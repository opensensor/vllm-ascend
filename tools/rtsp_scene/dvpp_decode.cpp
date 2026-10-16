// SPDX-License-Identifier: Apache-2.0
// Host demux; DVPP decodes/resizes to raw RGB or NV12 for optional hardware JPEG.
#include <acl/acl.h>
#include <acl/dvpp/hi_dvpp.h>
#include "sample_clock.h"

extern "C" {
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/log.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr int MAX_IN_FLIGHT = 32;
constexpr int REFERENCE_FRAMES = 16;
constexpr int DISPLAY_FRAMES = 2;
constexpr int IO_TIMEOUT_SECONDS = 10;
constexpr int JPEG_QUALITY = 90;
constexpr int JPEG_BUFFER_BYTES_PER_PIXEL = 3;
constexpr int JPEG_BUFFER_ALIGNMENT = 64;
constexpr int MAX_JPEG_CHANNELS = 256;
constexpr int RGB_CHANNELS = 3;
constexpr int MICROSECONDS_PER_SECOND = 1000000;
using Clock = std::chrono::steady_clock;

void check(int status, const char* operation) {
  if (status != 0) {
    std::ostringstream message;
    message << operation << " failed: 0x" << std::hex << static_cast<unsigned>(status);
    throw std::runtime_error(message.str());
  }
}

struct DeviceBuffer {
  void* pointer = nullptr;
  explicit DeviceBuffer(size_t size) { check(hi_mpi_dvpp_malloc(0, &pointer, size), "dvpp_malloc"); }
  ~DeviceBuffer() {
    if (pointer) hi_mpi_dvpp_free(pointer);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Pending {
  DeviceBuffer input;
  DeviceBuffer output;
  Pending(size_t input_size, size_t output_size) : input(input_size), output(output_size) {}
};

class Decoder {
 public:
  using FrameConsumer = std::function<void(void*, size_t, int, int, int)>;
  Decoder(int device, int max_side, int frames, double fps, std::filesystem::path output, bool raw_rgb,
          bool manage_runtime = true, FrameConsumer consumer = {})
      : device_(device),
        max_side_(max_side),
        frames_(frames),
        fps_(fps),
        output_(std::move(output)),
        raw_rgb_(raw_rgb),
        manage_runtime_(manage_runtime),
        consumer_(std::move(consumer)) {}
  ~Decoder() {
    stop_ = true;
    if (sender_.joinable()) sender_.join();
    if (jpeg_channel_ >= 0) {
      hi_mpi_venc_stop_chn(jpeg_channel_);
      hi_mpi_venc_destroy_chn(jpeg_channel_);
    }
    // Buffers must outlive the decoder's use of them, including on error.
    if (channel_ >= 0) {
      hi_mpi_vdec_stop_recv_stream(channel_);
      hi_mpi_vdec_reset_chn(channel_);
      hi_mpi_vdec_destroy_chn(channel_);
    }
    pending_.clear();
    if (system_ready_) hi_mpi_sys_exit();
    if (context_ && manage_runtime_) aclrtDestroyContext(context_);
    if (device_ready_) aclrtResetDevice(device_);
    if (acl_ready_) aclFinalize();
    av_bsf_free(&filter_);
    avformat_close_input(&format_);
  }

  void run(const std::string& source) {
    av_log_set_level(AV_LOG_QUIET);  // URLs and credentials must never reach diagnostics.
    format_ = avformat_alloc_context();
    if (!format_) throw std::runtime_error("Cannot allocate demux context");
    format_->interrupt_callback = {interrupt, this};
    touch_deadline();
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "timeout", "5000000", 0);
    int opened = avformat_open_input(&format_, source.c_str(), nullptr, &options);
    av_dict_free(&options);
    check(opened, "open input");
    touch_deadline();
    check(avformat_find_stream_info(format_, nullptr), "read stream metadata");
    stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_index_ < 0) throw std::runtime_error("No video stream");
    auto* parameters = format_->streams[stream_index_]->codecpar;
    if (parameters->codec_id != AV_CODEC_ID_H264 && parameters->codec_id != AV_CODEC_ID_HEVC)
      throw std::runtime_error("DVPP backend requires H.264 or H.265; no CPU fallback");
    if (parameters->width <= 0 || parameters->height <= 0) throw std::runtime_error("Missing source dimensions");
    const bool h264 = parameters->codec_id == AV_CODEC_ID_H264;
    type_ = h264 ? HI_PT_H264 : HI_PT_H265;
    const AVBitStreamFilter* definition = av_bsf_get_by_name(h264 ? "h264_mp4toannexb" : "hevc_mp4toannexb");
    check(av_bsf_alloc(definition, &filter_), "allocate bitstream filter");
    check(avcodec_parameters_copy(filter_->par_in, parameters), "copy codec parameters");
    filter_->time_base_in = format_->streams[stream_index_]->time_base;
    check(av_bsf_init(filter_), "initialize bitstream filter");

    if (manage_runtime_) {
      check(aclInit(nullptr), "aclInit");
      acl_ready_ = true;
      check(aclrtSetDevice(device_), "set device");
      device_ready_ = true;
      check(aclrtCreateContext(&context_, device_), "create context");
    } else {
      // An initialized Torch worker owns this context and its lifetime.
      int32_t current_device = -1;
      check(aclrtGetDevice(&current_device), "get worker device");
      if (current_device != device_) throw std::runtime_error("Decoder and worker must use the same NPU");
      check(aclrtGetCurrentContext(&context_), "get worker context");
      if (!context_) throw std::runtime_error("Worker NPU context must be initialized before decoding");
    }
    check(hi_mpi_sys_init(), "dvpp system init");
    system_ready_ = true;
    hi_vdec_chn_attr attr{};
    attr.type = type_;
    attr.mode = HI_VDEC_SEND_MODE_FRAME;
    attr.pic_width = parameters->width;
    attr.pic_height = parameters->height;
    attr.stream_buf_size = parameters->width * parameters->height * 3 / 2;
    attr.frame_buf_cnt = REFERENCE_FRAMES + DISPLAY_FRAMES + 1;
    hi_pic_buf_attr buffer_attr{
        attr.pic_width,       attr.pic_height, 0, HI_DATA_BIT_WIDTH_8, HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420,
        HI_COMPRESS_MODE_NONE};
    attr.frame_buf_size = hi_vdec_get_pic_buf_size(type_, &buffer_attr);
    attr.video_attr.ref_frame_num = REFERENCE_FRAMES;
    attr.video_attr.temporal_mvp_en = HI_TRUE;
    attr.video_attr.tmv_buf_size = hi_vdec_get_tmv_buf_size(type_, attr.pic_width, attr.pic_height);
    for (int candidate = 0; candidate < VDEC_MAX_CHN_NUM; ++candidate) {
      int status = hi_mpi_vdec_create_chn(candidate, &attr);
      if (status == 0) {
        channel_ = candidate;
        break;
      }
      if (static_cast<unsigned>(status) != HI_ERR_VDEC_EXIST) check(status, "create VDEC channel");
    }
    if (channel_ < 0) throw std::runtime_error("All VDEC channels occupied");
    hi_vdec_chn_param channel_param{};
    check(hi_mpi_vdec_get_chn_param(channel_, &channel_param), "get channel parameters");
    channel_param.display_frame_num = DISPLAY_FRAMES;
    channel_param.video_param.dec_mode = HI_VIDEO_DEC_MODE_IPB;
    channel_param.video_param.out_order = HI_VIDEO_OUT_ORDER_DISPLAY;
    check(hi_mpi_vdec_set_chn_param(channel_, &channel_param), "set channel parameters");
    check(hi_mpi_vdec_start_recv_stream(channel_), "start VDEC");
    const double scale = static_cast<double>(max_side_) / std::max(parameters->width, parameters->height);
    width_ = std::max(32, static_cast<int>(std::round(parameters->width * scale / 2)) * 2);
    height_ = std::max(32, static_cast<int>(std::round(parameters->height * scale / 2)) * 2);
    hi_module_type modules[] = {HI_MOD_VDEC, HI_MOD_JPEGE};
    hi_img_base_info image{};
    image.width = width_;
    image.height = height_;
    image.pixel_format = raw_rgb_ ? HI_PIXEL_FORMAT_RGB_888 : HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    check(hi_mpi_sys_get_image_align_info(modules, raw_rgb_ ? 1 : 2, &image, &alignment_), "image buffer alignment");
    if (!raw_rgb_) initialize_jpeg();
    std::cerr << "DVPP_INFO {\"codec\":\"" << (h264 ? "h264" : "hevc") << "\",\"source_width\":" << parameters->width
              << ",\"source_height\":" << parameters->height << ",\"width\":" << width_ << ",\"height\":" << height_
              << ",\"device\":" << device_ << ",\"channel\":" << channel_ << ",\"pixel_format\":\""
              << (raw_rgb_ ? "rgb888" : "nv12") << "\",\"frame_format\":\"" << (raw_rgb_ ? "ppm" : "jpeg")
              << "\",\"jpeg_encoder\":\"" << (raw_rgb_ ? "none" : "ascend_jpege") << "\"}" << std::endl;
    sender_ = std::thread([this] { send(); });
    receive();
    stop_ = true;
    sender_.join();
    if (sender_error_) std::rethrow_exception(sender_error_);
    if (samples_ != frames_) throw std::runtime_error("Stream ended before all sampled frames arrived");
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    const double cpu = usage.ru_utime.tv_sec + usage.ru_stime.tv_sec +
                       (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / static_cast<double>(MICROSECONDS_PER_SECOND);
    std::cerr << "DVPP_STATS {\"decoded_frames\":" << decoded_ << ",\"sampled_frames\":" << samples_
              << ",\"missed_sample_slots\":" << missed_sample_slots_ << ",\"submitted_frames\":" << submitted_frames_
              << ",\"untimestamped_input_packets\":" << untimed_packets_
              << ",\"largest_input_pts_gap_seconds\":" << largest_input_pts_gap_
              << ",\"jpeg_bytes_copied_to_host\":" << jpeg_bytes_ << ",\"rgb_bytes_copied_to_host\":" << rgb_bytes_
              << ",\"cpu_seconds\":" << cpu << "}" << std::endl;
  }

 private:
  void initialize_jpeg() {
    hi_venc_chn_attr attributes{};
    auto& jpeg = attributes.venc_attr;
    jpeg.type = HI_PT_JPEG;
    jpeg.max_pic_width = jpeg.pic_width = width_;
    jpeg.max_pic_height = jpeg.pic_height = height_;
    const int bytes = width_ * height_ * JPEG_BUFFER_BYTES_PER_PIXEL;
    jpeg.buf_size = ((bytes + JPEG_BUFFER_ALIGNMENT - 1) / JPEG_BUFFER_ALIGNMENT) * JPEG_BUFFER_ALIGNMENT;
    jpeg.is_by_frame = HI_TRUE;
    jpeg.jpeg_attr.recv_mode = HI_VENC_PIC_RECV_SINGLE;
    for (int candidate = 0; candidate < MAX_JPEG_CHANNELS; ++candidate) {
      int status = hi_mpi_venc_create_chn(candidate, &attributes);
      if (status == 0) {
        jpeg_channel_ = candidate;
        break;
      }
      if (static_cast<unsigned>(status) != HI_ERR_VENC_EXIST) check(status, "create JPEGE channel");
    }
    if (jpeg_channel_ < 0) throw std::runtime_error("All JPEGE channels occupied");
    hi_venc_start_param receive{-1};
    check(hi_mpi_venc_start_chn(jpeg_channel_, &receive), "start JPEGE");
    hi_venc_jpeg_param quality{};
    check(hi_mpi_venc_get_jpeg_param(jpeg_channel_, &quality), "get JPEG quality");
    quality.qfactor = JPEG_QUALITY;
    check(hi_mpi_venc_set_jpeg_param(jpeg_channel_, &quality), "set JPEG quality");
  }
  static int interrupt(void* opaque) {
    auto* self = static_cast<Decoder*>(opaque);
    return self->stop_ || Clock::now().time_since_epoch().count() > self->deadline_;
  }
  void touch_deadline() {
    deadline_ = (Clock::now() + std::chrono::seconds(IO_TIMEOUT_SECONDS)).time_since_epoch().count();
  }
  void send_packet(AVPacket* packet) {
    if (packet->pts == AV_NOPTS_VALUE)
      throw std::runtime_error("Video packet has no PTS; cannot preserve evidence time");
    while (!stop_) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_.size() < MAX_IN_FLIGHT) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (stop_) return;
    auto buffers = std::make_unique<Pending>(packet->size, alignment_.img_buf_size);
    void* input = buffers->input.pointer;
    check(aclrtMemcpy(input, packet->size, packet->data, packet->size, ACL_MEMCPY_HOST_TO_DEVICE),
          "copy compressed packet");
    hi_vdec_stream stream{};
    stream.end_of_frame = HI_TRUE;
    stream.need_display = HI_TRUE;
    stream.pts = av_rescale_q(packet->pts, filter_->time_base_out, AVRational{1, MICROSECONDS_PER_SECOND});
    const double source_pts = static_cast<int64_t>(stream.pts) / static_cast<double>(MICROSECONDS_PER_SECOND);
    if (submitted_frames_ > 0)
      largest_input_pts_gap_ = std::max(largest_input_pts_gap_, source_pts - previous_input_pts_);
    previous_input_pts_ = source_pts;
    stream.len = packet->size;
    stream.addr = static_cast<hi_u8*>(input);
    hi_vdec_pic_info picture{};
    picture.width = width_;
    picture.height = height_;
    picture.width_stride = alignment_.width_stride;
    picture.height_stride = alignment_.height_stride;
    picture.pixel_format = raw_rgb_ ? HI_PIXEL_FORMAT_RGB_888 : HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    picture.vir_addr = reinterpret_cast<uint64_t>(buffers->output.pointer);
    picture.buffer_size = alignment_.img_buf_size;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_.emplace(input, std::move(buffers));
    }
    check(hi_mpi_vdec_send_stream(channel_, &stream, &picture, 1000), "send compressed frame");
    ++submitted_frames_;
  }
  void send() noexcept {
    AVPacket* packet = av_packet_alloc();
    AVPacket* filtered = av_packet_alloc();
    try {
      if (!packet || !filtered) throw std::runtime_error("Cannot allocate packet");
      check(aclrtSetCurrentContext(context_), "set sender context");
      bool keyframe_seen = false;
      while (!stop_) {
        touch_deadline();
        int status = av_read_frame(format_, packet);
        if (status == AVERROR_EOF) break;
        if (stop_) break;
        check(status, "read compressed packet");
        // RTSP can expose an initial access unit before its RTP clock is
        // anchored. Wait for a timestamped keyframe rather than invent PTS.
        if (packet->stream_index == stream_index_ && packet->pts == AV_NOPTS_VALUE) {
          ++untimed_packets_;
          if (keyframe_seen) throw std::runtime_error("Video input lost PTS after decoding began");
        }
        if (packet->stream_index != stream_index_ || packet->pts == AV_NOPTS_VALUE ||
            (!keyframe_seen && !(packet->flags & AV_PKT_FLAG_KEY))) {
          av_packet_unref(packet);
          continue;
        }
        keyframe_seen = true;
        check(av_bsf_send_packet(filter_, packet), "filter input packet");
        while ((status = av_bsf_receive_packet(filter_, filtered)) == 0) {
          send_packet(filtered);
          av_packet_unref(filtered);
        }
        if (status != AVERROR(EAGAIN) && status != AVERROR_EOF) check(status, "filter output packet");
      }
      if (!stop_) {
        check(av_bsf_send_packet(filter_, nullptr), "flush bitstream filter");
        int status;
        while ((status = av_bsf_receive_packet(filter_, filtered)) == 0) {
          send_packet(filtered);
          av_packet_unref(filtered);
        }
        if (status != AVERROR_EOF) check(status, "drain bitstream filter");
        hi_vdec_stream eos{};
        hi_vdec_pic_info picture{};
        eos.end_of_stream = HI_TRUE;
        check(hi_mpi_vdec_send_stream(channel_, &eos, &picture, 1000), "send end of stream");
      }
    } catch (...) {
      sender_error_ = std::current_exception();
    }
    av_packet_free(&filtered);
    av_packet_free(&packet);
    sender_done_ = true;
  }
  void save(const hi_video_frame_info& decoded) {
    if (raw_rgb_) {
      save_rgb(decoded.v_frame);
      return;
    }
    const auto& frame = decoded.v_frame;
    if (frame.pixel_format != HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420 || frame.width != static_cast<unsigned>(width_) ||
        frame.height != static_cast<unsigned>(height_) || frame.width_stride[0] < static_cast<unsigned>(width_))
      throw std::runtime_error("Unexpected decoded NV12 layout");
    auto jpeg_input = decoded;
    jpeg_input.v_frame.compress_mode = HI_COMPRESS_MODE_NONE;
    jpeg_input.v_frame.video_format = HI_VIDEO_FORMAT_LINEAR;
    jpeg_input.v_frame.field = HI_VIDEO_FIELD_FRAME;
    jpeg_input.v_frame.dynamic_range = HI_DYNAMIC_RANGE_SDR8;
    jpeg_input.v_frame.virt_addr[1] =
        static_cast<unsigned char*>(frame.virt_addr[0]) + frame.width_stride[0] * frame.height_stride[0];
    jpeg_input.v_frame.width_stride[1] = frame.width_stride[0];
    check(hi_mpi_venc_send_frame(jpeg_channel_, &jpeg_input, 1000), "submit NV12 to JPEGE");
    hi_venc_pack packet{};
    hi_venc_stream stream{};
    stream.pack = &packet;
    stream.pack_cnt = 1;
    check(hi_mpi_venc_get_stream(jpeg_channel_, &stream, 1000), "get hardware JPEG");
    std::exception_ptr failure;
    try {
      if (stream.pack_cnt != 1 || packet.len <= packet.offset || !packet.addr)
        throw std::runtime_error("Invalid hardware JPEG output packet");
      const size_t bytes = packet.len - packet.offset;
      std::vector<unsigned char> jpeg(bytes);
      check(aclrtMemcpy(jpeg.data(), bytes, packet.addr + packet.offset, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
            "copy compressed JPEG");
      if (bytes < 4 || jpeg[0] != 0xff || jpeg[1] != 0xd8 || jpeg[bytes - 2] != 0xff || jpeg[bytes - 1] != 0xd9)
        throw std::runtime_error("Hardware output is not a complete JPEG");
      char name[64];
      std::snprintf(name, sizeof(name), "frame-%06d.jpg", samples_ + 1);
      auto target = output_ / name;
      auto temporary = target.string() + ".tmp";
      FILE* file = std::fopen(temporary.c_str(), "wb");
      if (!file) throw std::runtime_error("Cannot write sampled frame");
      const bool written = std::fwrite(jpeg.data(), 1, bytes, file) == bytes;
      const int closed = std::fclose(file);
      if (!written || closed) throw std::runtime_error("Cannot finish sampled frame");
      std::filesystem::rename(temporary, target);
      jpeg_bytes_ += bytes;
    } catch (...) {
      failure = std::current_exception();
    }
    const int released = hi_mpi_venc_release_stream(jpeg_channel_, &stream);
    if (failure) std::rethrow_exception(failure);
    check(released, "release hardware JPEG");
  }
  void save_rgb(const hi_video_frame& frame) {
    const size_t row_bytes = static_cast<size_t>(width_) * RGB_CHANNELS;
    if (frame.pixel_format != HI_PIXEL_FORMAT_RGB_888 || frame.width != static_cast<unsigned>(width_) ||
        frame.height != static_cast<unsigned>(height_) || frame.width_stride[0] < row_bytes ||
        frame.height_stride[0] < static_cast<unsigned>(height_) || !frame.virt_addr[0])
      throw std::runtime_error("Unexpected decoded RGB layout");
    const size_t bytes = static_cast<size_t>(frame.width_stride[0]) * frame.height_stride[0];
    if (bytes > alignment_.img_buf_size) throw std::runtime_error("RGB stride exceeds the allocated device buffer");
    if (consumer_) {
      // Borrowed device memory remains valid only for this callback. The
      // consumer must finish all reads before returning and frame release.
      consumer_(frame.virt_addr[0], frame.width_stride[0], width_, height_, samples_);
      return;
    }
    std::vector<unsigned char> pixels(bytes);
    check(aclrtMemcpy(pixels.data(), bytes, frame.virt_addr[0], bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy decoded RGB");
    char name[64];
    std::snprintf(name, sizeof(name), "frame-%06d.ppm", samples_ + 1);
    auto target = output_ / name;
    auto temporary = target.string() + ".tmp";
    FILE* file = std::fopen(temporary.c_str(), "wb");
    if (!file) throw std::runtime_error("Cannot write sampled RGB frame");
    bool written = std::fprintf(file, "P6\n%d %d\n255\n", width_, height_) > 0;
    for (int row = 0; written && row < height_; ++row)
      written = std::fwrite(pixels.data() + static_cast<size_t>(row) * frame.width_stride[0], 1, row_bytes, file) ==
                row_bytes;
    const int closed = std::fclose(file);
    if (!written || closed) throw std::runtime_error("Cannot finish sampled RGB frame");
    std::filesystem::rename(temporary, target);
    rgb_bytes_ += bytes;
  }
  void receive() {
    bool first = true;
    int64_t origin = 0, previous = 0;
    int idle_after_eof = 0;
    rtsp_scene::SampleClock sampler(fps_);
    while (samples_ < frames_) {
      hi_video_frame_info frame{};
      hi_vdec_supplement_info supplement{};
      hi_vdec_stream stream{};
      int status = hi_mpi_vdec_get_frame(channel_, &frame, &supplement, &stream, 100);
      if (status != 0) {
        if (static_cast<unsigned>(status) != HI_ERR_VDEC_BUF_EMPTY) check(status, "get decoded frame");
        if (sender_done_ && ++idle_after_eof > 20) break;
        continue;
      }
      idle_after_eof = 0;
      std::exception_ptr failure;
      try {
        if (frame.v_frame.frame_flag != 0) throw std::runtime_error("DVPP reported a damaged/unsupported frame");
        if (stream.need_display && frame.v_frame.virt_addr[0]) {
          const int64_t pts = static_cast<int64_t>(frame.v_frame.pts);
          if (first) {
            origin = previous = pts;
            first = false;
          }
          if (pts < previous) throw std::runtime_error("Decoded PTS moved backwards");
          previous = pts;
          ++decoded_;
          const double seconds = (pts - origin) / static_cast<double>(MICROSECONDS_PER_SECOND);
          if (sampler.select(seconds)) {
            save(frame);
            std::cerr << "DVPP_FRAME {\"id\":" << samples_ << ",\"seconds\":" << seconds << "}" << std::endl;
            ++samples_;
            missed_sample_slots_ = sampler.missed();
          }
        }
      } catch (...) {
        failure = std::current_exception();
      }
      int released = hi_mpi_vdec_release_frame(channel_, &frame);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.erase(stream.addr);
      }
      if (failure) std::rethrow_exception(failure);
      check(released, "release decoded frame");
    }
  }

  int device_, max_side_, frames_, stream_index_ = -1, channel_ = -1;
  int jpeg_channel_ = -1;
  uint64_t jpeg_bytes_ = 0, rgb_bytes_ = 0;
  int width_ = 0, height_ = 0, samples_ = 0, decoded_ = 0;
  int64_t missed_sample_slots_ = 0;
  int submitted_frames_ = 0, untimed_packets_ = 0;
  double largest_input_pts_gap_ = 0, previous_input_pts_ = 0;
  double fps_;
  std::filesystem::path output_;
  bool raw_rgb_;
  bool manage_runtime_;
  FrameConsumer consumer_;
  AVFormatContext* format_ = nullptr;
  AVBSFContext* filter_ = nullptr;
  aclrtContext context_ = nullptr;
  hi_payload_type type_{};
  hi_img_align_info alignment_{};
  bool acl_ready_ = false, device_ready_ = false, system_ready_ = false;
  std::atomic<bool> stop_{false}, sender_done_{false};
  std::atomic<Clock::duration::rep> deadline_{0};
  std::mutex mutex_;
  std::map<void*, std::unique_ptr<Pending>> pending_;
  std::thread sender_;
  std::exception_ptr sender_error_;
};
}  // namespace

#ifndef RTSP_DVPP_LIBRARY
int main(int argc, char** argv) {
  umask(0077);
  try {
    if (argc != 6 && argc != 7)
      throw std::runtime_error(
          "Usage: dvpp_decode DEVICE MAX_SIDE FRAME_COUNT FPS OUTPUT_DIRECTORY [ppm|jpeg]; input URL/path on stdin");
    const int device = std::stoi(argv[1]), side = std::stoi(argv[2]), frames = std::stoi(argv[3]);
    const double fps = std::stod(argv[4]);
    if (device < 0 || side < 32 || side > 4096 || frames <= 0 || !std::isfinite(fps) || fps <= 0)
      throw std::runtime_error("Invalid device, dimensions, frame count or sample rate");
    std::string source;
    std::getline(std::cin, source);
    if (source.empty()) throw std::runtime_error("Missing input URL/path on stdin");
    const std::string format = argc == 7 ? argv[6] : "ppm";
    if (format != "ppm" && format != "jpeg") throw std::runtime_error("Invalid frame format; use ppm or jpeg");
    Decoder decoder(device, side, frames, fps, argv[5], format == "ppm");
    decoder.run(source);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "DVPP_ERROR " << error.what() << std::endl;
    return 1;
  }
}
#endif
