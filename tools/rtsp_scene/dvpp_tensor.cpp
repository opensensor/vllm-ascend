// SPDX-License-Identifier: Apache-2.0
// Experimental same-process decode. Never exports borrowed pointers across IPC.
#define RTSP_DVPP_LIBRARY
#include "dvpp_decode.cpp"
#include <torch/extension.h>
#include <pybind11/functional.h>

void CaptureTensor(const std::string& source, int device, int side, int count, double fps,
                   pybind11::function callback) {
  TORCH_CHECK(device >= 0 && side >= 32 && side <= 4096 && count > 0 && std::isfinite(fps) && fps > 0,
              "Invalid decode geometry");
  // Python's callback must synchronize before returning. The non-owning tensor
  // view and any view derived from it must never escape the callback lifetime.
  Decoder decoder(device, side, count, fps, {}, true, false,
                  [callback, device](void* pointer, size_t stride, int width, int height, int sample) {
                    auto options =
                        torch::TensorOptions().dtype(torch::kUInt8).device(c10::DeviceType::PrivateUse1, device);
                    auto tensor = torch::from_blob(
                        pointer, {height, width, 3}, {static_cast<int64_t>(stride), 3, 1}, [](void*) {}, options);
                    try {
                      callback(tensor, sample);
                    } catch (...) {
                      // Flush queued Torch reads before Decoder releases DVPP memory.
                      pybind11::module_::import("torch").attr("npu").attr("synchronize")(device);
                      throw;
                    }
                    pybind11::module_::import("torch").attr("npu").attr("synchronize")(device);
                  });
  decoder.run(source);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("capture", &CaptureTensor); }
