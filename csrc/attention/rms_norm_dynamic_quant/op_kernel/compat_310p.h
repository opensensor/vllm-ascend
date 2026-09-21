#ifndef RMS_NORM_DYNAMIC_QUANT_COMPAT_310P_H
#define RMS_NORM_DYNAMIC_QUANT_COMPAT_310P_H

// 310P (dav_m200, AICore 200) provides no native bfloat16_t; the BF16
// instantiation is never exercised at runtime (FP16-only on 310P) but must
// still compile. 910B/910C define bfloat16_t in __clang_cce_types.h.
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && !defined(__bfloat16_t_defined)
#define __bfloat16_t_defined
struct bfloat16_t {
    uint16_t val;
    bfloat16_t() = default;
    bfloat16_t(float v) : val(0) { (void)v; }
    operator float() const { return 0.f; }
};
#endif

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
// 310P has no DataCopyPad. Every variant in CANN's dav_m200 implementation
// (compiler/tikcpp/tikcfw/impl/dav_m200/kernel_operator_data_copy_impl.h) is a
// stub whose body is ASCENDC_REPORT_NOT_SUPPORT(), and that macro expands to
// nothing in a device release build -- so the call compiles, links, launches,
// moves no data and reports no error. A kernel that copies through it returns
// with its output untouched. Emulate it with 32B-granular DataCopy instead.
//
// The 310P tiling keeps this lossless (see RmsNormDynamicQuantTilingHelper):
//   * numLastDim is a multiple of 32, so every x / y / gamma / beta / smooth
//     row is a whole number of 32B blocks and copies exactly;
//   * every core owns a multiple of 8 rows and the row loop steps by a
//     multiple of 8, so every fp32 scale write starts on a 32B boundary.
// The one ragged write left is the final scale block of the last core, which
// runs off the end of the scale tensor -- the host allocates that tensor
// rounded up to 8 rows for exactly this reason.
#define RMS_NORM_DQ_NO_DATA_COPY_PAD 1

// GM -> UB. `count` rows of `len` elements: GM rows are contiguous, UB rows are
// padded out to a whole 32B block and then separated by `strideBlk` further 32B
// units -- what DataCopyPad(dst, src, {count, len * sizeof(T), 0, strideBlk}, {})
// does on 910B.
template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void CompatCopyGm2Ub(
    const R<T>& dst, const S<T>& src, const uint32_t len, const uint32_t count, const uint32_t strideBlk,
    const uint32_t elemsPerBlk)
{
    if (len == 0 || count == 0) {
        return;
    }
    const uint32_t blkPerRow = (len + elemsPerBlk - 1) / elemsPerBlk;
    if (len % elemsPerBlk == 0) {
        AscendC::DataCopyParams params{
            static_cast<uint16_t>(count), static_cast<uint16_t>(blkPerRow), 0, static_cast<uint16_t>(strideBlk)};
        AscendC::DataCopy(dst, src, params);
        return;
    }
    // Ragged row: round the read up to a whole block. The surplus lands in the
    // UB padding the tiling already reserves and is never read back.
    const uint32_t ubStep = (blkPerRow + strideBlk) * elemsPerBlk;
    const AscendC::DataCopyParams oneRow{1, static_cast<uint16_t>(blkPerRow), 0, 0};
    for (uint32_t i = 0; i < count; ++i) {
        AscendC::DataCopy(dst[i * ubStep], src[i * len], oneRow);
    }
}

// UB -> GM, the mirror of the above.
template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void CompatCopyUb2Gm(
    const R<T>& dst, const S<T>& src, const uint32_t len, const uint32_t count, const uint32_t strideBlk,
    const uint32_t elemsPerBlk)
{
    if (len == 0 || count == 0) {
        return;
    }
    const uint32_t blkPerRow = (len + elemsPerBlk - 1) / elemsPerBlk;
    if (len % elemsPerBlk == 0) {
        AscendC::DataCopyParams params{
            static_cast<uint16_t>(count), static_cast<uint16_t>(blkPerRow), static_cast<uint16_t>(strideBlk), 0};
        AscendC::DataCopy(dst, src, params);
        return;
    }
    const uint32_t ubStep = (blkPerRow + strideBlk) * elemsPerBlk;
    const AscendC::DataCopyParams oneRow{1, static_cast<uint16_t>(blkPerRow), 0, 0};
    for (uint32_t i = 0; i < count; ++i) {
        AscendC::DataCopy(dst[i * len], src[i * ubStep], oneRow);
        // Row i is written a whole block wide; its surplus is overwritten by
        // row i + 1, and MTE3 retires in issue order.
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
}
#endif

#endif
