#ifndef CHUNK_KDA_FWD_COMPAT_310P_H
#define CHUNK_KDA_FWD_COMPAT_310P_H

#ifndef __CCE_KT_TEST__
#include "kernel_operator.h"
#endif

// The 310P vector and cube pipelines share one physical AI Core. Select the
// unified CATLASS dispatch path before any common KDA headers are parsed.
#ifndef CATLASS_UNIFIED_CORE
#define CATLASS_UNIFIED_CORE 1
#endif

// The dav_m200 compiler has no native BF16 type. CATLASS references the type
// in templates that are not instantiated by the FP16-only 310P KDA path.
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && !defined(__bfloat16_t_defined)
#define __bfloat16_t_defined
#define CHUNK_KDA_FWD_COMPAT_310P_ACTIVE
struct bfloat16_t {
    uint16_t val;
    bfloat16_t() = default;
    bfloat16_t(float value) : val(0) { (void)value; }
    operator float() const { return 0.0F; }
};
#endif

// 310P has no fixpipe unit; post-matmul stores use MTE3.
#ifndef PIPE_FIX
#define PIPE_FIX PIPE_MTE3
#endif

#ifdef CHUNK_KDA_FWD_COMPAT_310P_ACTIVE
#define LoadDataWithSparse LoadDataWithSparseCal
namespace AscendC {
inline float ToFloat(bfloat16_t value)
{
    return static_cast<float>(value);
}
} // namespace AscendC
#endif

#endif
