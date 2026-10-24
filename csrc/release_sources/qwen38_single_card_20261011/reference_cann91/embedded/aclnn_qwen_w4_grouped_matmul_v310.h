
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QWEN_W4GROUPED_MATMUL_V310_H_
#define ACLNN_QWEN_W4GROUPED_MATMUL_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQwenW4GroupedMatmulV310GetWorkspaceSize
 * parameters :
 * x : required
 * codes : required
 * scale : required
 * offset : required
 * groupEnds : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW4GroupedMatmulV310GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *codes,
    const aclTensor *scale,
    const aclTensor *offset,
    const aclTensor *groupEnds,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQwenW4GroupedMatmulV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW4GroupedMatmulV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
