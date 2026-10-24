
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QWEN_W4GROUP_MATMUL_V310_H_
#define ACLNN_QWEN_W4GROUP_MATMUL_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQwenW4GroupMatmulV310GetWorkspaceSize
 * parameters :
 * x : required
 * codes : required
 * scale : required
 * offset : required
 * tiled : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW4GroupMatmulV310GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *codes,
    const aclTensor *scale,
    const aclTensor *offset,
    bool tiled,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQwenW4GroupMatmulV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW4GroupMatmulV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
