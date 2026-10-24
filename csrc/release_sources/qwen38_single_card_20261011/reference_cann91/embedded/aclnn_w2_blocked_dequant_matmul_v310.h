
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_W2BLOCKED_DEQUANT_MATMUL_V310_H_
#define ACLNN_W2BLOCKED_DEQUANT_MATMUL_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnW2BlockedDequantMatmulV310GetWorkspaceSize
 * parameters :
 * x : required
 * codes : required
 * blockScale : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnW2BlockedDequantMatmulV310GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *codes,
    const aclTensor *blockScale,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnW2BlockedDequantMatmulV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnW2BlockedDequantMatmulV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
