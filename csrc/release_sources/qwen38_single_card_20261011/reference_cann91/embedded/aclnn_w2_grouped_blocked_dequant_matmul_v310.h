
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_W2GROUPED_BLOCKED_DEQUANT_MATMUL_V310_H_
#define ACLNN_W2GROUPED_BLOCKED_DEQUANT_MATMUL_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnW2GroupedBlockedDequantMatmulV310GetWorkspaceSize
 * parameters :
 * x : required
 * codes : required
 * blockScale : required
 * groupEnds : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnW2GroupedBlockedDequantMatmulV310GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *codes,
    const aclTensor *blockScale,
    const aclTensor *groupEnds,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnW2GroupedBlockedDequantMatmulV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnW2GroupedBlockedDequantMatmulV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
