
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QWEN4EXP_PLE_DECODE_V310_H_
#define ACLNN_QWEN4EXP_PLE_DECODE_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQwen4ExpPleDecodeV310GetWorkspaceSize
 * parameters :
 * projected : required
 * hidden : required
 * normKeyWeight : required
 * normQueryWeight : required
 * normConvWeight : required
 * currentConvWeight : required
 * normEps : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwen4ExpPleDecodeV310GetWorkspaceSize(
    const aclTensor *projected,
    const aclTensor *hidden,
    const aclTensor *normKeyWeight,
    const aclTensor *normQueryWeight,
    const aclTensor *normConvWeight,
    const aclTensor *currentConvWeight,
    double normEps,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQwen4ExpPleDecodeV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwen4ExpPleDecodeV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
