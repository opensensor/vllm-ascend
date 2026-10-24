
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RMS_NORM_DYNAMIC_QUANT_H_
#define ACLNN_RMS_NORM_DYNAMIC_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRmsNormDynamicQuantGetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * smoothScale1Optional : optional
 * smoothScale2Optional : optional
 * betaOptional : optional
 * epsilon : optional
 * outputMaskOptional : optional
 * dstType : optional
 * y1Out : required
 * y2Out : required
 * scale1Out : required
 * scale2Out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNormDynamicQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *smoothScale1Optional,
    const aclTensor *smoothScale2Optional,
    const aclTensor *betaOptional,
    double epsilon,
    const aclBoolArray *outputMaskOptional,
    int64_t dstType,
    const aclTensor *y1Out,
    const aclTensor *y2Out,
    const aclTensor *scale1Out,
    const aclTensor *scale2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRmsNormDynamicQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNormDynamicQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
