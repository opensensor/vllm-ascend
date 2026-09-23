/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

// CANN's metadata prebuild does not inherit target compile definitions. Keep
// the 310P capability selection in the translation unit so metadata generation
// and host compilation see the same operator signatures.
#define KDA_310P_FP16_INPUT_ONLY 1
#include "chunk_kda_fwd_def.cpp"
