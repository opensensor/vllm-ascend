/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef SIDECARD_PEER_H
#define SIDECARD_PEER_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define SC_PEER_VERSION 5
#define SC_GPU_MAP_TRANSIENT 0
#define SC_GPU_MAP_PERSISTENT 1
#define SC_MAX_GPU_MAPPINGS 2
#define SC_PEER_MAX_BYTES (8ULL * 1024 * 1024)
#define SC_PEER_MAP_ONLY 0
#define SC_PEER_PIN_ONLY 1
#define SC_PEER_LOCAL_PIN_ONLY 2
#define SC_PEER_CUDA_TO_ASCEND 3
#define SC_PEER_ASCEND_TO_CUDA 4
#define SC_ASSIGNED_CHIP_FIRST 2
#define SC_ASSIGNED_CHIP_LAST 3
#define SC_ASCEND_VENDOR 0x19e5
#define SC_ASSIGNED_PCI_BUS 0x49
#define SC_LIFETIME_KEY 0x53494445
#define SC_LIFETIME_SUBKEY 1
#define SC_P2P_TABLE_VERSION 1

struct sc_peer_request {
  __u32 version;
  __u32 physical_chip;
  __s32 dmabuf_fd;
  __u32 operation;
  __u64 npu_va;
  __u64 length;
  __u64 pin_length;
  __u64 guard_va;
  __s32 map_status;
  __s32 pin_status;
  __s32 dma_status;
  __u32 gpu_segments;
  __u64 npu_page_size;
  __u64 gpu_mapped_bytes;
  __u32 revoked;
  __u32 cache_id;
  __u32 retained;
  __u32 reserved;
  __u32 gpu_mapping_mode;
  __u32 gpu_mapping_reused;
};

#define SC_PEER_EXECUTE _IOWR('S', 0x40, struct sc_peer_request)

#endif
