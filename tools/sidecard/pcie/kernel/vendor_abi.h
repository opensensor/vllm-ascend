/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef SC_VENDOR_ABI_H
#define SC_VENDOR_ABI_H

/* Exact structures and declarations from the installed Ascend kernel headers:
 * inc/driver/ascend_kernel_hal.h and dev_inc_open/inc/devdrv_interface.h.
 * Kept separate from the userspace ABI; kernel builds check exported versions
 * against the installed driver's Module.symvers.
 */
struct p2p_page_info {
  u64 pa;
  u64 reserved[4];
};

struct p2p_page_table {
  u32 version;
  u64 page_size;
  struct p2p_page_info* pages_info;
  u64 page_num;
  u64 reserved[4];
};

extern struct device* devdrv_get_pci_dev_by_devid(u32 dev_id);
extern int hal_kernel_p2p_get_pages(u64 va, u64 len, void (*free_callback)(void* data), void* data,
                                    struct p2p_page_table** page_table);
extern int hal_kernel_p2p_put_pages(struct p2p_page_table* page_table);
extern int devdrv_devmem_addr_h2d(u32 devid, phys_addr_t host_bar_addr, phys_addr_t* device_phy_addr);

struct svm_dma_desc_addr_info {
  u64 src_va;
  u64 dst_va;
  u64 size;
};
struct svm_dma_desc_handle {
  int pid;
  u32 key;
  u32 subkey;
};
struct svm_dma_desc {
  void* sq_addr;
  u32 sq_tail;
};
extern int hal_kernel_svm_dma_desc_create(struct svm_dma_desc_addr_info* addr_info, struct svm_dma_desc_handle* handle,
                                          struct svm_dma_desc* dma_desc);
extern void hal_kernel_svm_dma_desc_destroy(struct svm_dma_desc_handle* handle);
extern int hal_kernel_svm_dev_va_to_dma_addr(int hostpid, u32 logical_devid, u64 va, u64* dma_addr);

enum devdrv_dma_direction {
  DEVDRV_DMA_HOST_TO_DEVICE = 0,
  DEVDRV_DMA_DEVICE_TO_HOST = 1,
  DEVDRV_DMA_SYS_TO_SYS = 2,
};
enum devdrv_dma_data_type {
  DEVDRV_DMA_DATA_COMMON = 0,
  DEVDRV_DMA_DATA_PCIE_MSG,
  DEVDRV_DMA_DATA_TRAFFIC,
  DEVDRV_DMA_DATA_MANAGE,
  DEVDRV_DMA_DATA_TYPE_MAX,
};
struct devdrv_dma_node {
  u64 src_addr;
  u64 dst_addr;
  u32 size;
  enum devdrv_dma_direction direction;
  u32 loc_passid;
};
extern int devdrv_dma_sync_link_copy_extend(u32 dev_id, enum devdrv_dma_data_type type, int wait_type,
                                            struct devdrv_dma_node* dma_node, u32 node_cnt);
#define SC_DMA_WAIT_INTERRUPT 1
#define SC_MAX_DMA_NODES 128
#define SC_DMA_PAGE_BYTES 4096ULL

#endif
