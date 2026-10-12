/* SPDX-License-Identifier: GPL-2.0-only */
/* Restricted qualification of the Ascend DMA engine against GPU DMA-BUF. */
#include <linux/capability.h>
#include <linux/completion.h>
#include <linux/dma-buf.h>
#include <linux/dma-resv.h>
#include <linux/err.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/sched.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include "sidecard_peer.h"
#include "vendor_abi.h"

struct sc_npu_pin {
  atomic_t revoked;
  struct completion finished;
};

struct sc_retained {
  struct dma_buf* buffer;
  struct dma_buf_attachment* attachment;
  struct sg_table* table;
  struct svm_dma_desc_handle guard;
};

struct sc_context {
  struct mutex lock;
  pid_t owner;
  struct sc_retained* retained;
};

static void sc_npu_revoke(void* data) {
  struct sc_npu_pin* pin = data;
  atomic_set(&pin->revoked, 1);
  /* Do not call hal_kernel_p2p_put_pages from this vendor callback. */
  wait_for_completion(&pin->finished);
}

static int sc_pin_npu(struct sc_peer_request* request) {
  struct p2p_page_table* pages = NULL;
  struct sc_npu_pin pin;
  phys_addr_t device_address;
  int result;

  atomic_set(&pin.revoked, 0);
  init_completion(&pin.finished);
  result = hal_kernel_p2p_get_pages(request->npu_va, request->pin_length, sc_npu_revoke, &pin, &pages);
  if (result) return result;
  if (!pages || pages->version != SC_P2P_TABLE_VERSION || !pages->page_size || !pages->page_num || !pages->pages_info) {
    result = -EPROTO;
    goto release;
  }
  request->npu_page_size = pages->page_size;
  result = devdrv_devmem_addr_h2d(request->physical_chip, pages->pages_info[0].pa, &device_address);
release:
  request->revoked = atomic_read(&pin.revoked);
  complete_all(&pin.finished);
  hal_kernel_p2p_put_pages(pages);
  return result;
}

static int sc_make_nodes(struct sc_peer_request* request, struct sg_table* table, struct devdrv_dma_node* nodes,
                         u32* count) {
  struct scatterlist* entry;
  u64 offset = 0;
  unsigned int index;
  *count = 0;

  for_each_sgtable_dma_sg(table, entry, index) {
    u64 within = 0;
    while (within < sg_dma_len(entry) && offset < request->length) {
      u64 gpu_address = sg_dma_address(entry) + within;
      u64 npu_address;
      u64 bytes = min_t(u64, request->length - offset, sg_dma_len(entry) - within);
      int result;
      bytes = min_t(u64, bytes, SC_DMA_PAGE_BYTES - ((request->npu_va + offset) % SC_DMA_PAGE_BYTES));
      result = hal_kernel_svm_dev_va_to_dma_addr(task_tgid_nr(current), request->cache_id, request->npu_va + offset,
                                                 &npu_address);
      if (result) return result;
      if (*count &&
          nodes[*count - 1].src_addr + nodes[*count - 1].size ==
              (request->operation == SC_PEER_CUDA_TO_ASCEND ? gpu_address : npu_address) &&
          nodes[*count - 1].dst_addr + nodes[*count - 1].size ==
              (request->operation == SC_PEER_CUDA_TO_ASCEND ? npu_address : gpu_address)) {
        nodes[*count - 1].size += bytes;
      } else {
        struct devdrv_dma_node* node;
        if (*count >= SC_MAX_DMA_NODES) return -E2BIG;
        node = &nodes[(*count)++];
        node->src_addr = request->operation == SC_PEER_CUDA_TO_ASCEND ? gpu_address : npu_address;
        node->dst_addr = request->operation == SC_PEER_CUDA_TO_ASCEND ? npu_address : gpu_address;
        node->size = bytes;
        node->direction =
            request->operation == SC_PEER_CUDA_TO_ASCEND ? DEVDRV_DMA_HOST_TO_DEVICE : DEVDRV_DMA_DEVICE_TO_HOST;
      }
      within += bytes;
      offset += bytes;
    }
  }
  return offset == request->length ? 0 : -ENOSPC;
}

static int sc_pin_local(struct sc_peer_request* request, struct sg_table* table, struct sc_retained* retained) {
  struct svm_dma_desc_addr_info addresses = {
      .src_va = request->guard_va,
      .dst_va = request->npu_va,
      .size = request->pin_length,
  };
  struct svm_dma_desc_handle handle = {
      .pid = task_tgid_nr(current),
      .key = SC_LIFETIME_KEY, /* Private SIDE qualification namespace. */
      .subkey = SC_LIFETIME_SUBKEY,
  };
  struct svm_dma_desc descriptor = {0};
  struct devdrv_dma_node* nodes = NULL;
  u64 address;
  u32 count;
  int result;

  /* Creating a descriptor retains both ranges but does not submit its SQ. */
  result = hal_kernel_svm_dma_desc_create(&addresses, &handle, &descriptor);
  if (result) return result;
  result = -EADDRNOTAVAIL;
  /* Qualification established that kernel cache IDs equal physical 2/3. */
  request->cache_id = request->physical_chip;
  result = hal_kernel_svm_dev_va_to_dma_addr(handle.pid, request->cache_id, request->npu_va, &address);
  if (result || request->operation == SC_PEER_LOCAL_PIN_ONLY) goto release;
  nodes = kcalloc(SC_MAX_DMA_NODES, sizeof(*nodes), GFP_KERNEL);
  if (!nodes) {
    result = -ENOMEM;
    goto release;
  }
  result = sc_make_nodes(request, table, nodes, &count);
  if (result) goto release;
  request->pin_status = 0;
  request->dma_status = devdrv_dma_sync_link_copy_extend(request->physical_chip, DEVDRV_DMA_DATA_COMMON,
                                                         SC_DMA_WAIT_INTERRUPT, nodes, count);
  if (request->dma_status) {
    /* A negative SDK return does not prove DMA quiescence. Retain all mapped
     * resources and the HBM lifetime descriptor. The diagnostic caller must
     * remain alive with its allocations. No forced module removal or reset. */
    retained->guard = handle;
    request->retained = 1;
    __module_get(THIS_MODULE);
    kfree(nodes);
    return 0;
  }
release:
  kfree(nodes);
  hal_kernel_svm_dma_desc_destroy(&handle);
  return result;
}

static int sc_map_gpu(struct file* file, struct sc_peer_request* request) {
  struct dma_buf_attachment* attachment;
  struct dma_buf* buffer;
  struct sg_table* table;
  struct scatterlist* entry;
  struct device* device;
  struct sc_retained* retained;
  unsigned int index;
  int result = 0;

  device = devdrv_get_pci_dev_by_devid(request->physical_chip);
  if (!device || !dev_is_pci(device)) return -ENODEV;
  /* Restrict this host's experiment to the assigned Huawei PCI function. */
  if (to_pci_dev(device)->vendor != SC_ASCEND_VENDOR || to_pci_dev(device)->bus->number != SC_ASSIGNED_PCI_BUS ||
      to_pci_dev(device)->devfn != PCI_DEVFN(0, 0))
    return -EXDEV;
  retained = kzalloc(sizeof(*retained), GFP_KERNEL);
  if (!retained) return -ENOMEM;

  buffer = dma_buf_get(request->dmabuf_fd);
  if (IS_ERR(buffer)) {
    result = PTR_ERR(buffer);
    goto free_retained;
  }
  if (!buffer->exp_name || strcmp(buffer->exp_name, "nv_dmabuf")) {
    result = -EPROTOTYPE;
    goto put_buffer;
  }
  if (buffer->size < request->length) {
    result = -ENOSPC;
    goto put_buffer;
  }
  /* Static attachment: mappings are held until explicitly unmapped. */
  attachment = dma_buf_attach(buffer, device);
  if (IS_ERR(attachment)) {
    result = PTR_ERR(attachment);
    goto put_buffer;
  }
  dma_resv_lock(buffer->resv, NULL);
  table = dma_buf_map_attachment(attachment, DMA_BIDIRECTIONAL);
  if (IS_ERR(table)) {
    result = PTR_ERR(table);
    goto unlock_buffer;
  }
  request->gpu_segments = table->nents;
  for_each_sgtable_dma_sg(table, entry, index) request->gpu_mapped_bytes += sg_dma_len(entry);
  if (request->operation == SC_PEER_PIN_ONLY) request->pin_status = sc_pin_npu(request);
  if (request->operation >= SC_PEER_LOCAL_PIN_ONLY) request->pin_status = sc_pin_local(request, table, retained);
  if (request->retained) {
    retained->buffer = buffer;
    retained->attachment = attachment;
    retained->table = table;
    ((struct sc_context*)file->private_data)->retained = retained;
    dma_resv_unlock(buffer->resv);
    return 0;
  }
  dma_buf_unmap_attachment(attachment, table, DMA_BIDIRECTIONAL);
unlock_buffer:
  dma_resv_unlock(buffer->resv);
  dma_buf_detach(buffer, attachment);
put_buffer:
  dma_buf_put(buffer);
free_retained:
  kfree(retained);
  return result;
}

static long sc_ioctl_locked(struct file* file, unsigned int command, unsigned long argument) {
  struct sc_context* context = file->private_data;
  struct sc_peer_request request;

  if (!capable(CAP_SYS_RAWIO)) return -EPERM;
  if (context->retained) return -EBUSY;
  if (command != SC_PEER_EXECUTE) return -ENOTTY;
  if (copy_from_user(&request, (void __user*)argument, sizeof(request))) return -EFAULT;
  if (request.version != SC_PEER_VERSION ||
      (request.physical_chip != SC_ASSIGNED_CHIP_FIRST && request.physical_chip != SC_ASSIGNED_CHIP_LAST) ||
      request.operation > SC_PEER_ASCEND_TO_CUDA ||
      (request.operation == SC_PEER_MAP_ONLY && (request.npu_va || request.pin_length || request.guard_va)) ||
      (request.operation != SC_PEER_MAP_ONLY &&
       (!request.npu_va || !request.pin_length || request.pin_length > SC_PEER_MAX_BYTES ||
        request.length > request.pin_length || request.npu_va + request.pin_length < request.npu_va ||
        request.guard_va + request.pin_length < request.guard_va)) ||
      (request.operation >= SC_PEER_LOCAL_PIN_ONLY && !request.guard_va) || request.cache_id || request.retained ||
      request.reserved || !request.length || request.length > SC_PEER_MAX_BYTES)
    return -EINVAL;
  request.map_status = 0;
  request.pin_status = -EOPNOTSUPP;
  request.dma_status = -EOPNOTSUPP;
  request.gpu_segments = 0;
  request.gpu_mapped_bytes = 0;
  request.npu_page_size = 0;
  request.revoked = 0;
  request.retained = 0;
  request.map_status = sc_map_gpu(file, &request);
  return copy_to_user((void __user*)argument, &request, sizeof(request)) ? -EFAULT : 0;
}

static long sc_ioctl(struct file* file, unsigned int command, unsigned long argument) {
  struct sc_context* context = file->private_data;
  long result;
  if (context->owner != task_tgid_nr(current)) return -EPERM;
  mutex_lock(&context->lock);
  result = sc_ioctl_locked(file, command, argument);
  mutex_unlock(&context->lock);
  return result;
}

static int sc_open(struct inode* inode, struct file* file) {
  struct sc_context* context;
  if (!capable(CAP_SYS_RAWIO)) return -EPERM;
  context = kzalloc(sizeof(*context), GFP_KERNEL);
  if (!context) return -ENOMEM;
  mutex_init(&context->lock);
  context->owner = task_tgid_nr(current);
  file->private_data = context;
  return 0;
}

static int sc_release(struct inode* inode, struct file* file) {
  struct sc_context* context = file->private_data;
  if (!context->retained) kfree(context);
  /* Failed DMA retains the context, native descriptors, and module reference. */
  return 0;
}

static const struct file_operations sc_operations = {
    .owner = THIS_MODULE,
    .open = sc_open,
    .release = sc_release,
    .unlocked_ioctl = sc_ioctl,
};

static struct miscdevice sc_device = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "sidecard_peer",
    .fops = &sc_operations,
    .mode = 0600,
};

static int __init sc_initialize(void) { return misc_register(&sc_device); }
static void __exit sc_finalize(void) { misc_deregister(&sc_device); }

module_init(sc_initialize);
module_exit(sc_finalize);
MODULE_LICENSE("GPL");
MODULE_IMPORT_NS("DMA_BUF");
MODULE_DESCRIPTION("Restricted Ascend/CUDA DMA-BUF direct transfer qualification");
