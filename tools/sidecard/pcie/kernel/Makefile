# SPDX-License-Identifier: GPL-2.0-only
obj-m += sidecard_peer.o

KERNEL_BUILD ?= /lib/modules/$(shell uname -r)/build
ASCEND_SYMVERS ?= /var/lib/dkms/davinci_ascend/1.0/$(shell uname -r)/x86_64/module/Module.symvers
KBUILD_EXTRA_SYMBOLS := $(ASCEND_SYMVERS)

.PHONY: all clean
all:
	$(MAKE) -C $(KERNEL_BUILD) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KERNEL_BUILD) M=$(CURDIR) clean
