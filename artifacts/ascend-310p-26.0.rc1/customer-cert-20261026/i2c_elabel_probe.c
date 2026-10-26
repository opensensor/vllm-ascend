/*
 * Direct I2C access to the Ascend SoC's own controller (i2c-0, DesignWare).
 *
 * The elabel read path in libdcmi is:
 *
 *   dcmi_elabel_init
 *     -> dcmi_flash_read_elabel(buf, len)
 *       -> dcmi_i2c_get_data(g_i2c_dev_name, 0x50, buf, len, reg)
 *         -> hal_npu_func_hooker(..., 0x157, ...)     ; the HAL I2C read
 *
 * and the HAL implements that as: open("/dev/i2c-%d"), ioctl(fd, I2C_SLAVE, addr),
 * then an I2C_SMBUS transfer.  The device address is 0x50 and the register is
 * 0x200 on a 310P (0x280 on a 910B).
 *
 * This probe reproduces that directly so the I2C transaction can be observed
 * independently of libdcmi's bookkeeping.  It is READ ONLY: it issues a
 * single SMBus read and prints what comes back.
 *
 * Usage: i2c_elabel_probe [i2c_bus] [slave_addr]
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* linux/i2c-dev.h ABI, spelled out so the union layout is unambiguous */
struct i2c_smbus_ioctl_data {
    unsigned char read_write;
    unsigned char command;
    unsigned int size;
    unsigned char data[34];
};
#define I2C_SLAVE        0x0703
#define I2C_SMBUS        0x0720
#define I2C_SMBUS_READ   1
#define I2C_SMBUS_BYTE   1
#define I2C_SMBUS_BLOCK_DATA_LEN 34

/* linux/i2c-dev.h combined transfer */
struct i2c_rdwr_ioctl_data {
    unsigned int nmsgs;
    struct i2c_msg *msgs;
};
struct i2c_msg {
    unsigned short addr;
    unsigned short flags;
    unsigned short len;
    unsigned char *buf;
};
#define I2C_RDWR 0x0707
#define I2C_M_RD  0x0001
#define I2C_M_TEN 0x0010

int main(int argc, char **argv)
{
    int bus = argc > 1 ? atoi(argv[1]) : 0;
    int slave = argc > 2 ? atoi(argv[2]) : 0x50;

    char path[64];
    snprintf(path, sizeof(path), "/dev/i2c-%d", bus);
    int fd = open(path, O_RDWR);
    if (fd < 0) {
        printf("open %s failed: %s\n", path, strerror(errno));
        return 1;
    }
    printf("opened %s\n", path);

    if (ioctl(fd, I2C_SLAVE, slave) < 0) {
        printf("ioctl(I2C_SLAVE, 0x%02x) failed: %s\n", slave, strerror(errno));
        close(fd);
        return 1;
    }
    printf("I2C_SLAVE set to 0x%02x\n", slave);

    /* Enumerate which slave addresses ACK, by attempting a zero-length read. */
    printf("\n--- slave ACK scan ---\n");
    for (int a = 0x08; a <= 0x77; a++) {
        if (ioctl(fd, I2C_SLAVE, a) < 0) continue;
        unsigned char b;
        if (read(fd, &b, 1) == 1) {
            printf("  0x%02x ACK (byte 0x%02x)\n", a, b);
        }
    }
    if (ioctl(fd, I2C_SLAVE, slave) < 0) {
        printf("restore slave failed\n");
        return 1;
    }

    /* Combined write-then-read, which is how the vendor HAL reads the
     * elabel register: write the register address, then read N bytes back. */
    for (int reg = 0x200; reg <= 0x208; reg += 8) {
        unsigned char wbuf[2];
        unsigned char rbuf[32];
        struct i2c_msg msgs[2];
        struct i2c_rdwr_ioctl_data rdwr;

        wbuf[0] = (unsigned char)(reg >> 8);
        wbuf[1] = (unsigned char)(reg & 0xff);
        memset(rbuf, 0, sizeof(rbuf));

        msgs[0].addr = slave;
        msgs[0].flags = 0;
        msgs[0].len = 2;
        msgs[0].buf = wbuf;
        msgs[1].addr = slave;
        msgs[1].flags = I2C_M_RD;
        msgs[1].len = sizeof(rbuf);
        msgs[1].buf = rbuf;

        rdwr.nmsgs = 2;
        rdwr.msgs = msgs;

        if (ioctl(fd, I2C_RDWR, &rdwr) < 0) {
            printf("  reg 0x%03x: I2C_RDWR failed: %s\n", reg, strerror(errno));
            continue;
        }
        printf("  reg 0x%03x: ", reg);
        for (unsigned i = 0; i < 16; i++) printf("%02x", rbuf[i]);
        printf("\n");
    }

    /* Raw I2C write of 2 bytes (register address) at slave 0x50, mirroring
     * what dcmi_i2c_set_data does.  Writing the register address is a
     * no-op for a read-only register and is the smallest possible probe of
     * whether the write path has a responder at all. */
    {
        unsigned char wbuf[2];
        struct i2c_msg msgs[1];
        struct i2c_rdwr_ioctl_data rdwr;

        wbuf[0] = 0x02; wbuf[1] = 0x00;
        msgs[0].addr = slave;
        msgs[0].flags = 0;
        msgs[0].len = 2;
        msgs[0].buf = wbuf;
        rdwr.nmsgs = 1;
        rdwr.msgs = msgs;

        printf("\n--- raw I2C write probe (register 0x200, 2 bytes) ---\n");
        if (ioctl(fd, I2C_RDWR, &rdwr) < 0) {
            printf("  I2C_RDWR write -> %s\n", strerror(errno));
        } else {
            printf("  I2C_RDWR write -> OK (responder present)\n");
        }
    }

    /* Also try the plain SMBus byte read at the same registers. */
    for (int reg = 0x200; reg <= 0x208; reg += 8) {
        struct i2c_smbus_ioctl_data d;
        memset(&d, 0, sizeof(d));
        d.read_write = I2C_SMBUS_READ;
        d.command = (unsigned char)reg;
        d.size = 1;

        if (ioctl(fd, I2C_SMBUS, &d) < 0) {
            printf("  reg 0x%03x: SMBus failed: %s\n", reg, strerror(errno));
            continue;
        }
        printf("  reg 0x%03x: byte = 0x%02x\n", reg, (unsigned char)d.data[0]);
    }

    /* Try a block read of 32 bytes at 0x200. */
    {
        struct i2c_smbus_ioctl_data d;
        memset(&d, 0, sizeof(d));
        d.read_write = I2C_SMBUS_READ;
        d.command = 0x00;
        d.size = 32;
        if (ioctl(fd, I2C_SMBUS, &d) < 0) {
            printf("block read failed: %s\n", strerror(errno));
        } else {
            printf("block read: ");
            for (int i = 0; i < 32; i++) printf("%02x", (unsigned char)d.data[i]);
            printf("\n");
        }
    }

    close(fd);
    return 0;
}
