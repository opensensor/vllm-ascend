// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdarg>
#include <cstddef>
#include <cstdio>

// Bounded diagnostic output shared by host tests and the isolated AI CPU probe.
class PleProbeText {
public:
    PleProbeText(char* data, size_t capacity) : data_(data), capacity_(capacity) {
        if (capacity_) data_[0] = '\0';
    }
    bool append(const char* format, ...) {
        if (!capacity_ || used_ == capacity_ - 1) { truncated_ = true; return false; }
        va_list args;
        va_start(args, format);
        int count = std::vsnprintf(data_ + used_, capacity_ - used_, format, args);
        va_end(args);
        if (count < 0) { truncated_ = true; return false; }
        if (static_cast<size_t>(count) >= capacity_ - used_) {
            used_ = capacity_ - 1;
            truncated_ = true;
            return false;
        }
        used_ += static_cast<size_t>(count);
        return true;
    }
    size_t size() const { return used_; }
    bool truncated() const { return truncated_; }
private:
    char* data_;
    size_t capacity_;
    size_t used_ = 0;
    bool truncated_ = false;
};
