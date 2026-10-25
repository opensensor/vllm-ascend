# Reuse immutable INT8 expert scale metadata

INT8-repacked banks pass their original FP16 scale buffers to the native
kernel, which already owns prepared immutable metadata. Other projection
layouts retain the FP32 scale contract. This avoids full-bank casts on every
token without changing the stored weights or arithmetic.

Three matched requests measured 3.4098, 3.3877 and 3.3886 tokens/s, versus
3.3404 median before the change. All response messages matched exactly.
The resident switch preserved worker PIDs, weight storage and request caches.
The installed Python source includes the change for future starts.
