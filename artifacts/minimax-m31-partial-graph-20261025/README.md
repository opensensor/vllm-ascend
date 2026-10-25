# Graph investigation follow-up

Attention-only graph shadow comparisons were exact on all 360 checked calls.
The first wrapper repeatedly captured context buckets and measured 3.204
median tokens/s. Keeping two buckets reduced captures by two thirds; its cold
sample was 3.205 and warm samples were 3.371 and 3.375, versus 3.340 eager.
All response messages matched. The wrapper was not retained: it still copies
inputs and metadata and pays capture cost at new context boundaries.

The retained-output variant of the earlier full-stage shadow still showed
layer differences. An eager-repeat control also showed differences, so that
comparison does not isolate a graph-specific arithmetic defect. It must not
be cited as evidence that graph execution itself is inaccurate.

The next prototype splits each decoder layer around sparse attention, leaving
context-dependent cache work eager and sharing persistent device buffers
between the captured pieces. Its initial private-pool version exhausted capture
memory on one chip and stalled the collective. That trial required restarting
the service. Work continues with a shared pool/stream and pairwise capture
memory admission. This directory does not qualify that prototype for serving.

Raw `.py.txt` sources preserve the executed diagnostic text. These are
experiments, not production launch commands.
