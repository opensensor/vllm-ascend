# Avoid slow integer routing work

The retained routing path counts and accumulates bounded routes in INT32,
then converts the final expert boundaries to INT64 for the native kernel.
Attention reuses its integer block quotient. The isolated graph probe measured
0.211 ms for the original count sequence and 0.070 ms for INT32 counts; floating
point alternatives were slower. All compared boundary values were identical.

Three matched serving requests measured 3.593674, 3.597727 and 3.594553 tokens/s,
versus 3.462157 median with partial graphs before this change. All messages
matched exactly. Workers and weight storage stayed resident. The native-load
and module replacement patterns continue to compose the activation and graphs.
The installed source includes these changes for future starts.
