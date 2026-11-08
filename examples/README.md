# Synthetic interface example — not a real protein feature cache

`synthetic_features.npy` is a deterministic float32 random array of shape19×1182, generated with NumPy seed20261108 only to check loading, dimensions and finite model execution. It is not an ESM-C/HMM extraction and its predicted S² values must not be interpreted biologically or reported as benchmark performance.

No measured labels, real protein sequence or training/test features are included in this example. For actual use supply correctly mapped ESM-C1152/HMM-legacy30 arrays for the complete authoritative chain.
