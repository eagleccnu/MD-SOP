# Frozen feature contract

For one complete protein chain of length L, use finite float32 arrays:

| Component | Shape | Frozen meaning |
|---|---|---|
| ESM-C | L × 1152 | ESM-C 600M; revision `fcc9d36f5fff97d9bb69a36e3252de61cc78c968` |
| HMM | L × 30 | HMM-v1-legacy30, same column definitions/transformation as the frozen preparation |
| Model input | L × 1182 | Concatenate ESM-C first, then HMM30; no extra input standardization |

ESM special tokens must not appear as residue rows. The chain sequence, feature row order, valid-label mask and labels must share the same residue identity and indexing. Sequence length is necessary but insufficient to establish this identity. Do not concatenate chains or silently truncate a mismatching array.

HMM30 is not an arbitrary 30-dimensional embedding: the legacy parser/search/database choices matter. Source preparation and parser code is archived in `frozen_training/feature_preparation_v2_1/`. It is historical, path-bound source, not a standalone feature-extraction installer. Model/HH-suite databases and sequence-search outputs are not included, and this package does not claim FASTA-to-feature reproducibility on its own.

The portable inference wrapper accepts `.npy` arrays with pickle loading disabled. A supplied FASTA must contain exactly one sequence and match L; identity/order of the actual feature rows remains the data preparer's responsibility. There is no fixed architectural maximum L, but whole-chain BiLSTM memory/time and actual validation lengths limit practical applicability; no arbitrary-long-chain accuracy guarantee is made.

Labels, when separately preparing training data, must be L-long with an explicit boolean mask. Missing experimental or MD targets must not be interpolated or set to zero for loss/scoring. Raw predictions may fall outside [0,1]; frozen reporting keeps them unclipped and reports out-of-range fractions.
