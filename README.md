# MD-SOP v2.2

**Sequence-based residue-level order-parameter prediction using MD-iRED and S-OPPE supervision.**

[中文说明](README.zh-CN.md) · [Model card](docs/MODEL_CARD.md) · [Feature contract](docs/FEATURES.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Results](results/RESULTS.md)

This is the frozen release preparation for experiment `v2_2_20261107`, assembled on 2026-11-08. It contains all six prespecified models, not a model selected retrospectively on Experimental10. No GitHub repository URL, DOI, or publication citation has been assigned by this preparation.

## What changed in v2.2?

The MD-iRED labels were replaced: within each simulation replica, S² was averaged over all complete, non-overlapping **5τc windows**, then the three replica means were equally averaged. Sequence features, homology split, masks, architecture, initialization and training recipe remained fixed relative to v2.1. S-OPPE and experimental labels did not change.

Features are **ESM-C 600M (1152) + HMM-v1-legacy30 (30) = 1182 dimensions per residue**. The model is a 128-dimensional projection followed by a two-layer BiLSTM (256 units per direction) and either a residue-wise (`w1`) or width-15 convolutional (`w15`) regression head. There is no VAE or contrastive pretraining stage in this MD-SOP release.

| Model family | Training supervision | Heads |
|---|---|---|
| `A_MD` | MD-iRED 5τc labels only | w1, w15 |
| `B_SOPPE` | S-OPPE labels only | w1, w15 |
| `C_MIX` | MD + S-OPPE, equal source loss weights | w1, w15 |

## Quick start: precomputed features → S²

Tested training/export environment: Python 3.9.20, PyTorch 2.5.1, NumPy 1.26.4; original training used CUDA 11.8. CPU inference does not require a GPU. See [environment instructions](docs/ENVIRONMENT.md) before installing; no system CUDA changes are required by this repository.

```bash
# Run from the repository root, in your dedicated Python environment.
python scripts/verify_release.py
python -m unittest discover -s tests -v

# Synthetic packaging example only: not a biological prediction benchmark.
python code/predict_features.py --model A_MD_w1 \
  --features examples/synthetic_features.npy --device cpu \
  --output outputs/synthetic_A_MD_w1.tsv

# Real inference: supply matching ESM-C and legacy HMM arrays for one chain.
python code/predict_features.py --model A_MD_w1 \
  --esmc /path/to/protein.esmc1152.npy --hmm /path/to/protein.hmm30.npy \
  --fasta /path/to/protein.fasta --device cpu \
  --output outputs/protein_s2.tsv
```

All six model IDs are accepted (`A_MD_w1`, `A_MD_w15`, `B_SOPPE_w1`, `B_SOPPE_w15`, `C_MIX_w1`, `C_MIX_w15`). Outputs are raw, unclipped, unsmoothed predictions at 1-based full-sequence positions; they are not PDB residue numbers. Existing output files are never silently overwritten. Row counts alone cannot establish residue identity: the feature preparer must preserve the exact sequence and residue order.

**This is not a one-command FASTA-to-prediction package.** The supplied wrapper accepts precomputed features. ESM-C weights, HH-suite, its search database and feature caches are not bundled. Arbitrary 30-dimensional HMM encodings are not interchangeable with the frozen legacy30 representation.

## Contents and frozen identities

```text
code/                  Portable feature-input inference + unchanged model definition
checkpoints/           Six inference-only exports, registry and tensor SHA256
configs/               Original frozen scientific configurations
frozen_training/       Original version-bound training/evaluation/preparation code
results/               18 primary scores, controls, histories and PNG/SVG figures
data/metadata/         Split IDs, sequence hashes and dataset counts; not raw labels
docs/                  Methods, caveats, data availability and publication checklist
manifests/             Scientific freeze and full repository file checksums
tests/                 CPU-only inference/feature-contract tests
```

The inference exports omit optimizer/scheduler/RNG state but retain **exactly the same model tensors** as the accepted best checkpoints. Source checkpoint SHA256, tensor SHA256 and best epoch are recorded for every model. No fine-tuning, quantization, smoothing or calibration was applied during packaging.

The original training code is intentionally **not path-rewritten**: its absolute project paths and fail-closed hash gates describe the historical run. It requires the original bound datasets and audit tree; it is not a portable full-training launcher. Do not bypass the gates or overwrite the frozen run. See [reproducibility boundaries](docs/REPRODUCIBILITY.md).

## Results and limitations

All six models are reported on MD-test (136 records), S-OPPE-test (273 records) and Experimental10 (10 proteins, 709 measured positions). Primary metrics are protein-record macro averages. Within v2.2, A_MD_w1 has the lowest MD/experimental RMSE point estimates (0.131770/0.112190), while B_SOPPE_w1 has the lowest S-OPPE-test RMSE (0.062105). Mixed training and w15 are not uniformly better.

These are **single-training-seed, historically reused holdout evaluations**, not a new blind benchmark. The experimental panel contains eight homology groups; its 24-comparison exact sign-flip/Holm family cannot attain 0.05 rejection resolution. Historical SOPPCL references have different training pipelines and known overlap concerns. Full100ns auxiliary scores are separate from the new 5τc primary target. See the model card and complete tables rather than treating any one metric as a universal ranking.

## Publication status

This directory is a locally prepared frozen package; no upload, remote repository creation or public disclosure was performed during preparation. Repository owner/name, visibility, software/weight licensing and final data-redistribution scope require owner confirmation. No license is invented here; see [LICENSE_PENDING.md](LICENSE_PENDING.md). Do not add credentials, private audit folders or raw datasets to a future commit.
