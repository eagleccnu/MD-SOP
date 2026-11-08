# Reproducibility layers and boundaries

## 1. Frozen release integrity and score aggregation

```bash
python scripts/verify_release.py
```

Uses the standard library only. Verifies file sizes/SHA256, all18 primary score slots,2514 per-protein records and primary macro aggregation, plus a heuristic token/private-key/private-endpoint scan. This is not a substitute for a human security/license review.

## 2. Portable feature-input inference

```bash
python -m unittest discover -s tests -v
python code/predict_features.py --model A_MD_w1 \
  --features examples/synthetic_features.npy --device cpu \
  --output outputs/synthetic_A_MD_w1.tsv
```

All six inference exports are verified against their file/tensor digests. Preparation also compares every exported tensor to the original best and compares CPU forward outputs on deterministic synthetic features. No training/optimization is performed by these release tests. The example is explicitly synthetic, not a biological feature cache.

## 3. Historical training/evaluation code

`frozen_training/` and `configs/` are byte-preserved scientific source/configuration, with original SHA256 in `manifests/source_snapshots.tsv`. Absolute workstation paths and the version-specific fail-closed gates are intentionally retained. Workspace/SSH-bootstrap scripts and operational host manifests are excluded from upload. Original server addresses, credentials, raw logs and private audit snapshots are not needed to run the portable inference wrapper.

Historical formal training uses `03_train_md_sop_v2_2.py`, `resolved_train_v2_2.yaml`, source-specific development manifests, untrained paired initialization and step3/4 acceptance gates. Evaluation uses `04_evaluate_md_sop_v2_2.py` and phase-specific model freeze manifests; step6/7 comparison modules implement fixed-target rescores and grouped statistics.

This repository alone does **not** recreate the full training dataset, remote dependency tree or original binding hierarchy. Those original paths are provenance, not a recommendation to create matching absolute directories on another machine. Training scripts cannot be launched portably just by changing one path. A new full retraining must establish a new run identifier and re-freeze all bindings after data preparation; do not relax hash gates or overwrite the accepted v2.2 run.

## 4. What is and is not fixed

Scientific configuration, masks/split identities, six checkpoint tensors, histories and score tables are frozen. Source checkpoint file SHA differs from inference-export file SHA because optimizer/scheduler/RNG state was intentionally omitted; canonical model-state SHA is equal. Inference checkpoints are not resume checkpoints.

Floating-point kernels may differ across devices/backends. GPU cuDNN stochastic cross-process resume is not promised bitwise identical; the original six runs were uninterrupted. No multi-seed validation or newly blind test was performed during release preparation. Synthetic smoke tests are packaging checks, not additional evidence of prediction accuracy.

## 5. Result preservation

All six routes and primary18 scores are retained. New/old models are compared only on the same target column; full100ns is auxiliary. `results/figures/` contains original frozen PNG/SVG figures, including loss trends and experimental per-protein curves. No figure was regenerated over the official source outputs during preparation. The new publication summary is derived from already accepted tables.

Private source-to-release path mappings, complete run/audit snapshots, raw target/feature/checkpoint pointers and before/after integrity receipts remain outside the upload repository. They are not a part of a future Git commit.
