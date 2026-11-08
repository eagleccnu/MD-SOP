# Environment and installation

Frozen training/export core: Python3.9.20, PyTorch2.5.1, CUDA runtime11.8, cuDNN90100, NumPy1.26.4, SciPy1.12.0, pandas2.2.2, matplotlib3.9.2, PyYAML6.0.2, pyarrow17.0.0. Original hardware: RTX4090, driver535.104.05. `environment/training_core_versions.json` records the historical values, not a promise that every transitive dependency is locked.

Use a dedicated environment rather than changing existing system software. Python3.9 is retained to document the historical experiment, not advertised as a current long-term support recommendation. This preparation installed or upgraded nothing on the workstation.

For CPU inference in an existing suitable Python environment:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
```

For the original CUDA11.8 PyTorch build:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-training.txt
```

These commands specify the relevant subset of [official PyTorch2.5.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v251); torchvision/torchaudio are not used. CUDA builds require a compatible NVIDIA driver. CPU inference does not require system CUDA. No fresh installation was tested by this packaging task; the existing frozen workstation environment was used for export/equivalence/smoke tests.

ESM-C SDK/weights and HH-suite/database are separate feature-preparation dependencies. Their absence from the model-training environment is expected for precomputed inputs. Requirements files are core pins, not redistributable copies of packages, CUDA libraries or third-party licenses.
