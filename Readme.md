# Neuro-NeRF: 3D Brain Reconstruction from Volumetric MRI

## 🎯 Project Motivation

The core motivation of this project is to generate a high-fidelity 3D model of
a human brain from 2D MRI scans. We apply the **Neural Radiance Fields (NeRF)**
approach by **overfitting a Multilayer Perceptron (MLP)** to a stack of 2D
slices.

Unlike traditional methods that store the brain as a static voxel grid, this
project treats it as an **Implicit Neural Representation (INR)** — the 3D
structure is "memorised" inside the network's weights as a continuous function:

$$f(x, y, z) \rightarrow \text{Intensity}$$

Once the network has converged, we can reconstruct the volume at *any*
coordinate, including positions between the original slices (super-resolution
along Z).

## 🏁 Goals

1. **Reconstruction quality** — measured by PSNR on held-out validation slices
2. **Inter-slice interpolation** — render anatomically plausible content at Z
   positions the model has never seen
3. **3D geometry extraction** — produce a Marching-Cubes mesh of the implicit
   surface for interactive viewing

## 🛠 Architecture

The codebase follows a strictly modular design — each module owns one concern,
with clear API boundaries documented in the file headers.

![Software Architecture](documentation/image.png)

| Module                                | Role                                                 |
|---------------------------------------|------------------------------------------------------|
| `input/data.py`                       | NIfTI / phantom volume providers (GPU-resident)      |
| `representation/model.py`             | `NeuralField` MLP + sinusoidal encoding              |
| `representation/sampler_def.py`       | 2D-pixel → 3D-coordinate strategies (Point, RaySlab) |
| `representation/trainer_def.py`       | Optimizer, loss, gradient clipping, PSNR metric      |
| `representation/train_loop.py`        | Train/val split, checkpointing, TensorBoard, early stopping |
| `output/renderer.py`                  | Slice rendering and ground-truth comparison plots    |
| `output/visualizer.py`                | Marching-Cubes mesh extraction + interactive Plotly viewer |

For the full technical reference (model details, sampler strategies, loss
formulation, hyperparameter rationale, PSNR interpretation) see
**[`documentation/training.md`](documentation/training.md)**.

## 🧪 Scientific Background

### What is an MRI?

Magnetic Resonance Imaging measures the signal intensity of hydrogen protons in
the body. The scanner produces a stack of 2D "slices". A significant challenge
is **slice thickness** (anisotropy) — useful information is lost in the gaps
between slices.

### Our approach vs. classical NeRF

| Aspect           | Classical NeRF                       | Neuro-NeRF                          |
|------------------|--------------------------------------|-------------------------------------|
| Input            | 2D photographs from external cameras | 2D internal MRI slices              |
| Predicts         | RGB color + density (opacity)        | Scalar tissue intensity             |
| Occlusion        | Yes (solid surfaces block each other)| No (volumetric, transparent data)   |
| Use case         | Novel-view synthesis                 | Inter-slice super-resolution        |

The volumetric `RaySlabSampler` corrects for the partial-volume effect by
treating each pixel as a thick column of tissue, integrating the implicit field
across the slab thickness — see [`documentation/training.md`](documentation/training.md).

---

## 🚀 Getting Started

### Option A — Google Colab (recommended)

The repo ships with a self-contained notebook that mounts Google Drive,
downloads the brain, runs training, and renders results. On a free T4 GPU the
full run takes 2–4 hours.

[**Open `colab_train.ipynb` in Colab →**](https://colab.research.google.com/github/TimurHegwein/Nerf---Neural-Radiance-Fields/blob/feat/colab-integration/colab_train.ipynb)

> Make sure to set **Runtime → Change runtime type → GPU** before running.

### Option B — Local

```bash
# 1. Install dependencies
pip install torch numpy matplotlib nibabel nilearn scikit-image tensorboard plotly

# 2. Download the ICBM 152 brain template
python fetch_brain.py

# 3. Train
python main.py

# 4. Monitor training (in a second terminal)
tensorboard --logdir runs/
```

### Outputs

| Path                                   | What it contains                          |
|----------------------------------------|-------------------------------------------|
| `checkpoints/brain_0.pth`              | Best model + config + metadata            |
| `runs/brain_0_experiment/`             | TensorBoard logs (loss, PSNR, LR)         |
| Inline matplotlib                      | GT vs Reconstruction vs Interpolation grid|

### Loading a trained model

```python
from output.visualizer import load_neural_field
model = load_neural_field("checkpoints/brain_0.pth", device="cuda")
# Render a slice at any Z position:
from output.renderer import NeuroRenderer
img = NeuroRenderer(model).render_slice(z_pos=0.3, resolution=256)
```

---

## 🗺 Roadmap

- [x] **Phase 1:** Modular MVP architecture with synthetic phantom data
- [x] **Phase 2:** GPU-resident data pipeline, RaySlab sampler, TensorBoard
- [x] **Phase 3:** Real human brain MRI (NIfTI) with stratified val split
- [x] **Phase 4:** Reproducible training (seeded), self-describing checkpoints, Colab notebook
- [ ] **Phase 5:** Hash-grid encoding (Instant-NGP) for ~10× faster convergence
- [ ] **Phase 6:** Held-out contiguous-block evaluation for honest extrapolation metrics
- [ ] **Phase 7:** Nerfstudio integration for real-time interactive visualisation

---

*Created for the course **Deep Learning for Computer Graphics** — TUM*
