# Training & Metrics — Technical Reference

This document explains *why* the training pipeline looks the way it does — model
architecture, loss formulation, sampler strategies, and how to read the metrics.

---

## Model: `NeuralField`

A coordinate-based MLP that maps a 3D point $(x, y, z) \in [-1, 1]^3$ to a
scalar intensity $\in [0, 1]$.

```
Input (x, y, z) ──► Sine Encoding ──► Linear → LayerNorm → SiLU
                                       (×N layers, with one concat-skip)
                                       ──► Linear → Sigmoid ──► Intensity
```

### Sine Encoding (Positional Encoding)

A bare MLP cannot represent high-frequency detail well — it has a low-frequency
inductive bias (the *spectral bias* from Rahaman et al. 2019). NeRF (Mildenhall
et al. 2020) addresses this by lifting each coordinate into a higher-dimensional
sinusoidal basis before the MLP sees it:

$$\gamma(p) = \big[\sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)\big]$$

With `num_freqs = 12`, each scalar coordinate becomes a 24-dim vector. The full
3D coordinate becomes a $3 \times 2 \times 12 = 72$-dim feature, which is what
the first MLP layer actually sees.

**Why exponential frequencies?** They cover all spatial scales from coarse
(skull outline) to fine (sulci, ventricles) without manual band tuning.

### Skip Connection

After every $\lfloor N/2 \rfloor$ layers we **concatenate** the encoded input
back into the hidden state. Without this, the high-frequency information from
the encoding gets diluted across many fully-connected layers.

The choice between **add** and **concat** matters: NeRF uses concat, which lets
the next layer linearly mix encoded coordinates and learned features however it
wants. Add-skip would force them onto the same scale, which fights the
LayerNorm before it.

### LayerNorm + SiLU

- **LayerNorm** stabilises activations across the wide hidden vector. Important
  with sinusoidal encodings, whose outputs span $[-1, 1]$ but the layers
  downstream can drift to wildly different scales.
- **SiLU** (`x * sigmoid(x)`, also known as Swish) is the de-facto default for
  coordinate MLPs — smoother gradient than ReLU and slightly better empirical
  convergence than GELU at this scale.

### Sigmoid Output

Forces the predicted intensity into $[0, 1]$, matching the percentile-clipped
NIfTI normalization. Avoids the model wandering off into negative or saturated
values.

---

## Sampler: 2D-to-3D Coordinate Lifting

Each MRI slice gives us pixel intensities at known (x, y, z) coordinates. The
sampler turns these pixels into supervision points for the network.

### `PointSampler` — discrete pixel sampling

Treats each pixel as an infinitesimal point at the centre of its slice. Random
pixels are drawn per training step. Simple, fast, but **ignores slice
thickness** — appropriate only for high-resolution isotropic scans.

### `RaySlabSampler` — volumetric integration *(default)*

Treats each pixel as a thin **slab** of tissue. For each ray:

1. Pick a random pixel and its (x, y) coordinate
2. Sample N points uniformly along the slab thickness (z-axis)
3. The model predicts intensities at all N points; their mean is compared to the
   pixel's actual intensity

This corrects for the **partial volume effect** — pixel intensities in real MRI
are line integrals through finite slab volume, not point measurements. Required
for accurate inter-slice interpolation.

#### How thick is one slab?

The slab thickness is derived from the volume's Z-resolution. For the ICBM 152
brain (189 slices along Z, ~1 mm physical spacing per slice from the NIfTI
header), each slab spans:

$$\text{thickness}_\text{norm} = \frac{2}{N_z} = \frac{2}{189} \approx 0.0106$$

in the normalised $[-1, 1]$ coordinate system, corresponding to roughly **1 mm**
of real tissue. With `num_samples_per_ray=8`, points are placed via
`torch.linspace(z_start, z_end, 8)` — endpoints inclusive, so 7 intervals — at
a spacing of:

$$\Delta z_\text{norm} = \frac{\text{thickness}}{7} \approx 0.00151 \;\;\; (\approx 0.14 \text{ mm})$$

So each ray queries the implicit field at 8 points roughly 0.14 mm apart along
Z, all sharing the same (x, y), and averages the predictions to get the
effective pixel intensity.

Default: `num_samples_per_ray=8`.

---

## Loss Function

$$\mathcal{L} = \underbrace{\frac{1}{B}\sum_i (f(p_i) - y_i)^2}_{\text{MSE}} +
                \lambda \underbrace{\mathbb{E}\big[|f(p) - f(p + \epsilon)|\big]}_{\text{TV (smoothness)}}$$

### MSE term

Direct supervision — predicted intensity should match the slab-integrated
ground-truth pixel value. PSNR is computed from this term alone for clean
metric reporting.

### TV (Total Variation) regulariser

Probes random points and their nearby neighbours, penalising abrupt jumps in
predicted intensity. Encourages the network to learn a **smooth** continuous
function rather than memorising a noisy point-cloud.

The implementation samples a perturbation $\epsilon \sim \mathcal{N}(0, 0.01^2)$
rather than computing axis-aligned spatial derivatives — strictly speaking this
is a *neighbour-consistency* prior, not classical TV. The effect is similar.

**Tuning:** `tv_weight=1e-7` is mild, used to suppress speckle without
over-smoothing the cortical detail. Higher values blur sulci.

---

## Optimizer

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Optimizer | Adam | Standard for coordinate MLPs |
| Learning Rate | `2e-3` | Higher than the NeRF default (1e-3) — works because the data is highly redundant (one volume only) |
| LR Schedule | `ExponentialLR(gamma=0.998)` | Drops by 0.2% per epoch — gentle decay, prevents getting stuck in late phase |
| Gradient Clipping | `max_norm=1.0` | Defends against the occasional gradient explosion from the concat-skip |

---

## Train / Val Split

**Stratified per-Z-bin sampling** with seed 42:

1. The Z-axis is split into `n_val` equal-width bins
2. One random slice is drawn from each bin
3. Result: validation slices are uniformly distributed across the volume — no
   clustered val regions, no biased coverage

Why not pure random shuffle? With only 19 val slices out of 189, a random sample
can clump (e.g., all from the temporal lobe), making the val score
unrepresentative. Why not stride-based (every Nth slice)? It's deterministic
across runs and seeds, harder to ablate.

---

## Metrics: Reading PSNR

PSNR (Peak Signal-to-Noise Ratio) is a logarithmic transformation of MSE for
data normalised to $[0, 1]$:

$$\text{PSNR} = -10 \cdot \log_{10}(\text{MSE})$$

### Quick reference

| MSE     | PSNR    | Visual meaning                       |
|---------|---------|--------------------------------------|
| 1.0     | 0 dB    | Random output                        |
| 0.1     | 10 dB   | Outline only                         |
| 0.01    | 20 dB   | Blurry but recognisable              |
| 0.001   | 30 dB   | Good — small artefacts visible       |
| 0.0001  | 40 dB   | Very good — barely distinguishable   |

**Rule of thumb:** +6 dB ≈ MSE quartered ≈ "twice as accurate".

### What "good" looks like for this setup

| Phase            | Train PSNR | Val PSNR  | Gap     |
|------------------|------------|-----------|---------|
| Early (Ep 1–10)  | 14–18 dB   | 10–16 dB  | 3–6 dB  |
| Mid (Ep 50–100)  | 22–28 dB   | 20–26 dB  | 2–3 dB  |
| Late (Ep 200+)   | 28–32 dB   | 26–30 dB  | < 2 dB  |

**Train/Val gap interpretation:**

- **< 2 dB:** healthy generalisation, model is interpolating smoothly between slices
- **2–4 dB:** mild overfitting, still useful
- **> 5 dB and growing:** memorisation — early stopping should already have kicked in

### What PSNR doesn't measure

PSNR correlates poorly with perceptual similarity. A 1-pixel-shifted image has
low PSNR but looks identical to humans. For perceptual quality use SSIM or
LPIPS — but for medical reconstruction where exact intensity matters, PSNR is
the appropriate metric.

---

## Checkpoint Format

`torch.save` writes a dict with:

```python
{
    "state_dict": <weights>,
    "config":     {"encoding_type": ..., "num_freqs": 12, "hidden_dim": 512, "num_layers": 6},
    "best_val_loss": float,
    "best_epoch":    int,
    "split_seed":    42,
    "val_indices":   [list of held-out Z indices],
}
```

The architecture config is stored *with the weights* so `output/visualizer.py`
can reconstruct the exact `NeuralField` instance without hardcoding
hyperparameters. This avoids the silent-shape-mismatch class of bugs.

The best checkpoint is **flushed to disk every 25 epochs** when val improves —
on Colab Free, an idle disconnect costs at most ~5 minutes of training rather
than the entire run.

---

## References

- Mildenhall et al. (2020), *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*
- Tancik et al. (2020), *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*
- Rahaman et al. (2019), *On the Spectral Bias of Neural Networks*
- Sitzmann et al. (2020), *Implicit Neural Representations with Periodic Activation Functions* (SIREN)
