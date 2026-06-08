# Stage 1 — Block-wise MLP for Thermoelectric Property Prediction

## Overview

Stage 1 is the foundation of the PINN pipeline. It replaces the flat Sequential
model from v4.py with a **block-wise Multi-Layer Perceptron (MLP)** built using
the Keras Functional API. The model takes 4 separate groups of physically related
features as inputs, processes each group through its own sub-network, merges them,
and predicts 4 thermoelectric targets simultaneously.

No physics constraints are added at this stage. The goal here is to get the
architecture, data pipeline, and preprocessing right before physics losses are
introduced in Stages 2–4.

---

## Dataset

| Property | Value |
|---|---|
| Source | `thermoelectric_ml_ready.csv` |
| Rows | 24,024 |
| Columns | 57 |
| Rows used (after target filter) | 24,019 |
| Train / Test split | 80% / 20% (19,215 train, 4,804 test) |

The ML-ready subset contains only materials that have Seebeck coefficient data.
Within this subset, all 8 thermoelectric properties have 100% coverage.

---

## Targets

ZT is not present in the dataset directly — it is computed from existing columns:

```
ZT_p = p-powerfact × T / pkappa     at T = 300 K
```

All four targets are transformed before training to make them easier to learn:

| Raw Target | Transformed Target | Transform | Why |
|---|---|---|---|
| `p-Seebeck` | `p-Seebeck_abs` | Absolute value | Sign only indicates n/p type; magnitude is what matters |
| `pcond` | `pcond_log` | `log10(x + 1)` | Spans many orders of magnitude (1 to ~100,000) |
| `pkappa` | `pkappa_log` | `log10(x + 1)` | Same reason — highly skewed distribution |
| `ZT_p` | `ZT_p_log` | `log10(ZT + 1)` | Compresses [0, 10] to [0, 1.04]; avoids near-zero variance |

---

## Feature Blocks

Features are split into 4 groups based on physical meaning. Each group is fed
into its own sub-network before being merged. This is the key architectural
difference from a flat MLP.

---

### Block 1 — Structural (17 features)

These describe the crystal structure and thermodynamic stability of the material.

| Feature | Description |
|---|---|
| `nat` | Number of atoms in the unit cell |
| `density` | Mass density (g/cm³) |
| `exfoliation_energy` | Energy cost to exfoliate a 2D layer (meV/atom) — NaN for 3D materials |
| `formation_energy_peratom` | Thermodynamic stability — how much energy is released when the material forms |
| `ehull` | Energy above the convex hull — how far the material is from the most stable phase |
| `spg_number_norm` | Space group number (1–230) normalized to [0, 1] |
| `dimensionality_enc` | Dimensionality encoded as integer: 3D=3, 2D=2, 1D=1, 0D=0 |
| `crys_cubic` | One-hot: is the crystal system cubic? |
| `crys_hexagonal` | One-hot: is the crystal system hexagonal? |
| `crys_monoclinic` | One-hot: is the crystal system monoclinic? |
| `crys_orthorhombic` | One-hot: is the crystal system orthorhombic? |
| `crys_tetragonal` | One-hot: is the crystal system tetragonal? |
| `crys_triclinic` | One-hot: is the crystal system triclinic? |
| `crys_trigonal` | One-hot: is the crystal system trigonal? |
| `exfoliation_energy_missing` | Binary flag: 1 if exfoliation_energy was missing (i.e. 3D material) |

**Why this block matters:** Crystal symmetry and stability determine what
transport mechanisms are even possible. High-symmetry crystals (cubic, trigonal)
tend to have higher carrier mobility. The energy above hull tells us if the
material is synthesizable — an unstable material with great ZT is useless.

**Sub-network size: Dense(32)**

---

### Block 2 — Electronic (15 features)

These describe the electronic structure — how electrons behave in the material.

| Feature | Description |
|---|---|
| `optb88vdw_bandgap` | Bandgap from OptB88vdW DFT functional (eV) — primary bandgap estimate |
| `mbj_bandgap` | Bandgap from modified Becke-Johnson functional (eV) — more accurate for semiconductors |
| `hse_gap` | Bandgap from HSE hybrid functional (eV) — most accurate but computed for fewer materials |
| `avg_elec_mass` | Average electron effective mass (in units of electron rest mass) |
| `avg_hole_mass` | Average hole effective mass (in units of electron rest mass) |
| `epsx` | Static dielectric constant along x-axis |
| `epsy` | Static dielectric constant along y-axis |
| `epsz` | Static dielectric constant along z-axis |
| `mepsx` | Electronic part of dielectric constant along x-axis |
| `mepsy` | Electronic part of dielectric constant along y-axis |
| `mepsz` | Electronic part of dielectric constant along z-axis |
| `slme` | Spectroscopic limited maximum efficiency — solar cell efficiency proxy |
| `spillage` | Topological spillage — indicator of topologically non-trivial band structure |
| `magmom_oszicar` | Magnetic moment from OSZICAR file (μB) |
| `magmom_outcar` | Magnetic moment from OUTCAR file (μB) — more converged estimate |

**Why this block matters:** The Seebeck coefficient and electrical conductivity
are direct consequences of the electronic structure. Effective masses determine
carrier mobility — lighter electrons → higher conductivity. Bandgap determines
whether a material is semiconducting (good for thermoelectrics) or metallic.
The three bandgap columns capture the same property at different levels of DFT
accuracy, giving the model a richer picture.

**Sub-network size: Dense(64)** — largest block because electronic features are
the most information-rich for thermoelectric prediction.

---

### Block 3 — Thermal / Transport (8 features)

These describe how heat and charge move through the material — the most directly
relevant features for thermoelectric performance.

| Feature | Description |
|---|---|
| `n-Seebeck` | n-type Seebeck coefficient (μV/K) — used as a cross-type predictor for p-type |
| `n-powerfact` | n-type power factor S²σ (μW/mK²) at 600K, 10²⁰/cm³ doping |
| `ncond` | n-type electrical conductivity (S/m) |
| `nkappa` | n-type thermal conductivity (W/mK) |
| `optb88vdw_total_energy` | Total DFT energy of the unit cell (eV) |
| `Tc_supercon` | Superconducting transition temperature (K) — NaN for non-superconductors |
| `max_ir_mode` | Maximum infrared-active phonon mode frequency (cm⁻¹) |
| `min_ir_mode` | Minimum infrared-active phonon mode frequency (cm⁻¹) |

**Why n-type features predict p-type targets:** The paper (Choudhary et al.)
showed a Spearman correlation of 0.71 between n-type and p-type Seebeck
coefficients across the dataset. Materials with good n-type transport generally
also have good p-type transport due to shared band structure. IR phonon modes
are included because low-frequency phonon modes are responsible for high lattice
thermal conductivity — a key factor in ZT.

**Sub-network size: Dense(32)**

---

### Block 4 — Mechanical (10 features)

These describe the mechanical and piezoelectric properties of the material.

| Feature | Description |
|---|---|
| `bulk_modulus_kv` | Bulk modulus in Voigt approximation (GPa) — resistance to compression |
| `shear_modulus_gv` | Shear modulus in Voigt approximation (GPa) — resistance to shear |
| `poisson` | Poisson's ratio — lateral strain response |
| `dfpt_piezo_max_dielectric` | Maximum total dielectric constant from DFPT |
| `dfpt_piezo_max_dielectric_electronic` | Electronic contribution to dielectric constant |
| `dfpt_piezo_max_dielectric_ionic` | Ionic contribution to dielectric constant |
| `dfpt_piezo_max_eij` | Maximum piezoelectric stress coefficient eij (C/m²) |
| `dfpt_piezo_max_dij` | Maximum piezoelectric strain coefficient dij (pC/N) |
| `max_efg` | Maximum electric field gradient at nuclear sites |
| `efg` | Mean electric field gradient |

**Why this block matters:** Soft materials (low bulk/shear modulus) tend to have
low lattice thermal conductivity because phonons scatter more easily — this
directly helps ZT. The Grüneisen parameter (anharmonicity) correlates with
Poisson's ratio. Dielectric constants from DFPT complement the electronic block's
static dielectric values and capture ionic screening effects.

**Sub-network size: Dense(16)** — smallest block; mechanical properties are
useful but less directly predictive than electronic features.

---

## Preprocessing Pipeline

```
Raw CSV (24,024 rows, 57 cols)
    │
    ├─ [1] Force pd.to_numeric on all feature/target columns
    │       → converts string "na"/"None" to proper NaN
    │
    ├─ [2] Encode categoricals
    │       crys       → one-hot (7 columns)
    │       spg_number → divide by 230 → [0, 1]
    │       dimensionality → integer map {3D:3, 2D:2, 1D:1, 0D:0}
    │
    ├─ [3] Compute ZT_p = p-powerfact × 300 / pkappa, clip to [0, 10]
    │
    ├─ [4] Concat all blocks into df_work, deduplicate columns immediately
    │
    ├─ [5] Drop rows missing any target → 24,019 rows remain
    │
    ├─ [6] Add missingness flags for 4 sparse columns
    │       (exfoliation_energy, slme, Tc_supercon, spillage)
    │
    ├─ [7] Median imputation per column
    │       using np.nanmedian on .values array (guaranteed scalar)
    │
    ├─ [8] Outlier clipping at Q1 − 3×IQR and Q3 + 3×IQR per column
    │
    ├─ [9] Target transforms (abs, log10)
    │
    └─ [10] RobustScaler per block (fit on train, apply to test)
            RobustScaler uses median/IQR instead of mean/std
            → robust to remaining outliers after clipping
```

---

## Neural Network Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│                                                                 │
│  Structural (17) │ Electronic (15) │ Thermal (8) │ Mech (10)   │
└────────┬─────────┴───────┬─────────┴──────┬──────┴──────┬──────┘
         │                 │                │             │
         ▼                 ▼                ▼             ▼
   ┌──────────┐      ┌──────────┐    ┌──────────┐  ┌──────────┐
   │ Dense(32)│      │ Dense(64)│    │ Dense(32)│  │ Dense(16)│
   │   ReLU   │      │   ReLU   │    │   ReLU   │  │   ReLU   │
   │ L2(1e-4) │      │ L2(1e-4) │    │ L2(1e-4) │  │ L2(1e-4) │
   │    BN    │      │    BN    │    │    BN    │  │    BN    │
   └────┬─────┘      └────┬─────┘    └────┬─────┘  └────┬─────┘
        │                 │               │              │
        └────────┬─────────┘               └──────┬───────┘
                 └──────────────┬──────────────────┘
                                │
                          Concatenate
                         (32+64+32+16 = 144)
                                │
                         ┌──────▼──────┐
                         │  Dense(256) │
                         │    ReLU     │
                         │  L2(1e-4)   │
                         │     BN      │
                         │  Dropout30% │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  Dense(128) │
                         │    ReLU     │
                         │  L2(1e-4)   │
                         │     BN      │
                         │  Dropout20% │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  Dense(64)  │
                         │    ReLU     │
                         │  Dropout10% │
                         └──────┬──────┘
                                │
              ┌─────────────────┼──────────────────┐─────────────┐
              ▼                 ▼                  ▼             ▼
        ┌──────────┐      ┌──────────┐      ┌──────────┐  ┌──────────┐
        │ Dense(1) │      │ Dense(1) │      │ Dense(1) │  │ Dense(1) │
        │  linear  │      │  linear  │      │  linear  │  │  linear  │
        └────┬─────┘      └────┬─────┘      └────┬─────┘  └────┬─────┘
             │                 │                  │              │
        p-Seebeck_abs     pcond_log          pkappa_log      ZT_p_log

                        OUTPUT LAYER (4 targets)
```

---

## Component Explanations

### ReLU Activation
Rectified Linear Unit: `f(x) = max(0, x)`. Outputs zero for negative inputs,
identity for positive. Used in all hidden layers because it avoids the vanishing
gradient problem that plagued older activations like sigmoid/tanh, and trains
much faster in deep networks.

### Linear Activation (output heads)
The output layer uses no activation — raw linear output. This is correct because
our targets are continuous real numbers with no hard bounds (after log transform
they can be any real value). Using sigmoid or softplus would artificially cap the
range.

### Batch Normalization (BN)
After each Dense layer in the sub-blocks and shared trunk. Normalizes the
activations of each mini-batch to have mean≈0 and variance≈1. This:
- Prevents internal covariate shift (activations drifting during training)
- Allows higher learning rates
- Acts as a mild regularizer
- Stabilizes training significantly, especially with the diverse scales across
  our 4 feature blocks

### Dropout
Randomly sets a fraction of neurons to zero during each training batch:
- 30% dropout after Dense(256) — strongest regularization at the widest layer
- 20% dropout after Dense(128)
- 10% dropout after Dense(64) — lightest at the bottleneck before outputs

Dropout is disabled during inference (`model.predict`). It prevents the model
from relying on any single neuron, forcing redundant representations — reduces
overfitting.

### L2 Regularization (`kernel_regularizer=l2(1e-4)`)
Adds `λ × sum(weights²)` to the loss function. Penalizes large weights, keeping
them small and distributed. Applied to all Dense layers except the output heads.
`1e-4` is a mild penalty — enough to regularize without hurting capacity.

### RobustScaler
Used instead of StandardScaler because materials data has heavy-tailed
distributions with outliers even after IQR clipping. RobustScaler uses
median and interquartile range instead of mean and standard deviation,
making it insensitive to remaining extreme values. A separate scaler is
fit per block so each block's features are independently normalized.

### Separate Output Heads
Each of the 4 targets gets its own `Dense(1)` layer branching from the same
`Dense(64)` trunk. This matters because:
- Each property has different sensitivity to the shared representation
- Gradients for each target flow back independently through their own head
- The model can weight each target's contribution separately
- In Stages 2–4, the physics loss will need individual predictions for S, σ, κ
  to compute `ZT = S²σT/κ` — separate heads make this trivial

---

## Training Configuration

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimizer | Adam (lr=0.001) | Adaptive learning rate, standard for deep learning |
| Loss | MSE | Penalizes large errors strongly; appropriate for regression |
| Batch size | 64 | Balance between gradient noise and memory |
| Max epochs | 300 | Upper bound; early stopping usually triggers first |
| Early stopping patience | 30 epochs | Stops if val_loss doesn't improve for 30 epochs |
| LR reduction factor | 0.5 | Halves learning rate on plateau |
| LR reduction patience | 15 epochs | Triggers LR halving after 15 stagnant epochs |
| Min learning rate | 1e-6 | Floor to prevent LR from collapsing to zero |
| Validation split | 20% of train | ~3,843 samples held out during training |

---

## Results

| Target | MAE | R² | Notes |
|---|---|---|---|
| p-Seebeck_abs (μV/K) | 25.43 | 0.935 | Strong — 93.5% variance explained |
| pcond_log | 0.133 | 0.967 | Excellent |
| pkappa_log | 0.098 | 0.964 | Excellent |
| ZT_p_log | ~0.000 | — | R² unreliable (near-zero variance) |

**ZT additional metrics:**
- Pearson r = **0.980** — model ranks materials by ZT with 98% correlation to truth
- Spearman r = **0.952** — rank-order correlation, robust to outliers

**Mean R² across first 3 targets: 0.955**

Pearson/Spearman are the right metrics for ZT because most materials cluster
near ZT≈0 — only exceptional thermoelectrics exceed ZT=1. What matters
scientifically is whether the model correctly identifies which materials rank
higher, not the absolute value. A Pearson r of 0.98 means the model is
effectively screening materials by thermoelectric potential.

---

## What Stage 1 Does NOT Do

- **No physics constraints** — the model does not know that `ZT = S²σT/κ`. It could
  predict values of S, σ, κ, and ZT that are mathematically inconsistent with each
  other and would never be flagged.
- **No guarantee of physical plausibility** — a prediction where `pkappa = 0` and
  `ZT = 5` with low `p-powerfact` is mathematically impossible but Stage 1 would
  not penalize it.

These limitations are addressed in Stages 2–4 by adding physics residual terms
to the loss function.

---

## Files

| File | Description |
|---|---|
| `stage1_blockwise_mlp.py` | Full Stage 1 training script |
| `thermoelectric_ml_ready.csv` | Input dataset (24,024 rows, 57 columns) |

---

## How to Run

```bash
pip install tensorflow scikit-learn pandas numpy scipy
python stage1_blockwise_mlp.py
```

Place `thermoelectric_ml_ready.csv` in the same directory as the script.
Expected runtime: 5–15 minutes depending on hardware (early stopping usually
triggers around epoch 80–120).