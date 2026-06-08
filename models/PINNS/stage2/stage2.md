# Thermoelectric Property Prediction — PINN Model (Stages 1–3)

A Physics-Informed Neural Network (PINN) built to predict thermoelectric transport properties from DFT-computed material descriptors. The model is developed in four progressive stages, each adding physical constraints on top of the previous one. This document covers Stages 1–3.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Target Properties](#3-target-properties)
4. [Input Features](#4-input-features)
5. [Target Transforms](#5-target-transforms)
6. [Stage 1 — Baseline Block MLP](#6-stage-1--baseline-block-mlp)
7. [Stage 2 — Power Factor Physics Loss](#7-stage-2--power-factor-physics-loss)
8. [Stage 3 — ZT Consistency Loss](#8-stage-3--zt-consistency-loss)
9. [Physics Constraints — Full Derivation](#9-physics-constraints--full-derivation)
10. [Why Log Space?](#10-why-log-space)
11. [Why Empirical Offsets?](#11-why-empirical-offsets)
12. [Results Summary](#12-results-summary)
13. [What Comes Next — Stage 4](#13-what-comes-next--stage-4)

---

## 1. Problem Statement

Thermoelectric materials convert heat into electricity. Their efficiency is governed by the dimensionless figure of merit:

$$ZT = \frac{S^2 \sigma}{\kappa} \cdot T$$

where:
- $S$ = Seebeck coefficient (µV/K) — the voltage generated per degree of temperature difference
- $\sigma$ = electrical conductivity (S/m)
- $\kappa$ = thermal conductivity (W/m·K)
- $T$ = temperature (K), fixed at 300 K here

Computing these properties from first principles (DFT) is expensive. The goal is to train a neural network that predicts all four properties simultaneously from cheaper structural and electronic descriptors, while enforcing that predictions are **physically self-consistent** — i.e., the predicted $S$, $\sigma$, $\kappa$ should actually produce the predicted $ZT$ according to the formula above.

A standard neural network ignores this constraint entirely. A PINN bakes it into the loss function.

---

## 2. Dataset

- Source: `thermoelectric_ml_ready.csv` — a compiled dataset of p-type thermoelectric properties from the JARVIS-DFT database
- Size: several thousand materials with DFT-computed properties
- Temperature: T = 300 K (all transport properties)
- Categorical columns encoded: `crys` (crystal system), `spg_number` (space group), `dimensionality` (0D/1D/2D/3D)

**Preprocessing:**
- Numeric coercion on all feature and target columns
- `ZT_p` computed as `p-powerfact * 300 / pkappa`, clipped to [0, 10]
- Rows dropped only if any of the 4 main targets are missing
- Missing value flags created for sparse columns (`exfoliation_energy`, `slme`, `Tc_supercon`, `spillage`)
- Median imputation for remaining NaNs
- Outlier clipping at IQR ± 3× per column

---

## 3. Target Properties

The model predicts four p-type thermoelectric properties:

| Target column | Physical quantity | Units | Transform applied |
|---|---|---|---|
| `p-Seebeck_abs` | Seebeck coefficient (absolute value) | µV/K | None (raw, after abs) |
| `pcond_log` | Electrical conductivity | S/m | log10(x + 1) |
| `pkappa_log` | Thermal conductivity | W/m·K | log10(x + 1) |
| `ZT_p_log` | Figure of merit ZT | dimensionless | log10(ZT + 1) |

All four are further scaled with `RobustScaler` before being fed as training targets (main heads output in scaled space; evaluation inverts the scaler).

---

## 4. Input Features

Features are organised into four domain-motivated **input blocks**. Each block gets its own sub-network before merging into a shared trunk. This allows the model to learn block-specific representations before integrating them.

### Block S — Structural (32 neurons)
Core structural descriptors plus crystal system encoding:

| Feature | Description |
|---|---|
| `nat` | Number of atoms in unit cell |
| `density` | Mass density (g/cm³) |
| `exfoliation_energy` | Energy to exfoliate a layer (eV/atom) |
| `formation_energy_peratom` | Thermodynamic stability (eV/atom) |
| `ehull` | Distance to convex hull — metastability |
| `crys_*` | One-hot: crystal system (cubic, hexagonal, etc.) |
| `spg_number_norm` | Space group number normalised to [0, 1] |
| `dimensionality_enc` | 0D=0, 1D=1, 2D=2, 3D=3 |
| `exfoliation_energy_missing` | Missingness flag |

### Block E — Electronic (64 neurons)
Electronic structure descriptors:

| Feature | Description |
|---|---|
| `optb88vdw_bandgap` | DFT bandgap (optB88vdW) |
| `mbj_bandgap` | mBJ corrected bandgap |
| `hse_gap` | HSE06 hybrid bandgap |
| `avg_elec_mass` | Average electron effective mass |
| `avg_hole_mass` | Average hole effective mass |
| `epsx/y/z` | Dielectric tensor components |
| `mepsx/y/z` | Electronic dielectric components |
| `slme` | Spectroscopic limited max efficiency |
| `spillage` | Topological spillage indicator |
| `magmom_oszicar/outcar` | Magnetic moment (two sources) |

### Block T — Thermal (32 neurons)
Thermal and energy descriptors:

| Feature | Description |
|---|---|
| `n-Seebeck` | n-type Seebeck (cross-type information) |
| `n-powerfact` | n-type power factor |
| `ncond` | n-type electrical conductivity |
| `nkappa` | n-type thermal conductivity |
| `optb88vdw_total_energy` | DFT total energy |
| `Tc_supercon` | Superconducting transition temperature |
| `max_ir_mode` / `min_ir_mode` | Phonon IR modes (thermal transport proxy) |

### Block M — Mechanical (16 neurons)
Mechanical and piezoelectric descriptors:

| Feature | Description |
|---|---|
| `bulk_modulus_kv` | Voigt bulk modulus |
| `shear_modulus_gv` | Voigt shear modulus |
| `poisson` | Poisson ratio |
| `dfpt_piezo_max_dielectric` | Max total dielectric (DFPT) |
| `dfpt_piezo_max_dielectric_electronic` | Electronic dielectric contribution |
| `dfpt_piezo_max_dielectric_ionic` | Ionic dielectric contribution |
| `dfpt_piezo_max_eij` | Max piezoelectric stress coefficient |
| `dfpt_piezo_max_dij` | Max piezoelectric strain coefficient |
| `max_efg` / `efg` | Electric field gradient |

---

## 5. Target Transforms

Raw thermoelectric properties span many orders of magnitude (e.g., conductivity can range from 10⁻² to 10⁶ S/m). Training a neural network directly on such ranges causes gradient instability. Three transforms are applied:

**Absolute value** for Seebeck: p-type Seebeck can be negative, but only the magnitude matters for ZT. Taking |S| removes the sign ambiguity.

**log10(x + 1)** for conductivity, kappa, and ZT: This compresses the dynamic range, makes the distribution more Gaussian, and — critically — makes the physics identity linear (see Section 9).

**RobustScaler** on all four transformed targets: Centers on median, scales by IQR. More robust than StandardScaler for heavy-tailed distributions common in materials data.

---

## 6. Stage 1 — Baseline Block MLP

**File:** `stage1_blockwise_mlp.py`

### What it does
Establishes the baseline architecture with no physics constraints. The goal is to confirm the block-wise Functional API design works and gives reasonable R².

### Architecture

```
inp_structural  (S_dim)  → Dense(32, ReLU) → BN  ─┐
inp_electronic  (E_dim)  → Dense(64, ReLU) → BN  ─┤
inp_thermal     (T_dim)  → Dense(32, ReLU) → BN  ─┤→ Concatenate
inp_mechanical  (M_dim)  → Dense(16, ReLU) → BN  ─┘
                                                     ↓
                                          Dense(256, ReLU) → BN → Dropout(0.3)
                                                     ↓
                                          Dense(128, ReLU) → BN → Dropout(0.2)
                                                     ↓
                                          Dense(64,  ReLU) → Dropout(0.1)
                                                     ↓
                          ┌──────────────────────────────────────┐
                          ↓          ↓           ↓          ↓
                     Dense(1)   Dense(1)    Dense(1)   Dense(1)
                     Seebeck    Cond        Kappa      ZT
                          └──────────────────────────────────────┘
                                       main_output (4)
```

- Activation: ReLU throughout, linear output heads
- Regularisation: L2(1e-4) on all Dense layers in trunk
- Batch Normalisation after each sub-block and each trunk layer
- Dropout: 0.3 → 0.2 → 0.1 (decreasing through trunk)
- Optimiser: Adam (lr=0.001)
- Loss: MSE on scaled targets
- Epochs: 300, batch size 64, early stopping patience 30

### Results

| Target | R² |
|---|---|
| p-Seebeck_abs | 0.935 |
| pcond_log | ~0.96 |
| pkappa_log | ~0.97 |
| ZT Pearson r | ~0.97 |
| **Mean R² (S, σ, κ)** | **0.955** |

This is the benchmark every subsequent stage must match or beat.

---

## 7. Stage 2 — Power Factor Physics Loss

**File:** `stage2_pinn_pf_loss.py`

### What it does
Adds the Power Factor physics constraint as an auxiliary loss term. This forces the model's predictions of Seebeck and conductivity to be internally consistent with the Power Factor identity:

$$PF = S^2 \cdot \sigma$$

### Why a custom training loop?
Keras's `.fit()` computes loss only on model outputs vs. labels. The physics residual is a derived quantity — it involves arithmetic between two output heads and a ground-truth column (`p-powerfact`). Keras cannot express this natively, so a manual `tf.GradientTape` training loop is used.

### The problem with earlier attempts
The first naive attempt computed the physics residual in **scaled space** — i.e., using RobustScaler-transformed values of $\hat{S}$ and $\hat{\sigma}$. This is meaningless: the scaler shifts and squashes $S$ and $\sigma$ independently, so $\hat{S}^2 \cdot \hat{\sigma}$ has no physical interpretation. The result was a physics loss of −2.7 billion, which flooded the entire model with garbage gradients and degraded Seebeck R² from 0.935 to 0.761.

### The fix — Auxiliary Heads in Log Space

Two additional output heads are added to predict $S$ and $\sigma$ in **unscaled log space**, purely for use in the physics residual:

| Aux Head | Predicts | Activation |
|---|---|---|
| `S_log_head` | log10(\|p-Seebeck\| + 1) | softplus (enforces ≥ 0) |
| `sig_log_head` | log10(pcond + 1) | softplus (enforces ≥ 0) |

These heads are trained with their own data loss ($L_{aux}$) against the ground-truth log-transformed values, independently of the main scaled targets.

### Physics Residual in Log Space

In log10 space, the PF identity becomes linear:

$$\log_{10}(PF + 1) \approx 2 \cdot \log_{10}(|S| + 1) + \log_{10}(\sigma + 1) - \text{UNIT\_OFFSET}$$

The **UNIT_OFFSET** absorbs the unit-conversion factor between the dataset columns (µV/K, S/m, µW/cm·K²) and ensures the residual starts near zero at initialisation. It is computed **empirically** from the data (median of $2 \cdot S\_log + sig\_log - \log_{10}(PF+1)$ over valid rows) — no hardcoding, no assumptions about unit systems.

### Loss Function

$$L_{total} = L_{data} + \lambda_{aux} \cdot L_{aux} + \lambda_{PF} \cdot L_{PF}$$

| Term | Formula | Weight | Purpose |
|---|---|---|---|
| $L_{data}$ | MSE(main\_pred\_scaled, y\_scaled) | 1.0 | Fit the 4 main targets |
| $L_{aux}$ | MSE(aux\_pred, [S\_log, sig\_log]) | λ_aux = 1.0 | Keep aux heads grounded |
| $L_{PF}$ | MSE(2·S\_log\_hat + sig\_log\_hat − offset, log\_PF\_true) | λ_PF = 0.1 | Enforce PF identity |

λ_PF = 0.1 is intentionally small — the physics loss is a **guide**, not the primary objective. Increasing it too much would let physics override data.

### Updated Architecture

```
[same 4-block trunk as Stage 1]
                    ↓
             Dense(64, ReLU) → Dropout(0.1)
                    ↓
   ┌────────────────────────────────────────────────┐
   ↓         ↓          ↓         ↓      ↓         ↓
Dense(1)  Dense(1)  Dense(1)  Dense(1) Dense(1) Dense(1)
Seebeck   Cond      Kappa     ZT       S_log    sig_log
[linear]  [linear] [linear] [linear] [softplus][softplus]
   └──── main_output (4) ────┘  └─── aux_output (2) ───┘
```

### Results

| Target | R² |
|---|---|
| p-Seebeck_abs | 0.9318 |
| pcond_log | 0.9687 |
| pkappa_log | 0.9613 |
| ZT Pearson r | 0.9783 |
| **Mean R² (S, σ, κ)** | **0.9539** |
| S_log (aux) | 0.9441 |
| sig_log (aux) | 0.9682 |
| **PF consistency R²** | **0.749** |

Stage 2 matches Stage 1 accuracy (mean R² 0.9539 vs 0.955) while adding partial physics consistency. PF R² of 0.749 is a large improvement over −12.5 but not yet tight — expected, because kappa is not yet in the loop.

---

## 8. Stage 3 — ZT Consistency Loss

**File:** `stage3_pinn_zt_loss.py`

### What it does
Adds the ZT consistency constraint. This is the most important physics loss because ZT is the target quantity that determines thermoelectric efficiency. It forces all three transport properties — $S$, $\sigma$, $\kappa$ — to be mutually consistent with the ZT formula.

### New Auxiliary Head

A third aux head is added for kappa:

| Aux Head | Predicts | Activation |
|---|---|---|
| `kappa_log_head` | log10(pkappa + 1) | softplus |

### ZT Physics Constraint

Starting from $ZT = PF \cdot T / \kappa = S^2 \sigma T / \kappa$, taking log10:

$$\log_{10}(ZT + 1) \approx 2 \cdot S\_log + sig\_log + \log_{10}(T) - \kappa\_log - \text{ZT\_OFFSET}$$

- $\log_{10}(T) = \log_{10}(300) \approx 2.477$ — a constant baked in at training time
- **ZT_OFFSET** is empirically computed (same method as UNIT_OFFSET) from rows where all three quantities and ZT are well-defined

### Loss Function

$$L_{total} = L_{data} + \lambda_{aux} \cdot L_{aux} + \lambda_{PF} \cdot L_{PF} + \lambda_{ZT} \cdot L_{ZT}$$

| Term | Weight | Reason |
|---|---|---|
| $L_{data}$ | 1.0 | Primary data fit |
| $L_{aux}$ | 1.0 | All 3 aux heads grounded |
| $L_{PF}$ | 0.1 | PF consistency (gentle) |
| $L_{ZT}$ | 1.0 | ZT consistency (strong — ZT is the goal) |

λ_ZT = 1.0 is higher than λ_PF because ZT is the primary metric. The kappa head adds a third anchor point, which also indirectly tightens PF consistency as a side effect.

### Updated Architecture

```
[same 4-block trunk]
                    ↓
             Dense(64, ReLU) → Dropout(0.1)
                    ↓
   ┌──────────────────────────────────────────────────────────┐
   ↓        ↓         ↓        ↓       ↓        ↓        ↓
Dense(1) Dense(1) Dense(1) Dense(1) Dense(1) Dense(1) Dense(1)
Seebeck   Cond    Kappa    ZT      S_log   sig_log  kappa_log
[linear] [linear][linear][linear][softplus][softplus][softplus]
   └──── main_output (4) ────┘   └────── aux_output (3) ──────┘
```

### Physics check output (Stage 3)

Two physics consistency scores are reported:

**PF check:** `2*S_log + sig_log - UNIT_OFFSET` vs `log10(PF_true + 1)`

**ZT check:** `2*S_log + sig_log + log10(T) - kappa_log - ZT_OFFSET` vs `log10(ZT_true + 1)`

Both should have R² closer to 1.0 than Stage 2 (especially ZT, which is now directly constrained).

---

## 9. Physics Constraints — Full Derivation

### Power Factor

In SI units:

$$PF = S^2 \cdot \sigma \quad [\text{W/m·K}^2]$$

Taking $\log_{10}$ of both sides (for large values where the +1 offset is negligible):

$$\log_{10}(PF) = 2 \log_{10}|S| + \log_{10}(\sigma)$$

The dataset uses mixed units (µV/K for $S$, S/m for $\sigma$, µW/cm·K² for PF). Rather than manually tracking conversion factors, an empirical offset is fit from the data:

$$\log_{10}(PF + 1) \approx 2 \cdot S\_log + sig\_log - \text{UNIT\_OFFSET}$$

where `UNIT_OFFSET` ≈ 6.0 (the log of the unit-conversion factor).

### ZT Figure of Merit

$$ZT = \frac{S^2 \sigma}{\kappa} \cdot T = \frac{PF \cdot T}{\kappa}$$

In log space:

$$\log_{10}(ZT) = \log_{10}(PF) + \log_{10}(T) - \log_{10}(\kappa)$$

Substituting the PF expression:

$$\log_{10}(ZT + 1) \approx 2 \cdot S\_log + sig\_log + \log_{10}(T) - \kappa\_log - \text{ZT\_OFFSET}$$

The ZT_OFFSET accounts for the accumulated unit offsets and the softening of the +1 terms.

---

## 10. Why Log Space?

Three reasons:

1. **Dynamic range.** Thermoelectric properties span orders of magnitude. Log-transforming puts them on a scale that neural networks can learn stably.

2. **Linearity of physics.** The power law $PF = S^2 \sigma$ becomes a linear identity $\log PF = 2 \log S + \log \sigma$ in log space. Linear physics constraints produce clean, bounded gradients. In raw space, the squaring operation produces gradients that explode with the magnitude of S.

3. **Consistent units for aux heads.** Both `S_log` and `sig_log` use the transform $\log_{10}(x + 1)$, putting them on the same scale (~0 to 7). This makes the MSE of the aux loss and the physics residual directly comparable, avoiding the need to tune scale-compensation hyperparameters.

---

## 11. Why Empirical Offsets?

The dataset mixes unit systems across columns. Hardcoding a unit-conversion factor would require:
- Knowing the exact units of every column
- Verifying consistency across the full dataset
- Manually re-deriving if the dataset changes

The empirical approach — computing `median(2*S_log + sig_log - log_PF)` from the data directly — is:
- **Robust:** uses the median, not the mean, so outliers don't corrupt it
- **Self-consistent:** derived from the exact same rows the model trains on
- **Zero-initialisation:** because the offset is subtracted, the physics residual starts near 0 at epoch 0, which means physics gradients start small and grow as the model learns — the right direction

---

## 12. Results Summary

| Metric | Stage 1 | Stage 2 | Stage 3 (expected) |
|---|---|---|---|
| Seebeck R² | 0.935 | **0.932** | ≥ 0.93 |
| Cond R² | ~0.96 | **0.969** | ≥ 0.96 |
| Kappa R² | ~0.97 | **0.961** | ≥ 0.96 |
| Mean R² (S,σ,κ) | 0.955 | **0.954** | ≥ 0.95 |
| ZT Pearson r | ~0.97 | **0.978** | ↑ |
| PF consistency R² | — | **0.749** | ↑ |
| ZT consistency R² | — | — | target ≥ 0.85 |
| Physics-aware | ✗ | Partial (PF) | Full (PF + ZT) |

---

## 13. What Comes Next — Stage 4

Stage 4 adds the **Wiedemann-Franz law** as a third physics constraint:

$$\kappa_e = L_0 \cdot \sigma \cdot T$$

where $L_0 = 2.44 \times 10^{-8}$ W·Ω·K⁻² is the Lorenz number. This connects electrical and thermal conductivity. It holds exactly for metals; for semiconductors it is approximate, so its weight λ_WF is kept small (≈ 0.05).

Stage 4 also includes a **lambda sweep** — automatically testing 4 configurations of (λ_PF, λ_ZT, λ_WF) and plotting R² vs. physics consistency scores, giving a quantitative view of the accuracy–consistency tradeoff.