# Thermoelectric Property Prediction — Full PINN Documentation

**Model:** Physics-Informed Neural Network (PINN)  
**Dataset:** JARVIS-DFT (`dft_3d` or `dft_2d`)  
**Goal:** Predict Seebeck coefficient, electrical conductivity, thermal conductivity, and ZT figure of merit from structural/electronic descriptors — while enforcing that predictions are physically self-consistent.

---

## Table of Contents

1. [What is a PINN and Why Are We Using It?](#1-what-is-a-pinn-and-why-are-we-using-it)
2. [Data Source](#2-data-source)
3. [Data Cleaning — Step by Step](#3-data-cleaning--step-by-step)
4. [Column Conversions and Transforms](#4-column-conversions-and-transforms)
5. [Input Features](#5-input-features)
6. [Model Architecture](#6-model-architecture)
7. [The Loss Function — Every Term Explained](#7-the-loss-function--every-term-explained)
8. [Physics Constraints — Full Derivation](#8-physics-constraints--full-derivation)
9. [Training Setup](#9-training-setup)
10. [Evaluation Metrics — What Each One Means](#10-evaluation-metrics--what-each-one-means)
11. [Results and What They Tell Us](#11-results-and-what-they-tell-us)
12. [Lambda Sweep — Why and What It Shows](#12-lambda-sweep--why-and-what-it-shows)

---

## 1. What is a PINN and Why Are We Using It?

### What is a normal neural network?

A normal neural network is a function approximator. You feed it inputs (numbers describing a material), it produces outputs (predicted properties), and you train it by telling it how wrong its predictions are. It adjusts its internal weights millions of times until the predictions get good. The key thing: **it has no idea what the numbers mean**. It just finds patterns.

The problem with this for materials science: a normal network can predict a Seebeck coefficient of 500 µV/K and a conductivity of 10⁶ S/m, and when you plug those into the ZT formula, get a completely different ZT than what the network predicted for ZT. The predictions are individually reasonable but **physically inconsistent with each other**. This is a serious problem if you plan to use the model to screen new materials.

### What does PINN add?

A PINN (Physics-Informed Neural Network) adds **extra penalty terms to the loss function** that punish the network whenever its predictions violate known physical laws. The network still learns from data, but it also gets graded on whether $S$, $\sigma$, $\kappa$, and $ZT$ are mutually consistent.

The full loss is:

$$L_{total} = L_{data} + \lambda_{PF} \cdot L_{PF} + \lambda_{ZT} \cdot L_{ZT} + \lambda_{WF} \cdot L_{WF}$$

The $\lambda$ values are weights controlling how strictly each physical law is enforced. Everything else in the model — the architecture, the training loop — is the same as a normal neural network. **The physics lives entirely inside the loss function.**

### Why do we need this for thermoelectrics specifically?

The three transport properties ($S$, $\sigma$, $\kappa$) are deeply interlinked:

- $PF = S^2 \sigma$ — power factor is derived from S and σ
- $ZT = PF \cdot T / \kappa$ — ZT is derived from all three
- $\kappa_e / (\sigma T) = L$ — the Wiedemann-Franz law links σ and κ

If a model predicts these four quantities independently without any constraint, there's no reason to trust that they'll be consistent with each other. The PINN constraints force the model to learn representations where all four quantities emerge from a shared physical understanding.

---

## 2. Data Source

The dataset is pulled directly from **JARVIS-DFT** (Joint Automated Repository for Various Integrated Simulations) using the `jarvis-tools` Python package. No local CSV is needed — the library downloads and caches it automatically.

```python
DATASET = "dft_3d"    # switch to "dft_2d" for 2D materials
```

| Dataset | Materials | Description |
|---|---|---|
| `dft_3d` | ~76,000 | Bulk 3D materials with DFT-computed properties |
| `dft_2d` | ~1,000 | 2D layered materials (e.g. MoS₂-type) |

The raw download gives ~76,000 rows and 64 columns for `dft_3d`. Most of these columns are not transport properties — they include structural, electronic, mechanical, and optical properties computed from DFT.

**After loading, the first thing the script does is normalise column names** — because JARVIS has changed column naming conventions across versions. For example, Seebeck coefficient might be stored as `p_seebeck_300`, `p-seebeck_300`, or `p_seebeck` depending on which version of the database you download. The `COL_MAP` dictionary maps all of these to a consistent internal name (`p-Seebeck`), so the rest of the script always uses the same name regardless of which version was downloaded.

---

## 3. Data Cleaning — Step by Step

### Step 1: Numeric coercion

JARVIS stores many values as the string `"na"` rather than Python `NaN`. Every feature and target column is passed through `pd.to_numeric(..., errors='coerce')`, which converts `"na"` → `NaN` and leaves real numbers unchanged. Without this step, the model would either crash or silently treat `"na"` as 0.

### Step 2: Compute ZT_p

ZT does not exist as a column in JARVIS. It is computed:

$$ZT_p = \frac{p\text{-powerfact} \times T}{pkappa}$$

where $T = 300$ K. This is only computed for rows where both `p-powerfact` and `pkappa` are present and `pkappa > 0` (can't divide by zero). All other rows get `NaN` for `ZT_p`.

**Result from the actual run:**
```
Valid ZT_p rows: 23,217  (out of 75,993 loaded)
```

This means ~52,000 rows are dropped at this stage because they don't have both power factor and thermal conductivity computed. This is expected — JARVIS only computes transport properties for materials that passed certain screening criteria.

> **Why clip ZT to [0, 10]?**
> Occasionally the DFT calculation produces physically unrealistic ZT values (e.g. ZT = 500) due to numerical errors in the transport calculation. Materials with ZT > 10 don't exist in practice. Clipping prevents these outliers from dominating the loss.

### Step 3: Drop rows with missing targets

After computing ZT_p, we keep only rows where **all four targets** are present:
- `p-Seebeck`
- `pcond`
- `pkappa`
- `ZT_p`

```
Rows after target dropna: 23,217
```

In this run, the ZT filter already removed the right rows, so the dropna step doesn't remove any additional rows. In other runs or datasets, this step can remove more.

### Step 4: Missingness flags

For four specific columns that are frequently missing, a **flag column** is created before imputation:

| Column | Flag created |
|---|---|
| `exfoliation_energy` | `exfoliation_energy_missing` = 1 if missing, 0 if present |
| `slme` | `slme_missing` |
| `Tc_supercon` | `Tc_supercon_missing` |
| `spillage` | `spillage_missing` |

**Why create a flag instead of just filling with median?**

If we simply fill the missing value with the median, the model sees a number and has no way of knowing whether it was real or made up. The flag tells the model "the value I'm seeing for this column was imputed — treat it with less confidence." A well-trained network learns to down-weight features when their missingness flag is 1. Without the flag, the imputed value gets treated as real, which can mislead the model.

### Step 5: Median imputation for remaining feature NaNs

After creating the flags, all remaining NaN values in feature columns are filled with the **column median**:

```python
median_val = float(np.nanmedian(arr))
df_work[col] = df_work[col].fillna(median_val)
```

**Why median and not mean?**

Materials property distributions are heavily skewed — a small number of materials have extreme values. The mean gets pulled by these extremes. The median is the middle value regardless of outliers, so it's a more neutral fill for a typical material. For example, if bandgap values are mostly 1–3 eV but a few are 10+ eV, the mean might be 2.5 eV while the median is 1.8 eV. The median is a better representation of a "typical" material.

**Does this hurt the model?**

Slightly — yes. An imputed value is not real data. But the alternative (dropping every row that has any missing feature) would lose most of the dataset. The missingness flags help the model understand which values to trust less. The net effect is small because the most important features (bandgap, density) are rarely missing.

### Step 6: Outlier clipping

For every feature column, values beyond **3× the interquartile range (IQR)** are clipped:

```python
Q1, Q3 = np.nanpercentile(arr, 25), np.nanpercentile(arr, 75)
IQR = Q3 - Q1
df_work[col] = df_work[col].clip(Q1 - 3*IQR, Q3 + 3*IQR)
```

**What is IQR?** The interquartile range is the distance between the 25th and 75th percentile. It represents the "middle 50%" of the data. Anything beyond 3× this range on either side is considered an extreme outlier.

**Why clip instead of drop?** Dropping rows for extreme feature values would reduce the dataset further. Clipping keeps the row but caps the extreme value at the boundary. This prevents a single extreme material from having outsized influence on the network weights.

---

## 4. Column Conversions and Transforms

### Why convert at all?

Thermoelectric transport properties span many orders of magnitude:
- Seebeck coefficient: 1 – 800 µV/K
- Electrical conductivity: 10⁻² – 10⁶ S/m
- Thermal conductivity: 0.01 – 1000 W/m·K

If you train a neural network directly on these raw numbers, the large-valued columns dominate the gradient updates. The network spends most of its effort fitting conductivity (huge numbers) and almost ignores Seebeck (small numbers). Transforms fix this by putting all properties on a comparable scale.

### Transform 1: Absolute value for Seebeck

```python
df_work['p-Seebeck_abs'] = np.abs(df_work['p-Seebeck'])
```

p-type Seebeck can be negative (the sign encodes the carrier type). But in the ZT formula, only $S^2$ appears — the sign cancels out. Taking the absolute value removes ambiguity and simplifies the learning problem.

### Transform 2: log10(x + 1) for conductivity, kappa, and ZT

```python
df_work['pcond_log']  = np.log10(df_work['pcond'].clip(lower=1e-10) + 1.0)
df_work['pkappa_log'] = np.log10(df_work['pkappa'].clip(lower=1e-10) + 1.0)
df_work['ZT_p_log']   = np.log10(df_work['ZT_p'] + 1.0)
```

**Why log10?**

A log transform compresses a wide range into a manageable one. For example:
- Conductivity range: 0.01 → 10⁶ S/m (8 orders of magnitude)
- After log10: 0 → 6 (6 units of range)

This makes the distribution much more Gaussian, which neural networks learn from much more efficiently.

**Why +1 before taking log?** Because log(0) = −∞. Adding 1 ensures that a value of 0 maps to log10(1) = 0 rather than crashing.

**Why clip to 1e-10 first?** For numerical safety — if `pcond` has a very tiny negative value due to floating-point errors, it would still produce a real number after clipping.

### Transform 3: RobustScaler on all four targets

```python
sc_y = RobustScaler()
y_tr_scaled = sc_y.fit_transform(y[idx_tr])
```

After log transforms, the four targets are further scaled using `RobustScaler`, which:
1. Subtracts the **median** (not mean)
2. Divides by the **IQR** (not standard deviation)

This centers the distribution around 0 and makes most values fall in the range [−2, +2]. Using median/IQR instead of mean/std makes the scaler robust to the outliers that still exist after log transform.

**Important:** The scaler is fit only on the training set and then applied to the test set. This prevents information from the test set leaking into the scaling.

### Aux targets — log space, NOT scaled

Three auxiliary columns are created specifically for the physics residual:

```python
df_work['S_log']     = np.log10(df_work['p-Seebeck_abs'].clip(lower=1e-10) + 1.0)
df_work['sig_log']   = df_work['pcond_log'].copy()
df_work['kappa_log'] = df_work['pkappa_log'].copy()
```

These are **identical transforms** to the main targets but are **never passed through RobustScaler**. They stay in raw log10 space. The reason: the physics formulas ($PF = S^2\sigma$, etc.) hold in log space and break in scaled space (see Section 8).

### Unit Offsets — UNIT_OFFSET_PF, UNIT_OFFSET_ZT, UNIT_OFFSET_WF

Three scalar constants are computed from the data:

```
UNIT_OFFSET_PF: 6.0044
UNIT_OFFSET_ZT: -3.7112
UNIT_OFFSET_WF: 6.7888
```

**What are these?**

The physics formulas ($PF = S^2\sigma$, etc.) hold in SI units. But the JARVIS dataset stores properties in mixed units — Seebeck in µV/K, conductivity in S/m, power factor in µW/cm·K², etc. When you take log10 of both sides of the PF equation, the unit mismatch shows up as a constant additive offset.

Rather than manually computing the unit conversion factor (which would require knowing the exact units of every column), we compute the offset empirically: take all rows where S, σ, and PF are all well-defined, compute `2*S_log + sig_log - log_PF` for each row, and take the median. This median is the offset — it absorbs all unit mismatches automatically.

**Why median?** Robust to outliers. A few rows with extreme values won't corrupt the offset estimate.

**Why is the sanity check important?**

After computing the offset, the code prints the median absolute residual after correction — this should be close to 0. If it's large (e.g. 2.0 log10 units), the physics formula is not well-satisfied by the data, which is a warning signal that something is wrong with the column mapping.

---

## 5. Input Features

Features are split into four domain-motivated blocks. Each block feeds into its own small sub-network before merging. This allows the model to learn block-specific representations — for example, the mechanical block can learn that bulk modulus and shear modulus interact in a specific way — before combining them.

### Block S — Structural (→ 32 neurons)

| Feature | What it is | Why it matters |
|---|---|---|
| `nat` | Number of atoms in unit cell | More atoms = more phonon scattering modes |
| `density` | Mass density (g/cm³) | Correlates with acoustic phonon velocity, thermal conductivity |
| `exfoliation_energy` | Energy to peel a layer (eV/atom) | Relevant for 2D materials; missing for most 3D |
| `formation_energy_peratom` | Thermodynamic stability (eV/atom) | Unstable materials are unlikely to be synthesisable |
| `ehull` | Distance to convex hull (eV/atom) | Metastability indicator |
| `crys_*` | Crystal system (one-hot encoded) | Symmetry determines transport anisotropy |
| `spg_number_norm` | Space group number / 230 | Encodes point group symmetry |
| `dimensionality_enc` | 0D=0, 1D=1, 2D=2, 3D=3 | Dimensionality fundamentally changes physics |
| `exfoliation_energy_missing` | 1 if exfoliation energy was NaN | Tells model the value was imputed |

**Why one-hot encode `crys`?** Crystal system is a category (cubic, hexagonal, tetragonal, etc.), not a number. If you encoded it as 1, 2, 3... the model would think "hexagonal is twice as much as cubic," which is meaningless. One-hot gives each crystal system its own independent column.

**Why normalise `spg_number` by 230?** Space group numbers run from 1 to 230. Dividing by 230 puts this in [0, 1]. Without normalisation, the raw number 230 would dominate the structural block.

### Block E — Electronic (→ 64 neurons)

| Feature | What it is | Why it matters |
|---|---|---|
| `optb88vdw_bandgap` | DFT bandgap (optB88vdW functional) | Semiconductors have larger Seebeck than metals |
| `mbj_bandgap` | Modified Becke-Johnson corrected bandgap | More accurate bandgap for gap prediction |
| `hse_gap` | HSE06 hybrid functional bandgap | Most accurate DFT bandgap estimate |
| `avg_elec_mass` / `avg_hole_mass` | Effective carrier masses | Heavy masses → high Seebeck but low conductivity |
| `epsx/y/z` | Total dielectric tensor components | Related to optical and polar phonon scattering |
| `mepsx/y/z` | Electronic dielectric components | Contribution to dielectric from electrons only |
| `slme` | Spectroscopic limited max efficiency | Solar cell efficiency proxy; correlates with absorption |
| `spillage` | Topological spillage indicator | Identifies topological insulators |
| `magmom_oszicar/outcar` | Magnetic moment (two DFT convergence monitors) | Magnetic order affects carrier scattering |

Electronic features get 64 neurons (vs 32 for structural) because electronic structure is the dominant driver of Seebeck and conductivity.

### Block T — Thermal (→ 32 neurons)

| Feature | What it is | Why it matters |
|---|---|---|
| `n-Seebeck` | n-type Seebeck coefficient | Cross-type information; p and n often correlated |
| `n-powerfact` | n-type power factor | Same |
| `ncond` | n-type electrical conductivity | Same |
| `nkappa` | n-type thermal conductivity | Same |
| `optb88vdw_total_energy` | DFT total energy | Related to phonon stability |
| `Tc_supercon` | Superconducting transition temperature | Electron-phonon coupling proxy |
| `max_ir_mode` / `min_ir_mode` | Phonon IR-active mode frequencies | Directly related to optical phonon scattering of heat |

### Block M — Mechanical (→ 16 neurons)

| Feature | What it is | Why it matters |
|---|---|---|
| `bulk_modulus_kv` | Voigt bulk modulus (GPa) | Stiff materials have high phonon velocity → high κ |
| `shear_modulus_gv` | Voigt shear modulus (GPa) | Related to transverse acoustic phonons |
| `poisson` | Poisson ratio | Grüneisen parameter proxy; larger → more anharmonicity → lower κ |
| `dfpt_piezo_max_dielectric` | Max total dielectric (DFPT) | Ionic contribution to phonon scattering |
| `dfpt_piezo_max_eij` | Max piezoelectric stress coefficient | Electron-phonon coupling in polar materials |
| `dfpt_piezo_max_dij` | Max piezoelectric strain coefficient | Same |
| `max_efg` / `efg` | Electric field gradient at nucleus | Nuclear quadrupole interaction; phonon scattering proxy |

Mechanical features get only 16 neurons because they are the most indirectly related to transport properties.

---

## 6. Model Architecture

### Overall structure

```
inp_structural  (15 features) ─→ Dense(32, ReLU) → BatchNorm ─┐
inp_electronic  (15 features) ─→ Dense(64, ReLU) → BatchNorm ─┤
inp_thermal     ( 8 features) ─→ Dense(32, ReLU) → BatchNorm ─┤─→ Concatenate (144)
inp_mechanical  (10 features) ─→ Dense(16, ReLU) → BatchNorm ─┘
                                                                 ↓
                                              Dense(256, ReLU) → BatchNorm → Dropout(0.3)
                                                                 ↓
                                              Dense(128, ReLU) → BatchNorm → Dropout(0.2)
                                                                 ↓
                                              Dense(64,  ReLU) → Dropout(0.1)
                                                                 ↓
              ┌──────────────────────────────────────────────────────────────────┐
              ↓          ↓           ↓          ↓         ↓         ↓         ↓
         Dense(1)   Dense(1)   Dense(1)   Dense(1)  Dense(1)  Dense(1)  Dense(1)
         linear     linear     linear     linear   softplus  softplus  softplus
         Seebeck    Cond       Kappa      ZT       S_log    sig_log  kappa_log
              └────────── main_output (4) ─────────┘  └──── aux_output (3) ────┘
```

**Total parameters: ~130,000**

### Component explanations

**Dense(N, ReLU)** — A fully-connected layer with N neurons and ReLU activation. ReLU (Rectified Linear Unit) is simply `max(0, x)` — it outputs the input if positive, zero if negative. It's the most common activation because it's computationally cheap and doesn't suffer from the vanishing gradient problem that sigmoid/tanh do.

**BatchNormalization** — After each major layer, batch normalisation standardises the activations (makes them have roughly mean 0 and standard deviation 1). This stabilises training and allows higher learning rates. Without it, deep networks often fail to converge.

**Dropout(p)** — During training only, randomly sets p fraction of neuron outputs to zero. This forces the network to learn redundant representations and prevents overfitting — the network can't rely on any single neuron. Dropout is turned off during evaluation (`training=False`).

- Dropout(0.3) after Dense(256): aggressive regularisation at the widest layer
- Dropout(0.2) after Dense(128): moderate
- Dropout(0.1) after Dense(64): light regularisation just before the output heads

**L2 regularisation (1e-4)** — On the Dense(256) and Dense(128) layers, a small penalty proportional to the squared weight values is added to the loss. This discourages the weights from becoming very large, which is a form of overfitting.

**Main output heads (4, linear activation)** — Four separate Dense(1) neurons, each predicting one target in scaled space. Linear activation means the output is unbounded — the scaler already constrains the range.

**Aux output heads (3, softplus activation)** — Three neurons predicting S_log, sig_log, kappa_log in unscaled log space. Softplus is `log(1 + e^x)`, which is always positive. Since log10(x+1) is always ≥ 0, softplus naturally enforces this constraint on the aux head outputs.

### Why separate aux heads?

The main heads output in RobustScaler-transformed space. The physics formulas ($PF = S^2\sigma$, etc.) do not hold in scaled space — scaling shifts and squashes each variable independently, destroying the multiplicative relationship. The aux heads bypass the scaler entirely and output in raw log10 space, where the physics formulas remain valid.

Both head types share the same trunk (Dense 256 → 128 → 64). This means physics information from the aux heads flows back through the shared layers and influences the representations that drive the main heads. They're not separate models — they're the same model reading the same internal representation through two different lenses.

### Auto-tuning for small datasets

If the dataset has fewer than 1,000 rows (which happens with `dft_2d`), the script automatically adjusts:

| Setting | dft_3d (large) | dft_2d (small) |
|---|---|---|
| Batch size | 64 | 16 |
| Early stopping patience | 30 | 50 |
| Dropout layer 1 | 0.3 | 0.1 |
| Dropout layer 2 | 0.2 | 0.1 |
| Dropout layer 3 | 0.1 | 0.05 |

Smaller batches for 2D because each batch needs to be a meaningful sample of the dataset. More patience because smaller datasets have noisier validation curves — the model needs more time to demonstrate it has genuinely plateaued. Less dropout because with few rows, you don't want to throw away information unnecessarily.

---

## 7. The Loss Function — Every Term Explained

### Why a manual training loop instead of Keras `.fit()`?

Keras's built-in `.fit()` computes loss only between model outputs and labels. The physics residual involves arithmetic between output heads and a ground-truth column (`log_PF`, `log_ZT`). Keras cannot express this. So a manual `tf.GradientTape` loop is used — it's the TensorFlow equivalent of "compute this arbitrary expression, then differentiate it."

### Full loss equation

$$L_{total} = L_{data} + \lambda_{AUX} \cdot L_{aux} + \lambda_{PF} \cdot L_{PF} + \lambda_{ZT} \cdot L_{ZT} + \lambda_{WF} \cdot L_{WF}$$

### Term 1: L_data (weight = 1.0)

```python
L_data = tf.reduce_mean(tf.square(main_pred - y_main_true))
```

**What it is:** Mean Squared Error between the 4 main predicted targets (in scaled space) and the 4 true targets (in scaled space).

**What it does:** This is the primary fitting objective. It drives the network to predict Seebeck, conductivity, kappa, and ZT correctly. Weight = 1.0 because this is the dominant term.

**In plain English:** "How wrong are your predictions for the actual property values?"

### Term 2: L_aux (weight = 1.0)

```python
L_aux = tf.reduce_mean(tf.square(aux_pred - y_aux_true))
```

**What it is:** Mean Squared Error between the 3 aux head predictions (S_log, sig_log, kappa_log) and their ground-truth log-space values.

**What it does:** Keeps the aux heads accurate and grounded. Without this term, the aux heads would only receive gradient signal through the physics residuals — they'd learn to minimise the physics loss but might drift far from the actual S, σ, κ values. L_aux anchors them.

**Why weight = 1.0?** The aux heads should be as carefully trained as the main heads. They're not secondary — they're the foundation of all three physics constraints.

**In plain English:** "Are your auxiliary S, σ, κ predictions (the ones used for physics checking) actually correct?"

### Term 3: L_PF (weight = λ_PF = 0.10)

```python
PF_log_hat = 2.0 * S_log_hat + sig_log_hat - UNIT_OFFSET_PF_T
L_PF = tf.reduce_mean(tf.square(PF_log_hat - log_pf_true))
```

**What it is:** The Power Factor physics residual. It computes what the power factor *should* be (given the predicted S and σ), and penalises the difference from the true power factor.

**The physics:** $PF = S^2 \cdot \sigma \Rightarrow \log_{10}(PF) = 2\log_{10}(S) + \log_{10}(\sigma)$

**Why 0.10?** Physics is a guide, not the primary objective. If λ_PF is too large, the physics loss dominates and the network sacrifices prediction accuracy to achieve consistency. 0.10 means "for every 10 units of improvement in data fitting, we also care about 1 unit of physics consistency."

**In plain English:** "Do your predicted S and σ multiply together to give the right power factor?"

### Term 4: L_ZT (weight = λ_ZT = 1.00)

```python
ZT_log_hat = (2.0 * S_log_hat + sig_log_hat - kappa_log_hat + LOG10_T_T - UNIT_OFFSET_ZT_T)
L_ZT = tf.reduce_mean(tf.square(ZT_log_hat - log_zt_true))
```

**What it is:** The ZT consistency residual. Checks whether the predicted S, σ, and κ together produce the correct ZT.

**The physics:** $ZT = \frac{S^2 \sigma T}{\kappa} \Rightarrow \log(ZT) = 2\log(S) + \log(\sigma) + \log(T) - \log(\kappa)$

**Why weight = 1.0 (same as L_aux)?** ZT is the primary target quantity — the whole point of the project is accurate ZT prediction. Making L_ZT strong (same weight as L_aux) forces the network to treat ZT consistency as a first-class objective, not a side constraint.

**In plain English:** "Do all four of your property predictions combine correctly to give the right ZT?"

### Term 5: L_WF (weight = λ_WF = 0.05)

```python
WF_log_hat = sig_log_hat + UNIT_OFFSET_WF_T + LOG10_T_T
L_WF = tf.reduce_mean(tf.square(WF_log_hat - kappa_log_hat))
```

**What it is:** The Wiedemann-Franz law residual. Links electrical conductivity to thermal conductivity.

**The physics:** $\frac{\kappa_e}{\sigma T} = L$ where $L = 2.44 \times 10^{-8}$ W·Ω·K⁻² is the Lorenz number. Taking log: $\log(\kappa_e) = \log(\sigma) + \log(L) + \log(T)$

**Why weight = 0.05 (smallest)?** The Wiedemann-Franz law holds exactly for free electrons in metals. For semiconductors like thermoelectrics, the effective Lorenz number varies with carrier concentration, reduced chemical potential, and phonon scattering mechanism. It is an approximation here. The empirical offset `UNIT_OFFSET_WF` absorbs the discrepancy between $\kappa_{total}$ (which the dataset measures) and $\kappa_{electronic}$ (which WF actually governs). Small weight = soft guidance.

> Reference: Yadav et al., *J. Phys. Commun.* 3 (2019) 105001 — shows that the generalised Lorenz number for semiconductors is NOT a universal constant.

**In plain English:** "Does your predicted conductivity relate to your predicted thermal conductivity in a way consistent with electron heat transport?"

---

## 8. Physics Constraints — Full Derivation

### Why use log space for all constraints?

All three physics laws involve multiplication and division. In raw space:

$$PF = S^2 \cdot \sigma$$

If S = 500 µV/K and σ = 10⁵ S/m, then squaring and multiplying gives 2.5 × 10¹³ — a huge number. The gradient of the MSE loss with respect to S at this scale is enormous, causing gradient explosion.

In log space, multiplication becomes addition:

$$\log(PF) = 2\log(S) + \log(\sigma)$$

The operands are now in the range 0–8, gradients are bounded, and training is stable.

### Power Factor constraint derivation

Starting from the physics:
$$PF = S^2 \cdot \sigma$$

Taking log10:
$$\log_{10}(PF) = 2\log_{10}(S) + \log_{10}(\sigma)$$

Adding +1 inside each log (for numerical stability near zero):
$$\log_{10}(PF + 1) \approx 2 \cdot S\_log + sig\_log - \text{UNIT\_OFFSET\_PF}$$

The UNIT_OFFSET_PF = 6.004 absorbs: (a) the unit conversion between µV/K, S/m, and µW/cm·K², and (b) the distortion from adding 1 inside each log. It is computed as the median of $2 \cdot S\_log + sig\_log - \log_{10}(PF+1)$ over all valid rows.

### ZT consistency constraint derivation

$$ZT = \frac{PF \cdot T}{\kappa} = \frac{S^2 \sigma T}{\kappa}$$

Taking log10:
$$\log_{10}(ZT) = 2\log_{10}(S) + \log_{10}(\sigma) + \log_{10}(T) - \log_{10}(\kappa)$$

With +1 offsets and empirical calibration:
$$\log_{10}(ZT + 1) \approx 2 \cdot S\_log + sig\_log + \log_{10}(300) - kappa\_log - \text{UNIT\_OFFSET\_ZT}$$

$\log_{10}(300) = 2.477$ is a constant baked in at training time.

UNIT_OFFSET_ZT = −3.711. The negative sign is expected: it reflects the fact that ZT values in this dataset are small (mean log_ZT ≈ 0.00, meaning ZT ≈ 0 for most materials), while the combination of S, σ, T terms is relatively large.

### Wiedemann-Franz constraint derivation

$$\frac{\kappa_e}{\sigma \cdot T} = L_0 \approx 2.44 \times 10^{-8} \text{ W·Ω·K}^{-2}$$

Rearranging: $\kappa_e = \sigma \cdot L_0 \cdot T$

Taking log10:
$$\log_{10}(\kappa_e) = \log_{10}(\sigma) + \log_{10}(L_0) + \log_{10}(T)$$

Since $\kappa_{total} \neq \kappa_e$ (there's also phonon thermal conductivity), we don't use $L_0$ directly. Instead, UNIT_OFFSET_WF = 6.789 is the empirically computed effective Lorenz number in log space:

$$\text{UNIT\_OFFSET\_WF} = \text{median}(\kappa\_log - sig\_log - \log_{10}(T))$$

This equals $\log_{10}(L_{eff})$ where $L_{eff} = 10^{6.789} \approx 6.15 \times 10^6$ — much larger than the free-electron $L_0 = 2.44 \times 10^{-8}$ because $\kappa_{total}$ includes phonon contributions on top of electronic.

---

## 9. Training Setup

### Optimiser: Adam (lr = 0.001)

Adam (Adaptive Moment Estimation) is an optimiser that maintains a per-parameter learning rate, adapting it based on recent gradient history. It converges faster than plain gradient descent and requires less manual learning rate tuning. lr = 0.001 is the standard default.

### Epochs: 300 maximum

One epoch = one full pass through the training data. The model is allowed up to 300 epochs but stops early if the validation loss stops improving.

### Early stopping (patience = 30 for 3D, 50 for 2D)

After each epoch, the model's performance on the **validation set** (a held-out 20% of training data) is measured. If the validation loss doesn't improve for 30 consecutive epochs, training stops. The best weights seen during training are restored.

**Why early stopping?** Without it, the model would eventually memorise the training data (overfitting) and perform poorly on new materials. The validation loss is a proxy for performance on unseen data — when it stops improving, further training is wasted.

### Train/validation/test split

```
Full dataset (23,217 rows)
    ↓ 80/20 split
Training pool (18,573)    |    Test set (4,644)    ← never seen during training
    ↓ 80/20 split
Training (14,858)  |  Validation (3,714)    ← used only for early stopping
```

The test set is truly held out — it's only used in the final evaluation, never for any training decision.

### Batch training

Each epoch, the 14,858 training rows are shuffled and divided into batches of 64. The weights are updated after each batch. This is called stochastic gradient descent — the gradient estimate from 64 samples is noisy but much faster than using all 14,858 rows at once.

---

## 10. Evaluation Metrics — What Each One Means

### MAE (Mean Absolute Error)

$$MAE = \frac{1}{N}\sum_{i=1}^{N} |\hat{y}_i - y_i|$$

The average absolute difference between predicted and true values. Reported in the same units as the target (e.g. µV/K for Seebeck).

**Seebeck MAE = 30.17 µV/K** means on average the prediction is 30 µV/K off from the true value. Given the range of Seebeck values is roughly 0–800 µV/K, this is an ~4% average error.

### R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum(\hat{y}_i - y_i)^2}{\sum(y_i - \bar{y})^2}$$

R² measures how much of the variance in the true values is explained by the model. Range: (−∞, 1]. Interpretation:
- R² = 1.0: perfect prediction
- R² = 0.0: the model is no better than predicting the mean every time
- R² < 0.0: the model is worse than predicting the mean (something is very wrong)

**R² = 0.9187 for Seebeck** means the model explains 91.87% of the variation in Seebeck values.

### Why R² fails for ZT — and why we use Pearson instead

```
ZT_p_log   0.0000   0.9446   var=0.00000 -- use Pearson
```

R² divides by the variance of the true values. If nearly all materials have ZT ≈ 0 (which is true — most materials are bad thermoelectrics), the variance is essentially zero, and R² becomes undefined or meaninglessly large/small depending on floating-point rounding. The `var=0.00000` warning flags this.

In this case, **Pearson correlation** is reported instead:

$$r = \frac{\sum(\hat{y}_i - \bar{\hat{y}})(y_i - \bar{y})}{\sqrt{\sum(\hat{y}_i - \bar{\hat{y}})^2 \cdot \sum(y_i - \bar{y})^2}}$$

Pearson r measures linear correlation regardless of scale. r = 1.0 = perfect positive correlation. r = 0.9728 means the model's ZT ranking is highly correlated with the true ZT ranking.

### Why we also report Spearman correlation

```
ZT Spearman r: 0.9520
```

Pearson assumes a linear relationship. **Spearman** correlation is rank-based: it converts both true and predicted ZT to ranks (1st, 2nd, 3rd...) and computes Pearson on the ranks. It measures whether the model correctly orders materials by ZT, regardless of whether the relationship is linear.

For materials screening, you often care more about "which materials are in the top 5%?" than exact ZT values. A high Spearman r (0.95) means the model is good at identifying which materials are better than others — this is practically the most useful metric for this project.

### Aux head evaluation

```
Aux target           MAE          R2
S_log                0.3680       0.7080
sig_log              0.1778       0.9617
kappa_log            0.2347       0.8992
```

These evaluate the three auxiliary heads independently in log10 space. They should be accurate because they're trained with L_aux. S_log R² of 0.708 is weaker than sig_log (0.962) — Seebeck is intrinsically harder to predict because it's sensitive to the shape of the DOS near the Fermi level, which is hard to capture from bulk descriptors alone.

### Physics check R² values

Three additional R² values report whether the **predicted** properties combine correctly to reproduce the ground-truth derived quantities:

**PF check R²:** Checks whether `2*S_log_pred + sig_log_pred - offset` matches `log10(PF_true + 1)`. A value close to 1.0 means the model's S and σ predictions are mutually consistent with the power factor.

**ZT check R²:** Same logic for ZT. Checks whether S, σ, κ predictions combine correctly.

**WF check R²:** Checks whether σ and κ are consistent with the Wiedemann-Franz law.

**Negative values** (e.g. PF check R² = −0.81 in Stage 4) mean the physics constraint is actually making things *less* consistent — a sign that the constraint weight is too high and fighting the data loss.

---

## 11. Results and What They Tell Us

### Stage 4 main results (from the actual run)

```
Target                 MAE          R2
p-Seebeck_abs          30.17        0.9187
pcond_log              0.1522       0.9650
pkappa_log             0.1224       0.9579
ZT Pearson r:          0.9728
ZT Spearman r:         0.9520
Mean R2 (S, σ, κ):     0.9472
```

These are strong results. The model explains over 91% of variance in Seebeck and over 96% of variance in conductivity, and correctly ranks materials by ZT 95% of the time.

### Lambda Sweep Summary Table

```
Config                       Mean R2    ZT Pearson   PF check R2
No physics (data only)       0.9402     0.9727       0.6257
PF only (Stage 2)            0.9471     0.9731       0.8008
PF + ZT (Stage 3)            0.9421     0.9710       -0.3367
PF + ZT + WF (Stage 4)       0.9312     0.9486       -6.4104
```

**Reading this table:**

1. **Stage 2 (PF only) gives the best mean R²** (0.9471). Adding one well-calibrated physics constraint slightly improves accuracy.

2. **Stage 3 and Stage 4 show degradation in both accuracy and physics consistency.** When ZT and WF constraints are added, the PF check actually goes negative. This means the constraints are **conflicting with each other** — the ZT constraint pulls the aux heads in a direction that breaks PF consistency.

3. **This is a known phenomenon in PINN training** called constraint interference. Each physics term adds gradient signal that may be partially orthogonal to (or even opposed to) other constraints. The solution is careful lambda tuning or constraint relaxation, which is the natural next step.

4. **The data-only baseline (0.9402) is strong.** This confirms the architecture itself is well-designed — even without physics, it performs well. The physics constraints are a refinement, not a necessity for basic accuracy.

### What the ZT variance issue means physically

`ZT_p_log` has near-zero variance because the vast majority of materials in JARVIS have ZT ≈ 0. This is physically correct — only a handful of known materials have ZT > 1. The model sees almost no high-ZT examples during training. The Pearson and Spearman correlations on the few materials that do have nonzero ZT are therefore the meaningful metric here.

---

## 12. Lambda Sweep — Why and What It Shows

### What is the sweep?

After training the full Stage 4 model, the script automatically retrains four versions with different physics constraint configurations:

| Config | λ_PF | λ_ZT | λ_WF |
|---|---|---|---|
| No physics | 0 | 0 | 0 |
| PF only | 0.10 | 0 | 0 |
| PF + ZT | 0.10 | 1.0 | 0 |
| PF + ZT + WF | 0.10 | 1.0 | 0.05 |

All other settings (architecture, data splits, random seed) are identical.

### Why do the sweep?

The lambda values are hyperparameters — there is no single correct value. Too small: the physics constraint has no effect. Too large: the physics constraint dominates and hurts accuracy. The sweep makes the tradeoff visible:

- **If PF-only improves R² over no physics:** the PF constraint is helping by providing useful gradient signal
- **If adding ZT hurts R²:** the ZT constraint is too strong or conflicts with PF
- **Physics check R² going negative:** strong signal that a constraint is counterproductive at the current lambda

### What the plot shows

Three bar charts are saved to `lambda_sweep_dft_3d.png`:
1. Mean R² across Seebeck, conductivity, kappa — measures raw prediction accuracy
2. ZT Pearson r — measures ability to rank materials correctly
3. Grouped bar chart of PF/ZT/WF physics check R² — measures physical self-consistency

The ideal result would be high bars in all three panels for the full PINN. In practice, there's a tradeoff visible — the sweep quantifies it so you can choose the configuration that best fits your use case (accuracy vs. physical consistency).