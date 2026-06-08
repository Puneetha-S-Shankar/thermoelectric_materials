"""
STAGE 2: PINN -- Power Factor Physics Residual (Correct, Unit-Aware)
=====================================================================

The physics identity  PF = S^2 * sigma  holds in SI-consistent units.
But the dataset columns use mixed units (uV/K, S/m, uW/cm*K^2), so:

    log10(PF + 1) = 2*log10(|S|+1) + log10(sigma+1) - UNIT_OFFSET

where UNIT_OFFSET (~6) is the log10 of the unit-conversion factor,
computed EMPIRICALLY from the data so no hardcoding is needed.

KEY DESIGN (v3 -- unit-aware):
  Main heads (4): p-Seebeck_abs, pcond_log, pkappa_log, ZT_p_log  [scaled space]
  Aux  heads (2): S_log, sig_log                                    [unscaled log space]

  Physics residual (L_PF) uses ONLY the aux heads:
    PF_log_hat = 2*S_log_hat + sig_log_hat - UNIT_OFFSET
    L_PF = MSE(PF_log_hat, log10(PF_true + 1))

  Because UNIT_OFFSET is subtracted, the residual starts near 0 at
  initialization -- gradients are small, clean, and well-conditioned.

  Total loss:
    L_total = L_data + LAMBDA_AUX * L_aux + LAMBDA_PF * L_PF
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 2: PINN -- Power Factor Physics Loss (Correct Log-Space Aux Heads)")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD + ENCODE
# ============================================================================
df = pd.read_csv(
    'C:\\Users\\Puneetha\\thermoelectric_dataset\\output\\thermoelectric_ml_ready.csv',
    low_memory=False
)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

crys_dummies = pd.get_dummies(df['crys'], prefix='crys').astype(float)
df['spg_number_norm']  = pd.to_numeric(df['spg_number'], errors='coerce').fillna(0) / 230.0
dim_map = {'3D': 3, '2D': 2, '1D': 1, '0D': 0}
df['dimensionality_enc'] = df['dimensionality'].map(dim_map).fillna(2)

T_val = 300.0
df_zt = df.copy()
for col in ['p-powerfact', 'pkappa', 'p-Seebeck', 'pcond']:
    df_zt[col] = pd.to_numeric(df_zt[col], errors='coerce')

mask = (
    df_zt['p-powerfact'].notna() &
    df_zt['pkappa'].notna() &
    (df_zt['pkappa'] > 0)
)
df_zt['ZT_p'] = np.nan
df_zt.loc[mask, 'ZT_p'] = (
    df_zt.loc[mask, 'p-powerfact'] * T_val / df_zt.loc[mask, 'pkappa']
)
df_zt['ZT_p'] = df_zt['ZT_p'].clip(0, 10)
print(f"Valid ZT_p rows: {mask.sum():,}")

# ============================================================================
# STEP 2: FEATURE BLOCKS
# ============================================================================
structural_base = ['nat', 'density', 'exfoliation_energy',
                   'formation_energy_peratom', 'ehull']
electronic_cols = ['optb88vdw_bandgap', 'mbj_bandgap', 'hse_gap',
                   'avg_elec_mass', 'avg_hole_mass',
                   'epsx', 'epsy', 'epsz', 'mepsx', 'mepsy', 'mepsz',
                   'slme', 'spillage', 'magmom_oszicar', 'magmom_outcar']
thermal_cols    = ['n-Seebeck', 'n-powerfact', 'ncond', 'nkappa',
                   'optb88vdw_total_energy', 'Tc_supercon',
                   'max_ir_mode', 'min_ir_mode']
mechanical_cols = ['bulk_modulus_kv', 'shear_modulus_gv', 'poisson',
                   'dfpt_piezo_max_dielectric', 'dfpt_piezo_max_dielectric_electronic',
                   'dfpt_piezo_max_dielectric_ionic', 'dfpt_piezo_max_eij',
                   'dfpt_piezo_max_dij', 'max_efg', 'efg']

target_cols = ['p-Seebeck', 'pcond', 'pkappa', 'ZT_p']
all_numeric = structural_base + electronic_cols + thermal_cols + mechanical_cols

# ============================================================================
# STEP 3: BUILD df_work
# ============================================================================
for col in all_numeric + target_cols + ['p-powerfact']:
    if col in df_zt.columns:
        df_zt[col] = pd.to_numeric(df_zt[col], errors='coerce')

numeric_exist = [c for c in all_numeric if c in df_zt.columns]

df_work = pd.concat([
    df_zt[numeric_exist].reset_index(drop=True),
    df_zt[target_cols + ['p-powerfact']].reset_index(drop=True),
    crys_dummies.reset_index(drop=True),
    df_zt[['spg_number_norm', 'dimensionality_enc']].reset_index(drop=True)
], axis=1)

df_work = df_work.loc[:, ~df_work.columns.duplicated()].copy()
df_work = df_work.dropna(subset=target_cols)
print(f"Rows after target dropna: {len(df_work):,}")

# Missingness flags
for col in ['exfoliation_energy', 'slme', 'Tc_supercon', 'spillage']:
    if col in df_work.columns:
        df_work[f'{col}_missing'] = df_work[col].isna().astype(float)

# Impute
numeric_exist = [c for c in numeric_exist if c in df_work.columns]
for col in numeric_exist:
    df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
    arr = df_work[col].values.astype(np.float64)
    median_val = float(np.nanmedian(arr))
    df_work[col] = df_work[col].fillna(0.0 if np.isnan(median_val) else median_val)

# Clip outliers
for col in numeric_exist:
    arr = df_work[col].values.astype(np.float64)
    Q1, Q3 = float(np.nanpercentile(arr, 25)), float(np.nanpercentile(arr, 75))
    df_work[col] = df_work[col].clip(Q1 - 3*(Q3-Q1), Q3 + 3*(Q3-Q1))

# ============================================================================
# STEP 4: TARGET TRANSFORMS  (main 4 targets)
# ============================================================================
df_work['p-Seebeck_abs'] = np.abs(df_work['p-Seebeck'])
df_work['pcond_log']     = np.log10(df_work['pcond'].clip(lower=1e-10) + 1.0)
df_work['pkappa_log']    = np.log10(df_work['pkappa'].clip(lower=1e-10) + 1.0)
df_work['ZT_p']          = df_work['ZT_p'].clip(lower=0, upper=10)
df_work['ZT_p_log']      = np.log10(df_work['ZT_p'] + 1.0)
targets_proc = ['p-Seebeck_abs', 'pcond_log', 'pkappa_log', 'ZT_p_log']

# ============================================================================
# STEP 5: AUX TARGETS FOR PHYSICS HEADS  (log space, consistent units)
# ============================================================================
# Both are log10(x + 1) -- same functional form, same scale.
# The physics identity in this space is:
#   log10(PF + 1)  ~=  2 * S_log + sig_log   (approximately, for large PF)
# More precisely, we use the exact log10(PF + 1) as ground truth.

df_work['S_log']      = np.log10(df_work['p-Seebeck_abs'].clip(lower=1e-10) + 1.0)
df_work['sig_log']    = df_work['pcond_log'].copy()   # identical transform
df_work['log_PF']     = np.log10(
    df_work['p-powerfact'].fillna(0).clip(lower=0) + 1.0
)

aux_targets = ['S_log', 'sig_log']   # aux head training targets

# ------------------------------------------------------------------
# UNIT OFFSET: log10(PF) = 2*log10(|S|) + log10(sigma) - UNIT_OFFSET
# This offset accounts for the unit-conversion factor between the
# dataset columns (uV/K, S/m, uW/cm*K^2 etc.) and cancels the
# systematic bias in the physics residual so L_PF starts near 0.
# Computed from rows where all three quantities are well-defined.
# ------------------------------------------------------------------
mask_pf_valid = (
    (df_work['S_log'] > 0.1) &
    (df_work['sig_log'] > 0.1) &
    (df_work['log_PF'] > 0.1)
)
residuals = (
    2.0 * df_work.loc[mask_pf_valid, 'S_log']
    + df_work.loc[mask_pf_valid, 'sig_log']
    - df_work.loc[mask_pf_valid, 'log_PF']
)
UNIT_OFFSET = float(residuals.median())

print(f"\nAux target stats:")
print(f"  S_log      : min={df_work['S_log'].min():.3f}, max={df_work['S_log'].max():.3f}, "
      f"mean={df_work['S_log'].mean():.3f}")
print(f"  sig_log    : min={df_work['sig_log'].min():.3f}, max={df_work['sig_log'].max():.3f}, "
      f"mean={df_work['sig_log'].mean():.3f}")
print(f"  log_PF     : min={df_work['log_PF'].min():.3f}, max={df_work['log_PF'].max():.3f}, "
      f"mean={df_work['log_PF'].mean():.3f}")
print(f"  UNIT_OFFSET: {UNIT_OFFSET:.4f} log10-units  "
      f"(from {mask_pf_valid.sum():,} valid rows)")
print(f"  Identity check: 2*S_log + sig_log - UNIT_OFFSET ~ log_PF "
      f"(median residual after correction: "
      f"{float((residuals - UNIT_OFFSET).abs().median()):.4f} log10 units)")

# ============================================================================
# STEP 6: ASSEMBLE BLOCK ARRAYS
# ============================================================================
missing_flags = [c for c in df_work.columns if c.endswith('_missing')]

s_cols = list(dict.fromkeys(
    [c for c in structural_base    if c in df_work.columns] +
    [c for c in crys_dummies.columns if c in df_work.columns] +
    (['spg_number_norm']    if 'spg_number_norm'    in df_work.columns else []) +
    (['dimensionality_enc'] if 'dimensionality_enc' in df_work.columns else []) +
    [c for c in missing_flags if 'exfoliation' in c]
))
e_cols = list(dict.fromkeys([c for c in electronic_cols  if c in df_work.columns]))
t_cols = list(dict.fromkeys([c for c in thermal_cols     if c in df_work.columns]))
m_cols = list(dict.fromkeys([c for c in mechanical_cols  if c in df_work.columns]))

print(f"\nBlock sizes -- S:{len(s_cols)}, E:{len(e_cols)}, "
      f"T:{len(t_cols)}, M:{len(m_cols)}")

X_s = df_work[s_cols].values.astype(np.float32)
X_e = df_work[e_cols].values.astype(np.float32)
X_t = df_work[t_cols].values.astype(np.float32)
X_m = df_work[m_cols].values.astype(np.float32)
y   = df_work[targets_proc].values.astype(np.float32)       # main targets (4)
y_aux = df_work[aux_targets].values.astype(np.float32)      # aux targets  (2)
log_PF_all = df_work['log_PF'].values.astype(np.float32).reshape(-1, 1)

# Safety
for name, arr in [('S', X_s), ('E', X_e), ('T', X_t), ('M', X_m),
                   ('y', y), ('y_aux', y_aux), ('log_PF', log_PF_all)]:
    bad = np.isnan(arr).sum() + np.isinf(arr).sum()
    if bad > 0:
        print(f"  WARNING {name}: {bad} bad values -- zeroing")
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Dataset: {len(y):,} samples")

# ============================================================================
# STEP 7: TRAIN / TEST SPLIT + SCALE
# ============================================================================
idx = np.arange(len(y))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)

def ss(X, it, ite):
    sc = RobustScaler()
    return sc.fit_transform(X[it]), sc.transform(X[ite]), sc

X_s_tr, X_s_te, sc_s = ss(X_s, idx_tr, idx_te)
X_e_tr, X_e_te, sc_e = ss(X_e, idx_tr, idx_te)
X_t_tr, X_t_te, sc_t = ss(X_t, idx_tr, idx_te)
X_m_tr, X_m_te, sc_m = ss(X_m, idx_tr, idx_te)

# Main targets -- scaled (RobustScaler)
sc_y = RobustScaler()
y_tr_scaled = sc_y.fit_transform(y[idx_tr])
y_te_raw    = y[idx_te]

# Aux targets -- NOT scaled (already in log space, ~[0, 7] range)
# Keeping them unscaled makes the physics residual interpretable in log-units
y_aux_tr  = y_aux[idx_tr]
y_aux_te  = y_aux[idx_te]

# log_PF -- NOT scaled (same log space as aux targets)
log_PF_tr = log_PF_all[idx_tr]
log_PF_te = log_PF_all[idx_te]

print(f"Train: {len(idx_tr):,}  |  Test: {len(idx_te):,}")

# ============================================================================
# STEP 8: BUILD MODEL WITH 2 AUX PHYSICS HEADS
# ============================================================================
tf.random.set_seed(42)
np.random.seed(42)

def sub_block(inp, units, name):
    x = layers.Dense(units, activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name=f'{name}_dense')(inp)
    return layers.BatchNormalization(name=f'{name}_bn')(x)

inp_s = keras.Input(shape=(X_s_tr.shape[1],), name='structural')
inp_e = keras.Input(shape=(X_e_tr.shape[1],), name='electronic')
inp_t = keras.Input(shape=(X_t_tr.shape[1],), name='thermal')
inp_m = keras.Input(shape=(X_m_tr.shape[1],), name='mechanical')

merged = layers.Concatenate(name='merge')([
    sub_block(inp_s, 32, 'struct'),
    sub_block(inp_e, 64, 'elec'),
    sub_block(inp_t, 32, 'therm'),
    sub_block(inp_m, 16, 'mech')
])

x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(merged)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.1)(x)

# Main output heads (4 targets -- scaled space)
out_seebeck = layers.Dense(1, activation='linear', name='seebeck')(x)
out_cond    = layers.Dense(1, activation='linear', name='conductivity')(x)
out_kappa   = layers.Dense(1, activation='linear', name='kappa')(x)
out_zt      = layers.Dense(1, activation='linear', name='ZT')(x)
main_out    = layers.Concatenate(name='main_output')([out_seebeck, out_cond, out_kappa, out_zt])

# Auxiliary physics heads (2 targets -- unscaled log space)
# S_log_head  -> log10(|p-Seebeck| + 1)   in uV/K log units
# sig_log_head-> log10(pcond + 1)          in S/m log units
# These are the heads that feed the physics residual -- CONSISTENT units.
out_S_log   = layers.Dense(1, activation='softplus', name='S_log_head')(x)   # softplus: output >= 0
out_sig_log = layers.Dense(1, activation='softplus', name='sig_log_head')(x)  # softplus: output >= 0
aux_out     = layers.Concatenate(name='aux_output')([out_S_log, out_sig_log])

model = keras.Model(
    inputs=[inp_s, inp_e, inp_t, inp_m],
    outputs=[main_out, aux_out],
    name='pinn_stage2_v2'
)

print(f"\nModel: {model.count_params():,} parameters")
print("  Main heads: seebeck, conductivity, kappa, ZT  (scaled space)")
print("  Aux heads:  S_log, sig_log                    (unscaled log space)")

# ============================================================================
# STEP 9: LOSS WEIGHTS + OPTIMIZER
# ============================================================================
# LAMBDA_PF: weight for physics residual
#   - Too large: physics dominates, model ignores data
#   - Too small: no benefit
#   - 0.05 is a safe starting point for log-space PF residual
# Physics residual weight -- safe because L_PF now starts near 0
LAMBDA_PF  = 0.1
# Aux head data loss weight -- keeps S_log, sig_log grounded
LAMBDA_AUX = 1.0

optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Bake the empirical unit offset into a constant tensor
UNIT_OFFSET_T = tf.constant(UNIT_OFFSET, dtype=tf.float32)

@tf.function
def train_step(Xs, Xe, Xt, Xm, y_main_true, y_aux_true, log_pf_true):
    with tf.GradientTape() as tape:
        main_pred, aux_pred = model([Xs, Xe, Xt, Xm], training=True)

        # 1) Data loss: MSE on 4 main targets (scaled space)
        L_data = tf.reduce_mean(tf.square(main_pred - y_main_true))

        # 2) Aux head data loss: MSE on S_log, sig_log (unscaled log space)
        L_aux  = tf.reduce_mean(tf.square(aux_pred - y_aux_true))

        # 3) Physics residual -- unit-corrected, starts near 0:
        #    PF = S^2 * sigma  [in SI-consistent units]
        #    log10(PF + 1) = 2*log10(|S|+1) + log10(sigma+1) - UNIT_OFFSET
        #    UNIT_OFFSET absorbs the unit-conversion factor between columns.
        S_log_hat   = aux_pred[:, 0:1]
        sig_log_hat = aux_pred[:, 1:2]
        PF_log_hat  = 2.0 * S_log_hat + sig_log_hat - UNIT_OFFSET_T
        L_PF        = tf.reduce_mean(tf.square(PF_log_hat - log_pf_true))

        L_total = L_data + LAMBDA_AUX * L_aux + LAMBDA_PF * L_PF

    grads = tape.gradient(L_total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return L_total, L_data, L_aux, L_PF

@tf.function
def val_step(Xs, Xe, Xt, Xm, y_main_true, y_aux_true, log_pf_true):
    main_pred, aux_pred = model([Xs, Xe, Xt, Xm], training=False)
    L_data = tf.reduce_mean(tf.square(main_pred - y_main_true))
    L_aux  = tf.reduce_mean(tf.square(aux_pred - y_aux_true))
    S_log_hat   = aux_pred[:, 0:1]
    sig_log_hat = aux_pred[:, 1:2]
    PF_log_hat  = 2.0 * S_log_hat + sig_log_hat - UNIT_OFFSET_T
    L_PF        = tf.reduce_mean(tf.square(PF_log_hat - log_pf_true))
    L_total = L_data + LAMBDA_AUX * L_aux + LAMBDA_PF * L_PF
    return L_total, L_data, L_aux, L_PF

# ============================================================================
# STEP 10: TRAINING LOOP
# ============================================================================
EPOCHS     = 300
BATCH_SIZE = 64
PATIENCE   = 30

n_val = int(len(idx_tr) * 0.2)
n_tr  = len(idx_tr) - n_val

def t(a): return tf.constant(a.astype(np.float32))

tr_s  = t(X_s_tr[:n_tr]);      vl_s  = t(X_s_tr[n_tr:])
tr_e  = t(X_e_tr[:n_tr]);      vl_e  = t(X_e_tr[n_tr:])
tr_t  = t(X_t_tr[:n_tr]);      vl_t  = t(X_t_tr[n_tr:])
tr_m  = t(X_m_tr[:n_tr]);      vl_m  = t(X_m_tr[n_tr:])
tr_y  = t(y_tr_scaled[:n_tr]); vl_y  = t(y_tr_scaled[n_tr:])
tr_ya = t(y_aux_tr[:n_tr]);    vl_ya = t(y_aux_tr[n_tr:])
tr_pf = t(log_PF_tr[:n_tr]);   vl_pf = t(log_PF_tr[n_tr:])

nb = int(np.ceil(n_tr / BATCH_SIZE))
best_val = np.inf; pat = 0; best_w = None
hist = {'loss': [], 'val_loss': [], 'L_data': [], 'L_aux': [], 'L_PF': []}

print(f"\nTraining: LAMBDA_PF={LAMBDA_PF}, LAMBDA_AUX={LAMBDA_AUX}")
print("Physics residual uses aux heads in log space -- consistent units\n")

for ep in range(EPOCHS):
    perm = np.random.permutation(n_tr)
    ep_L, ep_Ld, ep_La, ep_Lpf = [], [], [], []

    for b in range(nb):
        bi = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        L, Ld, La, Lpf = train_step(
            tf.gather(tr_s, bi), tf.gather(tr_e, bi),
            tf.gather(tr_t, bi), tf.gather(tr_m, bi),
            tf.gather(tr_y, bi), tf.gather(tr_ya, bi),
            tf.gather(tr_pf, bi)
        )
        ep_L.append(float(L)); ep_Ld.append(float(Ld))
        ep_La.append(float(La)); ep_Lpf.append(float(Lpf))

    vL, vLd, vLa, vLpf = val_step(vl_s, vl_e, vl_t, vl_m, vl_y, vl_ya, vl_pf)
    vL = float(vL)

    hist['loss'].append(np.mean(ep_L))
    hist['val_loss'].append(vL)
    hist['L_data'].append(np.mean(ep_Ld))
    hist['L_aux'].append(np.mean(ep_La))
    hist['L_PF'].append(np.mean(ep_Lpf))

    if ep % 20 == 0:
        print(f"Ep {ep:3d} | val={vL:.4f} | L_data={np.mean(ep_Ld):.4f} | "
              f"L_aux={np.mean(ep_La):.4f} | L_PF={np.mean(ep_Lpf):.4f}")

    if vL < best_val - 1e-5:
        best_val = vL; pat = 0; best_w = model.get_weights()
    else:
        pat += 1
        if pat >= PATIENCE:
            print(f"\nEarly stopping at epoch {ep}")
            break

model.set_weights(best_w)
print(f"\nBest val loss: {best_val:.4f}")

# ============================================================================
# STEP 11: EVALUATE
# ============================================================================
print("\n[EVALUATION]")

main_pred_scaled, aux_pred = model.predict(
    [X_s_te, X_e_te, X_t_te, X_m_te], verbose=0
)
y_pred = sc_y.inverse_transform(main_pred_scaled)

print(f"\n{'Target':<22} {'MAE':<12} {'R2':<10} {'Note'}")
print("-" * 70)
for i, tgt in enumerate(targets_proc):
    mae = mean_absolute_error(y_te_raw[:, i], y_pred[:, i])
    r2  = r2_score(y_te_raw[:, i], y_pred[:, i])
    var = float(np.var(y_te_raw[:, i]))
    note = f"var={var:.5f} -- use Pearson" if var < 0.01 else ""
    print(f"{tgt:<22} {mae:<12.4f} {r2:<10.4f} {note}")

r2_valid = np.mean([r2_score(y_te_raw[:, i], y_pred[:, i]) for i in range(3)])
print(f"\n  Mean R2 (Seebeck, cond, kappa): {r2_valid:.4f}")

zt_true = y_te_raw[:, 3]
zt_pred = y_pred[:, 3]
pr, _ = pearsonr(zt_true, zt_pred)
sr, _ = spearmanr(zt_true, zt_pred)
print(f"  ZT Pearson r:  {pr:.4f}")
print(f"  ZT Spearman r: {sr:.4f}")

# ---- Aux head evaluation (log-space accuracy) ----
print(f"\n[AUX HEAD EVALUATION] (log10 space)")
print(f"{'Aux target':<20} {'MAE':<12} {'R2':<10}")
print("-" * 44)
for i, tgt in enumerate(aux_targets):
    mae = mean_absolute_error(y_aux_te[:, i], aux_pred[:, i])
    r2  = r2_score(y_aux_te[:, i], aux_pred[:, i])
    print(f"{tgt:<20} {mae:<12.4f} {r2:<10.4f}")

# ---- Physics consistency check (log space, unit-corrected) ----
log_PF_true_te = log_PF_te.flatten()
S_log_pred     = aux_pred[:, 0]      # log10(|S_pred| + 1)
sig_log_pred   = aux_pred[:, 1]      # log10(sigma_pred + 1)
# Apply the same UNIT_OFFSET used in training
log_PF_pred    = 2.0 * S_log_pred + sig_log_pred - UNIT_OFFSET

valid = log_PF_true_te > 0.01
if valid.sum() > 10:
    pf_r2  = r2_score(log_PF_true_te[valid], log_PF_pred[valid])
    pf_mae = mean_absolute_error(log_PF_true_te[valid], log_PF_pred[valid])
    residual_bias = float(np.mean(log_PF_pred[valid] - log_PF_true_te[valid]))
    print(f"\n[Physics Check] 2*S_log + sig_log - {UNIT_OFFSET:.3f} vs log10(PF_true + 1)")
    print(f"  R2   = {pf_r2:.4f}  (closer to 1.0 = better physics consistency)")
    print(f"  MAE  = {pf_mae:.4f} log10 units")
    print(f"  Bias = {residual_bias:+.4f} log10 units  (should be ~0 if offset correct)")
else:
    print("\n  Not enough valid PF samples for physics check")

print("\n[Stage 2 complete] Next: stage3 adds ZT consistency loss.")