"""
STAGE 1: Block-wise MLP using Keras Functional API
====================================================
- Expands from 13 features to all 4 physics-meaningful blocks
- Uses Functional API (not Sequential) — required for multi-input architecture
- No physics loss yet — just gets the architecture right
- Targets: p-Seebeck, pcond, pkappa, ZT_p (computed from columns)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 1: Block-wise MLP (Functional API, Expanded Features)")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
df = pd.read_csv('C:\\Users\\Puneetha\\thermoelectric_dataset\\output\\thermoelectric_ml_ready.csv', low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ============================================================================
# STEP 2: ENCODE CATEGORICAL COLUMNS
# ============================================================================
print("\n[STEP 2] Encoding categorical columns...")

crys_dummies = pd.get_dummies(df['crys'], prefix='crys').astype(float)
df['spg_number_norm']  = pd.to_numeric(df['spg_number'], errors='coerce').fillna(0) / 230.0
dim_map = {'3D': 3, '2D': 2, '1D': 1, '0D': 0}
df['dimensionality_enc'] = df['dimensionality'].map(dim_map).fillna(2)

print(f"  crys one-hot: {crys_dummies.shape[1]} columns")
print(f"  spg_number normalized, dimensionality encoded")

# ============================================================================
# STEP 3: COMPUTE ZT TARGET (p-type, T=300K)
# ============================================================================
T = 300.0
print("\n[STEP 3] Computing ZT_p target...")

df_zt = df.copy()

# Force these to numeric first in case of string artefacts
for col in ['p-powerfact', 'pkappa']:
    df_zt[col] = pd.to_numeric(df_zt[col], errors='coerce')

mask_valid = (
    df_zt['p-powerfact'].notna() &
    df_zt['pkappa'].notna() &
    (df_zt['pkappa'] > 0)
)
df_zt['ZT_p'] = np.nan
df_zt.loc[mask_valid, 'ZT_p'] = (
    df_zt.loc[mask_valid, 'p-powerfact'] * T /
    df_zt.loc[mask_valid, 'pkappa']
)
df_zt['ZT_p'] = df_zt['ZT_p'].clip(0, 10)
print(f"  Valid ZT_p rows: {mask_valid.sum():,}")

# ============================================================================
# STEP 4: DEFINE FEATURE COLUMNS PER BLOCK
# ============================================================================
print("\n[STEP 4] Defining feature blocks...")

structural_base = [
    'nat', 'density', 'exfoliation_energy',
    'formation_energy_peratom', 'ehull'
]
electronic_cols = [
    'optb88vdw_bandgap', 'mbj_bandgap', 'hse_gap',
    'avg_elec_mass', 'avg_hole_mass',
    'epsx', 'epsy', 'epsz', 'mepsx', 'mepsy', 'mepsz',
    'slme', 'spillage', 'magmom_oszicar', 'magmom_outcar'
]
thermal_cols = [
    'n-Seebeck', 'n-powerfact', 'ncond', 'nkappa',
    'optb88vdw_total_energy', 'Tc_supercon',
    'max_ir_mode', 'min_ir_mode'
]
mechanical_cols = [
    'bulk_modulus_kv', 'shear_modulus_gv', 'poisson',
    'dfpt_piezo_max_dielectric', 'dfpt_piezo_max_dielectric_electronic',
    'dfpt_piezo_max_dielectric_ionic', 'dfpt_piezo_max_eij', 'dfpt_piezo_max_dij',
    'max_efg', 'efg'
]
target_cols  = ['p-Seebeck', 'pcond', 'pkappa', 'ZT_p']
all_numeric  = structural_base + electronic_cols + thermal_cols + mechanical_cols

# ============================================================================
# STEP 5: FORCE ALL FEATURE + TARGET COLUMNS TO NUMERIC
# ============================================================================
print("\n[STEP 5] Force-converting all columns to numeric...")

for col in all_numeric + target_cols:
    if col in df_zt.columns:
        df_zt[col] = pd.to_numeric(df_zt[col], errors='coerce')

# ============================================================================
# STEP 6: BUILD df_work — one concat, deduplicate immediately
# ============================================================================
print("\n[STEP 6] Building working dataframe...")

numeric_exist = [c for c in all_numeric if c in df_zt.columns]
target_exist  = [c for c in target_cols if c in df_zt.columns]

df_work = pd.concat([
    df_zt[numeric_exist].reset_index(drop=True),
    df_zt[target_exist].reset_index(drop=True),
    crys_dummies.reset_index(drop=True),
    df_zt[['spg_number_norm', 'dimensionality_enc']].reset_index(drop=True)
], axis=1)

# Deduplicate columns immediately — prevents the Series/median bug
df_work = df_work.loc[:, ~df_work.columns.duplicated()].copy()

# Drop rows missing any target
df_work = df_work.dropna(subset=target_exist)
print(f"  Rows after target dropna: {len(df_work):,}")

# ============================================================================
# STEP 7: MISSINGNESS FLAGS
# ============================================================================
print("\n[STEP 7] Adding missingness flags...")

sparse_cols = ['exfoliation_energy', 'slme', 'Tc_supercon', 'spillage']
for col in sparse_cols:
    if col in df_work.columns:
        df_work[f'{col}_missing'] = df_work[col].isna().astype(float)

# ============================================================================
# STEP 8: FORCE NUMERIC + IMPUTE WITH SCALAR MEDIAN
# ============================================================================
print("\n[STEP 8] Imputing missing values...")

numeric_exist = [c for c in numeric_exist if c in df_work.columns]

for col in numeric_exist:
    # pd.to_numeric converts any leftover strings to NaN
    df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
    # Cast to float64 array — np.nanmedian on this always returns a plain float
    arr = df_work[col].values.astype(np.float64)
    median_val = float(np.nanmedian(arr))
    if np.isnan(median_val):
        median_val = 0.0
    df_work[col] = df_work[col].fillna(median_val)

remaining_nan = int(df_work[numeric_exist].isna().sum().sum())
print(f"  Remaining NaNs in features: {remaining_nan}")

# ============================================================================
# STEP 9: OUTLIER CLIPPING (3xIQR)
# ============================================================================
print("\n[STEP 9] Clipping outliers (3x IQR)...")

for col in numeric_exist:
    arr = df_work[col].values.astype(np.float64)
    Q1  = float(np.nanpercentile(arr, 25))
    Q3  = float(np.nanpercentile(arr, 75))
    IQR = Q3 - Q1
    df_work[col] = df_work[col].clip(Q1 - 3*IQR, Q3 + 3*IQR)

# ============================================================================
# STEP 10: TARGET TRANSFORMS
# ============================================================================
print("\n[STEP 10] Transforming targets...")

df_work['p-Seebeck_abs'] = np.abs(df_work['p-Seebeck'])
df_work['pcond_log']     = np.log10(df_work['pcond'].clip(lower=1e-10) + 1.0)
df_work['pkappa_log']    = np.log10(df_work['pkappa'].clip(lower=1e-10) + 1.0)

# ZT fix: clip to reasonable range first, then log10(ZT + 1)
# log10(ZT + 1e-6) causes near-zero variance since most ZT << 1
# log10(ZT + 1) maps [0, 10] -> [0, 1.04] — much better spread
df_work['ZT_p'] = df_work['ZT_p'].clip(lower=0, upper=10)
df_work['ZT_p_log'] = np.log10(df_work['ZT_p'] + 1.0)

# Print ZT distribution so we can see the spread
zt_vals = df_work['ZT_p_log']
print(f"  ZT_p_log stats: min={zt_vals.min():.4f}, max={zt_vals.max():.4f}, "
      f"mean={zt_vals.mean():.4f}, std={zt_vals.std():.4f}")

targets_proc = ['p-Seebeck_abs', 'pcond_log', 'pkappa_log', 'ZT_p_log']

# ============================================================================
# STEP 11: ASSEMBLE BLOCK ARRAYS
# ============================================================================
print("\n[STEP 11] Assembling block arrays...")

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

print(f"  Block sizes — S:{len(s_cols)}, E:{len(e_cols)}, "
      f"T:{len(t_cols)}, M:{len(m_cols)}")

X_s = df_work[s_cols].values.astype(np.float32)
X_e = df_work[e_cols].values.astype(np.float32)
X_t = df_work[t_cols].values.astype(np.float32)
X_m = df_work[m_cols].values.astype(np.float32)
y   = df_work[targets_proc].values.astype(np.float32)

# Sanity check
for name, arr in [('S', X_s), ('E', X_e), ('T', X_t), ('M', X_m), ('y', y)]:
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING block {name}: {n_nan} NaNs, {n_inf} Infs — filling with 0")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if name == 'S': X_s = arr
        elif name == 'E': X_e = arr
        elif name == 'T': X_t = arr
        elif name == 'M': X_m = arr
        elif name == 'y': y   = arr

print("  Block arrays ready")

# ============================================================================
# STEP 12: TRAIN / TEST SPLIT + SCALE
# ============================================================================
print("\n[STEP 12] Splitting and scaling...")

idx = np.arange(len(y))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

X_s_tr, X_s_te = X_s[idx_train], X_s[idx_test]
X_e_tr, X_e_te = X_e[idx_train], X_e[idx_test]
X_t_tr, X_t_te = X_t[idx_train], X_t[idx_test]
X_m_tr, X_m_te = X_m[idx_train], X_m[idx_test]
y_tr,   y_te   = y[idx_train],   y[idx_test]

def fit_scale(train, test):
    sc = RobustScaler()
    return sc.fit_transform(train), sc.transform(test), sc

X_s_tr, X_s_te, sc_s = fit_scale(X_s_tr, X_s_te)
X_e_tr, X_e_te, sc_e = fit_scale(X_e_tr, X_e_te)
X_t_tr, X_t_te, sc_t = fit_scale(X_t_tr, X_t_te)
X_m_tr, X_m_te, sc_m = fit_scale(X_m_tr, X_m_te)

sc_y        = RobustScaler()
y_tr_scaled = sc_y.fit_transform(y_tr)

print(f"  Train: {len(idx_train):,}  |  Test: {len(idx_test):,}")

# ============================================================================
# STEP 13: BUILD BLOCK-WISE MODEL (Functional API)
# ============================================================================
print("\n[STEP 13] Building model...")

tf.random.set_seed(42)
np.random.seed(42)

def sub_block(inp, units, name):
    x = layers.Dense(units, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name=f'{name}_dense')(inp)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    return x

inp_s = keras.Input(shape=(X_s_tr.shape[1],), name='structural')
inp_e = keras.Input(shape=(X_e_tr.shape[1],), name='electronic')
inp_t = keras.Input(shape=(X_t_tr.shape[1],), name='thermal')
inp_m = keras.Input(shape=(X_m_tr.shape[1],), name='mechanical')

b_s = sub_block(inp_s, 32, 'struct')
b_e = sub_block(inp_e, 64, 'elec')
b_t = sub_block(inp_t, 32, 'therm')
b_m = sub_block(inp_m, 16, 'mech')

merged = layers.Concatenate(name='merge')([b_s, b_e, b_t, b_m])

x = layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4))(merged)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.1)(x)

out_seebeck = layers.Dense(1, activation='linear', name='seebeck')(x)
out_cond    = layers.Dense(1, activation='linear', name='conductivity')(x)
out_kappa   = layers.Dense(1, activation='linear', name='kappa')(x)
out_zt      = layers.Dense(1, activation='linear', name='ZT')(x)

output = layers.Concatenate(name='output')(
    [out_seebeck, out_cond, out_kappa, out_zt]
)

model = keras.Model(
    inputs=[inp_s, inp_e, inp_t, inp_m],
    outputs=output,
    name='blockwise_mlp'
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ============================================================================
# STEP 14: TRAIN
# ============================================================================
print("\n[STEP 14] Training...")

history = model.fit(
    [X_s_tr, X_e_tr, X_t_tr, X_m_tr],
    y_tr_scaled,
    validation_split=0.2,
    epochs=300,
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=30,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=15, min_lr=1e-6, verbose=0
        )
    ],
    verbose=1
)

print(f"Stopped at epoch: {len(history.history['loss'])}")

# ============================================================================
# STEP 15: EVALUATE
# ============================================================================
print("\n[STEP 15] Evaluation...")

y_pred_scaled = model.predict(
    [X_s_te, X_e_te, X_t_te, X_m_te], verbose=0
)
y_pred = sc_y.inverse_transform(y_pred_scaled)

print(f"\n{'Target':<22} {'MAE':<12} {'R2':<10} {'Note'}")
print("-" * 65)
for i, tgt in enumerate(targets_proc):
    mae = mean_absolute_error(y_te[:, i], y_pred[:, i])
    r2  = r2_score(y_te[:, i], y_pred[:, i])
    # R2 is unreliable when variance is very low — flag it
    variance = float(np.var(y_te[:, i]))
    note = f"var={variance:.5f} — R2 unreliable" if variance < 0.01 else ""
    print(f"{tgt:<22} {mae:<12.4f} {r2:<10.4f} {note}")

# For ZT specifically, also show correlation — more meaningful than R2 here
from scipy.stats import pearsonr, spearmanr
zt_true = y_te[:, 3]
zt_pred = y_pred[:, 3]
pearson_r,  _ = pearsonr(zt_true, zt_pred)
spearman_r, _ = spearmanr(zt_true, zt_pred)
print(f"\n  ZT_p_log additional metrics:")
print(f"    Pearson r:  {pearson_r:.4f}")
print(f"    Spearman r: {spearman_r:.4f}")
print(f"    (These matter more than R2 when variance is low)")

r2_valid = [r2_score(y_te[:, i], y_pred[:, i]) for i in range(3)]  # exclude ZT
print(f"\n  R2 for first 3 targets (reliable): {np.mean(r2_valid):.4f}")
print(f"  ZT Pearson r: {pearson_r:.4f}")
print("\nStage 1 complete. Next: run stage2_pinn_pf_loss.py")