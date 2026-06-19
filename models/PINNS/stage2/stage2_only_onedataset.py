"""
STAGE 4: FULL PINN -- Power Factor + ZT Consistency + Wiedemann-Franz
======================================================================
Data pulled directly from JARVIS-DFT via jarvis-tools (no local CSV needed).

================================================================
  *** CHANGE THIS ONE LINE TO SWITCH BETWEEN 3D AND 2D ***

    DATASET = "dft_3d"    <- 3D materials (~55,000 entries)
    DATASET = "dft_2d"    <- 2D materials (~1,000 entries)

================================================================

PHYSICS CONSTRAINTS IN THIS FILE:
  1. Power Factor (PF):  PF = S^2 * sigma         -> log: 2*S_log + sig_log = log_PF   (lambda_PF = 0.10)
  2. ZT Consistency:     ZT = S^2 * sigma * T / kappa
                         -> log: 2*S_log + sig_log - kappa_log + log10(T) = log_ZT     (lambda_ZT = 1.00)
  3. Wiedemann-Franz:    kappa_e / sigma = L * T  -> log: sig_log + log10(L_eff) + log10(T) = kappa_log
                         (lambda_WF = 0.05, approximate -- see notes below)

All three residuals are computed on UNSCALED LOG-SPACE AUX HEADS only
(S_log, sig_log, kappa_log), never on the RobustScaler'd main heads.
Each constraint has its own empirically-derived offset constant
(UNIT_OFFSET_PF, UNIT_OFFSET_ZT, UNIT_OFFSET_WF) computed as the median
residual over the training set -- this absorbs unit mismatches between
dataset columns and (for WF) the gap between kappa_total and kappa_electronic.

Reference for WF constraint formulation:
  Yadav, Deshmukh, Roberts, Jisrawi, Valluri,
  "An Analytic Study of the Wiedemann-Franz Law and the Thermoelectric
  Figure of Merit", J. Phys. Commun. 3 (2019) 105001.
  ZT = alpha^2 * sigma / kappa ; WF: kappa/(sigma*T) -> Lorenz number L.
  The paper shows the generalized Lorenz number is NOT a universal constant
  for semiconductors (depends on reduced chemical potential & scattering
  parameter r), so here L_eff is fit empirically from the data rather than
  fixed to the free-electron value L0 = 2.44e-8 W*Ohm/K^2. This is why
  lambda_WF is kept small (0.05) -- it's the loosest of the three constraints.

LAMBDA SWEEP:
  At the end, the script automatically retrains with 4 different lambda
  configurations and prints/plots a comparison so you can show how the
  physics constraints trade off against each other and against pure data loss.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# >>> CHANGE THIS ONE LINE TO SWITCH DATASET <<<
# ============================================================================
DATASET = "dft_3d"    # options: "dft_3d"  |  "dft_2d"
# ============================================================================

print("=" * 80)
print(f"STAGE 4: FULL PINN -- PF + ZT + Wiedemann-Franz  |  dataset={DATASET}")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD FROM JARVIS
# ============================================================================
try:
    from jarvis.db.figshare import data as jdata
except ImportError:
    raise ImportError(
        "jarvis-tools not installed.\n"
        "Run:  pip install jarvis-tools"
    )

print(f"\nDownloading {DATASET} from JARVIS-DFT (cached after first run)...")
raw = jdata(DATASET)
df = pd.DataFrame(raw)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ============================================================================
# STEP 2: COLUMN NAME NORMALISATION
# ============================================================================
COL_MAP = {
    # Seebeck
    'p_seebeck_300': 'p-Seebeck',
    'p-seebeck_300': 'p-Seebeck',
    'p_seebeck':     'p-Seebeck',

    # Electrical conductivity
    'p_eleccond_300': 'pcond',
    'p-eleccond_300': 'pcond',
    'p_eleccond':     'pcond',

    # Thermal conductivity
    'kappa_l':        'pkappa',
    'kappa':          'pkappa',

    # Power factor
    'p_powerfact_300': 'p-powerfact',
    'p-powerfact_300': 'p-powerfact',
    'p_powerfact':     'p-powerfact',

    # n-type equivalents
    'n_seebeck_300':   'n-Seebeck',
    'n-seebeck_300':   'n-Seebeck',
    'n_seebeck':       'n-Seebeck',
    'n_eleccond_300':  'ncond',
    'n-eleccond_300':  'ncond',
    'n_eleccond':      'ncond',
    'n_kappa_l':       'nkappa',
    'n_kappa':         'nkappa',
    'n_powerfact_300': 'n-powerfact',
    'n-powerfact_300': 'n-powerfact',
    'n_powerfact':     'n-powerfact',

    # Structure
    'atoms.composition.reduced_formula': 'formula',
    'spg_number':      'spg_number',
    'spg_number_dft':  'spg_number',
    'crys':            'crys',
    'crystal_system':  'crys',
}

df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

target_candidates = ['p-Seebeck', 'pcond', 'pkappa', 'p-powerfact',
                     'n-Seebeck', 'ncond', 'nkappa', 'n-powerfact']
print("\nTarget columns found:")
for c in target_candidates:
    found = c in df.columns
    sample = f"  (sample: {df[c].dropna().iloc[0] if found and df[c].notna().any() else 'N/A'})"
    print(f"  {'OK' if found else 'MISSING'}  {c}{sample if found else ''}")

# ============================================================================
# STEP 3: ENCODE CATEGORICALS
# ============================================================================
if 'crys' in df.columns:
    crys_dummies = pd.get_dummies(df['crys'], prefix='crys').astype(float)
else:
    print("  WARNING: 'crys' column not found -- skipping one-hot encoding")
    crys_dummies = pd.DataFrame(index=df.index)

if 'spg_number' in df.columns:
    df['spg_number_norm'] = pd.to_numeric(df['spg_number'], errors='coerce').fillna(0) / 230.0
else:
    df['spg_number_norm'] = 0.0

dim_map = {'3D': 3, '2D': 2, '1D': 1, '0D': 0}
if 'dimensionality' in df.columns:
    df['dimensionality_enc'] = df['dimensionality'].map(dim_map).fillna(
        3 if DATASET == 'dft_3d' else 2
    )
else:
    df['dimensionality_enc'] = 3 if DATASET == 'dft_3d' else 2

# ============================================================================
# STEP 4: COMPUTE ZT_p
# ============================================================================
T_val = 300.0
for col in ['p-powerfact', 'pkappa', 'p-Seebeck', 'pcond']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'p-powerfact' in df.columns and 'pkappa' in df.columns:
    mask_zt = df['p-powerfact'].notna() & df['pkappa'].notna() & (df['pkappa'] > 0)
    df['ZT_p'] = np.nan
    df.loc[mask_zt, 'ZT_p'] = df.loc[mask_zt, 'p-powerfact'] * T_val / df.loc[mask_zt, 'pkappa']
    df['ZT_p'] = df['ZT_p'].clip(0, 10)
    print(f"\nValid ZT_p rows: {mask_zt.sum():,}")
else:
    raise ValueError(
        "Cannot compute ZT_p: 'p-powerfact' or 'pkappa' missing after column normalisation.\n"
        "Check COL_MAP above and add the correct column name for this dataset."
    )

# ============================================================================
# STEP 5: FEATURE BLOCKS
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
# STEP 6: BUILD df_work
# ============================================================================
for col in all_numeric + target_cols + ['p-powerfact']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

numeric_exist = [c for c in all_numeric if c in df.columns]

df_work = pd.concat([
    df[numeric_exist].reset_index(drop=True),
    df[target_cols + ['p-powerfact']].reset_index(drop=True),
    crys_dummies.reset_index(drop=True),
    df[['spg_number_norm', 'dimensionality_enc']].reset_index(drop=True)
], axis=1)

df_work = df_work.loc[:, ~df_work.columns.duplicated()].copy()
df_work = df_work.dropna(subset=target_cols)
print(f"Rows after target dropna: {len(df_work):,}")

if len(df_work) < 100:
    raise ValueError(
        f"Only {len(df_work)} rows after filtering -- too few to train.\n"
        "Check that transport property columns exist in this dataset."
    )

# Missingness flags
for col in ['exfoliation_energy', 'slme', 'Tc_supercon', 'spillage']:
    if col in df_work.columns:
        df_work[f'{col}_missing'] = df_work[col].isna().astype(float)

# Impute with median
numeric_exist = [c for c in numeric_exist if c in df_work.columns]
for col in numeric_exist:
    df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
    arr = df_work[col].values.astype(np.float64)
    median_val = float(np.nanmedian(arr))
    df_work[col] = df_work[col].fillna(0.0 if np.isnan(median_val) else median_val)

# Clip outliers at IQR * 3
for col in numeric_exist:
    arr = df_work[col].values.astype(np.float64)
    Q1 = float(np.nanpercentile(arr, 25))
    Q3 = float(np.nanpercentile(arr, 75))
    df_work[col] = df_work[col].clip(Q1 - 3*(Q3-Q1), Q3 + 3*(Q3-Q1))

# ============================================================================
# STEP 7: TARGET TRANSFORMS
# ============================================================================
df_work['p-Seebeck_abs'] = np.abs(df_work['p-Seebeck'])
df_work['pcond_log']     = np.log10(df_work['pcond'].clip(lower=1e-10) + 1.0)
df_work['pkappa_log']    = np.log10(df_work['pkappa'].clip(lower=1e-10) + 1.0)
df_work['ZT_p']          = df_work['ZT_p'].clip(lower=0, upper=10)
df_work['ZT_p_log']      = np.log10(df_work['ZT_p'] + 1.0)
targets_proc = ['p-Seebeck_abs', 'pcond_log', 'pkappa_log', 'ZT_p_log']

# ============================================================================
# STEP 8: AUX TARGETS + UNIT OFFSETS (PF, ZT, WF)
# ============================================================================
df_work['S_log']     = np.log10(df_work['p-Seebeck_abs'].clip(lower=1e-10) + 1.0)
df_work['sig_log']   = df_work['pcond_log'].copy()
df_work['kappa_log'] = df_work['pkappa_log'].copy()
df_work['log_PF']    = np.log10(
    df_work['p-powerfact'].fillna(0).clip(lower=0) + 1.0
)
df_work['log_ZT']    = df_work['ZT_p_log'].copy()
aux_targets = ['S_log', 'sig_log', 'kappa_log']

log10_T = float(np.log10(T_val))

# --- UNIT_OFFSET_PF: 2*S_log + sig_log - log_PF -----------------------------
mask_pf_valid = (
    (df_work['S_log'] > 0.1) &
    (df_work['sig_log'] > 0.1) &
    (df_work['log_PF'] > 0.1)
)
residuals_pf = (
    2.0 * df_work.loc[mask_pf_valid, 'S_log']
    + df_work.loc[mask_pf_valid, 'sig_log']
    - df_work.loc[mask_pf_valid, 'log_PF']
)
UNIT_OFFSET_PF = float(residuals_pf.median())

# --- UNIT_OFFSET_ZT: 2*S_log + sig_log - kappa_log + log10(T) - log_ZT ------
# NOTE: ZT_p is often ~0 (log_ZT = log10(ZT_p+1) ~ 0), so a strict
# "log_ZT > 0.001" filter can leave zero valid rows -> NaN offset, which
# poisons the entire loss (all terms share one backward pass). Instead,
# only require S_log/sig_log/kappa_log to be meaningfully nonzero, and fall
# back to UNIT_OFFSET_PF - UNIT_OFFSET_WF (algebraically consistent) if
# nothing passes even that.
mask_zt_valid = (
    (df_work['S_log'] > 0.1) &
    (df_work['sig_log'] > 0.1) &
    (df_work['kappa_log'] > 0.01)
)
residuals_zt = (
    2.0 * df_work.loc[mask_zt_valid, 'S_log']
    + df_work.loc[mask_zt_valid, 'sig_log']
    - df_work.loc[mask_zt_valid, 'kappa_log']
    + log10_T
    - df_work.loc[mask_zt_valid, 'log_ZT']
)
UNIT_OFFSET_ZT = float(residuals_zt.median())
if not np.isfinite(UNIT_OFFSET_ZT):
    # Fallback: PF says 2*S_log+sig_log ~ log_PF + OFFSET_PF
    # WF says   sig_log - kappa_log ~ -OFFSET_WF - log10(T)
    # => 2*S_log+sig_log-kappa_log+log10(T) ~ log_PF + OFFSET_PF - OFFSET_WF
    # so a reasonable offset (relative to log_ZT~0) is OFFSET_PF - OFFSET_WF
    UNIT_OFFSET_ZT = UNIT_OFFSET_PF - UNIT_OFFSET_WF
    print(f"  WARNING: UNIT_OFFSET_ZT had no valid rows -- "
          f"using fallback OFFSET_PF - OFFSET_WF = {UNIT_OFFSET_ZT:.4f}")

# --- UNIT_OFFSET_WF: kappa_log - sig_log - log10(T)  (-> log10(L_eff)) ------
mask_wf_valid = (
    (df_work['sig_log'] > 0.1) &
    (df_work['kappa_log'] > 0.01)
)
residuals_wf = (
    df_work.loc[mask_wf_valid, 'kappa_log']
    - df_work.loc[mask_wf_valid, 'sig_log']
    - log10_T
)
UNIT_OFFSET_WF = float(residuals_wf.median())  # this IS log10(L_eff)

print(f"\nAux target stats:")
print(f"  S_log         : mean={df_work['S_log'].mean():.3f}, max={df_work['S_log'].max():.3f}")
print(f"  sig_log       : mean={df_work['sig_log'].mean():.3f}, max={df_work['sig_log'].max():.3f}")
print(f"  kappa_log     : mean={df_work['kappa_log'].mean():.3f}, max={df_work['kappa_log'].max():.3f}")
print(f"  log_PF        : mean={df_work['log_PF'].mean():.3f}")
print(f"  log_ZT        : mean={df_work['log_ZT'].mean():.3f}")
print(f"  UNIT_OFFSET_PF: {UNIT_OFFSET_PF:.4f}  (from {mask_pf_valid.sum():,} valid rows)")
print(f"  UNIT_OFFSET_ZT: {UNIT_OFFSET_ZT:.4f}  (from {mask_zt_valid.sum():,} valid rows)")
print(f"  UNIT_OFFSET_WF: {UNIT_OFFSET_WF:.4f}  (= log10(L_eff), from {mask_wf_valid.sum():,} valid rows)")

# ============================================================================
# STEP 9: ASSEMBLE ARRAYS
# ============================================================================
missing_flags = [c for c in df_work.columns if c.endswith('_missing')]

s_cols = list(dict.fromkeys(
    [c for c in structural_base         if c in df_work.columns] +
    [c for c in crys_dummies.columns    if c in df_work.columns] +
    (['spg_number_norm']    if 'spg_number_norm'    in df_work.columns else []) +
    (['dimensionality_enc'] if 'dimensionality_enc' in df_work.columns else []) +
    [c for c in missing_flags if 'exfoliation' in c]
))
e_cols = list(dict.fromkeys([c for c in electronic_cols if c in df_work.columns]))
t_cols = list(dict.fromkeys([c for c in thermal_cols    if c in df_work.columns]))
m_cols = list(dict.fromkeys([c for c in mechanical_cols if c in df_work.columns]))

print(f"\nBlock sizes -- S:{len(s_cols)}, E:{len(e_cols)}, T:{len(t_cols)}, M:{len(m_cols)}")

X_s = df_work[s_cols].values.astype(np.float32)
X_e = df_work[e_cols].values.astype(np.float32)
X_t = df_work[t_cols].values.astype(np.float32)
X_m = df_work[m_cols].values.astype(np.float32)
y          = df_work[targets_proc].values.astype(np.float32)
y_aux      = df_work[aux_targets].values.astype(np.float32)
log_PF_all = df_work['log_PF'].values.astype(np.float32).reshape(-1, 1)
log_ZT_all = df_work['log_ZT'].values.astype(np.float32).reshape(-1, 1)

for name, arr in [('S', X_s), ('E', X_e), ('T', X_t), ('M', X_m),
                   ('y', y), ('y_aux', y_aux),
                   ('log_PF', log_PF_all), ('log_ZT', log_ZT_all)]:
    bad = np.isnan(arr).sum() + np.isinf(arr).sum()
    if bad > 0:
        print(f"  WARNING {name}: {bad} bad values -- zeroing")
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Dataset: {len(y):,} samples")

# ============================================================================
# STEP 10: TRAIN / TEST SPLIT + SCALE
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

sc_y = RobustScaler()
y_tr_scaled = sc_y.fit_transform(y[idx_tr])
y_te_raw    = y[idx_te]

y_aux_tr  = y_aux[idx_tr]
y_aux_te  = y_aux[idx_te]
log_PF_tr = log_PF_all[idx_tr]
log_PF_te = log_PF_all[idx_te]
log_ZT_tr = log_ZT_all[idx_tr]
log_ZT_te = log_ZT_all[idx_te]

print(f"Train: {len(idx_tr):,}  |  Test: {len(idx_te):,}")

# ============================================================================
# STEP 11: AUTO-TUNE HYPERPARAMS FOR SMALL DATASETS (dft_2d)
# ============================================================================
n_total = len(y)

if n_total < 1000:
    BATCH_SIZE = 16
    PATIENCE   = 50
    DROPOUT_1  = 0.1
    DROPOUT_2  = 0.1
    DROPOUT_3  = 0.05
    print(f"\n[2D mode] Small dataset ({n_total} rows) -- "
          f"batch={BATCH_SIZE}, patience={PATIENCE}, reduced dropout")
else:
    BATCH_SIZE = 64
    PATIENCE   = 30
    DROPOUT_1  = 0.3
    DROPOUT_2  = 0.2
    DROPOUT_3  = 0.1
    print(f"\n[3D mode] Large dataset ({n_total} rows) -- "
          f"batch={BATCH_SIZE}, patience={PATIENCE}")

# ============================================================================
# STEP 12: BUILD MODEL  (4 main heads + 3 aux heads: S_log, sig_log, kappa_log)
# ============================================================================
def sub_block(inp, units, name):
    x = layers.Dense(units, activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name=f'{name}_dense')(inp)
    return layers.BatchNormalization(name=f'{name}_bn')(x)

def build_model(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

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
    x = layers.Dropout(DROPOUT_1)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(DROPOUT_3)(x)

    # Main heads -- 4 targets in scaled space
    out_seebeck = layers.Dense(1, activation='linear', name='seebeck')(x)
    out_cond    = layers.Dense(1, activation='linear', name='conductivity')(x)
    out_kappa   = layers.Dense(1, activation='linear', name='kappa')(x)
    out_zt      = layers.Dense(1, activation='linear', name='ZT')(x)
    main_out    = layers.Concatenate(name='main_output')([out_seebeck, out_cond, out_kappa, out_zt])

    # Aux heads -- S_log, sig_log, kappa_log -- unscaled log space
    out_S_log     = layers.Dense(1, activation='softplus', name='S_log_head')(x)
    out_sig_log   = layers.Dense(1, activation='softplus', name='sig_log_head')(x)
    out_kappa_log = layers.Dense(1, activation='softplus', name='kappa_log_head')(x)
    aux_out = layers.Concatenate(name='aux_output')([out_S_log, out_sig_log, out_kappa_log])

    model = keras.Model(
        inputs=[inp_s, inp_e, inp_t, inp_m],
        outputs=[main_out, aux_out],
        name=f'pinn_stage4_{DATASET}'
    )
    return model

# ============================================================================
# STEP 13: TRAINING / EVAL HELPERS (parameterised by lambdas)
# ============================================================================
EPOCHS = 300
UNIT_OFFSET_PF_T = tf.constant(UNIT_OFFSET_PF, dtype=tf.float32)
UNIT_OFFSET_ZT_T = tf.constant(UNIT_OFFSET_ZT, dtype=tf.float32)
UNIT_OFFSET_WF_T = tf.constant(UNIT_OFFSET_WF, dtype=tf.float32)
LOG10_T_T        = tf.constant(log10_T, dtype=tf.float32)

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
tr_zt = t(log_ZT_tr[:n_tr]);   vl_zt = t(log_ZT_tr[n_tr:])

nb = int(np.ceil(n_tr / BATCH_SIZE))


def make_train_step(model, optimizer, lam_aux, lam_pf, lam_zt, lam_wf):
    @tf.function
    def train_step(Xs, Xe, Xt, Xm, y_main_true, y_aux_true, log_pf_true, log_zt_true):
        with tf.GradientTape() as tape:
            main_pred, aux_pred = model([Xs, Xe, Xt, Xm], training=True)

            L_data = tf.reduce_mean(tf.square(main_pred - y_main_true))
            L_aux  = tf.reduce_mean(tf.square(aux_pred - y_aux_true))

            S_log_hat     = aux_pred[:, 0:1]
            sig_log_hat   = aux_pred[:, 1:2]
            kappa_log_hat = aux_pred[:, 2:3]

            # --- Constraint 1: Power Factor  PF = S^2 * sigma -------------
            PF_log_hat = 2.0 * S_log_hat + sig_log_hat - UNIT_OFFSET_PF_T
            L_PF = tf.reduce_mean(tf.square(PF_log_hat - log_pf_true))

            # --- Constraint 2: ZT consistency  ZT = S^2*sigma*T/kappa -----
            ZT_log_hat = (2.0 * S_log_hat + sig_log_hat - kappa_log_hat
                          + LOG10_T_T - UNIT_OFFSET_ZT_T)
            L_ZT = tf.reduce_mean(tf.square(ZT_log_hat - log_zt_true))

            # --- Constraint 3: Wiedemann-Franz  kappa_e/sigma = L*T -------
            WF_log_hat = sig_log_hat + UNIT_OFFSET_WF_T + LOG10_T_T
            L_WF = tf.reduce_mean(tf.square(WF_log_hat - kappa_log_hat))

            L_total = (L_data
                       + lam_aux * L_aux
                       + lam_pf  * L_PF
                       + lam_zt  * L_ZT
                       + lam_wf  * L_WF)

        grads = tape.gradient(L_total, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return L_total, L_data, L_aux, L_PF, L_ZT, L_WF
    return train_step


def make_val_step(model, lam_aux, lam_pf, lam_zt, lam_wf):
    @tf.function
    def val_step(Xs, Xe, Xt, Xm, y_main_true, y_aux_true, log_pf_true, log_zt_true):
        main_pred, aux_pred = model([Xs, Xe, Xt, Xm], training=False)

        L_data = tf.reduce_mean(tf.square(main_pred - y_main_true))
        L_aux  = tf.reduce_mean(tf.square(aux_pred - y_aux_true))

        S_log_hat     = aux_pred[:, 0:1]
        sig_log_hat   = aux_pred[:, 1:2]
        kappa_log_hat = aux_pred[:, 2:3]

        PF_log_hat = 2.0 * S_log_hat + sig_log_hat - UNIT_OFFSET_PF_T
        L_PF = tf.reduce_mean(tf.square(PF_log_hat - log_pf_true))

        ZT_log_hat = (2.0 * S_log_hat + sig_log_hat - kappa_log_hat
                      + LOG10_T_T - UNIT_OFFSET_ZT_T)
        L_ZT = tf.reduce_mean(tf.square(ZT_log_hat - log_zt_true))

        WF_log_hat = sig_log_hat + UNIT_OFFSET_WF_T + LOG10_T_T
        L_WF = tf.reduce_mean(tf.square(WF_log_hat - kappa_log_hat))

        L_total = (L_data
                   + lam_aux * L_aux
                   + lam_pf  * L_PF
                   + lam_zt  * L_ZT
                   + lam_wf  * L_WF)
        return L_total, L_data, L_aux, L_PF, L_ZT, L_WF
    return val_step


def train_model(lam_aux, lam_pf, lam_zt, lam_wf, seed=42, verbose=True, tag=""):
    model = build_model(seed=seed)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    train_step = make_train_step(model, optimizer, lam_aux, lam_pf, lam_zt, lam_wf)
    val_step   = make_val_step(model, lam_aux, lam_pf, lam_zt, lam_wf)

    best_val = np.inf; pat = 0; best_w = None
    hist = {'loss': [], 'val_loss': [], 'L_data': [], 'L_aux': [],
            'L_PF': [], 'L_ZT': [], 'L_WF': []}

    if verbose:
        print(f"\n{'='*70}")
        print(f"Training [{tag}]  lambda_AUX={lam_aux}  lambda_PF={lam_pf}  "
              f"lambda_ZT={lam_zt}  lambda_WF={lam_wf}")
        print(f"Dataset: {DATASET}  |  train rows: {n_tr}  |  val rows: {n_val}")
        print(f"{'='*70}")

    for ep in range(EPOCHS):
        perm = np.random.permutation(n_tr)
        ep_L, ep_Ld, ep_La, ep_Lpf, ep_Lzt, ep_Lwf = [], [], [], [], [], []

        for b in range(nb):
            bi = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            L, Ld, La, Lpf, Lzt, Lwf = train_step(
                tf.gather(tr_s, bi), tf.gather(tr_e, bi),
                tf.gather(tr_t, bi), tf.gather(tr_m, bi),
                tf.gather(tr_y, bi), tf.gather(tr_ya, bi),
                tf.gather(tr_pf, bi), tf.gather(tr_zt, bi)
            )
            ep_L.append(float(L));     ep_Ld.append(float(Ld))
            ep_La.append(float(La));   ep_Lpf.append(float(Lpf))
            ep_Lzt.append(float(Lzt)); ep_Lwf.append(float(Lwf))

        vL, vLd, vLa, vLpf, vLzt, vLwf = val_step(vl_s, vl_e, vl_t, vl_m, vl_y, vl_ya, vl_pf, vl_zt)
        vL = float(vL)

        hist['loss'].append(np.mean(ep_L))
        hist['val_loss'].append(vL)
        hist['L_data'].append(np.mean(ep_Ld))
        hist['L_aux'].append(np.mean(ep_La))
        hist['L_PF'].append(np.mean(ep_Lpf))
        hist['L_ZT'].append(np.mean(ep_Lzt))
        hist['L_WF'].append(np.mean(ep_Lwf))

        if verbose and ep % 20 == 0:
            print(f"Ep {ep:3d} | val={vL:.4f} | L_data={np.mean(ep_Ld):.4f} | "
                  f"L_aux={np.mean(ep_La):.4f} | L_PF={np.mean(ep_Lpf):.4f} | "
                  f"L_ZT={np.mean(ep_Lzt):.4f} | L_WF={np.mean(ep_Lwf):.4f}")

        if not np.isfinite(vL):
            if verbose:
                print(f"  WARNING: val loss is NaN at epoch {ep} -- stopping this run")
            break

        if vL < best_val - 1e-5:
            best_val = vL; pat = 0; best_w = model.get_weights()
        else:
            pat += 1
            if pat >= PATIENCE:
                if verbose:
                    print(f"\nEarly stopping at epoch {ep}")
                break

    if best_w is None:
        # Never improved (e.g. NaN'd immediately) -- keep current weights
        # rather than crashing on set_weights(None).
        if verbose:
            print("  WARNING: no improving checkpoint found -- "
                  "returning current (possibly NaN) weights")
        best_val = vL if 'vL' in dir() else best_val
    else:
        model.set_weights(best_w)
    if verbose:
        print(f"Best val loss: {best_val:.4f}")
    return model, hist, best_val


# ============================================================================
# STEP 14: EVALUATION HELPER
# ============================================================================
def evaluate_model(model, tag=""):
    print(f"\n[EVALUATION] {tag} | dataset={DATASET}")

    main_pred_scaled, aux_pred = model.predict(
        [X_s_te, X_e_te, X_t_te, X_m_te], verbose=0
    )
    y_pred = sc_y.inverse_transform(main_pred_scaled)

    results = {}

    print(f"\n{'Target':<22} {'MAE':<12} {'R2':<10} {'Note'}")
    print("-" * 70)
    for i, tgt in enumerate(targets_proc):
        mae = mean_absolute_error(y_te_raw[:, i], y_pred[:, i])
        r2  = r2_score(y_te_raw[:, i], y_pred[:, i])
        var = float(np.var(y_te_raw[:, i]))
        note = f"var={var:.5f} -- use Pearson" if var < 0.01 else ""
        print(f"{tgt:<22} {mae:<12.4f} {r2:<10.4f} {note}")
        results[f'{tgt}_mae'] = mae
        results[f'{tgt}_r2']  = r2

    r2_valid = np.mean([r2_score(y_te_raw[:, i], y_pred[:, i]) for i in range(3)])
    print(f"\n  Mean R2 (Seebeck, cond, kappa): {r2_valid:.4f}")
    results['mean_r2_3props'] = r2_valid

    zt_true = y_te_raw[:, 3]
    zt_pred = y_pred[:, 3]
    pr, _ = pearsonr(zt_true, zt_pred)
    sr, _ = spearmanr(zt_true, zt_pred)
    print(f"  ZT Pearson r:  {pr:.4f}")
    print(f"  ZT Spearman r: {sr:.4f}")
    results['zt_pearson']  = pr
    results['zt_spearman'] = sr

    print(f"\n[AUX HEAD EVALUATION] (log10 space)")
    print(f"{'Aux target':<20} {'MAE':<12} {'R2':<10}")
    print("-" * 44)
    for i, tgt in enumerate(aux_targets):
        mae = mean_absolute_error(y_aux_te[:, i], aux_pred[:, i])
        r2  = r2_score(y_aux_te[:, i], aux_pred[:, i])
        print(f"{tgt:<20} {mae:<12.4f} {r2:<10.4f}")
        results[f'{tgt}_aux_mae'] = mae
        results[f'{tgt}_aux_r2']  = r2

    S_log_pred     = aux_pred[:, 0]
    sig_log_pred   = aux_pred[:, 1]
    kappa_log_pred = aux_pred[:, 2]

    # --- PF physics check ---------------------------------------------------
    log_PF_true_te = log_PF_te.flatten()
    log_PF_pred = 2.0 * S_log_pred + sig_log_pred - UNIT_OFFSET_PF
    valid_pf = log_PF_true_te > 0.01
    if valid_pf.sum() > 10:
        pf_r2  = r2_score(log_PF_true_te[valid_pf], log_PF_pred[valid_pf])
        pf_mae = mean_absolute_error(log_PF_true_te[valid_pf], log_PF_pred[valid_pf])
        pf_bias = float(np.mean(log_PF_pred[valid_pf] - log_PF_true_te[valid_pf]))
        print(f"\n[Physics Check: PF]  2*S_log + sig_log - {UNIT_OFFSET_PF:.3f} vs log10(PF_true+1)")
        print(f"  R2   = {pf_r2:.4f}")
        print(f"  MAE  = {pf_mae:.4f} log10 units")
        print(f"  Bias = {pf_bias:+.4f} log10 units")
        results['pf_check_r2']  = pf_r2
        results['pf_check_mae'] = pf_mae
        results['pf_check_bias'] = pf_bias

    # --- ZT physics check ----------------------------------------------------
    log_ZT_true_te = log_ZT_te.flatten()
    log_ZT_pred = 2.0 * S_log_pred + sig_log_pred - kappa_log_pred + log10_T - UNIT_OFFSET_ZT
    valid_zt = log_ZT_true_te > 0.001
    if valid_zt.sum() > 10:
        zt_r2  = r2_score(log_ZT_true_te[valid_zt], log_ZT_pred[valid_zt])
        zt_mae = mean_absolute_error(log_ZT_true_te[valid_zt], log_ZT_pred[valid_zt])
        zt_bias = float(np.mean(log_ZT_pred[valid_zt] - log_ZT_true_te[valid_zt]))
        print(f"\n[Physics Check: ZT]  2*S_log + sig_log - kappa_log + log10(T) - {UNIT_OFFSET_ZT:.3f} vs log10(ZT_true+1)")
        print(f"  R2   = {zt_r2:.4f}")
        print(f"  MAE  = {zt_mae:.4f} log10 units")
        print(f"  Bias = {zt_bias:+.4f} log10 units")
        results['zt_check_r2']  = zt_r2
        results['zt_check_mae'] = zt_mae
        results['zt_check_bias'] = zt_bias

    # --- WF physics check -----------------------------------------------------
    log_WF_pred = sig_log_pred + UNIT_OFFSET_WF + log10_T
    wf_r2  = r2_score(kappa_log_pred, log_WF_pred)
    wf_mae = mean_absolute_error(kappa_log_pred, log_WF_pred)
    wf_bias = float(np.mean(log_WF_pred - kappa_log_pred))
    print(f"\n[Physics Check: WF]  sig_log + log10(L_eff={10**UNIT_OFFSET_WF:.3e}) + log10(T) vs kappa_log")
    print(f"  R2   = {wf_r2:.4f}")
    print(f"  MAE  = {wf_mae:.4f} log10 units")
    print(f"  Bias = {wf_bias:+.4f} log10 units")
    results['wf_check_r2']  = wf_r2
    results['wf_check_mae'] = wf_mae
    results['wf_check_bias'] = wf_bias

    return results


# ============================================================================
# STEP 15: TRAIN THE PRIMARY (FULL) MODEL
# ============================================================================
LAMBDA_AUX = 1.0
LAMBDA_PF  = 0.10
LAMBDA_ZT  = 1.00
LAMBDA_WF  = 0.05

model, hist, best_val = train_model(
    LAMBDA_AUX, LAMBDA_PF, LAMBDA_ZT, LAMBDA_WF,
    seed=42, verbose=True, tag="FULL PINN (PF+ZT+WF)"
)
results_full = evaluate_model(model, tag="FULL PINN (PF+ZT+WF)")

print(f"\n[Stage 4 complete | {DATASET}] Full PINN trained with PF + ZT + WF constraints.")

# ============================================================================
# STEP 16: LAMBDA SWEEP -- 4 CONFIGURATIONS
# ============================================================================
print(f"\n{'#'*80}")
print("# LAMBDA SWEEP -- comparing physics-constraint configurations")
print(f"{'#'*80}")

sweep_configs = [
    {"tag": "No physics (data only)", "lam_aux": 1.0, "lam_pf": 0.0,  "lam_zt": 0.0,  "lam_wf": 0.0},
    {"tag": "PF only (Stage 2)",       "lam_aux": 1.0, "lam_pf": 0.10, "lam_zt": 0.0,  "lam_wf": 0.0},
    {"tag": "PF + ZT (Stage 3)",       "lam_aux": 1.0, "lam_pf": 0.10, "lam_zt": 1.00, "lam_wf": 0.0},
    {"tag": "PF + ZT + WF (Stage 4, full)", "lam_aux": 1.0, "lam_pf": 0.10, "lam_zt": 1.00, "lam_wf": 0.05},
]

sweep_results = []
for cfg in sweep_configs:
    m, h, bv = train_model(
        cfg["lam_aux"], cfg["lam_pf"], cfg["lam_zt"], cfg["lam_wf"],
        seed=42, verbose=False, tag=cfg["tag"]
    )
    res = evaluate_model(m, tag=cfg["tag"])
    res['tag'] = cfg["tag"]
    res['lam_pf'] = cfg["lam_pf"]
    res['lam_zt'] = cfg["lam_zt"]
    res['lam_wf'] = cfg["lam_wf"]
    res['best_val'] = bv
    sweep_results.append(res)

# ============================================================================
# STEP 17: SWEEP SUMMARY TABLE
# ============================================================================
print(f"\n{'='*100}")
print("SWEEP SUMMARY")
print(f"{'='*100}")
header = (f"{'Config':<28} {'Mean R2':<10} {'ZT Pearson':<12} "
          f"{'PF check R2':<13} {'ZT check R2':<13} {'WF check R2':<13}")
print(header)
print("-" * 100)
for r in sweep_results:
    print(f"{r['tag']:<28} "
          f"{r.get('mean_r2_3props', float('nan')):<10.4f} "
          f"{r.get('zt_pearson', float('nan')):<12.4f} "
          f"{r.get('pf_check_r2', float('nan')):<13.4f} "
          f"{r.get('zt_check_r2', float('nan')):<13.4f} "
          f"{r.get('wf_check_r2', float('nan')):<13.4f}")

# ============================================================================
# STEP 18: PLOT SWEEP COMPARISON
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
tags = [r['tag'] for r in sweep_results]
short_tags = [t.split('(')[0].strip() for t in tags]

axes[0].bar(short_tags, [r.get('mean_r2_3props', 0) for r in sweep_results], color='steelblue')
axes[0].set_title('Mean R2 (Seebeck, cond, kappa)')
axes[0].set_ylim(0.8, 1.0)
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(short_tags, [r.get('zt_pearson', 0) for r in sweep_results], color='seagreen')
axes[1].set_title('ZT Pearson r')
axes[1].set_ylim(0.8, 1.0)
axes[1].tick_params(axis='x', rotation=30)

x = np.arange(len(short_tags))
width = 0.25
axes[2].bar(x - width, [r.get('pf_check_r2', 0) for r in sweep_results], width, label='PF check R2', color='indianred')
axes[2].bar(x,         [r.get('zt_check_r2', 0) for r in sweep_results], width, label='ZT check R2', color='goldenrod')
axes[2].bar(x + width, [r.get('wf_check_r2', 0) for r in sweep_results], width, label='WF check R2', color='mediumpurple')
axes[2].set_xticks(x)
axes[2].set_xticklabels(short_tags, rotation=30, ha='right')
axes[2].set_title('Physics Consistency Checks')
axes[2].legend()

plt.tight_layout()
plot_path = f"lambda_sweep_{DATASET}.png"
plt.savefig(plot_path, dpi=150)
print(f"\nSweep comparison plot saved to: {plot_path}")

print(f"\n[Stage 4 sweep complete | {DATASET}]")