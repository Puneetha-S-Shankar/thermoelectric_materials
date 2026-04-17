"""
ROBUST TRAINING WITH SPECIAL PREPROCESSING
Handles: extreme scales, zero values, negative values, outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("ROBUST TRAINING WITH SPECIAL PREPROCESSING")
print("=" * 100)

# ============================================================================
# STEP 1: LOAD
# ============================================================================
print("\n[STEP 1] Load data")
df = pd.read_csv('C:\\Users\\Puneetha\\thermoelectric_dataset\\output\\thermoelectric_ml_ready.csv', low_memory=False)
print(f"✓ Loaded: {len(df):,} materials")

# Keep only rows with all targets
targets = ['p-Seebeck', 'pcond', 'pkappa']
df_clean = df[df[targets].notna().all(axis=1)].copy()
print(f"✓ Rows with all targets: {len(df_clean):,}")

# ============================================================================
# STEP 2: HANDLE EXTREME TARGET VALUES
# ============================================================================
print("\n[STEP 2] Handle extreme target values")

for target in targets:
    vals = df_clean[target].values
    print(f"\n{target}:")
    print(f"  Original range: [{vals.min():.2e}, {vals.max():.2e}]")
    print(f"  Negative values: {(vals < 0).sum()}")
    print(f"  Zero values: {(vals == 0).sum()}")

# SOLUTION: Take absolute values (p-Seebeck negatives are fine in absolute value)
# and shift values so minimum is not zero
print("\n  Handling:")

df_clean['p-Seebeck_abs'] = np.abs(df_clean['p-Seebeck'])
print(f"  ✓ p-Seebeck: converted to absolute values")

# For conductivity and thermal cond: shift to avoid zero
# Add 1 to all values so min is at least 1
df_clean['pcond_shifted'] = df_clean['pcond'] + 1.0
df_clean['pkappa_shifted'] = df_clean['pkappa'] + 1.0
print(f"  ✓ pcond: shifted by +1 to avoid zeros")
print(f"  ✓ pkappa: shifted by +1 to avoid zeros")

# Take log to compress huge range
df_clean['pcond_log'] = np.log10(df_clean['pcond_shifted'])
df_clean['pkappa_log'] = np.log10(df_clean['pkappa_shifted'])
print(f"  ✓ Applied log10 to compress scales")

# Check new ranges
print(f"\nAfter processing:")
print(f"  p-Seebeck_abs:  [{df_clean['p-Seebeck_abs'].min():.2f}, {df_clean['p-Seebeck_abs'].max():.2f}]")
print(f"  pcond_log:      [{df_clean['pcond_log'].min():.2f}, {df_clean['pcond_log'].max():.2f}]")
print(f"  pkappa_log:     [{df_clean['pkappa_log'].min():.2f}, {df_clean['pkappa_log'].max():.2f}]")

# ============================================================================
# STEP 3: SELECT & CLEAN FEATURES
# ============================================================================
print("\n[STEP 3] Select features with >50% coverage")

# Features with good coverage
features_good = [
    'optb88vdw_bandgap',
    'avg_elec_mass',
    'avg_hole_mass',
    'ehull',
    'formation_energy_peratom',
    'optb88vdw_total_energy',
    'n-Seebeck',
    'n-powerfact',
    'ncond',
    'nkappa',
    'spg_number',
    'nat',
    'density'
]

features_available = []
for feat in features_good:
    if feat in df_clean.columns:
        coverage = df_clean[feat].notna().sum() / len(df_clean)
        if coverage > 0.5:
            features_available.append(feat)

print(f"✓ Selected {len(features_available)} features")

# ============================================================================
# STEP 4: HANDLE FEATURE OUTLIERS
# ============================================================================
print("\n[STEP 4] Handle feature outliers")

df_features = df_clean[features_available].copy()

for feat in features_available:
    vals = df_features[feat].dropna()
    
    # Use IQR method to detect outliers
    Q1 = vals.quantile(0.25)
    Q3 = vals.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR  # 3x IQR threshold
    upper_bound = Q3 + 3 * IQR
    
    outliers = ((vals < lower_bound) | (vals > upper_bound)).sum()
    
    if outliers > 0:
        print(f"  {feat:<30} {outliers:>4,} outliers ({100*outliers/len(vals):>5.1f}%)")
        
        # Clip outliers instead of removing
        df_features[feat] = df_features[feat].clip(lower=lower_bound, upper=upper_bound)

print(f"  ✓ Outliers clipped to 3x IQR bounds")

# ============================================================================
# STEP 5: IMPUTE MISSING VALUES
# ============================================================================
print("\n[STEP 5] Impute missing values (use median)")

for feat in features_available:
    median = df_features[feat].median()
    if pd.isna(median):
        median = 0
    df_features[feat].fillna(median, inplace=True)

df_features = df_features.dropna()
print(f"✓ Missing values imputed and cleaned")

# ============================================================================
# STEP 6: PREPARE TRAINING DATA
# ============================================================================
print("\n[STEP 6] Prepare training data")

# Filter targets to match features
targets_processed = ['p-Seebeck_abs', 'pcond_log', 'pkappa_log']
df_processed = df_clean.loc[df_features.index, targets_processed].copy()

# Make sure no NaN
df_processed = df_processed.dropna()
df_features = df_features.loc[df_processed.index]

print(f"✓ Final dataset: {len(df_features):,} materials")

X = df_features.values
y = df_processed.values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale using RobustScaler (handles outliers better)
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print(f"\nScaled data:")
print(f"  X_train: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
print(f"  y_train: mean={y_train_scaled.mean():.4f}, std={y_train_scaled.std():.4f}")

# ============================================================================
# STEP 7: BUILD MODEL
# ============================================================================
print("\n[STEP 7] Build neural network")

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    
    layers.Dense(3)  # 3 outputs
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("Architecture: 128 → 64 → 32 → 3")

# ============================================================================
# STEP 8: TRAIN
# ============================================================================
print("\n[STEP 8] Train model")

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)
    ],
    verbose=1
)

# ============================================================================
# STEP 9: EVALUATE
# ============================================================================
print("\n" + "=" * 100)
print("[STEP 9] RESULTS")
print("=" * 100)

y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate in PROCESSED space
mae_processed = mean_absolute_error(y_test, y_pred)
rmse_processed = np.sqrt(mean_squared_error(y_test, y_pred))
r2_processed = r2_score(y_test, y_pred)

print(f"\nIn processed space (log scale):")
print(f"  p-Seebeck_abs: R² = {r2_score(y_test[:, 0], y_pred[:, 0]):.4f}")
print(f"  pcond_log:     R² = {r2_score(y_test[:, 1], y_pred[:, 1]):.4f}")
print(f"  pkappa_log:    R² = {r2_score(y_test[:, 2], y_pred[:, 2]):.4f}")
print(f"\n  Overall MAE: {mae_processed:.4f}")
print(f"  Overall R²:  {r2_processed:.4f}")

# ============================================================================
# STEP 10: VISUALIZE
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

target_names = ['p-Seebeck (abs)', 'pcond (log10)', 'pkappa (log10)']
colors = ['#0084d1', '#ff8f00', '#00c853']

# Row 1: Predictions
for idx in range(3):
    ax = axes[0, idx]
    y_true = y_test[:, idx]
    y_pred_col = y_pred[:, idx]
    r2 = r2_score(y_true, y_pred_col)
    
    ax.scatter(y_true, y_pred_col, alpha=0.5, s=20, color=colors[idx])
    min_v = min(y_true.min(), y_pred_col.min())
    max_v = max(y_true.max(), y_pred_col.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2)
    
    ax.set_xlabel('Actual', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=10, fontweight='bold')
    ax.set_title(f'{target_names[idx]}\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Row 2: Residuals
for idx in range(3):
    ax = axes[1, idx]
    y_true = y_test[:, idx]
    y_pred_col = y_pred[:, idx]
    residuals = y_true - y_pred_col
    
    ax.scatter(y_pred_col, residuals, alpha=0.5, s=20, color=colors[idx])
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=10, fontweight='bold')
    ax.set_title(f'{target_names[idx]} - Residuals', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('robust_training_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: robust_training_results.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

summary = f"""
ROBUST PREPROCESSING & TRAINING

Problems Fixed:
  1. ✓ Extreme scales: Applied log10 to compress ranges
  2. ✓ Negative values: Converted p-Seebeck to absolute value
  3. ✓ Zero values: Shifted by +1 before log
  4. ✓ Feature outliers: Clipped to 3x IQR bounds
  5. ✓ Missing features: Used RobustScaler (handles outliers)

Dataset: {len(df_features):,} materials
Features: {len(features_available)} reliable features

Model:
  - 4 hidden layers with BatchNorm & Dropout
  - Multi-output (3 targets simultaneously)
  - RobustScaler for preprocessing
  - Early stopping + learning rate reduction

Results (in processed space):
  p-Seebeck_abs: R² = {r2_score(y_test[:, 0], y_pred[:, 0]):.4f}
  pcond_log:     R² = {r2_score(y_test[:, 1], y_pred[:, 1]):.4f}
  pkappa_log:    R² = {r2_score(y_test[:, 2], y_pred[:, 2]):.4f}
  
  Overall: R² = {r2_processed:.4f}, MAE = {mae_processed:.4f}

Interpretation:
  - R² > 0.5: Model learning well
  - 0.2 < R² < 0.5: Moderate performance
  - R² < 0.2: Poor learning
"""

print(summary)

with open('robust_training_summary.txt', 'w') as f:
    f.write(summary)

print("=" * 100)
print("✓ TRAINING COMPLETE")
print("=" * 100)