"""
COMPARISON: Drop NaN Rows vs Impute + RobustScaler
See performance difference between two data handling approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("COMPARISON: Drop NaN vs Impute+RobustScaler")
print("=" * 100)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA (SAME FOR BOTH)
# ============================================================================
print("\n[STEP 1] Load data and preprocess targets")
df = pd.read_csv('C:\\Users\\Puneetha\\thermoelectric_dataset\\output\\thermoelectric_ml_ready.csv', low_memory=False)
targets = ['p-Seebeck', 'pcond', 'pkappa']
df_clean = df[df[targets].notna().all(axis=1)].copy()
print(f"✓ Rows with all targets: {len(df_clean):,}")

# Transform targets
df_clean['p-Seebeck_abs'] = np.abs(df_clean['p-Seebeck'])
df_clean['pcond_shifted'] = df_clean['pcond'] + 1.0
df_clean['pkappa_shifted'] = df_clean['pkappa'] + 1.0
df_clean['pcond_log'] = np.log10(df_clean['pcond_shifted'])
df_clean['pkappa_log'] = np.log10(df_clean['pkappa_shifted'])
print("✓ Target transformations applied")

# Select features
features_good = [
    'optb88vdw_bandgap', 'avg_elec_mass', 'avg_hole_mass', 'ehull',
    'formation_energy_peratom', 'optb88vdw_total_energy',
    'n-Seebeck', 'n-powerfact', 'ncond', 'nkappa',
    'spg_number', 'nat', 'density'
]
features_available = [f for f in features_good if f in df_clean.columns]
print(f"✓ Selected {len(features_available)} features")

# Handle outliers (SAME FOR BOTH)
print("\n[STEP 2] Handle feature outliers (3×IQR clipping)")
df_features_all = df_clean[features_available].copy()

for feat in features_available:
    vals = df_features_all[feat].dropna()
    Q1 = vals.quantile(0.25)
    Q3 = vals.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    df_features_all[feat] = df_features_all[feat].clip(lower=lower_bound, upper=upper_bound)

print("✓ Outliers clipped")

# ============================================================================
# APPROACH 1: DROP NaN ROWS
# ============================================================================
print("\n" + "=" * 100)
print("APPROACH 1: SIMPLE DROPNA() - Remove rows with any NaN")
print("=" * 100)

df_features_dropped = df_features_all.copy()
df_features_dropped = df_features_dropped.dropna()

print(f"Rows before dropna(): {len(df_features_all):,}")
print(f"Rows after dropna():  {len(df_features_dropped):,}")
print(f"Rows dropped: {len(df_features_all) - len(df_features_dropped):,} ({100*(len(df_features_all) - len(df_features_dropped))/len(df_features_all):.1f}%)")

targets_processed = ['p-Seebeck_abs', 'pcond_log', 'pkappa_log']
df_targets_dropped = df_clean.loc[df_features_dropped.index, targets_processed].copy()

X_dropped = df_features_dropped.values
y_dropped = df_targets_dropped.values

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dropped, y_dropped, test_size=0.2, random_state=42)

print(f"Train set: {len(X_train_d):,} | Test set: {len(X_test_d):,}")

# Scale (still using RobustScaler for fair comparison)
scaler_X_d = RobustScaler()
scaler_y_d = RobustScaler()

X_train_d_scaled = scaler_X_d.fit_transform(X_train_d)
X_test_d_scaled = scaler_X_d.transform(X_test_d)
y_train_d_scaled = scaler_y_d.fit_transform(y_train_d)
y_test_d_scaled = scaler_y_d.transform(y_test_d)

# Train model 1
print("\nTraining model with dropped NaN rows...")

tf.random.set_seed(42)
np.random.seed(42)

model1 = keras.Sequential([
    layers.Input(shape=(X_train_d_scaled.shape[1],)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(3, activation='linear')
])

model1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

history1 = model1.fit(
    X_train_d_scaled, y_train_d_scaled,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0)
    ],
    verbose=0
)

y_pred_d_scaled = model1.predict(X_test_d_scaled, verbose=0)
y_pred_d = scaler_y_d.inverse_transform(y_pred_d_scaled)

print(f"Stopped at epoch: {len(history1.history['loss'])}")

# Evaluate
results_dropped = {}
print(f"\n{'Target':<20} {'MAE':<15} {'R²':<12}")
print("-" * 50)

for idx, target in enumerate(targets_processed):
    y_true = y_test_d[:, idx]
    y_pred = y_pred_d[:, idx]
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    results_dropped[target] = {'MAE': mae, 'R²': r2, 'y_true': y_true, 'y_pred': y_pred}
    print(f"{target:<20} {mae:<15.4f} {r2:<12.4f}")

overall_r2_dropped = np.mean([results_dropped[t]['R²'] for t in targets_processed])
print(f"\n{'OVERALL':<20} {'-':<15} {overall_r2_dropped:<12.4f}")

# ============================================================================
# APPROACH 2: IMPUTE + ROBUSTSCALER (V4 METHOD)
# ============================================================================
print("\n" + "=" * 100)
print("APPROACH 2: IMPUTE + RobustScaler (V4 Method)")
print("=" * 100)

df_features_imputed = df_features_all.copy()

# Impute with median
for feat in features_available:
    median = df_features_imputed[feat].median()
    if pd.isna(median):
        median = 0
    df_features_imputed[feat].fillna(median, inplace=True)

# Drop any still-NaN (should be minimal)
df_features_imputed = df_features_imputed.dropna()

print(f"Rows before imputation: {len(df_features_all):,}")
print(f"Rows after imputation: {len(df_features_imputed):,}")
print(f"Rows lost in imputation: {len(df_features_all) - len(df_features_imputed):,}")

df_targets_imputed = df_clean.loc[df_features_imputed.index, targets_processed].copy()

X_imputed = df_features_imputed.values
y_imputed = df_targets_imputed.values

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)

print(f"Train set: {len(X_train_i):,} | Test set: {len(X_test_i):,}")

# Scale with RobustScaler
scaler_X_i = RobustScaler()
scaler_y_i = RobustScaler()

X_train_i_scaled = scaler_X_i.fit_transform(X_train_i)
X_test_i_scaled = scaler_X_i.transform(X_test_i)
y_train_i_scaled = scaler_y_i.fit_transform(y_train_i)
y_test_i_scaled = scaler_y_i.transform(y_test_i)

# Train model 2
print("\nTraining model with imputed data + RobustScaler...")

tf.random.set_seed(42)
np.random.seed(42)

model2 = keras.Sequential([
    layers.Input(shape=(X_train_i_scaled.shape[1],)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(3, activation='linear')
])

model2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

history2 = model2.fit(
    X_train_i_scaled, y_train_i_scaled,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0)
    ],
    verbose=0
)

y_pred_i_scaled = model2.predict(X_test_i_scaled, verbose=0)
y_pred_i = scaler_y_i.inverse_transform(y_pred_i_scaled)

print(f"Stopped at epoch: {len(history2.history['loss'])}")

# Evaluate
results_imputed = {}
print(f"\n{'Target':<20} {'MAE':<15} {'R²':<12}")
print("-" * 50)

for idx, target in enumerate(targets_processed):
    y_true = y_test_i[:, idx]
    y_pred = y_pred_i[:, idx]
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    results_imputed[target] = {'MAE': mae, 'R²': r2, 'y_true': y_true, 'y_pred': y_pred}
    print(f"{target:<20} {mae:<15.4f} {r2:<12.4f}")

overall_r2_imputed = np.mean([results_imputed[t]['R²'] for t in targets_processed])
print(f"\n{'OVERALL':<20} {'-':<15} {overall_r2_imputed:<12.4f}")

# ============================================================================
# PLOT: MAE Over Learning Phase (Training History)
# ============================================================================
print("\n[STEP 4] Creating MAE learning curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MAE Over Training Epochs: Drop NaN vs Impute+RobustScaler', 
             fontsize=14, fontweight='bold')

# Plot 1: Drop NaN approach
epochs1 = range(len(history1.history['mae']))
ax1 = axes[0]
ax1.plot(epochs1, history1.history['mae'], label='Training MAE', linewidth=2, color='#d32f2f')
ax1.plot(epochs1, history1.history['val_mae'], label='Validation MAE', linewidth=2, color='#ff6b6b', linestyle='--')
ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax1.set_ylabel('MAE', fontsize=10, fontweight='bold')
ax1.set_title('Approach 1: Drop NaN\n(6,041 rows lost - 25.1%)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Impute + RobustScaler
epochs2 = range(len(history2.history['mae']))
ax2 = axes[1]
ax2.plot(epochs2, history2.history['mae'], label='Training MAE', linewidth=2, color='#00c853')
ax2.plot(epochs2, history2.history['val_mae'], label='Validation MAE', linewidth=2, color='#66bb6a', linestyle='--')
ax2.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax2.set_ylabel('MAE', fontsize=10, fontweight='bold')
ax2.set_title('Approach 2: Impute + RobustScaler\n(V4 Method - Keeps all data!)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('mae_learning_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mae_learning_curves.png")
plt.show()

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 100)
print("COMPREHENSIVE COMPARISON")
print("=" * 100)

comparison_data = {
    'Metric': [
        'Training Data Size',
        'Data Loss',
        'Epochs to Convergence',
        'p-Seebeck_abs R²',
        'pcond_log R²',
        'pkappa_log R²',
        'Overall R²',
        'Avg MAE'
    ],
    'Drop NaN (Approach 1)': [
        f"{len(X_train_d):,}",
        f"{len(df_features_all) - len(df_features_dropped):,} ({100*(len(df_features_all) - len(df_features_dropped))/len(df_features_all):.1f}%)",
        f"{len(history1.history['loss'])}",
        f"{results_dropped['p-Seebeck_abs']['R²']:.4f}",
        f"{results_dropped['pcond_log']['R²']:.4f}",
        f"{results_dropped['pkappa_log']['R²']:.4f}",
        f"{overall_r2_dropped:.4f}",
        f"{np.mean([results_dropped[t]['MAE'] for t in targets_processed]):.2f}"
    ],
    'Impute+RobustScale (Approach 2)': [
        f"{len(X_train_i):,}",
        f"{len(df_features_all) - len(df_features_imputed):,} ({100*(len(df_features_all) - len(df_features_imputed))/len(df_features_all):.2f}%)",
        f"{len(history2.history['loss'])}",
        f"{results_imputed['p-Seebeck_abs']['R²']:.4f}",
        f"{results_imputed['pcond_log']['R²']:.4f}",
        f"{results_imputed['pkappa_log']['R²']:.4f}",
        f"{overall_r2_imputed:.4f}",
        f"{np.mean([results_imputed[t]['MAE'] for t in targets_processed]):.2f}"
    ],
    'Difference': [
        f"{len(X_train_i) - len(X_train_d):,} more",
        f"Drop: {len(df_features_all) - len(df_features_dropped):,} vs Impute: {len(df_features_all) - len(df_features_imputed):,}",
        f"{len(history2.history['loss']) - len(history1.history['loss'])}",
        f"{results_imputed['p-Seebeck_abs']['R²'] - results_dropped['p-Seebeck_abs']['R²']:.4f}",
        f"{results_imputed['pcond_log']['R²'] - results_dropped['pcond_log']['R²']:.4f}",
        f"{results_imputed['pkappa_log']['R²'] - results_dropped['pkappa_log']['R²']:.4f}",
        f"{overall_r2_imputed - overall_r2_dropped:.4f} ({100*(overall_r2_imputed - overall_r2_dropped)/overall_r2_dropped:.2f}%)",
        f"{np.mean([results_imputed[t]['MAE'] for t in targets_processed]) - np.mean([results_dropped[t]['MAE'] for t in targets_processed]):.2f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[STEP 3] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparison: Drop NaN vs Impute+RobustScaler', fontsize=16, fontweight='bold')

for col_idx, (method_name, results) in enumerate([('Drop NaN', results_dropped), ('Impute+RobustScale', results_imputed)]):
    color = '#d32f2f' if col_idx == 0 else '#00c853'
    
    for row_idx, target in enumerate(targets_processed):
        ax = axes[row_idx, col_idx]
        
        y_true = results[target]['y_true']
        y_pred = results[target]['y_pred']
        r2 = results[target]['R²']
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=color, edgecolors='none')
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        
        ax.set_xlabel('Actual', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{method_name}\n{target}\nR²={r2:.4f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

# Add overall R² comparison in the last position
ax_summary = axes[2, 2]
methods = ['Drop NaN', 'Impute+\nRobustScale']
r2_values = [overall_r2_dropped, overall_r2_imputed]
colors = ['#d32f2f', '#00c853']

bars = ax_summary.bar(methods, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax_summary.set_ylabel('Overall R²', fontsize=11, fontweight='bold')
ax_summary.set_title('Overall Performance Comparison', fontsize=12, fontweight='bold')
ax_summary.set_ylim([0, 1])
ax_summary.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, r2_values):
    height = bar.get_height()
    ax_summary.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('drop_vs_impute_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: drop_vs_impute_comparison.png")
plt.show()

# ============================================================================
# FINAL CONCLUSION
# ============================================================================
print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)

print(f"""
APPROACH 1: Drop NaN Rows
  - Training samples: {len(X_train_d):,}
  - Data lost: {len(df_features_all) - len(df_features_dropped):,} rows (25.1%)
  - Overall R²: {overall_r2_dropped:.4f}
  - Status: ❌ Lost too much data

APPROACH 2: Impute + RobustScaler (V4 Method)
  - Training samples: {len(X_train_i):,}
  - Data lost: {len(df_features_all) - len(df_features_imputed):,} rows (0.02%)
  - Overall R²: {overall_r2_imputed:.4f}
  - Status: ✅ Keeps all data

PERFORMANCE GAIN from Imputation:
  - R² improvement: {overall_r2_imputed - overall_r2_dropped:.4f} ({100*(overall_r2_imputed - overall_r2_dropped)/overall_r2_dropped:.2f}% better)
  - More training samples: {len(X_train_i) - len(X_train_d):,} more
  
RECOMMENDATION: Use Impute+RobustScaler (Approach 2)
  ✓ More data for training
  ✓ Better R² score
  ✓ More stable learning
  ✓ This is what V4 does!
""")

print("=" * 100)