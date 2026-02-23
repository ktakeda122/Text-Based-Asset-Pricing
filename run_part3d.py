"""Part 3(d): OOS Forecasting of Industrial Production Growth using Embeddings.
Expanding-window Ridge regression with GCV for alpha selection.
Target: indprol1 (1-month-ahead industrial production growth).
Features: 250-dimensional monthly mean embeddings.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import time
import warnings
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

t0 = time.time()

print("=" * 70)
print("  Part 3(d): OOS Forecasting of Industrial Production Growth")
print("  Features: 250-dim Monthly Mean Embeddings")
print("=" * 70)

# -- 1. Build monthly embedding features ----------------------------------
df_emb = pd.read_parquet('articles_with_embeddings.parquet')
embeddings = np.array(df_emb['embedding'].tolist())
df_emb['date'] = pd.to_datetime(df_emb['display_date']).dt.to_period('M').dt.to_timestamp()

emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=emb_cols)
emb_df['date'] = df_emb['date'].values
emb_monthly = emb_df.groupby('date').mean().reset_index()

print(f"Embedding features: {len(emb_cols)} dims, {emb_monthly.shape[0]} months")

# -- 2. Merge with macro --------------------------------------------------
macro = pd.read_csv('macro.csv', parse_dates=['date'])
df = pd.merge(macro[['date', 'indprol1']], emb_monthly, on='date').sort_values('date').reset_index(drop=True)
df = df[df['indprol1'].notna()].reset_index(drop=True)

X_all = df[emb_cols].values.astype(np.float64)
y_all = df['indprol1'].values
dates_final = df['date'].values

n_total = len(y_all)
print(f"Dataset: {n_total} months, {len(emb_cols)} embedding features")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Target: indprol1 (mean={y_all.mean():.6f}, std={y_all.std():.6f})")

# -- 3. Expanding-window Ridge forecast ------------------------------------
min_train = int(n_total * 0.50)
n_test = n_total - min_train

print(f"\nExpanding window setup:")
print(f"  Initial training: {min_train} months ({pd.Timestamp(dates_final[0]).date()} to {pd.Timestamp(dates_final[min_train-1]).date()})")
print(f"  Test period:      {n_test} months ({pd.Timestamp(dates_final[min_train]).date()} to {pd.Timestamp(dates_final[-1]).date()})")

alphas = np.logspace(0, 6, 15)

y_pred_oos = np.full(n_test, np.nan)
selected_alphas = []

for i in range(n_test):
    t = min_train + i

    # Standardize using only training data (no look-ahead)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_all[:t])
    X_test_sc = scaler.transform(X_all[t:t+1])

    # Ridge with efficient leave-one-out GCV
    ridge = RidgeCV(alphas=alphas, gcv_mode='svd')
    ridge.fit(X_train_sc, y_all[:t])
    y_pred_oos[i] = ridge.predict(X_test_sc)[0]
    selected_alphas.append(ridge.alpha_)

    if (i + 1) % 25 == 0 or i == 0 or i == n_test - 1:
        running_r2 = r2_score(y_all[min_train:min_train+i+1], y_pred_oos[:i+1]) if i > 0 else float('nan')
        elapsed = time.time() - t0
        print(f"  Month {i+1:3d}/{n_test}: running OOS R2 = {running_r2:+.4f}, alpha = {ridge.alpha_:.0f}, elapsed = {elapsed:.1f}s", flush=True)

# -- 4. Final Results ------------------------------------------------------
y_actual = y_all[min_train:]
oos_r2 = r2_score(y_actual, y_pred_oos)
oos_mse = np.mean((y_actual - y_pred_oos) ** 2)

# Historical mean baseline
hist_mean_pred = np.array([y_all[:min_train+i].mean() for i in range(n_test)])
r2_hist_mean = r2_score(y_actual, hist_mean_pred)

elapsed_total = time.time() - t0

print(f"\n{'='*70}")
print(f"  RESULTS: OOS Forecasting of indprol1")
print(f"{'='*70}")
print(f"  Model:          Ridge Regression with LOO-GCV alpha selection")
print(f"  Features:       {len(emb_cols)} embedding dimensions (monthly mean)")
print(f"  Training start: {pd.Timestamp(dates_final[0]).date()}")
print(f"  Test period:    {pd.Timestamp(dates_final[min_train]).date()} to {pd.Timestamp(dates_final[-1]).date()} ({n_test} months)")
print(f"  Runtime:        {elapsed_total:.1f}s")
print(f"")
print(f"  OOS R-squared (Embeddings):       {oos_r2:+.4f}")
print(f"  OOS R-squared (LLM Topics, 2c):   -0.0478")
print(f"  OOS R-squared (Historical Mean):   {r2_hist_mean:+.4f}")
print(f"  OOS MSE:                           {oos_mse:.8f}")
print(f"")
print(f"  Median Ridge alpha selected: {np.median(selected_alphas):.0f}")
print(f"{'='*70}")
