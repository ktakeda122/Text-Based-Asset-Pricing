"""Part 3(a): Embedding-based prediction of mret using Lasso (k=5) + OLS.

1. Load articles_with_embeddings.parquet
2. Aggregate embeddings to monthly mean (element-wise average of 250-dim vectors)
3. Merge with macro.csv on date (align with mret)
4. Lasso path → select alpha for k=5 non-zero coefficients → OLS
5. Print R2 and compare to topic-based models
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

print("=" * 70)
print("  Part 3(a): Embeddings → Lasso (k=5) → OLS on mret")
print("=" * 70)

# -- 1. Load embeddings ---------------------------------------------------
df_emb = pd.read_parquet('articles_with_embeddings.parquet')
print(f"Loaded {len(df_emb)} articles with embeddings")

# Extract embeddings into a numpy array
embeddings = np.array(df_emb['embedding'].tolist())
print(f"Embedding shape: {embeddings.shape}")

# -- 2. Aggregate to monthly mean -----------------------------------------
df_emb['date'] = pd.to_datetime(df_emb['display_date']).dt.to_period('M').dt.to_timestamp()

emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=emb_cols)
emb_df['date'] = df_emb['date'].values

# Element-wise mean per month
emb_monthly = emb_df.groupby('date').mean().reset_index()
print(f"Monthly aggregation: {emb_monthly.shape[0]} months, {len(emb_cols)} embedding dimensions")

# -- 3. Merge with macro --------------------------------------------------
macro = pd.read_csv('macro.csv', parse_dates=['date'])
df = pd.merge(macro[['date', 'mret']], emb_monthly, on='date').sort_values('date').reset_index(drop=True)
print(f"Merged dataset: {df.shape[0]} months")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

X = df[emb_cols].values.astype(np.float64)
y = df['mret'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -- 4. Lasso path → select k=5 → OLS ------------------------------------
alphas_grid = np.logspace(-6, 1, 200)
alphas_out, coefs, _ = lasso_path(X_scaled, y, alphas=alphas_grid, max_iter=50000)
n_nonzero = np.sum(coefs != 0, axis=0)

# Select alpha yielding exactly 5 non-zero coefficients
target_k = 5
mask_k = n_nonzero == target_k
if mask_k.any():
    alpha_sel = alphas_out[mask_k].min()
    idx_sel = np.where(alphas_out == alpha_sel)[0][0]
    print(f"\nFound alpha with exactly k={target_k} non-zero coefficients")
else:
    idx_sel = np.argmin(np.abs(n_nonzero - target_k))
    alpha_sel = alphas_out[idx_sel]
    print(f"\nClosest k={n_nonzero[idx_sel]} (target was {target_k})")

selected_mask = coefs[:, idx_sel] != 0
selected_cols = [emb_cols[j] for j in range(len(emb_cols)) if selected_mask[j]]
print(f"Alpha = {alpha_sel:.6f}, selected {len(selected_cols)} embedding dimensions")

# OLS with selected features
X_ols = sm.add_constant(df[selected_cols])
ols = sm.OLS(y, X_ols).fit()

# -- 5. Results and comparison --------------------------------------------
print(f"\n{'='*70}")
print(f"  RESULTS: Embeddings Lasso (k=5) + OLS on mret")
print(f"{'='*70}")
print(f"  Alpha:    {alpha_sel:.6f}")
print(f"  R2:       {ols.rsquared:.4f}")
print(f"  Adj R2:   {ols.rsquared_adj:.4f}")
print(f"  F-stat:   {ols.fvalue:.2f} (p={ols.f_pvalue:.4f})")
print(f"  N:        {ols.nobs:.0f}")
print(f"")
print(f"  Selected embedding dimensions ({len(selected_cols)}):")
for c in selected_cols:
    sig = '***' if ols.pvalues[c] < 0.01 else '**' if ols.pvalues[c] < 0.05 else '*' if ols.pvalues[c] < 0.1 else ''
    print(f"    {c:10s}  coef={ols.params[c]:+12.6f}  p={ols.pvalues[c]:.4f} {sig}")

print(f"\n{'='*70}")
print(f"  COMPARISON TO PRIOR MODELS (all Lasso k=5 + OLS on mret)")
print(f"{'='*70}")
print(f"  Part 1(a) Pre-built Topics (180 features):   R2 = 0.1079")
print(f"  Part 1(e) Raw Word Counts (18,093 features): R2 = 0.1971")
print(f"  Part 2(a) LLM-Gen Topics (2,025 features):   R2 = 0.1302")
print(f"  Part 3(a) Embeddings (250 dimensions):        R2 = {ols.rsquared:.4f}")
print(f"{'='*70}")

# Also try different k values for context
print(f"\n  Embedding R2 across different k values:")
for k in [3, 5, 7, 10, 15, 20]:
    mk = n_nonzero == k
    if mk.any():
        a_k = alphas_out[mk].min()
        i_k = np.where(alphas_out == a_k)[0][0]
    else:
        i_k = np.argmin(np.abs(n_nonzero - k))
    sel_k = coefs[:, i_k] != 0
    sel_cols_k = [emb_cols[j] for j in range(len(emb_cols)) if sel_k[j]]
    if sel_cols_k:
        ols_k = sm.OLS(y, sm.add_constant(df[sel_cols_k])).fit()
        actual_k = sum(sel_k)
        print(f"    k={actual_k:3d}: R2 = {ols_k.rsquared:.4f}, Adj R2 = {ols_k.rsquared_adj:.4f}")

print(f"\n  Full OLS summary for k=5:")
print(ols.summary())
