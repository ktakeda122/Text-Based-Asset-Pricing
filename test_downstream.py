"""Test downstream pipeline (Steps 3-4) on 500-article checkpoint."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# ── Step 3: Monthly Aggregation ──────────────────────────────────────────────
print("=" * 70, flush=True)
print("  STEP 3: Monthly Aggregation", flush=True)
print("=" * 70, flush=True)

df_llm = pd.read_parquet('articles_with_topics.parquet')
print(f"Loaded {len(df_llm)} articles with LLM topics", flush=True)
print(f"Sample topics:", flush=True)
for i in range(5):
    print(f"  {df_llm['generated_topics'].iloc[i]}", flush=True)

empties = (df_llm['generated_topics'] == '').sum()
print(f"Empty topics: {empties}/{len(df_llm)}", flush=True)

# Year-month column
df_llm['date'] = pd.to_datetime(df_llm['display_date']).dt.to_period('M').dt.to_timestamp()

# Build DTM from LLM topics
llm_vec = CountVectorizer(stop_words='english')
llm_dtm = llm_vec.fit_transform(df_llm['generated_topics'])
llm_feat_names = llm_vec.get_feature_names_out()
llm_cols = ['llm_' + f for f in llm_feat_names]

print(f"\nLLM topic DTM: {llm_dtm.shape[0]} docs x {llm_dtm.shape[1]} unique terms", flush=True)

# Aggregate to monthly
llm_dtm_df = pd.DataFrame(llm_dtm.toarray(), columns=llm_cols)
llm_dtm_df['date'] = df_llm['date'].values
llm_monthly = llm_dtm_df.groupby('date').sum().reset_index()

# Merge with macro
macro = pd.read_csv('macro.csv', parse_dates=['date'])
df_2a = pd.merge(macro, llm_monthly, on='date')

print(f"Monthly LLM features: {llm_monthly.shape[0]} months x {len(llm_cols)} terms", flush=True)
print(f"Merged with macro: {df_2a.shape[0]} observations", flush=True)
print(f"Date range: {df_2a['date'].min().date()} to {df_2a['date'].max().date()}", flush=True)

# ── Step 4: Lasso (k=5) + OLS on mret ───────────────────────────────────────
print(f"\n{'=' * 70}", flush=True)
print("  STEP 4: Lasso (k=5) + OLS on mret", flush=True)
print("=" * 70, flush=True)

try:
    X_llm = df_2a[llm_cols].values.astype(float)
    scaler_llm = StandardScaler()
    X_llm_scaled = scaler_llm.fit_transform(X_llm)
    y_mret = df_2a['mret'].values

    print(f"X shape: {X_llm_scaled.shape}, y shape: {y_mret.shape}", flush=True)

    alphas_grid = np.logspace(-6, 1, 200)
    alphas_out, coefs, _ = lasso_path(X_llm_scaled, y_mret, alphas=alphas_grid, max_iter=50000)
    n_nonzero = np.sum(coefs != 0, axis=0)

    print(f"Lasso path: {len(alphas_out)} alphas, max nonzero = {n_nonzero.max()}", flush=True)

    # Select alpha for k=5
    mask_5 = n_nonzero == 5
    if mask_5.any():
        alpha_sel = alphas_out[mask_5].min()
        idx_sel = np.where(alphas_out == alpha_sel)[0][0]
    else:
        idx_sel = np.argmin(np.abs(n_nonzero - 5))
        alpha_sel = alphas_out[idx_sel]
        print(f"  (No exact k=5 found; closest k={n_nonzero[idx_sel]})", flush=True)

    selected_mask = coefs[:, idx_sel] != 0
    selected_topics = [llm_cols[j] for j in range(len(llm_cols)) if selected_mask[j]]

    if not selected_topics:
        print("  No features selected — insufficient data for regression.", flush=True)
    else:
        X_ols = sm.add_constant(df_2a[selected_topics])
        ols = sm.OLS(y_mret, X_ols).fit()

        print(f"\n  Alpha = {alpha_sel:.6f}", flush=True)
        print(f"  R2 = {ols.rsquared:.4f}, Adj R2 = {ols.rsquared_adj:.4f}", flush=True)
        print(f"  Selected LLM topics ({len(selected_topics)}):", flush=True)
        for t in selected_topics:
            nm = t[4:]  # strip 'llm_' prefix
            sig = '***' if ols.pvalues[t] < 0.01 else '**' if ols.pvalues[t] < 0.05 else '*' if ols.pvalues[t] < 0.1 else ''
            print(f"    {nm:30s}  coef={ols.params[t]:+10.6f}  p={ols.pvalues[t]:.4f} {sig}", flush=True)

        print(f"\n  -- Comparison to Part 1 --", flush=True)
        print(f"  Part 1(a) Topics R2 (k=5):      0.1079", flush=True)
        print(f"  Part 1(e) Raw Counts R2 (k=5):   0.1971", flush=True)
        print(f"  Part 2(a) LLM Topics R2 (k=5):   {ols.rsquared:.4f}  (based on {len(df_llm)} articles)", flush=True)

        print(f"\n{ols.summary()}", flush=True)

except Exception as e:
    print(f"\n  Regression failed: {e}", flush=True)
    print("  This is expected with only 500 articles — limited monthly overlap.", flush=True)

print(f"\n{'=' * 70}", flush=True)
print("  Pipeline test complete!", flush=True)
print("=" * 70, flush=True)
