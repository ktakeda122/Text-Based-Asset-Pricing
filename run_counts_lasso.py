import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings, sys
warnings.filterwarnings('ignore')

# ── Load and build monthly DTM ──────────────────────────────────────────────
articles = pd.read_parquet('articles.pq')
macro = pd.read_csv('macro.csv', parse_dates=['date'])
topics = pd.read_csv('topics.csv', parse_dates=['date'])

vectorizer = CountVectorizer(stop_words='english')
dtm = vectorizer.fit_transform(articles['headline'])
feature_names = vectorizer.get_feature_names_out()

# Prefix count columns to avoid name collision with macro columns
count_cols = ['w_' + f for f in feature_names]
# Map for display: w_xyz -> xyz
display_name = {c: c[2:] for c in count_cols}

# Aggregate counts to monthly sums
articles['date'] = articles['display_date'].dt.to_period('M').dt.to_timestamp()
dtm_df = pd.DataFrame(dtm.toarray(), columns=count_cols)
dtm_df['date'] = articles['date'].values
dtm_monthly = dtm_df.groupby('date').sum().reset_index()

# Merge with macro
df = pd.merge(macro, dtm_monthly, on='date')
df_topics = pd.merge(macro, topics, on='date')
topic_cols = [c for c in topics.columns if c != 'date']

X_counts = df[count_cols].values.astype(float)
scaler = StandardScaler()
X_counts_scaled = scaler.fit_transform(X_counts)

X_topics = df_topics[topic_cols].values
scaler_t = StandardScaler()
X_topics_scaled = scaler_t.fit_transform(X_topics)

print(f"Monthly DTM: {df.shape[0]} months x {len(count_cols)} terms")
print(f"Topics:      {df_topics.shape[0]} months x {len(topic_cols)} topics")


# ── Helpers ──────────────────────────────────────────────────────────────────
def lasso_select_ols(y, X_scaled, col_names, df_use, target_k=5):
    alphas_grid = np.logspace(-6, 1, 200)
    alphas_out, coefs, _ = lasso_path(X_scaled, y, alphas=alphas_grid, max_iter=50000)
    n_nonzero = np.sum(coefs != 0, axis=0)

    mask_k = n_nonzero == target_k
    if mask_k.any():
        idx = np.where(alphas_out == alphas_out[mask_k].min())[0][0]
    else:
        idx = np.argmin(np.abs(n_nonzero - target_k))

    sel_mask = coefs[:, idx] != 0
    sel_names = [col_names[j] for j in range(len(col_names)) if sel_mask[j]]
    if len(sel_names) == 0:
        return [], 0.0

    X_ols = sm.add_constant(df_use[sel_names])
    ols = sm.OLS(y, X_ols).fit()
    return sel_names, ols.rsquared


def find_k_for_target_r2(y, X_scaled, col_names, df_use, target_r2):
    """Find smallest k non-zero Lasso coefficients whose OLS R2 >= target_r2."""
    alphas_grid = np.logspace(-7, 1, 300)
    alphas_out, coefs, _ = lasso_path(X_scaled, y, alphas=alphas_grid, max_iter=50000)
    n_nonzero = np.sum(coefs != 0, axis=0)

    for k in sorted(set(n_nonzero)):
        if k == 0:
            continue
        mask_k = n_nonzero == k
        if not mask_k.any():
            continue
        idx = np.where(alphas_out == alphas_out[mask_k].min())[0][0]
        sel_mask = coefs[:, idx] != 0
        sel_names = [col_names[j] for j in range(len(col_names)) if sel_mask[j]]
        if len(sel_names) == 0:
            continue
        X_ols = sm.add_constant(df_use[sel_names])
        ols = sm.OLS(y, X_ols).fit()
        if ols.rsquared >= target_r2:
            return k, ols.rsquared, sel_names
    return None, None, None


# ── Run for all outcomes ─────────────────────────────────────────────────────
vol_cols_macro = [c for c in macro.columns if c.endswith('_vol')]
outcome_vars = ['mret', 'vol', 'indpro', 'indprol1'] + vol_cols_macro

print("\n" + "=" * 140)
print(f"{'Outcome':<15} {'Topics R2':>10} {'Counts R2':>10}  "
      f"{'k to match':>10} {'Matched R2':>10}   Selected count terms (k=5)")
print("=" * 140)

topics_r2s = []
counts_r2s = []
k_matches = []

for i, outcome in enumerate(outcome_vars):
    sys.stderr.write(f'\r{i+1}/{len(outcome_vars)} {outcome:20s}')
    sys.stderr.flush()

    y_t = df_topics[outcome].values
    y_c = df[outcome].values

    # Topics: 5 non-zero
    t_names, t_r2 = lasso_select_ols(y_t, X_topics_scaled, topic_cols, df_topics, target_k=5)

    # Counts: 5 non-zero
    c_names, c_r2 = lasso_select_ols(y_c, X_counts_scaled, count_cols, df, target_k=5)

    # How many counts needed to match topics R2?
    k_match, matched_r2, _ = find_k_for_target_r2(y_c, X_counts_scaled, count_cols, df, t_r2)

    topics_r2s.append(t_r2)
    counts_r2s.append(c_r2)
    k_matches.append(k_match)

    c_display = ", ".join(display_name[c] for c in c_names[:5]) if c_names else "N/A"
    k_str = str(k_match) if k_match else ">max"
    m_str = f"{matched_r2:.4f}" if matched_r2 else "N/A"

    print(f"{outcome:<15} {t_r2:10.4f} {c_r2:10.4f}  {k_str:>10} {m_str:>10}   {c_display}")

sys.stderr.write('\n')

# ── Detailed: mret ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("DETAILED: mret with 5 count terms")
print("=" * 80)
y_mret = df['mret'].values
c5_names, _ = lasso_select_ols(y_mret, X_counts_scaled, count_cols, df, target_k=5)
if c5_names:
    X_ols = sm.add_constant(df[c5_names])
    ols = sm.OLS(y_mret, X_ols).fit()
    # Rename for readable output
    X_ols_disp = X_ols.rename(columns=display_name)
    ols_disp = sm.OLS(y_mret, X_ols_disp).fit()
    print(ols_disp.summary())

# ── Summary ─────────────────────────────────────────────────────────────────
topics_r2s = np.array(topics_r2s)
counts_r2s = np.array(counts_r2s)
k_valid = [k for k in k_matches if k is not None]

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"With 5 features each:")
print(f"  Topics  mean R2: {topics_r2s.mean():.4f}  median: {np.median(topics_r2s):.4f}")
print(f"  Counts  mean R2: {counts_r2s.mean():.4f}  median: {np.median(counts_r2s):.4f}")
print(f"  Topics > Counts in {(topics_r2s > counts_r2s).sum()}/{len(outcome_vars)} outcomes")
print(f"\nTo match Topics R2 with counts:")
if k_valid:
    print(f"  Mean k needed:   {np.mean(k_valid):.1f}")
    print(f"  Median k needed: {np.median(k_valid):.1f}")
    print(f"  Min k:           {np.min(k_valid)}")
    print(f"  Max k:           {np.max(k_valid)}")
print(f"  Outcomes where counts could not match: {sum(1 for k in k_matches if k is None)}/{len(outcome_vars)}")
