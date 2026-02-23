import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, lasso_path
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings, sys
warnings.filterwarnings('ignore')

# Load data
macro = pd.read_csv('macro.csv', parse_dates=['date'])
topics = pd.read_csv('topics.csv', parse_dates=['date'])
df = pd.merge(macro, topics, on='date')

topic_cols = [c for c in topics.columns if c != 'date']
X = df[topic_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# All outcome variables
vol_cols = [c for c in macro.columns if c.endswith('_vol')]
outcome_vars = ['vol', 'indpro', 'indprol1'] + vol_cols

alphas_grid = np.logspace(-6, 1, 200)

results = []

for i, outcome in enumerate(outcome_vars):
    y = df[outcome].values
    sys.stderr.write(f'\r{i+1}/{len(outcome_vars)} {outcome:20s}')
    sys.stderr.flush()

    # Use lasso_path for speed
    alphas_out, coefs, _ = lasso_path(X_scaled, y, alphas=alphas_grid, max_iter=50000)
    # coefs shape: (n_features, n_alphas)
    n_nonzero = np.sum(coefs != 0, axis=0)

    # Find alpha yielding 5 non-zero coefficients
    mask_5 = n_nonzero == 5
    if mask_5.any():
        alpha_5 = alphas_out[mask_5].min()
        idx = np.where(alphas_out == alpha_5)[0][0]
    else:
        idx = np.argmin(np.abs(n_nonzero - 5))
        alpha_5 = alphas_out[idx]

    selected_mask = coefs[:, idx] != 0
    n_selected = selected_mask.sum()
    selected_topics = [topic_cols[j] for j in range(len(topic_cols)) if selected_mask[j]]

    if n_selected == 0:
        results.append({
            'outcome': outcome, 'alpha': alpha_5, 'n_nonzero': 0,
            'r2': 0.0, 'adj_r2': 0.0, 'f_pvalue': 1.0,
            'topics': [], 'ols_coefs': [], 'ols_pvals': [],
        })
        continue

    # OLS
    X_ols = sm.add_constant(df[selected_topics])
    ols_model = sm.OLS(y, X_ols).fit()

    results.append({
        'outcome': outcome,
        'alpha': alpha_5,
        'n_nonzero': n_selected,
        'r2': ols_model.rsquared,
        'adj_r2': ols_model.rsquared_adj,
        'f_pvalue': ols_model.f_pvalue,
        'topics': selected_topics,
        'ols_coefs': [ols_model.params[t] for t in selected_topics],
        'ols_pvals': [ols_model.pvalues[t] for t in selected_topics],
    })

sys.stderr.write('\n')

# =========================================================
# Print summary table
# =========================================================
print('=' * 120)
print(f'{"Outcome":<15} {"R2":>7} {"AdjR2":>7} {"F p-val":>10} {"#sel":>4}   Selected Topics')
print('=' * 120)
for r in results:
    topic_str = ', '.join(r['topics'])
    print(f'{r["outcome"]:<15} {r["r2"]:7.4f} {r["adj_r2"]:7.4f} {r["f_pvalue"]:10.2e} {r["n_nonzero"]:4d}   {topic_str}')
print('=' * 120)

# =========================================================
# Detailed results for vol, indpro, indprol1
# =========================================================
print('\n' + '=' * 100)
print('DETAILED RESULTS FOR KEY MACRO OUTCOMES')
print('=' * 100)

for r in results[:3]:
    print(f'\n--- {r["outcome"]} (R2={r["r2"]:.4f}, Adj R2={r["adj_r2"]:.4f}, alpha={r["alpha"]:.6f}) ---')
    for t, c, p in sorted(zip(r['topics'], r['ols_coefs'], r['ols_pvals']),
                           key=lambda x: abs(x[1]), reverse=True):
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f'  {t:40s}  coef={c:+12.4f}  p={p:.4f} {sig}')

# =========================================================
# Detailed results for individual volatilities
# =========================================================
print('\n' + '=' * 100)
print('DETAILED RESULTS FOR INDUSTRY VOLATILITIES')
print('=' * 100)

for r in results[3:]:
    print(f'\n--- {r["outcome"]} (R2={r["r2"]:.4f}, Adj R2={r["adj_r2"]:.4f}) ---')
    for t, c, p in sorted(zip(r['topics'], r['ols_coefs'], r['ols_pvals']),
                           key=lambda x: abs(x[1]), reverse=True):
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f'  {t:40s}  coef={c:+12.4f}  p={p:.4f} {sig}')

# =========================================================
# Summary statistics
# =========================================================
r2_vals = np.array([r['r2'] for r in results])
print(f'\n\n{"=" * 60}')
print(f'SUMMARY ACROSS ALL {len(results)} OUTCOMES')
print(f'{"=" * 60}')
print(f'Mean R2:   {np.mean(r2_vals):.4f}')
print(f'Median R2: {np.median(r2_vals):.4f}')
print(f'Min R2:    {np.min(r2_vals):.4f} ({results[int(np.argmin(r2_vals))]["outcome"]})')
print(f'Max R2:    {np.max(r2_vals):.4f} ({results[int(np.argmax(r2_vals))]["outcome"]})')

# R2 by category
r2_macro = [r['r2'] for r in results[:3]]
r2_indvol = [r['r2'] for r in results[3:]]
print(f'\nMacro outcomes (vol, indpro, indprol1) mean R2: {np.mean(r2_macro):.4f}')
print(f'Industry volatilities mean R2:                  {np.mean(r2_indvol):.4f}')

# Topic frequency
from collections import Counter
all_topics = []
for r in results:
    all_topics.extend(r['topics'])
freq = Counter(all_topics)
print(f'\nMost frequently selected topics across all {len(results)} outcomes:')
for t, count in freq.most_common(25):
    print(f'  {t:40s}  selected {count:2d}/{len(results)} times')
