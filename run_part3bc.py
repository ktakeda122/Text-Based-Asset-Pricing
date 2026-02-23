"""Part 3(b) & 3(c): Topic Recovery from Embeddings.

Can the 250-dim monthly embedding features reconstruct specific topic signals?

Part 3(b): Regress pre-built topics (from topics.csv) on monthly embeddings.
Part 3(c): Regress LLM-generated topic counts on monthly embeddings.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import warnings
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# SHARED: Build monthly embedding features
# ═══════════════════════════════════════════════════════════════════════════
df_emb = pd.read_parquet('articles_with_embeddings.parquet')
embeddings = np.array(df_emb['embedding'].tolist())
df_emb['date'] = pd.to_datetime(df_emb['display_date']).dt.to_period('M').dt.to_timestamp()

emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=emb_cols)
emb_df['date'] = df_emb['date'].values
emb_monthly = emb_df.groupby('date').mean().reset_index()

print(f"Monthly embeddings: {emb_monthly.shape[0]} months x {len(emb_cols)} dims")

# ═══════════════════════════════════════════════════════════════════════════
# PART 3(b): Recover pre-built topics from topics.csv
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Part 3(b): Recovering Pre-Built Topics from Embeddings")
print("=" * 70)

topics = pd.read_csv('topics.csv', parse_dates=['date'])
topic_cols = [c for c in topics.columns if c != 'date']

# Merge embeddings with topics on date
df_3b = pd.merge(topics, emb_monthly, on='date').sort_values('date').reset_index(drop=True)
print(f"Merged: {df_3b.shape[0]} months, {len(topic_cols)} topics, {len(emb_cols)} emb dims")

X_emb_3b = df_3b[emb_cols].values.astype(np.float64)
scaler_3b = StandardScaler()
X_emb_3b_sc = scaler_3b.fit_transform(X_emb_3b)

# Ridge regression for each topic (Y = topic attention, X = 250 embeddings)
alphas = np.logspace(-2, 4, 50)
results_3b = []

for tc in topic_cols:
    y = df_3b[tc].values
    ridge = RidgeCV(alphas=alphas, gcv_mode='svd')
    ridge.fit(X_emb_3b_sc, y)
    y_pred = ridge.predict(X_emb_3b_sc)
    r2 = r2_score(y, y_pred)
    results_3b.append({'topic': tc, 'r2': r2, 'alpha': ridge.alpha_})

results_3b.sort(key=lambda x: x['r2'], reverse=True)

# Print top and bottom topics
print(f"\n  {'Topic':<40} {'R2':>8}  {'Alpha':>8}")
print(f"  {'-'*40} {'-'*8}  {'-'*8}")

print(f"\n  TOP 20 BEST RECOVERED:")
for r in results_3b[:20]:
    bar = '#' * int(r['r2'] * 30)
    print(f"  {r['topic']:<40} {r['r2']:8.4f}  {r['alpha']:8.1f}  {bar}")

print(f"\n  BOTTOM 10 WORST RECOVERED:")
for r in results_3b[-10:]:
    bar = '#' * int(r['r2'] * 30)
    print(f"  {r['topic']:<40} {r['r2']:8.4f}  {r['alpha']:8.1f}  {bar}")

r2_vals_3b = [r['r2'] for r in results_3b]
print(f"\n  SUMMARY (all {len(topic_cols)} topics):")
print(f"    Mean R2:   {np.mean(r2_vals_3b):.4f}")
print(f"    Median R2: {np.median(r2_vals_3b):.4f}")
print(f"    Min R2:    {np.min(r2_vals_3b):.4f}  ({results_3b[-1]['topic']})")
print(f"    Max R2:    {np.max(r2_vals_3b):.4f}  ({results_3b[0]['topic']})")
print(f"    R2 > 0.5:  {sum(1 for r in r2_vals_3b if r > 0.5)}/{len(topic_cols)}")
print(f"    R2 > 0.7:  {sum(1 for r in r2_vals_3b if r > 0.7)}/{len(topic_cols)}")
print(f"    R2 > 0.9:  {sum(1 for r in r2_vals_3b if r > 0.9)}/{len(topic_cols)}")

# Highlight the 2 representative topics
print(f"\n  FEATURED TOPICS (detailed):")
for target_name in ['Recession', 'Oil market', 'Federal Reserve', 'Financial crisis', 'Elections', 'Terrorism']:
    match = [r for r in results_3b if r['topic'] == target_name]
    if match:
        r = match[0]
        rank = results_3b.index(r) + 1
        print(f"    {r['topic']:<30s}  R2 = {r['r2']:.4f}  (rank {rank}/{len(topic_cols)})")

# ═══════════════════════════════════════════════════════════════════════════
# PART 3(c): Recover LLM-generated topics from embeddings
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Part 3(c): Recovering LLM-Generated Topics from Embeddings")
print("=" * 70)

llm_articles = pd.read_parquet('articles_with_topics.parquet')
llm_articles['date'] = pd.to_datetime(llm_articles['display_date']).dt.to_period('M').dt.to_timestamp()

llm_vec = CountVectorizer(stop_words='english')
llm_dtm = llm_vec.fit_transform(llm_articles['generated_topics'])
llm_names = llm_vec.get_feature_names_out()
llm_cols = ['llm_' + f for f in llm_names]

# Aggregate to monthly using sparse matrix multiplication
dates_llm = llm_articles['date'].values
unique_dates = np.sort(pd.unique(dates_llm))
date_to_idx = {d: i for i, d in enumerate(unique_dates)}
row_idx = np.array([date_to_idx[d] for d in dates_llm])
agg = sparse.csr_matrix((np.ones(len(dates_llm)), (row_idx, np.arange(len(dates_llm)))),
                          shape=(len(unique_dates), len(dates_llm)))
llm_monthly_arr = (agg @ llm_dtm).toarray()
llm_monthly_df = pd.DataFrame(llm_monthly_arr, columns=llm_cols)
llm_monthly_df['date'] = unique_dates

print(f"LLM topics: {len(llm_names)} unique terms, {llm_monthly_df.shape[0]} months")

# Merge with embeddings
df_3c = pd.merge(llm_monthly_df, emb_monthly, on='date').sort_values('date').reset_index(drop=True)
print(f"Merged: {df_3c.shape[0]} months")

X_emb_3c = df_3c[emb_cols].values.astype(np.float64)
scaler_3c = StandardScaler()
X_emb_3c_sc = scaler_3c.fit_transform(X_emb_3c)

# Pick top 50 most frequent LLM terms for analysis
total_freq = llm_monthly_arr.sum(axis=0)
top50_idx = total_freq.argsort()[::-1][:50]
top50_llm_cols = [llm_cols[i] for i in top50_idx]

results_3c = []
for lc in top50_llm_cols:
    y = df_3c[lc].values.astype(float)
    ridge = RidgeCV(alphas=alphas, gcv_mode='svd')
    ridge.fit(X_emb_3c_sc, y)
    y_pred = ridge.predict(X_emb_3c_sc)
    r2 = r2_score(y, y_pred)
    freq = total_freq[llm_cols.index(lc)]
    results_3c.append({'term': lc[4:], 'r2': r2, 'alpha': ridge.alpha_, 'freq': freq})

results_3c.sort(key=lambda x: x['r2'], reverse=True)

print(f"\n  {'LLM Term':<25} {'Freq':>6} {'R2':>8}  {'Alpha':>8}")
print(f"  {'-'*25} {'-'*6} {'-'*8}  {'-'*8}")

print(f"\n  TOP 20 BEST RECOVERED:")
for r in results_3c[:20]:
    bar = '#' * int(r['r2'] * 30)
    print(f"  {r['term']:<25} {r['freq']:6.0f} {r['r2']:8.4f}  {r['alpha']:8.1f}  {bar}")

print(f"\n  BOTTOM 10 WORST RECOVERED:")
for r in results_3c[-10:]:
    bar = '#' * int(r['r2'] * 30)
    print(f"  {r['term']:<25} {r['freq']:6.0f} {r['r2']:8.4f}  {r['alpha']:8.1f}  {bar}")

r2_vals_3c = [r['r2'] for r in results_3c]
print(f"\n  SUMMARY (top 50 LLM terms by frequency):")
print(f"    Mean R2:   {np.mean(r2_vals_3c):.4f}")
print(f"    Median R2: {np.median(r2_vals_3c):.4f}")
print(f"    Min R2:    {np.min(r2_vals_3c):.4f}  ({results_3c[-1]['term']})")
print(f"    Max R2:    {np.max(r2_vals_3c):.4f}  ({results_3c[0]['term']})")
print(f"    R2 > 0.5:  {sum(1 for r in r2_vals_3c if r > 0.5)}/{len(results_3c)}")
print(f"    R2 > 0.7:  {sum(1 for r in r2_vals_3c if r > 0.7)}/{len(results_3c)}")

# Also run on ALL LLM terms (not just top 50)
all_r2s = []
for j in range(len(llm_cols)):
    y = df_3c[llm_cols[j]].values.astype(float)
    if y.std() == 0:
        continue
    ridge = RidgeCV(alphas=alphas, gcv_mode='svd')
    ridge.fit(X_emb_3c_sc, y)
    y_pred = ridge.predict(X_emb_3c_sc)
    all_r2s.append(r2_score(y, y_pred))

print(f"\n  ALL {len(all_r2s)} LLM TERMS (with non-zero variance):")
print(f"    Mean R2:   {np.mean(all_r2s):.4f}")
print(f"    Median R2: {np.median(all_r2s):.4f}")
print(f"    R2 > 0.5:  {sum(1 for r in all_r2s if r > 0.5)}/{len(all_r2s)}")

# ═══════════════════════════════════════════════════════════════════════════
# OVERALL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  OVERALL: Can Embeddings Reconstruct Topic Signals?")
print(f"{'='*70}")
print(f"  Pre-built topics (180 topics from topics.csv):")
print(f"    Mean R2 = {np.mean(r2_vals_3b):.4f}, Median = {np.median(r2_vals_3b):.4f}")
print(f"    {sum(1 for r in r2_vals_3b if r > 0.5)}/{len(r2_vals_3b)} topics recovered with R2 > 0.5")
print(f"")
print(f"  LLM-generated topics (top 50 terms from Part 2a):")
print(f"    Mean R2 = {np.mean(r2_vals_3c):.4f}, Median = {np.median(r2_vals_3c):.4f}")
print(f"    {sum(1 for r in r2_vals_3c if r > 0.5)}/{len(r2_vals_3c)} terms recovered with R2 > 0.5")
print(f"{'='*70}")
