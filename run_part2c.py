"""Part 2(c): OOS Forecasting of Industrial Production Growth using LLM Topics.
Expanding-window Ridge regression with GCV for alpha selection.
Target: indprol1 (1-month-ahead industrial production growth).
Features: LLM-generated topics from base/neutral prompt only.
"""
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import time
import warnings
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

t0 = time.time()

print("=" * 70, flush=True)
print("  Part 2(c): OOS Forecasting of Industrial Production Growth", flush=True)
print("  Features: LLM-Generated Topics (Base Prompt)", flush=True)
print("=" * 70, flush=True)

# -- 1. Build monthly LLM topic features ----------------------------------
llm_articles = pd.read_parquet('articles_with_topics.parquet')
llm_articles['date'] = pd.to_datetime(llm_articles['display_date']).dt.to_period('M').dt.to_timestamp()

llm_vec = CountVectorizer(stop_words='english')
llm_dtm = llm_vec.fit_transform(llm_articles['generated_topics'])  # sparse
llm_names = llm_vec.get_feature_names_out()
llm_cols = ['llm_' + f for f in llm_names]

# Aggregate to monthly using sparse matrix multiplication
dates_llm = llm_articles['date'].values
unique_dates = np.sort(pd.unique(dates_llm))
date_to_idx = {d: i for i, d in enumerate(unique_dates)}
row_indices = np.array([date_to_idx[d] for d in dates_llm])
agg = sparse.csr_matrix((np.ones(len(dates_llm)), (row_indices, np.arange(len(dates_llm)))),
                          shape=(len(unique_dates), len(dates_llm)))
llm_monthly_arr = (agg @ llm_dtm).toarray()

print(f"LLM features: {llm_monthly_arr.shape[1]} terms, {llm_monthly_arr.shape[0]} months", flush=True)

# -- 2. Merge with macro --------------------------------------------------
macro = pd.read_csv('macro.csv', parse_dates=['date'])

monthly_df = pd.DataFrame(llm_monthly_arr, columns=llm_cols)
monthly_df['date'] = unique_dates

df = pd.merge(macro[['date', 'indprol1']], monthly_df, on='date').sort_values('date').reset_index(drop=True)

# Drop NaN target rows
mask = df['indprol1'].notna()
df = df[mask].reset_index(drop=True)

X_all = df[llm_cols].values.astype(np.float64)
y_all = df['indprol1'].values
dates_final = df['date'].values

n_total = len(y_all)
print(f"Dataset: {n_total} months, {len(llm_cols)} LLM features", flush=True)
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}", flush=True)
print(f"Target: indprol1 (1-month-ahead industrial production growth)", flush=True)
print(f"  Mean: {y_all.mean():.6f}, Std: {y_all.std():.6f}", flush=True)

# -- 3. Expanding-window Ridge forecast ------------------------------------
min_train = int(n_total * 0.50)
n_test = n_total - min_train

print(f"\nExpanding window setup:", flush=True)
print(f"  Initial training: {min_train} months ({pd.Timestamp(dates_final[0]).date()} to {pd.Timestamp(dates_final[min_train-1]).date()})", flush=True)
print(f"  Test period:      {n_test} months ({pd.Timestamp(dates_final[min_train]).date()} to {pd.Timestamp(dates_final[-1]).date()})", flush=True)

alphas = np.logspace(0, 6, 15)

y_pred_oos = np.full(n_test, np.nan)
selected_alphas = []

for i in range(n_test):
    t = min_train + i

    # Standardize using only training data (no look-ahead)
    scaler = StandardScaler(with_mean=False)
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
        eta = elapsed / (i + 1) * (n_test - i - 1) / 60 if i > 0 else 0
        print(f"  Month {i+1:3d}/{n_test}: running OOS R2 = {running_r2:+.4f}, alpha = {ridge.alpha_:.0f}, ETA = {eta:.1f}min", flush=True)

# -- 4. Final Results ------------------------------------------------------
y_actual = y_all[min_train:]
oos_r2 = r2_score(y_actual, y_pred_oos)
oos_mse = np.mean((y_actual - y_pred_oos) ** 2)

# Historical mean baseline
hist_mean_pred = np.array([y_all[:min_train+i].mean() for i in range(n_test)])
r2_hist_mean = r2_score(y_actual, hist_mean_pred)

elapsed_total = time.time() - t0

print(f"\n{'='*70}", flush=True)
print(f"  RESULTS: OOS Forecasting of indprol1 (Industrial Production Growth)", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Model:          Ridge Regression with LOO-GCV alpha selection", flush=True)
print(f"  Features:       {len(llm_cols)} LLM-generated topic terms (base prompt)", flush=True)
print(f"  Training start: {pd.Timestamp(dates_final[0]).date()}", flush=True)
print(f"  Test period:    {pd.Timestamp(dates_final[min_train]).date()} to {pd.Timestamp(dates_final[-1]).date()} ({n_test} months)", flush=True)
print(f"  Runtime:        {elapsed_total:.1f}s", flush=True)
print(f"", flush=True)
print(f"  OOS R-squared:  {oos_r2:.4f}", flush=True)
print(f"  OOS MSE:        {oos_mse:.8f}", flush=True)
print(f"", flush=True)
print(f"  Baseline (expanding hist. mean) OOS R2: {r2_hist_mean:.4f}", flush=True)
print(f"  Median Ridge alpha selected: {np.median(selected_alphas):.0f}", flush=True)
print(f"{'='*70}", flush=True)
