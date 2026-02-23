"""Full pipeline: batched LLM topic generation + Lasso/OLS analysis."""
import pandas as pd
import numpy as np
import time
import os
import warnings
from tqdm import tqdm
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

load_dotenv()
gemini_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)
print("API Connected!", flush=True)

API_BATCH_SIZE = 50
CHECKPOINT_EVERY = 500

# ── Parsers ──────────────────────────────────────────────────────────────────
def _parse_numbered_lines(text, expected_n):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    cleaned = []
    for line in lines:
        for sep in ['. ', ') ', ': ']:
            idx = line.find(sep)
            if idx != -1 and line[:idx].strip().isdigit():
                line = line[idx + len(sep):]
                break
        cleaned.append(line.strip())
    if len(cleaned) == expected_n:
        return cleaned
    return None

def _parse_lenient(text, expected_n):
    result = _parse_numbered_lines(text, expected_n)
    if result is not None:
        return result
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    cleaned = []
    for line in lines:
        for sep in ['. ', ') ', ': ']:
            idx = line.find(sep)
            if idx != -1 and line[:idx].strip().isdigit():
                line = line[idx + len(sep):]
                break
        cleaned.append(line.strip())
    if len(cleaned) >= expected_n:
        return cleaned[:expected_n]
    return cleaned + [""] * (expected_n - len(cleaned))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Generate topics
# ══════════════════════════════════════════════════════════════════════════════
articles = pd.read_parquet('articles.pq')
output_file = 'articles_with_topics.parquet'

if os.path.exists(output_file):
    df_done = pd.read_parquet(output_file)
    start_idx = len(df_done)
    all_topics = list(df_done['generated_topics'])
    print(f"Resuming from checkpoint: {start_idx} / {len(articles)} done", flush=True)
else:
    start_idx = 0
    all_topics = []

all_headlines = articles['headline'].tolist()
remaining = all_headlines[start_idx:]
n_api_calls = (len(remaining) + API_BATCH_SIZE - 1) // API_BATCH_SIZE

if n_api_calls > 0:
    print(f"Processing {len(remaining)} remaining articles ({n_api_calls} API calls)", flush=True)

    t0 = time.time()
    last_checkpoint = start_idx

    for batch_idx in tqdm(range(0, len(remaining), API_BATCH_SIZE), desc="API calls", total=n_api_calls):
        batch = remaining[batch_idx:batch_idx + API_BATCH_SIZE]
        n = len(batch)

        numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(batch))
        user_prompt = f"""Below are {n} newspaper headlines, numbered 1 to {n}.

For EACH headline, extract exactly 1-3 keyword topics about the economic or financial risk/theme.

RULES:
- Output exactly {n} lines, numbered 1 to {n}.
- Each line must be: <number>. <1-3 keywords>
- Do NOT skip any line. Do NOT add extra lines.
- Do NOT repeat the headline -- only output keywords.

Headlines:
{numbered}

Output:"""

        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a financial analyst summarizing potential economic or market risks from news articles.",
                        temperature=0.3,
                        max_output_tokens=4096,
                    ),
                )
                text = response.text.strip()
                parsed = _parse_numbered_lines(text, n)
                if parsed is not None:
                    all_topics.extend(parsed)
                    break
                else:
                    tqdm.write(f"  Wrong count at {start_idx + batch_idx}, retrying ({attempt+1}/5)...")
                    time.sleep(2)
                    if attempt == 4:
                        all_topics.extend(_parse_lenient(text, n))
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "ResourceExhausted" in error_str:
                    wait = 2 ** attempt * 5
                    tqdm.write(f"  Rate limited ({attempt+1}/5), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"  Error ({attempt+1}/5): {error_str[:100]}")
                    if attempt == 4:
                        all_topics.extend([""] * n)
                    else:
                        time.sleep(3)

        time.sleep(1)

        if len(all_topics) - last_checkpoint >= CHECKPOINT_EVERY:
            df_cp = articles.iloc[:len(all_topics)].copy()
            df_cp['generated_topics'] = all_topics
            df_cp.to_parquet(output_file, index=False)
            last_checkpoint = len(all_topics)
            elapsed_total = time.time() - t0
            rate = (len(all_topics) - start_idx) / elapsed_total
            eta = (len(all_headlines) - len(all_topics)) / rate / 60 if rate > 0 else 0
            tqdm.write(f"  Checkpoint: {len(all_topics)}/{len(all_headlines)} (ETA: {eta:.0f} min)")

    df_final = articles.copy()
    df_final['generated_topics'] = all_topics
    df_final.to_parquet(output_file, index=False)

    gen_time = time.time() - t0
    empties = sum(1 for t in all_topics if not t)
    print(f"\nGeneration done! {len(all_topics)} articles in {gen_time/60:.1f} min, {empties} empty", flush=True)
else:
    print("All articles already generated!", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Monthly Aggregation
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print("  STEP 3: Monthly Aggregation", flush=True)
print(f"{'='*70}", flush=True)

df_llm = pd.read_parquet(output_file)
df_llm['date'] = pd.to_datetime(df_llm['display_date']).dt.to_period('M').dt.to_timestamp()

llm_vec = CountVectorizer(stop_words='english')
llm_dtm = llm_vec.fit_transform(df_llm['generated_topics'])
llm_feat_names = llm_vec.get_feature_names_out()
llm_cols = ['llm_' + f for f in llm_feat_names]

llm_dtm_df = pd.DataFrame(llm_dtm.toarray(), columns=llm_cols)
llm_dtm_df['date'] = df_llm['date'].values
llm_monthly = llm_dtm_df.groupby('date').sum().reset_index()

macro = pd.read_csv('macro.csv', parse_dates=['date'])
df_2a = pd.merge(macro, llm_monthly, on='date')

print(f"LLM DTM: {llm_dtm.shape[1]} unique terms from {len(df_llm)} articles", flush=True)
print(f"Monthly: {llm_monthly.shape[0]} months, merged: {df_2a.shape[0]} obs", flush=True)
print(f"Date range: {df_2a['date'].min().date()} to {df_2a['date'].max().date()}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Lasso (k=5) + OLS on mret
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print("  STEP 4: Lasso (k=5) + OLS on mret", flush=True)
print(f"{'='*70}", flush=True)

try:
    X_llm = df_2a[llm_cols].values.astype(float)
    scaler_llm = StandardScaler()
    X_llm_scaled = scaler_llm.fit_transform(X_llm)
    y_mret = df_2a['mret'].values

    alphas_grid = np.logspace(-6, 1, 200)
    alphas_out, coefs, _ = lasso_path(X_llm_scaled, y_mret, alphas=alphas_grid, max_iter=50000)
    n_nonzero = np.sum(coefs != 0, axis=0)

    mask_5 = n_nonzero == 5
    if mask_5.any():
        alpha_sel = alphas_out[mask_5].min()
        idx_sel = np.where(alphas_out == alpha_sel)[0][0]
    else:
        idx_sel = np.argmin(np.abs(n_nonzero - 5))
        alpha_sel = alphas_out[idx_sel]
        print(f"  (Closest k={n_nonzero[idx_sel]})", flush=True)

    selected_mask = coefs[:, idx_sel] != 0
    selected_topics = [llm_cols[j] for j in range(len(llm_cols)) if selected_mask[j]]

    X_ols = sm.add_constant(df_2a[selected_topics])
    ols = sm.OLS(y_mret, X_ols).fit()

    print(f"  Alpha = {alpha_sel:.6f}", flush=True)
    print(f"  R2 = {ols.rsquared:.4f}, Adj R2 = {ols.rsquared_adj:.4f}", flush=True)
    print(f"  Selected LLM topics ({len(selected_topics)}):", flush=True)
    for t in selected_topics:
        nm = t[4:]
        sig = '***' if ols.pvalues[t] < 0.01 else '**' if ols.pvalues[t] < 0.05 else '*' if ols.pvalues[t] < 0.1 else ''
        print(f"    {nm:30s}  coef={ols.params[t]:+10.6f}  p={ols.pvalues[t]:.4f} {sig}", flush=True)

    print(f"\n  -- Comparison to Part 1 --", flush=True)
    print(f"  Part 1(a) Topics R2 (k=5):      0.1079", flush=True)
    print(f"  Part 1(e) Raw Counts R2 (k=5):   0.1971", flush=True)
    print(f"  Part 2(a) LLM Topics R2 (k=5):   {ols.rsquared:.4f}", flush=True)
    print(f"\n{ols.summary()}", flush=True)

except Exception as e:
    print(f"  Regression failed: {e}", flush=True)

print(f"\n{'='*70}", flush=True)
print("  PIPELINE COMPLETE", flush=True)
print(f"{'='*70}", flush=True)
