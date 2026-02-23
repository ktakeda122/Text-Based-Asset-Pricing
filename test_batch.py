"""Test batched topic generation on 100 articles (2 batches of 50)."""
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

import google.generativeai as genai
from tqdm import tqdm

genai.configure(api_key="AIzaSyARCOQdJ5grxpIuHf_sLr5zd5-Ma3jyE-k")
print("API Connected!", flush=True)

# ── Helper parsers ───────────────────────────────────────────────────────────
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

def _parse_numbered_lines_lenient(text, expected_n):
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

# ── Batch generation ─────────────────────────────────────────────────────────
BATCH_SIZE = 50
RPM_LIMIT = 15
MIN_INTERVAL = 60.0 / RPM_LIMIT

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="You are a financial analyst summarizing potential economic or market risks from news articles.",
    generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=4096),
)

articles = pd.read_parquet('articles.pq')
test_headlines = articles['headline'].head(100).tolist()
print(f"Testing on {len(test_headlines)} articles ({len(test_headlines)//BATCH_SIZE} batches of {BATCH_SIZE})...", flush=True)

all_topics = []
last_request_time = 0
t0 = time.time()

for batch_idx in range(0, len(test_headlines), BATCH_SIZE):
    batch = test_headlines[batch_idx:batch_idx + BATCH_SIZE]
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

    elapsed = time.time() - last_request_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    for attempt in range(5):
        try:
            last_request_time = time.time()
            print(f"\n  Sending batch {batch_idx//BATCH_SIZE + 1} ({n} articles)...", flush=True)
            response = model.generate_content(user_prompt)
            text = response.text.strip()

            parsed = _parse_numbered_lines(text, n)
            if parsed is not None:
                all_topics.extend(parsed)
                print(f"  OK: parsed {len(parsed)} topics", flush=True)
                break
            else:
                lenient = _parse_numbered_lines_lenient(text, n)
                print(f"  Lenient parse: got {len(lenient)} (expected {n})", flush=True)
                if attempt == 4:
                    all_topics.extend(lenient)
                else:
                    print(f"  Retrying...", flush=True)
                    time.sleep(5)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str or "rate" in error_str.lower():
                wait = 2 ** attempt * 15
                print(f"  Rate limited (attempt {attempt+1}/5), waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  Error: {error_str[:150]}", flush=True)
                if attempt == 4:
                    all_topics.extend([""] * n)
                else:
                    time.sleep(5)

elapsed_total = time.time() - t0
print(f"\n{'='*80}", flush=True)
print(f"Completed {len(all_topics)} topics in {elapsed_total:.1f}s", flush=True)
print(f"API calls: {len(test_headlines)//BATCH_SIZE} (vs {len(test_headlines)} without batching)", flush=True)
empties = sum(1 for t in all_topics if not t)
print(f"Empty results: {empties}/{len(all_topics)}", flush=True)
print(f"{'='*80}", flush=True)

for i in range(len(all_topics)):
    flag = " *** EMPTY ***" if not all_topics[i] else ""
    print(f"  [{i:3d}] {all_topics[i]}{flag}", flush=True)
