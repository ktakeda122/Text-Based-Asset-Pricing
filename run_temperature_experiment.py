"""Part 2(b)(ii): Temperature experiment — compare output variability at temp=0.1 vs temp=0.9."""
import pandas as pd
import numpy as np
import time
import warnings
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

load_dotenv()
gemini_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)
print("API Connected!", flush=True)

SYSTEM_MSG = "You are a financial analyst summarizing potential economic or market risks from news articles."

# Select 20 random articles (fixed seed for reproducibility)
articles = pd.read_parquet('articles.pq')
np.random.seed(42)
sample_idx = np.random.choice(len(articles), 20, replace=False)
sample_headlines = articles['headline'].iloc[sample_idx].tolist()

print(f"Selected {len(sample_headlines)} random articles for temperature experiment\n", flush=True)

def generate_single_batch(headlines, temperature):
    """Generate topics for a list of headlines at given temperature."""
    n = len(headlines)
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
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
                    system_instruction=SYSTEM_MSG,
                    temperature=temperature,
                    max_output_tokens=4096,
                ),
            )
            text = response.text.strip()
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            cleaned = []
            for line in lines:
                for sep in ['. ', ') ', ': ']:
                    idx = line.find(sep)
                    if idx != -1 and line[:idx].strip().isdigit():
                        line = line[idx + len(sep):]
                        break
                cleaned.append(line.strip())
            if len(cleaned) >= n:
                return cleaned[:n]
            return cleaned + [""] * (n - len(cleaned))
        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {str(e)[:100]}", flush=True)
            time.sleep(3)
    return [""] * n

# Run 3 times at each temperature
results = {}
for temp in [0.1, 0.9]:
    results[temp] = []
    for run in range(3):
        print(f"Temperature={temp}, Run {run+1}/3...", flush=True)
        topics = generate_single_batch(sample_headlines, temp)
        results[temp].append(topics)
        time.sleep(2)

# ── Analysis ──────────────────────────────────────────────────────────────────
print(f"\n{'='*100}", flush=True)
print(f"  TEMPERATURE EXPERIMENT RESULTS", flush=True)
print(f"{'='*100}", flush=True)

print(f"\n{'Headline':<60s} | {'Temp=0.1 (3 runs)':<60s} | {'Temp=0.9 (3 runs)':<60s}", flush=True)
print("-" * 185, flush=True)

identical_low = 0
identical_high = 0

for i in range(20):
    headline = sample_headlines[i][:57] + "..." if len(sample_headlines[i]) > 57 else sample_headlines[i]

    low_topics = [results[0.1][r][i] for r in range(3)]
    high_topics = [results[0.9][r][i] for r in range(3)]

    # Check if all 3 runs are identical
    if len(set(low_topics)) == 1:
        identical_low += 1
    if len(set(high_topics)) == 1:
        identical_high += 1

    # Print first row with headline
    print(f"{headline:<60s} | {low_topics[0]:<60s} | {high_topics[0]:<60s}", flush=True)
    for r in range(1, 3):
        marker_l = " " if low_topics[r] == low_topics[0] else "*"
        marker_h = " " if high_topics[r] == high_topics[0] else "*"
        print(f"{'':60s} |{marker_l}{low_topics[r]:<59s} |{marker_h}{high_topics[r]:<59s}", flush=True)
    print(flush=True)

# ── Summary statistics ────────────────────────────────────────────────────────
print(f"{'='*100}", flush=True)
print(f"  SUMMARY", flush=True)
print(f"{'='*100}", flush=True)
print(f"  Temperature 0.1:", flush=True)
print(f"    Identical across all 3 runs: {identical_low}/20 articles ({100*identical_low/20:.0f}%)", flush=True)
print(f"    Changed across runs:         {20-identical_low}/20 articles ({100*(20-identical_low)/20:.0f}%)", flush=True)

print(f"  Temperature 0.9:", flush=True)
print(f"    Identical across all 3 runs: {identical_high}/20 articles ({100*identical_high/20:.0f}%)", flush=True)
print(f"    Changed across runs:         {20-identical_high}/20 articles ({100*(20-identical_high)/20:.0f}%)", flush=True)

# Count unique outputs per article
unique_low = [len(set(results[0.1][r][i] for r in range(3))) for i in range(20)]
unique_high = [len(set(results[0.9][r][i] for r in range(3))) for i in range(20)]

print(f"\n  Average unique outputs per article:", flush=True)
print(f"    Temp=0.1: {np.mean(unique_low):.2f} (out of 3 runs)", flush=True)
print(f"    Temp=0.9: {np.mean(unique_high):.2f} (out of 3 runs)", flush=True)

print(f"\n  Interpretation:", flush=True)
if identical_low > identical_high:
    print(f"    Low temperature (0.1) produces MORE consistent outputs -- {identical_low} vs {identical_high} identical.", flush=True)
    print(f"    High temperature (0.9) introduces MORE variability in topic extraction.", flush=True)
elif identical_low < identical_high:
    print(f"    Surprisingly, high temperature produced more consistent outputs.", flush=True)
else:
    print(f"    Both temperatures showed similar consistency.", flush=True)

print(f"\n{'='*100}", flush=True)
print(f"  TEMPERATURE EXPERIMENT COMPLETE", flush=True)
print(f"{'='*100}", flush=True)
