"""Part 2(b)(iii): Lookahead bias experiment --basic vs constrained prompt on 2007-2008 articles."""
import pandas as pd
import numpy as np
import time
import warnings
from google import genai
from google.genai import types

warnings.filterwarnings('ignore')

client = genai.Client(api_key="AIzaSyARCOQdJ5grxpIuHf_sLr5zd5-Ma3jyE-k")
print("API Connected!", flush=True)

# ── Select 10 articles from late 2007 to mid-2008 (GFC period) ───────────────
articles = pd.read_parquet('articles.pq')
articles['date'] = pd.to_datetime(articles['display_date'])

gfc_mask = (articles['date'] >= '2007-06-01') & (articles['date'] <= '2008-09-30')
gfc_articles = articles[gfc_mask].reset_index(drop=True)
print(f"Articles in Jun 2007 - Sep 2008: {len(gfc_articles)}", flush=True)

# Pick 10 diverse articles spanning the period
np.random.seed(123)
selected_idx = np.random.choice(len(gfc_articles), 10, replace=False)
selected = gfc_articles.iloc[sorted(selected_idx)].reset_index(drop=True)

print(f"\nSelected 10 GFC-era articles:", flush=True)
for i, row in selected.iterrows():
    print(f"  [{i}] {row['date'].date()} --{row['headline']}", flush=True)

# ── Helper: generate topics for a list of headlines with a given system prompt ─
def generate_topics(headlines, system_msg, temperature=0.3):
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
                    system_instruction=system_msg,
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

# ── Prompt 1: Basic (no time constraint --may exhibit lookahead bias) ────────
basic_system = "You are a financial analyst summarizing potential economic or market risks from news articles."

print(f"\nGenerating with BASIC prompt...", flush=True)
basic_topics = generate_topics(selected['headline'].tolist(), basic_system)
time.sleep(3)

# ── Prompt 2: Constrained (no future knowledge) ─────────────────────────────
constrained_system = (
    "You are a financial analyst working in the year the article was published. "
    "Identify economic or financial risks based ONLY on information that was available at the exact time of publication. "
    "You absolutely must NOT reference, mention, or allude to ANY future events such as the 2008 financial crisis, "
    "the Lehman Brothers collapse, the housing market crash, TARP, the Great Recession, or any other events "
    "that occurred AFTER the article's publication date. "
    "Analyze ONLY what the headline says at face value, as if you have no knowledge of what happens next."
)

print(f"Generating with CONSTRAINED prompt (no lookahead)...", flush=True)
constrained_topics = generate_topics(selected['headline'].tolist(), constrained_system)

# ── Side-by-side comparison ──────────────────────────────────────────────────
print(f"\n{'='*120}", flush=True)
print(f"  LOOKAHEAD BIAS EXPERIMENT: Basic vs. Constrained Prompt", flush=True)
print(f"{'='*120}", flush=True)

# Keywords that indicate lookahead bias
lookahead_keywords = ['crash', 'crisis', 'lehman', 'collapse', 'recession', 'tarp',
                      'bailout', 'meltdown', 'contagion', 'systemic', 'subprime',
                      'great recession', '2008', 'financial crisis']

basic_bias_count = 0
constrained_bias_count = 0

for i in range(10):
    headline = selected.iloc[i]['headline']
    date = selected.iloc[i]['date'].date()
    basic = basic_topics[i]
    constrained = constrained_topics[i]

    # Check for lookahead keywords
    basic_lower = basic.lower()
    constrained_lower = constrained.lower()

    basic_flags = [kw for kw in lookahead_keywords if kw in basic_lower]
    constrained_flags = [kw for kw in lookahead_keywords if kw in constrained_lower]

    if basic_flags:
        basic_bias_count += 1
    if constrained_flags:
        constrained_bias_count += 1

    basic_marker = f" !! LOOKAHEAD: {basic_flags}" if basic_flags else ""
    constrained_marker = f" !! LOOKAHEAD: {constrained_flags}" if constrained_flags else ""

    print(f"\n  [{i}] {date} --{headline}", flush=True)
    print(f"      Basic:       {basic}{basic_marker}", flush=True)
    print(f"      Constrained: {constrained}{constrained_marker}", flush=True)

    if basic != constrained:
        print(f"      >>> DIFFERENT", flush=True)
    else:
        print(f"      >>> SAME", flush=True)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*120}", flush=True)
print(f"  SUMMARY", flush=True)
print(f"{'='*120}", flush=True)
print(f"  Basic prompt:       {basic_bias_count}/10 articles had lookahead-indicative keywords", flush=True)
print(f"  Constrained prompt: {constrained_bias_count}/10 articles had lookahead-indicative keywords", flush=True)

n_different = sum(1 for i in range(10) if basic_topics[i] != constrained_topics[i])
print(f"  Outputs that differed: {n_different}/10", flush=True)

print(f"\n  Interpretation:", flush=True)
if basic_bias_count > constrained_bias_count:
    print(f"    The constrained prompt successfully reduced lookahead bias.", flush=True)
    print(f"    The basic prompt used future-knowledge terms in {basic_bias_count} articles,", flush=True)
    print(f"    while the constrained prompt reduced this to {constrained_bias_count}.", flush=True)
elif basic_bias_count == constrained_bias_count == 0:
    print(f"    Neither prompt showed obvious lookahead bias in keyword topics.", flush=True)
    print(f"    The LLM may be summarizing at too high a level to exhibit bias,", flush=True)
    print(f"    or the 1-3 keyword format limits the opportunity for bias to appear.", flush=True)
else:
    print(f"    Results are mixed --both prompts showed similar levels of lookahead-indicative keywords.", flush=True)

print(f"\n  Lookahead keywords checked: {', '.join(lookahead_keywords)}", flush=True)

print(f"\n{'='*120}", flush=True)
print(f"  LOOKAHEAD BIAS EXPERIMENT COMPLETE", flush=True)
print(f"{'='*120}", flush=True)
