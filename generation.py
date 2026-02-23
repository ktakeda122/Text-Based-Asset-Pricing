
#------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm
import google.generativeai as genai
import os
from dotenv import load_dotenv

#------------------------------------------------------------------------------
# Login into API
#------------------------------------------------------------------------------
load_dotenv()
gemini_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=gemini_key)
print('API Connected!')

#------------------------------------------------------------------------------

def generate_topics_batch(articles, model_name="gemini-2.5-flash", temperature=0.3,
                          persona=None, system_prompt=None,
                          batch_size=50, max_retries=5, rpm_limit=15):
    """
    Generate 1-3 keyword topics per article using batched API calls.
    Sends `batch_size` articles per request to minimize API call count.
    """
    all_topics = []
    min_interval = 60.0 / rpm_limit

    # Build system instruction
    system_msg = system_prompt or "You are a financial analyst summarizing potential economic or market risks from news articles."
    if persona == "bull":
        system_msg += " You are an overly optimistic investor who sees opportunity in every situation."
    elif persona == "bear":
        system_msg += " You are a deeply skeptical investor who sees risk and danger in market developments."

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_msg,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=4096,
        ),
    )

    n_batches = (len(articles) + batch_size - 1) // batch_size
    last_request_time = 0

    for batch_idx in tqdm(range(n_batches), desc="Batch progress"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(articles))
        batch = articles[start:end]
        n = len(batch)

        # Build numbered list of articles
        numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(batch))

        user_prompt = f"""Below are {n} newspaper headlines, numbered 1 to {n}.

For EACH headline, extract exactly 1-3 keyword topics about the economic or financial risk/theme.

RULES:
- Output exactly {n} lines, numbered 1 to {n}.
- Each line must be: <number>. <1-3 keywords>
- Do NOT skip any line. Do NOT add extra lines.
- Do NOT repeat the headline — only output keywords.

Headlines:
{numbered}

Output:"""

        # Proactive throttle
        elapsed = time.time() - last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        # Retry loop
        for attempt in range(max_retries):
            try:
                last_request_time = time.time()
                response = model.generate_content(user_prompt)
                text = response.text.strip()

                # Parse numbered lines
                parsed = _parse_numbered_lines(text, n)

                if parsed is not None:
                    all_topics.extend(parsed)
                    break
                else:
                    tqdm.write(f"  Batch {batch_idx}: got wrong count, retrying ({attempt+1}/{max_retries})...")
                    time.sleep(5)
                    if attempt == max_retries - 1:
                        # Fallback: pad with empty strings
                        lines = _parse_numbered_lines_lenient(text, n)
                        all_topics.extend(lines)

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "ResourceExhausted" in error_str or "rate" in error_str.lower():
                    wait = 2 ** attempt * 15
                    tqdm.write(f"  Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"  Error: {error_str[:120]}")
                    if attempt == max_retries - 1:
                        all_topics.extend([""] * n)
                    else:
                        time.sleep(5)

    return all_topics


def _parse_numbered_lines(text, expected_n):
    """Parse 'N. topic keywords' lines. Returns list of N strings or None if count wrong."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    # Strip leading number + period/parenthesis
    cleaned = []
    for line in lines:
        # Remove patterns like "1. ", "1) ", "1: "
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
    """Best-effort parse: return exactly expected_n items, padding or truncating."""
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

    # Pad or truncate
    if len(cleaned) >= expected_n:
        return cleaned[:expected_n]
    return cleaned + [""] * (expected_n - len(cleaned))


#------------------------------------------------------------------------------

if __name__ == "__main__":
    file_path = "articles.pq"
    df_articles = pd.read_parquet(file_path)

    articles = df_articles['headline'].tolist()

    # Basic generation
    df_articles['generated_topics'] = generate_topics_batch(articles, temperature=0.3)

    # With a bear persona
    df_articles['generated_topics_bear'] = generate_topics_batch(articles, temperature=0.3, persona='bear')

#------------------------------------------------------------------------------
