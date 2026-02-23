"""
Generate text embeddings for WSJ headlines using Google Gemini API.

Uses the `text-embedding-004` model with output_dimensionality=250.
Processes all ~10,200 headlines from articles.pq in batches of 100.
Saves result to articles_with_embeddings.parquet.
"""

import numpy as np
import pandas as pd
import time
import sys
import io
from tqdm import tqdm
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── API setup ────────────────────────────────────────────────────────────────
load_dotenv()
gemini_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)
print("API Connected!", flush=True)

MODEL = "gemini-embedding-001"
DIMENSIONS = 250
BATCH_SIZE = 100

# ── Load articles ────────────────────────────────────────────────────────────
df = pd.read_parquet("articles.pq")
headlines = df["headline"].tolist()
print(f"Loaded {len(headlines)} headlines", flush=True)

# ── Checkpoint logic ─────────────────────────────────────────────────────────
import os
CHECKPOINT_FILE = "embeddings_checkpoint.npy"
if os.path.exists(CHECKPOINT_FILE):
    checkpoint = np.load(CHECKPOINT_FILE)
    start_idx = len(checkpoint)
    all_embeddings = list(checkpoint)
    print(f"Resuming from checkpoint: {start_idx}/{len(headlines)} done", flush=True)
else:
    start_idx = 0
    all_embeddings = []

# ── Generate embeddings in batches ───────────────────────────────────────────
remaining = headlines[start_idx:]
n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

if n_batches > 0:
    print(f"Processing {len(remaining)} headlines in {n_batches} batches (dim={DIMENSIONS})", flush=True)
    t0 = time.time()

    for batch_idx in tqdm(range(0, len(remaining), BATCH_SIZE), desc="Embedding", total=n_batches):
        batch = remaining[batch_idx:batch_idx + BATCH_SIZE]

        for attempt in range(5):
            try:
                result = client.models.embed_content(
                    model=MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=DIMENSIONS,
                    ),
                )
                batch_embeddings = [e.values for e in result.embeddings]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "ResourceExhausted" in error_str:
                    wait = 2 ** attempt * 5
                    tqdm.write(f"  Rate limited ({attempt+1}/5), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"  Error ({attempt+1}/5): {error_str[:120]}")
                    if attempt == 4:
                        # Fill with zeros on total failure
                        all_embeddings.extend([[0.0] * DIMENSIONS] * len(batch))
                    else:
                        time.sleep(2)

        # Checkpoint every 1000 articles
        if len(all_embeddings) % 1000 < BATCH_SIZE:
            np.save(CHECKPOINT_FILE, np.array(all_embeddings))

    elapsed = time.time() - t0
    print(f"\nEmbedding done! {len(all_embeddings)} articles in {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
else:
    print("All embeddings already generated!", flush=True)

# ── Save to parquet ──────────────────────────────────────────────────────────
df["embedding"] = all_embeddings
df.to_parquet("articles_with_embeddings.parquet", index=False)

# Clean up checkpoint
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

# Verify
print(f"\nSaved articles_with_embeddings.parquet", flush=True)
print(f"  Shape: {df.shape}", flush=True)
print(f"  Embedding dim: {len(all_embeddings[0])}", flush=True)
print(f"  Sample embedding (first 5 values): {all_embeddings[0][:5]}", flush=True)
zero_count = sum(1 for e in all_embeddings if all(v == 0.0 for v in e))
print(f"  Zero embeddings (failures): {zero_count}", flush=True)
