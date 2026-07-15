"""Fetch a seeded random 200-question sample of MMLU and save it to disk.

Stream-and-save: streaming=True reads only the test split's shards (never the
huge auxiliary_train split a non-streaming load would pull). The test split is
subject-sorted, so an unbiased sample REQUIRES a shuffle buffer that covers the
whole split (14,042 rows, a few MB of text) -- a small buffer would draw only
from the first subjects alphabetically. SEED makes the sample reproducible:
same seed -> same 200 questions, byte for byte. No model, no tokenizer here.
"""

from collections import Counter

from datasets import load_dataset, Dataset

N = 200
SEED = 42
BUFFER_SIZE = 20_000          # >= split size (14,042) -> uniform over the full split
OUT_DIR = f"data/mmlu_{N}"

stream = load_dataset("cais/mmlu", "all", split="test", streaming=True)
sampled = stream.shuffle(seed=SEED, buffer_size=BUFFER_SIZE).take(N)

# Materialize into a normal on-disk dataset (answer is inferred as int64; the
# original ClassLabel type is not preserved, but the 0-3 value is).
rows = list(sampled)
ds = Dataset.from_list(rows)
ds.save_to_disk(OUT_DIR)

LETTERS = "ABCD"
print(f"Saved {len(ds)} rows to {OUT_DIR}  (seed={SEED}, buffer={BUFFER_SIZE})")
print("Columns:", ds.column_names)

# subject spread: the whole point of this re-fetch -- must NOT be one subject
counts = Counter(ds["subject"])
print(f"\n=== Subject distribution ({len(counts)} distinct subjects) ===")
for subject, count in counts.most_common():
    print(f"  {count:>3}  {subject}")

first = ds[0]
print("\n=== First row ===")
print("Subject: ", first["subject"])
print("Question:", first["question"])
print("Choices:")
for letter, choice in zip(LETTERS, first["choices"]):
    print(f"  {letter}. {choice}")
print("Answer (int): ", first["answer"])
print("Answer (letter):", LETTERS[first["answer"]])
