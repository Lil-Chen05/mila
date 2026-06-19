"""Fetch a 20-row slice of MMLU and save it to disk for the experiment.

Stream-and-save (not full-download): with streaming=True only the rows we
.take() are pulled, then Dataset.from_list turns them into a normal on-disk
dataset. Mirrors fetch_trivia.py. No model, no tokenizer, no parsing here.
"""

from datasets import load_dataset, Dataset

# streaming=True pulls rows on demand instead of downloading the whole split.
# Config "all" combines every MMLU subject; we read the test split.
stream = load_dataset("cais/mmlu", "all", split="test", streaming=True)

# Grab only the first 20 rows. NOTE: no shuffle, so these will all come from
# whichever subject sorts first in the split — fine for a pipeline test.
rows = list(stream.take(20))

# Turn those rows into a normal on-disk dataset (answer is inferred as int64;
# the original ClassLabel type is not preserved, but the 0-3 value is).
ds = Dataset.from_list(rows)
ds.save_to_disk("data/mmlu_20")

LETTERS = "ABCD"
first = ds[0]

print(f"Saved {len(ds)} rows to data/mmlu_20")
print("Columns:", ds.column_names)
print("\n=== First row ===")
print("Subject: ", first["subject"])
print("Question:", first["question"])
print("Choices:")
for letter, choice in zip(LETTERS, first["choices"]):
    print(f"  {letter}. {choice}")
print("Answer (int): ", first["answer"])
print("Answer (letter):", LETTERS[first["answer"]])
