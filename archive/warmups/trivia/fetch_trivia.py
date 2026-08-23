from datasets import load_dataset, Dataset

# streaming=True means rows are pulled on demand instead of
# downloading the whole (tens-of-GB) dataset.
stream = load_dataset("mandarjoshi/trivia_qa", "rc",
                      split="train", streaming=True)

# Grab only the first 50 rows.
rows = list(stream.take(50))

# Turn those 50 rows into a normal on-disk dataset.
ds = Dataset.from_list(rows)
ds.save_to_disk("data/trivia_50")

print(f"Saved {len(ds)} rows to data/trivia_50")
print("Columns:", ds.column_names)
print("First question:", ds[0]["question"])
