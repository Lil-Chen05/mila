from datasets import load_from_disk

ds = load_from_disk("data/trivia_50")
print(f"Rows: {len(ds)}")
print(f"Columns: {ds.column_names}\n")

row = ds[0]

def short(x, n=200):
    s = str(x)
    return s if len(s) <= n else s[:n] + f"  ...[{len(s)} chars total]"

print("=== Simple string fields ===")
print("question:       ", row["question"])
print("question_id:    ", row["question_id"])
print("question_source:", row["question_source"])

print("\n=== answer (a structured dict) ===")
for k, v in row["answer"].items():
    print(f"  {k}: {short(v)}")

# entity_pages and search_results are dicts of parallel lists
for field in ["entity_pages", "search_results"]:
    obj = row[field]
    print(f"\n=== {field} (dict of parallel lists) ===")
    print("  sub-keys:", list(obj.keys()))
    first_key = next(iter(obj))
    n_items = len(obj[first_key])
    print(f"  number of evidence items in this row: {n_items}")
    if n_items > 0:
        print("  --- first item across each sub-key ---")
        for k in obj:
            print(f"    {k}: {short(obj[k][0])}")
