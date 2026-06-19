import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load an open model via transformers. Defaults to a small instruct model;
# override with MODEL_NAME to try others.
model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
print(f"Loading model: {model_name}")

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {model.__class__.__name__} ({n_params:,} parameters)")

# Read one datapoint from the TriviaQA rows saved earlier by fetch_trivia.py.
ds = load_from_disk("data/trivia_50")
row = ds[0]
question = row["question"]
gold = row["answer"]["value"]

# Instruct models need their chat template; hand-formatting degrades answers.
messages = [{"role": "user", "content": f"Answer this trivia question concisely: {question}"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)

# Causal LMs echo the prompt, so decode only the newly generated tokens.
gen = out[0][inputs["input_ids"].shape[1]:]
answer = tok.decode(gen, skip_special_tokens=True)

print("\n=== Result ===")
print("Question:        ", question)
print("Generated answer:", answer.strip())
print("Gold answer:     ", gold)
