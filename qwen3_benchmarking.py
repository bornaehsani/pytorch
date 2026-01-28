import os
import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# -----------------------------
# Configuration
# -----------------------------
MODELS = [
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-0.6B",
]

NUM_RUNS = 100
DEVICE = "mps"
DTYPE = torch.bfloat16

# -----------------------------
# Benchmark loop
# -----------------------------
results = {}

for model_name in MODELS:
    print(f"\n==============================")
    print(f"Benchmarking {model_name}")
    print(f"==============================")

    times = []

    # Load tokenizer ONCE per model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_auth_token=True
    )

    for run in range(NUM_RUNS):
        print(f"Run {run + 1}/{NUM_RUNS} ...", end=" ")

        # Ensure clean state between runs
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        time.sleep(0.5)

        start = time.perf_counter()
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.tie_word_embeddings = False
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=DTYPE,
            device_map={"": DEVICE},
            trust_remote_code=True
        )
        load_time = time.perf_counter() - start
        times.append(load_time)

        print(f"{load_time:.2f}s")

        # Cleanup
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        time.sleep(0.5)

    cold = times[0]
    warm_avg = sum(times[1:]) / (NUM_RUNS - 1)

    results[model_name] = {
        "cold": cold,
        "warm_avg": warm_avg,
        "all": times,
    }

# -----------------------------
# Summary
# -----------------------------
print("\n\n========= SUMMARY =========")
for model_name, data in results.items():
    print(f" >> \n{model_name}")
    print(f"    - Cold load (run 1): {data['cold']:.2f}s")
    print(f"    - Warm avg (runs 2–{NUM_RUNS}): {data['warm_avg']:.2f}s")


"""
# -----------------------------
# Step 3: Generate text (example)
# -----------------------------
# Pick the first model for generation

model_name = MODELS[0]
print(f"\nLoading {model_name} for generation example...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_auth_token=True
)

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.tie_word_embeddings = False

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
    trust_remote_code=True
)
model.eval()

# Example chat prompt
messages = [
    {"role": "system", "content": "You are a concise technical assistant."},
    {"role": "user", "content": "Explain why Apple MPS struggles with large transformer models."}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

print("\n--- Model output ---\n")
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
"""
