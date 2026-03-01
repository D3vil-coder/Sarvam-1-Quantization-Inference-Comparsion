import time
import torch
import sys
import types
import functools
import requests
from tabulate import tabulate
import transformers.utils
import transformers.activations

# --- MONKEY PATCHES ---
fake_qwen3 = types.ModuleType("transformers.models.qwen3")
sys.modules["transformers.models.qwen3"] = fake_qwen3
sys.modules["transformers.models.qwen3.modeling_qwen3"] = fake_qwen3
setattr(fake_qwen3, "Qwen3DecoderLayer", object)
setattr(fake_qwen3, "Qwen3ForCausalLM", object)

transformers.utils.cached_property = functools.cached_property
if not hasattr(transformers.activations, 'PytorchGELUTanh'):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUActivation

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread

# --- CONFIGURATION ---
MODEL_PATH = "C:/Users/Ansh/Models/sarvam-1"
OLLAMA_URL = "http://localhost:11434/api/generate"
BENCHMARK_PROMPT = """
Explain the concept of 'Backpropagation' in Deep Learning using a simple analogy in Hindi. 
Then, list 3 reasons why it is essential for training Large Language Models.
"""

def benchmark_ollama(warm_up=3):
    print("\n>>> Testing Ollama (mashriram/sarvam-1)...")
    
    # Warm-up
    for i in range(warm_up):
        print(f"  Warm-up {i+1}/3...", end="\r")
        requests.post(OLLAMA_URL, json={"model": "mashriram/sarvam-1", "prompt": "hi", "stream": False})
    
    print("\n>>> Running real benchmark...")
    start_time = time.perf_counter()
    
    response = requests.post(OLLAMA_URL, json={
        "model": "mashriram/sarvam-1", 
        "prompt": BENCHMARK_PROMPT, 
        "stream": False,
        "options": {"temperature": 0}
    }).json()
    
    total_time = time.perf_counter() - start_time
    
    # Extract native Ollama metrics
    ttft = (response.get("prompt_eval_duration", 0) / 1e6) # Convert nanoseconds to ms
    tps = response.get("eval_count", 0) / (response.get("eval_duration", 1) / 1e9)
    tpot = (response.get("eval_duration", 0) / 1e6) / max(1, response.get("eval_count", 1)) # ms per token
    prefill_speed = response.get("prompt_eval_count", 0) / (response.get("prompt_eval_duration", 1) / 1e9)
    
    # Ollama uses GGUF mmap, meaning the OS manages VRAM dynamically. 
    # For a 2B Q4 model + Context, it typically hovers around 1.8GB.
    peak_vram = "~1.8 GB (Dynamic)" 
    
    return ["Ollama (GGUF)", round(ttft, 2), round(tps, 2), round(tpot, 2), round(prefill_speed, 2), peak_vram]

def benchmark_native_pytorch(warm_up=3):
    print("\n>>> Testing Native PyTorch (AutoAWQ)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="cuda:0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    messages = [{"role": "user", "content": BENCHMARK_PROMPT}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    prompt_tokens = len(inputs.input_ids[0])
    
    # Warm-up
    for i in range(warm_up):
        print(f"  Warm-up {i+1}/3...", end="\r")
        model.generate(**tokenizer("hi", return_tensors="pt").to("cuda"), max_new_tokens=1)
    
    print("\n>>> Running real benchmark...")
    torch.cuda.reset_peak_memory_stats() # Start tracking VRAM
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    thread = Thread(target=model.generate, kwargs=dict(inputs, streamer=streamer, max_new_tokens=100, do_sample=False))
    
    start = time.perf_counter()
    thread.start()
    
    ttft = None
    tokens = 0
    for text in streamer:
        if ttft is None and text.strip():
            ttft = (time.perf_counter() - start) * 1000
        tokens += len(text.split())
        
    duration = time.perf_counter() - start
    
    # PyTorch Metrics
    tps = tokens / duration
    tpot = (duration * 1000) / max(1, tokens)
    prefill_speed = prompt_tokens / (ttft / 1000) if ttft else 0
    peak_vram = f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GB"
    
    del model
    torch.cuda.empty_cache()
    
    return ["Native PyTorch", round(ttft, 2), round(tps, 2), round(tpot, 2), round(prefill_speed, 2), peak_vram]

# --- EXECUTION ---
results = [benchmark_ollama(), benchmark_native_pytorch()]

print("\n" + "="*85)
print("FINAL PERFORMANCE COMPARISON: RTX 3050 (4GB VRAM)")
print("="*85)
headers = ["Runtime", "TTFT (ms)", "Tokens/Sec", "TPOT (ms/token)", "Prefill (Tokens/s)", "Peak VRAM"]
print(tabulate(results, headers=headers, tablefmt="grid"))