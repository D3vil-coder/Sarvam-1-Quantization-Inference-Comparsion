import time
import torch
from tensorrt_llm.bindings.executor import Executor, ExecutorConfig, KvCacheConfig
from tensorrt_llm.bindings.executor import Request, SamplingConfig, ModelType
from transformers import AutoTokenizer

def get_true_vram_gb():
    # Gets the actual physical VRAM used on the GPU, not just PyTorch's slice
    free_mem, total_mem = torch.cuda.mem_get_info()
    return (total_mem - free_mem) / (1024**3)

def main():
    engine_dir = "/models/sarvam-1-engine"
    tokenizer_dir = "/models/sarvam-1"

    print("\n>>> Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    print(">>> Booting Raw C++ Executor...")
    kv_config = KvCacheConfig(free_gpu_memory_fraction=0.3)
    executor_config = ExecutorConfig(kv_cache_config=kv_config)
    executor = Executor(engine_dir, ModelType.DECODER_ONLY, executor_config)

    prompt = """Explain the concept of 'Backpropagation' in Deep Learning using a simple analogy in Hindi. 
Then, list 3 reasons why it is essential for training Large Language Models."""

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        formatted_prompt = prompt

    input_ids = tokenizer.encode(formatted_prompt)
    pad_token = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    
    sampling_config = SamplingConfig(temperature=0.0)

    print("\n>>> Warming up C++ Engine (Synchronously)...")
    warmup_ids = tokenizer.encode("hi")
    for _ in range(3):
        w_req = Request(
            input_token_ids=warmup_ids,
            max_tokens=5,
            streaming=False,
            sampling_config=sampling_config,
            end_id=tokenizer.eos_token_id,
            pad_id=pad_token
        )
        w_id = executor.enqueue_request(w_req)
        # THE FIX: We must actively wait for the warmup to finish before firing the next one
        while True:
            resps = executor.await_responses(w_id)
            if resps:
                res = resps[-1].result if hasattr(resps[-1], 'result') else resps[-1].get_result()
                if res.is_final:
                    break

    # Baseline VRAM before benchmark
    vram_before = get_true_vram_gb()

    req = Request(
        input_token_ids=input_ids,
        max_tokens=150,
        streaming=True,
        sampling_config=sampling_config,
        end_id=tokenizer.eos_token_id,
        pad_id=pad_token
    )

    print(f">>> Running Benchmark... (VRAM footprint: {vram_before:.2f} GB)")
    start_time = time.perf_counter()

    request_id = executor.enqueue_request(req)

    ttft = None
    tokens = 0
    output_ids = []
    is_done = False

    while not is_done:
        responses = executor.await_responses(request_id)
        for response in responses:
            if response.has_error():
                print(f"\n[!] C++ Executor Error: {response.error_msg}")
                is_done = True
                break
            
            res = response.result if hasattr(response, 'result') else response.get_result()
            
            if ttft is None:
                ttft = time.perf_counter() - start_time
            
            out_tokens = res.output_token_ids
            if len(out_tokens) > 0 and isinstance(out_tokens[0], list):
                new_tokens = out_tokens[0]
            else:
                new_tokens = out_tokens

            output_ids.extend(new_tokens)
            tokens += len(new_tokens)
            
            if res.is_final:
                is_done = True
                break

    duration = time.perf_counter() - start_time
    
    # Calculate True Metrics
    ttft_ms = ttft * 1000 if ttft else 0
    tps = tokens / duration if duration > 0 else 0
    tpot_ms = ((duration - ttft) * 1000) / max(1, tokens - 1) if ttft else 0
    peak_vram = get_true_vram_gb()

    print("\n" + "="*65)
    print("FINAL C++ EXECUTOR PERFORMANCE: RTX 3050 (4GB VRAM)")
    print("="*65)
    print(f"TTFT (Time to First Token): {ttft_ms:.2f} ms")
    print(f"TPOT (Time Per Output):     {tpot_ms:.2f} ms/token")
    print(f"Throughput:                 {tps:.2f} tokens/sec")
    print(f"Total Time:                 {duration:.2f} seconds")
    print(f"Hardware VRAM Footprint:    {peak_vram:.2f} GB")
    print("-" * 65)
    
    full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"[TRT-LLM Output Preview]:\n{full_text[:300]}...\n")

if __name__ == '__main__':
    main()
