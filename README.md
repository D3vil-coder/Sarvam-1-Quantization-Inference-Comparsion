# High-Performance Edge Inference: Optimizing Sarvam-1 for 4GB VRAM Constraints

## 🎯 Intent and Scope
The objective of this project is to achieve state-of-the-art inference performance for `sarvamai/sarvam-1` on a severely constrained **NVIDIA RTX 3050 Laptop GPU (4GB VRAM)**. 

Standard high-level deployments often fail on 4GB cards due to unmanaged KV cache allocation and framework overhead. This experiment demonstrates a rigorous optimization pipeline—moving from a failing PyTorch baseline to a highly efficient, bare-metal **TensorRT-LLM C++ Executor**. 

The engineering goals were:
1. **Low Latency:** Achieving sub-25ms Time-To-First-Token (TTFT).
2. **Memory Fencing:** Hard-capping the VRAM footprint at ~3.0 GB to ensure system stability.
3. **Throughput Maximization:** Outperforming consumer runtimes like `llama.cpp` by leveraging NVIDIA’s low-level C++ bindings.

---

## 💻 Hardware & Environment
* **GPU:** NVIDIA GeForce RTX 3050 (Laptop)
* **VRAM:** 4.00 GB Physical
* **Platform:** Docker on WSL2 (Windows Subsystem for Linux)
* **Model:** `sarvamai/sarvam-1` (2B Parameters)

---

## 🛠️ Optimization Methodology

### Step 1: W4A16 AWQ Quantization
The original FP16 weights of Sarvam-1 exceed 5GB, making a 4GB deployment physically impossible. 
* **Action:** Utilized NVIDIA `modelopt` to perform **INT4 Activation-Aware Weight Quantization (AWQ)**.
* **Result:** Reduced the model weight footprint sufficiently to fit within 4GB, while maintaining the model's linguistic integrity.

### Step 2: The Native PyTorch Bottleneck
Initial testing used the Hugging Face `transformers` library to run the **INT4 AWQ** quantized model.
* **Observation:** Despite the 4-bit weights, the native PyTorch/Transformers implementation attempted to reserve **4.8+ GB of VRAM** for the KV cache and workspace. 
* **Result:** Significant performance degradation due to memory swapping, resulting in a throughput of only **~1.15 TPS**. This proved that quantization alone is insufficient without low-level memory management.

### Step 3: Establishing the `llama.cpp` Baseline
To benchmark against standard consumer-grade optimization, the model was run via Ollama (`llama.cpp`).
* **Result:** Success. 
* **Metrics:** Delivered a stable **~58.00 TPS** with a TTFT of **~30.00 ms**, establishing a baseline for localized inference.

### Step 4: Engineering the TensorRT-LLM C++ Solution



#### Phase A: AOT Engine Compilation
The AWQ checkpoint was compiled into a static C++ execution engine. To survive the build on 4GB VRAM, the engine was "fenced" with strict architectural limits:
* `max_batch_size`: 1
* `max_input_len`: 512
* `max_seq_len`: 1024

#### Phase B: Bypassing Framework Overhead
Attempts to use high-level TRT-LLM wrappers (the `LLM()` Python class or `trtllm-serve`) failed on the RTX 3050. These "enterprise-ready" abstractions default to massive runtime allocations (e.g., 2048 batch slots) that triggered OS-level OOM kills in the WSL2 container. 

#### Phase C: Bare-Metal C++ Implementation
To achieve the target metrics, the Python high-level API was discarded in favor of the **`tensorrt_llm.bindings.executor.Executor`**.
* **Synchronous Warmup:** Implemented a blocking warmup sequence to ensure the C++ graph was fully primed before benchmarking.


---

## 📊 Comparative Performance Results

The following metrics were captured during synchronous, blocking inference:

| Metric | PyTorch (INT4 AWQ) | `llama.cpp` (Ollama) | TRT-LLM C++ (Executor) |
| :--- | :--- | :--- | :--- |
| **Throughput (TPS)** | 1.15 | ~58.00 | **86.77** *(~50% Speedup)* |
| **Time To First Token (TTFT)** | 4301.81ms | ~30.00 ms | **22.75 ms** |
| **Time Per Token (TPOT)** | ~866ms | ~17.00 ms | **11.45 ms** |
| **Total Memory Footprint** | 4.80+ GB | Dynamic(OS) | **3.03 GB (Locked)** |

---

## 🚀 Conclusion
By stripping away the abstraction layers and targeting the **TensorRT-LLM C++ Executor**, this project achieved a **~50% throughput increase** over standard optimized runtimes on identical hardware. This confirms that for 4GB VRAM edge devices, direct C++ bindings are the only way to unlock the full potential of NVIDIA hardware, delivering an impressive **86.77 TPS** while maintaining a stable 1GB VRAM safety margin for the host system.