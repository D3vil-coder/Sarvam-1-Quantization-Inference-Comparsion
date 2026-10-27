# Sarvam-1 inference on a 4 GB GPU

An exploratory project studying quantization and inference for [Sarvam-1](https://huggingface.co/sarvamai/sarvam-1) on an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM. The repository contains benchmarking scripts for a TensorRT-LLM Executor engine and for Ollama and PyTorch.

The original experiment explored W4A16 AWQ quantization and a TensorRT-LLM engine with batch size 1, a maximum input length of 512, and a maximum sequence length of 1,024. The original notes describe using Docker in WSL2. The quantization and engine-build scripts, checkpoints, engine files, dependency versions, and raw benchmark logs are not included here.

**Status:** The historical results below are self-reported observations, not a validated comparison between runtimes. The scripts use different timing, token-counting and memory methods, and their model artifacts and generation settings have not been shown to match. The earlier 50% speedup claim is therefore not established.

## Repository

- [`trt_benchmark.py`](trt_benchmark.py) uses the TensorRT-LLM Python bindings for the C++ Executor, runs three synchronous warmups, and issues one streaming request. It reads an engine from `/models/sarvam-1-engine` and tokenizer files from `/models/sarvam-1`. The script requests up to 150 generated tokens and sets the KV cache free-GPU-memory fraction to 0.3.
- [`Ollama_Pytorch_benchmark.py`](Ollama_Pytorch_benchmark.py) sends requests to a local Ollama server for `mashriram/sarvam-1` and loads a local model with Transformers for a PyTorch run. Its local Windows model path is hard-coded and must be adapted. It performs warmups and requests up to 100 new tokens in the PyTorch path.

These scripts require the appropriate model artifacts and compatible installations of PyTorch, Transformers, and their respective runtimes. The combined script also imports AutoAWQ, Requests and tabulate and applies compatibility monkey patches. Version pins and an end-to-end setup have not been verified, so the repository does not currently provide a guaranteed reproduction command.

## Historical observations

The prior README reported the following values for the RTX 3050 laptop setup. The raw outputs and full configurations are not available in this repository, so the numbers cannot currently be independently checked. They are retained as a record of that report and **must not be treated as an apples-to-apples performance ranking**.

| Value as previously reported | PyTorch | Ollama | TensorRT-LLM |
| --- | ---: | ---: | ---: |
| Throughput | 1.15 TPS | about 58 TPS | 86.77 TPS |
| Time to first token | 4,301.81 ms | about 30 ms | 22.75 ms |
| Time per output token | about 866 ms | about 17 ms | 11.45 ms |
| Memory footprint | over 4.8 GB | OS-managed | 3.03 GB |

## Measurement limitations

- The TensorRT-LLM script divides its counted output tokens by request wall time. The Ollama script divides `eval_count` by Ollama's `eval_duration`. These denominators describe different intervals.
- The PyTorch path counts whitespace-separated text fragments from the streamer, not model tokens. Its reported throughput and per-token time should not be interpreted as tokenizer-token metrics.
- The Ollama field labeled time to first token comes from `prompt_eval_duration`, not a measured first streamed token. The TensorRT-LLM script records the first response time, which requires validation as a first-token measurement.
- The Ollama memory value in the script is a hard-coded estimate. The TensorRT-LLM script samples device-wide used memory before and after inference; the latter is not a continuously measured peak. PyTorch uses its allocator's peak counter after resetting it. These are different memory measurements.
- The TensorRT-LLM script adds token IDs from each response. Its interpretation of streamed IDs as new tokens should be checked against the installed runtime's response semantics. The cache fraction is a configuration value, not a demonstrated hard VRAM cap.\n- Output lengths, prompt formatting, checkpoint or quantization artifacts, and runtime settings require alignment before a comparative benchmark. The script does not evaluate output quality, so it cannot establish that quantization preserves quality.

## Next steps

Record exact model revisions, artifacts, versions, GPU and container configuration. Run the same prompts and generation limits through each backend, count tokenizer output tokens consistently, and measure wall time, first-token latency, decode throughput and peak GPU memory with the same definitions. Save raw results and repeat runs, then evaluate quality separately from speed and memory.
