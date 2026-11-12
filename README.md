# danieLLM

A small LLM inference engine for Qwen3-0.6B, in Python and CUDA C++. The goal of this project is for me to get a better grasp of what it's like to build the core of a modern serving engine myself. 

The engine (model, KV cache, block manager, scheduler) is Python on PyTorch. Performance-critical ops are custom CUDA kernels built as PyTorch extensions with torch.utils.cpp_extension, and large matmuls go through cuBLAS. This is the same split that engines such as vLLM and SGLang use. Weights are loaded directly from safetensors, and Hugging Face's modeling code is only used to produce reference outputs.

Status: Writing PyTorch forward pass!


