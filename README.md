# cppDL

A high-performance deep learning inference library written in C++ from scratch with **NO** external dependencies.

## Performance Summary

| Benchmark | Configuration | Result |
|-----------|---------------|--------|
| GPT-2 Inference | no KV cache, 50 tokens | **35.43 tok/s** (PyTorch CPU: 19 tok/s, +86%) |
| GEMM (FP32) | 4096x4096, 8192x8192, 16384x16384 | **825 GFLOP/s** sustained |
| Multi-Head Attention | seq=512, embed=8192, heads=8 | **343 GFLOP/s** (413.0ms) |
| Single-Head Attention | seq=512, embed=8192, heads=1 | **359 GFLOP/s** (394.9ms) |

### Some more important benchmarks can be found below in the **Benchmarks** section.

*Run `multi_head_attention_test()`, `gemm_test()`, `gpt2_test()` or `attention_test` in the test bed to reproduce these benchmarks.!*

## Features

### Core Infrastructure
- **Arena-based memory allocator**: Deterministic allocation patterns, zero malloc during inference
- **Tensor API**: Shape/stride management, broadcasting, contiguous/non-contiguous view support
- **Safetensor parser**: Direct loading of Hugging Face model weights without Python dependencies
- **BPE tokenizer**: Vocabulary training, save/load, encoding/decoding with merge rules

### Neural Network Operations
- **Layers**: Linear (optimized GEMM backend), ReLU, Sigmoid, GELU, Softmax, LayerNorm
- **Attention**: Single-head and Multi-head Attention with or without causal masking
- **Models**: GPT-2 inference implementation

### CRUSHBLAS Backend
- **Microkernel**: 6x16, 4x8 and register-blocked FMA kernel for AVX2/AVX-512
- **Cache blocking**: 256x256x256 tiles optimized for L2/L3 cache hierarchy
- **Matrix packing**: Contiguous memory layout for optimal cache line utilization
- **Parallelization**: OpenMP with `collapse(2)` scheduling across M and N dimensions
- **Transpose-aware**: Native support for transposed operands without explicit transpose

## Building

### Prerequisites
- C++23 compiler (GCC 13+, Clang 16+, MSVC 2022+)
- CMake 3.20+
- Ninja (Linux) or MinGW (Windows)
- OpenMP (optional, for parallelization)
- CPU with AVX2 support (AVX-512 recommended for best performance)

### Linux
```bash
cmake --preset linux-release
./build.sh linux-release linux-release-build ON OFF OFF
./run.sh
```

### Windows
```bash
cmake --preset mingw-release
./build mingw-release mingw-release-build 1 0 0
./run
```

### Build Script Arguments
```
./build.sh <preset> <build-preset> <AVX> <BLAS> <DEBUG>
```
| Argument | Values | Description |
|----------|--------|-------------|
| `AVX` | ON/OFF (1/0 on Windows) | Enable AVX2/AVX-512 SIMD |
| `BLAS` | ON/OFF (1/0 on Windows) | Enables Multi-threaded and packed BLAS (not recommended for now) |
| `DEBUG` | ON/OFF (1/0 on Windows) | Enable debug assertions and logging |

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_AVX256` | ON | Enable AVX2/AVX-512 intrinsics |
| `DEBUG_CPPDL` | OFF | Enable debug macro and assertions |
| `USE_VULKAN` | OFF | Enable Vulkan compute backend (WIP) |
| `USE_OPENGL` | OFF | Enable OpenGL compute backend (WIP) |

## Usage Examples

### Headers
```cpp
#include "neural_core/neural_network.hpp"
#include "tensor_core/tensor.hpp"
#include "attention_core/attention.hpp"
#include "tokenizer_core/tokenizer.hpp"
#include "model_core/gpt2.hpp"
#include "safetensor_core/safetensor_reader.h"
// ... and so on
```

### Neural Network Inference
```cpp
#include "neural_core/neural_network.hpp"

// Define network architecture
neural::nn network;
network.add_linear(16384, 8192);
network.add_relu();
network.add_linear(8192, 4096);
network.add_relu();
network.add_linear(4096, 1024);
network.add_sigmoid();

// Allocate persistent memory for weights
size_t mem_size = network.mem_reqs();
neural::alloc_pool persistent_arena(mem_size * sizeof(float));
network.init(persistent_arena);

// Allocate temporary memory for activations
neural::alloc_pool temp_arena(mem_size * sizeof(float));

// Prepare input tensor
neural::neural_view input;
input.tensor.tensor_data      = input_data;
input.tensor.shape.ndim       = 2;
input.tensor.shape.dims[0]    = batch_size;
input.tensor.shape.dims[1]    = 16384;
input.tensor.shape.strides[0] = 16384;
input.tensor.shape.strides[1] = 1;

// Run inference
neural::neural_view output = network.forward(input, temp_arena);
```

### Multi-Head Attention
```cpp
#include "attention_core/attention.hpp"

size_t embed_dim = 8192;
size_t num_heads = 8;
size_t seq_len   = 512;

atten::multi_head_attention attn(embed_dim, num_heads);

// Initialize and load weights
memory::neural_arena persistent_arena(/* weight memory */);
attn.init(persistent_arena);
// If causal-masking is not wanted, simply set all bias data pointers to 0. 
attn.load_weights(wq_data, wk_data, wv_data, wo_data, bq_data, bk_data, bv_data, bo_data);

// Prepare input tensor [seq_len, embed_dim]
tens::tensor input;
input.tensor_data      = input_data;
input.shape.ndim       = 2;
input.shape.dims[0]    = seq_len;
input.shape.dims[1]    = embed_dim;
input.shape.strides[0] = embed_dim;
input.shape.strides[1] = 1;

// Forward pass
memory::neural_arena temp_arena(/* activation memory */);
tens::tensor output = attn.forward(input, temp_arena);
```

### Direct GEMM Operations
```cpp
#include "CRUSHBLAS_MODULE/core/BLAS/level3/level3.hpp"

// Define matrix views
level3::mat_ops_view A = {
    .row_view = M,
    .col_view = K,
    .leading_dimension = K,
    .data_view = a_data
};

level3::mat_ops_view B = {
    .row_view = K,
    .col_view = N,
    .leading_dimension = N,
    .data_view = b_data
};

level3::mat_ops_view C = {
    .row_view = M,
    .col_view = N,
    .leading_dimension = N,
    .data_view = c_data
};

// C = alpha * A @ B + beta * C
level3::blas::crush_gemm(
    level3::transpose_gemm::no_transpose,
    level3::transpose_gemm::no_transpose,
    A, B,
    1.0f,  // alpha
    0.0f,  // beta
    C
);
```

### BPE Tokenization
```cpp
#include "tokenizer_core/tokenizer.hpp"

bpe::bpe_tokenizer tokenizer;

// Train on corpus
std::string corpus = /* load training text */;
size_t num_merges = 1000;
tokenizer.train(corpus, num_merges);

// Encode text to token IDs
std::vector<bpe::g_tokenid> tokens = tokenizer.encode("Hello world");

// Decode token IDs back to text
std::string decoded = tokenizer.decode(tokens);

// Save/load vocabulary
tokenizer.save_model("vocab.txt", "merges.txt");
tokenizer.load_model("vocab.txt", "merges.txt");
```

## Benchmarks

All benchmarks performed on AMD Ryzen 9 9950X3D and an Intel i9-9900K with OpenMP parallelization enabled.

### GEMM Performance (FP32)

| Matrix Size | GFLOP | Time | Throughput |
|-------------|-------|------|------------|
| 4096 x 4096 | 137.4 | 0.167s | **825 GFLOP/s** |
| 8192 x 8192 | 1099.5 | 1.33s | **825 GFLOP/s** |
| 16384 x 16384 | 8796.1 | 10.66s | **825 GFLOP/s** |

### Attention Performance (FP32)

| Configuration | Params | Time | GFLOP | GFLOP/s |
|---------------|--------|------|-------|---------|
| Single-Head Attention | seq=512, embed=8192, heads=1 | 394.9ms | 141.8 | **359 GFLOP/s** |
| Multi-Head Attention | seq=512, embed=8192, heads=8 | 413.0ms | 141.8 | **343 GFLOP/s** |

*Attention FLOP calculation: 3 × seq × embed² (QKV projections) + 2 × seq² × embed (attention scores) + seq × embed² (output projection)*

### GPT-2 Inference (Small, 124M params, FP32)

| Metric | Value |
|--------|-------|
| Tokens/sec (no KV cache) | **35.43 tok/s** |
| GFLOP/s (average) | ~535 GFLOP/s |
| PyTorch CPU baseline | 19 tok/s |
| Speedup vs PyTorch | **+86%** |

### PyTorch Comparison Benchmark
```python
torch.set_num_threads(os.cpu_count())
#  torch.set_num_interop_threads(16) or 32 or 1 or however many your CPU can support
torch.set_num_interop_threads(32)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()
# ... prompt
num_new_tokens = 50

# Warmup
with torch.inference_mode():
    for _ in range(3):
        tokens = prompt.copy()
        for _ in range(10):
            x = torch.tensor([tokens], dtype=torch.long)
            logits = model(x, use_cache=False).logits[0, -1]
            tokens.append(int(torch.argmax(logits)))

# Benchmark
with torch.inference_mode():
    tokens = prompt.copy()
    start = time.perf_counter()
    for _ in range(num_new_tokens):
        x = torch.tensor([tokens], dtype=torch.long)
        logits = model(x, use_cache=False).logits[0, -1]
        tokens.append(int(torch.argmax(logits)))
    elapsed = time.perf_counter() - start

# Result: 19 tok/s
```

## Project Structure

```
cppDL/
├── core/
│   ├── include/
│   │   ├── CRUSHBLAS_MODULE/          # Custom BLAS backend (submodule)
│   │   │   └── core/
│   │   │       └── BLAS/
│   │   │           └── level3/        # GEMM kernels
│   │   ├── neural_core/               # Neural network layers
│   │   ├── tensor_core/               # Tensor operations
│   │   ├── attention_core/            # Attention mechanisms
│   │   ├── tokenizer_core/            # BPE tokenizer
│   │   ├── model_core/                # Model implementations (GPT-2)
│   │   ├── safetensor_core/           # Weight loading
│   │   ├── memory_core/               # Arena allocators
│   │   └── logger_core/               # Logging utilities
│   └── impl/                          # Implementation files
│       ├── neural_core_impl/
│       ├── tensor_core_impl/
│       ├── attention_core_impl/
│       ├── tokenizer_core_impl/
│       ├── model_core_impl/
│       └── safetensor_core_impl/
├── test_bed/
│   └── src/                           # Tests and benchmarks
├── CMakeLists.txt
├── CMakePresets.json
├── build.sh / build.bat
└── run.sh / run.bat
```

## Roadmap

### In Progress
- [ ] Custom Vulkan compute shader backend for GPU kernel execution
- [ ] Fused Tensor Ops, Fused Neural Network Ops, Fused Attention
- [ ] Different Attention mechanisms
- [ ] JIT/Compiler for automatic CPU/GPU code generation
- [ ] Auto-grad wrapper with operator fusion for training support

### Planned
- [ ] KV caching for faster autoregressive generation
- [ ] INT8/FP16 quantization support
- [ ] ONNX model loading
- [ ] Additional architectures (LLaMA, Mistral, etc.)
- [ ] Custom user-level GPU drivers for direct kernel dispatch

## Technical Notes

### Memory Management
cppDL uses a two-arena allocation strategy:
1. **Persistent arena**: Holds model weights, allocated once at initialization
2. **Temporary arena**: Holds intermediate activations, reset between forward passes

This eliminates malloc/free overhead during inference and ensures predictable memory behavior.

### SIMD Optimization
The CRUSHBLAS GEMM kernel uses:
- 6x16 microkernel with 12 YMM/ZMM accumulator registers
- Register blocking to minimize load/store operations in the inner loop
- Software prefetching for next cache line
- Loop unrolling (4x) in the K dimension

### Why Faster Than PyTorch?
The 86% speedup over PyTorch CPU inference comes from:
1. **Static memory allocation**: No Python object overhead or dynamic allocation
2. **Fused operations**: Reduced memory bandwidth pressure
3. **Simplified dispatch**: Direct function calls vs. PyTorch's dynamic dispatch
4. **Cache-friendly layouts**: Data arranged for sequential access patterns

Note: PyTorch's BLAS backend (OpenBLAS/MKL) achieves higher raw GEMM throughput. The end-to-end advantage comes from system-level optimizations, not kernel performance.

## License

MIT

## Acknowledgments

- **CRUSHBLAS**: Custom BLAS library developed as part of this project
- **GPT-2**: Weights loaded via Hugging Face safetensors format
- Architecture inspired by llama.cpp, ggml, and tinygrad
