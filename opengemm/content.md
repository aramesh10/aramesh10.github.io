# Introducing OpenGEMM

```python
import opengemm as og

c = og.gemm(a, b)                    # C[M, N] = A[M, K] @ B[N, K].T
c = og.gemm(a, b, sfa, sfb)          # block-scaled: nvfp4, mxfp8, mxfp4

og.emit_kernel(a, b, file="kernel.cu")      # emits .cu/.cuh for this shape
c = og.run_kernel("k.cu", a, b)             # compiles emitted kernel and runs it
```

OpenGEMM is a library containing GEMM kernels implemented in CUDA for B200. After autotuning, OpenGEMM is able to emit the raw CUDA `.cu` and `.cuh` files optimized for a given shape for any datatype, not requiring any other libraries or abstraction layers. 

## Motivation 

I have had a few projects now where having a cuBLAS-level performance GEMM kernel for a particular floating point type and shape implemented in CUDA would have been useful. 

GEMM kernels are ubiquitous and there aren't many repositories with a pure CUDA and PTX implementation of GEMM given a particular shape and datatype for B200s without a library behind it, often behind a DSL abstraction, or code within a blog post. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/tree/main/deep_gemm/include/deep_gemm/impls) and ThunderKittens are both great projects, but the library abstraction required a human or agent to manually parse through each file to get the underlying CUDA implementation. While agents automate this, I saw this as a wasteful cost of tokens and time.

My initial ambition was to recreate cuBLAS for the major devices, including other NVIDIA GPUs like H100 and A100, but also AMD's, Google's TPUs, and Apple's Metal. This is a larger undertaking than anticipated, even with agents, and I have a limited bandwidth, so I decided to release it in its current state, which supports GEMM kernels for most datatypes on B200.

### Agents

I provide agents reference implementations from blog posts and GitHub repositories I find online when optimizing a kernel. `emit_kernel` was motivated from this workflow as I was working on these kernels. Especially [as tool calls are transitioning to become Python code snippets](https://www.anthropic.com/engineering/code-execution-with-mcp), it seemed useful to have a function that will emit the CUDA files for these agents to use. Agents writing CUDA can save tokens and time using an existing optimized solution instead of rederiving the kernel from scratch each time. 

### Future

I would like to support GEMM implementations for other devices, in particular for the following:

- NVIDIA: B100s, B300s, H100s, H200s
- AMD: MI300X, MI325X, MI350X, MI355X
- Google: TPU v6e, TPU7x

I am also interested in implementing and emitting kernels in multiple languages, including CUTLASS, CuTe DSL, HIP, Pallas, and TileLang. In addition, adding support for different epilogues including for collective communication, similar to [QuACK](https://github.com/Dao-AILab/quack/tree/main) is one of the first areas I would expand this towards.

## Other thoughts

### Open source cuBLAS

cuBLAS is closed source as of 09/13/2026. Prior to agents, an open-source cuBLAS library in CUDA would have been an immense effort to implement for an outsider, giving NVIDIA a competitive advantage. With agents, these closed source libraries can quickly become democratized. Although Jensen believes the [future of AI is open and proprietary](https://blogs.nvidia.com/blog/ai-future-open-and-proprietary/), cuBLAS operates at a low part of the stack, which has a large downstream impact on the tech built on it. cuBLAS, as with any closed source library, will become open source with or without NVIDIA's permission. GEMM is already a solved problem and cuBLAS has the implementations already. Open-sourcing cuBLAS would accelerate the advancement of model serving, pushing token costs down and making AI more accessible to the token poor. It benefits NVIDIA by entrenching consumers into their hardware with the software stack the open-source community would inevitably develop from it. It's beneficial for society and NVIDIA to make cuBLAS open-source as soon as possible, but its current business plan does not seem to reflect an understanding of the long-term changes in software from the advancement of these models.