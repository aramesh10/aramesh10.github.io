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

I made OpenGEMM as I had many projects where having a cuBLAS-level performance GEMM kernel for a particular floating point type and shape implemented in CUDA would have been useful. 

GEMM kernels are ubiquitous and there isn't any repository with a pure CUDA and PTX implementation of GEMM for a particular shape and datatype for B200s without a library abstraction behind it, a DSL implementation, or code buried in a blog post. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/tree/main/deep_gemm/include/deep_gemm/impls) has open source implementations of GEMM and more, but there was a library abstraction behind it, requiring a human or agent to manually parse through each file. While agent performance is improving, I saw the need for GEMM kernels implemented in CUDA as ubiquitous and solvable without the need for agents so I started this repository. 

My initial ambition was to recreate cuBLAS for the major devices, including other NVIDIA GPUs like H100 and A100, but also AMD's, Google's TPUs, and Apple's Metal. However, this requires a lot more effort and I do think it is useful in its current state.

### Agents

Often when asking an agent to iterate on a kernel, I provide it reference implementations from blog posts and GitHub repositories I find online. `emit_kernel` was motivated from this workflow as I was working on these kernels. Especially [as tool calls are transitioning to become Python code snippets](https://www.anthropic.com/engineering/code-execution-with-mcp), it seemed useful to have a function that will emit the CUDA files for these agents to use. Agents writing CUDA can save tokens and time using an existing optimized solution instead of rederiving the kernel from scratch each time. 

### Future

I would like to support GEMM implementations for other devices, in particular for the following:

- NVIDIA: B100s, B300s, H100s, H200s
- AMD: MI300X, MI325X, MI350X, MI355X
- Google: TPU v6e, TPU7x

I am also interested in implementing and emitting kernels in multiple languages, including CUTLASS, CuTe DSL, HIP, Pallas, and TileLang. In addition, adding support for different epilogues including for collective communication, similar to [QuACK](https://github.com/Dao-AILab/quack/tree/main) is one of the first areas I would expand this towards.

## Other thoughts

### Open source cuBLAS

cuBLAS is closed source as of 09/13/2026. Prior to agents, an open-source cuBLAS library in CUDA would have been an immense effort to implement for an outsider, giving NVIDIA a competitive advantage and aligning with its business plan. With agents, these closed source libraries can quickly become democratized. Although Jensen believes the [future of AI is open and proprietary](https://blogs.nvidia.com/blog/ai-future-open-and-proprietary/) as AI is more than just a model and each industry handles its own unique challenges, cuBLAS does play an important role at a very low level of the AI stack. Open-sourcing cuBLAS can be beneficial to NVIDIA's long-term position as it would help to entrench consumers into their hardware if the software stack grows from it while pushing society's Pareto frontier by accelerating the rate at which technology improves. I do believe NVIDIA can help accelerate development of ML serving while remaining competitive from a business standpoint and that it is more beneficial to do it sooner than later. Unless NVIDIA pivots into developing its own models with its own hardware.