# 1.Profilling and Benchmarking
- 测试的 GPU 为 4060 LAPTOP

## End-to-End Benchmarking
- 因为显存不够，所以仅对 small, medium, large 的模型做了测试，仅对 small, medium的模型做了 forward+backward 测试

- (b)：经过 5 次预热后，标准差极小，例如 Small 模型的前向传播耗时为 27.84 ± 0.37 ms，完整步骤（前向+反向）耗时为 95.50 ± 1.83 ms。
- (c)：无预热时，由于首次运行的冷启动开销（如 CUDA 内核即时编译和内存分配），数据波动极大；进行 1-2 次预热时，初始化可能尚未完全完成，标准差小很多，但是与预热五次的还有一点差距。

- warmup=5, steps=10:

| Model Size   |   Context Length | Mode         | Time (ms)       |
|:-------------|-----------------:|:-------------|:----------------|
| small        |              128 | Forward Only | 27.84 ± 0.37     |
| small        |              256 | Forward Only | 63.47 ± 1.95     |
| medium       |              128 | Forward Only | 92.29 ± 3.95     |
| medium       |              256 | Forward Only | 171.68 ± 6.60    |
| large        |              128 | Forward Only | 179.11 ± 1.37     |
| large        |              256 | Forward Only | 8542.52 ± 1139.16(OOM) |

| Model Size   |   Context Length | Mode      | Time (ms)      |
|:-------------|-----------------:|:----------|:---------------|
| small        |              128 | Full Step | 95.50 ± 1.83   |
| small        |              256 | Full Step | 175.00 ± 5.07  |
| medium       |              128 | Full Step | 300.17 ± 9.70  |
| medium       |              256 | Full Step | 553.07 ± 14.19 |
| large        |              128 | Full Step | 1462.43 ± 41.08(OOM) |


- warmup=0, steps=10

| Model Size   |   Context Length | Mode         | Time (ms)      |
|:-------------|-----------------:|:-------------|:---------------|
| small        |              128 | Forward Only | 62.93 ± 101.51 |
| small        |              256 | Forward Only | 70.64 ± 13.82  |
| medium       |              128 | Forward Only | 100.19 ± 4.06  |
| medium       |              256 | Forward Only | 184.73 ± 12.47 |
| large        |              128 | Forward Only | 241.77 ± 100.90   |
| large        |              256 | Forward Only | 10662.79 ± 628.70(OOM) |

| Model Size   |   Context Length | Mode      | Time (ms)       |
|:-------------|-----------------:|:----------|:----------------|
| small        |              128 | Full Step | 134.06 ± 123.51 |
| small        |              256 | Full Step | 194.87 ± 22.13  |
| medium       |              128 | Full Step | 319.49 ± 20.86  |
| medium       |              256 | Full Step | 572.62 ± 38.22  |
| large        |              128 | Full Step | 1546.02 ± 146.03(OOM) |

- warmup=1, steps=10 (仅进行部分测试)

| Model Size   |   Context Length | Mode      | Time (ms)      |
|:-------------|-----------------:|:----------|:---------------|
| small        |              128 | Full Step | 96.92 ± 2.85   |
| small        |              256 | Full Step | 185.39 ± 9.61  |
| medium       |              128 | Full Step | 304.94 ± 10.20 |
| medium       |              256 | Full Step | 566.98 ± 26.15 |


## Nsight Systems Profiler
- 测试时，small: context-length=1024时OOM；medium: context-length=512时OOM；large: context-length=128时OOM

- (a) 总体而言，模型越大，python代码相比 nvtx 的差距越大，即越慢
- (b) 该题使用 medium, context-length=256时的数据
    - forward pass，耗时最长的 kernel为：ampere_sgemm_128x64_tn 以及 ampere_sgemm_64x64_tn，各花费约 21.142ms(18次), 21.051ms(44次)
    - forward + backward：有四个 kernel 耗时相近，如下表

|Time	|Total| Time|	Instances|	
|:-----|------|----:|:-----------|
|16.4%	|93.842 ms|	143|	ampere_sgemm_128x64_nn|
|16.3%	|93.573 ms|	73|	ampere_sgemm_64x64_nt|
|11.9%	|68.230 ms|	49|	ampere_sgemm_64x64_tn|
|11.4%|	65.531 ms|	120|	ampere_sgemm_128x64_tn|
    
- (c) 除矩阵乘法外，各种逐元素操作（Element-wise）的 Kernel 也占据了一定运行时间。包括：
    - `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)`: 占据 6.0%
    - `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`: 占据 3.9%

- (d) 矩阵乘法部分，和 (b) 一致，非矩阵乘法部分前几个如下
    - `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)`: 7.6%
    - `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`: 5.2%
    - `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`: 4.3%
    - `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)`: 2.4%
    - `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)`: 2.2%
    - `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`: 2.0%

- (e) 基于medium, ctx=256的数据，对比如下表：
    - 明显发现 softmax的 FLOPs 比 matmul 小很多，但是runtimes却没有明显差距
    - matmul：$4·b·n^2·d$；softmax: $3⋅b⋅h⋅n^2$
    - b: batch, n:seq_len, d: d_model, h: num_heads

|     | softmax | matmul |
|-----|---------|--------|
|FLOPs/layer|  0.01GFLOPs   |  1.07GFLOPs   |
|runtimes|70.34 ms | 88.08 ms |



- forward pass:

| Model Size   |   Context Length | Time (ms)      |
|:-------------|-----------------:|:---------------|
| small        |              128 | 30.40 ± 5.95   |
| small        |              256 | 57.60 ± 104.59 |
| small        |              512 | 61.04 ± 121.55 |
| medium       |              128 | 85.39 ± 120.07 |
| medium       |              256 | 89.74 ± 138.14 |

- backward pass:

| Model Size   |   Context Length | Time (ms)      |
|:-------------|-----------------:|:---------------|
| small        |              128 | 66.12 ± 2.21   |
| small        |              256 | 153.01 ± 21.13 |
| small        |              512 | 385.47 ± 2.81  |
| medium       |              128 | 240.99 ± 16.38 |
| medium       |              256 | 500.86 ± 14.60 |

- optimizer step:

| Model Size   |   Context Length | Time (ms)   |
|:-------------|-----------------:|:------------|
| small        |              128 | 0.36 ± 0.11 |
| small        |              256 | 0.38 ± 0.12 |
| small        |              512 | 0.44 ± 0.15 |
| medium       |              128 | 0.50 ± 0.12 |
| medium       |              256 | 0.51 ± 0.09 |

total:

| Model Size   |   Context Length | Time (ms)       |
|:-------------|-----------------:|:----------------|
| small        |              128 | 96.89 ± 6.35    |
| small        |              256 | 210.99 ± 106.70 |
| small        |              512 | 446.95 ± 121.58 |
| medium       |              128 | 326.87 ± 121.18 |
| medium       |              256 | 591.11 ± 138.91 |

## Mixed Precesion

- (a) ToyModel
    - parameters：保持FP32
    - Output of FFN: FP16
    - Output of layer norm：FP32
    - logits：FP16
    - Loss：FP32
    - Gradients：FP32

- (b) LayerNorm 中对数值精度最敏感的部分是 均值（mean）和方差（variance）的计算以及归一化（除以标准差）; BF16 具有与 FP32 相同的指数范围，相对FP16 不易发生下溢/溢出

- (c) 混合精度下测试结果如下，warmup=5, steps=10，相比FP32，可见速度快了许多

| Model Size   |   Context Length | Mode         | Time (ms)    |
|:-------------|-----------------:|:-------------|:-------------|
| small        |              128 | Forward Only | 23.18 ± 6.02 |
| small        |              256 | Forward Only | 26.56 ± 0.45 |
| medium       |              128 | Forward Only | 45.29 ± 2.54 |
| medium       |              256 | Forward Only | 80.87 ± 0.92 |
| large        |              128 | Forward Only | 92.46 ± 1.37  |
| large        |              256 | Forward Only | 233.93 ± 4.54 |


| Model Size   |   Context Length | Mode      | Time (ms)     |
|:-------------|-----------------:|:----------|:--------------|
| small        |              128 | Full Step | 61.95 ± 1.27  |
| small        |              256 | Full Step | 94.02 ± 2.03  |
| medium       |              128 | Full Step | 172.92 ± 3.40 |
| medium       |              256 | Full Step | 271.71 ± 6.26 |
| large        |              128 | Full Step | 12234.11 ± 286.27(OOM) |

## Profiling Memory
- 因为显存问题，使用 large 模型
- bf16 与 fp32下 training loop 的 context_length 为 128 时均出现OOM
- 以下为不同条件下的 peak memory

|precision/| 128(infer) | 256(infer) | 512(infer) | 128(train) |
|---------|-------------|------------|------------|------------|
|bf16     |5.53GB       |5.56GB      |5.77GB      |7.56GB      |
|fp32     |3.78GB       |3.87GB      |4.16GB      |7.52GB      |

- (a) 以下分别是 fp32 时的 memory timeline, 分别为 context128-train, context256-forward；可见 forward 的内存占用比较平滑，而 train 的内存占用会有起伏

<img src="image/ctx128-train-fp32.png" alt="128-train" width="500">

<img src="image/ctx256-forw-fp32.png" alt="256-forward" width="500">

- (b) 见上表

- (c) 见上表，相同条件下，mixed precision 确实占用了更多的内存

- (d) 对于 2.7B 模型来说，当 context_length 为 1024 时，activations 的 size 为 10MB
    - $1024 \times 2056 \times 4 / 1024^2$
    - 对于 large 模型，其占用为 5 MB (context_length为128时约为 0.8MB，考虑batch以及layer后为 90MB)


# 2.Optimizing Attention

## benchmark attention:

- seq_len=8196 时即 OOM，表中的 Memory 统计的是反向传播开始前的内存占用

|   d_model |   seq_len | Forward (ms)   | Backward (ms)   | Memory (GB)     |
|----------:|----------:|:---------------|:----------------|:----------------|
|        16 |       256 | 0.28 ± 0.04    | 0.48 ± 0.08     | 0.0203 ± 0.0000 |
|        16 |      1024 | 2.04 ± 0.09    | 5.10 ± 0.23     | 0.0814 ± 0.0000 |
|        16 |      4096 | 34.05 ± 0.73   | 79.71 ± 1.52    | 1.0397 ± 0.0000 |
|        32 |       256 | 0.27 ± 0.02    | 0.48 ± 0.08     | 0.0208 ± 0.0000 |
|        32 |      1024 | 2.06 ± 0.09    | 5.06 ± 0.13     | 0.0833 ± 0.0000 |
|        32 |      4096 | 34.66 ± 0.89   | 80.89 ± 1.74    | 1.0475 ± 0.0000 |
|        64 |       256 | 0.38 ± 0.09    | 0.65 ± 0.18     | 0.0218 ± 0.0000 |
|        64 |      1024 | 2.12 ± 0.12    | 5.40 ± 0.38     | 0.0873 ± 0.0000 |
|        64 |      4096 | 35.51 ± 1.13   | 82.92 ± 2.66    | 1.0631 ± 0.0000 |
|       128 |       256 | 0.32 ± 0.09    | 0.63 ± 0.19     | 0.0238 ± 0.0000 |
|       128 |      1024 | 2.31 ± 0.32    | 5.49 ± 0.39     | 0.0951 ± 0.0000 |
|       128 |      4096 | 37.38 ± 1.25   | 85.13 ± 2.36    | 1.0944 ± 0.0000 |

## benchmark JIT-compiled Attention

### torch compile

- compile 之后的 attention 测试结果如下

|   d_model |   seq_len | Forward (ms)   | Backward (ms)   | Memory (GB)     |
|----------:|----------:|:---------------|:----------------|:----------------|
|        16 |       256 | 0.18 ± 0.02    | 0.33 ± 0.04     | 0.0203 ± 0.0000 |
|        16 |      1024 | 0.73 ± 0.09    | 1.95 ± 0.18     | 0.0804 ± 0.0000 |
|        16 |      4096 | 9.70 ± 0.18    | 28.10 ± 0.18    | 1.0242 ± 0.0000 |
|        32 |       256 | 0.22 ± 0.02    | 0.33 ± 0.03     | 0.0208 ± 0.0000 |
|        32 |      1024 | 0.73 ± 0.07    | 1.88 ± 0.13     | 0.0824 ± 0.0000 |
|        32 |      4096 | 9.77 ± 0.17    | 27.87 ± 0.20    | 1.0320 ± 0.0000 |
|        64 |       256 | 0.26 ± 0.05    | 0.47 ± 0.11     | 0.0218 ± 0.0000 |
|        64 |      1024 | 0.75 ± 0.05    | 2.01 ± 0.11     | 0.0863 ± 0.0000 |
|        64 |      4096 | 9.95 ± 0.31    | 28.38 ± 0.62    | 1.0476 ± 0.0000 |
|       128 |       256 | 0.24 ± 0.04    | 0.52 ± 0.10     | 0.0237 ± 0.0000 |
|       128 |      1024 | 0.90 ± 0.07    | 2.38 ± 0.15     | 0.0941 ± 0.0000 |
|       128 |      4096 | 11.10 ± 0.31   | 30.82 ± 0.57    | 1.0789 ± 0.0000 |

- compile 之后的 model 测试结果如下，相比 end-to-end benchmark的结果快了许多：

| Model Size   |   Context Length | Mode      | Time (ms)     |
|:-------------|-----------------:|:----------|:--------------|
| small        |              128 | Full Step | 40.50 ± 0.67  |
| small        |              256 | Full Step | 62.51 ± 0.80  |
| medium       |              128 | Full Step | 128.03 ± 1.97 |
| medium       |              256 | Full Step | 191.48 ± 4.55 |


| Model Size   |   Context Length | Mode         | Time (ms)     |
|:-------------|-----------------:|:-------------|:--------------|
| small        |              128 | Forward Only | 22.64 ± 10.66 |
| small        |              256 | Forward Only | 14.15 ± 0.37  |
| medium       |              128 | Forward Only | 28.12 ± 0.53  |
| medium       |              256 | Forward Only | 49.09 ± 1.36  |

### FlashAttention

- 由于测量 backward 时 retain_graph 选项与 torch.comlile ,因此 FlashAttention2Triton 的 backward 调用了内部compile 的函数，没有在外部进行 torch.compile

- 因内存限制，seq_len最长测试到 4096

- FlashAttention-fb16

|   d_model |   seq_len |   Forward (ms) |   Backward (ms) |   End-to-End (ms) |
|----------:|----------:|---------------:|----------------:|------------------:|
|        16 |       128 |           0.02 |            0.11 |              0.13 |
|        16 |       256 |           0.02 |            0.21 |              0.23 |
|        16 |       512 |           0.04 |            0.65 |              0.67 |
|        16 |      1024 |           0.11 |            2.59 |              2.7  |
|        16 |      2048 |           0.38 |            9.88 |             10.1  |
|        16 |      4096 |           1.48 |           38.28 |             40.24 |
|        32 |       128 |           0.01 |            0.13 |              0.16 |
|        32 |       256 |           0.02 |            0.23 |              0.24 |
|        32 |       512 |           0.05 |            0.67 |              0.69 |
|        32 |      1024 |           0.17 |            2.64 |              2.78 |
|        32 |      2048 |           0.63 |            9.8  |             10.66 |
|        32 |      4096 |           2.52 |           39.71 |             40.93 |
|        64 |       128 |           0.02 |            0.12 |              0.16 |
|        64 |       256 |           0.04 |            0.25 |              0.28 |
|        64 |       512 |           0.07 |            0.75 |              0.77 |
|        64 |      1024 |           0.25 |            2.71 |              2.93 |
|        64 |      2048 |           0.89 |            9.98 |             10.89 |
|        64 |      4096 |           3.44 |           39.44 |             42.66 |
|       128 |       128 |           0.03 |            0.12 |              0.14 |
|       128 |       256 |           0.05 |            0.26 |              0.28 |
|       128 |       512 |           0.14 |            0.82 |              0.92 |
|       128 |      1024 |           0.48 |            2.94 |              3.44 |
|       128 |      2048 |           1.78 |           10.61 |             12.44 |
|       128 |      4096 |           6.92 |           41.33 |             48.25 |

- regular attention-bf16

|   d_model |   seq_len |   Forward (ms) |   Backward (ms) |   End-to-End (ms) |
|----------:|----------:|---------------:|----------------:|------------------:|
|        16 |       128 |           0.11 |            0.19 |              0.27 |
|        16 |       256 |           0.11 |            0.22 |              0.34 |
|        16 |       512 |           0.31 |            0.69 |              0.85 |
|        16 |      1024 |           2.02 |            4.59 |              6.53 |
|        16 |      2048 |           8.2  |           18.92 |             26.91 |
|        16 |      4096 |          33.21 |           76.27 |            112.02 |
|        32 |       128 |           0.16 |            0.19 |              0.31 |
|        32 |       256 |           0.18 |            0.31 |              0.4  |
|        32 |       512 |           0.3  |            0.72 |              0.88 |
|        32 |      1024 |           2.01 |            4.67 |              6.59 |
|        32 |      2048 |           8.26 |           19.12 |             27.15 |
|        32 |      4096 |          33.39 |           75.72 |            108.9  |
|        64 |       128 |           0.12 |            0.2  |              0.26 |
|        64 |       256 |           0.18 |            0.25 |              0.38 |
|        64 |       512 |           0.35 |            0.79 |              1.04 |
|        64 |      1024 |           2.17 |            4.84 |              6.82 |
|        64 |      2048 |           8.43 |           18.9  |             27.65 |
|        64 |      4096 |          33.48 |           76.26 |            109.37 |
|       128 |       128 |           0.16 |            0.23 |              0.41 |
|       128 |       256 |           0.14 |            0.37 |              0.49 |
|       128 |       512 |           0.36 |            0.83 |              1.12 |
|       128 |      1024 |           2.1  |            4.85 |              6.92 |
|       128 |      2048 |           8.38 |           19.27 |             28.02 |
|       128 |      4096 |          34.14 |           78.62 |            111.82 |

- FlashAttention2-fp32

|   d_model |   seq_len |   Forward (ms) |   Backward (ms) |   End-to-End (ms) |
|----------:|----------:|---------------:|----------------:|------------------:|
|        16 |       128 |           0.02 |            0.12 |              0.12 |
|        16 |       256 |           0.02 |            0.25 |              0.26 |
|        16 |       512 |           0.05 |            0.83 |              0.85 |
|        16 |      1024 |           0.18 |            3.2  |              3.34 |
|        16 |      2048 |           0.66 |           12.09 |             12.95 |
|        16 |      4096 |           2.6  |           50.07 |             51.22 |
|        32 |       128 |           0.02 |            0.1  |              0.13 |
|        32 |       256 |           0.04 |            0.26 |              0.29 |
|        32 |       512 |           0.09 |            0.84 |              0.89 |
|        32 |      1024 |           0.33 |            3.2  |              3.46 |
|        32 |      2048 |           1.22 |           12.14 |             13.37 |
|        32 |      4096 |           4.78 |           47.99 |             53.11 |
|        64 |       128 |           0.03 |            0.11 |              0.13 |
|        64 |       256 |           0.05 |            0.28 |              0.3  |
|        64 |       512 |           0.15 |            0.87 |              0.96 |
|        64 |      1024 |           0.48 |            3.28 |              3.71 |
|        64 |      2048 |           1.83 |           12.54 |             14.12 |
|        64 |      4096 |           6.97 |           50.73 |             59.09 |
|       128 |       128 |           0.04 |            0.12 |              0.15 |
|       128 |       256 |           0.1  |            0.31 |              0.38 |
|       128 |       512 |           0.3  |            0.96 |              1.23 |
|       128 |      1024 |           1.03 |            3.67 |              4.72 |
|       128 |      2048 |           3.86 |           13.91 |             18.07 |
|       128 |      4096 |          15.19 |           54.43 |             70.04 |


- regular attention-fp32

|   d_model |   seq_len |   Forward (ms) |   Backward (ms) |   End-to-End (ms) |
|----------:|----------:|---------------:|----------------:|------------------:|
|        16 |       128 |           0.11 |            0.23 |              0.29 |
|        16 |       256 |           0.18 |            0.38 |              0.47 |
|        16 |       512 |           0.65 |            1.73 |              2.35 |
|        16 |      1024 |           4.09 |            9.72 |             13.72 |
|        16 |      2048 |          16.39 |           38.11 |             55.3  |
|        16 |      4096 |          67.55 |          779.04 |           1793.41 |
|        32 |       128 |           0.1  |            0.21 |              0.29 |
|        32 |       256 |           0.18 |            0.36 |              0.53 |
|        32 |       512 |           0.64 |            1.75 |              2.4  |
|        32 |      1024 |           4.31 |           11.66 |             16.57 |
|        32 |      2048 |          21.46 |           44.28 |             56.18 |
|        32 |      4096 |          71.39 |          717.89 |           1682.84 |
|        64 |       128 |           0.23 |            0.28 |              0.32 |
|        64 |       256 |           0.26 |            0.44 |              0.56 |
|        64 |       512 |           0.65 |            1.79 |              2.43 |
|        64 |      1024 |           4.3  |            9.84 |             14.68 |
|        64 |      2048 |          16.88 |           43.84 |             56.77 |
|        64 |      4096 |          66.98 |          692.61 |           1706.75 |
|       128 |       128 |           0.11 |            0.24 |              0.36 |
|       128 |       256 |           0.2  |            0.43 |              0.56 |
|       128 |       512 |           0.71 |            1.9  |              2.8  |
|       128 |      1024 |           4.98 |           11.53 |             16.03 |
|       128 |      2048 |          17.71 |           42.99 |             85.39 |
|       128 |      4096 |          70.71 |          690.58 |           1593.33 |

# 3.Distributed Data Parallel Training

- distributed_communication_single_node

| Device   |   World Size | Data Size   |   Avg Time (s) |
|:---------|-------------:|:------------|---------------:|
| CPU      |            2 | 1MB         |       0.000464 |
| CPU      |            2 | 10MB        |       0.002807 |
| CPU      |            2 | 100MB       |       0.028007 |
| CPU      |            2 | 1GB         |       0.284217 |
| CPU      |            4 | 1MB         |       0.001046 |
| CPU      |            4 | 10MB        |       0.004889 |
| CPU      |            4 | 100MB       |       0.050327 |
| CPU      |            4 | 1GB         |       0.500881 |
| CPU      |            6 | 1MB         |       0.002371 |
| CPU      |            6 | 10MB        |       0.006956 |
| CPU      |            6 | 100MB       |       0.079248 |
| CPU      |            6 | 1GB         |       0.715771 |

- naive ddp benchmarking

基础配置如下，使用cpu：

```json
model={
    "vocab_size": 200,
    "context_length": 16,
    "d_model": 64,
    "num_layers": 4,
    "num_heads": 2,
    "d_ff": 256,
    "rope_theta": 10000.0
},
training={
    "precision": "fp32",
}
```
结果如下，可见通信有固定开销：

|   World Size |   Batch Size |   Avg Step Time (s) |   Avg Comm Time (s) | Comm Proportion   |
|-------------:|-------------:|--------------------:|--------------------:|:------------------|
|            2 |           16 |              0.0563 |              0.0227 | 40.33%            |
|            2 |           32 |              0.0501 |              0.0206 | 41.01%            |
|            2 |           64 |              0.052  |              0.0197 | 37.90%            |
|            2 |          128 |              0.0604 |              0.021  | 34.79%            |

- ddp flat benchmarking

|   World Size |   Batch Size |   Avg Step Time (s) |   Avg Comm Time (s) | Comm Proportion   |
|-------------:|-------------:|--------------------:|--------------------:|:------------------|
|            2 |           16 |              0.0414 |              0.0069 | 16.75%            |
|            2 |           32 |              0.0329 |              0.0072 | 21.97%            |
|            2 |           64 |              0.0489 |              0.0057 | 11.58%            |
|            2 |          128 |              0.0557 |              0.0068 | 12.27%            |

- benchmark individual ddp

|   World Size |   Batch Size |   Avg Step Time (s) |   Avg Comm Time (s) | Comm Proportion   |
|-------------:|-------------:|--------------------:|--------------------:|:------------------|
|            2 |           16 |              0.0381 |              0.0139 | 36.37%            |
|            2 |           32 |              0.0417 |              0.0139 | 33.30%            |
|            2 |           64 |              0.0516 |              0.0133 | 25.68%            |
|            2 |          128 |              0.0596 |              0.0141 | 23.59%            |

- bucket ddp:batch_size设置为了 64
- (a) 因为在 cpu 上测试，所以结果参考性有限，可以看到通信时间相对减少许多，大部分在 backward 阶段

|   World Size |   Bucket Size (MB) |   Avg Step Time (s) |   Avg Comm Time (s) | Comm Proportion   |
|-------------:|-------------------:|--------------------:|--------------------:|:------------------|
|            2 |                  1 |              0.1444 |              0.0095 | 6.57%             |
|            2 |                 10 |              0.0993 |              0.0079 | 7.92%             |
|            2 |                100 |              0.0776 |              0.0085 | 10.95%            |
|            2 |               1000 |              0.1125 |              0.0083 | 7.39%             |

- (b) equation
    - 单个桶通信耗时：$T_{comm} = \frac{s}{n_b \cdot w} + o$
    - 单个桶计算耗时：$T_{comp} = \frac{s}{n_b \cdot w}$
    - 总时间： $$T_{total} = T_{comp} + n_b \cdot T_{comm} = \frac{s}{n_b \cdot w} + n_b \cdot \left( \frac{s}{n_b \cdot w} + o \right)$$

    - 通信总开销：$Overhead = T_{total} - \frac{s}{w}= \frac{s}{n_b \cdot w} + n_b \cdot o$

    - 最优桶大小：将 $n_b=\frac{s}{b}$ 代入 $Overhead$，得到：$$\text{Overhead}(b) = \frac{b}{w} + \frac{s \cdot o}{b}$$
    - 求导得最优桶大小：$b = \sqrt{s \cdot w \cdot o}$

- communication_accounting:
    - 假设输入形状：[batch_size, seq_len, d_model] (b, s, d_model)
- (a) 
    - 每blcok参数数量：$param_per_block = 2 * d_{ff} * d_{model}$
    - memory of weights: $M_w = 4 * num_blocks * param_perblock$
    - accumulated gradients: 同 $M_w$
    - optimizer states: 当使用 AdamW 时，为 $2 * M_w$
    - activation(bf16): $2 * num_blocks * b * s * (d_{model} + d_{ff})$ 
    - 计算得到参数，梯度，优化器状态占用内存：3276GB; 激活值占用内存：$16.7b*s$ MB
    - 不考虑激活值，需要 H100 数量：41 个
- (b)
    - 共享的内存：$17547264*b*s + 3517578215424 (bytes)$
    - 忽略激活值，需要的设备数为：$N_{FSDP}=35$
- (c)
    - 假设单设备的批次大小和序列长度分别为：$b, s$，即单设备处理token总数为 $b·s$
    - 计算时间：$$t_{comp} = \frac{2 \cdot 2 \cdot (b \cdot s) \cdot d_{model} \cdot d_{ff}}{Y \cdot C} = \frac{4 \cdot (b \cdot s) \cdot d_{model} \cdot d_{ff}}{Y \cdot C}$$
    - FSDP通信：$$t_{comm, FSDP} = \frac{4 \cdot d_{model} \cdot d_{ff}}{Y \cdot M_X \cdot W_{ici}}$$
    - TP通信：$$t_{comm, TP} = \frac{2 \cdot 2 \cdot (b \cdot s) \cdot d_{model}}{M_Y \cdot W_{ici}} = \frac{4 \cdot (b \cdot s) \cdot d_{model}}{M_Y \cdot W_{ici}}$$
    - 求解 compute bound，必须满足：$t_{comp} \ge t_{comm, FSDP}$ 且 $t_{comp} \ge t_{comm, TP}$
    - 化简得到TP限制（已符合）：$$d_{ff} \ge \frac{Y \cdot C}{M_Y \cdot W_{ici}}$$
    - FSDP限制：$$(b \cdot s) \ge \frac{C}{M_X \cdot W_{ici}} \approx 7931.03$$
    - 故全局 batch size 约为： $B_{overall}=(b·s)·X \approx 126896$
- (d) 引入 pipeline parallelism，降低通信精度等


# 4.Optimizer State Sharding
- world size 默认为 2

- 一般情况

|   Batch Size |   Avg Total Step Time (s) |   Avg Optim Step Time (s) |   Mem After Init (MB) |   Mem Before Optim (MB) |   Mem After Optim (MB) |
|-------------:|--------------------------:|--------------------------:|----------------------:|------------------------:|-----------------------:|
|           16 |                    0.8424 |                    0.2535 |                1616.2 |                 1394.43 |                1395.05 |
|           32 |                    0.9351 |                    0.2529 |                1616.2 |                 1395.64 |                1396.24 |
|           64 |                    1.2517 |                    0.2524 |                1616.2 |                 1414.29 |                1414.58 |
|          128 |                      2.21 |                    0.2511 |                1616.2 |                 1398.19 |                1398.23 |

- 开启 state sharing 的情况

|   Batch Size |   Avg Total Step Time (s) |   Avg Optim Step Time (s) |   Mem After Init (MB) |   Mem Before Optim (MB) |   Mem After Optim (MB) |
|-------------:|--------------------------:|--------------------------:|----------------------:|------------------------:|-----------------------:|
|           16 |                    0.9132 |                    0.3204 |                1616.2 |                 1400.05 |                1400.38 |
|           32 |                    1.0514 |                    0.3202 |                1616.2 |                 1398.27 |                1398.58 |
|           64 |                    1.4145 |                    0.3195 |                1616.2 |                 1413.45 |                 1413.5 |
|          128 |                    2.2896 |                    0.3205 |                1616.2 |                  1395.1 |                1395.14 |


# 补充

- DDP 重新使用 kaggle 的 $T4 \times 2$ 测试如下 

```json
model={
    "vocab_size": 10000,
    "context_length": 32,
    "d_model": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "d_ff": 4096,
    "rope_theta": 10000.0
},
training={
    "precision": "fp32",
}
```

- navie ddp:

|   World Size |   Batch Size |   Avg Step Time (s) |   Avg Comm Time (s) | Comm Proportion   |
|-------------:|-------------:|--------------------:|--------------------:|:------------------|
|            2 |           16 |              1.0085 |              0.4556 | 45.18%            |
|            2 |           32 |               1.275 |              0.4635 | 36.35%            |
|            2 |           64 |              1.7419 |              0.4627 | 26.57%            |
|            2 |          128 |              2.5958 |              0.4636 | 17.86%            |

- flat_ddp:

|   World Size |   Batch Size |   Avg Step Time (s) |   Avg Comm Time (s) | Comm Proportion   |
|-------------:|-------------:|--------------------:|--------------------:|:------------------|
|            2 |           16 |               1.015 |              0.4834 | 47.62%            |
|            2 |           32 |              1.2357 |              0.4865 | 39.37%            |
|            2 |           64 |              1.6982 |              0.4853 | 28.58%            |
|            2 |          128 |              2.6457 |              0.4936 | 18.66%            |

- individul_ddp:

|   World Size |   Batch Size |   Avg Step Time (s) |   Avg Comm Time (s) | Comm Proportion   |
|-------------:|-------------:|--------------------:|--------------------:|:------------------|
|            2 |           16 |              0.8454 |              0.0158 | 1.87%             |
|            2 |           32 |              0.9465 |              0.0159 | 1.68%             |
|            2 |           64 |              1.2471 |              0.0156 | 1.25%             |
|            2 |          128 |              2.2154 |              0.0157 | 0.71%             |

- bucketed_ddp(batch size=64):

|   World Size |   Bucket Size (MB) |   Avg Step Time (s) |   Avg Comm Time (s) | Comm Proportion   |
|-------------:|-------------------:|--------------------:|--------------------:|:------------------|
|            2 |                  1 |              1.3303 |              0.0148 | 1.11%             |
|            2 |                 10 |              1.3208 |              0.0183 | 1.39%             |
|            2 |                100 |              1.3799 |              0.0296 | 2.14%             |
|            2 |               1000 |              1.5235 |              0.0297 | 1.95%             |