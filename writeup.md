# Profilling and Benchmarking
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
