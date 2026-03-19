# Profilling and Benchmarking
- 测试的 GPU 为 4060 LAPTOP

## End-to-End Benchmarking
- 因为显存不够，所以仅对 small, medium 的模型仅做了 forward+backward 测试

| Model Size   |   Context Length | Mode      | Time (ms)        |
|:-------------|-----------------:|:----------|:-----------------|
| small        |              128 | Full Step | 138.45 ± 19.48   |
| small        |              256 | Full Step | 207.59 ± 3.25    |
| medium       |              128 | Full Step | 445.70 ± 11.52   |
| medium       |              256 | Full Step | 6904.29 ± 656.46 |

- 以下是除 xl, 2.7B 模型的 forward 测试表：

| Model Size   |   Context Length | Mode         | Time (ms)       |
|:-------------|-----------------:|:-------------|:----------------|
| small        |              128 | Forward Only | 26.31 ± 1.44    |
| small        |              256 | Forward Only | 58.32 ± 1.31    |
| medium       |              128 | Forward Only | 86.58 ± 2.10    |
| medium       |              256 | Forward Only | 166.53 ± 3.70   |
| large        |              128 | Forward Only | 174.09 ± 4.50   |
| large        |              256 | Forward Only | 1703.45 ± 91.78 |


## Nsight Systems Profiler