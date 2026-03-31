import os
import timeit
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    use_gpu = False
    if use_gpu and torch.cuda.is_available():
        backend = "nccl"
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        backend = "gloo"
        device = torch.device("cpu")
        
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device

def benchmark_all_reduce(rank, world_size, byte_size, name, results):
    device = setup(rank, world_size)
    num_elements = byte_size // 4 # float32
    data = torch.randn(num_elements, dtype=torch.float32, device=device)
    num_warmup = 5
    num_iters = 10

    def do_all_reduce():
        dist.all_reduce(data, async_op=False)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    # warmup
    for _ in range(num_warmup):
        do_all_reduce()

    dist.barrier()

    # timing
    start = timeit.default_timer()
    for _ in range(num_iters):
        do_all_reduce()
    end = timeit.default_timer()

    total_time = (end - start) 
    avg_time = total_time / num_iters

    gathered_times = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_times, avg_time)

    if rank == 0:
        overall_avg_time = sum(gathered_times) / world_size
        print(f"Workers: {world_size} | Size: {name:>5} | Avg Time: {overall_avg_time:.6f} seconds")
        results.append({
            "Device": device.type.upper(),
            "World Size": world_size,
            "Data Size": name,
            "Avg Time (s)": f"{overall_avg_time:.6f}"
        })

if __name__ == "__main__":
    manager = mp.Manager()
    results = manager.list()

    sizes =[
        ("1MB", 1024**2),
        ("10MB", 10 * 1024**2),
        ("100MB", 100 * 1024**2),
        ("1GB", 1024**3)
    ]
    world_sizes = [2, 4, 6]
    
    for ws in world_sizes:
        print(f"\n--- Benchmarking with {ws} Processes ---")
        for name, size_bytes in sizes:
            mp.spawn(
                fn=benchmark_all_reduce, 
                args=(ws, size_bytes, name, results), 
                nprocs=ws, 
                join=True
            )
    
    df = pd.DataFrame(list(results))

    print("\n" + "="*80)
    print(f" 📊 Distributed All-Reduce Benchmark Results")
    print("="*80 + "\n")
    
    print(df.to_markdown(index=False))