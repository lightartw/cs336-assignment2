import gc
import os
import timeit
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd

from cs336_systems.config import Config
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW
from cs336_systems.parallel.individual_ddp import individual_ddp
from cs336_systems.parallel.optimizer_share import optimizer_share


def get_cpu_memory_mb():
    gc.collect() 
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def setup(rank, world_size, use_gpu):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if use_gpu and torch.cuda.is_available():
        backend = "nccl"
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        backend = "gloo"
        device = torch.device("cpu")
        
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device

def get_model(config: Config, device):
    mc = config.model
    init_model = BasicsTransformerLM(
        vocab_size=mc.vocab_size,
        context_length=mc.context_length,
        d_model=mc.d_model,
        num_layers=mc.num_layers,
        num_heads=mc.num_heads,
        d_ff=mc.d_ff,
        rope_theta=mc.rope_theta
    ).to(device) 

    model = individual_ddp(init_model)
    return model

def communicate(model: torch.nn.Module, world_size):
    model.finish_gradient_synchronization()


def bench_optimizer(rank, world_size, config: Config, results, use_shared_optim=False):
    mc = config.model
    tc = config.training
    use_gpu = tc.device == "cuda"
    device = setup(rank, world_size, use_gpu)
    dist.barrier()

    # init model
    torch.manual_seed(42)
    model = get_model(config, device)
    dist.barrier()
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        mem_after_init = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        mem_after_init = get_cpu_memory_mb()

    if use_shared_optim:
        optimizer = optimizer_share(model.parameters(), AdamW)
    else:
        optimizer = AdamW(model.parameters())

    # init data
    assert tc.batch_size % world_size == 0
    local_bs = tc.batch_size // world_size
    torch.manual_seed(100 + rank) 
    local_x = torch.randint(0, mc.vocab_size, (local_bs, mc.context_length), device=device)
    local_y = torch.randint(0, mc.vocab_size, (local_bs, mc.context_length), device=device) 
    
    # benchmark config
    num_warmup = 5
    num_iters = 10
    total_step_time_acc = 0.0
    total_optim_time_acc = 0.0
    mem_before_optim = 0.0
    mem_after_optim = 0.0

    # train
    for step in range(num_warmup + num_iters):
        optimizer.zero_grad()
        
        start_step = timeit.default_timer()
        logits = model(local_x)
        loss = cross_entropy(logits, local_y)
        loss.backward()

        communicate(model, world_size)
        clip_gradient(model.parameters(), max_norm=1.0)
        
        # start bench optim
        if step == 0:
            if device.type == 'cuda' and step == num_warmup:
                mem_before_optim = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                mem_before_optim = get_cpu_memory_mb()

        if device.type == 'cuda': 
            torch.cuda.synchronize(device)
        start_optim = timeit.default_timer()

        optimizer.step()

        if device.type == 'cuda': 
            torch.cuda.synchronize(device)
        end_optim = timeit.default_timer()
        end_step = timeit.default_timer()

        if step == 0:
            if device.type == 'cuda' and step == num_warmup:
                mem_after_optim = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                mem_after_optim = get_cpu_memory_mb()

        if step >= num_warmup:
            total_step_time_acc += (end_step - start_step)
            total_optim_time_acc += (end_optim - start_optim)

    # statistics
    avg_step_time = total_step_time_acc / num_iters
    avg_optim_time = total_optim_time_acc / num_iters

    gathered_step_times = [None for _ in range(world_size)]
    gathered_optim_times = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_step_times, avg_step_time)
    dist.all_gather_object(gathered_optim_times, avg_optim_time)

    if rank == 0:
        overall_step_time = sum(gathered_step_times) / world_size
        overall_optim_time = sum(gathered_optim_times) / world_size
        
        results.append({
            "World Size": world_size,
            "Batch Size": tc.batch_size,
            "Avg Total Step Time (s)": round(overall_step_time, 4),
            "Avg Optim Step Time (s)": round(overall_optim_time, 4),
            "Mem After Init (MB)": round(mem_after_init, 2), 
            "Mem Before Optim (MB)": round(mem_before_optim, 2),
            "Mem After Optim (MB)": round(mem_after_optim, 2),
        })
        
    dist.barrier()
    dist.destroy_process_group()



# TODO: not fully test
if __name__ == "__main__":
    manager = mp.Manager()
    results = manager.list()
    
    world_size = 2
    batch_sizes = [16, 32, 64, 128]
    use_shared_optim = True
    
    print(f"\n--- Starting optim Benchmark, World Size: {world_size} ---")
    
    for bs in batch_sizes:
        config = Config(
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
                "batch_size": bs, 
                "precision": "fp32",
                "device": "cpu", 
                "is_compile": False
            }
        )
        
        mp.spawn(
            fn=bench_optimizer, 
            args=(world_size, config, results, use_shared_optim), 
            nprocs=world_size, 
            join=True
        )
    
    df = pd.DataFrame(list(results))

    print("\n" + "="*80)
    print(f" Optim Benchmark Results (Deliverable)")
    print("="*80 + "\n")
    print(df.to_markdown(index=False)) 