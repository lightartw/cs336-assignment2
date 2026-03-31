import os
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd

from cs336_systems.config import Config
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW


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


def bench_naive_ddp(rank, world_size, config: Config, results):
    device = setup(rank, world_size)
    mc = config.model
    tc = config.training
    dist.barrier()

    # init model
    torch.manual_seed(42)
    model = BasicsTransformerLM(
        vocab_size=mc.vocab_size,
        context_length=mc.context_length,
        d_model=mc.d_model,
        num_layers=mc.num_layers,
        num_heads=mc.num_heads,
        d_ff=mc.d_ff,
        rope_theta=mc.rope_theta
    ).to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    dist.barrier()
    
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
    total_comm_time_acc = 0.0

    # train
    for step in range(num_warmup + num_iters):
        optimizer.zero_grad()
        if device.type == 'cuda': 
            torch.cuda.synchronize(device)
        
        start_step = timeit.default_timer()
        logits = model(local_x)
        loss = cross_entropy(logits, local_y)
        loss.backward()

        if device.type == 'cuda': 
            torch.cuda.synchronize(device)
        start_comm = timeit.default_timer()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        
        if device.type == 'cuda': 
            torch.cuda.synchronize(device)
        end_comm = timeit.default_timer()

        clip_gradient(model.parameters(), max_norm=1.0)
        optimizer.step()

        if device.type == 'cuda': 
            torch.cuda.synchronize(device)
        end_step = timeit.default_timer()

        if step >= num_warmup:
            total_step_time_acc += (end_step - start_step)
            total_comm_time_acc += (end_comm - start_comm)

    # statistics
    avg_step_time = total_step_time_acc / num_iters
    avg_comm_time = total_comm_time_acc / num_iters

    gathered_step_times = [None for _ in range(world_size)]
    gathered_comm_times = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_step_times, avg_step_time)
    dist.all_gather_object(gathered_comm_times, avg_comm_time)

    if rank == 0:
        overall_step_time = sum(gathered_step_times) / world_size
        overall_comm_time = sum(gathered_comm_times) / world_size
        comm_proportion = (overall_comm_time / overall_step_time) * 100
        
        results.append({
            "World Size": f"{world_size}",
            "Batch Size": tc.batch_size,
            "Avg Step Time (s)": round(overall_step_time, 4),
            "Avg Comm Time (s)": round(overall_comm_time, 4),
            "Comm Proportion": f"{comm_proportion:.2f}%"
        })

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    manager = mp.Manager()
    results = manager.list()
    
    world_size = 2
    
    batch_sizes = [16, 32, 64, 128]
    
    print(f"\n--- Starting Naive DDP Benchmark, World Size: {world_size} ---")
    
    for bs in batch_sizes:
        config = Config(
            model={
                "vocab_size": 100,
                "context_length": 16,
                "d_model": 64,
                "num_layers": 2,
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
            fn=bench_naive_ddp, 
            args=(world_size, config, results), 
            nprocs=world_size, 
            join=True
        )
    
    df = pd.DataFrame(list(results))

    print("\n" + "="*80)
    print(f" 📊 Naive DDP Training Benchmark Results (Deliverable)")
    print("="*80 + "\n")
    print(df.to_markdown(index=False))