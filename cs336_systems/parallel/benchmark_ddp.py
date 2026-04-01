import os
import timeit
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.multiprocessing as mp
import pandas as pd
from enum import Enum

from cs336_systems.config import Config
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW
from cs336_systems.parallel.individual_ddp import individual_ddp
from cs336_systems.parallel.bucket_ddp import bucket_ddp


class DDPType(str, Enum):
    NAIVE = "base_ddp"       
    FLAT = "flat_ddp"       
    INDIVIDUAL = "individual_ddp" 
    BUCKETED = "bucketed_ddp"     


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

def get_model(config: Config, ddptype: DDPType, device, bucket_size_mb: float | None=None):
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

    if ddptype == DDPType.NAIVE or ddptype == DDPType.FLAT:
        for param in init_model.parameters():
            dist.broadcast(param.data, src=0)
        return init_model
    elif ddptype == DDPType.INDIVIDUAL:
        model = individual_ddp(init_model)
        return model
    elif ddptype == DDPType.BUCKETED:
        assert bucket_size_mb is not None
        model = bucket_ddp(init_model, bucket_size_mb)
        return model

def communicate(model: torch.nn.Module, world_size, ddptype: DDPType):
    if ddptype == DDPType.NAIVE:
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
    elif ddptype == DDPType.FLAT:
        grads = [
            p.grad.data for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        flat_grads = _flatten_dense_tensors(grads)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= world_size

        updated_grads = _unflatten_dense_tensors(flat_grads, grads)
        for old_grad, new_grad in zip(grads, updated_grads):
            old_grad.copy_(new_grad)
    elif ddptype == DDPType.INDIVIDUAL:
        model.finish_gradient_synchronization()
    elif ddptype == DDPType.BUCKETED:
        model.finish_gradient_synchronization()


def bench_ddp(rank, world_size, config: Config, results, ddptype: DDPType, bucket_size_mb: float | None=None):
    device = setup(rank, world_size)
    mc = config.model
    tc = config.training
    dist.barrier()

    # init model
    torch.manual_seed(42)
    model = get_model(config, ddptype, device, bucket_size_mb)
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

        communicate(model, world_size, ddptype)
        
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
        
        if bucket_size_mb is not None:
            results.append({
                "World Size": f"{world_size}",
                "Bucket Size (MB)": bucket_size_mb,
                "Avg Step Time (s)": round(overall_step_time, 4),
                "Avg Comm Time (s)": round(overall_comm_time, 4),
                "Comm Proportion": f"{comm_proportion:.2f}%"
            })
        else:
            results.append({
                "World Size": f"{world_size}",
                "Batch Size": tc.batch_size,
                "Avg Step Time (s)": round(overall_step_time, 4),
                "Avg Comm Time (s)": round(overall_comm_time, 4),
                "Comm Proportion": f"{comm_proportion:.2f}%"
            })

    dist.barrier()
    dist.destroy_process_group()

def non_bucket():
    manager = mp.Manager()
    results = manager.list()
    
    world_size = 2
    batch_sizes = [16, 32, 64, 128]
    ddptype = DDPType.INDIVIDUAL
    bucket_size_mb = None
    
    print(f"\n--- Starting DDP Benchmark, World Size: {world_size} ---")
    
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
            fn=bench_ddp, 
            args=(world_size, config, results, ddptype, bucket_size_mb), 
            nprocs=world_size, 
            join=True
        )
    
    df = pd.DataFrame(list(results))

    print("\n" + "="*80)
    print(f" 📊 DDP Training Benchmark Results (Deliverable)")
    print("="*80 + "\n")
    print(df.to_markdown(index=False)) 

def bucket():
    manager = mp.Manager()
    results = manager.list()
    
    world_size = 2
    bucket_sizes_mb = [1, 10, 100, 1000] 
    ddptype = DDPType.BUCKETED
    
    print(f"\n--- Starting DDP Benchmark, World Size: {world_size} ---")
    
    for bucket_size in bucket_sizes_mb:
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
                "batch_size": 64, 
                "precision": "fp32",
                "device": "cpu", 
                "is_compile": False
            }
        ) 
        
        mp.spawn(
            fn=bench_ddp, 
            args=(world_size, config, results, ddptype, bucket_size),
            nprocs=world_size, 
            join=True
        )
    
    df = pd.DataFrame(list(results))

    print("\n" + "="*80)
    print(f" 📊 DDP Training Benchmark Results (Deliverable)")
    print("="*80 + "\n")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    bucket()