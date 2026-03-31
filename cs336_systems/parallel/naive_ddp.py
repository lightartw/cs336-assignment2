import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import copy

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


def train_naive_ddp(rank, world_size, config: Config):
    device = setup(rank, world_size)
    mc = config.model
    tc = config.training
    num_step = 5
    dist.barrier()

    # init model
    torch.manual_seed(42)
    single_model = BasicsTransformerLM(
        vocab_size=mc.vocab_size,
        context_length=mc.context_length,
        d_model=mc.d_model,
        num_layers=mc.num_layers,
        num_heads=mc.num_heads,
        d_ff=mc.d_ff,
        rope_theta=mc.rope_theta
    ).to(device)
    for param in single_model.parameters():
        dist.broadcast(param.data, src=0)
    dist.barrier()
    
    single_optimizer = AdamW(single_model.parameters())
    ddp_model = copy.deepcopy(single_model)
    ddp_optimizer = AdamW(ddp_model.parameters())

    # init data
    total_examples = tc.batch_size * num_step
    torch.manual_seed(100)
    all_x = torch.randint(0, mc.vocab_size, (total_examples, mc.context_length))
    all_y = torch.randint(0, mc.vocab_size, (total_examples, mc.context_length))
    
    # train
    for step in range(num_step):
        start_idx = step * tc.batch_size
        end_idx = start_idx + tc.batch_size
        global_x = all_x[start_idx:end_idx]
        global_y = all_y[start_idx:end_idx]

        assert tc.batch_size % world_size == 0
        local_bs = tc.batch_size // world_size
        local_x = global_x[rank * local_bs: (rank + 1) * local_bs]
        local_y = global_y[rank * local_bs: (rank + 1) * local_bs]

        # single process
        if rank == 0:
            single_optimizer.zero_grad()
            single_logits = single_model(global_x)
            single_loss = cross_entropy(single_logits, global_y)
            single_loss.backward()
            clip_gradient(single_model.parameters(), max_norm=1.0)
            single_optimizer.step()

        # navie ddp
        ddp_optimizer.zero_grad()
        ddp_logits = ddp_model(local_x)
        ddp_loss = cross_entropy(ddp_logits, local_y)
        ddp_loss.backward()

        for param in ddp_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size

        clip_gradient(ddp_model.parameters(), max_norm=1.0)
        ddp_optimizer.step()

        if rank == 0:
            try:
                for (n1, p1), (n2, p2) in zip(
                    single_model.named_parameters(),
                    ddp_model.named_parameters()
                ):
                    torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-5)
                print(f"✅ Step {step+1} passed")
            except AssertionError as e:
                print(f"❌ Step {step+1} FAILED")
                print(e)
                dist.abort()

    if rank == 0:
        print("🎉 ALL STEPS PASSED!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
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
            "batch_size": 16,
            "precision": "fp32",
            "device": "cpu",
            "is_compile": False
        }
    )

    world_size = 4
    mp.spawn(
        train_naive_ddp,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
