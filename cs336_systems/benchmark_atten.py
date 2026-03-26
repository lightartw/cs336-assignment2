import torch
import triton
import timeit
import statistics
import logging

from cs336_systems.config import Config

logger = logging.getLogger(__name__)

class AttentionBenchmarker:
    def __init__(self, config: 'Config | str'):
        if isinstance(config, str):
            config = Config.from_json(config)
        tc = config.training
        self.device = tc.device

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _compute_stats(self, times, description: str):
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        logger.info(f"[{description:20s}] Mean: {mean_time:.2f} ms/GB | Std: {std_time:.2f} ms/GB")
        return mean_time, std_time

    def benchmark_attention_step(
        self,
        attn_fn,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
        num_warmups: int = 10,
        num_trials: int = 100,
    ):
        forward_times =[]
        backward_times =[]
        memory_samples = []
        
        if Q.grad is not None: Q.grad = None
        if K.grad is not None: K.grad = None
        if V.grad is not None: V.grad = None

        try:
            for i in range(num_warmups + num_trials):
                is_active_step = (i >= num_warmups)
                
                # ================= 1. Forward Pass =================
                self._sync()
                start_fwd = timeit.default_timer()
                
                out = attn_fn(Q, K, V, mask)
                
                self._sync()
                end_fwd = timeit.default_timer()
                
                if is_active_step:
                    forward_times.append((end_fwd - start_fwd) * 1000)
                    current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    memory_samples.append(current_mem)     
                
                # ================= 2. Backward Pass =================
                loss = out.sum()
                
                self._sync()
                start_bwd = timeit.default_timer()
                
                loss.backward()
                
                self._sync()
                end_bwd = timeit.default_timer()
                
                if is_active_step:
                    backward_times.append((end_bwd - start_bwd) * 1000)
                    
                Q.grad = None
                K.grad = None
                V.grad = None
                
            fwd_stats = self._compute_stats(forward_times, "Forward Pass")
            bwd_stats = self._compute_stats(backward_times, "Backward Pass")
            memory_stats = self._compute_stats(memory_samples, "Memory states")
            
            return {
                "forward_mean_ms": fwd_stats[0],
                "forward_std_ms": fwd_stats[1],
                "backward_mean_ms": bwd_stats[0],
                "backward_std_ms": bwd_stats[1],
                "memory_gb": memory_stats[0],
                "memory_std": memory_stats[1],
                "oom": False
            }
        
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("CUDA Out of Memory Error encountered!")
            return {"oom": True}
        

def benchmark_flash(n_heads, seq_len, d_model, dtype, attn_fn, is_causal=True, warmups=25, trials=100):
    q, k, v = [
        torch.randn(1, n_heads, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]

    # 1. Forward
    fwd_ms = triton.testing.do_bench(
        lambda: attn_fn(q, k, v, is_causal), 
        warmup=warmups, 
        rep=trials
    )

    # 2. Backward
    o = attn_fn(q, k, v, is_causal)
    loss = o.sum()
    def bwd_pass():
        q.grad = k.grad = v.grad = None
        loss.backward(retain_graph=True)
        
    bwd_ms = triton.testing.do_bench(
        bwd_pass, 
        warmup=warmups, 
        rep=trials
    )

    # 3. Forward + Backward(end-to-end)
    def full_pass():
        q.grad = k.grad = v.grad = None
        out = attn_fn(q, k, v, is_causal)
        loss = out.sum()
        loss.backward()
        
    e2e_ms = triton.testing.do_bench(
        full_pass, 
        warmup=warmups, 
        rep=trials
    )

    return fwd_ms, bwd_ms, e2e_ms