import contextlib
import torch
import timeit
import statistics
import logging
import torch.cuda.nvtx as nvtx

from cs336_systems.config import Config
import cs336_basics
from cs336_basics.model import BasicsTransformerLM, annotated_scaled_dot_product_attention
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW

logger = logging.getLogger(__name__)


def get_random_batch(batch_size: int, vocab_size: int, context_length: int, device: str) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device)

@contextlib.contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled:
        with nvtx.range(name):
            yield
    else:
        yield

class ModelBenchmarker:
    def __init__(self, config: 'Config | str'):
        if isinstance(config, str):
            config = Config.from_json(config)
        mc = config.model
        tc = config.training

        self.device = tc.device
        self.precision = tc.precision

            
        self.model = BasicsTransformerLM(**mc.model_dump())
        self.model.to(self.device)
        self.model.train()
        if tc.is_compile:
            logger.info(f"Compiling the model on {self.device}...")
            if self.device == "cuda":
                torch.set_float32_matmul_precision('high')
            self.model = torch.compile(self.model)


        self.optimizer = AdamW(self.model.parameters())

        self.x = get_random_batch(
            tc.batch_size,
            mc.vocab_size,
            mc.context_length,
            self.device
        )
        self.y = get_random_batch(
            tc.batch_size,
            mc.vocab_size,
            mc.context_length,
            self.device
        )

    def _get_autocast(self):
        precision = self.precision
        if precision == "bf16":
            return torch.autocast(device_type=self.device, dtype=torch.bfloat16)
        elif precision == "fp16":
            return torch.autocast(device_type=self.device, dtype=torch.float16)
        else:
            return contextlib.nullcontext()

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _compute_stats(self, times, description: str):
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        logger.info(f"[{description:20s}] Mean: {mean_time:.2f} ms | Std: {std_time:.2f} ms")
        return mean_time, std_time

    def benchmark_step(
        self,
        only_forward: bool = False,
        num_warmups: int = 5,
        num_trials: int = 10,
        use_nvtx: bool = False,
        profile_memory: bool = False,
    ):
        if use_nvtx:
            cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        step_times = []
        ctx = self._get_autocast()
        

        if profile_memory and self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for i in range(num_warmups + num_trials):
            is_active_step = (i >= num_warmups)
            current_use_nvtx = use_nvtx and is_active_step

            if profile_memory and self.device == "cuda" and i == num_warmups:
                torch.cuda.memory._record_memory_history(max_entries=1000000)
            
            self._sync()
            start = timeit.default_timer()
            # ================= Forward =================
            with nvtx_range("forward", current_use_nvtx):
                with ctx:
                    if only_forward:
                        with torch.no_grad():
                            logits = self.model(self.x)
                    else:
                        logits = self.model(self.x)
                        loss = cross_entropy(logits, self.y)

            # ================= Backward =================
            if not only_forward:
                with nvtx_range("backward", current_use_nvtx):
                    loss.backward()
                    clip_gradient(self.model.parameters(), max_norm=1.0)
                with nvtx_range("optimizer_step", current_use_nvtx):
                    self.optimizer.zero_grad(set_to_none=True)
                    self.optimizer.step()

            self._sync()
            end = timeit.default_timer()
            if is_active_step:
                step_times.append((end - start) * 1000)

        mode = "Forward Only" if only_forward else "Forward + Backward"
        results = {
            "step": self._compute_stats(step_times, mode)
        }

        if profile_memory and self.device == "cuda":
            torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"[Memory] Peak: {peak_mem:.2f} GB")
            results["memory_gb"] = peak_mem

        return results

# =================== test =========================
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    def get_abs_path(rel_path: str) -> str:
        return os.path.normpath(os.path.join(current_dir, rel_path))

    config = get_abs_path("./model_config.json")

    benchmarker = ModelBenchmarker(config)
    benchmarker.benchmark_step(only_forward=False, num_warmups=1)
