import contextlib
import numpy as np
import torch
import timeit
import statistics
import logging

from cs336_systems.config import Config
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch

logger = logging.getLogger(__name__)

class ModelBenchmarker:
    def __init__(self, config: 'Config | str'):
        if isinstance(config, str):
            config = Config.from_json(config)
        self.mc = config.model
        self.oc = config.optimizer
        self.tc = config.training

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        precision = self.tc.precision
        if precision == "bf16":
            self.autocast_context = torch.autocast(device_type=self.device, dtype=torch.bfloat16)
        elif precision == "fp16":
            self.autocast_context = torch.autocast(device_type=self.device, dtype=torch.float16)
        else:
            self.autocast_context = contextlib.nullcontext()
        self.precision = precision
        
        self.model = BasicsTransformerLM(**self.mc.model_dump())
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.oc.lr,
            betas=(self.oc.beta1, self.oc.beta2),
            eps=self.oc.eps,
            weight_decay=self.oc.weight_decay
        )
        
        self.model.train()
        dataset_length = self.mc.context_length * 100
        data = np.random.randint(0, self.mc.vocab_size, size=(dataset_length,), dtype=np.int32)
        self.x, self.y = get_batch(data, self.tc.batch_size, self.mc.context_length, self.device)

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
    
    def _compute_stats(self, times, description: str):
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        logger.info(f"[{description:20s}] Mean: {mean_time:.2f} ms | Std: {std_time:.2f} ms")
        return mean_time, std_time

    def benchmark_step(self, only_forward: bool = False, num_warmups=5, num_trials=20):
        times =[]
        
        for i in range(num_warmups + num_trials):
            # start timing
            self._sync()
            start = timeit.default_timer()
            
            if only_forward:        # 推理
                with torch.no_grad(), self.autocast_context:
                    logits = self.model(self.x)
            else:                   # 训练
                self.optimizer.zero_grad(set_to_none=True)
                with self.autocast_context:
                    logits = self.model(self.x)
                    loss = cross_entropy(logits, self.y)
                loss.backward()
                clip_gradient(self.model.parameters(), max_norm=self.tc.max_norm)
                self.optimizer.step()

            self._sync()
            end = timeit.default_timer()
            
            if i >= num_warmups:
                times.append((end - start) * 1000)
        
        mode = "Forward Only" if only_forward else "Forward + Backward"
        return self._compute_stats(times, mode)



# =================== test =========================
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    def get_abs_path(rel_path: str) -> str:
        return os.path.normpath(os.path.join(current_dir, rel_path))

    config = get_abs_path("./model_config.json")

    benchmarker = ModelBenchmarker(config)
    benchmarker.benchmark_step(only_forward=False, num_warmups=1)
