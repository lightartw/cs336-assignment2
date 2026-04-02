from collections.abc import Callable
from typing import Any, Type

import torch.distributed as dist
from torch.optim import Optimizer


class optimizer_share(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        if dist.is_initialized():
            self.rank = dist.get_rank()           
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        self.params_with_owner = []
                
        super().__init__(params, defaults=kwargs)
        self.optimizer = optimizer_cls(self.param_groups)
        

    def step(self, closure: Callable | None = None, **kwargs):
        loss = self.optimizer.step(closure, **kwargs)
        
        for owner_rank, param in self.params_with_owner:
            dist.broadcast(param.data, src=owner_rank)

        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        params = param_group["params"]
        for i, param in enumerate(params):
            owner_rank = i % self.world_size
            self.params_with_owner.append((owner_rank, param))
        
        shared_param = params[self.rank :: self.world_size]
        param_group["params"] = shared_param

        super().add_param_group(param_group)