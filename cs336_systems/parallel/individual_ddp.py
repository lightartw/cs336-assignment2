import torch
import torch.distributed as dist


class individual_ddp(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        
        if dist.is_initialized():
            self.rank = dist.get_rank()           
            self.world_size = dist.get_world_size()
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
        else:
            self.rank = 0
            self.world_size = 1


        self.handles = []

        def hook(param: torch.Tensor):
            if param.grad is not None:
                handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append(handle)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
    
        if self.world_size > 1:
            for param in self.module.parameters():
                if param.grad is not None:
                    param.grad.data /= self.world_size
        
        self.handles.clear()