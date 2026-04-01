import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class bucket_ddp(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
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

        # init bucket
        max_bytes = bucket_size_mb * 1024 * 1024
        current_bucket = []
        current_bucket_bytes = 0
        self.buckets = []

        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad:
                continue

            param_bytes = param.numel() * param.element_size()
            if current_bucket_bytes + param_bytes > max_bytes and len(current_bucket) > 0:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_bucket_bytes = 0
            
            current_bucket.append(param)
            current_bucket_bytes += param_bytes
        if len(current_bucket) > 0:
            self.buckets.append(current_bucket)


        self.param2bucket = {}
        self.bucket_ready_counts = []
        for bucket_id, bucket_params in enumerate(self.buckets):
            self.bucket_ready_counts.append(len(bucket_params))
            for p in bucket_params:
                self.param2bucket[p] = bucket_id    
        

        self.handles = []

        def hook(param: torch.Tensor):
            bucket_id = self.param2bucket[param]
            self.bucket_ready_counts[bucket_id] -= 1
            if self.bucket_ready_counts[bucket_id] == 0:
                bucket_params = self.buckets[bucket_id]
                grads = [p.grad.data for p in bucket_params]

                flat_grads = _flatten_dense_tensors(grads)
                handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append((handle, flat_grads, grads))


        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle, flat_grads, old_grad in self.handles:
            handle.wait()
    
            if self.world_size > 1:
                flat_grads /= self.world_size

                updated_grads = _unflatten_dense_tensors(flat_grads, old_grad)
                for old_grad, new_grad in zip(old_grad, updated_grads):
                    old_grad.copy_(new_grad)
        
        self.handles.clear()

        for bucket_id, bucket_params in enumerate(self.buckets):
            self.bucket_ready_counts[bucket_id] = len(bucket_params)