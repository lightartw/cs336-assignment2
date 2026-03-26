import torch
import triton
import triton.language as tl
import math

class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                is_causal: bool=False):
        """
        FlashAttention-2 forward pass in pure PyTorch.

        Args:
            ctx (FunctionCtx) 
            Q (torch.Tensor): (..., N_q, d)
            K (torch.Tensor): (..., N_k, d)
            V (torch.Tensor): (..., N_k, d)
            is_causal (bool, optional): 

        Returns:
            O (torch.Tensor): (..., N_q, d)
        
        Saved for backward :
            L (torch.Tensor): LogSumExp 
                (..., N_q)
            Q, K, V, O: 
        """
        B_q, B_k = 32, 32

        N_q = Q.shape[-2]
        d   = Q.shape[-1]
        N_k = K.shape[-2]

        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)

        O = Q.new_zeros(Q.shape)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=torch.float32)

        for i in range(T_q):
            start_i = i * B_q
            end_i = min((i+1) * B_q, N_q)
            Q_i = Q[..., start_i:end_i, :]

            scalar_shape = Q_i.shape[:-1] 
            O_i_accum = Q_i.new_zeros(Q_i.shape)
            l_i = Q_i.new_zeros(scalar_shape)
            m_i = Q_i.new_full(scalar_shape, float("-inf"))

            for j in range(T_k):
                start_j = j * B_k
                end_j = min((j + 1) * B_k, N_k)
                K_j = K[..., start_j:end_j, :]
                V_j = V[..., start_j:end_j, :]

                S_ij = Q_i @ K_j.transpose(-2, -1) / math.sqrt(d)    
                m_i_new = torch.maximum(m_i, torch.max(S_ij, dim=-1).values)        
                P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))

                scale = torch.exp(m_i - m_i_new)
                l_i_new = scale * l_i + P_ij.sum(dim=-1)
                O_i_accum = scale.unsqueeze(-1) * O_i_accum + P_ij @ V_j

                m_i = m_i_new
                l_i = l_i_new
            
            O_i = O_i_accum / l_i.unsqueeze(-1)
            L_i = m_i + torch.log(l_i)

            O[..., start_i:end_i, :] = O_i
            L[..., start_i:end_i] = L_i

        ctx.save_for_backward(L, Q, K, V, O)
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
    

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr=False,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
   
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # load
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))
    
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    m_i = tl.full([Q_TILE_SIZE], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)

    offset_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(0, T_k):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1))
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1))

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            offset_k = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = offset_q[:, None] >= offset_k[None, :]
            S_ij = tl.where(mask, S_ij, -1e6)

        m_i_new = tl.maximum(m_i, tl.max(S_ij, axis=1))
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        
        alpha = tl.exp(m_i - m_i_new)
        l_i = alpha * l_i + tl.sum(P_ij, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(P_ij.to(V_j.dtype), V_j, acc=acc)

        m_i = m_i_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    O_i = acc / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i.to(O_ptr.dtype.element_ty), boundary_check=(0, 1))
    L_ptrs = L_ptr + batch_index * stride_lb + offset_q * stride_lq
    mask_q = offset_q < N_QUERIES
    tl.store(L_ptrs, L_i.to(L_ptr.dtype.element_ty), mask=mask_q)


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                is_causal: bool=False):
        """
        FlashAttention-2 forward pass in triton.
        """
        B_q, B_k = 32, 32

        N_q = Q.shape[-2]
        d   = Q.shape[-1]
        N_k = K.shape[-2]

        T_q = math.ceil(N_q / B_q)

        O = Q.new_zeros(Q.shape)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=torch.float32)

        batch_size = Q.numel() // (Q.shape[-2] * Q.shape[-1])
        Q_3d = Q.view(batch_size, N_q, d)
        K_3d = K.view(batch_size, N_k, d)
        V_3d = V.view(batch_size, N_k, d)
        O_3d = O.view(batch_size, N_q, d)
        L_2d = L.view(batch_size, N_q)
        scale = 1.0 / math.sqrt(d)

        flash_fwd_kernel[(T_q, batch_size,)](
            Q_3d, K_3d, V_3d,
            O_3d, L_2d,
            Q_3d.stride(0), Q_3d.stride(1), Q_3d.stride(2),
            K_3d.stride(0), K_3d.stride(1), K_3d.stride(2),
            V_3d.stride(0), V_3d.stride(1), V_3d.stride(2),
            O_3d.stride(0), O_3d.stride(1), O_3d.stride(2),
            L_2d.stride(0), L_2d.stride(1),
            N_QUERIES=N_q, N_KEYS=N_k,
            scale=scale,
            D=d,
            Q_TILE_SIZE=B_q,
            K_TILE_SIZE=B_k,
            is_causal=is_causal
        )
        
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError