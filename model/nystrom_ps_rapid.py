import torch
from torch import nn, einsum

import math
from einops import rearrange, reduce
import torch.nn.functional as F


def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

class Nystrom_attention_partial_structure(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, num_landmarks = 64, pinv_iterations = 6, residual = True, residual_conv_kernel = 33, eps = 1e-8,dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.eps = eps
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        
        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, partial_structure):
        #x shape [batch, patch_num, emb_dim]
        b,patch_num,_ = x.shape
        _,ptotal,_ = partial_structure.shape
        p_num = ptotal / patch_num
        m = self.num_landmarks
        eps = self.eps
        iters = self.pinv_iterations

        padding = 0
        remainder = patch_num % m
        if remainder > 0:
            padding = m - remainder
            x = F.pad(x, (0, 0, padding, 0), value = 0)

        patch_plus = patch_num + padding
        padding1 = 0
        remainder1 = (patch_plus + ptotal) % m
        if remainder1 > 0:
            padding1 = m - remainder1    
            partial_structure = F.pad(partial_structure, (0, 0, padding1, 0), value = 0)

        #partial_strcuture [batch, patch_num*p_num, emb_dim]
        combined_x = torch.cat((x,partial_structure),dim=1)
        
        #combined_x [batch, patch_num*(p_num+1),emb_dim]

        qkv = self.to_qkv(combined_x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        #q shape [batch, heads_num, patch_num*(p_num+1), d]
        #k shape [batch, heads_num, patch_num*(p_num+1), d]
        masked_q = q[:,:,0:patch_plus, :]
        
        masked_q = masked_q * self.scale
        
        
        l1 = math.ceil(patch_plus / m)
        l2 = math.ceil((ptotal + padding1 + patch_plus) / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(masked_q, landmark_einops_eq, 'sum', l = l1)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l2)
        
        q_landmarks /= l1
        k_landmarks /= l2
        
        
        
        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, masked_q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)
        
        
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))

        
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        
        if self.residual:
            out += (self.res_conv(v)[:,:,0:patch_plus, :])
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out[:, -patch_num:, :]
        