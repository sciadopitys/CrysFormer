import torch
from torch import nn

import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from nystrom_attention import Nystromformer
import model.nystrom_ps_rapid as nystrom_att

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Layer normalization of current tokens before feedforward block
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Transformer feedforward block
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Layer normalization of current tokens and partial structure tokens before attention
class PreNorm_partial_structure(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, partial_structure, **kwargs):
        return self.fn(self.norm(x), self.norm_p(partial_structure), **kwargs)

class Transformer_partial_structure(nn.Module):
    def __init__(self, patch_dim, num_patches, num_patches_ps, patch_height, patch_width, frame_patch_size, num_partial_structure, dim, depth, heads, dim_head, mlp_dim, same_partial_structure_emb=True,dropout = 0.):
        super().__init__()

        # Positional encoding for partial structures
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches*num_partial_structure, dim))
        
        self.layers = nn.ModuleList([])
        self.same_partial_structure_emb=same_partial_structure_emb
        if self.same_partial_structure_emb:
            # Create transformer blocks of attention block followed by feedforward block
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm_partial_structure(dim, nystrom_att.Nystrom_attention_partial_structure(dim, heads = heads, dim_head = dim_head, num_landmarks = 64, residual = True, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))

            # Single patch embedding for partial structures, used in all transformer layers
            self.partial_structure_to_patch_embedding = nn.Sequential(
                Rearrange('b p c (f pf) (h p1) (w p2) -> b (p f h w) (c p1 p2 pf)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm_partial_structure(dim, nystrom_att.Nystrom_attention_partial_structure(dim, heads = heads, dim_head = dim_head, num_landmarks = 64, residual = True, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
            self.partial_structure_rearrange=Rearrange('b p c (f pf) (h p1) (w p2) -> b (p f h w) (c p1 p2 pf)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size)
            self.partial_structure_emb_layers=nn.ModuleList([])

            # Provide a different partial structure token embedding to each transformer layer
            for _ in range(depth):
                self.partial_structure_emb_layers.append(nn.Sequential(
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, dim),
                    nn.LayerNorm(dim),
                ))
    def forward(self, x, partial_structure):
        if self.same_partial_structure_emb:

            # Create constant partial structure token embedding
            partial_structure = self.partial_structure_to_patch_embedding(partial_structure)
            _,total_partial_patches,_ = partial_structure.shape
            partial_structure = partial_structure + self.pos_embedding[:,:total_partial_patches,:]
            
            for attn, ff in self.layers:

                # Attention followed by feedforward
                x = attn(x,partial_structure) + x    
                x = ff(x) + x
                
            return x
        else:
            partial_structure = self.partial_structure_rearrange(partial_structure)
            _,total_partial_patches,_ = partial_structure.shape
            for i in range(len(self.layers)):

                # Block-specific partial structure embeddiing
                partial_structure_i=self.partial_structure_emb_layers[i](partial_structure)
                partial_structure_i = partial_structure_i + self.pos_embedding[:,:total_partial_patches,:]

                # Attention followed by feedforward
                x = self.layers[i][0](x,partial_structure_i) + x
                x = self.layers[i][1](x) + x
                
            return x


#MIT License

#Copyright (c) 2019 Andy Brock

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

# Post-transformer residual blocks
class BigGANBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels

        self.relu = nn.ReLU(inplace=True)

        self.bn1=nn.BatchNorm3d(in_channels)
        self.conv1=nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)

        self.bn2=nn.BatchNorm3d(in_channels)

        self.conv2=nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        
        self.bn3=nn.BatchNorm3d(out_channels)

        self.conv3=nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        
        self.bn4=nn.BatchNorm3d(out_channels)

        self.conv4=nn.Conv3d(out_channels, out_channels, kernel_size=1)

        self.learnable_sc = in_channels != out_channels
        if self.learnable_sc:
            self.conv_sc = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        
        res=x

        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.conv4(x)
        
        if self.learnable_sc:       
            res = self.conv_sc(res)

        x = x + res
        return x

class ViT_vary_encoder_decoder_partial_structure(nn.Module):
    def __init__(self, args, num_partial_structure, image_height, image_width, image_patch_size, frames, frame_patch_size, ps_size, dim, depth, heads, mlp_dim, same_partial_structure_emb, transformer="normal" ,pool = 'cls', channels = 10, dim_head = 64, dropout = 0., emb_dropout = 0., biggan_block_num=2, recycle = False):
        super().__init__()


        self.FFT=args.FFT
        self.iFFT=args.iFFT
        self.activation=args.activation
        self.FFT_skip=args.FFT_skip
        same_partial_structure_emb=args.same_partial_structure_emb

        # Define initial convolutions on Patterson map and partial structure
        if recycle: # First convolutional layer accepts an additional input channel during recycling runs
            self.conv1 = nn.Conv3d(in_channels=2, out_channels=10, kernel_size=7, padding=3, bias=False, padding_mode='circular')
        else:
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=7, padding=3, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm3d(10)
        channels=10

        self.conv1_p = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=7, padding=3, bias=False, padding_mode='circular')
        self.bn1_p = nn.BatchNorm3d(10)

        patch_height, patch_width = pair(image_patch_size)

        self.patch_height=patch_height
        self.patch_width=patch_width
        self.frame_patch_size=frame_patch_size

        # Calculate number of patches derived from Patterson map and set of partial structure
        self.num_patches = (math.ceil(image_height / patch_height)) * math.ceil(image_width / patch_width) * math.ceil(frames / frame_patch_size)
        self.num_patches_ps = self.num_patches + ((math.ceil(ps_size / patch_height)) * math.ceil(ps_size / patch_width) * math.ceil(ps_size / frame_patch_size) * num_partial_structure)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        # Patch embedding operations
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_partial_structure(patch_dim, self.num_patches, self.num_patches_ps, patch_height, patch_width, frame_patch_size, num_partial_structure, dim, depth, heads, dim_head, mlp_dim, same_partial_structure_emb, dropout)

        # Conversion back into 3D shape
        self.from_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
        )

        # Post-transformer CNN
        self.biggan_block_num=biggan_block_num
        self.bigGAN_layers=nn.ModuleList([])
        for i in range(biggan_block_num):
            self.bigGAN_layers.append(BigGANBlock(10,10))

        self.conv2 = nn.Conv3d(in_channels=10, out_channels=1, kernel_size=3, padding=1)


    def forward(self, x, ps):

        # Remove dummy batch dimension
        x = torch.squeeze(x,0)
        partial_structure = torch.squeeze(ps, 0)

        # Zero-pad Patterson and partial structure inputs such that all dimensions are divisible by corresponding patch dimension
        x_shape_original = x.shape
        pad_list=[]
        res=x.shape[4]% self.patch_width
        if not(res==0):
            pad_list.append((self.patch_width-res)//2)
            pad_list.append((self.patch_width-res)-(self.patch_width-res)//2)
        else:
            pad_list.append(0)
            pad_list.append(0)


        res=x.shape[3]% self.patch_height
        if not(res==0):
            pad_list.append((self.patch_height-res)//2)
            pad_list.append((self.patch_height-res)-(self.patch_height-res)//2)
        else:
            pad_list.append(0)
            pad_list.append(0)


        res=x.shape[2]% self.frame_patch_size
        if not(res==0):
            pad_list.append((self.frame_patch_size-res)//2)
            pad_list.append((self.frame_patch_size-res)-(self.frame_patch_size-res)//2)
        else:
            pad_list.append(0)
            pad_list.append(0)

        x= torch.nn.functional.pad(x,tuple(pad_list),"constant", 0)

        x_shape = x.shape
        # Initial CNN on and patch embedding on Patterson maps
        x = self.conv1(x)
        x = self.bn1(x) 
        
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        _ , num_partial_structure, _ ,_ ,_ = partial_structure.shape
        # Initial CNN on partial structures
        partial_structure = torch.unsqueeze(rearrange(partial_structure,'b p f h w -> (b p) f h w'),dim=1)
        # shape [batch*num_p,1,frame,height,width]
        partial_structure = self.conv1_p(partial_structure)
        partial_structure = self.bn1_p(partial_structure)
        # shape [batch*num_p,10,frame,height,width]
        partial_structure = rearrange(partial_structure,'(b p) c f h w -> b p c f h w', b=b, p=num_partial_structure)
        # shape [batch,num_p,10,frame,height,width]

        # Apply positional embedding to sequence of Patterson tokens
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        # Central vision transformer
        x = self.transformer(x,partial_structure)

        # Convert from tokens back to 3D shape
        x = self.from_patch_embedding(x)
        x = rearrange(x, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', f=x_shape[2] // self.frame_patch_size, h=x_shape[3] // self.patch_height, w=x_shape[4] // self.patch_width, p1 = self.patch_height, p2 = self.patch_width, pf = self.frame_patch_size, c=10)

        # Post-transformer CNN
        for i in range(self.biggan_block_num):
            x = self.bigGAN_layers[i](x)

        x = self.conv2(x)

        # Allow different final activations
        if self.activation=='tanh':
            x = torch.tanh(x)
        elif self.activation=='sigmoid':
            x = torch.sigmoid(x)
        elif self.activation=='None':
            x = x
        elif self.activation=='relu':
            x = torch.torch.nn.functional.relu(x)
            
        # Extract out regions corresponding to unpadded dimensions
        x= x[:,:,pad_list[4]:(x.shape[2]-pad_list[5]),pad_list[2]:(x.shape[3]-pad_list[3]),pad_list[0]:(x.shape[4]-pad_list[1])]
        
        return(x)



