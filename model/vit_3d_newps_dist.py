import torch
from torch import nn

import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from nystrom_attention import Nystromformer
import model.nystrom_ps_rapid as nystrom_att

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PreNorm_partial_structure(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, partial_structure, **kwargs):
        return self.fn(self.norm(x), self.norm_p(partial_structure), **kwargs)

class FeedForward_partial_structure(nn.Module):
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

class Attention_partial_structure(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, partial_structure):
        #x shape [batch, patch_num, emb_dim]
        _,patch_num,_ = x.shape
        #partial_strcuture [batch, patch_num*p_num, emb_dim]
        combined_x = torch.cat((x,partial_structure),dim=1)
        #combined_x [batch, patch_num*(p_num+1),emb_dim]

        qkv = self.to_qkv(combined_x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        #q shape [batch, heads_num, patch_num*(p_num+1), d]
        #k shape [batch, heads_num, patch_num*(p_num+1), d]
        masked_q = q[:,:,0:patch_num, :]
        #masked q shape [batch, heads_num, patch_num, d]
        dots = torch.matmul(masked_q, k.transpose(-1, -2)) * self.scale
        #dots shape [batch, heads_num, patch_num, patch_num*(p_num+1)]
        attn = self.attend(dots)
        attn = self.dropout(attn)
        #attn shape [batch, heads_num, patch_num, patch_num*(p_num+1)]
        #v shape [batch, heads_num, patch_num*(p_num+1), d]
        out = torch.matmul(attn, v)
        #out shape [batch, heads_num, patch_num, d]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer_partial_structure(nn.Module):
    def __init__(self, patch_dim, num_patches, num_patches_ps, patch_height, patch_width, frame_patch_size, num_partial_structure, dim, depth, heads, dim_head, mlp_dim, same_partial_structure_emb=True,dropout = 0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches*num_partial_structure, dim))
        self.layers = nn.ModuleList([])
        self.same_partial_structure_emb=same_partial_structure_emb
        if self.same_partial_structure_emb:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    #PreNorm_partial_structure(dim, Attention_partial_structure(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm_partial_structure(dim, nystrom_att.Nystrom_attention_partial_structure(dim, heads = heads, dim_head = dim_head, num_landmarks = 64, residual = True, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
            self.partial_structure_to_patch_embedding = nn.Sequential(
                Rearrange('b p c (f pf) (h p1) (w p2) -> b (p f h w) (c p1 p2 pf)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    #PreNorm_partial_structure(dim, Attention_partial_structure(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm_partial_structure(dim, nystrom_att.Nystrom_attention_partial_structure(dim, heads = heads, dim_head = dim_head, num_landmarks = 64, residual = True, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
            self.partial_structure_rearrange=Rearrange('b p c (f pf) (h p1) (w p2) -> b (p f h w) (c p1 p2 pf)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size)
            self.partial_structure_emb_layers=nn.ModuleList([])
            for _ in range(depth):
                self.partial_structure_emb_layers.append(nn.Sequential(
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, dim),
                    nn.LayerNorm(dim),
                ))
    def forward(self, x, partial_structure):
        if self.same_partial_structure_emb:
            
            partial_structure = self.partial_structure_to_patch_embedding(partial_structure)
            _,total_partial_patches,_ = partial_structure.shape
            partial_structure = partial_structure + self.pos_embedding[:,:total_partial_patches,:]
            
            for attn, ff in self.layers:
                
                x = attn(x,partial_structure) + x    
                x = ff(x) + x
                
            return x
        else:
            partial_structure = self.partial_structure_rearrange(partial_structure)
            _,total_partial_patches,_ = partial_structure.shape
            for i in range(len(self.layers)):
                partial_structure_i=self.partial_structure_emb_layers[i](partial_structure)
                partial_structure_i = partial_structure_i + self.pos_embedding[:,:total_partial_patches,:]
                x = self.layers[i][0](x,partial_structure_i) + x
                x = self.layers[i][1](x) + x
            return x
class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ViT_encoder_decoder(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 10, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=3, padding=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm3d(10)
        channels=10

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.from_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', f=frames // frame_patch_size, h=image_height // patch_height, w=image_width // patch_width, p1 = patch_height, p2 = patch_width, pf = frame_patch_size, c=10),
        )

        self.conv2 = nn.Conv3d(in_channels=10, out_channels=1, kernel_size=3, padding=1)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x) 
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.from_patch_embedding(x)

        x = self.conv2(x)

        x = torch.tanh(x)

        return(x)

class ViT_vary_encoder_decoder(nn.Module):
    def __init__(self, args, image_height, image_width, image_patch_size, frames, frame_patch_size, dim, depth, heads, mlp_dim, transformer="normal" ,pool = 'cls', channels = 10, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()


        self.FFT=args.FFT
        self.iFFT=args.iFFT
        self.activation=args.activation
        self.FFT_skip=args.FFT_skip
        transformer = args.transformer

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=3, padding=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm3d(10)
        channels=10
        
        patch_height, patch_width = pair(image_patch_size)

        self.patch_height=patch_height
        self.patch_width=patch_width
        self.frame_patch_size=frame_patch_size

        num_patches = (math.ceil(image_height / patch_height)) * math.ceil(image_width / patch_width) * math.ceil(frames / frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if transformer=="normal":
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        elif transformer=="Nystromformer":
            self.transformer = Nystromformer(dim = dim,
                                            depth = depth,
                                            heads = heads,
                                            num_landmarks = 64
                                            )

        self.from_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
        )

        self.conv2 = nn.Conv3d(in_channels=10, out_channels=1, kernel_size=3, padding=1)


    def forward(self, x):
        #x shape [batch,channel,frame,height,width]

        x = torch.squeeze(x,0)
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
        if self.FFT:
            x_size=x.size()
            if self.FFT_skip:
                x= torch.fft.fftn(x,dim=(2,3,4)).real + x
            else:
                x= torch.fft.fftn(x,dim=(2,3,4)).real
            assert x.size()==x_size
        
        x = self.conv1(x)
        x = self.bn1(x) 
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.from_patch_embedding(x)

        x = rearrange(x, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', f=x_shape[2] // self.frame_patch_size, h=x_shape[3] // self.patch_height, w=x_shape[4] // self.patch_width, p1 = self.patch_height, p2 = self.patch_width, pf = self.frame_patch_size, c=10)

        x = self.conv2(x)

        if self.iFFT:
            x = torch.fft.ifftn(x).real

        if self.activation=='tanh':
            x = torch.tanh(x)
        elif self.activation=='sigmoid':
            x = torch.sigmoid(x)
        elif self.activation=='None':
            x = x
        elif self.activation=='relu':
            x = torch.torch.nn.functional.relu(x)
        x= x[:,:,pad_list[4]:(x.shape[2]-pad_list[5]),pad_list[2]:(x.shape[3]-pad_list[3]),pad_list[0]:(x.shape[4]-pad_list[1])]
        return(x)


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

class ViT_vary_encoder_decoder_biggan_block(nn.Module):
    def __init__(self, args, image_height, image_width, image_patch_size, frames, frame_patch_size, dim, depth, heads, mlp_dim, transformer="Linformer" , pool = 'cls', channels = 10, dim_head = 64, dropout = 0., emb_dropout = 0., biggan_block_num=2):
        super().__init__()


        self.FFT=args.FFT
        self.iFFT=args.iFFT
        self.activation=args.activation
        self.FFT_skip=args.FFT_skip
        self.which_transformer = args.transformer
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=7, padding=3, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm3d(10)
        
        channels=10
        patch_height, patch_width = pair(image_patch_size)

        self.patch_height=patch_height
        self.patch_width=patch_width
        self.frame_patch_size=frame_patch_size


        num_patches = (math.ceil(image_height / patch_height)) * math.ceil(image_width / patch_width) * math.ceil(frames / frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if self.which_transformer=="normal":
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        elif self.which_transformer=="Nystromformer":
            self.transformer = Nystromformer(dim = dim,
                                            depth = depth,
                                            heads = heads,
                                            num_landmarks = 64
                                            )
        else:
            raise ValueError

        self.from_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
        )

        self.biggan_block_num=biggan_block_num
        self.bigGAN_layers=nn.ModuleList([])
        for i in range(biggan_block_num):
            self.bigGAN_layers.append(BigGANBlock(10,10))

        self.conv2 = nn.Conv3d(in_channels=10, out_channels=1, kernel_size=3, padding=1)

    def forward(self, combined_x):
        #x shape [batch,channel,frame,height,width]
        x = torch.squeeze(combined_x,0)
        
        #partial_structure [batch,num_p,frame,height,width]
        
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
        
        if self.FFT:
            x_size=x.size()
            if self.FFT_skip:
                x= torch.fft.fftn(x,dim=(2,3,4)).real + x
            else:
                x= torch.fft.fftn(x,dim=(2,3,4)).real
            assert x.size()==x_size
        
        x = self.conv1(x)
        x = self.bn1(x) 
        
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.from_patch_embedding(x)

        x = rearrange(x, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', f=x_shape[2] // self.frame_patch_size, h=x_shape[3] // self.patch_height, w=x_shape[4] // self.patch_width, p1 = self.patch_height, p2 = self.patch_width, pf = self.frame_patch_size, c=10)


        for i in range(self.biggan_block_num):
            x = self.bigGAN_layers[i](x)

        x = self.conv2(x)

        if self.iFFT:
            x = torch.fft.ifftn(x).real

        if self.activation=='tanh':
            x = torch.tanh(x)
        elif self.activation=='sigmoid':
            x = torch.sigmoid(x)
        elif self.activation=='None':
            x = x
        elif self.activation=='relu':
            x = torch.torch.nn.functional.relu(x)

        x= x[:,:,pad_list[4]:(x.shape[2]-pad_list[5]),pad_list[2]:(x.shape[3]-pad_list[3]),pad_list[0]:(x.shape[4]-pad_list[1])]
        return(x)

class ViT_vary_encoder_decoder_partial_structure(nn.Module):
    def __init__(self, args, num_partial_structure, image_height, image_width, image_patch_size, frames, frame_patch_size, ps_size, dim, depth, heads, mlp_dim, same_partial_structure_emb, transformer="normal" ,pool = 'cls', channels = 10, dim_head = 64, dropout = 0., emb_dropout = 0., biggan_block_num=2, recycle = False):
        super().__init__()


        self.FFT=args.FFT
        self.iFFT=args.iFFT
        self.activation=args.activation
        self.FFT_skip=args.FFT_skip
        transformer = "normal"
        same_partial_structure_emb=args.same_partial_structure_emb

        if recycle:
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

        self.num_patches = (math.ceil(image_height / patch_height)) * math.ceil(image_width / patch_width) * math.ceil(frames / frame_patch_size)
        self.num_patches_ps = self.num_patches + ((math.ceil(ps_size / patch_height)) * math.ceil(ps_size / patch_width) * math.ceil(ps_size / frame_patch_size) * num_partial_structure)
        patch_dim = channels * patch_height * patch_width * frame_patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if transformer=="normal":
            self.transformer = Transformer_partial_structure(patch_dim, self.num_patches, self.num_patches_ps, patch_height, patch_width, frame_patch_size, num_partial_structure, dim, depth, heads, dim_head, mlp_dim, same_partial_structure_emb, dropout)
        elif transformer=="Nystromformer":
            raise ValueError

        self.from_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
        )
        
        self.biggan_block_num=biggan_block_num
        self.bigGAN_layers=nn.ModuleList([])
        for i in range(biggan_block_num):
            self.bigGAN_layers.append(BigGANBlock(10,10))

        self.conv2 = nn.Conv3d(in_channels=10, out_channels=1, kernel_size=3, padding=1)


    def forward(self, x, ps):

        x = torch.squeeze(x,0)
        partial_structure = torch.squeeze(ps, 0)
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
        if self.FFT:
            x_size=x.size()
            if self.FFT_skip:
                x= torch.fft.fftn(x,dim=(2,3,4)).real + x
            else:
                x= torch.fft.fftn(x,dim=(2,3,4)).real
            assert x.size()==x_size
        
        x = self.conv1(x)
        x = self.bn1(x) 

        
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        _ , num_partial_structure, _ ,_ ,_ = partial_structure.shape
        partial_structure = torch.unsqueeze(rearrange(partial_structure,'b p f h w -> (b p) f h w'),dim=1)
        # shape [batch*num_p,1,frame,height,width]
        partial_structure = self.conv1_p(partial_structure)
        partial_structure = self.bn1_p(partial_structure)
        # shape [batch*num_p,10,frame,height,width]
        partial_structure = rearrange(partial_structure,'(b p) c f h w -> b p c f h w', b=b, p=num_partial_structure)
        # shape [batch,num_p,10,frame,height,width]

        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        
        x = self.transformer(x,partial_structure)
        
        x = self.from_patch_embedding(x)

        x = rearrange(x, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', f=x_shape[2] // self.frame_patch_size, h=x_shape[3] // self.patch_height, w=x_shape[4] // self.patch_width, p1 = self.patch_height, p2 = self.patch_width, pf = self.frame_patch_size, c=10)

        
        for i in range(self.biggan_block_num):
            x = self.bigGAN_layers[i](x)

        x = self.conv2(x)

        if self.iFFT:
            x = torch.fft.ifftn(x).real

        if self.activation=='tanh':
            x = torch.tanh(x)
        elif self.activation=='sigmoid':
            x = torch.sigmoid(x)
        elif self.activation=='None':
            x = x
        elif self.activation=='relu':
            x = torch.torch.nn.functional.relu(x)
        x= x[:,:,pad_list[4]:(x.shape[2]-pad_list[5]),pad_list[2]:(x.shape[3]-pad_list[3]),pad_list[0]:(x.shape[4]-pad_list[1])]
        
        return(x)
