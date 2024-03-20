from torch.utils.data import Dataset, DataLoader

import torch 
import torch.nn as nn
import torch.fft
import torch.cuda
import csv
import numpy as np
import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import math
import statistics
from vit_3d_newps_dist import ViT_encoder_decoder, ViT_vary_encoder_decoder, ViT_vary_encoder_decoder_partial_structure, ViT_vary_encoder_decoder_biggan_block
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #use GPU if possible
torch.backends.cudnn.benchmark = True

with open("test_new_dataset2.txt") as myfile: #select the first n_train examples as the training set, rest as validation set
    testlist = myfile.readlines()
testlist = [x.rstrip() for x in testlist]

with open("train_new_dataset2.txt") as myfile1: #select the first n_train examples as the training set, rest as validation set
    trainlist = myfile1.readlines()
trainlist = [x.rstrip() for x in trainlist]

full_list = trainlist + testlist


with open("new-dipeptide-AA-type-noclash_new2.list") as myfile1:
    examples = myfile1.readlines()
examples  = [x.rstrip() for x in examples]
        

class Dataset(torch.utils.data.Dataset):

  def __init__(self, pdbIDs):
        self.ids = pdbIDs
        
  def __len__(self):
        return len(self.ids)

  def __getitem__(self, index): #each example consists of a patterson map, electron density pair
        
        
        ID = self.ids[index]
        X = torch.load('patterson_pt_scaled/' + ID + '_patterson.pt')
        X = torch.unsqueeze(X, 0)
        
        
        for k in examples:
            line = k.split(" ")
            id1 = line[-1][:-4]
            
            if id1 == ID:
                num_res = len(line) - 1
                xlist = []
                for l in range(num_res):
                    cur_res = line[l]
                    res_t = torch.load('9_electron_density_pt_scaled/' + cur_res + '1.pt')
                    xlist.append(res_t)
          
            
        Xlist = torch.stack(xlist)
        X = torch.unsqueeze(X, 0)
        Xlist = torch.unsqueeze(Xlist, 0)

        y = torch.load('electron_density_pt_scaled/' + ID + '_fft.pt')
        y = torch.unsqueeze(y, 0)
        
        

        return X, Xlist, y  



#dataset_train = Dataset(trainlist)
#dataset_val = Dataset(testlist)
dataset_val = Dataset(full_list)

epsilon = 1e-8

def pearson_r_loss(output, target): #calculate pearson r coefficient for central region
    x = output
    y = target 
    
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx)) + epsilon) * torch.sqrt(torch.sum(torch.square(vy)) + epsilon)))
    return cost
    
    
def pearson_r_loss2(output, target): #calculate pearson r coefficient for central region

    x = output[0,:,:,:,:]
    y - target[0,:,:,:,:]
      
    
    batch = x.shape[0]
    cost = 0.0
    
    for i in range(batch):
    
        curx = x[i,:,:,:]
        cury = y[i,:,:,:]
        
        vx = curx - torch.mean(curx)
        vy = cury - torch.mean(cury)

        cost += (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx)) + epsilon) * torch.sqrt(torch.sum(torch.square(vy)) + epsilon)))
    return (cost / batch)
    
def fft_loss(patterson, electron_density):
    patterson = patterson[0,0,0,:,:]
    electron_density = electron_density[0,0,:,:,:]
    f1 = torch.fft.fftn(electron_density)
    f2 = torch.fft.fftn(torch.roll(torch.flip(electron_density, [0, 1, 2]), shifts=(1, 1, 1), dims=(0, 1, 2)))
    f3 = torch.mul(f1,f2)
    f4 = torch.fft.ifftn(f3)
    f4 = f4.real

    vx = f4 - torch.mean(f4)
    vy = patterson - torch.mean(patterson)

    cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx)) + epsilon) * torch.sqrt(torch.sum(torch.square(vy)) + epsilon)))
    return cost

test_loader = torch.utils.data.DataLoader(dataset=dataset_val, shuffle = False, batch_size= 1, num_workers = 4, pin_memory = True)

parser = argparse.ArgumentParser(description='simple distributed training job')

parser.add_argument('--total_epochs', default=99, type=int, help='Total epochs to train the model')
parser.add_argument('--lr_lambda', default=2, type=int, help='lr scheduler')
parser.add_argument('--max_frame_size',default=88, type=int, help='max size')
parser.add_argument('--max_image_height',default=72, type=int, help='max size')
parser.add_argument('--max_image_width',default=60, type=int, help='max size')
parser.add_argument('--ps_size',default=24, type=int, help='max size')
parser.add_argument('--patch_size',default=4, type=int, help='patch size')
parser.add_argument('--activation',default='tanh', type=str, help='activation function')
parser.add_argument('--FFT', default = False, help='FFT')
parser.add_argument('--iFFT', default = False, help='FFT')
parser.add_argument('--FFT_skip', default = False, help='FFT')
parser.add_argument('--transformer', default='Nystromformer',type=str , help='transformer type: normal or Nystromformer')

parser.add_argument('--dim',default=512, type=int, help='dim')
parser.add_argument('--depth',default=10, type=int, help='depth')
parser.add_argument('--heads',default=8, type=int, help='heads')
parser.add_argument('--mlp_dim',default=2048, type=int, help='mlp_dim')

parser.add_argument('--max_partial_structure',default=15, type=int, help='max number of partial_structure')
parser.add_argument('--same_partial_structure_emb', default = True, help='whether use same partial structure embeding layer each transformer layer')

parser.add_argument('--biggan_block_num',default=2, type=int, help='number of additional biggan block')
args = parser.parse_args()


if args.max_partial_structure>-1:
    model = ViT_vary_encoder_decoder_partial_structure(
        args=args,
        num_partial_structure = args.max_partial_structure, #max number of amino acid (partial structure) 
        image_height = args.max_image_height,          # max image size
        image_width = args.max_image_width,
        frames = args.max_frame_size,               # max number of frames
        image_patch_size = args.patch_size,     # image patch size
        frame_patch_size = args.patch_size,      # frame patch size
        ps_size = args.ps_size,
        dim = args.dim,
        depth = args.depth,
        heads = args.heads,
        mlp_dim = args.mlp_dim,
        same_partial_structure_emb=args.same_partial_structure_emb,
        dropout = 0.1,
        emb_dropout = 0.1,
        biggan_block_num=args.biggan_block_num,
        recycle = False
    ).to(device)
elif args.biggan_block_num>-1:
    model = ViT_vary_encoder_decoder_biggan_block(
        args=args,
        image_height = args.max_image_height,          # max image size
        image_width = args.max_image_width,
        frames = args.max_frame_size,               # max number of frames
        image_patch_size = args.patch_size,     # image patch size
        frame_patch_size = args.patch_size,      # frame patch size
        dim = args.dim,
        depth = args.depth,
        heads = args.heads,
        mlp_dim = args.mlp_dim,
        dropout = 0.1,
        emb_dropout = 0.1,
        biggan_block_num=args.biggan_block_num
    ).to(device)
else:
    model = ViT_vary_encoder_decoder(
        args=args,
        image_height = args.max_image_height,          # max image size
        image_width = args.max_image_width,
        frames = args.max_frame_size,               # max number of frames
        image_patch_size = args.patch_size,     # image patch size
        frame_patch_size = args.patch_size,      # frame patch size
        dim = args.dim,
        depth = args.depth,
        heads = args.heads,
        mlp_dim = args.mlp_dim,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)
    
    
#loading model state
checkpoint = torch.load('state.pth')

model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.MSELoss()

#evaluating model
model.eval() 
losses1 = []    
losses2 = []  
count = 0

with open('predictions/pearson.txt', 'w') as fwrite:
    with torch.no_grad():
        for x, ps, y in test_loader: 
            x, ps, y = x.to(device), ps.to(device), y.to(device)
            yhat = model(x, ps)
            loss1 = fft_loss(x, yhat)
            loss2 = pearson_r_loss(yhat, y)
            
            if count >= len(trainlist):
                fwrite.write(str(float(loss2.item())) + " " + (full_list[count]) + "\n")
                losses1.append(float(loss1.item()))
                losses2.append(float(loss2.item()))
            
            yhatc = yhat.cpu()
            torch.save(yhatc, 'predictions/' + (full_list[count]) + '.pt')
            torch.cuda.empty_cache()
            count += 1


hist, bin_edges = np.histogram(losses1)
print(hist)
print(bin_edges)

print(sum(losses1) / len(losses1))
print(statistics.pstdev(losses1))

hist, bin_edges = np.histogram(losses2)
print(hist)
print(bin_edges)

print(sum(losses2) / len(losses2))
print(statistics.pstdev(losses2))
