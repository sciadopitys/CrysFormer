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
import batchgen.dataset_initial

from model.vit_3d_newps_dist import ViT_encoder_decoder, ViT_vary_encoder_decoder, ViT_vary_encoder_decoder_partial_structure, ViT_vary_encoder_decoder_biggan_block
import random
import argparse

torch.set_float32_matmul_precision('high')

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #use GPU if possible
torch.backends.cudnn.benchmark = True

with open("example_ids/training_indices_new2.txt") as myfile2:
    indices = myfile2.readlines()
indlist  = [x.rstrip() for x in indices]

with open("example_ids/test_new_dataset2.txt") as myfile: #select the first n_train examples as the training set, rest as validation set
    testlist = myfile.readlines()
testlist = [x.rstrip() for x in testlist]

with open("example_ids/new-dipeptide-AA-type-noclash_new2.list") as myfile1:
    examples = myfile1.readlines()
examples  = [x.rstrip() for x in examples]


dataset_initial.create_batches()



class Dataset(torch.utils.data.Dataset):

  def __init__(self, pdbIDs):
        self.ids = pdbIDs

        
  def __len__(self):
        return len(self.ids)

  def __getitem__(self, index): #each example consists of a patterson map, electron density pair
        
        
        ID = self.ids[index]
        X = torch.load('patterson_pt_scaled/' + ID + '_patterson.pt')
        X = torch.unsqueeze(X, 0)

        #X1 = torch.load('predictions_15_angle_retrain/' + ID + '.pt')
        #X1 = X1[0,0,:,:,:]
            
        #X1 = torch.unsqueeze(X1, 0)
        
        
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
        
            
        #X_comb = torch.cat((X, X1), 0)
        #X = torch.unsqueeze(X_comb, 0)
        Xlist = torch.unsqueeze(Xlist, 0)
        X = torch.unsqueeze(X, 0)
        y = torch.load('electron_density_pt_scaled/' + ID + '_fft.pt')
        y = torch.unsqueeze(y, 0)
        
        
        return X, Xlist, y       


dataset_val = Dataset(testlist)
n_test = float(len(dataset_val))


class Dataset1(torch.utils.data.Dataset):

    def __init__(self, indices): 
        self.indices = indices
        
        
    def __getitem__(self, index):
    
        X = torch.load('patterson_pt_scaled_newps/train_' + str(index) + '_patterson.pt')  
        PS = torch.load('patterson_pt_scaled_newps/train_' + str(index) + '_ps.pt')
        y = torch.load('electron_density_pt_scaled_newps/train_' + str(index) + '.pt')
      
        return X, PS, y
        
    def __len__(self):
        return len(self.indices) - 1
        
dataset_train = Dataset1(indlist)
n_train = len(indlist)

epsilon = 1e-8

def pearson_r_loss(output, target): #calculate pearson r coefficient for central region

    x = output
    y = target  
    
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx)) + epsilon) * torch.sqrt(torch.sum(torch.square(vy)) + epsilon)))
    return cost
    
    
def pearson_r_loss3(output, target): #calculate pearson r coefficient for central region

    x = output[:,0,:,:,:]
    if target.dim() > 5:
        y = torch.squeeze(target, 0)[:,0,:,:,:]
    else:
        y = target[:,0,:,:,:]
    
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

#create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, shuffle = True, batch_size= 1, num_workers = 4, pin_memory = True)
test_loader = torch.utils.data.DataLoader(dataset=dataset_val, shuffle = False, batch_size= 1, num_workers = 4, pin_memory = True)


parser = argparse.ArgumentParser(description='simple distributed training job')

parser.add_argument('--total_epochs', default=80, type=int, help='Total epochs to train the model')
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
parser.add_argument('--depth',default=6, type=int, help='depth')
parser.add_argument('--heads',default=8, type=int, help='heads')
parser.add_argument('--mlp_dim',default=512, type=int, help='mlp_dim')

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



#loading pretrained model
#checkpoint = torch.load('state_15_var_retrain2.pth')
#model.load_state_dict(checkpoint['model_state_dict'])

#specify loss function, learning rate schedule, number of epochs
criterion = nn.MSELoss()
learning_rate = 2.5e-4
n_epochs = args.total_epochs
epoch = -1
accum = 4 #gradient accumulation; effective batch size is 13x
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=3e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0014, steps_per_epoch=(len(train_loader) // accum), epochs=n_epochs, pct_start=0.3, three_phase= False, div_factor=2.8, final_div_factor=40)

#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#loss = checkpoint['loss']
#epoch = checkpoint['epoch']

def mse_wrapper_loss(output, target):

    y = torch.squeeze(target, 0)
    return criterion(output, y)

clip = 1.0 #gradient clipping value
test1 = []
test2 = []
while epoch < n_epochs:
    model.train() 
    acc = 0.0
    
    if epoch >= 0:
        for i, (x, ps, y) in enumerate(train_loader):
              
            x, ps, y = x.to(device), ps.to(device), y.to(device)
            yhat = model(x, ps)                                             #apply model to current example
            loss_1 = mse_wrapper_loss(yhat, y)                              #evaluate loss
            loss_2 = (1 - pearson_r_loss2(yhat, y))
            loss = (0.9999 * loss_1) + (1e-4 * loss_2)
            acc += float(loss.item())
            loss = loss / accum            #needed due to gradient accumulation
            loss.backward()                                                 #compute and accumulate gradients for model parameters
            if (i+1) % accum == 0:                                          
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)    #gradient clipping
                optimizer.step()                                            #update model parameters only on accumulation epochs
                optimizer.zero_grad()                                       #clear (accumulated) gradients
                scheduler.step()
                torch.cuda.empty_cache()
    
    
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'epoch': epoch + 1,
            }, 'state.pth')
    
        
    model.eval() 
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    acc4 = 0.0
    with torch.no_grad(): #calculate loss and pearson r for all validation set elements
        for x, ps, y in test_loader: 
            x, ps, y = x.to(device), ps.to(device), y.to(device)
            
            yhat = model(x, ps)
            loss3 = criterion(yhat, y)
            loss4 = pearson_r_loss(yhat, y)
            loss5 = fft_loss(x, yhat)
            loss6 = pearson_r_loss2(yhat, y)
            acc1 += float(loss3.item())
            acc2 += float(loss4.item())
            acc3 += float(loss5.item())
            acc4 += float(loss6.item())
            torch.cuda.empty_cache()
    
    #store average value of metrics
    test1.append(acc1 / n_test)
    curacc = (acc2 / n_test)
    curacc2 = (acc3 / n_test)
    curacc3 = (acc4 / n_test)
    test2.append(curacc)
    print("%d %.10f %.6f %.6f %.6f %.10f" % (epoch, (acc / n_train), curacc, curacc2, curacc3, scheduler.get_last_lr()[0]))
    
    
    if (epoch % 3 == 0):
        dataset_initial.create_batches()  
    
    epoch += 1