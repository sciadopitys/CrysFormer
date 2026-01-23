from torch.utils.data import Dataset

import torch 
import torch.nn as nn
import torch.fft
import torch.cuda
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import math
import statistics
import batchgen.dataset_initial as dataset_init

import model.vit_3d_newps_dist as vit
import random
import argparse

# lower matrix multiplication precision
torch.set_float32_matmul_precision('high')

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

#use GPU if possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
#torch.backends.cudnn.benchmark = True

# load list of indices defining training batches
with open("example_ids/training_indices_new2.txt") as myfile2:
    indices = myfile2.readlines()
indlist  = [x.rstrip() for x in indices]

# load list of test set examples
with open("example_ids/test_new_dataset2.txt") as myfile: 
    testlist = myfile.readlines()
testlist = [x.rstrip() for x in testlist]

# load list of primary sequences for all dataset examples
with open("example_ids/new-dipeptide-AA-type-noclash_new2.list") as myfile1:
    examples = myfile1.readlines()
examples  = [x.rstrip() for x in examples]

# generate training batches
dataset_init.create_batches()

# test set dataset definition
class Dataset(torch.utils.data.Dataset):

  def __init__(self, pdbIDs):
        self.ids = pdbIDs

        
  def __len__(self):
        return len(self.ids)

  def __getitem__(self, index): #each example consists of patterson map and partial structure inputs, and output electron density 
        
        ID = self.ids[index]
      
        # load Patterson map
        X = torch.load('patterson_pt_scaled/' + ID + '_patterson.pt')
        X = torch.unsqueeze(X, 0)
        
        # load and combine set of partial structure maps
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
        
        # introduce dummy dimension for consistency with training batches     
        Xlist = torch.unsqueeze(Xlist, 0)
        X = torch.unsqueeze(X, 0)

        # load ground truth electron density
        y = torch.load('electron_density_pt_scaled/' + ID + '_fft.pt')
        y = torch.unsqueeze(y, 0)
        
        
        return X, Xlist, y       

# create test set
dataset_test = Dataset(testlist)
n_test = float(len(dataset_test))

# training set dataset definition
class Dataset1(torch.utils.data.Dataset):

    def __init__(self, indices): 
        self.indices = indices
        
        
    def __getitem__(self, index):

        # load stored user-generated batches
        X = torch.load('patterson_pt_scaled_newps/train_' + str(index) + '_patterson.pt')  
        PS = torch.load('patterson_pt_scaled_newps/train_' + str(index) + '_ps.pt')
        y = torch.load('electron_density_pt_scaled_newps/train_' + str(index) + '.pt')
      
        return X, PS, y
        
    def __len__(self):

        # number of total batches is one less than number of listed indices
        return len(self.indices) - 1

# create training set with size based on list of indices defining training batches
dataset_train = Dataset1(indlist)
n_train = len(indlist)

# calculate Pearson r correlation coefficient for single pair
def pearson_r_loss(output, target): 

    x = output
    y = target  
    
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx)) + epsilon) * torch.sqrt(torch.sum(torch.square(vy)) + epsilon)))
    return cost
    
# calculate Pearson correlation for a batch    
def pearson_r_loss2(output, target):

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

# calculate a Pearson correlation comparison between transformed prediction and corresponding Patterson input as a sanity check
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


# Create dataloaders. Test set has batch size 1, training set effective batch size specified by generated batches
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, shuffle = True, batch_size= 1, num_workers = 4, pin_memory = True)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test, shuffle = False, batch_size= 1, num_workers = 4, pin_memory = True)


# specify default values for model and training hyperparameters (can also be specified in command line)
parser = argparse.ArgumentParser(description='simple distributed training job')

parser.add_argument('--total_epochs', default=80, type=int, help='Total epochs to train the model')
parser.add_argument('--max_image_height',default=88, type=int, help='max size of Patterson/ground truth in first spatial dimension')
parser.add_argument('--max_image_width',default=72, type=int, help='max size of Patterson/ground truth in second spatial dimension')
parser.add_argument('--max_image_depth',default=60, type=int, help='max size of Patterson/ground truth in third spatial dimension')
parser.add_argument('--ps_size',default=24, type=int, help='maximum side length of cubic partial structures')
parser.add_argument('--patch_size',default=4, type=int, help='patch size (all dimensions)')
parser.add_argument('--activation',default='tanh', type=str, help='final activation function')

parser.add_argument('--dim',default=512, type=int, help='token embedding dimension')
parser.add_argument('--depth',default=6, type=int, help='transformer depth')
parser.add_argument('--heads',default=8, type=int, help='number of attention heads')
parser.add_argument('--mlp_dim',default=512, type=int, help='dimensionality within feedforward MLP')

parser.add_argument('--max_partial_structure',default=15, type=int, help='max number of partial_structures')
parser.add_argument('--same_partial_structure_emb', default = True, help='whether to use a constant partial structure embedding in each transformer layer')

parser.add_argument('--biggan_block_num',default=2, type=int, help='number of post-transformer BigGAN residual convolution')
args = parser.parse_args()

# create model with specified hyperparameters and send it to GPU
model = vit.ViT_vary_encoder_decoder_partial_structure(
    args=args,
    num_partial_structure = args.max_partial_structure, 
    image_height = args.max_image_height,         
    image_width = args.max_image_width,
    frames = args.max_image_depth,              
    image_patch_size = args.patch_size,    
    frame_patch_size = args.patch_size,      
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

# specify main loss function term, learning rate schedule, number of epochs
criterion = nn.MSELoss()
learning_rate = 2.5e-4
n_epochs = args.total_epochs
epoch = -1
accum = 4 #gradient accumulation; effective batch size is 4x

# AdamW optimizer with one-cycle learning rate schedule
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=3e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0014, steps_per_epoch=(len(train_loader) // accum), epochs=n_epochs, pct_start=0.3, three_phase= False, div_factor=2.8, final_div_factor=40)

#loading pretrained model
#checkpoint = torch.load('state.pth')
#model.load_state_dict(checkpoint['model_state_dict'])

#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#loss = checkpoint['loss']
#epoch = checkpoint['epoch']


# account for dummy dimension of desired ground truth
def mse_wrapper_loss(output, target):

    y = torch.squeeze(target, 0)
    return criterion(output, y)

clip = 1.0 #gradient clipping value
while epoch < n_epochs:
    model.train() 
    acc = 0.0 #for reporting current training set loss
    
    if epoch >= 0:
        for i, (x, ps, y) in enumerate(train_loader):
            # load tensors to GPU   
            x, ps, y = x.to(device), ps.to(device), y.to(device)

            #apply model to current example
            yhat = model(x, ps)                 

            # evaluate and add loss function terms
            loss_1 = mse_wrapper_loss(yhat, y)                              
            loss_2 = (1 - pearson_r_loss2(yhat, y))
            loss = (0.9999 * loss_1) + (1e-4 * loss_2)
            acc += float(loss.item())

            #compute and accumulate gradients for model parameters
            loss = loss / accum            #needed due to gradient accumulation
            loss.backward()                                                

            #update model parameters only on accumulation epochs
            if (i+1) % accum == 0:       

                #gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)    #gradient clipping
                
                optimizer.step()                                            

                 #clear (accumulated) gradients
                optimizer.zero_grad()             
                
                scheduler.step()
                torch.cuda.empty_cache()
    
        # save current model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'epoch': epoch + 1,
            }, 'state.pth')
    
    # evaluate test set metrics after each epoch    
    model.eval() 
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    with torch.no_grad(): 
        for x, ps, y in test_loader: 
            x, ps, y = x.to(device), ps.to(device), y.to(device)
            
            yhat = model(x, ps)
            loss3 = criterion(yhat, y)
            loss4 = pearson_r_loss(yhat, y)
            loss5 = fft_loss(x, yhat)
            acc1 += float(loss3.item())
            acc2 += float(loss4.item())
            acc3 += float(loss5.item())
            torch.cuda.empty_cache()
    
    curacc = (acc2 / n_test)
    curacc2 = (acc3 / n_test)

    # report epoch number, average training set loss, standard Pearson, comparison with original Patterson, and last learning rate
    print("%d %.10f %.6f %.6f %.10f" % (epoch, (acc / n_train), curacc, curacc2, scheduler.get_last_lr()[0]))
    
    # every 3 epochs, re-generate training batches to mix examples between batches
    if (epoch % 3 == 0):
        dataset_init.create_batches()  
    
    epoch += 1
