import torch 
import torch.fft
import torch.cuda

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import random
def shuffle_slice(a, start, stop):
    i = start
    while (i < (stop-1)):
        idx = random.randrange(i, stop)
        a[i], a[idx] = a[idx], a[i]
        i += 1


def create_batches():
    
    with open("example_ids/train_new_dataset2.txt") as myfile1: #select the first n_train examples as the training set, rest as validation set
        trainlist = myfile1.readlines()
    trainlist  = [x.rstrip() for x in trainlist]
    

    with open("example_ids/size_indices_new1.txt") as myfile: #select the first n_train examples as the training set, rest as validation set
        sindices = myfile.readlines()
    sindices  = [x.rstrip() for x in sindices]
    
    for i in range(len(sindices) - 1):
        start = int(sindices[i])
        end = int(sindices[i+1])
        shuffle_slice(trainlist, start, end)
        
    with open("example_ids/training_indices_new2.txt") as myfile2: #select the first n_train examples as the training set, rest as validation set
        indices = myfile2.readlines()
    indices  = [x.rstrip() for x in indices]

    
    with open("example_ids/new-dipeptide-AA-type-noclash_new2.list") as myfile1:
        examples = myfile1.readlines()
    examples  = [x.rstrip() for x in examples]
    

    for i in range(len(indices) - 1):
        start = int(indices[i])
        end = int(indices[i+1])
        xlist = []
        pslist = []
        ylist = []
        for j in range(start, end):
        
            new_x = torch.load('patterson_pt_scaled/' + trainlist[j] + '_patterson.pt')
            new_x = torch.unsqueeze(new_x, 0)

            
            #new_x3 = torch.load('predictions/' + trainlist[j] + '.pt')
            #new_x3 = new_x3[0,0,:,:,:]
            #new_x3 = torch.unsqueeze(new_x3, 0)
            
            
            for k in examples:
                line = k.split(" ")
                id = line[-1][:-4]
                
                if id == trainlist[j]:
                    num_res = len(line) - 1
                    xlist1 = []
                    for l in range(num_res):
                        cur_res = line[l]
                        res_t = torch.load('9_electron_density_pt_scaled/' + cur_res + '1.pt')
                        xlist1.append(res_t)
                    
            new_xlist = torch.stack(xlist1)
            
            
            
            #new_xcomb = torch.cat((new_x, new_x3), 0)
            #xlist.append(new_xcomb)
            xlist.append(new_x)
            pslist.append(new_xlist)
            new_y = torch.load('electron_density_pt_scaled/' + trainlist[j] + '_fft.pt')
            new_y = torch.unsqueeze(new_y, 0)
            ylist.append(new_y)
        
        data_x = torch.stack(xlist)
        data_ps = torch.stack(pslist)
        data_y = torch.stack(ylist)
        torch.save(data_x, 'patterson_pt_scaled_newps/train_' + str(i) + '_patterson.pt')  
        torch.save(data_ps, 'patterson_pt_scaled_newps/train_' + str(i) + '_ps.pt')
        torch.save(data_y, 'electron_density_pt_scaled_newps/train_' + str(i) + '.pt')
        