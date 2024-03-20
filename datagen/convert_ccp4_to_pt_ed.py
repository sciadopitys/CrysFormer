import gemmi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import sys
import torch


list_file = sys.argv[1]
split = sys.argv[2]
with open(list_file) as myfile2: #select the first n_train examples as the training set, rest as validation set
    ids = myfile2.readlines()
idlist  = [x.rstrip() for x in ids]

for x in idlist:
    input_file = '5_electron_density_ccp4_res_' + split + '/' + x + '_fft.ccp4'
    ccp4_map = gemmi.read_ccp4_map(input_file, setup = True)
    vectors = np.array(ccp4_map.grid, copy=False)
    tensor = torch.Tensor(vectors)
    torch.save(tensor, '7_electron_density_pt_' + split + '/' + x + '_fft.pt')