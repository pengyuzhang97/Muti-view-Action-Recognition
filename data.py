import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


#data1 = torch.randn(120*arg.batch_size, 64, 48)
data = np.load('data from cam1.npy')
data_ = torch.FloatTensor(data)
# Then I need to add one dimension which represents number of channels
data1 = torch.unsqueeze(data_, 1)
# create labels
label_in = np.load('labels from cam1.npy')
labels = torch.tensor(label_in)

data1111 = data1[0:50,:,:,:]
data_set = []
for  i in range(int(len(data1)/50)):
    data_set.append((data1[i*50:i*50+50,:,:,:], labels[i]))




dataloader1 = DataLoader(dataset=data_set, batch_size=50, shuffle=True)