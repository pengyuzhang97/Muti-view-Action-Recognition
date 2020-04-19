import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

data1 = torch.randn(120*50, 64, 48, 3)
labels = np.load('labels from cam1.npy')

data = np.load('data from cam1.npy')
data_ = torch.FloatTensor(data)
