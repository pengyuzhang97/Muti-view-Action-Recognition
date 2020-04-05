import torch
import torch.nn as nn
import numpy as np

data_ = torch.randn(5,25,64,48)
labels = np.ones(5)

train_dataset = []
for i in range(len(data_)):
	train_dataset.append((data_[i],labels[i]))