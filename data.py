import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# assuming I have 50 images from camera 1
data1 = torch.randn(120*50,64,48)

# Then I need to add one dimension which represents number of channels
data1 = torch.unsqueeze(data1,1)


# create labels
labels = torch.tensor(np.array(np.arange(1,121)))

train_loader1 = DataLoader(dataset=data1, batch_size=50,shuffle=False)




loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)