import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# assuming I have 50 images from camera 1
data1 = torch.randn(120*50,64,48)

# Then I need to add one dimension which represents number of channels
data1 = torch.unsqueeze(data1,1)


# create labels
labels = torch.tensor(np.repeat(np.arange(1,11),12))

dataloader1 = DataLoader(dataset=data1, batch_size=50, shuffle=False)


for i, images in enumerate(dataloader1):
    print(i)
    print(images)

# i = 0-119 index
#

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)