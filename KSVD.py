import torch

data = torch.randn(2,500)  # output from cnn

# batch size of lstm=1, sequence = 2, dimension of each sequence is 1x500
data_ = torch.unsqueeze(data,0) # 1 2 5(1x500)

test = data_[:,-1,:]