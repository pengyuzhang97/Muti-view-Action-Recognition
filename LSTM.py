import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=500,
                           hidden_size=64,
                           num_layers=25,
                           batch_first=False)
        self.linear = nn.Linear(64,10)

    def forward(self,x):
        out, (h_n,c_n) = self.lstm(data_)
        lstm_out = self.linear(out[:,-1,:])
        return lstm_out, h_n, c_n

data = torch.randn(25,500)  # output from cnn
# sequence = 2,!!batch size of lstm=1!!, dimension of each sequence is 1x500
# num_layers should euqal to batch size of cnn in my project
data_ = torch.unsqueeze(data,1) # 25 1 5(1x500)


module = LSTM()
out, h, c = module(data_)

