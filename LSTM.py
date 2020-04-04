import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = ConvNet()
        self.rnn = nn.LSTM(input_size=500,
                           hidden_size=64,
                           num_layers=1,
                           batch_first=False)
        self.linear = nn.Linear(64, 10)

    def forward(self, out):
        batch_size, timesteps, C, H, W = out.size()
#