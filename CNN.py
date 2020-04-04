#
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split, cross_val_score

class args:
    def __init__(self):
        self.batch_size = 25
        self.num_epochs = 25
        self.lr = 0.0005

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(6*4*50, 1000)
        self.fc2 = nn.Linear(1000, 500)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out



CNN = ConvNet()

print(CNN())


'''
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM,self).__init__()
        self.cnn = ConvNet()
        self.rnn = nn.LSTM(input_size=500,
                           hidden_size=64,
                           num_layers=1,
                           batch_first=False)
        self.linear = nn.Linear(64,10)

    def forward(self,out):
        batch_size, timesteps, C, H, W = out.size()
'''




