
#
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split, cross_val_score

class args:
    def __init__(self):
        self.batch_size = 50
        self.num_epochs = 1
        self.lr = 0.00005
        self.epoch = 90

arg = args()

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
        self.drop_out = nn.Dropout(0.5)
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

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM,self).__init__()
        self.cnn = ConvNet()
        self.lstm = nn.LSTM(input_size=500,
                           hidden_size=64,
                           num_layers=1,
                           batch_first=False)
        self.linear = nn.Linear(64,11)
    def forward(self,x):
        c_out = self.cnn(x)
        c_out = torch.unsqueeze(c_out,1)
        h0 = torch.randn(1,1,64) # initialize h0
        c0 = torch.randn(1,1,64) # initialize c0
        r_out, (h, c) = self.lstm(c_out,(h0,c0))
        h = self.linear(h)
        return r_out, h, c, c_out


model = CNN_LSTM()

##################################################################################################
model.cnn.layer1.load_state_dict(torch.load('5net_params_layer1.pkl'))
model.cnn.layer2.load_state_dict(torch.load('5net_params_layer2.pkl'))
model.cnn.layer3.load_state_dict(torch.load('5net_params_layer3.pkl'))
model.cnn.fc1.load_state_dict(torch.load('5net_params_linear1.pkl'))
model.cnn.fc2.load_state_dict(torch.load('5net_params_linear2.pkl'))

model.lstm.load_state_dict(torch.load('5net_params_lstm.pkl'))
model.linear.load_state_dict(torch.load('5net_params_linear.pkl'))
####################################################################################################

'''data = np.load('test data from cam5.npy') #############################################################################################
data_ = torch.FloatTensor(data)
# Then I need to add one dimension which represents number of channels
data1 = torch.unsqueeze(data_, 1)

# create labels
label_in = np.load('test labels from cam5.npy')##########################################################################################
labels = torch.tensor(label_in)

data_set = []
for  i in range(int(len(data1)/50)):
    data_set.append((data1[i*50:i*50+50,:,:,:], labels[i]))

testloader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)'''

with open('test data from cam5.txt','rb') as f:###########################################################
    test_loader = pickle.load(f)


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels_) in enumerate(test_loader):
        images = torch.squeeze(images, 0)
        '''Forward'''
        r_out, h, _, c_out = model(images)
        output = torch.squeeze((h), 1)
        _, predicted = torch.max(output.data, 1)
        if predicted.long() == labels_.long():
            correct = correct+1
            #print('ture is', labels_.long())

        #correct += (predicted.long() == labels_.long()).sum().item()