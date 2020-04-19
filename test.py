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
        self.lr = 0.0005
        self.epoch = 1

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

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM,self).__init__()
        self.cnn = ConvNet()
        self.lstm = nn.LSTM(input_size=500,
                           hidden_size=64,
                           num_layers=1, # h and c both have one layer
                           batch_first=False)
        self.linear = nn.Linear(64,11) # h->out requiring passing a fully-connected layer to match # of labels which is 11 different action
    def forward(self,x):
        c_out = self.cnn(x)
        c_out = torch.unsqueeze(c_out,1)
        h0 = torch.randn(1,1,64) # initialize h0
        c0 = torch.randn(1,1,64) # initialize c0
        r_out, (h, c) = self.lstm(c_out,(h0,c0))
        r_out = self.linear(r_out)
        h = self.linear(h)
        return r_out, h, c, c_out

model1 = CNN_LSTM()# camera 1
model2 = CNN_LSTM()# camera 2
model3 = CNN_LSTM()# camera 3
model4 = CNN_LSTM()# camera 4
model5 = CNN_LSTM()# camera 5
# requires 5 datasets, and each of them contains data from one camera


#data1 = torch.randn(120*arg.batch_size, 64, 48)
data = np.load('data from cam1.npy')
data_ = torch.FloatTensor(data)
# Then I need to add one dimension which represents number of channels
data1 = torch.unsqueeze(data_, 1)
# create labels
label_in = np.load('labels from cam1.npy')
labels = torch.tensor(label_in)
dataloader1 = DataLoader(dataset=data1, batch_size=50, shuffle=False)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=arg.lr)
# train model
loss_list = np.zeros(len(labels))
output = torch.tensor([])
correct = 0
acc_list = []
#output = torch.zeros(len(dataloader1),100) # 100 will not change unless changing fully connectted layer

for epoch in range(arg.epoch):
    for i, images in enumerate(dataloader1):

        '''Forward'''
        r_out, h, _, c_out = model1(images) # the last vector of r_out will be my feature vector
        output = torch.squeeze(h,1)
        loss = criterion(output, labels[i].unsqueeze(0).long())
        loss_list[i] = loss

        '''Backpropgation'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''Track accuracy'''
        total = labels.size(0)
        _, predicted = torch.max(output.data,1)
        if predicted.long() == labels[i].long():
            correct = correct+1
        #correct = (predicted.long() == labels[i].long()).sum().item()
        acc_list.append(correct/total)

        if (i+10) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, arg.epoch, i + 1, len(dataloader1), loss.item(),
                          (correct / total) * 100))
        if i == len(dataloader1)-1:
            print('Feature vector: {}%'.format(r_out[-1]))



