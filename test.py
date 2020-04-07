#
import pickle
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
        self.linear = nn.Linear(64,10) # h->out requiring passing a fully-connected layer to match # of labels which is 11 different action
    def forward(self,x):
        c_out = self.cnn(x)
        c_out = torch.unsqueeze(c_out,1)
        h0 = torch.randn(1,1,64) # initialize h0
        c0 = torch.randn(1,1,64) # initialize c0
        r_out, (h, c) = self.lstm(c_out,(h0,c0))
        r_out = self.linear(r_out)
        h = self.linear(h)
        return r_out, h, c


'''data1 = torch.randn(25,1,64,48)
# data1 should be from camera1, include 11 different actions
# elements of data1 is frame, so I use 25 as cnn_batch_size to get time series, and the channel is 1 because of gray scale images.
# input of cnn: batch size = 25 which means 25 frames; channels = 1; 64x48->pixels of each frame
# output from cnn should be 1x500
# input of lstm = 25x500; # of sequence should be 25, batch size is 1

data2 = torch.randn(25,1,64,48)
data3 = torch.randn(25,1,64,48)
data4 = torch.randn(25,1,64,48)
data5 = torch.randn(25,1,64,48)# the same defination as data1
'''



module1 = CNN_LSTM()# camera 1
'''module2 = CNN_LSTM()# camera 2
module3 = CNN_LSTM()# camera 3
module4 = CNN_LSTM()# camera 4
module5 = CNN_LSTM()# camera 5
# requires 5 datasets, and each of them contains data from one camera
'''


'''class dataload:
    def __init__(self,data):
        self.data = data
        self.data1 = torch.unsqueeze(self.data,1)
        self.labels = np.ones(len(self.data1))
        for'''

'''============================================================================================================================'''
'''==============================================='''
'''type and format of my input dataset'''
'''==============================================='''
# assuming I have 50 images from camera 1
data1 = torch.randn(120*arg.batch_size, 64, 48)

# Then I need to add one dimension which represents number of channels
data1 = torch.unsqueeze(data1, 1)


# create labels
labels = torch.tensor(np.repeat(np.arange(0,10),12))

# concatenate data and labels for future usage
'''train_dataset1 = []
for i in range(len(data1)):
	train_dataset1.append((data1[i],labels[i]))'''


# Until now I have created a training dataset which contains 50 elements(images), and all of them have already been labeled
# Nest step is to using 'Dataloader' to create batch size

dataloader1 = DataLoader(dataset=data1, batch_size=50, shuffle=False)

'''=========================================================================================================================='''



# training process
# Forget about the arg class above, let's testing our code under the condition that epoch is only 1

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(module1.parameters(), lr=arg.lr)

# train model
loss_list = np.zeros(120)
output = torch.tensor([])
acc_list = []
#output = torch.zeros(len(dataloader1),100) # 100 will not change unless changing fully connectted layer

for epoch in range(arg.epoch):
    for i, images in enumerate(dataloader1):

        '''Forward'''
        r_out, h, _ = module1(images) # the last vector of r_out will be my feature vector
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
        correct = (predicted.long() == labels.long()).sum().item()
        acc_list.append(correct/total)
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, arg.epoch, i + 1, len(dataloader1), loss.item(),
                          (correct / total) * 100))
            print(r_out[0])
        #loss_list[i] = loss
        '''output = torch.cat([output,torch.squeeze(out,dim=1)], dim=0)
        loss = criterion(output, labels[i].unsqueeze(0).long())'''
        #loss_list.append(loss.item())


        '''output = torch.tensor(np.array(output.append(out)))
        loss = criterion(output, labels)
        loss_list.append(loss.item())'''

'''I'm planning to implement mini-batch, the batch size will be 12 considering the total number of dataloader1 is 120, and the number of batch will be 10 '''
'''I am still going to us 11 different actions.'''
'''The next problem will be how to transfer my video to dataloader1'''

'''************************************************************'''
'''I need the last vector of r_out as my feature vector'''
'''************************************************************'''