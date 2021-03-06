#
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split, cross_val_score


def testdataset():
    global test_dataset

class args:
    def __init__(self):
        self.batch_size = 50
        self.num_epochs = 1
        self.lr = 0.00005
        self.epoch = 90

arg = args()


def decision(labels_):
    if labels_.long() == 0:
        vector[0, :] = r_out[-1]

    if labels_.long() == 1:
        vector[1, :] = r_out[-1]

    if labels_.long() == 2:
        vector[2, :] = r_out[-1]

    if labels_.long() == 3:
        vector[3, :] = r_out[-1]

    if labels_.long() == 4:
        vector[4, :] = r_out[-1]

    if labels_.long() == 5:
        vector[5, :] = r_out[-1]

    if labels_.long() == 6:
        vector[6, :] = r_out[-1]

    if labels_.long() == 7:
        vector[7, :] = r_out[-1]

    if labels_.long() == 8:
        vector[8, :] = r_out[-1]

    if labels_.long() == 9:
        vector[9, :] = r_out[-1]

    if labels_.long() == 10:
        vector[10, :] = r_out[-1]


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=2, stride=1, padding=0),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout(p=0.5)
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

model1 = CNN_LSTM()# camera 1
model2 = CNN_LSTM()# camera 2
model3 = CNN_LSTM()# camera 3
model4 = CNN_LSTM()# camera 4
model5 = CNN_LSTM()# camera 5


data = np.load('data from cam5.npy') #############################################################################################
data_ = torch.FloatTensor(data)
# Then I need to add one dimension which represents number of channels
data1 = torch.unsqueeze(data_, 1)

# create labels
label_in = np.load('labels from cam5.npy')##########################################################################################
labels = torch.tensor(label_in)

data_set = []
for  i in range(int(len(data1)/50)):
    data_set.append((data1[i*50:i*50+50,:,:,:], labels[i]))

train_loader, test_loader = train_test_split(data_set, test_size = 1/3)
with open('test data from cam5.txt', 'wb') as f:###########################################################
    pickle.dump(test_loader, f)




dataloader1 = DataLoader(dataset=train_loader, batch_size=1, shuffle=True)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model5.parameters(), lr=arg.lr)##################################################################################

# train model
loss_list = []
output = torch.tensor([])
correct = np.zeros(arg.epoch)
acc_list = []
total = np.zeros(arg.epoch)
vector = torch.zeros((11,64))
#output = torch.zeros(len(dataloader1),100) # 100 will not change unless changing fully connectted layer

for epoch in range(arg.epoch):
    for i, (images, labels_) in enumerate(dataloader1):
        images = torch.squeeze(images,0)
        '''Forward'''
        r_out, h, _, c_out = model5(images) ########################################################################################################
        output = torch.squeeze(h,1)
        loss = criterion(output, labels_.long())
        loss_list.append(loss.item())

        '''Backpropgation'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''Track accuracy'''
        total[epoch] = labels.size(0)
        _, predicted = torch.max(output.data,1)
        if predicted.long() == labels_.long():
            correct[epoch] = correct[epoch]+1
            decision(labels_)
        acc_list.append(correct/total[epoch])


        if (i+10) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, arg.epoch, i + 1, len(dataloader1), loss.item(),
                          (correct[epoch] / total[epoch]) * 100))
        '''if i == len(dataloader1)-1:
            print('Feature vector: {}%'.format(r_out[-1]))'''


##########################################################################################################
torch.save(model5.cnn.layer1.state_dict(), '5net_params_layer1.pkl')
torch.save(model5.cnn.layer2.state_dict(), '5net_params_layer2.pkl')
torch.save(model5.cnn.layer3.state_dict(), '5net_params_layer3.pkl')
torch.save(model5.cnn.fc1.state_dict(), '5net_params_linear1.pkl')
torch.save(model5.cnn.fc2.state_dict(), '5net_params_linear2.pkl')

torch.save(model5.lstm.state_dict(), '5net_params_lstm.pkl')
torch.save(model5.linear.state_dict(), '5net_params_linear.pkl')

np.save('loss_list 5',loss_list)
np.save('correct 5', correct)
np.save('feature vector 5',vector.detach().numpy())
##############################################################################################################
