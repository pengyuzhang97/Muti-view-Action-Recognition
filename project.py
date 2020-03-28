# BY Pengyu Zhang

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split, cross_val_score


# Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(12 * 12 * 128, 1000)
        self.fc2 = nn.Linear(1000, 8)

# Forward

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out




#Train the model
model = ConvNet()


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train loop
# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs): #epoch
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward
        outputs = model(images)
        loss = criterion(outputs, labels.long())
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted.long() == labels.long()).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

# Test the model
model.eval()
torch.save(model.layer1.state_dict(), 'net_params_layer1.pkl')
torch.save(model.layer2.state_dict(), 'net_params_layer2.pkl')
torch.save(model.fc1.state_dict(), 'net_params_linear1.pkl')
torch.save(model.fc2.state_dict(), 'net_params_linear2.pkl')
'''
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.long() == labels.long()).sum().item()
    print("accuracy: "+ str(correct/total))
'''
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in a_b_test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.long() == labels.long()).sum().item()
    print("accuracy: "+ str(correct/total))
