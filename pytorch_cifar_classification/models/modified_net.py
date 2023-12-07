import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)  # Increased kernel size and output channels
        self.bn1 = nn.BatchNorm2d(8)  # Added Batch Normalization
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)  # Reduced output channels, increased kernel size
        self.bn2 = nn.BatchNorm2d(12)  # Added Batch Normalization
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(12, 16, 3, padding=1)  # Further reduced output channels
        self.bn3 = nn.BatchNorm2d(16)  # Added Batch Normalization
        self.pool3 = nn.MaxPool2d(2, 2)

        # Define FC layers with modifications
        self.fc1 = nn.Linear(16 * 4 * 4, 100)  # Adjusted for the new dimensions and reduced neurons
        self.dropout1 = nn.Dropout(0.5)  # Added Dropout
        self.fc2 = nn.Linear(100, 64)
        self.dropout2 = nn.Dropout(0.5)  # Added Dropout
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Add sequence of CONV, BN, POOL layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = torch.flatten(x, 1)

        # Add FC layers with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
