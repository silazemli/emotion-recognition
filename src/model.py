
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout()

        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        # self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d((2, 2))

        self.halfpool = nn.MaxPool2d((2, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(64, num_classes)

        self.fc3 = nn.Linear(25600, 512) # absolute retardation
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        # x = self.dropout(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.dropout(x)
        x = self.pool(x)
        # x = F.relu(self.bn4(self.conv4(x)))
        x = x.flatten(1)

        # x = self.avgpool(x)
        # x = x.flatten(1)

        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # x = self.fc2(x)

        return x