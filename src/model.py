
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.do1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.do2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.do1(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # x = self.do1(x)

        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.avgpool(x)
        x = x.flatten(1)

        x = F.relu(self.fc1(x))
        x = self.do2(x)
        
        x = self.fc2(x)

        return x