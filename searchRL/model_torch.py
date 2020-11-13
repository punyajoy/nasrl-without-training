import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,actions):
        super(Net, self).__init__()
        kernel_1, filters_1, kernel_2, filters_2, kernel_3, filters_3, kernel_4, filters_4 = actions
        self.conv1 = nn.Conv2d(3, filters_1, kernel_1,stride=2)
        self.conv2 = nn.Conv2d(filters_1, filters_2, kernel_2,stride=1)
        self.conv3 = nn.Conv2d(filters_2, filters_3, kernel_3,stride=2)
        self.conv4 = nn.Conv2d(filters_3, filters_4, kernel_4,stride=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        finaldimension=self.get_final_dim(actions)
        self.fc3 = nn.Linear(filters_4*finaldimension*finaldimension,10)
    
    def get_final_dim(self,actions):
        in_dim=32
        for i in range(0,len(actions),2):
            in_dim = int((in_dim+2*1 - 1*(actions[i]-1))/2)
        return in_dim
    
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc3(x)
        return x

