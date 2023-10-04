import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# Encoder Model
# Extract high level features from raw frames
class Phi(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(3,32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(3,32, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(3,32, kernel_size=(3,3), stride=2, padding=1)

    def forward(self, x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(x))
        y = F.elu(self.conv3(x))
        y = F.elu(self.conv4(x))
        y = y.flatten(start_dim=1)
        return y
    
# Inverse Model
# Infer the action from encoded State(t) + State(t+1)
class Gnet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.l1 = nn.Linear(576, 256)
        self.l2 = nn.Linear(256, 12)
    
    def forward(self, state1, state2):
        x = torch.cat((state1,state2), dim=1)
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return F.softmax(y, dim = 1)


# Forward Prediction Model
# Infer the next encoded state 
# from the previous encoded state and the action
class Fnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(300, 256)
        self.l2 = nn.Linear(256, 288)
    
    def forward(self, state, action):
        action_ = torch.zeros(action.shape[0], 12)
        indices = torch.stack( (torch.arange(action.shapep[0]), action.squeeze()), dim=0 )
        indices = indices.tolist()
        action_[indices] = 1.
        x = torch.cat( (state, action_), dim=1 )
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return y

# Qnetwork to infer QValues
class Qnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                              kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32,32,kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32,32,kernel_size=(3,3), stride=2, padding=1)
        self.l1 = nn.Linear(288,100)
        self.l2 = nn.Linear(100,12)

    def forward(self, x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.flatten(start_dim = 2)
        y = y.view(y.shape[0],-1,32)
        y = y.flatten(start_dim = 1)
        y = F.elu(self.l1(y))
        y = self.l2(y)
        return y
        



        
        

