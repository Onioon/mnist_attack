import torch
from torch import nn
from torch.nn.modules import dropout

class Mnist_FCNN(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_class)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.model(x)
        return x


class MetaClassifier(nn.Module):
    def __init__(self):
        super(MetaClassifier, self).__init__()
        self.phi1 = nn.Sequential(
            nn.Linear(785, 1),
            nn.ELU(inplace = True)
        )
        self.phi2 = nn.Sequential(
            nn.Linear(513, 1),
            nn.ELU(inplace = True)
        )
        self.phi3 = nn.Sequential(
            nn.Linear(513, 1),
            nn.ELU(inplace = True)
        )
        self.rho = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3):
        n1 = self.phi1(x1)
        L1 = n1.sum(1)

        n1 = n1.permute(0,2,1).expand(-1,256,-1)
        x2 = torch.cat((x2,n1),2)
        n2 = self.phi2(x2)
        L2 = n2.sum(1)

        n2 = n2.permute(0,2,1).expand(-1,2,-1)
        x3 = torch.cat((x3,n2),2)
        n3 = self.phi3(x3)
        L3 = n3.sum(1)

        concat = torch.cat((L1, L2, L3), 1)
        return self.rho(concat)