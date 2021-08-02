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