import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(
            self,
            hidden_num = 40,
            dropout = 0.1,
            nonlin = torch.nn.Sigmoid(),
            input_dim = 70

    
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_num)
        self.fc2 = nn.Linear(hidden_num,7)
        self.non_lin = nonlin
        self.dropout = dropout
        
    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.non_lin(hidden)
        output = self.fc2(hidden)
        return output