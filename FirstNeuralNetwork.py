import torch
import torch.nn as nn
from icecream import ic


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(3, 2)

    def forward(self, x):
        h = torch.sigmoid(self.linear1(x))
        ic(h)
        o = torch.sigmoid(self.linear2(h))
        ic(o)
        return o

if __name__ == '__main__':
    model = Model()
    print(model)
    print(list(model.parameters()))
    X = torch.randn((3,2))
    ic(X)
    print(model(X))