import torch.nn as nn
# 模型部分
class Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hiddle_1, n_hiddle_2, n_hiddle_3, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hiddle_1), nn.BatchNorm1d(n_hiddle_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hiddle_1, n_hiddle_2), nn.BatchNorm1d(n_hiddle_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hiddle_2, n_hiddle_3), nn.BatchNorm1d(n_hiddle_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hiddle_3, out_dim))
        self.sofmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.sofmax(x)
        return x
