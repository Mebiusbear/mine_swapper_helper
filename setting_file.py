import torch.nn as nn
from torchvision import transforms
import torch


def label_index(index,len_label_name):
    z = torch.zeros(len_label_name)
    z[index] = 1
    return index


INIT_SIZE = 24

# 数据准备部分
label_name_dict = {
    "num_1" : "1",
    "num_2" : "2",
    "num_3" : "3",
    "num_4" : "4",
    "num_5" : "5",
    "kong"  : "k",
    "qizi"  : "q",
}
label_name = list(label_name_dict.keys()) 
len_label_name = len(label_name)
label_dict = {
    label_name[i] : label_index(i,len_label_name) for i in range (len_label_name)
}
loader = transforms.Compose([transforms.CenterCrop(INIT_SIZE),
                                transforms.ToTensor()])
def loader_func(img):
    return loader(img).unsqueeze(0).reshape(1,INIT_SIZE,INIT_SIZE)

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


# 训练部分
in_dim= INIT_SIZE*INIT_SIZE
n_hiddle_1=200
n_hiddle_2=50
n_hiddle_3=25

epoch = 20
learning_rate = 1e-3
batch_size = 8
model_name = "model_param/h1_%d_h2_%d_h3_%d_e_%d.pkl"%(n_hiddle_1,n_hiddle_2,n_hiddle_3,epoch)