from torch.utils.data import Dataset,DataLoader  # 导入Dataset后可以使用“help(Dataset)查看官方文档”
from PIL import Image                 # 借助PIL库导入数据图片
import os                             # 借助os库来用路径读入数据
from torchvision import transforms
import torch.nn as nn
import torch
import numpy as np
from torch import optim, nn
from torch.autograd import Variable
import shutil

import __init__

def label_index(index,len_label_name):
    z = torch.zeros(len_label_name)
    z[index] = 1
    return z

def get_pre_name(pre):
    pre = pre.detach().numpy().tolist()[0]
    max1 = max(pre)
    pre_index = pre.index(max1)
    pre_name = label_name[pre_index]
    return pre_name



loader = transforms.Compose([transforms.ToTensor()])


label_name = ["num_1","num_2","num_3","num_4","num_5","lei","kong","zha","qizi"]
len_label_name = len(label_name)
label_dict = {
    label_name[i] : label_index(i,len_label_name) for i in range (len_label_name)
}

batch_size = 5
epoch = 100
learning_rate = 1e-3

class Mydata(Dataset):                  # 根据官方文档，自己创建的类必须继承Dataset
    def __init__(self,root_dir,label_dir):          # 初始化操作,传入图片所在的根目录路径（root_dir）和label的路径（label_dir）获得一个路径列表（img_path）
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)  # 用join把路径拼接一起可以避免一些因“/”引发的错误
        self.img_path = os.listdir(self.path)                   # 将该路径下的所有文件变成一个列表
        self.imgs = [os.path.join(self.root_dir,self.label_dir,img) for img in self.img_path]

    def __getitem__(self,idx):                                # 使用index(简写为idx)获取某个数据
        # img_name = self.img_path[idx]                           # img_path列表里每个元素就是对应图片文件名
        # img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)	# 获得对应图片路径
        img = Image.open(self.imgs[idx]).convert("L")                                   # 使用PIL库下Image工具，打开对应路径图片
        img = loader(img).unsqueeze(0).reshape(1,30,30)
        label = self.label_dir                                              # 本数据集label就是文件名，如“ants”（虽然命名为dir看似路径，实则视作字符串会更容易理解）
        label = label_dict[label]
        return img,label     # 返回对应图片和图片的label

    def __len__(self):
        return len(self.imgs)



root_dir = "./pic"
train_dataset = Mydata(root_dir,label_name[0])
for ln in label_name[1:]:
    train_dataset += Mydata(root_dir,ln)

train_dataset = DataLoader(dataset=train_dataset, batch_size=batch_size,num_workers=0,shuffle=True)
train_dataset = list(train_dataset)

class Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hiddle_1, n_hiddle_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hiddle_1), nn.BatchNorm1d(n_hiddle_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hiddle_1, n_hiddle_2), nn.BatchNorm1d(n_hiddle_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hiddle_2, out_dim))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def train():
    criterion = nn.BCEWithLogitsLoss()
    model = Batch_Net(in_dim=30*30,n_hiddle_1=300,n_hiddle_2=100,out_dim=len_label_name)
    opitimizer = optim.SGD(model.parameters(), lr=learning_rate)


    for i in range(epoch):
        for img, label in train_dataset:
            batch_len = len(img)
            img = img.view(batch_len, -1)
            img = Variable(img)
            label = Variable(label)
            # forward
            out = model(img)
            loss = criterion(out, label)
            # backward
            opitimizer.zero_grad()
            loss.backward()
            opitimizer.step()
    torch.save(model.state_dict(),"model_param/h1_300_h2_100_e_80.pkl")

if __name__ == "__main__":
    train()



# unclass_test()
