import torch.nn as nn
from PIL import Image
import torch
from torchvision import transforms
import os
import shutil

loader = transforms.Compose([transforms.ToTensor()])
label_name = ["num_1","num_2","num_3","num_4","num_5","lei","kong","zha","qizi"]
len_label_name = len(label_name)

def get_pre_name(pre):
    pre = pre.detach().numpy().tolist()[0]
    max1 = max(pre)
    pre_index = pre.index(max1)
    pre_name = label_name[pre_index]
    return pre_name

def test(model,pic_abspath):
    img = Image.open(pic_abspath).convert("L")
    img = loader(img).unsqueeze(0).reshape(1,1,30,30)
    img = img.view(1, -1)
    model.eval()
    pre = model(img)
    return get_pre_name(pre)


def unclass_test(model):
    unclass_dir = "pic/unclass"
    testclass_dir = "pic/test_class"
    unclass_list = os.listdir(unclass_dir)
    for name in unclass_list:
        pre_name = test(model,os.path.join(unclass_dir,name))
        out_dir = os.path.join(testclass_dir,pre_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        shutil.move(os.path.join(unclass_dir,name),os.path.join(out_dir,name))

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

def discri():
    criterion = nn.BCEWithLogitsLoss()
    model = Batch_Net(in_dim=30*30,n_hiddle_1=300,n_hiddle_2=100,out_dim=len_label_name)
    model.load_state_dict(torch.load("/Users/bear/Documents/GitHub/saolei/model_param/h1_300_h2_100_e_80.pkl"))
    unclass_test(model)

if __name__ == "__main__":
    discri()