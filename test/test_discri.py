import torch.nn as nn
from PIL import Image
import torch
from torchvision import transforms
import os
import shutil
import sys
import matplotlib.pyplot as plt
sys.path.append("./")
import setting_file
import cut


# 数据准备部分
loader = setting_file.loader_func
label_name = setting_file.label_name
label_name_dict = setting_file.label_name_dict
len_label_name = setting_file.len_label_name

# 模型部分
Batch_Net = setting_file.Batch_Net
in_dim = setting_file.in_dim
n_hiddle_1 = setting_file.n_hiddle_1
n_hiddle_2 = setting_file.n_hiddle_2
n_hiddle_3 = setting_file.n_hiddle_3

# 训练部分
batch_size = setting_file.batch_size
epoch = setting_file.epoch
learning_rate = setting_file.learning_rate
model_name = setting_file.model_name


def get_pre_name(pre):
    pre_index = torch.max(pre.data,1)[1][0]
    # pre = pre.detach().numpy().tolist()[0]
    # max1 = max(pre)
    # pre_index = pre.index(max1)
    # print (pre_index)
    pre_name = label_name[pre_index]
    return pre_name

def test(model,pic_abspath):
    img = Image.open(pic_abspath)
    img = loader(img)
    img = img.view(1, -1)
    model.eval()
    pre = model(img)
    return get_pre_name(pre)

def initial():
    criterion = nn.BCEWithLogitsLoss()
    model = Batch_Net(in_dim,n_hiddle_1,n_hiddle_2,n_hiddle_3,out_dim=len_label_name)
    model.load_state_dict(torch.load(model_name))
    return criterion, model

if __name__ == "__main__":
    criterion, model = initial()
    print (test(model,"./pic/expand_dataset/unclass/6.png"))
    