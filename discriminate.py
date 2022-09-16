import torch.nn as nn
from PIL import Image
import torch
from torchvision import transforms
import os
import shutil
import setting_file
import cut
import matplotlib.pyplot as plt

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
    pre = pre.detach().numpy().tolist()[0]
    max1 = max(pre)
    pre_index = pre.index(max1)
    pre_name = label_name[pre_index]
    return pre_name

def test(model,pic_abspath):
    img = Image.open(pic_abspath).convert("L")
    img = loader(img)
    img = img.view(1, -1)
    model.eval()
    pre = model(img)
    return get_pre_name(pre)


def unclass_test(model):
    unclass_dir = "pic/expand_dataset/unclass"
    testclass_dir = "pic/expand_dataset/test_class"
    unclass_list = os.listdir(unclass_dir)
    if os.listdir(testclass_dir):
        print ("rm -r %s/*"%testclass_dir)
        os.system("rm -r %s/*"%testclass_dir)
    for name in unclass_list:
        pre_name = test(model,os.path.join(unclass_dir,name))
        out_dir = os.path.join(testclass_dir,pre_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        shutil.move(os.path.join(unclass_dir,name),os.path.join(out_dir,name))



def initial():
    criterion = nn.BCEWithLogitsLoss()
    model = Batch_Net(in_dim,n_hiddle_1,n_hiddle_2,n_hiddle_3,out_dim=len_label_name)
    model.load_state_dict(torch.load(model_name))
    return criterion, model

def discri(func):
    criterion, model = initial()
    func(model)

def get_all_pixel_discri(file_dir):
    criterion, model = initial()
    out = list()

    for i in range (480):        
        pic_abspath = os.path.join(file_dir,"%d.png"%i)
        img = Image.open(pic_abspath).convert("L")
        img = loader(img)
        img = img.view(1, -1)
        model.eval()
        pre = model(img)
        print ("%d.png : "%i,get_pre_name(pre) ," : ",label_name_dict[get_pre_name(pre)])

    #     out.append(label_name_dict[get_pre_name(pre)])
    # out_16_30 = list()
    # for i in range (16):
    #     out_16_30.append(out[i*30:(i+1)*30])
    # for i in out_16_30:
    #     print (i)
        


if __name__ == "__main__":
    discri(unclass_test)
    pass
    # get_all_pixel_discri("temp")