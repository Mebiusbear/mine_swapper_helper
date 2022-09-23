import os
from PIL import Image
import src.setting_file as setting_file
from torch.utils.data import Dataset,DataLoader
import numpy as np

# 数据准备部分
loader_func = setting_file.loader_func
label_name = setting_file.label_name
label_dict = setting_file.label_dict
len_label_name = setting_file.len_label_name

# 训练部分
batch_size = setting_file.batch_size
epoch = setting_file.epoch
learning_rate = setting_file.learning_rate
model_name = setting_file.model_name


# 训练部分数据导入
class Mydata_train(Dataset):
    def __init__(self,root_dir,label_dir): # 输入根目录与子目录
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir) # 合并路径 
        self.img_path = os.listdir(self.path) # 获取文件夹下所有图片
        self.imgs = [os.path.join(self.root_dir,self.label_dir,img) for img in self.img_path]

    def __getitem__(self,idx):
        img = Image.open(self.imgs[idx]) # 要用PIL打开图片
        img = loader_func("train",img)
        label = self.label_dir
        label = label_dict[label]
        return img,label

    def __len__(self):
        return len(self.imgs)

def train_dataset(dataset_root_dir):
    train_dataset = Mydata_train(dataset_root_dir,label_name[0])
    for ln in label_name[1:]:
        train_dataset += Mydata_train(dataset_root_dir,ln)

    train_dataset = DataLoader(dataset=train_dataset, batch_size=batch_size,num_workers=0,shuffle=True)

    return list(train_dataset)

# 识别部分数据导入
class Mydata_discri(Dataset):
    def __init__(self,img_list):
        self.img_list = img_list

    def __getitem__(self,idx):
        img = Image.fromarray(np.uint8(self.img_list[idx]*255))
        img = loader_func("discri",img)
        return img

    def __len__(self):
        return len(self.img_list)

def discri_dataset(img_list,discri_batch_size):
    discri_dataset = Mydata_discri(img_list)
    discri_dataset = DataLoader(dataset=discri_dataset, batch_size=discri_batch_size,num_workers=0,shuffle=False)

    return discri_dataset