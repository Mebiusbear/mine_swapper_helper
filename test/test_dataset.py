import torch
import sys
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os



sys.path.append('./')
import setting_file
# 数据准备部分
loader = setting_file.loader_func
label_name = setting_file.label_name
label_dict = setting_file.label_dict
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
        img = loader(img)
        label = self.label_dir                                              # 本数据集label就是文件名，如“ants”（虽然命名为dir看似路径，实则视作字符串会更容易理解）
        label = label_dict[label]
        return img,label     # 返回对应图片和图片的label

    def __len__(self):
        return len(self.imgs)


root_dir = "./pic/dataset"
train_dataset = Mydata(root_dir,label_name[0])
for ln in label_name[1:]:
    train_dataset += Mydata(root_dir,ln)

train_dataset = DataLoader(dataset=train_dataset, batch_size=batch_size,num_workers=0,shuffle=False)
train_dataset = list(train_dataset)


for data in train_dataset:
    img,label = data
    print (label)