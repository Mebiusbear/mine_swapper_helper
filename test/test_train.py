from torch.utils.data import Dataset,DataLoader  # 导入Dataset后可以使用“help(Dataset)查看官方文档”
from PIL import Image                 # 借助PIL库导入数据图片
import os                             # 借助os库来用路径读入数据
import torch
from torch import optim, nn
from torch.autograd import Variable
import logging
import sys
sys.path.append('./')
import setting_file
import __init__


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
        img = Image.open(self.imgs[idx])                                 # 使用PIL库下Image工具，打开对应路径图片
        img = loader(img)
        label = self.label_dir                                              # 本数据集label就是文件名，如“ants”（虽然命名为dir看似路径，实则视作字符串会更容易理解）
        label = label_dict[label]
        return img,label     # 返回对应图片和图片的label

    def __len__(self):
        return len(self.imgs)

root_dir = "./pic/dataset"
train_dataset = Mydata(root_dir,label_name[0]) + Mydata(root_dir,label_name[3])


train_dataset = DataLoader(dataset=train_dataset, batch_size=batch_size,num_workers=0,shuffle=True)
train_dataset = list(train_dataset)


def train():
    logging.info("h1_%d_h2_%d_h3_%d_e_%d"%(n_hiddle_1,n_hiddle_2,n_hiddle_3,epoch))
    criterion = nn.CrossEntropyLoss()
    model = Batch_Net(in_dim,n_hiddle_1,n_hiddle_2,n_hiddle_3,out_dim=4)
    opitimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for i in range(epoch):
        loss_total = 0
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
            loss_total += loss / batch_len
        log_mes = "\nepoch %d : "%i + str(float(loss_total))
        logging.info(log_mes)

    torch.save(model.state_dict(),model_name)

if __name__ == "__main__":
    # train()
    model_name_split = model_name.split("/")
    model_name_split[1] = "t14_" + model_name_split[1]
    model_name = "/".join(model_name_split)

    train()