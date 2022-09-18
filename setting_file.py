import torch.nn as nn
from torchvision import transforms
import torch
import logging

def make_log():
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关  此时是INFO
    # 第二步，创建一个handler，用于写入日志文件
    logfile = 'log_file/log%d.txt'%epoch
    fh = logging.FileHandler(logfile, mode='w')  # open的打开模式这里可以进行参考
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)   # 输出到console的log等级的开关
    # 第四步，定义handler的输出格式（时间，文件，行数，错误级别，错误提示）
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

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
def loader_func_1(img):
    img = img.convert("L")
    return loader(img).unsqueeze(0).reshape(1,INIT_SIZE,INIT_SIZE)
def loader_func_2(img):
    img = img.convert("RGB")
    return loader(img).unsqueeze(0).reshape(3,INIT_SIZE,INIT_SIZE)


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
loader_func = loader_func_2


if loader_func == loader_func_1:
    in_dim = INIT_SIZE*INIT_SIZE
elif loader_func == loader_func_2:
    in_dim = 3*INIT_SIZE*INIT_SIZE
    
    
n_hiddle_1=1600
n_hiddle_2=800
n_hiddle_3=200

epoch = 20
learning_rate = 1e-3
batch_size = 4
model_name = "model_param/h1_%d_h2_%d_h3_%d_e_%d.pkl"%(n_hiddle_1,n_hiddle_2,n_hiddle_3,epoch)