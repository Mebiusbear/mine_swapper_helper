from fileinput import filename
import torch.nn as nn
from torchvision import transforms
import torch
import logging


BOX_SIZE = 30
INIT_SIZE = 28
EASY_POINTS = (9,9)
DIFFICULT_POINTS = (16,30)
DATASET_ROOT_DIR = "./pic/dataset"

BASE_POINT = (307.91595458984375, 408.39251708984375)


# 日志部分
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


# 数据准备部分
def label_index(index,len_label_name):
    z = torch.zeros(len_label_name)
    return index
label_name_dict = {
    "num_1" :  1,
    "num_2" :  2,
    "num_3" :  3,
    "num_4" :  4,
    "num_5" :  5,
    "kong"  :  0,
    "qizi"  : -1,
    "safe"  : -2,
}
label_name = list(label_name_dict.keys()) 
len_label_name = len(label_name)
label_dict = {
    label_name[i] : label_index(i,len_label_name) for i in range (len_label_name)
}
def loader_func(mode,img):
    if mode == "train":
        trans_crop = transforms.RandomCrop(INIT_SIZE)
    elif mode == "discri":
        trans_crop = transforms.CenterCrop(INIT_SIZE)
    else:
        raise "no mode"
    loader = transforms.Compose([trans_crop,transforms.Resize(INIT_SIZE),transforms.ToTensor()])
    return loader(img.convert("RGB")).unsqueeze(0).reshape(3,INIT_SIZE,INIT_SIZE)


# 训练参数
in_dim = 3 * INIT_SIZE * INIT_SIZE
n_hiddle_1 = int (in_dim / 2)
n_hiddle_2 = int (in_dim / 8)
n_hiddle_3 = int (in_dim / 8 / 7)

epoch = 60
learning_rate = 1e-3
batch_size = 32
model_name = "./model_param/h1_%d_h2_%d_h3_%d_e_%d.pkl"%(n_hiddle_1,n_hiddle_2,n_hiddle_3,epoch)
