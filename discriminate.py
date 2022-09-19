import torch.nn as nn
from PIL import Image
import torch
import os
import shutil
import setting_file
import cut
import matplotlib.pyplot as plt
import numpy as np

# 数据准备部分
loader_func = setting_file.loader_func
label_name = setting_file.label_name
label_name_dict = setting_file.label_name_dict
len_label_name = setting_file.len_label_name

# 模型部分
Batch_Net = setting_file.Batch_Net
in_dim = setting_file.in_dim
n_hiddle_1 = setting_file.n_hiddle_1
n_hiddle_2 = setting_file.n_hiddle_2
n_hiddle_3 = setting_file.n_hiddle_3
initial = setting_file.initial

# 识别部分
model_name = setting_file.model_name
EASY_ROW, EASY_COL = setting_file.EASY_POINTS
DIFFICULT_ROW, DIFFICULT_COL = setting_file.DIFFICULT_POINTS
get_pre_name = setting_file.get_pre_name
difficult_get_all_pixel_discri = setting_file.difficult_get_all_pixel_discri
easy_get_all_pixel_discri = setting_file.easy_get_all_pixel_discri


# 选择扫雷图片，然后扩充数据库
def expand_dataset(func,filename):
    model_eval = initial()[1].eval()
    label_row_col,img_row_col = func(model_eval,filename)

    testclass_dir = "pic/expand_dataset/test_class"
    if os.listdir(testclass_dir):
        print ("rm -r %s/*"%testclass_dir)
        os.system("rm -r %s/*"%testclass_dir)

    for i,label_row in enumerate(label_row_col):
        for j,label in enumerate(label_row):
            out_dir = os.path.join(testclass_dir,label)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            nfn = os.path.join(out_dir,"%d%d.png"%(i,j))
            plt.imsave(nfn,img_row_col[i][j])


if __name__ == "__main__":
    filename = "./pic/expand_dataset/used_pic/Wechat_6.png"
    expand_dataset(difficult_get_all_pixel_discri,filename)