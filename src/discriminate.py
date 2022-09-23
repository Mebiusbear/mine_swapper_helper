import os
import matplotlib.pyplot as plt
import torch
import time

import src.cut as cut
import src.setting_file as setting_file
import src.dataset_file as dataset_file
import src.net_file as net_file

# 数据准备部分
label_name = setting_file.label_name
label_name_dict = setting_file.label_name_dict
len_label_name = setting_file.len_label_name

# 模型部分
Batch_Net = net_file.Batch_Net
in_dim = setting_file.in_dim
n_hiddle_1 = setting_file.n_hiddle_1
n_hiddle_2 = setting_file.n_hiddle_2
n_hiddle_3 = setting_file.n_hiddle_3

# 识别部分
model_name = setting_file.model_name
EASY_POINTS = setting_file.EASY_POINTS
DIFFICULT_POINTS = setting_file.DIFFICULT_POINTS


def initial(): # 初始化，返回调节器和模型
    model = Batch_Net(in_dim,n_hiddle_1,n_hiddle_2,n_hiddle_3,out_dim=len_label_name)
    model.load_state_dict(torch.load(model_name))
    return model.eval()

def get_pre_name(pre): # 输入预测tensor，返回预测名字列表
    # pre_index = torch.max(pre.data,1)[1][idx] # 求第idx行最大值
    pre_index_list = torch.max(pre.data,1)[1]
    pre_name_list = [label_name[pre_index] for pre_index in pre_index_list]
    return pre_name_list

def get_all_pixel_discri_kernel(level): # 图片分割的核心程序，可选择为模式：简单、一般、复杂

    if level == "difficult":
        ROW,COL = DIFFICULT_POINTS
    elif level == "easy":
        ROW,COL = EASY_POINTS
        
    def func(model_eval,filename):
        res = cut.cut_func(level,filename)
        res_row_col = [res[i*ROW:(i+1)*ROW] for i in range (COL)]

        dataset_discri = dataset_file.discri_dataset(res,ROW)
        out_row_col = [get_pre_name( \
                model_eval( \
                data.view(ROW, -1) )) for data in dataset_discri]
        return out_row_col,res_row_col
    return func


# 选择扫雷图片，然后扩充数据库
def expand_dataset(level,filename):
    model_eval = initial()
    func = get_all_pixel_discri_kernel(level)
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

def get_matrix(level,filename):
    import numpy as np
    matrix = np.zeros((30,16))
    model_eval = initial()
    func = get_all_pixel_discri_kernel(level)
    label_row_col,_ = func(model_eval,filename)
    
    for i,row in enumerate(label_row_col):
        matrix[i,:] = np.array(list(map(lambda x:label_name_dict[x],row)))
    return matrix.T


if __name__ == "__main__":
    filename = "./mine_swapper/temp/0.png"
    # expand_dataset("difficult",filename)

    start = time.time()
    for _ in range (10):
        get_matrix("difficult",filename)
    print ("Use time : ", time.time()-start)