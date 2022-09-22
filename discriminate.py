import os
import shutil
import setting_file
import matplotlib.pyplot as plt
import time

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
get_all_pixel_discri_kernel = setting_file.get_all_pixel_discri_kernel

# 选择扫雷图片，然后扩充数据库
def expand_dataset(level,filename):
    model_eval = initial()[1].eval()
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
    model_eval = initial()[1].eval()
    func = get_all_pixel_discri_kernel(level)
    label_row_col,_ = func(model_eval,filename)
    
    for i,row in enumerate(label_row_col):
        row_label = np.array(list(map(lambda x:label_name_dict[x],row)))
        matrix[i,:] = row_label
    return matrix.T


if __name__ == "__main__":
    filename = "./mine_swapper/temp/WX20220922-165434.png"
    # filename = "./pic/expand_dataset/used_pic/Wechat_9.png"
    start = time.time()
    expand_dataset("difficult",filename)
    
    print ("Use time : ", time.time()-start)