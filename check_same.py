import os
import numpy as np
import matplotlib.pyplot as plt


def check_same(file_dir):
    file_list = os.listdir(file_dir)

    res = list()
    for file_name in file_list:
        file_ralname = os.path.join(file_dir,file_name)
        im = plt.imread(file_ralname)
        for k in res:
            if np.allclose(k,im):
                break
        else:
            res.append(im)
    print (len(res))

label_name = ["num_1","num_2","num_3","num_4","num_5","lei","kong","zha","qizi"]

for ln in label_name:
    print (ln,end = " : ")
    check_same(os.path.join("pic",ln))