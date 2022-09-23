import matplotlib.pyplot as plt
import numpy as np
import src.setting_file as setting_file


EASY_POINTS = setting_file.EASY_POINTS
DIFFICULT_POINTS = setting_file.DIFFICULT_POINTS
SIZE = setting_file.BOX_SIZE

# 适配各种难度的裁剪
def cut_func(level,filename):
    if level == "difficult":
        b_row,b_col = 143,15
        ROW,COL = DIFFICULT_POINTS
    elif level == "easy":
        b_row,b_col = 145,15
        ROW,COL = EASY_POINTS
        
    im = plt.imread(filename)
    im = im[b_row:(b_row+ROW*30),b_col:(b_col+COL*30)]
    reshape_im = im.reshape(ROW,SIZE,COL,SIZE,4)
    res = np.array([reshape_im[i,:,j,:,:] for j in range (COL) for i in range (ROW)])
    return res

