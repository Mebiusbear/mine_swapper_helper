from fileinput import filename
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import setting_file

EASY_ROW, EASY_COL = setting_file.EASY_POINTS
DIFFICULT_ROW, DIFFICULT_COL = setting_file.DIFFICULT_POINTS
SIZE = setting_file.BOX_SIZE

# 适配各种难度的裁剪
def cut_func(level,filename):
    if level == "difficult":
        b_row,b_col = 143,15
        ROW,COL = DIFFICULT_ROW,DIFFICULT_COL
    elif level == "easy":
        b_row,b_col = 145,15
        ROW,COL = EASY_ROW,EASY_COL
        
    im = plt.imread(filename)
    im = im[b_row:(b_row+ROW*30),b_col:(b_col+COL*30)]
    reshape_im = im.reshape(ROW,SIZE,COL,SIZE,4)
    res = [reshape_im[i,:,j,:,:] for j in range (COL) for i in range (ROW)]
    return res

def difficult_cut(filename):
    im = plt.imread(filename)
    im = im[143:623,15:915,:]
    
    all_1 = im.reshape(DIFFICULT_ROW,SIZE,DIFFICULT_COL,SIZE,4)
    res = list()
    for i in range (DIFFICULT_ROW):
        for j in range (DIFFICULT_COL):
            res.append(all_1[i,:,j,:,:])
    return res

def easy_cut(filename):
    im = plt.imread(filename)
    im = im[145:415,15:285,:]
    all_1 = im.reshape(EASY_ROW,SIZE,EASY_COL,SIZE,4)
    res = list()
    for i in range (EASY_ROW):
        for j in range (EASY_COL):
            res.append(all_1[i,:,j,:,:])
    return res


def topil(res):
    ans = list()
    for im in res:
        pil_img = Image.fromarray(np.uint8(im*255))
        ans.append(pil_img)
    return ans

if __name__ == "__main__":
    filename = r"pic\expand_dataset\used_pic\Wechat_6.png"
    res = cut_func("difficult",filename)