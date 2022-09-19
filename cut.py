import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import setting_file

EASY_ROW, EASY_COL = setting_file.EASY_POINTS
DIFFICULT_ROW, DIFFICULT_COL = setting_file.DIFFICULT_POINTS
SIZE = 30

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

def save(im_list):
    for i,im in enumerate(im_list):
        plt.imsave("temp/%d.png"%i,im)

def class_unclass(res):

    for i,little_im in enumerate(res):
        plt.imsave("pic/expand_dataset/unclass/%d.png"%i,little_im)

if __name__ == "__main__":
    res = cut("./pic/expand_dataset/used_pic/Wechat_6.png")
    # save(res)
    class_unclass(res)
