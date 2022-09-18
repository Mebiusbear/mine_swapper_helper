import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def cut(filename):
    im = plt.imread(filename)
    im = im[143:623,15:915,:]

    SIZE = 30

    all_1 = im.reshape(16,30,SIZE,SIZE,4)

    res = list()

    for i in range (16):
        for j in range (30):
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
    res = cut("./pic/expand_dataset/used_pic/Wechat_2.png")
    # save(res)
    class_unclass(res)
