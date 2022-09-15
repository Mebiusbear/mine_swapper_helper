import matplotlib.pyplot as plt
import numpy as np
import PIL

im = plt.imread("test_4.png")
im = im[143:623,15:915,:]

SIZE = 30

all_1 = im.reshape(16,30,SIZE,SIZE,4)
little_im = list()
res = list()
res.append(all_1[0,:,0,:,:])

for i in range (16):
    for j in range (30):
        for k in res:
            if np.allclose(k,all_1[i,:,j,:,:]):
                break
        else:
            res.append(all_1[i,:,j,:,:])

for i,little_im in enumerate(res):
    plt.imsave("pic/unclass/%d.png"%i,little_im)