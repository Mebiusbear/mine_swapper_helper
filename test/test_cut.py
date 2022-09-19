import sys
import matplotlib.pyplot as plt
sys.path.append("./")
import cut


filename = "./pic/expand_dataset/used_pic/Wechat_8.png"
res = cut.easy_cut(filename)
# ans = cut.topil(res)

plt.imshow(res[3])
plt.show()
