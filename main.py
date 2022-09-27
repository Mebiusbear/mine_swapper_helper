from src import (train,discriminate,mine_helper,click_safe)
import time


# train.train()

# start = time.time()
# filename = "/Users/bear/Documents/GitHub/saolei/pic/expand_dataset/used_pic/Wechat_6.png"
# for i in range (10):
#     discriminate.get_matrix("difficult",filename)
# print ("used time : ", time.time()-start)

# -------------------------------
# train.train()

# -------------------------------
# mine_helper.main()

# -------------------------------
filename = "/Users/bear/Documents/GitHub/saolei/pic/expand_dataset/used_pic/12.png"
discriminate.expand_dataset("difficult",filename)

# -------------------------------
# click_safe.jiaozhun()