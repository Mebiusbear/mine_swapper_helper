import matplotlib.pyplot as plt


filename = "./log_file/log200.txt"
with open (filename,"r") as f:
    rls = f.readlines()

res = list()
for rl in rls:
    if "epoch" in rl:
        res.append (float(rl.rstrip().split(":")[-1]))
        
x = range(len(res))

plt.plot(x,res)
plt.show()