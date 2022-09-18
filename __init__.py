import os
import platform

print ("Run on : ", platform.system())
if platform.system() == "Darwin":
    os.system("find . -name '.DS_Store' -type f -delete")

if platform.system() == "Windows":
    for root, dirs, files in os.walk("./"):
        if ".DS_Store" in files :
            DS_abspath = os.path.join(root,".DS_Store")
            os.remove(DS_abspath)
    # os.system("dir /s /a .DS_Store > temp_1.txt")
    # with open ("temp_1.txt","r") as f:
    #     a = f.readlines()
    #     for i in a:
    #         if "Github\Mine_swapper_helper" in i:
    #             DS_abspath = os.path.join(i.rstrip().split(" ")[1],".DS_Store")
    #             print ("rm %s"%DS_abspath)
    #             os.remove(DS_abspath)