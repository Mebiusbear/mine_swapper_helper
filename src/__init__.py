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
