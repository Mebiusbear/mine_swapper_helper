import os
import platform

if platform.system() == "darwin":
    os.system("find . -name '.DS_Store' -type f -delete")