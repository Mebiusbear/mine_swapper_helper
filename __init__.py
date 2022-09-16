import os
import platform

if platform.system() == "Darwin":
    os.system("find . -name '.DS_Store' -type f -delete")
