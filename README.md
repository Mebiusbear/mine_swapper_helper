# How to Use
## 0.配置环境
```
pip install -r requierment.txt
```

## 1.制作数据集
```
unzip pic.zip
mkdir log_file
```

## 2.完善数据集
+ trainset 扩充 100 each filedir

## 3.使用
+ 训练
```
None
```
+ 数独
```
python main.py
```

# Node
+ 提供三个训练好的模型

# TODO
Version 1.0.0 :white_check_mark:
---
+ ~~多通道线性层 (finish)~~
+ ~~1与4的区分测试~~
+ ~~loss作图,matplotlib~~
+ ~~准确预测每一个数字~~
+ ~~创建简单分割和困难分割~~
+ ~~写cut-discrame-workflow~~
+ ~~改造easy，difficult，normal function~~
+ ~~开始写入数独内容~~
+ ~~自动点击安全点~~
+ ~~cut_func返回矩阵状态的数组~~
+ ~~改为并行discri~~
+ ~~优化扫雷程序（文件夹排序)~~


Version 1.1.0
---
+ 自动点击第一个点
+ 自动获得扫雷锚定点
+ 如果没有找到安全点，点击最小概率点
+ 不再需要temp文件夹
+ 解决func : get_all_pixel_discri_kernel图像需要倒置问题
+ 生成扫雷gif


Version 1.2.0
---
+ setting_file 直接判断三个难度
+ 写dataset教程
+ 写torch训练教程
+ 写test_dataset文件
+ 完善test_train,test_discri
+ 优化log_file

Version 1.3.0
---
+ workflow 化程序
+ 添加设置args
+ src加一个app文件夹


Version 2.0.0
---
+ 改写dfs
+ 加入分割算法
+ 换一个CNN
+ 插旗
+ 写一个初始化项目脚本
+ 对象化，初始设定难度

Version 2.1.0
---
+ 泛化各种扫雷
+ 适配windows

Version 3.0.0
---
+ 制作游戏
+ 尝试不用深度学习


# 流程

## 新数据集训练过程
+ 设定参数 epoch 60-80、h1 2352、h2 200、h3 100、batch 16、learning rate 1e-3 
+ python train.py 遇到pridict=1停下即可
+ python discriminate.py 查看pic/expand_dataset/test_class 分类是否正确
