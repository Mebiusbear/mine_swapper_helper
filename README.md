## zero part
```
pip install -r requierment.txt
unzip pic.zip
mkdir log_file
mkdir model_param
```

## first part
+ 制作数据集
```python
python cut.py
```
+ 标准化cut

## second part
+ trainset 扩充 100 echo filedir

## third part
+ 数独
+ 创建简单数独数学模型
+ 破解数独
+ pyqt制作游戏

# TODO
+ ~~多通道线性层 (finish)~~
+ ~~1与4的区分测试~~
+ ~~loss作图,matplotlib~~
+ ~~准确预测每一个数字~~
+ ~~创建简单分割和困难分割~~
+ ~~写cut-discrame-workflow~~
+ ~~改造easy，difficult，normal function~~
+ ~~开始写入数独内容~~
+ ~~自动点击安全点~~
+ 改写dfs
+ 加入分割算法
+ 自动获得启动点
+ 优化扫雷程序
+ 生成扫雷gif
+ 如果没有，点击最小概率点
+ 优化log_file
+ cut_func返回矩阵状态的数组
+ 解决func : get_all_pixel_discri_kernel图像需要倒置问题
+ 写dataset教程
+ 改为并行discri
+ 写test_dataset文件
+ 完善test_train,test_discri
+ 换一个CNN
+ workflow 化程序
+ 对齐初始点
+ 对象化，初始设定难度

# 流程

+ 设定参数 epoch 60-80、h1 800、h2 200、h3 50、batch 16、learning rate 1e-3 
+ python train.py 遇到pridict=1停下即可
+ python discriminate.py 查看pic/expand_dataset/test_class 分类是否正确