## zero part
```
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
+ 开始写入数独内容
+ 优化log_file
+ 写dataset教程
+ 改为并行discri
+ 写test_dataset文件
+ 完善test_train,test_discri
+ 换一个CNN
