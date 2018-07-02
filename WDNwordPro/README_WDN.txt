生成数据部分程序：
=======================================
WDNwordPro.sln：新建STK场景，生成可见性报表的c++项目程序原工程
readhelper.py：生成EXATA场景文件的config所需要的各种函数、类
confighandle.py：生成config文件的程序
firstdemo.py：循环配置参数EXATA启动仿真输出数据的程序


数据处理部分程序：
=======================================
radiohead.py：EXATAReader类，读取数据专用类（流聚合与业务聚合函数有一点小的问题，有些数据库处理的时候会报错）
feedbackprocess.py：将得到的query point 返回仿真中，生成新数据,更新数据集，反馈所需的仿真步骤和数据预处理类在这里
weirdfishes.py：（建模和优化函数的定义）
高斯混合模型类（对单数值的处理已经完成，正在对多数值的类进行处理，目前实现了2个指标的模型的结合）、
贝叶斯优化类（基本完成，对GMM模型起知道建议，现在未使用）、
网络评估类（生成value，生成归一化数据）、
强化学习类（状态信息存储，value或qos的state数据Dataframe）、

rawreader.py:对原始数据的数据库中的applicationlayer标签页下的各种指标进行时间序列曲线绘制
truelovewaits.py:数据处理程序（测试程序):目前的反馈过程都是在这里进行的
modifydrawer.py:对数据进行二次绘图，（反馈程序中的一些图有时候会有bug，需要进行二次绘制）
reckoner.py：高斯混合模型处理程序（分布图核密度估计图）
