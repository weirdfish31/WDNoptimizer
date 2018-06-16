2018/5/26
在weirdfishes.py中的高斯混合模型类中加入了聚类和nan数据处理函数

2018/5/24
生成数据部分程序：
=======================================
WDNwordPro.sln：新建STK场景，生成可见性报表的c++程序
readhelper.py：生成EXATA场景文件的config所需要的各种函数、类
confighandle.py：生成config文件的程序
firstdemo.py：循环配置参数EXATA启动仿真输出数据的程序


数据处理部分程序：
=======================================
radiohead.py：EXATAReader类，读取数据专用类

weirdfishes.py：高斯混合模型类（对单数值的处理已经完成，正在对多数值的类进行处理）、
贝叶斯优化类（基本完成）、网络评估类（生成value）、强化学习类（状态信息存储，value或qos指标）
完成了固定两个指标的数据合成，动态合成未完成

truelovewaits.py:数据处理程序（测试程序）

reckoner.py：高斯混合模型处理程序（分布图核密度估计图）

feedbackprocess.py：将得到的query point 返回仿真中，生成新数据