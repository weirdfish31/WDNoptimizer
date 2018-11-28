# 项目文件结构

**项目程序结构：**

![](http://m.qpic.cn/psb?/V10dYaiX2qXpCo/6hh83aSwj8hQuWW0yyGcuz.oRpxLyOCNf0imGzKxZ2g!/b/dDQBAAAAAAAA&bo=3AThAAAAAAADFws!&rf=viewer_4&t=5)

## 仿真配置部分程序：

* WDNwordPro.sln：新建STK场景，生成可见性报表的c++项目解决方案入口，具体的主程序在WDNwordPro文件夹中

* EXATAreadhelper.py：生成EXATA场景文件的config所需要的各种函数、类

- EXATAconfighandle.py：生成EXAT所需的各种config文件


- EXATAPDgenerator.py：循环配置参数EXATA启动仿真输出数据
- EXATALHSgenerator.py:对参数空间进行LHS采样，得到先验数据

## 数据处理部分程序：
- WDNexataReader.py：数据库读取预处理模块。读取数据专用模块，所有的数据处理函数都在其中
- WDNfeedback.py：反馈机制模块，将得到的query point 返回至仿真系统中，生成新数据,更新数据集，反馈所需的仿真步骤和数据预处理类在本程序中，本模块也可以进行先验数据的生成等
	 WDNoptimizer.py：概率建模优化函数定义模块，	多数值高斯混合模型优化类（对单数值的处理已经完成，目前实现了2个指标的模型的结合）、贝叶斯优化类（模型为高斯过程）、网络评估类（生成value，生成归一化数据）、记忆单元类（状态信息存储，value或qos的state数据Dataframe）、评估值高斯混合模型优化类，正在不断优化，最新版的模型是在高斯混合模型与高斯过程结合，建分簇进行高斯过程的拟合，得到网络的参数过程
- WDNcompare.py:未完成，将模型之间的比较模块在此模块中实现

## 实验部分程序：

- GPR_throughput_firstorder：多目标高斯模型迭代实验
- GMM_throughput_firstorder：多目标高斯混合模型迭代试验（混合模型有些错误）
- GMM_throughput_secondorder：多目标高斯混合模型UCB采集函数学习率实验（混合模型有问题）

上面三个实验没有太大参考价值，作为测试

- GMM_throughput_thirdorder：单目标高斯混合模型+高斯过程迭代试验（模型修正）目前效果不太理想
- New_Predi_Compare：将迭代之后的querypoint集合的预测值与实际仿真之进行比较（还未做完）

## 绘图程序：

- Drawer_Jitter_GMM.py：GMM混合模型绘图

- Drawer_RawData_Kde.py：KDE联合分布图（二维）

- Drawer_RawData_Statistic.py：原始数据统计曲线图

- Drawer_Throughput_GMM.py：吞吐量的GMM混合模型图，基于最开始的混合模型

- AFqueryPoint.py：querypoint的迭代序列二维绘图

  