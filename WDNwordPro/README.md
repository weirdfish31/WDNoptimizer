# 项目文件结构

## 仿真配置部分程序：

* WDNwordPro.sln：新建STK场景，生成可见性报表的c++项目程序原工程文件

* EXATAreadhelper.py：生成EXATA场景文件的config所需要的各种函数、类

- EXATAconfighandle.py：生成config文件的程序


- EXATAfirstdemo.py：循环配置参数EXATA启动仿真输出数据的程序



## 数据处理部分程序：
- WDNexataReader.py：数据库读取预处理模块。读取数据专用类，流聚合与业务聚合函数有一点小的问题，有些数据库处理的时候会报错

- WDNfeedback.py：反馈机制模块，将得到的query point 返回仿真中，生成新数据,更新数据集，反馈所需的仿真步骤和数据预处理类在这里

   WDNoptimizer.py：概率建模优化函数定义模块，	多数值高斯混合模型优化类（对单数值的处理已经完成，目前实现了2个指标的模型的结合）、贝叶斯优化类（模型为高斯过程）、网络评估类（生成value，生成归一化数据）、记忆单元类（状态信息存储，value或qos的state数据Dataframe）、评估值高斯混合模型优化类

- Drawer_RawData_Statistic.py:对原始数据的数据库中的业务层输出指标进行时间序列曲线绘制

- Drawer_RawData_Kde.py：对原始数据中的某两列绘制核密度估计图

- Drawer_Throughput_GMM.py:2簇的两种业务的吞吐量联合高斯混合模型的迭代过程热力图

- Drawer_Jitter_GMM.py:4簇的时延抖动与丢包里的联合高斯混合模型的迭代过程热力图



## 实验部分程序：

- GPR_throughput_firstorder：多目标高斯模型迭代实验
- GMM_throughput_firstorder：多目标高斯混合模型迭代试验（混合模型有些错误）
- GMM_throughput_secondorder：多目标高斯混合模型UCB采集函数学习率实验（混合模型有问题）

上面三个实验没有太大参考价值，作为测试



- GMM_throughput_thirdorder：单目标高斯混合模型+高斯过程迭代试验（模型修正）
- New_Predi_Compare：将迭代之后的querypoint集合的预测值与实际仿真之进行比较（还未做完）