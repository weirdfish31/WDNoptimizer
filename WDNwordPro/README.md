# Project composition

**program composition：**

![](http://m.qpic.cn/psb?/V10dYaiX2qXpCo/klMq3rQ0b9**ZMElqI8ouN4EvWPhqP69rH2ZKpF4vq0!/b/dFIBAAAAAAAA&bo=WQS4AAAAAAADF9c!&rf=viewer_4&t=5)

## Configuration：

- WDNwordPro.sln：Create a new STK scenario and create a c++ project solution entry for visibility access reports. The specific main program is in the WDNWordPro folder.
- EXATAreadhelper.py：Various functions and classes, generator config for EXATA scenario files
- EXATAconfighandle.py：生成EXAT所需的各种config文件
- EXATAPDgenerator.py：循环配置参数EXATA启动仿真输出数据
- EXATALHSgenerator.py:对参数空间进行LHS采样，得到先验数据

## Method：
- WDNexataReader.py：数据库读取预处理模块。读取数据专用模块，所有的数据处理函数都在其中
- WDNfeedback.py：反馈机制模块，将得到的query point 返回至仿真系统中，生成新数据,更新数据集，反馈所需的仿真步骤和数据预处理类在本程序中，本模块也可以进行先验数据的生成等
- WDNoptimizer.py：概率建模优化函数定义模块，	多数值高斯混合模型优化类（对单数值的处理已经完成，目前实现了2个指标的模型的结合）、贝叶斯优化类（模型为高斯过程）、网络评估类（生成value，生成归一化数据）、记忆单元类（状态信息存储，value或qos的state数据Dataframe）、评估值高斯混合模型优化类，正在不断优化，最新版的模型是在高斯混合模型与高斯过程结合，建分簇进行高斯过程的拟合，得到网络的参数过程
- WDNcompare.py:未完成，将模型之间的比较模块在此模块中实现
- WDNTagDataHandler.py：对已生成的数据库进行读取并进行分簇标记等一些列的操作函数类

## Experimnet：
- ETC.in the OLD_


## Result：
- ETC.in the OLD_





## TIPS:

- WDNwordPro项目程序运行结束会生成nodetxt，outAccess文件夹，存放仿真需要的可见性csv报表和卫星轨道txt文件
- nodetxt\outAccess文件夹内存有已压缩的全部报表文件和轨道文件




