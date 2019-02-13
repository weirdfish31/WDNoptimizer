# Project composition

**program composition：**

![](http://m.qpic.cn/psb?/V10dYaiX2qXpCo/klMq3rQ0b9**ZMElqI8ouN4EvWPhqP69rH2ZKpF4vq0!/b/dFIBAAAAAAAA&bo=WQS4AAAAAAADF9c!&rf=viewer_4&t=5)

## Configuration：

- WDNwordPro.sln：Create a new STK scenario and create a c++ project solution entry for visibility access reports. The specific main program is in the WDNWordPro folder.
- EXATAreadhelper.py：Various functions and classes, generator config for EXATA scenario files
- EXATAconfighandle.py：Generate various configuration files for EXATA
- EXATAPDgenerator.py：Cyclic configurate EXATA and iterate simulation
- EXATALHSgenerator.py:LHS sampling in parameter dimension to obtain prior data

## Method：
- WDNexataReader.py：The database preprocessing module. Read data dedicated module
- WDNfeedback.py：Feedback mechanism module, which returns the query point to the simulation system, generates new data, updates data sets, the simulation steps and data preprocessing classes needed for feedback. In this program, the module can also generate prior data, etc.
- WDNoptimizer.py：概率建模优化函数定义模块，	多数值高斯混合模型优化类（对单数值的处理已经完成，目前实现了2个指标的模型的结合）、贝叶斯优化类（模型为高斯过程）、网络评估类（生成value，生成归一化数据）、记忆单元类（状态信息存储，value或qos的state数据Dataframe）、评估值高斯混合模型优化类，正在不断优化，最新版的模型是在高斯混合模型与高斯过程结合，建分簇进行高斯过程的拟合，得到网络的参数过程
- WDNcompare.py:The comparison module between models is implemented in this module
- WDNTagDataHandler.py：Operating function classes that read and cluster the generated database 

## Experimnet：
- ETC.in the OLD_


## Result：
- ETC.in the OLD_





## TIPS:

- WDNwordPro项目程序运行结束会生成nodetxt，outAccess文件夹，存放仿真需要的可见性csv报表和卫星轨道txt文件
- nodetxt\outAccess文件夹内存有已压缩的全部报表文件和轨道文件




