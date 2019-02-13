# Project composition

**program composition：**

![](http://m.qpic.cn/psb?/V10dYaiX2qXpCo/klMq3rQ0b9**ZMElqI8ouN4EvWPhqP69rH2ZKpF4vq0!/b/dFIBAAAAAAAA&bo=WQS4AAAAAAADF9c!&rf=viewer_4&t=5)

## Configuration：

- WDNwordPro.sln：
>Create a new STK scenario and create a c++ project solution entry for visibility access reports. The specific main program is in the WDNWordPro folder.
- EXATAreadhelper.py：
>Various functions and classes, generator config for EXATA scenario files
- EXATAconfighandle.py：
Generate various configuration files for EXATA
- EXATAPDgenerator.py：
Cyclic configurate EXATA and iterate simulation
- EXATALHSgenerator.py:
LHS sampling in parameter dimension to obtain prior data

## Method：
- WDNexataReader.py：
The database preprocessing module. Read data dedicated module
- WDNfeedback.py：
Feedback mechanism module, which returns the query point to the simulation system, generates new data, updates data sets, the simulation steps and data preprocessing classes needed for feedback. In this program, the module can also generate prior data, etc.
- WDNoptimizer.py：
Probabilistic Modeling Optimizing Function Definition Module, Majority-Valued Gauss Mixture Model Optimizing Class (the processing of single value has been completed, and the combination of two index models has been realized at present), Bayesian Optimizing Class (the model is a Gauss process), Network Evaluation Class (generating value, generating normalized data), Memory Unit Class (State Data Frame of state information storage, value or qos) The latest version of the model combines the Gauss mixture model with the Gauss process, builds clusters to fit the Gauss process, and obtains the parameters of the network.
- WDNcompare.py:
The comparison module between models is implemented in this module
- WDNTagDataHandler.py：
Operating function classes that read and cluster the generated database 

## Experimnet：
- ETC.in the OLD_


## Result：
- ETC.in the OLD_





## TIPS:

- WDNwordPro项目程序运行结束会生成nodetxt，outAccess文件夹，存放仿真需要的可见性csv报表和卫星轨道txt文件
- nodetxt\outAccess文件夹内存有已压缩的全部报表文件和轨道文件




