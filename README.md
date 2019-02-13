# WDNoptimizer


![](http://m.qpic.cn/psb?/V10dYaiX2qXpCo/kHxKVg2sa1DKs6NtDU.qe6mxeo*Ae34F.J*gz4YQrGs!/b/dLgAAAAAAAAA&bo=WAWwAQAAAAARF8w!&rf=viewer_4&t=5)

**Wide Dynamic Network optimizer** ：

>This project is mainly used to quickly configure and simulate the next generation space information network system scenarios in EXATA. According to the statistical analysis of simulation data, the hybrid probability process model between multi-design parameters and performance indicators is established, and the network design is optimizing by bayesian method's iteration.
  
**Background：**:

>As a strategic support for global multi-service demand in the future, the next generation of space information network (SIN) has many design difficulties and uncertainties. Combined with the construction requirements and the need for representation of design elements with the expression of SIN system QoS performance, sometimes we could simulate the exact scenario and obtain the evaluation but we don’t know how to change the network’s design parameters to optimize it. so, based on the Posterior probability method, Aiming at the space-time characteristics of Space information network, using the specific network simulation platform, we proposed A novel method which represent the distribution layer-design elements’ compound uncertainty within based on hybrid probability process (HPP) model and the few shots iteration strategy of SIN. Bayesian method is used to fit the posterior surrogate model with high dimension design elements of the system and the QoS evaluation index. Compared with the origin surrogate model and the general Bayesian optimization method, better fitting results are obtained, which will guarantee the quantization of next generation space information network system’s design factors.




TIPS： STK 9.2.3，EXATA 5.4 needed

1）STK9.2.3：It provides basic orbit parameters and visibility parameter reports for constellation satellite simulation. 

2）EXATA5.4：Core unit of simulation optimization

3）LHSMDU：To provide the basic function of Latin hypercube sampling for pre-sampling module, it is necessary to install the module locally.gituhub：<https://github.com/sahilm89/lhsmdu>







**Composition description：**

1）WDNwordPro： Main folder

- configfile： EXATA configuration json files
- Debug： STK's access report and satellites trace generator program
- nodetxt： 存放给EXATA生成场景配置文件的节点坐标txt文件，实例地面用户节点与遥感卫星节点（WDNwordPro程序生成）
- outAccess： 存放给EXATA生成link相关配置文件的STK节点之间的csv可见性报表，只有一部分示意（WDNwordPro程序生成）
- OutConfigfile：存放EXATA中的场景配置文件，包括网络场景的仿真配置基本参数文件，链路拓扑文件，网络各层的基本参数配置文件，业务流配置文件，此文件夹中的文件直接提供给EXATA仿真核心模块进行网络场景的仿真，所有的参数优化都是基于这种可配置的config文件进行操作
- WDNwordPro：在STK中生成对应卫星场景的主程序，提供相应的网络仿真拓扑
- Figure：存放输出的图片
- OutStorefile：存放每一次仿真得到的参数配置信息文件备份
- history：存放各个数据集的设计参数组合的list文件，先验数据的集合（未补完）
- OLD_：存放旧的实验或绘图程序
- LabelData：存放评估聚类完成数据特征值，以便迭代时不需要重新进行聚类
- QP：history Query Point listfile


2）TestData：Simulation data folder
- LHSprior：prior sampling data
- LHSMSE：testdata
- ITER_Data：iteration data








Contact：
- E-Mail：weirdfish31@whu.edu.cn
- QQ:179177098







