# WDNoptimizer


![](http://m.qpic.cn/psb?/V10dYaiX2qXpCo/0SPhkx5sNaSuEeacMekY29I6MWUc74IBZKpx89se1mM!/b/dFQBAAAAAAAA&bo=2APrAQAAAAADNyM!&rf=viewer_4&t=5)

**Wide Dynamic Network optimizer** ：

此项目主要是用于对EXATA中的卫星通信网络场景进行快速配置与仿真，针对仿真数据进行统计分析，建立设计参数与性能指标之间的贝叶斯概率模型，通过迭代对设计进行优化。



TIPS： 1）需要安装 STK 9.2.3，EXATA 5.4

1）STK9.2.3：提供基本的星座卫星仿真的轨道参数，可见性参数报表等，独立使用也非常便捷

2）EXATA5.4：仿真优化的核心单元

3）LHSMDU模块：为预采样的模块，提供拉丁差立方采样的基本函数，需要在本地进行安装该模块。gituhub地址：<https://github.com/sahilm89/lhsmdu>







**组成说明：**

WDNwordPro文件夹： 主程序文件夹，包括生成EXATA所需的连通性表表、各种config文件，仿真数据处理python程序等

- configfile文件夹： 主要存放生成EXATA配置文件所需的各种参数配置json文件，包括场景中移动节点文件夹、FTP参数json文件、地面站节点参数、链路参数json文件、移动节点名称配置文件等
- Debug文件夹： STK生成可见性报表，生成节点坐标轨迹文件的exe程序文件夹
- nodetxt文件夹： 存放给EXATA生成场景配置文件的节点坐标txt文件（WDNwordPro程序生成）
- outAccess文件夹： 存放给EXATA生成link相关配置文件的STK节点之间的csv可见性报表（WDNwordPro程序生成）
- OutConfigfile文件夹：存放EXATA中的场景配置文件，包括网络场景的仿真配置基本参数文件，链路拓扑文件，网络各层的基本参数配置文件，业务流配置文件，此文件夹中的文件直接提供给EXATA仿真核心模块进行网络场景的仿真，所有的参数优化都是基于这种可配置的config文件进行操作
- WDNwordPro文件夹：在STK中生成对应卫星场景的主程序，提供相应的网络仿真拓扑
- Figure文件夹：存放输出的图片
- OutStorefile文件夹：存放每一次仿真得到的参数配置信息文件备份



TestData文件夹：存放测试用仿真数据，包括先验数据（由WDNwordPro文件夹中的EXATAPDgenerator文件生成）、迭代数据（由个次试验迭代生成，这里我们放一些例子）
- LHSprior文件夹：存放先验采样数据（实例，未包括全部数据）











联系方式：weirdfish31@whu.edu.cn



未补完