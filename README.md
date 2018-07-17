# WDNoptimizer
![](http://m.qpic.cn/psb?/V10dYaiX2qXpCo/0KCd0cw4u2u68rWY1EbR8xBM2Jh38OZb18PU01Ht1Vw!/b/dC4BAAAAAAAA&bo=zQPzAc0D8wEDORw!&rf=viewer_4&t=5)

**Wise Dynamic Network optimizer** 此项目主要是用于对EXATA中的卫星通信网络场景进行快速配置与仿真，针对仿真数据进行统计分析，建立仿真模型，对设计进行优化。



TIPS： 1）需要安装 STK 9.2.3，EXATA 5.4。

组成说明：

WDNwordPro文件夹： 主程序文件夹，包括生成EXATA所需的连通性表表、各种config文件，仿真数据处理python程序等

- configfile文件夹： 主要存放生成EXATA配置文件所需的各种参数配置json文件，包括场景中移动节点文件夹、FTP参数json文件、地面站节点参数、链路参数json文件、移动节点名称配置文件等
- Debug文件夹： STK生成可见性报表，生成节点坐标轨迹文件的exe程序文件夹
- nodetxt文件夹： 存放给EXATA生成场景配置文件的节点坐标txt文件
- outAccess文件夹： 存放给EXATA生成link相关配置文件的STK节点之间的csv可见性报表

未补完