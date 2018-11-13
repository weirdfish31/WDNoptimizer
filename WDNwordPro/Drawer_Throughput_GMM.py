# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:31:31 2018

@author: WDN
用来对superapp和trafficgenerator的throughput进行GMM模型的绘制
"""
import WDNexataReader#读取数据
import WDNoptimizer#建模，画图，AF函数
import WDNfeedback#反馈
import pandas as pd
import numpy as np


"高斯混合模型动态优化过程预处理================================================="
superappinterval=[20]
#superappsize=[8000,10000,12000,14000,16000,18000,20000,22000,24000]
superappsize=[16000,30000]
vbrinterval=[30]
#vbrsize=[8000,10000,12000,14000,16000,18000]
vbrsize=[24000]
trafinterval=[30]
trafsize=[8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,32000,34000,36000]
#trafsize=[22000]

memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
distriubuteculsterdata=pd.DataFrame()

#dataset='test_ REQUEST-SIZE EXP 18000 _ 2000'
#radio REQUEST-SIZE EXP 24000 _ 18000 _ RND EXP 22000
figpath="./Figure/"
datapath_old='G:/testData/2DGMM(16000_8000-36000)/'
datapath='G:/testData/2cluster_throughput_GMM/'
iternum=0



"读取训练数据集================================================================"
"""用来读取原始数据集，得到priordataset，绘制聚类图，拟合的GMM热力图"""
for sappi_i in superappinterval:
    for sapps_i in superappsize:
        for vbri_i in vbrinterval:
            for vbrs_i in vbrsize:
                for trafi_i in trafinterval: 
                    for trafs_i in trafsize:
                        gamer=WDNoptimizer.GMMmultiOptimizationUnit(cluster=2)
                        tempmemoryset=WDNoptimizer.MemoryUnit()
                        for i in range(30):
                            """
                            读取数据，对数据进行分类处理
                            """
                            dataset='radio REQUEST-SIZE DET '+str(sapps_i)+' _ '+str(vbrs_i)+' _ RND DET '+str(trafs_i)+' _'+str(i)
                            readdb=WDNexataReader.ExataDBreader()#实例化
                            readdb.opendataset(dataset,datapath_old)#读取特定路径下的数据库
                            readdb.appnamereader()#读取业务层的业务名称
                            readdb.appfilter()#将业务名称分类至三个list
                            readdb.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
                            readdb.inputparainsert(sappi_i,sapps_i,vbri_i,vbrs_i,trafi_i,trafs_i)
                            #将每条流的业务设计参数加入类中的字典
                            print(sapps_i,vbrs_i,trafs_i)
                            "======================以上步骤不可省略，下方处理可以根据需求修改"                       
                            """
                            评估部分，对于三种不同的业务有不同的权重:时延、抖动、丢包率、吞吐量
                            vbr:        [1,2,3,4]
                            trafficgen: [5,6,7,8]
                            superapp:   [9,10,11,12]
                            vbr,superapp,trafficgen
                            """
                            eva=WDNoptimizer.EvaluationUnit()
                            superapp=readdb.meandata('superapp')
                            eva.calculateMetricEvaValue(superapp)
                            vbr=readdb.meandata('vbr')
                            eva.calculateMetricEvaValue(vbr)
                            trafficgen=readdb.meandata('trafficgen')
                            eva.calculateMetricEvaValue(trafficgen)                      
                            """
                            状态动作保存：当前状态、评估值、动作、收益
                            如果是第一次仿真，动作与收益为缺省值null
                            记忆单元的数据包括：
                                1）当前状态：6个输入量 三个业务的包大小，发包时间间隔
                                2）评估值：value=eva.evaluationvalue()
                                3）动作：6个输入量与上一次仿真中修改的变化量
                                4）收益：动作之后的评估值的变化，遇上一次评估值之间的差
                            """
                            state=[sappi_i,sapps_i,vbri_i,vbrs_i,trafi_i,trafs_i]
                            print(state)
                            qos=eva.qoslist
                            memoryset.qosinserter(state=state,qos=qos)                    
                            tempmemoryset.qosinserter(state=state,qos=qos)
                        tempdataset=gamer.dropNaNworker(tempmemoryset.qosmemoryunit)
                        tempdataset=gamer.presortworker(tempdataset,col1='traf_throughput',col2='sapp_throughput')
                        tempdataset=gamer.clusterworker(tempdataset,col1='traf_throughput',col2='sapp_throughput',count=iternum)
                        distriubuteculsterdata=distriubuteculsterdata.append(tempdataset)                        
                        iternum=iternum+1
                        
"数据预处理===================================================================="
import WDNoptimizer
distriubuteculsterdata=distriubuteculsterdata.reset_index(drop=True)
priordataset=memoryset.qosmemoryunit#将原始的数据保存到内存中
qosgmmgamer=WDNoptimizer.GMMmultiOptimizationUnit(cluster=2)#实例化GMM模型
print(distriubuteculsterdata)#这个聚类结果是分别对每一组数据进行聚类之后聚合而成的数据
print(priordataset)
"AF函数======================================================================="
listaaa=[[63670,63990],[63979,240],[24773,63990],[92,63993],[63730,32765],[34308,63966],[43992,43885],
         [38937,47683],[44847,38317],[63158,315],[36956,47748],[30279,63982],[31258,53938],[29958,63988],
         [45047,35287],[44082,34484],[32056,52314],[37898,43319],[37711,44133],[36070,47304],[37326,44649],
         [37307,44596],[28670,53413],[27218,54348],[36729,43344],[34307,47951],[36263,44555],[33717,48207],
         [24941,56549],[33413,46767]]
print(type(listaaa[1]))
ttt=np.array([41,485])
"画图+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
##fitz=7 16
qosgmmgamer.gmmbuilder(distriubuteculsterdata,fitx=1,fity=5,fitz=17)#生成traf_messagecompletionrate均值，标准差平面的预测结果，用于画图
qosgmmgamer.gmmbuilder(distriubuteculsterdata,fitx=1,fity=5,fitz=9)#生成sapp_jitter均值标准差的平面的预测结果，用于画图
qosgmmgamer.multiGMMbuilder(distriubuteculsterdata,fitz=9,fita=17)#生成多指标的加权平面，保存的功能还未实现，需要实现
"要绘制多指标合成的曲面，必须进行上面两个步骤，生成"
qosgmmgamer.mulitgragher(data=distriubuteculsterdata,test=ttt,path=figpath)#多指标合成的画图

"反馈函数+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
"""根据原始数据集的模型和质询点，仿真X次，读取新的数据，加入到Priordataset，绘图，并找到下一个质询点"""
simucount=1
for i in listaaa:
    ttt=np.array(i)
    teaser=WDNfeedback.FeedBackWorker()#实例化反馈类
    teaser.updateQuerypointworker(ttt)#更新反馈参数
    newdata=teaser.updatetrainningsetworker(path=datapath,point=ttt,count=30)
    priordataset=priordataset.append(newdata)#将新数据加入至原始训练集中
    newgammer=WDNoptimizer.GMMmultiOptimizationUnit(cluster=2)#实例化GMM模型
    newdataset=newgammer.dropNaNworker(newdata)#去掉nan数据
    iternum=iternum+1
    newdataset=gamer.presortworker(newdataset,col1='traf_throughput',col2='sapp_throughput')
    newdataset=newgammer.clusterworker(newdataset,col1='traf_throughput',col2='sapp_throughput',count=iternum)#kmeans++聚类
    distriubuteculsterdata=distriubuteculsterdata.append(newdataset)
    "上面的新数据聚类完成，下面进行画图和querypoint的更新"
    newgammer.gmmbuilder(distriubuteculsterdata,fitx=1,fity=5,fitz=17)#生成traf_messagecompletionrate均值，标准差平面的预测结果，用于画图
    newgammer.gmmbuilder(distriubuteculsterdata,fitx=1,fity=5,fitz=9)#生成sapp_jitter均值标准差的平面的预测结果，用于画图
    newgammer.multiGMMbuilder(distriubuteculsterdata,fitz=9,fita=17)#生成多指标的加权平面，保存的功能还未实现，需要实现
    newgammer.mulitgragher(data=distriubuteculsterdata,test=ttt,path=figpath,count=simucount)#多指标合成的画图
    simucount=simucount+1#计数，修改文件名称


    
    
    
    
    
    
    
    
    
    
