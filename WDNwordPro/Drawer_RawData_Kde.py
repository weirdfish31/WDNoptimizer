# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:16:25 2018

@author: WDN
研究在相同设计参数的情况下，仿真的评估值的分布情况
本程序2用来对大量重复实验中的各种Qos结果进行KDE绘图
"""
import WDNexataReader#读取数据
import WDNoptimizer#建模，画图，AF函数
import pandas as pd



superappinterval=[20]
#superappsize=[8000,10000,12000,14000,16000,18000,20000,22000,24000]
superappsize=[12000]
vbrinterval=[30]
#vbrsize=[8000,10000,12000,14000,16000,18000]
vbrsize=[16000]
trafinterval=[30]
#trafsize=[8000,10000,12000,14000,16000,18000,20000,22000,24000]
trafsize=[30000]

flowdata=pd.DataFrame()#所有数据库的流聚合
appdata=pd.DataFrame()#所有数据库的某种业务的聚合
memoryset=WDNoptimizer.ReinforcementLearningUnit()#记忆单元，存储每次的状态

#dataset='test_ REQUEST-SIZE EXP 18000 _ 2000'
#radio REQUEST-SIZE EXP 24000 _ 18000 _ RND EXP 22000

figpath="./Figure/"
datapath='G:/testData/12000_16000_30000/'
#datapath='./OutConfigfile/'

for i in range(120):
    for sappi_i in superappinterval:
        for sapps_i in superappsize:
            for vbri_i in vbrinterval:
                for vbrs_i in vbrsize:
                    for trafi_i in trafinterval: 
                        for trafs_i in trafsize:
                            i=i+354
                            """
                            读取数据，对数据进行分类处理
                            """
                            
                            dataset='radio REQUEST-SIZE DET '+str(sapps_i)+' _ '+str(vbrs_i)+' _ RND DET '+str(trafs_i)+' _'+str(i)
                            readdb=WDNexataReader.ExataDBreader()#实例化
                            readdb.opendataset(dataset,datapath)#读取特定路径下的数据库
                            readdb.appnamereader()#读取业务层的业务名称
                            readdb.appfilter()#将业务名称分类至三个list
                            readdb.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
                            readdb.inputparainsert(sappi_i,sapps_i,vbri_i,vbrs_i,trafi_i,trafs_i)
                            #将每条流的业务设计参数加入类中的字典
                            print(sapps_i,vbrs_i,trafs_i)
                            #===================================================以上步骤不可省略，下方处理可以根据需求修改
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
                            value=eva.evaluationvalue()                     
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
                            print(qos)
                            memoryset.insertmemoryunit(state=state,value=value)
                            memoryset.qosinserter(state=state,qos=qos)
                            
                        
print(memoryset.qosmemoryunit)
import seaborn as sns 
sns.jointplot('traf_messagecompletionrate','sapp_jitter',data=memoryset.qosmemoryunit, kind='kde')        
#==============================================================================                 
#GMMgamer=weirdfishes.GMMOptimizationUnit()
#data=GMMgamer.dropNaNworker(memoryset.memoryunit)                          
#c=GMMgamer.clusterworker(data,col1="value",col2="trafs")
#print(c)
#d=GMMgamer.componentselecter(c,0)
#print(d)
#bayesgamer=weirdfishes.BayesianOptimizationUnit()
#bayesgamer.gussianproccessfitter(d)
#test=bayesgamer.acquisitionfunction(kappa=0.67)
#bayesgamer.heatpointer(test)

# =============================================================================
# #EM算法
# from sklearn.mixture import GMM
# gmm = GMM(n_components=3,n_iter=1000).fit(c)
# print(gmm)
# labels = gmm.predict(c)
# print(labels)
# plt.scatter(c[:, 0], c[:, 1],c=labels, s=40, cmap='viridis')
# probs = gmm.predict_proba(c)
# print(probs[:5].round(3))
# size = 50 * probs.max(1) ** 2  # 由圆点面积反应概率值的差异
# plt.scatter(c[:, 0], c[:, 1], c=labels, cmap='viridis', s=size)
# =============================================================================

















