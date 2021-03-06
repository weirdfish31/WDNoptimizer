# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:47:07 2018

@author: WDN
第三次实验，进行单目标(value)优化的修改，和聚类方式的修正
各个组合的数据进行混合模型拟合对各簇概率、各簇均值进行高斯过程建模、
"""
import WDNexataReader#读取数据
import WDNoptimizer#建模，画图，AF函数
import WDNfeedback#反馈
import pandas as pd
import numpy as np



"高斯混合模型动态优化过程预处理================================================="
"目前还是原来的先验数据，带实现LHS基于灵敏度分析的预采样"
superappinterval=[20]
#superappsize=[8000,10000,12000,14000,16000,18000,20000,22000,24000]
superappsize=[16000,30000]
vbrinterval=[30]
#vbrsize=[8000,10000,12000,14000,16000,18000]
vbrsize=[24000]
trafinterval=[30]
trafsize=[8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,32000,34000,36000]
#trafsize=[10000]
#trafsize=[22000]
#记录AF函数的每次选择
listaaa=[]


appdata=pd.DataFrame()#所有数据库的某种业务的聚合
memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
#distriubuteculsterdata=pd.DataFrame()#存储每次读取数据之后的分别聚类(均值，标签)的结果
valuegmmgamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#实例化GMM模型

#dataset='test_ REQUEST-SIZE EXP 18000 _ 2000'
#radio REQUEST-SIZE EXP 24000 _ 18000 _ RND EXP 22000
figpath="./Figure/"
datapath='G:/testData/2DGMM(16000_8000-36000)/'
newdatapath='./OutConfigfile/'
iternum=0


"读取训练数据集================================================================"
"""用来读取原始数据集，得到memoryset.probmemoryunit，绘制聚类图，拟合的GMM热力图"""

outlogfile = open('./queryPoint.log', 'w')
for sappi_i in superappinterval:
    for sapps_i in superappsize:
        for vbri_i in vbrinterval:
            for vbrs_i in vbrsize:
                for trafi_i in trafinterval: 
                    for trafs_i in trafsize:
                        """
                        gamer:对每一次一个输入组合的全部采样点做一次聚类
                        """
                        gamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#用来对一组参数中的数据聚类
                        tempmemoryset=WDNoptimizer.MemoryUnit()
                        state=[sappi_i,sapps_i,vbri_i,vbrs_i,trafi_i,trafs_i]
                        for i in range(20):
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
#                            print(sapps_i,vbrs_i,trafs_i)
                            "======================以上步骤不可省略，下方处理可以根据需求修改"
                            """
                            评估部分，对于三种不同的业务有不同的权重:时延、抖动、丢包率、吞吐量
                            vbr:        [1,2,3,4] delay,jitter,messagecompleterate,throughtput
                            trafficgen: [5,6,7,8]
                            superapp:   [9,10,11,12]
                            vbr,superapp,trafficgen
                            """
                            eva=WDNoptimizer.EvaluationUnit()
                            vbr=readdb.meandata('vbr')
                            eva.calculateMetricEvaValue(vbr)
                            trafficgen=readdb.meandata('trafficgen')
                            eva.calculateMetricEvaValue(trafficgen)
                            superapp=readdb.meandata('superapp')
                            eva.calculateMetricEvaValue(superapp)
                            value=eva.evaluationvalue()
                            """
                            状态动作保存：当前状态、评估值、动作、收益(目前的动作和收益没有用暂时放这里)
                            如果是第一次仿真，动作与收益为缺省值null
                            记忆单元的数据包括：
                                1）当前状态：6个输入量 三个业务的包大小，发包时间间隔
                                2）评估值：value=eva.evaluationvalue()
                            """
                            print(state)
                            """
                            "memoryset用来保存原始数据信息" 
                            "tempmemoryset用来进行读取数据是的聚类"
                            """
                            memoryset.valueinserter(state=state,value=value)
                            tempmemoryset.valueinserter(state=state,value=value)
                            print(eva.normalizedata)
                        """
                        "去掉nan数据"
                        "聚类"
                        "聚类后的数据保存到memoryset.probmemoryunit中"，最终得到每簇的均值、标签、概率
                        """
                        tempdataset=gamer.dropNaNworker(tempmemoryset.memoryunit)
                        tempdataset=gamer.presortworker(tempdataset,col1='vbri',col2='value')
                        tempdataset=gamer.clusterworker(tempdataset,col1='vbri',col2='value',count=iternum)
                        a=np.mean(tempdataset[tempdataset['label']==0]['value'])
                        b=np.mean(tempdataset[tempdataset['label']==1]['value'])   
                        if a<b:
                            part0=tempdataset.loc[tempdataset['label']==0]
                            part0.loc[:,'label']=0
                            part1=tempdataset.loc[tempdataset['label']==1]
                            part1.loc[:,'label']=1
#                            distriubuteculsterdata=distriubuteculsterdata.append(part0)
#                            distriubuteculsterdata=distriubuteculsterdata.append(part1)
                            """
                            计算混合模型中第一簇的概率，目前问题中的模型分为两簇，
                            计算一簇模型的概率自然可以得到另一簇的概率
                            """
                            probOf1=len(part1)/len(tempdataset)
                            probOf0=1-probOf1
                            value1=np.mean(part1[part1['label']==1]['value'])
                            value0=np.mean(part0[part0['label']==0]['value'])
                            memoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
                            memoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)
                        elif a>b:
                            part0=tempdataset.loc[tempdataset['label']==0]
                            part0.loc[:,'label']=1
                            part1=tempdataset.loc[tempdataset['label']==1]
                            part1.loc[:,'label']=0
#                            distriubuteculsterdata=distriubuteculsterdata.append(part0)
#                            distriubuteculsterdata=distriubuteculsterdata.append(part1)
                            probOf1=len(part0)/len(tempdataset)
                            probOf0=1-probOf1
                            value1=np.mean(part0[part0['label']==1]['value'])
                            value0=np.mean(part1[part1['label']==0]['value'])
                            memoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
                            memoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)                            
                        iternum=iternum+1
                        
                        
                        
                        
"数据预处理====目前预处理都是在读取数据时完成===================================="
#import WDNoptimizer
#distriubuteculsterdata=distriubuteculsterdata.reset_index(drop=True)
priordataset=memoryset.memoryunit#将原始的数据保存到内存中
print(memoryset.probmemoryunit)#这个数据是value均值、分簇概率，标签的综合数据，下面将利用这个数据进行GMM建模
print(priordataset)#原始数据包括state，value
#len(distriubuteculsterdata[distriubuteculsterdata['label']==0])
"建模GMM模型==================================================================="
valuegmmgamer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=6,label=0)#第一簇高斯过程模型
valuegmmgamer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=7,label=0)#第一簇概率高斯过程模型
valuegmmgamer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=6,label=1)#第二簇高斯过程模型
valuegmmgamer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=7,label=1)#第二簇概率高斯过程模型
#valuegmmgamer.obj['reg_prob_1']#test
#valuegmmgamer.obj['reg_value_0']#test
#valuegmmgamer.obj['err_value_1']#test
"AF函数模型===================================================================="
"""
需要对目前的AF函数UCB进行修改
目前有两簇的output，err，均值较大簇的prob
目前的AF函数为valueUCBhelper
"""
ttt=valuegmmgamer.valueUCBhelper(memoryset.probmemoryunit,kappa=1)
tu=ttt.tolist()
listaaa.append(tu)

"画图+++++++++++++++++++++++++++++++++++++++++++未完成+++++++++++++++++++++++"
"要绘制多指标合成的曲面，目前已经有模型参数，obj中提供"
valuegmmgamer.valuegragher(data=memoryset.probmemoryunit,qp=ttt,path=figpath)#多指标合成的画图

"反馈函数+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
"""根据原始数据集的模型和质询点，仿真X次，读取新的数据，加入到Priordataset，绘图，并找到下一个质询点"""
simucount=1
"把querypoint存储到log文件中"
for i in range(25):
    teaser=WDNfeedback.FeedBackWorker()#实例化反馈类
    teaser.updateQuerypointworker(ttt)#更新反馈参数
    "将反馈次数和querypoint写入log文件"
# =============================================================================
#     querypoint=str(ttt)
#     writeStr = "%s : {%s}\n" % (simucount, querypoint)
#     outlogfile.write(writeStr)
# =============================================================================
    teaser.runTest(count=10)#仿真
    newdata=teaser.updatetrainningsetworker(path=newdatapath,point=ttt,count=10,style='value')
    priordataset=priordataset.append(newdata)#将新数据加入至原始训练集中
    newgammer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#5实例化GMM模型
    iternum=iternum+1
    newdataset=newgammer.dropNaNworker(newdata)#去掉nan数据
    newdataset=newgammer.presortworker(newdataset,col1='vbri',col2='value')
    newdataset=newgammer.clusterworker(newdataset,col1='vbri',col2='value',count=iternum)#kmeans++聚类
    a=np.mean(newdataset[newdataset['label']==0]['value'])
    b=np.mean(newdataset[newdataset['label']==1]['value'])  
#    print(newdataset.loc[newdataset['label']==0])
    
    if a<b:
        part0=newdataset.loc[newdataset['label']==0]
        part0.loc[:,'label']=0
        part1=newdataset.loc[newdataset['label']==1]
        part1.loc[:,'label']=1
        probOf1=len(part1)/len(newdataset)
        probOf0=1-probOf1
        value1=np.mean(part1[part1['label']==1]['value'])
#        print(np.mean(part1[part1['label']==1]['value']))
        value0=np.mean(part0[part0['label']==0]['value'])
        memoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
        memoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)     

    elif a>b:
        part0=newdataset.loc[newdataset['label']==0]
        part0.loc[:,'label']=1
        part1=newdataset.loc[newdataset['label']==1]
        part1.loc[:,'label']=0
        probOf1=len(part0)/len(newdataset)
        probOf0=1-probOf1
        value1=np.mean(part0[part0['label']==1]['value'])
#        print(np.mean(part1[part1['label']==1]['value']))
        value0=np.mean(part1[part1['label']==0]['value'])
        memoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
        memoryset.probinserter(state=state,value=value0,prob=probOf0,label=0) 
#    distriubuteculsterdata=distriubuteculsterdata.append(newdataset)
    "上面的新数据聚类完成，下面进行画图和querypoint的更新"
    newgammer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=6,label=0)#第一簇高斯过程模型
    newgammer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=7,label=0)#第一簇概率高斯过程模型
    newgammer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=6,label=1)#第二簇高斯过程模型
    newgammer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=7,label=1)#第二簇概率高斯过程模型
    ttt=newgammer.valueUCBhelper(data=memoryset.probmemoryunit,kappa= 1)#AF函数
    tu=ttt.tolist()
    listaaa.append(tu)
    newgammer.valuegragher(data=memoryset.probmemoryunit,qp=ttt,path=figpath,count=simucount)#多指标合成的画图
    simucount=simucount+1#计数，修改文件名称
#记录每次AF选点的参数
with open('querypoint_log.txt','w') as f:
    f.write('\n')
    f.write(str(listaaa))
