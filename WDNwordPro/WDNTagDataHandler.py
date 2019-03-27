# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 22:23:25 2019

@author: WDN
读取迭代数据，进行标签，存放至LabelData
主要是评估值的标签 因为涉及到多目标优化的问题，还未进行QoS的标签功能
"""
import WDNexataReader#读取数据
import WDNoptimizer#建模，画图，AF函数
import pandas as pd
import numpy as np
import WDNfeedback
import WDNoptimizer
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

pd.set_option('display.max_rows',None)

# =============================================================================
# z=[]#序号
# superappinterval=[20]#superapp视频业务，需要的时延抖动小，吞吐量大
# superappsize=[]
# vbrinterval=[30]#vbr其他义务
# vbrsize=[24000]
# #vbrsize=[50000]
# trafinterval=[30]#trafficgenerator图像流，需要的丢包率小，吞吐量大
# trafsize=[]
# 
# #appdata=pd.DataFrame()#所有数据库的某种业务的聚合（三种业务分开处理）
# memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
# #valuegmmgamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#实例化GMM模型
# iternum=0#迭代的记数，在读取先验数据时记为零
# "选择路径与文件名，读取TXT文件中迭代的state数据=============================================="
# #datapath='E:/WDNoptimizer/LHSprior/'#LHS先验数据的存放位置
# #datapath='E:/WDNoptimizer/LHSMSE50000/'#LHS先验数据的存放位置
# datapath='E:/WDNoptimizer/LHSMSE24000/'#LHS先验数据的存放位置
# "-----------------------------------------------------------------------------------------"
# #statefilename="./history/priorstate_test.txt"#存储的先验数据的state列表txt文件
# #statefilename="./history/priorstate50000_all.txt"#存储的先验数据的state列表txt文件
# statefilename="./history/priorstate24000_all.txt"#存储的先验数据的state列表txt文件
# 
# with open(statefilename, 'r') as file_to_read:
#   while True:
#       lines = file_to_read.readline() # 整行读取数据
#       if not lines:
#           "这里还有点数据格式上的问题，目前是这样处理，需要对生成的state文件进行处理"
#           break
#           pass
#       z,si_tmp,ss_tmp,vi_tmp,vs_tmp,ti_tmp,ts_tmp = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
# #      superappinterval.append(si_tmp)  # 添加新读取的数据
#       superappsize.append(ss_tmp)
# #      vbrinterval.append(vi_tmp)
# #      vbrsize.append(vs_tmp)
# #      trafinterval.append(ti_tmp)
#       trafsize.append(ts_tmp) 
#       pass
#   pass
# print(superappsize)
# print(trafsize)
# 
# "读取训练数据集================================================================"
# """用来读取原始数据集，得到memoryset.probmemoryunit，绘制聚类图，拟合的GMM热力图"""
# 
# #outlogfile = open('./queryPoint.log', 'w')
# for sappi_i in superappinterval:
#     for vbri_i in vbrinterval:
#         for vbrs_i in vbrsize:
#             for trafi_i in trafinterval: 
#                 for count_i in range(len(superappsize)):
#                     """
#                     gamer:对每一次一个输入组合的全部采样点做一次聚类
#                     """
#                     gamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#用来对一组参数中的数据聚类
#                     tempmemoryset=WDNoptimizer.MemoryUnit()
#                     state=[sappi_i,superappsize[count_i],vbri_i,vbrs_i,trafi_i,trafsize[count_i]]
#                     for i in range(20):
#                         """
#                         读取数据，对数据进行分类处理
#                         """
#                         dataset='radio REQUEST-SIZE DET '+str(superappsize[count_i])+' _ '+str(vbrs_i)+' _ RND DET '+str(trafsize[count_i])+' _'+str(i)
#                         print(dataset)
#                         readdb=WDNexataReader.ExataDBreader()#实例化
#                         readdb.opendataset(dataset,datapath)#读取特定路径下的数据库
#                         readdb.appnamereader()#读取业务层的业务名称
#                         readdb.appfilter()#将业务名称分类至三个list
#                         readdb.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
#                         readdb.inputparainsert(sappi_i,superappsize[count_i],vbri_i,vbrs_i,trafi_i,trafsize[count_i])
#                         #将每条流的业务设计参数加入类中的字典
#                         "======================以上步骤不可省略，下方处理可以根据需求修改"
#                         """
#                         评估部分，对于三种不同的业务有不同的权重:时延、抖动、丢包率、吞吐量
#                         vbr:        [1,2,3,4] delay,jitter,messagecompleterate,throughtput
#                         trafficgen: [5,6,7,8]
#                         superapp:   [9,10,11,12]
#                         vbr,superapp,trafficgen
#                         """
#                         eva=WDNoptimizer.EvaluationUnit()
#                         vbr=readdb.meandata('vbr')
#                         eva.calculateMetricEvaValue(vbr)
#                         trafficgen=readdb.meandata('trafficgen')
#                         eva.calculateMetricEvaValue(trafficgen)
#                         superapp=readdb.meandata('superapp')
#                         eva.calculateMetricEvaValue(superapp)
#                         value=eva.evaluationvalue()
#                         """
#                         状态动作保存：当前状态、评估值、动作、收益(目前的动作和收益没有用暂时放这里)
#                         如果是第一次仿真，动作与收益为缺省值null
#                         记忆单元的数据包括：
#                             1）当前状态：6个输入量 三个业务的包大小，发包时间间隔
#                             2）评估值：value=eva.evaluationvalue()
#                         """
#                         print(state)
#                         """
#                         "memoryset用来保存原始数据信息" 
#                         "tempmemoryset用来进行读取数据是的聚类"
#                         """
# #                        memoryset.valueinserter(state=state,value=value)#原始的数据报的评估值的存储，这里注释
#                         tempmemoryset.valueinserter(state=state,value=value)
#                         print(eva.normalizedata)
#                     """
#                     "去掉nan数据"
#                     "聚类"
#                     "聚类后的数据保存到memoryset.probmemoryunit中"，最终得到每簇的均值、标签、概率
#                     """
#                     tempdataset=gamer.dropNaNworker(tempmemoryset.memoryunit)
#                     tempdataset=gamer.presortworker(tempdataset,col1='vbri',col2='value')
#                     tempdataset=gamer.clusterworker(tempdataset,col1='vbri',col2='value',count=iternum)
#                     a=np.mean(tempdataset[tempdataset['label']==0]['value'])
#                     b=np.mean(tempdataset[tempdataset['label']==1]['value'])   
#                     if a<b:
#                         part0=tempdataset.loc[tempdataset['label']==0]
#                         part0.loc[:,'label']=0
#                         part1=tempdataset.loc[tempdataset['label']==1]
#                         part1.loc[:,'label']=1
# #                            distriubuteculsterdata=distriubuteculsterdata.append(part0)
# #                            distriubuteculsterdata=distriubuteculsterdata.append(part1)
#                         """
#                         计算混合模型中第一簇的概率，目前问题中的模型分为两簇，
#                         计算一簇模型的概率自然可以得到另一簇的概率
#                         """
#                         probOf1=len(part1)/len(tempdataset)
#                         probOf0=1-probOf1
#                         value1=np.mean(part1[part1['label']==1]['value'])
#                         value0=np.mean(part0[part0['label']==0]['value'])
#                         memoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
#                         memoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)
#                     elif a>b:
#                         part0=tempdataset.loc[tempdataset['label']==0]
#                         part0.loc[:,'label']=1
#                         part1=tempdataset.loc[tempdataset['label']==1]
#                         part1.loc[:,'label']=0
# #                            distriubuteculsterdata=distriubuteculsterdata.append(part0)
# #                            distriubuteculsterdata=distriubuteculsterdata.append(part1)
#                         probOf1=len(part0)/len(tempdataset)
#                         probOf0=1-probOf1
#                         value1=np.mean(part0[part0['label']==1]['value'])
#                         value0=np.mean(part1[part1['label']==0]['value'])
#                         memoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
#                         memoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)                            
#                     iternum=iternum+1#聚类绘图需要，如不需要绘图则不需要这个参数
#                     
#                         
#                         
#                         
# "数据预处理====目前预处理都是在读取数据时完成===================================="
# #distriubuteculsterdata=distriubuteculsterdata.reset_index(drop=True)
# print(memoryset.probmemoryunit)#这个数据是value均值、分簇概率，标签的综合数据，下面将利用这个数据进行GMM建模
# 
# 
# 
# with open('./LabelData/LHS24000.txt','w') as f:#记录每次AF选点的参数
#     f.write('\n')
#     f.write(str(memoryset.probmemoryunit))#写入标记过的数据的DataFrame
#     
#     
# """
# 下面的程序使用来读取迭代数据的模块
# """
# 
# itermemoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
# newdatapath='E:/WDNoptimizer/GMM_i60_t10/'#新产生的数据的存放位置
# listGMM_i30=[[63709 ,63998 ],[63976 ,283   ],[24064 ,63980 ],[3     ,63876 ],[63973 ,35465 ],[34495 ,63986 ],
#             [45181 ,48150 ],[45054 ,48390 ],[33249 ,63986 ],[53340 ,34500 ],[42206 ,44016 ],[40228 ,45406 ],
#             [28527 ,63949 ],[63955 ,27404 ],[63706 ,217   ],[35303 ,49160 ],[23476 ,63994 ],[37209 ,46716 ],
#             [37468 ,46169 ],[32289 ,49794 ],[44269 ,37267 ],[63885 ,97    ],[22327 ,63966 ],[32624 ,63994 ],
#             [42535 ,39082 ],[32249 ,47910 ],[32600 ,47396 ],[41789 ,37975 ],[33214 ,46769 ],[43986 ,34883 ],
#             [32157 ,47348 ],[63716 ,83    ],[32437 ,49253 ],[42618 ,36109 ],[32586 ,48885 ],[44583 ,33893 ],
#             [31098 ,49494 ],[31923 ,46178 ],[31281 ,49419 ],[30590 ,50553 ],[30665 ,50260 ],[19424 ,63995 ],
#             [29946 ,50780 ],[33361 ,44745 ],[30368 ,50305 ],[50054 ,23937 ],[29988 ,50953 ],[32619 ,45542 ],
#             [31920 ,46328 ],[29687 ,51218 ],[30208 ,50682 ],[30433 ,50334 ],[34874 ,47880 ],[37153 ,41022 ],
#             [29175 ,51017 ],[35732 ,47203 ],[35318 ,63967 ],[28391 ,49667 ],[36563 ,43955 ],[28394 ,49467 ]]
# 
# "循环画图，读取新数据迭代画图++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
# """根据原始数据集的模型和质询点，仿真X次，读取新的数据，加入到Priordataset，绘图，并找到下一个质询点"""
# 
# for i in listGMM_i30:
#     ttt=np.array(i)
#     print(i[0])
#     gamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)
#     teaser=WDNfeedback.FeedBackWorker()#实例化反馈类
#     teaser.updateQuerypointworker(ttt)#更新反馈参数
#     newdata=teaser.updatetrainningsetworker(path=newdatapath,point=ttt,count=10,style='value')
#     newgammer=WDNoptimizer.GMMmultiOptimizationUnit(cluster=2)#实例化GMM模型
#     newdataset=newgammer.dropNaNworker(newdata)#去掉nan数据
#     newdataset=newgammer.presortworker(newdataset,col1='vbri',col2='value')
#     newdataset=newgammer.clusterworker(newdataset,col1='vbri',col2='value',count=iternum)#kmeans++聚类
#     state=[superappinterval[0],i[0],vbrinterval[0],vbrsize[0],trafinterval[0],i[1]]
#     a=np.mean(newdataset[newdataset['label']==0]['value'])
#     b=np.mean(newdataset[newdataset['label']==1]['value']) 
#     if a<b:
#         part0=tempdataset.loc[tempdataset['label']==0]
#         part0.loc[:,'label']=0
#         part1=tempdataset.loc[tempdataset['label']==1]
#         part1.loc[:,'label']=1
#         """
#         计算混合模型中第一簇的概率，目前问题中的模型分为两簇，
#         计算一簇模型的概率自然可以得到另一簇的概率
#         """
#         probOf1=len(part1)/len(tempdataset)
#         probOf0=1-probOf1
#         value1=np.mean(part1[part1['label']==1]['value'])
#         value0=np.mean(part0[part0['label']==0]['value'])
#         itermemoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
#         itermemoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)
#     elif a>b:
#         part0=tempdataset.loc[tempdataset['label']==0]
#         part0.loc[:,'label']=1
#         part1=tempdataset.loc[tempdataset['label']==1]
#         part1.loc[:,'label']=0
#         probOf1=len(part0)/len(tempdataset)
#         probOf0=1-probOf1
#         value1=np.mean(part0[part0['label']==1]['value'])
#         value0=np.mean(part1[part1['label']==0]['value'])
#         itermemoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
#         itermemoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)    
#     iternum=iternum+1
# 
# print(itermemoryset.probmemoryunit)#这个数据是value均值、分簇概率，标签的综合数据，下面将利用这个数据进行GMM建模
# 
# 
# 
# with open('./LabelData/ITER_GMM24000_i60_t10.txt','w') as f:#记录每次AF选点的参数
#     f.write('\n')
#     f.write(str(itermemoryset.probmemoryunit))#写入标记过的数据的DataFrame
# =============================================================================







class TaggedDataHandler:
    """
    对原始的仿真数据进行处理，得到聚类与评估后的特征值，在进行下一步的比较操作
    针对的是MPP模型的数据
    """
    def __init__(self):
        """
        初始化
        """
    def GridDataTagWriter(self,vbrs=24000,count_i=20,superappsize=[16000,30000],trafsize=[8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,32000,34000,36000],datapath='D:/WDNoptimizer/2DGMM(16000_8000-36000)/',savefilename='./LabelData/2DGMM(16000_8000-36000).txt'):
        """
        读取主观栅格数据
        """
        superappinterval=[20]
        vbrinterval=[30]
        vbrsize=[]
        vbrsize.append(vbrs)
        trafinterval=[30]
        memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
        iternum=0
        "读取训练数据集================================================================"
        """用来读取原始数据集，得到memoryset.probmemoryunit，绘制聚类图，拟合的GMM热力图"""
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
                                for i in range(count_i):
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
                                    probOf1=len(part0)/len(tempdataset)
                                    probOf0=1-probOf1
                                    value1=np.mean(part0[part0['label']==1]['value'])
                                    value0=np.mean(part1[part1['label']==0]['value'])
                                    memoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
                                    memoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)                            
                                iternum=iternum+1
                                
        "数据预处理====目前预处理都是在读取数据时完成===================================="
#        priordataset=memoryset.memoryunit#将原始的数据保存到内存中
        print(memoryset.probmemoryunit)#这个数据是value均值、分簇概率，标签的综合数据，下面将利用这个数据进行GMM建模
#        print(priordataset)#原始数据包括state，value
        with open(savefilename,'w') as f:#记录每次AF选点的参数
            f.write('\n')
            f.write(str(memoryset.probmemoryunit))#写入标记过的数据的DataFrame  
        print("mission completed！！！")
               
        
        
        
    def IterDataTagWriter(self,vbrs=24000,count_i=10,path='E:/WDNoptimizer/GMM_i60_t10/',QPlist=[],savefilename='./LabelData/ITER_GMM24000_i60_t10.txt'):
        """
        读取QP列表的点，对迭代数据进行处理
        """
        superappinterval=[20]#superapp视频业务，需要的时延抖动小，吞吐量大
#        superappsize=[]
        vbrinterval=[30]#vbr其他义务
        vbrsize=[]
        vbrsize.append(vbrs)
        #vbrsize=[50000]
        trafinterval=[30]#trafficgenerator图像流，需要的丢包率小，吞吐量大
#        trafsize=[]
        iternum=0#迭代的记数，在读取先验数据时记为零
        itermemoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
        newdatapath=path#新产生的数据的存放位置
        listGMM_i30=QPlist
        
        "循环画图，读取新数据迭代画图++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        """根据原始数据集的模型和质询点，仿真X次，读取新的数据，加入到Priordataset，绘图，并找到下一个质询点"""
        
        for i in listGMM_i30:
            ttt=np.array(i)
            teaser=WDNfeedback.FeedBackWorker()#实例化反馈类
            teaser.updateQuerypointworker(ttt)#更新反馈参数
            newdata=teaser.updatetrainningsetworker(path=newdatapath,point=ttt,count=count_i,style='value')
            newgammer=WDNoptimizer.GMMmultiOptimizationUnit(cluster=2)#实例化GMM模型
            newdataset=newgammer.dropNaNworker(newdata)#去掉nan数据
            newdataset=newgammer.presortworker(newdataset,col1='vbri',col2='value')
            newdataset=newgammer.clusterworker(newdataset,col1='vbri',col2='value',count=iternum)#kmeans++聚类
            state=[superappinterval[0],i[0],vbrinterval[0],vbrsize[0],trafinterval[0],i[1]]
            a=np.mean(newdataset[newdataset['label']==0]['value'])
            b=np.mean(newdataset[newdataset['label']==1]['value']) 
            if a<b:
                part0=newdataset.loc[newdataset['label']==0]
                part0.loc[:,'label']=0
                part1=newdataset.loc[newdataset['label']==1]
                part1.loc[:,'label']=1
                """
                计算混合模型中第一簇的概率，目前问题中的模型分为两簇，
                计算一簇模型的概率自然可以得到另一簇的概率
                """
                probOf1=len(part1)/len(newdataset)
                probOf0=1-probOf1
                value1=np.mean(part1[part1['label']==1]['value'])
                value0=np.mean(part0[part0['label']==0]['value'])
                itermemoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
                itermemoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)
            elif a>b:
                part0=newdataset.loc[newdataset['label']==0]
                part0.loc[:,'label']=1
                part1=newdataset.loc[newdataset['label']==1]
                part1.loc[:,'label']=0
                probOf1=len(part0)/len(newdataset)
                probOf0=1-probOf1
                value1=np.mean(part0[part0['label']==1]['value'])
                value0=np.mean(part1[part1['label']==0]['value'])
                itermemoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
                itermemoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)
            iternum=iternum+1
        print(itermemoryset.probmemoryunit)#这个数据是value均值、分簇概率，标签的综合数据，下面将利用这个数据进行GMM建模
        with open(savefilename,'w') as f:#记录每次AF选点的参数
            f.write('\n')
            f.write(str(itermemoryset.probmemoryunit))#写入标记过的数据的DataFrame  
        print("mission complete！！！")
    
    
    
    
    def PriorDataTagWriter_state(self,count=20,path='E:/WDNoptimizer/LHSprior/',filename="./history/priorstate_test.txt",savefilename='./LabelData/LHS24000.txt'):
        """
        读取原始先验数据的STATE文件，对先验数据进行评估聚类，并保存
        """
        z=[]#序号
        superappinterval=[]#superapp视频业务，需要的时延抖动小，吞吐量大
        superappsize=[]
        vbrinterval=[]#vbr其他义务
        vbrsize=[]
        trafinterval=[]#trafficgenerator图像流，需要的丢包率小，吞吐量大
        trafsize=[]
        memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
        iternum=0#迭代的记数，在读取先验数据时记为零
        "选择路径与文件名，读取TXT文件中迭代的state数据=============================================="
        datapath=path
#        datapath='E:/WDNoptimizer/LHSprior/'#LHS先验数据的存放位置
#        datapath='E:/WDNoptimizer/LHSMSE50000/'#LHS先验数据的存放位置
#        datapath='E:/WDNoptimizer/LHSMSE24000/'#LHS先验数据的存放位置
        "-----------------------------------------------------------------------------------------"
        statefilename=filename
#        statefilename="./history/priorstate_test.txt"#存储的先验数据的state列表txt文件
#        statefilename="./history/priorstate50000_all.txt"#存储的先验数据的state列表txt文件
#        statefilename="./history/priorstate24000_all.txt"#存储的先验数据的state列表txt文件 
        with open(statefilename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline() # 整行读取数据
                if not lines:
                    "这里还有点数据格式上的问题，目前是这样处理，需要对生成的state文件进行处理"
                    break
                    pass
                z,si_tmp,ss_tmp,vi_tmp,vs_tmp,ti_tmp,ts_tmp = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                superappinterval.append(math.ceil(si_tmp))
                superappsize.append(math.ceil(ss_tmp))
                vbrinterval.append(math.ceil(vi_tmp))
                vbrsize.append(math.ceil(vs_tmp))
                trafinterval.append(math.ceil(ti_tmp))
                trafsize.append(math.ceil(ts_tmp))
                pass
            pass
        "读取训练数据集================================================================"
        """用来读取原始数据集，得到memoryset.probmemoryunit，绘制聚类图，拟合的GMM热力图"""
        for count_i in range(len(superappsize)):
            """
            gamer:对每一次一个输入组合的全部采样点做一次聚类
            """
            gamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#用来对一组参数中的数据聚类
            tempmemoryset=WDNoptimizer.MemoryUnit()
            state=[superappinterval[count_i],superappsize[count_i],vbrinterval[count_i],vbrsize[count_i],trafinterval[count_i],trafsize[count_i]]
            for i in range(count):
                """
                读取数据，对数据进行分类处理
                """
                dataset = 'radio'+str(superappinterval[count_i])+"_"+str(superappsize[count_i])+"_"+str(vbrinterval[count_i])+"_"+str(vbrsize[count_i])+"_"+str(trafinterval[count_i])+"_"+str(trafsize[count_i])+'_'+str(i)
                print(dataset)
                readdb=WDNexataReader.ExataDBreader()#实例化
                readdb.opendataset(dataset,datapath)#读取特定路径下的数据库
                readdb.appnamereader()#读取业务层的业务名称
                readdb.appfilter()#将业务名称分类至三个list
                readdb.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
                readdb.inputparainsert(superappinterval[count_i],superappsize[count_i],vbrinterval[count_i],vbrsize[count_i],trafinterval[count_i],trafsize[count_i])
                #将每条流的业务设计参数加入类中的字典
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
                "tempmemoryset用来进行读取数据是的聚类"
                """
                tempmemoryset.valueinserter(state=state,value=value)
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
                probOf1=len(part0)/len(tempdataset)
                probOf0=1-probOf1
                value1=np.mean(part0[part0['label']==1]['value'])
                value0=np.mean(part1[part1['label']==0]['value'])
                memoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
                memoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)                            
            iternum=iternum+1#聚类绘图需要，如不需要绘图则不需要这个参数
        print(memoryset.probmemoryunit)
        with open(savefilename,'w') as f:#记录每次AF选点的参数
            f.write('\n')
            f.write(str(memoryset.probmemoryunit))#写入标记过的数据的DataFrame
        print('mission completed ! ! !')                    
    
    
    
    
    
    
    
    
    
    
    
    def PriorDataTagWriter(self,vbrs=24000,count=20,path='E:/WDNoptimizer/LHSprior/',filename="./history/priorstate_test.txt",savefilename='./LabelData/LHS24000.txt'):
        """
        读取原始先验数据的STATE文件，对先验数据进行评估聚类，并保存
        """
        z=[]#序号
        superappinterval=[20]#superapp视频业务，需要的时延抖动小，吞吐量大
        superappsize=[]
        vbrinterval=[30]#vbr其他义务
        vbrsize=[]
        vbrsize.append(vbrs)
        trafinterval=[30]#trafficgenerator图像流，需要的丢包率小，吞吐量大
        trafsize=[]
        memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
        iternum=0#迭代的记数，在读取先验数据时记为零
        "选择路径与文件名，读取TXT文件中迭代的state数据=============================================="
        datapath=path
#        datapath='E:/WDNoptimizer/LHSprior/'#LHS先验数据的存放位置
#        datapath='E:/WDNoptimizer/LHSMSE50000/'#LHS先验数据的存放位置
#        datapath='E:/WDNoptimizer/LHSMSE24000/'#LHS先验数据的存放位置
        "-----------------------------------------------------------------------------------------"
        statefilename=filename
#        statefilename="./history/priorstate_test.txt"#存储的先验数据的state列表txt文件
#        statefilename="./history/priorstate50000_all.txt"#存储的先验数据的state列表txt文件
#        statefilename="./history/priorstate24000_all.txt"#存储的先验数据的state列表txt文件 
        with open(statefilename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline() # 整行读取数据
                if not lines:
                    "这里还有点数据格式上的问题，目前是这样处理，需要对生成的state文件进行处理"
                    break
                    pass
                z,si_tmp,ss_tmp,vi_tmp,vs_tmp,ti_tmp,ts_tmp = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                superappsize.append(ss_tmp)
                trafsize.append(ts_tmp) 
                pass
            pass
        "读取训练数据集================================================================"
        """用来读取原始数据集，得到memoryset.probmemoryunit，绘制聚类图，拟合的GMM热力图"""
        for sappi_i in superappinterval:
            for vbri_i in vbrinterval:
                for vbrs_i in vbrsize:
                    for trafi_i in trafinterval: 
                        for count_i in range(len(superappsize)):
                            """
                            gamer:对每一次一个输入组合的全部采样点做一次聚类
                            """
                            gamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#用来对一组参数中的数据聚类
                            tempmemoryset=WDNoptimizer.MemoryUnit()
                            state=[sappi_i,superappsize[count_i],vbri_i,vbrs_i,trafi_i,trafsize[count_i]]
                            for i in range(count):
                                """
                                读取数据，对数据进行分类处理
                                """
                                dataset='radio REQUEST-SIZE DET '+str(superappsize[count_i])+' _ '+str(vbrs_i)+' _ RND DET '+str(trafsize[count_i])+' _'+str(i)
                                print(dataset)
                                readdb=WDNexataReader.ExataDBreader()#实例化
                                readdb.opendataset(dataset,datapath)#读取特定路径下的数据库
                                readdb.appnamereader()#读取业务层的业务名称
                                readdb.appfilter()#将业务名称分类至三个list
                                readdb.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
                                readdb.inputparainsert(sappi_i,superappsize[count_i],vbri_i,vbrs_i,trafi_i,trafsize[count_i])
                                #将每条流的业务设计参数加入类中的字典
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
                                "tempmemoryset用来进行读取数据是的聚类"
                                """
                                tempmemoryset.valueinserter(state=state,value=value)
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
                                probOf1=len(part0)/len(tempdataset)
                                probOf0=1-probOf1
                                value1=np.mean(part0[part0['label']==1]['value'])
                                value0=np.mean(part1[part1['label']==0]['value'])
                                memoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
                                memoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)                            
                            iternum=iternum+1#聚类绘图需要，如不需要绘图则不需要这个参数
        print(memoryset.probmemoryunit)
        with open(savefilename,'w') as f:#记录每次AF选点的参数
            f.write('\n')
            f.write(str(memoryset.probmemoryunit))#写入标记过的数据的DataFrame
        print('mission completed ! ! !')                        
    def LabelDataReader(self,filename="./LabelData/LHS24000_test.txt"):
        """
        读取TXT文件中迭代的标签数据特征值
        """
        z=[]#序号
        superappinterval=[]#superapp视频业务，需要的时延抖动小，吞吐量大
        superappsize=[]
        vbrinterval=[]#vbr其他义务
        vbrsize=[]
        trafinterval=[]#trafficgenerator图像流，需要的丢包率小，吞吐量大
        trafsize=[]
        value=[]
        prob=[]
        label=[]
        '---------------------------------------------------------------------------'
        labeldatafilename=filename#存储的先验数据的state列表txt文件
        with open(labeldatafilename, 'r') as file_to_read:
            
            while True:
                
                lines = file_to_read.readline() # 整行读取数据
                if not lines:
                    "这里还有点数据格式上的问题，目前是这样处理，需要对生成的state文件进行处理"
                    break
                    pass
                
                z,si_tmp,ss_tmp,vi_tmp,vs_tmp,ti_tmp,ts_tmp,v_tmp,p_tmp,l_tmp = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                superappinterval.append(si_tmp)  # 添加新读取的数据
                superappsize.append(ss_tmp)
                vbrinterval.append(vi_tmp)
                vbrsize.append(vs_tmp)
                trafinterval.append(ti_tmp)
                trafsize.append(ts_tmp) 
                value.append(v_tmp)
                prob.append(p_tmp)
                label.append(l_tmp)   
                pass
            pass
        '存入字典，得到DataFrame，方便进行后续操作,ps：字典转dataframe方便许多'
        dd={'sappi':superappinterval,
            'sapps':superappsize,
            'vbri':vbrinterval,
            'vbrs':vbrsize,
            'trafi':trafinterval,
            'trafs':trafsize,
            'value':value,
            'prob':prob,
            'label':label}
        data=pd.DataFrame(dd)
        print(data)
        print('mission completed ! ! !')    
        return data
    
    def RawDataReader(self,filename="./LabelData/LHS24000_test.txt"):
        """
        读取TXT文件中迭代的标签数据特征值
        """
        z=[]#序号
        superappinterval=[]#superapp视频业务，需要的时延抖动小，吞吐量大
        superappsize=[]
        vbrinterval=[]#vbr其他义务
        vbrsize=[]
        trafinterval=[]#trafficgenerator图像流，需要的丢包率小，吞吐量大
        trafsize=[]
        value=[]
        '---------------------------------------------------------------------------'
        labeldatafilename=filename#存储的先验数据的state列表txt文件
        with open(labeldatafilename, 'r') as file_to_read:
            
            while True:
                
                lines = file_to_read.readline() # 整行读取数据
                if not lines:
                    "这里还有点数据格式上的问题，目前是这样处理，需要对生成的state文件进行处理"
                    break
                    pass
                
                z,si_tmp,ss_tmp,vi_tmp,vs_tmp,ti_tmp,ts_tmp,v_tmp = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                superappinterval.append(si_tmp)  # 添加新读取的数据
                superappsize.append(ss_tmp)
                vbrinterval.append(vi_tmp)
                vbrsize.append(vs_tmp)
                trafinterval.append(ti_tmp)
                trafsize.append(ts_tmp) 
                value.append(v_tmp)
                pass
            pass
        '存入字典，得到DataFrame，方便进行后续操作,ps：字典转dataframe方便许多'
        dd={'sappi':superappinterval,
            'sapps':superappsize,
            'vbri':vbrinterval,
            'vbrs':vbrsize,
            'trafi':trafinterval,
            'trafs':trafsize,
            'value':value,}
        data=pd.DataFrame(dd)
        print(data)
        print('mission completed ! ! !')    
        return data    
    
    def ComparePrinter(self,MPP,GPR,RF,style='MSE'):
        """
        MSE参数的不同模型的比较画图
        """
        sns.set_style("whitegrid")
        plt.figure('Line fig',figsize=(20,6))
        plt.xlabel('Iteration Times')
        plt.ylabel(style)
        plt.title(style,fontsize='xx-large')
        plt.scatter(x=range(len(MPP)),y=MPP,marker='*',c='r')
        plt.scatter(x=range(len(GPR)),y=GPR,marker='o',c='black')
        plt.scatter(x=range(len(RF)),y=RF,marker='o',c='blue')
        plt.plot(MPP,color='r', linewidth=2, alpha=0.6,label='HPP')
        plt.plot(GPR,color='black', linewidth=2, alpha=0.6,label='GPR')
        plt.plot(RF,color='blue', linewidth=2, alpha=0.6,label='RF')
        plt.legend(fontsize='x-large')
            
    def ValueComparePrinter(self,MPP,GPR,RF,sim,style='prediction-simulation'):
        """
        MSE参数的不同模型的比较画图
        """
        sns.set_style("whitegrid")
        plt.figure('Line fig',figsize=(20,6))
        plt.xlabel('Iteration Times',fontsize='xx-large')
        plt.ylabel('predict-simulation',fontsize='xx-large')
        plt.title('value ',fontsize='xx-large')
        
        plt.scatter(x=range(len(MPP)),y=MPP,marker='*',c='r')
        plt.scatter(x=range(len(GPR)),y=GPR,marker='.',c='black')
        plt.scatter(x=range(len(RF)),y=RF,marker='o',c='blue')
        plt.plot(MPP,color='r', linewidth=2, alpha=0.6,label='MSP')
        plt.plot(GPR,color='black', linewidth=2, alpha=0.6,label='GP')
        plt.plot(RF,color='blue', linewidth=2, alpha=0.6,label='RF')
        plt.plot(sim,color='green', linewidth=2, alpha=0.6,label='simulation')
        plt.legend(fontsize='xx-large')

        
    def ZoominPrinter(self,mpp,gpr,rfr,style='R-Squared'):
        """
        进行zoomin的绘图
        """
        sns.set_style("whitegrid")
        fig,ax = plt.subplots(figsize=[20, 6])
        x=range(len(gpr))
        if style=='R-Squared':
            ax.scatter(x,mpp,marker='*',c='r')
            ax.scatter(x,gpr,marker='.',c='black')
            ax.scatter(x,rfr,marker='o',c='blue')
            ax.plot(x,rfr,color='blue', linewidth=2, alpha=0.6,label='RF')
            ax.plot(x,gpr,color='black', linewidth=2, alpha=0.6,label='GP')
            ax.plot(x,mpp,color='r', linewidth=2, alpha=0.6,label='MSP')
            ax.legend(fontsize='xx-large')
            plt.ylabel('R-squared',fontsize='xx-large')
            plt.xlabel('Iteration Times',fontsize='xx-large')
            plt.title('R-squared',fontsize='xx-large')
            
            axins=zoomed_inset_axes(ax, 5, loc=5)  # zoom = 6
            axins.plot(x,gpr,color='black', linewidth=2, alpha=0.6)
            axins.plot(x,rfr,color='blue', linewidth=2, alpha=0.6)
            axins.plot(x,mpp,color='r', linewidth=2, alpha=0.6)
            axins.scatter(x,mpp,marker='*',c='r')
            axins.scatter(x,gpr,marker='.',c='black')
            axins.scatter(x,rfr,marker='o',c='blue')
            
            # sub region of the original image
            x1, x2, y1, y2 = 50,55,-0.5,1.5
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            # fix the number of ticks on the inset axes
            axins.yaxis.get_major_locator().set_params(nbins=5)
            axins.xaxis.get_major_locator().set_params(nbins=5)
            
            plt.xticks(visible=True)
            plt.yticks(visible=True)
            plt.legend(fontsize='x-large')
            # draw a bbox of the region of the inset axes in the parent axes and
            # connecting lines between the bbox and the inset axes area
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            
            plt.draw()
            plt.show()
            
        elif style=='MSE':
            ax.scatter(x,mpp,marker='*',c='r')
            ax.scatter(x,gpr,marker='.',c='black')
            ax.scatter(x,rfr,marker='o',c='blue')
            ax.plot(x,rfr,color='blue', linewidth=2, alpha=0.6,label='RF')
            ax.plot(x,gpr,color='black', linewidth=2, alpha=0.6,label='GP')
            ax.plot(x,mpp,color='r', linewidth=2, alpha=0.6,label='MSP')
            ax.legend(fontsize='xx-large')
            plt.ylabel('MSE',fontsize='xx-large')
            plt.xlabel('Iteration Times',fontsize='xx-large')
            plt.title('MSE',fontsize='xx-large')
            
            axins=zoomed_inset_axes(ax, 5, loc=5)  # zoom = 6
            axins.plot(x,gpr,color='black', linewidth=2, alpha=0.6)
            axins.plot(x,rfr,color='blue', linewidth=2, alpha=0.6)
            axins.plot(x,mpp,color='r', linewidth=2, alpha=0.6)
            axins.scatter(x,mpp,marker='*',c='r')
            axins.scatter(x,gpr,marker='.',c='black')
            axins.scatter(x,rfr,marker='o',c='blue')
            
            # sub region of the original image
            x1, x2, y1, y2 = 50,55,0,0.03
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            # fix the number of ticks on the inset axes
            axins.yaxis.get_major_locator().set_params(nbins=5)
            axins.xaxis.get_major_locator().set_params(nbins=5)
            
            plt.xticks(visible=True)
            plt.yticks(visible=True)
            plt.legend(fontsize='x-large')
            # draw a bbox of the region of the inset axes in the parent axes and
            # connecting lines between the bbox and the inset axes area
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            
            plt.draw()
            plt.show()

        
        
        

class ModelCompareHandler:
    """
    针对处理过后的数据，重新进行建模与MSE比较绘图
    针对的是MPP模型，GPR模型
    目前正在进行RF（随机森林模型的制作）
    """
    def __init__(self):
        """
        初始化
        """
    def MPPmodelRebuilder(self,data):
        """
        基于数据，重建MPP模型
        """
        self.MPP=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#实例化MPP模型
        '---------------------------------------------------'
        self.MPP.gpbuilder(data,fitx=1,fity=5,fitz=6,label=0)#第一簇高斯过程模型
        self.MPP.gpbuilder(data,fitx=1,fity=5,fitz=7,label=0)#第一簇概率高斯过程模型
        self.MPP.gpbuilder(data,fitx=1,fity=5,fitz=6,label=1)#第二簇高斯过程模型
        self.MPP.gpbuilder(data,fitx=1,fity=5,fitz=7,label=1)#第二簇概率高斯过程模型        
    def MPPmodelRebuilder_state(self,data):
        """
        基于数据，重建MPP模型
        """
        self.MPP=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#实例化MPP模型
        '---------------------------------------------------'
        self.MPP.gpbuilder_state(data,fitz=6,label=0)#第一簇高斯过程模型
        self.MPP.gpbuilder_state(data,fitz=7,label=0)#第一簇概率高斯过程模型
        self.MPP.gpbuilder_state(data,fitz=6,label=1)#第二簇高斯过程模型
        self.MPP.gpbuilder_state(data,fitz=7,label=1)#第二簇概率高斯过程模型
                
    def GPRmodelRebuiler(self,data):
        """
        基于数据进行GPR模型的重建
        """
        self.GPR=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=1)#实例化MPP模型
        self.GPR.gpbuilder(data,fitx=1,fity=5,fitz=6,label=1)#第二簇高斯过程模型
        self.GPR.gpbuilder(data,fitx=1,fity=5,fitz=6,label=0)#第一簇高斯过程模型
    def GPRmodelRebuiler_state(self,data):
        """
        基于数据进行GPR模型的重建
        """
        self.GPR=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=1)#实例化MPP模型
        self.GPR.gpbuilder_state(data,fitz=6,label=1)#第二簇高斯过程模型
        self.GPR.gpbuilder_state(data,fitz=6,label=0)#第一簇高斯过程模型
        
    def RFmodelRebuilder(self,data):
        """
        基于数据进行RF回归模型的重建
        """
        self.RFR=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=1)#实例化RF模型
        self.RFR.rfbuilder(data,fitx=1,fity=5,fitz=6,label=0)#第一簇随机森林模型
        self.RFR.rfbuilder(data,fitx=1,fity=5,fitz=6,label=1)#第二簇随机森林模型
    def RFmodelRebuilder_state(self,data):
        """
        基于数据进行RF回归模型的重建
        """
        self.RFR=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=1)#实例化RF模型
        self.RFR.rfbuilder_state(data,fitz=6,label=0)#第一簇随机森林模型
        self.RFR.rfbuilder_state(data,fitz=6,label=1)#第二簇随机森林模型
    
    def RFpredicter(self,testdata):
        """
        基于此类中的基于标签过后的数据重新生成的RF模型，输入测试数据，进行评估值，概率的预测
        """
        bounds=pd.DataFrame()
        bounds['sapps']=testdata['sapps'][testdata['label']==1]
        bounds['trafs']=testdata['trafs'][testdata['label']==1]
        try_data = np.array(bounds)
        mean0=self.RFR.obj['reg_value_0'].predict(try_data)
        mean1=self.RFR.obj['reg_value_1'].predict(try_data)
        dd={'mean0':mean0,
            'mean1':mean1}
        data=pd.DataFrame(dd)
        return data
    def RFpredicter_state(self,testdata):
        """
        基于此类中的基于标签过后的数据重新生成的RF模型，输入测试数据，进行评估值，概率的预测
        """
        bounds=pd.DataFrame()
        bounds['sappi']=testdata['sappi'][testdata['label']==1]
        bounds['sapps']=testdata['sapps'][testdata['label']==1]
        bounds['vbri']=testdata['vbri'][testdata['label']==1]
        bounds['vbrs']=testdata['vbrs'][testdata['label']==1]
        bounds['trafi']=testdata['trafi'][testdata['label']==1]
        bounds['trafs']=testdata['trafs'][testdata['label']==1]
        try_data = np.array(bounds)
        mean0=self.RFR.obj['reg_value_0'].predict(try_data)
        mean1=self.RFR.obj['reg_value_1'].predict(try_data)
        dd={'mean0':mean0,
            'mean1':mean1}
        data=pd.DataFrame(dd)
        return data        
             
    def MPPpredicter(self,testdata):
        """
        基于此类中的基于标签过后的数据重新生成的MPP模型，输入测试数据，进行评估值，概率的预测
        """
        bounds=pd.DataFrame()
        bounds['sapps']=testdata['sapps'][testdata['label']==1]
        bounds['trafs']=testdata['trafs'][testdata['label']==1]
        try_data = np.array(bounds)
        mean0,std0=self.MPP.obj['reg_value_0'].predict(try_data,return_std=True)
        mean1,std1=self.MPP.obj['reg_value_1'].predict(try_data,return_std=True)
        prob0=self.MPP.obj['reg_prob_0'].predict(try_data,return_std=False)
        prob1=self.MPP.obj['reg_prob_1'].predict(try_data,return_std=False)
        dd={'mean0':mean0,
            'std0':std0,
            'prob0':prob0,
            'mean1':mean1,
            'std1':std1,
            'prob1':prob1}
        data=pd.DataFrame(dd)
        return data        
    def MPPpredicter_state(self,testdata):
        """
        基于此类中的基于标签过后的数据重新生成的MPP模型，输入测试数据，进行评估值，概率的预测
        """
        bounds=pd.DataFrame()
        bounds['sappi']=testdata['sappi'][testdata['label']==1]
        bounds['sapps']=testdata['sapps'][testdata['label']==1]
        bounds['vbri']=testdata['vbri'][testdata['label']==1]
        bounds['vbrs']=testdata['vbrs'][testdata['label']==1]
        bounds['trafi']=testdata['trafi'][testdata['label']==1]
        bounds['trafs']=testdata['trafs'][testdata['label']==1]
        try_data = np.array(bounds)
        mean0,std0=self.MPP.obj['reg_value_0'].predict(try_data,return_std=True)
        mean1,std1=self.MPP.obj['reg_value_1'].predict(try_data,return_std=True)
        prob0=self.MPP.obj['reg_prob_0'].predict(try_data,return_std=False)
        prob1=self.MPP.obj['reg_prob_1'].predict(try_data,return_std=False)
        dd={'mean0':mean0,
            'std0':std0,
            'prob0':prob0,
            'mean1':mean1,
            'std1':std1,
            'prob1':prob1}
        data=pd.DataFrame(dd)
        return data          
    
    def GPRpredicter(self,testdata):
        """
        基于此类中的基于标签过后的数据重新生成模型，输入测试数据，进行评估值，概率的预测
        1)必须先对GPR模型重新进行建模
        2)这里对两簇的数据都进行预测
        """
        bounds=pd.DataFrame()
        bounds['sapps']=testdata['sapps'][testdata['label']==1]
        bounds['trafs']=testdata['trafs'][testdata['label']==1]
        try_data = np.array(bounds)
        mean0,std0=self.GPR.obj['reg_value_0'].predict(try_data,return_std=True)
        mean1,std1=self.GPR.obj['reg_value_1'].predict(try_data,return_std=True)
#        mean=(mean0+mean1)/2
#        std=(std0+std1)/2
        dd={'mean':mean0,#这里mean为第一簇的均值
            'std':std0}
        data=pd.DataFrame(dd)
        return data
    def GPRpredicter_state(self,testdata):
        """
        基于此类中的基于标签过后的数据重新生成模型，输入测试数据，进行评估值，概率的预测
        1)必须先对GPR模型重新进行建模
        2)这里对两簇的数据都进行预测
        """
        bounds=pd.DataFrame()
        bounds['sappi']=testdata['sappi'][testdata['label']==1]
        bounds['sapps']=testdata['sapps'][testdata['label']==1]
        bounds['vbri']=testdata['vbri'][testdata['label']==1]
        bounds['vbrs']=testdata['vbrs'][testdata['label']==1]
        bounds['trafi']=testdata['trafi'][testdata['label']==1]
        bounds['trafs']=testdata['trafs'][testdata['label']==1]
        try_data = np.array(bounds)
        mean0,std0=self.GPR.obj['reg_value_0'].predict(try_data,return_std=True)
        mean1,std1=self.GPR.obj['reg_value_1'].predict(try_data,return_std=True)
#        mean=(mean0+mean1)/2
#        std=(std0+std1)/2
        dd={'mean':mean0,#这里mean为第一簇的均值
            'std':std0}
        data=pd.DataFrame(dd)
        return data                

    def MPPMSE(self,testdata,predictdata):
        """
        MPP模型的均的MSE误差
        """
        aaa=mean_squared_error(testdata['value'][testdata['label']==1],predictdata['mean1'])+mean_squared_error(testdata['value'][testdata['label']==0],predictdata['mean0'])
        aaa=aaa/2
        return aaa
#        a=testdata['value'][testdata['label']==1].reset_index(drop=True)
#        b=predictdata['mean1'].reset_index(drop=True)
#        c=testdata['value'][testdata['label']==0].reset_index(drop=True)
#        d=predictdata['mean0'].reset_index(drop=True)
#        MPPMSE1=(a-b)*(a-b)
#        w=MPPMSE1.sum()
#        print(w)
#        MPPMSE0=(c-d)*(c-d)
#        q=MPPMSE0.sum()
#        print(q)
#        MPPMSE=(w+q)/len(testdata)
#        print(MPPMSE)
#        return MPPMSE
    
    def GPRMSE(self,testdata,predictdata):
        """
        GPR模型的均的MSE误差
        """
        aaa=mean_squared_error(testdata['value'][testdata['label']==1]*testdata['prob'][testdata['label']==1],predictdata['mean'])+mean_squared_error(testdata['value'][testdata['label']==0]*testdata['prob'][testdata['label']==0],predictdata['mean'])
        aaa=aaa/2
        return aaa
#        a=testdata['value'][testdata['label']==1].reset_index(drop=True)
#        b=predictdata['mean'].reset_index(drop=True)
#        c=testdata['value'][testdata['label']==0].reset_index(drop=True)
#        d=predictdata['mean'].reset_index(drop=True)
#        GPRMSE1=(a-b)*(a-b)
#        w=GPRMSE1.sum()
#        print(w)
#        GPRMSE0=(c-d)*(c-d)
#        q=GPRMSE0.sum()
#        print(q)
#        GPRMSE=(w+q)/len(testdata)
#        print(GPRMSE)
#        return GPRMSE
    
    def RFMSE(self,testdata,predictdata):
        """
        RF模型的均的MSE误差,这里也与GPR一样用mean0进行计算
        """
        aaa=mean_squared_error(testdata['value'][testdata['label']==1],predictdata['mean0'])+mean_squared_error(testdata['value'][testdata['label']==0],predictdata['mean0'])
        aaa=aaa
        return aaa
#        a=testdata['value'][testdata['label']==1].reset_index(drop=True)
#        b=predictdata['mean0'].reset_index(drop=True)
#        c=testdata['value'][testdata['label']==0].reset_index(drop=True)
#        d=predictdata['mean1'].reset_index(drop=True)
#        RFMSE1=(a-b)*(a-b)
#        w=RFMSE1.sum()
#        print(w)
#        RFMSE0=(c-b)*(c-b)
#        q=RFMSE0.sum()
#        print(q)
#        RFMSE=(w+q)/len(testdata)
#        print(RFMSE)
#        return RFMSE

        
    def iterpredicthandler(self,priordata):
        """
        对模型的下一个点进行预测其输出
        """





















