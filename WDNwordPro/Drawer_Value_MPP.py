# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:39:18 2018

@author: WDN
基本完成，基于最新的LHS数据，可以进行迭代过程之后的画图
得到的是评估值过程、概率过程和最终MPP模型的热力图
三个热力图在一起
"""
import WDNexataReader#读取数据
import WDNoptimizer#建模，画图，AF函数
import WDNfeedback#反馈
import pandas as pd
import numpy as np



"高斯混合模型动态优化过程预处理================================================="
z=[]#序号
superappinterval=[20]#superapp视频业务，需要的时延抖动小，吞吐量大
superappsize=[]
vbrinterval=[30]#vbr其他义务
vbrsize=[24000]
trafinterval=[30]#trafficgenerator图像流，需要的丢包率小，吞吐量大
trafsize=[]
listaaa=[]#用于存储querypoint的序列
appdata=pd.DataFrame()#所有数据库的某种业务的聚合（三种业务分开处理）
memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
valuegmmgamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#实例化MPP模型

figpath="./Figure/"#图像的存放位置
#datapath='G:/testData/2DGMM(16000_8000-36000)/'#先验数据的存放位置
datapath='D:/WDNoptimizer/LHSprior/'#LHS先验数据的存放位置
iternum=0#迭代的记数，在读取先验数据时记为零

"读取TXT文件中迭代的state数据==================================================="
statefilename="./history/priorstate_test.txt"#存储的先验数据的state列表txt文件
with open(statefilename, 'r') as file_to_read:
  while True:
      lines = file_to_read.readline() # 整行读取数据
      if not lines:
          "这里还有点数据格式上的问题，目前是这样处理，需要对生成的state文件进行处理"
          break
          pass
      z,si_tmp,ss_tmp,vi_tmp,vs_tmp,ti_tmp,ts_tmp = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
#      superappinterval.append(si_tmp)  # 添加新读取的数据
      superappsize.append(ss_tmp)
#      vbrinterval.append(vi_tmp)
#      vbrsize.append(vs_tmp)
#      trafinterval.append(ti_tmp)
      trafsize.append(ts_tmp) 
      pass
  pass
print(superappsize)
print(trafsize)

"读取训练数据集================================================================"
"""用来读取原始数据集，得到memoryset.probmemoryunit，绘制聚类图，拟合的GMM热力图"""

#outlogfile = open('./queryPoint.log', 'w')
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
                    for i in range(20):
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
"AF函数模型===================================================================="
"""
需要对目前的AF函数UCB进行修改
目前有两簇的output，err，均值较大簇的prob
目前的AF函数为HPP_WDUCBhelper
"""
ttt=valuegmmgamer.HPP_WDUCBhelper(memoryset.probmemoryunit,kappa=5,iternum=30,count=0)
tu=ttt.tolist()
listaaa.append(tu)

"画图+++++++++++++++++++++++++++++++++++++++++++===========================================+++++++++++++++++++++++"
"要绘制多指标合成的曲面，目前已经有模型参数，obj中提供"
valuegmmgamer.valuegragher_three(data=memoryset.probmemoryunit,qp=ttt,path=figpath)#多指标合成的画图                 




"新数据的位置和QP的列表==================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++===="
# =============================================================================
# newdatapath='E:/WDNoptimizer/2cluster_throughput_GMM/'#新产生的数据的存放位置
# listaaa=[[63670,63990],[63979,240],[24773,63990],[92,63993],[63730,32765],[34308,63966],[43992,43885],
#          [38937,47683],[44847,38317],[63158,315],[36956,47748],[30279,63982],[31258,53938],[29958,63988],
#          [45047,35287],[44082,34484],[32056,52314],[37898,43319],[37711,44133],[36070,47304],[37326,44649],
#          [37307,44596],[28670,53413],[27218,54348],[36729,43344],[34307,47951],[36263,44555],[33717,48207],
#          [24941,56549],[33413,46767]]
# =============================================================================
"====================================================================================================================="
#newdatapath='E:/WDNoptimizer/MPP_k5_i30_t10_p1/'#新产生的数据的存放位置
newdatapath='D:/WDNoptimizer/MPP_k5_i60_t10_p1/'#新产生的数据的存放位置
listMPP_k5_i30_p1=[[63701, 63986], [63934, 63958], [32, 63966], [93, 63795], [41, 63832], [122, 63901], 
                   [107, 63951], [25, 63986], [47, 63945], [38, 63836], [91, 63858], [28, 63802],  
                   [21, 63785], [123, 63954], [54, 63986], [126, 63765], [90, 63996], [104, 63922], [183, 63991],
                   [187, 63980], [145, 63951], [51, 63674], [52, 63702], [3, 63652], [3, 59988], [22642, 31954],
                   [22738, 32552], [22649, 33128], [22850, 33467]]

listMPP_k5_i60_t10_p1=[[62504, 168], [62419, 302], [62429, 187], [62577, 231], [62357, 298], [62486, 92], [62509, 177], 
                       [62448, 296], [62571, 230], [62382, 156], [62449, 184], [62412, 186], [62360, 204], [62419, 161], 
                       [62447, 199], [62445, 197], [62434, 209], [62493, 169], [62498, 200], [62485, 227], [62373, 212], 
                       [62360, 234], [62422, 231], [62513, 251], [62405, 146], [62407, 231], [62509, 270], [62454, 182], 
                       [62388, 172], [62389, 314], [62485, 237], [62436, 109], [62501, 210], [62461, 182], [62586, 197], 
                       [62517, 189], [62506, 71], [62470, 147], [62471, 136], [62324, 239], [62399, 309], [62404, 261], 
                       [62457, 158], [62447, 245], [62477, 266], [62397, 143], [62404, 191], [62569, 231], [62404, 307], 
                       [62403, 229], [9119, 7702], [9101, 7765], [62498, 278], [19730, 48152], [19693, 48142], [19745, 48171],
                       [19770, 48191], [19792, 48144], [19777, 48112], [19702, 48160]]

"循环画图，读取新数据迭代画图++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
"""根据原始数据集的模型和质询点，仿真X次，读取新的数据，加入到Priordataset，绘图，并找到下一个质询点"""
simucount=1
for i in listMPP_k5_i60_t10_p1:
    ttt=np.array(i)
    gamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)
    teaser=WDNfeedback.FeedBackWorker()#实例化反馈类
    teaser.updateQuerypointworker(ttt)#更新反馈参数
    newdata=teaser.updatetrainningsetworker(path=newdatapath,point=ttt,count=10,style='value')
    priordataset=priordataset.append(newdata)#将新数据加入至原始训练集中
    newgammer=WDNoptimizer.GMMmultiOptimizationUnit(cluster=2)#实例化GMM模型
    newdataset=newgammer.dropNaNworker(newdata)#去掉nan数据
    newdataset=newgammer.presortworker(newdataset,col1='vbri',col2='value')
    newdataset=newgammer.clusterworker(newdataset,col1='vbri',col2='value',count=iternum)#kmeans++聚类

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
        value0=np.mean(part1[part1['label']==0]['value'])
        memoryset.probinserter(state=state,value=value1,prob=probOf1,label=1)
        memoryset.probinserter(state=state,value=value0,prob=probOf0,label=0)    
    iternum=iternum+1
#    distriubuteculsterdata=distriubuteculsterdata.append(newdataset)
    "上面的新数据聚类完成，下面进行画图和querypoint的更新"
    valuegmmgamer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=6,label=0)#第一簇高斯过程模型
    valuegmmgamer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=7,label=0)#第一簇概率高斯过程模型
    valuegmmgamer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=6,label=1)#第二簇高斯过程模型
    valuegmmgamer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=7,label=1)#第二簇概率高斯过程模型
    valuegmmgamer.valuegragher_three(data=memoryset.probmemoryunit,qp=ttt,path=figpath,count=simucount)#多指标合成的画图         
    simucount=simucount+1#计数，修改文件名称


    
    
    
    
    
    
    
    
    
    
