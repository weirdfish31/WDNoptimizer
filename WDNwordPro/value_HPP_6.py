# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:39:18 2018

@author: WDN
修改了数据的读取方式和模型，修改了AF函数的策略，部分指标的权重
重新进行命名规则
"""
import WDNexataReader#读取数据
import WDNoptimizer#建模，画图，AF函数
import WDNfeedback#反馈
import pandas as pd
import numpy as np
import math


"高斯混合模型动态优化过程预处理================================================="
z=[]#序号
superappinterval=[]#superapp视频业务，需要的时延抖动小，吞吐量大
superappsize=[]
vbrinterval=[]#vbr其他义务
vbrsize=[]
trafinterval=[]#trafficgenerator图像流，需要的丢包率小，吞吐量大
trafsize=[]
listaaa=[]#用于存储querypoint的序列
appdata=pd.DataFrame()#所有数据库的某种业务的聚合（三种业务分开处理）
memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
valuegmmgamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#实例化GMM模型

figpath="./Figure/"#图像的存放位置
newdatapath='./OutConfigfile/'#新产生的数据的存放位置
iternum=0#迭代的记数，在读取先验数据时记为零
"选择路径与文件名，读取TXT文件中迭代的state数据=============================================="
datapath='G:/data/LHS6D_prior/'#LHS先验数据的存放位置
#datapath='E:/WDNoptimizer/LHSMSE50000/'#LHS先验数据的存放位置
#datapath='E:/WDNoptimizer/LHSMSE24000/'#LHS先验数据的存放位置
"-----------------------------------------------------------------------------------------"
statefilename="G:/data/LHS6D_prior/priorLHS6D_train.txt"#存储的先验数据的state列表txt文件
#statefilename="./history/priorstate50000_all.txt"#存储的先验数据的state列表txt文件
#statefilename="./history/priorstate24000_all.txt"#存储的先验数据的state列表txt文件

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
print(superappsize)
print(trafsize)
print(vbrsize)
vbrsize = [ math.ceil(x) for x in vbrsize ]
vbrsize


"读取训练数据集================================================================"
"""用来读取原始数据集，得到memoryset.probmemoryunit，绘制聚类图，拟合的GMM热力图"""
for count_i in range(len(superappsize)):
    """
    gamer:对每一次一个输入组合的全部采样点做一次聚类
    """
    gamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#用来对一组参数中的数据聚类
    tempmemoryset=WDNoptimizer.MemoryUnit()
    state=[superappinterval[count_i],superappsize[count_i],vbrinterval[count_i],vbrsize[count_i],trafinterval[count_i],trafsize[count_i]]
    for i in range(20):
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
#import WDNoptimizer
#distriubuteculsterdata=distriubuteculsterdata.reset_index(drop=True)
priordataset=memoryset.memoryunit#将原始的数据保存到内存中
print(memoryset.probmemoryunit)#这个数据是value均值、分簇概率，标签的综合数据，下面将利用这个数据进行GMM建模
print(priordataset)#原始数据包括state，value
#len(distriubuteculsterdata[distriubuteculsterdata['label']==0])

"建模HPP模型==================================================================="

valuegmmgamer.gpbuilder_state(memoryset.probmemoryunit,fitz=6,label=0)#第一簇高斯过程模型
valuegmmgamer.gpbuilder_state(memoryset.probmemoryunit,fitz=7,label=0)#第一簇概率高斯过程模型
valuegmmgamer.gpbuilder_state(memoryset.probmemoryunit,fitz=6,label=1)#第二簇高斯过程模型
valuegmmgamer.gpbuilder_state(memoryset.probmemoryunit,fitz=7,label=1)#第二簇概率高斯过程模型

"AF函数模型===================================================================="
"""
需要对目前的AF函数UCB进行修改
目前有两簇的output，err，均值较大簇的prob
目前的AF函数为valueUCBhelper
"""
ttt=valuegmmgamer.valueUCBhelper_HPP_state(memoryset.probmemoryunit,kappa=0,iternum=100,count=0)
tu=ttt.tolist()
listaaa.append(tu)
with open('QP_HPP_D6.txt','a') as f:#记录每次AF选点的参数
    f.write(str(tu)+',')#写入listaaa，querypoint的序列

"反馈函数+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
"""根据原始数据集的模型和质询点，仿真X次，读取新的数据，加入到Priordataset，绘图，并找到下一个质询点"""
simucount=1
"把querypoint存储到log文件中"
for i in range(100):
    "将反馈次数和querypoint写入log文件"
    teaser=WDNfeedback.FeedBackWorker(ttt[0],ttt[1],ttt[2],ttt[3],ttt[4],ttt[5])#实例化反馈类
#    teaser.updateQuerypointworker(ttt)#更新反馈参数
    "将反馈次数和querypoint写入log文件"
# =============================================================================
#     querypoint=str(ttt)
#     writeStr = "%s : {%s}\n" % (simucount, querypoint)
#     outlogfile.write(writeStr)
# =============================================================================
    teaser.runTest_state(count=20)#仿真
    state=[ttt[0],ttt[1],ttt[2],ttt[3],ttt[4],ttt[5]]
    newdata=teaser.updatetrainningsetworker_state(state,path=newdatapath,count=20,style='value')
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
    newgammer.gpbuilder_state(memoryset.probmemoryunit,fitz=6,label=0)#第一簇高斯过程模型
    newgammer.gpbuilder_state(memoryset.probmemoryunit,fitz=7,label=0)#第一簇概率高斯过程模型
    newgammer.gpbuilder_state(memoryset.probmemoryunit,fitz=6,label=1)#第二簇高斯过程模型
    newgammer.gpbuilder_state(memoryset.probmemoryunit,fitz=7,label=1)#第二簇概率高斯过程模型
    ttt=newgammer.valueUCBhelper_HPP_state(memoryset.probmemoryunit,kappa=0,iternum=100,count=simucount)
    tu=ttt.tolist()
    listaaa.append(tu)  
    simucount=simucount+1#计数，修改文件名称
    
    with open('QP_HPP_D6.txt','a') as f:#记录每次AF选点的参数
        f.write('\n')
        f.write(str(tu)+',')#写入listaaa，querypoint的序列

    
    
    
    
    
    
    
    
    
    
    
    
    
    

