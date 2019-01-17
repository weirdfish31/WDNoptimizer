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
import numpy as np


"读数据================================================"
z=[]#序号
superappinterval=[20]#superapp视频业务，需要的时延抖动小，吞吐量大
superappsize=[]
vbrinterval=[30]#vbr其他义务
#vbrsize=[50000]
vbrsize=[24000]
trafinterval=[30]#trafficgenerator图像流，需要的丢包率小，吞吐量大
trafsize=[]
listaaa=[]#用于存储querypoint的序列
appdata=pd.DataFrame()#所有数据库的某种业务的聚合（三种业务分开处理）
memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
valuegmmgamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#实例化GMM模型

figpath="./Figure/"#图像的存放位置
#datapath='E:/WDNoptimizer/LHSprior/'#LHS先验数据的存放位置
newdatapath='./OutConfigfile/'#新产生的数据的存放位置
iternum=0#迭代的记数，在读取先验数据时记为零

"选择路径与文件名，读取TXT文件中迭代的state数据=============================================="
#datapath='G:/testData/2DGMM(16000_8000-36000)/'#先验数据的存放位置
#datapath='E:/WDNoptimizer/LHSMSE50000/'#LHS先验数据的存放位置
datapath='E:/WDNoptimizer/LHSMSE24000/'#LHS先验数据的存放位置
"-----------------------------------------------------------------------------------------"
#statefilename="./history/priorstate50000_all.txt"#存储的先验数据的state列表txt文件
#statefilename="./priorstate_test.txt"#存储的先验数据的state列表txt文件
statefilename="./history/priorstate24000_all.txt"#存储的先验数据的state列表txt文件


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
                        print(str(superappsize[count_i]))
                        dataset='radio REQUEST-SIZE DET '+str(superappsize[count_i])+' _ '+str(vbrs_i)+' _ RND DET '+str(trafsize[count_i])+' _'+str(i)
                        readdb=WDNexataReader.ExataDBreader()#实例化
                        readdb.opendataset(dataset,datapath)#读取特定路径下的数据库
                        readdb.appnamereader()#读取业务层的业务名称
                        readdb.appfilter()#将业务名称分类至三个list
                        readdb.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
                        readdb.inputparainsert(sappi_i,superappsize[count_i],vbri_i,vbrs_i,trafi_i,trafsize[count_i])
                        #将每条流的业务设计参数加入类中的字典
                        print(superappsize[count_i],vbrs_i,trafsize[count_i])
                        #===================================================以上步骤不可省略，下方处理可以根据需求修改
                        """
                        评估部分，对于三种不同的业务有不同的权重:时延、抖动、丢包率、吞吐量
                        vbr:        [1,2,3,4]
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
                        状态动作保存：当前状态、评估值、动作、收益
                        如果是第一次仿真，动作与收益为缺省值null
                        记忆单元的数据包括：
                            1）当前状态：6个输入量 三个业务的包大小，发包时间间隔
                            2）评估值：value=eva.evaluationvalue()
                        "memoryset用来保存原始数据信息" 
                        "tempmemoryset用来进行读取数据是的聚类"
                        """
                        state=[sappi_i,superappsize[count_i],vbri_i,vbrs_i,trafi_i,trafsize[count_i]]
                        qos=eva.qoslist
                        print(qos)
                        memoryset.valueinserter(state=state,value=value)
                        memoryset.qosinserter(state=state,qos=qos)
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
                        
print(memoryset.memoryunit)
print(memoryset.qosmemoryunit)
priordataset=memoryset.memoryunit#将原始的数据保存到内存中
print(memoryset.probmemoryunit)#这个数据是value均值、分簇概率，标签的综合数据，下面将利用这个数据进行GMM建模
print(priordataset)#原始数据包括state，value
import seaborn as sns 
sns.jointplot('sapps','value',data=memoryset.probmemoryunit, kind='kde')        
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

















