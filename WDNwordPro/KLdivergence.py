# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:08:08 2019

@author: weird
"""
import pandas as pd
import numpy as np
import scipy.stats
import WDNTagDataHandler
import matplotlib.pyplot as plt 
import seaborn as sns
import WDNoptimizer
from sklearn.metrics import mean_squared_error
import math
import WDNexataReader#读取数据

teaser=WDNTagDataHandler.TaggedDataHandler()#实例化
traindatafilename='./LabelData/LHS_6D_train.txt'
iterdatafilename='./LabelData/LHS_6D_2.txt'
readtestfilename='./LabelData/LHS_6D_1.txt'
simulationrawdataname='./LabelData/LHS6D_raw.txt'

testdatapath="D:/WDNoptimizer/access/LHS6D_prior2/"
teststate="D:/WDNoptimizer/access/LHS6D_prior2/priorLHS6D.txt"

traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
iterdata=teaser.LabelDataReader(filename=iterdatafilename)#迭代数据的读取

rawdata=teaser.RawDataReader(filename=simulationrawdataname)

#traindata=traindata.append(iterdata).reset_index(drop=True)#迭代数据加入先验数据

MPPMSE=[]
GPRMSE=[]
RFMSE=[]
KLd=[]
"循环迭代的数据，对每次迭代的模型进行MSE的计算"
"这里的MSE计算根据模型的不同分别进行，MPP模型中的各簇与仿真值对应的各簇进行计算，GPR中直接进行计算"
for i in range(int(len(traindata)/2)):
    trainset=traindata[0:(2*(i+1))]
    print(trainset)
    gamer=WDNTagDataHandler.ModelCompareHandler()
    '建模'
    gamer.MPPmodelRebuilder_state(trainset)
    gamer.GPRmodelRebuiler_state(trainset)
    gamer.RFmodelRebuilder_state(trainset)
    '预测'
    MPPdata=gamer.MPPpredicter_state(testdata)
    GPRdata=gamer.GPRpredicter_state(testdata)
    RFdata=gamer.RFpredicter_state(testdata)
    "KL散度"
    aaa=20*MPPdata['prob0']
    aaa=[math.ceil(x) for x in aaa]
    ccc=[20-x for x in aaa]
    kl=0
    for n in range(120):
        first=rawdata['value'][n*20:(n+1)*20]
        second=np.array(first)
        second[np.isnan(second)]=np.nanmean(second)
        second=np.sort(second)
        second=second/np.sum(second)
        bbb0= np.random.normal(loc=MPPdata['mean0'][n],scale=MPPdata['std0'][n],size=aaa[n])
        bbb1=np.random.normal(loc=MPPdata['mean1'][n],scale=MPPdata['std1'][n],size=ccc[n])
        bbb=np.append(bbb1,bbb0)
        bbb=bbb/np.sum(bbb)
        kl1=scipy.stats.entropy(bbb, second) 
#        kl1=np.sum(bbb*np.log(bbb/second))
        kl=kl+kl1
    KLd.append(kl)
    



gamer=WDNTagDataHandler.ModelCompareHandler()
'建模'
gamer.MPPmodelRebuilder_state(traindata)
gamer.GPRmodelRebuiler_state(traindata)
gamer.RFmodelRebuilder_state(traindata)
'预测'
MPPdata=gamer.MPPpredicter_state(testdata)
GPRdata=gamer.GPRpredicter_state(testdata)
RFdata=gamer.RFpredicter_state(testdata)



aaa=20*MPPdata['prob0']
aaa=[math.ceil(x) for x in aaa]
ccc=[20-x for x in aaa]
kl=0
for n in range(120):
    first=rawdata['value'][n*20:(n+1)*20]
    second=np.array(first)
    second[np.isnan(second)]=np.nanmean(second)
    second=np.sort(second)
    second=second/np.sum(second)
    bbb0= np.random.normal(loc=MPPdata['mean0'][n],scale=MPPdata['std0'][n],size=aaa[n])
    bbb1=np.random.normal(loc=MPPdata['mean1'][n],scale=MPPdata['std1'][n],size=ccc[n])
    bbb=np.append(bbb1,bbb0)
    bbb=bbb/np.sum(bbb)
    ddd=bbb/second
#    ddd=[x+0.0000001 for x in ddd]
    kl1=scipy.stats.entropy(bbb, second) 
#    kl1=np.sum(bbb*np.log(ddd))
    kl=kl+kl1
KLd.append(kl)













 
p=np.asarray([0.65,0.25,0.07,0.03])
p
q=np.array([0.6,0.25,0.1,0.05])
q
#方法一：根据公式求解
kl1=np.sum(bbb*np.log(bbb/second))
kl1


# =============================================================================
# """
# 读取原始先验数据的STATE文件，得到每个原始数据的value值
# """
# z=[]#序号
# superappinterval=[]#superapp视频业务，需要的时延抖动小，吞吐量大
# superappsize=[]
# vbrinterval=[]#vbr其他义务
# vbrsize=[]
# trafinterval=[]#trafficgenerator图像流，需要的丢包率小，吞吐量大
# trafsize=[]
# memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态
# iternum=0#迭代的记数，在读取先验数据时记为零
# "选择路径与文件名，读取TXT文件中迭代的state数据=============================================="
# datapath=testdatapath
# "-----------------------------------------------------------------------------------------"
# statefilename=teststate
# with open(statefilename, 'r') as file_to_read:
#     while True:
#         lines = file_to_read.readline() # 整行读取数据
#         if not lines:
#             "这里还有点数据格式上的问题，目前是这样处理，需要对生成的state文件进行处理"
#             break
#             pass
#         z,si_tmp,ss_tmp,vi_tmp,vs_tmp,ti_tmp,ts_tmp = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
#         superappinterval.append(math.ceil(si_tmp))
#         superappsize.append(math.ceil(ss_tmp))
#         vbrinterval.append(math.ceil(vi_tmp))
#         vbrsize.append(math.ceil(vs_tmp))
#         trafinterval.append(math.ceil(ti_tmp))
#         trafsize.append(math.ceil(ts_tmp))
#         pass
#     pass
# "读取训练数据集================================================================"
# """用来读取原始数据集，得到memoryset.probmemoryunit，绘制聚类图，拟合的GMM热力图"""
# for count_i in range(len(superappsize)):
#     tempmemoryset=WDNoptimizer.MemoryUnit()
#     state=[superappinterval[count_i],superappsize[count_i],vbrinterval[count_i],vbrsize[count_i],trafinterval[count_i],trafsize[count_i]]
#     for i in range(20):
#         """
#         读取数据，对数据进行分类处理
#         """
#         dataset = 'radio'+str(superappinterval[count_i])+"_"+str(superappsize[count_i])+"_"+str(vbrinterval[count_i])+"_"+str(vbrsize[count_i])+"_"+str(trafinterval[count_i])+"_"+str(trafsize[count_i])+'_'+str(i)
#         print(dataset)
#         readdb=WDNexataReader.ExataDBreader()#实例化
#         readdb.opendataset(dataset,datapath)#读取特定路径下的数据库
#         readdb.appnamereader()#读取业务层的业务名称
#         readdb.appfilter()#将业务名称分类至三个list
#         readdb.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
#         readdb.inputparainsert(superappinterval[count_i],superappsize[count_i],vbrinterval[count_i],vbrsize[count_i],trafinterval[count_i],trafsize[count_i])
#         #将每条流的业务设计参数加入类中的字典
#         "======================以上步骤不可省略，下方处理可以根据需求修改"
#         """
#         评估部分，对于三种不同的业务有不同的权重:时延、抖动、丢包率、吞吐量
#         vbr:        [1,2,3,4] delay,jitter,messagecompleterate,throughtput
#         trafficgen: [5,6,7,8]
#         superapp:   [9,10,11,12]
#         vbr,superapp,trafficgen
#         """
#         eva=WDNoptimizer.EvaluationUnit()
#         vbr=readdb.meandata('vbr')
#         eva.calculateMetricEvaValue(vbr)
#         trafficgen=readdb.meandata('trafficgen')
#         eva.calculateMetricEvaValue(trafficgen)
#         superapp=readdb.meandata('superapp')
#         eva.calculateMetricEvaValue(superapp)
#         value=eva.evaluationvalue()
#         """
#         状态动作保存：当前状态、评估值、动作、收益(目前的动作和收益没有用暂时放这里)
#         如果是第一次仿真，动作与收益为缺省值null
#         记忆单元的数据包括：
#             1）当前状态：6个输入量 三个业务的包大小，发包时间间隔
#             2）评估值：value=eva.evaluationvalue()
#         "tempmemoryset用来进行读取数据是的聚类"
#         """
#         tempmemoryset.valueinserter(state=state,value=value)
#         memoryset.valueinserter(state,value)                          
#     iternum=iternum+1#聚类绘图需要，如不需要绘图则不需要这个参数
# print(memoryset.memoryunit)
# savefilename='./LabelData/LHS6D_raw2.txt'
# with open(savefilename,'w') as f:#记录每次AF选点的参数
#     f.write('\n')
#     f.write(str(memoryset.memoryunit))#写入标记过的数据的DataFrame
# print('mission completed ! ! !')                    
# =============================================================================
















































