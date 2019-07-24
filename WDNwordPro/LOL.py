# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:44:05 2019

@author: weird
用来测试WDNTagDataHandler,主要是进行模型之间的迭代比较MSE
比较的指标有MSE,R-Squared,JS,KL
"""

import pandas as pd
import numpy as np
import WDNTagDataHandler
import matplotlib.pyplot as plt 
import seaborn as sns
import math
#import iterationreport



"选择路径与文件名，读取TXT文件中迭代的state数据=================================="
datapath='D:/WDNoptimizer/access/LHS6D_prior2/'#LHS先验数据的存放位置
statefilename="D:/WDNoptimizer/access/LHS6D_prior2/priorLHS6D.txt"#存储的先验数据的state列表txt文件
taggedDatafilename='./LabelData/LHS_6D_2.txt'
"-----------------------------------------------------------------------------"
traindatafilename='./LabelData/LHSprior_test.txt'
iterdatafilename='./LabelData/GPR24000_k5_i60_t10.txt'
griddatafilename='./LabelData/2DGMM(16000_8000-36000)_test.txt'
readtestfilename='./LabelData/LHS24000_test.txt'
'-----------------------------------------------------------------------------'
newdatapath='D:/WDNoptimizer/access/access_RF_I100_K0_new/'
savedataname='./LabelData/LHS6D_access_RF_K0_I100.txt'

teaser=WDNTagDataHandler.TaggedDataHandler()#实例化

'读取原始数据进行分类标签保存（先验数据，迭代数据）'
#teaser.PriorDataTagWriter_state(count=20,path=datapath,filename=statefilename,savefilename=taggedDatafilename)#先验数据的处理
#teaser.IterDataTagWriter_state(count_i=10,path=newdatapath,QPlist=RF_D6_I100_K0,savefilename=savedataname)#迭代数据的处理
#teaser.GridDataTagWriter()#主观栅格数据的处理

'MSE=========================================================================='
traindatafilename='./LabelData/LHS_6D_train.txt'
#iterdatafilename='./LabelData/LHS_6D_2.txt'
iterdatafilename='./LabelData/LHS6D_access_HPP_K5_I130.txt'
readtestfilename='./LabelData/LHS_6D_1.txt'


traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
iterdata=teaser.LabelDataReader(filename=iterdatafilename)#迭代数据的读取

traindata=traindata.append(iterdata).reset_index(drop=True)#迭代数据加入先验数据
print(traindata)
MPPMSE=[]
GPRMSE=[]
RFMSE=[]
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
    'MSE'
    MPPMSE1=gamer.MPPMSE(testdata,MPPdata)
    GPRMSE1=gamer.GPRMSE(testdata,GPRdata)
    RFMSE1=gamer.RFMSE(testdata,RFdata)
    MPPMSE.append(MPPMSE1)
    GPRMSE.append(GPRMSE1)
    RFMSE.append(RFMSE1)
"绘图"

#teaser.ComparePrinter(MPPMSE,GPRMSE,RFMSE,style='MSE')
teaser.ZoominPrinter(MPPMSE,GPRMSE,RFMSE,style='MSE')
#print(RFMSE)
#print(MPPMSE)
#print(GPRMSE)
#print(MPPMSE[0],GPRMSE[0])
#print(MPPMSE[1],GPRMSE[1])
#print(MPPMSE[2],GPRMSE[2])
#print(MPPMSE[5],GPRMSE[5])
#print(MPPMSE[11],GPRMSE[11])
#print(MPPMSE[30],GPRMSE[31])
#print(MPPMSE[50],GPRMSE[50])


'R方=========================================================================='
traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
iterdata=teaser.LabelDataReader(filename=iterdatafilename)#迭代数据的读取
traindata=traindata.append(iterdata).reset_index(drop=True)#迭代数据加入先验数据
r_mpp=[]
r_gpr=[]
r_rfr=[]
"循环迭代的数据，对每次迭代的模型进行MSE的计算"
"这里的MSE计算根据模型的不同分别进行，MPP模型中的各簇与仿真值对应的各簇进行计算，GPR中直接进行计算"
'目前的随机森林模型中不存在方差的特征值，需要考虑如何得到'
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
    'MSE'
    MPPMSE1=gamer.MPPMSE(testdata,MPPdata)
    GPRMSE1=gamer.GPRMSE(testdata,GPRdata)
    RFMSE1=gamer.RFMSE(testdata,RFdata)
    'R方'
    r_mpp.append(1-MPPMSE1/(np.var(testdata['value'][testdata['label']==1])+np.var(testdata['value'][testdata['label']==0])))
    r_gpr.append(1-GPRMSE1/(np.var(testdata['value'][testdata['label']==1])+np.var(testdata['value'][testdata['label']==0])))
    r_rfr.append(1-RFMSE1/(np.var(testdata['value'][testdata['label']==1])+np.var(testdata['value'][testdata['label']==0])))
   
"绘图"
teaser.ZoominPrinter(r_mpp,r_gpr,r_rfr,style='R-Squared')
#print(r_mpp)
#print(r_gpr)
#print(r_rfr)



'一次迭代实验中仿真值与预测值的不同模型的对比=========================================='
traindatafilename='./LabelData/LHS_6D_train.txt'
#iterdatafilename='./LabelData/ITER_MPP24000_k5_i30_t10_p1_test.txt'
#iterdatafilename='./LabelData/ITER_MPP24000_k5_i60_t10_p1_test.txt'
#iterdatafilename='./LabelData/HPP24000_k5_i60_t10.txt'
iterdatafilename='./LabelData/LHS6D_access_HPP_K0_I100.txt'
#iterdatafilename='./LabelData/ITER_GMM24000_i60_t10_test.txt'
readtestfilename='./LabelData/LHS_6D_1.txt'
#readtestfilename='./LabelData/24000testset.txt'

'读数据'
traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
iterdata=teaser.LabelDataReader(filename=iterdatafilename)#迭代数据的读取

mpppredictlist=[]
gprpredictlist=[]
rfpredictlist=[]
simulationlist=[]
'基于迭代数据进行迭代建模，得到下一个点的预测值'
for i in range(int(len(iterdata)/2)):
    trainset=traindata.append(iterdata[0:(i*2)]).reset_index(drop=True)
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
    '这里的比较有待商榷额，因为是不同模型，目前是这种方式进行value的比较'
    mppoutput=MPPdata['mean0'][i]*MPPdata['prob0'][i]+MPPdata['mean1'][i]*MPPdata['prob1'][i]
    mpppredictlist.append(mppoutput)
    gproutput=GPRdata['mean'][i]
    gprpredictlist.append(gproutput)
    rfoutput=RFdata['mean0'][i]
    rfpredictlist.append(rfoutput)
    '仿真的数据'
    simuoutput=iterdata['value'][2*i]*iterdata['prob'][2*i]+iterdata['value'][2*i+1]*iterdata['prob'][2*i+1]
    simulationlist.append(simuoutput)


"绘图"
teaser.ValueComparePrinter(mpppredictlist,gprpredictlist,rfpredictlist,simulationlist,style='predict-simulated')



'不同先验数据采样方式得到数据的相同模型的拟合比较结果-----------------------------'
traindatafilename='./LabelData/2Diteration/LHS24000_new.txt'
griddatafilename='./LabelData/2Diteration/2DGMM(16000_8000-36000)_test.txt'
readtestfilename='./LabelData/2Diteration/LHS24000_new.txt'
#readtestfilename='./LabelData/24000testset.txt'
"-----------------------------------------------------------------------------"
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
griddata=teaser.LabelDataReader(filename=griddatafilename)#栅格数据的读取
MPPMSE_LHS=[]
MPPMSE_RG=[]
#traindata=traindata.append(iterdata).reset_index(drop=True)#迭代数据加入先验数据
"循环迭代的数据，对每次迭代的模型进行MSE的计算"
"这里的MSE计算根据模型的不同分别进行，MPP模型中的各簇与仿真值对应的各簇进行计算，GPR中直接进行计算"
for i in range(int(len(traindata)/2)):
    trainsetLHS=traindata[0:(2*(i+1))]
    trainsetRG=griddata[0:(2*(i+1))]
    gamerlhs=WDNTagDataHandler.ModelCompareHandler()
    gamerrg=WDNTagDataHandler.ModelCompareHandler()
    '建模'
    gamerrg.MPPmodelRebuilder(trainsetRG)
    gamerlhs.MPPmodelRebuilder(trainsetLHS)
    '预测'
    MPPdatarg=gamerrg.MPPpredicter(testdata)
    MPPdatalhs=gamerlhs.MPPpredicter(testdata)
    'MSE'
    MPPMSElhs=gamerlhs.MPPMSE(testdata,MPPdatalhs)
    MPPMSErg=gamerrg.MPPMSE(testdata,MPPdatarg)

    MPPMSE_LHS.append(math.log(MPPMSElhs))
    MPPMSE_RG.append(math.log(MPPMSErg))

"绘图"
#sns.set_style("whitegrid")
plt.figure('Line fig',figsize=(27,8))
plt.xlabel('Data Counts',fontsize='xx-large')
plt.ylabel('Log-MSE',fontsize='xx-large')
#plt.title('value ',fontsize='xx-large')

plt.scatter(x=range(len(MPPMSE_LHS)),y=MPPMSE_LHS,marker='.',c='black')
plt.scatter(x=range(len(MPPMSE_RG)),y=MPPMSE_RG,marker='o',c='blue')

plt.plot(MPPMSE_LHS,color='r', linewidth=2, alpha=0.6,label='MSP_LHS')
plt.plot(MPPMSE_RG,color='b', linewidth=2, alpha=0.6,label='MSP_RANDOMGRID')
plt.legend(fontsize='xx-large')





'不同的combination 比较 surrogate Model----------------------------------------'
traindatafilename='./LabelData/LHS_6D_train.txt'

iterdatafilename0='./LabelData/LHS6D_access_HPP_K0_I100.txt'
iterdatafilename1='./LabelData/LHS6D_access_HPP_K1_I100.txt'
iterdatafilename2='./LabelData/LHS6D_access_HPP_K2_I100.txt'
iterdatafilename4='./LabelData/LHS6D_access_HPP_K4_I100.txt'
iterdatafilename5='./LabelData/LHS6D_access_HPP_K5_I130.txt'

gprname0='./LabelData/LHS6D_access_GPR_K0_I100.txt'
gprname2='./LabelData/LHS6D_access_GPR_K2_I100.txt'
gprname5='./LabelData/LHS6D_access_GPR_K5_I100.txt'

rfname0='./LabelData/LHS6D_access_RF_K0_I100.txt'
rfname2='./LabelData/LHS6D_access_RF_K2_I100.txt'
rfname5='./LabelData/LHS6D_access_RF_K5_I100.txt'

#iterdatafilename='./LabelData/ITER_GMM24000_i60_t10_test.txt'
#readtestfilename='./LabelData/LHS24000_test.txt'
#readtestfilename='./LabelData/24000testset.txt'
readtestfilename='./LabelData/LHS_6D_1.txt'

'仿真值与预测值的对比===分簇绘图================================================='
'读数据'
traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取

iterdata0=teaser.LabelDataReader(filename=iterdatafilename0)#迭代HPP0数据的读取
iterdata1=teaser.LabelDataReader(filename=iterdatafilename1)#迭代HPP0数据的读取
iterdata2=teaser.LabelDataReader(filename=iterdatafilename2)#迭代HPP5数据的读取
iterdata4=teaser.LabelDataReader(filename=iterdatafilename4)#迭代HPP0数据的读取
iterdata5=teaser.LabelDataReader(filename=iterdatafilename5)#迭代RFR5数据的读取
gprdata0=teaser.LabelDataReader(filename=gprname0)
gprdata2=teaser.LabelDataReader(filename=gprname2)
gprdata5=teaser.LabelDataReader(filename=gprname5)
rfdata0=teaser.LabelDataReader(filename=rfname0)
rfdata2=teaser.LabelDataReader(filename=rfname2)
rfdata5=teaser.LabelDataReader(filename=rfname5)

HPP0predictlist=[]
HPP1predictlist=[]
HPP2predictlist=[]
HPP4predictlist=[]
HPP5predictlist=[]
gprpre0=[]
gprpre2=[]
gprpre5=[]
rfpre0=[]
rfpre2=[]
rfpre5=[]

simulationlist0=[]
simulationlist1=[]
simulationlist2=[]
simulationlist4=[]
simulationlist5=[]
gprsim0=[]
gprsim2=[]
gprsim5=[]
rfsim0=[]
rfsim2=[]
rfsim5=[]

d0=[]
d1=[]
d2=[]
d4=[]
d5=[]
gprdiff0=[]
gprdiff2=[]
gprdiff5=[]
rfdiff0=[]
rfdiff2=[]
rfdiff5=[]



'基于迭代数据进行迭代建模，得到下一个点的预测值'
for i in range(int(len(iterdata0)/2)):
    trainset0=traindata.append(iterdata0[0:(i*2)]).reset_index(drop=True)
    trainset1=traindata.append(iterdata1[0:(i*2)]).reset_index(drop=True)
    trainset2=traindata.append(iterdata2[0:(i*2)]).reset_index(drop=True)
    trainset4=traindata.append(iterdata4[0:(i*2)]).reset_index(drop=True)
    trainset5=traindata.append(iterdata5[0:(i*2)]).reset_index(drop=True)
    
    gprtrain0=traindata.append(gprdata0[0:(i*2)]).reset_index(drop=True)
    gprtrain2=traindata.append(gprdata2[0:(i*2)]).reset_index(drop=True)
    gprtrain5=traindata.append(gprdata5[0:(i*2)]).reset_index(drop=True)
    rftrain0=traindata.append(rfdata0[0:(i*2)]).reset_index(drop=True)
    rftrain2=traindata.append(rfdata2[0:(i*2)]).reset_index(drop=True)
    rftrain5=traindata.append(rfdata5[0:(i*2)]).reset_index(drop=True)
    
    gamer0=WDNTagDataHandler.ModelCompareHandler()
    gamer1=WDNTagDataHandler.ModelCompareHandler()
    gamer2=WDNTagDataHandler.ModelCompareHandler()
    gamer4=WDNTagDataHandler.ModelCompareHandler()
    gamer5=WDNTagDataHandler.ModelCompareHandler()
    gpr0=WDNTagDataHandler.ModelCompareHandler()
    gpr2=WDNTagDataHandler.ModelCompareHandler()
    gpr5=WDNTagDataHandler.ModelCompareHandler()
    '建模'
    gamer0.MPPmodelRebuilder_state(trainset0)
    gamer0.MPPmodelRebuilder_state(trainset1)
    gamer2.MPPmodelRebuilder_state(trainset2)
    gamer0.MPPmodelRebuilder_state(trainset4)
    gamer5.MPPmodelRebuilder_state(trainset5)
    gpr0.GPRmodelRebuiler_state(gprtrain0)
    gpr2.GPRmodelRebuiler_state(gprtrain2)
    gpr5.GPRmodelRebuiler_state(gprtrain5)
    gamer0.RFmodelRebuilder_state(rftrain0)
    gamer2.RFmodelRebuilder_state(rftrain2)
    gamer5.RFmodelRebuilder_state(rftrain5)
    '预测'
    HPP0data=gamer0.MPPpredicter_state(trainset0)
    HPP1data=gamer0.MPPpredicter_state(trainset1)
    HPP2data=gamer2.MPPpredicter_state(trainset2)
    HPP4data=gamer0.MPPpredicter_state(trainset4)
    HPP5data=gamer5.MPPpredicter_state(trainset5)
    gpr0data=gpr0.GPRpredicter_state(gprtrain0)
    gpr2data=gpr2.GPRpredicter_state(gprtrain2)
    gpr5data=gpr5.GPRpredicter_state(gprtrain5)
    rf0data=gamer0.RFpredicter_state(rftrain0)
    rf2data=gamer2.RFpredicter_state(rftrain2)
    rf5data=gamer5.RFpredicter_state(rftrain5)
#    GPRdata=gamer.GPRpredicter(testdata)
    '这里的比较有待商榷额，因为是不同模型，目前是这种方式进行value的比较'
    'HPP-WDUCB'
    HPP0output=HPP0data['mean0'][i]*HPP0data['prob0'][i]+HPP0data['mean1'][i]*HPP0data['prob1'][i]
    HPP0predictlist.append(HPP0output)
    HPP1output=HPP1data['mean0'][i]*HPP1data['prob0'][i]+HPP1data['mean1'][i]*HPP1data['prob1'][i]
    HPP1predictlist.append(HPP1output)
    HPP2output=HPP2data['mean0'][i]*HPP2data['prob0'][i]+HPP2data['mean1'][i]*HPP2data['prob1'][i]
    HPP2predictlist.append(HPP2output)
    HPP4output=HPP4data['mean0'][i]*HPP4data['prob0'][i]+HPP4data['mean1'][i]*HPP4data['prob1'][i]
    HPP4predictlist.append(HPP4output)
    HPP5output=HPP5data['mean0'][i]*HPP5data['prob0'][i]+HPP5data['mean1'][i]*HPP5data['prob1'][i]
    HPP5predictlist.append(HPP5output)
    'GPR-WDUCB'
    gpr0output=gpr0data['mean'][i]
    gprpre0.append(gpr0output)
    gpr2output=gpr2data['mean'][i]
    gprpre2.append(gpr2output)
    gpr5output=gpr5data['mean'][i]
    gprpre5.append(gpr5output)
    'RF-WDUCB'
    rf0output=(rf0data['mean0'][i]+rf0data['mean1'][i])/2
    rfpre0.append(rf0output)
    rf2output=(rf2data['mean0'][i]+rf2data['mean1'][i])/2
    rfpre2.append(rf2output)
    rf5output=(rf5data['mean0'][i]+rf5data['mean1'][i])/2
    rfpre5.append(rf5output)



#    mpppredictlist0.append(HPPdata['mean0'][i])
#    mpppredictlist1.append(HPPdata['mean1'][i])
#    gproutput=GPRdata['mean'][i]
#    gprpredictlist.append(gproutput)
    '仿真的数据'
    'HPP-WDUCB'
    simuoutput0=iterdata0['value'][2*i]*iterdata0['prob'][2*i]+iterdata0['value'][2*i+1]*iterdata0['prob'][2*i+1]
    simulationlist0.append(simuoutput0)
    simuoutput1=iterdata1['value'][2*i]*iterdata1['prob'][2*i]+iterdata1['value'][2*i+1]*iterdata1['prob'][2*i+1]
    simulationlist1.append(simuoutput1)
    simulation2=iterdata2['value'][2*i]*iterdata2['prob'][2*i]+iterdata2['value'][2*i+1]*iterdata2['prob'][2*i+1]
    simulationlist2.append(simulation2)
    simuoutput4=iterdata4['value'][2*i]*iterdata4['prob'][2*i]+iterdata4['value'][2*i+1]*iterdata4['prob'][2*i+1]
    simulationlist4.append(simuoutput4)
    simulation5=iterdata5['value'][2*i]*iterdata5['prob'][2*i]+iterdata5['value'][2*i+1]*iterdata5['prob'][2*i+1]
    simulationlist5.append(simulation5)
    'GPR-WDUCB'
    gprsimoutput0=gprdata0['value'][2*i]*gprdata0['prob'][2*i]+gprdata0['value'][2*i+1]*gprdata0['prob'][2*i+1]
    gprsim0.append(gprsimoutput0)
    gprsimoutput2=gprdata2['value'][2*i]*gprdata2['prob'][2*i]+gprdata2['value'][2*i+1]*gprdata2['prob'][2*i+1]
    gprsim2.append(gprsimoutput2)
    gprsimoutput5=gprdata5['value'][2*i]*gprdata5['prob'][2*i]+gprdata5['value'][2*i+1]*gprdata5['prob'][2*i+1]
    gprsim5.append(gprsimoutput5)
    'RF-WDUCB'
    rfsimoutput0=rfdata0['value'][2*i]*rfdata0['prob'][2*i]+rfdata0['value'][2*i+1]*rfdata0['prob'][2*i+1]
    rfsim0.append(rfsimoutput0)
    rfsimoutput2=rfdata2['value'][2*i]*rfdata2['prob'][2*i]+rfdata2['value'][2*i+1]*rfdata2['prob'][2*i+1]
    rfsim2.append(rfsimoutput2)
    rfsimoutput5=rfdata5['value'][2*i]*rfdata5['prob'][2*i]+rfdata5['value'][2*i+1]*rfdata5['prob'][2*i+1]
    rfsim5.append(rfsimoutput5)    
    '仿真与预测的差值difference'
    'HPP-WDUCB'
    difference_hpp0=abs(HPP0output-simuoutput0)
    d0.append(difference_hpp0)
    difference_hpp1=abs(HPP1output-simuoutput1)
    d1.append(difference_hpp1)
    difference_hpp2=abs(HPP2output-simulation2)
    d2.append(difference_hpp2)
    difference_hpp4=abs(HPP4output-simuoutput4)
    d4.append(difference_hpp4)
    difference_hpp5=abs(HPP5output-simulation5)
    d5.append(difference_hpp5)
    'GPR-WDUCB'
    a0=abs(gpr0output-gprsimoutput0)
    gprdiff0.append(a0)
    a2=abs(gpr2output-gprsimoutput2)
    gprdiff2.append(a2)
    a5=abs(gpr5output-gprsimoutput5)
    gprdiff5.append(a5)    
    'RF-WDUCB'
    b0=abs(rf0output-rfsimoutput0)
    rfdiff0.append(b0)
    b2=abs(rf2output-rfsimoutput2)
    rfdiff2.append(b2)
    b5=abs(rf5output-rfsimoutput5)
    rfdiff5.append(b5)   

"绘图"

#sns.set_style("whitegrid")
plt.figure('Line fig',figsize=(20,3))
plt.xlabel('Iteration Times',fontsize='xx-large')
plt.ylabel('prediction-simulation',fontsize='xx-large')
plt.title('value ',fontsize='xx-large')
#plt.scatter(x=range(len(HPP0predictlist)),y=HPP0predictlist,marker='.',c='r',label='HPP0')
plt.scatter(x=range(len(HPP1predictlist)),y=HPP1predictlist,marker='o',c='r',label='pre_HPP')
#plt.scatter(x=range(len(HPP2predictlist)),y=HPP2predictlist,marker='o',c='r',label='HPP2')
#plt.scatter(x=range(len(HPP4predictlist)),y=HPP4predictlist,marker='o',c='r',label='HPP4')
#plt.scatter(x=range(len(HPP5predictlist)),y=HPP5predictlist,marker='*',c='blue',label='HPP5')
#plt.scatter(x=range(len(gprpre0)),y=gprpre0,marker='o',c='b',label='GPR0')
plt.scatter(x=range(len(gprpre2)),y=gprpre2,marker='o',c='b',label='pre_GPR')
#plt.scatter(x=range(len(gprpre5)),y=gprpre5,marker='o',c='black',label='GPR5')
#plt.scatter(x=range(len(rfpre2)),y=rfpre0,marker='o',c='black',label='RF0')
plt.scatter(x=range(len(rfpre2)),y=rfpre2,marker='o',c='black',label='pre_RF')
#plt.scatter(x=range(len(rfpre5)),y=rfpre5,marker='o',c='black',label='RF5')


#plt.plot(simulationlist0,color='r', linewidth=1, linestyle=':',alpha=0.6,label='sim_HPP_K0')
#plt.plot(simulationlist1,color='g', linewidth=1, linestyle=':',alpha=0.6,label='sim_HPP_k1')
plt.plot(simulationlist2,color='r', linewidth=1, linestyle=':',alpha=0.6,label='sim_HPP_WDUCB')
#plt.plot(simulationlist4,color='b', linewidth=1, linestyle=':',alpha=0.6,label='sim_HPP_k4')
#plt.plot(simulationlist5,color='w', linewidth=1, linestyle=':',alpha=0.6,label='sim_HPP_k5')
#plt.plot(gprsim0,color='b', linewidth=1, linestyle=':',alpha=0.6,label='sim_GPR_k0')
plt.plot(gprsim2,color='b', linewidth=1, linestyle=':',alpha=0.6,label='sim_GPR_UCB')
#plt.plot(gprsim5,color='b', linewidth=1, linestyle=':',alpha=0.6,label='sim_GPR_k5')
#plt.plot(rfsim0,color='g', linewidth=1, linestyle=':',alpha=0.6,label='sim_RF_k0')
plt.plot(rfsim2,color='black', linewidth=1, linestyle=':',alpha=0.6,label='sim_RF_EI')
#plt.plot(rfsim5,color='b', linewidth=1, linestyle=':',alpha=0.6,label='sim_RF_k5')

#plt.plot(d0,color='r', linewidth=2, alpha=0.6,label='diff_HPP_k0')
#plt.plot(d1,color='y', linewidth=2, alpha=0.6,label='diff_HPP_k1')
plt.plot(d2,color='r', linewidth=2, alpha=0.6,label='diff_HPP')
#plt.plot(d4,color='gray', linewidth=2, alpha=0.6,label='diff_HPP_k4')
#plt.plot(d5,color='b', linewidth=2, alpha=0.6,label='diff_HPP_k5')
#plt.plot(gprdiff0,color='b', linewidth=2, alpha=0.6,label='diff_GPR_k0')
plt.plot(gprdiff2,color='b', linewidth=2, alpha=0.6,label='diff_GPR')
#plt.plot(gprdiff5,color='r', linewidth=2, alpha=0.6,label='diff_GPR_k5')
#plt.plot(rfdiff0,color='r', linewidth=2, alpha=0.6,label='diff_RF_k0')
plt.plot(rfdiff2,color='black', linewidth=2, alpha=0.6,label='diff_RF')
#plt.plot(rfdiff5,color='b', linewidth=2, alpha=0.6,label='diff_RF_k5')

plt.legend(fontsize='large')


'diiference feature'

KLM0=[]
KLM1=[]
KLM2=[]
KLM4=[]
KLM5=[]
KLvar0=[]
KLvar1=[]
KLvar2=[]
KLvar4=[]
KLvar5=[]

gprm0=[]
gprm2=[]
gprm5=[]
gprvar0=[]
gprvar2=[]
gprvar5=[]

rfm0=[]
rfm2=[]
rfm5=[]
rfvar0=[]
rfvar2=[]
rfvar5=[]

for j in range(len(d0)):
#    k=3
#    kld0=np.array(d0[j*k:(j+1)*k])
#    kld1=np.array(d1[j*k:(j+1)*k])
#    kld2=np.array(d2[j*k:(j+1)*k])
#    kld4=np.array(d1[j*k:(j+1)*k])
#    kld5=np.array(d5[j*k:(j+1)*k])
#    a0=np.array(gprdiff0[j*k:(j+1)*k])
#    a2=np.array(gprdiff2[j*k:(j+1)*k])
#    a5=np.array(gprdiff5[j*k:(j+1)*k])
#    b0=np.array(rfdiff0[j*k:(j+1)*k])
#    b2=np.array(rfdiff2[j*k:(j+1)*k])
#    b5=np.array(rfdiff5[j*k:(j+1)*k])

    kld0=np.array(d0[0:(j+1)])
    kld1=np.array(d1[0:(j+1)])
    kld2=np.array(d2[0:(j+1)])
    kld4=np.array(d4[0:(j+1)])
    kld5=np.array(d5[0:(j+1)])
    a0=np.array(gprdiff0[0:(j+1)])
    a2=np.array(gprdiff2[0:(j+1)])
    a5=np.array(gprdiff5[0:(j+1)])
    b0=np.array(rfdiff0[0:(j+1)])
    b2=np.array(rfdiff2[0:(j+1)])
    b5=np.array(rfdiff5[0:(j+1)])
    
#    mean0=kld0.sum()/(k)
#    var0=(kld0*kld0).sum()/(k)-mean0**2
#    mean1=kld1.sum()/(k)
#    var1=(kld1*kld1).sum()/(k)-mean1**2 
#    mean2=kld2.sum()/(k)
#    var2=(kld2*kld2).sum()/(k)-mean2**2 
#    mean4=kld4.sum()/(k)
#    var4=(kld4*kld4).sum()/(k)-mean4**2    
#    mean5=kld5.sum()/(k)
#    var5=(kld5*kld5).sum()/(k)-mean5**2    
#    m0=a0.sum()/(k)
#    v0=(a0*a0).sum()/(k)-m0**2
#    m2=a2.sum()/(k)
#    v2=(a2*a2).sum()/(k)-m2**2
#    m5=a5.sum()/(k)
#    v5=(a5*a5).sum()/(k)-m5**2
#    rm0=b0.sum()/(k)
#    rv0=(b0*b0).sum()/(k)-m0**2
#    rm2=b2.sum()/(k)
#    rv2=(b2*b2).sum()/(k)-m2**2
#    rm5=b5.sum()/(k)
#    rv5=(b5*b5).sum()/(k)-m5**2

    mean0=kld0.sum()/(j+1)
    var0=(kld0*kld0).sum()/(j+1)-mean0**2
    mean1=kld1.sum()/(j+1)
    var1=(kld1*kld1).sum()/(j+1)-mean1**2 
    mean2=kld2.sum()/(j+1)
    var2=(kld2*kld2).sum()/(j+1)-mean2**2 
    mean4=kld4.sum()/(j+1)
    var4=(kld4*kld4).sum()/(j+1)-mean4**2    
    mean5=kld5.sum()/(j+1)
    var5=(kld5*kld5).sum()/(j+1)-mean5**2    
    m0=a0.sum()/(j+1)
    v0=(a0*a0).sum()/(j+1)-m0**2
    m2=a2.sum()/(j+1)
    v2=(a2*a2).sum()/(j+1)-m2**2
    m5=a5.sum()/(j+1)
    v5=(a5*a5).sum()/(j+1)-m5**2
    rm0=b0.sum()/(j+1)
    rv0=(b0*b0).sum()/(j+1)-m0**2
    rm2=b2.sum()/(j+1)
    rv2=(b2*b2).sum()/(j+1)-m2**2
    rm5=b5.sum()/(j+1)
    rv5=(b5*b5).sum()/(j+1)-m5**2
    
    KLM0.append(mean0)
    KLM1.append(mean1)
    KLM2.append(mean2)
    KLM4.append(mean4)
    KLM5.append(mean5)
    KLvar0.append(var0)
    KLvar1.append(var1)
    KLvar2.append(var2)
    KLvar4.append(var4)
    KLvar5.append(var5)
    gprm0.append(m0)
    gprm2.append(m2)
    gprm5.append(m5)
    gprvar0.append(v0)
    gprvar2.append(v2)
    gprvar5.append(v5)
    rfm0.append(rm0)
    rfm2.append(rm2)
    rfm5.append(rm5)
    rfvar0.append(rv0)
    rfvar2.append(rv2)
    rfvar5.append(rv5)   
    





"绝对差值之间均值方差绘图"
#sns.set_style("whitegrid")
plt.figure('Line fig',figsize=(8,6))

plt.xlabel('Iteration Count',fontsize='xx-large')
plt.ylabel('Difference',fontsize='xx-large')
plt.title('the difference features of conbinations',fontsize='xx-large')
#plt.plot(KLM0,color='r', linewidth=3, alpha=0.6,label='diff-HPP-WDUCB-K0-mean')
#plt.plot(KLM1,color='r', linewidth=2.5, alpha=0.6,label='diff-HPP-WDUCB-K1-mean')
#plt.plot(KLM2,color='b', linewidth=3, alpha=0.6,label='diff-HPP-WDUCB-K2-mean')
#plt.plot(KLM4,color='k', linewidth=2.5, alpha=0.6,label='diff-HPP-WDUCB-K4-mean')
#plt.plot(KLM5,color='orange', linewidth=3, alpha=0.6,label='diff-HPP-WDUCB-K5-mean')
plt.plot(gprm0,color='m', linewidth=2.5, alpha=0.6,label='diff-GPR-UCB-K0-mean')
plt.plot(gprm2,color='b', linewidth=2.5, alpha=0.6,label='diff-GPR-UCB-K2-mean')
plt.plot(gprm5,color='orange', linewidth=2.5, alpha=0.6,label='diff-GPR-UCB-K5-mean')
#plt.plot(rfm0,color='yellow', linewidth=1, alpha=0.6,label='diff-RF-EI-mean')
#plt.plot(rfm2,color='g', linewidth=1, alpha=0.6,label='diff-RF-EI-mean')
#plt.plot(rfm5,color='r', linewidth=1, alpha=0.6,label='diff-RF-PI-mean')


#plt.plot(KLvar0,color='r', linewidth=2.5, alpha=0.6,label='diff-HPP0-var')
#plt.plot(KLvar1,color='y', linewidth=2.5, alpha=0.6,label='diff-HPP1-var')
#plt.plot(KLvar2,color='g', linewidth=2.5, alpha=0.6,label='diff-HPP2-var')
#plt.plot(KLvar4,color='b', linewidth=2.5, alpha=0.6,label='diff-HPP4-var')
#plt.plot(KLvar5,color='gray', linewidth=2.5, alpha=0.6,label='diff-HPP5-var')
#plt.plot(gprvar0,color='black', linewidth=1, alpha=0.6,label='diff-GPR0-var')
#plt.plot(gprvar2,color='b', linewidth=1, alpha=0.6,label='diff-GPR2-var')
#plt.plot(gprvar5,color='black', linewidth=1, alpha=0.6,label='diff-GPR5-var')
#plt.plot(rfvar0,color='black', linewidth=1, alpha=0.6,label='diff-RF2-var')
#plt.plot(rfvar2,color='black', linewidth=1, alpha=0.6,label='diff-RF2-var')
#plt.plot(rfvar5,color='r', linewidth=1, alpha=0.6,label='diff-RF5-var')
plt.legend(fontsize='xx-large')




'柱状图'
KLM0[97]

plt.figure('Line fig',figsize=(8,6))
list_00 = [KLM0[2], KLM0[27], KLM0[50], KLM0[75],KLM0[97]]
list_01 = [KLM1[2], KLM1[27], KLM1[50], KLM1[75],KLM1[97]]
list_02 = [KLM2[2], KLM2[27], KLM2[50], KLM2[75],KLM2[97]]
list_04 = [KLM4[2], KLM4[27], KLM4[50], KLM4[75],KLM4[97]]
list_05 = [KLM5[2], KLM5[27], KLM5[50], KLM5[75],KLM5[97]]


list_00 = [gprm0[2], gprm0[25], gprm0[51], gprm0[75],gprm0[97]]
#list_01 = [KLM1[2], KLM1[27], KLM1[50], KLM1[75],KLM1[97]]
list_02 = [gprm2[2], gprm2[25], gprm2[51], gprm2[75],gprm2[97]]
#list_04 = [KLM4[2], KLM4[27], KLM4[50], KLM4[75],KLM4[97]]
list_05 = [gprm5[2], gprm5[25], gprm5[51], gprm5[75],gprm5[97]]



plt.title('the difference of K ',fontsize='xx-large')
plt.xlabel('Iteration Count',fontsize='xx-large')
plt.ylabel('difference',fontsize='xx-large')
name_list = ['2', '25', '50', '75','100']
x = list(range(len(name_list)))
total_width, n = 0.8, 3
width = total_width / n
plt.bar(x, list_00, width=width, label='k=0', tick_label=name_list, fc='m')
for i in range(len(x)):
	x[i] = x[i] + width
#plt.bar(x, list_01, width=width, label='k=1', fc='r')
#for i in range(len(x)):
#	x[i] = x[i] + width
plt.bar(x, list_02, width=width, label='k=2', fc='b')
for i in range(len(x)):
	x[i] = x[i] + width
#plt.bar(x, list_04, width=width, label='k=4', fc='k')
#for i in range(len(x)):
#	x[i] = x[i] + width
plt.bar(x, list_05, width=width, label='k=5', fc='orange')
plt.legend(fontsize='large')

































'仿真值与预测值的对比===旧数据集================================================='
'读数据'
traindatafilename='./LabelData/2Diteration/LHSprior_test.txt'
iterdatafilename1='./LabelData/2Diteration/HPP24000_k5_i60_t10.txt'
iterdatafilename2='./LabelData/2Diteration/GPR24000_k5_i60_t10.txt'
readtestfilename='./LabelData/2Diteration/LHS24000_new.txt'

traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
iterdata1=teaser.LabelDataReader(filename=iterdatafilename1)#迭代HPP数据的读取
iterdata2=teaser.LabelDataReader(filename=iterdatafilename2)#迭代GPR数据的读取

mpppredictlist=[]
gprpredictlist=[]
simulationlist1=[]
simulationlist2=[]
d1=[]
d2=[]
'基于迭代数据进行迭代建模，得到下一个点的预测值'
for i in range(int(len(iterdata1)/2)):
    trainset1=traindata.append(iterdata1[0:(i*2)]).reset_index(drop=True)
    trainset2=traindata.append(iterdata2[0:(i*2)]).reset_index(drop=True)
    gamer=WDNTagDataHandler.ModelCompareHandler()
    '建模'
    gamer.MPPmodelRebuilder(trainset1)
    gamer.GPRmodelRebuiler(trainset2)
    '预测'
    HPPdata=gamer.MPPpredicter(testdata)
    GPRdata=gamer.GPRpredicter(testdata)
    '这里的比较有待商榷额，因为是不同模型，目前是这种方式进行value的比较'
    mppoutput=HPPdata['mean0'][i]*HPPdata['prob0'][i]+HPPdata['mean1'][i]*HPPdata['prob1'][i]
    mpppredictlist.append(mppoutput)
#    mpppredictlist0.append(HPPdata['mean0'][i])
#    mpppredictlist1.append(HPPdata['mean1'][i])
    gproutput=GPRdata['mean'][i]
    gprpredictlist.append(gproutput)
    '仿真的数据'
    simuoutput1=iterdata1['value'][2*i]*iterdata1['prob'][2*i]+iterdata1['value'][2*i+1]*iterdata1['value'][2*i+1]
    simulationlist1.append(simuoutput1)
    simulation2=iterdata2['value'][2*i]*iterdata2['prob'][2*i]+iterdata2['value'][2*i+1]*iterdata2['value'][2*i+1]
    simulationlist2.append(simulation2)
    '绝对差值'
    difference_hpp=abs(mppoutput-simuoutput1)
    d1.append(difference_hpp)
    difference_gpr=abs(gproutput-simulation2)
    d2.append(difference_gpr)


"绘图"


plt.figure('Line fig',figsize=(27,8))
plt.xlabel('Iterations',fontsize='xx-large')
plt.ylabel('Prediction-Simulation',fontsize='xx-large')
#plt.title('value ',fontsize='xx-large')
plt.scatter(x=range(len(gprpredictlist)),y=gprpredictlist,marker='.',c='black',label='GPR')
plt.scatter(x=range(len(mpppredictlist)),y=mpppredictlist,marker='o',c='red',label='HPP')


plt.plot(simulationlist1,color='r', linewidth=2, linestyle=':',alpha=0.6,label='sim_HPP')
plt.plot(simulationlist2,color='black', linewidth=2, linestyle=':',alpha=0.6,label='sim_GP')
plt.plot(d1,color='r', linewidth=2, alpha=0.6,label='diff_HPP')
plt.plot(d2,color='black', linewidth=2, alpha=0.6,label='diff_GP')
plt.legend(fontsize='xx-large')









# =============================================================================
# '仿真值与预测值的对比分簇绘图========没啥意义 占时不需要==============='
# 
# '读数据'
# traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
# testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
# iterdata=teaser.LabelDataReader(filename=iterdatafilename)#迭代数据的读取
# mpppredictlist0=[]
# mpppredictlist1=[]
# gprpredictlist=[]
# rfpredictlist=[]
# simulationlist0=[]
# simulationlist1=[]
# '基于迭代数据进行迭代建模，得到下一个点的预测值'
# for i in range(int(len(iterdata)/2)):
#     trainset=traindata.append(iterdata[0:(i*2)]).reset_index(drop=True)
#     print(trainset)
#     gamer=WDNTagDataHandler.ModelCompareHandler()
#     '建模'
#     gamer.MPPmodelRebuilder(trainset)
#     gamer.GPRmodelRebuiler(trainset)
#     gamer.RFmodelRebuilder(trainset)
#     '预测'
#     MPPdata=gamer.MPPpredicter(testdata)
#     GPRdata=gamer.GPRpredicter(testdata)
#     RFdata=gamer.RFpredicter(testdata)
#     '这里的比较有待商榷额，因为是不同模型，目前是这种方式进行value的比较'
# #    mppoutput0=MPPdata['mean0'][i]*MPPdata['prob0'][i]+MPPdata['mean1'][i]*MPPdata['prob1'][i]
# 
#     mpppredictlist0.append(MPPdata['mean0'][i])
#     mpppredictlist1.append(MPPdata['mean1'][i])
#     gproutput=GPRdata['mean'][i]
#     gprpredictlist.append(gproutput)
#     rfoutput=RFdata['mean0'][i]
#     rfpredictlist.append(rfoutput)
#     '仿真的数据'
# #    simuoutput=iterdata['value'][2*i]*iterdata['prob'][2*i]+iterdata['value'][2*i+1]*iterdata['value'][2*i+1]
#     simulationlist0.append(iterdata['value'][2*i])
#     simulationlist1.append(iterdata['value'][2*i+1])
# 
# "绘图"
# 
# sns.set_style("whitegrid")
# plt.figure('Line fig',figsize=(20,6))
# plt.xlabel('Iteration Times')
# plt.ylabel('predict-simulated')
# plt.title('value ',fontsize='xx-large')
# 
# plt.scatter(x=range(len(gprpredictlist)),y=gprpredictlist,marker='.',c='black')
# plt.scatter(x=range(len(rfpredictlist)),y=rfpredictlist,marker='o',c='blue')
# 
# 
# plt.plot(mpppredictlist0,color='r', linewidth=2, alpha=0.6,label='MPP0')
# plt.plot(mpppredictlist1,color='r', linewidth=2, alpha=0.6,label='MPP1')
# plt.plot(gprpredictlist,color='black', linewidth=2, alpha=0.6,label='GPR')
# plt.plot(rfpredictlist,color='blue', linewidth=2, alpha=0.6,label='RF')
# plt.plot(simulationlist0,color='green', linewidth=2, alpha=0.6,label='simulate0')
# plt.plot(simulationlist1,color='green', linewidth=2, alpha=0.6,label='simulate1')
# plt.legend(fontsize='x-large')
# =============================================================================





































