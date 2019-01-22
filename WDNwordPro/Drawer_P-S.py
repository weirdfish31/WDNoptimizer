# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:18:54 2019

@author: WDN
2cluster 3time GPR AF query point list
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import WDNexataReader#读取数据
import WDNoptimizer#建模，画图，AF函数
import WDNfeedback#反馈

"实验1：2簇GPR模型60次迭代，每次3次仿真"
"""
比较的是GP模型的迭代，仿真值与预测值之间的关系，未完成，需要修改

"""
figpath="./Figure/"
GMMpath="D:/WDNoptimizer/GMM_i30/"
MPPpath="D:/WDNoptimizer/MPP_k5_i30_p1/"
xmin,xmax=0,65000
ymin,ymax=0,65000
xset,yset=np.meshgrid(np.arange(xmin,xmax, 500), np.arange(ymin,ymax, 500))

listaaa=[[63988 ,63060 ],[63979 ,47    ],[482   ,63975 ],[63986 ,33949 ],[538   ,63993 ],[36558 ,63960 ],
         [36792 ,63937 ],[48721 ,41062 ],[45332 ,43212 ],[63967 ,32802 ],[39670 ,48671 ],[34465 ,63962 ],
         [43896 ,42928 ],[63796 ,188   ],[33883 ,63965 ],[47457 ,38319 ],[46853 ,38737 ],[63987 ,34471 ],
         [39297 ,45425 ],[39465 ,45347 ],[63822 ,63819 ],[38843 ,45217 ],[45795 ,36783 ],[36496 ,47493 ],
         [46133 ,36282 ],[34219 ,49887 ],[34131 ,50097 ],[34031 ,50030 ],[30725 ,63996 ],[44420 ,38519 ],
         [42790 ,40597 ],[35925 ,46644 ],[35351 ,46947 ],[35464 ,46955 ],[35196 ,47188 ],[45238 ,36876 ],
         [34430 ,47700 ],[46008 ,35768 ],[31944 ,49767 ],[31601 ,50071 ],[31486 ,50094 ],[31977 ,49792 ],
         [31375 ,50182 ],[32198 ,63970 ],[38228 ,45129 ],[38356 ,45030 ],[32659 ,47497 ],[32100 ,47936 ],
         [40923 ,43976 ],[31637 ,48206 ],[30943 ,48581 ],[30598 ,48622 ],[30105 ,49038 ],[40257 ,44526 ],
         [28203 ,50370 ],[28449 ,50303 ],[27881 ,50677 ],[28039 ,63981 ],[28307 ,49208 ],[37792 ,44687 ]]

listddd=[[63979 ,47    ],[482   ,63975 ],[63986 ,33949 ],[538   ,63993 ],[36558 ,63960 ],
         [36792 ,63937 ],[48721 ,41062 ],[45332 ,43212 ],[63967 ,32802 ],[39670 ,48671 ],[34465 ,63962 ],
         [43896 ,42928 ],[63796 ,188   ],[33883 ,63965 ],[47457 ,38319 ],[46853 ,38737 ],[63987 ,34471 ],
         [39297 ,45425 ],[39465 ,45347 ],[63822 ,63819 ],[38843 ,45217 ],[45795 ,36783 ],[36496 ,47493 ],
         [46133 ,36282 ],[34219 ,49887 ],[34131 ,50097 ],[34031 ,50030 ],[30725 ,63996 ],[44420 ,38519 ],
         [42790 ,40597 ],[35925 ,46644 ],[35351 ,46947 ],[35464 ,46955 ],[35196 ,47188 ],[45238 ,36876 ],
         [34430 ,47700 ],[46008 ,35768 ],[31944 ,49767 ],[31601 ,50071 ],[31486 ,50094 ],[31977 ,49792 ],
         [31375 ,50182 ],[32198 ,63970 ],[38228 ,45129 ],[38356 ,45030 ],[32659 ,47497 ],[32100 ,47936 ],
         [40923 ,43976 ],[31637 ,48206 ],[30943 ,48581 ],[30598 ,48622 ],[30105 ,49038 ],[40257 ,44526 ],
         [28203 ,50370 ],[28449 ,50303 ],[27881 ,50677 ],[28039 ,63981 ],[28307 ,49208 ],[37792 ,44687 ],
         [37792 ,44687 ]]

listaaaGMM=[[63709 ,63998 ],[63976 ,283   ],[24064 ,63980 ],[3     ,63876 ],[63973 ,35465 ],[34495 ,63986 ],
            [45181 ,48150 ],[45054 ,48390 ],[33249 ,63986 ],[53340 ,34500 ],[42206 ,44016 ],[40228 ,45406 ],
            [28527 ,63949 ],[63955 ,27404 ],[63706 ,217   ],[35303 ,49160 ],[23476 ,63994 ],[37209 ,46716 ],
            [37468 ,46169 ],[32289 ,49794 ],[44269 ,37267 ],[63885 ,97    ],[22327 ,63966 ],[32624 ,63994 ],
            [42535 ,39082 ],[32249 ,47910 ],[32600 ,47396 ],[41789 ,37975 ],[33214 ,46769 ],[43986 ,34883 ],
            [32157 ,47348 ],[63716 ,83    ],[32437 ,49253 ],[42618 ,36109 ],[32586 ,48885 ],[44583 ,33893 ],
            [31098 ,49494 ],[31923 ,46178 ],[31281 ,49419 ],[30590 ,50553 ],[30665 ,50260 ],[19424 ,63995 ],
            [29946 ,50780 ],[33361 ,44745 ],[30368 ,50305 ],[50054 ,23937 ],[29988 ,50953 ],[32619 ,45542 ],
            [31920 ,46328 ],[29687 ,51218 ],[30208 ,50682 ],[30433 ,50334 ],[34874 ,47880 ],[37153 ,41022 ],
            [29175 ,51017 ],[35732 ,47203 ],[35318 ,63967 ],[28391 ,49667 ],[36563 ,43955 ],[28394 ,49467 ]]

listdddGMM=[[63976 ,283   ],[24064 ,63980 ],[3     ,63876 ],[63973 ,35465 ],[34495 ,63986 ],
            [45181 ,48150 ],[45054 ,48390 ],[33249 ,63986 ],[53340 ,34500 ],[42206 ,44016 ],[40228 ,45406 ],
            [28527 ,63949 ],[63955 ,27404 ],[63706 ,217   ],[35303 ,49160 ],[23476 ,63994 ],[37209 ,46716 ],
            [37468 ,46169 ],[32289 ,49794 ],[44269 ,37267 ],[63885 ,97    ],[22327 ,63966 ],[32624 ,63994 ],
            [42535 ,39082 ],[32249 ,47910 ],[32600 ,47396 ],[41789 ,37975 ],[33214 ,46769 ],[43986 ,34883 ],
            [32157 ,47348 ],[63716 ,83    ],[32437 ,49253 ],[42618 ,36109 ],[32586 ,48885 ],[44583 ,33893 ],
            [31098 ,49494 ],[31923 ,46178 ],[31281 ,49419 ],[30590 ,50553 ],[30665 ,50260 ],[19424 ,63995 ],
            [29946 ,50780 ],[33361 ,44745 ],[30368 ,50305 ],[50054 ,23937 ],[29988 ,50953 ],[32619 ,45542 ],
            [31920 ,46328 ],[29687 ,51218 ],[30208 ,50682 ],[30433 ,50334 ],[34874 ,47880 ],[37153 ,41022 ],
            [29175 ,51017 ],[35732 ,47203 ],[35318 ,63967 ],[28391 ,49667 ],[36563 ,43955 ],[28394 ,49467 ],
            [28394 ,49467 ]]

listMPP_k5_i30_p1=[[63701, 63986], [63934, 63958], [32, 63966], [93, 63795], [41, 63832], [122, 63901], 
                   [107, 63951], [25, 63986], [47, 63945], [38, 63836], [91, 63858], [28, 63802],  
                   [21, 63785], [123, 63954], [54, 63986], [126, 63765], [90, 63996], [104, 63922], [183, 63991],
                   [187, 63980], [145, 63951], [51, 63674], [52, 63702], [3, 63652], [3, 59988], [22642, 31954],
                   [22738, 32552], [22649, 33128], [22850, 33467]]

#==============================================================================
"querypoint list 转换"
#querypoint=pd.DataFrame(listaaa)
#destinatonpoint=pd.DataFrame(listddd)
querypointGMM=pd.DataFrame(listaaaGMM)
destinatonpointGMM=pd.DataFrame(listdddGMM)
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
datapath='D:/2DGMM(16000_8000-36000)/'
newdatapath='D:/2cluster_10times_GMM/'
#datapath='./OutConfigfile/'
iternum=0

"读取训练数据集================================================================"
"""用来读取原始数据集，得到priordataset，绘制聚类图，拟合的GMM热力图"""
for sappi_i in superappinterval:
    for sapps_i in superappsize:
        for vbri_i in vbrinterval:
            for vbrs_i in vbrsize:
                for trafi_i in trafinterval: 
                    for trafs_i in trafsize:
                        gamer=WDNoptimizer.GMMOptimizationUnit(cluster=2)
                        tempmemoryset=WDNoptimizer.ReinforcementLearningUnit()
                        for i in range(30):
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
                        a=np.mean(tempdataset[tempdataset['label']==0]['traf_throughput'])+np.mean(tempdataset[tempdataset['label']==0]['sapp_throughput'])
                        b=np.mean(tempdataset[tempdataset['label']==1]['traf_throughput'])+np.mean(tempdataset[tempdataset['label']==1]['sapp_throughput'])    
                        if a<b:
                            part0=tempdataset.loc[tempdataset['label']==0]
                            part0.loc[:,'label']=0
                            part1=tempdataset.loc[tempdataset['label']==1]
                            part0.loc[:,'label']=1
                            distriubuteculsterdata=distriubuteculsterdata.append(part0)
                            distriubuteculsterdata=distriubuteculsterdata.append(part1)
                        elif a>b:
                            part0=tempdataset.loc[tempdataset['label']==0]
                            part0.loc[:,'label']=1
                            part1=tempdataset.loc[tempdataset['label']==1]
                            part0.loc[:,'label']=0
                            distriubuteculsterdata=distriubuteculsterdata.append(part0)
                            distriubuteculsterdata=distriubuteculsterdata.append(part1)
#                        distriubuteculsterdata=distriubuteculsterdata.append(tempdataset)
                        iternum=iternum+1
"数据预处理===================================================================="
distriubuteculsterdata=distriubuteculsterdata.reset_index(drop=True)
priordataset=memoryset.qosmemoryunit#将原始的数据保存到内存中
qosgmmgamer=WDNoptimizer.GMMOptimizationUnit(cluster=2)#实例化GMM模型
qosgmmgamer.weightchanger(distriubuteculsterdata)
print(distriubuteculsterdata)#这个聚类结果是分别对每一组数据进行聚类之后聚合而成的数据
"根据querypoint返回模型进行数值的预测+++++++++++++++++++++++++++++++++++++++++++"
predictmean1=[]
predictmean2=[]
simulationmean1=[]
simulationmean2=[]

for i in listaaaGMM:
    "先根据目前的querypoint进行模型上的预测，报货两簇的均值，然后按大小来排列"
    ttt=np.array([i])
    asd=qosgmmgamer.querypredicter(data=distriubuteculsterdata,querypoint=ttt,fitz=9,fita=17)
    asd[0]=asd[0].tolist()
    asd[1]=asd[1].tolist()
    if asd[0][0]<asd[1][0]:
        predictmean1.append(asd[0][0])
        predictmean2.append(asd[1][0])
    elif asd[0][0]>asd[1][0]:
        predictmean1.append(asd[1][0])
        predictmean2.append(asd[0][0])
    "然后按照querypoint读取数据库数据"
    ttt=np.array(i)
    teaser=WDNfeedback.FeedBackWorker()#实例化反馈类
    teaser.updateQuerypointworker(ttt)#更新反馈参数
    newdata=teaser.updatetrainningsetworker(path=newdatapath,point=ttt,count=10)
    priordataset=priordataset.append(newdata)
    newgammer=WDNoptimizer.GMMOptimizationUnit(cluster=2)#实例化GMM模型
    newdataset=newgammer.dropNaNworker(newdata)#去掉nan数据
    newdataset=gamer.presortworker(newdataset,col1='traf_throughput',col2='sapp_throughput')
    newdataset=newgammer.clusterworker(newdataset,col1='traf_throughput',col2='sapp_throughput',count=iternum)#kmeans++聚类
    distriubuteculsterdata=distriubuteculsterdata.append(newdataset)
    a=np.mean(newdataset[newdataset['label']==0]['traf_throughput'])+np.mean(newdataset[newdataset['label']==0]['sapp_throughput'])
    b=np.mean(newdataset[newdataset['label']==1]['traf_throughput'])+np.mean(newdataset[newdataset['label']==1]['sapp_throughput'])    
    if a<b:
        simulationmean1.append(a) 
        simulationmean2.append(b)
    elif a>b:
        simulationmean1.append(b) 
        simulationmean2.append(a)

print(predictmean1)
print(predictmean2)
print(simulationmean1)
print(simulationmean2)
difference=[]
for i in range(len(predictmean1)):
    sumpredict=predictmean1[i]+predictmean2[i]
    sumsimulation=simulationmean1[i]+simulationmean2[i]
    differ=sumpredict-sumsimulation
    difference.append(differ)
"画图========================================================================="
plt.figure('Line fig',figsize=(20,10))
plt.xlabel('iteration times')
plt.ylabel('throughput')
plt.title('Throughput value ',fontsize='xx-large')
plt.plot(predictmean1,color='r', linewidth=2, alpha=0.6,label='predictmean1')
plt.plot(predictmean2,color='r', linewidth=2, alpha=0.6,label='predictmean2')
plt.plot(simulationmean1,color='black', linewidth=2, alpha=0.6,label='simulationmean1')
plt.plot(simulationmean2,color='black', linewidth=2, alpha=0.6,label='simulationmean2')
plt.plot(difference,color='blue', linewidth=2, alpha=0.6,label='difference')
plt.legend(fontsize='x-large')



































        
ttt=np.array([37468 ,46169 ])
teaser=WDNfeedback.FeedBackWorker()#实例化反馈类
teaser.updateQuerypointworker(ttt)#更新反馈参数
newdata=teaser.updatetrainningsetworker(path=newdatapath,point=ttt,count=10)
newgammer=WDNoptimizer.GMMOptimizationUnit(cluster=2)#实例化GMM模型
newdataset=newgammer.dropNaNworker(newdata)#去掉nan数据
newdataset=gamer.presortworker(newdataset,col1='traf_throughput',col2='sapp_throughput')
newdataset=newgammer.clusterworker(newdataset,col1='traf_throughput',col2='sapp_throughput',count=iternum)#kmeans++聚类
#distriubuteculsterdata=distriubuteculsterdata.append(newdataset)
a=np.mean(newdataset[newdataset['label']==0]['traf_throughput'])+np.mean(newdataset[newdataset['label']==0]['sapp_throughput'])
b=np.mean(newdataset[newdataset['label']==1]['traf_throughput'])+np.mean(newdataset[newdataset['label']==1]['sapp_throughput'])  
print(newdataset)
a  
b
part0=newdataset.loc[newdataset['label']==1]
print(part0)
part0.loc[:,'label']=0
print(part0)
print(newdataset)
if a<b:
    part0=newdataset.loc[newdataset['label']==0]
    part0.loc[:,'label']=0
    part1=newdataset.loc[newdataset['label']==1]
    part0.loc[:,'label']=1
    distriubuteculsterdata=distriubuteculsterdata.append(part0)
    distriubuteculsterdata=distriubuteculsterdata.append(part1)
elif a>b:
    part0=newdataset.loc[newdataset['label']==0]
    part0.loc[:,'label']=1
    part1=newdataset.loc[newdataset['label']==1]
    part0.loc[:,'label']=0
    distriubuteculsterdata=distriubuteculsterdata.append(part0)
    distriubuteculsterdata=distriubuteculsterdata.append(part1)

ttt=np.array([[28527 ,63949 ]])    
asd=qosgmmgamer.querypredicter(data=distriubuteculsterdata,querypoint=ttt,fitz=9,fita=17)
asd[0]=asd[0].tolist()
asd[1]=asd[1].tolist()
print(asd[0][0])
if asd[0]<asd[1]:
    predictmean1.append(asd[0][0])
    predictmean2.append(asd[1][0])
elif asd[0]>asd[1]:
    predictmean1.append(asd[1])
    predictmean2.append(asd[0])

print(predictmean1)
print(predictmean2)







# =============================================================================
# #数据处理部分
# #质询点坐标列表
# qxgmm=np.array(querypointGMM[0])
# qxgmm=qxgmm.tolist()
# qygmm=np.array(querypointGMM[1])
# qygmm=qygmm.tolist()
# qx = np.array(querypoint[0])
# qx=qx.tolist()
# qy = np.array(querypoint[1])
# qy=qy.tolist()
# #下一个质询点坐标列表
# dxgmm=np.array(destinatonpointGMM[0])
# dxgmm=dxgmm.tolist()
# dygmm=np.array(destinatonpointGMM[1])
# dygmm=dygmm.tolist()
# dx = np.array(destinatonpoint[0])
# dx=dx.tolist()
# dy = np.array(destinatonpoint[1])
# dy=dy.tolist()
# #均值点坐标列表
# mxgmm=[]
# mygmm=[]
# for i in range(len(qxgmm)):
#     mxi=sum(qxgmm[:i])/(i+1)
#     myi=sum(qygmm[:i])/(i+1)
#     mxgmm.append(mxi)
#     mygmm.append(myi)
# mx=[]
# my=[]
# for i in range(len(qx)):
#     mxi=sum(qx[:i])/(i+1)
#     myi=sum(qy[:i])/(i+1)
#     mx.append(mxi)
#     my.append(myi)
# 
# #创建图并命名
# for i in range(len(qx)):
#     plt.figure('Line fig',figsize=(21,10))
#     #设置x轴、y轴名称
#     plt.subplot(121)
#     plt.xlabel('superapp packet size')
#     plt.ylabel('trafficgenerator packet size')
#     plt.title('GPR Query Point Selection',fontsize='xx-large')
#     plt.xlim(xmin, xmax)
#     plt.ylim(ymin, ymax)
#     plt.plot(qx[:i], qy[:i], color='r', linewidth=1, alpha=0.6,label='Query Trace')
#     plt.plot(mx[:i],my[:i],color='black',linewidth=1,alpha=0.3,linestyle='--',label='Mean Trace')
#     plt.scatter(qx[:i],qy[:i],s=75,c='blue',marker='.',label='Query List')
#     plt.scatter(mx[:i],my[:i],s=30,c='gray',marker='o')
#     plt.scatter(qx[i],qy[i],s=180,c='red',marker='*',label='Next Point')
#     plt.scatter(mx[i],my[i],s=100,c='black',marker='o',label='Mean point')
#     plt.legend(loc=6,fontsize='x-large')
#     #设置x轴、y轴名称
#     plt.subplot(122)
#     plt.xlabel('superapp packet size')
#     plt.ylabel('trafficgenerator packet size')
#     plt.title('GMM Query Point Selection',fontsize='xx-large')
#     plt.xlim(xmin, xmax)
#     plt.ylim(ymin, ymax)
#     plt.plot(qxgmm[:i], qygmm[:i], color='r', linewidth=1, alpha=0.6,label='Query Trace')
#     plt.plot(mxgmm[:i],mygmm[:i],color='black',linewidth=1,alpha=0.3,linestyle='--',label='Mean Trace')
#     plt.scatter(qxgmm[:i],qygmm[:i],s=75,c='blue',marker='.',label='Query List')
#     plt.scatter(mxgmm[:i],mygmm[:i],s=30,c='gray',marker='o')
#     plt.scatter(qxgmm[i],qygmm[i],s=180,c='red',marker='*',label='Next Point')
#     plt.scatter(mxgmm[i],mygmm[i],s=100,c='black',marker='o',label='Mean point')
#     plt.legend(loc=6,fontsize='x-large')
#     plt.savefig(figpath+'querypoint'+str(i)+".jpg")
#     plt.show()
# =============================================================================



























