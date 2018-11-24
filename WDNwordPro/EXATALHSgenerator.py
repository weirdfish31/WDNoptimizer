# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 09:45:41 2018

@author: WDN
Prior Data Generator
加入LHS采样方法，对空间中的数据进行采样，同时进行仿真，得到先验数据
"""
import WDNfeedback#反馈
import pandas as pd
import numpy as np
import lhsmdu

    
    

"利用LHS在空间中采样，这里用二维采样============================================="
k = lhsmdu.sample(2, 10) # Latin Hypercube Sampling with multi-dimensional uniformit

"VBR其他流---------------------------------------------------------------------"
vbrinterval=[30]
#vbrsize=[8000,10000,12000,14000,16000,18000]
vbrsize=[24000]
"Superapp视频流----------------------------------------------------------------"
superappinterval=[20]
#superappsize=[8000,10000,12000,14000,16000,18000,20000,22000,24000]
#superappsize=[16000,30000]
x1 = np.array(k[0])*64000
x1=np.floor(x1)
superappsize=x1[0].tolist()
print("superapp size is "+str(superappsize))
"Trafficgenerator图像流--------------------------------------------------------"
trafinterval=[30]
#trafsize=[8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,32000,34000,36000]
#trafsize=[22000]
y1 = np.array(k[1])*64000
y1=np.floor(y1)
trafsize=y1[0].tolist()
print("trafficgenerator size is "+str(superappsize))
print(len(trafsize))
trafsize[0]


"保存state组合在log文件中（txt）"
statememory=pd.DataFrame()#用来给予所有的采样的从设计参数集合

#dataset='test_ REQUEST-SIZE EXP 18000 _ 2000'
#radio REQUEST-SIZE EXP 24000 _ 18000 _ RND EXP 22000
figpath="./Figure/"
datapath='G:/testData/2DGMM(16000_8000-36000)/'
newdatapath='./OutConfigfile/'



"生成训练数据集================================================================"
"""用来遍历采样点生成原始数据集，得到先验数据库"""

#outlogfile = open('./queryPoint.log', 'w')
for sappi_i in superappinterval:
    for vbri_i in vbrinterval:
        for vbrs_i in vbrsize:
            for trafi_i in trafinterval: 
                for count_i in range(len(superappsize)):
                    sapps_i=superappsize[count_i]
                    trafs_i=trafsize[count_i]
                    state=[sappi_i,sapps_i,vbri_i,vbrs_i,trafi_i,trafs_i]
                    statememory=statememory.append(state)
                    ttt=np.array([sapps_i,trafs_i])
                    teaser=WDNfeedback.FeedBackWorker(superinter=sappi_i,vbrinter=vbri_i,vbrsize=vbrs_i,trafinter=trafi_i)#实例化反馈类
                    teaser.updateQuerypointworker(ttt)#更新反馈参数
                    "将反馈次数和querypoint写入log文件"
                    teaser.runTest(count=1)#仿真
                
with open('priorstate.txt','w') as f:
    f.write(str(statememory))
    f.write('\n')