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
import math
pd.set_option('display.max_rows',None)
    
    

"利用LHS在空间中采样，这里用二维采样============================================="
k = lhsmdu.sample(6, 120) # Latin Hypercube Sampling with multi-dimensional uniformit
-"VBR其他流---------------------------------------------------------------------"
x1 = np.array(k[0])*100
x1=np.floor(x1)
vbrinterval=x1[0].tolist()
vbrinterval = [ math.ceil(x) for x in vbrinterval ]
vbrinterval
vbrinterval = [ x+1 for x in vbrinterval ]
vbrinterval
x1 = np.array(k[1])*64000
x1=np.floor(x1)
vbrsize=x1[0].tolist()
vbrsize = [ math.ceil(x) for x in vbrsize ]
vbrsize = [ x+1 for x in vbrsize ]
"Superapp视频流----------------------------------------------------------------"
x1 = np.array(k[2])*100
x1=np.floor(x1)
superappinterval=x1[0].tolist()
superappinterval = [ math.ceil(x) for x in superappinterval ]
superappinterval = [ x+1 for x in superappinterval ]
x1 = np.array(k[3])*64000
x1=np.floor(x1)
superappsize=x1[0].tolist()
superappsize = [ math.ceil(x) for x in superappsize ]
superappsize = [ x+1 for x in superappsize ]
"Trafficgenerator图像流--------------------------------------------------------"
x1 = np.array(k[4])*100
x1=np.floor(x1)
trafinterval=x1[0].tolist()
trafinterval = [ math.ceil(x) for x in trafinterval ]
trafinterval = [ x+1 for x in trafinterval ]
y1 = np.array(k[5])*64000
y1=np.floor(y1)
trafsize=y1[0].tolist()
trafsize = [ math.ceil(x) for x in trafsize ]
trafsize = [ x+1 for x in trafsize ]




"保存state组合在log文件中（txt）"
statememory=pd.DataFrame()#用来给予所有的采样的从设计参数集合                                                                                                                        

for count_i in range(len(superappsize)):
    state=[superappinterval[count_i],superappsize[count_i],vbrinterval[count_i],vbrsize[count_i],trafinterval[count_i],trafsize[count_i]]
    data=pd.DataFrame([state])
    statememory=statememory.append(data)
with open('priorLHS6D.txt','w') as f:
    f.write(str(statememory))
    f.write('\n')

"生成训练数据集================================================================"
"""用来遍历采样点生成原始数据集，得到先验数据库"""

#outlogfile = open('./queryPoint.log', 'w')
for count_i in range(len(superappsize)):
    teaser=WDNfeedback.FeedBackWorker(superinter=superappinterval[count_i],supersize=superappsize[count_i],vbrsize=vbrsize[count_i],vbrinter=vbrinterval[count_i],trafinter=trafinterval[count_i],trafsize=trafsize[count_i])#实例化反馈类
#    teaser.updateQuerypointworker(ttt)#更新反馈参数
    "将反馈次数和querypoint写入log文件"
    teaser.runTest_state(count=20)#仿真
                

    
