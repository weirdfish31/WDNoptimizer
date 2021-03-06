# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:58:48 2018

@author: WDN
用来读取原始数据 画图表
"""
import WDNexataReader#读取数据
import WDNoptimizer#建模，画图，AF函数
import pandas as pd


flowdata=pd.DataFrame()#所有数据库的流聚合
appdata=pd.DataFrame()#所有数据库的某种业务的聚合
memoryset=WDNoptimizer.MemoryUnit()#记忆单元，存储每次的状态

#dataset='test_ REQUEST-SIZE EXP 18000 _ 2000'
#radio REQUEST-SIZE EXP 24000 _ 18000 _ RND EXP 22000

figpath="./Figure/"
datapath='G:/testData/2DGMM(16000_8000-36000)/'
#datapath='./OutConfigfile/'

"""
读取数据，对数据进行分类处理
"""
flowdata=pd.DataFrame()#所有数据库的流聚合
appdata=pd.DataFrame()#所有数据库的某种业务的聚合


#dataset='radio REQUEST-SIZE DET '+str(sapps_i)+' _ '+str(vbrs_i)+' _ RND DET '+str(trafs_i)+' _'+str(i)
dataset="radio REQUEST-SIZE DET 16000 _ 24000 _ RND DET 8000 _9"
readdb=WDNexataReader.ExataDBreader()#实例化
readdb.opendataset(dataset,datapath)#读取特定路径下的数据库
readdb.appnamereader()#读取业务层的业务名称
readdb.appfilter()#将业务名称分类至三个list
readdb.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
readdb.inputparainsert(20,16000,30,24000,30,8000)
#将每条流的业务设计参数加入类中的字典

#===================================================以上步骤不可省略，下方处理可以根据需求修改
a=readdb.flowaggregator(20,16000,30,24000,30,8000)
#将同一个源到目的业务数据聚合在一起输出一个dataframe，一个数据库生成一个流聚合frame
flowdata=flowdata.append(a)
b=readdb.alltypeaggregator(20,16000,30,24000,30,8000,'trafficgen')
#将同一种业务，如VBR数据聚合在一起 输出一个dataframe，
#一个数据库可以生成三个，VBR，Superapp，Trafficgen三种业务聚合frame
appdata=appdata.append(b)
#=============================================================================
"""
画图
"""
readdb.rawdatadrawer(figpath,c='red')



                             


