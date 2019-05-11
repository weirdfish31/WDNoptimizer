# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:47:07 2018

@author: WDN
test：进行文件修改测试
从第三次实验中copy来的
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
valuegmmgamer=WDNoptimizer.GMMvalueOptimizaitonUnit(cluster=2)#实例化GMM模型

figpath="./Figure/"#图像的存放位置
#datapath='G:/testData/2DGMM(16000_8000-36000)/'#先验数据的存放位置
datapath='G:/testData/LHSprior/'#LHS先验数据的存放位置
newdatapath='./OutConfigfile/'#新产生的数据的存放位置
iternum=0#迭代的记数，在读取先验数据时记为零

"读取TXT文件中迭代的state数据==================================================="
statefilename="./priorstate_test.txt"#存储的先验数据的state列表txt文件
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
#valuegmmgamer.obj['reg_prob_1']#test
#valuegmmgamer.obj['reg_value_0']#test
#valuegmmgamer.obj['err_value_1']#test
"AF函数模型===================================================================="
"""
需要对目前的AF函数UCB进行修改
目前有两簇的output，err，均值较大簇的prob
目前的AF函数为valueUCBhelper
"""
ttt=valuegmmgamer.valueUCBhelper(memoryset.probmemoryunit,kappa=1)
tu=ttt.tolist()
listaaa.append(tu)

"画图+++++++++++++++++++++++++++++++++++++++++++未完成+++++++++++++++++++++++"
"要绘制多指标合成的曲面，目前已经有模型参数，obj中提供"
valuegmmgamer.valuegragher(data=memoryset.probmemoryunit,qp=ttt,path=figpath)#多指标合成的画图

"反馈函数+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
"""根据原始数据集的模型和质询点，仿真X次，读取新的数据，加入到Priordataset，绘图，并找到下一个质询点"""
simucount=1
"把querypoint存储到log文件中"
for i in range(25):
    teaser=WDNfeedback.FeedBackWorker()#实例化反馈类
    teaser.updateQuerypointworker(ttt)#更新反馈参数
    "将反馈次数和querypoint写入log文件"
# =============================================================================
#     querypoint=str(ttt)
#     writeStr = "%s : {%s}\n" % (simucount, querypoint)
#     outlogfile.write(writeStr)
# =============================================================================
    teaser.runTest(count=10)#仿真
    newdata=teaser.updatetrainningsetworker(path=newdatapath,point=ttt,count=20,style='value')
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
    newgammer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=6,label=0)#第一簇高斯过程模型
    newgammer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=7,label=0)#第一簇概率高斯过程模型
    newgammer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=6,label=1)#第二簇高斯过程模型
    newgammer.gpbuilder(memoryset.probmemoryunit,fitx=1,fity=5,fitz=7,label=1)#第二簇概率高斯过程模型
    ttt=newgammer.valueUCBhelper(data=memoryset.probmemoryunit,kappa= 1)#AF函数
    tu=ttt.tolist()
    listaaa.append(tu)
    newgammer.valuegragher(data=memoryset.probmemoryunit,qp=ttt,path=figpath,count=simucount)#多指标合成的画图
    simucount=simucount+1#计数，修改文件名称
with open('querypoint_log.txt','w') as f:#记录每次AF选点的参数
    f.write('\n')
    f.write(str(listaaa))#写入listaaa，querypoint的序列




"原始未模块化的BOA过程=========================================================="
# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt  
# from sklearn.gaussian_process import GaussianProcessRegressor  
# from sklearn.gaussian_process.kernels import RBF,Matern, ConstantKernel as C  
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import scipy
# train_data = np.array(memoryset.memoryunit)
# print(train_data)
# X=train_data[:,[1,5]]
# print(X)
# y=train_data[:,6]
# print(y)
# #kernel = C(0.1, (0.001,0.1))*RBF(0.5,(1e-4,10))
# kernel = Matern(nu=2.5)  
# reg = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=0.1)  
# reg.fit(X, y)                                                         
# x_min, x_max = 0, 64000  
# y_min, y_max = 0, 64000
# xset, yset = np.meshgrid(np.arange(x_min, x_max, 500), np.arange(y_min, y_max, 500))
# output,err = reg.predict(np.c_[xset.ravel(), yset.ravel()],return_std=True)  
# output,err = output.reshape(xset.shape),err.reshape(xset.shape) 
# sigma = np.sum(reg.predict(X, return_std=True)[1])  
# up,down = output*(1+1.96*err), output*(1-1.96*err) 
# =============================================================================


# =============================================================================
# 
# def acquisitionfunction(x,gp,kappa):
#     mean,std=gp.predict(x,return_std=True)
#     return mean + kappa*std
# 
# bounds=pd.DataFrame()
# bounds_throughly=pd.DataFrame()
# x_tries = np.random.uniform(1, 64001,size=(100000))
# y_tries = np.random.uniform(1, 64001,size=(100000))
# bounds['sapps']=x_tries
# bounds['trafs']=y_tries
# try_data = np.array(bounds)
# 
# ys=acquisitionfunction(try_data,gp=reg,kappa=0.67)
# try_max=try_data[ys.argmax()]
# max_acq=ys.max()
# print(try_max)
# print(max_acq)
# 
# x_tries_throughly = np.random.uniform(1, 64001,size=(250))
# y_tries_throughly = np.random.uniform(1, 64001,size=(250))
# bounds_throughly['sapps']=x_tries
# bounds_throughly['trafs']=y_tries
# try_data_throughly = np.array(bounds_throughly)
# 
# for x_try in try_data_throughly:
#     # Find the minimum of minus the acquisition function
#     res = scipy.optimize.minimize(lambda x: -acquisitionfunction(x.reshape(-1, 2), gp=reg,kappa=0.67),
#                    x_try.reshape(-1, 2),
#                    bounds=((1,64001),(1,64001)),
#                    method="L-BFGS-B")
#     if max_acq is None or -res.fun[0] >= max_acq:
#         try_max = res.x
#         max_acq = -res.fun[0]
#  
# test =np.array([try_max])
# print(test)    
# =============================================================================




# =============================================================================
# 
# predictstate=[20,int(test[0,0]+1),30,18000,30,int(test[0,1]+1)]
# print(predictstate)
# predictdataset='radio REQUEST-SIZE DET '+str(int(test[0,0]+1))+' _ '+str(vbrs_i)+' _ RND DET '+str(int(test[0,1]+1))
# outdatapath='./OutConfigfile/'
# readdb1=radiohead.ExataDBreader()#实例化
# readdb1.opendataset(predictdataset,outdatapath)#读取特定路径下的数据库
# readdb1.appnamereader()#读取业务层的业务名称
# readdb1.appfilter()#将业务名称分类至三个list
# readdb1.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
# eva=WDNoptimizer.EvaluationUnit()
# vbr=readdb1.meandata('vbr')
# eva.calculateMetricEvaValue(vbr)
# trafficgen=readdb1.meandata('trafficgen')
# eva.calculateMetricEvaValue(trafficgen)
# superapp=readdb1.meandata('superapp')
# eva.calculateMetricEvaValue(superapp)
# value=eva.evaluationvalue()
# print(value)
# memoryset.insertmemoryunit(state=predictstate,value=value)
# print(memoryset.memoryunit)
# #memoryset.memoryunit.drop([81,82,83],inplace=True)
# 
# 
# =============================================================================


# =============================================================================
# fig = plt.figure(figsize=(21,10))  
# ax1 = fig.add_subplot(121, projection='3d')  
# surf = ax1.plot_surface(xset,yset,output, cmap=cm.coolwarm,linewidth=0, antialiased=False)  
# surf_u = ax1.plot_wireframe(xset,yset,up,colors='lightgreen',linewidths=1,  
#                             rstride=10, cstride=2, antialiased=True)  
# surf_d = ax1.plot_wireframe(xset,yset,down,colors='lightgreen',linewidths=1,  
#                             rstride=10, cstride=2, antialiased=True)  
# 
# ax1.scatter(train_data[:,1],train_data[:,5],train_data[:,6],c='blue')  
# ax1.set_title('the predict mean value at ('+str(test[0,0])+'  '+str(test[0,1])+'): {0} '.format(reg.predict(test)[0]))  
# ax1.set_xlabel('sapps')  
# ax1.set_ylabel('trafs')  
# ax1.set_zlabel('value')  
# 
# ax = fig.add_subplot(122)  
# s = ax.scatter(train_data[:,1],train_data[:,5],train_data[:,6],cmap=plt.cm.viridis,c='red')
# im=ax.imshow(output, interpolation='bilinear', origin='lower',  
#                extent=(x_min, x_max-1, y_min, y_max), aspect='auto')
#   
# ax.set_title('the predict mean ')  
# ax.hlines(test[0,1],x_min, x_max-1)  
# ax.vlines(test[0,0],y_min, y_max)  
# ax.text(test[0,0],test[0,1],'{0}'.format(reg.predict(test)[0]),ha='left',
#         va='bottom',color='k',size=15,rotation=0)  
# ax.set_xlabel('sapps')  
# ax.set_ylabel('trafs')  
# plt.subplots_adjust(left=0.05, top=0.95, right=0.95)
# plt.colorbar(mappable=im,ax=ax)
# plt.show() 
# =============================================================================


#print(rawvbrdata.mean())
#print(type(rawvbrdata.mean()))
#data=readdb.deviationormalizer(dataset=rawvbrdata)
#print(data)

"""统计绘图模块
drawwer module
"""
#readdb.boxdrawer('sapps','offeredload',dataset=flowdata)
#readdb.stripplotdrawer('sapps','offeredload',dataset=flowdata)
#readdb.barplotdrawer('sapps','jitter','vbrs',dataset=flowdata)
#readdb.scatterdrawer('sapps','vbrs','throughput',dataset=flowdata)
#readdb.rawdatadrawer(figpath,c='coral')
#readdb.kdedrawer("delay","hopcount",figpath)



























