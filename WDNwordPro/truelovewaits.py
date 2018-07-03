# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:29:15 2018

@author: WDN
main program
主程序 大部分的读取数据，画图，BOA的实现代码都在这一部分

目前修改：2018/6/26
1）将原始数据的聚类方式设计的更加科学：分别对每一组数据进行聚类之后在进行合成
2）添加了querypoint写入log文件的命令

"""
import radiohead#读取数据
import weirdfishes#建模，画图，AF函数
import feedbackprocess#反馈
import pandas as pd
import numpy as np



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

flowdata=pd.DataFrame()#所有数据库的流聚合
appdata=pd.DataFrame()#所有数据库的某种业务的聚合
memoryset=weirdfishes.ReinforcementLearningUnit()#记忆单元，存储每次的状态

#dataset='test_ REQUEST-SIZE EXP 18000 _ 2000'
#radio REQUEST-SIZE EXP 24000 _ 18000 _ RND EXP 22000
figpath="./Figure/"
datapath='G:/testData/2DGMM(16000_8000-36000)/'
newdatapath='./OutConfigfile/'



"读取训练数据集================================================================"
"""用来读取原始数据集，得到priordataset，绘制聚类图，拟合的GMM热力图"""
def runTest():
    distriubuteculsterdata=pd.DataFrame()
    for sappi_i in superappinterval:
        for sapps_i in superappsize:
            for vbri_i in vbrinterval:
                for vbrs_i in vbrsize:
                    for trafi_i in trafinterval: 
                        for trafs_i in trafsize:
                            gamer=weirdfishes.GMMOptimizationUnit(cluster=2)
                            tempmemoryset=weirdfishes.ReinforcementLearningUnit()
                            for i in range(60):
                                """
                                读取数据，对数据进行分类处理
                                """
                                dataset='radio REQUEST-SIZE DET '+str(sapps_i)+' _ '+str(vbrs_i)+' _ RND DET '+str(trafs_i)+' _'+str(i)
                                readdb=radiohead.ExataDBreader()#实例化
                                readdb.opendataset(dataset,datapath)#读取特定路径下的数据库
                                readdb.appnamereader()#读取业务层的业务名称
                                readdb.appfilter()#将业务名称分类至三个list
                                readdb.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
                                readdb.inputparainsert(sappi_i,sapps_i,vbri_i,vbrs_i,trafi_i,trafs_i)
                                #将每条流的业务设计参数加入类中的字典
                                print(sapps_i,vbrs_i,trafs_i)
                                "======================以上步骤不可省略，下方处理可以根据需求修改"
#                                a=readdb.flowaggregator(sappi_i,sapps_i,vbri_i,vbrs_i,trafi_i,trafs_i)
#                                #将同一个源到目的业务数据聚合在一起输出一个dataframe，一个数据库生成一个流聚合frame
#                                flowdata=flowdata.append(a)
#                                b=readdb.alltypeaggregator(sappi_i,sapps_i,vbri_i,vbrs_i,trafi_i,trafs_i,'trafficgen')
#                                将同一种业务，如VBR数据聚合在一起 输出一个dataframe，
#                                一个数据库可以生成三个，VBR，Superapp，Trafficgen三种业务聚合frame
#                                appdata=appdata.append(b)                         
                                """
                                评估部分，对于三种不同的业务有不同的权重:时延、抖动、丢包率、吞吐量
                                vbr:        [1,2,3,4]
                                trafficgen: [5,6,7,8]
                                superapp:   [9,10,11,12]
                                vbr,superapp,trafficgen
                                """
                                eva=weirdfishes.EvaluationUnit()
                                superapp=readdb.meandata('superapp')
                                eva.calculateMetricEvaValue(superapp)
                                vbr=readdb.meandata('vbr')
                                eva.calculateMetricEvaValue(vbr)
                                trafficgen=readdb.meandata('trafficgen')
                                eva.calculateMetricEvaValue(trafficgen)
#                                value=eva.evaluationvalue()
#                                print(value)                        
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
#                                memoryset.insertmemoryunit(state=state,value=value)
                                memoryset.qosinserter(state=state,qos=qos)    
                                tempmemoryset.qosinserter(state=state,qos=qos)
                            tempdataset=gamer.dropNaNworker(tempmemoryset.qosmemoryunit)
                            tempdataset=gamer.clusterworker(tempdataset,col1='traf_throughput',col2='sapp_throughput')
                            distriubuteculsterdata=distriubuteculsterdata.append(tempdataset)
                            
    "数据预处理===================================================================="
    import weirdfishes
    priordataset=memoryset.qosmemoryunit#将原始的数据保存到内存中
    qosgmmgamer=weirdfishes.GMMOptimizationUnit(cluster=2)#实例化GMM模型
#    testdata=qosgmmgamer.dropNaNworker(memoryset.qosmemoryunit)#去掉nan数据
#    print(testdata)
    print(distriubuteculsterdata)#这个聚类结果是分别对每一组数据进行聚类之后聚合而成的数据
    print(priordataset)
#    testdata=qosgmmgamer.clusterworker(testdata,col1='traf_throughput',col2='sapp_throughput')#kmeans++聚类
    
    "AF函数======================================================================="
#    bbb=qosgmmgamer.acquisitionfunctionmethod2(testdata,0.6,1,5,16)#单指标的AF函数设计2
#    ccc=qosgmmgamer.acquisitionfunctionmethod1(testdata,0.6,1,5,16)#单指标的AF函数设计1
    ttt=qosgmmgamer.multiUCBhelper(data=distriubuteculsterdata,kappa= 0.7,fitz=9,fita=17)#多指标的AF函数
#    ttt=np.array([63670,63990])
    "画图+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
#    fitz=7 17
    qosgmmgamer.gmmbuilder(distriubuteculsterdata,fitx=1,fity=5,fitz=17)#生成traf_messagecompletionrate均值，标准差平面的预测结果，用于画图
    qosgmmgamer.gmmbuilder(distriubuteculsterdata,fitx=1,fity=5,fitz=9)#生成sapp_jitter均值标准差的平面的预测结果，用于画图
    qosgmmgamer.multiGMMbuilder(distriubuteculsterdata,fitz=9,fita=17)#生成多指标的加权平面，保存的功能还未实现，需要实现
    "要绘制多指标合成的曲面，必须进行上面两个步骤，生成"
    qosgmmgamer.mulitgragher(data=distriubuteculsterdata,test=ttt,path=figpath)#多指标合成的画图
    
#    qosgmmgamer.heatgragher(testdata,ttt,fitz=7)#绘图
#    qosgmmgamer.heatgragher(testdata,ttt,fitz=16)#绘图
    
    "反馈函数+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    """根据原始数据集的模型和质询点，仿真X次，读取新的数据，加入到Priordataset，绘图，并找到下一个质询点"""
    simucount=1
    "把querypoint存储到log文件中"
    outlogfile = open('./queryPoint.log', 'w')
    
    for i in range(30):
#        print(memoryset.qosmemoryunit)
        teaser=feedbackprocess.FeedBackWorker()#实例化反馈类
        teaser.updateQuerypointworker(ttt)#更新反馈参数
#        print(ttt)
#        ttt=np.array([[41,485]])
        "将反馈次数和querypoint写入log文件"
        querypoint=str(ttt)
        writeStr = "%s : {%s}\n" % (simucount, querypoint)
        outlogfile.write(writeStr)
        teaser.runTest(count=30)#仿真
        newdata=teaser.updatetrainningsetworker(path=newdatapath,dataset=priordataset,point=ttt,count=30)
        priordataset=priordataset.append(newdata)#将新数据加入至原始训练集中
#        print(priordataset)
        newgammer=weirdfishes.GMMOptimizationUnit(cluster=2)#实例化GMM模型
        newdataset=newgammer.dropNaNworker(newdata)#去掉nan数据
        newdataset=newgammer.clusterworker(newdataset,col1='traf_throughput',col2='sapp_throughput',count=simucount)#kmeans++聚类
        distriubuteculsterdata=distriubuteculsterdata.append(newdataset)
        "上面的新数据聚类完成，下面进行画图和querypoint的更新"
        newgammer.gmmbuilder(distriubuteculsterdata,fitx=1,fity=5,fitz=17)#生成traf_messagecompletionrate均值，标准差平面的预测结果，用于画图
        newgammer.gmmbuilder(distriubuteculsterdata,fitx=1,fity=5,fitz=9)#生成sapp_jitter均值标准差的平面的预测结果，用于画图
        newgammer.multiGMMbuilder(distriubuteculsterdata,fitz=9,fita=17)#生成多指标的加权平面，保存的功能还未实现，需要实现
        newgammer.mulitgragher(data=distriubuteculsterdata,test=ttt,path=figpath,count=simucount)#多指标合成的画图
        simucount=simucount+1#计数，修改文件名称
        ttt=newgammer.multiUCBhelper(data=distriubuteculsterdata,kappa= 0.7,fitz=9,fita=17)#AF函数








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
# eva=weirdfishes.EvaluationUnit()
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



























