# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:55:46 2018

@author: WDN

Bayesian Optimization Units
Evaluation Units
ReinforcementLearning Units
"""
#评估，增强学习记忆单元所需
import math
import radiohead
import numpy as np
import pandas as pd
#贝叶斯优化所需：
import matplotlib.pyplot as plt  
from sklearn.gaussian_process import GaussianProcessRegressor  
from sklearn.gaussian_process.kernels import RBF,Matern, ConstantKernel as C  
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy
from sklearn import mixture
from sklearn.mixture import GMM
from sklearn.cluster import KMeans

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GMMOptimizationUnit:
    """
    利用高斯混合模型对value进行建模
    包含所需的数据处理过程：去除NAN数据
    包含聚类过程：KMeans++聚类方法（目前只有Kmeans)
    """
    def __init__(self,cluster=2):
        self.xmin,self.xmax=0,64000
        self.ymin,self.ymax=0,64000
        self.xset,self.yset=np.meshgrid(np.arange(self.xmin,self.xmax, 500), np.arange(self.ymin,self.ymax, 500))
        self.n_clusters=cluster
        self.kernel=Matern(nu=2.5)
        self.wirecolor=['mediumpurple','lightgreen','gold','maroon']
    
    def componentselecter(self,data,i):
        testdata=data[data.label==i]
        testdata=testdata.reset_index(drop=True)
        return testdata
        
    def dropNaNworker(self,data):
        testdata=data.dropna(axis=0,how='any')
        testdata=testdata.reset_index(drop=True)
        return testdata
    def gmmbuilder(self,data):
        """
        根据聚类的结果，对用以标签下的数据进行GP回归，得到均值标准差
        """
        self.obj={}
        for i in range(self.n_clusters):
            testdata=data[data['label']==i]
            testdata=testdata.reset_index(drop=True)
            self.npdata=np.array(testdata)
            self.reg=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
            self.reg.fit(self.npdata[:,[1,5]],self.npdata[:,6])
            self.obj['output'+str(i)],self.obj['err'+str(i)]=self.reg.predict(np.c_[self.xset.ravel(),self.yset.ravel()],return_std=True)
            self.obj['output'+str(i)],self.obj['err'+str(i)]=self.obj['output'+str(i)].reshape(self.xset.shape),self.obj['err'+str(i)].reshape(self.xset.shape)
            self.obj['sigma'+str(i)]=np.sum(self.reg.predict(self.npdata[:,[1,5]],return_std=True)[1])
            self.obj['up'+str(i)],self.obj['down'+str(i)]=self.obj['output'+str(i)]*(1+1.96*self.obj['err'+str(i)]),self.obj['output'+str(i)]*(1-1.96*self.obj['err'+str(i)])
    def heatgragher(self,test):
        """
        绘制热力图和预测的下一个点，坐标是自适应的
        """
        fig = plt.figure(figsize=(21,10))  
        ax1 = fig.add_subplot(121, projection='3d')
        for i in range(self.n_clusters):
            self.obj['surf'+str(i)]=ax1.plot_surface(self.xset,self.yset,self.obj['output'+str(i)], cmap=plt.get_cmap('rainbow'),linewidth=0, antialiased=False)
            self.obj['surf_u'+str(i)]=ax1.plot_wireframe(self.xset,self.yset,self.obj['up'+str(i)],colors=self.wirecolor[i],linewidths=1,  
                                    rstride=10, cstride=2, antialiased=True)
            self.obj['surf_d'+str(i)]=ax1.plot_wireframe(self.xset,self.yset,self.obj['down'+str(i)],colors=self.wirecolor[i],linewidths=1,  
                                    rstride=10, cstride=2, antialiased=True)
        ax1.scatter(self.npdata[:,1],self.npdata[:,5],self.npdata[:,6],c='black')  
        ax1.set_title('the predict mean value at ('+str(test[0,0])+'  '+str(test[0,1])+'): {0} '.format(self.reg.predict(test)[0]))  
        ax1.set_xlabel('sapps')  
        ax1.set_ylabel('trafs')  
        ax1.set_zlabel('value') 
        
        ax = fig.add_subplot(122)  
        s = ax.scatter(self.npdata[:,1],self.npdata[:,5],self.npdata[:,6],cmap=plt.cm.viridis,c='red')
        im=ax.imshow(self.obj['output1'], interpolation='bilinear', origin='lower',  
                       extent=(self.xmin, self.xmax-1, self.ymin, self.ymax), aspect='auto')
          
        ax.set_title('the predict mean ')  
        ax.hlines(test[0,1],self.xmin, self.xmax-1)  
        ax.vlines(test[0,0],self.ymin, self.ymax)  
        ax.text(test[0,0],test[0,1],'{0}'.format(self.reg.predict(test)[0]),ha='left',
                va='bottom',color='k',size=15,rotation=0)  
        ax.set_xlabel('sapps')  
        ax.set_ylabel('trafs')  
        plt.subplots_adjust(left=0.05, top=0.95, right=0.95)
        plt.colorbar(mappable=im,ax=ax)
        plt.show() 
        
        
            
    def clusterworker(self,data,col1,col2):
        """
        SKlearn.cluster里面自带的KMeans函数，这里我们只取了数据的二维，方便画图
        """
        value=np.array(data[col1])
        value1=value.reshape((-1,1))
        trafs=np.array(data[col2])
        trafs1=trafs.reshape((-1,1))
        c=np.hstack((trafs1,value1))
        c=c[:,::-1]
        estimator=KMeans(n_clusters=self.n_clusters,max_iter=1000)
        estimator.fit(c)
        lable_pred=estimator.labels_
        #统计各类的数据的数目
        r1=pd.Series(estimator.labels_).value_counts()
        print(r1)
        r = pd.concat([data, pd.Series(estimator.labels_, index = data.index)], axis = 1)
        r.rename(columns={0:'label'},inplace=True)
        print(r)
        for i in range(len(c)):
            if int(lable_pred[i])==0:
                plt.scatter(c[i][0],c[i][1],color='red')
            if int(lable_pred[i])==1:
                plt.scatter(c[i][0],c[i][1],color='orangered')
            if int(lable_pred[i])==2:
                plt.scatter(c[i][0],c[i][1],color='gold')
            if int(lable_pred[i])==3:
                plt.scatter(c[i][0],c[i][1],color='violet')
        plt.show()
        return r
    
    def EMworker(self,data):
        """
        SKLearn里面自带的EM函数，这里我们只取了数据的二维，方便画图
        """
        value=np.array(data['value'])
        value1=value.reshape((-1,1))
        trafs=np.array(data['trafs'])
        trafs1=trafs.reshape((-1,1))
        c=np.hstack((trafs1,value1))
        c=c[:,::-1]
        gmm = GMM(n_components=3,n_iter=1000).fit(c)
        print(gmm)
        labels = gmm.predict(c)
        print(labels)
        plt.scatter(c[:, 0], c[:, 1],c=labels, s=40, cmap='viridis')
        probs = gmm.predict_proba(c)
        print(probs[:5].round(3))
        size = 50 * probs.max(1) ** 2  # 由圆点面积反应概率值的差异
        plt.scatter(c[:, 0], c[:, 1], c=labels, cmap='viridis', s=size)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class BayesianOptimizationUnit:
    """
    贝叶斯优化类：
    用作进行对业务层设计参数与输出的评分之间建立概率模型，进行设计指导
    
    这个函数是针对业务层输入参数进行设计的，数据结构不是针对所有的数据
    画图函数是三维的函数，变量连续的变量,核函数是matern函数，nu=2.5 后续做成可调整的 
    x为superapp的包大小（0,64000）
    y是trafficgenerator的包大小（0,64000）
    z是系统综合评分value
    """
    def __init__(self):
        """
        初始化BOA可函数等参数
        """
        self.kernel = Matern(nu=2.5)
        self.xmin,self.xmax=0,64000
        self.ymin,self.ymax=0,64000
        self.xset,self.yset=np.meshgrid(np.arange(self.xmin,self.xmax, 500), np.arange(self.ymin,self.ymax, 500))

    
    def gussianproccessfitter(self,data):
        """
        输入dataframe，进行GP拟合
        """
        self.train_data=np.array(data)
        self.reg=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
        self.reg.fit(self.train_data[:,[1,5]],self.train_data[:,6])
        self.output,self.err=self.reg.predict(np.c_[self.xset.ravel(),self.yset.ravel()],return_std=True)
        self.output,self.err=self.output.reshape(self.xset.shape),self.err.reshape(self.xset.shape)
        self.sigma=np.sum(self.reg.predict(self.train_data[:,[1,5]],return_std=True)[1])
        self.up,self.down=self.output*(1+1.96*self.err),self.output*(1-1.96*self.err)        
        
        
    def heatpointer(self,test):
        """
        绘制热力图和预测的下一个点，坐标是自适应的
        """
        fig = plt.figure(figsize=(21,10))  
        ax1 = fig.add_subplot(121, projection='3d')  
        surf = ax1.plot_surface(self.xset,self.yset,self.output, cmap=cm.coolwarm,linewidth=0, antialiased=False)  
        surf_u = ax1.plot_wireframe(self.xset,self.yset,self.up,colors='lightgreen',linewidths=1,  
                                    rstride=10, cstride=2, antialiased=True)  
        surf_d = ax1.plot_wireframe(self.xset,self.yset,self.down,colors='lightgreen',linewidths=1,  
                                    rstride=10, cstride=2, antialiased=True)  
        
        ax1.scatter(self.train_data[:,1],self.train_data[:,5],self.train_data[:,6],c='blue')  
        ax1.set_title('the predict mean value at ('+str(test[0,0])+'  '+str(test[0,1])+'): {0} '.format(self.reg.predict(test)[0]))  
        ax1.set_xlabel('sapps')  
        ax1.set_ylabel('trafs')  
        ax1.set_zlabel('value')  
        
        ax = fig.add_subplot(122)  
        s = ax.scatter(self.train_data[:,1],self.train_data[:,5],self.train_data[:,6],cmap=plt.cm.viridis,c='red')
        im=ax.imshow(self.output, interpolation='bilinear', origin='lower',  
                       extent=(self.xmin, self.xmax-1, self.ymin, self.ymax), aspect='auto')
          
        ax.set_title('the predict mean ')  
        ax.hlines(test[0,1],self.xmin, self.xmax-1)  
        ax.vlines(test[0,0],self.ymin, self.ymax)  
        ax.text(test[0,0],test[0,1],'{0}'.format(self.reg.predict(test)[0]),ha='left',
                va='bottom',color='k',size=15,rotation=0)  
        ax.set_xlabel('sapps')  
        ax.set_ylabel('trafs')  
        plt.subplots_adjust(left=0.05, top=0.95, right=0.95)
        plt.colorbar(mappable=im,ax=ax)
        plt.show() 
        
    def UCBmethod(self,x,gp,kappa):
        """
        upper confidence bound 方法
        根据随机过程的方差和均值进行选择，不会陷入局部最优
        这种做法比较的是置信区间内的最大值，尽管看起来简单，但是实际效果却意外的好
        """
        mean,std=gp.predict(x,return_std=True)
        return mean + kappa*std
    
    def EImethod(self,x,gp,xi):
        """
        expected improvement 方法
        explore时，应该选择那些具有比较大方差的点，而在exploit时，则应该优先考虑均值大的点
        EI使用的是数学期望，因此"大多少"这个因素被考虑在内
        """
        mean, std = gp.predict(x, return_std=True)
        z = (mean - self.ymax - xi)/std
        return (mean - self.ymax - xi)*np.linalg.norm.cdf(z) + std*np.linalg.norm.pdf(z)
        
        
        
    def POImethod(self,x,gp,xi):
        """
        probability of improvement 方法
        这种方法考虑让新的采样提升最大值的概率最大
        POI是一个概率函数，描述的是新的点能比当前最大值大的概率，但是大多少并不关心
        """
        mean,std=gp.predict(x,return_std=True)
        z=(mean-self.ymax-xi)/std
        return np.linalg.norm.cdf(z)
    
        
    def acquisitionfunction(self,kappa):
        """
        1）先随机选取100000点进行估计，得到UCB值最大的坐标
        2）再选取250个点进行拟牛顿法与1）中得到的最大值进行比较
        3）得到下一个仿真的点向量
        """
        bounds=pd.DataFrame()
        x_tries = np.random.uniform(1, 64001,size=(100000))
        y_tries = np.random.uniform(1, 64001,size=(100000))
        bounds['sapps']=x_tries
        bounds['trafs']=y_tries
        try_data = np.array(bounds)
        ys=self.UCBmethod(try_data,gp=self.reg,kappa=kappa)
        try_max=try_data[ys.argmax()]
        max_acq=ys.max()
        print(try_max)
        print(max_acq)
        try:
            bounds_throughly=pd.DataFrame()
            x_tries_throughly = np.random.uniform(1, 64001,size=(250))
            y_tries_throughly = np.random.uniform(1, 64001,size=(250))
            bounds_throughly['sapps']=x_tries_throughly
            bounds_throughly['trafs']=y_tries_throughly
            try_data_throughly = np.array(bounds_throughly)
            for x_try in try_data_throughly:
                # Find the minimum of minus the acquisition function
                res = scipy.optimize.minimize(lambda x: -self.UCBmethod(x.reshape(-1, 2), gp=self.reg,kappa=kappa),
                               x_try.reshape(-1, 2),
                               bounds=((1,64001),(1,64001)),
                               method="L-BFGS-B")
                if max_acq is None or -res.fun[0] >= max_acq:
                    try_max = res.x
                    max_acq = -res.fun[0]
        except:
            print('L-BFGS-B didn\'t work!!!')
        test =np.array([try_max])
        print(test)
        return test
        
    def insertnewworker(self,test,dataframe,path='./OutConfigfile/',vbrs_i=18000):
        """
        将acquisitionFunction选取的点仿真后得到的数据库读取出来，加入到先验数据中
        """
        predictstate=[20,int(test[0,0]+1),30,int(vbrs_i),30,int(test[0,1]+1)]
        predictdataset='radio REQUEST-SIZE DET '+str(int(test[0,0]+1))+' _ '+str(vbrs_i)+' _ RND DET '+str(int(test[0,1]+1))          
        readdb1=radiohead.ExataDBreader()
        readdb1.opendataset(predictdataset,path)
        readdb1.appnamereader()#读取业务层的业务名称
        readdb1.appfilter()#将业务名称分类至三个list
        readdb1.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
        eva=EvaluationUnit()
        vbr=readdb1.meandata('vbr')
        eva.calculateMetricEvaValue(vbr)
        trafficgen=readdb1.meandata('trafficgen')
        eva.calculateMetricEvaValue(trafficgen)
        superapp=readdb1.meandata('superapp')
        eva.calculateMetricEvaValue(superapp)
        value=eva.evaluationvalue()
        dataframe.insertmemortunit(state=predictstate,value=value)
        return dataframe
        
        
        
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EvaluationUnit:
    """
    性能评估类：
    用来对业务仿真数据进行评分
    
    目前所有的评估指标权重相等：  分为三种业务：trafficgenerator，superapp，cbr 
                                每种业务分四种指标：吞吐量、时延、时延抖动、消息完成率
    
    得到value
    """
    def __init__(self):
        self.weight= np.array([[0.0825,0.0825,0.0825,0.0825],
                               [0.0825,0.0825,0.0825,0.0825],
                               [0.0825,0.0825,0.0825,0.0825]], dtype=float) 
        self.normalizedata=[]
        self.qoslist=[]
    def calculateMetricEvaValue(self,dataset):
        """
        归一化处理：对于四种不同的指标，进行不同的归一化处理
        before the evaluation proccess must modify the raw data
        
        """
        #delay  x=3,y=0.5,flag=0
        delay = dataset['delay']
        a = self.exponentNormalizer(x=3,y=0.5,flag=0,value=delay)
        #Jitter xmax=0.1,y=0.001,flag=0
        jitter = dataset['jitter']
        b = self.exponentNormalizer(x=0.1,y=0.001,flag=0,value=jitter)
        #massagecompleterate x=0.9,y=0.1 flag=0
        packetoss = 1-dataset['messagecompletionrate']
        c = self.exponentNormalizer(x=0.9,y=0.1,flag=0,value=packetoss)
        #throughput max=100000000
        throughput = dataset['throughput']
        if throughput>100000000:
            d = 1
        else:
            d = self.sqrtNormalizer(x=100000000,value=throughput)
        self.normalizedata.append([a,b,c,d])
        self.qoslist=self.qoslist+[a,b,c,d]
    def evaluationvalue(self):
        """
        输出评估分数，对归一化之后的数据乘权重向量，得到分数
        output the evaluation result
        """
        a=np.array(self.normalizedata)
        b=self.weight
        s=0
        value=np.multiply(a,b)
        s=sum(map(sum,value))
        return s

    def exponentNormalizer(self,x,y,flag,value):
        """
        指数归一化函数
        ExponentNormalizer,calculate the q about the inputed parameters
        """
        q=0
        if (y>0 and y<1 and x>0):
            if flag==0:
                q = math.log(y) / (-x)
                return math.exp(-q * value)
            elif flag==1:
                q = math.log(1 - y) / (-x)
                return (1 - math.exp(-q * value))
        return 0.0
    
    def sqrtNormalizer(self,x,value):
        """
        平方根归一化函数
        throughputnormalizer
        """
        if value>0:
            k = 1 / math.sqrt(x)
            return k * math.sqrt(value)
        else:
            return 0.0
    
        
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ReinforcementLearningUnit:
    """
    强化学习类：
    未完成，现实现了state数据的存储功能
    目前暂时是用来存储输入的设计指标，评估分数，设计差值（动作），增益（与上一次仿真使用的设计指标和分数）
    """
    def __init__(self):
        """
        存储记忆单元
        save the memoryunit
        """
        self.memoryunit=pd.DataFrame()
        self.count=0
        self.qosmemoryunit=pd.DataFrame()
        
#        self.n_actions
#        self.n_states
#        self.lr#学习速率
#        self.gamma
#        self.epsilon_max
#        self.replace_target_iter
#        self.memory_size
#        self.batch_size=batch_size
#        
#        self.epsilon_increment = e_greedy_increment
#        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
    def qosinserter(self,state,qos):
        """
        添加state和qos指标的归一化值（state6个设计指标,qos12个指标）
        """
        aa=pd.DataFrame()
        aa['sappi']=[state[0]]
        aa['sapps']=[state[1]]
        aa['vbri']=[state[2]]
        aa['vbrs']=[state[3]]
        aa['trafi']=[state[4]]
        aa['trafs']=[state[5]]
        aa['sapp_delay']=[qos[0]]
        aa['sapp_jitter']=[qos[1]]
        aa['sapp_messagecompletionrate']=[qos[2]]
        aa['sapp_throughput']=[qos[3]]
        
        aa['vbr_delay']=[qos[4]]
        aa['vbr_jitter']=[qos[5]]
        aa['vbr_messagecompletionrate']=[qos[6]]
        aa['vbr_throughput']=[qos[7]]
        
        aa['traf_delay']=[qos[8]]
        aa['traf_jitter']=[qos[9]]
        aa['traf_messagecompletionrate']=[qos[10]]
        aa['traf_throughput']=[qos[11]]

        print(aa)
        self.qosmemoryunit=self.qosmemoryunit.append(aa)
        self.qosmemoryunit=self.qosmemoryunit.reset_index(drop=True)
        
    def insertmemoryunit(self,state,value):
        """
        添加新的state在记忆单元
        目前的业务设计参数有6种，随着以后的进行应该会增加
        append the new dataframe from the simulaiton
        目前不需要action gain参数 暂时注释掉
        """
        aa=pd.DataFrame()
        aa['sappi']=[state[0]]
        aa['sapps']=[state[1]]
        aa['vbri']=[state[2]]
        aa['vbrs']=[state[3]]
        aa['trafi']=[state[4]]
        aa['trafs']=[state[5]]
        aa['value']=[value]
# =============================================================================
#         if len(self.memoryunit)==0:
#             action=[0,0,0,0,0,0]
#             aa['sappi_a']=[action[0]]
#             aa['sapps_a']=[action[1]]
#             aa['vbri_a']=[action[2]]
#             aa['vbrs_a']=[action[3]]
#             aa['trafi_a']=[action[4]]
#             aa['trafs_a']=[action[5]]
#             aa['gain']=[0]
#         else:
#             aa['sappi_a']=[state[0]-self.memoryunit['sappi'][len(self.memoryunit['sappi'])-1]]
#             aa['sapps_a']=[state[1]-self.memoryunit['sapps'][len(self.memoryunit['sapps'])-1]]
#             aa['vbri_a']=[state[2]-self.memoryunit['vbri'][len(self.memoryunit['vbri'])-1]]
#             aa['vbrs_a']=[state[3]-self.memoryunit['vbrs'][len(self.memoryunit['vbrs'])-1]]
#             aa['trafi_a']=[state[4]-self.memoryunit['trafi'][len(self.memoryunit['trafi'])-1]]
#             aa['trafs_a']=[state[5]-self.memoryunit['trafs'][len(self.memoryunit['trafs'])-1]]
#             aa['gain']=[value-self.memoryunit['value'][len(self.memoryunit['value'])-1]]
# =============================================================================
        print(aa)
        self.memoryunit=self.memoryunit.append(aa)
        self.memoryunit=self.memoryunit.reset_index(drop=True)
        self.count=self.count+1

        

                        
            
        
        













      
        
        
        
        
        
        
        
        
        
        
        
        
        
