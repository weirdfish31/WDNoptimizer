# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:55:46 2018

@author: WDN

Bayesian Optimization Units
Evaluation Units
ReinforcementLearning Units
主要针对连续数值优化的类
"""
#评估，增强学习记忆单元所需
import math
import WDNexataReader
import numpy as np
import pandas as pd
#贝叶斯优化所需：
import matplotlib.pyplot as plt  
from sklearn.gaussian_process import GaussianProcessRegressor  
from sklearn.gaussian_process.kernels import RBF,Matern, ConstantKernel as C 
from sklearn.ensemble import RandomForestRegressor 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy
from sklearn import mixture
#from sklearn.mixture import GMM
from sklearn.cluster import KMeans
import time
import lhsmdu


class GMMvalueOptimizaitonUnit:
    """
    模型计入了概率过程与格策之间的主观权重，AF函数也相应的进行了高度的可配置设计
    利用高斯混合模型对多目标的value进行建模（模型中分簇的概率并没有进行）
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
        self.obj={}
        self.qosname=[]

    def valuegragher_three(self,data,path,qp,count=0,fita=6,fitb=7):
        """
        绘图，单指标的图与多指标合成的3D图
        目前的画图函数，更新了三种不同颜色的热力图：评估值、概率值、综合HPP模型（目前是两个value平面替代）
        """
        collist=data.columns.values.tolist()
        model1=collist[fita]
        model2=collist[fitb]
        
        "======================================================================"
        qp=qp.tolist()
        qp=np.array([qp])
        npdata=np.array(data)
#        fig = plt.figure()  
        fig = plt.figure(figsize=(19,6))  
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(self.xset,self.yset,self.obj['output_'+model1+'_1'], cmap=plt.get_cmap('magma'),linewidth=0, antialiased=False)
        ax1.plot_wireframe(self.xset,self.yset,self.obj['up_'+model1+'_1'],colors='lightgreen',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax1.plot_wireframe(self.xset,self.yset,self.obj['down_'+model1+'_1'],colors='lightgreen',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
#        ax1.scatter(npdata[:,1],npdata[:,5],npdata[:,7],c='black')  
        ax1.set_title('The value process surface')  
        ax1.set_xlabel('sapps')  
        ax1.set_ylabel('trafs')  
        ax1.set_zlabel('value_0') 
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"       
        ax4 = fig.add_subplot(132, projection='3d')
        ax4.plot_surface(self.xset,self.yset,self.obj['output_'+model2+'_1'], cmap=plt.get_cmap('viridis'),linewidth=0, antialiased=False)
        ax4.plot_wireframe(self.xset,self.yset,self.obj['up_'+model2+'_1'],colors='gold',linewidths=1,  
                               rstride=10, cstride=2, antialiased=True)
        ax4.plot_wireframe(self.xset,self.yset,self.obj['down_'+model2+'_1'],colors='gold',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
#        ax4.scatter(npdata[:,1],npdata[:,5],npdata[:,7],c='black')  
        ax4.set_title('The probability process output')  
        ax4.set_xlabel('sapps')  
        ax4.set_ylabel('trafs')  
        ax4.set_zlabel('probability')
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"       
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_surface(self.xset,self.yset,self.obj['output_value_1'], cmap=plt.get_cmap('rainbow'),linewidth=0, antialiased=False)
        ax3.plot_wireframe(self.xset,self.yset,self.obj['up_value_1'],colors='thistle',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax3.plot_wireframe(self.xset,self.yset,self.obj['down_value_1'],colors='pink',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax3.plot_surface(self.xset,self.yset,self.obj['output_value_0'], cmap=plt.get_cmap('jet'),linewidth=0, antialiased=False)
        ax3.plot_wireframe(self.xset,self.yset,self.obj['up_value_0'],colors='thistle',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax3.plot_wireframe(self.xset,self.yset,self.obj['down_value_0'],colors='pink',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax3.scatter(npdata[:,1],npdata[:,5],npdata[:,7],c='black')  
        ax3.set_title('the predict output at ('+str(qp[0,0])+'  '+str(qp[0,1])+'): {0} '.format(self.reg.predict(qp)[0]))  
        ax3.set_xlabel('sapps')  
        ax3.set_ylabel('trafs')  
        ax3.set_zlabel('value_total') 
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
# =============================================================================
#         ax = fig.add_subplot(122)  
#         s = ax.scatter(npdata[:,1],npdata[:,5],npdata[:,6],cmap=plt.cm.viridis,c='red')
#         im=ax.imshow(self.obj['output_value_1'], interpolation='bilinear', origin='lower',  
#                        extent=(self.xmin, self.xmax-1, self.ymin, self.ymax), aspect='auto')
#         
#         ax.set_title('the predict mean ')  
#         ax.hlines(qp[0,1],self.xmin, self.xmax-1)  
#         ax.vlines(qp[0,0],self.ymin, self.ymax)  
#         ax.set_xlabel('sapps')  
#         ax.set_ylabel('trafs') 
#         plt.colorbar(mappable=im,ax=ax)
# =============================================================================
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        plt.subplots_adjust(left=0.03, top=0.97, right=0.97)
        plt.savefig(path+'HPP_i60_multi'+str(count)+".jpg")
        plt.show() 
        
    def valuegragher_two(self,data,path,qp,count=0):
        """
        绘图，单指标的图与多指标合成的3D图
        迭代过程的绘图，双层热力图与平面query point 采样图
        """
        qp=qp.tolist()
        qp=np.array([qp])
        npdata=np.array(data)
        fig = plt.figure(figsize=(21,10))  
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"       
        ax3 = fig.add_subplot(121, projection='3d')
        ax3.plot_surface(self.xset,self.yset,self.obj['output_value_1'], cmap=plt.get_cmap('rainbow'),linewidth=0, antialiased=False)
        ax3.plot_wireframe(self.xset,self.yset,self.obj['up_value_1'],colors='gold',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax3.plot_wireframe(self.xset,self.yset,self.obj['down_value_1'],colors='lightgreen',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax3.plot_surface(self.xset,self.yset,self.obj['output_value_0'], cmap=plt.get_cmap('rainbow'),linewidth=0, antialiased=False)
        ax3.plot_wireframe(self.xset,self.yset,self.obj['up_value_0'],colors='gold',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax3.plot_wireframe(self.xset,self.yset,self.obj['down_value_0'],colors='lightgreen',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax3.scatter(npdata[:,1],npdata[:,5],npdata[:,7],c='black')  
        ax3.set_title('the predict output at ('+str(qp[0,0])+'  '+str(qp[0,1])+'): {0} '.format(self.reg.predict(qp)[0]))  
        ax3.set_xlabel('sapps')  
        ax3.set_ylabel('trafs')  
        ax3.set_zlabel('value_total') 
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        ax = fig.add_subplot(122)  
        s = ax.scatter(npdata[:,1],npdata[:,5],npdata[:,6],cmap=plt.cm.viridis,c='red')
        im=ax.imshow(self.obj['output_value_1'], interpolation='bilinear', origin='lower',  
                       extent=(self.xmin, self.xmax-1, self.ymin, self.ymax), aspect='auto')
        
        ax.set_title('the predict mean ')  
        ax.hlines(qp[0,1],self.xmin, self.xmax-1)  
        ax.vlines(qp[0,0],self.ymin, self.ymax)  
        ax.set_xlabel('sapps')  
        ax.set_ylabel('trafs')  
        plt.subplots_adjust(left=0.05, top=0.95, right=0.95)
        plt.colorbar(mappable=im,ax=ax)
        
        plt.savefig(path+'GMM_multi'+str(count)+".jpg")
        plt.show() 

    def valuegragher_one(self,data,path,qp,count=0):
        """
        绘图，单指标的图与多指标合成的3D图
        目前没用
        """
        qp=qp.tolist()
        qp=np.array([qp])
        npdata=np.array(data)
#        fig = plt.figure()  
        fig = plt.figure(figsize=(21,10))  
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(self.xset,self.yset,self.obj['output_value_0'], cmap=plt.get_cmap('rainbow'),linewidth=0, antialiased=False)
        ax1.plot_wireframe(self.xset,self.yset,self.obj['up_value_0'],colors='gold',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax1.plot_wireframe(self.xset,self.yset,self.obj['down_value_0'],colors='lightgreen',linewidths=1,  
                                rstride=10, cstride=2, antialiased=True)
        ax1.scatter(npdata[:,1],npdata[:,5],npdata[:,7],c='black')  
        ax1.set_title('the predict output at ('+str(qp[0,0])+'  '+str(qp[0,1])+'): {0} '.format(self.reg.predict(qp)[0]))  
        ax1.set_xlabel('sapps')  
        ax1.set_ylabel('trafs')  
        ax1.set_zlabel('value_0') 
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"       
        ax = fig.add_subplot(122)  
        s = ax.scatter(npdata[:,1],npdata[:,5],npdata[:,6],cmap=plt.cm.viridis,c='red')
        im=ax.imshow(self.obj['output_value_0'], interpolation='bilinear', origin='lower',  
                       extent=(self.xmin, self.xmax-1, self.ymin, self.ymax), aspect='auto')
        ax.set_title('the predict mean ')  
        ax.hlines(qp[0,1],self.xmin, self.xmax-1)  
        ax.vlines(qp[0,0],self.ymin, self.ymax)  
        ax.set_xlabel('sapps')  
        ax.set_ylabel('trafs')  
        plt.subplots_adjust(left=0.05, top=0.95, right=0.95)
        plt.colorbar(mappable=im,ax=ax)
        plt.savefig(path+'GPR_multi'+str(count)+".jpg")
        plt.show() 
        
    def UCBmethodhelper_alpha(self,x,gp,kappa,iternum,count):
        """
        upper confidence bound 方法
        根据随机过程的方差和均值进行选择，不会陷入局部最优 加入了k的衰减因子使k值随着迭代次数变化
        这种做法比较的是置信区间内的最大值，尽管看起来简单，但是实际效果却意外的好
        """
        mean,std=gp.predict(x,return_std=True)
        steplength=kappa/iternum
        a=kappa-count*steplength
        if a<0:
            a=0.1
        return mean + a*std
        
    
    def valueUCBhelper_alpha(self,data,kappa,iternum,count,proportion=1,fitx=1,fity=5,fitz=6):
        """
        将不同聚类得到的预测结果存入dataframe，生成对100000个随机点的预测的reg模型
        value的UCB值相加(概率加权求和)
        则根据聚类得到的权重加权得到UCB之和，得到选择的最大UCB值的query point
        1）在alpha版本中af函数加入了proportion参数，进行两个分粗的重要性的主观评价，更加针对PNTRC系统中的特定QoS性能
        2）提供了最简单的策略自适应的过程，kappa值随着迭代的进行变化（递减）
        3）提供了可调整的相应平面选择参数（）
        在目前的版本中要是出现了抽样的数值为0的情况，仿真会中断
        而且目前只有两簇的情况，没有进行更多簇的考虑
        """
        times  = time.clock() 
        bounds=pd.DataFrame()
        x_tries = np.random.uniform(0, 64000,size=(200000))
        y_tries = np.random.uniform(0, 64000,size=(200000))
        bounds['sapps']=x_tries
        bounds['trafs']=y_tries
        try_data = np.array(bounds)
        collist=data.columns.values.tolist()
        value=collist[fitz]
        "对各簇的模型和进行predict"
        ys0=self.UCBmethodhelper_alpha(try_data,gp=self.obj['reg_'+str(value)+'_'+str(0)],kappa=kappa,iternum=iternum,count=count)
        prob0=self.obj['reg_prob_0'].predict(try_data,return_std=False)
        ys1=self.UCBmethodhelper_alpha(try_data,gp=self.obj['reg_'+str(value)+'_'+str(1)],kappa=kappa,iternum=iternum,count=count)
        prob1=self.obj['reg_prob_1'].predict(try_data,return_std=False)
        "对各簇的概率与预测UCB值进行加权，这里的UCB值和概率都是nparray数据结构"
        UCB=ys0*prob0+proportion*ys1*prob1
        "对UCB中的最大值进行选择，在try_data中得到相应的querypoint"
        try_max=try_data[UCB.argmax()]
        try_max=try_max.astype(int)#为了EXATA配置文件，把询问点改为整数型
        if try_max[0]==0:
            try_max[0]=try_max[0]+1
        if try_max[1]==0:
            try_max[1]=try_max[1]+1
        max_acq=UCB.max()
        print(max_acq)
        print(try_max)
        timee = time.clock()
        rtime = timee - times
        print('the value-AF run time is : %fS' % rtime)
        return try_max 
        
    def valueUCBhelper_HPP(self,data,kappa,fitx=1,fity=5,fitz=6):
        """
        GMM的GMM-UCB模型,固定的策略
        """
        times  = time.clock() 
        bounds=pd.DataFrame()
        x_tries = np.random.uniform(0, 64000,size=(200000))
        y_tries = np.random.uniform(0, 64000,size=(200000))
        bounds['sapps']=x_tries
        bounds['trafs']=y_tries
        try_data = np.array(bounds)
#        componentmodel={}
#        UCBdic={}
        collist=data.columns.values.tolist()
        value=collist[fitz]
        "对各簇的模型和进行predict"
        ys0=self.UCBmethodhelper(try_data,gp=self.obj['reg_'+str(value)+'_'+str(0)],kappa=kappa)
        prob0=self.obj['reg_prob_0'].predict(try_data,return_std=False)
        ys1=self.UCBmethodhelper(try_data,gp=self.obj['reg_'+str(value)+'_'+str(1)],kappa=kappa)
        prob1=self.obj['reg_prob_1'].predict(try_data,return_std=False)
        "对各簇的概率与预测UCB值进行加权，这里的UCB值和概率都是nparray数据结构"
        UCB=ys0*prob0+ys1*prob1
#        aaa=pd.DataFrame(UCBdic)
#        for i in range(aaa.shape[0]):
#            for j in range(self.n_clusters):
#                aaa.iloc[i,j]=aaa.iloc[i,j]*self.componentweight[str(j)]
#        aaa['total']=aaa.apply(lambda x: x.sum(), axis=1)
#        ucbarray=np.array(aaa['total'])
        "对UCB中的最大值进行选择，在try_data中得到相应的querypoint"
        try_max=try_data[UCB.argmax()]
        try_max=try_max.astype(int)#为了EXATA配置文件，把询问点改为整数型
        if try_max[0]==0:
            try_max[0]=try_max[0]+1
        if try_max[1]==0:
            try_max[1]=try_max[1]+1
        max_acq=UCB.max()
        print(max_acq)
        print(try_max)
        timee = time.clock()
        rtime = timee - times
        print('the value-AF run time is : %fS' % rtime)
        return try_max 

    def valueUCBhelper_GPR_state(self,data,kappa,iternum,count,proportion=1,fitz=6):
        times  = time.clock() 
        bounds=pd.DataFrame()
        superappsize = np.random.uniform(16, 64000,size=(200000))
        superappsize = [ math.ceil(x) for x in superappsize ]
        superappsize = [ x+1 for x in superappsize ]
        trafsize = np.random.uniform(16, 64000,size=(200000))
        trafsize = [ math.ceil(x) for x in trafsize ]
        trafsize = [ x+1 for x in trafsize ]        
        superappinterval=np.random.uniform(0, 100,size=(200000))#superapp视频业务，需要的时延抖动小，吞吐量大
        superappinterval = [ math.ceil(x) for x in superappinterval ]
        superappinterval = [ x+1 for x in superappinterval ]
        vbrinterval=np.random.uniform(0, 100,size=(200000))
        vbrinterval = [ math.ceil(x) for x in vbrinterval ]
        vbrinterval = [ x+1 for x in vbrinterval ]        #vbr其他义务
        vbrsize=np.random.uniform(16, 64000,size=(200000))
        vbrsize = [ math.ceil(x) for x in vbrsize ]
        vbrsize = [ x+1 for x in vbrsize ]        
        trafinterval=np.random.uniform(0, 100,size=(200000))#trafficgenerator图像流，需要的丢包率小，吞吐量大
        trafinterval = [ math.ceil(x) for x in trafinterval ]
        trafinterval = [ x+1 for x in trafinterval ]
            
        bounds['superappinterval']=superappinterval
        bounds['superappsize']=superappsize
        bounds['vbrinterval']=vbrinterval
        bounds['vbrsize']=vbrsize
        bounds['trafinterval']=trafinterval
        bounds['trafsize']=trafsize
        
        try_data = np.array(bounds)
        collist=data.columns.values.tolist()
        value=collist[fitz]
        
        
        "对各簇的模型和进行predict"
        ys0=self.UCBmethodhelper_alpha(try_data,gp=self.obj['reg_'+str(value)+'_'+str(0)],kappa=kappa,iternum=iternum,count=count)
#        ys0=self.UCBmethodhelper(try_data,gp=self.obj['reg_'+str(value)+'_'+str(0)],kappa=kappa)
        prob0=self.obj['reg_prob_0'].predict(try_data,return_std=False)
#        ys1=self.UCBmethodhelper(try_data,gp=self.obj['reg_'+str(value)+'_'+str(1)],kappa=kappa)
#        prob1=self.obj['reg_prob_1'].predict(try_data,return_std=False)
        "对各簇的概率与预测UCB值进行加权，这里的UCB值和概率都是nparray数据结构"
        UCB=ys0*prob0
#        aaa=pd.DataFrame(UCBdic)
#        for i in range(aaa.shape[0]):
#            for j in range(self.n_clusters):
#                aaa.iloc[i,j]=aaa.iloc[i,j]*self.componentweight[str(j)]
#        aaa['total']=aaa.apply(lambda x: x.sum(), axis=1)
#        ucbarray=np.array(aaa['total'])
        "对UCB中的最大值进行选择，在try_data中得到相应的querypoint"
        try_max=try_data[UCB.argmax()]
        try_max=try_max.astype(int)#为了EXATA配置文件，把询问点改为整数型
        if try_max[0]==0:
            try_max[0]=try_max[0]+1
        if try_max[1]==0:
            try_max[1]=try_max[1]+1
        if try_max[2]==0:
            try_max[2]=try_max[2]+1
        if try_max[3]==0:
            try_max[3]=try_max[3]+1            
        if try_max[4]==0:
            try_max[4]=try_max[4]+1            
        if try_max[5]==0:
            try_max[5]=try_max[5]+1               
            
        max_acq=UCB.max()
        print(max_acq)
        print(try_max)
        timee = time.clock()
        rtime = timee - times
        print('the value-AF run time is : %fS' % rtime)
        return try_max 

        
        
    
    def valueUCBhelper_GPR(self,data,kappa,iternum,count,proportion=1,fitx=1,fity=5,fitz=6):
        """
        GPR的GP-UCB模型,修改了AF函数，加入了收敛因子
        """
        times  = time.clock() 
        bounds=pd.DataFrame()
        x_tries = np.random.uniform(0, 64000,size=(200000))
        y_tries = np.random.uniform(0, 64000,size=(200000))
        bounds['sapps']=x_tries
        bounds['trafs']=y_tries
        try_data = np.array(bounds)
#        componentmodel={}
#        UCBdic={}
        collist=data.columns.values.tolist()
        value=collist[fitz]
        "对各簇的模型和进行predict"
        ys0=self.UCBmethodhelper_alpha(try_data,gp=self.obj['reg_'+str(value)+'_'+str(0)],kappa=kappa,iternum=iternum,count=count)
#        ys0=self.UCBmethodhelper(try_data,gp=self.obj['reg_'+str(value)+'_'+str(0)],kappa=kappa)
        prob0=self.obj['reg_prob_0'].predict(try_data,return_std=False)
#        ys1=self.UCBmethodhelper(try_data,gp=self.obj['reg_'+str(value)+'_'+str(1)],kappa=kappa)
#        prob1=self.obj['reg_prob_1'].predict(try_data,return_std=False)
        "对各簇的概率与预测UCB值进行加权，这里的UCB值和概率都是nparray数据结构"
        UCB=ys0*prob0
#        aaa=pd.DataFrame(UCBdic)
#        for i in range(aaa.shape[0]):
#            for j in range(self.n_clusters):
#                aaa.iloc[i,j]=aaa.iloc[i,j]*self.componentweight[str(j)]
#        aaa['total']=aaa.apply(lambda x: x.sum(), axis=1)
#        ucbarray=np.array(aaa['total'])
        "对UCB中的最大值进行选择，在try_data中得到相应的querypoint"
        try_max=try_data[UCB.argmax()]
        try_max=try_max.astype(int)#为了EXATA配置文件，把询问点改为整数型
        if try_max[0]==0:
            try_max[0]=try_max[0]+1
        if try_max[1]==0:
            try_max[1]=try_max[1]+1
        max_acq=UCB.max()
        print(max_acq)
        print(try_max)
        timee = time.clock()
        rtime = timee - times
        print('the value-AF run time is : %fS' % rtime)
        return try_max  
    
    def UCBmethodhelper(self,x,gp,kappa):
        """
        upper confidence bound 方法
        根据随机过程的方差和均值进行选择，不会陷入局部最优
        这种做法比较的是置信区间内的最大值，尽管看起来简单，但是实际效果却意外的好
        """
        mean,std=gp.predict(x,return_std=True)
        return mean + kappa*std
        
    def gpbuilder(self,data,fitx=1,fity=5,fitz=6,label=1):
        """
        根据聚类的结果，对用以标签下的数据进行GP回归，得到均值标准差和响应平面。在这里我们用的是3维的过程
        高斯过程的拟合，将GP的相应平面存入实体的obj字典
        """
        collist=data.columns.values.tolist()
        value=collist[fitz]
        self.qosname.append(value)
        testdata=data[data['label']==label]
        testdata=testdata.reset_index(drop=True)
        self.npdata=np.array(testdata)
        self.reg=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
        self.obj['reg_'+value+'_'+str(label)]=self.reg.fit(self.npdata[:,[fitx,fity]],self.npdata[:,fitz])
        self.obj['output_'+value+'_'+str(label)],self.obj['err_'+value+'_'+str(label)]=self.obj['reg_'+value+'_'+str(label)].predict(np.c_[self.xset.ravel(),self.yset.ravel()],return_std=True)
        self.obj['output_'+value+'_'+str(label)],self.obj['err_'+value+'_'+str(label)]=self.obj['output_'+value+'_'+str(label)].reshape(self.xset.shape),self.obj['err_'+value+'_'+str(label)].reshape(self.xset.shape)
#        self.obj['sigma_'+str(label)]=np.sum(self.reg.predict(self.npdata[:,[1,5]],return_std=True)[1])
        self.obj['up_'+value+'_'+str(label)],self.obj['down_'+value+'_'+str(label)]=self.obj['output_'+value+'_'+str(label)]*(1+1.96*self.obj['err_'+value+'_'+str(label)]),self.obj['output_'+value+'_'+str(label)]*(1-1.96*self.obj['err_'+value+'_'+str(label)])
    def gpbuilder_state(self,data,fitz=6,label=1):
        """
        多维度高斯过程，拟合
        """
        collist=data.columns.values.tolist()
        value=collist[fitz]
        self.qosname.append(value)
        testdata=data[data['label']==label]
        testdata=testdata.reset_index(drop=True)
        self.npdata=np.array(testdata)
        self.reg=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
        if fitz==6:
            self.obj['reg_'+value+'_'+str(label)]=self.reg.fit(self.npdata[:,0:fitz],self.npdata[:,fitz])
        elif fitz==7:
            self.obj['reg_'+value+'_'+str(label)]=self.reg.fit(self.npdata[:,0:fitz-1],self.npdata[:,fitz])      

    def valueUCBhelper_HPP_state(self,data,kappa,iternum,count,proportion=1,fitz=6):
        """
        将不同聚类得到的预测结果存入dataframe，生成对100000个随机点的预测的reg模型
        value的UCB值相加(概率加权求和)
        则根据聚类得到的权重加权得到UCB之和，得到选择的最大UCB值的query point
        1）在alpha版本中af函数加入了proportion参数，进行两个分粗的重要性的主观评价，更加针对PNTRC系统中的特定QoS性能
        2）提供了最简单的策略自适应的过程，kappa值随着迭代的进行变化（递减）
        3）提供了可调整的相应平面选择参数（）
        在目前的版本中要是出现了抽样的数值为0的情况，仿真会中断
        而且目前只有两簇的情况，没有进行更多簇的考虑
        多维的UCB策略的拟合
        """
        times  = time.clock() 
        bounds=pd.DataFrame()
        superappsize = np.random.uniform(16, 64000,size=(200000))
        superappsize = [ math.ceil(x) for x in superappsize ]
        superappsize = [ x+1 for x in superappsize ]
        trafsize = np.random.uniform(16, 64000,size=(200000))
        trafsize = [ math.ceil(x) for x in trafsize ]
        trafsize = [ x+1 for x in trafsize ]        
        superappinterval=np.random.uniform(0, 100,size=(200000))#superapp视频业务，需要的时延抖动小，吞吐量大
        superappinterval = [ math.ceil(x) for x in superappinterval ]
        superappinterval = [ x+1 for x in superappinterval ]
        vbrinterval=np.random.uniform(0, 100,size=(200000))
        vbrinterval = [ math.ceil(x) for x in vbrinterval ]
        vbrinterval = [ x+1 for x in vbrinterval ]        #vbr其他义务
        vbrsize=np.random.uniform(16, 64000,size=(200000))
        vbrsize = [ math.ceil(x) for x in vbrsize ]
        vbrsize = [ x+1 for x in vbrsize ]        
        trafinterval=np.random.uniform(0, 100,size=(200000))#trafficgenerator图像流，需要的丢包率小，吞吐量大
        trafinterval = [ math.ceil(x) for x in trafinterval ]
        trafinterval = [ x+1 for x in trafinterval ]
        
      
        bounds['superappinterval']=superappinterval
        bounds['superappsize']=superappsize
        bounds['vbrinterval']=vbrinterval
        bounds['vbrsize']=vbrsize
        bounds['trafinterval']=trafinterval
        bounds['trafsize']=trafsize
        
        try_data = np.array(bounds)
        collist=data.columns.values.tolist()
        value=collist[fitz]
        "对各簇的模型和进行predict"
        ys0=self.UCBmethodhelper_alpha(try_data,gp=self.obj['reg_'+str(value)+'_'+str(0)],kappa=kappa,iternum=iternum,count=count)
        prob0=self.obj['reg_prob_0'].predict(try_data,return_std=False)
        ys1=self.UCBmethodhelper_alpha(try_data,gp=self.obj['reg_'+str(value)+'_'+str(1)],kappa=kappa,iternum=iternum,count=count)
        prob1=self.obj['reg_prob_1'].predict(try_data,return_std=False)
        "对各簇的概率与预测UCB值进行加权，这里的UCB值和概率都是nparray数据结构"
        UCB=ys0*prob0+proportion*ys1*prob1
        "对UCB中的最大值进行选择，在try_data中得到相应的querypoint"
        try_max=try_data[UCB.argmax()]
        try_max=try_max.astype(int)#为了EXATA配置文件，把询问点改为整数型
        if try_max[0]==0:
            try_max[0]=try_max[0]+1
        if try_max[1]==0:
            try_max[1]=try_max[1]+1
        if try_max[2]==0:
            try_max[2]=try_max[2]+1
        if try_max[3]==0:
            try_max[3]=try_max[3]+1            
        if try_max[4]==0:
            try_max[4]=try_max[4]+1            
        if try_max[5]==0:
            try_max[5]=try_max[5]+1               
            
        max_acq=UCB.max()
        print(max_acq)
        print(try_max)
        timee = time.clock()
        rtime = timee - times
        print('the value-AF run time is : %fS' % rtime)
        return try_max 














































































    def rfbuilder(self,data,fitx=1,fity=5,fitz=6,label=1):
        '''
        根据数据进行随机森林回归
        '''
        collist=data.columns.values.tolist()
        value=collist[fitz]
        self.qosname.append(value)
        testdata=data[data['label']==label]
        testdata=testdata.reset_index(drop=True)
        self.npdata=np.array(testdata)
        self.reg=RandomForestRegressor(n_estimators=10,n_jobs=1)
        self.obj['rfreg_'+value+'_'+str(label)]=self.reg.fit(self.npdata[:,[fitx,fity]],self.npdata[:,fitz])
        self.obj['rfoutput_'+value+'_'+str(label)]=self.obj['rfreg_'+value+'_'+str(label)].predict(np.c_[self.xset.ravel(),self.yset.ravel()])
        self.obj['rfoutput_'+value+'_'+str(label)]=self.obj['rfoutput_'+value+'_'+str(label)].reshape(self.xset.shape)
#        self.obj['sigma_'+str(label)]=np.sum(self.reg.predict(self.npdata[:,[1,5]],return_std=True)[1])              
    def rfbuilder_state(self,data,fitz=6,label=1):
        '''
        根据数据进行随机森林回归
        '''
        collist=data.columns.values.tolist()
        value=collist[fitz]
        self.qosname.append(value)
        testdata=data[data['label']==label]
        testdata=testdata.reset_index(drop=True)
        self.npdata=np.array(testdata)
        self.reg=RandomForestRegressor(n_estimators=10,n_jobs=1)
        if fitz==6:
            self.obj['reg_'+value+'_'+str(label)]=self.reg.fit(self.npdata[:,0:fitz],self.npdata[:,fitz])
        elif fitz==7:
            self.obj['reg_'+value+'_'+str(label)]=self.reg.fit(self.npdata[:,0:fitz-1],self.npdata[:,fitz])   



    def componentselecter(self,data,i):
        """
        选择相应的簇类
        """
        testdata=data[data.label==i]
        testdata=testdata.reset_index(drop=True)
        return testdata
    def dropNaNworker(self,data):
        """
        去除nan数据
        """
        testdata=data.dropna(axis=0,how='any')
        testdata=testdata.reset_index(drop=True)
        return testdata        
         
    def presortworker(self,data,col1,col2):
        """
        用来对数据进行排序，排序后进行聚类
        """
        data=data.sort_values(by=[col1,col2])
        data=data.reset_index(drop=True)
        return data
        
    def clusterworker(self,data,col1,col2,count=0):
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
        self.r1=pd.Series(estimator.labels_).value_counts()
        self.samplecount=data.iloc[:,0].size
#        for i in range(self.n_clusters):
#            self.componentweight[str(i)]=self.r1[i]/self.samplecount
        print("The sample count is "+str(self.samplecount))
#        print(self.r1)
        r = pd.concat([data, pd.Series(estimator.labels_, index = data.index)], axis = 1)
        r.rename(columns={0:'label'},inplace=True)
        for i in range(len(c)):
            if int(lable_pred[i])==0:
                plt.scatter(c[i][0],c[i][1],color='red')
            if int(lable_pred[i])==1:
                plt.scatter(c[i][0],c[i][1],color='blue')
            if int(lable_pred[i])==2:
                plt.scatter(c[i][0],c[i][1],color='gold')
            if int(lable_pred[i])==3:
                plt.scatter(c[i][0],c[i][1],color='violet')
#        plt.savefig('./Figure/Cluster'+str(count)+".jpg")#保存聚类图像
        plt.show()
        return r
    
    def EMworker(self,data):
        """
        SKLearn里面自带的EM函数，这里我们只取了数据的二维，方便画图
        目前未完成
        """
        value=np.array(data['value'])
        value1=value.reshape((-1,1))
        trafs=np.array(data['trafs'])
        trafs1=trafs.reshape((-1,1))
        c=np.hstack((trafs1,value1))
        c=c[:,::-1]
        gmm = mixture.GaussianMixture(n_components=2,n_iter=1000).fit(c)
        print(gmm)
        labels = gmm.predict(c)
        print(labels)
        plt.scatter(c[:, 0], c[:, 1],c=labels, s=40, cmap='viridis')
        probs = gmm.predict_proba(c)
        print(probs[:5].round(3))
        size = 50 * probs.max(1) ** 2  # 由圆点面积反应概率值的差异
        plt.scatter(c[:, 0], c[:, 1], c=labels, cmap='viridis', s=size)

    

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GMMmultiOptimizationUnit:
    """
    模型和AF函数有改变：模型进行了GMM的变换，概率为常数
    利用高斯混合模型对多目标的value进行建模（模型中分簇的概率并没有进行）
    针对的是特定的两个QoS性能上进行建模，是多目标优化的问题的迭代解决方案，目前展示未使用
    目前在GMM_throughput_secondorder.py实验中用到此类
    """
    def __init__(self,cluster=2):
        self.xmin,self.xmax=0,64000
        self.ymin,self.ymax=0,64000
        self.xset,self.yset=np.meshgrid(np.arange(self.xmin,self.xmax, 500), np.arange(self.ymin,self.ymax, 500))
        self.n_clusters=cluster
        self.kernel=Matern(nu=2.5)
        self.wirecolor=['mediumpurple','lightgreen','gold','maroon']
        self.componentweight={}#各components的权重，根据样本数分配
        self.obj={}
        self.qosname=[]
    
    def componentselecter(self,data,i):
        testdata=data[data.label==i]
        testdata=testdata.reset_index(drop=True)
        return testdata
    def dropNaNworker(self,data):
        testdata=data.dropna(axis=0,how='any')
        testdata=testdata.reset_index(drop=True)
        return testdata

    def multiGMMbuilder(self,data,fitz=7,fita=16):
        """
        its important
        根据历史选择的Qos指标得到的GMM模型，综合合成一个总体的分布，
        从原来的直接对数据进行权重分配编程对预测后的模型之间的权重分配
        固定的两个指标的合成加权,目前在GMM_throughput_secondorder.py实验中用到
        """
        collist=data.columns.values.tolist()
        value1=collist[fitz]
        value2=collist[fita]
        for i in range(self.n_clusters):
            self.obj['output_total_'+str(i)]=self.obj['output_'+value1+'_'+str(i)]+self.obj['output_'+value2+'_'+str(i)]
            self.obj['err_total_'+str(i)]=self.obj['err_'+value1+'_'+str(i)]+self.obj['err_'+value2+'_'+str(i)]
            self.obj['up_total_'+str(i)],self.obj['down_total_'+str(i)]=self.obj['output_total_'+str(i)]*(1+1.96*self.obj['err_total_'+str(i)]),self.obj['output_total_'+str(i)]*(1-1.96*self.obj['err_total_'+str(i)])
    
    def mulitgragher(self,data,path,test,count=0):
        """
        绘图，单指标的图与多指标合成的3D图
        """
        test=test.tolist()
        test=np.array([test])
        npdata=np.array(data)
        fig = plt.figure(figsize=(21,10))     
        ax3 = fig.add_subplot(121, projection='3d')
        for i in range(self.n_clusters):
            ax3.plot_surface(self.xset,self.yset,self.obj['output_total_'+str(i)], cmap=plt.get_cmap('rainbow'),linewidth=0, antialiased=False)
            ax3.plot_wireframe(self.xset,self.yset,self.obj['up_total_'+str(i)],colors='gold',linewidths=1,  
                                    rstride=10, cstride=2, antialiased=True)
            ax3.plot_wireframe(self.xset,self.yset,self.obj['down_total_'+str(i)],colors='lightgreen',linewidths=1,  
                                    rstride=10, cstride=2, antialiased=True)
#        ax3.scatter(npdata[:,1],npdata[:,5],npdata[:,7],c='black')  
        ax3.set_title('the predict mean output at ('+str(test[0,0])+'  '+str(test[0,1])+'): {0} '.format(self.reg.predict(test)[0]))  
        ax3.set_xlabel('sapps')  
        ax3.set_ylabel('trafs')  
        ax3.set_zlabel('value') 
        ax = fig.add_subplot(122)  
        s = ax.scatter(npdata[:,1],npdata[:,5],npdata[:,6],cmap=plt.cm.viridis,c='red')
        im=ax.imshow(self.obj['output_total_0'], interpolation='bilinear', origin='lower',  
                       extent=(self.xmin, self.xmax-1, self.ymin, self.ymax), aspect='auto')         
        ax.set_title('the predict mean ')  
        ax.hlines(test[0,1],self.xmin, self.xmax-1)  
        ax.vlines(test[0,0],self.ymin, self.ymax)  
        ax.set_xlabel('sapps')  
        ax.set_ylabel('trafs')  
        plt.subplots_adjust(left=0.05, top=0.95, right=0.95)
        plt.colorbar(mappable=im,ax=ax)
        
        plt.savefig(path+'GMM_multi'+str(count)+".jpg")
        plt.show() 
        
    
    def gmmbuilder(self,data,fitx=1,fity=5,fitz=6):
        """
        根据聚类的结果，对用以标签下的数据进行GP回归，得到均值标准差
        """
        collist=data.columns.values.tolist()
        value=collist[fitz]
        self.qosname.append(value)
        for i in range(self.n_clusters):
            testdata=data[data['label']==i]
            testdata=testdata.reset_index(drop=True)
            self.npdata=np.array(testdata)
            self.reg=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
            self.reg.fit(self.npdata[:,[fitx,fity]],self.npdata[:,fitz])
            self.obj['output_'+value+'_'+str(i)],self.obj['err_'+value+'_'+str(i)]=self.reg.predict(np.c_[self.xset.ravel(),self.yset.ravel()],return_std=True)
            self.obj['output_'+value+'_'+str(i)],self.obj['err_'+value+'_'+str(i)]=self.obj['output_'+value+'_'+str(i)].reshape(self.xset.shape),self.obj['err_'+value+'_'+str(i)].reshape(self.xset.shape)
            self.obj['sigma_'+str(i)]=np.sum(self.reg.predict(self.npdata[:,[1,5]],return_std=True)[1])
            self.obj['up_'+value+'_'+str(i)],self.obj['down_'+value+'_'+str(i)]=self.obj['output_'+value+'_'+str(i)]*(1+1.96*self.obj['err_'+value+'_'+str(i)]),self.obj['output_'+value+'_'+str(i)]*(1-1.96*self.obj['err_'+value+'_'+str(i)])
    
    def querypredicter(self,data,querypoint,fitx=1,fity=5,fitz=9,fita=17):
        """
        1)引入数据对模型进行建模《两个模型superapp_throughput,trafficgenerator_throughput
        2）对query point 进行预测
        3）加权*
        目前在画图程序Drawer_compare_predit_simulate_0.py中使用
        """
        collist=data.columns.values.tolist()
        value=collist[fitz]
        self.qosname.append(value)
        componentmodel={}
        clusterpredictmean=[]
        print(type(querypoint))
        for i in range(self.n_clusters):
            testdata=data[data['label']==i]
            testdata=testdata.reset_index(drop=True)
            npdata=np.array(testdata)
            componentmodel['reg'+str(fitz)+str(i)]=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
            componentmodel['reg'+str(fitz)+str(i)].fit(npdata[:,[fitx,fity]],npdata[:,fitz])
            mean1=componentmodel['reg'+str(fitz)+str(i)].predict(querypoint)
            componentmodel['reg'+str(fita)+str(i)]=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
            componentmodel['reg'+str(fita)+str(i)].fit(npdata[:,[fitx,fity]],npdata[:,fita])
            mean2=componentmodel['reg'+str(fita)+str(i)].predict(querypoint)
            totalmean=mean1+mean2
            clusterpredictmean.append(totalmean)
        return clusterpredictmean
            
    def multiUCBhelper(self,data,kappa,fitx=1,fity=5,fitz=7,fita=16):
        """
        将不同聚类得到的预测结果存入dataframe，生成对100000个随机点的预测的reg模型
        不同指标的UCB值相加
        则根据聚类得到的权重加权得到UCB之和，得到选择的最大UCB值的query point
        """
        times  = time.clock() 
        bounds=pd.DataFrame()
        x_tries = np.random.uniform(0, 64000,size=(100000))
        y_tries = np.random.uniform(0, 64000,size=(100000))
        bounds['sapps']=x_tries
        bounds['trafs']=y_tries
        try_data = np.array(bounds)
        componentmodel={}
        UCBdic={}
        for i in range(self.n_clusters):
            testdata=data[data['label']==i]
            testdata=testdata.reset_index(drop=True)
            npdata=np.array(testdata)
            componentmodel['reg'+str(fitz)+str(i)]=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
            componentmodel['reg'+str(fitz)+str(i)].fit(npdata[:,[fitx,fity]],npdata[:,fitz])
            componentmodel['reg'+str(fita)+str(i)]=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
            componentmodel['reg'+str(fita)+str(i)].fit(npdata[:,[fitx,fity]],npdata[:,fita])
            ys=self.UCBmethodhelper(try_data,gp=componentmodel['reg'+str(fitz)+str(i)],kappa=kappa)+self.UCBmethodhelper(try_data,gp=componentmodel['reg'+str(fita)+str(i)],kappa=kappa)
            UCBdic["ucb"+str(i)]=ys
        aaa=pd.DataFrame(UCBdic)
        for i in range(aaa.shape[0]):
            for j in range(self.n_clusters):
                aaa.iloc[i,j]=aaa.iloc[i,j]*self.componentweight[str(j)]
        aaa['total']=aaa.apply(lambda x: x.sum(), axis=1)
        ucbarray=np.array(aaa['total'])
        try_max=try_data[ucbarray.argmax()]
        try_max=try_max.astype(int)#为了EXATA配置文件，把询问点改为整数型
        max_acq=ucbarray.max()
        print(max_acq)
        print(try_max)
        timee = time.clock()
        rtime = timee - times
        print('the multi-AF run time is : %fS' % rtime)
        return try_max    
    
    def heatgragher(self,data,test,fitz=6):
        """
        绘制热力图和预测的下一个点，坐标是自适应的
        """
        test=test.tolist()
        test=np.array([test])
        collist=data.columns.values.tolist()
        value=collist[fitz]
        npdata=np.array(data)
        fig = plt.figure(figsize=(21,10))  
        ax1 = fig.add_subplot(121, projection='3d')
        for i in range(self.n_clusters):
            ax1.plot_surface(self.xset,self.yset,self.obj['output_'+value+'_'+str(i)], cmap=plt.get_cmap('rainbow'),linewidth=0, antialiased=False)
            ax1.plot_wireframe(self.xset,self.yset,self.obj['up_'+value+'_'+str(i)],colors=self.wirecolor[i],linewidths=1,  
                                    rstride=10, cstride=2, antialiased=True)
            ax1.plot_wireframe(self.xset,self.yset,self.obj['down_'+value+'_'+str(i)],colors=self.wirecolor[i],linewidths=1,  
                                    rstride=10, cstride=2, antialiased=True)
        ax1.scatter(npdata[:,1],npdata[:,5],npdata[:,6],c='black')  
        ax1.set_title('the predict mean output at ('+str(test[0,0])+'  '+str(test[0,1])+'): {0} '.format(self.reg.predict(test)[0]))  
        ax1.set_xlabel('sapps')  
        ax1.set_ylabel('trafs')  
        ax1.set_zlabel('value') 
        """
        单层的热力图目前GMM模型不好画
        """
#        ax = fig.add_subplot(122)  
#        s = ax.scatter(self.npdata[:,1],self.npdata[:,5],self.npdata[:,6],cmap=plt.cm.viridis,c='red')
#        im=ax.imshow(self.obj['output_'+value+'_1'], interpolation='bilinear', origin='lower',  
#                       extent=(self.xmin, self.xmax-1, self.ymin, self.ymax), aspect='auto')
#          
#        ax.set_title('the predict mean ')  
#        ax.hlines(test[0,1],self.xmin, self.xmax-1)  
#        ax.vlines(test[0,0],self.ymin, self.ymax)  
#        ax.text(test[0,0],test[0,1],'{0}'.format(self.reg.predict(test)[0]),ha='left',
#                va='bottom',color='k',size=15,rotation=0)  
#        ax.set_xlabel('sapps')  
#        ax.set_ylabel('trafs')  
#        plt.subplots_adjust(left=0.05, top=0.95, right=0.95)
#        plt.colorbar(mappable=im,ax=ax)
        plt.show() 
    def weightchanger(self,data):
        """
        对于已经进行聚类之后的数据，需要加入这个功能对每个簇的模型的权重进行修改
        这里的各簇的概率是一样的，概率为一常数
        """
        value=np.array(data['vbrs'])
        value1=value.reshape((-1,1))
        trafs=np.array(data['label'])
        trafs1=trafs.reshape((-1,1))
        c=np.hstack((trafs1,value1))
        c=c[:,::-1]
        estimator=KMeans(n_clusters=self.n_clusters,max_iter=1000)
        estimator.fit(c)
        #统计各类的数据的数目
        self.r1=pd.Series(estimator.labels_).value_counts()
        self.samplecount=data.iloc[:,0].size
        for i in range(self.n_clusters):
            self.componentweight[str(i)]=self.r1[i]/self.samplecount
    
    def UCBmethodhelper(self,x,gp,kappa):
        """
        upper confidence bound 方法
        根据随机过程的方差和均值进行选择，不会陷入局部最优
        这种做法比较的是置信区间内的最大值，尽管看起来简单，但是实际效果却意外的好
        """
        mean,std=gp.predict(x,return_std=True)
        return mean + kappa*std
    
    def acquisitionfunctionmethod2(self,data,kappa,fitx=1,fity=5,fitz=6):
        """
        设计2
        将不同聚类的得到的预测结果存入dataframe，
        则根据聚类得到的权重加权得到UCB之和，得到选择的最大UCB值的query point
        """
        times  = time.clock() 
        bounds=pd.DataFrame()
        x_tries = np.random.uniform(0, 64000,size=(100000))
        y_tries = np.random.uniform(0, 64000,size=(100000))
        bounds['sapps']=x_tries
        bounds['trafs']=y_tries
        try_data = np.array(bounds)
        componentmodel={}
        UCBdic={}
        for i in range(self.n_clusters):
            testdata=data[data['label']==i]
            testdata=testdata.reset_index(drop=True)
            npdata=np.array(testdata)
            componentmodel['reg'+str(i)]=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
            componentmodel['reg'+str(i)].fit(npdata[:,[fitx,fity]],npdata[:,fitz])
            ys=self.UCBmethodhelper(try_data,gp=componentmodel['reg'+str(i)],kappa=kappa)
            UCBdic["ucb"+str(i)]=ys
        aaa=pd.DataFrame(UCBdic)
        for i in range(aaa.shape[0]):
            for j in range(self.n_clusters):
                aaa.iloc[i,j]=aaa.iloc[i,j]*self.componentweight[str(j)]
        aaa['total']=aaa.apply(lambda x: x.sum(), axis=1)
        ucbarray=np.array(aaa['total'])
        try_max=try_data[ucbarray.argmax()]
        try_max=try_max.astype(int)#为了EXATA配置文件，把询问点改为整数型
        max_acq=ucbarray.max()
        print(max_acq)
        print(try_max)
        timee = time.clock()
        rtime = timee - times
        print('the AFmethod2 run time is : %fS' % rtime)
        return try_max
                
    
    def acquisitionfunctionmethod1(self,data,kappa,fitx=1,fity=5,fitz=6):
        """
        设计1：
        1）先随机选取100000点进行估计，得到每个components上的UCB值最大的坐标
        2）得到n个query point
        目前仿真没办法一次仿真几个，这个方法暂时用不了
        """
        times  = time.clock() 
        bounds=pd.DataFrame()
        x_tries = np.random.uniform(0, 64000,size=(100000))
        y_tries = np.random.uniform(0, 64000,size=(100000))
        bounds['sapps']=x_tries
        bounds['trafs']=y_tries
        try_data = np.array(bounds)
        aaa=pd.DataFrame()
        for i in range(self.n_clusters):
            testdata=data[data['label']==i]
            testdata=testdata.reset_index(drop=True)
            npdata=np.array(testdata)
            reg=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
            reg.fit(npdata[:,[fitx,fity]],npdata[:,fitz])
            ys=self.UCBmethodhelper(try_data,gp=reg,kappa=kappa)
            try_max=try_data[ys.argmax()]
            try_max=try_max.astype(int)#为了EXATA配置文件，把询问点改为整数型
            max_acq=ys.max()
            print(try_max)
            print(max_acq)
            test =pd.DataFrame([try_max])
            aaa=aaa.append(test)
        aaa=aaa.reset_index(drop=True)
        timee = time.clock()
        rtime = timee - times
        print('theAFmethod1 run time is : %fS' % rtime)
        return aaa
    
    def presortworker(self,data,col1,col2):
        """
        用来对数据进行排序，排序后进行聚类
        """
        data=data.sort_values(by=[col1,col2])
        data=data.reset_index(drop=True)
        return data
        
    def clusterworker(self,data,col1,col2,count=0):
        """
        SKlearn自带的KMeans函数，这里我们只取了数据的二维，方便画图
        KMEANs聚类方式
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
        self.r1=pd.Series(estimator.labels_).value_counts()
        self.samplecount=data.iloc[:,0].size
        for i in range(self.n_clusters):
            self.componentweight[str(i)]=self.r1[i]/self.samplecount
        print("The sample count is "+str(self.samplecount))
        print(self.r1)
        r = pd.concat([data, pd.Series(estimator.labels_, index = data.index)], axis = 1)
        r.rename(columns={0:'label'},inplace=True)
        for i in range(len(c)):
            if int(lable_pred[i])==0:
                plt.scatter(c[i][0],c[i][1],color='red')
            if int(lable_pred[i])==1:
                plt.scatter(c[i][0],c[i][1],color='blue')
            if int(lable_pred[i])==2:
                plt.scatter(c[i][0],c[i][1],color='gold')
            if int(lable_pred[i])==3:
                plt.scatter(c[i][0],c[i][1],color='violet')
#        plt.savefig('./Figure/Cluster_Kmeans_'+str(count)+".jpg")
        plt.show()
        return r
    
    def EMworker(self,data,col1='value',col2='trafs',count=0):
        """
        SKLearn里面自带的EM函数，这里我们只取了数据的二维，方便画图
        EM聚类标签方式,未完成，目前没用到EM
        """
        value=np.array(data[col1])
        value1=value.reshape((-1,1))
        trafs=np.array(data[col2])
        trafs1=trafs.reshape((-1,1))
        c=np.hstack((trafs1,value1))
        c=c[:,::-1]
        gmm = mixture.GaussianMixture(n_components=2,n_iter=1000).fit(c)
        print(gmm)
        labels = gmm.predict(c)
        print(labels)
        plt.scatter(c[:, 0], c[:, 1],c=labels, s=40, cmap='viridis')
        probs = gmm.predict_proba(c)
        print(probs[:5].round(3))
        size = 50 * probs.max(1) ** 2  # 由圆点面积反应概率值的差异
        plt.scatter(c[:, 0], c[:, 1], c=labels, cmap='viridis', s=size)
#        plt.savefig('./Figure/Cluster_EM_'+str(count)+".jpg")
        plt.show()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class BayesianOptimizationUnit:
    """
    贝叶斯优化类：最原初的BO类，使用的是GP过程
    用作进行对业务层设计参数与输出的评分之间建立概率模型，进行设计指导
    这个函数是针对业务层输入参数进行设计的，数据结构不是针对所有的数据
    画图函数是三维的函数，变量连续的变量,核函数是matern函数，nu=2.5（多维空间中的nu为2.5） 后续做成可调整的 
    x为superapp的包大小（0,64000）
    y是trafficgenerator的包大小（0,64000）
    z是系统综合评分value
    目前这个类中的GPR模型只适用于对单目标（评估值）进行优化，
    在做GMMM模型之间的多目标优化问题的时候需要用GMM模型（多簇）和GMM模型（单簇退化为高斯过程模型）进行比较
    """
    def dropNaNworker(self,data):
        testdata=data.dropna(axis=0,how='any')
        testdata=testdata.reset_index(drop=True)
        return testdata
    
    def __init__(self):
        """
        初始化BOA可函数等参数
        """
        self.kernel = Matern(nu=2.5)
        self.xmin,self.xmax=0,64000
        self.ymin,self.ymax=0,64000
        self.xset,self.yset=np.meshgrid(np.arange(self.xmin,self.xmax, 500), np.arange(self.ymin,self.ymax, 500))
        self.componentweight={}#各components的权重，根据样本数分配
        self.obj={}
        self.qosname=[]
    
    def gussianproccessfitter(self,data,fitx=1,fity=5,fitz=6):
        """
        输入dataframe，进行GP拟合
        output:预测的均值Mean of predictive distribution a query points
        err:预测的标准差Standard deviation of predictive distribution at query points. Only returned when return_std is True.
        """
        self.train_data=np.array(data)
        self.reg=GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=10,alpha=0.1)
        self.reg.fit(self.train_data[:,[fitx,fity]],self.train_data[:,fitz])
        self.output,self.err=self.reg.predict(np.c_[self.xset.ravel(),self.yset.ravel()],return_std=True)
        self.output,self.err=self.output.reshape(self.xset.shape),self.err.reshape(self.xset.shape)
        self.sigma=np.sum(self.reg.predict(self.train_data[:,[1,5]],return_std=True)[1])
        self.up,self.down=self.output*(1+1.96*self.err),self.output*(1-1.96*self.err)        
        
        
    def heatpointer(self,test):
        """
        绘制热力图和预测的下一个点，坐标是自适应的
        最原初得的GP版本的绘图，对于输出的控制需要进行修改
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
        这个af函数是最原初得的UCBmethod版本，用到了拟牛顿法，直接将最简单的GP过程进行搜索，目前的实验中用来进行比较试验
        """
        bounds=pd.DataFrame()
        x_tries = np.random.uniform(0, 64000,size=(100000))
        y_tries = np.random.uniform(0, 64000,size=(100000))
        bounds['sapps']=x_tries
        bounds['trafs']=y_tries
        try_data = np.array(bounds)
        ys=self.UCBmethod(try_data,gp=self.reg,kappa=kappa)
        try_max=try_data[ys.argmax()]
        try_max=try_max.astype(int)#为了EXATA配置文件，把询问点改为整数型
        max_acq=ys.max()
        print(try_max)
        print(max_acq)
        try:
            bounds_throughly=pd.DataFrame()
            x_tries_throughly = np.random.uniform(0, 64000,size=(50))
            y_tries_throughly = np.random.uniform(0, 64000,size=(50))
            bounds_throughly['sapps']=x_tries_throughly
            bounds_throughly['trafs']=y_tries_throughly
            try_data_throughly = np.array(bounds_throughly)
            for x_try in try_data_throughly:
                # Find the minimum of minus the acquisition function
                res = scipy.optimize.minimize(lambda x: -self.UCBmethod(x.reshape(-1, 2), gp=self.reg,kappa=kappa),
                               x_try.reshape(-1, 2),
                               bounds=((0,64000),(0,64000)),
                               method="L-BFGS-B")
                if max_acq is None or -res.fun[0] >= max_acq:
                    try_max = res.x
                    try_max=try_max.astype(int)#为了EXATA配置文件，把询问点改为整数型
                    max_acq = -res.fun[0]
        except:
            print('L-BFGS-B didn\'t work!!!')
        test =np.array([try_max])
        print(test)
        return test
        
    def insertnewworker(self,test,dataframe,path='./OutConfigfile/',vbrs_i=24000):
        """
        将acquisitionFunction选取的点仿真后得到的数据库读取出来，加入到先验数据中
        """
        predictstate=[20,int(test[0,0]+1),30,int(vbrs_i),30,int(test[0,1]+1)]
        predictdataset='radio REQUEST-SIZE DET '+str(int(test[0,0]+1))+' _ '+str(vbrs_i)+' _ RND DET '+str(int(test[0,1]+1))          
        readdb1=WDNexataReader.ExataDBreader()
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
    目前所有的评估指标权重相等：  分为三种业务：其他业务流vbr，视频流superapp，图像流trafficgenerator
                                每种业务分四种指标：时延delay、时延抖动jitter、消息完成率messagecompleterate（mcr）、吞吐量throughput
                                构建4*3的评估权重矩阵self.weight
    得到最终的每次仿真的评估值value
    """
    def __init__(self):
        self.weight= np.array([[0.0425,0.0425,0.0425,0.0425],
                               [0.045,0.18,0.045,0.18],
                               [0.025,0.016,0.16,0.25]], dtype=float) 
        self.normalizedata=[]#用来存储归一化结果，格式为dataframe，用来提供给计算评估值
        self.qoslist=[]#用来存储不同业务的qos指标归一化结果，格式为list直接为外部调用
    def calculateMetricEvaValue(self,dataset):
        """
        归一化处理：对于四种不同的指标，进行不同的归一化处理
        这里的归一化处理很重要，在不同量纲之间与不同QoS指标之间如何要比较，需要在最初进行主观判定
        """
        #delay  x=3,y=0.5,flag=0
        delay = dataset['delay']
        a = self.exponentNormalizer(x=0.25,y=0.5,flag=0,value=delay)
        #Jitter xmax=0.1,y=0.001,flag=0
        jitter = dataset['jitter']
        b = self.exponentNormalizer(x=0.1,y=0.001,flag=0,value=jitter)
        #massagecompleterate x=0.9,y=0.1 flag=0
        packetoss = 1-dataset['messagecompletionrate']
        c = self.exponentNormalizer(x=0.9,y=0.1,flag=0,value=packetoss)
        #throughput max=100000000
        throughput = dataset['throughput']
        if throughput>12000000:
            d = 1
        else:
            d = self.sqrtNormalizer(x=12000000,value=throughput)
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
class MemoryUnit:
    """
    记忆单元类：
    现实现了state数据的存储功能包括probability，value，label，state，各种业务流的qos性能均值
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
        self.probmemoryunit=pd.DataFrame()

        
    def probinserter(self,state,value,prob,label):
        """
        添加状态、均值、概率、标签在记忆单元
        目前的业务设计参数有6种，随着以后的进行应该会增加
        添加状态（state），评估均值（value），各簇概率（prob），标签（label）
        保存在probmemoryunit中
        """
        aa=pd.DataFrame()
        aa['sappi']=[state[0]]
        aa['sapps']=[state[1]]
        aa['vbri']=[state[2]]
        aa['vbrs']=[state[3]]
        aa['trafi']=[state[4]]
        aa['trafs']=[state[5]]
        aa['value']=[value]
        aa['prob']=[prob]
        aa['label']=[label]
        self.probmemoryunit=self.probmemoryunit.append(aa)
        self.probmemoryunit=self.probmemoryunit.reset_index(drop=True)
        self.count=self.count+1
        
        
        
    def qosinserter(self,state,qos):
        """
        添加状态、QoS指标的归一化值（state6个设计指标,qos12个指标）
        保存在qosmemoryunit中
        """
        aa=pd.DataFrame()
        aa['sappi']=[state[0]]
        aa['sapps']=[state[1]]
        aa['vbri']=[state[2]]
        aa['vbrs']=[state[3]]
        aa['trafi']=[state[4]]
        aa['trafs']=[state[5]]
        ""
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

        self.qosmemoryunit=self.qosmemoryunit.append(aa)
        self.qosmemoryunit=self.qosmemoryunit.reset_index(drop=True)
        self.count=self.count+1
        
    def valueinserter(self,state,value):
        """
        添加状态与评估值在记忆单元
        保存在memoryunit中
        """
        aa=pd.DataFrame()
        aa['sappi']=[state[0]]
        aa['sapps']=[state[1]]
        aa['vbri']=[state[2]]
        aa['vbrs']=[state[3]]
        aa['trafi']=[state[4]]
        aa['trafs']=[state[5]]
        aa['value']=[value]

        self.memoryunit=self.memoryunit.append(aa)
        self.memoryunit=self.memoryunit.reset_index(drop=True)
        self.count=self.count+1

        

                        
            
        
        













      
        
        
        
        
        
        
        
        
        
        
        
        
        
