# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:29:21 2018

@author: WDN
1）不同模型之间的相同AF策略的比较
2）相同模型之间，模型预测值与仿真真实之之间的比较
3）相同模型的不同AF1策略之间的比较
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

class WDNcompreUnit:
    """
    数据处理
    """
    def __init___(self):

        self.GPRpath="G:/testData/1cluster_3times_GPR/"
        self.GMMpath="G:/testdata/2cluster_10times_GMM"
# =============================================================================
#         self.figpath="./Figure/"
#         self.xmin,self.xmax=0,64000
#         self.ymin,self.ymax=0,64000
# =============================================================================
        self.xset,self.yset=np.meshgrid(np.arange(self.xmin,self.xmax, 500), np.arange(self.ymin,self.ymax, 500))
    
    def querylistfilter(self,list):
        """
        对每组的querypoint进行数据提取，得到每个店的坐标序列和均值的坐标序列
        """
        querypoint=pd.DataFrame(list)
        qx = np.array(querypoint[0])
        qx=qx.tolist()
        qy = np.array(querypoint[1])
        qy=qy.tolist()
        mx=[]
        my=[]
        for i in range(len(qx)):
            mxi=sum(qx[:i])/(i+1)
            myi=sum(qy[:i])/(i+1)
            mx.append(mxi)
            my.append(myi)
        return qx,qy,mx,my
    
    def comparedrawer(self,listgpr,listdgpr,listgmm,listdgmm,):
        """
        对两组不同的querylist进行迭代列表的比较
        """
        qx,qy,mx,my=self.querylistfilter(listgpr)
        dx,dy,useless1,useless2=self.querylistfilter(listdgpr)
        qxgmm,qygmm,mxgmm,mygmm=self.querylistfilter(listgmm)
        dxgmm,dygmm,useless3,uesless4=self.querylistfilter(listdgmm)
        
        for i in range(len(qx)):
            plt.figure('Line fig',figsize=(21,10))
            #设置x轴、y轴名称
            plt.subplot(121)
            plt.xlabel('superapp packet size')
            plt.ylabel('trafficgenerator packet size')
            plt.title('GPR Query Point Selection',fontsize='xx-large')
            plt.xlim(0, 65000)
            plt.ylim(0, 65000)
            plt.plot(qx[:i], qy[:i], color='r', linewidth=1, alpha=0.6,label='Query Trace')
            plt.plot(mx[:i],my[:i],color='black',linewidth=1,alpha=0.3,linestyle='--',label='Mean Trace')
            plt.scatter(qx[:i],qy[:i],s=75,c='blue',marker='.',label='Query List')
            plt.scatter(mx[:i],my[:i],s=30,c='gray',marker='o')
            plt.scatter(qx[i],qy[i],s=180,c='red',marker='*',label='Next Point')
            plt.scatter(mx[i],my[i],s=100,c='black',marker='o',label='Mean point')
            plt.legend(loc=6,fontsize='x-large')
            #设置x轴、y轴名称
            plt.subplot(122)
            plt.xlabel('superapp packet size')
            plt.ylabel('trafficgenerator packet size')
            plt.title('GMM Query Point Selection',fontsize='xx-large')
            plt.xlim(0, 65000)
            plt.ylim(0, 65000)
            plt.plot(qxgmm[:i], qygmm[:i], color='r', linewidth=1, alpha=0.6,label='Query Trace')
            plt.plot(mxgmm[:i],mygmm[:i],color='black',linewidth=1,alpha=0.3,linestyle='--',label='Mean Trace')
            plt.scatter(qxgmm[:i],qygmm[:i],s=75,c='blue',marker='.',label='Query List')
            plt.scatter(mxgmm[:i],mygmm[:i],s=30,c='gray',marker='o')
            plt.scatter(qxgmm[i],qygmm[i],s=180,c='red',marker='*',label='Next Point')
            plt.scatter(mxgmm[i],mygmm[i],s=100,c='black',marker='o',label='Mean point')
            plt.legend(loc=6,fontsize='x-large')
            plt.savefig('./Figure/'+'querypoint'+str(i)+".jpg")
            plt.show()
        





    def iterresultcompare(self,querylist):
        """
        根据已知的querypointlist，读取已有的仿真数据，得到每次迭代之后模型，进而得到具体点的预测的均值
        再根据历史的仿真的数据，进行再每次迭代的情况下，仿真的真实数据与预测的数据之间的比较
        """
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    