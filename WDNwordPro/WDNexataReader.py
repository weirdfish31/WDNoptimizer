# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 10:28:19 2018

@author: WDN
Datareader Units
读取原始数据和数据处理函数
"""
from __future__ import print_function
import sqlite3
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns 
import numpy as np 

class ExataDBreader:
    """
    打开EXATA仿真之后生成的DB文件
    并进行一定的分类：1）业务种类之间的分类2）节点之间的分类
    绘制一定的统计图表：直方图2D，3D，箱型图等
    """
    def opendataset(self,dataset,path):
        """
        打开DB文件
        open the dataset type .DB
        """
        self.conn=sqlite3.connect(path+dataset+'.db')
        self.c = self.conn.cursor()
        
    def __init__(self):
        """
        Initialization
        """
        self.appname=set([])
        self.appsent={}
        self.appreceived={}
        self.apppersent={}
        self.appperreceived={}
        self.appofferedload={}
        self.appthroughput={}
        self.appmessagecompletionrate={}
        self.appdelay={}
        self.appjitter={}
        self.apphopcount={}
        
        self.superappname=[]
        self.ftpname=[]
        self.vbrname=[]
        self.trafficgenname=[]
        
        self.appinterval={}
        self.apppacketsize={}
        
    def appnamereader(self):
        """
        读取DB文件中所有的业务名称
        read all the name of app
        """
        cursor=self.c.execute("SELECT ApplicationName  from APPLICATION_Summary")
        for row in cursor:
            self.appname.add(row[0])
        print(self.appname)
    
    def appfilter(self):
        """
        业务分类，目前三大类：trafficgenerator，vbr，superapp
        This function is about to difference the type of app into four different list
        """
        for app in self.appname:
            if (app.find('VBR') == -1 and app.find('FTP')==-1 and app.find('TrafficGen')==-1):
                self.superappname.append(str(app))
                self.vbrname.append(app+'-VBR')
                self.ftpname.append(app+'-FTP')
                self.trafficgenname.append(app+'-TrafficGen')
            else:
                continue
        
    def flowaggregator(self,sappi,sapps,vbri,vbrs,trafi,trafs):
        """
        按照节点，将业务流聚合在一起
        in the traffic-flow vision
        aggregate the four different app into one traffic flow
        in order to discover the relationship between the correlate parameters
        """
        times = time.clock()
        flowaggregator=pd.DataFrame()
        totalreceived = []
        totalsent = []
        offeredload = []
        throughput = []
        perreceived=[]
        persent=[]
        messagecompletionrate = []
        delay =[]
        jitter=[]
        hopcount=[]
        for app in self.superappname:
            totalreceived=totalreceived+(np.array(self.appreceived[app])+
                                         np.array(self.appreceived[app+'-VBR'])+
                                         np.array(self.appreceived[app+'-TrafficGen'])).tolist()
            totalsent=totalsent+(np.array(self.appsent[app])+
                                 np.array(self.appsent[app+'-VBR'])+
                                 np.array(self.appsent[app+'-TrafficGen'])).tolist()
            offeredload=offeredload+(np.array(self.appofferedload[app])+
                                     np.array(self.appofferedload[app+'-VBR'])+
                                     np.array(self.appofferedload[app+'-TrafficGen'])).tolist()
            throughput=throughput+(np.array(self.appthroughput[app])+
                                   np.array(self.appthroughput[app+'-VBR'])+
                                   np.array(self.appthroughput[app+'-TrafficGen'])).tolist()
            perreceived=perreceived+(np.array(self.appperreceived[app])+
                                     np.array(self.appperreceived[app+'-VBR'])+
                                     np.array(self.appperreceived[app+'-TrafficGen'])).tolist()
            persent=persent+(np.array(self.apppersent[app])+
                             np.array(self.apppersent[app+'-VBR'])+
                             np.array(self.apppersent[app+'-TrafficGen'])).tolist()
            messagecompletionrate=messagecompletionrate+(np.array(self.appmessagecompletionrate[app])+
                                                         np.array(self.appmessagecompletionrate[app+'-VBR'])+
                                                         np.array(self.appmessagecompletionrate[app+'-TrafficGen'])).tolist()
            delay=delay+(np.array(self.appdelay[app])+
                         np.array(self.appdelay[app+'-VBR'])+
                         np.array(self.appdelay[app+'-TrafficGen'])).tolist()
            jitter=jitter+(np.array(self.appjitter[app])+
                           np.array(self.appjitter[app+'-VBR'])+
                           np.array(self.appjitter[app+'-TrafficGen'])).tolist()
            hopcount=hopcount+(np.array(self.apphopcount[app])+
                               np.array(self.apphopcount[app+'-VBR'])+
                               np.array(self.apphopcount[app+'-TrafficGen'])).tolist()
        lens=len(hopcount)
        flowaggregator['sappi']=[sappi]*lens
        flowaggregator['sapps']=[sapps]*lens
        flowaggregator['vbri']=[vbri]*lens
        flowaggregator['vbrs']=[vbrs]*lens
        flowaggregator['trafi']=[trafi]*lens
        flowaggregator['trafs']=[trafs]*lens
        
        flowaggregator['received']=totalreceived
        flowaggregator['sent']=totalsent
        flowaggregator['perreceived']=perreceived
        flowaggregator['persent']=persent
        flowaggregator['offeredload']=offeredload
        flowaggregator['throughput']=throughput
        flowaggregator['messagecompletionrate']=messagecompletionrate
        flowaggregator['delay']=delay
        flowaggregator['jitter']=jitter
        flowaggregator['hopcount']=hopcount
        return flowaggregator
        timee = time.clock()
        rtime = timee - times
        print('the flowaggregator time is : %fs' % rtime)

    def alltypeaggregator(self,sappi,sapps,vbri,vbrs,trafi,trafs,apptype):
        """
        按照业务类型，将业务流聚合在一起
        in the app-type vision to discover the difference between different input parameters
        aggregate the same type of app into one dataframe each simulaiton
        apptype:superapp vbr trafficgen
        """
        times = time.clock()
        appaggregator=pd.DataFrame()
        totalreceived = []
        totalsent = []
        offeredload = []
        throughput = []
        perreceived=[]
        persent=[]
        messagecompletionrate = []
        delay =[]
        jitter=[]
        hopcount=[]
        if apptype=='superapp':
            for app in self.superappname:
                totalreceived=totalreceived+self.appreceived[app].copy()
                totalsent=totalsent+self.appsent[app].copy()
                offeredload=offeredload+self.appofferedload[app].copy()
                throughput=throughput+self.appthroughput[app].copy()
                perreceived=perreceived+self.appperreceived[app].copy()
                persent=persent+self.apppersent[app].copy()
                messagecompletionrate=messagecompletionrate+self.appmessagecompletionrate[app].copy()
                delay=delay+self.appdelay[app].copy()
                jitter=jitter+self.appjitter[app].copy()
                hopcount=hopcount+self.apphopcount[app].copy()
            lens=len(hopcount)
            appaggregator['sappi']=[sappi]*lens
            appaggregator['sapps']=[sapps]*lens
            appaggregator['received']=totalreceived
            appaggregator['sent']=totalsent
            appaggregator['perreceived']=perreceived
            appaggregator['persent']=persent
            appaggregator['offeredload']=offeredload
            appaggregator['throughput']=throughput
            appaggregator['messagecompletionrate']=messagecompletionrate
            appaggregator['delay']=delay
            appaggregator['jitter']=jitter
            appaggregator['hopcount']=hopcount
        elif apptype=='vbr':
            for app in self.vbrname:
                totalreceived=totalreceived+self.appreceived[app].copy()
                totalsent=totalsent+self.appsent[app].copy()
                offeredload=offeredload+self.appofferedload[app].copy()
                throughput=throughput+self.appthroughput[app].copy()
                perreceived=perreceived+self.appperreceived[app].copy()
                persent=persent+self.apppersent[app].copy()
                messagecompletionrate=messagecompletionrate+self.appmessagecompletionrate[app].copy()
                delay=delay+self.appdelay[app].copy()
                jitter=jitter+self.appjitter[app].copy()
                hopcount=hopcount+self.apphopcount[app].copy()
            lens=len(hopcount)
            appaggregator['vbri']=[vbri]*lens
            appaggregator['vbrs']=[vbrs]*lens
            appaggregator['received']=totalreceived
            appaggregator['sent']=totalsent
            appaggregator['perreceived']=perreceived
            appaggregator['persent']=persent
            appaggregator['offeredload']=offeredload
            appaggregator['throughput']=throughput
            appaggregator['messagecompletionrate']=messagecompletionrate
            appaggregator['delay']=delay
            appaggregator['jitter']=jitter
            appaggregator['hopcount']=hopcount
        elif apptype=='trafficgen':
            for app in self.trafficgenname:
                totalreceived=totalreceived+self.appreceived[app].copy()
                totalsent=totalsent+self.appsent[app].copy()
                offeredload=offeredload+self.appofferedload[app].copy()
                throughput=throughput+self.appthroughput[app].copy()
                perreceived=perreceived+self.appperreceived[app].copy()
                persent=persent+self.apppersent[app].copy()
                messagecompletionrate=messagecompletionrate+self.appmessagecompletionrate[app].copy()
                delay=delay+self.appdelay[app].copy()
                jitter=jitter+self.appjitter[app].copy()
                hopcount=hopcount+self.apphopcount[app].copy()
            lens=len(hopcount)
            appaggregator['trafi']=[trafi]*lens
            appaggregator['trafs']=[trafs]*lens
            appaggregator['received']=totalreceived
            appaggregator['sent']=totalsent
            appaggregator['perreceived']=perreceived
            appaggregator['persent']=persent
            appaggregator['offeredload']=offeredload
            appaggregator['throughput']=throughput
            appaggregator['messagecompletionrate']=messagecompletionrate
            appaggregator['delay']=delay
            appaggregator['jitter']=jitter
            appaggregator['hopcount']=hopcount            
        return appaggregator
        timee = time.clock()
        rtime = timee - times
        print('the alltypeaggregator time is : %fs' % rtime)
        
    def boxdrawer(self,a,b,dataset):
        """
        2D箱图
        boxplot,provide interval packetsize and the other parameters 
        """
        sns.set(style="whitegrid", color_codes=True)
        plt.subplots(figsize = (16, 10))
        sns.boxplot(x=a, y=b, data=dataset)
        plt.ylabel(b, fontsize=16)
        plt.xlabel(a, fontsize=16)
        
    def stripplotdrawer(self,a,b,dataset):
        """
        
        stripplot,provide interval packetsize and the other parameters 
        """
        sns.set(style="whitegrid", color_codes=True)
        plt.subplots(figsize = (16, 10))
        sns.stripplot(x=a, y=b, data=dataset,jitter=True)
        plt.ylabel(b, fontsize=16)
        plt.xlabel(a, fontsize=16)
    
    def barplotdrawer(self,a,b,hues,dataset):
        """
        2D柱状图
        stripplot,provide interval packetsize and the other parameters 
        """
        sns.set(style="whitegrid", color_codes=True)
        plt.subplots(figsize = (16, 10))
        sns.barplot(x=a, y=b, data=dataset,hue=hues)
        plt.ylabel(b, fontsize=16)
        plt.xlabel(a, fontsize=16)
    
    def scatterdrawer(self,a,b,c,dataset):
        '''
        3D点状图
        3D-scatter ,provide interval packetsize and the other parameters 
        default marker is ^
        '''
        fig=plt.figure(figsize=(16,10))
#        ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        x=dataset[a]
        y=dataset[b]
        z=dataset[c]
        #ax.scatter(x, y, z, color='r')
        ax.scatter(x, y, z, marker='^')
        ax.set_xlabel(a,fontsize=20)
        ax.set_ylabel(b,fontsize=20)
        ax.set_zlabel(c,fontsize=20)  
        plt.show()
        
    def kdedrawer(self,a,b,path):
        """
        2D分布图
        apptype: Superapp
        KDE parameterstype: delay jitter hopcount throughput
        This function can print one of KDE between two QoS parameters
        """
        times = time.clock()
        para1=[]
        para2=[]
        dataset=pd.DataFrame()
        if a=="delay":
            for app in self.superappname:
                para1=para1+self.appdelay[app].copy()
        elif a=="jitter":
            for app in self.superappname:
                para1=para1+self.appjitter[app].copy()
        elif a=="hopcount":
            for app in self.superappname:
                para1=para1+self.apphopcount[app].copy()
        elif a=="throughput":
            for app in self.superappname:
                para1=para1+self.appthroughput[app].copy()                      
        if b=="delay":
            for app in self.superappname:
                para2=para2+self.appdelay[app].copy()
        elif b=="jitter":
            for app in self.superappname:
                para2=para2+self.appjitter[app].copy()
        elif b=="hopcount":
            for app in self.superappname:
                para2=para2+self.apphopcount[app].copy()
        elif b=="throughput":
            for app in self.superappname:
                para2=para2+self.appthroughput[app].copy()               
        dataset[a]=para1
        dataset[b]=para2
        with sns.axes_style('white'):  
            APPKDE=sns.jointplot(a,b,data=dataset, kind='kde') 
            APPKDE.savefig(path+a+b+"KDE.jpg") 
        timee = time.clock()
        rtime = timee - times
        print('the kdedrawer time is : %fs' % rtime)
    
    def rawdatadrawer(self,path,c):
        """
        每条流的原始数据统计图表：折线图
        This function print all of the app in dataset 
        including:rawsent rawreceived persent perreceived throughput 
                  messagecompletionrate delay jitter hopcount
        c=colour
        """
        for app in self.appname:
            plt.figure(figsize=(20,20))
            plt.style.use("ggplot")
            plt.subplot(711)
            plt.plot(self.appperreceived[app],'-', color=c,linewidth = '1')
            plt.title(app+' perreceived',fontsize=15)    
            plt.subplot(712)
            plt.plot(self.apppersent[app],'-', color=c,linewidth = '1')
            plt.title(app+' persent',fontsize=15)    
            plt.subplot(713)
            plt.plot(self.appthroughput[app],'-', color=c,linewidth = '1')
            plt.title(app+' throughput',fontsize=15)
            plt.subplot(714)
            plt.plot(self.appmessagecompletionrate[app],'-', color=c,linewidth = '1')
            plt.title(app+' messagecompletionrate',fontsize=15)
            plt.subplot(715)
            plt.plot(self.appdelay[app],'-', color=c,linewidth = '1')
            plt.title(app+' delay',fontsize=15)
            plt.subplot(716)
            plt.plot(self.appjitter[app],'-', color=c,linewidth = '1')
            plt.title(app+' jitter',fontsize=15)
            plt.subplot(717)
            plt.plot(self.apphopcount[app],'-', color=c,linewidth = '1')
            plt.title(app+' hopcount',fontsize=15)
            plt.savefig(path+app+".jpg")
        
    def appdatareader(self):
        """
        将数据业务流名称读进类中的字典
        read data from the specify dataset into dictionary in appname ordered
        """
        times = time.clock()
        timestamp = []
        totalreceived = []
        totalsent = []
        offeredload = []
        throughput = []
        perreceived=[]
        persent=[]
        messagecompletionrate = []
        delay =[]
        jitter=[]
        hopcount=[]
        for app in self.appname:
            cursor2=self.c.execute("SELECT Timestamp, BytesSent, BytesReceived, OfferedLoad, Throughput, MessageCompletionRate, Delay, Jitter, HopCount, ApplicationName  from APPLICATION_Summary")
            for row in cursor2:
                if row[9] == app:
                    timestamp.append(int(row[0]))
                    totalreceived.append(float(row[2]))
                    totalsent.append(float(row[1]))
                    if row[3]=='1.#INF00000':
                        offeredload.append(float(0))
                    else:
                        offeredload.append(float(row[3]))
                    throughput.append(float(row[4]))
                    messagecompletionrate.append(float(row[5]))
                    delay.append(float(row[6]))
                    if row[7] is None:
                        jitter.append(float(0))
                    else:     
                        jitter.append(float(row[7]))
                    hopcount.append(float(row[8]))
            for i in range(len(totalreceived)):
                if i == 0:
                    perreceived.append(totalreceived[i])
                else:
                    perreceived.append(totalreceived[i]-totalreceived[i-1])
            for j in range(len(totalsent)):
                if j == 0:
                    persent.append(totalsent[j])
                else:
                    persent.append(totalsent[j]-totalsent[j-1])
            self.appsent[app]=totalsent.copy()
            self.appreceived[app]=totalreceived.copy()
            self.apppersent[app]=persent.copy()
            self.appperreceived[app]=perreceived.copy()    
            self.appofferedload[app]=offeredload.copy()
            self.appthroughput[app]=throughput.copy()
            self.appmessagecompletionrate[app]=messagecompletionrate.copy()
            self.appdelay[app]=delay.copy()
            self.appjitter[app]=jitter.copy()
            self.apphopcount[app]=hopcount.copy() 
            timestamp.clear()
            persent.clear()
            perreceived.clear()
            totalsent.clear()
            totalreceived.clear()
            offeredload.clear()
            throughput.clear()
            messagecompletionrate.clear()
            delay.clear()
            jitter.clear()
            hopcount.clear()  
        timee = time.clock()
        rtime = timee - times
        print('the appdatareader time is : %fs' % rtime)
        
    def inputparainsert(self,sappi,sapps,vbri,vbrs,trafi,trafs):
        """
        在案业务名字分类的字典中，插入设计参数
        to inzert the application design parameter into the dictionary 
        FTP application has no interval and packetsize parameters 
        """
        times = time.clock()
        for app in self.superappname:
            lens=len(self.appreceived[app])
            interval1=[sappi]*lens
            packetsize1=[sapps]*lens
            self.appinterval[app]=interval1.copy()
            self.apppacketsize[app]=packetsize1.copy()
            interval2=[vbri]*lens
            packetsize2=[vbrs]*lens
            self.appinterval[app+'-VBR']=interval2.copy()
            self.apppacketsize[app+'-VBR']=packetsize2.copy()
            interval3=[trafi]*lens
            packetsize3=[trafs]*lens
            self.appinterval[app+'TrafficGen']=interval3.copy()
            self.apppacketsize[app+'TrafficGen']=packetsize3.copy()
        timee = time.clock()
        rtime = timee - times
        print('the inputparainsert time is : %fs' % rtime)

    def meandata(self,apptype):
        """
        对特定业务类型，进行取平均值操作，并返回
        """
        times = time.clock()
        appaggregator=pd.DataFrame()
        totalreceived = []
        totalsent = []
        offeredload = []
        throughput = []
        perreceived=[]
        persent=[]
        messagecompletionrate = []
        delay =[]
        jitter=[]
        hopcount=[]
        if apptype=='superapp':
            for app in self.superappname:
                totalreceived=totalreceived+self.appreceived[app].copy()
                totalsent=totalsent+self.appsent[app].copy()
                offeredload=offeredload+self.appofferedload[app].copy()
                throughput=throughput+self.appthroughput[app].copy()
                perreceived=perreceived+self.appperreceived[app].copy()
                persent=persent+self.apppersent[app].copy()
                messagecompletionrate=messagecompletionrate+self.appmessagecompletionrate[app].copy()
                delay=delay+self.appdelay[app].copy()
                jitter=jitter+self.appjitter[app].copy()
                hopcount=hopcount+self.apphopcount[app].copy()
            appaggregator['received']=totalreceived
            appaggregator['sent']=totalsent
            appaggregator['perreceived']=perreceived
            appaggregator['persent']=persent
            appaggregator['offeredload']=offeredload
            appaggregator['throughput']=throughput
            appaggregator['messagecompletionrate']=messagecompletionrate
            appaggregator['delay']=delay
            appaggregator['jitter']=jitter
            appaggregator['hopcount']=hopcount
        elif apptype=='vbr':
            for app in self.vbrname:
                totalreceived=totalreceived+self.appreceived[app].copy()
                totalsent=totalsent+self.appsent[app].copy()
                offeredload=offeredload+self.appofferedload[app].copy()
                throughput=throughput+self.appthroughput[app].copy()
                perreceived=perreceived+self.appperreceived[app].copy()
                persent=persent+self.apppersent[app].copy()
                messagecompletionrate=messagecompletionrate+self.appmessagecompletionrate[app].copy()
                delay=delay+self.appdelay[app].copy()
                jitter=jitter+self.appjitter[app].copy()
                hopcount=hopcount+self.apphopcount[app].copy()
            appaggregator['received']=totalreceived
            appaggregator['sent']=totalsent
            appaggregator['perreceived']=perreceived
            appaggregator['persent']=persent
            appaggregator['offeredload']=offeredload
            appaggregator['throughput']=throughput
            appaggregator['messagecompletionrate']=messagecompletionrate
            appaggregator['delay']=delay
            appaggregator['jitter']=jitter
            appaggregator['hopcount']=hopcount
        elif apptype=='trafficgen':
            for app in self.trafficgenname:
                totalreceived=totalreceived+self.appreceived[app].copy()
                totalsent=totalsent+self.appsent[app].copy()
                offeredload=offeredload+self.appofferedload[app].copy()
                throughput=throughput+self.appthroughput[app].copy()
                perreceived=perreceived+self.appperreceived[app].copy()
                persent=persent+self.apppersent[app].copy()
                messagecompletionrate=messagecompletionrate+self.appmessagecompletionrate[app].copy()
                delay=delay+self.appdelay[app].copy()
                jitter=jitter+self.appjitter[app].copy()
                hopcount=hopcount+self.apphopcount[app].copy()
            appaggregator['received']=totalreceived
            appaggregator['sent']=totalsent
            appaggregator['perreceived']=perreceived
            appaggregator['persent']=persent
            appaggregator['offeredload']=offeredload
            appaggregator['throughput']=throughput
            appaggregator['messagecompletionrate']=messagecompletionrate
            appaggregator['delay']=delay
            appaggregator['jitter']=jitter
            appaggregator['hopcount']=hopcount            
        return appaggregator.mean()
        timee = time.clock()
        rtime = timee - times
        print('the alltypeaggregator time is : %fs' % rtime)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
        
        
                    
                   
                    
                    
                    