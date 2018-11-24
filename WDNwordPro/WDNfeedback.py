# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:40:27 2018

@author: WDN
自优化的0反馈过程：
在得到query point之后（根据acquisition function得到），需要将query的参数加入进2EXATA的配置文件中
并进行仿真

修改：2018/6/26
对读取数据库的函数进行了优化（增加了数据库路径的选项）
"""
import  os
import time
import shutil
import json
import pandas as pd
import WDNexataReader#读取数据
import WDNoptimizer#建模，画图，AF函数

class FeedBackWorker:
    """
    1）将质询点代入仿真配置参数文件（json文件）中
    2）重新进行仿真配置文件生成
    3）进行一个数量的重复仿真
    4）*读取数据库
    """
    def __init__(self,superinter=20,supersize=30000,vbrinter=30,vbrsize=24000,trafinter=30,trafsize=34000):
        """
        initial function
        """
        self.figpath="./Figure/"
        self.outdatapath='./OutConfigfile/'
        
        self.superapinterval = ' REQUEST-INTERVAL EXP '+str(superinter)+'MS '
        self.superappsize=" REQUEST-SIZE DET "+str(supersize)+" "
        
        self.vbrInterval = " "+str(vbrinter)+"MS "
        self.vbrsize=" "+str(vbrsize)+" "
        
        self.trafficgeninterval=' DET '+str(trafinter)+'MS '
        self.trafficgensize=' RND DET '+str(trafsize)+' '
        
        self.deliveryType = ' DELIVERY-TYPE UNRELIABLE '
        self.routing ='OSPFv2'
        self.satBand = '1G'
        self.satDrop = '0.0006'
        self.rsBand = '0.03G'
        self.rsDrop = '0.0015'
        self.acquisitioncount=1
        self.querydatasetlist=[]
    
    def updatetrainningsetworker(self,path,point,count=60,style='qos'):
        """
        将querypoint得到的仿真数据读取并加入到原始训练集中
        目前是固定4个参数，2个参数可变
        """
        appSize_i=self.superappsize
        trafficgensize_i=self.trafficgensize
        vbrsize_i=self.vbrsize
        simname='radio'+appSize_i+"_"+vbrsize_i+"_"+trafficgensize_i
        self.querydatasetlist.append(simname)#将每次调用此函数时的数据库名字保存至list中
        newdata=WDNoptimizer.MemoryUnit()
        datapath=path
#        datapath='G:/testData/GMM1/'
        if style=='qos':            
            for i in range(count):
                dataset=simname+'_'+str(i)
                reader=WDNexataReader.ExataDBreader()
                reader.opendataset(dataset,datapath)#读取特定路径下的数据库
                reader.appnamereader()#读取业务层的业务名称
                reader.appfilter()#将业务名称分类至三个list
                reader.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
                reader.inputparainsert(20,point[0],30,24000,30,point[1])
                "================================================================="
                eva=WDNoptimizer.EvaluationUnit()
                superapp=reader.meandata('superapp')
                eva.calculateMetricEvaValue(superapp)
                vbr=reader.meandata('vbr')
                eva.calculateMetricEvaValue(vbr)
                trafficgen=reader.meandata('trafficgen')
                eva.calculateMetricEvaValue(trafficgen)
                state=[20,point[0],30,24000,30,point[1]]
                print(state)
                qos=eva.qoslist
    #           memoryset.insertmemoryunit(state=state,value=value)
                newdata.qosinserter(state=state,qos=qos)
            return newdata.qosmemoryunit
        elif style=='value':
            for i in range(count):
                dataset=simname+'_'+str(i)
                reader=WDNexataReader.ExataDBreader()#实例化
                reader.opendataset(dataset,datapath)#读取特定路径下的数据库
                reader.appnamereader()#读取业务层的业务名称
                reader.appfilter()#将业务名称分类至三个list
                reader.appdatareader()#将每个业物流的输出数据存到实例化的类中的字典里面
                reader.inputparainsert(20,point[0],30,24000,30,point[1])
                "================================================================="
                eva=WDNoptimizer.EvaluationUnit()
                superapp=reader.meandata('superapp')
                eva.calculateMetricEvaValue(superapp)
                vbr=reader.meandata('vbr')
                eva.calculateMetricEvaValue(vbr)
                trafficgen=reader.meandata('trafficgen')
                eva.calculateMetricEvaValue(trafficgen)
                state=[20,point[0],30,24000,30,point[1]]
                print(state)
                value=eva.evaluationvalue()
    #           memoryset.insertmemoryunit(state=state,value=value)
                newdata.valueinserter(state=state,value=value)
            return newdata.memoryunit

    
    def updateQuerypointworker(self,point):
        '''
        将querypoint的参数加入实例化的反馈类中，更新参数
        '''
        #ttt=np.array([[40000,20000]])
        #ttt[0][0]=sapps,ttt[0][1]=trafs
        self.superappsize=" REQUEST-SIZE DET "+str(point[0])+" "
        self.trafficgensize=' RND DET '+str(point[1])+' '
    
    
    def runTest(self,count=60):
        """
        配置仿真参数，生成配置文件，运行仿真
        """
#        outlogfile = open('./outThrd.log', 'w')
        delivery_i=self.deliveryType
        routing_i=self.routing
        rsBand_i=self.rsBand
        satBand_i=self.satBand
        rsDrop_i=self.rsDrop
        satDrop_i=self.satDrop
        appInterval_i=self.superapinterval
        appSize_i=self.superappsize
        vbrInterval_i=self.vbrInterval
        vbrsize_i=self.vbrsize
        trafficgeninterval_i=self.trafficgeninterval
        trafficgensize_i=self.trafficgensize
        self.acquisitioncount=self.acquisitioncount+1
        for i in range(count):
            linkconfig = self.jsonread('./configfile/linkConfig.json')
            linkconfig['Sat'][0] = satBand_i
            linkconfig['Sat'][1] = satDrop_i
            linkconfig['RS'][0] = rsBand_i
            linkconfig['RS'][1] = rsDrop_i
            linkconfig['Ground'][0] = rsBand_i
            linkconfig['Ground'][1] = rsDrop_i
            linkconfig['ROUTING'] = routing_i
            appconfig = self.jsonread('./configfile/SuperappConfig.json')
            vbrconfig = self.jsonread('./configfile/VBRConfig.json')
            trafconfig = self.jsonread('./configfile/TrafficGenConfig.json')
            vbrconfig["MEAN-INTERVAL"]= vbrInterval_i
            vbrconfig["ITEM-SIZE"]=vbrsize_i
            trafconfig["PACKET-INTERVAL"]=trafficgeninterval_i
            trafconfig["PACKET-SIZE"]=trafficgensize_i
            appconfig['REQUEST-INTERVAL'] = appInterval_i
            appconfig['REQUEST-SIZE'] = appSize_i
            appconfig['DELIVERY-TYPE'] = delivery_i
            # write the parameter to file
            self.jsonwrite(linkconfig, './configfile/linkConfig.json')
            self.jsonwrite(appconfig, './configfile/SuperappConfig.json')
            self.jsonwrite(vbrconfig, './configfile/VBRConfig.json')
            self.jsonwrite(trafconfig, './configfile/TrafficGenConfig.json')
            # run the simulation and store the simulation out file
            simName = 'radio'+appSize_i+"_"+vbrsize_i+"_"+trafficgensize_i+'_'+str(i)
            simStr = 'EXPERIMENT-NAME ' + simName + '\n'
            simename = './OutConfigfile/sim.name'
            self.writefile(simename, simStr)
#            paraStr = 'SatBand: %s, RSBand: %s, Drop: %s, Route:%s, Interval: %s , Delivery: %s' % (satBand_i, rsBand_i, rsDrop_i, routing_i, appInterval_i, delivery_i)
#            writeStr = "%s : {%s}\n" % (simName, paraStr)
#            outlogfile.write(writeStr)
#            print(writeStr)
            if False:
                continue
            try:
#                self.runWDNwordPro()
                self.runbuildconfigfile()
                self.runexata()
                """
                暂时注释
                """
#                self.runOutfileStore(simName)
            except:
                continue
      
        
        
    def jsonread(self,filename):
        return json.loads(open(filename).read())

    def jsonwrite(self,dir, filename):
        datastr = json.dumps(dir, sort_keys=True, indent=2)
        outfile = open(filename, 'w')
        outfile.write(datastr)

    def writefile(self,file, wStr):
        fsimname = open(file, 'w')
        fsimname.write(wStr)
        
    def newfile(self,path):
        path=path.strip()
        path=path.rstrip("\\")
        # 判断路径是否存在
        isExists=os.path.exists(path)
        # 不存在
        if not isExists:
            # 创建目录操作函数
            os.makedirs(path)
            # print(path+' 创建成功')
            return True
        #存在
        else:
            # print(path+' 目录已存在')
            return False    
        
    def runOutfileStore(self,folderName):
        times = time.clock()
        outprefolder = './OutStorefile/'
        inprefolder = './configfile/'
        outfolder = outprefolder + folderName
        outconfig = outfolder + '/configfile/'
        shutil.copytree(inprefolder, outconfig)
        simoutfolder = './OutConfigfile/'
        statfilename = '/'+ folderName + '.stat'
        dbfilename = '/' + folderName + '.db'
        appfilename = '/' + folderName + '.app'
        appfile = simoutfolder + 'Scenario.app'
        shutil.copyfile(simoutfolder + statfilename, outfolder + statfilename)
        shutil.copyfile(simoutfolder + dbfilename, outfolder + dbfilename)
        shutil.copyfile(appfile, outfolder + appfilename)
        timee = time.clock()
        rtime = timee - times
        print('the store file time is : %fS' % rtime)    
        
    def runexata(self):
        times = time.clock()
        batname = 'runexata.bat'
        os.system(batname)
        timee = time.clock()
        rtime = timee - times
        print('the run exata simulation time is : %fs' % rtime)   
        
    def runbuildconfigfile(self):
        times = time.clock()
        batname = 'runconfig.bat'
        os.system(batname)
        timee = time.clock()
        rtime = timee - times
        print('the .config file run time is : %fS' % rtime)
        
    def RSSelect(self):
        times  = time.clock() 
        batName = 'runselect.bat'
        os.system(batName)
        timee = time.clock()
        rtime = timee - times
        print('the config remote sensor select file run time is : %fS' % rtime)
        
    def runWDNwordPro(self):
        times = time.clock()
        batname = 'runexe.bat'
        os.system(batname)
        timee = time.clock()
        rtime = timee - times
        print('the .nodes file run time is : %fS' % rtime)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        