# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:40:27 2018

@author: WDN
自优化的0反馈过程：
在得到query point之后（根据acquisition function得到），需要将query的参数加入进2EXATA的配置文件中
并进行仿真
"""
import  os
import time
import shutil
import json
import pandas as pd

class FeedBackWorker:
    """
    1）将质询点代入仿真配置参数文件（json文件）中
    2）重新进行仿真配置文件生成
    3）进行一个数量的重复仿真
    4）*读取数据库
    """
    def __init__(self):
        """
        initial function
        """
        self.figpath="./Figure/"
        self.outdatapath='./OutConfigfile/'
        
        self.superapinterval = [' REQUEST-INTERVAL EXP 20MS ']
        self.superappsize=[" REQUEST-SIZE DET 30000 "]
        
        self.vbrInterval = [ " 30MS "]
        self.vbrsize=[" 24000 "]
        
        self.trafficgeninterval=[' DET 30MS ']
        self.trafficgensize=[' RND DET 34000 ']
        
        self.deliveryType = [' DELIVERY-TYPE UNRELIABLE ']
        self.routing =['OSPFv2']
        self.satBand = ['1G']
        self.satDrop = ['0.0006']
        self.rsBand = ['0.03G']
        self.rsDrop = ['0.0015']
        
        
    def runTest(self,count=60):
        """
        配置仿真参数，生成配置文件，运行仿真
        """
        outlogfile = open('./outThrd.log', 'w')
        for delivery_i in self.deliveryType:
            for routing_i in self.routing:
                for rsBand_i in self.rsBand:
                    for satBand_i in self.satBand:
                        for rsDrop_i in self.rsDrop:
                            for satDrop_i in self.satDrop:
                                for appInterval_i in self.superapinterval:
                                    for appSize_i in self.superappsize:
                                        for vbrInterval_i in self.vbrInterval:  
                                            for vbrsize_i in self.vbrsize:
                                                for trafficgeninterval_i in self.trafficgeninterval:
                                                    for trafficgensize_i in self.trafficgensize:
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
                                                            paraStr = 'SatBand: %s, RSBand: %s, Drop: %s, Route:%s, Interval: %s , Delivery: %s' % (satBand_i, rsBand_i, rsDrop_i, routing_i, appInterval_i, delivery_i)
                                                            writeStr = "%s : {%s}\n" % (simName, paraStr)
                                                            outlogfile.write(writeStr)
                                                            print(writeStr)
                                                            if False:
                                                                continue
                                                            try:
    #                                                            runWDNwordPro()
                                                                self.runbuildconfigfile()
                                                                self.runexata()
                                                                """
                                                                暂时注释
                                                                """
        #                                                            self.runOutfileStore(simName)
                                                            except:
                                                                continue
        #                                                    except Exception as e:
        #                                                        ferro = open(simName + '.erro', 'w')
    #                                                            ferro.write(writeStr)
                                                            
                                                            
                                                            
                                                            
                                                            
                                                            
                                                            
        
        
        
        
        
        
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        