# -*- coding: utf-8 -*-
"""
this is the Scenario test build process,
###1: run the generateSource.py file under the ./configfile folder,
        to select the all remote sensor that can trigger a traffic,
        the usage is :
            python generateSource.py 15 6 numRs RS
###2: run the WDNwordPro.exe programme under the ./Debug,
        to generate the Satellite walker nodes file , the related config file is :
        Scenary.sat, ScenaryRS.sat, Source.RSsat, GroundNode.Position
        the usage is :
            WDNwordPro.exe
###3: run the WNDconfigHandle.py to build the Scenario config file,
        the input config file : appConfig.json, tcpbuffer.json, linkConfig.json,
        under the ./configfile folder, so the modify the files represents the different config of net
        the usage is :
            python WNDconfigHandle.py
###4: run the exata to simulate and generate the stats file and the database file, in the ./OutConfigfile 
        the usage is:
         exata WDNscenario.config -simulation
###5: read the .stats file and the .db file to analysis the application performance
"""

import  os
import time
import shutil
import json

import weirdfishes
import pandas as pd

# pythonexe = 'F:\\ProgramData\\Anaconda3\\python.exe'

def RSSelect():
    times  = time.clock() 
    batName = 'runselect.bat'
    os.system(batName)
    timee = time.clock()
    rtime = timee - times
    print('the config remote sensor select file run time is : %fS' % rtime)

def runWDNwordPro():
    times = time.clock()
    batname = 'runexe.bat'
    os.system(batname)
    timee = time.clock()
    rtime = timee - times
    print('the .nodes file run time is : %fS' % rtime)

def runbuildconfigfile():
    times = time.clock()
    batname = 'runconfig.bat'
    os.system(batname)
    timee = time.clock()
    rtime = timee - times
    print('the .config file run time is : %fS' % rtime)

def runexata():
    times = time.clock()
    batname = 'runexata.bat'
    os.system(batname)
    timee = time.clock()
    rtime = timee - times
    print('the run exata simulation time is : %fs' % rtime)

def runOutfileStore(folderName):
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

def newfile(path):
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

def jsonread(filename):
    return json.loads(open(filename).read())

def jsonwrite(dir, filename):
    datastr = json.dumps(dir, sort_keys=True, indent=2)
    outfile = open(filename, 'w')
    outfile.write(datastr)

def writefile(file, wStr):
    fsimname = open(file, 'w')
    fsimname.write(wStr)
    
# =============================================================================
# superappInterval = [' REQUEST-INTERVAL EXP 5MS ',' REQUEST-INTERVAL EXP 10MS ',
#                ' REQUEST-INTERVAL EXP 20MS ', ' REQUEST-INTERVAL EXP 30MS ',
#                ' REQUEST-INTERVAL EXP 40MS ',' REQUEST-INTERVAL EXP 60MS ',
#                ' REQUEST-INTERVAL EXP 80MS ', ' REQUEST-INTERVAL EXP 100MS ', 
#                ' REQUEST-INTERVAL EXP 120MS ',  ' REQUEST-INTERVAL EXP 140MS ',
#                ' REQUEST-INTERVAL EXP 160MS ', ' REQUEST-INTERVAL EXP 180MS ', 
#                ' REQUEST-INTERVAL EXP 200MS ',' REQUEST-INTERVAL EXP 250MS '
#                ]
# =============================================================================
superapinterval = [' REQUEST-INTERVAL EXP 20MS ']
superappsize=[" REQUEST-SIZE DET 30000 "]
# =============================================================================
# superappsize=[" REQUEST-SIZE EXP 8000 "," REQUEST-SIZE EXP 10000 "," REQUEST-SIZE EXP 12000 ",
#          " REQUEST-SIZE EXP 14000 "," REQUEST-SIZE EXP 16000 "," REQUEST-SIZE EXP 18000 ",
#          " REQUEST-SIZE EXP 20000 "," REQUEST-SIZE EXP 22000 "," REQUEST-SIZE EXP 24000 ",
#          " REQUEST-SIZE EXP 26000 "," REQUEST-SIZE EXP 28000 "," REQUEST-SIZE EXP 30000 ",
#          " REQUEST-SIZE EXP 32000 "," REQUEST-SIZE EXP 34000 "," REQUEST-SIZE EXP 36000 ",
#          ]
# =============================================================================

vbrInterval = [ " 30MS "]
vbrsize=[" 24000 "]
# =============================================================================
# vbrsize=[" 8000 "," 10000 "," 12000 ",
#          " 14000 "," 16000 "," 18000 ",
#          ]
# =============================================================================
trafficgeninterval=[' DET 30MS ']
#trafficgensize=[' RND DET 12000 ']
# =============================================================================
# trafficgensize=[' RND DET 8000 ',' RND DET 10000 ',' RND DET 12000 ',
#                 ' RND DET 14000 ',' RND DET 16000 ',' RND DET 18000 ',
#                 ' RND DET 20000 ',' RND DET 22000 ',' RND DET 24000 ',
#                 ' RND DET 26000 ',' RND DET 28000 ',' RND DET 30000 ',
#                 ' RND DET 32000 ',' RND DET 34000 ',' RND DET 36000 ',
#                 ]
# =============================================================================
trafficgensize=[' RND DET 24000 ',' RND DET 26000 ',' RND DET 28000 ',' RND DET 30000 ',
                ' RND DET 32000 ',' RND DET 34000 ',' RND DET 36000 ',]
deliveryType = [' DELIVERY-TYPE UNRELIABLE ']
routing =['OSPFv2']

satBand = ['1G']
satDrop = ['0.0006']

rsBand = ['0.03G']
rsDrop = ['0.0015']



flowdata=pd.DataFrame()#所有数据库的流聚合
appdata=pd.DataFrame()#所有数据库的某种业务的聚合
memoryset=weirdfishes.ReinforcementLearningUnit()#记忆单元，存储每次的状态

#dataset='test_ REQUEST-SIZE EXP 18000 _ 2000'
#radio REQUEST-SIZE EXP 24000 _ 18000 _ RND EXP 22000

figpath="./Figure/"
outdatapath='./OutConfigfile/'


def runTest():
#    i = 0
    outlogfile = open('./outThrd.log', 'w')
    for appInterval_i in superapinterval:
        for appSize_i in superappsize:
            for delivery_i in deliveryType:
                for routing_i in routing:
                    for rsBand_i in rsBand:
                        for satBand_i in satBand:
                            for rsDrop_i in rsDrop:
                                for satDrop_i in satDrop:
                                    for vbrInterval_i in vbrInterval:  
                                        for vbrsize_i in vbrsize:
                                            for trafficgeninterval_i in trafficgeninterval:
                                                for trafficgensize_i in trafficgensize:
                                                    for i in range(60):                                                        
#                                                        i = i +43
                                                        linkconfig = jsonread('./configfile/linkConfig.json')
                                                        linkconfig['Sat'][0] = satBand_i
                                                        linkconfig['Sat'][1] = satDrop_i
                                                        linkconfig['RS'][0] = rsBand_i
                                                        linkconfig['RS'][1] = rsDrop_i
                                                        linkconfig['Ground'][0] = rsBand_i
                                                        linkconfig['Ground'][1] = rsDrop_i
                                                        linkconfig['ROUTING'] = routing_i
                                                        appconfig = jsonread('./configfile/SuperappConfig.json')
                                                        vbrconfig = jsonread('./configfile/VBRConfig.json')
                                                        trafconfig = jsonread('./configfile/TrafficGenConfig.json')
                                                        vbrconfig["MEAN-INTERVAL"]= vbrInterval_i
                                                        vbrconfig["ITEM-SIZE"]=vbrsize_i
                                                        trafconfig["PACKET-INTERVAL"]=trafficgeninterval_i
                                                        trafconfig["PACKET-SIZE"]=trafficgensize_i
                                                        appconfig['REQUEST-INTERVAL'] = appInterval_i
                                                        appconfig['REQUEST-SIZE'] = appSize_i
                                                        appconfig['DELIVERY-TYPE'] = delivery_i
                                                        # write the parameter to file
                                                        jsonwrite(linkconfig, './configfile/linkConfig.json')
                                                        jsonwrite(appconfig, './configfile/SuperappConfig.json')
                                                        jsonwrite(vbrconfig, './configfile/VBRConfig.json')
                                                        jsonwrite(trafconfig, './configfile/TrafficGenConfig.json')
                                                        # run the simulation and store the simulation out file
                                                        simName = 'radio'+appSize_i+"_"+vbrsize_i+"_"+trafficgensize_i+'_'+str(i)
                                                        simStr = 'EXPERIMENT-NAME ' + simName + '\n'
                                                        simename = './OutConfigfile/sim.name'
                                                        writefile(simename, simStr)
                                                        paraStr = 'SatBand: %s, RSBand: %s, Drop: %s, Route:%s, Interval: %s , Delivery: %s' % (satBand_i, rsBand_i, rsDrop_i, routing_i, appInterval_i, delivery_i)
                                                        writeStr = "%s : {%s}\n" % (simName, paraStr)
                                                        outlogfile.write(writeStr)
                                                        print(writeStr)
                                                        if False:
                                                            continue
                                                        try:
#                                                            runWDNwordPro()
                                                            runbuildconfigfile()
                                                            runexata()
                                                            #+++++++++++++++++++++++++++++++++++
     
                                                            #+++++++++++++++++++++++++++++++++++
                                                            """
                                                            暂时注释
                                                            """
    #                                                            runOutfileStore(simName)
                                                        except:
                                                            continue
    #                                                    except Exception as e:
    #                                                        ferro = open(simName + '.erro', 'w')
    #                                                        ferro.write(writeStr)







if __name__ == '__main__':
    runTest()
