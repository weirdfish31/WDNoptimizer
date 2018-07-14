import os
import csv
import datetime
import numpy as np
import json
# starttime = datetime.datetime.strptime('3 Nov 2013 04:00:00', '%d %b %Y %H:%M:%S')

class readhelper:
           
    def readsatfile(self):
        satfile = open('./configfile/Scenary.sat')

        while True:
            line = satfile.readline()
            if not line:
                break
            if line.startswith('#'):
                continue
            if line.startswith('rootname'):
                linelist = line.split(':')
                self.rootname = linelist[1].strip()
            
            if line.startswith('numPlans'):
                linelist = line.split(':')
                self.numPlans = int(linelist[1])
            
            if line.startswith('numSatOfPlans'):
                linelist = line.split(':')
                self.numSatOfPlans = int(linelist[1])

            if line.startswith('starttime'):
                linelist = line.split('|')
                
                strtemp = linelist[1].split('.')
                self.starttime = datetime.datetime.strptime(strtemp[0], '%d %b %Y %H:%M:%S')

            if line.startswith('endtime'):
                linelist = line.split('|')
                strtemp = linelist[1].split('.')
                self.endtime = datetime.datetime.strptime(strtemp[0], '%d %b %Y %H:%M:%S')
                self.timedur = (self.endtime - self.starttime).seconds

    def readNodeConfigFile(self):
        nodelist = []
        file = open('./OutConfigfile/Scenario.nodeconfig')
        while True:
            line = file.readline()
            if not line or line.startswith('#'):
                break
            linelist = line.split()
            linelist[0] = int(linelist[0].strip().strip('[]'))
            if linelist[2].startswith('Sat'):
                linelist[1] = 1
            elif linelist[2].startswith('RS'):
                linelist[1] = 2
            elif linelist[2].startswith('Aircraft'):
                linelist[1] = 3
            else: 
                linelist[1] = 0
            nodelist.append(linelist)
        return nodelist

    def readnodePosition(self):
        """
        Read all ground node and all Remote sensor Sat to store the name in the gnodelist
        """
        gnodelist = []
        file = open('./configfile/GroundNode.Position')
        while True:
            line = file.readline()
            if not line:
                break
            if line.startswith('#'):
                continue
            templist = line.split()
            gnodelist.append(templist[0])
        fileRs = open('./configfile/Source.RSsat')
        while True:
            line = fileRs.readline()
            if not line:
                break
            if line.startswith('#'):
                continue
            templist = line.split()
            gnodelist.append(templist[0])
        self.gnodelist = gnodelist
        fileMN = open('./configfile/MobileNode.name')
        while True:
            line = fileMN.readline()
            if not line:
                break
            if line.startswith('#'):
                continue
            templist = line.split()
            gnodelist.append(templist[0])
        self.gnodelist = gnodelist
    def __init__(self):
        """
        self.nodelist = [[nodeid, nodetype,nodename]]
        self.gnodelist = [groundnodename]
        self.allAccess = [[groundnodename, nodeallaccessList]]
        nodeallaccessList = [[pathnode, sarttime, endtime, dur]]
        self.applist = [[srcid, destid, starttime,srcname, destname]]
        self.appAccess = [[pathnode, sarttime, endtime, dur]]
        self.GToSatList = [[ IPstr, pathnode]]
        self.GToSatTop = [pathnode]
        self.IPandFault = [[IPstr,[pathnode, sarttime, endtime, dur]]]
        """
        self.nodelist = self.readNodeConfigFile()
        self.readsatfile()
        self.readnodePosition()
        self.datetimeFormat = '%d %b %Y %H:%M:%S'
        self.allAccess = []
        self.numapp = 3
        self.appdur = 0
        self.applist = []
        self.appAccess = []
        self.GtoSatBand = 30000000
        self.SatToSatBand = 100000000
        self.GtoSatList = []
        self.GToSatTop = []
        self.IPandFault = []
        self.simdur = 10000
        self.link = json.loads(open('./configfile/linkConfig.json').read())
        self.routing = 'OSPFv2'
        self.modeltype = {'Sat':'WIRELESS LINK', 'RS':'WIRELESS LINK','Ground':'WIRELESS LINK','Aircraft':'WIRELESS LINK'}
    def createlinktop(self):
        """
        create the sat-link between comsats
        
        """
        linkfile = open('./OutConfigfile/satlink.topconfig','w')
        netindex = 0
        for i in range(1,self.numPlans):
            for j in range(1,self.numSatOfPlans + 1):
                
                curSat = self.getSatname(i,j)
                curId = self.getthenodeid(curSat)

                SplanSat = self.getSatname(i,j+1)
                SplanId = self.getthenodeid(SplanSat)

                AsplansSat = self.getSatname(i + 1,j)
                AsplansId = self.getthenodeid(AsplansSat)


                netindex = netindex + 1
                linkfile.write('LINK N8-'+str(netindex)+'.0 { ' 
                + str(curId) + ', ' + str(SplanId) + '}\n')
                linkfile.write('[ N8-'+str(netindex)+'.0] ' 
                + 'LINK-PHY-TYPE WIRELESS\n')
                linkfile.write('[ N8-'+str(netindex)+'.0] ' 
                + 'LINK-BANDWIDTH '+ self.bandhelper(self.link['Sat'][0]) +'\n')
                linkfile.write('[ N8-'+str(netindex)+'.0] ' 
                + 'FIXED-COMMS-DROP-PROBABILITY '+ self.bandhelper(self.link['Sat'][-1]) +'\n')
                linkfile.write('[ N8-'+str(netindex)+'.0] ' 
                + 'DUMMY-GUI-SYMMETRIC-LINK YES\n')
                linkfile.write('[ N8-'+str(netindex)+'.0] ' 
                + 'ROUTING-PROTOCOL '+ self.link['ROUTING'] +'\n')

                netindex = netindex + 1
                
                linkfile.write('LINK N8-'+str(netindex)+'.0 { ' 
                + str(curId) + ', ' + str(AsplansId) + '}\n')
                linkfile.write('[ N8-'+str(netindex)+'.0] ' 
                + 'LINK-PHY-TYPE WIRELESS\n')
                linkfile.write('[ N8-'+str(netindex)+'.0] ' 
                + 'LINK-BANDWIDTH '+ self.bandhelper(self.link['Sat'][0])+'\n')
                linkfile.write('[ N8-'+str(netindex)+'.0] ' 
                + 'FIXED-COMMS-DROP-PROBABILITY '+ self.bandhelper(self.link['Sat'][-1]) +'\n')
                linkfile.write('[ N8-'+str(netindex)+'.0] ' 
                + 'DUMMY-GUI-SYMMETRIC-LINK YES\n')
                linkfile.write('[ N8-'+str(netindex)+'.0] ' 
                + 'ROUTING-PROTOCOL '+ self.link['ROUTING'] +'\n')

    def bandhelper(self, bandstr):
        if bandstr.endswith('G'):
            return str(float(bandstr[:-1]) * 1000000000)
        if bandstr.endswith('M'):
            return str(float(bandstr[:-1]) * 1000000)
        return  bandstr

    def getthenodeid(self, Satname):
        Satid = 0
        for node in self.nodelist:
            if node[2] == Satname:
                Satid = node[0]
                break
        return Satid
    def getSatname(self, i, j):
        if i > self.numPlans:
            i = i % self.numPlans
        if j > self.numSatOfPlans:
            j = j % self.numSatOfPlans
        istr = str(i)
        if self.numPlans > 9 and i <=9:
            istr = '0' + istr
        jstr = str(j)
        if self.numSatOfPlans > 9 and j <=9:
            jstr = '0'+jstr
        Satname = self.rootname + istr + jstr

        return Satname

    def csvreadhandle(self):
        """
        read the all .csv file in outaccess folder
        and handle the .csv file 
        
        """
        allcsvFilename = [csvFile for csvFile in os.listdir('./outAccess/') if csvFile.endswith('.csv')]
        # for csvFile in os.listdir('./outAccess/'):
        #     if not csvFile.endswith('.csv'):
        #         continue
        prefolder = './outAccess/'
        preoutfolder = './outAccessSort/'

        for gnode in self.gnodelist:
            gnodeAccess = self.getthegnodecsvlit(prefolder=prefolder,
                                                preoutfolder=preoutfolder,
                                                gnode=gnode,
                                                allcsvFilename = allcsvFilename,
                                                fileFlag = False)           
            gnodeHandledAccess = self.AcessPreHandle(gnodeAccess,'')
            self.allAccess.append(list((gnode,gnodeHandledAccess)))

    def getthegnodecsvlit(self,preoutfolder,prefolder,gnode ,allcsvFilename,fileFlag):
        gnodeAccess = []
        gnodeallcsv = [filename for filename in allcsvFilename if filename.startswith(gnode)]
        for gnodecsv in gnodeallcsv:
            gnodecsvreader = csv.reader(open(prefolder+gnodecsv))
            for row in gnodecsvreader:
                if not row:
                    continue
                for rowdata in row:
                    rowstr = rowdata.strip()
                    if len(rowstr) < 66:
                        continue
                    rowdatalist = rowstr.split('    ')
                    rowlist = [x for x in rowdatalist if x]
                    if not rowlist[0].isnumeric():
                        continue
                    starttimestr =  rowlist[1].split('.')
                    startime = datetime.datetime.strptime(starttimestr[0],self.datetimeFormat) 
                    difsatrt = (startime-self.starttime).seconds
                    endtimestr = rowlist[2].split('.')
                    endtime = datetime.datetime.strptime(endtimestr[0], self.datetimeFormat) 
                    difendt = (endtime - self.starttime).seconds
                    if difendt == 0 and difsatrt > 0:
                        difendt = 86400
                    dur = difendt-difsatrt
                    # singleaccess.append(gnodecsv.rstrip('.csv'))
                    # singleaccess.append(difsatrt)
                    # singleaccess.append(difendt)
                    # singleaccess.append(dur)
                    tempsinglerowlist = list((gnodecsv.rstrip('.csv'), difsatrt, difendt, dur))
                    gnodeAccess.append(tempsinglerowlist)
                    # dur = int(rowlist[3])
        gnodeAccess.sort(key = lambda x:(x[1],x[2]))
        if fileFlag:
            gnodefileout = open(preoutfolder+gnode+'.nodecsv', 'w')
            for singlerow in gnodeAccess:
                gnodefileout.write(str(singlerow) + '\n') 
        return gnodeAccess

    def AcessPreHandle(self, accessList,gnode = ''):
        resList =[]
        resList.append(accessList[0])
        for index, item in enumerate(accessList):
            # item = copy.copy(item)
            if index + 1 == len(accessList):
                break
            if accessList[index+1][1] <= item[1] and  accessList[index+1][2] >= item[2] and item[1] == 0:
                
                #print('del')
                if item in resList:
                    resList.remove(item)
                resList.append(accessList[index+1])
                # resList.append(copy.copy(accessList[index+1]))
                continue 

            resLastItem = resList[-1]
            if resLastItem[2] > self.simdur:
                break
            itemnext = accessList[index+1]
            if item[1] <= resLastItem[2] and itemnext[1] >= resLastItem[2]:
                # print(resLastItem)
                # print(item)
                # print(itemnext)
                resList.append(item)
                if item[2] <= itemnext[1]:
                    resList.remove(item)
                    i = 1
                    while True:
                        itemlast = accessList[index-i]
                        if itemlast[2] >= itemnext[1]:
                            resList.append(itemlast)
                            break
                        i = i + 1
        if gnode:
            outfolder = './outAccessSelect/'
            fgnode = open(outfolder + gnode + '.access', 'w')
            for x in resList:
                fgnode.write(str(x) + '\n')
        return resList

    def generateGGapp(self):
        groundStation = []
        for x in self.gnodelist:
            if not x.startswith('RS'):
                if not x.startswith('Aircraft'):
                    groundStation.append(x)
        numapp = self.numapp
        print(numapp)
        applist = []
        appset = set(applist)
        for i in range(numapp):
            while True:
                src = groundStation[np.random.randint(0, len(groundStation))]
                groundStation.remove(src)
                dest = groundStation[np.random.randint(0, len(groundStation))]
                groundStation.remove(dest)
#                     src not in appset and src != dest and dest not in appset
                if src not in appset and src != dest :
                    applist.append((src, dest))
                    appset.add(src)
                    appset.add(dest)
                    break

        appconfig = json.loads(open('./configfile/SuperappConfig.json').read())
        vbrconfig = json.loads(open('./configfile/VBRConfig.json').read())
        ftpconfig = json.loads(open('./configfile/FTPConfig.json').read())
        trafficgenconfig = json.loads(open('./configfile/TrafficGenConfig.json').read())
        fileappconfig = open('./OutConfigfile/Scenario.app' ,'w')
        for item in applist:
            srcid = self.getthenodeid(item[0])
            destid = self.getthenodeid(item[1])
            appName = item[0] + '-' + item[1]
            ftpstime = np.random.exponential(scale=int(ftpconfig["START-TIME"]))
            ftptime = int(round(ftpstime)) + 1
            ftpITS = np.random.exponential(scale=int(ftpconfig["ITEM-TO-SEND"]))
            ftpITS1 = int(round(ftpITS)) + 1
            fileappconfig.write('SUPER-APPLICATION ' + str(srcid) + ' ' + str(destid)
                                + appconfig["START-TIME"] + appconfig["DURATION"]
                                + appconfig["DELIVERY-TYPE"] + appconfig["CONNECTION-RETRY"] 
                                + appconfig["REQUEST-NUM"] + appconfig["REQUEST-SIZE"]
                                + appconfig["REQUEST-INTERVAL"] 
                                + appconfig["REPLAY"]
                                +" APPLICATION-NAME " + appName + "\n")
        
            appName1 = item[0] + '-' + item[1]+'-VBR'
            fileappconfig.write('VBR ' + str(srcid) + ' ' + str(destid)
                                + vbrconfig["ITEM-SIZE"] + vbrconfig["MEAN-INTERVAL"]
                                + vbrconfig["START-TIME"] + vbrconfig["END-TIME"] 
                                +" APPLICATION-NAME " + appName1 + "\n")
            
            appName2 = item[0] + '-' + item[1]+'-FTP'
            fileappconfig.write('FTP ' + str(srcid) + ' ' + str(destid)
                                + ' ' + str(ftpITS1) + ' ' + str(ftptime) + 'S' 
                                +" APPLICATION-NAME " + appName2 + "\n")
            
            appName3 = item[0] + '-' + item[1]+'-TrafficGen'
            fileappconfig.write('TRAFFIC-GEN ' + str(srcid) + ' ' + str(destid)
                                + ' ' + trafficgenconfig["START-TIME"] + trafficgenconfig["DURATION"] 
                                +trafficgenconfig["PACKET-SIZE"]+trafficgenconfig["PACKET-INTERVAL"]
                                +trafficgenconfig["PROBABILITY"]+trafficgenconfig["LEAKY-BUCKET"]
                                +" APPLICATION-NAME " + appName3 + "\n")
    def generateRSGapp(self):
        groundStation = [x for x in self.nodelist if  x[1]==0]
        remoteSensors = [x for x in self.nodelist if  x[1]==2]
        numapp = self.numapp
        applist = []
        appset = set(applist)
        for i in range(numapp):
            while True:
                src = remoteSensors[np.random.randint(0, len(remoteSensors))][2]
                dest = groundStation[np.random.randint(0, len(groundStation))][2]
                if src not in appset and dest not in appset:
                    applist.append((src, dest))
                    appset.add(src)
                    appset.add(dest)
                    break
        appconfig = json.loads(open('./configfile/SuperappConfig.json').read())
        fileappconfig = open('./OutConfigfile/Scenario.app' ,'w')
        appType = 'SUPER-APPLICATION '
        for item in applist:
            srcid = self.getthenodeid(item[0])
            destid = self.getthenodeid(item[1])
            appName = item[0] + '-' + item[1]
            fileappconfig.write(appType + str(srcid) + ' ' + str(destid)
                                + appconfig["START-TIME"] + appconfig["DURATION"]
                                + appconfig["DELIVERY-TYPE"] + appconfig["CONNECTION-RETRY"] 
                                + appconfig["REQUEST-NUM"] + appconfig["REQUEST-SIZE"]
                                + appconfig["REQUEST-INTERVAL"] 
                                + appconfig["REPLAY"]
                                +" APPLICATION-NAME " + appName + "\n")    
        
    def generateGStopFault(self):
        """ 
        generate the ground to sat and the remote senosr to sat top
        and the single top fault time
        self.allAccess = [[groundnodename, nodeallaccessList]]
        nodeallaccessList = [[pathnode, sarttime, endtime, dur]]
        # LINK-PHY-TYPE WIRELESS
        LINK N8-190.0.1.0 { 1, 4 }
        LINK-PHY-TYPE WIRELESS
        DUMMY-GUI-SYMMETRIC-LINK YES
        LINK-BANDWIDTH 10000000000
        """
        netIndex = 0
        fileGroundTop = open('./OutConfigfile/GroundSat.top-config', 'w')
        fileGroundFault = open('./OutConfigfile/GroundSat.fault', 'w')
        for item in self.allAccess:
            for nodepath in item[1]:
                netIndex = netIndex + 1
                srcTOSat = nodepath[0].split('-')
                srcName = srcTOSat[0]
                SatName = srcTOSat[1]
                srcid = self.getthenodeid(srcName)
                Satid = self.getthenodeid(SatName)
                net = self.generateIPStr(netIndex)
                fileGroundTop.write('LINK '+ net + ' { ' + str(srcid) + ', ' 
                                        + str(Satid) + '} #'+nodepath[0]+'\n')
                fileGroundTop.write('['+ net +']' + ' LINK-PHY-TYPE WIRELESS\n')
                fileGroundTop.write('['+ net +']' + ' DUMMY-GUI-SYMMETRIC-LINK YES\n')
                if srcName.startswith('RS'):
                    fileGroundTop.write('['+ net +']' + ' LINK-BANDWIDTH ' 
                                            + self.bandhelper(self.link['RS'][0]) + '\n')
                    fileGroundTop.write('['+ net +']' + ' LINK-PROPAGATION-RAIN-INTENSITY ' 
                                            + self.bandhelper(self.link['RS'][1]) + '\n')
                    fileGroundTop.write('['+ net +']' + ' FIXED-COMMS-DROP-PROBABILITY ' 
                                            + self.bandhelper(self.link['RS'][-1]) + '\n')
                elif srcName.startswith('Aircraft'):
                    fileGroundTop.write('['+ net +']' + ' LINK-BANDWIDTH ' 
                                            + self.bandhelper(self.link['Aircraft'][0]) + '\n')
                    fileGroundTop.write('['+ net +']' + ' LINK-PROPAGATION-RAIN-INTENSITY ' 
                                            + self.bandhelper(self.link['Aircraft'][1]) + '\n')
                    fileGroundTop.write('['+ net +']' + ' LINK-PROPAGATION-CLIMATE ' 
                                            + self.bandhelper(self.link['Aircraft'][2]) + '\n')
                    fileGroundTop.write('['+ net +']' + ' LINK-PROPAGATION-REFRACTIVITY ' 
                                            + self.bandhelper(self.link['Aircraft'][3]) + '\n')                    
                    fileGroundTop.write('['+ net +']' + ' FIXED-COMMS-DROP-PROBABILITY ' 
                                            + self.bandhelper(self.link['Aircraft'][-1]) + '\n')
                else:
                    fileGroundTop.write('['+ net +']' + ' LINK-BANDWIDTH ' 
                                            + self.bandhelper(self.link['Ground'][0]) + '\n')
                    fileGroundTop.write('['+ net +']' + ' LINK-PROPAGATION-RAIN-INTENSITY ' 
                                            + self.bandhelper(self.link['Ground'][1]) + '\n')
                    fileGroundTop.write('['+ net +']' + ' LINK-PROPAGATION-CLIMATE ' 
                                            + self.bandhelper(self.link['Ground'][2]) + '\n')
                    fileGroundTop.write('['+ net +']' + ' LINK-PROPAGATION-REFRACTIVITY ' 
                                            + self.bandhelper(self.link['Ground'][3]) + '\n')
                    fileGroundTop.write('['+ net +']' + ' FIXED-COMMS-DROP-PROBABILITY ' 
                                            + self.bandhelper(self.link['Ground'][-1]) + '\n')
                    
                fileGroundTop.write('['+ net +']' + ' ROUTING-PROTOCOL ' + self.link['ROUTING'] + '\n')
                topStart = nodepath[1]
                topEnd = nodepath[2]
                faultIp = net[3:-1] + '2'
                if topStart > 1:
                    fileGroundFault.write('INTERFACE-FAULT '+ faultIp 
                                           + ' 0S ' + str(topStart) + 'S\n')
                fileGroundFault.write('INTERFACE-FAULT '+ faultIp 
                                           + ' ' + str(topEnd) + 'S 0S\n')
              
                
    def getTheNodeAllAccess(self, Nodename):
        for allAccessNode in self.allAccess:
            if allAccessNode[0] == Nodename:
                return allAccessNode[1]
    
    def generateTCPwindow(self):
        """
        TCP-MSS 1024
        TCP-SEND-BUFFER 16384
        TCP-RECEIVE-BUFFER 16384
        """
        tcpfile = open('./OutConfigfile/sim.tcpbuffer', 'w')
        tcpconfig = json.loads(open('./configfile/tcpbuffer.json').read())
        tcpfile.write(tcpconfig["MSS"] + "\n")
        tcpfile.write(tcpconfig['Send'] + '\n')
        tcpfile.write(tcpconfig['Recv'] + '\n')
    
    def starttimeFind(self,startime,accessList):
        """
        give a start time from the accessList return the overlap list
        """
        sindex = 0
        eindex = sindex
        for index , listItem in  enumerate(accessList):
            if listItem[1] <= startime and listItem[2] >= startime:
                sindex = index
            if listItem[1] <= startime+self.appdur and listItem[2] >= startime+self.appdur:
                eindex = index + 1
            
            if sindex != 0 and eindex != 0:
                break
        
        if sindex == 0 and eindex == 0:
            print('fuck not find the accesslist')
        resList = accessList[sindex : eindex]
        return resList
                      
    def generateIPStr(self, index):
        x = 100
        y = 0
        z = 1
        if index <= 200:
            z = index
        elif index <= 200*200:
            y = index / 200
            z = index % 200
        elif index <= 200*200*100:
            x = index / (200*200) + 100
            index = index % (200*200) + 100
            if index <= 200:
                z = index
            elif index <= 200*200:
                y = index / 200
                z = index % 200
        IPstr = 'N8-'+str(int(x)) +'.'+ str(int(y))+'.' + str(int(z))+'.0'
        return IPstr

    def generateTheFaultFile(self):
        """
        from the self.appAccess list Generating the switch list between Ground Node and Sat
        """
        faultfile = open('./OutConfigfile/WDNscen.fault', 'w')
        
        for x in self.appAccess:
            IpStr = self.findThePathNodeTopIpStr(x[0])
            for y in self.IPandFault:
                if IpStr == y[0]:
                    y[1].append(x)
                    break
            else:
                temlist = []
                temlist.append(x)
                self.IPandFault.append(list((IpStr,temlist)))
        for fuck in self.IPandFault:
            IptempStr = fuck[0]
            # print(fuck[1])
            fuck[1].sort( key = lambda x: x[1])
            start = fuck[1][0][1]
            end = fuck[1][-1][2]
            if start > 0 :
                startStr = 'INTERFACE-FAULT ' + IptempStr[3:len(IptempStr)-1] + '2 0S ' + str(start) + 'S\n' 
                faultfile.write(startStr)
            endStr = 'INTERFACE-FAULT ' + IptempStr[3:len(IptempStr)-1] + '2 ' + str(end) + 'S 0S\n'
            faultfile.write(endStr)

            for index , g in enumerate(fuck[1]):
                if index + 1 == len(fuck[1]):
                    break
                nextG = fuck[1][index + 1]
                if g[2] < nextG[1]:
                    temStr = ('INTERFACE-FAULT ' + IptempStr[3:len(IptempStr)-1] + '2 0S ' + 
                                                             str(g[2]) + 'S ' +str(nextG[1])+'S\n')
                    faultfile.write(temStr)
      
    def findThePathNodeTopIpStr(self, pathNode):
        for x in self.GtoSatList:
            if x[1] == pathNode:
                return x[0]
            
            

            
            

            
            

        

            
                
            
        


        







# if __name__ == '__main__':
#     nodelist = readNodeConfigFiel()

    # print(nodelist)


