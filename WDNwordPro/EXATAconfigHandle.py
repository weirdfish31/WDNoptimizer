import time
import EXATAreadhelper
# starttime = datetime.datetime.strptime('3 Nov 2013 04:00:00', '%d %b %Y %H:%M:%S')

def handlefun():
    s = time.clock()
    readconfig = EXATAreadhelper.readhelper()
#    print(readconfig.gnodelist)
#    print(readconfig.nodelist)
    readconfig.createlinktop()
    readconfig.csvreadhandle()
    readconfig.generateGStopFault()
    readconfig.generateGGapp()
    
# =============================================================================
#     readconfig.generateTCPwindow()
#     readconfig.generateTheFaultFile()
# =============================================================================
    e = time.clock()
    print('time is ' + str(e - s) + 's')
    for x in readconfig.GtoSatList:
        print(x)

if __name__ == '__main__':
     handlefun()


