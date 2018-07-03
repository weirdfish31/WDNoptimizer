# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:09:45 2018

@author: WDN
"""
from __future__ import print_function
import sqlite3
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import VAR
import numpy as np
import pandas as pd

#!/usr/bin/python


mpl.style.use('ggplot')
conn = sqlite3.connect('./OutConfigfile/Second001.db')
print ("Opened database successfully")
c = conn.cursor()
cursor =c.execute("SELECT Timestamp, BytesSent, BytesReceived, ApplicationName, MessageCompletionRate, Throughput, Delay, Jitter, Hopcount  from APPLICATION_Summary")
BR=[]
BS=[]
MCR=[]
Throughput=[]
Delay=[]
Jitter=[]
Hopcount=[]
winseg=500
para=['BS','BR','Throughput','Delay','Jitter','Hopcount']
null=[0,0,0,0,0,0]
for row in cursor:
    if row[3]=='Facility40-Facility43-FTP':
        BS.append(row[1])
        BR.append(row[2])
        MCR.append(row[4])
        Throughput.append(row[5])
        Delay.append(row[6])
        Jitter.append(row[7])
        Hopcount.append(row[8])
df=pd.DataFrame()
df3_diff=pd.DataFrame()
grangerpattern=pd.DataFrame({'BS':null, 'BR':null,'Throughput':null,'Delay':null,'Jitter':null,'Hopcount':null,'cause':para})
print(grangerpattern)
grangerpattern=grangerpattern[['cause','BS','BR','Throughput','Delay','Jitter','Hopcount']]
print(grangerpattern)
df['BR']=BR
df['BS']=BS
df['MCR']=MCR
df['Throughput']=Throughput
df['Delay']=Delay
df['Jitter']=Jitter
df['Hopcount']=Hopcount
df2 =pd.DataFrame(df[df.BS>0]).reset_index()
print(df2)
#del df2['index']


#first order difference
df3_diff['BR']=np.diff(df2.BR)
df3_diff['BS']=np.diff(df2.BS)
df3_diff['Hopcount']=np.diff(df2.Hopcount)
df3_diff['Throughput']=np.diff(df2.Throughput)
df3_diff['Delay']=np.diff(df2.Delay)
df3_diff['Jitter']=np.diff(df2.Jitter)
print(df3_diff)
# =============================================================================
# plt.figure(figsize=(24,10))
# plt.style.use("ggplot")
# plt.subplot(211)
# plt.plot(df3_diff['BR'],'-',linewidth = '1')
# plt.title('BR',fontsize=15)
# plt.subplot(212)
# plt.plot(df3_diff['BS'],'-',linewidth = '1')
# plt.title('BS',fontsize=15)
# plt.savefig("./est.jpg")
# =============================================================================
date1 = pd.date_range('11/19/2017', periods=len(df3_diff.index), freq='S')
df3_diff.index=pd.DatetimeIndex(date1)
#stability test
# =============================================================================
# plot_acf(df3_diff.BR).show()
# plot_acf(df3_diff.BS).show()
# plot_acf(df3_diff.Hopcount).show()
# plot_acf(df3_diff.Throughput).show()
# plot_acf(df3_diff.Delay).show()
# plot_acf(df3_diff.Jitter).show()
# =============================================================================


#set  Vector autoregression model

# make a VAR model print granger causality pattern

for i in range(int(10000/winseg)):
    model = VAR(df3_diff[i*winseg:(i+1)*winseg])
#    print(df3_diff[i*winseg:(i+1)*winseg])
    model.select_order(5)
    results = model.fit(maxlags=5, ic='aic')
#    print(grangerpattern)
    for para1 in para:
        for para2 in para:
            if(results.test_causality(para1, [para2],signif=0.05, kind='f')['conclusion'])=="reject":
                grangerpattern.loc[[para.index(para2)],[para1]]=1
            else:
                grangerpattern.loc[[para.index(para2)],[para1]]=0
    print(grangerpattern)
    grangerpattern.to_csv(path_or_buf='./pattern/seg'+str(i)+'.txt', sep=' ')

# =============================================================================
# results.test_causality('BR', ['BS'],signif=0.01, kind='f')
# results.test_causality('Hopcount', ['BR'],signif=0.01, kind='f')
# results.test_causality('Hopcount',['Delay'],signif=0.01,kind='f')
# =============================================================================

