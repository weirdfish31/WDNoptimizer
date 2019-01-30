# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:13:01 2019

@author: WDN
用来测试WDNTagDataHandler,主要是进行模型（）之间的迭代比较MSE
"""
import pandas as pd
import numpy as np
import WDNTagDataHandler
import matplotlib.pyplot as plt 
import seaborn as sns
import WDNoptimizer
from sklearn.metrics import mean_squared_error



"选择路径与文件名，读取TXT文件中迭代的state数据=============================================="
vbr=24000
#vbr=50000
"-----------------------------------------------------------------------------------------"
#datapath='D:/WDNoptimizer/LHSprior/'#LHS先验数据的存放位置
datapath='D:/WDNoptimizer/LHSMSE50000/'#LHS先验数据的存放位置
#datapath='D:/WDNoptimizer/LHSMSE24000/'#LHS先验数据的存放位置
"-----------------------------------------------------------------------------------------"
#statefilename="./history/priorstate_test.txt"#存储的先验数据的state列表txt文件
#statefilename="./history/priorstate50000_all.txt"#存储的先验数据的state列表txt文件
statefilename="./history/priorstate24000_all.txt"#存储的先验数据的state列表txt文件
"-----------------------------------------------------------------------------------------"
#taggedDatafilename='./LabelData/LHSprior.txt'
#taggedDatafilename='./LabelData/LHS50000.txt'
taggedDatafilename='./LabelData/LHS24000.txt'
"-----------------------------------------------------------------------------------------"
traindatafilename='./LabelData/LHSprior_test.txt'
#iterdatafilename='./LabelData/ITER_MPP24000_k5_i30_t10_p1_test.txt'
#iterdatafilename='./LabelData/ITER_MPP24000_k5_i60_t10_p1_test.txt'
iterdatafilename='./LabelData/HPP24000_k5_i60_t10.txt'
iterdatafilename='./LabelData/GPR24000_k5_i60_t10.txt'
griddatafilename='./LabelData/2DGMM(16000_8000-36000)_test.txt'
#iterdatafilename='./LabelData/ITER_GMM24000_i60_t10_test.txt'
readtestfilename='./LabelData/LHS24000_test.txt'
#readtestfilename='./LabelData/24000testset.txt'
'-----------------------------------------------------------------------------------------'

#newdatapath='D:/WDNoptimizer/GMM_i60_t10/'#新产生的数据的存放位置
#newdatapath='D:/WDNoptimizer/GPR_i25_t10/'
#newdatapath='D:/WDNoptimizer/iter_09/'
#newdatapath='D:/WDNoptimizer/iter_00/'
#newdatapath='D:/WDNoptimizer/HPP_k5_i60_t10_p1/'
#newdatapath='D:/WDNoptimizer/MPP_k5_i30_t10_p1/'
newdatapath='D:/WDNoptimizer/GPR_k5_i60_t10/'
#newdatapath='D:/WDNoptimizer/MPP_k5_i60_t10_p1/'

#savedataname='./LabelData/ITER_GMM24000_i60_t10.txt'
#savedataname='./LabelData/ITER_MPP24000_k5_i30_t10_p1.txt'
#savedataname='./LabelData/ITER_GPR24000_i25_t10.txt'
#savedataname='./LabelData/ITER_iter0_9.txt'
#savedataname='./LabelData/ITER_iter0_0.txt'
#savedataname='./LabelData/ITER_HPP24000_k5_i60_t10_p1.txt'
savedataname='./LabelData/ITER_GPR24000_k5_i60_t10.txt'
#savedataname='./LabelData/ITER_MPP24000_k5_i60_t10_p1.txt'



listGMM_i60_t10=[[63709 ,63998 ],[63976 ,283   ],[24064 ,63980 ],[3     ,63876 ],[63973 ,35465 ],[34495 ,63986 ],
                 [45181 ,48150 ],[45054 ,48390 ],[33249 ,63986 ],[53340 ,34500 ],[42206 ,44016 ],[40228 ,45406 ],
                 [28527 ,63949 ],[63955 ,27404 ],[63706 ,217   ],[35303 ,49160 ],[23476 ,63994 ],[37209 ,46716 ],
                 [37468 ,46169 ],[32289 ,49794 ],[44269 ,37267 ],[63885 ,97    ],[22327 ,63966 ],[32624 ,63994 ],
                 [42535 ,39082 ],[32249 ,47910 ],[32600 ,47396 ],[41789 ,37975 ],[33214 ,46769 ],[43986 ,34883 ],
                 [32157 ,47348 ],[63716 ,83    ],[32437 ,49253 ],[42618 ,36109 ],[32586 ,48885 ],[44583 ,33893 ],
                 [31098 ,49494 ],[31923 ,46178 ],[31281 ,49419 ],[30590 ,50553 ],[30665 ,50260 ],[19424 ,63995 ],
                 [29946 ,50780 ],[33361 ,44745 ],[30368 ,50305 ],[50054 ,23937 ],[29988 ,50953 ],[32619 ,45542 ],
                 [31920 ,46328 ],[29687 ,51218 ],[30208 ,50682 ],[30433 ,50334 ],[34874 ,47880 ],[37153 ,41022 ],
                 [29175 ,51017 ],[35732 ,47203 ],[35318 ,63967 ],[28391 ,49667 ],[36563 ,43955 ],[28394 ,49467 ]]

listGPR_i60_t3=[[63988 ,63060 ],[63979 ,47    ],[482   ,63975 ],[63986 ,33949 ],[538   ,63993 ],[36558 ,63960 ],
                [36792 ,63937 ],[48721 ,41062 ],[45332 ,43212 ],[63967 ,32802 ],[39670 ,48671 ],[34465 ,63962 ],
                [43896 ,42928 ],[63796 ,188   ],[33883 ,63965 ],[47457 ,38319 ],[46853 ,38737 ],[63987 ,34471 ],
                [39297 ,45425 ],[39465 ,45347 ],[63822 ,63819 ],[38843 ,45217 ],[45795 ,36783 ],[36496 ,47493 ],
                [46133 ,36282 ],[34219 ,49887 ],[34131 ,50097 ],[34031 ,50030 ],[30725 ,63996 ],[44420 ,38519 ],
                [42790 ,40597 ],[35925 ,46644 ],[35351 ,46947 ],[35464 ,46955 ],[35196 ,47188 ],[45238 ,36876 ],
                [34430 ,47700 ],[46008 ,35768 ],[31944 ,49767 ],[31601 ,50071 ],[31486 ,50094 ],[31977 ,49792 ],
                [31375 ,50182 ],[32198 ,63970 ],[38228 ,45129 ],[38356 ,45030 ],[32659 ,47497 ],[32100 ,47936 ],
                [40923 ,43976 ],[31637 ,48206 ],[30943 ,48581 ],[30598 ,48622 ],[30105 ,49038 ],[40257 ,44526 ],
                [28203 ,50370 ],[28449 ,50303 ],[27881 ,50677 ],[28039 ,63981 ],[28307 ,49208 ],[37792 ,44687 ]]

iter0_9=[[63962, 63909], [63972, 218], [9, 63836], [63977, 31048], [30965, 63967], [34415, 63976], [693, 63983], 
         [44577, 42897], [41411, 44445], [63572, 38], [35239, 46748], [48234, 30737], [32246, 46846], [45938, 31509], 
         [35230, 63998], [29892, 45772], [44206, 30474], [29049, 46222], [29559, 45840], [46735, 19362], [37667, 40023], 
         [37182, 40375], [27724, 47216], [39355, 36747], [63969, 30803], [28218, 46361], [27347, 47160], [36095, 38965], 
         [26589, 63989], [26110, 49825], [28726, 63969], [42705, 63965], [26302, 47293], [34007, 39634], [26203, 47530], 
         [37809, 32993], [26287, 47444], [24457, 63981], [35220, 36998], [44994, 21664], [46097, 72], [41264, 27707], 
         [39762, 30598], [39015, 31451], [26089, 49806], [36778, 34568], [38935, 43331], [39546, 29392], [46723, 23216], 
         [32233, 37749], [32163, 38034], [46526, 22997], [15109, 63974], [24358, 49055], [26154, 63989], [24675, 63981], 
         [33437, 35404], [63991, 22700], [25901, 45785], [35752, 32587], [25348, 47192], [24440, 48063], [36468, 47040],
         [24648, 47199], [36471, 31810], [38161, 47560], [45630, 20847], [31449, 36912], [25039, 63983], [24064, 50190], 
         [40327, 63990], [24189, 49662], [33065, 34653], [46174, 19784], [24514, 49856], [35592, 31800], [24773, 63996], 
         [25115, 63991], [22047, 63950], [24239, 52173], [26373, 63999], [24102, 51303], [18041, 52605], [33556, 50837], 
         [39910, 63995], [34084, 63996], [32946, 49818], [32183, 49980], [31568, 49907], [31043, 37833], [63955, 63860], 
         [26393, 51843], [26577, 51746], [23622, 63992], [38929, 28645], [45900, 18956], [36513, 30961], [31415, 37394], 
         [38538, 29474], [43695, 21168]]

iter0_0=[[57405, 47321], [35159, 37803], [35849, 35489], [35228, 36138], [35874, 36224], [35253, 37219], [35445, 36353], 
         [35737, 36082], [43942, 24887], [32747, 37895], [33705, 35596], [33675, 35152], [33708, 36548], [33751, 36086], 
         [34088, 35220], [33959, 34654], [34160, 35855], [34118, 35518], [34176, 36187], [34119, 35597], [34085, 35137], 
         [34308, 34868], [34159, 35059], [34304, 34892], [34244, 34807], [34294, 34678], [34163, 34765], [34321, 34523], 
         [33754, 33483], [34020, 34924], [33604, 33249], [34019, 35101], [33890, 34857], [34097, 34626], [34041, 34616],
         [33944, 34610], [33443, 32358], [33846, 34929], [33811, 34859], [33614, 34926], [33747, 35073], [32991, 33175], 
         [32901, 33125], [32864, 33205], [33754, 35517], [33606, 35326], [32953, 32736], [32831, 32819], [32744, 32772], 
         [33447, 36056], [32834, 37365], [32985, 37292], [33778, 35886], [33856, 35935], [32908, 32710], [33778, 35901], 
         [33734, 35898], [32834, 32564], [33860, 35683], [33864, 35580], [33027, 31985], [32823, 32248], [33652, 36052], 
         [33647, 36149], [33223, 37133], [31963, 39743], [30364, 37772], [32902, 38066], [33090, 37736], [29144, 40721], 
         [30787, 37824], [31387, 40896], [29262, 40447], [32205, 39682], [30949, 37239], [31171, 36847], [32117, 39213], 
         [32384, 38929], [32592, 38684], [31519, 40910], [31860, 40307], [31170, 36977], [33199, 37867], [30955, 37691], 
         [33302, 37933], [31649, 36418], [33592, 37682], [33491, 37543], [33462, 37534], [33163, 38402], [33237, 38155], 
         [33435, 38076], [31882, 36083], [32000, 35972], [32146, 35770], [32150, 35850], [32074, 35794], [33573, 37566], 
         [32218, 35511], [32277, 35543]]

listGPR_i25_t10=[[22605, 32779], [22635, 32892], [22663, 32814], [22620, 33077], [22630, 33037], [22650, 33078], 
                 [22698, 33099], [22735, 33144], [22637, 33191], [22702, 33057], [22638, 33215], [22672, 33155],
                 [22695, 33179], [22694, 33081], [22680, 33141], [22688, 33244], [22653, 33241], [22780, 33173], 
                 [22595, 33267], [22623, 33306], [22695, 33256], [22734, 33334], [22723, 33364], [22662, 33399], 
                 [22592, 33415]]

listMPP_k5_i30_t10_p1=[[63701, 63986], [63934, 63958], [32, 63966], [93, 63795], [41, 63832], [122, 63901], 
                       [107, 63951], [25, 63986], [47, 63945], [38, 63836], [91, 63858], [28, 63802],  
                       [21, 63785], [123, 63954], [54, 63986], [126, 63765], [90, 63996], [104, 63922], [183, 63991],
                       [187, 63980], [145, 63951], [51, 63674], [52, 63702], [3, 63652], [3, 59988], [22642, 31954],
                       [22738, 32552], [22649, 33128], [22850, 33467]]

listMPP_k5_i60_t10_p1=[[62504, 168], [62419, 302], [62429, 187], [62577, 231], [62357, 298], [62486, 92], [62509, 177], 
                       [62448, 296], [62571, 230], [62382, 156], [62449, 184], [62412, 186], [62360, 204], [62419, 161], 
                       [62447, 199], [62445, 197], [62434, 209], [62493, 169], [62498, 200], [62485, 227], [62373, 212], 
                       [62360, 234], [62422, 231], [62513, 251], [62405, 146], [62407, 231], [62509, 270], [62454, 182], 
                       [62388, 172], [62389, 314], [62485, 237], [62436, 109], [62501, 210], [62461, 182], [62586, 197], 
                       [62517, 189], [62506, 71], [62470, 147], [62471, 136], [62324, 239], [62399, 309], [62404, 261], 
                       [62457, 158], [62447, 245], [62477, 266], [62397, 143], [62404, 191], [62569, 231], [62404, 307], 
                       [62403, 229], [9119, 7702], [9101, 7765], [62498, 278], [19730, 48152], [19693, 48142], [19745, 48171],
                       [19770, 48191], [19792, 48144], [19777, 48112], [19702, 48160]]

HPP_k5_i60_t10_p1=[[63978, 63978], [63950, 63909], [63976, 63867], [63887, 63920], [63927, 63765], [63942, 63848],
                   [63807, 63792], [63984, 63524], [63956, 63983], [63831, 63879], [63986, 63543], [63916, 63960],
                   [63969, 63836], [63961, 63847], [63961, 63953], [63976, 63865], [63983, 63915], [63954, 63960], 
                   [63859, 63848], [63982, 63904], [63953, 63814], [63980, 63949], [63862, 63976], [63877, 63809], 
                   [63913, 63831], [63873, 63958], [63897, 63889], [63917, 63874], [63927, 63919], [63991, 63733], 
                   [63917, 63950], [63974, 63666], [63953, 63814], [63888, 63947], [63954, 63714], [63974, 63943], 
                   [84, 63916], [70, 63953], [75, 63905], [617, 63978], [174, 63851], [409, 63938], [46, 63868], 
                   [150, 63960], [152, 63955], [157, 63994], [58, 63907], [97, 63957], [63961, 40113], [37179, 33577], 
                   [33370, 30402], [31792, 28680], [30867, 27753], [30309, 27407], [29834, 27168], [29431, 27008], [29137, 26881],
                   [28904, 26864], [28669, 26957], [28476, 26951]]

GPR_k5_i60_t10=[[63886, 63854], [63957, 63852], [63926, 63981], [63983, 63957], [63937, 63621], [63924, 63982], 
                [63840, 63906], [63995, 63680], [63921, 63821], [63812, 63951], [63983, 63769], [63866, 63859], 
                [63879, 63901], [63985, 63860], [63981, 63768], [63994, 63843], [63893, 63974],
                [63864, 63985], [63903, 63942], [63869, 63815], [63984, 63966], [63900, 63692], [63964, 63946], 
                 [63928, 63881], [63983, 63774], [63988, 63777], [63974, 63737], [63919, 63998], [63968, 63991], 
                 [121, 63993], [94, 63975], [129, 63839], [180, 63867], [30, 63990], [98, 63993], [13, 63955], 
                 [79, 63999], [93, 63761], [225, 63994], [23, 63962], [187, 63951], [25, 63847], [186, 63999],
                 [219, 63959], [78, 63942], [37, 63885], [830, 63974], [4790, 63443], [34725, 35576], [32426, 33090], 
                 [30909, 31352], [29983, 30206], [29241, 29556], [28810, 29190], [28551, 28755], [28245, 28629], [27922, 28354],
                 [27745, 28426], [27496, 28234]]

teaser=WDNTagDataHandler.TaggedDataHandler()#实例化


'读取原始数据进行分类标签保存（先验数据，迭代数据）'
#teaser.PriorDataTagWriter(vbrs=vbr,count=20,path=datapath,filename=statefilename,savefilename=taggedDatafilename)#先验数据的处理
teaser.IterDataTagWriter(vbrs=vbr,count_i=10,path=newdatapath,QPlist=GPR_k5_i60_t10,savefilename=savedataname)#迭代数据的处理
#teaser.GridDataTagWriter()#主观栅格数据的处理

'MSE========================================================================================='
traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
iterdata=teaser.LabelDataReader(filename=iterdatafilename)#迭代数据的读取
griddat=teaser.LabelDataReader(filename=griddatafilename)#栅格数据的读取
traindata=traindata.append(iterdata).reset_index(drop=True)#迭代数据加入先验数据
print(traindata)
MPPMSE=[]
GPRMSE=[]
RFMSE=[]
"循环迭代的数据，对每次迭代的模型进行MSE的计算"
"这里的MSE计算根据模型的不同分别进行，MPP模型中的各簇与仿真值对应的各簇进行计算，GPR中直接进行计算"
for i in range(int(len(traindata)/2)):
    trainset=traindata[0:(2*(i+1))]
    print(trainset)
    gamer=WDNTagDataHandler.ModelCompareHandler()
    '建模'
    gamer.MPPmodelRebuilder(trainset)
    gamer.GPRmodelRebuiler(trainset)
    gamer.RFmodelRebuilder(trainset)
    '预测'
    MPPdata=gamer.MPPpredicter(testdata)
    GPRdata=gamer.GPRpredicter(testdata)
    RFdata=gamer.RFpredicter(testdata)
    
    'MSE'
    MPPMSE1=gamer.MPPMSE(testdata,MPPdata)
    GPRMSE1=gamer.GPRMSE(testdata,GPRdata)
    RFMSE1=gamer.RFMSE(testdata,RFdata)
    MPPMSE.append(MPPMSE1)
    GPRMSE.append(GPRMSE1)
    RFMSE.append(RFMSE1)
#    MPPMSE.append(mean_squared_error(testdata['value'][testdata['label']==1],MPPdata['mean1'])+mean_squared_error(testdata['value'][testdata['label']==0],MPPdata['mean0']))
#    GPRMSE.append(mean_squared_error(testdata['value'][testdata['label']==1],GPRdata['mean'])+mean_squared_error(testdata['value'][testdata['label']==0],GPRdata['mean']))
#    RFMSE.append(mean_squared_error(testdata['value'][testdata['label']==1],RFdata['mean1'])+mean_squared_error(testdata['value'][testdata['label']==0],RFdata['mean0']))
"绘图"
#teaser.ComparePrinter(MPPMSE,GPRMSE,RFMSE,style='MSE')
teaser.ZoominPrinter(MPPMSE,GPRMSE,RFMSE,style='MSE')
print(RFMSE)
print(MPPMSE)
print(GPRMSE)
#print(MPPMSE[0],GPRMSE[0])
#print(MPPMSE[1],GPRMSE[1])
#print(MPPMSE[2],GPRMSE[2])
#print(MPPMSE[5],GPRMSE[5])
#print(MPPMSE[11],GPRMSE[11])
#print(MPPMSE[30],GPRMSE[31])
#print(MPPMSE[50],GPRMSE[50])


'R方=========================================================================='
traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
iterdata=teaser.LabelDataReader(filename=iterdatafilename)#迭代数据的读取
traindata=traindata.append(iterdata).reset_index(drop=True)#迭代数据加入先验数据
r_mpp=[]
r_gpr=[]
r_rfr=[]
"循环迭代的数据，对每次迭代的模型进行MSE的计算"
"这里的MSE计算根据模型的不同分别进行，MPP模型中的各簇与仿真值对应的各簇进行计算，GPR中直接进行计算"
'目前的随机森林模型中不存在方差的特征值，需要考虑如何得到'
for i in range(int(len(traindata)/2)):
    trainset=traindata[0:(2*(i+1))]
    print(trainset)
    gamer=WDNTagDataHandler.ModelCompareHandler()
    '建模'
    gamer.MPPmodelRebuilder(trainset)
    gamer.GPRmodelRebuiler(trainset)
    gamer.RFmodelRebuilder(trainset)
    '预测'
    MPPdata=gamer.MPPpredicter(testdata)
    GPRdata=gamer.GPRpredicter(testdata)
    RFdata=gamer.RFpredicter(testdata)
    'MSE'
    MPPMSE1=gamer.MPPMSE(testdata,MPPdata)
    GPRMSE1=gamer.GPRMSE(testdata,GPRdata)
    RFMSE1=gamer.RFMSE(testdata,RFdata)
    'R方'
    r_mpp.append(1-MPPMSE1/(np.var(testdata['value'][testdata['label']==1])+np.var(testdata['value'][testdata['label']==0])))
    r_gpr.append(1-GPRMSE1/(np.var(testdata['value'][testdata['label']==1])+np.var(testdata['value'][testdata['label']==0])))
    r_rfr.append(1-RFMSE1/(np.var(testdata['value'][testdata['label']==1])+np.var(testdata['value'][testdata['label']==0])))
   
"绘图"
teaser.ZoominPrinter(r_mpp,r_gpr,r_rfr,style='R-Squared')

print(r_mpp)
print(r_gpr)
print(r_rfr)




'仿真值与预测值的对比============================================================='
'读数据'
traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
iterdata=teaser.LabelDataReader(filename=iterdatafilename)#迭代数据的读取

mpppredictlist=[]
gprpredictlist=[]
rfpredictlist=[]
simulationlist=[]
'基于迭代数据进行迭代建模，得到下一个点的预测值'
for i in range(int(len(iterdata)/2)):
    trainset=traindata.append(iterdata[0:(i*2)]).reset_index(drop=True)
    print(trainset)
    gamer=WDNTagDataHandler.ModelCompareHandler()
    '建模'
    gamer.MPPmodelRebuilder(trainset)
    gamer.GPRmodelRebuiler(trainset)
    gamer.RFmodelRebuilder(trainset)
    '预测'
    MPPdata=gamer.MPPpredicter(testdata)
    GPRdata=gamer.GPRpredicter(testdata)
    RFdata=gamer.RFpredicter(testdata)
    '这里的比较有待商榷额，因为是不同模型，目前是这种方式进行value的比较'
    mppoutput=MPPdata['mean0'][i]*MPPdata['prob0'][i]+MPPdata['mean1'][i]*MPPdata['prob1'][i]
    mpppredictlist.append(mppoutput)
    gproutput=GPRdata['mean'][i]
    gprpredictlist.append(gproutput)
    rfoutput=RFdata['mean0'][i]
    rfpredictlist.append(rfoutput)
    '仿真的数据'
    simuoutput=iterdata['value'][2*i]*iterdata['prob'][2*i]+iterdata['value'][2*i+1]*iterdata['prob'][2*i+1]
    simulationlist.append(simuoutput)


"绘图"
teaser.ValueComparePrinter(mpppredictlist,gprpredictlist,rfpredictlist,simulationlist,style='predict-simulated')

#sns.set_style("whitegrid")
#plt.figure('Line fig',figsize=(20,6))
#plt.xlabel('Iteration Times')
#plt.ylabel('predict-simulated')
#plt.title('value ',fontsize='xx-large')
#
#plt.scatter(x=range(len(mpppredictlist)),y=mpppredictlist,marker='*',c='r')
#plt.scatter(x=range(len(gprpredictlist)),y=gprpredictlist,marker='.',c='black')
#plt.scatter(x=range(len(rfpredictlist)),y=rfpredictlist,marker='o',c='blue')
#plt.plot(mpppredictlist,color='r', linewidth=2, alpha=0.6,label='MPP')
#plt.plot(gprpredictlist,color='black', linewidth=2, alpha=0.6,label='GPR')
#plt.plot(rfpredictlist,color='blue', linewidth=2, alpha=0.6,label='RF')
#plt.plot(simulationlist,color='green', linewidth=2, alpha=0.6,label='simulate')
#plt.legend(fontsize='x-large')

'仿真值与预测值的对比===分簇绘图======================='
'读数据'
traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
iterdata=teaser.LabelDataReader(filename=iterdatafilename)#迭代数据的读取
mpppredictlist0=[]
mpppredictlist1=[]
gprpredictlist=[]
rfpredictlist=[]
simulationlist0=[]
simulationlist1=[]
'基于迭代数据进行迭代建模，得到下一个点的预测值'
for i in range(int(len(iterdata)/2)):
    trainset=traindata.append(iterdata[0:(i*2)]).reset_index(drop=True)
    print(trainset)
    gamer=WDNTagDataHandler.ModelCompareHandler()
    '建模'
    gamer.MPPmodelRebuilder(trainset)
    gamer.GPRmodelRebuiler(trainset)
    gamer.RFmodelRebuilder(trainset)
    '预测'
    MPPdata=gamer.MPPpredicter(testdata)
    GPRdata=gamer.GPRpredicter(testdata)
    RFdata=gamer.RFpredicter(testdata)
    '这里的比较有待商榷额，因为是不同模型，目前是这种方式进行value的比较'
#    mppoutput0=MPPdata['mean0'][i]*MPPdata['prob0'][i]+MPPdata['mean1'][i]*MPPdata['prob1'][i]

    mpppredictlist0.append(MPPdata['mean0'][i])
    mpppredictlist1.append(MPPdata['mean1'][i])
    gproutput=GPRdata['mean'][i]
    gprpredictlist.append(gproutput)
    rfoutput=RFdata['mean0'][i]
    rfpredictlist.append(rfoutput)
    '仿真的数据'
#    simuoutput=iterdata['value'][2*i]*iterdata['prob'][2*i]+iterdata['value'][2*i+1]*iterdata['value'][2*i+1]
    simulationlist0.append(iterdata['value'][2*i])
    simulationlist1.append(iterdata['value'][2*i+1])

"绘图"

sns.set_style("whitegrid")
plt.figure('Line fig',figsize=(20,6))
plt.xlabel('Iteration Times')
plt.ylabel('predict-simulated')
plt.title('value ',fontsize='xx-large')

plt.scatter(x=range(len(gprpredictlist)),y=gprpredictlist,marker='.',c='black')
plt.scatter(x=range(len(rfpredictlist)),y=rfpredictlist,marker='o',c='blue')


plt.plot(mpppredictlist0,color='r', linewidth=2, alpha=0.6,label='MPP0')
plt.plot(mpppredictlist1,color='r', linewidth=2, alpha=0.6,label='MPP1')
plt.plot(gprpredictlist,color='black', linewidth=2, alpha=0.6,label='GPR')
plt.plot(rfpredictlist,color='blue', linewidth=2, alpha=0.6,label='RF')
plt.plot(simulationlist0,color='green', linewidth=2, alpha=0.6,label='simulate0')
plt.plot(simulationlist1,color='green', linewidth=2, alpha=0.6,label='simulate1')
plt.legend(fontsize='x-large')





'不同先验数据采样方式得到的不同的模型结果'
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
griddata=teaser.LabelDataReader(filename=griddatafilename)#栅格数据的读取
MPPMSE_LHS=[]
MPPMSE_RG=[]

"循环迭代的数据，对每次迭代的模型进行MSE的计算"
"这里的MSE计算根据模型的不同分别进行，MPP模型中的各簇与仿真值对应的各簇进行计算，GPR中直接进行计算"
for i in range(int(len(traindata)/2)):
    trainsetLHS=traindata[0:(2*(i+1))]
    trainsetRG=griddata[0:(2*(i+1))]
    gamerlhs=WDNTagDataHandler.ModelCompareHandler()
    gamerrg=WDNTagDataHandler.ModelCompareHandler()
    '建模'
    gamerrg.MPPmodelRebuilder(trainsetRG)
    gamerlhs.MPPmodelRebuilder(trainsetLHS)

    '预测'
    MPPdatarg=gamerrg.MPPpredicter(testdata)
    MPPdatalhs=gamerlhs.MPPpredicter(testdata)
    'MSE'
    MPPMSElhs=gamerlhs.MPPMSE(testdata,MPPdatalhs)
    MPPMSErg=gamerrg.MPPMSE(testdata,MPPdatarg)

    MPPMSE_LHS.append(MPPMSElhs)
    MPPMSE_RG.append(MPPMSErg)

"绘图"
print(MPPMSE_LHS)
print(MPPMSE_RG)

sns.set_style("whitegrid")
plt.figure('Line fig',figsize=(20,6))
plt.xlabel('Data Count')
plt.ylabel('MSE')
plt.title('value ',fontsize='xx-large')

plt.scatter(x=range(len(MPPMSE_LHS)),y=MPPMSE_LHS,marker='.',c='black')
plt.scatter(x=range(len(MPPMSE_RG)),y=MPPMSE_RG,marker='o',c='blue')

plt.plot(MPPMSE_LHS,color='r', linewidth=2, alpha=0.6,label='HPP_LHS')
plt.plot(MPPMSE_RG,color='b', linewidth=2, alpha=0.6,label='HPP_RANDOMGRID')
plt.legend(fontsize='x-large')

'----------------------------------------------------------------------------------------------------------'
traindatafilename='./LabelData/LHSprior_test.txt'
#iterdatafilename='./LabelData/ITER_MPP24000_k5_i30_t10_p1_test.txt'
#iterdatafilename='./LabelData/ITER_MPP24000_k5_i60_t10_p1_test.txt'
iterdatafilename1='./LabelData/HPP24000_k5_i60_t10.txt'
iterdatafilename2='./LabelData/GPR24000_k5_i60_t10.txt'
griddatafilename='./LabelData/2DGMM(16000_8000-36000)_test.txt'
#iterdatafilename='./LabelData/ITER_GMM24000_i60_t10_test.txt'
readtestfilename='./LabelData/LHS24000_test.txt'
#readtestfilename='./LabelData/24000testset.txt'

iterdatafilename1='./LabelData/HPP24000_k5_i60_t10.txt'
iterdatafilename2='./LabelData/GPR24000_k5_i60_t10.txt'

'仿真值与预测值的对比===分簇绘图======================='
'读数据'
traindata=teaser.LabelDataReader(filename=traindatafilename)#训练数据读取
testdata=teaser.LabelDataReader(filename=readtestfilename)#测试数据的读取
iterdata1=teaser.LabelDataReader(filename=iterdatafilename1)#迭代HPP数据的读取
iterdata2=teaser.LabelDataReader(filename=iterdatafilename2)#迭代GPR数据的读取
mpppredictlist=[]
gprpredictlist=[]

simulationlist1=[]
simulationlist2=[]
d1=[]
d2=[]
'基于迭代数据进行迭代建模，得到下一个点的预测值'
for i in range(int(len(iterdata1)/2)):
    trainset1=traindata.append(iterdata1[0:(i*2)]).reset_index(drop=True)
    trainset2=traindata.append(iterdata2[0:(i*2)]).reset_index(drop=True)
    gamer=WDNTagDataHandler.ModelCompareHandler()
    '建模'
    gamer.MPPmodelRebuilder(trainset1)
    gamer.GPRmodelRebuiler(trainset2)
    '预测'
    HPPdata=gamer.MPPpredicter(testdata)
    GPRdata=gamer.GPRpredicter(testdata)
    '这里的比较有待商榷额，因为是不同模型，目前是这种方式进行value的比较'
    mppoutput=HPPdata['mean0'][i]*HPPdata['prob0'][i]+HPPdata['mean1'][i]*HPPdata['prob1'][i]
    mpppredictlist.append(mppoutput)
#    mpppredictlist0.append(HPPdata['mean0'][i])
#    mpppredictlist1.append(HPPdata['mean1'][i])
    gproutput=GPRdata['mean'][i]
    gprpredictlist.append(gproutput)
    '仿真的数据'
    simuoutput1=iterdata1['value'][2*i]*iterdata1['prob'][2*i]+iterdata1['value'][2*i+1]*iterdata1['value'][2*i+1]
    simulationlist1.append(simuoutput1)
    simulation2=iterdata2['value'][2*i]*iterdata2['prob'][2*i]+iterdata2['value'][2*i+1]*iterdata2['value'][2*i+1]
    simulationlist2.append(simulation2)
    '绝对差值'
    difference_hpp=abs(mppoutput-simuoutput1)
    d1.append(difference_hpp)
    difference_gpr=abs(gproutput-simulation2)
    d2.append(difference_gpr)


"绘图"

sns.set_style("whitegrid")
plt.figure('Line fig',figsize=(20,6))
plt.xlabel('Iteration Times')
plt.ylabel('predict-simulated')
plt.title('value ',fontsize='xx-large')
plt.scatter(x=range(len(gprpredictlist)),y=gprpredictlist,marker='.',c='black',label='GPR')
plt.scatter(x=range(len(mpppredictlist)),y=mpppredictlist,marker='o',c='blue',label='HPP')


#plt.plot(mpppredictlist,color='r', linewidth=2, alpha=0.6,label='HPP')
#plt.plot(gprpredictlist,color='black', linewidth=2, alpha=0.6,label='GPR')
plt.plot(simulationlist1,color='r', linewidth=2, linestyle=':',alpha=0.6,label='simulate_HPP')
plt.plot(simulationlist2,color='black', linewidth=2, linestyle=':',alpha=0.6,label='simulate_GPR')
plt.plot(d1,color='r', linewidth=2, alpha=0.6,label='difference_HPP')
plt.plot(d2,color='black', linewidth=2, alpha=0.6,label='difference_GPR')
plt.legend(fontsize='x-large')








































