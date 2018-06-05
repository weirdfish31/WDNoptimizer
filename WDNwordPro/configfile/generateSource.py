import os
import sys
import numpy as np 

if len(sys.argv) != 5:
    sys.exit(1)

numPlans = int(sys.argv[1])
numSats = int(sys.argv[2])

numSorce = int(sys.argv[3])

pre = sys.argv[4]

resSouce = []

for i in range(numSorce):
    while True:
        p = np.random.randint(1, numPlans + 1)
        s = np.random.randint(1, numSats + 1)
        if (p,s) not in resSouce:
            resSouce.append((p,s))
            break

def fhelper(x,mx):
    if mx > 9 and x <= 9:
        return '0'+ str(x)
    else:
        return str(x)
f = open('./Source.RSsat', 'w')
for item in resSouce:
    sName = pre + fhelper(item[0], numPlans) + fhelper(item[1], numSats)
    f.write(sName+'\n')
print('end select')

    


