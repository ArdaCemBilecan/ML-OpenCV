import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv('Ads_CTR_Optimisation.csv')

import math
N = 10000 # 10.000 click
d = 10  
#Ri(n)
awards = [0] * d # firstly all awards are zero
#Ni(n)
clicks = [0] * d 
total = 0 # toplam odul
chosen = []
for n in range(1,N):
    ad = 0 # chosen advert
    max_ucb = 0
    for i in range(0,d):
        if(clicks[i] > 0):
            avg = awards[i] / clicks[i]
            delta = math.sqrt(3/2* math.log(n)/clicks[i]) # formula
            ucb = avg + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i          
    chosen.append(ad)
    clicks[ad] = clicks[ad]+ 1
    award = datas.values[n,ad] 
    awards[ad] = awards[ad]+ award
    total = total + award
print('Total Award:')   
print(total)

plt.hist(chosen)
plt.show()







