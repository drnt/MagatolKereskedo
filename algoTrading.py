import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import itertools

# close all open figures if any
plt.close('all')

#generating a semi deterministic time series
# with two cycle and random noise

# parameters
steps = 2000
sigma = 3.0
fr1, ampl1 = 100.0, 10.0 #long term cycle
fr2, ampl2 = 40.0, 20.0 #short term cycle

# generate random numbers
np.random.seed(1)
e = np.random.randn(steps)

# calculate time series
t = np.arange(1.0,steps + 1.0)
result = ampl1 * np.sin(math.pi*t/fr1) + ampl2 * np.sin(math.pi*t/fr2) + sigma*e


# plot the result
"""
plt.plot(result)
plt.ylabel('some numbers')
plt.show()
"""

# -------- calculate moving averages based on pandas
# --- create DataFrame
df = pd.DataFrame(result,columns=['A'])

# create moving averages
movaveSteps=[5,10,20,30,60]
movaveSteps=[2,5,7]

for step in movaveSteps:    
    df[('mave',step)]= pd.rolling_mean(df['A'],window=step)

# calculate differences between movaves
# and lags
lagSteps=[1,5,20]
lagSteps=[0,1,2]
comb_movaveSteps = list(itertools.combinations(movaveSteps,2))
for comb in comb_movaveSteps:
    df[('maveDiff',comb)]= df['mave',comb[0]]-df['mave',comb[1]]
    for lag in lagSteps:
        df[('maveDiffShifted',comb,lag)]=df[('maveDiff',comb)].shift(lag)


#calculate profitability of holding position for one step





        

print df.tail(10)

