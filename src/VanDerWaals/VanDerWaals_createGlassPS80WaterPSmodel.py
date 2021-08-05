#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:03:57 2019

@author: sm8857
"""

import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator, Rbf
from math import sqrt, log, exp, erf, sinh, cosh, pi
from numba import njit, float32, int32
from numba import jit
import time
import pandas as pd 




## LOAD THE DATA GENERATED AND FIT THE AI MODEL 

parameterSweep = pd.read_csv('VanDerWaals_parameterSweep_GlassPS80WaterPS.csv', delimiter = ',')

Ts = parameterSweep.Temperature
Cs = parameterSweep.Concentration
Hs = parameterSweep.Distance
As = parameterSweep.Hamaker


#print(time.time()-t0)
# #rbfi = Rbf(Ts, Cs, Hs, As)
h0s = np.logspace(-10,-5.5,100)
h0s = [log(h0s[i])/log(10) for i in range(len(h0s))]
T0s = np.linspace(20,100,20)
C0s = np.logspace(-5,3,20)
C0s = [log(C0s[i])/log(10) for i in range(len(C0s))]
data = np.zeros((len(T0s),len(C0s),len(h0s)))
for i in range(len(T0s)):
    for j in range(len(C0s)):
        for k in range(len(h0s)):
            #(k+j*len(h0s)+i*len(h0s)*len(C0s))
            data[i][j][k] = As[k+j*len(h0s)+i*len(h0s)*len(C0s)]

print(time.time()-t0)
my_interpolating_function = RegularGridInterpolator((T0s,C0s,h0s),data)
# #spline = sp.interpolate.Rbf(T0s,C0s,h0s,z,function='thin_plate')
print(time.time()-t0)

from joblib import dump, load

dump(my_interpolating_function, 'VanDerWaals_Model_GlassPS80WaterPS.joblib') 
my_interpolating_function= load('VanDerWaals_Model_GlassPS80WaterPS.joblib')         


idT = 1
idC = 0
Tcelsius = T0s[idT]
print(T0s[idT])
c0s = C0s[idC]
print(C0s[idC])
dataShow = [data[idT][idC][k] for k in range(len(h0s))]

h0s = np.logspace(-10,-5.5,100)


fig, ax = plt.subplots()
ax.plot(h0s*1e9,dataShow,'+',color='dodgerblue', markersize=4, label='Calculated, C = 10^{-5} M')


h0sF = np.logspace(-10,-5.5,1000)
#Ts = np.linspace(0,100,11)
#Cs = np.logspace(-5,3,11)
yRbf = []

for i in range(len(h0sF)):
    #print(my_interpolating_function([[Tcelsius,c0s,log(h0s[i])/log(10)]]))
    yRbf.append(my_interpolating_function([[Tcelsius,c0s,log(h0sF[i])/log(10)]]))
    #print(time.time()-t0)
#h0sF = [log(h0sF[i])/log(10) for i in range(len(h0sF))]

ax.plot(h0sF*1e9,yRbf,color='lightskyblue', label='Fit, C = 10^{-5} M')            


idT = 1
idC = 10
Tcelsius = T0s[idT]
print(T0s[idT])
c0s = C0s[idC]
print(C0s[idC])
dataShow2 = [data[idT][idC][k] for k in range(len(h0s))]
ax.plot(h0s*1e9,dataShow2,'+',color='darkviolet', markersize=4, label='Calculated, C = 0.16 M')


h0sF = np.logspace(-10,-5.5,1000)
#Ts = np.linspace(0,100,11)
#Cs = np.logspace(-5,3,11)
yRbf2 = []

for i in range(len(h0sF)):
    #print(my_interpolating_function([[Tcelsius,c0s,log(h0s[i])/log(10)]]))
    yRbf2.append(my_interpolating_function([[Tcelsius,c0s,log(h0sF[i])/log(10)]]))
    #print(time.time()-t0)
#h0sF = [log(h0sF[i])/log(10) for i in range(len(h0sF))]

ax.plot(h0sF*1e9,yRbf2,color='plum', label='Fit, C = 0.16 M') 


idT = 1
idC = 19
Tcelsius = T0s[idT]
print(T0s[idT])
c0s = C0s[idC]
print(C0s[idC])
dataShow2 = [data[idT][idC][k] for k in range(len(h0s))]
ax.plot(h0s*1e9,dataShow2,'+',color='crimson', markersize=4, label='Calculated, C = 10^{3} M')


h0sF = np.logspace(-10,-5.5,1000)
#Ts = np.linspace(0,100,11)
#Cs = np.logspace(-5,3,11)
yRbf2 = []

for i in range(len(h0sF)):
    #print(my_interpolating_function([[Tcelsius,c0s,log(h0s[i])/log(10)]]))
    yRbf2.append(my_interpolating_function([[Tcelsius,c0s,log(h0sF[i])/log(10)]]))
    #print(time.time()-t0)
#h0sF = [log(h0sF[i])/log(10) for i in range(len(h0sF))]

ax.plot(h0sF*1e9,yRbf2,color='lightpink',label='Fit, C = 10^{3} M') 



ax.set_xlabel('Height $h$ (log scale) (nm)')
ax.set_ylabel('Hamaker Constant $A_{GPWP}(h)/k_B T$')
plt.xscale('log')

ax.legend(loc='upper right',
          fontsize=10,
          frameon=False)

ax.set_ylim(0,3.5)
plt.tight_layout()
plt.savefig('VanDerWaals_GP80WP_HeightPlots.pdf')
plt.show()
#
            
...

