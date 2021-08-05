#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:03:57 2019

@author: sophie Marbach

This function creates the model to calculate the Hamaker constants more quickly. 

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


## 1 - LOAD THE DATA GENERATED AND FIT WITH A CLASSIC INTERPOLATION TO GENERATE A "MODEL"

parameterSweep = pd.read_csv('VanDerWaals_parameterSweep_GlassWaterPS.csv', delimiter = ',')

Ts = parameterSweep.Temperature
Cs = parameterSweep.Concentration
Hs = parameterSweep.Distance
As = parameterSweep.Hamaker


t0 = time.time()

# Do a bit of data transformation, actually doing the fits in log scale for C and h makes more sense... 
h0s = np.logspace(-10,-5.5,100)
h0s = [log(h0s[i])/log(10) for i in range(len(h0s))]
T0s = np.linspace(20,100,20)
C0s = np.logspace(-5,3,20)
C0s = [log(C0s[i])/log(10) for i in range(len(C0s))]
data = np.zeros((len(T0s),len(C0s),len(h0s)))
for i in range(len(T0s)):
    for j in range(len(C0s)):
        for k in range(len(h0s)):
            data[i][j][k] = As[k+j*len(h0s)+i*len(h0s)*len(C0s)]

print(time.time()-t0)
my_interpolating_function = RegularGridInterpolator((T0s,C0s,h0s),data)
print(time.time()-t0)

from joblib import dump, load

dump(my_interpolating_function, 'VanDerWaals_Model_GlassWaterPS.joblib') 

## 2 - LOAD THE "MODEL" GENERATED - and compare it with the calculated points

my_interpolating_function= load('VanDerWaals_Model_GlassWaterPS.joblib')


idT = 1
idC = 0

Tcelsius = T0s[idT]
print("Working temperature is")
print(T0s[idT])
c0s = C0s[idC]
print("Working salt concentration is")
print(C0s[idC])
dataShow = [data[idT][idC][k] for k in range(len(h0s))]

h0s = np.logspace(-10,-5.5,100)

fig, ax = plt.subplots()
ax.plot(h0s*1e9,dataShow,'o',color='dodgerblue', markersize=4, label='Calculated, C = 10^{-5} M')

h0sF = np.logspace(-10,-5.5,1000)

hamakerFit = []

for i in range(len(h0sF)):
    hamakerFit.append(my_interpolating_function([[Tcelsius,c0s,log(h0sF[i])/log(10)]]))
    

ax.plot(h0sF*1e9,hamakerFit,color='lightskyblue', label='Fit, C = 10^{-5} M')            


ax.set_xlabel('Height $h$ (log scale) (nm)')
ax.set_ylabel('Hamaker Constant $A_{GWP}(h)/k_B T$')
ax.set_ylim(0,3.5)
plt.xscale('log')

ax.legend(loc='upper right',
          fontsize=10,
          frameon=False)

plt.tight_layout()
plt.savefig('VanDerWaals_GWP_HeightPlot.pdf')
plt.show()
#
            
...

