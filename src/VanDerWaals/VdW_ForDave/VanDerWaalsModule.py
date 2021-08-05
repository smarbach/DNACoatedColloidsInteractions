#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:48:52 2020

@author: sophie marbach

This is the module to compute and plot van der waals potentials
"""
import math
import numpy as np
from math import log 
pi = math.pi
from joblib import load
import matplotlib.pyplot as plt

# Van der Waals Potential
def VanDerWaalsPotential(T,c0,h,R):

    # in this function
    # T has to be in CELSIUS ! 
    # and c0 in mol/L
    # and h in m
    # and R in m
    
    myHamakerFunction = load('VanDerWaals_Model_GlassWaterPS.joblib')
    hMod =log(h)/log(10)
    cMod = log(c0)/log(10)
    hamakerConstant = myHamakerFunction([[T,cMod,hMod]])
    phiVdW = - hamakerConstant/6*(2*R/h*(h+R)/(h+2*R) - log((h+2*R)/h))

    return(phiVdW)
        
# 1 - PHYSICAL PARAMETERS
radius = 3*10**(-6) # radius of particles in m
temperature = 25 # temperature in celsius
csalt = 0.14 # salt concentration in mol/L
allheights = np.linspace(1e-10,200e-9,200) #all heights to evaluate the profile on

# 2 - CALCULATE THE POTENTIAL

phiVdW = 0*allheights
for ip in range(len(allheights)):
    height = allheights[ip]
    # van der waals potential
    if log(height) < - 5.5*log(10): #outside this interval of heights we haven't calculated data on the Hamaker constant
        if log(height) > -10*log(10):
            phiVdW[ip] = VanDerWaalsPotential(temperature,csalt,height,radius) 
        else:
            phiVdW[ip] = 0
    else:
        phiVdW[ip] = 0
        
# 3 - PLOT

fig, ax = plt.subplots()
ax.plot(allheights*1e9,phiVdW,color='dodgerblue',label='Van der Waals Potential')

ax.set_ylim(-10,0) 

#potential2,lambdaV = compute_potential_profile_depletion(allheights, radius, PSdensity, gravity, saltConcentration, T, PSCharge, GlassCharge, 0)
#ax.plot(allheights*1e9,potential2)

ax.set_xlabel('Height $h_0$ (nm)')
ax.set_ylabel('Van der Waals Potential $\Phi_{vdW}(h_0)/k_B T$')

ax.legend(loc='lower center',
          fontsize=10,
          frameon=False)
#
#
plt.tight_layout()
plt.savefig('VanDerWaalsPotential.pdf')

plt.show()
