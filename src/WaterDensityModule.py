#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:48:52 2020

@author: sophie marbach
"""

import numpy as np
from scipy import interpolate
# might not need all the below codes I think
# import scipy.spatial as spatial
# import scipy.sparse.linalg as spla
# from scipy.linalg import cholesky
# from scipy.linalg import solve_triangular
# from functools import partial
# import copy
# import inspect
# import time
import sys
# import scipy.sparse as sp
import Pair_Lubrication as PL

#import matplotlib.gridspec as gridspec
#import matplotlib.pyplot as plt
#import matplotlib
#from matplotlib import ticker
#from matplotlib import cm
#from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                               AutoMinorLocator)

# ANOTHER SOLVENT
#waterDensity = [[24,1.00596],[30,1.00298],[35,0.99911],[40,0.99583],[45,0.99154],[50,0.98699],[55,0.98122],[60,0.97606]] ####
#waterDensity = [[24,1.00596],[90,1.00596]] ####

waterDensity = [[22,1.00531], \
[30,	1.00306], \
[40,	0.9994], \
[45,	0.99736], \
[50,	0.99511], \
[55,	0.9927], \
[60,	0.99013], \
[65,	0.98676], \
[70,	0.98345]]

def density(T) :
    Tbase =  273.15
    Tc = []
    densities = []
    for i in range(len(waterDensity)):
        Tc.append(waterDensity[i][0] + Tbase)
        densities.append(waterDensity[i][1])
    
    densities = [densities[i]*10**-3/(1*10**-6) for i in range(len(densities))]
    tck = interpolate.splrep(Tc, densities, s=None,k=1)
    return(interpolate.splev(T, tck, der=0))

def viscosityWater(T):
    A = 2.414e-5
    B = 247.8 
    C = 140
    return(A*10**(B/(T - C)))

def ONeillFrictionTangential(delta):

    deltas = [0.003202,0.005004,0.0453,0.1276,0.5431,1.3524,2.7622,9.0677]
    Fx = [4.0223,3.7863,2.6475,2.1514,1.5675,1.3079,1.1738,1.0591] # data from Brenner/Cox/Goldman full expansion
    
    if delta > 9.0677:
        friction = 1.0
    elif delta > 0.01:
        tck = interpolate.splrep(deltas, Fx, s=None,k=1)
        friction = interpolate.splev(delta, tck, der=0)
    else:        
        friction =  -8/15*np.log(delta) + 0.9588 #2.65a fromula in Brenner/Cox/Goldman
        
    return(friction)


def BrennanSphereSphereTangential(delta,srcpath):
    Lub = PL.Pair_Lubrication()
    Lub.load_WS_coefficient_interp_data(srcpath)
    Lub.load_JO_coefficient_interp_data(srcpath)
    Lub.set_WS_coefficient_interp_functions()
    Lub.set_JO_coefficient_interp_functions()
    
    #print("all set")
    r_i = np.array([0.0,0.0,0.0]) # pos of 1
    r_j = np.array([0.0,0.0,2+delta]) #pos of 2
    eta = 1.0 #viscosity -- set them to 1 so you don't get weird things
    a = 1.0# radius -- set them to 1 so you don't get weird things -- and then multiply by R and by eta
    Res = Lub.Resist(r_i, r_j, eta, a)
    #print("resistance: ")
    #print(Res)
    #Mob = np.linalg.pinv(Res)
    #print("Mobility: ")
    #print(Mob)
    return(Res[1][1]/(6*np.pi))
    




# Uncomment the following if you want a nice plot of the density fit
#
#allTs = [i for i in np.linspace(20,80,100)][:];
#allDensities = np.zeros(len(allTs))
#
#for i in range(len(allTs)):
#    allDensities[i] = density(allTs[i]+Tbase)*10**-3
#
#result = {'labels': [d for d in allDensities], 'data': allTs}
#
##print(result['data'])
##print(result['labels'])
#
#matplotlib.rcParams.update(
#    {'font.sans-serif': 'Arial',
#     'font.size': 8,
#     'font.family': 'Arial',
#     'mathtext.default': 'regular',
#     'axes.linewidth': 0.35, 
#     'axes.labelsize': 8,
#     'xtick.labelsize': 7,
#     'ytick.labelsize': 7,     
#     'lines.linewidth': 0.35,
#     'legend.frameon': False,
#     'legend.fontsize': 7,
#     'xtick.major.width': 0.3,
#     'xtick.minor.width': 0.3,
#     'ytick.major.width': 0.3,
#     'ytick.minor.width': 0.3,
#     'xtick.major.size': 1.5,
#     'ytick.major.size': 1.5,
#     'xtick.minor.size': 1,
#     'ytick.minor.size': 1,
#    })
#
#Tc = []
#densities = []
#for i in range(len(waterDensity)):
#    Tc.append(waterDensity[i][0] )
#    densities.append(waterDensity[i][1])
#    
#fig6, ax6 = plt.subplots(figsize = [3,2])
#ax6.plot(result['data'],result['labels'],':',color='black',lw = 1, label = 'Interpolation')
#
#ax6.plot(Tc,densities,'.', color='crimson', lw = 2, ms = 8, label = 'Experimental Data')
#    
#ax6.set_xlabel('Temperature (\N{DEGREE SIGN}C)')
#ax6.set_ylabel('Water density (g/cm$^3$)')
#
#
## ax6.spines['right'].set_visible(False)
## ax6.spines['top'].set_visible(False)
## # Only show ticks on the left and bottom spines
## ax6.yaxis.set_ticks_position('left')
## ax6.xaxis.set_ticks_position('bottom')
#
#ax6.xaxis.set_major_locator(MultipleLocator(20))
#ax6.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#ax6.xaxis.set_minor_locator(MultipleLocator(5))
#
#ax6.yaxis.set_major_locator(MultipleLocator(0.005))
#ax6.yaxis.set_minor_locator(MultipleLocator(0.01))
#
#ax6.legend(loc='lower left',handlelength=1, fontsize = 6,markerscale = 0.8,labelspacing = 0.5) 
#plt.tight_layout()
#plt.savefig('densities.eps', format='eps')
#plt.savefig('densities.pdf', format='pdf')
#
#plt.savefig('densities.svg', format='svg', transparent = True)
#plt.show()
