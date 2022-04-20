#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:48:52 2020

@author: sophie marbach
"""
import math
import numpy as np
from scipy import interpolate
from math import log , exp, sqrt
pi = math.pi


# Constants
Na = 6.02*10**23
Rg = 8.314 #J/(mol.K); #perfect gas constant
Tbase = 273.15
kB = Rg/Na; #boltzmann constant



# temperetures and concentrations in weight per volume
# Data from Alexandridis Macromolecules, 1994
cmcT = [[20,4],[25,0.7],[30,0.1],[35,0.025],[40,0.008]] ####
F127M = 12600 #g/mol for F127 (Sigma Aldrich data)

def cmc(T) :
    
    Tc = []
    cmconcentrations = []
    for i in range(len(cmcT)):
        Tc.append(cmcT[i][0] + Tbase)
        cmconcentrations.append(cmcT[i][1])
    
    cmconcentrations = [log(cmconcentrations[i]*10**-2*10**3/(F127M)) for i in range(len(cmconcentrations))]
    tck = interpolate.splrep(Tc, cmconcentrations, s=None,k=1)
    logc = interpolate.splev(T, tck, der=0)
    return(exp(logc))



# Uncomment the following if you want a nice plot of the cmc fit

#allTs = [i for i in np.linspace(0,100,100)][:];
#allCs = np.zeros(len(allTs))
#
#for i in range(len(allTs)):
#    allCs[i] = cmc(allTs[i]+Tbase)
#
#result = {'labels': [d for d in allCs], 'data': allTs}
#print(result['data'])
#print(result['labels'])



# temperetures and number of particules and spherical radius in Angstroms
# Data from Wanka Macromolecules, 1994 for F127 Micelles
NaggRdata = [[20,7,17],[25,37,57],[30,67,69.4],[35,82,74],[40,97,78.5],[45,106,80.8]] ####

def NaggR(T) :
    
    Tc = []
    Naggdat = []
    Rdat = []
    for i in range(len(NaggRdata)):
        Tc.append(NaggRdata[i][0] + Tbase)
        Naggdat.append(NaggRdata[i][1])
        Rdat.append(NaggRdata[i][2])
    
    tckr = interpolate.splrep(Tc, Rdat, s=None,k=1)
    R = interpolate.splev(T, tckr, der=0)
    tckn = interpolate.splrep(Tc, Naggdat, s=None,k=1)
    Nagg = interpolate.splev(T, tckn, der=0)
    
    return(R,Nagg)


# Uncomment the following if you want a nice plot of the aggregation and radius of micelles data

#allTs = [i for i in np.linspace(0,100,1000)][:];
#allRs = np.zeros(len(allTs))
#allNs = np.zeros(len(allTs))
#
#
#for i in range(len(allTs)):
#    allRs[i],allNs[i] = NaggR(allTs[i]+Tbase)
#
#result = {'radii': [d for d in allRs], 'aggregate': [n for n in allNs], 'temperature': allTs}
#print(result['radii'])
#print(result['aggregate'])
#print(result['temperature'])


def DepletionPotential(T,cm0,allhs,Radius,optColloid, *args, **kwargs):
    
    aggRadius = kwargs.get('aggRadius',0)
    depletionType = kwargs.get('depletionType','default')
    # size and number in the aggregate at that temperature
    if depletionType == 'F127':
        # recorded data for F127
        Ragg,Nagg = NaggR(T)
        Ragg = Ragg*10**(-10) # convert the radius from angstroms back to meters
        #print(Ragg)
        
        # concentration of aggregates
        cagg = (cm0 - cmc(T))*Na/Nagg*10**3 #this is still in number/L so that's why we use 10**3 factor here
        Ragg = Ragg*2/sqrt(pi) #This is to account for the fact that F127 is a polymer and hence the relevant radius 
        # is not necessarily the end to end radius
    else:
        Ragg = aggRadius #Whatever is inputted here would have to take this into account
        cagg = cm0
        #cagg = 25.5*10**(18) # data for bechinger    
        #Ragg = 501*10**(-9) # data for crockers
        #Ragg = 101*10**(-9) # data for Bechinger
        
    #print(cagg) 
    
    vFraction = cagg*4/3*pi*Ragg**3
    pfactor = (1+vFraction + vFraction**2 - vFraction**3)/(1-vFraction)**3 #Carnahagan-Stirling equation of state
    #cagg = 280/(31.5*10**6)*Na
    #print(cm0)
    #print(cmc(T))
    phi = 0*allhs
    #print(Ragg)
    #print(allhs)
    if cagg > 0:
        for ih in range(len(allhs)):
            h = allhs[ih]
            if h < 2*Ragg:
                if optColloid:
                    phi[ih] = -cagg*pfactor*pi/6*((2*Ragg-h)**2*(3*Radius + 2*Ragg + h/2))
                else:
                    phi[ih] = -cagg*pfactor*pi*(4*Radius*(Ragg**2) + (4/3)*(Ragg**3) + 1/3*(h**3) + (Radius - Ragg)*(h**2) - 4*Radius*Ragg*h)

    #print(Ragg)
    return(phi)
        
    