#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:34:25 2020

@author: sophie marbach
"""

import numpy as np
from scipy.stats import poisson
import math
from scipy.interpolate import InterpolatedUnivariateSpline

def ShotnoiseDistortedPotential (DV,beta,allheights,potentialIn,T):
    
    potential = 0*potentialIn
    for ip in range(len(potentialIn)):
        p = potentialIn[ip]
        if math.isinf(p):
            potential[ip] = 0
        else:
            potential[ip] = p
    maxp = max(potential)
    for ip in range(len(potentialIn)):
        p = potentialIn[ip]
        if math.isinf(p):
            potential[ip] = maxp
        else:
            potential[ip] = p
    
    
    Na = 6.022*10**23;
    Rg = 8.314; #J/(mol.K); #perfect gas constant
    kB = Rg/Na;
    Nvalues = list(range(1, 2*DV))
    kBT = kB*T
    ProbaDistorted = []
    """
    Original probability distribution of h with added shot noise (analytics)
    """
    PhiNormers = [np.exp(-phi) for phi in potential]
    PhiNorm = sum(PhiNormers)
    
    print(PhiNorm)
    phiFunc = InterpolatedUnivariateSpline(allheights,potential,k=1)
    
    for h in allheights:
        #hEval = [h - 1/beta*np.log(DV/n) for n in Nvalues]
        #phiEval = [phiFunc(hval) for hval in hEval]
        Factors = [poisson.pmf(n,DV)*np.exp(-phiFunc(h - 1/beta*np.log(DV/n)))/PhiNorm for n in Nvalues]
        ProbaDistorted.append(sum(Factors))
    

    PhiDistorted = - np.log(ProbaDistorted) + np.log(kBT)   
        
    return PhiDistorted