#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:34:25 2020

@author: sophie marbach
"""

import numpy as np
from scipy.stats import norm, poisson
import math
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import trapz
from math import sqrt, exp, pi

from scipy.integrate import quad

def ShotnoiseDistortedPotential (DV,beta,allheights,potentialIn,T):
    
    #DV = imposed number of photons
    
    
    
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
    phimin = np.min(potential)

    PhiNormers = [np.exp(-potential[h0i]-phimin)*(allheights[h0i+1]- allheights[h0i]) for h0i in range(len(allheights)-1)]
    PhiNorm = sum(PhiNormers)
    
    print(PhiNorm)
    phiFunc = InterpolatedUnivariateSpline(allheights,potential,k=1)
    
    #1st step is to create the interpolated values of noise
    Pconvolve = []
    N0 = DV
    #for dh in allheights:
    #    Pconvolve.append(poisson.pmf(N0,N0*exp(-beta*(dh)))*N0)
    #pnoiseFunc = InterpolatedUnivariateSpline(allheights,Pconvolve,k=1)

    def pNoiseFunc(dh):
        if -beta*dh < 800:
            #if (N0*exp(-beta*(dh))) < 5*N0:
            return(poisson.pmf(N0,N0*exp(-beta*(dh)))*N0)
            #else:
            #   return(0)
        else:
            return(0)

    for h in allheights:
        
        #hEval = [h - 1/beta*np.log(DV/n) for n in Nvalues]
        #phiEval = [phiFunc(hval) for hval in hEval]
        

        
        # Factors = [np.exp(-potential[h0i]-phimin)*pNoiseFunc(h-allheights[h0i])/PhiNorm* \
        #            (allheights[h0i+1]- allheights[h0i-1])/2 for h0i in range(1,len(allheights)-1)]
            
        # ProbaDistorted.append(sum(Factors))
        
        Factors = [np.exp(-potential[h0i]-phimin)*pNoiseFunc(h-allheights[h0i])/PhiNorm for h0i in range(len(allheights))]
        # pIntFactors = InterpolatedUnivariateSpline(allheights,Factors,k=1)
        # ProbaDistortedh,err = quad(pIntFactors,allheights[0],allheights[-1])
        ProbaDistortedh = trapz(Factors, allheights)
        ProbaDistorted.append(ProbaDistortedh)
    

    PhiDistorted = - np.log(ProbaDistorted) + np.log(kBT)   
    
    
    
    ########### old formulas to compute noise
    # potential = 0*potentialIn
    # for ip in range(len(potentialIn)):
    #     p = potentialIn[ip]
    #     if math.isinf(p):
    #         potential[ip] = 0
    #     else:
    #         potential[ip] = p
    # maxp = max(potential)
    # for ip in range(len(potentialIn)):
    #     p = potentialIn[ip]
    #     if math.isinf(p):
    #         potential[ip] = maxp
    #     else:
    #         potential[ip] = p
    
    
    # Na = 6.022*10**23;
    # Rg = 8.314; #J/(mol.K); #perfect gas constant
    # kB = Rg/Na;
    # Nvalues = list(range(1, 2*DV))
    # kBT = kB*T
    # ProbaDistorted = []
    # """
    # Original probability distribution of h with added shot noise (analytics)
    # """
    # PhiNormers = [np.exp(-phi) for phi in potential]
    # PhiNorm = sum(PhiNormers)
    
    # print(PhiNorm)
    # phiFunc = InterpolatedUnivariateSpline(allheights,potential,k=1)
    
    # for h in allheights:
    #     #hEval = [h - 1/beta*np.log(DV/n) for n in Nvalues]
    #     #phiEval = [phiFunc(hval) for hval in hEval]
    #     Factors = [poisson.pmf(n,DV)*np.exp(-phiFunc(h - 1/beta*np.log(DV/n)))/PhiNorm for n in Nvalues]
    #     ProbaDistorted.append(sum(Factors))
    

    # PhiDistorted = - np.log(ProbaDistorted) + np.log(kBT)   
        
    return PhiDistorted


def arbitraryNoise (sigma,allheights,potentialIn,T):
    
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
    
    
    ProbaDistorted = []
    """
    Original probability distribution of h with added gaussian noise (analytics)
    """
    PhiNormers = [np.exp(-phi) for phi in potential]
    PhiNorm = sum(PhiNormers)
    Na = 6.022*10**23;
    Rg = 8.314; #J/(mol.K); #perfect gas constant
    kB = Rg/Na;
    kBT = kB*T
    
    print(PhiNorm)
    phiFunc = InterpolatedUnivariateSpline(allheights,potential,k=1)
    hnV = np.linspace(-10*sigma,10*sigma,500)
    
    
    nweights = [exp(-(hn)**2/(2*sigma**2)) for hn in hnV]
    normw = sum(nweights)
    
    for h in allheights:
        
        #hEval = [h - 1/beta*np.log(DV/n) for n in Nvalues]
        #phiEval = [phiFunc(hval) for hval in hEval]
        
        Factors = [1/normw*exp(-(hn)**2/(2*sigma**2))*np.exp(-phiFunc(h - hn))/PhiNorm for hn in hnV]
        ProbaDistorted.append(sum(Factors))
    

    PhiDistorted = - np.log(ProbaDistorted) + np.log(kBT)  
    
    # for h in allheights:
        
    #     #hEval = [h - 1/beta*np.log(DV/n) for n in Nvalues]
    #     #phiEval = [phiFunc(hval) for hval in hEval]
        
    #     Factors = [1/normw*exp(-(hn)**2/(2*sigma**2))*phiFunc(h-hn) for hn in hnV]
    #     ProbaDistorted.append(sum(Factors))
    

    # PhiDistorted = ProbaDistorted
        
    return PhiDistorted