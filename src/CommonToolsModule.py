#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:40:55 2020

@author: sm8857
"""

import math
import numpy as np
from math import inf
from scipy.integrate import trapz
import matplotlib.pyplot as plt


from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
pi = math.pi


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)



def CompareData(x1,y1,x2,y2,xmax):

    fitPotentialCalc = interpolate.splrep(x2, y2, s=None,k=1)

    indexH = next(x for x, val in enumerate(x1) if val < xmax)
    y21 = interpolate.splev(x1, fitPotentialCalc, der=0)
    
    # calculate the l2 norms
    error = sum((y1[i]-y21[i])**2*abs(x1[i+1] - x1[i-1]) for i in range(indexH,len(y1)-1))
    errorNorm = sum(y1[i]**2*abs(x1[i+1] - x1[i-1]) for i in range(indexH,len(y1)-1))
    errorTot = error/errorNorm
    
    return(errorTot)
    
def CenterData(x1,y1,scalex):

    minx = x1[y1.argmin()]
    miny = min(y1)

    x1o = [(x1[i] - minx)*scalex for i in range(len(x1))]
    y1o = [y1[i] - miny for i in range(len(y1))]
    
    return(x1o,y1o)
    
def CenterDataNoYmove(x1,y1,scalex):

    minx = x1[y1.argmin()]

    x1o = [(x1[i] - minx)*scalex for i in range(len(x1))]
    
    return(x1o)    
    

def CleverIntegralInf(hsCut,allhs,Energies):
    # returns F(h) = int_h^(hsCut) Energies dh
    
    ftotfit = InterpolatedUnivariateSpline(allhs,Energies, k=1)
    currentInt = 0
    thresholdReached = 0
    cleverIntegral = np.zeros(len(allhs))
    for ip in range(len(allhs)):
        idh = len(allhs) - ip - 1
        if hsCut > allhs[idh] and allhs[idh] >= 0:
            if thresholdReached:
                y = ftotfit.integral(allhs[idh],allhs[idh+1])
            else:   
                y = ftotfit.integral(allhs[idh],hsCut)
                thresholdReached = 1
            currentInt += y
            cleverIntegral[idh] = currentInt
            
        elif allhs[idh] >= 0:
            cleverIntegral[idh] = 0
        elif allhs[idh] < 0:
            cleverIntegral[idh] = inf
          
    return(cleverIntegral)


def BruteIntegral(hsCut,allhs,Energies):
    # returns F(h) = int_h^(hsCut) Energies dh
    
    ftotfit = InterpolatedUnivariateSpline(allhs,Energies, k=1)

    cleverIntegral = np.zeros(len(allhs))
    for ip in range(len(allhs)):
        if hsCut > allhs[ip] and allhs[ip] >= 0:
            cleverIntegral[ip] = ftotfit.integral(allhs[ip],hsCut)

    return(cleverIntegral)

def CleverIntegral0(allhs,Energies):
    # returns F(h) = int_0^h Energies dh
    
    ftotfit = InterpolatedUnivariateSpline(allhs,Energies, k=1)
    currentInt = 0
    thresholdReached = 0
    nh = len(allhs)
    cleverIntegral = np.zeros(nh)
    for ip in range(len(allhs)):
        if allhs[ip] >= 0:
            if thresholdReached:
                y = ftotfit.integral(allhs[ip-1],allhs[ip])
            else:   
                if allhs[ip] == 0:
                    y = 0
                else: 
                    print('This integration may not work')
                    y = ftotfit.integral(0,allhs[ip])
                thresholdReached = 1
            currentInt += y
            cleverIntegral[ip] = currentInt
        
        else:
            cleverIntegral[ip] = inf
                
    return(cleverIntegral)
    
def DerjaguinIntegral(hsCut,Radius,allhs,Energies,optcolloidcolloidFlag):
    # this derjaguin integral takes in a surface energy and outputs an energy
    y = CleverIntegralInf(hsCut,allhs,Energies)
    #y = BruteIntegral(hsCut,allhs,Energies)
    if optcolloidcolloidFlag:
        prefac = 2*pi*Radius/2
    else:
        prefac = 2*pi*Radius
    return(prefac*y)