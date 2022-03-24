#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:10:45 2021

@author: sm8857
"""
import numpy as np
from scipy.optimize import fsolve

def MorsePotential(r,De,re,a):
    
    V = De*(np.exp(-2*a*(r-re)) - 2*np.exp(-a*(r-re)))
    
    return(V)

def FrenkelPotential(ra,rc,sigma,epsilon):
    
    alpha = 2*(rc/sigma)**2*(3/2/((rc/sigma)**2-1))**3
    rmin = rc*(3/(1+2*(rc/sigma)**2))**(1/2)
    
    V  = []
    
    for r in ra:
        if r > rc:
            V.append(0)
        else:
            V.append(epsilon*alpha*((sigma/r)**2 -1)*((rc/r)**2-1)**2)
    
    
    return(V)


def LennardJonesPotential(r,rc,sigma,epsilon):
    
    V  = 4*epsilon*((sigma/(r-rc))**12 - (sigma/(r-rc))**6)
        
    
    return(V)





def GeneralLennardJonesPotential(r,rc,m,sigma,epsilon):
    
    V  = 4*epsilon/m*(m*(sigma/(r-rc))**(2*m) - m*(sigma/(r-rc))**m)
        
    
    return(V)


def WCApotential(ra,sigma,epsilon):
    
    V  = []
    
    for r in ra:
        if r > sigma*2**(1/6):
            V.append(0)
        else:
            V.append(LennardJonesPotential(r,sigma,epsilon) + epsilon)
            
        
    return(V)


def LennardJonesPotentialTrunk(ra,rc,sigma,epsilon):
    
    
    rmax = sigma*2.5
    V = []
    for r in ra:
        if r-rc > rmax:
            V.append(0)
        else:
            V.append( LennardJonesPotential(r,rc,sigma,epsilon) - LennardJonesPotential(rc+rmax,rc,sigma,epsilon) ) 
            
    
    return(V)

def LennardJonesPotentialTrunkEimposed(ra,rc,sigma,epsilonMin):
    
    rmax = sigma*2.5
    #print(sigma)
    #print(rc)
    
    def ELJ(eps):
        return(eps + LennardJonesPotential(rc+rmax,rc,sigma,eps) - epsilonMin)
    
    #print(ELJ(epsilonMin))
    
    epsilonV = fsolve(ELJ, epsilonMin)
    epsilon = epsilonV[0]
    #print(epsilon)
    #epsilon = epsilonMin
    
    V = []
    for r in ra:
        if r-rc > rmax:
            V.append(0)
        else:
            V.append( LennardJonesPotential(r,rc,sigma,epsilon) - LennardJonesPotential(rc+rmax,rc,sigma,epsilon) ) 
            
    
    return(V)


def SplineLennardJonesPotentialTrunkEimposed(ra,rm,sigma,epsilonMin):
    
    rs = (26/7)**(1/6)*sigma
    epsilon = epsilonMin #the min is the same because the Spline is not shifted
    rc = 67/48*rs
    a = -24192/3211*(epsilon/rs**2)
    b = -(387072/61009)*(epsilon/rs**3)

    
    V = []
    for r in ra:
        if r-rm > rc:
            V.append(0)
        elif r-rm > rs: 
            V.append( a*(r-rm - rc)**2 + b*(r-rm - rc)**3  )
        else:
            V.append( 4*epsilon*((sigma/(r-rm))**12 - (sigma/(r-rm))**6) ) 
            
    
    return(V)

def MorsePotentialTrunkEimposed(ra,re,a,epsilonMin):
    
    rmax = 2.5/a
    #print(sigma)
    #print(rc)
    
    def ELJ(eps):
        return(eps + MorsePotential(re+rmax,eps,re,a) - epsilonMin)
    
    #print(ELJ(epsilonMin))
    
    epsilonV = fsolve(ELJ, epsilonMin)
    epsilon = epsilonV[0]
    #print(epsilon)
    #epsilon = epsilonMin
    
    V = []
    for r in ra:
        if r-re > rmax:
            V.append(0)
        else:
            V.append( MorsePotential(r,epsilon,re,a) - MorsePotential(re+rmax,epsilon,re,a) ) 
            
    
    return(V)

def GeneralLennardJonesPotentialTrunk(ra,rc,m, sigma,epsilon):
    
    
    rmax = sigma*2.5
    V = []
    for r in ra:
        if r-rc > rmax:
            V.append(0)
        else:
            V.append( GeneralLennardJonesPotential(r,rc,sigma,epsilon) - GeneralLennardJonesPotential(rc+rmax,rc,sigma,epsilon) ) 
            
    
    return(V)

def GeneralLennardJonesPotentialTrunkEimposed(ra,rc,m, sigma,epsilonMin):
    
    rmax = sigma*2.5
    def ELJ(eps):
        return(eps + GeneralLennardJonesPotential(rc+rmax,rc,m,sigma,eps) - epsilonMin)
    
    epsilonV = fsolve(ELJ, epsilonMin)
    epsilon = epsilonV[0]
    
    V = []
    for r in ra:
        if r-rc > rmax:
            V.append(0)
        else:
            V.append( GeneralLennardJonesPotential(r,rc,m,sigma,epsilon) - GeneralLennardJonesPotential(rc+rmax,rc,m,sigma,epsilon) ) 
            
    
    return(V)