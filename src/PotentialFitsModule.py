#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:10:45 2021

@author: sm8857
"""
import numpy as np


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
