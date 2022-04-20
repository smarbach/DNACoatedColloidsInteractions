#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:48:52 2020

@author: sophie marbach
"""
import math
import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata, RegularGridInterpolator, Rbf
from math import log , exp, sqrt
pi = math.pi
from joblib import dump, load
from CommonToolsModule import DerjaguinIntegral

def hamakerFactorPlate(h, Radius):
    return(1/6*(2*Radius/h*(h+Radius)/(h+2*Radius) - log((h+2*Radius)/h)))  #L1.42 eq. of parsegian book

def hamakerFactorColloid(h,Radius):
    z2 = (2*Radius + h)**2
    return(1/3*(Radius**2/(z2 - 4*Radius**2) + Radius**2/z2 - (1/2)*log(1 - 4*Radius**2/z2))) #L1.41 eq. of parsegian book
    
def VanDerWaalsPotential(T,c0,allhs,Radius,srcpath,optColloid):

    # in this function T has to be in CELSIUS ! 
    # and c0 has to be in mol/L
    # and h in m
    myHamakerFunction = load(srcpath+'VanDerWaals_Model_GlassWaterPS.joblib')

    cMod = log(c0)/log(10)
    # vdW potential between plate and sphere
    phiVdW = 0*allhs
    for ip in range(len(allhs)):
        h = allhs[ip]
        if log(h) < - 5.5*log(10):
            if log(h) > -10*log(10):
                hMod =log(h)/log(10)
                if T < 20:
                    T = 20 #if temperature is out of bounds fix it at min/max values
                if T > 100:
                    T = 100 
                    
                hamakerConstant = myHamakerFunction([[T,cMod,hMod]])
                if optColloid:    
                    phiVdW[ip] = - hamakerConstant*hamakerFactorColloid(h, Radius)
                else:
                    phiVdW[ip] = - hamakerConstant*hamakerFactorPlate(h, Radius)
                
    # vdW potential between sphere and sphere
    #z2 = (2*Radius + h)**2
    #phiVdW = - hamakerConstant/3*(Radius**2/(z2 - 4*Radius**2) + Radius**2/z2 - (1/2)*log(1 - 4*Radius**2/z2))
    
    return(phiVdW)


def VanDerWaalsPotentialDer(T,c0,allhs,Radius,srcpath,optColloid):

    # in this function T has to be in CELSIUS ! 
    # and c0 has to be in mol/L
    # and h in m
    myHamakerFunction = load(srcpath+'VanDerWaals_Model_GlassWaterPS.joblib')

    cMod = log(c0)/log(10)
    # vdW potential between plate and sphere
    phiVdW = 0*allhs
    for ip in range(len(allhs)):
        h = allhs[ip]
        if log(h) < - 5.5*log(10):
            if log(h) > -10*log(10):
                hMod =log(h)/log(10)
                if T < 20:
                    T = 20 #if temperature is out of bounds fix it at min/max values
                if T > 100:
                    T = 100
                hamakerConstant = myHamakerFunction([[T,cMod,hMod]])
                if optColloid:    
                    phiVdW[ip] = - hamakerConstant/(12*pi*h**2)
                else:
                    phiVdW[ip] = - hamakerConstant/(12*pi*h**2)
                
    # vdW potential between sphere and sphere
    #z2 = (2*Radius + h)**2
    #phiVdW = - hamakerConstant/3*(Radius**2/(z2 - 4*Radius**2) + Radius**2/z2 - (1/2)*log(1 - 4*Radius**2/z2))
    
    return(phiVdW)
    
def VanDerWaalsPotentialGlassOnPS(T,c0,allhs,Radius,srcpath,optColloid):

    # in this function T has to be in CELSIUS ! 
    # and c0 has to be in mol/L
    # and h in m
    myHamakerFunction = load(srcpath+'VanDerWaals_Model_GlassPS80WaterPS.joblib')

    cMod = log(c0)/log(10)
    # vdW potential between plate and sphere
    phiVdW = 0*allhs
    for ip in range(len(allhs)):
        h = allhs[ip]
        if log(h) < - 5.5*log(10):
            if log(h) > -10*log(10):
                hMod =log(h)/log(10)
                if T < 20:
                    T = 20 #if temperature is out of bounds fix it at min/max values
                if T > 100:
                    T = 100 
                
                hamakerConstant = myHamakerFunction([[T,cMod,hMod]])
                if optColloid:    
                    phiVdW[ip] = - hamakerConstant/(12*pi*h**2)
                else:
                    phiVdW[ip] = - hamakerConstant/(12*pi*h**2)
                
    # vdW potential between sphere and sphere
    #z2 = (2*Radius + h)**2
    #phiVdW = - hamakerConstant/3*(Radius**2/(z2 - 4*Radius**2) + Radius**2/z2 - (1/2)*log(1 - 4*Radius**2/z2))
    
    return(phiVdW)    

def VanDerWaalsPotentialGlassOnPSDer(T,c0,allhs,Radius,srcpath,optColloid):

    # in this function T has to be in CELSIUS ! 
    # and c0 has to be in mol/L
    # and h in m
    myHamakerFunction = load(srcpath+'VanDerWaals_Model_GlassPS80WaterPS.joblib')

    cMod = log(c0)/log(10)
    # vdW potential between plate and sphere
    phiVdW = 0*allhs
    for ip in range(len(allhs)):
        h = allhs[ip]
        if log(h) < - 5.5*log(10):
            if log(h) > -10*log(10):
                hMod =log(h)/log(10)
                if T < 20:
                    T = 20 #if temperature is out of bounds fix it at min/max values
                if T > 100:
                    T = 100 
                
                hamakerConstant = myHamakerFunction([[T,cMod,hMod]])
                if optColloid:    
                    phiVdW[ip] = - hamakerConstant/(12*pi*h**2)
                else:
                    phiVdW[ip] = - hamakerConstant/(12*pi*h**2)
                
    # vdW potential between sphere and sphere
    #z2 = (2*Radius + h)**2
    #phiVdW = - hamakerConstant/3*(Radius**2/(z2 - 4*Radius**2) + Radius**2/z2 - (1/2)*log(1 - 4*Radius**2/z2))
    
    return(phiVdW)    
    
def VanDerWaalsPotentialPS(T,c0,allhs,Radius,srcpath,optColloid):

    # in this function T has to be in CELSIUS ! 
    # and c0 has to be in mol/L
    # and h in m
    myHamakerFunction = load(srcpath+'VanDerWaals_Model_PSWaterPS.joblib')

    cMod = log(c0)/log(10)
    # vdW potential between plate and sphere
    phiVdW = 0*allhs
    for ip in range(len(allhs)):
        h = allhs[ip]
        if log(h) < - 5.5*log(10):
            if log(h) > -10*log(10):
                hMod =log(h)/log(10)
                hamakerConstant = myHamakerFunction([[T,cMod,hMod]])
                if optColloid:    
                    phiVdW[ip] = - hamakerConstant*hamakerFactorColloid(h, Radius)
                else:
                    phiVdW[ip] = - hamakerConstant*hamakerFactorPlate(h, Radius)
                
    # vdW potential between sphere and sphere
    #z2 = (2*Radius + h)**2
    #phiVdW = - hamakerConstant/3*(Radius**2/(z2 - 4*Radius**2) + Radius**2/z2 - (1/2)*log(1 - 4*Radius**2/z2))
    
    return(phiVdW)   


def VanDerWaalsPotentialPSDer(T,c0,allhs,Radius,srcpath,optColloid):

    # in this function T has to be in CELSIUS ! 
    # and c0 has to be in mol/L
    # and h in m
    myHamakerFunction = load(srcpath+'VanDerWaals_Model_PSWaterPS.joblib')

    cMod = log(c0)/log(10)
    # vdW potential between plate and sphere
    phiVdW = 0*allhs
    for ip in range(len(allhs)):
        h = allhs[ip]
        if log(h) < - 5.5*log(10):
            if log(h) > -10*log(10):
                hMod =log(h)/log(10)
                hamakerConstant = myHamakerFunction([[T,cMod,hMod]])
                if optColloid:    
                    phiVdW[ip] = - hamakerConstant/(12*pi*h**2)
                else:
                    phiVdW[ip] = - hamakerConstant/(12*pi*h**2)
                
    # vdW potential between sphere and sphere
    #z2 = (2*Radius + h)**2
    #phiVdW = - hamakerConstant/3*(Radius**2/(z2 - 4*Radius**2) + Radius**2/z2 - (1/2)*log(1 - 4*Radius**2/z2))
    
    return(phiVdW)     

def VanDerWaalsPotentialHamaDer(T,c0,allhs,Radius,srcpath,optColloid,hamakerConstant):

    # in this function T has to be in CELSIUS ! 
    # and c0 has to be in mol/L
    # and h in m
    # vdW potential between plate and sphere
    phiVdW = 0*allhs
    for ip in range(len(allhs)):
        h = allhs[ip]
        if optColloid:    
            phiVdW[ip] = - hamakerConstant/(12*pi*h**2)
        else:
            phiVdW[ip] = - hamakerConstant/(12*pi*h**2)
        

    return(phiVdW)     

def VanDerWaalsPotentialHama(T,c0,allhs,Radius,srcpath,optColloid,hamakerConstant):

    # in this function T has to be in CELSIUS ! 
    # and c0 has to be in mol/L
    # and h in m
    
    # vdW potential between plate and sphere
    phiVdW = 0*allhs
    for ip in range(len(allhs)):
        h = allhs[ip]
        if optColloid:    
            phiVdW[ip] = - hamakerConstant*hamakerFactorColloid(h, Radius)
        else:
            phiVdW[ip] = - hamakerConstant*hamakerFactorPlate(h, Radius)
                
    # vdW potential between sphere and sphere
    #z2 = (2*Radius + h)**2
    #phiVdW = - hamakerConstant/3*(Radius**2/(z2 - 4*Radius**2) + Radius**2/z2 - (1/2)*log(1 - 4*Radius**2/z2))
    
    return(phiVdW)   
        
def VanDerWaalsPotentialFull(T,c0,allhs,Radius,slideType,srcpath,optColloid,  *args, **kwargs): 
    
    
    hamaker = kwargs.get('hamaker',None)
    #just a plate plate interaction
    if slideType == 'Glass': 
        phiVdW = VanDerWaalsPotential(T,c0,allhs,Radius,srcpath,optColloid)
    elif slideType == 'PSonGlass':
        phiVdW = VanDerWaalsPotentialGlassOnPS(T,c0,allhs,Radius,srcpath,optColloid)          
    elif slideType == 'PS':
        phiVdW = VanDerWaalsPotentialPS(T,c0,allhs,Radius,srcpath,optColloid)        
    else:
        hamaker = kwargs.get('hamaker',None)
        phiVdW = VanDerWaalsPotentialHama(T,c0,allhs,Radius,srcpath,optColloid,hamaker)        

    return(phiVdW)


def VanDerWaalsPotentialDejarguin(T,c0,allhs,Radius,slideType,srcpath,optColloid,  *args, **kwargs): 
    
    
    
    if slideType == 'Glass': 
        phiVdWh = VanDerWaalsPotentialDer(T,c0,allhs,Radius,srcpath,optColloid)
        phiVdW = DerjaguinIntegral(allhs[-1],Radius,allhs,phiVdWh,optColloid)            
    elif slideType == 'PSonGlass':
        phiVdWh = VanDerWaalsPotentialGlassOnPSDer(T,c0,allhs,Radius,srcpath,optColloid)     
        phiVdW = DerjaguinIntegral(allhs[-1],Radius,allhs,phiVdWh,optColloid)            
    elif slideType == 'PS':
        phiVdWh = VanDerWaalsPotentialPSDer(T,c0,allhs,Radius,srcpath,optColloid) 
        phiVdW = DerjaguinIntegral(allhs[-1],Radius,allhs,phiVdWh,optColloid)            
    else:
        hamaker = kwargs.get('hamaker',None)
        phiVdWh = VanDerWaalsPotentialHamaDer(T,c0,allhs,Radius,srcpath,optColloid,hamaker) 
        phiVdW = DerjaguinIntegral(allhs[-1],Radius,allhs,phiVdWh,optColloid)      

    return(phiVdW)