#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:48:52 2020

@author: sophie marbach
"""
import math
import numpy as np
from CommonToolsModule import DerjaguinIntegral
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline
from math import log , exp, sqrt, inf, erf
pi = math.pi
import matplotlib.pyplot as plt


# Constants
Na = 6.02*10**23
Rg = 8.314 #J/(mol.K); #perfect gas constant
Tbase = 273.15
kB = Rg/Na; #boltzmann constant

def hMilner(sigma,ell,N,evFac):
    # equilibrium brush height according to Milner
    # evFac = reduced excluded volume factor (non dimensional)
    excludedVolume =  evFac*ell**3 #this is a simplifying assumption. 
    hMilner = (12/pi**2*sigma*ell**2*excludedVolume)**(1/3)*N
    return(hMilner)

def rMushroom(ell,N,nature):
    
     # equilibrium brush radius according to Polymer Physics, Rubinstein
    if nature == 'PEO':
        CinfinityPEO = 6.7
        lPEO = 0.148*1e-9
        radiusEndtoEnd = sqrt(CinfinityPEO*lPEO**2*(3*N))
    elif nature == 'DNA':
        radiusEndtoEnd = sqrt(ell**2*N)
    
    radiusGiration = radiusEndtoEnd/sqrt(6) #I'm not sure if it's the gyration or end-to-end that matters most.. 
    return(radiusEndtoEnd)
    
def elleff(ell1,N1,evFac1,ell2,N2,evFac2):
    # effective persistence length of a mixed brush
    # # keep the R length constant for the effective ell
    # elle = sqrt((N1*ell1**2 + N2*ell2**2)/(N1+N2)) 
    # # and then calculate the effective excluded volume
    # elleff53 = (evFac1*ell1**(5))**(1/3)*N1 + (evFac2*ell2**(5))**(1/3)*N2
    # eve = 1/elle**5*( elleff53/(N1+N2) )**3

    elleff53 = (evFac1*ell1**(5))**(1/3)*N1 + (evFac2*ell2**(5))**(1/3)*N2 #this is a simplifying assumption. 
    elle = (elleff53/(N1+N2))**(3/5)
    eve = 1.0
    
    
    # that's one possible model... that conserves the total height at least. 
    return(elle,eve)

def elleffMushroom(R1,R2,Nb):
    elle = sqrt(R1**2+R2**2)/sqrt(Nb)
    # effective persistence length of a mixed brush
    return(elle)


def findStretchedEll(sigma,ell,N,evFac,slideType):
    h = hMilner(sigma,ell,N,evFac)
    if N < 200:
        hadd = 4e-9
    elif N < 400:
        hadd = 4e-9
    else:
        hadd = 4e-9
        
    
    hstretch = h+hadd

   # model = 'increaseDensity'
    #model = 'increaseDensityProportionaly'
  #  model = 'constantIncrease' #model 1 -- also works for model 1b
    model = 'constantHighIncrease' #model 2
 #   model = 'constantVeryHighIncrease' #model 3 - doesn't work well
 #   model = 'increaseDensityProportionaly' #model 4
#    model = 'constantHighIncreaseWithBottom' #model 6
#    model = 'noIncrease' #model 5
    if model == 'increaseDensity':
    
        elle = ell
        #presence of F127 stretched by about 10 nm the polymers
        def searchElleff(sigmaeff):
            hsearch = hMilner(sigmaeff,ell,N,evFac)
            return(hstretch-hsearch)
        
        if N < 200:
            #print(searchElleff(sigma))
            #print(searchElleff(20*sigma))
            #print(searchElleff(100*sigma))
    
            sigmae = brentq(searchElleff,sigma,100*sigma)
        else:
            sigmae = brentq(searchElleff,sigma,10*sigma)
    
    elif model == 'increaseEll':
        sigmae = sigma
        #presence of F127 stretched by about 10 nm the polymers
        def searchElleff2(elleff):
            hsearch = hMilner(sigma,elleff,N,evFac)
            return(hstretch-hsearch)
        
        elle = brentq(searchElleff2,ell*0.1,ell*4)
    
    elif model == 'constantIncrease':#works well with 11.7 density on bottom layer
                                
        sigmae = sigma+ 1/(11e-9)**2 #increase the density by a constant amount
        elle = ell
        
    elif model == 'constantHighIncrease':
        if slideType == 'Glass': #works well with 9 density on bottom layer for *0.35 or with 8.5 for *0.45
            sigmae = sigma #+ 1/(11e-9)**2 #increase the density by a constant amount
            elle = ell
        else:
            sigmae = sigma+ 1/(4e-9)**2 #increase the density by a constant amount
            elle = ell
    
    elif model == 'constantHighIncreaseWithBottom':
        if slideType == 'Glass': #works well with 2 on bottom for *0.45 (5) *0.30 (5.5) -- 1.5 on bottom and 0.30
            sigmae = sigma*1.5 #increase the density by a constant amount 
            elle = ell
        else:
            sigmae = sigma+ 1/(4e-9)**2 #increase the density by a constant amount
            elle = ell
    
    elif model == 'constantVeryHighIncrease':
        if slideType == 'Glass': #works well with ?? density on bottom layer
            sigmae = sigma #+ 1/(11e-9)**2 #increase the density by a constant amount
            elle = ell
        else:
            sigmae = sigma+ 1/(3e-9)**2 #increase the density by a constant amount
            elle = ell
    
    elif model == 'increaseDensityProportionaly':
    
        if slideType == 'Glass': #works (2/1.5) well with 12 density on bottom layer
            #                    # with 1.7/1.7 works well with 11 density on bottom layer
                                 # with 1/1.7  works well with 12 density on bottom layer. 
            
            sigmae = sigma*1#s.70
            elle = ell
        else:
            sigmae = sigma*1.70 #increase the density by 45%
            elle = ell
    
    elif model == 'increaseDensityProportionalyNotBottom':
        if slideType == 'Glass': #works well with 9.5 density on bottom layer
            #sigmae = sigma*2.70 #increase the density by 170% = +4nm of F127
            hadd = 4e-9
            hstretch = h+hadd
            
            def searchElleff(sigmaeff):
                hsearch = hMilner(sigmaeff,ell,N,evFac)
                return(hstretch-hsearch)
            sigmae = brentq(searchElleff,sigma,100*sigma)
            elle = ell
        else:
            sigmae = sigma*1.50 #increase the density by 45%
            elle = ell
    
    elif model == 'noIncrease':
        if slideType == 'Glass': #works well with 9.5 density on bottom layer
            #sigmae = sigma*2.70 #increase the density by 170% = +4nm of F127
            hadd = 4e-9
            hstretch = h+hadd
            
            def searchElleff(sigmaeff):
                hsearch = hMilner(sigmaeff,ell,N,evFac)
                return(hstretch-hsearch)
            sigmae = brentq(searchElleff,sigma,100*sigma)
            # sigmae = sigma 
            elle = ell
        else:
            sigmae = sigma #the increase will be contained in the wPEO
            elle = ell

    
    return(sigmae,elle)


def findStretchedEllMushroom(ellbtemp,Nb):
    Rb = rMushroom(ellbtemp,Nb,'DNA')
    Rbfind = Rb+4e-9
    elle = Rbfind/sqrt(Nb)
    return(elle)


def eMilnerPrefactor(sigma,ell,N,evFac):
    excludedVolume = evFac*ell**3 #this is a simplifying assumption for the excludedVolume
    prefactor = (pi**2/12)**(1/3)*(sigma*excludedVolume/ell)**(2/3)*N
    return(prefactor)


def aMilner(sigma,ell,N,h,evFac):
    excludedVolume = evFac*ell**3; #this is a simplifying assumption for the excludedVolume
    Afac = N/ell**2*(N*excludedVolume*ell**2/(h)*(sigma) + pi**2*h**2/(24*N**2));
    return(Afac)

def eMilner(h,sigma,ell,N,evFac):
    # based on the Milner-Cates theory
    heq = hMilner(sigma,ell,N,evFac)
    #print(heq)
    x = h/heq#[h[ip]/heq for ip in range(len(h))]
    #print(x)
    prefactor  = eMilnerPrefactor(sigma,ell,N,evFac)*sigma #you need the surface energy
    
    if x<1:
        phi = prefactor*(1./(2*x) + 1/2*x**2 - 1/10*x**5  - 9/10);
    else:
        phi = inf
    # phi = np.zeros(len(h))

    # for ip in range(len(h)):
    #     if x[ip]>1:
    #         phi[ip] = 0
    #     elif x[ip]<0:
    #         phi[ip] = inf
    #     else:
    #         phi[ip] = -prefactor*(1./(2*x[ip]) + 1/2*x[ip]**2 - 1/10*x[ip]**5  - 9/10);
    #print(phi)
    return(phi)


def eMushroom(h,R,sigma):
    # based on the Eq. (15) of supmat
    x = h/R#[h[ip]/heq for ip in range(len(h))]
    #print(x)
    prefactor = sigma #you need the surface energy
    
    # this is Dolan's model, it is quite sharp...
    if x<1.248:
        phi = -prefactor*np.log( (sqrt(2*pi/(3*x**2)))*2*(np.exp(-pi**2/(6*x**2)) +np.exp(-2**2*pi**2/(6*x**2)) ) )
    else:
        phi = -prefactor*np.log(1-2*np.exp(-1/2*3*x**2)+2*np.exp(-2**2/2*3*x**2)-2*np.exp(-3**2/2*3*x**2))
    
    # this is Dolan's model, but the order 0 approximation
    # if x<1.248:
    #     phi = prefactor*(np.log( (sqrt((3*x**2)/(2*pi)))/2) + pi**2/(6*x**2))
    # else:
    #     phi = -prefactor*np.log(1-2*np.exp(-1/2*3*x**2)) #+2*np.exp(-2**2/2*3*x**2)-2*np.exp(-3**2/2*3*x**2))
    
    
    #this is Daan's model
    #phi = -prefactor*np.log(erf(sqrt(3*x**2/2)))
    
    return(phi)



def eMilnerBridge(h,sigma,ell,N,heq,evFac):
    # based on the Milner-Cates theory
    #heq = hMilner(sigma,ell,N)
    #print(heq)
    x = h/heq#[h[ip]/heq for ip in range(len(h))]
    #print(x)
    prefactor  = eMilnerPrefactor(sigma,ell,N,evFac) #you need an energy
    
    if x<1:
        phi = prefactor*(+ 1/10*x**5);
    else:
        phi = inf
    # phi = np.zeros(len(h))

    # for ip in range(len(h)):
    #     if x[ip]>1:
    #         phi[ip] = 0
    #     elif x[ip]<0:
    #         phi[ip] = inf
    #     else:
    #         phi[ip] = -prefactor*(1./(2*x[ip]) + 1/2*x[ip]**2 - 1/10*x[ip]**5  - 9/10);
    #print(phi)
    return(phi)


def redistributeHeights(hs,h1,h2):
    h1out = [h1/(h1+h2)*hs[ip] for ip in range(len(hs))]
    h2out = [h2/(h1+h2)*hs[ip] for ip in range(len(hs))]
    return(h1out,h2out)


def StericPotentialMilner(h,Radius,sigma,ell,N,evFac,optcolloid):
    # based on the Milner-Cates theory
    heq = hMilner(sigma,ell,N,evFac)
    #print(heq)
    x = [h[ip]/heq for ip in range(len(h))]
    #print(x)
    excludedVolume = evFac*ell**3
    if optcolloid:
        prefactor  = 2*pi*Radius*sigma**2*excludedVolume*N**2/2
    else:
        prefactor  = 2*pi*Radius*sigma**2*excludedVolume*N**2
    
    phi = np.zeros(len(h))

    for ip in range(len(h)):
        
        if x[ip]>1:
            phi[ip] = 0
        elif x[ip]<0:
            phi[ip] = inf
        else:
            phi[ip] = -prefactor*(1/2*log(x[ip]) + 1/6*x[ip]**3 - 1/60*x[ip]**6 - 9/10*x[ip] + 3/4)
    #print(phi)
    return(phi)

def StericPotentialMushroom(h,Radius,sigma,ell,N,evFac,optcolloid):
    # based on the mushroom model ... 
    # for now this is not well coded in ! have to use expressions for steric repulsion indeed !
    heq = hMilner(sigma,ell,N,evFac)
    #print(heq)
    x = [h[ip]/heq for ip in range(len(h))]
    #print(x)
    excludedVolume = evFac*ell**3
    if optcolloid:
        prefactor  = 2*pi*Radius*sigma**2*excludedVolume*N**2/2
    else:
        prefactor  = 2*pi*Radius*sigma**2*excludedVolume*N**2
    
    phi = np.zeros(len(h))

    for ip in range(len(h)):
        
        if x[ip]>1:
            phi[ip] = 0
        elif x[ip]<0:
            phi[ip] = inf
        else:
            phi[ip] = -prefactor*(1/2*log(x[ip]) + 1/6*x[ip]**3 - 1/60*x[ip]**6 - 9/10*x[ip] + 3/4)
    #print(phi)
    return(phi)

def StericPotentialMilner2faces(h,Radius,sigma,ell,N,evFac):
    # based on the Milner-Cates theory
    heq = hMilner(sigma,ell,N,evFac)
    #print(heq)
    x = [h[ip]/heq for ip in range(len(h))]
    #print(x)
    excludedVolume = evFac*ell**3
    prefactor  = 2*pi*Radius*sigma**2*excludedVolume*N**2
    
    phi = np.zeros(len(h))

    for ip in range(len(h)):
        
        if x[ip]>1:
            phi[ip] = 0
        elif x[ip]<0:
            phi[ip] = inf
        else:
            phi[ip] = -2*prefactor*(1/2*log(x[ip]) + 1/6*x[ip]**3 - 1/60*x[ip]**6 - 9/10*x[ip] + 3/4)*2
    #print(phi)
    return(phi)

def facingPotentialMilner(Nb,ellb,sigmab,evFacb,Nt,ellt,sigmat,evFact,heights):
    
    prefactorb  = eMilnerPrefactor(sigmab,ellb,Nb,evFacb)
    prefactort  = eMilnerPrefactor(sigmat,ellt,Nt,evFact)
    
    disc = 100 #discretization factor
    
    sigma1 = np.linspace(0,sigmab,disc)
    sigma2 = np.linspace(0,sigmat,disc)
    es = np.zeros((disc,disc))
    h1s = np.zeros(disc)
    h2s = np.zeros(disc)    
    
    N1 = Nb
    N2 = Nt
    ell1 = ellb
    ell2 = ellt 
    w1 = evFacb*ell1**3
    w2 = evFact*ell2**3
    eV1 = evFacb
    eV2 = evFact
    
    h1max = hMilner(sigmab,ell1,N1,evFacb)
    h2max = hMilner(sigmat,ell2,N2,evFact)
    
    potential = np.zeros(len(heights))
    hmean1 = np.zeros(len(heights))
    hmean2 = np.zeros(len(heights))
   
    firsttime = 1
    for ih in range(len(heights)):
        id = len(heights)-1-ih
        height = heights[id]
        # look at decreasing heights
        if height > h1max+h2max:
            if firsttime:  
                for is1 in range(disc):
                    s1 = sigma1[is1]
                    h1 = hMilner(s1,ell1,N1,eV1)
                    h1s[is1] = h1
                    h1start = h1    
                    for is2 in range(disc):
                        s2 = sigma2[is2]
                        h2 = hMilner(s2,ell2,N2,eV2)
                        h2s[is2] = h2
                        if is1 > 0 and is2 > 0: 
                            es[is1][is2] = sigmab*aMilner(s1,ell1,N1,h1,eV1) + sigmat*aMilner(s2,ell2,N2,h2,eV2) #brushes assume their equilibrium height
                        elif is1 > 0:
                            es[is1][is2] = sigmab*aMilner(s1,ell1,N1,h1,eV1)
                        elif is2 > 0:
                            es[is1][is2] = sigmat*aMilner(s2,ell2,N2,h2,eV2)
                            #then do trapezoidal integral in 2D
                potentialBase = 1/(sigmab*sigmat)*trapz(trapz(es,sigma1,0),sigma2) - 9/10*(1)*(sigmab*prefactorb + sigmat*prefactort)
                firsttime = 0   
            potential[id] = potentialBase
            hmean1[id] = h1max
            hmean2[id] = h2max
        else:
            for is1 in range(disc):
                s1 = sigma1[is1]
                h1 = h1s[is1]
                h1start = h1    
                for is2 in range(disc):
                    s2 = sigma2[is2]
                    h2 = h2s[is2]
                    if is1 > 0 and is2 > 0: 
                        if height <= (h1+h2):
                           def dEdh(h):
                               return( sigmab*(N1*((h*pi**2)/(12*N1**2) - (ell1**2*N1*s1*w1)/h**2))/ell1**2 + \
                                      sigmat*(N2*(-(((height - h)*pi**2)/(12*N2**2)) + (ell2**2*N2*s2*w2)/(height - h)**2))/ell2**2)
                           #h1real = fsolve(dEdh,h1start)
                           h1real = brentq(dEdh,0,height)
                           es[is1][is2] = (sigmab*aMilner(s1,ell1,N1,h1real,eV1) + sigmat*aMilner(s2,ell2,N2,height - h1real),eV2) 
                           h1start = h1real
                    elif is1 > 0:
                        if height <= (h1+h2):
                            es[is1][is2] = sigmab*aMilner(s1,ell1,N1,height,eV1)
                    elif is2 > 0:
                        if height <= (h1+h2):
                            es[is1][is2] = sigmat*aMilner(s2,ell2,N2,height,eV2)
                           # don't need to fill in the other grid parts since they do not change
            #then do trapezoidal integral in 2D for the potential
            potential[id] = 1/(sigmab*sigmat)*trapz(trapz(es,sigma1,0),sigma2) -9/10*(1)*(sigmab*prefactorb + sigmat*prefactort)
            #the effective brush heights correspond to the equilibrium heights at this density. 
            hmean1[id] = h1real
            hmean2[id] = height - h1real
        

    return(potential,hmean1,hmean2)


def facingPotentialMilnerSym(Nb,ellb,sigmab,evFacb,Nt,ellt,sigmat,evFact,heights):
    
    
    disc = 100 #discretization factor
    
    sigma1 = sigmab
    sigma2 = sigmat
    

    
    N1 = Nb
    N2 = Nt
    ell1 = ellb
    ell2 = ellt 
    w1 = evFacb*ell1**3
    w2 = evFact*ell2**3
    eV1 = evFacb
    eV2 = evFact
    
    h1 = hMilner(sigma1,ell1,N1,evFacb)
    h2 = hMilner(sigma2,ell2,N2,evFact)
    

    
    potential = np.zeros(len(heights))
    hmean1 = np.zeros(len(heights))
    hmean2 = np.zeros(len(heights))
   
    for ih in range(len(heights)):
        height = heights[ih]
        if height < h1+h2:
            
            def energyMin(h):
                return(eMilner(h, sigma1, ell1, N1, eV1) + eMilner(height-h, sigma2, ell2, N2, eV2))
            
            if sigma1 == sigma2 and N1 == N2 and ell1 == ell2:
                hmean1[ih] = height/2
                hmean2[ih] = height/2
                potential[ih] = energyMin(height/2)
            else:
                
                hvals = np.linspace(max(0,height-h2),min(h1,height),disc)
                Evals = [energyMin(hval) for hval in hvals]
                #print(Evals)
                lowestEnergyIndex = np.array(Evals).argmin()
                #print(lowestEnergyIndex)
                # if ih%100 == 0:
                #     plt.plot(hvals,Evals,'r')
                #     plt.show()
                #     plt.pause(1)
                    
                potential[ih] = Evals[lowestEnergyIndex]
                hmean1[ih] = hvals[lowestEnergyIndex]
                hmean2[ih] = height - hvals[lowestEnergyIndex]
        else:
            hmean1[ih] = h1
            hmean2[ih] = h2
    #print(potential)    

    return(potential,hmean1,hmean2)



def facingPotentialMushroom(Rb,sigmab,Rt,sigmat,heights):
    
    # this is in a regime where we assume that mushrooms can not really interpenetrate. 
    # this is because they are assumed to be dense enough -- so to speak
    
    disc = 100 #discretization factor
    
    potential = np.zeros(len(heights))
    hmean1 = np.zeros(len(heights))
    hmean2 = np.zeros(len(heights))
   
    sigma1 = sigmab
    sigma2 = sigmat
    
    R2 = Rt
    R1 = Rb
    
    for ih in range(len(heights)):
        height = heights[ih]
        
        # def energyMin(h):
        #     return(eMushroom(h, R1,sigma1) + eMushroom(height-h,R2,sigma2))
        
        # if Rb == Rt:
        #     hmean1[ih] = height/2
        #     hmean2[ih] = height/2
        #     potential[ih] = energyMin(height/2) 
        # else:
            
        #     hvals = np.linspace(max(0,height-R2),min(R1,height),disc)
        #     Evals = [energyMin(hval) for hval in hvals]
        #     #print(Evals)
        #     lowestEnergyIndex = np.array(Evals).argmin()
        #     #print(lowestEnergyIndex)
        #     potential[ih] = Evals[lowestEnergyIndex]
        #     hmean1[ih] = hvals[lowestEnergyIndex]
        #     hmean2[ih] = height - hvals[lowestEnergyIndex]

    # actually ignore the presence of one and the other
    #print(potential)    
        hmean1[ih] = height/2 #these values don't matter here since mushroom brushes don't interact
        hmean2[ih] = height/2
        potential[ih] = eMushroom(height, R1,sigma1) + eMushroom(height,R2,sigma2)
    
    return(potential,hmean1,hmean2)


# def facingPotentialMilnerBridge(Nb,ellb,sigmab,Nt,ellt,sigmat,heights):
#     # this facing potential also computes the free energy for two tethers to bridge
    
#     prefactorb  = eMilnerPrefactor(sigmab,ellb,Nb)
#     prefactort  = eMilnerPrefactor(sigmat,ellt,Nt)
    
#     disc = 100 #discretization factor
    
#     sigma1 = np.linspace(0,sigmab,disc)
#     sigma2 = np.linspace(0,sigmat,disc)
#     es = np.zeros((disc,disc))
#     esref = np.zeros((disc,disc))
#     esb = np.zeros((disc,disc))
#     h1s = np.zeros(disc)
#     h2s = np.zeros(disc)    
    
#     N1 = Nb
#     N2 = Nt
#     ell1 = ellb
#     ell2 = ellt 
#     w1 = ell1**3
#     w2 = ell2**3
    
#     h1max = hMilner(sigmab,ell1,N1)
#     h2max = hMilner(sigmat,ell2,N2)
    
#     potential = np.zeros(len(heights))
#     potentialRef = np.zeros(len(heights))
#     potentialB = np.zeros(len(heights))

#     hmean1 = np.zeros(len(heights))
#     hmean2 = np.zeros(len(heights))
    
 
#     firsttime = 1
#     for ih in range(len(heights)):
#         id = len(heights)-1-ih
#         height = heights[id]
#         # look at decreasing heights
#         if height > h1max+h2max: #brushes assume their equilibrium height
#             if firsttime:  
#                 for is1 in range(disc):
#                     s1 = sigma1[is1]
#                     h1 = hMilner(s1,ell1,N1)
#                     h1s[is1] = h1
#                     h1start = h1    
#                     for is2 in range(disc):
#                         s2 = sigma2[is2]
#                         h2 = hMilner(s2,ell2,N2)
#                         h2s[is2] = h2
#                         if is1 > 0 and is2 > 0: 
#                             es[is1][is2] = sigmab*aMilner(s1,ell1,N1,h1) + sigmat*aMilner(s2,ell2,N2,h2) 
#                             esref[is1][is2] = aMilner(s1,ell1,N1,h1) + aMilner(s2,ell2,N2,h2) #reference facing energy for a pair
#                         elif is1 > 0:
#                             es[is1][is2] = sigmab*aMilner(s1,ell1,N1,h1)
#                             esref[is1][is2] = aMilner(s1,ell1,N1,h1)
#                         elif is2 > 0:
#                             es[is1][is2] = sigmat*aMilner(s2,ell2,N2,h2)
#                             esref[is1][is2] = aMilner(s2,ell2,N2,h2)
#                             #then do trapezoidal integral in 2D
#                 potentialBase = 1/(sigmab*sigmat)*trapz(trapz(es,sigma1,0),sigma2) - 9/10*(1)*(sigmab*prefactorb + sigmat*prefactort)
#                 potentialBaseRef = 1/(sigmab*sigmat)*trapz(trapz(esref,sigma1,0),sigma2) - 9/10*(1)*(prefactorb + prefactort)
#                 firsttime = 0   
#             potential[id] = potentialBase
#             potentialRef[id] = potentialBaseRef
#             hmean1[id] = h1max
#             hmean2[id] = h2max
#         else:
#             for is1 in range(disc):
#                 s1 = sigma1[is1]
#                 h1 = h1s[is1]
#                 h1start = h1    
#                 for is2 in range(disc):
#                     s2 = sigma2[is2]
#                     h2 = h2s[is2]
#                     if is1 > 0 and is2 > 0: 
#                         if height <= (h1+h2):
#                            def dEdh(h):
#                                return( sigmab*(N1*((h*pi**2)/(12*N1**2) - (ell1**2*N1*s1*w1)/h**2))/ell1**2 + \
#                                       sigmat*(N2*(-(((height - h)*pi**2)/(12*N2**2)) + (ell2**2*N2*s2*w2)/(height - h)**2))/ell2**2)
#                            h1real = brentq(dEdh,0,height)
#                            es[is1][is2] = (sigmab*aMilner(s1,ell1,N1,h1real) + sigmat*aMilner(s2,ell2,N2,height - h1real)) 
#                            esref[is1][is2] = (aMilner(s1,ell1,N1,h1real) + aMilner(s2,ell2,N2,height - h1real)) 
#                            h1start = h1real
#                     elif is1 > 0:
#                         if height <= (h1+h2):
#                             es[is1][is2] = sigmab*aMilner(s1,ell1,N1,height)
#                             esref[is1][is2] = aMilner(s1,ell1,N1,height)
#                     elif is2 > 0:
#                         if height <= (h1+h2):
#                             es[is1][is2] = sigmat*aMilner(s2,ell2,N2,height)
#                             esref[is1][is2] = aMilner(s2,ell2,N2,height)
#                            # don't need to fill in the other grid parts since they do not change
#             #then do trapezoidal integral in 2D for the potential
#             potential[id] = 1/(sigmab*sigmat)*trapz(trapz(es,sigma1,0),sigma2) -9/10*(1)*(sigmab*prefactorb + sigmat*prefactort)
#             potentialRef[id] = 1/(sigmab*sigmat)*trapz(trapz(esref,sigma1,0),sigma2) -9/10*(1)*(prefactorb + prefactort)
#             #the effective brush heights correspond to the equilibrium heights at this density. 
#             hmean1[id] = h1real
#             hmean2[id] = height - h1real

#     # now calculate the bridging energy
    
#     #for ih in range(len(heights)):
#     #    id = len(heights)-1-ih
#     #    height = heights[id]
#     #    # look at decreasing heights
#     #    for is1 in range(disc):
#     #        s1 = sigma1[is1]
#     #        h1 = h1s[is1]
#     #        h1start = h1    
#     #        for is2 in range(disc):
#     #            s2 = sigma2[is2]
#     #            h2 = h2s[is2]
#     #            if is1 > 0 and is2 > 0: 
#     #                def dEdh(h):
#     #                    return( (N1*((h*pi**2)/(12*N1**2) - (ell1**2*N1*s1*w1)/h**2))/ell1**2 + \
#     #                           (N2*(-(((height - h)*pi**2)/(12*N2**2)) + (ell2**2*N2*s2*w2)/(height - h)**2))/ell2**2)
#     #                h1real = fsolve(dEdh,h1start)
#     #                esb[is1][is2] = (aMilner(s1,ell1,N1,h1real) + aMilner(s2,ell2,N2,height - h1real)) 
#     #                h1start = h1real
#     #            elif is1 > 0:
#     #                esb[is1][is2] = aMilner(s1,ell1,N1,height)
#     #            elif is2 > 0:
#     #                esb[is1][is2] = aMilner(s2,ell2,N2,height)
#     #    #then do trapezoidal integral in 2D for the potential
#     #    potentialB[id] = 1/(sigmab*sigmat)*trapz(trapz(esb,sigma1,0),sigma2) -9/10*(1)*(prefactorb + prefactort)
#         #the effective brush heights correspond to the equilibrium heights at this density. 
#         #hb1[id] = h1real
#         #hb2[id] = height - h1real 
#         #this will be useful to access the electrostatic energy... though in a first approximation let's just assume heights are not changed by bridging much 

    
#     #potentialBridgeOnly = potentialB - potentialRef 

#     potentialBridgeOnly = 0*potentialB
    
#     #plt.plot(esb[0][:])
#     #plt.plot(esb[10][:])
#     #plt.plot(esb[20][:])
#     #plt.plot(heights,potentialB)
#     #plt.plot(heights,potentialRef )
#     #plt.plot(heights,potentialBridgeOnly)
#     #plt.show()

#     return(potential,potentialBridgeOnly,hmean1,hmean2)

def determineRelevantHeights(N1b,N2b,N2t,N1t,ell1,ell2,eV1,eV2,sigmab,sigmat,h0,nresolution,slabHeight,model,mushroomFlag):

    if mushroomFlag == 1:
        #these brushes are mushroom like
        R1b = rMushroom(ell1,N1b,'PEO')
        R2b = rMushroom(ell2,N2b,'DNA')
            
        R1t = rMushroom(ell1,N1t,'PEO')
        print(R1t)
        R2t = rMushroom(ell2,N2t,'DNA')
    
        if R1b + R2b + R1t + R2t > 0:
           #now determine relevant height range to explore the potential on
            Rb = sqrt(R1b**2 + R2b**2)
            Rt = sqrt(R1t**2 + R2t**2)
            htot = Rb + Rt
            a = np.linspace(h0+(htot)*0.05,h0+(htot)*0.9,30*nresolution)
            b = np.linspace(h0+(htot)*0.91,150e-9,20*nresolution)
            c = np.linspace(151e-9,h0+(htot)+600e-9,10*nresolution)   
            d = np.linspace(h0+(htot)+700e-9,slabHeight,3) # THIS IS A PARAMETER FROM FAN'S SETUP
            allheights = np.concatenate((a,b,c,d),axis=None)
            
            print('The equilibrium total thickness of the bottom brush (mushroom model)  is (nm): ',(Rb)*1e9)
            print('The equilibrium total thickness of the top brush (mushroom model) is (nm): ',(Rt)*1e9)
            
        else:
            a = np.linspace(10e-9,400e-9,20*nresolution) #all heights to evaluate the profile on
            d = np.linspace(410e-9,slabHeight,3) # THIS IS A PARAMETER FROM FAN'S SETUP
            
            allheights = np.concatenate((a,d),axis=None)
    
    
    else:
        #these brushes are brushes
        #first determine heterogeneous brush parameters, if any
        h1b = hMilner(sigmab,ell1,N1b,eV1)
        h2b = hMilner(sigmab,ell2,N2b,eV2)
            
        h1t = hMilner(sigmat,ell1,N1t,eV1)
        h2t = hMilner(sigmat,ell2,N2t,eV2)

        if h1b+h1t+h2b+h2t > 0:
            if model == 'Unified':
                htot = h1b+h1t+h2b+h2t + 10e-9 #because there will be the increase in F127
                 #now determine relevant height range to explore the potential on
                a = np.linspace(h0+(htot)*0.3,h0+(htot)*0.55,10*nresolution)
                b = np.linspace(h0+(htot)*0.555,h0+(htot)*1.5,60*nresolution)
                c = np.linspace(h0+(htot)*1.505,h0+(htot)*3.0,20*nresolution)
                cb = np.linspace(h0+(htot)*3.005,h0+(htot)+999e-9,10*nresolution)   
                d = np.linspace(h0+(htot)+1000e-9,slabHeight,10) # THIS IS A PARAMETER FROM FAN'S SETUP
                allheights = np.concatenate((a,b,c,cb,d),axis=None)
                
                # allheights = np.linspace(h0 + htot*0.852, h0 + htot*1.05,60*nresolution)
                
                print('The equilibrium total thickness of the bottom brush is (nm): ',((h1b+h2b))*1e9)
                print('The equilibrium total thickness of the top brush is (nm): ', ((h1t+h2t))*1e9)
                #print(b)
            else:
                #now determine relevant height range to explore the potential on
                a = np.linspace(h0+(h1t+h2t+h1b+h2b)*0.3,h0+(h1t+h2t+h1b+h2b)*0.9,30*nresolution)
                b = np.linspace(h0+(h1t+h2t+h1b+h2b)*0.91,h0+(h1t+h2t+h1b+h2b)*3.0,20*nresolution)
                c = np.linspace(h0+(h1t+h2t+h1b+h2b)*3.01,h0+(h1t+h2t+h1b+h2b)+999e-9,10*nresolution)   
                d = np.linspace(h0+(h1t+h2t+h1b+h2b)+1000e-9,slabHeight,10) # THIS IS A PARAMETER FROM FAN'S SETUP
                allheights = np.concatenate((a,b,c,d),axis=None)
            
                print('The equilibrium total thickness of the bottom brush is (nm): ',((h1b+h2b))*1e9)
                print('The equilibrium total thickness of the top brush is (nm): ', ((h1t+h2t))*1e9)
                #print(b)
        else:
            a = np.linspace(10e-9,400e-9,20*nresolution) #all heights to evaluate the profile on
            d = np.linspace(410e-9,slabHeight,3) # THIS IS A PARAMETER FROM FAN'S SETUP
            
            allheights = np.concatenate((a,d),axis=None)
        
    #print(allheights)

    return(allheights)

    
def calculateCompressedHeights(N1b,N2b,N2t,N1t,ell1,ell2,eV1,eV2,sigmab,sigmat,h,bridgeOption,accountForF127Stretch,mushroomFlag,slideType) :
    
    lbot = 0
    ltop = 0
    
    
    if mushroomFlag == 1:
        #first determine heterogeneous brush parameters, if any
        R1b = rMushroom(ell1,N1b,'PEO')
        R2b = rMushroom(ell2,N2b,'DNA')
        Nb = N1b + N2b
        if Nb > 0:
            ellbtemp = elleffMushroom(R1b,R2b,Nb)
            if accountForF127Stretch == True:
                #if slideType == "Glass":
                #    ellb = ellbtemp
                #else:                
                # do it regardless of slide type
                ellb = findStretchedEllMushroom(ellbtemp,Nb)
                print('Effective persistence length for the bottom brush (nm)',ellb*1e9)
            else:
                ellb = ellbtemp
        else:
            ellb = ell1 #doesn't matter which one you pick cause there's nothing there
        Rb = rMushroom(ellb,Nb,'DNA')
        print('Effective radius for the bottom brush (nm)',Rb*1e9)
        lbot = Rb
        
        R1t = rMushroom(ell1,N1t,'PEO')
        R2t = rMushroom(ell2,N2t,'DNA')
        Nt = N1t + N2t
        if Nt > 0:
            ellttemp = elleffMushroom(R1t,R2t,Nt)
            if accountForF127Stretch == True:
                ellt = findStretchedEllMushroom(ellttemp,Nt)
                print('Effective persistence length for the bottom brush (nm)',ellt*1e9)
            else:
                ellt = ellttemp
        else:
            ellt = ell1 #doesn't matter which one you pick cause there's nothing there
        Rt = rMushroom(ellt,Nt,'DNA')
        print('Effective radius for the top brush (nm)',Rt*1e9)
        ltop = Rt
        
        #then determine compressed height
        if Nb == 0:
            ht = h
            hb = 0
            Emin = [eMushroom(h[i],Rt,sigmat) for i in range(len(h))]  #will be determined at a later stage so take 0 for now
            Ebridge = np.zeros(len(h)) #no bridging
        elif Nt == 0:
            ht = 0
            hb = h
            Emin = [eMushroom(h[i],Rb,sigmab) for i in range(len(h))] #will be determined at a later stage so take 0 for now
            Ebridge = np.zeros(len(h)) #no bridging
        else:
            Emin,hb,ht = facingPotentialMushroom(Rb,sigmab,Rt,sigmat,h) # Emin is an energy per unit area
            Ebridge = np.zeros(len(h))
        
        #redistribute those heights on appropriate quantities
        if Nb > 0:
            h1,h2 = redistributeHeights(hb,R1b,R2b)    
        else: 
            h1 = 0*h
            h2 = 0*h
        if Nt > 0:
            h4,h3 = redistributeHeights(ht,R1t,R2t)     
        else:
            h3 = 0*h
            h4 = 0*h
        hrest1 = Rb
        hrest2 = Rt
        #print(h4[-1]+h3[-1])
        sigmabtot = sigmab
        sigmattot = sigmat
        eVb = eV1
        eVt = eV1
        
        
    else:
        #first determine heterogeneous brush parameters, if any
        h1b = hMilner(sigmab,ell1,N1b,eV1)
        h2b = hMilner(sigmab,ell2,N2b,eV2)
        print('Bare heights of the bottom brush (nm)',h1b*1e9, h2b*1e9)
        Nb = N1b + N2b
        if Nb > 0:
            ellb,eVb = elleff(ell1,N1b,eV1,ell2,N2b,eV2)
            print('Effective excluded volume of the bottom brush ',eVb)
            if accountForF127Stretch == True:
                if slideType == "Glass":
                    sigmabtot = sigmab
                    sigmabtot,ellb = findStretchedEll(sigmab,ellb,Nb,eVb,"Glass") 
                else:
                    #find increased density based on stretched length of full brush
                    #sigmabtot = findStretchedEll(sigmab,ellb,Nb,slideType) 
                    #find ncreased density based on stretched of just the PEO brush
                    # if N1b > 0:
                    #     sigmabtot,ellb = findStretchedEll(sigmab,ell1,N1b,eV1,slideType) 
                    # else:
                        # actually I think this should just add to the whole brush... 
                    sigmabtot,ellb = findStretchedEll(sigmab,ellb,Nb,eVb,"") 
                print('Effective density of the bottom brush (nm^2)',1/sigmabtot*1e9*1e9)
            else:
                sigmabtot = sigmab
        else:
            ellb = ell1 #doesn't matter which one you pick cause there's nothing there
            eVb = eV1
            sigmabtot = sigmab
            
        h1t = hMilner(sigmat,ell1,N1t,eV1)
        h2t = hMilner(sigmat,ell2,N2t,eV2)
        print('Bare heights of the top brush (nm)',h1t*1e9, h2t*1e9)
        Nt = N1t + N2t
        if Nt > 0:
            ellt,eVt = elleff(ell1,N1t,eV1,ell2,N2t,eV2)            
            print('Effective excluded volume of the bottom brush ',eVt)
            print('Total height with effective parameters', hMilner(sigmat,ellt,Nt,eVt)*1e9)
            if accountForF127Stretch == True:
                #find increased density based on stretched length of full brush
                #sigmattot = findStretchedEll(sigmat,ellt,Nt,slideType)
                #find ncreased density based on stretched of just the PEO brush
                #if N1t > 0:
                #    sigmattot,ellt= findStretchedEll(sigmat,ell1,N1t,eV1,slideType)
                #else:
                # actually I think this should just add to the whole brush... 
                sigmattot,ellt = findStretchedEll(sigmat,ellt,Nt,eVt,"")
                print('Effective density of the top brush (nm^2)',1/sigmattot*1e9*1e9)
            else:
                sigmattot = sigmat
        else:
            ellt = ell1 #doesn't matter which one you pick cause there's nothing there
            eVt = eV1
            sigmattot = sigmat
        
        lbot = hMilner(sigmabtot,ellb,Nb,eVb)
        print("effective length of bottom brush",lbot*1e9)
        ltop = hMilner(sigmattot,ellt,Nt,eVt)
        print("effective length of top brush",ltop*1e9)
        
        #then determine compressed height
        if Nb == 0:
            ht = h
            hb = 0
            Emin = np.zeros(len(h)) #will be determined at a later stage so take 0 for now
            Ebridge = np.zeros(len(h)) #no bridging
        elif Nt == 0:
            ht = 0
            hb = h
            Emin = np.zeros(len(h)) #will be determined at a later stage so take 0 for now
            Ebridge = np.zeros(len(h)) #no bridging

        else:
            # all of those are pretty obsolete
            #if bridgeOption == 1:
            #    print("Calculating with bridging potential")
            #    Emin,Ebridge,hb,ht = facingPotentialMilnerBridge(Nb,ellb,sigmab,Nt,ellt,sigmat,h) # Emin is an energy per unit area, Ebridge an energy
            #else:
            #print("Calculating symmetric potential")
            
            Emin,hb,ht = facingPotentialMilnerSym(Nb,ellb,sigmabtot,eVb, Nt,ellt,sigmattot,eVt, h) # Emin is an energy per unit area
            Ebridge = np.zeros(len(h))
                
        #redistribute those heights on appropriate quantities
        if Nb > 0:
            h1,h2 = redistributeHeights(hb,h1b,h2b)    
        else: 
            h1 = 0*h
            h2 = 0*h
        if Nt > 0:
            h4,h3 = redistributeHeights(ht,h1t,h2t)     
        else:
            h3 = 0*h
            h4 = 0*h
        hrest1 = h1b+h2b+h1t+h2t
        hrest2 = h1b+h2b+h1t+h2t
        #print(h1,h2,h3,h4)
        
    return(Emin,Ebridge,h1,h2,h3,h4,ellb,ellt,eVb,eVt,Nb,Nt,hrest1,hrest2,sigmabtot,sigmattot,lbot,ltop)

  
def StericPotentialFull(allhs,R,sigmat,ellet,Nt,eVt, sigmab,elleb,Nb,eVb, hrest,Emins,optcolloidcolloidFlag,mushroomFlag,allQ1s,allQ2s):
    
    phiSter = 0*allhs
    
    if Nb == 0:
        if Nt == 0:
            phiSter = 0
        else:
            if mushroomFlag:
                phiSter = DerjaguinIntegral(hrest*100,R,allhs,Emins,optcolloidcolloidFlag)    
            else:
                phiSter = StericPotentialMilner(allhs,R,sigmat,ellet,Nt,eVt, optcolloidcolloidFlag)
    else:
        if Nt == 0:
            if mushroomFlag:
                phiSter = DerjaguinIntegral(hrest*100,R,allhs,Emins,optcolloidcolloidFlag)    
            else:
                phiSter = StericPotentialMilner(allhs,R,sigmab,elleb,Nb,eVb, optcolloidcolloidFlag)
        else:
            #print("Using dejarguin integral")
            #print(hrest)
            #print(allhs)
            # that's the case where you have to integrate the potential
            # I'M NOT SURE THIS IS THE RIGHT SIGMA FOR TOP VS BOTTOM... THERE'S A DIMENSIONS PROBLEM IF WE DON"T PUT IN SIGMA... 
            #phiSter = StericPotentialMilner2faces(allhs/2,R,sigmab,elleb,Nb)
            if mushroomFlag == 0: 
                phiSter = DerjaguinIntegral(hrest*100,R,allhs,Emins,optcolloidcolloidFlag) 
            else:
                EQ1s = [- sigmat*np.log(q1) for q1 in allQ1s]
                EQ2s = [- sigmab*np.log(q2) for q2 in allQ2s]
                EminsInt = np.zeros(len(Emins))
                
                
                for i in range(len(Emins)):
                    EminsInt[i] = max(EQ1s[i] + EQ2s[i],Emins[i])
                    EminsInt[i] = Emins[i] # use only Dolan's model
                #plt.loglog(allhs,EminsInt)
                #plt.loglog(allhs,Emins)
                #plt.show()
                #plt.pause(0.2)
                #print(EQ1s)
                #print(Emins)
                phiSter = DerjaguinIntegral(hrest*100,R,allhs,EminsInt,optcolloidcolloidFlag)          
    
    
    #print(Emins)         
    #print(phiSter)
    return phiSter