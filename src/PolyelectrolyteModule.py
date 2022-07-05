#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:12:46 2020

@author: sophie marbach
"""

import math
import numpy as np
from CommonToolsModule import DerjaguinIntegral
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline
from math import log , exp, sqrt, inf, tanh
pi = math.pi
import matplotlib.pyplot as plt
from StericModule import elleff, hMilner, StericPotentialMilner2faces, findStretchedEll

# Constants
Na = 6.02*10**23
Rg = 8.314 #J/(mol.K); #perfect gas constant
Tbase = 273.15
kB = Rg/Na; #boltzmann constant

def eVol(ell,eVparameter):
    #excluded volume
    return (ell*eVparameter)**3

def parabolicProfile(x, a, b):
    return a - b*x**2

def compressedACD(N,kappa,kappap,zp,v0,c0salt,h,Gamma1,sigmaP,B): 

    Cc =  (zp*kappa**2*h*B/(c0salt*v0*kappap**3)*exp(-kappap*h) - Gamma1)/(1 - exp(-2*kappap*h))
    Dc = (zp*kappa**2*h*B/(c0salt*v0*kappap**3) - exp(-kappap*h)*Gamma1)/(1 - exp(-2*kappap*h))
    Ac =  (h**2)/3*B + (N*sigmaP*v0/h - zp*Gamma1/(kappap*h))*kappap**2/kappa**2 #does not vanish for zp = 0
    return(Ac,Cc,Dc)

def uncompressedACD(N,kappa,kappap,zp,v0,c0salt,d,h,Gamma1,sigmaP,B): 
    
    def betaA():
        return zp*kappa**2*kappa/(2*c0salt*v0*kappap**2)*tanh(kappa*(d-h))
    def betaC():
        return (kappa*(1 + exp(-2*kappap*h))*tanh(kappa*(d-h))+kappap*(1 - exp(-2*kappap*h))) #does not vanish for zp = 0
    def beta_forC():
        return exp(-kappap*h)*zp*kappa**2*B/(2*c0salt*v0*kappap**2)*(kappa*tanh(kappa*(d-h))*(h**2 + 2/kappap**2)+2*h) - \
                (kappa*tanh(kappa*(d-h))+kappap)*Gamma1
    def beta_forD():
        return zp*kappa**2*B/(2*c0salt*v0*kappap**2)*(kappa*tanh(kappa*(d-h))*(h**2 + 2/kappap**2)+2*h) 
    def beta_forA():
        return zp*kappa**2*B/(2*c0salt*v0*kappap**2)*(kappa*tanh(kappa*(d-h))*(h**2 + 2/kappap**2)+2*h)           


    def alphaA():
        return h*kappa**2/kappap #does not vanish for zp = 0
    def alphaC(): 
        return -zp*(1 - exp(-2*kappap*h))
    def alpha_forC():
        return exp(-kappap*h)*N*v0*sigmaP*kappap + \
        exp(-kappap*h)*B*kappa**2/(2*c0salt*v0)*(2*c0salt*v0*h**3/(3*kappap)-2*h*zp**2/kappap**3) + \
        zp*(1-exp(-kappap*h))*Gamma1 #does not vanish for zp = 0
    def alpha_forD():
        return N*v0*sigmaP*kappap + \
        B*kappa**2/(2*c0salt*v0)*(2*c0salt*v0*h**3/(3*kappap)-2*h*zp**2/kappap**3)   #does not vanish for zp = 0
    def alpha_forA():
        return N*v0*sigmaP*kappap + \
        B*kappa**2/(2*c0salt*v0)*(2*c0salt*v0*h**3/(3*kappap)-2*h*zp**2/kappap**3)  #does not vanish for zp = 0
    
    def Gamma1_forA():   
        return - zp*Gamma1*(kappa*(exp(-1*kappap*h))*tanh(kappa*(d-h))+2*kappap*(- exp(-1*kappap*h))) + \
            (kappa*(1 + exp(-2*kappap*h))*tanh(kappa*(d-h))+kappap*(1 - exp(-2*kappap*h)))*zp*Gamma1
        
        
    def denom():
        return alphaC()*betaA() - alphaA()*betaC()
    
    
    Au = (-alpha_forA()*betaC() + alphaC()*beta_forA() + Gamma1_forA())/denom();
    Cu = (-alphaA()*beta_forC() + alpha_forC()*betaA())/denom();
    Cm = (-alphaA()*beta_forD() + alpha_forD()*betaA())/denom();
    Du = Gamma1*(zp*(1-exp(-kappap*h))*betaA() - alphaA()*exp(-kappap*h)*(kappa*tanh(kappa*(d-h))-kappap)) + Cm
                

    return(Au,Cu,Du)

def compressedPhi(N,kappa,kappap,zp,v0,c0salt,d,h,Gamma1,sigmaP,B,z):    
    
    Ac,Cc,Dc = compressedACD(N,kappa,kappap,zp,v0,c0salt,h,Gamma1,sigmaP,B)
    
    psic =  Cc*exp(-kappap*z) + Dc*exp(kappap*(z-h)) + \
        (zp*kappa**2/(2*c0salt*v0))*((Ac - B*z**2)/kappap**2 - 2*B/kappap**4)
    phic = (Ac- B*z**2) - zp*psic

    return(phic)
    
def uncompressedPhi(N,kappa,kappap,zp,v0,c0salt,d,h,Gamma1,sigmaP,B,z):    
    
    Au,Cu,Du = uncompressedACD(N,kappa,kappap,zp,v0,c0salt,d,h,Gamma1,sigmaP,B)
    
    psiu =  Cu*exp(-kappap*z) + Du*exp(kappap*(z-h)) + \
        (zp*kappa**2/(2*c0salt*v0))*((Au - B*z**2)/kappap**2 - 2*B/kappap**4)
    phiu = (Au - B*z**2 - zp*psiu)

    return(phiu)

def findPEequilibrium(N,ell,eV,sigmatot,charge,sigmaS,c0salt,lambdaD,eVparameter,accountForF127Stretch,slideType):
 
#def findPEequilibrium(N1,N2,ell1,ell2,eV1,eV2,sigmaP,charge,sigmaS,c0salt,lambdaD,eVparameter,accountForF127Stretch,slideType):
    
    
    
    # N = N1 + N2
    # if N > 0:
    #     ell,eV = elleff(ell1,N1,eV1,ell2,N2,eV2)
    #     print('Effective excluded volume of the  brush ',eV)
    #     # if accountForF127Stretch == True:
    #     #     if slideType == "Glass":
    #     #         sigmatot = sigmaP
    #     #         sigmatot,ell = findStretchedEll(sigmaP,ell,N,eV,"Glass") 
    #     #     else:
    #     #         #find increased density based on stretched length of full brush
    #     #         #sigmabtot = findStretchedEll(sigmab,ellb,Nb,slideType) 
    #     #         #find ncreased density based on stretched of just the PEO brush
    #     #         # if N1b > 0:
    #     #         #     sigmabtot,ellb = findStretchedEll(sigmab,ell1,N1b,eV1,slideType) 
    #     #         # else:
    #     #             # actually I think this should just add to the whole brush... 
    #     #         sigmatot,ell = findStretchedEll(sigmaP,ell,N,eV,"") 
    #     #     print('Effective density of the  brush (nm^2)',1/sigmatot*1e9*1e9)
    #     # else:
    #     # accounting for stretched brushes was already done at an earlier stage in the density sigmaP coming in
    #     sigmatot = sigmaP
    # else:
    #     ell = ell1 #doesn't matter which one you pick cause there's nothing there
    #     eV = eV1
    #     sigmatot = sigmaP
    
    # if N1 > 0:
    #     zp = charge/(N1+N2) #relative number of DNA charges per unit
    # elif abs(charge) >0:
    #     zp = charge/N2
    # else: 
    #     zp = 0
        
    zp = charge/N    
    
    htyp = hMilner(sigmatot,ell,N,eV) #typical height of the brush
    
    electron = 1.6*10**-19 #elementary electron charge
    v0 = eVol(ell,eVparameter) #use the persistence length here.. more likely %(pi/6)** simplification from Macromolecules, 1989, Misra Varanasi
    kappa = 1/lambdaD
    kappap = kappa*sqrt(1+ zp**2/(2*c0salt*v0));
    B = pi**2/(8*N**2*ell**2); # I keep every potential value /kBT because it just makes notations easier
    #sigmaS #surface charge (with usual units charge/m2)
    Gamma1 = - sigmaS/(2*electron*c0salt)*kappa/kappap**2 #- electron*sigmat/(eps0*epsW*kappap*kB*T); this is non dimensional
    
    def fu2(h):
        return(uncompressedPhi(N,kappa,kappap,zp,v0,c0salt,htyp*10,h,Gamma1,sigmatot,B,h))
    heq = brentq(fu2,htyp*0.9,htyp*10)
    
    # then you need to find the profile and fit it with a polynomial
    # this is actually unnecessary since the tip profile only depends on heq
    #allzs = np.linspace(0,heq,100)
    #allphis = np.zeros(len(allzs))
    
    
    #for iz in range(len(allzs)):
    #    z = allzs[iz]
    #    allphis[iz] = uncompressedPhi(kappa,kappap,zp,v0,c0salt,htyp*10,heq,Gamma1,sigmaP,B,z)
    
    #popt, pcov = curve_fit(parabolicProfile, allzs, allphis)
    #A = popt[0]
    #B = popt[1]
    #return(heq,A,B,N,zp,ell)
    #return(heq,N,zp,ell,eV)
    return(heq,zp)
    

def findPEequilibriumSym(N,ell,eV,zp,sigmaP,sigmaS,c0salt,lambdaD,eVparameter):
    
    htyp = hMilner(sigmaP,ell,N,eV) #typical height of the brush
    electron = 1.6*10**-19 #elementary electron charge
    v0 = eVol(ell,eVparameter) #use the persistence length here.. more likely %(pi/6)** simplification from Macromolecules, 1989, Misra Varanasi
    kappa = 1/lambdaD
    kappap = kappa*sqrt(1+ zp**2/(2*c0salt*v0));
    B = pi**2/(8*N**2*ell**2); # I keep every potential value /kBT because it just makes notations easier
    #sigmaS #surface charge (with usual units charge/m2)
    Gamma1 = - sigmaS/(2*electron*c0salt)*kappa/kappap**2 #- electron*sigmat/(eps0*epsW*kappap*kB*T); this is non dimensional
    
    def fu23(h):
        return(uncompressedPhi(N,kappa,kappap,zp,v0,c0salt,htyp*10,h,Gamma1,sigmaP,B,h))
    #heq = brentq(fu2,htyp*0.9,htyp*10)
    heq = fsolve(fu23,htyp)
    
    return (heq)

def calculateEffectiveSymmetricParam(Nb,Nt,ellb,ellt,eVb,eVt,sigmab,sigmat,chargeb,charget,sigmaSb,sigmaSt,c0salt,lambdaD,eVparameter,accountForF127Stretch,slideType):
    # this is only observed if there is DNA on both sides... 
    # if there is not DNA on both sides, I will deal with that later
    
    # first find equilibrium heights and profiles for each of the brushes
    #hb,Ab,Bb,Nb,zb,ellb = findPEequilibrium(N1b,N2b,ell1,ell2,sigmab,sigmaSb,c0salt,lambdaD)
    #ht,At,Bt,Nt,zt,ellt = findPEequilibrium(N1t,N2t,ell1,ell2,sigmat,sigmaSt,c0salt,lambdaD)
    
    
    htypb = hMilner(sigmab,ellb,Nb,eVb) #typical height of the brush
    print('Bottom height of  brush (nm)', htypb*1e9)
    htypb = hMilner(sigmat,ellt,Nt,eVt) #typical height of the brush
    print('Top height of  brush (nm)', htypb*1e9)
    print('charges',chargeb,charget)
    
    # # perform "presymmetrizing" rules 
    
    # # RULE SET PRESYMMETRIZING, BASED ON #2
    
    # hb,Nb,zb,ellb = findPEequilibrium(N1b,N2b,ell1,ell2,sigmab,0,sigmaSb,c0salt,lambdaD,eVparameter)
    # ht,Nt,zt,ellt = findPEequilibrium(N1t,N2t,ell1,ell2,sigmat,0,sigmaSt,c0salt,lambdaD,eVparameter)
    
    # hbtest,Nbtest,zb,ellbtest = findPEequilibrium(N1b,N2b,ell1,ell2,sigmab,chargeb,sigmaSb,c0salt,lambdaD,eVparameter)
    # httest,Nttest,zt,ellttest = findPEequilibrium(N1t,N2t,ell1,ell2,sigmat,charget,sigmaSt,c0salt,lambdaD,eVparameter)
    
    # A = (sigmab*ellb**5)**(1/3)*Nb + (sigmat*ellt**5)**(1/3)*Nt
    # B = sigmab**2*ellb**3*Nb**2 + sigmat**2*ellt**3*Nt**2
    # Cc = sigmab*zb*Nb/hb*sigmat*zt*Nt/ht
    # Dd = Nt/2 + Nb/2
    
    # heq = hb/2 + ht/2
    
    # N = Dd
    # sigmaMax = (2**(4/7)*B**(5/7))/(A**(9/7)*Dd**(1/7))
    # elleff = A**(6/7)/(2**(5/7)*B**(1/7)*Dd**(4/7))
    # zp = (A**(9/7)*sqrt(Cc)*heq)/(2**(4/7)*B**(5/7)*Dd**(6/7))
    
            
    
    # print('Individual equilibrium heights at this temperature are')
    # print('Bottom height, effective charge, effective persistence', hbtest, zb, ellbtest)
    # print('Top height, effective charge, effective persistence', httest, zt, ellttest)
    
    
    hb,zb = findPEequilibrium(Nb,ellb,eVb,sigmab,chargeb,sigmaSb,c0salt,lambdaD,eVparameter,accountForF127Stretch,slideType)
    ht,zt = findPEequilibrium(Nt,ellt,eVt,sigmat,charget,sigmaSt,c0salt,lambdaD,eVparameter,accountForF127Stretch,slideType)
    
            
    
    print('Individual equilibrium heights at this temperature are')
    print('Bottom height, effective charge, effective persistence', hb, zb, ellb)
    print('Top height, effective charge, effective persistence', ht, zt, ellt)
    
    # now search for symmetric parameters
    # the equivalent sigma will conserve the charge density squared
    #sigmaFac = sqrt(sigmat*sigmab) # that choice is more adjusted towards steric repulsion
    #sqrt(zt*Nt*sigmat*sigmab*Nb*zb/(hb*ht))/abs(Nb*zb/2+Nt*zt/2) #this has to be multiplied by h to get the right sigma # this is rather critical when zt = 0
    #sigmaMax = sigmaFac#*(ht+hb)/2
    #N = (Nb + Nt)/2
    #zp = sqrt(Nb*Nt*zb*zt/(hb*ht))*(hb+ht)/(2*N)#(Nb*zb+Nt*zt)/(Nb+Nt)
    sigmaS = sqrt(sigmaSb*sigmaSt) #that makes perfect sense in terms of interactions that scale as sigma*sigma
    
    
    #elleff = brentq(symmetricEll,ellb*0.1,ellb*3)
    
    # # RULE SET 8
    # # all the rule sets that do not have a solve to conserve the heights, basically they are not based on proper science.    
    # A = (sigmab*ellb**5)**(1/3)*Nb + (sigmat*ellt**5)**(1/3)*Nt
    # B = sigmab**2*ellb**3*Nb**2 + sigmat**2*ellt**3*Nt**2
    # Cc =  sigmab*zb*Nb/hb+zt*Nt*sigmat/ht
    
    # heq = hb/2 + ht/2
    
    # # then search for the uncompressed height
    # # def symmetricEll(Nn):
    # #     s = abs((2**(4)*B**(5))/(A**(9)*Nn**(1)))**(1/7)
    # #     ell = abs(A**(6)/(2**(5)*B**(1)*Nn**(4)))**(1/7)
    # #     z = -sqrt(Cc)*heq*abs((A**(9))/(2**(4)*B**(5)*Nn**(6)))**(1/7)
        
    # #     hse = findPEequilibriumSym(Nn,ell,z,s,sigmaS,c0salt,lambdaD,eVparameter)
    # #     return(hse - heq)    
    # # Nne = fsolve(symmetricEll,Nb/2+Nt/2) #find the elleff that conserves the total height of brushes    
    # # N = float(Nne)
    # # result = symmetricEll(Nne)
    # # print(result)
    # # #N = (16*B**5)/(A**9*sigmaMax**7)
    # # #elleff = (A**6*sigmaMax**4)/(8*B**3)
    # # #zp = -sqrt(Cc)*heq*((A**10*sigmaMax**(6))/B**5)/(16*A)
    # # sigmaMax = (2**(4/7)*B**(5/7))/(A**(9/7)*N**(1/7))
    # # elleff = A**(6/7)/(2**(5/7)*B**(1/7)*N**(4/7))
    # # zp = -sqrt(Cc)*heq*((A**(9/7))/(2**(4/7)*B**(5/7)*N**(6/7)))
    
    # def symmetricEll(ell):
    #     s = (2**(3/4)*B**(3/4)*ell**(1/4))/A**(3/2)
    #     Nn = A**(3/2)/(2*2**(1/4)*B**(1/4)*ell**(7/4))
    #     z = (Cc*heq*ell**(3/2))/(sqrt(2)*sqrt(B))
        
    #     hse = findPEequilibriumSym(Nn,ell,z,s,sigmaS,c0salt,lambdaD,eVparameter)
    #     return(hse - heq)    
    # elle = fsolve(symmetricEll,ellt) #find the elleff that conserves the total height of brushes    
    # elleff = float(elle)
    # result = symmetricEll(elleff)
    # print(result)
    # #N = (16*B**5)/(A**9*sigmaMax**7)
    # #elleff = (A**6*sigmaMax**4)/(8*B**3)
    # #zp = -sqrt(Cc)*heq*((A**10*sigmaMax**(6))/B**5)/(16*A)
    # sigmaMax = (2**(3/4)*B**(3/4)*elleff**(1/4))/A**(3/2)
    # N = A**(3/2)/(2*2**(1/4)*B**(1/4)*elleff**(7/4))
    # zp = (Cc*heq*elleff**(3/2))/(sqrt(2)*sqrt(B))
    
    
    # # RULE SET 7
    # # all the rule sets that do not have a solve to conserve the heights, basically they are not based on proper science.    
    # A = (sigmab*ellb**5)**(1/3)*Nb + (sigmat*ellt**5)**(1/3)*Nt
    # B = sigmab**2*ellb**3*Nb**2 + sigmat**2*ellt**3*Nt**2
    # Cc = sigmab*zb*Nb/hb*sigmat*zt*Nt/ht
    
    # heq = hb/2 + ht/2
    
    # # then search for the uncompressed height
    # # def symmetricEll(Nn):
    # #     s = abs((2**(4)*B**(5))/(A**(9)*Nn**(1)))**(1/7)
    # #     ell = abs(A**(6)/(2**(5)*B**(1)*Nn**(4)))**(1/7)
    # #     z = -sqrt(Cc)*heq*abs((A**(9))/(2**(4)*B**(5)*Nn**(6)))**(1/7)
        
    # #     hse = findPEequilibriumSym(Nn,ell,z,s,sigmaS,c0salt,lambdaD,eVparameter)
    # #     return(hse - heq)    
    # # Nne = fsolve(symmetricEll,Nb/2+Nt/2) #find the elleff that conserves the total height of brushes    
    # # N = float(Nne)
    # # result = symmetricEll(Nne)
    # # print(result)
    # # #N = (16*B**5)/(A**9*sigmaMax**7)
    # # #elleff = (A**6*sigmaMax**4)/(8*B**3)
    # # #zp = -sqrt(Cc)*heq*((A**10*sigmaMax**(6))/B**5)/(16*A)
    # # sigmaMax = (2**(4/7)*B**(5/7))/(A**(9/7)*N**(1/7))
    # # elleff = A**(6/7)/(2**(5/7)*B**(1/7)*N**(4/7))
    # # zp = -sqrt(Cc)*heq*((A**(9/7))/(2**(4/7)*B**(5/7)*N**(6/7)))
    
    # def symmetricEll(ell):
    #     s = (2**(3/4)*B**(3/4)*ell**(1/4))/A**(3/2)
    #     Nn = A**(3/2)/(2*2**(1/4)*B**(1/4)*ell**(7/4))
    #     z = -((sqrt(2)*sqrt(Cc)*heq*ell**(3/2))/sqrt(B))
        
        
    #     hse = findPEequilibriumSym(Nn,ell,z,s,sigmaS,c0salt,lambdaD,eVparameter)
    #     return(hse - heq)    
    # elle = fsolve(symmetricEll,ellt) #find the elleff that conserves the total height of brushes    
    # elleff = float(elle)
    # result = symmetricEll(elleff)
    # print(result)
    # #N = (16*B**5)/(A**9*sigmaMax**7)
    # #elleff = (A**6*sigmaMax**4)/(8*B**3)
    # #zp = -sqrt(Cc)*heq*((A**10*sigmaMax**(6))/B**5)/(16*A)
    # sigmaMax = (2**(3/4)*B**(3/4)*elleff**(1/4))/A**(3/2)
    # N = A**(3/2)/(2*2**(1/4)*B**(1/4)*elleff**(7/4))
    # zp = -((sqrt(2)*sqrt(Cc)*heq*elleff**(3/2))/sqrt(B))
    
    
    # ## RULE SET 6
    # # all the rule sets that do not have a solve to conserve the heights, basically they are not based on proper science.    
    # B = sigmab**2*ellb**3*Nb**2 + sigmat**2*ellt**3*Nt**2
    # Cc = sigmab*zb*Nb/hb+zt*Nt*sigmat/ht
    # Dd = Nt/2 + Nb/2
    
    # heq = hb/2 + ht/2
    
    # N = Dd
    # # conserving the charge itself does not work too well so let's we conserve the average charge density
    # # then search for the uncompressed height
    # def symmetricEll(ellf):
    #     sigmaM = sqrt(B/(2*ellf**3*N**2))
    #     z = np.sign(zb)*Cc/(2)*(heq/(N*sigmaM))
    #     hs = findPEequilibriumSym(N,ellf,z,sigmaM,sigmaS,c0salt,lambdaD,eVparameter)
    #     return(hs - heq)    
    # elleff = fsolve(symmetricEll,ellt) #find the elleff that conserves the total height of brushes 
    
    # sigmaMax = sqrt(B/(2*elleff**3*N**2))
    # zp = np.sign(zb)*Cc/(2)*(heq/(N*sigmaMax))
    
    ## RULE SET 5
    # A = (sigmab*ellb**5)**(1/3)*Nb + (sigmat*ellt**5)**(1/3)*Nt
    # B = sigmab**2*ellb**3*Nb**2 + sigmat**2*ellt**3*Nt**2

    
    # # else:
    # #     print('Using basic rules')
    # N = Nb/2+Nt/2
    # sigmaMax = (2**(4/7)*B**(5/7))/(A**(9/7)*N**(1/7))
    # elleff = A**(6/7)/(2**(5/7)*B**(1/7)*N**(4/7))
    # zp = zb*Nb/(2*N) + zt*Nt/(2*N)
    
    # heq = hb/2+ht/2
    
    
    #RULE SET 3
    
    B = sigmab**2*ellb**3*Nb**2 + sigmat**2*ellt**3*Nt**2
    Cc = sigmab*zb*Nb/hb*sigmat*zt*Nt/ht
    Dd = Nt/2 + Nb/2 #this is an "arbitrary choice" to conserve the total number of pathlengths
    eV = sqrt(eVt*eVb) # in any case I think this is 1
    
    
    heq = hb/2 + ht/2
    
    N = Dd
    
    #then search for the uncompressed height
    def symmetricEll(ellf):
        sigmaM = sqrt(B/(2*ellf**3*N**2)) #this conserves the steric forces
        z = np.sign(zb)*sqrt(Cc)*heq/(sigmaM*N) #a priori charges on both sides are of the same sign (this conserves the charge density^2)
        hs = findPEequilibriumSym(N,ellf,eV,z,sigmaM,sigmaS,c0salt,lambdaD,eVparameter) #conserve the equilibrium height
        return(hs - heq)    
    elleff = fsolve(symmetricEll,ellt) #find the elleff that conserves the total height of brushes    
    sigmaMax = sqrt(B/(2*elleff**3*N**2))
    zp = np.sign(zb)*sqrt(Cc)*heq/(sigmaMax*N) #a priori charges on both sides are of the same sign
    
    
    ## RULE SET 2

    # A = (sigmab*ellb**5)**(1/3)*Nb + (sigmat*ellt**5)**(1/3)*Nt
    # B = sigmab**2*ellb**3*Nb**2 + sigmat**2*ellt**3*Nt**2
    # Cc = sigmab*zb*Nb/hb*sigmat*zt*Nt/ht
    # Dd = Nt/2 + Nb/2
    
    # heq = hb/2 + ht/2
    
    # N = Dd
    # sigmaMax = (2**(4/7)*B**(5/7))/(A**(9/7)*Dd**(1/7))
    # elleff = A**(6/7)/(2**(5/7)*B**(1/7)*Dd**(4/7))
    # zp = (A**(9/7)*sqrt(Cc)*heq)/(2**(4/7)*B**(5/7)*Dd**(6/7))
    
    
    ## RULE SET 1
    
    # heq = ht/2 + hb/2 
    # A = (sigmab*ellb**5)**(1/3)*Nb + (sigmat*ellt**5)**(1/3)*Nt
    # B = sigmab**2*ellb**3*Nb**2 + sigmat**2*ellt**3*Nt**2
    # Cc = sigmab*zb*Nb/hb*sigmat*zt*Nt/ht
    # Dd = zt*Nt + zb*Nb

    # if abs(zb) > 0 or abs(zt) > 0:
    #     print('Using symmetric rule set #1')
    #     N = (B**5*abs(Dd)**7)/(8*A**9*Cc**(7/2)*heq**7)
    #     sigmaMax = (2*sqrt(Cc)*heq)/abs(Dd)
    #     elleff = (2*A**6*Cc**2*heq**4)/(B**3*Dd**4)
    #     zp = Dd/(2*N)
    
    # #N = 16*B**5/(A**9*sigmaMax**7)
    # #elleff = A**6*sigmaMax**4/(8*B**3)
    # #zp = np.sign(zb)*sqrt(C)*heq/(sigmaMax*N)    
    # #hs = findPEequilibriumSym(N,elleff,zp,sigmaMax,sigmaS,c0salt,lambdaD,eVparameter)

    
    # #N = Nb/2+Nt/2
    # #sigmaMax = sqrt(sigmab*sigmat)
    # #heq = ht/2 + hb/2 # conserve the height
    # # then search for the uncompressed height
    # #def symmetricEll(ellf):
    # #    #sigmaM = sqrt(B/(2*ellf**3*N**2))
    # #    Ne = sqrt(B/(2*ellf**3*sigmaMax**2))
    # #    #z = np.sign(zb)*sqrt(C)*heq/(sigmaM*N) #a priori charges on both sides are of the same sign
    # #    #hs = findPEequilibriumSym(N,ellf,z,sigmaM,sigmaS,c0salt,lambdaD,eVparameter)
    # #    z = np.sign(zb)*sqrt(C)*heq/(sigmaMax*Ne) #a priori charges on both sides are of the same sign
    # #    hs = findPEequilibriumSym(Ne,ellf,z,sigmaMax,sigmaS,c0salt,lambdaD,eVparameter)
        
    # #    return(hs - heq)    
    # #elleff = fsolve(symmetricEll,ellt) #find the elleff that conserves the total height of brushes    
    # #sigmaMax = sqrt(B/(2*elleff**3*N**2))
    # #N = sqrt(B/(2*elleff**3*sigmaMax**2))
    # #zp = np.sign(zb)*sqrt(C)*heq/(sigmaMax*N) #a priori charges on both sides are of the same sign
    
    
    
    
    hs = findPEequilibriumSym(N,eV,elleff,zp,sigmaMax,sigmaS,c0salt,lambdaD,eVparameter)
    
    #hb = hs
    #ht = hs
    
    # else:
    #     print('Using basic rules')
    #     N = (16*B**5)/(A**9*sigmat**(7/2)*sigmab**(7/2))
    #     sigmaMax = sqrt(sigmat*sigmab)
    #     elleff = A**6*sigmat**2*sigmab**2/(8*B**3)
    #     zp = zb*Nb/(2*N) + zt*Nt/(2*N)
        
    print('Effective symmetric parameters at that temperature are')
    print(elleff,N,sigmaMax,sigmaS,zp*N,heq,hs)
    
    return(elleff,eV,N,sigmaMax,sigmaS,zp,hb,ht)
    
    
# this will give typical A and B overlap at long distances - that will be used for Hybridization
# you can even take the assymetric ones, it should do the job


def polyelectrolytePotential(allheights,lambdaD,ell,N,sigmaMax,zp,c0salt,sigmaS,Radius,eVparameter,optcolloid):
# then you need to calculate the interaction energy, for that you can use symmetric configuration

    electron = 1.6*10**-19 #elementary electron charge
    v0 = eVol(ell,eVparameter) #use the persistence length here.. more likely %(pi/6)** simplification from Macromolecules, 1989, Misra Varanasi
    kappa = 1/lambdaD
    kappap = kappa*sqrt(1+ zp**2/(2*c0salt*v0));
    B = pi**2/(8*N**2*ell**2); # I keep every potential value /kBT because it just makes notations easier
    #sigmaS #surface charge (with usual units charge/m2)
    Gamma1 = - sigmaS/(2*electron*c0salt)*kappa/kappap**2 #- electron*sigmat/(eps0*epsW*kappap*kB*T); this is non dimensional

    polyEl = np.zeros(len(allheights))
    compressedEl = np.zeros(len(allheights))
    referenceSter = np.zeros(len(allheights))
    heq = hMilner(sigmaMax, ell, N,eVparameter)
    # determine first the critical heights for this type of brush
    disc = 100 #discretization factor -- I checked that results are well converged from this discretization factor onwards
    sigmas = np.linspace(0,sigmaMax,disc)
    hcrits = np.zeros(disc)
    height = np.mean(allheights)/2
    hreal = 0
    for isi in range(disc):        
        sigmaP = sigmas[isi]
        if sigmaP > 0:
            # now we have to figure the equilibrium height for this sigma
            htyp = hMilner(sigmaP,ell,N,eVparameter) #typical height of the brush

            def fc2(h):
                return(compressedPhi(N,kappa,kappap,zp,v0,c0salt,h,h,Gamma1,sigmaP,B,h))
            #h1critS = brentq(fc2,htyp*0.9,height)
            h1critS = fsolve(fc2,htyp)
            hcrits[isi] = h1critS
    
    #now you can compute the interaction energy
    for ih in range(len(allheights)):
        distance = allheights[ih]/2 #consider only half the distance for symmetric brushes
        Asigma = np.zeros(disc)
        for isi in range(disc):
            sigmaP = sigmas[isi]
            if sigmaP > 0:
            # if the layers are not squished
                if distance > hcrits[isi]: 
                    def fu2(h):
                        return(uncompressedPhi(N,kappa,kappap,zp,v0,c0salt,distance,h,Gamma1,sigmaP,B,h))
                    #hreal = brentq(fu2,hcrits[isi],distance)
                    hreal = fsolve(fu2,hcrits[isi]) # this is trying to solve that the potential vanishes at that height
                    Asigma[isi],emp1,emp2 = uncompressedACD(N,kappa,kappap,zp,v0,c0salt,distance,hreal,Gamma1,sigmaP,B)
                #otherwise the layers are squished
                else:
                    hreal = distance;
                    Asigma[isi],emp1,emp2 = compressedACD(N,kappa,kappap,zp,v0,c0salt,hreal,Gamma1,sigmaP,B)
        compressedEl[ih] = hreal # effective heights    
        polyEl[ih] = 2*N/(sigmaMax)*trapz(Asigma,sigmas) #factor 2 because there are 2 layers so twice the energy
        if distance < heq:
            referenceSter[ih] = 2*N*(sigmaMax*v0/ell)**(2/3)*(pi**2/12)**(1/3)*(heq/(2*distance) + (distance/heq)**2/2 - (distance/heq)**5/10 - 9/10)
    
    polyEl = polyEl - polyEl[-1] #substract the far field interaction
    
    for ih in range(len(allheights)):
        if polyEl[ih] < 1e-9:
            polyEl[ih] = 0
            
    
    # fig2, ax2 = plt.subplots(figsize=(6,4))
    # ax2.plot(allheights*1e9,-polyEl,label='integrated')
    # ax2.plot(allheights*1e9,-referenceSter,'--',label='calculus')
    # ax2.set_xlim(50,100)
    # ax2.set_ylim(-30,10)
    # plt.show() # this plot shows great agreement for no charges. 
    
    phiPE = DerjaguinIntegral(allheights[-1],Radius,allheights,sigmaMax*polyEl,optcolloid)
    #phiPE = DerjaguinIntegral(allheights[-1],Radius,allheights,sigmaMax*referenceSter)
    #phiPEanal = StericPotentialMilner2faces(allheights/2,Radius,sigmaMax,ell,N)

    #fig3, ax3 = plt.subplots(figsize=(6,4))
    #ax3.semilogy(allheights*1e9,phiPE,label='integrated')
    #ax3.semilogy(allheights*1e9,phiPEanal,label='calculus')
    #ax3.legend(loc='upper right',
    #          fontsize=10,
    #          frameon=False)
    #ax3.set_xlim(20,100)
    #plt.show()

    return(phiPE,compressedEl)
