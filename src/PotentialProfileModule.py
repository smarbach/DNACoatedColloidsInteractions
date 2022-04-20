#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:41:25 2020

@author: sophie marbach
"""

import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
from math import sqrt, log, exp, erf, sinh, cosh, pi, inf
from joblib import load
import pickle

from WaterDensityModule import density
from ElectrostaticsModule import permittivity, lambdaD, Graham, electrostaticPotential, electrostaticPotentialPlates
from DepletionModule import DepletionPotential
from VanDerWaalsModule import VanDerWaalsPotentialFull, VanDerWaalsPotentialDejarguin
from StericModule import calculateCompressedHeights, StericPotentialFull, determineRelevantHeights, hMilner
from HybridizationModule import bridgingPotential, compute_DNAbindingEnergy, bridgingConstants
from MicroscopicModule import calculateMicroscopicDetails
from PolyelectrolyteModule import calculateEffectiveSymmetricParam, polyelectrolytePotential
from UnifiedModule import unifiedPotentials

def compute_potential_profile(radius, PSdensity, gravity, saltConcentration, Tin, PSCharge, GlassCharge, \
                                cF127, topParameters, bottomParameters, ellDNAnotused, ellPEO, wPEO, slideType, gravityFactor, \
                                Sequence, srcpath, nresolution, micellePenetrationInsideBrush, PEmodel, eVparameter, \
                                slabH, bridgingOption, basename, \
                                criticalHeight, optcolloidcolloidFlag, optdeplFlag, optvdwFlag, mushroomFlag, porosityIn,DNAmodel,  *args, **kwargs):

    # now Tin can be a vector. 
    
    # Global physical constants
    electron = 1.6*10**-19 #elementary electron charge
    Na = 6.02*10**23
    Rg = 8.314 #J/(mol.K); #perfect gas constant
    Tbase = 273.15
    kB = Rg/Na #boltzmann constant
    
    ########## TRANSLATE ALL INPUT PARAMETERS INTO USI ###################
    Rt0 = radius*10**-6 #radius of the colloids (m)
    c0salt = (saltConcentration*10**3)*Na #salt concentration in parts/m^3 (the first number is in mol/L)
    rhoPS = PSdensity*10**-3/(1*10**-6) #density of PS in kg/m3 (tabulated)
    g = gravity #gravity in NYC
    sigmaPS = PSCharge# surface charge in C/m2 of polystyrene
    sigmaGlass = GlassCharge # surface charge in C/m2 of glass
    cm0 = cF127*1e-2*1e3/12600 #convert the F127 concentration in mol/L
    aggRadiusC = kwargs.get('aggRadius',0)*1e-9 #agregation radius in nm
    depletionTypeC = kwargs.get('depletionType','default')
    slabHeight = slabH*1e-6 #was given in um
    porosity = porosityIn*1e-9 #was given in nm
    hamakerC = kwargs.get('hamaker',3e-21)
    accountForDilatation = kwargs.get('dilatation', 0) # this will be an issue when rerunning codes, need to specify that that option is 1 !!
    
    ########## TRANSLATE SPECIFIC INPUT PARAMETERS INTO USI ###################
    h0b = bottomParameters[0]*10**(-9) #thickness of the bottom coating
    h0t = topParameters[0]*10**(-9) #thickness of the top coating
    
    N1b = bottomParameters[1] #number of PEO units
    N1t = topParameters[1] #number of PEO units
    

    sigmab = bottomParameters[4]/(10**-9)**2 # surface coverage 
    sigmat = topParameters[4]/(10**-9)**2 # surface coverage 

    #sigmaDNAb = sigmab*(bottomParameters[3]*electron) # electric surface coverage  in C/m^2
    #sigmaDNAt = sigmat*(topParameters[3]*electron) # electric surface coverage in C/m^2
    
    #chargeb = bottomParameters[3]
    #charget = topParameters[3]
    
    fstickyb = bottomParameters[5]
    fstickyt = topParameters[5]
    sigmastickyb = fstickyb*sigmab #sticky density on the bottom plate
    sigmastickyt = fstickyt*sigmat #sticky density on the top plate

    ell1 = ellPEO*10**(-9) # persistence length of PEO strands #here it's actually taken as the unit PEO length
    
    eVfac1 = wPEO
    eVfac2 = 1.0 #not known, probably around 1 since DNA is quite stiff.. 
    
    ellDNAinf = 0.75
    ellDNA0 = 2.09
    bNA = 5.83
    nNA = 2.01
    
    if PEmodel == "Electrostatics" or PEmodel == "Unified": 
        # then account for salt distortion of the DNA persistence length for the steric repulsion
        
        ellDNA = ellDNAinf + (ellDNA0 - ellDNAinf)/((bNA*saltConcentration)**(nNA/2)+1)
        
    else: 
            # do not do that for the polyelectrolyte model otherwise you are counting things twice.. ??
        #ellDNA = ellDNAinf #keep the infinite salt concentration (equivalent to no charges)
        ellDNA = ellDNAinf + (ellDNA0 - ellDNAinf)/((bNA*saltConcentration)**(nNA/2)+1)
        #print('the retained persistence length is')
        #print(ellDNA)    

    
    if DNAmodel == 'dsDNA':
        # now you have a mix of ss and ds
        bDNA = 0.34 #length of 1bp in a helix
        ellDNA = 3.4 #length of 10bp in a helix
        
    elif DNAmodel == 'ssDNA':
        bDNA = 0.56 #length (in nm) of a ssDNA nucleotide accounting for angles, from Chen paper
        #and ellDNA was defined previously
    elif DNAmodel == 'crocker':
        bDNA = 40/65
        ellDNA = 5
    elif DNAmodel == 'longssDNA':
        bDNA = 40/65
    print('the retained persistence length for DNA (at that salt concentration) is (nm):',ellDNA)


    N2b = bottomParameters[2]*bDNA/ellDNA #number of DNA units relevant for Milner model
    N2t = topParameters[2]*bDNA/ellDNA #number of DNA units relevant for Milner model
    print(N2b,N2t)

    chargeb = bottomParameters[3]
    charget = topParameters[3]

    sigmaDNAb = sigmab*(chargeb*electron) # electric surface coverage  in C/m^2
    sigmaDNAt = sigmat*(charget*electron) # electric surface coverage in C/m^2

    ell2 = ellDNA*10**(-9) # lpersistence ength of DNA strands


        
    ########## SOME FUNCTION DEFINITION FOR BOUYANCY ###################
    def rhoPST(Temp):
        # Temp is in K 
        if accountForDilatation:
            rhoPSt = (PSdensity-(0.000210*(Temp-Tbase-22)))*10**-3/(1*10**-6)
        else:
            rhoPSt = (PSdensity)*10**-3/(1*10**-6)
        #print("Dilatation at that temperature yields a PS density of")
        #print(rhoPSt/1000)
        return(rhoPSt)
    
    def RT(Temp):
        mass = Rt0**3*rhoPST(22+Tbase)
        #print(rhoPST(Temp))
        #print(mass)
        Rt = (mass/rhoPST(Temp))**(1/3)
        
        return(Rt)
        
    def meff(Temp):
        # Temp is in K 
        Rt = RT(Temp)
        return(4/3*pi*Rt**3*(rhoPST(Temp) - density(Temp))) #effective mass of the colloid immerged in water
    

    
    ##### INITIALIZING OUTPUT
    allheights = determineRelevantHeights(N1b,N2b,N2t,N1t,ell1,ell2,eVfac1,eVfac2,sigmab,sigmat,h0b+h0t,nresolution,slabHeight,PEmodel,mushroomFlag) #independent of temperature
    
    lenh = len(allheights)
    lent = len(Tin)
    sz = (lent,lenh)
    phikT = np.zeros(sz)
    phiEl = np.zeros(sz)
    phiDepl = np.zeros(sz)
    phiGrav = np.zeros(sz)
    phiVdW = np.zeros(sz)
    phiSter = np.zeros(sz)
    phiBridge = np.zeros(sz)
    phiPE = np.zeros(sz)
    NBridge = np.zeros(sz)
    Allsigmaabs = np.zeros(sz)
    AllKabs = np.zeros(sz)
    allQ1s = np.zeros(lenh) + 1
    allQ2s = np.zeros(lenh) + 1 
    DeltaG0s = np.zeros(lent)
    nInvolvedT = np.zeros(lent)
    gravityFactors = np.zeros(lent)
    Lvalues = np.zeros(lent)
    #allpunboundb = np.zeros(sz)
    
    allTs = [Tbase + Ti for Ti in Tin] 
    
    ########## CHECK FOR BRIDGING #############
    # bridging occurs if there is something else than a T in the DNA sequence
    bridging = 0
    if sigmastickyb*sigmastickyt > 0: 
        print('detecting if the sequence is sticky')
        for i in range(len(Sequence)-1):
            j=Sequence[i:i+1]
            if j !='T':
                bridging = 1
    
    
    
    ellDNAbond = ellDNAinf #+ (ellDNA0 - ellDNAinf)/((bNA*saltConcentration)**(nNA/2)+1) # for the bond length use the electrostatics model
    Nbond =  len(Sequence)*bDNA/ellDNAbond #number of DNA units relevant for Milner model in the bond
    ellBond = ellDNAbond*10**(-9) 
    hbond = hMilner(1/30*1e18,ellBond,Nbond,eVfac2) #average bond length -- keep it the same for everyone, with a median density
    hbond = 0 # actually for now do not account for this parameter -- since this is of the order of the fluctuations
    print('Typical length of the bond is (nm)', hbond*1e9)            
    
    #hbond = 3e-9
    
    ########## COMPUTE INTERACTIONS #############    
    ########## 1: HEIGHTS OF COATINGS ###################


    #if PEmodel == "Electrostatics": 
        #then find the heights as they are compressed
    if cF127 > 0:
        accountForF127Stretch = True
    else:
        accountForF127Stretch = False
        
    Emins,Ebridgeonly,heights1b,heights2b,heights2t,heights1t,elleb,ellet,eVb,eVt,Nb,Nt,hrest,hrestt,sigmab,sigmat,lbot,ltop = \
            calculateCompressedHeights(N1b,N2b,N2t,N1t,ell1,ell2,eVfac1,eVfac2,sigmab,sigmat,allheights-h0b-h0t, \
                                      bridging*0,accountForF127Stretch,mushroomFlag,slideType)  
        #print(Emins)
        #print(allheights)
     
        
    # this comes from height redistribution and should always be done like that no matter what the polymer brush is
    if mushroomFlag:
        h1eq = hrest
        h2eq = hrestt 
    else:
        h2eq = heights1b[-1]+heights2b[-1]
        h1eq = heights1t[-1]+heights2t[-1]
    print("Height of the top brush",h1eq*1e9)
    print("L0 parameter for finite stretching calculation is (top)",h1eq/((Nt**(1/2))*ellet))
    print("L0 parameter for finite stretching calculation is (bottom)",h2eq/((Nb**(1/2))*elleb))
        #plt.plot(allheights,Ebridgeonly)
        #plt.show()
        
    Emins = Emins - Emins[-1] # adjust for the inaccuracy of the trapz integration
    
    
    if bridging:
        print('There are sticky brushes')
        # compute binding energy once and for all
        bindingEnergies = compute_DNAbindingEnergy(Sequence,saltConcentration)
        # compute binding constants geometrically
        heightst = [heights1t[ih]+heights2t[ih] for ih in range(len(heights1t))]
        heightsb = [heights1b[ih]+heights2b[ih] for ih in range(len(heights1b))]
        #bindingEnergies = compute_DNAbindingEnergy(Sequence,saltConcentration)
        allKabsC, allQ1sC, allQ2sC = \
        bridgingConstants(allheights - h0b - h0t,hbond,h1eq,h2eq,sigmastickyb,sigmastickyt,sigmat,ellet,Nt,sigmab,elleb,Nb,heightst,heightsb,mushroomFlag, porosity)
    
    
    ########## 2: FREE ENERGIES ###################
    # now iterate on temperature
    idT = 0
    for T in allTs:        
        
        #print('#############################################')
        print('Temperature is (C)', T-Tbase)
        # 2 - a - electrostatic factors
        Rt = RT(T)
        #print("Dilatation at that temperature yields a radius increase of",Rt/Rt0)
        dW = density(T)
        #print('Water density is', dW)
        epsilon0r =  permittivity(T,density(T))
        lambdaV = lambdaD(T,c0salt) 
        #print('Debye Length is (nm)', lambdaV*1e9)
        gamma1 = Graham(T,lambdaV,sigmaPS)
        gamma2 = Graham(T,lambdaV,sigmaGlass)
        prefactorEl = pi*Rt*kB*T*epsilon0r/electron**2
        
        #print(prefactorEl)
        # electrostatics are now calculated after bridging
        
        
        # 2 - b - gravity
        mass = meff(T)
            
        #print('mass of the particle is (kg)', mass)
        if optcolloidcolloidFlag == 0: #only calculate gravity for surface colloid interactions
            if gravityFactor > 0:
                phiGrav[:][idT] = gravityFactor*allheights
                #print('PS Density enhancement is', gravityFactor*kB*T/(mass*g))
                gravityFactors[idT]  = gravityFactor
                
            else:
                phiGrav[:][idT] = mass*g*allheights/(kB*T)
                #print('gravity is (pN)', mass*g*1e12)
                gravityFactors[idT] =  mass*g/(kB*T)
        
        if optvdwFlag == 2:
            # 2 - c - Van der waals - plate/plate distance counts
            phiVdW[:][idT] = VanDerWaalsPotentialFull(T - Tbase,saltConcentration,allheights,Rt,slideType,srcpath,optcolloidcolloidFlag, hamaker = hamakerC)    
        elif optvdwFlag == 1:
            phiVdW[:][idT] = VanDerWaalsPotentialDejarguin(T - Tbase,saltConcentration,allheights,Rt,slideType,srcpath,optcolloidcolloidFlag, hamaker = hamakerC)    
        elif optvdwFlag == 0:
            phiVdW[:][idT] = [0 for h in allheights]
        
        
        if Nb+Nt == 0:
            print('There are no brushes on your polymer')
            # 2 - d - depletion
            realseparation = allheights - h0b - h0t -(heights1b[-1]+heights2b[-1]+heights1t[-1]+heights2t[-1]) 
            # maybe with a factor 3pi/16... depending on what counts for depletion... the total brush height or the average brush height... 
            #plt.plot(allheights,realseparation)
            if optdeplFlag:
                phiDepl[:][idT] = micellePenetrationInsideBrush*DepletionPotential(T,cm0,realseparation,Rt,optcolloidcolloidFlag, aggRadius = aggRadiusC, depletionType = depletionTypeC) #+h0t+heights1t[-1]+heights2t[-1] you also increase the effective radius of the particle here.
                                                                    # this has a very weak effect, so I would not account for it in an attempt to keep the model sane... 
                
            # 2 - e - steric potential --- no steric potential, no bridging
                
            # 2 - a - again electrostatics - you still need the plates interacting
            phiEl[:][idT] = electrostaticPotentialPlates(allheights,h0b,h0t,lambdaV,c0salt,Rt,gamma2,gamma1,optcolloidcolloidFlag)
                
            
        elif bridging == 0: #there are brushes but no bridging
            print('There are only passive brushes (non sticky)')
            if PEmodel == "Electrostatics" or PEmodel == "Unified": #then you need to follow a certain order to calculate different contributions
                # normally the unified model with bridging = 0 is the electrostatics or polyelectrolyte model depending on how you want to account for the charge
                
                # 2 - d - depletion
                realseparation = allheights - h0b - h0t -(heights1b[-1]+heights2b[-1]+heights1t[-1]+heights2t[-1]) 
                # maybe with a factor 3pi/16... depending on what counts for depletion... the total brush height or the average brush height... 
                #plt.plot(allheights,realseparation)
                if optdeplFlag:
                    phiDepl[:][idT] = micellePenetrationInsideBrush*DepletionPotential(T,cm0,realseparation,Rt,optcolloidcolloidFlag, aggRadius = aggRadiusC, depletionType = depletionTypeC) #+h0t+heights1t[-1]+heights2t[-1] you also increase the effective radius of the particle here.
                                                                    # this has a very weak effect, so I would not account for it in an attempt to keep the model sane... 
                
                # 2 - e - steric potential # this one does not depend on temperature but it has to be calculated after
                if idT == 0:
                    hmaxSteric = h1eq+h2eq
                    #print(hmaxSteric)
                    print('Calculating steric repulsion once')
                    phiSter[:][idT] = StericPotentialFull(allheights-h0t-h0b,Rt,sigmat,ellet,Nt,eVt,sigmab,elleb,Nb,eVb,hmaxSteric,Emins,optcolloidcolloidFlag,mushroomFlag, allQ1s, allQ2s)
                else:
                    phiSter[:][idT] = phiSter[:][0]
                    
                # 2 - a - again electrostatics - you still need the plates interacting
                phiEl[:][idT] = electrostaticPotentialPlates(allheights,h0b,h0t,lambdaV,c0salt,Rt,gamma2,gamma1,optcolloidcolloidFlag)
                    
                    
            elif PEmodel == "Polyelectrolyte": #then you need to follow another order
        
                
                # -- first you need effective brush parameters ---
                elleff,eVparameter,Neff,sigmaMax,sigmaS,zp,hb,ht = calculateEffectiveSymmetricParam(Nb,Nt,elleb,ellet,eVb,eVt,sigmab,sigmat,chargeb,charget,sigmaGlass,sigmaPS,c0salt,lambdaV,eVparameter,accountForF127Stretch,slideType)
                
                # 2 - e - steric potential - now calculate it with electrostatic effects #actually you also need to add it here... 
                phiPE[:][idT], compressedEl = polyelectrolytePotential(allheights-h0b-h0t,lambdaV,elleff,Neff,sigmaMax,zp,c0salt,sigmaS,Rt,eVparameter,optcolloidcolloidFlag)
                    
                # 2 - d - depletion
                realseparation = allheights - h0b - h0t - 2*compressedEl 
                # maybe with a factor 3pi/16... depending on what counts for depletion... the total brush height or the average brush height... 
                #plt.plot(allheights,realseparation)
                if optdeplFlag:
                    phiDepl[:][idT] = micellePenetrationInsideBrush*DepletionPotential(T,cm0,realseparation,Rt,optcolloidcolloidFlag, aggRadius = aggRadiusC, depletionType = depletionTypeC) #+h0t+heights1t[-1]+heights2t[-1] you also increase the effective radius of the particle here.
                
                
    
                # 2 - a - again electrostatics - you still need the plates interacting
                phiEl[:][idT] = electrostaticPotentialPlates(allheights,h0b,h0t,lambdaV,c0salt,Rt,gamma2,gamma1,optcolloidcolloidFlag)
                
            
            
        else:
            
            if PEmodel == "Electrostatics": #then you need to follow a certain order to calculate different contributions
    
                # 2 - d - depletion
                realseparation = allheights - h0b - h0t -(heights1b[-1]+heights2b[-1]+heights1t[-1]+heights2t[-1]) 
                # maybe with a factor 3pi/16... depending on what counts for depletion... the total brush height or the average brush height... 
                #plt.plot(allheights,realseparation)
                if optdeplFlag:
                    print('Calculated depletion')
                    phiDepl[:][idT] = micellePenetrationInsideBrush*DepletionPotential(T,cm0,realseparation,Rt,optcolloidcolloidFlag, aggRadius = aggRadiusC, depletionType = depletionTypeC) #+h0t+heights1t[-1]+heights2t[-1] you also increase the effective radius of the particle here.
                                                                    # this has a very weak effect, so I would not account for it in an attempt to keep the model sane... 
                
                
                    
                # 2 - f - bridging potential 
                if bridging:
                    heightst = [heights1t[ih]+heights2t[ih] for ih in range(len(heights1t))]
                    heightsb = [heights1b[ih]+heights2b[ih] for ih in range(len(heights1b))]
                    #bindingEnergies = compute_DNAbindingEnergy(Sequence,saltConcentration)
                    phiBridge[:][idT], NBridge[:][idT], Allsigmaabs[:][idT], sigmasticky, AllKabs[:][idT], DeltaG0s[idT], allQ1s, allQ2s = \
                    bridgingPotential(allheights - h0b - h0t,bindingEnergies,hbond,h1eq,h2eq,sigmastickyb,sigmastickyt,Rt,Sequence,saltConcentration,T,sigmat,ellet,Nt,sigmab,elleb,Nb,heightst,heightsb,mushroomFlag, porosity,optcolloidcolloidFlag, \
                                      allKabsC, allQ1sC, allQ2sC)
                #print('Calculated bridging potential')
                #print(hbond,h1eq,h2eq)
                
                # 2 - e - steric potential # this one does not depend on temperature but it has to be calculated after
                if idT == 0:
                    hmaxSteric = h1eq+h2eq
                    print('Calculating steric repulsion once')
                    phiSter[:][idT] = StericPotentialFull(allheights-h0t-h0b,Rt,sigmat,ellet,Nt,eVt,sigmab,elleb,Nb,eVb,hmaxSteric,Emins,optcolloidcolloidFlag,mushroomFlag, allQ1s, allQ2s)
                else:
                    phiSter[:][idT] = phiSter[:][0]
                
                # 2 - a - again electrostatics
        
                phiEl[:][idT] = electrostaticPotential(allheights,h0b,h0t,heights1b,heights1t,heights2b,heights2t, \
                                                    lambdaV,c0salt,sigmaDNAb,sigmaDNAt,Rt,sigmaGlass,sigmaPS,gamma2,gamma1,Allsigmaabs[:][idT], sigmasticky, \
                                                    bottomParameters[3]*electron, topParameters[3]*electron, optcolloidcolloidFlag)
                
                
                # fig2, ax2 = plt.subplots(figsize=(6,4))
                # ax2.plot(allheights*1e9,phiBridge[:][idT],label='bridging')
                # ax2.plot(allheights*1e9,[-p for p in phiSter[:][idT]],label='repulsion')
                # #ax2.semilogx(allheights*1e9,referenceSter,'--',label='calculus')
                # ax2.set_xlim(50,100)
                # ax2.set_ylim(-100,0)
                # plt.show() # this plot shows great agreement for no charges.
                
            
                
                    
                    
            elif PEmodel == "Polyelectrolyte": #then you need to follow another order
        
                
                
                # -- first you need effective brush parameters ---
                elleff,eVparameter,Neff,sigmaMax,sigmaS,zp,hb,ht = calculateEffectiveSymmetricParam(Nb,Nt,elleb,ellet,eVb,eVt,sigmab,sigmat,chargeb,charget,sigmaGlass,sigmaPS,c0salt,lambdaV,eVparameter,accountForF127Stretch,slideType)
                
                
                # 2 - e - steric potential - now calculate it with electrostatic effects #actually you also need to add it here... 
                phiPE[:][idT], compressedEl = polyelectrolytePotential(allheights-h0b-h0t,lambdaV,elleff,Neff,sigmaMax,zp,c0salt,sigmaS,Rt,eVparameter,optcolloidcolloidFlag)
                
                
                #print(compressedEl,hb,ht)
                # 2 - d - depletion
                realseparation = allheights - h0b - h0t - 2*compressedEl 
                # maybe with a factor 3pi/16... depending on what counts for depletion... the total brush height or the average brush height... 
                #plt.plot(allheights,realseparation)
                if optdeplFlag:
                    phiDepl[:][idT] = micellePenetrationInsideBrush*DepletionPotential(T,cm0,realseparation,Rt,optcolloidcolloidFlag, aggRadius = aggRadiusC, depletionType = depletionTypeC) #+h0t+heights1t[-1]+heights2t[-1] you also increase the effective radius of the particle here.
                
                
                # 2 - f - bridging potential 
                if bridging:
                    heightst = [compressedEl[ih] for ih in range(len(compressedEl))] #[heights1t[ih]+heights2t[ih] for ih in range(len(heights1t))]
                    heightsb = [compressedEl[ih] for ih in range(len(compressedEl))] #[heights1b[ih]+heights2b[ih] for ih in range(len(heights1b))]
                    
                    
                    if bridgingOption == "symmetric":
                        sigmaSticky = sqrt(sigmastickyb*sigmastickyt)
                        phiBridge[:][idT], NBridge[:][idT], Allsigmaabs[:][idT], sigmasticky, AllKabs[:][idT], DeltaG0s[idT], allQ1s, allQ2s = \
                        bridgingPotential(allheights - h0b - h0t,bindingEnergies,hbond,hb/2+ht/2,hb/2+ht/2,sigmaSticky,sigmaSticky,Rt,Sequence,saltConcentration,T,sigmaMax,elleff,Neff,sigmaMax,elleff,Neff,heightst,heightsb,mushroomFlag, porosity,optcolloidcolloidFlag, \
                                          allKabsC, allQ1sC, allQ2sC)
                    else:
                        phiBridge[:][idT], NBridge[:][idT], Allsigmaabs[:][idT], sigmasticky, AllKabs[:][idT], DeltaG0s[idT], allQ1s, allQ2s = \
                        bridgingPotential(allheights - h0b - h0t,bindingEnergies,hbond,hb,ht,sigmastickyb,sigmastickyt,Rt,Sequence,saltConcentration,T,sigmat,ellet,Nt,sigmab,elleb,Nb,heightst,heightsb,mushroomFlag, porosity,optcolloidcolloidFlag, \
                                          allKabsC, allQ1sC, allQ2sC)
                
                fig2, ax2 = plt.subplots(figsize=(6,4))
                ax2.plot(allheights*1e9,phiBridge[:][idT],label='bridging')
                ax2.plot(allheights*1e9,[-p for p in phiPE[:][idT]],label='repulsion')
                #ax2.semilogx(allheights*1e9,referenceSter,'--',label='calculus')
                ax2.set_xlim(50,100)
                ax2.set_ylim(-30,10)
                plt.show() # this plot shows great agreement for no charges.
                
                
                #print('Calculated bridging potential')
                #print(hbond,hb,ht)
    
                # 2 - a - again electrostatics - you still need the plates interacting
                phiEl[:][idT] = electrostaticPotentialPlates(allheights,h0b,h0t,lambdaV,c0salt,Rt,gamma2,gamma1,optcolloidcolloidFlag)
                
            elif PEmodel == "Unified": #then you need to follow a certain order to calculate different contributions
                # for now the new "unified" model is not able to account for different fractions of sticky ends. Please note that it takes in 
                # as argument already the fraction of sticky ends. This should be modified. 
                # the entropic contribution associated with that / and also asymmetric contributions / has to be formulated
                
                #bindingEnergies = compute_DNAbindingEnergy(Sequence,saltConcentration)
                # 2 - d - depletion
                realseparation = allheights - h0b - h0t -(heights1b[-1]+heights2b[-1]+heights1t[-1]+heights2t[-1]) 
                # maybe with a factor 3pi/16... depending on what counts for depletion... the total brush height or the average brush height... 
                #plt.plot(allheights,realseparation)
                if optdeplFlag:
                    phiDepl[:][idT] = micellePenetrationInsideBrush*DepletionPotential(T,cm0,realseparation,Rt,optcolloidcolloidFlag, aggRadius = aggRadiusC, depletionType = depletionTypeC) #+h0t+heights1t[-1]+heights2t[-1] you also increase the effective radius of the particle here.
                                                                    # this has a very weak effect, so I would not account for it in an attempt to keep the model sane... 
                
   
                
                # 2 - e - steric potential # this one does not depend on temperature but it has to be calculated after
                
                heightst = [heights1t[ih]+heights2t[ih] for ih in range(len(heights1t))]
                heightsb = [heights1b[ih]+heights2b[ih] for ih in range(len(heights1b))]
                
                modelUnified = 'minimizeForCompression'
                
                if modelUnified == 'accountForBindingCompressionOnly': 
                
                    phiBridge[:][idT], NBridge[:][idT], Allsigmaabs[:][idT], sigmasticky, AllKabs[:][idT], DeltaG0s[idT], allQ1s, allQ2s = \
                    bridgingPotential(allheights - h0b - h0t,bindingEnergies,hbond,h1eq,h2eq,sigmastickyb,sigmastickyt,Rt,Sequence,saltConcentration,T,sigmat,ellet,Nt,sigmab,elleb,Nb,heightst,heightsb,mushroomFlag, porosity,optcolloidcolloidFlag)
                    fractionfed = [sab for sab in Allsigmaabs[:][idT]]
                    #print(fractionfed)
                    #fractionfed = []
                    phiBridge[:][idT],phiSter[:][idT],NBridge[:][idT],Allsigmaabs[:][idT],sigmasticky,DeltaG0s[idT] = \
                     unifiedPotentials(allheights-h0t-h0b,h1eq,h2eq,sigmastickyb,sigmastickyt,Rt,Sequence,saltConcentration,T,sigmat,ellet,Nt,eVt,sigmab,elleb,Nb,eVb,bridging,optcolloidcolloidFlag,fractionfed)
    
                    phiBridge[:][idT], NBridge[:][idT], Allsigmaabs[:][idT], sigmasticky, AllKabs[:][idT], DeltaG0s[idT], allQ1s, allQ2s = \
                    bridgingPotential(allheights - h0b - h0t,bindingEnergies,hbond,h1eq,h2eq,sigmastickyb,sigmastickyt,Rt,Sequence,saltConcentration,T,sigmat,ellet,Nt,sigmab,elleb,Nb,heightst,heightsb,mushroomFlag, porosity,optcolloidcolloidFlag, \
                                      allKabsC, allQ1sC, allQ2sC)
                    
                elif modelUnified == 'minimizeForCompression': 
                
                    
                    if bridging:
                        phiBridgeBare, NBridge[:][idT], AllsigmaabsBare, sigmasticky, AllKabs[:][idT], DeltaG0s[idT], allQ1s, allQ2s = \
                        bridgingPotential(allheights - h0b - h0t,bindingEnergies,hbond,h1eq,h2eq,sigmastickyb,sigmastickyt,Rt,Sequence,saltConcentration,T,sigmat,ellet,Nt,sigmab,elleb,Nb,heightst,heightsb,mushroomFlag, porosity,optcolloidcolloidFlag, \
                                          allKabsC, allQ1sC, allQ2sC)
                        fractionfed = [kab for kab in AllKabs[:][idT]]
                    # uncomment for checking purposes only
                    #fractionfed = [0 for kab in AllKabs[:][idT]]
                    
                    # fig, ax = plt.subplots(figsize=[2,2])
    
                    # ax.plot(allheights,fractionfed)
                    # ax.set_ylim(-0.1,1)
                    # ax.set_xlim(30e-9,80e-9) 
                    # plt.show()
                    else:
                        fractionfed = [0 for h in allheights]
                    
                    # print(fractionfed)
                    heighttable = [h-h0t-h0b for h in allheights]
                    #fractionfed = []
                    phiBridge[:][idT],phiSter[:][idT],NBridge[:][idT],Allsigmaabs[:][idT],sigmasticky,DeltaG0s[idT] = \
                     unifiedPotentials(heighttable,h1eq,h2eq,sigmastickyb,sigmastickyt,Rt,Sequence,saltConcentration,T,sigmat,ellet,Nt,eVt,sigmab,elleb,Nb,eVb,bridging,optcolloidcolloidFlag,fractionfed)
    
                    # fig, ax = plt.subplots(figsize=[2,2])
    
                    # ax.plot(allheights,AllsigmaabsBare)
                    # ax.plot(allheights,Allsigmaabs[:][idT])
                    # ax.set_ylim(-0.1,1)
                    # ax.set_xlim(30e-9,80e-9) 
                    # plt.show()
                    
                    # this graph allowed to show that at low binding fractions the Daan Frenkel model and the Unified model are equivalent, 
                    # provided the right interaction energy is chosen... 
                    
                    # # checking that the steric repulsion corresponds
                    # hmaxSteric = h1eq+h2eq
                    # phiSterCheck = StericPotentialFull(allheights-h0t-h0b,Rt,sigmat,ellet,Nt,eVt,sigmab,elleb,Nb,eVb,hmaxSteric,Emins,optcolloidcolloidFlag,mushroomFlag, allQ1s, allQ2s)

                    # fig, ax = plt.subplots(figsize=[2,2])
    
                    # ax.plot(allheights,phiSterCheck)
                    # ax.plot(allheights,phiSter[:][idT])
                    # ax.set_ylim(-1,6)
                    # ax.set_xlim(40e-9,70e-9) 
                    # plt.show()
                    # # unified was checked in the absence of binding energy and agrees well with the steric part 


                    
    
                elif modelUnified == 'minimizeForCompressionSimpleInt': 
                
                    
                    fractionfed = []
                    phiBridge[:][idT],phiSter[:][idT],NBridge[:][idT],Allsigmaabs[:][idT],sigmasticky,DeltaG0s[idT] = \
                     unifiedPotentials(allheights-h0t-h0b,h1eq,h2eq,sigmastickyb,sigmastickyt,Rt,Sequence,saltConcentration,T,sigmat,ellet,Nt,eVt,sigmab,elleb,Nb,eVb,bridging,optcolloidcolloidFlag,fractionfed)
    
                        
                
                AllKabs[:][idT] = 1.0 # assign arbitrary value here - DeltaGeff won't mean anything here and is actually hard to calculate
                
                print('Calculated bridging potential')
                print(h1eq,h2eq)
                
                # 2 - a - again electrostatics
        
                phiEl[:][idT] = electrostaticPotential(allheights,h0b,h0t,heights1b,heights1t,heights2b,heights2t, \
                                                     lambdaV,c0salt,sigmaDNAb,sigmaDNAt,Rt,sigmaGlass,sigmaPS,gamma2,gamma1,Allsigmaabs[:][idT], sigmasticky, \
                                                     bottomParameters[3]*electron, topParameters[3]*electron, optcolloidcolloidFlag)
                         

            
                
                # plt.loglog(allheights*1e9,-phiBridge[:][idT],'x')    
                # plt.loglog(allheights*1e9,phiSter[:][idT],'o')
                # plt.loglog(allheights*1e9,-phiVdW[:][idT],'k')
                # #plt.xlim((h1eq+h2eq)*0.9*1e9,(h1eq+h2eq)*1.1*1e9)
                # plt.show()
                # plt.pause(1)
                    
        phikT[:][idT] = phiEl[:][idT] + phiGrav[:][idT] + phiDepl[:][idT] + phiVdW[:][idT] + phiSter[:][idT] + phiBridge[:][idT] + phiPE[:][idT]
        idT += 1

    ########## 3: MICROSCOPIC PROPERTIES ###################
    # now find microscopic properties
    # for these its fine to stick to the radius at room temperature
    #print(sigmastickyb,sigmastickyt,Rt0)
    areaFactor = min(sigmastickyb,sigmastickyt)*pi*Rt0**2
    hMinsT,hAvesT,nConnectedT,areaT,depthT,widthT,xvalues,svalues,sticky,punbound,deltaGeff,Rconnected, NAves, nInvolvedT\
        = calculateMicroscopicDetails(allheights,phikT,phiBridge,NBridge,Allsigmaabs,AllKabs,0.9,Rt0,lent,criticalHeight,optcolloidcolloidFlag,gravityFactors,areaFactor)
    # This was an olde estimate of the number of involved bonds
    #hAvesT = Lvalues
    # indT = 0
    # for rC in Rconnected:
    #     if rC < 10:
    #         nInvolvedT[indT] = rC**2*min(sigmastickyb*sigmastickyt)*pi*Rt0**2
    #         indT +=1
    #     else: 
    #         nInvolvedT[indT] = float("NaN")
    
    ########## 4: SAVE THE DATA ###################
    my_save= {"allheights":allheights, "phikT":phikT, "lambdaV":lambdaV, "phiEl":phiEl, "phiGrav":phiGrav, \
              "phiDepl": phiDepl, "phiVdW":phiVdW, "phiSter":phiSter, "phiBridge":phiBridge, \
           "phiPE":phiPE, "hMinsT":hMinsT,"hAvesT":hAvesT,"nConnectedT":nConnectedT,"areaT":areaT, \
           "depthT":depthT, "widthT":widthT, "xvalues":xvalues, "svalues":svalues, "sticky":sticky, \
           "punbound":punbound,"deltaGeff":deltaGeff,"DeltaG0s":DeltaG0s,"Rconnected":Rconnected, \
           "nInvolvedT":nInvolvedT, "NAves":NAves, "gravityFactors": gravityFactors, "Tin":Tin, "NBridge":NBridge}
    with open(basename+'.pickle', 'wb') as f:
        pickle.dump(my_save, f)
    #with open(basename+'.pickle', 'rb') as g:
    #    loaded_obj = pickle.load(g)

    return(allheights,phikT,lambdaV, phiEl, phiGrav, phiDepl, phiVdW, phiSter, phiBridge, \
           phiPE, hMinsT,hAvesT,nConnectedT,areaT,depthT,widthT,xvalues,svalues,sticky,punbound, \
           deltaGeff,DeltaG0s,Rconnected,nInvolvedT,NAves,gravityFactors)

def load_potential_profile(basename):
    with open(basename+'.pickle', 'rb') as g:
        loaded_obj = pickle.load(g)


    allheights = loaded_obj["allheights"]
    phikT = loaded_obj["phikT"]
    lambdaV = loaded_obj["lambdaV"]
    phiEl = loaded_obj["phiEl"]
    phiGrav = loaded_obj["phiGrav"]
    phiDepl = loaded_obj["phiDepl"]
    phiVdW = loaded_obj["phiVdW"]
    phiSter = loaded_obj["phiSter"]
    phiBridge = loaded_obj["phiBridge"]
    phiPE = loaded_obj["phiPE"]
    hMinsT = loaded_obj["hMinsT"]
    hAvesT = loaded_obj["hAvesT"]
    nConnectedT = loaded_obj["nConnectedT"]
    areaT = loaded_obj["areaT"]
    depthT = loaded_obj["depthT"]
    widthT = loaded_obj["widthT"]
    xvalues = loaded_obj["xvalues"]
    svalues = loaded_obj["svalues"]
    sticky = loaded_obj["sticky"]
    punbound = loaded_obj["punbound"]
    deltaGeff = loaded_obj["deltaGeff"]
    DeltaG0s = loaded_obj["DeltaG0s"]
    Rconnected = loaded_obj["Rconnected"]
    nInvolvedT = loaded_obj["nInvolvedT"]
    NAves = loaded_obj["NAves"]

    # my_save= {"allheights":allheights, "phikT":phikT, "lambdaV":lambdaV, "phiEl":phiEl, "phiGrav":phiGrav, \
    #           "phiDepl": phiDepl, "phiVdW":phiVdW, "phiSter":phiSter, "phiBridge":phiBridge, \
    #        "phiPE":phiPE, "hMinsT":hMinsT,"hAvesT":hAvesT,"nConnectedT":nConnectedT,"areaT":areaT, \
    #        "depthT":depthT, "widthT":widthT, "xvalues":xvalues, "svalues":svalues, "sticky":sticky, \
    #        "punbound":punbound,"deltaGeff":deltaGeff,"DeltaG0s":DeltaG0s,"Rconnected":Rconnected, \
    #        "nInvolvedT":nInvolvedT, "NAves":NAves}
    

        
    return(allheights,phikT,lambdaV, phiEl, phiGrav, phiDepl, phiVdW, phiSter, phiBridge, \
           phiPE, hMinsT,hAvesT,nConnectedT,areaT,depthT,widthT,xvalues,svalues,sticky,punbound, \
           deltaGeff,DeltaG0s,Rconnected,nInvolvedT,NAves)










### UNMAINTAINED CODES.... 


def compute_potential_profile_steric(allheights, radius, PSdensity, gravity, saltConcentration, \
                              Tin, PSCharge, GlassCharge, cF127,densityTether, \
                              ellPEO,NPEO,bottomCoating):
    
    # Global physical constants
    electron = 1.6*10**-19 #elementary electron charge
    Na = 6.02*10**23
    Rg = 8.314 #J/(mol.K); #perfect gas constant
    Tbase = 273.15
    kB = Rg/Na #boltzmann constant
    
    ########## TRANSLATE ALL INPUT PARAMETERS INTO USI ###################
    R = radius*10**-6 #radius of the colloids (m)
    c0salt = (saltConcentration*10**3)*Na #salt concentration in parts/m^3 (the first number is in mol/L)
    rhoPS = PSdensity*10**-3/(1*10**-6) #density of PS in kg/m3 (tabulated)
    g = gravity #gravity in NYC
    sigmaPS = PSCharge# surface charge in C/m2 of polystyrene
    sigmaGlass = GlassCharge # surface charge in C/m2 of glass
    sigma = densityTether/(10**-9)**2 # surface coverage 
    ell = ellPEO*10**(-9) # length of PEO strands
    N = NPEO
    cm0 = cF127*1e-2*1e3/12600 #convert the F127 concentration in mol/L
    bottomH = bottomCoating*10**(-9) #thickness of the bottom coating
    
    def meff(Temp):
        return(4/3*pi*R**3*(rhoPS - density(Temp))) #effective mass of the colloid
        
    
    phikT = 0*allheights
    phiEl = 0*allheights
    phiDepl = 0*allheights
    phiGrav = 0*allheights
    phiVdW = 0*allheights
    phiSter = 0*allheights
    
    T = Tbase + Tin
    
    epsilon0r =  permittivity(T,density(T))
    lambdaV = lambdaD(T,c0salt) 
    gamma1 = Graham(T,lambdaV,sigmaPS)
    gamma2 = Graham(T,lambdaV,sigmaGlass)    
    mass = meff(T)
    
    #print(lambdaV)
    #print(mass)
    
    for ip in range(len(allheights)):
        h = allheights[ip]
        # electrostatic repulsion - plate/plate distance counts
        phiEl[ip] = 64*pi*R*kB*T*epsilon0r/electron**2*gamma1*gamma2*exp(-h/lambdaV) 
        # gravity - plate/plate distance counts
        phiGrav[ip] = mass*g*h/(kB*T)
        # depletion - coating/coating distance counts
        heq = hMilner(sigma,ell,N)
        #print(heq)
        phiDepl[ip] = DepletionPotential(T,cm0,h-heq-bottomH,R)
        # van der waals - plate/plate distance counts
        if log(h) < - 5.5*log(10):
            if log(h) > -10*log(10):
                phiVdW[ip] = VanDerWaalsPotential(Tin,saltConcentration,h,R) # in here the concentration goes in mol/L
            else:
                phiVdW[ip] = 0
        else:
            phiVdW[ip] = 0
        # steric interactions with the surface - plate/static counting counts
        if h > bottomH:
            phiSter[ip] = StericPotentialMilner(h-bottomH,R,sigma,ell,N)
        else:
            phiSter[ip] = inf
            
        phikT[ip] = phiEl[ip] + phiGrav[ip] + phiDepl[ip] + phiVdW[ip] + phiSter[ip]
    
    
    
    return(phikT,lambdaV, phiEl, phiGrav, phiDepl, phiVdW, phiSter)


def compute_potential_profile_VdW(allheights, radius, PSdensity, gravity, saltConcentration, Tin, PSCharge, GlassCharge, cF127):
    
    # Global physical constants
    electron = 1.6*10**-19 #elementary electron charge
    Na = 6.02*10**23
    Rg = 8.314 #J/(mol.K); #perfect gas constant
    Tbase = 273.15
    kB = Rg/Na #boltzmann constant
    
    ########## TRANSLATE ALL INPUT PARAMETERS INTO USI ###################
    R = radius*10**-6 #radius of the colloids (m)
    c0salt = (saltConcentration*10**3)*Na #salt concentration in parts/m^3 (the first number is in mol/L)
    rhoPS = PSdensity*10**-3/(1*10**-6) #density of PS in kg/m3 (tabulated)
    g = gravity #gravity in NYC
    sigmaPS = PSCharge# surface charge in C/m2 of polystyrene
    sigmaGlass = GlassCharge # surface charge in C/m2 of glass
    cm0 = cF127*1e-2*1e3/12600 #convert the F127 concentration in mol/L
    
    
    def meff(Temp):
        return(4/3*pi*R**3*(rhoPS - density(Temp))) #effective mass of the colloid
        
    
    phikT = 0*allheights
    phiEl = 0*allheights
    phiDepl = 0*allheights
    phiGrav = 0*allheights
    phiVdW = 0*allheights
    
    
    T = Tbase + Tin
    
    epsilon0r =  permittivity(T,density(T))
    lambdaV = lambdaD(T,c0salt) 
    gamma1 = Graham(T,lambdaV,sigmaPS)
    gamma2 = Graham(T,lambdaV,sigmaGlass)    
    mass = meff(T)
    
    #print(lambdaV)
    #print(mass)
    
    for ip in range(len(allheights)):
        h = allheights[ip]
        # electrostatic repulsion
        phiEl[ip] = 64*pi*R*kB*T*epsilon0r/electron**2*gamma1*gamma2*exp(-h/lambdaV) 
        # gravity
        phiGrav[ip] = mass*g*h/(kB*T)
        # depletion
        phiDepl[ip] = DepletionPotential(T,cm0,h,R)
        # van der waals
        if log(h) < - 6.5*log(10):
            if log(h) > -10*log(10):
                phiVdW[ip] = VanDerWaalsPotential(Tin,saltConcentration,h,R)
            else:
                phiVdW[ip] = 0
        else:
            phiVdW[ip] = 0
    
        phikT[ip] = phiEl[ip] + phiGrav[ip] + phiDepl[ip] + phiVdW[ip]
        
    return(phikT,lambdaV, phiEl, phiGrav, phiDepl, phiVdW)


def compute_potential_profile_depletion(allheights, radius, PSdensity, gravity, saltConcentration, Tin, PSCharge, GlassCharge, cF127):
    
    # Global physical constants
    electron = 1.6*10**-19 #elementary electron charge
    Na = 6.02*10**23
    Rg = 8.314 #J/(mol.K); #perfect gas constant
    Tbase = 273.15
    kB = Rg/Na #boltzmann constant
    
    ########## TRANSLATE ALL INPUT PARAMETERS INTO USI ###################
    R = radius*10**-6 #radius of the colloids (m)
    c0salt = (saltConcentration*10**3)*Na #salt concentration in parts/m^3 (the first number is in mol/L)
    rhoPS = PSdensity*10**-3/(1*10**-6) #density of PS in kg/m3 (tabulated)
    g = gravity #gravity in NYC
    sigmaPS = PSCharge# surface charge in C/m2 of polystyrene
    sigmaGlass = GlassCharge # surface charge in C/m2 of glass
    cm0 = cF127*1e-2*1e3/12600 #convert the F127 concentration in mol/L
    
    
    def meff(Temp):
        return(4/3*pi*R**3*(rhoPS - density(Temp))) #effective mass of the colloid
        
    
    phikT = 0*allheights
    
    T = Tbase + Tin
    
    epsilon0r =  permittivity(T,density(T))
    lambdaV = lambdaD(T,c0salt) 
    gamma1 = Graham(T,lambdaV,sigmaPS)
    gamma2 = Graham(T,lambdaV,sigmaGlass)    
    mass = meff(T)
    
    #print(lambdaV)
    #print(mass)
    
    for ip in range(len(allheights)):
        h = allheights[ip]
        phiNaked = 64*pi*R*kB*T*epsilon0r/electron**2*gamma1*gamma2*exp(-h/lambdaV) + mass*g*h/(kB*T)
        #phiNakedB = 16.5*exp(-h/lambdaV) + mass*g*h/(kB*T) # data for Bechinger take 22 C and 0.0009 in concentration to get the right lambda = 33nm
        
        phiDepl = DepletionPotential(T,cm0,h,R)
        phikT[ip] = phiNaked + phiDepl
        
    return(phikT,lambdaV)



def compute_potential_profile_naked(allheights, radius, PSdensity, gravity, saltConcentration, Tin, PSCharge, GlassCharge):
    
    # Global physical constants
    electron = 1.6*10**-19 #elementary electron charge
    Na = 6.02*10**23
    Rg = 8.314 #J/(mol.K); #perfect gas constant
    Tbase = 273.15
    kB = Rg/Na; #boltzmann constant
    
    ########## TRANSLATE ALL INPUT PARAMETERS INTO USI ###################
    R = radius*10**-6; #radius of the colloids (m)
    c0salt = (saltConcentration*10**3)*Na; #salt concentration in parts/m^3 (the first number is in mol/L)
    rhoPS = PSdensity*10**-3/(1*10**-6); #density of PS in kg/m3 (tabulated)
    g = gravity; #gravity in NYC
    sigmaPS = PSCharge;# surface charge in C/m2 of polystyrene
    sigmaGlass = GlassCharge; # surface charge in C/m2 of glass
    
    
    def meff(Temp):
        return(4/3*pi*R**3*(rhoPS - density(Temp))) #effective mass of the colloid
        
    
    phikT = 0*allheights
    T = Tbase + Tin
    
    epsilon0r =  permittivity(T,density(T))
    lambdaV = lambdaD(T,c0salt) 
    gamma1 = Graham(T,lambdaV,sigmaPS)
    gamma2 = Graham(T,lambdaV,sigmaGlass)    
    mass = meff(T)
    
    #print(lambdaV)
    #print(mass)
    
    for ip in range(len(allheights)):
        h = allheights[ip]
        phi = 64*pi*R*kB*T*epsilon0r/electron**2*gamma1*gamma2*exp(-h/lambdaV) + mass*g*h/(kB*T)
        phikT[ip] = phi
        
    return(phikT,lambdaV)
