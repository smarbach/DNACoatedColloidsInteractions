#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:18:52 2022

@author: sm8857
"""


import numpy as np
import pickle
from scipy import special


from WaterDensityModule import viscosityWater, ONeillFrictionTangential
from StericModule import calculateCompressedHeights, determineRelevantHeights
from MicroscopicModule import computeStickyParameters
from scipy import interpolate
from scipy.interpolate import interp1d

    
from math import sqrt
import math
pi = math.pi


def compute_diffusion_coefficients(radius, Kon, saltConcentration, Tin, \
                                cF127, topParameters, bottomParameters, ellDNAnotused, ellPEO, wPEO, slideType, \
                                srcpath, nresolution, eVparameter, PEmodel, DNAmodel, \
                                slabH, basename, kineticPerspective, \
                                criticalHeight,  mushroomFlag, \
                                nInvolvedT, NAves, potential, phiGrav, phiVdW, phiSter, punbound, \
                                optionInertia, mass, gravityFactors):
    
    
    print(Tin)
    # Global physical constants
    Na = 6.02*10**23
    Rg = 8.314 #J/(mol.K); #perfect gas constant
    Tbase = 273.15
    kB = Rg/Na #boltzmann constant
    
    ########## TRANSLATE ALL INPUT PARAMETERS INTO USI ###################
    Rt0 = radius*10**-6 #radius of the colloids (m)
    slabHeight = slabH*1e-6 #was given in um

    h0b = bottomParameters[0]*10**(-9) #thickness of the bottom coating
    h0t = topParameters[0]*10**(-9) #thickness of the top coating
    
    N1b = bottomParameters[1] #number of PEO units
    N1t = topParameters[1] #number of PEO units

    sigmab = bottomParameters[4]/(10**-9)**2 # surface coverage 
    sigmat = topParameters[4]/(10**-9)**2 # surface coverage 

    fstickyb = bottomParameters[5]
    fstickyt = topParameters[5]

    #sigmaDNAb = sigmab*(bottomParameters[3]*electron) # electric surface coverage  in C/m^2
    #sigmaDNAt = sigmat*(topParameters[3]*electron) # electric surface coverage in C/m^2
    
    #chargeb = bottomParameters[3]
    #charget = topParameters[3]
   

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
    
    ell2 = ellDNA*10**(-9) # lpersistence ength of DNA strands

    
    ########## 1: HEIGHTS OF COATINGS ###################
    allheights = determineRelevantHeights(N1b,N2b,N2t,N1t,ell1,ell2,eVfac1,eVfac2,sigmab,sigmat,h0b+h0t,nresolution,slabHeight,PEmodel,mushroomFlag) #independent of temperature
    
   


    #if PEmodel == "Electrostatics": 
        #then find the heights as they are compressed
    if cF127 > 0:
        accountForF127Stretch = True
    else:
        accountForF127Stretch = False
        
    Emins,Ebridgeonly,heights1b,heights2b,heights2t,heights1t,elleb,ellet,eVb,eVt,Nb,Nt,hrest,hrestt,sigmab,sigmat,lbot,ltop = \
            calculateCompressedHeights(N1b,N2b,N2t,N1t,ell1,ell2,eVfac1,eVfac2,sigmab,sigmat,allheights-h0b-h0t, \
                                      0,accountForF127Stretch,mushroomFlag,slideType)  
    
    ########## 2: Start Calculating diffusion coefficients ###################


    # find the melting point
    pmax = punbound[-1]
    Tm = 0
    for iT in range(len(Tin)):
        if Tm == 0 and punbound[iT] >= pmax/2:
            Tm = Tin[iT]/2 + Tin[iT-1]/2
    
    print('The melting T was',Tm)

    #Gamma0 = 6*pi*Rt0*viscosityWater(Tbase + Tplot)*2 #for hydro friction near the boundary (increased by a factor 2 naturally)
    
    #sigma = 1/(20e-18)
    L = N1b*ellPEO*1e-9+N2b*ellDNA*1e-9 #this will kind of assume top and bottom are symmetric
    lp = (N1b*ellPEO + N2b*ellDNA)/(N1b+N2b)
    
    if kineticPerspective == 'top':
         ctot = fstickyt*sigmat*1/(ltop)
    elif kineticPerspective == 'bottom':
         ctot = fstickyb*sigmab*1/(lbot)
    elif kineticPerspective == 'centered':
         ctot = sqrt(fstickyb*sigmab*1/(lbot)*fstickyt*sigmat*1/(ltop))
    
    
    qon = Kon*1/(1000*Na)*ctot # (from website * concentratino)
    
            
    Dslide = np.zeros(len(Tin))
    lbrush = (ltop+lbot)/2
    Dhop = np.zeros(len(Tin))
    Deff = np.zeros(len(Tin))
    
    Reff = 1 # I can't remember what this parameter is supposed to represent, maybe some effective geometrical modification

    muHydro = ONeillFrictionTangential(2*lbrush/Rt0) #for hydro friction near the boundary 
    print('Relative distance from surface',2*lbrush/Rt0)
    print('Hydrodynamic increase due to friction is ', muHydro)

    #print(Tin)
    for iT in range(len(Tin)):
        print(Tin[iT]-Tm)
        Tplot = Tin[iT]
        Gamma0 = 6*pi*Rt0*viscosityWater(Tbase + Tplot)*muHydro
        Gamma = Gamma0
 #       betae0 = deltaGeff[i] # there's more than 1 way to estimate that actually 
        if nInvolvedT[iT] <= NAves[iT]:
            Dhop[iT] = Dhop[-1]
            Dslide[iT] =  Dslide[-1]
            Deff[iT] = Deff[-1]
        else:
            
            onOverOff = NAves[iT]/(nInvolvedT[iT]-NAves[iT])
            lowestpotentialindex = np.array(potential[:][iT]).argmin()
            # figure out "average" location of the particle - considered as the location of the minimum... 
            hMins = allheights[lowestpotentialindex]
            # figure out "average" location of the particle - considered as the weighted average with the potential
            #phi = potential[:][iT]
            phi = phiGrav[:][iT] +  phiVdW[:][iT] + phiSter[:][iT] # this makes more sense because it really judges only the non-interacting potential part
                # which is the one that counts to determine the big Qon/Qoff for 3D immersion of the problem. 
            #62459 #64284 # gravity Factors
            #sticky,punboundForDiffusion = computeStickyParameters(allheights,phi,20e-9,lowestpotentialindex,1/gravityFactors[i]*1e9)
            lowestpotentialindexforMin = np.array(phi[:]).argmin()
            if gravityFactors[iT] == 0:
                punboundForDiffusion= 0
            else:
                sticky,punboundForDiffusion = computeStickyParameters(allheights,phi,20e-9,lowestpotentialindexforMin,1/gravityFactors[iT]*1e9)
            
            
            
            pbclose = 1 - punboundForDiffusion
            Nr = NAves[iT]/(nInvolvedT[iT])
            #def fsolveOnOff(xx):
            #    return(Nr/pbclose - xx*(1+xx)**(nInvolvedT[i]-1)/((1+xx)**nInvolvedT[i] - 1))
            #onOverOffacc = fsolve(fsolveOnOff,onOverOff)
            onOverOffacc = onOverOff #Nr/(pbclose-Nr) #this is fine since all this is already in the binding range
            #print(Tplot, Nr/pbclose,onOverOffacc,onOverOff)
            
            xacc = pbclose/((1-pbclose)*(1+onOverOffacc)**nInvolvedT[iT])
            if punboundForDiffusion < 1:
                x = punboundForDiffusion/(1-punboundForDiffusion)
                
                xest = pbclose/((1-pbclose)*(1+onOverOff)**nInvolvedT[iT])   #Off / On (big rates) #if you don't want this simply x = 0
                xest2 = (punboundForDiffusion*(1+onOverOff)**nInvolvedT[iT])/(1-punboundForDiffusion)
                #print('Off/(Off+On)',punboundForDiffusion, x, xest2, 1/xest, 'pbclose',pbclose,  1/xacc)
                #x = xest2
                qoff = qon/onOverOff
                Nbound = NAves[iT]*Reff
                k = 3*kB*(Tbase + Tplot)/(2*L*lp*1e-9) #factor 2 because they are facing each other
                gamma = 6*pi*viscosityWater(Tbase + Tplot)*lbrush #factor 2 because they are facing each other
                gammap =  Nbound*( gamma + k*(1/qoff)) # + gamma/k*(exp(-betae0)) ) )
                gammap2 = Nbound*( gamma + k*(1/qoff)) # + gamma/k*onOverOff*(nInvolvedT[i]-Nbound) ) ) # but actually this 2nd part is often quite negligible... 
                DeffT = min(kB*(Tbase + Tplot)/(Gamma+gammap)*1e12,kB*(Tbase + Tplot)/(Gamma)*1e12)
                
                
                Nin = nInvolvedT[iT]*Reff
                Nbo = NAves[iT]*Reff
                
                
                # now finally calculate the friction
                
                
                Gammaeff,p0,Gammab = interpolateNGamma(Nin,Nbo,qoff*Gamma/k,qon*Gamma/k,gamma/Gamma,mass*k/Gamma**2,optionInertia,Gamma)
               
                
                Dhop[iT] = (p0*kB*(Tbase + Tplot)/(Gamma)*1e12*(1+x)/(1+x*p0))
                Dslide[iT] = (DeffT/(1+x*p0))
                
            
                #print('T', Tplot,'qon',onOverOff*qoff,'qoff',qoff,'k',k,'Nmax',int(nInvolvedT[iT]),'gamma',gamma,'Gamma',Gamma)
                print('gR',gamma/Gamma,'qon',onOverOff*qoff*gamma/k,'off',qoff*gamma/k,'timescale',gamma/k,'Nmax',int(nInvolvedT[iT]))
                print('min time', np.min([1/(onOverOff*qoff*gamma/k), Gamma/gamma, 1/(qoff*gamma/k)]))
                
                Deff[iT] = (((Gammaeff - p0)/(1+x*p0) + p0*(1+x)/(1+x*p0))*kB*(Tbase + Tplot)/(Gamma)*1e12)
            
            else:   
                Dhop[iT] = kB*(Tbase + Tplot)/(Gamma)*1e12
                Dslide[iT] = 0
                Deff[iT] = kB*(Tbase + Tplot)/(Gamma)*1e12
            
            #print(Deff)
            #print(Gamma/(Gamma+gammap2))
            
            
    ########## 3: SAVE THE DATA ###################
    my_save= {"Tin":Tin, "Dhop":Dhop, "Tm":Tm, "Dslide":Dslide, "Deff":Deff}
    with open(basename+'_DiffusionProperties.pickle', 'wb') as f:
        pickle.dump(my_save, f)
    #with open(basename+'.pickle', 'rb') as g:
    #    loaded_obj = pickle.load(g)

    
    
    return(Tm, Dhop, Dslide, Deff)
    
def calculateProba(N,n,x):
    return(special.binom(N,n)*x**n*(1-x)**(N-n))


def calculateGammaFull(N,Nb,qoff,qon,gR):
    
    ubGmat = np.zeros([3*N,3*N])
    vec = np.zeros(3*N)
    for i in range(N):
       # it's ubG but starting u0 b1 G1 etc..
       # u line
       k = (i-1)
       ubGmat[(i-1)*3][(i-1)*3] = - k*qoff - (N-k)*qon - 1/gR; #uk
       if i > 0:
           ubGmat[(i-1)*3][(i-2)*3] = k*qoff; #uk-1
       ubGmat[(i-1)*3][(i-1)*3+1] = qon; #bk+1
       if i < N-1:
           ubGmat[(i-1)*3][(i)*3] = (N-k-1)*qon;
       
       # b line
       ubGmat[(i-1)*3+1][(i-1)*3+1] = - i*qoff - (N-i)*qon; #bk
       if i > 0:
           ubGmat[(i-1)*3+1][(i-2)*3+1] = (i-1)*qoff; #bk-1
       ubGmat[(i-1)*3+1][(i-1)*3] = qoff; #uk-1
       if i < N-1:
           ubGmat[(i-1)*3+1][(i)*3+1] = (N-i)*qon;
       ubGmat[(i-1)*3+1][(i-1)*3+2] = -1;
       # G line
       vec[(i-1)*3 + 2] = -1;
       ubGmat[(i-1)*3 + 2][(i-1)*3 + 2] = - (1 + i*gR); #uk-1
       ubGmat[(i-1)*3 + 2][(i-1)*3 + 1] = i; #uk-1
    
    
    Result = np.linalg.solve(ubGmat,vec)
    Gis = np.zeros(N)
    pis = np.zeros(N)
    x = qon/(qoff+qon)
    p0 = calculateProba(N,0,x)
    Geffm1 = p0 

    for i in range(N):
        Gis[i-1] = Result[(i-1)*3 + 2]
        pis[i-1] = calculateProba(N,i+1,x)
        Geffm1 += Gis[i-1]*pis[i-1]
    if Nb < N:   
        Nbi = int(Nb)
        y = Nb - Nbi
        Gnb = (1-y)*Gis[Nbi-1] + y*Gis[Nbi]
    else:
        Gnb = 0
    
    # print(pis)
    # print(Gis)
    
    return(Geffm1,Gnb,p0)


def calculateGammaFullInertia(N,Nb,qoff,qon,gR,m):
    
    ubGmat = np.zeros([3*N+1,3*N+1])
    vec = np.zeros(3*N+1)
    vec[0] = -1
    ubGmat[0][0] = -1 + m*(-N*qon)
    ubGmat[0][3] = m*N*qon
    
    #print(N)
    for i in range(N):
       # it's ubG but starting u0 b1 G1 etc..
       # u line
       k = (i-1)
       icode = i+1
       
       ubGmat[(icode-1)*3 + 1][(icode-1)*3 + 1] = - k*qoff - (N-k)*qon - 1/gR; #uk
       if icode > 1:
           ubGmat[(icode-1)*3 +1][(icode-2)*3+1] = k*qoff; #uk-1
       ubGmat[(icode-1)*3+1][(icode -1)*3+2] = qon; #bk+1
       if icode < N:
           ubGmat[(icode-1)*3+1][(icode)*3+1] = (N-k-1)*qon;
       
       # b line
       ubGmat[(icode-1)*3+2][(icode-1)*3+2] = - icode*qoff - (N-icode)*qon; #bk
       if icode > 1:
           ubGmat[(icode-1)*3+2][(icode-2)*3+2] = (icode-1)*qoff; #bk-1
       ubGmat[(icode-1)*3+2][(icode-1)*3+1] = qoff; #uk-1
       if icode < N:
           ubGmat[(icode-1)*3+2][(icode)*3+2] = (N-i)*qon;
       ubGmat[(icode-1)*3+2][(icode-1)*3+3] = -1;
       # G line
       vec[(icode-1)*3 + 3] = -1;
       ubGmat[(icode-1)*3 + 3][(icode-1)*3 + 3] = - (1 + icode*gR)+m*(-icode*qoff - (N-icode)*qoff); #uk-1
       ubGmat[(icode-1)*3 + 3][(icode-1)*3 + 2] = i; #uk-1
       if icode > 0:
           ubGmat[(icode-1)*3+3][(icode-2)*3+3] = - m*(-icode*qoff)
       if icode < N:
           ubGmat[(icode-1)*3+3][(icode)*3+3] = -m*(- (N-icode)*qon) 
    
    
    Result = np.linalg.solve(ubGmat,vec)
    Gis = np.zeros(N+1)
    pis = np.zeros(N+1)
    x = qon/(qoff+qon)
    p0 = calculateProba(N,0,x)
    Geffm1 = 0#p0 

    for i in range(N+1):
        icode = i+1
        Gis[icode-1] = Result[(icode-1)*3]
        pis[icode-1] = calculateProba(N,icode-1,x)
        Geffm1 += Gis[icode-1]*pis[icode-1]
    if Nb < N:   
        # do a weighted average to avoid seeing sudden drops
        Nbi = int(Nb)
        y = Nb - Nbi
        Gnb = (1-y)*Gis[Nbi-1] + y*Gis[Nbi]
            # corresponds to Nbi ... and to Nbi+1
    else:
        Gnb = 0
    
    # print(pis)
    # print(Gis)
    
    return(Geffm1,Gnb,p0)


def interpolateNGamma(N,Nb,qoff,qon,gR,m,optionInertia,Gamma):

    Nin = int(N)
    y = N - Nin
    

    if optionInertia == 'none':
        print(Nin)
        Neval = Nin+1
        GammaeffT,GammabT,p0T = calculateGammaFull(Neval,Nb,qoff*(Neval-Nb)/(N-Nb),qon,gR)
        if Nin > 0:
            Neval = Nin
            GammaeffM,GammabM,p0M = calculateGammaFull(Neval,Nb,qoff*(Neval-Nb)/(N-Nb),qon,gR)
        else:
            p0M = 1
            GammaeffM =Gamma
            GammabM = Gamma
    else:
        print(Nin)
        Neval = Nin+1
        GammaeffT,GammabT,p0T = calculateGammaFullInertia(Neval,Nb,qoff*(Neval-Nb)/(N-Nb),qon,gR,m)
        
        if Nin > 0:
            Neval = Nin
            GammaeffM,GammabM,p0M = calculateGammaFullInertia(Neval,Nb,qoff*(Neval-Nb)/(N-Nb),qon,gR,m)
        else:
            p0M = 1
            GammaeffM =Gamma
            GammabM = Gamma
            
    Gammaeff = y*GammaeffT + (1-y)*GammaeffM
    p0 = y*p0T + (1-y)*p0M
    Gammab = y*GammabT + (1-y)*GammabM

    return(Gammaeff,p0,Gammab)

def interpolateNGamma4thorder(N,Nb,qoff,qon,gR,m,optionInertia,Gamma):

    Nin = int(N)
    if N > 2:  
        Nevals = [Nin-1,Nin,Nin+1,Nin+2]
    elif N > 1:
        Nevals = [Nin,Nin+1,Nin+2]
    else:
        Nevals = [Nin+1,Nin+2]
        
        
    nev = len(Nevals)    
    GammaeffS = np.zeros(nev)
    p0S = np.zeros(nev)
    GammabS = np.zeros(nev)
    
    for ieval in range(nev):
        Neval=Nevals[ieval]
        if optionInertia == 'none':
            GammaeffT,GammabT,p0T = calculateGammaFull(Neval,Nb,qoff*(Neval-Nb)/(N-Nb),qon,gR)
        else:
            GammaeffT,GammabT,p0T = calculateGammaFullInertia(Neval,Nb,qoff*(Neval-Nb)/(N-Nb),qon,gR,m)
            
        GammaeffS[ieval]=GammaeffT
        p0S[ieval]=p0T
        GammabS[ieval]=GammabT
    print(Nevals)
    Gammaefffunc = interp1d(Nevals,GammaeffS,kind='quadratic')
    Gammaeff = Gammaefffunc(N)
    p0func = interp1d(Nevals,p0S,kind='quadratic')
    p0 = p0func(N)
    Gammabfunc = interp1d(Nevals,GammabS,kind='quadratic')
    Gammab = Gammabfunc(N)
    
           

    return(Gammaeff,p0,Gammab)
