#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:18:06 2022

@author: sm8857
"""
###############################################################################
# Routine to call the diffusion profile function 
# For more details on outputs (beyond the diffusion profile curve), refer to commented sections below
###############################################################################

import sys
srcpath='../src/'
sys.path.append(srcpath)

import math
pi = math.pi

import numpy as np
import matplotlib.pyplot as plt

from EnergyProfileCall import returnResultEnergy
from PotentialProfileModule import load_potential_profile
from KineticsModule import compute_diffusion_coefficients

###############################################################################
# input parameters
###############################################################################

# colloid parameters

optcolloidcolloid = 1   # if colloid-colloid interaction set to 1
                        # else this is colloid-flat surface interaction
radius = 3.5            # radius of colloid in microns
densities  = [1.0,20.0]     # density(ies) (g/cm^3) of particle cores
dilatationC = 0         # account for colloid material dilatation (should only be 1 if colloid is Polystyrene)
PSCharge  = -0.019      # charge (C/m^2) of (top) colloid
optionInertia = 'on'    # or none if you do not wish to account for the effect of inertia

# general experimental parameters

criticalHeight = 560/2      # HALF depth of focus (nm) or maximal height of colloids
                            # that determines wether they are bound or unbound
slabThick  =  76.2          # Total accessible height for colloids in microns
                            #(height of water slide)
saltConcentration  = 0.500  # Salt concentration in mol/L
optglassSurfaceFlag = 0     # 1 if surface is covered with glass (for surface charge electrostatics)
gravity  = 0                # 0 to remove gravity effects or 9.802 gravity in NYC (or any other gravity constant)
cF127  = 0.3                # (w/v %) surfactant concentration (for effective increased length of brushes)
reductionOfEta = 1.0        # in mPa.s, viscosity of solvent at room temperature (for water = 1.0). The code
                            # will account for viscosity changes with temperature

# coating properties

tetherSeq  = "ACCGCA"       # sticky DNA sequence
NDNAt  = 10                 # TOTAL number of DNA bases on top surface (colloid), including sticky part
NDNAb  = 20                 # TOTAL number of DNA bases on bottom surface
NPEOt  = 34                 # TOTAL number of polymer units (for example PEO units) on top 
NPEOb  = 772                # TOTAL number of polymer units (for example PEO units) on bottom
areat  = (3)**2             # area (nm^2) available for 1 strand on top surface
areab  = (7)**2             # area (nm^2) available for 1 strand on bottom surface
ft  = 0.15                  # fraction of sticky DNA on top
fb  = 0.01                  # fraction of sticky DNA on bottom
DNAmodel = 'ssDNA'          # or 'dsDNA' according to the nature of the non-sticky part
persistencePEO  = 0.368     # persistence length (nm) of additional polymer (here PEO)
DNACharge  = 0              # charge per base of DNA, (real from 0 to 1). Leave to 0 to avoid 
                            # lengthy electrostatic calculations, especially at high salt concentration
                            # also these calculations are potentially unstable.
wPEO = 1                    # polymer excluded volume (adjust to obtain measured brush height or leave to 1 for default value)
Kon = 1.616e+06             # in /M/s - check nablab.rice.edu/nabtools/kinetics.html to obtain the binding constant of your DNA sequence
kineticPerspective = 'top'  # which brush is at high density, that will be the one used to average properties over. 'top' or 'bottom' or 'centered' if both are alike.
   


# model parameters

optdeplFlag = 0     # 1 to calculate depletion interactions
depletionTypeC = 'F127' # 'F127' or 'other'
Ragg = 1            # specify agregation radius in nm if you want another type of depletion
optvdwFlag  = 1     # 1 to calculate van der Waals interactions
slideType = 'PS'    # 'Glass' for Glass facing PS; otherwise 'PSonGlass' for PS (80nm) on Glass facing PS, or 'PS' for PS facing PS or 'other'
hamakerC = 3e-12    # in which case you need to specify the hamaker constant
mushroomFlag  = 0   # 1 if brush is low density, ~ mushroom brush, otherwise 0
porosity = 0.       # (real from 0 to 1) partial penetration of micelles in brush
#modelAccuracy = 0   # 1 if you want high model accuracy (long calculation - disabled for now)


# output parameters

basename = 'MyDiffusionSimulation'  # for plots and data saves, saving extension

 
###############################################################################
# Main routine, do not change
###############################################################################

# first launch a general analysis of the interaction forces with temperature
nrefinement = 34
plottemperatureFast = np.linspace(20,100,nrefinement)
nresolution = 20 #resolution factor for the number of points on the potential profile
PSdensity = 1.0

result = returnResultEnergy(optcolloidcolloid,radius,criticalHeight,slabThick, \
                      saltConcentration,tetherSeq,NDNAt,NDNAb,NPEOt,NPEOb, \
                      areat,areab,ft,fb,persistencePEO,wPEO,DNACharge,PSCharge, \
                      optglassSurfaceFlag,PSdensity,gravity,optdeplFlag, \
                      cF127,optvdwFlag,mushroomFlag,porosity,DNAmodel,slideType,plottemperatureFast, \
                          depletionType = depletionTypeC, aggRadius = Ragg, hamaker = hamakerC, \
                          dilatation = dilatationC)



punbound = result['punbound']
NAves = result['NAves']
 
# refine the analysis on a smaller range of temperatures
valmax = punbound[-1]
isfound = 0
for iT in range(len(plottemperatureFast)):
    if isfound == 0 and NAves[iT] < 1:
        iTfound = iT
        isfound = 1
        
        
nrefinement = 50
Tm = plottemperatureFast[iTfound]
Tmax = Tm+3
Tmin = Tm-7
plottemperatureFast = np.linspace(Tmin,Tmax,nrefinement)


plt.figure(figsize=[3.6, 2.5])
ax = plt.subplot()
# and run it for the different material densities
for materialDensity in densities:
    
    if materialDensity == densities[0] or gravity != 0:
        # first figure out the potential profile in detail -- but only if mass counts run it several times
        result = returnResultEnergy(optcolloidcolloid,radius,criticalHeight,slabThick, \
                          saltConcentration,tetherSeq,NDNAt,NDNAb,NPEOt,NPEOb, \
                          areat,areab,ft,fb,persistencePEO,wPEO,DNACharge,PSCharge, \
                          optglassSurfaceFlag,materialDensity,gravity,optdeplFlag, \
                          cF127,optvdwFlag,mushroomFlag,porosity,DNAmodel,slideType,plottemperatureFast, \
                              depletionType = depletionTypeC, aggRadius = Ragg, hamaker = hamakerC, \
                              dilatation = dilatationC)
        
        allheights, potential,lambdaV, phiEl, phiGrav, phiDepl, phiVdW, phiSter,  phiBridge, phiPE, hMinsT, hAvesT, nConnectedT, \
                                            areaT, depthT, widthT, xvalues, svalues, sticky, punbound, deltaGeff, DeltaG0s,Rconnected, \
                                            nInvolvedT,NAves= \
                                            load_potential_profile('defaultSave')
                                    

    # some formatting of the input parameters is now needed
    mass = 4/3*pi*(materialDensity+1/2)*radius**3*1e-18*1e3 
    if gravity == 0:
        gravityFactors = 0*plottemperatureFast
        
    # Parameters of the polymers on the top plate
    topCoating = 0.0 #thickness in nm of top coating (if incompressible)
    #NPEOt = NPEO #148 #250 #772 #number of PEO units 
    #NDNAt = 20
    DNAcharget = -DNACharge*NDNAt #in e (# of unit charges)
    densityTetherTop = 1/areat #59 #/20 # density in 1/nm^2
    fractionStickyTop = ft
    topParameters = [topCoating,NPEOt,NDNAt,DNAcharget,densityTetherTop,fractionStickyTop]
    
    # Parameters of the polymers on the bottom plate
    bottomCoating = 0.0 #thickness in nm of bottom coating (if incompressible)
    #NPEOb = 0 #148 #250 #772 #number of PEO units 
    #NDNAb = 60
    DNAchargeb = -DNACharge*NDNAb
    densityTetherBottom = 1/areab # density in 1/nm^2
    fractionStickyBottom = fb
    bottomParameters = [bottomCoating,NPEOb,NDNAb,DNAchargeb,densityTetherBottom,fractionStickyBottom]
    
    # and now calculate the diffusion coefficient
    Tm, Dhop, Dslide, Deff = compute_diffusion_coefficients(radius, Kon, saltConcentration, plottemperatureFast, \
                                    cF127, topParameters, bottomParameters, 1.49, persistencePEO, wPEO, slideType, \
                                    srcpath, nresolution, 1.0, 'Electrostatic', DNAmodel, \
                                    slabThick, basename, kineticPerspective, \
                                    criticalHeight,  mushroomFlag,optcolloidcolloid, \
                                    nInvolvedT, NAves, potential, phiGrav, phiVdW, phiSter, punbound, 
                                    optionInertia, mass,gravityFactors, reductionOfEta)
    # not all of these parameters are actually used but its fine for now
    
    # and now make the plot
    if materialDensity < 10 and materialDensity > 0:
        ax.plot(plottemperatureFast,Deff[-1]+plottemperatureFast*0,'--',color= [0.5,0.5,0.5], label='$D_{0}$, bare',linewidth = 1)
        Dpoly = Deff
        ax.plot(plottemperatureFast,Deff,color= 'purple', label='$D_{eff}$, light',linewidth = 1)
    elif materialDensity > 10 and materialDensity < 30:
        ax.plot(plottemperatureFast,Deff,color=  'goldenrod', label='$D_{eff}$, heavy',linewidth = 1)
        Dgold = Deff
    else:
        ax.plot(plottemperatureFast,Deff,color= 'black', label='$D_{eff}$, other',linewidth = 1)
    

ax.set_xlabel('Temperature ($\degree$C) ')
ax.set_ylabel('Effective diffusion ($\mu$m$^2$/s)',labelpad = 1.5)
ax.grid(b=True, which='major', axis='y', color='gray', lw = 0.2, linestyle='--')
plt.tight_layout()
ax.set_xlim(Tmin,Tmax)
ax.legend(loc='upper left',handlelength=1,markerscale = 0.8,labelspacing = 0.5)
plt.savefig(basename + '_DiffusionPlot'+'.eps')
plt.show()    
    
# plot the number of bonds
plt.figure(figsize=[3.6, 2.5])
ax = plt.subplot()
ax.plot(plottemperatureFast,NAves,color= [0,0.5,0.5], label='bound, $\overline{N}$',linewidth = 1)
ax.plot(plottemperatureFast,nInvolvedT,color= [0.5,0.5,0.5], label='available, $N$',linewidth = 1)
ax.legend(loc='upper right',handlelength=1, markerscale = 0.8,labelspacing = 0.5)
ax.set_yscale("log")
ax.set_ylabel('Number of pairs',labelpad = 1.5)
ax.grid(b=True, which='major', axis='y', color='gray', lw = 0.2, linestyle='--')
ax.set_xlim(Tmin,Tmax) 
plt.tight_layout()
ax.set_ylim(1,2e2)
plt.savefig(basename + '_NumberOfBounds'+'.eps')
plt.show()    


# plot the relative difference
diffPlot = [(Dpoly[i] - Dgold[i])/(Dpoly[i]+Dgold[i])*2*100 for i in range(len(Dgold))]
print('Maximum difference between diffusion coeffs is', np.max(diffPlot))


plt.figure(figsize=[3.6, 2.5])
ax = plt.subplot()
ax.plot(plottemperatureFast,diffPlot,color= [0,0,0],linewidth = 1)
ax.set_ylabel('Difference (%)',labelpad = 1.5)
ax.grid(b=True, which='major', axis='y', color='gray', lw = 0.2, linestyle='--')
plt.tight_layout()
ax.set_ylim(0,10)
ax.set_xlim(Tmin,Tmax)
plt.savefig(basename + '_percentage'+'.eps')
plt.show() 