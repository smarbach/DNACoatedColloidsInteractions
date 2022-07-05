#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:24:55 2020

@author: marbach
"""

# Main routine to call the Melting curve function 
# For more details on outputs (beyond the melting curve), refer to commented section below
import sys
sys.path.append('../src/')
from EnergyProfileCall import returnResultEnergy, noiseProfile
import matplotlib.pyplot as plt
import numpy as np


###############################################################################
# input parameters
###############################################################################

# colloid parameters

optcolloidcolloid = 0   # if colloid-colloid interaction set to 1
                        # else this is colloid-flat surface interaction
radius = 2.5            # radius of colloid in microns
PSdensity  = 1.055      # density (g/cm^3) of PS particle at Room T (22 C)
dilatationC = 1          # account for colloid material dilatation (should only be 1 if colloid is Polystyrene)
PSCharge  = -0.019      # charge (C/m^2) of (top) colloid

# general experimental parameters

criticalHeight = 20         # HALF depth of focus (nm) or maximal height of colloids
                            # that determines wether they are bound or unbound
slabThick  = 250            # Total accessible height for colloids in microns
                            #(height of water slide)
saltConcentration  = 0.140  # Salt concentration in mol/L
optglassSurfaceFlag = 1     # 1 if surface is not covered with PS fine layer (for vdW calculation)
gravity  = 9.802            # gravity in NYC
cF127  = 0.3                # (w/v %) surfactant concentration (for effective increased length of brushes)
temperature = [60.0]          # relevant temperatures for the energy profile calculation

# brush properties


tetherSeq  = "ACCGCA"   # sticky DNA sequence
NDNAt  = 20             # TOTAL number of DNA bases on top surface (colloid), including sticky part
NDNAb  = 60             # TOTAL number of DNA bases on bottom surface
NPEOt  = 772            # TOTAL number of polymer units (for example PEO units) on top 
NPEOb  = 0              # TOTAL number of polymer units (for example PEO units) on bottom
areat  = (6.9)**2       # area (nm^2) available for 1 strand on top surface
areab  = (9.4)**2       # area (nm^2) available for 1 strand on bottom surface
ft  = 1.0               # fraction of sticky DNA on top
fb  = 1.0               # fraction of sticky DNA on bottom
DNAmodel = 'ssDNA'      # or 'dsDNA' according to the nature of the non-sticky part
persistencePEO  = 0.368 # persistence length (nm) of additional polymer (here PEO)
DNACharge  = 0          # charge per base of DNA, (real from 0 to 1). Leave to 0 to avoid 
                        # lengthy electrostatic calculations, especially at high salt concentration
                        # also these calculations are potentially unstable.

# model parameters

optdeplFlag = 0     # 1 to calculate depletion interactions
depletionTypeC = 'other' # 'F127' or 'other'
Ragg = 1            # specify agregation radius in nm if you want another type of depletion
optvdwFlag  = 1     # 1 to calculate van der Waals interactions
slideType = 'Glass' # 'Glass' for Glass facing PS; otherwise 'PSonGlass' for PS (80nm) on Glass facing PS, or 'PS' for PS facing PS or 'other'
hamakerC = 3e-12    # in which case you need to specify the hamaker constant
mushroomFlag  = 0   # 1 if brush is low density, ~ mushroom brush, otherwise 0
porosity = 0.       # (real from 0 to 1) partial penetration of micelles in brush
#modelAccuracy = 0   # 1 if you want high model accuracy (long calculation - disabled for now)
wPEO = 0.0978       # polymer excluded volume (can be adjusted to obtain measured brush height)

optNoise  = 0               # apply a noise kernel to prediction
optShot  = 1                # shot noise (1) or gaussian kernel (0)
photonTarget  = 1000         # target photon number for shot noise
penetrationDepth  = 100     # penetration depth of the TIRM in nm
gaussianWidth  = 10         # gaussian kernel width in nm



###############################################################################
# plot and print data
###############################################################################

result = returnResultEnergy(optcolloidcolloid,radius,criticalHeight,slabThick, \
                      saltConcentration,tetherSeq,NDNAt,NDNAb,NPEOt,NPEOb, \
                      areat,areab,ft,fb,persistencePEO,wPEO,DNACharge,PSCharge, \
                      optglassSurfaceFlag,PSdensity,gravity,optdeplFlag, \
                      cF127,optvdwFlag,mushroomFlag,porosity,DNAmodel,slideType,temperature, \
                          depletionType = depletionTypeC, aggRadius = Ragg, hamaker = hamakerC, \
                          dilatation = dilatationC)
print('The potential profile is calculated.')
    
    
    
if optNoise == 1:
    print('Now calculating the noise distortion, this may take some time')
    result = noiseProfile(result,optShot,photonTarget,penetrationDepth,gaussianWidth,temperature)
    
    
###############################################################################
# plot and print data
###############################################################################

allheights =   result['allheights'] # colloid-colloid or colloid-surface separation
potential = result['potential'] # potential profile
phiEl = result['phiEl'] # electrostatic interactions
phiGrav = result['phiGrav'] # gravity potential
phiDepl = result['phiDepl'] # depletion interactions
phiVdW = result['phiVdW'] # van der Waals interactions
phiSter = result['phiSter'] # steric interactions
phiBridge = result['phiBridge'] # binding interactions


indT = 0 #temperature index, in this example we're looking at just 1 temperature value
fig, ax6 = plt.subplots(figsize=[3.6,2.5])
ax6.plot(allheights*1e9,potential,':',color='black',lw = 1, label = 'potential')
if optNoise == 1:
    phi = result['phi'] # potential profile
    ax6.plot(allheights*1e9,[p-phi[-100]+potential[-100] for p in phi],color='black',lw = 1, label = 'potential with noise')
ax6.plot(allheights*1e9,phiSter,color='dodgerblue', lw = 1, label = 'steric')
ax6.plot(allheights*1e9,phiBridge,color='crimson', lw = 1, label = 'binding')
ax6.plot(allheights*1e9,phiVdW,color='orange', lw = 1, alpha = 0.7, label = 'van der Waals')
ax6.plot(allheights*1e9,phiDepl,color='hotpink', lw = 1, alpha = 0.7, label = 'depletion')
ax6.plot(allheights*1e9,phiEl,color='limegreen', lw = 1, alpha = 0.7, label = 'electrostatics')
ax6.plot(allheights*1e9,phiGrav,color='mediumblue', alpha = 0.7, lw = 1, label = 'gravity')
    
ax6.set_xlabel('Height $h$ (nm)')
ax6.set_ylabel('Potential $\Phi(h)$ ($k_B T$)')
ax6.set_xlim(28,98)
ax6.set_ylim(-8,6)

# ax6.spines['right'].set_visible(False)
# ax6.spines['top'].set_visible(False)
# # Only show ticks on the left and bottom spines
# ax6.yaxis.set_ticks_position('left')
# ax6.xaxis.set_ticks_position('bottom')


ax6.legend(loc='lower right',handlelength=1, fontsize = 6,markerscale = 0.8,labelspacing = 0.5) 
plt.tight_layout()
plt.savefig('interactionProfile'+'.pdf', format='pdf')
plt.show()
