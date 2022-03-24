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
from EnergyProfileCall import returnResultEnergy
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
wPEO = 0.0978            # polymer excluded volume (adjust to obtain measured brush height)

# model parameters

optdeplFlag = 0     # 1 to calculate van der Waals interactions
optvdwFlag  = 1     # 1 to calculate van der Waals interactions
mushroomFlag  = 0   # 1 if brush is low density, ~ mushroom brush, otherwise 0
porosity = 0.       # (real from 0 to 1) partial penetration of micelles in brush
#modelAccuracy = 0   # 1 if you want high model accuracy (long calculation - disabled for now)


###############################################################################
# plot and print data
###############################################################################

result = returnResultEnergy(optcolloidcolloid,radius,criticalHeight,slabThick, \
                      saltConcentration,tetherSeq,NDNAt,NDNAb,NPEOt,NPEOb, \
                      areat,areab,ft,fb,persistencePEO,wPEO,DNACharge,PSCharge, \
                      optglassSurfaceFlag,PSdensity,gravity,optdeplFlag, \
                      cF127,optvdwFlag,mushroomFlag,porosity,DNAmodel,temperature)

    
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
ax6.plot(allheights*1e9,potential[:][indT],':',color='black',lw = 1, label = 'potential')
ax6.plot(allheights*1e9,phiSter[:][indT],color='dodgerblue', lw = 1, label = 'steric')
ax6.plot(allheights*1e9,phiBridge[:][indT],color='crimson', lw = 1, label = 'binding')
ax6.plot(allheights*1e9,phiVdW[:][indT],color='orange', lw = 1, alpha = 0.7, label = 'van der Waals')
ax6.plot(allheights*1e9,phiDepl[:][indT],color='hotpink', lw = 1, alpha = 0.7, label = 'depletion')
ax6.plot(allheights*1e9,phiEl[:][indT],color='limegreen', lw = 1, alpha = 0.7, label = 'electrostatics')
ax6.plot(allheights*1e9,phiGrav[:][indT],color='mediumblue', alpha = 0.7, lw = 1, label = 'gravity')
    
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
plt.savefig('interactionProfile'+'.eps', format='eps')
plt.show()

