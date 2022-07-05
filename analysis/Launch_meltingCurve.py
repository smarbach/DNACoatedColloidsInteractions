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
from MeltingCurveCall import returnResult
import matplotlib.pyplot as plt



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
optglassSurfaceFlag = 1     # 1 if surface is covered with glass (for surface charge electrostatics)
gravity  = 9.802            # gravity in NYC
cF127  = 0.3                # (w/v %) surfactant concentration (for effective increased length of brushes)

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

optdeplFlag = 0     # 1 to calculate depletion interactions
depletionTypeC = 'other' # 'F127' or 'other'
Ragg = 1            # specify agregation radius in nm if you want another type of depletion
optvdwFlag  = 1     # 1 to calculate van der Waals interactions
slideType = 'Glass' # 'Glass' for Glass facing PS; otherwise 'PSonGlass' for PS (80nm) on Glass facing PS, or 'PS' for PS facing PS or 'other'
hamakerC = 3e-12    # in which case you need to specify the hamaker constant
mushroomFlag  = 0   # 1 if brush is low density, ~ mushroom brush, otherwise 0
porosity = 0.       # (real from 0 to 1) partial penetration of micelles in brush
#modelAccuracy = 0   # 1 if you want high model accuracy (long calculation - disabled for now)

# melting option
optThermo = 1                                                                # 1 = build the thermodynamic melting curve or 0 = kinetic one
diffusion = (1.38e-23*300)/(6*3.1415*0.001*radius*1e-6)*1e-9/(radius*1e-9)   # parameter for kinetic melting, vertical diffusion in the well
meltingTime = 1*60                                                           # parameter for kinetic melting, number of seconds to determine if particle is unbound


###############################################################################
# plot and print data
###############################################################################

result = returnResult(optcolloidcolloid,radius,criticalHeight,slabThick, \
                      saltConcentration,tetherSeq,NDNAt,NDNAb,NPEOt,NPEOb, \
                      areat,areab,ft,fb,persistencePEO,wPEO,DNACharge,PSCharge, \
                      optglassSurfaceFlag,PSdensity,gravity,optdeplFlag, \
                      cF127,optvdwFlag,mushroomFlag,porosity,DNAmodel, slideType, \
                          depletionType = depletionTypeC, aggRadius = Ragg, hamaker = hamakerC, \
                          dilatation = dilatationC, meltingOpt = optThermo, diffusionZ = diffusion, meltTime = meltingTime)
    
###############################################################################
# plot and print data
###############################################################################

#print(result['data']) # print temperature values
#print(result['labels']) # print unbound probability values

punbound =   result['data'] # unbound probability values
plottemperatureFast = result['labels'] # temperature values

DeltaH0 = result['DeltaH0']
DeltaS0 = result['DeltaS0']

print('Delta H0 (kcal/K/mol) is', DeltaH0, ' and Delta S0 (cal/mol) is',DeltaS0, 'for this sticky sequence')


pmax = punbound[-1]
Tm = 0
for iT in range(len(plottemperatureFast)):
    if Tm == 0 and punbound[iT] >= pmax/2:
        Tm = plottemperatureFast[iT]/2 + plottemperatureFast[iT-1]/2

print('The melting T was',Tm)


fig, ax = plt.subplots(figsize=[3.6,2.5])

ax.plot(plottemperatureFast,punbound,':o', \
              color='mediumblue', lw = 1, ms = 2, mew=1, \
              mec='mediumblue',  label = 'Melting curve') 

ax.plot([Tm,Tm],[0,100000], ':', color = 'gray', lw = 1.0, label = '$T_m = ${:.1f}%$^\circ$ C'.format(Tm) )

    
# ax.xaxis.set_major_locator(MultipleLocator(5))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# ax.xaxis.set_minor_locator(MultipleLocator(1))
    
ax.set_xlabel('Temperature (\N{DEGREE SIGN}C)')
ax.set_ylabel('Unbound probability')
ax.legend(loc='lower right',handlelength=1, fontsize = 6,markerscale = 0.8,labelspacing = 0.5)

ax.grid(b=True, which='major', axis='y', color='gray', lw = 0.2, linestyle='--')


ax.set_xlim(Tm-10,Tm+10)
ax.set_ylim(0,1)

# ax5.spines['right'].set_visible(False)
# ax5.spines['top'].set_visible(False)
# # Only show ticks on the left and bottom spines
# ax5.yaxis.set_ticks_position('left')
# ax5.xaxis.set_ticks_position('bottom')

fig.tight_layout()
plt.savefig('meltingCurve'+'.pdf', format='pdf')
plt.show()

