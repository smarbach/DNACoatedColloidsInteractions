from PotentialProfileModule import compute_potential_profile
import numpy as np

def returnResultEnergy(optcolloidcolloidFlag,radius,focal_length,slabThick, \
                          salt_concentration,tether_seq,NDNAt,NDNAb,NPEOt,NPEOb, \
                          areat,areab,ft,fb,persistencePEO,wPEO,DNACharge,PSCharge, \
                          optglassSurfaceFlag,PSdensity,gravity,optdeplFlag, \
                          cF127,optvdwFlag,mushroomFlag,porosity,DNAmodel,temperature):

    gravityFactor = 0 #no gravity correction here
    
    # Main Physical parameters of the colloid
    #radius = 2.5 #0.575 #3.0 # in microns
    #PSdensity = 1.055 #1.055 #density of PS in kg/m3  #really need to find a reference for that 
        
    # Chemistry parameters of the fluidic cell
    saltConcentration = salt_concentration #salt concentration in mol/L
    if optglassSurfaceFlag:
        slideType = "Glass" #select "Glass" or "PSonGlass" or "PS"
    else:
        slideType = "PS"
    slabH = slabThick #height in microns
    
    # DNA sequence
    tetherSeq = tether_seq #put only Ts if you don't want bridging - but if there is DNA on one side only it will not bridge
    #tetherSeq = "GCAG"
    #tetherSeq = ""
    if DNACharge == 0:
        PEmodel = "Electrostatics" #select "Polyelectrolyte" or "Electrostatics"
    else:
        PEmodel = "Polyelectrolyte"
    #bridgingOption = "symmetric" #-- you can put that on to symmetric but essentially you will see very little difference
    bridgingOption = "anti"
    
    
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
    
    # do NOT change
    ellDNA = 0.56#56#*(50/30)**(1/5) # this is actually the length of DNA nucleotide # not used in the code though (it's directly coded in)
    ellPEO = persistencePEO#514#423#0.464#*(50/30)**(1/5) #in nm # choose to account for F127 effective elongation
    eVparameter = 1.0 #
    
    # playing with parameters inside the code
    #PSCharge = -0.019 # in C/m^2
    if optglassSurfaceFlag:
        GlassCharge = -0.0005  # in C/m^2
    else:
        GlassCharge = PSCharge
    
    # depletion interactions
    #cF127 = 0 #fraction of F127 in mass per volume. Set to 0 if you do not want depletion forces
    #penetration = 0
    penetration = 1
    criticalHeight = focal_length
    
    
    # accuracy
    nresolution = 5 #resolution factor not too high for this web app

    
    basename = 'defaultSave'
    #myTemp = []
    # thos extreme values are dictated by the van der waals interactions

    #First coarse grained run
    #print("First run")
    #for Temperature in np.arange(273+Tmin,273+Tmax,5):
    #    myTemp.append(Temperature-273)
    myTemp = temperature
    srcpath='../src/'   
        
    allheights, potential,lambdaV, phiEl, phiGrav, phiDepl, phiVdW, phiSter,  phiBridge, phiPE, hMinsT, hAvesT, nConnectedT, \
                                    areaT, depthT, widthT, xvalues, svalues, sticky, punbound1, deltaGeff, DeltaG0s,Rconnected, \
                                    nInvolvedT,NAves, gravityFactors = \
                                    compute_potential_profile( \
                                    radius, PSdensity, gravity, saltConcentration, myTemp, PSCharge, GlassCharge, \
                                    cF127, topParameters, bottomParameters, ellDNA, ellPEO, wPEO, slideType, gravityFactor, \
                                    tetherSeq, srcpath, nresolution, penetration, PEmodel, eVparameter, slabH, bridgingOption, basename, \
                                    criticalHeight, optcolloidcolloidFlag, optdeplFlag, optvdwFlag, mushroomFlag, porosity, DNAmodel)
 
    
    result = {'allheights': allheights,  'potential': potential,  'phiEl': phiEl, \
              'phiGrav': phiGrav,  'phiDepl': phiDepl,  'phiVdW': phiVdW,  'phiSter': phiSter,  'phiBridge': phiBridge}
        
    return(result)
