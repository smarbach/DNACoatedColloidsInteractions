from PotentialProfileModule import compute_potential_profile
import numpy as np

def returnResult(optcolloidcolloidFlag,radius,focal_length,slabThick, \
                          salt_concentration,tether_seq,NDNAt,NDNAb,NPEOt,NPEOb, \
                          areat,areab,ft,fb,persistencePEO,wPEO,DNACharge,PSCharge, \
                          optglassSurfaceFlag,PSdensity,gravity,optdeplFlag, \
                          cF127,optvdwFlag,mushroomFlag,porosity,DNAmodel):

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
    Tmin = 20
    Tmax = 80

    #First coarse grained run
    #print("First run")
    #for Temperature in np.arange(273+Tmin,273+Tmax,5):
    #    myTemp.append(Temperature-273)
    myTemp = [t for t in range(Tmin,Tmax,5)]
    srcpath='../src/'   
        
    allheights, potential,lambdaV, phiEl, phiGrav, phiDepl, phiVdW, phiSter,  phiBridge, phiPE, hMinsT, hAvesT, nConnectedT, \
                                    areaT, depthT, widthT, xvalues, svalues, sticky, punbound1, deltaGeff, DeltaG0s,Rconnected, \
                                    nInvolvedT,NAves, gravityFactors = \
                                    compute_potential_profile( \
                                    radius, PSdensity, gravity, saltConcentration, myTemp, PSCharge, GlassCharge, \
                                    cF127, topParameters, bottomParameters, ellDNA, ellPEO, wPEO, slideType, gravityFactor, \
                                    tetherSeq, srcpath, nresolution, penetration, PEmodel, eVparameter, slabH, bridgingOption, basename, \
                                    criticalHeight, optcolloidcolloidFlag, optdeplFlag, optvdwFlag, mushroomFlag, porosity, DNAmodel)
 
    # These are some options I will have to add to make this work. 
                                
    #criticalHeight = 20
    #optcolloidcolloidFlag = 0                       
    #optdeplFlag = 0
    #optvdwFlag = 1
    #mushroomFlag = 0
    #porosity = 0 #or 11
    # change cF127 to 1 to get some stretching from F127

    valChange = punbound1[-1]/2+punbound1[0]/2
    
    indexRefine = next(x for x, val in enumerate(punbound1) if val > valChange)
    Trefinemin = myTemp[indexRefine] - 8 +273
    Trefinemax = myTemp[indexRefine] + 5 +273
   
    myTemp1 = myTemp
    myTemp = []
    Temperature = Trefinemin 
    
    #First coarse grained run
#    print("Second run")
    while Temperature < Trefinemax:
        myTemp.append(Temperature-273)
        #plt.plot(myTemp,Punbound)  
        
        if (Temperature >= Trefinemin and Temperature < Trefinemax) :
            tempStep = 1.0
        else:
            tempStep = 5.0
        Temperature += tempStep

    print(myTemp)

    allheights, potential,lambdaV, phiEl, phiGrav, phiDepl, phiVdW, phiSter,  phiBridge, phiPE, hMinsT, hAvesT, nConnectedT, \
                                    areaT, depthT, widthT, xvalues, svalues, sticky, punbound2, deltaGeff, DeltaG0s,Rconnected, \
                                    nInvolvedT,NAves, gravityFactors = \
                                    compute_potential_profile( \
                                    radius, PSdensity, gravity, saltConcentration, myTemp, PSCharge, GlassCharge, \
                                    cF127, topParameters, bottomParameters, ellDNA, ellPEO, wPEO, slideType, gravityFactor, \
                                    tetherSeq, srcpath, nresolution, penetration, PEmodel, eVparameter, slabH, bridgingOption, basename, \
                                    criticalHeight, optcolloidcolloidFlag, optdeplFlag, optvdwFlag, mushroomFlag, porosity, DNAmodel)
    
    valChange = punbound1[-1]/2+punbound1[0]/2
    
    indexRefine = next(x for x, val in enumerate(punbound2) if val > valChange)
    Trefinemin2 = myTemp[indexRefine] - 1 +273
    Trefinemax2 = myTemp[indexRefine] + 1 +273
   
    myTemp2 = myTemp
    myTemp = []
    Temperature = Trefinemin2 
    
    #First coarse grained run
#    print("Second run")
    while Temperature < Trefinemax2:
        myTemp.append(Temperature-273)
        #plt.plot(myTemp,Punbound)  
        
        if (Temperature >= Trefinemin2 and Temperature < Trefinemax2) :
            tempStep = 0.2
        else:
            tempStep = 5.0
        Temperature += tempStep

    allheights, potential,lambdaV, phiEl, phiGrav, phiDepl, phiVdW, phiSter,  phiBridge, phiPE, hMinsT, hAvesT, nConnectedT, \
                                    areaT, depthT, widthT, xvalues, svalues, sticky, punbound3, deltaGeff, DeltaG0s,Rconnected, \
                                    nInvolvedT,NAves, gravityFactors = \
                                    compute_potential_profile( \
                                    radius, PSdensity, gravity, saltConcentration, myTemp, PSCharge, GlassCharge, \
                                    cF127, topParameters, bottomParameters, ellDNA, ellPEO, wPEO, slideType, gravityFactor, \
                                    tetherSeq, srcpath, nresolution, penetration, PEmodel, eVparameter, slabH, bridgingOption, basename, \
                                    criticalHeight, optcolloidcolloidFlag, optdeplFlag, optvdwFlag, mushroomFlag, porosity, DNAmodel)
    

    myTemp3 = myTemp
    #print(myTemp1)
    #print(myTemp2)
    #print(myTemp3)
    #print(punbound1)
    #print(punbound2)
    #print(punbound3)

    myTempTot = [np.concatenate((myTemp1, myTemp2, myTemp3))]
    #print(myTempTot)
    punboundTot = [np.concatenate((punbound1, punbound2, punbound3))]
    #print(punboundTot)
    table = np.concatenate((myTempTot,punboundTot),axis = 0)
    #print(table)
    tableU = np.unique(table, axis=1)
    #print(tableU)
    sortedArr = tableU [ :, tableU[0].argsort()]
    #print(sortedArr)
    #punbound = [1,2,3]
    Lmax = int(np.size(sortedArr)/2)
    punboundout = [sortedArr[1][i] for i in range(Lmax)]
    #labels = [t for t in myTemp]
    myTempout = [sortedArr[0][i] for i in range(Lmax)]
    result = {'labels': myTempout,  'data': punboundout}
    return(result)
