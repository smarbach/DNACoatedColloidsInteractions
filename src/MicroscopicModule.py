#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:48:52 2020

@author: sophie marbach
"""
import math
import numpy as np
from CommonToolsModule import DerjaguinIntegral,CenterDataNoYmove, CenterData
from CenteringDataModule import centerPotentialGravity, centerPotentialGravityKnown, centerPotentialRemoveGravity, centerPotentialRemoveGravityKnown, centerPotentialRemoveGravityKnownExperiment

from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.special import factorial
from math import log , exp, sqrt, inf, erf, floor, ceil
pi = math.pi
import matplotlib.pyplot as plt
from numpy.random import randn
import matplotlib


# Constants
Na = 6.02*10**23
Rg = 8.314 #J/(mol.K); #perfect gas constant
Tbase = 273.15
kB = Rg/Na; #boltzmann constant

def computeStickyParameters(allhs,phi,thresh,idMin,gravHeight):

    #print(phi[idMin])
    #for phiv in phi:
    #    print(-phiv +phi[idMin] )
    weights = [exp(-phiv+phi[idMin]) for phiv in phi]
    ftotfit = InterpolatedUnivariateSpline(allhs,weights, k=1)
    
    res = 0
    res2 = 0
    threshh = allhs[idMin] + thresh
    thresh2 = allhs[idMin] + 600e-9 #600e-9 #min(gravHeight*1e-9*7,max(allhs)) # I changed this recently it used to be   
    # for Fan's data, it doesn't change much if the boundary is between 3 to 6*gravitational Height, but it changes otherwise. 
    # I think it's also because beyond 600nm I'm not resolving things super finely. 
    #print(threshh,allhs[-1])
    for ih in range(len(allhs)):
        if allhs[ih] > threshh and res == 0:
            res = ih
        if allhs[ih] >= thresh2 and res2 == 0:
            res2 = ih
    #print(res,res2)
    if res2 == 0:
        thresh2 = allhs[idMin] + 100e-9
        for ih in range(len(allhs)):
            if allhs[ih] > thresh2 and res2 == 0:
                res2 = ih
        if res2 == 0:
            res2 = len(allhs)-1
    if res == 0:
        res = len(allhs)-1
    
    #print(allhs[0],allhs[res],phi[idMin],allhs[idMin])
    
    sticky = ftotfit.integral(allhs[0],allhs[res])/sqrt(2*pi) #divide by sqrt(2pi) such that this is really the curvature radius
    
    norm = ftotfit.integral(allhs[0],allhs[res2])/sqrt(2*pi) #also divide normalization by sqrt(2pi)
    
    if gravHeight > 0:
        hgrav = gravHeight*1e-9
        addup = 1/sqrt(2*pi)*hgrav*exp(phi[idMin])*(-exp(-allhs[-1]/hgrav) + exp(-thresh2/hgrav))
        
        norm2 = ftotfit.integral(allhs[0],allhs[-1])/sqrt(2*pi) #also divide normalization by sqrt(2pi)
    else:
        addup = 0
    
    if threshh > thresh2: # where really only gravity matters
        sticky = ftotfit.integral(allhs[0],allhs[res2])/sqrt(2*pi) #divide by sqrt(2pi) such that this is really the curvature radius
        if gravHeight > 0:
            hgrav = gravHeight*1e-9
            addups = 1/sqrt(2*pi)*hgrav*exp(phi[idMin])*(-exp(-allhs[-1]/hgrav) + exp(-threshh/hgrav))
        
            #norm2 = ftotfit.integral(allhs[0],allhs[-1])/sqrt(2*pi) #also divide normalization by sqrt(2pi)
        else:
            addups = 0
        sticky = sticky + addups
    # print("this is supposed to be the norm", norm2)
    # print("this is supposed the shorter norm", norm)
    # print("this is supposed the total norm", norm+addup)
    # print("this is supposed the actual gravitational Height", gravHeight)
    # print("this is supposed to be the sticky parameter", sticky)
    
    punbound = 1 - sticky/(norm+addup)
    
    return(sticky,punbound)

def makeRadialMap(r,s,radius,zoom,proportion):
    
    
    radialfunc = InterpolatedUnivariateSpline(r,s, k=1)
    
    disc = 1001 #number of points on the map

     # number of pixels to make white
    rxp = floor(radius*zoom*disc/2)
    print(rxp)
    rxm = ceil(radius*zoom*disc/2 - proportion)
    Z = np.zeros((disc,disc))
    for ix in range(disc):
        for iy in range(disc):
            rad = 2*sqrt((ix-disc/2)**2 + (iy-disc/2)**2)/disc
            if rad < 1:
                Z[ix][iy] = radialfunc(rad/zoom)
            else:
                Z[ix][iy] = float("NAN")
                
            if rxp < disc/2:
                
                if rad*disc/2 < rxp and rad*disc/2 > rxm:
                    
                    Z[ix][iy] = float("NAN")
    
    
    return(Z)



    
def calculateMicroscopicDetails(allhs,potential,phiBridge,Nbridge,Allsigmaabs,AllKabs,sigmaCut,Radius,Nt,criticalHeight,optcolloid,gravityFactors,geoFactor):
    hMins = np.zeros(Nt)
    hAves = np.zeros(Nt)
    NAves = np.zeros(Nt)
    width = np.zeros(Nt)
    phiMins = np.zeros(Nt)
    Nconnected = np.zeros(Nt)
    Npotential = np.zeros(Nt)
    
    Rconnected = np.zeros(Nt)
    area = np.zeros(Nt)
    nh = len(allhs)
    xvalues = np.zeros((nh,Nt))
    pNb = np.zeros((nh-1,Nt))
    
    pplot = np.zeros(nh-1)
    Nplot = np.zeros(nh-1)
    
    svalues = np.zeros((nh,Nt))
    punbound = np.zeros(Nt)
    sticky = np.zeros(Nt)
    deltaGeff = np.zeros(Nt)
    
    # if criticalHeight <= 50:
    #     heightCut1 = 1e-7 #past 100 nm the particle is melted
    # else:
    #     heightCut1 = allhs[-1]
    # print("maximal height used to account for melting", heightCut1)
    heightCut2 = criticalHeight*1e-9 #past 20 nm -- this is the resolution that Fan has.  
    depthT  = -20
    
    # preliminary loop to extract the gravity slope at - 3kT
    
    
    #heightCentered, potentialCentered, gravityFactorInitModel = centerPotentialRemoveGravity(allheights2,potential2,0,gravmins,gravmaxs,h_adjust,h_adjustment,hframe)        
    
    idTslope = 0
    for idT in range(Nt):
        lowestpotentialindex = np.array(potential[:][idT]).argmin()
        phi = potential[:][idT]
        phiMin = phi[lowestpotentialindex]
        if (phiMin < -3 and idTslope == 0) or idT == Nt-1:
            idTslope = idT
            allheights2, potential2 = CenterData(allhs, phi, 1e9)
            h_adjustment = 0
            gravmins = 40
            hframe = 1 
            h_adjust = 1
            gravmaxs = 180
            
            heightCentered, potentialCentered, gravityFactorInitModel = centerPotentialRemoveGravity(allheights2,potential2,0,gravmins,gravmaxs,h_adjust,h_adjustment,hframe)        

    
    
    
    for idT in range(Nt):
        lowestpotentialindex = np.array(potential[:][idT]).argmin()

        
        # figure out "average" location of the particle - considered as the location of the minimum... 
        hMins[idT] = allhs[lowestpotentialindex]
        
        # figure out "average" location of the particle - considered as the weighted average with the potential
        phi = potential[:][idT]
        
        allheights2, potential2 = CenterData(allhs, phi, 1e9)
        
        # plt.plot(allheights2,potential2)
        # plt.show()
        hframe = 1
        if gravityFactors[idT] > 0:
            ghere = gravityFactorInitModel*gravityFactors[idT]/gravityFactors[idTslope]
            phiMin = phi[lowestpotentialindex]
            h_adjustment = 0
            h_adjust = 1
            if (phiMin < -2):
                gravmins = 40
                gravmaxs = 180
            else:
                gravmins = 50
                gravmaxs = 200
            heightCentered, potentialCentered, gravityFactor = centerPotentialRemoveGravityKnownExperiment(allheights2,potential2,0,gravmins,gravmaxs,h_adjust,h_adjustment,hframe,ghere)    
    
            gravHeight = 1/gravityFactor*1e9 #gravitational Height in nanometers
        else:
            heightCentered = allheights2
            potentialCentered = potential2
            gravHeight = np.max([np.max(allhs),1000]) 

        #find the min but only up to 30nm from center because the long tails sometimes don't look good
        hidmax = 0
        idh = 0
        while hidmax == 0 and idh < len(heightCentered):
            if heightCentered[idh] > 30:
                hidmax = idh
            idh += 1
        if idh == len(heightCentered):
            hidmax = len(heightCentered)-1
        
        lowestpotentialindexgravityremoved = np.array(potentialCentered[0:hidmax]).argmin()
        
        phiMin = phi[lowestpotentialindex]
        depthT = potentialCentered[lowestpotentialindexgravityremoved]
        phiMins[idT] = potentialCentered[lowestpotentialindexgravityremoved]
        # btw compute width of potential
        res1 = 0
        res2 = 0
        # width of the potential corresponding to a 5kT fluctuation
        for ir in range(nh):
            if phi[ir] < phiMin+5 and res1 == 0:
                res1 = ir
        for ir in range(nh):
            #the other limit has to be found going down, to avoid double potential
            if phi[nh-1-ir] < phiMin+5 and res2 == 0:
                res2 = nh-1-ir
        #print(res1,res2)
        width[idT] = allhs[res2] - allhs[res1]
        if AllKabs[idT][lowestpotentialindex] > 0:
            deltaGeff[idT]  = -log(AllKabs[idT][lowestpotentialindex])
        else:
            deltaGeff[idT] = float("NaN")
        probas = [exp(-phiv+phiMin) for phiv in phi] #boltzmann weights
        hprobas = [exp(-phi[iv]+phiMin)*allhs[iv] for iv in range(len(phi))] #boltzmann weights
        hAves[idT] = sum(hprobas)/sum(probas) # because the potential is so deep this is actually not so different from hMins (the same...)
        
        Nprobas = [exp(-phi[iv]+phiMin)*Nbridge[idT][iv] for iv in range(len(phi))] #boltzmann weights
        NAves[idT] = sum(Nprobas)/sum(probas)
        
        # you might also want to figure out the distribution of bonds
        
        distributionPlot = 0
        if distributionPlot == 1:
            
            
            matplotlib.rcParams.update(
                {'font.sans-serif': 'Arial',
                 'font.size': 8,
                 'font.family': 'Arial',
                 'mathtext.default': 'regular',
                 'axes.linewidth': 0.35, 
                 'axes.labelsize': 8,
                 'xtick.labelsize': 7,
                 'ytick.labelsize': 7,     
                 'lines.linewidth': 0.35,
                 'legend.frameon': False,
                 'legend.fontsize': 7,
                 'xtick.major.width': 0.3,
                 'xtick.minor.width': 0.3,
                 'ytick.major.width': 0.3,
                 'ytick.minor.width': 0.3,
                 'xtick.major.size': 1.5,
                 'ytick.major.size': 1.5,
                 'xtick.minor.size': 1,
                 'ytick.minor.size': 1,
                })
            
            
            # this is to plot the distribution of #of bonds. so basically p(N)dN ~ p(h)dh such that 
            # p(N)~ exp(-beta phi(N)) dh/dN and comparing to wether or not this is a Poisson distribution 
            dhdNbCentered = [(allhs[iv+1] - allhs[iv])/(Nbridge[idT][iv+1]- Nbridge[idT][iv]) for iv in range(len(probas)-1)]
            probaCentered = [probas[iv]/2 + probas[iv+1]/2 for iv in range(len(probas)-1)]
            
            for idh in range(len(probas)-1):
                pNb[idh][idT] = -dhdNbCentered[idh]*probaCentered[idh]/sum(probas)
                Nplot[idh] = (Nbridge[idT][idh+1] + Nbridge[idT][idh])/2
                if Nplot[idh] <0.1:
                    # I don't have resolution below that. 
                    pplot[idh] = 0
                else:
                    pplot[idh] = pNb[idh][idT]
            
            t = np.arange(0,150, 1)
            d = np.exp(-NAves[idT])*np.power(NAves[idT], t)/factorial(t)
            
            fig,ax = plt.subplots(figsize=(3.7,3.0))

            

            
            #print(Nplot)
            ax.plot([NAves[idT],NAves[idT]],[0,max(pplot)],color = 'black', label='Average $<N_{bound}>$',linewidth = 1.5)

            ax.plot(Nplot,pplot,color='crimson',label='Self-consistent distribution',linewidth = 1.5)
            ax.plot(t, [pd*max(pplot)/max(d) for pd in d],'--',color='lightpink', label='Poisson distribution \n with same average',linewidth = 1.5)


            ax.set_xlim(40.0,140)
            ax.legend(loc='upper left',handlelength=1, fontsize = 7,markerscale = 0.8,labelspacing = 0.5)   

            ax.set_xlabel('Number of bonds N')
            ax.set_ylabel('Distribution of bonds p(N)')
            fig.tight_layout()
    
            plt.savefig('MicroscopicDistribution.svg', format='svg', transparent = True)
            plt.savefig('MicroscopicDistribution.eps', format='eps')
            plt.show()


            
            plt.show()
            
            # at any height you can also compare the average number of bonds to the potential
            fig,ax = plt.subplots(figsize=(3.7,3.0))
            
            #print(Nbridge[0][idT])
            ax.plot([h*1e9 for h in allhs],[p for p in phiBridge[:][idT]],color='crimson', label='Binding (Self-consistent)',linewidth = 1.5)

            ax.plot([h*1e9 for h in allhs], [-n for n in Nbridge[:][idT]],'--',color='lightpink', label='Binding -$k_BTN_{bound}(h)$ \n (Poisson approximation)',linewidth = 1.5)
            
            ax.plot([h*1e9 for h in allhs],[p for p in potential[:][idT]],color = 'black',label='Full potential (Self-consistent)',linewidth = 1.5)
            ax.plot([h*1e9 for h in allhs],[potential[idT][ih] - Nbridge[idT][ih]- phiBridge[idT][ih] for ih in range(len(allhs))],'--',color='gray',  label='Full potential \n (Poisson approximation)',linewidth = 1.5)
            
            ax.set_xlim(37.5,57.5)
            ax.set_ylim(-60,3)
            ax.legend(loc='lower right',handlelength=1.5, fontsize = 7,markerscale = 0.8,labelspacing = 0.5)   

            ax.set_xlabel('Separation h (nm)')
            ax.set_ylabel('Potential $\phi(h)/k_BT$', labelpad = -2)
            fig.tight_layout()
    
            plt.savefig('MicroscopicPotential.svg', format='svg', transparent = True)
            plt.savefig('MicroscopicPotential.eps', format='eps')
            plt.show()
        
        
        
        # figure out the number of tethers attached in "average", e.g. corresponding to this location
        Nconnected[idT] = Nbridge[idT][lowestpotentialindex] # I have no clue why but somehow the indices reverse here.
        isfound = 0
        hmaxConnected = 0
        
        
        
        SigmaMax = Allsigmaabs[idT][lowestpotentialindex] # fraction of sticky ends at the center of the sphere for the lowest potential energy
        
        for idh in range(nh-lowestpotentialindex):
            # I don't think this criteria makes sense. instead I would argue for a another criteria on the fraction of sticky ends
            #if isfound == 0 and Nbridge[idT][lowestpotentialindex] - Nbridge[idT][lowestpotentialindex+idh] > 0.9*Nbridge[idT][lowestpotentialindex] and Nbridge[idT][lowestpotentialindex] > 1:
            sigmaHeight =  Allsigmaabs[idT][lowestpotentialindex+idh]
            if isfound == 0 and sigmaHeight < 0.1*SigmaMax and Nbridge[idT][lowestpotentialindex] > 1:
                isfound = 1
                hmaxConnected = allhs[idh+lowestpotentialindex]
        if isfound == 1 and abs(Radius - hmaxConnected + hMins[idT]) < Radius:
            if optcolloid:
                Rconnected[idT] = sqrt((Radius**2 - (Radius - hmaxConnected/2 + hMins[idT]/2)**2)/Radius**2)
            else:
                Rconnected[idT] = sqrt((Radius**2 - (Radius - hmaxConnected + hMins[idT])**2)/Radius**2)
        else:
            Rconnected[idT] = 10
        
        # finally figure out the typical "area in contact" 
        # there are several definitions for that. One option is to return the portion of the sphere 
        # where the local fraction is larger than say 50%. Another is to return the portion where its 90% of whats at the center. 
        # this cutoff is defined by sigmaCut
        
        
        
        # figure out radially the % of tethers bound knowing that the height is hMins[idT] = allhs[lowestpotentialindex]
        for idh in range(nh-lowestpotentialindex):
            heightsMax = allhs[idh+lowestpotentialindex] # current height investigated
            heightsMin = hMins[idT]
            if optcolloid:
                rinvest = Radius - heightsMax/2 + heightsMin/2
            else:
                rinvest = Radius - heightsMax + heightsMin
            if abs(rinvest) <= Radius:
                #xR = (Radius**2 - (Radius - heightsMax + heightsMin)**2)/Radius**2
                #print(xR)
                xvalues[idh][idT] = sqrt((Radius**2 - (rinvest)**2)/Radius**2)
                
                svalues[idh][idT] = Allsigmaabs[idT][idh+lowestpotentialindex]

        res = []
        for ir in range(nh):
            if svalues[ir][idT] < sigmaCut and res == []:
                res = ir

        area[idT] = xvalues[res][idT]
    
        
        sticky[idT],punbound[idT] = computeStickyParameters(allhs,phi,heightCut2,lowestpotentialindex,gravHeight)

        # now actually figure out for each height the N in contact 
        # i e the maximum number of potential bonds at any temperature
        # this will be done as a weighted average over distances
        # at each distance you interact with a maximum number of bonds
        
        Nheights = np.zeros(nh)
        sigmaMax = np.amax(Allsigmaabs)
        
        cutoff = 0.01
        isfound = 0

        cutoff = cutoff/2
        
        endid = len(allhs)-1
        for idh in range(nh):
            if Allsigmaabs[idT][idh] < cutoff*sigmaMax and isfound == 0:
                endid = idh
                isfound = 1
                

        sigmaHeights =  np.flip(Allsigmaabs[idT][0:endid])
        Heights = np.flip(allhs[0:endid])
        xs, indices = np.unique(sigmaHeights, return_index=True)
        hs = Heights[indices]
            
        if endid == 0:
            Npotential[idT] = 0

        else:
                
            #print(xs)
            #plt.plot(xs,'+')
            #plt.show()
            
                
                
            heightFunc = InterpolatedUnivariateSpline(xs,hs,k=1)
            
            # sometimes it's not exactly clear what this cutoff should be exactly, and for large particles its safer to take it to 0.01
            cutoffSigma = 0.01
            
            hmaxConnected = heightFunc(cutoffSigma*sigmaMax)
           
            isfound = 0
            for idh in range(nh):
                if isfound == 0 and Allsigmaabs[idT][idh] < cutoffSigma*sigmaMax: #Nbridge[idT][idh] > 1: 
                     isfound = 1
                     hmaxConnected2 = allhs[idh]
                     
            #print(hmaxConnected, hmaxConnected2)
                     
            for idh in range(nh):
                sigmaMin = Allsigmaabs[idT][idh]
                hMin = allhs[idh]
                isfound = 0
                # I don't think this criteria makes sense. instead I would argue for a another criteria on the fraction of sticky ends
                #if isfound == 0 and Nbridge[idT][lowestpotentialindex] - Nbridge[idT][lowestpotentialindex+idh] > 0.9*Nbridge[idT][lowestpotentialindex] and Nbridge[idT][lowestpotentialindex] > 1:
                # this one is better
                # let's first find how far you can go -- because heights are discrete, you'd better interpolate
                
                #non interpolated version
                # for idh2 in range(nh-idh):
                #     sigmaHeight =  Allsigmaabs[idT][idh+idh2]
                #     # if the max height has not been found yet and the sticky density is such that only 1% is sticky at this height. 
                #     # and if you know that at this height you can find at least some bonds
                    
                #     if isfound == 0 and sigmaHeight < 0.02*sigmaMax and sigmaMin > 0.02*sigmaMax: #Nbridge[idT][idh] > 1: 
                #         isfound = 1
                #         hmaxConnected = allhs[idh+idh2]
                if sigmaMin > cutoffSigma*sigmaMax and abs(Radius - hmaxConnected + hMin) < Radius: #s[idT]
                    if optcolloid:
                        Nheights[idh] = ((Radius**2 - (Radius - hmaxConnected/2 + hMin/2)**2)/Radius**2)*geoFactor
                    else:
                        Nheights[idh] = ((Radius**2 - (Radius - hmaxConnected + hMin)**2)/Radius**2)*geoFactor
                else:
                    Nheights[idh] = 0
    
            Npotentialprobas = [exp(-phi[iv]+phiMin)*Nheights[iv] for iv in range(len(phi))] #boltzmann weights
            Npotential[idT] = sum(Npotentialprobas)/sum(probas)
        # max number of potential bonds
        
        
    return(hMins,hAves,Nconnected,area,phiMins,width,xvalues,svalues,sticky,punbound,deltaGeff,Rconnected,NAves,Npotential)
        
        
    
    
def calculateMicroscopicDetailsDistorted(allhs,potential,Nt,gravityFactors):
    width = np.zeros(Nt)
    phiMins = np.zeros(Nt)
    nh = len(allhs)
    punbound = np.zeros(Nt)
    sticky = np.zeros(Nt)
    depthT = -20

    heightCut2 = 20e-9 #past 20 nm -- this is the resolution that Fan has.  
    
    idTslope = 0
    for idT in range(Nt):
        lowestpotentialindex = np.array(potential[:][idT]).argmin()
        phi = potential[:][idT]
        phiMin = phi[lowestpotentialindex]
        if (phiMin < -3 and idTslope == 0) or idT == Nt-1:
            idTslope = idT
            allheights2, potential2 = CenterData(allhs, phi, 1e9)
            h_adjustment = 0
            h_adjust = 1
            hframe =1 
            gravmins = 40
            gravmaxs = 180
            
            heightCentered, potentialCentered, gravityFactorInitModel = centerPotentialRemoveGravity(allheights2,potential2,0,gravmins,gravmaxs,h_adjust,h_adjustment,hframe)        


    
    for idT in range(Nt):
        lowestpotentialindex = np.array(potential[:][idT]).argmin()
        
        # figure out "average" location of the particle - considered as the weighted average with the potential
        phi = potential[:][idT]
        if np.isnan(phi[0]) == False:
            allheights2, potential2 = CenterData(allhs, phi, 1e9)
            hframe = 1.5
            ghere = gravityFactorInitModel*gravityFactors[idT]/gravityFactors[idTslope]
            phiMin = phi[lowestpotentialindex]
            h_adjustment = 0
            h_adjust = 1
            if (phiMin < -2):
                gravmins = 40
                gravmaxs = 180
            else:
                gravmins = 50
                gravmaxs = 200
            #print(ghere)
            heightCentered, potentialCentered, gravityFactor = centerPotentialRemoveGravityKnownExperiment(allheights2,potential2,0,gravmins,gravmaxs,h_adjust,h_adjustment,hframe,ghere)    

            gravHeight = 1/gravityFactor*1e9 #gravitational Height in nanometers

            hidmax = 0
            idh = 0
            while hidmax == 0 and idh < nh:
                if heightCentered[idh] > 30:
                    hidmax = idh
                idh += 1
            if hidmax == 0:
                hidmax = nh-1
                
                
            lowestpotentialindexgravityremoved = np.array(potentialCentered[0:hidmax]).argmin()
            
            phiMin = phi[lowestpotentialindex]
            depthT = potentialCentered[lowestpotentialindexgravityremoved]
            phiMins[idT] = depthT
            # btw compute width of potential
            res1 = 0
            res2 = 0
            # width of the potential corresponding to a 5kT fluctuation
            for ir in range(nh):
                if phi[ir] < phiMin+5 and res1 == 0:
                    res1 = ir
            for ir in range(nh):
                #the other limit has to be found going down, to avoid double potential
                if phi[nh-1-ir] < phiMin+5 and res2 == 0:
                    res2 = nh-1-ir
            #print(res1,res2)
            width[idT] = allhs[res2] - allhs[res1]
            
            sticky[idT],punbound[idT] = computeStickyParameters(allhs,phi,heightCut2,lowestpotentialindex,gravHeight)
        else:
            phiMins[idT] = -float('inf')
            
        
    return(phiMins,width,sticky,punbound)


    
def calculateMicroscopicDetailsExperiment(allhs,potential,gravh):
    
    #heightCut2 = 10000e-9 #past 20 nm -- this is the resolution that Fan has.  
    lowestpotentialindex = np.array(potential).argmin()
    #print(allhs[-1])
    sCutoff,sNoCutoff,FWHM,width2k = computeWidth(allhs,potential,lowestpotentialindex,gravh*1e9)
        
        
    return(sCutoff,sNoCutoff,FWHM,width2k)

def computeWidth(allhs,phi,idMin,gravHeight):

    weights = [exp(-phiv+phi[idMin]) for phiv in phi]
    ftotfit = InterpolatedUnivariateSpline(allhs,weights, k=1)
    
    
    weightsx = [(allhs[id]-allhs[idMin])**2*exp(-phi[id]+phi[idMin]) for id in range(len(allhs))]
    ftotfitx = InterpolatedUnivariateSpline(allhs,weightsx, k=1)
    
    xsquare = sqrt(sum(weightsx)/sum(weights))
    
    weightsh = [(allhs[id]-allhs[idMin])*exp(-phi[id]+phi[idMin]) for id in range(len(allhs))]
    ftotfith = InterpolatedUnivariateSpline(allhs,weightsh, k=1)
    
    
    #different methods to calculate widths.. 
    res = 0
    res2 = 0
    threshh = allhs[idMin] + 20e-9 #cutoff method
    
    thresh2 = allhs[idMin] + 600e-9 #600e-9 #min(gravHeight*1e-9*7,max(allhs)) # I changed this recently it used to be   
    endPlot = 250e-6
    # for Fan's data, it doesn't change much if the boundary is between 3 to 6*gravitational Height, but it changes otherwise. 
    # I think it's also because beyond 600nm I'm not resolving things super finely. 
     

    for ih in range(len(allhs)):
        if allhs[ih] > threshh and res == 0:
            res = ih
        if allhs[ih] >= thresh2 and res2 == 0:
            res2 = ih

    if res2 == 0:
        if allhs[-1] > 100e-9:
            #res2 = len(allhs)-1
            # but maybe actually this can be improved
            idStart = 0
            idMax = 0
            hStart = 80e-9
            for ih in range(len(allhs)):
                if allhs[ih] > hStart and idStart == 0:
                    idStart = ih
                if allhs[ih] > 200e-9 and idStart > 0 and idMax == 0:
                    idMax = ih
            xFit = allhs[idStart:idMax]
            yFit = phi[idStart:idMax]
            polyResult = np.polyfit(xFit,yFit,1)
            p = np.poly1d(polyResult)
            xFill = np.linspace(hStart,600e-9,100)
            yFill = p(xFill)
            #plt.show() 
    
            #plt.plot(allhs,phi,'.')
            phi = np.concatenate((phi[0:idStart-1],yFill),axis=None)
            allhs = np.concatenate((allhs[0:idStart-1],xFill),axis=None)
    
            
            
            #plt.plot(xFill,yFill)
            #plt.show()    
            
            weights = [exp(-phiv+phi[idMin]) for phiv in phi]
            ftotfit = InterpolatedUnivariateSpline(allhs,weights, k=1)
        
            res2 = 0
            for ih in range(len(allhs)):
               if allhs[ih] >= thresh2 and res2 == 0:
                    res2 = ih   
                #endPlot = 250e-6
        else:
            res = len(allhs)-1
#    if res == 0:
#        res = len(allhs)-1
        #endPlot = 250e-6
        

    
    #print(allhs[0],allhs[res],phi[idMin],allhs[idMin])
    #print(res,res2,allhs[res],allhs[res2])
    
    sticky1 = ftotfit.integral(allhs[0],allhs[res]) #divide by sqrt(2pi) such that this is really the curvature radius
    #print(sticky1,sticky)
    
    sticky = ftotfit.integral(allhs[0],thresh2) #divide by sqrt(2pi) such that this is really the curvature radius
    stickyX = ftotfitx.integral(allhs[0],thresh2) #divide by sqrt(2pi) such that this is really the curvature radius
    stickyH = ftotfith.integral(allhs[0],thresh2) #divide by sqrt(2pi) such that this is really the curvature radius
    
    #print(sticky1,sticky)
    if gravHeight > 0 and allhs[-1] > 100e-9:
        hgrav = gravHeight*1e-9
        #print(hgrav)
        addups = hgrav*exp(phi[idMin])*(-exp(-endPlot/hgrav) + exp(-(thresh2)/hgrav))
        def addh(x):
            return(- ((x-allhs[idMin])*hgrav + hgrav**2)*exp(-x/hgrav))
        
        #print(addh(endPlot)-addh(thresh2))
        addupsH = (addh(endPlot) - addh(thresh2))*exp(phi[idMin])
        
        def addx(x):
            return((-(x-allhs[idMin])**2*hgrav - 2*(x-allhs[idMin])*hgrav**2 - 2*hgrav**3)*exp(-x/hgrav))
        
        addupsX = (addx(endPlot) - addx(thresh2))*exp(phi[idMin])
        
        
            #norm2 = ftotfit.integral(allhs[0],allhs[-1])/sqrt(2*pi) #also divide normalization by sqrt(2pi)
    else:
        addups = 0
        addupsX = 0
        addupsH = 0
    #print(addups)
    #print(stickyX,sticky,addupsX,addups)
    xsquare = sqrt((stickyX+addupsX)/(sticky+addups))
    #print(stickyH,addupsH)
    #print(stickyH+addupsH)
    hsquare = ((stickyH+addupsH)/(sticky+addups))
    sticky = sticky + addups
    
    # now you can also do the same thing with gravity removed right so 
    
    # gravmins = 40
    # gravmaxs = 180
    # h_adjust = 0
    # h_adjustment = 0
    # hframe = 1
    # allheights2, potential2 = CenterData(allhs, phi, 1e9)
    
    # heightCentered, potentialCentered, gravityFactor = centerPotentialRemoveGravityKnownExperiment(allheights2,potential2,0,gravmins,gravmaxs,h_adjust,h_adjustment,hframe,1/hgrav)

    # idMin = np.array(potentialCentered).argmin()
    
    # weights = [exp(-phiv+potentialCentered[idMin]) for phiv in potentialCentered]
    # ftotfit = InterpolatedUnivariateSpline(allhs,weights, k=1)
    
    # sticky1removed = ftotfit.integral(allhs[0],allhs[res]) #divide by sqrt(2pi) such that this is really the curvature radius
    
    # if gravHeight > 0 and allhs[-1] > 100e-9:
    #     hgrav = gravHeight*1e-9
    #     addups = hgrav*exp(potentialCentered[idMin])*(-exp(-endPlot/hgrav) + exp(-(allhs[res2]-allhs[idMin])/hgrav))
        
    #         #norm2 = ftotfit.integral(allhs[0],allhs[-1])/sqrt(2*pi) #also divide normalization by sqrt(2pi)
    # else:
    #     addups = 0
    # sticky1removed = sticky1removed + addups
    
    # and you can do the half width thing
    # (once gravity is removed... )
    
    
    phiMin = phi[idMin]
    phiMax = phiMin/2
    #print(phiMin)
    id1 = 0
    id2 = 0
    for ih in range(len(allhs)):
        if phi[ih] < phiMax and id1 == 0:
            id1 = ih
        if phi[ih] > phiMax and id1 > 0 and id2 == 0:
            id2 = ih
            
    FWHM = allhs[id2] - allhs[id1]
    
    
    
    id1 = 0
    id2 = 0
    x = 2
    for i in range(len(allhs)):
        if id1 == 0 and phi[i] < phiMin + x: #first value below xkT
            id1 = i
        if id1 > 0 and phi[i] > phiMin + x and id2 == 0:
            id2 = i
            
    width2k = allhs[id2] - allhs[id1]

    
    return(xsquare,hsquare,FWHM, width2k)

   
def extractStatsFromData(allheights, potential, GV, thresh, hslab):
    
    allhs = np.flip(allheights)
    phi = np.flip(potential)
    nh = len(allhs)
    href = 20
    phiGrav = [phi[idh] + GV*(allhs[idh]+href)*1e-9  for idh in range(nh)]
    idMinp = np.array(phiGrav).argmin()
    
    

    weights = [exp(-phiGrav[idh] +phiGrav[idMinp]) for idh in range(nh)]
    ftotfit = InterpolatedUnivariateSpline(allhs,weights, k=1)
    

    sticky = ftotfit.integral(allhs[0],thresh)
    
    hgrav =  1e9/GV*(exp(-GV*(thresh+href)*1e-9+phiGrav[idMinp]) - exp(-GV*hslab*1e-9+phiGrav[idMinp]))

    punbound = hgrav/(hgrav+sticky)
    print(punbound)

    return(sticky,punbound)        
 
def calculateRateMelting(allheights,potential,D,time,Ntemp,temperatures,Tmin,Tmax):
    punbound = np.zeros(Ntemp)
    heightCut2 = 30e-9 #past 20 nm -- this is the resolution that Fan has. 
    for idT in range(Ntemp):
        T = temperatures[idT]
        if T > Tmin and T < Tmax:
        
            phi = potential[:][idT]
            allheights2, potential2 = CenterData(allheights, phi, 1)    
            
            weights1 = [exp(-p2) for p2 in potential2]
            int1 = InterpolatedUnivariateSpline(allheights2,weights1, k=1)
            
            weights2 = np.zeros(len(potential2))
            #potmax = np.max(potential2)
            indmin = np.array(potential2).argmin() 
            
            indmax = 0
            for ih in range(len(allheights2)):
                if indmax == 0 and allheights2[ih] > heightCut2:
                    indmax = ih
            
            for ih in range(indmin,indmax+1):
                weights2[ih] = exp(potential2[ih])*int1.integral(allheights2[0],allheights2[ih])      
            
            
            
            int2 = InterpolatedUnivariateSpline(allheights2[indmin:indmax+1],weights2[indmin:indmax+1], k=1)
            
            print(allheights2[indmin],allheights2[indmax])
            Tmfpt = 1/D*int2.integral(allheights2[indmin],allheights2[indmax])
            
            print(Tmfpt)
            
            #def ptime(t):
            #    return(1/Tmfpt*exp(-t/Tmfpt))
            
            #punbound[idT] = quad(ptime,0,time)
            punbound[idT] = 1 - exp(-time/Tmfpt)
            print(punbound[idT])
        elif T>=Tmax:
            punbound[idT] = 1
            
    print(punbound)
    return(punbound)
                

def createMatrixPropagator(Nx,dt,dx,dxPhi,D):
    
    P = np.zeros((Nx,Nx))
    # this is the L generator of an associated FPE 
    
    # let's 1st fill in the matrix core
    for i in range(1,Nx-1):
        P[i][i] = 1 - 2*D*dt/dx[i]**2
        P[i][i+1] = + D*dt/(dx[i]**2)
        P[i][i-1] = + D*dt/(dx[i]**2)
        # for the flux term you need to make sure to propagate upstream/downstream
        u = D*dxPhi[i]
        if u > 0:
            #then advect what's before
            P[i][i] += - u*dt/((dx[i]+ dx[i-1])/2)*1
            P[i][i-1] += - u*dt/((dx[i]+ dx[i-1])/2)*(-1)
        else:
            #then advect what's after
            P[i][i] += - u*dt/((dx[i]+ dx[i+1])/2)*(-1)
            P[i][i+1] += - u*dt/((dx[i]+ dx[i+1])/2)*(1)
    
    #let's fill in the boundary conditions, here we have only no flux at the left boundary
    
    P[0][1] = 1 #(-1/dx[0])/(dxPhi[0] - 1/dx[0]) - I think that's actually the right BC
        
    # f = 0 at the right boundary so no need to do anything there.
    
    return(P)

def createInvertedMatrixPropagator(Nx,dt,dx,dxPhi,D):
    
    P = np.zeros((Nx,Nx))
    # this is the L generator of an associated FPE 
    
    # let's 1st fill in the matrix core
    for i in range(1,Nx-1):
        P[i][i] = 1 + 2*D[i]*dt/dx[i]**2
        P[i][i+1] = - D[i]*dt/(dx[i]**2)
        P[i][i-1] = - D[i]*dt/(dx[i]**2)
        # for the flux term you need to make sure to propagate upstream/downstream
        u = D[i]*dxPhi[i]
        if u > 0:
            #then advect what's before
            P[i][i] += + u*dt/((dx[i]+ dx[i-1])/2)*1
            P[i][i-1] += + u*dt/((dx[i]+ dx[i-1])/2)*(-1)
        else:
            #then advect what's after
            P[i][i] += + u*dt/((dx[i]+ dx[i+1])/2)*(-1)
            P[i][i+1] += + u*dt/((dx[i]+ dx[i+1])/2)*(1)
    
    #let's fill in the boundary conditions, here we have only no flux at the left boundary
    
    P[0][0] = 1 #no flux will be implemented later so actually don't care about it now
        
    # f = 0 at the right boundary so no need to do anything there.
    P[-1][-1] = 1


    Pmat = np.linalg.inv(P)
    

    
    return(Pmat)


def calculateRateMeltingIntegral(allheights,potential,D,time,Ntemp,temperatures,Tmin,Tmax,Radius):
    punbound = np.zeros(Ntemp)
    heightCut2 = 20e-9 #past 20 nm -- this is the resolution that Fan has. 
    for idT in range(Ntemp):
        T = temperatures[idT]
        if T > Tmin and T < Tmax:
            NT = 100000
            dt = time/NT
            
            phi = potential[:][idT]
            allheights2, potential2 = CenterData(allheights, phi, 1)    
            
            indmin = np.array(potential2).argmin()             
            indmax = 0
            for ih in range(len(allheights2)):
                if indmax == 0 and allheights2[ih] > heightCut2:
                    indmax = ih
            
            imin = max(indmin-20,1)
            dxphi = [(phi[i+1] - phi[i-1])/(allheights2[i+1]-allheights2[i-1]) for i in range(imin,indmax+1)]
            dx = [(allheights2[i+1]-allheights2[i-1])/2 for i in range(imin,indmax+1)]
            Nx = indmax+1-imin
            
            G = np.zeros(Nx)
            G += 1
            G[-1] = 0
            #Pmat = createMatrixPropagator(Nx,dt,dx,dxphi,D)
            
            lowestpotentialindex = np.array(potential[idT][imin:indmax]).argmin()
            #print(allheights[lowestpotentialindex])
            Dh = np.zeros(Nx)
            def fhydro(h):
                return((6*h**2+2*Radius*h)/(6*h**2 + 9*Radius*h +2*Radius**2))
            
            for i in range(Nx):
                #if i > lowestpotentialindex-15:
                #    Dh[i] = D*fhydro((allheights[i+imin] - allheights[imin+lowestpotentialindex-15]))
                # simplify the expression for D to be constant because it's a reasonable simplification
                Dh[i] = D
            #print(Dh)        

            Pmat = createInvertedMatrixPropagator(Nx,dt,dx,dxphi,Dh)

            
            
            for it in range(1,NT):
                G = np.matmul(Pmat,G)
                G[0] = G[1] # implement no flux at the left boundary
                #if it % 50000== 0:
                #    plt.plot(G)
                #    plt.show()
                    
            punbound[idT] = 1 - G[indmin-imin] #take the unbound particles to have started at the minimum point
            
            print(punbound[idT])
        elif T>=Tmax:
            punbound[idT] = 1
            
    print(punbound)
    return(punbound)
    
def calculateMeltingTkinetic(allheights,potential,D,time,Ntemp,temperatures,Tmin,Tmax):
    # this function was used to see if the direct resolution was in accordance with the results of a stochastic integral
    NT = 10000000 #time discretization
    dt = time/NT
    N = 100 #number of particles
    
    dW = sqrt(2*D*dt)*randn(N,NT)
    X = np.zeros((Ntemp,N))
    heightCut2 = 30e-9
    counts = np.zeros(Ntemp)
    dxPhi = np.zeros((Ntemp,len(allheights)-1))
    hx = np.zeros((Ntemp,len(allheights)-1))
    for idT in range(Ntemp):
        phi = potential[:][idT]
        allheights2, potential2 = CenterData(allheights, phi, 1)    
        dxPhi[:][idT] = [-(phi[i+1]-phi[i])/(allheights2[i+1]- allheights2[i]) for i in range(len(allheights)-1)]
        hx[:][idT] = [(allheights2[i+1] + allheights2[i])/2 for i in range(len(allheights)-1)]
    friction = D
    for it in range(NT):
        if (it % 100000 == 1):
            print('achieved 1 percent more', it)
            for idT in range(Ntemp):
                for ix in range(N):
                    if X[idT][ix] > heightCut2:
                        X[idT][ix] = float('nan')
                        counts[idT] +=1

            
        for idT in range(Ntemp):
            T = temperatures[idT]
            if T > Tmin and T < Tmax:
                
                force = InterpolatedUnivariateSpline(hx[:][idT],dxPhi[:][idT], k=1)

                
                X[:][idT] = [  X[idT][ix] + friction*force(  X[idT][ix] )*dt + dW[ix][it] for ix in range(N)]
                
                            
            elif T>=Tmax:
                counts[idT] = N
                
    punbound = [c/N for c in counts]
    
    return(punbound)
            
            
            