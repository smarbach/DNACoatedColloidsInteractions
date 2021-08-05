#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:48:52 2020

@author: sophie marbach
"""
import math
import numpy as np
from CommonToolsModule import DerjaguinIntegral
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.integrate import trapz
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from math import log , exp, sqrt, inf, erf
pi = math.pi
import matplotlib.pyplot as plt
from StericModule import eMilnerBridge
from FiniteStretchingModule import loadStretchedProfile



import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

#import cmocean
import pickle



# Constants
Na = 6.02*10**23
Rg = 8.314 #J/(mol.K); #perfect gas constant
Tbase = 273.15
kB = Rg/Na; #boltzmann constant


def compute_DNAbindingEnergy(Sequence,saltConcentration): 
   
    #SequenceIn = "T"+Sequence
    # no need to include the T at the beginning
    
    # Sequence example = "ATGCAT"
    # Returns the enthalpy and entropy of binding for a DNA sequence
    # Parameters are from the unified santa lucia model
    H=0
    for i in range(len(Sequence)-1):
        j=Sequence[i:i+2]
        #print(j)
        dH = 0
        if j=='AA'or j=='TT': dH=-7.9  #7.9 was old
        elif j=='AT'or j=='TA': dH=-7.2  
        elif j=='CA' or j=='TG' : dH=-8.5
        elif j=='GT' or j== 'AC' : dH=-8.4     # somehow the santaLuccia parameters are not symmetric
        elif j=='CT' or j== 'AG': dH=-7.8
        elif j=='GA' or j== 'TC' : dH=-8.2
        elif j=='CG': dH=-10.6
        elif j=='GC': dH=-9.8
        elif j=='GG'or j=='CC': dH=-8
        H=H+dH

    # penalty correction for terminal sections
    if Sequence[0]=='G'or Sequence[0]=='C' : H=H+0.1
    elif Sequence[0]=='T'or Sequence[0]=='A': H=H+2.3
        
    if Sequence[-1]=='G'or Sequence[-1]=='C' :H=H+0.1 
    elif Sequence[-1]=='T'or Sequence[-1]=='A': H=H+2.3 
    
    S=0
    complementarity = 0
    selfcomplementary = False   #check if the sequence is self complementary
    if (len(Sequence)% 2) == 0: # if the sequence has an even number of bases it may be complementary
        complementarity = 1
        for i in range(int(len(Sequence)/2)-1):
            base = Sequence[i]
            baseOpposite = Sequence[len(Sequence) -1 - i]
            if base=='A' and baseOpposite!='T':
                complementarity = 0
            elif base=='T' and baseOpposite!='A':
                complementarity = 0
            elif base=='G' and baseOpposite!='C':
                complementarity = 0
            elif base=='C' and baseOpposite!='G':
                complementarity = 0
        if complementarity:
            selfcomplementary = True
            
    #print(complementarity)
    
    if selfcomplementary == True:
        SC = -1.4
    else: SC = 0

    S=S+SC
    
    for i in range(len(Sequence)-1):
        j=Sequence[i:i+2]
        dS = 0
        if j=='AA'or j=='TT': dS=-22.2 #22.2
        elif j=='AT': dS = -20.4
        elif j=='TA': dS = -21.3
        elif j=='CA' or j=='TG': dS=-22.7
        elif j=='GT' or j=='AC': dS=-22.4
        elif j=='CT' or j=='AG': dS=-21.0
        elif j=='GA' or j=='TC': dS=-22.2
        elif j=='CG': dS=-27.2
        elif j=='GC': dS=-24.4
        elif j=='GG'or j=='CC': dS=-19.9
        S=S+dS


    if Sequence[0]=='G'or Sequence[0]=='C' : S=S-2.8 # i'm not so sure about these
    elif Sequence[0]=='T'or Sequence[0]=='A': S=S+4.1
        
    if Sequence[-1]=='G'or Sequence[-1]=='C' :S=S-2.8
    elif Sequence[-1]=='T'or Sequence[-1]=='A': S=S+4.1
    #print('S is',S)
    
    S=S+0.368*(len(Sequence)-1)*np.log(saltConcentration)

    bindingEnergies = [H*4.184*10**3,S*4.184] #90 C (for the 6 bases)

    # for uncertainty measurements
    #bindingEnergies = [H*4.184*10**3*1.01,S*4.184] #94 C +4
    #bindingEnergies = [H*4.184*10**3,S*4.184*1.01] #86 C -4
    #bindingEnergies = [H*4.184*10**3,S*4.184*0.99] #94 C +4
    #bindingEnergies = [H*4.184*10**3*0.99,S*4.184] #86 C -4 (typically results in a 4degree change)


    print("Energy Factors Computed with Santaluccia", H, S)
    print("Bare melting T here is", bindingEnergies[0]/bindingEnergies[1] - 273.15) 
    # numbers where checked to be coherent with Unafold predictions
    #print(bindingEnergies)
    return(bindingEnergies)
    
def Kab(sigmaSticky, h, h1eqin, h2eqin, DeltaG0, hbond,N1,s1,ell1,N2,s2,ell2, h1c,h2c,mushroomFlag, porosity): 
    
    Q1h = 1
    Q2h = 1
    
    #approach a la Daan 1
    if mushroomFlag:
        intermingling = 'transparent'
        if intermingling == 'transparent':
        #for this one we will assume that the mushroom brushes do intermingle completely... 
        # which is not completely true but let's start like that -- later we can also add some penetration length scale
        # here the model is made for symmetric chains
        
            Req1 = h1eqin
            Req2 = h2eqin
            #print('Radius of the brush is',Req*1e9)
            
            Paint = erf(sqrt(3/2)*h/Req1)
            Pbint = erf(sqrt(3/2)*h/Req2)
            #Pabfac = 1/2*(3/(pi*Req**2))**(3/2)*exp(-3*h**2/(2*Req**2))*erf(sqrt(3/4)*h/Req)
            # then integrate exp(-3/4 rab^2/Req^2)*2*pi*rab drab on 0 Infinity 
            #Pabint = Pabfac*4*pi/3*Req**2
            
            Req = sqrt(Req1**2 + Req2**2)
            Pabint = sqrt(6/(pi*Req**2))*(erf(sqrt(3/2)*h*Req1/(Req2*Req)) + erf(sqrt(3/2)*h*Req2/(Req1*Req)))*exp(-3/2*h**2/Req**2)
            
            #value = (sigmaSticky*exp(-DeltaG0)*exp(-3*factorh/(8))*sqrt(12/(2*pi*Rtotr**2))*erf(sqrt(3*factorh/(8)))/((erf(sqrt(3*factorh/(2))))**2))/(Na*10**3)
            value = sigmaSticky*exp(-DeltaG0)*Pabint/(Paint*Pbint)/(Na*10**3)
            
            Kab_h = value
        elif intermingling == 'withPressure':
            lint = 10e-9
            heff = 1e-9
            
            Req = h1eqin
            
            Paint = erf(sqrt(3/2)*h/Req)
            Pbint = Paint
            Pabfac = 1/2*(3/(pi*Req**2))**(3/2)*exp(-3*h**2/(2*Req**2))*erf(sqrt(3/4)*h/Req)
            # then integrate exp(-3/4 rab^2/Req^2)*2*pi*rab drab on 0 Infinity 
            Pabint = Pabfac*4*pi/3*Req**2
            
            #value = (sigmaSticky*exp(-DeltaG0)*exp(-3*factorh/(8))*sqrt(12/(2*pi*Rtotr**2))*erf(sqrt(3*factorh/(8)))/((erf(sqrt(3*factorh/(2))))**2))/(Na*10**3)
            value = sigmaSticky*exp(-DeltaG0)*Pabint/(Paint*Pbint)/(Na*10**3)
            
            Kab_h = value
            
        #print("Calculating hybridization with mushroom potential")
    ######### NORMAL MWC 
    else:
        
        ellaprox = (ell1+ell2)/2
        Naprox = (N1+N2)/2
        saprox = (s1+s2)/2
        
        
        interpenetLength = (h/ellaprox)**(-0.18)*(saprox*ellaprox**2)**(-0.15)*ellaprox*Naprox**(0.51)
        #print("the typical interpenetration length is", interpenetLength*1e9)
        
        #formula 111 of the review on polymers
        #print("interpentration length at this distance", interpenetLength)
        # the values obtained here are of the order of 15nm-3nm for 34k according to how much the polymer is compressed. That seems ok. 
        
        # define sort of "equivalent" penetration lengths
        if porosity > 0:
            
            if h1eqin + h2eqin >= (h+hbond) + 2*porosity:
                #print(h1eqin,h2eqin)
                #expo = max(1.00001,(1 + 0.3**(((h1eqin+h2eqin)/(h+hbond)-1.3)*20+1)))
                expo = 1
                h1eq = expo*h1c*((h+hbond + 2*porosity)/(h1c+h2c))
                h2eq = expo*h2c*((h+hbond + 2*porosity)/(h1c+h2c))
                # print("applying porosity",h1eq,h2eq)
                
                
            
            else:
            
                h1eq = h1eqin
                h2eq = h2eqin
        else:
            
            h1eq = h1eqin
            h2eq = h2eqin
        
        
        # now define the intermingling equivalent reaction constant
        intermingling = 'withPressingCompression' #can be 'withPressing' or 'transparent' or 'withTail'
        
        if intermingling == 'transparent':
            # approach with the density distribution
            
            
            interactionLength = 0
            dh = -h+h1eq+h2eq
            if dh > 0: #if there is touching of the brushes
                #interactionLength = min(0.05*h,dh) # take 10% 
                #minh = 0.001e-9
                #maxh = 5e-9
                #Lh = 5e-9
                
                #interactionLength = min(dh,max(minh,maxh*(1 - dh*(maxh-minh)/maxh/Lh)))   # (or 1 nm) typical penetration length in each other 
                #print(interactionLength)
                interactionLength = min(h1c,h2c,interpenetLength,dh)
            
            #then actually the brushes are larger than expected
            
            h1e = min(h1c + h1c*(interactionLength)/(h1c+h2c), h1eq)
            h2e = min(h2c + h2c*(interactionLength)/(h1c+h2c),h2eq)
            
            def crosssection(z):
                #account for a penalty when they bind, this is actually too much
                #e1 = eMilnerBridge(z,s1,ell1,N1,h1eq)
                #e2 = eMilnerBridge(h - z + hbond,s2,ell2,N2,h2eq)
                #facz = exp(-e1-e2)
                facz = 1.0
                return( z*(h1e**(2) - z**2)**(1/2)*(h - z + hbond)*(h2e**2 - (h - z + hbond)**2)**(1/2)*facz)
            
            if h+hbond <= h2e or h+hbond <= h1e: #usually the top layer is the biggest one
                
                ####### THIS FIRST SET IS BASICALLY SQUISHING THE LAYERS AND KEEPING TAKING MORE AND MORE BONDS IN CONTACT AS THEY ARE PRESSED TOGETHER
                intmin = h-h2e+hbond
                intmax = h1e
        
                if h+hbond <= h2e:
                    # the top layer is being compressed
                    intmin = 0.0
        
                if h+hbond <= h1e:
                    intmax = h+hbond
        
                
                    # approach with the density distribution
                def crosssection1(z):
                    #e1 = eMilnerBridge(z,s1,ell1,N1,intmax)
                    #e2 = eMilnerBridge(h - z + hbond,s2,ell2,N2,intmin)
                    #facz = exp(-e1-e2)
                    facz = 1.0
                    return( z*(intmax**(2) - z**2)**(1/2)*(h - z + hbond)*((h+hbond-intmin)**2 - (h - z + hbond)**2)**(1/2)*facz)
                
                # then the odds of binding are complete
                crosss,err = quad(crosssection1,intmin,intmax)
                Kab_h = sigmaSticky*crosss*3/(intmax**3)*3/((h+hbond-intmin)**3)*exp(-DeltaG0)/(Na*10**3)
                
                #Kab_h = sigmaSticky*exp(-DeltaG0)/(Na*10**3) #this is wrong
            elif h+hbond <= h1e+h2e:
                # if they partially overlap
                crosss,err = quad(crosssection,h-h2e+hbond,h1e)
                Kab_h = sigmaSticky*crosss*3/(h1e**3)*3/(h2e**3)*exp(-DeltaG0)/(Na*10**3) #now this has the right dimensions.
            else:
                Kab_h = 0
        elif intermingling == 'withPressing':
            
            dh = -h+h1eq+h2eq
            if dh > 0: #if there is touching of the brushes
                #interactionLength = min(0.05*h,dh) # take 10% 
                interactionLength = min(h1c,h2c,interpenetLength,dh)   # (or 1 nm) typical penetration length in each other 
                
            
            # h1e = h1c*(h+interactionLength)/(h1c+h2c)
            # h2e = h2c*(h+interactionLength)/(h1c+h2c)
            # epsilon = 1e-12
            # this is the full concentration profile when the layers are pressed against each other. 
            def concentrationProfile(z0,heq,hmax):
                # print(z0,heq,hmax)
                return(3/(2*heq**3)*(2*z0*sqrt(hmax**2 - z0**2) + ((2/3*heq**3/hmax + (hmax**2)/3) - hmax**2)*z0/sqrt(hmax**2 - z0**2)))
            
            def crosssectionPressing(z):
                #interaction crosssection
                def c1(z):
                    # print('c1')
                    #print(z,h1eq,h1c)
                    if z > h1c:
                        print("problem")
                    return(concentrationProfile(z,h1eq,h1c))
                
                def c2(z):
                    # print('c2')
                    #print(h-z,h2eq,h2c)
                    return(concentrationProfile(h-interactionLength-z,h2eq,h2c))
                
                return( c1(z)*c2(z) )
            
            if h < h1eq+h2eq:
                # if they partially overlap
                # print(h1c+h2c,h)
                crosss,err = quad(crosssectionPressing,h-interactionLength-h2c,h1c)
                Kab_h = sigmaSticky*crosss*exp(-DeltaG0)/(Na*10**3) #now this has the right dimensions.
            else:
                Kab_h = 0
        elif intermingling == 'withPressingCompression':
            
            
            
            
            
            interactionLength = 0
            dh = -h+h1eq+h2eq
            if dh > 0: #if there is touching of the brushes
                #interactionLength = min(0.05*h,dh) # take 10% 
                #minh = 0.001e-9
                #maxh = 5e-9
                #Lh = 5e-9
                
                #interactionLength = min(dh,max(minh,maxh*(1 - dh*(maxh-minh)/maxh/Lh)))   # (or 1 nm) typical penetration length in each other 
                #print(interactionLength)
                interactionLength = min(h1c,h2c,interpenetLength,dh)
                # print(interactionLength)
            #then actually the brushes are larger than expected
            
            h1e = min(h1c + h1c*(interactionLength)/(h1c+h2c),h1eq)
            h2e = min(h2c + h2c*(interactionLength)/(h1c+h2c),h2eq)
            # h1e = h1c*(h+interactionLength)/(h1c+h2c)
            # h2e = h2c*(h+interactionLength)/(h1c+h2c)
            # epsilon = 1e-12
            # this is the full concentration profile when the layers are pressed against each other. 
            #print("These are heights to calculate bridging")
            #print(h1e*1e9,h2e*1e9,h1c*1e9,h2c*1e9,h1eq*1e9,h2eq*1e9,h*1e9)
            
            def concentrationProfile(z0,heq,hmax):
                #print(z0,heq,hmax)
                # if h/(h1eq+h2eq) > 0.9: 
                #         print(z0,heq,hmax)
                return(3/(2*heq**3)*(2*z0*sqrt(hmax**2 - z0**2) + ((2/3*heq**3/hmax + (hmax**2)/3) - hmax**2)*z0/sqrt(hmax**2 - z0**2)))
            
            def crosssectionPressing(z):
                #interaction crosssection
                def c1(z):
                    # print('c1')
                    #print(z,h1eq,h1c)
                    if z > h1e:
                        print("problem")
                    return(concentrationProfile(z,h1eq,h1e))
                
                def c2(z):
                    # print('c2')
                    
                    return(concentrationProfile(h-z,h2eq,h2e))
                
                return( c1(z)*c2(z) )
            
            if h < h1eq+h2eq:
                # if they partially overlap
                # print(h1c+h2c,h)
                if h-h2e < h1e:
                    
                    crosss,err = quad(crosssectionPressing,h-h2e,h1e)
                    Kab_h = sigmaSticky*crosss*exp(-DeltaG0)/(Na*10**3) #now this has the right dimensions.
                else:
                    Kab_h = 0
            else:
                Kab_h = 0
                
                
            #if you want plots of how they interact, uncomment this part for example 
        
            # if h/(h1eq+h2eq) > 0.52: 
            #     Hdiscretiz1 = np.linspace(0,h1e,5000)
            #     C1 = [concentrationProfile(z,h1eq,h1e)*sigmaSticky/(Na*10**3) for z in Hdiscretiz1]
            #     Hdiscretiz10 = np.linspace(0,h1c,5000)
            #     C10 = [concentrationProfile(z,h1eq,h1c)*sigmaSticky/(Na*10**3) for z in Hdiscretiz10]
            #     Hdiscretiz2 = np.linspace(h-h2e+1e-16,h,2000)
            #     C2 = [concentrationProfile(h-z,h2eq,h2e)*1/(9.5e-9)**2/(Na*10**3) for z in Hdiscretiz2]
            #     Hdiscretiz20 = np.linspace(h-h2c+1e-16,h,2000)
            #     C20 = [concentrationProfile(h-z,h2eq,h2c)*1/(9.5e-9)**2/(Na*10**3) for z in Hdiscretiz20]
            
            #     matplotlib.rcParams.update(
            #         {'font.sans-serif': 'Arial',
            #           'font.size': 8,
            #           'font.family': 'Arial',
            #           'mathtext.default': 'regular',
            #           'axes.linewidth': 0.35, 
            #           'axes.labelsize': 8,
            #           'xtick.labelsize': 7,
            #           'ytick.labelsize': 7,     
            #           'lines.linewidth': 0.35,
            #           'legend.frameon': False,
            #           'legend.fontsize': 7,
            #           'xtick.major.width': 0.3,
            #           'xtick.minor.width': 0.3,
            #           'ytick.major.width': 0.3,
            #           'ytick.minor.width': 0.3,
            #           'xtick.major.size': 1.5,
            #           'ytick.major.size': 1.5,
            #           'xtick.minor.size': 1,
            #           'ytick.minor.size': 1,
            #         })
            #     fig,ax = plt.subplots(figsize=(3.75,2.5))
                
            #     htot = (h1eq+h2eq)*1e9
            #     ax.set_xlim(0, 1)  # outliers only
            #     #ax.set_xlim(22,80)
                
                
                
            #     ax.fill_between([z*1e9/htot for z in Hdiscretiz1],0,C1, facecolor='orchid', # The fill color \
            #                   color='orchid', lw = 1, alpha=0.2)    
            #     ax.plot([z*1e9/htot for z in Hdiscretiz10],C10,'--', \
            #                   color='darkmagenta', lw = 1)          
            #     ax.fill_between([z*1e9/htot for z in Hdiscretiz2],0,C2,facecolor = 'mediumblue', \
            #                   color='mediumblue', lw = 1, alpha = 0.2)                       
            #     ax.plot([z*1e9/htot for z in Hdiscretiz20],C20,'--', \
            #                   color='darkblue', lw = 1)                       
                  
            #     # ax.xaxis.set_major_locator(MultipleLocator(10))
            #     # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #     # ax.xaxis.set_minor_locator(MultipleLocator(5))
            #     # ax.legend(loc='lower right',handlelength=1, fontsize = 6,markerscale = 0.8,labelspacing = 0.5)   
                
            #     ax.set_xlabel('Position between surfaces $z/(h^t_{eq}+h^b_{eq})$')
            #     ax.set_ylabel('Concentration profile \n of tip ends (mol/L)', labelpad = +2)
                
            #     fig.tight_layout()
                
            #     #plt.savefig('ConcentrationProfile/Profileh{}'.format(h) +'.svg', format='svg', transparent = True)
            #     plt.savefig('ConcentrationProfile/Profileh{}'.format(h) +'.pdf', format='pdf')
            #     ax.set_ylim(0, 0.0025)  # outliers only
                
            #     ax.set_xlim(0.10, 0.88)  # outliers only
            #     #plt.savefig('ConcentrationProfile/ProfileCloseUPh{}'.format(h) +'.svg', format='svg', transparent = True)
            #     plt.savefig('ConcentrationProfile/ProfileCloseUPh{}'.format(h) +'.pdf', format='pdf')
                
            #     plt.show()
                   
                
        elif intermingling == 'withTail':
            def concentrationProfile(z0,heq,hmax):
                    # print(z0,heq,hmax)
                return(3/(2*heq**3)*(2*z0*sqrt(hmax**2 - z0**2) + ((2/3*heq**3/hmax + (hmax**2)/3) - hmax**2)*z0/sqrt(hmax**2 - z0**2)))
            
            tail = "finitestretching" # "finitestretching"
            if tail == "mushroom":
                
                
                
                def c1(z):
                        # print('c1')
                        #print(z,h1eq,h1c)
                        
                    return(concentrationProfile(z,h1eq,h1c))
                
                def c2(z):
                        # print('c2')
                        #print(h-z,h2eq,h2c)
                    return(concentrationProfile(z,h2eq,h2c))
                
                dh = -h+h1eq+h2eq
                if dh > 0: #if there is touching of the brushes
                    #interactionLength = min(0.05*h,dh) # take 10% 
                    interactionLength = min(1e-9,dh) 
                else:
                    interactionLength = 0
                def crosssectionPressing(z):
                    #interaction crosssection
                    
                    return( c1(z)*c2(h-interactionLength-z) )
                
                if interactionLength > 0:
                    crosss,err = quad(crosssectionPressing,h-interactionLength-h2c,h1c)
                    Kab_h = sigmaSticky*crosss*exp(-DeltaG0)/(Na*10**3) #now this has the right dimensions.
               
                else:
                    # let's actually do a mushroom tail
                    Req1 = h1eq #h1eq**(-1/3)*ell1**(4/3)*N1**(2/3)
                    Req2 = h2eq #h2eq**(-1/3)*ell2**(4/3)*N2**(2/3)
                    #print('Radius of the brush is',Req*1e9)
                    
                    Paint = erf(sqrt(3/2)*h/Req1)
                    Pbint = erf(sqrt(3/2)*h/Req2)
                    #Pabfac = 1/2*(3/(pi*Req**2))**(3/2)*exp(-3*h**2/(2*Req**2))*erf(sqrt(3/4)*h/Req)
                    # then integrate exp(-3/4 rab^2/Req^2)*2*pi*rab drab on 0 Infinity 
                    #Pabint = Pabfac*4*pi/3*Req**2
                    
                    Req = sqrt(Req1**2 + Req2**2)
                    Pabint = sqrt(6/(pi*Req**2))*(erf(sqrt(3/2)*h*Req1/(Req2*Req)) + erf(sqrt(3/2)*h*Req2/(Req1*Req)))*exp(-3/2*h**2/Req**2)
                    
                    #value = (sigmaSticky*exp(-DeltaG0)*exp(-3*factorh/(8))*sqrt(12/(2*pi*Rtotr**2))*erf(sqrt(3*factorh/(8)))/((erf(sqrt(3*factorh/(2))))**2))/(Na*10**3)
                    value = sigmaSticky*exp(-DeltaG0)*Pabint/(Paint*Pbint)/(Na*10**3)
                    
                    Kab_h = value
                    
     
                    Q1h = Paint
                        

                    Q2h = Pbint
                    
                # if h > 70e-9:
                #     zaxis = np.linspace(0,h1eq,100)
                #     plt.plot(zaxis,[c1(z) for z in zaxis])
                #     zaxis = np.linspace(0,h,100)
                #     plt.plot(zaxis,[2*(3/(2*pi*h1eq**2))**(1/2)*exp(-(3/2)*(z**2)/h1eq**2) for z in zaxis])
                #     plt.show()
                #     plt.pause(2)
                #     zaxis = np.linspace(0,h2eq,100)
                #     plt.plot(zaxis,[c2(z) for z in zaxis])
                #     zaxis = np.linspace(0,h,100)
                #     plt.plot(zaxis,[2*(3/(2*pi*h2eq**2))**(1/2)*exp(-(3/2)*(z**2)/h2eq**2)for z in zaxis])
                #     plt.show()
                #     plt.pause(2)
                
            
            elif tail == "expand":
                xi1 = h1eq**(-1/3)*ell1**(4/3)*N1**(2/3)*pi**2/4/2
                #print(xi1)
                xi2 = h2eq**(-1/3)*ell2**(4/3)*N2**(2/3)*pi**2/4/2
                #print(xi2)
               
                # h1e = h1c*(h+interactionLength)/(h1c+h2c)
                # h2e = h2c*(h+interactionLength)/(h1c+h2c)
                # epsilon = 1e-12
                # this is the full concentration profile when the layers are pressed against each other. 
                
                
                frac = 0.93
                def concentrationProfileWithTail(z0,heq,hmax,xi):
                    # print(z0,heq,hmax)
                    
                    if hmax == heq:
                        frac = 0.95
                    else:
                        frac = 0.95
                    if z0 > hmax*frac: 
                        cbord = concentrationProfile(hmax*frac,heq,hmax)
                        c = cbord*exp(-pi*((z0-hmax*frac)/xi)**(3/2) )
                    else: 
                        c = concentrationProfile(z0,heq,hmax)
                    
                    return(c)
                
                def c1(z):
                        # print('c1')
                        #print(z,h1eq,h1c)
                        
                    return(concentrationProfileWithTail(z,h1eq,h1c,xi1))
                
                def c2(z):
                        # print('c2')
                        #print(h-z,h2eq,h2c)
                    return(concentrationProfileWithTail(z,h2eq,h2c,xi2))
                
                dh = -h+h1eq+h2eq
                if dh > 0: #if there is touching of the brushes
                    interactionLength = min(0.05*h,dh) # take 10% 
                    #interactionLength = min(1e-9,dh) 
                else:
                    interactionLength = 0
                def crosssectionPressing(z):
                    #interaction crosssection
                    
                    return( c1(z)*c2(h-interactionLength-z) )
                
                crosss,err = quad(crosssectionPressing,0,h-interactionLength)
                Kab_h = sigmaSticky*crosss*exp(-DeltaG0)/(Na*10**3) #now this has the right dimensions.
               
                if (h - h1eq+h2eq) > -(h1eq+h2eq)*0.95 and h < 200e-9:
                    
                    # count the relative loss of degrees of freedom upon compression of these specific profiles
                    
                    q1o,err = quad(c1,0,h)
                    q1far,err = quad(c1,0,200e-9)
                    Q1h = min(q1o/q1far,1) #very far distance
                    #print(Q1h,erf(sqrt(3/2)*h/h1eq))
                    
                    q2o,err = quad(c2,0,h)
                    q2far,err = quad(c2,0,200e-9)
                    Q2h = min(q2o/q2far,1) #very far distance
            
                # if h > 70e-9:
                #     zaxis = np.linspace(0,h,100)
                #     plt.plot(zaxis,[c1(z) for z in zaxis])
                #     zaxis = np.linspace(0,h1eq,100)
                #     plt.plot(zaxis,[concentrationProfile(z,h1eq,h1eq) for z in zaxis])
                #     zaxis = np.linspace(0,h,100)
                #     plt.plot(zaxis,[2*(3/(2*pi*h1eq**2))**(1/2)*exp(-(3/2)*(z**2)/h1eq**2) for z in zaxis])
                #     plt.show()
                #     plt.pause(2)
                #     zaxis = np.linspace(0,h,100)
                #     plt.plot(zaxis,[c2(z) for z in zaxis])
                #     zaxis = np.linspace(0,h2eq,100)
                #     plt.plot(zaxis,[concentrationProfile(z,h2eq,h2eq) for z in zaxis])
                #     zaxis = np.linspace(0,h,100)
                #     plt.plot(zaxis,[2*(3/(2*pi*h2eq**2))**(1/2)*exp(-(3/2)*(z**2)/h2eq**2) for z in zaxis])
                #     plt.show()
                #     plt.pause(2)
            elif tail=="finitestretching":
                srcpath = '../../src/'
                #print('this is the top height',h1eq*1e9)
                
                zvaluest,epst = loadStretchedProfile(srcpath,4.0)
                zvaluesb,epsb = loadStretchedProfile(srcpath,1.7)

                #fit the tails with gaussian profiles... 
                xfit = []
                yfit = []
                
                for iz in range(len(zvaluest)):
                    if zvaluest[iz] > 0.95:
                        if epst[iz] > 0:
                            xfit.append(zvaluest[iz])
                            yfit.append(np.log(epst[iz]))
                
                pt = np.polyfit(xfit, yfit, 2)
                
                xfit = []
                yfit = []
                
                for iz in range(len(zvaluest)):
                    if zvaluesb[iz] > 0.95:
                        if epsb[iz] > 0:
                            xfit.append(zvaluesb[iz])
                            yfit.append(np.log(epsb[iz]))
                
                pb = np.polyfit(xfit, yfit, 2)

                epstfunc = InterpolatedUnivariateSpline(zvaluest,epst, k=1) #interpolate.interp1d(zvaluest,epst, kind='linear') # 
                epsbfunc = InterpolatedUnivariateSpline(zvaluesb,epsb, k=1) #interpolate.interp1d(zvaluesb,epsb, kind='linear') # 

                
                def concentrationProfileFiniteStetch(z0,hmax,epsfunc,pfunc,zmax):
                    # print(z0,heq,hmax)
                    zeval = z0/hmax
                    if zeval > zmax:
                        conc = 0
                    elif zeval > 0.95:
                        
                        p = np.poly1d(pfunc)
                        conc = np.exp(p(zeval))*1/hmax  
                    else:    
                        conc = epsfunc(zeval)*1/hmax
                    return(conc)
                
                # intmin = h-h2eq+hbond
                # intmax = h1eq
        
                # if h+hbond <= h2eq:
                #     # the top layer is being compressed
                #     intmin = 0.0
        
                # if h+hbond <= h1eq:
                #     intmax = h+hbond
            
        
                h2max = h2eq
                h1max = h1eq
                if h <= h2eq: #usually the top layer is the biggest one
                
                    ####### THIS FIRST SET IS BASICALLY SQUISHING THE LAYERS AND KEEPING TAKING MORE AND MORE BONDS IN CONTACT AS THEY ARE PRESSED TOGETHER
                    h2max = h
                if h <= h1eq:
                    h1max = h
                    
                
                
                def c1(z):
                        # print('c1')
                        #print(z,h1eq,h1c)
                        
                    return(concentrationProfileFiniteStetch(z,h1max,epstfunc,pt,10/4.0))
                
                def c2(z):
                        # print('c2')
                        #print(h-z,h2eq,h2c)
                    return(concentrationProfileFiniteStetch(z,h2max,epsbfunc,pb,10/1.7))
                
                def crosssectionPressing(z):
                    #interaction crosssection
                    
                    return( c1(z)*c2(h-z) )
                
                crosss,err = quad(crosssectionPressing,0,h)
                Kab_h = sigmaSticky*crosss*exp(-DeltaG0)/(Na*10**3) #now this has the right dimensions.
               
                if (h - h1eq+h2eq) > -(h1eq+h2eq)*0.95 and h < 200e-9:
                    
                    # count the relative loss of degrees of freedom upon compression of these specific profiles
                    
                    q1o,err = quad(c1,0,h)
                    q1far,err = quad(c1,0,200e-9)
                    Q1h = min(q1o/q1far,1) #very far distance
                    #print(Q1h,erf(sqrt(3/2)*h/h1eq))
                    
                    q2o,err = quad(c2,0,h)
                    q2far,err = quad(c2,0,200e-9)
                    Q2h = min(q2o/q2far,1) #very far distance
                    
                    
                Kab_h = Kab_h/(Q1h*Q2h)
            
                # if h > 70e-9:
                    
                #     xi1 = h1eq**(-1/3)*ell1**(4/3)*N1**(2/3)*pi**2/4/2
                #     #print(xi1)
                #     xi2 = h2eq**(-1/3)*ell2**(4/3)*N2**(2/3)*pi**2/4/2
                #     #print(xi2)
                   
                #     # h1e = h1c*(h+interactionLength)/(h1c+h2c)
                #     # h2e = h2c*(h+interactionLength)/(h1c+h2c)
                #     # epsilon = 1e-12
                #     # this is the full concentration profile when the layers are pressed against each other. 
                    
                    
                #     frac = 0.93
                #     def concentrationProfileWithTail(z0,heq,hmax,xi):
                #         # print(z0,heq,hmax)
                        
                #         if hmax == heq:
                #             frac = 0.95
                #         else:
                #             frac = 0.95
                #         if z0 > hmax*frac: 
                #             cbord = concentrationProfile(hmax*frac,heq,hmax)
                #             c = cbord*exp(-pi*((z0-hmax*frac)/xi)**(3/2) )
                #         else: 
                #             c = concentrationProfile(z0,heq,hmax)
                        
                #         return(c)
                    
                #     zaxis = np.linspace(0,h,100)
                #     plt.plot(zaxis,[c1(z) for z in zaxis])
                #     plt.plot(zaxis,[epstfunc(z/h1max)*1/h1max for z in zaxis])
                #     zaxis = np.linspace(0,h1eq,100)
                #     plt.plot(zaxis,[3/(2*h1eq**3)*(2*z0*sqrt(h1eq**2 - z0**2)) for z0 in zaxis])
                #     plt.show()
                #     plt.pause(2)
                #     zaxis = np.linspace(0,h,100)
                #     plt.plot(zaxis,[c2(z) for z in zaxis])
                #     plt.plot(zaxis,[epsbfunc(z/h2max)*1/h2max for z in zaxis])
                #     zaxis = np.linspace(0,h2eq,100)
                #     plt.plot(zaxis,[3/(2*h2eq**3)*(2*z0*sqrt(h2eq**2 - z0**2)) for z0 in zaxis])
                #     plt.show()
                #     plt.pause(2)
    #*exp(-Ebridge)
    
    # it's not clear which model to use here for sure... 
    #phenomelogical approach
    #Kab_h = sigmaSticky*pi*(2.5e-9)**2*exp(-DeltaG0)*exp(-Ebridge)    
    
    #if h > h1eq + h2eq:
    #    Kab_h = 0
    #else:
    #    #Kab_h = sigmaSticky/(Na*10**3)*exp(-DeltaG0)*exp(-Ebridge)*(h1eq+h2eq-h)/(h1eq+h2eq)
    #    r = h/(h1eq+h2eq)
    #    Kab_h = sigmaSticky/(Na*10**3)*exp(-DeltaG0)*exp(-Ebridge)*(1/h1eq**(1/2)* \
    #                         9/sqrt(2)*(r**2 - 1)/(r)**(5/2))*(1/h2eq**(1/2)*(r**2 - 1)/(r)**(5/2))
    
    #pure approach
    #Kab_h = exp(-DeltaG0)*exp(-Ebridge)
        
    #    ######### COMPRESSED MWC 
    # # we can develop yet an even more advanced model. 
    # # as a means of testing its viability for now don't take the actual compressed heights but some proxi
    # #consider that they bind over a "fluctuation lengthscale
    # hfluct = 5e-9
    
    # #if h <= h1eqin+h2eqin :
    # #    h1c = h1eqin/(h1eqin+h2eqin)*h
    # #    h2c = h2eqin/(h1eqin+h2eqin)*h
    # #else:
    # #    h1c = h1eqin
    # #    h2c = h2eqin
    # # approach with the density distribution
    # def epsilon(z,heq,hc):
    #     return(2*z*(hc**(2) - z**2)**(1/2) + (heq**2 - h**2)*z*(hc**2 - z**2)**(-1/2))
    
    # def epsilonbare(z,heq):
    #     return(2*z*(heq**(2) - z**2)**(1/2))
    
    # def crosssection(z):
    #     return( epsilon(z,h1eqin,h1c)*epsilon((h - z + hbond -hfluct),h2eqin,h2c) )

    # def crosssectionbare(z):
    #     return( epsilonbare(z,h1eqin)*epsilonbare((h - z + hbond -hfluct),h2eqin) )

    

    # if h +hbond - hfluct <= h1eqin+h2eqin :
    #     if h + hbond <= h1eqin+h2eqin:
    #     # if they partially overlap
        
    #         crosss,err = quad(crosssection,h-h2eqin+hbond-hfluct,h1eqin)
    #         Kab_h = sigmaSticky*crosss*3/(h1eqin**3)*3/(h2eqin**3)*exp(-DeltaG0)/(Na*10**3)/4 #now this has the right dimensions.
    #     else:
    #         #if they only have a tiny overlap
    #         crosss,err = quad(crosssectionbare,h-h2eqin+hbond-hfluct,h1eqin)
    #         Kab_h = sigmaSticky*crosss*3/(h1eqin**3)*3/(h2eqin**3)*exp(-DeltaG0)/(Na*10**3)/4
            
    # else:
    #     Kab_h = 0    
    
    
    return(Kab_h,Q1h,Q2h)


def sigmaab(Kab_h,fractionSticky,sigmaSticky) :
    # those functions where thoroughly checked...
    f = fractionSticky
    return(1/(2*Kab_h)*(1+(f+1)*Kab_h-sqrt(1+((f-1)*Kab_h)**2+2*Kab_h*(1+f))))

def sigmaabatt(Kab_h,fractionSticky,sigmaSticky) :
    # those functions where thoroughly checked...
    f = fractionSticky
    sabatt= f*Kab_h + (-f - f**2)*Kab_h**2 + (f + 3*f**2 + f**3)*Kab_h**3  \
           -f*(1 + f)*(1 + f*(5 + f))*Kab_h**4  + \
           f*(1 + f*(10 + f*(20 + f*(10 + f))))*Kab_h**5  \
           -f*(1 + f)*(1 + f*(14 + f*(36 + f*(14 + f))))*Kab_h**6
    return(sabatt)

def unboundProbas(Kabs,sa,sb) :
    #inside this function goes the actual Kabs so Kab/sigmaSticky
    # those functions where thoroughly checked...
    unboundPa = (-1-Kabs*sb + Kabs*sa + sqrt((1 + Kabs*sb - Kabs*sa)**2 + 4*Kabs*sa))/(2*Kabs*sa) 
    unboundPb = (-1-Kabs*sa + Kabs*sb + sqrt((1 + Kabs*sa - Kabs*sb)**2 + 4*Kabs*sb))/(2*Kabs*sb)    
    
    return(unboundPa,unboundPb)


def fatt(Kab_h,fractionSticky,sigmaSticky) :
    f = fractionSticky
    if f == 1:
        fattvalues = 1/(2*Kab_h)*(-(1+2*Kab_h) + sqrt(1+2*2*Kab_h) + 2*2*Kab_h*log((1+sqrt(1+2*2*Kab_h))/2))
        fattvalue = fattvalues*sigmaSticky
    else:
        # Here we can make the improvement suggested by Stefano. 
        # Leaving the old code in case
        #def sigmaablambda(lam) :
        #    Kabh = Kab_h*exp(-lam)
        #    if Kabh < 0.0001:
        #        sigmar = sigmaSticky*sigmaabatt(Kabh,fractionSticky,sigmaSticky)
        #    else:
        #        sigmar = sigmaSticky*sigmaab(Kabh,fractionSticky,sigmaSticky)
        #    return(sigmar)
        #fnew,err = quad(sigmaablambda,0,10)
        #lim = 10
        #fold = fnew*2
        #
        #while (abs(fold/fnew-1) > 0.0001):
        #    fold = fnew
        #    lim = lim+10
        #    fnew,err = quad(sigmaablambda,0,lim)

        #fattvalue = fnew   
        #print(fattvalue)
    
        # Stefano's improvement
        # normally here Kab_h is big enough so no need to worry about that. 
        # I checked that this formulation was consistent with the numerical integration. 
        sa = fractionSticky*sigmaSticky
        sb = sigmaSticky
        pa,pb = unboundProbas(Kab_h/sigmaSticky,sa,sb)
        sab = sigmaSticky*sigmaab(Kab_h,fractionSticky,sigmaSticky)
        fattvalue = - sa*log(pa) - sb*log(pb) - sab
        #print(fattvalue)
        
    return(- fattvalue)    

def fattsupply(Kab_h,fractionSticky,sigmaSticky) :
    f = fractionSticky
    if f == 1:
        # make a series expansion when Kab_h approaches 0... 
        fsupplyatt_h = Kab_h - Kab_h**2 + 5/3*Kab_h**3 - 7/2*Kab_h**4 + 42/5*Kab_h**5 - 22*Kab_h**6 
    else:
        fsupplyatt_h = f*Kab_h - 1/2*(f*(1 + f))*Kab_h**2 + 1/3*f*(1 + f*(3 + f))*Kab_h**3 - \
 1/4*(f*(1 + f)*(1 + f*(5 + f)))*Kab_h**4 + \
 1/5*f*(1 + f*(10 + f*(20 + f*(10 + f))))*Kab_h**5 - \
 1/6*(f*(1 + f)*(1 + f*(14 + f*(36 + f*(14 + f)))))*Kab_h**6 
    return(- fsupplyatt_h*sigmaSticky)




def bridgingPotential(allhs,bindingEnergies,hbond,h1eq,h2eq,sb,st,Radius,tetherSeq,saltConcentration,T,s1,ell1,N1,s2,ell2,N2,allht,allhb,mushroomFlag, porosity,optcolloid):

    
    DeltaS0 = bindingEnergies[1] #-466; #entropic contribution of binding (J/mol.K)
    DeltaH0 = bindingEnergies[0] #-170*10**3; # -47.8*4.184*1*10**3; #enthalpic contribution of binding (J/mol)
    #print(DeltaS0)
    #print(DeltaH0)

    DeltaG0 = DeltaH0/(Rg*T)-DeltaS0/Rg;
    print('Delta G0/kT at this temperature is',DeltaG0)
    #print('Delta H0 and Delta S0 are, DeltaG0 at 50C')
    #print(DeltaH0,DeltaS0,DeltaH0/(Rg*((273.15+55.0)))-DeltaS0/Rg)
    
    if sb < st:
        sigmaSticky = st #highest density
        fractionSticky = sb/st
    elif st < sb:
        sigmaSticky = sb #highest density
        fractionSticky = st/sb #here f is smaller than 1
    else: # they are equal
        sigmaSticky = sb
        fractionSticky = 1 # in that case same amount of sticky top and bottom
    
    nh = len(allhs)
    
    allKabs = np.zeros(nh)
    allQ1s= np.zeros(nh) 
    allQ2s= np.zeros(nh)
    phiBridgeSurf = np.zeros(nh)
    allSigmaabs = np.zeros(nh)
    #allpunboundt = np.zeros(nh)
    #allpunboundb = np.zeros(nh)
    
    hsCut = allhs[-1]
    threshold = 0.1
    criteria1 = 0
    criteria2 = 0
        
    for ih in range(len(allhs)):
        h = allhs[ih]
        ht = allht[ih]
        hb = allhb[ih]
        # this Kab corresponds to J/2 = sigma Kab in the text. 
        Kab_h,Q1_h,Q2_h = Kab(sigmaSticky, h, h1eq,h2eq, DeltaG0, hbond,N1,s1,ell1,N2,s2,ell2,ht,hb,mushroomFlag, porosity)
        allKabs[ih] = Kab_h
        allQ1s[ih] = Q1_h
        allQ2s[ih] = Q2_h
        #if Kab_h > 0: 
        #    print("kab here is", Kab_h)
        ftot_h = 0
        #print(Kab_h)
        if Kab_h > threshold:
            ftot_h = fatt(Kab_h,fractionSticky,sigmaSticky) 
            allSigmaabs[ih] = sigmaSticky*sigmaab(Kab_h,fractionSticky,sigmaSticky) #the right prefactor is the largest density (this was checked)
        if Kab_h <= threshold :
            criteria1 = 1
        if np.isnan(ftot_h):
            criteria1 = 1
        if criteria1 == 1:
            if Kab_h == 0:
                criteria2 = 1
                hsCut = h
            else:
                ftot_h = fattsupply(Kab_h,fractionSticky,sigmaSticky)
                allSigmaabs[ih] = sigmaSticky*sigmaabatt(Kab_h,fractionSticky,sigmaSticky) #the right prefactor is the largest density (this was checked)
                if np.isnan(ftot_h) :
                    criteria2 = 1
            criteria1 = 0
        if criteria2:
            phiBridgeSurf[ih] = 0
        else :
            phiBridgeSurf[ih] = ftot_h
            #allpunboundt[ih], allpunboundb[ih] = unboundProbas(Kab_h/sigmaSticky,st,sb)
   

#    print('These are the Kabs values')
#    print(allKabs)
#    print(phiBridgeSurf/sigmaSticky)
     
    phiBridge = DerjaguinIntegral(hsCut,Radius,allhs,phiBridgeSurf,optcolloid)
    sigmaBridge = DerjaguinIntegral(hsCut,Radius,allhs,allSigmaabs,optcolloid) 
    #sigmaBridge is actually equal to the number of bonds in contact
    
    #print(allQ1s)
    #print(allKabs)
    #print(phiBridgeSurf)
    #print(phiBridge)
    #return(phiBridge,sigmaBridge,allpunboundt,allpunboundb)
    # fractionSticky*sigmaSticky that's the maximum bonds that you can make so it makes sense to renormalize in that way. 
    return(phiBridge,sigmaBridge,allSigmaabs/(fractionSticky*sigmaSticky),fractionSticky*sigmaSticky,allKabs,DeltaG0,allQ1s,allQ2s)
    
    
    
    
    
            
            
            