#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:14:41 2021

@author: sm8857
"""

import math
import numpy as np
from CommonToolsModule import DerjaguinIntegral
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.optimize import curve_fit
from scipy import optimize
#from scipy.optimize import Bounds

from scipy.integrate import quad
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline
from math import log , exp, sqrt, inf, tanh, tan, sin,cos
pi = math.pi
import matplotlib.pyplot as plt
from StericModule import elleff, hMilner, StericPotentialMilner2faces, findStretchedEll
from HybridizationModule import compute_DNAbindingEnergy

# Constants
Na = 6.02*10**23
Rg = 8.314 #J/(mol.K); #perfect gas constant
Tbase = 273.15
kB = Rg/Na; #boltzmann constant


    

# def facingPotentialUnifiedBridge(Nb,ellb,sigmab,Nt,ellt,sigmat,heights,dG0,fractionfed,sigmaSticky,fractionSticky):
#     # this facing potential also computes the free energy for two tethers to bridge
    
        
#     N1 = Nb
#     N2 = Nt
#     ell1 = ellb
#     ell2 = ellt 
#     w1 = ell1**3
#     w2 = ell2**3
#     sigma1 = sigmab
#     sigma2 = sigmat
    
#     h1max = hMilner(sigmab,ell1,N1)
#     h2max = hMilner(sigmat,ell2,N2)
    
#     potentialExcluded = np.zeros(len(heights))
#     potentialStretch = np.zeros(len(heights))
#     potentialStretchu = np.zeros(len(heights))
#     potentialStretchb = np.zeros(len(heights))
#     potentialBridge = np.zeros(len(heights))
    
#     Pvalues = np.zeros(len(heights))
#     fvalues = np.zeros(len(heights))
#     Lvalues = np.zeros(len(heights))

#     # in assymetric settings normally this would have to be optimized for
#     hBridge = [h/2 for h in heights]
    
#     # for each h value we need to determine the number of segments Ps within the intermediate void region
#     #print(h1max)
#     #print(sigma1*ell1**2)
#     #dG0 = 0.5*dG0
#     #dG0 = 1000

    
#     for ih in range(len(hBridge)):
#         if len(fractionfed) > 0:
#             if fractionfed[ih] > 0:
#                 dGeff = - np.log(fractionfed[ih])
#                 #print("effective interaction energy at that distance", dG0)
#             else: 
#                 dGeff = 1e16

#         else:
#             hint = 1e-9
#             k0 = 1/(Na*10**3)*sigma1/hint*exp(-dG0)
#             dGeff = -np.log(k0)
            
                
#         h = hBridge[ih]
        
#         # the following quantities are designed to be in a "non-dimensional setting" all expressed wrt heq (for lengths)
#         # and N for numbers of segments
        
#         def layer(PoverN,Hoverheq):
#             # this function computes the thickness of the layer (full) region
#             if PoverN == 0:
#                 Loverheq = Hoverheq
#             else:
#                 Loverheq = Hoverheq/(1+PoverN*pi/(2)*tan(pi*PoverN/2))
#             if Loverheq >= 1:
#                 Loverheq = 1
#             return(Loverheq)
        
#         def fraction(PoverN,Hoverheq,Loverheq):
#             if PoverN == 0:
#                 f = 0
#             else:
#  #               print(H,L,P)
#                 if PoverN > 0.5:
#                     f = 1/PoverN*(Hoverheq-Loverheq)/Hoverheq*(1-Loverheq**3)
#                 else:
#                     f = pi/2*tan(pi*PoverN/2)*Loverheq/Hoverheq*(1-Loverheq**3)
# #                print(f)
#             # if f < 0 :
#             #     print(f)
#             #     print(P)
#             #     print(L)
#             #     print(H)
#             #     print(heq)
#             # if f > 1:
#             #     print(P)
#             #     print(N)
#             #     print(L)
#             #     print(H)
#             #     print(heq)
            
#             if f > sigmaSticky/sigma1*fractionSticky:
#                 f =  sigmaSticky/sigma1*fractionSticky # maximal fraction in the sticky compartment... 
#             ##print(sigmaSticky/sigma1*fractionSticky)
            
#             return(f)
    
    
#         def Bfac(N):
#             return(pi**2/(8*N**2))
        
#         def Afac(PoverN,heqoverellsqtN,f,Loverheq,Hoverheq):
#             if PoverN == 0:
#                 AfNoverell2 = ((heqoverellsqtN)**2) * ( \
#                                 (pi**2/8)*Loverheq**2 + (pi**2/12)*(1-Loverheq**3)*1/Hoverheq )
#             else:
#                # Af = B*L**2 + w*sigma*ell**2*f*P/(H-L)
#                 AfNoverell2 = ((heqoverellsqtN)**2) * ( \
#                                 (pi**2/8)*Loverheq**2 + (pi**2/12)*f*PoverN/(Hoverheq-Loverheq) )
            
                
#             return(AfNoverell2)
                
                
        
#         # energies are divided by 1/ell^2 and sigma multiplied by ell^2 as needed 
#         def fexcluded(PoverN,heqoverellsqtN,f,Loverheq,Hoverheq,AfNoverell2):
            
#             if PoverN == 0:
#                 return( (heqoverellsqtN)**(2)*( pi**2/12 * 1/Hoverheq*(1-Loverheq**3)  * Loverheq**3 +
#                                                 pi**2/20 * Loverheq**5 + 
#                                                 pi**2/24 * ( 1/Hoverheq*(1-Loverheq**3))**2*Hoverheq
#                                                 ) )
#             else:
#                 return( (heqoverellsqtN)**(2)*( pi**2/12 *f * PoverN/(Hoverheq-Loverheq)* Loverheq**3 +
#                                                 pi**2/20 * Loverheq**5 + 
#                                                 pi**2/24 * f**2 * PoverN**2/(Hoverheq - Loverheq)**2*Hoverheq
#                                                 ) )
            
            
#             # if PoverN == 0:
#             #     return(1/(2*(pi**2/12))*(AfNoverell2**2*Loverheq*(heqoverellsqtN)**(-2) - \
#             #                                   2*AfNoverell2*(pi**2/(8))*Loverheq**3/3 + \
#             #                                       (pi**2/8)**2*Loverheq**5/5*(heqoverellsqtN)**2))
#             # else:
#             #     #return(1/(2*sigma*w*ell**4)*(A**2*L - 2*A*B*L**3/3 + B**2*L**5/5 + w**2*(sigma*ell**2)**2*f**2*P**2/(H-L)))
#             #     return(1/(2*(pi**2/12))*(AfNoverell2**2*Loverheq*(heqoverellsqtN)**(-2) - \
#             #                                   2*AfNoverell2*(pi**2/(8))*Loverheq**3/3 + \
#             #                                       (pi**2/8)**2*Loverheq**5/5*(heqoverellsqtN)**2 + \
#             #                                   (pi**2/12)**2*f*PoverN*1/Hoverheq*(1-Loverheq**3)*(heqoverellsqtN)**2 ) )
            
#         #def fexcluded0(w,sigma,A,L,B,f,H,P,ell):
#         #    return(1/(2*sigma*w*ell**4)*((B*L**2)**2*L - 2*(B*L**2)*B*L**3/3 + B**2*L**5/5))

#         def fstretch(PoverN,heqoverellsqtN,f,Loverheq,Hoverheq):
            
#             pip = pi*PoverN/2
#             if PoverN == 0:
#                 stretch = ((heqoverellsqtN)**2) * (Loverheq**2) *pi**2/(8) * 1/15 * ( 5 - 2*Loverheq**3)  
#                 stretchu = stretch
#                 stretchb = 0
#             else:
#                 stretchu = ((heqoverellsqtN)**2) * (Loverheq**2) *pi**2/(8) * 1/15 * ( 3*Loverheq**3  \
#                                     - 5/(pi)* f *1/sin(2*pip) *  \
#                                         (-5 + cos(2*pip) + 3*(1-PoverN)*pi*tan(pip)) )
                    
#                 stretchb = ((heqoverellsqtN)**2)*f/2* ( pi**2/4* Loverheq**2/(cos(pip))**2* \
#                                                       ((1-PoverN)/2 + (1/(2*pi))*sin(pip*2)) + \
#                                                           (Hoverheq-Loverheq)**2/PoverN )
#                 stretch = stretchu + stretchb
            
#             # pip = pi*PoverN/2
#             # if f == 0:
#             #     stretch = ((heqoverellsqtN)**2)*(Loverheq**5)*pi**2*3/(4*8)*(4/15 + 4/9*(1/Loverheq**3 -1))
#             #     stretchu = stretch
#             #     stretchb = 0
#             # else:
#             #     #print(P,pip,tan(pip),f,f/tan(pip),(1-L**3/heq**3)*(H-L)/H*2/pi)
#             #     stretchu = ((heqoverellsqtN)**2)*(Loverheq**5)*pi**2*3/(4*8)*(4/15 + \
#             #         (2*f*(4/tan(pip) + 3*(1 + (tan(pip))**2)* \
#             #           (pi*(PoverN-1) + sin(pip))  ) )/(9*Loverheq**3*pi) ) 
            
#             #     stretchb = ((heqoverellsqtN)**2)*f/2* ( pi**2/4* Loverheq**2/(cos(pip))**2* \
#             #                                           ((1-PoverN)/2 + (1/(2*pi))*sin(pip*2)) + \
#             #                                               (Hoverheq-Loverheq)**2/PoverN )
                    
#             #     stretch = stretchu + stretchb
            
#             return(stretch,stretchu,stretchb)
        
#         def fbridge(f,Loverheq,Hoverheq):
#             #print(f)
#             if len(fractionfed) > 0: #then entropic penalties are already accounted for
#                 if f == 0 or f == 1:
#                     e3 = (f*(dGeff))/2 #*sigma*ell**2
#                 #elif f <= 0:
#                 #    e3 = 1e100
#                 else:
#                     #print(f)
#                     e3 = (f*(dGeff) + f*np.log(f) + (1-f)*np.log(1-f))/2 
#                     # I'm not extra sure about this factor.. 
#             else:

#                 hinteff = max(Hoverheq*h1max-Loverheq*h1max,hint)
#                 k0 = hint/(hinteff)*exp(-dGeff)
#                 dGeff2 = -np.log(k0)
                
#                 if f == 0:
#                     e3 = 0
#                 elif f == 1:
#                     e3 = (f*(dGeff2 ))/2 #*sigma*ell**2 - np.log(f)
#                 #elif f <= 0:
#                 #    e3 = 1e100
#                 else:
#                     #print(f)
#                     e3 = (f*(dGeff2 ) + f*np.log(f) + (1-f)*np.log(1-f))/2 # I'm not extra sure about this factor.. 
            
#             return( e3)
        
#         def finfinity(heqoverellsqtN):
        
#             return(9/10*(pi**2/12)*(heqoverellsqtN)**2)
            
            
#         def allEnergies(PoverN):
#             Hoverheq = h/h1max
#             heqoverellsqtN = h1max/(ell1*sqrt(N1))
#             Loverheq = layer(PoverN,Hoverheq)
#             fvalue = fraction(PoverN,Hoverheq,Loverheq) 
#             Avalue = Afac(PoverN,heqoverellsqtN,fvalue,Loverheq,Hoverheq)
#             e1 = fexcluded(PoverN,heqoverellsqtN,fvalue,Loverheq,Hoverheq,Avalue)
#             e2,e2u,e2b = fstretch(PoverN,heqoverellsqtN,fvalue,Loverheq,Hoverheq)   
            
            
#             e3 = fbridge(fvalue,Loverheq,Hoverheq) #+ fvalue*(1e100)
#             return(e1+e2+e3)

#         def allEnergiesSplit(PoverN):
#             Hoverheq = h/h1max
#             heqoverellsqtN = h1max/(ell1*sqrt(N1))
#             Loverheq = layer(PoverN,Hoverheq)
#             fvalue = fraction(PoverN,Hoverheq,Loverheq) 
#             Avalue = Afac(PoverN,heqoverellsqtN,fvalue,Loverheq,Hoverheq)
#             e1 = fexcluded(PoverN,heqoverellsqtN,fvalue,Loverheq,Hoverheq,Avalue)
#             e2,e2u,e2b = fstretch(PoverN,heqoverellsqtN,fvalue,Loverheq,Hoverheq)   
            
#             if fvalue > 0:
#                 e2b = e2b/fvalue
#             if fvalue < 1: 
#                 e2u = e2u/(1-fvalue)
            
#             e3 = fbridge(fvalue,Loverheq,Hoverheq) #+ fvalue*(1e100)
#             return(e1,e2,e2u,e2b,e3,Loverheq,fvalue)

#         testForMinimization = 1
        
#         if testForMinimization == 0: 
#             # this part was used for debugging purposes
#             ftoFind = fractionfed[ih]
#             #print(ftoFind)
#             Ndisc = 100
#             pvals = np.linspace(0.0,1.0,Ndisc)
#             fvals = []
#             for ip in range(Ndisc):
#                 e1,e2,e2u,e2b,e3,Lv,fv = allEnergiesSplit(pvals[ip])
#                 fvals.append(abs(fv-ftoFind))
#             idmin = np.array(fvals).argmin()
#             Pvalue = pvals[idmin]  
#             e1,e2,e2u,e2b,e3,Lv,fv = allEnergiesSplit(Pvalue)
#             heqoverellsqtN = h1max/(ell1*sqrt(N1))
            
#         else:
#             Ndisc = 100
#             pvals = np.linspace(0.0,1.0,Ndisc)
#             evals = [allEnergies(p) for p in pvals]
#             idmin = np.array(evals).argmin()
            
#             e1s = np.zeros(Ndisc)        
#             e2s = np.zeros(Ndisc)        
#             e3s = np.zeros(Ndisc)     
#             e2us = np.zeros(Ndisc)    
#             e2bs = np.zeros(Ndisc) 
#             fractios = np.zeros(Ndisc)
#             for i in range(Ndisc):
#                 e1s[i],e2s[i],e2us[i],e2bs[i],e3s[i],lv,fractios[i] = allEnergiesSplit(pvals[i])
    
#             #plt.plot(pvals,fractios)
#             #plt.show()
        
#             pmin = pvals[idmin]
#             #print(pmin)
#             #print(fractios[idmin]*h1max/h)
#             if pmin == 0:
#                 optimumE = optimize.minimize_scalar(allEnergies, bounds=(0.0, 0.1), method='bounded')  
#             else:
#                 optimumE = optimize.minimize_scalar(allEnergies, bounds=(max(pmin*0.9,0.0), min(pmin*1.1,1.0)), method='bounded')  
#             Pvalue = optimumE.x
#             # find the value of P that minimizes energy
            
#             optimumE = optimize.minimize_scalar(allEnergies, bounds=(0.0, 1.0), method='bounded')  
#             Pvalue2 = optimumE.x
#             #Pvalue = Pvalue2 #not clear when /which is the best one.. 
#             #Pvalue = 0
#             #Pvalue = pvals[idmin]
#             # plt.plot(pvals,e1s,'x')
#             # plt.plot(pvals,e2s,'+')
#             # plt.plot(pvals,e3s,'o')
#             # plt.plot(pvals,fractios,'o')
            
#             # plt.plot(pvals,e1s,'x',label = 'excluded')
#             # plt.plot(pvals,e2s,'+',label = 'stretched')
#             # plt.plot(pvals,e3s,'o',label = 'bridge')
#             # plt.plot(pvals,e3s+e2s+e1s,'k', label = 'total')
#             # plt.title('H= {:.2f}'.format(h/h1max)+' $L_{eq}$')
#             # plt.legend()
#             # plt.pause(0.5)
#             # plt.show()
            
#             #Pvalue = 0
            
#             e1,e2,e2u,e2b,e3,Lv,fv = allEnergiesSplit(Pvalue)
#             heqoverellsqtN = h1max/(ell1*sqrt(N1))
        
#         #print(Pvalue,heqoverellsqtN)
        
#         #print(e1,e2,9/10*N1*(sigma1*w1/ell1)**(2/3)*(pi**2/12)**(1/3),ell1**2)
        
        
#         Pvalues[ih] = Pvalue
#         fvalues[ih] = fv*sigma1 # this is the number density of pairs
#         Lvalues[ih] = Lv
#         potentialExcluded[ih] = 2*sigma1*e1
#         potentialStretch[ih] = 2*sigma1*e2
#         potentialStretchu[ih] = 2*sigma1*e2u
#         potentialStretchb[ih] = 2*sigma1*e2b
#         potentialBridge[ih] = 2*sigma1*e3
        
#     print("effective interaction energy at that distance", dGeff)
#     # plt.plot([h/h1max for h in hBridge],[P/N1 for P in Pvalues],'+',label='fraction of segments bound')
#     # plt.plot([h/h1max for h in hBridge],[f/sigma1 for f in fvalues],'x', label ='fraction of tethers bound')
#     # plt.plot([h/h1max for h in hBridge],[l/h1max for l in Lvalues],'o',label='thickness of layer')
#     # plt.legend()
#     # plt.xlim(0,3.0)
#     # plt.ylim(0,1.0)
#     # plt.pause(3)
#     # plt.show()
    
#     # print([f/sigma1 for f in fvalues[-10:-1]])
    
    
#     einf = 2*sigma1*finfinity(heqoverellsqtN)
#     #print("energies at infinity",einf,potentialExcluded[-1]+potentialStretch[-1])
#     potentialExcluded = potentialExcluded-potentialExcluded[-1]
#     potentialStretch = potentialStretch-potentialStretch[-1]
    
#     einfh = np.zeros(len(hBridge))
#     for i in range(len(hBridge)):
#         if hBridge[i] <= h1max:
#             h = hBridge[i]
#             einfh[i] = einf/(sigma1*9/10)*(h1max/(2*h) + h**2/(2*h1max**2) - 1/10*h**5/h1max**5 - 9/10) 
    
#     # plt.plot([h/h1max for h in hBridge],[f/sigma1 for f in fvalues],'.',label='fraction')
#     # plt.plot([h/h1max for h in hBridge],[p for p in Pvalues],'v',label='P values')
#     # plt.legend()
#     # plt.xlim(0.0,1.5)
#     # plt.ylim(0,0.5)
#     # plt.pause(0.1)
#     # plt.show()
    
#     # plt.plot([h/h1max for h in hBridge],Lvalues,'.')
#     # plt.xlim(0.0,5)
#     # plt.ylim(0,1.2)
#     # plt.pause(1)
#     # plt.show()
    
#     plt.plot([h/h1max for h in hBridge],[p/sigma1 for p in potentialExcluded],'+',label='excluded')
#     plt.plot([h/h1max for h in hBridge],[p/sigma1 for p in potentialStretch], 'x',label ='stretched')
#     #plt.plot([h/h1max for h in hBridge],[p/sigma1 for p in potentialStretchb], 'v',label ='s bound')
#     plt.plot([h/h1max for h in hBridge],[p/sigma1 for p in potentialBridge],'o',label='bridge')
#     plt.plot([h/h1max for h in hBridge],einfh,'d',label='f=0',ms = 2)
#     plt.plot([h/h1max for h in hBridge],[(potentialStretch[p]+ potentialExcluded[p]+ potentialBridge[p])/sigma1 for p in range(len(potentialStretch))], 'k', lw=2,label ='full')
#     plt.plot([h/h1max for h in hBridge],[(potentialStretch[p]+ potentialExcluded[p])/sigma1 for p in range(len(potentialStretch))], '--',color= 'b', lw=2,label ='full wt bridge')
    
#     # print((potentialStretch[-1]+ potentialExcluded[-1]+ potentialBridge[-1])/sigma1)
#     # print((potentialExcluded[-1]+ potentialBridge[-1])/sigma1)
#     # print((potentialBridge[-1])/sigma1)
#     # print(hBridge[-1])
#     #print(einfh,[p/sigma1 for p in potentialStretch] )
    
#     #plt.plot(Lvalues,[f/sigma1 for f in fvalues],'.',label='fraction')
#     #plt.plot(Lvalues,[p for p in Pvalues],'v',label='P values')
#     plt.legend()
#     plt.xlim(0.7,1.15)
#     plt.ylim(-2,2)
#     plt.pause(1.0)
#     plt.show()
#     # #print("fraction values",[f/sigma1 for f in fvalues])
    
#     # plt.plot([h/h1max for h in hBridge],[p/sigma1 for p in potentialStretch],'k',label ='stretched')
#     # plt.plot([h/h1max for h in hBridge],[p/sigma1 for p in potentialStretchb], 'v',label ='bound')
#     # plt.plot([h/h1max for h in hBridge],[p/sigma1 for p in potentialStretchu], '.',label ='unbound')
#     # plt.legend()
#     # plt.xlim(0.5,4.5)
#     # plt.ylim(-5,10)
#     # plt.pause(0.5)
#     # plt.show()
    
#     # plt.plot([h/h1max for h in hBridge],[(potentialStretch[p]+ potentialExcluded[p]+ potentialBridge[p])/sigma1 for p in range(len(potentialStretch))], 'k', lw=2,label ='full')
    
#     # #plt.plot(Lvalues,[f/sigma1 for f in fvalues],'.',label='fraction')
#     # #plt.plot(Lvalues,[p for p in Pvalues],'v',label='P values')
#     # plt.legend()
#     # plt.xlim(0.5,1.5)
#     # plt.ylim(-10,1)
#     # plt.pause(0.1)
#     # plt.show()
    
#     return(Pvalues,fvalues,Lvalues,potentialExcluded,potentialStretch,potentialBridge)

def facingPotentialUnifiedBridgeAssymetric(Nb,ellb,sigmab,eVb,Nt,ellt,sigmat,eVt,heights,dG0,fractionfed,stickyb,stickyt):
    # this facing potential also computes the free energy for two tethers to bridge
    
        
    N1 = Nb
    N2 = Nt
    ell1 = ellb
    ell2 = ellt 
    eV1 = eVb
    eV2 = eVt
    w1 = eV1*ell1**3
    w2 = eV2*ell2**3
    sigma1 = sigmab
    sigma2 = sigmat
    sticky1 = stickyb
    sticky2 = stickyt
    fraction1 = stickyb/sigmab #fraction of sticky at the bottom
    fraction2 = stickyt/sigmat #fraction of sticky at the top
    ssticky = max(stickyb,stickyt)
    fsticky = min(stickyb,stickyt)/ssticky

    #print("fraction of sticky ends top and bottom", fraction1, fraction2)
    #print("density of sticky ends top and bottom", sticky1, sticky2)
    #print("maximum fraction of sticky ends",fsticky)
    
    h1max = hMilner(sigmab,ell1,N1,eV1)
    h2max = hMilner(sigmat,ell2,N2,eV2)
    
    potentialExcluded = np.zeros(len(heights))
    potentialStretch = np.zeros(len(heights))
    potentialStretchu = np.zeros(len(heights))
    potentialStretchb = np.zeros(len(heights))
    potentialBridge = np.zeros(len(heights))
    
    fvalues = np.zeros(len(heights))


    

    #print ("number of heights to optimize",len(heights))
    for ih in range(len(heights)):
        
        htot = heights[ih]
        
        
        
        #now you have to minimize the actually combined problem BUT with a common f of course... 
        # first define the energy expense
        if len(fractionfed) > 0:
            if fractionfed[ih] > 0:
                dGeff = - np.log(fractionfed[ih]) #the location of this center position may not come from where I intend it to come from... 
                #print("effective interaction energy at that distance", dGeff)
            else: 
                dGeff = 1e16

        else:
            hint = 1e-9
            k0 = 1/(Na*10**3)*sigma1/hint*exp(-dG0)
            dGeff = -np.log(k0)
            
                  
        # the following quantities are designed to be in a "non-dimensional setting" all expressed wrt heq (for lengths)
        # and N for numbers of segments
        
        def layer(PoverN,Hoverheq):
            # this function computes the thickness of the layer (full) region
            if PoverN == 0:
                Loverheq = Hoverheq
            else:
                Loverheq = Hoverheq/(1+PoverN*pi/(2)*tan(pi*PoverN/2))
            if Loverheq >= 1:
                Loverheq = 1
            return(Loverheq)
        
        def fraction(PoverN,Hoverheq,Loverheq,fraction):
            if PoverN == 0:
                f = 0
            else:
 #               print(H,L,P)
                if PoverN > 0.5:
                    f = 1/PoverN*(Hoverheq-Loverheq)/Hoverheq*(1-Loverheq**3)
                else:
                    f = pi/2*tan(pi*PoverN/2)*Loverheq/Hoverheq*(1-Loverheq**3)
            
            if f > fraction:
                f =  2 # maximal fraction in the sticky part...  # this shows that f is not the same for each side, but sigma*f = # bound should be the same
            
            return(f)
    
    
        def Bfac(N):
            return(pi**2/(8*N**2))
        
        def Afac(PoverN,heqoverellsqtN,f,Loverheq,Hoverheq):
            if PoverN == 0:
                AfNoverell2 = ((heqoverellsqtN)**2) * ( \
                                (pi**2/8)*Loverheq**2 + (pi**2/12)*(1-Loverheq**3)*1/Hoverheq )
            else:
               # Af = B*L**2 + w*sigma*ell**2*f*P/(H-L)
                AfNoverell2 = ((heqoverellsqtN)**2) * ( \
                                (pi**2/8)*Loverheq**2 + (pi**2/12)*f*PoverN/(Hoverheq-Loverheq) )
            
                
            return(AfNoverell2)
                
                
        
        # energies are divided by 1/ell^2 and sigma multiplied by ell^2 as needed 
        def fexcluded(PoverN,heqoverellsqtN,f,Loverheq,Hoverheq,AfNoverell2):
            
            if PoverN == 0:
                return( (heqoverellsqtN)**(2)*( pi**2/12 * 1/Hoverheq*(1-Loverheq**3)  * Loverheq**3 +
                                                pi**2/20 * Loverheq**5 + 
                                                pi**2/24 * ( 1/Hoverheq*(1-Loverheq**3))**2*Hoverheq
                                                ) )
            else:
                return( (heqoverellsqtN)**(2)*( pi**2/12 *f * PoverN/(Hoverheq-Loverheq)* Loverheq**3 +
                                                pi**2/20 * Loverheq**5 + 
                                                pi**2/24 * f**2 * PoverN**2/(Hoverheq - Loverheq)**2*Hoverheq
                                                ) )
            
            
            # if PoverN == 0:
            #     return(1/(2*(pi**2/12))*(AfNoverell2**2*Loverheq*(heqoverellsqtN)**(-2) - \
            #                                   2*AfNoverell2*(pi**2/(8))*Loverheq**3/3 + \
            #                                       (pi**2/8)**2*Loverheq**5/5*(heqoverellsqtN)**2))
            # else:
            #     #return(1/(2*sigma*w*ell**4)*(A**2*L - 2*A*B*L**3/3 + B**2*L**5/5 + w**2*(sigma*ell**2)**2*f**2*P**2/(H-L)))
            #     return(1/(2*(pi**2/12))*(AfNoverell2**2*Loverheq*(heqoverellsqtN)**(-2) - \
            #                                   2*AfNoverell2*(pi**2/(8))*Loverheq**3/3 + \
            #                                       (pi**2/8)**2*Loverheq**5/5*(heqoverellsqtN)**2 + \
            #                                   (pi**2/12)**2*f*PoverN*1/Hoverheq*(1-Loverheq**3)*(heqoverellsqtN)**2 ) )
            
        #def fexcluded0(w,sigma,A,L,B,f,H,P,ell):
        #    return(1/(2*sigma*w*ell**4)*((B*L**2)**2*L - 2*(B*L**2)*B*L**3/3 + B**2*L**5/5))

        def fstretch(PoverN,heqoverellsqtN,f,Loverheq,Hoverheq):
            
            pip = pi*PoverN/2
            if PoverN == 0:
                stretch = ((heqoverellsqtN)**2) * (Loverheq**2) *pi**2/(8) * 1/15 * ( 5 - 2*Loverheq**3)  
                stretchu = stretch
                stretchb = 0
            else:
                stretchu = ((heqoverellsqtN)**2) * (Loverheq**2) *pi**2/(8) * 1/15 * ( 3*Loverheq**3  \
                                    - 5/(pi)* f *1/sin(2*pip) *  \
                                        (-5 + cos(2*pip) + 3*(1-PoverN)*pi*tan(pip)) )
                    
                stretchb = ((heqoverellsqtN)**2)*f/2* ( pi**2/4* Loverheq**2/(cos(pip))**2* \
                                                      ((1-PoverN)/2 + (1/(2*pi))*sin(pip*2)) + \
                                                          (Hoverheq-Loverheq)**2/PoverN )
                stretch = stretchu + stretchb
            
            # pip = pi*PoverN/2
            # if f == 0:
            #     stretch = ((heqoverellsqtN)**2)*(Loverheq**5)*pi**2*3/(4*8)*(4/15 + 4/9*(1/Loverheq**3 -1))
            #     stretchu = stretch
            #     stretchb = 0
            # else:
            #     #print(P,pip,tan(pip),f,f/tan(pip),(1-L**3/heq**3)*(H-L)/H*2/pi)
            #     stretchu = ((heqoverellsqtN)**2)*(Loverheq**5)*pi**2*3/(4*8)*(4/15 + \
            #         (2*f*(4/tan(pip) + 3*(1 + (tan(pip))**2)* \
            #           (pi*(PoverN-1) + sin(pip))  ) )/(9*Loverheq**3*pi) ) 
            
            #     stretchb = ((heqoverellsqtN)**2)*f/2* ( pi**2/4* Loverheq**2/(cos(pip))**2* \
            #                                           ((1-PoverN)/2 + (1/(2*pi))*sin(pip*2)) + \
            #                                               (Hoverheq-Loverheq)**2/PoverN )
                    
            #     stretch = stretchu + stretchb
            
            return(stretch,stretchu,stretchb)
        
        
        def fbridge(f,Loverheq,Hoverheq,alpha): #in here goes the "real f" = sab/sstickymax; and alpha = sstickymin/sstickymax
            #print(f)
            if len(fractionfed) > 0: #then entropic penalties are already accounted for in the Daan Frenkel way
                if alpha > 1: 
                    print("fraction of sticky DNA is not compatible")
                if f == 0 or f == alpha:
                    e3 = (f*(dGeff))/2 #*sigma*ell**2
                #elif f <= 0:
                #    e3 = 1e100
                else:
                    #print(f)
                    #e3 = (f*(dGeff) + f*np.log(f) + (1-f)*np.log(1-f))/2 
                    e3 = (f*(dGeff+1) + f*np.log(f) + (1-f)*np.log(1-f) + (alpha-f)*np.log(1-f/alpha) - f*log(alpha) )/2 #this model fits exactly with the derivation by Frenkel/Mognetti
            else:

                # this part of the code is obsolete... 
                hinteff = max(Hoverheq*h1max-Loverheq*h1max,hint)
                k0 = hint/(hinteff)*exp(-dGeff)
                dGeff2 = -np.log(k0)
                
                if f == 0:
                    e3 = 0
                elif f == 1:
                    e3 = (f*(dGeff2 ))/2 #*sigma*ell**2 - np.log(f)
                #elif f <= 0:
                #    e3 = 1e100
                else:
                    #print(f)
                    e3 = (f*(dGeff2 +1 ) + f*np.log(f) + 2*(1-f)*np.log(1-f))/2 # I'm not extra sure about this factor.. 
            
            return( e3)
        
        def finfinity(heqoverellsqtN):
        
            return(9/10*(pi**2/12)*(heqoverellsqtN)**2)
        
        
        def allEnergies(PoverN1,hBridge):
            Hoverheq1 = hBridge/h1max
            Hoverheq2 = (htot-hBridge)/h2max
            
            heqoverellsqtN1 = h1max/(ell1*sqrt(N1))
            heqoverellsqtN2 = h2max/(ell2*sqrt(N2))
            
            flagNoSearch =0 # flag to determine wether it makes sense to compute energy with these parameters or if they are just inaccessible
            fvalue1 = 0
            fvalue2 = 0
            PoverN2 = 0
            Loverheq2 = 0 
            Loverheq1 = 0
            
            if PoverN1 > 0:
                if sticky2 < sticky1:
                    # if the 1 is the stickyest
                    
                    Loverheq1 = layer(PoverN1,Hoverheq1)
                    fvalue1 = fraction(PoverN1,Hoverheq1,Loverheq1,fraction1*fsticky) #fraction1*fsticky is the max value for fvalue1 actually
                    
                    if fvalue1 == 2: 
                        flagNoSearch = 1
                    else:
                        
                        fvalue2 = fvalue1*sigma1/sigma2 #necessary condition that the #of bonds top and bottom is coherent
                        if fvalue2 <= fraction2: #fraction2 is the max value for fvalue2 actually
                            def PoverNsearch(PoverN):
                                Loverheq2eval = layer(PoverN,Hoverheq2)
                                fvalue2eval = fraction(PoverN,Hoverheq2,Loverheq2eval,fraction2) 
                                return((fvalue2 - fvalue2eval))
                            PoverN2 = brentq(PoverNsearch,0,1)
                            #print("PoverN",PoverN1,PoverN2)
                            Loverheq2 = layer(PoverN2,Hoverheq2)
                            #print("Fractions bound at this distance",fvalue1,fvalue2)
                        else:
                            flagNoSearch = 1
                else:
                    # if the 2 is the stickyest
                    PoverN2 = PoverN1
                    Loverheq2 = layer(PoverN2,Hoverheq2)
                    fvalue2 = fraction(PoverN2,Hoverheq2,Loverheq2,fraction2*fsticky) #fraction2*fsticky is the max value for fvalue2 actually
                    if fvalue2 == 2: 
                        flagNoSearch = 1
                    else:
                        fvalue1 = fvalue2*sigma2/sigma1 #necessary condition that the #of bonds top and bottom is coherent
                        if fvalue1 <= fraction1:
                            def PoverNsearch(PoverN):
                                Loverheq1eval = layer(PoverN,Hoverheq1)
                                fvalue1eval = fraction(PoverN,Hoverheq1,Loverheq1eval,fraction1) 
                                return((fvalue1 - fvalue1eval))
                            PoverN1 = brentq(PoverNsearch,0,1)
                            #print("PoverN",PoverN1,PoverN2)
                            Loverheq1 = layer(PoverN1,Hoverheq1)
                            #print("Fractions bound at this distance",fvalue1,fvalue2)
                        else:
                            flagNoSearch = 1
            else:
                #then there's no bound in any of the layers so they are just trivially relaxed
                Loverheq1 = layer(PoverN1,Hoverheq1)
                Loverheq2 = layer(PoverN2,Hoverheq2)
                fvalue1 = 0
                fvalue2 = 0
                
            if flagNoSearch == 0: 
                
                Avalue1 = Afac(PoverN1,heqoverellsqtN1,fvalue1,Loverheq1,Hoverheq1)
                Avalue2 = Afac(PoverN2,heqoverellsqtN2,fvalue2,Loverheq2,Hoverheq2)
                
                
                e11 = fexcluded(PoverN1,heqoverellsqtN1,fvalue1,Loverheq1,Hoverheq1,Avalue1)
                e12 = fexcluded(PoverN2,heqoverellsqtN2,fvalue2,Loverheq2,Hoverheq2,Avalue2)
                
                e21,e2u,e2b = fstretch(PoverN1,heqoverellsqtN1,fvalue1,Loverheq1,Hoverheq1) 
                e22,e2u,e2b = fstretch(PoverN2,heqoverellsqtN2,fvalue2,Loverheq2,Hoverheq2) 
                
                if sticky2 < sticky1: 
                    fractionE3 = fvalue1/fraction1 #the maximum value of this object should be fsticky
                    e3 = sticky1*fbridge(fractionE3,Loverheq1,Hoverheq1,fsticky)*2
                else:
                    fractionE3 = fvalue2/fraction2 #the maximum value of this object should be fsticky
                    e3 = sticky2*fbridge(fractionE3,Loverheq1,Hoverheq1,fsticky)*2


            else:
                e11 = 1e20
                e12 = 1e20
                e21 = 1e20
                e22 = 1e20
                e3 = 1e20


            
            return((sigma1*(e11+e21) + sigma2*(e12+e22) + e3)*1e-16,fvalue1,fvalue2,Loverheq1,Loverheq2,sigma1*e11+sigma2*e12,sigma1*e21+sigma2*e22,e3)
        
        def allEnergiesNoBound():
            Hoverheq1 = 1.0
            Hoverheq2 = 1.0
            
            heqoverellsqtN1 = h1max/(ell1*sqrt(N1))
            heqoverellsqtN2 = h2max/(ell2*sqrt(N2))
            
            fvalue1 = 0
            fvalue2 = 0
            PoverN2 = 0
            PoverN1 = 0
            Loverheq1 = layer(PoverN1,Hoverheq1)
            Loverheq2 = layer(PoverN2,Hoverheq2)
            
            
            Avalue1 = Afac(PoverN1,heqoverellsqtN1,fvalue1,Loverheq1,Hoverheq1)
            Avalue2 = Afac(PoverN2,heqoverellsqtN2,fvalue2,Loverheq2,Hoverheq2)
                
                
            e11 = fexcluded(PoverN1,heqoverellsqtN1,fvalue1,Loverheq1,Hoverheq1,Avalue1)
            e12 = fexcluded(PoverN2,heqoverellsqtN2,fvalue2,Loverheq2,Hoverheq2,Avalue2)
                
            e21,e2u,e2b = fstretch(PoverN1,heqoverellsqtN1,fvalue1,Loverheq1,Hoverheq1) 
            e22,e2u,e2b = fstretch(PoverN2,heqoverellsqtN2,fvalue2,Loverheq2,Hoverheq2) 
                

            e3 = 0

            
            return((sigma1*(e11+e21) + sigma2*(e12+e22) + e3)*1e-16,fvalue1,fvalue2,Loverheq1,Loverheq2,sigma1*e11+sigma2*e12,sigma1*e21+sigma2*e22,e3)
        
        
        
        def energyMinimizer(hBridge):
            
            def energyFuncP(PoverN1):
                eInterest,other,other,other,other,other,other,other = allEnergies(PoverN1,hBridge)
                    
                return(eInterest)
        

            Ndisc = 10
            
            pmin = 0.0
            pmax = 1.0
            
            iterP = 0
            while iterP < 4:
            
                pvals = np.linspace(pmin,pmax,Ndisc)
                
                evals = [energyFuncP(p) for p in pvals]
                idmin = np.array(evals).argmin()
                
                if idmin == 0:
                    pmin = 0.0
                    pmax = pvals[idmin+1]
                elif idmin == Ndisc-1:
                    pmax = 1.0
                    pmin = pvals[idmin-1]
                else:
                    pmin = pvals[idmin-1]
                    pmax = pvals[idmin+1]
                iterP += 1
                
            # Ndisc = 20
            # pvals = np.linspace(pmin,pmax,Ndisc)
            
            # evals = [energyFuncP(p) for p in pvals]
            # idmin = np.array(evals).argmin()
                
                    
            
            #es = np.zeros(Ndisc)        
           
            #for i in range(Ndisc):
            #    es[i] = energyFuncP(pvals[i])
    
            # plt.plot(pvals,es)
            # plt.show()
            # plt.pause(2)
        
            # I could improve the detection of the minimum by improving this. 
            pmin = pvals[idmin]
            # but at least this scan covers the whole range so I'm sure that I'm finding the minimum
            #print(pmin)
            #print(fractios[idmin]*h1max/h)
            #if pmin == 0:
            #    optimumE = optimize.minimize_scalar(energyFuncP, bounds=(0.0, 0.1), method='bounded')  
            #else:
            #    optimumE = optimize.minimize_scalar(energyFuncP, bounds=(max(pmin*0.9,0.0), min(pmin*1.1,1.0)), method='bounded')  
            #Pvalue = optimumE.x
            Pvalue = pmin
            # find the value of P that minimizes energy
            Evalue = energyFuncP(Pvalue)
            
            return(Pvalue,Evalue)
        
        def Eh(hBridge):
            #define quantities to find the position of the bridge
            Pvalue,Evalue = energyMinimizer(hBridge)
            
            
            return(Evalue,Pvalue)
        
        #print("Entering minimization of energy")
        disc = 10
        
        if htot < h1max+h2max: #at least for brushes which is the case here there's no value beyond the bound point. 
            
            hmin = max(1e-11,(htot-h2max)*0.8)
            hmax = min((h1max)*1.2,htot*0.99)
            
            iterMin = 0
            
            while iterMin < 3:
            
                hvals =np.linspace(hmin,hmax,disc)
                
                hmid = htot*h1max/(h2max+h1max)
                #hvals = np.linspace(max(0,0.7*hmid),min(1.3*hmid,htot*0.99),disc)
                
                evals = np.zeros(len(hvals))
                pvals = np.zeros(len(hvals))
                
                for p in range(len(hvals)):
                    ehp,pp = Eh(hvals[p])
                    
                    evals[p] = ehp
                    pvals[p] = pp 
                    #[Eh(p) for p in hvals]
                idmin = np.array(evals).argmin()
                
                if idmin == 0:
                    hmax = hvals[idmin+1]
                elif idmin == disc -1 :
                    hmin = hvals[idmin-1]
                else:
                    hmin = hvals[idmin-1]
                    hmax = hvals[idmin+1]
                iterMin += 1
            
                # if ih%100 == 0:
                #     print(hmid,htot,h1max,h2max)
                #     plt.plot(hvals,evals)
                #     #plt.plot(hvals,pvals)
                #     plt.show()
                #     plt.pause(1)
            
            #optimumE = optimize.minimize_scalar(Eh, bounds=(0.01*htot,0.99*htot), method='bounded') 
            h1real = hvals[idmin] # optimumE.x
            
            # h1real = hmid
            
            Pvalue1,Etot = energyMinimizer(h1real)
            Etot,fv1,fv2,Lv1,Lv2,e1,e2,e3 = allEnergies(Pvalue1,h1real)
        else:
            Pvalue1 = 0
            h1real = htot*h1max/(h2max+h1max) #take any value, it shouldn't matter since Pvalue1 = 0
            Etot,fv1,fv2,Lv1,Lv2,e1,e2,e3 = allEnergiesNoBound() 
        
        #print("At this distance we have", Pvalue1, h1real*1e9, htot*1e9,hmid*1e9, hmid*1.5*1e9, Etot)
        
        
        fvalues[ih] = fv1*sigma1 # this is the number density of pairs (computed from one value but on other side its the same)
        #print("number density of pairs, should be equal", fv2*sigma2,fv1*sigma1) # this is the number density of pairs -- they should be equal

        potentialExcluded[ih] = e1
        potentialStretch[ih] = e2
        potentialBridge[ih] = e3

    #print("effective interaction energy at that distance", dGeff)

    

    #print("energies at infinity",einf,potentialExcluded[-1]+potentialStretch[-1])
    potentialExcluded = potentialExcluded-potentialExcluded[-1]
    potentialStretch = potentialStretch-potentialStretch[-1]
    
   

    return(fvalues,potentialExcluded,potentialStretch,potentialBridge)


def unifiedPotentials(allhs,h1eq,h2eq,sb,st,Radius,tetherSeq,saltConcentration,T,s1,ell1,N1,eV1,s2,ell2,N2,eV2,bridging,optcolloid,fractionfed):

    bindingEnergies = compute_DNAbindingEnergy(tetherSeq,saltConcentration)
    DeltaS0 = bindingEnergies[1] #-466; #entropic contribution of binding (J/mol.K)
    DeltaH0 = bindingEnergies[0] #-170*10**3; # -47.8*4.184*1*10**3; #enthalpic contribution of binding (J/mol)
    #print(DeltaS0)
    #print(DeltaH0)

    DeltaG0 = DeltaH0/(Rg*T)-DeltaS0/Rg;
    #print('Delta G0 at this temperature is (kT units)')
    #print(DeltaG0)
    #print('Delta H0 and Delta S0 are, DeltaG0 at 50C')
    #print(DeltaH0,DeltaS0,DeltaH0/(Rg*((273.15+50.0)))-DeltaS0/Rg)
    #print('densities',s1,s2,sb,st)    
    
    if sb < st:
        sigmaSticky = st #highest sticky density density
        fractionSticky = sb/st
    elif st < sb:
        sigmaSticky = sb #highest density
        fractionSticky = st/sb #here f is smaller than 1
    else: # they are equal
        sigmaSticky = sb
        fractionSticky = 1 # in that case same amount of sticky top and bottom
    #print('This is the fraction of sticky ends', sb, st, fractionSticky)
    
    
    # for now I will assume fraction sticky is 1, and same density top and bottom... 
    #h1max = hMilner(s1,ell1,N1,eV1)  
    #print("thickness of 1 side",h1max)
    #Pvalues,fvalues,Lvalues,potentialExcluded,potentialStretch,potentialBridge = \
    #    facingPotentialUnifiedBridge(N1,ell1,s1,N2,ell2,s2,allhs,DeltaG0,fractionfed,sigmaSticky,fractionSticky)
    
    ssticky1 = st
    ssticky2 = sb
    
    fvalues,potentialExcluded,potentialStretch,potentialBridge = \
        facingPotentialUnifiedBridgeAssymetric(N1,ell1,s1,eV1,N2,ell2,s2,eV2,allhs,DeltaG0,fractionfed,ssticky1,ssticky2)
    
    
    #print(Lvalues)
    #print(allhs)
    #Lvalue = DerjaguinIntegral(allhs[-1],Radius,allhs,[2*p for p in potentialBridge],optcolloid)
    
    # this was in the old symmetric days
    #phiBridge = DerjaguinIntegral(allhs[-1],Radius,allhs,[2*p for p in potentialBridge],optcolloid)
    #phiSter = DerjaguinIntegral(allhs[-1],Radius,allhs,2*potentialStretch+2*potentialExcluded,optcolloid)

    phiBridge = DerjaguinIntegral(allhs[-1],Radius,allhs,[p for p in potentialBridge],optcolloid)
    phiSter = DerjaguinIntegral(allhs[-1],Radius,allhs,potentialStretch+potentialExcluded,optcolloid)
    
    
    sigmaBridge = DerjaguinIntegral(allhs[-1],Radius,allhs,fvalues,optcolloid) 
    #sigmaBridge is actually equal to the number of bonds in contact
    #print(phiBridge)
    #print(phiSter)
    #return(phiBridge,sigmaBridge,allpunboundt,allpunboundb)
    # fractionSticky*sigmaSticky that's the maximum bonds that you can make so it makes sense to renormalize in that way. 
    return(phiBridge,phiSter,sigmaBridge,fvalues/(fractionSticky*sigmaSticky),fractionSticky*sigmaSticky,DeltaG0)
    
    
