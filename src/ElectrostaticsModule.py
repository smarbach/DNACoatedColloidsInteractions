#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:47:06 2020

@author: sm8857
"""

import numpy as np
from scipy.integrate import trapz

import matplotlib.pyplot as plt

from math import sqrt, log, exp, erf, sinh, cosh, pi, tanh, asinh
from WaterDensityModule import density
from scipy.integrate import quad
from CommonToolsModule import DerjaguinIntegral

## This whole file is a bunch of electrostatics helper

def permittivity(T,rhom) :
    
        # Calculates the permittivity of water at any temperature and density
        e0 =(4*10**-7*pi*(299792458)**2)**(-1);
        kB = 1.380658*10**-23; 
        Mw = 0.018015268;
        Na = 6.0221367*10**23;
        alpha = 1.636*10**-40;
        mu = 6.138*10**-30;
        rhoc = 322/Mw;
        Tc = 647.096;
        rho = rhom/Mw;
    
        N = [0]*12
        N[0] = 0.978224486826;
        N[1] = -0.957771379375;
        N[2] = 0.237511794148;
        N[3] = 0.714692244396;
        N[4] = -0.298217036956;
        N[5] = -0.108863472196;
        N[6] = 10**-1*0.949327488264;
        N[7] = -1*10**-2*0.980469816509;
        N[8] = 10**-4*0.165167634970;
        N[9] = 10**-4*0.937359795772;
        N[10] = -1*10**-9*0.123179218720;
        N[11] = 10**-2*0.196096504426;
    
        ik = [1,1,1,2,3,3,4,5,6,7,10];
        jk = [0.25,1,2.5,1.5,1.5,2.5,2,2,5,0.5,10];
        q = 1.2;
    
        g = 1 
        for i in range(0,11) : 
            g = g + N[i]*(rho/rhoc)**(ik[i])*(Tc/T)**(jk[i])
        g = g + N[11]*(rho/rhoc)*(T/228 - 1)**(-q);
    
        A = Na*mu**2*rho*g/(e0*kB*T);
        B = Na*alpha/(3*e0)*rho;
    
        epsilon = (1 + A + 5*B + sqrt(9 + 2*A + 18*B + A**2 + 10*A*B + 9*B**2))/(4 - 4*B);
    
        epsilonTot = e0*epsilon;
        
        return(epsilonTot)

def lambdaD(T,c0salt):
    
    kB = 1.380658*10**-23;
    electron = 1.6*10**-19 #elementary electron charge

    epsilon0r = permittivity(T,density(T))
    #print(epsilon0r)
    #print(density(T))
    lambdaD = sqrt(epsilon0r*kB*T/(electron**2*2*c0salt))
    return(lambdaD)
    
def Graham(T,lambdaD,sigma):
    
    kB = 1.380658*10**-23
    epsilon0r = permittivity(T,density(T))
    electron = 1.6*10**-19 #elementary electron charge
    
    gamma = tanh((1/2)*asinh(sigma*lambdaD*electron/(2*kB*T*epsilon0r)))
    return(gamma)  


def phiHd(rhoDNA,heq,lambdaD):
    
     phiDonnan = asinh(rhoDNA)
     phiHd = sinh(phiDonnan)*phiDonnan + (1-cosh(phiDonnan))
    
     return(phiHd)      


def phiH(rhoDNA,heq,lambdaD):
    
    def intePhi(z):
        return(3*rhoDNA*(1-(z/heq)**2)**(1/2)*(z/heq)*sinh(z/lambdaD)*exp(-heq/lambdaD))
    
    phiH,err = quad(intePhi,0,heq)
    
    return(phiH)
    
def gammaH(rhoDNA,heq,lambdaD):
    Vh = phiHd(rhoDNA,heq,lambdaD)
    gamma = tanh(Vh/4)
    return(gamma)

def gammaHs(rhoDNA,heq,lambdaD):
    Vh = phiH(rhoDNA,heq,lambdaD)
    gamma = tanh(Vh/4)
    return(gamma)  
      

def coth(x):
    return(1/tanh(x))



def electrostaticPotential(allhs,h01,h02,hP1s,hP2s,h1s,h2s,lambdaD,c0salt,sigmaDNA1,sigmaDNA2,Radius,sigmab,sigmat,gammab,gammat,allsigmaabs,sigmasticky,DNAcharge1,DNAcharge2,optcolloid):
    
    # comment that out if you want electrostatics to start at the tip of the brush.. Uncomment if you want to start at average brush height
    #hP1s = [3*pi/16*hP1s[ip] for ip in range(len(allhs))]
    #hP2s = [3*pi/16*hP2s[ip] for ip in range(len(allhs))]
    #h1s = [3*pi/16*h1s[ip] for ip in range(len(allhs))]
    #h2s = [3*pi/16*h2s[ip] for ip in range(len(allhs))]
    
    h1eq =  h1s[-1] + hP1s[-1]
    h2eq =  h2s[-1] + hP2s[-1]
    
    #empirical formula from data  - because the onset of electrostatics is actually a little later than expected...
    #reducFactor1 = ((h1eq/lambdaD - 5)*(-2.5)/35 + h1eq/lambdaD)/(h1eq/lambdaD)
    #reducFactor2 = ((h2eq/lambdaD - 5)*(-2.5)/35 + h2eq/lambdaD)/(h2eq/lambdaD)
    
    #print(reducFactor1)
    #print(reducFactor2)
    
    #hP1s = [hP1s[ip]*reducFactor1 for ip in range(len(allhs))] #this corresponds to the max of the parabola (1/sqrt(2))
    #hP2s = [hP2s[ip]*reducFactor2 for ip in range(len(allhs))]
    #h1s = [h1s[ip]*reducFactor1 for ip in range(len(allhs))]
    #h2s = [h2s[ip]*reducFactor2 for ip in range(len(allhs))]
    #h0 => inert coatings
    #hP => polymer coatings
    #h => DNA (charged) coatings
    
    #mode = 1
    
    #heightFactor = 1 #could be the average 3*pi/16. Just trying something here, beware
    
    electron = 1.6*10**-19 #elementary electron charge
    
    allrhoDNA1 = 0
    allrhoDNA2 = 0
    st = sigmat/(2*electron*c0salt)
    sb = sigmab/(2*electron*c0salt)
    phiEl = 0*allhs
    phiEl2 = 0*allhs
    phiEl3 = 0*allhs
    phiDL = 0*allhs
    phiDL2 = 0*allhs
    realH = 0*allhs
    h1eq =  h1s[-1] + hP1s[-1]
    h2eq =  h2s[-1] + hP2s[-1]
    
    
    heq = h1s[-1] + hP1s[-1] + h2s[-1] + hP2s[-1]
    
    prefactor = lambdaD*c0salt
    
    phiDLend = 0 
    
    if abs(sigmaDNA2) > 0 or abs(sigmaDNA1) > 0:
        
        for ip in range(len(h1s)):
                
            # the 1st step is to calculate the surface profile & then you do a Derjaguin integral
            
            # For that as a starting point I need the density of charges everywhere
            h1p = h1s[ip]+hP1s[ip]
            h2p = h2s[ip]+hP2s[ip]
            
            
            
            hl = allhs[ip] - h01 - h02
            
            #print(hl)
            #print(heq)
            
            model = 1 # this will take parabolic profiles being uniformly squished 
            #model = 2 # this will neglect concentration profiles interpenetrating
            #model = 3 # this will take parabolic profiles, but with sticky proportions given by hybridization
            #model = 4 # this will take uniform distributions, being uniformly squished 
            
            if phiDLend > 0 and hl > heq + 10*lambdaD:
                phiDL[ip] = phiDLend
            else:
                
                if model == 1:
                    # this will take parabolic profiles being uniformly squished 
                    allrhoDNA2 = sigmaDNA2/(2*electron*c0salt*lambdaD) #work in non dimensional scales
                    allrhoDNA1 = sigmaDNA1/(2*electron*c0salt*lambdaD) #work in non dimensional scales
                    
                    def rhoC(z):
                        if z*lambdaD < h1p:
                            t = z
                            H = h1p/lambdaD
                            rho = 3/2*allrhoDNA1*(1-(t/H)**2)/H
                        elif z*lambdaD > hl - h2p:
                            t = hl/lambdaD - z
                            H = h2p/lambdaD
                            rho = 3/2*allrhoDNA2*(1-(t/H)**2)/H
                        else:
                            rho = 0
                        return(rho)
                
                elif model == 2:
                    # this will neglect concentration profiles interpenetrating -- no squishing
                    allrhoDNA2 = sigmaDNA2/(2*electron*c0salt*lambdaD) #work in non dimensional scales
                    allrhoDNA1 = sigmaDNA1/(2*electron*c0salt*lambdaD) #work in non dimensional scales
                    
                    def rhoC(z):
                        rho = 0
                        if z*lambdaD < h1eq:
                            t = z
                            H = h1eq/lambdaD # key point for profiles NOT interpenetrating
                            rho = 3/2*allrhoDNA1*(1-(t/H)**2)/H
                        if z*lambdaD > hl - h2eq:
                            t = hl/lambdaD - z
                            H = h2eq/lambdaD
                            rho = 3/2*allrhoDNA2*(1-(t/H)**2)/H
                        
                        return(rho)
                
                elif model == 3:
                    
                    # this will take parabolic profiles, but with sticky proportions given by hybridization
                    
                    sigmaDNA1p = allsigmaabs[ip]*sigmasticky*DNAcharge1
                    sigmaDNA2p = allsigmaabs[ip]*sigmasticky*DNAcharge2
                    
                    allrhoDNA2 = sigmaDNA2p/(2*electron*c0salt*lambdaD) 
                    allrhoDNA1 = sigmaDNA1p/(2*electron*c0salt*lambdaD) 
                    
                    
                    def rhoC(z):
                        if z*lambdaD < h1p:
                            t = z
                            H = h1p/lambdaD
                            rho = 3/2*allrhoDNA1*(1-(t/H)**2)/H
                        elif z*lambdaD > hl - h2p:
                            t = hl/lambdaD - z
                            H = h2p/lambdaD
                            rho = 3/2*allrhoDNA2*(1-(t/H)**2)/H
                        else:
                            rho = 0
                        return(rho)
                    
                    
                
                elif model == 4:
                    
                     # this will take uniform distributions, being uniformly squished 
                    
                    allrhoDNA2 = sigmaDNA2/(2*electron*c0salt*(h2p))
                    allrhoDNA1 = sigmaDNA1/(2*electron*c0salt*(h1p))
                    
                    def rhoC(z):
                        if z*lambdaD < h1p:
                            rho = allrhoDNA1
                        elif z*lambdaD > hl - h2p:
                            rho = allrhoDNA2
                        else:
                            rho = 0
                        return(rho)
                
                
                #allRhoC = [rhoC(hv) for hv in allhsEval]
                
                #plt.plot(allRhoC)
                #plt.show()
                
                def int1(zp):
                    return(-exp(-zp)/2*rhoC(zp))
                    
                def int2(zp):
                    return(exp(zp-hl/lambdaD)/2*rhoC(zp))
                    
                try:
                    f1,err = quad(int1,0,hl/lambdaD)
                      
                    try:
                        f2,err = quad(int2,0,hl/lambdaD)
                        em2h = exp(-2*hl/lambdaD)
                        #emh = exp(-hl/lambdaD)
                        
                        def phiDens(z):
                            #ez = exp(z)
                            #ez = exp(z)
                            emzh = exp(-z -hl/lambdaD)
                            ezm2h = exp(z -2*hl/lambdaD)
                            #emz2h = exp(-z-2*hl/lambdaD)                        
                            emz = exp(-z)
                            ezmh = exp(z-hl/lambdaD)
                            ehmz = exp(-z + hl/lambdaD)
    
                            f1z,err = quad(int1,0,z)     
                            f2z,err = quad(int2,0,z)  
                            
                            def int1h(zp):
                                return(exp(-zp+z)/2*rhoC(zp))
                            allhsEval = np.linspace(z,hl/lambdaD,100)
                            f1h = [int1h(hv) for hv in allhsEval]
                            F1zh = trapz(f1h,allhsEval)
                            
                            #F1zh,err = quad(int1h,z,hl/lambdaD)
                            
                            # def int2h(zp):
                            #     return(exp(zp-z)/2*rhoC(zp))
                            
                            # allhsEval = np.linspace(z,hl/lambdaD,200)
                            # f2h = [int2h(hv) for hv in allhsEval]
                            # F2zh = trapz(f2h,allhsEval)
    
                            
                            #print(F2zh)
                            
                            
                            #Ve = 1/(1-em2h)*(sb*(emz +ezm2h)   +st*(emzh+ ezmh)) \
                            #    +  1/(1-em2h)*(- (emz+ez)*f1 + (-ezm2h + ez)*f1z + (emzh +ezmh)*f2 + (ehmz -emzh)*f2z)
                            
                            #dzVe = 1/(1-em2h)*(sb*(-emz + ezm2h)   +st*(- emzh+ ezmh)) \
                            #    +  1/(1-em2h)*( (emz-ez)*f1 + (-ezm2h + ez)*f1z + (-emzh +ezmh)*f2 + (-ehmz +emzh)*f2z)
                            
                            
                            Ve = 1/(1-em2h)*(sb*(emz +ezm2h)   +st*(emzh+ ezmh)) \
                                +  1/(1-em2h)*(- (emz)*f1 + (-ezm2h)*f1z + F1zh + (emzh +ezmh)*f2 + (ehmz -emzh)*f2z)
                            
                            dzVe = 1/(1-em2h)*(sb*(-emz + ezm2h)   +st*(- emzh+ ezmh)) \
                                +  1/(1-em2h)*( (emz)*f1 + (-ezm2h)*f1z + F1zh + (-emzh +ezmh)*f2 + (-ehmz +emzh)*f2z)
                            
                            
                            rhop = exp(-Ve);
                            rhom = exp(Ve);
                            return(dzVe**2 + rhop*log(rhop) + rhom*log(rhom) - rhop - rhom + 2)
                
                        
                        try:
                            
                            allhsEval = np.linspace(0,hl/lambdaD,100)
                            phiDensEval = [phiDens(hv) for hv in allhsEval]
                            
                            #print('print phiDensEval')
                            #print(phiDensEval)
                            phiDL[ip] = trapz(phiDensEval,allhsEval)
                            
                            #phiDL[ip],err = quad(phiDens,0,hl/lambdaD)
                            #print('errors')
                            #print(phiDLquad)
                            #print(phiDL[ip])
                            
                            phiDLend = phiDL[ip]
                            #endID = ip
                        except OverflowError:
                            phiDL[ip] = float('inf')
                            print('overflow  1')
                    except OverflowError:
                        phiDL[ip] = float('inf')
                        print('overflow  2')
                except OverflowError:
                    phiDL[ip] = float('inf')
                    print('overflow  3')
        
        phiDL = phiDL - phiDLend
        
    #    print('print phiDL')
    #    print(phiDL)
        
    #    print(endID)
        phiClean = 0*phiDL
        for ip in range(len(phiDL)):
            if phiDL[ip] > 0:
                phiClean[ip] = phiDL[ip]
                
        
            
        # Finally do the Derjaguin integral
        phiEl = DerjaguinIntegral(allhs[-1],Radius,allhs,prefactor*phiClean,optcolloid)

    else:
        # then its only charged walls
        #gammab = Graham(T,lambdaD,sigmab);
        #gammat = Graham(T,lambdaD,sigmat)

        prefactorEl = pi*Radius*2*lambdaD**2*c0salt #you need 2piR because its not going through the dejarguin integration
    #    print('prefactor EL')
    #    print(prefactorEl)
    #    elif sigmaDNA2 > 0:
        for ip in range(len(h1s)):
    #            
            realH[ip] = allhs[ip] - h02  - h01   #only substract your side of the brush
            if optcolloid:
                phiEl[ip]= prefactorEl*64*gammat*gammab*exp(-realH[ip]/lambdaD)/2
            else:
                phiEl[ip]= prefactorEl*64*gammat*gammab*exp(-realH[ip]/lambdaD)            
   

    # FOR NOW -- BECAUSE I AM CODING VERY BADLY -- LET'S CALCULATE ALSO WITH THE OLD GOUY WAY
    # this old part of code, not used for now, does not work if there's no DNA (it doesn't take th GC profile for the charged surfaces)

#     gamma2s = 0*allhs
#     gamma1s = 0*allhs
#     realH = 0*allhs

    
#     rhoDNA2 = sigmaDNA2/(2*electron*c0salt*(h2eq))
#     rhoDNA1 = sigmaDNA1/(2*electron*c0salt*(h1eq))
#     phi2 = phiHd(rhoDNA2,h2eq,lambdaD)
#     phi1 = phiHd(rhoDNA1,h1eq,lambdaD)
    
# #    print(lambdaD)
# #    print(c0salt)
#     prefactorEl = pi*Radius*2*lambdaD**2*c0salt #you need 2piR because its not going through the dejarguin integration
# #    print('prefactor EL')
# #    print(prefactorEl)
# #    elif sigmaDNA2 > 0:
#     for ip in range(len(h1s)):
# #            
# #            if mode == 0:
# #                allrhoDNA2 = sigmaDNA2/(2*electron*c0salt*h2s[ip])
# #            else:

#         h2p = h2s[ip]+hP2s[ip]
#         allrhoDNA2 = sigmaDNA2/(2*electron*c0salt*(h2eq))

#         h1p = h1s[ip]+hP1s[ip]
#         allrhoDNA1 = sigmaDNA1/(2*electron*c0salt*(h1eq))
#         gamma2s[ip] = gammaH(allrhoDNA2,h2eq,lambdaD)
#         gamma1s[ip] = gammaH(allrhoDNA1,h1eq,lambdaD)

        

#         realH[ip] = allhs[ip] - h02 - h2eq - h01 - h1eq  #only substract your side of the brush
#         if realH[ip] > 0:
# #            # we can do the Gouy-Chapman approximation of the electrostatic potential if realH is positive
#             phiEl2[ip]= prefactorEl*64*gamma1s[ip]*gamma2s[ip]*exp(-realH[ip]/lambdaD)
#             phiDL2[ip]= 64*gamma1s[ip]*gamma2s[ip]*exp(-realH[ip]/lambdaD)
#         else:
# #                # otherwise you are in the squished part -- when it's something more hybrid like that it could be more complex

#             h =  allhs[ip] - h02 - h01  # because this measures the moment where it first compressed
#             phiRef = phi2*h2eq/heq +  phi1*h1eq/heq
#             phiEl2[ip]= prefactorEl*( 2*phiRef/2*((h-heq)/lambdaD)**2 + 64*gamma1s[ip]*gamma2s[ip]*(h/lambdaD - heq/lambdaD + 1))    
#             #print(realH[ip]/lambdaD)
#             phiDL2[ip]= 64*gamma1s[ip]*gamma2s[ip]*exp(-realH[ip]/lambdaD)
            

    
#      # FOR NOW -- BECAUSE I AM CODING VERY BADLY -- LET'S CALCULATE ALSO WITH THE SUPER SQUISHY GOUY WAY
#     gamma2s = 0*allhs
#     gamma1s = 0*allhs
#     realH = 0*allhs

    
#     rhoDNA2 = sigmaDNA2/(2*electron*c0salt*(h2eq))
#     rhoDNA1 = sigmaDNA1/(2*electron*c0salt*(h1eq))
#     phi2 = phiHd(rhoDNA2,h2eq,lambdaD)
#     phi1 = phiHd(rhoDNA1,h1eq,lambdaD)
    
#     prefactorEl = pi*Radius*2*lambdaD**2*c0salt
    
# #    elif sigmaDNA2 > 0:
#     for ip in range(len(h1s)):
# #            
# #            if mode == 0:
# #                allrhoDNA2 = sigmaDNA2/(2*electron*c0salt*h2s[ip])
# #            else:

#         h2p = h2s[ip]+hP2s[ip]
#         allrhoDNA2 = sigmaDNA2/(2*electron*c0salt*(h2p))

#         h1p = h1s[ip]+hP1s[ip]
#         allrhoDNA1 = sigmaDNA1/(2*electron*c0salt*(h1p))
#         gamma2s[ip] = gammaHs(allrhoDNA2,h2eq,lambdaD)
#         gamma1s[ip] = gammaHs(allrhoDNA1,h1eq,lambdaD)

#         realH[ip] = allhs[ip] - h02 - h2eq - h01 - h1eq  #only substract your side of the brush
#         if realH[ip] > 0:
# #            # we can do the Gouy-Chapman approximation of the electrostatic potential if realH is positive
#             phiEl3[ip]= prefactorEl*64*gamma1s[ip]*gamma2s[ip]*exp(-realH[ip]/lambdaD)
#         else:
# #                # otherwise you are in the squished part -- when it's something more hybrid like that it could be more complex

#             h =  allhs[ip] - h02 - h01  # because this measures the moment where it first compressed
#             phiRef = phi2*h2eq/heq +  phi1*h1eq/heq
#             phiEl3[ip]= prefactorEl*( 2*phiRef/2*((h-heq)/lambdaD)**2 + 64*gamma1s[ip]*gamma2s[ip]*(heq/lambdaD - h/lambdaD + 1))    
    
    
    # Some prints/plots to check profiles agree relatively well
    
    #print('printing phiDL G C method')
    #print(phiDL2)
    #fig, ax = plt.subplots()
    #ax.plot(allhs-h1eq-h2eq,phiEl,label='Linear approximation')
    #ax.plot(allhs-h1eq-h2eq,phiEl3,label='Squishy G-C method')
    #ax.plot(allhs-h1eq-h2eq,phiEl2,label='G-C method')
    #ax.legend(loc='upper right',
    #      fontsize=10,
    #      frameon=False)
    #plt.show()
    #print('printing Electrostatic profiles')
    #print(phiEl2)
    #print(phiEl)
    #print('printing distance')
    #print(allhs-h1eq-h2eq)
    
    
    # UNCOMMENT THIS IF YOU WANT ANOTHER POTENTIAL PROFILE
    #phiEl = phiEl2
    
    return(phiEl)
 
def electrostaticPotentialPlates(allhs,h01,h02,lambdaD,c0salt,Radius,gammab,gammat,optcolloid):
    


    phiEl = 0*allhs
    realH = 0*allhs


    
    prefactorEl = pi*Radius*2*lambdaD**2*c0salt #you need 2piR because its not going through the dejarguin integration
    #    print('prefactor EL')
    #    print(prefactorEl)
    #    elif sigmaDNA2 > 0:
    for ip in range(len(allhs)):
    #            
        realH[ip] = allhs[ip] - h02  - h01   #only substract your side of the brush
        if optcolloid:
            phiEl[ip]= prefactorEl*64*gammat*gammab*exp(-realH[ip]/lambdaD)/2
        else:
            phiEl[ip]= prefactorEl*64*gammat*gammab*exp(-realH[ip]/lambdaD)
                      
   

    
    return(phiEl)       