#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:52:39 2019

@author: sm8857
"""

import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from math import sqrt, log, exp, erf, sinh, cosh, pi
from numba import njit, float32, int32
from numba import jit
import time

h0s = np.logspace(-10,-5.5,100)
Ts = np.linspace(20,100,20)
Cs = np.logspace(-5,3,20)
spinthickPS = 80*10**(-9)

#@numba.njit(float32(float32))
#def epsilonW(xin: float) -> float:
@jit(float32(float32),nopython=True,cache=True)
def epsilonW(xin):   
    
    WjWir = [0.315,1.05,1.40,3.04,6.38]
    FjWir = [0.144, 0.808,0.295,1.31,3.12]
    GjWir = [0.228,0.577,0.425,0.380,0.851]
    
    WjWuv = [1.25,1.52,1.73,1.98,2.26,2.81]
    FjWuv = [0.0619,0.131,0.277,0.607,0.780,2.14]
    GjWuv = [0.0775,0.134,0.234,0.311,0.450,0.951]
    
    DjWmw = [74.8]
    TjWmw = [1.05*10**(11)]    
    
    sumS = 1
    for i in range(1):
        sumS = sumS + DjWmw[i]/(1+xin/TjWmw[i])
    
    for i in range(5):
        sumS = sumS + FjWir[i]/(WjWir[i]**2 + xin**2*10**(-28) + GjWir[i]*10**(-14)*xin)
    
    for i in range(6):
        sumS = sumS + FjWuv[i]/(WjWuv[i]**2 + xin**2*10**(-32) + GjWuv[i]*10**(-16)*xin)
        
    
    return sumS
    


#print(epsilonW(0))

#@numba.njit(float32(float32))
#def epsilonG(xin: float) -> float:
@jit(float32(float32),nopython=True,cache=True)
def epsilonG(xin):
    
    WuvG = [1.911]
    CuvG = [1.282]
    sumS = 1+ CuvG[0]/(1 + xin**2*10**(-32)/WuvG[0]**2)
    
    return sumS
    

#print(epsilonW(0))

#@numba.njit(float32(float32))
#def epsilonPS(xin: float) -> float:
@jit(float32(float32),nopython=True,cache=True)
def epsilonPS(xin):
    WjPS = [0.965,2.13,1.67,3.05]
    FjPS = [0.337,2.24,1.03,3.16]
    GjPS = [0.0988,0.760,0.532,1.75]
    
    sumS = 1
    for i in range(4):
        sumS = sumS + FjPS[i]/(WjPS[i]**2 + xin**2*10**(-32) + GjPS[i]*10**(-16)*xin)
    
    return sumS


#print(epsilonPS(0))
##@numba.njit(float32(float32,float32))
##def rn(xin: float, h:float) -> float:
@jit(float32(float32, float32),cache=True,nopython=True)
def rn(xin,h):
    c = 299792458
    return 2*h*xin*sqrt(epsilonW(xin))/c

#print(rn(0,0))
##@numba.njit(float32(float32))
##def sW(x: float) -> float:
@jit(float32(float32),cache=True,nopython=True)
def sW(x):
    return x

#@numba.njit(float32(float32,float32,float32))
#def sPS(x: float,xin: float, h:float) -> float:
@jit(float32(float32, float32, float32),cache=True,nopython=True)
def sPS(x,xin,h):
    c = 299792458
    return sqrt(x**2 + (2*xin*h/c)**2*(epsilonPS(xin) - epsilonW(xin)))

@jit(float32(float32, float32, float32),cache=True,nopython=True)
def sG(x,xin,h):
    c = 299792458
    return sqrt(x**2 + (2*xin*h/c)**2*(epsilonG(xin) - epsilonW(xin)))

#@numba.njit(float32(float32, float32, float32))
#def DeltaWS(x: float,xin: float, h:float) -> float:
@jit(float32(float32, float32, float32),cache=True,nopython=True)
def DeltaWS(x,xin,h):
    if x ==0:
        ratio = 0
    else: 
        if xin == 0:
            ratio = (epsilonW(xin) - epsilonPS(xin))/(epsilonW(xin) + epsilonPS(xin))
        else:
            ratio = (epsilonW(xin)*sPS(x,xin,h) - epsilonPS(xin)*sW(x))/(epsilonW(xin)*sPS(x,xin,h) + epsilonPS(xin)*sW(x))
    return ratio


#@numba.njit(float32(float32, float32, float32))
#def DeltaWS(x: float,xin: float, h:float) -> float:

@jit(float32(float32, float32, float32),cache=True,nopython=True)
def DeltaSG(x,xin,h):
    if x ==0:
        ratio = 0
    else: 
        if xin == 0:
            ratio = (epsilonPS(xin) - epsilonG(xin))/(epsilonPS(xin) + epsilonG(xin))
        else:
            ratio = (epsilonPS(xin)*sG(x,xin,h) - epsilonG(xin)*sPS(x,xin,h))/(epsilonPS(xin)*sG(x,xin,h) + epsilonG(xin)*sPS(x,xin,h))
    return ratio

@jit(float32(float32, float32, float32, float32),cache=True,nopython=True)
def DeltaWGeff(x,xin,h,spinthick):
    ratio = (DeltaSG(x,xin,h)*exp(-sPS(x,xin,h)*spinthick/h) + DeltaWS(x,xin,h))/(1 + DeltaSG(x,xin,h)*DeltaWS(x,xin,h)*exp(-sPS(x,xin,h)*spinthick/h))
    return ratio

#@numba.njit(float32(float32, float32, float32))
#def DeltabarWS(x: float,xin: float, h:float) -> float:
@jit(float32(float32, float32, float32),cache=True,nopython=True)
def DeltabarWS(x,xin,h):
    if x ==0:
        ratio = 0
    else:
        if xin == 0:
            ratio = 0
        else:
            ratio = (sPS(x,xin,h) - sW(x))/(sPS(x,xin,h) + sW(x))
    return ratio



@jit(float32(float32, float32, float32),cache=True,nopython=True)
def DeltabarSG(x,xin,h):
    if x ==0:
        ratio = 0
    else:
        if xin == 0:
            ratio = 0
        else:
            ratio = (sG(x,xin,h) - sPS(x,xin,h))/(sG(x,xin,h) + sPS(x,xin,h))
    return ratio
    


@jit(float32(float32, float32, float32, float32),cache=True,nopython=True)
def DeltabarWGeff(x,xin,h,spinthick):
    ratio = (DeltabarSG(x,xin,h)*exp(-sPS(x,xin,h)*spinthick/h) + DeltabarWS(x,xin,h))/(1 + DeltabarSG(x,xin,h)*DeltabarWS(x,xin,h)*exp(-sPS(x,xin,h)*spinthick/h))
    return ratio


#@numba.njit(float32(float32, float32, float32))
#def integrandVdW(x: float,xin: float, h:float) -> float:
@jit(float32(float32, float32, float32, float32),cache=True,nopython=True)
def integrandVdW(x,xin,h,spinthick):
    return x*(log(1 - DeltaWS(x,xin,h)*DeltaWGeff(x,xin,h,spinthick)*exp(-x))+ log(1 - DeltabarWS(x,xin,h)*DeltabarWGeff(x,xin,h,spinthick)*exp(-x)))


#print(integrandVdW(10,0,10**(-9)))
##@numba.njit(float32(float32, float32, float32))
##def hamaker(lambdaD: float,T: float,h: float) -> float: # find the retarded hamaker constant at distance h

@jit(float32(float32, float32, float32, float32),cache=True)
def hamaker(lambdaD,T,h,spinthick):
        hbar = 1.054571817*10**(-34)
        kB = 1.380658*10**-23
        mpi = 3.14159265359
        A123 = 0
        for n in range(200): # accuracy at 3rd digit
            if n==0:
                prefactor = 1/2*(1+2*h/lambdaD)*exp(-2*h/lambdaD)
            else:
                prefactor=1
                
            xinv = 2*mpi*n*kB*T/hbar
            #print(epsilonW(xinv)) # the dielectric conductivities make sense.
            #print(epsilonPS(xinv))
            #print(DeltabarWS(rn(xinv),xinv))
            #print(xinv/c)
            
                        
            if n==0:
                rmax = 100
            else:
                rmax = 10*rn(xinv,h)
                avalue = integrandVdW(rmax,xinv,h,spinthick)
                while abs(avalue) > 0 :
                    rmax = rmax*2
                    avalue = integrandVdW(rmax,xinv,h,spinthick) # this is to make sure the integration will go as far as needed. 
            
            #hplots = np.linspace(rn(xinv),100,1000)
            #plt.plot(hplots,[integrandVdW(hp) for hp in hplots])
            #plt.show()
            
            #print(xinv)
            #print(log(1 - DeltaWS(10,xinv)**2*exp(-10)))
            #rmax = h0s[-1]*kB*T/(hbar*c)
            #print(rmax)
            #print(rmax)
#            print(n)
#            a123n,err = quad(integrandVdW,rn(xin),rmax*1000)
#            print(a123n)
#            a123n,err = quad(integrandVdW,rn(xin),rmax*10000)
#            print(a123n)
            
            #print(rn(xinv))        
            a123n,err = quad(integrandVdW,rn(xinv,h),rmax,args=(xinv,h,spinthick))
            #print("the contribution to the sum is")
            #print(a123n)
            #a123n,err = quad(integrandVdW,rn(xinv),rmax*10) 
            #print(a123n)
            #a123n,err = quad(integrandVdW,rn(xinv),rn(xinv)*10000000)
            #print(a123n) #those were helpful to show that this is indeed the right bound
            # to work with to have a nice precision
            
            A123 = A123+prefactor*a123n
            #print(A123)
            
        A123 = -3/2*A123     
        #print(A123)
        #A123 = 13*10**(-21)
        #A123 = 0
        #print(A123)
        return A123
        #return(A123)

        #return(A123)


kB = 1.380658*10**-23
electron = 1.6*10**-19; #elementary electron charge
Na = 6.02*10**23;
spinthickPS = 80*10**-9 #thickness of spin coated PS

t0 = time.time()
for Tcelsius in Ts:
    print(time.time()-t0)
    t0 = time.time()
    for c0s in Cs:
        T = 273+Tcelsius
        c0salt = (c0s*10**3)*Na; #salt concentration in parts/m^3 (the first number is in mol/L)
        waterDensity = [[24,1.00596],[30,1.00298],[35,0.99911],[40,0.99583],[45,0.99154],[50,0.98699],[55,0.98122],[60,0.97606]] ####

        def density(T) :
            
            Tc = []
            densities = []
            for i in range(len(waterDensity)):
                Tc.append(waterDensity[i][0])
                densities.append(waterDensity[i][1])
            
            densities = [densities[i]*10**-3/(1*10**-6) for i in range(len(densities))]
            tck = interpolate.splrep(Tc, densities, s=None,k=1)
            return(interpolate.splev(T-273, tck, der=0))

        def permittivity(T,rhom) :

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

        epsilon0r = permittivity(T,density(T));
        lambdaD = sqrt(epsilon0r*kB*T/(electron**2*c0salt));

        for i in range(len(h0s)):
            Fvalue = hamaker(lambdaD,T,h0s[i],spinthickPS)
            f= open("VanDerWaals_parameterSweep_GlassPS80WaterPS.txt","a+")
            f.write('{:.16e}'.format(Tcelsius)+','+'{:.16e}'.format(c0s)+','+'{:.16e}'.format(h0s[i])+','+'{:.16e}'.format(Fvalue)+'\n')
            f.close()
...
