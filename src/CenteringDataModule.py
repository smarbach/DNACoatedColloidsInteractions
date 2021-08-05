#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:10:41 2020

@author: sm8857
"""

import matplotlib.pyplot as plt
from numpy import exp, pi, sqrt, log
import numpy as np

from lmfit import Model
from CommonToolsModule import CompareData


def centerNakedParticlePotential(height,potential,lambdaD,checkoption,cutForDataFit):

    # fit the profile only for values smaller than 120nm (beyond the results are floppy)

    indexH = next(x for x, val in enumerate(height) if val < cutForDataFit)
    
    #print(indexH)
    xFit = [height[i] for i in range(indexH,len(height))]
    yFit = [potential[ip] for ip in range(indexH,len(height))]
    
    #print(xFit,yFit)
    # Model for the fit
    def nakedParticlePotential(x, B, G,kappa, C):
        #naked particle potential fit
        return (B*exp(-x*kappa)+ G*x + C)
    
    gmodel = Model(nakedParticlePotential)
    result = gmodel.fit(yFit, x=xFit, B=1, G=0.01, kappa=1/(lambdaD*1e9), C = 0)

    #print("For info, results from the initial fit of the raw data")
    #print(result.fit_report())
    if checkoption == 1:
        
        plt.plot(height, potential, 'bo')
        plt.plot(xFit, result.init_fit, 'k--', label='initial fit')
        plt.plot(xFit, result.best_fit, 'r-', label='best fit')
        plt.legend(loc='best')
        plt.show()
        
    bestvalues = result.best_values
    kappaV = bestvalues.get("kappa",0)
    #kappaV = 1/(lambdaD*1e9)
    BV = bestvalues.get("B",0)    
    GV = bestvalues.get("G",0)
    CV = bestvalues.get("C",0)
    
    hmin = log(kappaV*BV/GV)/kappaV
    phimin = BV*exp(-kappaV*hmin) + GV*hmin + CV
    
    potentialCentered = [potential[i] - phimin for i in range(len(potential))]
    heightCentered = [height[i] - hmin for i in range(len(potential))]
    
    # fit the profile again now that it is centered and get the fitted value of kappa 

    indexH = next(x for x, val in enumerate(heightCentered) if val < cutForDataFit)
    
    xFit = [heightCentered[i] for i in range(indexH,len(heightCentered))]
    yFit = [potentialCentered[ip] for ip in range(indexH,len(heightCentered))]

    def nakedParticlePotentialClean(x, B, G,kappa):
        #naked particle potential fit
        return (B*exp(-x*kappa)+ G*x - B)
    
    gmodel2 = Model(nakedParticlePotentialClean)
    result = gmodel2.fit(yFit, x=xFit, B=1, G=0.01, kappa=1/(lambdaD*1e9))
    
    bestvalues = result.best_values
    kappaV = bestvalues.get("kappa",0)
    BV = bestvalues.get("B",0)    
    GV = bestvalues.get("G",0)
    
    hmin = log(kappaV*BV/GV)/kappaV
    #print(hmin)
    #print(result.fit_report())
    allheights = np.linspace(min(height),max(height),cutForDataFit)
    potentialFitted = [nakedParticlePotentialClean(allheights[i],BV,GV,kappaV) for i in range(len(allheights))]
    error = CompareData(height,potential,allheights,potentialFitted,cutForDataFit)
    
    return heightCentered, potentialCentered, kappaV, error



def centerPotentialRemoveGravity(height,potential,checkoption,gravmin,gravmax,h_adjust,h_adjustment,hframe):

    # 1 - remove gravity
    # fit the gravity profile only for values smaller than 120nm (beyond the results are floppy)
    # yet larger than 30 nm (hole)
    correctIndices = []
    for i in range(len(height)):
        val = height[i]
        if val < gravmax and val > gravmin:
            correctIndices.append(i)
                
    #print(indexH)
    xFit = [height[i] for i in correctIndices]
    yFit = [potential[ip] for ip in correctIndices]
    
    # Model for the fit
    def gravityPotential(x, G, C):
        #gravity potential
        return (G*x + C)
    
    gmodel = Model(gravityPotential)
    result = gmodel.fit(yFit, x=xFit, G=0.01, C = 0)

    #print("For info, results from the gravity fit of the raw data")
    #print(result.fit_report())
    
    if checkoption == 1:
        fig, ax = plt.subplots()
        ax.plot(height, potential, 'bo')
        ax.plot(xFit, result.init_fit, 'k--', label='gravity initial fit')
        ax.plot(xFit, result.best_fit, 'r-', label='gravity best fit')
        ax.legend(loc='best')
        plt.show()
        
    bestvalues = result.best_values
    GV = bestvalues.get("G",0)
    CV = bestvalues.get("C",0)
    
    potentialGravityRemoved = [potential[i] - GV*height[i] - CV for i in range(len(potential))]
    potential = potentialGravityRemoved
    GV = GV*1e9
    
    # 2 - "find the real minimum"
    # fit the potential profile only between -5 and 5
    correctIndices = []
    for i in range(len(height)):
        val = height[i]
        if val < hframe and val > -hframe:
            correctIndices.append(i)   
    
    #print(indexH)
    xFit = [height[i] for i in correctIndices]
    yFit = [potential[ip] for ip in correctIndices]
    
    #print(xFit,yFit)
    # Model for the fit
    def quadraticWell(x, A, B, C):
        #gravity potential
        return (A*x**2 + B*x + C)
    
    while len(xFit) < 3:
        hframe += 1
        correctIndices = []
        for i in range(len(height)):
            val = height[i]
            if val < hframe and val > -hframe:
                correctIndices.append(i)   
        
        #print(indexH)
        xFit = [height[i] for i in correctIndices]
        yFit = [potential[ip] for ip in correctIndices]
        
    # plt.plot(xFit,yFit)
    # plt.show()
    # #plt.plot(height,potential)
    #plt.show()
        
    gmodel = Model(quadraticWell)
    result = gmodel.fit(yFit, x=xFit, A=0, B= 0, C = 0)

    #print("For info, results from the minimum fit of the raw data")
    #print(result.fit_report())
    

    if checkoption == 1:
        fig, ax = plt.subplots()
        ax.plot(height, potential, 'bo')
        ax.plot(xFit, result.init_fit, 'k--', label='minimum initial fit')
        ax.plot(xFit, result.best_fit, 'r-', label='minimum best fit')
        ax.legend(loc='best')
        ax.set_xlim(min(xFit)*1.3,max(xFit)*1.3)
        ax.set_ylim(min(yFit)*1.3,max(yFit)*1.3)
        plt.show()
        
    bestvalues = result.best_values
    AV = bestvalues.get("A",0)
    BV = bestvalues.get("B",0)
    CV = bestvalues.get("C",0)
    
    hmin = - BV/(2*AV)
    #phimin = CV - BV**2/(4*AV)
    
    potentialCentered = [potentialGravityRemoved[i] for i in range(len(potential))] # don't remove phimin here, this part was set up by gravity
    heightCentered = [height[i] - hmin*h_adjust + h_adjustment for i in range(len(potential))]
    
    return heightCentered, potentialCentered, GV

def centerPotentialGravity(height,potential,checkoption,gravmin,gravmax):

    
    # 1 - "find the real minimum"
    # fit the potential profile only between -5 and 5
    correctIndices = []
    for i in range(len(height)):
        val = height[i]
        if val < 5 and val > -5:
            correctIndices.append(i)    
    #print(indexH)
    xFit = [height[i] for i in correctIndices]
    yFit = [potential[ip] for ip in correctIndices]
    
    #print(xFit,yFit)
    # Model for the fit
    def quadraticWell(x, A, B, C):
        #gravity potential
        return (A*x**2 + B*x + C)
    
    gmodel = Model(quadraticWell)
    result = gmodel.fit(yFit, x=xFit, A=0, B= 0, C = 0)

    #print("For info, results from the minimum fit of the raw data")
    #print(result.fit_report())

    if checkoption == 1:
        fig, ax = plt.subplots()
        ax.plot(height, potential, 'bo')
        ax.plot(xFit, result.init_fit, 'k--', label='minimum initial fit')
        ax.plot(xFit, result.best_fit, 'r-', label='minimum best fit')
        ax.legend(loc='best')
        ax.set_xlim(min(xFit)*1.3,max(xFit)*1.3)
        ax.set_ylim(-2,2)
        plt.show()
        
    bestvalues = result.best_values
    AV = bestvalues.get("A",0)
    BV = bestvalues.get("B",0)
    CV = bestvalues.get("C",0)
    
    hmin = - BV/(2*AV)
    #phimin = CV - BV**2/(4*AV)
    
    potentialCentered = [potential[i] for i in range(len(potential))] # don't remove phimin here, this part was set up by gravity
    heightCentered = [height[i] - hmin for i in range(len(potential))]
    
    # 2 - align gravity potential
    # fit the gravity profile only for values smaller than 120nm (beyond the results are floppy)
    # yet larger than 30 nm (hole)
    correctIndices = []
    for i in range(len(height)):
        val = height[i]
        if val < gravmax and val > gravmin:
            correctIndices.append(i)
                
    #print(indexH)
    xFit = [height[i] for i in correctIndices]
    yFit = [potential[ip] for ip in correctIndices]
    
    #print(xFit,yFit)
    # Model for the fit
    def gravityPotential(x, G, C):
        #gravity potential
        return (G*x + C)
    
    gmodel = Model(gravityPotential)
    result = gmodel.fit(yFit, x=xFit, G=0.01, C = 0)

    #print("For info, results from the gravity fit of the raw data")
    #print(result.fit_report())
    
    if checkoption == 1:
        fig, ax = plt.subplots()
        ax.plot(height, potential, 'bo')
        ax.plot(xFit, result.init_fit, 'k--', label='gravity initial fit')
        ax.plot(xFit, result.best_fit, 'r-', label='gravity best fit')
        ax.legend(loc='best')
        plt.show()
        
    bestvalues = result.best_values
    CV = bestvalues.get("C",0)
    GV = bestvalues.get("G",0)*1e9

    potentialGravityAlined= [potentialCentered[i] - CV for i in range(len(potential))]
    
    
    return heightCentered, potentialGravityAlined, GV

def centerPotentialGravityKnown(height,potential,gravity,checkoption):

    
#    # 1 - "find the real minimum"
#    # fit the potential profile only between -5 and 5
#    correctIndices = []
#    for i in range(len(height)):
#        val = height[i]
#        if val < 5 and val > -5:
#            correctIndices.append(i)    
#    #print(indexH)
#    xFit = [height[i] for i in correctIndices]
#    yFit = [potential[ip] for ip in correctIndices]
#    
#    #print(xFit,yFit)
#    # Model for the fit
#    def quadraticWell(x, A, B, C):
#        #gravity potential
#        return (A*x**2 + B*x + C)
#    
#    gmodel = Model(quadraticWell)
#    result = gmodel.fit(yFit, x=xFit, A=0, B= 0, C = 0)
#
#    #print("For info, results from the minimum fit of the raw data")
#    #print(result.fit_report())
#
#    if checkoption == 1:
#        fig, ax = plt.subplots()
#        ax.plot(height, potential, 'bo')
#        ax.plot(xFit, result.init_fit, 'k--', label='minimum initial fit')
#        ax.plot(xFit, result.best_fit, 'r-', label='minimum best fit')
#        ax.legend(loc='best')
#        ax.set_xlim(min(xFit)*1.3,max(xFit)*1.3)
#        ax.set_ylim(-2,2)
#        plt.show()
#        
#    bestvalues = result.best_values
#    AV = bestvalues.get("A",0)
#    BV = bestvalues.get("B",0)
#    CV = bestvalues.get("C",0)
#    
#    hmin = - BV/(2*AV)
#    #phimin = CV - BV**2/(4*AV)
#    
#    potentialCentered = [potential[i] for i in range(len(potential))] # don't remove phimin here, this part was set up by gravity
#    heightCentered = [height[i] - hmin for i in range(len(potential))]
    
    # 2 - align gravity potential
    # fit the gravity profile only for values smaller than 120nm (beyond the results are floppy)
    # yet larger than 30 nm (hole)
    correctIndices = []
    for i in range(len(height)):
        val = height[i]
        if val < 200 and val > 100:
            correctIndices.append(i)
                
    #print(indexH)
    xFit = [height[i] for i in correctIndices]
    yFit = [potential[ip] for ip in correctIndices]
    
    #print(xFit,yFit)
    # Model for the fit
    def gravityPotential(x, C):
        #gravity potential
        return (gravity*1e-9*x + C)
    
    gmodel = Model(gravityPotential)
    result = gmodel.fit(yFit, x=xFit, C = 0)

    #print("For info, results from the gravity fit of the raw data")
    #print(result.fit_report())
    
    if checkoption == 1:
        fig, ax = plt.subplots()
        ax.plot(height, potential, 'bo')
        ax.plot(xFit, result.init_fit, 'k--', label='gravity initial fit')
        ax.plot(xFit, result.best_fit, 'r-', label='gravity best fit')
        ax.legend(loc='best')
        ax.set_ylim(-2,8)
        ax.set_xlim(-20,200) 
        plt.show()
        
    bestvalues = result.best_values
    CV = bestvalues.get("C",0)

    potentialGravityRemoved = [potential[i] - CV for i in range(len(potential))]
    
    
    return height, potentialGravityRemoved



# def centerPotentialRemoveGravityKnown(height,potential,gravity,checkoption):

#     # 1 - remove gravity
#     # fit the gravity profile only for values smaller than 120nm (beyond the results are floppy)
#     # yet larger than 30 nm (hole)
#     correctIndices = []
#     for i in range(len(height)):
#         val = height[i]
#         if val < 200 and val > 100:
#             correctIndices.append(i)
                
#     #print(indexH)
#     xFit = [height[i] for i in correctIndices]
#     yFit = [potential[ip] for ip in correctIndices]
    
#     #print(xFit,yFit)
#     # Model for the fit
#     def gravityPotential(x, C):
#         #gravity potential
#         return (gravity*1e-9*x + C)
    
#     gmodel = Model(gravityPotential)
#     result = gmodel.fit(yFit, x=xFit, C = 0)

#     #print("For info, results from the gravity fit of the raw data")
#     #print(result.fit_report())
    
    
    
#     if checkoption == 1:
#         fig, ax = plt.subplots()
#         ax.plot(height, potential, 'bo')
#         ax.plot(xFit, result.init_fit, 'k--', label='gravity initial fit')
#         ax.plot(xFit, result.best_fit, 'r-', label='gravity best fit')
#         ax.legend(loc='best')
#         ax.set_xlim(min(xFit)*0.5,max(xFit)*1.3)
#         plt.show()
        
#     bestvalues = result.best_values
#     CV = bestvalues.get("C",0)
    
#     potentialGravityRemoved = [potential[i] - gravity*1e-9*height[i] - CV for i in range(len(potential))]
    
#     # 2 - "find the real minimum"
#     # fit the potential profile only between -5 and 5
#     correctIndices = []
#     for i in range(len(height)):
#         val = height[i]
#         if val < 5 and val > -5:
#             correctIndices.append(i)    
#     #print(indexH)
#     xFit = [height[i] for i in correctIndices]
#     yFit = [potential[ip] for ip in correctIndices]
    
#     #print(xFit,yFit)
#     # Model for the fit
#     def quadraticWell(x, A, B, C):
#         #gravity potential
#         return (A*x**2 + B*x + C)
    
#     gmodel = Model(quadraticWell)
#     result = gmodel.fit(yFit, x=xFit, A=0, B= 0, C = 0)

#     #print("For info, results from the minimum fit of the raw data")
#     #print(result.fit_report())
    

#     if checkoption == 1:
#         fig, ax = plt.subplots()
#         ax.plot(height, potential, 'bo')
#         ax.plot(xFit, result.init_fit, 'k--', label='minimum initial fit')
#         ax.plot(xFit, result.best_fit, 'r-', label='minimum best fit')
#         ax.legend(loc='best')
#         ax.set_xlim(min(xFit)*1.3,max(xFit)*1.3)
#         plt.show()
        
#     bestvalues = result.best_values
#     AV = bestvalues.get("A",0)
#     BV = bestvalues.get("B",0)
#     CV = bestvalues.get("C",0)
    
#     hmin = - BV/(2*AV)
#     #phimin = CV - BV**2/(4*AV)
    
#     potentialCentered = [potentialGravityRemoved[i] for i in range(len(potential))] # don't remove phimin here, this part was set up by gravity
#     heightCentered = [height[i] - hmin for i in range(len(potential))]
    
#     return heightCentered, potentialCentered


def centerPotentialRemoveGravityKnownExperiment(height,potential,checkoption,gravmin,gravmax,h_adjust,h_adjustment,hframe,gravity):

    # 0 - center potential 
    
    lowestindex = np.array(potential).argmin()
    lowestheight = height[lowestindex]
    height = height - lowestheight
    
    # 1 - remove gravity
    # fit the gravity profile only for values larger/smaller than gravmin/gravmax 
    correctIndices = []
    for i in range(len(height)):
        val = height[i]
        if val < gravmax and val > gravmin:
            correctIndices.append(i)
                
    #print(indexH)
    xFit = [height[i] for i in correctIndices]
    yFit = [potential[ip] for ip in correctIndices]
    
    # print(xFit)
    # print(yFit)
    # # Model for the fit
    # plt.plot(xFit,yFit)
    # plt.show()
    def gravityPotential(x, C):
        #gravity potential
        return (gravity*1e-9*x + C)
    
    gmodel = Model(gravityPotential)
    result = gmodel.fit(yFit, x=xFit, C = 0)

    #print("For info, results from the gravity fit of the raw data")
    #print(result.fit_report())
    
    if checkoption == 1:
        fig, ax = plt.subplots()
        ax.plot(height[correctIndices[0]-60:correctIndices[-1]], potential[correctIndices[0]-60:correctIndices[-1]], 'bo')
        ax.plot(xFit, result.init_fit, 'k--', label='gravity initial fit')
        ax.plot(xFit, result.best_fit, 'r-', lw=2 , label='gravity best fit')
        ax.legend(loc='best')
        plt.show()
        
    bestvalues = result.best_values
    GV = gravity*1e-9
    CV = bestvalues.get("C",0)
    
    potentialGravityRemoved = [potential[i] - GV*(height[i]+ h_adjustment) - CV for i in range(len(potential))]
    
    GV = GV*1e9


    potentialCentered = potentialGravityRemoved
    heightCentered = height + h_adjustment
    
    
    return heightCentered, potentialCentered, GV


def centerPotentialRemoveGravityKnown(height,potential,checkoption,gravmin,gravmax,h_adjust,h_adjustment,hframe,gravity):

    # 1 - remove gravity
    # fit the gravity profile only for values smaller than 120nm (beyond the results are floppy)
    # yet larger than 30 nm (hole)
    correctIndices = []
    for i in range(len(height)):
        val = height[i]
        if val < gravmax and val > gravmin:
            correctIndices.append(i)
                
    #print(indexH)
    xFit = [height[i] for i in correctIndices]
    yFit = [potential[ip] for ip in correctIndices]
    
    # Model for the fit
    # plt.plot(xFit,yFit)
    # plt.show()
    def gravityPotential(x, C):
        #gravity potential
        return (gravity*1e-9*x + C)
    
    gmodel = Model(gravityPotential)
    result = gmodel.fit(yFit, x=xFit, C = 0)

    #print("For info, results from the gravity fit of the raw data")
    #print(result.fit_report())
    
    if checkoption == 1:
        fig, ax = plt.subplots()
        ax.plot(height, potential, 'bo')
        ax.plot(xFit, result.init_fit, 'k--', label='gravity initial fit')
        ax.plot(xFit, result.best_fit, 'r-', label='gravity best fit')
        ax.legend(loc='best')
        plt.show()
        
    bestvalues = result.best_values
    GV = gravity*1e-9
    CV = bestvalues.get("C",0)
    
    potentialGravityRemoved = [potential[i] - GV*height[i] - CV for i in range(len(potential))]
    
    GV = GV*1e9
    
    # 2 - "find the real minimum"
    # fit the potential profile only between -5 and 5
    correctIndices = []
    hidmax = 0
    idh = 0
    
    
    potential = potentialGravityRemoved
    
    while hidmax == 0 and idh < len(height):
        #print(height[idh])
        if height[idh] > 40:
            hidmax = idh
        idh += 1 
    if hidmax == 0:
        hidmax = len(height)-1
        
    idmin = np.array(potential[0:hidmax]).argmin()
    #print(idmin)
    #print(idmin)
    #print(height[idmin])
    
    correctIndices = []
    
    idc = 0
    for h in height:
        if h - height[idmin] < hframe and h - height[idmin] > -hframe:
            correctIndices.append(idc)
            #print(h - height[idmin])
        idc+=1
    xFit = [height[ip] - height[idmin] for ip in correctIndices]
    yFit = [potential[ip] for ip in correctIndices]
    
    print(xFit,yFit)
    # Model for the fit
    def quadraticWell(x, A, B, C):
        #gravity potential
        return (A*x**2 + B*x + C)
    
    gmodel = Model(quadraticWell)

    if len(xFit) >= 3:
        result = gmodel.fit(yFit, x=xFit, A=0, B= 0, C = 0)
    
        #print("For info, results from the minimum fit of the raw data")
        #print(result.fit_report())
        
    
        if checkoption == 1:
            fig, ax = plt.subplots()
            ax.plot(xFit, yFit, 'bo')
            ax.plot(xFit, result.init_fit, 'k--', label='minimum initial fit')
            ax.plot(xFit, result.best_fit, 'r-', label='minimum best fit')
            ax.legend(loc='best')
            ax.set_xlim(min(xFit)*1.3,max(xFit)*1.3)
            ax.set_ylim(min(yFit)*1.3,max(yFit)*1.3)
            plt.show()
            
        bestvalues = result.best_values
        AV = bestvalues.get("A",0)
        BV = bestvalues.get("B",0)
        CV = bestvalues.get("C",0)
        
        hmin = - BV/(2*AV)
        #phimin = CV - BV**2/(4*AV)
        #print(- hmin*h_adjust + h_adjustment)
        potentialCentered = [potentialGravityRemoved[i] for i in range(len(potential))] # don't remove phimin here, this part was set up by gravity
        heightCentered = [height[i] - height[idmin] - hmin*h_adjust + h_adjustment for i in range(len(potential))]
    
    else:
        potentialCentered = [potentialGravityRemoved[i] for i in range(len(potential))] # don't remove phimin here, this part was set up by gravity
        heightCentered = [height[i] - height[idmin] + h_adjustment for i in range(len(potential))]
    
    
    
    return heightCentered, potentialCentered, GV