#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:25:26 2021

@author: sm8857
"""

#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


#length0 = 5.0
def loadStretchedProfile(srcpath,length0):
    #df = pd.read_csv(srcpath+'FiniteStretching/'+'output_L5.0.txt', delimiter = "\t", skiprows = 2, header = None)
    df = np.loadtxt(srcpath+'FiniteStretching/'+'output_L{}.txt'.format(length0), skiprows = 3, usecols = (0,1,2))
    
    
    endTips = [df[i][2]*length0 for i in range(len(df))]
    zvalues = np.linspace(0,10/length0,1001)
    
    # plt.plot(zvalues, endTips)
    # plt.show()
    
    # print(df)
    
    return(zvalues,endTips)
