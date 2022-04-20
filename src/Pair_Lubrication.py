'''
Class to handle Lubrication corrections
'''
import numpy as np
from scipy import interpolate
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
from functools import partial
import copy
import inspect
import time
import sys
import scipy.sparse as sp

class Pair_Lubrication(object):
  '''
  Small class to handle a single body.
  '''  
  def __init__(self):
    '''
    Constructor. Take arguments like ...
    '''
    # Location as np.array.shape = 3
    self.mob_scalars_MB = None
    self.res_scalars_wall_MB_1 = None
    self.mob_scalars_wall_MB = None
    self.mob_scalars_WS = None
    self.mob_scalars_JO = None
    self.MB_Fn = None
    self.MB_wall_Fn = None
    self.MB_res_wall_Fn = None
    self.WS_Fn = None
  
  def load_WS_coefficient_interp_data(self,srcpath):
    self.mob_scalars_WS = np.loadtxt(srcpath+"/Resistance_Coefs/mob_scalars_WS.txt") #res_scalars_WS.txt
    print("loaded WS data")
    
  def load_JO_coefficient_interp_data(self,srcpath):
    self.mob_scalars_JO = np.loadtxt(srcpath+"/Resistance_Coefs/res_scalars_JO.txt")
    print("loaded JO data")
    
  def set_WS_coefficient_interp_functions(self, kind='linear'):
    mob_scalars_WS_11 = self.mob_scalars_WS[::2, :]
    mob_scalars_WS_12 = self.mob_scalars_WS[1::2, :]
    d_s = mob_scalars_WS_11[:, 0]
    x11a_WS = interpolate.interp1d(d_s, mob_scalars_WS_11[:,1], kind=kind)
    x12a_WS = interpolate.interp1d(d_s, mob_scalars_WS_12[:,1], kind=kind)
    y11a_WS = interpolate.interp1d(d_s, mob_scalars_WS_11[:,2], kind=kind)
    y12a_WS = interpolate.interp1d(d_s, mob_scalars_WS_12[:,2], kind=kind)
    y11b_WS = interpolate.interp1d(d_s, mob_scalars_WS_11[:,3], kind=kind)
    y12b_WS = interpolate.interp1d(d_s, mob_scalars_WS_12[:,3], kind=kind)
    x11c_WS = interpolate.interp1d(d_s, mob_scalars_WS_11[:,4], kind=kind)
    x12c_WS = interpolate.interp1d(d_s, mob_scalars_WS_12[:,4], kind=kind)
    y11c_WS = interpolate.interp1d(d_s, mob_scalars_WS_11[:,5], kind=kind)
    y12c_WS = interpolate.interp1d(d_s, mob_scalars_WS_12[:,5], kind=kind)
    self.WS_Fn = [x11a_WS, x12a_WS, y11a_WS, y12a_WS, y11b_WS, y12b_WS, x11c_WS, x12c_WS, y11c_WS, y12c_WS]

    
  def set_JO_coefficient_interp_functions(self, kind='linear'):
    mob_scalars_JO_11 = self.mob_scalars_JO[::2, :]
    mob_scalars_JO_12 = self.mob_scalars_JO[1::2, :]
    d_s = mob_scalars_JO_11[:, 0]
    x11a_JO = interpolate.interp1d(d_s, mob_scalars_JO_11[:,1], kind=kind)
    x12a_JO = interpolate.interp1d(d_s, mob_scalars_JO_12[:,1], kind=kind)
    y11a_JO = interpolate.interp1d(d_s, mob_scalars_JO_11[:,2], kind=kind)
    y12a_JO = interpolate.interp1d(d_s, mob_scalars_JO_12[:,2], kind=kind)
    y11b_JO = interpolate.interp1d(d_s, mob_scalars_JO_11[:,3], kind=kind)
    y12b_JO = interpolate.interp1d(d_s, mob_scalars_JO_12[:,3], kind=kind)
    x11c_JO = interpolate.interp1d(d_s, mob_scalars_JO_11[:,4], kind=kind)
    x12c_JO = interpolate.interp1d(d_s, mob_scalars_JO_12[:,4], kind=kind)
    y11c_JO = interpolate.interp1d(d_s, mob_scalars_JO_11[:,5], kind=kind)
    y12c_JO = interpolate.interp1d(d_s, mob_scalars_JO_12[:,5], kind=kind)
    self.JO_Fn = [x11a_JO, x12a_JO, y11a_JO, y12a_JO, y11b_JO, y12b_JO, x11c_JO, x12c_JO, y11c_JO, y12c_JO]

    
  def WS_Resist(self, r_norm):
    ''' 
    % WS_Resist computes the near-field lubrication resistance matrix for a pair of
    '''

    # compute the scalars. 
    X11A = self.WS_Fn[0](r_norm)
    X12A = self.WS_Fn[1](r_norm)
    Y11A = self.WS_Fn[2](r_norm)
    Y12A = self.WS_Fn[3](r_norm)
    Y11B = self.WS_Fn[4](r_norm)
    Y12B = self.WS_Fn[5](r_norm)
    X11C = self.WS_Fn[6](r_norm)
    X12C = self.WS_Fn[7](r_norm)
    Y11C = self.WS_Fn[8](r_norm)
    Y12C = self.WS_Fn[9](r_norm)
    
    
    Xa = np.array([[X11A, X12A], [X12A, X11A]])
    Ya = np.array([[Y11A, Y12A], [Y12A, Y11A]])
    Yb = np.array([[Y11B, -Y12B], [Y12B, -Y11B]])
    Xc = np.array([[X11C, X12C], [X12C, X11C]])
    Yc = np.array([[Y11C, Y12C], [Y12C, Y11C]])
    
    
    return Xa, Ya, Yb, Xc, Yc
  
  def JO_Resist_interp(self, r_norm):
    ''' 
    % WS_Resist computes the near-field lubrication resistance matrix for a pair of
    '''
    # compute the scalars. 
    X11A = self.JO_Fn[0](r_norm)
    X12A = self.JO_Fn[1](r_norm)
    Y11A = self.JO_Fn[2](r_norm)
    Y12A = self.JO_Fn[3](r_norm)
    Y11B = self.JO_Fn[4](r_norm)
    Y12B = self.JO_Fn[5](r_norm)
    X11C = self.JO_Fn[6](r_norm)
    X12C = self.JO_Fn[7](r_norm)
    Y11C = self.JO_Fn[8](r_norm)
    Y12C = self.JO_Fn[9](r_norm)
    
    
    Xa = np.array([[X11A, X12A], [X12A, X11A]])
    Ya = np.array([[Y11A, Y12A], [Y12A, Y11A]])
    Yb = np.array([[Y11B, -Y12B], [Y12B, -Y11B]])
    Xc = np.array([[X11C, X12C], [X12C, X11C]])
    Yc = np.array([[Y11C, Y12C], [Y12C, Y11C]])
    
    return Xa, Ya, Yb, Xc, Yc
    
  def AT_Resist(self, r_norm):
    ''' 
    % AT_Resist computes the near-field lubrication resistance matrix for a pair of
    % identical spheres from the scalars in Adam Townsend's paper and
    % Mathmatica code.
    '''
    epsilon = r_norm-2
    
    # compute the scalars. The fomular is from Adam Townsend's Mathmatica code.
    X11A = 0.995419E0+(0.25E0)*epsilon**(-1)+(0.225E0)*np.log(epsilon**(-1))+(0.267857E-1)*epsilon*np.log(epsilon**(-1))
    X12A = (-0.350153E0)+(-0.25E0)*epsilon**(-1)+(-0.225E0)*np.log(epsilon**(-1))+(-0.267857E-1)*epsilon*np.log(epsilon**(-1))
    Y11A = 0.998317E0+(0.166667E0)*np.log(epsilon**(-1))
    Y12A = (-0.273652E0)+(-0.166667E0)*np.log(epsilon**(-1))
    Y11B = (-0.666667E0)*(0.23892E0+(-0.25E0)*np.log(epsilon**(-1))+(-0.125E0)*epsilon*np.log(epsilon**(-1)))
    Y12B = (-0.666667E0)*((-0.162268E-2)+(0.25E0)*np.log(epsilon**(-1))+(0.125E0)*epsilon*np.log(epsilon**(-1)))
    X11C = 0.133333E1*(0.10518E1+(-0.125E0)*epsilon*np.log(epsilon**(-1)))
    X12C = 0.133333E1*((-0.150257E0)+(0.125E0)*epsilon*np.log(epsilon**(-1)))
    Y11C = 0.133333E1*(0.702834E0+(0.2E0)*np.log(epsilon**(-1))+(0.188E0)*epsilon*np.log(epsilon**(-1)))
    Y12C = 0.133333E1*((-0.27464E-1)+(0.5E-1)*np.log(epsilon**(-1))+(0.62E-1)*epsilon*np.log(epsilon**(-1)))
    
    Xa = np.array([[X11A, X12A], [X12A, X11A]])
    Ya = np.array([[Y11A, Y12A], [Y12A, Y11A]])
    Yb = np.array([[Y11B, -Y12B], [Y12B, -Y11B]])
    Xc = np.array([[X11C, X12C], [X12C, X11C]])
    Yc = np.array([[Y11C, Y12C], [Y12C, Y11C]])
    

    
    return Xa, Ya, Yb, Xc, Yc

  def Resist(self, r_i, r_j, eta, a, L=None):
    ''' 
    % AT_Resist computes the near-field lubrication resistance matrix for a pair of
    % identical spheres from the scalars in Adam Townsend's paper and
    % Mathmatica code.
    '''
    R_i = r_i
    R_j = r_j
    r = R_j-R_i
    #print r
    r = (1./a)*r
    r_norm = np.linalg.norm(r)
    #print r
    r_hat = r/r_norm
    print(r_hat)
    print(r_norm)
    #################################### BIG OL HACK
    #r_norm = max(r_norm,2.00011)
    ################################################
    
    AT_cutoff = (2+0.006-1e-8);
    WS_cutoff = (2+0.1+1e-8);
      

    if r_norm <= AT_cutoff:
      Xa, Ya, Yb, Xc, Yc = self.AT_Resist(r_norm)
      inv=False
    elif r_norm <= WS_cutoff:
      Xa, Ya, Yb, Xc, Yc = self.WS_Resist(r_norm)
      inv=True
    else:
      Xa, Ya, Yb, Xc, Yc = self.JO_Resist_interp(r_norm) 
      inv=False
    
    #print("this is the coefficient matrix you are interested in!!", Ya -- actually YA11)
    
    squeezeMat = np.outer(r_hat, r_hat)
    shearMat = np.eye(3) - squeezeMat
    vortMat = np.array([[0.0,    r_hat[2], -r_hat[1]],
                        [-r_hat[2], 0.0,    r_hat[0]],
                        [r_hat[1], -r_hat[0], 0.0]])
    
    if inv:
      A_factor = 1.0 / (6*np.pi*eta*a)
      B_factor = 1.0 / (6*np.pi*eta*a**2)
      C_factor = 1.0 / (6*np.pi*eta*a**3)
    else:
      A_factor = (6*np.pi*eta*a)
      B_factor = (6*np.pi*eta*a**2)
      C_factor = (6*np.pi*eta*a**3)
    
    A = A_factor*(np.kron(Xa, squeezeMat) + np.kron(Ya, shearMat))
    #print(A)
    #print(A_factor*np.kron(Xa, squeezeMat))
    #print(A_factor*np.kron(Ya, shearMat))
    #print(A_factor)
    B = B_factor*np.kron(Yb, vortMat)
    C = C_factor*(np.kron(Xc, squeezeMat) + np.kron(Yc, shearMat))
    
    P = np.kron(np.array([[1., 0., 0., 0.],[0., 0., 1., 0.],[0., 1., 0., 0.],[0., 0., 0., 1.]]), np.eye(3))
    ResistanceMix = np.block([[A, B],[B.T, C]])
    Resistance = np.dot(P,np.dot(ResistanceMix,P))
    
    if inv:
      Resistance = np.linalg.pinv(Resistance)
    
    return Resistance
