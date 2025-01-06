# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 21:04:18 2023

@author: KNebiolo

script intent: 
    The intent of this script is to use Sympy to solve a complex algebraic expression
    that will allow us to calculate tail beat frequency (Hz) by setting Lighthills 
    equation equal to drag.  
    
    Then we will import Webb 1975 Table 20 and examine relationships between:
    1) Amplitude (A) and length (L)
    2) Propulsive Wave (V) and Swimming Speed (U)
    3) Trailing Edge Span (B) and Length (L)
    
    Then we will create functions for Thrust and Frequency, and validate 
    the Sympy solution by recreating f values in Table 20
    
"""
# import modules
from sympy import *
import numpy as np
from scipy import interpolate
import pandas as pd
import os
import matplotlib.pyplot as plt

# make fancy math output
init_printing(use_unicode=True)

# declare workspaces
inputWS = r"J:\2819\005\Calcs\ABM\Data"
outputWS = r"J:\2819\005\Calcs\ABM\Output"

#%% Part 1: Solve For f

# delcare symbols used in expressions.  
D,ρ,S,C_D,U_f,U_w,π,B,f,A,U,θ,λ,m,W,w,V  = symbols ('D,ρ,S,C_D,U_f,U_w,π,B,f,A,U,θ,λ,m,W,w,V')

# Fully expand out the RHS of the equation, including original terms for m,w, and W (Lighthill 1970)
rhs = ((π*ρ*B**2)/4)*((f*A*π)/1.414)*((f*A*π)/1.414*(1-(U/V)))*U-(((π*ρ*B**2)/4)*((f*A*π)/1.414*(1-(U/V)))**2*U)/(2*cos(θ))

# Make the left hand side of the equation equal to drag.  
lhs = D

# declare domain for sympy, we are in Real numbers
R = Reals

# Solve for frequency algebraicly with sympy.solveset
print (solveset(Eq(lhs,simplify(rhs)),f,R))
#print (solveset(Eq(lhs,simplify(rhs)),U,R))

#%% Part 2: Fit functions to Webb 1975

# get data
dat = pd.read_csv(os.path.join(inputWS,'sockeye_Webb1975.csv'))
len_new = np.linspace(0,dat.Length.max(),100)
swim_new = np.linspace(0,dat.U.max(),100)

# fit univariate spline to amplitude data, interpolate, and plot
amplitude = interpolate.UnivariateSpline(dat.Length,dat.A,k = 2) 
amp_new = amplitude(len_new)
plt.plot(len_new,amp_new)
plt.plot(dat.Length,dat.A,'ro')
plt.show

# fit univariate spline to propulsive wave data, interpolate, and plot
wave = interpolate.UnivariateSpline(dat.U,dat.V,k = 1) 
wave_new = wave(swim_new)
plt.plot(swim_new,wave_new)
plt.plot(dat.U,dat.V,'ro')
plt.show

# fit univariate spline to trailing edge span data, interpolate, and plot
trail = interpolate.UnivariateSpline(dat.Length,dat.B,k = 1) 
trail_new = trail(len_new)
plt.plot(len_new,trail_new)
plt.plot(dat.Length,dat.B,'ro')
plt.show

#%% Part 3: Create Functions for Thrust and Frequency
# Develop a function for thrust
def thrust (U,L,f):
    '''Lighthill 1970 thrust equation. '''
    # density of freshwater assumed to be 1
    rho = 1.0 
    
    # theta that produces cos(theta) = 0.85
    theta = 32.
    
    # sockeye parameters (Webb 1975, Table 20)
    length_dat = np.array([5.,10.,15.,20.,25.,30.,40.,50.,60.])
    speed_dat = np.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.]) /100.
    amp_dat = np.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4]) / 100.
    wave_dat = np.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3])
    edge_dat = np.array([1.,2.,3.,4.,5.,6.,8.,10.,12.]) / 100.
    
    # fit univariate spline
    amplitude = interpolate.UnivariateSpline(length_dat,amp_dat,k = 2) 
    wave = interpolate.UnivariateSpline(speed_dat,wave_dat,k = 1) 
    trail = interpolate.UnivariateSpline(length_dat,edge_dat,k = 1) 
    
    # interpolate A, V, B
    A = amplitude(L)
    V = wave(U)
    B = trail(L) 
    
    # Calculate thrust
    m = (np.pi * rho * B**2)/4.
    W = (f * A * np.pi)/1.414
    w = W * (1 - U/V)
        
    thrust = m * W * w * U - (m * w**2 * U)/(2. * np.cos(np.radians(theta)))
    
    return (thrust)

# Develop a function for tailbeat frequency
def frequency (U,L,D):
    ''' Function for tailbeat frequency.  By setting Lighthill (1970) equations 
    equal to drag, we can solve for tailbeat frequency (Hz).  
    
    Density of water (rho) is assumed to be 1
    
    Input parameters for this function include:
        U = speed over ground (or swim speed?) (cm/s)
        _lambda = length of the propulsive wave
        L = length, converted to trailing edge span (cm) = 0.2L
        D = force of drag'''
        
    # density of freshwater assumed to be 1
    rho = 1.0 
    
    # theta that produces cos(theta) = 0.85
    theta = 32.
    
    # sockeye parameters (Webb 1975, Table 20)
    length_dat = np.array([5.,10.,15.,20.,25.,30.,40.,50.,60.])
    speed_dat = np.array([37.4,58.,75.1,90.1,104.,116.,140.,161.,181.]) / 100.
    amp_dat = np.array([1.06,2.01,3.,4.02,4.91,5.64,6.78,7.67,8.4]) / 100.
    wave_dat = np.array([53.4361,82.863,107.2632,131.7,148.125,166.278,199.5652,230.0044,258.3])
    edge_dat = np.array([1.,2.,3.,4.,5.,6.,8.,10.,12.]) / 100.
    
    # fit univariate spline
    amplitude = interpolate.UnivariateSpline(length_dat,amp_dat,k = 2) 
    wave = interpolate.UnivariateSpline(speed_dat,wave_dat,k = 1) 
    trail = interpolate.UnivariateSpline(length_dat,edge_dat,k = 1) 
    
    # interpolate A, V, B
    A = amplitude(L)
    V = wave(U)
    B = trail(L)    
    
    # now that we have all variables, solve for f
    sol1 = -1 * np.sqrt(D*V**2*np.cos(np.radians(theta))/(A**2*B**2*U*np.pi**3*rho*(U - V)*(-0.062518880701972*U - 0.125037761403944*V*np.cos(np.radians(theta)) + 0.062518880701972*V)))
    sol2 = np.sqrt(D*V**2*np.cos(np.radians(theta))/(A**2*B**2*U*np.pi**3*rho*(U - V)*(-0.062518880701972*U - 0.125037761403944*V*np.cos(np.radians(theta)) + 0.062518880701972*V)))
    
    return (sol1,sol2)

#%% Part 4: Validation
# calculate thrust
vthrust = np.vectorize(thrust)
dat['thrust_erg_per_s'] = vthrust(dat.U,dat.Length,dat.f)
dat['thrust_Nm'] = dat.thrust_erg_per_s / 10000000. 
dat['thrust_N'] = dat.thrust_Nm / (dat.Length / 100.)

# calculate frequency
vfrequency = np.vectorize(frequency)
dat['freq'] = vfrequency(dat.U,dat.Length,dat.thrust_erg_per_s)[1]




    
    
        
    
        
    