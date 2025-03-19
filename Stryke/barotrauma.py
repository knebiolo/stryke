# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:08:20 2025

@author: Andrew.Yoder
"""

import numpy as np
import scipy

def calc_v(q, a):
    """
    Estimates water velocities if not provided.
    
    inputs:
    q: discharge (m^3/s)
    a: cross sectional areas (m^2)
    
    outputs:
    v: velocity    
    """
    
    v = q/a
    
    return v

def calc_k_viscosity(d_viscosity, density):
    """
    Calculates kinetic viscosity from dynamic viscosity
    
    inputs:
    d_viscosity: dynamic viscosity
    density: density of water (kg/m^3)
    
    outputs:
    k_viscosity: kinetic viscosity
    """
    
    k_viscosity = d_viscosity/density
    
    return k_viscosity

def calc_friction(K, ps_d, v, k_v):
    """
    Approximates friction factor for Darcy-Weisbach. Uses Reynolds Number, assumes circular cross section.
    
    inputs:
    K: absolute roughness
    ps_d: penstock diameter
    v: flow velocity
    k_v: kinematic viscosity of the fluid  
    
    outputs:
    f: friction factor
    """
    
    # compute relative roughness from absolute roughness
    rel_r = K/ps_d
    
    # compute Reynolds number
    Re = v*ps_d/k_v
    
    # use an equation with similar accuracy Colebrook-White (eq 8.4 in Miller)
    f = 0.25 / (np.log((rel_r/(3.7*ps_d)) + (5.74/Re**0.9)))**2
    
    return f



def calc_h_loss(f, ps_l, ps_d, v):
    """
    Calculates total head loss due to friction from Darcy-Weisbach equation.
    
    inputs:
    f: friction factor
    ps_l: length of penstock
    ps_d: diameter of penstock
    v_head: velocity head at turbine inlet
    
    outputs:
    h_loss: total head loss due to friction
    
    """
    g = scipy.constants.g
    
    # calculate velocity head
    v_head = ((v*v)/2*g)
    
    # calculate head loss
    h_loss = f*(ps_l/ps_d)*v_head
    
    return h_loss

def calc_p_2(p_atm, density, h_D):
    """
    Estimates pressure for P2 as a function of atmospheric pressure and submergence. 
    
    inputs:
    p_atm: atmospheric pressure
    density: density of water (kg/m^3)
    g: acceleration due to gravity
    h_D: submergence depth of draft tube outlet
    
    outputs:
    p_2: pressure at downstream draft tube entrance
    """
    
    g = scipy.constants.g
    p_atm = scipy.constants.atm
    
    p_2 = p_atm + density*g*h_D
    
    return p_2

def calc_p_1(p_2, h_1, h_2, density, v_1, v_2, h_loss):
    """
    Calculates pressure for P1.
    
    inputs:
    p_2: pressure at downstream (draft tube entrance)
    h_1: elevation head at the upstream point
    h_2: elevation head at the downstream point
    density: density of water (kg/m^3)
    v_1: flow velocity at the upstream point
    v_2: flow velocity at the downstream point
    g: acceleration due to gravity
    h_loss: total hed losses
    
    outputs:
    p_1: pressure at the upstream (penstock exit)
    """
    
    g = scipy.constants.g
    
    p_1 = p_2 + 0.5*density*(v_1*v_1) - 0.5*density*(v_2*v_2) + density*g*(h_1-h_2) + density*g*h_loss
    
    return p_1

def barotrauma_surv_prob(p_ratio, beta_0, beta_1):
    """
    Calculates the barotrauma-related survival probablilty using a biological
    response model.
    
    inputs:
    p_ratio: the pressure ratio between p1 and p2.
    beta_0: specific coefficient for each endpoint determined by the logistic regression analysis
    beta_1: specific coefficient for each endpoint determined by the logistic regression analysis
        
    output:
    endpoint: the selected endpoint (i.e., injury, mortal injury, or immediate
                                     mortality)
    """
    
    endpoint = np.exp(beta_0 + beta_1 * p_ratio) / (1 + np.exp(beta_0 + beta_1 * p_ratio))
    
    return endpoint
    

if __name__ == "__main__":
    
    # calculate the pressure ratio  given a vector of depths where fish start
    # and facility information
    
    # inputs: 
    #     q: discharge (m^3/s)
    #     a: cross sectional areas (m^2)
    #     K: absolute roughness (mm)
    #     ps_d: penstock diameter (m)
    #     v: flow velocity (m/s)
    #     f_d: flow depth  (m)
    #     ps_l: penstock length (m)
    #     v_head: velocity head at turbine inlet (m/s)
    #     h_D: submergence depth of draft tube outlet (m)
    #     h_2: elevation head at the downstream point (m?)
        
    # calc velocities
    q = 1
    a = 0.5
    v_1 = calc_v(q, a)
    v_2 = calc_v(q, a)
    
    # calculate friction for total head loss
    K = 0.025 # absolute friction for new, smooth steel pipe
    ps_d = 0.8
    f_d = 2
    d_viscosity = 0.0010016 # @ 20C
    density = 998.2 # kg/m^3
    k_v = calc_k_viscosity(d_viscosity, density)
    f = calc_friction(K, ps_d, v_1, f_d, k_v)
    print(f"friction factor = {f:0.04f}")
    
    # calculate total head loss
    ps_l = 40
    v_head = 2
    h_loss = calc_h_loss(f, ps_l, ps_d, v_head)
    
    # calculate pressure at p2
    p_atm = scipy.constants.atm
    h_D = 5
    p_2 = calc_p_2(p_atm, density, h_D)
    
    # calculate pressure at p1
    h_2 = -10
    depths = np.array([0.5, 6, 3, 27, 8.498, -9.33])
    print(f"depths = {depths}")
    p_1 = calc_p_1(p_2, depths, h_2, density, v_1, v_2, h_loss)
    
    # pressure ratio
    p_ratio = p_1 / p_2
    print(f"pressure ratios = {p_ratio}")

    # calculate mortality endpoint P(X)
    beta_0 = -1.132
    beta_1 = 0.450
    endpoint = barotrauma_surv_prob(p_ratio, beta_0, beta_1)
    print(f"survival probability = {endpoint}")
