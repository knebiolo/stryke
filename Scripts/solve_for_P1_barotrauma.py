# -*- coding: utf-8 -*-
"""
@author: KNebiolo

script intent: 
    The intent of this script is to use Sympy to solve for the inlet pressure at a turbine
    runner.  This pressure should represent the maximum pressure experienced by a 
    fish during turbine passage.  
    
    We will use Bernoullis  principle:
    P_1 + 1/2(ρ*v)_1^2 + (ρ*g*h)_1= P_2 + 1/2(ρ*v)_2^2 +(ρ*g*h)_2+ (ρ*g*h)_loss   
    
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
P_1,P_2,ρ,v_1,v_2,g,h_1,h_2,h_loss  = symbols('P1,P2,ρ,v_1,v_2,g,h_1,h_2,h_loss')

# RHS of the equation
rhs = P_2 + 1/2 * (ρ*v_2)**2 + (ρ*g*h_2) + (ρ*g*h_loss)

# LHS of the equation.  
lhs = P_1 + 1/2 * (ρ*v_1)**2 + (ρ*g*h_1)

# declare domain for sympy, we are in Real numbers
R = Reals

# Solve for frequency algebraicly with sympy.solveset
print (solveset(Eq(lhs,rhs),P_1,R))

