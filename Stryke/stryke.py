# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:59:19 2020

@author: Kevin Nebiolo
@qaqc: Ish Deo

Stryke: Kleinschmidt Associates Turbine Blade Strike Simulation Model

The intent of Stryke is to model downstream passage mortality as a function of 
route of passage using Monte Carlo methods.  For fish passing via entrainment, 
individuals are exposed to turbine strike, which is modeled with the Franke et. 
al. 1997 equations.  For fish that pass via passage structures or spill, mortality
is assessed with a roll of the dice using survival metrics determined a priori 
or sourced from similar studies.  
"""

# create functions for Kaplan, Propeller & Francis units

