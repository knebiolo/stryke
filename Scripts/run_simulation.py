# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:40:39 2020

@author: Kevin Nebiolo

Create an Upper Barker Model to test out Stryke
"""
# Import Dependencies
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke\Stryke")
import stryke
import os
import pandas as pd

# read scenario worksheet
ws = r'K:\Jobs\868\022\Calcs\validation'
wks = 'HUC02_Schaghticoke.xlsx'

wks_dir = os.path.join(ws,wks)

simulation = stryke.simulation(ws,wks, output_name = 'HUC02_Schaticoke')

simulation.run()
simulation.summary()


