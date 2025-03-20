# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:40:39 2020

@author: Kevin Nebiolo

Create an Upper Barker Model to test out Stryke
"""
# Import Dependencies
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke\Stryke")
from stryke import simulation
import os
import pandas as pd

# read scenario worksheet
ws = r'C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke\Spreadsheet Interface'
wks = 'Input_Spreadsheet_v250304.xlsx'

wks_dir = os.path.join(ws,wks)

sim = simulation(ws,'new_sheet_alpha', wks)
sim.run()
sim.summary()
sim.close()


