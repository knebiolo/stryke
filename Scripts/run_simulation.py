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
ws = r'C:\Users\knebiolo\Desktop\Stryke\stryke\Spreadsheet Interface'
wks = 'Cabot_Beta_Test.xlsx'

wks_dir = os.path.join(ws,wks)

simulation = stryke.simulation(ws,wks, output_name = 'Cabot_Beta_Test')

simulation.run()
simulation.summary()


