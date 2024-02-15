# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:40:39 2020

@author: Kevin Nebiolo

Create an Upper Barker Model to test out Stryke
"""
# Import Dependencies
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke\stryke")
import stryke
import os
import pandas as pd

# read scenario worksheet
ws = r'C:\Users\knebiolo\Desktop\Beaver_Falls_Production\validation'
wks = 'townsend_validation.xlsx'

wks_dir = os.path.join(ws,wks)

simulation = stryke.simulation(ws,wks, output_name = 'townsend_validate')

simulation.run()
simulation.summary()

results = simulation.beta_df
day_sum = simulation.daily_summary
year_sum = simulation.cum_sum
length = simulation.length_summ

# summarize over iterations by Species and Flow Scenario

with pd.ExcelWriter(wks_dir,engine = 'openpyxl', mode = 'a') as writer:
    results.to_excel(writer,sheet_name = 'beta fit')
    day_sum.to_excel(writer,sheet_name = 'daily summary')    
    year_sum.to_excel(writer,sheet_name = 'yearly summary')
    length.to_excel(writer,sheet_name = 'length data')


