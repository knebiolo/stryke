# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:40:39 2020

@author: Kevin Nebiolo

Create an Upper Barker Model to test out Stryke
"""
# Import Dependencies
import stryke
import os
import pandas as pd

# read scenario worksheet
ws = r'C:\Users\knebiolo\Desktop\Francis Qc\stryke'
wks = 'Francis Testing v3.xlsx'

wks_dir = os.path.join(ws,wks)

simulation = stryke.simulation(ws,wks, output_name = 'francis_why_3', existing = False)

simulation.run()
simulation.summary()

results = simulation.beta_df
summary = simulation.summ_dat
length = simulation.length_df

with pd.ExcelWriter(wks_dir,engine = 'openpyxl', mode = 'a') as writer:
    results.to_excel(writer,sheet_name = 'beta fit')
    summary.to_excel(writer,sheet_name = 'summary')
    length.to_excel(writer,sheet_name = 'length data')

summary.to_csv(os.path.join(wks_dir,'francis_why_3_summary.csv'))
