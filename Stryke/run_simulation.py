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
ws = r'D:\Franklin Falls\Data'
wks = 'Stryke Franklin Falls White Sucker.xlsx'

wks_dir = os.path.join(ws,wks)

simulation = stryke.simulation(ws,wks, output_name = 'ff white sucker', existing = False)

simulation.run()
simulation.summary(whole_project_surv = True)

results = simulation.beta_df
day_sum = simulation.summ_dat
year_sum = simulation.cum_sum
length = simulation.length_df

# summarize over iterations by Species and Flow Scenario

with pd.ExcelWriter(wks_dir,engine = 'openpyxl', mode = 'a') as writer:
    results.to_excel(writer,sheet_name = 'beta fit')
    day_sum.to_excel(writer,sheet_name = 'daily summary')    
    year_sum.to_excel(writer,sheet_name = 'yearly summary')
    length.to_excel(writer,sheet_name = 'length data')

day_sum.to_csv(os.path.join(ws,'francis_why_3_summary.csv'))

