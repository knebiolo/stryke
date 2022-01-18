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
from matplotlib import pyplot as plt
from scipy.stats import pareto


# read scenario worksheet
ws = r'J:\4287\001\Calcs\Entrainment\Data'
wks = 'Rye_Stryke_Opekiska.xlsx'

wks_dir = os.path.join(ws,wks)

simulation = stryke_v3.simulation(ws,wks, output_name = 'opekiska')

simulation.run()
simulation.summary()

results = simulation.beta_df
summary = simulation.summ_dat

with pd.ExcelWriter(wks_dir,engine = 'openpyxl', mode = 'a') as writer:
    results.to_excel(writer,sheet_name = 'beta fit')
    summary.to_excel(writer,sheet_name = 'summary')