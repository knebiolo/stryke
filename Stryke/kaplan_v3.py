# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:40:39 2020

@author: Kevin Nebiolo

Create an Upper Barker Model to test out Stryke
"""
# Import Dependencies
import stryke_v3
import os
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pareto


# read scenario worksheet
ws = r'E:\Rye_Entrainment_4287001\Entrainment\Data'
wks = 'Rye_Stryke_Allegheny.xlsx'

wks_dir = os.path.join(ws,wks)

simulation = stryke_v3.simulation(ws,wks, output_name = 'allegheny')

simulation.run()
simulation.summary()

results = simulation.beta_df
summary = simulation.summ_dat

with pd.ExcelWriter(wks_dir,engine = 'openpyxl', mode = 'a') as writer:
    results.to_excel(writer,sheet_name = 'beta fit')
    summary.to_excel(writer,sheet_name = 'summary')

# show ramp up at unit 1
# unit_1 = beta_fit_df[beta_fit_df.state == 'Unit 1']

# fig, ax = plt.subplots()
# ax.plot(unit_1.species.values,unit_1.est.values)
# ax.fill_between(unit_1.species.values, 
#                 unit_1.ll.values,
#                 unit_1.ul.values, 
#                 color='b', alpha=.1)
# plt.show()