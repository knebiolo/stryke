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


# read scenario worksheet
ws = r'J:\1126\008\Calcs\Data'
wks = 'Cornell_QC_StrykeV3_v8.xlsx'

wks_dir = os.path.join(ws,wks)

beta_fit_df = stryke_v3.simulation(ws,wks, export_results = False)

with pd.ExcelWriter(wks_dir,engine = 'openpyxl', mode = 'a') as writer:
    beta_fit_df.to_excel(writer,sheet_name = 'summary')

# show ramp up at unit 1
unit_1 = beta_fit_df[beta_fit_df.state == 'Unit 1']

fig, ax = plt.subplots()
ax.plot(unit_1.species.values,unit_1.est.values)
ax.fill_between(unit_1.species.values, 
                unit_1.ll.values,
                unit_1.ul.values, 
                color='b', alpha=.1)
plt.show()