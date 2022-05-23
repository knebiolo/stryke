# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:37:53 2022

@author: KNebiolo
"""

import os
import pandas as pd
import stryke

# identify workspace
inputWS = r"J:\1508\028\Calcs\Entrainment\Data"

# get data
dat = pd.read_csv(os.path.join(inputWS,'fish_lengths.csv'))

results = {'species':[],
           'sustained_min':[],
           'burst_min':[],
           'sustained_med':[],
           'burst_med':[],
           'sustained_max':[],
           'burst_max':[]}

# for row in dataframe, calculate swim speed in burst and sustained, append to dictionary
for row in dat.iterrows():
    spc = row[1]['species']
    min_ft = row[1]['min_in']/12.0
    med_ft = row[1]['med_in']/12.0
    max_ft = row[1]['max_in']/12.0

    ar = row[1]['ar']

    results['species'].append(spc)
    
    results['sustained_min'].append(stryke.speed(min_ft,ar,0))
    results['burst_min'].append(stryke.speed(min_ft,ar,1))
    
    results['sustained_med'].append(stryke.speed(med_ft,ar,0))
    results['burst_med'].append(stryke.speed(med_ft,ar,1))
    
    results['sustained_max'].append(stryke.speed(max_ft,ar,0))
    results['burst_max'].append(stryke.speed(max_ft,ar,1))

results = pd.DataFrame.from_dict(results,orient = 'columns')
results.to_csv(os.path.join(inputWS,'calc_swim_speed2.csv'))
    