# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 21:00:15 2021

Script Intent: Work with EPRI entrainment database, filter and fit Pareto

@author: KNebiolo
"""
import sys
sys.path.append(r"C:\Users\KNebiolo\OneDrive - Kleinschmidt Associates, Inc\software\stryke\Stryke")
# import moduels
import stryke
import os
import numpy as np
import pandas as pd

stryke.enable_matplotlib_inline()

# connect to project directory
project = r"J:\868\022\Calcs\validation"

#%% Pass EPRI filter, fit distributions
fish = stryke.epri(Family = 'Salmonidae', Month = [1,2,3,4,5,6,7,8,9,10,11,12], HUC02= [7],NIDID= 'WI00816' )
epri_dat=fish.epri
#epri_dat.to_csv(os.path.join(r"C:\Users\Srogers\Desktop\EpriOutput",'AllHUCSampleSizeIssue'))
fish.ParetoFit()
fish.LogNormalFit()
fish.WeibullMinFit()
fish.plot()
fish.LengthSummary()

#%% Collect Data
# species data
family = fish.family
genus = fish.genus 
species = fish.species

# months
month = fish.month 

# HUC
huc = fish.HUC02

# presence and entrainment rate
presence = fish.presence 
max_ent_rate = fish.max_ent_rate 
sample_size = fish.sample_size

# weibull c, location, scale
weibull_p = fish.weibull_t
weibull_c = round(fish.dist_weibull[0],4)
weibull_loc = round(fish.dist_weibull[1],4)
weibull_scale = round(fish.dist_weibull[2],4)

# log normal b, location, scale
log_normal_p = fish.log_normal_t
log_normal_b = round(fish.dist_lognorm[0],4)
log_normal_loc = round(fish.dist_lognorm[1],4)
log_normal_scale = round(fish.dist_lognorm[2],4)

# pareto shape, location, scale
pareto_p = fish.pareto_t
pareto_b = round(fish.dist_pareto[0],4)
pareto_loc = round(fish.dist_pareto[1],4)
pareto_scale = round(fish.dist_pareto[2],4)

length_b = round(fish.len_dist[0],4)
length_loc = round(fish.len_dist[1],4)
length_scale = round(fish.len_dist[2],4)

#%% Run if results are good
row = np.array([family, genus, species, month, huc[0], presence, max_ent_rate,
               sample_size, weibull_p, weibull_c, weibull_loc, weibull_scale,
               log_normal_p, log_normal_b, log_normal_loc, log_normal_scale,
               pareto_p, pareto_b, pareto_loc, pareto_scale,
               length_b,length_loc,length_scale])
columns = ['family','genus','species','month','huc','presence','max_ent_rate',
           'sample_size','weibull_p','weibull_c','weibull_loc','weibull_scale',
           'log_normal_p,','log_normal_b','log_normal_loc','log_normal_scale',
           'pareto_p','pareto_b','pareto_loc','pareto_scale',
           'length_b','length_loc','length_scale']
new_row_df = pd.DataFrame([row],columns = columns)

try:
    results = pd.read_csv(os.path.join(project,'epri_fit.csv'))
except FileNotFoundError:
    results = pd.DataFrame(columns = columns)
    
results = pd.concat([results,new_row_df], ignore_index = True)
results.to_csv(os.path.join(project,'epri_fit.csv'), index = False)

