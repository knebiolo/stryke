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
fish = stryke.epri(Family = 'Catostomidae', Month = [1,2,12], HUC02= [4], NIDID= 'WI00757')
epri_dat=fish.epri
#epri_dat.to_csv(os.path.join(r"C:\Users\Srogers\Desktop\EpriOutput",'AllHUCSampleSizeIssue'))
fish.ParetoFit()
fish.LogNormalFit()
fish.WeibullMinFit()
fish.plot()
fish.LengthSummary()

fish.summary_output(project, dist = 'Log Normal')

