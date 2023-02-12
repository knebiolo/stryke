# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 10:10:28 2023

@author: KNebiolo

Script Intent: Compare entrainment rates generated from pumped storage facilities, 
once through cooling facilities, of different species, and with different technologies
to strengthen confidence in use of Extreme Value distribtutions to simulate 
impingement and entrainment impacts for an ecological risk assessment. 
"""

# import modules
import pandas as pd
import numpy as np
from scipy.stats import pareto, genextreme, genpareto, lognorm, weibull_min
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# declare workspaces
inputWS = r"C:\Users\knebiolo\Desktop\stryke\Data"
outputWS = r"C:\Users\knebiolo\Desktop\stryke\Writings"

# get data
alos = pd.read_csv(os.path.join(inputWS,"alos_egg_density.csv"))
camera = pd.read_csv(os.path.join(inputWS,"pump_storage_density.csv"))

# plot alos egg density 

# get eggs
alos = alos[alos.LifeStage == 'E']

# create a linespace
alos_x = np.linspace(0,alos.EggPerM3.max(),100)

# fit alos egg density to pareto, gen exteme, and weibull
alos_pareto = pareto.fit(alos.EggPerM3.values)
alos_genextreme = genextreme.fit(alos.EggPerM3.values)
alos_weibull = weibull_min.fit(alos.EggPerM3.values)

# simulate 1000 
alos_pareto_sim = pareto.rvs(alos_pareto[0], alos_pareto[1], alos_pareto[2], 1000)
alos_genex_sim = genextreme.rvs(alos_genextreme[0], alos_genextreme[1], alos_genextreme[2], 1000)
alos_weibull_sim = weibull_min.rvs(alos_weibull[0], alos_weibull[1], alos_weibull[2], 1000)


# make a figure
figSize = (4,4)
plt.figure()
fig, axs = plt.subplots(2,2,tight_layout = True,figsize = figSize)
axs[0,0].hist(alos.EggPerM3, color='darkorange', density = True)
axs[0,0].set_title('Alossa sapidissima Egg Density')
axs[0,0].set_xlabel('org per Mft3')
axs[0,1].hist(alos_pareto_sim, color='blue',lw=2, density = True)
axs[0,1].set_title('Pareto')
axs[0,1].set_xlabel('org per Mft3')
axs[1,0].hist(alos_genex_sim, color='blue',lw=2, density = True)
axs[1,0].set_title('Extreme')
axs[1,0].set_xlabel('org per Mft3')
axs[1,1].hist(alos_weibull_sim, color='darkorange',lw=2, density = True)
axs[1,1].set_title('Weibull')
axs[1,1].set_xlabel('org per Mft3')

plt.show()

# plot Carm counts

# get eggs
camera.dropna(inplace = True)

# fit alos egg density to pareto, gen exteme, and weibull
camera_pareto = pareto.fit(camera.FishPerMCF.values)
camera_genextreme = genextreme.fit(camera.FishPerMCF.values)
camera_weibull = weibull_min.fit(camera.FishPerMCF.values)

# simulate 1000 
camera_pareto_sim = pareto.rvs(camera_pareto[0], camera_pareto[1], camera_pareto[2], 1000)
camera_genex_sim = genextreme.rvs(camera_genextreme[0], camera_genextreme[1], camera_genextreme[2], 1000)
camera_weibull_sim = weibull_min.rvs(camera_weibull[0], camera_weibull[1], camera_weibull[2], 1000)


# make a figure
figSize = (4,4)
plt.figure()
fig, axs = plt.subplots(2,2,tight_layout = True,figsize = figSize)
axs[0,0].hist(camera.FishPerMCF, color='darkorange', density = True)
axs[0,0].set_title('Acoustic Imaging Camera FishPerMCF')
axs[0,0].set_xlabel('count')
axs[0,1].hist(alos_pareto_sim, color='blue',lw=2, density = True)
axs[0,1].set_title('Pareto')
axs[0,1].set_xlabel('count')
axs[1,0].hist(alos_genex_sim, color='blue',lw=2, density = True)
axs[1,0].set_title('Extreme')
axs[1,0].set_xlabel('count')
axs[1,1].hist(alos_weibull_sim, color='darkorange',lw=2, density = True)
axs[1,1].set_title('Weibull')
axs[1,1].set_xlabel('count')

plt.show()