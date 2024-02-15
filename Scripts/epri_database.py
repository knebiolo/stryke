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
import matplotlib.pyplot as plt
from scipy.stats import pareto, lognorm, genextreme, ks_2samp, weibull_min
import os
from matplotlib import rcParams
import numpy as np


font = {'family': 'serif','size': 10}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'

# connect to data pass simple filter to EPRI class


fish = stryke.epri(Family = 'Catostomidae', Month = [1,2,3,4,5,6,7,8,9,10,11,12], HUC02= [5])
epri_dat=fish.epri
#epri_dat.to_csv(os.path.join(r"C:\Users\Srogers\Desktop\EpriOutput",'AllHUCSampleSizeIssue'))
fish.ParetoFit()
fish.LogNormalFit()
fish.WeibullMinFit()

# get a sample
pareto_sample = pareto.rvs(fish.dist_pareto[0],fish.dist_pareto[1],fish.dist_pareto[2],1000)
lognorm_sample = lognorm.rvs(fish.dist_lognorm[0],fish.dist_lognorm[1],fish.dist_lognorm[2],1000)
weibull_sample = weibull_min.rvs(fish.dist_weibull[0],fish.dist_weibull[1],fish.dist_weibull[2],1000)

# get our observations
observations = fish.epri.FishPerMft3.values

# KS test comnpare distribution with observations are they from the same distribution?
t1 = ks_2samp(observations,pareto_sample,alternative = 'two-sided')
t2 = ks_2samp(observations,lognorm_sample,alternative = 'two-sided')
t3 = ks_2samp(observations,weibull_sample,alternative = 'two-sided')

# make a figure
figSize = (4,4)
plt.figure()
fig, axs = plt.subplots(2,2,tight_layout = True,figsize = figSize)
axs[0,0].hist(observations, color='darkorange', density = True)
axs[0,0].set_title('Observations')
axs[0,0].set_xlabel('org per Mft3')
axs[0,1].hist(pareto_sample, color='blue',lw=2, density = True)
axs[0,1].set_title('Pareto p = %s'%(round(t1[1],4)))
axs[0,1].set_xlabel('org per Mft3')
axs[1,0].hist(lognorm_sample, color='blue',lw=2, density = True)
axs[1,0].set_title('Log Normal p = %s'%(round(t2[1],4)))
axs[1,0].set_xlabel('org per Mft3')
axs[1,1].hist(weibull_sample, color='darkorange',lw=2, density = True)
axs[1,1].set_title('Weibull p = %s'%(round(t3[1],4)))
axs[1,1].set_xlabel('org per Mft3')

#plt.savefig(os.path.join(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke\Output",'emerald_shiner.png'), dpi = 700)
plt.show()

# make a figure
figSize = (4,4)
plt.figure()
fig, axs = plt.subplots(2,2,tight_layout = True,figsize = figSize)
axs[0,0].hist(np.log(observations), color='darkorange', density = True)
axs[0,0].set_title('Observations')
axs[0,0].set_xlabel('org per Mft3')
axs[0,1].hist(np.log(pareto_sample), color='blue',lw=2, density = True)
axs[0,1].set_title('Pareto p = %s'%(round(t1[1],4)))
axs[0,1].set_xlabel('org per Mft3')
axs[1,0].hist(np.log(lognorm_sample), color='blue',lw=2, density = True)
axs[1,0].set_title('Log Normal p = %s'%(round(t2[1],4)))
axs[1,0].set_xlabel('org per Mft3')
axs[1,1].hist(np.log(weibull_sample), color='darkorange',lw=2, density = True)
axs[1,1].set_title('Weibull p = %s'%(round(t3[1],4)))
axs[1,1].set_xlabel('org per Mft3')

#plt.savefig(os.path.join(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke\Output",'emerald_shiner.png'), dpi = 700)
plt.show()

# # ok, now do lengths
#fish = stryke.epri(Species = 'Ictalurus punctatus')
fish.LengthSummary()

plt.figure()
plt.hist(fish.lengths,color = 'r')
plt.hist(lognorm.rvs(fish.len_dist[0],fish.len_dist[1],fish.len_dist[2],len(fish.lengths)),color = 'b', alpha = 0.5)
#plt.savefig(os.path.join(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke\Output",'fuck.png'), dpi = 700)
plt.show()