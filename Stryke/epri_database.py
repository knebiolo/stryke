# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 21:00:15 2021

Script Intent: Work with EPRI entrainment database, filter and fit Pareto

@author: KNebiolo
"""

# import moduels
import stryke_v3 as stryke
import matplotlib.pyplot as plt
from scipy.stats import pareto, lognorm, genpareto, ks_2samp


# connect to data pass simple filter to EPRI class
fish = stryke.epri(Feeding_Guild = 'IC', Habitat = 'Pool', Month = [6,7,8])
fish.ParetoFit()

# get a sample
pareto_sample = pareto.rvs(fish.ent_dist[0],fish.ent_dist[1],fish.ent_dist[2],len(fish.epri))
plt.hist(pareto_sample,color = 'r')

genpareto_sample = genpareto.rvs(fish.ent_dist3[0],fish.ent_dist3[1],fish.ent_dist3[2],len(fish.epri))
plt.hist(genpareto_sample,color = 'g')

# plot the original data for comparison

observations = fish.epri.FishPerMft3.values
plt.hist(observations,color = 'b')

# KS test
t1 = ks_2samp(observations,pareto_sample,alternative = 'two-sided')
print (t1)
t2 = ks_2samp(observations,genpareto_sample,alternative = 'two-sided')
print (t2)

# # ok, now do lengths
#fish = stryke.epri(Species = 'Ictalurus punctatus')
fish.LengthSummary()

plt.hist(fish.lengths,color = 'r')
plt.hist(lognorm.rvs(fish.len_dist[0],fish.len_dist[1],fish.len_dist[2],len(fish.lengths)),color = 'b')
