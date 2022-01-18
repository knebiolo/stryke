# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 18:32:39 2021

@author: KNebiolo

Script Intent: Perform hydrologic analysis by extracting information from the
last 10 years at the 100 closest gages.  Then calculate the 10, 50 and 90%
exceedance flows by season.  With a data frame of watershed size and exceedance
flow by season, fit a linear regression with exceedance discharge as a funtion
of watershed size.  Examine for linearity, if assumption holds we can use the
function generated to predict exceedance flow at the project.

The major inputs are to filter the National Inventory of Dams feature class to
those dams impacted by the project.
"""

# import moduels
import stryke
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# create directories for data
dataWS = r"E:\Rye_Entrainment_4287001\stryke"

# connect to the data
nid = os.path.join(dataWS,'Data', 'rye_dams.shp')
gages = os.path.join(dataWS,'Data', 'gages.shp')
NID_to_gage = os.path.join(dataWS,'Data', 'NID_Near_Gage_200.csv')

# initialize the hydrologic analysis
flow = stryke.hydrologic(nid, gages, NID_to_gage, os.path.join(dataWS,'Output'))
print("Initialized hydrologic functions")

# calculate seasonal flow exceedances
flow.seasonal_exceedance({'Winter':[12,1,2],
                          'Spring':[3,4,5],
                          'Summer':[6,7,8],
                          'Fall':[9,10,11]},
                         HUC = '05')

print ("Calculated seasonal exceedance values")

# loop over exceedance flows, seasons, and dams, interpolate exceedance
# write to output
scenarios = pd.DataFrame()

# generate lists to iterate over
exc_list = ['exc_10','exc_50','exc_90']
dams = flow.dams
seasons = ['Winter','Spring','Summer','Fall']

for dam in dams:
    for season in seasons:
        for exceedance in exc_list:
            # fit flow data to curve and plot for season, dam, and exceedance
            flow.curve_fit(season,dam,exceedance)

            # make a figure to show your friends and family, and regulators too!
            fig = plt.figure(figsize = (6,6))
            ax = fig.add_subplot(111)
            ax.plot(flow.X,flow.Y,'bo',label = 'Near Gages')
            ax.plot(flow.DamX,flow.DamY,'ro',label = '%s, %s sq km, %s cfs'%(dam, round(flow.DamX,0),round(flow.DamY,0)))
            ax.legend()
            ax.set_xlabel('Drainage Area (sq km)')
            ax.set_ylabel('%s Percent Exceedance Flow'%(exceedance.split("_")[1]))
            ax.set_title("%s %s Percent Exceedance at Dam %s"%(season,exceedance.split("_")[1],dam))
            plt.savefig(os.path.join(dataWS,'Output','%s_%s_%s.png'%(dam,season,exceedance)),bbox_inches = 'tight', dpi = 900)

            plt.plot()

            # write to output file
            row = np.array([dam, season, exceedance, flow.DamX, flow.DamY, "%s %s %s Flow"%(dam,season,exceedance)])
            newRow = pd.DataFrame(np.array([row]),columns = ['NIDID','Season','Exceedance','Drain_sqkm','cfs','Scenario Name'])
            scenarios = scenarios.append(newRow)

scenarios.to_csv(os.path.join(dataWS,'Output','scenarios.csv'))





