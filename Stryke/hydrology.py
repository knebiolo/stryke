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
from scipy import interpolate 


# create directories for data
dataWS = r"J:\1852\010\Calcs\Flow Duration Curves"

# connect to the data
NID_to_gage = os.path.join(dataWS,'Data', 'NID_to_gage.csv')

# initialize the hydrologic analysis
flow = stryke.hydrologic(NID_to_gage, os.path.join(dataWS,'Output'))
print("Initialized hydrologic functions")

# generate list of exceedances to iterate over
exceedence = np.linspace(0,100,11)

# season dictionary
# season_dict = {'January':[1],
#                'February':[2],
#                'March':[3],
#                'April':[4],
#                'May':[5],
#                'June':[6],
#                'July':[7],
#                'August':[8],
#                'September':[9],
#                'October':[10],
#                'November':[11],
#                'December':[12]}
#'Annual':[1,2,3,4,5,6,7,8,9,10,11,12],
season_dict = {'January':[1],
                'February':[2],
                'March':[3],
                'April':[4],
                'May':[5],
                'June':[6],
                'July':[7],
                'August':[8],
                'September':[9],
                'October':[10],
                'November':[11],
                'December':[12]}

# calculate seasonal flow exceedances 
flow.seasonal_exceedance(season_dict, exceedence, HUC = 19)

print ("Calculated seasonal exceedance values")

# loop over exceedance flows, seasons, and dams, interpolate exceedance
# write to output
scenarios = pd.DataFrame()

# generate lists to iterate over
dams = flow.dams
seasons = season_dict.keys()

for dam in dams:
    for season in seasons:
        for exc in exceedence:
            # fit flow data to curve and plot for season, dam, and exceedance
            flow.curve_fit(season,dam,exc)

            # make a figure to show your friends and family, and regulators too!
            fig = plt.figure(figsize = (6,6))
            ax = fig.add_subplot(111)
            ax.plot(flow.X,flow.Y,'bo',label = 'Near Gages')
            ax.plot(flow.DamX,flow.DamY,'ro',label = '%s, %s sq km, %s cfs'%(dam, round(flow.DamX,0),round(flow.DamY,0)))
            ax.legend()
            ax.set_xlabel('Drainage Area (sq km)')
            ax.set_ylabel('%s Percent Exceedance Flow'%(exc))
            ax.set_title("%s %s Percent Exceedance at Dam %s"%(season,exc,dam))
            plt.savefig(os.path.join(dataWS,'Output','%s_%s_%s.png'%(dam,season,exc)),bbox_inches = 'tight', dpi = 900)

            plt.plot()

            # write to output file
            row = np.array([dam, season, exc, flow.DamX, flow.DamY, "%s %s %s Flow"%(dam,season,exc)])
            newRow = pd.DataFrame(np.array([row]),columns = ['NIDID','Season','Exceedance','Drain_sqkm','cfs','Scenario Name'])
            scenarios = scenarios.append(newRow)
            
scenarios = scenarios.astype({'Drain_sqkm':np.float32,'cfs':np.float32,'Exceedance':np.float32},copy = False)
scenarios.to_csv(os.path.join(dataWS,'Output','scenarios.csv'))
print ("Exported Scenarios")



# for key in season_dict:
#     # get this seasons data
#     season = scenarios[scenarios.Season == key] 
#     season.dropna(inplace = True)
    
#     # fit bspline to curve
#     t, c, k = interpolate.splrep(season.Exceedance, season.cfs, s= 0, k = 2)
    
#     # create a spline function and write to dictionary
#     spline = interpolate.BSpline(t, c, k, extrapolate = False)
    
#     x = np.linspace(0,100,101)
#     y = spline(x)
    
#     # Create Figure
#     fig = plt.figure(figsize = (6,4),dpi = 300)
#     ax = plt.axes()
#     ax.plot(x,y)
#     #ax.plot(pd.to_numeric(scenarios.Exceedance),pd.to_numeric(scenarios.cfs),'ro')
#     ax.grid(True, ls = '--',color = 'k', lw = 0.25)
#     ax.set_title('%s Flow Duration Curve'%(key),
#                   fontdict = {'family':'serif',
#                               'size':9})
#     ax.set_xlabel('Percent of Time Flow Equaled or Exceeded',
#                   fontdict = {'family':'serif',
#                                 'size':8})
#     ax.set_xticks(np.linspace(0,100,6))
#     ax.set_xticklabels(labels = np.linspace(0,100,6),
#                         fontdict = {'family':'serif',
#                                     'size':6})
#     ax.set_ylabel('Discharge (cfs)',
#                   fontdict = {'family':'serif',
#                               'size':8})
#     #ax.set_yticks(np.round(np.linspace(scenarios.cfs.min(),scenarios.cfs.max(),6),0))
#     #ax.set_yticklabels(labels = np.round(np.linspace(scenarios.cfs.min(),scenarios.cfs.max(),6),0), fontdict = {'family':'serif','size':6})

#     ax.set_yticks(np.linspace(0,500,6))
#     ax.set_yticklabels(labels = np.linspace(0,500,6), fontdict = {'family':'serif','size':6})

#     plt.savefig(os.path.join(dataWS,'Output','%s.png'%(key)),bbox_inches = 'tight', dpi = 900)
#     plt.show()
#     print ("%s Figure Exported"%(key))
    
    
    
    
