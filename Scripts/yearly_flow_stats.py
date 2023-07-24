# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:50:56 2020

@author: Kevin Nebiolo

Script Intent: use hydrofunctions to acquire USGS gage data.  Calculate monthly
and annual statistics (min, quartiles, mean), flow exceedance values on annual
and monthly periods, and produce annual and monthly flow duration curves.
"""

# import modules
import hydrofunctions as hf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#%matplotlib inline

# declare output workspace
dataWS = r"J:\577\036\Calcs\entrainment\Output"
# declare the USGS Gage
gage = '03107500'
# prorate your flow
prorate = 1
# identify project name for plotting
project_name = "Data Science Is Fun"

# get gage data object from web
gage_dat = hf.NWIS(site=str(gage), service='dv', start_date='1900-01-01')

# extract dataframe
df = gage_dat.df()

# replace column names
for j in gage_dat.df().columns:
    if '00060' in j:
        if 'qualifiers' not in j:
            if ':00000' in j or ':00003' in j:
                df.rename(columns = {j:'DAvgFlow'},inplace = True)

# reset index
df.reset_index(inplace = True)

#extract what we need
df = df[['datetimeUTC','DAvgFlow']]

# apply prorate
df['DAvgFlow_prorate'] = df.DAvgFlow * prorate

# convert to datetime
df['datetimeUTC'] = pd.to_datetime(df.datetimeUTC)

# extract month
df['year'] = pd.DatetimeIndex(df['datetimeUTC']).year
df = df[df.year != 2022]

# export tables for further analysis
#df.to_csv(os.path.join(dataWS,'gage_%s.csv'%(gage)))

# summarize by year
ann = df.groupby('year')['DAvgFlow_prorate'].mean().to_frame().reset_index()
ann.rename(columns = {'DAvgFlow_prorate':'YearlyAvgFlow'},inplace = True)

# calculate yearly exceedance probability 
ann['AnnualExcProb'] = ann['YearlyAvgFlow'].rank(ascending = False, method = 'first',pct = True) * 100

exc_10 = df[df.year == 1996]
exc_50 = df[df.year == 1978]
exc_90 = df[df.year == 1999]

exc_10.to_csv(os.path.join(dataWS,'exc_10.csv'))
exc_50.to_csv(os.path.join(dataWS,'exc_50.csv'))
exc_90.to_csv(os.path.join(dataWS,'exc_90.csv'))

emp_rates = pd.read_csv(r"C:\Users\knebiolo\Desktop\Beaver_Falls_Production\validation\yearly_ent_rate.csv")
yellow_perch = emp_rates[emp_rates.Common == 'Yellow perch']

exc_50['Month'] = pd.DatetimeIndex(exc_50.datetimeUTC).month

fish_ext = pd.DataFrame(columns = ['Date','Flow','Fish'])
idx = 0
for row in exc_50.iterrows():
    
    date = row[1]['datetimeUTC']
    flow = row[1]['DAvgFlow_prorate']
    month = row[1]['Month']
    if month == 12 or month == 1 or month == 2:
        fish = flow * yellow_perch[yellow_perch.season == 'winter'].AvgOfFishPerMft3.values
    elif month == 3 or month == 4 or month == 5:
        fish = flow * yellow_perch[yellow_perch.season == 'spring'].AvgOfFishPerMft3.values
    elif month == 6 or month == 7 or month == 8:
        fish = flow * yellow_perch[yellow_perch.season == 'summer'].AvgOfFishPerMft3.values 
    else: 
        fish = flow * yellow_perch[yellow_perch.season == 'fall'].AvgOfFishPerMft3.values
    if len(fish) == 0:
        new_row = [date,flow,0]
    else:
        new_row = [date,flow,fish[0]]
    fish_ext.loc[idx] = new_row
    idx = idx+1
print ('expansion complete')    
