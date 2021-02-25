# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:40:39 2020

@author: Kevin Nebiolo

Create an Upper Barker Model to test out Stryke
"""
# Import Dependencies
import stryke_v2
import os
import numpy as np
import sqlite3
import pandas as pd
from scipy.stats import beta

# read scenario worksheet
wks_dir = r"\\kleinschmidtusa.com\Condor\Jobs\455\108\Docs\Studies\Entrainment-Blade Strike\STRYKE_StevensCreek_sensitivity_v4.xlsx"
scenarios = pd.read_excel(wks_dir,'Scenarios',header = 0,index_col = None)

all_results = pd.DataFrame()
beta_dict = {}
for row in scenarios.iterrows():
    # extract the number of fish, their length,  and stdv from row
    n = row[1]['Fish']
    length = row[1]['Length']/12.
    stdv = row[1]['StDev']/12.
    species = row[1]['Species']
    flow_scen = row[1]['Flow']
    scen_num = row[1]['Scenario Number']

    # build list of movement probabilities
    p_list = [row[1]['Percent Bypass'],row[1]['Percent Spill'],row[1]['Percent Unit']]

    # built a parameter dictionary for the kaplan function
    param_dict = {'H':float(row[1]['H']),
                  'RPM':float(row[1]['RPM']),
                  'D':float(row[1]['D']),
                  'Q':float(row[1]['Q']),
                  'Q_per':float(row[1]['Q_per']),
                  'ada':float(row[1]['ada']),
                  'N':float(row[1]['N']),
                  'iota':float(row[1]['iota']),
                  'D1':float(row[1]['D1']),
                  'D2':float(row[1]['D2']),
                  'B':float(row[1]['B']),
                  '_lambda':float(row[1]['_lambda'])} # use USFWS value of 0.2


    # build a survival dictionary
    surv_dict = {'bypass':float(row[1]['Bypass Survival']),
                 'spill':float(row[1]['Spill Survival'])}

    scen_results = pd.DataFrame()

    # create an iterator
    for i in np.arange(0,row[1]['Iterations'],1):
        # create population of fish
        population = np.random.normal(length,stdv,n)

        # simulate choice of route
        route = np.random.choice(['bypass','spill','Francis'],n,p = p_list)

        # simulate survival draws
        draw = np.random.uniform(0.,1.,n)

        # vectorize STRYKE survival function
        v_surv_rate = np.vectorize(stryke_v2.node_surv_rate)
        rates = v_surv_rate(population,route,surv_dict,param_dict)

        # calculate survival
        survival = np.where(draw > rates,0,1)

        # build dataframe of this iterations results
        iteration = pd.DataFrame({'scenario_num':np.repeat(scen_num,n),
                                  'species':np.repeat(species,n),
                                  'flow_scenario':np.repeat(flow_scen,n),
                                  'iteration':np.repeat(i,n),
                                  'population':population,
                                  'route':route,
                                  'draw':draw,
                                  'rates':rates,
                                  'survival':survival,})
        scen_results = scen_results.append(iteration)
        all_results = all_results.append(iteration)

    # write scenario results to spreadsheet
    with pd.ExcelWriter(wks_dir,engine = 'openpyxl', mode = 'a') as writer:
        scen_results.to_excel(writer,sheet_name = '%s %s'%(species,flow_scen))

    # summarize scenario - whole project
    whole_proj_succ = scen_results.groupby(by = 'iteration').survival.sum().to_frame().reset_index(drop = False).rename(columns = {'survival':'successes'})
    whole_proj_count = scen_results.groupby(by = 'iteration').survival.count().to_frame().reset_index(drop = False).rename(columns = {'survival':'count'})

    # merge successes and counts
    whole_summ = whole_proj_succ.merge(whole_proj_count)

    # calculate probabilities, fit to beta, write to dictionary summarizing results
    whole_summ['prob'] = whole_summ['successes']/whole_summ['count']
    whole_params = beta.fit(whole_summ.prob.values)
    whole_median = beta.median(whole_params[0],whole_params[1],whole_params[2],whole_params[3])
    whole_std = beta.std(whole_params[0],whole_params[1],whole_params[2],whole_params[3])
    whole_95ci = beta.interval(alpha = 0.95,a = whole_params[0],b = whole_params[1],loc = whole_params[2],scale = whole_params[3])
    beta_dict['%s_%s'%('whole',scen_num)] = [whole_median,whole_std,whole_95ci[0],whole_95ci[1]]

    # summarize scenario - whole project
    route_succ = scen_results.groupby(by = ['iteration','route']).survival.sum().to_frame().reset_index(drop = False).rename(columns = {'survival':'successes'})
    route_count = scen_results.groupby(by = ['iteration','route']).survival.count().to_frame().reset_index(drop = False).rename(columns = {'survival':'count'})
    # merge successes and counts
    route_summ = route_succ.merge(route_count)
    # calculate probabilities
    route_summ['prob'] = route_summ['successes']/route_summ['count']

    # extract route specific dataframes and fit beta
    # bypass
    bypass = route_summ[route_summ.route == 'bypass']
    if len(bypass) > 0:
        bypass_params = beta.fit(bypass.prob.values)
        bypass_median = beta.median(bypass_params[0],bypass_params[1],bypass_params[2],bypass_params[3])
        bypass_std = beta.std(bypass_params[0],bypass_params[1],bypass_params[2],bypass_params[3])
        bypass_95ci = beta.interval(alpha = 0.95,a = bypass_params[0],b = bypass_params[1],loc = bypass_params[2],scale = bypass_params[3])
        beta_dict['%s_%s'%('bypass',scen_num)] = [bypass_median,bypass_std,bypass_95ci[0],bypass_95ci[1]]
    # spill
    spill = route_summ[route_summ.route == 'spill']
    if len(spill) > 0:
        spill_params = beta.fit(spill.prob.values)
        spill_median = beta.median(spill_params[0],spill_params[1],spill_params[2],spill_params[3])
        spill_std = beta.std(spill_params[0],spill_params[1],spill_params[2],spill_params[3])
        spill_95ci = beta.interval(alpha = 0.95,a = spill_params[0],b = spill_params[1],loc = spill_params[2],scale = spill_params[3])
        beta_dict['%s_%s'%('spill',scen_num)] = [spill_median,spill_std,spill_95ci[0],spill_95ci[1]]
    # unit
    unit = route_summ[route_summ.route == 'Francis']
    if len(unit) > 0:
        unit_params = beta.fit(unit.prob.values)
        unit_median = beta.median(unit_params[0],unit_params[1],unit_params[2],unit_params[3])
        unit_std = beta.std(unit_params[0],unit_params[1],unit_params[2],unit_params[3])
        unit_95ci = beta.interval(alpha = 0.95,a = unit_params[0],b = unit_params[1],loc = unit_params[2],scale = unit_params[3])
        beta_dict['%s_%s'%('unit',scen_num)] = [unit_median,unit_std,unit_95ci[0],unit_95ci[1]]

    print ("Completed Scenario %s %s"%(species,flow_scen))

# convert beta dict to dataframe
beta_fit_df = pd.DataFrame.from_dict(beta_dict,orient = 'index',columns = ['mean','std','ll','ul'])
with pd.ExcelWriter(wks_dir,engine = 'openpyxl', mode = 'a') as writer:
    beta_fit_df.to_excel(writer,sheet_name = 'summary')
writer.close()

