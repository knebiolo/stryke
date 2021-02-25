# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:59:19 2020

@author: Kevin Nebiolo
@qaqc: Isha Deo

Stryke: Kleinschmidt Associates Turbine Blade Strike Simulation Model

The intent of Stryke is to model downstream the passage mortality through a
theoretical hydroelectric facility.  The simulation will employ Monte Carlo
methods wihtin an individual based modeling framework.  Meaning we are
modeling the individual fates of a theoretical population of fish and
summarizing the results for a single simulation.  Then, we iterate that IBM
thousands of times and eventually, through black magic, we have a pretty good
estimate of what the overall downstream passage survival would be of a
theoretical population of fish through a theoretical hydroelectric facility.

For fish passing via entrainment, individuals are exposed to turbine strike,
which is modeled with the Franke et. al. 1997 equations.  For fish that pass
via passage structures or spill, mortality is assessed with a roll of the dice
using survival metrics determined a priori or sourced from similar studies.

Unfortunately units are in feet - wtf - why can't we get over ourselves and adopt
metric.  God damnit I hate us sometimes

"""
# import dependencies
import sqlite3
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import beta
import xlrd

# create the standard project database and directory structure
def create_proj_db(project_dir, dbName):
    ''' function creates empty project database, user can edit project parameters using
    DB Broswer for sqlite found at: http://sqlitebrowser.org/'''

    # first step creates a project directory if it doesn't already exist
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    data_dir = os.path.join(project_dir,'Data')                                # raw data goes here
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    output_dir = os.path.join(project_dir, 'Output')                           # intermediate data products, final data products and images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    scratch_dir = os.path.join(output_dir,'Scratch')
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    figures_dir = os.path.join(output_dir, 'Figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    dbDir = os.path.join(data_dir,dbName)

    # connect to and create the project geodatabase
    conn = sqlite3.connect(dbDir, timeout=30.0)
    c = conn.cursor()
    conn.commit()
    c.close()

def Kaplan(length, param_dict):
    '''Franke et al. TBS for Kaplan turbines.
    Inputs are length of fish and dictionary of turbine parameters'''

    # either sample parameters from statistical distributions, use a priori measures or extract from parameter dictionary
    g = 32.2
    H = param_dict['H']
    RPM = param_dict['RPM']
    D = param_dict['D']
    Q = param_dict['Q']
    #rR = np.array([0.75])
    rR = np.random.uniform(0.3,1.0,1) # where on the blade did the fish strike? - see Deng for more info #IPD: can you send me the reference for this? ~ discussed 2/5/20
    #Q_per = param_dict['Q_per']
    ada = param_dict['ada']
    N = param_dict['N']
    #Qopt = param_dict['Qopt']
    _lambda = param_dict['_lambda'] # use USFWS value of 0.2 #IPD: updated to mimic spreadsheet for upper barker

    # part 1 - calculate omega, the energy coefficient, discharge coefficient
    omega = RPM * ((2 * np.pi)/60)
    Ewd = (g * H) / (omega * D)**2
    Qwd = Q/ (omega * D**3)

    # part 2 - calculate angle of absolute flow to the axis of rotation
    a_a = np.arctan((np.pi * ada * Ewd)/(2 * Qwd * rR)) #IPD: np.arctan returns answer in radians

    # probability of strike * length of fish
    p_strike = _lambda * (N / (D)) * (np.cos(a_a)/(8 * Qwd) + np.sin(a_a)/(np.pi * rR)) # IPD: conversion to radians is redundant and incorrect ~ corrected 2/5/20
    # need to take cosine and sine of angle alpha a (a_a)

    return 1 - (p_strike * length)

def Propeller(length, param_dict):
    '''Franke et al. TBS for Kaplan turbines.
    Inputs are length of fish and dictionary of turbine parameters'''

    # either sample parameters from statistical distributions, use a priori measures or extract from parameter dictionary
    g = 32.2
    H = param_dict['H']
    RPM = param_dict['RPM']
    D = param_dict['D']
    Q = param_dict['Q']
    rR = np.random.uniform(0.3,1.0,1) # where on the blade did the fish strike? - see Deng for more info
    Q_per = param_dict['Q_per']
    ada = param_dict['ada']
    N = param_dict['N']
    Qopt = param_dict['Qopt'] #IPD: why not use Qopt for beta calculations?
    _lambda = param_dict['_lambda'] # use USFWS value of 0.2

    # part 1 - calculate omega, the energy coefficient, discharge coefficient
    omega = RPM * ((2 * np.pi)/60) # radians per second
    Ewd = (g * H) / (omega * D)**2
    Qwd = Q/ (omega * D**3)

    # part 2 - calculate angle of absolute flow to the axis of rotation
    beta = np.arctan((np.pi/8 * rR)/(Qwd * Q_per)) #IPD: what does Qper refer to? optimimum multiplier? ~ corrected 2/5/20
    a_a = np.arctan((np.pi/2 * Ewd * ada)/(Qwd * rR) + (np.pi/8 * rR)/Qwd - np.tan(beta)) #IPD: should be tan(beta) ~ corrected 2/5/20

    # probability of strike * length of fish
    p_strike = _lambda * (N / (D)) * (np.cos(a_a)/(8 * Qwd)) + np.sin(a_a)/(np.pi * rR)

    return 1 - (p_strike * length)

def Francis(length, param_dict):
    '''Franke et al. TBS for Francis Turbines.
    Inputs are length of fish and dictionary of turbine parameters'''

    # either sample parameters from statistical distributions, use a priori measures or extract from parameter dictionary
    g = 32.2
    H = param_dict['H']
    RPM = param_dict['RPM']
    D = param_dict['D']
    Q = param_dict['Q']
    Q_per = param_dict['Q_per']
    ada = param_dict['ada']
    N = param_dict['N']
    iota = param_dict['iota']
    D1 = param_dict['D1']
    D2 = param_dict['D2']
    B = param_dict['B']
    _lambda = param_dict['_lambda'] # use USFWS value of 0.2

    # part 1 - calculate omega, the energy coefficient, discharge coefficient
    omega = RPM * ((2 * np.pi)/60) # radians per second
    Ewd = (g * H) / (omega * D)**2
    Qwd = Q/ (omega * D**3)

    # part 2 - calculate alpha and beta
    beta = np.arctan((0.707 * np.pi/8)/(iota * Qwd * Q_per * np.power(D1/D2,3))) #IPD: what is Qper? relook @ this equation ~ discussed 2/5/20
    alpha = np.radians(90) - np.arctan((2 * np.pi * Ewd * ada)/Qwd * (B/D1) + (np.pi * 0.707**2)/(2 * Qwd) * (B/D1) * (np.power(D2/D1,2)) - 4 * 0.707 * np.tan(beta) * (B/D1) * (D1/D2)) #IPD: should be tan(beta) ~ corrected 2/5/20

    # probability of strike * length of fish
    p_strike = _lambda * (N / D) * (((np.sin(alpha) * (B/D1))/(2*Qwd)) + (np.cos(alpha)/np.pi))

    return 1 - (p_strike * length)

def Pump(length, param_dict):
    ''' pump mode calculations from fish entrainment analysis report:
        J:\1210\005\Docs\Entrainment\Entrainment Calcs\BladeStrike_CabotStation.xlsx'''

    # either sample parameters from statistical distributions, use a priori measures or extract from parameter dictionary
    g = 32.2
    H = param_dict['H']
    RPM = param_dict['RPM']
    D = param_dict['D']
    Q = param_dict['Q']
    Q_p = param_dict['Q_p']
    ada = param_dict['ada']
    N = param_dict['N']
    D1 = param_dict['D1']
    D2 = param_dict['D2']
    B = param_dict['B']
    _lambda = param_dict['_lambda'] # use USFWS value of 0.2
    gamma = param_dict['gamma'] # use value of 0.153 from Chief Joseph Kokanee project

    # part 1 - calculate omega, discharge coefficient
    omega = RPM * ((2 * np.pi)/60) # radians per second
    Qpwd = Q_p/ (omega * D**3)

    # part 2 - calculate beta
    beta_p = np.arctan((0.707 * np.pi/8)/(Qpwd * np.power(D1/D2,3)))

    # probability of strike * length of fish
    p_strike = gamma * (N / (0.707 * D2)) * (((np.sin(beta_p) * (B/D1))/(2*Qpwd)) + (np.cos(beta_p)/np.pi))

    return 1 - (p_strike * length)

def node_surv_rate(length,route,surv_dict,param_dict):
    # if survival function is a priori
    if route == 'bypass':
        # get the a priori survival rate from the table in sqlite
        prob = surv_dict[route]

    elif route == 'spill':
        prob = surv_dict[route]

    # if survival is assessed at a Kaplan turbine:
    elif route == 'Kaplan':
        # calculate the probability of strike as a function of the length of the fish and turbine parameters
        prob = Kaplan(length, param_dict)

    # if survival is assessed at a Propeller turbine:
    elif route == 'Propeller':
        # calculate the probability of strike as a function of the length of the fish and turbine parameters
        prob = Propeller(length, param_dict)

    # if survival is assessed at a Francis turbine:
    elif route == 'Francis':
        # calculate the probability of strike as a function of the length of the fish and turbine parameters
        prob = Francis(length, param_dict)

    # if survival is assessed at a turbine in pump mode:
    elif route == 'Pump':
        # calculate the probability of strike as a function of the length of the fish and turbine parameters
        prob = Pump(length, param_dict)

    return prob



def summary(dbDir):
    '''create a function to summarize the Monte Carlo simulation.

    I believe we care about the lengths of the simulated fish and survival % by simulation

    The only input is the project database'''

    # first let's get the data we wish to describe
    conn = sqlite3.connect(dbDir, timeout=30.0)
    fish = pd.read_sql('SELECT * FROM tblFish', con = conn)
    survival = pd.read_sql('SELECT * FROM tblSurvive', con = conn)
    completion = pd.read_sql('SELECT * FROM tblCompletion', con = conn)
    # let's desribe fish lengths with a histogram
    # plt.figure(figsize = (6,3))
    # fig, ax = plt.subplots()
    # ax.hist(fish.length.values,10,density = 1)
    # ax.set_xlabel('Length (ft)')
    # plt.show()

    # let's describe survival by node
    grouped = survival[['simulation','location','prob_surv','status']].groupby(['simulation','location']).agg({'prob_surv':'count','status':'sum'}).reset_index().rename(columns = {'prob_surv':'n','status':'p'})
    grouped['proportion'] = grouped.p / grouped.n
    # now fit a beta distribution to each node
    locations = grouped.location.unique()
    beta_dict = {}

    for i in locations:
        dat = grouped.loc[grouped.location == i]
        params = beta.fit(dat.proportion.values)
        beta_median = beta.median(params[0],params[1],params[2],params[3])
        beta_std = beta.std(params[0],params[1],params[2],params[3])
        beta_95ci = beta.interval(alpha = 0.95,a = params[0],b = params[1],loc = params[2],scale = params[3])
        beta_dict[i] = [beta_median,beta_std,beta_95ci[0],beta_95ci[1]]

    # now calculate whole project survival
    whole = completion[['simulation','status','completion']].groupby(['simulation']).agg({'status':'sum','completion':'sum'}).reset_index().rename(columns = {'status':'p','completion':'n'})
    whole ['proportion'] = whole.p / whole.n
    params = beta.fit(whole.proportion.values)
    beta_median = beta.median(params[0],params[1],params[2],params[3])
    beta_std = beta.std(params[0],params[1],params[2],params[3])
    beta_95ci = beta.interval(alpha = 0.95,a = params[0],b = params[1],loc = params[2],scale = params[3])
    beta_dict['whole project'] = [beta_median,beta_std,beta_95ci[0],beta_95ci[1]]

    beta_fit_df = pd.DataFrame.from_dict(beta_dict,orient = 'index',columns = ['mean','std','ll','ul'])
    del i, params
    #print (beta_dict)
    print (beta_fit_df)
    # make a plot for each beta distribution - we like visuals!
    # plt.figure(figsize = (6,3))
    # fig,ax = plt.subplots()
    # for i in beta_dict.keys():
    #     params = beta_dict[i]
    #     x = np.linspace(beta.ppf(0.01, params[0][0], params[0][1]),beta.ppf(0.99, params[0][0], params[0][1]), 100)
    #     ax.plot(x,beta.pdf(x, params[0][0], params[0][1]),label = i)
    # fig.legend()

    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tblSummary(Scenario INTEGER,
                                                                   FishLengthMean REAL,
                                                                   FishLengthSD REAL,
                                                                   NumFish INTEGER,
                                                                   NumPassedSuccess INTEGER,
                                                                   PercentSurvival REAL
                                                                   )''')
    conn.commit()
    completion = pd.read_sql('SELECT * FROM tblCompletion',conn)
    beta_fit_df.to_sql("tblBetaFit", con = conn, if_exists = 'replace')

    for sim in completion['simulation'].unique():
        subset = completion.loc[completion['simulation'] == sim].sum()
        c.execute("INSERT INTO tblSummary VALUES(%d,%f,%f,%d,%d,%f);"%(sim,fish.length.mean(),fish.length.std(),len(fish),subset['status'],subset['status']/subset['completion']))
    conn.commit()
    c.close()






