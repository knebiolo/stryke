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
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import beta
import xlrd
import networkx as nx
#import hydrofunctions as hf
#import geopandas as gp
import statsmodels.api as sm
import math
from scipy.stats import pareto, genextreme, genpareto, lognorm, weibull_min, gumbel_r
import h5py
#import tables
from numpy.random import default_rng
rng = default_rng()


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
    p_strike = _lambda * (N * length / D) * ((np.cos(a_a)/(8 * Qwd)) + np.sin(a_a)/(np.pi * rR)) # IPD: conversion to radians is redundant and incorrect ~ corrected 2/5/20
    # need to take cosine and sine of angle alpha a (a_a)

    return 1 - (p_strike)

def Propeller(length, param_dict):
    '''Franke et al. TBS for Kaplan turbines.
    Inputs are length of fish and dictionary of turbine parameters'''

    # either sample parameters from statistical distributions, use a priori measures or extract from parameter dictionary
    g = 32.2
    H = param_dict['H']
    RPM = param_dict['RPM']
    D = param_dict['D']
    Q = param_dict['Q']
    #rR = np.random.uniform(0.3,1.0,1) # where on the blade did the fish strike? - see Deng for more info
    rR = 0.75
    ada = param_dict['ada']
    N = param_dict['N']
    Qopt = param_dict['Qopt'] #IPD: why not use Qopt for beta calculations?
    Q_per = param_dict['Qper']
    _lambda = param_dict['_lambda'] # use USFWS value of 0.2

    # part 1 - calculate omega, the energy coefficient, discharge coefficient
    omega = RPM * ((2 * np.pi)/60) # radians per second
    Ewd = (g * H) / (omega * D)**2
    Qwd = Q/ (omega * D**3)
    Qwd_opt = Qopt/ (omega * D**3)

    # part 2 - calculate angle of absolute flow to the axis of rotation
    beta = np.arctan((np.pi/8 * rR)/Qwd_opt) #IPD: what does Qper refer to? optimimum multiplier? ~ corrected 2/5/20

    a_a = np.arctan((np.pi * Ewd * ada)/(2 * Qwd * rR) + (np.pi/8 * rR)/Qwd - np.tan(beta)) #IPD: should be tan(beta) ~ corrected 2/5/20

    # probability of strike * length of fish
    p_strike = _lambda * (N * length / D) * ((np.cos(a_a)/(8 * Qwd)) + np.sin(a_a)/(np.pi * rR))

    return 1 - (p_strike)

def Francis(length, param_dict):
    '''Franke et al. TBS for Francis Turbines.
    Inputs are length of fish and dictionary of turbine parameters'''

    # either sample parameters from statistical distributions, use a priori measures or extract from parameter dictionary
    g = 32.2
    H = param_dict['H']
    RPM = param_dict['RPM']
    D = param_dict['D']
    Q = param_dict['Q']
    Q_per = param_dict['Qper']
    #Q_opt = param_dict['Q_opt']
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
    beta = np.arctan((0.707 * (np.pi/8))/(iota * Qwd * np.power(D1/D2,3))) #IPD: what is Qper? relook @ this equation ~ discussed 2/5/20
    #beta = np.arctan((0.707 * (np.pi/8))/(iota * Qwd * Q_per * np.power(D1/D2,3))) #IPD: what is Qper? relook @ this equation ~ discussed 2/5/20

    #tan_beta = (0.707 * (np.pi/8))/(iota * Qwd * np.power(D1/D2,3)) #IPD: what is Qper? relook @ this equation ~ discussed 2/5/20
    
    #alpha = np.radians(90) - np.arctan((2 * np.pi * Ewd * ada)/Qwd * (B/D1) + (np.pi * 0.707**2)/(2 * Qwd) * (B/D1) * (np.power(D2/D1,2)) - 4 * 0.707 * np.tan(beta) * (B/D1) * (D1/D2)) #IPD: should be tan(beta) ~ corrected 2/5/20
    alpha = np.radians(90) + np.arctan((2 * np.pi * Ewd * ada)/Qwd * (B/D1) + (np.pi * 0.707**2)/(2 * Qwd) * (B/D1) * (np.power(D2/D1,2)) - 4 * 0.707 * np.tan(beta) * (B/D1) * (D1/D2)) #IPD: should be tan(beta) ~ corrected 2/5/20

    #alpha = np.radians(90) + np.arctan((2 * np.pi * Ewd * ada)/Qwd * (B/D1) + (np.pi * 0.707**2)/(2 * Qwd) * (B/D1) * (np.power(D2/D1,2)) - 4 * 0.707 * tan_beta * (B/D1) * (D1/D2)) #IPD: should be tan(beta) ~ corrected 2/5/20


    # probability of strike * length of fish
    p_strike = _lambda * (N * length/ D) * (((np.sin(alpha) * (B/D1))/(2*Qwd)) + (np.cos(alpha)/np.pi))

    return 1 - (p_strike)

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

def node_surv_rate(length,status,surv_fun,route,surv_dict,u_param_dict):

    if status == 0:
        return 0.0
    else:
        if surv_fun == 'a priori':
                prob = surv_dict[route]

        else:
            param_dict = u_param_dict[route]
            # if survival is assessed at a Kaplan turbine:
            if surv_fun == 'Kaplan':
                # calculate the probability of strike as a function of the length of the fish and turbine parameters
                prob = Kaplan(length, param_dict)

            # if survival is assessed at a Propeller turbine:
            elif surv_fun == 'Propeller':
                # calculate the probability of strike as a function of the length of the fish and turbine parameters
                prob = Propeller(length, param_dict)

            # if survival is assessed at a Francis turbine:
            elif surv_fun == 'Francis':
                # calculate the probability of strike as a function of the length of the fish and turbine parameters
                prob = Francis(length, param_dict)

            # if survival is assessed at a turbine in pump mode:
            elif surv_fun == 'Pump':
                # calculate the probability of strike as a function of the length of the fish and turbine parameters
                prob = Pump(length, param_dict)

        return np.float32(prob)

# create function that builds networkx graph object from nodes and edges in project database
def create_route(wks_dir):
    '''function creates a networkx graph object from information provided by
    nodes and edges found in standard project database.  the only input
    is the standard project database.'''

    nodes = pd.read_excel(wks_dir,'Nodes',header = 0,index_col = None, usecols = "B:D", skiprows = 9)
    edges = pd.read_excel(wks_dir,'Edges',header = 0,index_col = None, usecols = "B:D", skiprows = 8)

    # create empty route object
    route = nx.route = nx.DiGraph()

    # add nodes to route - nodes.loc.values
    route.add_nodes_from(nodes.Location.values)

    # create edges - iterate over edge rows to create edges
    weights = []
    for i in edges.iterrows():
        _from = i[1]['_from']
        _to = i[1]['_to']
        weight = i[1]['weight']
        route.add_edge(_from,_to,weight = weight)
        weights.append(weight)

    # return finished product and enjoy functionality of networkx
    return route

def movement (location, status, swim_speed, graph, intake_vel_dict, Q_dict, op_order):
    ''' 
    since fish follow the flow, we can determine the probability
    a fish will choose the powerhouse over spill if we knew the 
    the min flow requirement, the min operating discharge, the unit 
    capacity, and the current discharge.
    
    we need to apportion our movement probabilities by edge weights
    iterate through neighbors and assign them ranges 0 - 1.0 based on
    movement weights - (].
                        
    since we are only dealing with downstream oriented fish, if the 
    current location is equal to forebay, then we need to do this fancy
    logic.
    
    Discharge dictionary (Q_dict) contains the:  curr_Q =  current discharge,
                                                 min_Q = minimum operating flow,
                                                 env_Q = environmental flow,
                                                 sta_cap_Q = station capacity, 
                                                 then capacity by unit
                                                 
    Operation order dictionary (op_order), is a dictionary of unit (key) and 
    operating order (value), ex: {'U1':1,'U2':2}
    '''  
    curr_Q = Q_dict['curr_Q']   # current discharge
    min_Q = Q_dict['min_Q']     # minimum operating discharge
    sta_cap = Q_dict['sta_cap'] # station capacity
    env_Q = Q_dict['env_Q']     # min environmental discharge 
    
    # if the fish is alive
    if status == 1:
        # get neighbors
        neighbors = set(graph[location])
        neighbors = list(neighbors)
                      
        locs = []
        probs = []
        
        # if the location is the forebay, then we have to do a lot of stuff
        if location == 'forebay':
            # when current discharge less than the min operating flow - everything is spilled:
            if curr_Q <= min_Q:
                for i in neighbors:
                    if i[0] == 'U':
                        locs.append(i)
                        probs.append(0.)
                    else:
                        locs.append('spill')  
                        probs.append(round(1 - np.sum(probs),5))  

                    
            # When current discharge greater than the min operating flow but less than station capacity...
            elif min_Q < curr_Q <= sta_cap:
                # get flow remaining for production
                prod_Q = curr_Q - env_Q
                                
                for i in neighbors:
                    if i[0] == 'U':
                        unit_cap = Q_dict[i]
                        order = op_order[i]
                        
                        # list units that turn on before this one
                        prev_units = []
                        for u in op_order:
                            if op_order[u] < order:
                                prev_units.append(u)
                        
                        # calculate the amount of discharge going to this unit
                        if len(prev_units) == 0:
                            if prod_Q >= Q_dict[i]:
                                u_Q = Q_dict[i]
                            else:
                                u_Q = prod_Q
                        else:
                            # need to figure out how much discharge is going to other units
                            prev_Q = 0
                            for j in prev_units:
                                prev_Q = prev_Q + Q_dict[j]
                            
                            if prev_Q > prod_Q:
                                u_Q = 0.0
                            else:
                                u_Q = prod_Q - prev_Q
                        
                        # write data to arrays
                        locs.append(i)

                        probs.append(round(u_Q/curr_Q,5))
                        del u_Q, prev_units
                        
                for i in neighbors:
                    if i == 'spill':
                        locs.append('spill')  
                        probs.append(round(1-np.sum(probs),5))
                
            # When current discharge greater than the min operating flow AND station capacity...
            elif curr_Q > sta_cap:
                               
                for i in neighbors:
                    if i[0] == 'U':
                        locs.append(i)
                        probs.append(round(Q_dict[i]/curr_Q,5))
                for i in neighbors:
                    if i == 'spill':
                        locs.append('spill')  
                        probs.append(round(1 - np.sum(probs),5))          
                    
        # if the location isn't the forebay, it really only has 1 place to go
        else:
            locs.append(neighbors[0])
            probs.append(1)

        # generate a new location
        new_loc = np.random.choice(locs,1,p = probs)[0]
        del neighbors, locs, probs
        
        # filter out those fish that can escape intake velocity
        if np.sum(swim_speed) > 0:
            if 'U' in new_loc:
                if swim_speed > intake_vel_dict[new_loc]:
                    new_loc = 'spill'
   
    # if the fish is dead, it can't move
    else:
        new_loc = location

    return new_loc

def speed (L,A,M):
    '''vectorizable function to calculate swimming speed as a function of length.
    
    Sambilay 1990
    
    Inputs:
        A = caudal fin aspect ratio (fishbase y'all)
        L = fish length (cm)
        M = swimming mode where 0 = sustained and 1 = burst
        
    Output is given in kilometers per hour, conversion to feet per second 
    given by google.'''
        
    # log_sa = -0.828 + 0.6196 * np.log10(L*30.48) + 0.3478 * np.log10(A) + 0.7621 * M
    
    # return (10**log_sa) * 0.911344

    sa = 10**(-0.828 + 0.6196 * np.log10(L*30.48) + 0.3478 * np.log10(A) + 0.7621 * M)
    
    return sa * 0.911344

class simulation():
    ''' Python class object that initiates, runs, and holds data for a facility
    specific simulation'''
    def __init__ (self, proj_dir, wks, output_name, existing = False):
        if existing == False:
            # create workspace directory
            self.wks_dir = os.path.join(proj_dir,wks)
            
            # extract scenarios from input spreadsheet
            #self.routing = pd.read_excel(self.wks_dir,'Routing',header = 0,index_col = None, usecols = "B:G", skiprows = 9)

            # import nodes and create a survival function dictionary
            self.nodes = pd.read_excel(self.wks_dir,'Nodes',header = 0,index_col = None, usecols = "B:D", skiprows = 9)
            self.surv_fun_df = self.nodes[['Location','Surv_Fun']].set_index('Location')
            self.surv_fun_dict = self.surv_fun_df.to_dict('index')

            # get last river node
            max_river_node = 0
            for i in self.nodes.Location.values:
                i_split = i.split("_")
                if len(i_split) > 1:
                    river_node = int(i_split[2])
                    if river_node > max_river_node:
                        max_river_node = river_node

            # make a movement graph from input spreadsheet
            self.graph = create_route(self.wks_dir)
            print ("created a graph")

            # identify the number of moves that a fish can make
            path_list = nx.all_shortest_paths(self.graph,'river_node_0','river_node_%s'%(max_river_node))

            max_len = 0
            for i in path_list:
                path_len = len(i)
                if path_len > max_len:
                    max_len = path_len
            self.moves = np.arange(0,max_len-1,1)
            print ("identified the number of moves, %s"%(self.moves))

            # import unit parameters
            self.unit_params = pd.read_excel(self.wks_dir,'Unit Params', header = 0, index_col = None, usecols = "B:R", skiprows = 4)

            # get hydraulic capacity of facility
            self.flow_cap = self.unit_params.Qcap.sum()

            # identify unique flow scenarios
            self.scenarios_df = pd.read_excel(self.wks_dir,'Flow Scenarios',header = 0,index_col = None, usecols = "B:L", skiprows = 5, dtype = {'Gage':np.str})
            self.scenarios = self.scenarios_df['Scenario'].unique()

            # import population data
            self.pop = pd.read_excel(self.wks_dir,'Population',header = 0,index_col = None, usecols = "B:S", skiprows = 11)

            # create output HDF file
            self.proj_dir = proj_dir
            self.output_name = output_name

            # create hdf object with Pandas
            self.hdf = pd.HDFStore(os.path.join(self.proj_dir,'%s.h5'%(self.output_name)))

            # write study set up data to hdf store
            self.hdf['Scenarios'] = self.scenarios_df
            self.hdf['Population'] = self.pop
            self.hdf['Nodes'] = self.nodes
            self.hdf['Edges'] = pd.read_excel(self.wks_dir,'Edges',header = 0,index_col = None, usecols = "B:D", skiprows = 8)
            self.hdf['Unit_Parameters'] = self.unit_params
            #self.hdf['Routing'] = self.routing
            self.hdf.flush()
        else:
            self.wks_dir = os.path.join(proj_dir,wks)

            # create output HDF file
            self.proj_dir = proj_dir
            self.output_name = output_name

            # import nodes and create a survival function dictionary
            self.nodes = pd.read_excel(self.wks_dir,'Nodes',header = 0,index_col = None, usecols = "B:D", skiprows = 9)

            # get last river node
            max_river_node = 0
            for i in self.nodes.Location.values:
                i_split = i.split("_")
                if len(i_split) > 1:
                    river_node = int(i_split[2])
                    if river_node > max_river_node:
                        max_river_node = river_node

            # make a movement graph from input spreadsheet
            self.graph = create_route(self.wks_dir)

            # identify the number of moves that a fish can make
            path_list = nx.all_shortest_paths(self.graph,'river_node_0','river_node_%s'%(max_river_node))

            max_len = 0
            for i in path_list:
                path_len = len(i)
                if path_len > max_len:
                    max_len = path_len
            self.moves = np.arange(0,max_len-1,1)

    def run(self):
        str_size = dict()
        str_size['species'] = 30
        for i in self.moves:
            str_size['state_%s'%(i)] = 30

        for scen in self.scenarios:
            scen_num = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Scenario Number'].values[0]
            season = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Season'].values[0]
            scenario = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Scenario'].values[0]
            months = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Months'].values[0]
            min_Q = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Min_Op_Flow'].values[0]
            env_Q = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Env_Flow'].values[0]
            
            if type(months) != np.int64:
                month_list = months.split(",") 
                month_list = list(map(int, month_list))
                months = len(months.split(","))
                
            else:
                months = 1

            hours = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Hours'].values[0]

            # identify the species we need to simulate for this scenario
            species = self.pop[self.pop['Season'] == season].Species.unique()
            
            ''' get flow data for simulation here.  Flow can either be a single value to simulate
            a single migratory event or we can simulate over a daily hydrograph'''
            
            flow = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Flow'].values[0]
            flow_df = pd.DataFrame()
            
            if flow == 'hydrograph':
                gage = str(self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Gage'].values[0])
                prorate = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Prorate'].values[0]
                flow_year = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['FlowYear'].values[0]

                # get gage data object from web
                gage_dat = hf.NWIS(site = gage, service='dv', start_date='1900-01-01')

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

                # extract year
                df['year'] = pd.DatetimeIndex(df['datetimeUTC']).year
                df = df[df.year == flow_year]
                
                # get months
                df['month'] = pd.DatetimeIndex(df['datetimeUTC']).month
                for i in month_list:
                    flow_df = flow_df.append(df[df.month == i])
            

            # for each species, perform the simulation for n individuals x times
            for spc in species:
                # extract a single row based on season and species
                spc_dat = self.pop[(self.pop['Season'] == season) & (self.pop.Species == spc)]

                # get scipy log normal distribution paramters - note values in centimeters
                s = spc_dat.s.values[0]
                len_loc = spc_dat.location.values[0]
                len_scale = spc_dat.scale.values[0]

                # get a priori length 
                mean_len = spc_dat.Length_mean.values[0]
                sd_len = spc_dat.Length_sd.values[0]

                # get species name
                species = spc_dat.Species.values[0]

                # get the number of times we are going to iterate this thing
                iterations = spc_dat['Iterations'].values[0]
                
                # get probability of occurence
                if math.isnan(spc_dat.occur_prob.values[0]):
                    occur_prob = 1.0
                else:
                    occur_prob = spc_dat.occur_prob.values[0]

                # create empty holders for some dictionaries
                u_param_dict = {}
                surv_dict = {}
                intake_vel_dict = {}
                units = []
                op_order_dict = {}
                        
                for row in self.unit_params.iterrows():
                    unit = row[1]['Unit']
                    runner_type = row[1]['Runner Type']
                    intake_vel_dict[row[1]['Unit']] = row[1]['intake_vel']
                    units.append(unit)
                    op_order_dict[unit] = row[1]['op_order']
                        
                    # create parameter dictionary for every unit, a dictionary in a dictionary
                    if runner_type == 'Kaplan':

                        # built a parameter dictionary for the kaplan function
                        param_dict = {'H':float(row[1]['H']),
                                      'RPM':float(row[1]['RPM']),
                                      'D':float(row[1]['D']),
                                      'ada':float(row[1]['ada']),
                                      'N':float(row[1]['N']),
                                      'Qopt':float(row[1]['Qopt']),
                                      '_lambda':float(row[1]['lambda'])}
                        u_param_dict[unit] = param_dict

                    elif runner_type == 'Propeller':
                        # built a parameter dictionary for the kaplan function
                        param_dict = {'H':float(row[1]['H']),
                                      'RPM':float(row[1]['RPM']),
                                      'D':float(row[1]['D']),
                                      'ada':float(row[1]['ada']),
                                      'N':float(row[1]['N']),
                                      'Qopt':float(row[1]['Qopt']),
                                      'Qper':row[1]['Qper'],
                                      '_lambda':float(row[1]['lambda'])}
                        u_param_dict[unit] = param_dict                       
                        
                    elif runner_type == 'Francis':
                        # built a parameter dictionary for the Francis function
                        param_dict = {'H':float(row[1]['H']),
                                      'RPM':float(row[1]['RPM']),
                                      'D':float(row[1]['D']),
                                      'ada':float(row[1]['ada']),
                                      'N':float(row[1]['N']),
                                      'Qper':float (row[1]['Qper']),
                                      'iota' : float (row[1]['iota']),
                                      'D1' : float (row[1]['D1']),
                                      'D2' : float (row[1]['D2']),
                                      'B' : float (row[1]['B']),
                                      '_lambda':float(row[1]['lambda'])}
                        u_param_dict[unit] = param_dict

                # create survival dictionary, which is a dictionary of a priori surival rates
                for row in self.nodes.iterrows():
                    if row[1]['Surv_Fun'] == 'a priori':
                        surv_dict[row[1]['Location']] = row[1]['Survival']
                
                # create an empty dataframe to hold length 
                spc_length = pd.DataFrame()

                # create an iterator
                for i in np.arange(0,iterations,1):
                    if flow == 'hydrograph':
                        for row in flow_df.iterrows():
                            curr_Q = row[1]['DAvgFlow_prorate']
                            day = row[1]['datetimeUTC']
                            '''this is where we need to introduce daily flow, rather than 30 days
                            
                            we can also build the Q_dict here
                            
                            We also need to add Q to the u_param_dict which is a nested dictionary
                                    '''
                            Q_dict = {'curr_Q': curr_Q,
                                      'min_Q': min_Q,
                                      'env_Q': env_Q}
                            
                            # for unit in units, add curr_Q to u_param_dict, add each unit capacity to Q_dict
                            sta_cap = 0.0
                            for u in units:
                                u_param_dict[u]['Q'] = curr_Q
                                Q_dict[u] = self.unit_params[self.unit_params.Unit == u].Qcap.values[0]
                                sta_cap = sta_cap + self.unit_params[self.unit_params.Unit == u].Qcap.values[0]
                                
                            Q_dict['sta_cap'] = sta_cap
    
                            '''we need to roll the dice here and determine whether or not fish are present at site'''
                            presence_seed = np.random.uniform(0,1)
                            
                            if occur_prob >= presence_seed:

                                
                                # if we don't have pareto parameters, we are passing a population
                                if math.isnan(spc_dat.param1.values[0]):
                                    n = np.int(spc_dat.Fish.values[0])
        
                                # fit a pareto or generalized pareto using scipy
                                else:
                                    shape = spc_dat.param1.values[0]
                                    loc = spc_dat.param2.values[0]
                                    scale = spc_dat.param3.values[0]
                                    if spc_dat.dist.values[0] == 'Pareto':
                                        ent_rate = pareto.rvs(shape, loc, scale, 1, random_state=rng)
                                    elif spc_dat.dist.values[0] == 'Extreme':
                                        ent_rate = genextreme.rvs(shape, loc, scale, 1, random_state=rng)
                                    else:
                                        ent_rate = weibull_min.rvs(shape, loc, scale, 1, random_state=rng)
        
                                    ent_rate = np.abs(ent_rate)
                                    print ("Entrainment rate of %s %s during %s simulated"%(round(ent_rate[0],4),spc,scen))
            
                                    # apply order of magnitude filter, if entrainment rate is 1 order of magnitude larger than largest observed entrainment rate, reduce
                                    max_ent_rate = spc_dat.max_ent_rate.values[0]
            
                                    if np.log10(ent_rate[0]) > np.log10(max_ent_rate):
            
                                        # how many orders of magnitude larger is the simulated entrainment rate than the largest entrainment rate on record?
                                        magnitudes = np.ceil(np.log10(ent_rate[0])) - np.ceil(np.log10(max_ent_rate)) + 0.5
            
                                        if magnitudes < 1.:
                                            magnitudes = 1.
            
                                        # reduce by at least 1 order of magnitude
                                        ent_rate = np.abs(ent_rate / 10**magnitudes)
                                        print ("New entrainment rate of %s"%(round(ent_rate[0],4)))
            
                                    # because we are simulating passage via spill - we need the number of fish in the river at time, not just flowing through units
                                    Mft3 = (60 * 60 * hours * curr_Q)/1000000
        
                                    # calcualte sample size
                                    n = np.round(Mft3 * ent_rate,0)[0]
                            else:
                                n = 0
                                
                            if n > 0:
                                print ("Resulting in an entrainment event of %s %s"%(np.int(n),spc))
                                
                                if math.isnan(s) == False:
                                    # create population of fish - IN CM!!!!!
                                    population = np.abs(lognorm.rvs(s, len_loc, len_scale, np.int(n), random_state=rng))
                                    population = np.where(population > 150,150,population)
                                    # convert lengths in cm to feet
                                    population = population * 0.0328084
                                else:
                                    population = np.abs(np.random.normal(mean_len, sd_len, np.int(n)))/12.0
    
                                # calculate sustained swim speed (ft/s)
                                if math.isnan(spc_dat.caudal_AR.values[0]) == False:
                                    AR = spc_dat.caudal_AR
                                    v_speed = np.vectorize(speed,excluded = [1,2])
                                    swim_speed = v_speed(population,AR,0)
                                else:
                                    swim_speed = np.zeros(len(population))
    
    
                                print ("created population for %s iteration:%s day: %s"%(species,i,day))
                                # start this iterations dataframe
                                iteration = pd.DataFrame({'scenario_num':np.repeat(scen_num,np.int(n)),
                                                          'species':np.repeat(species,np.int(n)),
                                                          'flow_scenario':np.repeat(scenario,np.int(n)),
                                                          'season':np.repeat(season,np.int(n)),
                                                          'iteration':np.repeat(i,np.int(n)),
                                                          'day':np.repeat(day,np.int(n)),
                                                          'flow':np.repeat(curr_Q,np.int(n)),
                                                          'population':np.float32(population),
                                                          'state_0':np.repeat('river_node_0',np.int(n))})
                                
    
                                for k in self.moves:
                                    if k == 0:
                                        # initial status
                                        status = np.repeat(1,np.int(n))
                                    else:
                                        status = iteration['survival_%s'%(k-1)].values
    
                                    # initial location
                                    location = iteration['state_%s'%(k)].values
    
                                    def surv_fun_att(state,surv_fun_dict):
                                        fun_typ = surv_fun_dict[state]['Surv_Fun']
                                        return fun_typ
    
                                    v_surv_fun = np.vectorize(surv_fun_att,excluded = [1])
                                    surv_fun = v_surv_fun(location,self.surv_fun_dict)
    
                                    # simulate survival draws
                                    dice = np.random.uniform(0.,1.,np.int(n))
    
                                    # vectorize STRYKE survival function
                                    v_surv_rate = np.vectorize(node_surv_rate, excluded = [4,5])
                                    rates = v_surv_rate(population,status,surv_fun,location,surv_dict,u_param_dict)
    
                                    # calculate survival
                                    survival = np.where(dice <= rates,1,0)
    
                                    # simulate movement
                                    if k < max(self.moves):
                                        # vectorize movement function
                                        v_movement = np.vectorize(movement,excluded = [3,4,5,6])
                                        
                                        # have fish move to the next node
                                        move = v_movement(location, 
                                                          survival,
                                                          swim_speed,
                                                          self.graph,
                                                          intake_vel_dict,
                                                          Q_dict,
                                                          op_order_dict)
    
                                    # add onto iteration dataframe, attach columns
                                    iteration['draw_%s'%(k)] = np.float32(dice)
                                    iteration['rates_%s'%(k)] = np.float32(rates)
                                    iteration['survival_%s'%(k)] = np.float32(survival)
    
                                    if k < max(self.moves):
                                        iteration['state_%s'%(k+1)] = move
    
                                # save that data
                                iteration.to_hdf(self.hdf,'simulations/%s/%s'%(scen,spc), mode = 'a', format = 'table', append = True)
                                self.hdf.flush()                               
                                

                                # start filling in that summary dictionary
                                row = [spc,scenario,season,str(i),day,curr_Q,str(len(iteration))]

                                columns = ['species','scenario','season','iteration','day','flow','pop_size']

                                # figure out number entrained and number suvived
                                counts = iteration.groupby(by = ['state_2'])['survival_2']\
                                    .count().to_frame().reset_index().rename(columns = {'survival_2':'entrained'})
                                sums = iteration.groupby(by = ['state_2'])['survival_2']\
                                    .sum().to_frame().reset_index().rename(columns = {'survival_2':'survived'})

                                # merge and calculate entrainment survival
                                ent_stats = counts.merge(sums,how = 'left',on ='state_2', copy = False)
                                ent_stats.fillna(0,inplace = True)
                                ent_stats['mortality'] = ent_stats.entrained - ent_stats.survived

                                # for each unit, calculate the number entrained and the number killed
                                for u in units:
                                    udat = ent_stats[ent_stats.state_2 == u]
                                    if len(udat) > 0:
                                        columns.append('num_entrained_%s'%(u))
                                        row.append(str(udat.entrained.values[0]))
                                        columns.append('num_killed_%s'%(u))
                                        row.append(str(udat.mortality.values[0]))
                                    else:
                                        columns.append('num_entrained_%s'%(u))
                                        row.append(str(0))
                                        columns.append('num_killed_%s'%(u))
                                        row.append(str(0))
                                # extract population and iteration
                                length_dat = iteration[['population','flow_scenario','season','iteration','day','state_2','survival_2']]
    
                                # append to species length dataframe
                                spc_length = spc_length.append(length_dat, ignore_index = True)
                                    
                            else:
                                print ("No fish of this species on %s"%(day))

                                row = [spc,scenario,season,str(i),day,curr_Q,str(0)]


                                columns = ['species','scenario','season','iteration','day','flow','pop_size']

                                # for each unit, calculate the number entrained and the number killed
                                for u in units:
                                    columns.append('num_entrained_%s'%(u))
                                    row.append(str(0))
                                    columns.append('num_killed_%s'%(u))
                                    row.append(str(0))
                            
                            # write daily summary to hdf - first convert to dataframe
                            daily = pd.DataFrame(columns = columns)
                            daily.loc[0] = row
                            daily.to_hdf(self.hdf,'Daily',mode = 'a',format = 'table', append = True)
                            self.hdf.flush()
                        
                    else:
                        #create an iterator to simulate days for the number of months passed to the season
                        for j in np.arange(0,months * 30,1):
                            curr_Q = flow
                            #day = row[1]['datetimeUTC']
                            '''this is where we need to introduce daily flow, rather than 30 days
                            
                            we can also build the Q_dict here
                            
                            We also need to add Q to the u_param_dict which is a nested dictionary
                                    '''
                            Q_dict = {'curr_Q': curr_Q,
                                      'min_Q': min_Q,
                                      'env_Q': env_Q}
                            
                            # for unit in units, add curr_Q to u_param_dict, add each unit capacity to Q_dict
                            sta_cap = 0.0
                            for u in units:
                                u_param_dict[u]['Q'] = curr_Q
                                Q_dict[u] = self.unit_params[self.unit_params.Unit == u].Qcap.values[0]
                                sta_cap = sta_cap + self.unit_params[self.unit_params.Unit == u].Qcap.values[0]
                                
                            Q_dict['sta_cap'] = sta_cap
                            
                            '''we need to roll the dice here and determine whether or not fish are present at site'''
                            presence_seed = np.random.uniform(0,1)
                            
                            if occur_prob >= presence_seed:
                                # if we don't have pareto parameters, we are passing a population
                                if math.isnan(spc_dat.param1.values[0]):
                                    n = np.int(spc_dat.Fish.values[0])
        
                                # fit a pareto or generalized pareto using scipy
                                else:
                                    shape = spc_dat.param1.values[0]
                                    loc = spc_dat.param2.values[0]
                                    scale = spc_dat.param3.values[0]
                                    if spc_dat.dist.values[0] == 'Pareto':
                                        ent_rate = pareto.rvs(shape, loc, scale, 1, random_state=rng)
                                    elif spc_dat.dist.values[0] == 'Extreme':
                                        ent_rate = genextreme.rvs(shape, loc, scale, 1, random_state=rng)
                                    else:
                                        ent_rate = weibull_min.rvs(shape, loc, scale, 1, random_state=rng)
        
                                    ent_rate = np.abs(ent_rate)
                                    print ("Entrainment rate of %s %s during %s simulated"%(round(ent_rate[0],4),spc,scen))
            
                                    # apply order of magnitude filter, if entrainment rate is 1 order of magnitude larger than largest observed entrainment rate, reduce
                                    max_ent_rate = spc_dat.max_ent_rate.values[0]
            
                                    if np.log10(ent_rate[0]) > np.log10(max_ent_rate):
            
                                        # how many orders of magnitude larger is the simulated entrainment rate than the largest entrainment rate on record?
                                        magnitudes = np.ceil(np.log10(ent_rate[0])) - np.ceil(np.log10(max_ent_rate)) + 0.5
            
                                        if magnitudes < 1.:
                                            magnitudes = 1.
            
                                        # reduce by at least 1 order of magnitude
                                        ent_rate = np.abs(ent_rate / 10**magnitudes)
                                        print ("New entrainment rate of %s"%(round(ent_rate[0],4)))
            
                                    # because we are simulating passage via spill - we need the number of fish in the river at time, not just flowing through units
                                    Mft3 = (60 * 60 * hours * curr_Q)/1000000
        
                                    # calcualte sample size
                                    n = np.round(Mft3 * ent_rate,0)[0]
                            else:
                                n = 0                            
                            
                            if n > 0:
                                print ("Resulting in an entrainment event of %s %s"%(np.int(n),spc))
    
                                if math.isnan(s) == False:
                                    # create population of fish - IN CM!!!!!
                                    population = np.abs(lognorm.rvs(s, len_loc, len_scale, np.int(n), random_state=rng))
                                    population = np.where(population > 150,150,population)
                                    # convert lengths in cm to feet
                                    population = population * 0.0328084
                                else:
                                    population = np.abs(np.random.normal(mean_len, sd_len, np.int(n)))/12.0
    
                                # calculate sustained swim speed (ft/s)
                                if math.isnan(spc_dat.caudal_AR.values[0]) == False:
                                    AR = spc_dat.caudal_AR
                                    v_speed = np.vectorize(speed,excluded = [1,2])
                                    swim_speed = v_speed(population,AR,0)
                                else:
                                    swim_speed = np.zeros(len(population))
    
    
                                print ("created population for %s iteration:%s day: %s"%(species,i,j))
                                # start this iterations dataframe
                                iteration = pd.DataFrame({'scenario_num':np.repeat(scen_num,np.int(n)),
                                                          'species':np.repeat(species,np.int(n)),
                                                          'flow_scenario':np.repeat(scen,np.int(n)),
                                                          'season':np.repeat(season,np.int(n)),
                                                          'iteration':np.repeat(i,np.int(n)),
                                                          'day':np.repeat(j,np.int(n)),
                                                          'flow':np.repeat(flow,np.int(n)),
                                                          'population':np.float32(population),
                                                          'state_0':np.repeat('river_node_0',np.int(n))})
    
                                for k in self.moves:
                                    if k == 0:
                                        # initial status
                                        status = np.repeat(1,np.int(n))
                                    else:
                                        status = iteration['survival_%s'%(k-1)].values
    
                                    # initial location
                                    location = iteration['state_%s'%(k)].values
    
                                    def surv_fun_att(state,surv_fun_dict):
                                        fun_typ = surv_fun_dict[state]['Surv_Fun']
                                        return fun_typ
    
                                    v_surv_fun = np.vectorize(surv_fun_att,excluded = [1])
                                    surv_fun = v_surv_fun(location,self.surv_fun_dict)
    
                                    # simulate survival draws
                                    dice = np.random.uniform(0.,1.,np.int(n))
    
                                    # vectorize STRYKE survival function
                                    v_surv_rate = np.vectorize(node_surv_rate, excluded = [4,5])
                                    rates = v_surv_rate(population,status,surv_fun,location,surv_dict,u_param_dict)
    
                                    # calculate survival
                                    survival = np.where(dice <= rates,1,0)
    
                                    # simulate movement
                                    if k < max(self.moves):
                                        # vectorize movement function
                                        v_movement = np.vectorize(movement,excluded = [3,4,5,6])
                                        
                                        # have fish move to the next node
                                        move = v_movement(location, 
                                                          survival,
                                                          swim_speed,
                                                          self.graph,
                                                          intake_vel_dict,
                                                          Q_dict,
                                                          op_order_dict)
    
                                    # add onto iteration dataframe, attach columns
                                    iteration['draw_%s'%(k)] = np.float32(dice)
                                    iteration['rates_%s'%(k)] = np.float32(rates)
                                    iteration['survival_%s'%(k)] = np.float32(survival)
    
                                    if k <= max(self.moves):
                                        iteration['state_%s'%(k+1)] = move
    
                                # save that data
                                iteration.to_hdf(self.hdf,'simulations/%s/%s'%(scen,spc), mode = 'a', format = 'table', append = True)
                                self.hdf.flush()

                                                           
                                
                                # start filling in that summary dictionary
                                row = [spc,scenario,season,i,j,curr_Q,len(iteration)]
                                columns = ['species','scenario','season','iteration','day','flow','pop_size']

                                # figure out number entrained and number suvived
                                counts = iteration.groupby(by = ['state_2'])['survival_2']\
                                    .count().to_frame().reset_index().rename(columns = {'survival_2':'entrained'})
                                sums = iteration.groupby(by = ['state_2'])['survival_2']\
                                    .sum().to_frame().reset_index().rename(columns = {'survival_2':'survived'})

                                # merge and calculate entrainment survival
                                ent_stats = counts.merge(sums,how = 'left',on ='state_2', copy = False)
                                ent_stats.fillna(0,inplace = True)
                                ent_stats['mortality'] = ent_stats.entrained - ent_stats.survived

                                # for each unit, calculate the number entrained and the number killed
                                for u in units:
                                    udat = ent_stats[ent_stats.state_2 == u]
                                    if len(udat) > 0:
                                        columns.append('num_entrained_%s'%(u))
                                        row.append(udat.entrained.values[0])
                                        columns.append('num_killed_%s'%(u))
                                        row.append(udat.mortality.values[0])
                                    else:
                                        columns.append('num_entrained_%s'%(u))
                                        row.append(0)
                                        columns.append('num_killed_%s'%(u))
                                        row.append(0)
                                # extract population and iteration
                                length_dat = iteration[['population','flow_scenario','season','iteration','day','state_2','survival_2']]
                                length_dat.rename(columns = {'state_2':'state','survival_2':'survival'}, inplace = True)
    
                                # append to species length dataframe
                                spc_length = spc_length.append(length_dat, ignore_index = True)
                                    
                            else:
                                print ("No fish of this species on %s"%(day))

                                row = [spc,scenario,season,i,day,curr_Q,0]

                                columns = ['species','scenario','season','iteration','day','flow','pop_size']

                                # for each unit, calculate the number entrained and the number killed
                                for u in units:
                                    columns.append('num_entrained_%s'%(u))
                                    row.append(0)
                                    columns.append('num_killed_%s'%(u))
                                    row.append(0)
                            
                            # write daily summary to hdf - first convert to dataframe
                            daily = pd.DataFrame(columns = columns)
                            daily.loc[0] = row
                            daily['iteration'] = daily.iteration.astype(int)
                            daily['day'] = daily.day.astype(str)
                            daily['flow'] = daily.flow.astype(str)
                            daily['pop_size'] = daily.pop_size.astype(str)
                            filter_cols = [col for col in daily if col.startswith('num_entrained')]
                            for c in filter_cols:
                                daily[c] = daily[c].astype(str)
                            del c
                            daily.astype(dtype = {'iteration':np.int}, copy = False)
                            daily.to_hdf(self.hdf,'Daily',mode = 'a',format = 'table', append = True)
                            self.hdf.flush()
            
                # write species length to database
                spc_length.to_hdf(self.hdf,key = 'Length', mode = 'a', format = 'table', append = True)
            self.hdf.flush()
            print ("Completed Scenario %s %s"%(species,scen))                            
            
        print ("Completed Simulations - view results")
        self.hdf.flush()
        self.hdf.close()

    def summary(self):
        '''Function summarizes entrainment risk hdf file'''
        # create hdf store
        self.hdf = pd.HDFStore(os.path.join(self.proj_dir,'%s.h5'%(self.output_name)))

        # create some empty holders
        self.beta_dict = {}

        # get Population table
        pop = self.hdf['Population']
        species = pop.Species.unique()

        # get Scenarios
        scen = self.hdf['Scenarios']
        scens = scen.Scenario.unique()

        # get units
        units = self.hdf['Unit_Parameters'].Unit.values

        self.daily_summary = self.hdf['Daily']
        self.daily_summary.iloc[:,6:] = self.daily_summary.iloc[:,6:].astype(float)

        print ("iterate through species and scenarios and summarize")
        # now loop over dem species, scenarios, iterations and days then calculate them stats
        for i in species:
            # create empty dataframe to hold all lengths for this particular species
            spc_length = self.hdf['Length']
            
            # calculate length stats for this species
            self.length_summ = spc_length.groupby(['season','state','survival']).population.describe()
            print ("summarized length by season, state, and survival")

            for j in scens:
                # get daily data for this species/scenario
                dat = self.hdf['simulations/%s/%s'%(j,i)]

                # summarize species-scenario - whole project
                whole_proj_succ = dat.groupby(by = ['iteration','day'])['survival_%s'%(max(self.moves))]\
                    .sum().to_frame().reset_index(drop = False).rename(columns = {'survival_%s'%(max(self.moves)):'successes'})
                whole_proj_count = dat.groupby(by = ['iteration','day'])['survival_%s'%(max(self.moves))]\
                    .count().to_frame().reset_index(drop = False).rename(columns = {'survival_%s'%(max(self.moves)):'count'})

                # merge successes and counts
                whole_summ = whole_proj_succ.merge(whole_proj_count)

                # calculate probabilities, fit to beta, write to dictionary summarizing results
                whole_summ['prob'] = whole_summ['successes']/whole_summ['count']
                try:
                    whole_params = beta.fit(whole_summ.prob.values)
                    whole_median = beta.median(whole_params[0],whole_params[1],whole_params[2],whole_params[3])
                    whole_std = beta.std(whole_params[0],whole_params[1],whole_params[2],whole_params[3])
                    
                    #whole_95ci = beta.interval(alpha = 0.95,a = whole_params[0],b = whole_params[1],loc = whole_params[2],scale = whole_params[3])
                    lcl = beta.ppf(0.025,a = whole_params[0],b = whole_params[1],loc = whole_params[2],scale = whole_params[3])
                    ucl = beta.ppf(0.975,a = whole_params[0],b = whole_params[1],loc = whole_params[2],scale = whole_params[3])  
                    self.beta_dict['%s_%s_%s'%(j,i,'whole')] = [j,i,'whole',whole_median,whole_std,lcl,ucl]
                except:
                    continue
                #print ("whole project survival for %s in scenario %s iteraton %s expected to be %s (%s,%s)"%(i,j,k,np.round(whole_median,2),np.round(whole_95ci[0],2),np.round(whole_95ci[1],2)))
                for l in self.moves:
                    # we need to remove the fish that died at the previous state
                    if l > 0:
                        sub_dat = dat[dat['survival_%s'%(l-1)] == 1]
                    else:
                        sub_dat = dat

                    # group by iteration and state
                    route_succ = sub_dat.groupby(by = ['iteration','day','state_%s'%(l)])['survival_%s'%(l)]\
                        .sum().to_frame().reset_index(drop = False).rename(columns = {'survival_%s'%(l):'successes'})
                    route_count = sub_dat.groupby(by = ['iteration','day','state_%s'%(l)])['survival_%s'%(l)]\
                        .count().to_frame().reset_index(drop = False).rename(columns = {'survival_%s'%(l):'count'})

                    # merge successes and counts
                    route_summ = route_succ.merge(route_count)

                    # calculate probabilities
                    route_summ['prob'] = route_summ['successes']/route_summ['count']

                    # extract route specific dataframes and fit beta
                    states = route_summ['state_%s'%(l)].unique()
                    for m in states:
                        st_df = route_summ[route_summ['state_%s'%(l)] == m]
                        try:
                            st_params = beta.fit(st_df.prob.values)
                            st_median = beta.median(st_params[0],st_params[1],st_params[2],st_params[3])
                            st_std = beta.std(st_params[0],st_params[1],st_params[2],st_params[3])
    
                            lcl = beta.ppf(0.025,a = st_params[0],b = st_params[1],loc = st_params[2],scale = st_params[3])
                            ucl = beta.ppf(0.975,a = st_params[0],b = st_params[1],loc = st_params[2],scale = st_params[3])                                    
                        except:
                            st_median = 0.5
                            st_std = 1.0
                            lcl = 0.
                            ucl = 1.

                        # add results to beta dictionary
                        self.beta_dict['%s_%s_%s'%(j,i,m)] = [j,i,m,st_median,st_std,lcl,ucl]
                print ("Fit beta distributions to states")
                del dat

        self.beta_df = pd.DataFrame.from_dict(data = self.beta_dict, orient = 'index', columns = ['scenario number','species','state','survival rate','variance','ll','ul'])
        
        self.hdf.flush()
        self.hdf.close()
        
        # calculate total killed and total entrained
        self.daily_summary['total_killed'] = self.daily_summary.filter(regex = 'num_killed', axis = 'columns').sum(axis = 1)
        self.daily_summary['total_entrained'] = self.daily_summary.filter(regex = 'num_entrained', axis = 'columns').sum(axis = 1)
        try:
            self.daily_summary['day'] = self.daily_summary['day'].dt.tz_localize(None)
        except:
            pass
        
        # create yearly summary by summing on species, flow scenario, and iteration
        yearly_summary = self.daily_summary.groupby(by = ['species','scenario','iteration'])['pop_size','total_killed','total_entrained'].sum()
        yearly_summary.reset_index(inplace = True)

        cum_sum_dict = {'species':[],
                        'scenario':[],
                        'med_population':[],
                        'med_entrained':[], 
                        'med_dead':[],
                        'median_ent':[],
                        'lcl_ent':[],
                        'ucl_ent':[],
                        'prob_gt_10_entrained':[],
                        'prob_gt_100_entrained':[],
                        'prob_gt_1000_entrained':[],
                        'median_killed':[],
                        'lcl_killed':[],
                        'ucl_killed':[],
                        'prob_gt_10_killed':[],
                        'prob_gt_100_killed':[],
                        'prob_gt_1000_killed':[]}
        # daily summary
        for fishy in yearly_summary.species.unique():
            #iterate over scenarios
            for scen in yearly_summary.scenario.unique():
                # get data
                idat = yearly_summary[(yearly_summary.species == fishy) & (yearly_summary.scenario == scen)]
                
                # get cumulative sums and append to dictionary
                cum_sum_dict['species'].append(fishy)
                cum_sum_dict['scenario'].append(scen)
                cum_sum_dict['med_population'].append(idat.pop_size.median())
                cum_sum_dict['med_entrained'].append(idat.total_entrained.median())
                cum_sum_dict['med_dead'].append(idat.total_killed.median())
                
                day_dat = self.daily_summary[(self.daily_summary.species == fishy) & (self.daily_summary.scenario == scen)]
                
                # fit distribution to number entrained
                dist = weibull_min.fit(day_dat.total_entrained)
                probs = weibull_min.sf([10,100,1000],dist[0],dist[1],dist[2])
                
                median = weibull_min.median(dist[0],dist[1],dist[2])
                lcl = weibull_min.ppf(0.025,dist[0],dist[1],dist[2])
                ucl = weibull_min.ppf(0.975,dist[0],dist[1],dist[2])
                
                cum_sum_dict['median_ent'].append(median)
                cum_sum_dict['lcl_ent'].append(lcl)
                cum_sum_dict['ucl_ent'].append(ucl)
                cum_sum_dict['prob_gt_10_entrained'].append(probs[0])
                cum_sum_dict['prob_gt_100_entrained'].append(probs[1])
                cum_sum_dict['prob_gt_1000_entrained'].append(probs[2])
                
                # fit distribution to number killed
                dist = weibull_min.fit(day_dat.total_killed)
                probs = weibull_min.sf([10,100,1000],dist[0],dist[1],dist[2])
                
                median = weibull_min.median(dist[0],dist[1],dist[2])
                lcl = weibull_min.ppf(0.025,dist[0],dist[1],dist[2])
                ucl = weibull_min.ppf(0.975,dist[0],dist[1],dist[2])
                
                cum_sum_dict['median_killed'].append(median)
                cum_sum_dict['lcl_killed'].append(lcl)
                cum_sum_dict['ucl_killed'].append(ucl)
                cum_sum_dict['prob_gt_10_killed'].append(probs[0])
                cum_sum_dict['prob_gt_100_killed'].append(probs[1])
                cum_sum_dict['prob_gt_1000_killed'].append(probs[2])
        print ("Yearly summary complete")        
        # plt.figure()
        # plt.hist(day_dat.total_entrained,color = 'r')
        # #plt.savefig(os.path.join(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke\Output",'fuck.png'), dpi = 700)
        # plt.show()        
                
        self.cum_sum = pd.DataFrame.from_dict(cum_sum_dict,orient = 'columns')

class hydrologic():
    '''python class object that conducts flow exceedance analysis using recent USGS data as afunction of the contributing watershed size.
    We have develped a strong linear relationship between drainage area and flow exceedances for the 100 nearest gages to the dam.  With
    this relationship we can predict what a wet spring looks like.'''

    def __init__(self, nid_near_gage_dir, output_dir):
        ''' to initialize the hydrologic class, provide:
            dam_dir = a shapefile directory for affected dams, must be subset of National Inventory of Dams (USACE 2004)
            gage_dir = a shapefile directory of USGS gages active since 2009
            nid_near_gage_dir = link to nid_near_gage table, csv file which lists the 100 nearest USGS gages to every NID dam
            output_dir = link to the output directory'''
        # establish output directory
        self.output_dir = output_dir

        # import dams, gages, and the near table and construct the exceedance table
        #self.nid = gp.read_file(dam_dir)
        #self.gages_shp = gp.read_file(gage_dir)
        self.NID_to_gage = pd.read_csv(nid_near_gage_dir, dtype={'STAID': object})
        self.DAvgFlow = pd.DataFrame()
        self.dams = self.NID_to_gage.NIDID.unique()

        print("Data imported, proceed to data extraction and exceedance calculation ")
        #self.gages = self.gages_shp.STAID.unique()
        self.gages = self.NID_to_gage.STAID.unique() 
        self.nearest_gages = self.NID_to_gage.STAID.values.tolist()

        self.nearest_gages = set(self.nearest_gages)
        print ("There are %s near gages"%(len(self.nearest_gages)))
        for i in self.nearest_gages:
            active = self.NID_to_gage[self.NID_to_gage.STAID == i].ACTIVE09.values[0]
            if active == 'yes':
                try:
                    print ("Start analyzing gage %s"%(i))
                    # get gage data object from web
                    gage = hf.NWIS(site = str(i),service = 'dv', start_date = '1900-01-01')
    
                    # extract dataframe
                    df = gage.df()
    
                    # replace column names
                    for j in gage.df().columns:
                        if '00060' in j:
                            if 'qualifiers' not in j:
                                if ':00000' in j or ':00003' in j:
                                    df.rename(columns = {j:'DAvgFlow'},inplace = True)            # reset index
                    df.reset_index(inplace = True)
    
                    #extract what we need
                    df = df[['datetimeUTC','DAvgFlow']]
    
                    # convert to datetime
                    df['datetimeUTC'] = pd.to_datetime(df.datetimeUTC)
    
                    # extract month
                    df['month'] = pd.DatetimeIndex(df['datetimeUTC']).month
    
                    curr_gage = self.NID_to_gage[self.NID_to_gage.STAID == i]
    
                    curr_name = curr_gage.iloc[0]['STANAME']
    
                    curr_huc = curr_gage.iloc[0]['HUC02']
    
                    drain_sqkm = np.float(curr_gage.iloc[0]['DRAIN_SQKM'])
    
                    df['STAID'] = np.repeat(curr_gage.iloc[0]['STAID'], len(df.index))
                    df['Name'] = np.repeat(curr_name, len(df.index))
                    df['HUC02'] = np.repeat(curr_huc, len(df.index))
                    df['Drain_sqkm'] = np.repeat(drain_sqkm, len(df.index))
    
                    self.DAvgFlow = self.DAvgFlow.append(df)
    
                    print ("stream gage %s with a drainage area of %s square kilometers added to flow data."%(i,drain_sqkm))
                except:
                    continue

    def seasonal_exceedance(self, seasonal_dict, exceedence, HUC = None):
        '''function calculates the 90, 50, and 10 percent exceedance flows by season and writes to an output data frame.

        seasonal_dict = python dictionary consisting of season (key) with a Python list like object of month numbers (value)'''

        self.exceedance = pd.DataFrame()
        self.DAvgFlow['season'] = np.empty(len(self.DAvgFlow))
        DAvgFlow = self.DAvgFlow
        for key in seasonal_dict:
            for month in seasonal_dict[key]:
                self.DAvgFlow.loc[self.DAvgFlow['month'] == month, 'season'] = key

        # seasonal exceedance probability
        self.DAvgFlow['SeasonalExcProb'] = self.DAvgFlow.groupby(['season','STAID'])['DAvgFlow'].rank(ascending = False, method = 'first',pct = True) * 100

        for i in self.nearest_gages:
            # extract exceedance probabilities and add to exceedance data frame
            for key in seasonal_dict:
                season = self.DAvgFlow[(self.DAvgFlow.season == key) & (self.DAvgFlow.STAID == i)]
                if HUC is not None:
                    season = season[season.HUC02 == HUC]
                if len(season) > 0:
                    season.sort_values(by = 'SeasonalExcProb', ascending = False, inplace = True)
                    print('length of season dataframe is %s'%(len(season.index)))
                    for j in exceedence:
                        if j == 0:
                            exc = season.DAvgFlow.max()
                        else:
                            excdf = season[season.SeasonalExcProb <= np.float32(j)]
                            exc = excdf.DAvgFlow.min()
                            if exc < 0:
                                exc = 0.
                        print ("Gage %s has a %s percent exceedance flow of %s in %s"%(i,j,exc,key))
        
                        # get gage information from gage shapefile
                        curr_gage = self.NID_to_gage[self.NID_to_gage.STAID == str(i)]
                        curr_name = curr_gage.iloc[0]['STANAME']
                        curr_huc = np.int(curr_gage.iloc[0]['HUC02'])
                        drain_sqkm = np.float(curr_gage.iloc[0]['DRAIN_SQKM'])
                        row = np.array([i,curr_name,curr_huc,drain_sqkm,key,exc,j])
                        newRow = pd.DataFrame(np.array([row]),columns = ['STAID','Name','HUC02','Drain_sqkm','season','flow','exceedance'])
                        self.exceedance = self.exceedance.append(newRow)

    def curve_fit(self, season, dam, exceedance):
        '''function uses statsmodels to perform OLS regaression and describe exceedance probablity as a function of watershed size,

            required inputs inlcude:
                season = string object denoting current season
                dam = string object denoting current dam
                exccednace = string object pulling specific exceedance column, 'exc_90','exc_50','exc_10' '''

        # get dam data
        dam_df = self.NID_to_gage[self.NID_to_gage.NIDID == dam]

        # get feature id
        nidid = dam_df.iloc[0]['NIDID']

        # get drainage area in sq miles
        drain_sqmi = dam_df.iloc[0]['Drainage_a']
        drain_sqkm = drain_sqmi * 2.58999

        # filter exceedance
        seasonal_exceedance_df = self.exceedance[(self.exceedance.season == season) & (self.exceedance.exceedance == str(exceedance))]
        self.dat = seasonal_exceedance_df
        
        # extract X and y arrays
        self.X = seasonal_exceedance_df.Drain_sqkm.values.astype(np.float32)
        self.Y = seasonal_exceedance_df.flow.values.astype(np.float32)

        # fit a linear model
        model = sm.OLS(self.Y, self.X).fit()
        print ("-------------------------------------------------------------------------------")
        print ("strke fit an OLS regression model to the data with p-value of %s"%(model.f_pvalue))
        print (model.summary())
        print ("-------------------------------------------------------------------------------")
        if model.f_pvalue < 0.05:
            coef = model.params[0]
            exc_flow = drain_sqkm * coef
            self.DamX = np.float(drain_sqkm)
            self.DamY = np.float(exc_flow)
            print ("dam %s with a drainage area of %s sq km has a %s percent exceedance flow of %s in %s"%(dam,round(drain_sqkm,2),exceedance,round(exc_flow,2),season))
        else:
            print ("Model not significant, there is no exceedance flow estimate")
            self.DamX = np.nan
            self.DamY = np.nan
class epri():
    '''python class object that queries the EPRI entrainment database and fits
    a pareto distribution to the observations.'''

    def __init__(self,
                 states = None,
                 plant_cap = None,
                 Month = None,
                 Family = None,
                 Genus = None,
                 Species = None,
                 Feeding_Guild = None,
                 Habitat = None,
                 Water_Type = None,
                 Mussel_Host = None,
                 Seasonal_Migrant = None,
                 HUC02 = None,
                 NIDID = None,  
                 River = None):

        '''The EPRI database can be queried many different ways.  Note, these are
        optional named arguments, meaning the end user doesn't have to query the
        database at all.  In this instance the returned Pareto distribution
        parameters will be representative of the entire dataset.

        -states = list like object of state abbreviations or single state abbreviation
        -plant_cap = tuple object indicating the plant capacity (cfs) cutoff and the direction, i.e. (1000,'gt')
        -Month = list like object of Month integer or single Month integer object.
        -Family = list like object of Scientific Families or single Family string object
        -Feeding_Guild = list like object of abbreviated feeding guilds or single feeding guild string object.
            List of appropriate feeding guilds:
               CA = carnivore
               FF = filter feeder
               HE = herbivore
               IC = insectivorous cyprinid
               IN = invertivore
               OM = omnivore
               PR = parasite
        -Species = list like object of common names or single common name string object
        -Habitat = list like object of abbreviated habitat preferences or single preference string object
            List of appropriate habitat types:
                BEN = benthic
                BFS = benthic fluvial specialist
                FS = fluvial specialist
                Lit = littoral (near cover/shorelines)
                Pel = pelagic
                Pool = pool (minnows)
                RP = run/pool (minnows)
                RRP = riffle/run/pool (minnows)
        -Water_Type = list like object of abbreviated water size preferences or single preference string object
            List of appropriate water sizes:
                BW = big water
                SS = small stream
                Either = can be found in all river sizes
        -Mussel_Host = is this fish a mussel host?
            List of appropriate
                Yes
                No
                null
        -Seasonal Migrant = string object indicating migration season.
            List of appropriate seasons:
                'Spring', 'Spring/Fall', 'Spring-Summer','Fall','Fall-Winter'
        '''

        # import EPRI database


   
        self.epri = pd.read_csv(r"..\data\epri1997.csv",  encoding= 'unicode_escape')

        ''' I want to hook up stryke to the EPRI database when project loads, figure out how to do this cuz this is lame'''

        if NIDID is not None:
            if isinstance(NIDID,str):
                self.epri = self.epri[self.epri.NIDID != NIDID]
            else:
                self.epri = self.epri[~self.epri['NIDID'].isin(NIDID)]

        if states is not None:
            if isinstance(states,str):
                self.epri = self.epri[self.epri.State == states]
            else:
                self.epri = self.epri[self.epri['State'].isin(states)]

        if plant_cap is not None:
            if plant_cap[1] == '>':
                self.epri = self.epri[self.epri.Plant_cap_cfs > plant_cap[0]]
            else:
                self.epri = self.epri[self.epri.Plant_cap_cfs <= plant_cap[0]]

        if Month is not None:
            if isinstance(Month,str):
                self.epri = self.epri[self.epri.Month == Month]
            else:
                self.epri = self.epri[self.epri['Month'].isin(Month)]

        if Family is not None:
            if isinstance(Family,str):
                self.epri = self.epri[self.epri['Family'] == Family]
            else:
                self.epri = self.epri[self.epri['Family'].isin(Family)]

        if Species is not None:
            if isinstance(Species,str):
                self.epri = self.epri[self.epri.Species == Species]
            else:
                self.epri = self.epri[self.epri['Species'].isin(Species)]
                
        if Genus is not None:
            if isinstance(Genus,str):
                self.epri = self.epri[self.epri.Genus == Genus]
            else:
                self.epri = self.epri[self.epri['Genus'].isin(Genus)]

        if Feeding_Guild is not None:
            if isinstance(Feeding_Guild,str):
                self.epri = self.epri[self.epri.FeedingGuild == Feeding_Guild]
            else:
                self.epri = self.epri[self.epri['FeedingGuild'].isin(Feeding_Guild)]

        if Habitat is not None:
            if isinstance(Habitat,str):
                self.epri = self.epri[self.epri.Habitat == Habitat]
            else:
                self.epri = self.epri[self.epri['Habitat'].isin(Habitat)]

        if Water_Type is not None:
            if isinstance(Water_Type,str):
                self.epri = self.epri[self.epri.WaterType == Water_Type]
            else:
                self.epri = self.epri[self.epri['WaterType'].isin(Water_Type)]

        if Mussel_Host is not None:
            if isinstance(Mussel_Host,str):
                self.epri = self.epri[self.epri.Host == Mussel_Host]
            else:
                self.epri = self.epri[self.epri['Host'].isin(Mussel_Host)]

        if Seasonal_Migrant is not None:
            if isinstance(Seasonal_Migrant,str):
                self.epri = self.epri[self.epri.Migrant == Seasonal_Migrant]
            else:
                self.epri = self.epri[self.epri['Migrant'].isin(Seasonal_Migrant)]
        
        if HUC02 is not None:
            if isinstance(HUC02,int):
                self.epri = self.epri[self.epri.HUC02 == HUC02]
            else:
                self.epri = self.epri[self.epri['HUC02'].isin(HUC02)]
                
        if River is not None:
            if isinstance(River,str):
                self.epri = self.epri[self.epri.River == River]
            else:
                self.epri = self.epri[self.epri['River'].isin(River)]          
        # calculate probability of presence
        success = self.epri.Present.sum()
        trials = len(self.epri)
        print ("--------------------------------------------------------------------------------------------")        
        print ("out of %s potential samples %s had this species present for %s probability of presence"%(trials,success,round(float(success/trials),4)))
        print ("--------------------------------------------------------------------------------------------")
 
        self.epri.fillna(0,inplace = True)
        self.epri = self.epri[self.epri.Present == 1]

        
        #print ("With a maximum entrainment rate of %s and only %s percent of records acount for 80 percent of the entrainment"%(self.epri.FishPerMft3.max(),
        #                                                                                                                round(len(self.epri[self.epri.FishPerMft3 > self.epri.FishPerMft3.max()*0.8]) / len(self.epri) * 100,2)))
        print ("There are %s records left to describe entrainment rates"%(len(self.epri)))
        print ("The maximum entrainment rate for this fish is: %s"%(self.epri.FishPerMft3.max()))
        print ("--------------------------------------------------------------------------------------------")
    def ParetoFit(self):
        ''' Function fits a Pareto distribution to the epri dataset relating to
        the species of interest'''

        # fit a pareto and write to the object
        self.dist_pareto = pareto.fit(self.epri.FishPerMft3.values, floc = 0)
        print ("The Pareto distribution has a shape parameter of b: %s,  location: %s and scale: %s"%(round(self.dist_pareto[0],4),
                                                                                                      round(self.dist_pareto[1],4),
                                                                                                      round(self.dist_pareto[2],4)))
        print ("The Pareto mean is: %s"% (pareto.mean(self.dist_pareto[0],self.dist_pareto[1],self.dist_pareto[2])))
        print ("The Pareto variance is: %s"% (pareto.var(self.dist_pareto[0],self.dist_pareto[1],self.dist_pareto[2])))
        print ("The Pareto standard deviation is: %s"% (pareto.std(self.dist_pareto[0],self.dist_pareto[1],self.dist_pareto[2])))
        print ("--------------------------------------------------------------------------------------------")


    def ExtremeFit(self):
        ''' Function fits a generic extreme value distribution to the epri dataset relating to
        the species of interest'''

        # fit a pareto and write to the object
        self.dist_extreme = genextreme.fit(self.epri.FishPerMft3.values, floc = 0)
        print ("The Generic Extreme Value distribution has a shape parameter of c: %s,  location: %s and scale: %s"%(round(self.dist_extreme[0],4),
                                                                                                      round(self.dist_extreme[1],4),
                                                                                                      round(self.dist_extreme[2],4)))
        print ("The Generic Extreme Value mean is: %s"% (genextreme.mean(self.dist_extreme[0],self.dist_extreme[1],self.dist_extreme[2])))
        print ("The Generic Extreme Value variance is: %s"% (genextreme.var(self.dist_extreme[0],self.dist_extreme[1],self.dist_extreme[2])))
        print ("The Generic Extreme Value standard deviation is: %s"% (genextreme.std(self.dist_extreme[0],self.dist_extreme[1],self.dist_extreme[2])))
        print ("--------------------------------------------------------------------------------------------")

    def WeibullMinFit(self):
       ''' Function fits a Frechet distribution to the epri dataset relating to
       the species of interest'''

       # fit a pareto and write to the object
       self.dist_weibull = weibull_min.fit(self.epri.FishPerMft3.values, floc = 0)
       print ("The Weibull Max distribution has a shape parameter of c: %s,  location: %s and scale: %s"%(round(self.dist_weibull[0],4),
                                                                                                     round(self.dist_weibull[1],4),
                                                                                                     round(self.dist_weibull[2],4)))
       print ("The Weibull Max mean is: %s"% (weibull_min.mean(self.dist_weibull[0],self.dist_weibull[1],self.dist_weibull[2])))
       print ("The Weibull Max variance is: %s"% (weibull_min.var(self.dist_weibull[0],self.dist_weibull[1],self.dist_weibull[2])))
       print ("The Weibull Max standard deviation is: %s"% (weibull_min.std(self.dist_weibull[0],self.dist_weibull[1],self.dist_weibull[2])))
       print ("--------------------------------------------------------------------------------------------")

    def GumbelFit(self):
       ''' Function fits a Frechet distribution to the epri dataset relating to
       the species of interest'''

       # fit a pareto and write to the object
       self.dist_gumbel = gumbel_r.fit(self.epri.FishPerMft3.values)
       print ("The Gumbel distribution has a shape parameter of location: %s and scale: %s"%(round(self.dist_gumbel[0],4),
                                                                                                     round(self.dist_gumbel[1],4)))
       print ("--------------------------------------------------------------------------------------------")
    def LengthSummary(self):
        '''Function summarizes length for species of interest using EPRI database'''

        # sum up the number of observations within each size cohort
        cm_0_5 = np.int(self.epri['0_5'].sum())
        cm_5_10 = np.int(self.epri['5_10'].sum())
        cm_10_15 = np.int(self.epri['10_15'].sum())
        cm_15_20 = np.int(self.epri['15_20'].sum())
        cm_20_25 = np.int(self.epri['20_25'].sum())
        cm_25_38 = np.int(self.epri['25_38'].sum())
        cm_38_51 = np.int(self.epri['38_51'].sum())
        cm_51_64 = np.int(self.epri['51_64'].sum())
        cm_64_76 = np.int(self.epri['64_76'].sum())
        cm_GT76 = np.int(self.epri['GT76'].sum())

        # sample from uniform distribution within each size cohort
        cm_0_5_arr = np.random.uniform(low = 0, high = 5.0, size = cm_0_5)
        cm_5_10_arr = np.random.uniform(low = 5.0, high = 10.0, size = cm_5_10)
        cm_10_55_arr = np.random.uniform(low = 10.0, high = 15.0, size = cm_10_15)
        cm_15_20_arr = np.random.uniform(low = 15.0, high = 20.0, size = cm_15_20)
        cm_20_25_arr = np.random.uniform(low = 20.0, high = 25.0, size = cm_20_25)
        cm_25_38_arr = np.random.uniform(low = 25.0, high = 38.0, size = cm_25_38)
        cm_38_51_arr = np.random.uniform(low = 38.0, high = 51.0, size = cm_38_51)
        cm_51_64_arr = np.random.uniform(low = 51.0, high = 64.0, size = cm_51_64)
        cm_64_76_arr = np.random.uniform(low = 64.0, high = 76.0, size = cm_64_76)
        cm_GT76_arr = np.random.uniform(low = 76.0, high = 100.0, size = cm_GT76)

        # append them all together into 1 array
        self.lengths = np.concatenate((cm_0_5_arr,
                                       cm_5_10_arr,
                                       cm_10_55_arr,
                                       cm_15_20_arr,
                                       cm_20_25_arr,
                                       cm_25_38_arr,
                                       cm_38_51_arr,
                                       cm_51_64_arr,
                                       cm_64_76_arr,
                                       cm_GT76_arr),
                                      axis = 0)

        # now fit that array to a log normal
        self.len_dist = lognorm.fit(self.lengths)
        print("The log normal distribution has a shape parameter s: %s, location: %s and scale: %s"%(round(self.len_dist[0],4),
                                                                                                     round(self.len_dist[1],4),
                                                                                                     round(self.len_dist[2],4)))






