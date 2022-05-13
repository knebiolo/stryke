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
import hydrofunctions as hf
#import geopandas as gp
import statsmodels.api as sm
import math
from scipy.stats import pareto, genextreme, genpareto, lognorm, weibull_min
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
    #beta = np.arctan((0.707 * (np.pi/8))/(iota * Qwd * np.power(D1/D2,3))) #IPD: what is Qper? relook @ this equation ~ discussed 2/5/20
    #beta = np.arctan((0.707 * (np.pi/8))/(iota * Qwd * Q_per * np.power(D1/D2,3))) #IPD: what is Qper? relook @ this equation ~ discussed 2/5/20

    tan_beta = (0.707 * (np.pi/8))/(iota * Qwd * np.power(D1/D2,3)) #IPD: what is Qper? relook @ this equation ~ discussed 2/5/20
    
    #alpha = np.radians(90) - np.arctan((2 * np.pi * Ewd * ada)/Qwd * (B/D1) + (np.pi * 0.707**2)/(2 * Qwd) * (B/D1) * (np.power(D2/D1,2)) - 4 * 0.707 * np.tan(beta) * (B/D1) * (D1/D2)) #IPD: should be tan(beta) ~ corrected 2/5/20
    alpha = np.radians(90) - np.arctan((2 * np.pi * Ewd * ada)/Qwd * (B/D1) + (np.pi * 0.707**2)/(2 * Qwd) * (B/D1) * (np.power(D2/D1,2)) - 4 * 0.707 * tan_beta * (B/D1) * (D1/D2)) #IPD: should be tan(beta) ~ corrected 2/5/20


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

    nodes = pd.read_excel(wks_dir,'Nodes',header = 0,index_col = None, usecols = "B:C", skiprows = 9)
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

def movement (location, status, length, swim_speed, graph, intake_vel_dict):
    if status == 1:
        # get neighbors
        neighbors = graph[location]

        ''' we need to apportion our movement probabilities by edge weights
        iterate through neighbors and assign them ranges 0 - 1.0 based on
        movement weights - (]'''

        locs = []
        probs = []
        for i in neighbors:
            if graph[location][i]['weight'] > 0.0:
                locs.append(i)
                probs.append(graph[location][i]['weight'])
        new_loc = np.random.choice(locs,1,p = probs)[0]
        # filter out those fish that can escape intake velocity
        if np.sum(swim_speed) > 0:
            if 'U' in new_loc:
                if swim_speed > intake_vel_dict[new_loc]:
                    new_loc = 'spill'
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
        
    log_sa = -0.828 + 0.6196 * np.log10(L*30.48) + 0.3478 * np.log10(A) + 0.7621 * M
    
    return (10**log_sa) * 0.911344

class simulation():
    ''' Python class object that initiates, runs, and holds data for a facility
    specific simulation'''
    def __init__ (self, proj_dir, wks, output_name, existing = False):
        if existing == False:
            # create workspace directory
            self.wks_dir = os.path.join(proj_dir,wks)
            
            # extract scenarios from input spreadsheet
            self.routing = pd.read_excel(self.wks_dir,'Routing',header = 0,index_col = None, usecols = "B:G", skiprows = 9)

            # import nodes and create a survival function dictionary
            self.nodes = pd.read_excel(self.wks_dir,'Nodes',header = 0,index_col = None, usecols = "B:C", skiprows = 9)
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
            self.unit_params = pd.read_excel(self.wks_dir,'Unit Params', header = 0, index_col = None, usecols = "B:P", skiprows = 4)

            # join unit parameters to scenarios
            self.scenario_dat = self.routing.merge(self.unit_params, how = 'left', left_on = 'State', right_on = 'Unit')

            # get hydraulic capacity of facility
            self.flow_cap = self.unit_params.Qcap.sum()

            # identify unique flow scenarios
            self.scenarios_df = pd.read_excel(self.wks_dir,'Flow Scenarios',header = 0,index_col = None, usecols = "B:G", skiprows = 5)
            self.scenarios = self.scenarios_df['Scenario'].unique()

            # import population data
            self.pop = pd.read_excel(self.wks_dir,'Population',header = 0,index_col = None, usecols = "B:R", skiprows = 11)

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
            self.hdf['Routing'] = self.routing
            self.hdf.flush()
        else:
            self.wks_dir = os.path.join(proj_dir,wks)

            # create output HDF file
            self.proj_dir = proj_dir
            self.output_name = output_name

            # import nodes and create a survival function dictionary
            self.nodes = pd.read_excel(self.wks_dir,'Nodes',header = 0,index_col = None, usecols = "B:C", skiprows = 9)

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
            months = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Months'].values[0]
            if type(months) != np.int64:
                months = len(months.split(","))
            else:
                months = 1
            flow = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Flow'].values[0]
            hours = self.scenarios_df[self.scenarios_df['Scenario'] == scen]['Hours'].values[0]

            # identify the species we need to simulate for this scenario
            species = self.pop[self.pop['Season'] == season].Species.unique()

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

                # get scenario routing and node survival data for this species/flow scenario
                sc_dat = self.scenario_dat[self.scenario_dat['Scenario'] == scen]

                # create empty holders for some dictionaries
                u_param_dict = {}
                surv_dict = {}
                intake_vel_dict = {}
                # iterate through routing rows for this flow scenario
                for row in sc_dat.iterrows():

                    state = row[1]['State']
                    # create a unit parameter dictionary or add to the survival dictionary
                    if math.isnan(row[1]['lambda']) == False:
                        # get this unit's parameters
                        u_dat = self.unit_params[self.unit_params.Unit == state]
                        runner_type = u_dat['Runner Type'].values[0]
                        intake_vel_dict[u_dat.Unit.values[0]] = u_dat['intake_vel'].values[0]
                        
                        # create parameter dictionary for every unit, a dictionary in a dictionary
                        if runner_type == 'Kaplan':

                            # built a parameter dictionary for the kaplan function
                            param_dict = {'H':float(u_dat.H.values[0]),
                                          'RPM':float(u_dat.RPM.values[0]),
                                          'D':float(u_dat.D.values[0]),
                                          'Q':float(row[1]['Q']),
                                          'ada':float(u_dat.ada.values[0]),
                                          'N':float(u_dat.N.values[0]),
                                          'Qopt':float(u_dat.Qopt.values[0]),
                                          '_lambda':float(row[1]['lambda'])}
                            u_param_dict[state] = param_dict

                        elif runner_type == 'Propeller':
                            # built a parameter dictionary for the kaplan function
                            param_dict = {'H':float(row[1]['H']),
                                          'RPM':float(row[1]['RPM']),
                                          'D':float(row[1]['D']),
                                          'Q':float(row[1]['Q']),
                                          'ada':float(row[1]['ada']),
                                          'N':float(row[1]['N']),
                                          'Qopt':float(row[1]['Qopt']),
                                          'Qper':row[1]['Qper'],
                                          '_lambda':float(row[1]['lambda'])}
                            u_param_dict[state] = param_dict

                        # elif runner_type == 'Francis':
                        #     # built a parameter dictionary for the kaplan function
                        #     param_dict = {'H':float(row[1]['H']),
                        #                   'RPM':float(row[1]['RPM']),
                        #                   'D':float(row[1]['D']),
                        #                   'Q':float(row[1]['Q']),
                        #                   'ada':float(row[1]['ada']),
                        #                   'N':float(row[1]['N']),
                        #                   'Q_opt':float(row[1]['Qopt']),
                        #                   'Q_per':float(row[1]['Qper']),
                        #                   'iota':float(row[1]['iota']),
                        #                   'D1':float(row[1]['D1']),
                        #                   'D2':float(row[1]['D2']),
                        #                   'B':float(row[1]['B']),
                        #                   '_lambda':float(row[1]['lambda'])}
                        #     u_param_dict[state] = param_dict                            
                            
                        #print (u_param_dict)
                        elif runner_type == 'Francis':
                            # built a parameter dictionary for the Francis function
                            param_dict = {'H':float(row[1]['H']),
                                          'RPM':float(row[1]['RPM']),
                                          'D':float(row[1]['D']),
                                          'Q':float(row[1]['Q']),
                                          'ada':float(row[1]['ada']),
                                          'N':float(row[1]['N']),
                                          'Qper':float (row[1]['Qper']),
                                          'iota' : float (row[1]['iota']),
                                          'D1' : float (row[1]['D1']),
                                          'D2' : float (row[1]['D2']),
                                          'B' : float (row[1]['B']),
                                          '_lambda':float(row[1]['lambda'])}
                            u_param_dict[state] = param_dict

                        #fuck

                    else:
                         surv_dict[state] = row[1]['Survival']

                    # alter the graph edge weigths with probability of movement into the node
                    for edge in self.graph.edges:
                        to_edge = edge[1]
                        if state == to_edge:
                            self.graph[edge[0]][edge[1]]['weight'] = row[1]['Probability of Movement']

                # create an iterator
                for i in np.arange(0,iterations,1):

                    #create an iterator to simulate days for the number of months passed to the season
                    for j in np.arange(0,months * 30,1):

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
                            
                            # get max entrainment length cutoff (either too large to fit or too fast)
                            # if math.isnan(spc_dat.caudal_AR.values[0]) == False:
                            #     max_ent_len = spc_dat.ent_cutoff_in.values[0]/12.0
                            # else:
                            #     max_ent_len = 100.
    
                            if np.log10(ent_rate[0]) > np.log10(max_ent_rate):
    
                                # how many orders of magnitude larger is the simulated entrainment rate than the largest entrainment rate on record?
                                magnitudes = np.ceil(np.log10(ent_rate[0])) - np.ceil(np.log10(max_ent_rate)) + 0.5
    
                                if magnitudes < 1.:
                                    magnitudes = 1.
    
                                # reduce by at least 1 order of magnitude
                                ent_rate = np.abs(ent_rate / 10**magnitudes)
                                print ("New entrainment rate of %s"%(round(ent_rate[0],4)))
    
                            # because we are simulating passage via spill - we need the number of fish in the river at time, not just flowing through units
                            Mft3 = (60 * 60 * hours * flow)/1000000

                            # calcualte sample size
                            n = np.round(Mft3 * ent_rate,0)[0]
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
                                                      'iteration':np.repeat(i,np.int(n)),
                                                      'day':np.repeat(j,np.int(n)),
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
                                    v_movement = np.vectorize(movement,excluded = [4,5,6])
                                    
                                    # have fish move to the next node
                                    move = v_movement(location, 
                                                      survival,
                                                      population,
                                                      swim_speed,
                                                      self.graph,
                                                      intake_vel_dict)

                                # add onto iteration dataframe, attach columns
                                iteration['draw_%s'%(k)] = np.float32(dice)
                                iteration['rates_%s'%(k)] = np.float32(rates)
                                iteration['survival_%s'%(k)] = np.float32(survival)

                                if k < max(self.moves):
                                    iteration['state_%s'%(k+1)] = move

                            # save that data
                            self.hdf['simulations/%s/%s/%s/%s'%(scen,spc,i,j)] = iteration
                            self.hdf.flush()
                        else:
                            print ("No fish this day")

                print ("Completed Scenario %s %s"%(species,scen))

        print ("Completed Simulations - view results")
        self.hdf.flush()
        self.hdf.close()

    def summary(self,whole_project_surv = False):
        '''Function summarizes entrainment risk hdf file'''
        # create hdf store
        self.hdf = pd.HDFStore(os.path.join(self.proj_dir,'%s.h5'%(self.output_name)))

        # create some empty holders
        self.beta_dict = {}

        tree = dict(dict(dict()))

        print ("Iterate through database, develop simulation tree")
        for key in self.hdf.keys():
            if key[0:4] == '/sim':
                levels = key.split("/")
                scenario = levels[2]
                species = levels[3]
                iteration = levels[4]
                day = levels[5]
                # if this scenario isn't in the first key
                if scenario not in tree.keys():
                    tree['%s'%(scenario)] = {}

                # if this species isn't in the first subkey
                if species not in tree['%s'%(scenario)].keys():
                    tree['%s'%(scenario)]['%s'%(species)] = {}

                if iteration not in tree['%s'%(scenario)]['%s'%(species)].keys():
                    tree['%s'%(scenario)]['%s'%(species)]['%s'%(iteration)] = []

                tree['%s'%(scenario)]['%s'%(species)]['%s'%(iteration)].append(day)

        # get Population table
        pop = self.hdf['Population']
        species = pop.Species.unique()

        # get Scenarios
        scen = self.hdf['Scenarios']
        scens = scen.Scenario.unique()

        # get units
        units = self.hdf['Unit_Parameters'].Unit.values

        self.daily_summary = pd.DataFrame()

        # create a summary dictionary to pandas data frame by column
        self.summary = dict()
        self.summary['species'] = []
        self.summary['scenario'] = []
        self.summary['flow_scen'] = []
        self.summary['season'] = []
        self.summary['iteration'] = []
        self.summary['day'] = []
        self.summary['pop_size'] = []
        self.summary['length_median'] = []
        self.summary['length_min'] = []
        self.summary['length_max'] = []
        self.summary['length_q1'] = []
        self.summary['length_q3'] = []
        #self.summary['avg_strike_prob'] = []

        for u in units:
            self.summary['num_entrained_%s'%(u)] = []
            self.summary['num_killed_%s'%(u)] = []

        length_dict = {}

        # now loop over dem species, scenarios, iterations and days then calculate them stats
        for i in species:
            # create empty dataframe to hold all lengths for this particular species
            spc_length = pd.DataFrame(columns = ['population','flow_scenario','iteration','day'])

            for j in scens:
                # create empty dataframe for summary
                dat = pd.DataFrame()
                flow_scen = j.split(' ')[0]
                season = j.split(' ')[2]
                try:
                    # identify the number of iterations this species/scenario ran for
                    iters = tree[j][i].keys()

                    for k in iters:
                        days = tree[j][i][k]

                        for d in days:
                            # generate hdf key
                            key = "simulations/" + j + "/" + i + "/" + k + "/" + d

                            # extract key to pandas dataframe
                            day_dat = pd.read_hdf(self.hdf, key = key)

                            # append to summary data frame
                            dat = dat.append(day_dat)

                            # extract population and iteration
                            length_dat = day_dat[['population','flow_scenario','iteration','day']]

                            # append to species length dataframe
                            spc_length = spc_length.append(length_dat, ignore_index = True)

                            # start filling in that summary dictionary
                            self.summary['species'].append(i)
                            self.summary['scenario'].append(j)
                            self.summary['flow_scen'].append(flow_scen)
                            self.summary['season'].append(season)
                            self.summary['iteration'].append(k)
                            self.summary['day'].append(d)
                            self.summary['pop_size'].append(len(day_dat))
                            self.summary['length_median'].append(day_dat.population.median())
                            self.summary['length_min'].append(day_dat.population.min())
                            self.summary['length_max'].append(day_dat.population.max())
                            self.summary['length_q1'].append(day_dat.population.quantile(0.25))
                            self.summary['length_q3'].append(day_dat.population.quantile(0.75))
                            #self.summary['avg_strike_prob'].append(day_dat.rates_2.median())


                            # figure out number entrained and number suvived
                            counts = day_dat.groupby(by = ['state_2'])['survival_2']\
                                .count().to_frame().reset_index().rename(columns = {'survival_2':'entrained'})
                            sums = day_dat.groupby(by = ['state_2'])['survival_2']\
                                .sum().to_frame().reset_index().rename(columns = {'survival_2':'survived'})

                            # merge and calculate entrainment survival
                            ent_stats = counts.merge(sums,how = 'left',on ='state_2', copy = False)
                            ent_stats.fillna(0,inplace = True)
                            ent_stats['mortality'] = ent_stats.entrained - ent_stats.survived

                            # for each unit, calculate the number entrained and the number killed
                            for u in units:
                                udat = ent_stats[ent_stats.state_2 == u]
                                if len(udat) > 0:
                                    self.summary['num_entrained_%s'%(u)].append(udat.entrained.values[0])
                                    self.summary['num_killed_%s'%(u)].append(udat.mortality.values[0])
                                else:
                                    self.summary['num_entrained_%s'%(u)].append(0)
                                    self.summary['num_killed_%s'%(u)].append(0)

                        if whole_project_surv == True:
                            # summarize species-scenario - whole project
                            whole_proj_succ = dat.groupby(by = ['iteration','day'])['survival_%s'%(max(self.moves))]\
                                .sum().to_frame().reset_index(drop = False).rename(columns = {'survival_%s'%(max(self.moves)):'successes'})
                            whole_proj_count = dat.groupby(by = ['iteration','day'])['survival_%s'%(max(self.moves))]\
                                .count().to_frame().reset_index(drop = False).rename(columns = {'survival_%s'%(max(self.moves)):'count'})
    
                            # merge successes and counts
                            whole_summ = whole_proj_succ.merge(whole_proj_count)
    
                            # calculate probabilities, fit to beta, write to dictionary summarizing results
                            whole_summ['prob'] = whole_summ['successes']/whole_summ['count']
                            whole_params = beta.fit(whole_summ.prob.values)
                            whole_median = beta.median(whole_params[0],whole_params[1],whole_params[2],whole_params[3])
                            whole_std = beta.std(whole_params[0],whole_params[1],whole_params[2],whole_params[3])
                            
                            #whole_95ci = beta.interval(alpha = 0.95,a = whole_params[0],b = whole_params[1],loc = whole_params[2],scale = whole_params[3])
                            lcl = beta.ppf(0.025,a = whole_params[0],b = whole_params[1],loc = whole_params[2],scale = whole_params[3])
                            ucl = beta.ppf(0.975,a = whole_params[0],b = whole_params[1],loc = whole_params[2],scale = whole_params[3])  
                            self.beta_dict['%s_%s_%s'%(j,i,'whole')] = [j,i,'whole',whole_median,whole_std,lcl,ucl]
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
                except KeyError:
                    continue
            # calculate length stats for this species
            length_dict[i] = [spc_length.population.min(),
                              spc_length.population.quantile(q = 0.25),
                              spc_length.population.median(),
                              spc_length.population.quantile(q = 0.75),
                              spc_length.population.max()]

        if whole_project_surv == True:
            self.beta_df = pd.DataFrame.from_dict(data = self.beta_dict, orient = 'index', columns = ['scenario number','species','state','survival rate','variance','ll','ul'])
        self.summ_dat = pd.DataFrame.from_dict(data = self.summary, orient = 'columns')
        self.length_df = pd.DataFrame.from_dict(data = length_dict,orient = 'index', columns = ['min','q1','median','q3','max'])
        self.hdf.flush()
        self.hdf.close()
        
        # calculate total killed and total entrained
        self.summ_dat['total_killed'] = self.summ_dat.filter(regex = 'num_killed', axis = 'columns').sum(axis = 1)
        self.summ_dat['total_entrained'] = self.summ_dat.filter(regex = 'num_entrained', axis = 'columns').sum(axis = 1)
        yearly_summary = self.summ_dat.groupby(by = ['species','flow_scen','iteration'])['pop_size','total_killed','total_entrained'].sum()
        yearly_summary.reset_index(inplace = True)

        cum_sum_dict = {'species':[],
                        'flow_scen':[],
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
            for scen in yearly_summary.flow_scen.unique():
                # get data
                idat = yearly_summary[(yearly_summary.species == fishy) & (yearly_summary.flow_scen == scen)]
                
                # get cumulative sums and append to dictionary
                cum_sum_dict['species'].append(fishy)
                cum_sum_dict['flow_scen'].append(scen)
                cum_sum_dict['med_population'].append(idat.pop_size.median())
                cum_sum_dict['med_entrained'].append(idat.total_entrained.median())
                cum_sum_dict['med_dead'].append(idat.total_killed.median())
                
                day_dat = self.summ_dat[(self.summ_dat.species == fishy) & (self.summ_dat.flow_scen == scen)]
                
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
        self.dams = self.NID_to_gage.NIDID.values.tolist()

        print("Data imported, proceed to data extraction and exceedance calculation ")
        #self.gages = self.gages_shp.STAID.unique()
        self.gages = self.NID_to_gage.STAID.unique() 
        self.nearest_gages = self.NID_to_gage.STAID.values.tolist()

        self.nearest_gages = set(self.nearest_gages)
        print ("There are %s near gages"%(len(self.nearest_gages)))
        for i in self.nearest_gages:
            try:
                print ("Start analyzing gage %s"%(i))
                # get gage data object from web
                gage = hf.NWIS(site = str(i),service = 'dv', start_date = '2009-01-01')

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

    def seasonal_exceedance(self, seasonal_dict, HUC = None):
        '''function calculates the 90, 50, and 10 percent exceedance flows by season and writes to an output data frame.

        seasonal_dict = python dictionary consisting of season (key) with a Python list like object of month numbers (value)'''

        self.exceedance = pd.DataFrame()
        self.DAvgFlow['season'] = np.empty(len(self.DAvgFlow))
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
                season.sort_values(by = 'SeasonalExcProb', ascending = False, inplace = True)
                print('length of season dataframe is %s'%(len(season.index)))
                if len(season.index) > 0:

                    exc10df = season[season.SeasonalExcProb <= 10.]
                    exc10 = exc10df.DAvgFlow.min()
                    print ("Gage %s has a 10 percent exceedance flow of %s in %s"%(i,exc10,key))

                    exc50df = season[season.SeasonalExcProb <= 50.]
                    exc50 = exc50df.DAvgFlow.min()
                    print ("Gage %s has a 50 percent exceedance flow of %s in %s"%(i,exc50,key))

                    exc90df = season[season.SeasonalExcProb <= 90.]
                    exc90 = exc90df.DAvgFlow.min()
                    print ("Gage %s has a 90 percent exceedance flow of %s in %s"%(i,exc90,key))

                    # get gage information from gage shapefile
                    curr_gage = self.NID_to_gage[self.NID_to_gage.STAID == str(i)]
                    curr_name = curr_gage.iloc[0]['STANAME']
                    curr_huc = np.int(curr_gage.iloc[0]['HUC02'])
                    drain_sqkm = np.float(curr_gage.iloc[0]['DRAIN_SQKM'])
                    row = np.array([i,curr_name,curr_huc,drain_sqkm,key,exc90,exc50,exc10])
                    newRow = pd.DataFrame(np.array([row]),columns = ['STAID','Name','HUC02','Drain_sqkm','season','exc_90','exc_50','exc_10'])
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
        drain_sqmi = dam_df.iloc[0]['Drainage_A']
        drain_sqkm = drain_sqmi * 2.58999

        # extract the 100 nearest gages associated with this NID feature
        gages = self.NID_to_gage[self.NID_to_gage.NIDID == nidid].STAID.values

        # filter exceedance
        seasonal_exceedance_df = self.exceedance[self.exceedance.season == season]

        gage_dat = pd.DataFrame()

        for i in gages:
            dat = seasonal_exceedance_df[seasonal_exceedance_df.STAID == i]
            gage_dat = gage_dat.append(dat)

        # extract X and y arrays
        self.X = gage_dat.Drain_sqkm.values.astype(np.float32)
        self.Y = gage_dat['%s'%(exceedance)].values.astype(np.float32)

        # fit a linear model
        model = sm.OLS(self.Y, self.X).fit()
        print ("-------------------------------------------------------------------------------")
        print ("strke fit an OLS regression model to the data with p-value of %s"%(model.f_pvalue))
        print (model.summary())
        print ("-------------------------------------------------------------------------------")
        if model.f_pvalue < 0.05:
            coef = model.params[0]
            exc_flow = drain_sqkm * coef
            self.DamX = drain_sqkm
            self.DamY = exc_flow
            print ("dam %s with a drainage area of %s sq km has a %s percent exceedance flow of %s in %s"%(dam,round(drain_sqkm,2),exceedance.split("_")[1],round(exc_flow,2),season))

class epri():
    '''python class object that queries the EPRI entrainment database and fits
    a pareto distribution to the observations.'''

    def __init__(self,
                 states = None,
                 plant_cap = None,
                 Month = None,
                 Family = None,
                 Species = None,
                 Feeding_Guild = None,
                 Habitat = None,
                 Water_Type = None,
                 Mussel_Host = None,
                 Seasonal_Migrant = None):

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
        self.epri = pd.read_csv(r"J:\1508\028\Calcs\Entrainment\stryke\Data\epri1997.csv",  encoding= 'unicode_escape')
        ''' I want to hook up stryke to the EPRI database when project loads, figure out how to do this cuz this is lame'''

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

        self.epri.fillna(0,inplace = True)
        print ("--------------------------------------------------------------------------------------------")
        #print ("With a maximum entrainment rate of %s and only %s percent of records acount for 80 percent of the entrainment"%(self.epri.FishPerMft3.max(),
        #                                                                                                                round(len(self.epri[self.epri.FishPerMft3 > self.epri.FishPerMft3.max()*0.8]) / len(self.epri) * 100,2)))
        print ("There are %s records left to describe entrainment rates"%(len(self.epri)))
        print ("The maximum entrainment rate for this fish is: %s"%(self.epri.FishPerMft3.max()))
        print ("--------------------------------------------------------------------------------------------")
    def ParetoFit(self):
        ''' Function fits a Pareto distribution to the epri dataset relating to
        the species of interest'''

        # fit a pareto and write to the object
        self.dist_pareto = pareto.fit(self.epri.FishPerMft3.values)

        print ("The Pareto distribution has a shape parameter of b: %s,  location: %s and scale: %s"%(round(self.dist_pareto[0],4),
                                                                                                      round(self.dist_pareto[1],4),
                                                                                                      round(self.dist_pareto[2],4)))
        print ("%s percent of the entrainment events had 80 percent of the total entrainment impact"%(round(pareto.cdf(0.8,
                                                                                                                       self.dist_pareto[0],
                                                                                                                       self.dist_pareto[1],
                                                                                                                       self.dist_pareto[2]),2)))
    def ExtremeFit(self):
        ''' Function fits a generic extreme value distribution to the epri dataset relating to
        the species of interest'''

        # fit a pareto and write to the object
        self.dist_extreme = genextreme.fit(self.epri.FishPerMft3.values)

        print ("The Generic Extreme Value distribution has a shape parameter of c: %s,  location: %s and scale: %s"%(round(self.dist_extreme[0],4),
                                                                                                      round(self.dist_extreme[1],4),
                                                                                                      round(self.dist_extreme[2],4)))

    def WeibullMinFit(self):
       ''' Function fits a Frechet distribution to the epri dataset relating to
       the species of interest'''

       # fit a pareto and write to the object
       self.dist_weibull = weibull_min.fit(self.epri.FishPerMft3.values)

       print ("The Weibull Max distribution has a shape parameter of c: %s,  location: %s and scale: %s"%(round(self.dist_weibull[0],4),
                                                                                                     round(self.dist_weibull[1],4),
                                                                                                     round(self.dist_weibull[2],4)))


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






