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
import networkx as nx

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

def node_surv_rate(length,status,surv_fun,route,surv_dict,u_param_dict):
   
    if status == 0:
        return 0.0
    else:
        if surv_fun == 'a priori':
            if route == 'forebay':
                # get the a priori survival rate from the table in sqlite
                prob = surv_dict[route]
        
            elif route == 'spill':
                # get the a priori survival rate from the table in sqlite
                prob = surv_dict[route]
                
            elif route == 'tailrace':
                # get the a priori survival rate from the table in sqlite
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
    
        return prob

# create function that builds networkx graph object from nodes and edges in project database
def create_route(wks_dir):
    '''function creates a networkx graph object from information provided by 
    nodes and edges found in standard project database.  the only input
    is the standard project database.'''
       
    nodes = pd.read_excel(wks_dir,'Nodes',header = 0,index_col = None, usecols = "B:C", skiprows = [0,8])
    edges = pd.read_excel(wks_dir,'Edges',header = 0,index_col = None, usecols = "B:D", skiprows = [0,7])

    # create empty route object
    route = nx.route = nx.DiGraph()
    
    # add nodes to route - nodes.loc.values
    route.add_nodes_from(nodes.location.values)
    
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

def movement (location, status, graph):
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
    else:
        new_loc = location
                    
    return new_loc

def simulation(proj_dir,wks, export_results = False):
    wks_dir = os.path.join(proj_dir,wks)
    
    # extract scenarios from input spreadsheet    
    routing = pd.read_excel(wks_dir,'Routing',header = 0,index_col = None, usecols = "B:G", skiprows = [0,8])
    
    # import nodes and create a survival function dictionary
    nodes = pd.read_excel(wks_dir,'Nodes',header = 0,index_col = None, usecols = "B:C", skiprows = [0,8])
    surv_fun_df = nodes[['Location','Surv_Fun']].set_index('Location')
    surv_fun_dict = surv_fun_df.to_dict('index')
    
    # get last river node
    max_river_node = 0
    for i in nodes.Nodes.values:
        i_split = i.split("_")
        if len(i_split) > 1:
            river_node = i_split[1]
            if river_node > max_river_node:
                max_river_node = river_node
                
    
    # make a movement graph from input spreadsheet
    graph = create_route(wks_dir)
    print ("created a graph")
    
    
    # identify the number of moves that a fish can make
    path_list = nx.all_shortest_paths(graph,'river_node_0','river_node_%s'%(max_river_node))
    max_len = 0
    for i in path_list:
        path_len = len(i)
        if path_len > max_len:
            max_len = path_len
    moves = np.arange(0,max_len,1)
    print ("identified the number of moves, %s"%(moves))
    
    # import unit parameters
    unit_params = pd.read_excel(wks_dir,'Unit Params', header = 0, index_col = None, usecols = "B:O", skiprows = [0,3])
    
    # join unit parameters to scenarios
    scenario_dat = routing.join(unit_params, how = 'left', lsuffix = 'state', rsuffix = 'Unit')

    # identify unique flow scenarios
    scenarios_df = pd.read_excel(wks_dir,'Flow Scenarios',header = 0,index_col = None, usecols = "B:C", skiprows = [0,4])
    scenarios = scenarios_df['Flow Scenarios'].unique()
    
    # import population data
    pop = pd.read_excel(wks_dir,'Population',header = 0,index_col = None, usecols = "B:I", skiprows = [0,10])
    
    # create some empty holders
    all_results = pd.DataFrame()
    beta_dict = {}
    
    for scen in scenarios:
        scen_num = scenarios_df[scenarios_df['Flow Scenarios'] == scen]['Scenario Number'].values[0]
        # identify the species we need to simulate for this scenario
        species = pop[pop['Flow Scenario'] == scen].Species.unique()
        
        # for each species, perform the simulation for n individuals x times
        for spc in species:
 
            # extract the number of fish, their length,  and stdv from row
            spc_dat = pop[(pop['Flow Scenario'] == scen) & (pop.Species == spc)]
            
            if np.isnull(spc_dat.EntMean.values[0]):
                n = np.int(spc_dat.Fish.values[0])
            else:
                n = np.random.lognormal(mean = spc_dat.EntMean.values[0], 
                                        sigma = spc_dat.EntStDev.values[0], 
                                        size = 1)
            
            length = spc_dat.Length.values[0] / 12. 
            stdv = spc_dat.StDev.values[0] / 12. 
            species = spc_dat.Species.values[0] 
            flow_scen = spc_dat['Flow Scenario'].values[0]          
            iterations = spc_dat['Iterations'].values[0]
            # get scenario routing and node survival data for this species/flow scenario
            sc_dat = scenario_dat[scenario_dat['Flow Scenario'] == flow_scen]
            
            # create empty holders for some dictionaries
            u_param_dict = {}
            surv_dict = {}
            
            # iterate through routing rows for this flow scenario
            for row in sc_dat.iterrows():

                state = row[1]['State']
                # create a unit parameter dictionary or add to the survival dictionary
                if 'Unit' in state:
                    # get this unit's parameters
                    u_dat = unit_params[unit_params.Unit == state]
                    runner_type = u_dat['Runner Type'].values[0]

                    # create parameter dictionary for every unit, a dictionary in a dictionary
                    if runner_type == 'Kaplan':
                        
                        # built a parameter dictionary for the kaplan function
                        param_dict = {'H':float(row[1]['H']),
                                      'RPM':float(row[1]['RPM']),
                                      'D':float(row[1]['D']),
                                      'Q':float(row[1]['Q']),
                                      'ada':float(row[1]['ada']),
                                      'N':float(row[1]['N']),
                                      'Qopt':float(row[1]['Qopt']),
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
                    #print (u_param_dict)
                    #fuck
                    
                else:
                     surv_dict[state] = row[1]['Survival'] 
                
                # alter the graph edge weigths with probability of movement into the node
                for edge in graph.edges:
                    to_edge = edge[1]
                    if state == to_edge:
                        graph[edge[0]][edge[1]]['weight'] = row[1]['Probability of Movement']
            
            # create an empty dataframe for flow scenario results
            scen_results = pd.DataFrame()
            

            # create an iterator
            for i in np.arange(0,iterations,1):
                # create population of fish
                population = np.random.normal(length,stdv,n)
                print ("created population for %s iteration #:%s"%(species,i))
                # start this iterations dataframe
                iteration = pd.DataFrame({'scenario_num':np.repeat(scen_num,n),
                                          'species':np.repeat(species,n),
                                          'flow_scenario':np.repeat(flow_scen,n),
                                          'iteration':np.repeat(i,n),
                                          'population':population,
                                          'state_0':np.repeat('forebay',n)})
                
                for j in moves:
                    if j == 0:
                        # initial status
                        status = np.repeat(1,n)
                    else:
                        status = iteration['survival_%s'%(j-1)].values
    
                    # initial location
                    location = iteration['state_%s'%(j)].values
                    
                    def surv_fun_att(state,surv_fun_dict):
                        fun_typ = surv_fun_dict[state]['surv_fun']
                        return fun_typ
                    
                    v_surv_fun = np.vectorize(surv_fun_att,excluded = [1])
                    surv_fun = v_surv_fun(location,surv_fun_dict)
                    # simulate survival draws
                    dice = np.random.uniform(0.,1.,n)
            
                    # vectorize STRYKE survival function
                    v_surv_rate = np.vectorize(node_surv_rate, excluded = [4,5])
                    rates = v_surv_rate(population,status,surv_fun,location,surv_dict,u_param_dict)

                    # calculate survival
                    survival = np.where(dice > rates,0,1)
                    print ("assessed survival for state %s"%(j))
                    
                    # simulate movement 
                    if j < max(moves):
                        # vectorize movement function
                        v_movement = np.vectorize(movement,excluded = [2])
                        move = v_movement(location,survival,graph)
                        print ("assessed movement to state %s"%(j+1))
                    
                    # add onto iteration dataframe, attach columns
                    iteration['draw_%s'%(j)] = dice
                    iteration['rates_%s'%(j)] = rates
                    iteration['survival_%s'%(j)] = survival
    
                    if j < max(moves):
                        iteration['state_%s'%(j+1)] = move
                    

                scen_results = scen_results.append(iteration)
                all_results = all_results.append(iteration)
            if export_results == True:
                # write scenario results to csv for further inspection
                scen_results.to_csv(os.path.join(proj_dir,"%s_%s_scen_%s.csv"%(species,flow_scen,scen_num)))
            

            # summarize scenario - whole project
            whole_proj_succ = scen_results.groupby(by = 'iteration')['survival_%s'%(max(moves))].sum().to_frame().reset_index(drop = False).rename(columns = {'survival_%s'%(max(moves)):'successes'})
            whole_proj_count = scen_results.groupby(by = 'iteration')['survival_%s'%(max(moves))].count().to_frame().reset_index(drop = False).rename(columns = {'survival_%s'%(max(moves)):'count'})
        
            # merge successes and counts
            whole_summ = whole_proj_succ.merge(whole_proj_count)
        
            # calculate probabilities, fit to beta, write to dictionary summarizing results
            whole_summ['prob'] = whole_summ['successes']/whole_summ['count']
            whole_params = beta.fit(whole_summ.prob.values)
            whole_median = beta.median(whole_params[0],whole_params[1],whole_params[2],whole_params[3])
            whole_std = beta.std(whole_params[0],whole_params[1],whole_params[2],whole_params[3])
            whole_95ci = beta.interval(alpha = 0.95,a = whole_params[0],b = whole_params[1],loc = whole_params[2],scale = whole_params[3])
            
            beta_dict['%s_%s_%s'%(scen,spc,'whole')] = [scen,spc,'whole',whole_median,whole_std,whole_95ci[0],whole_95ci[1]]
                                                         
            print ("whole project survival for %s expected to be %s (%s,%s)"%(species,np.round(whole_median,2),np.round(whole_95ci[0],2),np.round(whole_95ci[1],2)))
            for j in moves:
                if j > 0:
                    scen_results = scen_results[scen_results['survival_%s'%(j-1)]==1]
                # summarize scenario - whole project
                route_succ = scen_results.groupby(by = ['iteration','state_%s'%(j)])['survival_%s'%(j)].sum().to_frame().reset_index(drop = False).rename(columns = {'survival_%s'%(j):'successes'})
                route_count = scen_results.groupby(by = ['iteration','state_%s'%(j)])['survival_%s'%(j)].count().to_frame().reset_index(drop = False).rename(columns = {'survival_%s'%(j):'count'})
                # merge successes and counts
                route_summ = route_succ.merge(route_count)
                # calculate probabilities
                route_summ['prob'] = route_summ['successes']/route_summ['count']
                # extract route specific dataframes and fit beta\
                states = route_summ['state_%s'%(j)].unique()
                for k in states:
                    st_df = route_summ[route_summ['state_%s'%(j)] == k]
                    st_params = beta.fit(st_df.prob.values)
                    st_median = beta.median(st_params[0],st_params[1],st_params[2],st_params[3])
                    st_std = beta.std(st_params[0],st_params[1],st_params[2],st_params[3])
                    st_95ci = beta.interval(alpha = 0.95,a = st_params[0],b = st_params[1],loc = st_params[2],scale = st_params[3])
                    beta_dict['%s_%s_%s'%(scen,spc,k)] = [scen,spc,k,st_median,st_std,st_95ci[0],st_95ci[1]]
                
            print ("Completed Scenario %s %s"%(species,flow_scen))

                
    
    print ("Completed Simulations - view results")
    return pd.DataFrame.from_dict(beta_dict,orient = 'index',columns = ['scenario','species','state','est','std','ll','ul'])






