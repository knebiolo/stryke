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
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import beta


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
    # mandatory project tables
    c.execute('''DROP TABLE IF EXISTS tblFrancis''')                           # table to hold standard Frank et. al. tbsm parameters for Francis by facility unit #
    c.execute('''DROP TABLE IF EXISTS tblKaplan''')                            # table to hold standard Frank et. al. tbsm parameters for Kaplan by facility unit # - yes even if they only have francis 
    c.execute('''DROP TABLE IF EXISTS tblPropeller''')                         # you guessed it, now one for propellers
    c.execute('''DROP TABLE IF EXISTS tblPump''')                         # you guessed it, now one for propellers
    c.execute('''DROP TABLE IF EXISTS tblNodes''')                             # placeholder for route - will hold a networkx graph object
    c.execute('''DROP TABLE IF EXISTS tblEdges''')
    c.execute('''CREATE TABLE tblNodes(location TEXT PRIMARY KEY, 
                                       surv_fun TEXT CHECK(surv_fun = "a priori" OR 
                                                           surv_fun = "Kaplan" OR 
                                                           surv_fun = "Francis" OR 
                                                           surv_fun = "Propeller" OR
                                                           surv_fun = "Pump"), 
                                       prob REAL)''')
    c.execute('''CREATE TABLE tblEdges(_from TEXT, 
                                       _to TEXT, 
                                       weight REAL,
                                       FOREIGN KEY (_from)
                                           REFERENCES tblNodes(location),
                                       FOREIGN KEY (_to)
                                           REFERENCES tblNodes(location))''')
    c.execute('''CREATE TABLE tblFrancis(unit TEXT, 
                                         H REAL, 
                                         RPM INTEGER, 
                                         D REAL,
                                         Q REAL, 
                                         Q_per REAL, 
                                         ada REAL, 
                                         N INTEGER,
                                         iota REAL, 
                                         D1 REAL, 
                                         D2 REAL, 
                                         B REAL,
                                         _lambda REAL,
                                         FOREIGN KEY (unit)
                                             REFERENCES tblNodes(location))''')
    c.execute('''CREATE TABLE tblKaplan(unit TEXT, 
                                        H REAL, 
                                        RPM INTEGER, 
                                        D REAL,
                                        Q REAL, 
                                        ada REAL, 
                                        N INTEGER, 
                                        Qopt REAL,
                                         _lambda REAL,
                                        FOREIGN KEY (unit)
                                             REFERENCES tblNodes(location))''')
    c.execute('''CREATE TABLE tblPropeller(unit TEXT, 
                                           H REAL, 
                                           RPM INTEGER, 
                                           D REAL, 
                                           Q REAL, 
                                           Q_per REAL, 
                                           ada REAL,
                                           N INTEGER, 
                                           Qopt REAL,
                                         _lambda REAL,                                           
                                           FOREIGN KEY (unit)
                                             REFERENCES tblNodes(location))''')

    c.execute('''CREATE TABLE tblPump(unit TEXT, 
                                           H REAL, 
                                           RPM INTEGER, 
                                           D REAL, 
                                           Q REAL, 
                                           Q_p REAL, 
                                           ada REAL,
                                           N INTEGER, 
                                           D1 REAL, 
                                           D2 REAL,
                                           B REAL,
                                           Qopt REAL,
                                           _lambda REAL,
                                           gamma REAL,                                        
                                           FOREIGN KEY (unit)
                                             REFERENCES tblNodes(location))''')
    conn.commit()   
    c.close()

# create function that builds networkx graph object from nodes and edges in project database
def create_route(dbDir):
    '''function creates a networkx graph object from information provided by 
    nodes and edges found in standard project database.  the only input
    is the standard project database.'''
    
    # get nodes and edges - this assumes the end user has updated the database!
    node_sql = "SELECT * FROM tblNodes"
    edges_sql = "SELECT * FROM tblEdges"
    conn = sqlite3.connect(dbDir, timeout=30.0)
    c = conn.cursor()   
    nodes = pd.read_sql(node_sql,con = conn)
    edges = pd.read_sql(edges_sql,con = conn)
    c.close()
    
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
        
    # make a plot for posterity     
    labels = {}
    pos = {}
    idx = 0
    for i in nodes.location.values:
        labels[idx] = i
        idx = idx + 1

    pos = nx.layout.kamada_kawai_layout(route, center = [0,0], scale = 5)
    plt.subplot(111)
    nx.draw_networkx_nodes(route, pos, node_size = 1000, node_color = 'white')
    nx.draw_networkx_labels(route, pos, font_size = 6)
    nx.draw_networkx_edges(route, pos, node_size = 1000, width = np.array(weights) * 2.0)
    plt.show()
    
    # return finished product and enjoy functionality of networkx
    return route

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
    
class fish():
    '''python class object describing an individual fish in our individual based
    model.  The fish has two primary functions (methods); they can survive and
    they can move.  If they are dead, they can no longer move.  
    
    The simulation ends for an individual fish when there are no more moves to 
    make or its dead.'''
    
    def __init__(self,species,len_params,route,dbDir,simulation,fish):
        self.species = species
        self.length = np.random.normal(len_params[0],len_params[1])
        self.route = route
        self.status = 1   # fish are alive at start
        self.complete = 0 # fish do not start the simulation in a completed step
        self.location = 'forebay'
        self.dbDir = dbDir
        self.simulation = simulation
        self.fish = fish
        conn = sqlite3.connect(self.dbDir, timeout=30.0)
        c = conn.cursor()  
        c.execute('''CREATE TABLE IF NOT EXISTS tblFish(simulation INTEGER, 
                                                           fish INTEGER,
                                                           length REAL)''')
        conn.commit()
        c.execute("INSERT INTO tblFish VALUES(%s,%s,%s);"%(simulation,fish,self.length))
        conn.commit()
        c.close()
    
    def survive(self):
        '''we apply the survival method at a node, therefore survival is a function 
        of location.  If survival is determined a priori we search the database
        for this node's survival probability.  If the fish is at a turbine, 
        survival is a function of the turbine, its operations, and the length of
        the fish.'''
        
        # first get the survival function type
        conn = sqlite3.connect(self.dbDir, timeout=30.0)
        c = conn.cursor()  
        surv_fun = pd.read_sql("SELECT * FROM tblNodes WHERE location == '%s'"%(self.location), con = conn).surv_fun.values[0]
        c.close()
            
        # if survival function is a priori
        if surv_fun == 'a priori':
            # get the a priori survival rate from the table in sqlite
            conn = sqlite3.connect(self.dbDir, timeout=30.0)
            c = conn.cursor()  
            prob = pd.read_sql("SELECT * FROM tblNodes WHERE location == '%s'"%(self.location), con = conn).prob.values[0]
            c.close()
            
        # if survival is assessed at a Kaplan turbine:
        elif surv_fun == 'Kaplan':
            # get turbine parameters
            conn = sqlite3.connect(self.dbDir, timeout=30.0)
            c = conn.cursor()  
            params = pd.read_sql("SELECT * FROM tblKaplan WHERE unit == '%s'"%(self.location), con = conn)
            c.close()
            # convert dataframe to dictionary by index - pass index [0]
            param_dict = pd.DataFrame.to_dict(params,'index')
            
            # calculate the probability of strike as a function of the length of the fish and turbine parameters
            prob = Kaplan(self.length, param_dict[0])[0]
        
        # if survival is assessed at a Propeller turbine:
        elif surv_fun == 'Propeller':
            # get turbine parameters
            conn = sqlite3.connect(self.dbDir, timeout=30.0)
            c = conn.cursor()  
            params = pd.read_sql("SELECT * FROM tblPropeller WHERE unit == '%s'"%(self.location), con = conn)
            c.close()
            # convert dataframe to dictionary by index - pass index [0]
            param_dict = pd.DataFrame.to_dict(params,'index')
            
            # calculate the probability of strike as a function of the length of the fish and turbine parameters
            prob = Propeller(self.length, param_dict[0])[0]
            
        # if survival is assessed at a Francis turbine:
        elif surv_fun == 'Francis':
            # get turbine parameters
            conn = sqlite3.connect(self.dbDir, timeout=30.0)
            c = conn.cursor()  
            params = pd.read_sql("SELECT * FROM tblFrancis WHERE unit == '%s'"%(self.location), con = conn)
            c.close()
            # convert dataframe to dictionary by index - pass index [0]
            param_dict = pd.DataFrame.to_dict(params,'index')
            
            # calculate the probability of strike as a function of the length of the fish and turbine parameters
            prob = Francis(self.length, param_dict[0])[0]

        # if survival is assessed at a turbine in pump mode:
        elif surv_fun == 'Pump':
            # get turbine parameters
            conn = sqlite3.connect(self.dbDir, timeout=30.0)
            c = conn.cursor()  
            params = pd.read_sql("SELECT * FROM tblPump WHERE unit == '%s'"%(self.location), con = conn)
            c.close()
            # convert dataframe to dictionary by index - pass index [0]
            param_dict = pd.DataFrame.to_dict(params,'index')
            
            # calculate the probability of strike as a function of the length of the fish and turbine parameters
            prob = Pump(self.length, param_dict[0])
            
        print ("Fish is at %s, the probability of surviving is %s"%(self.location, prob))
        
        # roll the dice of death - very dungeons and dragons of us . . . 
        dice = np.random.uniform(0.00,1.00,1)
        print ("Random draw: %s"%(dice))
        '''apply death logic:
        if our dice roll is greater than the probability of surviving, fish has died'''
        
        if dice > prob:
            print ("Fish has been killed <X>>>><")
            self.status = 0
            self.complete = 1
            conn = sqlite3.connect(self.dbDir, timeout=30.0)
            c = conn.cursor()  
            c.execute('''CREATE TABLE IF NOT EXISTS tblCompletion(simulation INTEGER, 
                                                               fish INTEGER,
                                                               status INTEGER,
                                                               completion INTEGER)''')
            conn.commit()
            c.execute("INSERT INTO tblCompletion VALUES(%s,%s,%s,%s);"%(self.simulation,self.fish,self.status,self.complete))
            conn.commit()
            c.close()
        else:
            print ("Fish has survived <0>>>><")
            
        # write results to project database
        conn = sqlite3.connect(self.dbDir, timeout=30.0)
        c = conn.cursor()  
        c.execute('''CREATE TABLE IF NOT EXISTS tblSurvive(simulation INTEGER, 
                                                           fish INTEGER,
                                                           location TEXT,
                                                           prob_surv REAL,
                                                           dice REAL,
                                                           status INTEGER)''')
        conn.commit()
        c.execute("INSERT INTO tblSurvive VALUES(%s,%s,'%s',%s,%s,%s);"%(self.simulation,self.fish,self.location,prob,dice[0],self.status))
        conn.commit()
        c.close()
            
    def move(self):
        '''we move between nodes after applying the survival function.  movement 
        is a random choice between available nodes and edge weight'''
        
        if self.status == 1:
            # get neighbors
            neighbors = self.route[self.location]

            if len(neighbors) > 0:
                u_prob = 0
                l_prob = 0
                move_prob_dict = {}
                
                ''' we need to apportion our movement probabilities by edge weights
                iterate through neighbors and assign them ranges 0 - 1.0 based on 
                movement weights - (]'''

                for i in neighbors:
                    u_prob = self.route[self.location][i]['weight'] + l_prob
                    move_prob_dict[i] = (l_prob,u_prob)
                    print ("If roll of dice is between %s and %s, fish will move to the %s"%(l_prob,u_prob,i))
                    l_prob = u_prob
                del i
                
                # role the dice of movement (arguably, this is not as catchy)
                dice = np.random.uniform(0.00,1.00,1)
                print ("Random draw: %s"%(dice))
                for i in move_prob_dict:
                    # if the dice role is between movement thresholds for the this neighbor...
                    if dice >= move_prob_dict[i][0] and dice < move_prob_dict[i][1]:
                        self.location = i
                        print ("Fish moved to %s"%(i))                      
            else:
                print ("Fish survived passage through project <0>>>><")
                self.complete = 1
                conn = sqlite3.connect(self.dbDir, timeout=30.0)
                c = conn.cursor()  
                c.execute('''CREATE TABLE IF NOT EXISTS tblCompletion(simulation INTEGER, 
                                                                   fish INTEGER,
                                                                   status INTEGER,
                                                                   completion INTEGER)''')
                conn.commit()
                c.execute("INSERT INTO tblCompletion VALUES(%s,%s,%s,%s);"%(self.simulation,self.fish,self.status,self.complete))
                conn.commit()
                c.close()                
            
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
        

    
    
    

