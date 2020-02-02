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
    c.execute('''DROP TABLE IF EXISTS tblNodes''')                             # placeholder for route - will hold a networkx graph object
    c.execute('''DROP TABLE IF EXISTS tblEdges''')
    c.execute('''CREATE TABLE tblNodes(location TEXT PRIMARY KEY, 
                                       surv_fun TEXT CHECK(surv_fun = "a priori" OR 
                                                           surv_fun = "Kaplan" OR 
                                                           surv_fun = "Francis" OR 
                                                           surv_fun = "Propeller"), 
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
    dist = {}
    for i in edges.iterrows():
        _from = i[1]['_from']
        _to = i[1]['_to']
        weight = i[1]['weight']
        route.add_edge(_from,_to,weight = weight)
        dist[_from] = {_to:5}
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
    rR = np.random.uniform(0.3,1.0,1) # where on the blade did the fish strike? - see Deng for more info
    #Q_per = param_dict['Q_per']
    ada = param_dict['ada']
    N = param_dict['N']
    #Qopt = param_dict['Qopt']
    _lambda = 0.2 # use USFWS value of 0.2
    
    # part 1 - calculate omega, the energy coefficient, discharge coefficient
    omega = RPM * ((2 * np.pi)/60)
    Ewd = (g * H) / (omega * D)**2
    Qwd = Q/ (omega * D)**3
    
    # part 2 - calculate angle of absolute flow to the axis of rotation
    a_a = np.arctan((np.pi * ada * Ewd)/(2 * Qwd * rR))
    
    # probability of strike * length of fish
    p_strike = _lambda * (N / (D * 12)) * (np.radians(a_a)/(8 * Qwd) + np.radians(a_a)/(np.pi * rR))
    
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
    Qopt = param_dict['Qopt']
    _lambda = 0.2 # use USFWS value of 0.2
    
    # part 1 - calculate omega, the energy coefficient, discharge coefficient
    omega = RPM * ((2 * np.pi)/60) # radians per second
    Ewd = (g * H) / (omega * D)**2
    Qwd = Q/ (omega * D)**3
    
    # part 2 - calculate angle of absolute flow to the axis of rotation
    beta = np.arctan((np.pi/8 * rR)/(Qwd * Q_per))
    a_a = np.arctan((np.pi/2 * Ewd * ada)/(Qwd * rR) + (np.pi/8 * rR)/Qwd - beta)
       
    # probability of strike * length of fish
    p_strike = _lambda * (N / (D * 12)) * (np.cos(a_a)/(8 * Qwd)) + np.sin(a_a)/(np.pi * rR)
    
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
    _lambda = 0.2 # use USFWS value of 0.2

    # part 1 - calculate omega, the energy coefficient, discharge coefficient
    omega = RPM * ((2 * np.pi)/60) # radians per second
    Ewd = (g * H) / (omega * D)**2
    Qwd = Q/ (omega * D)**3
    
    # part 2 - calculate alpha and beta
    beta = np.arctan((0.707 * np.pi/8)/(iota * Qwd * Q_per * np.power(D1/D2,3)))
    alpha = np.radians(90) - np.arctan((2 * np.pi * Ewd * ada)/Qwd * (B/D1) + (np.pi * 0.707**2)/(2 * Qwd) * (B/D1) * (np.power(D2/D1,2)) - 4 * 0.707 * beta * (B/D1) * (D1/D2))

    # probability of strike * length of fish
    p_strike = _lambda * (N / D) * (((np.sin(alpha) * (B/D1))/(2*Qwd)) + (np.cos(alpha)/np.pi))
    
    return 1 - (p_strike * length)

    
class fish():
    '''python class object describing an individual fish in our individual based
    model.  The fish has two primary functions (methods); they can survive and
    they can move.  If they are dead, they can no longer move.  
    
    The simulation ends for an individual fish when there are no more moves to 
    make or its dead.'''
    
    def __init__(self,species,len_params,route,dbDir):
        self.species = species
        self.length = np.random.lognormal(len_params[0],len_params[1])
        self.route = route
        self.status = 1   # fish are alive at start
        self.complete = 0 # fish do not start the simulation in a completed step
        self.location = 'forebay'
        self.dbDir = dbDir
    
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
            prob = Kaplan(self.length, param_dict[0])
        
        print ("Fish is at %s, the probability of surviving is %s"%(self.location, prob))
        # roll the dice of death - very dungeons and dragons of us . . . 
        dice = np.random.uniform(0.00,1.00,1)
        print ("Random draw: %s"%(dice))
        '''apply death logic:
        if our dice role is greater than the probability of surviving, fish has died'''
        
        if dice > prob:
            print ("Fish has been killed <X>>>><")
            self.status = 0
        else:
            print ("Fish has survived <0>>>><")
            
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
            

