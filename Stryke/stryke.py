# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:59:19 2020

@author: Kevin Nebiolo
@qaqc: Ish Deo

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
    c.execute('''CREATE TABLE tblFrancis(unit TEXT, H REAL, omega INTEGER, D REAL,
                                         Q REAL, Q_per REAL, ada REAL, N INTEGER,
                                         iota REAL, D1 REAL, D2 REAL, B REAL)''')
    c.execute('''CREATE TABLE tblKaplan(unit TEXT, H REAL, omega INTEGER, D REAL,
                                        Q REAL, ada REAL, N INTEGER, Qopt REAL)''')
    c.execute('''CREATE TABLE tblPropeller(unit TEXT, H REAL, omega INTEGER, 
                                           D REAL, Q REAL, Q_per REAL, ada REAL,
                                           N INTEGER, Qopt REAL)''')
    c.execute('''CREATE TABLE tblNodes(location TEXT, surv_fun TEXT, prob REAL)''')
    c.execute('''CREATE TABLE tblEdges(edge_id INTEGER, _from TEXT, _to TEXT, 
                                       weight REAL)''')

    
    conn.commit()   
    c.close()

# create function that builds networkx graph object from nodes and edges in project database
def create_route(dbDir):
    '''function creates a networkx graph object from information provided by 
    nodes and edges found in standard project database.  the only input
    is the standard project database.'''
    # get nodes and edges - this assumes the end user has updated the database
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
    weights = []
    # create edges - iterate over edge rows to create edges
    for i in edges.iterrows():
        _from = i[1]['_from']
        _to = i[1]['_to']
        weight = i[1]['weight']
        route.add_edge(_from,_to,weight = weight)
        weights.append(weight)
    # return the finished product and enjoy functionality of networkx
    labels = {}
    idx = 0
    for i in nodes.location.values:
        labels[idx] = i
        idx = idx + 1

    pos = nx.layout.spring_layout(route)
    plt.subplot(111)
    nx.draw_networkx_nodes(route, pos, node_size = 10, node_color = 'blue')
    nx.draw_networkx_labels(route, pos, font_size = 8)
    nx.draw_networkx_edges(route, pos, node_size = 10, width = weights * 1000)
    plt.show()
    return route

#def Kaplan:
    
#def Francis:
    
#def Propeller:

    
        
    

    
# create functions for Kaplan, Propeller & Francis units

