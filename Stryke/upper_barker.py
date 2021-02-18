# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:40:39 2020

@author: Kevin Nebiolo

Create an Upper Barker Model to test out Stryke
"""
# Import Dependencies
import stryke
import os
import numpy as np
import sqlite3
import pandas as pd
# set up workspaces
proj_dir = r"C:\Users\Kevin Nebiolo\Desktop\fuck"
dbName = "upper_barker15.db"
dbDir = os.path.join(proj_dir,'Data',dbName)

# create project
upper_barker = stryke.create_proj_db(proj_dir,dbName)
conn = sqlite3.connect(dbDir, timeout=30.0)
c = conn.cursor()

# IPD: Kevin - insert excel reading here
c.execute("INSERT INTO tblNodes VALUES ('tailrace','a priori')")
c.execute("INSERT INTO tblNodes VALUES ('spill','a priori')")
c.execute("INSERT INTO tblNodes VALUES ('bypass','a priori')")
c.execute("INSERT INTO tblNodes VALUES ('unit 1','Kaplan')")
c.execute("INSERT INTO tblNodes VALUES ('forebay','a priori')")

c.execute("INSERT INTO tblEdges VALUES ('spill','tailrace')")
c.execute("INSERT INTO tblEdges VALUES ('bypass','tailrace')")
c.execute("INSERT INTO tblEdges VALUES ('unit 1','tailrace')")
c.execute("INSERT INTO tblEdges VALUES ('forebay','bypass')")
c.execute("INSERT INTO tblEdges VALUES ('forebay','spill')")
c.execute("INSERT INTO tblEdges VALUES ('forebay','unit 1')")

c.execute("INSERT INTO tblKaplan VALUES ('unit 1',22,190,6.70,960,0.93,4,624,0.1)")

conn.commit()
c.close()

# create routing network and append to project database
route = stryke.create_route(dbDir)

# iterate through 10 simulations
for i in np.arange(0,10,1):
    # iterate over 100 fish
    for j in np.arange(0,50,1):
        # create a fish object, supply it with a species, mean, standard deviation, migration route, and database directory
        fish = stryke.fish('shad',(10.5/12,(0.5/12)), route, dbDir,i,j)
        # while fish is alive and it hasn't completed migrating through project
        while fish.status == 1 and fish.complete == 0:
            # assess survival at this node and write to database
            fish.survive()
            # move to the next node- do we care enough about this data to log it?
            fish.move()
        print ("Fish %s simulation %s complete"%(j,i))
# summarize
stryke.summary(dbDir)

