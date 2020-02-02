# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:40:39 2020

@author: Kevin Nebiolo

Create an Upper Barker Model to test out Stryke
"""
# Import Dependencies
import stryke
import os

# set up workspaces
proj_dir = r"C:\Users\Kevin Nebiolo\Desktop\StrykeTest"
dbName = "upper_barker.db"
dbDir = os.path.join(proj_dir,'Data',dbName)
# create project
#upper_barker = stryke.create_proj_db(proj_dir,dbName)

# create routing network and append to project database
route = stryke.create_route(dbDir)

# create a fish object, supply it with a species, log normal (mean, standard deviation) tuple, migration route, and database directory
fish = stryke.fish('shad',(1.,2.5), route, dbDir)

# while fish is alive and it hasn't completed migrating through project 
while fish.status == 1 and fish.complete == 0:
    # assess survival at this node
    fish.survive()
    # move to the next node
    fish.move()