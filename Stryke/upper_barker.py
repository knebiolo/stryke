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
proj_dir = r"J:\705\092\Calcs\Python"
dbName = "upper_barker.db"
dbDir = os.path.join(proj_dir,'Data',dbName)
# create project
#upper_barker = stryke.create_proj_db(proj_dir,dbName)

# create routing network and append to project database
route = stryke.create_route(dbDir)
