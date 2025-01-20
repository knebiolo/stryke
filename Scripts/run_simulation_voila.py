# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:40:39 2020

@author: Kevin Nebiolo

Create an Upper Barker Model to test out Stryke
"""

# Import Dependencies
import os
import sys

# Dynamic inputs from the Jupyter Notebook
stryke_path = sys.argv[1]  # Path to the stryke repository
excel_dir = sys.argv[2]    # Directory containing the Excel file
excel_file = sys.argv[3]   # Excel file name

# Add the stryke repository to sys.path
sys.path.append(stryke_path)

from Stryke import stryke

# Construct full path to the Excel file
ws = excel_dir
wks = excel_file
file_output = os.path.splitext(wks)[0]

# Run the simulation
simulation = stryke.simulation(ws,wks, output_name = file_output)
simulation.run()
simulation.summary()
