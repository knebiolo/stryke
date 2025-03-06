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

Unfortunately Franke et al.  units are imperial and functions have not been 
tested in metric.  When units are indicated to be in metric on the input spreadsheet,
stryke converts units from SI into imperial for blade strike survival estimate and
the entrainment rate estimate which are in units of fish per million cubic feet.

"""


# import dependencies
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')

from matplotlib import rcParams
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import beta
import xlrd
import networkx as nx
from Stryke.hydrofunctions import hydrofunctions as hf
#import hydrofunctions as hf
import requests
#import geopandas as gp
import statsmodels.api as sm
import math
from scipy.stats import pareto, genextreme, genpareto, lognorm, weibull_min, gumbel_r, ks_2samp, nbinom, norm
import h5py
#import tables
from numpy.random import default_rng
rng = default_rng()

# Now pass the session to hydrofunctions if possible


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

def enable_matplotlib_inline():
    try:
        from IPython import get_ipython  # Import the function to get the current IPython instance
        ipython_instance = get_ipython()  # Get the current IPython instance
        if ipython_instance is not None:  # Check if we're running in an IPython environment
            ipython_instance.run_line_magic('matplotlib', 'inline')  # Execute the magic command
            print("Enabled inline matplotlib plotting.")
    except ImportError:
        print("IPython is not available; inline plotting cannot be enabled.")

def hurdle_bootstrap_daily(daily_counts, n_days=365, rng=None):
    """
    Single-year bootstrap:
      - Probability of zero = fraction of zero in daily_counts
      - If nonzero, pick a positive count from the empirical distribution
    Returns array of length n_days with simulated daily entrainment.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Identify zero vs. positive
    zero_mask = (daily_counts == 0)
    p_zero = zero_mask.mean()
    pos_counts = daily_counts[~zero_mask]  # only the positives

    # If no positives, everything is zero
    if len(pos_counts) == 0:
        return np.zeros(n_days, dtype=int)

    # Draw random uniform for each day
    draws = rng.random(n_days)
    sim = np.zeros(n_days, dtype=int)

    # For days that are "nonzero", pick from pos_counts (with replacement)
    is_pos = (draws > p_zero)
    idx = rng.integers(0, len(pos_counts), size=n_days)
    sim[is_pos] = pos_counts[idx[is_pos]]
    return sim

class simulation():
    ''' Python class object that initiates, runs, and holds data for a facility
    specific simulation'''
    def __init__ (self, proj_dir, wks, output_name, existing = False):
        """
        Initializes a simulation class object to model individual-based entrainment 
        impacts at hydroelectric projects.
        
        This method sets up the simulation environment by creating necessary directories, 
        importing data from spreadsheets, creating data structures for simulation 
        parameters, and setting up an HDF5 file for output. If the 'existing' flag 
        is set to False, the method performs a full setup including directory 
        creation, data importation, and HDF5 file preparation.  For existing 
        simulations, it re-imports essential data and recreates the movement graph 
        based on node information.
        
        Parameters:
        - proj_dir (str): The root directory for the project where the workspace 
        and output files will be stored.
        - wks (str): The name of the workspace directory within the project directory 
        where input data spreadsheets are located.
        - output_name (str): The name of the output file (without extension) where 
        the simulation results will be stored.
        - existing (bool, optional): Flag to indicate whether this is a new simulation 
        setup (False) or an existing one (True).
          Default is False.
        
        Key Actions:
        - Workspace directory creation (for new simulations).
        - Data importation from Excel spreadsheets including nodes, survival 
        functions, unit parameters, flow scenarios,
          operating scenarios, and population data.
        - Creation of survival function dictionary from node data.
        - Calculation of the maximum river node to define the spatial boundary 
        of the simulation.
        - Hydraulic capacity calculation based on unit parameters.
        - Setup of unique flow and operating scenarios for the simulation.
        - HDF5 file creation for output, with storage of setup data for reproducibility 
        and analysis.
        - For existing simulations, re-imports node data and recreates the movement 
        graph to simulate fish movement through the system.
        
        The method leverages pandas for data manipulation and storage, numpy for 
        array operations, and networkx for graph-based movement simulations, 
        ensuring efficient and scalable data handling.
        """
        if existing == False:
            # create workspace directory
            self.wks_dir = os.path.join(proj_dir,wks)
            
            # extract scenarios from input spreadsheet
            #self.routing = pd.read_excel(self.wks_dir,'Routing',header = 0,index_col = None, usecols = "B:G", skiprows = 9)

            # import nodes and create a survival function dictionary
            self.nodes = pd.read_excel(self.wks_dir,
                                       sheet_name = 'Nodes',
                                       header = 0,
                                       index_col = None, 
                                       usecols = "B:D", 
                                       skiprows = 9)
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
            self.max_river_node = max_river_node

            # import unit parameters
            self.unit_params = pd.read_excel(self.wks_dir,
                                             sheet_name = 'Unit Params', 
                                             header = 0, 
                                             index_col = None,
                                             usecols = "B:T",
                                             skiprows = 4)
            self.unit_params.set_index('Unit',inplace = True)
            
            self.facility_params = pd.read_excel(self.wks_dir,
                                             sheet_name = 'Facilities', 
                                             header = 0, 
                                             index_col = None,
                                             usecols = "B:I",
                                             skiprows = 3)
            self.facility_params.set_index('Facility', inplace = True)
            # self.facility_params['Min_Op_Flow'] = self.facility_params.Min_Op_Flow.fillna(0)
            # self.facility_params['Env_Flow'] = self.facility_params.Env_Flow.fillna(0)  
            # self.facility_params['Bypass_Flow'] = self.facility_params.Bypass_Flow.fillna(0)

            # get hydraulic capacity of facility
            self.flow_cap = self.unit_params.groupby('Facility')['Qcap'].sum()

            # identify unique flow scenarios
            self.flow_scenarios_df = pd.read_excel(self.wks_dir,
                                                   sheet_name = 'Flow Scenarios',
                                                   header = 0,
                                                   index_col = None, 
                                                   usecols = "B:I", 
                                                   skiprows = 5, 
                                                   dtype = {'Gage':str})
            
            self.input_hydrograph_df = pd.read_excel(self.wks_dir,
                                                   sheet_name = 'Hydrology',
                                                   header = 0,
                                                   index_col = None, 
                                                   usecols = "B:C", 
                                                   skiprows = 3, 
                                                   dtype = {'Date':str,'Discharge':float})
            
            # fetch units: if metric, convert
            df = pd.read_excel(self.wks_dir, sheet_name='Background and Metadata')
            self.output_units  = df.iat[13, 1]
            
            # if units are metric, convert to imperial for entrainment rates and Franke equations
            if self.output_units == 'metric':#  make sure everything is converted from m3/s and meters to feet:
                self.input_hydrograph_df["Discharge"] = self.input_hydrograph_df["Discharge"] * 35.31469989
                self.unit_params['intake_vel'] = self.unit_params.intake_vel * 3.28084
                self.unit_params['H'] = self.unit_params.H * 3.28084
                self.unit_params['D'] = self.unit_params.D * 3.28084
                self.unit_params['Qopt'] = self.unit_params.Qopt * 35.31469989
                self.unit_params['Qcap'] = self.unit_params.Qcap * 35.31469989
                self.unit_params['B'] = self.unit_params.B * 3.28084
                self.unit_params['D1'] = self.unit_params.D1 * 3.28084
                self.unit_params['D2'] = self.unit_params.D2 * 3.28084
                self.facility_params['Exc_Rack_Spacing'] = self.facility_params.Exc_Rack_Spacing*0.0328084
            else:
                self.facility_params['Exc_Rack_Spacing'] = self.facility_params.Exc_Rack_Spacing/12.


            self.operating_scenarios_df = pd.read_excel(self.wks_dir,
                                                        sheet_name = 'Operating Scenarios', 
                                                        header = 0, 
                                                        index_col = None,
                                                        usecols = "B:I",
                                                        skiprows = 8)
            
            self.operating_scenarios_df['Scenario Name'] = self.operating_scenarios_df.Scenario + " " + self.operating_scenarios_df.Unit
            self.ops_scens = None
            
            self.flow_scenarios = self.flow_scenarios_df['Scenario'].unique()
            self.op_scenarios = self.operating_scenarios_df['Scenario Name'].unique()

            # import population data
            self.pop = pd.read_excel(self.wks_dir,
                                     sheet_name = 'Population',
                                     header = 0,
                                     index_col = None,
                                     usecols = "B:S", 
                                     skiprows = 11)
                        
            # create output HDF file
            self.proj_dir = proj_dir
            self.output_name = output_name

            # create hdf object with Pandas
            self.hdf = pd.HDFStore(os.path.join(self.proj_dir,'%s.h5'%(self.output_name)))

            # write study set up data to hdf store
            self.hdf['Flow Scenarios'] = self.flow_scenarios_df
            self.hdf['Operating Scenarios'] = self.operating_scenarios_df
            self.hdf['Population'] = self.pop
            self.hdf['Nodes'] = self.nodes
            self.hdf['Edges'] = pd.read_excel(self.wks_dir,
                                              sheet_name = 'Edges',
                                              header = 0,
                                              index_col = None, 
                                              usecols = "B:D", 
                                              skiprows = 8)
            
            self.hdf['Unit_Parameters'] = self.unit_params
            #self.hdf['Routing'] = self.routing
            self.hdf.flush()
            
        else:
            self.wks_dir = os.path.join(proj_dir,wks)

            # create output HDF file
            self.proj_dir = proj_dir
            self.output_name = output_name

            # import nodes and create a survival function dictionary
            self.nodes = pd.read_excel(self.wks_dir,
                                       sheet_name = 'Nodes',
                                       header = 0,
                                       index_col = None, 
                                       usecols = "B:D", 
                                       skiprows = 9)

            # get last river node
            max_river_node = 0
            for i in self.nodes.Location.values:
                i_split = i.split("_")
                if len(i_split) > 1:
                    river_node = int(i_split[2])
                    if river_node > max_river_node:
                        max_river_node = river_node

            # make a movement graph from input spreadsheet
            self.graph = self.create_route(self.wks_dir)

            # identify the number of moves that a fish can make
            path_list = nx.all_shortest_paths(self.graph,'river_node_0','river_node_%s'%(max_river_node))

            max_len = 0
            for i in path_list:
                path_len = len(i)
                if path_len > max_len:
                    max_len = path_len
            self.moves = np.arange(0,max_len-1,1)
            
    def Kaplan(self,length, param_dict):
        
        """
        Calculates the probability of a fish surviving a blade strike in a Kaplan 
        turbine based on the model proposed by Franke et al. 1997.
    
        The function takes into account various turbine parameters and the length 
        of the fish to compute the survival probability.  It employs a blend of 
        deterministic and stochastic elements to model the interaction between 
        fish and turbine blades, particularly considering the location of the 
        strike along the blade.
    
        Parameters:
        - length (float): The length of the fish in meters.
        - param_dict (dict): A dictionary containing key turbine parameters 
        necessary for the calculation. Expected keys and their
          meanings are as follows:
            - 'H': The net head of water across the turbine (ft).
            - 'RPM': The revolutions per minute of the turbine.
            - 'D': The diameter of the turbine runner (ft).
            - 'Q': The flow rate through the turbine (ft^3/s).
            - 'ada': Turbine efficiency.
            - 'N': The number of blades on the turbine.
            - '_lambda': The empirically derived constant for blade strike probability, 0.2 as 
            suggested by the U.S. Fish and Wildlife Service.
    
        The function first computes the energy coefficient (Ewd) and discharge 
        coefficient (Qwd) of the turbine, followed by the angle of absolute flow 
        to the axis of rotation. Utilizing these parameters, it estimates the 
        probability of a fish striking a blade and subsequently computes the 
        survival probability by subtracting the strike probability from one.
    
        Returns:
        - The probability of a fish surviving a blade strike in a Kaplan turbine, 
        ranging from 0 (no survival) to 1 (certain survival).
    
        Note: The function assumes the position of the fish strike along the blade 
        (rR) is uniformly distributed between 0.3 and 1.0, based on recommendations 
        from Deng et al. (please refer to the specific study for more details). 
    
        Example:
            param_dict = {'H': 30, 'RPM': 100, 'D': 5, 'Q': 200, 'ada': 0.8, 'N': 4, '_lambda': 0.2}
            fish_length = 0.5  # 50 cm
            survival_probability = Kaplan(fish_length, param_dict)
            print(f'Survival Probability: {survival_probability}')
        """
    
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
        p_strike = _lambda * (N * length / D) * ((np.cos(a_a)/(8 * Qwd)) +\
                                                 np.sin(a_a)/(np.pi * rR)) # IPD: conversion to radians is redundant and incorrect ~ corrected 2/5/20
        # need to take cosine and sine of angle alpha a (a_a)
    
        return 1 - (p_strike)
    
    def Propeller(self,length, param_dict):
        """
        Estimates the survival probability of a fish passing through a propeller 
        turbine, adapting the blade strike model from Franke et al. 1997. This 
        function considers the physical characteristics of the fish and turbine 
        operational parameters to calculate the likelihood of fish survival after 
        potential blade strikes.
    
        The calculation assumes a fixed position on the blade where the fish 
        strike might occur and incorporates both deterministic and empirical 
        parameters to model the interaction dynamics.
    
        Parameters:
        - length (float): The length of the fish in meters, which is a critical 
        factor in determining the probability of a blade strike.
        - param_dict (dict): A dictionary containing essential turbine parameters 
        for the calculation. Expected keys include:
            - 'H': The hydraulic head (ft).
            - 'RPM': The revolutions per minute of the turbine.
            - 'D': The diameter of the turbine runner (ft).
            - 'Q': The actual flow rate through the turbine (ft^3/s).
            - 'ada': Turbine efficiency.
            - 'N': The number of blades on the turbine.
            - 'Qopt': The optimum flow rate through the turbine (ft^3/s), used in
            beta angle calculations.
            - 'Qper': The percentage of the optimum flow rate, contributing to
            the beta calculation.
            - '_lambda': An empirical constant for blade strike probability, 0.2 as suggested 
            by the U.S. Fish and Wildlife Service.
    
        The function computes the energy coefficient (Ewd) and discharge coefficient (Qwd)
        to determine the turbine's operational state, followed by the calculation 
        of the absolute flow angle to the rotation axis (a_a). The strike probability 
        is then estimated using the fish length, turbine diameter, and the 
        calculated flow angle, with the survival probability being the complement of
        the strike probability.
    
        Returns:
        - The probability of a fish surviving a blade strike in a propeller turbine, 
        with values ranging from 0 (no survival) to 1 (certain survival).
    
        Note: The function assumes a fixed radial position (rR = 0.75) for the 
        fish strike on the blade, simplifying the model while maintaining relevance 
        to typical turbine conditions. This decision is based on empirical observations
        and simplifies the model's complexity.
    
        Example:
            param_dict = {'H': 25, 'RPM': 120, 'D': 3, 'Q': 180, 'ada': 0.7, 'N': 3, 
                          'Qopt': 160, 'Qper': 0.8, '_lambda': 0.2}
            fish_length = 0.4  # 40 cm
            survival_probability = Propeller(fish_length, param_dict)
            print(f'Survival Probability: {survival_probability}')
        """
        
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
    
        a_a = np.arctan((np.pi * Ewd * ada)/(2 * Qwd * rR) + (np.pi/8 * rR)/Qwd -\
                        np.tan(beta)) #IPD: should be tan(beta) ~ corrected 2/5/20
    
        # probability of strike * length of fish
        p_strike = _lambda * (N * length / D) * ((np.cos(a_a)/(8 * Qwd)) +\
                                                 np.sin(a_a)/(np.pi * rR))
    
        return 1 - (p_strike)
    
    def Francis(self,length, param_dict):
        """
        Calculates the survival probability of a fish passing through a Francis turbine,
        based on the blade strike model developed by Franke et al. 1997. This model 
        incorporates various turbine and biological parameters to estimate the likelihood 
        of a fish surviving after potential blade strikes within a Francis turbine.
    
        Parameters:
        - length (float): Length of the fish in feet. This parameter is crucial as it 
          directly influences the probability of blade strike.
        - param_dict (dict): A dictionary of turbine and operational parameters required 
          for the survival probability calculation. The expected keys and their descriptions 
          are as follows:
            - 'H': Net head of water across the turbine (in feet).
            - 'RPM': Rotational speed of the turbine runner (in revolutions per minute).
            - 'D': Diameter of the turbine runner (in feet).
            - 'Q': Volumetric flow rate through the turbine (in cubic feet per second).
            - 'Qper': Percentage of the optimum flow rate, contributing to the relative 
              flow angle calculation.
            - 'ada': Efficiency of the turbine, a factor in calculating the tangential 
              flow angle.
            - 'N': Number of blades on the turbine.
            - 'iota': Ratio between discharge with no exit swirl and optimum discharge, 
              used in calculating the relative flow angle (β).
            - 'D1': Diameter at the turbine entrance (in feet).
            - 'D2': Diameter at the turbine exit (in feet).
            - 'B': Width of a turbine blade (in feet).
            - '_lambda': Empirical constant for blade strike probability, recommended by 
              the U.S. Fish and Wildlife Service.
    
        The function first calculates the energy and discharge coefficients of the turbine, 
        followed by the relative flow angle (β) and the tangential flow angle upstream of 
        the runner (α). Utilizing these angles, it estimates the probability of a fish 
        striking a blade. The survival probability is then derived as the complement of 
        the strike probability.
    
        Returns:
        - float: The probability of a fish surviving a blade strike in a Francis turbine, 
          ranging from 0 (no survival) to 1 (certain survival).
    
        Example:
            param_dict = {'H': 100, 'RPM': 100, 'D': 10, 'Q': 500, 'Qper': 1.1, 'ada': 0.8,
                          'N': 4, 'iota': 1.1, 'D1': 5, 'D2': 10, 'B': 1, '_lambda': 0.2}
            fish_length = 1  # 1 foot
            survival_probability = Francis(fish_length, param_dict)
            print(f'Survival Probability: {survival_probability}')
        """
    
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
    
        # Calculations
        omega = RPM * ((2 * np.pi) / 60)  # Angular velocity (rad/s)
        Ewd = (g * H) / (omega * D)**2  # Energy coefficient
        Qwd = Q / (omega * D**3)  # Discharge coefficient
    
        # Relative flow angle (beta)
        beta = np.arctan((0.707 * (np.pi / 8)) / (iota * Q_per * (D1 / D2)**3))
    
        # Angle tangential of absolute flow (alpha)
        alpha = np.arctan((2 * np.pi * Ewd * ada) / Qwd * (B / D1) +
                          (np.pi * 0.707**2) / (2 * Qwd) * (B / D1) * (D2 / D1)**2 -
                          4 * 0.707 * np.tan(beta) * (B / D1) * (D1 / D2))
    
        # Probability of mortality from blade strike (M^d)
        p_strike = _lambda * (N * length / D) * ((np.sin(alpha) * (B / D1)) / (2 * Qwd) + np.cos(alpha) / np.pi)
    
        return 1 - p_strike  # Survival probability
    
    def Pump(self,length, param_dict):
        r''' pump mode calculations from fish entrainment analysis report:
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
    
    def node_surv_rate(self,length,status,surv_fun,route,surv_dict,u_param_dict):
        """
        Calculates the survival probability of a fish passing through a node in 
        the migratory network, taking into account the type of hydraulic structure 
        encountered (e.g., Kaplan, Propeller, Francis turbines, or pump mode operation) 
        and the fish's length. The method supports both predefined survival probabilities 
        (a priori values) and dynamic calculations based on turbine parameters.
        
        Parameters:
        - length (float): The length of the fish, which is a critical factor in 
        calculating the blade strike probability.
        - status (int): The status of the fish, where 0 indicates a condition 
        (such as mortality) that precludes further survival calculation.
        - surv_fun (str): The survival function or method to be used. This can be
        'a priori' for predefined survival probabilities, or the name of a specific
        turbine type ('Kaplan', 'Propeller', 'Francis') for dynamic survival 
        probability calculation. 'Pump' mode is also supported for scenarios 
        involving fish entrainment during pumping operations.
        - route (str): The migratory route or node identifier. In the case of 
        'a priori' survival probabilities, this corresponds to the key in the 
        'surv_dict' dictionary. For dynamic calculations, it specifies the turbine 
        for which parameters are provided in 'u_param_dict'.
        - surv_dict (dict): A dictionary containing predefined survival probabilities 
        for various routes or nodes, used when 'surv_fun' is set to 'a priori'. 
        The keys correspond to route identifiers, and the values are the survival 
        probabilities.
        - u_param_dict (dict): A dictionary of dictionaries containing turbine 
        parameters necessary for calculating survival probabilities. Each key
        corresponds to a route or turbine identifier, with the associated value 
        being another dictionary of parameters relevant to the specific turbine 
        type indicated by 'surv_fun'.
        
        Returns:
        - float: The calculated survival probability for the fish at the given 
        node, converted to a 32-bit float. If the 'status' parameter is 0, 
        indicating a non-survivable condition, the method immediately returns 0.0.
        
        This method integrates various survival probability models, including 
        those based on empirical data ('a priori') and dynamic models for
        different turbine types, providing a versatile tool for assessing fish
        survival in hydroelectric project simulations.
        """
    
        if status == 0:
            return 0.0
        else:
            if surv_fun == 'a priori':
                try:
                    prob = surv_dict[route]
                except:
                    print ('Problem with a priori survival function')
    
            else:
                #TODO add impingement logic, bring 
                
                #TODO add in barotrauma logic
                
                param_dict = u_param_dict[route]
                # if survival is assessed at a Kaplan turbine:
                if surv_fun == 'Kaplan':
                    # calculate the probability of strike as a function of the length of the fish and turbine parameters
                    prob = self.Kaplan(length, param_dict)
    
                # if survival is assessed at a Propeller turbine:
                elif surv_fun == 'Propeller':
                    # calculate the probability of strike as a function of the length of the fish and turbine parameters
                    prob = self.Propeller(length, param_dict)
    
                # if survival is assessed at a Francis turbine:
                elif surv_fun == 'Francis':
                    # calculate the probability of strike as a function of the length of the fish and turbine parameters
                    prob = self.Francis(length, param_dict)
    
                # if survival is assessed at a turbine in pump mode:
                elif surv_fun == 'Pump':
                    # calculate the probability of strike as a function of the length of the fish and turbine parameters
                    prob = self.Pump(length, param_dict)
            try:
                return np.float32(prob)
            except:
                print ('check')
    
    # create function that builds networkx graph object from nodes and edges in project database
    def create_route(self,wks_dir):
        """
        Constructs a directed graph representing the migratory network of fish, 
        using data from nodes and edges defined within a specified project database. 
        The graph is built using the NetworkX library, allowing for the utilization 
        of its extensive graph analysis functionalities.
    
        The method reads node and edge information from an Excel file located in the 
        provided directory, 'wks_dir'. Nodes represent points within the migratory 
        network (such as entry, exit, and decision points), while edges represent 
        the possible paths a fish can take between these nodes, along with associated 
        weights (e.g., probabilities or costs associated with each path).
    
        Parameters:
        - wks_dir (str): The directory path where the project database Excel file is 
          located. This file should contain 'Nodes' and 'Edges' sheets with the 
          necessary data to construct the network graph.
    
        The method performs the following steps:
        1. Reads the 'Nodes' and 'Edges' sheets from the Excel file, extracting relevant 
           data for graph construction.
        2. Initializes an empty directed graph (DiGraph) object using NetworkX.
        3. Adds nodes to the graph based on locations specified in the 'Nodes' data.
        4. Iterates over the 'Edges' data to add directed edges between nodes, assigning 
           weights to these edges as specified in the data.
        5. Calculates the maximum number of moves a fish can make within the network, 
           based on all shortest paths from a predefined start node ('river_node_0') 
           to the furthest node, identified by 'self.max_river_node'.
    
        The constructed graph and the maximum number of moves are stored in 'self.graph' 
        and 'self.moves', respectively, for use in subsequent analyses or simulations 
        within the broader context of the project.
    
        Note: This method assumes the presence of a start node labeled 'river_node_0' 
        and relies on 'self.max_river_node' being set prior to its call to determine 
        the endpoint for path calculations.
        """
    
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
        self.graph = route

        # identify the number of moves that a fish can make
        path_list = nx.all_shortest_paths(route,'river_node_0','river_node_%s'%(self.max_river_node))

        max_len = 0
        for i in path_list:
            path_len = len(i)
            if path_len > max_len:
                max_len = path_len
        self.moves = np.arange(0,max_len + 1,1)
    
    def movement (self,
                  location,
                  status, 
                  swim_speed, 
                  graph, 
                  intake_vel_dict,
                  Q_dict, 
                  op_order, 
                  cap_dict,
                  unit_fac_dict):
        """
        Simulates the movement of a fish through a hydroelectric project's
        infrastructure, considering operational conditions, the fish's swimming
        capabilities, and environmental requirements. It calculates the probability
        of the fish choosing various paths based on water discharge and unit status.
    
        Parameters:
        - location (str): Current location within the project infrastructure.
        - status (int): Survival status of the fish, 1 for alive and 0 for dead.
        - swim_speed (float): Swimming speed of the fish, for intake velocity resistance.
        - graph (networkx.Graph): Directed graph of the project, with nodes as locations
          and edges as paths between locations.
        - intake_vel_dict (dict): Maps each turbine to its intake velocity.
        - Q_dict (dict): Contains discharge information, including 'curr_Q' for current
          discharge, 'min_Q' for minimum operating discharge, 'env_Q' for environmental
          flow, 'sta_cap' for station capacity, and turbine capacities.
        - op_order (dict): Maps each turbine to its operational order, for sequence
          determination as discharge increases.
    
        Special logic is applied in the forebay area, considering the variety of paths
        (turbines vs. spillway) and operational conditions. Movement probabilities are
        calculated based on edge weights, apportioned by operational conditions and the
        fish's swimming abilities.
    
        Returns:
        - str: New location after movement simulation, which could be a specific turbine,
          'spill' for the spillway, or the same location if the fish cannot move.
    
        This method provides a detailed simulation of fish movement, incorporating
        environmental flows, operational priorities, and biological capabilities to
        inform management decisions and impact assessments within hydroelectric projects.
        """
        curr_Q = Q_dict['curr_Q']   # current discharge
        min_Q_dict = Q_dict['min_Q']     # minimum operating discharge
        sta_cap_dict = Q_dict['sta_cap'] # station capacity
        env_Q_dict = Q_dict['env_Q']     # min environmental discharge 
        bypass_Q_dict = Q_dict['bypass_Q'] # how much discharge through downstream bypass sluice
        
        # if the fish is alive
        if status == 1:
            # get neighbors
            nbors = list(graph.neighbors(location))
            #neighbors = list(neighbors)
                          
            locs = []
            probs = []
            
            if len(nbors) > 1:
                # Check for "spill" in any part of the string or if it starts with an uppercase 'U'
                found_spill = np.char.find(nbors, "spill") >= 0
                starts_with_U = np.char.startswith(nbors, "U")
                
                
                '''check to see if any neighbor starts with U - if it does, figure 
                out that facility it belongs to and get the station capacity''' 
                if np.any(starts_with_U):
                    for i in nbors:
                        # it's a unit
                        try:
                            facility = unit_fac_dict[i]

                        # nope it's a bypass
                        except:
                            for j in nbors:
                                if 'U' in j:
                                    facility = self.unit_params.at[j,'Facility']
                                    continue
                        sta_cap = sta_cap_dict[facility]
                        min_Q =  min_Q_dict[facility]
                        env_Q = env_Q_dict[facility]
                        bypass_Q = bypass_Q_dict[facility]
                                
                        # When current discharge greater than the min operating flow but less than station capacity...
                        if min_Q < curr_Q < sta_cap + bypass_Q:
                            # get flow remaining for production
                            excess = curr_Q - (sta_cap + env_Q + bypass_Q)
                            if excess >= 0:
                                prod_Q = curr_Q - env_Q - bypass_Q - excess
                            else:
                                prod_Q = curr_Q - env_Q - bypass_Q
                            
                            if i[0] == 'U':
                                unit_cap = Q_dict[i]
                                order = op_order[i]
                                
                                # list units that turn on before this one
                                prev_units = []
                                for u in op_order:
                                    fac = unit_fac_dict[u]
                                    if fac == facility:
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
                                    
                                    if prev_Q >= prod_Q:
                                        u_Q = 0.0
                                    else:
                                        u_Q = prod_Q - prev_Q
                                        if u_Q > unit_cap:
                                            u_Q = unit_cap
                                
                                # write data to arrays
                                locs.append(i)
        
                                probs.append(u_Q/(prod_Q  + bypass_Q))
                                del u_Q, prev_units
                            else:
                                locs.append(i)
                                probs.append(bypass_Q / (prod_Q + bypass_Q))
                        
                        # when the current discharge is larger than the station capacity and bypass flows
                        elif curr_Q >= sta_cap + bypass_Q:
                            excess = curr_Q - (sta_cap + bypass_Q + env_Q) 
                            if i[0] == 'U':
                                q_cap = cap_dict[i]
                                locs.append(i)
                                probs.append(q_cap / (sta_cap + bypass_Q))
                            else:
                                locs.append(i)
                                probs.append(bypass_Q / (sta_cap + bypass_Q))
        
                elif np.any(found_spill):
                    # first characterize discharge at this spillway - how much flow
                    # is provided for production and what is excess?
                    # get facility associated with this spillway
                    for i in nbors:
                        if 'spill' in i:
                            facilities = self.facility_params[self.facility_params.Spillway == i].index
                    del i

                    # how much discharge is accounted for at facilities associated with this spill?
                    sta_cap = 0 
                    min_Q = 0 
                    env_Q = 0
                    bypass_Q = 0
                    
                    # sum up all non spill flows associated with this spillway
                    for i in facilities:
                        sta_cap_fac = sta_cap_dict[i]
                        min_Q_fac =  min_Q_dict[i]
                        env_Q_fac = env_Q_dict[i]
                        bypass_Q_fac = bypass_Q_dict[i]
                        sta_cap = sta_cap + sta_cap_fac
                        min_Q = min_Q + min_Q_fac
                        env_Q = env_Q + env_Q
                        bypass_Q = bypass_Q + bypass_Q_fac
                        
                    # when current discharge less than the min operating flow - everything is spilled:
                    for i in nbors:

                        if curr_Q <= min_Q:
                            if 'spill' in i:
                                locs.append(i)  
                                probs.append(1.)  
                            else:
                                locs.append(i)
                                probs.append(0.)
                                
                        # when current discharge is greater than hydraulic capacity - excess water above the enviromental discharge are routed through spill:
                        elif curr_Q >= np.sum(list(sta_cap_dict.values())) + env_Q + bypass_Q:
                            #TODO 
                            ''' Is this logic correct, do we need to know what facility 
                            they are interacting with?  should there be a facility to spill
                            relationship?'''
                            excess = curr_Q - np.sum(list(sta_cap_dict.values())) - bypass_Q - env_Q
                            tot_spill = excess + env_Q
                            p_spill = tot_spill / curr_Q
                            if 'spill' in i:
                                locs.append(i)  
                                probs.append(p_spill)  
                            else:
                                locs.append(i)
                                probs.append(1 - p_spill)
                                
                        # when the current discharge is between min flows and station capacity - the spillway gets what it gets            
                        elif curr_Q < np.sum(list(sta_cap_dict.values())) + env_Q + bypass_Q:
                            p_env = env_Q / curr_Q
                            if 'spill' in i:
                                locs.append(i)  
                                probs.append(p_env)
                            else:
                                locs.append(i)
                                probs.append(1 - p_env)
                else:
                    # the weights you provided in the setup sheet determine probability of movement
                    for i in nbors:
                        locs.append(i)
                        edge = location, i
                        edge_weight = graph[location][i]["weight"]
                        probs.append(edge_weight)
                            
            # if there is only 1 neighbor there is only one place to go       
            elif len(nbors) == 1:
                locs.append(nbors[0])
                probs.append(1)                
            
            # if there aren't any neighbors - we have reached the end
            else:
                locs.append(location)
                probs.append(1)
    
            # generate a new location
            #probs = np.array(probs) / np.sum(np.array(probs))
            try:
                new_loc = np.random.choice(locs,1,p = probs)[0]
            except:
                print ('Problem with movement function')

            del nbors, locs, probs
            
            # # filter out those fish that can escape intake velocity
            # if np.sum(swim_speed) > 0:
            #     if 'U' in new_loc:
            #         if swim_speed > intake_vel_dict[new_loc]:
            #             new_loc = 'spill'
       
        # if the fish is dead, it can't move
        else:
            new_loc = location
            
        return new_loc

    def speed (L,A,M):
        """
        Calculates swimming speed based on fish length, caudal fin aspect ratio,
        and swimming mode, according to Sambilay 1990. The function is vectorizable,
        allowing for efficient calculations over arrays of inputs.
    
        Parameters:
        - L (float or array-like): Fish length in centimeters.
        - A (float or array-like): Caudal fin aspect ratio (dimensionless), obtained
          from fish morphological data.
        - M (int or array-like): Swimming mode, where 0 represents sustained swimming
          and 1 represents burst swimming.
    
        Returns:
        - float or array-like: Swimming speed in kilometers per hour. The conversion
          factor used (0.911344) translates the speed to feet per second for
          compatibility with certain applications.
    
        This model provides an estimate of swimming speeds for fish based on physical
        characteristics and behavior, useful in ecological and engineering studies
        related to aquatic locomotion and habitat interactions.
        """
    
        sa = 10**(-0.828 + 0.6196 * np.log10(L*30.48) + 0.3478 * np.log10(A) + 0.7621 * M)
        
        return sa * 0.911344
    
    def get_USGS_hydrograph(self, gage, prorate, flow_year): 
        """
        Retrieves and standardizes a hydrograph from the USGS for a specified flow year,
        and adjusts the flow based on watershed size. The function fetches gage data
        online, extracts relevant flow data, and applies a proration factor to account
        for differences in watershed size.
    
        Parameters:
        - gage (str): The USGS gage identifier for which the hydrograph is requested.
        - prorate (float): Factor by which to adjust the flow data, typically based on
          the relative size of the watershed of interest compared to that of the gage.
        - flow_year (int): The year for which flow data is to be retrieved and processed.
    
        Returns:
        - DataFrame: A pandas DataFrame containing the daily average flow data for the
          specified flow year, adjusted by the proration factor. The DataFrame includes
          columns for date, original daily average flow ('DAvgFlow'), prorated daily
          average flow ('DAvgFlow_prorate'), year, and month.
    
        Note: The function assumes the availability of USGS gage data online and may
        require internet access to fetch the data. The use of 'verify=False' in the
        requests.get call may lead to security warnings and should be used with caution.
        """
        # get gage data object from web
        start_date = '%s-01-01'%(flow_year)
        end_date = '%s-12-31'%(flow_year)
        
        gage_dat = hf.station.NWIS(site = gage, service='dv', start_date= start_date, end_date = end_date)

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
            
        return df
    
    def create_hydrograph(self, discharge_type, scen, scen_months, flow_scenarios_df, fixed_discharge = None):
        """
        Generates a hydrograph for simulation based on specified scenarios, either by
        importing and adjusting actual hydrograph data or by simulating a hydrograph
        with a fixed discharge rate across specified months.
    
        Parameters:
        - discharge_type (str): Specifies the type of discharge data to use; can be
          'hydrograph' for actual hydrograph data or 'fixed' for a constant discharge rate.
        - scen (str): The scenario identifier to filter the relevant data in the flow
          scenarios dataframe.
        - scen_months (list): List of integers representing the months to include in the
          hydrograph simulation.
        - flow_scenarios_df (DataFrame): A dataframe containing flow scenario data,
          including gage information, proration factors, and flow years for 'hydrograph'
          type scenarios.
        - fixed_discharge (float, optional): The fixed discharge rate (in cubic feet per
          second) to use for 'fixed' discharge type simulations.
    
        Returns:
        - DataFrame: A pandas DataFrame representing the hydrograph for the specified
          scenario, containing columns for dates, prorated daily average flow, and month.
    
        The function handles two main types of simulations: importing and adjusting USGS
        hydrograph data based on a proration factor or simulating a hydrograph with a
        constant discharge rate across the specified months.
        """
        flow_df = pd.DataFrame()
        
        scen_df = flow_scenarios_df[flow_scenarios_df.Scenario == scen]
        
        # if the discharge type is hydrograph - import hydrography and transform using prorate factor
        if discharge_type == 'hydrograph':
            gage = str(scen_df.at[scen_df.index[0],'Gage'])
            prorate = scen_df.at[scen_df.index[0],'Prorate']
            flow_year = scen_df.at[scen_df.index[0],'FlowYear']
            
            # if a gage number is present, fetch the usgs gage data
            if sum(char.isdigit() for char in gage) > 1:
                
                df = self.get_USGS_hydrograph(gage, prorate, flow_year)
                for i in scen_months:
                    flow_df = pd.concat([flow_df, df[df.month == i]])
            
            # if not, use the hydrograph data in the Hydrology sheet
            else: 
                flow_df = self.input_hydrograph_df
                # apply prorate
                flow_df['DAvgFlow_prorate'] = flow_df.Discharge * prorate
                # convert to datetime
                flow_df['datetimeUTC'] = pd.to_datetime(flow_df.Date)
                # extract year
                flow_df['year'] = pd.DatetimeIndex(flow_df['datetimeUTC']).year
                flow_df = flow_df[flow_df.year == flow_year]
                # get months
                flow_df['month'] = pd.DatetimeIndex(flow_df['datetimeUTC']).month
        
        # if it is a fixed discharge - simulate a hydrograph
        elif discharge_type == 'fixed':
            day_in_month_dict = {1:31,2:28,3:31,
                                 4:30,5:31,6:30,
                                 7:31,8:31,9:30,
                                 10:31,11:30,12:31}
            sim_hydro_dict = {}
            
            # for every month 
            for month in scen_months:
                days = day_in_month_dict[month]
                for day in np.arange(1,days+1,1):
                    date = "2023-" + str(month) + "-" + str(day)
                    sim_hydro_dict[date] = fixed_discharge
                            
                df = pd.DataFrame.from_dict(sim_hydro_dict,orient = 'index')  
                df.reset_index(inplace = True, drop = False)
                df.rename(columns = {'index':'datetimeUTC',0:'DAvgFlow_prorate'},inplace = True) 
                df['month'] = pd.to_datetime(df.datetimeUTC).dt.month
                if np.any(df.DAvgFlow_prorate.values < 0):
                    print ('prorated daily average flow value not found')
                #flow_df = flow_df.append(df)
                flow_df = pd.concat([df, flow_df])
        
        return flow_df
    
    def daily_hours(self, Q_dict, scenario, operations = 'independent'):
        """
        Simulates the daily operational hours of units in hydroelectric facilities,
        considering the facility type (Run-Of-River or peaking) and operational
        dependencies between units. The function adjusts flow rates based on operational
        hours and unit capacities.
    
        Parameters:
        - ops_df (DataFrame): Contains unit operation parameters, including potential
          log-normal distribution parameters for operation hours and probability of
          not operating.
        - q_cap_dict (dict): Maps each unit to its flow capacity (cubic feet per second).
        - op_order_dict (dict): Maps each unit to its operational order, important for
          facilities with dependent unit operations.
        - operations (str, optional): Defines the operational mode of the units; can be
          'independent' for units operating independently or 'dependent' for units with
          sequential dependencies. Defaults to 'independent'.
    
        Returns:
        - tuple: Contains total operational hours, total flow, a dictionary of operational
          hours per unit, and a dictionary of flow per unit.
    
        The function supports various operational scenarios, including fixed hours,
        hours determined by a log-normal distribution, or dependent operations where
        the operation of one unit depends on another. For Run-Of-River facilities,
        units operate 24/7, while peaking facilities may vary.
        """

        ops_df = self.operating_scenarios_df[self.operating_scenarios_df.Scenario == scenario]
        ops_df.set_index('Unit', inplace = True)
        facilities = ops_df.Facility.unique()
        
        seasonal_facs = self.facility_params[self.facility_params.Scenario == scenario]
        #seasonal_facs.set_index('Facility', inplace = True)
        # loop over units, build some dictionaries
        prev_unit_hours = None
        
        hours_dict = {}
        hours_operated = {}
        flow_dict = {}
        
        ''' this is incorrect, it does not account for current discharge 
        and whether or not the facility is capable of operation all units'''

        
        cum_Q = 0. # current amount of discharge passing through powerhouse
        # for each unit either simulate hours operated or write hours to dictionary
        for facility in facilities:
            curr_Q = Q_dict['curr_Q']   # current discharge
            min_Q = Q_dict['min_Q'][facility]     # minimum operating discharge
            sta_cap = Q_dict['sta_cap'][facility] # station capacity
            env_Q = Q_dict['env_Q'][facility]     # min environmental discharge 
            bypass_Q = Q_dict['bypass_Q'][facility] # how much discharge through downstream bypass sluice
            prod_Q = curr_Q - env_Q - bypass_Q # how much water is left to use for production

            fac_type = seasonal_facs.at[facility,'Operations']
            fac_units = self.unit_params[self.unit_params.Facility == facility]
            #fac_units.set_index('Unit', inplace = True)
            fac_units = fac_units.sort_values(by = 'op_order')
            
            # if operations are modeled with a distribution 
            if fac_type != 'run-of-river':
                for i in fac_units.index:
                    order = fac_units.at[i,'op_order']
                    # get log norm shape parameters
                    shape = ops_df.at[i,'shape']
                    location = ops_df.at[i,'location']
                    scale = ops_df.at[i,'scale']
                    
                    hours_operated[i] = lognorm.rvs(shape,location,scale,1000)
    
                    # flip a coin - see if this unit is running today
                    prob_not_operating = ops_df.at[i,'Prob_Not_Op']
                    
                    if operations == 'independent':
                        if np.random.uniform(0,1,1) <= prob_not_operating:
                            hours_dict[i] = 0.
                            flow_dict[i] = 0.
    
                        else:
                            # TODO Bad Creek Analysis halved hours - change back
                            hours = lognorm.rvs(shape,location,scale,1)[0] #* 0.412290503
    
                            if hours > 24.:
                                hours = 24.
                            elif hours < 0:
                                hours = 0.
                            hours_dict[i] = hours
                            flow_dict[i] = fac_units.at[i,'Qcap'] * hours * 3600.    
                            
                    elif operations == 'dependent':
                        # if this is the first unit to be operated
                        #TODO change this back to just 1 == 1 - updated for Bad Creek Analysis
                        if order == 1:# or i == 5:
                            if np.random.uniform(0,1,1) <= prob_not_operating:
                                hours_dict[i] = 0.
                                flow_dict[i] = 0.
        
                            else:
                                # TODO Bad Creek Analysis halved hours - change back
                                hours = lognorm.rvs(shape,location,scale,1)[0] #* 0.412290503
    
                                if hours > 24.:
                                    hours = 24.
                                elif hours < 0:
                                    hours = 0.
                                hours_dict[i] = hours
                                flow_dict[i] = fac_units.at[i,'Qcap'] * hours * 3600.
    
                        # if it is any other unit        
                        else:
                            prev_unit = fac_units[fac_units.op_order == order -1].index
                            prev_hours = hours_dict[prev_unit]
                            
                            # if the previous unit ran
                            if prev_hours > 0:
                                hours_remain = np.where(hours_operated[i] <= prev_hours, hours_operated[i], np.nan)
                                hours_remain = hours_remain[~np.isnan(hours_remain)]
                                if len(hours_remain) > 0:
                                    fit_to_remain = lognorm.fit(hours_remain)
                                    if np.random.uniform(0,1,1) <= prob_not_operating:
                                        hours_dict[i] = 0.
                                        flow_dict[i] = 0.
                                    else:
                                        # TODO Bad Creek Analysis halved hours - change back
                                        hours = lognorm.rvs(fit_to_remain[0],fit_to_remain[0],fit_to_remain[0],1)[0] #* 0.412290503
    
                                        if hours > 24.:
                                            hours = 24.
                                        elif hours < 0:
                                            hours = 0.
                                        hours_dict[i] = hours                        
                                        flow_dict[i] = fac_units.at[i,'Qcap'] * hours * 3600.
                                else:
                                    hours_dict[i] = 0.
                                    flow_dict[i] = 0.                            
                            else:
                                hours_dict[i] = 0.
                                flow_dict[i] = 0.
            # if it's run of river, units operate when there is water
            else:
                at_capacity = False

                for i in fac_units.index:
                    hours = ops_df.at[i, 'Hours']
                    u_cap = fac_units.at[i, 'Qcap']
                    # Determine the effective capacity limit
                    effective_cap = min(prod_Q, sta_cap)
                    
                    if cum_Q + u_cap <= effective_cap:
                        hours_dict[i] = hours
                        flow_dict[i] = u_cap * hours * 3600.
                        cum_Q = cum_Q + u_cap
                    else:
                        excess = cum_Q + u_cap - effective_cap
                        hours_dict[i] = hours
                        flow_dict[i] = (u_cap - excess) * hours * 3600
                        at_capacity = True
                    if at_capacity:  # Exit the for loop if we are at capacity
                            break

        # # implement Bad Creek algorithm here - is this method valid for new construction?
        # tot_hours = 0.
        # sum_pump_rate = 0.
        # bc1_sum_rate = 0.
        # for key in hours_operated:
        #     if key == 'U1' or key == 'U2' or key == 'U3' or key == 'U4':
        #         tot_hours = tot_hours + hours_operated[key]
        #         bc1_sum_rate = bc1_sum_rate + ops_df.at[unit,'Qcap']
        #     sum_pump_rate = sum_pump_rate + ops_df.at[unit,'Qcap']
                
        # volume = tot_hours * 3600. * bc1_sum_rate 
        # tot_bc1_bc2_time = volume / sum_pump_rate / 3600.
        # time_ratio = tot_bc1_bc2_time / tot_hours
        
        # for key in hours_operated:
        #     hours_operated[key] * time_ratio

        tot_flow = 0       
        tot_hours = 0   
                     
        for u in hours_dict.keys():
            tot_hours = tot_hours + hours_dict[u]
            tot_flow = tot_flow + flow_dict[u]
               
        ops_df.reset_index(drop = False, inplace = True)
            
        return tot_hours, tot_flow, hours_dict, flow_dict
        
    def population_sim(self, spc_df, curr_Q):
        """
        Simulates a population of fish based on species-specific parameters and
        operational conditions of a hydroelectric facility. The function uses
        distribution parameters to simulate entrainment rates, which are then
        adjusted based on operational conditions and historical data.
    
        Parameters:
        - spc_df (DataFrame): Contains species-specific parameters for simulating
          entrainment rates, including distribution type and parameters.
        - discharge_type (str): Type of discharge operation ('fixed' or variable).
        - tot_hours (float): Total operational hours of the facility.
        - tot_flow (float): Total flow through the facility in cubic feet.
        - curr_Q (float): Current discharge rate in cubic feet per second.
    
        Returns:
        - int: Simulated number of fish entrained, rounded to the nearest whole number.
    
        The simulation considers the discharge type, operational hours, and current
        flow to estimate the volume of water interacting with the fish population.
        The entrainment rate is drawn from the specified distribution and adjusted
        for feasibility based on historical data.
        """
        shape_col_num = spc_df.columns.get_loc('shape')
        loc_col_num = spc_df.columns.get_loc('location')
        scale_col_num = spc_df.columns.get_loc('scale')
        dist_col_num = spc_df.columns.get_loc('dist')
        
        shape = spc_df.iat[0,shape_col_num]
        loc = spc_df.iat[0,loc_col_num]
        scale = spc_df.iat[0,scale_col_num]
        dist = spc_df.iat[0,dist_col_num]

        if dist == 'Pareto':
            ent_rate = pareto.rvs(shape, loc, scale, 1, random_state=rng)
        elif dist == 'Extreme':
            ent_rate = genextreme.rvs(shape, loc, scale, 1, random_state=rng)
        elif dist == 'Log Normal':
            ent_rate = lognorm.rvs(shape, loc, scale, 1, random_state=rng)
        else:
            ent_rate = weibull_min.rvs(shape, loc, scale, 1, random_state=rng)

        ent_rate = np.abs(ent_rate)

        # apply order of magnitude filter, if entrainment rate is 1 order of magnitude larger than largest observed entrainment rate, reduce
        max_ent_rate = spc_df.max_ent_rate.values[0]

        if np.log10(ent_rate[0]) > np.log10(max_ent_rate):

            # how many orders of magnitude larger is the simulated entrainment rate than the largest entrainment rate on record?
            magnitudes = np.ceil(np.log10(ent_rate[0])) - np.ceil(np.log10(max_ent_rate)) + 0.5

            if magnitudes < 1.:
                magnitudes = 1.

            # reduce by at least 1 order of magnitude
            ent_rate = np.abs(ent_rate / 10**magnitudes)
            #print ("New entrainment rate of %s"%(round(ent_rate[0],4)))

        # flow per day in relation to million cubic FEET
        Mft3 = (60 * 60 * 24 * curr_Q)/1000000
        # flow per day in relation to million cubic METERS
        Mm3 = Mft3 * 35.31469989
        
        metric_units = ["cms","CMS"]
        if self.output_units == 'metric':
            daily_rate = Mm3
            ent_rate = ent_rate / 35.31469989
        else: 
            daily_rate = Mft3

        # calcualte sample size
        return np.round(daily_rate * ent_rate,0)[0]

    def run(self):
        """
        Executes a comprehensive simulation of fish populations navigating through
        a hydroelectric facility, accounting for various operational scenarios,
        species-specific behaviors, and environmental conditions. The function
        integrates multiple components including route creation, operational
        parameters setup, and survival probability calculations.
        
        The simulation workflow includes:
        1. Route creation for fish movement based on facility layout.
        2. Initialization of dictionaries for operational parameters, survival
           probabilities, intake velocities, unit capacities, and operational orders.
        3. Iteration over hydroelectric units to populate dictionaries with unit-
           specific operational data.
        4. Scenario-based simulation, considering different flow conditions and
           species presence.
        5. Population simulation for each species under each scenario, including
           entrainment rate calculations and survival assessments.
        6. Movement and survival simulation for individual fish, with results
           stored in a hierarchical data format (HDF) file for analysis.
        
        No parameters are passed directly to this function as it operates on
        the class attributes set during initialization and updated through
        other methods.
        
        The function logs progress and notable events (e.g., entrainment events,
        units not operating) throughout the simulation, providing insights into
        the simulation dynamics and outcomes.
        
        Outputs:
        The function generates and stores detailed simulation results in an HDF file,
        including daily summaries, entrainment statistics, and length distributions
        for simulated fish populations across different scenarios.
        """
        # create route and data object behind the scenes
        self.create_route(self.wks_dir)
        
        str_size = dict()
        str_size['species'] = 30
        for i in self.moves:
            str_size['state_%s'%(i)] = 30
            
        # create empty holders for some dictionaries
        u_param_dict = {}
        surv_dict = {}
        intake_vel_dict = {}
        units = []
        op_order_dict = {}
        q_cap_dict = {}
        unit_fac_dict = {}
        # for every unit, build a whole bunch of dictionaries to pass information        
        for index, row in self.unit_params.iterrows():
            unit = index
            unit_fac_dict[unit] = row['Facility']
            q_cap_dict[unit] = row['Qcap']

            runner_type = row['Runner Type']
            intake_vel_dict[unit] = row['intake_vel']
            units.append(unit)
            op_order_dict[unit] = row['op_order']
                
            # create parameter dictionary for every unit, a dictionary in a dictionary
            if runner_type == 'Kaplan':
                # built a parameter dictionary for the kaplan function
                param_dict = {'H':float(row['H']),
                              'RPM':float(row['RPM']),
                              'D':float(row['D']),
                              'ada':float(row['ada']),
                              'N':float(row['N']),
                              'Qopt':float(row['Qopt']),
                              '_lambda':float(row['lambda'])}
                u_param_dict[unit] = param_dict

            elif runner_type == 'Propeller':
                # built a parameter dictionary for the kaplan function
                param_dict = {'H':float(row['H']),
                              'RPM':float(row['RPM']),
                              'D':float(row['D']),
                              'ada':float(row['ada']),
                              'N':float(row['N']),
                              'Qopt':float(row['Qopt']),
                              'Qper':row['Qper'],
                              '_lambda':float(row['lambda'])}
                u_param_dict[unit] = param_dict                       
                
            elif runner_type == 'Francis':
                # built a parameter dictionary for the Francis function
                param_dict = {'H':float(row['H']),
                              'RPM':float(row['RPM']),
                              'D':float(row['D']),
                              'ada':float(row['ada']),
                              'N':float(row['N']),
                              'Qper':float (row['Qper']),
                              'iota' : float (row['iota']),
                              'D1' : float (row['D1']),
                              'D2' : float (row['D2']),
                              'B' : float (row['B']),
                              '_lambda':float(row['lambda'])}
                u_param_dict[unit] = param_dict

        # create survival dictionary, which is a dictionary of a priori surival rates
        for row in self.nodes.iterrows():
            if row[1]['Surv_Fun'] == 'a priori':
                surv_dict[row[1]['Location']] = row[1]['Survival']

        # for every scenario 
        for scen in self.flow_scenarios:
            
            # extract information about this scenario
            scen_df = self.flow_scenarios_df[self.flow_scenarios_df['Scenario'] == scen]
            
            scen_num = scen_df.iat[0,scen_df.columns.get_loc('Scenario Number')]
            season = scen_df.iat[0,scen_df.columns.get_loc('Season')]                                 
            scenario = scen_df.iat[0,scen_df.columns.get_loc('Scenario')]
            scen_months = scen_df.iat[0,scen_df.columns.get_loc('Months')]

            
            if scen_df.iat[0,scen_df.columns.get_loc('Flow')] == 'hydrograph':
                self.discharge_type = 'hydrograph'
            else:
                self.discharge_type = 'fixed'

            # convert scen months into a list of calendar months in this scenario 
            if type(scen_months) != np.int64:
                month_list = scen_months.split(",") 
                scen_months = list(map(int, month_list))
            else:
                scen_months = [scen_months]
            
            # get unit operations scenarios and extract data
            ops = self.operating_scenarios_df[self.operating_scenarios_df['Scenario'] == scenario]
            units = self.unit_params.index

            # identify the species we need to simulate for this scenario
            species = self.pop[self.pop['Scenario'] == scenario].Species.unique()
            
            # create a hydrograph for this scenario
            if self.discharge_type == 'hydrograph':
                flow_df = self.create_hydrograph(self.discharge_type,
                                                 scen, 
                                                 scen_months,
                                                 self.flow_scenarios_df)
            else:
                fixed_discharge = scen_df.iat[0,scen_df.columns.get_loc('Flow')]
                flow_df = self.create_hydrograph(self.discharge_type,
                                                 scen, 
                                                 scen_months, 
                                                 self.flow_scenarios_df, 
                                                 fixed_discharge = fixed_discharge)
                
            
            # for each species, perform the simulation for n individuals x times
            for spc in species:
                # extract a single row based on season and species
                spc_dat = self.pop[(self.pop['Scenario'] == scenario) & (self.pop.Species == spc)]

                # get scipy log normal distribution paramters - note values in centimeters
                s = spc_dat.iat[0,spc_dat.columns.get_loc('length shape')]
                len_loc = spc_dat.iat[0,spc_dat.columns.get_loc('length location')]
                len_scale = spc_dat.iat[0,spc_dat.columns.get_loc('length scale')]

                # get a priori length 
                mean_len = spc_dat.iat[0,spc_dat.columns.get_loc('Length_mean')]
                sd_len = spc_dat.iat[0,spc_dat.columns.get_loc('Length_sd')]

                # get species name
                species = spc_dat.iat[0,spc_dat.columns.get_loc('Species')]

                # get the number of times we are going to iterate this thing
                iterations = spc_dat.iat[0,spc_dat.columns.get_loc('Iterations')]
                
                # get probability of occurence
                if math.isnan(spc_dat.iat[0,spc_dat.columns.get_loc('occur_prob')]):
                    occur_prob = 1.0
                else:
                    occur_prob = spc_dat.iat[0,spc_dat.columns.get_loc('occur_prob')]

                # create an empty dataframe to hold length 
                spc_length = pd.DataFrame()

                # create an iterator
                for i in np.arange(0,iterations,1):
                    # for every row in the discharge dataframe, simulate an entrainment event and assess survival
                    for flow_row in flow_df.iterrows():
                        # get current discharge and day
                        curr_Q = flow_row[1]['DAvgFlow_prorate']
                        day = flow_row[1]['datetimeUTC']
                        
                        # create a Q dictionary - which is used for the movement function
                        Q_dict = {}
                        Q_dict['curr_Q'] = curr_Q
                        min_Q_dict = {}
                        env_Q_dict = {}
                        bypass_Q_dict = {}
                        for index, row in self.facility_params[self.facility_params.Scenario == scenario].iterrows():
                            min_Q = row['Min_Op_Flow']
                            env_Q = row['Env_Flow']
                            bypass_Q = row['Bypass_Flow']
                            fac = index
                            min_Q_dict[fac] = min_Q
                            env_Q_dict[fac] = env_Q
                            bypass_Q_dict[fac] = bypass_Q
                        Q_dict['min_Q'] = min_Q_dict
                        Q_dict['env_Q'] = env_Q_dict
                        Q_dict['bypass_Q'] = bypass_Q_dict

                        # for unit in units, add curr_Q to u_param_dict, add each unit capacity to Q_dict
                        sta_cap = {}
                        for u in units:
                            u_param_dict[u]['Q'] = curr_Q
                            unit_df = self.unit_params.loc[[u]]
                            fac = unit_df.iat[0,unit_df.columns.get_loc('Facility')]
                            if fac not in sta_cap:
                                sta_cap[fac] = 0
                            Q_dict[u] = unit_df.iat[0,unit_df.columns.get_loc('Qcap')]
                            sta_cap[fac] = sta_cap[fac] + unit_df.iat[0,unit_df.columns.get_loc('Qcap')]
                            
                        Q_dict['sta_cap'] = sta_cap

                        # Are units running today? if they are - test for occurence
                        tot_hours, tot_flow, hours_dict, flow_dict = self.daily_hours(Q_dict, scenario)
                        
                        if np.any(tot_hours > 0):
                            '''we need to roll the dice here and determine whether or not fish are present at site'''
                            presence_seed = np.random.uniform(0,1)
                            
                            if occur_prob >= presence_seed:
    
                                # if we are passing a population
                                if math.isnan(spc_dat.iat[0,spc_dat.columns.get_loc('shape')]):
                                    n = np.int(spc_dat.iat[0,spc_dat.columns.get_loc('Fish')])
        
                                # simulate a populution
                                else:
                                    n = self.population_sim(spc_dat,curr_Q)
                                
                                if np.int32(n) == 0:
                                    n = 1
                                    
                                #print ("Resulting in an entrainment event of %s %s"%(np.int32(n),spc))
                                
                                if math.isnan(s) == False:
                                    # create population of fish - IN CM!!!!!
                                    population = np.abs(lognorm.rvs(s, len_loc, len_scale, np.int32(n), random_state=rng))
                                    population = np.where(population > 150,150,population)
                                    # if self.output_units == 'metric':
                                    #     # convert cm to m
                                    #     population = population /100.
                                    # else:
                                    #     # convert lengths in cm to feet
                                    #     population = population * 0.0328084
                                        
                                    population = population * 0.0328084
                                else:
                                    population = np.abs(np.random.normal(mean_len, sd_len, np.int32(n)))/12.0
    
                                # calculate sustained swim speed (ft/s)
                                if math.isnan(spc_dat.U_crit.values[0]) == False:
                                    swim_speed = np.repeat(spc_dat.iat[0,spc_dat.columns.get_loc('U_crit')],len(population))
                                else:
                                    swim_speed = np.repeat(spc_dat.iat[0,spc_dat.columns.get_loc('U_crit')],len(population))
    
    
                                #print ("created population for %s iteration:%s day: %s"%(species,i,day))
                                # create a dataframe that tracks each fish
                                fishes = pd.DataFrame({'scenario_num':np.repeat(scen_num,np.int32(n)),
                                                          'species':np.repeat(species,np.int32(n)),
                                                          'flow_scenario':np.repeat(scenario,np.int32(n)),
                                                          'season':np.repeat(season,np.int32(n)),
                                                          'iteration':np.repeat(i,np.int32(n)),
                                                          'day':np.repeat(day,np.int32(n)),
                                                          'flow':np.repeat(curr_Q,np.int32(n)),
                                                          'population':np.float32(population),
                                                          'state_0':np.repeat('river_node_0',np.int32(n))})
                                
    
                                for k in self.moves:
                                    if k == 0:
                                        # initial status
                                        status = np.repeat(1,np.int32(n))
                                    else:
                                        status = fishes['survival_%s'%(k-1)].values
    
                                    # initial location
                                    location = fishes['state_%s'%(k)].values
    
                                    def surv_fun_att(state,surv_fun_dict):
                                        fun_typ = surv_fun_dict[state]['Surv_Fun']
                                        return fun_typ
    
                                    v_surv_fun = np.vectorize(surv_fun_att,excluded = [1])
                                    try:
                                        surv_fun = v_surv_fun(location,self.surv_fun_dict)
                                    except:
                                        print ('problem with survival function')
    
                                    # simulate survival draws
                                    dice = np.random.uniform(0.,1.,np.int32(n))
    
                                    # vectorize STRYKE survival function
                                    v_surv_rate = np.vectorize(self.node_surv_rate, excluded = [4,5])
                                    rates = v_surv_rate(population,status,surv_fun,location,surv_dict,u_param_dict)
    
                                    # calculate survival
                                    survival = np.where(dice <= rates,1,0)
    
                                    # simulate movement
                                    if k < max(self.moves):
                                        # vectorize movement function
                                        v_movement = np.vectorize(self.movement,excluded = [3,4,5,6,7,8])
                                        
                                        # have fish move to the next node
                                        move = v_movement(location, 
                                                          survival,
                                                          swim_speed,
                                                          self.graph,
                                                          intake_vel_dict,
                                                          Q_dict,
                                                          op_order_dict,
                                                          q_cap_dict,
                                                          unit_fac_dict)
    
                                    # add onto iteration dataframe, attach columns
                                    fishes['draw_%s'%(k)] = np.float32(dice)
                                    fishes['rates_%s'%(k)] = np.float32(rates)
                                    fishes['survival_%s'%(k)] = np.float32(survival)
    
                                    if k < max(self.moves):
                                        fishes['state_%s'%(k+1)] = move
    
                                # save that data
                                max_string_lengths = fishes.select_dtypes(include=['object']).apply(lambda x: x.str.len().max())


                                fishes.to_hdf(self.hdf,
                                              key=f'simulations/{scen}/{spc}',
                                              mode='a',
                                              format='table',
                                              append=True,
                                              min_itemsize=20)  # Replace 'column_name' with your actual column name
                                self.hdf.flush()                              
                                
                                # if needed, convert flow back to metric units
                                if self.output_units == 'metric':
                                    curr_Q = curr_Q * 0.02831683199881
                                
                                # start filling in that summary dictionar
                                daily_row_dict = {'species':['{:50}'.format(spc)],
                                                  'scenario':['{:50}'.format(scenario)],
                                                  'season':['{:50}'.format(season)],
                                                  'iteration':[np.int64(i)],
                                                  'day': [pd.to_datetime(day)],
                                                  'flow':[np.float64(curr_Q)],
                                                  'pop_size':[np.int64(len(fishes))]}
                                
                                # calculate daily entrainment survival calcs
                                # Identify columns that begin with 'state_' and 'survival_'
                                state_columns = [col for col in fishes.columns if col.startswith('state_')]
                                survival_columns = [col for col in fishes.columns if col.startswith('survival_')]
                                
                                # Ensure they are in the correct order
                                state_columns.sort()
                                survival_columns.sort()
                                
                                # Convert state columns to a NumPy array of fixed-width Unicode strings.
                                state_vals = fishes[state_columns].to_numpy(dtype='U50')  # shape (n, m)
                                
                                # Create a boolean mask: True where the state starts with 'U'
                                mask = np.char.startswith(state_vals, 'U')
                                
                                # Identify rows where any state column indicates entrainment.
                                entrained = mask.any(axis=1)
                                
                                # For entrained rows, find the index of the first occurrence of a "U".
                                first_U_index = np.where(entrained, mask.argmax(axis=1), -1)
                                
                                # Convert the survival columns to a NumPy array.
                                survival_vals = fishes[survival_columns].values
                                
                                # Pre-allocate a boolean array for survival.
                                survived = np.zeros(len(fishes), dtype=bool)
                                
                                # For rows that are entrained, check the survival value corresponding to the first "U" state.
                                rows = np.arange(len(fishes))
                                survived[entrained] = (survival_vals[rows[entrained], first_U_index[entrained]] == 1)
                                
                                # Add the computed columns back to the DataFrame.
                                fishes['is_entrained'] = entrained
                                fishes['survived_entrainment'] = survived
                                
                                # Count totals.
                                total_entrained = entrained.sum()
                                total_survived_entrained = survived.sum()
                                daily_row_dict['num_entrained'] = total_entrained
                                daily_row_dict['num_survived'] = total_survived_entrained
                                
                                # extract population and iteration
                                # TODO - we are exporting survival 2 again and using it for powerhouse survival - need to code around this for more complex simulations
                                length_dat = fishes[['population','flow_scenario','season','iteration','day','state_2','survival_2']]
    
                                # append to species length dataframe
                                spc_length = pd.concat([spc_length,length_dat], ignore_index = True)
       
                            else:
                                n = 0
                                #print ("No fish of this species on %s"%(day))
                                
                                # if needed, convert back to metric units
                                metric_units = ["cms","CMS"]
                                if self.output_units in metric_units:
                                    curr_Q = curr_Q * 0.02831683199881
                                
                                daily_row_dict = {'species':['{:50}'.format(spc)],
                                                  'scenario':['{:50}'.format(scenario)],
                                                  'season':['{:50}'.format(season)],
                                                  'iteration':[np.int64(i)],
                                                  'day': [pd.to_datetime(day)],
                                                  'flow':[np.float64(curr_Q)],
                                                  'pop_size':[np.int64(0)],
                                                  'num_entrained':[np.int64(0)],
                                                  'num_survived':[np.int64(0)]}
    
     
                        else:
                            
                            n = 0
                            #print ("Units not operating on %s"%(day))
                            
                            # if needed, convert back to metric units
                            metric_units = ["cms","CMS"]
                            if self.output_units in metric_units:
                                curr_Q = curr_Q * 0.02831683199881
                            
                            daily_row_dict = {'species':['{:50}'.format(spc)],
                                              'scenario':['{:50}'.format(scenario)],
                                              'season':['{:50}'.format(season)],
                                              'iteration':[np.int64(i)],
                                              'day': [pd.to_datetime(day)],
                                              'flow':[np.float64(curr_Q)],
                                              'pop_size':[np.int64(0)],
                                              'num_entrained':[np.int64(0)],
                                              'num_survived':[np.int64(0)]}
                                
                        daily = pd.DataFrame.from_dict(daily_row_dict, orient = 'columns')
                        
                        daily.to_hdf(self.hdf,
                                     key = 'Daily',
                                     mode = 'a',
                                     format = 'table',
                                     append = True)
                            
                           
                        self.hdf.flush()
                    
                    print ("Scenario %s Iteration %s for Species %s complete\n"%(scenario,i,species), flush=True)

                        
                # TODO - more of that state 2 survival 2 nonesense
                spc_length.to_hdf(self.hdf,
                                  key = 'Length',
                                  mode = 'a', 
                                  format = 'table',
                                  min_itemsize = {'flow_scenario':50,
                                                   'season':50,
                                                   'day':50,
                                                   'state_2':50},
                                  append = True)
            self.hdf.flush()
            print ("Completed Scenario %s %s\n"%(species,scen), flush=True)                            
            
        print ("Completed Simulations - view results\n", flush=True)
        self.hdf.flush()
        self.hdf.close()

    def summary(self):
        """
        Summarizes the results of fish entrainment simulations stored in an HDF file.
        This function aggregates and analyzes data across species, scenarios, and units
        to provide insights into entrainment risks and outcomes.
    
        The summary process includes:
        1. Accessing the HDF store and retrieving relevant data tables (Population,
           Flow Scenarios, Unit Parameters, Daily summaries, and Length distributions).
        2. Iterating through species and scenarios to calculate survival rates, length
           statistics, and entrainment outcomes.
        3. Fitting beta distributions to survival probabilities and summarizing the
           results.
        4. Aggregating daily data to provide cumulative summaries, including median
           population sizes, entrainment and mortality rates, and confidence intervals.
        5. Analyzing entrainment and mortality distributions to calculate probabilities
           of exceeding certain thresholds (e.g., 10, 100, 1000 individuals).
    
        Outputs:
        - Detailed summaries are printed to the console, providing an overview of
          simulation outcomes.
        - Key summary statistics, including beta distribution parameters and cumulative
          sums, are stored as class attributes for further analysis or reporting.
        - The function updates the HDF file with aggregated summary data and closes
          the file upon completion.
    
        Note: This function assumes that the HDF file contains the necessary data
        tables from previous simulation runs and that the file structure adheres to
        the expected format.
        """
        # create hdf store
        self.hdf = pd.HDFStore(os.path.join(self.proj_dir,'%s.h5'%(self.output_name)))

        # create some empty holders
        self.beta_dict = {}

        # get Population table
        pop = self.hdf['Population']
        species = pop.Species.unique()

        # get Scenarios
        scen = self.hdf['Flow Scenarios']
        scens = scen.Scenario.unique()

        # get units
        units = self.hdf['Unit_Parameters'].index

        self.daily_summary = self.hdf['Daily']
        self.daily_summary.iloc[:,6:] = self.daily_summary.iloc[:,6:].astype(float)

        print ("iterate through species and scenarios and summarize")
        # now loop over dem species, scenarios, iterations and days then calculate them stats
        for i in species:
            # # create empty dataframe to hold all lengths for this particular species
            # spc_length = self.hdf['Length']
            
            # # calculate length stats for this species
            # self.length_summ = spc_length.groupby(['season','state_2','survival_2']).population.describe()
            # print ("summarized length by season, state, and survival")

            for j in scens:
                # get daily data for this species/scenario
                try:
                    dat = self.hdf['simulations/%s/%s'%(j,i)]
                except:
                    continue

                # summarize species-scenario - whole project
                whole_proj_succ = dat.groupby(by = ['iteration','day'])['survival_%s'%(max(self.moves))]\
                    .sum().\
                        to_frame().\
                            reset_index(drop = False).\
                                rename(columns = {'survival_%s'%(max(self.moves)):'successes'})
                whole_proj_count = dat.groupby(by = ['iteration','day'])['survival_%s'%(max(self.moves))]\
                    .count().\
                        to_frame().\
                            reset_index(drop = False).\
                                rename(columns = {'survival_%s'%(max(self.moves)):'count'})

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
        
        try:
            self.daily_summary['day'] = self.daily_summary['day'].dt.tz_localize(None)
        except:
            pass


        # Step 0) group your daily_summary by year, as before:
        yearly_summary = self.daily_summary.groupby(
            ['species','scenario','iteration']
        )[['pop_size','num_survived','num_entrained']].sum()
        yearly_summary.reset_index(inplace=True)
        
        cum_sum_dict = {
            'species': [],
            'scenario': [],
            'med_population': [],
            'med_entrained': [],
            'med_survived': [],
            'mean_ent': [],
            'lcl_ent': [],
            'ucl_ent': [],
            'prob_gt_10_entrained': [],
            'prob_gt_100_entrained': [],
            'prob_gt_1000_entrained': []
        }
        
        # For each species & scenario combination:
        for fishy in yearly_summary.species.unique():
            for scen in yearly_summary.scenario.unique():
                
                # A) Basic "observed" year-level sums (like you did before)
                idat = yearly_summary[
                    (yearly_summary.species == fishy) &
                    (yearly_summary.scenario == scen)
                ]
                
                # median population, median entrained, median survived
                cum_sum_dict['species'].append(fishy)
                cum_sum_dict['scenario'].append(scen)
                cum_sum_dict['med_population'].append(idat.pop_size.median())
                cum_sum_dict['med_entrained'].append(idat.num_entrained.median())
                cum_sum_dict['med_survived'].append(idat.num_survived.median())
        
                # B) We want daily data for this scenario & species
                day_dat = self.daily_summary[
                    (self.daily_summary.species == fishy) &
                    (self.daily_summary.scenario == scen)
                ]
        
                daily_counts = day_dat.num_entrained.values
                n_actual_days = len(daily_counts)
        
                # If we have no daily data, everything is 0
                if n_actual_days == 0:
                    # No daily data => all probabilities 0, no annual variation
                    cum_sum_dict['mean_ent'].append(0.)
                    cum_sum_dict['lcl_ent'].append(0.)
                    cum_sum_dict['ucl_ent'].append(0.)
                    cum_sum_dict['prob_gt_10_entrained'].append(0.)
                    cum_sum_dict['prob_gt_100_entrained'].append(0.)
                    cum_sum_dict['prob_gt_1000_entrained'].append(0.)
                    continue
        
                # ------------------------------------------------------------------
                # 1) Estimate "daily" thresholds via big single-sample:
                # ------------------------------------------------------------------
                # We'll do a large synthetic "super-year" to get daily threshold rates
                big_n_days = 100000
                sim_big = hurdle_bootstrap_daily(daily_counts, n_days=big_n_days)
                prob_gt_10 = np.mean(sim_big > 10)
                prob_gt_100 = np.mean(sim_big > 100)
                prob_gt_1000 = np.mean(sim_big > 1000)
        
                cum_sum_dict['prob_gt_10_entrained'].append(prob_gt_10)
                cum_sum_dict['prob_gt_100_entrained'].append(prob_gt_100)
                cum_sum_dict['prob_gt_1000_entrained'].append(prob_gt_1000)
        
                # ------------------------------------------------------------------
                # 2) Estimate "yearly" distribution of total
                # ------------------------------------------------------------------
                # We'll do repeated "year" bootstraps. You can either:
                #  a) Use the same # of days as your original data
                #  b) Force 365 days
                # for simplicity, let's assume "year" means 365
                n_years = 2000
                rng = np.random.default_rng(42)
                year_sums = []
                for _ in range(n_years):
                    sim_year = hurdle_bootstrap_daily(daily_counts, n_days=365, rng=rng)
                    year_sums.append(sim_year.sum())
        
                year_sums = np.array(year_sums)
                mean_ent = year_sums.mean()
                lcl = np.percentile(year_sums, 2.5)
                ucl = np.percentile(year_sums, 97.5)
        
                cum_sum_dict['mean_ent'].append(mean_ent)
                cum_sum_dict['lcl_ent'].append(lcl)
                cum_sum_dict['ucl_ent'].append(ucl)
        
        print("Yearly summary complete.")
        
        # Convert to a DataFrame
        self.cum_sum = pd.DataFrame.from_dict(cum_sum_dict)
        
        # # Suppose you already have:
        # #   self.daily_summary
        # # as in your original code, with columns: 
        # #   ['species','scenario','iteration','pop_size','num_entrained','num_survived', ...]
        
        # # First, group to get the sum by year/species/scenario/iteration:
        # yearly_summary = self.daily_summary.groupby(
        #     by=['species','scenario','iteration']
        # )[['pop_size','num_survived','num_entrained']].sum()
        # yearly_summary.reset_index(inplace=True)
        
        # cum_sum_dict = {
        #     'species': [],
        #     'scenario': [],
        #     'med_population': [],
        #     'med_entrained': [],
        #     'med_survived': [],
        #     'mean_ent': [],
        #     'lcl_ent': [],
        #     'ucl_ent': [],
        #     'prob_gt_10_entrained': [],
        #     'prob_gt_100_entrained': [],
        #     'prob_gt_1000_entrained': []
        # }
        
        # for fishy in yearly_summary.species.unique():
        #     for scen in yearly_summary.scenario.unique():
        #         # Pull the "yearly" data for this species/scenario
        #         idat = yearly_summary[
        #             (yearly_summary.species == fishy) &
        #             (yearly_summary.scenario == scen)
        #         ]
        
        #         # Summaries at the yearly level
        #         cum_sum_dict['species'].append(fishy)
        #         cum_sum_dict['scenario'].append(scen)
        #         cum_sum_dict['med_population'].append(idat.pop_size.median())
        #         cum_sum_dict['med_entrained'].append(idat.num_entrained.median())
        #         cum_sum_dict['med_survived'].append(idat.num_survived.median())
        
        #         # ------------------------------------------------------------
        #         # 1) Hurdle approach for DAILY entrained counts
        #         # ------------------------------------------------------------
        #         day_dat = self.daily_summary[
        #             (self.daily_summary.species == fishy) &
        #             (self.daily_summary.scenario == scen)
        #         ]
        #         daily_counts = day_dat.num_entrained.values
        
        #         if len(daily_counts) == 0:
        #             # No daily data at all => fallback
        #             prob_gt_10 = 0.0
        #             prob_gt_100 = 0.0
        #             prob_gt_1000 = 0.0
        #         else:
        #             # Probability of a zero day
        #             num_zero = np.sum(daily_counts == 0)
        #             total_days = len(daily_counts)
        #             p_zero = num_zero / total_days
        #             p_nonzero = 1.0 - p_zero
        
        #             # Positive counts only
        #             pos_counts = daily_counts[daily_counts > 0]
        
        #             if len(pos_counts) < 2:
        #                 # If there's almost no positive data, can't fit a distribution
        #                 # fallback or set safe defaults
        #                 prob_gt_10 = 0.0 if len(pos_counts) == 0 else float(pos_counts.max() > 10)
        #                 prob_gt_100 = 0.0
        #                 prob_gt_1000 = 0.0
        #             else:
        #                 # Negative Binomial, method-of-moments
        #                 #   mean = mu
        #                 #   var = mu + alpha * mu^2 => alpha = (var - mu)/mu^2
        #                 mu_pos = np.mean(pos_counts)
        #                 var_pos = np.var(pos_counts, ddof=1)
        #                 if var_pos <= mu_pos:
        #                     # If variance <= mean, fallback to Poisson or clamp alpha=0
        #                     alpha = 0.0
        #                 else:
        #                     alpha = (var_pos - mu_pos) / (mu_pos**2)
        
        #                 # NB2 parameterization: r = 1/alpha, p = r/(r+mu)
        #                 # (If alpha=0 => pure Poisson)
        #                 if alpha < 1e-9:
        #                     # effectively Poisson
        #                     r = 1e9
        #                     p = mu_pos / (mu_pos + r)
        #                 else:
        #                     r = 1.0 / alpha
        #                     p = r / (r + mu_pos)
        
        #                 # Probability that daily count is > X
        #                 #   = p_zero*(0) + p_nonzero * nbinom.sf(X, r, p)
        #                 prob_gt_10 = p_zero + p_nonzero*nbinom.sf(10,  r, p)
        #                 prob_gt_100 = p_zero + p_nonzero*nbinom.sf(100, r, p)
        #                 prob_gt_1000 = p_zero + p_nonzero*nbinom.sf(1000, r, p)
        
        #         cum_sum_dict['prob_gt_10_entrained'].append(prob_gt_10)
        #         cum_sum_dict['prob_gt_100_entrained'].append(prob_gt_100)
        #         cum_sum_dict['prob_gt_1000_entrained'].append(prob_gt_1000)
        
        #         # ------------------------------------------------------------
        #         # 2) Fit distribution to YEARLY sum of entrained
        #         #    (keep your normal approach, or whatever you prefer)
        #         # ------------------------------------------------------------
        #         # For the yearly sums, idat.num_entrained:
        #         if len(idat) > 1:
        #             # Fit normal to the yearly sums
        #             mu_year, std_year = norm.fit(idat.num_entrained)
        #             mean_ent = mu_year
        #             lcl = norm.ppf(0.025, mu_year, std_year)
        #             ucl = norm.ppf(0.975, mu_year, std_year)
        #         else:
        #             # If there's only 0 or 1 data point for yearly sums, fallback
        #             mean_ent = float(idat.num_entrained.mean()) if len(idat) > 0 else 0.0
        #             lcl = mean_ent
        #             ucl = mean_ent
        
        #         cum_sum_dict['mean_ent'].append(mean_ent)
        #         cum_sum_dict['lcl_ent'].append(lcl)
        #         cum_sum_dict['ucl_ent'].append(ucl)
        
        # print("Yearly summary complete.")
        
        # # Convert the dictionary to a DataFrame
        # self.cum_sum = pd.DataFrame.from_dict(cum_sum_dict, orient='columns')
        
        # # create yearly summary by summing on species, flow scenario, and iteration
        
        # yearly_summary = self.daily_summary.groupby(by = ['species','scenario','iteration'])[['pop_size','num_survived','num_entrained']].sum()        
        # yearly_summary.reset_index(inplace = True)

        # cum_sum_dict = {'species':[],
        #                 'scenario':[],
        #                 'med_population':[],
        #                 'med_entrained':[], 
        #                 'med_survived':[],
        #                 'mean_ent':[],
        #                 'lcl_ent':[],
        #                 'ucl_ent':[],
        #                 'prob_gt_10_entrained':[],
        #                 'prob_gt_100_entrained':[],
        #                 'prob_gt_1000_entrained':[]}
        # # daily summary
        # for fishy in yearly_summary.species.unique():
        #     #iterate over scenarios
        #     for scen in yearly_summary.scenario.unique():
        #         # get data
        #         idat = yearly_summary[(yearly_summary.species == fishy) & (yearly_summary.scenario == scen)]
                
        #         # get cumulative sums and append to dictionary
        #         cum_sum_dict['species'].append(fishy)
        #         cum_sum_dict['scenario'].append(scen)
        #         cum_sum_dict['med_population'].append(idat.pop_size.median())
        #         cum_sum_dict['med_entrained'].append(idat.num_entrained.median())
        #         cum_sum_dict['med_survived'].append(idat.num_survived.median())
                
        #         day_dat = self.daily_summary[(self.daily_summary.species == fishy) & 
        #                                      (self.daily_summary.scenario == scen)]# &
        #                                      #(self.daily_summary.num_entrained > 0)]
                
        #         # fit distribution to number entrained per day
        #         dist = genpareto.fit(day_dat.num_entrained)
        #         probs_ent = genpareto.sf([10,100,1000],dist[0],dist[1],dist[2])
                
        #         # fit distribution to number entrained per year
        #         dist2 = norm.fit(idat.num_entrained)
        #         mean = norm.mean(dist2[0],dist2[1])
        #         lcl = norm.ppf(0.025,dist2[0],dist2[1])
        #         ucl = norm.ppf(0.975,dist2[0],dist2[1])

        #         cum_sum_dict['mean_ent'].append(mean)
        #         cum_sum_dict['lcl_ent'].append(lcl)
        #         cum_sum_dict['ucl_ent'].append(ucl)
        #         cum_sum_dict['prob_gt_10_entrained'].append(probs_ent[0])
        #         cum_sum_dict['prob_gt_100_entrained'].append(probs_ent[1])
        #         cum_sum_dict['prob_gt_1000_entrained'].append(probs_ent[2])
                
        # print ("Yearly summary complete")   
        
        self.cum_sum = pd.DataFrame.from_dict(cum_sum_dict,orient = 'columns')        
        results = self.beta_df
        day_sum = self.daily_summary
        year_sum = self.cum_sum
        #length = self.length_summ
        
        # summarize over iterations by Species and Flow Scenario
        
        with pd.ExcelWriter(self.wks_dir,engine = 'openpyxl', mode = 'a') as writer:
            results.to_excel(writer,sheet_name = 'beta fit')
            day_sum.to_excel(writer,sheet_name = 'daily summary')    
            year_sum.to_excel(writer,sheet_name = 'yearly summary')


        # plt.figure()
        # plt.hist(day_dat.total_entrained,color = 'r')
        # #plt.savefig(os.path.join(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke\Output",'fish.png'), dpi = 700)
        # plt.show()        
                


class hydrologic():
    """
    A Python class for conducting flow exceedance analysis using recent USGS data
    in relation to the contributing watershed size. It utilizes a linear relationship
    between drainage area and flow exceedances derived from the 100 nearest USGS
    gages to a given dam. This relationship is used to predict flow conditions such
    as what constitutes a wet spring.

    The class is designed to import relevant data, including dam locations and
    nearby USGS gages, and to calculate flow exceedances based on this data.
    """

    def __init__(self, nid_near_gage_dir, output_dir):
        """
        Initializes the hydrologic class with necessary data paths.

        Parameters:
        - nid_near_gage_dir (str): Path to the CSV file listing the 100 nearest
          USGS gages to every dam listed in the National Inventory of Dams (NID).
          The CSV should contain columns for NID dam identifiers, USGS gage
          identifiers (STAID), and other relevant metadata.
        - output_dir (str): Path to the directory where output files and analyses
          will be stored.

        Upon initialization, the class imports the NID-to-gage mapping data and
        prepares for flow exceedance calculations by setting up data structures
        for storing gage data and identified dams.
        """
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
        """
        Calculates seasonal flow exceedance probabilities for each gage and compiles
        the results into a DataFrame. The method considers predefined seasons and
        specific exceedance thresholds to evaluate flow rates.
    
        Parameters:
        - seasonal_dict (dict): A dictionary mapping seasons to lists of month numbers,
          defining the temporal scope of each season.
        - exceedence (list): A list of exceedance thresholds (percentages) for which
          flow rates are calculated.
        - HUC (int, optional): Hydrologic Unit Code to filter the analysis to a specific
          watershed or sub-watershed. If provided, only gages within the specified HUC
          are considered.
    
        The method iterates through the gages, applying the seasonal definitions and
        exceedance thresholds to calculate minimum flow rates that are exceeded at
        the specified percentages of time within each season. Results include gage
        identifiers, names, HUCs, drainage areas, seasons, calculated flow rates, and
        exceedance percentages.
    
        Outputs:
        - Updates the `exceedance` attribute of the class instance, storing the compiled
          exceedance data as a DataFrame.
        """

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
        """
        Performs an Ordinary Least Squares (OLS) regression to model the relationship
        between watershed size and flow exceedance probability for a specific dam
        and season. This method uses the statsmodels library to fit the model and
        provides a summary of the regression results.
    
        Parameters:
        - season (str): The season for which the analysis is conducted (e.g., 'Spring').
        - dam (str): The identifier for the dam of interest, corresponding to entries
          in the NID_to_gage DataFrame.
        - exceedance (str): Specifies the exceedance threshold being modeled (e.g.,
          'exc_90' for the 90th percentile exceedance flow).
    
        The function retrieves the relevant dam data, filters the exceedance DataFrame
        for the specified season and exceedance threshold, and extracts the drainage
        area and flow data for regression analysis.
    
        Outputs:
        - Prints a summary of the regression model, including the p-value and model
          coefficients. If the model is statistically significant (p-value < 0.05),
          it calculates and prints the exceedance flow for the specified dam based
          on its drainage area.
        - Updates class attributes with the regression data (X, Y), the predicted
          exceedance flow for the specified dam (DamY), and the dam's drainage area
          (DamX) for potential further analysis.
        """

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
        """
        A Python class for querying the Electric Power Research Institute (EPRI)
        entrainment database and fitting a distribution to the observed
        entrainment rates. The class allows for flexible querying of the database
        based on various ecological and hydrological parameters.
    
        The class supports analysis by state, plant capacity, date, taxonomy,
        feeding guild, habitat preference, water body type, and more, to explore
        entrainment patterns across different environmental and operational conditions.
        """
        def __init__(self, 
                     states = None, 
                     plant_cap = None,
                     Family = None,
                     Genus = None,
                     Species = None,
                     Month = None,
                     HUC02 = None,
                     HUC04 = None,
                     HUC06 = None,
                     HUC08 = None,
                     NIDID = None,
                     River = None):
            """
            Initializes the epri class by querying the EPRI database based on the
            provided criteria. Optional arguments allow for targeted analysis of
            specific ecological or hydrological contexts.
        
            Parameters are used to filter the database for analysis. If no parameters
            are provided, the analysis will consider the entire dataset.
        
            Parameters:
            - states: State abbreviations to filter the data.
            - plant_cap: Plant capacity (cfs) with a direction for filtering (> or <=).
            - Month: Numeric representation of months for filtering.
            - Family, Genus, Species: Taxonomic filters for the analysis..
            - HUC02, HUC04, HUC06, HUC08: Hydrologic Unit Codes for geographic filtering.
            - NIDID: National Inventory of Dams identifier for filtering by dams.
            - River: River names for filtering the dataset.
        
            Upon initialization, the relevant subset of the EPRI database is loaded
            for subsequent analysis.
            """
              
    
            # import EPRI database
            #data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'..\Data\epri1997.csv')
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data', 'epri1997.csv')
            data_dir = os.path.normpath(data_dir)  # Normalize path for OS compatibility
            
            
            self.epri = pd.read_csv(data_dir,  encoding= 'unicode_escape')
    
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
            
            if HUC02 is not None:
                if isinstance(HUC02,str):
                    self.epri = self.epri[self.epri.HUC02 == HUC02]
                else:
                    self.epri = self.epri[self.epri['HUC02'].isin(HUC02)]
                    
            if HUC04 is not None:
                if isinstance(HUC04,str):
                    self.epri = self.epri[self.epri.HUC04 == HUC04]
                else:
                    self.epri = self.epri[self.epri['HUC04'].isin(HUC04)]
                    
            if HUC06 is not None:
                if isinstance(HUC06,str):
                    self.epri = self.epri[self.epri.HUC06 == HUC06]
                else:
                    self.epri = self.epri[self.epri['HUC06'].isin(HUC06)]
                    
            if HUC08 is not None:
                if isinstance(HUC08, int):
                    self.epri = self.epri[self.epri.HUC08 == HUC08]
                else:
                    self.epri = self.epri[self.epri['HUC08'].isin(HUC08)]
 
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
            self.family = Family
            self.genus = Genus
            self.species = Species
            self.month = Month
            self.presence = round(float(success/trials),4)
            self.max_ent_rate = self.epri.FishPerMft3.max()
            self.sample_size = len(self.epri)
            self.HUC02 = HUC02
            self.HUC04 = HUC04

    
            
            #print ("With a maximum entrainment rate of %s and only %s percent of records acount for 80 percent of the entrainment"%(self.epri.FishPerMft3.max(),
            #                                                                                                                round(len(self.epri[self.epri.FishPerMft3 > self.epri.FishPerMft3.max()*0.8]) / len(self.epri) * 100,2)))
            print ("There are %s records left to describe entrainment rates"%(len(self.epri)))
            print ("The maximum entrainment rate for this fish is: %s"%(self.epri.FishPerMft3.max()))
            print ("--------------------------------------------------------------------------------------------")
        
        
        def ParetoFit(self):
            """
            Fits a Pareto distribution to the filtered EPRI dataset to model entrainment
            rates. This method provides an analysis of entrainment patterns based on
            the shape, location, and scale parameters of the distribution.
        
            Outputs detailed statistics of the fitted Pareto distribution, including the
            mean, variance, and standard deviation, offering insights into the
            entrainment rates' distribution characteristics within the selected dataset.
            """
            print("Starting ParetoFit...", flush=True)
    
            # fit a pareto and write to the object
            self.dist_pareto = pareto.fit(self.epri.FishPerMft3.values, floc = 0)
            #self.dist_pareto = pareto.fit(self.epri.FishPerMft3.values, floc = 0)
            print ("The Pareto distribution has a shape parameter of b: %s,  location: %s and scale: %s"%(round(self.dist_pareto[0],4),
                                                                                                          round(self.dist_pareto[1],4),
                                                                                                          round(self.dist_pareto[2],4)))
            print ("--------------------------------------------------------------------------------------------")
    
            print("Finished ParetoFit.", flush=True)
    
        def ExtremeFit(self):
            """
            Fits an Extreme Value distribution to the filtered EPRI dataset to model entrainment
            rates. This method provides an analysis of entrainment patterns based on
            the shape, location, and scale parameters of the distribution.
        
            Outputs detailed statistics of the fitted Pareto distribution, including the
            mean, variance, and standard deviation, offering insights into the
            entrainment rates' distribution characteristics within the selected dataset.
            """
    
            # fit a pareto and write to the object
            self.dist_extreme = genextreme.fit(self.epri.FishPerMft3.values, floc = 0)
            #self.dist_extreme = genextreme.fit(self.epri.FishPerMft3.values, floc = 0)
            print ("The Generic Extreme Value distribution has a shape parameter of c: %s,  location: %s and scale: %s"%(round(self.dist_extreme[0],4),
                                                                                                          round(self.dist_extreme[1],4),
                                                                                                          round(self.dist_extreme[2],4)))
            print ("--------------------------------------------------------------------------------------------")
    
        def WeibullMinFit(self):
            """
            Fits a Frachet distribution to the filtered EPRI dataset to model entrainment
            rates. This method provides an analysis of entrainment patterns based on
            the shape, location, and scale parameters of the distribution.
        
            Outputs detailed statistics of the fitted Pareto distribution, including the
            mean, variance, and standard deviation, offering insights into the
            entrainment rates' distribution characteristics within the selected dataset.
            """
            print("Starting WeibullMinFit...", flush=True)


            # fit a pareto and write to the object
            self.dist_weibull = weibull_min.fit(self.epri.FishPerMft3.values, floc = 0)
            #self.dist_weibull = weibull_min.fit(self.epri.FishPerMft3.values, floc = 0)
            print ("The Weibull Max distribution has a shape parameter of c: %s,  location: %s and scale: %s"%(round(self.dist_weibull[0],4),
                                                                                                          round(self.dist_weibull[1],4),
                                                                                                          round(self.dist_weibull[2],4)))
            print ("--------------------------------------------------------------------------------------------")
            # existing code
            print("Finished WeibullMinFit.", flush=True)
            
        def LogNormalFit(self):
            """
            Fits a Log Normal distribution to the filtered EPRI dataset to model entrainment
            rates. This method provides an analysis of entrainment patterns based on
            the shape, location, and scale parameters of the distribution.
        
            Outputs detailed statistics of the fitted Pareto distribution, including the
            mean, variance, and standard deviation, offering insights into the
            entrainment rates' distribution characteristics within the selected dataset.
            """
            print("Starting LogNormalFit...", flush=True)
       
            # fit a pareto and write to the object
            self.dist_lognorm = lognorm.fit(self.epri.FishPerMft3.values, floc = 0)
            #self.dist_lognorm = lognorm.fit(self.epri.FishPerMft3.values, floc = 0)
            print ("The Log Normal distribution has a shape parameter of b: %s,  location: %s and scale: %s"%(round(self.dist_lognorm[0],4),
                                                                                                          round(self.dist_lognorm[1],4),
                                                                                                          round(self.dist_lognorm[2],4)))
            print ("--------------------------------------------------------------------------------------------")
            print("Finished LogNormalFit.", flush=True)
    
        def GumbelFit(self):
            """
            Fits a Gumbel distribution to the filtered EPRI dataset to model entrainment
            rates. This method provides an analysis of entrainment patterns based on
            the shape, location, and scale parameters of the distribution.
        
            Outputs detailed statistics of the fitted Pareto distribution, including the
            mean, variance, and standard deviation, offering insights into the
            entrainment rates' distribution characteristics within the selected dataset.
            """
    
            # fit a pareto and write to the object
            self.dist_gumbel = gumbel_r.fit(self.epri.FishPerMft3.values,floc = 0)
            #self.dist_gumbel = gumbel_r.fit(self.epri.FishPerMft3.values)
            print ("The Gumbel distribution has a shape parameter of location: %s and scale: %s"%(round(self.dist_gumbel[0],4),
                                                                                                          round(self.dist_gumbel[1],4)))
            print ("--------------------------------------------------------------------------------------------")
           
        def LengthSummary(self):
            # Ensure all required columns exist; if not, create them with zeros.
            required_columns = ['0_5', '5_10', '10_15', '15_20', '20_25', 
                                '25_38', '38_51', '51_64', '64_76', 'GT76']
            for col in required_columns:
                if col not in self.epri.columns:
                    self.epri[col] = 0
        
            # Sum up the counts for each cohort.
            cm_0_5 = np.int32(self.epri['0_5'].sum())
            cm_5_10 = np.int32(self.epri['5_10'].sum())
            cm_10_15 = np.int32(self.epri['10_15'].sum())
            cm_15_20 = np.int32(self.epri['15_20'].sum())
            cm_20_25 = np.int32(self.epri['20_25'].sum())
            cm_25_38 = np.int32(self.epri['25_38'].sum())
            cm_38_51 = np.int32(self.epri['38_51'].sum())
            cm_51_64 = np.int32(self.epri['51_64'].sum())
            cm_64_76 = np.int32(self.epri['64_76'].sum())
            cm_GT76 = np.int32(self.epri['GT76'].sum())
        
            # Check if there is any data.
            total_counts = (cm_0_5 + cm_5_10 + cm_10_15 + cm_15_20 + cm_20_25 +
                            cm_25_38 + cm_38_51 + cm_51_64 + cm_64_76 + cm_GT76)
            if total_counts == 0:
                print("Warning: No fish length data available.")
                self.lengths = np.array([])
                self.len_dist = (np.nan, np.nan, np.nan)
                return
        
            # Generate arrays of lengths by sampling uniformly within each cohort.
            cm_0_5_arr = np.random.uniform(low=0, high=5.0, size=cm_0_5)
            cm_5_10_arr = np.random.uniform(low=5.0, high=10.0, size=cm_5_10)
            cm_10_15_arr = np.random.uniform(low=10.0, high=15.0, size=cm_10_15)
            cm_15_20_arr = np.random.uniform(low=15.0, high=20.0, size=cm_15_20)
            cm_20_25_arr = np.random.uniform(low=20.0, high=25.0, size=cm_20_25)
            cm_25_38_arr = np.random.uniform(low=25.0, high=38.0, size=cm_25_38)
            cm_38_51_arr = np.random.uniform(low=38.0, high=51.0, size=cm_38_51)
            cm_51_64_arr = np.random.uniform(low=51.0, high=64.0, size=cm_51_64)
            cm_64_76_arr = np.random.uniform(low=64.0, high=76.0, size=cm_64_76)
            cm_GT76_arr = np.random.uniform(low=76.0, high=100.0, size=cm_GT76)
        
            # Concatenate all arrays.
            try:
                self.lengths = np.concatenate((cm_0_5_arr,
                                                cm_5_10_arr,
                                                cm_10_15_arr,
                                                cm_15_20_arr,
                                                cm_20_25_arr,
                                                cm_25_38_arr,
                                                cm_38_51_arr,
                                                cm_51_64_arr,
                                                cm_64_76_arr,
                                                cm_GT76_arr), axis=0)
            except Exception as e:
                print("Error concatenating length arrays:", e)
                self.lengths = np.array([])
                self.len_dist = (np.nan, np.nan, np.nan)
                return
        
            # If the resulting array is empty, skip fitting.
            if self.lengths.size == 0:
                print("Warning: Concatenated lengths array is empty.")
                self.len_dist = (np.nan, np.nan, np.nan)
                return
        
            # Fit the concatenated array to a lognormal distribution.
            self.len_dist = lognorm.fit(self.lengths)
            #print(self.lengths)
            print("The log normal distribution has a shape parameter s: %s, location: %s and scale: %s" %
                  (round(self.len_dist[0], 4), round(self.len_dist[1], 4), round(self.len_dist[2], 4)))

            

           
        # def summary_output(self, output_dir, dist = 'Log Normal'):
        #     # species data
        #     if dist == 'Log Normal' or dist == 'Weibull' or dist == 'Pareto':
        #         family = self.family
        #         genus = self.genus 
        #         species = self.species
                
        #         # months
        #         month = self.month 
                
        #         huc02 = self.HUC02
                
        #         # presence and entrainment rate
        #         presence = self.presence 
        #         max_ent_rate = self.max_ent_rate 
        #         sample_size = self.sample_size
                
        #         # weibull c, location, scale
        #         weibull_p = self.weibull_t
        #         weibull_c = round(self.dist_weibull[0],4)
        #         weibull_loc = round(self.dist_weibull[1],4)
        #         weibull_scale = round(self.dist_weibull[2],4)
                
        #         # log normal b, location, scale
        #         log_normal_p = self.log_normal_t
        #         log_normal_b = round(self.dist_lognorm[0],4)
        #         log_normal_loc = round(self.dist_lognorm[1],4)
        #         log_normal_scale = round(self.dist_lognorm[2],4)
                
        #         pareto_p = self.pareto_t
        #         pareto_b = round(self.dist_pareto[0],4)
        #         pareto_loc = round(self.dist_pareto[1],4)
        #         pareto_scale = round(self.dist_pareto[2],4)
                
        #         length_b = round(self.len_dist[0],4)
        #         length_loc = round(self.len_dist[1],4)
        #         length_scale = round(self.len_dist[2],4)
                
        #         if dist == 'Log Normal':
        #             row = np.array([family, genus, species, log_normal_b, 
        #                             log_normal_loc, log_normal_scale, max_ent_rate, 
        #                             presence, length_b,length_loc,length_scale])
        #         elif dist == 'Weibull':
        #             row = np.array([family, genus, species, weibull_c, 
        #                             weibull_loc, weibull_scale, max_ent_rate, 
        #                             presence, length_b,length_loc,length_scale])    
        #         else:
        #             row = np.array([family, genus, species, pareto_b, 
        #                             pareto_loc, pareto_scale, max_ent_rate, 
        #                             presence, length_b,length_loc,length_scale])
                
        #         columns = ['family','genus','species','ent_shape','ent_loc',
        #                    'ent_scale','max_ent_rate','presence','length_b',
        #                    'length_loc','length_scale']
        #         new_row_df = pd.DataFrame([row],columns = columns)
                
        #         try:
        #             results = pd.read_csv(os.path.join(output_dir,'epri_fit.csv'))
        #         except FileNotFoundError:
        #             results = pd.DataFrame(columns = columns)
                    
        #         results = pd.concat([results,new_row_df], ignore_index = True)
        #         results.to_csv(os.path.join(output_dir,'epri_fit.csv'), index = False)
                
        #     else:
        #         return print('Distribution no supported by stryke')
        def plot_fish_lengths(lengths, output_folder="static"):
            """Generates a histogram of fish lengths and saves it as an image file."""
            plt.figure(figsize=(5, 3))
            plt.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel("Fish Length (cm)")
            plt.ylabel("Frequency")
            plt.title("Distribution of Fish Lengths")
        
            # Ensure output directory exists
            os.makedirs(output_folder, exist_ok=True)
        
            # Save plot
            plot_path = os.path.join(output_folder, "fish_lengths.png")
            plt.savefig(plot_path)
            plt.close()
            
            return "fish_lengths.png"  # Return filename
        
        def plot(self):
            """
            Generates a 2x2 subplot figure showing histograms (with natural log-transformed data)
            for the observed entrainment rates and for simulated samples drawn from the fitted Pareto,
            Log Normal, and Weibull distributions. It also performs KS tests comparing the observed data
            with each simulated sample, storing the KS p-values in the object.
            """
            from scipy.stats import ks_2samp, pareto, lognorm, weibull_min
            import matplotlib.pyplot as plt
            import numpy as np
        
            # Generate simulated samples using explicit conditionals.
            if self.dist_pareto is not None:
                pareto_sample = pareto.rvs(self.dist_pareto[0], loc=self.dist_pareto[1],
                                           scale=self.dist_pareto[2], size=1000)
            else:
                pareto_sample = np.array([])
        
            if self.dist_lognorm is not None:
                lognorm_sample = lognorm.rvs(self.dist_lognorm[0], loc=self.dist_lognorm[1],
                                             scale=self.dist_lognorm[2], size=1000)
            else:
                lognorm_sample = np.array([])
        
            if self.dist_weibull is not None:
                weibull_sample = weibull_min.rvs(self.dist_weibull[0], loc=self.dist_weibull[1],
                                                 scale=self.dist_weibull[2], size=1000)
            else:
                weibull_sample = np.array([])
        
            # Get the observed entrainment data.
            observations = self.epri.FishPerMft3.values
        
            # Perform KS tests to compare the observed data with each simulated sample.
            t1 = ks_2samp(observations, pareto_sample, alternative='two-sided')
            t2 = ks_2samp(observations, lognorm_sample, alternative='two-sided')
            t3 = ks_2samp(observations, weibull_sample, alternative='two-sided')
            self.pareto_t = round(t1[1], 4)
            self.log_normal_t = round(t2[1], 4)
            self.weibull_t = round(t3[1], 4)
        
            # Set matplotlib style parameters.
            plt.rcParams['font.size'] = 6
            plt.rcParams['font.family'] = 'serif'
            figSize = (4, 4)
            
            # Create a 2x2 subplot figure.
            fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=figSize)
            
            # Plot the histogram of the log-transformed observed data.
            axs[0, 0].hist(np.log(observations), color='darkorange', density=True)
            axs[0, 0].set_title('Observations')
            axs[0, 0].set_xlabel('org per Mft3')
            
            # Plot the histogram for the Pareto simulated sample.
            axs[0, 1].hist(np.log(pareto_sample), color='blue', lw=2, density=True)
            axs[0, 1].set_title('Pareto p = %s' % (self.pareto_t))
            axs[0, 1].set_xlabel('org per Mft3')
            
            # Plot the histogram for the Log Normal simulated sample.
            axs[1, 0].hist(np.log(lognorm_sample), color='blue', lw=2, density=True)
            axs[1, 0].set_title('Log Normal p = %s' % (self.log_normal_t))
            axs[1, 0].set_xlabel('org per Mft3')
            
            # Plot the histogram for the Weibull simulated sample.
            axs[1, 1].hist(np.log(weibull_sample), color='darkorange', lw=2, density=True)
            axs[1, 1].set_title('Weibull p = %s' % (self.weibull_t))
            axs[1, 1].set_xlabel('org per Mft3')
            
            
            return fig
            #plt.show()
            

        def summary_output(self, output_dir, dist='Log Normal'):
            """
            Generates a detailed formatted summary report that includes:
              - Query filters (Family, Genus, Species, Month, HUC02, etc.)
              - Basic statistics: Sample size, probability of presence, maximum entrainment rate
              - Fish length distribution: Mean and standard deviation of fish lengths
              - Fitted parameters for Pareto, Log Normal, and Weibull distributions along with their KS p-values
            The report is written to a file "epri_summary_report.txt" in output_dir and is also returned.
            """
            # Basic query information
            family = self.family if self.family is not None else "N/A"
            genus = self.genus if self.genus is not None else "N/A"
            species = self.species if self.species is not None else "N/A"
            month = self.month if self.month is not None else []
            huc02 = self.HUC02 if self.HUC02 is not None else "N/A"
            # Basic statistics
            presence = self.presence if hasattr(self, 'presence') else "N/A"
            max_ent_rate = self.max_ent_rate if hasattr(self, 'max_ent_rate') else "N/A"
            sample_size = self.sample_size if hasattr(self, 'sample_size') else "N/A"
            
            # Fish length distribution stats (if available)
            if hasattr(self, 'len_dist'):
                dist_lognorm_shape = round(self.len_dist[0], 4)
                dist_lognorm_loc = round(self.len_dist[1], 4)
                dist_lognorm_scale = round(self.len_dist[2], 4)
            else:
                dist_lognorm_shape = dist_lognorm_loc = dist_lognorm_scale = "N/A"
            
            # For Pareto:
            if hasattr(self, 'dist_pareto'):
                pareto_shape = round(self.dist_pareto[0], 4)
                pareto_loc = round(self.dist_pareto[1], 4)
                pareto_scale = round(self.dist_pareto[2], 4)
                pareto_ks = getattr(self, 'pareto_t', "N/A")
            else:
                pareto_shape = pareto_loc = pareto_scale = pareto_ks = "N/A"
            
            # For Log Normal:
            if hasattr(self, 'dist_lognorm'):
                lognorm_shape = round(self.dist_lognorm[0], 4)
                lognorm_loc = round(self.dist_lognorm[1], 4)
                lognorm_scale = round(self.dist_lognorm[2], 4)
                lognorm_ks = getattr(self, 'log_normal_t', "N/A")
            else:
                lognorm_shape = lognorm_loc = lognorm_scale = lognorm_ks = "N/A"
            
            # For Weibull:
            if hasattr(self, 'dist_weibull'):
                weibull_shape = round(self.dist_weibull[0], 4)
                weibull_loc = round(self.dist_weibull[1], 4)
                weibull_scale = round(self.dist_weibull[2], 4)
                weibull_ks = getattr(self, 'weibull_t', "N/A")
            else:
                weibull_shape = weibull_loc = weibull_scale = weibull_ks = "N/A"
            
            # Build the report lines.
            lines = []
            lines.append("--------------------------------------------------")
            lines.append("            EPRI Fitting Summary Report           ")
            lines.append("--------------------------------------------------")
            lines.append(f"Species: {family} {genus} {species}")
            lines.append("")
            lines.append("Filters Applied:")
            lines.append(f"   Months: {month}")
            lines.append(f"   HUC02: {huc02}")
            lines.append("")
            lines.append("Statistics:")
            lines.append(f"   Sample Size: {sample_size}")
            lines.append(f"   Presence: {presence}")
            lines.append(f"   Maximum Entrainment Rate: {max_ent_rate}")
            lines.append("")
            lines.append("Fish Length Distribution:")
            lines.append(f"   Shape: {dist_lognorm_shape}")
            lines.append(f"   Location: {dist_lognorm_loc}")
            lines.append(f"   Scale: {dist_lognorm_scale}")
#            lines.append(f"   Lengths:{self.lengths}")
            lines.append("")
            lines.append("Pareto Distribution:")
            lines.append(f"   Shape: {pareto_shape}")
            lines.append(f"   Location: {pareto_loc}")
            lines.append(f"   Scale: {pareto_scale}")
            lines.append(f"   KS p-value: {pareto_ks}")
            lines.append("")
            lines.append("Log Normal Distribution:")
            lines.append(f"   Shape: {lognorm_shape}")
            lines.append(f"   Location: {lognorm_loc}")
            lines.append(f"   Scale: {lognorm_scale}")
            lines.append(f"   KS p-value: {lognorm_ks}")
            lines.append("")
            lines.append("Weibull Distribution:")
            lines.append(f"   Shape: {weibull_shape}")
            lines.append(f"   Location: {weibull_loc}")
            lines.append(f"   Scale: {weibull_scale}")
            lines.append(f"   KS p-value: {weibull_ks}")
            lines.append("--------------------------------------------------")
            
            final_report = "\n".join(lines)
            
            # Write the report to a text file.
            report_file = os.path.join(output_dir, "epri_summary_report.txt")
            with open(report_file, "w") as f:
                f.write(final_report)
            
            print(final_report)
            return final_report



        # def summary_output(self, output_dir, dist='Log Normal'):
        #     """
        #     Generates a formatted summary report (as a multi‐line string) including:
        #       - Query filters (Family, Genus, Species, Month, HUC02, etc.)
        #       - Basic statistics: sample size, maximum entrainment rate, probability of presence
        #       - Fish length distribution: mean and standard deviation (if available)
        #       - Fitting results for the chosen distribution (shape, location, scale, KS p-value)
            
        #     The report is written to a text file ("epri_summary_report.txt") in output_dir.
        #     """
        #     # Check that the distribution is supported.
        #     if dist not in ['Log Normal', 'Weibull', 'Pareto']:
        #         return "Distribution not supported by stryke"
            
        #     # Species / query information.
        #     family = self.family if self.family is not None else "N/A"
        #     genus = self.genus if self.genus is not None else "N/A"
        #     species = self.species if self.species is not None else "N/A"
        #     month = self.month if self.month is not None else []
        #     huc02 = self.HUC02 if self.HUC02 is not None else "N/A"
        #     # Basic stats.
        #     presence = self.presence if hasattr(self, 'presence') else "N/A"
        #     max_ent_rate = self.max_ent_rate if hasattr(self, 'max_ent_rate') else "N/A"
        #     sample_size = self.sample_size if hasattr(self, 'sample_size') else "N/A"
            
        #     # Fish length distribution stats.
        #     if hasattr(self, 'lengths') and self.lengths.size > 0:
        #         mean_length = round(np.mean(self.lengths), 4)
        #         std_length = round(np.std(self.lengths), 4)
        #     else:
        #         mean_length = "N/A"
        #         std_length = "N/A"
            
        #     # Distribution parameters.
        #     if dist == 'Log Normal':
        #         shape_val = round(self.dist_lognorm[0], 4) if hasattr(self, 'dist_lognorm') else "N/A"
        #         loc_val   = round(self.dist_lognorm[1], 4) if hasattr(self, 'dist_lognorm') else "N/A"
        #         scale_val = round(self.dist_lognorm[2], 4) if hasattr(self, 'dist_lognorm') else "N/A"
        #         ks_p      = self.log_normal_t if hasattr(self, 'log_normal_t') else "N/A"
        #     elif dist == 'Weibull':
        #         shape_val = round(self.dist_weibull[0], 4) if hasattr(self, 'dist_weibull') else "N/A"
        #         loc_val   = round(self.dist_weibull[1], 4) if hasattr(self, 'dist_weibull') else "N/A"
        #         scale_val = round(self.dist_weibull[2], 4) if hasattr(self, 'dist_weibull') else "N/A"
        #         ks_p      = self.weibull_t if hasattr(self, 'weibull_t') else "N/A"
        #     else:  # Pareto
        #         shape_val = round(self.dist_pareto[0], 4) if hasattr(self, 'dist_pareto') else "N/A"
        #         loc_val   = round(self.dist_pareto[1], 4) if hasattr(self, 'dist_pareto') else "N/A"
        #         scale_val = round(self.dist_pareto[2], 4) if hasattr(self, 'dist_pareto') else "N/A"
        #         ks_p      = self.pareto_t if hasattr(self, 'pareto_t') else "N/A"
            
        #     # Build the report lines.
        #     lines = []
        #     lines.append("--------------------------------------------------")
        #     lines.append("            EPRI Fitting Summary Report           ")
        #     lines.append("--------------------------------------------------")
        #     lines.append(f"Species: {family} {genus} {species}")
        #     lines.append("")
        #     lines.append("Filters Applied:")
        #     lines.append(f"   Months: {month}")
        #     lines.append(f"   HUC02: {huc02}")
        #     # (Add additional filters if desired, e.g., states, plant_cap, etc.)
        #     lines.append("")
        #     lines.append("Statistics:")
        #     lines.append(f"   Sample Size: {sample_size}")
        #     lines.append(f"   Presence: {presence}")
        #     lines.append(f"   Maximum Entrainment Rate: {max_ent_rate}")
        #     lines.append("")
        #     lines.append("Fish Length Distribution:")
        #     lines.append(f"   Mean Length: {mean_length}")
        #     lines.append(f"   Standard Deviation: {std_length}")
        #     lines.append("")
        #     lines.append(f"{dist} Distribution:")
        #     lines.append(f"   Shape: {shape_val}")
        #     lines.append(f"   Location: {loc_val}")
        #     lines.append(f"   Scale: {scale_val}")
        #     lines.append(f"   KS p-value: {ks_p}")
        #     lines.append("--------------------------------------------------")
            
        #     final_report = "\n".join(lines)
            
        #     # Write the report to a text file.
        #     report_file = os.path.join(output_dir, "epri_summary_report.txt")
        #     with open(report_file, "w") as f:
        #         f.write(final_report)
            
        #     # Also print it.
        #     print(final_report)
            
        #     return final_report