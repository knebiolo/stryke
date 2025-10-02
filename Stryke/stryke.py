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
thousands of times and eventually, through black magic *cough (probability theory), 
we have a pretty good estimate of what the overall downstream passage survival 
would be of a theoretical population of fish through a theoretical hydroelectric 
facility.

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
# Force pandas to show everything (all columns and rows)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # Let it use full width
pd.set_option('display.colheader_justify', 'left')
import os
import matplotlib
matplotlib.use('Agg')

from matplotlib import rcParams
rcParams.update({'font.size': 8, 'font.family': 'sans-serif'})

import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import beta
import xlrd
import networkx as nx
from networkx.readwrite import json_graph
#from Stryke.hydrofunctions import hydrofunctions as hf
import hydrofunctions as hf
from .barotrauma import baro_injury_prob, baro_surv_prob, calc_v, calc_k_viscosity, calc_friction, calc_h_loss, calc_p_2, calc_p_1
#from Stryke.barotrauma import baro_injury_prob, baro_surv_prob, calc_v, calc_k_viscosity, calc_friction, calc_h_loss, calc_p_2, calc_p_1
import requests
#import geopandas as gp
import statsmodels.api as sm
import math
from scipy.stats import beta, pareto, genextreme, genpareto, lognorm, weibull_min, gumbel_r, ks_2samp, nbinom, norm
from scipy import constants
import h5py
#import tables
from numpy.random import default_rng
rng = default_rng()
import logging
logger = logging.getLogger(__name__)

# Diagnostic Control Flags
DIAGNOSTICS_ENABLED = True   # Set to False to disable ALL diagnostics
VERBOSE_DIAGNOSTICS = False  # Set to True for per-iteration/per-day details (SLOW!)
                             # When False: Only shows summaries and important events

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

# Function to compute mean and credible interval
def summarize_ci(series):
    mean = series.mean()
    lower_bound = np.percentile(series, 2.5)  # 2.5th percentile
    upper_bound = np.percentile(series, 97.5)  # 97.5th percentile
    return pd.Series({"mean": mean, "lower_95_CI": lower_bound, "upper_95_CI": upper_bound})
            
            
def bootstrap_mean_ci(data, n_bootstrap=10000, ci=95):
    """
    Calculate the mean and a bootstrap-based credible interval for the given data.
    
    Parameters:
        data (array-like): The data to sample.
        n_bootstrap (int): Number of bootstrap samples.
        ci (float): The credible interval percentage (default: 95).
    
    Returns:
        tuple: (mean, lower bound, upper bound)
    """
    means = []
    data = np.array(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper
            
def to_dataframe(data, numeric_cols=None, index_col=None):
    """Converts data (list/dict or DataFrame) to DataFrame and optionally converts columns."""
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    if index_col and index_col in df.columns:
        df.set_index(index_col, inplace=True, drop=False)
    return df

def _read_csv_if_exists_compat(file_path=None, *args, **kwargs):
    """
    Backward-compatible wrapper:
      - tolerates file_path=None / "" (returns None)
      - accepts optional numeric_cols kw and coerces those columns if present
    """
    numeric_cols = kwargs.pop("numeric_cols", None)

    if not file_path or (isinstance(file_path, str) and not file_path.strip()):
        return None
    if not isinstance(file_path, (str, bytes, os.PathLike)):
        raise TypeError(f"read_csv_if_exists(file_path=...) expected a path, got {type(file_path).__name__}")

    if not os.path.exists(file_path):
        # Match previous behavior: either return None or raise; returning None is kinder to UIs.
        # If you prefer hard-fail, change to: raise FileNotFoundError(...)
        return None

    df = pd.read_csv(file_path)
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def read_csv_if_exists(*args, **kwargs):
    return _read_csv_if_exists_compat(*args, **kwargs)


class simulation():
    ''' Python class object that initiates, runs, and holds data for a facility
    specific simulation'''
    def __init__ (self, proj_dir, output_name, wks = None, existing = False):
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
        if existing == False and wks:
            self.worksheet_import(proj_dir,wks, output_name)
    
        elif existing == True and wks:
            self.existing_import(proj_dir, wks, output_name)
        
        #logger.info('simulation object created')
            
 
    def existing_import(self, proj_dir, wks, output_name):
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
        
        self.edges = pd.read_excel(self.wks_dir,
                                   sheet_name = 'Edges',
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

    def worksheet_import(self, proj_dir, wks, output_name):
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
        self.edges = pd.read_excel(self.wks_dir,
                                   sheet_name = 'Edges',
                                   header = 0,
                                   index_col = None, 
                                   usecols = "B:D", 
                                   skiprows = 9)
        self.surv_fun_df = self.nodes[['Location','Surv_Fun']].set_index('Location')
        self.surv_fun_dict = self.surv_fun_df.to_dict('index')

        if len(self.nodes) > 1:
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
                                         usecols = "B:Y",
                                         skiprows = 4)
        self.unit_params.set_index('Unit',inplace = True)
        
        self.facility_params = pd.read_excel(self.wks_dir,
                                         sheet_name = 'Facilities', 
                                         header = 0, 
                                         index_col = None,
                                         usecols = "B:I",
                                         skiprows = 3)
        self.facility_params.set_index('Facility', inplace = True)

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
            # Convert barotrauma parameters from meters to feet
            self.unit_params['fb_depth'] = self.unit_params.fb_depth * 3.28084
            self.unit_params['ps_D'] = self.unit_params.ps_D * 3.28084
            self.unit_params['ps_length'] = self.unit_params.ps_length * 3.28084
            self.unit_params['submergence_depth'] = self.unit_params.submergence_depth * 3.28084
            # roughness is in mm, no conversion needed
            self.facility_params['Rack Spacing'] = self.facility_params['Rack Spacing']*0.0328084
        else:
            self.facility_params['Rack Spacing'] = self.facility_params['Rack Spacing']/12.


        self.operating_scenarios_df = pd.read_excel(self.wks_dir,
                                                    sheet_name = 'Operating Scenarios', 
                                                    header = 0, 
                                                    index_col = None,
                                                    usecols = "B:I",
                                                    skiprows = 8)
        
        self.operating_scenarios_df['OpScenario'] = self.operating_scenarios_df.Scenario + " " + self.operating_scenarios_df.Unit
        self.ops_scens = None
        
        self.flow_scenarios = self.flow_scenarios_df['Scenario'].unique()
        self.op_scenarios = self.operating_scenarios_df['Scenario'].unique()

        # import population data
        self.pop = pd.read_excel(self.wks_dir,
                                 sheet_name = 'Population',
                                 header = 0,
                                 index_col = None,
                                 usecols = "B:V", 
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
        self.hdf['Edges'] = self.edges
        
        self.hdf['Unit_Parameters'] = self.unit_params
        #self.hdf['Routing'] = self.routing
        self.hdf.flush()
    
    def webapp_import(self, data_dict, output_name):
        DIAGNOSTICS_ENABLED = True  # Set to False to disable diagnostics
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] webapp_import called with output_name: {output_name}")
            print(f"[DIAG] proj_dir: {data_dict.get('proj_dir', os.getcwd())}")
            print(f"[DIAG] data_dict keys: {list(data_dict.keys())}")
            for k, v in data_dict.items():
                print(f"  {k}: type={type(v)}")
        hdf_path = os.path.join(data_dict.get('proj_dir', os.getcwd()), f"{output_name}.h5")
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] HDF5 output path: {hdf_path}")
            print(f"[DIAG] HDF5 path exists before write: {os.path.exists(hdf_path)}")
        """
        Imports data for a new simulation from in-memory/webapp sources,
        and writes data to an HDF5 file.
        """
        # Store basic info.
        self.output_name = output_name
        self.output_units = data_dict.get('units_system')
        self.sim_mode = data_dict.get('simulation_mode')
        self.proj_dir = data_dict.get("proj_dir", os.getcwd())
        hdf_path = os.path.join(self.proj_dir, f"{output_name}.h5")
        # Set wks_dir for Excel export compatibility (web app uses HDF5 path as reference)
        self.wks_dir = hdf_path
        
        # Convert graph summary data to DataFrames.
        graph_summary = data_dict.get('graph_summary', {})
        self.nodes = to_dataframe(graph_summary.get('Nodes', []))
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] Nodes DataFrame shape: {self.nodes.shape}")
            print(f"[DIAG] Nodes DataFrame columns: {self.nodes.columns.tolist()}")
            print(f"[DIAG] Nodes DataFrame head:\n{self.nodes.head()}")
        try:
            self.edges = to_dataframe(graph_summary.get('Edges', []))
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Edges DataFrame shape: {self.edges.shape}")
                print(f"[DIAG] Edges DataFrame columns: {self.edges.columns.tolist()}")
                print(f"[DIAG] Edges DataFrame head:\n{self.edges.head()}")
        except Exception as e:
            logger.info("Single Unit Scenario Identified, No Movement")
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Edges DataFrame construction failed: {e}")
        self.surv_fun_df = self.nodes[['Location','Surv_Fun']].set_index('Location')
        self.surv_fun_dict = self.surv_fun_df.to_dict('index')
    
        
        # Build the simulation graph.
        sim_graph_data = data_dict.get('simulation_graph')
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] simulation_graph in data_dict: {sim_graph_data is not None}")
        if sim_graph_data is not None:
            G = json_graph.node_link_graph(sim_graph_data)
        else:
            G = nx.DiGraph()
            for _, row in self.nodes.iterrows():
                node_id = row.get("ID", row.get("Location"))
                G.add_node(node_id, **row.to_dict())
            for _, row in self.edges.iterrows():
                source = row.get("_from")
                target = row.get("_to")
                try:
                    weight = float(row.get("weight", 1.0))
                except (ValueError, TypeError):
                    weight = 1.0
                G.add_edge(source, target, Weight=weight)
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] Graph nodes: {G.number_of_nodes()}")
            print(f"[DIAG] Graph edges: {G.number_of_edges()}")
        
        if "graph_data" in data_dict:
            G = json_graph.node_link_graph(data_dict["graph_data"])
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] graph_data loaded. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
        try:
            path_list = list(nx.all_shortest_paths(G, 'river_node_0', 'river_node_1'))
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Shortest path list: {path_list}")
        except nx.NetworkXNoPath:
            logger.info("No path found between river_node_0 and river_node_1")
            if DIAGNOSTICS_ENABLED:
                print("[DIAG] No path found between river_node_0 and river_node_1")
        except nx.NodeNotFound as e:
            logger.info("NodeNotFound:")
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] NodeNotFound: {e}")
        
        if len(self.nodes) > 1:
            try:
                paths = list(nx.all_shortest_paths(G, 'river_node_0', 'river_node_1'))
                max_len = max(len(path) for path in paths)
                self.moves = np.arange(0, max_len + 1, 1)
                if DIAGNOSTICS_ENABLED:
                    print(f"[DIAG] Moves array: {self.moves}")
            except nx.NetworkXNoPath:
                logger.warning("No path found between river_node_0 and river_node_1.")
                self.moves = np.zeros(1, dtype=np.int32)
                if DIAGNOSTICS_ENABLED:
                    print("[DIAG] Moves array set to zeros due to no path.")
        else:
            self.moves = np.zeros(1, dtype=np.int32)
            if DIAGNOSTICS_ENABLED:
                print("[DIAG] Moves array set to zeros due to single node.")
        self.graph = G
        


        
        # 3. Unit Parameters.
        if "unit_parameters_file" in data_dict:
            self.unit_params = read_csv_if_exists(
                data_dict["unit_parameters_file"],
                numeric_cols=[
                    'B', 'D', 'D1', 'D2', 'H', 'N', 'Qcap', 'Qopt',
                    'RPM', 'ada', 'intake_vel', 'iota', 'lambda',
                    'op_order', 'roughness',
                    'fb_depth', 'ps_D', 'ps_length', 'submergence_depth', 'elevation_head'
                ],
                index_col="Unit_Name"
            )
            self.unit_params['elevation_head'] = self.unit_params.submergence_depth
            if self.unit_params is not None:
                self.unit_params['Unit_Name'] = self.unit_params.Facility + ' - Unit ' + self.unit_params.Unit.astype('str')
                self.unit_params.set_index('Unit_Name', inplace=True)
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Unit params DataFrame shape: {self.unit_params.shape}")
                print(f"[DIAG] Unit params DataFrame columns: {self.unit_params.columns.tolist()}")
                print(f"[DIAG] Unit params DataFrame head:\n{self.unit_params.head()}")
        elif "unit_parameters" in data_dict:
            self.unit_params = to_dataframe(data_dict["unit_parameters"])
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Unit params DataFrame shape: {self.unit_params.shape}")
                print(f"[DIAG] Unit params DataFrame columns: {self.unit_params.columns.tolist()}")
                print(f"[DIAG] Unit params DataFrame head:\n{self.unit_params.head()}")

        # 4. Facilities.
        if "facilities" in data_dict:
            self.facility_params = to_dataframe(data_dict["facilities"], numeric_cols=['Bypass Flow', 'Env Flow', 'Min Op Flow', 'Rack Spacing', 'Units'], index_col="Facility")
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Facility params DataFrame shape: {self.facility_params.shape}")
                print(f"[DIAG] Facility params DataFrame columns: {self.facility_params.columns.tolist()}")
                print(f"[DIAG] Facility params DataFrame head:\n{self.facility_params.head()}")

        # 5. Flow Scenarios.
        if "flow_scenarios" in data_dict:
            self.flow_scenarios_df = to_dataframe(data_dict["flow_scenarios"], numeric_cols=['FlowYear', 'Prorate'])
            self.flow_scenarios = self.flow_scenarios_df["Scenario"].unique()
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Flow scenarios DataFrame shape: {self.flow_scenarios_df.shape}")
                print(f"[DIAG] Flow scenarios DataFrame columns: {self.flow_scenarios_df.columns.tolist()}")
                print(f"[DIAG] Flow scenarios DataFrame head:\n{self.flow_scenarios_df.head()}")
        
        # 6. Operating Scenarios.
        if "operating_scenarios_file" in data_dict:
            self.operating_scenarios_df = read_csv_if_exists(
                data_dict["operating_scenarios_file"],
                numeric_cols=['Hours', 'Location', 'Prob Not Operating', 'Scale', 'Shape', 'Unit']
            )
            #logger.info('operating scenarios columns', self.operating_scenarios_df.columns.to_list())
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Operating scenarios DataFrame shape: {self.operating_scenarios_df.shape}")
                print(f"[DIAG] Operating scenarios DataFrame columns: {self.operating_scenarios_df.columns.tolist()}")
                print(f"[DIAG] Operating scenarios DataFrame head:\n{self.operating_scenarios_df.head()}")
        else:
            self.operating_scenarios_df = None
        
        if self.operating_scenarios_df is not None and "Scenario" in self.operating_scenarios_df.columns:
            self.op_scenarios = self.operating_scenarios_df["Scenario"].unique()
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] op_scenarios: {self.op_scenarios}")
        else:
            self.op_scenarios = []
        
        # 7. Population.
        if "population" in data_dict:
            pop_data = data_dict["population"]
            if isinstance(pop_data, dict):
                pop_data = [pop_data]
            self.pop = to_dataframe(pop_data, numeric_cols=['Iterations', 'Length_mean', 'Length_sd', 'U_crit',
                                                               'length location', 'length scale', 'length shape',
                                                               'location', 'max_ent_rate', 'occur_prob', 'scale', 'shape'])
            #logger.info('population dataframe columns:', self.pop.columns.to_list())
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Population DataFrame shape: {self.pop.shape}")
                print(f"[DIAG] Population DataFrame columns: {self.pop.columns.tolist()}")
                print(f"[DIAG] Population DataFrame head:\n{self.pop.head()}")
        # 8. Hydrograph.
        if "hydrograph_file" in data_dict:
            self.input_hydrograph_df = read_csv_if_exists(data_dict["hydrograph_file"])
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Hydrograph DataFrame shape: {self.input_hydrograph_df.shape if self.input_hydrograph_df is not None else 'None'}")
                if self.input_hydrograph_df is not None:
                    print(f"[DIAG] Hydrograph DataFrame columns: {self.input_hydrograph_df.columns.tolist()}")
                    print(f"[DIAG] Hydrograph DataFrame head:\n{self.input_hydrograph_df.head()}")

        
        # # 9. Unit Conversion.
        # if data_dict.get("units_system", "imperial") == "metric":
        #     if hasattr(self, "input_hydrograph_df") and self.input_hydrograph_df is not None and "DAvgFlow_prorate" in self.input_hydrograph_df.columns:
        #         self.input_hydrograph_df["DAvgFlow_prorate"] *= 35.3147
        
        
        # 10. Create HDF5 file and store DataFrames.
        if os.path.exists(hdf_path):
            os.remove(hdf_path)
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Existing HDF5 file removed: {hdf_path}")
        hdf_path = os.path.join(self.proj_dir, f"{output_name}.h5")
        self.hdf = pd.HDFStore(hdf_path, mode='w')
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] HDF5 file opened for writing: {hdf_path}")
        for key, df in [("Flow Scenarios", getattr(self, "flow_scenarios_df", None)),
                        ("Operating Scenarios", getattr(self, "operating_scenarios_df", None)),
                        ("Population", getattr(self, "pop", None)),
                        ("Nodes", self.nodes),
                        ("Edges", self.edges),
                        ("Unit_Parameters", getattr(self, "unit_params", None)),
                        ("Facilities", getattr(self, "facility_params", None)),
                        ("Hydrograph", getattr(self, "input_hydrograph_df", None))]:
            if df is not None:
                self.hdf[key] = df
                if DIAGNOSTICS_ENABLED:
                    print(f"[DIAG] Wrote {key} to HDF5. Shape: {df.shape}")
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] HDF5 file flushed.")
        self.hdf.flush()
        self.hdf.close()
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] HDF5 file closed.")
        
        # Save the HDFStore file path for later use in run()
        self.hdf_path = hdf_path
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] Final HDF5 path stored: {self.hdf_path}")
        
        if hasattr(self, "unit_params") and self.unit_params is not None:
            if "Qcap" in self.unit_params.columns and "Facility" in self.unit_params.columns:
                self.flow_cap = self.unit_params.groupby("Facility")["Qcap"].sum()
            else:
                self.flow_cap = None
        
        logger.info("Web app import completed. Data stored to %s",hdf_path)

    def create_graph_from_app(self, model_setup, session):
        """
        Create a NetworkX graph from web app data.
        
        If the model setup indicates a unit-only simulation,
        create a graph with a single node based on unit parameters.
        Otherwise, if there is a Cytoscape JSON stored in session (from the interactive graph editor),
        convert it to a graph using create_graph_from_json.
        
        Parameters:
        -----------
        model_setup : str
            The simulation model setup string (e.g., "single_unit_survival_only", etc.)
        session : flask session
            The session object containing web app data.
        
        Returns:
        --------
        G : networkx.DiGraph
            The generated graph.
        """
        import networkx as nx
    
        # If the model is unit-only, create a graph from unit_params.
        if model_setup in ["single_unit_survival_only", "single_unit_simulated_entrainment"]:
            G = nx.DiGraph()
            if self.unit_params is not None and not self.unit_params.empty:
                # Take the first row from unit_params for the single unit.
                unit_row = self.unit_params.iloc[0]
                # Assume the unit number is stored in the "Unit" column (or use index)
                unit_number = unit_row["Unit"] if "Unit" in unit_row else unit_row.name
                node_id = f"unit_{unit_number}"
                # Add the unit node with its attributes.
                G.add_node(node_id, label=str(unit_number), type="unit", **unit_row.to_dict())
            else:
                print("Warning: No unit parameters available for single unit model.")
            return G
        else:
            # Otherwise, look for a Cytoscape JSON graph in the session.
            graph_json = session.get("graph_data")
            if graph_json:
                return self.create_graph_from_json(graph_json)
            else:
                # If no graph data, return an empty graph.
                print("No graph data found in session; returning an empty graph.")
                return nx.DiGraph()

            
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
    
        # Clip to valid probability range [0, 1]
        return np.clip(1 - p_strike, 0.0, 1.0)
    
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
    
        # Clip to valid probability range [0, 1]
        return np.clip(1 - p_strike, 0.0, 1.0)
    
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
    
        # Clip to valid probability range [0, 1]
        return np.clip(1 - p_strike, 0.0, 1.0)  # Survival probability
    
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
        # ✅ Fixed: Incorporate fish length in p_strike to match Franke methodology
        p_strike = gamma * (N * length / (0.707 * D2)) * (((np.sin(beta_p) * (B/D1))/(2*Qpwd)) + (np.cos(beta_p)/np.pi))
    
        # Clip to valid probability range [0, 1]
        return np.clip(1 - p_strike, 0.0, 1.0)
    
    # def barotrauma(self,discharge,K,fish_depth,h_D, beta_0, beta_1, calculate = False):
    #     """
    #     Calculates the pressure differential to produce a survival probability 
    #     as fish pass from an acclimated depth in the impoundment, through the
    #     facility, and into the draft tube.
        
    #     It is assumed the highest pressure experienced by a fish occurs at the 
    #     end of the penstock at the guide vanes just before it enters the turbine
    #     and the lowest pressure (nearly atmospheric) is experienced by the fish 
    #     as it enters the draft tube after exiting the turbine runner.
        
    #     Parameters:
    #     - discharge (float): (m^3/s)
    #     - K (float): absolute roughness of the penstock material (mm) See table from Miller 1996.
    #     - ps_diameter (float): penstock diameter (m)
    #     - ps_length (float): penstock length, or length from intake to outflow? (m)
    #     - v_head (float): velocity head at turbine inlet (m/s)
    #     - fish_depth (array): a value for fish acclimation depth (m)
    #     - h_D (float): submergence depth of the draft tube outlet (m) assumed to be 2m
    #     - h_2 (float): elevation head at the downstream point (m)
    #     - beta_0 (float): regression coefficent
    #     - beta_1 (float): regression coefficient
        
    #     Returns:
    #     - endpoint (array): an array of survival probabilities for each fish depth
    #     """
    #     def scalarize(x):
    #         if isinstance(x, (np.ndarray, list)) and len(x) == 1:
    #             return x[0]
    #         if hasattr(x, "item") and np.ndim(x) == 0:
    #             return x.item()
    #         return x
    #     discharge = scalarize(discharge)
    #     K = scalarize(K)
    #     #ps_diameter = scalarize(ps_diameter)
    #     #ps_length = scalarize(ps_length)
    #     #v_head = scalarize(v_head)
    #     fish_depth = scalarize(fish_depth)
    #     h_D = scalarize(h_D)
    #     #h_2 = scalarize(h_2)
    #     beta_0 = scalarize(beta_0)
    #     beta_1 = scalarize(beta_1)
        
    #     # calc penstock area from diameter input
    #     #a = np.pi * (ps_diameter/2)**2
        
    #     #logger.debug('calculated ps area')
    #     # Either the spreadsheet inputs are imperial, or they get converted to
    #     # imperial in __init__. Therefore, they must always be converted to metric
    #     # for barotrauma math.   
        
    #     # convert imperial input units to metric for calcs
    #     discharge = discharge * 0.02831683199881 # cfs to cms
    #     #a = a * 0.092903                         # sq ft to sq m
    #     #ps_diameter = ps_diameter * 0.3048       # ft to m
    #     #ps_length = ps_length * 0.3048           # ft to m
    #     #v_head = v_head * 0.02831683199881       # cfs to cms
    #     fish_depth = fish_depth * 0.3048       # ft to m
    #     h_D = h_D * 0.3048                       # ft to m
    #     #h_2 = h_2 * 0.3048                       # ft to m
    #     #logger.debug('converted units')
    #     # calculate velocities
    #     # # if flow is different at input/outflow, probably pass v_1 and v_2 through the function instead
    #     # v_1 = calc_v(discharge,a)
    #     # v_2 = calc_v(discharge,a)
    #     # logger.debug('calculated velocity')
    #     # # calculate friction for total head loss
    #     # dynamic_viscosity = 0.0010016 # for water @ 20C
    #     density = 998.2 # kg/m^3 for water @ 20C
    #     # kinematic_viscosity = calc_k_viscosity(dynamic_viscosity, density)
    #     # friction = calc_friction(K, ps_diameter, v_1, kinematic_viscosity)
        
    #     # # calculate total head loss
    #     # head_loss = calc_h_loss(friction, ps_length, ps_diameter, v_1)
        
    #     # calculate pressure at p2
    #     p_atm = constants.atm
    #     p_2 = calc_p_2(p_atm, density, h_D)
        
    #     # calculate presure at p1
    #     #p_1 = calc_p_1(p_2, fish_depth, h_2, density, v_1, v_2, head_loss)
    #     p_1 = calc_p_2(p_atm, density, fish_depth)
        
    #     # calculate pressure ratio
    #     p_ratio = p_1/p_2
        
    #     # calculate survival rate
    #     endpoint = baro_surv_prob(p_ratio, beta_0, beta_1)

    #     return scalarize(endpoint)

    def node_surv_rate(self,
                       length,
                       u_crit,
                       status,
                       surv_fun,
                       route,
                       surv_dict,
                       u_param_dict,
                       barotrauma = False):
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
        def scalarize(x):
            if isinstance(x, (np.ndarray, list)) and len(x) == 1:
                return x[0]
            if hasattr(x, "item") and np.ndim(x) == 0:
                return x.item()
            return x
        
        # Patch all scalar args
        length = scalarize(length)
        u_crit = scalarize(u_crit)
        status = scalarize(status)
        surv_fun = scalarize(surv_fun)
        route = scalarize(route)

    
        if status == 0:
            return 0.0
        else:
            if surv_fun == 'a priori':
                try:
                    prob = surv_dict[route]
                except KeyError as e:
                    logger.debug(f'Problem with a priori survival function for {route}: {e}')
                    logger.debug(f'Available routes in surv_dict: {list(surv_dict.keys())}')
                    prob = 1.
                except Exception as e:
                    logger.error(f'Unexpected error in a priori survival for {route}: {e}')
                    prob = 1.
    
            else:
                param_dict = u_param_dict[route]

                ''' Impingement is a function of head width, rack spacing, 
                critical swim speed, and intake velocity.  Essentially, fish that 
                are too wide to fit through the rack, but too slow to escape the
                intake velocity are impinged, and we assume impingement is death'''
                intake_vel = param_dict['intake_vel']
                rack_spacing = param_dict['rack_spacing']
                if length/10. > rack_spacing and u_crit < intake_vel:
                    imp_surv_prob = 0.
                else:
                    imp_surv_prob = 1.

                #logger.debug('calculated impingement survival')
                
                # if survival is assessed at a Kaplan turbine:
                if surv_fun == 'Kaplan':
                    # calculate the probability of strike as a function of the length of the fish and turbine parameters
                    strike_surv_prob = self.Kaplan(length, param_dict)
    
                # if survival is assessed at a Propeller turbine:
                elif surv_fun == 'Propeller':
                    # calculate the probability of strike as a function of the length of the fish and turbine parameters
                    strike_surv_prob = self.Propeller(length, param_dict)
    
                # if survival is assessed at a Francis turbine:
                elif surv_fun == 'Francis':
                    # calculate the probability of strike as a function of the length of the fish and turbine parameters
                    strike_surv_prob = self.Francis(length, param_dict)
    
                # if survival is assessed at a turbine in pump mode:
                elif surv_fun == 'Pump':
                    # calculate the probability of strike as a function of the length of the fish and turbine parameters
                    strike_surv_prob = self.Pump(length, param_dict)
                    
                
                if barotrauma == True:
                    # Validate required barotrauma parameters are present
                    try:
                        if route not in self.unit_params.index:
                            logger.warning(f'Route {route} not found in unit_params, skipping barotrauma calculation')
                            baro_surv = 1.0
                        elif pd.isna(self.unit_params.loc[route, 'fb_depth']) or pd.isna(self.unit_params.loc[route, 'submergence_depth']):
                            logger.warning(f'Missing barotrauma parameters for {route}: fb_depth or submergence_depth is NaN')
                            baro_surv = 1.0
                        else:
                            # get constants
                            g = constants.g
                            p_atm = constants.atm
                            density = 998.2 # kg/m^3 for water @ 20C
                            
                            vertical_habitat_value = self.pop['vertical_habitat'].item()
                            if vertical_habitat_value == 'Pelagic':
                                d_1 = 0.01
                                d_2 = 0.33
                            elif vertical_habitat_value == 'Benthic':
                                d_1 = 0.8
                                d_2 = 1
                            else:
                                d_1 = 0.01
                                d_2 = 1
                                
                            # get regression slope and intercept (beta 1 and beta 0)
                            beta_0 = self.pop['beta_0'].item()
                            beta_1 = self.pop['beta_1'].item()
                            
                            # get forebay depth and create depth range for habitat preference
                            # Note: fb_depth already converted to feet in __init__ if metric
                            depth_1 = self.unit_params['fb_depth'][route] * d_1
                            depth_2 = self.unit_params['fb_depth'][route] * d_2
                            fish_depth = np.random.uniform(depth_1,depth_2,1)[0]
                            
                            # get submergence depth (already in feet if metric conversion applied)
                            h_D = self.unit_params['submergence_depth'][route]
                            
                            # Convert depths from feet to meters for pressure calculation
                            fish_depth_m = fish_depth * 0.3048
                            h_D_m = h_D * 0.3048
                            
                            # calculate pressure ratio using SI units
                            p_1 = p_atm + density*g*fish_depth_m
                            p_2 = p_atm + density*g*h_D_m
                            p_ratio = p_1/p_2
                            
                            # calculate injury/mortality probability from barotrauma
                            # Note: baro_injury_prob returns P(injury), not P(survival)
                            baro_injury = baro_injury_prob(p_ratio, beta_0, beta_1)
            
                            # survival probability considering blade strike and barotrauma
                            baro_surv = 1.0 - baro_injury
                    except Exception as e:
                        logger.error(f'Error calculating barotrauma for route {route}: {e}')
                        baro_surv = 1.0  
                else:
                    baro_surv = 1.
                
                #logger.debug('calculated barotrauma survival')
                # incoporate latent mortality
                #latent_survival = beta.rvs(1.02, 0.371, size=1)[0]
                latent_survival = 1.
                #logger.debug('calculated latent survial')
                
                # calculate turbine survival estimate
                prob = imp_surv_prob * strike_surv_prob * baro_surv * latent_survival
                prob = scalarize(prob)
            try:
                return np.float32(prob)
            except (ValueError, TypeError) as e:
                logger.error(f'Cannot convert probability {prob} to float32: {e}')
                return np.float32(1.0)  # Default to 100% survival on conversion error
    
    # create function that builds networkx graph object from nodes and edges in project database
    def create_route(self):
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
    
        if getattr(self, 'graph', None) is not None:
            # self.graph already exists, so no need to rebuild.
            pass
        else:
            nodes = self.nodes
            edges = self.edges
        
            # create empty route object
            route = nx.route = nx.DiGraph()
        
            # add nodes to route - nodes.loc.values
            route.add_nodes_from(nodes.Location.values)
            
            if len(nodes) > 1:
                # create edges - iterate over edge rows to create edges
                weights = []
                for i in edges.iterrows():
                    _from = i[1]['_from']
                    _to = i[1]['_to']
                    weight = i[1]['weight']
                    route.add_edge(_from,_to,weight = weight)
                    weights.append(weight)
                    
                # identify the number of moves that a fish can make
                path_list = nx.all_shortest_paths(route,'river_node_0','river_node_%s'%(self.max_river_node))
    
                max_len = 0
                for i in path_list:
                    path_len = len(i)
                    if path_len > max_len:
                        max_len = path_len
                self.moves = np.arange(0,max_len + 1,1)
            else:
                self.moves = np.zeros(1, dtype = np.int32)
        
            # return finished product and enjoy functionality of networkx
            self.graph = route
 
    def movement(self,
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
        capabilities, and operational sequencing.
        """
        curr_Q = Q_dict['curr_Q']
        min_Q_dict = Q_dict['min_Q']
        sta_cap_dict = Q_dict['sta_cap']
        env_Q_dict = Q_dict['env_Q']
        bypass_Q_dict = Q_dict['bypass_Q']
    
        if status != 1:
            return location  # Fish is dead
    
        nbors = np.array(list(graph.neighbors(location)), dtype=str)
        #logger.debug(f'neighbors: {nbors}')
        if nbors.size == 0:
            return location

        locs = []
        probs = []
    

        contains_U = np.char.find(nbors, 'U') >= 0
        found_spill = np.char.find(nbors, 'spill') >= 0
    
        if np.any(contains_U):
            for i in nbors:
                if 'U' in i:
                    # Resolve facility
                    try:
                        facility = unit_fac_dict[i]
                    except KeyError:
                        try:
                            facility = self.unit_params.at[i, 'Facility']
                        except Exception:
                            continue  # skip node if not found
    
                    sta_cap = sta_cap_dict.get(facility, 0.0)
                    min_Q = min_Q_dict.get(facility, 0.0)
                    env_Q = env_Q_dict.get(facility, 0.0)
                    bypass_Q = bypass_Q_dict.get(facility, 0.0)
    
                    # Determine usable production flow
                    if curr_Q > min_Q:
                        excess = curr_Q - (sta_cap + env_Q + bypass_Q)
                        prod_Q = max(curr_Q - env_Q - bypass_Q - max(excess, 0), 0.0)
                    else:
                        prod_Q = 0.0
    
                    # Reintroduce operation order logic
                    unit_cap = Q_dict.get(i, 0.0)
                    order = op_order[i]
                    prev_units = [
                        u for u in op_order
                        if unit_fac_dict.get(u, None) == facility and op_order[u] < order
                    ]
                    prev_Q = sum(Q_dict.get(pu, 0.0) for pu in prev_units)
    
                    if prev_Q >= prod_Q:
                        u_Q = 0.0  # Not enough flow left for this unit
                    else:
                        u_Q = min(prod_Q - prev_Q, unit_cap)
    
                    prob = u_Q / curr_Q if curr_Q > 0 else 0.0
                    locs.append(i)
                    probs.append(prob)
    
                else:  # Bypass path
                    facility = unit_fac_dict.get(i, None)
                    bypass_Q = bypass_Q_dict.get(facility, 0.0)
                    prob = bypass_Q / curr_Q if curr_Q > 0 else 0.0
                    locs.append(i)
                    probs.append(prob)
    
        elif np.any(found_spill):
            facilities = self.facility_params[self.facility_params.Spillway.isin(nbors)].index
            total_sta_cap = sum(sta_cap_dict.get(f, 0.0) for f in facilities)
            total_env_Q = sum(env_Q_dict.get(f, 0.0) for f in facilities)
            total_bypass_Q = sum(bypass_Q_dict.get(f, 0.0) for f in facilities)

            for i in nbors:
                #logger.debug(f"neighbor: {i}")
                if 'U' in i:  # Only unit nodes have min_Q
                    min_Q = min_Q_dict.get(i, 0.0)
                    if curr_Q <= min_Q:
                        prob = 0.0

                elif 'spill' in i:
                    # Handle spill logic independently
                    if curr_Q <= min_Q:
                        prob = 1.0
                    if curr_Q >= total_sta_cap + total_env_Q + total_bypass_Q:
                        spill_Q = curr_Q - total_sta_cap - total_bypass_Q
                        prob = max(spill_Q / curr_Q, 0.0)
                    else:
                        p_env = total_env_Q / curr_Q if curr_Q > 0 else 0.0
                        prob = p_env
                else:
                    prob = 1.0  # fallback/default for unknown node types?
    
                locs.append(i)
                probs.append(prob)

        else:
            # Fallback: edge weights from graph
            for i in nbors:
                locs.append(i)
                edge_weight = graph[location][i].get("weight", 1.0)
                probs.append(edge_weight)
    
        # Normalize probabilities
        locs = np.array(locs, dtype=str).flatten()  # force flat array of strings
        probs = np.array(probs, dtype=float).flatten()
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(len(probs)) / len(probs)
            
        # print("locs:", locs)
        # print("probs:", probs)
        # print("probs shape:", probs.shape)

        try:
            new_loc = np.random.choice(locs, p=probs).item()

        except Exception as e:
            print("Choice failed:", e)
            new_loc = location

        return str(new_loc)

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
        import logging
        logger = logging.getLogger("Stryke.create_hydrograph")
        flow_df = pd.DataFrame()
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] create_hydrograph called: discharge_type={discharge_type}, scen={scen}, scen_months={scen_months}", flush=True)
            
        scen_df = flow_scenarios_df[flow_scenarios_df.Scenario == scen]
        logger.info(f"create_hydrograph called with discharge_type={discharge_type}, scen={scen}, scen_months={scen_months}")
        logger.info(f"scen_df shape: {scen_df.shape}, columns: {scen_df.columns.tolist()}")
        #logger.info('scenario dataframe %s', scen_df.shape)
        # if the discharge type is hydrograph - import hydrography and transform using prorate factor
        if discharge_type == 'hydrograph':
            #logger.debug('hydrograph')
            #print ('discharge type: hydrograph')
            gage = str(scen_df.at[scen_df.index[0],'Gage'])
            prorate = scen_df.at[scen_df.index[0],'Prorate']
            flow_year = scen_df.at[scen_df.index[0],'FlowYear']
            
            # if a gage number is present, fetch the usgs gage data
            if sum(char.isdigit() for char in gage) > 1:
                df = self.get_USGS_hydrograph(gage, prorate, flow_year)
                print(f"[DIAG] USGS hydrograph DataFrame shape: {df.shape}, columns: {df.columns.tolist()}", flush=True)
                for i in scen_months:
                    month_df = df[df.month == i]
                    print(f"[DIAG] USGS hydrograph month {i}: rows={month_df.shape[0]}", flush=True)
                    flow_df = pd.concat([flow_df, month_df])
            
            # if not, use the hydrograph data in the Hydrology sheet
            else:
                df = self.input_hydrograph_df.copy()
                logger.info(f"Hydrograph input DataFrame columns: {df.columns.tolist()}")
                logger.info(f"Hydrograph input DataFrame head:\n{df.head()}\nShape: {df.shape}")
                # If 'Discharge' and 'Date' columns exist, use them (Hydrology sheet)
                if 'Discharge' in df.columns and 'Date' in df.columns:
                    df['DAvgFlow_prorate'] = df['Discharge'] * prorate
                    df['datetimeUTC'] = pd.to_datetime(df['Date'])
                    logger.info("Hydrology sheet detected. Applied proration and datetime conversion.")
                # If 'DAvgFlow_prorate' and 'datetimeUTC' exist, use them (uploaded hydrograph)
                elif 'DAvgFlow_prorate' in df.columns and 'datetimeUTC' in df.columns:
                    logger.info("Webapp hydrograph detected. Columns already processed.")
                else:
                    logger.error(f"Hydrograph DataFrame missing required columns. Columns: {df.columns.tolist()}")
                    raise KeyError("Hydrograph DataFrame must have either ['Discharge', 'Date'] or ['DAvgFlow_prorate', 'datetimeUTC'] columns.")
                # extract year
                df['year'] = pd.DatetimeIndex(df['datetimeUTC']).year
                logger.info(f"Filtering hydrograph for flow_year={flow_year}")
                df = df[df['year'] == flow_year]
                logger.info(f"After year filter: shape={df.shape}")
                # get months
                df['month'] = pd.DatetimeIndex(df['datetimeUTC']).month.astype(int)
                scen_months_int = [int(m) for m in scen_months]
                logger.info(f"Filtering months: {scen_months_int}")
                for i in scen_months_int:
                    month_df = df[df['month'] == i]
                    logger.info(f"Month {i}: rows={month_df.shape[0]}")
                    flow_df = pd.concat([flow_df, month_df])
                logger.info(f"Final flow_df shape: {flow_df.shape}")
            #print (flow_df)
        
        # if it is a fixed discharge - simulate a hydrograph
        elif discharge_type == 'fixed':
            #print ('discharge type: fixed')
            day_in_month_dict = {1:31,2:28,3:31,
                                 4:30,5:31,6:30,
                                 7:31,8:31,9:30,
                                 10:31,11:30,12:31}
            sim_hydro_dict = {}
            print(f"[DIAG] create_hydrograph: fixed discharge={fixed_discharge}", flush=True)
            
            # for every month 
            for month in scen_months:
                days = day_in_month_dict[month]
                for day in np.arange(1,days+1,1):
                    date = "2023-" + str(month) + "-" + str(day)
                    sim_hydro_dict[date] = fixed_discharge
                if DIAGNOSTICS_ENABLED:
                    print(f"[DIAG] Simulated hydrograph dict keys: {list(sim_hydro_dict.keys())[:5]} ...", flush=True)
                            
                df = pd.DataFrame.from_dict(sim_hydro_dict,orient = 'index')  
                df.reset_index(inplace = True, drop = False)
                df.rename(columns = {'index':'datetimeUTC',0:'DAvgFlow_prorate'},inplace = True)
                if DIAGNOSTICS_ENABLED:
                    print(f"[DIAG] Simulated hydrograph DataFrame shape: {df.shape}, columns: {df.columns.tolist()}", flush=True)
                df['month'] = pd.to_datetime(df.datetimeUTC).dt.month
                if np.any(df.DAvgFlow_prorate.values < 0):
                    logger.debug ('prorated daily average flow value not found')
                #flow_df = flow_df.append(df)
                flow_df = pd.concat([df, flow_df])
        
        # Validate that hydrograph is not empty
        if flow_df.empty:
            error_msg = (
                f"ERROR: Hydrograph is empty after filtering for scenario '{scen}'.\n"
                f"Requested months: {scen_months}, year: {flow_year if discharge_type == 'hydrograph' else 'N/A'}\n"
                f"Available data months: {self.input_hydrograph_df['datetimeUTC'].dt.month.unique().tolist() if 'datetimeUTC' in self.input_hydrograph_df.columns else 'Unknown'}\n"
                f"Available data years: {self.input_hydrograph_df['datetimeUTC'].dt.year.unique().tolist() if 'datetimeUTC' in self.input_hydrograph_df.columns else 'Unknown'}\n"
                f"Please ensure your hydrograph data covers the requested time period."
            )
            logger.error(error_msg)
            print(f"[DIAG][ERROR] {error_msg}", flush=True)
            raise ValueError(error_msg)
        
        logger.info(f"Returning hydrograph DataFrame with shape: {flow_df.shape}")
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
        if DIAGNOSTICS_ENABLED and VERBOSE_DIAGNOSTICS:
            print(f"[DIAG] daily_hours called: scenario={scenario}, Q_dict keys={list(Q_dict.keys())}", flush=True)


        ops_df = self.operating_scenarios_df[self.operating_scenarios_df.Scenario == scenario]
        #ops_df.set_index('Unit', inplace = True)
        facilities = ops_df.Facility.unique()
        
        try:
            seasonal_facs = self.facility_params[self.facility_params.Scenario == scenario]
        except (KeyError, AttributeError) as e:
            logger.debug(f'No Scenario column in facility_params or scenario {scenario} not found: {e}')
            seasonal_facs = self.facility_params
        except Exception as e:
            logger.warning(f'Unexpected error filtering facility_params by scenario: {e}')
            seasonal_facs = self.facility_params
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
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] daily_hours: facility={facility}, curr_Q={Q_dict['curr_Q']}, min_Q={Q_dict['min_Q'][facility]}, sta_cap={Q_dict['sta_cap'][facility]}", flush=True)
 
            curr_Q = Q_dict['curr_Q']   # current discharge
            min_Q = Q_dict['min_Q'][facility]     # minimum operating discharge
            sta_cap = Q_dict['sta_cap'][facility] # station capacity
            env_Q = Q_dict['env_Q'][facility]     # min environmental discharge 
            bypass_Q = Q_dict['bypass_Q'][facility] # how much discharge through downstream bypass sluice
            prod_Q = curr_Q - env_Q - bypass_Q # how much water is left to use for production

            fac_type = seasonal_facs.at[facility,'Operations']
            fac_units = self.unit_params[self.unit_params.Facility == facility]
            if not fac_units.index.equals(pd.RangeIndex(start=0, stop=len(fac_units), step=1)):
                fac_units.reset_index(drop=False, inplace=True)

            #fac_units.set_index('Unit', inplace = True)
            fac_units = fac_units.sort_values(by = 'op_order')
            #logger.debug('Facility Type is %s',fac_type)
            # if operations are modeled with a distribution 
            for i in fac_units.index:
                #logger.debug('Facility index is %s',i)
                if fac_type != 'run-of-river':
                    order = fac_units.at[i,'op_order']
                    # get log norm shape parameters
                    shape = ops_df.at[i,'shape']
                    location = ops_df.at[i,'location']
                    scale = ops_df.at[i,'scale']
                    
                    hours_operated[i] = lognorm.rvs(shape,location,scale,1000)
    
                    # flip a coin - see if this unit is running today
                    prob_not_operating = ops_df.at[i,'Prob_Not_Op']
                    
                    #if operations == 'independent':
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
                            
                else:
                    #logger.debug('start processing run of river facility unit %s',i)
                    at_capacity = False
                    cum_Q = 0.  # current cumulative discharge through the powerhouse
                    
                    # Don't set the index; just iterate over each unit row.
                    for idx, row in fac_units.iterrows():
                        # Assume each unit row has a unique identifier in a column (e.g., 'Unit')
                        # If ops_df has a matching row for each unit, you can merge or filter by that identifier.
                        #logger.debug('working on unit %s', row['Unit'])
                        # For example, if ops_df has a 'Unit' column:
                        unit_ops = ops_df[ops_df.Unit == row['Unit']]#[ops_df.Unit == idx
                        if unit_ops.empty:
                            logger.debug("No operations data found for unit %s", idx)
                            continue

                        hours = unit_ops.iloc[0]['Hours']  # pick the first matching row
                        u_cap = row['Qcap']
                        
                        # Determine the effective capacity limit
                        effective_cap = min(prod_Q, sta_cap)
                        
                        if cum_Q + u_cap <= effective_cap:
                            hours_dict[idx] = hours
                            flow_dict[idx] = u_cap * hours * 3600.
                            cum_Q += u_cap
                        else:
                            excess = cum_Q + u_cap - effective_cap
                            hours_dict[idx] = hours
                            flow_dict[idx] = (u_cap - excess) * hours * 3600.
                            at_capacity = True
                    
                        if at_capacity:  # Exit the loop if at capacity
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
        
    def population_sim(self,output_units, spc_df, curr_Q):
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
        if DIAGNOSTICS_ENABLED and VERBOSE_DIAGNOSTICS:
            print(f"[DIAG] population_sim called: output_units={output_units}, curr_Q={curr_Q}", flush=True)
            print(f"[DIAG] population_sim: spc_df shape={spc_df.shape}, columns={spc_df.columns.tolist()}", flush=True)


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

        # Check if the simulated rate is at least 10 times the maximum observed rate.
        if ent_rate[0] > 10 * max_ent_rate:
            # Compute the exact orders-of-magnitude difference.
            magnitude_diff = np.log10(ent_rate[0] / max_ent_rate)
            # Optionally, round up to the next whole number if you want to reduce more aggressively.
            magnitudes = np.ceil(magnitude_diff)
            # Reduce by the factor computed.
            ent_rate = np.abs(ent_rate / 10**magnitudes)
            # print("New entrainment rate of %s" % (round(ent_rate[0], 4)))

        # flow per day in relation to million cubic FEET
        Mft3 = (60 * 60 * 24 * curr_Q)/1000000
        # flow per day in relation to million cubic METERS
        Mm3 = Mft3 * 35.31469989
        
        daily_rate = Mft3 # we convert hydrograph to cfs on import, no need to convert
        # metric_units = ["cms","CMS"]
        # if output_units == 'metric':
        #     daily_rate = Mm3
        #     ent_rate = ent_rate / 35.31469989
        # else: 
        #     daily_rate = Mft3

        # calcualte sample size
        return np.round(daily_rate * ent_rate,0)[0]

    def run(self):
        """
        Executes a comprehensive simulation of fish populations navigating through
        a hydroelectric facility. Diagnostic print statements are included to help
        trace parameter values and workflow progress.
        """
        # Reopen HDF5 file for append mode (it was closed after webapp_import)
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] run() called. Reopening HDF5 file: {self.hdf_path}", flush=True)
        self.hdf = pd.HDFStore(self.hdf_path, mode='a')
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] HDF5 file reopened successfully. Current keys: {self.hdf.keys()}", flush=True)
        
        # Create route and associated data.
        self.create_route()
        #logger.debug('starting simulation')
        # Setup string size dictionary for formatting.
        str_size = {'species': 30}
        try:
            for i in self.moves:
                str_size['state_%s' % i] = 30
        except Exception as e:
            logger.debug("Error setting up string sizes:", e)
            str_size['state_0'] = 30
    
        # Initialize dictionaries for unit parameters.
        u_param_dict = {}
        surv_dict = {}
        intake_vel_dict = {}
        units = []
        op_order_dict = {}
        q_cap_dict = {}
        unit_fac_dict = {}
        #logger.debug('building unit dictionaries')
        #print("Setting up unit parameters...", flush=True)
        for index, row in self.unit_params.iterrows():
            unit = index
            unit_fac_dict[unit] = row['Facility']
            q_cap_dict[unit] = row['Qcap']
            runner_type = row['Runner Type']
            intake_vel_dict[unit] = row['intake_vel']
            units.append(unit)
            op_order_dict[unit] = row['op_order']
            rack_spacing = self.facility_params.at[row['Facility'],'Rack Spacing']
            penstock_D = row['ps_D'] #self.unit_params.at[row['Facility'],'ps_D']

            if np.isnan(penstock_D):
                barotrauma = False
            else:
                barotrauma = True
                
            # if np.isnan(rack_spacing):
            #     rack_spacing = 2 /12.
            
            if runner_type == 'Kaplan':
                param_dict = {'H': float(row['H']),
                              'RPM': float(row['RPM']),
                              'D': float(row['D']),
                              'ada': float(row['ada']),
                              'N': float(row['N']),
                              'Qopt': float(row['Qopt']),
                              'Qper': float(row['Qopt'] / row['Qcap']),
                              '_lambda': float(row['lambda']),
                              'intake_vel':float(row['intake_vel']),
                              'rack_spacing':float(rack_spacing)}
                u_param_dict[unit] = param_dict
            elif runner_type == 'Propeller':
                param_dict = {'H': float(row['H']),
                              'RPM': float(row['RPM']),
                              'D': float(row['D']),
                              'ada': float(row['ada']),
                              'N': float(row['N']),
                              'Qopt': float(row['Qopt']),
                              'Qper': float(row['Qopt'] / row['Qcap']),
                              '_lambda': float(row['lambda']),
                              'intake_vel':float(row['intake_vel']),
                              'rack_spacing':float(rack_spacing)}
                u_param_dict[unit] = param_dict
            elif runner_type == 'Francis':
                param_dict = {'H': float(row['H']),
                              'RPM': float(row['RPM']),
                              'D': float(row['D']),
                              'ada': float(row['ada']),
                              'N': float(row['N']),
                              'Qper': float(row['Qopt'] / row['Qcap']),
                              'iota': float(row['iota']),
                              'D1': float(row['D1']),
                              'D2': float(row['D2']),
                              'B': float(row['B']),
                              '_lambda': float(row['lambda']),
                              'intake_vel':float(row['intake_vel']),
                              'rack_spacing':float(rack_spacing)}
                u_param_dict[unit] = param_dict
        #("Completed unit parameters setup.", flush=True)
    
        # Create survival dictionary from nodes.
        for idx, row in self.nodes.iterrows():
            # Use the Location field as key (assuming ID equals Location).
            surv_dict[row['Location']] = row['Survival']
        #print("Survival dictionary created:", surv_dict, flush=True)
        #logger.debug('iterate over scenarios')
        # Iterate over each flow scenario.
        for scen in self.flow_scenarios:
            #print(f"Starting scenario {scen} now", flush=True)
            #logger.debug('start assessing scenario')
            try:
                scen_df = self.flow_scenarios_df[self.flow_scenarios_df['Scenario'] == scen]
            except Exception:
                logger.warning('Scenario not in flow scenarios dataframe')
            scen_num = scen_df.iat[0, scen_df.columns.get_loc('Scenario Number')]
            season = scen_df.iat[0, scen_df.columns.get_loc('Season')]
            scenario = scen_df.iat[0, scen_df.columns.get_loc('Scenario')]
            scen_months = scen_df.iat[0, scen_df.columns.get_loc('Months')]
    
            if scen_df.iat[0, scen_df.columns.get_loc('Flow')] == 'hydrograph':
                self.discharge_type = 'hydrograph'
            else:
                self.discharge_type = 'fixed'
    
            if type(scen_months) != np.int64:
                month_list = scen_months.split(",")
                scen_months = list(map(int, month_list))
            else:
                scen_months = [scen_months]
            try:
                ops = self.operating_scenarios_df[self.operating_scenarios_df['Scenario'] == scenario]
            except Exception:
                logger.warning('Scenario not in operating scenarios')
            
            species = self.pop.Species.unique() #[self.pop['Scenario'] == scenario].Species.unique()
            
            # Create hydrograph for this scenario.
            if self.discharge_type == 'hydrograph':
                flow_df = self.create_hydrograph(self.discharge_type, scen, scen_months, self.flow_scenarios_df)
            else:
                fixed_discharge = scen_df.iat[0, scen_df.columns.get_loc('Flow')]
                flow_df = self.create_hydrograph(self.discharge_type, scen, scen_months, self.flow_scenarios_df, fixed_discharge=fixed_discharge)

            # Diagnostics for flow_df (hydrograph)
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] flow_df shape: {flow_df.shape}")
                print(f"[DIAG] flow_df columns: {list(flow_df.columns)}")
                print(f"[DIAG] flow_df head:\n{flow_df.head()}")
                print(f"[DIAG] flow_df tail:\n{flow_df.tail()}")
                print(f"[DIAG] flow_df date range: {flow_df['datetimeUTC'].min()} to {flow_df['datetimeUTC'].max()}" if 'datetimeUTC' in flow_df.columns else "[DIAG] No datetimeUTC column in flow_df")
                print(f"[DIAG] flow_df DAvgFlow_prorate min/max: {flow_df['DAvgFlow_prorate'].min()} / {flow_df['DAvgFlow_prorate'].max()}" if 'DAvgFlow_prorate' in flow_df.columns else "[DIAG] No DAvgFlow_prorate column in flow_df")

            
            for spc in species:
                spc_dat = self.pop[(self.pop['Scenario'] == scenario) & (self.pop.Species == spc)]
                if spc_dat.empty:
                    continue
                
                #logger.info('Working on species %s',spc)
                # Extract lognormal parameters (in centimeters)
                s = spc_dat.iat[0, spc_dat.columns.get_loc('length shape')]
                len_loc = spc_dat.iat[0, spc_dat.columns.get_loc('length location')]
                len_scale = spc_dat.iat[0, spc_dat.columns.get_loc('length scale')]
                
                # extract length in ft
                mean_len = spc_dat.iat[0, spc_dat.columns.get_loc('Length_mean')]
                sd_len = spc_dat.iat[0, spc_dat.columns.get_loc('Length_sd')]
                
                # get other variables
                species_name = spc_dat.iat[0, spc_dat.columns.get_loc('Species')]
                iterations = spc_dat.iat[0, spc_dat.columns.get_loc('Iterations')]
                u_crit = spc_dat.iat[0, spc_dat.columns.get_loc('U_crit')]
                occur_prob = spc_dat.iat[0, spc_dat.columns.get_loc('occur_prob')]
                if math.isnan(occur_prob):
                    occur_prob = 1.0
        
                spc_length = pd.DataFrame()
                for i in np.arange(0, iterations, 1):
                    for flow_row in flow_df.iterrows():
                        curr_Q = flow_row[1]['DAvgFlow_prorate']
                        day = flow_row[1]['datetimeUTC']

                        if DIAGNOSTICS_ENABLED and VERBOSE_DIAGNOSTICS:
                            # Diagnostics for flow_row and curr_Q
                            print(f"[DIAG] curr_Q: {curr_Q}")

                        # Build Q_dict.
                        Q_dict = {'curr_Q': curr_Q}
                        min_Q_dict = {}
                        env_Q_dict = {}
                        bypass_Q_dict = {}
                        if 'Scenario' in self.facility_params.columns:
                            for index, row in self.facility_params[self.facility_params.Scenario == scenario].iterrows():
                                fac = index
                                min_Q_dict[fac] = row['Min_Op_Flow']
                                env_Q_dict[fac] = row['Env_Flow']
                                bypass_Q_dict[fac] = row['Bypass_Flow']
                        else:
                            for index, row in self.facility_params.iterrows():
                                fac = index
                                min_Q_dict[fac] = row['Min_Op_Flow']
                                env_Q_dict[fac] = row['Env_Flow']
                                bypass_Q_dict[fac] = row['Bypass_Flow']
                        Q_dict['min_Q'] = min_Q_dict
                        Q_dict['env_Q'] = env_Q_dict
                        Q_dict['bypass_Q'] = bypass_Q_dict
    
                        # Update Q_dict and sta_cap for units.
                        sta_cap = {}

                        for u in units:
                            u_param_dict[u]['Q'] = curr_Q
                            unit_df = self.unit_params.loc[[u]]
                            fac = unit_df.iat[0, unit_df.columns.get_loc('Facility')]
                            if fac not in sta_cap:
                                sta_cap[fac] = 0
                            Q_dict[u] = unit_df.iat[0, unit_df.columns.get_loc('Qcap')]
                            sta_cap[fac] += unit_df.iat[0, unit_df.columns.get_loc('Qcap')]
                            

                        Q_dict['sta_cap'] = sta_cap

                        tot_hours, tot_flow, hours_dict, flow_dict = self.daily_hours(Q_dict, scenario)
                        #logger.info('Q-Dict Built')
                        
                        if np.any(tot_hours > 0):
                            presence_seed = np.random.uniform(0, 1)
                            if occur_prob >= presence_seed:
                                if math.isnan(spc_dat.iat[0, spc_dat.columns.get_loc('shape')]):
                                    n = int(spc_dat.iat[0, spc_dat.columns.get_loc('Fish')])
                                else:
                                    n = self.population_sim(self.output_units, spc_dat, curr_Q)
                                if int(n) == 0:
                                    n = 1
    
                                try:
                                    if not math.isnan(s):
                                        population = np.abs(lognorm.rvs(s, len_loc, len_scale, int(n), random_state=rng))
                                        population = np.where(population > 150, 150, population)
                                        population = population * 0.0328084  # convert cm to feet
                                    else:
                                        print("Generating population using normal distribution", flush=True)
                                        population = np.abs(np.random.normal(mean_len, sd_len, int(n))) / 12.0                                    # print(f"Population of {len(population)} created for species {species_name} on day {day}", flush=True)
                                except Exception as e:

                                    continue
    
                                try:
                                    U_crit_val = spc_dat.iat[0, spc_dat.columns.get_loc('U_crit')]
                                except Exception as e:
                                    U_crit_val = 0
                                swim_speed = np.repeat(U_crit_val, len(population))
                                #logger.info('Population estimated')
                                if len(self.nodes) > 1:
                                    fishes = pd.DataFrame({
                                        'scenario_num': np.repeat(scen_num, int(n)),
                                        'species': np.repeat(species_name, int(n)),
                                        'flow_scenario': np.repeat(scenario, int(n)),
                                        'season': np.repeat(season, int(n)),
                                        'iteration': np.repeat(i, int(n)),
                                        'day': np.repeat(day, int(n)),
                                        'flow': np.repeat(curr_Q, int(n)),
                                        'population': np.float32(population),
                                        'state_0': np.repeat('river_node_0', int(n))
                                    })
                                else:
                                    fishes = pd.DataFrame({
                                        'scenario_num': np.repeat(scen_num, int(n)),
                                        'species': np.repeat(species_name, int(n)),
                                        'flow_scenario': np.repeat(scenario, int(n)),
                                        'season': np.repeat(season, int(n)),
                                        'iteration': np.repeat(i, int(n)),
                                        'day': np.repeat(day, int(n)),
                                        'flow': np.repeat(curr_Q, int(n)),
                                        'population': np.float32(population),
                                        'state_0': np.repeat(self.nodes.at[0, 'Location'], int(n))
                                    })
                                    
                                #logger.info('Starting movement')
                                
                                def scalarize(x):
                                    if isinstance(x, (list, np.ndarray)) and len(x) == 1:
                                        return x[0]
                                    if hasattr(x, "item") and np.ndim(x) == 0:
                                        return x.item()
                                    return x
                                
                                def safe_node_surv_rate(pop, swim, status, surv_fun, location, surv_dict, u_param_dict):
                                    try:
                                        pop = scalarize(pop)
                                        swim = scalarize(swim)
                                        status = scalarize(status)
                                        surv_fun = scalarize(surv_fun)
                                        location = scalarize(location)
                                        #logger.debug('scalarized variables')
                                        return self.node_surv_rate(pop, swim, status, surv_fun, location, surv_dict, u_param_dict, barotrauma = barotrauma)
                                    except Exception as e:
                                        print(f"Failed node_surv_rate at location={location} with error: {e}")
                                        raise
                                
                                for k in self.moves:
                                    #logger.info(f'start movement for node {k}')  
                                
                                    if k == 0:
                                        status_arr = np.repeat(1, int(n))
                                    else:
                                        status_arr = fishes[f'survival_{k-1}'].values
                                
                                    current_location = fishes[f'state_{k}'].values
                                    current_location = np.asarray(current_location).flatten()
                                
                                    #logger.info(f'current location: {current_location}')
                                
                                    def surv_fun_att(state, surv_fun_dict):
                                        return surv_fun_dict[state]['Surv_Fun']
                                
                                    v_surv_fun = np.vectorize(surv_fun_att, excluded=[1])
                                    try:
                                        surv_fun = v_surv_fun(current_location, self.surv_fun_dict)
                                    except Exception as e:
                                        logger.warning(f"Survival function fallback triggered: {e}")
                                        surv_fun = np.repeat("a priori", len(current_location))
                                
                                    dice = np.random.uniform(0.0, 1.0, int(n))
                                
                                    # Flatten inputs before vectorization
                                    population = np.asarray(population).flatten()
                                    swim_speed = np.asarray(swim_speed).flatten()
                                    status_arr = np.asarray(status_arr).flatten()
                                    surv_fun = np.asarray(surv_fun).flatten()
                                
                                    v_surv_rate = np.vectorize(safe_node_surv_rate, excluded=[5, 6])
                                    rates = v_surv_rate(population, swim_speed, status_arr, surv_fun, current_location, surv_dict, u_param_dict)
                                
                                    #logger.info('applied vectorized survival rate')
                                    survival = np.where(dice <= rates, 1, 0)
                                
                                    if k < max(self.moves):
                                        def safe_movement(location, status, speed):
                                            location = scalarize(location)
                                            status = scalarize(status)
                                            speed = scalarize(speed)
                                            return self.movement(location, status, speed,
                                                                 self.graph, intake_vel_dict, Q_dict,
                                                                 op_order_dict, q_cap_dict, unit_fac_dict)
                                
                                        v_movement = np.vectorize(safe_movement)
                                        move = v_movement(current_location, survival, swim_speed)
                                    else:
                                        move = current_location
                                
                                    #logger.info('applied vectorized movement')
                                
                                    fishes[f'draw_{k}'] = np.float32(dice)
                                    fishes[f'rates_{k}'] = np.float32(rates)
                                    fishes[f'survival_{k}'] = np.float32(survival)
                                
                                    if k < max(self.moves):
                                        fishes[f'state_{k+1}'] = np.asarray(move).astype(str)
                                
                                    #logger.info('finished movement iteration')
                                
                                #logger.info('Finished movement')

   
                                max_string_lengths = fishes.select_dtypes(include=['object']).apply(lambda x: x.str.len().max())
                                fishes.to_hdf(self.hdf,
                                              key=f'simulations/{scen}/{spc}',
                                              mode='a',
                                              format='table',
                                              append=True,
                                              min_itemsize=20)
                                self.hdf.flush()
    
                                if self.output_units == 'metric':
                                    curr_Q_report = curr_Q * 0.02831683199881
                                else:
                                    curr_Q_report = curr_Q
                                daily_row_dict = {
                                    'species': ['{:50}'.format(spc)],
                                    'scenario': ['{:50}'.format(scenario)],
                                    'season': ['{:50}'.format(season)],
                                    'iteration': [np.int64(i)],
                                    'day': [pd.to_datetime(day)],
                                    'flow': [np.float64(curr_Q_report)],
                                    'pop_size': [np.int64(len(fishes))]
                                }
                                # Identify state and survival columns.
                                state_columns = sorted([col for col in fishes.columns if col.startswith('state_')])
                                survival_columns = sorted([col for col in fishes.columns if col.startswith('survival_')])
                                
                                # Convert state columns to Unicode strings.
                                state_vals = fishes[state_columns].to_numpy(dtype='U50')
                                # Use np.char.find to detect any uppercase 'U' in each state.

                                # Mask: where 'U' is found per state column per row
                                mask = np.char.find(state_vals, 'U') >= 0
                                entrained = mask.any(axis=1)
                                
                                # Safer way to find first index of 'U' in each row
                                first_U_index = np.full(len(fishes), -1)
                                first_U_index[entrained] = mask[entrained].argmax(axis=1)
                                
                                # Access survival values only where entrained
                                survival_vals = fishes[survival_columns].values
                                survived = np.zeros(len(fishes), dtype=bool)
                                
                                valid_indices = first_U_index[entrained]
                                survived[entrained] = survival_vals[entrained, valid_indices] == 1

                                fishes['is_entrained'] = entrained
                                fishes['survived_entrainment'] = survived
                                total_entrained = entrained.sum()
                                total_survived_entrained = survived.sum()
                                daily_row_dict['num_entrained'] = total_entrained
                                daily_row_dict['num_survived'] = total_survived_entrained
    
                                daily = pd.DataFrame.from_dict(daily_row_dict, orient='columns')
                                
                                # Print daily summary for user tracking
                                if DIAGNOSTICS_ENABLED:
                                    print(f"Day {day} | Flow: {curr_Q_report:.1f} | Entrained: {total_entrained} | Survived: {total_survived_entrained}", flush=True)
                                
                                if DIAGNOSTICS_ENABLED and VERBOSE_DIAGNOSTICS:
                                    print(f"[DIAG] About to write 'Daily' DataFrame: shape={daily.shape}, columns={list(daily.columns)}", flush=True)
                                    print(f"[DIAG] DataFrame head:\n{daily.head()}", flush=True)
                                
                                daily.to_hdf(self.hdf,
                                             key='Daily',
                                             mode='a',
                                             format='table',
                                             append=True)
                                
                                if DIAGNOSTICS_ENABLED and VERBOSE_DIAGNOSTICS:
                                    print(f"[DIAG] Wrote 'Daily' to HDF5. Flushing...", flush=True)
                                self.hdf.flush()

                                if DIAGNOSTICS_ENABLED and VERBOSE_DIAGNOSTICS:
                                    try:
                                        print(f"[DIAG] Current HDF5 keys after write: {self.hdf.keys()}", flush=True)
                                    except Exception as e:
                                        print(f"[DIAG] Could not retrieve HDF5 keys: {e}", flush=True)
                                
                            else:
                                if self.output_units == 'metric':
                                    curr_Q_report = curr_Q * 0.02831683199881
                                else:
                                    curr_Q_report = curr_Q
                                daily_row_dict = {
                                    'species': ['{:50}'.format(spc)],
                                    'scenario': ['{:50}'.format(scenario)],
                                    'season': ['{:50}'.format(season)],
                                    'iteration': [np.int64(i)],
                                    'day': [pd.to_datetime(day)],
                                    'flow': [np.float64(curr_Q_report)],
                                    'pop_size': [np.int64(0)],
                                    'num_entrained': [np.int64(0)],
                                    'num_survived': [np.int64(0)]
                                }
                                daily = pd.DataFrame.from_dict(daily_row_dict, orient='columns')
                                
                                # Print daily summary for user tracking
                                if DIAGNOSTICS_ENABLED:
                                    print(f"Day {day} | Flow: {curr_Q_report:.1f} | No fish present", flush=True)
                                
                                if DIAGNOSTICS_ENABLED and VERBOSE_DIAGNOSTICS:
                                    print(f"[DIAG] About to write 'Daily' DataFrame (no fish): shape={daily.shape}, columns={list(daily.columns)}", flush=True)
                                    print(f"[DIAG] DataFrame head:\n{daily.head()}", flush=True)
                                daily.to_hdf(self.hdf,
                                             key='Daily',
                                             mode='a',
                                             format='table',
                                             append=True)
                                if DIAGNOSTICS_ENABLED and VERBOSE_DIAGNOSTICS:                                
                                    print(f"[DIAG] Wrote 'Daily' to HDF5 (no fish). Flushing...", flush=True)
                                #self.hdf.flush()
                                if DIAGNOSTICS_ENABLED and VERBOSE_DIAGNOSTICS:
                                    try:
                                        print(f"[DIAG] Current HDF5 keys after write: {self.hdf.keys()}", flush=True)
                                    except Exception as e:
                                        print(f"[DIAG] Could not retrieve HDF5 keys: {e}", flush=True)
                        
                        logger.info("Scenario %s Dat %s Iteration %s for Species %s complete",scenario,day,i,species_name)
                self.hdf.flush()
                logger.info("Completed Scenario %s for Species %s",scen,species)
                
            logger.info("Completed Simulations - view results")
            if DIAGNOSTICS_ENABLED:
                try:
                    print(f"[DIAG] Final HDF5 keys before close: {self.hdf.keys()}", flush=True)
                except Exception as e:
                    print(f"[DIAG] Could not retrieve HDF5 keys (file may be closed): {e}", flush=True)
            self.hdf.flush()
            self.hdf.close()


    def summary(self):
        """
        Summarizes the results of fish entrainment simulations stored in an HDF file.
        ...
        (documentation unchanged)
        """
        import os
        import pandas as pd
        import io
        from contextlib import redirect_stdout
        from scipy.stats import beta
    
        hdf_path = os.path.join(self.proj_dir, '%s.h5' % (self.output_name))
        
        # Use a context manager to open the HDF file for reading so it closes automatically.
        with pd.HDFStore(hdf_path, mode='r') as store:
            if DIAGNOSTICS_ENABLED:    
                print(f"[DIAG] Opened HDF5 file for summary: {hdf_path}", flush=True)
                print(f"[DIAG] HDF5 keys present: {store.keys()}", flush=True)
            # create some empty holders
            self.beta_dict = {}
    
            # get Population table
            pop = store['Population']
            species = pop.Species.unique()
    
            # get Scenarios
            scen = store['Flow Scenarios']
            scens = scen.Scenario.unique()
    
            # get units (if needed)
            units = store['Unit_Parameters'].index
            if DIAGNOSTICS_ENABLED:
                if '/Daily' in store.keys():
                    print(f"[DIAG] 'Daily' table found in HDF5. Shape: {store['Daily'].shape}", flush=True)
                else:
                    print(f"[DIAG][ERROR] 'Daily' table is missing from HDF5!", flush=True)
            self.daily_summary = store['Daily']
            self.daily_summary.iloc[:,6:] = self.daily_summary.iloc[:,6:].astype(float)
    
            # Print summary statistics header
            print("\n" + "="*60, flush=True)
            print("SIMULATION SUMMARY STATISTICS", flush=True)
            print("="*60, flush=True)
            
            logger.info("iterate through species and scenarios and summarize")
            for i in species:
                for j in scens:
                        try:
                            dat = store['simulations/%s/%s' % (j, i)]
                        except:
                            continue
    
                        # summarize species-scenario - whole project
                        whole_proj_succ = dat.groupby(by=['iteration','day'])['survival_%s' % (max(self.moves))]\
                            .sum().to_frame().reset_index(drop=False)\
                            .rename(columns={'survival_%s' % (max(self.moves)):'successes'})
                        whole_proj_count = dat.groupby(by=['iteration','day'])['survival_%s' % (max(self.moves))]\
                            .count().to_frame().reset_index(drop=False)\
                            .rename(columns={'survival_%s' % (max(self.moves)):'count'})
                        whole_summ = whole_proj_succ.merge(whole_proj_count)
                        whole_summ['prob'] = whole_summ['successes'] / whole_summ['count']
                        whole_summ.fillna(0, inplace=True)
                        
                        # give a summary of whole project survival
                        logger.info("==== Whole Project Survival Summary ====")

                        overall_mean = whole_summ['prob'].mean()
                        overall_min = whole_summ['prob'].min()
                        overall_max = whole_summ['prob'].max()
                        overall_std = whole_summ['prob'].std()
                        
                        # logger.info("Overall survival probability across all days & iterations:")
                        # logger.info("  Mean = %.4f, Min = %.4f, Max = %.4f, Std Dev = %.4f",
                        #            overall_mean, overall_min, overall_max, overall_std)
                        
                        #logger.info("Per-day survival probability summary:")
                        per_day = whole_summ.groupby('day')['prob'].agg(['mean', 'min', 'max', 'std']).reset_index()
                        
                        # for _, row in per_day.iterrows():
                        #     logger.info("  Day %s | Mean = %.4f, Min = %.4f, Max = %.4f, Std Dev = %.4f",
                        #                 row['day'], row['mean'], row['min'], row['max'], row['std'])

                        try:
                            # whole_params = beta.fit(whole_summ.prob.values)
                            # whole_median = beta.median(whole_params[0], whole_params[1], whole_params[2], whole_params[3])
                            # whole_std = beta.std(whole_params[0], whole_params[1], whole_params[2], whole_params[3])
                            # lcl = beta.ppf(0.025, a=whole_params[0], b=whole_params[1],
                            #                  loc=whole_params[2], scale=whole_params[3])
                            # ucl = beta.ppf(0.975, a=whole_params[0], b=whole_params[1],
                            #                  loc=whole_params[2], scale=whole_params[3])
                            
                            logger.info("==== Whole Project Beta Distribution Summary (Survival Probability) ====")

                            # Fit the beta distribution to survival probabilities
                            whole_params = beta.fit(whole_summ['prob'].values)
                            a, b, loc, scale = whole_params
                            
                            # Summary stats from the fitted distribution
                            mean = beta.mean(a, b, loc=loc, scale=scale)
                            std_dev = beta.std(a, b, loc=loc, scale=scale)
                            lcl = beta.ppf(0.025, a=a, b=b, loc=loc, scale=scale)
                            ucl = beta.ppf(0.975, a=a, b=b, loc=loc, scale=scale)
                            
                            # Log it cleanly
                            logger.info("Fitted Beta distribution parameters:")
                            logger.info("  alpha (a) = %.4f, beta (b) = %.4f, loc = %.4f, scale = %.4f", a, b, loc, scale)
                            logger.info("Distribution summary:")
                            logger.info("  Mean survival probability = %.4f", mean)
                            logger.info("  Std deviation (spread)     = %.4f", std_dev)
                            logger.info("  95%% CI from Beta fit       = [%.4f, %.4f]", lcl, ucl)
                            
                            self.beta_dict['%s_%s_%s' % (j, i, 'whole')] = [j, i, 'whole', mean, std_dev, lcl, ucl]
                        except:
                            continue
                        for l in self.moves:
                            if l > 0:
                                sub_dat = dat[dat['survival_%s' % (l-1)] == 1]
                            else:
                                sub_dat = dat
                            route_succ = sub_dat.groupby(by=['iteration','day','state_%s' % (l)])['survival_%s' % (l)]\
                                .sum().to_frame().reset_index(drop=False)\
                                .rename(columns={'survival_%s' % (l):'successes'})
                            route_count = sub_dat.groupby(by=['iteration','day','state_%s' % (l)])['survival_%s' % (l)]\
                                .count().to_frame().reset_index(drop=False)\
                                .rename(columns={'survival_%s' % (l):'count'})
                            route_summ = route_succ.merge(route_count)
                            route_summ['prob'] = route_summ['successes'] / route_summ['count']
                            states = route_summ['state_%s' % (l)].unique()
                            for m in states:
                                st_df = route_summ[route_summ['state_%s' % (l)] == m]
                                try:
                                    st_params = beta.fit(st_df.prob.values)
                                    st_median = beta.median(st_params[0], st_params[1], st_params[2], st_params[3])
                                    st_std = beta.std(st_params[0], st_params[1], st_params[2], st_params[3])
                                    lcl = beta.ppf(0.025, a=st_params[0], b=st_params[1],
                                                     loc=st_params[2], scale=st_params[3])
                                    ucl = beta.ppf(0.975, a=st_params[0], b=st_params[1],
                                                     loc=st_params[2], scale=st_params[3])
                                except (ValueError, RuntimeError) as e:
                                    logger.warning(f'Beta fitting failed for state {m}: {e}. Using default values.')
                                    st_median = 0.
                                    st_std = 0.
                                    lcl = 0.
                                    ucl = 0.
                                self.beta_dict['%s_%s_%s' % (j, i, m)] = [j,
                                                                          i,
                                                                          m,
                                                                          st_median,
                                                                          st_std, 
                                                                          lcl, 
                                                                          ucl]
                        logger.info("Fit beta distributions to states")
                        del dat
    
                self.beta_df = pd.DataFrame.from_dict(data=self.beta_dict, orient='index',
                                                       columns=['scenario number','species','state','survival rate','variance','ll','ul'])
                # try:
                #     self.daily_summary['day'] = self.daily_summary['day'].dt.tz_localize(None)
                # except:
                #     pass
    
                # Group daily_summary by year
                yearly_summary = self.daily_summary.groupby(['species','scenario','iteration'])[['pop_size','num_survived','num_entrained']].sum()
                yearly_summary.reset_index(inplace=True)
    
                # Build cumulative summary DataFrame from dictionary
                cum_sum_dict = {
                    'species': [],
                    'scenario': [],
                    'mean_yearly_entrainment': [],
                    'lcl_yearly_entrainment': [],
                    'ucl_yearly_entrainment': [],
                    '1_in_10_day_entrainment': [],
                    '1_in_100_day_entrainment': [],
                    '1_in_1000_day_entrainment': [],
                    'mean_yearly_mortality': [],
                    'lcl_yearly_mortality': [],
                    'ucl_yearly_mortality': [],
                    '1_in_10_day_mortality': [],
                    '1_in_100_day_mortality': [],
                    '1_in_1000_day_mortality': [],
                }
                for fishy in yearly_summary.species.unique():
                    for scen in yearly_summary.scenario.unique():
                        logger.debug('summarizing %s in %s',fishy,scen)
                        idat = yearly_summary[(yearly_summary.species == fishy) & (yearly_summary.scenario == scen)]
                        cum_sum_dict['species'].append(fishy)
                        cum_sum_dict['scenario'].append(scen)
                        day_dat = self.daily_summary[(self.daily_summary.species == fishy) & (self.daily_summary.scenario == scen)]
                        daily_counts = day_dat.num_entrained.values
                        n_actual_days = len(daily_counts)
                        if n_actual_days == 0:
                            cum_sum_dict['mean_yearly_entrainment'].append(0.)
                            cum_sum_dict['lcl_yearly_entrainment'].append(0.)
                            cum_sum_dict['ucl_yearly_entrainment'].append(0.)
                            cum_sum_dict['1_in_10_day_entrainment'].append(0.)
                            cum_sum_dict['1_in_100_day_entrainment'].append(0.)
                            cum_sum_dict['1_in_1000_day_entrainment'].append(0.)
                            cum_sum_dict['mean_yearly_mortality'].append(0.)
                            cum_sum_dict['lcl_yearly_mortality'].append(0.)
                            cum_sum_dict['ucl_yearly_mortality'].append(0.)
                            cum_sum_dict['1_in_10_day_mortality'].append(0.)
                            cum_sum_dict['1_in_100_day_mortality'].append(0.)
                            cum_sum_dict['1_in_1000_day_mortality'].append(0.)
                            continue
                        df = day_dat[['day','iteration','num_entrained','num_survived']]
                        if 'num_mortalities' not in df.columns:
                            df['num_mortalities'] = df['num_entrained'] - df['num_survived']
                            
                        # Assuming bootstrap_mean_ci and summarize_ci are available
                        entrained_mean, entrained_lower, entrained_upper = bootstrap_mean_ci(df['num_entrained'])
                        killed_mean, killed_lower, killed_upper = bootstrap_mean_ci(df['num_mortalities'])
                        
                        # calculate extreme 1 in x day values
                        return_periods = [10, 100, 1000]
                        quantile_levels = {T: 1 - 1/T for T in return_periods}
                        extreme_entrained = {T: df['num_entrained'].quantile(q) for T, q in quantile_levels.items()}
                        extreme_killed = {T: df['num_mortalities'].quantile(q) for T, q in quantile_levels.items()}
                        
                        # calculate year total 
                        year_tots = df.groupby(['iteration'])[['num_entrained','num_mortalities']].sum()
                        
                        # summarize
                        summary = year_tots.apply(summarize_ci)
                        cum_sum_dict['mean_yearly_entrainment'].append(summary.at['mean','num_entrained'])
                        cum_sum_dict['lcl_yearly_entrainment'].append(summary.at['lower_95_CI','num_entrained'])
                        cum_sum_dict['ucl_yearly_entrainment'].append(summary.at['upper_95_CI','num_entrained'])
                        cum_sum_dict['1_in_10_day_entrainment'].append(extreme_entrained[10])
                        cum_sum_dict['1_in_100_day_entrainment'].append(extreme_entrained[100])
                        cum_sum_dict['1_in_1000_day_entrainment'].append(extreme_entrained[1000])
                        cum_sum_dict['mean_yearly_mortality'].append(summary.at['mean','num_mortalities'])
                        cum_sum_dict['lcl_yearly_mortality'].append(summary.at['lower_95_CI','num_mortalities'])
                        cum_sum_dict['ucl_yearly_mortality'].append(summary.at['upper_95_CI','num_mortalities'])
                        cum_sum_dict['1_in_10_day_mortality'].append(extreme_killed[10])
                        cum_sum_dict['1_in_100_day_mortality'].append(extreme_killed[100])
                        cum_sum_dict['1_in_1000_day_mortality'].append(extreme_killed[1000])

                logger.info("Yearly summary complete.")
    
                self.cum_sum = pd.DataFrame.from_dict(cum_sum_dict, orient='columns')
                # Debug print shapes
                logger.info("Beta DF shape: %s",self.beta_df.shape)
                summary = self.cum_sum.iloc[0]
                
                logger.info("==== Yearly Summary (Entrainment & Mortality Statistics) ====")
                
                # 1. Entrainment mean and CI
                logger.info("Entrainment: Mean = %.4f, 95%% CI = [%.4f, %.4f]",
                            summary['mean_yearly_entrainment'],
                            summary['lcl_yearly_entrainment'],
                            summary['ucl_yearly_entrainment'])
                
                # 2. Mortality mean and CI
                logger.info("Mortality: Mean = %.4f, 95%% CI = [%.4f, %.4f]",
                            summary['mean_yearly_mortality'],
                            summary['lcl_yearly_mortality'],
                            summary['ucl_yearly_mortality'])
                
                # 3–5. Entrainment rare events
                logger.info("Entrainment risk levels (probability of exceedance):")
                logger.info("  1-in-10 day:    %.4f", summary['1_in_10_day_entrainment'])
                logger.info("  1-in-100 day:  %.4f", summary['1_in_100_day_entrainment'])
                logger.info("  1-in-1000 day: %.4f", summary['1_in_1000_day_entrainment'])
                
                # 6–8. Mortality rare events
                logger.info("Mortality risk levels (probability of exceedance):")
                logger.info("  1-in-10 day:    %.4f", summary['1_in_10_day_mortality'])
                logger.info("  1-in-100 day:  %.4f", summary['1_in_100_day_mortality'])
                logger.info("  1-in-1000 day: %.4f", summary['1_in_1000_day_mortality']) 
                    
                # Optionally, write these DataFrames to Excel (if needed)
                # Only try if wks_dir is an Excel file (not HDF5)
                try:
                    if self.wks_dir and self.wks_dir.endswith(('.xlsx', '.xls')):
                        with pd.ExcelWriter(self.wks_dir, engine='openpyxl', mode='a') as writer:
                            self.beta_df.to_excel(writer, sheet_name='beta fit')
                            self.daily_summary.to_excel(writer, sheet_name='daily summary')
                            self.cum_sum.to_excel(writer, sheet_name='yearly summary')
                    else:
                        logger.info('Web app mode detected - Excel export skipped (results in HDF5 only)')
                except (PermissionError, FileNotFoundError) as e:
                    logger.info(f'Cannot write to Excel file (likely web app run): {e}')
                except Exception as e:
                    logger.warning(f'Unexpected error writing to Excel: {e}')
    
            # Close summary statistics section
            print("="*60 + "\n", flush=True)
    
        # At this point the HDFStore opened in read mode is closed.
        # Now open the file in append mode to write the summary DataFrames.
        with pd.HDFStore(hdf_path, mode='a') as store:
            store.put("Daily_Summary", self.daily_summary, format="table", data_columns=True)
            store.put("Beta_Distributions", self.beta_df, format="table", data_columns=True)
            store.put("Yearly_Summary", self.cum_sum, format="table", data_columns=True)
    
        return
                
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
                except (KeyError, ValueError, IndexError) as e:
                    logger.warning(f'Failed to load stream gage {i}: {e}')
                    continue
                except Exception as e:
                    logger.error(f'Unexpected error loading stream gage {i}: {e}')
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