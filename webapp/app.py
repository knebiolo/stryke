# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:48:03 2025

@author: Kevin.Nebiolo
"""
import os
import subprocess
import sys
import shutil
import threading
import queue
#import datetime
import io
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, after_this_request, send_from_directory, session, Response, g
from flask import jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import zipfile
import time
import uuid
import re
import pandas as pd
from io import StringIO
import networkx as nx
from networkx.readwrite import json_graph
from datetime import timedelta
import numpy as np
from contextlib import redirect_stdout
import json

# Manually tell pyproj where PROJ is installed
os.environ["PROJ_DIR"] = "/usr"
os.environ["PROJ_LIB"] = "/usr/share/proj"
os.environ["PYPROJ_GLOBAL_CONTEXT"] = "ON"

try:
    import pyproj
except ImportError:
    subprocess.run(["pip", "install", "--no-cache-dir", "pyproj"])
    import pyproj
    
# Explicitly add the parent directory of Stryke to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Add the Stryke directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Stryke")))

# Import Stryke components
from Stryke import stryke
from Stryke.stryke import epri

# Create a global log queue
LOG_QUEUE = queue.Queue()

# A custom stream object that writes messages to the queue
class QueueStream:
    def __init__(self, q):
        self.q = q
    def write(self, message):
        if message.strip():
            self.q.put(message)
    def flush(self):
        pass

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['PASSWORD'] = 'expensive5rudabega!@1'  # Set your desired password here

# Set session lifetime to 1 day (adjust as needed)
app.permanent_session_lifetime = timedelta(days=1)

@app.before_request
def make_session_permanent():
    session.permanent = True

# ----------------- Password Protection -----------------


@app.before_request
def require_login_and_setup():
    # First, enforce login for protected endpoints.
    if not session.get('logged_in') and request.endpoint not in ['login', 'static', 'health']:
        return redirect(url_for('login'))
    
    # Only set up session directories if the user is logged in.
    if session.get('logged_in'):
        # Generate a unique directory identifier if one doesn't exist.
        if 'user_dir' not in session:
            session['user_dir'] = uuid.uuid4().hex

        # Create a session-specific upload directory.
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session['user_dir'])
        os.makedirs(user_upload_dir, exist_ok=True)

        # Create a session-specific simulation project directory.
        user_sim_folder = os.path.join(SIM_PROJECT_FOLDER, session['user_dir'])
        os.makedirs(user_sim_folder, exist_ok=True)

        # Store these directories in Flask's global context for easy access.
        g.user_upload_dir = user_upload_dir
        g.user_sim_folder = user_sim_folder

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == app.config['PASSWORD']:
            session['logged_in'] = True
            flash("Logged in successfully!")
            return redirect(url_for('index'))
        else:
            error = 'Invalid password. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    # Retrieve the user's unique directory from the session
    user_dir = session.get('user_dir')
    if user_dir:
        # Build the session-specific paths
        user_upload_dir = os.path.join(UPLOAD_FOLDER, user_dir)
        user_sim_folder = os.path.join(SIM_PROJECT_FOLDER, user_dir)
        # Clear the session-specific folders
        clear_folder(user_upload_dir)
        clear_folder(user_sim_folder)
    
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('login'))

# -------------------------------------------------------

# Define directories for uploads and simulation projects
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
SIM_PROJECT_FOLDER = os.path.join(os.getcwd(), 'simulation_project')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIM_PROJECT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/health")
def health():
    print("Health endpoint accessed")
    return "OK", 200

# def run_simulation(ws, wks, output_name):
#     old_stdout = sys.stdout
#     sys.stdout = QueueStream(LOG_QUEUE)
#     try:
#         simulation_instance = stryke.simulation(ws, wks, output_name=output_name)
#         simulation_instance.run()
#         simulation_instance.summary()
#     except Exception as e:
#         print(f"Error during simulation: {e}")
#     finally:
#         sys.stdout = old_stdout
#         LOG_QUEUE.put(None)

def run_simulation_in_background(ws, wks, output_name):
    old_stdout = sys.stdout
    sys.stdout = QueueStream(LOG_QUEUE)
    try:
        with stryke.simulation(ws, wks, output_name=output_name) as sim:
            sim.run()
            sim.summary()
            # sim.close() is automatically handled by __exit__

        output_file = os.path.join(ws, f"{output_name}.h5")
        if os.path.exists(output_file):
            print(f"Simulation output created: {output_file}")
        else:
            print("Error: Simulation output file was not created!")
    except Exception as e:
        print("Error during simulation:", e)
    finally:
        sys.stdout = old_stdout
        LOG_QUEUE.put("[Simulation Complete]")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_simulation():
    simulation_results = None
    output_filename = None
    if request.method == 'POST':
        if 'excel_file' not in request.files:
            flash('No file part in the request')
            return render_template('upload_simulation.html')
        
        file = request.files['excel_file']
        if file.filename == '':
            flash('No file selected')
            return render_template('upload_simulation.html')
        
        # Save the file in the session-specific upload directory
        up_file_path = os.path.join(g.user_upload_dir, file.filename)
        file.save(up_file_path)
        flash(f'File successfully uploaded: {file.filename}')
        
        # Copy file to session-specific simulation directory
        simulation_file_path = os.path.join(g.user_sim_folder, file.filename)
        shutil.copy(up_file_path, simulation_file_path)
        
        ws = g.user_sim_folder
        wks = file.filename
        output_name = 'Simulation_Output'
        
        try:
            simulation_thread = threading.Thread(
                target=run_simulation_in_background,
                args=(ws, wks, output_name)
            )
            simulation_thread.start()
            flash("Simulation started. Live log will appear below.")
            simulation_results = "Simulation is running..."
            output_filename = f"{output_name}.h5"
        except Exception as e:
            flash(f"Error starting simulation: {e}")
            return render_template('upload_simulation.html')
    
    return render_template('upload_simulation.html',
                           simulation_results=simulation_results,
                           output_filename=output_filename)

@app.route('/download_zip')
def download_zip():
    user_sim_folder = g.get("user_sim_folder", SIM_PROJECT_FOLDER)

    # Remove any old zip files.
    for fname in os.listdir(user_sim_folder):
        if fname.endswith(".zip"):
            old_path = os.path.join(user_sim_folder, fname)
            try:
                os.remove(old_path)
                print(f"Removed old ZIP: {old_path}")
            except Exception as e:
                print(f"Error removing old ZIP {old_path}: {e}")

    # Create a new ZIP file.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"simulation_results_{timestamp}.zip"
    zip_filepath = os.path.join(user_sim_folder, zip_filename)
    print(f"Creating ZIP file: {zip_filepath}")

    try:
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_name in os.listdir(user_sim_folder):
                # Skip files with these extensions (adjust as needed).
                if file_name.endswith((".hdf", ".h5", ".zip")):
                    continue
                file_path = os.path.join(user_sim_folder, file_name)
                if os.path.isfile(file_path):
                    try:
                        zipf.write(file_path, arcname=file_name)
                        print(f"Added to ZIP: {file_name}")
                    except Exception as e:
                        print(f"Skipping file {file_name} => {e}")
        print(f"ZIP file successfully created: {zip_filepath}")
    except Exception as e:
        print(f"Error creating ZIP file: {e}")
        flash("Failed to create ZIP file.")
        return redirect(url_for('some_error_page'))

    if os.path.exists(zip_filepath):
        return send_file(zip_filepath, as_attachment=True)
    else:
        flash("ZIP file not found.")
        return redirect(url_for('some_error_page'))


@app.route('/stream')
def stream():
    def event_stream():
        while True:
            message = LOG_QUEUE.get()
            if message is None:
                continue
            yield f"data: {message}\n\n"
            if message == "[Simulation Complete]":
                break
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/results')
def results():
    output_file = request.args.get('output_file')
    output_path = os.path.join(SIM_PROJECT_FOLDER, output_file)
    summary_text = "Simulation ran successfully. Please download the output file below."
    return render_template('results.html', output_file=output_file, summary=summary_text)

@app.route('/fit', methods=['GET', 'POST'])
def fit_distributions():
    summary_text = ""
    log_text = ""
    plot_filename = ""
    
    # Use the session-specific simulation folder if available; otherwise, fallback to the global folder.
    sim_folder = g.get("user_sim_folder", SIM_PROJECT_FOLDER)
    
    if request.method == 'POST':
        try:
            print("LOG: Received POST request for fitting.")
            old_stdout = sys.stdout
            mystdout = io.StringIO()
            sys.stdout = mystdout

            states = request.form.get('states', '').strip()
            plant_cap = request.form.get('plant_cap', '').strip()
            family = request.form.get('family', '').strip()
            genus = request.form.get('genus', '').strip()
            species = request.form.get('species', '').strip()
            month_str = request.form.get('month', '').strip()
            huc02_str = request.form.get('huc02', '').strip()
            huc04_str = request.form.get('huc04', '').strip()
            huc06_str = request.form.get('huc06', '').strip()
            huc08_str = request.form.get('huc08', '').strip()
            nidid = request.form.get('nidid', '').strip()
            river = request.form.get('river', '').strip()
            
            def parse_list(text):
                return [item.strip() for item in text.split(',') if item.strip()] if text else []
            
            try:
                month = [int(m) for m in parse_list(month_str)]
            except Exception:
                month = []
            try:
                huc02 = [int(x) for x in parse_list(huc02_str)]
            except Exception:
                huc02 = []
            try:
                huc04 = [int(x) for x in parse_list(huc04_str)]
            except Exception:
                huc04 = []
            try:
                huc06 = [int(x) for x in parse_list(huc06_str)]
            except Exception:
                huc06 = []
            try:
                huc08 = [int(x) for x in parse_list(huc08_str)]
            except Exception:
                huc08 = []
            
            filter_args = {}
            if states:
                filter_args["states"] = states
            if plant_cap:
                filter_args["plant_cap"] = plant_cap
            if family:
                filter_args["Family"] = family
            if genus:
                filter_args["Genus"] = genus
            if species:
                filter_args["Species"] = species
            if month:
                filter_args["Month"] = month
            if huc02:
                filter_args["HUC02"] = huc02
            if huc04:
                filter_args["HUC04"] = huc04
            if huc06:
                filter_args["HUC06"] = huc06
            if huc08:
                filter_args["HUC08"] = huc08
            if nidid:
                filter_args["NIDID"] = nidid
            if river:
                filter_args["River"] = river
            
            print(f"LOG: Extracted filter arguments - {filter_args}")

            try:
                print("LOG: Running EPRI fitting function...")
                fish = stryke.epri(**filter_args)
                fish.ParetoFit()
                fish.LogNormalFit()
                fish.WeibullMinFit()
                fish.LengthSummary()
                print("LOG: Fitting functions executed successfully.")
            except Exception as e:
                sys.stdout = old_stdout
                flash(f"Error during fitting: {e}")
                return render_template('fit_distributions.html', summary=summary_text, log_text=log_text, plot_filename=plot_filename)
            finally:
                captured_output = mystdout.getvalue()
                sys.stdout = old_stdout
    
            # Generate and save the first plot.
            plt.clf()
            fish.plot()
            plot_filename = 'fitting_results.png'
            plot_path = os.path.join(sim_folder, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            
            # Generate and save the histogram.
            plt.clf()
            plt.figure(figsize=(5, 3))
            plt.hist(fish.lengths.tolist(), bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel("Fish Length (cm)")
            plt.ylabel("Frequency")
            plt.title("Distribution of Fish Lengths")
            other_filename = 'fish_lengths.png'
            other_plot_path = os.path.join(sim_folder, other_filename)
            plt.savefig(other_plot_path)
            plt.close()
            
            summary_text = (
                "Distribution fitting complete for filters: "
                f"States: '{states}', Plant Capacity: '{plant_cap}', Family: '{family}', "
                f"Genus: '{genus}', Species: '{species}', Months: {month}, "
                f"HUC02: {huc02}, HUC04: {huc04}, HUC06: {huc06}, HUC08: {huc08}, "
                f"NIDID: '{nidid}', River: '{river}'."
            )
            
            try:
                # Write the summary output to a file in the session folder.
                report_text = fish.summary_output(sim_folder, dist='Log Normal')
            except Exception as e:
                report_text = f"Error generating detailed summary report: {e}"
            
            log_text = report_text
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_filename_full = os.path.join(sim_folder, "fitting_results_log.txt")
            with open(log_filename_full, "a") as log_file:
                log_file.write(f"{timestamp} - Query: {summary_text}\n")
                log_file.write(f"{timestamp} - Report: {report_text}\n")
                log_file.write("--------------------------------------------------\n")
            
            return render_template('fit_distributions.html', summary=summary_text, log_text=log_text, plot_filename=plot_filename)

        except Exception as e:
            sys.stdout = old_stdout
            error_message = f"ERROR: {e}"
            print(error_message)
            return render_template('fit_distributions.html', summary=error_message)
             
    else:
        return render_template('fit_distributions.html', summary=summary_text, log_text=log_text, plot_filename=plot_filename)

@app.route('/plot/<filename>')
def serve_plot(filename):
    # Use the session-specific simulation folder if available.
    sim_folder = g.get("user_sim_folder", SIM_PROJECT_FOLDER)
    file_path = os.path.join(sim_folder, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return "Plot not found", 404

    
@app.route("/plot_lengths")
def plot_lengths():
    lengths = session.get("lengths", [])
    filename = epri.plot_fish_lengths(lengths, SIM_PROJECT_FOLDER)
    return send_from_directory(SIM_PROJECT_FOLDER, filename)

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")



#----------------------------------------
@app.route('/create_project', methods=['GET', 'POST'])
def create_project():
    if request.method == 'POST':
        project_name = request.form.get('project_name')
        project_notes = request.form.get('project_notes')
        units = request.form.get('units')
        model_setup = request.form.get('model_setup')
        print(f'choice of units is {units}')
        
        # Save data to session or database
        session['project_name'] = project_name
        session['project_notes'] = project_notes
        session['units'] = units
        session['model_setup'] = model_setup
        session['proj_dir'] = g.user_sim_folder  # Set the project directory
        
        flash(f"Project '{project_name}' created successfully!")
        return redirect(url_for('flow_scenarios'))
    
    # For GET requests, render the project creation form
    return render_template('create_project.html')


def process_hydrograph_data(raw_data):
    # Read the pasted text as a tab-delimited file without a header
    df = pd.read_csv(StringIO(raw_data), delimiter="\t", header=None)
    
    # Try converting the first cell of the first row to datetime.
    # If it fails, assume the row is a header and drop it.
    try:
        pd.to_datetime(df.iloc[0, 0])
        header_present = False
    except Exception:
        header_present = True

    if header_present:
        df = df.drop(index=0).reset_index(drop=True)
    
    # Rename columns to match Stryke's schema:
    # Assume column 0 is the date and column 1 is the discharge.
    df.columns = ['datetimeUTC', 'DAvgFlow_prorate']
    
    # Convert the datetime column to proper datetime format.
    # This step can be adjusted to handle non-standard datetime formats.
    df['datetimeUTC'] = pd.to_datetime(df['datetimeUTC'], errors='coerce', infer_datetime_format=True)
    print ('hydrograph before to numeric:')
    print (df)    
    # Convert discharge values to numeric.
    df['DAvgFlow_prorate'] = pd.to_numeric(df['DAvgFlow_prorate'], errors='coerce')
    
    # Drop rows with invalid dates or missing discharge values.
    df.dropna(inplace=True)
    
    # Set the datetime column as the index if that is required downstream.
    #df.set_index('datetimeUTC', inplace=True)
    print ('hydrograph:')
    print (df)
    return df

@app.route('/flow_scenarios', methods=['GET', 'POST'])
def flow_scenarios():
    if request.method == 'POST':
        # Retrieve form fields
        scenario_type = request.form.get('scenario_type')
        scenario_name = request.form.get('scenario_name')
        scenario_number = request.form.get('scenario_number')
        season = request.form.get('season')
        months = request.form.get('months')
        
        # Depending on scenario type, get the appropriate data.
        if scenario_type == 'static':
            discharge = request.form.get('discharge')
            hydrograph_data = None
        else:
            discharge = None
            hydrograph_data = request.form.get('hydrograph_data')
        
        print("DEBUG: Received scenario_type:", scenario_type, flush=True)
        print("DEBUG: Received hydrograph_data:", hydrograph_data, flush=True)
        
        # Store form values in session.
        session['scenario_type'] = scenario_type
        session['scenario_name'] = scenario_name
        session['scenario_number'] = scenario_number
        session['season'] = season
        session['months'] = months
        session['discharge'] = discharge
        session['hydrograph_data'] = hydrograph_data
        
        # Build the flow scenario DataFrame.
        if scenario_type == 'hydrograph' and hydrograph_data and hydrograph_data.strip():
            # Process the pasted hydrograph data.
            df_hydro = process_hydrograph_data(hydrograph_data)
            units = session.get('units', 'metric')
            # Ensure the flow column is numeric.
            df_hydro['DAvgFlow_prorate'] = pd.to_numeric(df_hydro['DAvgFlow_prorate'], errors='coerce')
            if units == 'metric':
                df_hydro['DAvgFlow_prorate'] = df_hydro['DAvgFlow_prorate'] * 35.3147
            
            hydro_file_path = os.path.join(g.user_sim_folder, 'hydrograph.csv')
            df_hydro.to_csv(hydro_file_path, index=False)
            session['hydrograph_file'] = hydro_file_path
            
            # Extract flow_year from the hydrograph data.
            df_hydro.reset_index(inplace=True)
            flow_year = int(df_hydro.datetimeUTC.dt.year.iloc[0]) if not df_hydro.empty else None
            
            # Force the Flow column to be the string 'hydrograph'
            discharge_value = 'hydrograph'
            
            flow_scenario_df = pd.DataFrame({
                'Scenario': [scenario_name],
                'Scenario Number': [scenario_number],
                'Season': [season],
                'Months': [months],
                'Flow': [discharge_value],
                'Gage': [None],
                'FlowYear': [flow_year],
                'Prorate': [1]
            })
        else:
            # For static discharge scenario.
            if discharge and session.get('units', 'metric') == 'metric':
                discharge_converted = float(discharge) * 35.3147
            else:
                discharge_converted = discharge
            flow_scenario_df = pd.DataFrame({
                'Scenario': [scenario_name],
                'Scenario Number': [scenario_number],
                'Season': [season],
                'Months': [months],
                'Flow': [discharge_converted],
                'Gage': [None],
                'FlowYear': [None],
                'Prorate': [1]
            })
        
        # Debug print the flow scenario DataFrame.
        print("DEBUG: Flow scenario DataFrame:")
        print(flow_scenario_df, flush=True)
        
        # Store the flow scenario in session.
        session['flow_scenario'] = flow_scenario_df.to_dict(orient='records')
        flow_scenarios = session.get("flow_scenario", [])
        if isinstance(flow_scenarios, str):
            flow_scenarios = json.loads(flow_scenarios)
        
        flash("Flow Scenario saved successfully!")
        return redirect(url_for('facilities'))

    units = session.get('units', 'metric')
    return render_template('flow_scenarios.html', units=units)



# ----------------- New Code: Sync model_setup with simulation_mode -----------------
@app.before_request
def sync_simulation_mode():
    # If the project model setup has been defined, copy it to 'simulation_mode'
    if 'model_setup' in session:
        session['simulation_mode'] = session['model_setup']


@app.route('/facilities', methods=['GET', 'POST'])
def facilities():
    if request.method == 'POST':
        print("POST form data:", request.form)

        # Determine simulation mode and number of facilities.
        sim_mode = session.get('simulation_mode', 'multiple_powerhouses_simulated_entrainment_routing')
        num_facilities = 1 if sim_mode in ["single_unit_survival_only", "single_unit_simulated_entrainment"] \
                          else int(request.form.get('num_facilities', 1))

        facilities_data = []
        facility_units = {}  # Dictionary to store facility:units mapping

        # Retrieve the user's chosen unit system (either 'metric' or 'imperial').
        units = session.get('units', 'metric')

        for i in range(num_facilities):
            facility_name = request.form.get(f'facility_name_{i}')
            num_units = 1 if sim_mode in ["single_unit_survival_only", "single_unit_simulated_entrainment"] \
                        else int(request.form.get(f'num_units_{i}', 1))

            operations = request.form.get(f'operations_{i}')

            # Rack Spacing: user enters mm if metric, inches if imperial.
            rack_spacing_raw = request.form.get(f'rack_spacing_{i}')
            try:
                rack_spacing_val = float(rack_spacing_raw) if rack_spacing_raw else None
            except ValueError:
                rack_spacing_val = None

            if rack_spacing_val is not None:
                if units == 'metric':
                    # Convert from millimeters to feet:
                    # mm -> m: value/1000, then m -> ft: multiply by 3.28084
                    rack_spacing_converted = (rack_spacing_val / 1000.0) * 3.28084
                else:
                    # Convert inches to feet.
                    rack_spacing_converted = rack_spacing_val / 12.0
            else:
                rack_spacing_converted = None

            # Flow values: user enters values in cubic meters per second if metric.
            min_op_flow_raw = request.form.get(f'min_op_flow_{i}', 0)
            env_flow_raw = request.form.get(f'env_flow_{i}', 0)
            bypass_flow_raw = request.form.get(f'bypass_flow_{i}', 0)

            try:
                min_op_flow_val = float(min_op_flow_raw)
            except ValueError:
                min_op_flow_val = 0.0
            try:
                env_flow_val = float(env_flow_raw)
            except ValueError:
                env_flow_val = 0.0
            try:
                bypass_flow_val = float(bypass_flow_raw)
            except ValueError:
                bypass_flow_val = 0.0

            if units == 'metric':
                # Convert cubic meters per second to cubic feet per second.
                min_op_flow_converted = min_op_flow_val * 35.3147
                env_flow_converted = env_flow_val * 35.3147
                bypass_flow_converted = bypass_flow_val * 35.3147
            else:
                # If already in imperial (cubic feet per second), no conversion is needed.
                min_op_flow_converted = min_op_flow_val
                env_flow_converted = env_flow_val
                bypass_flow_converted = bypass_flow_val

            spillway = request.form.get(f'spillway_{i}', 'none')

            # Build a dictionary matching the expected Excel headers (the simulation reads columns B-I on the Facilities tab).
            facilities_data.append({
                "Facility": facility_name,
                "Operations": operations,
                "Rack Spacing": rack_spacing_converted,
                "Min_Op_Flow": min_op_flow_converted,
                "Env_Flow": env_flow_converted,
                "Bypass_Flow": bypass_flow_converted,
                "Spillway": spillway,
                "Units": num_units,
            })


            facility_units[facility_name] = num_units

        # Save the raw facilities data and the facility-units mapping into session for reporting.
        session['facilities_data'] = facilities_data
        session['facility_units'] = facility_units

        # Convert the facilities_data into a pandas DataFrame (with converted values) for simulation/reporting.
        import pandas as pd
        df_facilities = pd.DataFrame(facilities_data)
        # Enforce the expected column order.
        expected_columns = ['Facility', 'Operations', 'Rack Spacing', 'Min_Op_Flow', 'Env_Flow', 'Bypass_Flow', 'Spillway', 'Units']
        df_facilities = df_facilities[expected_columns]
        # Save a JSON-serialized version of the DataFrame into the session.
        session['facilities_dataframe'] = df_facilities.to_json(orient='records')

        flash(f"{num_facilities} facility(ies) saved successfully!")
        return redirect(url_for('unit_parameters'))  # Adjust for next page as needed.

    units = session.get('units', 'metric')
    scenario = session.get('scenario_name', 'Unknown Scenario')
    sim_mode = session.get('simulation_mode', 'multiple_powerhouses_simulated_entrainment_routing')
    return render_template('facilities.html', units=units, scenario=scenario, sim_mode=sim_mode)

@app.route('/unit_parameters', methods=['GET', 'POST'])
def unit_parameters():
    if request.method == 'POST':
        print("Received form data:")
        for key, value in request.form.items():
            print(f"{key} : {value}")

        # Merge form data into rows (each row represents one unit's parameters)
        rows = {}
        for key, value in request.form.items():
            parts = key.rsplit('_', 1)
            if len(parts) != 2:
                continue
            field_name, row_id = parts
            # Remove any trailing underscore and digits from field_name
            clean_field_name = re.sub(r'_\d+$', '', field_name)
            print(f"Key: {key} split into clean_field_name: {clean_field_name} and row_id: {row_id}")
            if row_id.isdigit():
                if row_id not in rows:
                    rows[row_id] = {}
                rows[row_id][clean_field_name] = value

        print("Merged rows:")
        for row_id, data in rows.items():
            print(f"Row {row_id}: {data}")

        # Convert merged rows to a list of dictionaries.
        unit_parameters_raw = list(rows.values())
        # Save raw data for reporting purposes.
        session['unit_parameters_raw'] = unit_parameters_raw

        # Create a DataFrame from the raw unit parameters.
        df_units = pd.DataFrame(unit_parameters_raw)
        
        # Rename columns to match the Excel sheet expected by Stryke.
        # Adjust these keys if your form uses different names.
        rename_map = {
            "facility": "Facility",  # if applicable; often auto-filled
            "unit": "Unit",          # if applicable; might be generated automatically
            "type": "Runner Type",
            "velocity": "intake_vel",
            "order": "op_order",
            "H": "H",
            "RPM": "RPM",
            "D": "D",
            "efficiency": "ada",
            "N": "N",
            "Qopt": "Qopt",
            "Qcap": "Qcap",
            "B": "B",
            "iota": "iota",
            "D1": "D1",
            "D2": "D2",
            "lambda": "lambda",
            "roughness": "roughness"
        }
        df_units.rename(columns=rename_map, inplace=True)
        
        # Retrieve the user's unit system.
        units = session.get('units', 'metric')
        
        # If units are metric, convert the appropriate fields to imperial.
        # Conversion factors: meters -> feet: multiply by 3.28084; m³/s -> ft³/s: multiply by 35.31469989.
        if units == 'metric':
            conv_length = 3.28084  # For fields measured in meters
            conv_flow = 35.31469989  # For fields measured in m³/s

            # List of columns to convert from meters to feet.
            length_fields = ["intake_vel", "H", "D", "B", "D1", "D2"]
            # List of columns to convert from m³/s to ft³/s.
            flow_fields = ["Qopt", "Qcap"]

            for col in length_fields:
                if col in df_units.columns:
                    # Convert to numeric and multiply by conversion factor.
                    df_units[col] = pd.to_numeric(df_units[col], errors='coerce') * conv_length

            for col in flow_fields:
                if col in df_units.columns:
                    df_units[col] = pd.to_numeric(df_units[col], errors='coerce') * conv_flow

        # Assume unit_params is your DataFrame containing the unit parameters
        unit_params_path = os.path.join(g.user_sim_folder, 'unit_params.csv')
        df_units.to_csv(unit_params_path, index=False)
        
        # Store only the file path in the session
        session['unit_params_file'] = unit_params_path
        # Save the converted DataFrame into session in JSON-serializable format.
        #session['unit_parameters_dataframe'] = df_units.to_json(orient='records')
        #session['unit_parameters'] = df_units.to_dict(orient='records')
        
        flash("Unit parameters saved successfully!")
        return redirect(url_for('operating_scenarios'))
    return render_template('unit_parameters.html')

@app.route('/operating_scenarios', methods=['GET', 'POST'])
def operating_scenarios():
    if request.method == 'POST':
        # We'll assume that the form keys follow a convention:
        # For run-of-river: "scenario_ror_0_1", "facility_ror_0_1", "unit_ror_0_1", "hours_ror_0_1"
        # For pumped storage/peaking: "scenario_ps_0_1", "facility_ps_0_1", "unit_ps_0_1", "hours_ps_0_1",
        # plus additional keys like "prob_not_operating_0_1", "shape_0_1", "location_0_1", "scale_0_1".
        ror_rows = {}  # For run-of-river rows.
        ps_rows = {}   # For pumped storage/peaking rows.
        
        for key, value in request.form.items():
            parts = key.split('_')
            if len(parts) >= 4:
                field = parts[0]            # e.g. "scenario"
                table_type = parts[1]       # either "ror" or "ps"
                row_id = parts[2] + "_" + parts[3]  # Combines outer_index and unit number.
                if table_type == "ror":
                    if row_id not in ror_rows:
                        ror_rows[row_id] = {}
                    ror_rows[row_id][field] = value
                elif table_type == "ps":
                    if row_id not in ps_rows:
                        ps_rows[row_id] = {}
                    ps_rows[row_id][field] = value
            else:
                # For keys that may not follow the standard pattern, e.g. "prob_not_operating_0_1"
                parts = key.rsplit('_', 1)
                if len(parts) == 2:
                    field, row_id = parts
                    # Assume these belong to the pumped storage/peaking table.
                    if row_id not in ps_rows:
                        ps_rows[row_id] = {}
                    ps_rows[row_id][field] = value
        
        # Convert grouped dictionaries to lists.
        ror_list = list(ror_rows.values())
        ps_list = list(ps_rows.values())
        
        import pandas as pd
        df_ror = pd.DataFrame(ror_list)
        df_ps = pd.DataFrame(ps_list)
        
        # For run-of-river rows, add pumped storage fields if missing.
        for col in ['prob_not_operating', 'shape', 'location', 'scale']:
            if col not in df_ror.columns:
                df_ror[col] = None
        
        # For pumped storage rows, ensure the common fields exist.
        for col in ['scenario', 'facility', 'unit', 'hours']:
            if col not in df_ps.columns:
                df_ps[col] = None
        
        # Combine the two DataFrames.
        df_os = pd.concat([df_ror, df_ps], ignore_index=True)
        
        # Reorder columns to match the Excel Operating Scenarios sheet.
        # Expected columns: "Scenario", "Facility", "Unit", "Hours",
        # "Prob Not Operating", "Shape", "Location", "Scale".
        df_os = df_os[['scenario', 'facility', 'unit', 'hours', 'prob_not_operating', 'shape', 'location', 'scale']]
        
        # Rename columns to title-case, matching the Excel sheet headers.
        df_os.rename(columns={
            'scenario': 'Scenario',
            'facility': 'Facility',
            'unit': 'Unit',
            'hours': 'Hours',
            'prob_not_operating': 'Prob Not Operating',
            'shape': 'Shape',
            'location': 'Location',
            'scale': 'Scale'
        }, inplace=True)
        
        # Save the combined operating scenarios to the session.
        #session['operating_scenarios'] = df_os.to_dict(orient='records')
        # Assume unit_params is your DataFrame containing the unit parameters
        op_scen_path = os.path.join(g.user_sim_folder, 'op_scen.csv')
        df_os.to_csv(op_scen_path, index=False)
        
        # Store only the file path in the session
        session['op_scen_file'] = op_scen_path        
        flash("Operating scenarios saved successfully!")
        return redirect(url_for('graph_editor'))  # Replace with your next route as needed.
        
    return render_template('operating_scenarios.html')

@app.route('/get_operating_scenarios', methods=['GET'])
def get_operating_scenarios():
    import pandas as pd
    import os
    operating_scenarios = []
    if 'op_scen_file' in session:
        os_file = session['op_scen_file']
        if os.path.exists(os_file):
            df_ops = pd.read_csv(os_file)
            # Optionally, select only the columns you need for the dropdown or display
            # For example, if you only need 'Scenario' and 'Unit', you could do:
            # df_ops = df_ops[['Scenario', 'Unit']]
            operating_scenarios = df_ops.to_dict(orient='records')
    return jsonify(operating_scenarios)

@app.route('/graph_editor', methods=['GET'])
def graph_editor():
    return render_template('graph_editor.html')


@app.route('/save_graph', methods=['POST'])
def save_graph():
    from networkx.readwrite import json_graph

    graph_data = request.get_json()
    session['raw_graph_data'] = graph_data

    summary_nodes = []
    simulation_nodes = {}  # Use node_id as the key
    summary_edges = []
    simulation_edges = []

    nodes = graph_data.get("elements", {}).get("nodes", [])
    for node in nodes:
        data = node.get("data", {})
        node_id = data.get("id")
        label = data.get("label", node_id)
        surv_fun = data.get("surv_fun", "default")
        survival_rate = data.get("survival_rate", None)

        # Swap the ID with Location
        summary_nodes.append({
            "ID": label,         # Use label as the ID
            "Location": node_id,   # Use node_id as the Location
            "Surv_Fun": surv_fun,
            "Survival": survival_rate
        })

        simulation_nodes[node_id] = {
            "ID": label,
            "Location": node_id,
            "Surv_Fun": surv_fun,
            "Survival": survival_rate
        }

    edges = graph_data.get("elements", {}).get("edges", [])
    for edge in edges:
        data = edge.get("data", {})
        source = data.get("source")
        target = data.get("target")
        weight = float(data.get("weight", "1.0"))

        summary_edges.append({
            "_from": source,
            "_to": target,
            "weight": weight
        })

        simulation_edges.append((source, target, {"weight": weight}))

    # Build the NetworkX graph
    G = nx.DiGraph()
    for node_id, attrs in simulation_nodes.items():
        G.add_node(node_id, **attrs)
    for source, target, attrs in simulation_edges:
        G.add_edge(source, target, **attrs)

    # Save graph in node-link format
    sim_graph_data = json_graph.node_link_data(G)
    session['simulation_graph'] = sim_graph_data

    session['graph_summary'] = {"Nodes": summary_nodes, "Edges": summary_edges}
    session['nodes_data'] = summary_nodes
    session['edges_data'] = summary_edges

    print("Saved Nodes:", list(G.nodes))
    print("Saved Edges:", list(G.edges))

    return jsonify(success=True, summary=session['graph_summary'])


@app.route('/get_unit_parameters', methods=['GET'])
def get_unit_parameters():
    unit_parameters = []
    if 'unit_params_file' in session:
        unit_params_file = session['unit_params_file']
        if os.path.exists(unit_params_file):
            df_unit = pd.read_csv(unit_params_file)
            # Trim to only the columns needed for the dropdown
            if 'Facility' in df_unit.columns and 'Unit' in df_unit.columns:
                df_unit = df_unit[['Facility', 'Unit']]
            unit_parameters = df_unit.to_dict(orient='records')
    return jsonify(unit_parameters)


@app.route('/population', methods=['GET', 'POST'])
def population():
    species_defaults = [
        {
            "name": "Mid Atlantic Micropterus in Spring",
            "dist": "Log Normal",
            "shape": "0.99",
            "location": "0",
            "scale": "0.0013",
            "max_ent_rate": "0.0413",
            "occur_prob": "0.4118",
            "length shape": "0.5907",
            "length location": "2.1245",
            "length scale": "15.9345"
        },
        # ...
    ]

    if request.method == 'POST':
        # Basic info
        species_name = request.form.get('species_name')
        common_name = request.form.get('common_name')
        scenario = request.form.get('scenario')
        simulate_choice = request.form.get('simulateChoice')
        iterations = request.form.get('iterations')

        pop_data = {
            "Species": species_name,
            "Common Name": common_name,
            "Scenario": scenario,
            "Simulate Choice": simulate_choice,
            "Iterations": iterations,
            "Entrainment Choice": None,  # Will fill if needed
            "Modeled Species": None,     # Will fill if needed
            # We'll fill shape, location, scale, etc. below
        }

        # Helper for float
        def safe_float(val):
            try: return float(val)
            except: return None

        units = session.get('units', 'metric')
        
        # If user chooses “entrainment event”
        if simulate_choice == 'entrainment event':
            entrainment_choice = request.form.get('entrainmentChoice')
            pop_data["Entrainment Choice"] = entrainment_choice

            # 1) Common user inputs
            Ucrit_input = request.form.get('Ucrit')
            length_mean_input = request.form.get('length_mean')
            length_sd_input = request.form.get('length_sd')

            # Convert Ucrit
            Ucrit_ft = None
            if Ucrit_input:
                try:
                    Ucrit_val = float(Ucrit_input)
                    Ucrit_ft = Ucrit_val * 3.28084 if units == 'metric' else Ucrit_val
                except:
                    pass
            pop_data["U_crit"] = Ucrit_ft

            # 2) If “modeled”
            if entrainment_choice == 'modeled':
                modeled_species = request.form.get('modeledSpecies')
                pop_data["Modeled Species"] = modeled_species

                selected_species = next((s for s in species_defaults if s["name"] == modeled_species), None)
                if selected_species:
                    # Grab shape, location, scale from species defaults
                    pop_data["shape"] = selected_species["shape"]
                    pop_data["location"] = selected_species["location"]
                    pop_data["scale"] = selected_species["scale"]
                    pop_data["max_ent_rate"] = selected_species["max_ent_rate"]
                    pop_data["occur_prob"] = selected_species["occur_prob"]
                    pop_data["dist"] = selected_species["dist"]
                    # Also set length shape/location/scale from species
                    pop_data["length shape"] = safe_float(selected_species["length shape"])
                    pop_data["length location"] = safe_float(selected_species["length location"])
                    pop_data["length scale"] = safe_float(selected_species["length scale"])
                    
                    # If user typed in length mean/sd, overwrite the length location/scale
                    if length_mean_input:
                        try:
                            length_mean_in = float(length_mean_input)
                            if units == 'metric': length_mean_in /= 25.4
                            pop_data["length location"] = length_mean_in
                        except:
                            pass
                    if length_sd_input:
                        try:
                            length_sd_in = float(length_sd_input)
                            if units == 'metric': length_sd_in /= 25.4
                            pop_data["length scale"] = length_sd_in
                        except:
                            pass
                else:
                    # If no species found
                    pop_data["shape"] = None
                    pop_data["location"] = None
                    pop_data["scale"] = None

            # 3) If “empirical”
            elif entrainment_choice == 'empirical':
                empirical_shape = request.form.get('empiricalShape')
                empirical_location = request.form.get('empiricalLocation')
                empirical_scale = request.form.get('empiricalScale')
                empirical_dist = request.form.get('empiricalDist')
                max_entrainment_rate = request.form.get('max_entrainment_rate')
                occurrence_probability = request.form.get('occurrence_probability')

                length_shape_input = request.form.get('length_shape')
                length_location_input = request.form.get('length_location')
                length_scale_input = request.form.get('length_scale')

                pop_data["Modeled Species"] = None
                pop_data["shape"] = empirical_shape
                pop_data["location"] = empirical_location
                pop_data["scale"] = empirical_scale
                pop_data["dist"] = empirical_dist
                pop_data["max_ent_rate"] = max_entrainment_rate
                pop_data["occur_prob"] = occurrence_probability

                pop_data["length shape"] = safe_float(length_shape_input)
                pop_data["length location"] = safe_float(length_location_input)
                pop_data["length scale"] = safe_float(length_scale_input)

        # 4) If user chooses “starting population” (or something else)
        else:
            # e.g. pop_data["Fish"] = request.form.get('starting_population')
            # or do nothing if you want
            pass

        # Now do final conversions for length_mean, length_sd
        # Maybe you want them always computed?
        length_mean_val = safe_float(request.form.get('length_mean'))
        if length_mean_val is not None:
            length_mean_in = length_mean_val / 25.4 if units == 'metric' else length_mean_val
        else:
            length_mean_in = None
        pop_data["Length_mean"] = length_mean_in

        length_sd_val = safe_float(request.form.get('length_sd'))
        if length_sd_val is not None:
            length_sd_in = length_sd_val / 25.4 if units == 'metric' else length_sd_val
        else:
            length_sd_in = None
        pop_data["Length_sd"] = length_sd_in

        # Possibly unify shape, location, scale again if you want
        # For instance:
        # pop_data["shape"] = pop_data.get("shape") or pop_data.get("Empirical Shape") or None

        # Store in session
        session['population_data'] = pop_data

        # Build DataFrame with standardized keys
        import pandas as pd
        df_population = pd.DataFrame([pop_data])

        expected_columns = [
            "Species", "Common Name", "Scenario", "Iterations", "Fish",
            "shape", "location", "scale",
            "max_ent_rate", "occur_prob",
            "Length_mean", "Length_sd", "U_crit",
            "length shape", "length location", "length scale"
        ]
        for col in expected_columns:
            if col not in df_population.columns:
                df_population[col] = None
        df_population = df_population[expected_columns]

        # DataFrame for simulation
        session['population_dataframe_for_sim'] = df_population.to_json(orient='records')

        # DataFrame for summary
        summary_column_mapping = {
            "Species": "Species Name",
            "Common Name": "Common Name",
            "Scenario": "Scenario",
            "Iterations": "Iterations",
            "Fish": "Fish",
            "shape": "Empirical Shape",
            "location": "Empirical Location",
            "scale": "Empirical Scale",
            "max_ent_rate": "Max Entr Rate",
            "occur_prob": "Occur Prob",
            "Length_mean": "Length Mean (in)",
            "Length_sd": "Length SD (in)",
            "U_crit": "Ucrit (ft/s)",
            "length shape": "Length Shape",
            "length location": "Length Location",
            "length scale": "Length Scale"
        }
        df_population_summary = df_population.rename(columns=summary_column_mapping)
        session['population_dataframe_for_summary'] = df_population_summary.to_json(orient='records')

        flash("Population parameters saved successfully!")
        return redirect(url_for('model_setup_summary'))

    # GET request
    return render_template('population.html', species_defaults=species_defaults)

@app.route('/model_summary')
def model_setup_summary():
    import os
    import json
    import pandas as pd

    # --- Unit Parameters ---
    unit_parameters = []
    unit_columns = []
    if 'unit_params_file' in session:
        unit_params_file = session['unit_params_file']
        print("Found unit_params_file in session:", unit_params_file)
        if os.path.exists(unit_params_file):
            try:
                df_unit = pd.read_csv(unit_params_file)
                unit_parameters = df_unit.to_dict(orient='records')
                unit_columns = list(df_unit.columns)
                print("Loaded unit parameters:", unit_parameters)
            except Exception as e:
                print("Error reading unit_params_file:", e)
        else:
            print("Unit parameters file not found on disk:", unit_params_file)
    else:
        print("No unit_params_file key in session.")

    # --- Operating Scenarios ---
    operating_scenarios = []
    if 'op_scen_file' in session:
        ops_file = session['op_scen_file']
        print("Found operating_scenarios_file in session:", ops_file)
        if os.path.exists(ops_file):
            try:
                df_ops = pd.read_csv(ops_file)
                operating_scenarios = df_ops.to_dict(orient='records')
                print("Loaded operating scenarios:", operating_scenarios)
            except Exception as e:
                print("Error reading operating_scenarios_file:", e)
        else:
            print("Operating scenarios file not found on disk:", ops_file)
    else:
        print("No operating_scenarios_file key in session.")

    # --- Flow Scenarios (stored as JSON in session) ---
    flow_scenarios = session.get("flow_scenario", [])
    if isinstance(flow_scenarios, str):
        try:
            flow_scenarios = json.loads(flow_scenarios)
        except Exception as e:
            print("Error decoding flow_scenario JSON:", e)
            flow_scenarios = []
    print("Flow scenarios:", flow_scenarios)

    # --- Graph Data ---
    graph_summary = session.get('graph_summary', {"Nodes": [], "Edges": []})
    graph_nodes = graph_summary.get('Nodes', [])
    graph_edges = graph_summary.get('Edges', [])
    print("Graph summary (Nodes):", graph_nodes)
    print("Graph summary (Edges):", graph_edges)

    # --- Other Data ---
    facilities_data = session.get('facilities_data', [])
    population_data_raw = session.get('population_dataframe_for_summary', '[]')
    try:
        population_parameters = json.loads(population_data_raw)
    except Exception as e:
        print("Error decoding population data:", e)
        population_parameters = []
    if isinstance(population_parameters, dict):
        population_parameters = [population_parameters]

    simulation_graph = session.get('simulation_graph', {})

    summary_data = {
        "Facilities": facilities_data,
        "Unit Parameters": unit_parameters,
        "Operating Scenarios": operating_scenarios,
        "Population": population_parameters,
        "Flow Scenarios": flow_scenarios,
        "Graph Summary": graph_summary,
        "Simulation Graph": simulation_graph
    }

    return render_template(
        "model_summary.html",
        summary_data=summary_data,
        unit_columns=unit_columns,
        unit_parameters=unit_parameters,
        operating_scenarios=operating_scenarios,
        flow_scenarios=flow_scenarios,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        facilities_data=facilities_data,
        population_parameters=population_parameters
    )

from flask import current_app  # Import at module level if desired


@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    from Stryke import stryke
    from flask import current_app
    print("DEBUG: session['proj_dir'] =", session.get("proj_dir"))

    # Build input dictionary from session
    data_dict = {
        "facilities": session.get("facilities_data"),
        "unit_parameters_file": session.get("unit_params_file"),
        "operating_scenarios_file": session.get("op_scen_file"),
        "population": session.get("population_data"),
        "flow_scenarios": session.get("flow_scenario"),
        "hydrograph_file": session.get("hydrograph_file"),
        "graph_data": session.get("simulation_graph"),
        "graph_summary": session.get("graph_summary"),
        "units_system": session.get("units", "imperial"),
        "simulation_mode": session.get("simulation_mode", "multiple_powerhouses_simulated_entrainment_routing"),
        "proj_dir": session.get("proj_dir")  # add this line!
    }

    # Setup simulation
    user_sim_folder = g.user_sim_folder
    sim = stryke.simulation(proj_dir=user_sim_folder, output_name="WebAppModel", wks=None)
    sim.webapp_import(data_dict, output_name="WebAppModel")

    # Push app context into background thread
    app_obj = current_app._get_current_object()
    simulation_thread = threading.Thread(
        target=run_simulation_in_background_custom,
        args=(sim, user_sim_folder, app_obj)
    )
    simulation_thread.start()

    flash("Simulation started! Check logs for progress.")
    return redirect(url_for("simulation_logs"))


def run_simulation_in_background_custom(sim_instance, user_sim_folder, app_obj):
    import sys
    from contextlib import redirect_stdout

    with app_obj.app_context():
        old_stdout = sys.stdout
        sys.stdout = QueueStream(LOG_QUEUE)
        try:
            sim_instance.run()
            sim_instance.summary()
            LOG_QUEUE.put("[Simulation Complete]")

            # Generate the report
            report_html = generate_report(sim_instance)
            report_path = os.path.join(user_sim_folder, "simulation_report.html")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_html)

            # Save path in a known location for retrieval in /report
            with open(os.path.join(user_sim_folder, "report_path.txt"), "w") as f:
                f.write(report_path)

        except Exception as e:
            print("Error during simulation:", e)
        finally:
            sys.stdout = old_stdout


@app.route('/simulation_logs')
def simulation_logs():
    return render_template('simulation_logs.html')

def generate_discharge_histogram_text(df, column="DAvgFlow_prorate", bins=10):
    """
    Generate a text-based histogram for discharge recurrence probability.
    The histogram is scaled to a maximum bar length of 40 characters.
    """
    data = df[column].dropna().values
    if len(data) == 0:
        return "<p>No discharge data available for histogram.</p>"
    counts, bin_edges = np.histogram(data, bins=bins)
    total = data.size
    lines = []
    lines.append("Discharge Recurrence Probability Histogram:")
    for i in range(len(counts)):
        low = bin_edges[i]
        high = bin_edges[i+1]
        count = counts[i]
        percentage = count / total * 100
        # Scale the bar to a maximum length of 40 characters
        bar_length = int((percentage / 100) * 40)
        bar = "*" * bar_length
        line = f"{low:8.2f} - {high:8.2f}: {bar} ({count} occurrences, {percentage:.1f}%)"
        lines.append(line)
    return "<pre>" + "\n".join(lines) + "</pre>"

def get_simulation_instance():
    # A simple simulation class to hold the needed attributes.
    class Simulation:
        pass
    sim = Simulation()
    # Retrieve the project directory from the session.
    sim.proj_dir = session.get('proj_dir')
    # Retrieve the output_name from the session, or use a default.
    sim.output_name = session.get('output_name', 'simulation_output')
    return sim

@app.route('/report')
def report():
    import os
    user_sim_folder = g.user_sim_folder
    path_file = os.path.join(user_sim_folder, "report_path.txt")

    if not os.path.exists(path_file):
        return "<p>Error: Report file path not found.</p>"

    with open(path_file, 'r') as f:
        report_path = f.read().strip()

    if not os.path.exists(report_path):
        return "<p>Error: Report file not found.</p>"

    with open(report_path, 'r', encoding='utf-8') as f:
        report_html = f.read()

    full_report_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Simulation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f7f7f7;
                margin: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1000px;
                margin: auto;
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #0056b3;
            }}
            h2, h3 {{
                color: #0056b3;
                border-bottom: 1px solid #ddd;
                padding-bottom: 4px;
            }}
            p {{
                line-height: 1.6;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
            }}
            th, td {{
                padding: 8px;
                border: 1px solid #ccc;
                text-align: left;
            }}
            .table-wrap {{
                overflow-x: auto;
            }}
            pre {{
                background: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
            }}
            .download-link {{
                display: inline-block;
                margin-top: 20px;
                background: #007BFF;
                color: white;
                padding: 10px 15px;
                text-decoration: none;
                border-radius: 4px;
            }}
            .download-link:hover {{
                background: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {report_html}
            <a href="/download_report" class="download-link">⬇ Download Report</a>
        </div>
    </body>
    </html>
    """
    return full_report_html

def generate_report(sim):
    """
    Generate the comprehensive HTML report for the simulation.
    This version ensures that all plot fonts render at least size 8,
    rounds numeric values in the Beta Distributions table to two decimals,
    and plots the daily histograms (entrainment and mortality) side by side.
    """
    import os
    import pandas as pd
    import io, base64
    from datetime import datetime
    import matplotlib.pyplot as plt

    # Set global minimum font size for all plots
    plt.rcParams.update({'font.size': 8})

    hdf_path = os.path.join(sim.proj_dir, f"{sim.output_name}.h5")
    if not os.path.exists(hdf_path):
        return "<p>Error: HDF file not found. Please run the simulation first.</p>"

    store = pd.HDFStore(hdf_path, mode='r')

    report_sections = [
        f"<h1>Simulation Report</h1>",
        f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        f"<p>HDF keys found: {store.keys()}</p>"
    ]

    # Helper: Wrap DataFrame HTML in a scrollable container.
    def enforce_horizontal(df, name=""):
        if df is None or df.empty:
            return f"<p>No {name} data available.</p>"
        shape_info = f"<p>{name} data shape: {df.shape}</p>"
        if df.shape[0] > 1 and df.shape[0] > df.shape[1]:
            df = df.T
            shape_info += f"<p>Transposed to shape: {df.shape}</p>"
        # For Beta Distributions, round numeric columns to 2 decimals.
        if name.lower() == "beta distributions":
            df = df.copy()
            for col in df.select_dtypes(include=["number"]).columns:
                df[col] = df[col].round(2)
        table_html = df.to_html(index=False, border=1, classes="table")
        return shape_info + f"<div style='overflow-x:auto;'>{table_html}</div>"

    def add_section(title, key):
        report_sections.append(f"<h2>{title}</h2>")
        if key in store.keys():
            df = store[key]
            report_sections.append(enforce_horizontal(df, title))
        else:
            report_sections.append(f"<p>No {title} data available.</p>")

    # Basic sections
    add_section("Nodes", "/Nodes")
    add_section("Edges", "/Edges")
    add_section("Unit Parameters", "/Unit_Parameters")
    add_section("Facilities", "/Facilities")
    add_section("Flow Scenarios", "/Flow Scenarios")
    add_section("Operating Scenarios", "/Operating Scenarios")
    add_section("Population", "/Population")

    # --- HYDROGRAPH SECTION: Time Series + Recurrence Histogram ---
    report_sections.append("<h2>Hydrograph Plots</h2>")
    if "/Hydrograph" in store.keys():
        hydrograph_df = store["/Hydrograph"]
        if 'datetimeUTC' in hydrograph_df.columns:
            hydrograph_df['datetimeUTC'] = pd.to_datetime(hydrograph_df['datetimeUTC'])

        def create_hydro_timeseries(df):
            plt.rcParams.update({'font.size': 8})
            fig = plt.figure(figsize=(6,4))
            plt.plot(df['datetimeUTC'], df['DAvgFlow_prorate'], marker='.', linestyle='-')
            plt.xlabel("Date")
            plt.ylabel("Discharge")
            plt.title("Hydrograph Time Series")
            plt.xticks(rotation=45)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        def create_hydro_hist(df):
            plt.rcParams.update({'font.size': 8})
            fig = plt.figure(figsize=(6,4))
            plt.hist(df["DAvgFlow_prorate"].dropna(), bins=10, edgecolor='black')
            plt.xlabel("Discharge")
            plt.ylabel("Frequency")
            plt.title("Recurrence Histogram")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        ts_b64 = create_hydro_timeseries(hydrograph_df)
        hist_b64 = create_hydro_hist(hydrograph_df)
        report_sections.append(f"""
        <div style="display:flex; gap:20px; justify-content:center; flex-wrap:wrap;">
            <div style="flex:1; min-width:300px; text-align:center;">
                <h3>Time Series</h3>
                <img src="data:image/png;base64,{ts_b64}" style="max-width:100%; height:auto;" />
            </div>
            <div style="flex:1; min-width:300px; text-align:center;">
                <h3>Recurrence Histogram</h3>
                <img src="data:image/png;base64,{hist_b64}" style="max-width:100%; height:auto;" />
            </div>
        </div>
        """)
    else:
        report_sections.append("<p>No hydrograph data available.</p>")

    # --- BETA DISTRIBUTIONS ---
    add_section("Beta Distributions", "/Beta_Distributions")

    # --- YEARLY SUMMARY PANEL (Iteration-based) ---
    yearly_df = store["/Yearly_Summary"] if "/Yearly_Summary" in store.keys() else None
    daily_df = store["/Daily"] if "/Daily" in store.keys() else None

    if daily_df is not None and not daily_df.empty:
        if 'num_survived' in daily_df.columns and 'pop_size' in daily_df.columns:
            daily_df['num_mortality'] = daily_df['pop_size'] - daily_df['num_survived']
        if 'iteration' in daily_df.columns:
            iteration_sums = daily_df.groupby('iteration').agg({
                'num_entrained': 'sum',
                'num_mortality': 'sum'
            }).reset_index()
        else:
            iteration_sums = None
    else:
        iteration_sums = None

    def create_iteration_hist(df, metric, title):
        plt.rcParams.update({'font.size': 8})
        fig = plt.figure()
        plt.hist(df[metric].dropna(), bins=10, edgecolor='black')
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel("Frequency")
        plt.title(title)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def render_yearly_panel(yearly_df, iteration_sums):
        if yearly_df is None or yearly_df.empty:
            return "<p>No yearly summary data available.</p>"
        row = yearly_df.iloc[0]  # Only one row expected
        panel_html = "<h2>Yearly Summary (Iteration-based)</h2>"
        for metric in ["entrainment", "mortality"]:
            if iteration_sums is not None and f'num_{metric}' in iteration_sums.columns:
                hist_b64 = create_iteration_hist(iteration_sums, f'num_{metric}', f"Total {metric.title()} Distribution by Iteration")
            else:
                hist_b64 = ""
            # Use expected column keys; adjust if needed.
            if metric == 'entrainment':
                abbv = 'ent'
            else:
                abbv = 'mort'
            mean_val = row.get(f"mean_yearly_{abbv}", "N/A")
            lcl_val = row.get(f"lcl_yearly_{abbv}", "N/A")
            ucl_val = row.get(f"ucl_yearly_{abbv}", "N/A")
            like10 = row.get(f"1_in_10_day_{metric}", "N/A")
            like100 = row.get(f"1_in_100_day_{metric}", "N/A")
            like1000 = row.get(f"1_in_1000_day_{metric}", "N/A")
            panel_html += f"""
            <div style="display:flex; flex-wrap:wrap; margin-bottom:20px; border:1px solid #ccc; padding:10px; border-radius:5px;">
                <div style="flex:1; min-width:300px; padding:10px; border-right:1px solid #ddd;">
                    <h3>Histogram ({metric.title()})</h3>
                    <div style="text-align:center;">
                        {'<img src="data:image/png;base64,' + hist_b64 + '" style="max-width:100%; height:auto;" />' if hist_b64 else "<p>No histogram data</p>"}
                    </div>
                </div>
                <div style="flex:1; min-width:300px; padding:10px;">
                    <h3>Statistics ({metric.title()})</h3>
                    <p><strong>Average Annual:</strong> {mean_val}</p>
                    <p><strong>95% CI:</strong> {lcl_val} - {ucl_val}</p>
                    <p><strong>1 in 10 day event:</strong> {like10}</p>
                    <p><strong>1 in 100 day event:</strong> {like100}</p>
                    <p><strong>1 in 1000 day event:</strong> {like1000}</p>
                </div>
            </div>
            """
        return panel_html

    if yearly_df is not None and not yearly_df.empty:
        panel_html = render_yearly_panel(yearly_df, iteration_sums)
        report_sections.append(panel_html)
    else:
        report_sections.append("<p>No yearly summary data available.</p>")

    # --- DAILY HISTOGRAMS SIDE BY SIDE ---
    report_sections.append("<h2>Daily Histograms</h2>")
    if daily_df is not None and not daily_df.empty:
        # Ensure daily mortality exists
        if 'num_mortality' not in daily_df.columns and 'num_survived' in daily_df.columns and 'pop_size' in daily_df.columns:
            daily_df['num_mortality'] = daily_df['pop_size'] - daily_df['num_survived']

        def create_daily_hist(data, col, title):
            plt.rcParams.update({'font.size': 8})
            fig = plt.figure()
            plt.hist(data[col].dropna(), bins=20, edgecolor='black')
            plt.xlabel(col.replace('_', ' ').title())
            plt.ylabel("Frequency")
            plt.title(title)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        entr_img = None
        mort_img = None
        if 'num_entrained' in daily_df.columns:
            entr_img = create_daily_hist(daily_df, 'num_entrained', 'Daily Entrainment Distribution')
        if 'num_mortality' in daily_df.columns:
            mort_img = create_daily_hist(daily_df, 'num_mortality', 'Daily Mortality Distribution')

        report_sections.append("""
        <div style="display:flex; gap:20px; justify-content:center; flex-wrap:wrap;">
        """)
        if entr_img:
            report_sections.append(f"""
            <div style="flex:1; min-width:300px; text-align:center;">
                <h3>Daily Entrainment</h3>
                <img src="data:image/png;base64,{entr_img}" style="max-width:100%; height:auto;" />
            </div>
            """)
        else:
            report_sections.append("""
            <div style="flex:1; min-width:300px; text-align:center;">
                <h3>Daily Entrainment</h3>
                <p>No 'num_entrained' data available.</p>
            </div>
            """)
        if mort_img:
            report_sections.append(f"""
            <div style="flex:1; min-width:300px; text-align:center;">
                <h3>Daily Mortality</h3>
                <img src="data:image/png;base64,{mort_img}" style="max-width:100%; height:auto;" />
            </div>
            """)
        report_sections.append("</div>")
    else:
        report_sections.append("<p>No daily data available.</p>")

    store.close()

    final_html = "\n".join(report_sections)
    full_report = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Simulation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f7f7f7;
                margin: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1800px;
                margin: auto;
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #0056b3;
            }}
            h2 {{
                color: #0056b3;
                border-bottom: 1px solid #ddd;
                padding-bottom: 4px;
                margin-top: 2rem;
            }}
            h3 {{
                color: #0056b3;
                margin-top: 1.5rem;
            }}
            p {{
                line-height: 1.6;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
            }}
            th, td {{
                padding: 8px;
                border: 1px solid #ccc;
                text-align: left;
            }}
            .table-wrap {{
                overflow-x: auto;
            }}
            pre {{
                background: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
            }}
            .download-link {{
                display: inline-block;
                margin-top: 20px;
                background: #007BFF;
                color: white;
                padding: 10px 15px;
                text-decoration: none;
                border-radius: 4px;
            }}
            .download-link:hover {{
                background: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {final_html}
            <a href="/download_report" class="download-link">Download Report</a>
        </div>
    </body>
    </html>
    """
    return full_report




@app.route('/download_report')
def download_report():
    report_path = session.get('report_path')
    if not report_path or not os.path.exists(report_path):
        return "<p>Error: Report file not found.</p>"

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return Response(
        content,
        mimetype='text/html',
        headers={"Content-Disposition": "attachment;filename=simulation_report.html"}
    )


# @app.route('/report')
# def report():
#     report_html = session.get('latest_report_html')

#     if not report_html:
#         return "<p>Error: Report not found in session. Please run the simulation first.</p>"

#     full_report_html = f"""<!DOCTYPE html>
#     <html lang="en">
#     <head>
#       <meta charset="UTF-8">
#       <title>Simulation Report</title>
#     </head>
#     <body>
#     {report_html}
#     <br><a href="/download_report">Download Report</a>
#     </body>
#     </html>"""
#     return full_report_html


# from flask import Response

# @app.route('/download_report')
# def download_report():
#     report_html = session.get('latest_report_html')
#     if not report_html:
#         return "<p>Error: Report not found in session.</p>"

#     full_report_html = f"""<!DOCTYPE html>
#     <html lang="en">
#     <head>
#       <meta charset="UTF-8">
#       <title>Simulation Report</title>
#     </head>
#     <body>
#     {report_html}
#     </body>
#     </html>"""

#     return Response(
#         full_report_html,
#         mimetype='text/html',
#         headers={"Content-Disposition": "attachment;filename=simulation_report.html"}
#     )


# @app.route('/download_results')
# def download_results():
#     try:
#         # Re-create the simulation object from stored session/previous import.
#         # This example assumes that your simulation object can be re-initialized using your webapp_import method.
#         sim_data = {
#             "graph_summary": session.get("graph_summary"),
#             "unit_parameters": session.get("unit_parameters"),
#             "facilities": session.get("facilities_data"),
#             "flow_scenarios": session.get("flow_scenario"),
#             "operating_scenarios": session.get("operating_scenarios"),
#             "population": session.get("population_data"),
#             "hydrograph": session.get("hydrograph_df")
#         }
#         sim = stryke.simulation(proj_dir=os.getcwd(), output_name="WebAppModel")
#         sim.webapp_import(sim_data, output_name="WebAppModel")
        
#         # Generate the report
#         report_html = generate_report(sim)
        
#         # Save the report to a file in the user's simulation folder
#         sim_folder = g.get("user_sim_folder", SIM_PROJECT_FOLDER)
#         report_path = os.path.join(sim_folder, "simulation_report.html")
#         with open(report_path, "w", encoding="utf-8") as f:
#             f.write(report_html)
        
#         return send_file(report_path, as_attachment=True)
#     except Exception as e:
#         flash(f"Error generating report: {e}")
#         return redirect(url_for("simulation_logs"))
# from weasyprint import HTML

# @app.route('/report_pdf')
# def report_pdf():
#     # Generate the full HTML report (using your existing report() function)
#     report_html = report()
#     try:
#         pdf = HTML(string=report_html).write_pdf()
#     except Exception as e:
#         return f"<p>Error converting report to PDF: {e}</p>", 500

#     response = Response(pdf, mimetype='application/pdf')
#     response.headers['Content-Disposition'] = 'attachment; filename=simulation_report.pdf'
#     return response

# Un Comment to Test Locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
