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
from flask import jsonify, stream_with_context
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
import traceback
import logging
import logging.handlers
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import tables
from werkzeug.exceptions import HTTPException, NotFound


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

from collections import defaultdict
SESSION_LOCKS = defaultdict(threading.Lock)

def get_session_lock(user_key: str) -> threading.Lock:
    return SESSION_LOCKS[user_key]


# Monkey-patch whichever Stryke import you use
# try:
#     import stryke as _stryke_mod           # common import style in your app
# except ImportError:
#     from Stryke import stryke as _stryke_mod  # if yours is packaged as Stryke/stryke.py

# # Patch the function on the module so internal calls use this tolerant version
# _stryke_mod.read_csv_if_exists = _read_csv_if_exists_compat

# # (Optional) quick sanity print so you see it once in logs
# print("[INIT] Patched Stryke.read_csv_if_exists to back-compat shim.", flush=True)

for name in ("Stryke.stryke", "stryke"):
    mod = sys.modules.get(name)
    if mod:
        setattr(mod, "read_csv_if_exists", _read_csv_if_exists_compat)

print("[INIT] Patched Stryke.read_csv_if_exists to back-compat shim.", flush=True)


# Create a custom stream that writes log messages to LOG_QUEUE
class QueueStream:
    def __init__(self, q):
        self.q = q
    def write(self, message):
        if message.strip():
            try:
                self.q.put(message)
            except Exception:
                pass
    def flush(self):
        pass
    
class QueueLogHandler(logging.Handler):
    def __init__(self, q): super().__init__(); self.q = q
    def emit(self, record):
        try:
            msg = self.format(record)
            # keep it simple: one line per log event
            self.q.put(msg.replace("\n", " | "))
        except Exception:
            pass

# Set up logging to use QueueStream
queue_stream = QueueStream(LOG_QUEUE)
stream_handler = logging.StreamHandler(queue_stream)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = []  # Clear any default handlers
logger.addHandler(stream_handler)


# ----------------- Password Protection -----------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['PASSWORD'] = 'expensive5rudabega!@1'  # Set your desired password here


# Set session lifetime to 1 day (adjust as needed)
app.permanent_session_lifetime = timedelta(days=1)

REQUIRED_FILE_KEYS = {
    "hydrograph_file": "Hydrograph CSV",
    "unit_parameters_file": "Unit parameters CSV",
    "operating_scenarios_file": "Operating scenarios CSV",
    # add any other hard requirements here
}

def _coerce_empty_to_none(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v

def _validate_and_normalize_inputs(data):
    """
    Returns (normalized_data, missing_keys, bad_paths)
    - missing_keys: required keys that are None/blank
    - bad_paths: required keys whose paths do not exist on disk
    """
    norm = dict(data)  # shallow copy
    missing, bad = [], []
    import os

    # Normalize strings: "" -> None
    for k, v in list(norm.items()):
        norm[k] = _coerce_empty_to_none(v)

    # Required files must be present and exist
    for key, label in REQUIRED_FILE_KEYS.items():
        fp = norm.get(key)
        if not fp:
            missing.append(f"{key} ({label})")
        elif not isinstance(fp, (str, bytes, os.PathLike)):
            bad.append(f"{key} → {repr(fp)}  (not a path)")
        elif not os.path.isfile(fp):
            bad.append(f"{key} → {fp}  (file not found)")

    return norm, missing, bad

@app.errorhandler(Exception)
def handle_exception(e):
    # Let real HTTP errors (404, 405, etc.) pass through untouched
    if isinstance(e, HTTPException):
        return e
    # Log server-side exceptions with traceback, return 500
    import traceback
    traceback.print_exc()
    return "Internal Server Error", 500

@app.errorhandler(NotFound)
def handle_404(e):
    # Simple 404, no scary traceback
    return "Not Found", 404


@app.before_request
def make_session_permanent():
    session.permanent = True
    
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the full traceback
    traceback_str = traceback.format_exc()
    print("Unhandled Exception:", traceback_str, flush=True)
    
    # Optionally return a custom 500 page
    return "Internal Server Error", 500

@app.before_request
def attach_session_paths_to_g():
    # Mirror session paths into g for routes/templates that expect them
    g.user_upload_dir = session.get('user_upload_dir')
    g.user_sim_folder = session.get('user_sim_folder')
    g.proj_dir = session.get('proj_dir')

# Helper function to clear folders
def clear_folder_recursive(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

@app.before_request
def require_login_and_setup():
    if not session.get('logged_in') and request.endpoint not in ['login', 'static', 'health']:
        return redirect(url_for('login'))
    
    if session.get('logged_in'):
        if 'user_dir' not in session:
            session['user_dir'] = uuid.uuid4().hex

        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session['user_dir'])
        user_sim_folder = os.path.join(SIM_PROJECT_FOLDER, session['user_dir'])

        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_sim_folder, exist_ok=True)

        # Explicitly store paths in session
        session['user_upload_dir'] = user_upload_dir
        session['user_sim_folder'] = user_sim_folder

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == app.config['PASSWORD']:
            session['logged_in'] = True
            session['user_dir'] = uuid.uuid4().hex  # Unique directory per session
            flash("Logged in successfully!")
            return redirect(url_for('index'))
        else:
            error = 'Invalid password. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    user_upload_dir = session.get('user_upload_dir')
    user_sim_folder = session.get('user_sim_folder')

    if user_upload_dir and os.path.exists(user_upload_dir):
        clear_folder(user_upload_dir)
        os.rmdir(user_upload_dir)

    if user_sim_folder and os.path.exists(user_sim_folder):
        clear_folder(user_sim_folder)
        os.rmdir(user_sim_folder)

    session.clear()
    flash("Logged out successfully.")
    print("User session and directories cleared successfully.")
    return redirect(url_for('login'))

# -------------------------------------------------------

# Define directories for uploads and simulation projects
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
SIM_PROJECT_FOLDER = os.path.join(os.getcwd(), 'simulation_project')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIM_PROJECT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from flask_session import Session  # <-- add this (the class)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(SIM_PROJECT_FOLDER, '_flask_session')
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
app.config['SESSION_PERMANENT'] = True
Session(app)

def cleanup_old_data():
    """Remove files and directories older than 24 hours in upload and simulation folders."""
    now = time.time()
    cutoff = now - 24 * 3600  # 24 hours ago

    for folder in (UPLOAD_FOLDER, SIM_PROJECT_FOLDER):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                # If it's a directory, check its last modified time
                mtime = os.path.getmtime(path)
                if mtime < cutoff:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            except Exception as e:
                app.logger.warning(f"Failed to clean up {path}: {e}")

    # Schedule next cleanup in one hour
    threading.Timer(3600, cleanup_old_data).start()

# Kick off the first cleanup one minute after startup
threading.Timer(60, cleanup_old_data).start()

@app.route("/health")
def health():
    print("Health endpoint accessed")
    return "OK", 200

def run_xls_simulation(ws, wks, output_name):
    old_stdout = sys.stdout
    sys.stdout = QueueStream(LOG_QUEUE)
    try:
        simulation_instance = stryke.simulation(ws, wks, output_name=output_name)
        simulation_instance.run()
        simulation_instance.summary()
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        sys.stdout = old_stdout
        LOG_QUEUE.put(None)

def run_xls_simulation_in_background(ws, wks, output_name, data_dict=None):
    """Background thread for running the simulation and streaming logs to LOG_QUEUE."""
    old_stdout = sys.stdout
    sys.stdout = QueueStream(LOG_QUEUE)
    try:
        logger.info("Starting simulation...")
        sim = stryke.simulation(ws, wks, output_name=output_name)
        # If your Excel path doesn’t need webapp_import data, this call is okay as-is:
        sim.webapp_import()
        sim.run()
        sim.summary()
        logger.info("Simulation completed successfully.")
    except Exception as e:
        logger.exception("Simulation failed.")
        LOG_QUEUE.put(f"[ERROR] Simulation failed: {str(e)}")
    finally:
        sys.stdout = old_stdout
        LOG_QUEUE.put("[Simulation Complete]")  # <- unified sentinel


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
    
        user_upload_dir = session.get('user_upload_dir')
        user_sim_folder = session.get('user_sim_folder')
        if not user_upload_dir or not user_sim_folder:
            flash('Session expired. Please log in again.')
            return redirect(url_for('login'))
    
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_sim_folder, exist_ok=True)
    
        up_file_path = os.path.join(user_upload_dir, file.filename)
        file.save(up_file_path)
        flash(f'File successfully uploaded: {file.filename}')
    
        simulation_file_path = os.path.join(user_sim_folder, file.filename)
        shutil.copy(up_file_path, simulation_file_path)
    
        ws = user_sim_folder
        wks = file.filename
        output_name = 'Simulation_Output'
    
        try:
            simulation_thread = threading.Thread(
                target=run_xls_simulation_in_background,
                args=(ws, wks, output_name),  # 4th arg is optional now
                daemon=True
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


# @app.route('/download_zip')
# def download_zip():
#     user_sim_folder = g.get("user_sim_folder", SIM_PROJECT_FOLDER)

#     # Remove any old zip files.
#     for fname in os.listdir(user_sim_folder):
#         if fname.endswith(".zip"):
#             old_path = os.path.join(user_sim_folder, fname)
#             try:
#                 os.remove(old_path)
#                 print(f"Removed old ZIP: {old_path}")
#             except Exception as e:
#                 print(f"Error removing old ZIP {old_path}: {e}")

#     # Create a new ZIP file.
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     zip_filename = f"simulation_results_{timestamp}.zip"
#     zip_filepath = os.path.join(user_sim_folder, zip_filename)
#     print(f"Creating ZIP file: {zip_filepath}")

#     try:
#         with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
#             for file_name in os.listdir(user_sim_folder):
#                 # Skip files with these extensions (adjust as needed).
#                 if file_name.endswith((".hdf", ".h5", ".zip")):
#                     continue
#                 file_path = os.path.join(user_sim_folder, file_name)
#                 if os.path.isfile(file_path):
#                     try:
#                         zipf.write(file_path, arcname=file_name)
#                         print(f"Added to ZIP: {file_name}")
#                     except Exception as e:
#                         print(f"Skipping file {file_name} => {e}")
#         print(f"ZIP file successfully created: {zip_filepath}")
#     except Exception as e:
#         print(f"Error creating ZIP file: {e}")
#         flash("Failed to create ZIP file.")
#         return redirect(url_for('some_error_page'))

#     if os.path.exists(zip_filepath):
#         return send_file(zip_filepath, as_attachment=True)
#     else:
#         flash("ZIP file not found.")
#         return redirect(url_for('some_error_page'))

from flask import Response, stream_with_context

@app.route('/stream')
def stream():
    def event_stream():
        import queue as _q
        while True:
            try:
                message = LOG_QUEUE.get(timeout=15)
            except _q.Empty:
                yield "data: [keepalive]\n\n"
                continue
            yield f"data: {message}\n\n"
            if message == "[Simulation Complete]":
                break
    return Response(event_stream(), mimetype="text/event-stream")


# @app.route('/stream')
# def stream():
#     @stream_with_context
#     def event_stream():
#         yield ": connected\n\n"  # open for proxies
#         while True:
#             try:
#                 message = LOG_QUEUE.get(timeout=15)
#                 if message is None:
#                     continue
#                 yield f"data: {message}\n\n"
#                 if message == "[Simulation Complete]":
#                     break
#             except queue.Empty:
#                 yield "event: ping\ndata: keepalive\n\n"

#     resp = Response(event_stream(), mimetype="text/event-stream")
#     resp.headers["Cache-Control"] = "no-cache"
#     resp.headers["X-Accel-Buffering"] = "no"
#     resp.headers["Connection"] = "keep-alive"
#     return resp

# @app.route('/stream')
# def stream():
#     @stream_with_context
#     def event_stream():
#         yield ": connected\n\n"  # open the pipe for proxies
#         while True:
#             try:
#                 message = LOG_QUEUE.get(timeout=15)  # don’t idle forever
#                 if message is None:
#                     continue  # ignore legacy None
#                 yield f"data: {message}\n\n"
#                 if message == "[Simulation Complete]":
#                     break
#             except queue.Empty:
#                 yield "event: ping\ndata: keepalive\n\n"  # heartbeat

#     resp = Response(event_stream(), mimetype="text/event-stream")
#     resp.headers["Cache-Control"] = "no-cache"
#     resp.headers["X-Accel-Buffering"] = "no"
#     resp.headers["Connection"] = "keep-alive"
#     return resp

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



@app.route('/create_project', methods=['GET', 'POST'])
def create_project():
    if request.method == 'POST':
        project_name = request.form.get('project_name')
        project_notes = request.form.get('project_notes')
        units = request.form.get('units')
        model_setup = request.form.get('model_setup')

        # Save project metadata in session
        session['project_name'] = project_name
        session['project_notes'] = project_notes
        session['units'] = units
        session['model_setup'] = model_setup

        # Use explicitly stored session directory
        session['proj_dir'] = session.get('user_sim_folder')
        flash(f"Project '{project_name}' created successfully!")
        print("Project directory set to:", session['proj_dir'])

        return redirect(url_for('flow_scenarios'))

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
    #print ('hydrograph before to numeric:')
    #print (df)    
    # Convert discharge values to numeric.
    df['DAvgFlow_prorate'] = pd.to_numeric(df['DAvgFlow_prorate'], errors='coerce')
    
    # Drop rows with invalid dates or missing discharge values.
    df.dropna(inplace=True)
    
    # Set the datetime column as the index if that is required downstream.
    #df.set_index('datetimeUTC', inplace=True)
    #print ('hydrograph:')
    #print (df)
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
        
        # print("DEBUG: Received scenario_type:", scenario_type, flush=True)
        # print("DEBUG: Received hydrograph_data:", hydrograph_data, flush=True)
        
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
            
            hydro_file_path = os.path.join(session['proj_dir'], 'hydrograph.csv')
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
        #print("DEBUG: Flow scenario DataFrame:")
        #print(flow_scenario_df, flush=True)
        
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
        #print("POST form data:", request.form)

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

        # Debug print the flow scenario DataFrame.
        #print("DEBUG: Facilities DataFrame:")
        #print(facilities_data, flush=True)

    units = session.get('units', 'metric')
    scenario = session.get('scenario_name', 'Unknown Scenario')
    sim_mode = session.get('simulation_mode', 'multiple_powerhouses_simulated_entrainment_routing')
    return render_template('facilities.html', units=units, scenario=scenario, sim_mode=sim_mode)

@app.route('/unit_parameters', methods=['GET', 'POST'])
def unit_parameters():
    if request.method == 'POST':
        # print("Received form data:")
        # for key, value in request.form.items():
        #     print(f"{key} : {value}")

        # Merge form data into rows (each row represents one unit's parameters)
        rows = {}
        for key, value in request.form.items():
            parts = key.rsplit('_', 1)
            if len(parts) != 2:
                continue
            field_name, row_id = parts
            # Remove any trailing underscore and digits from field_name
            clean_field_name = re.sub(r'_\d+$', '', field_name)
            #print(f"Key: {key} split into clean_field_name: {clean_field_name} and row_id: {row_id}")
            if row_id.isdigit():
                if row_id not in rows:
                    rows[row_id] = {}
                rows[row_id][clean_field_name] = value

        # print("Merged rows:")
        # for row_id, data in rows.items():
        #     print(f"Row {row_id}: {data}")

        # Convert merged rows to a list of dictionaries.
        unit_parameters_raw = list(rows.values())
        session['unit_parameters_raw'] = unit_parameters_raw

        # Create a DataFrame from the raw unit parameters.
        df_units = pd.DataFrame(unit_parameters_raw)
        
        # Update rename_map to include the new barotrauma fields.
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
            "roughness": "roughness",
            # New fields for barotrauma:
            "fb_depth": "fb_depth",                  # forebay depth
            "ps_D": "ps_D",                          # penstock diameter
            "ps_length": "ps_length",                # penstock length
            "submergence_depth": "submergence_depth",# submergence depth of draft tube
            "elevation_head": "elevation_head"        # elevation head at downstream point
        }
        df_units.rename(columns=rename_map, inplace=True)
        
        # Retrieve the user's unit system.
        units = session.get('units', 'metric')
        
        # Convert appropriate fields if using metric units.
        if units == 'metric':
            conv_length = 3.28084  # meters to feet
            conv_flow = 35.31469989  # m³/s to ft³/s

            # Update length_fields to include the new fields (assuming they are measured in meters).
            length_fields = ["intake_vel", "H", "D", "B", "D1", "D2",
                             "fb_depth", "ps_D", "ps_length", "submergence_depth", "elevation_head"]
            flow_fields = ["Qopt", "Qcap"]

            for col in length_fields:
                if col in df_units.columns:
                    df_units[col] = pd.to_numeric(df_units[col], errors='coerce') * conv_length

            for col in flow_fields:
                if col in df_units.columns:
                    df_units[col] = pd.to_numeric(df_units[col], errors='coerce') * conv_flow

        # Save the DataFrame as CSV.
        unit_params_path = os.path.join(session['proj_dir'], 'unit_params.csv')
        df_units.to_csv(unit_params_path, index=False)
        
        session['unit_params_file'] = unit_params_path

        # print("DEBUG: Unit Parameters DataFrame:")
        # print(df_units, flush=True)
        
        flash("Unit parameters saved successfully!")
        return redirect(url_for('operating_scenarios'))
    return render_template('unit_parameters.html')

# @app.route('/unit_parameters', methods=['GET', 'POST'])
# def unit_parameters():
#     if request.method == 'POST':
#         print("Received form data:")
#         for key, value in request.form.items():
#             print(f"{key} : {value}")

#         # Merge form data into rows (each row represents one unit's parameters)
#         rows = {}
#         for key, value in request.form.items():
#             parts = key.rsplit('_', 1)
#             if len(parts) != 2:
#                 continue
#             field_name, row_id = parts
#             # Remove any trailing underscore and digits from field_name
#             clean_field_name = re.sub(r'_\d+$', '', field_name)
#             print(f"Key: {key} split into clean_field_name: {clean_field_name} and row_id: {row_id}")
#             if row_id.isdigit():
#                 if row_id not in rows:
#                     rows[row_id] = {}
#                 rows[row_id][clean_field_name] = value

#         print("Merged rows:")
#         for row_id, data in rows.items():
#             print(f"Row {row_id}: {data}")

#         # Convert merged rows to a list of dictionaries.
#         unit_parameters_raw = list(rows.values())
#         # Save raw data for reporting purposes.
#         session['unit_parameters_raw'] = unit_parameters_raw

#         # Create a DataFrame from the raw unit parameters.
#         df_units = pd.DataFrame(unit_parameters_raw)
        
#         # Rename columns to match the Excel sheet expected by Stryke.
#         # Adjust these keys if your form uses different names.
#         rename_map = {
#             "facility": "Facility",  # if applicable; often auto-filled
#             "unit": "Unit",          # if applicable; might be generated automatically
#             "type": "Runner Type",
#             "velocity": "intake_vel",
#             "order": "op_order",
#             "H": "H",
#             "RPM": "RPM",
#             "D": "D",
#             "efficiency": "ada",
#             "N": "N",
#             "Qopt": "Qopt",
#             "Qcap": "Qcap",
#             "B": "B",
#             "iota": "iota",
#             "D1": "D1",
#             "D2": "D2",
#             "lambda": "lambda",
#             "roughness": "roughness"
#         }
#         df_units.rename(columns=rename_map, inplace=True)
        
#         # Retrieve the user's unit system.
#         units = session.get('units', 'metric')
        
#         # If units are metric, convert the appropriate fields to imperial.
#         # Conversion factors: meters -> feet: multiply by 3.28084; m³/s -> ft³/s: multiply by 35.31469989.
#         if units == 'metric':
#             conv_length = 3.28084  # For fields measured in meters
#             conv_flow = 35.31469989  # For fields measured in m³/s

#             # List of columns to convert from meters to feet.
#             length_fields = ["intake_vel", "H", "D", "B", "D1", "D2"]
#             # List of columns to convert from m³/s to ft³/s.
#             flow_fields = ["Qopt", "Qcap"]

#             for col in length_fields:
#                 if col in df_units.columns:
#                     # Convert to numeric and multiply by conversion factor.
#                     df_units[col] = pd.to_numeric(df_units[col], errors='coerce') * conv_length

#             for col in flow_fields:
#                 if col in df_units.columns:
#                     df_units[col] = pd.to_numeric(df_units[col], errors='coerce') * conv_flow

#         # Assume unit_params is your DataFrame containing the unit parameters
#         unit_params_path = os.path.join(session['proj_dir'], 'unit_params.csv')
#         df_units.to_csv(unit_params_path, index=False)
        
#         # Store only the file path in the session
#         session['unit_params_file'] = unit_params_path
#         # Save the converted DataFrame into session in JSON-serializable format.
#         #session['unit_parameters_dataframe'] = df_units.to_json(orient='records')
#         #session['unit_parameters'] = df_units.to_dict(orient='records')
        
#         # Debug print the flow scenario DataFrame.
#         print("DEBUG: Unit Parameters DataFrame:")
#         print(df_units, flush=True)
        
#         flash("Unit parameters saved successfully!")
#         return redirect(url_for('operating_scenarios'))
#     return render_template('unit_parameters.html')

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
        op_scen_path = os.path.join(session['proj_dir'], 'op_scen.csv')
        df_os.to_csv(op_scen_path, index=False)
        
        # Store only the file path in the session
        session['op_scen_file'] = op_scen_path        
        flash("Operating scenarios saved successfully!")
        return redirect(url_for('graph_editor'))  # Replace with your next route as needed.
    
        # print("DEBUG: Operating Scenarios DataFrame:")
        # print(df_os, flush=True)
        
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
    import json, pandas as pd

    graph_data = request.get_json()
    proj_dir = session.get('proj_dir')
    if not proj_dir:
        return jsonify(success=False, error="No project directory"), 400
    
    graph_data = request.get_json()
    session['raw_graph_data'] = graph_data
    logger.debug('graph data acquired')
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
    logger.debug('Nodes found')
    
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

    logger.debug('edges created')
    # Build the NetworkX graph
    G = nx.DiGraph()
    for node_id, attrs in simulation_nodes.items():
        G.add_node(node_id, **attrs)
    for source, target, attrs in simulation_edges:
        G.add_edge(source, target, **attrs)

    # Save graph in node-link format
    sim_graph_data = json_graph.node_link_data(G)
    
    # 1) Save full Cytoscape JSON and node-link JSON to disk
    full_path = os.path.join(proj_dir, 'graph_full.json')
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f)

    node_link_path = os.path.join(proj_dir, 'graph_node_link.json')
    with open(node_link_path, 'w', encoding='utf-8') as f:
        json.dump(sim_graph_data, f)

    # 2) Save summary CSVs (nice for auditing)
    nodes_csv = os.path.join(proj_dir, 'graph_nodes.csv')
    edges_csv = os.path.join(proj_dir, 'graph_edges.csv')
    pd.DataFrame(summary_nodes).to_csv(nodes_csv, index=False)
    pd.DataFrame(summary_edges).to_csv(edges_csv, index=False)

    # 3) Keep only pointers in session; drop heavy blobs
    session['graph_files'] = {
        'full': full_path,
        'node_link': node_link_path,
        'nodes_csv': nodes_csv,
        'edges_csv': edges_csv,
    }
    session['graph_summary'] = {"Nodes": summary_nodes, "Edges": summary_edges}
    session.pop('raw_graph_data', None)        # ⬅️ remove big object
    session.pop('simulation_graph', None)      # ⬅️ remove big object
    session.modified = True

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
            "name": "Ascipenser, Great Lakes, Annual",
            "dist": "Weibull",
            "shape": "1.6764",
            "location": "0",
            "scale": "0.0034",
            "max_ent_rate": "0.005",
            "occur_prob": "0.5833",
            "length shape": "0.0108",
            "length location": "-712.6184",
            "length scale": "738.7833"
        },
        {
            "name": "Alosa, Great Lakes, Annual",
            "dist": "Pareto",
            "shape": "0.2737",
            "location": "0",
            "scale": "0.0025",
            "max_ent_rate": "29.95",
            "occur_prob": "0.3889",
            "length shape": "0.091",
            "length location": "-10.9611",
            "length scale": "23.4268"
        },
        {
            "name": "Ambloplites, Great Lakes, Met Winter",
            "dist": "Pareto",
            "shape": "0.5974",
            "location": "0",
            "scale": "0.0052",
            "max_ent_rate": "1.85",
            "occur_prob": "0.3279",
            "length shape": "0.7897",
            "length location": "-0.9668",
            "length scale": "5.5452"
        },
        {
            "name": "Ambloplites, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.9106",
            "location": "0",
            "scale": "0.0424",
            "max_ent_rate": "1.85",
            "occur_prob": "0.3279",
            "length shape": "0.9569",
            "length location": "-0.0496",
            "length scale": "3.5772"
        },
        {
            "name": "Ambloplites, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "1.6537",
            "location": "0",
            "scale": "0.0623",
            "max_ent_rate": "5.5",
            "occur_prob": "0.9677",
            "length shape": "0.0148",
            "length location": "-369.7358",
            "length scale": "381.4015"
        },
        {
            "name": "Ambloplites, Great Lakes, Met Fall",
            "dist": "Log Normal",
            "shape": "1.7352",
            "location": "0",
            "scale": "0.0773",
            "max_ent_rate": "24.62",
            "occur_prob": "0.8495",
            "length shape": "0.0162",
            "length location": "-290.1324",
            "length scale": "302.3072"
        },
        {
            "name": "Ameiurus, Great Lakes, Met Winter",
            "dist": "Log Normal",
            "shape": "1.0344",
            "location": "0",
            "scale": "0.0143",
            "max_ent_rate": "0.12",
            "occur_prob": "0.3085",
            "length shape": "0.0982",
            "length location": "-19.9968",
            "length scale": "27.2126"
        },
        {
            "name": "Ameiurus, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.8906",
            "location": "0",
            "scale": "0.0244",
            "max_ent_rate": "2.78",
            "occur_prob": "0.5899",
            "length shape": "0.896",
            "length location": "-0.0183",
            "length scale": "3.7653"
        },
        {
            "name": "Ameiurus, Great Lakes, Met Summer",
            "dist": "Weibull",
            "shape": "0.4777",
            "location": "0",
            "scale": "0.0449",
            "max_ent_rate": "3.73",
            "occur_prob": "0.7143",
            "length shape": "0.1967",
            "length location": "-30.4348",
            "length scale": "45.0078"
        },
        {
            "name": "Ameiurus, Great Lakes, Met Fall",
            "dist": "Log Normal",
            "shape": "1.3804",
            "location": "0",
            "scale": "0.0128",
            "max_ent_rate": "0.57",
            "occur_prob": "0.5347",
            "length shape": "0.7149",
            "length location": "-1.336",
            "length scale": "10.1348"
        },
        {
            "name": "Amia, Great Lakes, Annual",
            "dist": "Log Normal",
            "shape": "1.2451",
            "location": "0",
            "scale": "0.0053",
            "max_ent_rate": "0.06",
            "occur_prob": "0.338",
            "length shape": "0.8228",
            "length location": "4.294",
            "length scale": "4.9567"
        },
        {
            "name": "Anguilla, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.5447",
            "location": "0",
            "scale": "0.0012",
            "max_ent_rate": "0.11",
            "occur_prob": "0.2381",
            "length shape": "0.2774",
            "length location": "-16.8846",
            "length scale": "64.446"
        },
        {
            "name": "Anguilla, Great Lakes, Met Summer",
            "dist": "Pareto",
            "shape": "0.415",
            "location": "0",
            "scale": "0.0019",
            "max_ent_rate": "0.51",
            "occur_prob": "0.6",
            "length shape": "0.1795",
            "length location": "-10.0737",
            "length scale": "50.5949"
        },
        {
            "name": "Anguilla, Great Lakes, Met Fall & Winter",
            "dist": "Log Normal",
            "shape": "0.9227",
            "location": "0",
            "scale": "0.0056",
            "max_ent_rate": "0.02",
            "occur_prob": "0.1071",
            "length shape": "9.2168",
            "length location": "36.3711",
            "length scale": "1.2944"
        },
        {
            "name": "Catostomus, Great Lakes, Met Winter",
            "dist": "Pareto",
            "shape": "0.5357",
            "location": "0",
            "scale": "0.0051",
            "max_ent_rate": "2.35",
            "occur_prob": "0.4918",
            "length shape": "0.197",
            "length location": "-17.815",
            "length scale": "32.4136"
        },
        {
            "name": "Catostomus, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.6709",
            "location": "0",
            "scale": "0.0011",
            "max_ent_rate": "2.01",
            "occur_prob": "0.6383",
            "length shape": "0.6587",
            "length location": "-2.1796",
            "length scale": "13.8869"
        },
        {
            "name": "Catostomus, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "2.2178",
            "location": "0",
            "scale": "0.0832",
            "max_ent_rate": "11.31",
            "occur_prob": "0.6768",
            "length shape": "0.6814",
            "length location": "-0.6951",
            "length scale": "3.3764"
        },
        {
            "name": "Catostomus, Great Lakes, Met Fall",
            "dist": "Weibull",
            "shape": "0.396",
            "location": "0",
            "scale": "0.0591",
            "max_ent_rate": "28.9",
            "occur_prob": "0.7374",
            "length shape": "0.133",
            "length location": "-4.1585",
            "length scale": "16.7352"
        },
        {
            "name": "Chrosomus, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.6815",
            "location": "0",
            "scale": "0.0292",
            "max_ent_rate": "3.57",
            "occur_prob": "0.4632",
            "length shape": "0.1572",
            "length location": "-13.8922",
            "length scale": "18.4307"
        },
        {
            "name": "Chrosomus, Great Lakes, Met Summer & Fall",
            "dist": "Log Normal",
            "shape": "1.6152",
            "location": "0",
            "scale": "0.0071",
            "max_ent_rate": "0.28",
            "occur_prob": "0.119",
            "length shape": "0.274",
            "length location": "-5.7931",
            "length scale": "9.7798"
        },
        {
            "name": "Coregonus, Great Lakes, Annual",
            "dist": "Log Normal",
            "shape": "0.8508",
            "location": "0",
            "scale": "0.0035",
            "max_ent_rate": "0.01",
            "occur_prob": "0.119",
            "length shape": "0.3129",
            "length location": "-4.0996",
            "length scale": "10.1031"
        },
        {
            "name": "Culaea, Great Lakes, Met Winter",
            "dist": "Log Normal",
            "shape": "1.4119",
            "location": "0",
            "scale": "0.0195",
            "max_ent_rate": "0.79",
            "occur_prob": "0.3158",
            "length shape": "0.4749",
            "length location": "-1.6826",
            "length scale": "4.6771"
        },
        {
            "name": "Culaea, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.4596",
            "location": "0",
            "scale": "0.0619",
            "max_ent_rate": "3.99",
            "occur_prob": "0.7885",
            "length shape": "0.5086",
            "length location": "-1.5036",
            "length scale": "4.3853"
        },
        {
            "name": "Culaea, Great Lakes, Met Summer",
            "dist": "Weibull",
            "shape": "1.1945",
            "location": "0",
            "scale": "0.0255",
            "max_ent_rate": "0.1",
            "occur_prob": "0.3684",
            "length shape": "0.4617",
            "length location": "-2.0154",
            "length scale": "5.271"
        },
        {
            "name": "Culaea, Great Lakes, Met Fall",
            "dist": "Weibull",
            "shape": "0.7098",
            "location": "0",
            "scale": "0.0397",
            "max_ent_rate": "0.22",
            "occur_prob": "0.3333",
            "length shape": "0.2715",
            "length location": "-6.0649",
            "length scale": "10.2824"
        },
        {
            "name": "Cyprinella, Great Lakes, Met Fall & Winter",
            "dist": "Weibull",
            "shape": "1.0455",
            "location": "0",
            "scale": "0.0308",
            "max_ent_rate": "0.18",
            "occur_prob": "0.3605",
            "length shape": "0.0071",
            "length location": "-415.9701",
            "length scale": "421.6351"
        },
        {
            "name": "Cyprinella, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "2.9356",
            "location": "0",
            "scale": "0.0008",
            "max_ent_rate": "4.29",
            "occur_prob": "0.6889",
            "length shape": "0.2899",
            "length location": "-6.0242",
            "length scale": "10.3543"
        },
        {
            "name": "Cyprinella, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "1.2126",
            "location": "0",
            "scale": "0.0218",
            "max_ent_rate": "0.14",
            "occur_prob": "0.6863",
            "length shape": "0.0055",
            "length location": "-522.4786",
            "length scale": "528.356"
        },
        {
            "name": "Cyprinus, Great Lakes, Met Spring & Summer",
            "dist": "Log Normal",
            "shape": "1.0631",
            "location": "0",
            "scale": "0.0033",
            "max_ent_rate": "0.03",
            "occur_prob": "0.1558",
            "length shape": "0.696",
            "length location": "-0.6124",
            "length scale": "8.2879"
        },
        {
            "name": "Cyprinus, Great Lakes, Met Fall",
            "dist": "Log Normal",
            "shape": "0.543",
            "location": "0",
            "scale": "0.004",
            "max_ent_rate": "0.01",
            "occur_prob": "0.3846",
            "length shape": "0.4721",
            "length location": "-4.168",
            "length scale": "13.2867"
        },
        {
            "name": "Dorosoma, Great Lakes, Met Spring and Summer",
            "dist": "Pareto",
            "shape": "0.2606",
            "location": "0",
            "scale": "0.0007",
            "max_ent_rate": "27.72",
            "occur_prob": "0.5",
            "length shape": "0.0643",
            "length location": "-31.8162",
            "length scale": "44.8908"
        },
        {
            "name": " Esox, Great Lakes, Met Fall & Winter",
            "dist": "Pareto",
            "shape": "0.6404",
            "location": "0",
            "scale": "0.0016",
            "max_ent_rate": "0.06",
            "occur_prob": "0.228",
            "length shape": "0.2138",
            "length location": "-27.0125",
            "length scale": "55.4757"
        },
        {
            "name": " Esox, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.3744",
            "location": "0",
            "scale": "0.0132",
            "max_ent_rate": "0.52",
            "occur_prob": "0.4414",
            "length shape": "0.3667",
            "length location": "-5.3217",
            "length scale": "30.9049"
        },
        {
            "name": " Esox, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "1.6164",
            "location": "0",
            "scale": "0.0177",
            "max_ent_rate": "0.4",
            "occur_prob": "0.5528",
            "length shape": "0.3435",
            "length location": "-3.6442",
            "length scale": "12.8822"
        },
        {
            "name": " Etheostoma, Great Lakes, Met Winter",
            "dist": "Pareto",
            "shape": "1.3824",
            "location": "0",
            "scale": "0.0033",
            "max_ent_rate": "0.05",
            "occur_prob": "0.0677",
            "length shape": "0.0643",
            "length location": "-41.3359",
            "length scale": "46.2126"
        },
        {
            "name": " Etheostoma, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.8802",
            "location": "0",
            "scale": "0.0254",
            "max_ent_rate": "13.5",
            "occur_prob": "0.506",
            "length shape": "0.054",
            "length location": "-49.769",
            "length scale": "54.7001"
        },   
        {
            "name": " Etheostoma, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "1.6969",
            "location": "0",
            "scale": "0.005",
            "max_ent_rate": "0.68",
            "occur_prob": "0.3743",
            "length shape": "0.4985",
            "length location": "-1.3703",
            "length scale": "4.0797"
        },  
        {
            "name": " Etheostoma, Great Lakes, Met Fall",
            "dist": "Log Normal",
            "shape": "1.3218",
            "location": "0",
            "scale": "0.0093",
            "max_ent_rate": "0.1",
            "occur_prob": "0.2164",
            "length shape": "0.472",
            "length location": "-2.1768",
            "length scale": "5.7151"
        },
        {
            "name": "Lepomis, Great Lakes, Met Winter",
            "dist": "Log Normal",
            "shape": "0.9881",
            "location": "0",
            "scale": "0.0033",
            "max_ent_rate": "0.18",
            "occur_prob": "0.2078",
            "length shape": "0.6577",
            "length location": "-1.1654",
            "length scale": "5.7882"
        },
        {
            "name": "Lepomis, Great Lakes, Met Spring",
            "dist": "Pareto",
            "shape": "0.3713",
            "location": "0",
            "scale": "0.0014",
            "max_ent_rate": "17.17",
            "occur_prob": "0.4696",
            "length shape": "0.6577",
            "length location": "-1.1654",
            "length scale": "5.7882"
        },
        {
            "name": "Lepomis, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "1.7938",
            "location": "0",
            "scale": "0.0264",
            "max_ent_rate": "7.61",
            "occur_prob": "0.6626",
            "length shape": "0.0175",
            "length location": "-300.5003",
            "length scale": "310.5752"
        },
        {
            "name": "Lepomis, Great Lakes, Met Fall",
            "dist": "Weibull",
            "shape": "0.5825",
            "location": "0",
            "scale": "0.1293",
            "max_ent_rate": "28.64",
            "occur_prob": "0.5967",
            "length shape": "0.5825",
            "length location": "-1.9339",
            "length scale": "6.6951"
        },
        {
            "name": "Lota, Great Lakes, Met Winter",
            "dist": "Pareto",
            "shape": "0.5486",
            "location": "0",
            "scale": "0.0033",
            "max_ent_rate": "1.32",
            "occur_prob": "0.1227",
            "length shape": "0.167",
            "length location": "-36.1855",
            "length scale": "59.929"
        },
        {
            "name": "Lota, Great Lakes, Met Spring",
            "dist": "Weibull",
            "shape": "0.6415",
            "location": "0",
            "scale": "0.0273",
            "max_ent_rate": "0.35",
            "occur_prob": "0.4902",
            "length shape": "0.4221",
            "length location": "6.6695",
            "length scale": "10.4073"
        },
        {
            "name": "Lota, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "1.3717",
            "location": "0",
            "scale": "0.0017",
            "max_ent_rate": "0.26",
            "occur_prob": "0.6078",
            "length shape": "0.5136",
            "length location": "-2.4821",
            "length scale": "9.1436"
        },
        {
            "name": "Lota, Great Lakes, Met Fall",
            "dist": "Weibull",
            "shape": "2.1153",
            "location": "0",
            "scale": "0.0042",
            "max_ent_rate": "0.007",
            "occur_prob": "0.3137",
            "length shape": "0.5506",
            "length location": "1.1179",
            "length scale": "15.9954"
        },
        {
            "name": "Luxilus, Great Lakes, Met Fall & Winter",
            "dist": "Log Normal",
            "shape": "2.6755",
            "location": "0",
            "scale": "0.1121",
            "max_ent_rate": "4.5",
            "occur_prob": "0.5041",
            "length shape": "0.0152",
            "length location": "-182.0854",
            "length scale": "189.282"
        },
        {
            "name": "Luxilus, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.5964",
            "location": "0",
            "scale": "0.159",
            "max_ent_rate": "1.45",
            "occur_prob": "0.4928",
            "length shape": "0.2284",
            "length location": "-5.031",
            "length scale": "13.0993"
        },
        {
            "name": "Luxilus, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "3.0011",
            "location": "0",
            "scale": "0.0706",
            "max_ent_rate": "5.18",
            "occur_prob": "0.7467",
            "length shape": "0.0109",
            "length location": "-253.1085",
            "length scale": "259.427"
        },
        {
            "name": "Micropterus, Great Lakes, Met Winter",
            "dist": "Log Normal",
            "shape": "0.7591",
            "location": "0",
            "scale": "0.0079",
            "max_ent_rate": "0.07",
            "occur_prob": "0.3077",
            "length shape": "0.1181",
            "length location": "-20.6739",
            "length scale": "29.5847"
        },
        {
            "name": "Micropterus, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.3713",
            "location": "0",
            "scale": "0.0081",
            "max_ent_rate": "0.13",
            "occur_prob": "0.4423",
            "length shape": "0.0192",
            "length location": "-554.6249",
            "length scale": "576.789"
        },
        {
            "name": "Micropterus, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "0.5405",
            "location": "0",
            "scale": "0.1621",
            "max_ent_rate": "6.05",
            "occur_prob": "0.6964",
            "length shape": "0.9155",
            "length location": "-0.3861",
            "length scale": "3.8903"
        },
        {
            "name": "Micropterus, Great Lakes, Met Fall",
            "dist": "Log Normal",
            "shape": "1.5437",
            "location": "0",
            "scale": "0.0283",
            "max_ent_rate": "1.27",
            "occur_prob": "0.6726",
            "length shape": "0.4116",
            "length location": "-1.1232",
            "length scale": "11.2873"
        },
        {
            "name": "Morone, Great Lakes, Met Fall & Winter",
            "dist": "Log Normal",
            "shape": "1.214",
            "location": "0",
            "scale": "0.0156",
            "max_ent_rate": "0.08",
            "occur_prob": "0.6667",
            "length shape": "0.4341",
            "length location": "1.834",
            "length scale": "8.0233"
        },
        {
            "name": "Morone, Great Lakes, Met Spring & Summer",
            "dist": "Log Normal",
            "shape": "2.1",
            "location": "0",
            "scale": "0.0244",
            "max_ent_rate": "0.42",
            "occur_prob": "0.4706",
            "length shape": "0.527",
            "length location": "1.0488",
            "length scale": "8.3202"
        },
        {
            "name": "Moxostoma, Great Lakes, Met Winter",
            "dist": "Log Normal",
            "shape": "1.3572",
            "location": "0",
            "scale": "0.0198",
            "max_ent_rate": "0.3",
            "occur_prob": "0.1837",
            "length shape": "0.7031",
            "length location": "-1.3125",
            "length scale": "7.2819"
        },
        {
            "name": "Moxostoma, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "0.9826",
            "location": "0",
            "scale": "0.0116",
            "max_ent_rate": "0.07",
            "occur_prob": "0.3472",
            "length shape": "0.109",
            "length location": "-123.0021",
            "length scale": "146.2813"
        },
        {
            "name": "Moxostoma, Great Lakes, Met Summer & Fall",
            "dist": "Log Normal",
            "shape": "1.3686",
            "location": "0",
            "scale": "0.0087",
            "max_ent_rate": "0.7",
            "occur_prob": "0.4833",
            "length shape": "0.5167",
            "length location": "-5.1992",
            "length scale": "18.7"
        },
        {
            "name": "Nocomis, Great Lakes, Annual",
            "dist": "Log Normal",
            "shape": "1.7634",
            "location": "0",
            "scale": "0.0123",
            "max_ent_rate": "0.2",
            "occur_prob": "0.337",
            "length shape": "0.7352",
            "length location": "-0.6709",
            "length scale": "3.9941"
        },
        {
            "name": "Notemigonus, Great Lakes, Met Winter",
            "dist": "Pareto",
            "shape": "0.5433",
            "location": "0",
            "scale": "0.0574",
            "max_ent_rate": "1.31",
            "occur_prob": "0.5102",
            "length shape": "0.1365",
            "length location": "-5.0983",
            "length scale": "12.5858"
        },
        {
            "name": "Notemigonus, Great Lakes, Met Spring",
            "dist": "Pareto",
            "shape": "0.3",
            "location": "0",
            "scale": "0.0008",
            "max_ent_rate": "1.53",
            "occur_prob": "0.64",
            "length shape": "0.7596",
            "length location": "-0.7386",
            "length scale": "4.2723"
        },
        {
            "name": "Notemigonus, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "2.8014",
            "location": "0",
            "scale": "0.0618",
            "max_ent_rate": "6.42",
            "occur_prob": "0.6543",
            "length shape": "0.0207",
            "length location": "-130.7952",
            "length scale": "136.8216"
        },
        {
            "name": "Notemigonus, Great Lakes, Met Fall",
            "dist": "Log Normal",
            "shape": "3.111",
            "location": "0",
            "scale": "0.0542",
            "max_ent_rate": "8.92",
            "occur_prob": "0.6914",
            "length shape": "0.0153",
            "length location": "-101.087",
            "length scale": "108.5648"
        },
        {
            "name": "Noturus, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.3064",
            "location": "0",
            "scale": "0.0073",
            "max_ent_rate": "0.06",
            "occur_prob": "0.3488",
            "length shape": "0.3015",
            "length location": "-10.8068",
            "length scale": "19.8945"
        },
        {
            "name": "Noturus, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "1.0047",
            "location": "0",
            "scale": "0.01",
            "max_ent_rate": "0.09",
            "occur_prob": "0.6889",
            "length shape": "0.2312",
            "length location": "-11.9586",
            "length scale": "20.4888"
        },
        {
            "name": "Notropis, Great Lakes, Met Winter",
            "dist": "Pareto",
            "shape": "0.9849",
            "location": "0",
            "scale": "0.0045",
            "max_ent_rate": "0.26",
            "occur_prob": "0.2647",
            "length shape": "0.0023",
            "length location": "-1153.8089",
            "length scale": "1160.4858"
        },
        {
            "name": "Notropis, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.4117",
            "location": "0",
            "scale": "0.007",
            "max_ent_rate": "0.64",
            "occur_prob": "0.4844",
            "length shape": "0.596",
            "length location": "-0.3735",
            "length scale": "4.5076"
        },
        {
            "name": "Notropis, Great Lakes, Met Summer",
            "dist": "Pareto",
            "shape": "0.4168",
            "location": "0",
            "scale": "0.0011",
            "max_ent_rate": "2.01",
            "occur_prob": "0.4346",
            "length shape": "0.0133",
            "length location": "-214.065",
            "length scale": "219.2391"
        },
        {
            "name": "Notropis, Great Lakes, Met Fall",
            "dist": "Log Noraml",
            "shape": "1.0752",
            "location": "0",
            "scale": "0.0065",
            "max_ent_rate": "0.23",
            "occur_prob": "0.23",
            "length shape": "0.5057",
            "length location": "-1.4499",
            "length scale": "4.2016"
        },
        {
            "name": "Oncorhynchus, Great Lakes, Met Winter",
            "dist": "Weibull",
            "shape": "0.6833",
            "location": "0",
            "scale": "0.0073",
            "max_ent_rate": "0.05",
            "occur_prob": "0.2303",
            "length shape": "0.0582",
            "length location": "-74.0199",
            "length scale": "94.144"
        },
        {
            "name": "Oncorhynchus, Great Lakes, Met Spring, Summer, & Fall",
            "dist": "Weibull",
            "shape": "0.6833",
            "location": "0",
            "scale": "0.0073",
            "max_ent_rate": "0.05",
            "occur_prob": "0.2303",
            "length shape": "0.0582",
            "length location": "-74.0199",
            "length scale": "94.144"
        },
        {
            "name": "Osmerus, Great Lakes, Annual",
            "dist": "Log Normal",
            "shape": "3.8764",
            "location": "0",
            "scale": "0.0012",
            "max_ent_rate": "400.12",
            "occur_prob": "0.5882",
            "length shape": "0.2795",
            "length location": "0.0606",
            "length scale": "7.795"
        },
        {
            "name": "Perca, Great Lakes, Met Winter",
            "dist": "Pareto",
            "shape": "0.3831",
            "location": "0",
            "scale": "24.4615",
            "max_ent_rate": "3.54",
            "occur_prob": "0.4426",
            "length shape": "0.1231",
            "length location": "-14.7021",
            "length scale": "24.4615"
        },
        {
            "name": "Perca, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "2.8141",
            "location": "0",
            "scale": "0.0871",
            "max_ent_rate": "48.32",
            "occur_prob": "0.8276",
            "length shape": "0.2716",
            "length location": "-9.5459",
            "length scale": "16.8826"
        },
        {
            "name": "Perca, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "3.2221",
            "location": "0",
            "scale": "0.2197",
            "max_ent_rate": "32.81",
            "occur_prob": "0.7849",
            "length shape": "0.537",
            "length location": "-1.1236",
            "length scale": "3.6604"
        },
        {
            "name": "Perca, Great Lakes, Met Fall",
            "dist": "Log Normal",
            "shape": "2.4307",
            "location": "0",
            "scale": "0.1207",
            "max_ent_rate": "34.7",
            "occur_prob": "0.7957",
            "length shape": "0.2627",
            "length location": "-0.3852",
            "length scale": "8.2808"
        },
        {
            "name": "Pimephales, Great Lakes, Met Winter",
            "dist": "Pareto",
            "shape": "1.8322",
            "location": "0",
            "scale": "0.0045",
            "max_ent_rate": "0.11",
            "occur_prob": "0.2329",
            "length shape": "0.0036",
            "length location": "-496.3134",
            "length scale": "503.4153"
        },
        {
            "name": "Pimephales, Great Lakes, Met Spring",
            "dist": "Weibull",
            "shape": "0.5481",
            "location": "0",
            "scale": "0.0316",
            "max_ent_rate": "2.63",
            "occur_prob": "0.4915",
            "length shape": "0.0043",
            "length location": "-444.0727",
            "length scale": "451.2193"
        },
        {
            "name": "Pimephales, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "1.3049",
            "location": "0",
            "scale": "0.0116",
            "max_ent_rate": "0.13",
            "occur_prob": "0.4419",
            "length shape": "0.009",
            "length location": "-294.8785",
            "length scale": "300.4008"
        },
        {
            "name": "Pimephales, Great Lakes, Met Fall",
            "dist": "Log Normal",
            "shape": "0.9676",
            "location": "0",
            "scale": "0.0062",
            "max_ent_rate": "0.24",
            "occur_prob": "0.2481",
            "length shape": "0.0049",
            "length location": "-525.1864",
            "length scale": "531.6184"
        },
        {
            "name": "Pomoxis, Great Lakes, Met Winter",
            "dist": "Log Normal",
            "shape": "1.7493",
            "location": "0",
            "scale": "0.0473",
            "max_ent_rate": "4.77",
            "occur_prob": "0.4032",
            "length shape": "0.2192",
            "length location": "-4.8473",
            "length scale": "12.5246"
        },
        {
            "name": "Pomoxis, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.5603",
            "location": "0",
            "scale": "0.0187",
            "max_ent_rate": "1.49",
            "occur_prob": "0.6129",
            "length shape": "0.3344",
            "length location": "-5.6749",
            "length scale": "15.676"
        },
        {
            "name": "Pomoxis, Great Lakes, Met Summer",
            "dist": "Weibull",
            "shape": "0.4162",
            "location": "0",
            "scale": "0.2537",
            "max_ent_rate": "4.81",
            "occur_prob": "0.6566",
            "length shape": "0.7861",
            "length location": "-0.5388",
            "length scale": "3.5888"
        },
        {
            "name": "Pomoxis, Great Lakes, Met Fall",
            "dist": "Log Normal",
            "shape": "2.3606",
            "location": "0",
            "scale": "1162",
            "max_ent_rate": "5.76",
            "occur_prob": "0.7576",
            "length shape": "0.1213",
            "length location": "-13.3987",
            "length scale": "20.7735"
        },
        {
            "name": "Rhinichthys, Great Lakes, Met Fall & Winter",
            "dist": "Log Normal",
            "shape": "1.1916",
            "location": "0",
            "scale": "0.0017",
            "max_ent_rate": "0.07",
            "occur_prob": "0.2636",
            "length shape": "0.0047",
            "length location": "-538.9506",
            "length scale": "545.041"
        },
        {
            "name": "Rhinichthys, Great Lakes, Met Spring & Summer",
            "dist": "Log Normal",
            "shape": "1.8139",
            "location": "0",
            "scale": "0.0015",
            "max_ent_rate": "0.54",
            "occur_prob": "0.475",
            "length shape": "0.01",
            "length location": "-166.1794",
            "length scale": "173.6912"
        },
        {
            "name": "Salmo, Great Lakes, Met Fall & Winter",
            "dist": "Log Normal",
            "shape": "1.4869",
            "location": "0",
            "scale": "0.0013",
            "max_ent_rate": "0.34",
            "occur_prob": "0.2679",
            "length shape": "0.9693",
            "length location": "-4.6793",
            "length scale": "5.3207"
        },
        {
            "name": "Salmo, Great Lakes, Met Spring & Summer",
            "dist": "Log Normal",
            "shape": "1.2363",
            "location": "0",
            "scale": "0.0058",
            "max_ent_rate": "0.2",
            "occur_prob": "0.3636",
            "length shape": "0.3118",
            "length location": "-8.0675",
            "length scale": "26.9977"
        },
        {
            "name": "Salvelinus, Great Lakes, Annual",
            "dist": "Log Normal",
            "shape": "1.6537",
            "location": "0",
            "scale": "0.0088",
            "max_ent_rate": "0.24",
            "occur_prob": "0.2273",
            "length shape": "0.0819",
            "length location": "-92.9551",
            "length scale": "110.736"
        },
        {
            "name": "Sander, Great Lakes, Met Fall & Winter",
            "dist": "Log Normal",
            "shape": "1.7446",
            "location": "0",
            "scale": "0.0363",
            "max_ent_rate": "0.85",
            "occur_prob": "0.5221",
            "length shape": "0.4177",
            "length location": "-1.0242",
            "length scale": "16.4924"
        },
        {
            "name": "Sander, Great Lakes, Met Spring & Summer",
            "dist": "Log Normal",
            "shape": "0.5305",
            "location": "0",
            "scale": "9.8148",
            "max_ent_rate": "5.22",
            "occur_prob": "0.7654",
            "length shape": "0.5305",
            "length location": "-2.2348",
            "length scale": "9.8148"
        },
        {
            "name": "Semotilus, Great Lakes, Met Fall & Winter",
            "dist": "Pareto",
            "shape": "0.5018",
            "location": "0",
            "scale": "0.0009",
            "max_ent_rate": "0.62",
            "occur_prob": "0.1186",
            "length shape": "0.124",
            "length location": "-27.1355",
            "length scale": "37.8381"
        },
        {
            "name": "Semotilus, Great Lakes, Met Spring & Summer",
            "dist": "Log Normal",
            "shape": "1.2676",
            "location": "0",
            "scale": "0.0069",
            "max_ent_rate": "0.12",
            "occur_prob": "0.3636",
            "length shape": "0.1264",
            "length location": "-22.8442",
            "length scale": "32.6316"
        },
        {
            "name": "Umbra, Great Lakes, Met Fall & Winter",
            "dist": "Log Normal",
            "shape": "0.8115",
            "location": "0",
            "scale": "0.0073",
            "max_ent_rate": "0.06",
            "occur_prob": "0.2824",
            "length shape": "0.167",
            "length location": "-15.5853",
            "length scale": "20.9652"
        },
        {
            "name": "Umbra, Great Lakes, Met Spring & Summer",
            "dist": "Log Normal",
            "shape": "1.8176",
            "location": "0",
            "scale": "0.0185",
            "max_ent_rate": "2.32",
            "occur_prob": "0.4936",
            "length shape": "0.0055",
            "length location": "-382.586",
            "length scale": "389.8177"
        },


        # ...
    ]

    if request.method == 'POST':
        #print("starting population post route", flush=True)
        
        # Print all form data
        form_data = dict(request.form)
        #print("Received form data:", form_data, flush=True)
        
        species_name = request.form.get('species_name')
        common_name = request.form.get('common_name')
        scenario = request.form.get('scenario')
        simulate_choice = request.form.get('simulateChoice')
        iterations = request.form.get('iterations')
        vertical_habitat = request.form.get('vertical_habitat')
        beta_0 = request.form.get('beta_0')
        beta_1 = request.form.get('beta_1')
        
        # print("Basic Info: species_name=%s, common_name=%s, scenario=%s, simulate_choice=%s, iterations=%s, vertical_habitat=%s, beta_0=%s, beta_1=%s" %
        #       (species_name, common_name, scenario, simulate_choice, iterations, vertical_habitat, beta_0, beta_1), flush=True)
        
        pop_data = {
            "Species": species_name,
            "Common Name": common_name,
            "Scenario": scenario,
            "Simulate Choice": simulate_choice,
            "Iterations": iterations,
            "Entrainment Choice": None,  # Will fill if needed
            "Modeled Species": None,     # Will fill if needed
        }
        #print("Initial pop_data:", pop_data, flush=True)
        
        # Helper for float conversion
        def safe_float(val):
            try:
                return float(val)
            except Exception as e:
                print("safe_float failed for value '{}': {}".format(val, e), flush=True)
                return None
    
        pop_data["vertical_habitat"] = vertical_habitat
        pop_data["beta_0"] = safe_float(beta_0)
        pop_data["beta_1"] = safe_float(beta_1)
        
        units = session.get('units', 'metric')
        
        # If user chooses “entrainment event”
        if simulate_choice == 'entrainment event':
            entrainment_choice = request.form.get('entrainmentChoice')
            pop_data["Entrainment Choice"] = entrainment_choice
            #print("Entrainment event selected, choice: {}".format(entrainment_choice), flush=True)
            
            Ucrit_input = request.form.get('Ucrit')
            length_mean_input = request.form.get('length_mean')
            length_sd_input = request.form.get('length_sd')
            
            # Convert Ucrit
            Ucrit_ft = None
            if Ucrit_input:
                try:
                    Ucrit_val = float(Ucrit_input)
                    Ucrit_ft = Ucrit_val * 3.28084 if units == 'metric' else Ucrit_val
                except Exception as e:
                    print("Ucrit conversion failed for input '{}': {}".format(Ucrit_input, e), flush=True)
            pop_data["U_crit"] = Ucrit_ft
            
            # If “modeled”
            if entrainment_choice == 'modeled':
                modeled_species = request.form.get('modeledSpecies')
                pop_data["Modeled Species"] = modeled_species
                #print("Modeled entrainment selected, species: {}".format(modeled_species), flush=True)
                
                selected_species = next((s for s in species_defaults if s["name"] == modeled_species), None)
                if selected_species:
                    pop_data["shape"] = selected_species["shape"]
                    pop_data["location"] = selected_species["location"]
                    pop_data["scale"] = selected_species["scale"]
                    pop_data["max_ent_rate"] = selected_species["max_ent_rate"]
                    pop_data["occur_prob"] = selected_species["occur_prob"]
                    pop_data["dist"] = selected_species["dist"]
                    pop_data["length shape"] = safe_float(selected_species["length shape"])
                    pop_data["length location"] = safe_float(selected_species["length location"])
                    pop_data["length scale"] = safe_float(selected_species["length scale"])
                    
                    if length_mean_input:
                        try:
                            length_mean_in = float(length_mean_input)
                            if units == 'metric': 
                                length_mean_in /= 25.4
                            pop_data["length location"] = length_mean_in
                        except Exception as e:
                            print("Length mean conversion failed: {}".format(e), flush=True)
                    if length_sd_input:
                        try:
                            length_sd_in = float(length_sd_input)
                            if units == 'metric':
                                length_sd_in /= 25.4
                            pop_data["length scale"] = length_sd_in
                        except Exception as e:
                            print("Length SD conversion failed: {}".format(e), flush=True)
                else:
                    pop_data["shape"] = None
                    pop_data["location"] = None
                    pop_data["scale"] = None
                #print("pop_data after modeled branch:", pop_data, flush=True)
            
            # If “empirical”
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
                #print("pop_data after empirical branch:", pop_data, flush=True)
        
        # If user chooses “starting population” (or something else)
        else:
            # Optionally process starting population if needed
            print("Starting population option selected", flush=True)
        
        # Final conversions for length_mean, length_sd
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
        
        #print("Final pop_data before DataFrame creation:", pop_data, flush=True)
        
        #session['population_data'] = pop_data
        
        import pandas as pd
        df_population = pd.DataFrame([pop_data])
        #print("DataFrame created with shape:", df_population.shape, flush=True)
        
        expected_columns = [
            "Species", "Common Name", "Scenario", "Iterations", "Fish",
            "vertical_habitat", "beta_0", "beta_1",
            "dist","shape", "location", "scale",
            "max_ent_rate", "occur_prob",
            "Length_mean", "Length_sd", "U_crit",
            "length shape", "length location", "length scale"
        ]
        for col in expected_columns:
            if col not in df_population.columns:
                df_population[col] = None
        df_population = df_population[expected_columns]
        #print("DataFrame after ensuring expected columns:", flush = True)
        #print (df_population, flush=True)
        
        #session['population_dataframe_for_sim'] = df_population.to_json(orient='records')
        # After creating and saving the DataFrame
        df_population_clean = df_population.where(pd.notnull(df_population), None)
        session['population_data_for_sim'] = df_population_clean.to_dict(orient='records')
        #print ('population dataframe for modeling:', session.get('population_data_for_sim'), flush=True)
        summary_column_mapping = {
            "Species": "Species Name",
            "Common Name": "Common Name",
            "Scenario": "Scenario",
            "Iterations": "Iterations",
            "Fish": "Fish",
            "vertical_habitat": "Vertical Habitat",
            "beta_0": "Beta 0",
            "beta_1": "Beta 1",
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
        
        # get the project directory
        proj_dir = session["proj_dir"]  # Or your configured project directory
        #print ('project directory:', proj_dir, flush = True)
        
        # make a csv path, save the dataframe and check
        pop_csv_path = os.path.join(proj_dir, "population_params.csv")
        df_population.to_csv(pop_csv_path, index=False)   
        session['population_csv_path'] = pop_csv_path
        df_check = pd.read_csv(pop_csv_path)
        print("CSV Headers:", df_check.columns.tolist(), flush=True)
        #print("Saved population parameters to file:", pop_csv_path, flush=True)

        #print("Population DataFrame for summary:", session.get('population_dataframe_for_summary'), flush=True)
        
        print("Population parameters saved successfully! Redirecting...", flush=True)
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
        #print("Found unit_params_file in session:", unit_params_file)
        if os.path.exists(unit_params_file):
            try:
                df_unit = pd.read_csv(unit_params_file)
                unit_parameters = df_unit.to_dict(orient='records')
                unit_columns = list(df_unit.columns)
                #print("Loaded unit parameters:", unit_parameters)
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
        #print("Found operating_scenarios_file in session:", ops_file)
        if os.path.exists(ops_file):
            try:
                df_ops = pd.read_csv(ops_file)
                operating_scenarios = df_ops.to_dict(orient='records')
                #print("Loaded operating scenarios:", operating_scenarios)
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
    #print("Flow scenarios:", flow_scenarios)

    # --- Graph Data ---
    graph_summary = session.get('graph_summary', {"Nodes": [], "Edges": []})
    graph_nodes = graph_summary.get('Nodes', [])
    graph_edges = graph_summary.get('Edges', [])
    #print("Graph summary (Nodes):", graph_nodes)
    #print("Graph summary (Edges):", graph_edges)

    # --- Other Data ---
    facilities_data = session.get('facilities_data', [])
    population_parameters = []

    pop_csv_path = session.get('population_csv_path')
    proj_dir = session.get('proj_dir')
    
    if not pop_csv_path and proj_dir:
        candidate = os.path.join(proj_dir, "population_params.csv")
        if os.path.exists(candidate):
            pop_csv_path = candidate
            session['population_csv_path'] = candidate  # repopulate pointer
            session.modified = True
    
    population_parameters = []
    if pop_csv_path and os.path.exists(pop_csv_path):
        df_pop = pd.read_csv(pop_csv_path)
        population_parameters = df_pop.to_dict(orient='records')
    else:
        # last-ditch fallback
        population_parameters = session.get('population_data_for_sim', [])

    # pop_csv_path = session.get('population_csv_path')
    
    # if pop_csv_path:
    #     print("Found population CSV file in session:", pop_csv_path, flush=True)
    #     if os.path.exists(pop_csv_path):
    #         try:
    #             df_pop = pd.read_csv(pop_csv_path)
    #             population_parameters = df_pop.to_dict(orient='records')
    #         except Exception as e:
    #             print("Error reading population CSV file:", e, flush=True)
    #     else:
    #         print("Population CSV file not found on disk:", pop_csv_path, flush=True)
    # else:
    #     print("No population CSV file key in session.", flush=True)
    #     # Use any in-memory data if we have it
    #     population_parameters = session.get('population_data_for_sim', [])

    simulation_graph = session.get('simulation_graph', {})
    #print ('Loaded simulation graph:', simulation_graph, flush = True)
    
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
    print ('model setup summary complete', flush = True)

def _attach_queue_logger(log_queue):
    h = QueueLogHandler(log_queue)
    h.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    h.setFormatter(fmt)
    # avoid duplicates if called twice
    for existing in logger.handlers:
        if isinstance(existing, QueueLogHandler):
            break
    else:
        logger.addHandler(h)

def _close_hdf5_handles(obj):
    """Best-effort: flush/close any HDF5/HDFStore handles hanging off the sim object."""
    candidates = []
    for name in dir(obj):
        # avoid dunder & bound methods
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
        except Exception:
            continue
        candidates.append(attr)

    for a in candidates:
        try:
            # pandas
            if hasattr(pd, "HDFStore") and isinstance(a, pd.HDFStore):
                try: a.flush()
                except Exception: pass
                try: a.close()
                except Exception: pass
            # PyTables file
            if tables and getattr(tables, "File", None) and isinstance(a, tables.File):
                try: a.flush()
                except Exception: pass
                try: a.close()
                except Exception: pass
            # h5py
            import h5py
            if isinstance(a, h5py.File):
                try: a.flush()
                except Exception: pass
                try: a.close()
                except Exception: pass
        except Exception:
            # ignore any introspection weirdness
            pass


def run_simulation_in_background_custom(data_dict, log_queue):
    start_ts = datetime.now()
    old_stdout = sys.stdout
    sys.stdout = QueueStream(log_queue)  # route print() to SSE
    try:
        print(f"[INFO] Simulation process started at {start_ts.isoformat()}", flush=True)

        proj_dir = data_dict.get('proj_dir')
        if not proj_dir:
            raise ValueError("proj_dir missing")
        os.makedirs(proj_dir, exist_ok=True)
        print(f"[INFO] Project directory: {proj_dir}", flush=True)

        print("[INFO] Creating simulation object...", flush=True)
        #sim = stryke.simulation(proj_dir=proj_dir, output_name="WebAppModel", wks=None)
        output_name = data_dict.get('output_name', 'WebAppModel')
        sim = stryke.simulation(proj_dir=proj_dir, output_name=output_name, wks=None)
        sim.webapp_import(data_dict, output_name=output_name)

        print("[INFO] Importing model inputs...", flush=True)
        #sim.webapp_import(data_dict, output_name="WebAppModel")
        sim.project_name  = data_dict.get('project_name')  or 'N/A'
        sim.project_notes = data_dict.get('project_notes') or 'N/A'
        sim.model_setup   = data_dict.get('model_setup')   or 'N/A'
        sim.units_session = data_dict.get('units')         or 'N/A'

        print("[INFO] Running simulation...", flush=True)
        sim.run()

        print("[INFO] Building summary...", flush=True)
        sim.summary()

        print("[INFO] Generating report HTML...", flush=True)
        report_html = generate_report(sim)
        report_path = os.path.join(proj_dir, "simulation_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        with open(os.path.join(proj_dir, "report_path.txt"), "w", encoding="utf-8") as f:
            f.write(report_path)
        print(f"[SUCCESS] Report written to {report_path}", flush=True)

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", flush=True)
        logger.exception("Simulation failed in background worker.")

    finally:
        sys.stdout = old_stdout
        try:
            log_queue.put("[Simulation Complete]")  # unified sentinel
            _close_hdf5_handles(sim)
        except Exception:
            pass

@app.route('/run_simulation', methods=['POST'], endpoint='run_simulation')
def run_simulation():
    # in run_simulation()
    output_name = f"WebAppModel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session['output_name'] = output_name  # so /report can find it later
    data_dict = {
        'proj_dir': session.get('proj_dir'),
        # ... existing fields ...
        'output_name': output_name,
    }

    proj_dir = session.get('proj_dir')
    if not proj_dir:
        flash("Session expired or project not initialized.")
        return redirect(url_for('index'))

    # population CSV is optional; don’t crash if missing
    pop_df = None
    pop_csv = session.get('population_csv_path')
    if pop_csv:
        try:
            pop_df = pd.read_csv(pop_csv, low_memory=False)
        except Exception as e:
            logger.exception("Failed reading population CSV")
            print(f"[WARN] population_csv_path unreadable: {e}", flush=True)

    graph_data = None
    graph_summary = session.get('graph_summary')
    gf = session.get('graph_files') or {}
    node_link_path = gf.get('node_link')
    if node_link_path and os.path.exists(node_link_path):
        with open(node_link_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

    data_dict = {
        'proj_dir':      proj_dir,
        'project_name':  session.get('project_name'),
        'project_notes': session.get('project_notes'),
        'model_setup':   session.get('model_setup'),
        'units':         session.get('units'),
        'facilities':    session.get('facilities_data'),
        'unit_parameters_file':   session.get('unit_params_file'),
        'operating_scenarios_file': session.get('op_scen_file'),
        'population':    pop_df,
        'flow_scenarios': session.get('flow_scenario'),
        'hydrograph_file': session.get('hydrograph_file'),
        'graph_data':     graph_data,
        'graph_summary':  graph_summary,
        'units_system':   session.get('units', 'imperial'),
        'simulation_mode': session.get('simulation_mode', 'multiple_powerhouses_simulated_entrainment_routing'),
    }

    t = threading.Thread(target=run_simulation_in_background_custom,
                         args=(data_dict, LOG_QUEUE),
                         daemon=True)
    t.start()
    flash("Simulation started! Check logs for progress.")

    # send user to a page that listens to /stream
    return redirect(url_for('simulation_logs'))

# @app.route('/simulation_logs')
# def simulation_logs():
#     return render_template('simulation_logs.html')

@app.route("/simulation_logs")
def simulation_logs():
    # this template should attach to your SSE /stream
    return render_template("simulation_logs.html")

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

@app.route('/debug_report_path')
def debug_report_path():
    import os
    try:
        proj_dir = session.get('proj_dir')
        path_file = os.path.join(proj_dir, "report_path.txt")
        if not os.path.exists(path_file):
            return f"<p>report_path.txt not found: {path_file}</p>", 404
        with open(path_file, 'r') as f:
            path_contents = f.read().strip()
        return f"<p>Contents of report_path.txt: {path_contents}</p>"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<p>Fatal error: {e}</p>", 500

@app.route('/report')
def report():
    proj_dir = session.get('proj_dir')
    marker = os.path.join(proj_dir, "report_path.txt")
    if os.path.exists(marker):
        with open(marker, 'r', encoding='utf-8') as f:
            report_path = f.read().strip()
    else:
        report_path = os.path.join(proj_dir, "simulation_report.html")
    
    if not os.path.exists(report_path):
        logger.debug("[ERROR] simulation_report.html does not exist")
        return f"<h1>Report not found: {report_path}</h1>", 404

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
    except UnicodeDecodeError:
        logger.debug("[WARNING] UTF-8 decode error — trying Latin-1 fallback")
        with open(report_path, 'r', encoding='latin1') as f:
            report_html = f.read()

    # If report_html is a complete HTML document, return it directly:
    return report_html

def generate_report(sim):
    """
    Generate the comprehensive HTML report for the simulation.
    Adds:
      - Robust HDF5 open with retry/backoff when the file is momentarily locked.
      - Guaranteed close of the HDFStore via try/finally.
      - Optional concurrent-read mode via env var HDF5_ALLOW_CONCURRENT_READS=1.
    """
    import os
    import time
    import io, base64
    from datetime import datetime
    import pandas as pd
    import matplotlib.pyplot as plt

    # Optional: allow reading while the writer still has the file open.
    # Enable in your env (Railway/compose) with HDF5_ALLOW_CONCURRENT_READS=1
    if os.getenv("HDF5_ALLOW_CONCURRENT_READS") == "1":
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    # Set global minimum font size for all plots
    plt.rcParams.update({'font.size': 8})

    hdf_path = os.path.join(sim.proj_dir, f"{sim.output_name}.h5")
    if not os.path.exists(hdf_path):
        return "<p>Error: HDF file not found. Please run the simulation first.</p>"

    # --- Open HDF with retry/backoff to survive transient file locks ---
    attempts = 12         # ~ a few seconds total with exponential backoff
    base_delay = 0.15
    store = None
    try:
        for i in range(attempts):
            try:
                store = pd.HDFStore(hdf_path, mode='r')
                break
            except Exception as e:
                msg = str(e)
                # HDF5 lock messages often contain these substrings
                if ("unable to lock file" in msg) or ("Resource temporarily unavailable" in msg):
                    if i == attempts - 1:
                        if 'logger' in globals():
                            logger.error("HDF5 busy after retries: %s", msg)
                        return "<p>The report file is busy. Please try again in a few seconds.</p>"
                    # Exponential backoff
                    time.sleep(base_delay * (1.5 ** i))
                    continue
                # Different error → surface it
                raise

        if 'logger' in globals():
            logger.debug('hdf file exists and opened %s', hdf_path)

        report_sections = [
            "<div style='margin: 10px;'>"
            "  <button onclick=\"window.location.href='/'\" style='padding:10px;'>Home and Logout</button>"
            "</div>",
            f"<h1>Simulation Report for Project: {getattr(sim, 'project_name', 'N/A')}</h1>",
            f"<p><strong>Project Notes:</strong> {getattr(sim, 'project_notes', 'N/A')}</p>",
            f"<p><strong>Model Setup:</strong> {getattr(sim, 'model_setup', 'N/A')}</p>",
            f"<p><strong>Units:</strong> {getattr(sim, 'units_session', 'N/A')}</p>",
            f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"<p>HDF keys found: {store.keys()}</p>"
        ]

        units = sim.output_units

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

        def add_section(title, key, units):
            report_sections.append(f"<h2>{title}</h2>")
            if key in store.keys():
                df = store[key]
                if units == 'metric':
                    if key == '/Unit_Parameters':
                        df['intake_vel'] = df.intake_vel * 0.3048
                        df['D'] = df.D * 0.3048
                        df['H'] = df.H * 0.3048
                        df['Qopt'] = df.Qopt * 0.0283168
                        df['Qcap'] = df.Qcap * 0.0283168
                        df['B'] = df.B * 0.3048
                        df['D1'] = df.D1 * 0.3048
                        df['D2'] = df.D2 * 0.3048
                    elif key == '/Facilities':
                        df['Bypass_Flow'] = df.Bypass_Flow * 0.0283168
                        df['Env_Flow'] = df.Env_Flow * 0.0283168
                        df['Min_Op_Flow'] = df.Min_Op_Flow * 0.0283168
                report_sections.append(enforce_horizontal(df, title))
            else:
                report_sections.append(f"<p>No {title} data available.</p>")

        # --- HYDROGRAPH SECTION: Time Series + Recurrence Histogram ---
        report_sections.append("<h2>Hydrograph Plots</h2>")
        if "/Hydrograph" in store.keys():
            hydrograph_df = store["/Hydrograph"]
            if 'datetimeUTC' in hydrograph_df.columns:
                hydrograph_df['datetimeUTC'] = pd.to_datetime(hydrograph_df['datetimeUTC'])
                if units == 'metric':
                    hydrograph_df['DAvgFlow_prorate'] = hydrograph_df.DAvgFlow_prorate * 0.0283168

            def create_hydro_timeseries(df):
                plt.rcParams.update({'font.size': 8})
                fig = plt.figure(figsize=(6, 4))
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
                fig = plt.figure(figsize=(6, 4))
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
        add_section("Beta Distributions", "/Beta_Distributions", units)

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
                if 'logger' in globals():
                    logger.debug('no daily df iteration column')
        else:
            iteration_sums = None
            if 'logger' in globals():
                logger.debug('no daily df')

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
            panel_html = "<h2>Seasonal Summary (by Iteration)</h2>"
            for metric in ["entrainment", "mortality"]:
                if metric == 'entrainment':
                    metric_key = 'entrained'
                else:
                    metric_key = 'mortality'

                if iteration_sums is not None and f'num_{metric_key}' in iteration_sums.columns:
                    hist_b64 = create_iteration_hist(iteration_sums, f'num_{metric_key}',
                                                     f"Total {metric.title()} by Iteration")
                else:
                    hist_b64 = ""

                mean_val = row.get(f"mean_yearly_{metric}", "N/A")
                lcl_val  = row.get(f"lcl_yearly_{metric}", "N/A")
                ucl_val  = row.get(f"ucl_yearly_{metric}", "N/A")
                like10   = row.get(f"1_in_10_day_{metric}", "N/A")
                like100  = row.get(f"1_in_100_day_{metric}", "N/A")
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
            if 'logger' in globals():
                logger.debug('something went wrong, yearly df empty')

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

            entr_img = create_daily_hist(daily_df, 'num_entrained', 'Daily Entrainment Distribution') \
                       if 'num_entrained' in daily_df.columns else None
            mort_img = create_daily_hist(daily_df, 'num_mortality', 'Daily Mortality Distribution') \
                       if 'num_mortality' in daily_df.columns else None

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

        # ---- finalize HTML ----
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
                h1 {{ color: #0056b3; }}
                h2 {{
                    color: #0056b3;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 4px;
                    margin-top: 2rem;
                }}
                h3 {{ color: #0056b3; margin-top: 1.5rem; }}
                p {{ line-height: 1.6; }}
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
                .table-wrap {{ overflow-x: auto; }}
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
                .download-link:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="container">
                {final_html}
                <a href="/download_report_zip" class="download-link">Download Report</a>
            </div>
        </body>
        </html>
        """
        return full_report

    finally:
        # Close the HDFStore no matter what happened above
        try:
            if store is not None:
                store.close()
        except Exception:
            pass



# def generate_report(sim):
#     """
#     Generate the comprehensive HTML report for the simulation.
#     This version ensures that all plot fonts render at least size 8,
#     rounds numeric values in the Beta Distributions table to two decimals,
#     and plots the daily histograms (entrainment and mortality) side by side.
#     """
#     import os
#     import pandas as pd
#     import io, base64
#     from datetime import datetime
#     import matplotlib.pyplot as plt

#     # Set global minimum font size for all plots
#     plt.rcParams.update({'font.size': 8})

#     hdf_path = os.path.join(sim.proj_dir, f"{sim.output_name}.h5")
#     if not os.path.exists(hdf_path):
#         return "<p>Error: HDF file not found. Please run the simulation first.</p>"
#     logger.debug('hdf file exists %s',hdf_path)
#     store = pd.HDFStore(hdf_path, mode='r')

#     report_sections = [
#         "<div style='margin: 10px;'>"
#         "  <button onclick=\"window.location.href='/'\" style='padding:10px;'>Home and Logout</button>"
#         "</div>",
#         f"<h1>Simulation Report for Project: {sim.project_name if hasattr(sim, 'project_name') else 'N/A'}</h1>",
#         f"<p><strong>Project Notes:</strong> {sim.project_notes if hasattr(sim, 'project_notes') else 'N/A'}</p>",
#         f"<p><strong>Model Setup:</strong> {sim.model_setup if hasattr(sim, 'model_setup') else 'N/A'}</p>",
#         f"<p><strong>Units:</strong> {sim.units_session if hasattr(sim, 'units_session') else 'N/A'}</p>",
#         f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
#         f"<p>HDF keys found: {store.keys()}</p>"
#     ]

#     units = sim.output_units
#     # Helper: Wrap DataFrame HTML in a scrollable container.
#     def enforce_horizontal(df, name=""):
#         if df is None or df.empty:
#             return f"<p>No {name} data available.</p>"
#         shape_info = f"<p>{name} data shape: {df.shape}</p>"
#         if df.shape[0] > 1 and df.shape[0] > df.shape[1]:
#             df = df.T
#             shape_info += f"<p>Transposed to shape: {df.shape}</p>"
#         # For Beta Distributions, round numeric columns to 2 decimals.
#         if name.lower() == "beta distributions":
#             df = df.copy()
#             for col in df.select_dtypes(include=["number"]).columns:
#                 df[col] = df[col].round(2)
#         table_html = df.to_html(index=False, border=1, classes="table")
#         return shape_info + f"<div style='overflow-x:auto;'>{table_html}</div>"

#     def add_section(title, key,  units):
#         report_sections.append(f"<h2>{title}</h2>")
#         if key in store.keys():
#             df = store[key]
#             if units == 'metric':
#                 if key == '/Unit_Parameters':
#                     df['intake_vel'] = df.intake_vel * 0.3048
#                     df['D'] = df.D * 0.3048
#                     df['H'] = df.H * 0.3048
#                     df['Qopt'] = df.Qopt * 0.0283168
#                     df['Qcap'] = df.Qcap * 0.0283168
#                     df['B'] = df.B * 0.3048
#                     df['D1'] = df.D1 * 0.3048
#                     df['D2'] = df.D2 * 0.3048
#                 elif key == '/Facilities':
#                     df['Bypass_Flow'] = df.Bypass_Flow * 0.0283168
#                     df['Env_Flow'] = df.Env_Flow * 0.0283168
#                     df['Min_Op_Flow'] = df.Min_Op_Flow * 0.0283168
                    
#             report_sections.append(enforce_horizontal(df, title))
#         else:
#             report_sections.append(f"<p>No {title} data available.</p>")

#     # Basic sections
#     # add_section("Nodes", "/Nodes", units)
#     # add_section("Edges", "/Edges", units)
#     # add_section("Unit Parameters", "/Unit_Parameters", units)
#     # add_section("Facilities", "/Facilities", units)
#     # add_section("Flow Scenarios", "/Flow Scenarios", units)
#     # add_section("Operating Scenarios", "/Operating Scenarios", units)
#     # add_section("Population", "/Population", units)
    
#     #logger.debug('finished basic data sections of report')
    
#     # --- HYDROGRAPH SECTION: Time Series + Recurrence Histogram ---
#     report_sections.append("<h2>Hydrograph Plots</h2>")
#     if "/Hydrograph" in store.keys():
#         hydrograph_df = store["/Hydrograph"]
#         if 'datetimeUTC' in hydrograph_df.columns:
#             hydrograph_df['datetimeUTC'] = pd.to_datetime(hydrograph_df['datetimeUTC'])
#             if units == 'metric':
#                 hydrograph_df['DAvgFlow_prorate'] = hydrograph_df.DAvgFlow_prorate * 0.0283168

#         def create_hydro_timeseries(df):
#             plt.rcParams.update({'font.size': 8})
#             fig = plt.figure(figsize=(6,4))
#             plt.plot(df['datetimeUTC'], df['DAvgFlow_prorate'], marker='.', linestyle='-')
#             plt.xlabel("Date")
#             plt.ylabel("Discharge")
#             plt.title("Hydrograph Time Series")
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             buf = io.BytesIO()
#             plt.savefig(buf, format='png', bbox_inches='tight')
#             plt.close(fig)
#             buf.seek(0)
#             return base64.b64encode(buf.getvalue()).decode('utf-8')

#         def create_hydro_hist(df):
#             plt.rcParams.update({'font.size': 8})
#             fig = plt.figure(figsize=(6,4))
#             plt.hist(df["DAvgFlow_prorate"].dropna(), bins=10, edgecolor='black')
#             plt.xlabel("Discharge")
#             plt.ylabel("Frequency")
#             plt.title("Recurrence Histogram")
#             plt.tight_layout()
#             buf = io.BytesIO()
#             plt.savefig(buf, format='png', bbox_inches='tight')
#             plt.close(fig)
#             buf.seek(0)
#             return base64.b64encode(buf.getvalue()).decode('utf-8')

#         ts_b64 = create_hydro_timeseries(hydrograph_df)
#         hist_b64 = create_hydro_hist(hydrograph_df)
#         report_sections.append(f"""
#         <div style="display:flex; gap:20px; justify-content:center; flex-wrap:wrap;">
#             <div style="flex:1; min-width:300px; text-align:center;">
#                 <h3>Time Series</h3>
#                 <img src="data:image/png;base64,{ts_b64}" style="max-width:100%; height:auto;" />
#             </div>
#             <div style="flex:1; min-width:300px; text-align:center;">
#                 <h3>Recurrence Histogram</h3>
#                 <img src="data:image/png;base64,{hist_b64}" style="max-width:100%; height:auto;" />
#             </div>
#         </div>
#         """)
#     else:
#         report_sections.append("<p>No hydrograph data available.</p>")
#     #logger.debug('finished hydrograph')
    
#     # --- BETA DISTRIBUTIONS ---
#     add_section("Beta Distributions", "/Beta_Distributions", units)

#     # --- YEARLY SUMMARY PANEL (Iteration-based) ---
#     yearly_df = store["/Yearly_Summary"] if "/Yearly_Summary" in store.keys() else None
#     daily_df = store["/Daily"] if "/Daily" in store.keys() else None

#     if daily_df is not None and not daily_df.empty:
#         if 'num_survived' in daily_df.columns and 'pop_size' in daily_df.columns:
#             daily_df['num_mortality'] = daily_df['pop_size'] - daily_df['num_survived']
#         if 'iteration' in daily_df.columns:
#             iteration_sums = daily_df.groupby('iteration').agg({
#                 'num_entrained': 'sum',
#                 'num_mortality': 'sum'
#             }).reset_index()
#         else:
#             iteration_sums = None
#             logger.debug('no daily df')
#     else:
#         iteration_sums = None
#         logger.debug('no daily df')

#     def create_iteration_hist(df, metric, title):
#         plt.rcParams.update({'font.size': 8})
#         fig = plt.figure()
#         plt.hist(df[metric].dropna(), bins=10, edgecolor='black')
#         plt.xlabel(metric.replace('_', ' ').title())
#         plt.ylabel("Frequency")
#         plt.title(title)
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png', bbox_inches='tight')
#         plt.close(fig)
#         buf.seek(0)
#         return base64.b64encode(buf.getvalue()).decode('utf-8')

#     def render_yearly_panel(yearly_df, iteration_sums):
#         if yearly_df is None or yearly_df.empty:
#             return "<p>No yearly summary data available.</p>"
#         row = yearly_df.iloc[0]  # Only one row expected
#         panel_html = "<h2>Seasonal Summary (by Iteration)</h2>"
#         for metric in ["entrainment", "mortality"]:
#             if metric == 'entrainment':
#                 metric = 'entrained'
#             if iteration_sums is not None and f'num_{metric}' in iteration_sums.columns:
#                 hist_b64 = create_iteration_hist(iteration_sums, f'num_{metric}', f"Total {metric.title()} by Iteration")
#             else:
#                 hist_b64 = ""
#             # Use expected column keys; adjust if needed.
#             if metric == 'entrained':
#                 metric = 'entrainment'

#             mean_val = row.get(f"mean_yearly_{metric}", "N/A")
#             lcl_val = row.get(f"lcl_yearly_{metric}", "N/A")
#             ucl_val = row.get(f"ucl_yearly_{metric}", "N/A")
#             like10 = row.get(f"1_in_10_day_{metric}", "N/A")
#             like100 = row.get(f"1_in_100_day_{metric}", "N/A")
#             like1000 = row.get(f"1_in_1000_day_{metric}", "N/A")
#             panel_html += f"""
#             <div style="display:flex; flex-wrap:wrap; margin-bottom:20px; border:1px solid #ccc; padding:10px; border-radius:5px;">
#                 <div style="flex:1; min-width:300px; padding:10px; border-right:1px solid #ddd;">
#                     <h3>Histogram ({metric.title()})</h3>
#                     <div style="text-align:center;">
#                         {'<img src="data:image/png;base64,' + hist_b64 + '" style="max-width:100%; height:auto;" />' if hist_b64 else "<p>No histogram data</p>"}
#                     </div>
#                 </div>
#                 <div style="flex:1; min-width:300px; padding:10px;">
#                     <h3>Statistics ({metric.title()})</h3>
#                     <p><strong>Average Annual:</strong> {mean_val}</p>
#                     <p><strong>95% CI:</strong> {lcl_val} - {ucl_val}</p>
#                     <p><strong>1 in 10 day event:</strong> {like10}</p>
#                     <p><strong>1 in 100 day event:</strong> {like100}</p>
#                     <p><strong>1 in 1000 day event:</strong> {like1000}</p>
#                 </div>
#             </div>
#             """
#         return panel_html
    


#     #logger.debug('finished creating yearly panel')
#     if yearly_df is not None and not yearly_df.empty:
#         panel_html = render_yearly_panel(yearly_df, iteration_sums)
#         report_sections.append(panel_html)
#     else:
#         report_sections.append("<p>No yearly summary data available.</p>")
#         logger.debug('something went wrong, yearly df empty')

#     # --- DAILY HISTOGRAMS SIDE BY SIDE ---
#     report_sections.append("<h2>Daily Histograms</h2>")
#     if daily_df is not None and not daily_df.empty:
#         # Ensure daily mortality exists
#         if 'num_mortality' not in daily_df.columns and 'num_survived' in daily_df.columns and 'pop_size' in daily_df.columns:
#             daily_df['num_mortality'] = daily_df['pop_size'] - daily_df['num_survived']

#         def create_daily_hist(data, col, title):
#             plt.rcParams.update({'font.size': 8})
#             fig = plt.figure()
#             plt.hist(data[col].dropna(), bins=20, edgecolor='black')
#             plt.xlabel(col.replace('_', ' ').title())
#             plt.ylabel("Frequency")
#             plt.title(title)
#             buf = io.BytesIO()
#             plt.savefig(buf, format='png', bbox_inches='tight')
#             plt.close(fig)
#             buf.seek(0)
#             return base64.b64encode(buf.getvalue()).decode('utf-8')

#         entr_img = None
#         mort_img = None
#         if 'num_entrained' in daily_df.columns:
#             entr_img = create_daily_hist(daily_df, 'num_entrained', 'Daily Entrainment Distribution')
#         if 'num_mortality' in daily_df.columns:
#             mort_img = create_daily_hist(daily_df, 'num_mortality', 'Daily Mortality Distribution')

#         report_sections.append("""
#         <div style="display:flex; gap:20px; justify-content:center; flex-wrap:wrap;">
#         """)
#         if entr_img:
#             report_sections.append(f"""
#             <div style="flex:1; min-width:300px; text-align:center;">
#                 <h3>Daily Entrainment</h3>
#                 <img src="data:image/png;base64,{entr_img}" style="max-width:100%; height:auto;" />
#             </div>
#             """)
#         else:
#             report_sections.append("""
#             <div style="flex:1; min-width:300px; text-align:center;">
#                 <h3>Daily Entrainment</h3>
#                 <p>No 'num_entrained' data available.</p>
#             </div>
#             """)
#         if mort_img:
#             report_sections.append(f"""
#             <div style="flex:1; min-width:300px; text-align:center;">
#                 <h3>Daily Mortality</h3>
#                 <img src="data:image/png;base64,{mort_img}" style="max-width:100%; height:auto;" />
#             </div>
#             """)
#         report_sections.append("</div>")
#     else:
#         report_sections.append("<p>No daily data available.</p>")

#     store.close()
#     #logger.debug('finished daily pannel')
#     final_html = "\n".join(report_sections)
#     full_report = f"""
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <title>Simulation Report</title>
#         <style>
#             body {{
#                 font-family: Arial, sans-serif;
#                 background: #f7f7f7;
#                 margin: 20px;
#                 color: #333;
#             }}
#             .container {{
#                 max-width: 1800px;
#                 margin: auto;
#                 background: #fff;
#                 padding: 20px;
#                 border-radius: 8px;
#                 box-shadow: 0 0 10px rgba(0,0,0,0.1);
#             }}
#             h1 {{
#                 color: #0056b3;
#             }}
#             h2 {{
#                 color: #0056b3;
#                 border-bottom: 1px solid #ddd;
#                 padding-bottom: 4px;
#                 margin-top: 2rem;
#             }}
#             h3 {{
#                 color: #0056b3;
#                 margin-top: 1.5rem;
#             }}
#             p {{
#                 line-height: 1.6;
#             }}
#             table {{
#                 width: 100%;
#                 border-collapse: collapse;
#                 margin: 1rem 0;
#             }}
#             th, td {{
#                 padding: 8px;
#                 border: 1px solid #ccc;
#                 text-align: left;
#             }}
#             .table-wrap {{
#                 overflow-x: auto;
#             }}
#             pre {{
#                 background: #f4f4f4;
#                 padding: 10px;
#                 border-radius: 5px;
#             }}
#             .download-link {{
#                 display: inline-block;
#                 margin-top: 20px;
#                 background: #007BFF;
#                 color: white;
#                 padding: 10px 15px;
#                 text-decoration: none;
#                 border-radius: 4px;
#             }}
#             .download-link:hover {{
#                 background: #0056b3;
#             }}
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             {final_html}
#             <a href="/download_report_zip" class="download-link">Download Report</a>
#         </div>
#     </body>
#     </html>
#     """
#     #logger.debug('report formatted')
#     return full_report

@app.route('/download_report')
def download_report():
    proj_dir = session.get('proj_dir')
    if not proj_dir:
        return "<p>Error: No project directory in session.</p>", 400

    marker = os.path.join(proj_dir, "report_path.txt")
    report_path = open(marker, 'r', encoding='utf-8').read().strip() if os.path.exists(marker) \
                  else os.path.join(proj_dir, "simulation_report.html")

    if not os.path.exists(report_path):
        return f"<p>Error: Report file not found at {report_path}.</p>", 404

    return send_file(report_path, as_attachment=True,
                     download_name="simulation_report.html", mimetype="text/html")

@app.get('/download_zip', endpoint='download_zip')
def download_zip_alias():
    # simple handoff to the real endpoint
    return redirect(url_for('download_report_zip'))

# Quick route map to spot mismatches (remove in prod)
@app.get('/_debug_routes')
def _debug_routes():
    lines = []
    for rule in app.url_map.iter_rules():
        methods = ",".join(sorted(m for m in rule.methods if m in ("GET","POST","PUT","DELETE","PATCH")))
        lines.append(f"{rule.endpoint:30s}  {methods:10s}  {rule}")
    return "<pre>" + "\n".join(sorted(lines)) + "</pre>"

@app.route('/download_report_zip')
def download_report_zip():
    proj_dir = session.get('proj_dir')
    if not proj_dir:
        return "<h1>Session missing proj_dir</h1>", 500
    if not os.path.exists(proj_dir):
        return f"<h1>Project directory not found: {proj_dir}</h1>", 404

    zip_path = os.path.join(proj_dir, "simulation_report.zip")
    # rebuild fresh zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(proj_dir):
            for file in files:
                fp = os.path.join(root, file)
                if os.path.abspath(fp) == os.path.abspath(zip_path):
                    continue
                zipf.write(fp, arcname=os.path.relpath(fp, start=proj_dir))

    from flask import after_this_request
    @after_this_request
    def _cleanup(resp):
        try:
            os.remove(zip_path)
        except Exception:
            pass
        return resp

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(zip_path, as_attachment=True, download_name=f"simulation_report_{timestamp}.zip")


# Un Comment to Test Locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded = True, use_reloader=False)
