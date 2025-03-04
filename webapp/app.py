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
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, after_this_request, send_from_directory, session, Response
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import zipfile
import time

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


# ----------------- Password Protection -----------------
@app.before_request
def require_login():

    if not session.get('logged_in') and request.endpoint not in ['login', 'static','health']:
        return redirect(url_for('login'))

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
    session.clear()
    # Optional: clear folders if desired
    clear_folder(UPLOAD_FOLDER)
    clear_folder(SIM_PROJECT_FOLDER)
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
    return "OK", 200

def run_simulation(ws, wks, output_name):
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
        
def run_simulation_in_background(ws, wks, output_name):
    old_stdout = sys.stdout
    sys.stdout = QueueStream(LOG_QUEUE)
    try:
        simulation_instance = stryke.simulation(ws, wks, output_name=output_name)
        simulation_instance.run()
        simulation_instance.summary()

        # Debugging: Check if file exists
        output_file = os.path.join(SIM_PROJECT_FOLDER, f"{output_name}.h5")
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
        
        up_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(up_file_path)
        flash(f'File successfully uploaded: {file.filename}')
        
        simulation_file_path = os.path.join(SIM_PROJECT_FOLDER, file.filename)
        shutil.copy(up_file_path, simulation_file_path)
        
        ws = SIM_PROJECT_FOLDER
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

    # 1) Remove any old zip files in SIM_PROJECT_FOLDER
    for fname in os.listdir(SIM_PROJECT_FOLDER):
        if fname.endswith(".zip"):
            old_path = os.path.join(SIM_PROJECT_FOLDER, fname)
            try:
                os.remove(old_path)
                print(f"Removed old ZIP: {old_path}")
            except Exception as e:
                print(f"Error removing old ZIP {old_path}: {e}")

    # 2) Now create a brand-new ZIP file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"simulation_results_{timestamp}.zip"
    zip_filepath = os.path.join(SIM_PROJECT_FOLDER, zip_filename)
    print(f"Creating ZIP file: {zip_filepath}")

    try:
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_name in os.listdir(SIM_PROJECT_FOLDER):
                # Skip HDF, H5, and .zip
                if file_name.endswith((".hdf", ".h5", ".zip")):
                    continue

                file_path = os.path.join(SIM_PROJECT_FOLDER, file_name)
                if os.path.isfile(file_path):
                    # Try to add
                    try:
                        zipf.write(file_path, arcname=file_name)
                        print(f"Added to ZIP: {file_name}")
                    except Exception as e:
                        print(f"Skipping file {file_name} => {e}")

        print(f"ZIP file successfully created: {zip_filepath}")
    except Exception as e:
        print(f"Error creating ZIP file: {e}")
        flash("Failed to create ZIP file.")
        return redirect(url_for('fit_distributions'))

    # 3) Return the new ZIP
    if os.path.exists(zip_filepath):
        return send_file(zip_filepath, as_attachment=True)
    else:
        flash("ZIP file not found.")
        return redirect(url_for('fit_distributions'))



# @app.route('/download_zip')
# def download_zip():
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     zip_filename = f"simulation_results_{timestamp}.zip"
#     zip_filepath = os.path.join(SIM_PROJECT_FOLDER, zip_filename)

#     print(f"Creating ZIP file: {zip_filepath}")

#     try:
#         with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
#             # Loop over each file in SIM_PROJECT_FOLDER
#             for file_name in os.listdir(SIM_PROJECT_FOLDER):
#                 file_path = os.path.join(SIM_PROJECT_FOLDER, file_name)

#                 # Skip HDF/H5 files
#                 if file_name.endswith((".hdf", ".h5", ".zip")):
#                     continue

#                 # Retry up to 5 times if file doesn't exist or is locked
#                 max_tries = 5
#                 wait_time = 0.5  # half a second
#                 for attempt in range(max_tries):
#                     if not os.path.exists(file_path):
#                         print(f"[Attempt {attempt+1}] {file_path} doesn't exist. "
#                               f"Waiting {wait_time}s then retrying.")
#                         time.sleep(wait_time)
#                         continue
#                     else:
#                         # Check if we can open it in read-binary mode
#                         try:
#                             with open(file_path, "rb") as test_f:
#                                 pass
#                             # If it opened successfully, break from retry loop
#                             break
#                         except Exception as e:
#                             print(f"[Attempt {attempt+1}] File locked: {file_path}, error: {e}. "
#                                   f"Waiting {wait_time}s then retrying.")
#                             time.sleep(wait_time)
#                 else:
#                     # If we exhausted all tries, skip this file
#                     print(f"Skipping file after {max_tries} attempts: {file_path}")
#                     continue

#                 # Finally, try adding to the ZIP
#                 try:
#                     for fname in os.listdir(SIM_PROJECT_FOLDER):
#                         if fname.endswith(".zip"):
#                             os.remove(os.path.join(SIM_PROJECT_FOLDER, fname))

#                     zipf.write(file_path, arcname=file_name)
#                     print(f"Added to ZIP: {file_name}")
#                 except Exception as e:
#                     print(f"Skipping file {file_name} => {e}")

#         print(f"ZIP file successfully created: {zip_filepath}")

#     except Exception as e:
#         print(f"Error creating ZIP file: {e}")
#         flash("Failed to create ZIP file.")
#         return redirect(url_for('fit_distributions'))  # or wherever you want to redirect

#     # Serve the ZIP file if it exists
#     if os.path.exists(zip_filepath):
#         # Optional: short wait so OS fully recognizes the new file
#         time.sleep(0.5)

#         response = send_file(zip_filepath, as_attachment=True)

#         @after_this_request
#         def cleanup(response):
#             try:
#                 print("Cleaning up after sending ZIP...")
#                 # Remove everything except this newly created ZIP
#                 for f in os.listdir(SIM_PROJECT_FOLDER):
#                     f_path = os.path.join(SIM_PROJECT_FOLDER, f)
#                     if f_path != zip_filepath:
#                         if os.path.isfile(f_path):
#                             os.remove(f_path)
#                 print("Cleanup complete.")
#             except Exception as exc:
#                 print(f"Error during cleanup: {exc}")
#             return response

#         return response
#     else:
#         flash("ZIP file not found.")
#         return redirect(url_for('fit_distributions'))  # or whichever page



# # Added download route for output file
# @app.route('/download/<filename>')
# def download_output(filename):
#     output_path = os.path.join(SIM_PROJECT_FOLDER, filename)
#     print(f"Looking for file: {output_path}")  # Debugging output
#     if os.path.exists(output_path):
#         print(f"File found: {output_path}")  # Debugging output
#         return send_file(output_path, as_attachment=True)
#     else:
#         print("Error: File not found!")  # Debugging output
#         flash("Output file not found.")
#         return redirect(url_for('upload_simulation'))

# @app.route('/download_distribution_zip')
# def download_distribution_zip():
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     zip_filename = f"distribution_fitting_results_{timestamp}.zip"
#     zip_filepath = os.path.join(SIM_PROJECT_FOLDER, zip_filename)

#     print(f"Creating Distribution ZIP file: {zip_filepath}")  # Debugging

#     try:
#         # Ensure file isn't still being written
#         if not os.path.exists(zip_filepath):
#             print(f"Error: ZIP file {zip_filepath} not found.")
#             flash("Distribution ZIP file not found.")
#             return redirect(url_for('fit_distributions'))

#         # Debugging: Check for locked files inside the ZIP
#         for file in os.listdir(SIM_PROJECT_FOLDER):
#             file_path = os.path.join(SIM_PROJECT_FOLDER, file)
#             if os.path.isfile(file_path):
#                 try:
#                     with open(file_path, "rb") as f:
#                         pass  # If this fails, the file is locked
#                 except Exception as e:
#                     print(f"Skipping locked file: {file_path} - Error: {e}")

#         # Small delay to ensure file write is completed
#         time.sleep(1)  

#         print(f"ZIP file {zip_filepath} exists. Preparing to send.")
#         response = send_file(zip_filepath, as_attachment=True)

#         @after_this_request
#         def cleanup(response):
#             """Delete all distribution fitting files after sending the ZIP."""
#             try:
#                 print("Cleaning up distribution fitting results folder...")
#                 for file in os.listdir(SIM_PROJECT_FOLDER):
#                     file_path = os.path.join(SIM_PROJECT_FOLDER, file)
#                     if file_path != zip_filepath:  # Keep latest ZIP, delete the rest
#                         if os.path.isfile(file_path):
#                             os.remove(file_path)
#                             print(f"Deleted: {file_path}")
#                         elif os.path.isdir(file_path):
#                             shutil.rmtree(file_path)
#                             print(f"Deleted directory: {file_path}")
#                 print("Cleanup complete.")
#             except Exception as e:
#                 print(f"Error during cleanup: {e}")
#             return response

#         print(f"ZIP file {zip_filepath} is being sent.")  # Debugging output
#         return response

#     except Exception as e:
#         print(f"Error serving ZIP file: {e}")
#         flash("Failed to serve ZIP file.")
#         return redirect(url_for('fit_distributions'))

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
    
            plt.clf()
            fish.plot()
            plot_filename = 'fitting_results.png'
            plot_path = os.path.join(SIM_PROJECT_FOLDER, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            
            plt.clf()
            plt.figure(figsize=(5, 3))
            plt.hist(fish.lengths.tolist(), bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel("Fish Length (cm)")
            plt.ylabel("Frequency")
            plt.title("Distribution of Fish Lengths")
        
            other_filename = 'fish_lengths.png'
            plot_path = os.path.join(SIM_PROJECT_FOLDER, other_filename)
            plt.savefig(plot_path)
            plt.close()
            
            summary_text = (
                "Distribution fitting complete for filters: "
                f"States: '{states}', Plant Capacity: '{plant_cap}', Family: '{family}', "
                f"Genus: '{genus}', Species: '{species}', Months: {month}, "
                f"HUC02: {huc02}, HUC04: {huc04}, HUC06: {huc06}, HUC08: {huc08}, "
                f"NIDID: '{nidid}', River: '{river}'."
            )
            
            try:
                report_text = fish.summary_output(SIM_PROJECT_FOLDER, dist='Log Normal')
            except Exception as e:
                report_text = f"Error generating detailed summary report: {e}"
            
            log_text = report_text
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_filename_full = os.path.join(SIM_PROJECT_FOLDER, "fitting_results_log.txt")
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
    file_path = os.path.join(SIM_PROJECT_FOLDER, filename)
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

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
