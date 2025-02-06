# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:48:03 2025

@author: Kevin.Nebiolo
"""
# webapp/app.py
import os
import sys
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
# Ensure your repository is in the Python path so you can import Stryke
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke")  # Adjust the path if needed
from Stryke import stryke
import threading
import queue
from flask import Response
import datetime 
import matplotlib.pyplot as plt
import io

# Create a global log queue
LOG_QUEUE = queue.Queue()

# A custom stream object that writes messages to the queue
class QueueStream:
    def __init__(self, q):
        self.q = q
    def write(self, message):
        # Only push non-empty messages (you can adjust filtering as needed)
        if message.strip():
            self.q.put(message)
    def flush(self):
        pass

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Define directories for uploads and simulation projects
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
SIM_PROJECT_FOLDER = os.path.join(os.getcwd(), 'simulation_project')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIM_PROJECT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def run_simulation(ws, wks, output_name):
    # Save the original stdout
    old_stdout = sys.stdout
    # Redirect stdout to our queue stream
    sys.stdout = QueueStream(LOG_QUEUE)
    try:
        simulation_instance = stryke.simulation(ws, wks, output_name=output_name)
        simulation_instance.run()
        simulation_instance.summary()
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        # Restore stdout
        sys.stdout = old_stdout
        # Put a sentinel value to indicate the simulation is done
        LOG_QUEUE.put(None)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'excel_file' not in request.files:
#             flash('No file part in the request')
#             return redirect(request.url)
        
#         file = request.files['excel_file']
#         if file.filename == '':
#             flash('No file selected')
#             return redirect(request.url)
        
#         # Save the uploaded file in UPLOAD_FOLDER
#         up_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(up_file_path)
#         flash(f'File successfully uploaded: {file.filename}')
        
#         # Copy the uploaded file to SIM_PROJECT_FOLDER
#         simulation_file_path = os.path.join(SIM_PROJECT_FOLDER, file.filename)
#         shutil.copy(up_file_path, simulation_file_path)
        
#         # Use the uploaded file for the simulation
#         ws = SIM_PROJECT_FOLDER  # Your simulation project folder
#         wks = file.filename      # The uploaded file name
#         output_name = 'Cabot_Beta_Test'
        
#         try:
#             # Start the simulation in a background thread
#             simulation_thread = threading.Thread(target=run_simulation, args=(ws, wks, output_name))
#             simulation_thread.start()
#             flash("Simulation started. You will see live output below.")
#         except Exception as e:
#             flash(f"Error starting simulation: {e}")
#             return redirect(request.url)
        
#         # Redirect to the results page (which will include live log updates)
#         return redirect(url_for('results', output_file=f"{output_name}.h5"))
    
#     return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    simulation_results = None
    simulation_log = None
    output_filename = None

    if request.method == 'POST':
        if 'excel_file' not in request.files:
            flash('No file part in the request')
            return render_template('upload_simulation.html')
        
        file = request.files['excel_file']
        if file.filename == '':
            flash('No file selected')
            return render_template('upload_simulation.html')
        
        # Save the uploaded file in UPLOAD_FOLDER.
        up_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(up_file_path)
        flash(f'File successfully uploaded: {file.filename}')
        
        # Copy the file to SIM_PROJECT_FOLDER (where the simulation expects it).
        simulation_file_path = os.path.join(SIM_PROJECT_FOLDER, file.filename)
        shutil.copy(up_file_path, simulation_file_path)
        
        # Set up the simulation using the uploaded file.
        ws = SIM_PROJECT_FOLDER     # Simulation project folder.
        wks = file.filename         # Uploaded file name.
        output_name = 'Simulation_Output'  # Use a fixed output name to overwrite previous results.
        
        try:
            simulation_instance = stryke.simulation(ws, wks, output_name=output_name)
            
            # (Optional) Capture printed output from simulation run.
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            try:
                simulation_instance.run()
                simulation_instance.summary()
            finally:
                simulation_log = mystdout.getvalue()
                sys.stdout = old_stdout
            
            flash("Simulation completed successfully!")
            output_filename = f"{output_name}.h5"  # The simulation output file.
            
            simulation_results = "Simulation ran successfully. You can download the output file below."
            
            # (Optional) Write a log entry to a simulation log file.
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_filename = os.path.join(SIM_PROJECT_FOLDER, "simulation_results_log.txt")
            with open(log_filename, "a") as log_file:
                log_file.write(f"{timestamp} - File: {wks}\n")
                log_file.write(f"{timestamp} - Result: Simulation completed successfully.\n")
                log_file.write("--------------------------------------------------\n")
            
        except Exception as e:
            flash(f"Error during simulation: {e}")
            return render_template('upload.html')
    
    return render_template('upload.html',
                           simulation_results=simulation_results,
                           simulation_log=simulation_log,
                           output_filename=output_filename)

@app.route('/download/<filename>')
def download_output(filename):
    output_path = os.path.join(SIM_PROJECT_FOLDER, filename)
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    else:
        flash("Output file not found.")
        return redirect(url_for('upload_simulation'))

@app.route('/stream')
def stream():
    def event_stream():
        # Continue streaming until we encounter our sentinel value (None)
        while True:
            message = LOG_QUEUE.get()
            if message is None:
                yield 'data: [Simulation Complete]\n\n'
                break
            yield f'data: {message}\n\n'
    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/results')
def results():
    output_file = request.args.get('output_file')
    output_path = os.path.join(SIM_PROJECT_FOLDER, output_file)
    summary_text = "Simulation ran successfully. Please download the output file below."
    return render_template('results.html', output_file=output_file, summary=summary_text)

# @app.route('/download/<filename>')
# def download_output(filename):
#     output_path = os.path.join(SIM_PROJECT_FOLDER, filename)
#     if os.path.exists(output_path):
#         return send_file(output_path, as_attachment=True)
#     else:
#         flash("Output file not found.")
#         return redirect(url_for('index'))
    
@app.route('/fit', methods=['GET', 'POST'])
def fit_distributions():
    if request.method == 'POST':
        # Retrieve parameters from the form.
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
        
        # Helper function to parse comma-separated values.
        def parse_list(text):
            return [item.strip() for item in text.split(',') if item.strip()] if text else None
      
        try:
            month = [int(m) for m in parse_list(month_str)]
        except Exception:
            month = None
        try:
            huc02 = [int(x) for x in parse_list(huc02_str)]
        except Exception:
            huc02 = None
        try:
            huc04 = [int(x) for x in parse_list(huc04_str)]
        except Exception:
            huc04 = None
        try:
            huc06 = [int(x) for x in parse_list(huc06_str)]
        except Exception:
            huc06 = None
        try:
            huc08 = [int(x) for x in parse_list(huc08_str)]
        except Exception:
            huc08 = None
        
        filter_args = {
            "states": states,
            "plant_cap": plant_cap,
            "Family": family,
            "Genus": genus,
            "Species": species,
            "Month": month,
            "HUC02": huc02,
            "HUC04": huc04,
            "HUC06": huc06,
            "HUC08": huc08,
            "NIDID": nidid,
            "River": river
        }
        
        try:
            fish = stryke.epri(**filter_args)
        except Exception as e:
            flash(f"Error initializing EPRI query: {e}")
            return render_template('fit_form.html')
        
        # Capture any printed output from the fitting functions.
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        try:
            fish.ParetoFit()
            fish.LogNormalFit()
            fish.WeibullMinFit()
            fish.LengthSummary()
        finally:
            captured_output = mystdout.getvalue()
            sys.stdout = old_stdout  # Restore stdout
        
        # Overwrite the plot file with a fixed filename.
        plt.clf()
        fish.plot()  # Generate the matplotlib figure.
        plot_filename = 'fitting_results.png'
        plot_path = os.path.join(SIM_PROJECT_FOLDER, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        
        summary_text = (
            "Distribution fitting complete for filters: "
            f"States: '{states}', Plant Capacity: '{plant_cap}', Family: '{family}', "
            f"Genus: '{genus}', Species: '{species}', Months: {month}, "
            f"HUC02: {huc02}, HUC04: {huc04}, HUC06: {huc06}, HUC08: {huc08}, "
            f"NIDID: '{nidid}', River: '{river}'."
        )
        log_text = "Fitting functions executed successfully.\n\n" + captured_output
        
        # Write a log entry to a results log file.
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_filename_full = os.path.join(SIM_PROJECT_FOLDER, "fitting_results_log.txt")
        with open(log_filename_full, "a") as log_file:
            log_file.write(f"{timestamp} - Query: {summary_text}\n")
            log_file.write(f"{timestamp} - Result: {log_text}\n")
            log_file.write("--------------------------------------------------\n")
        
        # Render the same page with the results now included.
        return render_template('fit_form.html',
                               summary=summary_text,
                               log_text=log_text,
                               plot_filename=plot_filename)
    
    # For GET requests, just render the form.
    return render_template('fit_form.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    

