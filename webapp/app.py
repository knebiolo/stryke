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
        
def run_simulation_in_background(ws, wks, output_name):
    # Redirect stdout to our custom stream
    old_stdout = sys.stdout
    sys.stdout = QueueStream(LOG_QUEUE)
    try:
        simulation_instance = stryke.simulation(ws, wks, output_name=output_name)
        simulation_instance.run()
        simulation_instance.summary()
    except Exception as e:
        print("Error during simulation:", e)
    finally:
        # Restore stdout and signal completion
        sys.stdout = old_stdout
        LOG_QUEUE.put("[Simulation Complete]")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_simulation():
    simulation_results = None
    output_filename = None
    # We no longer capture all output synchronously; we stream live via SSE.
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
        
        # Set up simulation parameters.
        ws = SIM_PROJECT_FOLDER     # Simulation project folder.
        wks = file.filename         # Uploaded file name.
        output_name = 'Simulation_Output'  # Fixed output name (will overwrite previous output).
        
        try:
            # Start the simulation in a background thread.
            simulation_thread = threading.Thread(
                target=run_simulation_in_background,
                args=(ws, wks, output_name)
            )
            simulation_thread.start()
            flash("Simulation started. Live log will appear below.")
            simulation_results = "Simulation is running..."
            output_filename = f"{output_name}.h5"  # Expected output file (when simulation completes)
        except Exception as e:
            flash(f"Error starting simulation: {e}")
            return render_template('upload_simulation.html')
    
    # Render the page with (or without) simulation results.
    return render_template('upload_simulation.html',
                           simulation_results=simulation_results,
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
        # Continue streaming until the sentinel message "[Simulation Complete]" is received.
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
    # Initialize variables for context.
    summary_text = ""
    log_text = ""
    plot_filename = ""
    
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
        
        # Build the filter dictionary only with non-empty values.
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
        
        # Redirect stdout BEFORE initializing fish so that __init__ output is captured.
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        try:
            # Initialize the EPRI query. This will capture print output from __init__.
            fish = stryke.epri(**filter_args)
            # Run the fitting functions.
            fish.ParetoFit()
            fish.LogNormalFit()
            fish.WeibullMinFit()
            fish.LengthSummary()
        except Exception as e:
            sys.stdout = old_stdout  # Restore stdout before handling the error.
            flash(f"Error during fitting: {e}")
            return render_template('fit_distributions.html', summary=summary_text, log_text=log_text, plot_filename=plot_filename)
        finally:
            captured_output = mystdout.getvalue()
            sys.stdout = old_stdout

        # Generate and save the plot (overwrite any previous file).
        plt.clf()
        fish.plot()  # Generate the matplotlib figure.
        plot_filename = 'fitting_results.png'  # Fixed filename.
        plot_path = os.path.join(SIM_PROJECT_FOLDER, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        
        # Create a summary text that includes the query.
        summary_text = (
            "Distribution fitting complete for filters: "
            f"States: '{states}', Plant Capacity: '{plant_cap}', Family: '{family}', "
            f"Genus: '{genus}', Species: '{species}', Months: {month}, "
            f"HUC02: {huc02}, HUC04: {huc04}, HUC06: {huc06}, HUC08: {huc08}, "
            f"NIDID: '{nidid}', River: '{river}'."
        )
        
        # Generate the new detailed report.
        try:
            report_text = fish.summary_output(SIM_PROJECT_FOLDER, dist='Log Normal')
        except Exception as e:
            report_text = f"Error generating detailed summary report: {e}"
        
        # Use the new report as our log_text.
        log_text = report_text
        
        # Write a log entry (optional).
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_filename_full = os.path.join(SIM_PROJECT_FOLDER, "fitting_results_log.txt")
        with open(log_filename_full, "a") as log_file:
            log_file.write(f"{timestamp} - Query: {summary_text}\n")
            log_file.write(f"{timestamp} - Report: {report_text}\n")
            log_file.write("--------------------------------------------------\n")
        
        return render_template('fit_distributions.html', summary=summary_text, log_text=log_text, plot_filename=plot_filename)
        
    
    # For GET requests, simply render the form.
    return render_template('fit_distributions.html', summary=summary_text, log_text=log_text, plot_filename=plot_filename)

@app.route('/plot/<filename>')
def serve_plot(filename):
    file_path = os.path.join(SIM_PROJECT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return "Plot not found", 404

    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    

