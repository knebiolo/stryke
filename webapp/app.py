# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import shutil
import threading
import queue
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Stryke")))

from Stryke import stryke
from Stryke.stryke import epri

LOG_QUEUE = queue.Queue()

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
app.config['PASSWORD'] = 'expensive5rudabega!@1'

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
SIM_PROJECT_FOLDER = os.path.join(os.getcwd(), 'simulation_project')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIM_PROJECT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------- Password Protection -----------------
@app.before_request
def require_login():
    if not session.get('logged_in') and request.endpoint not in ['login', 'static']:
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
    clear_folder(SIM_PROJECT_FOLDER)
    flash("Logged out successfully.")
    return redirect(url_for('login'))

# ----------------- Simulation Execution -----------------
def run_simulation_in_background(ws, wks, output_name):
    old_stdout = sys.stdout
    sys.stdout = QueueStream(LOG_QUEUE)
    try:
        print("LOG: Starting simulation...")
        simulation_instance = stryke.simulation(ws, wks, output_name=output_name)
        simulation_instance.run()
        simulation_instance.summary()
        print("LOG: Simulation completed.")
    except Exception as e:
        print(f"LOG: Error during simulation: {e}")
    finally:
        sys.stdout = old_stdout
        LOG_QUEUE.put("[Simulation Complete]")

@app.route('/upload', methods=['GET', 'POST'])
def upload_simulation():
    simulation_results = None
    output_filename = None

    if request.method == 'POST':
        if 'excel_file' not in request.files:
            flash('No file selected.')
            return render_template('upload_simulation.html')

        file = request.files['excel_file']
        if file.filename == '':
            flash('No file selected.')
            return render_template('upload_simulation.html')

        up_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(up_file_path)
        shutil.copy(up_file_path, os.path.join(SIM_PROJECT_FOLDER, file.filename))
        
        ws = SIM_PROJECT_FOLDER
        wks = file.filename
        output_name = 'Simulation_Output'

        try:
            print("LOG: Starting simulation thread...")
            simulation_thread = threading.Thread(
                target=run_simulation_in_background,
                args=(ws, wks, output_name)
            )
            simulation_thread.start()
            flash("Simulation started.")
            simulation_results = "Simulation is running..."
            output_filename = f"{output_name}.h5"
        except Exception as e:
            flash(f"Error starting simulation: {e}")
            return render_template('upload_simulation.html')

    return render_template('upload_simulation.html', simulation_results=simulation_results, output_filename=output_filename)

# ----------------- Distribution Fitting -----------------
@app.route('/fit', methods=['GET', 'POST'])
def fit_distributions():
    summary_text = ""
    log_text = ""
    plot_filename = ""
    other_filename = "fish_lengths.png"

    if request.method == 'POST':
        try:
            print("LOG: Received POST request for fitting.")
            old_stdout = sys.stdout
            mystdout = io.StringIO()
            sys.stdout = mystdout

            filter_args = {key: request.form.get(key, '').strip() for key in request.form.keys()}
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
                flash(f"Error during fitting: {e}")
                return render_template('fit_distributions.html', summary=summary_text, log_text=log_text)

            plt.clf()
            fish.plot()
            plot_path = os.path.join(SIM_PROJECT_FOLDER, "fitting_results.png")
            plt.savefig(plot_path)
            plt.close()

            plt.clf()
            plt.hist(fish.lengths.tolist(), bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel("Fish Length (cm)")
            plt.ylabel("Frequency")
            plt.title("Distribution of Fish Lengths")
            plt.savefig(os.path.join(SIM_PROJECT_FOLDER, other_filename))
            plt.close()

            return render_template('fit_distributions.html', summary="Fitting complete!", log_text=log_text, plot_filename="fitting_results.png", other_filename=other_filename)

        except Exception as e:
            flash(f"Error during fitting: {e}")
            return render_template('fit_distributions.html', summary=summary_text, log_text=log_text)

# ----------------- Serve Plots -----------------
@app.route('/plot/<filename>')
def serve_plot(filename):
    file_path = os.path.join(SIM_PROJECT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return "Plot not found", 404

# ----------------- Utility Functions -----------------
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path) if os.path.isfile(file_path) else shutil.rmtree(file_path)
                print(f"LOG: Deleted {file_path}")
            except Exception as e:
                print(f"LOG: Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
