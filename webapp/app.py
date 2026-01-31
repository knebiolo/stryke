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
from flask import jsonify, stream_with_context, current_app
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import zipfile
import time
import uuid
from collections import defaultdict
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
import secrets
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import tables
import filelock
from werkzeug.exceptions import HTTPException, NotFound, RequestEntityTooLarge
from werkzeug.utils import secure_filename

def _env_flag(name, default="0"):
    val = os.environ.get(name, default)
    return str(val).strip().lower() in ("1", "true", "yes", "on")

def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)

DIAGNOSTICS_ENABLED = _env_flag("STRYKE_WEBAPP_DIAGNOSTICS", "0")
LOG_LINE_MAX_CHARS = _env_int("STRYKE_LOG_LINE_MAX_CHARS", 2000)

# Manually tell pyproj where PROJ is installed
os.environ["PROJ_DIR"] = "/usr"
os.environ["PROJ_LIB"] = "/usr/share/proj"
os.environ["PYPROJ_GLOBAL_CONTEXT"] = "ON"

try:
    import pyproj
except ImportError as exc:
    raise ImportError(
        "pyproj is required but not installed. Install it in the environment before running the webapp."
    ) from exc
    
# Explicitly add the parent directory of Stryke to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Add the Stryke directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Stryke")))

# Import Stryke components
from Stryke import stryke
from Stryke.stryke import epri

# Use bounded queues to prevent memory issues if connection dies
# maxsize=1000 means we keep last 1000 messages, older ones dropped
RUN_QUEUES = defaultdict(lambda: queue.Queue(maxsize=1000))

def get_queue(run_id):
    return RUN_QUEUES[run_id]

def _read_csv_with_encoding_fallback(file_path, *args, **kwargs):
    """Read CSV with a small, explicit encoding fallback chain."""
    encoding = kwargs.pop("encoding", None)
    if encoding:
        return pd.read_csv(file_path, *args, encoding=encoding, **kwargs)

    encodings = ("utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "latin-1")
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, *args, encoding=enc, **kwargs)
            if enc != "utf-8":
                print(f"[WARN] read_csv_if_exists: used encoding={enc} for {file_path}", flush=True)
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    msg = f"Failed to decode CSV as UTF-8/UTF-16/Latin-1: {file_path}"
    if last_error is not None:
        raise ValueError(msg) from last_error
    raise ValueError(msg)

def _read_csv_if_exists_compat(file_path=None, *args, **kwargs):
    """
    Backward-compatible wrapper:
      - tolerates file_path=None / "" (returns None)
      - accepts optional numeric_cols kw and coerces those columns if present
    """
    numeric_cols = kwargs.pop("numeric_cols", None)
    index_col = kwargs.pop("index_col", None)

    if not file_path or (isinstance(file_path, str) and not file_path.strip()):
        return None
    if not isinstance(file_path, (str, bytes, os.PathLike)):
        raise TypeError(f"read_csv_if_exists(file_path=...) expected a path, got {type(file_path).__name__}")

    if not os.path.exists(file_path):
        # Match previous behavior: either return None or raise; returning None is kinder to UIs.
        # If you prefer hard-fail, change to: raise FileNotFoundError(...)
        return None

    df = _read_csv_with_encoding_fallback(file_path, *args, **kwargs)
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    if index_col:
        if index_col in df.columns:
            df.set_index(index_col, inplace=True, drop=False)
        elif index_col == "Unit_Name" and {"Facility", "Unit"} <= set(df.columns):
            df["Unit_Name"] = df["Facility"].astype(str) + " - Unit " + df["Unit"].astype(str)
            df.set_index("Unit_Name", inplace=True, drop=False)
        else:
            print(f"[WARN] read_csv_if_exists: index_col '{index_col}' not found in {file_path}", flush=True)
    return df

from collections import defaultdict
SESSION_LOCKS = defaultdict(threading.Lock)

def get_session_lock(user_key: str) -> threading.Lock:
    return SESSION_LOCKS[user_key]

for name in ("Stryke.stryke", "stryke"):
    mod = sys.modules.get(name)
    if mod:
        setattr(mod, "read_csv_if_exists", _read_csv_if_exists_compat)

print("[INIT] Patched Stryke.read_csv_if_exists to back-compat shim.", flush=True)
 
class QueueStream:
    """File-like object that writes text lines into a queue.Queue for SSE."""
    def __init__(self, q, prefix: str = "", log_file=None, max_line_length=None):
        self.q = q
        self.prefix = prefix or ""
        self.log_file = log_file
        self.max_line_length = max_line_length if max_line_length is not None else LOG_LINE_MAX_CHARS
        self._buf = []
        self._lock = threading.Lock()

    def _truncate_line(self, line: str) -> str:
        if self.max_line_length and len(line) > self.max_line_length:
            return line[: self.max_line_length] + " ...[truncated]"
        return line

    def write(self, s):
        if not s:
            return 0
        text = str(s)
        # Also write to debug log file if specified
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(text)
            except Exception:
                pass
        with self._lock:
            self._buf.append(text)
            joined = "".join(self._buf)
            lines = joined.splitlines(keepends=True)
            self._buf = []
            carry = ""
            for line in lines:
                if line.endswith("\n"):
                    # strip trailing newline and emit
                    line_out = self._truncate_line(line.rstrip("\n"))
                    try:
                        # Use put_nowait to avoid blocking if queue is full
                        # This prevents simulation from stalling if EventSource disconnects
                        self.q.put_nowait(self.prefix + line_out)
                    except queue.Full:
                        # Queue full - drop oldest messages to make room
                        # This happens when EventSource disconnects but simulation continues
                        try:
                            self.q.get_nowait()  # Remove oldest message
                            self.q.put_nowait(self.prefix + line_out)
                        except Exception:
                            pass  # If still failing, just drop this message
                    except Exception:
                        pass
                else:
                    carry += line
            if carry:
                self._buf.append(carry)
        return len(s)

    def flush(self):
        with self._lock:
            if self._buf:
                try:
                    joined = "".join(self._buf)
                    self.q.put(self.prefix + self._truncate_line(joined))
                except Exception:
                    pass
                self._buf.clear()

    # compatibility no-ops
    def close(self): self.flush()
    def isatty(self): return False

   
class QueueLogHandler(logging.Handler):
    def __init__(self, q): super().__init__(); self.q = q
    def emit(self, record):
        try:
            msg = self.format(record).replace("\n", " | ")
            if LOG_LINE_MAX_CHARS and len(msg) > LOG_LINE_MAX_CHARS:
                msg = msg[:LOG_LINE_MAX_CHARS] + " ...[truncated]"
            # keep it simple: one line per log event
            self.q.put_nowait(msg)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(msg)
            except Exception:
                pass
        except Exception:
            pass
        
def _attach_queue_log_handler(q):
    """Attach a per-run logging handler that writes to the SSE queue."""
    h = QueueLogHandler(q)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    # Attach to this module's logger AND the Stryke logger (if it emits logs)
    targets = [logging.getLogger(__name__), logging.getLogger("Stryke"), logging.getLogger("Stryke.stryke")]
    for lg in targets:
        lg.addHandler(h)
        # ✅ Disable propagation to prevent cross-contamination with root logger
        lg.propagate = False
    return h, targets

def _detach_queue_log_handler(h, targets):
    """Remove per-run logging handler and restore logger state."""
    for lg in targets:
        try:
            lg.removeHandler(h)
            # ✅ Re-enable propagation after cleanup
            lg.propagate = True
        except Exception:
            pass

# ----------------- Password Protection -----------------
app = Flask(__name__)

# Use a secure secret key from environment, fallback to a warning/dev key
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    app.logger.warning("FLASK_SECRET_KEY not set; using insecure dev key — set it in your environment!")
    app.secret_key = 'born4slippy4!'  # dev fallback only

# Password should always come from environment, never hardcoded
app.config['PASSWORD'] = os.environ.get('APP_PASSWORD')
if not app.config['PASSWORD']:
    app.logger.warning("APP_PASSWORD not set; using insecure dev password — set it in your environment!")
    app.config['PASSWORD'] = 'expensive5rudabega!@1'  # dev fallback only

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 200)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax")
app.config["SESSION_COOKIE_SECURE"] = _env_flag("SESSION_COOKIE_SECURE", False)
app.config["DEBUG_ROUTES_ENABLED"] = _env_flag("ENABLE_DEBUG_ROUTES", False)

ALLOWED_EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm"}
ALLOWED_JSON_EXTENSIONS = {".json"}
ALLOWED_STRYKE_EXTENSIONS = {".stryke"}

# Set session lifetime to 1 day (adjust as needed)
app.permanent_session_lifetime = timedelta(days=1)

REQUIRED_FILE_KEYS = {
    "hydrograph_file": "Hydrograph CSV",
    "unit_parameters_file": "Unit parameters CSV",
    "operating_scenarios_file": "Operating scenarios CSV",
    # add any other hard requirements here
}

# Ensure instance path exists; it's a good default for per-app writable data
os.makedirs(app.instance_path, exist_ok=True)

SESSION_ROOT = os.environ.get('SESSION_ROOT') or os.path.join(app.instance_path, 'sessions')
os.makedirs(SESSION_ROOT, exist_ok=True)

app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR=SESSION_ROOT,
    SESSION_PERMANENT=False,
)

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

def _validate_extension(filename: str, allowed: set) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed:
        raise ValueError(f"Invalid file type: {ext}")


def _collect_unit_param_warnings(unit_params_file):
    warnings = []
    if not unit_params_file or not os.path.exists(unit_params_file):
        warnings.append("Unit parameters file is missing.")
        return warnings

    try:
        df_unit = pd.read_csv(unit_params_file)
    except Exception as exc:
        warnings.append(f"Could not read unit parameters file: {exc}")
        return warnings

    if df_unit.empty:
        warnings.append("Unit parameters are empty.")
        return warnings

    missing_cols = [col for col in ("Qopt", "Qcap") if col not in df_unit.columns]
    if missing_cols:
        warnings.append(f"Missing required columns: {', '.join(missing_cols)}.")
        return warnings

    qopt = pd.to_numeric(df_unit["Qopt"], errors="coerce")
    qcap = pd.to_numeric(df_unit["Qcap"], errors="coerce")
    labels = df_unit.index.astype(str)
    if "Facility" in df_unit.columns and "Unit" in df_unit.columns:
        labels = df_unit["Facility"].astype(str) + " - Unit " + df_unit["Unit"].astype(str)
    elif "Unit" in df_unit.columns:
        labels = df_unit["Unit"].astype(str)

    invalid_qopt = qopt.isna() | (qopt <= 0)
    invalid_qcap = qcap.isna() | (qcap <= 0)
    if invalid_qopt.any():
        units = ", ".join(sorted(set(labels[invalid_qopt].tolist())))
        warnings.append(f"Qopt missing/invalid for units: {units}.")
    if invalid_qcap.any():
        units = ", ".join(sorted(set(labels[invalid_qcap].tolist())))
        warnings.append(f"Qcap missing/invalid for units: {units}.")

    return warnings


def _raise_if_unit_params_invalid(unit_params_file):
    warnings = _collect_unit_param_warnings(unit_params_file)
    if warnings:
        raise ValueError("Unit parameters invalid. " + " ".join(warnings))

@app.errorhandler(NotFound)
def handle_404(e):
    # Simple 404, no scary traceback
    return "Not Found", 404

@app.before_request
def make_session_permanent():
    session.permanent = True
    
@app.errorhandler(Exception)
def handle_exception(e):
    # Let real HTTP errors (404, 405, etc.) pass through to Flask’s default pages
    if isinstance(e, HTTPException):
        return e

    # Log full traceback server-side
    tb = traceback.format_exc()
    try:
        app.logger.exception("Unhandled exception")
    except Exception:
        print("Unhandled exception:", tb, flush=True)

    # Minimal safe response
    return "Internal Server Error", 500

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return f"Upload too large. Max upload size is {MAX_UPLOAD_MB} MB.", 413

def _get_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token

@app.context_processor
def inject_csrf_token():
    return {"csrf_token": _get_csrf_token()}

@app.before_request
def csrf_protect():
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not token or token != session.get("csrf_token"):
            return "Invalid CSRF token", 400

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
        max_attempts = _env_int("MAX_LOGIN_ATTEMPTS", 5)
        window_seconds = _env_int("LOGIN_WINDOW_SECONDS", 600)
        attempts = session.get("login_attempts", {"count": 0, "first_ts": time.time()})
        if time.time() - attempts.get("first_ts", 0) > window_seconds:
            attempts = {"count": 0, "first_ts": time.time()}
        if attempts.get("count", 0) >= max_attempts:
            return "Too many login attempts. Try again later.", 429

        if request.form.get('password') == app.config['PASSWORD']:
            session['logged_in'] = True
            session['user_dir'] = uuid.uuid4().hex  # Unique directory per session
            session.pop("login_attempts", None)
            flash("Logged in successfully!")
            return redirect(url_for('index'))
        else:
            attempts["count"] = attempts.get("count", 0) + 1
            session["login_attempts"] = attempts
            error = 'Invalid password. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    user_upload_dir = session.get('user_upload_dir')
    user_sim_folder = session.get('user_sim_folder')

    def _safe_rmtree(base_dir, target_dir):
        if not target_dir:
            return
        base_abs = os.path.abspath(base_dir)
        target_abs = os.path.abspath(target_dir)
        if not target_abs.startswith(base_abs + os.sep):
            raise PermissionError(f"Refusing to delete outside {base_abs}")
        if os.path.exists(target_abs):
            shutil.rmtree(target_abs, ignore_errors=False)

    try:
        if user_upload_dir:
            _safe_rmtree(app.config['UPLOAD_FOLDER'], user_upload_dir)
        if user_sim_folder:
            _safe_rmtree(SIM_PROJECT_FOLDER, user_sim_folder)
    except Exception as e:
        app.logger.warning("Logout cleanup failed: %s", e)

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
                                 # <-- correct
def cleanup_old_data():
    """Remove files and directories older than 24 hours in upload and simulation folders."""
    SESSION_DIR = app.config['SESSION_FILE_DIR']

    now = time.time()
    cutoff = now - 24 * 3600  # 24 hours ago

    for folder in (UPLOAD_FOLDER, SIM_PROJECT_FOLDER):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            # skip the session store
            if os.path.abspath(path) == os.path.abspath(SESSION_DIR):
                continue
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

class _SafeQueueStream:
    """Minimal stdout proxy to push lines into a Queue."""
    def __init__(self, q): self.q = q
    def write(self, s):
        if not s: 
            return
        # split to preserve line breaks without flooding
        for line in str(s).splitlines():
            try: self.q.put(line)
            except Exception: pass
    def flush(self): 
        pass

def run_xls_simulation_in_background(ws, wks, output_name, q, data_dict=None):
    import os, sys, logging
    log = logging.getLogger(__name__)
    success = False

    # resolve Excel path (handles bare filename vs absolute)
    excel_path = wks if os.path.isabs(wks) else os.path.join(ws, wks)
    if not os.path.isdir(ws):
        try: q.put(f"[ERROR] Invalid run directory: {ws}")
        finally: q.put("[Simulation Complete]")
        return
    if not os.path.exists(excel_path):
        try: q.put(f"[ERROR] Excel file not found: {excel_path}")
        finally: q.put("[Simulation Complete]")
        return

    # stream all prints + logger output to this run's queue
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = QueueStream(q)
    sys.stderr = QueueStream(q)
    h, targets = _attach_queue_log_handler(q)

    # optional file lock (safe if filelock not installed)
    h5_path = os.path.join(ws, f"{output_name}.h5")
    try:
        from filelock import FileLock
        lock = FileLock(h5_path + ".lock")
    except Exception:
        lock = None

    try:
        log.info("Starting simulation (XLS path)...")
        # handle older/newer stryke signatures
        try:
            sim = stryke.simulation(ws, wks, output_name=output_name)
        except TypeError:
            sim = stryke.simulation(ws, wks)

        sim.webapp_import(data_dict, output_name)
        if lock:
            with lock:
                sim.run(); sim.summary()
        else:
            sim.run(); sim.summary()
        success = True
            
        # Generate and save the report for XLS simulations
        log.info("Generating simulation report...")
        try:
            report_html = generate_report(sim)
            report_path = os.path.join(sim.proj_dir, 'simulation_report.html')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            # Create marker file for report location
            marker_path = os.path.join(sim.proj_dir, 'report_path.txt')
            with open(marker_path, 'w', encoding='utf-8') as f:
                f.write(report_path)
                
            log.info("Report generated and saved to %s", report_path)
        except Exception as e:
            log.warning("Failed to generate report: %s", e)

        if success:
            complete_marker = os.path.join(sim.proj_dir, "simulation_complete.flag")
            with open(complete_marker, "w", encoding="utf-8") as f:
                f.write(datetime.now().isoformat())

        log.info("Simulation completed successfully.")
    except Exception as e:
        log.exception("Simulation failed (XLS).")
        try: q.put(f"[ERROR] Simulation failed: {e}")
        except Exception: pass
    finally:
        # restore stdio
        sys.stdout, sys.stderr = old_stdout, old_stderr
        # detach handler so it doesn’t leak to future runs
        for lg in targets:
            try:
                lg.removeHandler(h)
            except Exception:
                pass
        try:
            q.put("[Simulation Complete]")
        except Exception:
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_simulation():
    simulation_results = None
    output_filename = None
    run_id = None
    
    
    if request.method == 'POST':
        if 'excel_file' not in request.files:
            flash('No file part in the request')
            return render_template('upload_simulation.html')
    
        file = request.files['excel_file']
        if not file or file.filename.strip() == '':
            flash('No file selected')
            return render_template('upload_simulation.html')

        safe_name = secure_filename(file.filename)
        if not safe_name:
            flash('Invalid file name')
            return render_template('upload_simulation.html')
        try:
            _validate_extension(safe_name, ALLOWED_EXCEL_EXTENSIONS)
        except ValueError:
            flash('Invalid file type. Please upload an Excel file (.xls, .xlsx, .xlsm).')
            return render_template('upload_simulation.html')
    
        user_upload_dir = session.get('user_upload_dir')
        user_sim_folder = session.get('user_sim_folder')
        if not user_upload_dir or not user_sim_folder:
            flash('Session expired. Please log in again.')
            return redirect(url_for('login'))
    
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_sim_folder, exist_ok=True)
    
        # Save the raw upload into the user's upload inbox
        up_file_path = os.path.join(user_upload_dir, safe_name)
        file.save(up_file_path)
        flash(f'File successfully uploaded: {safe_name}')
    
        # --- Create a unique run sandbox under this user's sim folder ---
        run_id = uuid.uuid4().hex
        run_dir = os.path.join(user_sim_folder, run_id)
        os.makedirs(run_dir, exist_ok=True)
    
        # Copy the uploaded Excel into the run sandbox
        simulation_file_path = os.path.join(run_dir, safe_name)
        shutil.copy(up_file_path, simulation_file_path)
    
        # Point Stryke at the run sandbox
        ws = run_dir # per-run directory (prevents collisions)
        wks = safe_name
        output_name = 'Simulation_Output'
    
        # So /report and other views know where to look for THIS run
        session['proj_dir'] = run_dir
        session['run_dir'] = run_dir
        session['output_name'] = output_name
        session['last_run_id'] = run_id
    
        # Bind the background worker to the per-run queue
        q = get_queue(run_id)
        simulation_thread = threading.Thread(
            target=run_xls_simulation_in_background,
            args=(ws, wks, output_name, q), # worker signature: (ws, wks, output_name, queue)
            daemon=True
            )
        simulation_thread.start()
    
        flash('Simulation started. Live log will appear below.')
        simulation_results = 'Simulation is running...'
        output_filename = f'{output_name}.h5'
    
    # GET or after POST: render the page. run_id is None on GET, set on POST
    return render_template(
        'upload_simulation.html',
        simulation_results=simulation_results,
        output_filename=output_filename,
        run_id=run_id
        )
            
        # simulation_file_path = os.path.join(user_sim_folder, file.filename)
        # shutil.copy(up_file_path, simulation_file_path)
    
        # ws = user_sim_folder
        # wks = file.filename
        # output_name = 'Simulation_Output'
    
        # try:
        #     simulation_thread = threading.Thread(
        #         target=run_xls_simulation_in_background,
        #         args=(ws, wks, output_name),  # 4th arg is optional now
        #         daemon=True
        #     )
        #     simulation_thread.start()
        #     flash("Simulation started. Live log will appear below.")
        #     simulation_results = "Simulation is running..."
        #     output_filename = f"{output_name}.h5"
        # except Exception as e:
        #     flash(f"Error starting simulation: {e}")
        #     return render_template('upload_simulation.html')
    
        # return render_template('upload_simulation.html',
        #                        simulation_results=simulation_results,
        #                        output_filename=output_filename)


@app.get('/stream')
def stream():
    run_id = request.args.get('run', '')
    if not run_id:
        return "Missing ?run=<run_id>", 400

    q = get_queue(run_id)
    user_root = session.get('user_sim_folder')
    run_dir = None
    if run_id and user_root:
        run_dir = os.path.join(user_root, run_id)
    if not run_dir:
        run_dir = session.get('run_dir') or session.get('proj_dir')
    if not run_dir or not os.path.exists(run_dir):
        # Fallback: locate run_dir by scanning SIM_PROJECT_FOLDER/<user_dir>/<run_id>
        try:
            for user_dir in os.listdir(SIM_PROJECT_FOLDER):
                candidate = os.path.join(SIM_PROJECT_FOLDER, user_dir, run_id)
                if os.path.isdir(candidate):
                    run_dir = candidate
                    break
        except Exception:
            run_dir = None
    log_file = os.path.join(run_dir, "simulation_debug.log") if run_dir else None

    def event_stream():
        import queue as _q
        file_pos = 0
        try:
            yield "data: [INFO] Log stream connected.\n\n"
            while True:
                try:
                    msg = q.get(timeout=30)  # Increased timeout to 30 seconds
                except _q.Empty:
                    if log_file and os.path.exists(log_file):
                        try:
                            size = os.path.getsize(log_file)
                            if size < file_pos:
                                file_pos = 0
                            if size > file_pos:
                                with open(log_file, "rb") as f:
                                    f.seek(file_pos)
                                    chunk = f.read()
                                    file_pos = f.tell()
                                text = chunk.decode("utf-8", errors="replace")
                                for line in text.splitlines():
                                    yield f"data: {line}\n\n"
                                continue
                        except Exception as exc:
                            yield f"data: [ERROR] Log tail failed: {exc}\n\n"
                            break
                    yield "data: [keepalive]\n\n"
                    continue
                except Exception as e:
                    yield f"data: [ERROR] {str(e)}\n\n"
                    break
                yield f"data: {msg}\n\n"
                if msg == "[Simulation Complete]":
                    break
        finally:
            RUN_QUEUES.pop(run_id, None)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # <— important for Nginx-like proxies
            "Connection": "keep-alive"
        },
    )

@app.get("/log_tail")
def log_tail():
    run_id = request.args.get("run", "")
    if not run_id:
        return jsonify({"error": "Missing run id"}), 400
    try:
        offset = int(request.args.get("offset", "0"))
    except ValueError:
        offset = 0

    user_root = session.get("user_sim_folder")
    run_dir = None
    if run_id and user_root:
        run_dir = os.path.join(user_root, run_id)
    if not run_dir:
        run_dir = session.get("run_dir") or session.get("proj_dir")
    if not run_dir or not os.path.exists(run_dir):
        try:
            for user_dir in os.listdir(SIM_PROJECT_FOLDER):
                candidate = os.path.join(SIM_PROJECT_FOLDER, user_dir, run_id)
                if os.path.isdir(candidate):
                    run_dir = candidate
                    break
        except Exception:
            run_dir = None

    log_file = os.path.join(run_dir, "simulation_debug.log") if run_dir else None
    if not log_file or not os.path.exists(log_file):
        return jsonify({"text": "", "offset": offset})

    try:
        size = os.path.getsize(log_file)
        if offset < 0 or offset > size:
            offset = 0
        with open(log_file, "rb") as f:
            f.seek(offset)
            chunk = f.read()
            new_offset = f.tell()
        text = chunk.decode("utf-8", errors="replace")
    except Exception as exc:
        return jsonify({"error": str(exc), "text": "", "offset": offset}), 500

    completed = False
    if run_dir:
        if os.path.exists(os.path.join(run_dir, "simulation_complete.flag")):
            completed = True
        elif os.path.exists(os.path.join(run_dir, "simulation_report.html")):
            completed = True
    return jsonify({"text": text, "offset": new_offset, "completed": completed})

def _safe_path(base_dir: str, *parts: str) -> str:
    """Join parts to base_dir and ensure the result stays inside base_dir."""
    base_abs = os.path.abspath(base_dir)
    target = os.path.abspath(os.path.join(base_abs, *parts))
    if not target.startswith(base_abs + os.sep):
        raise PermissionError("Not allowed")
    return target

def _get_active_run_dir():
    return session.get('run_dir') or session.get('proj_dir')

@app.route('/results')
def results():
    run_id = request.args.get('run', '')
    if not run_id:
        return "Missing ?run=<run_id>", 400
    output_file = request.args.get('output_file', 'Simulation_Output.h5')

    user_root = session.get('user_sim_folder', '')
    if not user_root:
        flash('Session expired. Please log in again.')
        return redirect(url_for('login'))

    run_dir = os.path.join(user_root, run_id)
    try:
        target_path = _safe_path(run_dir, output_file)
    except PermissionError:
        return "Not allowed", 403
    if not os.path.exists(target_path):
        return "Not found", 404

    return render_template('results.html',
                           output_file=os.path.basename(target_path),
                           run_id=run_id,
                           summary='Simulation ran successfully. You can download the output below.')

@app.route('/download')
def download():
    # require run token
    run_id = request.args.get('run', '')
    if not run_id:
        return "Missing ?run=<run_id>", 400

    filename = request.args.get('filename', '')
    if not filename:
        return "Missing ?filename=...", 400

    user_root = session.get('user_sim_folder', '')
    if not user_root:
        flash('Session expired. Please log in again.')
        return redirect(url_for('login'))

    run_dir = os.path.join(user_root, run_id)
    try:
        target_path = _safe_path(run_dir, filename)
    except PermissionError:
        return "Not allowed", 403

    if not os.path.exists(target_path):
        return "Not found", 404

    return send_file(target_path, as_attachment=True)

@app.route('/export_unit_params_template', methods=['POST'])
def export_unit_params_template():
    """
    Export unit parameters as a reusable template.
    Downloads directly to user's browser (saved to Downloads folder).
    """
    try:
        template_data = {
            'template_type': 'unit_parameters',
            'template_version': '1.0',
            'exported_date': pd.Timestamp.now().isoformat(),
            'project_name': session.get('project_name', 'Unknown')
        }
        
        # Get unit params file content
        user_folder = session.get('user_sim_folder')
        if user_folder and 'unit_params_file' in session:
            unit_params_path = session['unit_params_file']
            if os.path.exists(unit_params_path):
                with open(unit_params_path, 'r') as f:
                    template_data['unit_params_csv'] = f.read()
            else:
                flash('Unit parameters file not found')
                return redirect(request.referrer or url_for('unit_parameters'))
        else:
            flash('No unit parameters configured yet')
            return redirect(request.referrer or url_for('unit_parameters'))
        
        # Create JSON file
        template_json = json.dumps(template_data, indent=2)
        template_name = f"{template_data['project_name'].replace(' ', '_')}_unit_params_template.json"
        
        # Send as downloadable file (goes to Downloads folder)
        return Response(
            template_json,
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename={template_name}'}
        )
        
    except Exception as e:
        flash(f'Error exporting unit parameters template: {str(e)}')
        return redirect(request.referrer or url_for('unit_parameters'))

@app.route('/import_unit_params_template', methods=['POST'])
def import_unit_params_template():
    """
    Import unit parameters template.
    """
    try:
        if 'template_file' not in request.files:
            flash('No template file uploaded')
            return redirect(request.referrer or url_for('unit_parameters'))
        
        file = request.files['template_file']
        if file.filename == '':
            flash('No template file selected')
            return redirect(request.referrer or url_for('unit_parameters'))
        
        if not file.filename.endswith('.json'):
            flash('Template file must be a JSON file')
            return redirect(request.referrer or url_for('unit_parameters'))
        
        # Read and parse template
        template_data = json.loads(file.read().decode('utf-8'))
        
        # Validate template type
        if template_data.get('template_type') != 'unit_parameters':
            flash('Invalid template type. Expected unit parameters template.')
            return redirect(request.referrer or url_for('unit_parameters'))
        
        # Restore unit params file
        user_folder = session.get('user_sim_folder')
        if user_folder and 'unit_params_csv' in template_data:
            unit_params_path = os.path.join(user_folder, 'unit_params.csv')
            with open(unit_params_path, 'w') as f:
                f.write(template_data['unit_params_csv'])
            session['unit_params_file'] = unit_params_path
            flash('Unit parameters template imported successfully!')
        else:
            flash('Template does not contain unit parameters data')
        
        return redirect(url_for('unit_parameters'))
        
    except Exception as e:
        flash(f'Error importing unit parameters template: {str(e)}')
        return redirect(request.referrer or url_for('unit_parameters'))

@app.route('/export_graph_template', methods=['POST'])
def export_graph_template():
    """
    Export graph/network as a reusable template.
    Downloads directly to user's browser (saved to Downloads folder).
    """
    try:
        template_data = {
            'template_type': 'graph',
            'template_version': '1.0',
            'exported_date': pd.Timestamp.now().isoformat(),
            'project_name': session.get('project_name', 'Unknown')
        }
        
        # Get graph file content
        user_folder = session.get('user_sim_folder')
        if user_folder and 'graph_file' in session:
            graph_path = session['graph_file']
            if os.path.exists(graph_path):
                with open(graph_path, 'r') as f:
                    template_data['graph_json'] = f.read()
            else:
                flash('Graph file not found')
                return redirect(request.referrer or url_for('graph_editor'))
        else:
            flash('No graph configured yet')
            return redirect(request.referrer or url_for('graph_editor'))
        
        # Create JSON file
        template_json = json.dumps(template_data, indent=2)
        template_name = f"{template_data['project_name'].replace(' ', '_')}_graph_template.json"
        
        # Send as downloadable file (goes to Downloads folder)
        return Response(
            template_json,
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename={template_name}'}
        )
        
    except Exception as e:
        flash(f'Error exporting graph template: {str(e)}')
        return redirect(request.referrer or url_for('graph_editor'))

@app.route('/import_graph_template', methods=['POST'])
def import_graph_template():
    """
    Import graph/network template.
    """
    try:
        if 'template_file' not in request.files:
            flash('No template file uploaded')
            return redirect(request.referrer or url_for('graph_editor'))
        
        file = request.files['template_file']
        if file.filename == '':
            flash('No template file selected')
            return redirect(request.referrer or url_for('graph_editor'))
        
        if not file.filename.endswith('.json'):
            flash('Template file must be a JSON file')
            return redirect(request.referrer or url_for('graph_editor'))
        
        # Read and parse template
        template_data = json.loads(file.read().decode('utf-8'))
        
        # Validate template type
        if template_data.get('template_type') != 'graph':
            flash('Invalid template type. Expected graph template.')
            return redirect(request.referrer or url_for('graph_editor'))
        
        # Restore graph file
        user_folder = session.get('user_sim_folder')
        if user_folder and 'graph_json' in template_data:
            graph_path = os.path.join(user_folder, 'graph.json')
            with open(graph_path, 'w') as f:
                f.write(template_data['graph_json'])
            session['graph_file'] = graph_path
            flash('Graph template imported successfully!')
        else:
            flash('Template does not contain graph data')
        
        return redirect(url_for('graph_editor'))
        
    except Exception as e:
        flash(f'Error importing graph template: {str(e)}')
        return redirect(request.referrer or url_for('graph_editor'))

@app.route('/save_project', methods=['POST'])
def save_project():
    """
    Save entire project (all session data) as a single .stryke JSON file
    """
    try:
        sim_folder = g.get("user_sim_folder")
        if not sim_folder:
            flash('Session expired. Please log in again.')
            return redirect(url_for('login'))
        
        # Gather all project data from session files
        project_data = {
            'version': '1.0',
            'saved_date': datetime.now().isoformat(),
            'project_info': {},
            'flow_scenarios': {},
            'facilities': {},
            'unit_parameters': {},
            'graph': {},
            'population': {},
            'operating_scenarios': {}
        }
        
        # Project info
        project_csv = os.path.join(sim_folder, 'project.csv')
        if os.path.exists(project_csv):
            df = pd.read_csv(project_csv)
            project_data['project_info'] = df.to_dict(orient='records')
        
        # Flow scenarios
        flow_csv = os.path.join(sim_folder, 'flow.csv')
        if os.path.exists(flow_csv):
            df = pd.read_csv(flow_csv)
            project_data['flow_scenarios'] = df.to_dict(orient='records')
        
        # Hydrograph data (if exists)
        hydrograph_csv = os.path.join(sim_folder, 'hydrograph.csv')
        if os.path.exists(hydrograph_csv):
            df = pd.read_csv(hydrograph_csv)
            project_data['hydrograph'] = df.to_dict(orient='records')
        
        # Facilities
        facilities_csv = os.path.join(sim_folder, 'facilities.csv')
        if os.path.exists(facilities_csv):
            df = pd.read_csv(facilities_csv)
            project_data['facilities'] = df.to_dict(orient='records')
        
        # Unit parameters
        unit_params_csv = os.path.join(sim_folder, 'unit_params.csv')
        if os.path.exists(unit_params_csv):
            with open(unit_params_csv, 'r') as f:
                project_data['unit_parameters']['csv_content'] = f.read()
        
        # Graph/routing
        graph_json = os.path.join(sim_folder, 'graph.json')
        if os.path.exists(graph_json):
            with open(graph_json, 'r') as f:
                project_data['graph'] = json.load(f)
        
        # Population data
        population_csv = os.path.join(sim_folder, 'population.csv')
        if os.path.exists(population_csv):
            df = pd.read_csv(population_csv)
            project_data['population'] = df.to_dict(orient='records')
        
        # Operating scenarios
        op_scenarios_csv = os.path.join(sim_folder, 'operating_scenarios.csv')
        if os.path.exists(op_scenarios_csv):
            df = pd.read_csv(op_scenarios_csv)
            project_data['operating_scenarios'] = df.to_dict(orient='records')
        
        # Create filename from project name or use default
        project_name = 'stryke_project'
        if project_data['project_info']:
            project_name = project_data['project_info'][0].get('Project Name', 'stryke_project')
            # Clean filename
            project_name = re.sub(r'[^\w\s-]', '', project_name).strip().replace(' ', '_')
        
        filename = f"{project_name}.stryke"
        
        # Create JSON response
        json_data = json.dumps(project_data, indent=2)
        
        response = Response(
            json_data,
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename={filename}'
            }
        )
        
        flash('✅ Project saved successfully!')
        return response
        
    except Exception as e:
        flash(f'Error saving project: {str(e)}')
        return redirect(request.referrer or url_for('index'))

@app.route('/load_project', methods=['POST'])
def load_project():
    """
    Load a complete project from a .stryke JSON file
    """
    try:
        print("=== LOAD PROJECT CALLED ===", flush=True)
        sim_folder = g.get("user_sim_folder")
        if not sim_folder:
            print("No sim_folder found", flush=True)
            flash('Session expired. Please log in again.')
            return redirect(url_for('login'))
        
        # Check if file was uploaded
        if 'project_file' not in request.files:
            print("No project_file in request.files", flush=True)
            flash('No file selected')
            return redirect(url_for('create_project'))
        
        file = request.files['project_file']
        if file.filename == '':
            print("Empty filename", flush=True)
            flash('No file selected')
            return redirect(url_for('create_project'))
        
        if not file.filename.endswith('.stryke'):
            print(f"Invalid file type: {file.filename}", flush=True)
            flash('Invalid file type. Please upload a .stryke file')
            return redirect(url_for('create_project'))
        
        print(f"Loading project file: {file.filename}", flush=True)
        
        # Read and parse JSON
        content = file.read().decode('utf-8')
        project_data = json.loads(content)
        
        # Validate version
        if 'version' not in project_data:
            flash('Invalid project file format')
            return redirect(request.referrer or url_for('index'))
        
        # Clear existing session data
        for filename in ['project.csv', 'flow.csv', 'hydrograph.csv', 'facilities.csv', 
                         'unit_params.csv', 'graph.json', 'population.csv', 'operating_scenarios.csv']:
            filepath = os.path.join(sim_folder, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Restore project info
        if project_data.get('project_info'):
            df = pd.DataFrame(project_data['project_info'])
            print(f"DEBUG: project_info DataFrame shape: {df.shape}", flush=True)
            print(f"DEBUG: project_info columns: {list(df.columns)}", flush=True)
            if len(df) > 0:
                print(f"DEBUG: First row data: {df.iloc[0].to_dict()}", flush=True)
            df.to_csv(os.path.join(sim_folder, 'project.csv'), index=False)
            # Update session variables from loaded data
            if len(df) > 0:
                session['project_name'] = df.iloc[0].get('Project Name', '')
                session['project_notes'] = df.iloc[0].get('Project Notes', '')
                session['units'] = df.iloc[0].get('Units', 'metric')
                session['model_setup'] = df.iloc[0].get('Model Setup', '')
                print(f"SET session vars: name={session['project_name']}, units={session['units']}", flush=True)
        else:
            print("DEBUG: No project_info in loaded data!", flush=True)
        
        # Restore flow scenarios
        if project_data.get('flow_scenarios'):
            df = pd.DataFrame(project_data['flow_scenarios'])
            flow_csv_path = os.path.join(sim_folder, 'flow.csv')
            df.to_csv(flow_csv_path, index=False)
            # CRITICAL: Store flow_scenario for simulation
            session['flow_scenario'] = df.to_dict(orient='records')
            # Update session variables from loaded data
            if len(df) > 0:
                # Try both column name formats (with/without space)
                session['scenario_name'] = df.iloc[0].get('Scenario Name', df.iloc[0].get('Scenario', ''))
                session['scenario_number'] = str(df.iloc[0].get('Scenario Number', ''))
                session['season'] = df.iloc[0].get('Season', '')
                session['months'] = df.iloc[0].get('Months', '')
                # Check if it's a hydrograph or static discharge
                flow_val = df.iloc[0].get('Flow', df.iloc[0].get('Discharge'))
                if pd.notna(flow_val) and flow_val != 'hydrograph':
                    session['scenario_type'] = 'static'
                    session['discharge'] = str(flow_val)
                else:
                    session['scenario_type'] = 'hydrograph'
        
        # Restore hydrograph
        if project_data.get('hydrograph'):
            df = pd.DataFrame(project_data['hydrograph'])
            hydro_csv_path = os.path.join(sim_folder, 'hydrograph.csv')
            df.to_csv(hydro_csv_path, index=False)
            # CRITICAL: Store hydrograph_file for simulation
            session['hydrograph_file'] = hydro_csv_path
        
        # Restore facilities
        if project_data.get('facilities'):
            df = pd.DataFrame(project_data['facilities'])
            df.to_csv(os.path.join(sim_folder, 'facilities.csv'), index=False)
            # CRITICAL: Store facilities_data for simulation
            session['facilities_data'] = df.to_dict('records')
            # Build facility_units mapping for operating_scenarios page
            facility_units = {}
            for _, row in df.iterrows():
                facility_units[row['Facility']] = int(row['Units'])
            session['facility_units'] = facility_units
            print(f"DEBUG load_project: Stored {len(session['facilities_data'])} facilities and {len(facility_units)} facility_units in session", flush=True)
        
        # Restore unit parameters
        unit_params_data = project_data.get('unit_parameters')
        if isinstance(unit_params_data, dict) and unit_params_data.get('csv_content'):
            unit_params_path = os.path.join(sim_folder, 'unit_params.csv')
            with open(unit_params_path, 'w') as f:
                f.write(unit_params_data['csv_content'])
            session['unit_params_file'] = unit_params_path
        elif isinstance(unit_params_data, list) and unit_params_data:
            unit_params_path = os.path.join(sim_folder, 'unit_params.csv')
            pd.DataFrame(unit_params_data).to_csv(unit_params_path, index=False)
            session['unit_params_file'] = unit_params_path
        elif isinstance(unit_params_data, dict) and unit_params_data:
            unit_params_path = os.path.join(sim_folder, 'unit_params.csv')
            records = unit_params_data.get('records')
            if isinstance(records, list) and records:
                pd.DataFrame(records).to_csv(unit_params_path, index=False)
                session['unit_params_file'] = unit_params_path
            else:
                df_unit = pd.DataFrame(unit_params_data)
                if not df_unit.empty:
                    df_unit.to_csv(unit_params_path, index=False)
                    session['unit_params_file'] = unit_params_path
        
        # Restore graph
        if project_data.get('graph'):
            with open(os.path.join(sim_folder, 'graph.json'), 'w') as f:
                json.dump(project_data['graph'], f, indent=2)
            session['graph_data'] = project_data['graph']
            
            # CRITICAL: Generate graph_summary from loaded graph data for simulation
            graph_data = project_data['graph']
            summary_nodes = []
            summary_edges = []
            
            # Process nodes
            elements = graph_data.get("elements", {}) if isinstance(graph_data, dict) else {}
            nodes = elements.get("nodes", []) if isinstance(elements, dict) else []
            for node in nodes:
                data = (node or {}).get("data", {})
                node_id = data.get("id")
                if not node_id:
                    continue
                label = data.get("label", node_id)
                surv_fun = data.get("surv_fun", "default")
                survival_rate = data.get("survival_rate")
                
                summary_nodes.append({
                    "ID": label,          # Use label as the ID (human-friendly)
                    "Location": node_id,  # Use node_id as the Location (machine id)
                    "Surv_Fun": surv_fun,
                    "Survival": survival_rate,
                })
            
            # Process edges
            edges = elements.get("edges", []) if isinstance(elements, dict) else []
            for edge in edges:
                data = (edge or {}).get("data", {})
                source = data.get("source")
                target = data.get("target")
                if not source or not target:
                    continue
                # robust weight parsing
                w_raw = data.get("weight", 1.0)
                try:
                    weight = float(w_raw)
                except Exception:
                    weight = 1.0
                
                summary_edges.append({"_from": source, "_to": target, "weight": weight})
            
            # Store graph_summary in session for simulation
            session['graph_summary'] = {"Nodes": summary_nodes, "Edges": summary_edges}
            print(f"DEBUG load_project: Created graph_summary with {len(summary_nodes)} nodes, {len(summary_edges)} edges", flush=True)
        
        # Restore population
        if project_data.get('population'):
            df = pd.DataFrame(project_data['population'])
            pop_csv_path = os.path.join(sim_folder, 'population.csv')
            df.to_csv(pop_csv_path, index=False)
            session['population_csv_path'] = pop_csv_path

            # Ensure required columns exist for simulation
            expected_columns = [
                "Species", "Common Name", "Scenario", "Iterations", "Fish",
                "Simulate Choice", "Entrainment Choice", "Modeled Species",
                "vertical_habitat", "beta_0", "beta_1", "fish_type",
                "dist", "shape", "location", "scale",
                "max_ent_rate", "occur_prob",
                "Length_mean", "Length_sd", "U_crit",
                "length shape", "length location", "length scale"
            ]
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = None
            df = df[expected_columns]

            df_clean = df.where(pd.notnull(df), None)
            session['population_data_for_sim'] = df_clean.to_dict(orient='records')

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
            df_summary = df.rename(columns=summary_column_mapping)
            session['population_dataframe_for_summary'] = df_summary.to_json(orient='records')
        
        # Restore operating scenarios
        if project_data.get('operating_scenarios'):
            df = pd.DataFrame(project_data['operating_scenarios'])
            op_scen_path = os.path.join(sim_folder, 'operating_scenarios.csv')
            df.to_csv(op_scen_path, index=False)
            session['op_scen_file'] = op_scen_path
        
        # Set flag to indicate project was just loaded
        session['project_loaded'] = True
        # CRITICAL: Set proj_dir so other pages can save data
        session['proj_dir'] = sim_folder
        session.modified = True  # Force session to save
        
        # List what files were created
        print(f"FILES CREATED IN {sim_folder}:", flush=True)
        for fname in ['project.csv', 'flow.csv', 'hydrograph.csv', 'facilities.csv', 'unit_params.csv', 'graph.json', 'population.csv', 'operating_scenarios.csv']:
            fpath = os.path.join(sim_folder, fname)
            if os.path.exists(fpath):
                print(f"  ✓ {fname} exists", flush=True)
            else:
                print(f"  ✗ {fname} missing", flush=True)
        
        print(f"BEFORE REDIRECT: session keys={list(session.keys())}", flush=True)
        print(f"BEFORE REDIRECT: project_name={session.get('project_name')}", flush=True)
        print(f"BEFORE REDIRECT: proj_dir={session.get('proj_dir')}", flush=True)
        
        # Clear auto-save since we just loaded a project
        flash('✅ Project loaded successfully! All data has been restored.')
        
        print("Project loaded successfully, redirecting to create_project", flush=True)
        
        # Redirect to create_project to show loaded data
        return redirect(url_for('create_project'))
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}", flush=True)
        flash('Invalid project file format. File is corrupted or not a valid .stryke file.')
        return redirect(url_for('create_project'))
    except Exception as e:
        print(f"Error loading project: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        flash(f'Error loading project: {str(e)}')
        return redirect(url_for('create_project'))

@app.route('/export_partial_template', methods=['POST'])
@app.route('/fit', methods=['GET', 'POST'])
def fit_distributions():
    summary_text = ""
    log_text = ""
    plot_filename = ""
    other_filename = ""
    
    # ✅ Enforce session folder requirement - no fallback to global folder
    sim_folder = g.get("user_sim_folder")
    if not sim_folder:
        flash('Session expired. Please log in again.')
        return redirect(url_for('login'))
    
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

            # Small helpers to compute AD and information-criteria metrics locally
            import math as _math
            from scipy.stats import pareto as _pareto, lognorm as _lognorm, weibull_min as _weibull

            def anderson_darling_statistic_local(sample, cdf_fn):
                x = np.sort(np.asarray(sample))
                n = x.size
                if n == 0:
                    return None
                eps = 1e-12
                F = np.clip(cdf_fn(x), eps, 1.0 - eps)
                i = np.arange(1, n + 1)
                S = np.sum((2 * i - 1) * (np.log(F) + np.log(1.0 - F[::-1])))
                A2 = -n - S / n
                return float(A2)

            def compute_loglik_aic_local(obs, dist_obj, params, floc_fixed=True):
                if params is None:
                    return (None, None, None, None)
                obs = np.asarray(obs)
                n = obs.size
                if n == 0:
                    return (None, None, None, None)
                try:
                    sh, loc, scale = params[0], params[1], params[2]
                    pdf = dist_obj.pdf(obs, sh, loc=loc, scale=scale)
                    pdf = np.clip(pdf, 1e-300, None)
                    loglik = float(np.sum(np.log(pdf)))
                    k = 2 if floc_fixed else 3
                    aic = 2 * k - 2 * loglik
                    if n - k - 1 > 0:
                        aicc = aic + (2 * k * (k + 1)) / float(n - k - 1)
                    else:
                        aicc = aic
                    bic = _math.log(n) * k - 2 * loglik
                    return (loglik, aic, aicc, bic)
                except Exception:
                    return (None, None, None, None)

            # Prepare a container for passing structured metrics to the template
            fit_metrics = {}
            
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
                return render_template('fit_distributions.html', summary=summary_text, log_text=log_text, plot_filename=plot_filename, other_filename=other_filename, fit_metrics=fit_metrics)
            finally:
                captured_output = mystdout.getvalue()
                sys.stdout = old_stdout
    
            # Generate and save the first plot.
            # ✅ Use explicit figure to prevent matplotlib state bleed-over
            fig1 = plt.figure(figsize=(10, 6))
            fish.plot()
            plot_filename = 'fitting_results.png'
            plot_path = os.path.join(sim_folder, plot_filename)
            fig1.savefig(plot_path)
            plt.close(fig1)
            
            # Generate and save the histogram.
            # ✅ Use explicit figure to prevent matplotlib state bleed-over
            fig2 = plt.figure(figsize=(5, 3))
            plt.hist(fish.lengths.tolist(), bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel("Fish Length (cm)")
            plt.ylabel("Frequency")
            plt.title("Distribution of Fish Lengths")
            other_filename = 'fish_lengths.png'
            other_plot_path = os.path.join(sim_folder, other_filename)
            fig2.savefig(other_plot_path)
            plt.close(fig2)
            
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

            # Build structured fit metrics for display on the UI
            try:
                observations = np.asarray(fish.epri.FishPerMft3.values)
            except Exception:
                observations = np.array([])

            # helper to safely read params
            def _params_of(attr):
                try:
                    return getattr(fish, attr)
                except Exception:
                    return None

            # Pareto
            pareto_params = _params_of('dist_pareto')
            pareto_loglik, pareto_aic, pareto_aicc, pareto_bic = compute_loglik_aic_local(observations, _pareto, pareto_params, floc_fixed=True)
            try:
                pareto_ad = anderson_darling_statistic_local(observations, lambda x: _pareto.cdf(x, pareto_params[0], loc=pareto_params[1], scale=pareto_params[2])) if pareto_params is not None else None
            except Exception:
                pareto_ad = None
            fit_metrics['Pareto'] = {
                'params': pareto_params,
                'ks_p': getattr(fish, 'pareto_t', 'N/A'),
                'loglik': pareto_loglik,
                'aicc': pareto_aicc,
                'ad': pareto_ad,
            }

            # Log Normal
            lognorm_params = _params_of('dist_lognorm')
            lognorm_loglik, lognorm_aic, lognorm_aicc, lognorm_bic = compute_loglik_aic_local(observations, _lognorm, lognorm_params, floc_fixed=True)
            try:
                lognorm_ad = anderson_darling_statistic_local(observations, lambda x: _lognorm.cdf(x, lognorm_params[0], loc=lognorm_params[1], scale=lognorm_params[2])) if lognorm_params is not None else None
            except Exception:
                lognorm_ad = None
            fit_metrics['Log Normal'] = {
                'params': lognorm_params,
                'ks_p': getattr(fish, 'log_normal_t', 'N/A'),
                'loglik': lognorm_loglik,
                'aicc': lognorm_aicc,
                'ad': lognorm_ad,
            }

            # Weibull
            weibull_params = _params_of('dist_weibull')
            weibull_loglik, weibull_aic, weibull_aicc, weibull_bic = compute_loglik_aic_local(observations, _weibull, weibull_params, floc_fixed=True)
            try:
                weibull_ad = anderson_darling_statistic_local(observations, lambda x: _weibull.cdf(x, weibull_params[0], loc=weibull_params[1], scale=weibull_params[2])) if weibull_params is not None else None
            except Exception:
                weibull_ad = None
            fit_metrics['Weibull'] = {
                'params': weibull_params,
                'ks_p': getattr(fish, 'weibull_t', 'N/A'),
                'loglik': weibull_loglik,
                'aicc': weibull_aicc,
                'ad': weibull_ad,
            }

            # Only Pareto, Log Normal, and Weibull are included here
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_filename_full = os.path.join(sim_folder, "fitting_results_log.txt")
            with open(log_filename_full, "a") as log_file:
                log_file.write(f"{timestamp} - Query: {summary_text}\n")
                log_file.write(f"{timestamp} - Report: {report_text}\n")
                log_file.write("--------------------------------------------------\n")
            
            return render_template('fit_distributions.html', summary=summary_text, log_text=log_text, plot_filename=plot_filename, other_filename=other_filename, fit_metrics=fit_metrics)

        except Exception as e:
            sys.stdout = old_stdout
            error_message = f"ERROR: {e}"
            print(error_message)
            return render_template('fit_distributions.html', summary=error_message, plot_filename=plot_filename, other_filename=other_filename, fit_metrics=fit_metrics)
             
    else:
        return render_template('fit_distributions.html', summary=summary_text, log_text=log_text, plot_filename=plot_filename, other_filename=other_filename)

@app.route('/plot/<filename>')
def serve_plot(filename):
    # Use the session-specific simulation folder if available.
    sim_folder = g.get("user_sim_folder", SIM_PROJECT_FOLDER)
    if not filename.lower().endswith(".png"):
        return "Invalid file type", 400
    try:
        file_path = _safe_path(sim_folder, filename)
    except PermissionError:
        return "Not allowed", 403
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
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
    print(f"=== CREATE PROJECT CALLED === Method: {request.method}", flush=True)
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
        session['project_loaded'] = False  # Reset flag when creating new project

        # Use explicitly stored session directory
        session['proj_dir'] = session.get('user_sim_folder')
        
        # CRITICAL FIX: Write project data to CSV so it can be saved/loaded
        sim_folder = g.get('user_sim_folder')
        if sim_folder:
            project_csv = os.path.join(sim_folder, 'project.csv')
            project_df = pd.DataFrame([{
                'Project Name': project_name,
                'Project Notes': project_notes,
                'Units': units,
                'Model Setup': model_setup
            }])
            project_df.to_csv(project_csv, index=False)
            print(f"Wrote project data to {project_csv}", flush=True)
        
        flash(f"Project '{project_name}' created successfully!")
        print("Project directory set to:", session['proj_dir'])

        return redirect(url_for('flow_scenarios'))

    # GET request - session-first loading pattern
    # Check session first, then fall back to CSV
    print(f"Session keys: {list(session.keys())}", flush=True)
    project_name = session.get('project_name', '')
    project_notes = session.get('project_notes', '')
    units = session.get('units', 'metric')
    model_setup = session.get('model_setup', '')
    project_loaded = session.get('project_loaded', False)
    print(f"From session: name={project_name}, units={units}, setup={model_setup}, loaded={project_loaded}", flush=True)
    
    # If no session data, try loading from CSV
    if not project_name:
        sim_folder = g.get('user_sim_folder')
        if sim_folder:
            project_csv = os.path.join(sim_folder, 'project.csv')
            if os.path.exists(project_csv):
                try:
                    df = pd.read_csv(project_csv)
                    if len(df) > 0:
                        project_name = df.iloc[0].get('Project Name', '')
                        project_notes = df.iloc[0].get('Project Notes', '')
                        units = df.iloc[0].get('Units', 'metric')
                        model_setup = df.iloc[0].get('Model Setup', '')
                        # Populate session from CSV
                        session['project_name'] = project_name
                        session['project_notes'] = project_notes
                        session['units'] = units
                        session['model_setup'] = model_setup
                except Exception as e:
                    print(f"Error reading existing project data: {e}")
    
    print(f"Rendering create_project.html with: name={project_name}, loaded={project_loaded}", flush=True)
    try:
        return render_template('create_project.html',
                             project_name=project_name or '',
                             project_notes=project_notes or '',
                             units=units or 'metric',
                             model_setup=model_setup or '',
                             project_loaded=project_loaded)
    except Exception as e:
        print(f"ERROR rendering create_project.html: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise

def process_hydrograph_data(raw_data):
    # Try tab first, then comma if only one column detected
    df = pd.read_csv(StringIO(raw_data), delimiter="\t", header=None)
    if df.shape[1] == 1:
        df = pd.read_csv(StringIO(raw_data), delimiter=",", header=None)
    if df.shape[1] != 2:
        raise ValueError("Hydrograph data must have exactly two columns (date and flow), tab- or comma-delimited. Paste directly from Excel.")
    df.columns = ['datetimeUTC', 'DAvgFlow_prorate']
    df['datetimeUTC'] = pd.to_datetime(df['datetimeUTC'], errors='coerce', infer_datetime_format=True)
    df['DAvgFlow_prorate'] = pd.to_numeric(df['DAvgFlow_prorate'], errors='coerce')
    df.dropna(inplace=True)
    # Debug: print columns and first few rows
    print("Hydrograph columns:", df.columns.tolist(), flush=True)
    print("Hydrograph head:", df.head(), flush=True)
    if df.empty or 'datetimeUTC' not in df.columns:
        raise ValueError("No valid hydrograph data found. Please check your input format (two columns, no header, paste directly from Excel).")
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
            try:
                df_hydro = process_hydrograph_data(hydrograph_data)
            except Exception as e:
                flash(f"Hydrograph error: {e}")
                return redirect(url_for('flow_scenarios'))
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
        
        # CRITICAL FIX: Write flow scenario to CSV so it can be saved/loaded
        flow_csv_path = os.path.join(session['proj_dir'], 'flow.csv')
        flow_scenario_df.to_csv(flow_csv_path, index=False)
        print(f"Wrote flow scenario to {flow_csv_path}", flush=True)
        
        # Store the flow scenario in session.
        session['flow_scenario'] = flow_scenario_df.to_dict(orient='records')
        flow_scenarios = session.get("flow_scenario", [])
        if isinstance(flow_scenarios, str):
            flow_scenarios = json.loads(flow_scenarios)
        
        flash("Flow Scenario saved successfully!")
        return redirect(url_for('facilities'))

    # GET request - session-first loading pattern
    units = session.get('units', 'metric')
    project_loaded = session.get('project_loaded', False)
    
    # Check session first for flow scenario data
    scenario_name = session.get('scenario_name', '')
    scenario_number = session.get('scenario_number', '')
    season = session.get('season', '')
    months = session.get('months', '')
    scenario_type = session.get('scenario_type', '')
    discharge = session.get('discharge', '')
    hydrograph_data = session.get('hydrograph_data', '')
    
    # Load from CSV if no scenario data OR if hydrograph data is missing
    sim_folder = g.get('user_sim_folder')
    if sim_folder:
        flow_csv = os.path.join(sim_folder, 'flow.csv')
        if os.path.exists(flow_csv) and (not scenario_name or (scenario_type == 'hydrograph' and not hydrograph_data)):
            try:
                df = pd.read_csv(flow_csv)
                if len(df) > 0:
                    scenario_name = df.iloc[0].get('Scenario', '')
                    scenario_number = str(df.iloc[0].get('Scenario Number', ''))
                    season = df.iloc[0].get('Season', '')
                    months = df.iloc[0].get('Months', '')
                    
                    # Check if it's hydrograph or static
                    flow_val = df.iloc[0].get('Flow')
                    if flow_val == 'hydrograph' or pd.isna(flow_val):
                        scenario_type = 'hydrograph'
                        # Try to load hydrograph data
                        hydro_csv = os.path.join(sim_folder, 'hydrograph.csv')
                        if os.path.exists(hydro_csv):
                            df_hydro = pd.read_csv(hydro_csv)
                            # Convert back to display format (assuming stored in CFS)
                            if units == 'metric' and 'DAvgFlow_prorate' in df_hydro.columns:
                                df_hydro['DAvgFlow_prorate'] = df_hydro['DAvgFlow_prorate'] / 35.3147
                            # Format as tab-delimited string for textarea
                            hydrograph_data = df_hydro.to_csv(sep='\t', index=False, header=False)
                            print(f"Loaded hydrograph with {len(df_hydro)} rows", flush=True)
                    else:
                        scenario_type = 'static'
                        # Convert back to display units
                        discharge_val = float(flow_val)
                        if units == 'metric':
                            discharge_val = discharge_val / 35.3147
                        discharge = str(discharge_val)
                    
                    # Populate session from CSV
                    session['scenario_name'] = scenario_name
                    session['scenario_number'] = scenario_number
                    session['season'] = season
                    session['months'] = months
                    session['scenario_type'] = scenario_type
                    session['discharge'] = discharge
                    session['hydrograph_data'] = hydrograph_data
            except Exception as e:
                print(f"Error reading existing flow scenario data: {e}", flush=True)
    
    return render_template('flow_scenarios.html',
                         units=units,
                         project_loaded=project_loaded,
                         scenario_name=scenario_name,
                         scenario_number=scenario_number,
                         season=season,
                         months=months,
                         scenario_type=scenario_type,
                         discharge=discharge,
                         hydrograph_data=hydrograph_data)

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
        sim_mode = session.get('simulation_mode', 'multiple_powerhouses_simulated_entrainment')
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
        df_facilities = pd.DataFrame(facilities_data)
        # Enforce the expected column order.
        expected_columns = ['Facility', 'Operations', 'Rack Spacing', 'Min_Op_Flow', 'Env_Flow', 'Bypass_Flow', 'Spillway', 'Units']
        df_facilities = df_facilities[expected_columns]
        # Save a JSON-serialized version of the DataFrame into the session.
        session['facilities_dataframe'] = df_facilities.to_json(orient='records')
        
        # CRITICAL FIX: Write facilities to CSV so it can be saved/loaded
        facilities_csv_path = os.path.join(session['proj_dir'], 'facilities.csv')
        df_facilities.to_csv(facilities_csv_path, index=False)
        print(f"Wrote facilities to {facilities_csv_path}", flush=True)

        flash(f"{num_facilities} facility(ies) saved successfully!")
        return redirect(url_for('unit_parameters'))  # Adjust for next page as needed.

        # Debug print the flow scenario DataFrame.
        #print("DEBUG: Facilities DataFrame:")
        #print(facilities_data, flush=True)

    # GET request - load from session or CSV if available
    units = session.get('units', 'metric')
    scenario = session.get('scenario_name', 'Unknown Scenario')
    sim_mode = session.get('simulation_mode', 'multiple_powerhouses_simulated_entrainment_routing')
    project_loaded = session.get('project_loaded', False)
    
    # Try to load existing facilities data from CSV
    facilities_list = []
    sim_folder = g.get('user_sim_folder')
    print(f"DEBUG facilities GET: sim_folder={sim_folder}", flush=True)
    if sim_folder:
        facilities_csv = os.path.join(sim_folder, 'facilities.csv')
        print(f"DEBUG facilities GET: checking {facilities_csv}, exists={os.path.exists(facilities_csv)}", flush=True)
        if os.path.exists(facilities_csv):
            try:
                df = pd.read_csv(facilities_csv)
                print(f"DEBUG facilities GET: loaded CSV with shape {df.shape}, columns={list(df.columns)}", flush=True)
                print(f"DEBUG facilities GET: first row = {df.iloc[0].to_dict() if len(df) > 0 else 'EMPTY'}", flush=True)
                
                # Convert from stored (imperial) back to display units
                if units == 'metric':
                    # Rack spacing: ft to mm
                    if 'Rack Spacing' in df.columns:
                        df['Rack Spacing'] = df['Rack Spacing'] * 304.8  # ft to mm
                    # Flow values: cfs to cms
                    for col in ['Min_Op_Flow', 'Env_Flow', 'Bypass_Flow']:
                        if col in df.columns:
                            df[col] = df[col] / 35.3147
                
                facilities_list = df.to_dict('records')
                print(f"DEBUG facilities GET: facilities_list has {len(facilities_list)} items", flush=True)
                print(f"DEBUG facilities GET: facilities_list[0] = {facilities_list[0] if facilities_list else 'EMPTY'}", flush=True)
            except Exception as e:
                print(f"ERROR loading facilities from CSV: {e}", flush=True)
                import traceback
                traceback.print_exc()
    
    return render_template('facilities.html', 
                         units=units, 
                         scenario=scenario, 
                         sim_mode=sim_mode, 
                         project_loaded=project_loaded,
                         facilities_data=facilities_list)

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
            "penstock_id": "Penstock_ID",
            "penstock_qcap": "Penstock_Qcap",
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
            flow_fields = ["Qopt", "Qcap", "Penstock_Qcap"]

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
    
    # GET request - load existing unit parameters if available
    project_loaded = session.get('project_loaded', False)
    units = session.get('units', 'metric')
    unit_params_list = []
    
    sim_folder = g.get('user_sim_folder')
    
    # Load facilities data into session if not already there (needed for template)
    if sim_folder and 'facilities_data' not in session:
        facilities_csv = os.path.join(sim_folder, 'facilities.csv')
        if os.path.exists(facilities_csv):
            try:
                df_fac = pd.read_csv(facilities_csv)
                # Don't convert units - session stores display format from form submission
                session['facilities_data'] = df_fac.to_dict('records')
                # Build facility_units mapping
                facility_units = {}
                for _, row in df_fac.iterrows():
                    facility_units[row['Facility']] = int(row['Units'])
                session['facility_units'] = facility_units
                print(f"DEBUG: Populated session with {len(session['facilities_data'])} facilities", flush=True)
            except Exception as e:
                print(f"ERROR loading facilities for session: {e}", flush=True)
    
    if sim_folder:
        unit_params_csv = os.path.join(sim_folder, 'unit_params.csv')
        print(f"DEBUG unit_params GET: checking {unit_params_csv}, exists={os.path.exists(unit_params_csv)}", flush=True)
        if os.path.exists(unit_params_csv):
            try:
                df = pd.read_csv(unit_params_csv)
                print(f"DEBUG unit_params GET: loaded CSV with shape {df.shape}, columns={list(df.columns)}", flush=True)
                
                # Convert from stored (imperial) back to display units
                if units == 'metric':
                    conv_length = 3.28084  # feet to meters
                    conv_flow = 35.31469989  # ft³/s to m³/s
                    
                    length_fields = ["intake_vel", "H", "D", "B", "D1", "D2",
                                   "fb_depth", "ps_D", "ps_length", "submergence_depth", "elevation_head"]
                    flow_fields = ["Qopt", "Qcap", "Penstock_Qcap"]
                    
                    for col in length_fields:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce') / conv_length
                    
                    for col in flow_fields:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce') / conv_flow
                
                df_clean = df.where(pd.notnull(df), "")
                unit_params_list = df_clean.to_dict('records')
                print(f"DEBUG unit_params GET: converted to {len(unit_params_list)} records", flush=True)
                
                # Create a lookup dictionary: {"facility|unit": params_dict}
                # Use string keys instead of tuples for JSON serialization
                unit_params_lookup = {}
                for params in unit_params_list:
                    facility = str(params.get('Facility', '') or '').strip()
                    unit_raw = params.get('Unit', '')
                    if isinstance(unit_raw, (int, float, np.integer, np.floating)):
                        unit_raw = float(unit_raw)
                        unit = str(int(unit_raw)) if unit_raw.is_integer() else str(unit_raw)
                    else:
                        unit = str(unit_raw).strip()
                    key = f"{facility}|{unit}"  # String key instead of tuple
                    unit_params_lookup[key] = params
                print(f"DEBUG unit_params GET: created lookup with {len(unit_params_lookup)} entries", flush=True)
                
                # Store in session for template access
                session['unit_params_lookup'] = unit_params_lookup
            except Exception as e:
                print(f"ERROR loading unit parameters from CSV: {e}", flush=True)
                import traceback
                traceback.print_exc()
    
    return render_template('unit_parameters.html', 
                         project_loaded=project_loaded,
                         units=units)

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
        op_scen_path = os.path.join(session['proj_dir'], 'operating_scenarios.csv')
        df_os.to_csv(op_scen_path, index=False)
        print(f"Wrote operating scenarios to {op_scen_path}", flush=True)
        
        # Store only the file path in the session
        session['op_scen_file'] = op_scen_path        
        flash("Operating scenarios saved successfully!")
        return redirect(url_for('graph_editor'))  # Replace with your next route as needed.
    
        # print("DEBUG: Operating Scenarios DataFrame:")
        # print(df_os, flush=True)
        
    # GET request - load existing operating scenarios if available
    project_loaded = session.get('project_loaded', False)
    op_scen_list = []
    
    sim_folder = g.get('user_sim_folder')
    print(f"DEBUG op_scenarios GET: sim_folder={sim_folder}", flush=True)
    if sim_folder:
        op_scen_csv = os.path.join(sim_folder, 'operating_scenarios.csv')
        print(f"DEBUG op_scenarios GET: checking {op_scen_csv}, exists={os.path.exists(op_scen_csv)}", flush=True)
        if os.path.exists(op_scen_csv):
            try:
                df = pd.read_csv(op_scen_csv)
                print(f"DEBUG op_scenarios GET: loaded CSV with shape {df.shape}, columns={list(df.columns)}", flush=True)
                print(f"DEBUG op_scenarios GET: first row = {df.iloc[0].to_dict() if len(df) > 0 else 'EMPTY'}", flush=True)
                
                op_scen_list = df.to_dict('records')
                
                # Create a lookup dictionary: "facility|unit": op_scen_dict
                op_scen_lookup = {}
                for op_scen in op_scen_list:
                    facility = op_scen.get('Facility', '')
                    unit = op_scen.get('Unit', '')
                    # Normalize unit to integer string so keys match template indexing (which uses ints)
                    try:
                        unit_key = str(int(float(unit)))
                    except Exception:
                        unit_key = str(unit)
                    key = f"{facility}|{unit_key}"
                    op_scen_lookup[key] = op_scen
                print(f"DEBUG op_scenarios GET: created lookup with {len(op_scen_lookup)} entries", flush=True)
                
                # Store in session for template access
                session['op_scen_lookup'] = op_scen_lookup
            except Exception as e:
                print(f"ERROR loading operating scenarios from CSV: {e}", flush=True)
                import traceback
                traceback.print_exc()
    
    return render_template('operating_scenarios.html', 
                         project_loaded=project_loaded,
                         has_op_scen_data=len(op_scen_list) > 0)

@app.route('/get_operating_scenarios', methods=['GET'])
def get_operating_scenarios():
    import pandas as pd
    import os
    if DIAGNOSTICS_ENABLED:
        print("[DIAG] get_operating_scenarios called. Session keys:")
        for k in session.keys():
            print(f"  {k}")
    operating_scenarios = []
    if 'op_scen_file' in session:
        os_file = session['op_scen_file']
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] op_scen_file: {os_file}")
            print(f"[DIAG] File exists: {os.path.exists(os_file)}")
        if os.path.exists(os_file):
            df_ops = pd.read_csv(os_file)
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Operating scenarios DataFrame shape: {df_ops.shape}")
                print(f"[DIAG] Operating scenarios DataFrame columns: {df_ops.columns.tolist()}")
                print(f"[DIAG] Operating scenarios DataFrame head:\n{df_ops.head()}")
            # Optionally, select only the columns you need for the dropdown or display
            # For example, if you only need 'Scenario' and 'Unit', you could do:
            # df_ops = df_ops[['Scenario', 'Unit']]
            operating_scenarios = df_ops.to_dict(orient='records')
        if DIAGNOSTICS_ENABLED:
            print(f"[DIAG] Returning {len(operating_scenarios)} operating scenarios.")
    return jsonify(operating_scenarios)

@app.route('/graph_editor', methods=['GET'])
def graph_editor():
    project_loaded = session.get('project_loaded', False)
    graph_data = session.get('graph_data', {})
    
    # If no graph_data in session, try loading from file
    if not graph_data or not graph_data.get('elements'):
        sim_folder = g.get('user_sim_folder')
        if sim_folder:
            graph_file = os.path.join(sim_folder, 'graph.json')
            print(f"DEBUG graph_editor: checking {graph_file}, exists={os.path.exists(graph_file)}", flush=True)
            if os.path.exists(graph_file):
                try:
                    with open(graph_file, 'r') as f:
                        graph_data = json.load(f)
                    print(f"DEBUG graph_editor: loaded graph with {len(graph_data.get('elements', {}).get('nodes', []))} nodes", flush=True)
                    session['graph_data'] = graph_data
                except Exception as e:
                    print(f"ERROR loading graph from file: {e}", flush=True)
    
    print(f"DEBUG graph_editor: graph_data has {len(graph_data.get('elements', {}).get('nodes', []))} nodes", flush=True)
    return render_template('graph_editor.html', graph_data=graph_data, project_loaded=project_loaded)

@app.route('/save_graph', methods=['POST'])
def save_graph():
    log = current_app.logger

    proj_dir = session.get('proj_dir')
    if not proj_dir:
        return jsonify(success=False, error="No project directory"), 400

    try:
        graph_data = request.get_json(silent=False)
    except Exception as e:
        log.exception("Failed to parse JSON body")
        return jsonify(success=False, error=f"Invalid JSON: {e}"), 400

    if not isinstance(graph_data, dict):
        return jsonify(success=False, error="JSON body must be an object"), 400

    session['raw_graph_data'] = graph_data
    log.debug('Graph data acquired')

    summary_nodes = []
    simulation_nodes = {}  # keyed by node_id
    summary_edges = []
    simulation_edges = []

    # --- Nodes ---
    elements = graph_data.get("elements", {}) if isinstance(graph_data, dict) else {}
    nodes = elements.get("nodes", []) if isinstance(elements, dict) else []
    for node in nodes:
        data = (node or {}).get("data", {})
        node_id = data.get("id")
        if not node_id:
            # skip malformed nodes
            log.debug("Skipping node without id: %r", node)
            continue
        label = data.get("label", node_id)
        surv_fun = data.get("surv_fun", "default")
        survival_rate = data.get("survival_rate")

        summary_nodes.append({
            "ID": label,          # Use label as the ID (human-friendly)
            "Location": node_id,  # Use node_id as the Location (machine id)
            "Surv_Fun": surv_fun,
            "Survival": survival_rate,
        })

        simulation_nodes[node_id] = {
            "ID": label,
            "Location": node_id,
            "Surv_Fun": surv_fun,
            "Survival": survival_rate,
        }
    log.debug('Nodes processed: %d', len(simulation_nodes))

    # --- Edges ---
    edges = elements.get("edges", []) if isinstance(elements, dict) else []
    for edge in edges:
        data = (edge or {}).get("data", {})
        source = data.get("source")
        target = data.get("target")
        if not source or not target:
            log.debug("Skipping edge without source/target: %r", edge)
            continue
        # robust weight parsing
        w_raw = data.get("weight", 1.0)
        try:
            weight = float(w_raw)
        except Exception:
            log.debug("Invalid weight %r; defaulting to 1.0", w_raw)
            weight = 1.0

        summary_edges.append({"_from": source, "_to": target, "weight": weight})
        simulation_edges.append((source, target, {"weight": weight}))

    log.debug('Edges processed: %d', len(simulation_edges))

    # --- Build NetworkX graph ---
    G = nx.DiGraph()
    for node_id, attrs in simulation_nodes.items():
        G.add_node(node_id, **attrs)
    for source, target, attrs in simulation_edges:
        G.add_edge(source, target, **attrs)

    sim_graph_data = json_graph.node_link_data(G)

    # --- Write outputs ---
    try:
        os.makedirs(proj_dir, exist_ok=True)
        full_path = os.path.join(proj_dir, 'graph_full.json')
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False)
        
        # CRITICAL FIX: Also save as graph.json for save_project compatibility
        graph_json_path = os.path.join(proj_dir, 'graph.json')
        with open(graph_json_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False)

        node_link_path = os.path.join(proj_dir, 'graph_node_link.json');
        with open(node_link_path, 'w', encoding='utf-8') as f:
            json.dump(sim_graph_data, f, ensure_ascii=False);

        nodes_csv = os.path.join(proj_dir, 'graph_nodes.csv');
        edges_csv = os.path.join(proj_dir, 'graph_edges.csv');
        pd.DataFrame(summary_nodes).to_csv(nodes_csv, index=False);
        pd.DataFrame(summary_edges).to_csv(edges_csv, index=False);
    except Exception as e:
        log.exception("Failed writing graph artifacts")
        return jsonify(success=False, error=f"Write error: {e}"), 500

    # --- Record lightweight pointers & summaries in session ---
    session['graph_files'] = {
        'full': full_path,
        'node_link': node_link_path,
        'nodes_csv': nodes_csv,
        'edges_csv': edges_csv,
    }
    session['graph_summary'] = {"Nodes": summary_nodes, "Edges": summary_edges}
    session.pop('raw_graph_data', None)   # drop heavy blobs
    session.pop('simulation_graph', None)
    session.modified = True

    return jsonify(success=True, summary=session['graph_summary'])

@app.route('/get_unit_parameters', methods=['GET'])
def get_unit_parameters():
        if DIAGNOSTICS_ENABLED:
            print("[DIAG] get_unit_parameters called. Session keys:")
            for k in session.keys():
                print(f"  {k}")
        unit_parameters = []
        if 'unit_params_file' in session:
            unit_params_file = session['unit_params_file']
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] unit_params_file: {unit_params_file}")
                print(f"[DIAG] File exists: {os.path.exists(unit_params_file)}")
            if os.path.exists(unit_params_file):
                df_unit = pd.read_csv(unit_params_file)
                if DIAGNOSTICS_ENABLED:
                    print(f"[DIAG] Unit parameters DataFrame shape: {df_unit.shape}")
                    print(f"[DIAG] Unit parameters DataFrame columns: {df_unit.columns.tolist()}")
                    print(f"[DIAG] Unit parameters DataFrame head:\n{df_unit.head()}")
                # Trim to only the columns needed for the dropdown
                if 'Facility' in df_unit.columns and 'Unit' in df_unit.columns:
                    df_unit = df_unit[['Facility', 'Unit']]
                unit_parameters = df_unit.to_dict(orient='records')
            if DIAGNOSTICS_ENABLED:
                print(f"[DIAG] Returning {len(unit_parameters)} unit parameters.")
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
            "dist": "Log Normal",
            "shape": "1.50",
            "location": "0",
            "scale": "0.02",
            "max_ent_rate": "1.0",
            "occur_prob": "0.40",
            "length shape": "0.50",
            "length location": "-5.0",
            "length scale": "10.0"
        },
        {
            "name": "Perca, Great Lakes, Met Spring",
            "dist": "Log Normal",
            "shape": "1.50",
            "location": "0",
            "scale": "0.03",
            "max_ent_rate": "1.5",
            "occur_prob": "0.60",
            "length shape": "0.50",
            "length location": "-5.0",
            "length scale": "10.0"
        },
        {
            "name": "Perca, Great Lakes, Met Summer",
            "dist": "Log Normal",
            "shape": "1.50",
            "location": "0",
            "scale": "0.04",
            "max_ent_rate": "2.0",
            "occur_prob": "0.80",
            "length shape": "0.50",
            "length location": "-5.0",
            "length scale": "10.0"
        },
        {
            "name": "Perca, Great Lakes, Met Fall",
            "dist": "Log Normal",
            "shape": "1.50",
            "location": "0",
            "scale": "0.03",
            "max_ent_rate": "1.2",
            "occur_prob": "0.70",
            "length shape": "0.50",
            "length location": "-5.0",
            "length scale": "10.0"
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

    # Normalize species names: strip leading/trailing whitespace so entries like
    # " Esox" and " Etheostoma" won't include accidental spaces when matched.
    for _s in species_defaults:
        try:
            if isinstance(_s.get('name'), str):
                _s['name'] = _s['name'].strip()
        except Exception:
            # Be defensive: if an entry is malformed, skip it rather than crash
            continue

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
        pop_data["fish_type"] = request.form.get('fish_type')
        
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
                            if units == 'metric' : 
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
        
        df_population = pd.DataFrame([pop_data])
        #print("DataFrame created with shape:", df_population.shape, flush=True)
        
        expected_columns = [
            "Species", "Common Name", "Scenario", "Iterations", "Fish",
            "Simulate Choice", "Entrainment Choice", "Modeled Species",
            "vertical_habitat", "beta_0", "beta_1", "fish_type",
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
        pop_csv_path = os.path.join(proj_dir, "population.csv")
        df_population.to_csv(pop_csv_path, index=False)
        print(f"Wrote population data to {pop_csv_path}", flush=True)   
        session['population_csv_path'] = pop_csv_path
        df_check = pd.read_csv(pop_csv_path)
        print("CSV Headers:", df_check.columns.tolist(), flush=True)
        #print("Saved population parameters to file:", pop_csv_path, flush=True)

        #print("Population DataFrame for summary:", session.get('population_dataframe_for_summary'), flush=True)
        
        print("Population parameters saved successfully! Redirecting...", flush=True)
        flash("Population parameters saved successfully!")
        return redirect(url_for('model_setup_summary'))

    # GET request - load existing population data if available
    project_loaded = session.get('project_loaded', False)
    population_data = {}
    
    sim_folder = g.get('user_sim_folder')
    print(f"DEBUG population GET: sim_folder={sim_folder}", flush=True)
    if sim_folder:
        pop_csv = os.path.join(sim_folder, 'population.csv')
        print(f"DEBUG population GET: checking {pop_csv}, exists={os.path.exists(pop_csv)}", flush=True)
        if os.path.exists(pop_csv):
            try:
                df = pd.read_csv(pop_csv)
                print(f"DEBUG population GET: loaded CSV with shape {df.shape}, columns={list(df.columns)}", flush=True)
                if len(df) > 0:
                    # Add missing columns for backward compatibility with old project files
                    required_columns = ['Simulate Choice', 'Entrainment Choice', 'Modeled Species', 'fish_type']
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = None
                            print(f"DEBUG: Added missing column '{col}' to population data", flush=True)
                    
                    # For old files, try to infer Simulate Choice from Fish column
                    if pd.isna(df.iloc[0].get('Simulate Choice')) or df.iloc[0].get('Simulate Choice') == '':
                        if 'Fish' in df.columns and pd.notna(df.iloc[0]['Fish']):
                            df.at[0, 'Simulate Choice'] = 'starting population'
                            print(f"DEBUG: Inferred 'Simulate Choice' = 'starting population' from Fish column", flush=True)
                        elif 'max_ent_rate' in df.columns and pd.notna(df.iloc[0]['max_ent_rate']):
                            df.at[0, 'Simulate Choice'] = 'entrainment event'
                            df.at[0, 'Entrainment Choice'] = 'empirical'  # Default guess
                            print(f"DEBUG: Inferred 'Simulate Choice' = 'entrainment event'", flush=True)
                    
                    population_data = df.iloc[0].to_dict()
                    print(f"DEBUG population GET: population_data keys: {list(population_data.keys())}", flush=True)
                    
                    # Convert U_crit from ft/s (stored) to m/s (display) if units are metric
                    units = session.get('units', 'metric')
                    if units == 'metric' and 'U_crit' in population_data:
                        try:
                            if pd.notna(population_data['U_crit']):
                                ucrit_ft = float(population_data['U_crit'])
                                population_data['U_crit'] = ucrit_ft / 3.28084  # Convert ft/s to m/s
                                print(f"DEBUG: Converted U_crit from {ucrit_ft} ft/s to {population_data['U_crit']} m/s", flush=True)
                        except (ValueError, TypeError) as e:
                            print(f"ERROR converting U_crit: {e}", flush=True)
                    
                    # Convert Length_mean from inches (stored) to mm (display) if units are metric
                    if units == 'metric' and 'Length_mean' in population_data:
                        try:
                            if pd.notna(population_data['Length_mean']):
                                length_in = float(population_data['Length_mean'])
                                population_data['Length_mean'] = length_in * 25.4  # Convert inches to mm
                                print(f"DEBUG: Converted Length_mean from {length_in} in to {population_data['Length_mean']} mm", flush=True)
                        except (ValueError, TypeError) as e:
                            print(f"ERROR converting Length_mean: {e}", flush=True)
                    
                    # Convert Length_sd from inches (stored) to mm (display) if units are metric
                    if units == 'metric' and 'Length_sd' in population_data:
                        try:
                            if pd.notna(population_data['Length_sd']):
                                sd_in = float(population_data['Length_sd'])
                                population_data['Length_sd'] = sd_in * 25.4  # Convert inches to mm
                        except (ValueError, TypeError) as e:
                            print(f"ERROR converting Length_sd: {e}", flush=True)
                    
                    # Set beta_0 and beta_1 based on fish_type if they're missing (Pflugrath 2021 values)
                    if 'fish_type' in population_data and pd.notna(population_data['fish_type']):
                        fish_type = str(population_data['fish_type']).lower()
                        # Only set if beta values are missing or NaN
                        if pd.isna(population_data.get('beta_0')) or pd.isna(population_data.get('beta_1')):
                            if fish_type == 'physoclistous':
                                population_data['beta_0'] = -4.8085
                                population_data['beta_1'] = 3.33
                                print(f"DEBUG: Set beta values for Physoclistous: beta_0=-4.8085, beta_1=3.33", flush=True)
                            elif fish_type == 'physostomous':
                                population_data['beta_0'] = -4.93263
                                population_data['beta_1'] = 2.96225
                                print(f"DEBUG: Set beta values for Physostomous: beta_0=-4.93263, beta_1=2.96225", flush=True)
                        else:
                            print(f"DEBUG: beta values already set: beta_0={population_data.get('beta_0')}, beta_1={population_data.get('beta_1')}", flush=True)
                    
                    # Store processed data back to session for simulation
                    session['population_data_for_sim'] = [population_data]
            except Exception as e:
                print(f"ERROR loading population from CSV: {e}", flush=True)
                import traceback
                traceback.print_exc()
    
    return render_template('population.html', 
                         species_defaults=species_defaults, 
                         project_loaded=project_loaded,
                         population_data=population_data)

@app.route('/model_summary')
def model_setup_summary():
    # --- Unit Parameters ---
    unit_parameters = []
    unit_columns = []
    unit_params_file = session.get('unit_params_file')
    if unit_params_file:
        if os.path.exists(unit_params_file):
            try:
                df_unit = pd.read_csv(unit_params_file)
                unit_parameters = df_unit.to_dict(orient='records')
                unit_columns = list(df_unit.columns)
            except Exception as e:
                print("Error reading unit_params_file:", e)
        else:
            print("Unit parameters file not found on disk:", unit_params_file)
    else:
        print("No unit_params_file key in session.")
    unit_param_warnings = _collect_unit_param_warnings(unit_params_file)

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
    log = logging.getLogger(__name__)
    graph_summary = session.get('graph_summary')
    gf = session.get('graph_files') or {}
    node_link_path = gf.get('node_link')
    graph_data = {}
    graph_nodes = []
    graph_edges = []
    
    # Try to get graph from session first
    if 'graph_data' in session and session['graph_data']:
        graph_data = session['graph_data']
        if 'elements' in graph_data:
            graph_nodes = graph_data['elements'].get('nodes', [])
            graph_edges = graph_data['elements'].get('edges', [])
        print(f"DEBUG model_summary: Loaded graph from session: {len(graph_nodes)} nodes, {len(graph_edges)} edges", flush=True)
    # Fallback to file
    elif node_link_path and os.path.exists(node_link_path):
        try:
            with open(node_link_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            if 'elements' in graph_data:
                graph_nodes = graph_data['elements'].get('nodes', [])
                graph_edges = graph_data['elements'].get('edges', [])
            print(f"DEBUG model_summary: Loaded graph from file: {len(graph_nodes)} nodes, {len(graph_edges)} edges", flush=True)
        except Exception:
            log.exception('Failed reading node_link graph file: %s', node_link_path)

    # --- Population Data ---
    pop_df = []
    if 'population_data_for_sim' in session:
        pop_df = session['population_data_for_sim']
        print(f"DEBUG model_summary: Population data from session: {pop_df}", flush=True)
    
    # Ensure it's a list for template iteration
    if pop_df and not isinstance(pop_df, list):
        pop_df = [pop_df]

    # --- Passage Routes (targets of bifurcations) ---
    passage_routes = []
    if graph_edges:
        targets_by_source = {}
        for edge in graph_edges:
            data = edge.get('data', {}) if isinstance(edge, dict) else {}
            source = data.get('source')
            target = data.get('target')
            if source and target:
                targets_by_source.setdefault(source, set()).add(target)
        bifurcation_sources = {src for src, tgts in targets_by_source.items() if len(tgts) > 1}
        seen_targets = set()
        for edge in graph_edges:
            data = edge.get('data', {}) if isinstance(edge, dict) else {}
            source = data.get('source')
            target = data.get('target')
            if source in bifurcation_sources and target and target not in seen_targets:
                passage_routes.append(target)
                seen_targets.add(target)

    # --- Prepare data for template ---
    proj_dir = session.get('proj_dir', '')
    output_name = session.get('project_name', 'unnamed_project')
    
    # Return template with all collected data
    return render_template('model_summary.html',
                         unit_parameters=unit_parameters,
                         unit_columns=unit_columns,
                         unit_param_warnings=unit_param_warnings,
                         operating_scenarios=operating_scenarios,
                         flow_scenarios=flow_scenarios,
                         population_parameters=pop_df,  # Changed from population_data to match template
                         graph_summary=graph_summary,
                         graph_nodes=graph_nodes,  # Added
                         graph_edges=graph_edges,  # Added
                         passage_routes=passage_routes,
                         project_name=session.get('project_name'),
                         project_notes=session.get('project_notes'),
                         model_setup=session.get('model_setup'),
                         units=session.get('units'))

def run_simulation_in_background_custom(data_dict, q):
    """Background worker for UI-driven simulations."""
    import os, sys, logging
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    success = False
    stop_event = threading.Event()
    
    # Also write to a file so we can see everything
    proj_dir = data_dict.get('proj_dir')
    log_file = None
    if proj_dir:
        log_file = os.path.join(proj_dir, 'simulation_debug.log')
        with open(log_file, 'w') as f:
            f.write("=== SIMULATION LOG START ===\n")

    # stream all prints + logger output to this run's queue AND file
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = QueueStream(q, log_file=log_file)
    sys.stderr = QueueStream(q, log_file=log_file)
    h, targets = _attach_queue_log_handler(q)
    def _heartbeat():
        start_ts = time.time()
        while not stop_event.wait(30):
            elapsed = int(time.time() - start_ts)
            msg = f"[INFO] Simulation running... {elapsed // 60}m{elapsed % 60:02d}s elapsed"
            try:
                q.put_nowait(msg)
            except Exception:
                pass
            if log_file:
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                except Exception:
                    pass

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        try:
            q.put_nowait("[INFO] Simulation thread started.")
        except Exception:
            pass
        log.info("Starting UI-driven simulation...")
        proj_dir = data_dict.get('proj_dir')
        output_name = data_dict.get('output_name', 'simulation_output')
        
        if not proj_dir or not os.path.exists(proj_dir):
            raise ValueError(f"Invalid project directory: {proj_dir}")

        _raise_if_unit_params_invalid(data_dict.get('unit_parameters_file'))
            
        # Create simulation instance
        sim = stryke.simulation(proj_dir, output_name, existing=False)
        
        # Import webapp data
        try:
            q.put_nowait("[INFO] Importing webapp data...")
        except Exception:
            pass
        sim.webapp_import(data_dict, output_name)
        
        # Run simulation
        try:
            q.put_nowait("[INFO] Running simulation...")
        except Exception:
            pass
        sim.run()
        sim.summary()
        success = True
        
        # Generate and save the report
        log.info("Generating simulation report...")
        report_html = generate_report(sim)
        report_path = os.path.join(proj_dir, 'simulation_report.html')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        # Create marker file for report location
        marker_path = os.path.join(proj_dir, 'report_path.txt')
        with open(marker_path, 'w', encoding='utf-8') as f:
            f.write(report_path)
            
        log.info("Report generated and saved to %s", report_path)
        if success:
            complete_marker = os.path.join(proj_dir, "simulation_complete.flag")
            with open(complete_marker, "w", encoding="utf-8") as f:
                f.write(datetime.now().isoformat())
        log.info("Simulation completed successfully.")
    except Exception as e:
        log.exception("Simulation failed (UI-driven).")
        try:
            q.put(f"[ERROR] Simulation failed: {e}")
        except Exception:
            pass
    finally:
        stop_event.set()
        # restore stdio
        sys.stdout, sys.stderr = old_stdout, old_stderr
        # detach handler so it doesn't leak to future runs
        for lg in targets:
            try:
                lg.removeHandler(h)
            except Exception:
                pass
        try:
            q.put("[Simulation Complete]")
        except Exception:
            pass

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Start a simulation run from the model summary page."""
    import uuid
    
    # Generate unique run ID
    run_id = uuid.uuid4().hex
    q = get_queue(run_id)
    try:
        q.put_nowait("[INFO] Run queued. Waiting for worker thread...")
    except Exception:
        pass
    
    # Get project directory and output name
    proj_dir = session.get('proj_dir')
    output_name = session.get('project_name', 'simulation_output')
    
    if not proj_dir:
        flash("No project directory found. Please start from the beginning.")
        return redirect(url_for('index'))
        
    # Store run info in session
    session['last_run_id'] = run_id
    session['output_name'] = output_name
    session['last_run_started_at'] = time.time()

    user_root = session.get('user_sim_folder')
    if not user_root:
        flash("Session expired. Please log in again.")
        return redirect(url_for('login'))
    run_dir = os.path.join(user_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    # Create the log file immediately so the SSE tailer can show progress even
    # if the worker thread runs in a different process.
    try:
        log_path = os.path.join(run_dir, "simulation_debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[INFO] Run created at {datetime.now().isoformat()}\n")
    except Exception:
        pass
    session['run_dir'] = run_dir
    
    # Prepare data dictionary
    log = logging.getLogger(__name__)
    graph_summary = session.get('graph_summary')
    gf = session.get('graph_files') or {}
    node_link_path = gf.get('node_link')
    graph_data = {}
    
    # Try to load pre-generated node-link file (created when graph was manually built)
    if node_link_path and os.path.exists(node_link_path):
        try:
            with open(node_link_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            log.info('Loaded node-link graph from file: %s', node_link_path)
        except Exception:
            log.exception('Failed reading node_link graph file: %s', node_link_path)
    
    # If no node-link file, generate it from graph_summary (for loaded projects)
    if not graph_data and graph_summary:
        log.info('Generating node-link graph from graph_summary')
        import networkx as nx
        from networkx.readwrite import json_graph
        
        G = nx.DiGraph()
        
        # Add nodes from graph_summary
        for node in graph_summary.get('Nodes', []):
            node_id = node.get('Location')  # Use Location as the node ID
            if node_id:
                G.add_node(node_id,
                          ID=node.get('ID'),
                          Location=node.get('Location'),
                          Surv_Fun=node.get('Surv_Fun'),
                          Survival=node.get('Survival'))
        
        # Add edges from graph_summary
        for edge in graph_summary.get('Edges', []):
            source = edge.get('_from')
            target = edge.get('_to')
            weight = edge.get('weight', 1.0)
            if source and target:
                G.add_edge(source, target, weight=weight)
        
        # Convert to node-link format
        graph_data = json_graph.node_link_data(G)
        log.info('Generated node-link graph: %d nodes, %d edges', 
                len(graph_data.get('nodes', [])), len(graph_data.get('links', [])))
    
    # Population data
    pop_df = []
    if 'population_data_for_sim' in session:
        pop_df = session['population_data_for_sim']
    
    data_dict = {
        'proj_dir': run_dir,
        'project_name': session.get('project_name'),
        'project_notes': session.get('project_notes'),
        'model_setup': session.get('model_setup'),
        'units': session.get('units'),
        'facilities': session.get('facilities_data'),
        'unit_parameters_file': session.get('unit_params_file'),
        'operating_scenarios_file': session.get('op_scen_file'),
        'population': pop_df,
        'flow_scenarios': session.get('flow_scenario'),
        'hydrograph_file': session.get('hydrograph_file'),
        'graph_data': graph_data,
        'graph_summary': graph_summary,
        'units_system': session.get('units', 'imperial'),
        'simulation_mode': session.get('simulation_mode', 'multiple_powerhouses_simulated_entrainment_routing'),
        'output_name': output_name,
    }
    
    # Start the UI-driven worker with the per-run queue
    t = threading.Thread(target=run_simulation_in_background_custom,
                         args=(data_dict, q),
                         daemon=True)
    t.start()
    flash("Simulation started! Check logs for progress.")
    
    # Redirect to logs page
    return redirect(url_for('simulation_logs', run=run_id))

@app.route("/simulation_logs")
def simulation_logs():
    """
    Show simulation logs page. Checks for completed simulation first.
    """
    run_id = request.args.get('run', '')
    
    # Check if simulation has already completed by looking for output files
    user_root = session.get('user_sim_folder')
    proj_dir = None
    if run_id and user_root:
        proj_dir = os.path.join(user_root, run_id)
    if not proj_dir:
        proj_dir = session.get('run_dir') or session.get('proj_dir')
    simulation_status = 'running'  # Default
    report_path = None
    
    run_started_at = session.get('last_run_started_at')

    def is_fresh(path):
        if not run_started_at:
            return True
        try:
            return os.path.getmtime(path) >= run_started_at
        except Exception:
            return False

    if proj_dir and os.path.exists(proj_dir):
        # Check for simulation completion markers
        marker_file = os.path.join(proj_dir, 'report_path.txt')
        report_html = os.path.join(proj_dir, 'simulation_report.html')
        complete_marker = os.path.join(proj_dir, "simulation_complete.flag")
        output_name = session.get('output_name', 'simulation_output')
        output_h5 = os.path.join(proj_dir, f"{output_name}.h5")
        debug_log = os.path.join(proj_dir, 'simulation_debug.log')
        
        if os.path.exists(marker_file):
            # Read the report path from marker file
            try:
                with open(marker_file, 'r') as f:
                    report_path = f.read().strip()
                    if os.path.exists(report_path) and is_fresh(report_path):
                        simulation_status = 'completed'
                    else:
                        report_path = None
            except Exception:
                pass
        elif os.path.exists(report_html) and is_fresh(report_html):
            # Found report directly
            report_path = report_html
            simulation_status = 'completed'
        elif os.path.exists(complete_marker) and is_fresh(complete_marker):
            simulation_status = 'completed'
            report_path = None
        elif os.path.exists(output_h5) and is_fresh(output_h5) and not run_started_at:
            # Found H5 output (simulation completed but report might be missing)
            simulation_status = 'completed'
            report_path = None  # No HTML report, but sim finished
        
        # Check if simulation is actually in progress by looking at debug log
        # If debug log exists and was modified recently (< 60 seconds), sim is running
        if simulation_status == 'running' and os.path.exists(debug_log):
            try:
                import time
                mtime = os.path.getmtime(debug_log)
                if run_started_at and mtime < run_started_at:
                    pass
                elif time.time() - mtime > 60:
                    # Log hasn't been updated in 60 seconds - might be stalled
                    simulation_status = 'stalled'
            except Exception:
                pass

    return render_template("simulation_logs.html", 
                         run_id=run_id,
                         simulation_status=simulation_status,
                         report_path=report_path)

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
    sim.proj_dir = _get_active_run_dir()
    # Retrieve the output_name from the session, or use a default.
    sim.output_name = session.get('output_name', 'simulation_output')
    return sim

@app.route('/debug_report_path')
def debug_report_path():
    import os
    if not app.config.get("DEBUG_ROUTES_ENABLED"):
        return "Not Found", 404
    try:
        proj_dir = _get_active_run_dir()
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
    log = current_app.logger

    proj_dir = _get_active_run_dir()
    if not proj_dir or not os.path.isdir(proj_dir):
        flash('Session expired or run not initialized.')
        return redirect(url_for('upload_simulation'))

    marker = os.path.join(proj_dir, 'report_path.txt')
    if os.path.exists(marker):
        try:
            with open(marker, 'r', encoding='utf-8') as f:
                report_path = f.read().strip()
        except Exception:
            log.exception('Failed reading report_path marker: %s', marker)
            report_path = os.path.join(proj_dir, 'simulation_report.html')
    else:
        report_path = os.path.join(proj_dir, 'simulation_report.html')

    if not os.path.exists(report_path):
        log.warning('Report not found: %s', report_path)
        return f"<h1>Report not found: {report_path}</h1>", 404

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
    except UnicodeDecodeError:
        log.warning('UTF-8 decode error for %s — trying Latin-1 fallback', report_path)
        with open(report_path, 'r', encoding='latin1', errors='replace') as f:
            report_html = f.read()
    except Exception:
        log.exception('Failed reading report HTML: %s', report_path)
        return 'Failed to read report.', 500

    return Response(report_html, mimetype='text/html')


def generate_report(sim):
    """
    Generate the comprehensive HTML report for the simulation.
    Robust HDF5 open with retry/backoff; guaranteed close; headless plotting.
    """
    import os, time, io, base64, logging
    from datetime import datetime
    import pandas as pd

    # Headless backend for servers / gunicorn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = logging.getLogger(__name__)

    # Allow concurrent reads if configured
    if os.getenv("HDF5_ALLOW_CONCURRENT_READS") == "1":
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    plt.rcParams.update({'font.size': 8})

    hdf_path = os.path.join(sim.proj_dir, f"{sim.output_name}.h5")
    if not os.path.exists(hdf_path):
        log.warning("HDF not found at %s", hdf_path)
        return "<p>Error: HDF file not found. Please run the simulation first.</p>"

    attempts = 12
    base_delay = 0.15
    store = None
    try:
        # Open HDF with retry/backoff to survive transient file locks
        for i in range(attempts):
            try:
                store = pd.HDFStore(hdf_path, mode='r')
                break
            except Exception as e:
                msg = str(e)
                if ("unable to lock file" in msg) or ("Resource temporarily unavailable" in msg):
                    if i == attempts - 1:
                        log.error("HDF5 busy after retries: %s", msg)
                        return "<p>The report file is busy. Please try again in a few seconds.</p>"
                    time.sleep(base_delay * (1.5 ** i))
                    continue
                raise

        log.info("Report: HDF opened: %s", hdf_path)

        def _storer_nrows(hdf_store, key):
            try:
                st = hdf_store.get_storer(key)
                if st is not None and getattr(st, "nrows", None) is not None:
                    return int(st.nrows)
            except Exception:
                return None
            return None

        # --- Executive Summary Metrics ---
        log.info("Report: loading summary tables")
        yearly_df = store["/Yearly_Summary"] if "/Yearly_Summary" in store.keys() else None
        daily_df = store["/Daily"] if "/Daily" in store.keys() else None
        pop_df = store["/Population"] if "/Population" in store.keys() else None

        total_fish = None
        if pop_df is not None:
            for col_name in ("Pop_Size", "Population", "PopSize", "pop_size", "Individuals", "count"):
                if col_name in pop_df.columns:
                    total_fish = pop_df[col_name].sum()
                    break
        if total_fish is None and daily_df is not None and not daily_df.empty and {"species", "iteration", "pop_size"} <= set(daily_df.columns):
            total_fish = daily_df.groupby(["species", "iteration"])["pop_size"].first().sum()
        if total_fish is None:
            total_fish = 0

        mean_entr = mean_mort = prob_entr = 0.0
        overall_surv = 100.0
        if yearly_df is not None and not yearly_df.empty:
            first_row = yearly_df.iloc[0]
            mean_entr = float(first_row.get("mean_yearly_entrainment", 0.0) or 0.0)
            mean_mort = float(first_row.get("mean_yearly_mortality", 0.0) or 0.0)
            prob_entr = float(first_row.get("prob_entrainment", 0.0) or 0.0)

        total_entr = total_surv = 0.0
        if daily_df is not None and not daily_df.empty:
            total_entr = float(daily_df.get("num_entrained", pd.Series(dtype=float)).sum())
            total_surv = float(daily_df.get("num_survived", pd.Series(dtype=float)).sum())
            if total_entr > 0:
                overall_surv = (total_surv / total_entr) * 100.0

        # Get actual fish counts from simulation data for validation
        simulation_keys = [k for k in store.keys() if k.startswith('/simulations/')]
        total_fish_from_sims = 0
        if simulation_keys:
            log.info("Report: counting fish from %d simulation tables", len(simulation_keys))
        for key in simulation_keys:
            nrows = _storer_nrows(store, key)
            if nrows is not None:
                total_fish_from_sims += nrows
                continue
            try:
                sim_data = store[key]
                if isinstance(sim_data, pd.DataFrame):
                    total_fish_from_sims += len(sim_data)
            except Exception:
                pass
        
        # Check if barotrauma mode is active
        barotrauma_active = False
        try:
            unit_params_df = store.get("/Unit_Parameters")
            if unit_params_df is not None and 'ps_D' in unit_params_df.columns:
                barotrauma_active = not unit_params_df['ps_D'].isna().all()
        except (KeyError, AttributeError):
            unit_params_df = None
        
        # Get mortality component data for "Wheel of Death"
        # FIXED: Calculate mean across iterations, not sum across all days×iterations
        mortality_components = {}
        if daily_df is not None and not daily_df.empty:
            mort_cols = ['mortality_impingement', 'mortality_blade_strike', 'mortality_barotrauma']
            available_mort_cols = [c for c in mort_cols if c in daily_df.columns]
            if available_mort_cols:
                # Group by iteration, sum per iteration, then take mean across iterations
                if 'iteration' in daily_df.columns:
                    iter_totals = daily_df.groupby('iteration')[available_mort_cols].sum()
                    for col in available_mort_cols:
                        mortality_components[col.replace('mortality_', '')] = iter_totals[col].mean()
                else:
                    # Fallback if no iteration column
                    for col in available_mort_cols:
                        mortality_components[col.replace('mortality_', '')] = daily_df[col].sum()
            else:
                mortality_components = None
        else:
            mortality_components = None
        
        exec_summary_html = "<h2>Executive Summary</h2>"
        # Use simulation data as the authoritative source
        if total_fish_from_sims > 0:
            total_fish = total_fish_from_sims
        
        if any([total_fish, mean_entr, mean_mort, total_entr]):
            exec_summary_html += (
                f"""
            <div style=\"display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:15px; margin:20px 0;\">
                <div style=\"padding:15px; background:#e8f4f8; border-left:4px solid #0056b3; border-radius:5px;\">
                    <div style=\"font-size:12px; color:#666; text-transform:uppercase;\">Total Fish Simulated</div>
                    <div style=\"font-size:28px; font-weight:bold; color:#0056b3;\">{int(total_fish):,}</div>
                </div>
                <div style=\"padding:15px; background:#fff3cd; border-left:4px solid #ffc107; border-radius:5px;\">
                    <div style=\"font-size:12px; color:#666; text-transform:uppercase;\">Probability of Entrainment</div>
                    <div style=\"font-size:28px; font-weight:bold; color:#ffc107;\">{prob_entr:.2%}</div>
                </div>
                <div style=\"padding:15px; background:#d4edda; border-left:4px solid #28a745; border-radius:5px;\">
                    <div style=\"font-size:12px; color:#666; text-transform:uppercase;\">Overall Passage Survival</div>
                    <div style=\"font-size:28px; font-weight:bold; color:#28a745;\">{overall_surv:.1f}%</div>
                </div>
                <div style=\"padding:15px; background:#f8d7da; border-left:4px solid #dc3545; border-radius:5px;\">
                    <div style=\"font-size:12px; color:#666; text-transform:uppercase;\">Mean Annual Entrainment</div>
                    <div style=\"font-size:28px; font-weight:bold; color:#dc3545;\">{int(mean_entr):,}</div>
                </div>
                <div style=\"padding:15px; background:#e2e3e5; border-left:4px solid #6c757d; border-radius:5px;\">
                    <div style=\"font-size:12px; color:#666; text-transform:uppercase;\">Mean Annual Mortality</div>
                    <div style=\"font-size:28px; font-weight:bold; color:#6c757d;\">{int(mean_mort):,}</div>
                </div>
            </div>
            """
            )
        else:
            exec_summary_html += "<p>Summary metrics not available.</p>"

        # Data validation diagnostics
        baro_status = "🔴 ACTIVE" if barotrauma_active else "⚪ INACTIVE"
        baro_color = "#dc3545" if barotrauma_active else "#6c757d"
        diagnostic_html = f"""
        <details style="margin:20px 0; padding:15px; background:#f8f9fa; border-radius:5px;">
            <summary style="cursor:pointer; font-weight:bold; color:#0056b3;">📊 Data Source Diagnostics (Click to Expand)</summary>
            <div style="margin-top:10px; font-family:monospace; font-size:11px;">
                <p><strong>Fish Count Sources:</strong></p>
                <ul>
                    <li>Total from /simulations/ tables: <strong>{total_fish_from_sims:,}</strong> (all fish that completed passage)</li>
                    <li>Total from /Population table: <strong>{total_fish if total_fish != total_fish_from_sims else 'N/A'}</strong></li>
                    <li>Total entrained (from /Daily): <strong>{int(total_entr):,}</strong></li>
                    <li>Total survived (from /Daily): <strong>{int(total_surv):,}</strong></li>
                </ul>
                <p><strong>Mortality Model Configuration:</strong></p>
                <ul>
                    <li>Barotrauma Mode: <strong style="color:{baro_color};">{baro_status}</strong></li>
                    <li>Blade Strike Model: <strong>✅ ACTIVE</strong> (Franke 1997)</li>
                    <li>Impingement Screening: <strong>✅ ACTIVE</strong></li>
                </ul>
                <p><strong>HDF5 Tables Available:</strong> {', '.join(store.keys())}</p>
                <p style="color:#666; font-style:italic; margin-top:10px;">Note: Route usage counts show final passage destinations. Executive summary shows aggregated daily metrics.</p>
            </div>
        </details>
        """
        
        log.info("Report: assembling HTML sections")
        report_sections = [
            "<div style='margin: 10px;'>"
            "  <button onclick=\"window.location.href='/'\" style='padding:10px;'>Home and Logout</button>"
            "</div>",
            f"<h1>Simulation Report for Project: {getattr(sim, 'project_name', 'N/A')}</h1>",
            f"<p><strong>Project Notes:</strong> {getattr(sim, 'project_notes', 'N/A')}</p>",
            f"<p><strong>Model Setup:</strong> {getattr(sim, 'model_setup', 'N/A')}</p>",
            f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            exec_summary_html,
            diagnostic_html,
        ]

        units = getattr(sim, 'output_units', 'imperial')

        # Helper to render a df nicely (transposes tall tables, rounds Beta Distributions)
        def enforce_horizontal(df, name=""):
            if df is None or df.empty:
                return f"<p>No {name} data available.</p>"
            shape_info = f"<p>{name} data shape: {df.shape}</p>"
            if df.shape[0] > 1 and df.shape[0] > df.shape[1]:
                df = df.T
                shape_info += f"<p>Transposed to shape: {df.shape}</p>"
            if name.lower() == "beta distributions":
                df = df.copy()
                for col in df.select_dtypes(include=["number"]).columns:
                    df[col] = df[col].round(2)
            table_html = df.to_html(index=False, border=1, classes="table")
            return shape_info + f"<div style='overflow-x:auto;'>{table_html}</div>"

        # Table counter for consistent numbering
        table_counter = {'count': 1}
        
        def add_section(title, key, units_mode):
            report_sections.append(f"<h2>{title}</h2>")
            if key in store.keys():
                df = store[key]
                if units_mode == 'metric':
                    if key == '/Unit_Parameters':
                        for c, factor in [
                            ('intake_vel', 0.3048), ('D', 0.3048), ('H', 0.3048),
                            ('Qopt', 0.0283168), ('Qcap', 0.0283168), ('Penstock_Qcap', 0.0283168), ('B', 0.3048),
                            ('D1', 0.3048), ('D2', 0.3048), ('ps_D', 0.3048), ('ps_length', 0.3048),
                            ('fb_depth', 0.3048), ('submergence_depth', 0.3048), ('elevation_head', 0.3048)
                        ]:
                            if c in df.columns:
                                df[c] = df[c] * factor
                    elif key == '/Facilities':
                        for c in ['Bypass_Flow', 'Env_Flow', 'Min_Op_Flow']:
                            if c in df.columns:
                                df[c] = df[c] * 0.0283168
                report_sections.append(enforce_horizontal(df, title))
                # Add caption based on section
                if key == '/Facilities':
                    table_counter['count'] += 1
                    report_sections.append(f"<p style='font-size:0.9em; color:#666; margin-top:8px; font-style:italic;'>Table {table_counter['count']}: Facility-level parameters including bypass flows, environmental flows, and minimum operating conditions.</p>")
                elif key == '/Unit_Parameters':
                    table_counter['count'] += 1
                    report_sections.append(f"<p style='font-size:0.9em; color:#666; margin-top:8px; font-style:italic;'>Table {table_counter['count']}: Unit-specific turbine parameters including dimensions, capacities, and operational characteristics used in blade strike calculations.</p>")
                elif key == '/Operating Scenarios':
                    table_counter['count'] += 1
                    report_sections.append(f"<p style='font-size:0.9em; color:#666; margin-top:8px; font-style:italic;'>Table {table_counter['count']}: Operating priority scenarios defining unit dispatch order and operational constraints.</p>")
            else:
                report_sections.append(f"<p>No {title} data available.</p>")

        # Hydrograph plots
        log.info("Report: hydrograph plots")
        report_sections.append("<h2>Hydrograph Plots</h2>")
        if "/Hydrograph" in store.keys():
            hydrograph_df = store["/Hydrograph"]
            if 'datetimeUTC' in hydrograph_df.columns:
                hydrograph_df = hydrograph_df.copy()
                hydrograph_df['datetimeUTC'] = pd.to_datetime(hydrograph_df['datetimeUTC'])
                if units == 'metric' and 'DAvgFlow_prorate' in hydrograph_df.columns:
                    hydrograph_df['DAvgFlow_prorate'] = hydrograph_df['DAvgFlow_prorate'] * 0.0283168

            def create_hydro_timeseries(df):
                plt.rcParams.update({'font.size': 8})
                fig = plt.figure(figsize=(6, 4))
                if {'datetimeUTC', 'DAvgFlow_prorate'} <= set(df.columns):
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
                if 'DAvgFlow_prorate' in df.columns:
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
                    <p style="font-size:0.9em; color:#666; margin-top:8px; font-style:italic;">
                        Figure 1: Daily flow hydrograph showing temporal variation in discharge rates throughout the simulation period.
                    </p>
                </div>
                <div style="flex:1; min-width:300px; text-align:center;">
                    <h3>Recurrence Histogram</h3>
                    <img src="data:image/png;base64,{hist_b64}" style="max-width:100%; height:auto;" />
                    <p style="font-size:0.9em; color:#666; margin-top:8px; font-style:italic;">
                        Figure 2: Distribution of flow recurrence intervals showing frequency of different discharge magnitudes.
                    </p>
                </div>
            </div>
            """)
        else:
            report_sections.append("<p>No hydrograph data available.</p>")

        # Species Information
        if pop_df is not None and not pop_df.empty:
            report_sections.append("<h2>Species Information</h2>")
            # Include modeled distribution parameters for length
            species_cols = [c for c in ("Species", "Common Name", "length location", "length scale", "U_crit") if c in pop_df.columns]
            if species_cols:
                species_summary = pop_df[species_cols].copy()
                # Rename for better display
                rename_map = {
                    "length location": "Length Mean (modeled)",
                    "length scale": "Length SD (modeled)",
                    "U_crit": "Ucrit (ft/s)"
                }
                species_summary.rename(columns=rename_map, inplace=True)
                # Round numeric columns
                for col in species_summary.columns:
                    if species_summary[col].dtype in ['float64', 'float32']:
                        species_summary[col] = species_summary[col].round(2)
                report_sections.append(f"<div style='overflow-x:auto;'>{species_summary.to_html(index=False, border=1)}</div>")
                report_sections.append("<p style='font-size:0.9em; color:#666; margin-top:8px; font-style:italic;'>Table 1: Species characteristics including modeled length distribution parameters (mean and SD from statistical distribution) and swimming performance (Ucrit).</p>")
            else:
                report_sections.append("<p>Species metadata not available.</p>")
        else:
            report_sections.append("<p>No species information available.</p>")

        # Facility and Operating Data
        report_sections.append("<h2>Facility Configuration</h2>")
        report_sections.append(f"<p><strong>Note:</strong> Unit parameters are stored internally in imperial units (the native units for the Franke turbine equations). All values shown have been converted to {units}, then back to metric if applicable. Minor rounding differences may occur due to unit conversions.</p>")
        add_section("Facility Parameters", "/Facilities", units)
        add_section("Unit Parameters", "/Unit_Parameters", units)
        add_section("Operating Scenarios", "/Operating Scenarios", units)

        # Probability of Entrainment section
        report_sections.append("<h2>Probability of Entrainment</h2>")
        if yearly_df is not None and not yearly_df.empty and 'prob_entrainment' in yearly_df.columns:
            prob_entr = yearly_df.iloc[0]['prob_entrainment']
            report_sections.append(f"""
            <div style="padding:15px; background:#e8f4f8; border-left:4px solid #0056b3; border-radius:5px; margin:20px 0;">
                <p style="font-size:18px; margin:0;"><strong>Overall Probability of Entrainment:</strong> {prob_entr:.4f} ({prob_entr*100:.2f}%)</p>
                <p style="margin:5px 0 0 0; color:#555;">This represents the proportion of fish that become entrained across all days and iterations.</p>
            </div>
            """)
        else:
            report_sections.append("<p>Probability of entrainment data not available.</p>")

        # Survival Analysis and Diagnostics
        report_sections.append("<h2>Survival Analysis</h2>")
        
        # Add Mortality Factor Breakdown
        if mortality_components and sum(mortality_components.values()) > 0:
            report_sections.append("<h3>Mortality Factor Breakdown</h3>")
            
            # Create pie chart
            plt.rcParams.update({'font.size': 9})
            fig, ax = plt.subplots(figsize=(7, 7))
            
            labels = []
            values = []
            colors_map = {
                'impingement': '#e74c3c',      # Red
                'blade_strike': '#f39c12',     # Orange
                'barotrauma': '#9b59b6'        # Purple
            }
            colors = []
            
            for factor, count in mortality_components.items():
                if count > 0:
                    factor_name = factor.replace('_', ' ').title()
                    labels.append(factor_name)
                    values.append(count)
                    colors.append(colors_map.get(factor, '#95a5a6'))
            
            if values:
                wedges, texts, autotexts = ax.pie(
                    values, 
                    labels=labels,
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90,
                    textprops={'fontsize': 10, 'weight': 'bold'}
                )
                ax.set_title('Mortality by Cause\n(of entrained fish that died)', 
                           fontsize=12, weight='bold', pad=20)
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                buf.seek(0)
                wheel_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                total_mortality = sum(values)
                report_sections.append(f"""
                <div style="display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start;">
                    <div style="flex:1; min-width:300px; text-align:center;">
                        <img src="data:image/png;base64,{wheel_b64}" style="max-width:100%; height:auto;" />
                        <p style="font-size:0.9em; color:#666; margin-top:8px; font-style:italic;">
                            Figure 3: Proportional breakdown of fish mortality by causal factor (impingement, blade strike, and barotrauma).
                        </p>
                    </div>
                    <div style="flex:1; min-width:300px;">
                        <p style="margin:10px 0;"><strong>Total Mortalities Analyzed:</strong> {int(total_mortality):,}</p>
                        <table style="width:100%; border-collapse:collapse;">
                            <tr style="background:#f8f9fa;">
                                <th style="padding:8px; text-align:left; border:1px solid #dee2e6;">Cause</th>
                                <th style="padding:8px; text-align:right; border:1px solid #dee2e6;">Count</th>
                                <th style="padding:8px; text-align:right; border:1px solid #dee2e6;">%</th>
                            </tr>
                """)
                
                for i, (label, value) in enumerate(zip(labels, values)):
                    pct = (value / total_mortality) * 100
                    report_sections.append(f"""
                            <tr>
                                <td style="padding:8px; border:1px solid #dee2e6;">
                                    <span style="display:inline-block; width:12px; height:12px; background:{colors[i]}; margin-right:5px;"></span>
                                    {label}
                                </td>
                                <td style="padding:8px; text-align:right; border:1px solid #dee2e6;">{int(value):,}</td>
                                <td style="padding:8px; text-align:right; border:1px solid #dee2e6;">{pct:.1f}%</td>
                            </tr>
                    """)
                
                report_sections.append("""
                        </table>
                        <p style="margin-top:8px; color:#666; font-size:0.9em; font-style:italic;">
                            Table (Inset): Mortality counts and percentages by causal mechanism, showing relative contribution of each hazard type.
                        </p>
                        <p style="margin-top:8px; color:#666; font-size:12px; font-style:italic;">
                            Note: Mortality factors are tracked independently and may not sum to 100% if multiple factors contribute to the same mortality event.
                        </p>
                    </div>
                </div>
                """)
            else:
                report_sections.append("<p>No mortality component data to display.</p>")
        else:
            report_sections.append("<p><em>Mortality factor breakdown not available (add mortality component tracking to enable).</em></p>")
        
        if overall_surv < 70.0:  # Flag low survival rates
            baro_note = " <strong>Barotrauma is ACTIVE</strong> and may be contributing significantly to mortality." if barotrauma_active else " Barotrauma mode is inactive."
            report_sections.append(f"""
            <div style="padding:15px; background:#fff3cd; border-left:4px solid #ffc107; border-radius:5px; margin:20px 0;">
                <p style="margin:0;"><strong>⚠️ Low Survival Rate Detected ({overall_surv:.1f}%)</strong></p>
                <p style="margin:5px 0;">Potential causes for Francis turbine survival below 70%:</p>
                <ul style="margin:5px 0 0 20px;">
                    <li>High head operation increasing blade strike probability</li>
                    <li>Large fish size relative to turbine runner clearances</li>
                    <li>Off-design flow conditions reducing hydraulic efficiency</li>
                    <li>Barotrauma effects from rapid pressure changes</li>
                </ul>
                <p style="margin:5px 0 0 0; font-style:italic;">{baro_note} Check unit parameters (head, RPM, blade design) and fish length distributions.</p>
            </div>
            """)
        
        # Driver diagnostics (unit-level) for identifying outsized effects.
        driver_df = None
        if "/Driver_Diagnostics" in store.keys():
            driver_df = store["/Driver_Diagnostics"]
        elif "/Unit_Parameters" in store.keys():
            try:
                unit_params = store["/Unit_Parameters"]
                beta_units = store["/Beta_Distributions_Units"] if "/Beta_Distributions_Units" in store.keys() else None
                route_flows = store["/Route_Flows"] if "/Route_Flows" in store.keys() else None
                if isinstance(unit_params, pd.DataFrame) and not unit_params.empty:
                    tmp_diag = unit_params.copy()
                    tmp_diag["route"] = tmp_diag.index.astype(str)
                    preferred_cols = [
                        "H", "RPM", "D", "N", "Qopt", "Qcap",
                        "D1", "D2", "B", "ada", "intake_vel",
                        "ps_D", "ps_length", "fb_depth", "submergence_depth",
                        "elevation_head"
                    ]
                    keep_cols = ["route"] + [c for c in preferred_cols if c in tmp_diag.columns]
                    tmp_diag = tmp_diag[keep_cols]

                    if isinstance(beta_units, pd.DataFrame) and not beta_units.empty:
                        beta_units = beta_units.copy()
                        rename_map = {}
                        if "Passage Route" in beta_units.columns:
                            rename_map["Passage Route"] = "route"
                        if "state" in beta_units.columns:
                            rename_map["state"] = "route"
                        if "Mean" in beta_units.columns:
                            rename_map["Mean"] = "survival_mean"
                        if "survival rate" in beta_units.columns:
                            rename_map["survival rate"] = "survival_mean"
                        if "Variance" in beta_units.columns:
                            rename_map["Variance"] = "survival_variance"
                        if "variance" in beta_units.columns:
                            rename_map["variance"] = "survival_variance"
                        if "Lower 95% CI" in beta_units.columns:
                            rename_map["Lower 95% CI"] = "survival_lcl"
                        if "ll" in beta_units.columns:
                            rename_map["ll"] = "survival_lcl"
                        if "Upper 95% CI" in beta_units.columns:
                            rename_map["Upper 95% CI"] = "survival_ucl"
                        if "ul" in beta_units.columns:
                            rename_map["ul"] = "survival_ucl"
                        if rename_map:
                            beta_units.rename(columns=rename_map, inplace=True)

                        if "route" in beta_units.columns:
                            beta_keep = [
                                c for c in [
                                    "route",
                                    "survival_mean",
                                    "survival_variance",
                                    "survival_lcl",
                                    "survival_ucl"
                                ] if c in beta_units.columns
                            ]
                            if beta_keep:
                                beta_units = beta_units[beta_keep]
                                numeric_cols = [c for c in beta_keep if c != "route"]
                                if numeric_cols:
                                    beta_units = beta_units.groupby("route", as_index=False)[numeric_cols].mean()
                                tmp_diag = tmp_diag.merge(beta_units, on="route", how="left")

                    if isinstance(route_flows, pd.DataFrame) and not route_flows.empty:
                        rf = route_flows.copy()
                        if "route" in rf.columns and "discharge_cfs" in rf.columns:
                            rf["route"] = rf["route"].astype(str)
                            route_mean = rf.groupby("route")["discharge_cfs"].mean()
                            tmp_diag = tmp_diag.merge(
                                route_mean.rename("mean_discharge_cfs"),
                                left_on="route",
                                right_index=True,
                                how="left"
                            )
                            unit_routes = set(tmp_diag["route"].dropna().astype(str).tolist())
                            unit_mean = route_mean[route_mean.index.isin(unit_routes)]
                            unit_total = float(unit_mean.sum())
                            share_series = None
                            if unit_total > 0:
                                share_series = unit_mean / unit_total
                            else:
                                total_all = float(route_mean.sum())
                                if total_all > 0:
                                    share_series = route_mean / total_all
                            if share_series is not None:
                                tmp_diag = tmp_diag.merge(
                                    share_series.rename("flow_share"),
                                    left_on="route",
                                    right_index=True,
                                    how="left"
                                )

                    if "flow_share" in tmp_diag.columns and "survival_mean" in tmp_diag.columns:
                        tmp_diag["mortality_weight"] = tmp_diag["flow_share"] * (1 - tmp_diag["survival_mean"])
                    driver_df = tmp_diag
            except Exception:
                driver_df = None
        if isinstance(driver_df, pd.DataFrame) and not driver_df.empty:
            report_sections.append("<h3>Driver Diagnostics (Unit-Level)</h3>")
            report_sections.append(
                "<p style='margin-top:4px; color:#666; font-size:0.9em; font-style:italic;'>"
                "Flow share is computed from mean discharge in the Route_Flows table. "
                "Weighted Mortality Index = flow_share * (1 - survival_mean). "
                "Higher values indicate outsized contributions to project-wide mortality."
                "</p>"
            )

            display_df = driver_df.copy()
            if 'route' not in display_df.columns:
                display_df['route'] = display_df.index.astype(str)

            if units == 'metric':
                length_cols = [
                    'H', 'D', 'D1', 'D2', 'B', 'ps_D', 'ps_length',
                    'fb_depth', 'submergence_depth', 'elevation_head', 'intake_vel'
                ]
                flow_cols = ['Qopt', 'Qcap', 'Penstock_Qcap', 'mean_discharge_cfs']
                for col in length_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col] * 0.3048
                for col in flow_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col] * 0.0283168

            if 'flow_share' in display_df.columns:
                display_df['flow_share_pct'] = display_df['flow_share'] * 100.0

            sort_col = None
            if 'mortality_weight' in display_df.columns:
                sort_col = 'mortality_weight'
            elif 'flow_share' in display_df.columns:
                sort_col = 'flow_share'
            if sort_col:
                display_df = display_df.sort_values(sort_col, ascending=False)

            discharge_label = 'Mean Discharge (m³/s)' if units == 'metric' else 'Mean Discharge (cfs)'
            col_candidates = [
                ('route', 'Route'),
                ('flow_share_pct', 'Flow Share (%)'),
                ('mean_discharge_cfs', discharge_label),
                ('mean_abs_pct_off_qopt', 'Mean % Off Qopt'),
                ('pct_hours_outside_qopt_band', 'Hours Outside Qopt Band (%)'),
                ('total_hours', 'Total Hours'),
                ('survival_mean', 'Survival Mean'),
                ('survival_lcl', 'Survival LCL'),
                ('survival_ucl', 'Survival UCL'),
                ('mortality_weight', 'Weighted Mortality Index'),
                ('H', 'H'),
                ('RPM', 'RPM'),
                ('D', 'D'),
                ('N', 'N'),
                ('Qopt', 'Qopt'),
                ('Qcap', 'Qcap'),
                ('ps_D', 'ps_D'),
                ('ps_length', 'ps_length'),
            ]

            display_cols = [c for c, _ in col_candidates if c in display_df.columns]
            if display_cols:
                display_df = display_df[display_cols]
                display_df = display_df.rename(columns={c: label for c, label in col_candidates if c in display_cols})
                for col in display_df.columns:
                    if pd.api.types.is_numeric_dtype(display_df[col]):
                        display_df[col] = display_df[col].round(4)
                report_sections.append(
                    f"<div style='overflow-x:auto;'>{display_df.to_html(index=False, border=1, classes='table')}</div>"
                )
            else:
                report_sections.append("<p>No driver diagnostics columns available.</p>")

            report_sections.append(
                "<p style='margin-top:4px; color:#666; font-size:0.9em; font-style:italic;'>"
                "Note: Double-runner penstocks can be approximated by representing each runner as a separate unit "
                "or splitting penstock flow if routing effects appear overstated."
                "</p>"
            )

        # Beta distributions - show all passage routes (units and interior nodes) with ALL iterations
        report_sections.append("<h3>Survival Probability Distributions</h3>")
        # Attempt to display an explanatory caption produced by the simulation summarizer.
        caption_text = None
        try:
            if "/Beta_Distributions_Units" in store.keys():
                meta = store.get_storer("Beta_Distributions_Units").attrs
            elif "/Beta_Distributions" in store.keys():
                meta = store.get_storer("Beta_Distributions").attrs
            else:
                meta = None
            if meta is not None and hasattr(meta, 'beta_caption'):
                caption_text = meta.beta_caption
        except Exception:
            caption_text = None

        if caption_text:
            report_sections.append(f"<p style='margin-top:4px; color:#666; font-size:0.9em; font-style:italic;'>{caption_text}</p>")
        else:
            report_sections.append("<p style='margin-top:4px; color:#666; font-size:0.9em; font-style:italic;'><strong>Note:</strong> Means shown are empirical averages of per-iteration, per-day survival probabilities; 95% intervals are bootstrap estimates. Direct beta fits can be sensitive to exact 0/1 values and are not used for the reported means.</p>")
        if "/Beta_Distributions_Units" in store.keys():
            add_section("Beta Distributions (Passage Routes Only)", "/Beta_Distributions_Units", units)
        else:
            add_section("Beta Distributions (All Routes)", "/Beta_Distributions", units)

        # Flow vs Entrainment Analysis  
        log.info("Report: flow vs entrainment analysis")
        report_sections.append("<h2>Flow vs Entrainment Relationship</h2>")
        if daily_df is not None and not daily_df.empty and 'flow' in daily_df.columns:
            df_flow = daily_df.copy()

            def create_entrainment_scatter(data):
                plt.rcParams.update({'font.size': 8})
                fig = plt.figure(figsize=(6, 4))
                plt.scatter(data['flow'], data['num_entrained'], alpha=0.6, s=20, color='#ff8c00', edgecolors='none')
                plt.xlabel('Flow (cfs)' if units == 'imperial' else 'Flow (m³/s)')
                plt.ylabel('Number Entrained')
                plt.title('Entrainment vs Flow')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                return base64.b64encode(buf.getvalue()).decode('utf-8')

            if 'num_entrained' in df_flow.columns:
                scatter_entr = create_entrainment_scatter(df_flow)
                report_sections.append(
                    f"<div style='text-align:center;'>"
                    f"<img src=\"data:image/png;base64,{scatter_entr}\" style='max-width:100%; height:auto;' />"
                    "<p style='font-size:0.9em; color:#666; margin-top:8px; font-style:italic;'>"
                    "Figure 4: Scatter plot showing relationship between flow rate and number of fish entrained, helping identify flow-dependent entrainment patterns."
                    "</p>"
                    "</div>"
                )
            else:
                report_sections.append("<p>Entrainment data not available.</p>")
        else:
            report_sections.append("<p>Flow relationship data not available.</p>")

        # Passage Route Usage Analysis  
        log.info("Report: passage route usage aggregation starting")
        report_sections.append("<h2>Passage Route Usage</h2>")
        simulation_keys = [k for k in store.keys() if k.startswith('/simulations/')]
        combined_route_counts = pd.Series(dtype=float)
        route_surv_sum = defaultdict(float)
        route_surv_count = defaultdict(float)
        units_label = "m³/s" if units == 'metric' else "cfs"
        route_flow_df = store["/Route_Flows"] if "/Route_Flows" in store.keys() else None
        if isinstance(route_flow_df, pd.DataFrame) and not route_flow_df.empty:
            route_flow_df = route_flow_df.copy()
        else:
            route_flow_df = None
        need_estimated_flow = route_flow_df is None
        discharge_records = []
        flow_conversion = 0.0283168 if units == 'metric' else 1.0
        
        # Load graph to identify bifurcation points (nodes with multiple downstream options)
        # Bifurcation nodes are where fish make passage decisions (e.g., unit vs spillway)
        bifurcation_map = {}  # Maps source node -> list of possible target nodes
        all_targets = set()  # All nodes that are targets of bifurcations (actual passage routes)
        try:
            # Use sim.proj_dir instead of session (to avoid "Working outside of request context" error)
            session_proj_dir = getattr(sim, 'proj_dir', None) or os.path.dirname(hdf_path)
            graph_json_path = os.path.join(session_proj_dir, 'graph.json')
            if os.path.exists(graph_json_path):
                with open(graph_json_path, 'r') as f:
                    graph_data = json.load(f)
                    if 'elements' in graph_data and 'edges' in graph_data['elements']:
                        # Build adjacency list of outgoing edges from each node
                        edge_map = {}
                        for edge in graph_data['elements']['edges']:
                            source = edge['data']['source']
                            target = edge['data']['target']
                            if source not in edge_map:
                                edge_map[source] = []
                            edge_map[source].append(target)
                        
                        # Identify bifurcation nodes (multiple outgoing edges)
                        for source, targets in edge_map.items():
                            if len(targets) > 1:
                                bifurcation_map[source] = targets
                                # The targets of bifurcations are the actual passage routes
                                all_targets.update(targets)
                        
                        print(f"[DEBUG] Identified {len(bifurcation_map)} bifurcation nodes: {list(bifurcation_map.keys())}", flush=True)
                        print(f"[DEBUG] Passage route targets: {all_targets}", flush=True)
        except Exception as e:
            print(f"[WARNING] Could not load graph for bifurcation analysis: {e}", flush=True)
            bifurcation_map = {}
            all_targets = set()

        for idx_key, key in enumerate(simulation_keys):
            if idx_key == 0 or (idx_key + 1) % 5 == 0:
                log.info("Report: processing simulation table %d/%d", idx_key + 1, len(simulation_keys))
            sim_data = store[key]
            if not isinstance(sim_data, pd.DataFrame):
                continue
            state_cols = [c for c in sim_data.columns if c.startswith('state_')]
            if not state_cols:
                continue

            # For each fish, identify ALL passage routes they took (for multi-facility scenarios)
            # Each fish may pass through multiple bifurcations (e.g., facility1 then facility2)
            # We need to count each passage event separately
            all_passage_events = []
            
            # Identify survival columns for mapping to state columns (if present)
            survival_cols = [c for c in sim_data.columns if c.startswith('survival_')]

            for idx, row in sim_data.iterrows():
                # Prefer the persistent fish identifier stored in the row (if present).
                # Pandas may load a column named 'index' (from the HDF) while the DataFrame
                # also has its own row index. Use the stored 'index' column when available
                # so deduplication across days works correctly.
                fish_identifier = row.get('index') if 'index' in sim_data.columns else idx
                # Get the sequence of states this fish passed through
                fish_path = [row[col] for col in state_cols if pd.notna(row[col])]
                
                if bifurcation_map:
                    # Find ALL bifurcation targets in this fish's path
                    # Each one represents a separate passage event
                    for pos, state in enumerate(fish_path):
                        if state in all_targets:
                            # Map state position -> survival column (survival_{pos-1}) when possible
                            survived = None
                            try:
                                # Map the state position to the same-index survival column
                                # (simulation writes survival_k for state_k)
                                if pos >= 0 and survival_cols and len(survival_cols) > pos:
                                    surv_col = survival_cols[pos]
                                    survived = row.get(surv_col)
                                    # Ensure we check the extracted value
                                    if pd.notna(survived):
                                        try:
                                            survived = int(survived)
                                        except Exception:
                                            pass
                            except Exception:
                                survived = None
                            # This is a passage route (target of a bifurcation decision)
                            all_passage_events.append({
                                'fish_id': fish_identifier,
                                'passage_route': state,
                                'iteration': row.get('iteration'),
                                'day': row.get('day'),
                                'flow': row.get('flow'),
                                'survived': survived
                            })
                else:
                    # Fallback: Find first non-river_node state
                    # This works for simple cases but may include intermediate nodes
                    for pos, state in enumerate(fish_path):
                        if 'river_node' not in state.lower():
                            survived = None
                            try:
                                if pos > 0 and survival_cols and len(survival_cols) >= pos:
                                    surv_col = survival_cols[pos-1]
                                    survived = row.get(surv_col)
                            except Exception:
                                survived = None
                            all_passage_events.append({
                                'fish_id': fish_identifier,
                                'passage_route': state,
                                'iteration': row.get('iteration'),
                                'day': row.get('day'),
                                'flow': row.get('flow'),
                                'survived': survived
                            })
                            break
            
            # Convert passage events to DataFrame
            if all_passage_events:
                passage_df = pd.DataFrame(all_passage_events)

                # Deduplicate according to policy B: count unique fish per iteration.
                # This ensures the report shows the average number of individual fish
                # passing each route per iteration, instead of counting repeated
                # daily events for the same fish multiple times.
                if not passage_df.empty:
                    # Because fish indices are assigned per-day (they reset each day),
                    # we must treat the unique fish instance as the tuple (fish_id, iteration, day).
                    # Policy B (average across iterations) requires computing per-iteration
                    # totals by summing unique fish per day within that iteration.
                    passage_df_unique_iter = passage_df.drop_duplicates(subset=['fish_id', 'iteration', 'day'])
                    # unique per day for discharge/flow estimation (same as above)
                    passage_df_unique_day = passage_df.drop_duplicates(subset=['fish_id', 'iteration', 'day'])
                else:
                    passage_df_unique_iter = passage_df
                    passage_df_unique_day = passage_df

                # Debug: report raw and deduped sizes
                try:
                    print(f"[ROUTE AGG DEBUG] raw_events={len(passage_df)}, unique_iter_events={len(passage_df_unique_iter)}, unique_day_events={len(passage_df_unique_day)}", flush=True)
                except Exception:
                    pass

                # Count passage events by route using unique fish per iteration
                route_counts = passage_df_unique_iter['passage_route'].value_counts()
                # Filter out any remaining river nodes (shouldn't happen, but defensive)
                route_counts = route_counts[~route_counts.index.str.contains('river_node', case=False, na=False)]
                combined_route_counts = combined_route_counts.add(route_counts, fill_value=0)

                # Accumulate survival stats by route (avoid second pass)
                if 'survived' in passage_df_unique_iter.columns:
                    try:
                        surv_grp = passage_df_unique_iter.groupby('passage_route')['survived'].agg(['count', 'sum'])
                        for route, row in surv_grp.iterrows():
                            route_surv_sum[route] += float(row.get('sum', 0.0) or 0.0)
                            route_surv_count[route] += float(row.get('count', 0.0) or 0.0)
                    except Exception:
                        pass

                # Store passage_df_unique_day for discharge calculations if needed
                if need_estimated_flow and 'flow' in passage_df_unique_day.columns:
                    sim_data_for_discharge = passage_df_unique_day
                else:
                    sim_data_for_discharge = None
            else:
                sim_data_for_discharge = None
                # Debug: report that this simulation key produced no passage events
                print(f"[ROUTE AGG DEBUG] simulation key={key} produced 0 passage events; sim_rows={len(sim_data.index)}", flush=True)
                continue

            if need_estimated_flow and sim_data_for_discharge is not None:
                # Use the passage events DataFrame for discharge calculations
                sim_sub = sim_data_for_discharge[['iteration', 'day', 'flow', 'passage_route']].copy()
                # Filter out river nodes from discharge calculations
                sim_sub = sim_sub[~sim_sub['passage_route'].str.contains('river_node', case=False, na=False)]
                if sim_sub.empty:
                    continue
                    
                sim_sub['flow_converted'] = sim_sub['flow'].astype(float) * flow_conversion

                per_day_totals = (
                    sim_sub.groupby(['iteration', 'day']).size().rename('total_fish').reset_index()
                )
                per_day_flows = (
                    sim_sub.groupby(['iteration', 'day'])['flow_converted'].first().reset_index()
                )
                route_day_counts = (
                    sim_sub.groupby(['iteration', 'day', 'passage_route'])
                    .size()
                    .rename('route_count')
                    .reset_index()
                )
                route_day_counts = route_day_counts.merge(per_day_totals, on=['iteration', 'day'], how='left')
                route_day_counts = route_day_counts.merge(per_day_flows, on=['iteration', 'day'], how='left')
                route_day_counts = route_day_counts[route_day_counts['total_fish'] > 0]

                if not route_day_counts.empty:
                    route_day_counts['estimated_discharge'] = (
                        route_day_counts['route_count'] / route_day_counts['total_fish']
                    ) * route_day_counts['flow_converted']
                    route_day_counts.rename(columns={'passage_route': 'Passage Route'}, inplace=True)
                    discharge_records.append(route_day_counts[['Passage Route', 'route_count', 'total_fish', 'estimated_discharge']])
            try:
                del sim_data
            except Exception:
                pass

        if not combined_route_counts.empty:
            # Debug: log combined counts BEFORE computing summaries
            try:
                print(f"[ROUTE AGG DEBUG] combined_route_counts raw_sum={combined_route_counts.sum()}, entries={len(combined_route_counts)}", flush=True)
            except Exception:
                pass

            # Get number of iterations from daily_df or default to 1
            num_iterations = 1
            if daily_df is not None and 'iteration' in daily_df.columns:
                num_iterations = daily_df['iteration'].nunique()

            # Keep both total counts (sum across iterations) and mean per-iteration for reference
            total_counts = combined_route_counts.copy()
            try:
                mean_counts = total_counts / num_iterations
            except Exception:
                mean_counts = total_counts

            # Debug: log num_iterations and total entrained from Daily (if present)
            try:
                total_daily_entr = float(daily_df.get('num_entrained', pd.Series(dtype=float)).sum()) if daily_df is not None else 0.0
            except Exception:
                total_daily_entr = None
            print(f"[ROUTE AGG DEBUG] num_iterations={num_iterations}, total_entrained_from_Daily={total_daily_entr}", flush=True)
            try:
                print(f"[ROUTE AGG DEBUG] total_counts_sum={total_counts.sum()}, mean_counts_sum={mean_counts.sum()}, entries={len(total_counts)}", flush=True)
            except Exception:
                pass

            # Use totals for reporting (user expects simulation totals, not per-iteration means)
            combined_route_counts = total_counts.sort_values(ascending=False)

            # Calculate actual entrainment rate from route data (using totals)
            total_passage_fish = combined_route_counts.sum()
            turbine_routes = combined_route_counts[combined_route_counts.index.str.contains('U', case=True, na=False)]
            entrained_fish = turbine_routes.sum()
            actual_entr_rate = (entrained_fish / total_passage_fish) * 100 if total_passage_fish > 0 else 0
            
            # Add entrainment summary
            report_sections.append(f"""
            <div style="padding:15px; background:#fff3cd; border-left:4px solid #ffc107; border-radius:5px; margin:20px 0;">
                <p style="margin:0;"><strong>Entrainment Analysis from Route Data:</strong></p>
                <ul style="margin:5px 0 0 20px;">
                    <li>Total fish tracked through passage: <strong>{int(total_passage_fish):,}</strong></li>
                    <li>Fish entrained through turbines: <strong>{int(entrained_fish):,}</strong> ({actual_entr_rate:.1f}%)</li>
                    <li>Fish passing via other routes: <strong>{int(total_passage_fish - entrained_fish):,}</strong> ({100-actual_entr_rate:.1f}%)</li>
                </ul>
            </div>
            """)
            
            plt.rcParams.update({'font.size': 8})
            fig, ax1 = plt.subplots(figsize=(6, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(combined_route_counts)))
            ax1.pie(
                combined_route_counts.values,
                labels=combined_route_counts.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax1.set_title('Fish Passage Distribution by Route')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            pie_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Prepare table showing simulation TOTALS (not per-iteration means)
            counts_int = combined_route_counts.values.astype(int)
            counts_formatted = [f"{x:,}" for x in counts_int]
            route_table = pd.DataFrame({
                'Passage Route': combined_route_counts.index,
                # Display formatted totals with thousands separators for readability
                'Number of Fish': counts_formatted,
                'Percentage': (combined_route_counts.values / combined_route_counts.sum() * 100).round(1)
            })

            # Build survival by route table from accumulated counters
            try:
                if route_surv_count:
                    surv_tbl = pd.DataFrame({
                        'passage_route': list(route_surv_count.keys()),
                        'Total Fish': [int(route_surv_count[r]) for r in route_surv_count.keys()],
                        'Survived': [int(route_surv_sum.get(r, 0.0)) for r in route_surv_count.keys()],
                    })
                    surv_tbl['Survival %'] = (
                        (surv_tbl['Survived'] / surv_tbl['Total Fish']).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100
                    ).round(1)
                else:
                    surv_tbl = None
            except Exception as e:
                print(f"[ROUTE AGG DEBUG] survival aggregation failed: {e}", flush=True)
                surv_tbl = None

            # Append survival table to the report if available
            if surv_tbl is not None and not surv_tbl.empty:
                # Build HTML table for survival
                try:
                    surv_html = surv_tbl[['passage_route', 'Total Fish', 'Survived', 'Survival %']].to_html(index=False, border=1)
                    report_sections.append('<h3>Survival by Passage Route</h3>')
                    report_sections.append(surv_html)
                except Exception:
                    pass

            report_sections.append(
                "<div style='display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start;'>"
                "<div style='flex:1; min-width:300px; text-align:center;'>"
                f"<img src=\"data:image/png;base64,{pie_b64}\" style='max-width:100%; height:auto;' />"
                "</div>"
                "<div style='flex:1; min-width:320px;'>"
                "<h3>Route Usage Summary</h3>"
                f"{route_table.to_html(index=False, border=1)}"
                "<p style='font-size:0.9em; color:#666; margin-top:8px; font-style:italic;'>"
                "Table: Fish passage counts, entrainment events, and mortality statistics by route, showing how fish navigate through different facility pathways."
                "</p>"
                "</div>"
                "</div>"
            )
        else:
            report_sections.append("<p>Route usage data not available.</p>")

        discharge_heading = "Actual Discharge by Passage Route" if not need_estimated_flow else "Estimated Discharge by Passage Route"
        discharge_summary = None
        if route_flow_df is not None and {'route', 'discharge_cfs'}.issubset(route_flow_df.columns):
            processed_flows = route_flow_df.copy()
            processed_flows['route'] = processed_flows['route'].astype(str)
            processed_flows['day'] = pd.to_datetime(processed_flows['day'], errors='coerce')
            processed_flows['discharge_value'] = processed_flows['discharge_cfs'].astype(float)
            if units == 'metric':
                processed_flows['discharge_value'] = processed_flows['discharge_value'] * 0.0283168
            discharge_summary = processed_flows.groupby('route').agg(
                total_discharge=('discharge_value', 'sum'),
                mean_discharge=('discharge_value', 'mean'),
                median_discharge=('discharge_value', 'median'),
                days_observed=('discharge_value', 'count')  # Count total observations (iteration × day), not unique days
            ).reset_index().rename(columns={'route': 'Passage Route'})
        elif discharge_records:
            discharge_df = pd.concat(discharge_records, ignore_index=True)
            discharge_summary = discharge_df.groupby('Passage Route').agg(
                total_discharge=('estimated_discharge', 'sum'),
                mean_discharge=('estimated_discharge', 'mean'),
                median_discharge=('estimated_discharge', 'median'),
                days_observed=('estimated_discharge', 'count'),
                total_route_fish=('route_count', 'sum')
            ).reset_index()
        
        report_sections.append(f"<h3>{discharge_heading}</h3>")
        if discharge_summary is not None and not discharge_summary.empty:
            # FIXED: Filter out river nodes before display
            discharge_summary = discharge_summary[
                ~discharge_summary['Passage Route'].str.contains('river_node', case=False, na=False)
            ]
            
            # Map machine IDs to human-readable names
            nodes_df = store["/Nodes"] if "/Nodes" in store.keys() else None
            if nodes_df is not None and not nodes_df.empty:
                # Create mapping from Location (machine ID) to ID (human name)
                node_name_map = dict(zip(nodes_df['Location'], nodes_df['ID']))
                discharge_summary['Passage Route'] = discharge_summary['Passage Route'].map(
                    lambda x: node_name_map.get(x, x)  # Use mapping or keep original if not found
                )
            
            discharge_summary = discharge_summary.sort_values('total_discharge', ascending=False)
            total_fish_all_routes = combined_route_counts.sum()
            if total_fish_all_routes and total_fish_all_routes > 0:
                discharge_summary['Fish Share (%)'] = discharge_summary['Passage Route'].map(
                    lambda r: (combined_route_counts.get(r, 0.0) / total_fish_all_routes) * 100
                ).fillna(0).round(1)
            elif 'total_route_fish' in discharge_summary.columns:
                route_totals = discharge_summary['total_route_fish'].sum()
                discharge_summary['Fish Share (%)'] = discharge_summary['total_route_fish'].div(route_totals).fillna(0).mul(100).round(1)
            else:
                discharge_summary['Fish Share (%)'] = 0.0

            display_cols = [
                'Passage Route',
                'Fish Share (%)',
                f'Mean Discharge ({units_label})',
                f'Median Discharge ({units_label})',
                f'Total Discharge ({units_label}·day)',
                'Days Sampled'
            ]

            discharge_display = discharge_summary.rename(columns={
                'mean_discharge': f'Mean Discharge ({units_label})',
                'median_discharge': f'Median Discharge ({units_label})',
                'total_discharge': f'Total Discharge ({units_label}·day)',
                'days_observed': 'Days Sampled'
            })

            for col in [f'Mean Discharge ({units_label})', f'Median Discharge ({units_label})', f'Total Discharge ({units_label}·day)']:
                discharge_display[col] = discharge_display[col].map(lambda x: f"{x:,.2f}")
            discharge_display['Fish Share (%)'] = discharge_display['Fish Share (%)'].map(lambda x: f"{x:.1f}")
            if 'Days Sampled' in discharge_display.columns:
                discharge_display['Days Sampled'] = discharge_display['Days Sampled'].fillna(0).astype(int)

            report_sections.append(
                "<div style='display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start;'>"
                "<div style='flex:1; min-width:320px;'>"
                f"{discharge_display[display_cols].to_html(index=False, border=1)}"
                "<p style='font-size:0.9em; color:#666; margin-top:8px; font-style:italic;'>"
                "Table: Total discharge volume through each passage route over the simulation period, including mean daily discharge and number of days sampled."
                "</p>"
                "</div>"
            )

            top_routes = discharge_summary.head(10).reset_index(drop=True)
            plt.rcParams.update({'font.size': 8})
            fig, ax = plt.subplots(figsize=(7, max(3, len(top_routes) * 0.4)))
            ax.barh(top_routes['Passage Route'], top_routes['total_discharge'], color='#5a9bd4')
            ax.set_xlabel(f'Total Discharge ({units_label}·day)')
            ax.set_ylabel('Passage Route')
            ax.set_title('Top Routes by Discharge')
            ax.invert_yaxis()
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            discharge_bar_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            report_sections[-1] += (
                "<div style='flex:1; min-width:320px; text-align:center;'>"
                f"<img src=\"data:image/png;base64,{discharge_bar_b64}\" style='max-width:100%; height:auto;' />"
                "</div>"
                "</div>"
            )
        else:
            report_sections.append("<p>Insufficient data to report discharge by passage route.</p>")

        # Yearly summary panel (iteration-based)
        yearly_df = yearly_df if yearly_df is not None else (store["/Yearly_Summary"] if "/Yearly_Summary" in store.keys() else None)
        daily_df  = daily_df if daily_df is not None else (store["/Daily"] if "/Daily" in store.keys() else None)

        if daily_df is not None and not daily_df.empty:
            df = daily_df.copy()
            if 'num_mortality' not in df.columns and {'num_survived','pop_size'} <= set(df.columns):
                df['num_mortality'] = df['pop_size'] - df['num_survived']

            def compute_one_in_n_stats(values, ns=(10, 100, 1000)):
                """Compute literal 1-in-N day thresholds: value exceeded on average once every N days.
                Returns dict n -> threshold (float) and observed_count.
                """
                out = {}
                total_days = len(values)
                for n in ns:
                    if total_days == 0:
                        out[n] = {'threshold': 0.0, 'observed_days': 0, 'total_days': 0}
                        continue
                    q = 1.0 - 1.0 / float(n)
                    try:
                        thr = float(np.nanquantile(values, q))
                    except Exception:
                        thr = float(values.quantile(q)) if hasattr(values, 'quantile') else 0.0
                    observed = int((values >= thr).sum())
                    out[n] = {'threshold': thr, 'observed_days': observed, 'total_days': total_days}
                return out

            def create_daily_hist(data, col, title, one_in_n_stats=None):
                plt.rcParams.update({'font.size': 8})
                fig = plt.figure()
                if col in data.columns:
                    # Include zeros by filling NaN with 0 instead of dropping
                    values = data[col].fillna(0)
                    plt.hist(values, bins=20, edgecolor='black')
                    # Draw vertical lines for thresholds
                    if one_in_n_stats:
                        colors = {10: '#d95f02', 100: '#1b9e77', 1000: '#7570b3'}
                        for n, info in one_in_n_stats.items():
                            thr = info.get('threshold', 0.0)
                            if thr is None:
                                continue
                            plt.axvline(thr, color=colors.get(n, 'black'), linestyle='--', linewidth=1.2, label=f'1-in-{n}: {thr:.0f}')
                        plt.legend(fontsize=8)
                plt.xlabel(col.replace('_', ' ').title())
                plt.ylabel("Frequency")
                plt.title(title)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                return base64.b64encode(buf.getvalue()).decode('utf-8')

            # Precompute one-in-N stats for entrainment and mortality
            entr_stats = None
            mort_stats = None
            if 'num_entrained' in df.columns:
                values = df['num_entrained'].fillna(0)
                entr_stats = compute_one_in_n_stats(values, ns=(10, 100, 1000))
            if 'num_mortality' in df.columns:
                values_m = df['num_mortality'].fillna(0)
                mort_stats = compute_one_in_n_stats(values_m, ns=(10, 100, 1000))

            entr_img = create_daily_hist(df, 'num_entrained', 'Daily Entrainment Distribution', one_in_n_stats=entr_stats) if 'num_entrained' in df.columns else None
            mort_img = create_daily_hist(df, 'num_mortality', 'Daily Mortality Distribution', one_in_n_stats=mort_stats) if 'num_mortality' in df.columns else None

            # Build a two-column panel: left = entrainment histogram & summary; right = mortality histogram & summary
            left_col = []
            right_col = []

            entr_html = (
                f'<img src="data:image/png;base64,{entr_img}" style="max-width:100%; height:auto;" />'
                if entr_img else "<p>No 'num_entrained' data available.</p>"
            )
            left_col.append("<div style='flex:1; min-width:300px; text-align:center;'>")
            left_col.append("<h3>Daily Entrainment</h3>")
            left_col.append(entr_html)
            left_col.append("<p style='font-size:0.9em; color:#666; margin-top:8px; font-style:italic;'>Figure 5: Daily distribution of fish entrainment events showing frequency of different entrainment magnitudes.</p>")

            # One-in-N summary for entrainment
            if entr_stats:
                entr_summary_lines = []
                for n, info in entr_stats.items():
                    thr = info['threshold']
                    observed = info['observed_days']
                    total = info['total_days']
                    observed_str = f"(observed {observed} days → ~1 in {int(total/observed) if observed>0 else '∞'} days)" if observed>0 else "(observed 0 days)"
                    entr_summary_lines.append(f"1-in-{n} day event: {int(thr)} fish {observed_str}")
                entr_summary_html = '<br/>'.join(entr_summary_lines)
                left_col.append(f"<div style='text-align:left; padding:10px; background:#eef7ff; border-radius:6px; margin:8px;'>{entr_summary_html}</div>")

            left_col.append("</div>")

            # Mortality column
            if mort_img:
                right_col.append("<div style='flex:1; min-width:300px; text-align:center;'>")
                right_col.append("<h3>Daily Mortality</h3>")
                right_col.append(f"<img src='data:image/png;base64,{mort_img}' style='max-width:100%; height:auto;' />")
                right_col.append("<p style='font-size:0.9em; color:#666; margin-top:8px; font-style:italic;'>Figure 6: Daily distribution of fish mortalities showing frequency of different mortality counts per day.</p>")
                if mort_stats:
                    mort_summary_lines = []
                    for n, info in mort_stats.items():
                        thr = info['threshold']
                        observed = info['observed_days']
                        total = info['total_days']
                        observed_str = f"(observed {observed} days → ~1 in {int(total/observed) if observed>0 else '∞'} days)" if observed>0 else "(observed 0 days)"
                        mort_summary_lines.append(f"1-in-{n} day mortality: {int(thr)} fish {observed_str}")
                    mort_summary_html = '<br/>'.join(mort_summary_lines)
                    right_col.append(f"<div style='text-align:left; padding:10px; background:#fff0f0; border-radius:6px; margin:8px;'>{mort_summary_html}</div>")
                right_col.append("</div>")

            # Append combined panel
            combined_html = "<div style='display:flex; gap:20px; justify-content:center; flex-wrap:wrap'>" + "".join(left_col) + "".join(right_col) + "</div>"
            report_sections.append(combined_html)
        else:
            report_sections.append("<p>No daily data available.</p>")

        # --- Annual summary: Mean annual entrainment & mortality with 95% CI ---
        annual_summary_html = None
        try:
            # Prefer Yearly_Summary written by the simulation (single-row summary)
            if yearly_df is not None and not yearly_df.empty:
                row = yearly_df.iloc[0]
                entr_mean = row.get('mean_yearly_entrainment', None)
                entr_lcl = row.get('lcl_yearly_entrainment', None)
                entr_ucl = row.get('ucl_yearly_entrainment', None)
                mort_mean = row.get('mean_yearly_mortality', None)
                mort_lcl = row.get('lcl_yearly_mortality', None)
                mort_ucl = row.get('ucl_yearly_mortality', None)
                if any(v is not None for v in (entr_mean, entr_lcl, entr_ucl, mort_mean, mort_lcl, mort_ucl)):
                    def fmt(v):
                        try:
                            return f"{int(v):,}"
                        except Exception:
                            return 'N/A'
                    annual_summary_html = f"<div style='display:flex; gap:20px; flex-wrap:wrap; margin:10px 0;'>"
                    annual_summary_html += f"<div style='flex:1; min-width:220px; padding:10px; background:#f0f8ff; border-radius:6px;'><strong>Mean Annual Entrainment</strong><br/>{fmt(entr_mean)}<br/><small style='color:#666;'>95% CI: {fmt(entr_lcl)} – {fmt(entr_ucl)}</small></div>"
                    annual_summary_html += f"<div style='flex:1; min-width:220px; padding:10px; background:#fff6f6; border-radius:6px;'><strong>Mean Annual Mortality</strong><br/>{fmt(mort_mean)}<br/><small style='color:#666;'>95% CI: {fmt(mort_lcl)} – {fmt(mort_ucl)}</small></div>"
                    annual_summary_html += "</div>"
            # Fallback: compute per-iteration sums from daily table if available
            if annual_summary_html is None and daily_df is not None and not daily_df.empty:
                if 'iteration' in daily_df.columns:
                    try:
                        by_iter = daily_df.groupby('iteration').agg({'num_entrained':'sum','num_mortality':'sum'})
                        # compute mean and empirical 95% interval
                        entr_vals = by_iter['num_entrained'].values
                        mort_vals = by_iter['num_mortality'].values if 'num_mortality' in by_iter.columns else None
                        def emp_stats(arr):
                            a = np.array(arr, dtype=float)
                            mean = np.nanmean(a)
                            lcl = np.nanpercentile(a, 2.5) if a.size>0 else np.nan
                            ucl = np.nanpercentile(a, 97.5) if a.size>0 else np.nan
                            return mean, lcl, ucl
                        em, el, eu = emp_stats(entr_vals)
                        if mort_vals is not None:
                            mm, ml, mu = emp_stats(mort_vals)
                        else:
                            mm = ml = mu = np.nan
                        def fmt2(v):
                            try:
                                return f"{int(round(float(v))):,}"
                            except Exception:
                                return 'N/A'
                        annual_summary_html = f"<div style='display:flex; gap:20px; flex-wrap:wrap; margin:10px 0;'>"
                        annual_summary_html += f"<div style='flex:1; min-width:220px; padding:10px; background:#f0f8ff; border-radius:6px;'><strong>Mean Annual Entrainment</strong><br/>{fmt2(em)}<br/><small style='color:#666;'>95% CI: {fmt2(el)} – {fmt2(eu)}</small></div>"
                        annual_summary_html += f"<div style='flex:1; min-width:220px; padding:10px; background:#fff6f6; border-radius:6px;'><strong>Mean Annual Mortality</strong><br/>{fmt2(mm)}<br/><small style='color:#666;'>95% CI: {fmt2(ml)} – {fmt2(mu)}</small></div>"
                        annual_summary_html += "</div>"
                    except Exception:
                        annual_summary_html = None
        except Exception:
            annual_summary_html = None

        if annual_summary_html:
            # Insert minimal, unstyled annual summary lines per request
            # Format: "Average Annual Entrapment: YYY (XXX - ZZZ)"
            try:
                # crude parsing of the html created above to extract numbers (we built it),
                # but safer to reconstruct from available numeric variables if present.
                # Prefer Yearly_Summary values
                if yearly_df is not None and not yearly_df.empty:
                    row = yearly_df.iloc[0]
                    e_mean = row.get('mean_yearly_entrainment', None)
                    e_lcl = row.get('lcl_yearly_entrainment', None)
                    e_ucl = row.get('ucl_yearly_entrainment', None)
                    m_mean = row.get('mean_yearly_mortality', None)
                    m_lcl = row.get('lcl_yearly_mortality', None)
                    m_ucl = row.get('ucl_yearly_mortality', None)
                else:
                    # fallback values used when we computed stats from iterations
                    e_mean = em if 'em' in locals() else None
                    e_lcl = el if 'el' in locals() else None
                    e_ucl = eu if 'eu' in locals() else None
                    m_mean = mm if 'mm' in locals() else None
                    m_lcl = ml if 'ml' in locals() else None
                    m_ucl = mu if 'mu' in locals() else None

                def s(v):
                    try:
                        return f"{int(round(float(v))):,}"
                    except Exception:
                        return 'N/A'

                report_sections.append("<h2>Annual Summary</h2>")
                report_sections.append(f"<p>Average Annual Entrainment: {s(e_mean)} ({s(e_lcl)} - {s(e_ucl)})</p>")
                report_sections.append(f"<p>Average Annual Mortality: {s(m_mean)} ({s(m_lcl)} - {s(m_ucl)})</p>")
            except Exception:
                # In case of unexpected errors, skip adding the block
                pass

        # Time Series Plots - Daily Entrainment with Error Bars
        log.info("Report: daily average entrainment time series")
        report_sections.append("<h2>Daily Average Entrainment Over Time</h2>")
        if daily_df is not None and not daily_df.empty and 'day' in daily_df.columns:
            df_ts = daily_df.copy()
            
            # Calculate daily statistics with error bars
            daily_stats = df_ts.groupby('day').agg({
                'num_entrained': ['mean', 'std', 'count'],
                'flow': 'mean'
            }).reset_index()
            
            # Flatten column names
            daily_stats.columns = ['day', 'entr_mean', 'entr_std', 'entr_count', 'flow_mean']
            daily_stats['day'] = pd.to_datetime(daily_stats['day'], errors='coerce')
            daily_stats.sort_values('day', inplace=True)
            
            # Calculate standard error for error bars
            daily_stats['entr_se'] = daily_stats['entr_std'] / np.sqrt(daily_stats['entr_count'])
            daily_stats['entr_se'] = daily_stats['entr_se'].fillna(0)

            def create_entrainment_timeseries_with_error(data):
                plt.rcParams.update({'font.size': 8})
                fig, ax1 = plt.subplots(figsize=(10, 5))
                
                # Plot entrainment with error bars
                ax1.errorbar(data['day'], data['entr_mean'], yerr=data['entr_se'], 
                           color='#ff8c00', marker='o', linestyle='-', linewidth=1.5, 
                           markersize=3, capsize=3, alpha=0.8)
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Number Entrained', color='#ff8c00')
                ax1.tick_params(axis='y', labelcolor='#ff8c00')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)

                # Add flow on secondary axis
                if 'flow_mean' in data.columns:
                    ax2 = ax1.twinx()
                    ax2.plot(data['day'], data['flow_mean'], color='gray', alpha=0.4, linewidth=1)
                    ax2.set_ylabel('Flow', color='gray')
                    ax2.tick_params(axis='y', labelcolor='gray')

                plt.title('Daily Average Entrainment with Standard Error')
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                return base64.b64encode(buf.getvalue()).decode('utf-8')

            if 'num_entrained' in df_ts.columns and not daily_stats.empty:
                ts_entr = create_entrainment_timeseries_with_error(daily_stats)
                report_sections.append(
                    "<div style='text-align:center;'>"
                    f"<img src=\"data:image/png;base64,{ts_entr}\" style='max-width:100%; height:auto;' />"
                    "<p style='font-size:0.9em; color:#666; margin-top:8px; font-style:italic;'>"
                    "Figure 7: Daily average entrainment over time with standard error bars (orange) and overlaid flow rates (gray). Shows temporal patterns and variability in fish entrainment."
                    "</p>"
                    "</div>"
                )
            else:
                report_sections.append("<p>Entrainment time series data not available.</p>")
        else:
            report_sections.append("<p>Time series data not available.</p>")

        # Finalize HTML
        log.info("Report: final HTML assembly")
        final_html = "\n".join(report_sections)
        full_report = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Simulation Report</title>
<style>
 body {{ font-family: Arial, sans-serif; background:#f7f7f7; margin:20px; color:#333; }}
 .container {{ max-width: 1800px; margin:auto; background:#fff; padding:20px; border-radius:8px; box-shadow:0 0 10px rgba(0,0,0,.1); }}
 h1 {{ color:#0056b3; }} h2 {{ color:#0056b3; border-bottom:1px solid #ddd; padding-bottom:4px; margin-top:2rem; }}
 h3 {{ color:#0056b3; margin-top:1.5rem; }} p {{ line-height:1.6; }}
 table {{ width:100%; border-collapse:collapse; margin:1rem 0; }}
 th,td {{ padding:8px; border:1px solid #ccc; text-align:left; }}
 .download-link {{ display:inline-block; margin-top:20px; background:#007BFF; color:white; padding:10px 15px; text-decoration:none; border-radius:4px; }}
 .download-link:hover {{ background:#0056b3; }}
</style></head><body><div class="container">
{final_html}
<a href="/download_report_zip" class="download-link">Download Report</a>
</div></body></html>"""
        return full_report

    finally:
        try:
            if store is not None:
                store.close()
        except Exception:
            pass



@app.route('/download_report')
def download_report():
    proj_dir = _get_active_run_dir()
    if not proj_dir:
        return "<p>Error: No project directory in session.</p>", 400

    marker = os.path.join(proj_dir, 'report_path.txt')
    report_path = open(marker, 'r', encoding='utf-8').read().strip() if os.path.exists(marker) \
                  else os.path.join(proj_dir, 'simulation_report.html')

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
    if not app.config.get("DEBUG_ROUTES_ENABLED"):
        return "Not Found", 404
    lines = []
    for rule in app.url_map.iter_rules():
        methods = ",".join(sorted(m for m in rule.methods if m in ("GET","POST","PUT","DELETE","PATCH")))
        lines.append(f"{rule.endpoint:30s}  {methods:10s}  {rule}")
    return "<pre>" + "\n".join(sorted(lines)) + "</pre>"

@app.route('/download_report_zip')
def download_report_zip():
    proj_dir = _get_active_run_dir()
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
