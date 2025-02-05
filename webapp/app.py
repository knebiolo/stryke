# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:48:03 2025

@author: Kevin.Nebiolo
"""

import sys
# Add your local repository path to sys.path
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\stryke")

from flask import Flask, render_template
from Stryke import stryke  # Now this works because Python can find it

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, Stryke Web App!"

if __name__ == '__main__':
    app.run(debug=True)
