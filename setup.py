# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:22:05 2025

@author: Kevin.Nebiolo
"""

# setup.py
from setuptools import setup, find_packages

setup(
    name='Stryke',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your runtime dependencies here, e.g.:
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'xlrd',
        'networkx',
        'statsmodels',
        'scipy',
        'h5py'
    ],
)
