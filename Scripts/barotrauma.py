# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:36:10 2024

@author: Kevin.Nebiolo

script intent: develop a function to calculate pressure ratio 
"""

def calculate_inlet_pressure(
    h1, h2, Q, A_penstock, A_draft_tube, h_loss, submergence_depth, 
    g=9.81, rho=1000, atm_pressure=101325
):
    """
    Calculates the inlet pressure (P1) and pressure ratio (P1/P2) for a hydraulic system.
    Adjusts velocities based on discharge (Q) and cross-sectional areas (A).
    
    Parameters:
    - h1, h2: Elevation at inlet and outlet (meters).
    - Q: Discharge (m³/s).
    - A_penstock, A_draft_tube: Cross-sectional areas of penstock and draft tube (m²).
    - h_loss: Head losses (meters).
    - submergence_depth: Submergence at draft tube outlet (meters).
    - g: Acceleration due to gravity, default 9.81 m/s².
    - rho: Water density, default 1000 kg/m³.
    - atm_pressure: Atmospheric pressure, default 101,325 Pa.

    Returns:
    - dict: Includes inlet pressure (P1), outlet pressure (P2), and pressure ratio (P1/P2).
    """
    # Calculate velocities
    v1 = Q / A_penstock  # Velocity at the penstock inlet
    v2 = Q / A_draft_tube  # Velocity at the draft tube outlet

    # Calculate downstream pressure (P2)
    P2 = atm_pressure + rho * g * submergence_depth

    # Velocity-derived terms
    velocity_term = 0.5 * rho * (v1**2 - v2**2)

    # Gravitational potential energy (elevation term)
    elevation_term = rho * g * (h1 - h2)

    # Head loss contribution
    head_loss_term = rho * g * h_loss

    # Calculate inlet pressure (P1)
    P1 = P2 + velocity_term + elevation_term + head_loss_term

    # Calculate pressure ratio (P1/P2)
    pressure_ratio = P1 / P2

    return {
        "inlet_pressure": P1,
        "outlet_pressure": P2,
        "pressure_ratio": pressure_ratio,
        "velocity_inlet": v1,
        "velocity_outlet": v2
    }

h1 = 1500
h2 = 1470 
Q = 400  
A_penstock = 5 
A_draft_tube = 6
h_loss = 5 
submergence_depth = 3
g=9.81
rho=1000
atm_pressure=101325

pressures = calculate_inlet_pressure(h1, h2, Q, A_penstock, A_draft_tube, h_loss, submergence_depth)

# Calculate velocities
v1 = Q / A_penstock  # Velocity at the penstock inlet
v2 = Q / A_draft_tube  # Velocity at the draft tube outlet

# Calculate downstream pressure (P2)
P2 = atm_pressure + rho * g * submergence_depth

# Velocity-derived terms
velocity_term = 0.5 * rho * (v1**2 - v2**2)

# Gravitational potential energy (elevation term)
elevation_term = rho * g * (h1 - h2)

# Head loss contribution
head_loss_term = rho * g * h_loss

# Calculate inlet pressure (P1)
P1 = P2 + velocity_term + elevation_term + head_loss_term

# Calculate pressure ratio (P1/P2)
pressure_ratio = P1 / P2
