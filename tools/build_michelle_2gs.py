"""Rebuild the "2GS YOY" single-Francis-unit project (Michelle's client report,
generated 2026-07-14) from the numbers published in her HTML report, so it can
be re-run headlessly against the current engine.

Her HTML report shows unit parameters already converted to metric for display;
webapp_import expects native (imperial) units, so every length/velocity/flow
value below is the metric report value multiplied back to feet/cfs. Facility
Rack Spacing and the population U_crit field are shown unconverted in her
report (native ft / ft-s), so those are used as-is.

Usage:
    python tools/build_michelle_2gs.py --out DIR [--iterations N] [--unit-D-ft D]
"""
import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

M_TO_FT = 3.28084
CMS_TO_CFS = 35.31469989


def build(out_root, iterations=50, unit_d_ft=None, unit_b_ft=None):
    run_dir = out_root
    os.makedirs(run_dir, exist_ok=True)

    # Metric values straight from the "Unit Parameters" table in her report.
    unit_row = {
        'Facility': '2GS', 'Unit': 1, 'Penstock_ID': 1,
        'Penstock_Qcap': 10.999992 * CMS_TO_CFS,
        'Runner Type': 'Francis',
        'intake_vel': 0.33 * M_TO_FT,
        'op_order': 1,
        'ps_D': 4.3 * M_TO_FT, 'ps_length': 10.0 * M_TO_FT,
        'fb_depth': 5.0 * M_TO_FT, 'submergence_depth': 5.0 * M_TO_FT,
        'roughness': 0.025,
        'H': 10.23 * M_TO_FT, 'RPM': 240,
        'D': (unit_d_ft if unit_d_ft is not None else 1.0 * M_TO_FT),
        'ada': 0, 'N': 15,
        'Qopt': 8.999994 * CMS_TO_CFS, 'Qcap': 10.999992 * CMS_TO_CFS,
        'B': (unit_b_ft if unit_b_ft is not None else 1.08 * M_TO_FT),
        'iota': 1.1,
        'D1': 1.02 * M_TO_FT, 'D2': 1.6 * M_TO_FT, 'lambda': 0.2,
    }
    units = pd.DataFrame([unit_row])
    unit_path = os.path.join(run_dir, 'unit_params.csv')
    units.to_csv(unit_path, index=False)

    # Facility Parameters table in her report is shown unconverted (native feet).
    rack_spacing_ft = 0.098425
    qcap_cfs = unit_row['Qcap']
    facilities = [{
        'Facility': '2GS', 'Operations': 'run-of-river', 'Rack Spacing': rack_spacing_ft,
        'Min_Op_Flow': 4.999996, 'Env_Flow': 0.0, 'Bypass_Flow': 0.0,
        'Spillway': 'none', 'Units': 1,
    }]

    ops = pd.DataFrame([{
        'Scenario': 'fall', 'Facility': '2GS', 'Unit': 1.0, 'Hours': 24.0,
        'Prob Not Operating': None, 'Shape': None, 'Location': None, 'Scale': None,
    }])
    ops_path = os.path.join(run_dir, 'operating_scenarios.csv')
    ops.to_csv(ops_path, index=False)

    # Always at Qcap -> static flow scenario (her report: "No hydrograph data available").
    flow_scenarios = [{
        'Scenario': 'fall', 'Scenario Number': 1, 'Season': 'fall', 'Months': '9,10,11',
        'Flow': qcap_cfs, 'Gage': None, 'FlowYear': None, 'Prorate': 1,
    }]

    # Population row: Walleye, "Sander, Great Lakes, Met Fall & Winter" preset
    # (current app.py values), U_crit shown unconverted (native ft/s) in her
    # species table as 1.08.
    population = [{
        'Species': 'Sander vitreus', 'Common Name': 'Walleye', 'Scenario': 'fall',
        'Iterations': iterations, 'Fish': None,
        'Simulate Choice': 'entrainment event', 'Entrainment Choice': 'modeled',
        'Modeled Species': 'Sander, Great Lakes, Met Fall & Winter',
        'vertical_habitat': 'Pelagic', 'beta_0': -4.8085, 'beta_1': 3.33,
        'fish_type': 'physoclistous',
        'dist': 'Pareto', 'shape': 0.3980874898, 'location': 0.0, 'scale': 0.002010039,
        'max_ent_rate': 0.85, 'occur_prob': 0.451, 'Length_mean': None, 'Length_sd': None,
        'U_crit': 0.33 * M_TO_FT,
        'length shape': 0.4177, 'length location': -1.0242, 'length scale': 16.4924,
    }]

    graph_summary = {
        'Nodes': [
            {'ID': 'river_node_0', 'Location': 'river_node_0', 'Surv_Fun': 'a priori', 'Survival': 1.0},
            {'ID': '2GS - Unit 1', 'Location': '2GS - Unit 1', 'Surv_Fun': 'Francis', 'Survival': 0.0},
            {'ID': 'river_node_1', 'Location': 'river_node_1', 'Surv_Fun': 'a priori', 'Survival': 1.0},
        ],
        'Edges': [
            {'_from': 'river_node_0', '_to': '2GS - Unit 1', 'weight': 1.0},
            {'_from': '2GS - Unit 1', '_to': 'river_node_1', 'weight': 1.0},
        ],
    }
    import networkx as nx
    from networkx.readwrite import json_graph
    graph = nx.DiGraph()
    for node in graph_summary['Nodes']:
        graph.add_node(node['Location'], ID=node['ID'], Location=node['Location'],
                        Surv_Fun=node['Surv_Fun'], Survival=node['Survival'])
    for edge in graph_summary['Edges']:
        graph.add_edge(edge['_from'], edge['_to'], weight=edge['weight'])
    graph_data = json_graph.node_link_data(graph)

    return {
        'proj_dir': run_dir, 'project_name': '2GS YOY', 'project_notes': 'Rebuilt for engine QC',
        'model_setup': 'single_unit_survival_only', 'units': 'metric', 'facilities': facilities,
        'unit_parameters_file': unit_path, 'operating_scenarios_file': ops_path, 'population': population,
        'flow_scenarios': flow_scenarios, 'graph_data': graph_data, 'graph_summary': graph_summary,
        'units_system': 'metric', 'simulation_mode': 'single_unit_survival_only', 'output_name': 'michelle_2gs',
        'hydrograph_file': None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--unit-D-ft', type=float, dest='unit_d_ft')
    parser.add_argument('--unit-B-ft', type=float, dest='unit_b_ft')
    args = parser.parse_args()

    data = build(args.out, args.iterations, args.unit_d_ft, args.unit_b_ft)
    import Stryke.stryke as stryke
    os.environ['STRYKE_RANDOM_SEED'] = str(args.seed)
    sim = stryke.simulation(data['proj_dir'], data['output_name'], existing=False)
    sim.webapp_import(data, data['output_name'])
    sim.run()
    sim.summary()
    with pd.HDFStore(os.path.join(data['proj_dir'], f"{data['output_name']}.h5"), mode='r') as store:
        daily = store['Daily']
    totals = daily[['num_entrained', 'num_survived', 'mortality_impingement',
                     'mortality_blade_strike', 'mortality_barotrauma']].sum()
    entrained = totals['num_entrained']
    mort = entrained - totals['num_survived']
    iters = daily['iteration'].nunique()
    result = pd.DataFrame([{
        'iterations': iters,
        'mean_entrained': entrained / iters, 'mean_mortalities': mort / iters,
        'mortality_rate_of_entrained': mort / entrained if entrained else float('nan'),
        'impingement': totals['mortality_impingement'] / iters,
        'blade_strike': totals['mortality_blade_strike'] / iters,
        'barotrauma': totals['mortality_barotrauma'] / iters,
        'escaped': (daily['num_escaped'].sum() / iters) if 'num_escaped' in daily.columns else float('nan'),
    }])
    result.to_csv(os.path.join(data['proj_dir'], 'result.csv'), index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
