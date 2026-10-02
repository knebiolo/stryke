"""Run a saved .stryke project headlessly through the web app's simulation pipeline.

Mirrors the web app path (simulation.webapp_import -> run -> summary) using the
inputs embedded in the project file, with optional overrides for sensitivity
checks. Prints entrained / mortality totals and the cause breakdown.

Usage:
    python tools/run_stryke_project.py PROJECT.stryke --out DIR [--iterations N] [--seed S]
        [--preset-version corrected|flawed|original] [--unit-B-ft 1.64] [--label name]
"""
import argparse
import io
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from kakabeka_seasonal_ab import VERSIONS, graph_inputs, load_presets  # noqa: E402

PRESET_FIELDS = ('dist', 'shape', 'location', 'scale', 'max_ent_rate', 'occur_prob',
                 'length shape', 'length location', 'length scale')


def build(project_path, out_root, label, iterations, preset_version, unit_b_ft):
    with open(project_path, encoding='utf-8') as handle:
        project = json.load(handle)
    run_dir = os.path.join(out_root, label)
    os.makedirs(run_dir, exist_ok=True)

    units = pd.read_csv(io.StringIO(project['unit_parameters']['csv_content']))
    if unit_b_ft is not None:
        units['B'] = unit_b_ft
    unit_path = os.path.join(run_dir, 'unit_params.csv')
    units.to_csv(unit_path, index=False)

    ops = pd.DataFrame(project['operating_scenarios']).dropna(subset=['Scenario', 'Facility'], how='all')
    ops_path = os.path.join(run_dir, 'operating_scenarios.csv')
    ops.to_csv(ops_path, index=False)

    hydro_path = os.path.join(run_dir, 'hydrograph.csv')
    pd.DataFrame(project['hydrograph'])[['datetimeUTC', 'DAvgFlow_prorate']].to_csv(hydro_path, index=False)

    population = [dict(row) for row in project['population']]
    if len(population) != 1:
        raise ValueError(f'{project_path}: engine supports one population row per run, found {len(population)}')
    for row in population:
        if iterations is not None:
            row['Iterations'] = iterations
        if preset_version is not None:
            presets = load_presets(VERSIONS[preset_version])
            name = str(row['Modeled Species']).strip()
            if name not in presets:
                raise KeyError(f'Preset {name!r} not found in {preset_version} species_defaults')
            for field in PRESET_FIELDS:
                value = presets[name][field]
                row[field] = value if field == 'dist' else float(value)

    graph_summary, graph_data = graph_inputs(project)
    info = project['project_info'][0]
    return {
        'proj_dir': run_dir, 'project_name': label, 'project_notes': f'Headless run of {os.path.basename(project_path)}',
        'model_setup': info['Model Setup'], 'units': info['Units'], 'facilities': project['facilities'],
        'unit_parameters_file': unit_path, 'operating_scenarios_file': ops_path, 'population': population,
        'flow_scenarios': project['flow_scenarios'], 'graph_data': graph_data, 'graph_summary': graph_summary,
        'units_system': info['Units'], 'simulation_mode': info['Model Setup'], 'output_name': label,
        'hydrograph_file': hydro_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project')
    parser.add_argument('--out', required=True)
    parser.add_argument('--label', default='asis')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--preset-version', choices=sorted(VERSIONS))
    parser.add_argument('--unit-B-ft', type=float, dest='unit_b_ft')
    args = parser.parse_args()

    data = build(args.project, args.out, args.label, args.iterations, args.preset_version, args.unit_b_ft)
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
        'label': args.label, 'iterations': iters,
        'mean_entrained': entrained / iters, 'mean_mortalities': mort / iters,
        'mortality_rate': mort / entrained if entrained else float('nan'),
        'impingement': totals['mortality_impingement'] / iters,
        'blade_strike': totals['mortality_blade_strike'] / iters,
        'barotrauma': totals['mortality_barotrauma'] / iters,
        'escaped': (daily['num_escaped'].sum() / iters) if 'num_escaped' in daily.columns else float('nan'),
    }])
    result.to_csv(os.path.join(data['proj_dir'], 'result.csv'), index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
