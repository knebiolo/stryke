"""Seasonal A/B/C check of Great Lakes entrainment presets on Kakabeka.

Runs the web app's simulation pipeline (simulation.webapp_import -> run -> summary)
headlessly for Walleye (Sander) and White Sucker (Catostomus) across four
meteorological-season scenarios, once per preset version:

- original : species_defaults at commit 5799c70 (before the deduplication refit)
- flawed   : species_defaults at HEAD (genus-wide, all-season refit)
- corrected: species_defaults in the working tree (seasonal, Great Lakes refit)

Facility, unit, graph, operating, and hydrograph inputs come from saved
Kakabeka .stryke projects and are identical across versions. Jan-Feb 2022
hydrograph days are relabelled to 2021 so the Winter scenario gets a full
Dec-Feb season from a single FlowYear.

Usage:
    python tools/kakabeka_seasonal_ab.py --iterations 20 --seed 20260930
"""
import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DOWNLOADS = os.path.join(os.path.expanduser('~'), 'Downloads')
EXISTING_TEMPLATE = os.path.join(DOWNLOADS, 'Kakabeka-OLD-NP (1).stryke')
NEW_TEMPLATE = os.path.join(DOWNLOADS, 'Kakabeka_New (4).stryke')
HYDROGRAPH_SOURCE = os.path.join(DOWNLOADS, 'KakabekaExisting.stryke')

SEASONS = {
    'Winter': '12,1,2',
    'Spring': '3,4,5',
    'Summer': '6,7,8',
    'Fall': '9,10,11',
}
SPECIES = {
    'Walleye': {
        'Species': 'Sander',
        'presets': {
            'Winter': 'Sander, Great Lakes, Met Fall & Winter',
            'Spring': 'Sander, Great Lakes, Met Spring & Summer',
            'Summer': 'Sander, Great Lakes, Met Spring & Summer',
            'Fall': 'Sander, Great Lakes, Met Fall & Winter',
        },
        'fish_type': 'physoclistous',
        'beta_0': -4.8085,
        'beta_1': 3.33,
        'vertical_habitat': 'Pelagic',
    },
    'White Sucker': {
        'Species': 'Catostomus',
        'presets': {season: f'Catostomus, Great Lakes, Met {season}' for season in SEASONS},
        'fish_type': 'physostomous',
        'beta_0': -4.93263,
        'beta_1': 2.96225,
        'vertical_habitat': 'Benthic',
    },
}
VERSIONS = {'original': '5799c70', 'flawed': 'HEAD', 'corrected': None}
GIT = r'C:\Users\Kevin.Nebiolo\AppData\Local\Programs\Git\bin\git.exe'


def _parse_species_defaults(text):
    match = re.search(r'species_defaults\s*=\s*\[', text)
    if match is None:
        raise RuntimeError('species_defaults not found in app.py source')
    start = text.index('[', match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == '[':
            depth += 1
        elif text[index] == ']':
            depth -= 1
            if depth == 0:
                return ast.literal_eval(text[start:index + 1])
    raise RuntimeError('Unterminated species_defaults literal')


def load_presets(revision):
    if revision is None:
        with open(os.path.join(PROJECT_ROOT, 'webapp', 'app.py'), encoding='utf-8') as handle:
            text = handle.read()
    else:
        text = subprocess.run(
            [GIT, 'show', f'{revision}:webapp/app.py'],
            cwd=PROJECT_ROOT, capture_output=True, text=True, encoding='utf-8', check=True,
        ).stdout
    return {entry['name'].strip(): entry for entry in _parse_species_defaults(text)}


def _load_project(path):
    with open(path, encoding='utf-8') as handle:
        return json.loads(handle.read())


def full_year_hydrograph():
    hydro = pd.DataFrame(_load_project(HYDROGRAPH_SOURCE)['hydrograph'])
    dates = pd.to_datetime(hydro['datetimeUTC'])
    if len(hydro) != 365 or dates.min() != pd.Timestamp('2021-03-01') or dates.max() != pd.Timestamp('2022-02-28'):
        raise ValueError(f'Unexpected hydrograph span in {HYDROGRAPH_SOURCE}: {dates.min()}..{dates.max()} ({len(hydro)} days)')
    dates = dates.where(dates.dt.year == 2021, dates - pd.DateOffset(years=1))
    hydro['datetimeUTC'] = dates.dt.strftime('%Y-%m-%d')
    return hydro[['datetimeUTC', 'DAvgFlow_prorate']]


def graph_inputs(project):
    elements = project['graph']['elements']
    nodes = [
        {'ID': n['data'].get('label', n['data']['id']), 'Location': n['data']['id'],
         'Surv_Fun': n['data'].get('surv_fun', 'default'), 'Survival': n['data'].get('survival_rate')}
        for n in elements['nodes']
    ]
    edges = [
        {'_from': e['data']['source'], '_to': e['data']['target'], 'weight': float(e['data'].get('weight', 1.0))}
        for e in elements['edges']
    ]
    import networkx as nx
    from networkx.readwrite import json_graph
    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(node['Location'], ID=node['ID'], Location=node['Location'],
                       Surv_Fun=node['Surv_Fun'], Survival=node['Survival'])
    for edge in edges:
        graph.add_edge(edge['_from'], edge['_to'], weight=edge['weight'])
    return {'Nodes': nodes, 'Edges': edges}, json_graph.node_link_data(graph)


def population_rows(presets, iterations, u_crit, common_name, season):
    # The barotrauma path reads self.pop[...].item(), so each run holds exactly one population row.
    rows = []
    spec = SPECIES[common_name]
    for preset_season, preset_name in spec['presets'].items():
        if preset_season != season:
            continue
        preset = presets.get(preset_name)
        if preset is None:
            raise KeyError(f'Preset {preset_name!r} missing from species_defaults')
        rows.append({
                'Species': spec['Species'], 'Common Name': common_name, 'Scenario': season,
                'Iterations': iterations, 'Fish': None,
                'Simulate Choice': 'entrainment event', 'Entrainment Choice': 'modeled',
                'Modeled Species': preset_name, 'vertical_habitat': spec['vertical_habitat'],
                'beta_0': spec['beta_0'], 'beta_1': spec['beta_1'], 'fish_type': spec['fish_type'],
                'dist': preset['dist'], 'shape': float(preset['shape']), 'location': float(preset['location']),
                'scale': float(preset['scale']), 'max_ent_rate': float(preset['max_ent_rate']),
                'occur_prob': float(preset['occur_prob']), 'Length_mean': None, 'Length_sd': None,
                'U_crit': u_crit, 'length shape': float(preset['length shape']),
                'length location': float(preset['length location']),
                'length scale': float(preset['length scale']),
            })
    return rows


def build_run(template_path, label, version, presets, iterations, u_crit, out_root, hydrograph,
              common_name, season):
    project = _load_project(template_path)
    run_name = f"{label}_{version}_{SPECIES[common_name]['Species']}_{season}"
    run_dir = os.path.join(out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    unit_path = os.path.join(run_dir, 'unit_params.csv')
    with open(unit_path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(project['unit_parameters']['csv_content'])

    ops = pd.DataFrame(project['operating_scenarios']).dropna(subset=['Scenario', 'Facility'], how='all')
    template_scenario = ops['Scenario'].unique()
    if len(template_scenario) != 1:
        raise ValueError(f'{template_path}: expected one operating scenario, found {template_scenario}')
    ops_path = os.path.join(run_dir, 'operating_scenarios.csv')
    ops.assign(Scenario=season).to_csv(ops_path, index=False)

    hydro_path = os.path.join(run_dir, 'hydrograph.csv')
    hydrograph.to_csv(hydro_path, index=False)

    flow_scenarios = [{
        'Scenario': season, 'Scenario Number': 1, 'Season': season, 'Months': SEASONS[season],
        'Flow': 'hydrograph', 'Gage': None, 'FlowYear': 2021, 'Prorate': 1,
    }]
    graph_summary, graph_data = graph_inputs(project)
    info = project['project_info'][0]
    return {
        'proj_dir': run_dir,
        'project_name': run_name,
        'project_notes': f'Seasonal preset A/B ({version})',
        'model_setup': info['Model Setup'],
        'units': info['Units'],
        'facilities': project['facilities'],
        'unit_parameters_file': unit_path,
        'operating_scenarios_file': ops_path,
        'population': population_rows(presets, iterations, u_crit, common_name, season),
        'flow_scenarios': flow_scenarios,
        'graph_data': graph_data,
        'graph_summary': graph_summary,
        'units_system': info['Units'],
        'simulation_mode': info['Model Setup'],
        'output_name': run_name,
        'hydrograph_file': hydro_path,
    }


def run_one(data_dict, seed):
    import Stryke.stryke as stryke
    os.environ['STRYKE_RANDOM_SEED'] = str(seed)
    sim = stryke.simulation(data_dict['proj_dir'], data_dict['output_name'], existing=False)
    sim.webapp_import(data_dict, data_dict['output_name'])
    sim.run()
    sim.summary()
    with pd.HDFStore(os.path.join(data_dict['proj_dir'], f"{data_dict['output_name']}.h5"), mode='r') as store:
        daily = store['Daily']
    return daily


def summarize(per_iter, label, version):
    """per_iter: rows of (species, scenario, iteration, num_entrained, mort) across all seasonal runs."""
    rows = []
    for (species, scenario), grp in per_iter.groupby(['species', 'scenario']):
        entrained = grp['num_entrained'].sum()
        rows.append({'config': label, 'version': version, 'species': species, 'scenario': scenario,
                     'mean_entrained': grp['num_entrained'].mean(),
                     'mean_mortalities': grp['mort'].mean(),
                     'mortality_rate': grp['mort'].sum() / entrained if entrained else np.nan})
    annual = per_iter.groupby(['species', 'iteration'])[['num_entrained', 'mort']].sum().reset_index()
    for species, grp in annual.groupby('species'):
        entrained = grp['num_entrained'].sum()
        rows.append({'config': label, 'version': version, 'species': species, 'scenario': 'ANNUAL',
                     'mean_entrained': grp['num_entrained'].mean(),
                     'mean_mortalities': grp['mort'].mean(),
                     'mortality_rate': grp['mort'].sum() / entrained if entrained else np.nan})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--u-crit-ftps', type=float, default=2.0,
                        help='U_crit (ft/s) applied identically to both species in every version')
    parser.add_argument('--out', default=os.path.join(os.environ.get('TEMP', PROJECT_ROOT), 'stryke_kakabeka_ab'))
    parser.add_argument('--configs', default='existing,new')
    parser.add_argument('--versions', default=','.join(VERSIONS))
    args = parser.parse_args()

    templates = {'existing': EXISTING_TEMPLATE, 'new': NEW_TEMPLATE}
    hydrograph = full_year_hydrograph()
    os.makedirs(args.out, exist_ok=True)
    results = []
    out_csv = os.path.join(args.out, 'kakabeka_seasonal_ab_summary.csv')
    for version in args.versions.split(','):
        presets = load_presets(VERSIONS[version])
        for label in args.configs.split(','):
            frames = []
            for common_name in SPECIES:
                for season in SEASONS:
                    print(f'=== Running {label} / {version} / {common_name} / {season} ===', flush=True)
                    data_dict = build_run(templates[label], label, version, presets, args.iterations,
                                          args.u_crit_ftps, args.out, hydrograph, common_name, season)
                    daily = run_one(data_dict, args.seed)
                    if daily.empty:
                        raise RuntimeError(f"No Daily rows for {data_dict['output_name']}")
                    daily = daily.assign(mort=daily['num_entrained'] - daily['num_survived'])
                    per_iter = daily.groupby('iteration')[['num_entrained', 'mort']].sum().reset_index()
                    frames.append(per_iter.assign(species=common_name, scenario=season))
            results.extend(summarize(pd.concat(frames, ignore_index=True), label, version))
            pd.DataFrame(results).to_csv(out_csv, index=False)
    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    print(f'\nWrote {out_csv}')


if __name__ == '__main__':
    main()
