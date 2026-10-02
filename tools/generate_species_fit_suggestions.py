"""Generate suggested best-fit distributions for species_defaults in webapp/app.py

This script will:
- Parse the `species_defaults` list literal from `webapp/app.py`.
- Resolve genus, meteorological months, and Great Lakes HUC02=4 from each preset name.
- Fit Pareto, lognormal, and Weibull distributions to the matching positive EPRI rates.
- Select by AICc, using Anderson-Darling as a tie-breaker, and report sample metadata.
- Write a CSV and PDF review report without modifying `webapp/app.py`.

Usage: run this in the project root where your Python environment has the dependencies installed.

Note: This script only *suggests* changes. It does not modify `webapp/app.py`.
"""
import re
import ast
import os
import csv
import sys
import traceback
import math
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
APP_PY = os.path.join(PROJECT_ROOT, 'webapp', 'app.py')
OUTPUT_CSV = os.environ.get(
    'STRYKE_FIT_OUTPUT_CSV',
    os.path.join(PROJECT_ROOT, 'species_fit_suggestions.csv'),
)

# Delay importing heavy libraries until needed
# Ensure repo root is on sys.path so local package imports work when running script
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def extract_species_defaults(app_py_path):
    text = open(app_py_path, 'r', encoding='utf-8').read()
    # Find the start of the species_defaults assignment
    m = re.search(r"species_defaults\s*=\s*\[", text)
    if not m:
        raise RuntimeError('Could not find species_defaults in app.py')
    start = m.start()
    # Find the matching closing bracket by scanning characters
    idx = text.find('[', start)
    if idx == -1:
        raise RuntimeError('Malformed species_defaults')
    depth = 0
    end_idx = None
    for i in range(idx, len(text)):
        ch = text[i]
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx is None:
        raise RuntimeError('Could not find end of species_defaults')
    list_text = text[idx:end_idx+1]
    # Use ast.literal_eval to safely parse Python literal
    species_list = ast.literal_eval(list_text)
    return species_list


def guess_genus_from_name(name):
    # Expect names like 'Ascipenser, Great Lakes, Annual' or 'Micropterus, Great Lakes, Met Spring'
    if not name:
        return None
    first = name.split(',')[0].strip()
    # If first has multiple words, take first token (genus)
    genus = first.split()[0]
    return genus


METEOROLOGICAL_SEASON_MONTHS = {
    'Winter': (12, 1, 2),
    'Spring': (3, 4, 5),
    'Summer': (6, 7, 8),
    'Fall': (9, 10, 11),
}
GREAT_LAKES_HUC02 = 4


def parse_preset_filters(name):
    """Resolve the genus, month list, and Great Lakes HUC02 from a preset name."""
    genus, separator, label = name.partition(', Great Lakes, ')
    if not separator or not genus.strip():
        raise ValueError(f'Unsupported species-default name: {name!r}')

    label = label.strip()
    if label == 'Annual':
        months = tuple(range(1, 13))
    else:
        if not label.startswith('Met '):
            raise ValueError(f'Unsupported season label in species default: {name!r}')
        season_names = [
            part.strip()
            for part in re.split(r'\s*(?:&|,|\band\b)\s*', label[4:])
            if part.strip()
        ]
        unknown = [
            season for season in season_names
            if season not in METEOROLOGICAL_SEASON_MONTHS
        ]
        if unknown:
            raise ValueError(f'Unknown season(s) {unknown!r} in species default {name!r}')
        months = tuple(sorted({
            month
            for season in season_names
            for month in METEOROLOGICAL_SEASON_MONTHS[season]
        }))
        if not months:
            raise ValueError(f'No months resolved for species default {name!r}')

    return genus.strip(), months, GREAT_LAKES_HUC02


def anderson_darling_statistic(sample, cdf_fn):
    """Compute Anderson-Darling statistic for sample given CDF function.
    cdf_fn should accept an array and return CDF values in [0,1]."""
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


def compute_loglik_aic(obs, dist_obj, params, floc_fixed=True):
    """Compute log-likelihood, AIC, AICc, and BIC for observations under a scipy.stats distribution object.
    params is a tuple (shape, loc, scale) as returned by scipy fit calls.
    We assume floc_fixed=True (so loc was fixed) and count k=2 parameters (shape & scale).
    Returns (loglik, aic, aicc, bic) or (None, None, None, None) on error.
    """
    if params is None:
        raise ValueError('Cannot calculate fit statistics without fitted parameters.')
    obs = np.asarray(obs, dtype=float)
    n = obs.size
    if n == 0:
        raise ValueError('Cannot calculate fit statistics without observations.')
    shape, loc, scale = params[0], params[1], params[2]
    logpdf = dist_obj.logpdf(obs, shape, loc=loc, scale=scale)
    if not np.isfinite(logpdf).all():
        raise ValueError('Fitted distribution produced non-finite log probabilities.')
    loglik = float(np.sum(logpdf))
    k = 2 if floc_fixed else 3
    aic = 2 * k - 2 * loglik
    aicc = aic + (2 * k * (k + 1) / float(n - k - 1) if n - k - 1 > 0 else aic)
    bic = math.log(n) * k - 2 * loglik
    return (loglik, aic, aicc, bic)


def fit_distributions(observations):
    """Fit supported positive-rate distributions and select by AICc, then AD."""
    from scipy.stats import pareto, lognorm, weibull_min

    values = np.asarray(observations, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Distribution fitting requires finite, strictly positive rates.')

    distributions = {
        'Pareto': pareto,
        'Log Normal': lognorm,
        'Weibull': weibull_min,
    }
    metrics = {}
    for name, distribution in distributions.items():
        params = distribution.fit(values, floc=0)
        loglik, aic, aicc, _ = compute_loglik_aic(values, distribution, params)
        ad = anderson_darling_statistic(
            values,
            lambda sample, dist=distribution, fitted=params: dist.cdf(sample, *fitted),
        )
        metrics[name] = {
            'params': params,
            'loglik': loglik,
            'aic': aic,
            'aicc': aicc,
            'ad': ad,
        }

    criterion_key = 'aicc' if values.size > 3 else 'aic'
    criterion = 'AICc' if criterion_key == 'aicc' else 'AIC'
    best_score = min(result[criterion_key] for result in metrics.values())
    candidates = [
        (name, result)
        for name, result in metrics.items()
        if result[criterion_key] - best_score <= 2.0
    ]
    best_name, best_result = min(candidates, key=lambda item: item[1]['ad'])
    decision_reason = (
        f'{criterion} tie; selected by AD ({best_result["ad"]:.4f})'
        if len(candidates) > 1
        else f'{criterion} lowest ({best_score:.3f})'
    )
    return best_name, best_result['params'], metrics, decision_reason, criterion


def run_fits_and_select_best(preset_name):
    from Stryke.stryke import epri

    genus, months, huc02 = parse_preset_filters(preset_name)
    fish = epri(Genus=genus, HUC02=[huc02], Month=list(months))
    rates = np.asarray(fish.epri.FishPerMft3.values, dtype=float)
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError(f'Invalid entrainment rates found for {preset_name!r}.')
    n_present = rates.size
    n_zero = int(np.count_nonzero(rates == 0))
    observations = rates[rates > 0]
    if observations.size == 0:
        raise ValueError(f'No positive entrainment rates found for {preset_name!r}.')
    fish.epri = fish.epri.loc[fish.epri.FishPerMft3 > 0].copy()

    # Run the standard three fits only
    fish.ParetoFit()
    fish.LogNormalFit()
    fish.WeibullMinFit()
    fig = fish.plot()

    # Collect p-values
    def get_p(val):
        try:
            return float(val)
        except Exception:
            return -1.0

    pareto_p = get_p(getattr(fish, 'pareto_t', -1))
    lognorm_p = get_p(getattr(fish, 'log_normal_t', -1))
    weibull_p = get_p(getattr(fish, 'weibull_t', -1))

    # For each distribution compute loglik/AIC/AICc/BIC and AD statistic
    metrics = {}
    from scipy.stats import pareto as _pareto, lognorm as _lognorm, weibull_min as _weibull

    # Pareto
    pareto_params = getattr(fish, 'dist_pareto', None)
    pareto_loglik, pareto_aic, pareto_aicc, pareto_bic = compute_loglik_aic(observations, _pareto, pareto_params, floc_fixed=True)
    try:
        pareto_ad = anderson_darling_statistic(observations, lambda x: _pareto.cdf(x, pareto_params[0], loc=pareto_params[1], scale=pareto_params[2])) if pareto_params is not None else None
    except Exception:
        pareto_ad = None

    # Lognormal
    lognorm_params = getattr(fish, 'dist_lognorm', None)
    lognorm_loglik, lognorm_aic, lognorm_aicc, lognorm_bic = compute_loglik_aic(observations, _lognorm, lognorm_params, floc_fixed=True)
    try:
        lognorm_ad = anderson_darling_statistic(observations, lambda x: _lognorm.cdf(x, lognorm_params[0], loc=lognorm_params[1], scale=lognorm_params[2])) if lognorm_params is not None else None
    except Exception:
        lognorm_ad = None

    # Weibull
    weibull_params = getattr(fish, 'dist_weibull', None)
    weibull_loglik, weibull_aic, weibull_aicc, weibull_bic = compute_loglik_aic(observations, _weibull, weibull_params, floc_fixed=True)
    try:
        weibull_ad = anderson_darling_statistic(observations, lambda x: _weibull.cdf(x, weibull_params[0], loc=weibull_params[1], scale=weibull_params[2])) if weibull_params is not None else None
    except Exception:
        weibull_ad = None

    metrics.update({
        'pareto_loglik': pareto_loglik, 'pareto_aic': pareto_aic, 'pareto_aicc': pareto_aicc, 'pareto_bic': pareto_bic, 'pareto_ad': pareto_ad,
        'lognorm_loglik': lognorm_loglik, 'lognorm_aic': lognorm_aic, 'lognorm_aicc': lognorm_aicc, 'lognorm_bic': lognorm_bic, 'lognorm_ad': lognorm_ad,
        'weibull_loglik': weibull_loglik, 'weibull_aic': weibull_aic, 'weibull_aicc': weibull_aicc, 'weibull_bic': weibull_bic, 'weibull_ad': weibull_ad,
    })

    # Decision algorithm: primary = lowest AICc, secondary = lowest AD (smaller better), tertiary = highest KS p-value
    # Decide whether fixed-loc or free-loc Gamma is preferable
    aicc_map = {
        'Pareto': pareto_aicc if pareto_aicc is not None else float('inf'),
        'Log Normal': lognorm_aicc if lognorm_aicc is not None else float('inf'),
        'Weibull': weibull_aicc if weibull_aicc is not None else float('inf'),
    }
    # pick lowest AICc
    criterion = 'AICc' if observations.size > 3 else 'AIC'
    criterion_map = aicc_map if criterion == 'AICc' else {
        'Pareto': pareto_aic,
        'Log Normal': lognorm_aic,
        'Weibull': weibull_aic,
    }
    best_by_aicc = min(criterion_map.items(), key=lambda kv: kv[1])
    # Check for ties within delta_aicc
    delta_aicc = 2.0
    candidates = [k for k, v in criterion_map.items() if abs(v - best_by_aicc[1]) <= delta_aicc]
    if len(candidates) == 1:
        best_dist = candidates[0]
        reason = f'{criterion} lowest ({best_by_aicc[1]:.3f})'
    else:
        # tie-breaker: AD statistic (smaller better)
        ad_map = {
            'Pareto': pareto_ad if pareto_ad is not None else float('inf'),
            'Log Normal': lognorm_ad if lognorm_ad is not None else float('inf'),
            'Weibull': weibull_ad if weibull_ad is not None else float('inf'),
        }
        best_by_ad = min(((d, ad_map[d]) for d in candidates), key=lambda kv: kv[1])
        # If AD available, pick that; otherwise fallback to KS p-value
        if math.isfinite(best_by_ad[1]):
            best_dist = best_by_ad[0]
            reason = f'{criterion} tie; selected by AD ({best_by_ad[1]:.4f})'
        else:
            p_map = {
                'Pareto': pareto_p,
                'Log Normal': lognorm_p,
                'Weibull': weibull_p,
            }
            # restrict to candidates
            p_map = {k: p_map[k] for k in candidates}
            best_dist = max(p_map.items(), key=lambda kv: kv[1])[0]
            reason = f'{criterion} tie; selected by KS p-value'

    # Grab params
    params = {'shape': None, 'location': None, 'scale': None}
    try:
        if best_dist == 'Pareto' and getattr(fish, 'dist_pareto', None) is not None:
            d = fish.dist_pareto
            params['shape'], params['location'], params['scale'] = d[0], d[1], d[2]
        elif best_dist == 'Log Normal' and getattr(fish, 'dist_lognorm', None) is not None:
            d = fish.dist_lognorm
            params['shape'], params['location'], params['scale'] = d[0], d[1], d[2]
        elif best_dist == 'Weibull' and getattr(fish, 'dist_weibull', None) is not None:
            d = fish.dist_weibull
            params['shape'], params['location'], params['scale'] = d[0], d[1], d[2]
        
    except Exception:
        traceback.print_exc()

    best_dist, selected_params, fit_metrics, reason, criterion = fit_distributions(observations)
    params = {
        'shape': float(selected_params[0]),
        'location': float(selected_params[1]),
        'scale': float(selected_params[2]),
    }
    metrics = {}
    metric_prefixes = {
        'Pareto': 'pareto',
        'Log Normal': 'lognorm',
        'Weibull': 'weibull',
    }
    for name, result in fit_metrics.items():
        prefix = metric_prefixes[name]
        metrics[f'{prefix}_loglik'] = result['loglik']
        metrics[f'{prefix}_aic'] = result['aic']
        metrics[f'{prefix}_aicc'] = result['aicc']
        metrics[f'{prefix}_ad'] = result['ad']

    return {
        'genus': genus,
        'months': months,
        'huc02': huc02,
        'n_present': n_present,
        'n_positive': int(observations.size),
        'n_zero': n_zero,
        'occur_prob': float(fish.presence),
        'max_ent_rate': float(fish.max_ent_rate),
        'best_dist': best_dist,
        'pareto_p': pareto_p,
        'lognorm_p': lognorm_p,
        'weibull_p': weibull_p,
        'gamma_p': getattr(fish, 'gamma_t', None),
        'extreme_p': getattr(fish, 'extreme_t', 'N/A'),
        'shape': params['shape'],
        'location': params['location'],
        'scale': params['scale'],
        'metrics': metrics,
        'decision_reason': reason,
        'criterion': criterion,
        'plot_fig': fig,
    }


def main():
    print('Extracting species_defaults from app.py...')
    species = extract_species_defaults(APP_PY)
    print(f'Found {len(species)} species entries')

    rows = []
    pdf_path = os.environ.get(
        'STRYKE_FIT_OUTPUT_PDF',
        os.path.join(PROJECT_ROOT, 'species_fit_report.pdf'),
    )
    pdf = PdfPages(pdf_path)
    # Optional environment override to limit number of species processed for quick testing
    try:
        limit = int(os.environ.get('SUGGEST_LIMIT')) if os.environ.get('SUGGEST_LIMIT') else None
    except Exception:
        limit = None
    for i, sp in enumerate(species, 1):
        if limit is not None and i > limit:
            break
        name = sp.get('name') if isinstance(sp, dict) else None
        print(f'[{i}/{len(species)}] Processing: {name}')
        genus = guess_genus_from_name(name)
        if not genus:
            print('  Could not guess genus, skipping')
            rows.append({'name': name, 'error': 'no genus'})
            continue
        try:
            res = run_fits_and_select_best(name)
            # Flatten metrics
            m = res.get('metrics', {})
            row = {
                'name': name,
                'genus': genus,
                'huc02': res.get('huc02'),
                'months': ','.join(map(str, res.get('months', ()))),
                'n_present': res.get('n_present'),
                'n_positive': res.get('n_positive'),
                'n_zero': res.get('n_zero'),
                'occur_prob': res.get('occur_prob'),
                'max_ent_rate': res.get('max_ent_rate'),
                'best_dist': res.get('best_dist'),
                'decision_reason': res.get('decision_reason'),
                'pareto_p': res.get('pareto_p'), 'lognorm_p': res.get('lognorm_p'), 'weibull_p': res.get('weibull_p'), 'gamma_p': res.get('gamma_p'), 'extreme_p': res.get('extreme_p'),
                'shape': res.get('shape'), 'location': res.get('location'), 'scale': res.get('scale'),
                'pareto_loglik': m.get('pareto_loglik'), 'pareto_aic': m.get('pareto_aic'), 'pareto_aicc': m.get('pareto_aicc'), 'pareto_bic': m.get('pareto_bic'), 'pareto_ad': m.get('pareto_ad'),
                'lognorm_loglik': m.get('lognorm_loglik'), 'lognorm_aic': m.get('lognorm_aic'), 'lognorm_aicc': m.get('lognorm_aicc'), 'lognorm_bic': m.get('lognorm_bic'), 'lognorm_ad': m.get('lognorm_ad'),
                'weibull_loglik': m.get('weibull_loglik'), 'weibull_aic': m.get('weibull_aic'), 'weibull_aicc': m.get('weibull_aicc'), 'weibull_bic': m.get('weibull_bic'), 'weibull_ad': m.get('weibull_ad'),
                'gamma_loglik': m.get('gamma_loglik'), 'gamma_aic': m.get('gamma_aic'), 'gamma_aicc': m.get('gamma_aicc'), 'gamma_bic': m.get('gamma_bic'), 'gamma_ad': m.get('gamma_ad'),
                'gamma_free_loglik': m.get('gamma_free_loglik'), 'gamma_free_aic': m.get('gamma_free_aic'), 'gamma_free_aicc': m.get('gamma_free_aicc'), 'gamma_free_bic': m.get('gamma_free_bic'), 'gamma_free_ad': m.get('gamma_free_ad'), 'gamma_free_loc': m.get('gamma_free_loc'),
                'extreme_loglik': m.get('extreme_loglik'), 'extreme_aic': m.get('extreme_aic'), 'extreme_aicc': m.get('extreme_aicc'), 'extreme_bic': m.get('extreme_bic'), 'extreme_ad': m.get('extreme_ad'),
                'error': None,
            }
            rows.append(row)
            print('  Suggested best:', res.get('best_dist'))
            # If a figure was produced, add a page to the PDF with the figure and metrics text
            fig = res.get('plot_fig')
            if fig is not None:
                try:
                    # Add a header to the fig with species name
                    fig.suptitle(f"{name} — suggested: {res.get('best_dist')} ({res.get('decision_reason')})", fontsize=10)
                    # Add a small metrics textbox at the bottom of the figure
                    try:
                        metrics = res.get('metrics', {})
                        txt_lines = []
                        for key in ('pareto_aicc','lognorm_aicc','weibull_aicc','gamma_aicc','gamma_free_aicc','extreme_aicc'):
                            v = metrics.get(key)
                            if v is not None:
                                txt_lines.append(f"{key}: {v:.3f}")
                        # include AD stats if present
                        for key in ('pareto_ad','lognorm_ad','weibull_ad','gamma_ad','gamma_free_ad','extreme_ad'):
                            v = metrics.get(key)
                            if v is not None:
                                txt_lines.append(f"{key}: {v:.4f}")
                        fig.text(0.02, 0.02, '\n'.join(txt_lines), fontsize=8, va='bottom', ha='left', family='monospace')
                    except Exception:
                        pass
                    pdf.savefig(fig)
                    fig.clf()
                except Exception:
                    try:
                        pdf.savefig()
                    except Exception:
                        pass
        except (ValueError, ZeroDivisionError) as e:
            print('  Error processing:', e)
            traceback.print_exc()
            rows.append({'name': name, 'error': str(e)})

    # Close PDF
    try:
        pdf.close()
        print('\nWrote PDF report to', pdf_path)
    except Exception:
        print('\nFailed to write PDF report')

    # Write CSV with extended columns
    fieldnames = [
        'name', 'genus', 'huc02', 'months', 'n_present', 'n_positive',
        'n_zero', 'occur_prob', 'max_ent_rate', 'best_dist', 'decision_reason',
        'pareto_p', 'lognorm_p', 'weibull_p', 'gamma_p',
        'shape', 'location', 'scale',
        'pareto_loglik', 'pareto_aic', 'pareto_aicc', 'pareto_bic', 'pareto_ad',
        'lognorm_loglik', 'lognorm_aic', 'lognorm_aicc', 'lognorm_bic', 'lognorm_ad',
        'weibull_loglik', 'weibull_aic', 'weibull_aicc', 'weibull_bic', 'weibull_ad',
        'error'
    ]
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fieldnames}
            writer.writerow(out)

    print('\nWrote suggestions to', OUTPUT_CSV)

if __name__ == '__main__':
    main()
