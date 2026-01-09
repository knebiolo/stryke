"""Generate suggested best-fit distributions for species_defaults in webapp/app.py

This script will:
- Parse the `species_defaults` list literal from `webapp/app.py`.
- For each species entry, attempt to extract a genus name from the `name` field.
- Query the EPRI fitter via `stryke.epri(Genus=genus)` and run Pareto/LogNormal/Weibull/Gamma fits.
- Collect KS p-values and distribution parameters, choose the best distribution.
- Write out a CSV `species_fit_suggestions.csv` with recommended dist and parameters.

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
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'species_fit_suggestions.csv')

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
        return (None, None, None, None)
    obs = np.asarray(obs)
    n = obs.size
    if n == 0:
        return (None, None, None, None)
    try:
        # Unpack params: scipy may return (shape, loc, scale) or (c, loc, scale)
        shape, loc, scale = params[0], params[1], params[2]
        # Compute pdf
        pdf = dist_obj.pdf(obs, shape, loc=loc, scale=scale)
        # Avoid zeros
        pdf = np.clip(pdf, 1e-300, None)
        loglik = float(np.sum(np.log(pdf)))
        # parameter count k: if loc fixed -> 2 else 3
        k = 2 if floc_fixed else 3
        aic = 2 * k - 2 * loglik
        if n - k - 1 > 0:
            aicc = aic + (2 * k * (k + 1)) / float(n - k - 1)
        else:
            aicc = aic
        bic = math.log(n) * k - 2 * loglik
        return (loglik, aic, aicc, bic)
    except Exception:
        return (None, None, None, None)


def run_fits_and_select_best(genus):
    # Import here to avoid top-level dependency when not running script
    try:
        from Stryke.stryke import epri
    except Exception:
        # allow import via package name
        import Stryke
        from Stryke.stryke import epri

    filter_args = {}
    if genus:
        filter_args['Genus'] = genus
    # Call epri
    try:
        fish = epri(**filter_args)
    except Exception as e:
        raise RuntimeError(f'epri query failed for genus {genus}: {e}')

    # Run the standard three fits only
    fish.ParetoFit()
    fish.LogNormalFit()
    fish.WeibullMinFit()
    # Ensure plot() called to compute KS tests (plot() computes and stores pareto_t, log_normal_t, weibull_t, gamma_t)
    fig = None
    try:
        fig = fish.plot()
    except Exception:
        traceback.print_exc()

    # Collect p-values
    def get_p(val):
        try:
            return float(val)
        except Exception:
            return -1.0

    pareto_p = get_p(getattr(fish, 'pareto_t', -1))
    lognorm_p = get_p(getattr(fish, 'log_normal_t', -1))
    weibull_p = get_p(getattr(fish, 'weibull_t', -1))

    # Observations
    try:
        observations = np.asarray(fish.epri.FishPerMft3.values)
    except Exception:
        observations = np.array([])

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
    best_by_aicc = min(aicc_map.items(), key=lambda kv: kv[1])
    # Check for ties within delta_aicc
    delta_aicc = 2.0
    candidates = [k for k, v in aicc_map.items() if abs(v - best_by_aicc[1]) <= delta_aicc]
    if len(candidates) == 1:
        best_dist = candidates[0]
        reason = f'AICc lowest ({best_by_aicc[1]:.3f})'
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
            reason = f'AICc tie; selected by AD ({best_by_ad[1]:.4f})'
        else:
            p_map = {
                'Pareto': pareto_p,
                'Log Normal': lognorm_p,
                'Weibull': weibull_p,
            }
            # restrict to candidates
            p_map = {k: p_map[k] for k in candidates}
            best_dist = max(p_map.items(), key=lambda kv: kv[1])[0]
            reason = 'AICc tie; selected by KS p-value'

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

    return {
        'genus': genus,
        'best_dist': best_dist,
        'pareto_p': pareto_p,
        'lognorm_p': lognorm_p,
        'weibull_p': weibull_p,
        'gamma_p': gamma_p,
        'extreme_p': getattr(fish, 'extreme_t', 'N/A'),
        'shape': params['shape'],
        'location': params['location'],
        'scale': params['scale'],
        'metrics': metrics,
        'decision_reason': reason,
        'plot_fig': fig,
    }


def main():
    print('Extracting species_defaults from app.py...')
    species = extract_species_defaults(APP_PY)
    print(f'Found {len(species)} species entries')

    rows = []
    pdf_path = os.path.join(PROJECT_ROOT, 'species_fit_report.pdf')
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
            res = run_fits_and_select_best(genus)
            # Flatten metrics
            m = res.get('metrics', {})
            row = {
                'name': name,
                'genus': genus,
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
        except Exception as e:
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
        'name', 'genus', 'best_dist', 'decision_reason',
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
