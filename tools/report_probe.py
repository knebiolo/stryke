import os
import sys
sys.path.insert(0, r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\stryke')
import pandas as pd
import numpy as np
from Stryke.stryke import bootstrap_mean_ci

h5 = os.path.join(r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\stryke','same_old.h5')
print('HDF path:', h5)
print('Exists:', os.path.exists(h5))

if not os.path.exists(h5):
    sys.exit(1)

store = pd.HDFStore(h5, mode='r')
print('\nKeys in HDF:')
for k in store.keys():
    print(' ', k)

# Population length fields
if '/Population' in store.keys():
    pop = store['/Population']
    print('\nPopulation table columns:', list(pop.columns))
    if 'Length_mean' in pop.columns:
        print('Population Length_mean examples:')
        print(pop[['Species','Scenario','Length_mean','Length_sd']].head(10))
    else:
        print('No Length_mean in Population table')
else:
    print('\nNo /Population table found')

# Beta distributions (units)
if '/Beta_Distributions_Units' in store.keys():
    beta_units = store['/Beta_Distributions_Units']
    print('\nBeta_Distributions_Units head:')
    print(beta_units.head(10))
    # try to find Unit 1 and Unit 2 rows
    mask_u1 = beta_units['Passage Route'].str.contains('Unit 1', case=False, na=False)
    mask_u2 = beta_units['Passage Route'].str.contains('Unit 2', case=False, na=False)
    if mask_u1.any():
        print('\nUnit 1 beta row:')
        print(beta_units[mask_u1])
    if mask_u2.any():
        print('\nUnit 2 beta row:')
        print(beta_units[mask_u2])
else:
    print('\nNo /Beta_Distributions_Units table found')

# Whole project empirical mean from a simulation table
sim_keys = [k for k in store.keys() if k.startswith('/simulations/')]
if sim_keys:
    key = sim_keys[0]
    print('\nUsing simulations table:', key)
    dat = store[key]
    # find survival_* columns
    surv_cols = [c for c in dat.columns if c.startswith('survival_')]
    print('Found survival cols:', surv_cols[:10])
    if surv_cols:
        # find max suffix
        try:
            moves = [int(c.split('_')[-1]) for c in surv_cols]
            max_move = max(moves)
        except Exception:
            max_move = 0
        surv_col = f'survival_{max_move}'
        if surv_col in dat.columns:
            whole_proj_succ = dat.groupby(by=['iteration','day'])[surv_col].sum().to_frame().reset_index(drop=False).rename(columns={surv_col:'successes'})
            whole_proj_count = dat.groupby(by=['iteration','day'])[surv_col].count().to_frame().reset_index(drop=False).rename(columns={surv_col:'count'})
            whole_summ = whole_proj_succ.merge(whole_proj_count)
            whole_summ['prob'] = whole_summ['successes'] / whole_summ['count']
            whole_summ.fillna(0, inplace=True)
            mean_emp = whole_summ['prob'].mean()
            std_emp = whole_summ['prob'].std()
            mean_boot, lcl_boot, ucl_boot = bootstrap_mean_ci(whole_summ['prob'].values, n_bootstrap=2000, ci=95)
            print('\nWhole-project survival empirical mean:', mean_emp)
            print('Std:', std_emp)
            print('Bootstrap mean:', mean_boot)
            print('Bootstrap 95% CI:', (lcl_boot, ucl_boot))
        else:
            print('Survival column', surv_col, 'not found in table')
    else:
        print('No survival_* columns found in simulation table')
else:
    print('\nNo simulation tables found in HDF')

store.close()
print('\nDone')
