import pandas as pd
import os

HDF = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'same_old.h5')
print('HDF path:', HDF)
store = pd.HDFStore(HDF)
keys = [k for k in store.keys() if k.startswith('/simulations/')]
print('Simulation keys found:', keys)
if not keys:
    raise SystemExit(1)

key = keys[0]
print('Inspecting', key)
df = store[key]
print('Columns:', df.columns.tolist())
state_cols = [c for c in df.columns if c.startswith('state_')]
surv_cols = [c for c in df.columns if c.startswith('survival_')]
print('state_cols:', state_cols)
print('survival_cols:', surv_cols)

def show_row(i):
    row = df.iloc[i]
    print('\nRow index', i)
    for c in state_cols:
        print(f'{c}:', row.get(c))
    for c in surv_cols:
        print(f'{c}:', row.get(c))
    # show iteration/day
    print('iteration:', row.get('iteration'), 'day:', row.get('day'))

for i in range(min(12, len(df))):
    show_row(i)

store.close()
