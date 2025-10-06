import pandas as pd
import numpy as np

h5 = r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\stryke\same_old.h5'
store = pd.HDFStore(h5, mode='r')
if '/Daily' not in store.keys():
    print('No /Daily table found in', h5)
else:
    df = store['/Daily']
    if 'num_entrained' not in df.columns:
        print('/Daily has no num_entrained column')
    else:
        v = df['num_entrained'].fillna(0).astype(float)
        total_days = len(v)
        print(f'Total days in /Daily: {total_days}')
        for n in (10,100,1000):
            q = 1.0 - 1.0/float(n)
            thr = float(np.nanquantile(v, q))
            obs = int((v >= thr).sum())
            approx = (total_days/obs) if obs>0 else float('inf')
            print(f'1-in-{n} threshold = {thr:.1f} fish; observed {obs} days; approx 1-in-{int(approx) if obs>0 else "inf"} days')
store.close()
