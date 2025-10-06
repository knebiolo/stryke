import pandas as pd
import os

HDF = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'same_old.h5')
print('HDF path:', HDF)
store = pd.HDFStore(HDF)
keys = [k for k in store.keys() if k.startswith('/simulations/')]
print('Simulation keys found:', keys)
key = keys[0]
df = store[key]
state_cols = [c for c in df.columns if c.startswith('state_')]
surv_cols = [c for c in df.columns if c.startswith('survival_')]
print('state_cols:', state_cols)
print('surv_cols:', surv_cols)

all_targets = set()
# For this diagnostic, treat any state containing 'Unit' as a target
for c in df.columns:
    pass

def extract_for_row(i):
    row = df.iloc[i]
    fish_path = [row[col] for col in state_cols if pd.notna(row[col])]
    recs = []
    for pos, state in enumerate(fish_path):
        if 'Unit' in str(state):
            surv = None
            chosen_col = None
            if pos >= 0 and surv_cols and len(surv_cols) > pos:
                chosen_col = surv_cols[pos]
                surv = row.get(chosen_col)
            recs.append((i, pos, state, chosen_col, surv))
    return recs

rows_with_units = []
for i in range(len(df)):
    rows_with_units.extend(extract_for_row(i))

for r in rows_with_units[:50]:
    print('row_idx', r[0], 'pos', r[1], 'state', r[2], 'chosen_surv_col', r[3], 'value', r[4])

store.close()
