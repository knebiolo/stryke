import os
import pandas as pd

h5_path = r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\stryke\same_old.h5'
print('Opening', h5_path)
with pd.HDFStore(h5_path, mode='r') as store:
    print('Keys:', store.keys())
    sim_key = '/simulations/summer/Micropterus'
    if sim_key not in store.keys():
        print('Simulation key not found:', sim_key)
    else:
        df = store[sim_key]
        print('Loaded simulations DF shape:', df.shape)
        # Find state and survival columns
        state_cols = [c for c in df.columns if c.startswith('state_')]
        survival_cols = [c for c in df.columns if c.startswith('survival_')]
        print('state_cols:', state_cols)
        print('survival_cols:', survival_cols)

        # Build passage events: first non-river_node state is passage route
        events = []
        for idx, row in df.iterrows():
            # prefer explicit stored 'index' if available
            fish_id = row.get('index') if 'index' in df.columns else idx
            iteration = row.get('iteration') if 'iteration' in df.columns else None
            day = row.get('day') if 'day' in df.columns else None
            flow = row.get('flow') if 'flow' in df.columns else None
            # find first non-river_node state
            passage_route = None
            for c in state_cols:
                val = row.get(c)
                if pd.isna(val):
                    continue
                if 'river_node' not in str(val).lower():
                    passage_route = val
                    break
            # find survival for the node (matching the index of the first 'U' would be better)
            # As a conservative approach, take the last survival column value as final survival
            survived = None
            if survival_cols:
                try:
                    # assume survival columns align with states and take last non-null value
                    surv_vals = [row.get(c) for c in survival_cols]
                    surv_vals = [v for v in surv_vals if not pd.isna(v)]
                    if surv_vals:
                        survived = int(surv_vals[-1])
                except Exception:
                    survived = None
            if passage_route:
                events.append({'fish_id': fish_id, 'iteration': iteration, 'day': day, 'passage_route': passage_route, 'flow': flow, 'survived': survived})

        events_df = pd.DataFrame(events)
        print('\nReconstructed events shape:', events_df.shape)
        print('Unique fish ids in events:', events_df['fish_id'].nunique())

        # Policy B dedupe by (fish_id, iteration, day)
        dedup = events_df.drop_duplicates(subset=['fish_id', 'iteration', 'day'])
        print('After dedupe (Policy B):', dedup.shape)

        # Per-iteration counts
        if 'iteration' in dedup.columns and dedup['iteration'].notnull().any():
            per_iter = dedup.groupby(['iteration', 'passage_route']).size().unstack(fill_value=0)
            mean_per_route = per_iter.mean(axis=0).sort_values(ascending=False)
            print('\nMean per-iteration route counts:')
            print(mean_per_route)
            print('\nPer-iteration table:')
            print(per_iter.head(10))
        else:
            print('\nNo iteration info available; showing dedup counts:')
            print(dedup['passage_route'].value_counts())

        # Survival by route
        if 'survived' in dedup.columns and dedup['survived'].notnull().any():
            surv = dedup.groupby('passage_route')['survived'].agg(['count', 'sum', 'mean'])
            surv.columns = ['Total', 'Survived', 'Survival Rate']
            surv['Survival %'] = surv['Survival Rate'] * 100
            print('\nSurvival by route:')
            print(surv)
        else:
            print('\nNo explicit survival data available in deduped events')
