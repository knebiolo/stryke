import h5py, os
h5 = r'C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\stryke\simulation_report_20251006_132251\same_old.h5'
with h5py.File(h5,'r') as f:
    ds = f['simulations']['summer']['Micropterus']['table']
    print('dataset shape', ds.shape, 'dtype', ds.dtype)
    # fields: 'index', 'values_block_0', 'values_block_1', 'values_block_2', 'values_block_3'
    # values_block_3 appears to be S24 with length 8 -> likely state_ columns
    state_field = ds.dtype.names[-1]
    print('state field name:', state_field)
    cnt_rows = 0
    fish_with = set()
    entries = ds[:]
    for rec in entries:
        idx = int(rec['index'])
        # values_block_3 is an array of bytes (strings)
        states = rec[state_field]
        # convert bytes to str and strip
        path = []
        for s in states:
            try:
                ss = s.decode('utf-8') if isinstance(s, bytes) else str(s)
            except Exception:
                ss = str(s)
            ss = ss.strip('\x00')
            if ss:
                path.append(ss)
        # check for Unit/spill
        found=False
        for s in path:
            if ('Unit' in s) or ('spill' in s) or ('Spill' in s):
                cnt_rows += 1
                import h5py
                import os
                import pandas as pd

                # Path to extracted HDF from user zip
                h5 = r'C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\stryke\simulation_report_20251006_132251\same_old.h5'

                def decode_bytes_array(arr):
                    out = []
                    for s in arr:
                        try:
                            ss = s.decode('utf-8') if isinstance(s, bytes) else str(s)
                        except Exception:
                            ss = str(s)
                        ss = ss.strip('\x00')
                        if ss:
                            out.append(ss)
                    return out


                with h5py.File(h5, 'r') as f:
                    ds = f['simulations']['summer']['Micropterus']['table']
                    print('dataset shape', ds.shape, 'dtype', ds.dtype)
                    # Inspect dtype names -> last field is likely the state strings block
                    state_field = ds.dtype.names[-1]
                    print('state field name:', state_field)

                    entries = ds[:]

                    # Reconstruct passage events
                    events = []
                    for rec in entries:
                        fish_idx = int(rec['index'])
                        # attempt to read numeric blocks for iteration/day/flow
                        iteration = None
                        flow = None
                        # values_block_1 often contains float flow
                        if 'values_block_1' in ds.dtype.names:
                            try:
                                vb1 = rec['values_block_1']
                                if hasattr(vb1, '__len__') and len(vb1) > 0:
                                    flow = float(vb1[0])
                            except Exception:
                                flow = None
                        # values_block_2 may contain iteration/day ints
                        if 'values_block_2' in ds.dtype.names:
                            try:
                                vb2 = rec['values_block_2']
                                if hasattr(vb2, '__len__') and len(vb2) > 0:
                                    # Heuristic: first element may be iteration id if small integer
                                    maybe_iter = int(vb2[0])
                                    if 0 <= maybe_iter <= 1000:
                                        iteration = maybe_iter
                            except Exception:
                                iteration = None

                        states = decode_bytes_array(rec[state_field])
                        # Heuristic: find first passage-like state (Unit, Spill, GS, or anything not 'river_node')
                        passage_route = None
                        for s in states:
                            low = s.lower()
                            if 'river_node' in low:
                                continue
                            # pick states that look like facility/unit/spill
                            if 'unit' in s or 'spill' in s or 'gs' in s or 'matabitchuan' in s.lower() or 'unit' in s.lower():
                                passage_route = s
                                break
                            # fallback: any non-river_node
                            if passage_route is None:
                                passage_route = s

                        # attempt to get day string if present in the states list (e.g., '2024-06-02')
                        day = None
                        for s in states:
                            if s.count('-') == 2 and s.split('-')[0].isdigit():
                                day = s
                                break

                        if passage_route:
                            events.append({'fish_id': fish_idx, 'iteration': iteration, 'day': day, 'passage_route': passage_route, 'flow': flow})

                    # Print compact summary
                    print('\nReconstructed events:', len(events))
                    unique_indices = set(int(r['index']) for r in entries)
                    print('unique indices in raw table (sample window):', len(unique_indices))

                    # Build DataFrame and compute policy-B counts (unique per iteration)
                    ev_df = pd.DataFrame(events)
                    if ev_df.empty:
                        print('\nNo passage events reconstructed')
                    else:
                        # Dedupe by (fish_id, iteration, day) because fish IDs are per-day.
                        # If 'iteration' is missing, this reduces to per-day dedupe by fish_id/day.
                        dedup = ev_df.drop_duplicates(subset=['fish_id', 'iteration', 'day'])

                        # If iteration is present, compute per-iteration route counts then average across iterations.
                        if 'iteration' in dedup.columns and dedup['iteration'].notnull().any():
                            per_iter = dedup.groupby(['iteration', 'passage_route']).size().unstack(fill_value=0)
                            mean_per_route = per_iter.mean(axis=0).sort_values(ascending=False)
                            # Print only top 12 routes to keep output compact
                            print('\nPolicy B (mean unique fish per iteration) -- top routes:')
                            print(mean_per_route.head(12))
                        else:
                            # fallback: compute counts across dedup set and treat as single iteration equivalents
                            route_counts = dedup['passage_route'].value_counts()
                            print('\n[INFO] iteration not present; using deduped counts as single-iteration estimates:')
                            print(route_counts)

                        print('\nTotal unique deduped fish rows used:', len(dedup))
