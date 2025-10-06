import h5py
import pandas as pd
import numpy as np

h5_path = r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\stryke\same_old.h5'

print("="*70)
print("STRYKE SIMULATION ANALYSIS - Policy B Verification")
print("="*70)

with h5py.File(h5_path, 'r') as f:
    print("\n📊 HDF5 Structure:")
    print(f"Available keys: {list(f.keys())}")
    
    # Get simulations table
    sim_table = f['simulations']['summer']['Micropterus']['table']
    print(f"\n🐟 Simulations table shape: {sim_table.shape}")
    
    entries = sim_table[:]
    
    # Extract passage events with all relevant fields
    events = []
    for rec in entries:
        fish_id = int(rec['index'])
        
        # Extract iteration from values_block_2
        iteration = None
        if 'values_block_2' in sim_table.dtype.names:
            try:
                vb2 = rec['values_block_2']
                if hasattr(vb2, '__len__') and len(vb2) > 0:
                    iteration = int(vb2[0])
            except:
                pass
        
        # Extract flow from values_block_1
        flow = None
        if 'values_block_1' in sim_table.dtype.names:
            try:
                vb1 = rec['values_block_1']
                if hasattr(vb1, '__len__') and len(vb1) > 0:
                    flow = float(vb1[0])
            except:
                pass
        
        # Decode state strings
        state_field = sim_table.dtype.names[-1]
        states = []
        for s in rec[state_field]:
            try:
                ss = s.decode('utf-8') if isinstance(s, bytes) else str(s)
                ss = ss.strip('\x00').strip()
                if ss:
                    states.append(ss)
            except:
                pass
        
        # Find day (date string)
        day = None
        for s in states:
            if '-' in s and len(s) == 10:  # Date format
                day = s
                break
        
        # Find passage route (Unit 1, Unit 2, or spillway)
        passage_route = None
        for s in states:
            s_lower = s.lower()
            if 'unit' in s_lower or 'spillway' in s_lower:
                passage_route = s
                break
        
        # Extract survival from values_block_0 (last few elements often survival flags)
        survival = None
        if 'values_block_0' in sim_table.dtype.names:
            try:
                vb0 = rec['values_block_0']
                # Survival typically stored as 1.0 (survived) or 0.0 (died)
                # Often in positions 2-4 in the block
                if hasattr(vb0, '__len__') and len(vb0) > 2:
                    survival = float(vb0[2])  # Common position for survival flag
            except:
                pass
        
        if passage_route:
            events.append({
                'fish_id': fish_id,
                'iteration': iteration,
                'day': day,
                'passage_route': passage_route,
                'flow': flow,
                'survival': survival
            })

# Convert to DataFrame for analysis
df = pd.DataFrame(events)

print(f"\n📈 Total passage events reconstructed: {len(df)}")
print(f"Unique fish IDs: {df['fish_id'].nunique()}")
print(f"Unique days: {df['day'].nunique()}")
print(f"Iterations present: {sorted(df['iteration'].dropna().unique())}")

# Apply Policy B: Deduplicate by (fish_id, iteration, day)
print("\n" + "="*70)
print("POLICY B: Unique Fish Per Day Per Iteration")
print("="*70)

df_deduped = df.drop_duplicates(subset=['fish_id', 'iteration', 'day'])
print(f"\nAfter deduplication: {len(df_deduped)} unique fish instances")

# Calculate per-iteration route counts
if df_deduped['iteration'].notnull().any():
    per_iter_counts = df_deduped.groupby(['iteration', 'passage_route']).size().unstack(fill_value=0)
    mean_per_route = per_iter_counts.mean(axis=0).sort_values(ascending=False)
    
    print("\n🎯 Mean Fish Per Iteration (Policy B):")
    for route, count in mean_per_route.items():
        print(f"  {route}: {count:.1f} fish/iteration")
    
    total_mean = mean_per_route.sum()
    print(f"\n  TOTAL: {total_mean:.1f} fish/iteration")
    
    print("\n📊 Per-Iteration Breakdown:")
    print(per_iter_counts)
    
    # Calculate percentages
    print("\n📈 Passage Route Distribution:")
    for route, count in mean_per_route.items():
        pct = (count / total_mean) * 100
        print(f"  {route}: {pct:.1f}%")

# Survival analysis
print("\n" + "="*70)
print("SURVIVAL ANALYSIS BY ROUTE")
print("="*70)

if 'survival' in df_deduped.columns and df_deduped['survival'].notnull().any():
    survival_by_route = df_deduped.groupby('passage_route')['survival'].agg(['count', 'sum', 'mean'])
    survival_by_route.columns = ['Total Fish', 'Survived', 'Survival Rate']
    survival_by_route['Survival %'] = survival_by_route['Survival Rate'] * 100
    print("\n" + survival_by_route.to_string())
else:
    print("\nSurvival data not available in reconstructed events")
    print("(Survival flags may be in different positions in values_block_0)")

# Flow analysis
print("\n" + "="*70)
print("FLOW CONDITIONS")
print("="*70)

if 'flow' in df.columns and df['flow'].notnull().any():
    print(f"\nFlow range: {df['flow'].min():.1f} - {df['flow'].max():.1f} cfs")
    print(f"Mean flow: {df['flow'].mean():.1f} cfs")
    print(f"Median flow: {df['flow'].median():.1f} cfs")

print("\n" + "="*70)
print("✅ Analysis Complete!")
print("="*70)
