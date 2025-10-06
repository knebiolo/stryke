import h5py, os, sys
h5 = r'C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\stryke\simulation_report_20251006_132251\same_old.h5'
if not os.path.exists(h5):
    print('HDF not found at', h5); sys.exit(1)
print('Opening', h5)
with h5py.File(h5, 'r') as f:
    def recurse(g, indent=0):
        for k, v in g.items():
            prefix = '  ' * indent
            if isinstance(v, h5py.Group):
                print(f"{prefix}Group: {k} (members={len(v)})")
                recurse(v, indent+1)
            else:
                # dataset
                print(f"{prefix}Dataset: {k} shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
    recurse(f)
    print('\nTop-level keys:', list(f.keys()))
    # Try to inspect /Daily if present
    if 'Daily' in f:
        try:
            d = f['Daily']
            print('\n/Daily group members:', list(d.keys()))
            # if this is a table, find 'block0_values' or similar
            for k in d.keys():
                obj = d[k]
                print('  ', k, type(obj), getattr(obj, 'shape', None))
        except Exception as e:
            print('Error reading /Daily:', e)
    # Look for simulations group
    sims = [k for k in f.keys() if k.startswith('simulations') or k.startswith('/simulations')]
    # h5py strips leading slash so use any group containing 'simulations'
    sims = [k for k in f.keys() if 'simulations' in k]
    print('\nCandidates for simulations groups at top-level or deeper:')
    def find_groups(g, path=''):
        found=[]
        for k,v in g.items():
            p = f"{path}/{k}" if path else k
            if isinstance(v, h5py.Group):
                if 'simulations' in k:
                    found.append(p)
                found += find_groups(v, p)
        return found
    sim_groups = find_groups(f)
    print(sim_groups)
    # If there is a simulations group deeper, print its members
    for sg in sim_groups:
        print('\nInspecting', sg)
        grp = f[sg]
        for k,v in grp.items():
            print('  ', k, 'type', 'Group' if isinstance(v,h5py.Group) else 'Dataset', 'shape', getattr(v,'shape',None))
print('Done')
