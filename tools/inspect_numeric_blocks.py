import h5py

h5 = r'C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\stryke\simulation_report_20251006_132251\same_old.h5'

def decode_bytes_array(arr):
    out = []
    for s in arr:
        try:
            ss = s.decode('utf-8') if isinstance(s, bytes) else str(s)
        except Exception:
            ss = str(s)
        ss = ss.strip('\x00')
        out.append(ss)
    return out

with h5py.File(h5, 'r') as f:
    ds = f['simulations']['summer']['Micropterus']['table']
    print('dtype names:', ds.dtype.names)
    entries = ds[:40]
    for i, rec in enumerate(entries):
        print('\n--- row', i, 'raw index:', rec['index'])
        for name in ds.dtype.names:
            val = rec[name]
            if isinstance(val, (bytes, bytearray)):
                print(name, 'bytes:', val)
            else:
                print(name, type(val), val)
        # decode last field which is array of bytes
        states = decode_bytes_array(rec[ds.dtype.names[-1]])
        print('decoded states:', states)
