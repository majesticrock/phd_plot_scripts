import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
N=10000
df   = load_all(f"lattice_cut/./T_C/bcc/N={N}/", "T_C.json.gz").sort_values('omega_D')

def find_near_duplicates(arr, tol=1e-6):
    arr = np.array(arr, dtype=float)
    arr_sorted = np.sort(arr)
    diffs = np.diff(arr_sorted)
    dup_mask = np.abs(diffs) < tol
    if not np.any(dup_mask):
        return False, []
    
    # collect the pairs that are too close
    duplicates = []
    for i in np.where(dup_mask)[0]:
        duplicates.append((arr_sorted[i], arr_sorted[i+1]))
    
    # optionally, flatten/unique the duplicate values
    flat_dupes = np.unique(np.round(np.array(duplicates).flatten(), 6))
    return True, flat_dupes.tolist()

# Apply the function and unpack results
df["has_duplicates"], df["duplicate_values"] = zip(*df["temperatures"].apply(find_near_duplicates))

# Filter only those with duplicates
filtered_df = df[df["has_duplicates"]]
print(filtered_df)