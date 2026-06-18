import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

#df = load_pickle(f"lattice_cut/T_C/bcc", "T_C.pkl").query("N==10000")
df = load_pickle(f"lattice_cut/bcc/N=16000", "resolvents.pkl")

#filtered = df.query("U==0.01 & E_F==-0.5 & omega_D==0.02 & g>2").sort_values('g')
#for _, row in filtered.iterrows():
#    print(row["g"])
    
filtered = df.query("E_F==-0.5 & omega_D==0.02 & g==2.2").sort_values('g')
for _, row in filtered.iterrows():
    print(row["U"])