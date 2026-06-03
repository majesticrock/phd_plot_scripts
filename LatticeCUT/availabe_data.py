import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

df = load_pickle(f"lattice_cut/./T_C/bcc", "T_C.pkl")

filtered = df.query("N==10000 & U==0.0 & E_F==-0.5 & omega_D==0.05 & g>1.0 & g<1.3").sort_values('g')

print(filtered["g"])