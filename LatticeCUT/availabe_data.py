from mrock.get_data import *
data_loader = DataLoader()

#df = data_loader.load_pickle(f"lattice_cut", f"T_C/bcc", "T_C.pkl").query("N==10000")
df = data_loader.load_pickle(f"lattice_cut", f"bcc/N=16000", "resolvents.pkl")

#filtered = df.query("U==0.01 & E_F==-0.5 & omega_D==0.02 & g>2").sort_values('g')
#for _, row in filtered.iterrows():
#    print(row["g"])
    
filtered = df.query("E_F==-0.5 & omega_D==0.02 & g==2.2").sort_values('g')
for _, row in filtered.iterrows():
    print(row["U"])