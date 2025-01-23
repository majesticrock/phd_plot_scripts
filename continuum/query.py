import __path_appender as __ap
__ap.append()

from get_data import load_panda, continuum_params
pd_data = load_panda("continuum", "offset_10", "resolvents.json.gz",
                    **continuum_params(N_k=20000, T=0, 
                                       coulomb_scaling=1, 
                                       screening=1e-4, 
                                       k_F=4.25, 
                                       g=1, 
                                       omega_D=10))

print(pd_data["Delta_max"])