import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

E_F = -0.5
SYSTEM = 'bcc'
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=2.0,
                                         U=0.0, 
                                         E_F=E_F,
                                         omega_D=0.02))

energies = main_df['energies']
sp_dos = main_df['dos']
gaps = main_df['Delta']
E = np.sqrt((energies - E_F)**2 + gaps**2)

normal_state_filling = np.sum(sp_dos[energies < E_F]) * (energies[1] - energies[0])


expec_n_in_sc = 0.5 * (1 - (energies - E_F) / E)
sc_state_filling = np.sum(sp_dos * expec_n_in_sc) * (energies[1] - energies[0])

print(normal_state_filling)
print(sc_state_filling)