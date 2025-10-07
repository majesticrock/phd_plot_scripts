import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

BAND_WIDTH=300
DIR = "test"
main_df = load_panda("HHG", f"{DIR}/powerlaw1_laser/PiFlux", "occupations.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=BAND_WIDTH, 
                                  field_amplitude=1, photon_energy=5.25, 
                                  tau_diag=15, tau_offdiag=-1, t0=8))

nx, nz = main_df["upper_band"][0].shape
x = np.linspace(0, np.pi, nx, endpoint=False)
z = np.linspace(-np.pi, np.pi, nz, endpoint=False)

X, Z = np.meshgrid(x, z, indexing='ij')
grad = np.sin(Z) * np.cos(Z) / np.sqrt(np.cos(X)**2 + np.cos(Z)**2)

j_z = np.array([ np.sum(grad * (upper - lower)) for upper, lower in zip(main_df["upper_band"], main_df["lower_band"]) ])

fig, ax = plt.subplots()
time = np.linspace(0, 1, len(j_z))

ax.plot(time, j_z)

plt.show()