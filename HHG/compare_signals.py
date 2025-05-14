import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

import current_density_fourier as cdf

fig, ax = cdf.create_frame()


main_df = load_panda("HHG", "test_sthread/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=100))
cdf.add_current_density_to_plot(main_df, ax, "Base", shift=1)
main_df = load_panda("HHG", "test0/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=100))
cdf.add_current_density_to_plot(main_df, ax, "MPI base", shift=1, ls="--")
main_df = load_panda("HHG", "test_openmp_base/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=100))
cdf.add_current_density_to_plot(main_df, ax, "openMP base", shift=1, ls="-.")



cdf.add_verticals(main_df["frequencies"],ax)

ax.legend(loc="upper right")
fig.tight_layout()
plt.show()