import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

from scipy.fft import rfftfreq

main_df = load_panda("HHG", "test", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=10, field_amplitude=1.6, photon_energy=5.25))

times = np.linspace(main_df["t_begin"], main_df["t_end"], main_df["n_measurements"], endpoint=False)

fig, ax = plt.subplots()    
ax.plot(times, main_df["alphas"] - main_df["betas"])

ax.set_xlabel(legend(r"t / T_\mathrm{L}"))
ax.set_ylabel(legend(r"j(\omega)"))

fig.tight_layout()
plt.show()