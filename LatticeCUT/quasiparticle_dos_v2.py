import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
from scipy.ndimage import gaussian_filter1d

E_F = -0.5
SYSTEM = 'bcc'
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=2.5,
                                         U=0, 
                                         E_F=E_F,
                                         omega_D=0.02))

def quasiparticle_dos(energies, gaps, sp_dos, n_points=400, E_range=None, sigma=None):
    # restrict to nonzero gaps
    mask = gaps != 0
    energies = energies[mask]
    gaps = gaps[mask]
    sp_dos = sp_dos[mask]

    # quasiparticle dispersion (positive and negative)
    E_qp_pos = np.sqrt((energies - E_F)**2 + gaps**2)
    E_qp_neg = -E_qp_pos

    # combine positive and negative bands
    E_qp = np.concatenate([E_qp_pos, E_qp_neg])
    weights = np.concatenate([sp_dos, sp_dos])

    # determine output range
    if E_range is None:
        Emin, Emax = E_qp.min(), E_qp.max()
    else:
        Emin, Emax = E_range

    # make grid
    E_grid = np.linspace(Emin, Emax, n_points)

    # histogram with weights = sp_dos
    qp_dos, _ = np.histogram(E_qp, bins=n_points, range=(Emin, Emax), weights=weights, density=False)

    # normalize to bin width so it looks like a DOS
    bin_width = (Emax - Emin) / n_points
    qp_dos = qp_dos / bin_width

    if sigma is not None:
        qp_dos = gaussian_filter1d(qp_dos, sigma)

    return E_grid, qp_dos


E_grid, qp_dos = quasiparticle_dos(main_df["energies"], main_df["Delta"], main_df["dos"])

fig, ax = plt.subplots()
ax.plot(E_grid, qp_dos)
ax.set_xlabel("E")
ax.set_ylabel("Quasiparticle DOS")

fig.tight_layout()
plt.show()
