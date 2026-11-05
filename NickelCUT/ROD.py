import matplotlib.pyplot as plt
from load_flow_files import *
from nickel_cut_parameters import FLOW_PARAMETERS
import numpy as np

data = load_flow_file(**FLOW_PARAMETERS, force_json=False, dense=False)

fig, ax = plt.subplots()

l_times = data["l_times"]
ax.plot(l_times, data["residual_offdiagonalities"], "-o", label=r"$\mathcal{H}$")
ax.plot(l_times, data["max_interactions"] / 10, "-s", label=r"$\max |U|$ / 10")
ax.plot(l_times, data["derivative_residual_offdiagonalities"] / 10, "-x", label="$\partial_\ell \mathcal{H}$ / 10")

print("l_max =", l_times[-1])

ax.set_xlabel(r"$\ell \cdot t$")
ax.set_ylabel(r"$\mathrm{ROD} / t$")

ax.set_ylim(0, None)
ax.set_xlim(0, l_times[-1])
ax.legend()

fig.tight_layout()

k = np.linspace(-np.pi, np.pi, data["L"], endpoint=False)
cos_x, cos_y = np.meshgrid(np.cos(k), np.cos(k))
energies = -2. * (
    (cos_x + cos_y) 
    + 2 * FLOW_PARAMETERS["tprime"] * cos_x * cos_y
) - data["E_F"]

integrated_ROD = np.trapezoid(data["residual_offdiagonalities"]**2, l_times) * data["L"] * data["L"]
H0_norm = np.sqrt(np.sum(energies**2))
print("Integrated ROD =", integrated_ROD, "||H_0|| =", H0_norm, "   relative =", integrated_ROD / H0_norm)

plt.show()