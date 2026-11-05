import numpy as np
import matplotlib.pyplot as plt

from load_flow_files import load_flow_file
from hartree_fock import solve_hartree_fock
from nickel_cut_parameters import FLOW_PARAMETERS


dense = False
data = load_flow_file(**FLOW_PARAMETERS, resume_num="", force_json=False, dense=dense)

L = data["L"]
ELL_INDEX = 0 if dense else -1
hf_dispersion, chemical_potential, _ = solve_hartree_fock(
    data["extracted_channels"][ELL_INDEX],
    beta=data["beta"],
    target_filling=data["filling"],
)
dispersion = hf_dispersion


def extract_hoppings(dispersion, L):
    """Return symmetry-averaged Fourier coefficients on the finite periodic grid."""
    dispersion = np.asarray(dispersion).reshape(L, L)

    # The momentum samples run from -pi to pi, rather than from 0 to 2*pi.
    displacement = np.where(
        np.arange(L) <= L // 2,
        np.arange(L),
        np.arange(L) - L,
    )
    phase = (-1.0) ** (
        displacement[:, np.newaxis] + displacement[np.newaxis, :]
    )
    coefficients = np.fft.ifft2(dispersion) * phase

    def signed_index(index):
        return index if index <= L // 2 else index - L

    visited = set()
    hoppings = []
    for rx in range(L):
        for ry in range(L):
            if (rx, ry) in visited:
                continue

            orbit = {
                (sx * rx % L, sy * ry % L)
                for sx in (-1, 1)
                for sy in (-1, 1)
            }
            orbit.update(
                (sx * ry % L, sy * rx % L)
                for sx in (-1, 1)
                for sy in (-1, 1)
            )
            visited.update(orbit)

            dx, dy = min(
                tuple(sorted(
                    (abs(signed_index(ix)), abs(signed_index(iy))),
                    reverse=True,
                ))
                for ix, iy in orbit
            )
            # the sign is just convention; we use the same as Metzner in the fRG review
            value = -np.mean([coefficients[ix, iy] for ix, iy in orbit]).real
            hoppings.append((dx, dy, np.hypot(dx, dy), value, len(orbit)))

    return sorted(hoppings, key=lambda hopping: (hopping[2], hopping[0], hopping[1]))


def initial_hopping(dx, dy):
    distance_squared = dx * dx + dy * dy
    if distance_squared == 0:
        return data["E_F"]
    if distance_squared == 1:
        return 1.0
    if distance_squared == 2:
        return FLOW_PARAMETERS["tprime"]
    return 0.0


hoppings = extract_hoppings(dispersion, L)
print("Hartree-Fock chemical potential =", chemical_potential)
print("ROD of plotted data =", data["residual_offdiagonalities"][ELL_INDEX], ", index =", ELL_INDEX)
print("Symmetry-averaged Hartree-Fock hopping coefficients (displacement, distance, value):")
for dx, dy, distance, value, _ in hoppings:
    print(f"({dx:2d}, {dy:2d}), {distance:7.3f}, {value: .8g}")

fig, ax = plt.subplots()
ax.plot(
    [hopping[2] for hopping in hoppings],
    [hopping[3] for hopping in hoppings],
    marker="o",
    label="After Hartree-Fock",
)
final_minus_initial = [
    value - initial_hopping(dx, dy)
    for dx, dy, _, value, _ in hoppings
]
ax.plot(
    [hopping[2] for hopping in hoppings],
    final_minus_initial,
    marker="s",
    linestyle="--",
    label=r"Initial $-$ final",
)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel(r"Distance $|\mathbf{R}|$ (lattice spacings)")
ax.set_ylabel(r"Hopping coefficient $t_{\mathbf{R}}$")
ax.set_title("Hartree-Fock real-space hopping coefficients")
ax.legend()

ax.set_xlim(0, 5)

plt.show()
