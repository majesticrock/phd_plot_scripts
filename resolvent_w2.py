import numpy as np
import matplotlib.pyplot as plt
import gzip
# Calculates the resolvent in w^2

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

Ts = np.array([0.])
Us = np.array([-1.0, -1.25, -1.5, -1.75, -2.0])
Vs = np.array([-0.5])

folder = "data/L=50/"
name_suffix = "SC"
fig, ax = plt.subplots()
types = ["higgs", "phase"]#
lss = ["-", "--", "-."]

plot_upper_lim = 2

for q, T in enumerate(Ts):
    for r, U in enumerate(Us):
        for s, V in enumerate(Vs):
            name = f"T={T}/U={U}_V={V}/"

            file = f"{folder}{name}one_particle.dat.gz"
            with gzip.open(file, 'rt') as f_open:
                one_particle = np.abs(np.loadtxt(f_open).flatten())

            roots = np.array([np.min(one_particle) * 2, np.max(one_particle) * 2])**2
            a_inf = (roots[0] + roots[1]) * 0.5
            b_inf = ((roots[1] - roots[0]) * 0.25)

            #ax.axvspan(np.sqrt(roots[1]), np.sqrt(roots[0]), alpha=.2, color="purple", label="Continuum")
            for idx, type in enumerate(types):
                file = f"{folder}{name}resolvent_{type}_{name_suffix}.dat.gz"
                with gzip.open(file, 'rt') as f_open:
                    M = np.loadtxt(f_open)
                    A = M[0]
                    B = M[1]

                w_vals = 10000
                w_lin = np.linspace(0, plot_upper_lim, w_vals, dtype=complex)**2
                w_lin += 1e-3j
                off = 1

                def terminator(w):
                    p = w - a_inf
                    q = 4 * b_inf**2
                    root = np.sqrt(np.real(p**2 - q), dtype=complex)
                    return_arr = np.zeros(len(w), dtype=complex)
                    for i in range(0, len(w)):
                        if(w[i] > roots[0]):
                            return_arr[i] = (p[i] - root[i]) / (2. * b_inf**2)
                        else:
                            return_arr[i] = (p[i] + root[i]) / (2. * b_inf**2)

                    return return_arr

                deviation_from_inf = np.zeros(len(A) - 1)
                for i in range(0, len(A) - 1):
                    deviation_from_inf[i] = abs((A[i] - a_inf) / a_inf) + abs((np.sqrt(B[i + 1]) - b_inf) / b_inf)

                off_termi = len(A) - 1 - np.argmin(deviation_from_inf)
                print("Terminating at i=", np.argmin(deviation_from_inf))
                def dos(w):
                    for i in range(0, len(w)):
                        if(w[i].real > roots[0] and w[i].real < roots[1]):
                            w[i] = w[i].real

                    G = w - A[len(A) - off_termi] - B[len(B) - off_termi] * terminator( w )
                    for j in range(len(A) - off_termi - 1, -1, -1):
                        G = w - A[j] - B[j + 1] / G
                    return B[0] / G

                if(idx == 0):
                    ax.plot(np.sqrt(w_lin.real), -dos( np.copy(w_lin) ).imag, color=colors[q+r+s],
                        linestyle=lss[idx], linewidth=(plt.rcParams["lines.linewidth"]+idx*2),
                        label=f"$U={U}$")
                else:
                    ax.plot(np.sqrt(w_lin.real), -dos( np.copy(w_lin) ).imag, color=colors[q+r+s],
                        linestyle=lss[idx], linewidth=(plt.rcParams["lines.linewidth"]+idx*2))
                #ax.plot(B, "x", label=folder)
                #        #label=f"$V={V}$")

ax.set_yscale("log")
ax.legend()
ax.set_xlabel(r"$\epsilon / t$")
fig.tight_layout()

import os
plt.savefig(f"python/build/{os.path.basename(__file__).split('.')[0]}.pdf")
plt.show()
