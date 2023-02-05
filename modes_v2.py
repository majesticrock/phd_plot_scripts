import matplotlib.pyplot as plt
import numpy as np

nameU = "-5.00"

M = np.loadtxt(f"data/T0.1/U_modes/{nameU}.txt").transpose()
#M = 1 / M

N = np.loadtxt(f"data/T0.1/U_modes/{nameU}_one_particle.txt").transpose().flatten()

M.sort()
N.sort()

sigma = 0.03
def gauss(x, mu):
    return np.exp((-(x - mu)**2) / (2*sigma)) / np.sqrt(2*np.pi*sigma)

lims = [min(M) - 1, max(M) + 1]
#lims = [-12, 12]
size = M.size
x_lin = np.linspace(lims[0], lims[1], size)

data = np.zeros(size)
for i in range(0, len(M)):
    data += gauss(x_lin, M[i])

data /= max(data)
data *= 16

plt.plot(x_lin, data, label="DoS")
plt.plot(M, np.linspace(min(data), max(data), len(M)), "x", label="Spectrum")
#plt.plot(x_lin, N, "o", mfc="none", label="One Particle")

#plt.xlim(lims[0], lims[1])
#plt.ylim(-20, 20)
plt.xlabel(r"$\epsilon / t$")
plt.ylabel(r"DoS /  a.u.")
plt.legend()
plt.tight_layout()

import os
plt.savefig(f"python/build/{os.path.basename(__file__).split('.')[0]}_U={nameU}.pdf")
plt.show()
