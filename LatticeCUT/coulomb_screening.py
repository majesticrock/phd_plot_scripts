import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'bcc'
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=0,
                                         U=0, 
                                         E_F=0,
                                         omega_D=0.02))

E_F = -0.5
xi  = main_df['energies'] - E_F
rho = main_df['dos']

# let's assume W=1eV
# Further assume a unit cell 1x1x1 angstroms^3 -> volume = 1 angstrom^3
# alpha = e^2 / (epsilon_0) = 180.951282 angstroms * eV
N_k = 100
W = 1.
a = 1.0
alpha = 180.951282 / a**3
rho_F = rho[np.argmin(np.abs(xi))] / W
screening = 2 * rho_F * alpha

b1 = (2*np.pi/a) * np.array([ 1, 1,-1])
b2 = (2*np.pi/a) * np.array([ 1,-1, 1])
b3 = (2*np.pi/a) * np.array([-1, 1, 1])

total = 0.
for i in range(N_k):
    for j in range(N_k):
        for k in range(N_k):

            u = (i + 0.5)/N_k - 0.5
            v = (j + 0.5)/N_k - 0.5
            w = (k + 0.5)/N_k - 0.5

            kvec = u*b1 + v*b2 + w*b3

            total += alpha / (np.sum(kvec**2) + screening)
total /= N_k**3
mu = total * rho_F
print("Integral over first BZ =", total, "eV       or unitless = ", mu)

mu_star = mu / (1 + mu * np.log((E_F + 1) / (0.04)))
print("mu^* = ", mu_star)



Gs = []
for n1 in range(-1,2):
    for n2 in range(-1,2):
        for n3 in range(-1,2):
            if n1 == n2 == n3 == 0:
                continue
            Gs.append(n1*b1+n2*b2+n3*b3)

Gs = np.array(Gs)
def inside_BZ(k,k2):
    k2 = np.dot(k,k)
    return np.all(np.sum((k-Gs)**2,axis=1) >= k2)

def f(k2):
    return alpha / (np.sum(k2**2) + screening)
N = 1000000

kmax = 3*np.pi/a
vals = np.zeros(N)
accepted = 0

while accepted < N:
    k = np.random.uniform(-kmax,kmax,3)
    k2 = np.dot(k,k)
    if inside_BZ(k, k2):
        vals[accepted] = alpha / (np.sum(k2) + screening)
        accepted += 1

average = np.mean(vals)
mu = average * rho_F
print("Integral over first BZ =", average, "eV       or unitless = ", mu)

mu_star = mu / (1 + mu * np.log((E_F + 1) / (0.04)))
print("mu^* = ", mu_star)