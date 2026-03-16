import numpy as np
import matplotlib.pyplot as plt

def occupation(eps, Delta):
    return 0.5 * (1 - eps / np.sqrt(eps**2 + np.abs(Delta)**2))

def pairing(eps, Delta):
    return -0.5 * Delta / np.sqrt(eps**2 + np.abs(Delta)**2)

Delta0 = 1. / 20.
k = np.linspace(-np.pi, np.pi, 4000, endpoint=False)
eps = np.linspace(-1, 1, 4000) / Delta0 #(np.cos(k) ) / Delta0 

occ0 = occupation(eps, 1)
pair0 = pairing(eps, 1)

fig, axes = plt.subplots(nrows=3, sharex=True)

shifts = np.array([0.01])

for shift in shifts:
    occ_diff = -(occupation(eps, 1 + shift) - occ0) / shift
    pair_diff = (-(pairing(eps, 1 + shift) - pair0) / shift)
    pair_shift = (-np.imag(pairing(eps, 1 + 1j * shift)) / shift)**2
    
    axes[0].plot(k / np.pi, pair_diff, label=rf'${shift}$')
    axes[1].plot(k / np.pi, occ_diff, label=rf'${shift}$')
    axes[2].plot(k / np.pi, pair_shift, label=rf'${shift}$')

    comp = 1. / (eps**2 + 1)
    comp *= np.max(pair_diff) / np.max(comp)
    axes[0].plot(k / np.pi, comp, ls="--")
    comp *= np.max(pair_shift) / np.max(comp)
    axes[2].plot(k / np.pi, comp, ls="--")
    
    comp = 1. / (eps * np.sqrt(eps**2 + 1))
    comp *= np.max(occ_diff) / np.max(comp)
    axes[1].plot(k / np.pi, comp, ls="--")
    
    


axes[1].set_xlabel(r'$k / \pi$')
axes[1].set_ylabel(r'$\langle n \rangle_0 - \langle n \rangle_H$')
axes[0].set_ylabel(r'$\langle c c \rangle_0 - \langle c c \rangle_H$')

axes[0].legend()
fig.tight_layout()

plt.show()