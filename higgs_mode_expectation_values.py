import numpy as np
import matplotlib.pyplot as plt

def occupation(eps, Delta):
    return 0.5 * (1 - eps / np.sqrt(eps**2 + Delta**2))

def pairing(eps, Delta):
    return -0.5 * Delta / np.sqrt(eps**2 + Delta**2)

Delta0 = 1. / 200.
k = np.linspace(-np.pi, np.pi, 4000, endpoint=False)
eps = (np.cos(k) ) / Delta0 

occ0 = occupation(eps, 1)
pair0 = pairing(eps, 1)

fig, axes = plt.subplots(nrows=2, sharex=True)
fig_r, axes_r = plt.subplots(nrows=2, sharex=True)

shifts = np.array([0.01, 0.5])

for shift in shifts:
    occ_diff = -(occupation(eps, 1 + shift) - occ0) / shift
    pair_diff = -(pairing(eps, 1 + shift) - pair0) / shift
    
    axes[1].plot(k / np.pi, occ_diff, label=rf'${shift}$')
    axes[0].plot(k / np.pi, pair_diff, label=rf'${shift}$')

    occ_real = np.fft.ifft(np.fft.ifftshift(occ_diff))
    pair_real = np.fft.ifft(np.fft.ifftshift(pair_diff))

    axes_r[1].plot(np.arange(len(occ_real)),   occ_real.real, label=rf'${shift}$')
    axes_r[0].plot(np.arange(len(pair_real)), pair_real.real, label=rf'${shift}$')
    

axes[1].set_xlabel(r'$k / \pi$')
axes[1].set_ylabel(r'$\langle n \rangle_0 - \langle n \rangle_H$')
axes[0].set_ylabel(r'$\langle c c \rangle_0 - \langle c c \rangle_H$')

axes[0].legend()
fig.tight_layout()

axes_r[1].set_xlabel('Site index')
axes_r[1].set_ylabel(r'$\langle n \rangle_0 - \langle n \rangle_H$')
axes_r[0].set_ylabel(r'$\langle c c \rangle_0 - \langle c c \rangle_H$')
axes_r[0].legend()
fig_r.tight_layout()

plt.show()