import numpy as np
import matplotlib.pyplot as plt
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
rho_F = 2*rho[np.argmin(np.abs(xi))] # factor 2 because of the spins

a = 1.0
N = 150
beta = 30
eta = 1e-3
q_dependence = True
omega_dependence = False

def fermi(x):
    return 1.0 / (np.exp(beta * x) + 1.0)

b1 = (2*np.pi/a) * np.array([ 0, 1, 1])
b2 = (2*np.pi/a) * np.array([ 1, 0, 1])
b3 = (2*np.pi/a) * np.array([ 1, 1, 0])

coeffs = (np.arange(N) - N/2) / N
X, Y, Z = np.meshgrid(coeffs, coeffs, coeffs, indexing="ij")
k = (
    X[..., None]*b1 +
    Y[..., None]*b2 +
    Z[..., None]*b3
)
cx = np.cos(k[...,0]*a/2)
sx = np.sin(k[...,0]*a/2)
cy = np.cos(k[...,1]*a/2)
sy = np.sin(k[...,1]*a/2)
cz = np.cos(k[...,2]*a/2)
sz = np.sin(k[...,2]*a/2)

def xi_from_k(k):
    return -(np.cos(k[...,0]*a/2) *
             np.cos(k[...,1]*a/2) *
             np.cos(k[...,2]*a/2)) - E_F
    
def xi_shifted(q):
    qx, qy, qz = q * a/2

    cq_x, sq_x = np.cos(qx), np.sin(qx)
    cq_y, sq_y = np.cos(qy), np.sin(qy)
    cq_z, sq_z = np.cos(qz), np.sin(qz)

    xi_q = -(
        (cx*cq_x - sx*sq_x) *
        (cy*cq_y - sy*sq_y) *
        (cz*cq_z - sz*sq_z)
    ) - E_F

    return xi_q

xi_k = xi_from_k(k)
f_k = fermi(xi_k)

def chi_q0_static(xi_k):
    fprime = -beta * np.exp(beta*xi_k) / (np.exp(beta*xi_k) + 1)**2
    return -2 * np.mean(fprime)

def full_lindhard(q, omega):
    xi_q = xi_shifted(q)
    f_q = fermi(xi_q)

    num = f_k - f_q
    den = omega + 1j*eta + xi_k - xi_q

    return -2 * np.mean(num / den)

def chi(q, omega):
    qnorm = np.linalg.norm(q)
    if qnorm < eta:
        if omega < eta:
            return chi_q0_static(xi_k)
        else:
            xi_q = xi_shifted(q)
            return -2. * np.mean(f_k * (xi_q - xi_k)) / omega**2
    return full_lindhard(q, omega)


if q_dependence:
    G = np.array([0,0,0])
    X = 0.5 * b1
    M = 0.5 * (b1 + b2)
    R = 0.5 * (b1 + b2 + b3)

    segments = [
        (G, X),
        (X, M),
        (M, R),
        (R, G),
    ]

    q_path = []
    tick_pos = [0]
    tick_labels = [r'$\Gamma$']

    for start, end in segments:
        seg = np.linspace(start, end, 100, endpoint=False)
        q_path.append(seg)
        tick_pos.append(tick_pos[-1] + len(seg))
        tick_labels.append(None)  # placeholder

    # append final endpoint
    q_path.append([segments[-1][1]])
    q_path = np.vstack(q_path)
    tick_labels = [r'$\Gamma$', 'X', 'M', 'R', r'$\Gamma$']

    fig, ax = plt.subplots()

    for o, omega in enumerate([0., 0.1, 0.5]):
        chi_vals = np.array([chi(q, omega) for q in q_path])
        x = np.arange(len(chi_vals))

        ax.plot(x, np.real(chi_vals), label=f"$\\omega={omega}$", c=f"C{o}", ls="-")

    ax.axhline(rho_F, ls=":", c="k", label=r"$\rho_F$")

    # vertical separators
    for p in tick_pos:
        ax.axvline(p, color='k', linestyle=':', alpha=0.6)

    # x ticks at symmetry points
    ax.set_xticks(tick_pos, tick_labels)
    ax.set_ylabel(r"$\Re \chi_0(q,\omega)$")
    ax.set_xlabel(r"$qa$")
    ax.legend()
    fig.tight_layout()
    plt.show()
    
if omega_dependence:
    fig, ax = plt.subplots()
    omegas = np.linspace(0, 0.5, 200)
    
    for q_ratio in [0.0001, 0.001, 0.01, 0.1]:
        Q = q_ratio * b1
        chis = np.array([chi(Q, omega) for omega in omegas])
        ax.plot(omegas, chis, label=f"$c={q_ratio}$")
    
    ax.set_ylabel(r"$\Re \chi_0(q,\omega)$")
    ax.set_xlabel(r"$\omega$")
    ax.legend()
    fig.tight_layout()
    plt.show()