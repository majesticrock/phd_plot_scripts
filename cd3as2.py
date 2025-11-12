# Python script to compute and plot tight-binding bands for the fluorite (CaF2) structure
# Assumptions:
# - Conventional cubic cell (4 Ca + 8 F = 12 orbitals per cell)
# - One orbital per atom (single s-like orbital)
# - Only nearest-neighbour hopping (Ca <-> F). No Ca-Ca or F-F hopping.
# - Hopping amplitude t (negative favors bonding)
# - On-site energies eps_Ca and eps_F
#
# The script builds a k-dependent 12x12 Bloch Hamiltonian H(k) and plots bands along a
# standard high-symmetry path in the cubic BZ: Gamma - X - W - K - Gamma - L
#
# Output: a matplotlib band-structure plot.
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh

# lattice constant (set to 1 for convenience)
a = 1.0

# fractional basis positions in the conventional cubic cell (units of a)
# 4 Ca positions (FCC)
ca_frac = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
])

# 8 F positions (tetrahedral sites located at (1/4,1/4,1/4)-type positions)
f_frac = np.array([
    #[0.25, 0.25, 0.25],
    [0.25, 0.25, 0.75],
    [0.25, 0.75, 0.25],
    [0.75, 0.25, 0.25],
    [0.75, 0.75, 0.75],
    #[0.75, 0.75, 0.25],
    [0.75, 0.25, 0.75],
    [0.25, 0.75, 0.75],
])

# combine into single basis list: first Ca then F (so H is 12x12)
positions_frac = np.vstack([ca_frac, f_frac])
n_ca = ca_frac.shape[0]
n_f = f_frac.shape[0]
N = positions_frac.shape[0]

# physical parameters (adjustable)
eps_ca = 0.0     # on-site energy for Ca
eps_f = 0.0      # on-site energy for F (offset shows ionic character)
t_cf = -1.0      # nearest-neighbour Ca-F hopping (negative typical)
cutoff = 0.6 * a # cutoff distance to decide nearest neighbours (in units of a)

# real-space lattice vectors for the conventional cubic cell
lat_vecs = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]) * a

# convert fractional positions to cartesian (in units of a)
positions_cart = positions_frac @ lat_vecs

# find all Ca-F hopping pairs including periodic images within a small supercell
# We'll consider translations R = (nx,ny,nz) with nx,ny,nz in {-1,0,1}
image_range = range(-1,2)
hoppings = []  # list of tuples (i_ca, j_f, R_vector, delta_vector, distance)
for i_ca in range(n_ca):
    r_ca = positions_cart[i_ca]
    for j_f in range(n_ca, N):
        r_f_base = positions_cart[j_f]
        for nx in image_range:
            for ny in image_range:
                for nz in image_range:
                    R = np.array([nx,ny,nz]) @ lat_vecs
                    delta = r_f_base + R - r_ca
                    dist = np.linalg.norm(delta)
                    if dist < cutoff + 1e-9:  # nearest neighbour
                        hoppings.append((i_ca, j_f, np.array([nx,ny,nz]), delta, dist))

# print a quick summary of found hoppings
print(f"Found {len(hoppings)} Ca-F hopping links (including periodic images).")
# Each Ca is coordinated to 8 F, and there are 4 Ca -> 32 unique Ca->F within cell images
# depending on counting we should see ~32 entries.
# Build Hamiltonian H(k): H_ii = eps (Ca or F), H_ij = sum_R t * exp(i k·delta) for Ca-F pairs

def H_of_k(k_frac):
    """
    k_frac: k-vector in fractional coordinates of reciprocal lattice (i.e., numbers in [0,1) refer to multiples of 2π/a)
    returns Hermitian NxN Hamiltonian matrix H(k).
    """
    k_cart = 2*np.pi * np.array(k_frac)  # because reciprocal vectors are 2π/a * unit vectors and a=1
    H = np.zeros((N,N), dtype=complex)
    # on-site
    for i in range(n_ca):
        H[i,i] = eps_ca
    for j in range(n_ca, N):
        H[j,j] = eps_f
    # hoppings: Ca (i) -> F (j)
    for (i_ca, j_f, Rvec, delta, dist) in hoppings:
        phase = np.exp(1j * np.dot(k_cart, delta))
        H[i_ca, j_f] += t_cf * phase
        H[j_f, i_ca] += np.conj(t_cf * phase)  # hermitian
    return H

# High-symmetry k-points in fractional coordinates for cubic BZ (conventional cubic cell)
# Using coordinates in units of reciprocal-lattice basis (i.e., Gamma = (0,0,0), X = (0.5,0,0), L = (0.5,0.5,0.5) etc.)
HS = {
    'Γ': np.array([0.0,0.0,0.0]),
    'X': np.array([0.5,0.0,0.0]),
    'W': np.array([0.5,0.25,0.0]),
    'K': np.array([0.375,0.375,0.75]),  # conventional coordinate for K in cubic path
    'L': np.array([0.5,0.5,0.5])
}

# build path: Gamma - X - W - K - Gamma - L
path = [('Γ','X'),('X','W'),('W','K'),('K','Γ'),('Γ','L')]

# number of k-points per segment
nk_seg = 200

k_points = []
k_node_positions = [0]
k_node_labels = []
for (a_label,b_label) in path:
    a = HS[a_label]
    b = HS[b_label]
    for i in range(nk_seg):
        t = i/(nk_seg)
        k_points.append(a*(1-t) + b*t)
    k_node_positions.append(len(k_points))
    k_node_labels.append(a_label)
# append final label of last node
k_node_labels.append(path[-1][1])

k_points = np.array(k_points)
# evaluate bands
bands = []
for k in k_points:
    Hk = H_of_k(k)
    ev = eigvalsh(Hk)
    bands.append(ev.real)
bands = np.array(bands).T  # shape (Nbands, Nk)

# plotting
fig, ax = plt.subplots(figsize=(7,5))
kx = np.arange(bands.shape[1])
for n in range(bands.shape[0]):
    ax.plot(kx, bands[n], linewidth=1.2)

# set x-ticks at high-symmetry nodes
ax.set_xticks(k_node_positions)
ax.set_xticklabels(k_node_labels)
ax.set_xlim(0, bands.shape[1]-1)
ax.set_ylabel('Energy (arbitrary units)')
ax.set_title('Tight-binding bands for fluorite structure (one orbital/site, Ca-F hopping only)')
ax.grid(alpha=0.3, linestyle='--')

# draw vertical lines at node positions
for pos in k_node_positions:
    ax.axvline(pos, linestyle='--', linewidth=0.6)

plt.tight_layout()
plt.show()

# Print brief notes on parameters so the user knows what to change
print("\nParameters used:")
print(f"  eps_ca = {eps_ca}, eps_f = {eps_f}, t_cf = {t_cf}, cutoff = {cutoff} (units of a)")
print("Change these variables at the top of the script to explore different regimes (ionic limit, stronger hopping, etc.).")
