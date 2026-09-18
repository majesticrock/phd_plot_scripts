import numpy as np
import matplotlib.pyplot as plt

from Momentum import MomentumGrid, Momentum

class Lindhard:
    def __init__(self, L, beta, dispersion):
        self.L = L
        self.N = L*L
        self.beta = beta
        self.dispersion = dispersion
        self.momenta = MomentumGrid(L)

    def fermi(self, energy):
        return 1.0 / (1.0 + np.exp(self.beta * energy))
    def deriv_fermi(self, energy):
        return -0.25 * self.beta / (np.cosh(0.5 * self.beta * energy)**2)

    def lindhard_susceptibility_for_q(self, q: Momentum):
        """Return [chi0(q)]_{k} = [f(eps_k)-f(eps_{k+q})]/(eps_{k+q}-eps_k)."""
        eps_k        = self.dispersion[self.momenta.flat_pos()]
        eps_k_plus_q = self.dispersion[(self.momenta + q).flat_pos()]
        delta_eps = eps_k_plus_q - eps_k

        mask = np.abs(delta_eps) < 1e-10

        chi0_diag = np.zeros_like(delta_eps, dtype=float)
        chi0_diag[mask] = -self.deriv_fermi(eps_k[mask])
        np.divide(
            self.fermi(eps_k) - self.fermi(eps_k_plus_q),
            delta_eps,
            out=chi0_diag,
            where=~mask,
        )
        
        return chi0_diag

    def susceptibility_heatmap_for_all_q(self):
        """Return bare susceptibility maps over the full q-grid."""
        lindhard_map = np.zeros((self.L, self.L), dtype=float)

        for iy in range(self.L):
            for ix in range(self.L):
                q = Momentum(self.L, ix, iy)
                lindhard_map[iy, ix] = np.sum(self.lindhard_susceptibility_for_q(q)) / self.N
        return lindhard_map

    def plot_susceptibility_along_high_symmetry_lines(self, ax=None, **kwargs):
        """Plot the bare susceptibility along Gamma -> X -> M -> Gamma."""
        n = self.L // 2

        gamma_to_x = [Momentum(self.L, n, iy) for iy in range(n, -1, -1)]
        x_to_m = [Momentum(self.L, ix, 0) for ix in range(n, -1, -1)]
        m_to_gamma = [Momentum(self.L, i, i) for i in range(n + 1)]
        path = gamma_to_x + x_to_m[1:] + m_to_gamma[1:]

        susceptibility = np.asarray([
            np.sum(self.lindhard_susceptibility_for_q(q)) / self.N
            for q in path
        ])
        positions = np.arange(len(path))
        symmetry_positions = [0, n, 2 * n, 3 * n]

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        else:
            fig = ax.get_figure()
        ax.plot(positions, susceptibility, **kwargs)
        ax.set_xticks(symmetry_positions)
        ax.set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
        ax.set_xlabel(r"$\mathbf{q}$")
        ax.set_ylabel(r"$\chi(\mathbf{q})$")
        ax.grid(axis="x", linestyle=":")

        return fig, ax


    def plot_susceptibility_heatmap(self):
        """Plot bare Lindhard and RPA susceptibility across all q values."""
        lindhard_map = self.susceptibility_heatmap_for_all_q()

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

        vmin = np.min(lindhard_map)
        vmin = 0. if vmin > 0. else vmin
        im0 = ax.imshow(lindhard_map, origin="lower", cmap="magma", vmin=vmin, vmax=np.max(lindhard_map))
        ax.set_xlabel(r"$q_x$")
        ax.set_ylabel(r"$q_y$")
        fig.colorbar(im0, ax=ax, label=r"$\chi_0(\mathbf{q})$")

        ticks = np.array([0, self.L // 4, self.L // 2, 3 * self.L // 4])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        tick_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$"]
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        return fig, ax


if __name__ == "__main__":
    L=40
    BETA=2.5
    x = np.linspace(-np.pi, np.pi, L, endpoint=False)
    dispersion = -2 * ( np.cos(x[:,None]) + np.cos(x[None,:]) ).flatten()
    
    chi_Q_ref = 2. * (np.sum(np.divide(
    np.tanh(dispersion * BETA/2), 
        2. * dispersion, 
        where=np.abs(dispersion) > 1e-10
    )) + (np.abs(dispersion) <= 1e-10).sum() * BETA / 4.) / (L*L)
    print(chi_Q_ref)
    
    sus = Lindhard(L, BETA, dispersion)
    sus.plot_susceptibility_heatmap()
    
    sus.plot_susceptibility_along_high_symmetry_lines()
    plt.show()
