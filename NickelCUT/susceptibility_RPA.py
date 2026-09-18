import numpy as np
import matplotlib.pyplot as plt

from Momentum import Momentum
from susceptibility_lindhard import Lindhard

class RPA:
    def __init__(self, L, beta, dispersion, full_vertex):
        self.L = L
        self.N = L * L
        self.full_vertex = full_vertex
        self.lindhard = Lindhard(L, beta, dispersion)

    def gamma_matrix_for_q(self, q: Momentum):
        """Return the interaction vertex for q."""
        if self.full_vertex.ndim == 2:
            return self.full_vertex
        return self.full_vertex[:, :, q.pos]

    def susceptibility_matrix_for_q(self, q: Momentum):
        """Solve chi_s(q) = [I - chi_0(q) Gamma_s(q)]^-1 chi_0(q)."""
        chi0 = np.diag(self.lindhard.lindhard_susceptibility_for_q(q))
        system_matrix = np.eye(self.N) - chi0 @ self.gamma_matrix_for_q(q)
        return np.linalg.inv(system_matrix) @ chi0

    def instability_finder(self, q: Momentum, fig=None, ax=None):
        chi0 = self.lindhard.lindhard_susceptibility_for_q(q)
        chi0_sqrt = np.sqrt(chi0)
        
        print(np.min(chi0), np.min(chi0_sqrt))
        
        system_matrix = np.diag(chi0_sqrt) @ self.gamma_matrix_for_q(q) @ np.diag(chi0_sqrt)
        
        print("||S-S^T||^2 =", np.sum((system_matrix - system_matrix.T)**2))
        
        eigenvalues, eigenvectors = np.linalg.eigh(system_matrix)
        max_idx = np.argmax(eigenvalues)
        
        ev_map = (eigenvectors[:,max_idx] / chi0_sqrt).reshape(L, L)
        vmax = np.max(ev_map)
        vmin = np.min(ev_map)
        
        if (fig is None):
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        image = ax.imshow(ev_map, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_xlabel(r"$q_x$")
        ax.set_ylabel(r"$q_y$")
        fig.colorbar(image, ax=ax, label=r"$v_\mathrm{max}$")
    
        ticks = np.array([0, self.L // 4, self.L // 2, 3 * self.L // 4])
        tick_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$"]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    
        ax.set_title(f"Largest eigenvalue = {eigenvalues[max_idx]}")
    
        return fig, ax
    

    def rpa_susceptibility_for_q(self, q: Momentum):
        return np.sum(self.susceptibility_matrix_for_q(q)) / self.N

    def susceptibility_heatmap_for_all_q(self):
        """Return the RPA susceptibility map over the full q-grid."""
        rpa_map = np.zeros((self.L, self.L), dtype=float)

        for iy in range(self.L):
            for ix in range(self.L):
                try:
                    rpa_map[iy, ix] = self.rpa_susceptibility_for_q(
                        Momentum(self.L, ix, iy)
                    )
                except np.linalg.LinAlgError:
                    rpa_map[iy, ix] = np.nan

        return rpa_map

    def plot_susceptibility_heatmap(self, fig=None, ax=None):
        """Plot the RPA susceptibility across the full q-grid."""
        rpa_map = self.susceptibility_heatmap_for_all_q()
        finite_values = rpa_map[np.isfinite(rpa_map)]
        vmax = np.max(finite_values) if finite_values.size else 1.0

        if (fig is None):
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        image = ax.imshow(rpa_map, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_xlabel(r"$q_x$")
        ax.set_ylabel(r"$q_y$")
        fig.colorbar(image, ax=ax, label=r"$\chi_s(\mathbf{q})$")

        ticks = np.array([0, self.L // 4, self.L // 2, 3 * self.L // 4])
        tick_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$"]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        return fig, ax

    def plot_susceptibility_along_high_symmetry_lines(self, ax=None, **kwargs):
        """Plot the RPA susceptibility along Gamma -> X -> M -> Gamma."""
        n = self.L // 2
        gamma_to_x = [Momentum(self.L, n, iy) for iy in range(n, -1, -1)]
        x_to_m = [Momentum(self.L, ix, 0) for ix in range(n, -1, -1)]
        m_to_gamma = [Momentum(self.L, i, i) for i in range(n + 1)]
        path = gamma_to_x + x_to_m[1:] + m_to_gamma[1:]

        susceptibility = np.asarray([
            self.rpa_susceptibility_for_q(q) for q in path
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
        ax.set_ylabel(r"$\chi (\mathbf{q})$")
        ax.grid(axis="x", linestyle=":")

        return fig, ax


if __name__ == "__main__":
    from load_full_flow_file import load_full_flow_state
    from nickel_cut_parameters import FLOW_PARAMETERS
    data = load_full_flow_state(subdir="", **FLOW_PARAMETERS, resume_num="", force_json=False)

    L = data["L"]
    N = L * L
    beta = 5.5 #1.0 / data["T"] if data["T"] > 0.0 else 32
    
    #full_vertex = 2.7 * np.ones((N, N)) / N
    #x = np.linspace(-np.pi, np.pi, L, endpoint=False)
    #dispersion = -2 * (np.cos(x[:, None]) + np.cos(x[None, :])).flatten()

    dispersion = data["epsilon_tilde"] + FLOW_PARAMETERS["U_0"] / 2
    full_vertex = (data["interactions_differing_spin"] - data["interactions_same_spin"])

    rpa = RPA(L, beta, dispersion, full_vertex)
    print("Filling:", np.sum(rpa.lindhard.fermi(dispersion)) / N)
    #rpa.plot_susceptibility_heatmap()
    #rpa.lindhard.plot_susceptibility_heatmap()
    
    fig_hs, ax_hs = rpa.plot_susceptibility_along_high_symmetry_lines(label="RPA")
    rpa.lindhard.plot_susceptibility_along_high_symmetry_lines(ax_hs, label="Lindhard")
    ax_hs.legend()
    
    rpa.instability_finder(Momentum(L, 0, 0))
    
    plt.show()