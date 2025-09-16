import numpy as np
import numpy.linalg as alg
from scipy import integrate
import matplotlib.pyplot as plt

S = 0.5
H0 = np.array([0., 0., 1.])
dt = 0.01
t_end = 40

class m_class:
    def __init__(self):
        self.history = [S * np.array([np.sin(np.pi - 0.1), 0, np.cos(np.pi - 0.1)]) ]
    
    def before_zero(self, t=None):
        return S * np.array([np.sin(np.pi - 0.1), 0, np.cos(np.pi - 0.1)]) 
    
    def get(self, t):
        if t < 0: 
            return self.before_zero(t)
        index = int(t // dt) 
        frac = (t % dt) / dt 
        if frac == 0 or index + 1 >= len(self.history):
            if index >= len(self.history) - 1:
                return self.history[-1]
            return self.history[index]
        return (1 - frac) * self.history[index] + frac * self.history[index + 1]
    
    def add(self, m):
        self.history.append(m)

def simulate(abscissae, TAU=0, J=2):
    print(f"tau={TAU}", f"J={J}")
    GAMMA_S = 0.2 * J * S
    def right_side(t, m, history):
        if TAU != 0:
            n_delayed = H0 + J * history.get(t - TAU)
        else:
            n_delayed = H0 + J * m
        norm = alg.norm(n_delayed)
        if norm > 1e-12:
            n_delayed /= norm

        term1_2 = np.cross(m, H0) + (2 * GAMMA_S * (S * n_delayed - m))
        term3 = GAMMA_S * (n_delayed * np.dot(n_delayed, m) - m * np.dot(n_delayed, n_delayed))
        return term1_2 - term3

    m_vec = m_class()
    ts = np.linspace(0, t_end, int(t_end / dt), endpoint=False)
    for t in ts:
        solution = integrate.solve_ivp(right_side, t_span=(t, t + dt), y0=m_vec.get(t), args=(m_vec, ), t_eval=[t + dt])
        m_vec.add(solution.y[:, -1])
        
    return np.array([m_vec.get(ab) for ab in abscissae])

Js = [2, 8, 100]
Taus = [0.1, 0.5]

fig, axes = plt.subplots(nrows=len(Taus), ncols=len(Js), sharex=True, sharey=True, figsize=(12.8, 9.6))
for ax, tau in zip(axes[:, 0], Taus):
    ax.set_ylabel(f"$m$, $\\tau={tau}$")
for ax in axes[-1]:
    ax.set_xlabel("$t$ $(1/h0)$")
for ax, J in zip(axes[0], Js):
    ax.set_title(f"$J={J}$")
t_plot = np.append(np.linspace(-5, 0, 20, endpoint=False), np.linspace(0, t_end, int(t_end / dt), endpoint=False))

for axs, TAU in zip(axes, Taus):
    for ax, J in zip(axs, Js):
        m_plot = simulate(abscissae=t_plot, TAU=0, J=J)
        ax.plot(t_plot, m_plot[:, 0], label="$m_x$, $\\tau=0$", ls="-")
        ax.plot(t_plot, m_plot[:, 2], label="$m_z$, $\\tau=0$", ls="-")

        m_plot = simulate(abscissae=t_plot, TAU=TAU, J=J)
        ax.plot(t_plot, m_plot[:, 0], label="$m_x$, delayed", ls="--")
        ax.plot(t_plot, m_plot[:, 2], label="$m_z$, delayed", ls="--")

axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
plt.show()