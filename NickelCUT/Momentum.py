import math
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Momentum:
    L: int
    x: int
    y: int = 0

    def __post_init__(self):
        if self.L % 2:
            raise ValueError("L must be even")
        object.__setattr__(self, "x", self.x % self.L)
        object.__setattr__(self, "y", self.y % self.L)

    @property
    def kx(self):
        return math.pi * (2 * self.x / self.L - 1)

    @property
    def ky(self):
        return math.pi * (2 * self.y / self.L - 1)

    @property
    def pos(self):
        return self.x + self.L * self.y

    def __index__(self):
        return self.x + self.L * self.y

    def __add__(self, other):
        assert self.L == other.L
        L = self.L
        return Momentum(
            L,
            (self.x + other.x + L // 2) % L,
            (self.y + other.y + L // 2) % L,
        )

    def __sub__(self, other):
        assert self.L == other.L
        L = self.L
        return Momentum(
            L,
            (self.x - other.x + L // 2) % L,
            (self.y - other.y + L // 2) % L,
        )

    def __neg__(self):
        return Momentum(self.L, self.L // 2, self.L // 2) - self

    def __repr__(self):
        return f"Momentum(idx=({self.x},{self.y}), k=({self.kx/math.pi:.3f}π,{self.ky/math.pi:.3f}π))"


def Gamma(L):
    return Momentum(L, L // 2, L // 2)

def Q(L):
    return Momentum(L, 0, 0)

class MomentumGrid:
    def __init__(self, L):
        if L % 2:
            raise ValueError("L must be even")

        self.L = L
        self.x, self.y = np.meshgrid(
            np.arange(L),
            np.arange(L),
            indexing="xy",
        )

    def _idx(self, x, y):
        return x + self.L * y

    def __sub__(self, other):
        L = self.L

        if isinstance(other, Momentum):
            # Shift the whole BZ
            x = (self.x - other.x + L//2) % L
            y = (self.y - other.y + L//2) % L
            return self._idx(x, y)

        elif isinstance(other, MomentumGrid):
            # Pairwise q - p
            x = (self.x.ravel()[None, :] - other.x.ravel()[:, None] + L//2) % L
            y = (self.y.ravel()[None, :] - other.y.ravel()[:, None] + L//2) % L
            return self._idx(x, y)

        return NotImplemented

    def __add__(self, other):
        if not isinstance(other, Momentum):
            return NotImplemented

        L = self.L
        x = (self.x + other.x + L // 2) % L
        y = (self.y + other.y + L // 2) % L
        return MomentumGrid(L, x, y)

    def __neg__(self):
        tmp = MomentumGrid(self.L)
        L = self.L
        tmp.x = (-self.x + L) % L
        tmp.y = (-self.y + L) % L
        return tmp

    def pos(self):
        return self._idx(self.x, self.y)
    
    def flat_pos(self):
        return self.pos().flatten()
    
    
class MomentumHalfFillingFS:
    def __init__(self, L):
        if L % 2:
            raise ValueError("L must be even")

        self.L = L
        self.x = np.concatenate([np.arange(L), [0], np.arange(L - 1, 0, -1)])
        self.y = np.concatenate([np.arange(L//2, L), [0], np.arange(L - 1, 0, -1), np.arange(L//2)])

    def _idx(self, x, y):
        return x + self.L * y

    def kx_ky(self):
        return np.array([math.pi * (2 * self.x / L - 1), math.pi * (2 * self.y / L - 1)]).T

    def __sub__(self, other):
        L = self.L

        if isinstance(other, Momentum):
            # Shift the whole BZ
            x = (self.x - other.x + L//2) % L
            y = (self.y - other.y + L//2) % L
            return self._idx(x, y)

        elif isinstance(other, MomentumHalfFillingFS):
            # Pairwise q - p
            x = (self.x.ravel() - other.x.ravel() + L//2) % L
            y = (self.y.ravel() - other.y.ravel() + L//2) % L
            return self._idx(x, y)

        return NotImplemented

    def __add__(self, other):
        if not isinstance(other, Momentum):
            return NotImplemented

        L = self.L
        x = (self.x + other.x + L // 2) % L
        y = (self.y + other.y + L // 2) % L
        return MomentumHalfFillingFS(L, x, y)

    def __neg__(self):
        tmp = MomentumHalfFillingFS(self.L)
        L = self.L
        tmp.x = (-self.x + L) % L
        tmp.y = (-self.y + L) % L
        return tmp

    def as_list_of_strings(self):
        return [
            f"Momentum(idx=({x},{y}))"
                for x, y in zip(self.x, self.y) ]

    def pos(self):
        return self._idx(self.x, self.y)
    
    def flat_pos(self):
        return self.pos().flatten()
    

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mc
    from load_full_flow_file import load_full_flow_file

    data = load_full_flow_file("cpp/NickelCUT/build/test")

    L = data["L"]

    # turns a L*L 1D array into a L x L 2D array
    def convert_1d_to_2d(arr):
        return np.reshape(arr, (-1, L))

    k_FS = MomentumHalfFillingFS(L)

    cmap = plt.get_cmap("inferno")
    norm = mc.Normalize(data["l_times"][0], data["l_times"][-1])
    fig, ax = plt.subplots()
    for i in range(data["number_of_data_points"]):
        if (i % 5 != 0):
            continue
        dispersion = convert_1d_to_2d(data["flow_states"][i]["dispersion"])
        ax.plot(dispersion[k_FS.x, k_FS.y].flatten(), c=cmap(norm(data["l_times"][i])))
    
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\varepsilon$")
    
    plt.show()