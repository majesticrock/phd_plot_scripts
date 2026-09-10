import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Momentum:
    """
    A single momentum on an L x L periodic momentum grid.

    The momentum indices are

        x, y = 0, ..., L-1

    and the physical momenta are

        kx = pi * (2*x/L - 1)
        ky = pi * (2*y/L - 1).

    The linear position in an L x L array is

        pos = x + L*y.
    """

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
        """
        Linear position in an L x L array.
        """
        return self.x + self.L * self.y

    def flat_pos(self):
        """
        One-element 1D array containing the linear position.
        """
        return np.asarray(self.pos).reshape(-1)

    def __index__(self):
        return self.pos

    def __add__(self, other):
        if not isinstance(other, Momentum):
            return NotImplemented

        if self.L != other.L:
            raise ValueError(
                "Cannot add momenta with different lattice sizes"
            )

        L = self.L

        return Momentum(
            L,
            (self.x + other.x + L // 2) % L,
            (self.y + other.y + L // 2) % L,
        )

    def __sub__(self, other):
        if not isinstance(other, Momentum):
            return NotImplemented

        if self.L != other.L:
            raise ValueError(
                "Cannot subtract momenta with different lattice sizes"
            )

        L = self.L

        return Momentum(
            L,
            (self.x - other.x + L // 2) % L,
            (self.y - other.y + L // 2) % L,
        )

    def __neg__(self):
        return Momentum(
            self.L,
            self.L // 2,
            self.L // 2,
        ) - self

    def __repr__(self):
        return (
            f"Momentum("
            f"idx=({self.x},{self.y}), "
            f"k=({self.kx / math.pi:.3f}π,"
            f"{self.ky / math.pi:.3f}π)"
            f")"
        )


def Gamma(L):
    """
    Gamma = (0, 0).
    """
    return Momentum(L, L // 2, L // 2)


def Q(L):
    """
    Q = (pi, pi), represented by the index (0, 0).
    """
    return Momentum(L, 0, 0)


class MomentumGrid:
    """
    The complete L x L momentum grid represented as a flat array
    of L**2 momenta.

    This representation is particularly convenient for Bethe-Salpeter
    calculations because a momentum variable can be turned into a
    row or column variable simply with

        K = MomentumGrid(L)[:, None]
        P = MomentumGrid(L)[None, :]

    giving

        K.shape == (L**2, 1)
        P.shape == (1, L**2)

    and hence

        K - P

    has shape

        (L**2, L**2).
    """

    def __init__(self, L, x=None, y=None):
        if L % 2:
            raise ValueError("L must be even")

        self.L = L

        if x is None and y is None:
            # Flattened momentum grid.
            #
            # pos = x + L*y
            #
            # First x varies fastest, then y.
            self.x = np.tile(np.arange(L), L)
            self.y = np.repeat(np.arange(L), L)

        elif x is None or y is None:
            raise ValueError(
                "x and y must either both be provided or both be None"
            )

        else:
            self.x = np.asarray(x) % L
            self.y = np.asarray(y) % L

            # Allow NumPy broadcasting between x and y.
            self.x, self.y = np.broadcast_arrays(
                self.x,
                self.y,
            )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.x.shape

    @property
    def size(self):
        return self.x.size

    # ------------------------------------------------------------------
    # Physical momentum
    # ------------------------------------------------------------------

    @property
    def kx(self):
        return math.pi * (2 * self.x / self.L - 1)

    @property
    def ky(self):
        return math.pi * (2 * self.y / self.L - 1)

    # ------------------------------------------------------------------
    # Linear position
    # ------------------------------------------------------------------

    @property
    def pos(self):
        """
        Linear momentum index.

        Preserves the shape of the MomentumGrid.

        Examples:

            MomentumGrid(L).pos.shape
                -> (L**2,)

            MomentumGrid(L)[:, None].pos.shape
                -> (L**2, 1)

            MomentumGrid(L)[None, :].pos.shape
                -> (1, L**2)
        """
        return self.x + self.L * self.y

    def flat_pos(self):
        """
        Return the momentum indices as a flat 1D array.
        """
        return self.pos.reshape(-1)

    # ------------------------------------------------------------------
    # NumPy-style indexing
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """
        Preserve MomentumGrid under NumPy-style slicing.

        Examples
        --------
        K0 = MomentumGrid(L)

        K0[:, None].shape
            -> (L**2, 1)

        K0[None, :].shape
            -> (1, L**2)
        """
        return MomentumGrid(
            self.L,
            self.x[key],
            self.y[key],
        )

    # ------------------------------------------------------------------
    # Momentum arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        L = self.L

        if isinstance(other, Momentum):
            if self.L != other.L:
                raise ValueError(
                    "Cannot add momenta with different lattice sizes"
                )

            return MomentumGrid(
                L,
                (self.x + other.x + L // 2) % L,
                (self.y + other.y + L // 2) % L,
            )

        if isinstance(other, MomentumGrid):
            if self.L != other.L:
                raise ValueError(
                    "Cannot add momentum grids with different lattice sizes"
                )

            return MomentumGrid(
                L,
                (self.x + other.x + L // 2) % L,
                (self.y + other.y + L // 2) % L,
            )

        return NotImplemented

    def __sub__(self, other):
        L = self.L

        if isinstance(other, Momentum):
            if self.L != other.L:
                raise ValueError(
                    "Cannot subtract momenta with different lattice sizes"
                )

            return MomentumGrid(
                L,
                (self.x - other.x + L // 2) % L,
                (self.y - other.y + L // 2) % L,
            )

        if isinstance(other, MomentumGrid):
            if self.L != other.L:
                raise ValueError(
                    "Cannot subtract momentum grids with different lattice sizes"
                )

            return MomentumGrid(
                L,
                (self.x - other.x + L // 2) % L,
                (self.y - other.y + L // 2) % L,
            )

        return NotImplemented

    def __neg__(self):
        L = self.L

        return MomentumGrid(
            L,
            (-self.x + L) % L,
            (-self.y + L) % L,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"MomentumGrid("
            f"L={self.L}, "
            f"shape={self.shape}, "
            f"size={self.size}"
            f")"
        )