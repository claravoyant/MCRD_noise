"""1D finite-difference grid and Laplacian, matching Appendix A's discretization.

The paper uses a periodic domain (Eq. A3): l_i = i*a, a = L/N. Boundary condition
for your system was NOT specified, so this module defaults to periodic (flagged in
run_comparison.py's output) but also offers a no-flux (Neumann) option -- switching
is a one-line change (`bc="neumann"`), since both conserve total mass under diagonal
diffusion and the rest of the solver doesn't care which one is in use.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

VALID_BCS = ("periodic", "neumann")


@dataclass(frozen=True)
class Grid:
    N: int
    L: float
    bc: str

    @property
    def dx(self) -> float:
        return self.L / self.N

    @property
    def x(self) -> np.ndarray:
        return np.arange(self.N) * self.dx


def build_laplacian(grid: Grid) -> sp.csr_matrix:
    """Second-order finite-difference Laplacian (without the 1/dx**2 division deferred
    to the caller -- this returns the *unscaled* second-difference operator scaled by
    1/dx**2, i.e. the actual discrete Laplacian).
    """
    if grid.bc not in VALID_BCS:
        raise ValueError(f"bc must be one of {VALID_BCS}, got {grid.bc!r}")

    N = grid.N
    main = -2.0 * np.ones(N)
    off = 1.0 * np.ones(N - 1)
    Lmat = sp.diags([off, main, off], offsets=[-1, 0, 1], format="lil")

    if grid.bc == "periodic":
        Lmat[0, N - 1] = 1.0
        Lmat[N - 1, 0] = 1.0
    else:  # neumann (no-flux): mirror ghost point, u_{-1} = u_1, u_N = u_{N-2}
        Lmat[0, 1] = 2.0
        Lmat[N - 1, N - 2] = 2.0

    Lmat = Lmat.tocsr() / grid.dx**2
    return Lmat
