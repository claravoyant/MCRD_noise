"""Reaction kinetics for the Gef/Pak1 mass-conserving reaction-diffusion (MCRD) model.

State vector u_vec = (u, q, v, w, p):
    u - membrane-bound Gef
    q - membrane-bound Gef + Pak1
    v - dephosphorylated cytosolic Gef
    w - phosphorylated cytosolic Gef
    p - cytosolic Pak1

Reaction kinetics (paper's Eq. (2)-equivalent for this system):

    f1 = v*(a*u**2 + f) - b*u - c*u*p        # du/dt reaction part
    f2 = c*u*p - d*q                          # dq/dt
    f3 = -v*(a*u**2 + f) + b*u + e*w          # dv/dt
    f4 = d*q - e*w                            # dw/dt
    f5 = -c*u*p + d*q                         # dp/dt

f sums to zero pointwise over (f1+f2+f3+f4) and over (f5+f2), which is exactly what
guarantees n_GEF = u+q+v+w and n_pak = p+q are conserved by the reaction term alone,
independent of the diffusion discretization or boundary condition.
"""

from dataclasses import dataclass

import numpy as np

_SPECIES = ("u", "q", "v", "w", "p")


@dataclass(frozen=True)
class ReactionParams:
    """Reaction-rate constants. Fill in with fitted/known values before trusting results."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


def reaction(u_vec: np.ndarray, params: ReactionParams) -> np.ndarray:
    """Evaluate f(u_vec) pointwise.

    Parameters
    ----------
    u_vec : ndarray, shape (5, N)
        Rows are (u, q, v, w, p); columns are spatial grid points. Also accepts a
        flat shape (5,) for a single (homogeneous) state.
    params : ReactionParams

    Returns
    -------
    ndarray, same shape as u_vec
    """
    u, q, v, w, p = u_vec
    a, b, c, d, e, f = params.a, params.b, params.c, params.d, params.e, params.f

    react = a * u**2 + f
    f1 = v * react - b * u - c * u * p
    f2 = c * u * p - d * q
    f3 = -v * react + b * u + e * w
    f4 = d * q - e * w
    f5 = -c * u * p + d * q

    return np.stack([f1, f2, f3, f4, f5], axis=0)


def jacobian(u_vec: np.ndarray, params: ReactionParams) -> np.ndarray:
    """Analytical Jacobian d f_i / d u_j, evaluated pointwise.

    Returns
    -------
    ndarray, shape (N, 5, 5) if u_vec has shape (5, N), or (5, 5) for a single state.
    Row i, column j is d f_i / d u_j, in species order (u, q, v, w, p).
    """
    u, q, v, w, p = u_vec
    a, b, c, d, e, f = params.a, params.b, params.c, params.d, params.e, params.f

    scalar_input = np.asarray(u).ndim == 0
    u_arr, v_arr, p_arr = np.atleast_1d(u), np.atleast_1d(v), np.atleast_1d(p)
    n = u_arr.shape[0]

    react = a * u_arr**2 + f
    J = np.zeros((n, 5, 5))

    # row 0: d f1/d(u,q,v,w,p)
    J[:, 0, 0] = 2 * a * u_arr * v_arr - b - c * p_arr
    J[:, 0, 2] = react
    J[:, 0, 4] = -c * u_arr

    # row 1: d f2/d(...)
    J[:, 1, 0] = c * p_arr
    J[:, 1, 1] = -d
    J[:, 1, 4] = c * u_arr

    # row 2: d f3/d(...)
    J[:, 2, 0] = -2 * a * u_arr * v_arr + b
    J[:, 2, 2] = -react
    J[:, 2, 3] = e

    # row 3: d f4/d(...)
    J[:, 3, 1] = d
    J[:, 3, 3] = -e

    # row 4: d f5/d(...)
    J[:, 4, 0] = -c * p_arr
    J[:, 4, 1] = d
    J[:, 4, 4] = -c * u_arr

    return J[0] if scalar_input else J


def equilibrium_from_u(u: float, params: ReactionParams, N_pak: float) -> np.ndarray:
    """Closed-form (q, v, w, p) given u at homogeneous reactive+mass equilibrium.

    NOTE: the (v, w, p) formulas here are corrected relative to the vector as
    originally transcribed. Cross-checking against fsolve on the raw 5-equation
    system (f1=f2=f3=0, q+p=N_pak, u+q+v+w=N_GEF) showed the original v/w/p
    formulas were each other's values shifted by one slot -- e.g. the expression
    originally labeled "v" (d*N_pak/(c*u+d)) is algebraically p (it depends only on
    the Pak1-binding constants c, d, which makes sense for free Pak1; it should not,
    and structurally cannot, depend on the Gef activation/phosphorylation constants
    a, b, e, f -- which is what made the mislabeling detectable in the first place).
    The mapping below (q unchanged; v <- old "w" formula; w <- old "p" formula;
    p <- old "v" formula) was verified to zero the reaction residuals and hit the
    target masses to machine precision for multiple parameter sets.
    """
    a, b, c, d, e, f = params.a, params.b, params.c, params.d, params.e, params.f
    denom = c * u + d
    q = N_pak * c * u / denom
    p = d * N_pak / denom
    v = (b * u * denom + d * N_pak * c * u) / ((a * u**2 + f) * denom)
    w = d * N_pak * c * u / (e * denom)
    return np.array([u, q, v, w, p])


def solve_homogeneous_equilibrium(
    N_GEF: float, N_pak: float, params: ReactionParams, *, tol: float = 1e-8
) -> list[np.ndarray]:
    """Solve for homogeneous reactive-equilibrium states of the well-mixed system.

    Solves the quartic in u derived from f(u_vec) = 0 subject to q+p = N_pak and
    u+q+v+w = N_GEF, then reconstructs (q, v, w, p) from each valid root of u.

    Returns
    -------
    list of ndarray, each shape (5,) -- one per physically valid (real, positive,
    finite) equilibrium found. The paper's analogous derivation notes at most 3 real
    positive roots occur in practice, though up to 4 are possible in principle.
    """
    a, b, c, d, e, f = params.a, params.b, params.c, params.d, params.e, params.f

    c4 = a * c
    c3 = a * d + N_pak * a * c + d * N_pak * a * c / e - N_GEF * a * c
    c2 = f * c + b * c - N_GEF * a * d
    c1 = (
        f * d
        + N_pak * c * f
        + b * d
        + d * N_pak * c
        + d * N_pak * c * f / e
        - N_GEF * f * c
    )
    c0 = -N_GEF * f * d

    roots = np.roots([c4, c3, c2, c1, c0])

    equilibria = []
    for r in roots:
        if abs(r.imag) > tol * max(abs(r.real), 1.0):
            continue
        u = r.real
        if u <= 0:
            continue
        state = equilibrium_from_u(u, params, N_pak)
        if np.any(~np.isfinite(state)) or np.any(state < -tol):
            continue
        # Sanity check: does this state actually zero the reaction term and hit the
        # mass targets? (Catches algebra/sign slips rather than silently trusting the
        # closed form.)
        resid = reaction(state, params)
        mass_gef = state[0] + state[1] + state[2] + state[3]
        mass_pak = state[4] + state[1]
        if (
            np.max(np.abs(resid[:3])) > 1e-6 * max(1.0, np.max(np.abs(state)))
            or abs(mass_gef - N_GEF) > 1e-6 * max(1.0, N_GEF)
            or abs(mass_pak - N_pak) > 1e-6 * max(1.0, N_pak)
        ):
            continue
        equilibria.append(state)

    return equilibria
