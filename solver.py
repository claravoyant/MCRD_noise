"""Crank-Nicolson time-stepper for the 5-species Gef/Pak1 MCRD system, generalizing
Appendix A of the paper (Eqs. A5-A14) from 2 species to 5.

Because diffusion is diagonal (D = Diag(D_u, D_q, D_v, D_w, D_p), no cross-diffusion),
the linear (diffusive) part of the Crank-Nicolson update decouples into 5 independent
per-species linear solves -- exactly the structure of the paper's Au/Av, Bu/Bv blocks,
just with 5 blocks instead of 2. The nonlinear reaction term f(u_vec) couples the 5
species pointwise at each grid location (not across grid locations), so it's handled
with the same fixed-point (Picard) iteration the paper uses for its nonlinear phi term,
generalized to vector-valued f.

Noise (independent OU fields on u and q, from noise.py) is injected as an explicit
source term in the Crank-Nicolson right-hand side, trapezoidal-averaged like every
other explicit term in the scheme -- see the note at the top of noise.py.

ADAPTIVE STEP-HALVING: the fixed-point (Picard) iteration on the reaction term only
converges if dt/2 * |lambda_max(J_reaction)| stays below ~1, where J_reaction is the
*local* reaction Jacobian -- not just its value at the homogeneous equilibrium. Since
a * u**2 dominates that Jacobian, a large enough excursion in u (e.g. from unconstrained
noise) shrinks the safe dt substantially: at this system's fitted parameters, u only
needs to reach ~5x its equilibrium value before dt=0.0005's safety margin is gone (see
the eigenvalue-vs-u check that motivated this). Rather than requiring a single global dt
small enough for the worst excursion the *entire* run might ever produce -- which can be
far smaller than what's needed 99.9% of the time -- `run()` now calls `_advance`, which
retries a non-converging step at half the timestep (with its own freshly drawn OU noise
sub-steps, not a split of the already-drawn full-step noise), recursing down to `min_dt`
before giving up. This only pays the extra cost on the rare steps that actually need it.

This does NOT guarantee the underlying SDE itself stays bounded forever -- the reaction
term's a*u**2 autocatalysis has no upper saturation, and nothing constrains the OU noise
on (u, q). If a given noise realization genuinely drives the state to a finite-time
blow-up, adaptive step-halving will track that accurately (resolving right up to the
blow-up) rather than crash earlier from a Picard failure -- but it will still eventually
hit `min_dt` and raise, correctly, rather than run forever. That distinction (numerical
stiffness vs. a real blow-up) is exactly what this makes visible instead of masking.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from grid import Grid, build_laplacian
from model import ReactionParams, reaction
from noise import OUNoiseField, WhiteNoiseField

_SPECIES = ("u", "q", "v", "w", "p")
_NOISY_SPECIES_IDX = (0, 1)  # u, q


@dataclass
class SimulationResult:
    times: np.ndarray
    snapshots: np.ndarray  # shape (n_saved, 5, N)
    mean: np.ndarray  # shape (n_saved, 5) -- spatial mean per species over time
    std: np.ndarray  # shape (n_saved, 5) -- spatial std-dev per species over time
    mass_gef: np.ndarray  # shape (n_saved,) -- spatial average of u+q+v+w
    mass_pak: np.ndarray  # shape (n_saved,) -- spatial average of p+q
    grid: Grid
    noisy: bool


class MCRDSolver:
    def __init__(
        self,
        grid: Grid,
        D: dict,
        params: ReactionParams,
        dt: float,
        fp_tol: float = 1e-9,
        fp_maxiter: int = 50,
    ):
        """
        Parameters
        ----------
        grid : Grid
        D : dict mapping species name -> diffusion constant, e.g.
            {"u": D_u, "q": D_q, "v": D_v, "w": D_w, "p": D_p}
        params : ReactionParams
        dt : float, timestep
        """
        self.grid = grid
        self.params = params
        self.dt = dt
        self.fp_tol = fp_tol
        self.fp_maxiter = fp_maxiter

        missing = [s for s in _SPECIES if s not in D]
        if missing:
            raise ValueError(f"Missing diffusion constants for species: {missing}")
        self.D = np.array([D[s] for s in _SPECIES])

        self._L = build_laplacian(grid)
        self._I = sp.identity(grid.N, format="csr")
        self._op_cache: dict[float, tuple[list, list]] = {}
        self._get_diffusion_op(dt)  # eagerly build/cache the nominal dt, as before

    def _get_diffusion_op(self, dt: float):
        """(factorized_A_list, B_list) for a given dt -- built once and cached. Only
        ever called with something other than self.dt when a step is being bisected."""
        if dt not in self._op_cache:
            factorized, B = [], []
            for D_s in self.D:
                lam = D_s * dt / 2.0
                A_s = (self._I - lam * self._L).tocsc()
                B_s = self._I + lam * self._L
                factorized.append(spla.splu(A_s))
                B.append(B_s)
            self._op_cache[dt] = (factorized, B)
        return self._op_cache[dt]

    def _step(
        self, u_vec: np.ndarray, phi_n: np.ndarray, noise_rhs: np.ndarray | None, dt: float
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """One Crank-Nicolson step of size dt, with fixed-point iteration on the
        reaction term.

        Returns (u_next, phi_next, n_iterations_used).
        """
        factorized, B = self._get_diffusion_op(dt)
        base_rhs = np.stack(
            [B[s] @ u_vec[s] for s in range(5)], axis=0
        )
        if noise_rhs is not None:
            base_rhs = base_rhs + noise_rhs

        phi_k = phi_n.copy()
        u_new = u_vec.copy()
        for it in range(1, self.fp_maxiter + 1):
            rhs = base_rhs + (dt / 2.0) * (phi_n + phi_k)
            for s in range(5):
                u_new[s] = factorized[s].solve(rhs[s])
            phi_new = reaction(u_new, self.params)
            resid = np.max(np.abs(phi_new - phi_k))
            phi_k = phi_new
            if resid < self.fp_tol:
                return u_new, phi_new, it

        return u_new, phi_k, self.fp_maxiter

    def _advance(
        self,
        u_vec: np.ndarray,
        phi_n: np.ndarray,
        dt: float,
        noisy: bool,
        noise_fields: dict | None,
        min_dt: float,
        depth: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance (u_vec, phi_n) by dt. If the fixed-point iteration doesn't converge
        at this dt, roll back the OU noise fields (their state was already advanced by
        the failed attempt) and retry as two independent half-steps, each drawing its
        own fresh noise -- recursing until it converges or dt/2 < min_dt."""
        noise_rhs = None
        saved_noise_state = None
        if noisy:
            N = u_vec.shape[1]
            saved_noise_state = {idx: f.value.copy() for idx, f in noise_fields.items()}
            noise_rhs = np.zeros((5, N))
            for idx, field_ in noise_fields.items():
                eta_n = field_.value.copy()
                eta_np1 = field_.step(dt)
                noise_rhs[idx] = (dt / 2.0) * (eta_n + eta_np1)

        u_new, phi_new, n_iter = self._step(u_vec, phi_n, noise_rhs, dt)
        if n_iter < self.fp_maxiter:
            return u_new, phi_new

        if dt / 2 < min_dt:
            raise RuntimeError(
                f"Fixed-point iteration did not converge even at dt={dt:.3e} "
                f"(floor min_dt={min_dt:.3e}, {depth} bisection(s) deep). This is no "
                "longer a fixed-dt stiffness issue -- the state is likely undergoing a "
                "genuine finite-time blow-up (unbounded reaction autocatalysis + "
                "unconstrained noise, with nothing capping either)."
            )

        if noisy:
            for idx, field_ in noise_fields.items():
                field_.value = saved_noise_state[idx]

        print(f"  [adaptive] step not converging at dt={dt:.3e} (depth {depth}); "
              f"bisecting to dt={dt / 2:.3e}")
        half = dt / 2.0
        u_mid, phi_mid = self._advance(u_vec, phi_n, half, noisy, noise_fields, min_dt, depth + 1)
        return self._advance(u_mid, phi_mid, half, noisy, noise_fields, min_dt, depth + 1)

    def run(
        self,
        u0: np.ndarray,
        n_steps: int,
        *,
        save_every: int = 1,
        noisy: bool = False,
        noise_type: str = "ou",
        noise_A: float | None = None,
        noise_tau_c: float | None = None,
        noisy_species: tuple[int, ...] = _NOISY_SPECIES_IDX,
        seed: int | None = None,
        min_dt: float | None = None,
    ) -> SimulationResult:
        """Integrate n_steps forward from initial condition u0 (shape (5, N)).

        noise_type : "ou" (default) for the paper's temporally-correlated
            Ornstein-Uhlenbeck noise (requires noise_tau_c), or "white" for
            temporally-uncorrelated Gaussian noise at the same integrated strength A
            (see noise.py's WhiteNoiseField docstring for how A carries over between
            the two). Both are already spatially uncorrelated (iid per grid point).
        min_dt : smallest timestep _advance is allowed to bisect down to before
            raising (see _advance / the adaptive-step-halving note above). Defaults
            to self.dt / 1024 (10 halvings) if not given.
        """
        if min_dt is None:
            min_dt = self.dt / 1024
        u = u0.copy().astype(float)
        phi = reaction(u, self.params)
        N = self.grid.N

        rng = np.random.default_rng(seed)
        noise_fields = None
        if noisy:
            if noise_A is None:
                raise ValueError("noise_A is required when noisy=True")
            if noise_type == "ou":
                if noise_tau_c is None:
                    raise ValueError(
                        "noise_tau_c is required when noisy=True and noise_type='ou'"
                    )
                noise_fields = {
                    idx: OUNoiseField(N=N, A=noise_A, tau_c=noise_tau_c, rng=rng)
                    for idx in noisy_species
                }
            elif noise_type == "white":
                noise_fields = {
                    idx: WhiteNoiseField(N=N, A=noise_A, dt=self.dt, rng=rng)
                    for idx in noisy_species
                }
            else:
                raise ValueError(
                    f"Unknown noise_type: {noise_type!r} (expected 'ou' or 'white')"
                )

        n_saved = n_steps // save_every + 1
        times = np.zeros(n_saved)
        snapshots = np.zeros((n_saved, 5, N))
        mean = np.zeros((n_saved, 5))
        std = np.zeros((n_saved, 5))
        mass_gef = np.zeros(n_saved)
        mass_pak = np.zeros(n_saved)

        def record(idx, t, state):
            times[idx] = t
            snapshots[idx] = state
            mean[idx] = state.mean(axis=1)
            std[idx] = state.std(axis=1)
            mass_gef[idx] = (state[0] + state[1] + state[2] + state[3]).mean()
            mass_pak[idx] = (state[4] + state[1]).mean()

        record(0, 0.0, u)
        save_idx = 1

        for step in range(1, n_steps + 1):
            try:
                u, phi = self._advance(u, phi, self.dt, noisy, noise_fields, min_dt)
            except RuntimeError as ex:
                # u/phi here are still whatever the last step that DID succeed left behind
                # -- the failed step's `u, phi = ...` assignment never happened -- so this
                # is the exact pre-failure state, not just whatever was last saved on the
                # save_every grid (which can lag up to save_every-1 steps behind). Package
                # everything recorded so far, plus this final state, into a SimulationResult
                # and attach it to the exception rather than losing it: `except RuntimeError
                # as ex: partial = ex.partial_result`.
                last_t = (step - 1) * self.dt
                already_saved = save_idx > 0 and np.isclose(times[save_idx - 1], last_t)
                if already_saved:
                    partial_times, partial_snapshots = times[:save_idx], snapshots[:save_idx]
                    partial_mean, partial_std = mean[:save_idx], std[:save_idx]
                    partial_mass_gef, partial_mass_pak = mass_gef[:save_idx], mass_pak[:save_idx]
                else:
                    partial_times = np.concatenate([times[:save_idx], [last_t]])
                    partial_snapshots = np.concatenate([snapshots[:save_idx], u[None]], axis=0)
                    partial_mean = np.concatenate([mean[:save_idx], [u.mean(axis=1)]], axis=0)
                    partial_std = np.concatenate([std[:save_idx], [u.std(axis=1)]], axis=0)
                    partial_mass_gef = np.concatenate(
                        [mass_gef[:save_idx], [(u[0] + u[1] + u[2] + u[3]).mean()]]
                    )
                    partial_mass_pak = np.concatenate(
                        [mass_pak[:save_idx], [(u[4] + u[1]).mean()]]
                    )
                partial_result = SimulationResult(
                    times=partial_times,
                    snapshots=partial_snapshots,
                    mean=partial_mean,
                    std=partial_std,
                    mass_gef=partial_mass_gef,
                    mass_pak=partial_mass_pak,
                    grid=self.grid,
                    noisy=noisy,
                )
                err = RuntimeError(
                    f"...at step {step} (t={step * self.dt:.4f}): {ex}\n"
                    f"Trajectory up to t={last_t:.4f} ({partial_times.size} points, including "
                    "the exact pre-failure state) is preserved -- catch this exception and "
                    "read its `.partial_result` attribute rather than losing it."
                )
                err.partial_result = partial_result
                err.failed_step = step
                err.failed_time = step * self.dt
                raise err from ex

            if step % save_every == 0:
                record(save_idx, step * self.dt, u)
                save_idx += 1

        return SimulationResult(
            times=times,
            snapshots=snapshots,
            mean=mean,
            std=std,
            mass_gef=mass_gef,
            mass_pak=mass_pak,
            grid=self.grid,
            noisy=noisy,
        )
