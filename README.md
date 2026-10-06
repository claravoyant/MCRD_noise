# Gef/Pak1 mass-conserving reaction-diffusion model with Ornstein-Uhlenbeck active noise

Solver for a 5-species Gef/Pak1 mass-conserving reaction-diffusion (MCRD) model in 1D,
with optional additive noise on the membrane-bound species. The noise model follows
Dao et al., *Noise-Induced Localized Patterns in Excitable Media: Amplitude versus Persistence*
(arXiv/journal: **TODO**), adapted to this model (see "Differences from the reference scheme").

All quantities (concentration, time, length) are in nondimensional model units.

## Layout

```
src/
  grid.py     1D grid and finite-difference Laplacian (periodic or Neumann/no-flux)
  model.py    reaction kinetics, Jacobian, homogeneous equilibria
  noise.py    OU noise field (OUNoiseField) and white-noise control (WhiteNoiseField)
  solver.py   Crank-Nicolson stepper with fixed-point iteration and adaptive step halving
examples/
  run_example.py   short demo run with the parameters below
```

## Quick start

```bash
pip install -r requirements.txt
python examples/run_example.py                 # demo, T = 5
python examples/run_example.py --T 1000        # full-length run (2e6 steps; slow)
python examples/run_example.py --white         # white-noise control
python examples/run_example.py --A 0.02 --tau_c 1.0 --out run.npz
```

## Model

State (u, q, v, w, p): membrane Gef, membrane Gef+Pak1, dephosphorylated cytosolic Gef,
phosphorylated cytosolic Gef, cytosolic Pak1. Reactions (`src/model.py`):

```
f_u = v(a u^2 + f) - b u - c u p
f_q = c u p - d q
f_v = -v(a u^2 + f) + b u + e w
f_w = d q - e w
f_p = -c u p + d q
```

The reaction terms conserve N_GEF = u+q+v+w and N_PAK = p+q. Diffusion is diagonal.

## Noise

Noise is added to the equations for u and q only (v, w, p receive none directly):

    ds/dt = D_s d^2 s/dl^2 + f_s + eta_s(l,t),   s in {u, q}

Each eta_s is an independent Ornstein-Uhlenbeck process, uncorrelated in space:

    d eta/dt = -eta/tau_c + (sqrt(2A)/tau_c) xi(l,t),   <xi xi'> = delta(l-l') delta(t-t')

giving stationary statistics `<eta eta'> = delta(l-l') (A/tau_c) exp(-|t-t'|/tau_c)`, where
`A = sigma_eta^2 tau_c` is the integrated noise strength. Same A and tau_c for both species.

- **Discretization:** Euler-Maruyama, `eta^{n+1} = (1 - dt/tau_c) eta^n + sqrt(2 A dt)/tau_c * zeta`,
  `zeta ~ N(0,1)` i.i.d. per grid point and species.
- **Initialization:** eta is drawn from its stationary distribution N(0, A/tau_c).
- **White-noise control** (`noise_type="white"`): fresh draw each step,
  `eta = sqrt(2A/dt) zeta`, so the integrated strength A is comparable to the OU case.

## Numerics

- Crank-Nicolson for diffusion and reaction; nonlinear reaction term solved by fixed-point
  (Picard) iteration, tolerance 1e-9 (sup norm of the change in the reaction term), max 100 iterations.
- Noise enters the Crank-Nicolson right-hand side as a trapezoidal average,
  `(dt/2)(eta^n + eta^{n+1})`. eta is advanced once per step, before the fixed-point iteration,
  since it does not depend on the fields.
- **Adaptive step halving:** if the iteration does not converge, the step is retried as two
  half steps (fresh noise drawn for each), down to `dt/1024`, after which a `RuntimeError`
  is raised (interpreted as finite-time blow-up). The exception carries `.partial_result`.

## Parameters (parameter set 1)

| Quantity | Value |
|---|---|
| a, b, c, d, e, f | 2.38/4, 0.185, 151.6/2, 0.552, 0.828, 1.0 |
| D_u = D_q | 0.05 |
| D_v = D_w = D_p | 5.0 |
| N_GEF, N_PAK | 2.0, 0.492 |
| L, N, boundary | 7.0, 200, Neumann (no-flux) |
| dt | 5e-4 |
| T | 1000 (2e6 steps) |
| tau_c | 1.0 |
| A | 0.005, 0.01, 0.015, 0.02 (sweep) |
| Initial condition | homogeneous state x (1 + 0.01 * N(0,1)) per species and grid point |
| Replicates | 10 per A |

Parameter set 2 (L = 3, N_GEF = 6.5, N_PAK = 2.0, A in {0, 0.001, 0.01, 0.1}):
**TODO: add reaction/diffusion parameters and dt.**

## Reproducibility notes

- Sweep seeds: initial condition `default_rng(3000 + rep + 100000*attempt)`, noise
  `seed = round(A*1e6) + rep + 100000*attempt`.
- Replicates that blew up (step halving hit the floor) were discarded and rerun with the next
  attempt's seeds, up to 20 attempts. Reported statistics therefore condition on non-blow-up
  runs. **TODO: report how many runs were discarded per A.**

## Caveats

- **Mass conservation is broken by the noise.** The reaction terms and diffusion conserve
  N_GEF and N_PAK, but additive zero-mean noise on u and q does not. Mass is conserved only on
  average; `SimulationResult.mass_gef` / `mass_pak` track the drift.
- **No positivity constraint** is imposed, so concentrations can go negative.
- **Grid dependence:** eta is i.i.d. per grid point with variance A/tau_c independent of the
  grid spacing, so the effective noise depends on N (N = 200 throughout).
- The reaction term has unbounded autocatalysis (a u^2); strong noise can cause blow-up.

## Differences from the reference scheme

Dao et al. describe adding the noise increment after the Crank-Nicolson step converges and
do not fully specify the operator-splitting order. Here the noise is included inside the
Crank-Nicolson update as described above. The OU noise model itself (Eqs. 2-4 and A1 of the
reference) is unchanged.

## Citation

**TODO:** cite this repository and Dao et al.
