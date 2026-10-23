"""Resource estimates at a target accuracy (noiseless circuits, finite shots).

Each Hadamard-test shot returns one +-1 outcome for Re g(k tau) or Im g(k tau). With M
shots, Re g is estimated as 2 * Binomial(M, (1 + Re g)/2) / M - 1 (same for Im). The
number of shots needed for a root-mean-square energy error (bias included) at the target
is found by Monte Carlo.

Shot allocation:
- Krylov D: M shots for each of the 2D circuits (Re and Im of g(k tau), k = 1..D).
- SPE degree d: a total budget M split over the odd lags j <= d in proportion to |c_j|
  (importance sampling, as in [3]), half for Re and half for Im of each g(j tau).

Cost metric: total number of controlled-evolution steps executed over all shots, which
is proportional to QPU time (the real-time analogue of [3]'s total walk-operator calls).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .krylov import krylov_energy
from .phase_est import SPEParams, spe_crossing_energy


def _sample(value: float, shots: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = np.clip((1 + value) / 2, 0, 1)
    return 2 * rng.binomial(shots, p) / np.maximum(shots, 1) - 1


def sampled_g(g_exact: np.ndarray, shots: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shot-noisy g; shots[k] = shots for Re and for Im of g(k tau) (0 = not measured)."""
    g = np.zeros_like(g_exact)
    g[0] = 1.0
    for k in range(1, len(g_exact)):
        if shots[k] > 0:
            g[k] = _sample(g_exact[k].real, shots[k], rng) + 1j * _sample(g_exact[k].imag, shots[k], rng)
    return g


def sampled_g_batch(g_exact: np.ndarray, shots: np.ndarray, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """n_trials independent shot-noisy copies of g (vectorized ``sampled_g``)."""
    k = np.flatnonzero(shots > 0)
    G = np.zeros((n_trials, len(g_exact)), complex)
    G[:, 0] = 1.0
    m = shots[k][None, :]
    for part, sign in ((g_exact[k].real, 1.0), (g_exact[k].imag, 1j)):
        pr = np.clip((1 + part) / 2, 0, 1)[None, :]
        G[:, k] += sign * (2 * rng.binomial(np.broadcast_to(m, (n_trials, len(k))), pr) / m - 1)
    return G


def krylov_shots(dim: int, shots_per_circuit: int, k_len: int) -> np.ndarray:
    s = np.zeros(k_len, dtype=int)
    s[1:dim + 1] = shots_per_circuit
    return s


def spe_shots(d: int, total: int, k_len: int, p0_lower_bound: float = 0.5) -> np.ndarray:
    p = SPEParams.from_degree(d, p0_lower_bound)
    w = np.abs(p.coefficients) / p.one_norm
    s = np.zeros(k_len, dtype=int)
    s[p.odd_degrees] = np.maximum(1, np.round(0.5 * total * w)).astype(int)  # per Re and per Im
    return s


def rmse(estimate_fn, g_exact, shots, e_exact, n_trials, rng) -> float:
    errs = np.array([estimate_fn(sampled_g(g_exact, shots, rng)) - e_exact for _ in range(n_trials)])
    return float(np.sqrt(np.mean(errs**2)))


def shots_for_target(make_shots, estimate_fn, g_exact, e_exact, target, grid, n_trials=300, seed=7):
    """Smallest budget on ``grid`` (log-interpolated) with RMSE <= target. Returns (budget, curve)."""
    rng = np.random.default_rng(seed)
    curve = [(int(m), rmse(estimate_fn, g_exact, make_shots(int(m)), e_exact, n_trials, rng)) for m in grid]
    for (m0, r0), (m1, r1) in zip(curve[:-1], curve[1:]):
        if r0 > target >= r1:
            t = (np.log(target) - np.log(r0)) / (np.log(r1) - np.log(r0))
            return float(np.exp(np.log(m0) + t * (np.log(m1) - np.log(m0)))), curve
    return None, curve


@dataclass
class CostSummary:
    algorithm: str
    setting: str
    distinct_circuits: int
    max_steps: int
    max_rzz: int
    max_duration_us: float
    shots_total: float | None
    total_steps_all_shots: float | None
    qpu_time_s: float | None  # sum of run durations over all shots (no reset/readout overhead)


def krylov_cost(dim, shots_per_circuit, step_ns, prep_ns, rzz_per_step) -> CostSummary:
    ks = np.arange(1, dim + 1)
    total_shots = 2 * dim * shots_per_circuit
    steps = 2 * shots_per_circuit * ks.sum()
    t = 2 * shots_per_circuit * np.sum(prep_ns + ks * step_ns) * 1e-9
    return CostSummary("Krylov (QKSD)", f"D = {dim}", 2 * dim, int(dim), int(rzz_per_step * dim),
                       (prep_ns + dim * step_ns) / 1e3, float(total_shots), float(steps), float(t))


def spe_cost(d, total_shots, step_ns, prep_ns, rzz_per_step, k_len, p0_lower_bound=0.5) -> CostSummary:
    s = spe_shots(d, int(total_shots), k_len, p0_lower_bound)
    j = np.arange(k_len)
    steps = 2 * np.sum(s * j)
    t = 2 * np.sum(s * (prep_ns + j * step_ns) * (s > 0)) * 1e-9
    n_circ = 2 * int(np.count_nonzero(s))
    return CostSummary("SPE (CDF)", f"d = {d}", n_circ, int(d), int(rzz_per_step * d), (prep_ns + d * step_ns) / 1e3,
                       float(2 * s.sum()), float(steps), float(t))


def krylov_estimator(dim, tau, c0, threshold=1e-8):
    return lambda g: krylov_energy(g, dim, tau, c0, threshold).energy


def spe_estimator(d, tau, c0, p0_lower_bound=0.5):
    return lambda g: spe_crossing_energy(g, d, tau, c0, p0_lower_bound).energy


# ---------------------------------------------------------------------------
# Direct validation of a shot budget with many trials (catches rare large errors)
# ---------------------------------------------------------------------------


def error_stats(errors: np.ndarray, success: np.ndarray | None = None) -> dict:
    a = np.abs(errors)
    out = {"rmse": float(np.sqrt(np.mean(errors**2))), "median_abs": float(np.median(a)),
           "frac_above_10mHa": float(np.mean(a > 1e-2)), "max_abs": float(a.max()), "n_trials": int(len(errors))}
    if success is not None:
        out["frac_failed"] = float(np.mean(~success))
    return out


def krylov_trial_errors(dim, shots_per_circuit, g_exact, e_exact, tau, c0, n_trials, seed):
    rng = np.random.default_rng(seed)
    s = krylov_shots(dim, shots_per_circuit, len(g_exact))
    est = krylov_estimator(dim, tau, c0)
    G = sampled_g_batch(g_exact, s, n_trials, rng)
    return np.array([est(row) - e_exact for row in G]), None


def spe_trial_errors(d, total_shots, g_exact, e_exact, tau, c0, n_trials, seed, p0_lower_bound=0.5):
    from .phase_est import spe_crossing_energy_batch

    rng = np.random.default_rng(seed)
    s = spe_shots(d, total_shots, len(g_exact), p0_lower_bound)
    errs, oks = [], []
    for a in range(0, n_trials, 2000):  # bounded memory
        G = sampled_g_batch(g_exact, s, min(2000, n_trials - a), rng)
        e, ok = spe_crossing_energy_batch(G, d, tau, c0, p0_lower_bound)
        errs.append(e - e_exact)
        oks.append(ok)
    return np.concatenate(errs), np.concatenate(oks)


def validate_budget(trial_fn, start: float, target: float, n_trials: int, seed: int = 11, step: float = 1.1,
                    max_iter: int = 25) -> dict:
    """Increase the budget from ``start`` by factors of ``step`` until RMSE (n_trials) <= target.

    Returns the validated budget, its statistics, and the statistics at the starting value.
    """
    m = float(start)
    first = None
    history = []
    for _ in range(max_iter):
        errs, ok = trial_fn(int(np.ceil(m)), n_trials, seed)
        st = error_stats(errs, ok)
        history.append({"budget": int(np.ceil(m)), **st})
        if first is None:
            first = st
        if st["rmse"] <= target:
            return {"budget": int(np.ceil(m)), "stats": st, "start_budget": int(np.ceil(start)), "start_stats": first,
                    "history": history}
        m *= step
    raise RuntimeError(f"budget not validated after {max_iter} increases (last RMSE {st['rmse']:.3g})")
