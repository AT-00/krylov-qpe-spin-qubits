"""Statistical phase estimation (SPE) from g(k tau): CDF + binary search, as in [3].

The cumulative spectral distribution of |HF> in the rescaled variable x = tau E',

    C(x) = sum_k p_k Theta(x - tau E'_k),    p_k = |<E_k|HF>|^2,

is approximated by replacing Theta with a periodic Heaviside smoothed by a Gaussian of
width sigma (a scaled error function) and truncated to odd Fourier modes j <= d:

    F(y)   = 1/2 + sum_{j odd <= d} c_j sin(j y),   c_j = 2/(pi j) * exp(-j^2 sigma^2 / 2)
    C~(x)  = 1/2 + sum_{j odd <= d} c_j Im[ e^{i j x} g(j tau) ].

Only g at odd multiples of tau up to d (the maximal number of controlled steps) is needed,
matching the index set S1 = {0} U {2j+1} of [3]. Parameters follow the guarantee used
there (Lin & Tong): with eta a known lower bound on p0/2 and eps = eta/8,

    sigma = sqrt(2 ln(1/eps)) / d          (truncation error ~ eps)
    delta = sigma sqrt(2) erfcinv(2 eps)  (smoothing error <= eps outside +-delta)

and the binary search below then returns x* with |x* - tau E'_0| <= delta. This bound is
the standard guarantee for the error-function construction; it is checked here only in the
tested regime (default p0 lower bound 0.5, the H2 data in tests/), not proven for every
parameter choice. [3] uses a Chebyshev/Bessel construction with tighter bounds, which do
not transfer automatically to this Gaussian-smoothed step.
Requires tau |E'| < pi/3 (guaranteed by tau = pi / (3 lambda)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfcinv


@dataclass(frozen=True)
class SPEParams:
    d: int  # maximal degree = maximal number of controlled steps (odd)
    eta: float  # lower bound on p0, divided by 2
    sigma: float
    delta: float
    eps: float

    @staticmethod
    def from_degree(d: int, p0_lower_bound: float = 0.5) -> "SPEParams":
        if d < 1 or d % 2 == 0:
            raise ValueError("d must be a positive odd integer")
        eta = p0_lower_bound / 2
        eps = eta / 8
        sigma = np.sqrt(2 * np.log(1 / eps)) / d
        delta = sigma * np.sqrt(2) * float(erfcinv(2 * eps))
        return SPEParams(d=d, eta=eta, sigma=float(sigma), delta=float(delta), eps=eps)

    @property
    def odd_degrees(self) -> np.ndarray:
        return np.arange(1, self.d + 1, 2)

    @property
    def coefficients(self) -> np.ndarray:
        j = self.odd_degrees
        return 2 / (np.pi * j) * np.exp(-0.5 * (j * self.sigma) ** 2)

    @property
    def one_norm(self) -> float:
        """F = sum |c_j| (sets the sample count M = 2 F^2 / eta^2 in [3])."""
        return float(np.sum(np.abs(self.coefficients)))


def cdf_estimate(x, g: np.ndarray, params: SPEParams) -> np.ndarray:
    """C~(x) from g[k] = g(k tau), k = 0..>=d."""
    if len(g) < params.d + 1:
        raise ValueError(f"need g up to k = {params.d}")
    j = params.odd_degrees
    x = np.atleast_1d(np.asarray(x, dtype=float))
    phases = np.exp(1j * np.outer(x, j)) * g[j][None, :]
    return 0.5 + (params.coefficients[None, :] * phases.imag).sum(axis=1)


@dataclass
class SPEResult:
    energy: float  # Ha, including c0
    x: float  # rescaled ground-state estimate
    params: SPEParams
    n_iterations: int
    success: bool = True  # False: no interior crossing (estimate pinned at the search-window edge)


def spe_energy(g: np.ndarray, d: int, tau: float, c0: float = 0.0, p0_lower_bound: float = 0.5) -> SPEResult:
    """Binary search of Lin & Tong as used in [3]: threshold 3 eta / 4, shifts 2 delta / 3."""
    p = SPEParams.from_degree(d, p0_lower_bound)
    x_left, x_right = -np.pi / 3, np.pi / 3
    n = 0
    while x_right - x_left > 2 * p.delta:
        x_mid = 0.5 * (x_left + x_right)
        if cdf_estimate(x_mid, g, p)[0] > 0.75 * p.eta:
            x_right = x_mid + 2 * p.delta / 3  # ground state is left of x_mid + delta
        else:
            x_left = x_mid - 2 * p.delta / 3  # ground state is right of x_mid - delta
        n += 1
        if n > 200:
            break
    x_star = 0.5 * (x_left + x_right)
    return SPEResult(energy=float(x_star / tau + c0), x=float(x_star), params=p, n_iterations=n)


def spe_crossing_energy(g: np.ndarray, d: int, tau: float, c0: float = 0.0, p0_lower_bound: float = 0.5,
                        grid: int = 4001) -> SPEResult:
    """Continuous readout: the smallest x with C~(x) = 3 eta / 4 (the point the binary search brackets).

    The bisection output moves in jumps when g changes (it is a quantizer of resolution
    ~delta), so its finite differences are zero or a jump. The crossing point x_c is a
    smooth function of g and stays within delta of the bisection result; it is the
    readout used for noise sweeps and sensitivities.
    """
    from scipy.optimize import brentq

    p = SPEParams.from_degree(d, p0_lower_bound)
    level = 0.75 * p.eta
    xs = np.linspace(-np.pi / 3, np.pi / 3, grid)
    c = cdf_estimate(xs, g, p) - level
    idx = np.flatnonzero((c[:-1] <= 0) & (c[1:] > 0))
    if c[0] > 0 or len(idx) == 0:
        # C~ is already above the level at the left edge, or never crosses it: no interior
        # crossing. The edge value is returned (as before) and flagged as a failure.
        x_star, ok = (xs[0] if c[0] > 0 else xs[-1]), False
    else:
        i = idx[0]
        x_star, ok = brentq(lambda x: cdf_estimate(x, g, p)[0] - level, xs[i], xs[i + 1], xtol=1e-14), True
    return SPEResult(energy=float(x_star / tau + c0), x=float(x_star), params=p, n_iterations=0, success=ok)


def spe_crossing_energy_batch(G: np.ndarray, d: int, tau: float, c0: float = 0.0, p0_lower_bound: float = 0.5,
                              grid: int = 4001, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``spe_crossing_energy`` for many g vectors (rows of G). Returns (energies, success).

    Same grid, same first-sign-change rule and same root (bisection to 1e-13 rad instead of
    brentq); tests/ check agreement with the scalar version.
    """
    p = SPEParams.from_degree(d, p0_lower_bound)
    level = 0.75 * p.eta
    j = p.odd_degrees
    xs = np.linspace(-np.pi / 3, np.pi / 3, grid)
    A = p.coefficients[:, None] * np.sin(np.outer(j, xs))
    B = p.coefficients[:, None] * np.cos(np.outer(j, xs))
    energies = np.empty(len(G))
    success = np.empty(len(G), dtype=bool)
    for a in range(0, len(G), chunk):
        gj = G[a:a + chunk][:, j]
        re, im = gj.real, gj.imag
        c = 0.5 + re @ A + im @ B - level
        up = (c[:, :-1] <= 0) & (c[:, 1:] > 0)
        has = up.any(axis=1) & (c[:, 0] <= 0)
        i = np.argmax(up, axis=1)
        lo, hi = xs[i], xs[np.minimum(i + 1, grid - 1)]
        for _ in range(40):  # bisection on the exact C~ (vectorized over trials)
            mid = 0.5 * (lo + hi)
            cm = 0.5 + np.sum(p.coefficients * (np.sin(np.outer(mid, j)) * re + np.cos(np.outer(mid, j)) * im),
                              axis=1) - level
            hi = np.where(cm > 0, mid, hi)
            lo = np.where(cm > 0, lo, mid)
        x = 0.5 * (lo + hi)
        x = np.where(has, x, np.where(c[:, 0] > 0, xs[0], xs[-1]))
        energies[a:a + chunk] = x / tau + c0
        success[a:a + chunk] = has
    return energies, success


def spe_sample_count(d: int, p0_lower_bound: float = 0.5) -> float:
    """Total Hadamard-test samples M = 2 F^2 / eta^2 from [3] (before success-probability boosting)."""
    p = SPEParams.from_degree(d, p0_lower_bound)
    return 2 * p.one_norm**2 / p.eta**2
