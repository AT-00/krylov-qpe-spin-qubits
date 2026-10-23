"""Noise sweeps and central-difference sensitivities for both estimators.

An estimator maps the realization-averaged Hadamard-test data g(k tau) to an energy. That
is what an experiment does: every shot sees a different noise realization, so measured
expectation values are noise averages. Estimators are nonlinear in g, so error bars come
from a delete-one-block jackknife over noise realizations.

Sensitivities at an operating point Gamma_0, relative step h:

    S_j     = dE/dGamma_j ~ [E(G_j(1+h)) - E(G_j(1-h))] / (2 h G_j)        (Ha us)
    eps_j   = Gamma_j S_j / (E(Gamma_0) - E_noiseless)                       (dimensionless)

eps_j is normalized by the *noise-induced* shift, not by the total error, because SPE's
total error also contains its resolution bias (~1/d), which no noise source controls.
For Krylov D = 2 the noiseless error is ~0, so both normalizations coincide.
All configurations share the same realizations (common random numbers).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .noisy import SOURCES, NoiseConfig

Estimator = Callable[[np.ndarray], float]


def jackknife(samples: np.ndarray, stat: Callable[[np.ndarray], float], n_blocks: int = 20) -> tuple[float, float]:
    """Estimate stat(mean g) and its jackknife standard error. samples: (n_real, K+1)."""
    n = len(samples)
    n_blocks = min(n_blocks, n)
    full = stat(samples.mean(axis=0))
    blocks = np.array_split(np.arange(n), n_blocks)
    total = samples.sum(axis=0)
    loo = np.array([stat((total - samples[b].sum(axis=0)) / (n - len(b))) for b in blocks])
    se = np.sqrt((n_blocks - 1) / n_blocks * np.sum((loo - loo.mean()) ** 2))
    return float(full), float(se)


def paired_jackknife(a: np.ndarray, b: np.ndarray, stat: Callable[[np.ndarray, np.ndarray], float],
                     n_blocks: int = 20) -> tuple[float, float]:
    """Jackknife for a statistic of two sample sets on the same realizations (keeps pairing)."""
    n = len(a)
    n_blocks = min(n_blocks, n)
    full = stat(a.mean(0), b.mean(0))
    blocks = np.array_split(np.arange(n), n_blocks)
    ta, tb = a.sum(0), b.sum(0)
    loo = np.array([stat((ta - a[bl].sum(0)) / (n - len(bl)), (tb - b[bl].sum(0)) / (n - len(bl))) for bl in blocks])
    se = np.sqrt((n_blocks - 1) / n_blocks * np.sum((loo - loo.mean()) ** 2))
    return float(full), float(se)


def sweep_configs(defaults: NoiseConfig, factors) -> tuple[list[NoiseConfig], list[tuple[str, float]]]:
    d = defaults.as_dict()
    configs, keys = [], []
    for s in SOURCES:
        for f in factors:
            configs.append(NoiseConfig.only(s, d[s] * f))
            keys.append((s, float(d[s] * f)))
    return configs, keys


@dataclass
class SweepCurve:
    source: str
    gammas: list[float]
    energy: list[float]
    energy_se: list[float]
    default_gamma: float


def sweep_curves(samples: np.ndarray, keys: list[tuple[str, float]], estimator: Estimator, defaults: NoiseConfig,
                 n_blocks: int = 20) -> list[SweepCurve]:
    """samples: (n_configs, n_real, K+1) aligned with keys from sweep_configs."""
    out = []
    for s in SOURCES:
        rows = [i for i, (src, _) in enumerate(keys) if src == s]
        stats = [jackknife(samples[i], estimator, n_blocks) for i in rows]
        out.append(SweepCurve(s, [keys[i][1] for i in rows], [e for e, _ in stats], [se for _, se in stats],
                              defaults.as_dict()[s]))
    return out


def sensitivity_configs(point: NoiseConfig, rel_step: float) -> list[NoiseConfig]:
    """[noiseless, point, then (+, -) for each source]."""
    g0 = point.as_dict()
    cfgs = [NoiseConfig(), point]
    for s in SOURCES:
        cfgs.append(point.with_(**{s: g0[s] * (1 + rel_step)}))
        cfgs.append(point.with_(**{s: g0[s] * (1 - rel_step)}))
    return cfgs


@dataclass
class Sensitivity:
    operating_point: dict[str, float]
    rel_step: float
    energy_noiseless: float
    energy: float
    energy_se: float
    noise_shift: float  # E(point) - E(noiseless)
    noise_shift_se: float
    dE_dGamma: dict[str, float]  # Ha us
    dE_dGamma_se: dict[str, float]
    relative: dict[str, float]
    relative_se: dict[str, float]

    def to_dict(self):
        return asdict(self)


def sensitivity_from_samples(samples: np.ndarray, point: NoiseConfig, rel_step: float, estimator: Estimator,
                             n_blocks: int = 20) -> Sensitivity:
    """samples aligned with sensitivity_configs(point, rel_step)."""
    g0 = point.as_dict()
    e_free = estimator(samples[0].mean(0))  # noiseless pulses: identical in every realization
    e0, e0_se = jackknife(samples[1], estimator, n_blocks)
    shift = e0 - e_free
    grad, grad_se, rel, rel_se = {}, {}, {}, {}
    for j, s in enumerate(SOURCES):
        plus, minus = samples[2 + 2 * j], samples[3 + 2 * j]
        h = rel_step * g0[s]
        grad[s], grad_se[s] = paired_jackknife(plus, minus, lambda a, b: (estimator(a) - estimator(b)) / (2 * h),
                                               n_blocks)
        # relative sensitivity: ratio with the noise-induced shift, jackknifed jointly (pairs plus/minus/point)
        trip = np.concatenate([plus, minus, samples[1]], axis=1)
        k = plus.shape[1]

        def ratio(m, k=k, h=h, s=s):
            return g0[s] * (estimator(m[:k]) - estimator(m[k:2 * k])) / (2 * h) / (estimator(m[2 * k:]) - e_free)

        rel[s], rel_se[s] = jackknife(trip, ratio, n_blocks)
    return Sensitivity(g0, rel_step, e_free, e0, e0_se, shift, e0_se, grad, grad_se, rel, rel_se)


def chemical_accuracy_budget(curve: SweepCurve, e_exact: float, threshold: float = 1.6e-3) -> float | None:
    """Largest Gamma (1/us) with |E - E_exact| <= threshold, log-log interpolation; None if not crossed."""
    g = np.log(np.asarray(curve.gammas))
    err = np.abs(np.asarray(curve.energy) - e_exact)
    for i in range(len(g) - 1):
        if err[i] <= threshold < err[i + 1] and err[i] > 0:
            t = (np.log(threshold) - np.log(err[i])) / (np.log(err[i + 1]) - np.log(err[i]))
            return float(np.exp(g[i] + t * (g[i + 1] - g[i])))
    return None
