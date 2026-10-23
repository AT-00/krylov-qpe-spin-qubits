"""Synthetic checks of the analysis layer (jackknife, central differences, CRN pairing)."""

import numpy as np
import pytest

from krylov_qpe.noisy import NoiseConfig
from krylov_qpe.sensitivity import (
    chemical_accuracy_budget,
    jackknife,
    sensitivity_configs,
    sensitivity_from_samples,
    SweepCurve,
)


def test_jackknife_of_mean_equals_standard_error():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    est, se = jackknife(x, lambda m: m[1], n_blocks=400)
    assert est == pytest.approx(x[:, 1].mean())
    assert se == pytest.approx(x[:, 1].std(ddof=1) / np.sqrt(400), rel=1e-10)


def test_sensitivity_recovers_known_model():
    """E = a_qs*G_qs^2 + a_w*G_w + a_p*G_p^2 + a_x*G_x^2 (+ shared noise) -> exact derivatives and ratios."""
    point = NoiseConfig(quasistatic=0.02, white=0.001, pink=0.1, exchange=0.2)
    coef = {"quasistatic": -3.0, "white": -1.0, "pink": -0.8, "exchange": -0.05}
    power = {"quasistatic": 2, "white": 1, "pink": 2, "exchange": 2}

    def model(cfg):
        d = cfg.as_dict()
        return sum(coef[s] * d[s] ** power[s] for s in d)

    cfgs = sensitivity_configs(point, 0.1)
    rng = np.random.default_rng(1)
    common = rng.normal(scale=1e-3, size=(300, 1))  # realization noise shared by all configs (CRN)
    samples = np.stack([np.repeat(model(c) + common, 2, axis=1) for c in cfgs])  # "g" with 2 columns

    sens = sensitivity_from_samples(samples, point, 0.1, lambda m: m[0])
    g0 = point.as_dict()
    shift = model(point) - model(NoiseConfig())
    for s in g0:
        exact = coef[s] * power[s] * g0[s] ** (power[s] - 1)
        assert sens.dE_dGamma[s] == pytest.approx(exact, rel=1e-9)  # central difference is exact here
        assert sens.dE_dGamma_se[s] == pytest.approx(0, abs=1e-12)  # CRN cancels shared noise
        assert sens.relative[s] == pytest.approx(g0[s] * exact / shift, rel=0.02)
    assert sum(sens.relative.values()) == pytest.approx(
        sum(power[s] * coef[s] * g0[s] ** power[s] for s in g0) / shift, rel=0.02)


def test_budget_interpolation():
    c = SweepCurve("pink", [0.01, 0.1, 1.0], [-1.0001, -1.001, -1.1], [0, 0, 0], 0.1)
    # |E - E_exact| = 0.1 mHa, 1 mHa, 100 mHa; threshold 10 mHa -> log-log midpoint
    g = chemical_accuracy_budget(c, -1.0, threshold=0.01)
    assert g == pytest.approx(10 ** -0.5, rel=1e-9)
