"""Optional exact accelerator for SpinPulse's pulse integration.

SpinPulse 1.1.3 integrates a pulse layer with ``spin_pulse.transpilation.utils.propagate``:
one ``scipy.linalg.expm`` per 1 ns time slice, then a sequential matrix-product reduce.
``propagate_eigh`` computes the same product with batched Hermitian eigendecompositions
and a pairwise (tree) product. It agrees with the original to ~1e-14 and is 2-3x faster
for the 2- and 4-dimensional slices used here (tests/test_noisy.py checks agreement on
full noisy trajectories).

``enable()`` rebinds the name used inside ``spin_pulse.transpilation.pulse_layer``;
``disable()`` restores SpinPulse's own function. Nothing else in SpinPulse is changed.
"""

from __future__ import annotations

import numpy as np
from spin_pulse.transpilation import pulse_layer, utils

_ORIGINAL = utils.propagate


def propagate_eigh(H: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """Same contract as spin_pulse.transpilation.utils.propagate: U = prod_i exp(-i sum_j coeff[j,i] H[j])."""
    h_tot = np.einsum("jkl,ji->ikl", H, coeff)
    w, v = np.linalg.eigh(h_tot)
    us = (v * np.exp(-1j * w)[:, None, :]) @ np.conj(np.swapaxes(v, 1, 2))
    us = us[::-1]  # later times to the left, as in the original reduce
    while len(us) > 1:
        if len(us) % 2:
            us = np.concatenate([us[:-2], (us[-2] @ us[-1])[None]])
        us = us[0::2] @ us[1::2]
    return us[0]


def enable() -> None:
    pulse_layer.propagate = propagate_eigh


def disable() -> None:
    pulse_layer.propagate = _ORIGINAL


def is_enabled() -> bool:
    return pulse_layer.propagate is propagate_eigh
