"""Real-time quantum Krylov subspace diagonalization (QKSD) from g(k tau).

Basis |psi_k> = U^k |HF>, U = exp(-i H' tau), k = 0..D-1. All projected matrices are
Toeplitz in the lag k - j, so they follow from the same Hadamard-test data g:

    S_jk = <psi_j|psi_k>   = g((k-j) tau)
    T_jk = <psi_j|U|psi_k> = g((k-j+1) tau)          (lags -(D-1) .. D)

with g(-t) = conj(g(t)). We solve the *unitary* pencil T v = mu S v and read the energies
from the eigenphases, E = -arg(mu)/tau + c0. This needs no measurement of H itself (the
real-time analogue of [3], where the projected matrices are built from Chebyshev moments
alone). It is exact when U is the exact propagator, as here.

Regularization (as in [3]): diagonalize S, discard eigenvectors with eigenvalue below
``threshold * max eigenvalue``, and solve the pencil in the remaining subspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _lag(g: np.ndarray, m: int) -> complex:
    return g[m] if m >= 0 else np.conj(g[-m])


def krylov_matrices(g: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """S and T (dim x dim) from g[k] = g(k tau), k = 0..dim (needs len(g) >= dim + 1)."""
    if len(g) < dim + 1:
        raise ValueError(f"need g up to lag {dim}, got {len(g) - 1}")
    S = np.array([[_lag(g, k - j) for k in range(dim)] for j in range(dim)])
    T = np.array([[_lag(g, k - j + 1) for k in range(dim)] for j in range(dim)])
    return S, T


@dataclass
class KrylovResult:
    energy: float  # lowest eigen-energy (Ha, including c0)
    energies: np.ndarray  # all retained eigen-energies, sorted
    kept: int  # retained dimension after regularization
    s_eigs: np.ndarray  # eigenvalues of S (descending)


def krylov_energy(g: np.ndarray, dim: int, tau: float, c0: float = 0.0, threshold: float = 1e-8) -> KrylovResult:
    S, T = krylov_matrices(g, dim)
    S = 0.5 * (S + S.conj().T)  # exactly Hermitian (it is, up to rounding)
    s, V = np.linalg.eigh(S)
    order = np.argsort(s)[::-1]
    s, V = s[order], V[:, order]
    keep = s > threshold * s[0]
    Vk = V[:, keep] / np.sqrt(s[keep])[None, :]  # whitened basis: Vk^H S Vk = I
    mu = np.linalg.eigvals(Vk.conj().T @ T @ Vk)
    energies = np.sort(-np.angle(mu) / tau + c0)
    return KrylovResult(energy=float(energies[0]), energies=energies, kept=int(keep.sum()), s_eigs=s)
