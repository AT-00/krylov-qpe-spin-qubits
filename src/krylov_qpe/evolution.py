"""Controlled time evolution and the Hadamard test.

Shared quantum primitive for both algorithms:

    g(k tau) = <HF| U^k |HF>,   U = exp(-i H' tau),   H' = H - c0 I,

measured with one ancilla: prepare |+>_a |HF>, apply controlled-U k times, then
<X_a> = Re g and <Y_a> = Im g. The identity part c0 only adds a known global
phase, so it is put back classically (energies are shifted by c0).

Qubit layout: ancilla = qubit 0, system = qubit 1 (neighbours on SpinPulse's linear chain).

Controlled step. For the 1-qubit tapered H' = gz Z + gx X = b (n . sigma),
    U = Ry(phi) Rz(2 theta) Ry(-phi),  theta = b tau,  phi = atan2(gx, gz),
and controlled-Rz(2 theta) = Rz_sys(theta) RZZ(-theta), so one controlled step is
one RZZ plus single-qubit rotations (checked against scipy.linalg.expm in tests/).

No fast-forwarding. For this 1-qubit Hamiltonian U^k could be compiled as a single
controlled rotation of angle k*theta. A many-body Hamiltonian cannot be fast-forwarded,
so to keep the cost structure of the real problem, U^k is applied as k separate
controlled steps: circuit depth and duration grow linearly with evolution time.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .hamiltonian import H2Problem

ANCILLA, SYSTEM = 0, 1
OBS_X = SparsePauliOp("IX")  # X on qubit 0 (little-endian label)
OBS_Y = SparsePauliOp("IY")


def default_tau(problem: H2Problem) -> float:
    """tau = pi / (3 lambda): keeps tau*|E'| < pi/3 using only the Pauli 1-norm bound."""
    return float(np.pi / (3.0 * problem.traceless_one_norm))


def step_angles(problem: H2Problem, tau: float) -> tuple[float, float]:
    b = float(np.hypot(problem.gx, problem.gz))
    return b * tau, float(np.arctan2(problem.gx, problem.gz))


def prep_circuit(problem: H2Problem) -> QuantumCircuit:
    qc = QuantumCircuit(2, name="prep")
    qc.h(ANCILLA)
    if problem.hf_index == 1:
        qc.x(SYSTEM)
    return qc


def controlled_step_circuit(problem: H2Problem, tau: float) -> QuantumCircuit:
    """Controlled-exp(-i H' tau), ancilla = control."""
    theta, phi = step_angles(problem, tau)
    qc = QuantumCircuit(2, name="c-U")
    qc.ry(-phi, SYSTEM)
    qc.rz(theta, SYSTEM)
    qc.rzz(-theta, ANCILLA, SYSTEM)
    qc.ry(phi, SYSTEM)
    return qc


def hadamard_test_circuit(problem: H2Problem, tau: float, k: int) -> QuantumCircuit:
    """Full logical circuit for g(k tau): prep + k controlled steps (measurement basis omitted)."""
    qc = prep_circuit(problem)
    step = controlled_step_circuit(problem, tau)
    for _ in range(k):
        qc.compose(step, inplace=True)
    return qc


def g_from_state(state: np.ndarray) -> complex:
    """g = <X_a> + i <Y_a> for a 2-qubit statevector (qubit 0 = ancilla)."""
    sv = Statevector(state)
    return complex(np.real(sv.expectation_value(OBS_X)) + 1j * np.real(sv.expectation_value(OBS_Y)))


def statevector_g(problem: H2Problem, tau: float, k_max: int) -> np.ndarray:
    """Noiseless g(k tau), k = 0..k_max, by simulating the circuits (validation of the primitive)."""
    return np.array([g_from_state(Statevector(hadamard_test_circuit(problem, tau, k)).data) for k in range(k_max + 1)])
