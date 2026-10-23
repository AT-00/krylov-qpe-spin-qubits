# --------------------------------------------------------------------------------------
# _to_circuit_fixed below is derived from SpinPulse (spin_pulse/transpilation/pulse_layer.py,
# PulseLayer.to_circuit, version 1.1.3).
#
# (C) Copyright Quobly 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license at http://www.apache.org/licenses/LICENSE-2.0.
#
# Modified by Amalia Tsitsekli, 2026: the einsum strings embedding the operands'
# single-qubit Hamiltonians in two-qubit layers are exchanged, and propagation is
# called through the pulse_layer module. The rest of the function is unchanged.
# --------------------------------------------------------------------------------------
"""Correction for a qubit-ordering issue in SpinPulse 1.1.3 (``PulseLayer.to_circuit``).

In a two-qubit layer, SpinPulse adds the single-qubit Hamiltonians of the two operands
(their detuning ramps and frequency-noise terms) to the two-qubit Hamiltonian as

    H0 = einsum("dij,ab->diajb", H0, I)   # H0 (x) I
    H1 = einsum("dij,ab->daibj", H1, I)   # I (x) H1

and appends the resulting 4x4 unitary to ``[q0, q1]``. Qiskit is little-endian: in a
matrix appended to [q0, q1], q0 is the *right-most* tensor factor. So ``H0 (x) I`` acts on
q1 and ``I (x) H1`` on q0, i.e. each operand's local terms are applied to the other
qubit. The Heisenberg term is symmetric and unaffected. ``tests/test_noisy.py`` contains a
minimal asymmetric example that fails without this patch.

``apply()`` replaces ``PulseLayer.to_circuit`` by a copy of SpinPulse's function in which
only the two einsum strings are exchanged. Propagation goes through
``pulse_layer.propagate`` (the module attribute), so ``fast_propagate.enable()`` keeps
working. ``remove()`` restores SpinPulse's original method.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from spin_pulse.transpilation import pulse_layer

_ORIGINAL = pulse_layer.PulseLayer.to_circuit


def _to_circuit_fixed(self):
    """SpinPulse 1.1.3 ``PulseLayer.to_circuit`` with the operand embeddings corrected."""
    circ = QuantumCircuit(self.qubits)
    indices_treated = []
    for pulse_sequence in self.twoq_pulse_sequences:
        qubits = pulse_sequence.qubits
        H, coeff = pulse_sequence.to_hamiltonian()
        H0, coeff0 = self.oneq_pulse_sequences[qubits[0]._index].to_hamiltonian()
        # I (x) H0: acts on qubits[0], the least significant factor in Qiskit's convention
        H0 = np.einsum("dij,ab->daibj", H0, np.eye(2)).reshape(H0.shape[0], H0.shape[1] * 2, H0.shape[2] * 2)
        H1, coeff1 = self.oneq_pulse_sequences[qubits[1]._index].to_hamiltonian()
        # H1 (x) I: acts on qubits[1]
        H1 = np.einsum("dij,ab->diajb", H1, np.eye(2)).reshape(H1.shape[0], H1.shape[1] * 2, H1.shape[2] * 2)
        H = np.concatenate((H, H0, H1), axis=0)
        coeff = np.concatenate((coeff, coeff0, coeff1), axis=0)
        indices_treated += [qubits[0]._index, qubits[1]._index]
        matrix = pulse_layer.propagate(H, coeff)
        circ.append(UnitaryGate(matrix, "2q_" + f"{self.duration}"), qubits)

    for i in range(self.num_qubits):
        if i not in indices_treated:
            H, coeff = self.oneq_pulse_sequences[i].to_hamiltonian()
            matrix = pulse_layer.propagate(H, coeff)
            circ.append(UnitaryGate(matrix, "1q_" + f"{self.duration}"), [self.qubits[i]])
    return circ


def apply() -> None:
    pulse_layer.PulseLayer.to_circuit = _to_circuit_fixed


def remove() -> None:
    pulse_layer.PulseLayer.to_circuit = _ORIGINAL


def is_applied() -> bool:
    return pulse_layer.PulseLayer.to_circuit is _to_circuit_fixed
