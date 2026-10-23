"""H2 Hamiltonian: OpenFermion (+PySCF) -> Jordan-Wigner -> Z2 tapering -> Qiskit SparsePauliOp.

Qubit convention
----------------
OpenFermion orders spin-orbitals (0a, 0b, 1a, 1b) and Jordan-Wigner maps spin-orbital p
to qubit p. Qiskit Pauli labels are little-endian; the conversion below builds operators
from (pauli, index) pairs, so qubit k in OpenFermion is qubit k in Qiskit.

Tapering
--------
The 4-qubit H2/STO-3G Hamiltonian commutes with Z0Z1, Z0Z2 and Z0Z3. The Hartree-Fock
determinant |q0 q1 q2 q3> = |1100> has eigenvalues (+1, -1, -1) for them, and so does the
only other determinant it couples to (|0011>). Fixing these eigenvalues with OpenFermion's
``taper_off_qubits`` removes three qubits and leaves a 1-qubit Hamiltonian

    H = c0 I + gz Z + gx X,

whose ground energy equals FCI and in which Hartree-Fock is |0> (both checked in tests/).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openfermion import (
    MolecularData,
    QubitOperator,
    get_fermion_operator,
    jordan_wigner,
    taper_off_qubits,
)
from openfermionpyscf import run_pyscf
from qiskit.quantum_info import SparsePauliOp

H2_EQ_BOND_LENGTH = 0.735  # Angstrom
H2_STABILIZERS = (("Z0 Z1", +1), ("Z0 Z2", -1), ("Z0 Z3", -1))


def qubit_operator_to_sparse_pauli_op(qop: QubitOperator, n_qubits: int) -> SparsePauliOp:
    """OpenFermion QubitOperator -> Qiskit SparsePauliOp with identical qubit indices."""
    entries = []
    for term, coeff in qop.terms.items():
        entries.append(("".join(p for _, p in term), [q for q, _ in term], coeff))
    op = SparsePauliOp.from_sparse_list(entries, num_qubits=n_qubits).simplify(atol=1e-12)
    return SparsePauliOp(op.paulis, np.real_if_close(op.coeffs).astype(complex))


@dataclass(frozen=True)
class H2Problem:
    """Everything the algorithms need about the molecule."""

    bond_length: float
    jw_op: SparsePauliOp  # 4-qubit Jordan-Wigner Hamiltonian
    op: SparsePauliOp  # 1-qubit tapered Hamiltonian
    hf_energy: float  # PySCF
    fci_energy: float  # PySCF
    c0: float  # identity coefficient (handled classically)
    gz: float
    gx: float
    hf_index: int  # computational basis state of HF in the tapered space

    @property
    def traceless_one_norm(self) -> float:
        """lambda = sum of |coefficients| of the non-identity Pauli terms (spectral-norm bound)."""
        return abs(self.gz) + abs(self.gx)

    @property
    def exact_energies(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.op.to_matrix().real)

    @property
    def exact_ground_energy(self) -> float:
        return float(self.exact_energies[0])

    @property
    def hf_state(self) -> np.ndarray:
        psi = np.zeros(2, complex)
        psi[self.hf_index] = 1.0
        return psi

    @property
    def ground_state_overlap(self) -> float:
        """p0 = |<E0|HF>|^2."""
        _, vecs = np.linalg.eigh(self.op.to_matrix())
        return float(abs(vecs[self.hf_index, 0]) ** 2)


def h2_problem(bond_length: float = H2_EQ_BOND_LENGTH, basis: str = "sto-3g") -> H2Problem:
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond_length))]
    mol = run_pyscf(MolecularData(geometry, basis, multiplicity=1, charge=0), run_scf=True, run_fci=True)
    jw = jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian()))
    stabilizers = [sign * QubitOperator(s) for s, sign in H2_STABILIZERS]
    tapered = taper_off_qubits(jw, stabilizers)
    tapered.compress(1e-12)
    n_tapered = 1 + max((q for term in tapered.terms for q, _ in term), default=0)
    if n_tapered != 1:
        raise RuntimeError(f"expected a 1-qubit tapered Hamiltonian, got {n_tapered} qubits")
    op = qubit_operator_to_sparse_pauli_op(tapered, 1)
    coeffs = {str(p): float(c.real) for p, c in zip(op.paulis, op.coeffs)}
    c0, gz, gx = coeffs.get("I", 0.0), coeffs.get("Z", 0.0), coeffs.get("X", 0.0)
    if abs(coeffs.get("Y", 0.0)) > 1e-12:
        raise RuntimeError("unexpected Y term in tapered H2 Hamiltonian")
    # Hartree-Fock = the computational basis state whose energy equals the HF energy.
    diag = np.real(np.diag(op.to_matrix()))
    hf_index = int(np.argmin(abs(diag - mol.hf_energy)))
    if abs(diag[hf_index] - mol.hf_energy) > 1e-8:
        raise RuntimeError("Hartree-Fock state not found in the tapered basis")
    return H2Problem(
        bond_length=bond_length,
        jw_op=qubit_operator_to_sparse_pauli_op(jw, 4),
        op=op,
        hf_energy=float(mol.hf_energy),
        fci_energy=float(mol.fci_energy),
        c0=c0,
        gz=gz,
        gx=gx,
        hf_index=hf_index,
    )


def exact_g(problem: H2Problem, times) -> np.ndarray:
    """g(t) = <HF| exp(-i H' t) |HF> for the traceless part H' = H - c0 I (exact)."""
    hp = problem.op.to_matrix() - problem.c0 * np.eye(2)
    evals, vecs = np.linalg.eigh(hp)
    w = abs(vecs[problem.hf_index, :]) ** 2
    t = np.asarray(times, dtype=float)
    return (w[None, :] * np.exp(-1j * np.outer(t, evals))).sum(axis=1)
