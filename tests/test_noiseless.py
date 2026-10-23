import numpy as np
import pytest
from openfermion import MolecularData, get_fermion_operator, get_sparse_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from krylov_qpe.evolution import (
    controlled_step_circuit,
    default_tau,
    hadamard_test_circuit,
    statevector_g,
)
from krylov_qpe.hamiltonian import exact_g, h2_problem
from krylov_qpe.krylov import krylov_energy, krylov_matrices
from krylov_qpe.phase_est import SPEParams, cdf_estimate, spe_energy

CHEM_ACC = 1.6e-3


@pytest.fixture(scope="module")
def h2():
    return h2_problem()


def test_reference_energies(h2):
    assert h2.fci_energy == pytest.approx(-1.13731, abs=1e-5)
    assert h2.exact_ground_energy == pytest.approx(h2.fci_energy, abs=1e-10)
    # HF is a basis state of the tapered space with the HF energy
    assert np.real(h2.hf_state.conj() @ h2.op.to_matrix() @ h2.hf_state) == pytest.approx(h2.hf_energy, abs=1e-10)


def test_tapering_preserves_sector_spectrum(h2):
    """Tapered spectrum == JW spectrum in the HF symmetry sector (OpenFermion reference matrix)."""
    mol = run_pyscf(MolecularData([("H", (0, 0, 0)), ("H", (0, 0, 0.735))], "sto-3g", 1, 0))
    full = get_sparse_operator(jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian())), 4).toarray()
    # OpenFermion is big-endian: |q0 q1 q2 q3>; HF = 1100 -> index 12, doubly excited 0011 -> 3
    block = full[np.ix_([12, 3], [12, 3])]
    # the block is invariant under H (no coupling to other determinants)
    others = [i for i in range(16) if i not in (12, 3)]
    assert np.allclose(full[np.ix_(others, [12, 3])], 0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(block), h2.exact_energies, atol=1e-10)


def test_controlled_step_is_exact(h2):
    tau = default_tau(h2)
    X, Z = np.array([[0, 1], [1, 0]]), np.diag([1.0, -1.0])
    U = expm(-1j * tau * (h2.gx * X + h2.gz * Z))
    # little-endian: qubit 0 (ancilla) is the least significant factor
    CU = np.kron(np.eye(2), np.diag([1, 0])) + np.kron(U, np.diag([0, 1]))
    np.testing.assert_allclose(Operator(controlled_step_circuit(h2, tau)).data, CU, atol=1e-12)


def test_hadamard_test_matches_exact_g(h2):
    tau = default_tau(h2)
    np.testing.assert_allclose(statevector_g(h2, tau, 15), exact_g(h2, tau * np.arange(16)), atol=1e-12)
    assert hadamard_test_circuit(h2, tau, 5).count_ops()["rzz"] == 5  # no fast-forwarding


def test_tau_avoids_aliasing(h2):
    tau = default_tau(h2)
    assert np.max(np.abs(tau * (h2.exact_energies - h2.c0))) < np.pi / 3


def test_krylov_converges_at_dim_2(h2):
    tau = default_tau(h2)
    g = exact_g(h2, tau * np.arange(12))
    S, T = krylov_matrices(g, 3)
    np.testing.assert_allclose(S, S.conj().T, atol=1e-14)
    assert np.linalg.eigvalsh(S)[0] == pytest.approx(0, abs=1e-12)  # Krylov space is 2-dimensional
    assert abs(krylov_energy(g, 1, tau, h2.c0).energy - h2.fci_energy) > CHEM_ACC
    for dim in (2, 3, 5, 8):
        r = krylov_energy(g, dim, tau, h2.c0)
        assert abs(r.energy - h2.fci_energy) < 1e-9
        assert r.kept == 2


def test_cdf_is_a_smoothed_step(h2):
    tau = default_tau(h2)
    g = exact_g(h2, tau * np.arange(256))
    p = SPEParams.from_degree(255)
    x0, x1 = tau * (h2.exact_energies - h2.c0)
    c = cdf_estimate([x0 - 3 * p.delta, x0 + 3 * p.delta, x1 + 3 * p.delta], g, p)
    assert c[0] == pytest.approx(0, abs=p.eps)
    assert c[1] == pytest.approx(h2.ground_state_overlap, abs=p.eps)
    assert c[2] == pytest.approx(1, abs=p.eps)


def test_spe_within_guarantee_and_converges(h2):
    tau = default_tau(h2)
    g = exact_g(h2, tau * np.arange(2100))
    errs = []
    for d in (31, 127, 511, 2047):
        r = spe_energy(g, d, tau, h2.c0)
        err = abs(r.energy - h2.fci_energy)
        assert err <= r.params.delta / tau + 1e-12
        errs.append(err)
    assert errs[-1] < CHEM_ACC
    assert errs[-1] < errs[0] / 10
