import numpy as np
import pytest
from qiskit.quantum_info import Operator
from spin_pulse import ExperimentalEnvironment, PulseCircuit
from spin_pulse.environment.noise import NoiseType

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm as _expm

from krylov_qpe import fast_propagate, spinpulse_patch
from krylov_qpe.evolution import controlled_step_circuit, default_tau, g_from_state
from krylov_qpe.hamiltonian import exact_g, h2_problem
from krylov_qpe.krylov import krylov_energy
from krylov_qpe.noisy import (
    DEFAULT_OPERATING_POINT,
    NoiseConfig,
    TraceBundle,
    bundle_for,
    compile_hadamard_test,
    generate_unit_traces,
    make_hardware_specs,
    noisy_g_samples,
    trajectory_g,
)


@pytest.fixture(scope="module")
def setup():
    prob = h2_problem()
    tau = default_tau(prob)
    return prob, tau, compile_hadamard_test(prob, tau)


def test_noiseless_pulses_close_to_exact(setup):
    prob, tau, comp = setup
    g = trajectory_g(comp, 10, None)
    np.testing.assert_allclose(g, exact_g(prob, tau * np.arange(11)), atol=2e-3)  # non-adiabatic pulse error only
    assert abs(krylov_energy(g, 2, tau, prob.c0).energy - prob.fci_energy) < 1.6e-3


def test_fast_propagate_matches_spinpulse(setup):
    prob, tau, comp = setup
    unit = generate_unit_traces(comp.hs, comp.run_duration_ns(8), 3)
    bundle = bundle_for(unit, DEFAULT_OPERATING_POINT.scaled(3.0), comp.hs)
    fast_propagate.disable()
    ref = trajectory_g(comp, 8, bundle)
    fast_propagate.enable()
    try:
        fast = trajectory_g(comp, 8, bundle)
    finally:
        fast_propagate.disable()
    np.testing.assert_allclose(fast, ref, atol=1e-11)
    assert abs(ref[8] - exact_g(prob, [8 * tau])[0]) > 1e-3  # noise actually acted


def test_bundle_equals_spinpulse_environment(setup):
    """One step integrated with SpinPulse's own environment == through our summed-trace container."""
    prob, tau, comp = setup
    hs = comp.hs
    env = ExperimentalEnvironment(hs, NoiseType.PINK, T2S=6_000, TJS=3_000, duration=2**16, segment_duration=2**16,
                                  seed=11)
    native = PulseCircuit.from_circuit(comp.step.isa, hs, exp_env=env)  # attached at t_lab = 0
    u_native = Operator.from_circuit(native.to_circuit()).data
    bundle = TraceBundle(hs, [t.values for t in env.time_traces], [t.values for t in env.time_traces_coupling])
    mine = PulseCircuit.from_circuit(comp.step.isa, hs)
    mine.t_lab = 0
    mine.attach_time_traces(bundle)
    np.testing.assert_allclose(Operator.from_circuit(mine.to_circuit()).data, u_native, atol=1e-12)
    ideal = Operator(controlled_step_circuit(prob, tau)).data
    assert np.abs(u_native - ideal).max() > 1e-3


@pytest.mark.parametrize("noise_type,seg", [(NoiseType.QUASISTATIC, 2**15), (NoiseType.WHITE, 1), (NoiseType.PINK, 2**16)])
def test_rescaling_equals_regeneration(noise_type, seg):
    hs = make_hardware_specs(2)
    kw = dict(duration=max(seg, 2**15), segment_duration=seg, seed=5)
    a = ExperimentalEnvironment(hs, noise_type, T2S=10_000.0, TJS=10_000.0, **kw)
    b = ExperimentalEnvironment(hs, noise_type, T2S=4_000.0, TJS=4_000.0, **kw)
    f = (10_000 / 4_000) ** (0.5 if noise_type is NoiseType.WHITE else 1.0)
    for x, y in zip(a.time_traces + a.time_traces_coupling, b.time_traces + b.time_traces_coupling):
        np.testing.assert_allclose(x.values * f, y.values, rtol=1e-10, atol=1e-15)


def test_zero_noise_and_reproducibility(setup):
    prob, tau, comp = setup
    G = noisy_g_samples(comp, 4, [NoiseConfig(), NoiseConfig(pink=0.2, exchange=0.4)], 4, n_jobs=1)
    np.testing.assert_allclose(G[0], np.broadcast_to(trajectory_g(comp, 4, None), G[0].shape), atol=1e-12)
    G2 = noisy_g_samples(comp, 4, [NoiseConfig(pink=0.2, exchange=0.4)], 4, n_jobs=1)
    np.testing.assert_array_equal(G[1], G2[0])
    assert np.all(np.abs(G[1][:, 1:]) <= 1 + 1e-9)


def test_block_splitting_is_exact(setup):
    """Realizations simulated in blocks equal one run (resumable caching relies on this)."""
    _, _, comp = setup
    cfg = [NoiseConfig(quasistatic=0.05, pink=0.2)]
    whole = noisy_g_samples(comp, 3, cfg, 5, n_jobs=1)
    parts = np.concatenate([noisy_g_samples(comp, 3, cfg, 2, n_jobs=1, first_realization=0),
                            noisy_g_samples(comp, 3, cfg, 3, n_jobs=1, first_realization=2)], axis=1)
    np.testing.assert_array_equal(whole, parts)


def test_g_from_state_conventions():
    plus = np.array([1, 1, 0, 0]) / np.sqrt(2)  # ancilla |+>, system |0>  -> g = 1
    assert g_from_state(plus) == pytest.approx(1.0)
    phase = np.array([1, -1j, 0, 0]) / np.sqrt(2)  # |0> + e^{-i pi/2}|1> on the ancilla -> g = -i
    assert g_from_state(phase) == pytest.approx(-1j)


def _offset_on_q0_acts_on(hs) -> tuple[float, float]:
    """Frequency offset on qubit 0 only, during the first two-qubit (exchange) layer of a bare RZZ.

    Returns |overlap| of the extra unitary with a Z rotation (phase pi/2) on qubit 0 and on qubit 1.
    """
    qc = QuantumCircuit(2)
    qc.rzz(0.6, 0, 1)
    pc = PulseCircuit.from_circuit(qc, hs)
    layer = next(lay for lay in pc.pulse_layers if lay.twoq_pulse_sequences)
    trace = np.zeros(pc.duration)
    trace[layer.t_start:layer.t_start + layer.duration] = (np.pi / 2) / layer.duration
    bundle = TraceBundle(hs, [trace, np.zeros(pc.duration)], [np.zeros(pc.duration)])
    clean = Operator.from_circuit(PulseCircuit.from_circuit(qc, hs).to_circuit()).data
    noisy_pc = PulseCircuit.from_circuit(qc, hs)
    noisy_pc.t_lab = 0
    noisy_pc.attach_time_traces(bundle)
    extra = Operator.from_circuit(noisy_pc.to_circuit()).data @ clean.conj().T

    def overlap(label):  # either sign: echo pulses may flip Z
        return max(abs(np.trace(_expm(-1j * sgn * np.pi / 4 * SparsePauliOp(label).to_matrix()).conj().T @ extra)) / 4
                   for sgn in (1, -1))

    return overlap("IZ"), overlap("ZI")  # little-endian: "IZ" = Z on qubit 0


def test_local_terms_stay_on_their_qubit_in_two_qubit_layers():
    """Asymmetric check of the SpinPulse 1.1.3 embedding issue, and that the patch fixes it."""
    hs = make_hardware_specs(2)
    assert spinpulse_patch.is_applied()  # applied at import of krylov_qpe.noisy
    on_q0, on_q1 = _offset_on_q0_acts_on(hs)
    assert on_q0 == pytest.approx(1.0, abs=1e-4) and on_q1 < 0.6
    spinpulse_patch.remove()
    try:
        on_q0, on_q1 = _offset_on_q0_acts_on(hs)
        assert on_q1 == pytest.approx(1.0, abs=1e-4) and on_q0 < 0.6  # as released: lands on qubit 1
    finally:
        spinpulse_patch.apply()


def test_patch_reaches_worker_processes(setup):
    """Workers (joblib) must use the patched layer: compare with the in-process trajectory."""
    prob, tau, comp = setup
    cfg = [NoiseConfig(pink=0.2, exchange=0.4)]
    unit = generate_unit_traces(comp.hs, comp.run_duration_ns(3), 0)
    expected = trajectory_g(comp, 3, bundle_for(unit, cfg[0], comp.hs))
    got = noisy_g_samples(comp, 3, cfg, 2, n_jobs=2)[0, 0]
    np.testing.assert_allclose(got, expected, atol=1e-12)
    unpatched = noisy_g_samples(comp, 3, cfg, 2, n_jobs=2, spinpulse_fix=False)[0, 0]
    assert np.abs(unpatched - expected).max() > 1e-4
