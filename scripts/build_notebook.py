"""Build notebooks/demo.ipynb (the source of truth for the notebook's cells)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Quantum Krylov vs. statistical phase estimation on simulated silicon spin qubits\n\n"
       "H₂ ground-state energy from one shared quantum primitive, the Hadamard test of "
       "g(kτ) = ⟨HF|e^{−iHkτ}|HF⟩, post-processed two ways: real-time Krylov subspace diagonalization "
       "and statistical phase estimation. The circuits are executed at pulse level under spin-qubit noise "
       "with **SpinPulse** (Quobly, arXiv:2601.10435). Everything is **simulation**.\n\n"
       "Live cells use few noise realizations (about 1 min in total); the figures at the end come from "
       "`scripts/run_all.py`."),
    code("import sys, json, warnings\nfrom pathlib import Path\nwarnings.filterwarnings('ignore')\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT / 'src'))\nimport numpy as np"),
    md("## 1. Hamiltonian: OpenFermion → Jordan–Wigner → Z₂ tapering → 1 qubit"),
    code("from krylov_qpe.hamiltonian import h2_problem\n"
         "prob = h2_problem()\n"
         "print('4-qubit JW terms:', prob.jw_op.size)\n"
         "print('tapered H =', prob.op)\n"
         "print(f'E_FCI = {prob.fci_energy:.6f} Ha, E_HF = {prob.hf_energy:.6f} Ha, p0 = |<E0|HF>|^2 = {prob.ground_state_overlap:.4f}')"),
    md("## 2. Shared primitive: controlled evolution + Hadamard test\n"
       "One controlled step is one RZZ plus single-qubit rotations; U^k is applied as k steps (no fast-forwarding)."),
    code("from krylov_qpe.evolution import default_tau, controlled_step_circuit, statevector_g\n"
         "from krylov_qpe.hamiltonian import exact_g\n"
         "tau = default_tau(prob)\n"
         "print(f'tau = {tau:.4f} 1/Ha')\n"
         "print('max |g_circuit - g_exact| over k<=20:', abs(statevector_g(prob, tau, 20) - exact_g(prob, tau*np.arange(21))).max())\n"
         "controlled_step_circuit(prob, tau).draw()"),
    md("## 3. Noiseless estimators"),
    code("from krylov_qpe.krylov import krylov_energy\nfrom krylov_qpe.phase_est import spe_energy, spe_crossing_energy\n"
         "g = exact_g(prob, tau*np.arange(2100)); E0 = prob.fci_energy\n"
         "for D in (1, 2, 3):\n"
         "    print(f'Krylov D={D}: error {1e3*(krylov_energy(g, D, tau, prob.c0).energy - E0):+.2e} mHa')\n"
         "for d in (63, 255, 2047):\n"
         "    print(f'SPE d={d:4d}: bisection {1e3*(spe_energy(g, d, tau, prob.c0).energy-E0):+7.2f} mHa, '\n"
         "          f'crossing {1e3*(spe_crossing_energy(g, d, tau, prob.c0).energy-E0):+7.2f} mHa')"),
    md("## 4. SpinPulse: compile and run with noise\n"
       "Hardware model from SpinPulse's *ParametrizingfromQPUSpecs* example (1 µs X and ZZ gates). "
       "Noise rates Γ = 1/T* in 1/µs. Importing `krylov_qpe.noisy` applies a correction for a qubit-ordering "
       "issue in SpinPulse 1.1.3's two-qubit layers (see README, Implementation notes)."),
    code("from krylov_qpe.noisy import compile_hadamard_test, noisy_g_samples, DEFAULT_OPERATING_POINT, NoiseConfig, trajectory_g\n"
         "from krylov_qpe import spinpulse_patch\n"
         "print('SpinPulse qubit-ordering correction applied:', spinpulse_patch.is_applied())\n"
         "comp = compile_hadamard_test(prob, tau)\n"
         "print('step:', dict(comp.step.isa.count_ops()), f'{comp.step.duration_ns} ns | prep:', dict(comp.prep.isa.count_ops()), f'{comp.prep.duration_ns} ns')\n"
         "comp.step.pulse.plot(hardware_specs=comp.hs)"),
    md("A small live run. Krylov at D = 2 is ill-conditioned (1 − |g(τ)| ≈ 0.02), so a few hundred realizations "
       "still leave errors of several mHa; the full results use N = 5000."),
    code("from krylov_qpe.sensitivity import jackknife\n"
         "from krylov_qpe.resources import krylov_estimator\n"
         "N = 200\n"
         "G = noisy_g_samples(comp, 2, [NoiseConfig(), DEFAULT_OPERATING_POINT], N)\n"
         "f = krylov_estimator(2, tau, prob.c0)\n"
         "for name, gs in zip(['noiseless pulses', 'operating point'], G):\n"
         "    e, se = jackknife(gs, f)\n"
         "    print(f'Krylov D=2, {name:17s}: E - E_FCI = {1e3*(e-E0):+7.2f} ± {1e3*se:.2f} mHa  (N={N})')"),
    md("## 5. Full results (`python scripts/run_all.py`)"),
    code("from IPython.display import Image, display\n"
         "res = json.loads((ROOT / 'results' / 'results.json').read_text())\n"
         "for alg in ('krylov_high_statistics', 'spe'):\n"
         "    s = res['noisy'][alg]['sensitivity']\n"
         "    print(f\"{alg}: E - E_FCI = {1e3*(s['energy']-E0):+.1f} ± {1e3*s['energy_se']:.1f} mHa, noise shift {1e3*s['noise_shift']:+.1f} mHa\")\n"
         "    for k in s['relative']:\n"
         "        print(f\"   {k:12s} relative sensitivity {s['relative'][k]:+.2f} ± {s['relative_se'][k]:.2f}\")\n"
         "for f in ['fig1_convergence.png', 'fig2_sweeps.png', 'fig3_sensitivity.png']:\n"
         "    display(Image(filename=str(ROOT / 'figures' / f)))"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = ROOT / "notebooks" / "demo.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
