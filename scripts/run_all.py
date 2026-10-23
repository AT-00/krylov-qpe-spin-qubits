"""Regenerate every number and figure.

    python scripts/run_all.py --max-minutes 100   # full run (~3 h on 2 cores); rerun until complete
    python scripts/run_all.py --quick             # smoke run (~15 min)

Writes results/results.json (or results_quick.json) and calls make_figures.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from krylov_qpe.evolution import default_tau  # noqa: E402
from krylov_qpe.hamiltonian import exact_g, h2_problem  # noqa: E402
from krylov_qpe.krylov import krylov_energy  # noqa: E402
from krylov_qpe import spinpulse_patch  # noqa: E402
from krylov_qpe.noisy import (  # noqa: E402
    PINK_SEGMENT_NS,
    DEFAULT_OPERATING_POINT,
    SPINPULSE_EXAMPLE_POINT,
    compile_hadamard_test,
    noisy_g_samples,
    trajectory_g,
)
from krylov_qpe.phase_est import SPEParams, spe_crossing_energy, spe_energy  # noqa: E402
from krylov_qpe.resources import (  # noqa: E402
    krylov_cost,
    krylov_estimator,
    krylov_shots,
    krylov_trial_errors,
    shots_for_target,
    spe_cost,
    spe_estimator,
    spe_shots,
    spe_trial_errors,
    validate_budget,
)
from krylov_qpe.sensitivity import (  # noqa: E402
    chemical_accuracy_budget,
    jackknife,
    sensitivity_configs,
    sensitivity_from_samples,
    sweep_configs,
    sweep_curves,
)

CHEM = 1.6e-3
D_KRYLOV = 2  # Krylov dimension for the noisy comparison (exact noiselessly for H2)
D_SPE = 63  # SPE degree for the noisy comparison (longest run: 63 controlled steps)
K_KRYLOV = 6  # steps simulated in the Krylov runs (D up to 6 for the dimension plot)
FACTORS = np.logspace(-1, 1, 7)
REL_STEP = 0.1
KRYLOV_THRESHOLDS = (1e-3, 1e-2)


class Incomplete(Exception):
    pass


VERSIONS = {p: version(p) for p in ["spin-pulse", "qiskit", "openfermion", "pyscf", "numpy", "scipy"]}


def fingerprint(comp, patched: bool) -> str:
    """Everything outside a stage's own arguments that changes simulated samples."""
    src = repr((VERSIONS, PINK_SEGMENT_NS, patched, str(comp.hs), comp.step.duration_ns, comp.prep.duration_ns,
                float(comp.tau), [str(i) for i in comp.step.isa.data], [str(i) for i in comp.prep.isa.data]))
    return hashlib.sha1(src.encode()).hexdigest()[:10]


def cached_samples(tag, comp, k_max, configs, n, seed, jobs, fast, deadline, block=50, patched=True):
    """noisy_g_samples in blocks of realizations, each cached in results/cache/ (resumable).

    The cache key covers the stage arguments plus fingerprint() (package versions, 1/f
    segment, SpinPulse patch on/off, hardware specs, compiled circuits). Raises Incomplete
    when the time budget runs out; rerunning the same command resumes from cached blocks.
    """
    key_src = repr((tag, k_max, [c.as_dict() for c in configs], seed, fingerprint(comp, patched)))
    key = hashlib.sha1(key_src.encode()).hexdigest()[:12]
    cache = ROOT / "results" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for a in range(0, n, block):
        b = min(n, a + block)
        f = cache / f"{tag}_{key}_r{a}-{b}.npy"
        if f.exists():
            parts.append(np.load(f))
            continue
        if time.time() > deadline:
            raise Incomplete(f"{tag}: stopped at realization {a}/{n} (time budget); rerun to resume")
        t = time.time()
        G = noisy_g_samples(comp, k_max, configs, b - a, seed, jobs, fast=fast, first_realization=a,
                            spinpulse_fix=patched)
        np.save(f, G)
        parts.append(G)
        print(f"  {tag}: realizations {a}-{b} of {n} in {time.time() - t:.0f}s", flush=True)
    return np.concatenate(parts, axis=1)


def krylov_high_statistics(comp, prob, op, n, seed, jobs, fast, deadline=float("inf"), patched=True):
    """Krylov D = 2 sensitivities with many more realizations (runs need only 2 steps).

    Uses an independent seed stream (seed + 1) from the main Krylov run.
    """
    e0 = prob.exact_ground_energy
    cfgs = sensitivity_configs(op, REL_STEP) + [SPINPULSE_EXAMPLE_POINT]
    t = time.time()
    G = cached_samples("krylov_hi", comp, D_KRYLOV, cfgs, n, seed + 1, jobs, fast, deadline, block=500,
                       patched=patched)
    f = krylov_estimator(D_KRYLOV, comp.tau, prob.c0)
    sens = sensitivity_from_samples(G[:-1], op, REL_STEP, f)
    e_ex, e_ex_se = jackknife(G[-1], f)
    print(f"krylov high-statistics: {len(cfgs)} configs x {n} realizations in {time.time() - t:.0f}s")
    print(f"  E(op) - E0 = {1e3 * (sens.energy - e0):.2f} +- {1e3 * sens.energy_se:.2f} mHa;"
          f" noise shift {1e3 * sens.noise_shift:.2f} mHa; example point {1e3 * (e_ex - e0):.2f} +- {1e3 * e_ex_se:.2f}")
    for src in sens.relative:
        print(f"   {src:12s} dE/dG {1e3 * sens.dE_dGamma[src]:9.3f} +- {1e3 * sens.dE_dGamma_se[src]:.3f} mHa.us"
              f" | relative {sens.relative[src]:.3f} +- {sens.relative_se[src]:.3f}")
    return {"n_realizations": n, "seed": seed + 1, "sensitivity": sens.to_dict(),
            "spinpulse_example": {"energy": e_ex, "se": e_ex_se}}


def c2l(z):
    return {"re": np.real(z).tolist(), "im": np.imag(z).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-krylov", type=int, default=None)
    ap.add_argument("--n-spe", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-fast-propagate", action="store_true", help="use SpinPulse's own integrator")
    ap.add_argument("--n-krylov-hi", type=int, default=None, help="realizations for the high-statistics Krylov run")
    ap.add_argument("--krylov-hi-only", action="store_true",
                    help="only (re)compute the high-statistics Krylov run and merge it into an existing results file")
    ap.add_argument("--no-spinpulse-patch", action="store_true",
                    help="use SpinPulse 1.1.3's PulseLayer.to_circuit as released (see spinpulse_patch.py)")
    ap.add_argument("--n-validate", type=int, default=None, help="trials for validating shot budgets")
    ap.add_argument("--max-minutes", type=float, default=float("inf"),
                    help="stop after this many minutes; completed blocks are cached and a rerun resumes")
    args = ap.parse_args()
    n_k = args.n_krylov or (40 if args.quick else 1000)
    n_s = args.n_spe or (12 if args.quick else 200)
    n_hi = args.n_krylov_hi or (200 if args.quick else 5000)
    t0 = time.time()
    deadline = t0 + 60 * args.max_minutes
    try:
        run(args, n_k, n_s, n_hi, t0, deadline)
    except Incomplete as exc:
        print(f"INCOMPLETE: {exc}")
        sys.exit(3)


def run(args, n_k, n_s, n_hi, t0, deadline):
    patched = not args.no_spinpulse_patch
    spinpulse_patch.apply() if patched else spinpulse_patch.remove()
    n_val = args.n_validate or (2000 if args.quick else 20000)
    prob = h2_problem()
    tau = default_tau(prob)
    e0 = prob.exact_ground_energy
    comp = compile_hadamard_test(prob, tau)
    step_ns, prep_ns = comp.step.duration_ns, comp.prep.duration_ns
    rzz_per_step = dict(comp.step.isa.count_ops()).get("rzz", 0)
    print(f"E0 {e0:.8f} | tau {tau:.4f} 1/Ha | step {step_ns} ns | prep {prep_ns} ns | p0 {prob.ground_state_overlap:.4f}")
    out = ROOT / "results" / ("results_quick.json" if args.quick else "results.json")
    if args.krylov_hi_only:
        res = json.loads(out.read_text())
        res["noisy"]["krylov_high_statistics"] = krylov_high_statistics(
            comp, prob, DEFAULT_OPERATING_POINT, n_hi, args.seed, args.jobs, not args.no_fast_propagate, deadline,
            patched)
        out.write_text(json.dumps(res, indent=1, default=float))
        print(f"merged into {out}")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py"), str(out)], check=True)
        return

    # ---------------- noiseless convergence ----------------
    g_ex = exact_g(prob, tau * np.arange(4200))
    g_pulse = trajectory_g(comp, D_SPE, None)
    dims = list(range(1, 7))
    d_grid = [7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095]
    noiseless = {
        "krylov_dims": dims,
        "krylov_exact": [krylov_energy(g_ex, D, tau, prob.c0).energy - e0 for D in dims],
        "krylov_pulse": [krylov_energy(g_pulse, D, tau, prob.c0, 1e-3).energy - e0 for D in dims],
        "spe_degrees": d_grid,
        "spe_bisection": [spe_energy(g_ex, d, tau, prob.c0).energy - e0 for d in d_grid],
        "spe_crossing": [spe_crossing_energy(g_ex, d, tau, prob.c0).energy - e0 for d in d_grid],
        "spe_delta": [SPEParams.from_degree(d).delta / tau for d in d_grid],
        "spe_pulse_crossing_d63": spe_crossing_energy(g_pulse, D_SPE, tau, prob.c0).energy - e0,
        "krylov_pulse_D2": krylov_energy(g_pulse, 2, tau, prob.c0).energy - e0,
    }
    print("noiseless done")

    # ---------------- resources at chemical accuracy (ideal gates: exact g, shot noise only) ----------------
    res_key = hashlib.sha1(repr((VERSIONS, float(tau), prob.c0, prob.gz, prob.gx, CHEM, D_KRYLOV, n_val, args.quick,
                                 step_ns, prep_ns, rzz_per_step, "v2")).encode()).hexdigest()[:10]
    res_cache = ROOT / "results" / "cache" / f"resources_{res_key}.json"
    if res_cache.exists():
        resources = json.loads(res_cache.read_text())
    else:
        trials = 60 if args.quick else 300
        # exact-estimator degree for 1.6 mHa (no shots), and the degree used for the shot budget
        d_exact = next(d for d in range(1023, 8192, 32) if d % 2 and
                       abs(spe_crossing_energy(g_ex, d, tau, prob.c0).energy - e0) <= CHEM)
        d_ca = next(d for d in range(1023, 8192, 64) if d % 2 and
                    abs(spe_crossing_energy(g_ex, d, tau, prob.c0).energy - e0) <= CHEM / np.sqrt(2))
        g_ca = exact_g(prob, tau * np.arange(d_ca + 1))
        # 1) coarse budget from a few hundred trials, 2) validated directly with n_val trials
        kr_m0, kr_curve = shots_for_target(lambda m: krylov_shots(D_KRYLOV, m, len(g_ex)),
                                           krylov_estimator(D_KRYLOV, tau, prob.c0), g_ex, e0, CHEM,
                                           np.logspace(3, 7, 9).astype(int), n_trials=trials)
        sp_m0, sp_curve = shots_for_target(lambda m: spe_shots(d_ca, m, len(g_ca)), spe_estimator(d_ca, tau, prob.c0),
                                           g_ca, e0, CHEM, np.logspace(3, 8, 11).astype(int),
                                           n_trials=max(30, trials // 3))
        kr_val = validate_budget(lambda m, n, sd: krylov_trial_errors(D_KRYLOV, m, g_ca, e0, tau, prob.c0, n, sd),
                                 kr_m0, CHEM, n_val)
        sp_val = validate_budget(lambda m, n, sd: spe_trial_errors(d_ca, m, g_ca, e0, tau, prob.c0, n, sd),
                                 sp_m0, CHEM, n_val)
        for v, d in ((sp_val, d_ca),):  # report actual allocated totals (rounding, >= 1 shot per lag)
            for h in v["history"]:
                h["actual_total_shots"] = int(2 * spe_shots(d, h["budget"], len(g_ca)).sum())
        costs = [asdict(krylov_cost(D_KRYLOV, kr_val["budget"], step_ns, prep_ns, rzz_per_step)),
                 asdict(spe_cost(d_ca, sp_val["budget"], step_ns, prep_ns, rzz_per_step, len(g_ca)))]
        resources = {"label": "ideal-gate resource estimate (exact g, shot noise only)",
                     "spe_d_exact_for_ca": d_exact, "spe_d_for_ca": d_ca,
                     "krylov_coarse_shots_per_circuit": kr_m0, "spe_coarse_total_budget": sp_m0,
                     "krylov_curve": kr_curve, "spe_curve": sp_curve,
                     "krylov_validation": kr_val, "spe_validation": sp_val, "n_validation_trials": n_val,
                     "costs_at_ca": costs, "run_d63": comp.resources(D_SPE), "run_D2": comp.resources(D_KRYLOV)}
        res_cache.parent.mkdir(parents=True, exist_ok=True)
        res_cache.write_text(json.dumps(resources, default=float))
    kv, sv = resources["krylov_validation"], resources["spe_validation"]
    print(f"resources: Krylov {kv['budget']} shots/circuit (RMSE {1e3 * kv['stats']['rmse']:.2f} mHa); "
          f"SPE d={resources['spe_d_for_ca']} {resources['costs_at_ca'][1]['shots_total']:.0f} shots "
          f"(RMSE {1e3 * sv['stats']['rmse']:.2f} mHa, tail {sv['stats']['frac_above_10mHa']:.1e}); "
          f"exact-estimator d for 1.6 mHa: {resources['spe_d_exact_for_ca']} ({time.time() - t0:.0f}s)")

    # ---------------- noisy runs (same realizations for every configuration) ----------------
    op = DEFAULT_OPERATING_POINT
    sens_cfgs = sensitivity_configs(op, REL_STEP)
    sw_cfgs, sw_keys = sweep_configs(op, FACTORS)
    configs = sens_cfgs + [SPINPULSE_EXAMPLE_POINT] + sw_cfgs
    i_ex = len(sens_cfgs)
    sl_sw = slice(i_ex + 1, None)

    est = {"krylov": krylov_estimator(D_KRYLOV, tau, prob.c0), "spe": spe_estimator(D_SPE, tau, prob.c0)}
    noisy = {}
    for name, k_max, n in (("krylov", K_KRYLOV, n_k), ("spe", D_SPE, n_s)):
        t = time.time()
        G = cached_samples(name, comp, k_max, configs, n, args.seed, args.jobs, not args.no_fast_propagate, deadline,
                           patched=patched)
        print(f"{name}: {len(configs)} configs x {n} realizations in {time.time() - t:.0f}s")
        f = est[name]
        sens = sensitivity_from_samples(G[:i_ex], op, REL_STEP, f)
        curves = sweep_curves(G[sl_sw], sw_keys, f, op)
        e_ex, e_ex_se = jackknife(G[i_ex], f)
        entry = {
            "n_realizations": n,
            "k_max": k_max,
            "sensitivity": sens.to_dict(),
            "sweeps": [asdict(c) for c in curves],
            "budget_gamma": {c.source: chemical_accuracy_budget(c, e0, CHEM) for c in curves},
            "points": {
                "noiseless_pulse": {"energy": sens.energy_noiseless, "se": 0.0},
                "operating_point": {"energy": sens.energy, "se": sens.energy_se, "noise": op.as_dict()},
                "spinpulse_example": {"energy": e_ex, "se": e_ex_se, "noise": SPINPULSE_EXAMPLE_POINT.as_dict()},
            },
            "g_mean": {"noiseless_pulse": c2l(G[0].mean(0)), "operating_point": c2l(G[1].mean(0)),
                       "spinpulse_example": c2l(G[i_ex].mean(0))},
        }
        if name == "spe":  # estimator failures (no interior CDF crossing) on the averaged data
            rows = list(range(i_ex + 1, len(configs)))
            fails = [not spe_crossing_energy(G[r].mean(0), D_SPE, tau, prob.c0).success for r in rows]
            entry["sweep_failed"] = {src: [f for (s_, _), f in zip(sw_keys, fails) if s_ == src]
                                     for src in dict.fromkeys(k for k, _ in sw_keys)}
            entry["n_sweep_failed"] = int(sum(fails))
        if name == "krylov":
            entry["vs_dim_kept"] = {key: {str(th): [krylov_energy(G[idx].mean(0), D, tau, prob.c0, th).kept
                                                    for D in dims] for th in KRYLOV_THRESHOLDS}
                                    for key, idx in (("operating_point", 1), ("spinpulse_example", i_ex))}
            entry["vs_dim"] = {}
            for key, idx in (("operating_point", 1), ("spinpulse_example", i_ex)):
                entry["vs_dim"][key] = {
                    str(th): [jackknife(G[idx], krylov_estimator(D, tau, prob.c0, th)) for D in dims] for th in KRYLOV_THRESHOLDS
                }
        else:
            degs = list(range(7, D_SPE + 1, 8))
            entry["vs_degree"] = {"degrees": degs}
            for key, idx in (("noiseless_pulse", 0), ("operating_point", 1), ("spinpulse_example", i_ex)):
                entry["vs_degree"][key] = [jackknife(G[idx], spe_estimator(d, tau, prob.c0)) for d in degs]
        noisy[name] = entry
        s = sens
        print(f"  E(op) - E0 = {1e3 * (s.energy - e0):.2f} +- {1e3 * s.energy_se:.2f} mHa;"
              f" noise shift {1e3 * s.noise_shift:.2f} mHa")
        for src in s.relative:
            print(f"   {src:12s} dE/dG {1e3 * s.dE_dGamma[src]:9.3f} +- {1e3 * s.dE_dGamma_se[src]:.3f} mHa.us"
                  f" | relative {s.relative[src]:.3f} +- {s.relative_se[src]:.3f}")

    noisy["krylov_high_statistics"] = krylov_high_statistics(comp, prob, op, n_hi, args.seed, args.jobs,
                                                             not args.no_fast_propagate, deadline, patched)

    results = {
        "meta": {"quick": args.quick, "seed": args.seed, "runtime_s": round(time.time() - t0, 1),
                 "python": platform.python_version(), "fast_propagate": not args.no_fast_propagate,
                 "spinpulse_patch": patched, "pink_segment_ns": PINK_SEGMENT_NS,
                 "seeds": {"krylov": args.seed, "spe": args.seed, "krylov_high_statistics": args.seed + 1},
                 "versions": VERSIONS,
                 "units": {"energy": "Hartree", "gamma": "1/us (Gamma = 1/T*)", "dE_dGamma": "Hartree*us"}},
        "problem": {"molecule": "H2", "basis": "sto-3g", "bond_length_A": prob.bond_length, "qubits_jw": 4,
                    "qubits_tapered": 1, "c0": prob.c0, "gz": prob.gz, "gx": prob.gx, "hf_energy": prob.hf_energy,
                    "fci_energy": prob.fci_energy, "exact_energies": prob.exact_energies.tolist(),
                    "p0": prob.ground_state_overlap, "tau": tau},
        "hardware": {"step_ns": step_ns, "prep_ns": prep_ns, "step_ops": dict(comp.step.isa.count_ops()),
                     "prep_ops": dict(comp.prep.isa.count_ops())},
        "settings": {"D_krylov": D_KRYLOV, "d_spe": D_SPE, "factors": FACTORS.tolist(), "rel_step": REL_STEP,
                     "spe_p0_lower_bound": 0.5, "krylov_threshold_main": 1e-8, "chemical_accuracy": CHEM},
        "noiseless": noiseless,
        "resources": resources,
        "noisy": noisy,
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote {out} ({results['meta']['runtime_s']} s)")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py"), str(out)], check=True)


if __name__ == "__main__":
    main()
