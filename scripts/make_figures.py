"""Figures from a results JSON:  python scripts/make_figures.py results/results.json"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

# Categorical palette in fixed order (validated for colour-vision deficiency); markers add a second channel.
SRC = {
    "quasistatic": dict(color="#2a78d6", marker="o", label="Quasi-static (freq.)"),
    "white": dict(color="#eb6834", marker="s", label="White (freq.)"),
    "pink": dict(color="#1baf7a", marker="D", label="1/f pink (freq.)"),
    "exchange": dict(color="#eda100", marker="^", label="Exchange J (1/f)"),
}
ALG = {"krylov": dict(color="#2a78d6", label="Krylov (QKSD, D = 2)"), "spe": dict(color="#eb6834", label="SPE (d = 63)")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
MHA = 1e3


def logerr(v, se):
    """Asymmetric error bars that stay positive on log axes."""
    v, se = np.asarray(v, float), np.asarray(se, float)
    return np.vstack([np.minimum(se, 0.85 * v), se])


def chem_line(ax, chem, x_text, ha="right", below=False, label="chemical accuracy (1.6 mHa)"):
    ax.axhline(chem, color=INK, lw=1, ls="--", zorder=2)
    ax.text(x_text, chem * (0.6 if below else 1.18), label, fontsize=8, color=INK, ha=ha)


def fig_convergence(res, out):
    """Fig 1: |dE| vs Krylov dimension and vs SPE degree, noiseless and noisy."""
    e0 = res["problem"]["fci_energy"]
    chem = res["settings"]["chemical_accuracy"] * MHA
    nl, kn, sn = res["noiseless"], res["noisy"]["krylov"], res["noisy"]["spe"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.32))

    floor = 1e-2
    dims = nl["krylov_dims"]
    ke = np.abs(nl["krylov_exact"]) * MHA
    exact_pts = ke < floor
    a1.plot(np.array(dims)[~exact_pts], ke[~exact_pts], color=MUTED, marker="o", ms=6, lw=0, label="exact g (no noise)")
    a1.plot(np.array(dims)[exact_pts], np.full(exact_pts.sum(), floor * 1.25), color=MUTED, marker="v", ms=7, lw=0,
            label="exact g: < 10⁻⁹ mHa")
    a1.plot(dims, np.maximum(np.abs(nl["krylov_pulse"]) * MHA, floor), color=MUTED, lw=1.2, ls="--",
            label="noiseless pulses")
    for key, st, lab in (("spinpulse_example", dict(color="#1baf7a", marker="D"), "SpinPulse example noise"),
                         ("operating_point", dict(color="#2a78d6", marker="s"), "operating point noise")):
        for th, ls in (("0.001", "-"), ("0.01", ":")):
            v = kn["vs_dim"][key][th]
            e = np.array([x[0] for x in v]) - e0
            se = np.array([x[1] for x in v])
            a1.errorbar(dims, np.abs(e) * MHA, yerr=logerr(np.abs(e) * MHA, se * MHA), color=st["color"], marker=st["marker"], ms=5, lw=1.8,
                        ls=ls, capsize=2, mec="white", mew=0.7)
    a1.set_yscale("log")
    a1.set_xlabel("Krylov dimension D")
    a1.set_ylabel("|E − E_FCI|  (mHa)")
    a1.set_title("Real-time Krylov (QKSD)", loc="left", fontsize=10.5, color=INK)
    chem_line(a1, chem, 6.1, "right", below=True)
    from matplotlib.lines import Line2D

    handles = a1.get_legend_handles_labels()[0] + [
        Line2D([], [], color="#1baf7a", marker="D", lw=1.8, label="SpinPulse example noise"),
        Line2D([], [], color="#2a78d6", marker="s", lw=1.8, label="operating point noise"),
        Line2D([], [], color=INK, lw=1.5, ls="-", label="S threshold 10⁻³"),
        Line2D([], [], color=INK, lw=1.5, ls=":", label="S threshold 10⁻²"),
    ]
    a1.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left", ncol=2, bbox_to_anchor=(-0.02, -0.16))
    a1.set_ylim(floor, 5e3)

    d = np.array(nl["spe_degrees"])
    a2.plot(d, np.abs(nl["spe_crossing"]) * MHA, color=MUTED, marker="o", ms=5, lw=1.5, label="exact g, CDF crossing")
    a2.plot(d, np.array(nl["spe_delta"]) * MHA, color=MUTED, lw=1, ls=":", label="guarantee δ/τ")
    vd = sn["vs_degree"]
    degs = np.array(vd["degrees"])
    for key, st, lab in (("spinpulse_example", dict(color="#1baf7a", marker="D"), "SpinPulse example noise"),
                         ("operating_point", dict(color="#2a78d6", marker="s"), "operating point noise")):
        e = np.array([x[0] for x in vd[key]]) - e0
        se = np.array([x[1] for x in vd[key]])
        a2.errorbar(degs, np.abs(e) * MHA, yerr=logerr(np.abs(e) * MHA, se * MHA), color=st["color"], marker=st["marker"], ms=5, lw=1.8,
                    capsize=2, mec="white", mew=0.7, label=lab)
    a2.axvspan(degs[-1] * 1.08, d[-1] * 1.3, color=GRID, alpha=0.5, zorder=0)
    a2.text(degs[-1] * 1.2, 2.2e3, "not simulated with noise\n(runs > 237 µs)", fontsize=8, color=MUTED, va="top")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("SPE degree d  (= longest run, in controlled steps)")
    a2.set_title("Statistical phase estimation (CDF)", loc="left", fontsize=10.5, color=INK)
    chem_line(a2, chem, d[-1] * 1.25)
    a2.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2, bbox_to_anchor=(-0.02, -0.16))
    a2.set_ylim(floor, 5e3)
    fig.suptitle("H₂ ground-state energy error, noiseless vs. SpinPulse noise (simulation)", x=0.06, ha="left",
                 fontsize=11, color=INK, y=1.01)
    fig.savefig(out)
    plt.close(fig)


def fig_sweeps(res, out):
    """Fig 2: noise-induced shift |E - E_noiseless| vs noise rate per source, both algorithms."""
    chem = res["settings"]["chemical_accuracy"] * MHA
    tau, c0 = res["problem"]["tau"], res["problem"]["c0"]
    e_floor = -np.pi / 3 / tau + c0  # bottom of the SPE search window: estimator failed
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, alg in zip(axes, ("krylov", "spe")):
        n = res["noisy"][alg]
        free = n["points"]["noiseless_pulse"]["energy"]
        failed_any = False
        for c in n["sweeps"]:
            st = SRC[c["source"]]
            g = np.array(c["gammas"])
            en = np.array(c["energy"])
            e = np.abs(en - free) * MHA
            se = np.array(c["energy_se"]) * MHA
            failed = (np.array(n["sweep_failed"][c["source"]], dtype=bool) if "sweep_failed" in n
                      else np.isclose(en, e_floor, atol=1e-9))
            ok = ~failed
            ax.errorbar(g[ok], e[ok], yerr=logerr(e[ok], se[ok]), color=st["color"], marker=st["marker"], ms=5, lw=1.8,
                        capsize=2, mec="white", mew=0.7, label=st["label"], zorder=3)
            if failed.any():
                failed_any = True
                ax.plot(g[failed], e[failed], ls="none", marker="x", ms=7, mew=1.6, color=st["color"], zorder=3)
            i0 = int(np.argmin(abs(np.log(g / c["default_gamma"]))))
            ax.plot(g[i0], e[i0], marker=st["marker"], ms=11, mfc="none", mec=INK, mew=1.1, zorder=4)
        chem_line(ax, chem, 1.2e-4, "left", label="1.6 mHa additional-noise budget")
        if failed_any:
            ax.text(0.02, 0.97, "× estimator failed\n   (crossing at edge of search window)", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color=MUTED)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e3)
        ax.set_xlabel("Noise rate Γ = 1/T*  (1/µs)")
        ax.set_title(ALG[alg]["label"] + f"  ·  N = {n['n_realizations']}", loc="left", fontsize=10.5, color=INK)
    axes[0].set_ylabel("noise-induced shift |E − E_noiseless|  (mHa)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].text(0.99, 0.02, "each source alone · open marker = operating point · ±1 s.e.", transform=axes[1].transAxes,
                 ha="right", fontsize=7.5, color=MUTED)
    fig.savefig(out)
    plt.close(fig)


def sensitivity_source(res, alg):
    """Krylov: prefer the high-statistics run when present."""
    if alg == "krylov" and "krylov_high_statistics" in res["noisy"]:
        hi = res["noisy"]["krylov_high_statistics"]
        return hi["sensitivity"], hi["n_realizations"]
    return res["noisy"][alg]["sensitivity"], res["noisy"][alg]["n_realizations"]


def fig_sensitivity(res, out):
    """Fig 3: dE/dGamma * Gamma (mHa) per source, Krylov vs SPE."""
    srcs = list(SRC)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(wspace=0.45))
    y = np.arange(len(srcs))
    hgt = 0.36
    for i, alg in enumerate(("krylov", "spe")):
        s, n_real = sensitivity_source(res, alg)
        g0 = s["operating_point"]
        absv = [g0[k] * s["dE_dGamma"][k] * MHA for k in srcs]
        abse = [g0[k] * s["dE_dGamma_se"][k] * MHA for k in srcs]
        rel = [s["relative"][k] for k in srcs]
        rele = [s["relative_se"][k] for k in srcs]
        off = (0.5 - i) * hgt
        a1.barh(y + off, absv, xerr=abse, height=hgt * 0.92, color=ALG[alg]["color"],
                label=f"{ALG[alg]['label']}, N = {n_real}",
                error_kw=dict(ecolor=INK, lw=0.9, capsize=2), zorder=3)
        a2.barh(y + off, rel, xerr=rele, height=hgt * 0.92, color=ALG[alg]["color"],
                error_kw=dict(ecolor=INK, lw=0.9, capsize=2), zorder=3)
    for a in (a1, a2):
        a.set_yticks(y, [SRC[k]["label"] for k in srcs], color=INK)
        a.axvline(0, color=MUTED, lw=0.8)
        a.grid(axis="y", visible=False)
        a.invert_yaxis()
    a1.set_xlabel("Γ · ∂E/∂Γ  (mHa)   — energy shift per relative change of Γ")
    a2.set_xlabel("(Γ/δE_noise) · ∂E/∂Γ   — does not sum to 1 (non-additive)")
    a1.set_title("Absolute sensitivity at the operating point", loc="left", fontsize=10.5, color=INK)
    a2.set_title("Relative sensitivity (elasticity)", loc="left", fontsize=10.5, color=INK)
    fig.legend(*a1.get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.savefig(out)
    plt.close(fig)


def main(path):
    res = json.loads(Path(path).read_text())
    FIG.mkdir(exist_ok=True)
    sfx = "_quick" if res["meta"].get("quick") else ""
    fig_convergence(res, FIG / f"fig1_convergence{sfx}.png")
    fig_sweeps(res, FIG / f"fig2_sweeps{sfx}.png")
    fig_sensitivity(res, FIG / f"fig3_sensitivity{sfx}.png")
    print("figures written to", FIG)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "results.json")
