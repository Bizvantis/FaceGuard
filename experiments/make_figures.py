"""Figures (vector PDF) and a LaTeX macro file with every number quoted in the paper.

Reads experiments/results/*.json and writes paper/figures/*.pdf and paper/generated/numbers.tex.
Colours: validated light-mode categorical palette (blue / orange / aqua) with redundant line styles and
direct labels so the figures also survive grey-scale printing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import FIGURES, RESULTS, ROOT  # noqa: E402

GEN = ROOT / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
     "grid": "#e1e0d9", "base": "#c3c2b7"}
COL, DCOL = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": C["base"], "axes.labelcolor": C["ink2"], "text.color": C["ink"],
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.3, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"[fig] {name}.pdf")


def panel_label(ax, text):
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom", color=C["ink"])


# ---------------------------------------------------------------- figures
def fig_parallax(par):
    A = np.array(par["angles"])
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.25), sharey=True)
    series = [("head", "Real head", C["blue"], "-"), ("flat", "Flat print", C["orange"], "--"), ("bent_R40", "Bent print ($R$=40 mm)", C["aqua"], ":")]
    for ax, key, thr, title in ((axes[0], "ratio", 18.0, "Yaw-ratio estimate"), (axes[1], "parallax", 0.8 * 18.0, "Homography-parallax estimate")):
        for name, label, color, ls in series:
            m = np.array(par[name][f"{key}_deg_mean"])
            s = np.array(par[name][f"{key}_deg_std"])
            ax.plot(A, m, color=color, ls=ls, label=label)
            ax.fill_between(A, m - s, m + s, color=color, alpha=0.10, lw=0)
        for sgn in (1, -1):
            ax.axhline(sgn * thr, color=C["ink"], lw=0.7, ls=(0, (4, 3)))
        ax.text(-58, thr + 2, f"threshold {thr:.1f}°", fontsize=6.5, color=C["ink2"])
        ax.set_xlabel("Rotation of the presented face / print (deg)")
        ax.set_title(title, color=C["ink"])
        ax.set_xlim(-60, 60)
        ax.set_ylim(-75, 75)
    axes[0].set_ylabel("Estimated yaw change (deg)")
    axes[0].legend(loc="lower right")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_parallax")


def fig_roc(lfw):
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for key, label, color, ls in (("full", "FaceGuard (aligned + flip TTA)", C["blue"], "-"),
                                  ("no_tta", "Without flip TTA", C["orange"], "--"),
                                  ("no_alignment", "Without landmark alignment", C["aqua"], ":")):
        v = lfw["verification"][key]
        far, tar = np.array(v["roc_far"]), np.array(v["roc_tar"])
        ok = far > 0
        ax.plot(far[ok], tar[ok], color=color, ls=ls, label=f"{label} ({100 * v['accuracy']:.2f}%)")
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 1)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("False accept rate (FAR)")
    ax.set_ylabel("True accept rate (TAR)")
    ax.legend(loc="lower right")
    save(fig, "fig_roc")


def fig_scores(lfw):
    op = lfw["operating_points"]
    e = np.array(op["hist"]["edges"])
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for key, label, color, ls in (("genuine", f"Same person ({op['n_genuine']:,} pairs)", C["blue"], "-"),
                                  ("impostor", f"Different people ({op['n_impostor']:,} pairs)", C["orange"], "--")):
        h = np.array(op["hist"][key])
        h = np.where(h > 0, h, np.nan)
        ax.step(e[:-1], h, where="post", color=color, ls=ls, label=label)
    for name, t in (("relaxed", 0.38), ("balanced", 0.45), ("strict", 0.52)):
        ax.axvline(t, color=C["ink"], lw=0.6, ls=(0, (4, 3)))
        # labels in the empty band above both curves, clear of the data
        ax.text(t + 0.006, 0.9, name, rotation=90, fontsize=6, color=C["ink2"], va="top")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.2)
    ax.set_xlim(-0.3, 1.0)
    ax.set_xlabel("Cosine similarity of the two face embeddings")
    ax.set_ylabel("Share of pairs per bin")
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    # legend above the plot area so it never covers the threshold lines
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.0, handlelength=1.6, borderaxespad=0.2)
    save(fig, "fig_scores")


def fig_openset(lfw):
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for key, label, color, ls in (("top3", "Top-3 mean (FaceGuard)", C["blue"], "-"), ("nn", "Nearest neighbor", C["orange"], "--"),
                                  ("centroid", "Nearest centroid", C["aqua"], ":")):
        cv = lfw["open_set"]["curves"][key]
        far, dr = np.array(cv["far"]), np.array(cv["dir"])
        ok = far > 0
        ax.plot(far[ok], dr[ok], color=color, ls=ls, label=label)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0.9, 1.0)
    ax.set_xlabel("False alarm rate (unknown people accepted)")
    ax.set_ylabel("Detection & identification rate")
    ax.legend(loc="lower right")
    save(fig, "fig_openset")


def fig_robust(rob):
    rows = rob["conditions"]
    clean = next(r for r in rows if r["kind"] == "clean")["tar_at_thr"] * 100
    specs = [("blur", "Gaussian blur $\\sigma$ (px)", lambda r: r["level"], False),
             ("gain", "Brightness gain", lambda r: r["level"], True),
             ("scale", "Face width (px)", lambda r: r["face_px"], True),
             ("jpeg", "JPEG quality", lambda r: r["level"], True),
             ("noise", "Noise $\\sigma$ (gray levels)", lambda r: r["level"], False)]
    fig, axes = plt.subplots(1, 5, figsize=(DCOL, 1.55), sharey=True)
    for ax, (kind, xlabel, xf, logx) in zip(axes, specs):
        rs = [r for r in rows if r["kind"] == kind]
        x = np.array([xf(r) for r in rs], float)
        y = np.array([100 * r["tar_at_thr"] for r in rs])
        if kind in ("blur", "noise"):
            x, y = np.r_[0, x], np.r_[clean, y]
        order = np.argsort(x)
        ax.plot(x[order], y[order], color=C["blue"], marker="o", ms=3.2, mec="white", mew=0.6)
        ax.axhline(clean, color=C["muted"], lw=0.7, ls=(0, (4, 3)))
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if kind == "gain":
            ax.set_xticks([0.15, 0.3, 0.5, 1, 2, 3])
            ax.set_xticklabels(["0.15", "0.3", "0.5", "1", "2", "3"])
        if kind == "scale":
            ax.set_xticks([11, 17, 24, 34, 47, 93])
            ax.set_xticklabels(["11", "17", "24", "34", "47", "93"])
        if kind == "jpeg":
            ax.set_xticks([4, 8, 15, 30, 100])
            ax.set_xticklabels(["4", "8", "15", "30", "orig."])
            ax.plot([100], [clean], color=C["blue"], marker="o", ms=3.2, mec="white", mew=0.6)
        if kind == "gain":
            ax.plot([1.0], [clean], color=C["blue"], marker="o", ms=3.2, mec="white", mew=0.6)
        if kind == "scale":
            ax.plot([93], [clean], color=C["blue"], marker="o", ms=3.2, mec="white", mew=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("TAR at 0.45 (%)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_robust")


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def fisher_p(k1, n1, k2, n2):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def fig_liveness(liv):
    sweep = liv["tau_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.05))
    ax = axes[0]
    tops = []
    for (noise, color, ls, mk), dx in zip(((0.5, C["blue"], "-", "o"), (1.0, C["orange"], "--", "s"), (2.0, C["aqua"], ":", "^")),
                                          (-0.35, 0.0, 0.35)):
        pts = sorted((s["tau"], s["frr"], s.get("n", 400)) for s in sweep if s["noise_px"] == noise)
        x = np.array([p[0] for p in pts], float) + dx  # small offsets keep the error bars apart
        y = np.array([100 * p[1] for p in pts])
        ci = np.array([wilson(round(p[1] * p[2]), p[2]) for p in pts]) * 100
        ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], color=color, ls=ls, marker=mk, ms=3.2, mec="white", mew=0.6,
                    elinewidth=0.6, capsize=1.5, label=f"landmark noise {noise:g} px")
        tops.append(ci[:, 1].max())
    top = max(tops)
    ax.set_ylim(0, top * 1.25)
    ax.axvline(18, color=C["ink"], lw=0.6, ls=(0, (4, 3)))
    ax.text(18.3, top * 1.12, "default", fontsize=6.5, color=C["ink2"])
    ax.set_xticks(sorted({s["tau"] for s in sweep}))
    ax.set_xlabel("Head-turn threshold $\\tau$ (deg)")
    ax.set_ylabel("Live users rejected, BPCER (%)")
    # legend above the plot area so it never hides a curve
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=0.9, handlelength=1.8, borderaxespad=0.2)
    panel_label(ax, "(a)")
    ax = axes[1]
    h = np.array(liv["live"]["full"]["time_hist"], float)
    ax.bar(np.arange(len(h)) + 0.5, 100 * h / h.sum(), width=0.8, color=C["blue"])
    ax.set_xlim(0, 20)
    ax.set_xlabel("Time to pass the 3-action check (s)")
    ax.set_ylabel("Sessions (%)")
    ax.grid(axis="x", visible=False)
    panel_label(ax, "(b)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_liveness")


def fig_runtime(rt):
    st = rt["stages"]["640x480"]  # the default camera resolution
    items = [("Face detection (YuNet)", "detect"), ("Alignment (similarity warp)", "align"), ("Embedding (SFace)", "embed"),
             ("Embedding + flip TTA", "embed_tta"), ("Face mesh (MediaPipe)", "mesh"),
             ("Full pipeline, per frame", "pipeline_default"), ("Full pipeline + mesh", "pipeline_with_mesh")]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    y = np.arange(len(items))[::-1]
    means = [st[k]["mean"] for _, k in items]
    stds = [st[k]["std"] for _, k in items]
    ax.barh(y, means, height=0.55, color=C["blue"], xerr=stds, error_kw={"elinewidth": 0.6, "ecolor": C["ink2"], "capsize": 1.5})
    ends = [m + s for m, s in zip(means, stds)]
    for yi, m, e in zip(y, means, ends):  # label just past the error bar, not on top of it
        ax.text(e + max(ends) * 0.03, yi, f"{m:.2f} ms" if m < 0.1 else f"{m:.1f} ms", va="center", fontsize=6.5, color=C["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels([t for t, _ in items])
    ax.set_xlim(0, max(ends) * 1.3)
    ax.set_xlabel("Latency at 640$\\times$480 (ms, mean $\\pm$ s.d.)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_runtime")


# ---------------------------------------------------------------- numbers
def macros(res: dict) -> str:
    lines = ["% Generated by experiments/make_figures.py - do not edit by hand."]

    def m(name, value, fmt="{:.2f}"):
        lines.append(f"\\newcommand{{\\{name}}}{{{fmt.format(value) if not isinstance(value, str) else value}}}")

    pct = lambda v: 100 * v  # noqa: E731
    lfw = res.get("lfw")
    if lfw:
        d = lfw["dataset"]
        m("lfwImages", d["images"], "{:,}"), m("lfwPeople", d["people"], "{:,}"), m("lfwPeopleTen", d["people_ge10"], "{}")
        m("lfwPeopleTwenty", d["people_ge20"], "{}"), m("lfwMultiFace", d["multi_face_images"], "{:,}")
        m("lfwFaceWidth", d["face_width_px_median"], "{:.0f}"), m("lfwYawMedian", d["abs_yaw_median"], "{:.1f}"), m("lfwYawNinety", d["abs_yaw_p90"], "{:.1f}")
        for key, tag in (("full", "Full"), ("no_tta", "NoTTA"), ("no_alignment", "NoAlign")):
            v = lfw["verification"][key]
            m(f"lfwAcc{tag}", pct(v["accuracy"])), m(f"lfwStd{tag}", pct(v["std"])), m(f"lfwAuc{tag}", v["auc"], "{:.4f}")
            m(f"lfwEer{tag}", pct(v["eer"])), m(f"lfwTarTwo{tag}", pct(v["tar_at_far_1e-2"])), m(f"lfwTarThree{tag}", pct(v["tar_at_far_1e-3"]))
            m(f"lfwThr{tag}", v["threshold"], "{:.3f}"), m(f"lfwGen{tag}", v["genuine_mean"], "{:.3f}"), m(f"lfwImp{tag}", v["impostor_mean"], "{:.3f}")
        op = lfw["operating_points"]
        m("opGenuine", op["n_genuine"], "{:,}"), m("opImpostor", op["n_impostor"], "{:,}"), m("opImpMax", op["impostor_max"], "{:.3f}")
        m("opImpQ", op["impostor_p9999"], "{:.3f}")
        for lvl, tag in (("relaxed", "Relaxed"), ("balanced", "Balanced"), ("strict", "Strict")):
            L = op["levels"][lvl]
            m(f"far{tag}", pct(L["far"]), "{:.4f}"), m(f"frr{tag}", pct(L["frr"]))
        m("unkGenuine", pct(op["levels"]["unknown_zone_0.30"]["genuine_below"])), m("unkImpostor", pct(op["levels"]["unknown_zone_0.30"]["impostor_below"]))
        m("adaptiveFar", pct(op["levels"]["adaptive_gate_0.57"]["far"]), "{:.4f}")
        idn = lfw["identification"]
        m("idPeople", idn["people"], "{}"), m("idImages", idn["images"], "{:,}")
        for g in (1, 2, 3, 5, 8):
            for strat, stag in (("nn", "Nn"), ("top3", "Topthree"), ("top5", "Topfive"), ("centroid", "Centroid")):
                k = f"g{g}/{strat}"
                if k in idn:
                    m(f"id{['', 'One', 'Two', 'Three', '', 'Five', '', '', 'Eight'][g]}{stag}", pct(idn[k]["mean"]))
        os_ = lfw["open_set"]
        for strat, tag in (("nn", "Nn"), ("top3", "Topthree"), ("centroid", "Centroid")):
            m(f"os{tag}DirTwo", pct(os_[f"{strat}/dir_at_far_0.01"]["mean"])), m(f"os{tag}DirThree", pct(os_[f"{strat}/dir_at_far_0.001"]["mean"]))
            for lvl, lt in (("relaxed", "Relaxed"), ("balanced", "Balanced"), ("strict", "Strict")):
                m(f"os{tag}Dir{lt}", pct(os_[f"{strat}/{lvl}/dir"]["mean"])), m(f"os{tag}Far{lt}", pct(os_[f"{strat}/{lvl}/far"]["mean"]))
        m("osUnknown", os_["unknown_probes"]["mean"], "{:,.0f}"), m("osKnown", os_["known_probes"]["mean"], "{:,.0f}"), m("osEnrolled", os_["enrolled_people"], "{}")
        for name, v in lfw["classifiers"].items():
            tag = {"Top-3 cosine (FaceGuard)": "Topthree", "1-NN cosine": "Nn", "Nearest centroid": "Centroid", "Linear SVM": "Svm",
                   "k-NN (k=5, cosine)": "Knn", "Logistic regression": "Logreg"}[name]
            m(f"cls{tag}", pct(v["mean"])), m(f"cls{tag}Std", pct(v["std"]))
        pe = lfw["pose_enrollment"]
        for s in ("frontal", "diverse", "random"):
            m(f"pose{s.capitalize()}Side", pe[f"{s}/own_score_nonfrontal"]["mean"], "{:.3f}")
            m(f"pose{s.capitalize()}Front", pe[f"{s}/own_score_frontal"]["mean"], "{:.3f}")
            m(f"pose{s.capitalize()}Rank", pct(pe[f"{s}/rank1"]["mean"]))
        m("posePeople", pe["people"], "{}"), m("poseSideFrac", pct(pe["nonfrontal_fraction"]["mean"]), "{:.1f}")
    oli = res.get("olivetti")
    if oli:
        m("oliRankOne", pct(oli["closed_set_accuracy"])), m("oliAuc", oli["auc"], "{:.4f}"), m("oliEer", pct(oli["eer"]))
        m("oliTarTwo", pct(oli["tar_at_far"]["0.01"])), m("oliTarThree", pct(oli["tar_at_far"]["0.001"]))
        m("oliDir", pct(oli["open_set_dir"])), m("oliFar", pct(oli["open_set_far"])), m("oliMs", oli["ms_per_image"], "{:.1f}")
    rob = res.get("robustness")
    if rob:
        m("robPairs", rob["pairs"], "{:,}")
        pick = {("clean", 0): "robClean", ("blur", 2): "robBlurTwo", ("blur", 4): "robBlurFour", ("gain", 0.15): "robGainLow",
                ("gain", 2.0): "robGainTwo", ("gain", 3.0): "robGainThree", ("scale", 0.25): "robScaleQuarter", ("scale", 0.18): "robScaleSmall",
                ("jpeg", 15): "robJpegFifteen", ("jpeg", 4): "robJpegFour", ("noise", 16): "robNoiseSixteen", ("noise", 32): "robNoiseThirtyTwo",
                ("mask", 1): "robMask", ("sunglasses", 1): "robSunglasses"}
        for r in rob["conditions"]:
            name = pick.get((r["kind"], r["level"]))
            if name:
                m(name, pct(r["tar_at_thr"]))
                m(name + "Auc", r["auc"], "{:.4f}")
        m("robMaxFar", pct(max(r["far_at_thr"] for r in rob["conditions"])))
        m("robScaleQuarterPx", next(r["face_px"] for r in rob["conditions"] if (r["kind"], r["level"]) == ("scale", 0.25)), "{:.0f}")
        m("robScaleSmallPx", next(r["face_px"] for r in rob["conditions"] if (r["kind"], r["level"]) == ("scale", 0.18)), "{:.0f}")
        m("robScaleSmallDet", pct(next(r["detected"] for r in rob["conditions"] if (r["kind"], r["level"]) == ("scale", 0.18))), "{:.1f}")
        if "impostor_comparisons" in rob:
            m("robImpN", rob["impostor_comparisons"], "{:,}"), m("robImpGallery", rob["impostor_gallery"], "{:,}")
            m("robFarLargeMax", pct(max(r["far_large"] for r in rob["conditions"])), "{:.4f}")
            m("robFalseAcceptsTotal", sum(r["false_accepts_large"] for r in rob["conditions"]), "{:,}")
            m("robImpMaxLarge", max(r["impostor_max_large"] for r in rob["conditions"]), "{:.3f}")
            m("robImpTotal", rob["impostor_comparisons"] * len(rob["conditions"]), "{:,}")
            worst = max(rob["conditions"], key=lambda r: r["far_large"])
            m("robFarWorstKind", {"blur": "blur", "gain": "brightness gain", "scale": "down-scaling", "jpeg": "JPEG compression",
                                  "noise": "noise", "mask": "the mask", "sunglasses": "sunglasses", "clean": "clean probes"}[worst["kind"]])
            m("robFarWorstFa", worst["false_accepts_large"], "{:,}")
            m("robFarWorstLevel", f"{worst['level']:g}")
            clean = next(r for r in rob["conditions"] if r["kind"] == "clean")
            m("robFarLargeClean", pct(clean["far_large"]), "{:.4f}"), m("robFaClean", clean["false_accepts_large"], "{:,}")
            m("robFarLargeMin", pct(min(r["far_large"] for r in rob["conditions"])), "{:.5f}")
            m("robFarSevereMax", pct(max(r["far_large"] for r in rob["conditions"]
                                         if (r["kind"], r["level"]) in (("blur", 4), ("blur", 6), ("gain", 3.0), ("scale", 0.12),
                                                                        ("jpeg", 4), ("noise", 32)))), "{:.4f}")
    par = res.get("parallax")
    if par:
        m("parFaces", par["faces"], "{}"), m("parKratio", par["k_ratio"], "{:.3f}"), m("parKpar", par["k_parallax"], "{:.3f}"), m("parNose", par["nose_depth_mm_mean"], "{:.1f}")
        if "faces_cal" in par:
            m("parFacesCal", par["faces_cal"], "{}"), m("parFacesEval", par["faces_eval"], "{}"), m("parSplitSeed", par["split_seed"], "{}")
            m("parKratioEval", par["k_ratio_eval_fit"], "{:.3f}"), m("parKparEval", par["k_parallax_eval_fit"], "{:.3f}")
        m("parFlatParQ", par["flat"]["parallax_p99_abs_deg"], "{:.1f}")
        i25 = par["angles"].index(25)
        for key, tag in (("head", "Head"), ("flat", "Flat"), ("bent_R60", "BentSixty"), ("bent_R40", "BentForty"), ("bent_R30", "BentThirty")):
            m(f"par{tag}RatioTwentyFive", par[key]["ratio_deg_mean"][i25], "{:.1f}"), m(f"par{tag}ParTwentyFive", par[key]["parallax_deg_mean"][i25], "{:.1f}")
            m(f"par{tag}ParMax", par[key]["parallax_max_abs_deg"], "{:.1f}"), m(f"par{tag}RatioMax", par[key]["ratio_max_abs_deg"], "{:.1f}")
    pose = res.get("pose")
    if pose:
        a = pose["agreement"]
        m("poseN", a["n"], "{:,}"), m("poseR", a["yaw_pearson_r"], "{:.3f}"), m("poseSlope", a["yaw_slope"], "{:.3f}"), m("poseMae", a["yaw_mae_deg"], "{:.2f}")
        m("posePitchR", a["pitch_pearson_r"], "{:.2f}"), m("poseJitter", pose["yaw_noise"]["mean_std_deg"], "{:.2f}"), m("poseJitterQ", pose["yaw_noise"]["p95_std_deg"], "{:.2f}")
    liv = res.get("liveness")
    if liv:
        L = liv["live"]
        # Monte-Carlo rates: one decimal (resolution 0.05-0.1 points at 1,000-2,000 sessions)
        m("livFrr", pct(L["full"]["frr"]), "{:.1f}"), m("livTime", L["full"]["time_mean"], "{:.1f}"), m("livTimeQ", L["full"]["time_p95"], "{:.1f}")
        m("livN", L["full"]["n"], "{:,}")
        for k, tag in (("no_parallax", "NoPar"), ("no_order_enforcement", "NoOrder"), ("no_smile_rule", "NoSmile"), ("turns_only", "Turns"),
                       ("noise_2px", "Noise"), ("no_confusion", "NoConf"), ("no_timing", "NoTiming"), ("window_2_5", "WinShort"),
                       ("window_5_0", "WinLong")):
            if k in L:
                m(f"livFrr{tag}", pct(L[k]["frr"]), "{:.1f}")
                m(f"livTime{tag}", L[k]["time_mean"], "{:.1f}")
        A = liv["attacks"]
        for k, v in A.items():
            k2 = k.replace("bent_R60", "bentRSixty").replace("bent_R40", "bentRForty").replace("bent_R30", "bentRThirty")
            k2 = k2.replace("window_2_5", "windowShort").replace("window_5_0", "windowLong")
            parts = [p for p in k2.replace("/", "_").split("_") if p]
            m("att" + "".join(p[0].upper() + p[1:] for p in parts), pct(v["pass_rate"]), "{:.1f}")
        m("livNAbl", L["no_parallax"]["n"], "{:,}")
        m("attPrintN", A["flat/full"]["n"], "{:,}"), m("attReplayN", A["replay/full"]["n"], "{:,}")
        m("attUniversalN", A["universal/full"]["n"], "{:,}")
        m("attPrintBound", 300 / A["flat/full"]["n"], "{:.1f}")  # rule of three: 95% upper bound when 0 of n pass
        s = liv["sequences"]["n=3,expr=True"]
        m("seqDistinct", s["distinct"], "{}"), m("seqMaxP", pct(s["max_prob"])), m("seqMinEnt", s["min_entropy_bits"])
        s2 = liv["sequences"]["n=3,expr=False"]
        m("seqDistinctTurns", s2["distinct"], "{}")
        # 95% Wilson intervals and two-sided Fisher exact tests for the key comparisons
        def cnt(r):
            return (round(r["frr"] * r["n"]) if "frr" in r else round(r["pass_rate"] * r["n"])), r["n"]
        for name, r in (("livFrr", L["full"]), ("attUniversalFull", A["universal/full"]), ("attReplayFull", A["replay/full"]),
                        ("attUniversalNoOrderNoTiming", A["universal/no_order_no_timing"])):
            lo, hi = wilson(*cnt(r))
            m(name + "Lo", pct(lo), "{:.1f}"), m(name + "Hi", pct(hi), "{:.1f}")
        hw_live = max((wilson(*cnt(r))[1] - wilson(*cnt(r))[0]) / 2 for r in L.values())
        hw_rep = max((wilson(*cnt(r))[1] - wilson(*cnt(r))[0]) / 2 for k, r in A.items() if k.startswith("replay"))
        hw_uni = max((wilson(*cnt(r))[1] - wilson(*cnt(r))[0]) / 2 for k, r in A.items() if k.startswith("universal"))
        m("ciHalfLive", pct(hw_live), "{:.1f}"), m("ciHalfReplay", pct(hw_rep), "{:.1f}"), m("ciHalfUniversal", pct(hw_uni), "{:.1f}")
        tests = {"pParallaxBpcer": (L["full"], L["no_parallax"]), "pTimingBpcer": (L["full"], L["no_timing"]),
                 "pUniFullNoOrder": (A["universal/full"], A["universal/no_order_enforcement"]),
                 "pUniFullNoTiming": (A["universal/full"], A["universal/no_timing"]),
                 "pUniFullNeither": (A["universal/full"], A["universal/no_order_no_timing"]),
                 "pRepFullNoOrder": (A["replay/full"], A["replay/no_order_enforcement"]),
                 "pUniShortWindow": (A["universal/full"], A["universal/window_2_5"])}
        for name, (a, b) in tests.items():
            p = fisher_p(*cnt(a), *cnt(b))
            m(name, "<0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.1 else f"{p:.3f}")
        S = liv.get("sensitivity", {})
        for k, tag in (("react_0.5", "ReactFast"), ("react_1.0", "ReactSlow"), ("react_1.4", "ReactSlowest"),
                       ("amp_22", "AmpSmall"), ("amp_38", "AmpLarge"), ("confusion_0.1", "ConfTen"), ("confusion_0.2", "ConfTwenty")):
            if k in S:
                m(f"sens{tag}", pct(S[k]["frr"]), "{:.1f}")
        B = liv.get("baselines", {})
        if B:
            m("baseN", B["yaw_turns/live"]["n"], "{:,}")
            for key, tag in (("yaw_turns", "Yaw"), ("blink_only", "Blink"), ("conventional", "Conv"), ("faceguard", "Fg")):
                for kind, kt in (("live", "Live"), ("flat", "Flat"), ("bent_R40", "Bent"), ("replay", "Replay"), ("universal", "Uni")):
                    r = B[f"{key}/{kind}"]
                    m(f"lb{tag}{kt}", pct(r["frr"] if "frr" in r else r["pass_rate"]), "{:.1f}")
            unis = [B[f"{k}/universal"]["pass_rate"] for k in ("yaw_turns", "blink_only", "conventional")]
            reps = [B[f"{k}/replay"]["pass_rate"] for k in ("yaw_turns", "blink_only", "conventional")]
            m("lbUniMin", pct(min(unis)), "{:.1f}"), m("lbUniMax", pct(max(unis)), "{:.1f}")
            m("lbRepMin", pct(min(reps)), "{:.1f}"), m("lbRepMax", pct(max(reps)), "{:.1f}")
    g = res.get("guard")
    if g:
        o = g["owner_at_desk"]
        m("gOwners", o["owners"], "{}"), m("gHours", o["simulated_hours"], "{:.1f}"), m("gFrames", o["frames"], "{:,}")
        m("gFalseIntruder", o["false_intruder_locks_per_hour"], "{:.2f}"), m("gFalseWalk", o["false_walkaway_locks_per_hour_no_typing"], "{:.2f}")
        m("gMatchFrac", pct(o["status_fraction"].get("match", 0))), m("gUnknownFrac", pct(o["status_fraction"].get("unknown", 0)))
        m("gUncertainFrac", pct(o["status_fraction"].get("uncertain", 0))), m("gLongestGap", max(o["longest_unrecognised_s"]), "{:.1f}")
        shld = o["events"].get("typing/notify/shoulder", 0)
        m("gShldAlerts", shld, "{}"), m("gShldAlertsHour", shld / o["simulated_hours"], "{:.1f}")
        for kind, tag in (("intruder", "Intr"), ("replace", "Repl"), ("shoulder", "Shld")):
            L = g[f"latency_{kind}"]
            m(f"g{tag}Rate", pct(L["detected_within_20s"]), "{:.1f}"), m(f"g{tag}Mean", L["mean_s"] or 0, "{:.2f}")
            m(f"g{tag}Med", L["median_s"] or 0, "{:.2f}"), m(f"g{tag}Q", L["p95_s"] or 0, "{:.2f}"), m(f"g{tag}N", L["trials"], "{}")
            m(f"g{tag}Missed", round(L["trials"] * (1 - L["detected_within_20s"])), "{}")
        missed = lambda k: round(g[f"latency_{k}"]["trials"] * (1 - g[f"latency_{k}"]["detected_within_20s"]))  # noqa: E731
        m("gReplExtra", missed("replace") - missed("intruder"), "{}")
        m("gLockBound", 3.0 / o["simulated_hours"], "{:.2f}")  # rule of three: 95% upper bound with zero events
        m("gMinutes", o["simulated_hours"] * 60 / o["owners"], "{:.0f}")
    gd = res.get("guard_diag")
    if gd:
        a = gd["owner_stream"]["totals"]
        m("gdPhotos", a["photos"], "{}"), m("gdMulti", a["faces>=2"], "{}"), m("gdSecond", a["unknown second face"], "{}")
        m("gdSecondPct", pct(a["unknown second face"] / a["photos"]), "{:.1f}")
        for kind, tag in (("intruder", "Intr"), ("shoulder", "Shld")):
            c = gd["stranger_classification"][kind]
            m(f"gd{tag}Acted", c.get("unknown (acted on)", 0), "{}"), m(f"gd{tag}Quality", c.get("quality gate", 0), "{}")
            m(f"gd{tag}Uncertain", c.get("uncertain band", 0), "{}"), m(f"gd{tag}Owner", c.get("accepted as the owner", 0), "{}")
            m(f"gd{tag}Total", sum(c.values()), "{}")
    rt = res.get("runtime")
    if rt:
        s = rt["stages"]["640x480"]
        for k, tag in (("detect", "Det"), ("align", "Align"), ("embed", "Emb"), ("embed_tta", "EmbTta"), ("mesh", "Mesh"),
                       ("pipeline_default", "Pipe"), ("pipeline_embed_every_frame", "PipeAll"), ("pipeline_with_mesh", "PipeMesh")):
            m(f"rt{tag}", s[k]["mean"], "{:.2f}" if s[k]["mean"] < 0.1 else "{:.1f}")
        m("rtCpuLive", rt["cpu_live_view"]["cpu_percent_of_machine"], "{:.1f}"), m("rtCpuGuard", rt["cpu_guard_background"]["cpu_percent_of_machine"], "{:.1f}")
        m("rtCpuLiveCore", rt["cpu_live_view"]["cpu_percent_of_one_core"], "{:.0f}"), m("rtCpuGuardCore", rt["cpu_guard_background"]["cpu_percent_of_one_core"], "{:.0f}")
        m("rtFps", rt["cpu_live_view"]["frames_displayed_per_s"], "{:.1f}"), m("rtMem", rt["memory_after_benchmark_mb"], "{:.0f}")
        m("rtLoad", rt["load"]["detector_and_recognizer_ms"], "{:.0f}"), m("rtLoadMesh", rt["load"]["mesh_ms"], "{:.0f}")
    return "\n".join(lines) + "\n"


def table_robust(rob) -> str:
    names = {"clean": "Clean probe", "blur": "Gaussian blur, $\\sigma$ (px)", "gain": "Brightness gain", "scale": "Down-scaling factor",
             "jpeg": "JPEG quality", "noise": "Gaussian noise, $\\sigma$", "mask": "Occlusion: face mask", "sunglasses": "Occlusion: sunglasses"}
    rows, last = [], None
    for r in rob["conditions"]:
        if last is not None and r["kind"] != last and r["kind"] != "sunglasses":
            rows.append("\\midrule")
        label = names[r["kind"]] if r["kind"] != last else ""
        level = "--" if r["kind"] in ("clean", "mask", "sunglasses") else f"{r['level']:g}"
        rows.append(f"{label} & {level} & {r['face_px']:.0f} & {100 * r['detected']:.1f} & {r['auc']:.4f} & "
                    f"{100 * r['tar_at_thr']:.1f} & {100 * r['far_at_thr']:.2f} \\\\")
        last = r["kind"]
    return "\n".join(rows) + "\n"


def table_liveness(liv) -> str:
    A = liv["attacks"]
    rows = []
    label = {"flat": "Flat print (tilted up to 75$^\\circ$)", "bent_R60": "Bent print, $R{=}60$ mm", "bent_R40": "Bent print, $R{=}40$ mm",
             "bent_R30": "Bent print, $R{=}30$ mm"}
    for kind in ("flat", "bent_R60", "bent_R40", "bent_R30"):
        rows.append(f"{label[kind]} & {100 * A[f'{kind}/full']['pass_rate']:.1f} & {100 * A[f'{kind}/turns_only']['pass_rate']:.1f} & "
                    f"{100 * A[f'{kind}/turns_only_no_parallax']['pass_rate']:.1f} \\\\")
    return "\n".join(rows) + "\n"


def table_baselines(liv) -> str:
    """Liveness protocols under the same simulation: BPCER and APCER per attack (%), 95% Wilson interval in brackets."""
    B = liv["baselines"]
    names = [("yaw_turns", "Head turns, yaw ratio only"), ("blink_only", "Single blink prompt"),
             ("conventional", "Random actions, yaw ratio"), ("faceguard", "\\sys{} (full protocol)")]
    rows = []
    for key, label in names:
        cells = []
        for kind in ("live", "flat", "bent_R40", "replay", "universal"):
            r = B[f"{key}/{kind}"]
            rate = r["frr"] if "frr" in r else r["pass_rate"]
            cells.append(f"{100 * rate:.1f}")
        bold = key == "faceguard"
        cells = [f"\\textbf{{{c}}}" if bold else c for c in cells]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def recognition_baselines(base, lfw):
    """Rows and macros for same-protocol recognition baselines (LFW View 2)."""
    from scipy.stats import ttest_rel
    R = base["results"]
    lines, rows = [], []
    lat = base.get("latency_tta_ms", {})
    specs = [("sface", "SFace (\\sys{})", 38.7), ("arcface_mbf", "ArcFace MobileFaceNet", base["model_mb"]["arcface_mbf"]),
             ("arcface_r50", "ArcFace ResNet-50", base["model_mb"]["arcface_r50"])]
    for key, label, mb in specs:
        r = R[key]
        ms = f"{lat[key]['mean_ms']:.1f}" if key in lat else "--"
        bold = key == "sface"
        acc = f"{100 * r['accuracy']:.2f}$\\pm${100 * r['std']:.2f}"
        rows.append(f"{label} & {mb:.1f} & {ms} & {acc} & {r['auc']:.4f} & {100 * r['eer']:.2f} & "
                    f"{100 * r['tar_at_far_1e-2']:.2f} & {100 * r['tar_at_far_1e-3']:.2f} \\\\")
    def p(a, b):
        v = float(ttest_rel(R[a]["fold_acc"], R[b]["fold_acc"]).pvalue)
        return "<0.001" if v < 0.001 else f"{v:.2f}" if v >= 0.1 else f"{v:.3f}"
    macros = {"pTta": p("sface", "sface_no_tta"), "pSfaceMbf": p("sface", "arcface_mbf"), "pSfaceRfifty": p("sface", "arcface_r50")}
    for key, tag in (("arcface_mbf", "Mbf"), ("arcface_r50", "Rfifty")):
        macros[f"base{tag}Acc"] = f"{100 * R[key]['accuracy']:.2f}"
        macros[f"base{tag}TarThree"] = f"{100 * R[key]['tar_at_far_1e-3']:.2f}"
        macros[f"base{tag}Eer"] = f"{100 * R[key]['eer']:.2f}"
        macros[f"base{tag}Mb"] = f"{base['model_mb'][key]:.1f}"
        if key in lat:
            macros[f"base{tag}Ms"] = f"{lat[key]['mean_ms']:.1f}"
    if "sface" in lat:
        macros["baseSfaceMs"] = f"{lat['sface']['mean_ms']:.1f}"
    return "\n".join(rows) + "\n", macros


def mpii_tables(m):
    """Rows + macros for the laptop-webcam (MPIIFaceGaze) evaluation."""
    rows, mac = [], {}
    names = {"sface": "SFace (\\sys{})", "arcface_mbf": "ArcFace MobileFaceNet", "arcface_r50": "ArcFace ResNet-50"}
    for k, label in names.items():
        v, e = m[k]["verification"], m[k]["enrollment"]
        thr = (f"{100 * v['balanced/far']:.4f} & {100 * v['balanced/frr']:.2f} & {100 * e['balanced/tar']:.2f} & "
               f"{100 * e['balanced/far']:.2f}") if k == "sface" else "-- & -- & -- & --"  # thresholds are calibrated for SFace only
        rows.append(f"{label} & {v['auc']:.5f} & {100 * v['eer']:.2f} & {100 * v['tar_at_far_1e-3']:.2f} & {thr} \\\\")
    s, e = m["sface"]["verification"], m["sface"]["enrollment"]
    mac.update({"mpImages": f"{m['detected']:,}", "mpPeople": str(m["people"]), "mpPer": str(m["per_person"]),
                "mpFacePx": f"{m['face_px_median']:.0f}", "mpGen": f"{s['genuine_pairs']:,}", "mpImp": f"{s['impostor_pairs']:,}",
                "mpAuc": f"{s['auc']:.5f}", "mpEer": f"{100 * s['eer']:.2f}", "mpTarThree": f"{100 * s['tar_at_far_1e-3']:.2f}",
                "mpTarFour": f"{100 * s['tar_at_far_1e-4']:.2f}",
                "mpFarBal": f"{100 * s['balanced/far']:.4f}", "mpFaBal": f"{s['balanced/false_accepts']:,}",
                "mpFrrBal": f"{100 * s['balanced/frr']:.2f}", "mpFaStrict": f"{s['strict/false_accepts']:,}",
                "mpFrrStrict": f"{100 * s['strict/frr']:.2f}", "mpFarRelaxed": f"{100 * s['relaxed/far']:.3f}",
                "mpImpMax": f"{s['impostor_max']:.3f}", "mpGenMean": f"{s['genuine_mean']:.3f}",
                "mpProbes": f"{e['probes']:,}", "mpEnrollTar": f"{100 * e['balanced/tar']:.2f}",
                "mpEnrollFa": str(e["balanced/false_accepts"]), "mpEnrollFarRelaxed": f"{100 * e['relaxed/far']:.1f}",
                "mpEnrollImpMax": f"{e['impostor_best_max']:.3f}",
                "mpDaysMin": str(min(m["days_per_person"].values())), "mpDaysMax": str(max(m["days_per_person"].values()))})
    return "\n".join(rows) + "\n", mac


def demo_tables(d):
    rows, mac = [], {}
    order = [("sex", ("male", "female")), ("ethnicity", ("white", "black", "asian", "indian")), ("age", ("youth", "middle aged", "senior"))]
    label = {"middle aged": "Middle-aged"}
    for attr, groups in order:
        for g in groups:
            r = d["attributes"][attr].get(g)
            if not r:
                continue
            lo, hi = r["balanced/fmr_ci"]
            flo, fhi = r["balanced/fnmr_ci"]
            rows.append(f"{label.get(g, g.capitalize())} & {r['people']:,} & {r['genuine_pairs']:,} & {100 * r['balanced/fnmr']:.2f} "
                        f"({100 * flo:.1f}--{100 * fhi:.1f}) & {r['impostor_pairs']:,} & {100 * r['balanced/fmr']:.4f} "
                        f"({100 * lo:.4f}--{100 * hi:.4f}) \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    A = d["attributes"]
    tag = {"male": "Male", "female": "Female", "white": "White", "black": "Black", "asian": "Asian", "indian": "Indian",
           "youth": "Youth", "middle aged": "Middle", "senior": "Senior"}
    for attr, groups in order:
        for g in groups:
            r = A[attr].get(g)
            if r:
                mac[f"dm{tag[g]}Fmr"] = f"{100 * r['balanced/fmr']:.4f}"
                mac[f"dm{tag[g]}Fnmr"] = f"{100 * r['balanced/fnmr']:.2f}"
                mac[f"dm{tag[g]}People"] = f"{r['people']:,}"
                mac[f"dm{tag[g]}Gen"] = f"{r['genuine_pairs']:,}"
    mac["dmSexFmrRatio"] = f"{A['sex']['female']['balanced/fmr'] / A['sex']['male']['balanced/fmr']:.1f}"
    mac["dmSexFnmrRatio"] = f"{A['sex']['female']['balanced/fnmr'] / A['sex']['male']['balanced/fnmr']:.1f}"
    mac["dmEthFmrRatio"] = f"{A['ethnicity']['black']['balanced/fmr'] / A['ethnicity']['white']['balanced/fmr']:.0f}"
    mac["dmAsianWhiteFmrRatio"] = f"{A['ethnicity']['asian']['balanced/fmr'] / A['ethnicity']['white']['balanced/fmr']:.0f}"
    mac["dmLabeled"] = f"{d['labeled_people']:,}"
    return "\n".join(rows) + "\n", mac


def pilot_tables(p):
    ev, v1 = p["evaluation_v2"], p["pilot_v1"]
    pc = lambda x: f"{100 * x:.0f}"  # noqa: E731
    ci = lambda c: f"{100 * c[0]:.0f}--{100 * c[1]:.0f}"  # noqa: E731
    rows = []

    def unlock(key, label, attack=False):
        r = ev.get(key)
        if not r:
            return
        ok = r["verified"]
        rate = ok / r["n"]
        what = "accepted" if attack else "passed"
        rows.append(f"{label} & {r['n']} & {ok} {what} ({pc(rate)}\\%; {ci(r['ci'])}) \\\\")
    unlock("owner_live", "Owner, full protocol")
    unlock("owner_turns", "Owner, turns only")
    vol = ev.get("volunteer_live")
    if vol:
        rows.append(f"Volunteer, full protocol (impostor) & {vol['n']} & {vol['false_accepts']} accepted; live in "
                    f"{round(vol['live_rate'] * vol['n'])} of {vol['n']} \\\\")
    rows.append("\\midrule")
    unlock("phone_full", "Phone photo vs.\\ full protocol", True)
    unlock("phone_turns", "Phone photo vs.\\ turns, parallax gate", True)
    unlock("phone_turns_noparallax", "Phone photo vs.\\ turns, no gate", True)
    rows.append("\\midrule")
    g = ev.get("guard_desk")
    if g:
        rows.append(f"Owner at desk ({g['hours'] * 60:.0f} s $\\times60$) & 1 & {g['intruder_locks'] + g['absence_locks']} false locks, "
                    f"{g['shoulder_alerts']} alerts \\\\".replace(f"({g['hours'] * 60:.0f} s $\\times60$)", f"({g['hours'] * 3600:.0f}~s)"))
    for key, label in (("guard_walkaway", "Walk-away lock"), ("guard_intruder", "Stranger sits down: lock"),
                       ("guard_shoulder", "Shoulder-surfer: alert"), ("guard_lockwatch", "Locked PC: photograph")):
        r = ev.get(key)
        if r:
            rows.append(f"{label} & {r['trials']} & {r['detected']} of {r['trials']}, {r['since_event_median']:.1f}~s after the cue \\\\")
    o = v1["owner_live"]
    mac = {"rwPilotTrials": str(v1["trials"]), "rwPilotOwner": str(o["n"]), "rwPilotOwnerPass": str(o["verified"]),
           "rwPilotDiff": str(o["reasons"].get("A different face appeared", 0)),
           "rwPilotBlink": str(o["reasons"].get("Timed out waiting for: blink your eyes", 0)),
           "rwPilotNoResp": str(o["reasons"].get("No response to the prompt in time", 0))}
    ol, ot = ev["owner_live"], ev["owner_turns"]
    att = [ev[k] for k in ("phone_full", "phone_turns", "phone_turns_noparallax") if k in ev]
    n_att = sum(a["n"] for a in att)
    mac.update({"rwOwnerN": str(ol["n"] + ot["n"]), "rwOwnerPass": str(ol["verified"] + ot["verified"]),
                "rwFps": f"{ol['fps_median']:.0f}", "rwOwnerSim": f"{ol['similarity_median']:.2f}",
                "rwAttN": str(n_att), "rwAttAcc": str(sum(a["verified"] for a in att)),
                "rwAttYawMax": f"{max(a['max_yaw_change']['max'] for a in att if a['max_yaw_change']['max'] is not None):.1f}",
                "rwAttParMax": f"{max(a['max_parallax']['max'] for a in att if a['max_parallax']['max'] is not None):.1f}",
                "rwOwnerParMed": f"{ol['max_parallax']['median']:.0f}",
                "rwVolN": str(vol["n"]) if vol else "0", "rwVolFa": str(vol["false_accepts"]) if vol else "0",
                "rwVolSim": f"{vol['max_similarity_to_owner']:.2f}" if vol else "--",
                "rwVolLive": str(round(vol["live_rate"] * vol["n"])) if vol else "0",
                "rwAttUpper": f"{100 * wilson(0, n_att)[1]:.0f}"})
    for key, tag in (("guard_walkaway", "Walk"), ("guard_intruder", "Intr"), ("guard_shoulder", "Shld"), ("guard_lockwatch", "Lockw")):
        if key in ev:
            mac[f"rw{tag}Lat"] = f"{ev[key]['latency_median']:.1f}"
            mac[f"rw{tag}Beep"] = f"{ev[key]['since_event_median']:.1f}"
    if "guard_desk" in ev:
        mac["rwDeskS"] = f"{ev['guard_desk']['hours'] * 3600:.0f}"
    return "\n".join(rows) + "\n", mac


def main():
    res = {n: load(n) for n in ("lfw", "olivetti", "robustness", "parallax", "pose", "liveness", "guard", "guard_diag", "runtime",
                                "baselines", "mpii", "demographics", "realworld")}
    newmac = {}
    for key, fn, fname, macro in (("mpii", mpii_tables, "tab_mpii.tex", "MpiiRows"), ("demographics", demo_tables, "tab_demo.tex", "DemoRows"),
                                  ("realworld", pilot_tables, "tab_pilot.tex", "PilotRows")):
        if res[key]:
            rows, mac = fn(res[key])
            (GEN / fname).write_text(f"\\newcommand{{\\{macro}}}{{%\n" + rows + "}\n", encoding="utf-8")
            newmac.update(mac)
    if res["liveness"] and "baselines" in res["liveness"]:
        (GEN / "tab_baselines.tex").write_text("\\newcommand{\\BaselineRows}{%\n" + table_baselines(res["liveness"]) + "}\n",
                                               encoding="utf-8")
    extra = {}
    if res["baselines"]:
        rows, extra = recognition_baselines(res["baselines"], res["lfw"])
        (GEN / "tab_recog.tex").write_text("\\newcommand{\\RecogRows}{%\n" + rows + "}\n", encoding="utf-8")
    # Table rows are wrapped in macros: \input inside a tabular breaks booktabs rules ("Misplaced \noalign").
    if res["robustness"]:
        (GEN / "tab_robust.tex").write_text("\\newcommand{\\RobustRows}{%\n" + table_robust(res["robustness"]) + "}\n",
                                            encoding="utf-8")
    if res["liveness"] and "flat/turns_only_no_parallax" in res["liveness"].get("attacks", {}):
        (GEN / "tab_liveness.tex").write_text("\\newcommand{\\PrintRows}{%\n" + table_liveness(res["liveness"]) + "}\n",
                                              encoding="utf-8")
    if res["parallax"]:
        fig_parallax(res["parallax"])
    if res["lfw"]:
        fig_roc(res["lfw"])
        fig_scores(res["lfw"])
        fig_openset(res["lfw"])
    if res["robustness"]:
        fig_robust(res["robustness"])
    if res["liveness"] and "tau_sweep" in res["liveness"]:
        fig_liveness(res["liveness"])
    if res["runtime"]:
        fig_runtime(res["runtime"])
    extra.update(newmac)
    text = macros(res) + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in extra.items())
    (GEN / "numbers.tex").write_text(text, encoding="utf-8")
    print(f"[tex] {GEN / 'numbers.tex'}")


if __name__ == "__main__":
    main()
