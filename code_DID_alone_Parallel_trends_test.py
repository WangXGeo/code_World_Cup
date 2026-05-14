# -*- coding: utf-8 -*-
"""
World Cup DID (TWFE) + Parallel Trends (treated = type1+type2 vs type0)
======================================================================
Main goals:
1) Multi-scale DID effects on GSR (bub/500/1000/5000/10000)
2) Build vs Post sustainability (Build=Post Wald, retention ratio Post/Build)
3) Main host (type2) vs Co-host (type1) stronger greening? (Wald within phase)
4) Parallel trends using pre-award 5 years:
   - Linear pretrend test: treated*t slope in pre window
   - Event-study (treated vs control): joint lead test + figure

Panel:
- entity: id (ENTITY_COL)
- time  : year (TIME_COL)
- type  : 0 control, 1 co-host, 2 main host

Phases:
- Build: [AWARD_YEAR, HOST_YEAR-1]
- Post : [HOST_YEAR+1, HOST_YEAR+POST_HORIZON]   (default horizon=5)
- Event year excluded by design (optional)

Dependencies:
pip install pandas numpy matplotlib linearmodels
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS


# =======================
# USER CONFIG
# =======================
CSV_PATH = r"year_buffers.csv"
OUT_DIR  = r"2010_South_Africa"

# Key years
AWARD_YEAR = 2004
HOST_YEAR  = 2010
POST_HORIZON = 5            # post-period upper bound = HOST_YEAR + POST_HORIZON
PRE_YEARS = 5               # number of pre-award years used for parallel-trends tests
EXCLUDE_EVENT_YEAR = True   # whether to exclude the host/event year

# Columns
ENTITY_COL = "id"
TIME_COL   = "year"
TYPE_COL   = "type"

# Outcomes (buffers)
Y_COLS = ["GSR_bub", "GSR_500", "GSR_1000", "GSR_5000", "GSR_10000"]

# Controls
CONTROLS = ["NTL", "MAT", "AP"]

# Parallel trends switches
DO_PRETREND_LINEAR = True
DO_EVENT_STUDY     = True
EVENT_MAX_LAG      = POST_HORIZON   # maximum event-study lag
EVENT_BASE_REL     = -1             # omitted event-study baseline period
# =======================


# =======================
# Helpers
# =======================
def ensure_columns(df: pd.DataFrame, cols: list) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")


def stars(p: float) -> str:
    if not np.isfinite(p):
        return "na"
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return "ns"


def stars_table(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def sig_symbol(p: float) -> str:
    # bracket symbol on plot: **(1%), *(5%), †(10%)
    if not np.isfinite(p):
        return ""
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p < 0.10: return "†"
    return ""


def get_term(res, term: str):
    b  = float(res.params.get(term, np.nan))
    se = float(res.std_errors.get(term, np.nan))
    p  = float(res.pvalues.get(term, np.nan))
    lo = b - 1.96 * se
    hi = b + 1.96 * se
    return b, se, p, lo, hi


def wald_linear(res, weights: dict):
    """
    Wald test for linear restriction: sum_k weights[k] * beta_k = 0
    weights: {term_name: weight}
    """
    names = list(res.params.index)
    for t in weights:
        if t not in names:
            return np.nan
    w = np.zeros((1, len(names)))
    for t, wt in weights.items():
        w[0, names.index(t)] = float(wt)
    try:
        test = res.wald_test(w)
        return float(test.pval)
    except Exception:
        return np.nan


def wald_joint_zero(res, terms: list):
    """
    Joint Wald test H0: all terms == 0.
    If covariance is singular, return NaN instead of crashing.
    """
    names = list(res.params.index)
    valid = [t for t in terms if t in names]
    if len(valid) == 0:
        return np.nan
    W = np.zeros((len(valid), len(names)))
    for i, t in enumerate(valid):
        W[i, names.index(t)] = 1.0
    try:
        test = res.wald_test(W)
        return float(test.pval)
    except Exception:
        return np.nan


def fit_twfe(panel: pd.DataFrame, y: str, xcols: list):
    model = PanelOLS(
        panel[y].astype(float),
        panel[xcols].astype(float),
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False
    )
    res = model.fit(cov_type="clustered", cluster_entity=True)
    return res


def drop_constant_or_allzero_cols(panel: pd.DataFrame, cols: list):
    """Drop columns with no variation (all zeros / constant) in the current panel."""
    keep = []
    for c in cols:
        s2 = panel[c].dropna()
        if s2.empty:
            continue
        if s2.nunique() <= 1:
            continue
        keep.append(c)
    return keep


def make_full_rank_cols(panel: pd.DataFrame, cols: list, tol: float = 1e-10):
    """
    Greedy selection of linearly independent columns to avoid singular X'X.
    Suitable for event-study (few dozen columns).
    """
    if not cols:
        return []

    X = panel[cols].to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if X.shape[0] == 0:
        return []

    selected = []
    current = None
    current_rank = 0

    for j, c in enumerate(cols):
        xj = X[:, [j]]
        if current is None:
            if np.linalg.norm(xj) > tol:
                selected.append(c)
                current = xj
                current_rank = np.linalg.matrix_rank(current, tol=tol)
            continue

        trial = np.hstack([current, xj])
        r = np.linalg.matrix_rank(trial, tol=tol)
        if r > current_rank:
            selected.append(c)
            current = trial
            current_rank = r

    return selected


def add_bracket(ax, x1, x2, y, text, dh=0.08, barh=0.06, fs=8):
    y0 = y + dh
    ax.plot([x1, x1, x2, x2], [y0, y0 + barh, y0 + barh, y0],
            linewidth=0.8, alpha=0.9)
    ax.text((x1 + x2) / 2, y0 + barh + dh * 0.35, text,
            ha="center", va="bottom", fontsize=fs)


def parse_scale_label(y_col: str) -> str:
    if y_col.endswith("_bub"):
        return "bub"
    for s in ["500", "1000", "5000", "10000"]:
        if y_col.endswith(f"_{s}"):
            return f"{s}m"
    return y_col


def scale_sort_key(scale_label: str) -> int:
    order = {"bub": 0, "500m": 1, "1000m": 2, "5000m": 3, "10000m": 4}
    return order.get(scale_label, 999)


# =======================
# Parallel trends (treated = type1+type2)
# =======================
def pretrend_linear_test_treated(df: pd.DataFrame, y_col: str):
    """
    Pre-award window: [AWARD_YEAR-PRE_YEARS, AWARD_YEAR-1]
    Treated = 1 if type in {1,2}, Control = type==0
    Model: y ~ treated*t + controls + FE
    Test: p-value of treated slope
    """
    y0 = AWARD_YEAR - PRE_YEARS
    y1 = AWARD_YEAR - 1
    sub = df[(df[TIME_COL] >= y0) & (df[TIME_COL] <= y1)].copy()
    if sub.empty:
        return {"pretrend_window": f"{y0}-{y1}", "p_treated_slope": np.nan}

    sub["t_pre"] = sub[TIME_COL] - y0

    sub[TYPE_COL] = pd.to_numeric(sub[TYPE_COL], errors="coerce")
    sub = sub.dropna(subset=[TYPE_COL])
    sub[TYPE_COL] = sub[TYPE_COL].astype(int)

    sub["treated"] = (sub[TYPE_COL].isin([1, 2])).astype(int)
    sub["T_slope"] = sub["treated"] * sub["t_pre"]

    panel = sub.sort_values([ENTITY_COL, TIME_COL]).set_index([ENTITY_COL, TIME_COL])
    xcols = ["T_slope"] + CONTROLS
    panel = panel.dropna(subset=[y_col] + xcols)
    if panel.empty:
        return {"pretrend_window": f"{y0}-{y1}", "p_treated_slope": np.nan}

    res = fit_twfe(panel, y_col, xcols)
    p = float(res.pvalues.get("T_slope", np.nan))
    return {"pretrend_window": f"{y0}-{y1}", "p_treated_slope": p}


def event_study_treated(df: pd.DataFrame, y_col: str, out_dir: Path):
    """
    Event study centered at AWARD_YEAR:
    treated = 1 if type in {1,2}, control = type==0
    rel = year - AWARD_YEAR
    Include leads [-PRE_YEARS..-2] and lags [0..EVENT_MAX_LAG], baseline rel=EVENT_BASE_REL (-1).
    Joint test on all leads (parallel trends).
    """
    sub = df.copy()
    sub["rel"] = sub[TIME_COL] - AWARD_YEAR

    rel_lo = -PRE_YEARS
    rel_hi = EVENT_MAX_LAG
    sub = sub[(sub["rel"] >= rel_lo) & (sub["rel"] <= rel_hi)].copy()

    if EXCLUDE_EVENT_YEAR:
        sub = sub[sub[TIME_COL] != HOST_YEAR].copy()

    sub[TYPE_COL] = pd.to_numeric(sub[TYPE_COL], errors="coerce")
    sub = sub.dropna(subset=[TYPE_COL])
    sub[TYPE_COL] = sub[TYPE_COL].astype(int)

    sub["treated"] = (sub[TYPE_COL].isin([1, 2])).astype(int)

    rel_vals = list(range(rel_lo, rel_hi + 1))
    if EVENT_BASE_REL in rel_vals:
        rel_vals.remove(EVENT_BASE_REL)

    xcols = []
    term_map = []  # (rel, term)
    for r in rel_vals:
        col_r = f"R{r:+d}"
        sub[col_r] = (sub["rel"] == r).astype(int)
        term = f"T_{col_r}"                 # treated × rel dummy
        sub[term] = sub["treated"] * sub[col_r]
        xcols.append(term)
        term_map.append((r, term))

    xcols += CONTROLS

    panel = sub.sort_values([ENTITY_COL, TIME_COL]).set_index([ENTITY_COL, TIME_COL])
    panel = panel.dropna(subset=[y_col] + xcols)
    if panel.empty:
        return None

    # Avoid singular matrices: remove constant columns and keep a full-rank set.
    xcols_clean = drop_constant_or_allzero_cols(panel, xcols)
    xcols_clean = make_full_rank_cols(panel, xcols_clean)

    # Keep control variables where possible and place them after event-study terms.
    ctrls = [c for c in CONTROLS if c in xcols_clean]
    others = [c for c in xcols_clean if c not in CONTROLS]
    xcols_clean = others + ctrls

    res = fit_twfe(panel, y_col, xcols_clean)

    # joint lead test: rel in [-PRE_YEARS..-2]
    lead_terms = [t for r, t in term_map if (r <= -2 and t in res.params.index)]
    p_leads = wald_joint_zero(res, lead_terms) if lead_terms else np.nan

    # build plot df
    rows = []
    for r, term in term_map:
        if term not in res.params.index:
            continue
        b, se, p, lo, hi = get_term(res, term)
        rows.append({"rel": r, "b_pp": b * 100, "lo_pp": lo * 100, "hi_pp": hi * 100, "p": p})
    est = pd.DataFrame(rows).sort_values("rel")

    # even if est empty, still return p-value
    if est.empty:
        return {"p_leads_treated": p_leads}

    # plot
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(4.2, 2.35))
    ax.grid(axis="y", linewidth=0.4, alpha=0.22)
    ax.set_axisbelow(True)

    C = "#4C78A8"
    x = est["rel"].to_numpy()
    y = est["b_pp"].to_numpy()
    lo = est["lo_pp"].to_numpy()
    hi = est["hi_pp"].to_numpy()
    yerr = np.vstack([y - lo, hi - y])

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o", markersize=3.8,
        markerfacecolor=C, markeredgecolor=C,
        ecolor=C, color=C,
        capsize=2.0, elinewidth=0.9, capthick=0.9,
        linewidth=0,
        label="treated (type1+type2)"
    )
    ax.plot(x, y, linewidth=0.9, alpha=0.55, color=C)

    ax.axvline(0, linewidth=0.8, alpha=0.6)
    ax.axhline(0, linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Event time (year - award_year)")
    ax.set_ylabel("Effect on GSR (pp)")
    ax.set_title(f"Event-study (treated vs control): {y_col}", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    ax.text(0.01, 0.02, f"Lead joint test p={p_leads:.3f}",
            transform=ax.transAxes, fontsize=8, va="bottom")

    fig.tight_layout(pad=0.6)
    safe_y = y_col.replace("/", "_")
    fig_png = out_dir / f"{HOST_YEAR}_EventStudy_Treated_{safe_y}.png"
    fig_pdf = out_dir / f"{HOST_YEAR}_EventStudy_Treated_{safe_y}.pdf"
    fig.savefig(fig_png, dpi=600)
    fig.savefig(fig_pdf)
    plt.close(fig)

    return {
        "p_leads_treated": p_leads,
        "eventstudy_png": str(fig_png),
        "eventstudy_pdf": str(fig_pdf),
    }


# =======================
# Two-phase DID (type1 vs type0, type2 vs type0)
# =======================
def run_two_phase(df_raw: pd.DataFrame, y_col: str, out_dir: Path) -> dict:
    base_cols = [ENTITY_COL, TIME_COL, TYPE_COL, y_col] + CONTROLS
    ensure_columns(df_raw, base_cols)

    df = df_raw.copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").astype("Int64")
    df = df.dropna(subset=[TIME_COL])
    df[TIME_COL] = df[TIME_COL].astype(int)

    df[TYPE_COL] = pd.to_numeric(df[TYPE_COL], errors="coerce").astype("Int64")
    df = df.dropna(subset=[TYPE_COL])
    df[TYPE_COL] = df[TYPE_COL].astype(int)

    # outcome
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce").clip(0, 1)

    # phases
    build_lo = AWARD_YEAR
    build_hi = HOST_YEAR - 1
    post_lo  = HOST_YEAR + 1
    post_hi  = HOST_YEAR + POST_HORIZON

    df["build"] = ((df[TIME_COL] >= build_lo) & (df[TIME_COL] <= build_hi)).astype(int)
    df["post"]  = ((df[TIME_COL] >= post_lo) & (df[TIME_COL] <= post_hi)).astype(int)

    if EXCLUDE_EVENT_YEAR:
        df = df[df[TIME_COL] != HOST_YEAR].copy()

    # type dummies
    df["type1"] = (df[TYPE_COL] == 1).astype(int)
    df["type2"] = (df[TYPE_COL] == 2).astype(int)
    has_type2 = df["type2"].sum() > 0

    # interactions
    df["T1_build"] = df["type1"] * df["build"]
    df["T1_post"]  = df["type1"] * df["post"]
    if has_type2:
        df["T2_build"] = df["type2"] * df["build"]
        df["T2_post"]  = df["type2"] * df["post"]

    df = df.sort_values([ENTITY_COL, TIME_COL])
    panel = df.set_index([ENTITY_COL, TIME_COL])

    xcols = ["T1_build", "T1_post"] + (["T2_build", "T2_post"] if has_type2 else []) + CONTROLS
    panel = panel.dropna(subset=[y_col] + xcols)
    if panel.empty:
        raise ValueError(f"No valid rows for outcome={y_col}. Check missing values.")

    res = fit_twfe(panel, y_col, xcols)

    # terms
    b1, se1, p1, lo1, hi1 = get_term(res, "T1_build")
    b2, se2, p2, lo2, hi2 = get_term(res, "T1_post")
    if has_type2:
        b3, se3, p3, lo3, hi3 = get_term(res, "T2_build")
        b4, se4, p4, lo4, hi4 = get_term(res, "T2_post")
    else:
        b3 = se3 = p3 = lo3 = hi3 = np.nan
        b4 = se4 = p4 = lo4 = hi4 = np.nan

    # tests
    # within group sustainability: Build = Post
    p_t1_phase = wald_linear(res, {"T1_build": 1, "T1_post": -1})
    p_t2_phase = wald_linear(res, {"T2_build": 1, "T2_post": -1}) if has_type2 else np.nan

    # main host vs co-host within phase
    p_build_t2_vs_t1 = wald_linear(res, {"T2_build": 1, "T1_build": -1}) if has_type2 else np.nan
    p_post_t2_vs_t1  = wald_linear(res, {"T2_post": 1, "T1_post": -1}) if has_type2 else np.nan

    # retention ratio Post/Build
    eps = 1e-10
    ret_t1 = (b2 / b1) if np.isfinite(b1) and abs(b1) > eps else np.nan
    ret_t2 = (b4 / b3) if np.isfinite(b3) and abs(b3) > eps else np.nan

    # console summary
    print(f"\n=== Outcome: {y_col} ===")
    print(f"Build: {build_lo}-{build_hi} | Post: {post_lo}-{post_hi} | EventYearExcluded={EXCLUDE_EVENT_YEAR}")
    print("[Wald: H0 Build = Post (within group)]")
    print(f" type1: p={p_t1_phase:.4f} -> {stars(p_t1_phase)}")
    if has_type2:
        print(f" type2: p={p_t2_phase:.4f} -> {stars(p_t2_phase)}")
    if has_type2:
        print("[Wald: H0 type2 = type1 (within phase)]")
        print(f" Build: p={p_build_t2_vs_t1:.4f} -> {stars(p_build_t2_vs_t1)}")
        print(f" Post : p={p_post_t2_vs_t1:.4f}  -> {stars(p_post_t2_vs_t1)}")
    print("[Retention ratio = Post/Build]")
    print(f" type1: {ret_t1 if np.isfinite(ret_t1) else np.nan}")
    if has_type2:
        print(f" type2: {ret_t2 if np.isfinite(ret_t2) else np.nan}")

    # output tables
    meta = {
        "award_year": AWARD_YEAR,
        "host_year": HOST_YEAR,
        "post_end": HOST_YEAR + POST_HORIZON,
        "Outcome": y_col,
        "N": int(res.nobs),
        "Entities": int(panel.index.get_level_values(0).nunique()),
        "Years": int(panel.index.get_level_values(1).nunique()),
        "Entity FE": "Yes",
        "Year FE": "Yes",
        "Cluster": "Entity",
        "Controls": ", ".join(CONTROLS),

        "Wald_T1_BuildEqPost_p": p_t1_phase,
        "Wald_T2_BuildEqPost_p": p_t2_phase,
        "Wald_Build_T2EqT1_p": p_build_t2_vs_t1,
        "Wald_Post_T2EqT1_p": p_post_t2_vs_t1,
        "Retention_T1_PostOverBuild": ret_t1,
        "Retention_T2_PostOverBuild": ret_t2,
    }

    terms = [
        ("Co-host (type1)", "Build", "T1_build", b1, se1, p1, lo1, hi1),
        ("Co-host (type1)", "Post",  "T1_post",  b2, se2, p2, lo2, hi2),
    ]
    if has_type2:
        terms += [
            ("Main host (type2)", "Build", "T2_build", b3, se3, p3, lo3, hi3),
            ("Main host (type2)", "Post",  "T2_post",  b4, se4, p4, lo4, hi4),
        ]

    rows = []
    for group, phase, term, b, se, p, lo, hi in terms:
        rows.append({
            **meta,
            "Group": group,
            "Phase": phase,
            "term": term,
            "Effect(pp)": b * 100,
            "SE(pp)": se * 100,
            "CI95_low(pp)": lo * 100,
            "CI95_high(pp)": hi * 100,
            "p": p,
            "Paper cell": f"{b*100:.3f}{stars_table(p)} ({se*100:.3f})",
        })

    out_long = pd.DataFrame(rows)
    out_wide = out_long.pivot(index=["Outcome", "Group"], columns="Phase", values="Paper cell").reset_index()

    safe_y = y_col.replace("/", "_")
    long_csv = out_dir / f"{HOST_YEAR}_TwoPhase_LONG_{safe_y}.csv"
    wide_csv = out_dir / f"{HOST_YEAR}_TwoPhase_WIDE_{safe_y}.csv"
    stat_csv = out_dir / f"{HOST_YEAR}_TwoPhase_STATS_{safe_y}.csv"
    out_long.to_csv(long_csv, index=False, encoding="utf-8-sig")
    out_wide.to_csv(wide_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([meta]).to_csv(stat_csv, index=False, encoding="utf-8-sig")

    # plot (Build vs Post)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })

    phase_order = ["Build", "Post"]
    plot_df = out_long.copy()
    plot_df["Phase"] = pd.Categorical(plot_df["Phase"], categories=phase_order, ordered=True)
    plot_df = plot_df.sort_values(["Group", "Phase"])

    build_label = f"Build ({build_lo}–{build_hi})"
    post_label  = f"Post ({post_lo}–{post_hi})"
    x = np.arange(2)

    C1 = "#B07AA1"
    C2 = "#7DBA7A"
    color_map = {"Co-host (type1)": C1, "Main host (type2)": C2}
    grp_order = ["Co-host (type1)"] + (["Main host (type2)"] if has_type2 else [])
    offsets = [-0.07, 0.07] if has_type2 else [0.0]

    fig, ax = plt.subplots(figsize=(3.40, 2.25))
    ax.grid(axis="y", linewidth=0.4, alpha=0.22)
    ax.set_axisbelow(True)

    plotted = {}
    for gi, g in enumerate(grp_order):
        subg = plot_df[plot_df["Group"] == g]
        yv = subg["Effect(pp)"].to_numpy()
        lo = subg["CI95_low(pp)"].to_numpy()
        hi = subg["CI95_high(pp)"].to_numpy()
        yerr = np.vstack([yv - lo, hi - yv])

        xs = x + offsets[gi]
        c = color_map[g]

        ax.errorbar(xs, yv, yerr=yerr, fmt="o", markersize=4.2,
                    markerfacecolor=c, markeredgecolor=c,
                    ecolor=c, color=c,
                    capsize=2.0, elinewidth=0.9, capthick=0.9,
                    linewidth=0, label=g)
        ax.plot(xs, yv, linewidth=0.9, alpha=0.55, color=c)
        plotted[g] = {"x": xs, "hi": hi}

    ax.set_xticks(x)
    ax.set_xticklabels([build_label, post_label])
    ax.set_ylabel("Effect on GSR (pp)")
    ax.set_title(y_col, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")

    # bracket for phase difference if significant
    lab1 = sig_symbol(p_t1_phase)
    if lab1 and "Co-host (type1)" in plotted:
        xs = plotted["Co-host (type1)"]["x"]
        y_br = float(np.nanmax(plotted["Co-host (type1)"]["hi"]))
        add_bracket(ax, xs[0], xs[1], y_br, lab1, dh=0.10, barh=0.08, fs=8)

    if has_type2:
        lab2 = sig_symbol(p_t2_phase)
        if lab2 and "Main host (type2)" in plotted:
            xs = plotted["Main host (type2)"]["x"]
            y_br = float(np.nanmax(plotted["Main host (type2)"]["hi"]))
            add_bracket(ax, xs[0], xs[1], y_br, lab2, dh=0.10, barh=0.08, fs=8)

    fig.tight_layout(pad=0.6)
    fig_png = out_dir / f"{HOST_YEAR}_TwoPhase_Effects_{safe_y}.png"
    fig_pdf = out_dir / f"{HOST_YEAR}_TwoPhase_Effects_{safe_y}.pdf"
    fig.savefig(fig_png, dpi=600)
    fig.savefig(fig_pdf)
    plt.close(fig)

    return {
        "Outcome": y_col,
        "Scale": parse_scale_label(y_col),

        "T1_Build_pp": b1 * 100,
        "T1_Post_pp":  b2 * 100,
        "T1_Build_p":  p1,
        "T1_Post_p":   p2,
        "Wald_T1_BuildEqPost_p": p_t1_phase,
        "Retention_T1_PostOverBuild": ret_t1,

        "T2_Build_pp": b3 * 100,
        "T2_Post_pp":  b4 * 100,
        "T2_Build_p":  p3,
        "T2_Post_p":   p4,
        "Wald_T2_BuildEqPost_p": p_t2_phase,
        "Retention_T2_PostOverBuild": ret_t2,

        "Wald_Build_T2EqT1_p": p_build_t2_vs_t1,
        "Wald_Post_T2EqT1_p":  p_post_t2_vs_t1,

        "TwoPhase_LONG": str(long_csv),
        "TwoPhase_WIDE": str(wide_csv),
        "TwoPhase_STATS": str(stat_csv),
        "TwoPhase_FIG_PNG": str(fig_png),
        "TwoPhase_FIG_PDF": str(fig_pdf),
    }


def plot_scale_heterogeneity(summary_df: pd.DataFrame, out_dir: Path):
    if summary_df.empty:
        return
    df = summary_df.copy()
    df["ScaleOrder"] = df["Scale"].apply(scale_sort_key)
    df = df.sort_values("ScaleOrder")

    x = np.arange(len(df))
    labels = df["Scale"].tolist()

    C1 = "#B07AA1"
    C2 = "#7DBA7A"

    # Build
    fig, ax = plt.subplots(figsize=(4.3, 2.35))
    ax.grid(axis="y", linewidth=0.4, alpha=0.22)
    ax.set_axisbelow(True)
    ax.plot(x, df["T1_Build_pp"], marker="o", linewidth=0.9, alpha=0.75, color=C1, label="type1 Build")
    if df["T2_Build_pp"].notna().any():
        ax.plot(x, df["T2_Build_pp"], marker="o", linewidth=0.9, alpha=0.75, color=C2, label="type2 Build")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Effect (pp)")
    ax.set_title("Scale heterogeneity: Build", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.6)
    fig.savefig(out_dir / f"{HOST_YEAR}_ScaleHeterogeneity_Build.png", dpi=600)
    fig.savefig(out_dir / f"{HOST_YEAR}_ScaleHeterogeneity_Build.pdf")
    plt.close(fig)

    # Post
    fig, ax = plt.subplots(figsize=(4.3, 2.35))
    ax.grid(axis="y", linewidth=0.4, alpha=0.22)
    ax.set_axisbelow(True)
    ax.plot(x, df["T1_Post_pp"], marker="o", linewidth=0.9, alpha=0.75, color=C1, label="type1 Post")
    if df["T2_Post_pp"].notna().any():
        ax.plot(x, df["T2_Post_pp"], marker="o", linewidth=0.9, alpha=0.75, color=C2, label="type2 Post")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Effect (pp)")
    ax.set_title("Scale heterogeneity: Post", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.6)
    fig.savefig(out_dir / f"{HOST_YEAR}_ScaleHeterogeneity_Post.png", dpi=600)
    fig.savefig(out_dir / f"{HOST_YEAR}_ScaleHeterogeneity_Post.pdf")
    plt.close(fig)


# =======================
# Main
# =======================
def main():
    in_path = Path(CSV_PATH)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # sanity check core columns
    ensure_columns(df, [ENTITY_COL, TIME_COL, TYPE_COL] + CONTROLS)

    all_rows = []
    pretrend_rows = []
    event_rows = []

    for y in Y_COLS:
        ensure_columns(df, [y])  # outcome exists?

        # 1) Two-phase DID (type1 & type2 separate)
        summ = run_two_phase(df, y, out_dir)
        all_rows.append(summ)

        # 2) Pretrend linear test (treated=type1+type2)
        if DO_PRETREND_LINEAR:
            pt = pretrend_linear_test_treated(df, y)
            pretrend_rows.append({"Outcome": y, "Scale": parse_scale_label(y), **pt})

        # 3) Event-study (treated=type1+type2)
        if DO_EVENT_STUDY:
            es = event_study_treated(df, y, out_dir)
            if es is not None:
                event_rows.append({"Outcome": y, "Scale": parse_scale_label(y), **es})

    # combined summary
    summary_df = pd.DataFrame(all_rows)
    summary_df["ScaleOrder"] = summary_df["Scale"].apply(scale_sort_key)
    summary_df = summary_df.sort_values("ScaleOrder").drop(columns=["ScaleOrder"])
    summary_csv = out_dir / f"{HOST_YEAR}_TwoPhase_Summary_ALL.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # scale heterogeneity plots
    plot_scale_heterogeneity(summary_df, out_dir)

    # pretrend outputs
    if pretrend_rows:
        pre_df = pd.DataFrame(pretrend_rows)
        pre_df["ScaleOrder"] = pre_df["Scale"].apply(scale_sort_key)
        pre_df = pre_df.sort_values("ScaleOrder").drop(columns=["ScaleOrder"])
        pre_csv = out_dir / f"{HOST_YEAR}_Pretrend_Linear_TreatedVsControl_ALL.csv"
        pre_df.to_csv(pre_csv, index=False, encoding="utf-8-sig")
        print("Saved pretrend linear summary:", pre_csv)

    # event-study outputs
    if event_rows:
        es_df = pd.DataFrame(event_rows)
        es_df["ScaleOrder"] = es_df["Scale"].apply(scale_sort_key)
        es_df = es_df.sort_values("ScaleOrder").drop(columns=["ScaleOrder"])
        es_csv = out_dir / f"{HOST_YEAR}_EventStudy_TreatedVsControl_Summary_ALL.csv"
        es_df.to_csv(es_csv, index=False, encoding="utf-8-sig")
        print("Saved event-study summary:", es_csv)

    print("\n=== ALL DONE ===")
    print("Two-phase summary:", summary_csv)
    print("Scale plots saved:")
    print(" -", out_dir / f"{HOST_YEAR}_ScaleHeterogeneity_Build.png")
    print(" -", out_dir / f"{HOST_YEAR}_ScaleHeterogeneity_Post.png")


if __name__ == "__main__":
    main()