"""
Many-Facet Rasch Model (MFRM) — 2-Facet Binomial Rating Scale Model
====================================================================
Facets: Essay Ability (B_n) × Rater Severity (D_i)

Model (logit scale):
    logit E(x_ni / m) = B_n - D_i

where:
  x_ni  = score rater i gives essay n  (0 ≤ x_ni ≤ m)
  m     = maximum possible score
  B_n   = essay ability (higher = better quality)
  D_i   = rater severity (positive = harsh, negative = lenient)
           Convention: mean(D_i) = 0  (rater facet defines scale origin)

Estimation: Joint Maximum Likelihood (JMLE) via Newton-Raphson iteration.
Fit stats:  Infit MSQ and Outfit MSQ (same interpretation as FACETS / TOEFL).

References
----------
Linacre, J.M. (1994). Many-facet Rasch measurement. MESA Press.
Bond, T.G. & Fox, C.M. (2015). Applying the Rasch Model (3rd ed.). Routledge.
Eckes, T. (2015). Introduction to Many-Facet Rasch Measurement. Peter Lang.
"""

from __future__ import annotations
import numpy as np


# ── Numerically stable sigmoid ────────────────────────────────────────────────

def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (avoids overflow)."""
    x = np.clip(x, -500.0, 500.0)
    pos = x >= 0
    result = np.empty_like(x, dtype=float)
    result[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    result[~pos] = ex / (1.0 + ex)
    return result


# ── Fit flag (TOEFL / FACETS conventions) ────────────────────────────────────

def _fit_flag(msq) -> str:
    """Classify an infit/outfit MSQ value using standard Rasch thresholds."""
    if msq is None or (isinstance(msq, float) and np.isnan(msq)):
        return "no data"
    if msq > 1.5:
        return "underfit"    # rater/essay is unpredictable
    if msq < 0.5:
        return "overfit"     # rater/essay is too predictable (may be cloning)
    if msq > 1.3:
        return "marginal"    # worth monitoring
    return "acceptable"


# ── Core JMLE engine ──────────────────────────────────────────────────────────

def compute_mfrm(
    score_matrix: dict,
    max_score: int = 22,
    n_iter: int = 300,
    tol: float = 1e-4,
) -> dict | None:
    """
    Fit the 2-facet binomial Rasch model using JMLE.

    Parameters
    ----------
    score_matrix : dict  {essay_id: {rater_id: total_score}}
        Missing ratings are allowed (not every rater rates every essay).
    max_score    : int   Maximum possible score on the rubric.
    n_iter       : int   Maximum JMLE iterations.
    tol          : float Convergence criterion (max |Δparameter|).

    Returns
    -------
    dict with keys:
        rater_results  – per-rater measures, SE, infit/outfit, flags
        essay_results  – per-essay measures, SE, infit/outfit, fair scores
        rater_sep / rater_rel  – separation index and reliability (rater facet)
        essay_sep / essay_rel  – separation index and reliability (essay facet)
        converged / iterations – convergence diagnostics
        max_score              – passed through for display
    Returns None if there is insufficient data (< 2 essays or < 2 raters).
    """

    # ── 1. Index mapping ──────────────────────────────────────────────
    essay_ids = sorted(score_matrix.keys())
    rater_ids = sorted({rid for d in score_matrix.values() for rid in d})
    n_essays, n_raters = len(essay_ids), len(rater_ids)

    if n_essays < 2 or n_raters < 2:
        return None

    ei = {eid: i for i, eid in enumerate(essay_ids)}
    ri = {rid: i for i, rid in enumerate(rater_ids)}

    m = float(max_score)

    # ── 2. Data matrix  (NaN = not rated) ────────────────────────────
    X = np.full((n_essays, n_raters), np.nan)
    for eid, ratings in score_matrix.items():
        for rid, score in ratings.items():
            if rid in ri:
                X[ei[eid], ri[rid]] = float(score)

    valid = ~np.isnan(X)            # boolean mask of observed cells

    # Require at least 2 observations per row and column for estimation
    row_obs = valid.sum(axis=1)
    col_obs = valid.sum(axis=0)
    if (row_obs >= 1).sum() < 2 or (col_obs >= 1).sum() < 2:
        return None

    # ── 3. Initialise parameters ──────────────────────────────────────
    with np.errstate(invalid='ignore'):
        essay_props = np.nanmean(X, axis=1) / m
    essay_props = np.clip(essay_props, 0.02, 0.98)
    B = np.log(essay_props / (1.0 - essay_props))   # essay ability (logits)
    D = np.zeros(n_raters)                            # rater severity (logits)

    # ── 4. JMLE iterations ────────────────────────────────────────────
    converged = False
    iteration = 0

    for iteration in range(n_iter):
        B_prev = B.copy()
        D_prev = D.copy()

        # Expected scores and variances under current parameters
        theta = B[:, np.newaxis] - D[np.newaxis, :]   # (n_essays × n_raters)
        P = _expit(theta)
        E = m * P               # expected score
        V = m * P * (1.0 - P)  # binomial variance

        # -- Update essay abilities (B_n) ---------------------------------
        # Newton step: B_n += (Σ_i observed - Σ_i expected) / Σ_i variance
        for n in range(n_essays):
            mask = valid[n]
            if mask.sum() < 1:
                continue
            num = np.nansum(X[n, mask] - E[n, mask])
            den = np.nansum(V[n, mask])
            if den > 1e-9:
                B[n] += num / den
        # No centering of B — essays float on the scale defined by raters

        # -- Update rater severities (D_i) --------------------------------
        # Newton step: D_i += (Σ_n expected - Σ_n observed) / Σ_n variance
        # Sign is reversed vs. B because D opposes the score in the model.
        for i in range(n_raters):
            mask = valid[:, i]
            if mask.sum() < 1:
                continue
            num = np.nansum(E[mask, i] - X[mask, i])
            den = np.nansum(V[mask, i])
            if den > 1e-9:
                D[i] += num / den
        D -= D.mean()   # centre raters → defines the scale origin

        # -- Convergence check --------------------------------------------
        delta = max(
            np.max(np.abs(B - B_prev)),
            np.max(np.abs(D - D_prev)),
        )
        if delta < tol:
            converged = True
            break

    # ── 5. Final expected values and residuals ────────────────────────
    theta = B[:, np.newaxis] - D[np.newaxis, :]
    P = _expit(theta)
    E = m * P
    V = m * P * (1.0 - P)

    with np.errstate(invalid='ignore', divide='ignore'):
        Z2 = np.where(valid & (V > 1e-9), ((X - E) ** 2) / V, np.nan)
        # Z2[n,i] = squared standardised residual for cell (n,i)

    # ── 6. Standard errors: SE(B_n) = 1/√(Σ_i V_ni) ─────────────────
    B_se = np.array([
        1.0 / np.sqrt(np.nansum(V[n, valid[n]]))
        if valid[n].any() else np.nan
        for n in range(n_essays)
    ])
    D_se = np.array([
        1.0 / np.sqrt(np.nansum(V[valid[:, i], i]))
        if valid[:, i].any() else np.nan
        for i in range(n_raters)
    ])

    # ── 7. Infit / Outfit MSQ ─────────────────────────────────────────
    #
    #   Outfit MSQ  = mean(z²)            [unweighted; sensitive to outliers]
    #   Infit  MSQ  = Σ(V·z²) / Σ(V)     [information-weighted; more robust]
    #
    # Both ≈ 1.0 for data fitting the model.
    # > 1.5 → underfit (erratic)   < 0.5 → overfit (Guttman-like pattern)

    def _fit(z2_slice, v_slice):
        ok = ~np.isnan(z2_slice)
        if ok.sum() < 2:
            return None, None
        outfit = float(np.mean(z2_slice[ok]))
        denom  = float(np.sum(v_slice[ok]))
        infit  = float(np.sum(v_slice[ok] * z2_slice[ok]) / denom) if denom > 1e-9 else None
        return (
            round(infit,  3) if infit  is not None else None,
            round(outfit, 3),
        )

    rater_infit  = np.full(n_raters, np.nan)
    rater_outfit = np.full(n_raters, np.nan)
    for i in range(n_raters):
        mask = valid[:, i]
        if mask.sum() >= 2:
            inf, out = _fit(Z2[mask, i], V[mask, i])
            rater_infit[i]  = inf  if inf  is not None else np.nan
            rater_outfit[i] = out if out is not None else np.nan

    essay_infit  = np.full(n_essays, np.nan)
    essay_outfit = np.full(n_essays, np.nan)
    for n in range(n_essays):
        mask = valid[n]
        if mask.sum() >= 2:
            inf, out = _fit(Z2[n, mask], V[n, mask])
            essay_infit[n]  = inf  if inf  is not None else np.nan
            essay_outfit[n] = out if out is not None else np.nan

    # ── 8. Separation index & reliability ────────────────────────────
    #
    #   Observed SD² = True SD² + Mean SE²   (variance components)
    #   True SD      = √max(0, Obs_SD² − Mean_SE²)
    #   Separation G = True SD / Mean SE
    #   Reliability  = G² / (1 + G²)          [analogous to Cronbach α]
    #
    # Interpretation (rater facet): G > 2 → at least 3 distinct severity levels
    # are reliably separated. G < 1 → measures are unreliable.

    def _sep_rel(measures, se_arr):
        se_valid = se_arr[~np.isnan(se_arr)]
        if len(se_valid) < 2:
            return None, None
        obs_sd   = float(np.std(measures, ddof=1))
        mean_se  = float(np.mean(se_valid))
        true_sd  = float(np.sqrt(max(0.0, obs_sd ** 2 - mean_se ** 2)))
        sep      = true_sd / mean_se if mean_se > 1e-9 else 0.0
        rel      = sep ** 2 / (1.0 + sep ** 2)
        return round(sep, 2), round(rel, 2)

    rater_sep, rater_rel = _sep_rel(D, D_se)
    essay_sep, essay_rel = _sep_rel(B, B_se)

    # ── 9. Fair average scores ────────────────────────────────────────
    #
    # The "fair score" for essay n is the expected score from a rater of
    # average severity (D = 0), i.e. fair_n = expit(B_n) × m.
    # This removes systematic rater effects from the essay score, giving
    # a more equitable comparison across essays rated by different raters.

    fair_scores = {
        eid: round(float(_expit(B[n])) * m, 2)
        for n, eid in enumerate(essay_ids)
    }

    # Also compute original group mean per essay for comparison
    with np.errstate(invalid='ignore'):
        essay_obs_means = {
            eid: round(float(np.nanmean(X[n])), 2)
            for n, eid in enumerate(essay_ids)
        }

    # ── 10. Assemble output dicts ─────────────────────────────────────

    def _fmt(v):
        if v is None:
            return None
        try:
            if np.isnan(v):
                return None
        except (TypeError, ValueError):
            pass
        return round(float(v), 3)

    rater_results = {}
    for i, rid in enumerate(rater_ids):
        d   = float(D[i])
        inf = _fmt(rater_infit[i])
        out = _fmt(rater_outfit[i])
        rater_results[rid] = {
            "measure":    round(d, 3),
            "se":         _fmt(D_se[i]),
            "ci_lo":      round(d - 1.96 * float(D_se[i]), 2) if _fmt(D_se[i]) else None,
            "ci_hi":      round(d + 1.96 * float(D_se[i]), 2) if _fmt(D_se[i]) else None,
            "infit_msq":  inf,
            "outfit_msq": out,
            "infit_flag": _fit_flag(inf),
            "severity":   "lenient" if d < -0.5 else ("harsh" if d > 0.5 else "calibrated"),
        }

    essay_results = {}
    for n, eid in enumerate(essay_ids):
        b   = float(B[n])
        inf = _fmt(essay_infit[n])
        out = _fmt(essay_outfit[n])
        essay_results[eid] = {
            "measure":    round(b, 3),
            "se":         _fmt(B_se[n]),
            "ci_lo":      round(b - 1.96 * float(B_se[n]), 2) if _fmt(B_se[n]) else None,
            "ci_hi":      round(b + 1.96 * float(B_se[n]), 2) if _fmt(B_se[n]) else None,
            "infit_msq":  inf,
            "outfit_msq": out,
            "infit_flag": _fit_flag(inf),
            "fair_score": fair_scores[eid],
            "obs_mean":   essay_obs_means[eid],
            "n_raters":   int(valid[n].sum()),
        }

    return {
        "rater_results": rater_results,
        "essay_results": essay_results,
        "rater_ids":     rater_ids,
        "essay_ids":     essay_ids,
        "rater_sep":     rater_sep,
        "rater_rel":     rater_rel,
        "essay_sep":     essay_sep,
        "essay_rel":     essay_rel,
        "converged":     converged,
        "iterations":    iteration + 1,
        "max_score":     max_score,
        "n_essays":      n_essays,
        "n_raters":      n_raters,
    }
