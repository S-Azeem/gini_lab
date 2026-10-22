"""
Checkpoints for the Gini prep lab.

Each check compares your output to the answer key with deliberately loose tolerances:
your binning, feature set and modelling choices can legitimately differ from mine.
A ❌ means "go and look", not "you are wrong". Hints never give the number away.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

_KEY_PATH = Path(__file__).resolve().parent / "_answers" / "answer_key.json"
_KEY = json.loads(_KEY_PATH.read_text()) if _KEY_PATH.exists() else {}


def _ok(msg):  print(f"  ✅ {msg}")
def _bad(msg): print(f"  ❌ {msg}")
def _info(msg): print(f"  ℹ️  {msg}")
def _todo(name): print(f"  ⏳ {name}: not done yet"); return False


def _needs_key(fn):
    import functools
    @functools.wraps(fn)
    def w(*a, **k):
        if not _KEY:
            print(f"(answer key not found at {_KEY_PATH} — skipping {fn.__name__})"); return None
        try:
            return fn(*a, **k)
        except Exception as e:  # half-finished TODOs land here
            print(f"  ⏳ {fn.__name__}: couldn't check yet ({type(e).__name__}: {e})"); return False
    return w


def _close(x, key, tol, label, hint):
    if abs(x - _KEY[key]) <= tol:
        _ok(f"{label} = {x:.4f}")
        return True
    _bad(f"{label} = {x:.4f} — outside the expected range. Hint: {hint}")
    return False


# ------------------------------------------------------------------ module 1
@_needs_key
def m1(dev, oot):
    print("Checkpoint 1 · target & sample")
    if dev is None or oot is None: return _todo("dev/oot")
    if "bad" not in dev: return _bad("dev needs a 'bad' column")
    good = True
    if len(dev) + len(oot) != _KEY["n_loans"]:
        _bad("dev + oot should cover every loan exactly once"); good = False
    else:
        _ok(f"{len(dev):,} dev / {len(oot):,} OOT loans")
    good &= _close(dev.bad.mean(), "dev_bad_rate", 0.003, "dev bad rate",
                   "bad = reached 90+ DPD (dpd_bucket 3) at mob ≤ 12. Loans that close early are goods.")
    good &= _close(oot.bad.mean(), "oot_bad_rate", 0.003, "OOT bad rate", "same definition as dev")
    return good


# ------------------------------------------------------------------ module 3
@_needs_key
def m3(iv, account_feature_names):
    print("Checkpoint 3 · binning, WoE, IV")
    if iv is None: return _todo("iv")
    iv = pd.Series(iv).sort_values(ascending=False)
    good = _close(iv.get("credit_score", np.nan), "iv_credit_score", 0.10, "IV(credit_score)",
                  "≈5% min bin size, ≤8 bins, monotonic merges, WoE = ln(%good/%bad) with a 0.5 floor.")
    top10 = list(iv.index[:10])
    if any(f in account_feature_names for f in top10):
        _ok("an account-history feature makes the IV top 10 — the transaction data is pulling its weight")
    else:
        _bad("no account-history feature in the IV top 10 — check your aggregations"); good = False
    suspicious = [f for f in iv.index if iv[f] > 0.6]
    _info(f"features with IV > 0.6 (read these before module 4): {suspicious}")
    return good


# ------------------------------------------------------------------ module 4
@_needs_key
def m4(exclude):
    print("Checkpoint 4 · leakage & conceptual review")
    if not exclude: return _todo("EXCLUDE")
    leaks = set(_KEY["leaks"])
    found = leaks & set(exclude)
    if found == leaks:
        _ok("both planted leaks excluded")
    else:
        _bad(f"{len(leaks - found)} leaking feature(s) still in. Ask of every field: was this known, "
             "and computed only from information available, at the decision date?")
    if "interest_rate" in exclude:
        _ok("interest_rate excluded — defensible: it is an output of the old risk assessment")
    else:
        _info("you kept interest_rate. Be ready to defend it: price is set from the old scorecard, "
              "so the new model partly learns the old one.")
    return found == leaks


# ------------------------------------------------------------------ module 5
@_needs_key
def m5(selected, val_gini):
    print("Checkpoint 5 · MIV forward selection")
    if not selected: return _todo("selected")
    good = True
    if {"util_ratio", "revolving_bal_to_limit"} <= set(selected):
        _bad("util_ratio AND revolving_bal_to_limit both selected — your correlation cap isn't biting"); good = False
    if set(selected) & set(_KEY["leaks"]):
        _bad("a leaking feature got selected"); good = False
    if 5 <= len(selected) <= 15:
        _ok(f"{len(selected)} features selected")
    else:
        _bad(f"{len(selected)} features — expected roughly 6–12. Check your MIV stopping threshold (≈0.02)"); good = False
    if val_gini >= _KEY["val_gini"] - 0.04:
        _ok(f"validation Gini {val_gini:.3f}")
    else:
        _bad(f"validation Gini {val_gini:.3f} looks low"); good = False
    return good


# ------------------------------------------------------------------ module 6
@_needs_key
def m6(params_woe, stab_plain, stab_enet):
    print("Checkpoint 6 · scorecard + elastic net stability")
    good = True
    if params_woe is not None:
        p = pd.Series(params_woe).drop("const", errors="ignore")
        if (p < 0).all():
            _ok("all WoE coefficients negative (WoE = ln(good/bad), target = bad) — signs consistent")
        else:
            _bad(f"positive WoE coefficient(s): {list(p[p >= 0].index)}. That usually means a "
                 "correlated feature is fighting another one — drop or re-bin"); good = False
    else:
        _todo("params_woe"); good = False
    if stab_plain is None or stab_enet is None: return _todo("stability tables")
    flip = stab_plain.loc["util_ratio", "neg_share"]
    _info(f"plain logistic: util_ratio coefficient negative in {flip:.0%} of bootstraps "
          f"(true effect is positive), sd {stab_plain.loc['util_ratio', 'sd']:.3f}")
    _info(f"elastic net:    negative in {stab_enet.loc['util_ratio', 'neg_share']:.0%}, "
          f"sd {stab_enet.loc['util_ratio', 'sd']:.3f}")
    if stab_enet.loc["util_ratio", "sd"] < stab_plain.loc["util_ratio", "sd"]:
        _ok("elastic net stabilised the correlated pair")
    else:
        _bad("elastic net isn't more stable — is your penalty strong enough (try C ≈ 0.05)?"); good = False
    return good


# ------------------------------------------------------------------ module 7
@_needs_key
def m7(gini_oot, psi_post, csi_post, calib_ratio_post):
    print("Checkpoint 7 · validation")
    if gini_oot is None: return _todo("validation")
    good = _close(gini_oot, "oot_gini", 0.05, "OOT Gini", "score OOT with dev-fitted bins and WoE — no refitting")
    _info(f"score PSI (dev vs OOT from Jul-2019) = {psi_post:.3f} — note what it does and doesn't tell you")
    top = pd.Series(csi_post).idxmax()
    if top == "credit_score" and pd.Series(csi_post).max() > 0.25:
        _ok(f"CSI flags credit_score ({pd.Series(csi_post).max():.2f}) even though the score PSI looked calm")
    else:
        _bad(f"top CSI feature is {top}. Compute CSI on the dev bins for each selected feature"); good = False
    if calib_ratio_post > 1.1:
        _ok(f"post-Jul-2019 actual/predicted = {calib_ratio_post:.2f} → under-prediction despite a healthy Gini")
    else:
        _bad(f"actual/predicted = {calib_ratio_post:.2f} — compute it for OOT loans from 2019-07 only"); good = False
    return good


# ------------------------------------------------------------------ module 8
@_needs_key
def m8(score_fn):
    print("Checkpoint 8 · points scaling")
    if score_fn is None: return _todo("score_fn")
    tests = {0.02: 600.0, 0.04: 580.0, 0.0099: 620.0}
    out = {p: score_fn(np.array([p]))[0] for p in tests}
    # 600 at good:bad 50:1, PDO 20  →  odds 50 ⇔ p = 1/51
    exp = {p: 600 + 20 / np.log(2) * np.log(((1 - p) / p) / 50) for p in tests}
    if all(abs(out[p] - exp[p]) < 0.5 for p in tests):
        _ok("600 points at 50:1, 20 points to double the odds")
        return True
    _bad(f"got {[(p, round(out[p], 1)) for p in tests]}. Score = offset + factor·ln(good:bad odds)")
    return False


# ------------------------------------------------------------------ module 9
@_needs_key
def m9a(params, train_gini):
    print("Checkpoint 9a · 12-month behavioural PD")
    if params is None: return _todo("pd12 params")
    good = True
    u = [k for k in params.index if "unemp" in k]
    if u and params[u[0]] > 0:
        _ok(f"unemployment coefficient {params[u[0]]:.3f} > 0")
    else:
        _bad("no positive unemployment term — you need it to make the PD forward-looking"); good = False
    good &= _close(train_gini, "pd12_gini", 0.05, "snapshot Gini", "features: bureau_score, dpd_bucket, mob, dti, unemployment")
    return good


@_needs_key
def m9b(lifetime_pd, annualise):
    print("Checkpoint 9b · lifetime PD mechanics")
    if lifetime_pd is None: return _todo("functions")
    hz = np.full((1, 24), 0.01)
    lp = lifetime_pd(hz)[0]
    good = True
    if abs(lp - (1 - 0.99 ** 24)) < 1e-9: _ok(f"lifetime PD from monthly hazards = {lp:.4f}")
    else: _bad("lifetime PD = 1 − Π(1 − h_t)"); good = False
    a = annualise(np.array([lp]), np.array([24]))[0]
    if abs(a - (1 - 0.99 ** 12)) < 1e-9: _ok(f"annualised = {a:.4f}")
    else: _bad("annualised PD = 1 − (1 − LPD)^(12 / remaining months)"); good = False
    return good


@_needs_key
def m9c(grid):
    print("Checkpoint 9c · SICR threshold")
    if grid is None: return _todo("sicr grid")
    best = grid.sort_values("youden", ascending=False).iloc[0]
    good = _close(best.youden, "sicr_best_youden", 0.06, "best Youden J", "flag = (ratio > k AND Δann_PD > a) OR dpd ≥ 1")
    if 0.03 <= best.stage2_share <= 0.30: _ok(f"Stage 2 share at the optimum {best.stage2_share:.1%}")
    else: _bad(f"Stage 2 share {best.stage2_share:.1%} looks off"); good = False
    _info(f"your optimum: k = {best.k}, absolute floor = {best['abs']:.3f}. Now argue whether you'd actually use it.")
    return good


@_needs_key
def m9d(summary, ecl_by_scenario):
    print("Checkpoint 9d · ECL")
    if summary is None: return _todo("ECL summary")
    good = True
    cov = summary["ecl"] / summary["gca"]
    if cov.is_monotonic_increasing: _ok(f"coverage rises by stage: {', '.join(f'{c:.1%}' for c in cov)}")
    else: _bad("coverage should rise Stage 1 → 2 → 3"); good = False
    tot = summary["ecl"].sum()
    if abs(tot / _KEY["ecl_total"] - 1) < 0.30: _ok(f"total ECL £{tot:,.0f}")
    else: _bad(f"total ECL £{tot:,.0f} is far from mine — check EAD amortisation, discounting and the 12m cut for Stage 1"); good = False
    w = {"base": .5, "upside": .2, "downside": .3}
    weighted = sum(w[s] * ecl_by_scenario[s] for s in w)
    if weighted > ecl_by_scenario["base"]:
        _ok(f"probability-weighted ECL is {weighted / ecl_by_scenario['base'] - 1:.1%} above base — the non-linearity")
    else:
        _bad("weighted ECL should exceed base-only ECL"); good = False
    return good


@_needs_key
def m9e(segment, uplift):
    print("Checkpoint 9e · backtest & overlay")
    if segment is None: return _todo("overlay")
    good = True
    if "dti" in segment.lower(): _ok(f"segment: {segment}")
    else: _bad("that isn't where the model is missing. Backtest predicted vs actual by every model driver"); good = False
    if _KEY["uplift_lo"] <= uplift <= _KEY["uplift_hi"]: _ok(f"uplift ×{uplift:.2f}")
    else: _bad(f"uplift ×{uplift:.2f} out of range — actual / predicted 12m default rate for the segment"); good = False
    return good
