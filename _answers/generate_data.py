"""
Synthetic UK unsecured personal-loan book for the Gini prep lab.

DON'T READ THIS BEFORE DOING THE EXERCISES - it contains the true data-generating
process, i.e. the answers. Kept so the data is reproducible.

Outputs (in ../data):
  loans.csv            one row per loan, application-time fields (+ two planted leaks)
  account_history.csv  6 monthly current-account summaries per applicant, pre-application
  performance.parquet  monthly loan-level panel, 2016-01 .. 2022-12
  macro.csv            unemployment history + 3 forward scenarios (2023-2032)
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(20261022)
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

N = 40_000
sig = lambda x: 1 / (1 + np.exp(-x))

# ---------------------------------------------------------------- macro
months = pd.period_range("2016-01", "2022-12", freq="M")
anchors = {"2016-01": 5.1, "2016-12": 4.8, "2017-12": 4.4, "2018-12": 4.0, "2019-12": 3.8,
           "2020-03": 4.0, "2020-12": 5.1, "2021-06": 4.8, "2021-12": 4.1, "2022-12": 3.8}
a_idx = [months.get_loc(pd.Period(k, "M")) for k in anchors]
unemp_hist = np.interp(np.arange(len(months)), a_idx, list(anchors.values()))
unemp_hist += rng.normal(0, 0.03, len(months))

fut = pd.period_range("2023-01", "2032-12", freq="M")
t = np.arange(len(fut))
base = 3.8 + 0.5 * (1 - np.exp(-t / 12))                          # drifts to ~4.3
upside = 3.8 - 0.3 * (1 - np.exp(-t / 12))                        # ~3.5
downside = 3.8 + 3.0 * np.exp(-((t - 20) / 14) ** 2) + 0.5 * (1 - np.exp(-t / 12))  # peak ~6.9 mid-2024
macro_hist = pd.DataFrame({"month": months.astype(str), "unemployment": unemp_hist.round(2),
                           "type": "history"})
macro_fut = pd.concat([pd.DataFrame({"month": fut.astype(str), "unemployment": s.round(2), "type": n})
                       for n, s in [("base", base), ("upside", upside), ("downside", downside)]])
macro = pd.concat([macro_hist, macro_fut], ignore_index=True)
macro.to_csv(OUT / "macro.csv", index=False)
unemp_by_m = dict(zip(months, unemp_hist))

# ---------------------------------------------------------------- applicants
orig = pd.PeriodIndex(rng.choice(pd.period_range("2016-01", "2020-12", freq="M"), N), freq="M")
q = rng.normal(0, 1, N)                      # latent credit quality (bureau visible)
b = rng.normal(0, 1, N)                      # latent cash-flow stress (NOT in bureau; visible in account data)

v2 = orig >= pd.Period("2019-07", "M")       # bureau moved to a new score scale
credit_score = np.where(v2, 728 + 74 * q + rng.normal(0, 22, N), 680 + 55 * q + rng.normal(0, 20, N))
credit_score = np.clip(credit_score.round(), 300, 999).astype(int)

util = sig(-0.6 * q + 0.3 * b + rng.normal(0, 0.8, N) - 0.4)
revolving = np.clip(util * 0.96 + rng.normal(0, 0.03, N), 0, 1.2)     # near-duplicate of util
inq = rng.poisson(np.exp(0.15 - 0.35 * q + 0.15 * b))
has_dlq = rng.random(N) < sig(-1.3 - 0.9 * q)
msd = np.where(has_dlq, np.ceil(72 * rng.beta(1.2, 1.6, N)), np.nan)
income = np.exp(rng.normal(np.log(31_000), 0.42, N)).round(-2)
loan_amount = np.clip(np.exp(rng.normal(np.log(9_000), 0.6, N)), 1_000, 35_000).round(-2)
term = rng.choice([24, 36, 48, 60], N, p=[0.15, 0.35, 0.25, 0.25])
dti = np.clip(rng.beta(2.4, 5.5, N) + 0.04 * b - 0.03 * (np.log(income) - np.log(31_000)), 0.01, 0.85)
emp_years = np.round(rng.exponential(6, N), 1)
homeowner = (rng.random(N) < sig(0.4 * q + 0.05 * (emp_years - 6) - 0.2)).astype(int)
regions = np.array(["North West", "North East", "Yorkshire", "Midlands", "London",
                    "South East", "South West", "Scotland", "Wales"])
reg_eff = dict(zip(regions, [0.12, 0.15, 0.08, 0.05, -0.05, -0.12, -0.08, 0.04, 0.10]))
region = rng.choice(regions, N, p=[.12, .05, .09, .16, .15, .15, .09, .11, .08])
district = np.array([f"{r[:2].upper()}{i:03d}" for r, i in zip(region, rng.integers(0, 160, N))])
channel = rng.choice(["direct", "aggregator", "broker"], N, p=[0.5, 0.35, 0.15])
ch_eff = {"direct": -0.1, "aggregator": 0.05, "broker": 0.25}

eps = rng.normal(0, 0.6, N)                 # unobservable
z0 = (-0.80 * q + 0.60 * b + 1.0 * (util - 0.4) + 2.4 * (dti - 0.3) + 0.09 * inq
      + 0.30 * (has_dlq & (msd <= 12)) + 0.10 * (has_dlq & (msd > 12))
      - 0.20 * homeowner - 0.015 * (emp_years - 6)
      + 0.25 * np.log(loan_amount / income * 4) + np.vectorize(reg_eff.get)(region)
      + np.vectorize(ch_eff.get)(channel) + eps)

rate = np.round(0.055 + 0.07 * sig(0.9 * (-0.8 * q + 1.4 * (util - 0.4) + 2.4 * (dti - .3))) + rng.normal(0, .015, N), 4)

# ---------------------------------------------------------------- account history (6m pre-application)
rows = []
for k in range(-6, 0):
    trend = (k + 6) / 5                       # 0 -> 1 towards application
    infl = income / 12 * np.exp(rng.normal(0, 0.12, N)) * (1 - 0.05 * np.clip(b, 0, None) * trend)
    outf = infl * np.exp(rng.normal(0.02 + 0.03 * b, 0.08, N))
    od = rng.poisson(np.exp(0.4 - 0.3 * q + 0.35 * b + 0.2 * b * trend).clip(0, 30))
    rdd = rng.poisson(np.exp(-2.6 - 0.3 * q + 0.55 * b + 0.6 * np.clip(b, 0, None) * trend))
    gamb = np.where(rng.random(N) < sig(-1.8 + 0.6 * b), np.exp(rng.normal(3.5 + 0.5 * b, 0.9, N)), 0)
    bal = np.maximum(-1500, income / 24 * np.exp(rng.normal(0, .5, N)) - 400 * b - 300 * od / 5)
    sal = (rng.random(N) < sig(2.5 - 0.4 * b)).astype(int)
    rows.append(pd.DataFrame({"loan_id": np.arange(N), "month_offset": k,
                              "total_inflow": infl.round(2), "total_outflow": outf.round(2),
                              "avg_balance": bal.round(2), "overdraft_days": np.minimum(od, 30),
                              "returned_dd_count": rdd, "gambling_spend": gamb.round(2),
                              "salary_credit": sal}))
acct = pd.concat(rows).sort_values(["loan_id", "month_offset"])
acct.to_csv(OUT / "account_history.csv", index=False)

# ---------------------------------------------------------------- monthly performance simulation
def mob_curve(m):
    return -0.9 * np.exp(-m / 3) + 0.35 * np.exp(-((m - 14) / 12) ** 2) - 0.004 * m

all_m = pd.period_range("2016-01", "2022-12", freq="M")
state = np.full(N, -1)          # -1 not yet originated, 0 cur, 1 30-59, 2 60-89, 3 default, 9 closed
qt = q.copy()
bal = loan_amount.astype(float).copy()
mr = rate / 12
pmt = loan_amount * mr / (1 - (1 + mr) ** (-term))
mob = np.zeros(N, int)
def_month = np.full(N, None, object)
close_month = np.full(N, None, object)
close_reason = np.full(N, None, object)
months_in_def = np.zeros(N, int)
ever30_12 = np.zeros(N, bool)
recs = []
for m in all_m:
    starting = (orig == m)
    state[starting] = 0
    act = state >= 0
    act &= state != 9
    live = act & (state < 3)
    mob[act] += 1
    # bureau drift
    qt[live] += rng.normal(0, 0.08, live.sum())
    shock = live & (rng.random(N) < 0.007)                              # life event (job loss etc.)
    qt[shock] -= rng.normal(2.4, 0.5, shock.sum())
    z = z0 - 0.80 * (qt - q)
    u = unemp_by_m[m]
    col = 0.0
    if m >= pd.Period("2022-03", "M"):
        col = min(1.0, (m - pd.Period("2022-03", "M")).n / 6)         # cost-of-living ramp
    lp = -5.15 + 0.62 * z + mob_curve(mob) + 0.33 * (u - 4.5) + 0.95 * col * (dti > 0.40)
    h = sig(lp)
    r = rng.random(N)
    new = state.copy()
    s0, s1, s2 = live & (state == 0), live & (state == 1), live & (state == 2)
    new[s0 & (r < h)] = 1
    roll1 = sig(-0.1 + 0.25 * z)
    new[s1] = np.where(r[s1] < roll1[s1], 2, 0)
    roll2 = sig(1.0 + 0.2 * z)
    new[s2] = np.where(r[s2] < roll2[s2], 3, 0)
    ever30_12 |= (new >= 1) & (new < 9) & (mob <= 12) & live
    newly_def = (new == 3) & (state == 2)
    def_month[newly_def] = str(m)
    # amortise performing / delinquent (delinquent accrue - keep balance)
    perf = live & (new == 0)
    bal[perf] = np.maximum(0, bal[perf] * (1 + mr[perf]) - pmt[perf])
    # prepay / maturity
    pre = perf & (rng.random(N) < 0.011)
    mat = perf & ((mob >= term) | (bal < 1))
    # write-off after 6 months in default
    months_in_def[new == 3] += 1
    wo = (new == 3) & (months_in_def > 6)
    for mask, why in [(pre & ~mat, "prepaid"), (mat, "matured"), (wo, "written_off")]:
        close_month[mask] = str(m); close_reason[mask] = why
    show = act & ~wo
    score_now = np.where(new >= 1, -45 * np.minimum(new, 3) / 1.5, 0)
    bureau = np.clip(680 + 55 * qt + rng.normal(0, 10, N) + score_now, 300, 999).round()
    idx = np.where(show)[0]
    recs.append(pd.DataFrame({"loan_id": idx, "month": str(m), "mob": mob[idx],
                              "balance": bal[idx].round(2), "dpd_bucket": new[idx],
                              "bureau_score": bureau[idx].astype(int)}))
    new[pre | mat | wo] = 9
    state = new

perf = pd.concat(recs, ignore_index=True)
perf["dpd_bucket"] = perf["dpd_bucket"].clip(upper=3)
perf.to_parquet(OUT / "performance.parquet", index=False)

# ---------------------------------------------------------------- loans table (+ planted leaks)
first12 = perf[perf.mob <= 12].groupby("loan_id").dpd_bucket.max().reindex(range(N)).fillna(0)
bad12 = (first12 >= 3).astype(int).values
# LEAK 1: post-origination collections activity
collections_letters = rng.poisson(np.exp(-2.2 + 2.6 * ever30_12 + 0.8 * bad12))
# LEAK 2: target-encoded district bad rate computed over the WHOLE book, own row included
tmp = pd.DataFrame({"d": district, "bad": bad12})
g = tmp.groupby("d").bad.agg(["sum", "count"])
prior = bad12.mean()
district_rate = ((g["sum"] + 5 * prior) / (g["count"] + 5)).reindex(district).values

loans = pd.DataFrame({
    "loan_id": np.arange(N), "orig_month": orig.astype(str), "loan_amount": loan_amount,
    "term_months": term, "interest_rate": rate, "channel": channel, "region": region,
    "postcode_district": district, "credit_score": credit_score, "util_ratio": util.round(4),
    "revolving_bal_to_limit": revolving.round(4), "num_inquiries_6m": inq,
    "months_since_last_delinq": msd, "annual_income": income, "dti": dti.round(4),
    "emp_years": emp_years, "homeowner": homeowner,
    "collections_letters_12m": collections_letters,
    "district_bad_rate": district_rate.round(4),
    "close_month": close_month, "close_reason": close_reason, "default_month": def_month,
})
loans.to_csv(OUT / "loans.csv", index=False)
print(f"loans {len(loans):,}  panel rows {len(perf):,}  12m bad rate {bad12.mean():.3%}")
print("by vintage year:", pd.Series(bad12).groupby(orig.year).mean().round(4).to_dict())
print("ever default:", loans.default_month.notna().mean().round(4))
