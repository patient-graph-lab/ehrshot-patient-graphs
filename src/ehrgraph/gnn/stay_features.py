"""Stay-level feature blocks for the GNN / linear experiments: age and sex, admission
context, prior utilisation, a comorbidity score and key labs.

Every block returns a `FeatureBlock` whose rows follow `tg.nodes` (a list of
`(patient_id, prediction_time)`), exactly like `dx_svd` and `clmbr`, so it can be named in
`FEATURE_SETS` and flows through `assemble` / `build_pyg_data` / `standardize` unchanged.

TIME RULE (what keeps these features honest): a feature may only use events at or before the
stay's prediction time `t`. Each builder enforces that itself and says how. Nothing here uses
an outcome. Missing values are filled with a median taken over all stays (label-free, the same
kind of transductive step as the SVD) and flagged with a `*_missing` column.

Which blocks depend on the diagnosis `source` ("safe" vs "leaky")?
  * `comorbidity` is computed from the diagnosis vectors, so it FOLLOWS `source` and belongs in
    LEAKABLE_BLOCKS (a cheating record contains later diagnoses, so its score is inflated).
  * `demo`, `admission`, `prior_use`, `labs` use only pre-`t` events and are identical under
    every condition, like CLMBR. They are NOT in LEAKABLE_BLOCKS.

Input tables (plain DataFrames; adapt the column names in the loader, not here). `patient_id`
must have the same type as in `tg.nodes` (both are cast to str here):
  person : patient_id, birth (datetime), sex ("F" / "M" / anything else = unknown)
  visits : patient_id, start (datetime), kind ("inpatient" | "ed" | anything else)
  labs   : patient_id, time (datetime), name (short key, e.g. "creatinine"), value (float)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ehrgraph.gnn.node_features import FeatureBlock


def _nodes_frame(nodes: list) -> pd.DataFrame:
    return pd.DataFrame({"patient_id": [str(n[0]) for n in nodes], "t": pd.to_datetime([n[1] for n in nodes])})


def _fill(values: np.ndarray):
    """Median-fill a column; returns (filled, missing_flag). All-missing -> zeros."""
    v = np.asarray(values, dtype=np.float64)
    miss = ~np.isfinite(v)
    med = np.nanmedian(v) if (~miss).any() else 0.0
    return np.where(miss, med, v).astype(np.float32), miss.astype(np.float32)


# ------------------------------------------------------------------ age and sex
def demo_block(nodes: list, person: pd.DataFrame) -> FeatureBlock:
    """Age in years AT the prediction time, and sex. Uses only birth date and sex."""
    nf = _nodes_frame(nodes)
    pr = person.assign(patient_id=person["patient_id"].astype(str)).drop_duplicates("patient_id").set_index("patient_id")
    birth = pd.to_datetime(nf["patient_id"].map(pr["birth"]))
    age = ((nf["t"] - birth).dt.days / 365.25).clip(0, 110).to_numpy()
    sex = nf["patient_id"].map(pr["sex"]).astype(str).str.upper().str[:1]
    age_f, age_miss = _fill(age)
    cols = [age_f, (sex == "F").to_numpy(np.float32), (~sex.isin(["F", "M"])).to_numpy(np.float32), age_miss]
    return FeatureBlock("demo", np.column_stack(cols), ["age_years", "sex_female", "sex_unknown", "age_missing"])


# ------------------------------------------------------------------ admission context
def admission_block(nodes: list, visits: pd.DataFrame, within: str = "1D", ed_window: str = "1D") -> FeatureBlock:
    """What kind of encounter the stay is. `current visit` = the latest visit that STARTED at or
    before t and no more than `within` earlier (for a prediction made at admission that is the
    admission itself; for one made at discharge after a long stay raise `within`). Uses only
    its start time and kind, never its end or length. Also: an ED visit in the `ed_window`
    before t, and the hour / weekday of t."""
    nf = _nodes_frame(nodes)
    v = visits.assign(patient_id=visits["patient_id"].astype(str), start=pd.to_datetime(visits["start"]))
    kind = v["kind"].astype(str).str.lower()
    v = v.assign(is_ip=(kind == "inpatient").astype(float), is_ed=(kind == "ed").astype(float),
                 is_other=(~kind.isin(["inpatient", "ed"])).astype(float))
    by_pat = {pid: g.sort_values("start") for pid, g in v.groupby("patient_id")}
    w, edw = pd.Timedelta(within), pd.Timedelta(ed_window)
    out = np.zeros((len(nf), 4), dtype=np.float32)          # ip, ed, other, found
    ed_prior = np.zeros(len(nf), dtype=np.float32)
    for i, (pid, t) in enumerate(zip(nf["patient_id"], nf["t"])):
        g = by_pat.get(pid)
        if g is None:
            continue
        starts = g["start"].to_numpy()
        j = np.searchsorted(starts, np.datetime64(t), side="right") - 1      # latest start <= t
        if j >= 0 and t - pd.Timestamp(starts[j]) <= w:
            out[i] = [g["is_ip"].iat[j], g["is_ed"].iat[j], g["is_other"].iat[j], 1.0]
        lo = np.searchsorted(starts, np.datetime64(t - edw), side="left")
        ed_prior[i] = float(g["is_ed"].to_numpy()[lo:j + 1].sum() > 0) if j >= lo else 0.0
    hour = nf["t"].dt.hour.to_numpy() + nf["t"].dt.minute.to_numpy() / 60
    x = np.column_stack([out, ed_prior, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                         (nf["t"].dt.dayofweek.to_numpy() >= 5).astype(np.float32)])
    return FeatureBlock("admission", x.astype(np.float32),
                        ["visit_inpatient", "visit_ed", "visit_other", "visit_found", "ed_in_prior_window",
                         "hour_sin", "hour_cos", "weekend"])


# ------------------------------------------------------------------ prior utilisation
def prior_use_block(nodes: list, visits: pd.DataFrame, days: int = 365, lag: str = "0D") -> FeatureBlock:
    """Number of earlier inpatient admissions and ED visits in the `days` before t, and the days
    since the last inpatient admission (capped at 2 * days). Counts only visits that started
    STRICTLY BEFORE `t - lag`, so the current visit is excluded when lag = 0 and the prediction
    is made at its start. For a prediction made at the end of a visit, set `lag` to about that
    visit's length so the visit does not count itself."""
    nf = _nodes_frame(nodes)
    v = visits.assign(patient_id=visits["patient_id"].astype(str), start=pd.to_datetime(visits["start"]))
    kind = v["kind"].astype(str).str.lower()
    ip = {pid: np.sort(g["start"].to_numpy()) for pid, g in v[kind == "inpatient"].groupby("patient_id")}
    ed = {pid: np.sort(g["start"].to_numpy()) for pid, g in v[kind == "ed"].groupby("patient_id")}
    horizon, lag = pd.Timedelta(days=days), pd.Timedelta(lag)
    x = np.zeros((len(nf), 3), dtype=np.float32)
    for i, (pid, t) in enumerate(zip(nf["patient_id"], nf["t"])):
        hi, lo = np.datetime64(t - lag), np.datetime64(t - lag - horizon)
        for k, d in ((0, ip.get(pid)), (1, ed.get(pid))):
            if d is not None:
                x[i, k] = np.searchsorted(d, hi, side="left") - np.searchsorted(d, lo, side="left")
        d = ip.get(pid)
        last = np.searchsorted(d, hi, side="left") - 1 if d is not None else -1
        x[i, 2] = min((t - pd.Timestamp(d[last])).days, 2 * days) if last >= 0 else 2 * days
    x[:, :2] = np.log1p(x[:, :2])
    x[:, 2] = np.log1p(x[:, 2])
    return FeatureBlock("prior_use", x, [f"log_n_inpatient_{days}d", f"log_n_ed_{days}d", "log_days_since_inpatient"])


# ------------------------------------------------------------------ comorbidity (from the diagnosis vectors)
def _rng(prefix: str, a: int, b: int, width: int = 2):
    return [f"{prefix}{i:0{width}d}" for i in range(a, b + 1)]

# (name, weight, ICD-10 prefixes, ICD-9 prefixes); codes are matched with the dot removed.
# A compact version of the Quan et al. Charlson coding. Check it against a published implementation
# (for example the `comorbidipy` package) before quoting the score in a paper.
_CHARLSON = [
    ("mi", 1, ["I21", "I22", "I252"], ["410", "412"]),
    ("chf", 1, ["I099", "I110", "I130", "I132", "I255", "I42", "I43", "I50", "P290"],
     ["39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493", "4254", "4255", "4256", "4257", "4258", "4259", "428"]),
    ("pvd", 1, ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559", "Z958", "Z959"],
     ["0930", "4373", "440", "441", "4431", "4432", "4433", "4434", "4435", "4436", "4437", "4438", "4439", "4471", "5571", "5579", "V434"]),
    ("cvd", 1, ["G45", "G46", "H340"] + _rng("I", 60, 69), ["36234"] + _rng("", 430, 438, 3)),
    ("dementia", 1, ["F00", "F01", "F02", "F03", "F051", "G30", "G311"], ["290", "2941", "3312"]),
    ("copd", 1, ["I278", "I279", "J684", "J701", "J703"] + _rng("J", 40, 47) + _rng("J", 60, 67),
     ["4168", "4169", "5064", "5081", "5088"] + _rng("", 490, 505, 3)),
    ("rheumatic", 1, ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
     ["4465", "7100", "7101", "7102", "7103", "7104", "7140", "7141", "7142", "7148", "725"]),
    ("peptic_ulcer", 1, ["K25", "K26", "K27", "K28"], ["531", "532", "533", "534"]),
    ("liver_mild", 1, ["B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715", "K717", "K73", "K74",
                       "K760", "K762", "K763", "K764", "K768", "K769", "Z944"],
     ["07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709", "570", "571", "5733", "5734", "5738", "5739", "V427"]),
    ("diabetes", 1, [f"E{d}{s}" for d in (10, 11, 13, 14) for s in (0, 1, 6, 8, 9)], ["2500", "2501", "2502", "2503", "2508", "2509"]),
    ("diabetes_complic", 2, [f"E{d}{s}" for d in (10, 11, 13, 14) for s in (2, 3, 4, 5, 7)], ["2504", "2505", "2506", "2507"]),
    ("hemiplegia", 2, ["G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831", "G832", "G833", "G834", "G839"],
     ["3341", "342", "343", "3440", "3441", "3442", "3443", "3444", "3445", "3446", "3449"]),
    ("renal", 2, ["I120", "I131", "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992"] + [f"N03{i}" for i in range(2, 8)] + [f"N05{i}" for i in range(2, 8)],
     ["40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493", "582", "585", "586", "5880", "V420", "V451", "V56"] + [f"583{i}" for i in range(8)]),
    ("cancer", 2, _rng("C", 0, 26) + _rng("C", 30, 34) + _rng("C", 37, 41) + ["C43"] + _rng("C", 45, 58) + _rng("C", 60, 76) + _rng("C", 81, 85) + ["C88"] + _rng("C", 90, 97),
     _rng("", 140, 172, 3) + _rng("", 174, 195, 3) + _rng("", 200, 208, 3) + ["2386"]),
    ("liver_severe", 3, ["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767"],
     ["4560", "4561", "4562", "5722", "5723", "5724", "5725", "5726", "5727", "5728"]),
    ("metastatic", 6, ["C77", "C78", "C79", "C80"], ["196", "197", "198", "199"]),
    ("aids", 6, ["B20", "B21", "B22", "B23", "B24"], ["042", "043", "044"]),
]
_OVERRIDES = {"liver_mild": "liver_severe", "diabetes": "diabetes_complic", "cancer": "metastatic"}   # the milder one is dropped


def _norm_code(code: str):
    """('icd10' | 'icd9' | None, code without vocabulary prefix, dots or spaces, upper case)."""
    s = str(code).upper()
    vocab = s.split("/")[0] if "/" in s else ""
    c = s.split("/")[-1].replace(".", "").replace(" ", "")
    if "ICD10" in vocab:
        return "icd10", c
    if "ICD9" in vocab:
        return "icd9", c
    if vocab:
        return None, c
    return ("icd9" if (c[:1] == "V" or not c[:1].isalpha()) else "icd10"), c    # no vocabulary prefix: ICD-9 is numeric or V..; ICD-10 starts with another letter


def comorbidity_block(matrix, codes: list, name: str = "comorbidity") -> FeatureBlock:
    """Charlson indicators (one 0/1 column per condition) and the weighted score, computed from
    the SAME binary diagnosis matrix that feeds `dx_svd`. So with source="leaky" it sees the
    whole record and is inflated, exactly like the diagnosis features: list it in LEAKABLE_BLOCKS."""
    M = sp.csr_matrix(matrix)
    norm = [_norm_code(c) for c in codes]
    ind = {}
    for cname, _w, p10, p9 in _CHARLSON:
        p10, p9 = tuple(p10), tuple(p9)
        mask = np.array([(sys == "icd10" and c.startswith(p10)) or (sys == "icd9" and c.startswith(p9)) for sys, c in norm])
        ind[cname] = (np.asarray((M[:, np.flatnonzero(mask)] > 0).sum(axis=1)).ravel() > 0) if mask.any() else np.zeros(M.shape[0], bool)
    weights = {c: w for c, w, *_ in _CHARLSON}
    score = np.zeros(M.shape[0])
    for cname in ind:
        on = ind[cname] & ~ind[_OVERRIDES[cname]] if cname in _OVERRIDES else ind[cname]
        score += weights[cname] * on
    x = np.column_stack([score] + [ind[c].astype(float) for c in ind]).astype(np.float32)
    return FeatureBlock(name, x, ["charlson_score"] + [f"cm_{c}" for c in ind],
                        meta={"codes_matched": {c: int(v.sum()) for c, v in ind.items()}})


# ------------------------------------------------------------------ labs
def labs_block(nodes: list, labs: pd.DataFrame, names: list, window: str = "1D", clip=(0.01, 0.99)) -> FeatureBlock:
    """The most recent value of each lab in `names` within `window` BEFORE (or at) t, a missing
    flag per lab, and the number of labs measured. Values are clipped to the `clip` quantiles
    over all stays (label-free) to tame extremes.

    CHECK `prediction_time`. If a task predicts AT ADMISSION (guo_los), labs drawn on the
    admission day after the admission time are in the FUTURE and are correctly excluded here;
    most labs will then be missing. Using them means moving the prediction time later (a
    different task), not a feature choice."""
    nf = _nodes_frame(nodes)
    L = labs.assign(patient_id=labs["patient_id"].astype(str), time=pd.to_datetime(labs["time"]))
    w = pd.Timedelta(window)
    cols, colnames, nmeas = [], [], np.zeros(len(nf), dtype=np.float32)
    for lab in names:
        g = L[L["name"] == lab].sort_values("time")
        times = {pid: (d["time"].to_numpy(), d["value"].to_numpy(dtype=float)) for pid, d in g.groupby("patient_id")}
        v = np.full(len(nf), np.nan)
        for i, (pid, t) in enumerate(zip(nf["patient_id"], nf["t"])):
            d = times.get(pid)
            if d is None:
                continue
            j = np.searchsorted(d[0], np.datetime64(t), side="right") - 1
            if j >= 0 and t - pd.Timestamp(d[0][j]) <= w:
                v[i] = d[1][j]
        if np.isfinite(v).any():
            lo, hi = np.nanquantile(v, clip)
            v = np.where(np.isfinite(v), np.clip(v, lo, hi), np.nan)
        filled, miss = _fill(v)
        cols += [filled, miss]
        colnames += [f"lab_{lab}", f"lab_{lab}_missing"]
        nmeas += 1 - miss
    return FeatureBlock("labs", np.column_stack(cols + [nmeas]).astype(np.float32), colnames + ["n_labs_measured"])


def make_stay_blocks(nodes: list, person=None, visits=None, labs=None, lab_names=None, **kw) -> dict:
    """The blocks that do not depend on the diagnosis source; compute once per task and pass as
    `extra_blocks` to `make_blocks` / `build_pyg_data`. `kw`: within, ed_window, days, lag, window."""
    out = {}
    if person is not None:
        out["demo"] = demo_block(nodes, person)
    if visits is not None:
        out["admission"] = admission_block(nodes, visits, **{k: kw[k] for k in ("within", "ed_window") if k in kw})
        out["prior_use"] = prior_use_block(nodes, visits, **{k: kw[k] for k in ("days", "lag") if k in kw})
    if labs is not None and lab_names:
        out["labs"] = labs_block(nodes, labs, lab_names, **{k: kw[k] for k in ("window",) if k in kw})
    return out
