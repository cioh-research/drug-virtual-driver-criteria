"""Evidence synthesis: derive state-wise relative effects (delta) of ocular features from the verified
evidence files (evidence/*.json), with full provenance.  Every number is read from the evidence JSON
(values copied from the opened sources by the evidence-extraction step); manual entries are limited to
(i) unit conversions stated in the source and (ii) percentage changes quoted verbatim, and are flagged.

Output: results/params.json
  features, alert baseline (mean, SD, source), per state: delta_E (evidence-quantified), delta_D
  (direction-informed sensitivity), provenance list, k studies, total n, range of reported deltas.
"""
import json, os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVID = os.path.join(BASE, "evidence")
RESULTS_DIR = os.path.join(BASE, "results"); os.makedirs(RESULTS_DIR, exist_ok=True)
E = {s: json.load(open(os.path.join(EVID, f"{s}.json"), encoding="utf-8")) for s in ("drowsiness", "alcohol", "opioid", "cannabis")}

FEATS = ["P", "BD", "SL", "PV", "G", "FD", "D", "A", "V"]
NAMES = {"P": "Eye closure (PERCLOS / %TEC)", "BD": "Blink duration", "SL": "Saccade latency", "PV": "Saccade peak velocity",
         "G": "Smooth-pursuit gain", "FD": "Fixation duration", "D": "Pupil diameter", "A": "PLR constriction amplitude (relative)",
         "V": "PLR constriction velocity"}
UNITS = {"P": "%", "BD": "ms", "SL": "ms", "PV": "deg/s", "G": "", "FD": "ms", "D": "mm", "A": "%", "V": "mm/s"}


def m(state, sid, metric, idx=0):
    st = next(s for s in E[state]["studies"] if s["id"] == sid)
    ms = [x for x in st["measures"] if x["metric"] == metric]
    return st, ms[idx]


def rel(state, sid, metric, idx=0, n=None, manual_pct=None, note="", base=None, imp=None):
    """relative change of one study arm (impaired vs baseline)."""
    st, x = m(state, sid, metric, idx)
    if manual_pct is not None:
        d = manual_pct / 100.0; how = "reported % change"
    elif x.get("baseline_mean") is not None and x.get("impaired_mean") is not None:
        d = (x["impaired_mean"] - x["baseline_mean"]) / x["baseline_mean"]; how = "means"
    elif x.get("pct_change") is not None:
        d = x["pct_change"] / 100.0; how = "reported % change"
    else:
        raise ValueError(f"no numeric value {sid} {metric}")
    return dict(study=sid, n=n if n is not None else st.get("n"), delta=float(d), how=how, metric=metric,
                base=x.get("baseline_mean") if base is None else base, imp=x.get("impaired_mean") if imp is None else imp, units=x.get("units"), note=note,
                title=st.get("title"), year=st.get("year"))


def relv(state, sid, metric, sel="", n=None, manual_pct=None, note=""):
    """like rel(), but selects the measure whose units/evidence text contains `sel` (round-2 additions)."""
    st = next(x for x in E[state]["studies"] if x["id"] == sid)
    ms = [x for x in st["measures"] if x["metric"] == metric and (sel.lower() in (x.get("units") or "").lower() + " " + (x.get("evidence_text") or "").lower())]
    num = [x for x in ms if (x.get("baseline_mean") is not None and x.get("impaired_mean") is not None) or x.get("pct_change") is not None]
    ms = num if (num and manual_pct is None) else ms
    if not ms:
        raise ValueError(f"no measure {sid} {metric} {sel}")
    idx = st["measures"].index(ms[0]); same = [x for x in st["measures"] if x["metric"] == metric]
    return rel(state, sid, metric, same.index(ms[0]), n=n, manual_pct=manual_pct, note=note)


def manual(sid, n, delta, metric, how, note="", base=None, imp=None, units=None, year=None):
    """entry computed from values quoted in the evidence file (formula stated in `how`)."""
    return dict(study=sid, n=n, delta=float(delta), how=how, metric=metric, base=base, imp=imp, units=units, note=note, title=None, year=year)


def direc(state, sid, metric, idx=0):
    st, x = m(state, sid, metric, idx)
    return dict(study=sid, n=st.get("n"), direction=x["direction"], metric=metric)


# ---------------------------------------------------------------- curated evidence map
Q = {  # quantified effects (relative change); state -> feature -> list
    "DR": {
        "BD": [rel("drowsiness", "Shiferaw2018", "blink_duration_ms"), rel("drowsiness", "Lee2016", "blink_duration_ms", manual_pct=25.0, base=100, imp=125, note="abstract 100 vs 125 ms (P < 0.05); Table 2 reports 0.13 vs 0.18 s (n.s.)"),
               rel("drowsiness", "Watling2013", "blink_duration_ms", note="time-on-task, morning drive")],
        "SL": [relv("drowsiness", "Meyhofer2017", "saccade_latency_ms", "prosaccade latency", n=18, note="24 h TSD, control group")],
        "PV": [rel("drowsiness", "Stone2019", "saccade_peak_velocity_deg_s", manual_pct=-20, note="source: decreased '10 to 30%' after 20-25 h awake; midpoint used"),
               relv("drowsiness", "Meyhofer2017", "saccade_peak_velocity_deg_s", "prosaccade peak", n=18, note="24 h TSD, control group"),
               manual("Muehl2018", 24, (-38.8 - (-5.1)) / 352.0, "saccade_peak_velocity_deg_s", "difference of change scores (-38.8 vs -5.1 deg/s) relative to the normative alert value 352 deg/s",
                      note="conference abstract; change scores", year=2018)],
        "G": [rel("drowsiness", "Stone2019", "smooth_pursuit_gain", note="~-20% peak at ~23 h awake"),
              relv("drowsiness", "Meyhofer2017", "smooth_pursuit_gain", "step-ramp", n=18, note="step-ramp gain; sinusoidal gain unchanged")],
        "FD": [rel("drowsiness", "Shiferaw2018", "fixation_duration_ms", note="n.s.")],
        "A": [rel("drowsiness", "Rozanowski2015", "plr_amplitude_mm_or_pct")],
        "V": [rel("drowsiness", "Kaifie2021", "plr_constriction_velocity_mm_s", note="device units"), rel("drowsiness", "Rozanowski2015", "plr_constriction_velocity_mm_s")],
    },
    "AC": {
        # Cori et al. (2023), Table 1 (full text): Optalert %TEC and blink duration, model-estimated means, baseline vs BAC 0.08%.
        # %TEC has a near-zero baseline (same device as Anderson 2013), so the ABSOLUTE difference (+0.36 percentage points) is
        # added to the alert reference (1.99%), as for drowsiness; it is expressed here relative to that reference.
        "P": [manual("Cori2023", 12, (0.55 - 0.19) / 1.99, "perclos_pct", "%TEC 0.19 -> 0.55 (baseline vs BAC 0.08%); absolute difference +0.36 points added to the alert reference 1.99%",
                     base=0.19, imp=0.55, units="% time with eyes closed", note="Optalert; BAC 0.08%; absolute difference applied", year=2023)],
        "BD": [manual("Cori2023", 12, (0.37 - 0.29) / 0.29, "blink_duration_ms", "0.29 -> 0.37 s (baseline vs BAC 0.08%), model-estimated means",
                      base=290, imp=370, units="ms (reported in s)", note="Optalert; BAC 0.08%", year=2023),
               # Zemblys et al. (2024), Table 8 (Manual setup, LMM): blink duration intercept 87.47 ms, slope +31.44 ms per 1 permille BrAC; evaluated at 0.8 permille (= BAC 0.08%) for comparability with Cori
               manual("Zemblys2024", 35, (31.44 * 0.8) / 87.47, "blink_duration_ms", "LMM: 87.47 ms + 31.44 ms per 1 permille BrAC (Manual setup); evaluated at 0.8 permille",
                      base=87.47, imp=87.47 + 31.44 * 0.8, units="ms", note="test-track driving; 5-camera Smart Eye", year=2024)],
        "SL": [rel("alcohol", "Vorstius2008", "saccade_latency_ms", 0, note="pro-saccades, BrAC ~0.07%"),
               relv("alcohol", "Ojanpera2023", "saccade_latency_ms", "pro-saccade", n=9, note="BrAC 1.0-1.2 per mille"),
               relv("alcohol", "Zoethout2009", "saccade_latency_ms", "", n=12, note="IV clamp 0.6 g/L; n.s.")],
        "PV": [rel("alcohol", "Lehtinen1979", "saccade_peak_velocity_deg_s", note="BAC 0.056-0.116%"),
               relv("alcohol", "Zoethout2009", "saccade_peak_velocity_deg_s", "", n=12, note="IV clamp 0.6 g/L; n.s.")],
        "G": [rel("alcohol", "Didier2025", "smooth_pursuit_gain", n=62, note="light drinkers (n = 62), BrAC ~0.09%"),
              relv("alcohol", "Ojanpera2023", "smooth_pursuit_gain", "", n=9, note="BrAC 1.0-1.2 per mille"),
              relv("alcohol", "Tyson2021", "smooth_pursuit_gain", "steady-state", n=16, manual_pct=-22, note="-22% at 0.055% BAC"),
              relv("alcohol", "Zoethout2009", "smooth_pursuit_gain", "", n=12, note="proxy: % time in smooth pursuit")],
        "FD": [rel("alcohol", "Silva2017", "fixation_duration_ms"), rel("alcohol", "Ziedman1980", "fixation_duration_ms", 1, note="roadway dwell, BAC ~0.084%"),
               # Zemblys et al. (2024), Table 7 (Manual setup, LMM): fixation duration intercept 486.28 ms, slope +334.22 ms per 1 permille; evaluated at 0.8 permille (same rule as blink duration).
               # Fixations closer than 400 ms / 2 deg were merged by the authors, so the measure is closer to on-road fixation/dwell than to laboratory fixation.
               manual("Zemblys2024", 35, (334.22 * 0.8) / 486.28, "fixation_duration_ms", "LMM: 486.28 ms + 334.22 ms per 1 permille BrAC (Manual setup); evaluated at 0.8 permille",
                      base=486.28, imp=486.28 + 334.22 * 0.8, units="ms", note="test-track driving; merged fixations (< 400 ms, < 2 deg)", year=2024)],
        "D": [rel("alcohol", "Didier2025", "pupil_diameter_mm", manual_pct=(1.175 / 1.086 - 1) * 100, note="ratio of normalised means: alcohol +17.5% vs placebo +8.6% (1.175/1.086 - 1)"),
              rel("alcohol", "Silva2017", "pupil_diameter_mm", note="n.s."),
              relv("alcohol", "LobatoRincon2013", "pupil_diameter_mm", "basal", n=19, note="BrAC 0.49 mg/L")],
        # Lobato-Rincon et al. (2013), Table 2: four stimuli (white 5600 K, blue, green, red). Rule: where several lighting/stimulus
        # conditions are reported, the one closest to daytime broadband conditions is used -> white light (red light: -16.9%).
        "A": [manual("LobatoRincon2013", 19, (1.96 / 6.24) / (1.88 / 5.88) - 1, "plr_amplitude_mm_or_pct",
                     "white-light constriction amplitude 1.88 -> 1.96 mm divided by basal diameter 5.88 -> 6.24 mm (relative amplitude)", base=1.88, imp=1.96,
                     note="white light (5600 K); n.s. (p = 0.39); red light alone gives -16.9%", year=2013)],
        "V": [rel("alcohol", "Kaifie2021", "plr_constriction_velocity_mm_s", note="n.s.; device units")],
    },
    "OP": {
        "D": [rel("opioid", "Knaggs2004", "pupil_diameter_mm", 0, note="morphine/codeine, analgesic dose"),
              rel("opioid", "Rollins2014", "pupil_diameter_mm", note="high-dose remifentanil"),
              rel("opioid", "McKay2021", "pupil_diameter_mm"), rel("opioid", "McKay2022", "pupil_diameter_mm"),
              rel("opioid", "McKay2026", "pupil_diameter_mm", note="reported as >48% (lower bound)"),
              rel("opioid", "Kongsgaard2019", "pupil_diameter_mm", n=56, note="chronic opioid patients (dose increase, n=36) vs volunteers (n=20)"),
              manual("Meissner2023", 42, (-0.672 - 0.417) / 2, "pupil_diameter_mm", "mean of -67.2% (hydromorphone, 6.4 -> 2.1 mm) and -41.7% (morphine, 7.2 -> 4.2 mm), dark-adapted",
                     note="IV infusion, paired crossover", year=2023)],
        "PV": [relv("opioid", "Feit2014", "saccade_peak_velocity_deg_s", "prosaccade (latency task)", n=46, note="methadone maintenance (HIV-negative subgroup), pre- vs post-dose (tolerant)")],
        "A": [manual("Rollins2014", 10, (0.4 / 2.5) / (2.5 / 5.6) - 1, "plr_amplitude_mm_or_pct",
                     "absolute PLR amplitude divided by the pupil diameter of the same condition (rule ii): 2.5/5.6 mm -> 0.4/2.5 mm; ratio of group means (approximation)",
                     base=100 * 2.5 / 5.6, imp=100 * 0.4 / 2.5, units="% (derived)", note="high-dose remifentanil; ratio of group means", year=2014),
              rel("opioid", "McKay2021", "plr_amplitude_mm_or_pct"),
              rel("opioid", "Kongsgaard2019", "plr_amplitude_mm_or_pct", n=56, note="cross-sectional")],
        "V": [rel("opioid", "McKay2021", "plr_constriction_velocity_mm_s"), rel("opioid", "Kongsgaard2019", "plr_constriction_velocity_mm_s", n=54, note="cross-sectional; velocity endpoint n = 34 + 20")],
    },
    "CB": {
        "SL": [dict(study="Zuurman2008", n=12, delta=20.0 / 177.2, how="THC-placebo difference +20 ms (95% CI 10-30) relative to normative latency 177.2 ms",
                    metric="saccade_latency_ms", base=None, imp=None, units="ms", note="vaporised THC", title=None, year=2008),
               relv("cannabis", "Kleinloog2012", "saccade_latency_ms", "", n=49, note="vaporised THC; n.s.")],
        "PV": [dict(study="Zuurman2008", n=12, delta=14.0 / 352.0, how="THC-placebo difference +14 deg/s (95% CI -4 to 32) relative to normative peak velocity 352 deg/s; nonsignificant point estimate retained",
                    metric="saccade_peak_velocity_deg_s", base=None, imp=None, units="deg/s", note="n.s.", title=None, year=2008),
               relv("cannabis", "Kleinloog2012", "saccade_peak_velocity_deg_s", "", n=49, note="vaporised THC; n.s.")],
        "G": [relv("cannabis", "Kleinloog2012", "other", "smooth pursuit", n=49, note="proxy: % time in smooth pursuit; n.s.")],
        "BD": [relv("cannabis", "Manning2023", "blink_duration_ms", "mean over drive", n=14, note="CBD 100 mg + THC 5 mg sublingual")],
        "FD": [relv("cannabis", "Manning2023", "fixation_duration_ms", "40-min", n=14, note="simulated drive")],
        "D": [rel("cannabis", "KleineBrueggeney2025", "pupil_diameter_mm", n=10, note="IV THC; THC arm, 20-min endpoint n = 10"), rel("cannabis", "OrtizPeregrina2020", "pupil_diameter_mm", 1, note="low mesopic; n.s."),
              relv("cannabis", "Marcotte2023", "pupil_diameter_mm", "median", n=184, note="medians; parallel groups; p = 0.06")],
        "A": [rel("cannabis", "KleineBrueggeney2025", "plr_amplitude_mm_or_pct", 0, n=10, note="THC arm, 20-min endpoint n = 10"), rel("cannabis", "Campobasso2020", "plr_amplitude_mm_or_pct", n=14),
              manual("Godbole2024", 84, ((20.4 / 33.8) / (23.7 / 29.6) - 1 + (20.1 / 32.9) / (23.7 / 29.6) - 1) / 2, "plr_amplitude_mm_or_pct",
                     "% light reflex 33.8 -> 20.4 (occasional) and 32.9 -> 20.1 (daily), each divided by the no-use group change 29.6 -> 23.7; mean of both",
                     note="same-cohort preprint Table 2", year=2024)],
        "V": [rel("cannabis", "KleineBrueggeney2025", "plr_constriction_velocity_mm_s", n=8, note="n.s.; THC arm, 20-min endpoint n = 8; 40-80% constriction-interval slope"), rel("cannabis", "Campobasso2020", "plr_constriction_velocity_mm_s", n=14)],
    },
}
# eye-closure (PERCLOS-type) for drowsiness is modelled on an absolute scale (Anderson 2013, %TEC; SE converted to SD)
P_DR = dict(study="Anderson2013", n=28, base=1.99, imp=6.34, sd_base=0.25 * np.sqrt(28), sd_imp=0.97 * np.sqrt(28),  # 28 of 29 participants analysed
            how="means; SD converted from reported SE (SD = SE*sqrt(n))", note="Optalert %TEC (time with eyes fully closed) after 30 h awake")

D_ONLY = {  # direction-only evidence (no extractable magnitude)
    "DR": {"D": [direc("drowsiness", "Russo2003", "pupil_diameter_mm"), direc("drowsiness", "Daguet2019", "pupil_diameter_mm"), direc("drowsiness", "Morad2000", "pupil_diameter_mm"),
                 direc("drowsiness", "VanEgroo2019", "pupil_diameter_mm")],
           "SL": [direc("drowsiness", "Collet2020", "saccade_latency_ms")],
           "PV": [direc("drowsiness", "Zils2005", "saccade_peak_velocity_deg_s"), direc("drowsiness", "Fransson2008SD", "saccade_peak_velocity_deg_s"), direc("drowsiness", "Russo2003", "saccade_peak_velocity_deg_s"),
                  direc("drowsiness", "DiStasi2014", "saccade_peak_velocity_deg_s"), direc("drowsiness", "Caldwell2003AFRL", "saccade_peak_velocity_deg_s"), direc("drowsiness", "Hirvonen2010", "saccade_peak_velocity_deg_s")],
           "G": [direc("drowsiness", "Fransson2008SD", "smooth_pursuit_gain"), direc("drowsiness", "Tyson2023", "smooth_pursuit_gain")],
           "FD": [direc("drowsiness", "Muehl2018", "fixation_duration_ms"), direc("drowsiness", "Winkler2017", "fixation_duration_ms")],
           "P": [direc("drowsiness", "Dinges1998", "perclos_pct"), direc("drowsiness", "Jackson2016TIP", "perclos_pct"), direc("drowsiness", "Cellini2023", "perclos_pct"),
                 direc("drowsiness", "Jackson2016AAP", "eyelid_closure_other")],
           "BD": [direc("drowsiness", "Ingre2006", "blink_duration_ms"), direc("drowsiness", "Schleicher2008", "blink_duration_ms"), direc("drowsiness", "ShekariSoleimanloo2019", "blink_duration_ms"),
                  direc("drowsiness", "Palmer1994", "blink_duration_ms"), direc("drowsiness", "Anund2009", "blink_duration_ms")]},
    "AC": {"P": [direc("alcohol", "Cori2023", "eyelid_closure_other"), direc("alcohol", "Beideman1977", "eyelid_closure_other")], "SL": [direc("alcohol", "Didier2025", "saccade_latency_ms"), direc("alcohol", "Roche2014", "saccade_latency_ms"), direc("alcohol", "Fransson2010", "saccade_latency_ms"),
                  direc("alcohol", "Dumont2010", "saccade_latency_ms"), direc("alcohol", "Wegner1999", "saccade_latency_ms")],
           "PV": [direc("alcohol", "Didier2025", "saccade_peak_velocity_deg_s"), direc("alcohol", "Fransson2010", "saccade_peak_velocity_deg_s"), direc("alcohol", "Tyson2021", "saccade_peak_velocity_deg_s"),
                  direc("alcohol", "Dumont2010", "saccade_peak_velocity_deg_s"), direc("alcohol", "Zoethout2012ch5", "saccade_peak_velocity_deg_s")],
           "G": [direc("alcohol", "Roche2014", "smooth_pursuit_gain"), direc("alcohol", "Fransson2010", "smooth_pursuit_gain"), direc("alcohol", "Dumont2010", "smooth_pursuit_gain")],
           "FD": [direc("alcohol", "Koch2023", "fixation_duration_ms"), direc("alcohol", "Deuber2025", "fixation_duration_ms")]},
    "OP": {"PV": [direc("opioid", "Feit2013", "saccade_peak_velocity_deg_s"), direc("opioid", "FeitEPA1229_2014", "saccade_peak_velocity_deg_s")],
           "G": [direc("opioid", "Rottach2002", "smooth_pursuit_gain")],
           "P": [direc("opioid", "PorathWaller2021", "eyelid_closure_other"), direc("opioid", "Porath2019", "eyelid_closure_other")]},
    "CB": {"SL": [direc("cannabis", "Ploner2002", "saccade_latency_ms")], "G": [direc("cannabis", "LiemMoolenaar2010", "other")]},
}

# ---------------------------------------------------------------- quality appraisal (applied uniformly, independent of effect direction)
# Core-set rules: published 2000 or later; peer-reviewed journal article or full peer-reviewed conference paper;
# controlled design with n >= 8 (verifiable); instrumented ocular measurement; adult population;
# one report per sample (re-analyses, abstracts of the same study and reports whose sample overlap cannot be excluded
# are reduced to the most complete report).
APPRAISAL = {
    "Lehtinen1979": "published before 2000", "Ziedman1980": "published before 2000 (technical report)",
    "Beideman1977": "published before 2000", "Wegner1999": "published before 2000", "Palmer1994": "published before 2000 (technical report)",
    "Dinges1998": "published before 2000 (technical report)",
    "Watling2013": "conference paper, not peer reviewed", "Muehl2018": "conference abstract", "Winkler2017": "conference abstract",
    "Zoethout2012ch5": "thesis chapter (same programme as Zoethout et al., 2009)", "Caldwell2003AFRL": "technical report, not peer reviewed",
    "Feit2013": "conference abstract (same programme as Feit et al., 2014)", "FeitEPA1229_2014": "conference abstract (same programme as Feit et al., 2014)",
    "Walecki2016": "conference abstract (same programme as Feit et al., 2014)",
    "Rottach2002": "n < 8", "DiStasi2014": "sample size not verifiable on the opened record",
    "PorathWaller2021": "examiner ratings, not instrumented (technical report)", "Porath2019": "examiner ratings, not instrumented",
    "Tyson2023": "re-analysis of Stone et al. (2019) data", "Jackson2016AAP": "same group, protocol and year as Jackson et al. (2016, TIP); overlap not excludable",
    "ShekariSoleimanloo2019": "same laboratory on-road sleep-deprivation protocol as Shiferaw et al. (2018); overlap not excludable",
    "McKay2022": "same laboratory and remifentanil protocol as McKay & Larson (2021); overlap not excludable",
    "McKay2026": "same laboratory and remifentanil protocol as McKay & Larson (2021); overlap not excludable",
    "Tyson2021": "reported values not verifiable against the published article",
    "Marcotte2023": "examiner ratings, not instrumented (officer-estimated pupil size during field sobriety testing, 0.5-mm steps)",
}
EVIDENCE_SET = os.environ.get("EVIDENCE_SET", "core")   # "core" (main analysis) or "full" (sensitivity)
if EVIDENCE_SET == "core":
    for _m in (Q, D_ONLY):
        for _s in _m:
            for _f in list(_m[_s]):
                _m[_s][_f] = [x for x in _m[_s][_f] if x["study"] not in APPRAISAL]
    # Godbole et al. (2024): the peer-reviewed article reports the direction only; magnitudes exist only in a
    # same-cohort preprint, so in the core set the study contributes direction, not magnitude.
    Q["CB"]["A"] = [x for x in Q["CB"]["A"] if x["study"] != "Godbole2024"]
    D_ONLY["CB"].setdefault("A", []).append(direc("cannabis", "Godbole2024", "plr_amplitude_mm_or_pct"))
    # Direction-only reports enter the computation only where no quantified study exists (scenario D).  Where a cell is
    # already quantified they only confirm an established direction; in the core set they are not counted as evidence
    # (well-established eyelid findings for drowsiness are cited through the review by Abe, 2023).
    SUPPORTING = {}
    for _s in D_ONLY:
        for _f in list(D_ONLY[_s]):
            if Q.get(_s, {}).get(_f) or (_s == "DR" and _f == "P"):
                SUPPORTING.setdefault(_s, {})[_f] = [x["study"] for x in D_ONLY[_s][_f]]
                D_ONLY[_s][_f] = []
else:
    SUPPORTING = {}


# ---------------------------------------------------------------- alert-state normative baselines
def pooled(arms):
    n = np.array([a[0] for a in arms], float); mu = np.array([a[1] for a in arms]); sd = np.array([a[2] for a in arms])
    mbar = (n * mu).sum() / n.sum()
    # within-study pooled SD: between-subject variability under a single protocol (between-study
    # differences reflect stimulus/device protocols and are excluded)
    var = ((n - 1) * sd ** 2).sum() / (n - 1).sum()
    return float(mbar), float(np.sqrt(var))

ALERT = {
    "P": (1.99, float(0.25 * np.sqrt(28)), "Anderson et al. (2013), %TEC first 16 h awake (SE converted to SD)"),
    "BD": (179.88, 24.89, "Shiferaw et al. (2018), rested on-road drive"),
    "SL": (177.2, 18.52, "Bargary et al. (2017), n = 1058"),
    "PV": (352.0, 50.0, "Hopf et al. (2018), 15-deg saccades, n = 100"),
    "G": (0.795, 0.138, "Bargary et al. (2017), n = 1058"),
    "FD": (561.38, 108.02, "Shiferaw et al. (2018), rested on-road drive"),
    "D": (4.03, 0.90, "Schröder et al. (2018), photopic 16 lx, n = 91"),
}
A_arms = [(14, 23.06, 4.83), (20, 40.2, 10.8), (18, 37.35, 4.81), (14, 23.5, 6.7), (10, 24.21, 5.11)]  # Kleine-B. THC baseline n = 14  # Campobasso, Kongsgaard, McKay2021, Kleine-B., Rozanowski
V_arms = [(14, 3.96, 0.50), (20, 5.12, 1.09), (18, 3.30, 0.51), (14, 4.00, 0.84), (10, 3.98, 0.82)]
ALERT["A"] = (*pooled(A_arms), "n-weighted mean and within-study pooled SD of baseline arms of Campobasso 2020, Kongsgaard 2019, McKay 2021, Kleine-Brueggeney 2025, Różanowski 2015")
ALERT["V"] = (*pooled(V_arms), "n-weighted mean and within-study pooled SD of baseline arms of Campobasso 2020, Kongsgaard 2019, McKay 2021, Kleine-Brueggeney 2025, Różanowski 2015")


EXCLUDE_PROXY = os.environ.get("EXCLUDE_PROXY", "0") == "1"   # sensitivity: drop pursuit-time-fraction proxies of pursuit gain
if EXCLUDE_PROXY:
    for _s in Q.values():
        if "G" in _s:
            _s["G"] = [x for x in _s["G"] if "proxy" not in (x.get("note") or "")]


def summarise(lst):
    n = np.array([x["n"] or 1 for x in lst], float); d = np.array([x["delta"] for x in lst])
    return dict(delta=float((n * d).sum() / n.sum()), k=len(lst), n=int(n.sum()), lo=float(d.min()), hi=float(d.max()))


STATES = ["AL", "DR", "AC", "OP", "CB"]
out = {"features": FEATS, "names": NAMES, "units": UNITS,
       "alert": {f: dict(mean=ALERT[f][0], sd=ALERT[f][1], source=ALERT[f][2]) for f in FEATS}, "states": {}}
for s in STATES[1:]:
    rec = {}
    for f in FEATS:
        q = Q[s].get(f, []); dl = D_ONLY[s].get(f, [])
        cell = dict(quant=q, dir_only=dl)
        if s == "DR" and f == "P":
            cell.update(delta=(P_DR["imp"] - P_DR["base"]) / P_DR["base"], k=1, n=P_DR["n"], lo=None, hi=None, absolute=P_DR)
        elif q:
            cell.update(summarise(q))
        else:
            cell.update(delta=0.0, k=0, n=0, lo=None, hi=None)
        dirs = [x["direction"] for x in dl]
        sign = 0
        if dirs and all(x == "increase" for x in dirs): sign = 1
        if dirs and all(x == "decrease" for x in dirs): sign = -1
        cell["dir_sign"] = sign
        rec[f] = cell
    out["states"][s] = rec


# Sensitivity scenario W (weak opioid effect): the pooled pupil effect (mostly IV / high-dose evidence) is scaled so that
# the pupil-diameter change is -14.1% (about 0.57 mm on a 4.03 mm alert pupil), at the lower end of the 0.58-1.83 mm
# maximum constriction reported after oral tramadol 100 mg across CYP2D6 genotypes (Slanar et al., 2007).
W_TARGET = -0.141


def archetypes(scenario="E"):
    mu, sd = {}, {}
    mu["AL"] = np.array([ALERT[f][0] for f in FEATS]); sd["AL"] = np.array([ALERT[f][1] for f in FEATS])
    cv = sd["AL"] / mu["AL"]
    for s in STATES[1:]:
        mm, ss = [], []
        for i, f in enumerate(FEATS):
            c = out["states"][s][f]
            if s == "DR" and f == "P":
                mm.append(P_DR["imp"]); ss.append(P_DR["sd_imp"]); continue
            d = c["delta"]
            if scenario == "W" and s == "OP" and f in ("D", "A", "V"):
                d = d * W_SCALE  # oral/therapeutic-dose sensitivity (see W_SCALE)
            if scenario == "D" and c["k"] == 0 and c["dir_sign"] != 0:
                d = 0.5 * cv[i] * c["dir_sign"]  # half a between-subject SD in the reported direction
            mm.append(mu["AL"][i] * (1 + d)); ss.append(cv[i] * mu["AL"][i] * (1 + d))
        mu[s] = np.array(mm); sd[s] = np.array(ss)
    return mu, sd


out["evidence_set"] = EVIDENCE_SET
out["supporting_not_counted"] = SUPPORTING
out["excluded_by_appraisal"] = APPRAISAL if EVIDENCE_SET == "core" else {}
_ids = set()
for _s in out["states"].values():
    for _c in _s.values():
        _ids |= {x["study"] for x in _c["quant"]} | {x["study"] for x in _c["dir_only"]}
_ids |= {"Anderson2013", "Shiferaw2018", "NORM_Bargary2017", "NORM_Hopf2018", "NORM_Schroder2018", "Campobasso2020", "Kongsgaard2019",
         "McKay2021", "KleineBrueggeney2025", "Rozanowski2015"}
out["contributing_ids"] = sorted(_ids)
W_SCALE = W_TARGET / out["states"]["OP"]["D"]["delta"]
out["W_scale"] = W_SCALE
for sc in ("E", "D", "W"):
    mu, sd = archetypes(sc)
    out[f"MU_{sc}"] = {s: mu[s].tolist() for s in STATES}; out[f"SD_{sc}"] = {s: sd[s].tolist() for s in STATES}
json.dump(out, open(os.path.join(RESULTS_DIR, os.environ.get("PARAMS_NAME", "params.json")), "w", encoding="utf-8"), indent=1, default=float)

if __name__ == "__main__":
    print("alert:", {f: (round(ALERT[f][0], 3), round(ALERT[f][1], 3)) for f in FEATS})
    for s in STATES[1:]:
        print(s, {f: (f"{100 * out['states'][s][f]['delta']:+.1f}%", out['states'][s][f]['k'], out['states'][s][f]['dir_sign']) for f in FEATS})
    for sc in ("E", "D"):
        print(sc, {s: [round(v, 3) for v in out[f'MU_{sc}'][s]] for s in STATES})
