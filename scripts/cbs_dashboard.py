#!/usr/bin/env python3
"""CBS (Cystathionine Beta-Synthase Deficiency / Classical Homocystinuria) Epilepsy Dashboard.

CBS encodes Cystathionine Beta-Synthase, a 551-aa cytoplasmic PLP-dependent enzyme at 21q22.3
that catalyzes the FIRST STEP of the transsulfuration pathway:
  Homocysteine + L-Serine → Cystathionine  (irreversible, PLP/B6-dependent)
This is the ONLY route for homocysteine to exit the remethylation cycle into transsulfuration.

TRANSSULFURATION PATHWAY — CBS IS THE GATEWAY ENZYME:

  Methionine → SAM (via MAT) → SAH → Homocysteine (Hcy)
                                              │
                         ┌────────────────────┤
                         │  Remethylation:    │  Transsulfuration:
                         │  MTR (MeCbl)       │  CBS (PLP/B6)  ← GATEWAY
                         │  Hcy → Methionine  │  Hcy + Serine → Cystathionine
                         │  (MTHFR provides   │       │
                         │   5-methylTHF)     │   CTH (cystathionase)
                         └────────────────────┘       ↓
                                                  Cysteine + α-Ketobutyrate
                                                  ↓
                                           Taurine, Glutathione, SO4²⁻

CBS LOF RESULT — HHcy VERY HIGH + METHIONINE ELEVATED (KEY DIFFERENCE FROM ALL REMETHYLATION DEFECTS):
  → Hcy CANNOT enter transsulfuration → Hcy trapped → tHcy 100–500 µmol/L (HIGHEST OF ALL HHcy disorders)
  → Methionine ACCUMULATES (HIGH 60–500 µmol/L): catabolic (transsulfuration) route blocked.
    KEY: In cblE/cblG/MTHFR, methionine is LOW (remethylation to methionine is impaired).
         In CBS deficiency, methionine is HIGH (remethylation produces methionine normally but
         transsulfuration to clear it is blocked → methionine piles up).
  → Cystathionine ABSENT/undetectable: CBS cannot make it → pathognomonic negative.
  → Cysteine LOW / conditionally essential: cystathionine → cysteine via CTH; with CBS blocked,
    cysteine must come entirely from diet.
  → MMA NORMAL: propionyl-CoA / AdoCbl arm entirely intact; MMA axis independent of CBS.
  → MeCbl NORMAL: cobalamin processing (LMBRD1, ABCD4, MMACHC, MMADHC, MMAB, MTRR, MTR) intact.
  → AdoCbl NORMAL: same.
  → C3 NORMAL: no C3 elevation → traditional NBS amino-acid panel may detect METHIONINE elevation
    (some panels also now measure total homocysteine; methionine >200 µmol/L triggers recall).
  → Serum folate NORMAL or SLIGHTLY LOW: remethylation (MTHFR → MTR) is INTACT; no methylfolate
    trap (unlike cblE/cblG where serum folate is HIGH, and unlike MTHFR where it is LOW).
    Betaine supplementation uses BHMT pathway (also generates methionine from Hcy).
  → Megaloblastic anemia ABSENT: no methylfolate trap; MTHFR and MTR functioning normally.
  → Ectopia lentis (lens dislocation): PATHOGNOMONIC for CBS — homocysteine cross-links fibrillin
    in the lens zonules → dislocation. 90% by age 10 (untreated). NOT seen in cblE/cblG/MTHFR.
  → Marfanoid habitus: homocysteine disrupts fibrillin-1 cross-linking → connective tissue disease.
  → Thromboembolism: HIGHEST RISK of all HHcy disorders — HHcy damages vascular endothelium,
    activates platelets, promotes coagulation. 50–60% risk by age 30 untreated.
  → Intellectual disability (IDD): 60–65%; more severe in B6-non-responsive.
  → Psychiatric: 50–60% schizophrenia-like (mechanism: HHcy → N-methyl-D-aspartate receptor).
  → Epilepsy: 20–30% (less prominent than in cobalamin disorders; mechanism: thrombotic strokes +
    HHcy-mediated neuronal excitotoxicity; seizures often secondary to cerebrovascular events).
  → Osteoporosis / skeletal: CBS makes cystathionine → cysteine → glutathione; glutathione deficiency
    → oxidative bone damage; also HHcy inhibits lysyl oxidase → impaired collagen cross-linking.

B6 (PYRIDOXINE) RESPONSIVENESS — THE DEFINING FEATURE OF CBS (UNIQUE AMONG ALL HHcy DISORDERS):
  PLP (pyridoxal-5'-phosphate, active B6) is the catalytic cofactor of CBS.
  B6-responsive CBS (50% of patients):
    → Certain missense alleles (p.Ile278Thr, p.Glu302Lys, p.Asp444Asn) have residual CBS enzyme
      that is thermolabile or has reduced PLP affinity; high-dose B6 saturates the PLP binding
      site → restores partial enzyme activity → HHcy normalizes or drops >50% on B6 alone.
    → Response criterion: tHcy falls >50% on 100–500 mg B6/day.
    → Phenotype: generally milder; ectopia lentis later; IDD less severe.
  B6-non-responsive CBS (40%):
    → Null alleles, catalytic-core missense variants (p.Gly307Ser, p.Arg121Cys, p.Arg369Cys);
      NO residual enzyme activity; B6 cannot rescue → betaine + methionine restriction required.
    → Phenotype: severe; early ectopia lentis; high thrombosis; severe IDD.
  Mild / attenuated CBS (10%):
    → partial activity (5–20% residual); adult-onset mild HHcy; discovered incidentally.

CBS PROTEIN BIOLOGY:
  551 amino acids; 63 kDa monomer; active as tetramer (2x2) or octamer (4x2 dimers).
  N-terminal heme-binding domain (regulatory; unique among PLP enzymes — CBS is the ONLY
    human PLP enzyme with a heme regulatory motif; heme responds to CO, NO, and redox state).
  Catalytic core (AA ~70–400): PLP binding (Lys119 forms Schiff base); substrate binding pockets
    for homocysteine and serine; catalytic lysine conserved.
  C-terminal regulatory domain (AA 400–551): CBS domain (named after this protein);
    SAM allosteric activation site — SAM ACTIVATES CBS (opposite of MTHFR where SAM inhibits).
    When methionine is high → SAM high → CBS activated → more Hcy shunted to transsulfuration.
    This is the homeostatic valve: CBS responds to SAM to increase flux when methionine is high.
  Heme b cofactor: non-covalently bound; b-type heme (Fe²⁺); regulatory, NOT catalytic;
    reduces CBS activity in the oxidized (Fe³⁺) state; CO and NO modulate CBS via heme.

CRITICAL BIOCHEMICAL DIFFERENCES — CBS vs cblE/cblG/MTHFR:
  Feature                    CBS              cblE (MTRR)       cblG (MTR)        MTHFR
  tHcy (µmol/L)              ↑↑↑ 100–500     ↑ 40–200          ↑ 40–200          ↑↑ 50–300
  Methionine                 HIGH ↑↑          LOW               LOW               LOW
  MMA                        NORMAL           NORMAL            NORMAL            NORMAL
  MeCbl fibroblasts          NORMAL           ABSENT            ABSENT            NORMAL
  AdoCbl fibroblasts         NORMAL           NORMAL            NORMAL            NORMAL
  Serum folate               NORMAL           HIGH              HIGH              LOW/NORMAL
  Megaloblastic anemia       ABSENT           90%               80%               Rare
  NBS detection              ~60% (Met↑)      INVISIBLE         INVISIBLE         INVISIBLE
  Ectopia lentis             90% ← UNIQUE     ABSENT            ABSENT            ABSENT
  Marfanoid habitus          80% ← UNIQUE     ABSENT            ABSENT            ABSENT
  Thromboembolism            50–60% ← HIGH    25%               20%               25%
  B6 response                50% ← UNIQUE     None              None              None
  Betaine                    Level A          Level A           Level A           Level A
  Methionine restriction     Level A ← KEY    Not needed        Not needed        Not needed
  B6 (pyridoxine)            Level A (50%)    Not indicated     Not indicated     Not indicated
  Folinic acid               Level C          Level A           Level A           Level B
  Riboflavin                 Level C          Not indicated     Not indicated     Level A

BIOMARKERS (plasma, steady-state, untreated):
  Plasma tHcy: 100–500 µmol/L (B6-responsive: 60–200; B6-NR-severe: 200–500)
  Plasma methionine: 60–500 µmol/L ELEVATED (KEY POSITIVE — different from ALL remethylation defects)
  Urine MMA: NORMAL (<5 mmol/mol Cr)
  Plasma cystathionine: ABSENT / undetectable (CBS cannot make cystathionine from Hcy+serine)
  Plasma cysteine: LOW 15–40 µmol/L (transsulfuration blocked → dietary source only)
  MeCbl fibroblasts: NORMAL (cobalamin system intact)
  AdoCbl fibroblasts: NORMAL
  C3 on NBS: NORMAL (no MMA)
  Plasma methionine on NBS: ELEVATED → detectable by amino acid panel (~60% detected at birth)
  Serum folate: NORMAL (no methylfolate trap)
  CBC/MCV: NORMAL (no megaloblastic anemia)
  Ophthalmology: ectopia lentis on slit-lamp exam — dislocated lens (inferiorly in CBS, superiorly in Marfan)

KEY VARIANTS IN CBS:
  p.Ile278Thr (c.833T>C): MOST COMMON; European/Irish; B6-RESPONSIVE; moderate-severe phenotype;
    thermolabile enzyme; 3–10% residual activity at 37°C but responds to B6/PLP saturation.
  p.Gly307Ser: second most common; B6-NON-RESPONSIVE; severe; null-like; catalytic domain.
  p.Glu302Lys: B6-RESPONSIVE; catalytic domain; residual enzyme rescued by PLP.
  p.Arg369Cys: B6-NON-RESPONSIVE; severe; catalytic core disruption; early thrombosis.
  p.Arg121Cys: B6-NON-RESPONSIVE; severe; heme-binding region disruption; null phenotype.
  p.Asp444Asn: B6-RESPONSIVE; attenuated; C-terminal near CBS/SAM domain; mild HHcy.
  c.IVS11+1G>A: splice donor site; severe null; European.

OMIM: Gene *613381 (CBS) · Disease #236200 (Homocystinuria due to CBS deficiency)
Chromosome: 21q22.3 · Inheritance: AUTOSOMAL RECESSIVE
Prevalence: ~1/200,000–344,000 births (most common cause of classical homocystinuria);
  Ireland highest frequency (~1/65,000 due to p.Ile278Thr founder)
"""

from __future__ import annotations
import random

random.seed(91)   # reproducible synthetic cohort — seed 91 for CBS


# ── helpers ───────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"CBS-{i:03d}"


def _sex(i: int) -> str:
    # Autosomal recessive — both sexes equally affected
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    # B6-responsive classical 50%, B6-NR-severe 40%, mild-attenuated 10%
    r = random.random()
    if r < 0.50:
        return "b6_responsive_classical"
    elif r < 0.90:
        return "b6_nr_severe"
    else:
        return "mild_attenuated"


def _genotype(i: int, pheno: str) -> str:
    b6r_alleles = [
        "p.Ile278Thr / p.Ile278Thr",
        "p.Ile278Thr / p.Glu302Lys",
        "p.Ile278Thr / c.IVS11+1G>A",
        "p.Glu302Lys / p.Glu302Lys",
        "p.Ile278Thr / p.Asp444Asn",
        "p.Asp444Asn / p.Glu302Lys",
    ]
    b6nr_alleles = [
        "p.Gly307Ser / p.Gly307Ser",
        "p.Gly307Ser / p.Arg369Cys",
        "p.Arg121Cys / p.Gly307Ser",
        "p.Arg369Cys / p.Arg121Cys",
        "p.Gly307Ser / c.IVS11+1G>A",
        "p.Arg121Cys / c.IVS11+1G>A",
    ]
    mild_alleles = [
        "p.Asp444Asn / p.Asp444Asn",
        "p.Ile278Thr / p.Asp444Asn",
        "p.Glu302Lys / p.Asp444Asn",
    ]
    if pheno == "b6_responsive_classical":
        return random.choice(b6r_alleles)
    elif pheno == "b6_nr_severe":
        return random.choice(b6nr_alleles)
    else:
        return random.choice(mild_alleles)


def _homocysteine(pheno: str) -> float:
    """tHcy µmol/L — CBS is HIGHEST of all HHcy disorders."""
    if pheno == "b6_responsive_classical":
        return round(random.uniform(60, 220), 1)
    elif pheno == "b6_nr_severe":
        return round(random.uniform(220, 510), 1)
    else:
        return round(random.uniform(30, 80), 1)


def _methionine(pheno: str) -> float:
    """Plasma methionine µmol/L — ELEVATED in CBS (KEY POSITIVE vs LOW in cblE/cblG/MTHFR)."""
    if pheno == "b6_responsive_classical":
        return round(random.uniform(80, 320), 1)
    elif pheno == "b6_nr_severe":
        return round(random.uniform(200, 520), 1)
    else:
        return round(random.uniform(40, 110), 1)


def _mma_urine(pheno: str) -> float:
    """MMA mmol/mol Cr — NORMAL in ALL CBS phenotypes."""
    return round(random.uniform(1.0, 4.5), 1)


def _cysteine(pheno: str) -> float:
    """Plasma cysteine µmol/L — LOW in CBS (transsulfuration blocked)."""
    if pheno == "b6_nr_severe":
        return round(random.uniform(12, 32), 1)
    elif pheno == "b6_responsive_classical":
        return round(random.uniform(18, 42), 1)
    else:
        return round(random.uniform(25, 55), 1)


def _serum_folate(pheno: str) -> float:
    """Serum folate nmol/L — NORMAL in CBS (no methylfolate trap)."""
    return round(random.uniform(12, 32), 1)


def _hemoglobin(pheno: str) -> float:
    """Hemoglobin g/dL — NORMAL (no megaloblastic anemia in CBS)."""
    base = 13.5 if random.random() > 0.5 else 12.5
    return round(base + random.uniform(-1.5, 1.5), 1)


def _c3(pheno: str) -> float:
    """C3 acylcarnitine µmol/L — NORMAL (MMA arm intact)."""
    return round(random.uniform(0.4, 1.2), 2)


def _onset_months(pheno: str) -> int:
    """Age at symptom onset (months)."""
    if pheno == "b6_nr_severe":
        return random.randint(6, 24)      # infantile
    elif pheno == "b6_responsive_classical":
        return random.randint(18, 60)     # toddler/pre-school
    else:
        return random.randint(120, 360)   # adult onset (mild)


def _diag_months(pheno: str, onset: int) -> int:
    """Age at diagnosis (months)."""
    delay = random.randint(3, 36)
    if pheno == "mild_attenuated":
        delay = random.randint(12, 120)
    return onset + delay


def _ectopia_lentis(pheno: str) -> bool:
    """Lens dislocation — PATHOGNOMONIC for CBS."""
    if pheno == "b6_nr_severe":
        return random.random() < 0.92
    elif pheno == "b6_responsive_classical":
        return random.random() < 0.75
    else:
        return random.random() < 0.25


def _marfanoid(pheno: str) -> bool:
    if pheno == "b6_nr_severe":
        return random.random() < 0.88
    elif pheno == "b6_responsive_classical":
        return random.random() < 0.65
    else:
        return random.random() < 0.20


def _thromboembolism(pheno: str) -> bool:
    if pheno == "b6_nr_severe":
        return random.random() < 0.65
    elif pheno == "b6_responsive_classical":
        return random.random() < 0.35
    else:
        return random.random() < 0.10


def _idd(pheno: str) -> bool:
    if pheno == "b6_nr_severe":
        return random.random() < 0.85
    elif pheno == "b6_responsive_classical":
        return random.random() < 0.55
    else:
        return random.random() < 0.10


def _osteoporosis(pheno: str) -> bool:
    if pheno == "b6_nr_severe":
        return random.random() < 0.62
    elif pheno == "b6_responsive_classical":
        return random.random() < 0.42
    else:
        return random.random() < 0.15


def _psychiatric(pheno: str) -> bool:
    if pheno == "b6_nr_severe":
        return random.random() < 0.62
    elif pheno == "b6_responsive_classical":
        return random.random() < 0.45
    else:
        return random.random() < 0.10


def _seizures(pheno: str) -> bool:
    if pheno == "b6_nr_severe":
        return random.random() < 0.38
    elif pheno == "b6_responsive_classical":
        return random.random() < 0.22
    else:
        return random.random() < 0.08


def _nbs(pheno: str) -> bool:
    """NBS detection via methionine elevation."""
    if pheno == "b6_nr_severe":
        return random.random() < 0.78  # methionine very high → usually detected
    elif pheno == "b6_responsive_classical":
        return random.random() < 0.55  # methionine elevated → partially detected
    else:
        return random.random() < 0.20  # mild → often missed


def _b6_responsive(pheno: str) -> bool:
    return pheno == "b6_responsive_classical"


def _aed(sz: bool, pheno: str) -> str:
    if not sz:
        return "none"
    choices = ["levetiracetam", "valproate (AVOID — B6 depletion)", "lamotrigine", "pyridoxine alone", "levetiracetam + pyridoxine"]
    weights = [0.40, 0.05, 0.25, 0.15, 0.15]
    return random.choices(choices, weights=weights)[0]


def _treatment_response(pheno: str) -> str:
    if pheno == "b6_responsive_classical":
        return random.choice(["excellent on B6+betaine", "good on B6+betaine", "partial on B6, complete with betaine added"])
    elif pheno == "b6_nr_severe":
        return random.choice(["good on betaine+met-restriction", "partial on betaine+met-restriction", "moderate on betaine+met-restriction+carnitine"])
    else:
        return random.choice(["excellent on betaine alone", "stable on diet alone"])


def _build_cohort() -> list[dict]:
    patients = []
    for i in range(1, 41):
        pheno = _phenotype(i)
        geno = _genotype(i, pheno)
        onset = _onset_months(pheno)
        sz = _seizures(pheno)
        patients.append({
            "patient_id": _pid(i),
            "sex": _sex(i),
            "phenotype": pheno,
            "genotype": geno,
            "onset_months": onset,
            "diag_months": _diag_months(pheno, onset),
            "homocysteine_umol_l": _homocysteine(pheno),
            "methionine_umol_l": _methionine(pheno),
            "mma_mmol_mol_cr": _mma_urine(pheno),
            "cysteine_umol_l": _cysteine(pheno),
            "serum_folate_nmol_l": _serum_folate(pheno),
            "hemoglobin_g_dl": _hemoglobin(pheno),
            "c3_umol_l": _c3(pheno),
            "mecbl_normal": True,     # ALL CBS patients — cobalamin system intact
            "adocbl_normal": True,    # ALL CBS patients
            "mma_normal": True,       # ALL CBS patients
            "nbs_detected": _nbs(pheno),
            "ectopia_lentis": _ectopia_lentis(pheno),
            "marfanoid": _marfanoid(pheno),
            "thromboembolism": _thromboembolism(pheno),
            "idd": _idd(pheno),
            "osteoporosis": _osteoporosis(pheno),
            "psychiatric": _psychiatric(pheno),
            "seizures": sz,
            "b6_responsive": _b6_responsive(pheno),
            "aed": _aed(sz, pheno),
            "treatment_response": _treatment_response(pheno),
        })
    return patients


_PATIENTS: list[dict] = _build_cohort()


# ── public API functions ───────────────────────────────────────────────────────

def get_overview() -> dict:
    n = len(_PATIENTS)

    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    avg = lambda key: round(sum(p[key] for p in _PATIENTS) / n, 1)

    b6r_pts = [p for p in _PATIENTS if p["phenotype"] == "b6_responsive_classical"]
    nr_pts  = [p for p in _PATIENTS if p["phenotype"] == "b6_nr_severe"]
    mild_pts = [p for p in _PATIENTS if p["phenotype"] == "mild_attenuated"]

    pheno_dist = {
        "b6_responsive_classical": {"n": len(b6r_pts), "pct": round(len(b6r_pts) / n * 100)},
        "b6_nr_severe":            {"n": len(nr_pts),  "pct": round(len(nr_pts)  / n * 100)},
        "mild_attenuated":         {"n": len(mild_pts), "pct": round(len(mild_pts) / n * 100)},
    }

    return {
        "gene": "CBS",
        "full_name": "Cystathionine Beta-Synthase Deficiency (Classical Homocystinuria)",
        "chromosome": "21q22.3",
        "inheritance": "Autosomal Recessive",
        "omim_gene": "OMIM *613381",
        "omim_disease": "OMIM #236200",
        "protein_size": "551 aa; PLP (B6)-dependent; heme regulatory domain; SAM-activated",
        "prevalence": "~1/200,000–344,000 births; Ireland ~1/65,000 (p.Ile278Thr founder)",
        "cohort_n": n,
        "phenotype_distribution": pheno_dist,
        "function": (
            "CBS catalyzes the condensation of homocysteine (Hcy) with L-serine to form "
            "cystathionine — the FIRST and RATE-LIMITING step of the transsulfuration pathway. "
            "This is the gateway for Hcy to exit the methionine cycle and be irreversibly "
            "committed to cysteine synthesis. CBS uses PLP (pyridoxal-5'-phosphate, active B6) "
            "as a catalytic cofactor (Lys119 Schiff base) and is allosterically ACTIVATED by "
            "SAM. The enzyme also carries a unique heme regulatory domain responsive to CO/NO. "
            "Downstream, cystathionine is cleaved by CTH (cystathionase) → cysteine + "
            "α-ketobutyrate → taurine, glutathione (GSH), and sulfate."
        ),
        "mechanism": (
            "CBS LOF → homocysteine cannot enter transsulfuration → accumulates to very high "
            "levels (100–500 µmol/L, the HIGHEST of all inherited HHcy disorders). Concurrently, "
            "methionine is ELEVATED (HIGH) because the downstream catabolic route is blocked — "
            "this is the KEY METABOLIC DIFFERENCE distinguishing CBS from all remethylation "
            "defects (cblE/cblG/MTHFR), where methionine is LOW because remethylation itself "
            "is impaired. Cystathionine is ABSENT (CBS cannot make it). Cysteine is LOW (only "
            "dietary source). Thromboembolism risk is the HIGHEST of all HHcy disorders due to "
            "direct HHcy-mediated endothelial damage and platelet hyperactivation. Ectopia lentis "
            "is PATHOGNOMONIC — HHcy cross-links fibrillin in lens zonules causing dislocation. "
            "B6 (PLP) supplementation restores partial CBS activity in ~50% of patients "
            "(those with residual-activity alleles like p.Ile278Thr) — a response unique to CBS."
        ),
        "key_negative": (
            "KEY POSITIVES (unique to CBS among HHcy disorders): "
            "(1) Methionine HIGH (60–500 µmol/L) — reverse of cblE/cblG/MTHFR where methionine is LOW. "
            "(2) Ectopia lentis (lens dislocation) — pathognomonic; 90% untreated; absent in all cobalamin disorders. "
            "(3) Marfanoid habitus (80%) — connective tissue; never in cblE/cblG/MTHFR. "
            "(4) B6-responsiveness in 50% — no other HHcy disorder responds to B6. "
            "(5) Thromboembolism risk HIGHEST — 50–60% by age 30 untreated. "
            "(6) NBS-detectable via methionine (~60%) — unlike cblE/cblG/MTHFR (all NBS-invisible). "
            "KEY NEGATIVES (shared with cblE/cblG/MTHFR): "
            "(1) MMA NORMAL — distinguishes CBS from combined MMA+HHcy (cblC/cblD/cblF/cblJ/cblX). "
            "(2) MeCbl NORMAL — cobalamin system intact (unlike cblE/cblG where MeCbl is absent). "
            "(3) Serum folate NORMAL — no methylfolate trap (unlike cblE/cblG HIGH, unlike MTHFR LOW). "
            "(4) Megaloblastic anemia ABSENT — no methylfolate trap mechanism. "
            "(5) C3 NORMAL — NBS via Met, not C3 acylcarnitine."
        ),
        "nbs_primary": (
            "Amino acid panel — methionine elevation (>60–100 µmol/L triggers recall on most panels). "
            "~60% detected at birth; B6-non-responsive severe patients almost always detected "
            "(methionine very high); mild/B6-responsive patients may be missed."
        ),
        "nbs_secondary": (
            "Some programs now add total homocysteine (tHcy) to NBS — improves CBS detection "
            "significantly. Ophthalmology: ectopia lentis on slit-lamp (inferior dislocation "
            "in CBS, vs superior in Marfan syndrome). Urine amino acids: cystathionine absent; "
            "homocystine present (oxidized dimer of Hcy in urine)."
        ),
        "kpis": {
            "avg_homocysteine_umol_l":  avg("homocysteine_umol_l"),
            "avg_methionine_umol_l":    avg("methionine_umol_l"),
            "avg_cysteine_umol_l":      avg("cysteine_umol_l"),
            "avg_serum_folate_nmol_l":  avg("serum_folate_nmol_l"),
            "avg_hemoglobin_g_dl":      avg("hemoglobin_g_dl"),
            "pct_seizures":             pct(lambda p: p["seizures"]),
            "pct_ectopia_lentis":       pct(lambda p: p["ectopia_lentis"]),
            "pct_marfanoid":            pct(lambda p: p["marfanoid"]),
            "pct_thromboembolism":      pct(lambda p: p["thromboembolism"]),
            "pct_idd":                  pct(lambda p: p["idd"]),
            "pct_osteoporosis":         pct(lambda p: p["osteoporosis"]),
            "pct_psychiatric":          pct(lambda p: p["psychiatric"]),
            "pct_b6_responsive":        pct(lambda p: p["b6_responsive"]),
            "pct_nbs_detected":         pct(lambda p: p["nbs_detected"]),
            "pct_mma_normal":           pct(lambda p: p["mma_normal"]),
            "pct_mecbl_normal":         pct(lambda p: p["mecbl_normal"]),
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)
    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Focal seizures (thrombotic/stroke-related)", "pct": 55},
        {"type": "Generalized tonic-clonic", "pct": 30},
        {"type": "Absence / myoclonic", "pct": 10},
        {"type": "Infantile spasms (neonatal-severe)", "pct": 5},
    ]

    metabolic_triggers = [
        {"trigger": "High-methionine diet (meat, dairy, legumes without restriction)",
         "pct": 92, "mechanism": "Worsens HHcy; methionine floods pathway → CBS-blocked → more Hcy accumulates"},
        {"trigger": "Fasting / catabolic illness",
         "pct": 68, "mechanism": "Protein catabolism releases methionine → HHcy surge; thrombosis risk increases"},
        {"trigger": "Nitrous oxide (N2O) — MODERATE risk",
         "pct": 45, "mechanism": "N2O oxidizes MTR cobalamin; MTR pathway intact in CBS but elevated HHcy sensitizes; not absolute CI unlike cblE/cblG"},
        {"trigger": "Estrogen / oral contraceptives",
         "pct": 42, "mechanism": "Combined thrombogenic effect with HHcy; absolute contraindication in CBS"},
        {"trigger": "Missed betaine / B6 doses",
         "pct": 75, "mechanism": "HHcy rebounds rapidly without BHMT bypass; thrombosis can occur within days of missed betaine"},
        {"trigger": "Surgery / immobility",
         "pct": 38, "mechanism": "Immobility + HHcy = extreme DVT/PE risk; anticoagulation mandatory perioperatively"},
    ]

    treatments = [
        {
            "treatment": "Pyridoxine (B6) — 100–500 mg/day",
            "level": "Level A (B6-responsive patients only, 50%)",
            "mechanism": (
                "PLP is CBS cofactor; high-dose B6 saturates PLP binding site in residual-activity "
                "alleles (p.Ile278Thr, p.Glu302Lys) → restores partial CBS enzyme activity → "
                "HHcy normalizes (>50% reduction). B6 testing is MANDATORY before finalizing treatment: "
                "if responsive, B6 alone may normalize HHcy. B6-non-responsive patients do NOT benefit "
                "and should proceed directly to betaine + methionine restriction."
            ),
        },
        {
            "treatment": "Betaine (trimethylglycine, TMG) — 100–200 mg/kg/day (max 6–20 g/day)",
            "level": "Level A (ALL CBS patients — B6-responsive AND non-responsive)",
            "mechanism": (
                "Betaine provides methyl groups for the BHMT (betaine-homocysteine methyltransferase) "
                "pathway: Hcy + betaine → methionine (via BHMT, liver/kidney). This is a CBS-INDEPENDENT "
                "remethylation route. Even when CBS is blocked, betaine drives Hcy into methionine "
                "via BHMT → reduces HHcy. KEY CAUTION: in severe CBS, betaine-driven methionine "
                "production can further RAISE methionine (already elevated); monitor closely. "
                "Target tHcy <50 µmol/L; betaine alone achieves 50–70% reduction."
            ),
        },
        {
            "treatment": "Methionine restriction (low-Met diet + Met-free amino acid formula)",
            "level": "Level A (especially B6-non-responsive patients)",
            "mechanism": (
                "Restrict dietary methionine intake → less substrate for homocysteine production "
                "(methionine → SAM → SAH → Hcy cycle). Methionine-free amino acid formula provides "
                "complete essential amino acids (except methionine) to support growth. Target plasma "
                "methionine 20–80 µmol/L (in range). CRITICAL: cysteine MUST be supplemented (conditionally "
                "essential in CBS — transsulfuration pathway blocked → no endogenous cysteine synthesis)."
            ),
        },
        {
            "treatment": "Cystine/cysteine supplementation",
            "level": "Level B (all CBS patients on restricted diet)",
            "mechanism": (
                "Transsulfuration blocked → cysteine not being synthesized → cysteine is conditionally "
                "essential in CBS. Supplement as cystine 100–200 mg/kg/day in formula. Without cysteine, "
                "GSH (glutathione) synthesis impaired → oxidative stress; taurine also reduced."
            ),
        },
        {
            "treatment": "Folic acid — 5 mg/day",
            "level": "Level B (support remethylation / BHMT efficiency)",
            "mechanism": (
                "Folate supports remethylation cycle efficiency; BHMT pathway indirectly depends on "
                "folate-driven THF regeneration. Also reduces cardiovascular risk. Given that CBS "
                "patients have intact MTHFR/MTR, folate is not Level A (no methylfolate trap); "
                "but it supports overall remethylation."
            ),
        },
        {
            "treatment": "Aspirin (antithrombotic) — 75–100 mg/day",
            "level": "Level A (adults; thrombosis prophylaxis)",
            "mechanism": (
                "CBS untreated → 50–60% thromboembolism risk by age 30. HHcy directly damages "
                "vascular endothelium, activates platelets, promotes coagulation. Low-dose aspirin "
                "reduces platelet aggregation. Anticoagulation (LMWH/warfarin) for documented "
                "thrombosis or perioperative cover."
            ),
        },
        {
            "treatment": "Riboflavin (B2) — 10–20 mg/day",
            "level": "Level C (general support)",
            "mechanism": (
                "B2 supports FAD-dependent enzymes in the methionine cycle. NOT Level A for CBS "
                "(unlike MTHFR where B2 is Level A for FAD-dependent MTHFR activity)."
            ),
        },
    ]

    drug_risks = [
        {
            "agent": "Nitrous oxide (N2O)",
            "risk": "MODERATE (not absolute CI — different from cblE/cblG where it is ABSOLUTE CI)",
            "mechanism": (
                "N2O irreversibly oxidizes cobalamin on MTR → MTR inactivated. In CBS, "
                "MTR and the cobalamin system are INTACT (unlike cblE/cblG where they are already "
                "compromised). However, severely elevated HHcy in CBS makes the cobalamin/MTR system "
                "more vulnerable to oxidative stress from N2O. Use with CAUTION, not as absolute CI. "
                "Pretreat with methionine + betaine; supplement B12 perioperatively."
            ),
        },
        {
            "agent": "Estrogen / oral contraceptives",
            "risk": "HIGH RISK — ABSOLUTE CONTRAINDICATION (combined thrombogenic risk)",
            "mechanism": (
                "Estrogen already increases thrombosis risk; combined with HHcy-mediated endothelial "
                "damage in CBS → DVT/PE risk extremely high. Absolute CI in CBS women."
            ),
        },
        {
            "agent": "Valproate",
            "risk": "HIGH RISK — depletes B6 (pyridoxine)",
            "mechanism": (
                "Valproate accelerates B6 catabolism → reduces PLP availability for CBS in B6-responsive "
                "patients → worsens HHcy. Use levetiracetam first-line for seizures in CBS."
            ),
        },
        {
            "agent": "Antifolates (methotrexate, trimethoprim)",
            "risk": "MODERATE RISK",
            "mechanism": (
                "Antifolates block DHFR → reduce THF → impair folate-dependent remethylation "
                "support; worsen HHcy in CBS. Use alternatives."
            ),
        },
        {
            "agent": "High-methionine diet (meat, dairy, eggs unrestricted)",
            "risk": "HIGH RISK — direct substrate loading",
            "mechanism": (
                "Excess methionine → excess Hcy production → exponential HHcy worsening; "
                "transsulfuration still blocked → Hcy cannot be cleared."
            ),
        },
        {
            "agent": "Surgery / immobilization without anticoagulation",
            "risk": "HIGH RISK — DVT/PE",
            "mechanism": (
                "Immobility + HHcy + surgery-induced hypercoagulability = extreme thrombosis risk. "
                "LMWH bridging therapy mandatory perioperatively."
            ),
        },
    ]

    variants = [
        {"variant": "p.Ile278Thr (c.833T>C)", "b6": "B6-RESPONSIVE", "prevalence": "Most common European/Irish; ~30% of CBS alleles", "severity": "Moderate; thermolabile enzyme → responds to high B6/PLP"},
        {"variant": "p.Gly307Ser",             "b6": "B6-NON-RESPONSIVE", "prevalence": "Second most common; ~25% of CBS alleles", "severity": "Severe; catalytic domain; null-like"},
        {"variant": "p.Glu302Lys",             "b6": "B6-RESPONSIVE", "prevalence": "~5% CBS alleles", "severity": "Moderate-severe; responds well to B6"},
        {"variant": "p.Arg369Cys",             "b6": "B6-NON-RESPONSIVE", "prevalence": "~5% CBS alleles", "severity": "Severe; catalytic core disruption; early thrombosis"},
        {"variant": "p.Arg121Cys",             "b6": "B6-NON-RESPONSIVE", "prevalence": "~4% CBS alleles", "severity": "Severe; heme-binding region; null phenotype"},
        {"variant": "p.Asp444Asn",             "b6": "B6-RESPONSIVE", "prevalence": "~3% CBS alleles", "severity": "Attenuated; C-terminal near SAM domain; mild HHcy"},
        {"variant": "c.IVS11+1G>A",            "b6": "B6-NON-RESPONSIVE", "prevalence": "~3% CBS alleles; European", "severity": "Severe; splice null"},
    ]

    return {
        "patient_sample": _PATIENTS[:12],
        "seizure_types":      seizure_types,
        "metabolic_triggers": metabolic_triggers,
        "treatments":         treatments,
        "drug_risks":         drug_risks,
        "variant_breakdown":  variants,
        "biomarker_ranges": {
            "homocysteine_umol_l": {
                "b6_responsive_classical": "60–220 µmol/L (KEY: highest of all isolated HHcy disorders)",
                "b6_nr_severe":           "220–510 µmol/L (severely elevated)",
                "mild_attenuated":         "30–80 µmol/L",
                "note": "ALL ELEVATED vs normal <15 µmol/L; CBS is highest HHcy of all inborn HHcy disorders",
            },
            "methionine_umol_l": {
                "b6_responsive_classical": "80–320 µmol/L ELEVATED",
                "b6_nr_severe":           "200–520 µmol/L ELEVATED",
                "mild_attenuated":         "40–110 µmol/L ELEVATED",
                "note": "KEY POSITIVE: HIGH in CBS; LOW in cblE/cblG/MTHFR (remethylation defects)",
            },
            "mma": "NORMAL (<5 mmol/mol Cr) — ALL patients; MMA arm completely intact",
            "cystathionine": "ABSENT — CBS cannot form cystathionine from Hcy+serine; pathognomonic",
            "cysteine_umol_l": "LOW (12–55 µmol/L); transsulfuration blocked → conditionally essential",
            "serum_folate": "NORMAL (12–32 nmol/L) — NO methylfolate trap; MTHFR/MTR intact",
            "mecbl_adocbl": "BOTH NORMAL — cobalamin system (LMBRD1, ABCD4, MMACHC, MTR, MTRR) fully intact",
            "c3": "NORMAL — no C3 elevation; NBS via methionine, not acylcarnitines",
        },
        "kpi_pcts": {
            "ectopia_lentis":   pct(lambda p: p["ectopia_lentis"]),
            "marfanoid":        pct(lambda p: p["marfanoid"]),
            "thromboembolism":  pct(lambda p: p["thromboembolism"]),
            "idd":              pct(lambda p: p["idd"]),
            "osteoporosis":     pct(lambda p: p["osteoporosis"]),
            "psychiatric":      pct(lambda p: p["psychiatric"]),
            "seizures":         pct(lambda p: p["seizures"]),
            "b6_responsive":    pct(lambda p: p["b6_responsive"]),
            "nbs_detected":     pct(lambda p: p["nbs_detected"]),
        },
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "gene":         "CBS (Cystathionine Beta-Synthase)",
            "gene_omim":    "*613381",
            "disease_omim": "#236200 (Homocystinuria due to CBS deficiency)",
            "protein":      "551 amino acids; heme regulatory domain (N-terminal); catalytic PLP domain; CBS/SAM regulatory domain (C-terminal)",
            "cofactor":     "PLP (pyridoxal-5'-phosphate, active B6) — catalytic (Lys119 Schiff base); B6-responsive alleles have partial PLP affinity rescue",
            "heme":         "Heme b (Fe²⁺) — regulatory; unique among PLP enzymes; responds to CO, NO, redox; NOT catalytic",
            "locus":        "21q22.3",
            "inheritance":  "Autosomal Recessive (biallelic LOF); both sexes equally affected",
            "reaction":     "L-Homocysteine + L-Serine → L-Cystathionine + H₂O (PLP-dependent, irreversible)",
            "pathway":      "Transsulfuration pathway — Step 1 (CBS) and Step 2 (CTH/cystathionase) → cysteine → taurine, GSH, sulfate",
            "regulation":   "Allosterically ACTIVATED by SAM (S-adenosylmethionine) — CBS is the homeostatic 'overflow valve' for excess methionine",
        },
        "key_concepts": [
            {
                "concept": "Methionine is HIGH in CBS (vs LOW in all remethylation defects)",
                "explanation": (
                    "This is the single most important distinguishing biochemical feature. "
                    "In cblE/cblG/MTHFR, remethylation of Hcy → methionine is impaired → methionine falls. "
                    "In CBS, methionine production (via MTR remethylation) is INTACT; the problem is in "
                    "downstream catabolism (transsulfuration) — methionine piles up because it cannot be "
                    "committed to cystathionine synthesis. "
                    "Plasma methionine 60–500 µmol/L (normal 20–40 µmol/L) is the first clue to CBS on NBS."
                ),
            },
            {
                "concept": "B6 (pyridoxine) responsiveness — UNIQUE to CBS among all HHcy disorders",
                "explanation": (
                    "No other inherited HHcy disorder responds to B6. In CBS, PLP (active B6) is the "
                    "catalytic cofactor. ~50% of CBS patients carry alleles (p.Ile278Thr, p.Glu302Lys) "
                    "with residual CBS enzyme that has reduced PLP affinity; high-dose B6 (100–500 mg/day) "
                    "saturates PLP binding → restores partial CBS activity → HHcy normalizes. "
                    "B6 responsiveness must be tested BEFORE finalizing treatment (trial period 3–4 weeks). "
                    "B6-non-responsive patients need betaine + methionine restriction."
                ),
            },
            {
                "concept": "Ectopia lentis — PATHOGNOMONIC for CBS",
                "explanation": (
                    "Homocysteine cross-links and disrupts fibrillin-1 in lens zonules → lens dislocation. "
                    "CBS: dislocation is INFERIOR (vs SUPERIOR in Marfan syndrome — a critical clinical distinction). "
                    "90% of untreated CBS patients develop ectopia lentis by age 10; earlier and more severe in "
                    "B6-non-responsive. ABSENT in cblE/cblG/MTHFR. If ectopia lentis is present, CBS must be "
                    "excluded before any other HHcy cause is assigned."
                ),
            },
            {
                "concept": "Thromboembolism — HIGHEST RISK of all HHcy disorders (50–60% by age 30 untreated)",
                "explanation": (
                    "Hcy directly damages vascular endothelium (eNOS inhibition, oxidative stress), "
                    "activates platelet aggregation, promotes coagulation. CBS produces the highest Hcy "
                    "levels of all inborn HHcy disorders → highest thrombosis risk. "
                    "DVT, PE, cerebral venous thrombosis, arterial stroke ALL reported. "
                    "Aspirin + long-term betaine are thrombosis prophylaxis Level A. "
                    "Estrogen is absolutely contraindicated (additive thrombogenic effect)."
                ),
            },
            {
                "concept": "Cystathionine ABSENT — a pathognomonic biochemical marker",
                "explanation": (
                    "CBS is the only enzyme that synthesizes cystathionine in humans. CBS LOF → "
                    "plasma and urine cystathionine is ABSENT or undetectable. This is pathognomonic. "
                    "Cysteine becomes conditionally essential → must supplement cystine in formula."
                ),
            },
            {
                "concept": "NBS detection via methionine (~60%) — unlike cblE/cblG/MTHFR",
                "explanation": (
                    "cblE/cblG/MTHFR are all NBS-INVISIBLE (no elevated acylcarnitines or "
                    "standard amino acids detectable). CBS is partially NBS-detectable via "
                    "ELEVATED methionine on amino acid panel. B6-non-responsive severe: ~78% detected. "
                    "Mild/B6-responsive: ~55% detected (methionine may be borderline). "
                    "Addition of total homocysteine (tHcy) to NBS improves detection significantly."
                ),
            },
            {
                "concept": "MMA NORMAL — distinguishes CBS from combined MMA+HHcy disorders",
                "explanation": (
                    "cblC (MMACHC), cblD (MMADHC), cblF (LMBRD1), cblJ (ABCD4), cblX (HCFC1) all "
                    "have ELEVATED MMA + HHcy. CBS has NORMAL MMA. This excludes all combined disorders. "
                    "But CBS still needs to be distinguished from isolated HHcy remethylation defects "
                    "(cblE/cblG/MTHFR) — which is done via methionine level (HIGH in CBS, LOW in others) "
                    "and ectopia lentis / marfanoid habitus (present in CBS, absent in others)."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disease": "cblE (MTRR) / cblG (MTR) — remethylation defects",
                "distinguishing": (
                    "KEY DIFFERENCE: Methionine is LOW in cblE/cblG (remethylation impaired → methionine not made). "
                    "Methionine is HIGH in CBS (remethylation intact; transsulfuration exit blocked). "
                    "Additional: ectopia lentis and marfanoid habitus ABSENT in cblE/cblG; PRESENT in CBS. "
                    "MeCbl absent in cblE/cblG (cobalamin defect); NORMAL in CBS (cobalamin intact). "
                    "Serum folate HIGH in cblE/cblG (methylfolate trap); NORMAL in CBS. "
                    "Megaloblastic anemia 80–90% in cblE/cblG; ABSENT in CBS. "
                    "NBS: cblE/cblG invisible; CBS ~60% detected (methionine elevation)."
                ),
            },
            {
                "disease": "MTHFR deficiency — remethylation defect (folate axis)",
                "distinguishing": (
                    "KEY DIFFERENCE: Methionine LOW in MTHFR (no 5-methylTHF → MTR starved → methionine not made). "
                    "Methionine HIGH in CBS. "
                    "Serum folate LOW/NORMAL in MTHFR; NORMAL in CBS (no methylfolate trap in either). "
                    "MeCbl NORMAL in both MTHFR and CBS (cobalamin intact). "
                    "Ectopia lentis in CBS (90%); ABSENT in MTHFR. "
                    "Riboflavin Level A in MTHFR (FAD-cofactor); Level C only in CBS. "
                    "Methionine supplement Level A in MTHFR (product deficiency); RESTRICTED in CBS (methionine already high). "
                    "White matter disease prominent in MTHFR (82%); less so in CBS (thrombotic mechanism)."
                ),
            },
            {
                "disease": "Combined MMA+HHcy disorders (cblC/MMACHC, cblD/MMADHC, cblF/LMBRD1, cblJ/ABCD4, cblX/HCFC1)",
                "distinguishing": (
                    "KEY DIFFERENCE: MMA ELEVATED (200–5000 mmol/mol Cr) in combined disorders; "
                    "MMA NORMAL in CBS (transsulfuration defect, MMA arm intact). "
                    "MeCbl ABSENT in combined disorders; NORMAL in CBS. "
                    "C3 elevated on NBS in combined disorders; NORMAL in CBS (detected via methionine). "
                    "Combined disorders do NOT cause ectopia lentis, marfanoid habitus, or B6 response."
                ),
            },
            {
                "disease": "Marfan syndrome (FBN1, 15q21)",
                "distinguishing": (
                    "Ectopia lentis: in Marfan, dislocation is SUPERIOR; in CBS, INFERIOR. "
                    "Marfan habitus (tall, arachnodactyly, pectus) present in BOTH Marfan and CBS. "
                    "KEY: Marfan has NORMAL homocysteine (no HHcy). "
                    "Simple plasma tHcy measurement distinguishes: CBS has very high tHcy; Marfan normal. "
                    "Genetic testing: FBN1 (Marfan) vs CBS (homocystinuria). "
                    "Thrombosis much more prominent in CBS than Marfan."
                ),
            },
            {
                "disease": "Pyridoxine-dependent epilepsy (ALDH7A1 — antiquitin deficiency)",
                "distinguishing": (
                    "Both conditions involve B6/PLP metabolism. "
                    "ALDH7A1 deficiency: alpha-aminoadipic semialdehyde (alpha-AASA) elevated; "
                    "pipecolic acid elevated; NO HHcy; NO methionine elevation. "
                    "CBS: isolated HHcy + methionine elevation; NOT alpha-AASA or pipecolic acid. "
                    "ALDH7A1 seizures are severe, onset neonatal to early infantile, B6-responsive. "
                    "CBS seizures are secondary (thrombosis), onset later, B6 response relates to "
                    "CBS enzyme restoration, not direct PLP-dependent GABA synthesis like ALDH7A1."
                ),
            },
            {
                "disease": "Nutritional B12 deficiency (vegan / maternal)",
                "distinguishing": (
                    "Nutritional B12 deficiency → MTR cobalamin-depleted → HHcy + megaloblastic anemia. "
                    "KEY: serum B12 LOW in nutritional deficiency; NORMAL in CBS (cobalamin absorption intact). "
                    "Methionine: LOW in nutritional B12 deficiency (MTR impaired → methionine not made); "
                    "HIGH in CBS (MTR intact; transsulfuration blocked). "
                    "Ectopia lentis: ABSENT in nutritional B12 deficiency; PRESENT in CBS. "
                    "Megaloblastic anemia: present in B12 deficiency; ABSENT in CBS. "
                    "Response: rapid complete response to B12 injection in nutritional deficiency; "
                    "CBS requires betaine + methionine restriction + B6 (if responsive)."
                ),
            },
            {
                "disease": "Hypermethioninemia (MAT1A deficiency / glycine N-methyltransferase deficiency)",
                "distinguishing": (
                    "MAT1A deficiency: methionine VERY HIGH (may exceed 1000 µmol/L); Hcy normal or mildly elevated. "
                    "CBS: BOTH methionine HIGH and Hcy VERY HIGH. "
                    "MAT1A: typically benign; no ectopia lentis; no thrombosis; no IDD. "
                    "Distinguisher: tHcy is the key — very high in CBS, near-normal in MAT1A."
                ),
            },
            {
                "disease": "CBS vs cblG (MTR) — the methionine level test",
                "distinguishing": (
                    "Summary differential — the single most useful test: "
                    "Plasma methionine: "
                    "CBS → HIGH (60–500 µmol/L): transsulfuration blocked → methionine accumulates. "
                    "cblG → LOW (<15 µmol/L): MTR cannot remethylate Hcy → methionine not produced. "
                    "cblE → LOW: same as cblG. "
                    "MTHFR → LOW: no 5-methylTHF → MTR starved → methionine not made. "
                    "Therefore: HIGH methionine + HIGH Hcy = CBS (or rare combined hypermethioninemia+HHcy). "
                    "LOW methionine + HIGH Hcy = remethylation defect (cblE, cblG, MTHFR). "
                    "This single test narrows diagnosis before gene panel results are available."
                ),
            },
        ],
    }


# ── entry point (for quick local verification) ───────────────────────────────

if __name__ == "__main__":
    import json
    overview = get_overview()
    bd = get_breakdown()
    defs = get_definitions()

    print(f"Gene: {overview['gene']}  |  N={overview['cohort_n']}")
    print(f"Avg tHcy: {overview['kpis']['avg_homocysteine_umol_l']} µmol/L  "
          f"(expected ~200–300 across all phenotypes)")
    print(f"Avg methionine: {overview['kpis']['avg_methionine_umol_l']} µmol/L  "
          f"(expected HIGH ~150–300 — KEY POSITIVE vs LOW in cblE/cblG/MTHFR)")
    print(f"Avg cysteine: {overview['kpis']['avg_cysteine_umol_l']} µmol/L  "
          f"(expected LOW ~20–35)")
    print(f"% ectopia lentis: {overview['kpis']['pct_ectopia_lentis']}%  "
          f"(expected ~70–80% across all)")
    print(f"% B6-responsive: {overview['kpis']['pct_b6_responsive']}%  "
          f"(expected ~50%)")
    print(f"% thromboembolism: {overview['kpis']['pct_thromboembolism']}%  "
          f"(expected ~45–50% across all)")
    print(f"% NBS detected: {overview['kpis']['pct_nbs_detected']}%  "
          f"(expected ~60% — higher than cblE/cblG/MTHFR which are 0%)")
    print(f"% MeCbl normal: {overview['kpis']['pct_mecbl_normal']}%  "
          f"(expected 100% — cobalamin intact)")
    print(f"% MMA normal: {overview['kpis']['pct_mma_normal']}%  (expected 100%)")
    print(f"Phenotypes: {overview['phenotype_distribution']}")
    print(f"\nBreakdown treatment count: {len(bd['treatments'])}")
    print(f"Definitions differential dx count: {len(defs['differential_diagnosis'])}")
    print("\nAll OK.")
