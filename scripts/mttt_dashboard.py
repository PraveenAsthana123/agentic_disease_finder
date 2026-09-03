#!/usr/bin/env python3
"""MT-TT — Mitochondrially Encoded tRNA-Thr — CPEO / Myopathy / Cardiomyopathy (HCM)
Combined CI+CIV Deficiency (mt-translation fingerprint) | H-strand rCRS 15888–15953

MT-TT (OMIM *590094) encodes mitochondrial tRNA-Thr (UGU anticodon, reading Thr codons
ACA/ACC/ACG/ACT), located on the H-strand at rCRS 15888–15953 (66 nt). MT-TT is situated
between MT-TE (ends 14742, L-strand) and MT-TP (starts 15956, L-strand) in the final
tRNA-cluster of the human mitochondrial genome, flanking the D-loop control region.

m.15923A>G is the most commonly reported pathogenic MT-TT mutation (~35%), causing
CPEO + myopathy + cardiomyopathy (HCM > DCM) + lactic acidosis with combined CI+CIV
deficiency — the mt-translation fingerprint shared by all pathogenic mt-tRNA mutations
that do not involve mt-ribosomal proteins (CII NORMAL — nuclear SDH).

CARDIOMYOPATHY is the most DISTINCTIVE MT-TT feature — hypertrophic cardiomyopathy
(HCM) is present in ~55-65% of MT-TT patients, significantly more prominent than in
MT-TS2 or MT-TL2, and mandates annual echocardiography + Holter monitoring.
Cardiac conduction defects (WPW pattern, HV-block) occur in ~20% — overlap with KSS
if large deletion spans MT-TT.

  MT-TT gene              OMIM *590094
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Cardiomyopathy
                          Lactic Acidosis / SNHL / Exercise Intolerance
  Protein product         tRNA-Thr (UGU anticodon) — 66 nucleotides; RNA gene
                          Thr codons: ACA, ACC, ACG, ACT (UGU anticodon, wobble position 34)
  Genome                  Mitochondrial DNA (mtDNA), H-strand, rCRS 15888–15953
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    Between MT-TE (14742, L-strand) and MT-TP (15956, L-strand)
                          Immediately 5' of the D-loop control region

CARDIOMYOPATHY SURVEILLANCE PROTOCOL:
  Annual echocardiography (LV mass index, LVEF, LVOTO gradient) + 24h Holter ECG.
  WPW-pattern / short PR-interval → full electrophysiology study (ablation if accessory
  pathway symptomatic). HCM → beta-blocker first-line; avoid amiodarone (mt-toxic).
  Cardiac transplant has been performed in MT-TT cardiomyopathy (end-stage HCM/DCM).

NUCLEAR DDx — TARS2 (mt-Threonyl-tRNA Synthetase):
  TARS2 mutations cause combined CI+CIV deficiency with identical biochemical fingerprint
  to MT-TT pathogenic variants. AR biallelic TARS2 → WES-detectable; maternal inheritance
  ABSENT; similar CPEO + myopathy + lactic acidosis phenotype but LESS cardiomyopathy.
  TARS2 is the enzyme charging tRNA-Thr with threonine; loss-of-function impairs all
  mt-Thr aminoacylation, causing similar (though often milder) OXPHOS deficiency.
"""

import random
from collections import Counter

SEED = 799
N_PATIENTS = 40

VARIANTS = [
    ("m.15923A>G", "Anticodon loop (position 34 — wobble base UGU)", "~35%; most common; CPEO + myopathy + cardiomyopathy (HCM) + lactic acidosis; combined CI+CIV fingerprint; adult onset 20–40 yr; exercise intolerance prominent"),
    ("m.15924A>G", "Anticodon (position 35 — middle anticodon base)", "~25%; CPEO + exercise intolerance + myopathy; moderate CI+CIV reduction; cardiomyopathy ~40%; SNHL ~35%; adult onset; overlap TARS2 DDx"),
    ("m.15928G>A", "Variable loop / anticodon stem boundary", "~22%; multisystem — CPEO + cardiomyopathy + SNHL; CI+CIV combined deficiency; lactic acidosis moderate; cardiac conduction defect in ~25%; HARS2/ANT1 DDx"),
    ("m.15940A>G", "T-stem (position 49 area)", "~10%; exercise intolerance + myopathy + lactic acidosis; CPEO less prominent; CI+CIV combined; SNHL ~30%; adult onset 30–50 yr"),
    ("Large deletion", "MT-TT–spanning deletion (KSS / CPEO region)", "~8%; deletion spanning MT-TT ± MT-TE/MT-TP/D-loop → KSS/CPEO; multi-complex OXPHOS; annual Holter/echo mandatory; often sporadic"),
]

PHENOTYPES = [
    "CPEO + myopathy + HCM cardiomyopathy — adult heteroplasmy 45–75%",
    "CPEO + exercise intolerance + lactic acidosis — moderate heteroplasmy",
    "Multisystem — CPEO + cardiomyopathy + SNHL + lactic acidosis — high heteroplasmy",
    "Cardiomyopathy (HCM) dominant + mild myopathy — cardiac presentation first",
    "Exercise intolerance + myopathy + SNHL — low-moderate heteroplasmy",
]

TRIGGERS = [
    "Intercurrent illness / fever", "Fasting / prolonged NPO", "Anaesthetic agents (propofol)",
    "VPA/valproate", "Physiological stress / surgery", "Amiodarone (mt-toxic)",
    "Linezolid antibiotic", "High-dose statins", "Aminoglycosides (cochlear OXPHOS)",
]

TREATMENTS = [
    ("CoQ10 (ubiquinol)", "Level C", "Mitochondrial cofactor; 10–30 mg/kg/day divided doses; ubiquinol preferred; CI+CIV combined deficiency standard adjunct"),
    ("Riboflavin (B2)", "Level C", "50–200 mg/day; FAD cofactor; CI+CIII support; low risk; continue long-term"),
    ("Thiamine (B1)", "MANDATORY empiric", "10–20 mg/kg/day IV acutely; PDH cofactor; empiric before workup; BTBGD exclusion first"),
    ("Biotin", "MANDATORY empiric", "10 mg/day; BTD/SLC19A3 exclusion empiric; withdraw only after BTBGD excluded"),
    ("L-Carnitine", "Level C", "50–100 mg/kg/day; secondary carnitine deficiency common; correction mandatory"),
    ("Beta-blockers (HCM)", "STANDARD — cardiac", "Metoprolol / bisoprolol — first-line for HCM + outflow obstruction; avoid amiodarone (mt-toxic); rate control for AF"),
    ("Elamipretide", "Phase 2 trials", "Cardiolipin stabiliser; cardioprotective in mt-cardiomyopathy; investigational; promising in HCM cases"),
    ("ICD / cardiac monitoring", "MANDATORY surveillance", "Annual echo + 24h Holter; ICD if EF <35% or malignant arrhythmia; WPW → EP study + ablation"),
]

ABSOLUTE_CI = [
    ("Metformin", "ABSOLUTE CI", "Complex I inhibitor → fatal lactic acidosis in CI-deficient skeletal muscle; ALL MT-TT carriers — even those without overt diabetes"),
    ("VPA / valproate", "ABSOLUTE CI", "mt-ribosome inhibitor + CoA sequestration + POLG inhibition; catastrophic in CI+CIV deficiency; use LEV"),
    ("Propofol", "ABSOLUTE CI (PRIS)", "CIV inhibitor + propofol infusion syndrome; lethal in pre-existing CI+CIV deficiency; use sevoflurane"),
    ("Linezolid", "ABSOLUTE CI", "mt-23S rRNA inhibition → blocks ALL 13 mtDNA-encoded OXPHOS translations; compounding CI+CIV failure"),
    ("Chloramphenicol", "ABSOLUTE CI", "mt-ribosome inhibitor (70S); blocks mt-translation globally; catastrophic in tRNA-Thr defect"),
    ("Amiodarone", "ABSOLUTE CI / HIGH CAUTION", "Accumulates in mitochondria; inhibits OXPHOS electron transport; worsens CI+CIV deficiency; use alternative rhythm control (flecainide cautiously; sotalol cautiously)"),
    ("High-dose statins", "CAUTION / RELATIVE CI", "CoQ10 depletion + mitochondrial myopathy worsening; if mandatory, use low-dose pravastatin; monitor CK + lactate"),
    ("Ketogenic diet", "CONTRAINDICATED", "High-fat low-carb → catabolism + acetyl-CoA → impairs CI-dependent NADH oxidation in mt-disease; worsens lactic acidosis"),
]


def _make_patients():
    rng = random.Random(SEED)
    pts = []
    variant_pool = ["m.15923A>G"] * 14 + ["m.15924A>G"] * 10 + ["m.15928G>A"] * 9 + \
                   ["m.15940A>G"] * 4 + ["Large deletion"] * 3
    rng.shuffle(variant_pool)
    for i, var in enumerate(variant_pool[:N_PATIENTS]):
        if var == "m.15923A>G":
            het = rng.randint(55, 82)
            ci = rng.randint(22, 42)
            civ = rng.randint(20, 40)
            cii = rng.randint(88, 104)
            onset = rng.randint(18, 42)
            cpeo = True
            myo = True
            cardio = rng.random() < 0.68
            snhl = rng.random() < 0.40
            dm = rng.random() < 0.06
            rrf = rng.random() < 0.72
            leigh = False
        elif var == "m.15924A>G":
            het = rng.randint(45, 72)
            ci = rng.randint(28, 48)
            civ = rng.randint(26, 46)
            cii = rng.randint(88, 106)
            onset = rng.randint(22, 50)
            cpeo = rng.random() < 0.90
            myo = True
            cardio = rng.random() < 0.42
            snhl = rng.random() < 0.35
            dm = rng.random() < 0.04
            rrf = rng.random() < 0.65
            leigh = False
        elif var == "m.15928G>A":
            het = rng.randint(50, 78)
            ci = rng.randint(20, 40)
            civ = rng.randint(18, 38)
            cii = rng.randint(86, 103)
            onset = rng.randint(15, 40)
            cpeo = rng.random() < 0.85
            myo = True
            cardio = rng.random() < 0.62
            snhl = rng.random() < 0.48
            dm = rng.random() < 0.05
            rrf = rng.random() < 0.70
            leigh = False
        elif var == "m.15940A>G":
            het = rng.randint(40, 65)
            ci = rng.randint(32, 52)
            civ = rng.randint(30, 50)
            cii = rng.randint(89, 107)
            onset = rng.randint(28, 55)
            cpeo = rng.random() < 0.70
            myo = True
            cardio = rng.random() < 0.30
            snhl = rng.random() < 0.30
            dm = rng.random() < 0.03
            rrf = rng.random() < 0.58
            leigh = False
        else:  # large deletion
            het = rng.randint(25, 55)
            ci = rng.randint(18, 35)
            civ = rng.randint(16, 33)
            cii = rng.randint(82, 100)
            onset = rng.randint(8, 25)
            cpeo = True
            myo = True
            cardio = rng.random() < 0.75
            snhl = rng.random() < 0.60
            dm = rng.random() < 0.08
            rrf = rng.random() < 0.80
            leigh = rng.random() < 0.15
        lactate = round(rng.uniform(2.8, 8.5) if ci < 35 else rng.uniform(1.8, 5.2), 1)
        pts.append({
            "id": f"TT-{i+1:03d}",
            "variant": var,
            "sex": rng.choice(["M", "F"]),
            "age_onset_yr": onset,
            "heteroplasmy_blood_pct": het,
            "ci_pct": ci,
            "civ_pct": civ,
            "cii_pct": cii,
            "lactate_mmol_L": lactate,
            "cpeo": cpeo,
            "myopathy": myo,
            "cardiomyopathy": cardio,
            "snhl": snhl,
            "diabetes_mellitus": dm,
            "ragged_red_fibres": rrf,
            "leigh_like": leigh,
        })
    return pts


def get_overview():
    pts = _make_patients()
    n = len(pts)

    def pct(key): return round(100 * sum(1 for p in pts if p[key]) / n)

    avg = lambda key: round(sum(p[key] for p in pts) / n, 1)

    phenotype_counts = Counter()
    for p in pts:
        if p["cpeo"] and p["cardiomyopathy"] and p["myopathy"]:
            phenotype_counts["CPEO + myopathy + HCM cardiomyopathy"] += 1
        elif p["cpeo"] and p["myopathy"] and not p["cardiomyopathy"]:
            phenotype_counts["CPEO + myopathy (no cardiomyopathy)"] += 1
        elif p["cardiomyopathy"] and not p["cpeo"]:
            phenotype_counts["Cardiomyopathy (HCM) dominant + mild myopathy"] += 1
        elif p["snhl"] and p["myopathy"]:
            phenotype_counts["Multisystem — CPEO + SNHL + cardiomyopathy"] += 1
        else:
            phenotype_counts["Exercise intolerance + myopathy + lactic acidosis"] += 1

    pheno_dist = [{"phenotype": k, "count": v, "pct": round(100 * v / n)}
                  for k, v in sorted(phenotype_counts.items(), key=lambda x: -x[1])]

    hmap = [
        {"range": "Blood <30% (any variant)", "phenotype": "Asymptomatic carrier; exercise intolerance subclinical; cardiology screen annually", "management": "Surveillance echo+Holter annually; avoid absolute CIs; no treatment unless symptomatic"},
        {"range": "Blood 30–50% (m.15923A>G)", "phenotype": "Mild exercise intolerance + myalgia; CPEO early-ptosis; HCM mild/absent; lactic acidosis on exertion", "management": "CoQ10 + riboflavin + thiamine; cardiology co-management; CPEO monitoring"},
        {"range": "Blood 50–70% (m.15923A>G)", "phenotype": "CPEO + myopathy + HCM; lactic acidosis; exertional dyspnoea; SNHL emerging", "management": "Full mitochondrial cocktail; beta-blocker for HCM; annual echo; audiometry; avoid triggers"},
        {"range": "Blood >70% (m.15923A>G/15928G>A)", "phenotype": "Severe CPEO + myopathy + progressive HCM; cardiac conduction defects; SNHL significant; lactic acidosis at rest", "management": "Cardiology + neurology co-management; ICD if EF <35%; cochlear implant consideration; ICU readiness for crises"},
        {"range": "Large deletion (any blood level)", "phenotype": "KSS phenotype: CPEO + cardiomyopathy + retinal pigmentation + ataxia + cardiac conduction block; multi-OXPHOS", "management": "Annual Holter mandatory; pacemaker if HV block; ophthalmology; endocrinology (diabetes risk 15%); ICD consideration"},
    ]

    return {
        "cohort_statistics": {
            "n_patients": n,
            "seed": SEED,
            "avg_heteroplasmy_blood_pct": avg("heteroplasmy_blood_pct"),
            "avg_ci_activity_pct_normal": avg("ci_pct"),
            "avg_civ_activity_pct_normal": avg("civ_pct"),
            "avg_cii_activity_pct_normal": avg("cii_pct"),
            "pct_cpeo": pct("cpeo"),
            "pct_myopathy": pct("myopathy"),
            "pct_cardiomyopathy": pct("cardiomyopathy"),
            "pct_snhl": pct("snhl"),
            "pct_diabetes_mellitus": pct("diabetes_mellitus"),
            "pct_ragged_red_fibres": pct("ragged_red_fibres"),
            "pct_leigh_like": pct("leigh_like"),
            "avg_age_onset_yr": avg("age_onset_yr"),
        },
        "phenotype_distribution": pheno_dist,
        "heteroplasmy_clinical_map": hmap,
        "key_molecular_features": [
            "tRNA-Thr (UGU anticodon, 66 nt) — H-strand rCRS 15888–15953 — OMIM *590094",
            "Combined CI+CIV deficiency (mt-translation fingerprint); CII NORMAL (nuclear SDH)",
            "Thr codons decoded: ACA, ACC, ACG, ACT (UGU anticodon: wobble at position 34)",
            "CARDIOMYOPATHY (HCM > DCM) ~55–65% — most DISTINCTIVE MT-TT clinical feature",
            "CPEO (ptosis + ophthalmoplegia) ~80% — progressive over decades",
            "Myopathy (proximal weakness + fatigue) ~90% — ragged-red fibres on muscle biopsy",
            "SNHL (sensorineural hearing loss) ~40–48% — annual audiometry mandatory",
            "Lactic acidosis (blood lactate >2.5 mmol/L) ~75% — exertional; may be resting at high heteroplasmy",
            "H-strand encoded — standard NGS coverage adequate (unlike L-strand MT-TE/MT-TP)",
            "Nuclear DDx: TARS2 (mt-Threonyl-tRNA synthetase, AR biallelic, WES-detectable)",
            "Large deletions spanning MT-TT → KSS phenotype: CPEO + conduction block + retinal pigmentation",
            "AMIODARONE ABSOLUTE CI — mt-OXPHOS inhibitor; use alternative rhythm control",
            "Cardiac transplant performed in end-stage MT-TT HCM/DCM — report to international registry",
        ],
        "cohort_summary_features": [
            f"40-patient cohort, seed-{SEED}: {pct('cpeo')}% CPEO, {pct('cardiomyopathy')}% cardiomyopathy (HCM predominant), {pct('myopathy')}% myopathy",
            f"Mean heteroplasmy (blood): {avg('heteroplasmy_blood_pct')}% — muscle underestimates blood by 10–15%; biopsy for definitive",
            f"Mean CI activity: {avg('ci_pct')}% normal; Mean CIV: {avg('civ_pct')}% normal; Mean CII: {avg('cii_pct')}% (NORMAL — nuclear SDH)",
            f"Mean age of onset: {avg('age_onset_yr')} yr (range: ~8–55 yr depending on variant and heteroplasmy)",
            f"SNHL: {pct('snhl')}% — annual audiometry; cochlear implant effective if <75 dB HL loss; verify CI+CIV in cochlea",
            f"RRF (ragged-red fibres): {pct('ragged_red_fibres')}% — COX-negative SDH-positive; Gomori trichrome diagnostic",
            f"Diabetes: {pct('diabetes_mellitus')}% (low — unlike MT-TE MIDM or MT-TL1 MIDD; DM in MT-TT mostly KSS large deletion patients)",
            "Cardiac surveillance MANDATORY: annual echo (LV mass, LVEF, LVOTO) + 24h Holter (arrhythmia screening)",
            "Metformin ABSOLUTE CI even if DM arises in KSS/MT-TT — use insulin; amiodarone ABSOLUTE CI for arrhythmia management",
            "WES MISSES MT-TT — dedicated mtDNA panel with H-strand coverage required; blood mtDNA acceptable (H-strand gene)",
        ],
        "clinical_alerts": [
            {"alert": "CARDIOMYOPATHY (HCM) — Annual Echo + Holter MANDATORY", "detail": "~55–65% of MT-TT patients; HCM > DCM; cardiac conduction defects ~20%; WPW pattern → EP study; AMIODARONE ABSOLUTE CI"},
            {"alert": "AMIODARONE ABSOLUTE CI", "detail": "Accumulates in mitochondria; inhibits OXPHOS electron transport; worsens CI+CIV deficiency; use flecainide/sotalol cautiously; beta-blockers preferred"},
            {"alert": "VPA ABSOLUTE CI — Use LEV", "detail": "mt-ribosome inhibitor + CoA sequestration; catastrophic in tRNA-Thr CI+CIV deficiency; document at every encounter"},
            {"alert": "PROPOFOL ABSOLUTE CI (PRIS)", "detail": "CIV inhibitor; lethal in CI+CIV deficiency; use sevoflurane for all anaesthetics"},
            {"alert": "METFORMIN ABSOLUTE CI", "detail": "CI inhibitor → fatal lactic acidosis; even if DM appears in MT-TT/KSS; use insulin"},
            {"alert": "BTBGD (SLC19A3) MANDATORY EXCLUSION", "detail": "Treatable Leigh-like mimic; give thiamine + biotin empirically; exclude before MT-TT diagnosis"},
        ],
    }


def get_breakdown():
    pts = _make_patients()

    variant_summaries = []
    for var_info in VARIANTS:
        vname = var_info[0]
        vpts = [p for p in pts if p["variant"] == vname]
        if not vpts:
            continue
        n = len(vpts)
        avg = lambda key: round(sum(p[key] for p in vpts) / n, 1)
        pct = lambda key: round(100 * sum(1 for p in vpts if p[key]) / n)
        variant_summaries.append({
            "variant": vname,
            "region": var_info[1],
            "n": n,
            "avg_heteroplasmy_blood_pct": avg("heteroplasmy_blood_pct"),
            "avg_ci_activity_pct": avg("ci_pct"),
            "avg_civ_activity_pct": avg("civ_pct"),
            "pct_cpeo": pct("cpeo"),
            "pct_myopathy": pct("myopathy"),
            "pct_cardiomyopathy": pct("cardiomyopathy"),
            "pct_snhl": pct("snhl"),
            "pct_diabetes_mellitus": pct("diabetes_mellitus"),
            "note": var_info[2],
        })

    trigger_rates = []
    rng = random.Random(SEED + 1)
    for trigger in TRIGGERS:
        trigger_rates.append({"trigger": trigger, "pct": rng.randint(15, 72)})
    trigger_rates.sort(key=lambda x: -x["pct"])

    treatment_info = [{"agent": t[0], "evidence": t[1], "note": t[2]} for t in TREATMENTS]

    ci_info = [{"agent": a[0], "category": a[1], "rationale": a[2]} for a in ABSOLUTE_CI]

    per_patient = []
    for p in pts:
        per_patient.append({
            "id": p["id"], "variant": p["variant"], "sex": p["sex"],
            "age_onset_yr": p["age_onset_yr"],
            "heteroplasmy_blood_pct": p["heteroplasmy_blood_pct"],
            "ci_pct": p["ci_pct"], "civ_pct": p["civ_pct"], "cii_pct": p["cii_pct"],
            "lactate_mmol_L": p["lactate_mmol_L"],
            "cpeo": p["cpeo"], "myopathy": p["myopathy"],
            "cardiomyopathy": p["cardiomyopathy"],
            "snhl": p["snhl"],
            "diabetes_mellitus": p["diabetes_mellitus"],
            "ragged_red_fibres": p["ragged_red_fibres"],
            "leigh_like": p["leigh_like"],
        })

    n = len(pts)
    return {
        "variant_summaries": variant_summaries,
        "per_patient": per_patient,
        "trigger_rates": trigger_rates,
        "treatment_info": treatment_info,
        "contraindication_info": ci_info,
        "biochemical_fingerprint": {
            "CI_pct_normal": f"{round(sum(p['ci_pct'] for p in pts)/n,1)}",
            "CIV_pct_normal": f"{round(sum(p['civ_pct'] for p in pts)/n,1)}",
            "CII_pct_normal": f"{round(sum(p['cii_pct'] for p in pts)/n,1)} (NORMAL — nuclear-encoded SDH)",
            "pattern": "CI + CIV reduced; CII NORMAL → mt-translation fingerprint (tRNA-Thr defect)",
            "BN_PAGE": "CI band absent/reduced; CIV (COX) absent/reduced; CII (SDH) band PRESENT and NORMAL; CIII may be additionally reduced in large deletions spanning MT-CYB",
            "muscle_histochemistry": "RRF on Gomori trichrome; COX-negative fibres; SDH-positive ragged fibres; HCM — concentric LV hypertrophy on echo; cardiac muscle biopsy shows similar COX-negative fibres in severe cases",
            "H_strand_NGS_note": "MT-TT (15888-15953) is H-strand encoded — standard mtDNA NGS coverage is adequate; no L-strand coverage pitfall (unlike MT-TE, MT-TP, MT-ND6)",
        },
        "ddx_comparison": [
            {
                "gene": "MT-TL1 (MELAS/MIDD)",
                "disease": "MELAS + MIDD (m.3243A>G)",
                "oxphos": "CI+CIII+CIV pan-OXPHOS",
                "distinguisher": "Stroke-like episodes (SLE) ABSENT in MT-TT; MIDD = maternal DM+deafness (m.3243A>G) vs MT-TT minimal DM; MT-TL1 pan-OXPHOS (CI+CIII+CIV) vs MT-TT CI+CIV only; cardiomyopathy less prominent in MT-TL1 than MT-TT",
            },
            {
                "gene": "MT-TK (MERRF)",
                "disease": "MERRF + MSL — Myoclonic Epilepsy",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "Myoclonic epilepsy (90%) ABSENT in MT-TT; MSL/Madelung lipomatosis (10–25%) ABSENT in MT-TT; cardiomyopathy less prominent in MERRF; m.8344A>G most common MERRF mutation",
            },
            {
                "gene": "MT-TH (tRNA-His)",
                "disease": "Combined CI+CIV — Leigh-like + CPEO + Cardiomyopathy",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "Leigh-like MRI more prominent in MT-TH; similar HCM pattern; MT-TH m.12147G>A most common; SNHL less distinctive in MT-TH; cardiomyopathy overlap but MT-TH Leigh-like at high heteroplasmy vs MT-TT CPEO-dominant",
            },
            {
                "gene": "MT-TS2 (tRNA-Ser AGY)",
                "disease": "SHORTEST mt-tRNA — CPEO + SNHL + Myopathy",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "Isolated SNHL at low heteroplasmy DISTINCTIVE of MT-TS2; LESS cardiomyopathy than MT-TT; 59 nt (shortest mt-tRNA vs MT-TT 66 nt); m.12258C>A most common; NO cardiac transplant reports",
            },
            {
                "gene": "TARS2 (mt-Threonyl-tRNA Synthetase)",
                "disease": "Combined CI+CIV Deficiency — AR nuclear",
                "oxphos": "CI+CIV (same fingerprint via tRNA-Thr aminoacylation failure)",
                "distinguisher": "WES-detectable AR biallelic gene; MATERNAL inheritance ABSENT; TARS2 charges tRNA-Thr — functional mimicry of MT-TT pathogenic variants; CPEO + myopathy but LESS cardiomyopathy; cardiomyopathy is a useful clinical discriminator for MT-TT vs TARS2",
            },
            {
                "gene": "ANT1 / SLC25A4 (adPEO)",
                "disease": "Autosomal Dominant CPEO (adPEO) + Cardiomyopathy",
                "oxphos": "mtDNA instability → secondary deletions; CPEO + myopathy",
                "distinguisher": "AD inheritance (paternal transmission possible) vs MATERNAL only; ANT1 mutations → mtDNA deletions (secondary); cardiomyopathy overlap with MT-TT but AD pattern distinguishes; WES-detectable; POLG/ANT1 panel concurrent with mtDNA sequencing",
            },
            {
                "gene": "POLG (SANDO / Alpers)",
                "disease": "mtDNA depletion / deletion; AR nuclear",
                "oxphos": "Pan-OXPHOS; mtDNA depletion",
                "distinguisher": "HEPATOPATHY (30–50%) ABSENT in MT-TT; mtDNA depletion on Southern/qPCR ABSENT in MT-TT (stable mtDNA copy number); WES detects POLG; neuropathy prominent in POLG vs peripheral myopathy in MT-TT",
            },
            {
                "gene": "SLC19A3 (BTBGD)",
                "disease": "Biotin-Thiamine-Responsive BGD",
                "oxphos": "Normal OXPHOS biochemistry",
                "distinguisher": "TREATABLE — thiamine + biotin → MRI normalisation; Leigh-like MRI; NORMAL CI/CIV biochemistry; MANDATORY first exclusion; especially important in high-heteroplasmy MT-TT cases with Leigh-like MRI",
            },
        ],
    }


def get_definitions():
    return {
        "gene_biology": {
            "MT-TT": "Mitochondrially encoded tRNA-Thr (UGU anticodon, reads ACA/ACC/ACG/ACT); 66 nt RNA gene; H-strand rCRS 15888–15953; flanked by MT-TE (ends 14742, L-strand) and MT-TP (starts 15956, L-strand); immediately 5' of the D-loop control region; OMIM *590094",
            "tRNA_Thr_function": "Delivers threonine to the mt-ribosome A-site during translation of all 13 mtDNA-encoded OXPHOS subunits; UGU anticodon reads Thr codons ACA (wobble position 34, inosine modification), ACC, ACG, ACT; mt-aminoacyl-tRNA synthetase: TARS2 (mt-threonyl-tRNA synthetase); TARS2 mutations cause combined OXPHOS deficiency indistinguishable biochemically from MT-TT pathogenic variants — AR nuclear, WES-detectable",
            "H_strand_encoding": "MT-TT is encoded on the HEAVY (H) strand — standard mtDNA NGS coverage is adequate. Unlike L-strand genes (MT-TE, MT-TP, MT-ND6), no special L-strand coverage QC is required. H-strand capture panels (Illumina TruSight, Agilent SureSelect) provide reliable coverage at 15888–15953.",
            "mt_translation_fingerprint": "CI + CIV reduced; CII (SDH, nuclear-encoded) NORMAL; tRNA-Thr is required for inserting Thr into all 13 mtDNA-encoded OXPHOS polypeptides; CIII may be additionally reduced in large deletions spanning MT-CYB (14747–15887), but tRNA-Thr point mutations specifically affect CI (7 ND-subunits) and CIV (3 CO-subunits) most prominently",
            "cardiomyopathy_mechanism": "Cardiac muscle has the highest mitochondrial density in the body and greatest dependence on OXPHOS for ATP (90% mitochondrial vs 10% glycolytic in cardiomyocytes). CI+CIV deficiency in cardiomyocytes → ATP depletion → myofibrillar dysfunction → HCM (initially compensatory hypertrophy) → eventual DCM (decompensated dilation) → arrhythmias + conduction defects. WPW-pattern pre-excitation occurs in ~15% due to accessory pathway development in the hypertrophied septum.",
            "TARS2_nuclear_DDx": "TARS2 (threonyl-tRNA synthetase, mitochondrial; OMIM *610957) charges mt-tRNA-Thr with threonine — functional equivalent to the MT-TT tRNA. Biallelic TARS2 loss-of-function → impaired mt-tRNA-Thr aminoacylation → same downstream CI+CIV OXPHOS deficiency as MT-TT pathogenic variants. Clinical phenotype overlaps: CPEO + myopathy + lactic acidosis + combined CI+CIV deficiency. Key discriminator: TARS2 is AR (paternal transmission possible), WES-detectable, and causes LESS cardiomyopathy than MT-TT; maternal inheritance pattern clinches MT-TT diagnosis.",
            "rCRS_position": "H-strand rCRS 15888–15953 (66 nt); BETWEEN MT-TE (ends 14742) and MT-TP (starts 15956); 14-nt intergenic spacer (15954–15955 + control region boundary); MT-TT is the penultimate mt-tRNA gene before the D-loop (control region, 16024–576)",
            "large_deletion_KSS": "Large deletions spanning MT-TT commonly co-delete MT-TE, MT-TP, and portions of the D-loop or MT-CYB → KSS phenotype (CPEO + cardiac conduction block + retinal pigmentation < age 20 yr + ataxia + cerebellar signs). Deletions typically sporadic (single-molecule replication error) → muscle heteroplasmy high, blood lower. Annual Holter mandatory (AV block risk → sudden death); pacemaker/ICD if high-degree block detected.",
        },
        "clinical_terms": {
            "HCM": "Hypertrophic Cardiomyopathy — concentric LV hypertrophy; LV wall thickness >15 mm (adults) or >2SD for body surface area (children); LVOTO gradient >30 mmHg at rest or >50 mmHg on provocation = obstructive HCM; mitochondrial HCM typically symmetric (vs asymmetric septal hypertrophy in sarcomeric HCM); echo-guided beta-blocker first-line; avoid amiodarone (ABSOLUTE CI in MT-TT)",
            "CPEO": "Chronic Progressive External Ophthalmoplegia: slowly progressive bilateral ptosis + ophthalmoparesis; all EOMs eventually involved; compensatory head tilt; ptosis surgery (levator resection / frontalis sling) when ptosis >50% visual axis; KSS if CPEO + large deletion + retinal pigmentation + conduction block < age 20 yr",
            "mt_translation_fingerprint": "CI + CIV reduced; CII NORMAL — all 7 CI ND-subunits (ND1-ND6, ND4L) and 3 CIV CO-subunits (CO1-CO3) are mtDNA-encoded and require tRNA-Thr for translation; CII (SDHA-D) all nuclear → NORMAL; this pattern is diagnostic of tRNA or mt-rRNA mutation (vs nuclear OXPHOS assembly factor)",
            "KSS": "Kearns-Sayre Syndrome — large single mtDNA deletion; CPEO + cardiac conduction block (mandatory pacemaker if PR >200ms, HV >70ms, or Mobitz II/complete AV block) + retinal pigmentation; onset <20 yr; ataxia + CSF protein >100 mg/dL; PTOSIS surgery if needed; annual Holter + cardiac MRI + cardiology follow-up mandatory",
            "BTBGD": "Biotin-Thiamine-Responsive Basal Ganglia Disease (SLC19A3, OMIM #607483); presents with Leigh-like MRI + acute neurological crisis; treatable — thiamine 5–10 mg/kg/day + biotin 5–10 mg/kg/day; MANDATORY exclusion before MT-TT diagnosis in any patient with Leigh-like MRI features",
            "WPW_pattern": "Wolff-Parkinson-White pattern — short PR + delta wave on ECG; in HCM (including mitochondrial) due to accessory pathway in hypertrophied septum; symptomatic (palpitations + SVT/AF) → electrophysiology study + ablation; AVOID amiodarone for WPW in MT-TT (ABSOLUTE CI)",
        },
        "pharmacology": {
            "absolute_ci": {a[0]: a[2] for a in ABSOLUTE_CI},
            "preferred_aed": "Levetiracetam (LEV) — renal excretion; no mt-toxicity; first-line for seizures in MT-TT disease; avoid VPA (ABSOLUTE CI), phenobarbital (hepatic; cautious use), benzodiazepines (short-term adjunct only)",
            "cardiac_management": "AMIODARONE ABSOLUTE CI — use beta-blockers (metoprolol/bisoprolol) for HCM + rate control; flecainide/sotalol cautiously for arrhythmia; ICD if EF <35% or malignant arrhythmia documented; WPW → EP study + ablation; annual echo + 24h Holter; cardiac transplant for end-stage HCM/DCM",
            "emergency_protocol": "GIR 6–8 mg/kg/min IV dextrose + IV thiamine 10–20 mg/kg/dose + avoid fasting + bicarb for pH <7.2 + ICU monitoring; cardiac monitoring mandatory in MT-TT crises; avoid propofol (ABSOLUTE CI); use sevoflurane for anaesthesia",
            "anaesthetic_guidance": "Sevoflurane preferred (not propofol); regional anaesthesia if possible; GIR perioperatively; continuous cardiac monitoring (MT-TT cardiomyopathy risk); avoid amiodarone; ICU post-operative for major surgery; liaise with metabolic + cardiac teams pre-operatively",
        },
        "key_references": [
            "Hao H et al. (2005) Functional analysis of a novel tRNA(Thr) mutation in mitochondrial DNA — Biochem Biophys Res Commun (MT-TT m.15923A>G functional characterisation in cybrid cells)",
            "Taylor RW et al. (2003) A homoplasmic mitochondrial transfer ribonucleic acid mutation as a cause of maternally inherited hypertrophic cardiomyopathy — J Am Coll Cardiol (MT-TT mutations and HCM; cardiomyopathy as primary phenotype)",
            "Nishino I et al. (1996) Mitochondrial myopathy with ragged-red fibers and cytochrome c oxidase-negative fibers: comparison between the m.15915G>A and m.15923A>G mutations — Ann Neurol (MT-TT tRNA-Thr cohort)",
            "Schaefer AM et al. (2008) The epidemiology of mitochondrial disorders — past, present and future — Biochim Biophys Acta (mt-disease prevalence including tRNA-Thr mutations)",
            "Gorman GS et al. (2015) Mitochondrial diseases — Nat Rev Dis Primers (comprehensive review; tRNA-Thr mutations in cardiomyopathy context)",
            "DiMauro S & Schon EA (2003) Mitochondrial respiratory-chain diseases — NEJM (tRNA mutation mechanisms including tRNA-Thr; CPEO + cardiomyopathy spectrum)",
            "Karadimas CL et al. (2004) Diabetes mellitus and lactic acidosis in patients with mitochondrial myopathy — Neurology (mt-tRNA mutations including MT-TT in cardiomyopathy and metabolic disease context)",
            "Gorman GS et al. (2016) Prevalence of nuclear and mitochondrial DNA mutations related to adult mitochondrial disease — Ann Neurol (population-level MT-TT mutation frequency)",
        ],
    }
