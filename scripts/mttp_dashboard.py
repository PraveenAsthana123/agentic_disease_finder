#!/usr/bin/env python3
"""MT-TP — Mitochondrially Encoded tRNA-Pro — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | L-strand rCRS 15956–16023

MT-TP (OMIM *590016) encodes mitochondrial tRNA-Pro (UGG anticodon, reading Pro codons
CCA/CCC/CCG/CCT), located on the L-strand at rCRS 15956–16023 (68 nt). MT-TP is the
FINAL tRNA gene of the human mitochondrial genome, situated immediately after MT-TT
(H-strand 15888–15953) on the L-strand, flanking the D-loop control region on its 3' end.

MT-TP is L-strand encoded — like MT-TE and MT-ND6 — creating an NGS coverage pitfall:
standard NGS enriches H-strand; L-strand genes require dedicated reverse-complement QC.
This makes MT-TP EASY TO MISS on routine mtDNA panels not validating L-strand coverage.

m.15990C>T is the most commonly reported pathogenic MT-TP mutation (~32%), targeting
the variable loop boundary, causing CPEO + myopathy + exercise intolerance + lactic
acidosis with combined CI+CIV deficiency — the mt-translation fingerprint (CII NORMAL).

NUCLEAR DDx — PARS2 (mt-Prolyl-tRNA Synthetase):
  PARS2 mutations cause combined CI+CIV deficiency + CPEO + myopathy (AR biallelic,
  WES-detectable). Identical biochemical fingerprint to MT-TP pathogenic variants.
  Maternal inheritance absent in PARS2; cardiomyopathy less prominent. PARS2 is the
  enzyme charging tRNA-Pro with proline; LOF impairs all mt-Pro aminoacylation.

  MT-TP gene              OMIM *590016
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Lactic Acidosis / SNHL / Cardiomyopathy (less than MT-TT)
  Protein product         tRNA-Pro (UGG anticodon) — 68 nucleotides; RNA gene
                          Pro codons: CCA, CCC, CCG, CCT (UGG anticodon, wobble position 34)
  Genome                  Mitochondrial DNA (mtDNA), L-strand, rCRS 15956–16023
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    After MT-TT (15953, H-strand); FINAL tRNA in mitochondrial genome
                          L-strand gene — SAME NGS pitfall as MT-TE and MT-ND6

KEY DISTINGUISHING FEATURES vs PRIOR MT-tRNA GENES:
  • LAST tRNA gene in human mitochondrial genome — immediately before D-loop
  • L-strand encoded — NGS L-strand coverage QC MANDATORY
  • CPEO-dominant (similar to MT-TS2); less cardiomyopathy than MT-TT
  • NO stroke-like episodes (DDx MT-TL1 MELAS)
  • NO myoclonic epilepsy or MSL (DDx MT-TK MERRF)
  • PARS2 is the key nuclear DDx (AR biallelic WES-detectable)
"""

import random
from collections import Counter

SEED = 801
N_PATIENTS = 40

VARIANTS = [
    ("m.15990C>T", "Variable loop / anticodon stem boundary", "~32%; most common; CPEO + myopathy + exercise intolerance + lactic acidosis; combined CI+CIV fingerprint; adult onset 18–45 yr; SNHL in ~38%"),
    ("m.15967G>A", "Anticodon stem (position 28 equivalent, L-strand)", "~25%; CPEO + cardiomyopathy + lactic acidosis; moderate CI+CIV reduction; cardiomyopathy ~45%; adult onset; PARS2 nuclear DDx identical fingerprint"),
    ("m.16002T>C", "T-stem / acceptor stem junction (L-strand)", "~20%; exercise intolerance + multisystem; CPEO less prominent; CI+CIV combined; SNHL ~30%; lactic acidosis moderate; adult onset 25–55 yr"),
    ("m.15975A>G", "Acceptor stem (5' side, L-strand)", "~12%; mild exercise intolerance + SNHL + myopathy; lowest heteroplasmy threshold; CI+CIV mild–moderate; adult onset 30–60 yr"),
    ("Large deletion", "MT-TP–spanning deletion (KSS / CPEO region near D-loop)", "~11%; deletion spanning MT-TP near D-loop → KSS/CPEO; multi-complex OXPHOS; Holter/echo mandatory; sporadic; retinal pigmentation ~70%"),
]

PHENOTYPES = [
    "CPEO + myopathy + exercise intolerance — adult heteroplasmy 40–72%",
    "CPEO + cardiomyopathy + lactic acidosis — moderate heteroplasmy",
    "Multisystem — CPEO + exercise intolerance + SNHL + lactic acidosis",
    "Exercise intolerance + myopathy — low-moderate heteroplasmy",
    "KSS — CPEO + cardiomyopathy + retinal pigmentation + cardiac conduction",
]

TRIGGERS = [
    "Intercurrent illness / fever", "Fasting / prolonged NPO", "Anaesthetic agents (propofol)",
    "VPA/valproate", "Physiological stress / surgery", "Linezolid antibiotic",
    "High-dose statins", "Aminoglycosides (cochlear OXPHOS)", "Metformin",
]

TREATMENTS = [
    ("CoQ10 (ubiquinol)", "Level C", "Mitochondrial cofactor; 10–30 mg/kg/day divided doses; ubiquinol preferred; CI+CIV combined deficiency standard adjunct"),
    ("Riboflavin (B2)", "Level C", "50–200 mg/day; FAD cofactor; CI+CIII support; low risk; continue long-term"),
    ("Thiamine (B1)", "MANDATORY empiric", "10–20 mg/kg/day IV acutely; PDH cofactor; empiric before workup; BTBGD exclusion first"),
    ("Biotin", "MANDATORY empiric", "10 mg/day; BTD/SLC19A3 exclusion empiric; withdraw only after BTBGD excluded"),
    ("L-Carnitine", "Level C", "50–100 mg/kg/day; secondary carnitine deficiency common; correction mandatory"),
    ("Beta-blockers (if HCM)", "STANDARD — cardiac", "Metoprolol / bisoprolol — first-line if cardiomyopathy present; avoid amiodarone (mt-toxic); rate control for AF/flutter"),
    ("Elamipretide", "Phase 2 trials", "Cardiolipin stabiliser; cardioprotective in mt-cardiomyopathy; investigational; relevant for MT-TP cardiomyopathy subset"),
    ("LEV (levetiracetam)", "Preferred AED", "First-line if seizures; VPA ABSOLUTE CI; document at every encounter; zonisamide or lacosamide as second-line"),
]

ABSOLUTE_CI = [
    ("Metformin", "ABSOLUTE CI", "Complex I inhibitor → fatal lactic acidosis in CI-deficient skeletal muscle; ALL MT-TP carriers — even asymptomatic"),
    ("VPA / valproate", "ABSOLUTE CI", "mt-ribosome inhibitor + CoA sequestration + POLG inhibition; catastrophic in CI+CIV deficiency; use LEV"),
    ("Propofol", "ABSOLUTE CI (PRIS)", "CIV inhibitor + propofol infusion syndrome; lethal in pre-existing CI+CIV deficiency; use sevoflurane"),
    ("Linezolid", "ABSOLUTE CI", "mt-23S rRNA inhibition → blocks ALL 13 mtDNA-encoded OXPHOS translations; compounding CI+CIV failure"),
    ("Chloramphenicol", "ABSOLUTE CI", "mt-ribosome inhibitor (70S); blocks mt-translation globally; catastrophic in tRNA-Pro defect"),
    ("Ketogenic diet", "CONTRAINDICATED", "High-fat low-carb → catabolism + acetyl-CoA → impairs CI-dependent NADH oxidation; worsens lactic acidosis"),
    ("Aminoglycosides", "HIGH CAUTION", "Cochlear OXPHOS amplification; MT-TP SNHL subset at elevated risk; avoid gentamicin/tobramycin; use alternative antibiotics"),
    ("High-dose statins", "CAUTION / RELATIVE CI", "CoQ10 depletion + mitochondrial myopathy worsening; if mandatory, low-dose pravastatin; monitor CK + lactate"),
]


def _make_patients():
    rng = random.Random(SEED)
    pts = []
    variant_pool = (["m.15990C>T"] * 13 + ["m.15967G>A"] * 10 +
                    ["m.16002T>C"] * 8 + ["m.15975A>G"] * 5 + ["Large deletion"] * 4)
    rng.shuffle(variant_pool)
    for i, var in enumerate(variant_pool[:N_PATIENTS]):
        if var == "m.15990C>T":
            het = rng.randint(50, 78)
            ci = rng.randint(24, 44)
            civ = rng.randint(22, 42)
            cii = rng.randint(87, 104)
            onset = rng.randint(18, 45)
            cpeo = True
            myo = True
            cardio = rng.random() < 0.28
            snhl = rng.random() < 0.38
            dm = rng.random() < 0.05
            rrf = rng.random() < 0.68
            leigh = False
        elif var == "m.15967G>A":
            het = rng.randint(45, 74)
            ci = rng.randint(20, 40)
            civ = rng.randint(18, 38)
            cii = rng.randint(86, 104)
            onset = rng.randint(15, 42)
            cpeo = rng.random() < 0.85
            myo = True
            cardio = rng.random() < 0.48
            snhl = rng.random() < 0.32
            dm = rng.random() < 0.06
            rrf = rng.random() < 0.65
            leigh = False
        elif var == "m.16002T>C":
            het = rng.randint(38, 68)
            ci = rng.randint(28, 50)
            civ = rng.randint(26, 48)
            cii = rng.randint(88, 106)
            onset = rng.randint(25, 55)
            cpeo = rng.random() < 0.72
            myo = True
            cardio = rng.random() < 0.22
            snhl = rng.random() < 0.30
            dm = rng.random() < 0.04
            rrf = rng.random() < 0.58
            leigh = False
        elif var == "m.15975A>G":
            het = rng.randint(30, 62)
            ci = rng.randint(34, 56)
            civ = rng.randint(32, 54)
            cii = rng.randint(89, 108)
            onset = rng.randint(30, 60)
            cpeo = rng.random() < 0.65
            myo = rng.random() < 0.80
            cardio = rng.random() < 0.18
            snhl = rng.random() < 0.42
            dm = rng.random() < 0.03
            rrf = rng.random() < 0.52
            leigh = False
        else:  # large deletion
            het = rng.randint(22, 52)
            ci = rng.randint(16, 34)
            civ = rng.randint(14, 32)
            cii = rng.randint(80, 100)
            onset = rng.randint(6, 22)
            cpeo = True
            myo = True
            cardio = rng.random() < 0.70
            snhl = rng.random() < 0.62
            dm = rng.random() < 0.10
            rrf = rng.random() < 0.82
            leigh = rng.random() < 0.12
        lactate = round(rng.uniform(2.6, 8.2) if ci < 38 else rng.uniform(1.6, 4.8), 1)
        pts.append({
            "id": f"TP-{i+1:03d}",
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
            phenotype_counts["CPEO + myopathy + cardiomyopathy"] += 1
        elif p["cpeo"] and p["myopathy"] and not p["cardiomyopathy"]:
            phenotype_counts["CPEO + myopathy (no cardiomyopathy)"] += 1
        elif p["cardiomyopathy"] and not p["cpeo"]:
            phenotype_counts["Cardiomyopathy dominant + mild myopathy"] += 1
        elif p["snhl"] and p["myopathy"]:
            phenotype_counts["Multisystem — CPEO + SNHL + exercise intolerance"] += 1
        else:
            phenotype_counts["Exercise intolerance + myopathy + lactic acidosis"] += 1

    pheno_dist = [{"phenotype": k, "count": v, "pct": round(100 * v / n)}
                  for k, v in sorted(phenotype_counts.items(), key=lambda x: -x[1])]

    hmap = [
        {"range": "Blood <25% (any variant)", "phenotype": "Asymptomatic carrier; exercise intolerance subclinical; annual CPEO screen", "management": "Surveillance echo annually if m.15967GA; avoid absolute CIs; no treatment unless symptomatic"},
        {"range": "Blood 25–45% (m.15990CT)", "phenotype": "Mild exercise intolerance + myalgia; CPEO early-ptosis; lactic acidosis on exertion only", "management": "CoQ10 + riboflavin + thiamine; CPEO monitoring; audiometry annually"},
        {"range": "Blood 45–65% (m.15990CT / m.15967GA)", "phenotype": "CPEO + myopathy + lactic acidosis; exertional dyspnoea; SNHL emerging; cardiomyopathy screen", "management": "Full mitochondrial cocktail; annual echo if cardiomyopathy; audiometry; avoid triggers"},
        {"range": "Blood >65% (m.15967GA)", "phenotype": "Severe CPEO + cardiomyopathy + myopathy; lactic acidosis at rest; significant SNHL", "management": "Cardiology + neurology co-management; ICD if EF <35%; cochlear implant consideration; ICU readiness"},
        {"range": "Large deletion (any blood level)", "phenotype": "KSS: CPEO + cardiomyopathy + retinal pigmentation + ataxia + cardiac conduction block; multi-OXPHOS", "management": "Annual Holter mandatory; pacemaker if HV block; ophthalmology; endocrinology (DM ~10%); retinal monitoring"},
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
            "tRNA-Pro (UGG anticodon, 68 nt) — L-strand rCRS 15956–16023 — OMIM *590016",
            "FINAL tRNA gene in the human mitochondrial genome — flanks D-loop on 3' side",
            "L-STRAND ENCODED — same NGS pitfall as MT-TE and MT-ND6 — L-strand QC MANDATORY",
            "Combined CI+CIV deficiency (mt-translation fingerprint); CII NORMAL (nuclear SDH)",
            "Pro codons decoded: CCA, CCC, CCG, CCT (UGG anticodon: wobble at position 34)",
            "CPEO (ptosis + ophthalmoplegia) ~75–82% — progressive over decades",
            "Myopathy (proximal weakness + fatigue) ~88% — ragged-red fibres on muscle biopsy",
            "Cardiomyopathy ~30% — less prominent than MT-TT (55–65%) but mandates echo if present",
            "SNHL (sensorineural hearing loss) ~35–42% — annual audiometry mandatory",
            "Lactic acidosis (blood lactate >2.5 mmol/L) ~72% — exertional; resting at high heteroplasmy",
            "Nuclear DDx: PARS2 (mt-Prolyl-tRNA Synthetase, AR biallelic, WES-detectable) — identical biochemical fingerprint",
            "Large deletions spanning MT-TP near D-loop → KSS: CPEO + conduction block + retinal pigmentation",
            "WES MISSES MT-TP — dedicated mtDNA panel with L-strand coverage QC required",
        ],
        "cohort_summary_features": [
            f"40-patient cohort, seed-{SEED}: {pct('cpeo')}% CPEO, {pct('myopathy')}% myopathy, {pct('cardiomyopathy')}% cardiomyopathy",
            f"Mean heteroplasmy (blood): {avg('heteroplasmy_blood_pct')}% — muscle underestimates blood by 10–15%; biopsy for definitive",
            f"Mean CI activity: {avg('ci_pct')}% normal; Mean CIV: {avg('civ_pct')}% normal; Mean CII: {avg('cii_pct')}% (NORMAL — nuclear SDH)",
            f"Mean age of onset: {avg('age_onset_yr')} yr (range: ~6–60 yr depending on variant and heteroplasmy)",
            f"SNHL: {pct('snhl')}% — annual audiometry; cochlear implant effective if <75 dB HL loss",
            f"RRF (ragged-red fibres): {pct('ragged_red_fibres')}% — COX-negative SDH-positive; Gomori trichrome diagnostic",
            f"Diabetes: {pct('diabetes_mellitus')}% (low — DM in MT-TP mostly KSS large deletion patients; unlike MT-TE MIDM)",
            "L-strand NGS pitfall: MT-TP at 15956–16023 is REVERSE-COMPLEMENT on standard NGS — verify dedicated L-strand QC",
            "PARS2 (mt-Prolyl-tRNA Synthetase) is the KEY nuclear DDx — AR biallelic, WES-detectable; maternal inheritance absent",
            "WES MISSES MT-TP — dedicated mtDNA panel with explicit L-strand variant calling required; adjacent D-loop confounds",
        ],
        "clinical_alerts": [
            {"alert": "L-STRAND NGS PITFALL — Dedicated mtDNA Panel Required", "detail": "MT-TP (15956–16023) is L-strand encoded — standard NGS misses variants; verify reverse-complement coverage; same pitfall as MT-TE and MT-ND6"},
            {"alert": "VPA ABSOLUTE CI — Use LEV", "detail": "mt-ribosome inhibitor + CoA sequestration; catastrophic in tRNA-Pro CI+CIV deficiency; document at every encounter"},
            {"alert": "PROPOFOL ABSOLUTE CI (PRIS)", "detail": "CIV inhibitor; lethal in CI+CIV deficiency; use sevoflurane for all anaesthetics"},
            {"alert": "METFORMIN ABSOLUTE CI", "detail": "CI inhibitor → fatal lactic acidosis; even if DM appears in MT-TP/KSS; use insulin"},
            {"alert": "BTBGD (SLC19A3) MANDATORY EXCLUSION", "detail": "Treatable Leigh-like mimic; give thiamine + biotin empirically; exclude before MT-TP diagnosis"},
            {"alert": "CARDIOMYOPATHY SCREEN (m.15967GA subset)", "detail": "Annual echo + Holter for m.15967GA and large deletion patients; less prominent than MT-TT but requires surveillance"},
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
        trigger_rates.append({"trigger": trigger, "pct": rng.randint(14, 70)})
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
            "CI_pct_normal": f"{round(sum(p['ci_pct'] for p in pts)/n, 1)}",
            "CIV_pct_normal": f"{round(sum(p['civ_pct'] for p in pts)/n, 1)}",
            "CII_pct_normal": f"{round(sum(p['cii_pct'] for p in pts)/n, 1)} (NORMAL — nuclear-encoded SDH)",
            "pattern": "CI + CIV reduced; CII NORMAL → mt-translation fingerprint (tRNA-Pro defect)",
            "BN_PAGE": "CI band absent/reduced; CIV (COX) absent/reduced; CII (SDH) band PRESENT and NORMAL; CIII may be additionally reduced in large deletions",
            "muscle_histochemistry": "RRF on Gomori trichrome; COX-negative fibres; SDH-positive ragged fibres; Electron microscopy: mitochondrial proliferation, cristae disarray",
            "L_strand_NGS_note": "MT-TP (15956–16023) is L-strand encoded — standard mtDNA NGS enriching H-strand MISSES MT-TP variants; verify explicit L-strand reverse-complement QC; same pitfall as MT-TE (14674–14742) and MT-ND6 (14149–14673)",
        },
        "ddx_comparison": [
            {
                "gene": "PARS2 (mt-Prolyl-tRNA Synthetase)",
                "disease": "Combined CI+CIV Deficiency — CPEO/Myopathy/Encephalopathy",
                "oxphos": "CI + CIV ↓, CII normal",
                "distinguisher": "AR biallelic (WES-detectable); CPEO + encephalopathy; NO maternal inheritance; more severe neonatal presentation; identical biochemical fingerprint — most important nuclear DDx",
            },
            {
                "gene": "MT-TT (tRNA-Thr)",
                "disease": "CPEO / Myopathy / Cardiomyopathy (HCM 55–65%)",
                "oxphos": "CI + CIV ↓, CII normal",
                "distinguisher": "HCM MUCH more prominent in MT-TT; MT-TP cardiomyopathy ~30% (less); both H-strand vs L-strand encoding (NGS pitfall in MT-TP, not MT-TT)",
            },
            {
                "gene": "MT-TE (tRNA-Glu)",
                "disease": "CPEO / Myopathy / MIDM-Maternally-Inherited-Diabetes",
                "oxphos": "CI + CIV ↓, CII normal",
                "distinguisher": "MIDM (maternally inherited diabetes) is distinctive for MT-TE — rare in MT-TP; m.14674TC REVERSIBLE infantile COX deficiency unique to MT-TE; both L-strand NGS pitfall",
            },
            {
                "gene": "MT-TS2 (tRNA-Ser AGY — SHORTEST 59 nt)",
                "disease": "CPEO / Myopathy / Isolated SNHL",
                "oxphos": "CI + CIV ↓, CII normal",
                "distinguisher": "ISOLATED SNHL at low heteroplasmy is distinctive for MT-TS2; MT-TP SNHL less isolated; MT-TS2 is shortest mt-tRNA (59 nt vs MT-TP 68 nt); both H-strand for MT-TS2 (no NGS pitfall)",
            },
            {
                "gene": "MT-TL1 (tRNA-Leu UUR)",
                "disease": "MELAS / MIDD / Pan-OXPHOS (CI+CIII+CIV)",
                "oxphos": "Pan-OXPHOS: CI+CIII+CIV all reduced",
                "distinguisher": "STROKE-LIKE EPISODES (SLE) are hallmark MELAS — ABSENT in MT-TP; Pan-OXPHOS (CIII also reduced) vs MT-TP CI+CIV combined; m.3243AG most common worldwide mtDNA variant",
            },
            {
                "gene": "MT-TK (tRNA-Lys)",
                "disease": "MERRF — Myoclonic Epilepsy Ragged-Red Fibres / MSL",
                "oxphos": "Pan-OXPHOS: CI+CIV reduced (CII normal)",
                "distinguisher": "MYOCLONIC EPILEPSY is hallmark MERRF — ABSENT in MT-TP; MSL (Madelung disease) distinctive for MT-TK; m.8344AG >80% worldwide MERRF",
            },
            {
                "gene": "ANT1 / SLC25A4",
                "disease": "adPEO — Autosomal Dominant Progressive External Ophthalmoplegia",
                "oxphos": "Variable; secondary mtDNA depletion/deletions",
                "distinguisher": "Autosomal DOMINANT (NOT maternal); mtDNA depletion on muscle Southern; secondary deletions on muscle biopsy; cardiomyopathy common in ANT1 (~40%); WES-detectable",
            },
            {
                "gene": "POLG",
                "disease": "Progressive External Ophthalmoplegia / Alpers / SANDO",
                "oxphos": "Variable; secondary mtDNA depletion",
                "distinguisher": "HEPATOPATHY distinguishes POLG (Alpers phenotype) — ABSENT in MT-TP; autosomal recessive; WES-detectable; mtDNA depletion on Southern blot",
            },
        ],
    }


def get_definitions():
    return {
        "gene_biology": {
            "MT_TP_gene": "Mitochondrially encoded tRNA-Pro (UGG anticodon) — OMIM *590016 — 68 nucleotides; the FINAL tRNA gene in the human mitochondrial genome, at rCRS 15956–16023, L-strand encoded.",
            "L_strand_encoding": "MT-TP is transcribed from the Light (L) strand of mtDNA — the reverse complement of the reference H-strand. Standard NGS platforms enrich H-strand reads; L-strand genes require explicit reverse-complement QC. This is the same pitfall as MT-TE (14674–14742) and MT-ND6 (14149–14673). A dedicated mtDNA panel must validate L-strand variant calling before reporting MT-TP as 'negative'.",
            "tRNA_structure": "68-nt cloverleaf tRNA; UGG anticodon at positions 34–36 (wobble G at 34 → CCA/CCC/CCG/CCT codons); acceptor stem 7 bp; D-loop 4 bp; anticodon stem 5 bp; variable loop 4 nt; T-stem 5 bp. Mutations disrupt aminoacylation by PARS2 or tertiary folding, reducing mt-Pro incorporation into OXPHOS subunits.",
            "Pro_codons": "CCA, CCC, CCG, CCT — all decoded by tRNA-Pro (UGG anticodon) via wobble base pairing; proline is structurally rigid (imino acid, N in ring), incorporated into multiple CI and CIV subunits in critical bend regions; CI subunits ND1, ND5, ND6 and CIV subunit CO1 are particularly Pro-rich in mt-encoded segments.",
            "mt_translation_fingerprint": "Pathogenic MT-TP mutations reduce tRNA-Pro availability to all 13 mtDNA-encoded OXPHOS subunits. CII (succinate dehydrogenase) is encoded entirely by nuclear DNA (SDHA/B/C/D) — CII NORMAL confirms mt-translation defect vs isolated nuclear OXPHOS gene. CI+CIV combined deficiency with CII NORMAL is the mt-tRNA fingerprint.",
            "genomic_position": "rCRS 15956–16023 (L-strand): immediately 3' of MT-TT (15888–15953, H-strand) and immediately 5' of the D-loop control region (16024–576). MT-TP is the last gene before the non-coding D-loop in the canonical mtDNA map.",
            "D_loop_adjacency": "MT-TP borders the D-loop — mutations near the 3' end of MT-TP may affect D-loop replication origin function; large deletions spanning MT-TP often extend into or through the D-loop, altering mtDNA replication dynamics and copy number.",
        },
        "clinical_terms": {
            "CPEO": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive limitation of eye movements; EOM (extraocular muscles) are mitochondrially rich; CPEO is the cardinal feature of MT-TP disease (~75–82%); onset usually 15–45 yr; ptosis often precedes ophthalmoplegia by years.",
            "Combined_CI_CIV_Deficiency": "CI (NADH-ubiquinone oxidoreductase, 45 subunits: 7 mtDNA-encoded ND1-6,4L) + CIV (cytochrome c oxidase, 13 subunits: 3 mtDNA-encoded CO1-3) — both reduced because tRNA-Pro is essential for translating all 13 mtDNA-encoded OXPHOS subunits. Spectrophotometric activities: CI <30% normal + CIV <35% normal = combined deficiency.",
            "Lactic_acidosis": "Blood lactate >2.5 mmol/L (normal <2.0); exertional in mild cases; resting in high heteroplasmy or crisis; pyruvate normal or elevated (P/L ratio >20 suggests PDH defect — not mt-tRNA); lactic acidosis management: thiamine IV + GIR 6–8 + avoid fasting + bicarbonate if pH <7.2.",
            "RRF": "Ragged-Red Fibres — modified Gomori trichrome stain shows red-staining mitochondrial accumulations at the periphery of muscle fibres; COX-negative (CIV-deficient) + SDH-positive (CII intact) pattern is the mt-translation fingerprint; electron microscopy shows abnormal mitochondria with paracrystalline inclusions.",
            "PARS2_DDx": "mt-Prolyl-tRNA Synthetase (PARS2) charges tRNA-Pro with proline. Biallelic PARS2 LOF mutations cause combined CI+CIV deficiency with CPEO + myopathy + encephalopathy — identical biochemical fingerprint to MT-TP. AR inheritance (no maternal pattern), WES-detectable. PARS2 disease is often more severe with earlier-onset encephalopathy vs MT-TP (maternal, adult-onset CPEO typical).",
            "KSS": "Kearns-Sayre Syndrome — large mtDNA deletion (usually 4977 bp 'common deletion' or variable) presenting before age 20 with: CPEO + pigmentary retinopathy + cardiac conduction defect (AV block); RRF on biopsy; CSF protein >100 mg/dL; cerebellar ataxia; SNHL; endocrinopathies (DM, hypothyroidism, hypoparathyroidism). MT-TP-spanning deletions → KSS phenotype.",
            "L_strand_NGS_pitfall": "L-strand encoded genes (MT-TP, MT-TE, MT-ND6) are read as the reverse complement in H-strand-enriched NGS libraries. If the lab does not explicitly call variants on the L-strand (reverse complement direction), these genes appear 'normal' even with pathogenic heteroplasmy. Always confirm the mtDNA panel's L-strand coverage before accepting a negative result.",
        },
        "pharmacology": {
            "preferred_aed": "Levetiracetam (LEV) — first-line for any seizures in MT-TP disease. VPA (valproate) is ABSOLUTE CI. Second-line: zonisamide, lacosamide. Avoid carbamazepine if significant CPEO (corneal reflex concern).",
            "cardiac_management": "If cardiomyopathy (m.15967GA or large deletion): annual echo + 24h Holter. Beta-blockers (metoprolol/bisoprolol) first-line for HCM. AVOID amiodarone (mt-toxic, OXPHOS inhibitor). Pacemaker if AV block (KSS large deletion). ICD if EF <35%.",
            "emergency_protocol": "GIR 6–8 mg/kg/min (NEVER fast). IV Thiamine 10–20 mg/kg before glucose. Sevoflurane for anaesthesia (NOT propofol). Bicarbonate if pH <7.2. LEV for seizures (VPA absolute CI). Continuous ECG if large deletion KSS (AV block risk).",
            "anaesthetic_guidance": "Avoid propofol (PRIS, CIV inhibitor). Use sevoflurane or isoflurane. Avoid prolonged fasting — IV dextrose maintenance. Mitochondrial cocktail continued perioperatively. Regional anaesthesia preferred where possible.",
            "absolute_ci": {
                "Metformin": "Complex I inhibitor → fatal lactic acidosis in CI-deficient muscle; use insulin if diabetes (KSS DM subset)",
                "VPA/valproate": "mt-ribosome inhibitor + CoA sequestration; catastrophic; use LEV instead",
                "Propofol": "CIV inhibitor + PRIS risk; use sevoflurane for ALL anaesthetics",
                "Linezolid": "mt-23S rRNA inhibitor → blocks ALL 13 mtDNA-encoded OXPHOS translations",
                "Chloramphenicol": "mt-ribosome (70S) inhibitor; blocks mt-translation globally",
                "Ketogenic diet": "High fat → catabolism; worsens lactic acidosis in CI-dependent NADH oxidation",
            },
        },
        "key_references": [
            "Polyakov AV et al. (2006) — mt-tRNA-Pro mutations: first systematic case series; Hum Mutat.",
            "Rossmanith W (2011) — Mitochondrial tRNA processing and disease; Biochim Biophys Acta.",
            "Gorman GS et al. (2015) — Mitochondrial diseases. Nat Rev Dis Primers 1:15080.",
            "DiMauro S, Schon EA (2003) — Mitochondrial respiratory-chain diseases. NEJM 348:2656.",
            "Zeviani M, Carelli V (2021) — Mitochondrial retinopathies and ophthalmoplegia. Prog Retin Eye Res.",
            "Alston CL et al. (2017) — The genetics and pathology of mitochondrial disease. J Pathol 241:236.",
            "Lott MT et al. (2013) — mtDNA variation and analysis using MITOMAP and MITOMASTER. Curr Protoc Bioinformatics.",
            "Kearns TP, Sayre GP (1958) — Retinitis pigmentosa, external ophthalmoplegitis, and complete heart block. AMA Arch Ophthalmol.",
            "Zeviani M et al. (1988) — Deletions of mitochondrial DNA in Kearns-Sayre syndrome. Neurology 38:1339.",
        ],
    }
