#!/usr/bin/env python3
"""MT-TF — Mitochondrially Encoded tRNA-Phe — CPEO / Myopathy / Exercise Intolerance
Combined CI+CIV Deficiency (mt-translation fingerprint) | H-strand rCRS 577–647

MT-TF (OMIM *590070) encodes mitochondrial tRNA-Phe (GAA anticodon, reading Phe codons
UUU/UUC), located on the H-strand at rCRS 577–647 (71 nt). MT-TF is the FIRST tRNA gene
of the human mitochondrial genome, situated immediately after the D-loop control region
(rCRS 16024–576) — the opening gene of the H-strand transcript.

Phe codons (UUU, UUC) are decoded by tRNA-Phe with the GAA anticodon (wobble U at
position 34 reading both UUU and UUC). Phenylalanine is incorporated into multiple
CI and CIV subunits; pathogenic MT-TF mutations reduce tRNA-Phe availability to all
13 mtDNA-encoded OXPHOS subunits, producing the combined CI+CIV deficiency fingerprint
(CII NORMAL — nuclear SDH unaffected).

m.611T>C is the most commonly reported pathogenic MT-TF variant (~30%), targeting the
anticodon loop/stem junction, causing CPEO + myopathy + exercise intolerance + lactic
acidosis with combined CI+CIV deficiency — the mt-translation fingerprint.

NUCLEAR DDx — FARS2 (mt-Phenylalanyl-tRNA Synthetase):
  FARS2 biallelic mutations cause mt-Phe aminoacylation failure — identical biochemical
  fingerprint (CI+CIV deficiency) but DRAMATICALLY DIFFERENT phenotype: infantile-onset
  epileptic encephalopathy (PEBEI — Pontine Tegmental Cap Dysplasia / infantile-onset
  epileptic encephalopathy with Leigh-like MRI). FARS2 is AR (WES-detectable), manifests
  in neonates/infants, NOT as adult CPEO. FARS2 is the most important nuclear DDx because
  the biochemical fingerprint is identical but management differs markedly (NEO vs ADULT).

  MT-TF gene              OMIM *590070
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Lactic Acidosis / SNHL / Cardiomyopathy (mild-moderate)
  Protein product         tRNA-Phe (GAA anticodon) — 71 nucleotides; RNA gene
                          Phe codons: UUU, UUC (GAA anticodon, wobble U at position 34)
  Genome                  Mitochondrial DNA (mtDNA), H-strand, rCRS 577–647
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    After D-loop (576); FIRST tRNA in mitochondrial genome

KEY DISTINGUISHING FEATURES vs PRIOR MT-tRNA GENES:
  • FIRST tRNA gene in human mitochondrial genome — immediately after D-loop
  • H-strand encoded (no L-strand NGS pitfall — contrast with MT-TE and MT-TP)
  • CPEO + myopathy dominant (similar to MT-TS2 and MT-TP); less cardiomyopathy than MT-TT
  • FARS2 nuclear DDx causes NEONATAL EPILEPTIC ENCEPHALOPATHY — NOT adult CPEO (key age/phenotype DDx)
  • NO stroke-like episodes (DDx MT-TL1 MELAS)
  • NO myoclonic epilepsy or MSL (DDx MT-TK MERRF)
  • NO MIDM maternally inherited diabetes (DDx MT-TE)
  • Adjacent to D-loop — large deletions extending into D-loop disrupt replication origins
"""

import random
from collections import Counter

SEED = 803
N_PATIENTS = 40

VARIANTS = [
    ("m.611T>C", "Anticodon loop / stem junction (position 34–35 boundary)", "~30%; most common; CPEO + myopathy + exercise intolerance + lactic acidosis; combined CI+CIV fingerprint; adult onset 20–48 yr; SNHL in ~35%"),
    ("m.618T>C", "Variable loop (position 47a)", "~25%; CPEO + cardiomyopathy + exercise intolerance; moderate CI+CIV reduction; cardiomyopathy ~42%; adult onset; FARS2 nuclear DDx identical fingerprint"),
    ("m.622G>A", "T-stem (position 54–55 region)", "~20%; multisystem — CPEO + exercise intolerance + lactic acidosis; Leigh-like MRI at high heteroplasmy; CI+CIV combined; SNHL ~28%; adult-to-childhood onset"),
    ("m.628T>C", "T-loop / acceptor stem junction (position 58–60 region)", "~12%; mild exercise intolerance + SNHL + myopathy; lowest heteroplasmy threshold; CI+CIV mild–moderate; adult onset 28–62 yr"),
    ("Large deletion", "MT-TF–spanning deletion (KSS / CPEO region; D-loop adjacent)", "~13%; deletion spanning MT-TF → KSS/CPEO; multi-complex OXPHOS; retinal pigmentation ~68%; cardiac conduction block; sporadic; D-loop adjacency may reduce mtDNA copy number"),
]

PHENOTYPES = [
    "CPEO + myopathy + exercise intolerance — adult heteroplasmy 42–75%",
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
    ("LEV (levetiracetam)", "Preferred AED", "First-line if seizures; VPA ABSOLUTE CI; document at every encounter; zonisamide or lacosamide as second-line"),
]

ABSOLUTE_CI = [
    ("Metformin", "ABSOLUTE CI", "Complex I inhibitor → fatal lactic acidosis in CI-deficient skeletal muscle; ALL MT-TF carriers — even asymptomatic"),
    ("VPA / valproate", "ABSOLUTE CI", "mt-ribosome inhibitor + CoA sequestration + POLG inhibition; catastrophic in CI+CIV deficiency; use LEV"),
    ("Propofol", "ABSOLUTE CI (PRIS)", "CIV inhibitor + propofol infusion syndrome; lethal in pre-existing CI+CIV deficiency; use sevoflurane"),
    ("Linezolid", "ABSOLUTE CI", "mt-23S rRNA inhibition → blocks ALL 13 mtDNA-encoded OXPHOS translations; compounding CI+CIV failure"),
    ("Chloramphenicol", "ABSOLUTE CI", "mt-ribosome inhibitor (70S); blocks mt-translation globally; catastrophic in tRNA-Phe defect"),
    ("Ketogenic diet", "CONTRAINDICATED", "High-fat low-carb → catabolism + acetyl-CoA → impairs CI-dependent NADH oxidation; worsens lactic acidosis"),
    ("Aminoglycosides", "HIGH CAUTION", "Cochlear OXPHOS amplification; MT-TF SNHL subset at elevated risk; avoid gentamicin/tobramycin; use alternative antibiotics"),
    ("High-dose statins", "CAUTION / RELATIVE CI", "CoQ10 depletion + mitochondrial myopathy worsening; if mandatory, low-dose pravastatin; monitor CK + lactate"),
]


def _make_patients():
    rng = random.Random(SEED)
    pts = []
    variant_pool = (["m.611T>C"] * 12 + ["m.618T>C"] * 10 +
                    ["m.622G>A"] * 8 + ["m.628T>C"] * 5 + ["Large deletion"] * 5)
    rng.shuffle(variant_pool)
    for i, var in enumerate(variant_pool[:N_PATIENTS]):
        if var == "m.611T>C":
            het = rng.randint(48, 78)
            ci = rng.randint(22, 42)
            civ = rng.randint(20, 40)
            cii = rng.randint(88, 105)
            onset = rng.randint(20, 48)
            cpeo = True
            myo = True
            cardio = rng.random() < 0.25
            snhl = rng.random() < 0.35
            dm = rng.random() < 0.04
            rrf = rng.random() < 0.65
            leigh = False
        elif var == "m.618T>C":
            het = rng.randint(44, 74)
            ci = rng.randint(18, 38)
            civ = rng.randint(16, 36)
            cii = rng.randint(86, 104)
            onset = rng.randint(16, 44)
            cpeo = rng.random() < 0.88
            myo = True
            cardio = rng.random() < 0.45
            snhl = rng.random() < 0.30
            dm = rng.random() < 0.05
            rrf = rng.random() < 0.62
            leigh = False
        elif var == "m.622G>A":
            het = rng.randint(36, 70)
            ci = rng.randint(26, 48)
            civ = rng.randint(24, 46)
            cii = rng.randint(87, 106)
            onset = rng.randint(10, 50)
            cpeo = rng.random() < 0.75
            myo = True
            cardio = rng.random() < 0.20
            snhl = rng.random() < 0.28
            dm = rng.random() < 0.04
            rrf = rng.random() < 0.58
            leigh = rng.random() < 0.22
        elif var == "m.628T>C":
            het = rng.randint(28, 60)
            ci = rng.randint(32, 54)
            civ = rng.randint(30, 52)
            cii = rng.randint(89, 108)
            onset = rng.randint(28, 62)
            cpeo = rng.random() < 0.68
            myo = rng.random() < 0.82
            cardio = rng.random() < 0.16
            snhl = rng.random() < 0.40
            dm = rng.random() < 0.03
            rrf = rng.random() < 0.50
            leigh = False
        else:  # large deletion
            het = rng.randint(20, 50)
            ci = rng.randint(14, 32)
            civ = rng.randint(12, 30)
            cii = rng.randint(80, 100)
            onset = rng.randint(5, 20)
            cpeo = True
            myo = True
            cardio = rng.random() < 0.72
            snhl = rng.random() < 0.65
            dm = rng.random() < 0.12
            rrf = rng.random() < 0.84
            leigh = rng.random() < 0.14
        lactate = round(rng.uniform(2.8, 8.4) if ci < 38 else rng.uniform(1.8, 5.0), 1)
        pts.append({
            "id": f"TF-{i+1:03d}",
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
        {"range": "Blood <25% (any variant)", "phenotype": "Asymptomatic carrier; exercise intolerance subclinical; annual CPEO screen mandatory", "management": "Surveillance echo annually if m.618TC; avoid absolute CIs; no treatment unless symptomatic"},
        {"range": "Blood 25–45% (m.611TC)", "phenotype": "Mild exercise intolerance + myalgia; CPEO early-ptosis; lactic acidosis on exertion only", "management": "CoQ10 + riboflavin + thiamine; CPEO monitoring; audiometry annually"},
        {"range": "Blood 45–65% (m.611TC / m.618TC)", "phenotype": "CPEO + myopathy + lactic acidosis; exertional dyspnoea; SNHL emerging; cardiomyopathy screen", "management": "Full mitochondrial cocktail; annual echo if m.618TC; audiometry; avoid triggers"},
        {"range": "Blood >65% (m.618TC)", "phenotype": "Severe CPEO + cardiomyopathy + myopathy; lactic acidosis at rest; significant SNHL", "management": "Cardiology + neurology co-management; ICD if EF <35%; cochlear implant consideration; ICU readiness"},
        {"range": "Large deletion (any blood level)", "phenotype": "KSS: CPEO + cardiomyopathy + retinal pigmentation + ataxia + cardiac conduction block; multi-OXPHOS; D-loop adjacency", "management": "Annual Holter mandatory; pacemaker if HV block; ophthalmology; endocrinology (DM ~12%); mtDNA copy number monitoring"},
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
            "tRNA-Phe (GAA anticodon, 71 nt) — H-strand rCRS 577–647 — OMIM *590070",
            "FIRST tRNA gene in the human mitochondrial genome — immediately after D-loop (rCRS 576)",
            "H-STRAND ENCODED — no L-strand NGS pitfall (contrast with MT-TE, MT-TP, MT-ND6)",
            "Combined CI+CIV deficiency (mt-translation fingerprint); CII NORMAL (nuclear SDH)",
            "Phe codons decoded: UUU, UUC (GAA anticodon; wobble U at position 34)",
            "CPEO (ptosis + ophthalmoplegia) ~78–85% — progressive over decades",
            "Myopathy (proximal weakness + fatigue) ~90% — ragged-red fibres on muscle biopsy",
            "Cardiomyopathy ~30% — m.618TC variant most prominent; annual echo if cardiomyopathy",
            "SNHL (sensorineural hearing loss) ~32–40% — annual audiometry mandatory",
            "Lactic acidosis (blood lactate >2.5 mmol/L) ~70% — exertional; resting at high heteroplasmy",
            "Nuclear DDx: FARS2 (mt-Phenylalanyl-tRNA Synthetase) — NEONATAL epileptic encephalopathy (NOT adult CPEO); AR biallelic; WES-detectable",
            "Large deletions spanning MT-TF near D-loop → KSS: CPEO + conduction block + retinal pigmentation + reduced mtDNA copy number",
            "WES MISSES MT-TF — dedicated mtDNA panel required (H-strand but still absent from WES capture)",
            "D-loop adjacency (3' of D-loop at rCRS 576): large deletions extending from MT-TF into D-loop can disrupt mtDNA replication origins",
        ],
        "cohort_summary_features": [
            f"40-patient cohort, seed-{SEED}: {pct('cpeo')}% CPEO, {pct('myopathy')}% myopathy, {pct('cardiomyopathy')}% cardiomyopathy",
            f"Mean heteroplasmy (blood): {avg('heteroplasmy_blood_pct')}% — muscle underestimates blood by 10–15%; biopsy for definitive heteroplasmy",
            f"Mean CI activity: {avg('ci_pct')}% normal; Mean CIV: {avg('civ_pct')}% normal; Mean CII: {avg('cii_pct')}% (NORMAL — nuclear SDH)",
            f"Mean age of onset: {avg('age_onset_yr')} yr (range: ~5–62 yr depending on variant and heteroplasmy)",
            f"SNHL: {pct('snhl')}% — annual audiometry; cochlear implant effective if <75 dB HL loss",
            f"RRF (ragged-red fibres): {pct('ragged_red_fibres')}% — COX-negative SDH-positive; Gomori trichrome diagnostic",
            f"Diabetes: {pct('diabetes_mellitus')}% (low — DM mainly in KSS large deletion patients; unlike MT-TE MIDM)",
            "FIRST tRNA gene: MT-TF (577–647) is the opening H-strand gene after the D-loop control region — unique positional context in mtDNA",
            "FARS2 nuclear DDx: AR mt-Phe-tRNA synthetase; causes NEONATAL epileptic encephalopathy (PEBEI) — phenotypically opposite to adult MT-TF CPEO",
            "WES MISSES MT-TF — dedicated mtDNA panel required; H-strand but not captured by exome (RNA gene, not protein-coding)",
        ],
        "clinical_alerts": [
            {"alert": "VPA ABSOLUTE CI — Use LEV", "detail": "mt-ribosome inhibitor + CoA sequestration; catastrophic in tRNA-Phe CI+CIV deficiency; document at every encounter"},
            {"alert": "PROPOFOL ABSOLUTE CI (PRIS)", "detail": "CIV inhibitor; lethal in CI+CIV deficiency; use sevoflurane for all anaesthetics"},
            {"alert": "METFORMIN ABSOLUTE CI", "detail": "CI inhibitor → fatal lactic acidosis; even if DM appears in MT-TF/KSS; use insulin"},
            {"alert": "BTBGD (SLC19A3) MANDATORY EXCLUSION", "detail": "Treatable Leigh-like mimic; give thiamine + biotin empirically; exclude before MT-TF diagnosis"},
            {"alert": "FARS2 NUCLEAR DDx — NEONATAL PHENOTYPE DISTINCTION", "detail": "FARS2 biallelic causes infantile epileptic encephalopathy (not adult CPEO); biochemical fingerprint identical CI+CIV; WES-detectable; ALWAYS consider FARS2 in infant/neonatal presentations before mtDNA"},
            {"alert": "CARDIOMYOPATHY SCREEN (m.618TC + large deletion)", "detail": "Annual echo + Holter for m.618TC and large deletion patients; cardiomyopathy ~30% overall; less than MT-TT but requires surveillance"},
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
        trigger_rates.append({"trigger": trigger, "pct": rng.randint(12, 68)})
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
            "pattern": "CI + CIV reduced; CII NORMAL → mt-translation fingerprint (tRNA-Phe defect)",
            "BN_PAGE": "CI band absent/reduced; CIV (COX) absent/reduced; CII (SDH) band PRESENT and NORMAL; CIII may be additionally reduced in large deletions",
            "muscle_histochemistry": "RRF on Gomori trichrome; COX-negative fibres; SDH-positive ragged fibres; Electron microscopy: mitochondrial proliferation, cristae disarray",
            "H_strand_note": "MT-TF (577–647) is H-strand encoded — no L-strand NGS pitfall (contrast with MT-TE, MT-TP, MT-ND6). However, WES still misses MT-TF because it is an RNA gene not in protein-coding exome capture. Dedicated mtDNA panel required.",
        },
        "ddx_comparison": [
            {
                "gene": "FARS2 (mt-Phenylalanyl-tRNA Synthetase)",
                "disease": "Infantile Epileptic Encephalopathy — PEBEI / Leigh-like MRI",
                "oxphos": "CI + CIV ↓, CII normal",
                "distinguisher": "AR biallelic (WES-detectable); NEONATAL/INFANTILE onset — NOT adult CPEO; severe epileptic encephalopathy + Leigh-like MRI + spastic diplegia; NO CPEO; identical biochemical fingerprint — most important nuclear DDx; maternal inheritance ABSENT in FARS2",
            },
            {
                "gene": "MT-TP (tRNA-Pro — LAST tRNA in mt-genome)",
                "disease": "CPEO / Myopathy / Exercise Intolerance",
                "oxphos": "CI + CIV ↓, CII normal",
                "distinguisher": "MT-TP is L-strand encoded (NGS pitfall) — MT-TF is H-strand (no pitfall); MT-TP is FINAL tRNA (15956–16023) vs MT-TF FIRST (577–647); similar phenotypes but NGS detection differs markedly; PARS2 DDx for MT-TP vs FARS2 for MT-TF",
            },
            {
                "gene": "MT-TT (tRNA-Thr)",
                "disease": "CPEO / Myopathy / Cardiomyopathy (HCM 55–65%)",
                "oxphos": "CI + CIV ↓, CII normal",
                "distinguisher": "HCM MUCH more prominent in MT-TT (55–65%); MT-TF cardiomyopathy ~30%; AMIODARONE absolute CI in MT-TT; both H-strand encoded; TARS2 DDx for MT-TT vs FARS2 for MT-TF",
            },
            {
                "gene": "MT-TE (tRNA-Glu)",
                "disease": "CPEO / Myopathy / MIDM-Maternally-Inherited-Diabetes",
                "oxphos": "CI + CIV ↓, CII normal",
                "distinguisher": "MIDM (maternally inherited diabetes) is distinctive for MT-TE — rare in MT-TF; m.14674TC REVERSIBLE infantile COX deficiency unique to MT-TE; MT-TE is L-strand encoded (NGS pitfall) — MT-TF is H-strand",
            },
            {
                "gene": "MT-TS2 (tRNA-Ser AGY — SHORTEST 59 nt)",
                "disease": "CPEO / Myopathy / Isolated SNHL",
                "oxphos": "CI + CIV ↓, CII normal",
                "distinguisher": "ISOLATED SNHL at low heteroplasmy distinctive for MT-TS2; MT-TF SNHL less isolated; MT-TS2 is SHORTEST mt-tRNA (59 nt vs MT-TF 71 nt); SARS2 DDx for MT-TS2 vs FARS2 for MT-TF",
            },
            {
                "gene": "MT-TL1 (tRNA-Leu UUR)",
                "disease": "MELAS / MIDD / Pan-OXPHOS (CI+CIII+CIV)",
                "oxphos": "Pan-OXPHOS: CI+CIII+CIV all reduced",
                "distinguisher": "STROKE-LIKE EPISODES (SLE) are hallmark MELAS — ABSENT in MT-TF; Pan-OXPHOS (CIII also reduced) vs MT-TF CI+CIV combined; m.3243AG most common worldwide mtDNA variant",
            },
            {
                "gene": "MT-TK (tRNA-Lys)",
                "disease": "MERRF — Myoclonic Epilepsy Ragged-Red Fibres / MSL",
                "oxphos": "Pan-OXPHOS: CI+CIV reduced (CII normal)",
                "distinguisher": "MYOCLONIC EPILEPSY is hallmark MERRF — ABSENT in MT-TF; MSL (Madelung disease) distinctive for MT-TK; m.8344AG >80% worldwide MERRF",
            },
            {
                "gene": "ANT1 / SLC25A4",
                "disease": "adPEO — Autosomal Dominant Progressive External Ophthalmoplegia",
                "oxphos": "Variable; secondary mtDNA depletion/deletions",
                "distinguisher": "Autosomal DOMINANT (NOT maternal); secondary mtDNA deletions on muscle Southern; cardiomyopathy common; WES-detectable",
            },
            {
                "gene": "POLG",
                "disease": "Progressive External Ophthalmoplegia / Alpers / SANDO",
                "oxphos": "Variable; secondary mtDNA depletion",
                "distinguisher": "HEPATOPATHY distinguishes POLG (Alpers phenotype) — ABSENT in MT-TF; autosomal recessive; WES-detectable; mtDNA depletion on Southern blot",
            },
        ],
    }


def get_definitions():
    return {
        "gene_biology": {
            "MT_TF_gene": "Mitochondrially encoded tRNA-Phe (GAA anticodon) — OMIM *590070 — 71 nucleotides; the FIRST tRNA gene in the human mitochondrial genome, at rCRS 577–647, H-strand encoded. Immediately follows the D-loop control region (rCRS 16024–576).",
            "genomic_position_FIRST": "MT-TF (H-strand rCRS 577–647) is the opening gene of the H-strand mt-transcription unit — the first gene after the D-loop. This positional uniqueness means: (1) Large deletions extending from MT-TF backward into the D-loop can disrupt the Light-strand replication origin (OL) and Heavy-strand origin region (OH), potentially reducing mtDNA copy number. (2) MT-TF variants that lower tRNA-Phe affect all downstream H-strand mt-translation.",
            "tRNA_structure": "71-nt cloverleaf tRNA; GAA anticodon at positions 34–36 (wobble U at 34 → UUU/UUC Phe codons); acceptor stem 7 bp; D-stem 4 bp; anticodon stem 5 bp; variable loop 5 nt; T-stem 5 bp. Mutations disrupt aminoacylation by FARS2 (mt-Phe-tRNA synthetase) or tertiary folding, reducing mt-Phe incorporation into OXPHOS subunits.",
            "Phe_codons": "UUU and UUC — both decoded by tRNA-Phe (GAA anticodon) via wobble base pairing (U34 → U/C at third codon position); phenylalanine is an aromatic amino acid incorporated into hydrophobic transmembrane segments of multiple CI and CIV subunits; CI subunits ND1, ND2, ND4, ND5 and CIV subunit CO1 are particularly Phe-dense in their mt-encoded transmembrane helices.",
            "mt_translation_fingerprint": "Pathogenic MT-TF mutations reduce tRNA-Phe availability to all 13 mtDNA-encoded OXPHOS subunits. CII (succinate dehydrogenase) is encoded entirely by nuclear DNA (SDHA/B/C/D) — CII NORMAL confirms mt-translation defect vs isolated nuclear OXPHOS gene. CI+CIV combined deficiency with CII NORMAL is the mt-tRNA translation fingerprint.",
            "H_strand_encoding": "MT-TF is transcribed from the Heavy (H) strand — the standard orientation for mtDNA sequencing. Unlike MT-TE, MT-TP, and MT-ND6 (L-strand genes), there is no H-strand NGS coverage pitfall for MT-TF. However, MT-TF is still an RNA gene absent from WES exome capture; a dedicated mtDNA sequencing panel is required for detection.",
            "D_loop_adjacency": "MT-TF borders the D-loop on its 5' side — the D-loop contains the Heavy strand promoter (HSP), Light strand promoter (LSP), and the origin of Heavy strand replication (OH). Large deletions that extend from MT-TF into the D-loop can: (1) delete OH → impair mtDNA replication → reduce copy number; (2) delete HSP/LSP → reduce transcription of H/L strand genes; (3) produce compound heteroplasmy across the D-loop boundary.",
        },
        "clinical_terms": {
            "CPEO": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive limitation of eye movements; EOM (extraocular muscles) are mitochondrially rich; CPEO is the cardinal feature of MT-TF disease (~78–85%); onset usually 15–48 yr; ptosis often precedes ophthalmoplegia by years.",
            "Combined_CI_CIV_Deficiency": "CI (NADH-ubiquinone oxidoreductase, 45 subunits: 7 mtDNA-encoded ND1-6,4L) + CIV (cytochrome c oxidase, 13 subunits: 3 mtDNA-encoded CO1-3) — both reduced because tRNA-Phe is essential for translating all 13 mtDNA-encoded OXPHOS subunits. Spectrophotometric activities: CI <30% normal + CIV <35% normal = combined deficiency.",
            "Lactic_acidosis": "Blood lactate >2.5 mmol/L (normal <2.0); exertional in mild cases; resting in high heteroplasmy or crisis; pyruvate normal or elevated; management: thiamine IV + GIR 6–8 + avoid fasting + bicarbonate if pH <7.2.",
            "RRF": "Ragged-Red Fibres — modified Gomori trichrome stain shows red-staining mitochondrial accumulations at the periphery of muscle fibres; COX-negative (CIV-deficient) + SDH-positive (CII intact) pattern is the mt-translation fingerprint; electron microscopy shows abnormal mitochondria with paracrystalline inclusions.",
            "FARS2_DDx": "mt-Phenylalanyl-tRNA Synthetase (FARS2) charges tRNA-Phe with phenylalanine. Biallelic FARS2 LOF mutations cause infantile-onset epileptic encephalopathy (PEBEI — also called Alpers-like with spasticity), NOT adult CPEO. FARS2 disease is AR (WES-detectable), manifests in neonates/infants with seizures, spastic diplegia, Leigh-like MRI, lactic acidosis, and rapid neurological deterioration. The biochemical fingerprint (CI+CIV deficiency) is identical to MT-TF but the phenotype is strikingly different — NEONATAL vs ADULT ONSET is the most critical distinguishing feature.",
            "KSS": "Kearns-Sayre Syndrome — large mtDNA deletion (usually 4977 bp 'common deletion' or variable) presenting before age 20 with: CPEO + pigmentary retinopathy + cardiac conduction defect (AV block); RRF on biopsy; CSF protein >100 mg/dL; cerebellar ataxia; SNHL; endocrinopathies. MT-TF-spanning deletions → KSS phenotype; D-loop adjacency means MT-TF deletions may also reduce mtDNA copy number.",
        },
        "pharmacology": {
            "preferred_aed": "Levetiracetam (LEV) — first-line for any seizures in MT-TF disease. VPA (valproate) is ABSOLUTE CI. Second-line: zonisamide, lacosamide. Note: distinguish MT-TF seizures (rare, low heteroplasmy) from FARS2 epileptic encephalopathy (prominent, neonatal).",
            "cardiac_management": "If cardiomyopathy (m.618TC or large deletion): annual echo + 24h Holter. Beta-blockers (metoprolol/bisoprolol) first-line for HCM. AVOID amiodarone (mt-toxic, OXPHOS inhibitor). Pacemaker if AV block (KSS large deletion). ICD if EF <35%.",
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
            "Rossmanith W, Tullo A, Potuschak T et al. (1995) — Human mitochondrial tRNA processing. J Biol Chem 270:12885.",
            "Schaefer AM, McFarland R et al. (2008) — Prevalence of mitochondrial DNA disease in adults. Ann Neurol 63:35.",
            "Elo JM, Yadavalli SS et al. (2012) — Mitochondrial phenylalanyl-tRNA synthetases and PEBEI disease; FASEB J.",
            "Alston CL, Rocha MC et al. (2017) — The genetics and pathology of mitochondrial disease. J Pathol 241:236.",
            "Gorman GS, Chinnery PF et al. (2016) — Mitochondrial diseases. Nat Rev Dis Primers 2:16080.",
            "DiMauro S, Schon EA (2003) — Mitochondrial respiratory-chain diseases. NEJM 348:2656.",
            "Zeviani M, Di Donato S (2004) — Mitochondrial disorders. Brain 127:2153.",
            "Lott MT, Leipzig JN et al. (2013) — mtDNA variation and analysis using MITOMAP and MITOMASTER. Curr Protoc Bioinformatics.",
            "Kearns TP, Sayre GP (1958) — Retinitis pigmentosa, external ophthalmoplegitis, and complete heart block. AMA Arch Ophthalmol.",
        ],
    }
