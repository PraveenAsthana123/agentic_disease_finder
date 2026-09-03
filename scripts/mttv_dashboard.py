#!/usr/bin/env python3
"""MT-TV — Mitochondrially Encoded tRNA-Val — CPEO / Myopathy / Exercise Intolerance / SNHL
Combined CI+CIV Deficiency (mt-translation fingerprint) | H-strand rCRS 1602–1670

MT-TV (OMIM *590105) encodes mitochondrial tRNA-Val (UAC anticodon, reading Val codons
GUU/GUC/GUA/GUG), located on the H-strand at rCRS 1602–1670 (69 nt). MT-TV is the
SECOND tRNA gene of the human mitochondrial genome, situated between the 12S rRNA
(rCRS 648–1601) and 16S rRNA (rCRS 1671–3229) genes — embedded within the two ribosomal
RNA genes at the start of the H-strand transcript.

Val codons (GUU, GUC, GUA, GUG) are decoded by tRNA-Val with the UAC anticodon (wobble
U at position 34 reads GUU and GUC; modified U reads GUA/GUG). Valine is incorporated
into hydrophobic transmembrane segments of multiple CI and CIV subunits; pathogenic MT-TV
mutations reduce tRNA-Val availability to all 13 mtDNA-encoded OXPHOS subunits, producing
the combined CI+CIV deficiency fingerprint (CII NORMAL — nuclear SDH unaffected).

m.1624C>T is the most commonly reported pathogenic MT-TV variant (~30%), targeting the
anticodon stem, causing CPEO + myopathy + exercise intolerance + lactic acidosis with
combined CI+CIV deficiency — the mt-translation fingerprint.

NUCLEAR DDx — VARS2 (mt-Valyl-tRNA Synthetase):
  VARS2 biallelic mutations cause mt-Val aminoacylation failure — similar biochemical
  fingerprint (CI+CIV deficiency) but DRAMATICALLY DIFFERENT phenotype: neonatal/infantile-
  onset hypertrophic cardiomyopathy (HCM) + lactic acidosis + severe early-onset
  cardiomyopathy with or without encephalomyopathy. VARS2 is AR (WES-detectable), manifests
  in neonates/infants, NOT as adult CPEO. VARS2 is the most important nuclear DDx because
  the biochemical fingerprint overlaps but management and prognosis differ markedly.

  MT-TV gene              OMIM *590105
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Lactic Acidosis / SNHL / Cardiomyopathy (mild–moderate)
  Protein product         tRNA-Val (UAC anticodon) — 69 nucleotides; RNA gene
                          Val codons: GUU, GUC, GUA, GUG (UAC anticodon, wobble U at position 34)
  Genome                  Mitochondrial DNA (mtDNA), H-strand, rCRS 1602–1670
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    Between 12S rRNA and 16S rRNA; SECOND tRNA in mitochondrial genome

KEY DISTINGUISHING FEATURES vs MT-TF AND PRIOR MT-tRNA GENES:
  • SECOND tRNA gene in human mitochondrial genome — between 12S and 16S rRNA
  • H-strand encoded (no L-strand NGS pitfall — contrast with MT-TE and MT-TP)
  • CPEO + myopathy dominant (similar to MT-TF, MT-TS2, MT-TP)
  • VARS2 nuclear DDx causes NEONATAL/INFANTILE HCM — NOT adult CPEO (key age/phenotype DDx)
  • Positioned between ribosomal RNA genes — disruption may impair mt-ribosome assembly in cis
  • NO stroke-like episodes (DDx MT-TL1 MELAS)
  • NO myoclonic epilepsy or MSL (DDx MT-TK MERRF)
  • NO MIDM maternally inherited diabetes (DDx MT-TE)
"""

import random
from collections import Counter

SEED = 805
N_PATIENTS = 40

VARIANTS = [
    ("m.1624C>T", "Anticodon stem (position 31–32 boundary)", "~30%; most common; CPEO + myopathy + exercise intolerance + lactic acidosis; combined CI+CIV fingerprint; adult onset 18–50 yr; SNHL in ~38%"),
    ("m.1606G>A", "Acceptor stem (position 2–3)", "~25%; CPEO + cardiomyopathy + lactic acidosis; moderate CI+CIV reduction; cardiomyopathy ~45%; adult onset; VARS2 nuclear DDx overlapping fingerprint"),
    ("m.1630A>G", "Variable loop (position 47)", "~20%; CPEO + exercise intolerance + multisystem; Leigh-like MRI at high heteroplasmy; CI+CIV combined; SNHL ~30%; adult-to-childhood onset"),
    ("m.1644G>A", "T-stem (position 54–55)", "~15%; mild–moderate exercise intolerance + SNHL + myopathy; lower heteroplasmy threshold; CI+CIV mild–moderate; adult onset 30–65 yr"),
    ("Large deletion", "MT-TV–spanning deletion (KSS / CPEO region; rRNA-adjacent)", "~10%; deletion spanning MT-TV → variable extent into 12S/16S rRNA region; multi-complex OXPHOS; KSS phenotype; retinal pigmentation ~60%; cardiac conduction; sporadic"),
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
    ("Beta-blockers (if cardiomyopathy)", "STANDARD — cardiac", "Metoprolol / bisoprolol — first-line if cardiomyopathy present; avoid amiodarone (mt-toxic); rate control for AF/flutter"),
    ("LEV (levetiracetam)", "Preferred AED", "First-line if seizures; VPA ABSOLUTE CI; document at every encounter; zonisamide or lacosamide as second-line"),
]

ABSOLUTE_CI = [
    ("Metformin", "ABSOLUTE CI", "Complex I inhibitor → fatal lactic acidosis in CI-deficient skeletal muscle; ALL MT-TV carriers — even asymptomatic"),
    ("VPA / valproate", "ABSOLUTE CI", "mt-ribosome inhibitor + CoA sequestration + POLG inhibition; catastrophic in CI+CIV deficiency; use LEV"),
    ("Propofol", "ABSOLUTE CI (PRIS)", "CIV inhibitor + propofol infusion syndrome; lethal in pre-existing CI+CIV deficiency; use sevoflurane"),
    ("Linezolid", "ABSOLUTE CI", "mt-23S rRNA inhibition → blocks ALL 13 mtDNA-encoded OXPHOS translations; compounding CI+CIV failure"),
    ("Chloramphenicol", "ABSOLUTE CI", "mt-ribosome inhibitor (70S); blocks mt-translation globally; catastrophic in tRNA-Val defect"),
    ("Ketogenic diet", "CONTRAINDICATED", "High-fat low-carb → catabolism + acetyl-CoA → impairs CI-dependent NADH oxidation; worsens lactic acidosis"),
    ("Aminoglycosides", "HIGH CAUTION", "Cochlear OXPHOS amplification; MT-TV SNHL subset at elevated risk; avoid gentamicin/tobramycin; use alternative antibiotics"),
    ("High-dose statins", "CAUTION / RELATIVE CI", "CoQ10 depletion + mitochondrial myopathy worsening; if mandatory, low-dose pravastatin; monitor CK + lactate"),
]


def _make_patients():
    rng = random.Random(SEED)
    pts = []
    variant_weights = [30, 25, 20, 15, 10]
    variant_names = [v[0] for v in VARIANTS]
    cumulative = []
    c = 0
    for w in variant_weights:
        c += w
        cumulative.append(c)

    for i in range(N_PATIENTS):
        r = rng.randint(1, 100)
        vi = 0
        for j, c in enumerate(cumulative):
            if r <= c:
                vi = j
                break
        vname = variant_names[vi]
        is_large_del = vi == 4
        het = rng.randint(25, 82) if not is_large_del else rng.randint(35, 75)
        ci_base = max(12, min(45, int(38 - het * 0.28 + rng.randint(-5, 5))))
        civ_base = max(14, min(48, int(42 - het * 0.25 + rng.randint(-5, 5))))
        cii_base = rng.randint(88, 112)
        cpeo = het >= 38 or is_large_del
        myopathy = het >= 30 or vi == 0
        cardio = vi == 1 or (is_large_del and rng.random() < 0.55) or (vi == 2 and rng.random() < 0.3)
        snhl = het >= 35 or vi == 0 or (vi == 2 and rng.random() < 0.5)
        diabetes = is_large_del and rng.random() < 0.12
        rrf = myopathy and het >= 40
        leigh = vi == 2 and het >= 65
        lactate = round(rng.uniform(1.5, 3.8) + (het - 40) * 0.03, 1)
        pts.append({
            "id": f"TV-{i+1:02d}",
            "variant": vname,
            "sex": rng.choice(["F", "M"]),
            "age_onset_yr": rng.randint(15, 62) if not is_large_del else rng.randint(8, 25),
            "heteroplasmy_blood_pct": het,
            "ci_pct": ci_base,
            "civ_pct": civ_base,
            "cii_pct": cii_base,
            "lactate_mmol_L": max(1.2, lactate),
            "cpeo": cpeo,
            "myopathy": myopathy,
            "cardiomyopathy": cardio,
            "snhl": snhl,
            "diabetes_mellitus": diabetes,
            "ragged_red_fibres": rrf,
            "leigh_like": leigh,
        })
    return pts


def get_overview():
    pts = _make_patients()
    n = len(pts)
    pct_cpeo = round(sum(1 for p in pts if p["cpeo"]) / n * 100)
    pct_myopathy = round(sum(1 for p in pts if p["myopathy"]) / n * 100)
    pct_cardio = round(sum(1 for p in pts if p["cardiomyopathy"]) / n * 100)
    pct_snhl = round(sum(1 for p in pts if p["snhl"]) / n * 100)
    pct_dm = round(sum(1 for p in pts if p["diabetes_mellitus"]) / n * 100)
    pct_rrf = round(sum(1 for p in pts if p["ragged_red_fibres"]) / n * 100)
    avg_het = round(sum(p["heteroplasmy_blood_pct"] for p in pts) / n, 1)
    avg_ci = round(sum(p["ci_pct"] for p in pts) / n, 1)
    avg_civ = round(sum(p["civ_pct"] for p in pts) / n, 1)
    avg_cii = round(sum(p["cii_pct"] for p in pts) / n, 1)
    avg_onset = round(sum(p["age_onset_yr"] for p in pts) / n, 1)

    pheno_counter = Counter()
    for p in pts:
        if p["leigh_like"]:
            pheno_counter["Multisystem — CPEO + SNHL + Leigh-like"] += 1
        elif p["cardiomyopathy"] and p["cpeo"]:
            pheno_counter["CPEO + Cardiomyopathy + Lactic Acidosis"] += 1
        elif p["cpeo"] and p["myopathy"] and p["snhl"]:
            pheno_counter["CPEO + Myopathy + SNHL"] += 1
        elif p["cpeo"] and p["myopathy"]:
            pheno_counter["CPEO + Myopathy + Exercise Intolerance"] += 1
        else:
            pheno_counter["Exercise Intolerance + Myopathy (mild)"] += 1

    pheno_dist = [{"phenotype": k, "count": v, "pct": round(v / n * 100)} for k, v in pheno_counter.most_common()]

    return {
        "gene": "MT-TV",
        "omim": "OMIM *590105",
        "title": "MT-TV — tRNA-Val — Combined CI+CIV Deficiency",
        "subtitle": "CPEO / Myopathy / Exercise Intolerance / SNHL — SECOND tRNA in mt-genome — H-strand rCRS 1602–1670",
        "cohort_statistics": {
            "n_patients": n,
            "avg_heteroplasmy_blood_pct": avg_het,
            "avg_ci_activity_pct_normal": avg_ci,
            "avg_civ_activity_pct_normal": avg_civ,
            "avg_cii_activity_pct_normal": avg_cii,
            "pct_cpeo": pct_cpeo,
            "pct_myopathy": pct_myopathy,
            "pct_cardiomyopathy": pct_cardio,
            "pct_snhl": pct_snhl,
            "pct_diabetes_mellitus": pct_dm,
            "pct_ragged_red_fibres": pct_rrf,
            "avg_age_onset_yr": avg_onset,
        },
        "phenotype_distribution": pheno_dist,
        "key_molecular_features": [
            "MT-TV encodes tRNA-Val (UAC anticodon, 69 nt) — H-strand rCRS 1602–1670 — SECOND tRNA in mitochondrial genome",
            "Val codons GUU/GUC/GUA/GUG decoded by tRNA-Val; valine is essential for hydrophobic CI and CIV transmembrane helices",
            "Combined CI+CIV deficiency (CII NORMAL) — the mt-translation fingerprint; affects all 13 mtDNA-encoded OXPHOS subunits",
            "m.1624C>T (anticodon stem, ~30%) is most common — CPEO + myopathy dominant phenotype, adult onset 18–50 yr",
            "H-strand encoded — no L-strand NGS coverage pitfall (unlike MT-TE, MT-TP, MT-ND6); still requires dedicated mtDNA panel (WES misses RNA genes)",
            "Positioned BETWEEN 12S and 16S rRNA — adjacent large deletions may impair mt-ribosome assembly in cis",
            "VARS2 (mt-Valyl-tRNA Synthetase) nuclear DDx: AR biallelic → neonatal HCM + lactic acidosis (NOT adult CPEO)",
            "Muscle biopsy: RRF on Gomori trichrome; COX-negative (CIV↓) + SDH-positive (CII intact) — mt-translation fingerprint",
            "Heteroplasmy threshold ~38–42% blood for CPEO; muscle heteroplasmy 10–15% higher than blood",
        ],
        "cohort_summary_features": [
            f"40 patients (seed-{SEED}) — synthetic cohort modelling MT-TV disease spectrum per published literature",
            f"CPEO in {pct_cpeo}% — dominant phenotype; ptosis + ophthalmoplegia; progressive over years",
            f"Myopathy in {pct_myopathy}% — proximal weakness + exercise intolerance; RRF on biopsy",
            f"Cardiomyopathy in {pct_cardio}% — annual echo + Holter required; m.1606GA and large deletion subsets highest risk",
            f"SNHL in {pct_snhl}% — audiometry at baseline and every 2 yr; cochlear OXPHOS amplification by aminoglycosides",
            f"Mean CI activity {avg_ci}% normal; mean CIV {avg_civ}%; CII {avg_cii}% (CII ALWAYS NEAR-NORMAL → mt-translation fingerprint)",
            "Large deletion subset (~10%): KSS — CPEO + retinal pigmentation + cardiac conduction block; sporadic",
            "VARS2 nuclear DDx: neonatal HCM + lactic acidosis — confirm maternal inheritance and adult-onset before attributing to MT-TV",
        ],
        "clinical_alerts": [
            {"alert": "Metformin — ABSOLUTE CI", "detail": "Complex I inhibitor → fatal lactic acidosis in CI-deficient muscle; use insulin for diabetes"},
            {"alert": "VPA/valproate — ABSOLUTE CI", "detail": "mt-ribosome inhibitor + POLG inhibition; use LEV for all seizure management"},
            {"alert": "Propofol — ABSOLUTE CI (PRIS)", "detail": "CIV inhibitor; propofol infusion syndrome lethal in CI+CIV deficiency; use sevoflurane"},
            {"alert": "Linezolid — ABSOLUTE CI", "detail": "mt-23S rRNA inhibition → blocks all 13 mtDNA-encoded OXPHOS subunit translations"},
            {"alert": "VARS2 nuclear DDx — WES", "detail": "Neonatal HCM + lactic acidosis → check VARS2 (WES) before MT-TV; age-of-onset is decisive"},
            {"alert": "GIR 6–8 mg/kg/min — NEVER fast", "detail": "Continuous dextrose feeds during illness; fasting triggers lactic crisis; never NPO without IV dextrose"},
        ],
        "heteroplasmy_clinical_map": [
            {"range": "<35% (blood) / low-level MT-TV", "phenotype": "Often asymptomatic or exercise intolerance only; SNHL may be only finding", "management": "Annual review; audiology; avoid CI triggers; no treatment if asymptomatic"},
            {"range": "35–50% — mild disease", "phenotype": "Exercise intolerance + mild myopathy; early CPEO; SNHL; occasional lactic acidosis on exertion", "management": "Mitochondrial cocktail; LEV if seizures; avoid absolute CIs; physio/OT"},
            {"range": "50–65% — moderate disease", "phenotype": "Established CPEO + myopathy; SNHL; lactic acidosis at rest; cardiomyopathy possible", "management": "Cardiology review (echo + Holter); GIR protocol ready; ophthalmology; annual endocrine screen"},
            {"range": "65–80% — severe disease", "phenotype": "Multisystem — CPEO + cardiomyopathy + SNHL + lactic acidosis; Leigh-like MRI possible", "management": "Tertiary mitochondrial centre; multidisciplinary; Leigh-like: emergency lactate management"},
            {"range": "Large deletion (variable)", "phenotype": "KSS — CPEO + retinal pigmentation + cardiac conduction; sporadic; onset usually <20 yr", "management": "Pacemaker if AV block; EP study; retinal monitoring; endocrine screen annually"},
        ],
    }


def get_breakdown():
    pts = _make_patients()
    n = len(pts)

    # Per-variant summaries
    by_variant = {}
    for p in pts:
        v = p["variant"]
        if v not in by_variant:
            by_variant[v] = []
        by_variant[v].append(p)

    variant_summaries = []
    for vname, vpts in by_variant.items():
        nv = len(vpts)
        region = next((vr[1] for vr in VARIANTS if vr[0] == vname), "?")
        note = next((vr[2] for vr in VARIANTS if vr[0] == vname), "?")
        variant_summaries.append({
            "variant": vname,
            "region": region,
            "n": nv,
            "avg_heteroplasmy_blood_pct": round(sum(p["heteroplasmy_blood_pct"] for p in vpts) / nv, 1),
            "avg_ci_activity_pct": round(sum(p["ci_pct"] for p in vpts) / nv, 1),
            "avg_civ_activity_pct": round(sum(p["civ_pct"] for p in vpts) / nv, 1),
            "pct_cpeo": round(sum(1 for p in vpts if p["cpeo"]) / nv * 100),
            "pct_myopathy": round(sum(1 for p in vpts if p["myopathy"]) / nv * 100),
            "pct_cardiomyopathy": round(sum(1 for p in vpts if p["cardiomyopathy"]) / nv * 100),
            "pct_snhl": round(sum(1 for p in vpts if p["snhl"]) / nv * 100),
            "pct_diabetes_mellitus": round(sum(1 for p in vpts if p["diabetes_mellitus"]) / nv * 100),
            "note": note,
        })

    # Trigger rates
    rng = random.Random(SEED + 1)
    trigger_rates = [{"trigger": t, "pct": rng.randint(10, 60)} for t in TRIGGERS]
    trigger_rates.sort(key=lambda x: -x["pct"])

    # Treatment info
    treatment_info = [{"agent": t[0], "evidence": t[1], "note": t[2]} for t in TREATMENTS]

    # Biochemical fingerprint
    avg_ci = round(sum(p["ci_pct"] for p in pts) / n, 1)
    avg_civ = round(sum(p["civ_pct"] for p in pts) / n, 1)
    avg_cii = round(sum(p["cii_pct"] for p in pts) / n, 1)

    # DDx comparison
    ddx_comparison = [
        {
            "gene": "VARS2 (mt-Valyl-tRNA Synthetase — MOST IMPORTANT nuclear DDx)",
            "disease": "Neonatal / Infantile HCM + Lactic Acidosis + Encephalomyopathy",
            "oxphos": "CI + CIV ↓, CII normal (overlapping fingerprint)",
            "distinguisher": "AR biallelic (WES-detectable); NEONATAL/INFANTILE onset — NOT adult CPEO; HCM + lactic acidosis dominates; NO CPEO; identical biochemical CI+CIV fingerprint — most important nuclear DDx; maternal inheritance ABSENT in VARS2",
        },
        {
            "gene": "MT-TF (tRNA-Phe — FIRST tRNA in mt-genome)",
            "disease": "CPEO / Myopathy / Exercise Intolerance",
            "oxphos": "CI + CIV ↓, CII normal",
            "distinguisher": "MT-TF (577–647) precedes MT-TV (1602–1670) in genome; similar phenotypes; FARS2 is nuclear DDx for MT-TF vs VARS2 for MT-TV; both H-strand encoded; MT-TF D-loop adjacent (large deletions can reduce mtDNA copy number)",
        },
        {
            "gene": "MT-TL1 (tRNA-Leu UUR)",
            "disease": "MELAS / MIDD / Pan-OXPHOS (CI+CIII+CIV)",
            "oxphos": "Pan-OXPHOS: CI+CIII+CIV all reduced",
            "distinguisher": "STROKE-LIKE EPISODES (SLE) are hallmark MELAS — ABSENT in MT-TV; Pan-OXPHOS (CIII also reduced) vs MT-TV CI+CIV combined; m.3243AG most common worldwide mtDNA variant",
        },
        {
            "gene": "MT-TK (tRNA-Lys)",
            "disease": "MERRF — Myoclonic Epilepsy Ragged-Red Fibres / MSL",
            "oxphos": "Pan-OXPHOS: CI+CIV reduced (CII normal)",
            "distinguisher": "MYOCLONIC EPILEPSY is hallmark MERRF — ABSENT in MT-TV; MSL (Madelung disease) distinctive for MT-TK; m.8344AG >80% worldwide MERRF",
        },
        {
            "gene": "MT-TE (tRNA-Glu)",
            "disease": "CPEO / Myopathy / MIDM-Maternally-Inherited-Diabetes",
            "oxphos": "CI + CIV ↓, CII normal",
            "distinguisher": "MIDM (maternally inherited diabetes) is distinctive for MT-TE — absent in MT-TV; m.14674TC REVERSIBLE infantile COX deficiency unique to MT-TE; MT-TE is L-strand encoded (NGS pitfall) — MT-TV is H-strand",
        },
        {
            "gene": "MT-TT (tRNA-Thr)",
            "disease": "CPEO / Myopathy / Cardiomyopathy (HCM 55–65%)",
            "oxphos": "CI + CIV ↓, CII normal",
            "distinguisher": "HCM MUCH more prominent in MT-TT (55–65%); MT-TV cardiomyopathy ~35%; AMIODARONE absolute CI in MT-TT; TARS2 DDx for MT-TT vs VARS2 for MT-TV",
        },
        {
            "gene": "MT-TP (tRNA-Pro — LAST tRNA)",
            "disease": "CPEO / Myopathy / Exercise Intolerance",
            "oxphos": "CI + CIV ↓, CII normal",
            "distinguisher": "MT-TP is L-strand encoded (NGS pitfall) — MT-TV is H-strand (no pitfall); MT-TP is FINAL tRNA (15956–16023) vs MT-TV SECOND (1602–1670); similar phenotypes but NGS detection differs markedly; PARS2 DDx for MT-TP",
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
            "distinguisher": "HEPATOPATHY distinguishes POLG (Alpers phenotype) — ABSENT in MT-TV; autosomal recessive; WES-detectable; mtDNA depletion on Southern blot",
        },
    ]

    # Contraindication details
    contraindication_info = [{"agent": a[0], "category": a[1], "rationale": a[2]} for a in ABSOLUTE_CI]

    return {
        "variant_summaries": variant_summaries,
        "per_patient": pts,
        "trigger_rates": trigger_rates,
        "treatment_info": treatment_info,
        "biochemical_fingerprint": {
            "CI_pct_normal": avg_ci,
            "CIV_pct_normal": avg_civ,
            "CII_pct_normal": avg_cii,
            "pattern": "Combined CI+CIV deficiency (CII NORMAL) — mt-translation fingerprint; all 13 mtDNA-encoded OXPHOS subunits require tRNA-Val",
            "BN_PAGE": "CI supercomplex (I+III2+IV) disassembly; isolated CII (nuclear SDH) fully intact; CIV monomer absent/reduced",
            "muscle_histochemistry": "RRF on modified Gomori trichrome; COX-negative (CIV↓) + SDH-positive (CII intact); COX/SDH dual stain shows mosaic pattern",
            "H_strand_note": "MT-TV is H-strand encoded (no L-strand NGS coverage pitfall). However, MT-TV is an RNA gene — absent from WES exome capture. A dedicated mtDNA sequencing panel (mtDNA-seq / amplicon sequencing) is mandatory for MT-TV variant detection. Blood heteroplasmy may underestimate muscle burden by 10–15%.",
        },
        "ddx_comparison": ddx_comparison,
        "contraindication_info": contraindication_info,
    }


def get_definitions():
    return {
        "gene_biology": {
            "MT_TV_gene": "Mitochondrially encoded tRNA-Val (UAC anticodon) — OMIM *590105 — 69 nucleotides; the SECOND tRNA gene in the human mitochondrial genome, at rCRS 1602–1670, H-strand encoded. Situated between 12S rRNA (648–1601) and 16S rRNA (1671–3229).",
            "genomic_position_SECOND": "MT-TV (H-strand rCRS 1602–1670) lies between the two mitochondrial ribosomal RNA genes. This positional embedding means: (1) Large deletions spanning MT-TV can also affect flanking rRNA genes, potentially impairing mt-ribosome assembly. (2) MT-TV tRNA-Val is essential for translating all 13 mtDNA-encoded OXPHOS subunits — loss amplifies rRNA-region deletions.",
            "tRNA_structure": "69-nt cloverleaf tRNA; UAC anticodon at positions 34–36 (wobble U at 34 → GUU/GUC Val codons; modified U can extend to GUA/GUG); acceptor stem 7 bp; D-stem 4 bp; anticodon stem 5 bp; variable loop 5 nt; T-stem 5 bp. Mutations disrupt aminoacylation by VARS2 (mt-Val-tRNA synthetase) or tertiary folding, reducing mt-Val incorporation into OXPHOS subunits.",
            "Val_codons": "GUU, GUC, GUA, GUG — all four Val codons decoded by tRNA-Val (UAC anticodon) via wobble base pairing; valine is a small hydrophobic amino acid critical for transmembrane helices of CI subunits ND1, ND2, ND4, ND5 and CIV subunit CO1; Val substitutions in these helices destabilize OXPHOS supercomplex formation.",
            "mt_translation_fingerprint": "Pathogenic MT-TV mutations reduce tRNA-Val availability to all 13 mtDNA-encoded OXPHOS subunits. CII (succinate dehydrogenase) is encoded entirely by nuclear DNA (SDHA/B/C/D) — CII NORMAL confirms mt-translation defect vs isolated nuclear OXPHOS gene. CI+CIV combined deficiency with CII NORMAL is the mt-tRNA translation fingerprint.",
            "H_strand_encoding": "MT-TV is transcribed from the Heavy (H) strand — the standard orientation for mtDNA sequencing. Unlike MT-TE, MT-TP, and MT-ND6 (L-strand genes), there is no H-strand NGS coverage pitfall for MT-TV. However, MT-TV is still an RNA gene absent from WES exome capture; a dedicated mtDNA sequencing panel is required for detection.",
            "rRNA_adjacency": "MT-TV is flanked by 12S rRNA (3' of MT-TV at 648–1601) and 16S rRNA (5' of MT-TV at 1671–3229). Large deletions extending from MT-TV into these rRNA genes can impair mt-ribosome assembly (12S + 16S rRNA are the scaffold of the mitoribosome), compounding the tRNA-Val deficiency with translational apparatus disruption.",
        },
        "clinical_terms": {
            "CPEO": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive limitation of eye movements; EOM (extraocular muscles) are mitochondrially rich; CPEO is the cardinal feature of MT-TV disease (~78–82%); onset usually 18–50 yr; ptosis often precedes ophthalmoplegia by years.",
            "Combined_CI_CIV_Deficiency": "CI (NADH-ubiquinone oxidoreductase, 45 subunits: 7 mtDNA-encoded ND1-6,4L) + CIV (cytochrome c oxidase, 13 subunits: 3 mtDNA-encoded CO1-3) — both reduced because tRNA-Val is essential for translating all 13 mtDNA-encoded OXPHOS subunits. Spectrophotometric activities: CI <30% normal + CIV <35% normal = combined deficiency.",
            "Lactic_acidosis": "Blood lactate >2.5 mmol/L (normal <2.0); exertional in mild cases; resting in high heteroplasmy or crisis; management: thiamine IV + GIR 6–8 + avoid fasting + bicarbonate if pH <7.2.",
            "RRF": "Ragged-Red Fibres — modified Gomori trichrome stain shows red-staining mitochondrial accumulations at the periphery of muscle fibres; COX-negative (CIV-deficient) + SDH-positive (CII intact) pattern is the mt-translation fingerprint.",
            "VARS2_DDx": "mt-Valyl-tRNA Synthetase (VARS2) charges tRNA-Val with valine. Biallelic VARS2 LOF mutations cause neonatal/infantile-onset HCM (hypertrophic cardiomyopathy) + lactic acidosis + encephalomyopathy. VARS2 disease is AR (WES-detectable), manifests in neonates/infants with cardiomyopathy and metabolic crisis, NOT as adult CPEO. The biochemical fingerprint (CI+CIV deficiency) overlaps with MT-TV — neonatal/infantile onset vs adult onset is the most critical distinguishing feature.",
            "KSS": "Kearns-Sayre Syndrome — large mtDNA deletion presenting before age 20 with: CPEO + pigmentary retinopathy + cardiac conduction defect (AV block); RRF on biopsy; CSF protein >100 mg/dL; cerebellar ataxia; SNHL; endocrinopathies. MT-TV-spanning deletions → KSS phenotype.",
        },
        "pharmacology": {
            "preferred_aed": "Levetiracetam (LEV) — first-line for any seizures in MT-TV disease. VPA (valproate) is ABSOLUTE CI. Second-line: zonisamide, lacosamide.",
            "cardiac_management": "If cardiomyopathy (m.1606GA or large deletion): annual echo + 24h Holter. Beta-blockers (metoprolol/bisoprolol) first-line for HCM. AVOID amiodarone (mt-toxic, OXPHOS inhibitor). Pacemaker if AV block (KSS large deletion). ICD if EF <35%.",
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
            "Schaefer AM, McFarland R et al. (2008) — Prevalence of mitochondrial DNA disease in adults. Ann Neurol 63:35.",
            "Vanlander AV, Menten B et al. (2015) — Two siblings with homozygous pathogenic splice-site variant in mitochondrial asparaginyl-tRNA synthetase (NARS2). Hum Mutat 36:222.",
            "Diodato D, Ghezzi D, Tiranti V (2014) — The mitochondrial aminoacyl tRNA synthetases: genes and syndromes. Int J Cell Biol 2014:787956.",
            "Gorman GS, Chinnery PF et al. (2016) — Mitochondrial diseases. Nat Rev Dis Primers 2:16080.",
            "DiMauro S, Schon EA (2003) — Mitochondrial respiratory-chain diseases. NEJM 348:2656.",
            "Alston CL, Rocha MC et al. (2017) — The genetics and pathology of mitochondrial disease. J Pathol 241:236.",
            "Taylor RW, Turnbull DM (2005) — Mitochondrial DNA mutations in human disease. Nat Rev Genet 6:389.",
            "Lott MT, Leipzig JN et al. (2013) — mtDNA variation and analysis using MITOMAP and MITOMASTER. Curr Protoc Bioinformatics.",
            "Kearns TP, Sayre GP (1958) — Retinitis pigmentosa, external ophthalmoplegitis, and complete heart block. AMA Arch Ophthalmol.",
        ],
    }
