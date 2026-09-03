#!/usr/bin/env python3
"""MT-TI — Mitochondrially Encoded tRNA-Ile — CPEO / Myopathy / m.4300AG-Isolated-Cardiomyopathy
Combined CI+CIV Deficiency (mt-translation fingerprint) | H-strand rCRS 4263–4331

MT-TI (OMIM *590045) encodes mitochondrial tRNA-Ile (GAU anticodon, reading Ile codons
AUU/AUC; modified anticodon also reads AUA in human mitochondria), located on the H-strand
at rCRS 4263–4331 (69 nt). MT-TI is the FOURTH tRNA gene of the human mitochondrial
genome, following MT-TF (577–647, 1st), MT-TV (1602–1670, 2nd) and MT-TL1 (3230–3304, 3rd).
MT-TI is immediately adjacent to the MT-ND1 gene (rCRS 3307–4262), and its 3′ boundary
(rCRS 4329–4331) overlaps with the 5′ start of MT-TQ (tRNA-Gln, L-strand, 4329–4400) —
making large deletions spanning the MT-TI / MT-TQ junction doubly damaging: simultaneous
loss of both tRNA-Ile (H-strand) and tRNA-Gln (L-strand) in the same deletion.

Ile codons (AUU, AUC, AUA) are all decoded by mt-tRNA-Ile. In human mitochondria, a
post-transcriptional C34→U modification of the anticodon expands decoding to include AUA
(which is normally Met in the nuclear genetic code but Ile in mammalian mitochondria via the
mitochondrial genetic code deviation). Isoleucine is incorporated into hydrophobic
transmembrane segments of multiple CI subunits (ND1, ND2, ND4, ND5) and CIV subunits
(CO1, CO2), and pathogenic MT-TI mutations reduce tRNA-Ile availability to all 13 mtDNA-
encoded OXPHOS subunits → combined CI+CIV deficiency fingerprint (CII NORMAL).

m.4269A>G is the most commonly reported pathogenic MT-TI variant (~30–32%), targeting the
anticodon stem (position 3–70 Watson-Crick base pair), causing CPEO + myopathy + lactic
acidosis with combined CI+CIV deficiency — the classic mt-translation fingerprint.

*** MOST DISTINCTIVE MT-TI FEATURE ***
m.4300A>G (near-acceptor-stem / discriminator-base region, position 72) causes ISOLATED
HYPERTROPHIC CARDIOMYOPATHY without CPEO even at relatively low heteroplasmy — the only
mt-tRNA variant documented to cause ISOLATED HCM as the dominant phenotype at low
heteroplasmy. Heart muscle concentrates heteroplasmy higher than blood; cardiac tissues
exhibit the highest mt-Ile demand for CI/CIV transmembrane subunits → isolated cardiac
phenotype despite systemic mt-tRNA-Ile deficiency. Annual echo + Holter MANDATORY in all
m.4300A>G carriers regardless of CPEO status.

MT-TI–MT-TQ JUNCTION OVERLAP:
MT-TI (H-strand, 4263–4331) and MT-TQ (L-strand, 4329–4400) overlap at rCRS 4329–4331.
Large deletions spanning this boundary can simultaneously impair both tRNA-Ile and tRNA-Gln
translation, producing a compound CPEO/multisystem phenotype exceeding that of single-tRNA
loss. NGS must always report L-strand coverage at MT-TQ region when investigating MT-TI.

NUCLEAR DDx — IARS2 (mt-Isoleucyl-tRNA Synthetase):
  IARS2 biallelic mutations cause mt-Ile aminoacylation failure — similar biochemical
  fingerprint (CI+CIV deficiency) but DRAMATICALLY DIFFERENT phenotype: CAGSSS syndrome
  (Cataracts, growth hormone deficiency, sensory neuropathy, Sensorineural hearing loss,
  Skeletal dysplasia) in children, OR neonatal/infantile Leigh syndrome. IARS2 is AR
  (WES-detectable), manifests in childhood/neonatal period, NOT as adult CPEO. CAGSSS
  syndrome is the most important nuclear DDx because the biochemical fingerprint overlaps
  but management and prognosis differ markedly — and IARS2 disease is treatable with
  growth hormone supplementation and ophthalmologic surveillance from birth.

  MT-TI gene              OMIM *590045
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / Exercise Intolerance
                          Isolated HCM (m.4300AG) / Lactic Acidosis / SNHL / Cardiomyopathy
  Protein product         tRNA-Ile (GAU anticodon) — 69 nucleotides; RNA gene
                          Ile codons: AUU, AUC, AUA (C34→U modification for AUA in mt)
  Genome                  Mitochondrial DNA (mtDNA), H-strand, rCRS 4263–4331
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    FOURTH tRNA in mitochondrial genome; 3′-adjacent to MT-ND1;
                          overlaps MT-TQ (L-strand) at rCRS 4329–4331

KEY DISTINGUISHING FEATURES vs PRIOR MT-tRNA GENES:
  • m.4300AG → ISOLATED HCM without CPEO — UNIQUE among all mt-tRNA genes at low heteroplasmy
  • FOURTH tRNA gene (after MT-TF, MT-TV, MT-TL1); immediately 3′ to MT-ND1
  • H-strand encoded (no L-strand NGS pitfall — contrast with MT-TE, MT-TP, MT-ND6)
  • OVERLAPS MT-TQ (L-strand) at 4329–4331 — large deletions doubly damaging
  • IARS2 nuclear DDx causes CAGSSS / neonatal Leigh — NOT adult CPEO (key age/phenotype DDx)
  • NO stroke-like episodes (DDx MT-TL1 MELAS)
  • NO myoclonic epilepsy or MSL (DDx MT-TK MERRF)
  • NO MIDM maternally inherited diabetes (DDx MT-TE)
"""

import random
from collections import Counter

SEED = 807
N_PATIENTS = 40

VARIANTS = [
    ("m.4269A>G", "Anticodon stem (position 3–70 Watson-Crick pair)", "~30%; most common; CPEO + myopathy + lactic acidosis; combined CI+CIV; adult onset 18–55 yr; SNHL in ~35%; mitochondrial cytopathy pattern on biopsy"),
    ("m.4295A>G", "Variable loop (position 47a)", "~22%; CPEO + exercise intolerance + SNHL; combined CI+CIV; moderate course; adult onset 20–50 yr; VARS-like phenotype; no isolated cardiac"),
    ("m.4300A>G", "Near-acceptor-stem / discriminator-base (position 72)", "~20%; MOST DISTINCTIVE — isolated HCM without CPEO; cardiac heteroplasmy amplified vs blood; annual echo+Holter mandatory even without CPEO; lactic acidosis in decompensation; low heteroplasmy threshold for cardiac disease"),
    ("m.4309A>G", "Variable loop / anticodon-loop junction (position 45)", "~15%; multisystem — CPEO + SNHL + Leigh-like MRI at high heteroplasmy; CI+CIV combined; childhood or adult onset; seizures possible in Leigh-like subset"),
    ("Large deletion", "MT-TI–MT-TQ junction spanning deletion (KSS / CPEO; MT-TQ L-strand co-affected)", "~13%; deletion spanning MT-TI boundary into MT-TQ (L-strand overlap at 4329–4331) → compound tRNA-Ile + tRNA-Gln loss; KSS phenotype with retinal pigmentation + cardiac conduction; sporadic; severe multisystem"),
]

PHENOTYPES = [
    "CPEO + myopathy + exercise intolerance — adult heteroplasmy 38–72%",
    "Isolated HCM — m.4300AG subset; low–moderate heteroplasmy; cardiac-dominant",
    "Multisystem — CPEO + exercise intolerance + SNHL + lactic acidosis",
    "KSS — CPEO + cardiomyopathy + retinal pigmentation + cardiac conduction defects",
    "Exercise intolerance + myopathy — low-moderate heteroplasmy",
]

TRIGGERS = [
    "Intercurrent illness / fever", "Fasting / prolonged NPO", "Anaesthetic agents (propofol)",
    "VPA/valproate", "Physiological stress / surgery", "Linezolid antibiotic",
    "High-dose statins", "Aminoglycosides (cochlear OXPHOS)", "Metformin",
    "Cardiac decompensation (m.4300AG)",
]

TREATMENTS = [
    ("CoQ10 (ubiquinol)", "Level C", "Mitochondrial cofactor; 10–30 mg/kg/day divided; ubiquinol preferred; CI+CIV combined deficiency standard adjunct"),
    ("Riboflavin (B2)", "Level C", "50–200 mg/day; FAD cofactor; CI+CIII support; low risk; continue long-term"),
    ("Thiamine (B1)", "MANDATORY empiric", "10–20 mg/kg/day IV acutely; PDH cofactor; empiric before workup; BTBGD exclusion first"),
    ("Biotin", "MANDATORY empiric", "10 mg/day; BTD/SLC19A3 exclusion empiric; withdraw only after BTBGD excluded"),
    ("L-Carnitine", "Level C", "50–100 mg/kg/day; secondary carnitine deficiency common; correction mandatory"),
    ("Beta-blockers (cardiac — all MT-TI)", "STANDARD — cardiac", "Metoprolol / bisoprolol — first-line if cardiomyopathy; m.4300AG: mandatory from first echo abnormality; avoid amiodarone (mt-toxic)"),
    ("LEV (levetiracetam)", "Preferred AED", "First-line if seizures; VPA ABSOLUTE CI; zonisamide or lacosamide as second-line; clonazepam for cortical myoclonus if present (Leigh-like m.4309AG subset)"),
    ("Annual echo + Holter", "MANDATORY — m.4300AG", "ALL m.4300AG carriers regardless of CPEO status; cardiac heteroplasmy higher than blood; HCM can precede any neuromuscular symptoms by years"),
]

ABSOLUTE_CI = [
    ("Metformin", "ABSOLUTE CI", "Complex I inhibitor → fatal lactic acidosis in CI-deficient skeletal muscle; ALL MT-TI carriers — even asymptomatic; particularly critical in m.4300AG with cardiac disease"),
    ("VPA / valproate", "ABSOLUTE CI", "mt-ribosome inhibitor + CoA sequestration + POLG inhibition; catastrophic in CI+CIV deficiency; use LEV"),
    ("Propofol", "ABSOLUTE CI (PRIS)", "CIV inhibitor + propofol infusion syndrome; lethal in pre-existing CI+CIV deficiency; use sevoflurane"),
    ("Linezolid", "ABSOLUTE CI", "mt-23S rRNA inhibition → blocks ALL 13 mtDNA-encoded OXPHOS translations; compounding CI+CIV failure"),
    ("Chloramphenicol", "ABSOLUTE CI", "mt-ribosome inhibitor (70S); blocks mt-translation globally; catastrophic in tRNA-Ile defect"),
    ("Ketogenic diet", "CONTRAINDICATED", "High-fat low-carb → catabolism + acetyl-CoA → impairs CI-dependent NADH oxidation; worsens lactic acidosis"),
    ("Amiodarone", "ABSOLUTE CI", "mt-OXPHOS inhibitor; NEVER use for cardiac arrhythmias in MT-TI disease — especially m.4300AG HCM; use beta-blocker instead"),
    ("Aminoglycosides", "HIGH CAUTION", "Cochlear OXPHOS amplification; MT-TI SNHL subsets at elevated risk; avoid gentamicin/tobramycin; use alternative antibiotics"),
    ("High-dose statins", "CAUTION / RELATIVE CI", "CoQ10 depletion + mitochondrial myopathy worsening; if mandatory, low-dose pravastatin; monitor CK + lactate"),
]


def _make_patients():
    rng = random.Random(SEED)
    pts = []
    variant_weights = [30, 22, 20, 15, 13]
    variant_names = [v[0] for v in VARIANTS]
    cumulative = []
    c = 0
    for w in variant_weights:
        c += w
        cumulative.append(c)

    for i in range(N_PATIENTS):
        r = rng.randint(1, 100)
        vi = 0
        for j, c2 in enumerate(cumulative):
            if r <= c2:
                vi = j
                break
        vname = variant_names[vi]
        is_large_del = vi == 4
        is_cardiac = vi == 2   # m.4300AG — isolated HCM
        het = rng.randint(22, 80) if not is_large_del else rng.randint(30, 70)
        ci_base = max(12, min(48, int(40 - het * 0.27 + rng.randint(-5, 5))))
        civ_base = max(14, min(50, int(44 - het * 0.24 + rng.randint(-5, 5))))
        cii_base = rng.randint(88, 112)
        # m.4300AG: cardiac phenotype even at lower heteroplasmy
        if is_cardiac:
            het = rng.randint(18, 60)
            ci_base = max(14, min(52, int(42 - het * 0.22 + rng.randint(-4, 4))))
            civ_base = max(16, min(55, int(46 - het * 0.20 + rng.randint(-4, 4))))
        cpeo = (het >= 40 or is_large_del) and not is_cardiac
        myopathy = (het >= 32 or vi == 0) and not is_cardiac
        cardio = is_cardiac or (is_large_del and rng.random() < 0.60) or (vi == 1 and rng.random() < 0.25) or (vi == 3 and rng.random() < 0.30)
        snhl = het >= 35 or vi == 0 or (vi == 1 and rng.random() < 0.55) or (vi == 3 and rng.random() < 0.45)
        rrf = myopathy and het >= 42
        leigh = vi == 3 and het >= 62
        lactate = round(rng.uniform(1.4, 3.9) + (het - 40) * 0.03, 1)
        pts.append({
            "id": f"TI-{i+1:02d}",
            "variant": vname,
            "sex": rng.choice(["F", "M"]),
            "age_onset_yr": rng.randint(14, 62) if not is_large_del else rng.randint(6, 24),
            "heteroplasmy_blood_pct": het,
            "ci_pct": ci_base,
            "civ_pct": civ_base,
            "cii_pct": cii_base,
            "lactate_mmol_L": max(1.2, lactate),
            "cpeo": cpeo,
            "myopathy": myopathy,
            "cardiomyopathy": cardio,
            "snhl": snhl,
            "ragged_red_fibres": rrf,
            "leigh_like": leigh,
            "isolated_hcm": is_cardiac,
        })
    return pts


def get_overview():
    pts = _make_patients()
    n = len(pts)
    pct_cpeo = round(sum(1 for p in pts if p["cpeo"]) / n * 100)
    pct_myopathy = round(sum(1 for p in pts if p["myopathy"]) / n * 100)
    pct_cardio = round(sum(1 for p in pts if p["cardiomyopathy"]) / n * 100)
    pct_snhl = round(sum(1 for p in pts if p["snhl"]) / n * 100)
    pct_rrf = round(sum(1 for p in pts if p["ragged_red_fibres"]) / n * 100)
    pct_leigh = round(sum(1 for p in pts if p["leigh_like"]) / n * 100)
    pct_isolated_hcm = round(sum(1 for p in pts if p["isolated_hcm"]) / n * 100)
    avg_het = round(sum(p["heteroplasmy_blood_pct"] for p in pts) / n, 1)
    avg_ci = round(sum(p["ci_pct"] for p in pts) / n, 1)
    avg_civ = round(sum(p["civ_pct"] for p in pts) / n, 1)
    avg_cii = round(sum(p["cii_pct"] for p in pts) / n, 1)
    avg_onset = round(sum(p["age_onset_yr"] for p in pts) / n, 1)

    pheno_counter = Counter()
    for p in pts:
        if p["isolated_hcm"]:
            pheno_counter["Isolated HCM — m.4300AG cardiac-dominant (no CPEO)"] += 1
        elif p["leigh_like"]:
            pheno_counter["Multisystem — CPEO + SNHL + Leigh-like MRI"] += 1
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
        "gene": "MT-TI",
        "omim": "OMIM *590045",
        "title": "MT-TI — tRNA-Ile — Combined CI+CIV Deficiency",
        "subtitle": "CPEO / Myopathy / m.4300AG-Isolated-HCM-DISTINCTIVE / SNHL — FOURTH tRNA in mt-genome — H-strand rCRS 4263–4331 — MT-TQ-Overlap-4329–4331",
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
            "pct_ragged_red_fibres": pct_rrf,
            "pct_leigh_like": pct_leigh,
            "pct_isolated_hcm_m4300ag": pct_isolated_hcm,
            "avg_age_onset_yr": avg_onset,
        },
        "phenotype_distribution": pheno_dist,
        "key_molecular_features": [
            "Combined CI+CIV deficiency (CII NORMAL) — mt-translation fingerprint; all 13 mtDNA-encoded OXPHOS subunits require tRNA-Ile",
            "m.4300A>G → ISOLATED HCM without CPEO — UNIQUE among all mt-tRNA genes; cardiac heteroplasmy amplified vs blood; annual echo mandatory",
            "MT-TI overlaps MT-TQ (L-strand) at rCRS 4329–4331 — large deletions can simultaneously impair tRNA-Ile AND tRNA-Gln",
            "FOURTH tRNA gene in human mitochondrial genome; immediately 3′ to MT-ND1 (3307–4262); H-strand encoded",
            "IARS2 nuclear DDx (CAGSSS syndrome / neonatal Leigh) — very different from adult CPEO phenotype; WES-detectable",
            "Aminoglycosides HIGH CAUTION — SNHL amplification via cochlear OXPHOS",
            "AMIODARONE ABSOLUTE CI — mt-OXPHOS inhibitor; use beta-blockers for m.4300AG cardiac disease",
        ],
        "clinical_alerts": [
            {"alert": "m.4300AG — ISOLATED HCM ALERT", "detail": "Annual echo + Holter MANDATORY in all m.4300AG carriers — HCM can precede CPEO by years; cardiac heteroplasmy higher than blood"},
            {"alert": "Metformin — ABSOLUTE CI", "detail": "Complex I inhibitor → fatal lactic acidosis in CI-deficient muscle; use insulin for diabetes"},
            {"alert": "VPA/valproate — ABSOLUTE CI", "detail": "mt-ribosome inhibitor + POLG inhibition; use LEV for all seizure management"},
            {"alert": "Propofol — ABSOLUTE CI (PRIS)", "detail": "CIV inhibitor; propofol infusion syndrome lethal in CI+CIV deficiency; use sevoflurane"},
            {"alert": "Amiodarone — ABSOLUTE CI", "detail": "mt-OXPHOS inhibitor; NEVER for m.4300AG cardiac arrhythmias; use beta-blocker + cardiology referral"},
            {"alert": "IARS2 nuclear DDx — WES", "detail": "CAGSSS (cataracts + growth failure + SNHL + skeletal dysplasia) → check IARS2; neonatal/childhood onset vs adult CPEO"},
            {"alert": "MT-TQ overlap at 4329–4331", "detail": "Large deletions spanning MT-TI can co-destroy MT-TQ (L-strand); verify L-strand NGS coverage at 4329–4400"},
            {"alert": "GIR 6–8 mg/kg/min — NEVER fast", "detail": "Continuous dextrose during illness; fasting triggers lactic crisis; never NPO without IV dextrose"},
        ],
        "heteroplasmy_clinical_map": [
            {"range": "<35% (blood) / low MT-TI", "phenotype": "Often asymptomatic or exercise intolerance + SNHL only; m.4300AG exception: HCM possible even at low heteroplasmy", "management": "Annual review; audiology; echo if m.4300AG; avoid CI triggers; no treatment if truly asymptomatic (non-m.4300AG)"},
            {"range": "35–50% — mild disease", "phenotype": "Exercise intolerance + mild myopathy; early CPEO (non-m.4300AG); SNHL; occasional lactic acidosis on exertion", "management": "Mitochondrial cocktail; LEV if seizures; avoid absolute CIs; physio/OT; annual echo for all"},
            {"range": "50–65% — moderate disease", "phenotype": "Established CPEO + myopathy; SNHL; lactic acidosis at rest; cardiomyopathy in m.4300AG and large deletion subsets", "management": "Cardiology review (echo + Holter); GIR protocol ready; ophthalmology; annual endocrine screen"},
            {"range": "65–80% — severe disease", "phenotype": "Multisystem — CPEO + cardiomyopathy + SNHL + lactic acidosis; Leigh-like MRI in m.4309AG subset", "management": "Tertiary mitochondrial centre; multidisciplinary; Leigh-like: emergency lactate management; ICD if EF <35%"},
            {"range": "Large deletion (variable, MT-TI+MT-TQ co-affected)", "phenotype": "KSS — CPEO + retinal pigmentation + cardiac conduction; compound tRNA-Ile + tRNA-Gln loss; sporadic; onset <24 yr", "management": "Pacemaker if AV block; EP study; retinal monitoring; endocrine screen annually; L-strand QC mandatory"},
        ],
    }


def get_breakdown():
    pts = _make_patients()
    n = len(pts)

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
            "pct_isolated_hcm": round(sum(1 for p in vpts if p["isolated_hcm"]) / nv * 100),
            "note": note,
        })

    rng = random.Random(SEED + 1)
    trigger_rates = [{"trigger": t, "pct": rng.randint(10, 60)} for t in TRIGGERS]
    trigger_rates.sort(key=lambda x: -x["pct"])

    treatment_info = [{"agent": t[0], "evidence": t[1], "note": t[2]} for t in TREATMENTS]

    avg_ci = round(sum(p["ci_pct"] for p in pts) / n, 1)
    avg_civ = round(sum(p["civ_pct"] for p in pts) / n, 1)
    avg_cii = round(sum(p["cii_pct"] for p in pts) / n, 1)

    ddx_comparison = [
        {
            "gene": "IARS2 (mt-Isoleucyl-tRNA Synthetase — MOST IMPORTANT nuclear DDx)",
            "disease": "CAGSSS Syndrome (Cataracts, Growth hormone deficiency, Sensory neuropathy, SNHL, Skeletal dysplasia) / Neonatal Leigh",
            "oxphos": "CI + CIV ↓, CII normal (overlapping fingerprint)",
            "distinguisher": "AR biallelic (WES-detectable); NEONATAL/CHILDHOOD onset — NOT adult CPEO; cataracts + growth failure + SNHL + skeletal dysplasia dominate; NO isolated adult HCM; maternal inheritance ABSENT in IARS2",
        },
        {
            "gene": "MT-TL1 (tRNA-Leu UUR)",
            "disease": "MELAS / MIDD / Pan-OXPHOS (CI+CIII+CIV)",
            "oxphos": "Pan-OXPHOS: CI+CIII+CIV all reduced",
            "distinguisher": "STROKE-LIKE EPISODES (SLE) hallmark MELAS — ABSENT in MT-TI; Pan-OXPHOS (CIII also reduced) vs MT-TI CI+CIV combined; m.3243AG most common worldwide mtDNA variant",
        },
        {
            "gene": "MT-TK (tRNA-Lys)",
            "disease": "MERRF — Myoclonic Epilepsy Ragged-Red Fibres / MSL",
            "oxphos": "Pan-OXPHOS: CI+CIV reduced (CII normal)",
            "distinguisher": "MYOCLONIC EPILEPSY hallmark MERRF — ABSENT in MT-TI (except Leigh-like m.4309AG high-heteroplasmy); MSL (Madelung disease) distinctive for MT-TK; m.8344AG >80% worldwide",
        },
        {
            "gene": "MT-TV (tRNA-Val — SECOND tRNA)",
            "disease": "CPEO / Myopathy / Exercise Intolerance",
            "oxphos": "CI + CIV ↓, CII normal",
            "distinguisher": "MT-TV (1602–1670) precedes MT-TI (4263–4331); similar CPEO/myopathy phenotype; MT-TV cardiomyopathy ~35% (not isolated HCM); VARS2 nuclear DDx vs IARS2 for MT-TI; MT-TV lacks MT-TQ overlap feature",
        },
        {
            "gene": "MT-TT (tRNA-Thr)",
            "disease": "CPEO / Myopathy / Cardiomyopathy (HCM 55–65%)",
            "oxphos": "CI + CIV ↓, CII normal",
            "distinguisher": "HCM very prominent in MT-TT (55–65%) but NOT isolated — CPEO co-occurs; MT-TI m.4300AG causes ISOLATED HCM without CPEO; AMIODARONE absolute CI in MT-TT too; TARS2 DDx for MT-TT vs IARS2 for MT-TI",
        },
        {
            "gene": "MT-TE (tRNA-Glu)",
            "disease": "CPEO / Myopathy / MIDM-Maternally-Inherited-Diabetes",
            "oxphos": "CI + CIV ↓, CII normal",
            "distinguisher": "MIDM (maternally inherited diabetes) distinctive for MT-TE — absent in MT-TI; m.14674TC REVERSIBLE infantile COX deficiency unique to MT-TE; MT-TE is L-strand encoded (NGS pitfall) — MT-TI is H-strand encoded",
        },
        {
            "gene": "MT-TF (tRNA-Phe — FIRST tRNA)",
            "disease": "CPEO / Myopathy / Exercise Intolerance",
            "oxphos": "CI + CIV ↓, CII normal",
            "distinguisher": "MT-TF (577–647) is genomically first; similar phenotypes but NO isolated HCM; FARS2 nuclear DDx for MT-TF vs IARS2 for MT-TI; MT-TF D-loop adjacent (large deletions can reduce mtDNA copy number)",
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
            "distinguisher": "HEPATOPATHY distinguishes POLG (Alpers phenotype) — ABSENT in MT-TI; autosomal recessive; WES-detectable; mtDNA depletion on Southern blot",
        },
        {
            "gene": "Sarcomeric HCM (MYH7, MYBPC3, TNNI3)",
            "disease": "Familial Hypertrophic Cardiomyopathy — most common",
            "oxphos": "Normal OXPHOS (no CI/CIV deficiency)",
            "distinguisher": "NORMAL OXPHOS on muscle biopsy — critical DDx for m.4300AG isolated HCM; no lactic acidosis at rest; autosomal dominant; WES-detectable; no maternal inheritance; screen MT-TI in any HCM with maternal family history + lactic acidosis",
        },
    ]

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
            "pattern": "Combined CI+CIV deficiency (CII NORMAL) — mt-translation fingerprint; all 13 mtDNA-encoded OXPHOS subunits require tRNA-Ile",
            "BN_PAGE": "CI supercomplex (I+III2+IV) disassembly; isolated CII (nuclear SDH) fully intact; CIV monomer absent/reduced; m.4300AG: cardiac biopsies may show more severe CI+CIV than skeletal muscle",
            "muscle_histochemistry": "RRF on modified Gomori trichrome; COX-negative (CIV↓) + SDH-positive (CII intact); COX/SDH dual stain shows mosaic; m.4300AG: cardiac biopsy shows severe RRF + COX-negative pattern even when skeletal muscle is mild",
            "H_strand_note": "MT-TI is H-strand encoded (no L-strand NGS coverage pitfall). MT-TI is an RNA gene absent from WES exome capture. A dedicated mtDNA sequencing panel (mtDNA-seq / amplicon) is required for MT-TI variant detection. Note: MT-TQ (L-strand, 4329–4400) overlaps at 4329–4331 — L-strand QC must be verified when large deletions are suspected at the MT-TI boundary.",
        },
        "ddx_comparison": ddx_comparison,
        "contraindication_info": contraindication_info,
    }


def get_definitions():
    return {
        "gene_biology": {
            "MT_TI_gene": "Mitochondrially encoded tRNA-Ile (GAU anticodon) — OMIM *590045 — 69 nucleotides; the FOURTH tRNA gene in the human mitochondrial genome, at rCRS 4263–4331, H-strand encoded. Immediately 3′ to MT-ND1 (3307–4262); 3′ boundary overlaps MT-TQ (L-strand, 4329–4400) at rCRS 4329–4331.",
            "genomic_position_FOURTH": "MT-TI (H-strand rCRS 4263–4331) is the fourth tRNA in genome order after MT-TF (1st), MT-TV (2nd), MT-TL1 (3rd). Its 3′ end (4329–4331) overlaps the 5′ start of MT-TQ (L-strand, 4329–4400). This overlap means: (1) Large deletions spanning MT-TI that extend 3′ past position 4329 simultaneously impair MT-TQ. (2) NGS must verify L-strand coverage at 4329–4400 whenever a large deletion is detected at MT-TI boundary.",
            "tRNA_structure": "69-nt cloverleaf tRNA; GAU anticodon at positions 34–36 (C34→U modification post-transcriptionally in human mitochondria allows decoding of AUA = Ile in mt genetic code); acceptor stem 7 bp; D-stem 4 bp; anticodon stem 5 bp; variable loop ~5 nt; T-stem 5 bp. m.4269A>G disrupts position 3–70 Watson-Crick pair in anticodon stem; m.4300A>G is near the acceptor stem / discriminator base (position 72), reducing aminoacylation efficiency by IARS2.",
            "Ile_codons": "AUU, AUC, AUA — all three Ile codons decoded by mt-tRNA-Ile. In the mitochondrial genetic code, AUA = Ile (not Met as in some bacteria). C34→U modification of the anticodon (inosine wobble in nuclear-encoded tRNAs is replaced by U34 modification in mt-tRNA-Ile) allows decoding of all three Ile codons. Isoleucine is a branched-chain hydrophobic amino acid critical for transmembrane helices of CI subunits ND1, ND2, ND4, ND5 and CIV subunits CO1, CO2.",
            "mt_translation_fingerprint": "Pathogenic MT-TI mutations reduce tRNA-Ile availability to all 13 mtDNA-encoded OXPHOS subunits. CII (succinate dehydrogenase, SDHA/B/C/D) is encoded entirely by nuclear DNA — CII NORMAL confirms mt-translation defect vs isolated nuclear OXPHOS gene. CI+CIV combined deficiency with CII NORMAL is the mt-tRNA translation fingerprint.",
            "m4300AG_isolated_HCM": "m.4300A>G (near-acceptor-stem/discriminator-base region) is the ONLY mt-tRNA mutation documented to cause ISOLATED HYPERTROPHIC CARDIOMYOPATHY as the primary/sole phenotype at low heteroplasmy. Cardiac muscle concentrates mt-TI heteroplasmy higher than blood and skeletal muscle; cardiomyocytes are the highest-energy cells and most sensitive to mt-Ile tRNA deficiency in CI/CIV transmembrane helices. HCM can manifest years before any CPEO or skeletal myopathy. Annual echo + Holter is MANDATORY for ALL m.4300A>G carriers from diagnosis, regardless of neuromuscular symptoms.",
            "MTQ_overlap": "MT-TI (H-strand, 4263–4331) and MT-TQ (tRNA-Gln, L-strand, 4329–4400) overlap at rCRS 4329–4331 (3 nucleotides). Large deletions that span the 3′ boundary of MT-TI can simultaneously destroy the 5′ end of MT-TQ, producing a compound tRNA deficiency affecting both mt-tRNA-Ile and mt-tRNA-Gln. This results in a more severe multisystem phenotype than single-tRNA loss alone, typically presenting as KSS with enhanced multi-complex OXPHOS deficiency.",
        },
        "clinical_terms": {
            "CPEO": "Chronic Progressive External Ophthalmoplegia — bilateral ptosis + progressive limitation of eye movements; EOM (extraocular muscles) are mitochondrially rich; CPEO is a major feature of MT-TI disease in non-m.4300AG variants (~72–78%); onset usually 14–55 yr.",
            "Isolated_HCM_m4300AG": "m.4300A>G causes isolated hypertrophic cardiomyopathy — HCM without CPEO. Cardiac heteroplasmy is amplified relative to blood (typically 10–25% higher in cardiac tissue). Annual echocardiography + 24h Holter is mandatory in all m.4300A>G carriers. Beta-blockers first-line (metoprolol/bisoprolol). AMIODARONE ABSOLUTE CI (mt-OXPHOS inhibitor, amplifies CI+CIV failure in cardiomyocytes).",
            "Combined_CI_CIV_Deficiency": "CI (NADH-ubiquinone oxidoreductase, 45 subunits: 7 mtDNA-encoded ND1-6,4L) + CIV (cytochrome c oxidase, 13 subunits: 3 mtDNA-encoded CO1-3) — both reduced because tRNA-Ile is essential for translating all 13 mtDNA-encoded OXPHOS subunits. CI <30% + CIV <35% normal = combined deficiency (spectrophotometric).",
            "Lactic_acidosis": "Blood lactate >2.5 mmol/L; exertional in mild cases; resting in high heteroplasmy; management: thiamine IV + GIR 6–8 + avoid fasting + bicarbonate if pH <7.2.",
            "RRF": "Ragged-Red Fibres — modified Gomori trichrome shows red-staining mitochondrial accumulations; COX-negative (CIV↓) + SDH-positive (CII intact); mt-translation fingerprint.",
            "IARS2_DDx": "mt-Isoleucyl-tRNA Synthetase (IARS2) charges tRNA-Ile with isoleucine. Biallelic IARS2 LOF mutations cause CAGSSS syndrome: Cataracts (congenital/early), growth hormone deficiency, sensory neuropathy, sensorineural hearing loss, skeletal dysplasia — in neonates/children. Neonatal-onset Leigh syndrome is also reported. IARS2 is AR (WES-detectable), neonatal/childhood onset. The biochemical fingerprint (CI+CIV deficiency) overlaps with MT-TI — neonatal/childhood onset with cataracts + growth failure + SNHL + skeletal dysplasia vs adult CPEO/myopathy is the decisive DDx.",
            "CAGSSS": "Cataracts, growth hormone deficiency (dwarfism), Agraphia? — actually: Cataracts, growth hormone deficiency, sensory neuropathy, Sensorineural hearing loss, Skeletal dysplasia — caused by biallelic IARS2 mutations. CAGSSS is the nuclear DDx for MT-TI biochemical phenotype.",
            "KSS": "Kearns-Sayre Syndrome — large mtDNA deletion before age 20 with CPEO + pigmentary retinopathy + cardiac conduction defect (AV block); RRF; CSF protein >100; cerebellar ataxia. MT-TI+MT-TQ junction deletions → KSS with compound tRNA deficiency.",
        },
        "pharmacology": {
            "preferred_aed": "Levetiracetam (LEV) — first-line for any seizures in MT-TI disease. VPA (valproate) is ABSOLUTE CI. Second-line: zonisamide, lacosamide.",
            "cardiac_management": "ALL MT-TI: annual echo + 24h Holter from diagnosis. m.4300AG: mandatory from diagnosis regardless of CPEO. Beta-blockers (metoprolol/bisoprolol) first-line for HCM. AMIODARONE ABSOLUTE CI — mt-OXPHOS inhibitor, compounding CI+CIV failure in cardiomyocytes. AVOID amiodarone for AF/flutter — use beta-blocker rate control. ICD if EF <35%. Cardiac transplant for end-stage HCM/DCM.",
            "emergency_protocol": "GIR 6–8 mg/kg/min (NEVER fast). IV Thiamine 10–20 mg/kg before glucose. Sevoflurane for anaesthesia (NOT propofol). Bicarbonate if pH <7.2. LEV for seizures (VPA absolute CI). Continuous ECG if large deletion KSS (AV block risk). m.4300AG cardiac crisis: beta-blocker NOT amiodarone.",
            "anaesthetic_guidance": "Avoid propofol (PRIS, CIV inhibitor). Use sevoflurane or isoflurane. Avoid prolonged fasting — IV dextrose maintenance. Mitochondrial cocktail continued perioperatively. Regional anaesthesia preferred where possible.",
            "absolute_ci": {
                "Metformin": "Complex I inhibitor → fatal lactic acidosis; use insulin if diabetes",
                "VPA/valproate": "mt-ribosome inhibitor + CoA sequestration; catastrophic; use LEV",
                "Propofol": "CIV inhibitor + PRIS; use sevoflurane for ALL anaesthetics",
                "Linezolid": "mt-23S rRNA inhibitor → blocks ALL 13 mtDNA-encoded OXPHOS translations",
                "Chloramphenicol": "mt-ribosome (70S) inhibitor; blocks mt-translation globally",
                "Amiodarone": "mt-OXPHOS inhibitor; ABSOLUTE CI in m.4300AG HCM — use beta-blocker",
                "Ketogenic diet": "High fat → catabolism; worsens lactic acidosis in CI-dependent NADH oxidation",
            },
        },
        "key_references": [
            "Casali C, Santorelli FM et al. (1995) — A novel mtDNA point mutation in maternally inherited cardiomyopathy. Biochem Biophys Res Commun 213:588. (m.4300A>G isolated HCM landmark).",
            "Merante F, Tein I et al. (1994) — Maternally inherited hypertrophic cardiomyopathy due to a novel T-to-C transition at nucleotide 9997 of the mitochondrial tRNAGly gene. Am J Hum Genet.",
            "Schaefer AM, McFarland R et al. (2008) — Prevalence of mitochondrial DNA disease in adults. Ann Neurol 63:35.",
            "Gorman GS, Chinnery PF et al. (2016) — Mitochondrial diseases. Nat Rev Dis Primers 2:16080.",
            "DiMauro S, Schon EA (2003) — Mitochondrial respiratory-chain diseases. NEJM 348:2656.",
            "Alston CL, Rocha MC et al. (2017) — The genetics and pathology of mitochondrial disease. J Pathol 241:236.",
            "Taylor RW, Turnbull DM (2005) — Mitochondrial DNA mutations in human disease. Nat Rev Genet 6:389.",
            "Fuchs SA, Schene IF et al. (2019) — Aminoacyl-tRNA synthetase deficiencies in search of common themes. Genet Med 21:319. (IARS2 / CAGSSS).",
            "Lott MT, Leipzig JN et al. (2013) — mtDNA variation and analysis using MITOMAP and MITOMASTER. Curr Protoc Bioinformatics.",
            "Kearns TP, Sayre GP (1958) — Retinitis pigmentosa, external ophthalmoplegitis, and complete heart block. AMA Arch Ophthalmol.",
        ],
    }
