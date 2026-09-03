#!/usr/bin/env python3
"""MT-TS2 — Mitochondrially Encoded tRNA-Ser(AGY) — CPEO / Myopathy / SNHL
Combined CI+CIV Deficiency (mt-translation fingerprint)

MT-TS2 (OMIM *590085) encodes mitochondrial tRNA for serine (AGY codons: AGU/AGC),
anticodon GCU, rCRS H-strand positions 12207–12265 (59 bp). At 59 nt, MT-TS2 is the
SHORTEST mitochondrial tRNA gene. Like MT-TH and MT-TK, MT-TS2 is essential for
translating ALL 13 mtDNA-encoded OXPHOS subunits — mutations impair mt-translation,
causing combined CI + CIV deficiency (CII NORMAL — the mt-translation fingerprint).

m.12258C>A (variable loop, position 47 of tRNA-Ser AGY) is the most commonly reported
pathogenic MT-TS2 mutation; it disrupts tRNA folding, impairs Ser aminoacylation, and
causes CPEO + myopathy + sensorineural hearing loss (SNHL). Unlike MT-TH, cardiomyopathy
is less prominent; unlike MT-TK, MSL and myoclonic epilepsy are absent.

  MT-TS2 gene             OMIM *590085
  Primary disease         Combined CI+CIV Deficiency — CPEO / Myopathy / SNHL
                          Exercise Intolerance / Lactic Acidosis
                          Multisystem Encephalomyopathy (high heteroplasmy)
  Protein product         tRNA-Ser(AGY) (GCU anticodon) — 59 nucleotides; RNA gene
                          SHORTEST mitochondrial tRNA gene (59 nt vs 69-74 nt others)
  Genome                  Mitochondrial DNA (mtDNA), H-strand, rCRS 12207–12265
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    Between MT-TH (12138–12206) and MT-TL2 (12266–12336)
  Key mutation            m.12258C>A — variable loop disruption →
                            impaired tRNA-Ser(AGY) fold → pan-OXPHOS (CI+CIV)

HETEROPLASMY THRESHOLD (m.12258C>A — blood underestimates by ~10-15%):
  <50% blood:             Asymptomatic carrier / isolated SNHL only
  50-65% blood:           CPEO + SNHL + exercise intolerance (partial phenotype)
  65-80% blood:           Full CPEO + myopathy + lactic acidosis
  >80% blood:             Multisystem encephalomyopathy + multi-organ + lactic crisis

SHORTEST MITOCHONDRIAL tRNA GENE — MT-TS2 UNIQUE FEATURES:
  At 59 nt, MT-TS2 is the shortest human mitochondrial tRNA gene (vs 69 nt MT-TH/MT-TK,
  74 nt MT-TL1). The compact structure makes it more susceptible to folding disruption
  from single point mutations in any structural domain (D-arm/anticodon-arm truncated
  relative to standard tRNA). This explains the relatively broad phenotypic spectrum
  from mild SNHL at low heteroplasmy to multisystem disease at high heteroplasmy.

PAN-OXPHOS (CI + CIV) — KEY DISTINGUISHER:
  MT-TS2 mutations impair mt-ribosome aminoacylation of Ser(AGY) → defective translation
  of ALL 13 mtDNA-encoded OXPHOS subunits → CI + CIV reduced; CIII/CV variable;
  CII (SDH — nuclear-encoded) NORMAL → CII NORMAL is the mt-translation fingerprint.
  BN-PAGE: CI + CIV reduced pattern; CII band normal.

MT-TS2 TRIAD — CPEO + MYOPATHY + SNHL:
  1. CPEO (chronic progressive external ophthalmoplegia — ptosis + ophthalmoplegia)
  2. Myopathy (limb + oculopharyngeal; RRF in muscle biopsy)
  3. SNHL (sensorineural hearing loss — can be isolated at low heteroplasmy)
  Plus: exercise intolerance, lactic acidosis, variable encephalomyopathy.
  NO myoclonic epilepsy as cardinal feature (distinguishes from MT-TK MERRF).
  NO stroke-like episodes (distinguishes from MT-TL1 MELAS).
  NO MSL/Madelung lipomatosis (distinguishes from MT-TK MERRF).
  Less cardiomyopathy than MT-TH (distinguishes from MT-TH).
"""

import random
from collections import Counter

SEED = 793
N_PATIENTS = 40

VARIANTS = [
    ("m.12258C>A", "Variable loop (position 47)", "~38% of MT-TS2 cases; variable loop disruption impairs tRNA tertiary fold; combined CI+CIV; CPEO + myopathy + SNHL; Jaksch 1998 first description"),
    ("m.12236T>C", "Anticodon stem (position 26)", "~25%; anticodon stem disruption; exercise intolerance + CPEO; adult onset; slower progression than m.12258C>A; SNHL variable"),
    ("m.12246G>A", "Anticodon loop (position 36)", "~20%; anticodon loop disruption; multisystem encephalomyopathy + SNHL; higher heteroplasmy needed for severe phenotype"),
    ("m.12270T>C", "Acceptor stem (position 60)", "~12%; acceptor stem disruption; myopathy + lactic acidosis; less CPEO; RRF on muscle biopsy; adult presentation"),
    ("Large deletion", "Multi-gene spanning", "~5%; deletion spanning MT-TS2 ± MT-TH/MT-TL2 region → KSS/CPEO/Pearson; multi-complex OXPHOS; annual Holter mandatory"),
]

PHENOTYPES = [
    "CPEO + myopathy + SNHL — intermediate heteroplasmy",
    "Exercise intolerance + lactic acidosis — low-moderate heteroplasmy",
    "Multisystem encephalomyopathy + CPEO + SNHL — high heteroplasmy",
    "KSS overlap (large deletion: CPEO + retinal pigment + cardiac block)",
    "Asymptomatic / isolated SNHL — low heteroplasmy",
]

TRIGGERS = [
    "Intercurrent illness (fever/infection)",
    "Fasting / prolonged NPO",
    "Valproate / contraindicated drug exposure",
    "Surgery / general anaesthesia (propofol)",
    "Extreme physical exertion",
    "Alcohol",
]

ABSOLUTE_CI = [
    ("Metformin", "ABSOLUTE CI", "Complex I inhibitor → fatal lactic acidosis in CI-deficient states; applies to ALL MT-TS2 carriers regardless of phenotype severity or heteroplasmy level"),
    ("Valproate (VPA)", "ABSOLUTE CI", "mt-ribosome inhibitor + CoA sequestration + POLG inhibition; impairs tRNA-Ser aminoacylation recovery; hepatotoxic in CI+CIV deficiency"),
    ("Linezolid", "ABSOLUTE CI", "mt-23S rRNA inhibitor → blocks synthesis of all 13 mtDNA OXPHOS subunits; compounding CI+CIV failure; fatal in mitochondrial myopathy"),
    ("Chloramphenicol", "ABSOLUTE CI", "mt-ribosome inhibitor; same mechanism as linezolid for all mtDNA-encoded OXPHOS subunits; fatal myelosuppression risk in mitochondrial disease"),
    ("Propofol", "ABSOLUTE CI (PRIS)", "Propofol infusion syndrome — inhibits OXPHOS at CIV; amplified lethality with pre-existing CI+CIV deficiency; use sevoflurane instead"),
    ("KD (Ketogenic Diet)", "CONTRAINDICATED", "CI+CIV failure → FADH2 backup blocks beta-oxidation; ketogenic metabolism requires functional OXPHOS; catastrophic ketoacidosis risk"),
    ("IV tPA", "CAUTION / CI", "Multisystem MT-TS2 may mimic ischaemic stroke → IV tPA haemorrhage risk in metabolic lesions; exclude ischaemia with DWI/PWI/MRA before any tPA"),
    ("Aminoglycosides", "HIGH CAUTION", "Cochlear OXPHOS amplification risk; MT-TS2 SNHL patients have vulnerable spiral ganglion neurons; aminoglycosides can precipitate/worsen SNHL dramatically"),
]

TREATMENTS = [
    ("Thiamine B1", "Level C", "MANDATORY empiric — cofactor for PDH/KGDH; lactic acidosis crisis: 10-20 mg/kg IV; oral maintenance 100-300 mg/day; SLC19A3/BTD exclusion first"),
    ("Biotin", "Level C", "MANDATORY empiric — BTBGD/BTD exclusion (treatable Leigh-like mimic); oral 10-20 mg/day; low cost/risk, high benefit if BTBGD missed"),
    ("CoQ10 Ubiquinol", "Level C", "300-1200 mg/day divided doses; electron carrier CI→CII→CIII; ubiquinol preferred over ubiquinone in CI+CIV deficiency; titrate over weeks"),
    ("Riboflavin B2", "Level C", "100-400 mg/day; FAD/FMN cofactor for CI N-module and CIII; may marginally improve residual CI+CIV activity; trial 3-6 months minimum"),
    ("L-Carnitine", "Level C", "1-3 g/day; mitochondrial beta-oxidation support; carnitine depletion common in CI+CIV deficiency; avoid masking VPA toxicity symptoms"),
    ("Levetiracetam (LEV)", "Preferred AED", "Renal excretion; no mitochondrial toxicity; first-line for seizures in MT-TS2 disease; avoid VPA (ABSOLUTE CI); avoid CBZ/OXC (worsens fatigue)"),
    ("GIR 6-8 mg/kg/min IV dextrose", "Acute crisis", "NEVER fast — fasting triggers lactic crisis; IV glucose infusion during illness/perioperative NPO; prevents catabolism of CI+CIV-deficient cells"),
    ("Ophthalmology", "MANDATORY", "Annual ocular motility exam + ptosis assessment + fundoscopy; prism correction for diplopia; ptosis surgery last resort; fundoscopy for KSS overlap"),
    ("Audiology / cochlear implant", "Effective for SNHL", "MT-TS2 SNHL responds to cochlear implant; refer early (before profound loss); audiological monitoring every 6-12 months; avoid aminoglycosides"),
]

rng = random.Random(SEED)

def _make_patients():
    patients = []
    variant_weights = [0.38, 0.25, 0.20, 0.12, 0.05]
    phenotype_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
    for i in range(N_PATIENTS):
        var_idx = rng.choices(range(len(VARIANTS)), weights=variant_weights)[0]
        phen_idx = rng.choices(range(len(PHENOTYPES)), weights=phenotype_weights)[0]
        het_base = rng.randint(42, 92)
        age_onset = rng.randint(0, 58)
        has_cpeo = phen_idx in (0, 2, 3)
        has_myopathy = phen_idx in (0, 1, 2)
        has_encephalo = phen_idx == 2
        has_kss = phen_idx == 3
        has_snhl = rng.random() < 0.62
        has_cardio = rng.random() < 0.15  # less than MT-TH
        has_dm = rng.random() < 0.10
        has_neuropathy = rng.random() < 0.28
        has_rrm = rng.random() < 0.45
        patients.append({
            "id": f"MTTS2-{i+1:03d}",
            "variant": VARIANTS[var_idx][0],
            "phenotype": PHENOTYPES[phen_idx],
            "heteroplasmy_blood": het_base,
            "age_onset": age_onset,
            "cpeo": has_cpeo,
            "myopathy": has_myopathy,
            "encephalomyopathy": has_encephalo,
            "kss": has_kss,
            "snhl": has_snhl,
            "cardiomyopathy": has_cardio,
            "diabetes": has_dm,
            "neuropathy": has_neuropathy,
            "ragged_red_fibres": has_rrm,
            "ci_activity": round(rng.uniform(8, 38), 1),
            "civ_activity": round(rng.uniform(10, 42), 1),
            "cii_activity": round(rng.uniform(85, 115), 1),
            "lactate": round(rng.uniform(1.8, 11.5), 1),
            "sex": rng.choice(["M", "F"]),
        })
    return patients

_PATIENTS = _make_patients()


def get_overview():
    pts = _PATIENTS
    n = len(pts)
    variant_counts = Counter(p["variant"] for p in pts)
    phenotype_counts = Counter(p["phenotype"] for p in pts)
    avg_het = round(sum(p["heteroplasmy_blood"] for p in pts) / n, 1)
    avg_onset = round(sum(p["age_onset"] for p in pts) / n, 1)
    avg_ci = round(sum(p["ci_activity"] for p in pts) / n, 1)
    avg_civ = round(sum(p["civ_activity"] for p in pts) / n, 1)
    avg_cii = round(sum(p["cii_activity"] for p in pts) / n, 1)
    pct_cpeo = round(sum(1 for p in pts if p["cpeo"]) / n * 100)
    pct_myopathy = round(sum(1 for p in pts if p["myopathy"]) / n * 100)
    pct_snhl = round(sum(1 for p in pts if p["snhl"]) / n * 100)
    pct_cardio = round(sum(1 for p in pts if p["cardiomyopathy"]) / n * 100)
    pct_rrm = round(sum(1 for p in pts if p["ragged_red_fibres"]) / n * 100)

    return {
        "gene": "MT-TS2",
        "omim_gene": "590085",
        "full_name": "Mitochondrially Encoded tRNA-Ser(AGY) — tRNA-Serine (AGY codons)",
        "rCRS_position": "H-strand 12207–12265 (59 nt) — SHORTEST mitochondrial tRNA",
        "flanking_genes": "MT-TH (12138–12206) → MT-TS2 → MT-TL2 (12266–12336) → MT-ND5",
        "inheritance": "MATERNAL — heteroplasmic",
        "primary_disease": "Combined CI+CIV Deficiency — CPEO / Myopathy / SNHL",
        "key_mutation": "m.12258C>A (variable loop position 47 — ~38% of MT-TS2 cases)",
        "oxphos_fingerprint": "CI + CIV reduced; CII (SDH — nuclear-encoded) NORMAL — mt-translation fingerprint",
        "unique_feature": "SHORTEST mt-tRNA at 59 nt (vs 69 nt MT-TH/MT-TK; 74 nt MT-TL1) — compact structure amplifies folding vulnerability",
        "cohort_statistics": {
            "n_patients": n,
            "avg_heteroplasmy_blood_pct": avg_het,
            "avg_age_onset_yr": avg_onset,
            "avg_ci_activity_pct_normal": avg_ci,
            "avg_civ_activity_pct_normal": avg_civ,
            "avg_cii_activity_pct_normal": avg_cii,
            "pct_cpeo": pct_cpeo,
            "pct_myopathy": pct_myopathy,
            "pct_snhl": pct_snhl,
            "pct_cardiomyopathy": pct_cardio,
            "pct_ragged_red_fibres": pct_rrm,
        },
        "variant_distribution": [
            {"variant": v[0], "region": v[1], "count": variant_counts.get(v[0], 0),
             "pct": round(variant_counts.get(v[0], 0) / n * 100), "note": v[2]}
            for v in VARIANTS
        ],
        "phenotype_distribution": [
            {"phenotype": ph, "count": phenotype_counts.get(ph, 0),
             "pct": round(phenotype_counts.get(ph, 0) / n * 100)}
            for ph in PHENOTYPES
        ],
        "heteroplasmy_clinical_map": [
            {"range": "<50% blood", "phenotype": "Asymptomatic / isolated SNHL", "management": "Annual audiological review, genetic counselling, avoid aminoglycosides"},
            {"range": "50–65% blood", "phenotype": "CPEO + SNHL + exercise intolerance", "management": "Ophthalmology, audiology, CoQ10 ubiquinol, avoid CI drugs"},
            {"range": "65–80% blood", "phenotype": "Full CPEO + myopathy + lactic acidosis", "management": "Multidisciplinary: neurology + ophthalmology + audiology; thiamine empiric"},
            {"range": ">80% blood", "phenotype": "Multisystem encephalomyopathy + multi-organ + crisis", "management": "ICU: GIR 6–8, IV thiamine, bicarb if pH <7.2, avoid fasting/propofol"},
        ],
        "key_molecular_features": [
            "MT-TS2 encodes tRNA-Ser(AGY) (GCU anticodon) — 59 nt RNA gene; SHORTEST human mitochondrial tRNA",
            "Flanked by MT-TH (12138–12206) and MT-TL2 (12266–12336); immediately adjacent to MT-TH",
            "CI + CIV combined deficiency (both mt-encoded); CII NORMAL (nuclear) = mt-translation fingerprint",
            "m.12258C>A disrupts variable loop position 47 → impairs tRNA-Ser(AGY) tertiary folding → aminoacylation failure",
            "Blood heteroplasmy underestimates by 10–15%; muscle biopsy preferred for threshold assessment",
            "WES/WGS misses MT-TS2 — dedicated mtDNA panel (NGS or long-read) mandatory",
            "Isolated SNHL at low heteroplasmy (<50%) is a DISTINCTIVE feature — may predate CPEO/myopathy by years",
            "CPEO prominent (like MT-TH); less cardiomyopathy than MT-TH; NO MSL (vs MT-TK MERRF)",
            "NO myoclonic epilepsy as cardinal (distinguishes from MT-TK MERRF)",
            "NO stroke-like episodes (distinguishes from MT-TL1 MELAS)",
            "BTBGD (SLC19A3) MANDATORY exclusion — treatable Leigh-like mimic; thiamine + biotin empiric",
            "Aminoglycosides HIGH CAUTION — cochlear OXPHOS amplification worsens SNHL dramatically",
        ],
        "clinical_alerts": [
            {"alert": "METFORMIN ABSOLUTE CI", "detail": "Complex I inhibitor → fatal lactic acidosis; applies to ALL MT-TS2 carriers including subclinical/SNHL-only"},
            {"alert": "VPA ABSOLUTE CI", "detail": "mt-ribosome inhibitor + CoA sequestration + POLG; catastrophic in CI+CIV deficiency; use LEV instead"},
            {"alert": "PROPOFOL ABSOLUTE CI (PRIS)", "detail": "CIV inhibitor + propofol infusion syndrome; amplified lethality in pre-existing CI+CIV deficiency; use sevoflurane"},
            {"alert": "LINEZOLID ABSOLUTE CI", "detail": "mt-23S rRNA inhibition blocks all 13 mtDNA OXPHOS translations — compounding CI+CIV failure"},
            {"alert": "AMINOGLYCOSIDES HIGH CAUTION", "detail": "Cochlear OXPHOS amplification in MT-TS2 SNHL; can precipitate profound deafness; avoid gentamicin/amikacin/tobramycin"},
            {"alert": "NEVER FAST", "detail": "Fasting triggers acute lactic crisis; GIR 6–8 mg/kg/min IV dextrose mandatory during illness/perioperative NPO"},
            {"alert": "KD CONTRAINDICATED", "detail": "CI+CIV failure blocks beta-oxidation FADH2 processing; ketogenic diet causes catastrophic ketoacidosis"},
            {"alert": "BTBGD EXCLUSION MANDATORY", "detail": "SLC19A3 Biotin-Thiamine-Responsive Basal Ganglia Disease — treatable Leigh-like mimic; MANDATORY first exclusion"},
        ],
        "cohort_summary_features": [
            f"{pct_cpeo}% have CPEO (ptosis + ophthalmoplegia)",
            f"{pct_myopathy}% have myopathy (limb + oculopharyngeal)",
            f"{pct_snhl}% have sensorineural hearing loss",
            f"{pct_cardio}% have cardiomyopathy (less prominent than MT-TH)",
            f"{pct_rrm}% have ragged-red fibres on muscle biopsy",
            f"Mean CI activity {avg_ci}% of normal (CIV {avg_civ}%; CII {avg_cii}% — NORMAL)",
            f"Mean blood heteroplasmy {avg_het}% (muscle ~10–15% higher)",
            f"Mean age at onset {avg_onset} yr (range neonatal–58 yr)",
            "MT-TS2 = SHORTEST mt-tRNA (59 nt) — broader phenotypic spectrum from isolated SNHL to multisystem",
        ],
    }


def get_breakdown():
    pts = _PATIENTS
    by_variant = {}
    for p in pts:
        v = p["variant"]
        if v not in by_variant:
            by_variant[v] = []
        by_variant[v].append(p)

    variant_summaries = []
    for v_name, v_desc, v_note in VARIANTS:
        grp = by_variant.get(v_name, [])
        if not grp:
            continue
        avg_het = round(sum(x["heteroplasmy_blood"] for x in grp) / len(grp), 1)
        avg_ci = round(sum(x["ci_activity"] for x in grp) / len(grp), 1)
        avg_civ = round(sum(x["civ_activity"] for x in grp) / len(grp), 1)
        pct_cpeo = round(sum(1 for x in grp if x["cpeo"]) / len(grp) * 100)
        pct_myopathy = round(sum(1 for x in grp if x["myopathy"]) / len(grp) * 100)
        pct_snhl = round(sum(1 for x in grp if x["snhl"]) / len(grp) * 100)
        pct_cardio = round(sum(1 for x in grp if x["cardiomyopathy"]) / len(grp) * 100)
        variant_summaries.append({
            "variant": v_name,
            "region": v_desc,
            "note": v_note,
            "n": len(grp),
            "avg_heteroplasmy_blood_pct": avg_het,
            "avg_ci_activity_pct": avg_ci,
            "avg_civ_activity_pct": avg_civ,
            "pct_cpeo": pct_cpeo,
            "pct_myopathy": pct_myopathy,
            "pct_snhl": pct_snhl,
            "pct_cardiomyopathy": pct_cardio,
        })

    per_patient = [
        {
            "id": p["id"],
            "variant": p["variant"],
            "sex": p["sex"],
            "age_onset_yr": p["age_onset"],
            "heteroplasmy_blood_pct": p["heteroplasmy_blood"],
            "ci_pct": p["ci_activity"],
            "civ_pct": p["civ_activity"],
            "cii_pct": p["cii_activity"],
            "lactate_mmol_L": p["lactate"],
            "cpeo": p["cpeo"],
            "myopathy": p["myopathy"],
            "kss": p["kss"],
            "snhl": p["snhl"],
            "cardiomyopathy": p["cardiomyopathy"],
            "ragged_red_fibres": p["ragged_red_fibres"],
            "neuropathy": p["neuropathy"],
            "phenotype_short": PHENOTYPES.index(p["phenotype"]) + 1,
        }
        for p in pts
    ]

    trigger_rates = [
        {"trigger": t, "pct": rng.randint(22, 82)} for t in TRIGGERS
    ]
    trigger_rates.sort(key=lambda x: -x["pct"])

    treatment_info = [
        {"agent": t[0], "evidence": t[1], "note": t[2]} for t in TREATMENTS
    ]
    ci_info = [
        {"agent": a[0], "category": a[1], "rationale": a[2]} for a in ABSOLUTE_CI
    ]

    return {
        "variant_summaries": variant_summaries,
        "per_patient": per_patient,
        "trigger_rates": trigger_rates,
        "treatment_info": treatment_info,
        "contraindication_info": ci_info,
        "biochemical_fingerprint": {
            "CI_pct_normal": f"{round(sum(p['ci_activity'] for p in pts)/len(pts),1)}",
            "CIV_pct_normal": f"{round(sum(p['civ_activity'] for p in pts)/len(pts),1)}",
            "CII_pct_normal": f"{round(sum(p['cii_activity'] for p in pts)/len(pts),1)} (NORMAL — nuclear-encoded)",
            "pattern": "CI + CIV reduced; CII NORMAL → mt-translation fingerprint",
            "BN_PAGE": "CI band absent/reduced; CIV (COX) absent/reduced; CII band PRESENT (SDH-positive)",
            "muscle_histochemistry": "RRF on Gomori trichrome; COX-negative fibres at moderate-high heteroplasmy; SDH-positive ragged fibres",
        },
        "ddx_comparison": [
            {
                "gene": "MT-TH",
                "disease": "Combined CI+CIV — Leigh-like + CPEO + Cardiomyopathy",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "More prominent cardiomyopathy (HCM/DCM); Leigh-like MRI more frequent at lower heteroplasmy; less isolated SNHL presentation; MT-TH immediately flanks MT-TS2 (12138 vs 12207 rCRS)",
            },
            {
                "gene": "MT-TK",
                "disease": "MERRF (Myoclonic Epilepsy + RRF + MSL)",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "Myoclonic epilepsy cardinal (90%); MSL/lipomatosis (10-25%) — ABSENT in MT-TS2; progressive myoclonic epilepsy archetype; NO isolated SNHL; Shoffner 1990",
            },
            {
                "gene": "MT-TL1",
                "disease": "MELAS (Stroke-like Episodes + MIDD)",
                "oxphos": "CI+CIII+CIV (pan-OXPHOS)",
                "distinguisher": "Stroke-like episodes crossing vascular territories (ABSENT in MT-TS2); MRI cortical/subcortical NOT bilateral BG; SSSVS; MIDD at low heteroplasmy; Goto 1990",
            },
            {
                "gene": "SURF1",
                "disease": "Leigh Syndrome (isolated CIV)",
                "oxphos": "CIV isolated (CI/CII/CIII normal)",
                "distinguisher": "Isolated CIV (not combined CI+CIV); COX-negative muscle; AR nuclear gene; WES detectable; NO maternal inheritance; hairy-vascular lesions in some",
            },
            {
                "gene": "SLC19A3 (BTBGD)",
                "disease": "Biotin-Thiamine-Responsive BGD",
                "oxphos": "Normal OXPHOS biochemistry",
                "distinguisher": "TREATABLE — thiamine + biotin → dramatic MRI recovery; Leigh-like signal abnormality on MRI but normal CI/CIV biochemistry; MANDATORY first exclusion before MT-TS2 confirmed",
            },
        ],
    }


def get_definitions():
    return {
        "gene_biology": {
            "MT-TS2": "Mitochondrially encoded tRNA-Ser(AGY) (GCU anticodon); 59 nt RNA gene; H-strand rCRS 12207–12265; SHORTEST human mitochondrial tRNA gene; required for mt-ribosomal synthesis of all 13 mtDNA OXPHOS subunits; located between MT-TH and MT-TL2",
            "tRNA-Ser_AGY_function": "Delivers serine to the mt-ribosome A-site during translation of ALL 13 mtDNA-encoded subunits; anticodon GCU recognises AGU and AGC (Ser-AGY codons); mt-aminoacyl-tRNA synthetase: SARS2 (mitochondrial seryl-tRNA synthetase); SARS2 mutations cause HUPRA syndrome (phenocopy consideration)",
            "mt_translation_fingerprint": "CI + CIV reduced; CII (SDH, nuclear-encoded) NORMAL; BN-PAGE: CI and CIV bands absent/reduced; CII band present; distinguishes mt-tRNA mutations from isolated nuclear gene OXPHOS defects",
            "rCRS_position": "H-strand rCRS 12207–12265 (59 bp); IMMEDIATELY flanked by MT-TH (12138–12206) and MT-TL2 (12266–12336); this tRNA cluster (MT-TH, MT-TS2, MT-TL2) spans rCRS 12138–12336 — important for large deletion mapping",
            "SHORTEST_tRNA": "At 59 nt, MT-TS2 is the shortest human mitochondrial tRNA. Standard cytoplasmic tRNAs are 73-93 nt; mt-tRNAs range 59-75 nt. The compact MT-TS2 structure (truncated D-arm and variable loop) amplifies the functional impact of single-nucleotide mutations — a point mutation anywhere destabilises the entire fold more readily than in longer mt-tRNAs.",
        },
        "clinical_terms": {
            "CPEO": "Chronic Progressive External Ophthalmoplegia: slowly progressive bilateral ptosis + ophthalmoparesis; EOMs eventually all affected; most prominent MT-TS2 feature; KSS if large deletion (CPEO + retinal pigmentation + cardiac conduction block <20 yr)",
            "Isolated_SNHL": "Sensorineural hearing loss as SOLE presentation at low heteroplasmy (<50%) — DISTINCTIVE for MT-TS2; may predate CPEO and myopathy by years; cochlear implant effective; audiological monitoring every 6-12 months mandatory",
            "mt_translation_fingerprint": "CI + CIV reduced; CII NORMAL — reflects that all 7 CI ND-subunits (ND1-ND6, ND4L) and all 3 CIV CO-subunits (CO1-CO3) are mtDNA-encoded and require tRNA-Ser(AGY); CII (SDHA-D) all nuclear-encoded → NORMAL",
            "heteroplasmy_threshold": "Minimum mutant mtDNA load for disease expression; MT-TS2 m.12258C>A: ~50% blood for isolated SNHL; ~65% for CPEO+myopathy; >80% for multisystem encephalomyopathy; blood underestimates muscle by 10–15%",
            "BTBGD": "Biotin-Thiamine-Responsive Basal Ganglia Disease (SLC19A3); Leigh-like MRI; TREATABLE with thiamine + biotin; MANDATORY first exclusion before attributing multisystem features to MT-TS2",
            "KSS": "Kearns-Sayre Syndrome: CPEO + retinal pigmentary degeneration + cardiac conduction block onset <20 yr; caused by large mtDNA deletions (spanning MT-TS2 ± flanking genes); annual Holter mandatory (complete heart block risk)",
            "HUPRA_syndrome": "HUPRA (Hyperuricemia, Pulmonary hypertension, Renal failure, Alkalosis) — caused by SARS2 mutations (mt-seryl-tRNA synthetase, phenocopy of MT-TS2); AR nuclear; WES detectable; DIFFERENT from maternal mt-tRNA MT-TS2 mutations",
        },
        "pharmacology": {
            "absolute_ci": {a[0]: a[2] for a in ABSOLUTE_CI},
            "preferred_aed": "Levetiracetam (LEV) — renal excretion; no mt-toxicity; first-line for seizures in MT-TS2 disease; avoid VPA (ABSOLUTE CI), CBZ/OXC (worsens fatigue in myopathy)",
            "emergency_protocol": "GIR 6–8 mg/kg/min IV dextrose + IV thiamine 10–20 mg/kg + avoid fasting + bicarb for pH <7.2 + ICU if multisystem crisis",
            "aminoglycoside_note": "AVOID gentamicin/amikacin/tobramycin/streptomycin — cochlear OXPHOS amplification worsens MT-TS2 SNHL; if systemic infection mandates aminoglycoside, use lowest effective dose + TDM + ENT monitoring",
        },
        "key_references": [
            "Jaksch M et al. (1998) Mutations of tRNA(Ser)(UCN) and tRNA(His) genes causing hereditary mitochondrial myopathy — Hum Mol Genet (first systematic MT-TS2 mutation study)",
            "Hao H et al. (2005) Mitochondrial tRNA(Ser(AGY)) mutation m.12258C>A in maternally inherited CPEO and SNHL — Biochem Biophys Res Commun",
            "Rossmanith W (2012) Of P and Z clusters: mitochondrial tRNA 3' end processing activities — FEBS Lett (MT-TS2 structural biology, shortest mt-tRNA)",
            "Gorman GS et al. (2015) Mitochondrial diseases — Nat Rev Dis Primers (comprehensive review including mt-tRNA spectrum)",
            "DiMauro S & Schon EA (2003) Mitochondrial respiratory-chain diseases — NEJM (tRNA mutation review including MT-TS2 context)",
            "Zeviani M & Carelli V (2021) Mitochondrial retinopathies — Int J Mol Sci (OXPHOS disease spectrum, mt-tRNA phenotypes)",
        ],
    }
