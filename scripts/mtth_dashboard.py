#!/usr/bin/env python3
"""MT-TH — Mitochondrially Encoded tRNA-His — Combined CI+CIV Deficiency
Leigh-like Syndrome / CPEO / Cardiomyopathy / Sensorineural Hearing Loss

MT-TH (OMIM *590080) encodes the mitochondrial tRNA for histidine (anticodon GUG),
rCRS H-strand positions 12138–12206 (69 bp). Like MT-TK and MT-TL1, MT-TH is a tRNA
gene essential for translating ALL 13 mtDNA-encoded OXPHOS subunits — mutations impair
mt-translation globally, causing pan-OXPHOS deficiency (predominantly CI + CIV;
CII/SDH nuclear-encoded: NORMAL — the mt-translation biochemical fingerprint).

m.12147G>A (D-loop/variable-loop junction of tRNA-His) is the most commonly reported
MT-TH pathogenic mutation; it disrupts tRNA tertiary folding, impairs Lys-tRNA
aminoacylation, and causes combined CI+CIV deficiency → Leigh-like infantile presentation.

  MT-TH gene             OMIM *590080
  Primary disease        Combined CI+CIV Deficiency — Leigh-like Syndrome
                         CPEO (Chronic Progressive External Ophthalmoplegia)
                         Cardiomyopathy (HCM/DCM)
                         Exercise Intolerance / Myopathy / SNHL
  Protein product        tRNA-His (GUG anticodon) — 69 nucleotides; RNA gene
  Genome                 Mitochondrial DNA (mtDNA), H-strand, rCRS 12138–12206
  Inheritance            MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location   Between MT-ND4 (10760–12137) and MT-TS2 (12207–12265)
  Key mutation           m.12147G>A — D-loop/variable-loop junction disruption →
                           impaired tRNA tertiary folding → pan-OXPHOS (CI+CIV)

HETEROPLASMY THRESHOLD (m.12147G>A — blood underestimates by ~10-15%):
  <55% blood:             Asymptomatic carrier / mild exercise intolerance
  55-70% blood:           CPEO + SNHL + mild myopathy (partial phenotype)
  70-85% blood:           Full Leigh-like + CPEO + cardiomyopathy + SNHL
  >85% blood:             Severe Leigh + multi-organ failure + lactic acidosis crisis
  Muscle biopsy preferred for heteroplasmy in equivocal blood cases (>20% higher)

PAN-OXPHOS (CI + CIV PREDOMINANTLY) — KEY DISTINGUISHER:
  MT-TH mutations impair mt-ribosome aminoacylation of His → defective translation
  of ALL 13 mtDNA-encoded OXPHOS subunits → CI + CIV reduced; CIII/CV variable;
  CII (SDH — nuclear-encoded) NORMAL → CII NORMAL is the mt-translation fingerprint.
  BN-PAGE: CI + CIV reduced pattern; CII band normal.
  MT-TH CI+CIV pattern ≠ isolated CI (Leigh/LHON) ≠ isolated CIV (SURF1/SCO2).

MT-TH TRIAD — LEIGH-LIKE + CPEO + CARDIOMYOPATHY:
  1. Leigh-like syndrome (bilateral symmetric BG/brainstem MRI at high heteroplasmy)
  2. CPEO (chronic progressive external ophthalmoplegia — ptosis + ophthalmoplegia)
  3. Cardiomyopathy (HCM or DCM — mandatory annual echo/Holter)
  Plus: SNHL, lactic acidosis, myopathy, dementia, peripheral neuropathy.
  NO stroke-like episodes (distinguishes from MT-TL1 MELAS).
  NO MSL (distinguishes from MT-TK MERRF).
  NO primary LHON (distinguishes from MT-ND4/MT-ND1/MT-ND6).

CPEO — KEY FEATURE (more prominent than MT-TK/MT-TL1):
  Progressive bilateral ptosis + external ophthalmoplegia (EOMs spared for gaze until
  late); diplopia rare; Kearns-Sayre overlap if deletion spans MT-TH region;
  annual ophthalmology mandatory; prism glasses for ptosis/diplopia compensation;
  ptosis surgery last resort (risk of corneal exposure keratopathy).
"""

import random
from collections import Counter

SEED = 791
N_PATIENTS = 40

VARIANTS = [
    ("m.12147G>A", "D-loop/variable-loop junction", "~40% of MT-TH cases; disrupts tRNA-His tertiary fold; impairs His aminoacylation; combined CI+CIV deficiency; classic Leigh-like infantile phenotype + CPEO; most studied MT-TH mutation"),
    ("m.12192G>A", "Anticodon stem-loop", "~25%; anticodon stem-loop structural disruption; CPEO + myopathy + adult-onset; slower progression; milder CI+CIV residual activity than m.12147G>A"),
    ("m.12183T>C", "Acceptor stem", "~20%; acceptor stem disruption; moderate combined CI+CIV; Leigh-like childhood onset; SNHL prominent; CPEO variable penetrance"),
    ("m.12169T>C", "D-loop", "~10%; D-loop region; CPEO + cardiomyopathy adult-onset; mild CI+CIV; SNHL; may mimic MT-TL1 but no stroke-like episodes"),
    ("Large deletion", "Multi-gene spanning", "~5%; deletion spanning MT-TH ± MT-TS2/MT-ND4 region → KSS/CPEO/Pearson; multi-complex OXPHOS; annual Holter mandatory; KSS triad if complete"),
]

PHENOTYPES = [
    "Leigh-like (bilateral BG/brainstem MRI + CI+CIV) — high heteroplasmy",
    "CPEO + myopathy + SNHL — intermediate heteroplasmy",
    "Cardiomyopathy (HCM/DCM) + CPEO + lactic acidosis",
    "KSS overlap (large deletion: CPEO + cardiomyopathy + retinal pigment)",
    "Exercise intolerance / asymptomatic carrier — low heteroplasmy",
]

TRIGGERS = [
    "Intercurrent illness (fever/infection)",
    "Fasting / prolonged NPO",
    "Valproate / contraindicated drug exposure",
    "Surgery / general anaesthesia",
    "Extreme physical exertion",
    "Alcohol",
]

ABSOLUTE_CI = [
    ("Metformin", "ABSOLUTE CI", "Complex I inhibitor → fatal lactic acidosis in CI-deficient states; applies to ALL MT-TH carriers regardless of phenotype severity"),
    ("Valproate (VPA)", "ABSOLUTE CI", "mt-ribosome inhibitor + CoA sequestration + POLG inhibition; impairs tRNA-His aminoacylation recovery; hepatotoxic in CI+CIV deficiency"),
    ("Linezolid", "ABSOLUTE CI", "mt-23S rRNA inhibitor → blocks synthesis of all 13 mtDNA OXPHOS subunits including ND1-6, CO1-3, ATP6/8, CYB; compounding CI+CIV failure"),
    ("Chloramphenicol", "ABSOLUTE CI", "mt-ribosome inhibitor; same mechanism as linezolid for mtDNA-encoded OXPHOS; fatal myelosuppression risk in mitochondrial disease"),
    ("Propofol", "ABSOLUTE CI (PRIS)", "Propofol infusion syndrome — inhibits OXPHOS at CIV; amplified lethality with pre-existing CI+CIV deficiency; use sevoflurane instead"),
    ("KD (Ketogenic Diet)", "CONTRAINDICATED", "CI+CIV failure → FADH2 backup blocks beta-oxidation; ketogenic metabolism requires functional OXPHOS; catastrophic ketoacidosis risk"),
    ("IV tPA", "CAUTION / CI", "Leigh-like MRI mimics ischaemic stroke → IV tPA carries haemorrhage risk in metabolic lesions; NEVER give without DWI/PWI/MRA to exclude ischaemia"),
    ("Statins", "HIGH CAUTION", "CoQ10 depletion via mevalonate pathway blockade; CI+CIV deficiency worsens with additional CoQ10 depletion; lowest dose + ubiquinol supplementation"),
]

TREATMENTS = [
    ("Thiamine B1", "Level C", "MANDATORY empiric — cofactor for PDH/KGDH; lactic acidosis crisis: 10-20 mg/kg IV; oral maintenance 100-300 mg/day; SLC19A3/BTD exclusion first"),
    ("Biotin", "Level C", "MANDATORY empiric — BTBGD/BTD exclusion (treatable Leigh mimic); oral 10-20 mg/day"),
    ("CoQ10 Ubiquinol", "Level C", "300-1200 mg/day in divided doses; electron carrier between CI→CII→CIII; ubiquinol > ubiquinone in CI+CIV deficiency; titrate slowly"),
    ("Riboflavin B2", "Level C", "100-400 mg/day; FAD/FMN cofactor for CI N-module NDUFV2 and CIII; may improve residual CI+CIV activity marginally"),
    ("L-Carnitine", "Level C", "1-3 g/day; mitochondrial beta-oxidation support; carnitine depletion common in CI+CIV deficiency; avoid in VPA toxicity (may mask)"),
    ("Levetiracetam (LEV)", "Preferred AED", "Renal excretion; no mitochondrial toxicity; first-line for seizures in MT-TH disease; avoid VPA (ABSOLUTE CI), CBZ/OXC (sodium channel — worsens fatigue)"),
    ("GIR 6-8 mg/kg/min IV dextrose", "Acute crisis", "NEVER fast — fasting triggers lactic crisis; IV glucose infusion during illness/perioperative; prevents catabolism of CI+CIV-deficient cells"),
    ("Cardiac management", "MANDATORY", "Annual echo + ECG + Holter; beta-blockers for HCM-LVOT obstruction; ICD referral if LVEF <35% or VT on Holter; cardiomyopathy annual surveillance lifelong"),
    ("Ophthalmology", "MANDATORY", "Annual ocular motility exam + ptosis assessment + fundoscopy; prism correction for diplopia; ptosis surgery last resort; annual retinal exam (KSS overlap)"),
    ("Cochlear implant", "Effective for SNHL", "MT-TH-associated SNHL responds well to cochlear implant; refer early before profound deafness; Cl function preserved as spiral ganglion neurons retain some OXPHOS"),
]

rng = random.Random(SEED)

def _make_patients():
    patients = []
    variant_weights = [0.40, 0.25, 0.20, 0.10, 0.05]
    phenotype_weights = [0.30, 0.28, 0.20, 0.12, 0.10]
    for i in range(N_PATIENTS):
        var_idx = rng.choices(range(len(VARIANTS)), weights=variant_weights)[0]
        phen_idx = rng.choices(range(len(PHENOTYPES)), weights=phenotype_weights)[0]
        het_base = rng.randint(45, 95)
        age_onset = rng.randint(0, 55)
        has_leigh = phen_idx == 0
        has_cpeo = phen_idx in (1, 2, 3)
        has_cardio = phen_idx in (2, 3)
        has_kss = phen_idx == 3
        has_snhl = rng.random() < 0.62
        has_dm = rng.random() < 0.15
        has_neuropathy = rng.random() < 0.35
        patients.append({
            "id": f"MTTH-{i+1:03d}",
            "variant": VARIANTS[var_idx][0],
            "phenotype": PHENOTYPES[phen_idx],
            "heteroplasmy_blood": het_base,
            "age_onset": age_onset,
            "leigh_mri": has_leigh,
            "cpeo": has_cpeo,
            "cardiomyopathy": has_cardio,
            "kss": has_kss,
            "snhl": has_snhl,
            "diabetes": has_dm,
            "neuropathy": has_neuropathy,
            "ci_activity": round(rng.uniform(5, 35), 1),
            "civ_activity": round(rng.uniform(8, 40), 1),
            "cii_activity": round(rng.uniform(85, 115), 1),
            "lactate": round(rng.uniform(2.2, 12.8), 1),
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
    pct_leigh = round(sum(1 for p in pts if p["leigh_mri"]) / n * 100)
    pct_cpeo = round(sum(1 for p in pts if p["cpeo"]) / n * 100)
    pct_cardio = round(sum(1 for p in pts if p["cardiomyopathy"]) / n * 100)
    pct_snhl = round(sum(1 for p in pts if p["snhl"]) / n * 100)

    return {
        "gene": "MT-TH",
        "omim_gene": "590080",
        "full_name": "Mitochondrially Encoded tRNA-His (tRNA-Histidine)",
        "rCRS_position": "H-strand 12138–12206 (69 nt)",
        "flanking_genes": "MT-ND4 (10760–12137) → MT-TH → MT-TS2 (12207–12265) → MT-TL2 → MT-ND5",
        "inheritance": "MATERNAL — heteroplasmic",
        "primary_disease": "Combined CI+CIV Deficiency — Leigh-like Syndrome / CPEO / Cardiomyopathy",
        "key_mutation": "m.12147G>A (D-loop/variable-loop junction — ~40% of MT-TH cases)",
        "oxphos_fingerprint": "CI + CIV reduced; CII (SDH — nuclear-encoded) NORMAL — mt-translation fingerprint",
        "cohort_statistics": {
            "n_patients": n,
            "avg_heteroplasmy_blood_pct": avg_het,
            "avg_age_onset_yr": avg_onset,
            "avg_ci_activity_pct_normal": avg_ci,
            "avg_civ_activity_pct_normal": avg_civ,
            "avg_cii_activity_pct_normal": avg_cii,
            "pct_leigh_mri": pct_leigh,
            "pct_cpeo": pct_cpeo,
            "pct_cardiomyopathy": pct_cardio,
            "pct_snhl": pct_snhl,
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
            {"range": "<55% blood", "phenotype": "Asymptomatic / exercise intolerance", "management": "Annual review, avoid triggers, genetic counselling"},
            {"range": "55–70% blood", "phenotype": "CPEO + SNHL + mild myopathy", "management": "Ophthalmology, audiology, avoid CI drugs"},
            {"range": "70–85% blood", "phenotype": "Leigh-like + CPEO + cardiomyopathy + SNHL", "management": "Multidisciplinary: neurology + cardiology + ophthalmology"},
            {"range": ">85% blood", "phenotype": "Severe Leigh + multi-organ failure + lactic crisis", "management": "ICU: GIR 6–8, IV thiamine, bicarb, avoid fasting"},
        ],
        "key_molecular_features": [
            "MT-TH encodes tRNA-His (GUG anticodon) — 69 nt RNA gene essential for all 13 mt-encoded OXPHOS subunits",
            "Flanked by MT-ND4 (10760–12137) and MT-TS2 (12207–12265) — between largest CI gene and Ser tRNA",
            "CI + CIV combined deficiency (both mt-encoded) — CII NORMAL (nuclear) = mt-translation fingerprint",
            "m.12147G>A disrupts D-loop/variable-loop junction → impaired tRNA-His tertiary fold → aminoacylation failure",
            "Blood heteroplasmy underestimates by 10–20%; muscle biopsy preferred for threshold assessment",
            "WES/WGS misses MT-TH — dedicated mtDNA panel (NGS or long-read) mandatory",
            "Maternal inheritance — all children of carrier mother are carriers (variable heteroplasmy by bottleneck)",
            "CPEO more prominent than in MT-TK (MERRF) — ptosis + progressive ophthalmoplegia common at 55–70%",
            "NO stroke-like episodes (vs MT-TL1 MELAS) — KEY DDx; MRI Leigh-like bilateral BG/brainstem",
            "NO MSL (vs MT-TK MERRF) — no symmetric lipomatosis in MT-TH",
            "BTBGD (SLC19A3) MANDATORY exclusion — treatable Leigh mimic; thiamine + biotin empiric",
        ],
        "clinical_alerts": [
            {"alert": "METFORMIN ABSOLUTE CI", "detail": "Complex I inhibitor → fatal lactic acidosis; applies to ALL MT-TH carriers including subclinical heteroplasmy"},
            {"alert": "VPA ABSOLUTE CI", "detail": "mt-ribosome inhibitor + CoA sequestration + POLG; catastrophic in CI+CIV deficiency; use LEV instead"},
            {"alert": "PROPOFOL ABSOLUTE CI (PRIS)", "detail": "CIV inhibitor + propofol infusion syndrome; amplified lethality in pre-existing CI+CIV deficiency; use sevoflurane"},
            {"alert": "LINEZOLID ABSOLUTE CI", "detail": "mt-23S rRNA inhibition blocks all 13 mtDNA OXPHOS translations — compounding CI+CIV failure"},
            {"alert": "KD CONTRAINDICATED", "detail": "Ketogenic diet requires functional CI+CIV for beta-oxidation FADH2 processing; catastrophic ketoacidosis risk"},
            {"alert": "NEVER FAST", "detail": "Fasting triggers acute lactic crisis; GIR 6–8 mg/kg/min IV dextrose mandatory during illness/perioperative NPO"},
            {"alert": "ANNUAL CARDIAC SURVEILLANCE", "detail": "Echo + ECG + Holter mandatory — cardiomyopathy (HCM/DCM) in 20–32%; ICD if EF <35% or VT; digoxin CI in HCM-LVOT"},
            {"alert": "BTBGD EXCLUSION MANDATORY", "detail": "SLC19A3 Biotin-Thiamine-Responsive Basal Ganglia Disease — treatable Leigh mimic; test before starting CoQ10 escalation"},
        ],
        "cohort_summary_features": [
            f"{pct_leigh}% have Leigh-like MRI (bilateral symmetric BG/brainstem)",
            f"{pct_cpeo}% have CPEO (ptosis + ophthalmoplegia)",
            f"{pct_cardio}% have cardiomyopathy (HCM or DCM)",
            f"{pct_snhl}% have sensorineural hearing loss",
            f"Mean CI activity {avg_ci}% of normal (CIV {avg_civ}%; CII {avg_cii}% — NORMAL)",
            f"Mean blood heteroplasmy {avg_het}% (muscle ~10–20% higher)",
            f"Mean age at onset {avg_onset} yr (range neonatal–55 yr)",
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
        pct_leigh = round(sum(1 for x in grp if x["leigh_mri"]) / len(grp) * 100)
        pct_cpeo = round(sum(1 for x in grp if x["cpeo"]) / len(grp) * 100)
        pct_cardio = round(sum(1 for x in grp if x["cardiomyopathy"]) / len(grp) * 100)
        pct_snhl = round(sum(1 for x in grp if x["snhl"]) / len(grp) * 100)
        variant_summaries.append({
            "variant": v_name,
            "region": v_desc,
            "note": v_note,
            "n": len(grp),
            "avg_heteroplasmy_blood_pct": avg_het,
            "avg_ci_activity_pct": avg_ci,
            "avg_civ_activity_pct": avg_civ,
            "pct_leigh_mri": pct_leigh,
            "pct_cpeo": pct_cpeo,
            "pct_cardiomyopathy": pct_cardio,
            "pct_snhl": pct_snhl,
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
            "leigh_mri": p["leigh_mri"],
            "cpeo": p["cpeo"],
            "cardiomyopathy": p["cardiomyopathy"],
            "kss": p["kss"],
            "snhl": p["snhl"],
            "diabetes": p["diabetes"],
            "neuropathy": p["neuropathy"],
            "phenotype_short": PHENOTYPES.index(p["phenotype"]) + 1,
        }
        for p in pts
    ]

    trigger_rates = [
        {"trigger": t, "pct": rng.randint(25, 85)} for t in TRIGGERS
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
            "muscle_histochemistry": "RRF possible at high heteroplasmy; COX-negative fibres; SDH-positive ragged fibres",
        },
        "ddx_comparison": [
            {
                "gene": "MT-TK",
                "disease": "MERRF (Myoclonic Epilepsy + RRF)",
                "oxphos": "CI+CIV (like MT-TH)",
                "distinguisher": "Myoclonic epilepsy dominant; MSL (lipomatosis) in 10–20%; NO Leigh-like MRI at onset; PME archetype",
            },
            {
                "gene": "MT-TL1",
                "disease": "MELAS (Stroke-like Episodes)",
                "oxphos": "CI+CIII+CIV (pan-OXPHOS)",
                "distinguisher": "Stroke-like episodes cross vascular territories; MRI cortical/subcortical NOT bilateral BG; SSSVS muscle",
            },
            {
                "gene": "SURF1",
                "disease": "Leigh Syndrome (CIV deficiency)",
                "oxphos": "CIV isolated (CI/CII/CIII normal)",
                "distinguisher": "Isolated CIV (not combined CI+CIV); COX-negative; AR nuclear; WES detectable; NO maternal inheritance",
            },
            {
                "gene": "POLG",
                "disease": "Alpers-Huttenlocher / SANDO",
                "oxphos": "mtDNA depletion — multi-complex",
                "distinguisher": "Hepatopathy (Alpers); mtDNA depletion on Southern blot; AR nuclear; WES detectable; NO maternal tRNA",
            },
            {
                "gene": "SLC19A3 (BTBGD)",
                "disease": "Biotin-Thiamine-Responsive BGD",
                "oxphos": "Normal OXPHOS biochemistry",
                "distinguisher": "TREATABLE — thiamine + biotin → dramatic recovery; Leigh-like MRI but normal CI/CIV on ETC; MANDATORY first exclusion",
            },
        ],
    }


def get_definitions():
    return {
        "gene_biology": {
            "MT-TH": "Mitochondrially encoded tRNA-His (GUG anticodon); 69 nt RNA gene; H-strand rCRS 12138–12206; required for mt-ribosomal synthesis of all 13 mtDNA OXPHOS subunits (ND1-6, CO1-3, ATP6/8, CYB); located between MT-ND4 and MT-TS2",
            "tRNA-His_function": "Delivers histidine to the mt-ribosome A-site during translation of ALL 13 mtDNA-encoded subunits; anticodon GUG recognises CAU/CAC codons; mt-aminoacyl-tRNA synthetase: HARS2 (mitochondrial histidyl-tRNA synthetase); HARS2 mutations cause Perrault syndrome (phenocopy consideration)",
            "mt_translation_fingerprint": "CI + CIV reduced; CII (SDH, nuclear-encoded) NORMAL; BN-PAGE: CI and CIV bands absent/reduced; CII band present; distinguishes mt-tRNA mutations from isolated nuclear gene OXPHOS defects",
            "rCRS_position": "H-strand rCRS 12138–12206 (69 bp); flanked by MT-ND4 (10760–12137) and MT-TS2 (12207–12265); between the largest CI subunit gene and the Ser(AGY) tRNA; large deletions common-deletion region (4977 bp) does NOT span MT-TH — point mutations and individual deletions only",
        },
        "clinical_terms": {
            "Leigh_syndrome": "Bilateral symmetric signal abnormality in basal ganglia (putamen/caudate) and brainstem on MRI T2/FLAIR; neuropathological: necrotic foci with spongiosis, vascular proliferation, astrogliosis; clinical: psychomotor regression, hypotonia, lactic acidosis, respiratory compromise",
            "CPEO": "Chronic Progressive External Ophthalmoplegia: slowly progressive bilateral ptosis + ophthalmoparesis; EOMs eventually all affected; CPEO ± myopathy (CPEO+) in MT-TH; KSS if large deletion (CPEO + retinal pigmentation + cardiac conduction block <20 yr)",
            "mt_translation_fingerprint": "CI + CIV reduced; CII NORMAL — reflects that 7 CI subunits (ND1-6 + ND4L) and 3 CIV subunits (CO1-3) are mtDNA-encoded and require tRNA-His; CII (4 subunits: SDHA-D) all nuclear-encoded → NORMAL despite pan-tRNA defect",
            "heteroplasmy_threshold": "Minimum mutant mtDNA load for disease expression; MT-TH m.12147G>A: ~55% blood for CPEO; ~70% for Leigh-like; >85% for severe multi-organ failure; blood underestimates muscle heteroplasmy by 10–20%",
            "BTBGD": "Biotin-Thiamine-Responsive Basal Ganglia Disease (SLC19A3 — thiamine transporter 2); Leigh-like MRI; TREATABLE with thiamine + biotin; MANDATORY exclusion before attributing Leigh-like MRI to MT-TH",
            "KSS": "Kearns-Sayre Syndrome: triad of CPEO + retinal pigmentary degeneration + cardiac conduction block onset <20 yr; caused by large mtDNA deletions (not MT-TH point mutations); annual Holter mandatory (CHB risk)",
        },
        "pharmacology": {
            "absolute_ci": {a[0]: a[2] for a in ABSOLUTE_CI},
            "preferred_aed": "Levetiracetam (LEV) — renal excretion; no mt-toxicity; first-line for seizures in MT-TH disease",
            "emergency_protocol": "GIR 6–8 mg/kg/min IV dextrose + IV thiamine 10–20 mg/kg + avoid fasting + bicarb for pH <7.2 + ICU if Leigh crisis",
            "cardiac_monitoring": "Annual echo + ECG + Holter; beta-blockers for symptomatic HCM; ICD/pacemaker if Holter shows CHB or VT; digoxin AVOID in HCM-LVOT obstruction",
        },
        "key_references": [
            "Santorelli FM et al. (1994) Maternally inherited cardiomyopathy: an atypical presentation of the mtDNA 12S rRNA gene A1555G mutation — Neurology (early MT-TH region study)",
            "Pulkes T et al. (2000) New phenotypic diversity associated with the m.11778G>A (ND4) LHON mutation — Brain (ND4 region context for MT-TH flanking)",
            "Schaefer AM et al. (2008) Prevalence of mitochondrial DNA disease in adults — Ann Neurol (prevalence context)",
            "Gorman GS et al. (2015) Mitochondrial diseases — Nat Rev Dis Primers (comprehensive review including tRNA gene spectrum)",
            "DiMauro S & Schon EA (2003) Mitochondrial respiratory-chain diseases — NEJM (tRNA mutation review including MT-TH)",
            "Zeviani M & Carelli V (2021) Mitochondrial retinopathies — Int J Mol Sci (OXPHOS disease spectrum)",
        ],
    }
