#!/usr/bin/env python3
"""MT-TL2 — Mitochondrially Encoded tRNA-Leu(CUN) — CPEO / Myopathy / Optic Atrophy
Combined CI+CIV Deficiency (mt-translation fingerprint)

MT-TL2 (OMIM *590055) encodes mitochondrial tRNA for leucine (CUN codons: CUA/CUC/CUG/CUU),
anticodon UAA, rCRS H-strand positions 12266–12336 (71 bp). MT-TL2 is one of two mt-tRNA-Leu
genes (the other being MT-TL1 at rCRS 3230–3304 encoding tRNA-Leu(UUR), the MELAS gene).
Like all mt-tRNA genes, MT-TL2 is essential for translating ALL 13 mtDNA-encoded OXPHOS
subunits — mutations impair mt-translation, causing combined CI + CIV deficiency
(CII NORMAL — the mt-translation fingerprint).

m.12311T>C (anticodon stem, position 32 of tRNA-Leu CUN) is the most commonly reported
pathogenic MT-TL2 mutation; it disrupts tRNA folding and causes CPEO + myopathy.
m.12308A>G (anticodon loop, position 33) is notable for its association with
CPEO + optic atrophy — a distinguishing feature from MT-TS2 and MT-TH.

  MT-TL2 gene             OMIM *590055
  Primary disease         Combined CI+CIV Deficiency — CPEO / Myopathy / Optic Atrophy
                          Exercise Intolerance / Lactic Acidosis
                          Multisystem Encephalomyopathy (high heteroplasmy)
  Protein product         tRNA-Leu(CUN) (UAA anticodon) — 71 nucleotides; RNA gene
                          CUN codons: CUA, CUC, CUG, CUU (Leu)
  Genome                  Mitochondrial DNA (mtDNA), H-strand, rCRS 12266–12336
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    Between MT-TS2 (12207–12265) and MT-ND5 (12337–14148)
  Key mutation            m.12311T>C — anticodon stem disruption →
                            impaired tRNA-Leu(CUN) fold → pan-OXPHOS (CI+CIV)

TWO LEUCINE tRNA GENES — KEY DISTINCTION:
  MT-TL1 (rCRS 3230–3304, UUR codons, UAA anticodon) → MELAS, MIDD — HIGH-SEVERITY
  MT-TL2 (rCRS 12266–12336, CUN codons, UAA anticodon) → CPEO, Myopathy, Optic Atrophy
  SAME anticodon (UAA) but DIFFERENT wobble position; different mutational spectrum;
  COMPLETELY DIFFERENT CLINICAL PHENOTYPE — MT-TL2 does NOT cause stroke-like episodes.

HETEROPLASMY THRESHOLD (m.12311T>C — blood underestimates by ~10-15%):
  <50% blood:             Asymptomatic carrier / isolated exercise intolerance
  50-65% blood:           CPEO + exercise intolerance + myopathy (partial phenotype)
  65-80% blood:           Full CPEO + optic atrophy + myopathy + lactic acidosis
  >80% blood:             Multisystem encephalomyopathy + multi-organ + lactic crisis

CPEO + OPTIC ATROPHY — MT-TL2 UNIQUE FEATURE:
  m.12308A>G is the only common mt-tRNA mutation causing CPEO PLUS optic atrophy
  as co-primary features. This dual optic + oculomotor phenotype arises because
  MT-TL2 CUN codons are used heavily by ND4 (the LHON subunit) — high CUN-codon
  density in ND4 mRNA makes RGCs/retinal neurons disproportionately vulnerable
  when tRNA-Leu(CUN) aminoacylation is impaired.

PAN-OXPHOS (CI + CIV) — KEY DISTINGUISHER:
  MT-TL2 mutations impair mt-ribosome aminoacylation of Leu(CUN) → defective translation
  of ALL 13 mtDNA-encoded OXPHOS subunits → CI + CIV reduced; CII (SDH — nuclear) NORMAL.
  BN-PAGE: CI + CIV reduced pattern; CII band normal — mt-translation fingerprint.

MT-TL2 TRIAD — CPEO + MYOPATHY + OPTIC ATROPHY:
  1. CPEO (chronic progressive external ophthalmoplegia — ptosis + ophthalmoplegia)
  2. Myopathy (limb + oculopharyngeal; RRF in muscle biopsy)
  3. Optic atrophy (DISTINCTIVE — less common in MT-TH and MT-TS2; prominent in m.12308A>G)
  Plus: exercise intolerance, lactic acidosis, SNHL, variable encephalomyopathy.
  NO myoclonic epilepsy as cardinal feature (distinguishes from MT-TK MERRF).
  NO stroke-like episodes (distinguishes from MT-TL1 MELAS — KEY DDx).
  NO MSL/Madelung lipomatosis (distinguishes from MT-TK MERRF).
  NO primary cardiomyopathy as the main feature (vs MT-TH).
"""

import random
from collections import Counter

SEED = 795
N_PATIENTS = 40

VARIANTS = [
    ("m.12311T>C", "Anticodon stem (position 32)", "~35% of MT-TL2 cases; anticodon stem loop disruption impairs tRNA-Leu(CUN) tertiary fold; combined CI+CIV; CPEO + myopathy; adult onset; Bindoff 1993 landmark report"),
    ("m.12315G>A", "Anticodon loop (position 36)", "~25%; anticodon loop disruption; CPEO + exercise intolerance; moderate CI+CIV reduction; adult presentation; variable SNHL"),
    ("m.12294G>A", "D-stem (position 15)", "~20%; D-stem disruption; multisystem encephalomyopathy + optic atrophy + CPEO; higher heteroplasmy needed for severe phenotype; CI+CIV fingerprint"),
    ("m.12308A>G", "Anticodon loop (position 33)", "~12%; DISTINCTIVE — CPEO + optic atrophy co-primary; RGC vulnerability from high CUN-codon density in ND4; may mimic LHON+CPEO overlap; Holt 1990"),
    ("Large deletion", "Multi-gene spanning", "~8%; deletion spanning MT-TL2 ± MT-TS2/MT-ND5 region → KSS/CPEO/Pearson; multi-complex OXPHOS; annual Holter mandatory (cardiac conduction block risk)"),
]

PHENOTYPES = [
    "CPEO + myopathy + optic atrophy — intermediate heteroplasmy",
    "Exercise intolerance + lactic acidosis — low-moderate heteroplasmy",
    "Multisystem encephalomyopathy + CPEO + optic atrophy — high heteroplasmy",
    "KSS overlap (large deletion: CPEO + retinal pigment + cardiac block)",
    "Asymptomatic / isolated exercise intolerance — low heteroplasmy",
]

TRIGGERS = [
    "Intercurrent illness (fever/infection)",
    "Fasting / prolonged NPO",
    "Valproate / contraindicated drug exposure",
    "Surgery / general anaesthesia (propofol)",
    "Extreme physical exertion",
    "Alcohol",
    "Aminoglycoside exposure (SNHL precipitant)",
]

ABSOLUTE_CI = [
    ("Metformin", "ABSOLUTE CI", "Complex I inhibitor → fatal lactic acidosis in CI-deficient states; applies to ALL MT-TL2 carriers regardless of phenotype severity or heteroplasmy level"),
    ("Valproate (VPA)", "ABSOLUTE CI", "mt-ribosome inhibitor + CoA sequestration + POLG inhibition; impairs tRNA-Leu(CUN) aminoacylation recovery; hepatotoxic in CI+CIV deficiency"),
    ("Linezolid", "ABSOLUTE CI", "mt-23S rRNA inhibitor → blocks synthesis of all 13 mtDNA OXPHOS subunits; compounding CI+CIV failure; fatal in mitochondrial myopathy"),
    ("Chloramphenicol", "ABSOLUTE CI", "mt-ribosome inhibitor; same mechanism as linezolid for all mtDNA-encoded OXPHOS subunits; fatal myelosuppression risk in mitochondrial disease"),
    ("Propofol", "ABSOLUTE CI (PRIS)", "Propofol infusion syndrome — inhibits OXPHOS at CIV; amplified lethality with pre-existing CI+CIV deficiency; use sevoflurane instead"),
    ("KD (Ketogenic Diet)", "CONTRAINDICATED", "CI+CIV failure → FADH2 backup blocks beta-oxidation; ketogenic metabolism requires functional OXPHOS; catastrophic ketoacidosis risk"),
    ("Ethambutol", "CAUTION — OPTIC ATROPHY", "Optic nerve toxin; synergistic optic toxicity in MT-TL2 patients with m.12308A>G (pre-existing RGC/optic nerve vulnerability); avoid if possible"),
    ("Amiodarone", "CAUTION", "Mitochondrial membrane toxin; may worsen lactic acidosis in CI+CIV deficiency; alternative rate control preferred"),
]

TREATMENTS = [
    ("Thiamine B1", "Level C", "MANDATORY empiric — cofactor for PDH/KGDH; lactic acidosis crisis: 10-20 mg/kg IV; oral maintenance 100-300 mg/day; SLC19A3/BTD exclusion first"),
    ("Biotin", "Level C", "MANDATORY empiric — BTBGD/BTD exclusion (treatable Leigh-like mimic); oral 10-20 mg/day; low cost/risk, high benefit if BTBGD missed"),
    ("CoQ10 Ubiquinol", "Level C", "300-1200 mg/day divided doses; electron carrier CI→CII→CIII; ubiquinol preferred over ubiquinone in CI+CIV deficiency; titrate over weeks"),
    ("Riboflavin B2", "Level C", "100-400 mg/day; FAD/FMN cofactor for CI N-module and CIII; may marginally improve residual CI+CIV activity; trial 3-6 months minimum"),
    ("L-Carnitine", "Level C", "1-3 g/day; mitochondrial beta-oxidation support; carnitine depletion common in CI+CIV deficiency; avoid masking VPA toxicity symptoms"),
    ("Levetiracetam (LEV)", "Preferred AED", "Renal excretion; no mitochondrial toxicity; first-line for seizures in MT-TL2 disease; avoid VPA (ABSOLUTE CI); avoid CBZ/OXC (worsens fatigue)"),
    ("GIR 6-8 mg/kg/min IV dextrose", "Acute crisis", "NEVER fast — fasting triggers lactic crisis; IV glucose infusion during illness/perioperative NPO; prevents catabolism of CI+CIV-deficient cells"),
    ("Ophthalmology", "MANDATORY", "Annual ocular motility exam + ptosis + fundoscopy + OCT (for optic atrophy); prism for diplopia; ptosis surgery last resort; m.12308A>G: neuro-ophthalmology mandatory"),
    ("Idebenone (Raxone)", "Level C — optic atrophy", "Consider for MT-TL2 m.12308A>G optic atrophy patients (analogy to LHON); 900 mg/day; evidence weaker than LHON; discuss risk/benefit"),
    ("Audiology", "Annual", "Audiological monitoring every 6-12 months; cochlear implant effective for SNHL; avoid aminoglycosides (cochlear OXPHOS amplification risk)"),
]

rng = random.Random(SEED)

def _make_patients():
    patients = []
    variant_weights = [0.35, 0.25, 0.20, 0.12, 0.08]
    phenotype_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
    for i in range(N_PATIENTS):
        var_idx = rng.choices(range(len(VARIANTS)), weights=variant_weights)[0]
        phen_idx = rng.choices(range(len(PHENOTYPES)), weights=phenotype_weights)[0]
        het_base = rng.randint(42, 92)
        age_onset = rng.randint(0, 60)
        has_cpeo = phen_idx in (0, 2, 3)
        has_myopathy = phen_idx in (0, 1, 2)
        has_encephalo = phen_idx == 2
        has_kss = phen_idx == 3
        has_optic_atrophy = rng.random() < (0.58 if VARIANTS[var_idx][0] == "m.12308A>G" else 0.30)
        has_snhl = rng.random() < 0.52
        has_cardio = rng.random() < 0.12
        has_dm = rng.random() < 0.12
        has_neuropathy = rng.random() < 0.25
        has_rrm = rng.random() < 0.48
        patients.append({
            "id": f"MTTL2-{i+1:03d}",
            "variant": VARIANTS[var_idx][0],
            "phenotype": PHENOTYPES[phen_idx],
            "heteroplasmy_blood": het_base,
            "age_onset": age_onset,
            "cpeo": has_cpeo,
            "myopathy": has_myopathy,
            "encephalomyopathy": has_encephalo,
            "kss": has_kss,
            "optic_atrophy": has_optic_atrophy,
            "snhl": has_snhl,
            "cardiomyopathy": has_cardio,
            "diabetes": has_dm,
            "neuropathy": has_neuropathy,
            "ragged_red_fibres": has_rrm,
            "ci_activity": round(rng.uniform(10, 40), 1),
            "civ_activity": round(rng.uniform(12, 45), 1),
            "cii_activity": round(rng.uniform(85, 115), 1),
            "lactate": round(rng.uniform(1.8, 11.0), 1),
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
    pct_optic = round(sum(1 for p in pts if p["optic_atrophy"]) / n * 100)
    pct_snhl = round(sum(1 for p in pts if p["snhl"]) / n * 100)
    pct_cardio = round(sum(1 for p in pts if p["cardiomyopathy"]) / n * 100)
    pct_rrm = round(sum(1 for p in pts if p["ragged_red_fibres"]) / n * 100)

    return {
        "gene": "MT-TL2",
        "omim_gene": "590055",
        "full_name": "Mitochondrially Encoded tRNA-Leu(CUN) — tRNA-Leucine (CUN codons: CUA/CUC/CUG/CUU)",
        "rCRS_position": "H-strand 12266–12336 (71 nt)",
        "flanking_genes": "MT-TS2 (12207–12265) → MT-TL2 → MT-ND5 (12337–14148)",
        "inheritance": "MATERNAL — heteroplasmic",
        "primary_disease": "Combined CI+CIV Deficiency — CPEO / Myopathy / Optic Atrophy",
        "key_mutation": "m.12311T>C (anticodon stem position 32 — ~35% of MT-TL2 cases)",
        "distinctive_mutation": "m.12308A>G (anticodon loop position 33 — CPEO + Optic Atrophy co-primary; ~12%)",
        "oxphos_fingerprint": "CI + CIV reduced; CII (SDH — nuclear-encoded) NORMAL — mt-translation fingerprint",
        "unique_feature": "One of TWO mt-tRNA-Leu genes (MT-TL1 = MELAS/UUR; MT-TL2 = CPEO/Optic Atrophy/CUN); m.12308A>G causes CPEO + optic atrophy — rare dual optic+oculomotor phenotype",
        "anticodon": "UAA (same as MT-TL1 but different wobble; decodes CUN codons vs UUR codons)",
        "cohort_statistics": {
            "n_patients": n,
            "avg_heteroplasmy_blood_pct": avg_het,
            "avg_age_onset_yr": avg_onset,
            "avg_ci_activity_pct_normal": avg_ci,
            "avg_civ_activity_pct_normal": avg_civ,
            "avg_cii_activity_pct_normal": avg_cii,
            "pct_cpeo": pct_cpeo,
            "pct_myopathy": pct_myopathy,
            "pct_optic_atrophy": pct_optic,
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
            {"range": "<50% blood", "phenotype": "Asymptomatic / isolated exercise intolerance", "management": "Annual ophthalmic + neurological review; genetic counselling; avoid CI drugs"},
            {"range": "50–65% blood", "phenotype": "CPEO + exercise intolerance + myopathy", "management": "Ophthalmology, CoQ10 ubiquinol, thiamine empiric, avoid CI drugs"},
            {"range": "65–80% blood", "phenotype": "Full CPEO + optic atrophy + myopathy + lactic acidosis", "management": "Multidisciplinary: neurology + neuro-ophthalmology + audiology; idebenone consider for optic atrophy"},
            {"range": ">80% blood", "phenotype": "Multisystem encephalomyopathy + multi-organ + crisis", "management": "ICU: GIR 6–8, IV thiamine, bicarb if pH <7.2, avoid fasting/propofol/VPA"},
        ],
        "key_molecular_features": [
            "MT-TL2 encodes tRNA-Leu(CUN) (UAA anticodon) — 71 nt RNA gene; rCRS H-strand 12266–12336",
            "Flanked by MT-TS2 (12207–12265) and MT-ND5 (12337–14148); directly 3' of the MT-TH/MT-TS2/MT-TL2 tRNA cluster",
            "CI + CIV combined deficiency (both mt-encoded); CII NORMAL (nuclear) = mt-translation fingerprint",
            "m.12311T>C disrupts anticodon stem loop → impairs tRNA-Leu(CUN) tertiary folding → aminoacylation failure",
            "m.12308A>G causes CPEO + optic atrophy — RGC vulnerability from high CUN-codon density in ND4 mRNA",
            "TWO mt-Leu-tRNA genes: MT-TL1 (UUR/MELAS) vs MT-TL2 (CUN/CPEO-Optic atrophy) — COMPLETELY DIFFERENT PHENOTYPES",
            "Blood heteroplasmy underestimates by 10–15%; muscle biopsy preferred for threshold assessment",
            "WES/WGS misses MT-TL2 — dedicated mtDNA panel (NGS or long-read) mandatory",
            "NO stroke-like episodes (KEY DDx from MT-TL1 MELAS — same UAA anticodon, entirely different clinical disease)",
            "NO myoclonic epilepsy as cardinal (distinguishes from MT-TK MERRF)",
            "NO MSL/Madelung lipomatosis (distinguishes from MT-TK MERRF)",
            "BTBGD (SLC19A3) MANDATORY exclusion — treatable Leigh-like mimic; thiamine + biotin empiric",
        ],
        "clinical_alerts": [
            {"alert": "METFORMIN ABSOLUTE CI", "detail": "Complex I inhibitor → fatal lactic acidosis; applies to ALL MT-TL2 carriers including subclinical/exercise-only"},
            {"alert": "VPA ABSOLUTE CI", "detail": "mt-ribosome inhibitor + CoA sequestration + POLG; catastrophic in CI+CIV deficiency; use LEV instead"},
            {"alert": "PROPOFOL ABSOLUTE CI (PRIS)", "detail": "CIV inhibitor + propofol infusion syndrome; amplified lethality in pre-existing CI+CIV deficiency; use sevoflurane"},
            {"alert": "LINEZOLID ABSOLUTE CI", "detail": "mt-23S rRNA inhibition blocks all 13 mtDNA OXPHOS translations — compounding CI+CIV failure"},
            {"alert": "ETHAMBUTOL CAUTION (OPTIC ATROPHY)", "detail": "Optic nerve toxin; synergistic risk in MT-TL2 m.12308A>G patients with pre-existing RGC vulnerability; avoid if possible"},
            {"alert": "NEVER FAST", "detail": "Fasting triggers acute lactic crisis; GIR 6–8 mg/kg/min IV dextrose mandatory during illness/perioperative NPO"},
            {"alert": "KD CONTRAINDICATED", "detail": "CI+CIV failure blocks beta-oxidation FADH2 processing; ketogenic diet causes catastrophic ketoacidosis"},
            {"alert": "BTBGD EXCLUSION MANDATORY", "detail": "SLC19A3 Biotin-Thiamine-Responsive Basal Ganglia Disease — treatable Leigh-like mimic; MANDATORY first exclusion"},
        ],
        "cohort_summary_features": [
            f"{pct_cpeo}% have CPEO (ptosis + ophthalmoplegia)",
            f"{pct_myopathy}% have myopathy (limb + oculopharyngeal)",
            f"{pct_optic}% have optic atrophy (DISTINCTIVE — higher than MT-TS2/MT-TH; hallmark of m.12308A>G)",
            f"{pct_snhl}% have sensorineural hearing loss",
            f"{pct_cardio}% have cardiomyopathy (less prominent than MT-TH)",
            f"{pct_rrm}% have ragged-red fibres on muscle biopsy",
            f"Mean CI activity {avg_ci}% of normal (CIV {avg_civ}%; CII {avg_cii}% — NORMAL)",
            f"Mean blood heteroplasmy {avg_het}% (muscle ~10–15% higher)",
            f"Mean age at onset {avg_onset} yr (range neonatal–60 yr)",
            "MT-TL2 = second mt-Leu-tRNA (CUN codons); SAME UAA anticodon as MT-TL1 but COMPLETELY DIFFERENT DISEASE (no MELAS/stroke-like)",
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
        pct_optic = round(sum(1 for x in grp if x["optic_atrophy"]) / len(grp) * 100)
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
            "pct_optic_atrophy": pct_optic,
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
            "optic_atrophy": p["optic_atrophy"],
            "snhl": p["snhl"],
            "cardiomyopathy": p["cardiomyopathy"],
            "ragged_red_fibres": p["ragged_red_fibres"],
            "neuropathy": p["neuropathy"],
            "phenotype_short": PHENOTYPES.index(p["phenotype"]) + 1,
        }
        for p in pts
    ]

    trigger_rates = [
        {"trigger": t, "pct": rng.randint(20, 80)} for t in TRIGGERS
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
                "gene": "MT-TL1",
                "disease": "MELAS (Stroke-like Episodes + MIDD)",
                "oxphos": "CI+CIII+CIV (pan-OXPHOS)",
                "distinguisher": "MOST CRITICAL DDx — SAME tRNA-Leu gene family (UAA anticodon); COMPLETELY DIFFERENT DISEASE: stroke-like episodes crossing vascular territories (ABSENT in MT-TL2); MRI cortical/subcortical; MIDD at low heteroplasmy; SSSVS; m.3243A>G most common; Goto 1990",
            },
            {
                "gene": "MT-TS2",
                "disease": "Combined CI+CIV — CPEO / Myopathy / SNHL (SHORTEST mt-tRNA)",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "Immediately flanks MT-TL2 (12207–12265 vs 12266–12336); both cause CPEO+myopathy; MT-TS2 more isolated SNHL at low heteroplasmy; MT-TL2 more optic atrophy (m.12308A>G); MT-TS2 = 59 nt (shortest); MT-TL2 = 71 nt",
            },
            {
                "gene": "MT-TH",
                "disease": "Combined CI+CIV — Leigh-like + CPEO + Cardiomyopathy",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "More prominent cardiomyopathy (HCM/DCM); Leigh-like MRI more frequent; less optic atrophy than MT-TL2 m.12308A>G; cardiomyopathy annual echo mandatory; immediately upstream of MT-TS2→MT-TL2",
            },
            {
                "gene": "MT-TK",
                "disease": "MERRF (Myoclonic Epilepsy + RRF + MSL)",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "Myoclonic epilepsy cardinal (90%); MSL/lipomatosis (10-25%) — ABSENT in MT-TL2; progressive myoclonic epilepsy archetype; no optic atrophy; Shoffner 1990",
            },
            {
                "gene": "SLC19A3 (BTBGD)",
                "disease": "Biotin-Thiamine-Responsive BGD",
                "oxphos": "Normal OXPHOS biochemistry",
                "distinguisher": "TREATABLE — thiamine + biotin → dramatic MRI recovery; Leigh-like signal on MRI but normal CI/CIV biochemistry; MANDATORY first exclusion before MT-TL2 confirmed",
            },
            {
                "gene": "OPA1 / DOA",
                "disease": "Dominant Optic Atrophy (isolated)",
                "oxphos": "Variable — CII normal; CIV may be low in DOA-plus",
                "distinguisher": "OPA1 optic atrophy is insidious, bilateral, blue-yellow dyschromatopsia; AD nuclear; WES detectable; NO CPEO, NO myopathy, NO lactic acidosis vs MT-TL2 m.12308A>G where optic atrophy accompanies CPEO+myopathy",
            },
        ],
    }


def get_definitions():
    return {
        "gene_biology": {
            "MT-TL2": "Mitochondrially encoded tRNA-Leu(CUN) (UAA anticodon); 71 nt RNA gene; H-strand rCRS 12266–12336; required for mt-ribosomal synthesis of all 13 mtDNA OXPHOS subunits; located immediately 3' of MT-TS2 and 5' of MT-ND5",
            "tRNA_Leu_CUN_function": "Delivers leucine to the mt-ribosome A-site during translation of ALL 13 mtDNA-encoded subunits for CUN codons (CUA/CUC/CUG/CUU); anticodon UAA; mt-aminoacyl-tRNA synthetase: LARS2 (mitochondrial leucyl-tRNA synthetase); LARS2 mutations cause HLASA syndrome (phenocopy consideration)",
            "two_mt_Leu_tRNA": "Human mtDNA encodes two leucine tRNAs: MT-TL1 (rCRS 3230–3304, UUR codons, anticodon UAA → decodes UUA/UUG) for MELAS/MIDD phenotypes, and MT-TL2 (rCRS 12266–12336, CUN codons, anticodon UAA → decodes CUA/CUC/CUG/CUU) for CPEO/Optic Atrophy. Same anticodon, entirely different codons and diseases.",
            "mt_translation_fingerprint": "CI + CIV reduced; CII (SDH, nuclear-encoded) NORMAL; BN-PAGE: CI and CIV bands absent/reduced; CII band present; distinguishes mt-tRNA mutations from isolated nuclear gene OXPHOS defects",
            "rCRS_position": "H-strand rCRS 12266–12336 (71 bp); flanked by MT-TS2 (12207–12265) and MT-ND5 (12337–14148); part of the tRNA cluster (MT-TH/MT-TS2/MT-TL2) spanning rCRS 12138–12336",
            "optic_atrophy_mechanism": "m.12308A>G disrupts anticodon loop position 33 — CUN codons are used at high frequency in MT-ND4 mRNA (LHON subunit). Impaired tRNA-Leu(CUN) aminoacylation disproportionately reduces ND4 synthesis → retinal ganglion cell CI deficiency → optic atrophy. This makes MT-TL2 m.12308A>G mechanistically related to LHON despite being a tRNA mutation, not a structural gene mutation.",
        },
        "clinical_terms": {
            "CPEO": "Chronic Progressive External Ophthalmoplegia: slowly progressive bilateral ptosis + ophthalmoparesis; EOMs eventually all affected; prominent MT-TL2 feature; KSS if large deletion (CPEO + retinal pigmentation + cardiac conduction block <20 yr)",
            "Optic_Atrophy": "Optic nerve degeneration → painless progressive visual loss + central scotoma + colour vision impairment; DISTINCTIVE for MT-TL2 m.12308A>G; unlike LHON (acute/subacute), MT-TL2 optic atrophy is insidious alongside CPEO; OCT and VEP mandatory annually",
            "mt_translation_fingerprint": "CI + CIV reduced; CII NORMAL — reflects that all 7 CI ND-subunits (ND1-ND6, ND4L) and all 3 CIV CO-subunits (CO1-CO3) are mtDNA-encoded and require tRNA-Leu(CUN); CII (SDHA-D) all nuclear-encoded → NORMAL",
            "heteroplasmy_threshold": "Minimum mutant mtDNA load for disease expression; MT-TL2 m.12311T>C: ~50% blood for exercise intolerance; ~65% for CPEO+myopathy; >80% for multisystem encephalomyopathy; blood underestimates muscle by 10–15%",
            "BTBGD": "Biotin-Thiamine-Responsive Basal Ganglia Disease (SLC19A3); Leigh-like MRI; TREATABLE with thiamine + biotin; MANDATORY first exclusion before attributing multisystem features to MT-TL2",
            "KSS": "Kearns-Sayre Syndrome: CPEO + retinal pigmentary degeneration + cardiac conduction block onset <20 yr; caused by large mtDNA deletions spanning MT-TL2 ± flanking genes; annual Holter mandatory (complete heart block risk)",
            "HLASA": "HLASA (Hyperuricemia, Lactic Acidosis, Sideroblastic Anemia) — caused by LARS2 mutations (mt-leucyl-tRNA synthetase, CUN-arm); AR nuclear; WES detectable; DIFFERENT from maternal MT-TL2 mt-tRNA mutations; PIEZO2 association in some pedigrees",
        },
        "pharmacology": {
            "absolute_ci": {a[0]: a[2] for a in ABSOLUTE_CI},
            "preferred_aed": "Levetiracetam (LEV) — renal excretion; no mt-toxicity; first-line for seizures in MT-TL2 disease; avoid VPA (ABSOLUTE CI), CBZ/OXC (worsens fatigue in myopathy)",
            "emergency_protocol": "GIR 6–8 mg/kg/min IV dextrose + IV thiamine 10–20 mg/kg + avoid fasting + bicarb for pH <7.2 + ICU if multisystem crisis",
            "optic_atrophy_note": "For m.12308A>G patients: idebenone 900 mg/day (Level C; analogous to LHON); avoid ethambutol/amiodarone/chloramphenicol; annual OCT + VEP; early referral neuro-ophthalmology; visual rehabilitation planning",
        },
        "key_references": [
            "Bindoff LA et al. (1993) Mitochondrial function in chronic progressive external ophthalmoplegia — Brain (landmark MT-TL2 myopathy study including m.12311T>C)",
            "Holt IJ et al. (1990) A new mitochondrial disease associated with mitochondrial DNA heteroplasmy — Am J Hum Genet (early mt-tRNA CPEO reports including MT-TL2 region)",
            "Moraes CT et al. (1993) Mitochondrial DNA deletions in progressive external ophthalmoplegia — Neurology (large deletions spanning MT-TL2 → KSS/CPEO)",
            "Gorman GS et al. (2015) Mitochondrial diseases — Nat Rev Dis Primers (comprehensive review including mt-tRNA spectrum)",
            "DiMauro S & Schon EA (2003) Mitochondrial respiratory-chain diseases — NEJM (tRNA mutation review including MT-TL2 context)",
            "Zeviani M & Carelli V (2021) Mitochondrial retinopathies — Int J Mol Sci (OXPHOS disease spectrum, mt-tRNA and optic nerve phenotypes)",
            "Sasarman F et al. (2008) Tissue-specific cofactor dependencies and heterologous complementation of mt-tRNA synthetase defects — Hum Mol Genet (LARS2/HARS2 mechanistic framework applicable to MT-TL2 phenocopy DDx)",
        ],
    }
