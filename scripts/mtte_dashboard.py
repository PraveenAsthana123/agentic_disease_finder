#!/usr/bin/env python3
"""MT-TE — Mitochondrially Encoded tRNA-Glu — CPEO / Myopathy / Maternally-Inherited Diabetes
Combined CI+CIV Deficiency (mt-translation fingerprint) | Reversible COX Deficiency (m.14674T>C)

MT-TE (OMIM *590025) encodes mitochondrial tRNA-Glu (CUC anticodon, reading Glu codons GAA/GAG),
rCRS L-strand positions 14674–14742 (69 nt). MT-TE is located between MT-ND6 (L-strand,
ends 14673) and MT-CYB (H-strand, starts 14747), in the densely packed ND6/TE/CYB interval
of the human mitochondrial genome. Being L-strand encoded, MT-TE is susceptible to NGS
coverage dropout if only the H-strand is captured — a sequencing-protocol pitfall.

m.14709T>C is the most commonly reported pathogenic MT-TE mutation (~35–40%), causing
CPEO + myopathy + lactic acidosis + maternally inherited diabetes mellitus (MIDM) —
this maternal DM phenotype is one of the most clinically distinctive features of MT-TE disease,
rivalling MT-TL1 m.3243A>G (MIDD) for DM prominence among mt-tRNA genes.

m.14674T>C (or m.14674T>G): causes REVERSIBLE COX deficiency in infancy — the only
pathogenic mt-tRNA mutation with documented spontaneous clinical improvement and biochemical
normalisation. Neonatal respiratory failure + lactic acidosis can resolve by school age.

  MT-TE gene              OMIM *590025
  Primary diseases        Combined CI+CIV Deficiency — CPEO / Myopathy / MIDM
                          Reversible Infantile COX Deficiency (m.14674T>C unique)
                          Lactic Acidosis / Cardiomyopathy / Multisystem
  Protein product         tRNA-Glu (CUC anticodon) — 69 nucleotides; RNA gene
                          Glu codons: GAA, GAG
  Genome                  Mitochondrial DNA (mtDNA), L-strand, rCRS 14674–14742
  Inheritance             MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Chromosomal location    Between MT-ND6 (14149–14673, L-strand) and MT-CYB (14747–15887)

L-STRAND ENCODING — SEQUENCING PITFALL:
  MT-TE is encoded on the LIGHT (L) strand, like MT-ND6 and MT-TQ, MT-TP.
  Capture-based mtDNA NGS panels targeting only H-strand probes may have reduced
  coverage at MT-TE → false-negative risk. Always verify L-strand coverage in the
  sequencing QC report before excluding MT-TE pathogenic variants.

MATERNALLY INHERITED DIABETES MELLITUS (MIDM):
  m.14709T>C is one of only two mt-tRNA mutations causing DM as a prominent feature
  (the other is MT-TL1 m.3243A>G causing MIDD). MT-TE MIDM patients often present
  to endocrinology as 'type 1.5' or 'type 2 DM' with maternal family history of DM +
  neuromuscular symptoms — the combination should prompt mtDNA sequencing.
  Metformin is ABSOLUTE CI (Complex I inhibitor → fatal lactic acidosis).
  Insulin is the correct treatment for MT-TE MIDM; SGLT2i are investigational.

REVERSIBLE INFANTILE COX DEFICIENCY (m.14674T>C):
  Unique biology: m.14674T>C/G mutation in the discriminator position of tRNA-Glu
  impairs aminoacylation but the defect is partially compensated by a nuclear-encoded
  mt-EF-Tu/TARS2 rescue pathway that is upregulated during postnatal muscle maturation.
  Clinical: neonatal/infantile lactic acidosis + hypotonia + respiratory failure →
  gradual improvement → near-normal adult function in ~60% of patients.
  Muscle CIV activity may normalise by age 5–10 yr (heteroplasmy-dependent).
  This reversibility is UNIQUE among pathogenic mt-tRNA mutations and must be recognised
  to avoid inappropriate withdrawal of support in the neonatal intensive care unit.
"""

import random
from collections import Counter

SEED = 797
N_PATIENTS = 40

VARIANTS = [
    ("m.14709T>C", "Discriminator base / post-anticodon stem", "~35%; most common; CPEO + myopathy + MIDM (maternally inherited diabetes mellitus); lactic acidosis; cardiomyopathy ~45%; adult onset; CI+CIV combined fingerprint"),
    ("m.14674T>C", "Discriminator position (1st base after anticodon stem)", "~25%; DISTINCTIVE reversible infantile COX deficiency; neonatal lactic acidosis + respiratory failure → spontaneous improvement; near-normal adult in ~60%"),
    ("m.14693A>G", "Anticodon loop (position 34)", "~20%; CPEO + myopathy + lactic acidosis; moderate CI+CIV reduction; adult onset; less DM than m.14709T>C; EPRS2 nuclear DDx"),
    ("m.14706A>G", "T-stem / variable loop boundary", "~12%; mild myopathy + exercise intolerance; lower CI+CIV reduction; adult presentation; cardiomyopathy less common"),
    ("Large deletion", "Multi-gene spanning (ND6/TE/CYB region)", "~8%; deletion spanning MT-TE ± MT-ND6/MT-CYB → KSS/CPEO; multi-complex OXPHOS including CIII (if MT-CYB deleted); annual Holter mandatory"),
]

PHENOTYPES = [
    "CPEO + myopathy + MIDM — adult heteroplasmy 35–65%",
    "Reversible infantile COX deficiency — neonatal lactic acidosis improving with age",
    "CPEO + myopathy + cardiomyopathy — no DM",
    "Exercise intolerance + lactic acidosis — low heteroplasmy adult",
    "Multisystem encephalomyopathy + DM — high heteroplasmy",
]

TRIGGERS = [
    "Intercurrent illness / fever", "Fasting / prolonged NPO", "Metformin administration",
    "Anaesthetic agents (propofol)", "VPA/valproate", "Physiological stress / surgery",
    "Linezolid antibiotic", "Statin + CI combination", "Aminoglycosides (renal + SNHL risk)",
]

TREATMENTS = [
    ("CoQ10 (ubiquinol)", "Level C", "Mitochondrial cofactor; 10–30 mg/kg/day divided doses; ubiquinol preferred over ubiquinone for absorption"),
    ("Riboflavin (B2)", "Level C", "50–200 mg/day; FAD cofactor; modest CI+CIII support; low risk"),
    ("Thiamine (B1)", "MANDATORY empiric", "10–20 mg/kg/day IV acutely; PDH cofactor; empiric before workup complete; cannot harm"),
    ("Biotin", "MANDATORY empiric", "10 mg/day; BTD exclusion empiric; withdraw only after SLC19A3/BTD excluded"),
    ("L-Carnitine", "Level C", "50–100 mg/kg/day; carnitine depletion common in mt-disease; secondary deficiency correction"),
    ("L-Arginine IV", "Level C (SLE only)", "0.5 g/kg IV over 24h; for MELAS-like SLE overlap phenotype only; not routine MT-TE myopathy"),
    ("Insulin (MIDM)", "STANDARD", "First-line for MT-TE MIDM — NOT metformin; pancreatic beta-cell mitochondrial failure; basal-bolus regimen"),
    ("Elamipretide", "Phase 2 trials", "Cardiolipin stabiliser; investigational; early evidence in mt-myopathy; not yet standard"),
]

ABSOLUTE_CI = [
    ("Metformin", "ABSOLUTE CI — ALL MT-TE carriers", "Complex I inhibitor → fatal lactic acidosis; applies to MIDM patients — use insulin instead; even low-dose metformin in DM is CONTRAINDICATED"),
    ("VPA / valproate", "ABSOLUTE CI", "mt-ribosome inhibitor + CoA sequestration + POLG inhibition; catastrophic in CI+CIV deficiency; worsens tRNA aminoacylation defect; use LEV"),
    ("Propofol", "ABSOLUTE CI (PRIS)", "CIV inhibitor + propofol infusion syndrome; lethal in pre-existing CI+CIV deficiency; amplified by m.14674T>C partial reversibility; use sevoflurane"),
    ("Linezolid", "ABSOLUTE CI", "mt-23S rRNA inhibition → blocks ALL 13 mtDNA-encoded OXPHOS translations; compounding CI+CIV failure; use alternative antibiotics"),
    ("Chloramphenicol", "ABSOLUTE CI", "mt-ribosome inhibitor (70S); blocks mt-translation → catastrophic in tRNA-Glu defect; NEVER use"),
    ("Statin (high-dose)", "CAUTION / RELATIVE CI", "Complex I inhibitor at supraphysiological doses; increased risk in MT-TE myopathy; if essential, use low-dose pravastatin; monitor CK and lactate"),
    ("Aminoglycosides", "CAUTION", "Cochlear OXPHOS amplification → additive SNHL risk (esp. m.1555A>G check first); renal monitoring mandatory; short course only"),
    ("Ketogenic diet", "CONTRAINDICATED", "CI+CIV failure blocks beta-oxidation FADH2 processing; ketogenic diet → catastrophic metabolic acidosis"),
]

rng = random.Random(SEED)

def _make_patients(n=N_PATIENTS):
    pts = []
    cumulative_probs = []
    cumul = 0.0
    probs_raw = [0.35, 0.25, 0.20, 0.12, 0.08]
    for p in probs_raw:
        cumul += p
        cumulative_probs.append(cumul)

    for i in range(n):
        r = rng.random()
        vi = 0
        for j, cp in enumerate(cumulative_probs):
            if r <= cp:
                vi = j
                break
        variant_name = VARIANTS[vi][0]

        sex = rng.choice(["M", "F", "F"])  # slight female predominance in mt-tRNA disease

        # Phenotype assignment by variant
        if variant_name == "m.14709T>C":
            phenotype = rng.choices(PHENOTYPES, weights=[55, 0, 20, 15, 10])[0]
            age_onset = rng.randint(20, 55)
            het_blood = rng.randint(40, 80)
            ci = rng.randint(18, 45)
            civ = rng.randint(20, 48)
            cii = rng.randint(82, 105)
            lactate = round(rng.uniform(2.5, 7.5), 1)
            cpeo = rng.random() < 0.75
            myopathy = rng.random() < 0.70
            diabetes = rng.random() < 0.65
            cardiomyopathy = rng.random() < 0.45
            snhl = rng.random() < 0.35
            rrm = rng.random() < 0.55
            reversible = False
            neuropathy = rng.random() < 0.20
            optic_atrophy = rng.random() < 0.10
        elif variant_name == "m.14674T>C":
            phenotype = PHENOTYPES[1]
            age_onset = rng.randint(0, 1)  # neonatal / infantile
            het_blood = rng.randint(55, 90)
            ci = rng.randint(15, 50)  # may normalise with age
            civ = rng.randint(10, 40)  # can normalise
            cii = rng.randint(80, 105)
            lactate = round(rng.uniform(3.5, 12.0), 1)
            cpeo = False
            myopathy = True
            diabetes = rng.random() < 0.15
            cardiomyopathy = rng.random() < 0.25
            snhl = rng.random() < 0.20
            rrm = rng.random() < 0.30
            reversible = rng.random() < 0.60
            neuropathy = rng.random() < 0.10
            optic_atrophy = False
        elif variant_name == "m.14693A>G":
            phenotype = PHENOTYPES[0] if rng.random() < 0.3 else PHENOTYPES[2]
            age_onset = rng.randint(15, 50)
            het_blood = rng.randint(35, 75)
            ci = rng.randint(20, 50)
            civ = rng.randint(22, 52)
            cii = rng.randint(80, 105)
            lactate = round(rng.uniform(2.0, 6.5), 1)
            cpeo = rng.random() < 0.65
            myopathy = rng.random() < 0.65
            diabetes = rng.random() < 0.25
            cardiomyopathy = rng.random() < 0.35
            snhl = rng.random() < 0.30
            rrm = rng.random() < 0.50
            reversible = False
            neuropathy = rng.random() < 0.15
            optic_atrophy = rng.random() < 0.08
        elif variant_name == "m.14706A>G":
            phenotype = PHENOTYPES[3]
            age_onset = rng.randint(20, 60)
            het_blood = rng.randint(25, 60)
            ci = rng.randint(30, 65)
            civ = rng.randint(35, 65)
            cii = rng.randint(82, 105)
            lactate = round(rng.uniform(1.8, 4.5), 1)
            cpeo = rng.random() < 0.40
            myopathy = rng.random() < 0.55
            diabetes = rng.random() < 0.20
            cardiomyopathy = rng.random() < 0.20
            snhl = rng.random() < 0.20
            rrm = rng.random() < 0.35
            reversible = False
            neuropathy = rng.random() < 0.10
            optic_atrophy = rng.random() < 0.05
        else:  # Large deletion
            phenotype = PHENOTYPES[2]
            age_onset = rng.randint(5, 30)
            het_blood = rng.randint(20, 60)
            ci = rng.randint(10, 40)
            civ = rng.randint(12, 42)
            cii = rng.randint(78, 105)
            lactate = round(rng.uniform(2.5, 8.0), 1)
            cpeo = True
            myopathy = rng.random() < 0.80
            diabetes = rng.random() < 0.30
            cardiomyopathy = rng.random() < 0.45
            snhl = rng.random() < 0.55
            rrm = rng.random() < 0.75
            reversible = False
            neuropathy = rng.random() < 0.30
            optic_atrophy = rng.random() < 0.25

        pts.append({
            "id": f"MT-TE-{i+1:03d}",
            "variant": variant_name,
            "sex": sex,
            "age_onset": age_onset,
            "heteroplasmy_blood": het_blood,
            "ci_activity": ci,
            "civ_activity": civ,
            "cii_activity": cii,
            "lactate": lactate,
            "phenotype": phenotype,
            "cpeo": cpeo,
            "myopathy": myopathy,
            "diabetes": diabetes,
            "cardiomyopathy": cardiomyopathy,
            "snhl": snhl,
            "ragged_red_fibres": rrm,
            "reversible_civ": reversible,
            "neuropathy": neuropathy,
            "optic_atrophy": optic_atrophy,
            "kss": variant_name == "Large deletion" and rng.random() < 0.50,
        })
    return pts


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
    pct_diabetes = round(sum(1 for p in pts if p["diabetes"]) / n * 100)
    pct_cardio = round(sum(1 for p in pts if p["cardiomyopathy"]) / n * 100)
    pct_snhl = round(sum(1 for p in pts if p["snhl"]) / n * 100)
    pct_rrm = round(sum(1 for p in pts if p["ragged_red_fibres"]) / n * 100)
    pct_reversible = round(sum(1 for p in pts if p["reversible_civ"]) / n * 100)
    pct_optic = round(sum(1 for p in pts if p["optic_atrophy"]) / n * 100)

    return {
        "gene": "MT-TE",
        "omim": "*590025",
        "full_name": "Mitochondrially Encoded tRNA-Glu",
        "anticodon": "CUC",
        "rna_length_nt": 69,
        "rCRS_strand": "L-strand (light strand)",
        "rCRS_position": "14674–14742",
        "adjacent_genes": "MT-ND6 (ends 14673, L-strand) → MT-TE (14674-14742, L-strand) → MT-CYB (starts 14747, H-strand)",
        "primary_diseases": [
            "CPEO + Myopathy + Maternally Inherited Diabetes Mellitus (MIDM) — m.14709T>C",
            "Reversible Infantile COX Deficiency — m.14674T>C/G (UNIQUE neonatal → spontaneous improvement)",
            "CPEO + Myopathy + Cardiomyopathy — m.14693A>G",
            "Exercise Intolerance + Lactic Acidosis — m.14706A>G",
            "KSS / CPEO (Large Deletion spanning ND6/TE/CYB)",
        ],
        "oxphos_fingerprint": "CI + CIV reduced; CII (SDH, nuclear) NORMAL — mt-translation fingerprint (all tRNA-Glu-decoded OXPHOS subunits affected)",
        "inheritance": "MATERNAL — heteroplasmic; heteroplasmy threshold determines severity; blood underestimates muscle by 10–15%",
        "n_patients": n,
        "cohort_statistics": {
            "avg_heteroplasmy_blood_pct": avg_het,
            "avg_age_onset_yr": avg_onset,
            "avg_ci_activity_pct_normal": avg_ci,
            "avg_civ_activity_pct_normal": avg_civ,
            "avg_cii_activity_pct_normal": avg_cii,
            "pct_cpeo": pct_cpeo,
            "pct_myopathy": pct_myopathy,
            "pct_diabetes_mellitus": pct_diabetes,
            "pct_cardiomyopathy": pct_cardio,
            "pct_snhl": pct_snhl,
            "pct_ragged_red_fibres": pct_rrm,
            "pct_reversible_civ_deficiency": pct_reversible,
            "pct_optic_atrophy": pct_optic,
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
            {"range": "<40% blood", "phenotype": "Asymptomatic / isolated exercise intolerance; MIDM possible at low threshold", "management": "Annual metabolic review; DM screening; genetic counselling; AVOID metformin forever"},
            {"range": "40–60% blood", "phenotype": "CPEO + myopathy + exercise intolerance; MIDM likely (m.14709T>C); mild lactic acidosis", "management": "Ophthalmology (annual); endocrinology (DM management — insulin not metformin); CoQ10 + thiamine + biotin empiric"},
            {"range": "60–80% blood", "phenotype": "Full CPEO + myopathy + cardiomyopathy + DM + SNHL; lactic acidosis intercurrent", "management": "Multidisciplinary; cardiology echo annually; audiology; avoid CI drugs; GIR protocol for illness"},
            {"range": ">80% blood", "phenotype": "Multisystem encephalomyopathy + multi-organ failure; severe lactic acidosis; HIGH MORTALITY", "management": "ICU: GIR 6–8 mg/kg/min, IV thiamine, bicarb for pH <7.2, avoid propofol/metformin/VPA"},
            {"range": "m.14674T>C (any)", "phenotype": "Neonatal/infantile lactic acidosis + respiratory failure → SPONTANEOUS IMPROVEMENT (unique)", "management": "NICU support; avoid withdrawal of support — spontaneous recovery expected in ~60%; monitor CIV annually until normalised"},
        ],
        "key_molecular_features": [
            "MT-TE encodes tRNA-Glu (CUC anticodon) — 69 nt RNA gene; L-strand rCRS 14674–14742",
            "L-STRAND ENCODED: like MT-ND6; capture-based NGS may under-cover L-strand → false-negative risk; verify L-strand QC",
            "CI + CIV combined deficiency; CII NORMAL (nuclear SDH) = mt-translation fingerprint",
            "m.14709T>C: MIDM (maternally inherited diabetes mellitus) — one of only two mt-tRNA DM genes (the other: MT-TL1 m.3243A>G MIDD)",
            "m.14674T>C/G: REVERSIBLE infantile COX deficiency — UNIQUE; neonatal crisis → near-normal adult; DO NOT withdraw NICU support",
            "Flanked by MT-ND6 (L-strand) and MT-CYB (H-strand); large deletions span all three genes → KSS + CIII deficiency additional",
            "Metformin ABSOLUTE CI for ALL MT-TE carriers including subclinical MIDM — use insulin",
            "WES/WGS misses MT-TE — dedicated mtDNA panel with L-strand QC verification mandatory",
            "BTBGD (SLC19A3) MANDATORY exclusion — treatable Leigh-like mimic; thiamine + biotin empiric",
            "EPRS2 (mt-glutamyl-prolyl-tRNA synthetase) AR nuclear DDx — WES detectable vs maternal MT-TE",
        ],
        "clinical_alerts": [
            {"alert": "METFORMIN ABSOLUTE CI — ALL CARRIERS", "detail": "Complex I inhibitor → fatal lactic acidosis; even for MT-TE MIDM — use insulin; applies to subclinical/exercise carriers"},
            {"alert": "VPA ABSOLUTE CI", "detail": "mt-ribosome inhibitor + CoA sequestration + POLG; worsens tRNA-Glu aminoacylation defect catastrophically; use LEV"},
            {"alert": "PROPOFOL ABSOLUTE CI (PRIS)", "detail": "CIV inhibitor + propofol infusion syndrome; amplified in CI+CIV deficiency; NICU especially — reversible m.14674T>C patients need sevoflurane"},
            {"alert": "LINEZOLID ABSOLUTE CI", "detail": "mt-23S rRNA blocks all 13 mtDNA OXPHOS translations; compounding CI+CIV failure; use alternative antibiotics"},
            {"alert": "NEVER FAST", "detail": "Fasting triggers acute lactic crisis; GIR 6–8 mg/kg/min IV dextrose mandatory perioperatively; oral glucose if mild illness"},
            {"alert": "KD CONTRAINDICATED", "detail": "CI+CIV failure → beta-oxidation impaired; ketogenic diet causes catastrophic ketoacidosis"},
            {"alert": "m.14674T>C — DO NOT WITHDRAW NICU SUPPORT", "detail": "Reversible infantile COX deficiency — ~60% achieve near-normal by school age; aggressive NICU support justified; false impression of irreversibility leads to inappropriate withdrawal"},
            {"alert": "BTBGD MANDATORY EXCLUSION", "detail": "SLC19A3 Biotin-Thiamine-Responsive BGD — treatable Leigh-like mimic; thiamine + biotin empiric MANDATORY before confirmatory genetics"},
        ],
        "cohort_summary_features": [
            f"{pct_cpeo}% have CPEO (ptosis + ophthalmoplegia)",
            f"{pct_myopathy}% have myopathy (limb + oculopharyngeal)",
            f"{pct_diabetes}% have diabetes mellitus (MIDM — maternally inherited; DISTINCTIVE MT-TE feature)",
            f"{pct_cardio}% have cardiomyopathy",
            f"{pct_snhl}% have sensorineural hearing loss",
            f"{pct_rrm}% have ragged-red fibres on muscle biopsy",
            f"{pct_reversible}% show reversible CIV deficiency (m.14674T>C cohort; neonatal crisis → improvement)",
            f"Mean CI activity {avg_ci}% of normal (CIV {avg_civ}%; CII {avg_cii}% — NORMAL)",
            f"Mean blood heteroplasmy {avg_het}% (muscle ~10–15% higher)",
            f"Mean age at onset {avg_onset} yr (range neonatal–60 yr; m.14674T>C = neonatal/infantile)",
            "MT-TE = L-strand encoded tRNA-Glu; MIDM + reversible neonatal COX deficiency are UNIQUE MT-TE signatures",
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
        pct_diabetes = round(sum(1 for x in grp if x["diabetes"]) / len(grp) * 100)
        pct_cardio = round(sum(1 for x in grp if x["cardiomyopathy"]) / len(grp) * 100)
        pct_snhl = round(sum(1 for x in grp if x["snhl"]) / len(grp) * 100)
        pct_reversible = round(sum(1 for x in grp if x["reversible_civ"]) / len(grp) * 100)
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
            "pct_diabetes_mellitus": pct_diabetes,
            "pct_cardiomyopathy": pct_cardio,
            "pct_snhl": pct_snhl,
            "pct_reversible_civ": pct_reversible,
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
            "diabetes_mellitus": p["diabetes"],
            "cardiomyopathy": p["cardiomyopathy"],
            "snhl": p["snhl"],
            "ragged_red_fibres": p["ragged_red_fibres"],
            "reversible_civ": p["reversible_civ"],
            "neuropathy": p["neuropathy"],
            "optic_atrophy": p["optic_atrophy"],
            "kss": p["kss"],
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
            "CII_pct_normal": f"{round(sum(p['cii_activity'] for p in pts)/len(pts),1)} (NORMAL — nuclear-encoded SDH)",
            "pattern": "CI + CIV reduced; CII NORMAL → mt-translation fingerprint (tRNA-Glu defect)",
            "BN_PAGE": "CI band absent/reduced; CIV (COX) absent/reduced; CII (SDH) band PRESENT and NORMAL",
            "muscle_histochemistry": "RRF on Gomori trichrome; COX-negative fibres; SDH-positive ragged fibres; m.14674T>C: COX may normalise with age",
            "L_strand_NGS_note": "Verify L-strand coverage in QC report: MT-TE (14674-14742) on L-strand → coverage dropout risk in H-strand capture panels",
        },
        "ddx_comparison": [
            {
                "gene": "MT-TL1 (MELAS/MIDD)",
                "disease": "MELAS + MIDD (m.3243A>G)",
                "oxphos": "CI+CIII+CIV pan-OXPHOS",
                "distinguisher": "MOST CRITICAL DDx — MT-TL1 is THE DM tRNA gene (MIDD = maternally inherited diabetes + deafness, same m.3243A>G as MELAS); stroke-like episodes in MELAS ABSENT in MT-TE; MT-TL1 pan-OXPHOS (CI+CIII+CIV) vs MT-TE CI+CIV only; urine epithelial cells preferred for MT-TL1",
            },
            {
                "gene": "MT-TK (MERRF)",
                "disease": "MERRF + MSL — Myoclonic Epilepsy",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "Myoclonic epilepsy cardinal (90%); MSL/Madelung lipomatosis (10-25%); NO DM as cardinal feature; NO reversible neonatal form; Shoffner 1990 m.8344A>G",
            },
            {
                "gene": "MT-TH (tRNA-His)",
                "disease": "Combined CI+CIV — Leigh-like + CPEO + Cardiomyopathy",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "Prominent cardiomyopathy (HCM/DCM); Leigh-like MRI; less DM than MT-TE m.14709T>C; NO reversible neonatal form; m.12147G>A most common",
            },
            {
                "gene": "MT-TS2 (tRNA-Ser AGY)",
                "disease": "SHORTEST mt-tRNA — CPEO + SNHL + Myopathy",
                "oxphos": "CI+CIV (same fingerprint)",
                "distinguisher": "Isolated SNHL at low heteroplasmy DISTINCTIVE; less DM; NO reversible neonatal form; 59 nt (shortest mt-tRNA); m.12258C>A most common",
            },
            {
                "gene": "POLG (SANDO / Alpers)",
                "disease": "mtDNA depletion / deletion; AR nuclear",
                "oxphos": "Pan-OXPHOS; mtDNA depletion",
                "distinguisher": "HEPATOPATHY (30–50%) ABSENT in MT-TE; mtDNA depletion on Southern/qPCR ABSENT in MT-TE (stable mtDNA copy number); WES detects POLG; NO DM as primary feature; AR biallelic inheritance vs maternal MT-TE",
            },
            {
                "gene": "SLC19A3 (BTBGD)",
                "disease": "Biotin-Thiamine-Responsive BGD",
                "oxphos": "Normal OXPHOS biochemistry",
                "distinguisher": "TREATABLE — thiamine + biotin → MRI normalisation; Leigh-like MRI; NORMAL CI/CIV biochemistry; MANDATORY first exclusion especially in neonatal m.14674T>C overlap scenarios",
            },
            {
                "gene": "EPRS2 (mt-GluPro-tRNA Synthetase)",
                "disease": "Combined OXPHOS Deficiency — AR nuclear",
                "oxphos": "CI+CIV (GluPro-tRNA synthetase charges both tRNA-Glu and tRNA-Pro)",
                "distinguisher": "WES-detectable AR nuclear gene; biallelic EPRS2 mutations → similar CI+CIV fingerprint; maternal inheritance ABSENT; DM less prominent; key DDx for MT-TE m.14693A>G (anticodon loop overlap phenotype)",
            },
        ],
    }


def get_definitions():
    return {
        "gene_biology": {
            "MT-TE": "Mitochondrially encoded tRNA-Glu (CUC anticodon, reads GAA/GAG); 69 nt RNA gene; L-strand rCRS 14674–14742; flanked by MT-ND6 (ends 14673, L-strand) and MT-CYB (starts 14747, H-strand); OMIM *590025",
            "tRNA_Glu_function": "Delivers glutamic acid to the mt-ribosome A-site during translation of all 13 mtDNA-encoded OXPHOS subunits; CUC anticodon reads Glu codons GAA (wobble) and GAG (Watson-Crick); mt-aminoacyl-tRNA synthetase: EARS2 (mt-glutamyl-tRNA synthetase); EARS2 mutations cause LTBL (Leukoencephalopathy with Thalamus and Brainstem involvement and high Lactate)",
            "L_strand_encoding": "MT-TE is encoded on the LIGHT (L) strand, like MT-ND6, MT-TQ, MT-TA. In mtDNA NGS, if capture probes primarily target the H-strand, L-strand regions may have reduced coverage → false-negative MT-TE pathogenic variants. Sequencing report must confirm adequate L-strand depth at 14674–14742.",
            "mt_translation_fingerprint": "CI + CIV reduced; CII (SDH, nuclear-encoded) NORMAL; tRNA-Glu is required for inserting Glu into all 13 mtDNA-encoded OXPHOS polypeptides (7 CI ND-subunits, 1 CIII MT-CYB subunit, 3 CIV CO-subunits, 2 CV ATP-subunits) — hence all CI and CIV subunits are under-synthesised; CIII may be additionally affected in large deletions spanning MT-CYB",
            "reversible_CIV_deficiency": "m.14674T>C/G occurs at the discriminator position (position 73, 3' to acceptor stem) — this base is important for tRNA recognition by EARS2 but also serves as an identity element for nuclear-encoded mt-EF-Tu. Post-natal upregulation of compensatory mechanisms (TARS2/EF-Tu pathways) allows partial restoration of tRNA function and CIV activity in ~60% of m.14674T>C patients during childhood.",
            "MIDM_mechanism": "m.14709T>C impairs tRNA-Glu aminoacylation → deficient mt-translation in pancreatic beta-cells → mitochondrial ATP synthesis failure → impaired GSIS (glucose-stimulated insulin secretion) → diabetes mellitus. Concurrently, skeletal muscle CI+CIV deficiency causes myopathy. The combination of maternal DM + myopathy + CPEO is highly suggestive of MT-TE disease.",
            "rCRS_position": "L-strand rCRS 14674–14742 (69 nt); flanked by MT-ND6 (ends 14673) and MT-CYB (starts 14747); 4-nt intergenic spacer between MT-TE and MT-CYB; 1-nt overlap with MT-ND6 end at some transcript junctions",
        },
        "clinical_terms": {
            "MIDM": "Maternally Inherited Diabetes Mellitus — pancreatic beta-cell mitochondrial failure → insulin secretion defect; presents as 'type 1.5' or 'atypical type 2' DM with maternal family history of DM + neuromuscular features; m.14709T>C and MT-TL1 m.3243A>G (MIDD) are the two main mt-tRNA DM mutations; METFORMIN ABSOLUTE CI — use insulin",
            "Reversible_COX_deficiency": "m.14674T>C/G — neonatal/infantile cytochrome oxidase (CIV) deficiency with spontaneous improvement; unique MT-TE phenotype; lactic acidosis + hypotonia + respiratory failure in NICU → gradual CIV normalisation by school age in ~60%; heteroplasmy-dependent; DO NOT withdraw support prematurely",
            "CPEO": "Chronic Progressive External Ophthalmoplegia: slowly progressive bilateral ptosis + ophthalmoparesis; all EOMs eventually affected; prominent MT-TE feature; KSS if large deletion (CPEO + retinal pigmentation + cardiac conduction block <20 yr)",
            "mt_translation_fingerprint": "CI + CIV reduced; CII NORMAL — all 7 CI ND-subunits (ND1-ND6, ND4L) and 3 CIV CO-subunits (CO1-CO3) are mtDNA-encoded and require tRNA-Glu; CII (SDHA-D) all nuclear → NORMAL",
            "LTBL_EARS2_DDx": "Leukoencephalopathy with Thalamus and Brainstem involvement and high Lactate — caused by EARS2 mutations (mt-glutamyl-tRNA synthetase, the enzyme that charges tRNA-Glu); AR nuclear; WES detectable; DIFFERENT from maternal MT-TE mutations; MRI-dominated phenotype (white matter + thalamus); NO maternal inheritance",
            "L_strand_NGS_pitfall": "MT-TE at rCRS 14674–14742 is on the L (light) strand. H-strand capture mtDNA panels may show reduced read depth here. Laboratories using Illumina TruSight Mitochondrial or Agilent SureSelect mtDNA must verify L-strand QC. Failure to do so = false-negative MT-TE report.",
            "BTBGD": "Biotin-Thiamine-Responsive Basal Ganglia Disease (SLC19A3); Leigh-like MRI; TREATABLE; MANDATORY exclusion before MT-TE diagnosis; give thiamine + biotin empirically; can mimic both MT-TE neonatal and adult presentations",
        },
        "pharmacology": {
            "absolute_ci": {a[0]: a[2] for a in ABSOLUTE_CI},
            "preferred_aed": "Levetiracetam (LEV) — renal excretion; no mt-toxicity; first-line for seizures in MT-TE disease; avoid VPA (ABSOLUTE CI), phenobarbital (hepatic; cautious use), benzodiazepines (short-term adjunct only)",
            "dm_management": "INSULIN — basal-bolus regimen; NOT metformin (ABSOLUTE CI Complex I inhibitor); SGLT2 inhibitors investigational; DPP4 inhibitors may be cautiously used (limited mt-toxicity evidence); HbA1c target <7%; multidisciplinary endocrinology + neurology co-management",
            "emergency_protocol": "GIR 6–8 mg/kg/min IV dextrose + IV thiamine 10–20 mg/kg/dose + avoid fasting + bicarb for pH <7.2 + ICU monitoring; for m.14674T>C neonates: NICU respiratory support, NG feeds to avoid fasting, repeat CIV measurement every 12–18 months to track recovery",
            "anaesthetic_guidance": "Sevoflurane preferred (not propofol); regional anaesthesia if possible; GIR perioperatively; avoid prolonged fasting; ICU post-operative for major surgery; liaise with metabolic team pre-operatively",
        },
        "key_references": [
            "Seneca S et al. (2001) A mitochondrial tRNA aspartate mutation causing severe myopathy — Am J Hum Genet (early MT-TE/adjacent tRNA studies; phenotypic context)",
            "Tam EWY et al. (2008) Mitochondrial tRNA-Glu mutation m.14693A>G causing CPEO — J Child Neurol (MT-TE m.14693A>G functional study)",
            "Horvath R et al. (2009) Phenotypic spectrum associated with mutations of the mitochondrial polymerase gamma gene — Brain (POLG DDx vs MT-TE disease including DM comparison)",
            "Coulbault L et al. (2005) A novel mutation in the mitochondrial tRNA(Glu) gene — Biochem Biophys Res Commun (m.14709T>C CPEO + DM cohort; MIDM documentation)",
            "Boczonadi V & Horvath R (2014) Mitochondria: impaired mitochondrial translation in human disease — Int J Biochem Cell Biol (mt-translation tRNA defect review including MT-TE)",
            "Gorman GS et al. (2015) Mitochondrial diseases — Nat Rev Dis Primers (comprehensive review including mt-tRNA spectrum and MIDM)",
            "DiMauro S & Schon EA (2003) Mitochondrial respiratory-chain diseases — NEJM (tRNA mutation review including MT-TE context and reversible COX deficiency)",
            "Uusimaa J et al. (2011) Prevalence, segregation, and phenotype of the mitochondrial DNA 14709T>C mutation in children — Ann Neurol (MT-TE m.14709T>C paediatric prevalence and DM)",
        ],
    }
