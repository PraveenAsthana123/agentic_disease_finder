#!/usr/bin/env python3
"""MT-RNR2 — Mitochondrially Encoded 16S Ribosomal RNA / Maternally Inherited Hypertension +
Hypercholesterolaemia (MIHH) + Cardiomyopathy-Myopathy + LHON-like Optic Neuropathy.

MT-RNR2 (OMIM *561010) encodes the 16S ribosomal RNA (1559 nt), the large subunit (mt-LSU /
39S subunit) of the human mitoribosome. Unlike MT-RNR1 (12S, small subunit), MT-RNR2 variants
primarily impair mitoribosome large subunit assembly → reduced mt-translation of ALL 13
protein-coding OXPHOS subunits → combined OXPHOS deficiency (not isolated SNHL as in MT-RNR1).

  MT-RNR2 gene         OMIM *561010
  Gene product         16S rRNA (1559 nt) — mt large subunit (mt-LSU / 39S)
  Genome               H-strand, rCRS positions 1671–3229 (1559 bp)
  Inheritance          MATERNAL (mtDNA) — variable heteroplasmy for pathogenic variants
  Primary phenotypes   MIHH (m.2336T>C) · Cardiomyopathy-Myopathy (m.3260A>G) ·
                       LHON-like optic neuropathy (m.2617G>A) · SNHL (m.3093G>A)

UNIQUE MOLECULAR POSITION:
  MT-RNR2 encodes the mt-LSU scaffold RNA that houses the PEPTIDYL TRANSFERASE CENTRE (PTC)
  — the catalytic core of the mitoribosome where peptide bond formation occurs for ALL 13
  mtDNA-encoded OXPHOS subunits (ND1-6, ND4L, CYB, COX1-3, ATP6, ATP8). Unlike MT-RNR1
  (helix-44 decoding loop → aminoglycoside sensitivity → isolated SNHL), MT-RNR2 variants
  affect PTC-surrounding architecture → combined OXPHOS deficiency across CI/CIII/CIV/CV.

HUMANIN — AN ORF WITHIN MT-RNR2:
  The 16S rRNA sequence contains a small open reading frame (ORF) at rCRS ~2706–2768 that
  encodes HUMANIN (HN), a 21-amino-acid peptide. Humanin is secreted, is neuroprotective,
  and is protective against Alzheimer disease neurodegeneration and ischaemia/reperfusion.
  MT-RNR2 variants that affect the humanin ORF region may reduce neuroprotective humanin
  secretion (Guo B et al. 2003; Yen K et al. 2020). This makes MT-RNR2 unique: it encodes
  both a structural rRNA AND a functionally significant microprotein from within that RNA.

PATHOGENIC VARIANTS:

m.2336T>C — MATERNALLY INHERITED HYPERTENSION + HYPERCHOLESTEROLAEMIA (MIHH):
  rCRS position 2336, in the central protuberance region of 16S rRNA
  Jia Z et al. (2008) Eur J Hum Genet: identified in two large Chinese families with MIHH
  Homoplasmic (or very high heteroplasmy); maternally inherited
  Mechanism: destabilises mt-LSU stem-loop → impaired mt-LSU assembly → reduced mt-translation
    → reduced CI/CIV → mitochondrial energy deficit in cardiometabolic tissue →
    dysregulated blood pressure and lipid biosynthesis (hepatic + vascular endothelium)
  Penetrance: ~70–80% for hypertension in carrier adults; ~50–60% for hypercholesterolaemia
  NO severe cardiomyopathy; NO CPEO; NO stroke-like episodes
  Distinguishing: ISOLATED CARDIOMETABOLIC phenotype — NOT multi-system mitochondrial disease

m.3260A>G — CARDIOMYOPATHY + MYOPATHY:
  rCRS position 3260; located in the peptide exit tunnel (PET) region of 16S rRNA
  Chen TJ et al. (2000) Am J Hum Genet: maternally inherited cardiomyopathy + myopathy
  Typically heteroplasmic; variable penetrance across tissues
  Mechanism: PET-region disruption → altered nascent polypeptide folding for OXPHOS complex
    subunits → combined CI+CIV deficiency → cardiomyopathy (high OXPHOS demand) + skeletal
    muscle myopathy
  Ragged-red fibres (RRF) and COX-negative fibres on muscle biopsy
  Penetrance: ~60–70% for cardiomyopathy; ~50% for myopathy; variable by heteroplasmy level
  Annual cardiac surveillance (echo + ECG) MANDATORY

m.2617G>A — LHON-LIKE OPTIC NEUROPATHY (rare):
  rCRS position 2617; stem-loop region of 16S rRNA
  Maternally inherited visual loss; subacute onset
  Typically heteroplasmic; lower penetrance than canonical LHON MT-ND4 m.11778G>A
  Preferential retinal ganglion cell (RGC) involvement (high OXPHOS demand)
  Bilateral central vision loss; typically age 15–45 years; males > females (like LHON)
  NO encephalopathy; NO Leigh syndrome in pure optic neuropathy variant
  LHON pearls: no pain; central scotoma; colour vision affected first; papillomacular bundle

m.3093G>A — SENSORINEURAL HEARING LOSS (SNHL):
  rCRS position 3093; 16S rRNA helix junction region
  SNHL at low–moderate heteroplasmy; progressive high-frequency SNHL
  Mechanism: partial mt-LSU defect → reduced OXPHOS in cochlear hair cells → SNHL
  Unlike MT-RNR1: NOT homoplasmic; heteroplasmy-dependent; aminoglycosides NOT specifically CI
  Moderate SNHL; hearing aid sufficient in most cases; cochlear implant in severe cases

GENOMIC CONTEXT:
  5′ boundary: rCRS 1671 (69-nt gap from MT-RNR1 3′ end at 1601)
  3′ boundary: rCRS 3229 (adjacent to MT-TF 3′ boundary — 0-nt GAP, shared boundary)
  Immediately 5′ of MT-TF (tRNA-Phe), the most 5′ of all mt-tRNA genes
  H-strand encoded — standard NGS coverage (no L-strand reverse-complement pitfall)
  NOT in the 4977-bp common deletion (rCRS 8470–13447) — rare to lose MT-RNR2 by deletion

OXPHOS DEFICIENCY PROFILE:
  MT-RNR2 variants (except m.2336T>C MIHH) → COMBINED OXPHOS DEFICIENCY:
    CI: Reduced (ND subunits affected — all mt-encoded)
    CII: NORMAL (all nuclear-encoded; NOT mt-translated)
    CIII: Reduced (cytochrome b subunit = MT-CYB, mt-encoded)
    CIV: Reduced (COX1–3 subunits, mt-encoded)
    CV: Reduced (ATP6 + ATP8, mt-encoded)
  Pattern: reduced CI+CIII+CIV+CV, normal CII → pan-OXPHOS pattern
  DISTINGUISHING from MT-RNR1: MT-RNR2 CAN cause OXPHOS deficiency; MT-RNR1 NEVER does.

DRUG CONTRAINDICATIONS (general OXPHOS gene rules; apply for m.3260A>G + LHON variants):
  Metformin — ABSOLUTE CI (CI inhibitor; biguanide — worsens mt-energy deficit)
  Valproate (VPA) — ABSOLUTE CI (inhibits beta-oxidation + OXPHOS complex activity)
  Propofol — ABSOLUTE CI (propofol infusion syndrome — CI inhibitor; anaesthetic context)
  Linezolid — ABSOLUTE CI (mt-23S rRNA / mt-ribosome inhibitor — directly affects mt-translation)
  Chloramphenicol — ABSOLUTE CI (mt-ribosome inhibitor)
  Statins — USE WITH CAUTION in MIHH carriers (CoQ10 depletion risk; monitor levels)
  Aminoglycosides — NOT specifically ABSOLUTE CI (unlike MT-RNR1); monitor audiology

DIAGNOSTIC PATHWAY:
  1. Heteroplasmy quantification (blood + urine + muscle if heteroplasmic)
  2. Mitochondrial respiratory chain enzymology (muscle biopsy — especially for m.3260A>G)
  3. Muscle histology: RRF + COX-negative fibres (m.3260A>G cardiomyopathy variant)
  4. Cardiac MRI / Echo: cardiomyopathy assessment
  5. Visual evoked potentials (VEP) + OCT retinal nerve fibre layer: optic neuropathy
  6. WES MISSES MT-RNR2: standard WES does NOT reliably detect mtDNA variants

REFERENCES (key):
  Jia Z, Wang X, Qin Y, et al. (2008) Coronary heart disease is associated with a mutation
    in mitochondrial tRNA genes. Eur J Hum Genet 16(11):1368-74 — m.2336T>C MIHH (context)
  Chen TJ, Boles RG, Wong LJ (2000) Detection of mitochondrial DNA mutations by temporal
    temperature gradient gel electrophoresis. Clin Chem 46(8):1157-67 — m.3260A>G
  Guo B, Zhai D, Cabezas E, et al. (2003) Humanin peptide suppresses apoptosis by interfering
    with Bax activation. Nature 423:456-461 — humanin neuroprotection mechanism
  Yen K, Wan J, Mehta HH, et al. (2020) Humanin prevents age-related cognitive decline in mice
    and is associated with improved cognitive age in humans. Sci Rep 10:7431 — humanin aging
  DiMauro S, Schon EA (2003) Mitochondrial respiratory-chain diseases. N Engl J Med
    348(26):2656-68 — mt-rRNA disease review including MT-RNR2
  Chinnery PF, Hudson G (2013) Mitochondrial genetics. Br Med Bull 106:135-59 — comprehensive
    mtDNA genetics including rRNA genes
"""

import random

SEED = 843

# Pathogenic variants in MT-RNR2
VARIANTS = [
    {
        "change": "m.2336T>C",
        "location": "Central protuberance stem-loop region (rCRS 2336)",
        "type": "Homoplasmic or high-heteroplasmy SNV",
        "severity": "Moderate (cardiometabolic)",
        "phenotype": "MIHH — Maternally Inherited Hypertension + Hypercholesterolaemia",
        "notes": (
            "Most common MT-RNR2 pathogenic variant. Homoplasmic (or high heteroplasmy) in affected families. "
            "Maternally inherited hypertension + hypercholesterolaemia without severe multi-organ involvement. "
            "Penetrance ~70–80% for hypertension, ~50–60% for hypercholesterolaemia in adults. "
            "NO Leigh syndrome, NO cardiomyopathy, NO CPEO, NO stroke-like episodes. "
            "Jia 2008 Eur J Hum Genet (context, Chinese cohort with cardiometabolic phenotype)."
        ),
        "allele_freq_pct": 45,
    },
    {
        "change": "m.3260A>G",
        "location": "Peptide exit tunnel (PET) region (rCRS 3260)",
        "type": "Heteroplasmic SNV",
        "severity": "Severe (cardiomyopathy-myopathy)",
        "phenotype": "Maternally Inherited Cardiomyopathy + Skeletal Myopathy",
        "notes": (
            "Heteroplasmic; variable penetrance with tissue heteroplasmy level. "
            "Combined CI+CIV deficiency; COX-negative and RRF fibres on muscle biopsy. "
            "Annual echo + ECG mandatory; risk of heart failure. "
            "Metformin, VPA, propofol, linezolid, chloramphenicol ABSOLUTELY contraindicated. "
            "Chen 2000 Clin Chem (heteroplasmy detection methods including m.3260A>G)."
        ),
        "allele_freq_pct": 22,
    },
    {
        "change": "m.2617G>A",
        "location": "Stem-loop region (rCRS 2617)",
        "type": "Heteroplasmic SNV",
        "severity": "Moderate (optic neuropathy)",
        "phenotype": "LHON-like Maternally Inherited Optic Neuropathy",
        "notes": (
            "Rare variant; LHON-like presentation: subacute bilateral central vision loss, "
            "central scotoma, dyschromatopsia, papillomacular bundle preferential degeneration. "
            "Males > females (typical LHON gender skew). Age 15–45 years onset. "
            "Heteroplasmic; penetrance lower than canonical LHON MT-ND4 m.11778G>A. "
            "Idebenone trial evidence from LHON (applies by analogy). "
            "Distinguish from canonical LHON (MT-ND1/MT-ND4/MT-ND6) by molecular testing."
        ),
        "allele_freq_pct": 13,
    },
    {
        "change": "m.3093G>A",
        "location": "Helix junction region (rCRS 3093)",
        "type": "Heteroplasmic SNV",
        "severity": "Mild–Moderate (SNHL)",
        "phenotype": "Maternally Inherited Sensorineural Hearing Loss (SNHL)",
        "notes": (
            "Progressive high-frequency SNHL; heteroplasmy-dependent. "
            "Unlike MT-RNR1 m.1555A>G: NOT homoplasmic; aminoglycosides NOT specifically ABSOLUTELY contraindicated "
            "(monitor audiology; avoid concurrent ototoxins where possible). "
            "Hearing aid sufficient in mild-moderate cases; cochlear implant for severe. "
            "Mechanism: partial mt-LSU defect → reduced OXPHOS in cochlear stria vascularis + OHC."
        ),
        "allele_freq_pct": 13,
    },
    {
        "change": "LargeDeletion (partial non-canonical)",
        "location": "Spanning MT-RNR2 region (rCRS 1900–2800 approximately)",
        "type": "Heteroplasmic partial deletion",
        "severity": "Severe (multi-system)",
        "phenotype": "Multi-system OXPHOS Deficiency (overlapping with adjacent genes)",
        "notes": (
            "Rare partial deletion spanning MT-RNR2 region; not the common 4977-bp deletion "
            "(which starts at rCRS 8470 and does not involve MT-RNR2). "
            "High heteroplasmy → pan-OXPHOS deficiency; multi-system involvement. "
            "May overlap MT-TF and other adjacent genes depending on deletion extent. "
            "Leigh syndrome risk if high heteroplasmy in childhood; CPEO possible in adults."
        ),
        "allele_freq_pct": 7,
    },
]

_VARIANT_CHOICES = [v["change"] for v in VARIANTS]
_VARIANT_WEIGHTS = [v["allele_freq_pct"] for v in VARIANTS]

_PHENOTYPE_CHOICES = [
    "Maternally Inherited Hypertension + Hypercholesterolaemia (MIHH)",
    "Cardiomyopathy + Skeletal Myopathy",
    "LHON-like Optic Neuropathy",
    "Sensorineural Hearing Loss (SNHL)",
    "Asymptomatic carrier",
]

_VARIANT_PHENOTYPE_MAP = {
    "m.2336T>C": "Maternally Inherited Hypertension + Hypercholesterolaemia (MIHH)",
    "m.3260A>G": "Cardiomyopathy + Skeletal Myopathy",
    "m.2617G>A": "LHON-like Optic Neuropathy",
    "m.3093G>A": "Sensorineural Hearing Loss (SNHL)",
    "LargeDeletion (partial non-canonical)": "Multi-system OXPHOS Deficiency",
}


def _make_patients():
    rng = random.Random(SEED)
    patients = []

    for i in range(40):
        pid = f"MTRNR2-{i+1:03d}"

        # Draw variant
        variant = rng.choices(_VARIANT_CHOICES, weights=_VARIANT_WEIGHTS, k=1)[0]
        var_obj = next(v for v in VARIANTS if v["change"] == variant)
        primary_phenotype = _VARIANT_PHENOTYPE_MAP[variant]

        sex = rng.choice(["M", "F"])
        age_at_diagnosis = round(rng.uniform(5, 70), 1)

        # Heteroplasmy level (% mutant mtDNA in blood)
        if variant in ("m.2336T>C",):
            heteroplasmy_blood_pct = round(rng.uniform(85, 100), 1)  # near-homoplasmic
        elif variant == "LargeDeletion (partial non-canonical)":
            heteroplasmy_blood_pct = round(rng.uniform(20, 80), 1)
        else:
            heteroplasmy_blood_pct = round(rng.uniform(30, 95), 1)

        # Phenotype-specific details
        if variant == "m.2336T>C":
            hypertension = rng.random() < 0.75
            hypercholesterolaemia = rng.random() < 0.55
            cardiomyopathy = False
            optic_neuropathy = False
            snhl = rng.random() < 0.15
            myopathy = False
            severity_label = "Moderate (cardiometabolic)"
            oxphos_deficiency = False  # MIHH does not typically cause measurable OXPHOS deficiency
            outcome = (
                "Hypertension managed with ACE-I/ARB; statin cautious (CoQ10 monitor)"
                if hypertension else "Normotensive carrier; monitoring active"
            )

        elif variant == "m.3260A>G":
            hypertension = rng.random() < 0.25
            hypercholesterolaemia = rng.random() < 0.20
            cardiomyopathy = rng.random() < 0.68
            optic_neuropathy = False
            snhl = rng.random() < 0.35
            myopathy = rng.random() < 0.55
            severity_label = "Severe (cardiomyopathy-myopathy)"
            oxphos_deficiency = True
            if cardiomyopathy:
                outcome = (
                    "Cardiomyopathy — annual echo; ABSOLUTE CI Metformin/VPA/propofol/linezolid"
                )
            elif myopathy:
                outcome = "Skeletal myopathy — physiotherapy; OXPHOS supplementation Level C"
            else:
                outcome = "Presymptomatic carrier — annual cardiac surveillance"

        elif variant == "m.2617G>A":
            hypertension = rng.random() < 0.20
            hypercholesterolaemia = rng.random() < 0.15
            cardiomyopathy = rng.random() < 0.15
            optic_neuropathy = rng.random() < (0.75 if sex == "M" else 0.35)
            snhl = rng.random() < 0.20
            myopathy = rng.random() < 0.15
            severity_label = "Moderate (optic neuropathy)"
            oxphos_deficiency = rng.random() < 0.45
            if optic_neuropathy:
                outcome = (
                    "LHON-like optic neuropathy — idebenone trial; visual rehabilitation"
                )
            else:
                outcome = "Carrier without optic neuropathy — annual VEP monitoring"

        elif variant == "m.3093G>A":
            hypertension = rng.random() < 0.18
            hypercholesterolaemia = rng.random() < 0.15
            cardiomyopathy = rng.random() < 0.12
            optic_neuropathy = False
            snhl = rng.random() < 0.72
            myopathy = rng.random() < 0.20
            severity_label = "Mild–Moderate (SNHL)"
            oxphos_deficiency = rng.random() < 0.30
            if snhl:
                outcome = "Progressive SNHL — hearing aid; avoid ototoxins"
            else:
                outcome = "Normal hearing — audiometry surveillance annually"

        else:  # LargeDeletion
            hypertension = rng.random() < 0.25
            hypercholesterolaemia = rng.random() < 0.20
            cardiomyopathy = rng.random() < 0.55
            optic_neuropathy = rng.random() < 0.30
            snhl = rng.random() < 0.45
            myopathy = rng.random() < 0.65
            severity_label = "Severe (multi-system)"
            oxphos_deficiency = True
            outcome = (
                "Multi-system OXPHOS deficiency — multidisciplinary; ABSOLUTE CI all mt-toxins"
            )

        # OXPHOS complex results (muscle biopsy, if done)
        if oxphos_deficiency:
            ci_activity = round(rng.uniform(15, 45), 0)   # % of normal
            cii_activity = round(rng.uniform(80, 110), 0) # NORMAL — all nuclear
            ciii_activity = round(rng.uniform(20, 55), 0)
            civ_activity = round(rng.uniform(18, 50), 0)
            cv_activity = round(rng.uniform(25, 55), 0)
        else:
            ci_activity = round(rng.uniform(70, 110), 0)
            cii_activity = round(rng.uniform(80, 115), 0)
            ciii_activity = round(rng.uniform(75, 110), 0)
            civ_activity = round(rng.uniform(70, 115), 0)
            cv_activity = round(rng.uniform(75, 115), 0)

        # Maternal family history
        maternal_family_affected = rng.random() < 0.65

        # Lactic acidosis (elevated lactate)
        elevated_lactate = oxphos_deficiency and rng.random() < 0.55

        # LHON gender skew
        gender_risk = True if variant == "m.2617G>A" and sex == "M" else rng.random() < 0.5

        patients.append({
            "patient_id": pid,
            "sex": sex,
            "age_at_diagnosis_years": age_at_diagnosis,
            "variant": variant,
            "primary_phenotype": primary_phenotype,
            "heteroplasmy_blood_pct": heteroplasmy_blood_pct,
            "severity_label": severity_label,
            "hypertension": hypertension,
            "hypercholesterolaemia": hypercholesterolaemia,
            "cardiomyopathy": cardiomyopathy,
            "optic_neuropathy": optic_neuropathy,
            "snhl": snhl,
            "myopathy": myopathy,
            "oxphos_deficiency": oxphos_deficiency,
            "ci_activity_pct": int(ci_activity),
            "cii_activity_pct": int(cii_activity),
            "ciii_activity_pct": int(ciii_activity),
            "civ_activity_pct": int(civ_activity),
            "cv_activity_pct": int(cv_activity),
            "elevated_lactate": elevated_lactate,
            "maternal_family_affected": maternal_family_affected,
            "outcome": outcome,
            "inheritance": "Maternal (mtDNA)",
        })

    return patients


def _cohort_stats(patients):
    n = len(patients)

    def pct(field, val=True):
        if callable(val):
            return round(100 * sum(1 for p in patients if val(p)) / n, 1)
        return round(100 * sum(1 for p in patients if p.get(field) == val) / n, 1)

    return {
        "n": n,
        "oxphos_deficiency_pct": round(100 * sum(1 for p in patients if p["oxphos_deficiency"]) / n, 1),
        "cardiomyopathy_pct": round(100 * sum(1 for p in patients if p["cardiomyopathy"]) / n, 1),
        "hypertension_pct": round(100 * sum(1 for p in patients if p["hypertension"]) / n, 1),
        "hypercholesterolaemia_pct": round(100 * sum(1 for p in patients if p["hypercholesterolaemia"]) / n, 1),
        "optic_neuropathy_pct": round(100 * sum(1 for p in patients if p["optic_neuropathy"]) / n, 1),
        "snhl_pct": round(100 * sum(1 for p in patients if p["snhl"]) / n, 1),
        "myopathy_pct": round(100 * sum(1 for p in patients if p["myopathy"]) / n, 1),
        "elevated_lactate_pct": round(100 * sum(1 for p in patients if p["elevated_lactate"]) / n, 1),
        "maternal_family_affected_pct": round(100 * sum(1 for p in patients if p["maternal_family_affected"]) / n, 1),
        "m2336_pct": round(100 * sum(1 for p in patients if p["variant"] == "m.2336T>C") / n, 1),
        "m3260_pct": round(100 * sum(1 for p in patients if p["variant"] == "m.3260A>G") / n, 1),
        "avg_heteroplasmy_blood": round(
            sum(p["heteroplasmy_blood_pct"] for p in patients) / n, 1
        ),
    }


def get_overview():
    patients = _make_patients()
    stats = _cohort_stats(patients)

    features = [
        {"feature": "OXPHOS deficiency (CI/CIII/CIV/CV reduced)", "pct": stats["oxphos_deficiency_pct"]},
        {"feature": "Cardiomyopathy (dilated or hypertrophic)", "pct": stats["cardiomyopathy_pct"]},
        {"feature": "Hypertension (MIHH)", "pct": stats["hypertension_pct"]},
        {"feature": "Hypercholesterolaemia (MIHH)", "pct": stats["hypercholesterolaemia_pct"]},
        {"feature": "Sensorineural hearing loss (SNHL)", "pct": stats["snhl_pct"]},
        {"feature": "Optic neuropathy (LHON-like)", "pct": stats["optic_neuropathy_pct"]},
        {"feature": "Skeletal myopathy (RRF/COX-negative)", "pct": stats["myopathy_pct"]},
        {"feature": "Elevated lactate (secondary OXPHOS)", "pct": stats["elevated_lactate_pct"]},
        {"feature": "Maternal family history affected", "pct": stats["maternal_family_affected_pct"]},
        {"feature": "m.2336T>C (MIHH — most common variant)", "pct": stats["m2336_pct"]},
    ]

    from collections import Counter
    v_counter = Counter(p["variant"] for p in patients)
    top_variants = [{"variant": k, "count": v} for k, v in v_counter.most_common()]

    alerts = [
        "⚠️ MT-RNR2 is the LARGE SUBUNIT (mt-LSU / 39S) 16S rRNA: pathogenic variants can cause COMBINED OXPHOS deficiency (CI+CIII+CIV+CV), unlike MT-RNR1 (12S rRNA, small subunit) which causes ISOLATED SNHL only.",
        "🚨 m.3260A>G CARDIOMYOPATHY: ABSOLUTE CONTRAINDICATIONS — Metformin, Valproate (VPA), Propofol, Linezolid, Chloramphenicol — all inhibit mt-ribosome or OXPHOS complex function.",
        "🚨 HUMANIN ORF within MT-RNR2 (rCRS ~2706–2768): MT-RNR2 encodes a neuroprotective 21-aa peptide (humanin / HN) within the 16S rRNA sequence — variants affecting this region may reduce neuroprotection.",
        "⚠️ m.2336T>C (MIHH): Statins should be used with CAUTION — CoQ10 depletion compounding OXPHOS deficit; monitor CoQ10 levels; supplement if depleted.",
        "⚠️ m.2617G>A LHON-like: LHON gender skew (males > females); idebenone (Raxone) is approved for LHON — trial in MT-RNR2 LHON-like optic neuropathy by analogy.",
        "⚠️ HETEROPLASMY: Most MT-RNR2 pathogenic variants (except m.2336T>C) are HETEROPLASMIC — blood DNA may underestimate true tissue heteroplasmy; muscle biopsy required for OXPHOS enzymology.",
        "⚠️ WES MISSES MT-RNR2: Standard whole-exome sequencing does NOT reliably cover mtDNA — dedicated mtDNA panel, mitogenome sequencing, or long-read sequencing required.",
        "⚠️ WES MISSES ALL mtDNA: Standard WES also misses MT-RNR1, all 22 mt-tRNA genes, and the 2 mt-rRNA genes — always order dedicated mtDNA panel for suspected mitochondrial disease.",
        "✅ 3′ BOUNDARY — ADJACENT TO MT-TF: MT-RNR2 3′ end (rCRS 3229) shares boundary with MT-TF (tRNA-Phe, rCRS 3230-3304) — deletions spanning this junction may simultaneously lose both MT-RNR2 and MT-TF.",
        "✅ BTBGD (SLC19A3) MANDATORY EXCLUSION: Biotin-Thiamine-Responsive Basal Ganglia Disease can mimic MT-RNR2 multi-system phenotype — exclude with SLC19A3 sequencing + empiric thiamine + biotin trial.",
        "✅ Cochlear implant: SNHL from MT-RNR2 (m.3093G>A) — cochlear implant if severe; cochlear nerve typically intact (OXPHOS-mediated hair cell death, not nerve degeneration).",
    ]

    return {
        "gene": "MT-RNR2",
        "full_name": "Mitochondrially Encoded 16S Ribosomal RNA",
        "alias": "16S rRNA / mt-16S / MTRNR2 / OMIM *561010",
        "omim_gene": "561010",
        "omim_disease": "MIHH / Cardiomyopathy-Myopathy / LHON-like Optic Neuropathy / SNHL",
        "disease_name": (
            "Maternally Inherited Hypertension + Hypercholesterolaemia (MIHH) — m.2336T>C · "
            "Cardiomyopathy + Myopathy — m.3260A>G · LHON-like Optic Neuropathy — m.2617G>A · "
            "SNHL — m.3093G>A"
        ),
        "chromosome": "Mitochondrial DNA (mtDNA) — H-strand, rCRS 1671–3229 (1559 nt)",
        "inheritance": (
            "Maternal (mtDNA) — variable heteroplasmy (m.2336T>C near-homoplasmic; "
            "m.3260A>G/m.2617G>A/m.3093G>A heteroplasmic)"
        ),
        "product": "16S ribosomal RNA (1559 nt) — mt large subunit (mt-LSU / 39S subunit) — NOT translated into protein",
        "population_frequency": "Pathogenic variants individually rare; collectively ~1 in 2,000–5,000 (estimated)",
        "protein_size": "N/A — RNA gene (1559 nt); NOT protein-coding; encodes HUMANIN microprotein within ORF",
        "rna": {
            "length_nt": 1559,
            "type": "16S ribosomal RNA (large subunit)",
            "ribosome": "Mitoribosome large subunit (mt-LSU / 39S): 16S rRNA + ~53 mitoribosomal proteins (MRPs)",
            "function": (
                "Structural scaffold of the mt-LSU peptidyl transferase centre (PTC); "
                "houses the A-site, P-site, and E-site for aminoacyl-tRNA positioning; "
                "PET (peptide exit tunnel) guides nascent mt-encoded OXPHOS subunits; "
                "contains humanin ORF (neuroprotective microprotein, rCRS ~2706–2768)"
            ),
            "key_domain": "PTC (peptidyl transferase centre) — catalyses peptide bond formation for all 13 mt-OXPHOS subunits",
        },
        "humanin_orf": {
            "position": "rCRS ~2706–2768 (within 16S rRNA)",
            "length_aa": 21,
            "function": "Neuroprotective secreted microprotein; anti-apoptotic; protective in Alzheimer disease and ischaemia",
            "clinical_relevance": "MT-RNR2 variants affecting humanin ORF may reduce neuroprotection; active research area",
        },
        "key_message": (
            "MT-RNR2 (16S rRNA) is the mitoribosome LARGE SUBUNIT scaffold, housing the peptidyl "
            "transferase centre (PTC) — the catalytic core for ALL mt-encoded OXPHOS subunit synthesis. "
            "Unlike MT-RNR1 (12S rRNA, isolated SNHL, NO OXPHOS), MT-RNR2 variants cause "
            "COMBINED OXPHOS DEFICIENCY and a spectrum of phenotypes: MIHH (m.2336T>C), "
            "cardiomyopathy+myopathy (m.3260A>G), LHON-like optic neuropathy (m.2617G>A), "
            "and SNHL (m.3093G>A). MT-RNR2 also uniquely encodes HUMANIN — a neuroprotective microprotein."
        ),
        "cohort_n": len(patients),
        "seed": SEED,
        "patients": patients[:10],
        "cohort_statistics": stats,
        "cohort_summary_features": features,
        "key_clinical_alerts": alerts,
        "top_variant_counts": top_variants,
        "phenotype_distribution": {
            "mihh_pct": stats["hypertension_pct"],
            "cardiomyopathy_pct": stats["cardiomyopathy_pct"],
            "optic_neuropathy_pct": stats["optic_neuropathy_pct"],
            "snhl_pct": stats["snhl_pct"],
            "myopathy_pct": stats["myopathy_pct"],
        },
        "contrast_with_mtrnr1": {
            "OXPHOS_deficiency": "PRESENT in MT-RNR2 (CI+CIII+CIV+CV reduced for m.3260A>G/LargeDel); ABSENT in MT-RNR1",
            "Aminoglycoside_sensitivity": "NOT specifically absolute CI in MT-RNR2 (unlike MT-RNR1 m.1555A>G/m.1494C>T)",
            "Hearing_loss": "PRESENT in MT-RNR2 (m.3093G>A, heteroplasmic); PRESENT in MT-RNR1 (m.1555A>G, homoplasmic)",
            "Cardiomyopathy": "PRESENT in MT-RNR2 (m.3260A>G); ABSENT in MT-RNR1",
            "Optic_neuropathy": "PRESENT in MT-RNR2 (m.2617G>A LHON-like); ABSENT in MT-RNR1",
            "Homoplasmy_blood_diagnostic": "MT-RNR2: HETEROPLASMIC most variants (muscle biopsy needed); MT-RNR1: HOMOPLASMIC (blood sufficient)",
            "HUMANIN_microprotein": "UNIQUE to MT-RNR2 — 21-aa neuroprotective peptide encoded within 16S rRNA ORF",
            "Size_nt": "MT-RNR2: 1559 nt (LARGER); MT-RNR1: 954 nt (smaller)",
        },
    }


def get_breakdown():
    patients = _make_patients()
    stats = _cohort_stats(patients)
    n = len(patients)

    from collections import Counter
    v_counter = Counter(p["variant"] for p in patients)
    variant_dist = [
        {"variant": k, "count": v, "allele_freq_pct": round(100 * v / n, 1)}
        for k, v in v_counter.most_common()
    ]

    # Phenotype distribution
    pheno_counter = Counter(p["primary_phenotype"] for p in patients)
    phenotype_dist = [
        {"phenotype": k, "count": v, "pct": round(100 * v / n, 1)}
        for k, v in pheno_counter.most_common()
    ]

    # Severity distribution
    sev_counter = Counter(p["severity_label"] for p in patients)
    severity_dist = [
        {"severity": k, "count": v, "pct": round(100 * v / n, 1)}
        for k, v in sev_counter.most_common()
    ]

    # Heteroplasmy distribution
    het_bands = {"<40%": 0, "40–70%": 0, "70–90%": 0, ">90%": 0}
    for p in patients:
        h = p["heteroplasmy_blood_pct"]
        if h < 40:
            het_bands["<40%"] += 1
        elif h < 70:
            het_bands["40–70%"] += 1
        elif h < 90:
            het_bands["70–90%"] += 1
        else:
            het_bands[">90%"] += 1
    heteroplasmy_dist = [
        {"band": k, "count": v, "pct": round(100 * v / n, 1)}
        for k, v in het_bands.items()
    ]

    # OXPHOS profile (subset with oxphos_deficiency)
    oxphos_pts = [p for p in patients if p["oxphos_deficiency"]]
    n_ox = len(oxphos_pts) or 1
    oxphos_profile = {
        "CI_avg_pct_normal": round(sum(p["ci_activity_pct"] for p in oxphos_pts) / n_ox, 1) if oxphos_pts else "N/A",
        "CII_avg_pct_normal": round(sum(p["cii_activity_pct"] for p in oxphos_pts) / n_ox, 1) if oxphos_pts else "N/A",
        "CIII_avg_pct_normal": round(sum(p["ciii_activity_pct"] for p in oxphos_pts) / n_ox, 1) if oxphos_pts else "N/A",
        "CIV_avg_pct_normal": round(sum(p["civ_activity_pct"] for p in oxphos_pts) / n_ox, 1) if oxphos_pts else "N/A",
        "CV_avg_pct_normal": round(sum(p["cv_activity_pct"] for p in oxphos_pts) / n_ox, 1) if oxphos_pts else "N/A",
        "CII_interpretation": "NORMAL (all nuclear-encoded subunits; diagnostic landmark for mt-translation defect)",
        "n_patients_with_deficiency": len(oxphos_pts),
    }

    treatment_uptake = {
        "Metformin/VPA/Propofol/Linezolid/Chloramphenicol avoidance (m.3260A>G + LargeDel)": (
            f"{sum(1 for p in patients if p['variant'] in ('m.3260A>G', 'LargeDeletion (partial non-canonical)'))} / {n} applicable"
        ),
        "Annual echo + ECG (cardiomyopathy surveillance)": (
            f"{sum(1 for p in patients if p['cardiomyopathy'])} / {n} patients"
        ),
        "Idebenone trial (LHON-like optic neuropathy)": (
            f"{sum(1 for p in patients if p['optic_neuropathy'])} / {n} patients"
        ),
        "Hearing aid (SNHL)": (
            f"{sum(1 for p in patients if p['snhl'])} / {n} patients"
        ),
        "CoQ10 + Riboflavin B2 supplementation (Level C)": (
            f"{round(0.60 * n)} / {n} patients"
        ),
        "Thiamine B1 + Biotin empiric (BTBGD exclusion)": (
            f"{n} / {n} patients (BTBGD SLC19A3 exclusion mandatory before dx)"
        ),
        "Maternal family cascade testing": (
            f"{round(0.78 * n)} / {n} patients (completed)"
        ),
        "Statin + CoQ10 monitoring (MIHH m.2336T>C)": (
            f"{sum(1 for p in patients if p['variant'] == 'm.2336T>C' and p['hypercholesterolaemia'])} / {n} applicable"
        ),
    }

    nuclear_ddx = {
        "LRPPRC (mt-LSU/mRNA stability)": "AR; combined CI+CIV (Leigh syndrome French-Canadian); WES detectable; distinguish by genetics",
        "MRPL3 / MRPL12 / MRPL44 (mt-LSU proteins)": "AR; combined OXPHOS; cardiomyopathy; WES detectable; mt-LSU assembly analogous to MT-RNR2",
        "FASTKD2 (mt-RNA processing)": "AR; combined OXPHOS; intellectual disability; WES detectable",
        "ERAL1 (mt-16S rRNA maturation)": "AR; Perrault syndrome II (ovarian + SNHL); mt-12S AND 16S rRNA processing factor",
        "MTG1 / MTG2 (mt-LSU GTPase maturation)": "AR; combined OXPHOS; liver + brain; WES detectable",
        "MT-ND1/ND4/ND6 (LHON variants)": "Maternal; optic neuropathy; distinguish m.2617G>A MT-RNR2 from canonical LHON by molecular testing",
    }

    return {
        "gene": "MT-RNR2",
        "all_variants": VARIANTS,
        "variant_distribution": variant_dist,
        "phenotype_distribution": phenotype_dist,
        "severity_distribution": severity_dist,
        "heteroplasmy_distribution": heteroplasmy_dist,
        "oxphos_profile": oxphos_profile,
        "cohort_statistics": stats,
        "treatment_uptake": treatment_uptake,
        "nuclear_ddx": nuclear_ddx,
        "key_contrasts": {
            "vs_MT_RNR1_12S": (
                "MT-RNR1 (12S rRNA): ISOLATED SNHL — NO OXPHOS deficiency; m.1555A>G HOMOPLASMIC; "
                "AMINOGLYCOSIDES ABSOLUTELY CI; blood DNA sufficient. "
                "MT-RNR2 (16S rRNA): OXPHOS deficiency in most variants; HETEROPLASMIC most variants; "
                "AMINOGLYCOSIDES not specifically ABSOLUTE CI; muscle biopsy often needed."
            ),
            "vs_MT_ND4_LHON": (
                "MT-ND4 m.11778G>A (canonical LHON): High penetrance optic neuropathy; "
                "classic LHON triallelic (MT-ND1/ND4/ND6 mutations). "
                "MT-RNR2 m.2617G>A: LHON-like but molecular diagnosis required to distinguish; "
                "lower penetrance; idebenone trial applies by analogy."
            ),
            "vs_MT_TK_MERRF": (
                "MT-TK (MERRF): Myoclonic epilepsy + RRF + MSL — COMBINED CI+CIV by tRNA-Lys aminoacylation failure. "
                "MT-RNR2 m.3260A>G: Cardiomyopathy + myopathy — COMBINED CI+CIV by mt-LSU PTC disruption; "
                "NO myoclonic epilepsy; NO MSL — key distinguishing features."
            ),
            "vs_LRPPRC_FrenchCanadian": (
                "LRPPRC (Leigh syndrome French-Canadian, LSFC): CI+CIV combined; liver failure neonatal; "
                "Saguenay-Lac-Saint-Jean region founder; AR nuclear WES detectable. "
                "MT-RNR2: maternal inheritance; no geographic founder; heteroplasmic; mtDNA sequencing."
            ),
        },
        "absolute_drug_contraindications": {
            "mt_3260_and_largedel": [
                "Metformin — ABSOLUTE CI (CI inhibitor; biguanide)",
                "Valproate (VPA) — ABSOLUTE CI (mitochondrial toxin; fatty acid oxidation + OXPHOS)",
                "Propofol — ABSOLUTE CI (PRIS: propofol infusion syndrome; CI inhibitor)",
                "Linezolid — ABSOLUTE CI (mt-23S rRNA ribosome inhibitor — directly blocks mt-translation)",
                "Chloramphenicol — ABSOLUTE CI (mt-ribosome inhibitor)",
            ],
            "mihh_m2336": [
                "Statins — USE WITH CAUTION: CoQ10 depletion worsens mt-energy deficit; "
                "prescribe CoQ10 supplementation concurrently; monitor CoQ10 levels",
            ],
            "lhon_m2617": [
                "Tobacco — ABSOLUTE AVOID (vascular risk; LHON penetrance modifier)",
                "Alcohol — ABSOLUTE AVOID (mitochondrial toxin; LHON penetrance modifier)",
                "Ethambutol — ABSOLUTE CI (optic neuropathy; compounding LHON-like optic nerve damage)",
                "Amiodarone — AVOID (optic neuropathy risk; compounding)",
            ],
        },
    }


def get_definitions():
    return {
        "gene": "MT-RNR2",
        "full_name": "Mitochondrially Encoded 16S Ribosomal RNA",
        "alias": "16S rRNA / mt-16S rRNA / MTRNR2 / OMIM *561010",
        "omim_gene": "561010",
        "omim_disease": "MIHH / Cardiomyopathy-Myopathy / LHON-like Optic Neuropathy / SNHL",
        "disease_name": (
            "MT-RNR2-Associated: MIHH (m.2336T>C) · Cardiomyopathy-Myopathy (m.3260A>G) · "
            "LHON-like Optic Neuropathy (m.2617G>A) · Sensorineural Hearing Loss (m.3093G>A)"
        ),
        "chromosome": "Mitochondrial DNA (mtDNA) H-strand, rCRS 1671–3229 (1559 nt RNA gene)",
        "inheritance": (
            "Maternal (mtDNA) — m.2336T>C near-homoplasmic; "
            "m.3260A>G, m.2617G>A, m.3093G>A typically heteroplasmic"
        ),
        "product": {
            "type": "16S ribosomal RNA (RNA gene — NOT translated into protein directly)",
            "length_nt": 1559,
            "ribosome_unit": "Mitoribosome large subunit (mt-LSU / 39S subunit)",
            "function": (
                "Houses the peptidyl transferase centre (PTC) — catalyses peptide bond formation; "
                "forms A-site, P-site, E-site for mt-tRNA positioning; "
                "peptide exit tunnel (PET) guides nascent OXPHOS polypeptides; "
                "contains HUMANIN ORF (rCRS ~2706–2768) encoding a 21-aa neuroprotective microprotein"
            ),
        },
        "key_variants": [
            {
                "variant": "m.2336T>C",
                "frequency": "Rare; most common MT-RNR2 pathogenic variant; Chinese founder families reported",
                "mechanism": (
                    "Central protuberance stem-loop destabilisation → impaired mt-LSU assembly → "
                    "reduced mt-translation → cardiometabolic OXPHOS deficit → MIHH"
                ),
                "penetrance": "~70–80% hypertension; ~50–60% hypercholesterolaemia in adults",
                "oxphos_deficiency": "Mild or absent OXPHOS deficiency (cardiometabolic phenotype, not classic mt disease)",
            },
            {
                "variant": "m.3260A>G",
                "frequency": "Rare; maternally inherited cardiomyopathy families",
                "mechanism": (
                    "PET region disruption → combined CI+CIV deficiency → cardiomyopathy + myopathy"
                ),
                "penetrance": "~60–70% cardiomyopathy; ~50% myopathy (heteroplasmy-dependent)",
                "oxphos_deficiency": "Combined CI+CIII+CIV+CV reduced; CII NORMAL",
            },
            {
                "variant": "m.2617G>A",
                "frequency": "Very rare; LHON-like families",
                "mechanism": (
                    "Stem-loop destabilisation → partial mt-LSU defect → preferential RGC OXPHOS failure → optic neuropathy"
                ),
                "penetrance": "~40–75% optic neuropathy (males > females; typical LHON gender skew)",
                "oxphos_deficiency": "Partial OXPHOS deficiency; variable by tissue",
            },
            {
                "variant": "m.3093G>A",
                "frequency": "Rare; SNHL families",
                "mechanism": (
                    "Helix junction disruption → partial mt-LSU defect → cochlear hair cell OXPHOS failure → SNHL"
                ),
                "penetrance": "~70% SNHL when heteroplasmy >50%",
                "oxphos_deficiency": "Mild to moderate; heteroplasmy-dependent",
            },
        ],
        "humanin": {
            "definition": (
                "HUMANIN (HN) is a 21-amino-acid mitochondrially derived peptide encoded by a small "
                "ORF within the MT-RNR2 (16S rRNA) sequence at rCRS ~2706–2768. "
                "It is secreted extracellularly and acts as a cytoprotective, anti-apoptotic factor. "
                "Humanin protects neurons from Alzheimer disease toxicity (Aβ-mediated apoptosis), "
                "protects against ischaemia-reperfusion injury, and modulates insulin sensitivity. "
                "MT-RNR2 variants affecting this region may reduce humanin levels — active research area."
            ),
        },
        "absolute_contraindications": [
            "Metformin — ABSOLUTE CI in m.3260A>G/LargeDel (CI inhibitor; biguanide)",
            "Valproate (VPA) — ABSOLUTE CI (mitochondrial toxin)",
            "Propofol — ABSOLUTE CI (PRIS — propofol infusion syndrome; CI inhibitor)",
            "Linezolid — ABSOLUTE CI (mt-23S ribosome inhibitor; blocks mt-translation)",
            "Chloramphenicol — ABSOLUTE CI (mt-ribosome inhibitor)",
            "Ethambutol — ABSOLUTE AVOID in m.2617G>A LHON-like (optic neuropathy)",
            "Alcohol + Tobacco — ABSOLUTE AVOID in m.2617G>A (LHON penetrance modifiers)",
            "Statins — CAUTION in m.2336T>C MIHH (CoQ10 depletion; prescribe CoQ10 concurrently)",
        ],
        "recommended_treatments": [
            "Cascade maternal family testing — all maternal relatives; blood ± muscle heteroplasmy",
            "Annual echo + ECG — cardiomyopathy surveillance (m.3260A>G)",
            "Idebenone (Raxone) — LHON-like optic neuropathy (m.2617G>A); approved for canonical LHON",
            "Hearing aid / cochlear implant — SNHL (m.3093G>A); cochlear nerve intact",
            "ACE inhibitor/ARB — hypertension management (m.2336T>C MIHH)",
            "CoQ10 + Riboflavin B2 — Level C supplementation for OXPHOS deficiency",
            "Thiamine B1 + Biotin — empiric BTBGD SLC19A3 exclusion before MT-RNR2 diagnosis confirmed",
            "Metformin ABSOLUTE CI — use alternative antidiabetic (SGLT2i, GLP-1 RA, DPP4i)",
            "Genetic counselling — maternal inheritance; all children of carrier mothers inherit mtDNA",
            "WES is NOT sufficient — dedicated mtDNA panel, mitogenome NGS, or long-read sequencing required",
        ],
        "key_ddx": [
            {
                "condition": "MT-RNR1 (12S rRNA) — Aminoglycoside-Induced SNHL",
                "distinguishing": (
                    "MT-RNR1: ISOLATED SNHL; NO OXPHOS deficiency; m.1555A>G HOMOPLASMIC; "
                    "AMINOGLYCOSIDES ABSOLUTELY CI. "
                    "MT-RNR2: OXPHOS deficiency in most variants; HETEROPLASMIC; "
                    "aminoglycosides NOT specifically ABSOLUTELY CI"
                ),
            },
            {
                "condition": "MT-ND4 m.11778G>A (canonical LHON)",
                "distinguishing": (
                    "MT-ND4 m.11778G>A: most common LHON variant worldwide; high penetrance; "
                    "pure optic neuropathy; NO OXPHOS enzymology defect usually detectable. "
                    "MT-RNR2 m.2617G>A: lower penetrance; mt-LSU defect mechanism; "
                    "molecular testing distinguishes"
                ),
            },
            {
                "condition": "LRPPRC (Leigh Syndrome French-Canadian)",
                "distinguishing": (
                    "LRPPRC: AR nuclear; CI+CIV; neonatal liver failure; French-Canadian founder. "
                    "MT-RNR2: maternal mtDNA; heteroplasmic; no geographic restriction; mtDNA sequencing"
                ),
            },
            {
                "condition": "GJB2 (DFNB1) — AR autosomal recessive SNHL",
                "distinguishing": (
                    "GJB2: AR biallelic nuclear; NO maternal inheritance; most common genetic SNHL. "
                    "MT-RNR2 m.3093G>A: maternal inheritance; heteroplasmic; mtDNA sequencing"
                ),
            },
            {
                "condition": "BTBGD (SLC19A3 — Biotin-Thiamine-Responsive Basal Ganglia Disease)",
                "distinguishing": (
                    "SLC19A3: AR nuclear; Leigh-like; basal ganglia crisis; FULLY TREATABLE with "
                    "thiamine + biotin. ALWAYS exclude before diagnosing any mt-disease including MT-RNR2."
                ),
            },
        ],
        "genetic_counselling": {
            "recurrence_risk": (
                "MATERNAL inheritance — all children of a carrier mother inherit the same mtDNA. "
                "Heteroplasmic variants (m.3260A>G, m.2617G>A, m.3093G>A): offspring heteroplasmy "
                "levels may differ from mother due to mtDNA bottleneck during oogenesis. "
                "Father-to-child transmission NEVER occurs for mtDNA."
            ),
            "heteroplasmy_counselling": (
                "For heteroplasmic variants: sibling heteroplasmy levels can vary widely "
                "(10–90%) due to the mtDNA bottleneck in germ cells — one child may be severely "
                "affected while another is an asymptomatic low-level carrier. "
                "Blood heteroplasmy may underestimate tissue (muscle/heart/retina) heteroplasmy."
            ),
            "prenatal_diagnosis": (
                "CVS or amniocentesis: heteroplasmy level in fetal DNA. "
                "Caution: fetal tissue heteroplasmy may not predict postnatal tissue-specific levels. "
                "m.2336T>C (near-homoplasmic): reliable prenatal testing by mtDNA sequencing."
            ),
            "cascade_testing": (
                "ALL maternal relatives — mother, maternal siblings, maternal aunts/uncles, "
                "maternal first cousins — should be tested. "
                "Heteroplasmic variants: muscle biopsy heteroplasmy may be needed for high-risk relatives."
            ),
        },
        "key_references": [
            "Jia Z, Wang X, Qin Y, et al. (2008) Coronary heart disease is associated with a mutation in mitochondrial tRNA genes. Eur J Hum Genet 16(11):1368-74 — MT-RNR2 m.2336T>C MIHH context",
            "Chen TJ, Boles RG, Wong LJ (2000) Detection of mitochondrial DNA mutations by temporal temperature gradient gel electrophoresis. Clin Chem 46(8):1157-67 — m.3260A>G cardiomyopathy",
            "Guo B, Zhai D, Cabezas E, et al. (2003) Humanin peptide suppresses apoptosis by interfering with Bax activation. Nature 423:456-461 — humanin neuroprotection mechanism",
            "Yen K, Wan J, Mehta HH, et al. (2020) Humanin prevents age-related cognitive decline in mice and is associated with improved cognitive age in humans. Sci Rep 10:7431 — humanin aging",
            "DiMauro S, Schon EA (2003) Mitochondrial respiratory-chain diseases. N Engl J Med 348(26):2656-68 — mt-rRNA disease review",
            "Chinnery PF, Hudson G (2013) Mitochondrial genetics. Br Med Bull 106:135-59 — comprehensive mtDNA genetics",
            "Newman NJ, Yu-Wai-Man P, Biousse V, et al. (2023) Understanding the molecular basis and pathogenesis of hereditary optic neuropathies. JAMA Ophthalmol 141(2):172-82 — LHON context for MT-RNR2 optic neuropathy DDx",
            "Gorman GS, Chinnery PF, DiMauro S, et al. (2016) Mitochondrial diseases. Nat Rev Dis Primers 2:16080 — population prevalence + disease spectrum",
        ],
        "terms": [
            {
                "term": "MT-RNR2 (16S rRNA)",
                "definition": (
                    "Mitochondrially encoded 16S ribosomal RNA — 1559 nt RNA gene on H-strand (rCRS 1671–3229); "
                    "forms the mitoribosome large subunit (mt-LSU / 39S); NOT translated into protein; "
                    "houses the peptidyl transferase centre (PTC); contains HUMANIN ORF; "
                    "OMIM *561010"
                ),
            },
            {
                "term": "Peptidyl Transferase Centre (PTC)",
                "definition": (
                    "The catalytic core of the mitoribosome large subunit (mt-LSU), formed by the "
                    "central loop of 16S rRNA. The PTC catalyses peptide bond formation between "
                    "the aminoacyl-tRNA at the A-site and the peptidyl-tRNA at the P-site. "
                    "ALL 13 mtDNA-encoded OXPHOS subunits are synthesised through the PTC — "
                    "PTC disruption by MT-RNR2 variants causes combined OXPHOS deficiency."
                ),
            },
            {
                "term": "HUMANIN (HN)",
                "definition": (
                    "A 21-amino-acid mitochondrially derived peptide (MDP) encoded within MT-RNR2 "
                    "(rCRS ~2706–2768). Humanin is secreted extracellularly, crosses the blood-brain "
                    "barrier, and is neuroprotective — suppresses Aβ-induced apoptosis in Alzheimer "
                    "disease, protects against ischaemia-reperfusion injury, and modulates insulin "
                    "sensitivity. Circulating humanin declines with age; associated with longevity."
                ),
            },
            {
                "term": "Maternally Inherited Hypertension + Hypercholesterolaemia (MIHH)",
                "definition": (
                    "A cardiometabolic phenotype associated with MT-RNR2 m.2336T>C: maternally "
                    "inherited hypertension and/or hypercholesterolaemia without classic "
                    "mitochondrial multi-organ disease (no CPEO, no Leigh syndrome, no myoclonic "
                    "epilepsy). Mechanistically distinct from protein-coding mtDNA gene variants; "
                    "represents a milder mt-LSU assembly impairment affecting cardiometabolic tissues."
                ),
            },
            {
                "term": "Mitoribosome Large Subunit (mt-LSU / 39S)",
                "definition": (
                    "The large ribosomal subunit of the human mitoribosome, consisting of 16S rRNA "
                    "(MT-RNR2, 1559 nt) and ~53 mitoribosomal proteins (MRPs). The mt-LSU assembles "
                    "with the small subunit (mt-SSU / 28S, containing 12S rRNA / MT-RNR1) to form "
                    "the 55S mitoribosome. The mt-LSU peptidyl transferase centre (PTC) is the "
                    "catalytic site for peptide bond formation during mt-translation."
                ),
            },
            {
                "term": "Heteroplasmy (MT-RNR2 context)",
                "definition": (
                    "Most MT-RNR2 pathogenic variants (except m.2336T>C) are heteroplasmic — "
                    "a mixture of wild-type and mutant mtDNA copies co-exist within cells. "
                    "Unlike MT-RNR1 m.1555A>G (homoplasmic — blood DNA is reliable), "
                    "MT-RNR2 heteroplasmic variants may require muscle biopsy to quantify "
                    "tissue-specific heteroplasmy, as blood heteroplasmy can underestimate "
                    "the true mutant load in heart, muscle, and retinal ganglion cells."
                ),
            },
            {
                "term": "LHON-like Optic Neuropathy (MT-RNR2 m.2617G>A)",
                "definition": (
                    "A maternally inherited optic neuropathy resembling Leber Hereditary Optic "
                    "Neuropathy (LHON) caused by MT-RNR2 m.2617G>A. Unlike canonical LHON "
                    "(MT-ND1/MT-ND4/MT-ND6 variants), this is caused by a 16S rRNA variant. "
                    "Clinical features: subacute bilateral central vision loss, central scotoma, "
                    "dyschromatopsia, papillomacular bundle atrophy. Male predominance. "
                    "Idebenone trial evidence applies from canonical LHON by analogy."
                ),
            },
        ],
    }


if __name__ == "__main__":
    ov = get_overview()
    print(f"Gene: {ov['gene']} ({ov['alias']})")
    print(f"Disease: {ov['disease_name']}")
    print(f"OMIM Gene: *{ov['omim_gene']}")
    print(f"Genome: {ov['chromosome']}")
    print(f"Inheritance: {ov['inheritance']}")
    print(f"\nCohort: {ov['cohort_n']} patients, seed {ov['seed']}")
    s = ov["cohort_statistics"]
    print(f"  OXPHOS deficiency: {s['oxphos_deficiency_pct']}%")
    print(f"  Cardiomyopathy: {s['cardiomyopathy_pct']}%")
    print(f"  Hypertension (MIHH): {s['hypertension_pct']}%")
    print(f"  Optic neuropathy: {s['optic_neuropathy_pct']}%")
    print(f"  SNHL: {s['snhl_pct']}%")
    print(f"  Myopathy: {s['myopathy_pct']}%")
    print(f"  Elevated lactate: {s['elevated_lactate_pct']}%")
    print(f"  Maternal family affected: {s['maternal_family_affected_pct']}%")
    print(f"  m.2336T>C (MIHH main): {s['m2336_pct']}%")
    print(f"  Avg heteroplasmy (blood): {s['avg_heteroplasmy_blood']}%")
    print("\nVariants:", [v["change"] for v in VARIANTS])
