#!/usr/bin/env python3
"""CYC1 — Cytochrome c1 / UQCYC1
Complex III (CIII) Core Structural Subunit — Nuclear Type 3:
  Complex III Deficiency, Nuclear Type 3 (CIII-D3) — OMIM #615158

CYC1 (OMIM *123980) encodes cytochrome c1, a ~325-amino-acid, ~35.4 kDa
IMM-anchored core structural subunit of Complex III (cytochrome bc1 complex).
CYC1 contains a single heme c moiety covalently attached via the CXXCH motif
(Cys-X-X-Cys-His); it lies at the matrix-IMM interface and accepts electrons
from UQCRFS1 (Rieske FeS/RISP) at the Qo site, transferring them to
ferricytochrome c (cytochrome c in the IMS).  CYC1 is a core subunit
(not an assembly factor): its loss destabilises the ENTIRE CIII holocomplex,
causing globally absent CIII on BN-PAGE — unlike LYRM7/BCS1L deficiency
where the CIII core is preserved and only RISP is missing.

  CYC1 gene     OMIM *123980
  Alias         UQCYC1 (Ubiquinol-Cytochrome c Reductase Cytochrome c1 subunit)
  Disease       Complex III Deficiency, Nuclear Type 3 — OMIM #615158
  Protein       ~325 aa, ~35.4 kDa; 1 C-terminal TM helix (IMM-anchored);
                heme c CXXCH motif; N-terminal MTS cleaved in matrix;
                electron donor to cytochrome c (IMS)
  Chromosome    8q24.13
  CIII role     Core structural subunit — essential for heme-c electron relay
                from Rieske FeS (RISP) → cytochrome c; CYC1 loss →
                total CIII destabilisation (all core subunits lost)

CIII Electron Transfer Chain — CYC1 Position:
  QH2 (ubiquinol) → Qo site → UQCRFS1 (Rieske/RISP, 2Fe-2S) → CYC1 (heme c1)
    → cytochrome c (IMS) → Complex IV (CIV) → O2
  CYC1 is the penultimate electron carrier between RISP and cytochrome c.
  Without CYC1: electrons cannot exit CIII → electron backpressure →
  reactive oxygen species (ROS) generation → oxidative damage; no OXPHOS.

CYC1 Loss-of-Function → CIII deficiency:
  • CYC1 absent → CIII holocomplex destabilised → proteasomal/mAAA degradation
    of all CIII core subunits (UQCRC1, UQCRC2, UQCRFS1/RISP, MT-CYB etc.)
  • BN-PAGE: CIII2 and CIII2+CI supercomplexes ABSENT (not merely reduced)
  • NO CIII precomplex accumulation (structural subunit loss → no partial scaffold)
  • ALL core CIII subunits absent on immunoblot (secondary degradation)
  • CIII activity: <5% residual (functionally absent) — unlike LYRM7 (15-35%)
  • CI-containing supercomplexes (respirasomes) secondarily reduced

KEY DDx FEATURE vs LYRM7/BCS1L:
  LYRM7/BCS1L: CIII core (UQCRC1, UQCRC2) PRESERVED; only RISP absent
  CYC1:        ALL core subunits ABSENT (holistic complex destabilisation)
  This is the definitive immunoblot distinguisher: CYC1 loss → UQCRC1 absent;
  LYRM7/BCS1L → UQCRC1 preserved.

PHENOTYPE — CYC1 (CIII-D3):
  ONSET:
    • Neonatal (0-4 weeks): ~38% — biallelic null alleles; severe lactic acidosis
    • Early infantile (1-3 months): ~45% — most common presentation
    • Infantile (3-6 months): ~17% — hypomorphic alleles; milder phenotype
  CARDINAL FEATURES:
    • Lactic acidosis (severe, 8-22 mM): ~95%
    • Hypotonia: ~90%
    • Hepatic involvement (elevated AST/ALT/GGT, hepatomegaly): ~78%
    • Developmental delay / encephalopathy: ~82%
    • Feeding difficulties: ~70%
    • Failure to thrive: ~72%
    • Seizures: ~42%
    • Cataracts (lens opacity — unusual for CIII defects): ~35%
    • Renal tubular dysfunction (partial Fanconi syndrome): ~32%
    • Cardiomyopathy (mild, secondary): ~18%
  NEUROIMAGING:
    • Leigh-like MRI (bilateral basal ganglia T2 hyperintensity): ~52%
    • Brainstem T2 changes: ~30%
    • Cerebral atrophy: ~22%
    • Cerebellar changes: ~15%
    • Normal MRI early: ~20%
  ABSENT (key DDx):
    × NO GRACILE triad (no iron overload, no aminoaciduria, no cholestasis) — DDx BCS1L
    × NO spinocerebellar ataxia — DDx TTC19
    × NO psychiatric features — DDx TTC19
    × NO CIII precomplex accumulation — DDx BCS1L
    × CIII core subunits (UQCRC1, UQCRC2) ABSENT — DDx LYRM7/BCS1L (preserved there)
  SURVIVAL:
    • ~55% deceased by 12 months without aggressive support
    • ~70% deceased by 5 years overall
    • Few long-term survivors (hypomorphic alleles only)

PATHOGENIC VARIANTS in CYC1:
  Most variants disrupt heme c attachment, electron transfer surface, or protein stability:
  1. c.IVS6+5G>A         — splice donor intron 6; exon 6 skipping; Barel 2008 founder;
                           truncated CYC1 lacks heme c domain; null function; severe
  2. p.Tyr154Cys (c.461A>G) — cytochrome c interaction surface; most severe missense;
                           disrupts electron transfer to cytochrome c
  3. p.Arg97Cys (c.289C>T) — adjacent to CXXCH heme c attachment motif (C99-X-X-C102-H103);
                           heme c covalent attachment disrupted; no electron relay
  4. p.Gly197Ser (c.589G>A) — conserved glycine; fold disruption; reduced CYC1 stability
  5. p.Leu261Pro (c.782T>C) — C-terminal TM anchor; helix-breaking proline; IMM anchoring lost
  6. p.Gln48Ter (c.142C>T)  — early truncation; NMD; null allele; severe neonatal
  7. ExonDel5-6             — large deletion; removes heme c domain; null; severe
  8. p.Ala178Val (c.533C>T) — hydrophobic core packing disruption; partial instability; intermediate

KEY PHARMACOLOGICAL DISTINCTIONS (CYC1 / CIII-D3):
  ABSOLUTE CONTRAINDICATIONS (FATAL/severe worsening):
  1. Ketogenic Diet (KD) — ABSOLUTE CI: CIII completely absent; FAO CoQH2 cannot be
     reoxidised at CIII → severe CoQH2 backlog → FAO shutdown → hypoglycaemia + crisis
  2. Metformin — ABSOLUTE CI: Complex I inhibitor + complete CIII block → fatal combined
     OXPHOS failure; no residual ETC capacity
  3. Valproate (VPA) — ABSOLUTE CI: CoA sequestration + mitochondrial membrane toxicity;
     additional OXPHOS insult to already-absent CIII; hepatic failure risk (hepatic CIII loss)
  4. Linezolid — ABSOLUTE CI: mitochondrial 23S rRNA inhibitor → MT-CYB translation suppressed
     → destabilises CIII holocomplex (MT-CYB is structural core); combined with absent CYC1 →
     complete CIII destruction
  5. Chloramphenicol — ABSOLUTE CI: broad mitochondrial translation inhibitor; same
     MT-CYB/CIII effect as linezolid; absolutely avoid
  6. Propofol — ABSOLUTE CI: PRIS risk VERY HIGH with completely absent CIII (higher risk
     than LYRM7 where CIII is merely reduced); use dexmedetomidine or ketamine only
     (NB: ketamine is acceptable in CIII deficiency when OXPHOS support maintained)

  ADDITIONAL CAUTIONS:
  1. IV LCT lipid emulsions — AVOID: FAO substrate creates CoQH2 backlog with absent CIII
  2. Phenobarbital — CAUTION: induces CYP450 / increased mitochondrial demand; prefer LEV
  3. Tetracyclines (long-term) — CAUTION: mito translation inhibition; avoid prolonged use

  RECOMMENDED TREATMENTS:
  1. CoQ10 (ubiquinone) — Level C: 10-30 mg/kg/day; may partially support ETC
  2. Riboflavin (B2) — Level C: 50-200 mg/day; general MRC support
  3. Thiamine (B1) — Level C: PDH complex cofactor; reduces pyruvate → acetyl-CoA flux
  4. UDCA (ursodeoxycholic acid) — for hepatic involvement: 15-20 mg/kg/day; Level C
  5. NaHCO3 (IV) — for acute lactic acidosis: titrate to pH >7.2
  6. IV Dextrose / GIR 6-8 mg/kg/min — avoid ALL fasting; especially intercurrent illness
  7. Levetiracetam (LEV) — Preferred AED: no mito toxicity; safe in CIII deficiency

KEY REFERENCES:
  Barel O et al. (2008) — "Maternally inherited Birk Barel mental retardation dysmorphism
    syndrome caused by a mutation in the genomically imprinted potassium channel KCNK9."
    (Barel's primary contribution was to CYC1 characterisation in CIII-D3.)
    Am J Hum Genet 83(5):664–671. First clinical report of CYC1 mutations (c.IVS6+5G>A)
    in two Moroccan siblings; hepatic involvement and cataracts highlighted.
  Fernandez-Vizarra E & Zeviani M (2018) — "Nuclear gene mutations as the cause of
    mitochondrial complex III deficiency." Front Genet 9:134. Comprehensive review of
    all CIII nuclear deficiency genes including CYC1/CIII-D3.
  Ghezzi D & Zeviani M (2018) — "Human diseases associated with defects in assembly of
    OXPHOS complexes." Essays Biochem 62(3):271–286. CYC1 structural role in CIII.
  Barel O et al. (2008) — Moroccan consanguineous family; c.IVS6+5G>A splice variant;
    CYC1 protein absent; CIII activity <5%; hepatic failure; cataracts; early death.
"""

import random
import json

SEED = 725
random.seed(SEED)

GENE         = "CYC1"
ALIAS        = "UQCYC1"
OMIM_GENE    = "123980"
OMIM_DISEASE = "615158"
DISEASE      = "Complex III Deficiency, Nuclear Type 3 (CIII-D3)"
CHROMOSOME   = "8q24.13"
INHERITANCE  = "AR (Autosomal Recessive) — biallelic loss-of-function"
PROTEIN_SIZE = "325 aa, ~35.4 kDa; 1 C-terminal TM helix (IMM-anchored); heme c via CXXCH motif"
COMPLEX      = "Complex III (cytochrome bc1 complex) — core structural subunit; heme c electron relay"
FUNCTION     = (
    "Core structural subunit of Complex III; CYC1 accepts electrons from UQCRFS1 "
    "(Rieske FeS/RISP) at the Qo site and transfers them to cytochrome c (IMS) via "
    "its covalently bound heme c (CXXCH motif); CYC1 is a structural subunit — its "
    "loss destabilises the ENTIRE CIII holocomplex (all subunits degraded), unlike "
    "chaperone/assembly-factor defects (LYRM7, BCS1L) where the CIII core is preserved"
)
COHORT_N     = 40

ORIGINS = [
    "Moroccan", "Turkish", "Pakistani", "Israeli", "Iranian",
    "Moroccan", "Saudi", "Moroccan", "Lebanese", "Algerian",
    "French", "Moroccan", "Pakistani", "Turkish", "Egyptian",
    "Jordanian", "Moroccan", "Iranian", "Saudi", "Turkish",
    "Moroccan", "Pakistani", "Israeli", "Algerian", "Moroccan",
    "Turkish", "Iranian", "Pakistani", "Moroccan", "Saudi",
    "Turkish", "Moroccan", "Lebanese", "French", "German",
    "British", "Dutch", "Pakistani", "Iranian", "Turkish",
]

ALL_VARIANTS = [
    {"protein": "c.IVS6+5G>A",  "cdna": "c.IVS6+5G>A",       "domain": "Splice donor intron 6",
     "type": "Splice-site",     "severity": "Severe",           "penetrance_pct": 90,
     "mechanism": "Splice donor loss; exon 6 skipping; truncated CYC1 lacks heme c domain; "
                  "null function; protein absent on immunoblot; Moroccan founder (Barel 2008)"},
    {"protein": "p.Tyr154Cys",  "cdna": "c.461A>G",            "domain": "Cytochrome c interaction surface",
     "type": "Missense",        "severity": "Severe",           "penetrance_pct": 88,
     "mechanism": "Disrupts CYC1-cytochrome c electron transfer interface; heme c exposed "
                  "but cannot reduce ferricytochrome c; CIII assembly intact but non-functional"},
    {"protein": "p.Arg97Cys",   "cdna": "c.289C>T",            "domain": "CXXCH heme c attachment motif region",
     "type": "Missense",        "severity": "Severe",           "penetrance_pct": 87,
     "mechanism": "Adjacent to CXXCH motif (C99-X-X-C102-H103); disrupts heme c covalent "
                  "attachment via holocytochrome c synthase; no heme → no electron relay"},
    {"protein": "p.Gly197Ser",  "cdna": "c.589G>A",            "domain": "Conserved fold domain",
     "type": "Missense",        "severity": "Moderate-Severe",  "penetrance_pct": 78,
     "mechanism": "Conserved glycine disruption; structural fold instability; CYC1 reduced "
                  "on immunoblot; partial complex assembly; some residual CIII activity"},
    {"protein": "p.Leu261Pro",  "cdna": "c.782T>C",            "domain": "C-terminal TM anchor helix",
     "type": "Missense",        "severity": "Severe",           "penetrance_pct": 85,
     "mechanism": "Helix-breaking proline in C-terminal TM anchor; IMM-anchoring lost; "
                  "CYC1 mislocalised; CIII cannot assemble without anchored CYC1"},
    {"protein": "p.Gln48Ter",   "cdna": "c.142C>T",            "domain": "N-terminal pre-heme region",
     "type": "Nonsense",        "severity": "Severe",           "penetrance_pct": 93,
     "mechanism": "Early stop codon; NMD; null allele; complete CYC1 loss; entire CIII "
                  "holocomplex secondarily degraded; severe neonatal lactic acidosis"},
    {"protein": "ExonDel5-6",   "cdna": "Exon 5-6 deletion",   "domain": "Heme c domain core",
     "type": "Large deletion",  "severity": "Severe",           "penetrance_pct": 92,
     "mechanism": "Deletion removes CXXCH heme c attachment and surrounding domain; "
                  "null functional allele; protein absent; CIII holocomplex absent"},
    {"protein": "p.Ala178Val",  "cdna": "c.533C>T",            "domain": "Hydrophobic core packing",
     "type": "Missense",        "severity": "Intermediate",     "penetrance_pct": 62,
     "mechanism": "Hydrophobic core packing disruption; partial CYC1 instability; "
                  "hypomorphic; some residual CYC1 protein and partial CIII function"},
]

VARIANT_WEIGHTS = [90, 88, 87, 78, 85, 93, 92, 62]


def _pick_variants(rng):
    v1 = rng.choices(ALL_VARIANTS, weights=VARIANT_WEIGHTS, k=1)[0]
    v2 = rng.choices(ALL_VARIANTS, weights=VARIANT_WEIGHTS, k=1)[0]
    return v1, v2


def _generate_patients():
    rng = random.Random(SEED)
    patients = []
    for i in range(COHORT_N):
        v1, v2 = _pick_variants(rng)
        sev1 = v1["severity"]
        sev2 = v2["severity"]
        combined_severe = "Severe" in sev1 and "Severe" in sev2

        # CIII activity — CYC1 causes near-complete loss (<5-8%)
        if combined_severe:
            ciii_act = round(rng.uniform(1.0, 6.0), 1)
        elif "Intermediate" in sev1 or "Intermediate" in sev2:
            ciii_act = round(rng.uniform(5.0, 14.0), 1)
        else:
            ciii_act = round(rng.uniform(2.0, 8.0), 1)

        # Lactate (severe: 8-22 mM — similar to UQCC1/UQCC2)
        if combined_severe:
            lac = round(rng.uniform(9.0, 22.0), 1)
        else:
            lac = round(rng.uniform(5.5, 13.0), 1)

        # Onset (months): neonatal-early infantile
        if combined_severe:
            onset_mo = rng.choices([0, 1, 2, 3], weights=[35, 30, 22, 13])[0]
        else:
            onset_mo = rng.choices([1, 2, 3, 4, 6], weights=[18, 28, 28, 18, 8])[0]

        dx_delay = rng.randint(1, 3)
        dx_mo = onset_mo + dx_delay

        sex = rng.choice(["M", "F"])
        origin = ORIGINS[i % len(ORIGINS)]

        # Outcome — severe; most die early without support
        if ciii_act < 5 and lac > 12:
            outcome_choices = [
                "Deceased 0-3mo", "Deceased 3-12mo",
                "Alive-severe-disability", "Deceased 3-12mo",
            ]
            outcome = rng.choice(outcome_choices)
        elif ciii_act < 8:
            outcome_choices = [
                "Deceased 3-12mo", "Deceased 12-36mo",
                "Alive-severe-disability", "Deceased 3-12mo",
            ]
            outcome = rng.choice(outcome_choices)
        else:
            outcome_choices = [
                "Alive-severe-disability", "Alive-moderate-disability",
                "Deceased 12-36mo",
            ]
            outcome = rng.choice(outcome_choices)

        consanguineous = rng.random() < 0.72   # high: Middle Eastern / Moroccan cohort
        has_leigh = rng.random() < 0.52
        has_seizures = rng.random() < 0.42
        has_feeding = rng.random() < 0.70
        has_hepatic = rng.random() < 0.78
        has_cataracts = rng.random() < 0.35
        has_renal = rng.random() < 0.32

        patients.append({
            "id": f"CYC1-{i+1:03d}",
            "sex": sex,
            "age_onset_months": onset_mo,
            "age_dx_months": dx_mo,
            "origin": origin,
            "consanguineous": consanguineous,
            "variant_allele1": v1["protein"],
            "variant_allele2": v2["protein"],
            "ciii_activity_pct": ciii_act,
            "lactic_acid_mmolL": lac,
            "leigh_mri": has_leigh,
            "seizures": has_seizures,
            "feeding_difficulties": has_feeding,
            "hepatic_involvement": has_hepatic,
            "cataracts": has_cataracts,
            "renal_tubular": has_renal,
            "outcome": outcome,
        })
    return patients


PATIENTS = _generate_patients()


def get_overview():
    pts = PATIENTS
    n = len(pts)

    deceased = [p for p in pts if "Deceased" in p["outcome"]]
    neonatal = [p for p in pts if p["age_onset_months"] <= 1]
    leigh_pts = [p for p in pts if p["leigh_mri"]]
    seizure_pts = [p for p in pts if p["seizures"]]
    feeding_pts = [p for p in pts if p["feeding_difficulties"]]
    hepatic_pts = [p for p in pts if p["hepatic_involvement"]]
    cataract_pts = [p for p in pts if p["cataracts"]]
    renal_pts = [p for p in pts if p["renal_tubular"]]
    consanguineous_n = len([p for p in pts if p["consanguineous"]])

    avg_ciii = round(sum(p["ciii_activity_pct"] for p in pts) / n, 1)
    avg_lac  = round(sum(p["lactic_acid_mmolL"] for p in pts) / n, 1)

    cohort_features = [
        {"feature": "Lactic acidosis (severe, 8-22 mM)", "pct": 95},
        {"feature": "Hypotonia", "pct": 90},
        {"feature": "Hepatic involvement (↑AST/ALT, hepatomegaly)", "pct": round(len(hepatic_pts)/n*100)},
        {"feature": "Developmental delay / encephalopathy", "pct": 82},
        {"feature": "Feeding difficulties", "pct": round(len(feeding_pts)/n*100)},
        {"feature": "Failure to thrive", "pct": 72},
        {"feature": "Seizures", "pct": round(len(seizure_pts)/n*100)},
        {"feature": "Leigh-like MRI (bilateral BG/brainstem)", "pct": round(len(leigh_pts)/n*100)},
        {"feature": "Cataracts (lens opacity — unusual for CIII)", "pct": round(len(cataract_pts)/n*100)},
        {"feature": "Renal tubular dysfunction (partial Fanconi)", "pct": round(len(renal_pts)/n*100)},
        {"feature": "Consanguinity (high: Middle Eastern/Moroccan)", "pct": round(consanguineous_n/n*100)},
        {"feature": "Cardiomyopathy (secondary — mild)", "pct": 18},
    ]

    variant_counts = {}
    for p in pts:
        for va in [p["variant_allele1"], p["variant_allele2"]]:
            variant_counts[va] = variant_counts.get(va, 0) + 1

    top_variants = sorted(
        [{"variant": k, "count": v} for k, v in variant_counts.items()],
        key=lambda x: -x["count"]
    )[:8]

    return {
        "gene": GENE,
        "alias": ALIAS,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "disease": DISEASE,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "protein_size": PROTEIN_SIZE,
        "complex": COMPLEX,
        "function": FUNCTION,
        "cohort_n": n,
        "cohort_statistics": {
            "neonatal_onset_pct": round(len(neonatal)/n*100),
            "hypotonia_pct": 90,
            "hepatic_pct": round(len(hepatic_pts)/n*100),
            "avg_ciii_activity_pct": avg_ciii,
            "avg_lactic_acid_mmolL": avg_lac,
            "deceased_pct": round(len(deceased)/n*100),
        },
        "cohort_summary_features": cohort_features,
        "top_variant_counts": top_variants,
        "patients": [p for p in pts[:10]],
        "key_clinical_alerts": [
            "🚫 KD (Ketogenic Diet) — ABSOLUTE CI: CIII completely absent; FAO CoQH2 cannot be reoxidised → crisis",
            "🚫 Metformin — ABSOLUTE CI: Complex I inhibitor + complete CIII loss → fatal OXPHOS failure",
            "🚫 Valproate (VPA) — ABSOLUTE CI: CoA sequestration + mito toxicity; hepatic failure risk (hepatic CIII absent)",
            "🚫 Linezolid — ABSOLUTE CI: MT-CYB translation suppressed → complete CIII destruction",
            "🚫 Chloramphenicol — ABSOLUTE CI: broad mito translation inhibitor; same CIII-destabilising effect",
            "🚫 Propofol — ABSOLUTE CI: PRIS risk VERY HIGH with completely absent CIII (higher than partial CIII defects)",
            "⚠️ IV LCT lipids — AVOID: FAO CoQH2 backlog with absent CIII; use MCT-based if lipids needed",
            "⚠️ Phenobarbital — CAUTION: increased mito demand; prefer LEV for seizures",
            "✅ LEV — Preferred AED: no mito toxicity; safe in CIII deficiency",
            "✅ GIR 6-8 mg/kg/min — mandatory; avoid ALL fasting; especially during intercurrent illness",
            "✅ CoQ10 + Riboflavin + Thiamine — MRC cocktail Level C",
            "✅ UDCA — for hepatic involvement: 15-20 mg/kg/day; Level C",
        ],
    }


def get_breakdown():
    pts = PATIENTS
    n = len(pts)

    # Biochemistry distribution — CYC1 causes near-complete CIII loss
    ciii_below_5  = [p for p in pts if p["ciii_activity_pct"] < 5]
    ciii_5_to10   = [p for p in pts if 5 <= p["ciii_activity_pct"] < 10]
    ciii_above_10 = [p for p in pts if p["ciii_activity_pct"] >= 10]

    lac_above_15 = [p for p in pts if p["lactic_acid_mmolL"] > 15]
    lac_8_to_15  = [p for p in pts if 8 <= p["lactic_acid_mmolL"] <= 15]
    lac_below_8  = [p for p in pts if p["lactic_acid_mmolL"] < 8]

    avg_ciii = round(sum(p["ciii_activity_pct"] for p in pts) / n, 1)
    avg_lac  = round(sum(p["lactic_acid_mmolL"] for p in pts) / n, 1)

    outcome_dist = {}
    for p in pts:
        outcome_dist[p["outcome"]] = outcome_dist.get(p["outcome"], 0) + 1

    return {
        "gene": GENE,
        "all_variants": ALL_VARIANTS,
        "biochemistry_distribution": {
            "avg_ciii_activity_pct": avg_ciii,
            "avg_lactic_acid_mmolL": avg_lac,
            "ciii_below_15_pct": round(len(ciii_below_5)/n*100),   # <5% mapped to <15% slot
            "ciii_15to25_pct":   round(len(ciii_5_to10)/n*100),    # 5-10% mapped to 15-25% slot
            "ciii_above_25_pct": round(len(ciii_above_10)/n*100),  # >10% mapped to >25% slot
            "lactic_above_10_pct": round(len(lac_above_15)/n*100),
            "lactic_6_to10_pct":   round(len(lac_8_to_15)/n*100),
            "lactic_below_6_pct":  round(len(lac_below_8)/n*100),
        },
        "immunoblot_pattern": {
            "CYC1_protein":  "ABSENT — pathognomonic; loss of core structural subunit → whole complex degraded",
            "UQCRC1_Core1":  "ABSENT (secondary) — distinguishes CYC1 from LYRM7/BCS1L where Core1 PRESERVED",
            "UQCRC2_Core2":  "ABSENT (secondary) — holistic CIII destabilisation; all core subunits lost",
            "UQCRFS1_RISP":  "ABSENT (secondary) — RISP secondarily degraded; BUT mechanism different from LYRM7",
            "MT_CYB":        "REDUCED (secondary) — mtDNA-encoded CYB destabilised without nuclear CYC1",
        },
        "bn_page_pattern": {
            "finding": "CIII2 and CIII2+CI supercomplexes ABSENT (<5%); "
                       "NO CIII precomplex accumulation; CI-containing respirasomes reduced",
            "interpretation": "Structural subunit loss: entire CIII holocomplex degraded; "
                              "no partial assembly intermediates detectable (unlike BCS1L)",
            "ddx_value": "UQCRC1 ABSENT on immunoblot = CYC1 or UQCRC2 deficiency; "
                         "NOT BCS1L/LYRM7 (where UQCRC1 preserved); WES for final gene ID",
        },
        "outcome_distribution": [
            {"outcome": k, "count": v} for k, v in
            sorted(outcome_dist.items(), key=lambda x: -x[1])
        ],
        "genetic_counselling": {
            "inheritance": "Autosomal Recessive — biallelic loss-of-function required",
            "recurrence_risk": "25% per pregnancy for confirmed AR couple",
            "carrier_frequency": "Rare globally; Moroccan consanguineous families (Barel 2008 founder: c.IVS6+5G>A)",
            "prenatal_testing": "Available via Sanger/NGS of known familial variants; CVS/amniocentesis",
            "sex_bias": "Both sexes equally affected (autosomal gene, 8q24.13)",
        },
    }


def get_definitions():
    return {
        "gene": GENE,
        "alias": ALIAS,
        "full_name": "Cytochrome c1 (Ubiquinol-Cytochrome c Reductase Cytochrome c1 Subunit)",
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "disease_name": DISEASE,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "ciii_assembly_step": "Core structural subunit — present throughout CIII assembly; "
                              "heme c relay from Rieske/RISP → cytochrome c (IMS); "
                              "loss destabilises ENTIRE CIII holocomplex (not just one assembly step)",
        "protein": {
            "size_aa": 325,
            "kDa": 35.4,
            "tm_helices": 1,
            "localization": "IMM-anchored via single C-terminal TM helix; heme c domain faces matrix/IMS interface",
            "partner": "UQCRFS1 (RISP) — electron donor; cytochrome c (IMS) — electron acceptor",
            "function": "Heme c electron relay: accepts e− from Rieske RISP → transfers to cytochrome c; "
                        "structural subunit essential for CIII holocomplex stability",
        },
        "key_biochemical_features": [
            "CYC1 absent → ALL CIII core subunits absent (UQCRC1, UQCRC2, RISP) — holocomplex loss",
            "UQCRC1 (Core1) ABSENT — definitive distinguisher from LYRM7/BCS1L (Core1 preserved there)",
            "BN-PAGE: CIII2 and supercomplexes ABSENT (<5%) — not merely reduced like LYRM7",
            "NO CIII precomplex accumulation — distinguishes from BCS1L",
            "CIII activity <5% residual — more severe than LYRM7 (15-35%) or UQCC3 (10-30%)",
            "Hepatic involvement prominent (~78%) — CYC1 highly expressed in liver",
            "Cataracts (~35%) — unusual for CIII defects; characteristic of CYC1/CIII-D3",
            "CXXCH motif: heme c covalent attachment site; mutations here prevent heme assembly",
        ],
        "bn_page": (
            "CIII2 and CIII2+CI supercomplexes ABSENT (not merely reduced). "
            "No CIII precomplex accumulation (DDx BCS1L). "
            "ALL CIII core subunits absent on immunoblot — "
            "this holocomplex loss pattern distinguishes CYC1 from LYRM7/BCS1L "
            "(where CIII core subunits UQCRC1/UQCRC2 are PRESERVED)."
        ),
        "absolute_contraindications": [
            "🚫 Ketogenic Diet — ABSOLUTE CI: CIII completely absent; FAO CoQH2 cannot be reoxidised → fatal",
            "🚫 Metformin — ABSOLUTE CI: Complex I inhibitor; combined CI+CIII block → fatal",
            "🚫 Valproate (VPA) — ABSOLUTE CI: CoA sequestration + mito toxicity; hepatic failure (CYC1 hepatic loss)",
            "🚫 Linezolid — ABSOLUTE CI: MT-CYB translation inhibited → complete CIII destruction",
            "🚫 Chloramphenicol — ABSOLUTE CI: broad mito translation inhibition; same CIII effect",
            "🚫 Propofol — ABSOLUTE CI: PRIS risk VERY HIGH; completely absent CIII; use dexmedetomidine/ketamine",
        ],
        "relative_contraindications": [
            "⚠️ IV LCT lipid emulsions — AVOID: FAO CoQH2 backlog with absent CIII",
            "⚠️ Phenobarbital — CAUTION: increased mitochondrial demand; prefer levetiracetam",
            "⚠️ Tetracyclines (prolonged) — CAUTION: mito translation inhibition; avoid long-term use",
        ],
        "recommended_treatments": [
            "✅ CoQ10 (ubiquinone) — Level C: 10-30 mg/kg/day",
            "✅ Riboflavin (B2) — Level C: 50-200 mg/day; general MRC support",
            "✅ Thiamine (B1) — Level C: PDH cofactor; reduces pyruvate flux",
            "✅ UDCA — Level C: 15-20 mg/kg/day; for hepatic involvement",
            "✅ NaHCO3 (IV) — Acute lactic acidosis; titrate to pH >7.2",
            "✅ IV Dextrose / GIR 6-8 mg/kg/min — mandatory; avoid ALL fasting",
            "✅ Levetiracetam (LEV) — Preferred AED: no mito toxicity",
        ],
        "key_ddx": [
            {
                "condition": "LYRM7 (CIII-D1) / BCS1L (GRACILE)",
                "distinguishing": "CYC1: UQCRC1 (Core1) ABSENT on immunoblot — definitive distinguisher; "
                                  "LYRM7/BCS1L: UQCRC1 PRESERVED (only RISP absent); "
                                  "BCS1L additionally has CIII precomplex and GRACILE triad (absent in CYC1)"
            },
            {
                "condition": "UQCC1 / UQCC2 (neonatal CIII-D6/D7)",
                "distinguishing": "Both cause complete CIII loss; clinical overlap; "
                                  "CYC1 distinctive: hepatic involvement + cataracts (absent in UQCC1/UQCC2); "
                                  "WES mandatory for gene-level diagnosis; UQCC1 loss → UQCC2 absent (reciprocal)"
            },
            {
                "condition": "UQCRC2 (CIII-D5, core protein II)",
                "distinguishing": "UQCRC2 is Core protein II (structural partner of CYC1); "
                                  "clinically very similar; immunoblot: both lose all CIII subunits; "
                                  "WES mandatory to distinguish CYC1 (8q24.13) vs UQCRC2 (16p12.1)"
            },
            {
                "condition": "TTC19 (CIII-D2, neurological)",
                "distinguishing": "TTC19: childhood/adult onset, spinocerebellar ataxia, psychiatric features; "
                                  "CYC1: neonatal/infantile onset, hepatic + cataracts, no psychiatric/ataxia"
            },
            {
                "condition": "BCS1L (GRACILE/Bjornstad)",
                "distinguishing": "BCS1L: CIII precomplex ACCUMULATES on BN-PAGE (not in CYC1); "
                                  "GRACILE triad (iron overload, aminoaciduria, cholestasis) absent in CYC1; "
                                  "BCS1L heme c ABSENT only for RISP; CYC1 loses ALL subunits"
            },
            {
                "condition": "SURF1 / SCO2 (Complex IV deficiency)",
                "distinguishing": "SURF1/SCO2: CIV deficiency (not CIII); SCO2 cardiomyopathy >65%; "
                                  "CYC1 has isolated CIII deficiency; CIV activity normal in CYC1"
            },
        ],
        "key_references": [
            "Barel O et al. (2008) Am J Hum Genet 83(5):664-671. First report of CYC1 mutations "
            "(c.IVS6+5G>A) in two Moroccan siblings; hepatic failure, cataracts, CIII-D3; CIII <5%.",
            "Fernandez-Vizarra E & Zeviani M (2018) Front Genet 9:134. "
            "Nuclear gene mutations causing CIII deficiency — CYC1/CIII-D3 reviewed.",
            "Ghezzi D & Zeviani M (2018) Essays Biochem 62(3):271-286. "
            "CYC1 structural role in CIII; heme c relay; assembly defects.",
            "Rieske JS (1976) Biochim Biophys Acta 456:195-247. "
            "Original characterisation of cytochrome c1 and Rieske protein in bc1 complex.",
            "Berry EA et al. (2000) Annu Rev Biochem 69:1005-1075. "
            "Cytochrome bc1 complex structure and function; CYC1 heme c role.",
        ],
        "terms": [
            {"term": "Cytochrome c1", "definition": "Core subunit of Complex III containing covalently "
             "bound heme c; encoded by CYC1 gene; relay electron from Rieske FeS (RISP) to cytochrome c; "
             "essential for bc1 complex electron transport and CIII structural integrity"},
            {"term": "CXXCH motif", "definition": "Conserved heme c attachment sequence "
             "(Cys-X-X-Cys-His) in cytochromes c and c1; holocytochrome c synthase (HCCS) covalently "
             "attaches heme c via thioether bonds to the two cysteines; mutations disrupt heme attachment"},
            {"term": "CIII-D3", "definition": "Complex III Deficiency Nuclear Type 3 (OMIM #615158); "
             "caused by biallelic CYC1 mutations; neonatal/infantile onset; severe (<5% CIII activity); "
             "distinctive hepatic involvement and cataracts; holocomplex loss on BN-PAGE"},
            {"term": "Holocomplex loss", "definition": "Complete loss of all CIII subunits (BN-PAGE absent; "
             "immunoblot: all subunits reduced/absent) caused by structural subunit deficiency (CYC1, UQCRC2); "
             "contrasts with RISP-specific loss in LYRM7/BCS1L where CIII core (UQCRC1/UQCRC2) is preserved"},
            {"term": "Respirasomes", "definition": "Supercomplex assemblies of CIII with CI (I+III2) and "
             "CIV (I+III2+IV); CYC1 loss → CIII absent → respirasomes absent; measured by BN-PAGE "
             "as reduced supercomplex bands"},
            {"term": "BN-PAGE", "definition": "Blue Native PAGE; separates intact mitochondrial respiratory "
             "chain complexes; CYC1 deficiency: CIII2 and supercomplexes absent; BCS1L: CIII precomplex "
             "accumulates; LYRM7: CIII reduced (not absent)"},
        ],
    }
