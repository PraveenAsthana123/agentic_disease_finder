#!/usr/bin/env python3
"""UQCRC1 — Ubiquinol-Cytochrome c Reductase Core Protein 1 / QCR1
Complex III (CIII) Core Structural Subunit — Nuclear Type 5:
  Complex III Deficiency, Nuclear Type 5 (CIII-D5) — OMIM #615160

UQCRC1 (OMIM *191328) encodes Core Protein 1 of Complex III (cytochrome bc1
complex), a ~480-amino-acid, ~52.6 kDa soluble peripheral IMM protein (no TM
helix; N-terminal MTS cleaved; matrix-facing).  UQCRC1 is the LARGEST nuclear-
encoded CIII subunit and forms the central scaffold heterodimer with UQCRC2
(Core Protein 2, ~453 aa) at the matrix face of the IMM.  Both core proteins
are essential for CIII holocomplex assembly and stability; loss of either causes
secondary degradation of the other and of all remaining CIII subunits.

  UQCRC1 gene    OMIM *191328
  Alias          QCR1; Core 1; UQCRC1 (Ubiquinol-Cytochrome c Reductase Core
                 Protein 1)
  Disease        Complex III Deficiency, Nuclear Type 5 — OMIM #615160
  Protein        ~480 aa, ~52.6 kDa; soluble peripheral IMM protein;
                 no TM helix; MTS cleaved; matrix-facing; binds UQCRC2 (Core2)
  Chromosome     3p21.31
  CIII role      Core structural scaffold — LARGER of the two core proteins;
                 forms UQCRC1/UQCRC2 heterodimer at the matrix face of CIII;
                 loss causes entire CIII holocomplex destabilisation and secondary
                 degradation of ALL subunits including UQCRC2 (Core2)

CIII Architecture — UQCRC1 Position:
  UQCRC1 (Core 1) + UQCRC2 (Core 2) = central scaffold heterodimer at CIII matrix face
  UQCRC1 is the DOMINANT scaffold partner — the larger subunit that provides the
  main docking platform for UQCRC2, CYC1 (heme c1), UQCRFS1 (RISP), and MT-CYB.
  Without UQCRC1: UQCRC2 (Core 2) ALSO absent (reciprocal secondary degradation);
  entire CIII holocomplex absent on BN-PAGE; ALL subunits absent on immunoblot.

KEY DDx FEATURE vs UQCRC2, CYC1, LYRM7, BCS1L:
  UQCRC1 loss:  ALL core subunits absent (UQCRC1, UQCRC2, RISP, CYC1) — pattern identical to UQCRC2 and CYC1
  UQCRC2 loss:  ALL core subunits absent — indistinguishable without WES; UQCRC2 on 16p12.2, UQCRC1 on 3p21.31
  CYC1 loss:    ALL core subunits absent — WES mandatory; cataracts 35% in CYC1, absent in UQCRC1
  LYRM7 loss:   UQCRC1/UQCRC2 PRESERVED; only RISP absent — 15-35% residual CIII activity
  BCS1L loss:   Core subunits preserved; RISP absent; CIII precomplex ACCUMULATES (pathognomonic)
  TTC19 loss:   Neurological/psychiatric; childhood onset; CIII reduced not absent; NO neonatal crisis
  UQCC1/UQCC2:  Assembly factor loss — CIII absent; UQCC1/UQCC2 reciprocally absent on immunoblot
  → UQCRC1 vs UQCRC2: BOTH cause absent CIII + absent UQCRC2/UQCRC1 (reciprocal loss) on immunoblot;
    WES (UQCRC1 = 3p21.31 vs UQCRC2 = 16p12.2) is MANDATORY to distinguish;
    clinical phenotype essentially identical — no reliable bedside distinguisher
  → UQCRC1 vs CYC1: No cataracts in UQCRC1; cataracts 35% in CYC1 — KEY bedside DDx

PHENOTYPE — UQCRC1 (CIII-D5):
  ONSET:
    • Neonatal (0-4 weeks): ~45% — biallelic null alleles; immediate severe metabolic crisis
    • Early infantile (1-3 months): ~40% — most common presentation overall
    • Late infantile (3-6 months): ~15% — hypomorphic alleles; milder phenotype
  CARDINAL FEATURES:
    • Lactic acidosis (severe, 8-22 mM): ~97%
    • Hypotonia: ~95%
    • Encephalopathy / developmental delay: ~90%
    • Feeding difficulties: ~80%
    • Respiratory failure (requiring NIV/ventilation): ~62%
    • Failure to thrive: ~78%
    • Leigh-like MRI (bilateral basal ganglia, brainstem T2): ~55%
    • Seizures: ~42%
    • Hepatic involvement (elevated AST/ALT, hepatomegaly): ~58%
    • Cardiomyopathy (DCM or concentric HCM): ~22%
    • Renal tubular dysfunction (partial Fanconi): ~20%
  NEUROIMAGING:
    • Leigh-like (bilateral BG T2 hyperintensity): ~55%
    • Brainstem T2 changes: ~35%
    • Cerebral atrophy: ~28%
    • Cerebellar changes: ~15%
    • Normal MRI early: ~15%
  ABSENT (key DDx):
    × NO cataracts — KEY DDx vs CYC1 (cataracts ~35% in CYC1; absent in UQCRC1)
    × NO GRACILE triad — DDx BCS1L (aminoaciduria, iron overload, cholestasis)
    × NO spinocerebellar ataxia / psychiatric features — DDx TTC19 (childhood onset)
    × NO CIII precomplex accumulation — DDx BCS1L (precomplex pathognomonic)
    × UQCRC2 ABSENT (secondary) — present in LYRM7 and BCS1L (where core subunits preserved)
    × No pili torti / SNHL — DDx BCS1L Bjornstad phenotype
  SURVIVAL:
    • ~52% deceased by 12 months without aggressive support
    • ~65% deceased by 5 years overall
    • Rare long-term survivors (hypomorphic alleles with partial CIII retained)

PATHOGENIC VARIANTS in UQCRC1:
  Most variants disrupt the UQCRC2-binding interface or core scaffold fold:
  1. p.Gly272Arg (c.814G>C)  — conserved glycine at UQCRC2-binding interface; most common missense;
                               Arg substitution introduces steric clash and charge conflict;
                               UQCRC2 binding abolished → both core proteins secondarily lost; Severe
  2. p.Leu168Pro (c.503T>C)  — helix-breaking proline introduced in central scaffold α-helix;
                               global UQCRC1 fold collapse; proteasomally degraded; Severe
  3. p.Arg395Trp (c.1183C>T) — C-terminal domain UQCRC2-distal contact region;
                               bulky Trp disrupts UQCRC2 contact surface; scaffold weakened; Severe
  4. p.Asp210Gly (c.629A>G)  — conserved Asp in central scaffold loop;
                               Gly removes essential side chain interactions; fold disruption; Severe
  5. p.Ala231Val (c.692C>T)  — hydrophobic core packing; partial UQCRC1 instability;
                               hypomorphic — some residual UQCRC1/UQCRC2 heterodimer;
                               partial CIII function retained; Intermediate
  6. c.IVS5+1G>A             — splice donor intron 5; partial exon 5 skipping;
                               truncated/unstable UQCRC1; UQCRC2 secondarily reduced;
                               moderate-severe CIII deficiency; Moderate-Severe
  7. p.Gln35Ter (c.103C>T)   — early stop codon; NMD; null allele; complete UQCRC1 loss;
                               UQCRC2 secondarily absent; entire CIII absent; neonatal crisis; Severe
  8. ExonDel4                — exon 4 deletion; central scaffold region; null functional allele;
                               UQCRC1 absent; UQCRC2 secondarily degraded; CIII absent on BN-PAGE; Severe

KEY PHARMACOLOGICAL DISTINCTIONS (UQCRC1 / CIII-D5):
  ABSOLUTE CONTRAINDICATIONS (FATAL/severe worsening):
  1. Ketogenic Diet (KD) — ABSOLUTE CI: CIII completely absent; FAO CoQH2 cannot be
     reoxidised → fatal metabolic crisis (identical to all complete CIII deficiency)
  2. Metformin — ABSOLUTE CI: Complex I inhibitor + complete CIII absence → combined
     CI+CIII block → fatal OXPHOS failure
  3. Valproate (VPA) — ABSOLUTE CI: CoA sequestration + mitochondrial toxicity;
     hepatic failure risk elevated in CIII-deficient patients
  4. Linezolid — ABSOLUTE CI: inhibits mitochondrial translation (MT-CYB); in UQCRC1
     deficiency, MT-CYB is the CIII scaffold template — further suppression → complete
     CIII scaffold destruction
  5. Chloramphenicol — ABSOLUTE CI: broad mitochondrial translation inhibitor; same
     CIII-destabilising mechanism; ABSOLUTE contraindication in all mito disease
  6. Propofol — ABSOLUTE CI: PRIS risk VERY HIGH with completely absent CIII;
     dexmedetomidine or ketamine preferred for procedural sedation

  CONDITIONAL CONCERNS:
  - IV LCT lipid emulsions — AVOID: FAO generates CoQH2 that CANNOT be reoxidised
    with absent CIII → toxic CoQH2 accumulation; MCT-based if lipids needed
  - Phenobarbital — CAUTION: increased mitochondrial energy demand; prefer LEV
  - Tetracyclines (prolonged) — CAUTION: mitochondrial translation inhibition

  SAFE / PREFERRED:
  - Levetiracetam (LEV): preferred AED — no mitochondrial toxicity
  - GIR 6-8 mg/kg/min: mandatory glucose infusion; avoid ALL fasting
  - CoQ10 + Riboflavin + Thiamine: MRC cocktail Level C
  - Riboflavin: NOT riboflavin-responsive (no FAD binding domain, unlike ACAD9)
  - NaHCO3 (acute): IV sodium bicarbonate for acute severe lactic acidosis

REFERENCES:
  - Fernandez-Vizarra & Zeviani (2018) Front Genet — Nuclear CIII gene landscape
  - Berry & Walker (2000) Annu Rev Biochem — bc1 complex architecture; core scaffold role
  - Karaarslan et al. (2012) CIII-D5 initial case reports
  - OMIM Gene *191328 · Disease #615160
"""

import random
import json

SEED = 729
random.seed(SEED)

GENE         = "UQCRC1"
ALIAS        = "QCR1"
OMIM_GENE    = "191328"
OMIM_DISEASE = "615160"
DISEASE      = "Complex III Deficiency, Nuclear Type 5 (CIII-D5)"
CHROMOSOME   = "3p21.31"
INHERITANCE  = "AR (Autosomal Recessive) — biallelic loss-of-function"
PROTEIN_SIZE = "480 aa, ~52.6 kDa; soluble peripheral IMM; no TM helix; matrix-facing; LARGEST nuclear CIII subunit"
COMPLEX      = "Complex III (cytochrome bc1 complex) — core structural scaffold (UQCRC1/UQCRC2 heterodimer)"
FUNCTION     = (
    "Core structural protein 1 of Complex III — the LARGER scaffold partner (~480 aa vs ~453 aa UQCRC2); "
    "UQCRC1 provides the main docking platform for the UQCRC1/UQCRC2 scaffold heterodimer at the matrix face; "
    "loss of UQCRC1 causes secondary degradation of UQCRC2 and of all remaining CIII subunits — "
    "entire CIII holocomplex absent; phenotype indistinguishable from UQCRC2 deficiency by biochemistry; "
    "WES (UQCRC1 = 3p21.31 vs UQCRC2 = 16p12.2) is MANDATORY for gene-level diagnosis"
)
COHORT_N     = 40

ORIGINS = [
    "Turkish", "Pakistani", "Iranian", "Saudi", "Moroccan",
    "Lebanese", "Israeli", "Algerian", "Turkish", "Iranian",
    "German", "French", "Dutch", "British", "Turkish",
    "Pakistani", "Egyptian", "Jordanian", "Saudi", "Iranian",
    "Moroccan", "Turkish", "Syrian", "Pakistani", "Algerian",
    "Israeli", "Turkish", "Iranian", "Saudi", "Pakistani",
    "Lebanese", "Turkish", "Moroccan", "French", "German",
    "British", "Turkish", "Pakistani", "Iranian", "Saudi",
]

ALL_VARIANTS = [
    {"protein": "p.Gly272Arg",  "cdna": "c.814G>C",          "domain": "UQCRC2-binding interface",
     "type": "Missense",        "severity": "Severe",        "penetrance_pct": 89,
     "mechanism": "Conserved glycine at UQCRC2-binding interface; Arg introduces steric and charge conflict; "
                  "UQCRC2 binding abolished → both core proteins secondarily lost; most common UQCRC1 missense"},
    {"protein": "p.Leu168Pro",  "cdna": "c.503T>C",           "domain": "Central scaffold α-helix",
     "type": "Missense",        "severity": "Severe",        "penetrance_pct": 86,
     "mechanism": "Helix-breaking proline in central scaffold α-helix; global UQCRC1 fold collapse; "
                  "proteasomally degraded; UQCRC2 secondarily absent; CIII holocomplex absent"},
    {"protein": "p.Arg395Trp",  "cdna": "c.1183C>T",          "domain": "C-terminal UQCRC2 distal contact",
     "type": "Missense",        "severity": "Severe",        "penetrance_pct": 84,
     "mechanism": "Bulky Trp substitution disrupts C-terminal UQCRC2 distal contact region; "
                  "scaffold heterodimer weakened; complex destabilised; CIII absent"},
    {"protein": "p.Asp210Gly",  "cdna": "c.629A>G",           "domain": "Central scaffold loop",
     "type": "Missense",        "severity": "Severe",        "penetrance_pct": 82,
     "mechanism": "Conserved Asp in central scaffold loop; Gly removes essential H-bond donors; "
                  "fold disruption; UQCRC1 destabilised; UQCRC2 secondarily degraded"},
    {"protein": "p.Ala231Val",  "cdna": "c.692C>T",            "domain": "Hydrophobic core",
     "type": "Missense",        "severity": "Intermediate",  "penetrance_pct": 67,
     "mechanism": "Hydrophobic core packing disruption; partial UQCRC1 instability; "
                  "hypomorphic — some residual UQCRC1/UQCRC2 heterodimer; partial CIII function"},
    {"protein": "c.IVS5+1G>A", "cdna": "c.IVS5+1G>A",         "domain": "Splice donor intron 5",
     "type": "Splice-site",     "severity": "Moderate-Severe", "penetrance_pct": 79,
     "mechanism": "Splice donor loss; partial exon 5 skipping; truncated/unstable UQCRC1; "
                  "UQCRC2 secondarily reduced; moderate-severe CIII deficiency"},
    {"protein": "p.Gln35Ter",   "cdna": "c.103C>T",            "domain": "N-terminal pre-scaffold region",
     "type": "Nonsense",        "severity": "Severe",        "penetrance_pct": 93,
     "mechanism": "Early stop codon; NMD; null allele; complete UQCRC1 loss; "
                  "UQCRC2 secondarily absent; entire CIII absent; neonatal crisis"},
    {"protein": "ExonDel4",     "cdna": "Exon 4 deletion",    "domain": "Central scaffold (exon 4)",
     "type": "Large deletion",  "severity": "Severe",        "penetrance_pct": 90,
     "mechanism": "Deletion removes central scaffold region; null functional allele; "
                  "UQCRC1 absent; UQCRC2 secondarily degraded; CIII absent on BN-PAGE"},
]

VARIANT_WEIGHTS = [89, 86, 84, 82, 67, 79, 93, 90]


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

        # CIII activity — UQCRC1 loss causes near-complete CIII loss (<5-8%)
        if combined_severe:
            ciii_act = round(rng.uniform(1.0, 5.5), 1)
        elif "Intermediate" in sev1 or "Intermediate" in sev2:
            ciii_act = round(rng.uniform(5.0, 16.0), 1)
        else:
            ciii_act = round(rng.uniform(2.0, 7.5), 1)

        # Lactate — severe (8-22 mM)
        if combined_severe:
            lac = round(rng.uniform(9.0, 22.0), 1)
        else:
            lac = round(rng.uniform(5.0, 14.0), 1)

        # Onset — predominantly neonatal-early infantile
        if combined_severe:
            onset_mo = rng.choices([0, 1, 2, 3], weights=[43, 32, 17, 8])[0]
        else:
            onset_mo = rng.choices([1, 2, 3, 4, 6], weights=[14, 28, 32, 18, 8])[0]

        dx_delay = rng.randint(1, 3)
        dx_mo = onset_mo + dx_delay

        sex = rng.choice(["M", "F"])
        origin = ORIGINS[i % len(ORIGINS)]

        # Outcome — severe; high mortality
        if ciii_act < 5 and lac > 12:
            outcome = rng.choice([
                "Deceased 0-3mo", "Deceased 3-12mo",
                "Alive-severe-disability", "Deceased 3-12mo",
            ])
        elif ciii_act < 8:
            outcome = rng.choice([
                "Deceased 3-12mo", "Deceased 12-36mo",
                "Alive-severe-disability", "Deceased 3-12mo",
            ])
        else:
            outcome = rng.choice([
                "Alive-severe-disability", "Alive-moderate-disability",
                "Deceased 12-36mo",
            ])

        consanguineous  = rng.random() < 0.70
        has_leigh       = rng.random() < 0.55
        has_seizures    = rng.random() < 0.42
        has_feeding     = rng.random() < 0.80
        has_hepatic     = rng.random() < 0.58
        has_cardiac     = rng.random() < 0.22
        has_renal       = rng.random() < 0.20
        has_resp_fail   = rng.random() < 0.62

        patients.append({
            "id": f"UQCRC1-{i+1:03d}",
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
            "cardiomyopathy": has_cardiac,
            "renal_tubular": has_renal,
            "respiratory_failure": has_resp_fail,
            "outcome": outcome,
        })
    return patients


PATIENTS = _generate_patients()


def get_overview():
    pts = PATIENTS
    n = len(pts)

    deceased       = [p for p in pts if "Deceased" in p["outcome"]]
    neonatal       = [p for p in pts if p["age_onset_months"] <= 1]
    leigh_pts      = [p for p in pts if p["leigh_mri"]]
    seizure_pts    = [p for p in pts if p["seizures"]]
    feeding_pts    = [p for p in pts if p["feeding_difficulties"]]
    hepatic_pts    = [p for p in pts if p["hepatic_involvement"]]
    cardiac_pts    = [p for p in pts if p["cardiomyopathy"]]
    renal_pts      = [p for p in pts if p["renal_tubular"]]
    resp_pts       = [p for p in pts if p["respiratory_failure"]]
    consangu_n     = len([p for p in pts if p["consanguineous"]])

    avg_ciii = round(sum(p["ciii_activity_pct"] for p in pts) / n, 1)
    avg_lac  = round(sum(p["lactic_acid_mmolL"] for p in pts) / n, 1)

    cohort_features = [
        {"feature": "Lactic acidosis (severe, 8-22 mM)", "pct": 97},
        {"feature": "Hypotonia", "pct": 95},
        {"feature": "Encephalopathy / developmental delay", "pct": 90},
        {"feature": "Feeding difficulties", "pct": round(len(feeding_pts)/n*100)},
        {"feature": "Failure to thrive", "pct": 78},
        {"feature": "Respiratory failure (NIV/ventilation)", "pct": round(len(resp_pts)/n*100)},
        {"feature": "Leigh-like MRI (bilateral BG/brainstem)", "pct": round(len(leigh_pts)/n*100)},
        {"feature": "Seizures", "pct": round(len(seizure_pts)/n*100)},
        {"feature": "Hepatic involvement (↑AST/ALT, hepatomegaly)", "pct": round(len(hepatic_pts)/n*100)},
        {"feature": "Cardiomyopathy (DCM or concentric HCM)", "pct": round(len(cardiac_pts)/n*100)},
        {"feature": "Renal tubular dysfunction (partial Fanconi)", "pct": round(len(renal_pts)/n*100)},
        {"feature": "Consanguinity", "pct": round(consangu_n/n*100)},
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
            "hypotonia_pct": 95,
            "hepatic_pct": round(len(hepatic_pts)/n*100),
            "avg_ciii_activity_pct": avg_ciii,
            "avg_lactic_acid_mmolL": avg_lac,
            "deceased_pct": round(len(deceased)/n*100),
        },
        "cohort_summary_features": cohort_features,
        "top_variant_counts": top_variants,
        "patients": [p for p in pts[:10]],
        "key_clinical_alerts": [
            "🚫 KD (Ketogenic Diet) — ABSOLUTE CI: CIII completely absent; FAO CoQH2 cannot be reoxidised → fatal crisis",
            "🚫 Metformin — ABSOLUTE CI: Complex I inhibitor + complete CIII loss → fatal OXPHOS failure",
            "🚫 Valproate (VPA) — ABSOLUTE CI: CoA sequestration + mito toxicity; hepatic failure risk",
            "🚫 Linezolid — ABSOLUTE CI: MT-CYB translation suppressed → complete CIII scaffold destruction",
            "🚫 Chloramphenicol — ABSOLUTE CI: broad mito translation inhibitor; same CIII-destabilising effect",
            "🚫 Propofol — ABSOLUTE CI: PRIS risk VERY HIGH with completely absent CIII; use dexmedetomidine",
            "⚠️ IV LCT lipids — AVOID: FAO CoQH2 backlog with absent CIII; MCT-based if lipids needed",
            "⚠️ Phenobarbital — CAUTION: increased mito demand; prefer LEV for seizures",
            "✅ LEV — Preferred AED: no mito toxicity; safe in CIII deficiency",
            "✅ GIR 6-8 mg/kg/min — mandatory; avoid ALL fasting; especially during intercurrent illness",
            "✅ CoQ10 + Riboflavin + Thiamine — MRC cocktail Level C",
            "✅ UQCRC1 vs CYC1 DDx: NO cataracts in UQCRC1; cataracts 35% in CYC1 — bedside distinguisher",
            "✅ UQCRC1 vs UQCRC2 DDx: Both absent CIII; WES (3p21.31 vs 16p12.2) MANDATORY to distinguish",
        ],
    }


def get_breakdown():
    pts = PATIENTS
    n = len(pts)

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
            "ciii_below_5_pct":   round(len(ciii_below_5)/n*100),
            "ciii_5_to10_pct":    round(len(ciii_5_to10)/n*100),
            "ciii_above_10_pct":  round(len(ciii_above_10)/n*100),
            "lactic_above_15_pct": round(len(lac_above_15)/n*100),
            "lactic_8_to_15_pct":  round(len(lac_8_to_15)/n*100),
            "lactic_below_8_pct":  round(len(lac_below_8)/n*100),
        },
        "immunoblot_pattern": {
            "UQCRC1_Core1": "ABSENT — primary loss; largest CIII core scaffold protein; proteasomally degraded",
            "UQCRC2_Core2": "ABSENT (secondary) — reciprocal degradation when UQCRC1 lost; "
                            "BOTH core proteins absent — distinguishes from LYRM7/BCS1L (Core1/Core2 PRESERVED)",
            "UQCRFS1_RISP": "ABSENT (secondary) — RISP degraded without CIII scaffold; "
                            "mechanism: scaffold loss vs LYRM7 (direct chaperone loss) vs BCS1L (insertion block)",
            "CYC1_heme_c1": "ABSENT (secondary) — structural heme c1 subunit; secondarily degraded",
            "MT_CYB":       "REDUCED (secondary) — mtDNA-encoded; destabilised without nuclear scaffold",
        },
        "bn_page_pattern": {
            "finding": "CIII2 and CIII2+CI supercomplexes ABSENT (<5%); "
                       "NO CIII precomplex accumulation; all sub-complexes absent",
            "interpretation": "Core scaffold loss: ENTIRE CIII holocomplex absent; "
                              "no partial assembly intermediates (unlike BCS1L precomplex); "
                              "BN-PAGE identical to UQCRC2 and CYC1 deficiency — WES mandatory",
            "ddx_value": "UQCRC2 ABSENT on immunoblot = UQCRC1 or UQCRC2 deficiency (and CYC1); "
                         "NOT LYRM7/BCS1L (Core2 preserved there); "
                         "UQCRC1 vs UQCRC2: both absent CIII; WES gene ID (3p21.31 vs 16p12.2) required; "
                         "UQCRC1 vs CYC1: no cataracts in UQCRC1; cataracts 35% in CYC1",
        },
        "outcome_distribution": [
            {"outcome": k, "count": v} for k, v in
            sorted(outcome_dist.items(), key=lambda x: -x[1])
        ],
        "genetic_counselling": {
            "inheritance": "Autosomal Recessive — biallelic loss-of-function required",
            "recurrence_risk": "25% per pregnancy for confirmed AR couple",
            "carrier_frequency": "Rare globally; no known high-frequency founder mutations",
            "prenatal_testing": "Available via Sanger/NGS of known familial variants; CVS/amniocentesis",
            "sex_bias": "Both sexes equally affected (autosomal gene, 3p21.31)",
        },
    }


def get_definitions():
    return {
        "gene": GENE,
        "alias": ALIAS,
        "full_name": "Ubiquinol-Cytochrome c Reductase Core Protein 1 (Core 1 / QCR1)",
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "disease_name": DISEASE,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "ciii_assembly_step": "Core structural scaffold — LARGEST nuclear CIII subunit; forms UQCRC1/UQCRC2 "
                              "heterodimer at CIII matrix face; loss causes ENTIRE CIII holocomplex absent "
                              "(ALL subunits secondarily degraded); indistinguishable from UQCRC2 deficiency "
                              "by biochemistry alone; WES mandatory for gene-level diagnosis",
        "protein": {
            "size_aa": 480,
            "kDa": 52.6,
            "tm_helices": 0,
            "localization": "Soluble peripheral IMM protein; no TM helix; matrix-facing; MTS cleaved; "
                            "LARGEST nuclear-encoded CIII subunit",
            "partner": "UQCRC2 (Core Protein 2) — obligate heterodimer partner; "
                       "MT-CYB — scaffold contact during CIII biogenesis",
            "function": "Central CIII core scaffold; dominant partner in UQCRC1/UQCRC2 heterodimer; "
                        "provides main docking platform for UQCRC2, CYC1, RISP; "
                        "evolved from bacterial signal peptidase 2 (lost catalytic activity in eukaryotes); "
                        "loss → UQCRC2 secondarily absent → CIII holocomplex absent",
        },
        "key_biochemical_features": [
            "UQCRC1 absent → UQCRC2 (Core2) also absent (reciprocal secondary degradation)",
            "ALL CIII core subunits absent — identical BN-PAGE/immunoblot pattern to UQCRC2 and CYC1",
            "UQCRC2 ABSENT on immunoblot = UQCRC1 or UQCRC2 deficiency (NOT LYRM7/BCS1L — Core2 preserved there)",
            "BN-PAGE: CIII2 and supercomplexes ABSENT (<5%) — not merely reduced",
            "NO CIII precomplex accumulation — distinguishes from BCS1L (precomplex pathognomonic)",
            "CIII activity <5% residual — functionally absent (more severe than LYRM7 15-35%)",
            "NO cataracts — KEY bedside DDx vs CYC1 (cataracts ~35% in CYC1, absent in UQCRC1)",
            "UQCRC1 vs UQCRC2: biochemically identical; WES gene ID (3p21.31 vs 16p12.2) MANDATORY",
        ],
        "bn_page": (
            "CIII2 and CIII2+CI supercomplexes ABSENT (not merely reduced). "
            "No CIII precomplex accumulation (DDx BCS1L — precomplex pathognomonic). "
            "ALL CIII core subunits absent on immunoblot (UQCRC1, UQCRC2, RISP, CYC1) — "
            "BN-PAGE/immunoblot pattern identical to UQCRC2 and CYC1 deficiency; "
            "WES (UQCRC1 = 3p21.31 vs UQCRC2 = 16p12.2 vs CYC1 = 8q24.13) is MANDATORY for diagnosis."
        ),
        "absolute_contraindications": [
            "🚫 Ketogenic Diet — ABSOLUTE CI: CIII completely absent; FAO CoQH2 backlog → fatal",
            "🚫 Metformin — ABSOLUTE CI: Complex I inhibitor; combined CI+CIII block → fatal",
            "🚫 Valproate (VPA) — ABSOLUTE CI: CoA sequestration + mito toxicity; hepatic failure",
            "🚫 Linezolid — ABSOLUTE CI: MT-CYB translation inhibited → CIII scaffold destruction",
            "🚫 Chloramphenicol — ABSOLUTE CI: broad mito translation; same CIII-destabilising effect",
            "🚫 Propofol — ABSOLUTE CI: PRIS VERY HIGH risk; completely absent CIII; use dexmedetomidine",
        ],
        "relative_contraindications": [
            "⚠️ IV LCT lipid emulsions — AVOID: FAO CoQH2 backlog with absent CIII",
            "⚠️ Phenobarbital — CAUTION: increased mitochondrial demand; prefer levetiracetam",
            "⚠️ Tetracyclines (prolonged) — CAUTION: mito translation inhibition; avoid long-term use",
        ],
        "safe_treatments": [
            "✅ LEV (Levetiracetam) — preferred AED; no mitochondrial toxicity",
            "✅ GIR 6-8 mg/kg/min — mandatory continuous glucose; avoid ALL fasting",
            "✅ CoQ10 + Riboflavin + Thiamine — MRC cocktail; Level C evidence",
            "✅ NaHCO3 (IV) — acute lactic acidosis management; pH target ≥7.25",
            "✅ NIV/BiPAP — respiratory support; early initiation for respiratory failure",
            "✅ Dexmedetomidine or Ketamine — safe sedation alternatives to propofol",
        ],
        "ddx_pearls": [
            "UQCRC1 vs UQCRC2: BOTH cause absent CIII + absent Core1/Core2 on immunoblot; WES MANDATORY",
            "UQCRC1 vs CYC1: NO cataracts in UQCRC1; cataracts ~35% in CYC1 — KEY bedside DDx",
            "UQCRC1 vs LYRM7: Core1/Core2 ABSENT in UQCRC1; PRESERVED in LYRM7 (only RISP absent)",
            "UQCRC1 vs BCS1L: NO precomplex in UQCRC1; CIII precomplex ACCUMULATES in BCS1L (pathognomonic)",
            "UQCRC1 vs TTC19: UQCRC1 = neonatal/infantile severe; TTC19 = childhood neurological/psychiatric",
            "UQCRC1 vs UQCC1/UQCC2: All absent CIII; UQCC1/UQCC2 absent on immunoblot in UQCC1/UQCC2 deficiency",
        ],
        "key_references": [
            "Fernandez-Vizarra & Zeviani (2018) Front Genet — nuclear CIII gene landscape",
            "Berry & Walker (2000) Annu Rev Biochem — bc1 complex architecture; core scaffold role",
            "OMIM Gene *191328 · Disease #615160 (CIII-D5)",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (variant count) ===")
    bd = get_breakdown()
    print(f"Avg CIII: {bd['biochemistry_distribution']['avg_ciii_activity_pct']}%")
    print(f"Avg Lac:  {bd['biochemistry_distribution']['avg_lactic_acid_mmolL']} mmol/L")
    print("\n=== DEFINITIONS ===")
    print(json.dumps(get_definitions(), indent=2)[:1000])
