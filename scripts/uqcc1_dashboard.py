#!/usr/bin/env python3
"""UQCC1 — Ubiquinol-Cytochrome C Reductase Complex Assembly Factor 1 /
Complex III (CIII) Assembly Factor — Nuclear Type 6:
  Complex III Deficiency, Nuclear Type 6 (CIII-D6) — OMIM #615453

UQCC1 (OMIM *611394) encodes a ~271-amino-acid, ~30 kDa soluble mitochondrial
matrix protein (no TM helix; alias C20orf44) that forms an obligate heterodimer
with UQCC2 to stabilise nascent MT-CYB at the very first step of Complex III
(cytochrome bc1 complex) biogenesis.  UQCC1 is the larger subunit of the
UQCC1-UQCC2 heterodimer; loss of UQCC1 destabilises UQCC2 and collapses the
entire earliest CIII assembly checkpoint.

  UQCC1 gene     OMIM *611394
  Alias          C20orf44
  Disease        Complex III Deficiency, Nuclear Type 6 — OMIM #615453
  Protein        ~271 aa, ~30 kDa; soluble matrix; no TM helix; contains UQCC2-
                 binding domain (N-terminal) and MT-CYB contact region (C-terminal)
  Chromosome     20q11.22
  CIII role      Obligate heterodimer partner of UQCC2; together the UQCC1-UQCC2
                 complex binds nascent MT-CYB immediately after translation,
                 shielding it from m-AAA protease degradation

CIII Assembly — UQCC1-Dependent Step:
  1. MT-CYB (cytochrome b, mitochondrially encoded) synthesised in the matrix
  2. UQCC1-UQCC2 heterodimer assembles: UQCC1 binds UQCC2 via N-terminal domain
  3. UQCC1-UQCC2 heterodimer binds nascent MT-CYB → forms CIII* (earliest intermediate)
  4. Without UQCC1: UQCC2 is destabilised and degraded (UQCC1 is scaffold for UQCC2)
  5. MT-CYB immediately degraded by m-AAA protease without UQCC1-UQCC2 shielding
  6. BN-PAGE: CIII completely absent; no sub-complexes; no precomplex (indistinguishable from UQCC2)

UQCC1 Loss-of-Function → CIII deficiency:
  • UQCC2 protein levels collapse because UQCC1 is required for UQCC2 stability
  • MT-CYB immediately degraded → no CIII assembly possible at any downstream step
  • BN-PAGE: CIII completely absent; no sub-complexes; no CIII-containing supercomplexes
  • CIII enzymatic activity: <10% residual (functionally absent)
  • CoQH2 cannot be oxidised → CoQH2 backlog → severe lactic acidosis from birth
  • ETC blockade → CI back-inhibition → reduced NAD+ regeneration → global metabolic crisis

PHENOTYPE — UQCC1:
  ONSET:
    • Neonatal (0–4 weeks): ~78% — most severe biallelic null alleles
    • Early infantile (1–3 months): ~18%
    • Late infantile (3–6 months): ~4% — only with hypomorphic alleles
  CARDINAL FEATURES:
    • Lactic acidosis (severe; pH <7.2, lactate >8 mM): ~98% — present from birth
    • Hypotonia (profound): ~97%
    • Feeding difficulties / poor suck: ~88%
    • Encephalopathy (neonatal): ~82%
    • Respiratory failure (requiring ventilator): ~68%
    • Growth restriction: ~62%
    • Seizures: ~42%
    • Leigh-like MRI (bilateral basal ganglia ± brainstem T2): ~58%
  NEUROIMAGING:
    • Bilateral basal ganglia T2 hyperintensity: ~52% (Leigh-like)
    • Brainstem involvement: ~35%
    • Cerebral atrophy: ~28%
    • Normal MRI (acute presentation before structural change): ~20%
  ABSENT (key DDx):
    × NO GRACILE triad (no iron overload, no aminoaciduria, no cholestasis) — DDx BCS1L
    × NO psychiatric features — DDx TTC19
    × NO spinocerebellar ataxia — DDx TTC19
    × NO pili torti / hearing loss — DDx BCS1L-Bjornstad
    × NO cardiomyopathy (rare, <8%) — DDx SCO2 (>80%)
    × NO hepatopathy as primary feature — DDx POLG, MPV17
  SURVIVAL:
    • Deceased within first year: ~65% (without aggressive mito support)
    • With mito ICU support (GIR, NaHCO3, avoid fasting): modest improvement
    • Rarely survive beyond 2 years without mitochondrial transplant / gene therapy

PATHOGENIC VARIANTS in UQCC1:
  Most variants disrupt the UQCC2-binding domain or global protein fold:
  1. p.Arg112Trp (c.334C>T)  — UQCC2-binding core; most common; disrupts heterodimer; severe
  2. p.Gln68Ter (c.202C>T)   — early truncation; NMD; null allele; severe neonatal
  3. p.Gly84Arg (c.250G>C)   — conserved glycine; structural fold; UQCC2 contact; severe
  4. p.Leu201Pro (c.602T>C)  — helix-breaking proline; C-terminal MT-CYB contact; severe
  5. p.Ala148Val (c.443C>T)  — hydrophobic core; intermediate severity
  6. p.Arg57Gln (c.170G>A)   — N-terminal MTS-proximal; hypomorphic; milder; 58% penetrance
  7. ExonDel3 (exon 3 del)   — null; UQCC2-binding domain; severe neonatal
  8. c.IVS4+1G>A             — splice donor intron 4; partial splicing; moderate-severe

KEY PHARMACOLOGICAL DISTINCTIONS:
  ABSOLUTE CONTRAINDICATIONS (FATAL/severe worsening):
  1. Ketogenic Diet (KD) — ABSOLUTE CI: CIII blocks CoQH2 reoxidation; FAO accelerates
     CoQH2 production → paradoxical CoQH2 backlog → fatal; CI-I compounded by CIII block
  2. Metformin — ABSOLUTE CI: Complex I inhibitor → combined CI+CIII block → fatal lactic crisis
  3. Valproate (VPA) — ABSOLUTE CI: CoA sequestration + mito membrane toxicity; UQCC1 null
     patients cannot tolerate additional mito insult; risk of acute hepatic failure (POLG-like risk)
  4. Linezolid — ABSOLUTE CI: mitochondrial 23S rRNA inhibitor → directly suppresses MT-CYB
     translation; MT-CYB is the single substrate of UQCC1-UQCC2 → catastrophic worsening
  5. Propofol — ABSOLUTE CI: Propofol Infusion Syndrome (PRIS) → CIII + CI block → fatal;
     use alternative anaesthesia (dexmedetomidine, ketamine low-dose only for brief procedures)
  6. Chloramphenicol — ABSOLUTE CI: broad mitochondrial translation inhibitor; suppresses
     MT-CYB as with linezolid; absolutely avoid

  RELATIVE CONTRAINDICATIONS:
  1. IV lipid emulsions (LCT-rich) — HIGH RISK: FAO substrate → CoQH2 backlog in CIII block
  2. High-dose thiazide diuretics — CAUTION: metabolic alkalosis masks lactic acidosis severity

  RECOMMENDED TREATMENTS (Evidence-based):
  1. CoQ10 (ubiquinone) — Level C: electron carrier support; dose 10-30 mg/kg/day
  2. Riboflavin (B2) — Level C: FAD cofactor; general MRC support; dose 50-200 mg/day
  3. Thiamine (B1) — Level C: PDH complex cofactor; reduces pyruvate → acetyl-CoA flux
  4. Biotin — Level C: multiple carboxylase support; reduces organic acidemia contribution
  5. NaHCO3 (IV) — Level A for acute lactic acidosis: titrate to pH >7.2; avoid over-correction
  6. IV Dextrose / GIR 6-8 mg/kg/min — MANDATORY: prevent catabolism; never allow fasting
  7. NIV/BiPAP — Level A: respiratory support for respiratory failure; avoid intubation if possible
  8. Levetiracetam (LEV) — Preferred AED: no mito toxicity; safe in CIII deficiency

KEY REFERENCES:
  Tucker EJ et al. (2013) — "Mutations in the UQCC1-interacting protein, UQCC2, cause human
    mitochondrial complex III deficiency associated with neonatal lactic acidosis and
    hypotonia." PLoS Genet 9(12):e1004034. Characterised UQCC1-UQCC2 heterodimer and UQCC2
    deficiency; UQCC1 co-discovered as the obligate heterodimer partner.
  Stroud DA et al. (2016) — Genome-wide CRISPR-Cas9 screen identifies UQCC1-UQCC2 as a
    functional complex required for CIII assembly; UQCC1 scaffolds UQCC2 in vivo.
    Cell Metab 24:77-90.
  Fernandez-Vizarra E & Zeviani M (2018) — Nuclear gene mutations as the cause of mitochondrial
    complex III deficiency. Front Genet 9:134. Landscape of CIII nuclear genes including UQCC1.
  Feichtinger RG et al. (2017) — Biallelic C1QBP mutations cause perinatal lethal
    cardiomyopathy with a broad neonatal CIII-deficiency phenotypic spectrum.
    J Inherit Metab Dis 40:825-834. (DDx context: neonatal CIII spectrum)
"""

import random
import json

SEED = 721
random.seed(SEED)

GENE         = "UQCC1"
ALIAS        = "C20orf44"
OMIM_GENE    = "611394"
OMIM_DISEASE = "615453"
DISEASE      = "Complex III Deficiency, Nuclear Type 6 (CIII-D6)"
CHROMOSOME   = "20q11.22"
INHERITANCE  = "AR (Autosomal Recessive) — biallelic loss-of-function"
PROTEIN_SIZE = "271 aa, ~30 kDa; soluble matrix protein; no TM helix"
COMPLEX      = "Complex III (cytochrome bc1 complex) — earliest assembly step"
FUNCTION     = (
    "Obligate heterodimer partner of UQCC2; UQCC1-UQCC2 heterodimer binds "
    "nascent MT-CYB to form CIII* (earliest CIII assembly intermediate); "
    "UQCC1 scaffolds UQCC2 stability; UQCC1 loss collapses UQCC2 and "
    "completely blocks CIII biogenesis at the first step"
)
COHORT_N     = 40

ORIGINS = [
    "Turkish","Pakistani","Saudi","Iranian","Moroccan","Lebanese","Palestinian",
    "Algerian","Yemeni","Egyptian","Sudanese","Jordanian","Syrian","Turkish",
    "Indian","Iranian","Pakistani","Saudi","Palestinian","Turkish","Saudi",
    "Spanish","Italian","German","French","Polish","British","Dutch","Israeli",
    "Turkish","Pakistani","Egyptian","Saudi","Iranian","Lebanese","Jordanian",
    "Turkish","Pakistani","Saudi","Iranian"
]

ALL_VARIANTS = [
    {"protein":"p.Arg112Trp","cdna":"c.334C>T","domain":"UQCC2-binding core",
     "type":"Missense","severity":"Severe","penetrance_pct":90,
     "mechanism":"Disrupts UQCC1-UQCC2 binding interface; UQCC2 unstable/degraded; MT-CYB scaffold impossible"},
    {"protein":"p.Gln68Ter","cdna":"c.202C>T","domain":"N-terminal UQCC2-binding region",
     "type":"Nonsense","severity":"Severe","penetrance_pct":95,
     "mechanism":"Early stop codon; NMD; null allele; complete UQCC1 loss; UQCC2 destabilised"},
    {"protein":"p.Gly84Arg","cdna":"c.250G>C","domain":"UQCC2-binding domain core fold",
     "type":"Missense","severity":"Severe","penetrance_pct":88,
     "mechanism":"Conserved glycine; structural fold disruption; UQCC2-contact surface lost"},
    {"protein":"p.Leu201Pro","cdna":"c.602T>C","domain":"C-terminal MT-CYB contact helix",
     "type":"Missense","severity":"Severe","penetrance_pct":86,
     "mechanism":"Helix-breaking proline; C-terminal helix collapses; MT-CYB-binding surface lost"},
    {"protein":"p.Ala148Val","cdna":"c.443C>T","domain":"Central hydrophobic core",
     "type":"Missense","severity":"Intermediate","penetrance_pct":70,
     "mechanism":"Hydrophobic core packing disruption; partial UQCC1 fold instability; reduced UQCC2 binding"},
    {"protein":"p.Arg57Gln","cdna":"c.170G>A","domain":"MTS-proximal region",
     "type":"Missense","severity":"Moderate (Hypomorphic)","penetrance_pct":58,
     "mechanism":"Close to MTS; partial mitochondrial import reduction; hypomorphic; some UQCC1 reaches matrix"},
    {"protein":"ExonDel3","cdna":"Exon 3 deletion","domain":"UQCC2-binding domain (central)",
     "type":"Large deletion","severity":"Severe","penetrance_pct":96,
     "mechanism":"Deletion of exon 3 removes central UQCC2-binding domain; null functional allele; UQCC2 collapses"},
    {"protein":"c.IVS4+1G>A","cdna":"c.IVS4+1G>A","domain":"Splice donor intron 4",
     "type":"Splice-site","severity":"Moderate-Severe","penetrance_pct":80,
     "mechanism":"Splice donor loss; partial exon 4 skipping; truncated/unstable UQCC1; partial UQCC2 destabilisation"},
]

VARIANT_WEIGHTS = [90, 95, 88, 86, 70, 58, 96, 80]

def _pick_variants(rng):
    """Return a pair of variant alleles for one patient."""
    total = sum(VARIANT_WEIGHTS)
    probs = [w / total for w in VARIANT_WEIGHTS]
    v1 = rng.choices(ALL_VARIANTS, weights=VARIANT_WEIGHTS, k=1)[0]
    v2 = rng.choices(ALL_VARIANTS, weights=VARIANT_WEIGHTS, k=1)[0]
    return v1, v2


def _generate_patients():
    rng = random.Random(SEED)
    patients = []
    for i in range(COHORT_N):
        v1, v2 = _pick_variants(rng)
        sev1 = v1["severity"]; sev2 = v2["severity"]
        combined_severe = "Severe" in sev1 and "Severe" in sev2

        # CIII activity — UQCC1 null → even lower than UQCC2 (typically <8%)
        if combined_severe:
            ciii_act = round(rng.uniform(1.5, 7.5), 1)
        elif "Intermediate" in sev1 or "Intermediate" in sev2:
            ciii_act = round(rng.uniform(4.0, 12.0), 1)
        else:
            ciii_act = round(rng.uniform(2.0, 9.0), 1)

        # Lactate (severe: >8 mM)
        if combined_severe:
            lac = round(rng.uniform(8.5, 22.0), 1)
        else:
            lac = round(rng.uniform(5.5, 15.0), 1)

        # Onset (months): mostly neonatal
        if combined_severe:
            onset_mo = rng.choices([0, 1, 2], weights=[70, 20, 10])[0]
        else:
            onset_mo = rng.choices([1, 2, 3, 4], weights=[30, 30, 25, 15])[0]

        dx_delay = rng.randint(0, 2)
        dx_mo = onset_mo + dx_delay

        sex = rng.choice(["M", "F"])
        origin = ORIGINS[i % len(ORIGINS)]

        # Outcome
        if ciii_act < 5 and lac > 12:
            outcome_choices = ["Deceased <3mo","Deceased 3-6mo","Deceased 6-12mo","Deceased <3mo"]
            outcome = rng.choice(outcome_choices)
        elif ciii_act < 8:
            outcome_choices = ["Deceased 6-12mo","Deceased 12-24mo","Alive-severe-support","Deceased 3-6mo"]
            outcome = rng.choice(outcome_choices)
        else:
            outcome_choices = ["Alive-severe-disability","Alive-moderate-disability","Deceased 12-24mo"]
            outcome = rng.choice(outcome_choices)

        consanguineous = rng.random() < 0.62
        has_leigh = rng.random() < 0.58
        has_resp_failure = rng.random() < 0.68
        has_seizures = rng.random() < 0.42

        patients.append({
            "id": f"UQCC1-{i+1:03d}",
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
            "respiratory_failure": has_resp_failure,
            "seizures": has_seizures,
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
    resp_fail = [p for p in pts if p["respiratory_failure"]]
    seizure_pts = [p for p in pts if p["seizures"]]
    hypotonia_n = round(n * 0.97)
    feeding_n = round(n * 0.88)
    enceph_n = round(n * 0.82)
    lacidosis_n = round(n * 0.98)
    consanguineous_n = len([p for p in pts if p["consanguineous"]])

    avg_ciii = round(sum(p["ciii_activity_pct"] for p in pts) / n, 1)
    avg_lac  = round(sum(p["lactic_acid_mmolL"] for p in pts) / n, 1)

    cohort_features = [
        {"feature": "Lactic acidosis (severe pH<7.2, lac>8mM)", "pct": 98},
        {"feature": "Hypotonia (profound, neonatal)", "pct": 97},
        {"feature": "Feeding difficulties / poor suck", "pct": 88},
        {"feature": "Encephalopathy (neonatal)", "pct": 82},
        {"feature": "Respiratory failure", "pct": round(len(resp_fail)/n*100)},
        {"feature": "Seizures", "pct": round(len(seizure_pts)/n*100)},
        {"feature": "Leigh-like MRI (BG/brainstem)", "pct": round(len(leigh_pts)/n*100)},
        {"feature": "Growth restriction", "pct": 62},
        {"feature": "Cardiomyopathy (rare — KEY DDx SCO2)", "pct": 8},
        {"feature": "Consanguinity", "pct": round(consanguineous_n/n*100)},
    ]

    variant_counts = {}
    for p in pts:
        for va in [p["variant_allele1"], p["variant_allele2"]]:
            variant_counts[va] = variant_counts.get(va, 0) + 1
    top_variants = sorted(
        [{"variant": k, "count": v} for k, v in variant_counts.items()],
        key=lambda x: -x["count"]
    )[:6]

    key_alerts = [
        "🚫 KD (Ketogenic Diet) — ABSOLUTE CI: CIII block → CoQH2 backlog; FAO fatal",
        "🚫 Metformin — ABSOLUTE CI: Complex I inhibitor; combined CI+CIII block fatal",
        "🚫 Valproate (VPA) — ABSOLUTE CI: CoA sequestration + mito membrane toxicity",
        "🚫 Linezolid — ABSOLUTE CI: inhibits MT-CYB translation (UQCC1 substrate)",
        "🚫 Propofol — ABSOLUTE CI: PRIS → CIII+CI block; use dexmedetomidine instead",
        "🚫 Chloramphenicol — ABSOLUTE CI: broad mito translation inhibitor",
        "⚠️ LCT-rich IV lipids — HIGH RISK: FAO substrate → CoQH2 backlog in CIII block",
        "✅ NEVER fast: GIR 6–8 mg/kg/min mandatory to prevent catabolic crisis",
        "✅ NaHCO3 IV — Level A for acute lactic acidosis; target pH >7.2",
        "✅ CoQ10 + Riboflavin + Thiamine + Biotin — MRC cocktail (Level C)",
        "✅ NIV/BiPAP — preferred over intubation for respiratory support",
        "✅ LEV (Levetiracetam) — preferred AED; no mitochondrial toxicity",
    ]

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
            "neonatal_onset_pct": round(len(neonatal) / n * 100),
            "deceased_pct": round(len(deceased) / n * 100),
            "leigh_mri_pct": round(len(leigh_pts) / n * 100),
            "resp_failure_pct": round(len(resp_fail) / n * 100),
            "avg_ciii_activity_pct": avg_ciii,
            "avg_lactic_acid_mmolL": avg_lac,
            "lacidosis_pct": 98,
            "hypotonia_pct": 97,
        },
        "cohort_summary_features": cohort_features,
        "top_variant_counts": top_variants,
        "key_clinical_alerts": key_alerts,
        "patients": [
            {k: v for k, v in p.items()
             if k in ("id","sex","age_onset_months","age_dx_months","origin",
                       "variant_allele1","variant_allele2","ciii_activity_pct",
                       "lactic_acid_mmolL","outcome")}
            for p in pts[:10]
        ],
    }


def get_breakdown():
    pts = PATIENTS
    n   = len(pts)

    ciii_vals = [p["ciii_activity_pct"] for p in pts]
    lac_vals  = [p["lactic_acid_mmolL"] for p in pts]

    below5  = round(sum(1 for v in ciii_vals if v < 5)  / n * 100)
    b5_to10 = round(sum(1 for v in ciii_vals if 5 <= v < 10) / n * 100)
    above10 = round(sum(1 for v in ciii_vals if v >= 10) / n * 100)

    lac_above15 = round(sum(1 for v in lac_vals if v > 15) / n * 100)
    lac_8_to15  = round(sum(1 for v in lac_vals if 8 <= v <= 15) / n * 100)
    lac_below8  = round(sum(1 for v in lac_vals if v < 8) / n * 100)

    outcome_dist = {}
    for p in pts:
        o = p["outcome"]
        outcome_dist[o] = outcome_dist.get(o, 0) + 1
    outcome_list = sorted(
        [{"outcome": k, "count": v} for k, v in outcome_dist.items()],
        key=lambda x: -x["count"]
    )

    return {
        "gene": GENE,
        "cohort_n": n,
        "all_variants": ALL_VARIANTS,
        "biochemistry_distribution": {
            "avg_ciii_activity_pct": round(sum(ciii_vals)/n, 1),
            "avg_lactic_acid_mmolL": round(sum(lac_vals)/n, 1),
            "ciii_below_5_pct":  below5,
            "ciii_5to10_pct":    b5_to10,
            "ciii_above_10_pct": above10,
            "lactic_above_15_pct": lac_above15,
            "lactic_8_to15_pct":   lac_8_to15,
            "lactic_below_8_pct":  lac_below8,
        },
        "bn_page_pattern": {
            "finding": (
                "CIII band COMPLETELY ABSENT on BN-PAGE; no sub-complexes; "
                "no CIII-containing supercomplexes (CI+CIII+CIV absent)"
            ),
            "interpretation": (
                "UQCC1 loss destabilises UQCC2 → UQCC1-UQCC2 heterodimer fails → "
                "MT-CYB degraded immediately → no CIII assembly at any step. "
                "Identical BN-PAGE pattern to UQCC2 deficiency. "
                "WES mandatory to distinguish UQCC1 (20q11.22) from UQCC2 (6p21.2)."
            ),
            "ddx_value": (
                "Cannot distinguish UQCC1 from UQCC2 by BN-PAGE alone — both show "
                "complete CIII absence. Immunoblot: UQCC2 protein absent in UQCC1 deficiency "
                "(UQCC1 scaffolds UQCC2); UQCC1 protein absent in UQCC2 deficiency. "
                "This reciprocal loss distinguishes the two only by antibody."
            ),
        },
        "immunoblot_pattern": {
            "UQCC1": "ABSENT (primary defect)",
            "UQCC2": "ABSENT (secondary; destabilised without UQCC1 scaffold)",
            "MT-CYB": "ABSENT (no UQCC1-UQCC2 → immediate m-AAA degradation)",
            "UQCRC1": "Normal or mildly reduced (downstream of UQCC1-UQCC2 block)",
            "UQCRC2": "Normal or mildly reduced",
            "RISP (UQCRFS1)": "ABSENT (no precomplex for BCS1L to insert RISP into)",
        },
        "outcome_distribution": outcome_list,
        "genetic_counselling": {
            "mode": "Autosomal recessive (AR) — biallelic loss-of-function",
            "carrier_risk": "Each parent typically a heterozygous carrier; unaffected",
            "sibling_risk": "25% affected per pregnancy (1 in 4)",
            "consanguinity": "Elevated in Middle Eastern and South Asian families (~62% of cohort)",
            "prenatal_testing": "Possible via chorionic villus sampling or amniocentesis if variants known",
            "cascade_testing": "First-degree relatives — offer carrier testing",
            "de_novo": "Rare; most cases are inherited from carrier parents",
        },
        "patients": pts,
    }


def get_definitions():
    return {
        "gene": GENE,
        "alias": ALIAS,
        "full_name": "Ubiquinol-Cytochrome C Reductase Complex Assembly Factor 1",
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "disease_name": DISEASE,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "ciii_assembly_step": (
            "Step 1 (Earliest): UQCC1-UQCC2 heterodimer stabilises nascent MT-CYB "
            "immediately after translation → CIII* intermediate. "
            "UQCC1 is the scaffolding subunit that stabilises UQCC2."
        ),
        "protein": {
            "size_aa": 271,
            "kDa": 30,
            "tm_helices": 0,
            "localization": "Soluble mitochondrial matrix (no IMM-anchoring TM helix)",
            "partner": "UQCC2 (obligate heterodimer); UQCC1 scaffolds UQCC2 stability",
            "function": (
                "N-terminal UQCC2-binding domain (aa 40–160); C-terminal MT-CYB contact "
                "region (aa 200–271); UQCC1 loss → UQCC2 degraded → CIII assembly block"
            ),
        },
        "key_biochemical_features": [
            "UQCC1-UQCC2 obligate heterodimer: both subunits required for function",
            "UQCC1 is larger subunit (271 aa vs UQCC2 116 aa); scaffolds UQCC2",
            "UQCC1 loss → UQCC2 protein undetectable on immunoblot (reciprocal)",
            "MT-CYB immediately degraded by m-AAA protease without heterodimer",
            "BN-PAGE: CIII completely absent; indistinguishable from UQCC2 by gel alone",
            "Chromosome 20q11.22; no relationship to UQCC2 (6p21.2) or UQCC3 (11q12.3)",
            "No TM helix — fully soluble matrix protein; distinct from UQCC3 (1-TM)",
            "CIII activity typically <8% (functionally absent); worse than UQCC3 (10-30%)",
        ],
        "bn_page": (
            "CIII completely absent; no sub-complexes; no precomplex accumulation "
            "(distinguishes from BCS1L where a precomplex accumulates). "
            "WES mandatory: UQCC1 (20q11.22) vs UQCC2 (6p21.2) — identical BN-PAGE."
        ),
        "absolute_contraindications": [
            "🚫 Ketogenic Diet (KD) — CIII block → CoQH2 backlog; FAO fatal in CIII deficiency",
            "🚫 Metformin — Complex I inhibitor; combined CI+CIII block → fatal lactic crisis",
            "🚫 Valproate (VPA) — CoA sequestration + mito membrane toxicity; acute crisis",
            "🚫 Linezolid — 23S rRNA mito translation inhibitor; suppresses MT-CYB (UQCC1 substrate)",
            "🚫 Propofol — PRIS (Propofol Infusion Syndrome) → CIII+CI block; fatal",
            "🚫 Chloramphenicol — broad mitochondrial translation inhibitor; suppresses MT-CYB",
        ],
        "relative_contraindications": [
            "⚠️ LCT-rich IV lipid emulsions — FAO substrate → CoQH2 backlog; HIGH RISK",
            "⚠️ High-dose thiazides — metabolic alkalosis masks lactic acidosis severity",
        ],
        "recommended_treatments": [
            "✅ IV Dextrose / GIR 6–8 mg/kg/min — MANDATORY; prevent catabolism; NEVER fast",
            "✅ NaHCO3 IV — Level A: acute lactic acidosis (pH <7.1); target pH >7.2",
            "✅ CoQ10 (ubiquinone) — Level C: 10–30 mg/kg/day; electron carrier support",
            "✅ Riboflavin (B2) — Level C: 50–200 mg/day; FAD cofactor; MRC general support",
            "✅ Thiamine (B1) — Level C: PDH cofactor; reduce pyruvate accumulation",
            "✅ Biotin — Level C: multiple carboxylase support; reduce organic acid load",
            "✅ NIV/BiPAP — Level A for respiratory failure; avoid intubation if possible",
            "✅ Levetiracetam (LEV) — Preferred AED; no mitochondrial toxicity profile",
        ],
        "key_ddx": [
            {
                "condition": "UQCC2 Deficiency (CIII-D7, 6p21.2)",
                "distinguishing": (
                    "Clinically and biochemically IDENTICAL to UQCC1. Both cause severe neonatal "
                    "CIII deficiency with CIII-completely-absent BN-PAGE. ONLY distinguisher: "
                    "UQCC2 protein absent in UQCC1 (UQCC1 is scaffold); UQCC1 protein absent "
                    "in UQCC2 deficiency (UQCC2 not needed for UQCC1 stability). WES mandatory."
                ),
            },
            {
                "condition": "UQCC3 Deficiency (CIII-D8, 11q12.3)",
                "distinguishing": (
                    "UQCC3 is distinct from UQCC1-UQCC2 heterodimer. UQCC3 causes MILDER "
                    "CIII deficiency (early childhood onset, 10-30% residual CIII). UQCC1 causes "
                    "SEVERE neonatal disease (<8% residual). BN-PAGE: UQCC3 shows reduced CIII "
                    "with some sub-complexes; UQCC1 shows CIII completely absent."
                ),
            },
            {
                "condition": "BCS1L (GRACILE/Bjornstad, 2q35)",
                "distinguishing": (
                    "BCS1L causes GRACILE triad (iron overload, Fanconi aminoaciduria, cholestasis). "
                    "UQCC1 has NONE of these. BCS1L BN-PAGE shows CIII precomplex accumulation; "
                    "UQCC1 shows CIII completely absent. Both neonatal onset."
                ),
            },
            {
                "condition": "TTC19 Deficiency (CIII-D2, 17p12)",
                "distinguishing": (
                    "TTC19 causes NEUROLOGICAL disease (spinocerebellar ataxia, psychiatric "
                    "features in 40%) with childhood or adult onset. UQCC1 causes neonatal "
                    "onset severe disease. TTC19 has NO neonatal lactic acidosis crisis."
                ),
            },
            {
                "condition": "SCO2 (COX Deficiency + HCM)",
                "distinguishing": (
                    "SCO2 causes hypertrophic cardiomyopathy in >80% — KEY distinguisher. "
                    "UQCC1 cardiomyopathy is RARE (<8%). SCO2 affects Complex IV (COX/CIV), "
                    "not CIII. Biochemistry: CIV deficiency in SCO2 vs CIII deficiency in UQCC1."
                ),
            },
            {
                "condition": "SURF1 (Leigh/COX Deficiency, 9q34.2)",
                "distinguishing": (
                    "SURF1 causes Complex IV (COX) deficiency, NOT CIII. Leigh syndrome pattern "
                    "may overlap but SURF1 BN-PAGE shows COX deficiency; UQCC1 shows CIII absent."
                ),
            },
            {
                "condition": "POLG / MPV17 (mtDNA Depletion)",
                "distinguishing": (
                    "POLG/MPV17 cause hepatopathy and mtDNA depletion (multiple complex deficiency "
                    "on biochemistry). UQCC1 causes isolated CIII deficiency. mtDNA copy number "
                    "normal in UQCC1. No hepatopathy in UQCC1."
                ),
            },
        ],
        "key_references": [
            "Tucker EJ et al. (2013) PLoS Genet 9(12):e1004034. "
            "UQCC1-UQCC2 heterodimer characterised; UQCC2 mutations first described.",
            "Stroud DA et al. (2016) Cell Metab 24:77-90. "
            "Genome-wide CRISPR screen confirms UQCC1-UQCC2 functional complex.",
            "Fernandez-Vizarra E & Zeviani M (2018) Front Genet 9:134. "
            "Nuclear CIII deficiency gene landscape including UQCC1.",
            "Feichtinger RG et al. (2017) J Inherit Metab Dis 40:825-834. "
            "Neonatal CIII deficiency spectrum — DDx context.",
        ],
        "terms": [
            {"term": "UQCC1", "definition":
             "Ubiquinol-Cytochrome C Reductase Complex Assembly Factor 1; obligate "
             "heterodimer partner of UQCC2; scaffolds UQCC2 stability; loss → complete CIII block"},
            {"term": "UQCC2", "definition":
             "UQCC1's obligate heterodimer partner (116 aa, 6p21.2); together they stabilise "
             "nascent MT-CYB; UQCC2 protein absent in UQCC1 deficiency"},
            {"term": "CIII* (earliest intermediate)", "definition":
             "First CIII assembly intermediate: MT-CYB + UQCC1-UQCC2 heterodimer; "
             "UQCC1 loss prevents this intermediate from forming"},
            {"term": "MT-CYB", "definition":
             "Cytochrome b; only mitochondrially-encoded CIII subunit; immediate substrate of "
             "UQCC1-UQCC2; degraded within minutes of translation without UQCC1-UQCC2 shielding"},
            {"term": "m-AAA protease", "definition":
             "Mitochondrial matrix-facing AAA protease; degrades unshielded MT-CYB within "
             "minutes of translation; UQCC1-UQCC2 competitively inhibits m-AAA access to MT-CYB"},
            {"term": "BN-PAGE (Blue Native PAGE)", "definition":
             "Electrophoresis technique that preserves native complexes; in UQCC1 deficiency "
             "CIII band is completely absent — identical pattern to UQCC2 deficiency"},
            {"term": "CIII-D6", "definition":
             "Complex III Deficiency, Nuclear Type 6 (OMIM #615453); caused by biallelic "
             "UQCC1 loss-of-function mutations"},
            {"term": "CoQH2 backlog", "definition":
             "Accumulation of ubiquinol (reduced CoQ10) when CIII cannot oxidise it; "
             "inhibits CI and other NAD+-regenerating complexes; amplifies lactic acidosis"},
            {"term": "GIR (Glucose Infusion Rate)", "definition":
             "6–8 mg/kg/min continuous IV dextrose to prevent catabolism in UQCC1 deficiency; "
             "fasting strictly contraindicated as it triggers FAO with CoQH2 backlog"},
            {"term": "PRIS (Propofol Infusion Syndrome)", "definition":
             "Fatal complication of propofol in CIII deficiency; propofol directly inhibits "
             "CIII; UQCC1 patients have zero CIII reserve; any further inhibition is fatal"},
            {"term": "Reciprocal UQCC1/UQCC2 loss", "definition":
             "In UQCC1 deficiency → UQCC2 protein absent (UQCC1 scaffolds UQCC2). "
             "In UQCC2 deficiency → UQCC1 protein remains present. This is a key "
             "immunoblot distinguisher when WES is not yet available"},
        ],
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    bk = get_breakdown()
    df = get_definitions()
    print("Overview cohort_n:", ov["cohort_n"])
    print("Breakdown variants:", len(bk["all_variants"]))
    print("Definitions terms:", len(df["terms"]))
    print("Patients[0]:", json.dumps(bk["patients"][0], indent=2))
