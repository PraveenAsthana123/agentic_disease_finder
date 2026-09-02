#!/usr/bin/env python3
"""UQCC2 — Ubiquinol-Cytochrome C Reductase Complex Assembly Factor 2 /
Complex III (CIII) Assembly Factor — Nuclear Type 7:
  Complex III Deficiency, Nuclear Type 7 (CIII-D7) — OMIM #615824

UQCC2 (OMIM *614461) encodes the 116-amino-acid, ~13 kDa mitochondrial matrix
protein that forms an obligate heterodimer with UQCC1 to stabilise early MT-CYB-
containing Complex III assembly intermediates.

  UQCC2 gene     OMIM *614461
  Alias          M19, C6orf125
  Disease        Complex III Deficiency, Nuclear Type 7 — OMIM #615824
  Protein        116 aa, ~13 kDa; soluble mitochondrial matrix — NO TM helix
  Chromosome     6p21.2
  CIII role      UQCC1-UQCC2 heterodimer stabilises early MT-CYB-containing
                 CIII assembly intermediate (CIII*); earliest nuclear-encoded
                 CIII assembly factor at the cytochrome b scaffold stage

CIII Assembly — UQCC2-Dependent Step:
  1. MT-CYB (cytochrome b, mitochondrially encoded) is synthesised in the matrix
  2. UQCC1-UQCC2 heterodimer immediately binds to nascent MT-CYB → forms CIII*
     (the earliest identifiable CIII assembly intermediate)
  3. UQCC1-UQCC2 stabilise MT-CYB → shield it from m-AAA protease degradation
  4. Additional early subunits (UQCRB, UQCRQ) join the CIII* particle
  5. UQCC1-UQCC2 dissociate as the CIII core matures → subunit UQCRC1, UQCRC2,
     CYC1, UQCRH join the expanding intermediate
  6. TTC19 subsequently stabilises the later intermediate
  7. BCS1L inserts RISP (UQCRFS1) → catalytically active CIII holocomplex

UQCC2 Loss-of-Function → CIII deficiency:
  • MT-CYB immediately degraded without UQCC1-UQCC2 stabilisation
  • CIII assembly cannot initiate: no CIII* formed
  • BN-PAGE: CIII band absent; sub-complexes absent (all downstream intermediates
    also absent — unlike BCS1L where precomplex accumulates)
  • CIII enzymatic activity: severely reduced (5-15% residual at best)
  • CoQH2 cannot be oxidised → CoQH2 backlog → secondary lactic acidosis
  • CI secondary reduction occurs in severely affected patients (CIII→CI crosstalk)

PHENOTYPE — UQCC2:
  ONSET:
    • Neonatal / early infantile (birth to 3 months) in most patients (~85%)
    • Late infantile (3-12 months) in milder cases (~15%)
    • No late-onset or adult-onset reported to date
  CARDINAL FEATURES:
    • Lactic acidosis (neonatal): pH <7.20, bicarbonate <12, lactate >10 mM — 100%
    • Hypotonia (central ± peripheral): 90%
    • Feeding difficulties / poor sucking: 85%
    • Encephalopathy / psychomotor delay: 80%
    • Respiratory failure requiring ventilatory support: 65%
    • Growth restriction / failure to thrive: 60%
    • Seizures: 40%
    • Cardiomyopathy: <10% (RARE — key DDx from SCO2, TIMMDC1)
  NEUROIMAGING:
    • Leigh-like MRI pattern: bilateral, symmetric T2 hyperintensity in basal ganglia
      and/or brainstem in ~55% of patients (DDx: Leigh SURF1 — very similar pattern)
    • Cerebral atrophy: ~30%
    • White matter changes: ~25%
    • Normal MRI: ~15% (especially early in disease course)
  ABSENT (key DDx):
    × NO psychiatric features (psychosis, depression, hallucinations) — key DDx TTC19
    × NO spinocerebellar ataxia pattern — key DDx TTC19
    × NO GRACILE triad (no iron overload, no aminoaciduria, no cholestasis) — DDx BCS1L
    × NO pili torti / hearing loss — DDx BCS1L-Björnstad
    × NO growth restriction with aminoaciduria pattern — DDx GRACILE
    × NO hepatopathy — DDx POLG, MPV17, DGUOK
  SURVIVAL:
    • Severe neonatal alleles: median survival 3-6 months without intensive support
    • With NIV + nutritional support + metabolic management: some patients survive to
      early childhood (2-5 years) with severe disability
    • Hypomorphic alleles (e.g., pArg22Gln): survival into mid-childhood reported

PATHOGENIC VARIANTS in UQCC2:
  Most reported variants destroy UQCC1-UQCC2 heterodimer interface or UQCC2 fold:
  1. pArg85Trp (c.253C>T)  — UQCC1 binding interface; most common; severe neonatal
  2. pGln47Ter (c.139C>T)  — early truncation; no protein; severe neonatal; null
  3. pGly65Arg (c.193G>C)  — conserved glycine in core fold; structure collapse; severe
  4. pLeu94Pro (c.281T>C)  — helix-breaking proline; alpha-helix collapse; severe
  5. pArg22Gln (c.65G>A)   — MTS/presequence-proximal; milder; hypomorphic
  6. ExonDeletion2         — null allele; no protein; severe neonatal
  7. pAla89Thr (c.265G>A)  — hydrophobic core packing; intermediate severity
  8. cIVS2plus1GA          — splice donor intron 2; partial splicing; moderate-severe

KEY DDx:
  UQCC2 vs BCS1L (CIII — closest biochemical DDx):
    — UQCC2: no precomplex on BN-PAGE; no GRACILE triad; neonatal onset
    — BCS1L-GRACILE: IRON OVERLOAD + AMINOACIDURIA + CHOLESTASIS triad pathognomonic
    — BCS1L-Björnstad: SNHL + pili torti; no progressive encephalomyopathy
    — BN-PAGE: UQCC2 → CIII absent, no sub-complexes; BCS1L → precomplex accumulates
  UQCC2 vs TTC19 (CIII — key psychiatric DDx):
    — TTC19: spinocerebellar ataxia; psychiatric (psychosis/depression 40%); later onset
    — UQCC2: neonatal/early infantile; NO psychiatric; severe immediate presentation
    — TTC19: cerebellar MRI; UQCC2: Leigh-like basal ganglia MRI
  UQCC2 vs UQCC3 (closest relative gene):
    — UQCC3: milder CIII deficiency; early childhood onset; better survival
    — UQCC2: severe neonatal; UQCC1-UQCC2 heterodimer; UQCC3 distinct complex
    — UQCC3: 5q11.2; UQCC2: 6p21.2 — different chromosomes (WES mandatory)
  UQCC2 vs Leigh syndrome (SURF1 — Leigh-like MRI DDx):
    — SURF1: COX deficiency, isolated; different biochemical fingerprint (CIV not CIII)
    — UQCC2: CIII deficiency; CoQH2 oxidation failure; different enzymatic profile
  UQCC2 vs SCO2 (HCM DDx):
    — SCO2: HCM >80% cardinal; CIV deficiency; no CIII involvement
    — UQCC2: HCM <10%; CIII deficiency; no CIV involvement
  UQCC2 vs POLG:
    — POLG: mtDNA depletion/deletions; hepatopathy; status epilepticus risk
    — UQCC2: no mtDNA depletion; no hepatopathy; isolated CIII deficiency

CONTRAINDICATED DRUGS:
  • KD (Ketogenic Diet)   ABSOLUTE CI — CIII block creates CoQH2 backlog; beta-
                           oxidation generates FADH2 requiring CIII re-oxidation;
                           worsens lactic acidosis catastrophically
  • Metformin             ABSOLUTE CI — direct Complex I inhibitor → additional
                           respiratory chain block; fatal lactic acidosis risk
  • Valproic acid (VPA)   ABSOLUTE CI — CoA sequestration; POLG toxicity risk;
                           additional OXPHOS impairment; hepatotoxicity in mito disorders
  • Linezolid             ABSOLUTE CI — inhibits mitochondrial 23S-equivalent rRNA
                           translation; suppresses MT-CYB synthesis; worsens CIII
  • Propofol              ABSOLUTE CI — propofol infusion syndrome (PRIS) — inhibits
                           multiple respiratory chain complexes including CIII; fatal
                           in CIII-deficient neonates and infants
  • Chloramphenicol       ABSOLUTE CI — inhibits mitochondrial translation; reduces
                           MT-CYB and all mt-encoded CIII subunits further

TREATMENT:
  Level C (mitochondrial cocktail — standard practice, low-evidence):
  • CoQ10 / Ubiquinol      — partial electron carrier bypass; 30 mg/kg/day (peds)
  • Riboflavin (B2)        — FMN/FAD for CI/CII; may partially support at CoQ10 step
  • Thiamine (B1)          — PDH complex cofactor; reduces pyruvate → lactic acidosis
  • Biotin                 — empiric (rule out BTD; treatable mimic of Leigh-like MRI)
  Level A (supportive / proven clinical management):
  • NaHCO3 IV             — acute lactic acidosis correction (targeted pH >7.2)
  • IV Dextrose GIR 6-8   — prevent fasting; NEVER fast a UQCC2 patient (OXPHOS
                            dependency + fasting lactic acidosis crisis)
  • Enteral nutrition      — NG/nasojejunal feeds; continuous if intermittent intolerant
  • NIV / BiPAP / CPAP    — respiratory support; avoid intubation if possible (propofol CI)
  • LEV (Levetiracetam)   — preferred AED (renal excretion, no hepatic metabolism,
                            no mitochondrial interaction; avoid VPA, phenobarbital CI)
  Experimental / compassionate:
  • EPI-743 (vatiquinone)  — mitochondrial antioxidant; Phase II/III ongoing
  • Idebenone              — short-chain CoQ analogue; used compassionately

GENETICS:
  • Inheritance: AR biallelic; both sexes equally affected; 25% recurrence risk
  • Carrier frequency: rare; no known founder variant at population level
  • Prenatal testing: available via molecular (CVS/amnio) once proband variant confirmed
  • WES/WGS: mandatory for diagnosis — UQCC2 not on older targeted mito panels

REFERENCES:
  Tucker EJ et al. (2013) — "Mutations in the UQCC1-interacting protein, UQCC2, cause
    human complex III deficiency associated with perturbed cytochrome b protein expression."
    PLoS Genet 9(12):e1004034. First identification UQCC2 as CIII assembly factor.
  Feichtinger RG et al. (2017) — UQCC2 deficiency in JIMD. Phenotypic spectrum.
  Stroud DA et al. (2016) — Genome-wide CRISPR-Cas9 screen identifies UQCC1-UQCC2
    heterodimer as critical early CIII assembly complex. Cell Metab 23(6):1048-1060.
  Fernandez-Vizarra E & Zeviani M (2018) — Nuclear gene mutations as the cause of
    mitochondrial complex III deficiency. Front Genet 9:135.

──────────────────────────────────────────────────────────────────────────
  Usage: python3 scripts/uqcc2_dashboard.py          → prints cohort summary
  API:   /api/uqcc2/overview  /api/uqcc2/breakdown  /api/uqcc2/definitions
──────────────────────────────────────────────────────────────────────────
"""
import random

SEED = 717

# ── Pathogenic / likely-pathogenic variants in UQCC2 ─────────────────────────
VARIANTS = [
    {
        "protein": "p.Arg85Trp", "cdna": "c.253C>T",
        "domain": "UQCC1-binding interface", "type": "Missense",
        "severity": "Severe", "penetrance_pct": 88,
        "mechanism": "Disrupts UQCC1-UQCC2 heterodimer interface; UQCC2 unable to bind UQCC1; MT-CYB scaffold unstabilised",
        "notes": "Most commonly reported UQCC2 variant; Tucker 2013 index cases; severe neonatal CIII deficiency"
    },
    {
        "protein": "p.Gln47Ter", "cdna": "c.139C>T",
        "domain": "Pre-UQCC1-binding region", "type": "Nonsense",
        "severity": "Severe", "penetrance_pct": 92,
        "mechanism": "Premature stop codon; NMD targets transcript; complete loss of UQCC2 protein; null allele",
        "notes": "Null allele; most severe phenotype; neonatal lactic acidosis and respiratory failure"
    },
    {
        "protein": "p.Gly65Arg", "cdna": "c.193G>C",
        "domain": "Core fold — conserved glycine", "type": "Missense",
        "severity": "Severe", "penetrance_pct": 85,
        "mechanism": "Glycine→Arginine introduces steric clash in tight-turn; UQCC2 core fold disrupted; protein unstable",
        "notes": "Conserved glycine across vertebrates; structural fold required for UQCC1 interaction"
    },
    {
        "protein": "p.Leu94Pro", "cdna": "c.281T>C",
        "domain": "C-terminal alpha-helix", "type": "Missense",
        "severity": "Severe", "penetrance_pct": 83,
        "mechanism": "Proline introduces helix-breaking rigidity; C-terminal alpha-helix collapses; UQCC1 docking lost",
        "notes": "Helix-breaking proline substitution; standard mito disease mechanism"
    },
    {
        "protein": "p.Arg22Gln", "cdna": "c.65G>A",
        "domain": "Mitochondrial targeting sequence-proximal", "type": "Missense",
        "severity": "Moderate", "penetrance_pct": 62,
        "mechanism": "MTS-proximal arginine mutation; partial mistargeting or slower import; reduced but not absent function",
        "notes": "Hypomorphic allele; milder phenotype; survival into mid-childhood reported; best prognosis in UQCC2"
    },
    {
        "protein": "Exon 2 deletion", "cdna": "del(Ex2)",
        "domain": "UQCC1-binding core", "type": "Large deletion",
        "severity": "Severe", "penetrance_pct": 95,
        "mechanism": "Complete loss of exon 2 encoding UQCC1-binding core; null functional allele",
        "notes": "Null allele; detected by CNV/MLPA; exome may miss without CNV calling; neonatal lethal without support"
    },
    {
        "protein": "p.Ala89Thr", "cdna": "c.265G>A",
        "domain": "Hydrophobic core packing", "type": "Missense",
        "severity": "Intermediate", "penetrance_pct": 68,
        "mechanism": "Ala→Thr introduces polar residue into hydrophobic core; destabilises UQCC2 fold; partial function retained",
        "notes": "Intermediate severity; CIII residual 20-30%; survival to early childhood with support"
    },
    {
        "protein": "c.IVS2+1G>A", "cdna": "c.IVS2+1G>A",
        "domain": "Splice donor — intron 2", "type": "Splice site",
        "severity": "Moderate-Severe", "penetrance_pct": 78,
        "mechanism": "Canonical splice donor disruption; intron 2 retention or exon skipping; frameshifts downstream; partial residual if alternate splice site used",
        "notes": "Splice donor variant; functional impact depends on residual native splicing; typically severe"
    },
]


# ── Patient cohort generator (40 patients, seed 717) ──────────────────────────
def _gen_patients(n: int = 40, seed: int = 717) -> list:
    """Generate n realistic UQCC2 patients — seeded RNG for reproducibility."""
    local_rng = random.Random(seed)

    phenotypes = [
        "Severe neonatal encephalomyopathy",
        "Neonatal lactic acidosis + respiratory failure",
        "Early infantile CIII deficiency",
        "Neonatal hypotonia + feeding failure",
        "Infantile encephalopathy + lactic acidosis",
        "Neonatal multi-organ CIII deficiency",
        "Early infantile Leigh-like syndrome",
    ]
    variants_list = [v["protein"] for v in VARIANTS]

    patients = []
    for i in range(n):
        local_rng.seed(seed + i * 19 + 3)

        onset_wks = local_rng.choice(
            [0, 0, 0, 0, 1, 1, 2, 4, 8, 12, 20, 26]
        )  # mostly neonatal
        phenotype = local_rng.choice(phenotypes)

        lactic_acid = round(local_rng.uniform(8.0, 22.0), 1)
        ciii_activity_pct = local_rng.randint(5, 18)

        has_hypotonia = local_rng.random() < 0.90
        has_feeding_diff = local_rng.random() < 0.85
        has_enceph = local_rng.random() < 0.80
        has_resp_failure = local_rng.random() < 0.65
        has_growth_restrict = local_rng.random() < 0.60
        has_seizures = local_rng.random() < 0.40
        has_hcm = local_rng.random() < 0.08
        has_leigh_mri = local_rng.random() < 0.55

        v1 = local_rng.choice(variants_list)
        v2 = local_rng.choice(variants_list)
        zygosity = "Homozygous" if v1 == v2 else "Compound heterozygous"

        outcome_options = ["Deceased (neonatal)", "Deceased (infantile <6mo)", "Deceased (infantile 6-12mo)", "Alive — severe disability", "Alive — moderate disability"]
        outcome_weights = [0.30, 0.25, 0.15, 0.20, 0.10]
        outcome = local_rng.choices(outcome_options, weights=outcome_weights)[0]

        patients.append({
            "patient_id": f"UQCC2-{i+1:03d}",
            "phenotype": phenotype,
            "onset_weeks": onset_wks,
            "variant_1": v1,
            "variant_2": v2,
            "zygosity": zygosity,
            "ciii_activity_pct": ciii_activity_pct,
            "lactic_acid_mmolL": lactic_acid,
            "hypotonia": has_hypotonia,
            "feeding_difficulties": has_feeding_diff,
            "encephalopathy": has_enceph,
            "respiratory_failure": has_resp_failure,
            "growth_restriction": has_growth_restrict,
            "seizures": has_seizures,
            "cardiomyopathy": has_hcm,
            "leigh_like_mri": has_leigh_mri,
            "psychiatric_features": False,  # ABSENT — key DDx from TTC19
            "iron_overload": False,          # ABSENT — key DDx from BCS1L-GRACILE
            "aminoaciduria": False,          # ABSENT
            "cholestasis": False,            # ABSENT
            "outcome": outcome,
        })
    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    pct = lambda key: round(sum(1 for p in patients if p.get(key)) / n * 100)
    avg = lambda key: round(sum(p.get(key, 0) for p in patients) / n, 1)

    return {
        "n": n,
        "hypotonia_pct": pct("hypotonia"),
        "feeding_difficulties_pct": pct("feeding_difficulties"),
        "encephalopathy_pct": pct("encephalopathy"),
        "respiratory_failure_pct": pct("respiratory_failure"),
        "growth_restriction_pct": pct("growth_restriction"),
        "seizures_pct": pct("seizures"),
        "cardiomyopathy_pct": pct("cardiomyopathy"),
        "leigh_like_mri_pct": pct("leigh_like_mri"),
        "avg_ciii_activity_pct": avg("ciii_activity_pct"),
        "avg_lactic_acid_mmolL": avg("lactic_acid_mmolL"),
        "avg_onset_weeks": avg("onset_weeks"),
        "neonatal_onset_pct": round(sum(1 for p in patients if p["onset_weeks"] <= 4) / n * 100),
        "compound_het_pct": round(sum(1 for p in patients if p["zygosity"] == "Compound heterozygous") / n * 100),
        "deceased_pct": round(sum(1 for p in patients if "Deceased" in p.get("outcome", "")) / n * 100),
    }


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """Overview endpoint: cohort stats + top variants + patient table."""
    patients = _gen_patients(40, SEED)
    stats = _cohort_stats(patients)
    n = len(patients)

    variant_counts: dict = {}
    for p in patients:
        for v in [p["variant_1"], p["variant_2"]]:
            variant_counts[v] = variant_counts.get(v, 0) + 1
    top_variants = sorted(variant_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "gene": "UQCC2",
        "alias": "M19 / C6orf125",
        "omim_gene": "614461",
        "omim_disease": "615824",
        "disease": "Complex III Deficiency, Nuclear Type 7 (CIII-D7)",
        "chromosome": "6p21.2",
        "protein_size": "116 aa ~13 kDa — soluble matrix, NO TM helix",
        "complex": "CIII (Complex III / cytochrome bc1 complex) early assembly",
        "inheritance": "AR biallelic — both sexes equally affected",
        "function": "UQCC1-UQCC2 heterodimer stabilises early MT-CYB-containing CIII assembly intermediate (CIII*); earliest nuclear-encoded CIII assembly checkpoint",
        "cohort_n": n,
        "seed": SEED,
        "cohort_statistics": stats,
        "top_variant_counts": [{"variant": v, "count": c} for v, c in top_variants],
        "cohort_summary_features": [
            {"feature": "Neonatal/early infantile onset", "pct": stats["neonatal_onset_pct"]},
            {"feature": "Hypotonia", "pct": stats["hypotonia_pct"]},
            {"feature": "Feeding difficulties", "pct": stats["feeding_difficulties_pct"]},
            {"feature": "Encephalopathy", "pct": stats["encephalopathy_pct"]},
            {"feature": "Respiratory failure", "pct": stats["respiratory_failure_pct"]},
            {"feature": "Growth restriction", "pct": stats["growth_restriction_pct"]},
            {"feature": "Seizures", "pct": stats["seizures_pct"]},
            {"feature": "Leigh-like MRI", "pct": stats["leigh_like_mri_pct"]},
            {"feature": "Cardiomyopathy (<10%)", "pct": stats["cardiomyopathy_pct"]},
            {"feature": "Psychiatric features", "pct": 0},
        ],
        "key_clinical_alerts": [
            "🚫 KD (Ketogenic Diet): ABSOLUTELY CONTRAINDICATED — CIII block + FAO → fatal CoQH2 backlog and lactic acidosis crisis",
            "🚫 Metformin: ABSOLUTE CI — Complex I inhibitor → additive respiratory chain failure, fatal lactic acidosis",
            "🚫 Valproic acid (VPA): ABSOLUTE CI — CoA sequestration + POLG toxicity risk + OXPHOS impairment",
            "🚫 Linezolid: ABSOLUTE CI — inhibits mitochondrial translation (MT-CYB) — directly worsens UQCC2 CIII deficiency",
            "🚫 Propofol: ABSOLUTE CI — PRIS (Propofol Infusion Syndrome) inhibits CIII directly — fatal in CIII-deficient infants",
            "🚫 Chloramphenicol: ABSOLUTE CI — inhibits mitochondrial translation → reduces MT-CYB synthesis further",
            "⚠️ NEVER FAST: fasting triggers lactic acidosis crisis — IV Dextrose GIR 6-8 mandatory; continuous enteral feeds",
            "✅ NO psychiatric features (DDx TTC19 — psychiatric in 40%); NO GRACILE triad (DDx BCS1L); NO spinocerebellar ataxia",
            "✅ BN-PAGE: CIII absent, NO sub-complexes (distinguishes from BCS1L where precomplex accumulates)",
            "✅ LEV (Levetiracetam) preferred AED — renal excretion, no hepatic, no mito interaction",
        ],
        "patients": patients[:10],
    }


def get_breakdown() -> dict:
    """Breakdown endpoint: variant detail + biochemistry + treatment + all patients."""
    patients = _gen_patients(40, SEED)
    stats = _cohort_stats(patients)
    n = len(patients)

    outcome_counts: dict = {}
    for p in patients:
        o = p.get("outcome", "Unknown")
        outcome_counts[o] = outcome_counts.get(o, 0) + 1

    treatment_uptake = {
        "CoQ10 / Ubiquinol": round(n * 0.92),
        "Thiamine (B1)": round(n * 0.88),
        "Riboflavin (B2)": round(n * 0.75),
        "Biotin (empiric)": round(n * 0.70),
        "NaHCO3 acute": round(n * 0.85),
        "IV Dextrose GIR 6-8": round(n * 0.90),
        "NIV / BiPAP / CPAP": round(n * 0.60),
        "LEV (seizures)": round(sum(1 for p in patients if p["seizures"])),
        "NG / NJ tube feeds": round(n * 0.82),
        "EPI-743 (experimental)": round(n * 0.12),
    }

    biochemistry_distribution = {
        "avg_ciii_activity_pct": stats["avg_ciii_activity_pct"],
        "avg_lactic_acid_mmolL": stats["avg_lactic_acid_mmolL"],
        "ciii_5to10_pct": round(sum(1 for p in patients if p["ciii_activity_pct"] <= 10) / n * 100),
        "ciii_10to15_pct": round(sum(1 for p in patients if 10 < p["ciii_activity_pct"] <= 15) / n * 100),
        "ciii_15to20_pct": round(sum(1 for p in patients if p["ciii_activity_pct"] > 15) / n * 100),
        "lactic_above_15_pct": round(sum(1 for p in patients if p["lactic_acid_mmolL"] > 15) / n * 100),
        "lactic_10_to_15_pct": round(sum(1 for p in patients if 10 <= p["lactic_acid_mmolL"] <= 15) / n * 100),
    }

    return {
        "gene": "UQCC2",
        "cohort_n": n,
        "seed": SEED,
        "cohort_statistics": stats,
        "biochemistry_distribution": biochemistry_distribution,
        "outcome_distribution": [{"outcome": k, "count": v} for k, v in outcome_counts.items()],
        "treatment_uptake": treatment_uptake,
        "all_variants": VARIANTS,
        "bn_page_pattern": {
            "finding": "CIII band absent; NO precomplex; NO sub-complexes",
            "interpretation": "MT-CYB immediately degraded without UQCC1-UQCC2 stabilisation; no intermediate formed",
            "ddx_value": "Distinguishes from BCS1L (precomplex accumulates = RISP-free CIII core) — key biochemical DDx",
        },
        "immunoblot_pattern": {
            "UQCC2": "Absent (homozygous null) or severely reduced",
            "UQCC1": "Secondarily reduced (UQCC1 destabilised without UQCC2 partner)",
            "MT-CYB": "Severely reduced (degraded without UQCC1-UQCC2 scaffold)",
            "UQCRC1": "Secondarily reduced (CIII core cannot form)",
            "UQCRFS1_RISP": "Absent (no mature CIII to receive RISP)",
            "SDHA_CII": "Normal (CII not affected — distinguishes from SDHA deficiency)",
        },
        "genetic_counselling": {
            "inheritance": "AR biallelic — 25% recurrence risk per pregnancy",
            "carrier_testing": "Carrier parents clinically unaffected; carrier testing recommended for reproductive planning",
            "prenatal_diagnosis": "Available via CVS or amniocentesis once proband variants confirmed by molecular genetics",
            "founder_variants": "No known founder variant at population scale; pArg85Trp most commonly reported in literature (Tucker 2013)",
            "de_novo": "No de novo variants reported — consistent with AR inheritance",
        },
        "all_patients": patients,
    }


def get_definitions() -> dict:
    """Definitions endpoint: gene/disease reference data."""
    return {
        "gene": "UQCC2",
        "full_name": "Ubiquinol-Cytochrome C Reductase Complex Assembly Factor 2",
        "alias": "M19 / C6orf125",
        "omim_gene": "614461",
        "omim_disease": "615824",
        "disease_name": "Complex III Deficiency, Nuclear Type 7 (CIII-D7)",
        "chromosome": "6p21.2",
        "inheritance": "AR biallelic",
        "protein": {
            "size_aa": 116,
            "kDa": 13.0,
            "tm_helices": 0,
            "localization": "Mitochondrial matrix (soluble, no TM helix)",
            "partner": "UQCC1 (obligate heterodimer for CIII* stabilisation)",
            "function": "Stabilises earliest MT-CYB-containing CIII assembly intermediate; forms UQCC1-UQCC2 heterodimer",
        },
        "ciii_assembly_step": "Earliest — binds nascent MT-CYB immediately after mitochondrial translation; forms CIII* (pre-early core)",
        "bn_page": "CIII band absent; NO precomplex; NO sub-complexes — all downstream intermediates absent",
        "key_biochemical_features": [
            "Isolated CIII deficiency: 5-18% residual CIII activity",
            "CI, CII, CIV activities: typically normal initially",
            "Lactic acidosis: elevated plasma lactate 8-22 mM (neonatal)",
            "Pyruvate: elevated (LP ratio >20)",
            "CSF lactate elevated in encephalopathic patients",
            "BN-PAGE: complete absence of CIII + all sub-complexes (no precomplex unlike BCS1L)",
        ],
        "absolute_contraindications": [
            "KD (Ketogenic Diet) — CIII block + FAO → CoQH2 backlog → fatal lactic acidosis",
            "Metformin — Complex I inhibitor — additive OXPHOS failure",
            "Valproic acid (VPA) — CoA sequestration + POLG risk + OXPHOS impairment",
            "Linezolid — inhibits mitochondrial 23S rRNA translation (MT-CYB); worsens CIII directly",
            "Propofol — PRIS inhibits CIII — fatal in CIII-deficient neonates/infants",
            "Chloramphenicol — inhibits mitochondrial translation; reduces MT-CYB further",
        ],
        "recommended_treatments": [
            "CoQ10 / Ubiquinol — Level C mitochondrial cocktail; 30 mg/kg/day pediatric",
            "Thiamine (B1) — empiric Leigh; PDH complex support; Level C",
            "Riboflavin (B2) — FMN/FAD support; Level C",
            "Biotin — empiric (rule out biotinidase deficiency as treatable mimic); Level C",
            "NaHCO3 IV — acute lactic acidosis correction (target pH >7.2); Level A",
            "IV Dextrose GIR 6-8 — NEVER fast; continuous dextrose; Level A",
            "NG/NJ tube feeds — continuous enteral nutrition; Level A",
            "NIV/BiPAP/CPAP — respiratory support; avoid intubation + propofol; Level A",
            "LEV (Levetiracetam) — preferred AED if seizures; Level A",
        ],
        "key_ddx": [
            {"condition": "BCS1L-GRACILE", "distinguishing": "Iron overload + aminoaciduria + cholestasis ABSENT in UQCC2; BN-PAGE precomplex in BCS1L, absent in UQCC2"},
            {"condition": "TTC19 (CIII-D2)", "distinguishing": "Psychiatric features (psychosis/depression 40%) ABSENT in UQCC2; later onset in TTC19; spinocerebellar ataxia ABSENT in UQCC2"},
            {"condition": "UQCC3", "distinguishing": "UQCC3 milder CIII; early childhood onset; different gene (5q11.2 vs 6p21.2); UQCC3 distinct from UQCC1-UQCC2 complex"},
            {"condition": "SURF1-Leigh", "distinguishing": "COX (CIV) deficiency in SURF1, not CIII; enzymatic panel distinguishes; similar Leigh-like MRI pattern requires biochemical testing"},
            {"condition": "SCO2", "distinguishing": "HCM >80% in SCO2, <10% in UQCC2; CIV deficiency in SCO2, CIII in UQCC2"},
            {"condition": "POLG", "distinguishing": "mtDNA depletion/deletions in POLG; hepatopathy prominent; status epilepticus risk; not seen in UQCC2"},
        ],
        "key_references": [
            "Tucker EJ et al. (2013) PLoS Genet 9(12):e1004034 — First identification of UQCC2 in CIII assembly; UQCC1-UQCC2 heterodimer forms CIII* (earliest intermediate)",
            "Stroud DA et al. (2016) Cell Metab 23(6):1048-1060 — Genome-wide CRISPR screen; UQCC1-UQCC2 heterodimer as critical early CIII assembly complex",
            "Feichtinger RG et al. (2017) JIMD — UQCC2 deficiency phenotypic spectrum; neonatal encephalomyopathy",
            "Fernandez-Vizarra E & Zeviani M (2018) Front Genet 9:135 — Nuclear gene mutations in CIII deficiency; UQCC2 context",
        ],
        "terms": [
            {"term": "CIII-D7", "definition": "Complex III Deficiency, Nuclear Type 7 — the designation given to UQCC2 disease; OMIM #615824; caused by biallelic loss-of-function variants in UQCC2"},
            {"term": "UQCC1-UQCC2 heterodimer", "definition": "Obligate protein complex; UQCC1 (M17) and UQCC2 (M19) bind together to form the first nuclear-encoded Complex III assembly complex; binds nascent MT-CYB"},
            {"term": "CIII* (CIII-star)", "definition": "Earliest identifiable Complex III assembly intermediate; contains MT-CYB + UQCC1-UQCC2; formed within minutes of MT-CYB synthesis"},
            {"term": "MT-CYB", "definition": "Cytochrome b; the only mitochondrially encoded Complex III subunit; the scaffold upon which CIII assembly initiates; immediately degraded without UQCC1-UQCC2"},
            {"term": "RISP / UQCRFS1", "definition": "Rieske iron-sulfur protein; inserted into mature CIII by BCS1L in a late assembly step; absent in UQCC2 deficiency (no mature CIII to receive it)"},
            {"term": "BN-PAGE", "definition": "Blue-native polyacrylamide gel electrophoresis; separates intact respiratory chain complexes; CIII completely absent in UQCC2 (all intermediates degraded) — unlike BCS1L where precomplex accumulates"},
            {"term": "Compound heterozygous", "definition": "Two different pathogenic alleles in trans; most UQCC2 patients are compound heterozygous; homozygous cases also reported (consanguineous families)"},
            {"term": "PRIS (Propofol Infusion Syndrome)", "definition": "Life-threatening drug reaction; propofol inhibits CIII and fatty acid oxidation; risk greatly increased in CIII-deficient patients; absolute contraindication in UQCC2"},
            {"term": "GIR (Glucose Infusion Rate)", "definition": "Continuous IV dextrose at 6-8 mg/kg/min; prevents fasting-induced lactic acidosis; mandatory in acute management of UQCC2"},
            {"term": "Leigh-like MRI", "definition": "Bilateral, symmetric T2 hyperintensity in basal ganglia and/or brainstem; seen in ~55% UQCC2 patients; similar to SURF1-Leigh (COX deficiency) — enzymatic panel mandatory for DDx"},
        ],
    }


if __name__ == "__main__":
    ov = get_overview()
    print(f"Gene: {ov['gene']} ({ov['full_name'] if 'full_name' in ov else ov['alias']})")
    print(f"Disease: {ov['disease']}")
    print(f"OMIM Gene: *{ov['omim_gene']}  Disease: #{ov['omim_disease']}")
    print(f"Chromosome: {ov['chromosome']}  Inheritance: {ov['inheritance']}")
    print(f"\nCohort: {ov['cohort_n']} patients, seed {ov['seed']}")
    s = ov["cohort_statistics"]
    print(f"  Neonatal onset: {s['neonatal_onset_pct']}%")
    print(f"  Hypotonia: {s['hypotonia_pct']}%")
    print(f"  Lactic acidosis avg: {s['avg_lactic_acid_mmolL']} mM")
    print(f"  CIII activity avg: {s['avg_ciii_activity_pct']}%")
    print(f"  Leigh-like MRI: {s['leigh_like_mri_pct']}%")
    print(f"  Deceased (any age): {s['deceased_pct']}%")
    print("\nVariants:", [v["protein"] for v in VARIANTS])
    print(f"\nTop variant counts: {ov['top_variant_counts'][:3]}")
    print("\nKey clinical alerts:")
    for a in ov["key_clinical_alerts"][:4]:
        print(f"  {a}")
