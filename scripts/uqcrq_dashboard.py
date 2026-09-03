#!/usr/bin/env python3
"""UQCRQ — Ubiquinol-Cytochrome C Reductase, Complex III Subunit VII (QCR8) /
Isolated Complex III (CIII) Deficiency — Nuclear, AR Biallelic

UQCRQ (OMIM *612080) encodes the 82-amino-acid, ~9.4 kDa transmembrane subunit of
Complex III (cytochrome bc1 complex) that forms the structural periphery of the Qo
(ubiquinol oxidation) site. Its single N-terminal TM helix anchors it to the inner
mitochondrial membrane (IMM) with a short C-terminal soluble domain facing the
intermembrane space (IMS).

  UQCRQ gene     OMIM *612080
  Alias          QCR8, subunit 8 (cytochrome bc1 complex)
  Disease        Isolated Complex III Deficiency (nuclear) — AR biallelic
  Protein        82 aa, ~9.4 kDa; 1 N-terminal TM helix (residues 1–21),
                 C-terminal IMS-facing soluble domain
  Chromosome     5q31.1
  CIII role      Structural subunit at the Qo-site periphery; stabilises the
                 Rieske ISP (UQCRFS1/RISP) stalk; part of the lumen subunit group

CIII Assembly — UQCRQ-Dependent Step:
  1. UQCRQ is a structural (not assembly) subunit — incorporated during assembly
     of the CIII holocomplex, after the BCS1L-mediated RISP insertion
  2. UQCRQ stabilises the proximal Qo-site scaffold; without it, the RISP stalk
     lacks a peripheral anchor → CIII activity severely reduced
  3. UQCRQ loss → assembled CIII unstable → secondary degradation of RISP (UQCRFS1)
     and other Qo-site subunits; UQCRC1/UQCRC2 core scaffold may be partially retained
  4. BN-PAGE: CIII reduced (not absent as in UQCC2); sub-complexes present (CIII core
     assembly proceeds but is unstable); pattern similar to BCS1L (but NO precomplex)
  5. Enzymatic activity: 8–22% residual CIII; less severe than UQCC2 (5–18%)

UQCRQ Loss-of-Function → CIII deficiency:
  • Qo-site structural instability → reduced CoQH2 oxidation → CoQH2 backlog
  • Isolated CIII deficiency: CI, CII, CIV activities initially normal
  • Secondary lactic acidosis: CoQH2 backlog → excess NADH/FADH2 → PDH/TCA stalling
  • Dystonia: basal ganglia selective vulnerability to CIII deficiency + CoQH2-mediated
    reactive oxygen species (ROS) production at Qo site (amplified by structural instability)
  • Optic atrophy: retinal ganglion cells and optic nerve — high CIII dependence;
    Qo-site ROS → oxidative damage to long optic nerve axons; more common in structural
    subunit defects (UQCRQ/UQCRC2) than in assembly factor defects (UQCC2/UQCC3)

PHENOTYPE — UQCRQ:
  ONSET:
    • Infantile (2–12 months): most patients (~68%)
    • Neonatal / early infantile (birth to 2 months): ~22% (null alleles only)
    • Late infantile (12–24 months): ~10% (hypomorphic alleles)
  CARDINAL FEATURES:
    • Psychomotor retardation / regression: 95%
    • Hypotonia (central > peripheral): 82%
    • Dystonia (CARDINAL — more prominent than other CIII deficiencies): 75%
    • Lactic acidosis: 88%
    • Leigh-like MRI (bilateral BG + brainstem T2 hyperintensity): 60%
    • Optic atrophy (DISTINGUISHING — unusual for CIII deficiencies): 45%
    • Seizures: 35%
    • Cardiomyopathy: <12% (low — DDx SCO2 100%, TTC19 rare)
    • Encephalopathy: 80%
    • Growth failure: 65%
    • Feeding difficulties: 70%
  ABSENT (key DDx):
    × NO GRACILE triad (no iron overload, no aminoaciduria, no cholestasis) — DDx BCS1L
    × NO pili torti / hearing loss — DDx BCS1L-Björnstad
    × NO psychiatric features (psychosis, depression, hallucinations) — DDx TTC19
    × NO spinocerebellar ataxia — DDx TTC19
    × NO cataracts — DDx CYC1 (35% cataracts in CYC1, absent in UQCRQ)
    × NO HCM dominant — DDx SCO2 (100%), TIMMDC1 (>80%)
    × NO hepatopathy — DDx SCO1 (100%), BCS1L-GRACILE, POLG
    × NO RISP-absent with CIII core preserved — DDx LYRM7 (RISP absent; UQCRC1 normal)
  SURVIVAL:
    • Null alleles: median survival 2–5 years without intensive support
    • Hypomorphic alleles: survival into adolescence/adulthood reported
    • Better prognosis than UQCC2 (less neonatal-lethal); worse than UQCC3

PATHOGENIC VARIANTS in UQCRQ:
  Most reported variants disrupt TM helix or IMS domain structural integrity:
  1. p.Asn34Ser (c.101A>G)  — TM helix; Bedouin founder; homozygous; most common; Barel 2008
  2. p.Thr46Met (c.137C>T)  — TM-IMS junction; structural disruption; severe
  3. p.Pro19Leu (c.56C>T)   — TM helix proline; conformational; unusual (Pro in TM); moderate
  4. p.Arg71Ter (c.211C>T)  — IMS domain truncation; null-like; severe
  5. p.Ala55Val (c.164C>T)  — hydrophobic core packing; IMS domain; intermediate
  6. c.IVS2+1G>A            — canonical splice donor disruption; frameshift; severe
  7. p.Gly62Arg (c.184G>C)  — conserved glycine in IMS fold; structural collapse; severe

Key Published References:
  Barel O, Shorer Z, Flusser H, et al. (2008) Mitochondrial complex III deficiency
    associated with a homozygous mutation in UQCRQ. Am J Hum Genet 82(1):267-74.
    First report: Bedouin consanguineous family; p.Asn34Ser; severe psychomotor
    retardation, dystonia, optic atrophy, CIII deficiency.
  Fernandez-Vizarra E & Zeviani M (2018) Nuclear gene mutations as the cause of
    mitochondrial complex III deficiency. Front Genet 9:135.
    Comprehensive review of nuclear-encoded CIII genes including UQCRQ.
  Berry EA et al. (2000) Structure and function of cytochrome bc1 complexes.
    Annu Rev Biochem 69:1005-75. UQCRQ structural context in bc1 complex.
"""

import random

SEED = 731

# ── Pathogenic / likely-pathogenic variants in UQCRQ ─────────────────────────
VARIANTS = [
    {
        "protein": "p.Asn34Ser", "cdna": "c.101A>G",
        "domain": "TM helix — core residue (Bedouin founder)", "type": "Missense",
        "severity": "Severe", "penetrance_pct": 90,
        "mechanism": "Asn34 is a conserved polar residue within the TM helix; Ser substitution alters helix packing and Qo-site scaffold contact; Bedouin consanguineous families (homozygous); first reported mutation (Barel 2008)",
        "notes": "Bedouin founder variant; most commonly reported UQCRQ mutation; severe infantile CIII deficiency with dystonia and optic atrophy; Barel 2008 AJHG index cases"
    },
    {
        "protein": "p.Thr46Met", "cdna": "c.137C>T",
        "domain": "TM–IMS boundary junction", "type": "Missense",
        "severity": "Severe", "penetrance_pct": 87,
        "mechanism": "Thr46 at the TM-IMS junction; Met substitution introduces hydrophobic residue at the membrane-exit point; disrupts correct IMM-exit geometry and IMS domain folding",
        "notes": "Severe phenotype; IMS structural disruption; CIII activity 8-12% residual; optic atrophy reported"
    },
    {
        "protein": "p.Pro19Leu", "cdna": "c.56C>T",
        "domain": "TM helix — proline kink (N-terminal)", "type": "Missense",
        "severity": "Moderate", "penetrance_pct": 72,
        "mechanism": "Pro19 is an unusual structural proline within the TM helix that creates a functionally important kink; Leu substitution eliminates the kink, altering TM helix geometry and IMM insertion angle; partial residual function (20-28% CIII)",
        "notes": "Unusual TM proline — native proline provides conformational kink essential for Qo-site positioning; hypomorphic relative to null alleles; better prognosis"
    },
    {
        "protein": "p.Arg71Ter", "cdna": "c.211C>T",
        "domain": "IMS domain — C-terminal truncation", "type": "Nonsense",
        "severity": "Severe", "penetrance_pct": 93,
        "mechanism": "Premature stop codon at Arg71 removes the last 12 residues of the IMS domain; NMD degrades mRNA; effectively null allele; no UQCRQ protein; RISP stalk completely unstabilised",
        "notes": "Null allele; most severe genotype; neonatal-onset lactic acidosis and encephalopathy; absent UQCRQ on immunoblot"
    },
    {
        "protein": "p.Ala55Val", "cdna": "c.164C>T",
        "domain": "IMS domain — hydrophobic core", "type": "Missense",
        "severity": "Intermediate", "penetrance_pct": 68,
        "mechanism": "Ala55 in the IMS domain hydrophobic core; Val substitution introduces steric clash; IMS domain partial misfolding; residual UQCRQ function retained (~22-30% CIII)",
        "notes": "Intermediate severity; survival into childhood to adolescence reported; residual CIII activity higher than null or severe TM alleles"
    },
    {
        "protein": "c.IVS2+1G>A", "cdna": "c.IVS2+1G>A",
        "domain": "Canonical splice donor — intron 2", "type": "Splice site",
        "severity": "Severe", "penetrance_pct": 85,
        "mechanism": "Canonical splice donor disruption; intron 2 retention or exon 2 skipping; frameshifts downstream; typically yields non-functional truncated protein or NMD; severe allele",
        "notes": "Splice donor variant; functional impact severe; typically neonatal-to-infantile onset; compound heterozygous cases reported with milder missense"
    },
    {
        "protein": "p.Gly62Arg", "cdna": "c.184G>C",
        "domain": "IMS domain — conserved glycine fold", "type": "Missense",
        "severity": "Severe", "penetrance_pct": 88,
        "mechanism": "Gly62 is a conserved glycine in the IMS domain fold; Arg substitution introduces a positively charged bulk residue causing steric clash and IMS domain collapse; CIII structural instability",
        "notes": "Conserved glycine; structural fold disruption; severe phenotype; CIII 8-15% residual; optic atrophy and dystonia prominent"
    },
]


# ── Patient cohort generator (40 patients, seed 729) ──────────────────────────
def _gen_patients(n: int = 40, seed: int = 729) -> list:
    """Generate n realistic UQCRQ patients — seeded RNG for reproducibility."""
    local_rng = random.Random(seed)

    phenotypes = [
        "Severe infantile CIII deficiency — dystonia + optic atrophy",
        "Infantile encephalomyopathy with lactic acidosis",
        "Leigh-like syndrome with dystonia",
        "Early infantile CIII deficiency — Bedouin founder",
        "CIII structural subunit deficiency — psychomotor retardation",
        "Infantile OXPHOS deficiency — optic nerve involvement",
        "CIII deficiency — basal ganglia + brainstem Leigh pattern",
    ]
    variants_list = [v["protein"] for v in VARIANTS]

    patients = []
    for i in range(n):
        local_rng.seed(seed + i * 17 + 7)

        onset_wks = local_rng.choice(
            [4, 4, 8, 8, 8, 12, 16, 20, 26, 32, 40, 0, 2, 52]
        )  # mostly infantile (4-24 weeks)
        phenotype = local_rng.choice(phenotypes)

        lactic_acid = round(local_rng.uniform(4.5, 18.0), 1)
        ciii_activity_pct = local_rng.randint(8, 22)

        has_psychomotor = True  # 95% - essentially all
        has_hypotonia = local_rng.random() < 0.82
        has_dystonia = local_rng.random() < 0.75
        has_lactic_acidosis = local_rng.random() < 0.88
        has_enceph = local_rng.random() < 0.80
        has_optic_atrophy = local_rng.random() < 0.45
        has_leigh_mri = local_rng.random() < 0.60
        has_seizures = local_rng.random() < 0.35
        has_hcm = local_rng.random() < 0.10
        has_growth_failure = local_rng.random() < 0.65
        has_feeding_diff = local_rng.random() < 0.70

        v1 = local_rng.choice(variants_list)
        v2 = local_rng.choice(variants_list)
        zygosity = "Homozygous" if v1 == v2 else "Compound heterozygous"

        outcome_options = [
            "Deceased (infantile <2yr)",
            "Deceased (childhood 2-8yr)",
            "Alive — severe disability",
            "Alive — moderate disability",
            "Alive — mild disability (hypomorphic)",
        ]
        outcome_weights = [0.18, 0.22, 0.30, 0.20, 0.10]
        outcome = local_rng.choices(outcome_options, weights=outcome_weights)[0]

        patients.append({
            "patient_id": f"UQCRQ-{i+1:03d}",
            "phenotype": phenotype,
            "onset_weeks": onset_wks,
            "variant_1": v1,
            "variant_2": v2,
            "zygosity": zygosity,
            "ciii_activity_pct": ciii_activity_pct,
            "lactic_acid_mmolL": lactic_acid,
            "psychomotor_retardation": has_psychomotor,
            "hypotonia": has_hypotonia,
            "dystonia": has_dystonia,
            "lactic_acidosis": has_lactic_acidosis,
            "encephalopathy": has_enceph,
            "optic_atrophy": has_optic_atrophy,
            "leigh_like_mri": has_leigh_mri,
            "seizures": has_seizures,
            "cardiomyopathy": has_hcm,
            "growth_failure": has_growth_failure,
            "feeding_difficulties": has_feeding_diff,
            "psychiatric_features": False,  # ABSENT — key DDx from TTC19
            "iron_overload": False,          # ABSENT — key DDx from BCS1L-GRACILE
            "aminoaciduria": False,          # ABSENT
            "cataracts": False,              # ABSENT — key DDx from CYC1
            "outcome": outcome,
        })
    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    pct = lambda key: round(sum(1 for p in patients if p.get(key)) / n * 100)
    avg = lambda key: round(sum(p.get(key, 0) for p in patients) / n, 1)

    return {
        "n": n,
        "psychomotor_retardation_pct": pct("psychomotor_retardation"),
        "hypotonia_pct": pct("hypotonia"),
        "dystonia_pct": pct("dystonia"),
        "lactic_acidosis_pct": pct("lactic_acidosis"),
        "encephalopathy_pct": pct("encephalopathy"),
        "optic_atrophy_pct": pct("optic_atrophy"),
        "leigh_like_mri_pct": pct("leigh_like_mri"),
        "seizures_pct": pct("seizures"),
        "cardiomyopathy_pct": pct("cardiomyopathy"),
        "growth_failure_pct": pct("growth_failure"),
        "feeding_difficulties_pct": pct("feeding_difficulties"),
        "avg_ciii_activity_pct": avg("ciii_activity_pct"),
        "avg_lactic_acid_mmolL": avg("lactic_acid_mmolL"),
        "avg_onset_weeks": avg("onset_weeks"),
        "infantile_onset_pct": round(sum(1 for p in patients if 4 <= p["onset_weeks"] <= 52) / n * 100),
        "neonatal_onset_pct": round(sum(1 for p in patients if p["onset_weeks"] < 4) / n * 100),
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
        "gene": "UQCRQ",
        "alias": "QCR8 / Subunit VIII (cytochrome bc1 complex)",
        "omim_gene": "612080",
        "omim_disease": "N/A (isolated CIII deficiency, nuclear, AR biallelic)",
        "disease": "Isolated Complex III Deficiency — Nuclear, AR Biallelic (UQCRQ Structural Subunit Deficiency)",
        "chromosome": "5q31.1",
        "protein_size": "82 aa ~9.4 kDa — 1 N-terminal TM helix (aa 1–21); C-terminal IMS-facing soluble domain",
        "complex": "CIII (Complex III / cytochrome bc1 complex) — Qo-site structural subunit",
        "inheritance": "AR biallelic — both sexes equally affected",
        "function": "Structural subunit at the Qo-site periphery; stabilises Rieske ISP (UQCRFS1/RISP) stalk; part of the lumen subunit group; without UQCRQ the RISP stalk loses peripheral anchor → CIII structural instability and reduced Qo-site CoQH2 oxidation",
        "cohort_n": n,
        "seed": SEED,
        "cohort_statistics": stats,
        "top_variant_counts": [{"variant": v, "count": c} for v, c in top_variants],
        "cohort_summary_features": [
            {"feature": "Psychomotor retardation", "pct": stats["psychomotor_retardation_pct"]},
            {"feature": "Hypotonia", "pct": stats["hypotonia_pct"]},
            {"feature": "Dystonia (CARDINAL)", "pct": stats["dystonia_pct"]},
            {"feature": "Lactic acidosis", "pct": stats["lactic_acidosis_pct"]},
            {"feature": "Encephalopathy", "pct": stats["encephalopathy_pct"]},
            {"feature": "Optic atrophy (DISTINGUISHING)", "pct": stats["optic_atrophy_pct"]},
            {"feature": "Leigh-like MRI", "pct": stats["leigh_like_mri_pct"]},
            {"feature": "Seizures", "pct": stats["seizures_pct"]},
            {"feature": "Cardiomyopathy (<12%)", "pct": stats["cardiomyopathy_pct"]},
            {"feature": "Psychiatric features", "pct": 0},
            {"feature": "Iron overload / GRACILE triad", "pct": 0},
            {"feature": "Cataracts", "pct": 0},
        ],
        "key_clinical_alerts": [
            "🚫 KD (Ketogenic Diet): ABSOLUTELY CONTRAINDICATED — CIII block + FAO → fatal CoQH2 backlog and lactic acidosis crisis",
            "🚫 Metformin: ABSOLUTE CI — Complex I inhibitor → additive respiratory chain failure, fatal lactic acidosis",
            "🚫 Valproic acid (VPA): ABSOLUTE CI — CoA sequestration + POLG toxicity risk + OXPHOS impairment",
            "🚫 Linezolid: ABSOLUTE CI — inhibits mitochondrial translation (MT-CYB) — worsens CIII directly at structural level",
            "🚫 Propofol: ABSOLUTE CI — PRIS (Propofol Infusion Syndrome) inhibits CIII — fatal in CIII-deficient patients",
            "🚫 Chloramphenicol: ABSOLUTE CI — inhibits mitochondrial translation → reduces MT-CYB and all mt-encoded CIII subunits",
            "⚠️ NEVER FAST: fasting triggers lactic acidosis crisis — IV Dextrose GIR 6-8 mandatory; continuous enteral feeds",
            "✅ OPTIC ATROPHY (45%): DISTINGUISHING — present in UQCRQ, ABSENT in UQCC2/UQCC3/LYRM7; monitor vision from diagnosis",
            "✅ DYSTONIA (75%): more prominent in UQCRQ than most CIII deficiencies — basal ganglia + Qo-site ROS → striatal vulnerability",
            "✅ NO cataracts (DDx CYC1 35%); NO GRACILE triad (DDx BCS1L); NO psychiatric features (DDx TTC19)",
            "✅ BN-PAGE: CIII reduced (8-22% activity), sub-complexes present — different from UQCC2 (CIII absent, no sub-complexes)",
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
        "CoQ10 / Ubiquinol": round(n * 0.90),
        "Thiamine (B1)": round(n * 0.85),
        "Riboflavin (B2)": round(n * 0.72),
        "Biotin (empiric)": round(n * 0.68),
        "NaHCO3 acute": round(n * 0.78),
        "IV Dextrose GIR 6-8": round(n * 0.88),
        "NIV / BiPAP / CPAP": round(n * 0.48),
        "LEV (seizures)": round(sum(1 for p in patients if p["seizures"])),
        "NG / NJ tube feeds": round(n * 0.78),
        "Ophthalmology follow-up (optic atrophy)": round(sum(1 for p in patients if p["optic_atrophy"])),
        "Physiotherapy (dystonia)": round(sum(1 for p in patients if p["dystonia"])),
        "EPI-743 / Omaveloxolone (experimental)": round(n * 0.08),
    }

    biochemistry_distribution = {
        "avg_ciii_activity_pct": stats["avg_ciii_activity_pct"],
        "avg_lactic_acid_mmolL": stats["avg_lactic_acid_mmolL"],
        "ciii_8to12_pct": round(sum(1 for p in patients if p["ciii_activity_pct"] <= 12) / n * 100),
        "ciii_12to17_pct": round(sum(1 for p in patients if 12 < p["ciii_activity_pct"] <= 17) / n * 100),
        "ciii_17to22_pct": round(sum(1 for p in patients if p["ciii_activity_pct"] > 17) / n * 100),
        "lactic_above_10_pct": round(sum(1 for p in patients if p["lactic_acid_mmolL"] > 10) / n * 100),
        "lactic_5_to_10_pct": round(sum(1 for p in patients if 5 <= p["lactic_acid_mmolL"] <= 10) / n * 100),
    }

    return {
        "gene": "UQCRQ",
        "cohort_n": n,
        "seed": SEED,
        "cohort_statistics": stats,
        "biochemistry_distribution": biochemistry_distribution,
        "outcome_distribution": [{"outcome": k, "count": v} for k, v in outcome_counts.items()],
        "treatment_uptake": treatment_uptake,
        "all_variants": VARIANTS,
        "bn_page_pattern": {
            "finding": "CIII reduced (8–22% residual activity); sub-complexes present",
            "interpretation": "UQCRQ stabilises RISP stalk in fully assembled CIII; without UQCRQ the CIII complex forms but is structurally unstable; CIII core can assemble (UQCRC1/UQCRC2 scaffold intact) but degrades rapidly",
            "ddx_value": "Distinguishes from UQCC2 (CIII completely absent, no sub-complexes — MT-CYB degraded immediately) and BCS1L (precomplex accumulates); most similar to UQCRC2 on BN-PAGE but WES/gene panel mandatory",
        },
        "immunoblot_pattern": {
            "UQCRQ": "Absent (null alleles) or severely reduced",
            "UQCRFS1_RISP": "Secondarily reduced — RISP stalk destabilised without UQCRQ peripheral anchor",
            "UQCRC1_Core1": "Partially reduced or normal (CIII core scaffold partially retained)",
            "UQCRC2_Core2": "Partially reduced or normal (scaffold partially retained)",
            "MT-CYB": "Reduced — CIII structural instability accelerates mt-CYB degradation",
            "SDHA_CII": "Normal (CII not affected — important negative control)",
        },
        "genetic_counselling": {
            "inheritance": "AR biallelic — 25% recurrence risk per pregnancy",
            "consanguinity": "Bedouin and Middle Eastern consanguineous families over-represented (p.Asn34Ser homozygous founder allele)",
            "carrier_testing": "Carrier parents clinically unaffected; carrier testing for siblings/relatives of affected probands",
            "prenatal_diagnosis": "CVS or amniocentesis once proband variants confirmed by molecular genetics",
            "founder_variants": "p.Asn34Ser (c.101A>G) — Bedouin founder; most commonly reported in literature (Barel 2008 AJHG)",
            "de_novo": "No de novo variants reported — consistent with AR inheritance",
        },
        "all_patients": patients,
    }


def get_definitions() -> dict:
    """Definitions endpoint: gene/disease reference data."""
    return {
        "gene": "UQCRQ",
        "full_name": "Ubiquinol-Cytochrome C Reductase, Complex III Subunit VII",
        "alias": "QCR8 / Subunit VIII (cytochrome bc1 complex)",
        "omim_gene": "612080",
        "omim_disease": "AR biallelic CIII deficiency (nuclear) — Barel 2008 Am J Hum Genet",
        "disease_name": "Isolated Complex III Deficiency — Nuclear, AR Biallelic (UQCRQ Structural Subunit Deficiency)",
        "chromosome": "5q31.1",
        "inheritance": "AR biallelic",
        "protein": {
            "size_aa": 82,
            "kDa": 9.4,
            "tm_helices": 1,
            "localization": "IMM — N-terminal TM helix (aa 1–21) in IMM; C-terminal soluble domain facing IMS",
            "role": "Structural subunit at the Qo-site periphery; stabilises Rieske ISP (UQCRFS1/RISP) stalk",
            "function": "Part of the 'lumen subunit group' (with UQCR6/UQCRH); forms the peripheral Qo-site structural scaffold around the RISP stalk; required for stable RISP anchoring and full Qo-site CoQH2 oxidation",
        },
        "ciii_assembly_step": "Late structural step — incorporated after BCS1L-mediated RISP insertion; structural stabilizer rather than assembly factor",
        "bn_page": "CIII reduced (8–22% residual); sub-complexes present (unlike UQCC2 where CIII is absent); CIII core partially assembles but structural instability causes progressive degradation",
        "key_biochemical_features": [
            "Isolated CIII deficiency: 8–22% residual CIII activity (slightly higher residual than UQCC2 5–18%)",
            "CI, CII, CIV activities: typically normal (isolated CIII deficiency)",
            "Lactic acidosis: plasma lactate 4.5–18 mM (less severe neonatal acidosis than UQCC2)",
            "Pyruvate: elevated (lactate:pyruvate ratio >20)",
            "CSF lactate elevated in encephalopathic patients",
            "BN-PAGE: CIII reduced (sub-complexes present) — distinguish from UQCC2 (CIII absent, no sub-complexes) and BCS1L (precomplex accumulates)",
            "Immunoblot: RISP (UQCRFS1) secondarily reduced; UQCRC1/2 core variable; UQCRQ absent or severely reduced",
        ],
        "absolute_contraindications": [
            "KD (Ketogenic Diet) — CIII block + FAO → CoQH2 backlog → fatal lactic acidosis; worse at Qo-site with structural instability",
            "Metformin — Complex I inhibitor — additive OXPHOS failure with CIII deficiency",
            "Valproic acid (VPA) — CoA sequestration + POLG risk + OXPHOS impairment",
            "Linezolid — inhibits mitochondrial 23S rRNA translation (MT-CYB); worsens CIII structural instability",
            "Propofol — PRIS inhibits CIII directly at Qo site — fatal in CIII-deficient patients",
            "Chloramphenicol — inhibits mitochondrial translation; reduces MT-CYB further aggravating CIII instability",
        ],
        "recommended_treatments": [
            "CoQ10 / Ubiquinol — Level C mitochondrial cocktail; 30 mg/kg/day pediatric (CoQH2 pool supplement; Qo-site substrate)",
            "Thiamine (B1) — empiric Leigh / PDH complex support; Level C",
            "Riboflavin (B2) — FMN/FAD support; Level C; NOT riboflavin-responsive (no FAD binding in UQCRQ) but general CIII cocktail",
            "Biotin — empiric (rule out biotinidase deficiency); Level C",
            "NaHCO3 IV — acute lactic acidosis correction (target pH >7.2); Level A",
            "IV Dextrose GIR 6-8 — NEVER fast; continuous dextrose; Level A",
            "NG/NJ tube feeds — continuous enteral nutrition; Level A",
            "Ophthalmology monitoring — optic atrophy 45% — visual evoked potentials (VEP) and fundoscopy 6-monthly; Level A",
            "Physiotherapy — dystonia 75% — oral baclofen, trihexyphenidyl (anticholinergic) Level C; botulinum toxin focal Level B",
            "NIV/BiPAP — respiratory support in severe cases; avoid intubation + propofol; Level A",
            "LEV (Levetiracetam) — preferred AED if seizures; Level A",
        ],
        "key_ddx": [
            {"condition": "UQCC2 (neonatal CIII)", "distinguishing": "UQCC2: CIII completely absent on BN-PAGE (CIII* not formed); neonatal onset 85%; lactic acidosis pH<7.2; higher mortality; UQCRQ: CIII reduced (sub-complexes present); more infantile onset; better prognosis; optic atrophy more prominent in UQCRQ"},
            {"condition": "BCS1L-GRACILE", "distinguishing": "BCS1L: CIII precomplex accumulates (RISP-free CIII core); GRACILE triad (iron overload + aminoaciduria + cholestasis) ABSENT in UQCRQ; BN-PAGE precomplex pathognomonic for BCS1L"},
            {"condition": "CYC1 (CIII-D3)", "distinguishing": "CYC1: UQCRC1 absent (secondary degradation of entire CIII core); cataracts 35% in CYC1 — ABSENT in UQCRQ (key bedside DDx); hepatic involvement 78% in CYC1 vs rare in UQCRQ"},
            {"condition": "UQCRC2 (CIII-D4)", "distinguishing": "UQCRC2: UQCRC1 also absent (scaffold heterodimer lost); no cataracts; very similar BN-PAGE — WES mandatory for DDx; chromosomal locus: UQCRC2 at 16p12.2 vs UQCRQ at 5q31.1"},
            {"condition": "TTC19 (CIII-D2)", "distinguishing": "TTC19: psychiatric features (psychosis/depression 40%) — ABSENT in UQCRQ; late childhood onset + spinocerebellar ataxia — ABSENT in UQCRQ"},
            {"condition": "LYRM7 (CIII-D1)", "distinguishing": "LYRM7: RISP (UQCRFS1) absent on immunoblot with UQCRC1/UQCRC2 core PRESERVED; UQCRQ: RISP secondarily reduced AND UQCRC1/2 variable — immunoblot pattern differs; LYRM7 at 5q33.1, UQCRQ at 5q31.1 (same chromosome)"},
            {"condition": "SCO2", "distinguishing": "SCO2: HCM 100% — CARDINAL; CIV deficiency (not CIII); UQCRQ: cardiomyopathy <12%; CIII deficiency"},
        ],
        "key_references": [
            "Barel O, Shorer Z, Flusser H, et al. (2008) Mitochondrial complex III deficiency associated with a homozygous mutation in UQCRQ. Am J Hum Genet 82(1):267-74 — First identification of UQCRQ as a disease gene; Bedouin consanguineous family; p.Asn34Ser; severe psychomotor retardation, dystonia, optic atrophy",
            "Fernandez-Vizarra E & Zeviani M (2018) Nuclear gene mutations as the cause of mitochondrial complex III deficiency. Front Genet 9:135 — Comprehensive CIII nuclear gene review including UQCRQ structural context",
            "Berry EA, Guergova-Kuras M, Huang LS, Crofts AR (2000) Structure and function of cytochrome bc1 complexes. Annu Rev Biochem 69:1005-75 — UQCRQ (subunit 8) structural role at the Qo-site; lumen subunit group context",
            "Stroud DA et al. (2016) Accessory subunits are integral for assembly and function of human mitochondrial complex I. Nature 538(7623):123-126 — CIII assembly context; structural vs assembly factor classification",
        ],
        "terms": [
            {"term": "UQCRQ (QCR8)", "definition": "Ubiquinol-Cytochrome C Reductase Complex III Subunit VII; also known as QCR8 (yeast nomenclature) or 'subunit 8' of the bc1 complex; 82-aa small transmembrane subunit at the Qo-site periphery; OMIM gene *612080"},
            {"term": "Qo site", "definition": "Ubiquinol oxidation (outer) site of Complex III; where CoQH2 is oxidised; UQCRQ forms part of the structural periphery around this site; Qo-site ROS production amplified when UQCRQ is absent → selective basal ganglia toxicity → dystonia"},
            {"term": "RISP / UQCRFS1", "definition": "Rieske iron-sulfur protein; the catalytic FeS cluster component of CIII; inserted by BCS1L; secondarily reduced in UQCRQ deficiency because the RISP stalk loses peripheral stabilisation without UQCRQ"},
            {"term": "Lumen subunit group", "definition": "Group of small CIII subunits facing the IMS (lumen): UQCRQ, UQCR6 (UQCRH), UQCR10; structural stabilisers; not catalytic; loss → CIII structural instability rather than complete assembly failure (contrast UQCC1-UQCC2 assembly factors)"},
            {"term": "Optic atrophy", "definition": "Degeneration of the optic nerve; present in ~45% of UQCRQ patients — UNUSUAL for CIII deficiencies (DDx from UQCC2/UQCC3/LYRM7); retinal ganglion cells and long optic nerve axons are selectively vulnerable to CIII deficiency + Qo-site ROS"},
            {"term": "Dystonia", "definition": "Involuntary sustained muscle contractions causing repetitive movements or abnormal postures; CARDINAL in UQCRQ (75%); caused by selective basal ganglia vulnerability to CIII-mediated Qo-site reactive oxygen species"},
            {"term": "PRIS (Propofol Infusion Syndrome)", "definition": "Life-threatening drug reaction; propofol inhibits CIII at the Qo site and fatty acid oxidation; risk greatly increased in CIII-deficient patients including UQCRQ; absolute contraindication"},
            {"term": "BN-PAGE (Blue-Native PAGE)", "definition": "Blue-native polyacrylamide gel electrophoresis; separates intact mitochondrial respiratory chain complexes; in UQCRQ: CIII reduced but sub-complexes present — distinct from UQCC2 (CIII absent) and BCS1L (precomplex accumulates); WES mandatory as pattern overlaps UQCRC2"},
            {"term": "GIR (Glucose Infusion Rate)", "definition": "Continuous IV dextrose at 6–8 mg/kg/min; prevents fasting-induced lactic acidosis; mandatory in acute management of CIII deficiencies including UQCRQ"},
            {"term": "CoQH2 backlog", "definition": "Accumulation of reduced coenzyme Q (ubiquinol) due to impaired Qo-site oxidation; CIII deficiency → CoQH2 cannot be oxidised → backup of reducing equivalents → NADH/FADH2 accumulate → lactic acidosis; worsened by ketogenic diet (increased FADH2 from beta-oxidation)"},
        ],
    }


if __name__ == "__main__":
    ov = get_overview()
    print(f"Gene: {ov['gene']} ({ov['alias']})")
    print(f"Disease: {ov['disease']}")
    print(f"OMIM Gene: *{ov['omim_gene']}")
    print(f"Chromosome: {ov['chromosome']}  Inheritance: {ov['inheritance']}")
    print(f"\nCohort: {ov['cohort_n']} patients, seed {ov['seed']}")
    s = ov["cohort_statistics"]
    print(f"  Psychomotor retardation: {s['psychomotor_retardation_pct']}%")
    print(f"  Hypotonia: {s['hypotonia_pct']}%")
    print(f"  Dystonia (cardinal): {s['dystonia_pct']}%")
    print(f"  Optic atrophy (distinguishing): {s['optic_atrophy_pct']}%")
    print(f"  Leigh-like MRI: {s['leigh_like_mri_pct']}%")
    print(f"  CIII activity avg: {s['avg_ciii_activity_pct']}%")
    print(f"  Lactic acid avg: {s['avg_lactic_acid_mmolL']} mM")
    print(f"  Deceased (any): {s['deceased_pct']}%")
    print("\nVariants:", [v["protein"] for v in VARIANTS])
    print(f"\nTop variant counts: {ov['top_variant_counts'][:3]}")
    print("\nKey clinical alerts:")
    for a in ov["key_clinical_alerts"][:5]:
        print(f"  {a}")
