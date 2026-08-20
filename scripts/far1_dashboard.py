#!/usr/bin/env python3
"""FAR1 / Rhizomelic Chondrodysplasia Punctata Type 4 (RCDP4) Epilepsy Dashboard — seed data module.

FAR1 encodes Fatty Acyl-CoA Reductase 1 (553 aa), a NADPH-dependent reductase localised at
the peroxisomal membrane that catalyses the reduction of long-chain fatty acyl-CoA to the
corresponding fatty alcohol. This fatty alcohol is then used by AGPS (Step 2 of plasmalogen
biosynthesis) to form the ether bond at sn-1 of dihydroxyacetone phosphate (DHAP).

ENZYME FUNCTION — FAR1 IN PLASMALOGEN PATHWAY:
  GNPAT (Step 1):  DHAP + acyl-CoA  →  1-acyl-DHAP                   [intact in RCDP4]
  FAR1 (parallel): Fatty acyl-CoA + NADPH  →  Fatty alcohol + NADP+  [BLOCKED in RCDP4]
  AGPS (Step 2):   1-acyl-DHAP + Fatty alcohol  →  1-alkyl-DHAP + Fatty acid
  FAR1 provides the fatty-alcohol substrate required by AGPS. Without FAR1, AGPS has no
  fatty alcohol → ether-bond formation impossible → plasmalogen synthesis collapses.

CRITICAL BIOCHEMICAL DISTINCTIONS:
  RCDP4 vs RCDP1 (PEX7): PEX7 intact in RCDP4 → PHYH (PTS2 cargo) IS imported → phytanic NORMAL.
    RCDP1: PEX7 absent → PHYH cannot be imported → phytanic ELEVATED. KEY DISTINCTION.
  RCDP4 vs RCDP2 (GNPAT): GNPAT intact in RCDP4 → 1-acyl-DHAP produced. But AGPS has no substrate
    (fatty alcohol absent) → ether bond cannot form. Near-identical biochemistry; gene sequencing only.
  RCDP4 vs RCDP3 (AGPS): In RCDP3, fatty alcohol IS produced (FAR1 intact) but AGPS cannot use it.
    In RCDP4, FAR1 absent → no fatty alcohol → AGPS has substrate missing. Net result identical.
  RCDP4 vs ZSD (PEX1/PEX6): VLCFA NORMAL in RCDP4 (PTS1 pathway intact). ZSD: VLCFA ELEVATED.
  All four RCDP types (PEX7/GNPAT/AGPS/FAR1): plasmalogens severely low, phenotype near-identical.
  ONLY GENE SEQUENCING distinguishes RCDP4 from RCDP2 (GNPAT) and RCDP3 (AGPS).

FAR1 PROTEIN BIOLOGY:
  FAR1: 553 aa, type II peroxisomal membrane protein (PMP), NADPH-dependent reductase.
  FAR2 (a paralogue, ~65% identity) also exists but FAR2 deficiency has not been established as a
  distinct human disease phenotype as of 2026.
  FAR1 is expressed ubiquitously; FAR2 primarily in gonadal and brain tissue.
  Both FAR1 and FAR2 localise to the peroxisomal membrane — NOT the peroxisomal matrix.
  FAR1 is the major contributor to fatty-alcohol supply for plasmalogen synthesis in CNS/myelin.

LOCUS: 11p15.3  |  OMIM GENE: *616107  |  OMIM DISEASE: #616154
PROTEIN: 553 aa, peroxisomal membrane (PMP — NOT PTS1/PTS2 soluble import)
EPIDEMIOLOGY: Very rare; ~15–20 cases reported worldwide (2026).
  May represent the rarest molecularly defined RCDP subtype.
  Initially described by Buchert et al., 2014 (Am J Hum Genet).
  Biochemically indistinguishable from RCDP2/RCDP3 → must include FAR1 on peroxisomal gene panels.

ALKYLGLYCEROL PHARMACOLOGY IN RCDP4:
  Alkylglycerols (batyl/chimyl/selachyl alcohol = 1-O-alkyl-sn-glycerol) are pre-formed ether lipids.
  In RCDP4, the block is upstream of AGPS (fatty alcohol absent), but alkylglycerols provide
  pre-formed ether-linked glycerol that bypasses BOTH the fatty-alcohol requirement AND the AGPS step.
  The ether bond in alkylglycerols is already formed — cells convert to ether phospholipid precursors.
  Same mechanism as for RCDP2/RCDP3; experimental Level C. Human data n<20 across all RCDP combined.

PHENOTYPIC SPECTRUM (broadly similar to RCDP1/RCDP2/RCDP3; may show slightly more clinical overlap
with RCDP2 and RCDP3 than with RCDP1 due to phytanic-normal status):
  Classic RCDP4: Rhizomelia + stippled epiphyses + cataracts + ichthyosis.
    Seizures 50-60%. Profound ID. Median survival < 5 years. Plasmalogens virtually absent.
  Intermediate RCDP4: Moderate skeletal, moderate ID, seizures 20-35%. Survival 10-20 y.
  Mild RCDP4: Minimal skeletal, variable ID, seizures 12-18%, adult survival possible.

EPILEPSY:
  Overall seizures: 50-60% classic, 20-35% intermediate, 12-18% mild.
  Types: Infantile spasms (IS) 33-38%, Focal 28-33%, Myoclonic 14-18%, GTCS 10-14%, SE 5-8%.
  Drug resistance: ~28-32% of those with seizures.

AED PHARMACOLOGY (same phytanic-normal risk profile as RCDP2 and RCDP3):
  LEV: FIRST-LINE (no peroxisomal interactions, no adrenal mechanism, safe all forms).
  ACTH: Level B for IS (preferred over VGB given cataract risk; very limited RCDP4-specific data).
  VGB: HIGH RISK — cataracts 55-62% + VGB visual field loss = additive risk. Not absolute CI
    (unlike ZSD retinopathy universal/severe), but HIGH RISK. Monthly VF/VEP if used.
  VPA: RELATIVE CI — hepatotoxicity; POLG1 MANDATORY (CPIC Grade A). Phytanic NORMAL in RCDP4
    (less phytanic-hepatic stress than RCDP1). Mitochondrial risk persists regardless.
  PHT/CBZ/OXC: CAN USE — no adrenal insufficiency in RCDP4 (unlike ABCD1 = ABSOLUTE CI).
  Alkylglycerol supplementation: BYPASSES FAR1 BLOCK (pre-formed ether lipid; no fatty-alcohol
    deficiency to compensate). Experimental Level C; same mechanism as for RCDP2/RCDP3.
  DHA: Level C (restores DHA-plasmalogen).
  No ERT (2026): FAR1 is peroxisomal membrane protein — not secreted, not lysosomal. ERT
    cannot cross peroxisomal membrane as functional enzyme replacement.
  No HSCT: Pathology is hypomyelination (plasmalogen deficiency), not neuroinflammation.
"""

import random
random.seed(45)


# ── Overview (KPIs + summary) ─────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 55,
        "classic_rcdp_pct": 38,
        "intermediate_rcdp_pct": 42,
        "mild_rcdp_pct": 20,
        "drug_resistance_pct": 28,
        "cataract_pct": 58,
        "ichthyosis_pct": 55,
        "rhizomelia_pct": 100,
        "stippling_pct": 80,
        "on_alkylglycerol_pct": 25,
        "on_dha_pct": 32,
        "omim_gene": "616107",
        "omim_disease": "616154",
        "locus": "11p15.3",
        "vlcfa_normal": True,
        "vlcfa_normal_pct": 100,
        "phytanic_normal_pct": 88,
        "plasmalogen_low_pct": 96,
        "inheritance": "Autosomal recessive (AR), biallelic LOF",
        "common_variant": "No founder mutation — all private/rare variants; ~15–20 cases worldwide 2026",
        "disease_mechanism": (
            "FAR1 (Fatty Acyl-CoA Reductase 1) is a 553 aa NADPH-dependent reductase at the "
            "peroxisomal membrane that produces the fatty alcohol substrate required by AGPS for "
            "ether-bond formation (Step 2 of plasmalogen biosynthesis). FAR1 LOF abolishes fatty "
            "alcohol supply → AGPS cannot form the ether bond despite intact GNPAT (Step 1). "
            "Plasmalogens severely low as in RCDP1/2/3, but phytanic acid is NORMAL (PEX7 intact → "
            "PHYH imported normally). Biochemically near-identical to RCDP2/GNPAT and RCDP3/AGPS — "
            "distinguished only by gene sequencing. FAR1 is a peroxisomal membrane protein (PMP), "
            "not a matrix enzyme imported via PTS1/PTS2."
        ),
        "nbs_positive_rate": (
            "Not in NBS; plasmalogens on DBS advocated; phytanic normal → not triggered by "
            "routine NBS peroxisomal screens. Rarest molecularly defined RCDP — often missed or "
            "diagnosed as RCDP2/RCDP3 without FAR1 sequencing."
        ),
        "key_concepts": [
            "FAR1 = Fatty Acyl-CoA Reductase 1 (553 aa, peroxisomal membrane protein/PMP, NADPH-dependent); produces fatty alcohol for AGPS Step 2",
            "RCDP4 accounts for <5% of RCDP; ~15–20 cases worldwide 2026; possibly rarest molecularly defined RCDP subtype",
            "FAR1 deficiency → no fatty alcohol → AGPS has no substrate → ether bond cannot form → plasmalogens severely low",
            "Plasmalogens (RBC) SEVERELY LOW — same as RCDP1/2/3; primary diagnostic biomarker (C16:0-DMA, C18:0-DMA)",
            "Phytanic acid NORMAL in RCDP4 (PEX7 intact → PHYH PTS2 import unaffected) — same as RCDP2/RCDP3, unlike RCDP1",
            "VLCFA NORMAL — PTS1 pathway intact, PEX1/PEX6 intact; if VLCFA elevated → ZSD, not RCDP4",
            "No founder mutation — all private/rare variants; gene panel must include FAR1 alongside GNPAT and AGPS",
            "Cataracts 55-62%, rhizomelia 100%, stippled epiphyses 80% — same clinical hallmarks as RCDP1/2/3",
            "VGB HIGH RISK — cataracts + VGB visual field constriction = additive visual impairment",
            "VPA RELATIVE CI — hepatotoxicity; POLG1 MANDATORY (CPIC A); phytanic NORMAL (less phytanic-hepatic stress vs RCDP1)",
            "PHT/CBZ/OXC CAN BE USED — no adrenal insufficiency in RCDP4 (unlike ABCD1 = ABSOLUTE CI)",
            "Alkylglycerols bypass FAR1 block — pre-formed ether lipid provides ether bond without need for fatty alcohol",
            "FAR1 is peroxisomal membrane protein (PMP) — not secreted/lysosomal; ERT not applicable",
            "No HSCT — hypomyelination (not neuroinflammation); HSCT arrests inflammatory demyelination only",
            "Distinguished from RCDP2/GNPAT and RCDP3/AGPS ONLY by gene sequencing — biochemical markers near-identical",
        ],
        "standards": [
            "Buchert R, Tawamie H, Smith C, et al. A peroxisomal disorder of severe intellectual disability, epilepsy, and cataracts due to fatty acyl-CoA reductase 1 deficiency. Am J Hum Genet. 2014;95(5):602–610.",
            "Honsho M, Fujiki Y. Plasmalogen homeostasis: regulation of plasmalogen biosynthesis and its physiological significance in mammals. FEBS Lett. 2017.",
            "Braverman NE et al. Rhizomelic chondrodysplasia punctata. GeneReviews. NCBI Bookshelf. 2015.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism. Biochim Biophys Acta. 2016.",
            "Wanders RJA, Waterham HR. Biochemistry of mammalian peroxisomes revisited. Annu Rev Biochem. 2006.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org.",
        ],
    }


# ── Breakdown (patients + seizures + treatments) ─────────────────────────────

def get_breakdown():
    etiologies = [
        {
            "name": "Classic RCDP4 (Null/Null or Severe LOF)",
            "pct": 38,
            "n": 15,
            "sex": "M/F equal",
            "onset_age": "Neonatal–3 months",
            "seizure_risk": "50–60%",
            "eeg": "Hypsarrhythmia (infantile spasms), multifocal spike-wave, burst-suppression",
            "mri": "Severe hypomyelination, reduced white matter T2 signal, cortical atrophy",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic null or severe LOF (frameshift/nonsense/large deletion) → complete FAR1 "
                "enzyme absence → no fatty alcohol produced → AGPS has no ether-bond substrate. "
                "Plasmalogens virtually absent. Rhizomelia severe, congenital cataracts, profound ID. "
                "All variants are private (no founder allele). Phytanic NORMAL (PEX7 intact). "
                "GNPAT (Step 1) and AGPS enzyme are intact but AGPS has no fatty-alcohol substrate."
            ),
        },
        {
            "name": "Intermediate RCDP4 (Null/Hypomorphic or Splice)",
            "pct": 42,
            "n": 17,
            "sex": "M/F equal",
            "onset_age": "3–18 months",
            "seizure_risk": "20–35%",
            "eeg": "Modified hypsarrhythmia, focal spikes, multifocal discharges, theta slowing",
            "mri": "Moderate hypomyelination, periventricular signal changes",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "One null + one hypomorphic allele → partial FAR1 reductase activity. "
                "Residual fatty alcohol production allows partial AGPS ether-bond formation. "
                "Plasmalogens 5–25% of normal. Moderate rhizomelia, moderate ID. Survival 10–20 y. "
                "Phytanic NORMAL; phytol restriction not required."
            ),
        },
        {
            "name": "Mild RCDP4 (Hypomorphic/Hypomorphic)",
            "pct": 20,
            "n": 8,
            "sex": "M/F equal",
            "onset_age": "6–36 months",
            "seizure_risk": "12–18%",
            "eeg": "Focal discharges, sparse multifocal spikes, may be near-normal",
            "mri": "Minimal white matter changes; myelination near-normal",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic hypomorphic alleles → residual FAR1 reductase activity (15–40% of normal). "
                "Fatty alcohol production partially restored → AGPS can form some ether bonds. "
                "Plasmalogens 20–55% of normal. Minimal rhizomelia, mild-moderate ID, adult survival. "
                "Mildest RCDP4 phenotype; may be diagnosed in childhood or adult life."
            ),
        },
    ]

    phenotypes = (
        [("Classic RCDP4", "Severe")] * 15
        + [("Intermediate RCDP4", "Intermediate")] * 17
        + [("Mild RCDP4", "Mild")] * 8
    )
    sexes = (["M"] * 20 + ["F"] * 20)
    random.shuffle(sexes)

    genotype_map = {
        "Classic RCDP4": ["Biallelic null (frameshift/nonsense)", "Null + severe splice", "Homozygous large deletion"],
        "Intermediate RCDP4": ["Null + hypomorphic missense", "Splice variant + missense", "Compound heterozygous missense"],
        "Mild RCDP4": ["Biallelic hypomorphic missense", "Compound hypomorphic", "Residual reductase alleles"],
    }

    seizure_map = {
        "Severe": ["Infantile spasms", "Multifocal clonic", "Myoclonic"],
        "Intermediate": ["Focal aware", "Focal to bilateral GTCS", "Myoclonic"],
        "Mild": ["Focal aware", "GTCS", None],
    }

    trigger_pool = [
        "Febrile illness", "Sleep deprivation", "Missed AED", "Stress / arousal",
        "Photic stimulation", "Metabolic decompensation", "Anaesthesia (low risk — no adrenal mechanism)",
    ]

    treatment_pool = [
        "LEV (first-line)", "ACTH (IS — Level B)", "CLB (adjunct — Level C)",
        "LTG (adjunct — Level C)", "DHA supplementation (Level C)",
        "Alkylglycerol supplementation (FAR1-block bypass — experimental Level C)",
    ]

    patients = []
    for i, ((pheno, severity), sex) in enumerate(zip(phenotypes, sexes)):
        pid = f"RCDP4-{i+1:02d}"
        age_mo = random.randint(0, 3) if severity == "Severe" else (
            random.randint(3, 18) if severity == "Intermediate" else random.randint(6, 36)
        )
        has_sz = random.random() < (0.55 if severity == "Severe" else 0.28 if severity == "Intermediate" else 0.15)
        seizure_type = random.choice(seizure_map[severity]) if has_sz else None
        sz_free = random.choice([True, False]) if has_sz else True
        drug_resistant = has_sz and not sz_free and random.random() < 0.38
        triggers = random.sample(trigger_pool, k=random.randint(2, 4)) if has_sz else []
        treatments = random.sample(treatment_pool, k=random.randint(1, 3))
        genotype = random.choice(genotype_map[pheno])
        patients.append({
            "id": pid,
            "phenotype": pheno,
            "severity": severity,
            "sex": sex,
            "onset_age_months": age_mo,
            "has_seizures": has_sz,
            "seizure_type": seizure_type,
            "seizure_free": sz_free,
            "drug_resistant": drug_resistant,
            "triggers": triggers,
            "treatments": treatments,
            "genotype": genotype,
            "phytanic_normal": True,
            "vlcfa_normal": True,
            "plasmalogen_severely_low": True,
            "cataracts": random.random() < 0.58,
            "ichthyosis": random.random() < 0.55,
            "rhizomelia": True,
            "stippled_epiphyses": random.random() < 0.80,
            "polg1_tested": True,
        })

    seizure_types = [
        {"type": "Infantile Spasms / Hypsarrhythmia", "pct": 35, "preferred_tx": "ACTH Level B (preferred over VGB — cataracts additive)"},
        {"type": "Focal (aware / impaired awareness)", "pct": 32, "preferred_tx": "LEV first-line"},
        {"type": "Myoclonic", "pct": 16, "preferred_tx": "LEV / CLB adjunct"},
        {"type": "Focal to Bilateral GTCS", "pct": 12, "preferred_tx": "LEV + CLB"},
        {"type": "Status Epilepticus", "pct": 6, "preferred_tx": "IV LEV / MDZ; fosphenytoin acceptable (no adrenal)"},
    ]

    triggers = [
        {"trigger": "Febrile illness", "pct": 58},
        {"trigger": "Sleep deprivation", "pct": 36},
        {"trigger": "Missed AED dose", "pct": 30},
        {"trigger": "Stress / arousal", "pct": 26},
        {"trigger": "Photic stimulation", "pct": 13},
        {"trigger": "Metabolic decompensation", "pct": 16},
        {"trigger": "Anaesthesia (low risk; no adrenal mechanism unlike ABCD1)", "pct": 8},
    ]

    monitoring = [
        {"parameter": "RBC plasmalogens (C16:0-DMA, C18:0-DMA)", "threshold": "Severely low (<5% normal)", "frequency": "Every 6 months"},
        {"parameter": "Plasma phytanic acid", "threshold": "Usually NORMAL (<30 µmol/L); monitor if diet uncertain", "frequency": "Annually"},
        {"parameter": "Plasma VLCFA (C26:0)", "threshold": "NORMAL (if elevated → reconsider ZSD/PBD)", "frequency": "At diagnosis; annually"},
        {"parameter": "LFTs (if VPA used — RELATIVE CI)", "threshold": "Normal → q3 months if VPA used", "frequency": "q3 months on VPA"},
        {"parameter": "EEG", "threshold": "Baseline + after treatment change", "frequency": "q6–12 months"},
        {"parameter": "Ophthalmology / Visual field / VEP", "threshold": "Baseline cataracts screen; VF mandatory if VGB used", "frequency": "Annually; monthly VF if VGB"},
        {"parameter": "DHA plasma", "threshold": "Low → supplement 200 mg/day (Level C)", "frequency": "Every 6 months"},
    ]

    lifecycle = [
        {"stage": "Neonatal (0–1 mo)", "features": "Stippled epiphyses on X-ray, rhizomelia, congenital cataracts; biochemistry: plasmalogens severely low, phytanic NORMAL, VLCFA NORMAL"},
        {"stage": "Infant (1–12 mo)", "features": "Infantile spasms onset (33-38%); hypsarrhythmia; ACTH Level B; VGB to be avoided (cataracts additive); DHA + alkylglycerol started"},
        {"stage": "Toddler (1–3 y)", "features": "Persistent multifocal epilepsy; hypomyelination on MRI; profound ID in classic; physiotherapy for rhizomelia"},
        {"stage": "Child (3–10 y)", "features": "Drug-resistant epilepsy 28%; contractures worsen; cataract surgery; intermediate forms attend school"},
        {"stage": "Adolescent (10–18 y)", "features": "Plateau or slow decline; mild forms show functional improvement; seizure frequency stabilises"},
        {"stage": "Adult (>18 y)", "features": "Mild RCDP4 only: adult survival, residual mild ID, possible employment; classic/intermediate rare >18y"},
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "SV2A modulator",
            "level": "First-line (Level A RCDP-class)",
            "dose": "20–60 mg/kg/day in 2 divided doses; IV for SE",
            "notes": "No peroxisomal interactions; no adrenal effect; no VLCFA/phytanic impact. Preferred across all RCDP subtypes.",
            "ci": "None specific to RCDP4",
        },
        {
            "drug": "ACTH / Corticotropin",
            "class": "Infantile spasms — hormone therapy",
            "level": "Level B (IS preferred over VGB)",
            "dose": "150 IU/m²/day IM × 2 weeks then taper",
            "notes": "Preferred over VGB for IS in RCDP4: avoids adding to cataract + VF risk. Adrenal axis intact — ACTH does not carry ABCD1-class risk. POLG1-status must be known.",
            "ci": "None specific; monitor glycaemia, BP, infection",
        },
        {
            "drug": "Alkylglycerol supplementation",
            "class": "Ether lipid precursor (experimental)",
            "level": "Level C (experimental)",
            "dose": "Batyl/chimyl/selachyl alcohol 100–200 mg/kg/day (no standard dose 2026)",
            "notes": "Provides pre-formed ether-linked glycerol downstream of FAR1 block. Bypasses both fatty-alcohol requirement (FAR1) and ether-bond step (AGPS). Same mechanism as RCDP2/RCDP3 use. Animal data positive; human n<20 across all RCDP types.",
            "ci": "None known; long-term safety uncharacterised",
        },
        {
            "drug": "DHA (Docosahexaenoic acid)",
            "class": "Omega-3 supplementation",
            "level": "Level C",
            "dose": "200 mg/day (infants); weight-based scaling in children",
            "notes": "Restores DHA-plasmalogen pools; safe, no drug interactions. Used in all RCDP subtypes for neuronal membrane support.",
            "ci": "None",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "Benzodiazepine — 1,5-BZD",
            "level": "Level C adjunct",
            "dose": "0.1–0.3 mg/kg/day in 2 divided doses (max 40 mg/day)",
            "notes": "Add-on for focal and myoclonic seizures. Sedation risk in severe forms. No peroxisomal interaction.",
            "ci": "None specific",
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "Sodium channel / glutamate modulator",
            "level": "Level C adjunct",
            "dose": "Slow titration; weight-based (titration per BNFc/Lexicomp)",
            "notes": "Useful add-on for focal seizures in intermediate/mild RCDP4. No peroxisomal interactions.",
            "ci": "None specific to RCDP4; standard SJS risk with rapid titration",
        },
    ]

    contraindications = [
        {
            "drug": "Vigabatrin (VGB)",
            "level": "HIGH RISK — cataracts + VF additive",
            "reason": (
                "Cataracts in 55-62% RCDP4 + VGB-induced irreversible visual field constriction = "
                "additive and cumulative visual impairment. Cataracts impair VF assessment reliability. "
                "NOT absolute CI (unlike ZSD where retinopathy is universal/severe), but HIGH RISK. "
                "Monthly VF/VEP mandatory if used; prefer ACTH for IS."
            ),
            "alternative": "ACTH (Level B, IS) · LEV (focal/myoclonic)",
        },
        {
            "drug": "Valproate (VPA)",
            "level": "RELATIVE CI — hepatotoxicity + POLG1 MANDATORY",
            "reason": (
                "Hepatotoxicity risk. POLG1 MANDATORY (CPIC Grade A) before VPA in any suspected "
                "mitochondrial disease context. Phytanic NORMAL in RCDP4 (less phytanic-driven hepatic "
                "stress vs RCDP1), but POLG1/mitochondrial risk persists. LFT q3 months if used."
            ),
            "alternative": "LEV (first-line) · LTG/CLB adjunct",
        },
        {
            "drug": "Phenytoin (PHT) / Fosphenytoin — routine use note",
            "level": "CAN USE (no adrenal — contrast ABCD1)",
            "reason": (
                "PHT/fosphenytoin carry no adrenal-insufficiency risk in RCDP4 (unlike ABCD1/X-ALD "
                "where PHT = ABSOLUTE CI via CYP3A4 cortisol drop → adrenal crisis). PHT CAN be used "
                "in RCDP4 when clinically indicated, including for SE. Standard cardiac monitoring applies."
            ),
            "alternative": "LEV preferred first-line; PHT acceptable for SE if LEV unavailable",
        },
        {
            "drug": "Typical antipsychotics (haloperidol, chlorpromazine)",
            "level": "RELATIVE CI — EPS risk in ID/hypomyelination",
            "reason": (
                "Severe ID + hypomyelination increases EPS sensitivity. Atypical preferred if needed "
                "(quetiapine, risperidone low-dose). Not an absolute CI but use with caution."
            ),
            "alternative": "Atypical antipsychotics if behavioural management required",
        },
        {
            "drug": "No ERT (2026) — FAR1 peroxisomal membrane protein",
            "level": "NOT APPLICABLE — ERT not feasible",
            "reason": (
                "FAR1 is a peroxisomal membrane protein (PMP; not secreted; not lysosomal). "
                "Systemic ERT cannot reach intracellular peroxisomal membranes as functional enzyme. "
                "No gene-therapy trials (IND/Phase I) in RCDP4 as of 2026 (too few cases worldwide)."
            ),
            "alternative": "Alkylglycerol bypass (experimental Level C) · DHA supplementation",
        },
    ]

    return {
        "etiologies": etiologies,
        "patients": patients,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "monitoring": monitoring,
        "lifecycle": lifecycle,
        "treatments": treatments,
        "contraindications": contraindications,
    }


# ── Definitions ───────────────────────────────────────────────────────────────

def get_definitions():
    return {
        "key_concepts": [
            "FAR1 = Fatty Acyl-CoA Reductase 1 (553 aa, peroxisomal membrane protein, NADPH-dependent); produces fatty alcohol for AGPS Step 2",
            "RCDP4: <5% of RCDP; ~15–20 cases worldwide 2026; biochemically indistinguishable from RCDP2/RCDP3 without FAR1 sequencing",
            "FAR1 deficiency → no fatty alcohol → AGPS substrate absent → ether bond cannot form → plasmalogens severely low (same as RCDP1/2/3)",
            "Plasmalogens (RBC) SEVERELY LOW — C16:0-DMA and C18:0-DMA virtually absent in classic RCDP4",
            "Phytanic acid NORMAL in RCDP4 (PEX7 intact → PHYH PTS2 import unaffected) — KEY DISTINCTION from RCDP1/PEX7",
            "VLCFA NORMAL — PTS1 pathway intact (PEX1/PEX6 intact); VLCFA elevated → ZSD, NOT RCDP4",
            "No founder mutation — all private variants; peroxisomal gene panel must include FAR1 after GNPAT and AGPS negative",
            "Cataracts 55-62%, rhizomelia 100%, stippled epiphyses 80%; same hallmarks as RCDP1/2/3 — ether-lipid-deficiency phenotype",
            "VGB HIGH RISK — cataracts (58%) + VGB visual field constriction = additive risk; prefer ACTH for IS",
            "VPA RELATIVE CI — hepatotoxicity; POLG1 MANDATORY (CPIC A); phytanic NORMAL (less phytanic-hepatic stress than RCDP1)",
            "PHT/CBZ/OXC CAN BE USED — no adrenal insufficiency in RCDP4 (unlike ABCD1 where PHT = ABSOLUTE CI)",
            "Alkylglycerols bypass FAR1 block — pre-formed ether lipid requires neither fatty alcohol (FAR1) nor ether-bond enzyme (AGPS)",
            "FAR1 is PMP (peroxisomal membrane protein) — not secreted/lysosomal; ERT not applicable unlike LSDs",
            "No HSCT — hypomyelination from plasmalogen deficiency, not neuroinflammation; HSCT irrelevant in RCDP",
            "Described first: Buchert et al. 2014 (Am J Hum Genet); FAR2 paralogue not yet linked to human disease as of 2026",
        ],
        "diagnostic_algorithm": [
            "Neonate with rhizomelia + stippled epiphyses + cataracts ± ichthyosis: order URGENT peroxisomal biochemistry.",
            "Step 1: Plasma VLCFA (C26:0). If ELEVATED → suspect ZSD (PEX1/PEX6/etc.); if NORMAL → continue RCDP workup.",
            "Step 2: RBC plasmalogens (C16:0-DMA, C18:0-DMA). If SEVERELY LOW → RCDP confirmed biochemically.",
            "Step 3: Plasma phytanic acid. If ELEVATED → RCDP1/PEX7; if NORMAL → RCDP2/GNPAT, RCDP3/AGPS, or RCDP4/FAR1.",
            "Step 4: Plasma pristanic acid (usually NORMAL in all RCDP — elevated in Adult Refsum if AMACR deficiency).",
            "Step 5: Order peroxisomal gene panel including PEX7, GNPAT, AGPS, FAR1 (and PHYH if phytanic elevated).",
            "Step 6: FAR1 sequencing — if GNPAT and AGPS negative with RCDP phenotype, FAR1 must be tested.",
            "Step 7: EEG if seizures suspected; MRI brain (hypomyelination pattern).",
            "Step 8: Ophthalmology — cataracts, VEP, VF baseline (critical before considering VGB).",
            "Step 9: Skeletal X-rays for stippling extent and rhizomelia quantification.",
            "Step 10: POLG1 sequencing MANDATORY before any VPA consideration (CPIC Grade A).",
            "Step 11: Start DHA supplementation + alkylglycerol if available; LEV for seizures.",
        ],
        "pharmacological_distinctions": [
            "LEV: FIRST-LINE across all RCDP subtypes including RCDP4 — no peroxisomal, adrenal, or VLCFA interactions.",
            "ACTH: PREFERRED over VGB for IS in RCDP4 — avoids adding VGB visual field constriction to existing cataract risk.",
            "VGB: HIGH RISK in RCDP4 — cataracts 55-62% + VGB VF constriction additive. NOT absolute CI as in ZSD (retinopathy universal) but avoid if ACTH available.",
            "VPA: RELATIVE CI — POLG1 MANDATORY (CPIC A). Phytanic NORMAL in RCDP4 (less phytanic-hepatic stress than RCDP1) but hepatotoxicity risk from VPA mechanism persists.",
            "PHT/CBZ/OXC: CAN USE in RCDP4 (no adrenal axis involvement). Contrast with ABCD1/X-ALD = ABSOLUTE CI (CYP3A4 cortisol drop → adrenal crisis).",
            "Alkylglycerols: Bypass FAR1 block by supplying pre-formed ether lipid. Pre-formed 1-O-alkyl-sn-glycerol bypasses the need for both fatty alcohol (FAR1) and ether-bond formation (AGPS). Experimental Level C.",
            "DHA: Level C across all RCDP; safe, no interactions; restores DHA-plasmalogen pool.",
            "No ERT: FAR1 is peroxisomal membrane protein — not secreted, not lysosomal. Cannot be delivered as ERT unlike lysosomal hydrolases.",
            "No HSCT: Hypomyelination from plasmalogen deficiency responds to ether-lipid supplementation, NOT donor marrow cells. HSCT halts inflammatory demyelination (ABCD1-CCALD) — irrelevant in RCDP.",
            "Typical antipsychotics: RELATIVE CI in severe ID + hypomyelination — increased EPS sensitivity. Use atypicals (quetiapine preferred).",
            "Anaesthesia: LOW risk vs ABCD1 (no adrenal crisis mechanism). Standard precautions for severe ID, airway, positioning (rhizomelia). No specific anaesthetic CI.",
            "Phytol-restricted diet: NOT required in RCDP4 (phytanic NORMAL, PEX7 intact). Contrast RCDP1/PEX7 where phytol restriction Level B.",
        ],
        "differential_diagnosis": [
            {
                "condition": "RCDP1 (PEX7)",
                "distinction": "Phytanic ELEVATED (PEX7 absent → PHYH cannot be imported via PTS2); in RCDP4 phytanic NORMAL. Otherwise clinically and biochemically near-identical. Most common RCDP (~90%).",
            },
            {
                "condition": "RCDP2 (GNPAT)",
                "distinction": "Biochemically near-identical to RCDP4 — phytanic NORMAL, VLCFA NORMAL, plasmalogens severely low. GNPAT (Step 1) vs FAR1 (fatty alcohol supplier for Step 2). Only gene sequencing distinguishes.",
            },
            {
                "condition": "RCDP3 (AGPS)",
                "distinction": "Biochemically near-identical to RCDP4 — phytanic NORMAL, VLCFA NORMAL, plasmalogens severely low. AGPS (Step 2 enzyme) vs FAR1 (Step 2 substrate provider). Only gene sequencing distinguishes.",
            },
            {
                "condition": "Zellweger Spectrum Disorder (PEX1 / PEX6)",
                "distinction": "VLCFA ELEVATED in ZSD (all PTS1 cargo import fails). In RCDP4 VLCFA NORMAL (PTS1 intact; FAR1 is a PMP, not PTS1 cargo). ZSD also has phytanic/pristanic elevated and pipecolic acid elevated.",
            },
            {
                "condition": "Adult Refsum Disease (PHYH / AMACR)",
                "distinction": "Phytanic ELEVATED (PHYH loss → phytanic accumulation). In RCDP4 phytanic NORMAL. Adult Refsum presents in adolescence/adulthood — no rhizomelia, no cataracts, no stippling. OMIM #266500.",
            },
            {
                "condition": "CDPX2 / EBP (X-linked chondrodysplasia punctata)",
                "distinction": "XL-dominant (hemizygous males lethal); features: asymmetric limb shortening, peroxisomal β-oxidation bypass. Phytanic NORMAL but cholesterol/sterol accumulation; different molecular pathway (EBP = emopamil binding protein). No peroxisomal plasmalogen defect.",
            },
            {
                "condition": "Non-peroxisomal skeletal dysplasias (e.g. Conradi-Hunermann)",
                "distinction": "Stippling on X-ray shared. Normal peroxisomal biochemistry distinguishes. Plasmalogens NORMAL. Phytanic NORMAL. VLCFA NORMAL. Diagnosis by clinical genetics + skeletal survey.",
            },
        ],
        "diagnostic_algorithm": [
            "Neonate with rhizomelia + stippled epiphyses + cataracts ± ichthyosis: order URGENT peroxisomal biochemistry.",
            "Step 1: Plasma VLCFA (C26:0). If ELEVATED → suspect ZSD (PEX1/PEX6/etc.); if NORMAL → continue RCDP workup.",
            "Step 2: RBC plasmalogens (C16:0-DMA, C18:0-DMA). If SEVERELY LOW → RCDP confirmed biochemically.",
            "Step 3: Plasma phytanic acid. If ELEVATED → RCDP1/PEX7; if NORMAL → RCDP2/GNPAT, RCDP3/AGPS, or RCDP4/FAR1.",
            "Step 4: Plasma pristanic acid (usually NORMAL in all RCDP; elevated in Adult Refsum if AMACR deficiency).",
            "Step 5: Order peroxisomal gene panel including PEX7, GNPAT, AGPS, FAR1 (and PHYH if phytanic elevated).",
            "Step 6: FAR1 sequencing REQUIRED if GNPAT and AGPS both negative with RCDP phenotype — RCDP4 underdiagnosed.",
            "Step 7: EEG if seizures suspected; MRI brain (hypomyelination pattern on T2-weighted sequences).",
            "Step 8: Ophthalmology — cataracts, VEP, VF baseline (critical before considering VGB).",
            "Step 9: Skeletal X-rays for stippling extent and rhizomelia quantification.",
            "Step 10: POLG1 sequencing MANDATORY before any VPA consideration (CPIC Grade A).",
            "Step 11: Start DHA supplementation + alkylglycerol (experimental) if available; LEV for seizures.",
            "Step 12: Multidisciplinary team — neurology, ophthalmology, orthopaedics, physiotherapy, clinical genetics.",
        ],
        "standards": [
            "Buchert R, Tawamie H, Smith C, et al. A peroxisomal disorder of severe intellectual disability, epilepsy, and cataracts due to fatty acyl-CoA reductase 1 deficiency. Am J Hum Genet. 2014;95(5):602–610.",
            "Honsho M, Fujiki Y. Plasmalogen homeostasis: regulation of plasmalogen biosynthesis and its physiological significance in mammals. FEBS Lett. 2017.",
            "Braverman NE et al. Rhizomelic chondrodysplasia punctata. GeneReviews. NCBI Bookshelf. 2015.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism. Biochim Biophys Acta. 2016.",
            "Wanders RJA, Waterham HR. Biochemistry of mammalian peroxisomes revisited. Annu Rev Biochem. 2006.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org.",
        ],
    }
