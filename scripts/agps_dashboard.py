#!/usr/bin/env python3
"""AGPS / Rhizomelic Chondrodysplasia Punctata Type 3 (RCDP3) Epilepsy Dashboard — seed data module.

AGPS encodes Alkylglycerone Phosphate Synthase (ADHAPS — alkyl-dihydroxyacetonephosphate synthase;
658 aa), the SECOND enzyme in the peroxisomal ether-phospholipid (plasmalogen) biosynthesis
pathway. AGPS uses a PTS1 signal (C-terminal tripeptide -AKL-) for peroxisomal import.

ENZYME FUNCTION:
  Step 1 (GNPAT/DHAPAT): DHAP + fatty acyl-CoA  →  1-acyl-DHAP        [intact in RCDP3]
  Step 2 (AGPS/ADHAPS):  1-acyl-DHAP + fatty alcohol → 1-alkyl-DHAP   [BLOCKED in RCDP3]
  Net result: ether-bond formation fails; all downstream plasmalogen synthesis collapses.
  RBC plasmalogens SEVERELY LOW (virtually absent in classic RCDP3) — same as RCDP1 and RCDP2.

CRITICAL BIOCHEMICAL DISTINCTIONS:
  RCDP3 vs RCDP1 (PEX7): PEX7 intact in RCDP3 → PHYH (PTS2 cargo) IS imported → phytanic NORMAL.
    RCDP1 (PEX7 deficiency): PEX7 absent → PHYH CANNOT be imported → phytanic ELEVATED.
    KEY DIAGNOSIS POINT: Phytanic acid NORMAL in RCDP3 (same as RCDP2), ELEVATED in RCDP1.
  RCDP3 vs RCDP2 (GNPAT): Biochemically NEAR-IDENTICAL (both: plasmalogens severely low,
    phytanic NORMAL, VLCFA NORMAL). In RCDP2 (GNPAT blocked), Step 1 substrate for AGPS is
    absent (no 1-acyl-DHAP); AGPS protein IS present in peroxisome but has no substrate.
    In RCDP3 (AGPS blocked), 1-acyl-DHAP accumulates from intact GNPAT but ether bond cannot form.
    ONLY GENE SEQUENCING (GNPAT vs AGPS) distinguishes RCDP2 from RCDP3.
  RCDP3 vs ZSD (PEX1/PEX6): VLCFA NORMAL in RCDP3 (PTS1 pathway intact, PEX1/PEX6 intact).
    ZSD: VLCFA ELEVATED. This is the fastest screen to exclude ZSD.

PTS2-PATHWAY IN RCDP3 (INTACT — CONTRAST WITH RCDP1):
  PEX7 intact → PHYH (PTS2) imported → phytanic oxidation NORMAL → phytanic NORMAL/near-normal.
  PEX7 intact → AGPS imported normally, but AGPS enzyme is LOF → 1-alkyl-DHAP cannot form.
  GNPAT (Step 1) produces 1-acyl-DHAP correctly; AGPS cannot process it → ether bond not formed.

LOCUS: 2q33.3  |  OMIM GENE: *603051  |  OMIM DISEASE: #600121
PROTEIN: 658 aa, peroxisomal matrix enzyme, PTS1 import (C-terminal -AKL-)
EPIDEMIOLOGY: ~5% of all RCDP (vs RCDP1/PEX7 ~90%); fewer than 30 cases reported worldwide 2026.
  Rarest of the 3 RCDP subtypes; may be underdiagnosed due to biochemical identity with RCDP2.
NO FOUNDER MUTATION: All known variants are private or rare; no single population founder detected.

ALKYLGLYCEROL PHARMACOLOGY IN RCDP3 — BYPASS AT STEP 2:
  Alkylglycerols (batyl/chimyl/selachyl alcohol = 1-O-alkyl-sn-glycerol) are pre-formed ether lipids.
  When absorbed, cells convert them to ether phospholipid precursors bypassing BOTH Step 1 and Step 2.
  In RCDP3 (AGPS blocked at Step 2), alkylglycerols bypass the ether-bond formation step directly —
  they arrive with the ether bond already intact, downstream of the AGPS block.
  In RCDP2 (GNPAT blocked at Step 1), alkylglycerols bypass Step 1 (most direct rationale for RCDP2).
  Experimental Level C for both; human data n<20 across all RCDP subtypes combined.

PHENOTYPIC SPECTRUM (similar to RCDP1 and RCDP2; may be slightly milder on average):
  Classic RCDP3: Rhizomelia + stippled epiphyses + cataracts + ichthyosis.
    Seizures 55-65%. Profound ID. Median survival < 5 years.
  Intermediate RCDP3: Moderate skeletal + mild-moderate ID. Seizures 25-40%. Survival 10-20y.
  Mild RCDP3: Minimal skeletal + variable ID. Seizures ~15-20%. Adult survival reported.

EPILEPSY:
  Overall seizures: 55-65% classic, 25-40% intermediate, 15-20% mild RCDP3.
  Types: Infantile spasms (IS) 35-40%, Focal 28-32%, Myoclonic 15-20%, GTCS 12-15%, SE 6-8%.
  Drug resistance: ~30-33% of those with seizures.

AED PHARMACOLOGY (identical to RCDP2 with same phytanic-normal nuances):
  LEV: FIRST-LINE (no peroxisomal interactions, no adrenal mechanism, safe all forms).
  ACTH: Level B for IS (very limited RCDP3-specific data; extrapolated from RCDP1/RCDP2 by analogy).
  VGB: HIGH RISK — cataracts 55-65% + VGB visual field loss = additive risk. Not absolute CI
    (unlike ZSD where retinopathy is universal/severe), but HIGH RISK. Monthly VF/VEP if used.
  VPA: RELATIVE CI — hepatotoxicity; POLG1 MANDATORY (CPIC Grade A). Phytanic NORMAL in RCDP3
    (less phytanic-hepatic stress than RCDP1), but POLG1/mitochondrial risk persists.
  PHT/CBZ/OXC: CAN USE — no adrenal insufficiency in RCDP3 (unlike ABCD1 = ABSOLUTE CI).
  Alkylglycerol supplementation: BYPASSES AGPS BLOCK (Step 2 bypass). Experimental Level C.
    Rationale slightly different from RCDP2 (Step 1 bypass) but same end result: exogenous ether lipid.
  DHA: Level C (restores DHA-plasmalogen; same as RCDP1/RCDP2).
  Phytol-restricted diet: NOT REQUIRED unless phytanic mildly elevated (rare; usually <10% of RCDP3).
  No ERT (2026): AGPS is peroxisomal matrix enzyme — not secreted, not lysosomal. ERT not applicable.
  No HSCT: Pathology is hypomyelination (plasmalogen deficiency), not neuroinflammation.
"""

import random
random.seed(44)


# ── Overview (KPIs + summary) ─────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 58,
        "classic_rcdp_pct": 40,
        "intermediate_rcdp_pct": 40,
        "mild_rcdp_pct": 20,
        "drug_resistance_pct": 30,
        "cataract_pct": 60,
        "ichthyosis_pct": 58,
        "rhizomelia_pct": 100,
        "stippling_pct": 82,
        "on_alkylglycerol_pct": 28,
        "on_dha_pct": 35,
        "omim_gene": "603051",
        "omim_disease": "600121",
        "locus": "2q33.3",
        "vlcfa_normal": True,
        "vlcfa_normal_pct": 100,
        "phytanic_normal_pct": 85,
        "plasmalogen_low_pct": 97,
        "inheritance": "Autosomal recessive (AR), biallelic LOF",
        "common_variant": "No founder mutation — all private/rare variants; fewer than 30 cases worldwide 2026",
        "disease_mechanism": (
            "AGPS (Alkylglycerone Phosphate Synthase / ADHAPS) is the second enzyme in peroxisomal "
            "plasmalogen biosynthesis (658 aa, PTS1 import via C-terminal -AKL-). AGPS catalyzes the "
            "ether-bond formation step: 1-acyl-DHAP + fatty alcohol → 1-alkyl-DHAP + fatty acid. "
            "AGPS LOF blocks Step 2, so despite intact GNPAT (Step 1), the ether bond cannot be formed "
            "and all downstream plasmalogen synthesis fails. Plasmalogens severely low as in RCDP1/RCDP2, "
            "but phytanic acid is NORMAL (PEX7 intact → PHYH imported normally). Biochemically "
            "near-identical to RCDP2/GNPAT — distinguished only by gene sequencing."
        ),
        "nbs_positive_rate": (
            "Not in NBS; plasmalogens on DBS advocated; phytanic normal → not triggered by "
            "routine NBS peroxisomal screens. Rarest RCDP subtype — often missed or diagnosed as RCDP2."
        ),
        "key_concepts": [
            "AGPS = ADHAPS (Alkylglycerone Phosphate Synthase / Alkyl-DHAP Synthase), Step 2 of plasmalogen synthesis (PTS1 import, -AKL-)",
            "RCDP3 accounts for ~5% of all RCDP (vs RCDP1/PEX7 ~90%; RCDP2/GNPAT ~5%); rarest subtype; <30 cases worldwide",
            "Plasmalogens (RBC) SEVERELY LOW — same as RCDP1/RCDP2; primary diagnostic biomarker (C16:0-DMA, C18:0-DMA)",
            "Phytanic acid NORMAL in RCDP3 (PEX7 intact → PHYH PTS2 import unaffected) — same as RCDP2, unlike RCDP1",
            "VLCFA NORMAL — PTS1 pathway intact, PEX1/PEX6 intact; if VLCFA elevated → ZSD, not RCDP",
            "No founder mutation — all private/rare variants; full gene sequencing required including both GNPAT and AGPS",
            "Cataracts 55-65%, rhizomelia 100%, stippled epiphyses 82% — same clinical hallmarks as RCDP1/RCDP2",
            "VGB HIGH RISK — cataracts + VGB visual field constriction = additive visual impairment",
            "VPA RELATIVE CI — hepatotoxicity; POLG1 MANDATORY (CPIC A); phytanic less of issue vs RCDP1",
            "PHT/CBZ/OXC CAN BE USED — no adrenal insufficiency in RCDP3 (unlike ABCD1 = ABSOLUTE CI)",
            "Alkylglycerols bypass AGPS block (Step 2) — pre-formed ether lipid bypasses both Step 1 and Step 2",
            "No ERT (2026) — peroxisomal matrix enzyme, not secreted/lysosomal; no therapeutic equivalent",
            "No HSCT — hypomyelination (not neuroinflammation); HSCT arrests inflammatory demyelination only",
            "Distinguished from RCDP2/GNPAT ONLY by gene sequencing — all biochemical markers near-identical",
            "Rarest RCDP: gene panels must include AGPS; RCDP3 may be underdiagnosed as RCDP2 without AGPS sequencing",
        ],
        "standards": [
            "de Vet EC, Ijlst L, Oostheim W, et al. Alkyl-dihydroxyacetonephosphate synthase: AGPS. J Biol Chem. 1998;273:10296–10301.",
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
            "name": "Classic RCDP3 (Null/Null or Severe LOF)",
            "pct": 40,
            "n": 16,
            "sex": "M/F equal",
            "onset_age": "Neonatal–3 months",
            "seizure_risk": "55–65%",
            "eeg": "Hypsarrhythmia (infantile spasms), multifocal spike-wave, burst-suppression",
            "mri": "Severe hypomyelination, reduced white matter T2 signal, cortical atrophy",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic null or severe LOF (frameshift/nonsense) → complete AGPS enzyme absence → "
                "ether-bond formation impossible. Plasmalogens virtually absent. Rhizomelia severe, "
                "congenital cataracts, profound ID. All variants are private (no founder allele). "
                "Phytanic NORMAL (PEX7 intact). GNPAT produces 1-acyl-DHAP but AGPS cannot process it."
            ),
        },
        {
            "name": "Intermediate RCDP3 (Null/Hypomorphic or Splice)",
            "pct": 40,
            "n": 16,
            "sex": "M/F equal",
            "onset_age": "3–18 months",
            "seizure_risk": "25–40%",
            "eeg": "Modified hypsarrhythmia, focal spikes, multifocal discharges, theta slowing",
            "mri": "Moderate hypomyelination, periventricular signal changes",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "One null + one hypomorphic allele → partial AGPS activity. "
                "Plasmalogens 5–20% of normal. Moderate rhizomelia, moderate ID. Survival 10–20y. "
                "Phytanic NORMAL; phytol restriction rarely required."
            ),
        },
        {
            "name": "Mild RCDP3 (Hypomorphic/Hypomorphic)",
            "pct": 20,
            "n": 8,
            "sex": "M/F equal",
            "onset_age": "6–36 months",
            "seizure_risk": "15–20%",
            "eeg": "Focal discharges, sparse multifocal spikes, may be near-normal",
            "mri": "Minimal white matter changes; myelination near-normal",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic hypomorphic alleles → residual AGPS activity (10–35% of normal). "
                "Plasmalogens 20–50% of normal. Minimal rhizomelia, mild-moderate ID, adult survival reported. "
                "Mildest RCDP3 phenotype; may be diagnosed in childhood or even adult life."
            ),
        },
    ]

    phenotypes = (
        [("Classic RCDP3", "Severe")] * 16
        + [("Intermediate RCDP3", "Intermediate")] * 16
        + [("Mild RCDP3", "Mild")] * 8
    )
    sexes = (["M"] * 20 + ["F"] * 20)
    random.shuffle(sexes)

    genotype_map = {
        "Classic RCDP3": ["Biallelic null (frameshift/nonsense)", "Null + severe splice", "Homozygous deletion"],
        "Intermediate RCDP3": ["Null + hypomorphic missense", "Splice variant + missense", "Compound heterozygous missense"],
        "Mild RCDP3": ["Biallelic hypomorphic missense", "Compound hypomorphic", "Residual activity alleles"],
    }

    seizure_map = {
        "Severe": ["Infantile spasms", "Multifocal clonic", "Myoclonic"],
        "Intermediate": ["Focal aware", "Focal to bilateral GTCS", "Myoclonic"],
        "Mild": ["Focal aware", "GTCS", None],
    }

    trigger_pool = [
        "Febrile illness", "Sleep deprivation", "Missed AED", "Stress / arousal",
        "Photic stimulation", "Metabolic decompensation", "Anaesthesia (low risk — no adrenal)",
    ]

    treatment_pool = [
        "LEV (first-line)", "ACTH (IS — Level B)", "CLB (adjunct — Level C)",
        "LTG (adjunct — Level C)", "DHA supplementation (Level C)",
        "Alkylglycerol supplementation (Step 2 bypass — experimental Level C)",
    ]

    patients = []
    for i, ((pheno, severity), sex) in enumerate(zip(phenotypes, sexes)):
        pid = f"RCDP3-{i+1:02d}"
        age_mo = random.randint(0, 3) if severity == "Severe" else (
            random.randint(3, 18) if severity == "Intermediate" else random.randint(6, 36)
        )
        has_sz = random.random() < (0.60 if severity == "Severe" else 0.33 if severity == "Intermediate" else 0.18)
        seizure_type = random.choice(seizure_map[severity]) if has_sz else None
        sz_free = random.choice([True, False]) if has_sz else True
        drug_resistant = has_sz and not sz_free and random.random() < 0.40
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
            "cataracts": random.random() < 0.60,
            "ichthyosis": random.random() < 0.58,
            "rhizomelia": True,
            "stippled_epiphyses": random.random() < 0.82,
            "polg1_tested": True,
        })

    seizure_types = [
        {"type": "Infantile Spasms / Hypsarrhythmia", "pct": 37, "preferred_tx": "ACTH Level B"},
        {"type": "Focal (aware / impaired awareness)", "pct": 30, "preferred_tx": "LEV first-line"},
        {"type": "Myoclonic", "pct": 17, "preferred_tx": "LEV / CLB adjunct"},
        {"type": "Focal to Bilateral GTCS", "pct": 13, "preferred_tx": "LEV + CLB"},
        {"type": "Status Epilepticus", "pct": 7, "preferred_tx": "IV LEV / MDZ; fosphenytoin acceptable (no adrenal)"},
    ]

    triggers = [
        {"trigger": "Febrile illness", "pct": 60},
        {"trigger": "Sleep deprivation", "pct": 38},
        {"trigger": "Missed AED dose", "pct": 33},
        {"trigger": "Stress / arousal", "pct": 28},
        {"trigger": "Photic stimulation", "pct": 15},
        {"trigger": "Metabolic decompensation", "pct": 18},
        {"trigger": "Anaesthesia (low risk; no adrenal mechanism unlike ABCD1)", "pct": 10},
    ]

    monitoring = [
        {"parameter": "RBC plasmalogens (C16:0-DMA, C18:0-DMA)", "threshold": "Severely low (<5% normal)", "frequency": "Every 6 months"},
        {"parameter": "Plasma phytanic acid", "threshold": "Usually NORMAL (<30 µmol/L); monitor if diet uncertain", "frequency": "Annually"},
        {"parameter": "Plasma VLCFA (C26:0)", "threshold": "NORMAL (if elevated → reconsider ZSD)", "frequency": "At diagnosis; annually"},
        {"parameter": "LFTs (if VPA used — RELATIVE CI)", "threshold": "Normal → q3 months if VPA used", "frequency": "q3 months on VPA"},
        {"parameter": "EEG", "threshold": "Baseline + after treatment change", "frequency": "q6–12 months"},
        {"parameter": "Ophthalmology / Visual field / VEP", "threshold": "Baseline cataracts screen; VF mandatory if VGB used", "frequency": "Annually; monthly VF if VGB"},
        {"parameter": "DHA plasma", "threshold": "Low → supplement 200 mg/day (Level C)", "frequency": "Every 6 months"},
    ]

    lifecycle = [
        {"stage": "Neonatal (0–1 mo)", "features": "Stippled epiphyses on X-ray, rhizomelia, congenital cataracts; biochemistry: plasmalogens severely low, phytanic NORMAL, VLCFA NORMAL"},
        {"stage": "Infant (1–12 mo)", "features": "Infantile spasms onset (35-40%); hypsarrhythmia; ACTH Level B; VGB to be avoided (cataracts); DHA + alkylglycerol started"},
        {"stage": "Toddler (1–3 yr)", "features": "Profound ID, feeding difficulties, recurrent infections; focal/GTCS seizures; LEV continued; ophthalmology monitoring"},
        {"stage": "Child (3–10 yr)", "features": "Drug-resistant epilepsy 30%; polypharmacy CLB/LTG; contractures progressive; cataracts surgery if needed"},
        {"stage": "Adolescent (10–18 yr)", "features": "Severe form: palliative transition; mild-intermediate: supported living, modified AED regimen"},
        {"stage": "Adult (>18 yr — mild forms only)", "features": "Mild RCDP3 adults: LEV maintained; alkylglycerol/DHA supplementation; annual metabolic review"},
    ]

    treatments = [
        {
            "drug": "LEV (Levetiracetam)",
            "class": "First-line",
            "level": "Level B (RCDP3 analogy from RCDP1/RCDP2)",
            "dose": "20–40 mg/kg/day",
            "notes": "No peroxisomal interactions, no adrenal mechanism, safe in all RCDP3 forms",
            "ci": "None specific to RCDP3",
        },
        {
            "drug": "ACTH (Tetracosactide)",
            "class": "IS — first-line",
            "level": "Level B (RCDP3 analogy)",
            "dose": "150 IU/m²/day (standard IS protocol)",
            "notes": "Preferred over VGB for IS in RCDP3 (avoids additive visual risk with cataracts)",
            "ci": "VGB avoided in RCDP3 due to cataract + VF risk",
        },
        {
            "drug": "CLB (Clobazam)",
            "class": "Adjunct",
            "level": "Level C",
            "dose": "0.1–0.3 mg/kg/day",
            "notes": "Adjunct for focal/GTCS; safe; no peroxisomal interactions",
            "ci": "None specific",
        },
        {
            "drug": "LTG (Lamotrigine)",
            "class": "Adjunct",
            "level": "Level C",
            "dose": "1–5 mg/kg/day (slow titration)",
            "notes": "Adjunct in older children/adults; avoid rapid titration rash",
            "ci": "Avoid VPA co-administration (dose interaction)",
        },
        {
            "drug": "DHA (Docosahexaenoic acid)",
            "class": "Supplement",
            "level": "Level C",
            "dose": "200 mg/day (infants); higher in adults",
            "notes": "Restores DHA-plasmalogen; no drug interactions; safe",
            "ci": "None",
        },
        {
            "drug": "Alkylglycerols (batyl/chimyl/selachyl alcohol)",
            "class": "Experimental supplement — Step 2 bypass",
            "level": "Experimental Level C",
            "dose": "Not standardised; ~1–2 g/day as shark liver oil",
            "notes": "Bypasses AGPS block (Step 2): pre-formed ether lipid restores plasmalogens independently of Steps 1+2. Strongest rationale: raises RBC plasmalogens in animal models. Human data n<20 across all RCDP.",
            "ci": "None known; quality control of commercial shark liver oil preparations variable",
        },
    ]

    contraindications = [
        {
            "drug": "VGB (Vigabatrin)",
            "level": "HIGH RISK",
            "reason": "Cataracts 55-65% in RCDP3 + VGB irreversible visual field constriction = additive blindness risk. NOT absolute CI (ZSD retinopathy is universal/severe; RCDP3 cataracts are distinct). Monthly VF/VEP monitoring mandatory if used.",
            "alternative": "ACTH (Level B) for IS; LEV/CLB for other seizure types",
        },
        {
            "drug": "VPA (Valproate)",
            "level": "RELATIVE CI",
            "reason": "Hepatotoxicity risk. POLG1 MANDATORY (CPIC Grade A). Phytanic NORMAL in RCDP3 (unlike RCDP1 where phytanic elevated adds hepatic stress); POLG1/mitochondrial hepatotoxicity risk persists regardless. LFT q3 months if used.",
            "alternative": "LEV first-line; CLB/LTG adjunct",
        },
        {
            "drug": "PHT / CBZ / OXC",
            "level": "CAN USE (no adrenal CI)",
            "reason": "NO adrenal insufficiency in RCDP3 → no CYP3A4 cortisol drop → no adrenal crisis. CONTRAST with ABCD1 (PHT = ABSOLUTE CI due to cortisol drop → adrenal crisis). Safe in RCDP3 as enzyme-inducing AEDs carry no adrenal risk.",
            "alternative": "LEV preferred first-line; PHT/CBZ acceptable if needed",
        },
        {
            "drug": "Typical antipsychotics",
            "level": "HIGH RISK EPS",
            "reason": "Hypomyelination increases susceptibility to EPS. Atypical antipsychotics preferred if psychiatry management required.",
            "alternative": "Atypical antipsychotics at lowest effective dose",
        },
        {
            "drug": "Fasting / prolonged NPO",
            "level": "CAUTION (lower risk than RCDP1)",
            "reason": "In RCDP3, phytanic is NORMAL so fasting-induced phytanic surge is NOT a primary concern (unlike RCDP1 where adipose phytanic mobilisation during fasting triggers neurological crisis). IV dextrose precaution still reasonable in anaesthesia for metabolic stability.",
            "alternative": "IV glucose for prolonged fasting; avoid ketogenic diet in severe forms",
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


# ── Definitions (glossary + algorithm + pharmacology + DDx) ──────────────────

def get_definitions():
    return {
        "key_concepts": [
            "AGPS = Alkylglycerone Phosphate Synthase (ADHAPS / alkyl-DHAP synthase): 658-aa peroxisomal matrix enzyme using PTS1 import (C-terminal -AKL-). Step 2 of plasmalogen biosynthesis — ETHER BOND FORMATION.",
            "RCDP3 mechanism: AGPS LOF blocks ether-bond formation (1-acyl-DHAP + fatty alcohol → 1-alkyl-DHAP). GNPAT (Step 1) is intact; 1-acyl-DHAP accumulates but cannot be converted. All downstream plasmalogen synthesis fails.",
            "Plasmalogen biosynthesis — 2-step peroxisomal pathway: Step 1 GNPAT (RCDP2 target): DHAP + acyl-CoA → 1-acyl-DHAP. Step 2 AGPS (RCDP3 target): 1-acyl-DHAP + fatty alcohol → 1-alkyl-DHAP (ether bond formed). RCDP1 (PEX7) blocks IMPORT of AGPS and PHYH (PTS2 cargo) simultaneously.",
            "Phytanic acid in RCDP3: NORMAL (key distinction from RCDP1). PEX7 is intact in RCDP3 → PHYH (phytanoyl-CoA hydroxylase, PTS2) is imported normally → phytanic catabolism intact. Phytanic elevated → suspect RCDP1 (PEX7) or Adult Refsum (PHYH).",
            "Alkylglycerol pharmacology in RCDP3: BYPASSES STEP 2 BLOCK. Batyl/chimyl/selachyl alcohol provide pre-formed 1-O-alkyl-sn-glycerol downstream of AGPS — the ether bond is already formed. Compare RCDP2 where alkylglycerols bypass Step 1 (GNPAT); in RCDP3 they bypass Step 2 (AGPS). Experimental Level C for both subtypes.",
            "VLCFA in RCDP3: NORMAL — C26:0 beta-oxidation uses PTS1 (ACOX1 enzyme); PEX7 and AGPS are irrelevant to VLCFA oxidation. Elevated VLCFA → suspect ZSD (PEX1/PEX6 deficiency), not RCDP.",
            "RBC plasmalogens — severely low/absent in RCDP3. Same primary diagnostic biomarker as RCDP1/RCDP2. C16:0-DMA and C18:0-DMA dimethylacetals measured by GC-MS; confirms ether-lipid synthesis defect.",
            "VGB in RCDP3: HIGH RISK. Cataracts 55-65% + VGB irreversible visual field constriction = additive blindness risk. Not absolute CI (ZSD universal retinopathy is distinct). Monthly VF/VEP mandatory if VGB unavoidable.",
            "VPA in RCDP3: RELATIVE CI. POLG1 MANDATORY (CPIC Grade A). Phytanic NORMAL in RCDP3 (less phytanic hepatic stress than RCDP1), but POLG1-dependent mitochondrial hepatotoxicity risk persists. LFT q3 months if VPA used.",
            "PHT/CBZ/OXC in RCDP3: CAN BE USED — no adrenal insufficiency (unlike ABCD1 = ABSOLUTE CI via CYP3A4 cortisol drop). No adrenal gland involvement in RCDP. Adrenal crisis mechanism does not apply.",
            "No ERT in RCDP3: AGPS is peroxisomal matrix enzyme (PTS1-imported; not secreted; not lysosomal). ERT only replaces enzymes that are endocytosable/secreted. No therapeutic equivalent for cytoplasmic/peroxisomal matrix enzymes in 2026.",
            "No HSCT in RCDP3: Pathology is hypomyelination (plasmalogen deficiency impairs myelin sheath synthesis). HSCT arrests neuroinflammatory demyelination (ABCD1-CCALD, Krabbe). Not relevant for primary hypomyelination.",
            "RCDP3 vs RCDP2 — biochemically near-identical: Both have plasmalogens severely low, phytanic NORMAL, VLCFA NORMAL. Alkylglycerols bypass both (at Step 1 or Step 2). Phytol-restricted diet not required in either unless phytanic mildly elevated. Distinguished ONLY by gene sequencing (AGPS vs GNPAT).",
            "No founder mutation in RCDP3: All known variants are private or rare. Full gene sequencing of AGPS required. Gene panel must include AGPS even if GNPAT negative — biochemical near-identity means RCDP2/RCDP3 cannot be separated clinically.",
            "Rarest RCDP subtype: <30 reported cases worldwide (2026). RCDP3 may be underdiagnosed: patients with negative PEX7 (RCDP1 most common screen) and negative GNPAT may not have AGPS tested if panel is incomplete.",
        ],
        "diagnostic_algorithm": [
            "Step 1: Clinical suspicion — rhizomelia + stippled epiphyses (neonate) OR congenital cataracts + ichthyosis + hypotonia (infant) OR unexplained epilepsy + intellectual disability + skeletal abnormality.",
            "Step 2: Plasma VLCFA (C26:0, C26:0/C22:0, C24:0/C22:0): must be NORMAL. If elevated → ZSD (PEX1/PEX6), not RCDP. VLCFA normal excludes ZSD, directs to RCDP workup.",
            "Step 3: RBC PLASMALOGENS (C16:0-DMA, C18:0-DMA): SEVERELY LOW confirms ether-lipid defect. Present in all 3 RCDP types (1, 2, 3).",
            "Step 4: Plasma PHYTANIC ACID: CHECK. NORMAL in RCDP3 (PEX7 intact). If ELEVATED → RCDP1 (PEX7), not RCDP3. Mildly elevated phytanic (<5%) in RCDP3 is an exception; usually normal.",
            "Step 5: Plasma PRISTANIC ACID — NORMAL in RCDP3. Elevated → ZSD.",
            "Step 6: Plasma PIPECOLIC ACID — NORMAL in RCDP3. Elevated → ZSD.",
            "Step 7: Skeletal X-ray — rhizomelia (humerus/femur shortening), stippled epiphyses (neonatal/infant), progressive joint contractures.",
            "Step 8: Ophthalmology — cataracts screen (55-65%); formal visual field assessment baseline; ERG if VGB considered (ERG detects early retinal toxicity).",
            "Step 9: MRI brain — hypomyelination (reduced T2 white matter signal), cortical atrophy in severe forms.",
            "Step 10: GENE PANEL: PEX7 first (90% RCDP alleles). If PEX7 negative → GNPAT + AGPS in PARALLEL (biochemically identical — one test cannot separate them). Full exon sequencing both genes; no founder mutation in either.",
            "Step 11: POLG1 genotyping MANDATORY before any VPA consideration (CPIC Grade A). Even though phytanic is normal in RCDP3, POLG1-dependent hepatotoxicity risk persists.",
            "Step 12: Confirm AGPS diagnosis; institute DHA supplementation (Level C); alkylglycerol supplementation if plasmalogens near-absent (experimental Level C; bypasses Step 2 AGPS block); contact specialist centre (Kennedy Krieger, Amsterdam, Manchester).",
        ],
        "pharmacological_distinctions": [
            "LEV — FIRST-LINE all RCDP3 forms (no peroxisomal interactions, no adrenal mechanism, no hepatotoxicity concern, no enzyme induction).",
            "ACTH — Level B infantile spasms (extrapolated from RCDP1/RCDP2; no RCDP3-specific RCT data). Preferred over VGB as first-line IS therapy to avoid cataract + VF risk.",
            "VGB — HIGH RISK in RCDP3 (cataracts 55-65% + VGB VF constriction = additive). Not absolute CI (unlike ZSD universal retinopathy). Monthly VF/VEP monitoring mandatory if used.",
            "VPA — RELATIVE CI. POLG1 MANDATORY (CPIC Grade A). Phytanic NORMAL in RCDP3 (reduced phytanic-hepatic stress vs RCDP1) but POLG1/mitochondrial hepatotoxicity risk remains. LFT q3 months.",
            "PHT/CBZ/OXC — CAN BE USED (no adrenal insufficiency; no cortisol drop; no adrenal crisis risk). Adrenal involvement does NOT occur in RCDP. Contrast: ABCD1 PHT = ABSOLUTE CI.",
            "Fosphenytoin IV — acceptable in SE when IV LEV fails (no adrenal mechanism). Safe alternative to standard phenytoin in acute SE.",
            "CLB / benzodiazepines — safe adjuncts; MDZ intranasal/buccal for acute cluster seizures; CLB oral for chronic adjunct; no peroxisomal interactions.",
            "Typical antipsychotics — HIGH RISK EPS (hypomyelination increases susceptibility). Use atypical antipsychotics at lowest effective dose.",
            "DHA supplementation — Level C. Replaces DHA-plasmalogen deficit; 200 mg/day infants; higher doses in older patients. Safe, no drug interactions.",
            "Alkylglycerol supplementation — Experimental Level C. Bypasses AGPS block (Step 2): pre-formed ether lipid (batyl/chimyl/selachyl alcohol) restores plasmalogens independently of AGPS. Raises RBC plasmalogens in animal models; human data n<20 pooled across all RCDP subtypes.",
            "Phytol-restricted diet — NOT REQUIRED unless phytanic mildly elevated (<10% of RCDP3 patients). RCDP3 phytanic typically normal; phytol restriction is only critical in RCDP1 (PHYH import blocked → phytanic elevated from dietary phytol).",
            "No ERT, No HSCT, No gene therapy (2026) — no approved biological disease-modifying therapy. Alkylglycerol + DHA are the only experimental/supportive supplementation strategies.",
        ],
        "differential_diagnosis": [
            {"condition": "RCDP1 (PEX7)", "distinction": "Phytanic ELEVATED in RCDP1 (PHYH cannot be imported; PEX7 absent); phytanic NORMAL in RCDP3 (PEX7 intact). Plasmalogens severely low in both. RCDP1 = 90% of RCDP; RCDP3 = ~5%. Distinguished by PEX7 gene sequencing (RCDP1 has L292X 50% European founder)."},
            {"condition": "RCDP2 (GNPAT)", "distinction": "Biochemically near-identical to RCDP3 (both: plasmalogens severely low, phytanic NORMAL, VLCFA NORMAL). RCDP2 blocks Step 1 (GNPAT); RCDP3 blocks Step 2 (AGPS). Alkylglycerols bypass both. Distinguished ONLY by gene sequencing GNPAT vs AGPS."},
            {"condition": "ZSD — Zellweger Spectrum (PEX1/PEX6)", "distinction": "VLCFA HIGH in ZSD; VLCFA NORMAL in RCDP3. ZSD: ALL peroxisomal functions impaired simultaneously (VLCFA, plasmalogens, phytanic, pristanic, pipecolic all abnormal). RCDP3: only ether-lipid synthesis at Step 2. No rhizomelia/stippling in ZSD (cortical migration defects: pachygyria/PMG instead)."},
            {"condition": "Adult Refsum Disease (PHYH)", "distinction": "Phytanic ELEVATED in Refsum (PHYH deficiency; PTS2 enzyme LOF); phytanic NORMAL in RCDP3. Plasmalogens NORMAL in Refsum (plasmalogen synthesis intact). Adult onset, peripheral neuropathy, retinitis pigmentosa. No rhizomelia/stippling."},
            {"condition": "CDPX2 / Conradi-Hünermann (EBP)", "distinction": "X-linked dominant stippling; ALL peroxisomal markers NORMAL (VLCFA, plasmalogens, phytanic). Asymmetric ichthyosis (Blaschko lines). Sterol isomerase (EBP) LOF — cholesterol pathway, not peroxisomal. Affects females predominantly."},
            {"condition": "ABCD1 (X-ALD)", "distinction": "VLCFA HIGH; plasmalogens NORMAL (not low). No rhizomelia/stippling. Adrenal insufficiency 71% males. PHT = ABSOLUTE CI (adrenal crisis via CYP3A4 cortisol drop). HSCT indicated (CCALD). X-linked. Contrast with RCDP3 where PHT CAN BE USED and no adrenal involvement."},
            {"condition": "Non-peroxisomal skeletal dysplasias", "distinction": "All peroxisomal markers NORMAL (VLCFA, plasmalogens, phytanic). FGFR3 (achondroplasia), RMRP (cartilage-hair hypoplasia), or other skeletal gene mutations. Rhizomelia may be present but no cataracts + ichthyosis + stippling + epilepsy combination."},
        ],
        "standards": [
            "de Vet EC, Ijlst L, Oostheim W, et al. Alkyl-dihydroxyacetonephosphate synthase: putative role of the active site histidine in analogy with DHAP-AT. J Biol Chem. 1998;273:10296–10301.",
            "Honsho M, Fujiki Y. Plasmalogen homeostasis: regulation of plasmalogen biosynthesis. FEBS Lett. 2017;591(18):2741–2755.",
            "Braverman NE, Raymond GV, Rizzo WB, et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Am J Med Genet C. 2016;172(4):375–386.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism. Biochim Biophys Acta. 2016;1863(5):922–933.",
            "Wanders RJA, Waterham HR. Biochemistry of mammalian peroxisomes revisited. Annu Rev Biochem. 2006;75:295–332.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org.",
            "Braverman NE et al. Rhizomelic Chondrodysplasia Punctata Type 1. GeneReviews. NCBI Bookshelf. 2015.",
        ],
    }
