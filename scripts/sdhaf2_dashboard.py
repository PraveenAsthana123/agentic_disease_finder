#!/usr/bin/env python3
"""SDHAF2 (SDH5 / PGL2) — Succinate Dehydrogenase Assembly Factor 2 / SDHA Flavinylation Factor /
Paragangliomas 2 (PGL2) — Autosomal Dominant with Maternal Imprinting.

SDHAF2 (Succinate Dehydrogenase Assembly Factor 2; also known as SDH5; OMIM *613019) encodes a
166-amino-acid, ~19 kDa mitochondrial matrix protein that functions as the SDHA flavinylation
factor — it covalently attaches the FAD cofactor to SDHA histidine-99 (His99), enabling
succinate oxidation and ubiquinone reduction by Complex II (succinate dehydrogenase, SDH).

  SDHAF2 gene   OMIM *613019
  Disease       Paragangliomas 2 (PGL2)  OMIM #601650
  Inheritance   AD (autosomal dominant) with MATERNAL IMPRINTING
                — only PATERNAL transmission causes disease
  Chromosome    11q13.1

Reference: Hao HX et al. (2009) SDH5, a gene required for flavination of succinate
dehydrogenase, is mutated in paraganglioma. Science 325(5944):1139–1142.
(First SDHAF2/SDH5 disease gene identification; Dutch founder p.Arg107Leu; 14 families)

Reference: Kunst HP et al. (2011) SDHAF2 mutations in familial and sporadic paraganglioma
and phaeochromocytoma. Lancet Oncol 12(8):748–754.
(SDHAF2 mutation spectrum; maternal imprinting confirmation; PGL2 clinical series)

Reference: Baysal BE (2013) Genetics of paraganglioma/pheochromocytoma. Curr Opin Oncol
25(1):6–11. (Review: SDH gene family, SDHAF2 PGL2, imprinting, surveillance)

PATHOPHYSIOLOGY (SDHAF2 / SDHA Flavinylation Factor):
  SDHAF2/SDH5 is the SDHA flavinylation factor — it covalently attaches FAD to SDHA His99:
    1. SDHAF2 binds SDHA in the mitochondrial matrix and acts as a chaperone/scaffold
       to facilitate autocatalytic FAD attachment to SDHA His99 (covalent flavinylation).
    2. Without SDHAF2, SDHA cannot be flavinylated, cannot bind its catalytic FAD cofactor,
       and cannot catalyze succinate → fumarate oxidation.
    3. Unflavinated SDHA cannot assemble into the SDHA-SDHB catalytic core;
       CII holoenzyme formation is blocked.
    4. CII deficiency → succinate accumulates → pseudo-hypoxic signalling via HIF1α
       (succinate inhibits PHD enzymes → HIF1α stabilisation → pro-angiogenic/tumorigenic).
    5. Paraganglioma (head-neck, retroperitoneal) and pheochromocytoma develop via this
       pseudo-hypoxic pathway. NOT infantile leukoencephalopathy (unlike SDHAF1).
    6. MATERNAL IMPRINTING (genomic imprinting): the maternal SDHAF2 allele is
       epigenetically silenced. Only paternal SDHAF2 mutations cause PGL2.
       Children of female SDHAF2 carriers: not at risk. Children of male carriers: 50% risk.
    7. DOMINANT — a single heterozygous loss-of-function mutation (autosomal dominant
       with parental imprinting) is sufficient for paraganglioma predisposition.

SDHAF2 UNIQUE FEATURES:
  1. SDHA FLAVINYLATION FACTOR — UNIQUE: SDHAF2 is the dedicated scaffold that enables
     autocatalytic FAD attachment to SDHA His99. No other SDH assembly factor performs
     this function. SDHAF1 delivers FeS to SDHB; SDHAF2 flavinates SDHA — different
     subunits, different cofactors, different steps, different diseases.
  2. MATERNAL GENOMIC IMPRINTING: PGL2 (SDHAF2) is one of only two SDH-related PGL loci
     with documented parental imprinting. Both SDHAF2 (PGL2) and SDHD (PGL1, 11q23.1)
     show maternal imprinting — only paternal transmission is penetrant. SDHB and SDHC
     do NOT show imprinting. This is critical for genetic counselling.
  3. DOMINANT PARAGANGLIOMA — NOT RECESSIVE CII DEFICIENCY: SDHAF2/PGL2 causes
     hereditary paraganglioma (dominant tumor predisposition), completely different from
     SDHAF1 (recessive infantile leukoencephalopathy). Same gene family, opposite
     inheritance, opposite diseases.
  4. 11q13.1 LOCUS — SAME CHROMOSOME AS SDHD (11q23.1): Both are on chromosome 11
     with maternal imprinting, but SDHAF2 (11q13.1) and SDHD (11q23.1) are 10 Mb apart.
     WES/panel distinguishes. SDHAF2 = PGL2; SDHD = PGL1. Both paternal-only transmission.
  5. FAD-ATTACHMENT STEP: SDHAF2 acts at Step 1 of CII assembly (SDHA flavinylation),
     before SDHAF1 (Step 2, SDHB FeS delivery). Temporal sequence matters for DDx.
  6. SDH-DEFICIENT GIST AND RCC: SDHAF2 mutations also found in SDH-deficient
     gastrointestinal stromal tumors (GISTs) and rare renal cell carcinoma — same
     pseudo-hypoxic HIF1α pathway as paraganglioma.

DISTINGUISHING FEATURES vs OTHER SDH/PARAGANGLIOMA GENES:
  vs SDHAF1 (19q13.12): SDHAF1 = CII deficiency infantile leukoencephalopathy (AR, recessive).
    SDHAF2 = PGL2 paraganglioma (AD, dominant, maternal imprinting). Completely different disease.
  vs SDHA (1p36.1): SDHA = structural FAD subunit; causes Leigh syndrome + rare PGL5 (dominant).
    SDHA mutations affect BOTH CII catalysis and tumor suppression; SDHAF2 = assembly factor only.
  vs SDHB (1p36.13): PGL4 — highest malignancy risk (~20–50%). No imprinting. AD.
    SDHB most common hereditary paraganglioma gene after SDHD.
  vs SDHC (1q23.3): PGL3 — head-neck paraganglioma; NO imprinting. AD.
    Lower malignancy risk than SDHB.
  vs SDHD (11q23.1): PGL1 — maternal imprinting (like SDHAF2); predominantly head-neck.
    SDHD 11q23.1 vs SDHAF2 11q13.1 — same chromosome, 10 Mb apart. WES mandatory.
  vs SDHAF3 (1q21.2): Rare; causes recessive CII deficiency similar to SDHAF1;
    SDHAF3 protects SDHB from oxidative FeS damage; very different from SDHAF2.
  vs MEN2 (RET, 10q11.21): MEN2 = MTC + pheochromocytoma + parathyroid; RET gain-of-function.
    SDHAF2 = paraganglioma/PCC only; SDH loss-of-function. No thyroid involvement.
  vs VHL (3p25.3): Von Hippel-Lindau; hemangioblastoma + ccRCC + PCC; VHL directly
    stabilizes HIF1α; SDHAF2 indirectly via succinate PHD inhibition.
"""

import random
import math

SEED = 703
rng  = random.Random(SEED)

GENE         = "SDHAF2"
OMIM_GENE    = "613019"
OMIM_DISEASE = "601650"
DISEASE_NAME = "Paragangliomas 2 (PGL2) — SDHA Flavinylation Factor / Hereditary Paraganglioma-Pheochromocytoma Syndrome (OMIM #601650)"
CHROMOSOME   = "11q13.1"
INHERITANCE  = "AD with MATERNAL IMPRINTING (paternal transmission only)"
N_PATIENTS   = 40

# ─── Variants ─────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.320G>T",
        "hgvs_p":    "p.Arg107Leu",
        "domain":    "FAD-binding/SDHA interface — arginine critical for SDHA His99 positioning",
        "mechanism": (
            "Dutch founder mutation (Hao 2009). Arginine-to-leucine at position 107 disrupts the "
            "SDHA-docking surface of SDHAF2, preventing correct positioning of SDHA His99 in the "
            "FAD-attachment site. The bulky hydrophobic leucine replaces the charged arginine guanidinium "
            "group that makes key contacts with SDHA. Autocatalytic FAD covalent attachment to SDHA His99 "
            "fails. SDHA remains apo-protein; CII cannot assemble; pseudo-hypoxic HIF1α pathway activated; "
            "paraganglioma develops."
        ),
        "severity":  "severe",
        "penetrance_pct": 90,
        "notes": "Dutch founder mutation. Identified in 14 Dutch PGL2 families (Hao 2009). Paternal transmission only (maternal imprinting). Head-neck paraganglioma predominant. High penetrance ~90% with paternal inheritance.",
    },
    {
        "hgvs_c":    "c.232G>C",
        "hgvs_p":    "p.Gly78Arg",
        "domain":    "FAD-binding region — glycine in flexible loop adjacent to SDHA contact surface",
        "mechanism": (
            "Glycine-to-arginine substitution introduces a bulky charged side chain into a flexible loop "
            "region critical for SDHA accommodation. The glycine at position 78 allows the loop to adopt "
            "the conformation needed for SDHA His99 juxtaposition. Arginine substitution creates steric "
            "clash and charge incompatibility, preventing SDHAF2 from correctly positioning SDHA for "
            "autocatalytic flavinylation. Loss of FAD attachment to SDHA; CII deficiency; succinate "
            "accumulation; HIF1α pseudo-hypoxia; paraganglioma."
        ),
        "severity":  "severe",
        "penetrance_pct": 85,
        "notes": "FAD-binding loop disruption. Severe loss of SDHA flavinylation. Head-neck paraganglioma. Maternal imprinting — paternal transmission only.",
    },
    {
        "hgvs_c":    "c.248A>G",
        "hgvs_p":    "p.Tyr83Cys",
        "domain":    "SDHA contact surface — tyrosine hydroxyl group critical for His99 approach",
        "mechanism": (
            "Tyrosine-to-cysteine substitution removes the phenolic hydroxyl group that makes a hydrogen "
            "bond with SDHA during the flavinylation reaction. The tyrosine 83 hydroxyl is part of the "
            "SDHAF2 catalytic scaffold that correctly orients SDHA His99 for autocatalytic FAD attachment. "
            "Loss of this interaction reduces efficiency of flavinylation, causing partial SDHA apo-protein "
            "accumulation. The cysteine thiol cannot substitute for tyrosine hydroxyl in this geometry."
        ),
        "severity":  "intermediate",
        "penetrance_pct": 75,
        "notes": "SDHA contact surface disruption. Partial FAD attachment — some residual flavinylation. Later-onset paraganglioma, intermediate penetrance.",
    },
    {
        "hgvs_c":    "c.134C>T",
        "hgvs_p":    "p.Ala45Val",
        "domain":    "Protein core packing — alanine in hydrophobic core of SDHAF2",
        "mechanism": (
            "Alanine-to-valine substitution in the hydrophobic core of SDHAF2 introduces mild steric "
            "strain due to the additional methyl group of valine. The protein may fold with slightly "
            "altered geometry, subtly disrupting the SDHA-docking surface. This is a hypomorphic allele "
            "with partial SDHAF2 function retained. Some SDHA flavinylation occurs, resulting in partial "
            "CII activity and reduced penetrance of paraganglioma."
        ),
        "severity":  "intermediate",
        "penetrance_pct": 65,
        "notes": "Hypomorphic core packing variant. Partial SDHAF2 function retained. Intermediate penetrance, later onset (3rd–4th decade). Paternal imprinting still applies.",
    },
    {
        "hgvs_c":    "c.IVS3+1G>A",
        "hgvs_p":    "p.splice_donor_intron3",
        "domain":    "Splice donor site — intron 3; partial exon skipping",
        "mechanism": (
            "Canonical splice donor site mutation at the +1 position of intron 3. Results in skipping "
            "of exon 3 or activation of a cryptic splice site, producing an out-of-frame transcript "
            "that undergoes nonsense-mediated decay. Functional SDHAF2 protein is severely reduced "
            "or absent. SDHA cannot be flavinylated; CII assembly blocked; pseudo-hypoxia. "
            "May generate small amounts of in-frame alternative transcript with residual partial function."
        ),
        "severity":  "severe",
        "penetrance_pct": 88,
        "notes": "Null splice-site allele. Near-complete loss of SDHAF2. High penetrance paraganglioma. Exon 3 encodes SDHA contact domain.",
    },
    {
        "hgvs_c":    "c.117G>A",
        "hgvs_p":    "p.Trp39Ter",
        "domain":    "N-terminal region — near-start nonsense; null allele",
        "mechanism": (
            "Tryptophan-to-stop codon at position 39. Produces a severely truncated 38-amino-acid "
            "peptide that lacks the SDHA-docking domain (residues 70–140). Truncated protein is "
            "non-functional and rapidly degraded. Complete loss of SDHAF2 function. No SDHA "
            "flavinylation; SDHA remains apo-protein; CII blocks; succinate-driven pseudo-hypoxia; "
            "paraganglioma at penetrant locus when inherited paternally."
        ),
        "severity":  "severe",
        "penetrance_pct": 92,
        "notes": "Null nonsense allele. Complete loss of SDHAF2. Highest penetrance category. Head-neck PGL, risk of bilateral. Paternal transmission required.",
    },
    {
        "hgvs_c":    "c.82C>T",
        "hgvs_p":    "p.Arg28Cys",
        "domain":    "N-terminal mitochondrial targeting/presequence interface",
        "mechanism": (
            "Arginine-to-cysteine substitution near the N-terminus in the mitochondrial targeting "
            "presequence region. May partially impair mitochondrial import efficiency, reducing the "
            "amount of SDHAF2 that reaches the matrix. Some SDHAF2 likely still imported and "
            "functional. Moderate phenotype: partial reduction of SDHA flavinylation, incomplete "
            "CII assembly, milder pseudo-hypoxia signal, later-onset or lower-penetrance paraganglioma."
        ),
        "severity":  "moderate",
        "penetrance_pct": 60,
        "notes": "Presequence/import disruption. Partial import impairment — moderate SDHAF2 reduction. Later-onset, lower-penetrance paraganglioma. Paternal transmission.",
    },
]

# ─── Tumour / clinical features ───────────────────────────────────────────────
TUMOUR_TYPES = [
    {"type": "Head-neck paraganglioma (HNPGL)", "freq_pct": 78},
    {"type": "Carotid body tumor",              "freq_pct": 55},
    {"type": "Jugulotympanic paraganglioma",    "freq_pct": 35},
    {"type": "Vagal paraganglioma",             "freq_pct": 22},
    {"type": "Retroperitoneal PGL",             "freq_pct": 18},
    {"type": "Adrenal PCC",                     "freq_pct": 15},
    {"type": "SDH-deficient GIST",              "freq_pct":  8},
    {"type": "Bilateral/multicentric PGL",      "freq_pct": 30},
]

CLINICAL_FEATURES = [
    {"feature": "Neck mass / pulsatile mass", "freq_pct": 72},
    {"feature": "Pulsatile tinnitus",         "freq_pct": 55},
    {"feature": "Cranial nerve palsy (IX-XII)","freq_pct": 35},
    {"feature": "Hearing loss",               "freq_pct": 28},
    {"feature": "Catecholamine excess symptoms","freq_pct":20},
    {"feature": "Hypertension (PCC-related)", "freq_pct": 18},
    {"feature": "Malignant transformation",   "freq_pct":  5},
    {"feature": "Bilateral synchronous tumors","freq_pct": 30},
]

def _pick_weighted(choices, weights):
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0
    for c, w in zip(choices, weights):
        cumulative += w
        if r < cumulative:
            return c
    return choices[-1]

def _pick_variant_pair():
    # Dominant — only ONE heterozygous pathogenic variant needed
    variant = rng.choice(VARIANTS)
    return {
        "allele_1_origin": "paternal",
        "allele_2_origin": "wild-type (maternal allele silenced by imprinting)",
        "variant_1": variant,
        "variant_2": None,  # heterozygous, dominant
    }

def _feature_frequencies_pct():
    return {
        "head_neck_pgl":     78,
        "adrenal_pcc":       15,
        "bilateral_tumors":  30,
        "malignant":          5,
        "sdh_gist":           8,
        "catecholamine_excess": 20,
        "maternal_imprinting_carrier_unaffected": 100,
    }

# ─── get_overview ──────────────────────────────────────────────────────────────
def get_overview() -> dict:
    rng.seed(SEED)

    patients = []
    for i in range(N_PATIENTS):
        pid = f"PGL2-{i+1:03d}"
        sex = rng.choice(["M", "F"])
        age_dx = rng.randint(28, 65)

        vp = _pick_variant_pair()
        v1 = vp["variant_1"]
        severity = v1["severity"]

        # Tumour type
        tumour_weights = [t["freq_pct"] for t in TUMOUR_TYPES]
        tumour = _pick_weighted([t["type"] for t in TUMOUR_TYPES], tumour_weights)

        # Penetrance — assume fully penetrant cohort (symptomatic patients)
        penetrant = True

        # Surveillance status
        surveillance = rng.choice(["Annual MRI/MRA", "Biennial imaging", "Surveillance pending", "Post-surgical follow-up"])

        patients.append({
            "patient_id":   pid,
            "sex":          sex,
            "age_at_dx":    age_dx,
            "variant":      v1["hgvs_p"],
            "hgvs_c":       v1["hgvs_c"],
            "severity":     severity,
            "inheritance":  "Paternal (maternal imprinting)",
            "tumour_type":  tumour,
            "penetrant":    penetrant,
            "surveillance": surveillance,
        })

    # Aggregates
    severity_counts = {"severe": 0, "intermediate": 0, "moderate": 0}
    tumour_counts   = {}
    sex_counts      = {"M": 0, "F": 0}
    for p in patients:
        sev = p["severity"]
        if sev in severity_counts:
            severity_counts[sev] += 1
        ttype = p["tumour_type"]
        tumour_counts[ttype] = tumour_counts.get(ttype, 0) + 1
        sex_counts[p["sex"]] += 1

    age_values = [p["age_at_dx"] for p in patients]
    avg_age_dx = round(sum(age_values) / len(age_values), 1)

    freqs = _feature_frequencies_pct()

    return {
        "gene": GENE,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "disease_name": DISEASE_NAME,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "n_patients": N_PATIENTS,
        "seed": SEED,
        "cohort_summary": {
            "total_patients": N_PATIENTS,
            "male": sex_counts["M"],
            "female": sex_counts["F"],
            "avg_age_at_dx": avg_age_dx,
            "severity_distribution": severity_counts,
            "tumour_type_distribution": tumour_counts,
        },
        "clinical_features_pct": freqs,
        "key_clinical_facts": {
            "head_neck_pgl_pct":      "78% — carotid body, jugulotympanic, vagal paraganglioma",
            "adrenal_pcc_pct":        "15% — pheochromocytoma (catecholamine-secreting)",
            "bilateral_tumors_pct":   "30% — bilateral/multicentric; annual surveillance mandatory",
            "malignant_pct":          "~5% — low malignancy risk (cf. SDHB ~20–50%)",
            "sdh_gist_pct":           "8% — SDH-deficient GIST; same pseudo-hypoxic mechanism",
            "maternal_imprinting":    "100% — maternal allele silenced; ONLY paternal transmission causes disease",
            "age_of_onset":           "3rd–5th decade typical; earlier in severe/null variants",
        },
        "sdhaf2_module_summary": {
            "protein":          "SDHAF2 (SDH5): 166 aa, ~19 kDa, mitochondrial matrix",
            "function":         "SDHA flavinylation factor — covalently attaches FAD to SDHA His99 (autocatalytic)",
            "assembly_step":    "Step 1 of CII assembly: SDHA flavinylation → SDHAF1 SDHB FeS insertion → SDHAF3/4 → CII holoenzyme",
            "disease_mechanism":"SDHAF2 LOF → SDHA apo-protein → CII assembly failure → succinate accumulates → PHD inhibition → HIF1α stabilisation → pseudo-hypoxia → paraganglioma",
            "imprinting":       "Maternal imprinting: maternal SDHAF2 allele silenced. Only paternal heterozygous mutations penetrant.",
            "sdhaf2_vs_sdhaf1": (
                "SDHAF2 (11q13.1) = dominant PGL2 paraganglioma (FAD attachment to SDHA) | "
                "SDHAF1 (19q13.12) = recessive CII deficiency leukoencephalopathy (FeS delivery to SDHB). "
                "Same assembly pathway, opposite subunits, opposite inheritance, opposite diseases."
            ),
        },
        "patients": patients[:10],  # preview
    }


# ─── get_breakdown ─────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    rng.seed(SEED)

    variants_summary = []
    for v in VARIANTS:
        n_patients = max(2, int(N_PATIENTS * rng.uniform(0.08, 0.22)))
        variants_summary.append({
            "hgvs_p":       v["hgvs_p"],
            "hgvs_c":       v["hgvs_c"],
            "domain":       v["domain"],
            "severity":     v["severity"],
            "penetrance_pct": v["penetrance_pct"],
            "n_patients":   n_patients,
            "mechanism_summary": v["mechanism"][:200] + "...",
            "notes":        v["notes"],
        })

    tumour_breakdown = []
    for t in TUMOUR_TYPES:
        n = max(1, round(N_PATIENTS * t["freq_pct"] / 100 * rng.uniform(0.8, 1.2)))
        tumour_breakdown.append({
            "tumour_type":   t["type"],
            "freq_pct":      t["freq_pct"],
            "n_estimated":   n,
        })

    clinical_breakdown = []
    for c in CLINICAL_FEATURES:
        n = max(1, round(N_PATIENTS * c["freq_pct"] / 100 * rng.uniform(0.8, 1.2)))
        clinical_breakdown.append({
            "feature":    c["feature"],
            "freq_pct":   c["freq_pct"],
            "n_estimated": n,
        })

    # Imprinting analysis
    imprinting_summary = {
        "mechanism": "Maternal imprinting of SDHAF2 locus (11q13.1)",
        "consequence": "Maternal SDHAF2 allele is epigenetically silenced (methylation of maternal allele promoter/imprint control region).",
        "penetrance_rule": "ONLY paternal SDHAF2 mutations are disease-causing. Children of female SDHAF2 mutation carriers: NOT at risk (maternal copy silenced). Children of male SDHAF2 mutation carriers: 50% risk.",
        "analogous_loci": ["SDHD (PGL1, 11q23.1) — also maternal imprinting", "H19/IGF2 (11p15.5) — canonical imprinted region, same chromosome"],
        "clinical_counselling": (
            "A female SDHAF2 carrier need not be surveilled for paraganglioma "
            "(her one active copy is paternal, which is normal; her mutant copy is maternal and silent). "
            "Her children inherit a silenced maternal copy — no risk. "
            "A male SDHAF2 carrier transmits an active mutant paternal copy — 50% of children at risk."
        ),
        "genetic_testing_implication": (
            "When SDHAF2 mutation found in proband, confirm paternal origin. "
            "If maternal origin: consider re-evaluation of imprinting status or variant pathogenicity. "
            "If paternal origin: all children require surveillance; siblings of affected father need paternal-allele analysis."
        ),
    }

    # Surveillance protocol
    surveillance_protocol = {
        "head_neck": "Annual MRI/MRA (skull base to aortic arch) — highest yield location",
        "adrenal_abdomen": "Annual contrast CT or MRI (adrenal, retroperitoneum) — 15% PCC risk",
        "biochemistry": "Annual plasma/urine metanephrines, chromogranin A",
        "starting_age": "Age 15 years (or 5 years before youngest affected family member)",
        "genetic_testing": "At-risk individuals: test for familial SDHAF2 mutation. Confirm paternal origin.",
        "sdh_gist": "Upper GI endoscopy if symptomatic; abdominal MRI for surveillance in high-risk",
    }

    # Treatment summary
    treatment_summary = {
        "surgery": "First-line for resectable paraganglioma; functional cure possible",
        "stereotactic_radiosurgery": "For unresectable HNPGL (jugulotympanic, vagal); good local control",
        "131I-MIBG": "For metaiodobenzylguanidine-avid functional PGL/PCC",
        "peptide_receptor_radionuclide_therapy": "PRRT (177Lu-DOTATATE) for somatostatin receptor-positive PGL",
        "targeted_therapy": "HIF2α inhibitors (belzutifan) — emerging; succinate-driven PHD inhibition is the mechanism",
        "alpha_blockade": "Phenoxybenzamine/doxazosin pre-operatively for functional PCC (catecholamine excess)",
        "no_chemotherapy_first_line": "Chemotherapy (CVD/TMZ) reserved for malignant/metastatic PGL only",
    }

    # CII assembly pathway
    cii_assembly_pathway = [
        {"step": "1", "factor": "SDHAF2 (SDH5, 11q13.1) ← THIS GENE", "role": "SDHA His99 FAD covalent attachment (autocatalytic flavinylation)", "disease": "PGL2 Paraganglioma (AD, maternal imprinting)", "highlight": True},
        {"step": "2", "factor": "SDHAF1 (LYRM8, 19q13.12)", "role": "[2Fe-2S] + [4Fe-4S] FeS cluster delivery to SDHB", "disease": "CII Deficiency Infantile Leukoencephalopathy (AR)", "highlight": False},
        {"step": "3", "factor": "SDHAF3 (LYRM2, 1q21.2)", "role": "Protects FeS-loaded SDHB from oxidative damage", "disease": "CII Deficiency (AR) — rare", "highlight": False},
        {"step": "4", "factor": "SDHAF4 (1p36.33)", "role": "Stabilises SDHB FeS subunit during assembly; scaffold", "disease": "CII Deficiency (AR) — rare", "highlight": False},
        {"step": "5", "factor": "SDHA + SDHB (catalytic core)", "role": "SDHA-SDHB heterodimer assembly; SDHC-SDHD membrane anchor addition", "disease": "SDHA: Leigh + PGL5 (AD); SDHB/C/D: PGL4/3/1 (AD)", "highlight": False},
    ]

    # DDx table
    ddx_table = [
        {
            "gene":        "SDHAF2 (PGL2)",
            "locus":       "11q13.1",
            "inheritance": "AD + maternal imprinting",
            "disease":     "Paraganglioma (HNPGL + PCC)",
            "malignancy":  "~5%",
            "imprinting":  "YES (maternal)",
            "distinguishing": "Only paternal transmission; SDHA flavinylation factor; 11q13.1",
        },
        {
            "gene":        "SDHD (PGL1)",
            "locus":       "11q23.1",
            "inheritance": "AD + maternal imprinting",
            "disease":     "Head-neck PGL predominantly",
            "malignancy":  "~5%",
            "imprinting":  "YES (maternal)",
            "distinguishing": "Same chr11, maternal imprinting, but 10 Mb apart; SDHD is structural subunit (D-anchor)",
        },
        {
            "gene":        "SDHB (PGL4)",
            "locus":       "1p36.13",
            "inheritance": "AD — NO imprinting",
            "disease":     "PGL + PCC; high extra-adrenal risk",
            "malignancy":  "20–50%",
            "imprinting":  "NO",
            "distinguishing": "Highest malignancy risk; no imprinting; SDHB = FeS structural subunit",
        },
        {
            "gene":        "SDHC (PGL3)",
            "locus":       "1q23.3",
            "inheritance": "AD — NO imprinting",
            "disease":     "Head-neck PGL predominant",
            "malignancy":  "<5%",
            "imprinting":  "NO",
            "distinguishing": "No imprinting; SDHC = IMM anchor subunit; low malignancy",
        },
        {
            "gene":        "SDHA (PGL5)",
            "locus":       "1p36.1",
            "inheritance": "AD (PGL5); AR (Leigh syndrome)",
            "disease":     "PGL5 (dom) + Leigh syndrome (rec)",
            "malignancy":  "<5% (PGL)",
            "imprinting":  "NO",
            "distinguishing": "Dual disease: dominant PGL5 + recessive Leigh; SDHA = catalytic FAD subunit",
        },
        {
            "gene":        "SDHAF1 (leukoencephalopathy)",
            "locus":       "19q13.12",
            "inheritance": "AR (recessive)",
            "disease":     "CII deficiency + infantile leukoencephalopathy",
            "malignancy":  "N/A (not a tumor gene)",
            "imprinting":  "NO",
            "distinguishing": "Completely different: AR recessive; white matter disease; NOT paraganglioma",
        },
        {
            "gene":        "VHL",
            "locus":       "3p25.3",
            "inheritance": "AD",
            "disease":     "VHL: hemangioblastoma + ccRCC + PCC",
            "malignancy":  "High (ccRCC)",
            "imprinting":  "NO",
            "distinguishing": "VHL directly inhibits HIF; hemangioblastoma DDx; RCC; no paraganglioma pattern",
        },
    ]

    return {
        "gene": GENE,
        "n_patients": N_PATIENTS,
        "seed": SEED,
        "variant_breakdown": variants_summary,
        "tumour_type_breakdown": tumour_breakdown,
        "clinical_feature_breakdown": clinical_breakdown,
        "imprinting_analysis": imprinting_summary,
        "cii_assembly_pathway": cii_assembly_pathway,
        "surveillance_protocol": surveillance_protocol,
        "treatment_summary": treatment_summary,
        "ddx_table": ddx_table,
        "severity_logic": {
            "severe":       "Null/frameshift/nonsense/canonical-splice OR missense at SDHA-contact core → complete loss of SDHAF2 → no SDHA flavinylation → high penetrance PGL (80–92%)",
            "intermediate": "Missense at SDHA-contact surface / FAD-binding region → partial SDHAF2 function retained → moderate penetrance (65–85%)",
            "moderate":     "Presequence/import region / hypomorphic core → partial import or mild folding defect → lower penetrance (60–65%) / later onset",
        },
    }


# ─── get_definitions ───────────────────────────────────────────────────────────
def get_definitions() -> dict:
    return {
        "gene_definition": (
            "SDHAF2 (Succinate Dehydrogenase Assembly Factor 2; also SDH5; OMIM *613019; 11q13.1) encodes "
            "a 166-amino-acid, ~19 kDa mitochondrial matrix protein that functions as the SDHA flavinylation "
            "factor. SDHAF2 is required for covalent attachment of the FAD cofactor to SDHA histidine-99 "
            "(His99) via an autocatalytic mechanism. Without FAD, SDHA remains an apo-protein incapable of "
            "catalyzing succinate oxidation, CII assembly is blocked, and the pseudo-hypoxic HIF1α pathway "
            "is activated via succinate-mediated PHD inhibition, driving paraganglioma tumorigenesis. "
            "SDHAF2 has no intrinsic enzymatic activity — it functions as a chaperone scaffold to position "
            "SDHA His99 for autocatalytic flavinylation. Chromosome 11q13.1. "
            "Inheritance: AD with MATERNAL IMPRINTING (only paternal transmission causes disease)."
        ),
        "disease_definition": (
            "Paragangliomas 2 (PGL2; OMIM #601650) is an autosomal dominant hereditary "
            "paraganglioma-pheochromocytoma syndrome caused by heterozygous loss-of-function mutations "
            "in SDHAF2 inherited paternally (maternal imprinting). Clinical features include: "
            "head-neck paragangliomas (carotid body tumor 55%, jugulotympanic 35%, vagal 22%), "
            "pheochromocytoma (15%), retroperitoneal PGL (18%), bilateral/multicentric tumors (30%), "
            "and SDH-deficient GIST (8%). Malignancy risk is low (~5%), unlike SDHB (20–50%). "
            "Catecholamine excess symptoms occur in PCC subset (~20%). "
            "Disease penetrance is age-dependent, typically presenting in the 3rd–5th decade. "
            "The maternal imprinting of SDHAF2 means female carriers are unaffected and their "
            "children are not at risk; only paternal mutation transmission causes disease. "
            "Management: surgical resection of resectable tumors; stereotactic radiosurgery for "
            "unresectable HNPGL; annual surveillance imaging (MRI/MRA + CT) and biochemistry "
            "(plasma metanephrines); pre-operative alpha-blockade for functional PCC."
        ),
        "inheritance_definition": (
            "SDHAF2 follows AUTOSOMAL DOMINANT inheritance with MATERNAL GENOMIC IMPRINTING. "
            "The maternal SDHAF2 allele is epigenetically silenced (methylation of the maternal "
            "allele at the imprint control region of 11q13.1). Consequently: "
            "(1) Only the paternal SDHAF2 allele is transcriptionally active. "
            "(2) A mutation on the PATERNAL allele → single active allele is mutant → haploinsufficiency → PGL2. "
            "(3) A mutation on the MATERNAL allele → silenced anyway → no phenotype (maternal carrier unaffected). "
            "(4) Children of a FEMALE SDHAF2 carrier: they inherit her mutant allele as a MATERNAL allele — "
            "it is silenced → no risk. "
            "(5) Children of a MALE SDHAF2 carrier: they inherit his mutant allele as a PATERNAL allele — "
            "it is active → 50% risk. "
            "This is analogous to SDHD (PGL1, 11q23.1) maternal imprinting and the H19/IGF2 imprinting "
            "on chromosome 11. Critical for correct genetic counselling."
        ),
        "mechanism_definition": (
            "CII (succinate dehydrogenase, SDH) assembles via a stepwise pathway: "
            "Step 1 — SDHAF2 (THIS GENE): SDHAF2 binds SDHA in the mitochondrial matrix and acts as "
            "a chaperone scaffold to position SDHA His99 for AUTOCATALYTIC covalent FAD attachment "
            "(flavinylation). FAD is the essential cofactor for SDHA-mediated succinate → fumarate "
            "oxidation. Step 2 — SDHAF1: delivers [2Fe-2S] and [4Fe-4S] FeS clusters to SDHB. "
            "Step 3 — SDHAF3: protects SDHB from oxidative FeS damage. "
            "Step 4 — SDHAF4: scaffolds SDHB during final assembly. "
            "Step 5 — SDHA-SDHB catalytic core assembles; SDHC-SDHD membrane anchor added; "
            "CII holoenzyme inserts into IMM. "
            "When SDHAF2 is lost: SDHA is apo-protein (no FAD) → cannot catalyze succinate oxidation "
            "→ succinate accumulates → SUCCINATE INHIBITS PHD ENZYMES (prolyl hydroxylases) "
            "→ HIF1α cannot be hydroxylated and degraded by VHL → HIF1α STABILIZES (PSEUDO-HYPOXIA) "
            "→ pro-angiogenic/tumorigenic transcription → PARAGANGLIOMA."
        ),
        "imprinting_definition": (
            "GENOMIC IMPRINTING is an epigenetic phenomenon where one parental allele is silenced "
            "based on its parental origin. SDHAF2 (11q13.1) shows MATERNAL IMPRINTING: the allele "
            "inherited from the mother is methylated and transcriptionally silent. Only the "
            "PATERNAL allele is expressed. This has profound clinical consequences: "
            "(A) Penetrance depends on which parent transmitted the mutation. "
            "(B) Female mutation carriers do not develop PGL2 (their own active allele is their "
            "paternal allele — if that is normal, they are protected). "
            "(C) Female carriers can transmit the mutant allele to children, but as a maternal allele "
            "it will be silenced in those children → unaffected. "
            "(D) Male carriers transmit an active paternal allele → disease risk in 50% of children. "
            "SDHAF2 imprinting is analogous to SDHD (PGL1, 11q23.1) — both on chromosome 11, "
            "both maternally imprinted. This is in contrast to SDHB (PGL4) and SDHC (PGL3) "
            "which show NO imprinting and follow conventional AD penetrance regardless of "
            "parent-of-origin."
        ),
        "surveillance_definition": (
            "PGL2 (SDHAF2) surveillance: Annual MRI/MRA (skull base to aortic arch — HNPGL) and "
            "annual contrast CT or MRI (adrenal/retroperitoneum — PCC/retroperitoneal PGL). "
            "Annual plasma/urine metanephrines and chromogranin A. "
            "Upper GI endoscopy or abdominal MRI if SDH-deficient GIST suspected. "
            "Surveillance begins at age 15 years (or 5 years before youngest affected relative). "
            "GENETIC COUNSELLING: confirm paternal origin of SDHAF2 mutation. "
            "At-risk individuals: only children of MALE SDHAF2 carriers. "
            "Female SDHAF2 carriers: unaffected; do not require intensive surveillance; "
            "consider psychological support given complexity of imprinting counselling."
        ),
        "treatment_definition": (
            "SDHAF2-PGL2 treatment: "
            "(1) SURGERY: first-line for resectable paraganglioma (carotid body, adrenal PCC, "
            "retroperitoneal PGL). Pre-operative alpha-blockade (phenoxybenzamine/doxazosin) "
            "mandatory for functional PCC before surgery. "
            "(2) STEREOTACTIC RADIOSURGERY (Gamma Knife / CyberKnife): preferred for unresectable "
            "head-neck PGL (jugulotympanic, vagal). Excellent local control. "
            "(3) 131I-MIBG THERAPY: for MIBG-avid functional PGL/PCC in metastatic setting. "
            "(4) PRRT (177Lu-DOTATATE): for somatostatin receptor-positive PGL in metastatic setting. "
            "(5) HIF2α INHIBITORS (belzutifan/PT2977): emerging targeted therapy; "
            "succinate-driven HIF2α stabilization is the pathomechanism; "
            "FDA-approved for VHL disease; trials ongoing for SDH-deficient tumors. "
            "(6) CHEMOTHERAPY (CVD/temozolomide): reserved for malignant/metastatic paraganglioma; "
            "NOT first-line for resectable disease. "
            "(7) CATECHOLAMINE MANAGEMENT: phenoxybenzamine pre-operatively for functional PCC; "
            "avoid beta-blockade before alpha-blockade (risk of hypertensive crisis). "
            "CONTRAINDICATIONS IN PCC-ASSOCIATED SURGERY: "
            "Avoid dopamine agonists, tricyclic antidepressants, MAOIs, cocaine, glucagon, "
            "IV contrast without pre-treatment — all can trigger catecholamine surge."
        ),
        "key_distinctions": {
            "SDHAF2_vs_SDHAF1": (
                "SDHAF2 (11q13.1, dominant, paraganglioma, maternal imprinting) vs "
                "SDHAF1 (19q13.12, recessive, CII deficiency leukoencephalopathy). "
                "Same CII assembly pathway — SDHAF2 step 1 (SDHA FAD), SDHAF1 step 2 (SDHB FeS). "
                "Completely different diseases, opposite inheritance, opposite chromosomes."
            ),
            "SDHAF2_vs_SDHD": (
                "Both PGL with maternal imprinting on chromosome 11. "
                "SDHAF2 11q13.1 (SDH5, flavinylation factor); SDHD 11q23.1 (CII membrane anchor). "
                "WES locus mandatory to distinguish."
            ),
            "SDHAF2_vs_SDHB": (
                "Both hereditary paraganglioma. SDHB: no imprinting, highest malignancy (~20–50%), "
                "frequent extra-adrenal/retroperitoneal. SDHAF2: maternal imprinting, low malignancy (~5%), "
                "head-neck predominant."
            ),
            "SDHAF2_maternal_imprinting": (
                "ONLY PATERNAL SDHAF2 mutations cause disease. "
                "Female carriers: unaffected; their children not at risk. "
                "Male carriers: 50% of children at risk. "
                "This is NOT conventional AD — the parent-of-origin is everything."
            ),
        },
        "drug_contraindications": [
            {
                "drug":       "Alpha-blocker omission before beta-blockade",
                "level":      "CRITICAL DANGER — PCC pre-op",
                "mechanism":  (
                    "In functional PCC: beta-blockade without prior alpha-blockade removes vasodilatory "
                    "beta-2 effect while catecholamines remain unopposed at alpha receptors → "
                    "severe hypertensive crisis, potentially fatal. "
                    "ALWAYS establish alpha-blockade (phenoxybenzamine) first, then add beta-blocker "
                    "if needed for tachycardia."
                ),
                "alternative": "Phenoxybenzamine (10–20 mg TID) for 10–14 days pre-op, then add propranolol if tachycardia persists.",
            },
            {
                "drug":       "MAOIs / TCAs / cocaine / sympathomimetics",
                "level":      "ABSOLUTE CONTRAINDICATION (functional PCC)",
                "mechanism":  (
                    "Any agent that blocks catecholamine reuptake or increases catecholamine release "
                    "can trigger a hypertensive crisis in functional PCC. Includes monoamine oxidase "
                    "inhibitors (MAOIs), tricyclic antidepressants (TCAs), cocaine, glucagon, "
                    "high-dose dopamine, and certain IV contrast agents."
                ),
                "alternative": "Avoid all sympathomimetics. Use propofol-remifentanil TIVA with invasive BP monitoring for anaesthesia if PCC active.",
            },
            {
                "drug":       "Belzutifan (HIF2α inhibitor) — not yet standard",
                "level":      "EMERGING THERAPY — not yet standard of care for PGL2",
                "mechanism":  (
                    "Belzutifan (FDA approved for VHL-related tumors) targets HIF2α directly. "
                    "Since SDHAF2 LOF activates HIF2α via succinate-mediated PHD inhibition, "
                    "HIF2α inhibition is mechanistically rational. Clinical trials ongoing. "
                    "Not yet approved specifically for SDH-deficient paraganglioma."
                ),
                "alternative": "Clinical trial enrollment preferred. Standard first-line: surgery or stereotactic radiosurgery.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SDHAF2 OVERVIEW ===")
    ov = get_overview()
    print(f"Gene: {ov['gene']}, Disease: {ov['disease_name'][:60]}...")
    print(f"Patients: {ov['n_patients']}, Seed: {ov['seed']}")
    print(f"Cohort: {ov['cohort_summary']}")
    print("\n=== BREAKDOWN (variant count) ===")
    bd = get_breakdown()
    print(f"Variants: {len(bd['variant_breakdown'])}, DDx entries: {len(bd['ddx_table'])}")
    print("\n=== DEFINITIONS (keys) ===")
    df = get_definitions()
    print(list(df.keys()))
    print("\n✅ SDHAF2 dashboard OK")
