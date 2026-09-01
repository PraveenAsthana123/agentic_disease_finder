#!/usr/bin/env python3
"""RRM2B Encephalomyopathic mtDNA Depletion Syndrome Dashboard.

Mitochondrial DNA Depletion Syndrome 8A (MDDS8A) = OMIM #612075
Also: RRM2B Deficiency / p53R2-Related Encephalomyopathy

RRM2B (ribonucleoside-diphosphate reductase subunit M2 B; also called p53R2;
351 aa; 8q22.3) encodes the p53-inducible small (R2) subunit of ribonucleotide
reductase (RNR), the sole source of dNTPs for mtDNA replication in post-mitotic
(non-dividing) cells.

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — mtDNA depletion disease; shared mechanism with
     POLG/DGUOK/MPV17/TK2/TWNK/SUCLA2
  2. KD = CONTRAINDICATED — OXPHOS-dependent beta-oxidation fails in mtDNA depletion
  3. Renal tubular acidosis (Fanconi syndrome) ~50-55% — DISTINCTIVE DDx from all other
     MDDS (TK2/SUCLA2/POLG do NOT have Fanconi); treat with NaHCO3/citrate + K+ replacement
  4. NO hepatopathy — KEY DDx from DGUOK/MPV17/TWNK/POLG (hepatocerebral); liver is
     NORMAL in RRM2B MDDS8A
  5. NO methylmalonic aciduria — KEY DDx from SUCLA2 (SUCLA2 has mild MMA 100%;
     RRM2B does NOT)
  6. Respiratory failure ~65% — leading cause of death; NIV critical; sleep study mandatory
  7. CK mildly elevated (100-800 U/L, <10x ULN) — KEY DDx from TK2 (CK very high >2000
     U/L, 90%); if CK > 2000, reconsider TK2
  8. SNHL ~35% — less prevalent than SUCLA2 (75%); but screen at diagnosis
  9. Leigh-like MRI ~40% — lower than SUCLA2 (80%); bilateral basal ganglia + brainstem
 10. Hypotonia 100% — universal, profound, neonatal/infantile onset
 11. Propofol = AVOID (PRIS risk in mitochondrial disease)
 12. LEV preferred AED — renal excretion, safe without hepatic or mitochondrial liability
 13. PEO5 (AD form): adult-onset ptosis + progressive external ophthalmoplegia + multiple
     mtDNA DELETIONS (not depletion); heterozygous dominant negative; NO early depletion

RRM2B BIOLOGY:
RRM2B (351 amino acids, 8q22.3) encodes the p53-inducible small (R2) subunit of
ribonucleotide reductase (RNR), a heterotetrameric enzyme (R1/R2 heterotetramer) that
reduces ribonucleoside diphosphates (NDPs) to deoxyribonucleoside diphosphates (dNDPs).
The dNTPs produced by RRM2B are transported into mitochondria via the SLC25A33/SLC25A36
nucleoside transporters for use by POLG.

RRM2B is unique in that it is:
  (a) p53-inducible: upregulated by DNA damage to support DNA repair dNTP synthesis
  (b) constitutively expressed in non-dividing/post-mitotic cells (neurons, muscle fibres)

This contrasts with RRM2, which is S-phase specific and absent in G0/G1 cells.
In post-mitotic tissue (where RRM2 is cell-cycle-silenced), RRM2B (p53R2) is therefore
the SOLE source of dNTPs for mitochondrial DNA replication by POLG. Without RRM2B,
post-mitotic cells cannot maintain the balanced dNTP pool needed for POLG fidelity,
leading to replication errors and depletion.

Domain architecture:
  N-terminal Fe-S scaffold / folate-binding region (aa 1-~100)
  Radical-generating tyrosine Tyr270 (human RRM2B; equivalent to Tyr122 in E. coli R2):
    essential for di-iron-centre radical chemistry
  Dinuclear iron cluster (aa ~150-250): essential for radical generation; two Fe ions
    chelated by Asp-Asp-His-Glu ligands; stabilises the tyrosyl radical
  C-terminal allosteric specificity site (aa ~270-351): substrate/effector binding

GENOTYPE-PHENOTYPE CORRELATION:
  Biallelic null (nonsense/frameshift) → severe; neonatal hypotonia; rapid mtDNA
    depletion; Fanconi 60%; median survival <3 years
  Missense compound het → moderate-severe; Fanconi 45%; may survive to childhood
    with supportive care
  Homozygous missense → variable; depends on residue criticality (Fe-S scaffold vs
    peripheral); range from severe to moderate
  Splice site/missense → variable; partial splice rescue may moderate phenotype
  Iberian founder (missense, MAF ~0.003 in gnomAD non-Finnish European; compound het) →
    moderate; enriched in Iberian/Mediterranean populations; Bornstein 2008 Hum Mutat
  Heterozygous dominant negative (R1-interface missense) → adult-onset PEO5; multiple
    mtDNA DELETIONS (not depletion); NO early depletion

KEY DDx PEARLS:
  vs TK2 (MDDS4A): TK2 → CK very high >2000 U/L in 90%; myopathic > encephalopathic;
    NO Fanconi; if CK >2000, reconsider TK2
  vs POLG: POLG → hepatopathy 80%; EPC (epilepsia partialis continua) 60%; NO Fanconi;
    European founder (Ala467Thr/Trp748Ser)
  vs DGUOK: DGUOK → hepatocerebral; nystagmus 90%; NO Fanconi
  vs SUCLA2 (MDDS10): SUCLA2 → mild MMA 100%; SNHL 75%; NO Fanconi; Faroe founder
  vs TWNK (MDDS7): TWNK → hepatocerebral 75%; IOSCA spinocerebellar ataxia; NO Fanconi
  Fanconi syndrome + hypotonia + lactic acidosis + NO hepatopathy + NO MMA →
    RRM2B until proven otherwise

References:
  Bourdon A et al. 2007 Nat Genet — first description of RRM2B mutations causing MDDS
  Spinazzola A et al. 2009 Hum Mol Genet — broader RRM2B phenotypic spectrum
  Bornstein B et al. 2008 Hum Mutat — additional families, Iberian founder
  Tyynismaa H et al. 2012 — adult-onset RRM2B spectrum (PEO5)
"""

import random
from datetime import date

SEED = 559  # 40-patient cohort seed


def get_overview() -> dict:
    """RRM2B MDDS8A — top-level overview for /api/rrm2b/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "RRM2B Encephalomyopathic mtDNA Depletion Syndrome (MDDS8A)",
        "gene": "RRM2B",
        "protein": "Ribonucleoside-Diphosphate Reductase Subunit M2 B (p53R2)",
        "protein_size_aa": 351,
        "locus": "8q22.3",
        "inheritance": (
            "Autosomal Recessive (AR) — biallelic for MDDS8A (encephalomyopathic depletion); "
            "Autosomal Dominant (AD) — heterozygous dominant negative for PEO5 "
            "(progressive external ophthalmoplegia 5; multiple mtDNA deletions, NOT depletion)"
        ),
        "omim_gene": "604712",
        "omim_disease": "612075",
        "mechanism": (
            "Biallelic RRM2B loss-of-function → loss of p53R2 (sole dNTP source in G0/post-mitotic cells) "
            "→ dNTP pool depletion in post-mitotic tissues (muscle, neuron) "
            "→ POLG replication stalling → mtDNA copy number depletion in muscle and brain (<30% normal). "
            "In dividing cells, RRM2 (S-phase) compensates; in post-mitotic neurons and myocytes, "
            "RRM2B (p53R2) is the ONLY RNR R2 subunit present, making it irreplaceable for mtDNA maintenance."
        ),
        "key_contraindications": [
            {
                "drug": "Valproic Acid (VPA) — ALL INDICATIONS",
                "level": "ABSOLUTE CONTRAINDICATION",
                "reason": (
                    "mtDNA depletion disease — VPA inhibits POLG (mtDNA polymerase gamma), "
                    "sequesters CoA as valproyl-CoA, and produces a hepatotoxic epoxide metabolite. "
                    "Shared mechanism across all MDDS (POLG, DGUOK, MPV17, TK2, TWNK, SUCLA2, RRM2B). "
                    "Fulminant hepatic failure and death reported. DOCUMENT AS ALLERGY-EQUIVALENT."
                ),
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "CONTRAINDICATED",
                "reason": (
                    "Forces OXPHOS-dependent beta-oxidation. RRM2B MDDS8A depletes ETC complexes "
                    "I/III/IV/V (all mtDNA-encoded) → OXPHOS failure → KD precipitates metabolic crisis."
                ),
            },
            {
                "drug": "Propofol",
                "level": "AVOID",
                "reason": "PRIS (propofol infusion syndrome) in mitochondrial disease — Complex II + FAO inhibition.",
            },
            {
                "drug": "Fasting",
                "level": "FORBIDDEN at all ages",
                "reason": (
                    "Provide emergency IV dextrose (GIR 6-8 mg/kg/min) during any nil-by-mouth period. "
                    "Issue emergency protocol letter. Fasting triggers metabolic decompensation and "
                    "lactic crisis in mtDNA depletion."
                ),
            },
        ],
        "pathognomonic_ddx": {
            "fanconi_syndrome": (
                "Renal tubular acidosis / Fanconi syndrome in ~52% — DISTINCTIVE vs ALL other MDDS. "
                "Features: glucosuria + aminoaciduria + phosphaturia + bicarbonuria (proximal tubule "
                "dysfunction). TK2, SUCLA2, POLG do NOT cause Fanconi. "
                "Fanconi + hypotonia + lactic acidosis + NO hepatopathy + NO MMA → RRM2B until proven otherwise."
            ),
            "no_hepatopathy": (
                "Liver is NORMAL in RRM2B MDDS8A — critical DDx from DGUOK, MPV17, TWNK, POLG "
                "(all hepatocerebral). Absence of liver involvement + Fanconi + hypotonia → suspect RRM2B."
            ),
            "no_methylmalonic_aciduria": (
                "NO methylmalonic aciduria — critical DDx from SUCLA2 (MMA 100%, mild) and MUT. "
                "Urine organic acids show NO elevated MMA in RRM2B MDDS8A."
            ),
            "ck_mildly_elevated": (
                "CK mildly elevated 100-800 U/L (<10x ULN) in ~60% — KEY DDx from TK2 where CK "
                ">2000 U/L in 90%. If CK >2000, reconsider TK2. RRM2B CK elevation is modest."
            ),
        },
        "cohort_summary": {
            "n": 40,
            "seed": SEED,
            "median_onset_months": 3,
            "median_diagnosis_months": 11,
            "top_presenting_features": [
                "Hypotonia (100%)",
                "Lactic acidosis (90%)",
                "Failure to thrive (85%)",
                "Proximal weakness (80%)",
                "Developmental delay/regression (75%)",
                "Respiratory failure (65%)",
                "Renal tubular acidosis / Fanconi syndrome (52%)",
                "CK mildly elevated (60%)",
            ],
        },
        "first_description": (
            "Bourdon A et al., Nat Genet 2007 — first description of RRM2B (p53R2) mutations causing MDDS; "
            "Spinazzola A et al., Hum Mol Genet 2009 — broader phenotypic spectrum"
        ),
        "iberian_founder": (
            "Iberian founder missense variant (MAF ~0.003 in gnomAD non-Finnish European); "
            "enriched in Iberian/Mediterranean origin families; Bornstein B et al., Hum Mutat 2008"
        ),
    }


def get_breakdown() -> dict:
    """RRM2B MDDS8A — breakdown for /api/rrm2b/breakdown."""
    rng = random.Random(SEED)

    genotypes = [
        {
            "genotype": "Nonsense/frameshift compound het",
            "fraction": 0.25,
            "example": "Biallelic truncating — e.g. nonsense/frameshift compound het (descriptive; specific alleles vary by family)",
            "phenotype": (
                "Severe early-onset; neonatal hypotonia; rapid mtDNA depletion (<15% muscle); "
                "Fanconi ~60%; median survival <3 years; respiratory failure leading cause of death"
            ),
        },
        {
            "genotype": "Missense/missense compound het",
            "fraction": 0.35,
            "example": "Compound heterozygous missense — one or both alleles affecting Fe-S scaffold or iron-cluster domain",
            "phenotype": (
                "Moderate-severe; Fanconi ~45%; may survive to childhood with aggressive supportive care; "
                "variable degree of encephalopathy depending on residue criticality"
            ),
        },
        {
            "genotype": "Homozygous missense",
            "fraction": 0.15,
            "example": "Homozygous missense — phenotype depends on domain; Fe-S scaffold → severe; peripheral residue → moderate",
            "phenotype": (
                "Variable; Fe-S scaffold or iron-cluster missense → severe; "
                "allosteric specificity-site missense → moderate; Fanconi ~40%"
            ),
        },
        {
            "genotype": "Splice site/missense compound het",
            "fraction": 0.12,
            "example": "Splice site / missense compound het — partial splice rescue may moderate phenotype",
            "phenotype": (
                "Variable; partial splice rescue may reduce severity; Fanconi ~35%; "
                "delayed-onset respiratory compromise possible"
            ),
        },
        {
            "genotype": "Iberian founder compound het",
            "fraction": 0.08,
            "example": (
                "Iberian founder missense (MAF ~0.003 gnomAD non-Finnish European) / "
                "second pathogenic allele; Iberian/Mediterranean origin"
            ),
            "phenotype": (
                "Moderate; Iberian/Mediterranean origin; lactic acidosis + Fanconi ~50%; "
                "SNHL present; Bornstein 2008 Hum Mutat families"
            ),
        },
        {
            "genotype": "Other / atypical",
            "fraction": 0.05,
            "example": "Atypical biallelic combinations including deep intronic, large deletions, or complex rearrangements",
            "phenotype": "Variable; includes atypical phenotypes; confirm with mtDNA copy number quantification in muscle",
        },
    ]

    phenotypes_dist = [
        ("Encephalomyopathic + Fanconi (severe)", 14),
        ("Encephalomyopathic without Fanconi (moderate-severe)", 16),
        ("Encephalomyopathic + SNHL (prominent)", 10),
    ]

    sex_choices = ["M", "F"]
    ethnic_groups = [
        "Iberian/Spanish",
        "Portuguese",
        "Italian",
        "North African/Moroccan",
        "Ashkenazi Jewish",
        "Sephardic Jewish",
        "Turkish",
        "Palestinian",
        "German",
        "British",
        "Pakistani",
        "Indian",
        "Chinese",
        "Japanese",
        "Brazilian",
    ]
    variant_descriptors = [
        ("Biallelic truncating (nonsense/frameshift)", "Biallelic null"),
        ("Missense/missense compound het (Fe-S scaffold)", "Missense comphet — Fe-S domain"),
        ("Missense/missense compound het (iron cluster)", "Missense comphet — iron cluster"),
        ("Homozygous missense (allosteric site)", "Homozygous missense — allosteric"),
        ("Iberian founder / second allele", "Iberian founder compound het"),
        ("Splice site / missense compound het", "Splice/missense comphet"),
        ("Homozygous missense (Fe-S scaffold)", "Homozygous missense — Fe-S, severe"),
        ("Nonsense / missense compound het", "Nonsense/missense comphet"),
    ]

    patients = []
    pid = 1
    for phenotype_name, count in phenotypes_dist:
        for _ in range(count):
            sex = rng.choice(sex_choices)
            ethnicity = rng.choice(ethnic_groups)
            variant = rng.choice(variant_descriptors)
            onset_mo = rng.randint(0, 6)
            diag_mo = onset_mo + rng.randint(2, 18)
            hypotonia = True  # 100%
            lactic_acidosis = rng.random() < 0.90
            failure_to_thrive = rng.random() < 0.85
            respiratory_failure = rng.random() < 0.65
            fanconi = rng.random() < 0.52
            ck_elevated = rng.random() < 0.60
            leigh_mri = rng.random() < 0.40
            snhl = rng.random() < 0.35
            proximal_weakness = rng.random() < 0.80
            dev_delay = rng.random() < 0.75
            lactate = round(rng.uniform(3.0, 9.5), 1) if lactic_acidosis else round(rng.uniform(1.1, 2.3), 1)
            ck_value = rng.randint(120, 750) if ck_elevated else rng.randint(40, 120)
            mtdna_pct = rng.randint(5, 28)  # <30% in affected tissues
            urine_glucose_fanconi = rng.choice([True, False]) if fanconi else False
            patients.append({
                "pid": pid,
                "sex": sex,
                "ethnicity": ethnicity,
                "phenotype": phenotype_name,
                "variant_class": variant[1],
                "variant_description": variant[0],
                "onset_months": onset_mo,
                "diagnosis_months": diag_mo,
                "hypotonia": hypotonia,
                "lactic_acidosis": lactic_acidosis,
                "peak_lactate_mmol": lactate,
                "failure_to_thrive": failure_to_thrive,
                "respiratory_failure": respiratory_failure,
                "fanconi_syndrome": fanconi,
                "glucosuria_on_fanconi": urine_glucose_fanconi,
                "ck_elevated": ck_elevated,
                "ck_iu_L": ck_value,
                "leigh_mri": leigh_mri,
                "snhl": snhl,
                "proximal_weakness": proximal_weakness,
                "developmental_delay": dev_delay,
                "mtdna_copy_pct_muscle": mtdna_pct,
                "hepatopathy": False,  # NEVER present in RRM2B MDDS8A
                "methylmalonic_aciduria": False,  # NEVER present in RRM2B
            })
            pid += 1

    total = len(patients)

    feature_prevalence = [
        {
            "feature": "Hypotonia",
            "pct": 100,
            "note": "Universal neonatal/infantile hypotonia; may be profound (ventilator-dependent in severe cases)",
        },
        {
            "feature": "Lactic Acidosis",
            "pct": 90,
            "note": (
                "Blood lactate >2.5 mmol/L fasting and post-prandial; pH <7.2 in crisis. "
                "Reflects OXPHOS impairment from mtDNA depletion. "
                "Monitor with bicarbonate and point-of-care lactate."
            ),
        },
        {
            "feature": "Failure to Thrive",
            "pct": 85,
            "note": "Feeding difficulties from hypotonia + dysphagia; NG tube often required from early infancy",
        },
        {
            "feature": "Proximal Weakness",
            "pct": 80,
            "note": "Hip and shoulder girdle weakness; progresses to loss of ambulation; EMG shows myopathic pattern",
        },
        {
            "feature": "Developmental Delay / Psychomotor Regression",
            "pct": 75,
            "note": (
                "Psychomotor regression correlates with degree of mtDNA depletion in brain; "
                "cognitive impairment variable; loss of acquired milestones during metabolic crises"
            ),
        },
        {
            "feature": "Respiratory Failure",
            "pct": 65,
            "note": (
                "Leading cause of death in RRM2B MDDS8A. "
                "Diaphragmatic and intercostal involvement; nocturnal hypoventilation precedes daytime. "
                "NIV mandatory when respiratory compromise develops; sleep study at diagnosis and every 6 months."
            ),
        },
        {
            "feature": "CK Mildly Elevated (<10x ULN, 100-800 U/L)",
            "pct": 60,
            "note": (
                "MILD elevation — KEY DDx from TK2 (CK >2000 U/L in 90%, very high). "
                "If CK >2000 U/L, strongly reconsider TK2 MDDS4A. "
                "RRM2B CK range: 100-800 U/L; reflects modest myopathic component."
            ),
        },
        {
            "feature": "Renal Tubular Acidosis / Fanconi Syndrome",
            "pct": 52,
            "note": (
                "PROXIMAL TUBULE DYSFUNCTION — DISTINCTIVE vs ALL other MDDS. "
                "Four hallmark features: glucosuria + aminoaciduria + phosphaturia + bicarbonuria. "
                "Mechanism: mitochondrial ATP depletion in proximal tubule cells → tubular transporter failure. "
                "TK2, SUCLA2, POLG, DGUOK, MPV17, TWNK do NOT cause Fanconi. "
                "Treat with NaHCO3/citrate 1-3 mEq/kg/day + K+ replacement + phosphate supplementation."
            ),
        },
        {
            "feature": "Leigh-like MRI (bilateral BG/brainstem T2 hyperintensity)",
            "pct": 40,
            "note": (
                "Bilateral T2/FLAIR hyperintensity in basal ganglia and/or brainstem. "
                "Less prevalent than SUCLA2 (80%) and other Leigh-predominant MDDS. "
                "Correlates with severe biallelic null genotype and deep mtDNA depletion in brain."
            ),
        },
        {
            "feature": "Sensorineural Hearing Loss (SNHL)",
            "pct": 35,
            "note": (
                "SNHL present but less prevalent than SUCLA2 (75%). "
                "ABR/OAE screening at diagnosis and every 6 months. "
                "Cochlear implants may be considered if SNHL ≥ 40 dB bilateral."
            ),
        },
        {
            "feature": "NO Hepatopathy",
            "pct": 0,
            "note": (
                "ABSENT — liver is NORMAL in RRM2B MDDS8A. "
                "Critical DDx from DGUOK, MPV17, TWNK, POLG (all hepatocerebral). "
                "Normal LFTs are a required feature for MDDS8A; elevated transaminases should prompt "
                "reconsideration of hepatocerebral MDDS or concomitant pathology."
            ),
        },
        {
            "feature": "NO Methylmalonic Aciduria",
            "pct": 0,
            "note": (
                "ABSENT — KEY DDx from SUCLA2 (MMA 100%, mild, 10-100 µmol/mmol creat). "
                "Urine organic acids show no elevated MMA in RRM2B MDDS8A. "
                "MMA present in urine → strongly consider SUCLA2/SUCLG1 instead."
            ),
        },
        {
            "feature": "NO Nystagmus",
            "pct": 0,
            "note": "ABSENT — nystagmus is DGUOK-pathognomonic (90%); not a feature of RRM2B",
        },
        {
            "feature": "NO 3-MGA-uria",
            "pct": 0,
            "note": "ABSENT — critical DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB (all have 3-MGA-uria)",
        },
    ]

    treatments = [
        {
            "tx": "Valproic Acid (VPA) — ALL INDICATIONS",
            "level": "ABSOLUTE CONTRAINDICATION",
            "note": (
                "NEVER prescribe in RRM2B MDDS8A or ANY mtDNA depletion syndrome. "
                "Three independent mechanisms of mitotoxicity: "
                "(1) POLG inhibition by VPA and 4-en-VPA metabolite; "
                "(2) CoA sequestration as valproyl-CoA → impairs beta-oxidation and TCA cycle; "
                "(3) hepatotoxic reactive epoxide (2-propyl-4-pentenoic acid epoxide). "
                "Combined: fulminant hepatic failure and death. Document VPA allergy-equivalent in ALL medical records."
            ),
        },
        {
            "tx": "Ketogenic Diet (KD)",
            "level": "CONTRAINDICATED",
            "note": (
                "Forces OXPHOS-dependent beta-oxidation. "
                "RRM2B depletes mtDNA-encoded ETC complexes I/III/IV/V. "
                "KD → fatty acid load → OXPHOS bottleneck → lactic acidosis exacerbated → metabolic crisis."
            ),
        },
        {
            "tx": "Propofol",
            "level": "AVOID — PRIS Risk",
            "note": (
                "Propofol Infusion Syndrome: Complex II inhibition + fatty-acid oxidation inhibition "
                "in mitochondrial disease → lipid accumulation, cardiac arrhythmia, metabolic acidosis. "
                "Anaesthesia: use ketamine or volatile agents (sevoflurane). "
                "Document AVOID in anaesthetic record and allergy list."
            ),
        },
        {
            "tx": "Fasting Prevention (strict)",
            "level": "A — Mandatory at All Ages",
            "note": (
                "Avoid fasting >3 h in neonates/infants, >5 h in older children. "
                "Nil-by-mouth periods (pre-op, intercurrent illness): IV dextrose (GIR 6-8 mg/kg/min). "
                "Continuous glucose monitoring during decompensation. "
                "Emergency protocol letter carried by family at all times."
            ),
        },
        {
            "tx": "Levetiracetam (LEV)",
            "level": "A — First-line AED",
            "note": (
                "Renal excretion only; no hepatic metabolism; no mitochondrial toxicity. "
                "IV loading 20-40 mg/kg for acute seizures. "
                "Maintenance 20-60 mg/kg/day divided q12h. "
                "Avoid phenobarbital/phenytoin/carbamazepine (hepatic induction, Complex I risk)."
            ),
        },
        {
            "tx": "Non-Invasive Ventilation (NIV)",
            "level": "A — Mandatory when respiratory compromise develops",
            "note": (
                "Respiratory failure is the leading cause of death in RRM2B MDDS8A (~65%). "
                "Initiate nocturnal NIV at first sign of diaphragmatic compromise (nocturnal SpO2 desaturations). "
                "Sleep study mandatory at diagnosis and every 6 months. "
                "Escalate to daytime NIV as needed. Tracheostomy if NIV fails."
            ),
        },
        {
            "tx": "Sodium Bicarbonate / Citrate (for Fanconi / RTA)",
            "level": "A — For Renal Tubular Acidosis (52%)",
            "note": (
                "For Fanconi syndrome / proximal RTA: NaHCO3 1-3 mEq/kg/day oral "
                "or Shohl's solution (sodium citrate/citric acid); titrate to serum bicarbonate 22-26 mEq/L. "
                "Monitor urine pH and serum electrolytes twice weekly in acute phase. "
                "Bicarbonate loss in Fanconi is ongoing — do not discontinue."
            ),
        },
        {
            "tx": "Potassium Replacement (for Fanconi-induced hypokalemia)",
            "level": "A — For Fanconi Syndrome",
            "note": (
                "Fanconi-induced urinary potassium wasting → hypokalemia → cardiac arrhythmia risk. "
                "KCl supplements titrated to serum K+ 3.8-4.5 mEq/L. "
                "Monitor serum K+ twice weekly in acute phase; weekly when stable."
            ),
        },
        {
            "tx": "Phosphate Supplementation (for Fanconi-induced hypophosphatemia)",
            "level": "B — For Fanconi Syndrome",
            "note": (
                "Fanconi-induced phosphaturia → hypophosphatemia → rickets risk (Fanconi rickets). "
                "K-Phos or Na-Phos supplements; dose guided by serum phosphate and urine phosphate/creatinine ratio. "
                "Monitor renal phosphate wasting; adjust dose quarterly."
            ),
        },
        {
            "tx": "L-Carnitine",
            "level": "B — Supportive",
            "note": (
                "50-100 mg/kg/day oral; replenishes carnitine lost in Fanconi-induced urinary wasting. "
                "Reduces acylcarnitine burden; supportive for beta-oxidation in residual OXPHOS function. "
                "Monitor free/total carnitine; target free carnitine >20 µmol/L."
            ),
        },
        {
            "tx": "Riboflavin (Vitamin B2)",
            "level": "C — Supportive",
            "note": (
                "100-400 mg/day oral; supportive for residual OXPHOS flavoprotein activity "
                "(Complex I and II FAD cofactor). Low risk; often included in mitochondrial supportive cocktail."
            ),
        },
        {
            "tx": "Vitamin C (Ascorbate)",
            "level": "C — Supportive",
            "note": (
                "500-1000 mg/day; radical quenching; may support residual RNR dinuclear iron-centre "
                "radical activity in RRM2B partial loss-of-function. "
                "Low risk; part of supportive antioxidant regimen."
            ),
        },
        {
            "tx": "dCyd/dThd Deoxynucleoside Supplementation",
            "level": "D / Investigational — NOT approved",
            "note": (
                "Pyrimidine deoxyribonucleoside (dCyd/dThd) bypass of RNR; analogous to TK2 "
                "deoxynucleoside therapy (which rescues TK2 MDDS in preclinical + early clinical data). "
                "Rationale: exogenous deoxyribonucleosides enter mitochondria via SLC transporters, "
                "bypassing the RRM2B RNR step to restore mtDNA copy number. "
                "Phase 2 investigation in progress; NOT approved; discuss with metabolic specialist."
            ),
        },
        {
            "tx": "Phenobarbital / Phenytoin / Carbamazepine",
            "level": "AVOID",
            "note": (
                "Hepatic enzyme induction; sodium channel blockers may exacerbate mitochondrial "
                "membrane dysfunction; phenytoin inhibits Complex I. Prefer LEV for seizures in RRM2B."
            ),
        },
        {
            "tx": "Genetic Counselling",
            "level": "A — Mandatory",
            "note": (
                "AR inheritance (MDDS8A): 25% recurrence risk per pregnancy. "
                "Prenatal diagnosis via CVS (11-13 weeks) or amniocentesis for known familial variants. "
                "Iberian/Mediterranean families: assess carrier status of extended family members "
                "given Iberian founder allele frequency (~MAF 0.003). "
                "Preimplantation genetic testing (PGT-M) available."
            ),
        },
    ]

    disease_timeline = [
        {
            "phase": "Neonatal / Early Infantile (0-3 months)",
            "events": (
                "Profound hypotonia often noted at delivery or in first days of life. "
                "Feeding difficulties → NG tube; failure to thrive. "
                "Blood lactate elevated (>3 mmol/L); urine organic acids: NO MMA (distinguishes from SUCLA2). "
                "Renal function: check urine glucose, amino acids, phosphate, bicarbonate for Fanconi. "
                "Normal liver function (distinguishes from hepatocerebral MDDS). "
                "First metabolic crisis may occur with first intercurrent illness."
            ),
        },
        {
            "phase": "Infantile (3-12 months)",
            "events": (
                "Progressive hypotonia; failure to achieve motor milestones. "
                "Brain MRI: Leigh-like changes in basal ganglia may appear (40% at some point). "
                "Fanconi syndrome confirmed on urine biochemistry in ~52%: "
                "glucosuria + aminoaciduria + phosphaturia + bicarbonuria. "
                "SNHL confirmed on ABR in ~35%; CK mildly elevated in ~60%. "
                "Respiratory assessment: sleep study for nocturnal hypoventilation. "
                "Genetic testing: RRM2B biallelic variants confirmed."
            ),
        },
        {
            "phase": "Early Toddler (1-2 years)",
            "events": (
                "Developmental plateau or regression; psychomotor delay. "
                "Fanconi management ongoing: NaHCO3 + K+ replacement + phosphate. "
                "Respiratory failure emerges in ~65% — NIV initiation; sleep study surveillance. "
                "Feeding support: NG or PEG placement for severe dysphagia. "
                "Lactic acidosis decompensation during febrile illness — emergency protocol letters critical."
            ),
        },
        {
            "phase": "Childhood (2-8 years)",
            "events": (
                "Phenotype-dependent: severe (null) → progressive respiratory decline, loss of ambulation; "
                "moderate (missense comphet, Iberian founder) → stable disability with support. "
                "Continued Fanconi management; risk of rickets from phosphaturia. "
                "Physiotherapy, occupational therapy; communication aids if nonverbal. "
                "SNHL management: hearing aids or cochlear implant evaluation."
            ),
        },
        {
            "phase": "Terminal / Late Phase",
            "events": (
                "Respiratory failure leading cause of death; NIV or tracheostomy. "
                "Palliative care discussions in progressive severe disease. "
                "PEO5 (AD RRM2B) patients present separately in adulthood with ptosis + ophthalmoplegia + "
                "multiple mtDNA DELETIONS (not depletion); this is a distinct phenotype from MDDS8A."
            ),
        },
    ]

    # MMA differential table — RRM2B vs SUCLA2 vs TK2
    mma_ddx_table = {
        "rrm2b_mma": "ABSENT — NO methylmalonic aciduria in RRM2B MDDS8A",
        "sucla2_mma_umol_mmol_creat_typical": "10-100 (mild MMA — metabolic bystander of TCA blockade)",
        "tk2_mma": "ABSENT — NO methylmalonic aciduria in TK2 MDDS4A",
        "rrm2b_ck_iu_L": "100-800 U/L (<10x ULN, mild) — 60% of patients",
        "sucla2_ck": "Normal to mildly elevated (<5x ULN) — encephalomyopathic, not myopathic",
        "tk2_ck_iu_L": ">2000 U/L (very high, >10-20x ULN) — myopathic marker, 90% of TK2",
        "rrm2b_fanconi": "YES — ~52%; DISTINCTIVE; proximal RTA unique to RRM2B among MDDS",
        "sucla2_fanconi": "NO — Fanconi absent in SUCLA2 MDDS10",
        "tk2_fanconi": "NO — Fanconi absent in TK2 MDDS4A",
        "rrm2b_hepatopathy": "NO — liver normal (non-hepatocerebral MDDS)",
        "sucla2_hepatopathy": "NO — liver normal (non-hepatocerebral MDDS)",
        "tk2_hepatopathy": "NO — liver normal (myopathic MDDS)",
        "note": (
            "Key clinical triad distinguishing RRM2B from SUCLA2: "
            "(1) NO MMA in RRM2B vs MMA 100% in SUCLA2; "
            "(2) Fanconi 52% in RRM2B vs 0% in SUCLA2; "
            "(3) SNHL 35% in RRM2B vs 75% in SUCLA2. "
            "Key triad distinguishing RRM2B from TK2: "
            "(1) CK mild (<800) in RRM2B vs CK very high (>2000) in TK2; "
            "(2) Fanconi 52% in RRM2B vs 0% in TK2; "
            "(3) encephalomyopathic in RRM2B vs predominantly myopathic in TK2."
        ),
    }

    # Fanconi syndrome feature details
    fanconi_features = [
        {
            "feature": "Glucosuria",
            "mechanism": (
                "Failure of SGLT2-mediated glucose reabsorption in proximal tubule S1/S2 segments "
                "due to mitochondrial ATP depletion → urine glucose positive despite normoglycemia"
            ),
            "test": "Urine dipstick glucose + serum glucose simultaneously; glucosuria with normal blood glucose confirms tubular origin",
            "significance": "Hallmark feature 1 of 4 Fanconi criteria in RRM2B",
        },
        {
            "feature": "Aminoaciduria",
            "mechanism": (
                "Failure of amino acid reabsorption transporters (SLC6A/SLC1A/SLC3A families) "
                "in proximal tubule due to ATP depletion → generalised aminoaciduria"
            ),
            "test": "Urine amino acids (quantitative): generalised hyperaminoaciduria across multiple amino acid classes",
            "significance": "Hallmark feature 2 of 4 Fanconi criteria; distinguishes from primary aminoacidurias",
        },
        {
            "feature": "Phosphaturia",
            "mechanism": (
                "Failure of NaPi-IIa/NaPi-IIc (SLC34A1/SLC34A3) phosphate transporters "
                "due to ATP depletion → renal phosphate wasting → hypophosphatemia → rickets risk"
            ),
            "test": "Serum phosphate (low) + urine phosphate/creatinine ratio (elevated); tubular maximum for phosphate (TmP/GFR) reduced",
            "significance": "Hallmark feature 3 of 4; if untreated → hypophosphatemic rickets + growth failure",
        },
        {
            "feature": "Bicarbonuria / Proximal RTA",
            "mechanism": (
                "Failure of proximal tubule bicarbonate reabsorption (NHE3/carbonic anhydrase II) "
                "due to ATP depletion → urinary bicarbonate wasting → metabolic acidosis with high urinary pH"
            ),
            "test": "Serum bicarbonate (low) + urine pH (inappropriately high >6.0 during acidosis) + FEHCO3 elevated",
            "significance": "Hallmark feature 4 of 4; distinguishes from distal RTA (type 1) where urine pH cannot be lowered",
        },
    ]

    return {
        "generated": date.today().isoformat(),
        "cohort_n": 40,
        "seed": SEED,
        "phenotype_distribution": [
            {"name": name, "n": count, "pct": round(count / 40 * 100)}
            for name, count in phenotypes_dist
        ],
        "genotype_breakdown": genotypes,
        "feature_prevalence": feature_prevalence,
        "treatments": treatments,
        "disease_timeline": disease_timeline,
        "mma_ddx_table": mma_ddx_table,
        "fanconi_features": fanconi_features,
        "patients_sample": patients[:8],
    }


def get_definitions() -> dict:
    """RRM2B MDDS8A — definitions for /api/rrm2b/definitions."""
    return {
        "generated": date.today().isoformat(),
        "terms": [
            {
                "term": "RRM2B / p53R2 — Ribonucleoside-Diphosphate Reductase Subunit M2 B",
                "definition": (
                    "RRM2B (351 amino acids, 8q22.3; also called p53R2) encodes the small (R2) subunit "
                    "of ribonucleotide reductase (RNR). Unlike RRM2 (the canonical S-phase R2 subunit), "
                    "RRM2B is constitutively expressed in non-dividing, post-mitotic cells and is "
                    "transcriptionally induced by p53 in response to DNA damage. "
                    "RRM2B forms a heterotetrameric RNR complex (R1₂/RRM2B₂) with the large R1 subunit "
                    "(encoded by RRM1). This complex reduces ribonucleoside diphosphates (NDPs) to "
                    "deoxyribonucleoside diphosphates (dNDPs), which are then phosphorylated to dNTPs "
                    "for use by POLG in mtDNA replication. "
                    "Domain architecture: N-terminal Fe-S scaffold/folate-binding region (aa 1-~100); "
                    "dinuclear iron cluster (aa ~150-250) stabilising the essential tyrosyl radical "
                    "(Tyr270); C-terminal allosteric specificity site (aa ~270-351). "
                    "In post-mitotic neurons and myocytes where RRM2 (S-phase) is absent, RRM2B is "
                    "the SOLE source of dNTPs for mtDNA maintenance."
                ),
            },
            {
                "term": "MDDS8A — Mitochondrial DNA Depletion Syndrome 8A (Encephalomyopathic Type)",
                "definition": (
                    "MDDS8A (OMIM #612075) is an autosomal recessive disease caused by biallelic "
                    "loss-of-function mutations in RRM2B. Clinical features: neonatal/infantile hypotonia "
                    "(100%), lactic acidosis (90%), failure to thrive (85%), proximal weakness (80%), "
                    "developmental delay/regression (75%), respiratory failure (65%), Fanconi syndrome "
                    "(52%), Leigh-like MRI (40%), SNHL (35%). "
                    "Crucially: NO hepatopathy (liver NORMAL — distinguishes from DGUOK/MPV17/TWNK/POLG) "
                    "and NO methylmalonic aciduria (distinguishes from SUCLA2 where MMA is 100%). "
                    "Renal Fanconi syndrome is the single most DISTINCTIVE feature of RRM2B among all MDDS. "
                    "mtDNA copy number in muscle: <30% normal (confirmed by Southern blot or qPCR)."
                ),
            },
            {
                "term": "Ribonucleotide Reductase (RNR) — Enzyme Mechanism and mtDNA Relevance",
                "definition": (
                    "Ribonucleotide reductase (RNR) is the rate-limiting enzyme for de novo dNTP synthesis. "
                    "Reaction: NDP + (reduced thioredoxin) → dNDP + (oxidised thioredoxin). "
                    "The radical mechanism requires: "
                    "(1) a stable tyrosyl radical (Tyr270 in RRM2B) generated by the dinuclear iron centre; "
                    "(2) radical transfer via a 35-Å proton-coupled electron transfer (PCET) pathway "
                    "from the R2 tyrosyl radical through a conserved pathway to the R1 active-site cysteine, "
                    "which directly reduces the substrate ribose 2'-OH. "
                    "This dNDP is phosphorylated to dNTP by nucleoside diphosphate kinases. "
                    "For mtDNA, the critical dNTPs (dATP, dGTP, dCTP, dTTP) produced by RRM2B are "
                    "transported into mitochondria via SLC25A33/SLC25A36 transporters for POLG. "
                    "Without balanced dNTP pools, POLG makes replication errors, stalls, or cannot "
                    "initiate replication → progressive mtDNA copy number depletion."
                ),
            },
            {
                "term": "Fanconi Syndrome / Renal Tubular Acidosis — Mechanism and DDx Value",
                "definition": (
                    "Fanconi syndrome is a generalised proximal tubule dysfunction characterised by "
                    "the four-hallmark tetrad: glucosuria + aminoaciduria + phosphaturia + bicarbonuria. "
                    "Mechanism in RRM2B MDDS8A: mitochondrial ATP depletion in proximal tubule cells "
                    "(which are highly mitochondria-dependent due to OXPHOS-driven secondary active "
                    "transport) → failure of basolateral Na+/K+-ATPase and apical cotransporters → "
                    "loss of reabsorption of glucose, amino acids, phosphate, and bicarbonate. "
                    "Clinical consequences: metabolic acidosis (bicarbonate loss), hypokalemia "
                    "(potassium wasting), hypophosphatemia (risk of Fanconi rickets), "
                    "volume depletion. "
                    "DDx CRITICAL: Fanconi syndrome in an infant with hypotonia + lactic acidosis + "
                    "NO hepatopathy + NO MMA → RRM2B MDDS8A until proven otherwise. "
                    "NO other MDDS (TK2, SUCLA2, POLG, DGUOK, MPV17, TWNK) causes Fanconi. "
                    "Treatment: NaHCO3/citrate + K+ replacement + phosphate supplementation."
                ),
            },
            {
                "term": "Post-Mitotic dNTP Supply — Why G0/G1 Cells Need RRM2B Specifically",
                "definition": (
                    "In proliferating (dividing) cells, RRM2 is expressed in S phase under E2F "
                    "transcriptional control and provides ample dNTPs for both nuclear and mitochondrial "
                    "DNA replication. In post-mitotic cells (terminally differentiated neurons, adult "
                    "skeletal and cardiac myocytes), RRM2 is cell-cycle silenced (absent in G0/G1). "
                    "RRM2B (p53R2) is constitutively expressed regardless of cell-cycle phase, "
                    "making it the SOLE source of dNTPs for mitochondrial DNA maintenance in these "
                    "cells. Mitochondria undergo continuous mtDNA replication (turnover) even in "
                    "post-mitotic cells to maintain the copy number required for OXPHOS function. "
                    "Without RRM2B, this maintenance replication fails → mtDNA depletion exclusively "
                    "or predominantly in post-mitotic tissues (brain, muscle), explaining the "
                    "encephalomyopathic phenotype of MDDS8A."
                ),
            },
            {
                "term": "PEO5 (AD RRM2B) — Multiple mtDNA Deletions vs Depletion",
                "definition": (
                    "PEO5 (Progressive External Ophthalmoplegia 5; AD-PEO caused by heterozygous "
                    "dominant-negative RRM2B variants) is a DISTINCT phenotype from MDDS8A. "
                    "Key distinction: "
                    "MDDS8A = biallelic (AR) → mtDNA DEPLETION (<30% copy number) in muscle/brain; "
                    "PEO5 = heterozygous (AD) dominant negative → multiple mtDNA DELETIONS "
                    "(normal copy number, multiple large-scale deletions on Southern blot). "
                    "PEO5 clinical features: adult-onset ptosis + progressive external ophthalmoplegia "
                    "(± proximal limb weakness); NO early encephalomyopathy; NO Fanconi syndrome; "
                    "Tyynismaa H et al. 2012 described the adult-onset spectrum. "
                    "Mechanism of dominant negative: R1-interface missense variant disrupts RRM2B-RRM1 "
                    "heterodimerisation, reducing but not abolishing dNTP supply → subtle imbalance "
                    "sufficient to cause replication slippage/deletion accumulation over decades, "
                    "not severe enough to cause early depletion."
                ),
            },
            {
                "term": "VPA Absolute Contraindication — mtDNA Depletion Mechanism",
                "definition": (
                    "Valproic acid is absolutely contraindicated in ALL mtDNA depletion syndromes "
                    "(RRM2B MDDS8A, SUCLA2 MDDS10, SUCLG1 MDDS9, POLG Alpers, DGUOK MDDS3, "
                    "MPV17 MDDS6, TK2 MDDS4A, TWNK MDDS7) via three independent mechanisms: "
                    "(1) VPA and its Δ4-metabolite (4-en-VPA) directly inhibit POLG (mtDNA polymerase "
                    "gamma), reducing mtDNA replication in cells already depleted by RRM2B loss — "
                    "additive catastrophic depletion; "
                    "(2) valproyl-CoA sequesters free CoA (forming valproyl-CoA), impairing "
                    "beta-oxidation and TCA cycle in energetically compromised mitochondria; "
                    "(3) reactive epoxide metabolite (2-propyl-4-pentenoic acid epoxide) is directly "
                    "hepatotoxic — causing fatal hepatic necrosis, especially in patients with pre-existing "
                    "mitochondrial compromise. Combined: fulminant hepatic failure and death reported "
                    "in paediatric mtDNA disease. No safe dose. Document VPA as allergy-equivalent."
                ),
            },
            {
                "term": "RRM2B vs TK2 — CK Differential (Critical Clinical DDx Pearl)",
                "definition": (
                    "CK is the single most powerful laboratory discriminator between RRM2B (MDDS8A) "
                    "and TK2 (MDDS4A) at bedside: "
                    "RRM2B MDDS8A: CK typically 100-800 U/L (<10x ULN) in ~60% of patients. "
                    "The modest CK reflects mild myopathic involvement secondary to encephalomyopathic depletion. "
                    "TK2 MDDS4A: CK typically >2000 U/L (often 2000-10,000 U/L, >10-20x ULN) in ~90% — "
                    "the hallmark of primary myopathic MDDS. TK2 directly depletes dCTP/dTTP in muscle "
                    "→ profound myopathy → massive CK leak. "
                    "Rule: If CK >2000 U/L in an infant with hypotonia + mtDNA depletion, "
                    "send TK2 gene test first. "
                    "If CK <800 U/L + Fanconi + NO MMA, send RRM2B. "
                    "If CK <800 U/L + mild MMA + SNHL + NO Fanconi, send SUCLA2."
                ),
            },
            {
                "term": "mtDNA Depletion — Diagnostic Threshold and Tissue Specificity",
                "definition": (
                    "mtDNA depletion is defined as a severe reduction in mtDNA copy number relative "
                    "to nuclear DNA in affected tissues. Diagnostic threshold: <30% of age-matched "
                    "control mean (commonly <20% in severely affected individuals). "
                    "Measured by: Southern blot (gold standard, can detect both depletion and deletions); "
                    "quantitative PCR (qPCR) of mtDNA/nDNA ratio (faster, widely available). "
                    "Tissue specificity in RRM2B: most severe in muscle and brain (post-mitotic, "
                    "dependent on RRM2B for dNTP supply). Liver and fibroblasts may show near-normal "
                    "copy number (dividing cells have RRM2 compensation) — a negative fibroblast "
                    "result does NOT exclude MDDS8A; always test muscle biopsy. "
                    "Degree of depletion correlates with phenotype severity: <15% → severe/fatal; "
                    "15-30% → moderate; >30% (low-level depletion) → mild or PEO5-like."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json

    print("=== RRM2B MDDS8A Dashboard — Overview ===")
    ov = get_overview()
    print(json.dumps(ov, indent=2))

    print("\n=== Breakdown (feature prevalence) ===")
    bk = get_breakdown()
    for f in bk["feature_prevalence"]:
        print(f"  {f['feature']}: {f['pct']}%")

    print("\n=== Fanconi Features ===")
    for ff in bk["fanconi_features"]:
        print(f"  {ff['feature']}: {ff['mechanism'][:60]}...")

    print("\n=== Definitions count ===")
    df = get_definitions()
    print(f"  {len(df['terms'])} definitions")
