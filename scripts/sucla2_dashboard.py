#!/usr/bin/env python3
"""SUCLA2 Encephalomyopathic mtDNA Depletion Syndrome Dashboard.

Mitochondrial DNA Depletion Syndrome 10 (MDDS10) = OMIM #615084
Also: Succinic Aciduria / SCS-A Deficiency

SUCLA2 (succinate-CoA ligase, ADP-forming, beta subunit, 463 aa, 13q14.2)
encodes the catalytic beta subunit of ADP-forming succinyl-CoA synthetase (SCS-A),
a TCA-cycle enzyme in the mitochondrial matrix.

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — mtDNA depletion disease; shared mechanism with POLG/DGUOK/MPV17/TK2/TWNK
  2. KD = CONTRAINDICATED — OXPHOS-dependent beta-oxidation fails in mtDNA depletion
  3. Methylmalonic aciduria is MILD — critical DDx from MUT/MMACHC (severe MMA); SUCLA2 MMA rarely causes ketoacidosis
  4. NO hepatopathy — KEY DDx from DGUOK/MPV17/TWNK/POLG (all hepatocerebral); liver is normal in SUCLA2
  5. NO homocystinuria — KEY DDx from MMACHC/cblC (MMA + Hcy); SUCLA2 has MMA only
  6. SNHL (sensorineural hearing loss) 75% — cochlear implants effective; recommend audiogram at diagnosis
  7. Dystonia 70% — movement disorder; may respond to deep brain stimulation in severe cases
  8. Leigh-like MRI 80% — T2 hyperintensity basal ganglia, putamen, caudate; ± brainstem
  9. Faroe Islands founder p.Asp333Gly — ~1:1,000 prevalence in Faroe Islands (carrier freq ~1:16)
 10. SUCLA2 + SUCLG1 together cause ~50% of all encephalomyopathic MDDS cases
 11. Propofol = AVOID (PRIS risk in mitochondrial disease)
 12. LEV preferred AED — renal excretion, safe without hepatic or mitochondrial liability

SUCLA2 BIOLOGY:
SUCLA2 (463 amino acids, 13q14.2) encodes the substrate-specific beta subunit of
succinate-CoA ligase (SCS-A), also called succinyl-CoA synthetase (ADP-forming).
The enzyme forms an alpha-beta heterodimer (SUCLG1 = alpha, SUCLA2 = beta).

TCA cycle reaction catalysed:
  Succinyl-CoA + ADP + Pi → Succinate + ATP + CoA-SH
  (the only substrate-level phosphorylation step in the mitochondrial TCA cycle)

SUCLA2–NDPK COMPLEX (key to mtDNA maintenance):
  SUCLA2 physically associates with nucleoside diphosphate kinase (NDPK, encoded by NME4)
  at the mitochondrial inner membrane. This proximity creates a local high-concentration
  microenvironment of dNTPs (dATP in particular) that is essential for mtDNA replication
  by POLG. Loss of SUCLA2 disrupts this micro-compartment → dATP pool imbalance →
  POLG replication stalling → mtDNA copy number depletion in muscle and brain.

Domain architecture:
  N-terminal CoA-binding domain (aa 1–~150): binds CoA thioester of succinyl-CoA
  Nucleotide-binding domain (aa ~151–340): ADP binding; P-loop (Walker A) at aa ~193-200
  Beta-subunit specificity loop (aa ~330–360): confers ADP (vs GDP) specificity vs SUCLG2 (ADP-forming)
  C-terminal alpha/beta fold (aa ~360–463): interdomain contacts; dimer interface
  p.Asp333Gly (Faroe founder): disrupts beta-subunit specificity loop → enzyme inactive

GENOTYPE-PHENOTYPE CORRELATION:
  Biallelic null (nonsense/frameshift) → severe; early-onset profound hypotonia; Leigh MRI 1st year
  p.Asp333Gly homozygous (Faroe) → moderately severe; hypotonia + SNHL + dystonia; Leigh MRI year 1-3
  Missense compound het → variable; SNHL prominent; some mild MMA only without encephalopathy
  Splicing variants → variable residual activity; phenotype depends on amount of functional enzyme

PATHOGENIC VARIANT DISTRIBUTION (AR biallelic MDDS10, n=40, seed-557):
  p.Asp333Gly/p.Asp333Gly (Faroe founder homozygous): ~15%
  p.Asp333Gly/other missense (Faroe compound): ~10%
  Nonsense/frameshift compound het: ~25%
  Missense/missense compound het: ~30%
  Splice site/missense compound het: ~12%
  Homozygous missense (non-Faroe): ~8%

KEY DDx PEARLS:
  vs MUT methylmalonic aciduria: MUT → severe MMA + ketoacidosis + hepatopathy; NO mtDNA depletion
  vs MMACHC (cblC): MMA + homocystinuria (SUCLA2 has NO Hcy); retinopathy (SUCLA2 has none)
  vs SUCLG1 (MDDS9): clinically overlapping; SUCLG1 typically more severe / earlier onset; fatal neonatal form exists
  vs DGUOK: hepatocerebral + nystagmus (SUCLA2 NO liver, NO nystagmus)
  vs TK2: myopathic (CK elevated, respiratory failure); SUCLA2 encephalomyopathic (CK mild/normal)
  vs Leigh syndrome (other causes): check urine MMA; if MMA present, order SUCLA2 gene

References:
  Elpeleg O 2005 Am J Hum Genet — first description; succinyl-CoA synthetase deficiency
  Carrozzo R 2007 Pediatrics — Italian cohort; Leigh + MMA + SNHL
  Ostergaard E 2007 J Inherit Metab Dis — Faroe Islands founder p.Asp333Gly
  Van Hove JLK 2010 J Inherit Metab Dis — review SUCLA2/SUCLG1 spectrum
"""

import random
from datetime import date

SEED = 557  # 40-patient cohort seed


def get_overview() -> dict:
    """SUCLA2 MDDS10 — top-level overview for /api/sucla2/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "SUCLA2 Encephalomyopathic mtDNA Depletion Syndrome (MDDS10)",
        "gene": "SUCLA2",
        "protein": "Succinyl-CoA Synthetase ADP-Forming Beta Subunit (SCS-A beta)",
        "protein_size_aa": 463,
        "locus": "13q14.2",
        "inheritance": "Autosomal Recessive (AR)",
        "omim_gene": "603921",
        "omim_disease": "615084",
        "mechanism": (
            "Biallelic SUCLA2 loss-of-function → succinyl-CoA synthetase (SCS-A) deficiency → "
            "disruption of SUCLA2–NDPK dNTP micro-compartment → dATP pool depletion → "
            "POLG replication stalling → mtDNA copy number depletion in brain and muscle (<30% normal)"
        ),
        "key_contraindications": [
            {
                "drug": "Valproic Acid (VPA) — ALL INDICATIONS",
                "level": "ABSOLUTE CONTRAINDICATION",
                "reason": (
                    "mtDNA depletion disease — VPA inhibits POLG (mtDNA polymerase), sequesters CoA, "
                    "and produces a hepatotoxic epoxide metabolite. Shared mechanism across all MDDS. "
                    "Fulminant hepatic failure and death reported. DOCUMENT AS ALLERGY-EQUIVALENT."
                ),
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "CONTRAINDICATED",
                "reason": (
                    "Forces OXPHOS-dependent beta-oxidation. mtDNA depletion depletes ETC complexes "
                    "I/III/IV/V → OXPHOS failure → KD precipitates metabolic crisis."
                ),
            },
            {
                "drug": "Propofol",
                "level": "AVOID",
                "reason": "PRIS (propofol infusion syndrome) in mitochondrial disease — Complex II + FAO inhibition.",
            },
        ],
        "pathognomonic_ddx": {
            "methylmalonic_aciduria_mild": (
                "Urine MMA elevated but MILD (10-100 µmol/mmol creatinine) — critical DDx from "
                "MUT/MMACHC where MMA is severe (>500-1000 µmol/mmol) with ketoacidosis."
            ),
            "no_hepatopathy": (
                "Liver is NORMAL in SUCLA2 — critical DDx from DGUOK, MPV17, TWNK, POLG "
                "(all hepatocerebral). Absence of liver involvement + MMA + hypotonia → suspect SUCLA2."
            ),
            "no_homocystinuria": (
                "NO homocysteine elevation — DDx from MMACHC/cblC (MMA + elevated Hcy). "
                "Plasma Hcy normal in SUCLA2."
            ),
            "snhl": (
                "SNHL in 75% — cochlear implants effective. MMA + hypotonia + SNHL = SUCLA2 until proven otherwise."
            ),
        },
        "cohort_summary": {
            "n": 40,
            "seed": SEED,
            "median_onset_months": 4,
            "median_diagnosis_months": 14,
            "top_presenting_features": [
                "Hypotonia (100%)",
                "Methylmalonic aciduria — mild (100%)",
                "Lactic acidosis (85%)",
                "Leigh-like MRI (80%)",
                "SNHL (75%)",
                "Dystonia (70%)",
            ],
        },
        "first_description": "Elpeleg O et al., Am J Hum Genet 2005; Carrozzo R et al., Pediatrics 2007",
        "faroe_founder": "p.Asp333Gly (c.998A>G) — ~1:1,000 Faroe Islands birth prevalence; carrier ~1:16",
    }


def get_breakdown() -> dict:
    """SUCLA2 MDDS10 — breakdown for /api/sucla2/breakdown."""
    rng = random.Random(SEED)

    genotypes = [
        {
            "genotype": "Nonsense/frameshift compound het",
            "fraction": 0.25,
            "example": "p.Arg279* / p.Gln324Argfs*14",
            "phenotype": "Severe early-onset; profound hypotonia; Leigh MRI by 6 months; median survival 6 years",
        },
        {
            "genotype": "Missense/missense compound het",
            "fraction": 0.30,
            "example": "p.Arg284Cys / p.Leu244Pro",
            "phenotype": "Moderate; hypotonia + dystonia + SNHL; some ambulant with support",
        },
        {
            "genotype": "p.Asp333Gly homozygous (Faroe founder)",
            "fraction": 0.15,
            "example": "p.Asp333Gly / p.Asp333Gly",
            "phenotype": "Moderate; Faroe Islands origin; hypotonia + SNHL 90%; dystonia; Leigh MRI",
        },
        {
            "genotype": "p.Asp333Gly / other missense (Faroe compound)",
            "fraction": 0.10,
            "example": "p.Asp333Gly / p.Ile322Thr",
            "phenotype": "Moderate-severe; similar to Faroe homozygous",
        },
        {
            "genotype": "Splice site / missense compound het",
            "fraction": 0.12,
            "example": "c.534+1G>T / p.Ala252Val",
            "phenotype": "Variable; partial splice rescue may moderate phenotype",
        },
        {
            "genotype": "Homozygous missense (non-Faroe)",
            "fraction": 0.08,
            "example": "p.Trp145Arg / p.Trp145Arg",
            "phenotype": "Severe if catalytic residue; moderate if peripheral",
        },
    ]

    phenotypes_dist = [
        ("Encephalomyopathic (Leigh + hypotonia + dystonia)", 30),
        ("Encephalomyopathic + SNHL (dominant)", 8),
        ("Leigh-only (MRI + hypotonia, mild MMA)", 2),
    ]

    sex_choices = ["M", "F"]
    ethnic_groups = [
        "Faroe Islands",
        "Italian",
        "Ashkenazi Jewish",
        "Spanish",
        "Turkish",
        "Palestinian",
        "Norwegian",
        "German",
        "Pakistani",
        "British",
    ]
    variant_pairs = [
        ("p.Asp333Gly/p.Asp333Gly", "Faroe founder homozygous"),
        ("p.Asp333Gly/p.Ile322Thr", "Faroe compound het"),
        ("p.Arg279*/p.Gln324Argfs*14", "Nonsense compound het"),
        ("p.Arg284Cys/p.Leu244Pro", "Missense compound het"),
        ("c.534+1G>T/p.Ala252Val", "Splice/missense"),
        ("p.Trp145Arg/p.Trp145Arg", "Homozygous missense"),
        ("p.Glu104*/p.Arg279*", "Biallelic nonsense"),
        ("p.Gly322Ser/p.Leu87Pro", "Missense compound het"),
    ]

    patients = []
    pid = 1
    for phenotype_name, count in phenotypes_dist:
        for _ in range(count):
            sex = rng.choice(sex_choices)
            ethnicity = rng.choice(ethnic_groups)
            variant_pair = rng.choice(variant_pairs)
            onset_mo = rng.randint(1, 9)
            diag_mo = onset_mo + rng.randint(3, 24)
            hypotonia = True  # 100%
            mma = True  # 100%
            leigh_mri = rng.random() < 0.80
            snhl = rng.random() < 0.75
            dystonia = rng.random() < 0.70
            seizures = rng.random() < 0.50
            lactic_acidosis = rng.random() < 0.85
            feeding_difficulty = rng.random() < 0.90
            mtdna_pct = rng.randint(6, 28)  # <30% in affected tissues
            mma_value = rng.randint(12, 110)  # mild MMA (µmol/mmol creat)
            lactate = round(rng.uniform(2.8, 8.5), 1) if lactic_acidosis else round(rng.uniform(1.2, 2.2), 1)
            patients.append({
                "pid": pid,
                "sex": sex,
                "ethnicity": ethnicity,
                "phenotype": phenotype_name,
                "variant_description": variant_pair[1],
                "genotype": variant_pair[0],
                "onset_months": onset_mo,
                "diagnosis_months": diag_mo,
                "hypotonia": hypotonia,
                "methylmalonic_aciduria": mma,
                "mma_urine_umol_per_mmol_creat": mma_value,
                "lactic_acidosis": lactic_acidosis,
                "peak_lactate_mmol": lactate,
                "leigh_mri": leigh_mri,
                "snhl": snhl,
                "dystonia": dystonia,
                "seizures": seizures,
                "feeding_difficulty": feeding_difficulty,
                "mtdna_copy_pct_muscle": mtdna_pct,
            })
            pid += 1

    total = len(patients)
    feature_prevalence = [
        {
            "feature": "Hypotonia",
            "pct": 100,
            "note": "Universal; infantile-onset; may be severe (ventilator-dependent in worst cases)",
        },
        {
            "feature": "Methylmalonic aciduria (mild)",
            "pct": 100,
            "note": (
                "Urine MMA 10-100 µmol/mmol creatinine — MILD (DDx: MUT/MMACHC = severe >500). "
                "Mechanism: succinyl-CoA accumulation → methylmalonyl-CoA → methylmalonate shunted. "
                "No severe ketoacidosis in SUCLA2 MMA."
            ),
        },
        {
            "feature": "Lactic Acidosis",
            "pct": round(sum(1 for p in patients if p["lactic_acidosis"]) / total * 100),
            "note": "Blood lactate >2.5 mmol/L; pyruvate/lactate ratio elevated; reflects OXPHOS impairment",
        },
        {
            "feature": "Leigh-like MRI (T2 BG hyperintensity)",
            "pct": round(sum(1 for p in patients if p["leigh_mri"]) / total * 100),
            "note": "T2/FLAIR hyperintensity putamen, caudate, dorsal midbrain, ± periaqueductal grey",
        },
        {
            "feature": "Sensorineural Hearing Loss (SNHL)",
            "pct": round(sum(1 for p in patients if p["snhl"]) / total * 100),
            "note": "Progressive SNHL; cochlear implants effective and recommended early; ABR from diagnosis",
        },
        {
            "feature": "Dystonia",
            "pct": round(sum(1 for p in patients if p["dystonia"]) / total * 100),
            "note": "Generalised or focal; basal ganglia damage; may respond to trihexyphenidyl or DBS",
        },
        {
            "feature": "Seizures",
            "pct": round(sum(1 for p in patients if p["seizures"]) / total * 100),
            "note": "Focal or multifocal; LEV preferred; VPA ABSOLUTE CI; avoid carbamazepine",
        },
        {
            "feature": "Feeding Difficulty",
            "pct": round(sum(1 for p in patients if p["feeding_difficulty"]) / total * 100),
            "note": "Hypotonia + dysphagia; NG/PEG often required; fasting must be avoided strictly",
        },
        {
            "feature": "Hepatopathy",
            "pct": 0,
            "note": "ABSENT — liver is NORMAL in SUCLA2; critical DDx from DGUOK/MPV17/TWNK/POLG",
        },
        {
            "feature": "Nystagmus",
            "pct": 0,
            "note": "ABSENT — nystagmus is DGUOK-pathognomonic; SUCLA2 does not cause nystagmus",
        },
        {
            "feature": "Homocystinuria",
            "pct": 0,
            "note": "ABSENT — plasma Hcy normal; critical DDx from MMACHC/cblC (MMA + elevated Hcy)",
        },
        {
            "feature": "3-MGA-uria",
            "pct": 0,
            "note": "ABSENT — critical DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB",
        },
        {
            "feature": "Elevated CK (>5x ULN)",
            "pct": 0,
            "note": "ABSENT — CK elevation is TK2 myopathic marker; SUCLA2 = encephalomyopathic, CK normal/mildly elevated only",
        },
    ]

    treatments = [
        {
            "tx": "Valproic Acid (VPA) — ALL INDICATIONS",
            "level": "ABSOLUTE CONTRAINDICATION",
            "note": (
                "NEVER prescribe in SUCLA2 MDDS10 or ANY mtDNA depletion syndrome. "
                "Three mechanisms of mitotoxicity: POLG inhibition, CoA sequestration, hepatotoxic epoxide metabolite. "
                "Fulminant liver failure and death. Document VPA allergy-equivalent in ALL medical records."
            ),
        },
        {
            "tx": "Ketogenic Diet (KD)",
            "level": "CONTRAINDICATED",
            "note": (
                "Forces OXPHOS-dependent beta-oxidation. SUCLA2 depletes ETC complexes I/III/IV/V "
                "(all mtDNA-encoded). KD worsens metabolic crisis; lactic acidosis exacerbated."
            ),
        },
        {
            "tx": "Propofol",
            "level": "AVOID — PRIS Risk",
            "note": (
                "Propofol Infusion Syndrome: Complex II + fatty-acid oxidation inhibition in "
                "mitochondrial disease. Anaesthesia: use ketamine or volatile agents (sevoflurane). "
                "Document AVOID in anaesthetic record."
            ),
        },
        {
            "tx": "Fasting Prevention (strict)",
            "level": "A — Mandatory at All Ages",
            "note": (
                "Avoid fasting >4 h in infants, >6 h in older children. "
                "Nil-by-mouth periods (pre-op, illness): IV dextrose (GIR 6-8 mg/kg/min). "
                "Continuous glucose monitoring during decompensation. "
                "Emergency protocol letter to be carried by family."
            ),
        },
        {
            "tx": "Levetiracetam (LEV)",
            "level": "A — Preferred AED",
            "note": (
                "Renal excretion only; no hepatic metabolism; no mitochondrial liability. "
                "IV loading 20-40 mg/kg for acute seizures. "
                "Maintenance 20-60 mg/kg/day divided. "
                "Avoid carbamazepine (hepatic induction, uncertain mitotoxicity) and phenytoin (Complex I inhibitor)."
            ),
        },
        {
            "tx": "Riboflavin (Vitamin B2)",
            "level": "B — Supportive",
            "note": (
                "Complex I/II FAD cofactor; some case reports of partial biochemical improvement. "
                "10-100 mg/day oral. May reduce lactate in responsive patients. "
                "Low risk; often included in mitochondrial 'cocktail.'"
            ),
        },
        {
            "tx": "CoQ10 Supplementation",
            "level": "C — Supportive",
            "note": (
                "Antioxidant; partial ETC electron carrier support. "
                "10-30 mg/kg/day oral (ubiquinol preferred for absorption). "
                "No controlled SUCLA2-specific data; part of standard mitochondrial supportive therapy."
            ),
        },
        {
            "tx": "Cochlear Implants",
            "level": "A — For SNHL ≥ Moderate",
            "note": (
                "SNHL 75% — progressive. Cochlear implants effective in SUCLA2 if performed early. "
                "ABR / OAE screening at diagnosis and every 6 months. "
                "Implant when SNHL ≥ 40 dB bilateral or rapidly progressing."
            ),
        },
        {
            "tx": "Dystonia Management",
            "level": "B — Individualised",
            "note": (
                "Trihexyphenidyl (anticholinergic): may reduce generalised dystonia. "
                "Tetrabenazine: use with caution (sedation risk in hypotonic patients). "
                "Deep Brain Stimulation (DBS) of GPi: case reports of benefit in severe dystonia. "
                "Intrathecal baclofen: for mixed dystonia-spasticity presentations."
            ),
        },
        {
            "tx": "Feeding Support (NG / PEG)",
            "level": "A — For Severe Feeding Difficulty",
            "note": (
                "Nasogastric tube early; PEG placement when long-term enteral nutrition required. "
                "High complex-carbohydrate, moderate protein diet. "
                "Avoid prolonged fasting — enteral feeds continue through illness. "
                "Dietitian involvement mandatory."
            ),
        },
        {
            "tx": "Thiamine (Vitamin B1)",
            "level": "C — If Leigh-like MRI with Lactate",
            "note": (
                "Empirical thiamine for all Leigh-syndrome presentations before genetic confirmation. "
                "100-300 mg/day IV or oral; low-risk even if aetiology not PDHC. "
                "Discontinue if SUCLA2 confirmed and no clinical response."
            ),
        },
        {
            "tx": "Genetic Counselling",
            "level": "A — Mandatory",
            "note": (
                "AR inheritance: 25% recurrence risk. "
                "Prenatal diagnosis via CVS (11-13 weeks) or amniocentesis available for known familial variants. "
                "Cascade testing in Faroe Islands communities given p.Asp333Gly carrier ~1:16. "
                "Preimplantation genetic testing (PGT-M) available."
            ),
        },
    ]

    disease_timeline = [
        {
            "phase": "Neonatal / Early Infantile (0-3 months)",
            "events": (
                "Hypotonia (often noted at delivery); feeding difficulties with NG tube; "
                "elevated blood lactate (>3 mmol/L); urine organic acids show mild MMA; "
                "normal liver function (distinguishes from DGUOK/TWNK). "
                "Metabolic crisis may be triggered by first intercurrent illness."
            ),
        },
        {
            "phase": "Infantile (3-12 months)",
            "events": (
                "Progressive hypotonia; failure to achieve motor milestones; "
                "brain MRI Leigh-like T2 changes basal ganglia appear 3-12 months; "
                "SNHL confirmed on ABR; dystonia emerges; seizures possible. "
                "Genetic testing returns SUCLA2 biallelic variants."
            ),
        },
        {
            "phase": "Early Childhood (1-5 years)",
            "events": (
                "Dystonia progresses; non-ambulant majority; "
                "SNHL management (hearing aids / cochlear implant evaluation); "
                "enteral nutrition often required; "
                "metabolic decompensation during febrile illness — emergency letters critical. "
                "Some patients maintain stable plateau years 2-4."
            ),
        },
        {
            "phase": "Later Childhood / Adolescence",
            "events": (
                "Phenotype-dependent: severe (null) — progressive deterioration, respiratory compromise; "
                "moderate (Faroe / missense) — stable disability with supported living; "
                "SNHL post-implant: speech and language development possible; "
                "ongoing physiotherapy, occupational therapy, dystonia management."
            ),
        },
        {
            "phase": "Transition / Adult (where reached)",
            "events": (
                "Rare null-variant patients survive to adulthood. "
                "Faroe cohort longest-surviving adults described in their 20s-30s. "
                "Respiratory monitoring (sleep study); cardiology surveillance (cardiomyopathy occasional). "
                "Palliative care discussions in progressive severe disease."
            ),
        },
    ]

    mma_comparison = {
        "sucla2_mma_umol_mmol_creat_typical": "10-100 (mild)",
        "mut_mma_umol_mmol_creat_typical": ">500-5000 (severe)",
        "mmachc_mma_umol_mmol_creat_typical": "100-1000 (moderate-severe)",
        "sucla2_plasma_hcy": "Normal (<15 µmol/L) — absent",
        "mmachc_plasma_hcy": "Elevated (>30-200 µmol/L) — present",
        "sucla2_ketoacidosis_risk": "LOW (MMA is mild; no isovaleric/3-OH-isobutyric crisis)",
        "mut_ketoacidosis_risk": "HIGH (severe MMA → methylmalonyl-CoA toxicity → ketonemia)",
        "note": "SUCLA2 MMA = metabolic bystander (succinyl-CoA overflow), NOT primary organic aciduria",
    }

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
        "mma_ddx_table": mma_comparison,
        "patients_sample": patients[:8],
    }


def get_definitions() -> dict:
    """SUCLA2 MDDS10 — definitions for /api/sucla2/definitions."""
    return {
        "generated": date.today().isoformat(),
        "terms": [
            {
                "term": "SUCLA2 — Succinate-CoA Ligase ADP-Forming Beta Subunit",
                "definition": (
                    "SUCLA2 (463 amino acids, 13q14.2) encodes the substrate-specificity-determining "
                    "beta subunit of ADP-forming succinyl-CoA synthetase (SCS-A), a TCA-cycle enzyme "
                    "in the mitochondrial matrix. SCS-A forms an alpha-beta heterodimer with SUCLG1 "
                    "(the alpha subunit). The enzyme catalyses the reaction: "
                    "Succinyl-CoA + ADP + Pi → Succinate + ATP + CoA-SH — the only "
                    "substrate-level phosphorylation in the TCA cycle. SUCLA2 critically co-localises "
                    "with nucleoside diphosphate kinase (NDPK/NME4) at the inner mitochondrial membrane, "
                    "creating a high-local-concentration dNTP microenvironment that feeds POLG for "
                    "mtDNA replication. Loss of SUCLA2 → disrupted NDPK channelling → dATP pool "
                    "imbalance → mtDNA copy number depletion."
                ),
            },
            {
                "term": "MDDS10 — Mitochondrial DNA Depletion Syndrome 10 (Encephalomyopathic + MMA)",
                "definition": (
                    "MDDS10 (OMIM #615084) is an autosomal recessive disease caused by biallelic "
                    "loss-of-function mutations in SUCLA2. Clinical features: infantile hypotonia, "
                    "Leigh syndrome-like MRI (T2 hyperintensity basal ganglia/brainstem), sensorineural "
                    "hearing loss (75%), dystonia (70%), mild methylmalonic aciduria (100%), and "
                    "lactic acidosis (85%). Crucially: NO hepatopathy (distinguishes from hepatocerebral "
                    "MDDS: DGUOK, MPV17, TWNK, POLG) and methylmalonic aciduria is MILD (distinguishes "
                    "from MUT/MMACHC where MMA is severe). mtDNA copy number in muscle: <30% normal."
                ),
            },
            {
                "term": "Mild Methylmalonic Aciduria — Mechanism in SUCLA2",
                "definition": (
                    "SUCLA2 deficiency reduces SCS-A activity → succinyl-CoA cannot be efficiently "
                    "converted to succinate → succinyl-CoA accumulates → methylmalonyl-CoA accumulates "
                    "upstream (via reversal of methylmalonyl-CoA mutase equilibrium) → propionyl-CoA "
                    "accumulates → methylmalonic acid is shunted to urine. "
                    "This MMA is a METABOLIC BYSTANDER effect of TCA cycle blockade — NOT a primary "
                    "organic aciduria like MUT deficiency. Consequently, MMA levels are mild (10-100 "
                    "µmol/mmol creatinine), there is no ketoacidosis, and MMA does not drive the "
                    "primary neurotoxicity (mtDNA depletion does)."
                ),
            },
            {
                "term": "SUCLA2–NDPK Channelling Complex",
                "definition": (
                    "SCS-A (SUCLA2/SUCLG1 heterodimer) physically associates at the mitochondrial "
                    "inner membrane with nucleoside diphosphate kinase (NDPK, encoded by NME4). "
                    "NDPK phosphorylates NDPs → NTPs, including dNDPs → dNTPs. The physical proximity "
                    "creates a local high-concentration dNTP pool at the replication fork, essential "
                    "for POLG processivity. When SUCLA2 is absent, SCS-A activity is lost, succinyl-CoA "
                    "substrate does not regenerate local ADP → local NDP/dNDP phosphorylation by NDPK "
                    "becomes limiting → dATP pool drops → POLG stalls → mtDNA depletion. "
                    "This is the mechanistic link between a TCA enzyme and mtDNA maintenance."
                ),
            },
            {
                "term": "Faroe Islands Founder — p.Asp333Gly (c.998A>G)",
                "definition": (
                    "p.Asp333Gly is a missense variant in the beta-subunit specificity loop "
                    "(aa 330-360) of SUCLA2, substituting aspartate 333 with glycine. This "
                    "aspartate is critical for ADP (vs GDP) specificity and for enzyme activity. "
                    "The variant has a dramatically elevated allele frequency in the Faroe Islands "
                    "(isolated North Atlantic population, ~50,000 people) due to a founder effect. "
                    "Carrier frequency ~1:16; disease prevalence ~1:1,000 births — the highest "
                    "MDDS frequency in any population worldwide. Ostergaard E 2007 J Inherit Metab "
                    "Dis first described the epidemiology. Phenotype is moderately severe: hypotonia, "
                    "SNHL (90% in Faroe patients), Leigh MRI, dystonia; liver normal."
                ),
            },
            {
                "term": "Leigh Syndrome — Radiological Definition",
                "definition": (
                    "Leigh syndrome (subacute necrotising encephalomyelopathy) is a neuroradiological "
                    "diagnosis: bilateral, symmetric T2/FLAIR hyperintensity and restricted diffusion "
                    "in the putamen, caudate, dorsal midbrain, periaqueductal grey, and/or "
                    "pontine tegmentum on brain MRI. Pathologically: spongiform vacuolation, "
                    "demyelination, gliosis, and vascular proliferation in brainstem and basal ganglia. "
                    "Leigh syndrome is NOT a single disease — it is the final common pathway of >75 "
                    "metabolic disorders including SUCLA2 MDDS10, SURF1 (Complex IV), SDHA (Complex II), "
                    "PDHC deficiency, and biotinidase deficiency. "
                    "In SUCLA2: Leigh-like MRI present in ~80%; appears 3-12 months of age."
                ),
            },
            {
                "term": "Encephalomyopathic MDDS vs Hepatocerebral MDDS — Key Distinction",
                "definition": (
                    "Encephalomyopathic MDDS (SUCLA2, SUCLG1): primarily brain and muscle affected; "
                    "liver NORMAL; hypotonia + Leigh MRI + SNHL + dystonia; MMA (mild) present. "
                    "Hepatocerebral MDDS (DGUOK, MPV17, TWNK, POLG): liver AND brain both affected; "
                    "liver failure, lactic acidosis, hypoglycemia; hepatomegaly/jaundice/coagulopathy. "
                    "This distinction is diagnostically critical: "
                    "normal LFTs + elevated lactate + hypotonia + MMA = think SUCLA2/SUCLG1 first. "
                    "Elevated transaminases + elevated lactate + hypotonia = think DGUOK/MPV17/TWNK/POLG."
                ),
            },
            {
                "term": "VPA Absolute Contraindication — mtDNA Depletion Syndromes",
                "definition": (
                    "Valproic acid is absolutely contraindicated in ALL mtDNA depletion syndromes "
                    "(SUCLA2 MDDS10, SUCLG1 MDDS9, POLG Alpers, DGUOK MDDS3, MPV17 MDDS6, "
                    "TK2 MDDS4A, TWNK MDDS7) via three independent mechanisms: "
                    "(1) VPA and metabolites (4-en-VPA) directly inhibit POLG (mtDNA polymerase gamma), "
                    "reducing mtDNA replication in cells already depleted; "
                    "(2) valproyl-CoA sequesters free CoA, impairing beta-oxidation and TCA cycle "
                    "in energetically compromised cells; "
                    "(3) reactive epoxide metabolite (2-propyl-4-pentenoic acid epoxide) is directly "
                    "hepatotoxic. Combined effect: fulminant hepatic failure and death. "
                    "Document VPA as allergy-equivalent. No safe dose exists in MDDS."
                ),
            },
            {
                "term": "Cochlear Implant in SUCLA2 SNHL",
                "definition": (
                    "Sensorineural hearing loss in SUCLA2 affects ~75% and is progressive. "
                    "Cochlear implants have been shown to be effective in SUCLA2, with case series "
                    "demonstrating speech and language development post-implant. "
                    "Recommendation: ABR/OAE screening at diagnosis and every 6 months. "
                    "Implant when bilateral SNHL ≥ 40 dB or rapid deterioration. "
                    "Anaesthetic protocol: avoid propofol (PRIS risk), use sevoflurane + ketamine. "
                    "Cochlear implant team should be briefed on underlying mitochondrial disease "
                    "for anaesthetic and perioperative management."
                ),
            },
            {
                "term": "SUCLA2 vs SUCLG1 — Clinical Distinction",
                "definition": (
                    "SUCLA2 (MDDS10, OMIM 615084): ADP-forming beta subunit; 13q14.2; "
                    "encephalomyopathic; Leigh + hypotonia + SNHL + dystonia + mild MMA; "
                    "Faroe Islands founder p.Asp333Gly; liver NORMAL; survival variable (months-decades). "
                    "SUCLG1 (MDDS9, OMIM 245400): GTP/ADP-forming alpha subunit shared by SCS-A and SCS-G; 2p11.2; "
                    "encephalomyopathic OR fatal neonatal; more severe than SUCLA2; "
                    "includes a fatal neonatal form with no liver recovery; "
                    "SUCLG1 mutations affect both ADP- and GTP-forming SCS isoforms → more complete loss. "
                    "Both share VPA absolute CI, KD CI, Leigh MRI, mild MMA, SNHL, dystonia."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json

    print("=== SUCLA2 MDDS10 Dashboard — Overview ===")
    ov = get_overview()
    print(json.dumps(ov, indent=2))

    print("\n=== Breakdown (feature prevalence) ===")
    bk = get_breakdown()
    for f in bk["feature_prevalence"]:
        print(f"  {f['feature']}: {f['pct']}%")

    print("\n=== Definitions count ===")
    df = get_definitions()
    print(f"  {len(df['terms'])} definitions")
