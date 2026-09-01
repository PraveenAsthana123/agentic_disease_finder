#!/usr/bin/env python3
"""SUCLG1 Encephalomyopathic mtDNA Depletion Syndrome Dashboard.

Mitochondrial DNA Depletion Syndrome 9 (MDDS9) = OMIM #612235
Also: SCS-Deficiency / Encephalomyopathic MDDS with Severe Methylmalonic Aciduria

SUCLG1 (succinyl-CoA ligase [GDP-forming] alpha subunit; 394 aa precursor / 362 aa mature; 2p11.2)
encodes the shared alpha subunit of BOTH isoforms of succinyl-CoA synthetase:
  SCS-A (ADP-forming):  SUCLG1 + SUCLA2 heterodimer → ADP
  SCS-G (GTP-forming):  SUCLG1 + SUCLG2 heterodimer → GTP
Loss of SUCLG1 abolishes BOTH SCS-A and SCS-G activity, explaining why SUCLG1 disease is
typically MORE SEVERE than SUCLA2 disease (which only loses SCS-A).

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — mtDNA depletion disease; shared mechanism with
     POLG/DGUOK/MPV17/TK2/TWNK/SUCLA2 (CoA sequestration + mtDNA depletion)
  2. KD = CONTRAINDICATED — OXPHOS-dependent beta-oxidation fails in mtDNA depletion
  3. MMA is SEVERE — urine MMA >500 µmol/mmol creatinine (often 500-3000); KEY DDx from
     SUCLA2 (MMA mild 10-100 µmol/mmol creat); SUCLG1 can cause MMA ketoacidosis
  4. HEPATOPATHY present ~70% — KEY DDx from SUCLA2 (NO hepatopathy); liver failure
     and elevated transaminases; SCS-G loss affects hepatic GTP-dependent processes
  5. C4-DC (succinylcarnitine) ELEVATED — PATHOGNOMONIC for SCS axis (both SUCLG1 and
     SUCLA2); order acylcarnitine profile when encephalomyopathy + MMA are found together
  6. Both SCS-A AND SCS-G deficient (vs SUCLA2: SCS-A only) → greater metabolic burden
     → typically earlier onset, more severe, higher mortality in neonatal period
  7. SNHL ~40% — less prevalent than SUCLA2 (75%); screen audiogram at diagnosis
  8. Leigh-like MRI ~60% — T2 hyperintensity basal ganglia + brainstem (similar pattern
     to SUCLA2 but often earlier and more extensive)
  9. Seizures ~65% — myoclonic and focal; may include neonatal seizures
 10. Hypotonia 100% — universal; neonatal onset; often more profound than SUCLA2
 11. Propofol = AVOID (PRIS risk — mitochondrial disease universal rule)
 12. LEV preferred AED — renal excretion, no hepatic metabolism, no CoA interaction
 13. NO specific disease-modifying therapy (unlike TK2 deoxynucleoside rescue); treatment
     is supportive: emergency letters, metabolic monitoring, symptom management

SUCLG1 BIOLOGY:
SUCLG1 (394 amino acids precursor including 32 aa MTS; 362 aa mature; 2p11.2) encodes
the alpha subunit that is shared between the two mitochondrial succinyl-CoA synthetase
(SCS) isoforms. Both isoforms catalyse the same TCA-cycle reaction:

  Succinyl-CoA + ADP/GDP + Pi → Succinate + CoA-SH + ATP/GTP

This is the sole substrate-level phosphorylation step in the TCA cycle occurring inside
the mitochondrial matrix. CoA-SH regeneration is essential for continued TCA flux.

SUCLG1 domains (alpha subunit):
  Mitochondrial targeting sequence (MTS): aa 1-32 (cleaved post-import)
  N-terminal CoA-binding domain: aa 33-~180 (thioesters of succinyl-CoA)
  Nucleotide-binding (lid) domain: aa ~180-280 (P-loop Walker A Gly-X-X-X-X-Gly-Lys)
  Dimer interface: aa ~280-362; contacts beta subunit (SUCLA2 or SUCLG2)
  Active site histidine phosphorylation (His259): catalytic phosphate intermediate

LOSS of SUCLG1 → BOTH SCS-A (SUCLG1+SUCLA2) and SCS-G (SUCLG1+SUCLG2) inactive:
  1. SCS-A loss → disrupts SUCLA2-NDPK dNTP micro-compartment → dATP pool ↓ → POLG
     stalls → mtDNA depletion in brain, muscle, liver (as in SUCLA2 but more extensive)
  2. SCS-G loss → GTP pool ↓ in hepatocytes → impaired PEPCK (GTP-dependent, rate-limiting
     step of gluconeogenesis) → fasting hypoglycaemia + hepatic dysfunction
  3. Succinyl-CoA accumulates → methylmalonyl-CoA accumulates (SEVERE, unlike SUCLA2 mild)
     → methylmalonic acid shunted to urine → SEVERE MMA with potential ketoacidosis
  4. CoA sequestration → global mitochondrial CoA shortage → OXPHOS impairment worsens

SUCCINYLCARNITINE (C4-DC) ELEVATION:
  Elevated succinylcarnitine (C4-DC) in acylcarnitine profile is a PATHOGNOMONIC
  marker of the SCS axis (SUCLG1 and SUCLA2). C4-DC (butanediylcarnitine, MW 289)
  is the carnitine conjugate of succinic acid. Differential: malonate semialdehyde
  dehydrogenase (ALDH6A1) deficiency — but ALDH6A1 does NOT cause mtDNA depletion.
  If C4-DC elevated + MMA elevated + hypotonia: test SUCLG1 and SUCLA2 in parallel.
  SUCLA2 C4-DC: present but often lower. SUCLG1 C4-DC: typically higher (both SCS lost).

HEPATOPATHY MECHANISM (SUCLG1 unique vs SUCLA2):
  Hepatocytes rely heavily on SCS-G (SUCLG1+SUCLG2) for GTP production. GTP is the
  obligate substrate for PEPCK (phosphoenolpyruvate carboxykinase), the rate-limiting
  step of hepatic gluconeogenesis. Loss of SCS-G → GTP deficit in hepatocytes → PEPCK
  impairment → fasting hypoglycaemia + hepatic failure. SUCLA2 (SCS-A only) does NOT
  affect SCS-G → liver GTP and PEPCK are intact → NO hepatopathy in SUCLA2 disease.

GENOTYPE-PHENOTYPE CORRELATION:
  Biallelic null (nonsense/frameshift) → catastrophic neonatal; both SCS-A and SCS-G
    ablated; severe hypotonia; lactic acidosis pH <7.1; hepatic failure first weeks;
    fatal within days to months without intensive support
  Missense affecting CoA-binding domain → severe encephalomyopathic; hepatopathy 80%;
    MMA severe; median survival 1-2 years
  Missense at dimer interface → moderate-severe; partial residual enzyme activity;
    hepatopathy present; MMA 300-800 µmol/mmol creat
  Walker A P-loop missense → variable; reduces nucleotide binding; severity depends on
    residual phosphorylation capacity
  Splice-site/missense compound het → variable; partial splice rescue may moderate

KEY DDx PEARLS:
  vs SUCLA2 (MDDS10): SUCLA2 → MMA MILD (10-100); NO hepatopathy; SNHL 75% (>SUCLG1);
    If MMA severe >500 + hepatopathy → SUCLG1 until proven otherwise
  vs MUT methylmalonic aciduria: MUT → severe MMA but NO mtDNA depletion; NO
    encephalomyopathy as primary; NO C4-DC; liver normal or secondarily injured
  vs MMACHC (cblC): MMA + homocystinuria (SUCLG1 NO Hcy); retinopathy;
    responds to hydroxocobalamin (SUCLG1 does not)
  vs DGUOK (MDDS3): hepatocerebral; nystagmus 90%; NO MMA; NO C4-DC
  vs TWNK (MDDS7): hepatocerebral; IOSCA spinocerebellar; NO MMA; NO C4-DC
  vs MPV17 (MDDS6): hepatocerebral; peripheral neuropathy; NO MMA; NO C4-DC
  vs POLG (Alpers): hepatocerebral; EPC (epilepsia partialis continua) 60%;
    NO MMA; NO C4-DC; European founder Ala467Thr/Trp748Ser
  SUCLG1 FINGERPRINT: severe MMA + C4-DC elevated + hepatopathy + hypotonia + mtDNA
    depletion + no homocystinuria → SUCLG1 panel first

References:
  Ostergaard E 2007 J Inherit Metab Dis — first description SUCLG1 mutations; MDDS9
  Carrozzo R 2007 Pediatrics — SUCLA2/SUCLG1 spectrum review; MMA severity comparison
  Van Hove JLK 2010 J Inherit Metab Dis — comprehensive review SCS deficiency spectrum
  Lamperti C 2012 Neurology — genotype-phenotype correlation SUCLG1
  Stiles AR 2015 Mol Genet Metab — hepatopathy mechanism in SUCLG1 vs SUCLA2
"""

import random
from datetime import date

SEED = 561  # 40-patient cohort seed


def get_overview() -> dict:
    """SUCLG1 MDDS9 — top-level overview for /api/suclg1/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "SUCLG1 Encephalomyopathic mtDNA Depletion Syndrome (MDDS9)",
        "gene": "SUCLG1",
        "protein": "Succinyl-CoA Ligase GDP-Forming Alpha Subunit (Shared SCS-A/SCS-G alpha)",
        "protein_size_aa": 394,
        "mature_protein_aa": 362,
        "locus": "2p11.2",
        "inheritance": "Autosomal Recessive (AR)",
        "omim_gene": "611224",
        "omim_disease": "612235",
        "mechanism": (
            "Biallelic SUCLG1 loss-of-function → BOTH SCS-A (SUCLG1+SUCLA2) AND SCS-G (SUCLG1+SUCLG2) "
            "deficient → succinyl-CoA accumulation → SEVERE methylmalonic aciduria (succinyl-CoA → "
            "methylmalonyl-CoA overflow) + disrupted SUCLA2-NDPK dNTP micro-compartment (SCS-A loss) → "
            "dATP pool depletion → POLG stalls → mtDNA copy number depletion in brain, muscle, liver; "
            "ADDITIONALLY SCS-G loss → hepatic GTP deficit → PEPCK impairment → hepatopathy + fasting "
            "hypoglycaemia (unique vs SUCLA2 which preserves SCS-G)"
        ),
        "key_contraindications": [
            {
                "drug": "Valproic Acid (VPA)",
                "level": "ABSOLUTE CONTRAINDICATION",
                "mechanism": (
                    "VPA inhibits POLG (mtDNA polymerase gamma) directly; CoA sequestration by "
                    "valproyl-CoA reduces mitochondrial CoA availability; epoxide metabolites cause "
                    "hepatotoxicity — synergistically lethal in SUCLG1 where mtDNA depletion, "
                    "CoA shortage, and hepatopathy are pre-existing vulnerabilities. "
                    "Same as all MDDS (POLG/DGUOK/MPV17/TK2/TWNK/SUCLA2)."
                ),
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "CONTRAINDICATED",
                "mechanism": (
                    "KD forces reliance on fatty acid beta-oxidation and OXPHOS for ATP production. "
                    "In SUCLG1 disease, mtDNA depletion impairs OXPHOS capacity → cells cannot sustain "
                    "the metabolic demand of KD → metabolic crisis, lactic acidosis, energy collapse. "
                    "Contrary to its use in channelopathies (SCN1A/KCNQ2), KD is contraindicated in "
                    "all mtDNA depletion syndromes."
                ),
            },
            {
                "drug": "Propofol",
                "level": "AVOID — PRIS Risk",
                "mechanism": (
                    "Propofol infusion syndrome (PRIS) risk universally elevated in mitochondrial disease. "
                    "Propofol inhibits Complex I and II of the ETC; in SUCLG1 (pre-existing OXPHOS "
                    "impairment from mtDNA depletion), additional ETC inhibition can precipitate PRIS: "
                    "metabolic acidosis, rhabdomyolysis, cardiac failure. Use alternative anaesthetics "
                    "(sevoflurane, ketamine at lower doses, dexmedetomidine)."
                ),
            },
        ],
        "key_features": {
            "hypotonia": {"pct": 100, "note": "Universal; neonatal/early infantile onset; often profound"},
            "methylmalonic_aciduria_severe": {"pct": 100, "note": "SEVERE MMA >500 µmol/mmol creat — KEY DDx from SUCLA2 (mild 10-100)"},
            "lactic_acidosis": {"pct": 100, "note": "Elevated blood lactate >3 mmol/L; reflects OXPHOS impairment"},
            "hepatopathy": {"pct": 70, "note": "KEY DDx from SUCLA2 (no hepatopathy); elevated transaminases ± liver failure"},
            "c4dc_succinylcarnitine": {"pct": 95, "note": "Elevated C4-DC on acylcarnitine — PATHOGNOMONIC for SCS axis; order with MMA"},
            "leigh_mri": {"pct": 60, "note": "T2 hyperintensity basal ganglia + brainstem; often earlier than SUCLA2"},
            "seizures": {"pct": 65, "note": "Myoclonic and focal; neonatal seizures in severe cases"},
            "dystonia": {"pct": 50, "note": "Movement disorder; less prominent than SUCLA2 (70%) due to earlier mortality"},
            "snhl": {"pct": 40, "note": "Less than SUCLA2 (75%); screen at diagnosis with ABR/OAE"},
            "feeding_difficulty": {"pct": 90, "note": "NG tube early; PEG for long-term enteral nutrition"},
            "fasting_hypoglycaemia": {"pct": 55, "note": "SCS-G loss → PEPCK impairment → gluconeogenesis failure; avoid fasting"},
        },
        "severity_vs_sucla2": (
            "SUCLG1 disease is TYPICALLY MORE SEVERE than SUCLA2 because SUCLG1 ablates BOTH SCS-A "
            "(shared with SUCLA2) and SCS-G (unique to SUCLG1 axis). SCS-G loss causes hepatopathy "
            "and fasting hypoglycaemia not seen in SUCLA2. Neonatal lethality is more common in "
            "biallelic null SUCLG1. SUCLA2+SUCLG1 together account for ~50% of all encephalomyopathic "
            "MDDS; when MMA is severe (>500) and hepatopathy is present, SUCLG1 is the leading diagnosis."
        ),
        "first_author": "Ostergaard E",
        "first_publication_year": 2007,
        "first_journal": "J Inherit Metab Dis",
        "cohort_n": 40,
        "seed": SEED,
    }


def get_breakdown() -> dict:
    """SUCLG1 MDDS9 — 40-patient cohort breakdown for /api/suclg1/breakdown."""
    rng = random.Random(SEED)

    genotypes = [
        {
            "variant_class": "Biallelic null (nonsense/frameshift)",
            "n": 10,
            "pct": 25,
            "note": (
                "Complete SCS-A + SCS-G loss; most severe phenotype; neonatal lactic acidosis, "
                "hepatic failure first weeks; MMA >1000 µmol/mmol creat; often fatal within months."
            ),
        },
        {
            "variant_class": "CoA-binding domain missense compound het",
            "n": 12,
            "pct": 30,
            "note": (
                "Impaired succinyl-CoA binding; severe encephalomyopathic + hepatopathy; "
                "MMA 500-900 µmol/mmol creat; median survival 1-3 years."
            ),
        },
        {
            "variant_class": "Dimer interface missense compound het",
            "n": 9,
            "pct": 22,
            "note": (
                "Impaired alpha–beta subunit dimerisation; partial residual SCS activity; "
                "moderate-severe; hepatopathy 60%; MMA 300-600 µmol/mmol creat."
            ),
        },
        {
            "variant_class": "P-loop (Walker A) missense",
            "n": 5,
            "pct": 13,
            "note": (
                "Reduced nucleotide (ADP/GDP) binding; variable residual activity; "
                "phenotype ranges moderate-severe; seizures prominent; C4-DC clearly elevated."
            ),
        },
        {
            "variant_class": "Splice-site/missense compound het",
            "n": 4,
            "pct": 10,
            "note": (
                "Partial splice rescue possible; variable severity; some patients survive "
                "to mid-childhood with supportive care; MMA 200-500 µmol/mmol creat."
            ),
        },
    ]

    phenotypes_dist = [
        ("Severe neonatal (biallelic null + hepatic failure)", 12),
        ("Encephalomyopathic + hepatopathy (CoA-domain missense)", 16),
        ("Encephalomyopathic + MRI-only (interface/splice)", 12),
    ]

    sex_choices = ["M", "F"]
    ethnic_groups = [
        "Italian",
        "Turkish",
        "Moroccan",
        "Pakistani",
        "Indian (South Asian)",
        "Palestinian",
        "Norwegian",
        "Spanish",
        "French",
        "German",
        "Egyptian",
    ]
    variant_pairs = [
        ("p.Arg313*/p.Gln178Argfs*8", "Biallelic null / nonsense-frameshift"),
        ("p.Gly209Arg/p.Thr224Ile", "CoA-domain missense compound het"),
        ("p.Leu153Pro/p.Arg202Cys", "CoA-domain missense compound het"),
        ("p.Ala284Val/p.Glu301Lys", "Dimer interface missense compound het"),
        ("p.Ser278Phe/p.Arg256His", "Dimer interface missense compound het"),
        ("p.Gly201Glu/p.Phe206Leu", "P-loop Walker A missense compound het"),
        ("c.387+1G>A/p.Thr193Met", "Splice-site/missense compound het"),
        ("p.Tyr180*/p.Arg250Trp", "Nonsense/missense compound het"),
    ]

    patients = []
    pid = 1
    for phenotype_name, count in phenotypes_dist:
        for _ in range(count):
            sex = rng.choice(sex_choices)
            ethnicity = rng.choice(ethnic_groups)
            variant_pair = rng.choice(variant_pairs)
            onset_mo = rng.randint(0, 4)  # earlier onset than SUCLA2
            diag_mo = onset_mo + rng.randint(2, 18)
            hypotonia = True  # 100%
            mma = True  # 100%
            hepatopathy = rng.random() < 0.70
            leigh_mri = rng.random() < 0.60
            snhl = rng.random() < 0.40
            dystonia = rng.random() < 0.50
            seizures = rng.random() < 0.65
            lactic_acidosis = True  # 100% in this severe disease
            feeding_difficulty = rng.random() < 0.90
            fasting_hypoglycaemia = rng.random() < 0.55
            mtdna_pct = rng.randint(4, 25)  # often more depleted than SUCLA2
            mma_value = rng.randint(300, 2800)  # SEVERE MMA
            lactate = round(rng.uniform(3.5, 12.0), 1)
            c4dc_elevated = rng.random() < 0.95
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
                "methylmalonic_aciduria_severe": mma,
                "mma_urine_umol_per_mmol_creat": mma_value,
                "lactic_acidosis": lactic_acidosis,
                "peak_lactate_mmol": lactate,
                "hepatopathy": hepatopathy,
                "fasting_hypoglycaemia": fasting_hypoglycaemia,
                "leigh_mri": leigh_mri,
                "snhl": snhl,
                "dystonia": dystonia,
                "seizures": seizures,
                "feeding_difficulty": feeding_difficulty,
                "mtdna_copy_pct_muscle": mtdna_pct,
                "c4dc_succinylcarnitine_elevated": c4dc_elevated,
            })
            pid += 1

    total = len(patients)
    feature_prevalence = [
        {
            "feature": "Hypotonia",
            "pct": 100,
            "note": "Universal; neonatal onset; often more profound than SUCLA2; ventilator-dependent in severe cases",
        },
        {
            "feature": "Methylmalonic Aciduria (SEVERE)",
            "pct": 100,
            "note": (
                "Urine MMA >500 µmol/mmol creatinine (often 500-3000) — SEVERE. "
                "KEY DDx: SUCLA2 MMA is MILD (10-100 µmol/mmol creat). "
                "SUCLG1 MMA can precipitate ketoacidosis unlike SUCLA2. "
                "Mechanism: succinyl-CoA → methylmalonyl-CoA overflow via both SCS-A and SCS-G loss."
            ),
        },
        {
            "feature": "Lactic Acidosis",
            "pct": 100,
            "note": (
                "Blood lactate typically >4 mmol/L; pyruvate/lactate ratio elevated (>20:1). "
                "Reflects OXPHOS impairment from mtDNA depletion. Often more severe than SUCLA2."
            ),
        },
        {
            "feature": "C4-DC (Succinylcarnitine) Elevated",
            "pct": round(sum(1 for p in patients if p["c4dc_succinylcarnitine_elevated"]) / total * 100),
            "note": (
                "PATHOGNOMONIC for SCS axis (SUCLG1 and SUCLA2 both show C4-DC elevation). "
                "Order acylcarnitine profile alongside urine organic acids in any child with "
                "encephalomyopathy + MMA. C4-DC levels often higher in SUCLG1 than SUCLA2."
            ),
        },
        {
            "feature": "Hepatopathy",
            "pct": round(sum(1 for p in patients if p["hepatopathy"]) / total * 100),
            "note": (
                "Elevated transaminases ± hepatomegaly ± liver failure. "
                "KEY DDx from SUCLA2 (NO hepatopathy). "
                "Mechanism: SCS-G loss → GTP deficit → PEPCK impairment → hepatocyte dysfunction. "
                "Monitor LFTs regularly; avoid hepatotoxic drugs."
            ),
        },
        {
            "feature": "Fasting Hypoglycaemia",
            "pct": round(sum(1 for p in patients if p["fasting_hypoglycaemia"]) / total * 100),
            "note": (
                "SCS-G loss → hepatic GTP ↓ → PEPCK (GTP-dependent, rate-limiting gluconeogenesis step) "
                "impaired → fasting hypoglycaemia. Prevent by continuous enteral feeds; "
                "emergency IV dextrose (GIR 8-10 mg/kg/min) during any illness."
            ),
        },
        {
            "feature": "Leigh-like MRI",
            "pct": round(sum(1 for p in patients if p["leigh_mri"]) / total * 100),
            "note": (
                "T2 hyperintensity basal ganglia (putamen, caudate), brainstem. "
                "Similar to SUCLA2 but often earlier onset and more extensive. "
                "60% SUCLG1 vs 80% SUCLA2 (lower in SUCLG1 because many severe cases die before "
                "full MRI workup)."
            ),
        },
        {
            "feature": "Seizures",
            "pct": round(sum(1 for p in patients if p["seizures"]) / total * 100),
            "note": (
                "Myoclonic, focal, and neonatal seizures. LEV preferred (renal excretion). "
                "VPA ABSOLUTE CI. Consider perampanel for refractory myoclonic. "
                "EEG: may show burst-suppression pattern in severe neonatal form."
            ),
        },
        {
            "feature": "Dystonia",
            "pct": round(sum(1 for p in patients if p["dystonia"]) / total * 100),
            "note": (
                "Movement disorder; less prevalent than SUCLA2 (70%) partly because "
                "severe cases do not survive long enough to develop prominent dystonia. "
                "Trihexyphenidyl/tetrabenazine cautiously; baclofen for mixed spasticity."
            ),
        },
        {
            "feature": "SNHL (Sensorineural Hearing Loss)",
            "pct": round(sum(1 for p in patients if p["snhl"]) / total * 100),
            "note": (
                "~40% — significantly less than SUCLA2 (75%). "
                "Mechanism less clear than SUCLA2; possibly relates to lower survival to auditory "
                "screening age in severe cases. Screen ABR at diagnosis in all SUCLG1 patients."
            ),
        },
        {
            "feature": "Feeding Difficulty",
            "pct": round(sum(1 for p in patients if p["feeding_difficulty"]) / total * 100),
            "note": "NG tube early; PEG for long-term; high complex-carbohydrate diet; avoid prolonged fasting",
        },
    ]

    treatments = [
        {
            "tx": "Emergency Metabolic Protocol (Illness Letter)",
            "level": "A — Mandatory",
            "note": (
                "All SUCLG1 patients require an emergency letter for any febrile illness or surgery. "
                "IV dextrose (GIR 8-10 mg/kg/min) to suppress catabolism; correct acidosis with NaHCO3; "
                "correct hypoglycaemia aggressively (SCS-G loss impairs gluconeogenesis). "
                "VPA ABSOLUTE CI even in emergency. Propofol AVOID."
            ),
        },
        {
            "tx": "LEV (Levetiracetam) — AED of Choice",
            "level": "A — Preferred AED",
            "note": (
                "Levetiracetam: renal excretion, no hepatic metabolism, no CoA interaction, "
                "no ETC inhibition. Safe in SUCLG1. Starting dose 10-20 mg/kg/day in 2 doses; "
                "titrate to 40-60 mg/kg/day. For myoclonic: perampanel or clonazepam (low doses). "
                "VPA remains ABSOLUTE CI regardless of seizure severity."
            ),
        },
        {
            "tx": "Riboflavin (Vitamin B2)",
            "level": "B — Supportive",
            "note": (
                "Complex I/II FAD cofactor; case reports of partial lactate reduction. "
                "10-100 mg/day oral. Low risk. Include in mitochondrial 'cocktail.' "
                "No controlled SUCLG1-specific data."
            ),
        },
        {
            "tx": "CoQ10 Supplementation",
            "level": "C — Supportive",
            "note": (
                "Antioxidant; partial ETC support. 10-30 mg/kg/day oral (ubiquinol preferred). "
                "No controlled SUCLG1-specific data; standard supportive mitochondrial therapy."
            ),
        },
        {
            "tx": "Continuous Enteral Nutrition",
            "level": "A — To Prevent Fasting",
            "note": (
                "Fasting hypoglycaemia from SCS-G/PEPCK impairment is a unique SUCLG1 risk. "
                "Continuous NG or PEG feeds overnight; never fast >4-6 hours. "
                "High complex-carbohydrate (50-60%), moderate protein (8-10%), restricted fat. "
                "KD CONTRAINDICATED. Dietitian involvement mandatory."
            ),
        },
        {
            "tx": "Hepatology Monitoring",
            "level": "A — Mandatory (given hepatopathy 70%)",
            "note": (
                "Monitor LFTs (ALT, AST, GGT, bilirubin) every 3 months. "
                "Liver ultrasound 6-monthly. Coagulation screen. "
                "Liver failure may require emergency support; no transplant data in SUCLG1 "
                "(unlike hepatocerebral MDDS such as DGUOK hepatic-only form). "
                "Avoid any hepatotoxic drugs."
            ),
        },
        {
            "tx": "Audiological Surveillance",
            "level": "B — For SNHL Screening",
            "note": (
                "ABR at diagnosis; repeat every 6 months. ~40% SNHL. "
                "Early hearing aids if SNHL ≥20 dB. Cochlear implants can be considered "
                "if SNHL ≥40 dB bilateral and patient is sufficiently stable. "
                "Neurological prognosis guides cochlear implant decision."
            ),
        },
        {
            "tx": "Thiamine Empirical",
            "level": "C — Pre-diagnosis Empirical",
            "note": (
                "Thiamine 100-300 mg/day IV empirically for all Leigh-syndrome presentations "
                "before genetic confirmation (thiamine-responsive PDHC mimics Leigh; low risk). "
                "Discontinue when SUCLG1 confirmed and no clinical response."
            ),
        },
        {
            "tx": "Dystonia Management",
            "level": "B — Individualised",
            "note": (
                "Trihexyphenidyl (anticholinergic): 0.5-2 mg/day, titrate slowly; "
                "monitor sedation in hypotonic patients. Tetrabenazine: low doses, monitor QTc. "
                "Intrathecal baclofen for severe mixed dystonia-spasticity. "
                "DBS (GPi) case reports; neurological stability required."
            ),
        },
        {
            "tx": "Genetic Counselling",
            "level": "A — Mandatory",
            "note": (
                "AR inheritance: 25% recurrence risk. Prenatal diagnosis via CVS (11-13 weeks) "
                "or amniocentesis for known familial variants. No population founder like SUCLA2 "
                "Faroe, but consanguineous populations (Turkish, Pakistani, Palestinian) enriched. "
                "Preimplantation genetic testing (PGT-M) available."
            ),
        },
    ]

    disease_timeline = [
        {
            "phase": "Neonatal (0-4 weeks)",
            "events": (
                "Hypotonia (often noted at delivery or within hours); profound neonatal lactic acidosis "
                "(pH <7.1 in severe); feeding failure requiring NG tube; elevated blood lactate >5 mmol/L; "
                "severe MMA on urine organic acids (>500 µmol/mmol creat); C4-DC elevated on acylcarnitine; "
                "liver function abnormal in severe cases. Neonatal seizures possible."
            ),
        },
        {
            "phase": "Early Infantile (1-3 months)",
            "events": (
                "Progressive hypotonia; failure to thrive; MRI brain shows Leigh-like T2 changes "
                "(basal ganglia, brainstem) — earlier and often more extensive than SUCLA2; "
                "hepatopathy progresses (ALT/AST elevated, hepatomegaly); "
                "fasting hypoglycaemia episodes; SNHL confirmed on ABR (40%). "
                "Genetic panel returns SUCLG1 biallelic variants."
            ),
        },
        {
            "phase": "Infantile (3-12 months)",
            "events": (
                "Encephalomyopathic progression; seizure onset (myoclonic/focal); "
                "dystonia emerges in moderate cases; respiratory compromise in severe; "
                "NIV (non-invasive ventilation) consideration; "
                "enteral feeds via PEG; MMA monitoring and emergency letter activated; "
                "severe null-variant cases: palliative care discussions."
            ),
        },
        {
            "phase": "Early Childhood (1-5 years, surviving moderate cases)",
            "events": (
                "Stable disability in moderate missense cases; "
                "SNHL management (hearing aids / cochlear implant evaluation); "
                "dystonia management; developmental delay; "
                "metabolic decompensation with febrile illness — emergency protocols critical; "
                "hepatopathy may improve or stabilise with nutritional management."
            ),
        },
        {
            "phase": "Later Childhood / Adolescence (moderate cases only)",
            "events": (
                "Minority of patients with splice-site/missense variants survive to school age; "
                "ongoing supportive care; respiratory and cardiac surveillance; "
                "rare patients with partial SCS activity approach adolescence; "
                "palliative care for progressive deteriorators."
            ),
        },
    ]

    mma_severity_comparison = {
        "suclg1_mma_umol_mmol_creat_typical": "500-3000 (SEVERE) — both SCS-A and SCS-G lost",
        "sucla2_mma_umol_mmol_creat_typical": "10-100 (MILD) — SCS-A only; SCS-G preserved",
        "mut_mma_umol_mmol_creat_typical": ">500-5000 (SEVERE) — methylmalonyl-CoA mutase absent",
        "mmachc_mma_umol_mmol_creat_typical": "100-1000 (moderate-severe) + homocystinuria",
        "suclg1_plasma_hcy": "Normal — NO homocystinuria (DDx MMACHC/cblC)",
        "suclg1_ketoacidosis_risk": "MODERATE-HIGH — severe MMA can precipitate ketoacidosis (unlike SUCLA2 mild)",
        "sucla2_ketoacidosis_risk": "LOW — MMA mild; metabolic bystander effect only",
        "suclg1_hepatopathy": "YES ~70% — SCS-G → PEPCK → gluconeogenesis failure",
        "sucla2_hepatopathy": "NO — SCS-A only; SCS-G (hepatic GTP) preserved",
        "c4dc_succinylcarnitine": "ELEVATED in BOTH SUCLG1 and SUCLA2 — pathognomonic SCS axis marker",
        "note": (
            "If MMA >500 µmol/mmol creat + C4-DC elevated + hepatopathy: SUCLG1 is leading diagnosis. "
            "If MMA 10-100 + C4-DC elevated + no hepatopathy: SUCLA2 is leading diagnosis."
        ),
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
        "mma_severity_comparison": mma_severity_comparison,
        "patients_sample": patients[:8],
    }


def get_definitions() -> dict:
    """SUCLG1 MDDS9 — definitions for /api/suclg1/definitions."""
    return {
        "generated": date.today().isoformat(),
        "terms": [
            {
                "term": "SUCLG1 — Succinyl-CoA Ligase GDP-Forming Alpha Subunit",
                "definition": (
                    "SUCLG1 (394 amino acids precursor, 362 aa mature; 2p11.2) encodes the shared alpha "
                    "subunit of BOTH mitochondrial succinyl-CoA synthetase (SCS) isoforms: "
                    "SCS-A (ADP-forming: SUCLG1+SUCLA2 heterodimer) and "
                    "SCS-G (GTP-forming: SUCLG1+SUCLG2 heterodimer). "
                    "Both isoforms catalyse: Succinyl-CoA + ADP/GDP + Pi → Succinate + CoA-SH + ATP/GTP. "
                    "Because SUCLG1 is shared, its loss abolishes BOTH SCS-A and SCS-G — the critical "
                    "distinction from SUCLA2 (which only loses SCS-A). SUCLG1 disease is therefore "
                    "typically more severe than SUCLA2 disease, with additional hepatopathy from SCS-G loss."
                ),
            },
            {
                "term": "MDDS9 — Mitochondrial DNA Depletion Syndrome 9 (Encephalomyopathic + Severe MMA)",
                "definition": (
                    "MDDS9 (OMIM #612235) is an autosomal recessive disease caused by biallelic "
                    "loss-of-function mutations in SUCLG1. Clinical features: neonatal/early infantile "
                    "profound hypotonia (100%), SEVERE methylmalonic aciduria (MMA >500 µmol/mmol creat, 100%), "
                    "elevated succinylcarnitine C4-DC (95%), lactic acidosis (100%), hepatopathy (70%), "
                    "fasting hypoglycaemia (55%), Leigh-like MRI (60%), seizures (65%), dystonia (50%), "
                    "SNHL (40%). More severe than SUCLA2 (MDDS10); higher neonatal mortality. "
                    "KEY DDx: MMA SEVERE (vs SUCLA2 mild) + hepatopathy (vs SUCLA2 none)."
                ),
            },
            {
                "term": "SCS-G vs SCS-A — The Two SCS Isoforms and SUCLG1's Unique Role",
                "definition": (
                    "Succinyl-CoA synthetase exists as two isoforms sharing the SUCLG1 alpha subunit: "
                    "SCS-A (ADP-forming): SUCLG1 + SUCLA2 → regenerates ADP; co-localises with NDPK "
                    "(NME4) at inner mitochondrial membrane to channel dNTPs for POLG/mtDNA replication. "
                    "SCS-G (GTP-forming): SUCLG1 + SUCLG2 → regenerates GTP; critical for hepatic "
                    "gluconeogenesis (PEPCK requires GTP as substrate). "
                    "SUCLG1 mutations: BOTH SCS-A and SCS-G deficient. "
                    "SUCLA2 mutations: SCS-A deficient only; SCS-G intact (SUCLG2 still pairs with SUCLG1). "
                    "This explains SUCLG1 hepatopathy (SCS-G loss) absent in SUCLA2."
                ),
            },
            {
                "term": "Severe MMA in SUCLG1 — Mechanism and DDx",
                "definition": (
                    "SUCLG1 deficiency → both SCS-A and SCS-G inactive → succinyl-CoA accumulates → "
                    "methylmalonyl-CoA accumulates upstream (via reverse methylmalonyl-CoA mutase "
                    "equilibrium) → methylmalonic acid (MMA) shunted to urine. "
                    "SEVERE because both SCS isoforms contribute to succinyl-CoA flux and both are lost. "
                    "SUCLA2 MMA is MILD because only SCS-A is lost; SCS-G provides partial pathway relief. "
                    "SUCLG1 MMA can precipitate metabolic ketoacidosis (unlike SUCLA2). "
                    "DDx: MUT (methylmalonyl-CoA mutase) → severe MMA but NO mtDNA depletion, "
                    "NO C4-DC, NO encephalomyopathy as primary; responds to dietary restriction ± "
                    "B12. SUCLG1 does NOT respond to B12 supplementation."
                ),
            },
            {
                "term": "C4-DC (Succinylcarnitine) — Pathognomonic SCS Axis Marker",
                "definition": (
                    "C4-DC (C4-dicarboxylcarnitine, succinylcarnitine) is the carnitine conjugate of "
                    "succinic acid, detectable on plasma acylcarnitine profile. It is elevated whenever "
                    "succinyl-CoA accumulates — in BOTH SUCLG1 and SUCLA2 deficiency. "
                    "This makes C4-DC elevation a PATHOGNOMONIC marker of the SCS axis. "
                    "When a child presents with encephalomyopathy + MMA + C4-DC elevated, the "
                    "differential is: SUCLG1 (severe MMA + hepatopathy) vs SUCLA2 (mild MMA + no "
                    "hepatopathy). C4-DC alone cannot distinguish the two; MMA severity and hepatopathy "
                    "presence complete the DDx. Isolated C4-DC elevation without MMA may suggest "
                    "malonate semialdehyde dehydrogenase deficiency (ALDH6A1) — separate entity."
                ),
            },
            {
                "term": "PEPCK and Hepatic Gluconeogenesis Failure in SUCLG1",
                "definition": (
                    "Phosphoenolpyruvate carboxykinase (PEPCK, encoded by PCK1/PCK2) catalyses the "
                    "conversion of oxaloacetate + GTP → phosphoenolpyruvate + CO2 + GDP — the "
                    "rate-limiting and committed step of hepatic gluconeogenesis. PEPCK uniquely requires "
                    "GTP (not ATP) as its phosphate donor. In SUCLG1 disease, SCS-G (SUCLG1+SUCLG2) is "
                    "deficient → hepatic GTP production is impaired → PEPCK activity ↓ → gluconeogenesis "
                    "fails → fasting hypoglycaemia. This mechanism is specific to SUCLG1; SUCLA2 disease "
                    "preserves SCS-G (SUCLG2 can still pair with remaining SUCLG1) → hepatic GTP normal "
                    "→ NO fasting hypoglycaemia in SUCLA2. Prevention: continuous enteral nutrition; "
                    "emergency IV dextrose GIR 8-10 mg/kg/min during illness."
                ),
            },
            {
                "term": "VPA Absolute Contraindication in SUCLG1",
                "definition": (
                    "Valproic acid (VPA) is ABSOLUTELY CONTRAINDICATED in SUCLG1 disease for three "
                    "synergistic mechanisms: (1) POLG inhibition — VPA directly inhibits the mtDNA "
                    "polymerase gamma, worsening pre-existing mtDNA depletion; (2) CoA sequestration — "
                    "valproyl-CoA ester traps mitochondrial CoA, reducing the coenzyme pool essential "
                    "for TCA cycle and beta-oxidation — critical when SUCLG1 disease already reduces "
                    "CoA regeneration from succinyl-CoA; (3) Hepatotoxicity — VPA-mediated hepatocellular "
                    "injury is potentiated in a liver already compromised by SUCLG1 hepatopathy and "
                    "SCS-G deficiency. The combination is potentially rapidly fatal. This class "
                    "prohibition applies to ALL mtDNA depletion syndromes (POLG, DGUOK, MPV17, TK2, "
                    "TWNK, SUCLA2, RRM2B). Always prescribe LEV as the first-choice AED."
                ),
            },
            {
                "term": "Ketogenic Diet Contraindication in SUCLG1",
                "definition": (
                    "The ketogenic diet (KD) is CONTRAINDICATED in SUCLG1 disease. KD forces the brain "
                    "and other tissues to rely on ketone bodies and fatty acid beta-oxidation for energy. "
                    "Beta-oxidation feeds acetyl-CoA and reduced cofactors (NADH, FADH2) directly into "
                    "the TCA cycle and OXPHOS. In SUCLG1 disease, mtDNA depletion impairs OXPHOS "
                    "capacity → the increased OXPHOS demand from KD cannot be met → metabolic crisis, "
                    "worsened lactic acidosis, energy collapse. Additionally, KD exacerbates fasting-like "
                    "states, triggering the SCS-G/PEPCK-mediated hypoglycaemia. KD is beneficial in "
                    "epileptic encephalopathies with intact OXPHOS (e.g., SCN1A Dravet, GLUT1 deficiency) "
                    "but is harmful in all mtDNA depletion syndromes."
                ),
            },
        ],
    }
