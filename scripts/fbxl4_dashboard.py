#!/usr/bin/env python3
"""FBXL4 Encephalomyopathic mtDNA Depletion Syndrome Dashboard.

Mitochondrial DNA Depletion Syndrome 13 (MDDS13) = OMIM #615471
Also known as: FBXL4-Related Encephalomyopathic mtDNA Depletion Syndrome

FBXL4 (F-box and leucine-rich repeat protein 4; 891 aa precursor / ~839 aa mature; 6q16.1-q16.2)
encodes a mitochondrially-targeted F-box protein that acts as a substrate adaptor within an
SCF-like (Skp1-Cullin-F-box) E3 ubiquitin ligase complex in the mitochondrial matrix. FBXL4
is essential for:
  1. Maintaining mtDNA copy number by balancing mitochondrial biogenesis vs. mitophagy
  2. Stabilizing mitochondrial network via regulation of fusion/fission dynamics
  3. Preventing excessive mitochondrial protein degradation

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — mtDNA depletion disease; same mechanism as
     POLG/DGUOK/MPV17/TK2/TWNK/SUCLA2/SUCLG1 (CoA sequestration + mtDNA depletion aggravation)
  2. KD = CONTRAINDICATED — OXPHOS-dependent beta-oxidation fails in mtDNA depletion
  3. NO MMA — KEY DDx from SUCLA2 (mild MMA) and SUCLG1 (severe MMA);
     FBXL4 does NOT affect succinyl-CoA pathway; urine organic acids: MMA normal
  4. NO C4-DC (succinylcarnitine) elevation — KEY DDx from SUCLA2 and SUCLG1 (SCS axis)
  5. MULTIPLE OXPHOS COMPLEX DEFICIENCY — characteristic in FBXL4: Complexes I, II, III, IV
     all depressed on muscle ETC analysis (vs many MDDS with single complex deficiency)
  6. mtDNA COPY NUMBER SEVERELY DEPLETED — cardinal finding; muscle biopsy <20% of normal
     (often 5-15%); also depleted in liver and brain
  7. Hypotonia 100% — universal; neonatal/early infantile; severe from birth
  8. Lactic acidosis 100% — SEVERE; blood lactate >5 mmol/L common; pH <7.1 in neonatal form
  9. Leigh-like MRI ~65% — T2 hyperintensity basal ganglia, brainstem, peri-aqueductal grey
 10. Seizures ~58% — infantile spasms, myoclonic, focal; LEV preferred AED
 11. NO nystagmus — KEY DDx from DGUOK (nystagmus 90% rotary/pendular PATHOGNOMONIC)
 12. NO Fanconi syndrome — KEY DDx from RRM2B (Fanconi 52%)
 13. CK normal or mildly elevated — KEY DDx from TK2 (CK very high, 90%)
 14. Propofol = AVOID (PRIS risk — mitochondrial disease universal rule)
 15. LEV preferred AED — renal excretion, no hepatic metabolism, no CoA interaction
 16. NO disease-modifying therapy proven (unlike TK2 deoxynucleoside rescue)

FBXL4 BIOLOGY:
FBXL4 (891 amino acids precursor including ~52 aa MTS; ~839 aa mature; 6q16.1-q16.2)
encodes the first mitochondrially-localized F-box protein identified. F-box proteins
are substrate-recognition subunits of SCF (Skp1-Cullin-F-box) E3 ubiquitin ligase complexes
that tag proteins for proteasomal degradation.

FBXL4 protein domains:
  Mitochondrial targeting sequence (MTS): aa 1-~52 (cleaved post-import into matrix)
  F-box domain: aa ~80-130 (Skp1 binding interface; substrate adaptor function)
  Leucine-Rich Repeat (LRR) domain: aa ~150-850 (13-16 LRRs; substrate-recognition module)
  C-terminal cap: aa ~850-839 (LRR capping; structural stability)

FBXL4 mitochondrial function:
The FBXL4-SCF-like complex within the mitochondrial matrix ubiquitinates target proteins
involved in mitophagy regulation (likely BNIP3L/NIX pathway components and DRP1-adaptor
proteins), preventing hyperactivation of mitochondrial degradation. Without FBXL4:

  1. Excessive mitophagy → mitochondrial mass ↓ → mtDNA copy number ↓↓
     (muscle: typically <20% normal; brain: 20-40% normal; liver: variable)
  2. Mitochondrial network fragmentation → impaired mitochondrial fusion cascade
     (OPA1/MFN1/MFN2 regulatory imbalance)
  3. OXPHOS assembly impaired → all ETC complexes (I, II, III, IV) + Complex V depressed
     → severe OXPHOS failure → lactic acidosis from impaired oxidative phosphorylation
  4. mtDNA depletion → further OXPHOS failure (mtDNA encodes 13 OXPHOS subunits)

This distinguishes FBXL4 from nucleotide pool MDDS (TK2, DGUOK, RRM2B, SUCLA2, SUCLG1)
where mtDNA depletion occurs via dNTP pool insufficiency during replication.
In FBXL4, mtDNA depletion results from mitophagy excess and mtDNA turnover dysregulation.

MULTI-COMPLEX OXPHOS DEFICIENCY:
ETC enzyme analysis in FBXL4 muscle biopsy characteristically shows:
  Complex I (NADH dehydrogenase): reduced (often <30% of normal; largely mtDNA-encoded)
  Complex II (succinate dehydrogenase): reduced-to-normal (fully nDNA-encoded; mild effect)
  Complex III (cytochrome bc1): reduced (mtDNA-encoded cytochrome b)
  Complex IV (cytochrome c oxidase): reduced (mtDNA-encoded COX subunits)
  Complex V (ATP synthase): reduced (mtDNA-encoded F0 subunits)
This pan-OXPHOS deficiency pattern differs from single-complex deficiencies (e.g., NDUFV1:
only CI) and confirms mtDNA template insufficiency rather than a single assembly factor defect.

GENOTYPE-PHENOTYPE CORRELATION:
  Biallelic null (nonsense/frameshift) → catastrophic neonatal; complete loss of FBXL4;
    extreme mtDNA depletion (muscle <10% normal); lactic acidosis pH <7.1; fatal within
    days to months without intensive support
  F-box domain missense → severe; impairs Skp1 binding and SCF complex assembly;
    encephalomyopathic + multi-complex OXPHOS deficiency; median survival 2-4 years
  LRR domain missense (homozygous) → moderate-severe; substrate recognition partially
    maintained; some residual FBXL4 activity; survival into mid-childhood possible
  Compound het (splice-site + missense) → variable; partial splice rescue may attenuate;
    OXPHOS deficiency often less severe; some patients survive to adolescence

KEY DDx PEARLS:
  vs SUCLA2 (MDDS10): SUCLA2 → MMA MILD (10-100 µmol/mmol creat) + SNHL + Dystonia +
    SUCLA2-NDPK axis; FBXL4 → NO MMA; NO C4-DC; multi-complex OXPHOS deficiency
  vs SUCLG1 (MDDS9): SUCLG1 → MMA SEVERE (500-3000) + hepatopathy 70% + C4-DC elevated;
    FBXL4 → NO MMA; NO C4-DC; NO hepatopathy (mild LFT elevation only)
  vs RRM2B (MDDS8A): RRM2B → Fanconi syndrome 52% (renal tubular acidosis, phosphaturia);
    FBXL4 → NO Fanconi; renal tubular function normal
  vs DGUOK (MDDS3): DGUOK → nystagmus 90% PATHOGNOMONIC; hepatocerebral ± hepatic-only;
    FBXL4 → NO nystagmus; global encephalomyopathic (brain + muscle >> liver)
  vs MPV17 (MDDS6): MPV17 → peripheral neuropathy 80%; hepatocerebral dominant;
    Navajo founder; FBXL4 → NO peripheral neuropathy; encephalomyopathic dominant
  vs TK2 (MDDS4A): TK2 → myopathic only (NO brain involvement); CK very high 90%;
    deoxynucleoside therapy; FBXL4 → encephalomyopathic; CK normal/mildly elevated
  vs POLG (Alpers): POLG → EPC 60% (epilepsia partialis continua); hepatopathy 80%;
    European founders (Ala467Thr/Trp748Ser); FBXL4 → NO EPC; hepatopathy rare;
    consanguineous (Turkish/Saudi/Egyptian) rather than Northern European
  FBXL4 FINGERPRINT: severe hypotonia + profound lactic acidosis + multi-complex OXPHOS
    deficiency + mtDNA depletion (muscle <20%) + NO MMA + NO C4-DC + NO nystagmus +
    NO Fanconi + encephalomyopathic → FBXL4 panel first

References:
  Gai X et al. 2013 Am J Hum Genet — first description; exome; FBXL4 mutations MDDS13
  Bonnen PE et al. 2013 Am J Hum Genet — independent discovery; FBXL4 role in MDDS
  Antoun G et al. 2013 J Med Genet — clinical series; multi-complex OXPHOS deficiency
  Huemer M et al. 2015 J Child Neurol — genotype-phenotype; LRR vs F-box severity
  Barøy T et al. 2016 Eur J Hum Genet — Scandinavian cohort; founder variants; natural history
  Sabouny R & Shutt TE 2020 Trends Endocrinol Metab — FBXL4 mechanism; mitophagy regulation
"""

import random
from datetime import date

SEED = 563  # 40-patient cohort seed


def get_overview() -> dict:
    """FBXL4 MDDS13 — top-level overview for /api/fbxl4/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "FBXL4 Encephalomyopathic mtDNA Depletion Syndrome (MDDS13)",
        "gene": "FBXL4",
        "protein": "F-box and Leucine-Rich Repeat Protein 4 (Mitochondrial Matrix SCF-Adaptor)",
        "protein_size_aa": 891,
        "mature_protein_aa": 839,
        "locus": "6q16.1-q16.2",
        "inheritance": "Autosomal Recessive (AR)",
        "omim_gene": "605654",
        "omim_disease": "615471",
        "mechanism": (
            "Biallelic FBXL4 loss-of-function → loss of mitochondrial SCF-like E3 ubiquitin adaptor "
            "→ excessive mitophagy + impaired mitochondrial network dynamics → mtDNA copy number "
            "severely depleted in all tissues (muscle <20% normal, brain 20-40%, liver variable) → "
            "all OXPHOS complexes (I, II, III, IV, V) depressed due to mtDNA template insufficiency "
            "→ profound lactic acidosis + encephalomyopathy. Mechanism differs from nucleotide-pool "
            "MDDS (TK2/DGUOK/SUCLA2): FBXL4 depletion results from mitophagy dysregulation, not "
            "dNTP pool deficiency. No MMA, no C4-DC — SCS axis intact."
        ),
        "key_contraindications": [
            {
                "drug": "Valproic Acid (VPA)",
                "level": "ABSOLUTE CONTRAINDICATION",
                "mechanism": (
                    "VPA inhibits POLG (mtDNA polymerase gamma) directly; CoA sequestration by "
                    "valproyl-CoA reduces mitochondrial CoA availability; epoxide metabolites cause "
                    "hepatotoxicity. In FBXL4 disease, pre-existing severe mtDNA depletion and OXPHOS "
                    "failure make any further mitochondrial insult potentially lethal. "
                    "Same absolute CI as all other MDDS (POLG/DGUOK/MPV17/TK2/TWNK/SUCLA2/SUCLG1)."
                ),
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "CONTRAINDICATED",
                "mechanism": (
                    "KD forces reliance on fatty acid beta-oxidation and OXPHOS for ATP production. "
                    "In FBXL4 disease, all OXPHOS complexes are depressed due to mtDNA depletion "
                    "→ cells cannot sustain the metabolic demand of KD → metabolic crisis, worsening "
                    "lactic acidosis, energy collapse. Contraindicated in all mtDNA depletion syndromes. "
                    "Note: KD is used in channelopathies (SCN1A/KCNQ2) but NOT in OXPHOS diseases."
                ),
            },
            {
                "drug": "Propofol",
                "level": "AVOID — PRIS Risk",
                "mechanism": (
                    "Propofol infusion syndrome (PRIS) risk universally elevated in mitochondrial disease. "
                    "Propofol inhibits Complex I and Complex II of the ETC; in FBXL4 (pre-existing "
                    "pan-OXPHOS deficiency from mtDNA depletion), additional ETC inhibition can "
                    "precipitate PRIS: metabolic acidosis, rhabdomyolysis, cardiac failure. "
                    "Use alternatives: sevoflurane (preferred), ketamine (low doses), dexmedetomidine."
                ),
            },
        ],
        "key_features": {
            "hypotonia": {"pct": 100, "note": "Universal; neonatal/early infantile onset; profound from birth; ventilator-dependent in severe forms"},
            "lactic_acidosis": {"pct": 100, "note": "SEVERE — blood lactate >5 mmol/L; pH <7.1 in neonatal; reflects pan-OXPHOS failure from mtDNA depletion"},
            "psychomotor_regression": {"pct": 95, "note": "Progressive encephalopathy; developmental arrest then regression; loss of milestones"},
            "multi_complex_oxphos_deficiency": {"pct": 80, "note": "Complexes I+III+IV (all mtDNA-encoded) + V depressed; pan-OXPHOS deficiency — KEY DDx from single-complex defects"},
            "leigh_mri": {"pct": 65, "note": "T2 hyperintensity basal ganglia, brainstem, peri-aqueductal grey; bilateral symmetric"},
            "seizures": {"pct": 58, "note": "Infantile spasms, myoclonic, focal; LEV preferred (renal excretion); VPA ABSOLUTE CI"},
            "feeding_difficulty": {"pct": 90, "note": "NG tube early; PEG for long-term; high complex-carbohydrate diet"},
            "growth_retardation": {"pct": 85, "note": "Failure to thrive; weight and length below 3rd centile; relates to metabolic inefficiency"},
            "mtdna_copy_depletion": {"pct": 100, "note": "Cardinal: muscle mtDNA <20% normal; often 5-15%; confirmed by Southern blot or qPCR on muscle biopsy"},
        },
        "no_mma_no_c4dc": (
            "FBXL4 disease does NOT elevate methylmalonic acid (MMA) and does NOT elevate "
            "C4-DC (succinylcarnitine) on acylcarnitine profile. The succinyl-CoA/SCS axis is "
            "fully intact — FBXL4 acts via mitophagy/ubiquitin pathway, not nucleotide pool or "
            "TCA-cycle. KEY DDx: if child has encephalomyopathy + profound lactic acidosis "
            "but MMA is NORMAL and acylcarnitines are NORMAL → FBXL4 panel; mtDNA copy number "
            "in muscle and multi-complex OXPHOS enzyme analysis."
        ),
        "first_author": "Gai X / Bonnen PE",
        "first_publication_year": 2013,
        "first_journal": "Am J Hum Genet (dual independent discovery)",
        "cohort_n": 40,
        "seed": SEED,
    }


def get_breakdown() -> dict:
    """FBXL4 MDDS13 — 40-patient cohort breakdown for /api/fbxl4/breakdown."""
    rng = random.Random(SEED)

    genotypes = [
        {
            "variant_class": "Biallelic null (nonsense/frameshift)",
            "n": 10,
            "pct": 25,
            "note": (
                "Complete FBXL4 protein loss; catastrophic neonatal; extreme mtDNA depletion "
                "(muscle <10% normal); lactic acidosis pH <7.1; pan-OXPHOS failure; "
                "often fatal within weeks to months; palliative care typically initiated."
            ),
        },
        {
            "variant_class": "F-box domain missense (homozygous or compound het)",
            "n": 14,
            "pct": 35,
            "note": (
                "F-box domain impairs Skp1-FBXL4 interaction → SCF-like complex assembly fails; "
                "near-complete loss of function; severe encephalomyopathic; multi-complex OXPHOS "
                "deficiency; mtDNA muscle 8-18% normal; median survival 2-5 years."
            ),
        },
        {
            "variant_class": "LRR domain missense (homozygous founder variant)",
            "n": 11,
            "pct": 27,
            "note": (
                "Leucine-rich repeat domain; partial substrate-recognition maintained; "
                "moderate-severe encephalomyopathic; OXPHOS deficiency less extreme; "
                "mtDNA muscle 15-25% normal; survival into mid-childhood; common in Turkish, "
                "Saudi, and Egyptian consanguineous populations."
            ),
        },
        {
            "variant_class": "Compound het (splice-site + missense)",
            "n": 5,
            "pct": 13,
            "note": (
                "Splice-site allele may have partial rescue; variable phenotype; "
                "some patients survive to adolescence; OXPHOS deficiency may fluctuate; "
                "lactic acidosis can be episodic rather than constant."
            ),
        },
    ]

    phenotypes_dist = [
        ("Severe neonatal (biallelic null / F-box domain)", 14),
        ("Encephalomyopathic + multi-complex OXPHOS deficiency (F-box/LRR)", 18),
        ("Moderate encephalomyopathic (LRR missense / splice compound het)", 8),
    ]

    sex_choices = ["M", "F"]
    ethnic_groups = [
        "Turkish (consanguineous)",
        "Saudi Arabian (consanguineous)",
        "Egyptian (consanguineous)",
        "Lebanese (consanguineous)",
        "Pakistani (consanguineous)",
        "Italian",
        "French",
        "German",
        "Norwegian",
        "Spanish",
        "Indian (South Asian)",
        "Moroccan",
        "Palestinian",
    ]
    variant_pairs = [
        ("p.Arg158*/p.Gln312*", "Biallelic nonsense"),
        ("p.Cys534Arg/p.Arg516Cys", "F-box domain missense compound het"),
        ("p.Glu95Lys/p.Thr109Met", "F-box domain missense compound het"),
        ("p.Gly748Arg/p.Gly748Arg", "LRR domain homozygous founder (Turkish)"),
        ("p.Leu781Pro/p.Leu781Pro", "LRR domain homozygous founder (Saudi)"),
        ("p.Arg512Gln/p.Arg512Gln", "LRR domain homozygous (Egyptian)"),
        ("c.1303+2T>A/p.Arg512His", "Splice-site/missense compound het"),
        ("p.Trp104*/p.Arg200Cys", "Nonsense/missense compound het"),
    ]

    patients = []
    pid = 1
    for phenotype_name, count in phenotypes_dist:
        for _ in range(count):
            sex = rng.choice(sex_choices)
            ethnicity = rng.choice(ethnic_groups)
            variant_pair = rng.choice(variant_pairs)
            onset_mo = rng.randint(0, 3)  # early onset
            diag_mo = onset_mo + rng.randint(3, 20)
            hypotonia = True  # 100%
            lactic_acidosis = True  # 100%
            psychomotor_regression = rng.random() < 0.95
            leigh_mri = rng.random() < 0.65
            seizures = rng.random() < 0.58
            feeding_difficulty = rng.random() < 0.90
            growth_retardation = rng.random() < 0.85
            multi_oxphos = rng.random() < 0.80
            mtdna_pct = rng.randint(5, 22)  # severely depleted
            lactate = round(rng.uniform(4.5, 14.0), 1)
            ph = round(rng.uniform(6.95, 7.25), 2)
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
                "lactic_acidosis": lactic_acidosis,
                "peak_lactate_mmol": lactate,
                "blood_ph_nadir": ph,
                "psychomotor_regression": psychomotor_regression,
                "leigh_mri": leigh_mri,
                "seizures": seizures,
                "feeding_difficulty": feeding_difficulty,
                "growth_retardation": growth_retardation,
                "multi_oxphos_deficiency": multi_oxphos,
                "mtdna_copy_pct_muscle": mtdna_pct,
                "mma_normal": True,  # KEY DDx
                "c4dc_normal": True,  # KEY DDx
            })
            pid += 1

    total = len(patients)
    feature_prevalence = [
        {
            "feature": "Hypotonia",
            "pct": 100,
            "note": "Universal; neonatal/early infantile; severe; ventilator-dependent in null-variant cases",
        },
        {
            "feature": "Lactic Acidosis (Severe)",
            "pct": 100,
            "note": (
                "Blood lactate >5 mmol/L characteristic; pH <7.1 in severe neonatal. "
                "Reflects pan-OXPHOS failure (all ETC complexes depressed due to mtDNA depletion). "
                "Most severe in biallelic null (lactate often >8 mmol/L)."
            ),
        },
        {
            "feature": "Psychomotor Regression",
            "pct": round(sum(1 for p in patients if p["psychomotor_regression"]) / total * 100),
            "note": (
                "Progressive encephalopathy; developmental arrest followed by regression. "
                "Language acquisition severely impaired; motor regression with spasticity. "
                "Reflects ongoing neuronal energy failure from mtDNA depletion."
            ),
        },
        {
            "feature": "Feeding Difficulty",
            "pct": round(sum(1 for p in patients if p["feeding_difficulty"]) / total * 100),
            "note": "NG tube in early infantile period; PEG for long-term; ensure continuous feeds; NO prolonged fasting",
        },
        {
            "feature": "Growth Retardation",
            "pct": round(sum(1 for p in patients if p["growth_retardation"]) / total * 100),
            "note": "Weight and length below 3rd centile; metabolic inefficiency + poor caloric intake from feeding difficulty",
        },
        {
            "feature": "Multi-Complex OXPHOS Deficiency",
            "pct": round(sum(1 for p in patients if p["multi_oxphos_deficiency"]) / total * 100),
            "note": (
                "ETC enzyme analysis (muscle biopsy): Complexes I, III, IV (all mtDNA-encoded) "
                "depressed; Complex V also reduced. Complex II (fully nDNA-encoded) may be "
                "relatively preserved. Pan-OXPHOS pattern confirms mtDNA template insufficiency. "
                "KEY DDx from single-complex deficiencies (CI-only: NDUFV1; CIV-only: SCO2/COX10)."
            ),
        },
        {
            "feature": "Leigh-like MRI",
            "pct": round(sum(1 for p in patients if p["leigh_mri"]) / total * 100),
            "note": (
                "T2 hyperintensity in basal ganglia (putamen, caudate), brainstem, "
                "peri-aqueductal grey; bilateral symmetric. Similar pattern to other MDDS "
                "but FBXL4 may show more diffuse involvement including cortex in severe forms."
            ),
        },
        {
            "feature": "Seizures",
            "pct": round(sum(1 for p in patients if p["seizures"]) / total * 100),
            "note": (
                "Infantile spasms (IS) most common in first year; myoclonic and focal seizures. "
                "LEV preferred (renal excretion, no mitochondrial toxicity). "
                "VPA ABSOLUTE CI. ACTH for infantile spasms (VGB second line — avoid VPA). "
                "EEG: hypsarrhythmia in spasm phase; multi-focal discharges."
            ),
        },
        {
            "feature": "mtDNA Copy Number Depleted",
            "pct": 100,
            "note": (
                "Cardinal diagnostic finding: mtDNA <20% of age-matched controls in skeletal muscle "
                "(often 5-15%). Confirmed by qPCR (mtDNA/nDNA ratio) or Southern blot. "
                "Brain and liver also depleted (less accessible). "
                "Distinguishes FBXL4 from primary OXPHOS assembly factor defects (normal mtDNA copy)."
            ),
        },
        {
            "feature": "MMA Normal (KEY DDx)",
            "pct": 100,
            "note": (
                "Urine MMA is NORMAL in FBXL4 — SCS axis is intact. "
                "KEY DDx: if lactic acidosis + encephalomyopathy + MMA normal → FBXL4 "
                "(vs SUCLA2 mild MMA 10-100 µmol/mmol; SUCLG1 severe MMA 500-3000 µmol/mmol). "
                "Order urine organic acids to exclude SCS-axis MDDS."
            ),
        },
    ]

    treatments = [
        {
            "tx": "Emergency Metabolic Protocol (Illness Letter)",
            "level": "A — Mandatory",
            "note": (
                "All FBXL4 patients require an emergency letter for any febrile illness or surgery. "
                "IV dextrose (GIR 8-10 mg/kg/min) to suppress catabolism and provide glucose substrate "
                "for glycolytic (non-OXPHOS) ATP generation; correct acidosis with NaHCO3. "
                "VPA ABSOLUTE CI even in emergency. Propofol AVOID — use sevoflurane/ketamine."
            ),
        },
        {
            "tx": "LEV (Levetiracetam) — AED of Choice",
            "level": "A — Preferred AED",
            "note": (
                "Levetiracetam: renal excretion, no hepatic metabolism, no CoA interaction, "
                "no known ETC inhibition. Safe profile in mitochondrial disease. "
                "Starting dose 10-20 mg/kg/day in 2 doses; titrate to 40-60 mg/kg/day. "
                "For infantile spasms: ACTH first-line; VGB second-line (VPA NEVER). "
                "VPA remains ABSOLUTE CI regardless of seizure type or severity."
            ),
        },
        {
            "tx": "Riboflavin (Vitamin B2)",
            "level": "B — Supportive",
            "note": (
                "FAD cofactor for Complex I and II; case reports of partial lactate reduction "
                "in some MDDS. 10-100 mg/day oral. Low risk/benefit ratio. "
                "Include in standard mitochondrial 'cocktail.' No FBXL4-specific controlled data."
            ),
        },
        {
            "tx": "CoQ10 (Ubiquinol preferred)",
            "level": "C — Supportive",
            "note": (
                "Antioxidant and ETC electron shuttle; 10-30 mg/kg/day oral. "
                "Ubiquinol (reduced form) preferred for absorption. "
                "No controlled FBXL4-specific data; standard supportive mitochondrial therapy."
            ),
        },
        {
            "tx": "Thiamine (Vitamin B1) — Empirical",
            "level": "C — Pre-diagnosis Empirical",
            "note": (
                "100-300 mg/day IV or oral empirically for all Leigh-syndrome presentations "
                "before genetic confirmation (thiamine-responsive PDHC mimics Leigh; very low risk). "
                "Discontinue when FBXL4 confirmed and no clinical response observed."
            ),
        },
        {
            "tx": "Continuous Enteral Nutrition",
            "level": "A — Anti-catabolism",
            "note": (
                "Continuous NG or PEG feeds to prevent fasting; fasting worsens lactic acidosis "
                "in OXPHOS disease by forcing dependence on ETC-coupled beta-oxidation. "
                "High complex-carbohydrate (50-60%), moderate protein (8-10%), low-moderate fat. "
                "KD STRICTLY CONTRAINDICATED. Dietitian involvement mandatory from diagnosis."
            ),
        },
        {
            "tx": "ACTH / Vigabatrin — Infantile Spasms",
            "level": "B — Condition-Specific",
            "note": (
                "If infantile spasms (hypsarrhythmia on EEG): ACTH 40-160 IU/day IM x4 weeks "
                "is first-line (West syndrome protocol). VGB (vigabatrin) second-line. "
                "Pyridoxine trial (100-200 mg/day x3 days) may be given empirically. "
                "VPA is ABSOLUTELY CONTRAINDICATED even for infantile spasms in FBXL4."
            ),
        },
        {
            "tx": "Respiratory Support (NIV / Ventilator)",
            "level": "A — If Respiratory Compromise",
            "note": (
                "Early NIV (BIPAP) for hypercapnic/hypoxic episodes; tracheostomy in severe "
                "progressive cases. Respiratory failure is a leading cause of death in FBXL4. "
                "Palliative care discussions early in biallelic-null severe neonatal presentations."
            ),
        },
        {
            "tx": "Genetic Counselling",
            "level": "A — Mandatory",
            "note": (
                "AR inheritance: 25% recurrence risk. Prenatal diagnosis via CVS (11-13 weeks) "
                "or amniocentesis for known familial FBXL4 variants. Enriched in consanguineous "
                "populations (Turkish, Saudi, Egyptian) — offer cascade testing. "
                "PGT-M (preimplantation genetic testing for monogenic disorders) available."
            ),
        },
        {
            "tx": "Muscle Biopsy + ETC Enzyme Analysis",
            "level": "A — Diagnostic",
            "note": (
                "Fresh-frozen quadriceps/deltoid biopsy for ETC enzyme assay (CI-CIV + CS ratio). "
                "Pan-OXPHOS deficiency expected; mtDNA quantification by qPCR on same biopsy. "
                "Biopsy guides clinical severity and confirms biochemical diagnosis before "
                "genetic sequencing returns."
            ),
        },
    ]

    disease_timeline = [
        {
            "phase": "Neonatal (0-4 weeks)",
            "events": (
                "Profound hypotonia at delivery; neonatal lactic acidosis (pH <7.1 in severe null variant); "
                "feeding failure requiring NG tube; blood lactate >5 mmol/L; "
                "neonatal seizures possible (infantile spasms develop later); "
                "urine organic acids: MMA NORMAL (critical DDx point vs SUCLA2/SUCLG1); "
                "acylcarnitines: C4-DC NORMAL (DDx vs SCS-axis MDDS); "
                "blood NBS: elevated lactate; no specific marker pathognomonic."
            ),
        },
        {
            "phase": "Early Infantile (1-3 months)",
            "events": (
                "Progressive hypotonia and hyporeflexia; failure to thrive; "
                "developmental delay becomes apparent (minimal visual following, poor social smile); "
                "MRI brain: early Leigh-like changes — T2 hyperintensity basal ganglia, brainstem; "
                "muscle biopsy: mtDNA copy <20% normal; pan-OXPHOS enzyme deficiency confirmed; "
                "genetic panel: FBXL4 biallelic variants identified; "
                "infantile spasms may emerge (EEG: hypsarrhythmia)."
            ),
        },
        {
            "phase": "Infantile (3-12 months)",
            "events": (
                "Encephalomyopathic progression; psychomotor regression; "
                "respiratory compromise in severe forms (NIV initiation); "
                "PEG tube for long-term enteral nutrition; "
                "seizure management (ACTH for spasms, LEV ongoing); "
                "growth failure below 3rd centile; recurrent metabolic decompensation with illness; "
                "emergency metabolic letters activated; palliation discussions for null-variant cases."
            ),
        },
        {
            "phase": "Early Childhood (1-4 years, moderate cases)",
            "events": (
                "Stable disability in moderate LRR-missense cases; "
                "limited motor development (sitting possible; standing/walking rarely achieved); "
                "communication severely impaired; "
                "seizure burden may decrease with age in some; "
                "metabolic decompensation risk with febrile illness remains high; "
                "respiratory physio and NIV; ophthalmology for optic atrophy screening."
            ),
        },
        {
            "phase": "Mid-Childhood / Adolescence (selected splice/compound het)",
            "events": (
                "Small minority with splice-site rescue or partial LRR residual activity survive; "
                "static disability with encephalopathy; communication via AAC devices; "
                "ongoing mitochondrial monitoring (LFTs, lactate, cardiac echo annually); "
                "palliative care integration for progressive deteriorators."
            ),
        },
    ]

    oxphos_profile = {
        "complex_I_NADH_dehydrogenase": {
            "typical_pct_control": "15-35",
            "note": "CI: mtDNA-encoded ND1-6 subunits; most commonly severely reduced in FBXL4",
        },
        "complex_II_succinate_dehydrogenase": {
            "typical_pct_control": "60-100",
            "note": "CII: fully nDNA-encoded; often relatively preserved — FBXL4 primary effect via mtDNA depletion",
        },
        "complex_III_cytochrome_bc1": {
            "typical_pct_control": "20-40",
            "note": "CIII: mtDNA-encoded cytochrome b; reduced proportionally to mtDNA depletion",
        },
        "complex_IV_cytochrome_c_oxidase": {
            "typical_pct_control": "10-30",
            "note": "CIV: mtDNA-encoded COX I/II/III subunits; often most severely reduced; COX histochemistry negative fibres",
        },
        "complex_V_ATP_synthase": {
            "typical_pct_control": "25-50",
            "note": "CV: mtDNA-encoded F0 subunits (ATP6/ATP8); reduced but often less severely than CIV",
        },
        "citrate_synthase": {
            "typical_result": "Normal or elevated",
            "note": "CS normalised data: used as mitochondrial mass marker; CS elevated if mitochondrial proliferation",
        },
        "pattern_interpretation": (
            "Pan-OXPHOS deficiency (CI + CIII + CIV ± CV reduced; CII relatively preserved) "
            "indicates mtDNA template insufficiency — all mtDNA-encoded subunits reduced while "
            "nDNA-encoded Complex II remains intact. This pattern discriminates FBXL4 from: "
            "(a) single-complex nDNA assembly factor defects (CI-only: NDUFV1; CIII-only: UQCRQ); "
            "(b) normal ETC with only mtDNA copy number as finding. "
            "Note: SUCLA2/SUCLG1 may also show pan-OXPHOS in muscle; DDx by MMA + C4-DC."
        ),
    }

    return {
        "generated": date.today().isoformat(),
        "disease": "FBXL4 MDDS13",
        "cohort_n": total,
        "seed": SEED,
        "patients_sample": patients[:8],
        "genotype_breakdown": genotypes,
        "phenotype_distribution": [
            {"name": n, "n": c, "pct": round(c / total * 100)} for n, c in phenotypes_dist
        ],
        "feature_prevalence": feature_prevalence,
        "treatments": treatments,
        "disease_timeline": disease_timeline,
        "oxphos_profile": oxphos_profile,
        "mma_c4dc_summary": {
            "fbxl4_mma": "NORMAL — SCS axis intact; no succinyl-CoA/methylmalonyl-CoA overflow",
            "fbxl4_c4dc": "NORMAL — no succinylcarnitine elevation; SCS enzymes unaffected",
            "sucla2_mma": "MILD (10-100 µmol/mmol creat) — SCS-A (ADP-forming) disrupted; SCS-G intact",
            "suclg1_mma": "SEVERE (500-3000 µmol/mmol creat) — both SCS-A and SCS-G disrupted",
            "sucla2_c4dc": "ELEVATED — succinylcarnitine pathognomonic for SCS axis",
            "suclg1_c4dc": "ELEVATED — succinylcarnitine pathognomonic for SCS axis",
            "fbxl4_fanconi": "ABSENT — renal tubular function normal (KEY DDx from RRM2B Fanconi 52%)",
            "fbxl4_nystagmus": "ABSENT — no rotary/pendular nystagmus (KEY DDx from DGUOK nystagmus 90%)",
            "key_diagnostic_clue": (
                "Encephalomyopathy + profound lactic acidosis + MMA NORMAL + C4-DC NORMAL + "
                "muscle mtDNA <20% + pan-OXPHOS deficiency → FBXL4 first on panel"
            ),
        },
    }


def get_definitions() -> dict:
    """FBXL4 MDDS13 — clinical definitions for /api/fbxl4/definitions."""
    return {
        "generated": date.today().isoformat(),
        "disease": "FBXL4 MDDS13",
        "terms": [
            {
                "term": "FBXL4 (F-box and Leucine-Rich Repeat Protein 4)",
                "definition": (
                    "FBXL4 encodes a 891 aa mitochondrially-targeted F-box protein localised to the "
                    "mitochondrial matrix. It acts as the substrate-recognition subunit of an SCF-like "
                    "(Skp1-Cullin-F-box) E3 ubiquitin ligase complex inside mitochondria. "
                    "FBXL4 is essential for maintaining mtDNA copy number (by limiting mitophagy) "
                    "and mitochondrial network integrity. Biallelic loss-of-function causes MDDS13 — "
                    "characterised by severe mtDNA depletion and pan-OXPHOS deficiency."
                ),
            },
            {
                "term": "MDDS13 (Mitochondrial DNA Depletion Syndrome 13)",
                "definition": (
                    "OMIM #615471. Autosomal recessive encephalomyopathic mtDNA depletion syndrome "
                    "caused by biallelic FBXL4 mutations. Hallmarks: profound neonatal/infantile "
                    "hypotonia, severe lactic acidosis, mtDNA copy number depleted in muscle (<20%), "
                    "pan-OXPHOS deficiency. NO MMA, NO C4-DC — SCS axis intact. First described 2013 "
                    "(Gai X et al. and Bonnen PE et al., independent simultaneous discovery)."
                ),
            },
            {
                "term": "F-box Domain",
                "definition": (
                    "A ~40 aa protein-protein interaction motif that mediates binding to Skp1, "
                    "which bridges the F-box protein to the Cullin scaffold of SCF E3 ligase complexes. "
                    "In FBXL4, F-box domain missense variants impair Skp1 binding, disrupting "
                    "SCF-like complex assembly in the mitochondrial matrix. F-box domain variants "
                    "generally cause severe disease."
                ),
            },
            {
                "term": "Leucine-Rich Repeat (LRR) Domain",
                "definition": (
                    "A repetitive structural motif (~20-29 aa per repeat; 13-16 repeats in FBXL4) "
                    "forming a curved solenoid structure that mediates substrate recognition. "
                    "In FBXL4, LRR variants impair recognition and ubiquitination of mitophagy "
                    "regulatory targets, leading to excessive mitochondrial degradation. "
                    "LRR variants often cause moderate-severe disease (more residual function vs F-box)."
                ),
            },
            {
                "term": "mtDNA Copy Number Depletion",
                "definition": (
                    "Reduction in mitochondrial DNA molecules per cell/tissue below normal range. "
                    "In FBXL4 MDDS13: skeletal muscle mtDNA typically <20% of age-matched controls "
                    "(normal: ~1000-5000 mtDNA molecules per cell in muscle). "
                    "Measured by quantitative PCR (mtDNA/nDNA ratio) or Southern blot on fresh-frozen "
                    "muscle biopsy. mtDNA depletion impairs synthesis of 13 OXPHOS subunits encoded "
                    "by mitochondrial genome → pan-OXPHOS deficiency."
                ),
            },
            {
                "term": "Pan-OXPHOS Deficiency",
                "definition": (
                    "Deficiency of all (or most) oxidative phosphorylation complexes (I, III, IV, V) "
                    "on ETC enzyme analysis of skeletal muscle. In FBXL4: complexes I, III, IV, V "
                    "are reduced (all contain mtDNA-encoded subunits); Complex II (fully nDNA-encoded) "
                    "may be relatively preserved. This pattern is the biochemical signature of mtDNA "
                    "template insufficiency — distinguished from single-complex defects caused by "
                    "mutations in complex-specific assembly factors."
                ),
            },
            {
                "term": "Mitophagy",
                "definition": (
                    "Selective autophagy of dysfunctional mitochondria via the autophagy pathway. "
                    "In healthy cells, mitophagy clears damaged mitochondria (PINK1/Parkin pathway "
                    "and receptor-mediated: BNIP3L/NIX, FUNDC1). FBXL4 loss leads to dysregulated "
                    "mitophagy (excessive degradation), reducing mitochondrial mass and mtDNA copy "
                    "number. This is distinct from the dNTP pool mechanism of TK2/DGUOK/RRM2B MDDS."
                ),
            },
            {
                "term": "Leigh Syndrome / Leigh-like MRI",
                "definition": (
                    "Leigh syndrome: subacute necrotizing encephalomyelopathy characterised by "
                    "bilateral symmetric T2-hyperintense lesions in basal ganglia, brainstem, "
                    "and/or cerebellum on MRI. Leigh-like: same MRI pattern without meeting full "
                    "clinical Leigh criteria. In FBXL4: Leigh-like MRI in ~65%, involving putamen, "
                    "caudate, brainstem (dorsal pons/midbrain), peri-aqueductal grey. "
                    "Caused by energy failure in high-metabolic-demand grey matter structures."
                ),
            },
            {
                "term": "VPA (Valproic Acid) — Absolute CI in All MDDS",
                "definition": (
                    "Valproic acid (valproate, sodium valproate, divalproex) is absolutely "
                    "contraindicated in ALL mtDNA depletion syndromes including FBXL4 MDDS13. "
                    "Mechanisms: (1) VPA directly inhibits POLG (mtDNA polymerase gamma) — worsening "
                    "mtDNA depletion; (2) valproyl-CoA sequesters mitochondrial CoA → OXPHOS impairment; "
                    "(3) epoxide metabolites cause hepatotoxicity. In pre-existing severe OXPHOS failure "
                    "(FBXL4), even short VPA exposure can precipitate fatal metabolic decompensation."
                ),
            },
            {
                "term": "LEV (Levetiracetam) — Preferred AED in Mitochondrial Disease",
                "definition": (
                    "Levetiracetam (Keppra) is the preferred antiseizure medication in mitochondrial "
                    "disease including FBXL4. Advantages: (1) renal excretion (no hepatic P450); "
                    "(2) no CoA sequestration; (3) no ETC complex inhibition; (4) no POLG inhibition; "
                    "(5) available IV for status epilepticus. Dosing: 10-20 mg/kg/day initial → "
                    "titrate 40-60 mg/kg/day in 2 divided doses. For infantile spasms: ACTH first, "
                    "VGB second (never VPA)."
                ),
            },
            {
                "term": "Emergency Metabolic Protocol (Sick Day Rules)",
                "definition": (
                    "A protocol for FBXL4 patients during febrile illness, surgery, or fasting: "
                    "(1) IV dextrose (GIR 8-10 mg/kg/min) to suppress catabolism and provide "
                    "glycolytic ATP substrate; (2) bicarbonate (NaHCO3) to correct metabolic acidosis "
                    "if pH <7.2; (3) avoid fasting >4-6 hours; (4) hospital admission for vomiting "
                    "preventing oral/enteral feeds; (5) VPA and propofol ABSOLUTELY AVOIDED. "
                    "Emergency letter should be carried by patient/family at all times."
                ),
            },
            {
                "term": "Acylcarnitine Profile — C4-DC (Succinylcarnitine)",
                "definition": (
                    "C4-DC (butanediylcarnitine; MW 289 Da) is the carnitine conjugate of succinic "
                    "acid. It is elevated in SUCLA2 and SUCLG1 (SCS axis defects) as a biomarker "
                    "of succinyl-CoA accumulation. In FBXL4 MDDS13, C4-DC is NORMAL — the "
                    "succinyl-CoA synthetase axis (SCS-A/SCS-G) is fully intact. KEY DDx: "
                    "normal acylcarnitines + normal MMA in a child with encephalomyopathy + "
                    "lactic acidosis → FBXL4 or other non-SCS MDDS (vs SUCLA2/SUCLG1)."
                ),
            },
            {
                "term": "Consanguinity and Founder Variants",
                "definition": (
                    "FBXL4 is enriched in consanguineous populations due to autosomal recessive "
                    "inheritance. Documented founder variants include: LRR domain p.Gly748Arg in "
                    "Turkish families; LRR domain p.Leu781Pro in Saudi families; additional founders "
                    "in Egyptian, Lebanese, and Pakistani populations. When consanguinity is identified "
                    "in a child with severe lactic acidosis and encephalomyopathy, include FBXL4 "
                    "early in the genetic panel alongside other AR mitochondrial disease genes."
                ),
            },
        ],
    }
