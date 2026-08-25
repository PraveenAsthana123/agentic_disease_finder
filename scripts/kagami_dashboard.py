#!/usr/bin/env python3
"""Kagami-Ogata Syndrome (KOS14) Dashboard.

Kagami-Ogata Syndrome is the OPPOSITE of Temple Syndrome at the SAME 14q32.3 locus.
  Principal genes: DLK1 (Delta-Like 1 Homolog, paternally expressed), RTL1 (Retrotransposon-Like 1,
                   paternally expressed), MEG3 (Maternally Expressed Gene 3, maternally expressed)
  Mechanism: MATERNAL LOF of 14q32.3 → EXCESS paternal DLK1 + RTL1, ABSENT maternal MEG3
  Most common cause: Paternal UPD14 (upd(14)pat) OR Maternal deletion 14q32.3 (~40% each)
  Result: DLK1 OVEREXPRESSED (both alleles paternally active) + RTL1 EXCESS + MEG3 ABSENT
  OMIM Disease: #608149 · Genes: DLK1 *176290 · MEG3 *601626 · RTL1 *603899

IMPRINTING MECHANISM — WHY MATERNAL LOF CAUSES KAGAMI-OGATA SYNDROME:
  DLK1 (Delta-Like 1 Homolog, 383 aa, 14q32.2): PATERNALLY expressed only (normally)
    — In upd(14)pat / maternal deletion: TWO paternal chr14 → DLK1 from BOTH → excess DLK1
    — DLK1 is an anti-adipogenic Notch-like transmembrane protein → excess may drive fetal overgrowth
    — DLK1 inhibits the hypothalamic GnRH pulse generator in fetal life
  RTL1 (Retrotransposon-Like 1, 14q32.31): PATERNALLY expressed only
    — RTL1as (antisense): MATERNALLY expressed (silenced in KOS14 = no RTL1as)
    — RTL1 encodes a protease essential for placental vascular integrity
    — RTL1 dosage: normally tight balance of RTL1 (paternal) vs RTL1as (maternal) regulation
    — In KOS14: RTL1 doubled + RTL1as absent → placentomegaly + mesenchymal dysplasia
    — RTL1 excess → placental villous trophoblast dysplasia → Placental Mesenchymal Dysplasia (PMD)
    — Placentomegaly → polyhydramnios → preterm birth → respiratory compromise
  MEG3 (Maternally Expressed Gene 3, 14q32.3): MATERNALLY expressed, lncRNA — ABSENT in KOS14
    — Normally activates p53 tumour suppressor pathway; absent in KOS14 → reduced p53 activity
    — MEG3 absence → hepatoblastoma predisposition (~5% lifetime risk in KOS14)
    — MEG3 absence → hypothalamic programming abnormalities → neonatal GH-IGF axis dysregulation

  NORMAL 14q32.3 imprinting map (key genes):
    Paternally expressed: DLK1, RTL1 (antisense RTL1as = maternally expressed)
    Maternally expressed: MEG3, MEG8, MEG9, MIATS (lncRNAs)
    IG-DMR (Intergenic DMR): methylated paternal, unmethylated maternal
    In KOS14: IG-DMR BIALLELICALLY methylated (both alleles = paternal pattern) = 100% methylation

FOUR GENETIC MECHANISMS (by frequency):
  1. Paternal UPD14 (upd(14)pat) (~40%): two paternal chromosome 14 copies
     — maternal chr14 absent → MEG3 absent (maternally expressed, both silenced paternally)
     — DLK1 from BOTH alleles: 2× excess
     — SNP array: LOH chr14 without copy number change (isodisomy) or partial LOH (heterodisomy)
     — recurrence risk: <1% (meiotic non-disjunction, predominantly paternal)
  2. Maternal deletion 14q32.3 (~40%): deletes MEG3 domain from MATERNAL chromosome
     — paternal chr14 intact → excess DLK1/RTL1 (unopposed) + no maternal MEG3
     — detected by CMA as copy number loss at 14q32.3 on maternal allele
     — parental studies mandatory: if from mother → 50% recurrence; de novo → ~1%
     — critical: maternal inheritance of deletion → KOS14; paternal inheritance → Temple Syndrome
  3. IG-DMR epimutation (maternal allele hypermethylation → paternal pattern): ~10%
     — maternal IG-DMR acquires abnormal methylation → MEG3 silenced on BOTH alleles
     — DLK1 expressed from both; copy number normal; UPD absent; only methylation abnormal
     — SNP array NORMAL → methylation test mandatory
  4. Paternal duplication 14q32.3 (~10%): extra copy of paternal DLK1-RTL1 region
     — increases DLK1 and RTL1 dosage beyond 2× → may cause more severe placental phenotype
     — detected by CMA as copy number GAIN at 14q32.3 of paternal origin

CLINICAL FEATURES — THE OPPOSITE OF TEMPLE SYNDROME:
  PATHOGNOMONIC SIGN:
    Coat-hanger ribs (bell-shaped thorax): ribs are HORIZONTAL or upward-slanting, short
      — creates "inverted bell" or "coat-hanger" appearance on chest X-ray
      — results from abnormal thoracic cage development; present in >95% of KOS14 cases
      — restricts lung expansion → neonatal respiratory failure
      — the sine qua non of Kagami-Ogata diagnosis; absent in Temple Syndrome

  PRENATAL / NEONATAL:
    Macrosomia (LGA at birth): ~60% (OPPOSITE of Temple = SGA)
    Polyhydramnios: ~75% (RTL1 excess → abnormal placental vascular drainage + swallowing difficulty)
    Placentomegaly: >90th percentile placental weight in ~70%
    Placental Mesenchymal Dysplasia (PMD): ~50% — disorganised mesenchymal villi, increased villous size
    Abdominal wall defects: umbilical hernia or omphalocele-like lax abdominal wall in ~40%
    Large tongue (macroglossia): ~50% (muscular hypertrophy)
    Facial dysmorphism: long philtrum, low-set ears, wide mouth, facial puffiness
    Preterm birth: ~60% due to polyhydramnios
    Neonatal respiratory failure: ~85% (coat-hanger thorax + preterm)
    High neonatal mortality: ~30-35% die in neonatal period despite intensive care

  GROWTH / DEVELOPMENT (survivors):
    Short stature in survivors: ~50% (due to prematurity + neonatal complications)
    Bell-shaped thorax persists: >90% of survivors have persistent thoracic deformity
    Macrocephaly: ~30%
    Hepatoblastoma risk: ~5% lifetime — hepatic surveillance mandatory (MEG3 absent = p53 reduced)
    Intellectual disability: moderate-severe in survivors (IQ ~40-70); mild in mildest cases
    Motor delay: severe in most survivors (neonatal ICU sequelae)
    Speech: severely limited; most survivors are non-verbal or minimally verbal

  EPILEPSY (~15-25% in survivors):
    MECHANISM: primarily hypoxic-ischemic encephalopathy (HIE) from neonatal respiratory failure
               — NOT primary epilepsy gene disorder (contrast Angelman with primary UBE3A epilepsy)
    Seizure types: focal onset (most common), infantile spasms (in HIE context), GTCS
    EEG: background abnormalities (suppression-burst in severe cases); focal slowing; may be normal
    Onset: neonatal (with HIE) or infancy (HIE sequelae)
    Treatment: standard AEDs — no absolute AED contraindications in KOS14
    Prognosis: depends on severity of HIE; focal seizures may be drug-responsive
    DRE: ~30% of those with epilepsy (HIE-related epilepsy is often drug-resistant)
    VPA risk: MODERATE (hepatic monitoring overlap with hepatoblastoma surveillance; weight gain rare)

  NO features of Temple Syndrome:
    NO central precocious puberty (CPP)
    NO truncal obesity
    NO significant GnRH dysregulation causing CPP
    NO hyperphagia (contrast PWS)
    NO food-seeking behavior

KEY BIOMARKERS / DIAGNOSIS:
  First test: Methylation analysis at 14q32.3 (IG-DMR and MEG3-DMR)
    — Abnormal KOS14: 100% methylation (both alleles methylated = both have paternal pattern)
    — Normal: ~50% methylation at IG-DMR
    — Sensitivity: >98% for upd(14)pat + maternal deletion + epimutation
    — SNP array alone misses epimutation + UPD (no copy number change)
  Second test: SNP array
    — Paternal UPD: LOH chr14 without CN change (isodisomy = entire chr14 LOH)
    — Maternal deletion: CN loss at 14q32.3
    — Parental studies confirm inheritance (maternal deletion from mother → 50% recurrence)
  DLK1 serum protein: ELEVATED (excess paternal DLK1) — emerging biomarker
  AFP (alpha-fetoprotein): monitor quarterly for hepatoblastoma (elevated AFP = first sign)
  Chest X-ray: coat-hanger ribs (horizontal, shortened) — PATHOGNOMONIC

KEY COMPARISONS:
  KOS14 vs Temple Syndrome: SAME LOCUS (14q32.3) OPPOSITE PARENT OPPOSITE PHENOTYPE
  KOS14 vs BWS: Beckwith-Wiedemann (11p15.5) — macrosomia + omphalocele + macroglossia (similar signs)
    — BWS: IGF2 excess; hepatoblastoma also; Wilms tumour risk (KOS14 does NOT have Wilms risk)
    — BWS: no coat-hanger ribs; chromosome 11 not 14
  KOS14 vs PMD alone: Placental Mesenchymal Dysplasia (PMD) can occur without KOS14
    — Always test for 14q32.3 methylation when PMD is found

TREATMENTS:
  Respiratory: mechanical ventilation Level A; CPAP Level A; surfactant if preterm Level A
  Hepatoblastoma surveillance: AFP q3-6mo + liver ultrasound q6mo → lifetime surveillance Level A
  Standard AEDs for epilepsy: LEV Level B (first-line focal); VGB Level B (infantile spasms)
  VPA: moderate risk — hepatic monitoring overlap; weight monitoring; use if indicated
  NO GnRH analog (no CPP in KOS14 — contrast Temple where GnRH analog is Level A)
  NO GH therapy (not indicated; no GH deficiency)
  Rib distraction surgery / thoracic expansion: experimental Level C in severe survivors

DRUG RISKS:
  VPA: MODERATE risk — hepatic enzyme elevation monitoring overlap with hepatoblastoma surveillance
  No AED is ABSOLUTELY contraindicated in KOS14 (contrast Angelman: CBZ/OXC absolute CI)
  Sedatives: HIGH RISK in acute neonatal period (respiratory compromise)
  Oxygen / prone positioning: mandatory in coat-hanger thorax
"""
import random
from typing import Any, Dict, List

SEED = 293
random.seed(SEED)

# ── Genetic Mechanisms ──────────────────────────────────────────────────────────
MECHANISMS: List[Dict[str, Any]] = [
    {
        "id": "upd14pat",
        "name": "Paternal UPD14 (upd(14)pat)",
        "frequency_pct": 40,
        "first_line_test": "Methylation IG-DMR (100% methylated) → SNP array LOH chr14",
        "cn_change": "None (UPD has no CN change — SNP array alone misses this)",
        "recurrence_risk_pct": 1,
        "notes": "Two paternal chr14; DLK1 × 2; MEG3 absent; RTL1 × 2",
    },
    {
        "id": "maternal_deletion",
        "name": "Maternal deletion 14q32.3",
        "frequency_pct": 40,
        "first_line_test": "CMA — copy number loss at 14q32.3; parental studies mandatory",
        "cn_change": "Loss at 14q32.3 on maternal allele",
        "recurrence_risk_pct": 50,  # if inherited from mother
        "notes": "De novo: ~1% recurrence. Inherited from mother: 50%. PARENT OF ORIGIN CRITICAL.",
    },
    {
        "id": "igdmr_epimutation",
        "name": "IG-DMR Epimutation (maternal hypermethylation)",
        "frequency_pct": 10,
        "first_line_test": "Methylation IG-DMR (100%) — CN normal; SNP array NORMAL",
        "cn_change": "None",
        "recurrence_risk_pct": 5,
        "notes": "Maternal allele acquires paternal methylation pattern; MEG3 silenced on both alleles",
    },
    {
        "id": "paternal_dup",
        "name": "Paternal duplication 14q32.3",
        "frequency_pct": 10,
        "first_line_test": "CMA — copy number GAIN at 14q32.3 on paternal allele",
        "cn_change": "Gain at 14q32.3 on paternal allele",
        "recurrence_risk_pct": 50,  # if from father
        "notes": "Extra DLK1/RTL1 copy; phenotype may be more severe; placental dysplasia",
    },
]

# ── Phenotypic Classes ────────────────────────────────────────────────────────────
PHENOTYPES: List[Dict[str, Any]] = [
    {
        "group": "Severe-Lethal",
        "pct": 30,
        "description": "Coat-hanger ribs + neonatal respiratory failure; die in neonatal period despite ICU",
        "key_features": ["Severe coat-hanger thorax", "Refractory respiratory failure", "Placentomegaly",
                         "Macrosomia", "Polyhydramnios", "Abdominal wall defects"],
        "survival": "Neonatal death (~30%)",
        "epilepsy_pct": 0,  # die before epilepsy onset
    },
    {
        "group": "Severe-Surviving",
        "pct": 50,
        "description": "Coat-hanger ribs + prolonged respiratory support; survive with significant morbidity",
        "key_features": ["Persistent bell-shaped thorax", "Prolonged ventilation", "Moderate-severe IDD",
                         "HIE risk", "Epilepsy risk 20-30%", "Hepatoblastoma surveillance"],
        "survival": "Survive into childhood/adulthood with support",
        "epilepsy_pct": 22,
    },
    {
        "group": "Moderate",
        "pct": 20,
        "description": "Milder thoracic dysplasia; respiratory distress without prolonged ventilation; better outcome",
        "key_features": ["Mild-moderate thoracic changes", "Mild IDD", "Near-normal ambulation",
                         "Hepatoblastoma risk still present"],
        "survival": "Survive; ambulatory; moderate disability",
        "epilepsy_pct": 10,
    },
]

# ── Known Variants / Molecular Subtypes ──────────────────────────────────────────
VARIANTS: List[Dict[str, Any]] = [
    {
        "variant": "upd(14)pat Isodisomy",
        "gene": "Whole chr14 paternal origin",
        "frequency_pct": 22,
        "phenotype_class": "Severe-Surviving",
        "mechanism_detail": "LOH entire chr14; risk of AR disease homozygosity",
        "dlk1_level": "2×",
        "rtl1_level": "2×",
        "meg3_level": "0",
    },
    {
        "variant": "upd(14)pat Heterodisomy",
        "gene": "Both paternal chr14 homologs",
        "frequency_pct": 18,
        "phenotype_class": "Moderate–Severe-Surviving",
        "mechanism_detail": "Partial LOH; no AR homozygosity risk; phenotype slightly milder",
        "dlk1_level": "2×",
        "rtl1_level": "2×",
        "meg3_level": "0",
    },
    {
        "variant": "Maternal del 14q32.3 de novo",
        "gene": "MEG3/IG-DMR region deleted (maternal)",
        "frequency_pct": 28,
        "phenotype_class": "Severe-Surviving",
        "mechanism_detail": "De novo deletion; recurrence <1%",
        "dlk1_level": "1.5–2×",
        "rtl1_level": "1.5–2×",
        "meg3_level": "0",
    },
    {
        "variant": "Maternal del 14q32.3 inherited",
        "gene": "MEG3/IG-DMR region deleted (maternal, from mother)",
        "frequency_pct": 12,
        "phenotype_class": "Severe-Surviving",
        "mechanism_detail": "From mother; 50% recurrence; carrier mother = NO phenotype (normal maternal silencing)",
        "dlk1_level": "1.5–2×",
        "rtl1_level": "1.5–2×",
        "meg3_level": "0",
    },
    {
        "variant": "IG-DMR Epimutation (maternal hypermethylation)",
        "gene": "14q32.3 methylation error",
        "frequency_pct": 10,
        "phenotype_class": "Moderate–Severe",
        "mechanism_detail": "SNP array NORMAL; methylation test essential; sporadic; recurrence ~5%",
        "dlk1_level": "2×",
        "rtl1_level": "2×",
        "meg3_level": "0",
    },
    {
        "variant": "Paternal dup 14q32.3",
        "gene": "DLK1/RTL1 gain (paternal)",
        "frequency_pct": 10,
        "phenotype_class": "Severe-Surviving–Severe-Lethal",
        "mechanism_detail": "CN gain; may worsen placental phenotype; parent-of-origin mandatory",
        "dlk1_level": "3×",
        "rtl1_level": "3×",
        "meg3_level": "absent/reduced",
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────────
TREATMENTS: List[Dict[str, Any]] = [
    {
        "treatment": "Mechanical Ventilation",
        "level": "A",
        "indication": "Neonatal respiratory failure from coat-hanger thorax + preterm birth",
        "drug_class": "Respiratory support",
        "notes": "MANDATORY in ~85% of neonates; prolonged ventilation weeks-months in severe cases",
        "contraindicated": False,
    },
    {
        "treatment": "CPAP / High-flow O₂",
        "level": "A",
        "indication": "Milder respiratory distress; step-down from ventilation",
        "drug_class": "Respiratory support",
        "notes": "Prone positioning beneficial for coat-hanger thorax geometry",
        "contraindicated": False,
    },
    {
        "treatment": "Surfactant (Beractant/Calfactant)",
        "level": "A",
        "indication": "Preterm KOS14 with respiratory distress syndrome (RDS)",
        "drug_class": "Surfactant",
        "notes": "Standard neonatal RDS treatment; preterm birth ~60% in KOS14",
        "contraindicated": False,
    },
    {
        "treatment": "Hepatoblastoma Surveillance (AFP + Ultrasound)",
        "level": "A",
        "indication": "ALL KOS14 patients — MEG3 absent reduces p53 activity → hepatoblastoma ~5% risk",
        "drug_class": "Surveillance",
        "notes": "AFP q3-6mo + liver ultrasound q6mo until age 5 (peak hepatoblastoma age 0-4); continue annually lifelong",
        "contraindicated": False,
    },
    {
        "treatment": "Levetiracetam (LEV)",
        "level": "B",
        "indication": "Focal epilepsy in KOS14 survivors (HIE sequelae)",
        "drug_class": "AED — SV2A ligand",
        "notes": "First-line focal AED; no hepatic risk; weight neutral; preferred in hepatoblastoma surveillance context",
        "contraindicated": False,
    },
    {
        "treatment": "Vigabatrin (VGB)",
        "level": "B",
        "indication": "Infantile spasms in KOS14 (HIE-related West syndrome)",
        "drug_class": "AED — GABA transaminase inhibitor",
        "notes": "ACTH preferred first-line for IS; VGB alternative; visual field monitoring mandatory",
        "contraindicated": False,
    },
    {
        "treatment": "ACTH (Adrenocorticotropic hormone)",
        "level": "B",
        "indication": "Infantile spasms / West syndrome in KOS14 (HIE context)",
        "drug_class": "Hormone therapy",
        "notes": "Level A for IS generally; Level B in KOS14 context (HIE IS may respond less than cryptogenic IS)",
        "contraindicated": False,
    },
    {
        "treatment": "Lamotrigine (LTG)",
        "level": "B",
        "indication": "Focal epilepsy in older survivors",
        "drug_class": "AED — sodium channel blocker",
        "notes": "Weight neutral; good tolerability; no hepatic risk",
        "contraindicated": False,
    },
    {
        "treatment": "Valproate (VPA)",
        "level": "C",
        "indication": "Generalised seizures in survivors where LEV/LTG insufficient",
        "drug_class": "AED — broad spectrum",
        "notes": "MODERATE RISK in KOS14: hepatic enzyme elevation monitoring overlaps hepatoblastoma surveillance (AFP elevation may be confounded); use only if other AEDs fail for generalised seizure types",
        "contraindicated": False,
    },
    {
        "treatment": "Rib Distraction / Thoracic Expansion Surgery",
        "level": "C",
        "indication": "Severe persistent thoracic restriction in survivors beyond neonatal period",
        "drug_class": "Surgical",
        "notes": "Experimental; few case series; may improve respiratory mechanics; Level C evidence (case reports only)",
        "contraindicated": False,
    },
    {
        "treatment": "GnRH Analog — NOT indicated",
        "level": "N/A",
        "indication": "NOT indicated in KOS14 (no CPP — contrast Temple Syndrome where Level A)",
        "drug_class": "Contraindicated context",
        "notes": "KOS14 has no DLK1 deficiency; DLK1 is excess; GnRH pulse generator inhibited (not activated); CPP absent",
        "contraindicated": True,
    },
    {
        "treatment": "GH Therapy — NOT indicated",
        "level": "N/A",
        "indication": "NOT indicated in KOS14 (no GH deficiency; RTL1 acts IGF2-like)",
        "drug_class": "Contraindicated context",
        "notes": "GH axis not deficient in KOS14; excess RTL1 has IGF2-like effects; GH supplementation not needed",
        "contraindicated": True,
    },
]

# ── Drug Risks ────────────────────────────────────────────────────────────────────
DRUG_RISKS: List[Dict[str, Any]] = [
    {
        "drug": "Valproate (VPA)",
        "risk_level": "MODERATE",
        "risk_type": "Hepatic monitoring complexity / hepatotoxicity",
        "mechanism": "VPA hepatic enzyme elevation + hepatoblastoma AFP monitoring = confounding; hepatotoxicity rare but serious; weight gain rare in survivors with KOS14 body habitus",
        "recommendation": "Use only if other AEDs insufficient for generalised seizures; monitor LFTs and AFP separately with oncology team",
        "absolute_ci": False,
    },
    {
        "drug": "Sedatives / Opioids",
        "risk_level": "HIGH",
        "risk_type": "Respiratory depression in coat-hanger thorax",
        "mechanism": "Coat-hanger thorax severely limits respiratory reserve; sedatives can precipitate respiratory failure",
        "recommendation": "Use with extreme caution in any KOS14 patient; respiratory monitoring mandatory",
        "absolute_ci": False,
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "risk_level": "LOW",
        "risk_type": "Not absolutely contraindicated in KOS14",
        "mechanism": "No specific contraindication in KOS14 (contrast Angelman where CBZ/OXC = absolute CI for myoclonic/atonic seizures); focal epilepsy in KOS14 survivors may respond to CBZ/OXC",
        "recommendation": "Can be used for focal epilepsy; verify seizure type first (not myoclonic/atonic as in Angelman)",
        "absolute_ci": False,
    },
    {
        "drug": "Long-chain fat parenteral nutrition",
        "risk_level": "LOW",
        "risk_type": "Increased hepatic load in hepatoblastoma-monitored patients",
        "mechanism": "KOS14 patients on hepatoblastoma surveillance; excess hepatic lipid load theoretically increases hepatic stress; monitor LFTs",
        "recommendation": "Standard neonatal PN acceptable; monitor liver enzymes in ICU setting",
        "absolute_ci": False,
    },
]

# ── EEG Patterns ─────────────────────────────────────────────────────────────────
EEG_PATTERNS: List[Dict[str, str]] = [
    {
        "pattern": "Suppression-Burst",
        "context": "Severe HIE in neonatal period — background suppression alternating with brief bursts",
        "significance": "Poor prognosis sign for survivors; associated with severe cortical injury",
        "frequency": "~20% of neonatal KOS14 with HIE",
    },
    {
        "pattern": "Hypsarrhythmia",
        "context": "West syndrome / infantile spasms in HIE survivors (age 3-12 months)",
        "significance": "HIE-related infantile spasms; chaotic high-amplitude multifocal discharges; treat with ACTH/VGB",
        "frequency": "~10% of KOS14 survivors",
    },
    {
        "pattern": "Focal Slowing + Focal Spikes",
        "context": "Post-HIE focal epilepsy in older survivors",
        "significance": "Structural focal epilepsy from HIE cortical injury; LEV/LTG responsive ~60%",
        "frequency": "~15% of survivors with epilepsy",
    },
    {
        "pattern": "Near-Normal / Diffuse Slowing",
        "context": "Moderate KOS14 survivors without significant HIE",
        "significance": "Background slowing commensurate with developmental delay; no epileptiform activity",
        "frequency": "~40% of moderate phenotype survivors",
    },
    {
        "pattern": "Normal EEG",
        "context": "Mild KOS14 or survivors beyond 5 years without seizures",
        "significance": "Normal EEG does not exclude KOS14; diagnosis is molecular",
        "frequency": "~15% of KOS14 survivors",
    },
]

# ── Patient Cohort Generation ─────────────────────────────────────────────────────
def _make_patients(n: int = 40) -> List[Dict[str, Any]]:
    rng = random.Random(SEED)
    patients = []

    pheno_pool = (
        ["Severe-Lethal"] * 12 +
        ["Severe-Surviving"] * 20 +
        ["Moderate"] * 8
    )
    rng.shuffle(pheno_pool)

    mech_pool = (
        ["upd14pat"] * 16 +
        ["maternal_deletion"] * 16 +
        ["igdmr_epimutation"] * 4 +
        ["paternal_duplication"] * 4
    )
    rng.shuffle(mech_pool)

    seizure_type_pool = ["Focal", "Infantile Spasms", "GTCS", "None"]
    aed_pool = ["Levetiracetam", "Vigabatrin", "ACTH", "Lamotrigine", "Valproate", "None"]

    for i in range(n):
        pid = f"KOS14-{293000 + i:06d}"
        sex = rng.choice(["M", "F"])
        phenotype = pheno_pool[i]
        mechanism = mech_pool[i]

        age_onset_mo = None
        if phenotype == "Severe-Lethal":
            age_onset_mo = 0  # neonatal death
            has_epilepsy = False
            seizure_type = None
            current_aed = None
            survived = False
        elif phenotype == "Severe-Surviving":
            survived = True
            has_epilepsy = rng.random() < 0.22
            if has_epilepsy:
                age_onset_mo = rng.randint(0, 18)
                seizure_type = rng.choice(["Focal", "Infantile Spasms", "GTCS"])
                current_aed = rng.choice(["Levetiracetam", "Vigabatrin", "ACTH", "Lamotrigine"])
            else:
                age_onset_mo = None
                seizure_type = None
                current_aed = None
        else:  # Moderate
            survived = True
            has_epilepsy = rng.random() < 0.10
            if has_epilepsy:
                age_onset_mo = rng.randint(6, 36)
                seizure_type = rng.choice(["Focal", "GTCS"])
                current_aed = rng.choice(["Levetiracetam", "Lamotrigine"])
            else:
                age_onset_mo = None
                seizure_type = None
                current_aed = None

        # Coat-hanger ribs: present in almost all
        coat_hanger_ribs = True if phenotype != "Moderate" else rng.random() < 0.75
        macrosomia = rng.random() < 0.60
        polyhydramnios = rng.random() < 0.75
        placentomegaly = rng.random() < 0.70
        abdominal_wall_defect = rng.random() < 0.40
        macroglossia = rng.random() < 0.50
        preterm = rng.random() < 0.60

        # Hepatoblastoma surveillance
        hepatoblastoma_surveillance = survived
        hepatoblastoma_detected = rng.random() < 0.05 if survived else False

        # DLK1 serum (elevated in KOS14 — both alleles paternally expressed)
        dlk1_serum_pct = rng.uniform(150, 280) if mechanism in ("upd14pat", "igdmr_epimutation") else rng.uniform(130, 220)

        # IG-DMR methylation (100% = both alleles methylated / paternal pattern)
        igdmr_methylation_pct = rng.uniform(92, 100)

        patients.append({
            "id": pid,
            "sex": sex,
            "phenotype_group": phenotype,
            "mechanism": mechanism,
            "survived_neonatal": survived,
            "coat_hanger_ribs": coat_hanger_ribs,
            "macrosomia": macrosomia,
            "polyhydramnios": polyhydramnios,
            "placentomegaly": placentomegaly,
            "abdominal_wall_defect": abdominal_wall_defect,
            "macroglossia": macroglossia,
            "preterm": preterm,
            "has_epilepsy": has_epilepsy,
            "seizure_onset_months": age_onset_mo,
            "seizure_type": seizure_type,
            "current_aed": current_aed,
            "hepatoblastoma_surveillance": hepatoblastoma_surveillance,
            "hepatoblastoma_detected": hepatoblastoma_detected,
            "dlk1_serum_pct_of_normal": round(dlk1_serum_pct, 1),
            "igdmr_methylation_pct": round(igdmr_methylation_pct, 1),
        })

    return patients


_PATIENTS = _make_patients(40)


# ── API Functions ─────────────────────────────────────────────────────────────────
def get_overview() -> Dict[str, Any]:
    pts = _PATIENTS
    n = len(pts)
    survived = [p for p in pts if p["survived_neonatal"]]
    n_survived = len(survived)
    epilepsy_n = sum(1 for p in pts if p["has_epilepsy"])
    coat_hanger_n = sum(1 for p in pts if p["coat_hanger_ribs"])
    macrosomia_n = sum(1 for p in pts if p["macrosomia"])
    polyhydramnios_n = sum(1 for p in pts if p["polyhydramnios"])
    placentomegaly_n = sum(1 for p in pts if p["placentomegaly"])
    abdominal_n = sum(1 for p in pts if p["abdominal_wall_defect"])
    macroglossia_n = sum(1 for p in pts if p["macroglossia"])
    preterm_n = sum(1 for p in pts if p["preterm"])
    hepatoblastoma_n = sum(1 for p in pts if p["hepatoblastoma_detected"])
    avg_dlk1 = round(sum(p["dlk1_serum_pct_of_normal"] for p in pts) / n, 1)
    avg_igdmr = round(sum(p["igdmr_methylation_pct"] for p in pts) / n, 1)

    mechanism_counts: Dict[str, int] = {}
    for p in pts:
        mechanism_counts[p["mechanism"]] = mechanism_counts.get(p["mechanism"], 0) + 1

    phenotype_counts: Dict[str, int] = {}
    for p in pts:
        phenotype_counts[p["phenotype_group"]] = phenotype_counts.get(p["phenotype_group"], 0) + 1

    seizure_types: Dict[str, int] = {}
    for p in pts:
        if p["seizure_type"]:
            seizure_types[p["seizure_type"]] = seizure_types.get(p["seizure_type"], 0) + 1

    aed_dist: Dict[str, int] = {}
    for p in pts:
        if p["current_aed"]:
            aed_dist[p["current_aed"]] = aed_dist.get(p["current_aed"], 0) + 1

    return {
        "disease": "Kagami-Ogata Syndrome (KOS14)",
        "gene": "DLK1 / RTL1 / MEG3",
        "locus": "14q32.3",
        "inheritance": "Genomic Imprinting — Maternal LOF",
        "mechanism": "Paternal UPD14 (~40%) / Maternal deletion 14q32.3 (~40%) / IG-DMR epimutation (~10%) / Paternal duplication (~10%)",
        "omim_gene_dlk1": "176290",
        "omim_gene_meg3": "601626",
        "omim_gene_rtl1": "603899",
        "omim_disease": "608149",
        "prevalence": "~1:50,000–100,000 (likely underdiagnosed due to high neonatal mortality)",
        "cohort_n": n,
        "seed": SEED,
        "kpis": {
            "neonatal_mortality_pct": round(100 * (n - n_survived) / n, 1),
            "coat_hanger_ribs_pct": round(100 * coat_hanger_n / n, 1),
            "polyhydramnios_pct": round(100 * polyhydramnios_n / n, 1),
            "macrosomia_pct": round(100 * macrosomia_n / n, 1),
            "preterm_pct": round(100 * preterm_n / n, 1),
            "epilepsy_pct": round(100 * epilepsy_n / n, 1),
            "hepatoblastoma_detected_pct": round(100 * hepatoblastoma_n / n, 1),
            "placentomegaly_pct": round(100 * placentomegaly_n / n, 1),
            "avg_dlk1_serum_pct_of_normal": avg_dlk1,
            "avg_igdmr_methylation_pct": avg_igdmr,
        },
        "mechanism_distribution": mechanism_counts,
        "phenotype_distribution": phenotype_counts,
        "seizure_types": seizure_types,
        "aed_distribution": aed_dist,
        "treatments": TREATMENTS,
        "drug_risks": DRUG_RISKS,
        "eeg_patterns": EEG_PATTERNS,
        "key_facts": [
            "Kagami-Ogata Syndrome = MATERNAL LOF at 14q32.3 — SAME LOCUS as Temple Syndrome, OPPOSITE PARENT",
            "Coat-hanger ribs (horizontal bell-shaped thorax) = PATHOGNOMONIC sign on chest X-ray (>95%)",
            "Paternal UPD14 (~40%) OR Maternal deletion 14q32.3 (~40%) — two equally common mechanisms",
            "DLK1 and RTL1 excess (paternal genes unopposed) → fetal overgrowth + placental dysplasia",
            "MEG3 absent (normally maternally expressed) → reduced p53 → hepatoblastoma risk ~5%",
            "Hepatoblastoma surveillance MANDATORY: AFP q3-6 months + liver ultrasound q6 months lifelong",
            "Epilepsy ~15-25% in SURVIVORS only; mechanism = HIE from neonatal respiratory failure",
            "NO absolute AED contraindications (contrast Angelman: CBZ/OXC absolute CI)",
            "VPA: MODERATE risk — hepatic monitoring overlaps hepatoblastoma surveillance; avoid if possible",
            "NO GnRH analog (contrast Temple Syndrome where CPP = Level A indication)",
            "IG-DMR methylation = 100% in KOS14 (both alleles methylated) vs 0% in Temple Syndrome",
            "SNP array alone misses UPD (40%) and epimutation (10%) — methylation test is FIRST",
            "Neonatal mortality ~30% despite ICU; surviving patients need lifelong multidisciplinary care",
        ],
    }


def get_breakdown() -> Dict[str, Any]:
    pts = _PATIENTS
    return {
        "patients": pts,
        "mechanisms": MECHANISMS,
        "phenotypes": PHENOTYPES,
        "variants": VARIANTS,
        "treatments": TREATMENTS,
        "drug_risks": DRUG_RISKS,
        "eeg_patterns": EEG_PATTERNS,
        "summary": {
            "total": len(pts),
            "survived_neonatal": sum(1 for p in pts if p["survived_neonatal"]),
            "neonatal_death": sum(1 for p in pts if not p["survived_neonatal"]),
            "with_epilepsy": sum(1 for p in pts if p["has_epilepsy"]),
            "coat_hanger_ribs": sum(1 for p in pts if p["coat_hanger_ribs"]),
            "polyhydramnios": sum(1 for p in pts if p["polyhydramnios"]),
            "macrosomia": sum(1 for p in pts if p["macrosomia"]),
            "placentomegaly": sum(1 for p in pts if p["placentomegaly"]),
            "hepatoblastoma_detected": sum(1 for p in pts if p["hepatoblastoma_detected"]),
        },
    }


def get_definitions() -> Dict[str, Any]:
    return {
        "disease_overview": {
            "Kagami_Ogata_Syndrome_KOS14": (
                "Kagami-Ogata Syndrome (KOS14, OMIM #608149) is a severe genomic imprinting disorder "
                "caused by maternal loss-of-function at the 14q32.3 imprinting domain — "
                "the OPPOSITE parent of origin compared to Temple Syndrome (paternal LOF at same locus). "
                "Two principal paternally expressed genes are dose-dysregulated: "
                "DLK1 (Delta-Like 1 Homolog, OMIM *176290) — anti-adipogenic Notch-like transmembrane protein — "
                "and RTL1 (Retrotransposon-Like 1, OMIM *603899) — placental vascular protease. "
                "The maternally expressed lncRNA MEG3 (OMIM *601626) is absent in KOS14, "
                "reducing p53 tumour suppressor activity and causing hepatoblastoma predisposition (~5%). "
                "Prevalence: ~1:50,000–100,000; likely underdiagnosed due to high neonatal mortality."
            ),
        },
        "pathognomonic_sign": {
            "Coat_Hanger_Ribs": (
                "Coat-hanger ribs are PATHOGNOMONIC for Kagami-Ogata Syndrome. "
                "On chest X-ray: ribs are horizontal (perpendicular to the spine) or upward-slanting, "
                "shortened, creating a bell-shaped or 'coat-hanger' thoracic silhouette. "
                "The normal thorax has downward-slanting ribs (30-45° angle); in KOS14 this angle is "
                "close to 0° or positive. "
                "This abnormal thoracic geometry severely restricts lung expansion and diaphragm movement, "
                "causing restrictive respiratory physiology. "
                "Mechanism: abnormal thoracic cage ossification driven by excess paternal RTL1/DLK1 "
                "affecting rib periosteum development. "
                "Present in >95% of KOS14 cases; absent in Temple Syndrome (same locus, opposite parent). "
                "Coat-hanger rib appearance is the most reliable clinical discriminator of KOS14 from "
                "other imprinting disorders (PWS, AS, Temple Syndrome)."
            ),
        },
        "imprinting_mechanism": {
            "14q32_Imprinting_Map_KOS14": (
                "The 14q32.3 imprinting domain in Kagami-Ogata Syndrome: "
                "Normal state: DLK1 and RTL1 expressed from paternal allele only; "
                "MEG3, MEG8, MEG9 expressed from maternal allele only. "
                "IG-DMR: METHYLATED on paternal allele (prevents MEG3 expression from paternal); "
                "UNMETHYLATED on maternal allele (allows MEG3 expression from maternal). "
                "In KOS14 (maternal LOF): maternal allele absent or silenced → "
                "IG-DMR is BIALLELICALLY METHYLATED (100% methylation — both alleles have paternal pattern) "
                "→ DLK1 expressed from BOTH alleles (2×) → RTL1 expressed from BOTH alleles (2×) "
                "→ MEG3 absent (maternally expressed; no maternal allele)."
            ),
            "RTL1_Mechanism": (
                "RTL1 (Retrotransposon-Like 1, 14q32.31) encodes a protease essential for "
                "placental villous trophoblast vascular integrity. "
                "Normally, RTL1 (paternal, expressed) is post-transcriptionally regulated by "
                "RTL1as (antisense RNA, maternal origin, expressed from maternal allele). "
                "In KOS14: RTL1 is doubled (2× paternal alleles) AND RTL1as is absent (no maternal allele). "
                "Unregulated excess RTL1 protease → disrupts placental villous vasculature → "
                "Placental Mesenchymal Dysplasia (PMD): abnormal villous hypertrophy, "
                "stem cell arteries thrombosis, loss of capillary structure. "
                "PMD → placentomegaly (very large placenta >90th percentile) → "
                "polyhydramnios (impaired fluid resorption) → preterm birth → respiratory failure."
            ),
            "MEG3_Absence_Hepatoblastoma": (
                "MEG3 (Maternally Expressed Gene 3) is a tumour-suppressor lncRNA that activates "
                "p53 pathway and represses cell proliferation. "
                "In KOS14: MEG3 absent from both alleles (maternally expressed gene; both alleles "
                "have paternal imprint silencing MEG3). "
                "Consequence: reduced p53 tumour suppressor activity in hepatocytes → "
                "hepatoblastoma predisposition (~5% lifetime risk). "
                "Peak hepatoblastoma age: 0-4 years (hepatoblastoma is a paediatric liver tumour). "
                "AFP (alpha-fetoprotein) is the biomarker: normally high at birth, falls by age 1; "
                "in hepatoblastoma: fails to fall or rises → AFP q3-6 months is mandatory surveillance. "
                "Liver ultrasound q6 months for morphological assessment."
            ),
        },
        "genetic_mechanisms": {
            "upd14pat_Mechanism": (
                "Paternal UPD14 (upd(14)pat) accounts for ~40% of Kagami-Ogata Syndrome. "
                "Patient has TWO paternal chromosome 14 copies, NO maternal chromosome 14. "
                "Subtypes: isodisomy (same paternal chr14 duplicated — LOH across all chr14 on SNP array) "
                "or heterodisomy (both paternal homologs — LOH only at centromeric region). "
                "Isodisomy creates risk for unmasking autosomal recessive conditions on chr14. "
                "Methylation result: IG-DMR shows 100% methylation (both alleles paternal pattern). "
                "Mechanism: meiotic non-disjunction (paternal meiosis II most common) + "
                "rescue of trisomic conceptus → paternal isodisomy."
            ),
            "Maternal_Deletion_14q32": (
                "Maternal deletion of 14q32.3 accounts for ~40% of KOS14. "
                "Deletion removes MEG3/IG-DMR region from the MATERNAL chromosome. "
                "Detected by CMA as copy number loss at 14q32.3. "
                "Parental studies MANDATORY: "
                "  — If from MOTHER: 50% recurrence risk (carrier mother = NO phenotype, "
                "    because normally maternal MEG3 is expressed and DLK1 maternally silenced; "
                "    mother has one paternal chr14 with intact DLK1 = NO KOS14); "
                "    her AFFECTED children received the maternal deletion + inherited normal paternal chr14 → "
                "    no maternal MEG3 → KOS14. "
                "  — If de novo: recurrence risk ~1%. "
                "CRITICAL PARENT-OF-ORIGIN RULE: "
                "  Same deletion PATERNALLY inherited from father → Temple Syndrome (paternal LOF). "
                "  Same deletion MATERNALLY inherited from mother → KOS14 (maternal LOF). "
                "  This is the cardinal demonstration of imprinting."
            ),
        },
        "diagnosis": {
            "First_Test_Methylation_IG_DMR": (
                "Methylation analysis at 14q32.3 (IG-DMR and MEG3-DMR) is the FIRST and most "
                "sensitive diagnostic test for Kagami-Ogata Syndrome. "
                "Normal result: ~50% methylation at IG-DMR (one paternal methylated + one maternal unmethylated). "
                "Abnormal KOS14 result: 90-100% methylation (both alleles methylated = paternal pattern). "
                "Sensitivity: >98% for upd(14)pat + maternal deletion + epimutation. "
                "CRITICAL: SNP array alone misses ~50% of KOS14 cases "
                "(upd(14)pat with no CN change: 40%; epimutation with no CN change: 10%). "
                "Always order methylation FIRST in the workup of coat-hanger rib phenotype."
            ),
            "Second_Test_SNP_Array_Parental": (
                "After abnormal methylation (100% at IG-DMR): SNP array identifies mechanism. "
                "upd(14)pat isodisomy: LOH across entire chr14 without CN change. "
                "Maternal deletion: CN loss at 14q32.3. "
                "Paternal duplication: CN gain at 14q32.3. "
                "Parental SNP array studies mandatory for recurrence risk counselling: "
                "confirm maternal origin of deletion (50% recurrence) vs de novo (~1%). "
                "DLK1 serum protein: ELEVATED in KOS14 (emerging biomarker; not yet routine 2026)."
            ),
            "KOS14_vs_BWS_Discriminators": (
                "Kagami-Ogata Syndrome vs Beckwith-Wiedemann Syndrome (BWS) — "
                "both have macrosomia + macroglossia + abdominal wall defects. "
                "DISCRIMINATING FEATURES:\n"
                "  Coat-hanger ribs: KOS14 = PATHOGNOMONIC (>95%); BWS = ABSENT\n"
                "  Chromosome locus: KOS14 = 14q32.3; BWS = 11p15.5 (IGF2/H19)\n"
                "  Tumour risk: KOS14 = hepatoblastoma ~5%; BWS = Wilms tumour (~5%) + hepatoblastoma + others\n"
                "  Respiratory failure: KOS14 = PRIMARY from thoracic dysplasia; BWS = less severe\n"
                "  Methylation test: KOS14 = IG-DMR 14q32.3; BWS = H19/IGF2 DMR 11p15.5\n"
                "  Limb hemihypertrophy: BWS common; KOS14 absent"
            ),
        },
        "treatment_details": {
            "Respiratory_Management": (
                "Neonatal respiratory failure in Kagami-Ogata Syndrome is the primary cause of "
                "mortality. The coat-hanger thorax creates a fixed restrictive defect: "
                "ribs cannot expand during inspiration, severely limiting tidal volume. "
                "Management: immediate intubation and mechanical ventilation in severe cases. "
                "Positive pressure ventilation may be difficult (poor compliance due to rigid cage). "
                "High-frequency oscillatory ventilation (HFOV) may improve gas exchange. "
                "Surfactant for preterm (60% of KOS14 neonates are preterm). "
                "Prone positioning may improve diaphragm mechanics with the inverted thoracic shape. "
                "Long-term survivors on home ventilator support have been reported."
            ),
            "Hepatoblastoma_Surveillance": (
                "Hepatoblastoma surveillance is MANDATORY for ALL surviving KOS14 patients. "
                "Protocol: AFP every 3-6 months + liver ultrasound every 6 months. "
                "Duration: at minimum until age 5 years (peak hepatoblastoma incidence 0-4 years); "
                "annual AFP + ultrasound for life thereafter (lower but non-zero risk persists). "
                "Normal AFP kinetics: high at birth (~100,000 ng/mL), falls by age 1-2 to adult levels. "
                "Interpret AFP in context of age-specific norms (paediatric oncology reference ranges). "
                "AFP elevation OR failure to decline as expected → urgent paediatric oncology referral. "
                "VPA can cause mild AFP elevation — coordinate AED choice with oncology team if AFP surveillance is active."
            ),
            "Epilepsy_Management_KOS14": (
                "Epilepsy in KOS14 survivors is SECONDARY to hypoxic-ischemic encephalopathy (HIE) "
                "from neonatal respiratory failure — NOT a primary epilepsy gene disorder. "
                "Key distinction from Angelman syndrome: NO absolute AED contraindications in KOS14. "
                "Seizure types in survivors: focal (most common), infantile spasms (HIE-West), GTCS. "
                "First-line for focal epilepsy: Levetiracetam (LEV) — Level B. "
                "First-line for infantile spasms: ACTH or Vigabatrin — Level B in HIE context. "
                "VPA: MODERATE risk (hepatic monitoring overlap) — use only if other AEDs fail "
                "for generalised seizure types; coordinate AFP surveillance with oncology. "
                "DRE (drug-resistant epilepsy): ~30% of KOS14 epilepsy (HIE-related). "
                "Epilepsy surgery: consider for focal cortical lesions from HIE; MRI first."
            ),
            "VPA_Moderate_Risk_KOS14": (
                "Valproate (VPA) is classified as MODERATE RISK in Kagami-Ogata Syndrome — "
                "NOT absolutely contraindicated, but requires careful coordination. "
                "Risk rationale: "
                "(1) Hepatic enzyme elevation from VPA may confound AFP interpretation in "
                "    patients under hepatoblastoma surveillance. "
                "(2) VPA can cause mild-moderate AFP elevation in paediatric patients, complicating "
                "    the primary tumour surveillance biomarker. "
                "(3) Hepatotoxicity risk (rare, idiosyncratic) adds complexity to LFT monitoring "
                "    already required for hepatoblastoma surveillance. "
                "Recommendation: prefer LEV or LTG as first-line AED; use VPA only if specifically "
                "indicated (e.g. myoclonic seizures not controlled by alternatives). "
                "If VPA is used: coordinate monthly LFT + AFP monitoring with paediatric oncology."
            ),
        },
        "key_comparisons": {
            "KOS14_vs_Temple_Syndrome": (
                "Kagami-Ogata Syndrome vs Temple Syndrome — SAME LOCUS 14q32.3, OPPOSITE PARENT:\n"
                "KOS14 (Maternal LOF): DLK1 2×, RTL1 2×, MEG3 0 → MACROSOMIA, coat-hanger ribs,\n"
                "  severe neonatal respiratory failure, hepatoblastoma risk, high mortality\n"
                "Temple Syndrome (Paternal LOF): DLK1 0, MEG3 2× → SGA, mild hypotonia,\n"
                "  truncal obesity, CPP, mild IDD, epilepsy 20-30%, low mortality\n"
                "SAME: 14q32.3 locus; methylation test; same gene products (opposite direction)\n"
                "DIFFERENT: phenotypes are mirror images; IG-DMR methylation 0% vs 100%;\n"
                "  DLK1 absent (TS14) vs elevated (KOS14); coat-hanger ribs only KOS14"
            ),
            "KOS14_vs_BWS": (
                "Kagami-Ogata Syndrome vs Beckwith-Wiedemann Syndrome (OMIM #130650):\n"
                "Both: macrosomia, macroglossia, abdominal wall defects, tumour predisposition\n"
                "KOS14-only: coat-hanger ribs (PATHOGNOMONIC), 14q32.3 locus, hepatoblastoma only\n"
                "BWS-only: Wilms tumour (in addition to hepatoblastoma), hemihypertrophy (30%),\n"
                "  11p15.5 locus, facial nevus flammeus, neonatal hypoglycemia (excess IGF2 → HI)\n"
                "Key test: KOS14 → IG-DMR 14q32.3 methylation; BWS → H19/IGF2 DMR 11p15.5"
            ),
            "KOS14_vs_Angelman": (
                "Kagami-Ogata Syndrome vs Angelman Syndrome (UBE3A, 15q11-q13):\n"
                "KOS14: severe neonatal phenotype; respiratory dominant; epilepsy SECONDARY (HIE);\n"
                "  NO absolute AED CI; VPA moderate risk; coat-hanger ribs\n"
                "Angelman: epilepsy PRIMARY (85%); CBZ/OXC ABSOLUTE CI; VPA Level A first-line;\n"
                "  happy demeanor; absent speech; ataxia; hand-flapping\n"
                "Key: epilepsy mechanism completely different; AED recommendations opposite"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_overview(), indent=2, default=str))
