#!/usr/bin/env python3
"""Temple Syndrome (TS14) Dashboard.

Temple Syndrome is a GENOMIC IMPRINTING disorder at the 14q32.3 MEG3/DLK1 locus.
  Principal genes: MEG3 (Maternally Expressed Gene 3, lncRNA), DLK1 (Delta-Like 1 Homolog)
  Mechanism: PATERNAL LOF of 14q32.3 → loss of DLK1 (paternally expressed only)
  Most common cause: Maternal UPD14 (upd(14)mat) — patient has TWO maternal chr14, NO paternal chr14
  Result: DLK1 ABSENT (paternally expressed, maternally silenced) + MEG3 BIALLELICALLY expressed

IMPRINTING MECHANISM — WHY PATERNAL LOF CAUSES TEMPLE SYNDROME:
  DLK1 (Delta-Like 1 Homolog, 383 aa, 14q32.2): PATERNALLY expressed only
    — maternal copy is epigenetically silenced by the intergenic DMR (IG-DMR) and MEG3-DMR
    — DLK1 encodes a transmembrane protein homologous to Notch ligands; inhibits adipogenesis
    — inhibits preadipocyte differentiation via Notch pathway in hypothalamic-pituitary axis
    — DLK1 null in sheep → increased adiposity + precocious puberty → phenocopies TS14
  MEG3 (Maternally Expressed Gene 3, 14q32.3): MATERNALLY expressed, lncRNA
    — maternal expression driven by IG-DMR hypomethylation (maternal allele = unmethylated)
    — paternal copy is silenced by CpG methylation of IG-DMR and MEG3-DMR
    — In upd(14)mat: TWO maternal alleles → MEG3 BIALLELIC overexpression
    — MEG3 targets: p53 tumour suppressor pathway, TGF-β signalling, angiogenesis
    — Biallelic MEG3 → enhanced p53 activity → apoptosis in GnRH neurons → CPP mechanism?

  NORMAL 14q32.3 imprinting map (key genes):
    Paternally expressed: DLK1, RTL1 (antisense: RTL1as — maternally expressed)
    Maternally expressed: MEG3, MEG8, MEG9, MIATS (lncRNAs)
    IG-DMR (Intergenic DMR): differentially methylated region between DLK1 and MEG3
      — METHYLATED on paternal allele, UNMETHYLATED on maternal allele
      — Primary imprint: established in germ cells; controls DLK1/MEG3 expression
    MEG3-DMR (secondary): methylated on paternal allele

FOUR GENETIC MECHANISMS (by frequency):
  1. Maternal UPD14 (upd(14)mat) (~70%): two maternal chromosome 14 copies
     — paternal chr14 absent → DLK1 absent (both copies maternally silenced)
     — detected by SNP array (LOH chr14) ± methylation analysis (IG-DMR methylation 0%)
     — isodisomy (same chr14 twice) vs heterodisomy (both maternal homologs present)
     — recurrence risk negligible (<1%) for UPD
  2. Paternal deletion 14q32.3 (~15%): deletes DLK1/MEG3 domain from PATERNAL chromosome
     — detected by CMA/FISH; may be de novo or inherited (paternally transmitted)
     — if deletion inherited from father: 50% recurrence (father's children at risk)
     — critical: deletion of DLK1 alone = TS14; larger deletions may extend phenotype
  3. Epimutation (IG-DMR hypermethylation on paternal allele converts to maternal pattern): rare
     — methylation study shows 0% paternal methylation at IG-DMR
     — SNP array normal (no UPD, no deletion)
     — cause: spontaneous error in post-fertilisation methylation maintenance
     — recurrence risk low but not negligible
  4. Maternal duplication 14q32.3: extremely rare; extra copy of maternal MEG3 region
     — doubles MEG3 dose without DLK1 loss; overlapping but milder TS14 phenotype

CLINICAL FEATURES:
  Neonatal/Infant:
    Small for gestational age (SGA): ~80-90% (weight <-2 SD at birth)
    Hypotonia: 75-85%; may be severe enough to require NG tube (PWS-like neonatal period)
    Feeding difficulties: 70-80%; failure to thrive in early infancy
    Polyhydramnios: 30-40% (prenatal)
    Small placenta: 50-60%

  Childhood/Growth:
    Short stature: nearly universal (height SDS -2 to -4); NOT due to GH deficiency typically
      — GH secretion usually NORMAL (contrast PWS where GH deficient >90%)
      — short stature mechanism: intrinsic skeletal growth limitation + prenatal growth failure
      — GH therapy can add 4-7 cm final height (off-label)
    Central precocious puberty (CPP): 50-60% females; less common in males
      — onset: 6-8 years (early puberty); mechanism: DLK1 normally inhibits GnRH → loss → CPP
      — treat with GnRH analog (leuprolide/triptorelin) — Level A; improves adult height
    Truncal obesity: 60-70% by adolescence (milder than PWS; food-seeking less pathological)
    Small hands and feet: ~70%
    Elfin facies: triangular face, upslanting palpebral fissures, frontal bossing, ear lobe pits

  Neurodevelopment:
    Intellectual disability: mild (IQ 65-80 range); ~80% affected; severe IDD <10%
    ADHD/behavioural: 40-50%; inattention, hyperactivity
    Anxiety: 30-40%
    Autism spectrum features: 15-20%; less severe than FXS or Angelman
    Hypotonia → delayed motor milestones (sitting 10-14 months; walking 18-24 months)
    Language: mild delay (first words 15-20 months); most develop functional speech

  Epilepsy (~20-30%):
    Seizure types: focal (most common), GTCS, absence, febrile seizures
    Onset: typically childhood (2-8 years)
    EEG: focal slowing, occasional focal spikes; NO pathognomonic pattern (contrast Angelman)
    Usually treatment-responsive; DRE uncommon (<10%)
    No absolute AED contraindications (contrast Angelman where CBZ/OXC = absolute CI)
    Recommended first-line: LEV, LTG, OXC (for focal), VPA (for generalised)
    VPA risk: weight gain (exacerbates truncal obesity) — moderate risk; use cautiously

  Skeletal:
    Scoliosis: 20-30%
    Mild brachydactyly
    Club foot (talipes equinovarus): 10-15%

  Endocrine:
    CPP: 50-60% (treat with GnRH analog)
    Insulin resistance / T2DM risk by adulthood: 20-30% (truncal obesity + CPP → hyperinsulinism)
    Thyroid function: usually normal
    Adrenal: usually normal

KEY BIOMARKERS / DIAGNOSIS:
  First test: MS-MLPA methylation analysis at 14q32.3 (IG-DMR and MEG3-DMR)
    — Abnormal: IG-DMR unmethylated on BOTH alleles = maternal pattern (no paternal methylation)
    — Sensitivity >98% for UPD14mat + deletion + epimutation
    — Normal methylation: consider other diagnoses
  Second test: SNP array
    — upd(14)mat: loss of heterozygosity (LOH) chr14 WITHOUT copy number change (isodisomy)
      OR presence of both maternal homologs (heterodisomy) + LOH
    — Paternal deletion: copy number loss at 14q32.3 (paternal allele)
  Confirmatory: UPD confirmation (parental studies + SNP array genotyping)
  DLK1 serum protein: REDUCED in TS14 (biomarker under study; not routine yet)

TREATMENT:
  1. GnRH ANALOG (leuprolide / triptorelin): Level A — for CPP (common ~50-60% females)
     — decelerates bone age advance; improves adult height prediction by 4-6 cm
     — duration: until age ~11 (girls) or bone age closure appropriate
  2. GH THERAPY: Level C (off-label use) — for short stature
     — ~4-7 cm height SDS improvement; no FDA approval for TS specifically
     — consider if height SDS <-3 and bone age not advanced
     — monitor: bone age, scoliosis progression, IGF-1 levels
  3. DIETARY MANAGEMENT: Level B — manage truncal obesity; not as severe as PWS
     — standard dietary counselling; no food locking required (unlike PWS)
     — emphasis on low glycaemic index diet (T2DM prevention)
  4. LEVETIRACETAM (LEV): Level B — focal/generalised seizures; weight-neutral; first-line
  5. LAMOTRIGINE (LTG): Level B — generalised/focal; weight-neutral; slow titration
  6. OXCARBAZEPINE (OXC): Level B — focal seizures; hyponatraemia monitoring
  7. VPA: Level C — effective but MODERATE RISK in TS14 (weight gain worsens obesity)
     — NOT absolute CI (unlike Angelman); use if seizure type warrants (generalised)
     — metabolic monitoring mandatory; avoid if BMI >95th percentile
  8. MELATONIN: Level B — sleep disturbance (common in TS14; hypotonia → fragmented sleep)
  9. SSRI (fluoxetine/sertraline): Level C — anxiety, behavioural; 30-40% response
  10. METFORMIN: Level C — insulin resistance / T2DM risk; weight-neutral

DRUG RISKS IN TS14:
  VPA: MODERATE RISK — weight gain worsens truncal obesity; not as extreme as PWS
  Antipsychotics (typical): HIGH RISK — weight gain; metabolic syndrome
  Topiramate: CAUTION — cognitive effects in IDD population; weight-neutral benefit
  GH therapy: caution — monitor for scoliosis progression (annual spinal imaging)
  No ABSOLUTE CI AED — unlike Angelman Syndrome (CBZ/OXC = absolute CI in AS)

KEY VARIANTS / BREAKPOINTS:
  upd(14)mat isodisomy: both chr14 from same maternal homolog; LOH + homozygosity
  upd(14)mat heterodisomy: both maternal homologs present; LOH at centromeric regions
  del(14)(q32.3): paternal deletion; size variable; minimal region = DLK1 + IG-DMR
  IG-DMR epimutation: abnormal methylation, normal CN, normal SNP → molecular diagnosis
  Paternal translocation: rare; unbalanced rearrangement

KEY EXAM TRAPS:
  TS14 vs PWS — SAME NEONATAL PRESENTATION, DIFFERENT LOCUS:
    Both: neonatal hypotonia + feeding failure + SGA → 'PWS-like'; methylation test differentiates
    PWS: 15q11-q13 paternal LOF; hyperphagia pathological; GH deficient; no CPP; epilepsy 10-15%
    TS14: 14q32.3 paternal LOF; CPP 50-60%; GH usually normal; obesity milder; epilepsy 20-30%
  GH in TS14 — NOT the same indication as PWS:
    PWS: GH deficient (>90%) → GH = Level A, start infancy, FDA approved
    TS14: GH usually normal secretion → GH = Level C, off-label, for short stature only
  CPP in TS14 — COMMON (50-60% females); treat aggressively:
    GnRH analog improves adult height significantly in TS14
    CPP is RARE in PWS; central in PWS is less problematic
  DLK1 deficiency alone is sufficient for TS14 full phenotype
    (confirmed by isolated paternal DLK1 deletion case series)
  Methylation TEST is FIRST TEST — not SNP array or CMA alone
    (UPD has normal CN → CMA-only will miss ~70% of TS14 cases)
  Epilepsy in TS14 (20-30%) vs Angelman (85%):
    TS14 epilepsy = focal, mild, treatment-responsive — very different from AS
    No absolute AED CI in TS14 (CBZ can be used for focal seizures)

KEY DIFFERENTIALS:
  Prader-Willi (15q11-q13 paternal): same neonatal hypotonia + feeding difficulty; hyperphagia >>TS14
  Angelman (15q11-q13 maternal): severe epilepsy + absent speech + happy demeanor; NO obesity
  Silver-Russell Syndrome (SRS, 11p15.5 / upd(7)mat): SGA + asymmetry + limb length discrepancy
  Kabuki (KMT2D): facial dysmorphism + intellectual disability; different locus
  Kagami-Ogata (upd(14)pat — OPPOSITE parent): coat-hanger ribs + abdominal wall defects; severe
  Sotos (NSD1): OVERGROWTH not SGA; tall stature, macrocephaly; opposite of TS14 growth
  Albright Hereditary Osteodystrophy (GNAS imprinting, 20q13.32): PHP1A; subcutaneous calcifications
"""

import random

SEED = 291
random.seed(SEED)

# ── Genetic mechanisms ────────────────────────────────────────────────────────
MECHANISMS = [
    {
        "mechanism": "Maternal UPD14 (upd(14)mat) — isodisomy",
        "freq": 45,
        "detection": "SNP array (LOH chr14, no CN change) + methylation (IG-DMR)",
        "phenotype": "Classic TS14 — full phenotype; CPP in ~60% females; recurrence <1%",
        "recurrence": "<1%",
        "notes": "Both chr14 from same maternal homolog; isodisomy → risk of autosomal recessive disease",
    },
    {
        "mechanism": "Maternal UPD14 (upd(14)mat) — heterodisomy",
        "freq": 25,
        "detection": "SNP array (LOH centromeric regions) + methylation",
        "phenotype": "Classic TS14 — full phenotype; CPP in ~55% females; recurrence <1%",
        "recurrence": "<1%",
        "notes": "Both maternal homologs present; LOH only at centromeric/pericentromeric regions",
    },
    {
        "mechanism": "Paternal deletion 14q32.3 (del(14q32.3)pat)",
        "freq": 15,
        "detection": "CMA/FISH (copy number loss at 14q32.3 on paternal allele) + methylation",
        "phenotype": "Classic TS14; if inherited paternally → 50% recurrence risk",
        "recurrence": "50% if paternally inherited; ~1% if de novo",
        "notes": "Minimal critical region includes DLK1 + IG-DMR; larger deletions → additional genes",
    },
    {
        "mechanism": "IG-DMR epimutation (paternal methylation → maternal pattern)",
        "freq": 12,
        "detection": "Methylation analysis only (SNP array normal, CN normal)",
        "phenotype": "Classic TS14 — full phenotype; milder in some series",
        "recurrence": "Low but uncertain; germline epimutation risk",
        "notes": "Cause: post-fertilisation methylation maintenance error; paternal IG-DMR loses methylation",
    },
    {
        "mechanism": "Maternal duplication 14q32.3",
        "freq": 3,
        "detection": "CMA (copy number gain at 14q32.3 on maternal allele) + methylation",
        "phenotype": "Overlapping but milder TS14; MEG3 overexpressed without DLK1 loss",
        "recurrence": "50% if maternally inherited",
        "notes": "Extra copy of maternal MEG3/MEG8 region; DLK1 may still be present from paternal",
    },
]

# ── Phenotypic groups ─────────────────────────────────────────────────────────
PHENOTYPES = [
    {"group": "Classic SGA + CPP + Short Stature", "pct": 50},
    {"group": "Hypotonic-Dominant (PWS-like neonatal)", "pct": 30},
    {"group": "Mild Attenuated (ascertained via CPP or developmental delay)", "pct": 20},
]

# ── Variants ──────────────────────────────────────────────────────────────────
VARIANTS = [
    {"variant": "upd(14)mat-iso", "freq_pct": 45, "severity": "Classic", "mechanism": "UPD isodisomy"},
    {"variant": "upd(14)mat-hetero", "freq_pct": 25, "severity": "Classic", "mechanism": "UPD heterodisomy"},
    {"variant": "del(14q32.3)-pat", "freq_pct": 15, "severity": "Classic-Severe", "mechanism": "Paternal deletion (CMA)"},
    {"variant": "IG-DMR_epimutation", "freq_pct": 12, "severity": "Classic-Mild", "mechanism": "Epimutation (methylation only)"},
    {"variant": "dup(14q32.3)-mat", "freq_pct": 3, "severity": "Mild", "mechanism": "Maternal duplication"},
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {"name": "GnRH Analog (leuprolide / triptorelin)",   "level": "A",  "target": "Central Precocious Puberty (CPP)",      "notes": "Start at diagnosis of CPP; improves adult height +4-7 cm; mandatory if CPP onset <8y female"},
    {"name": "GH Therapy (off-label)",                   "level": "C",  "target": "Short stature",                         "notes": "GH secretion usually NORMAL in TS14; off-label; +4-7 cm SDS if bone age not advanced"},
    {"name": "Levetiracetam (LEV)",                      "level": "B",  "target": "Focal/generalised seizures",            "notes": "Weight-neutral; broad-spectrum; first-line; no drug interactions"},
    {"name": "Lamotrigine (LTG)",                        "level": "B",  "target": "Focal/generalised seizures",            "notes": "Weight-neutral; slow titration required; avoid rapid increase"},
    {"name": "Oxcarbazepine (OXC)",                      "level": "B",  "target": "Focal seizures",                        "notes": "Effective focal; monitor sodium (hyponatraemia); no absolute CI in TS14"},
    {"name": "Valproate (VPA)",                          "level": "C",  "target": "Generalised/absence seizures",          "notes": "MODERATE RISK weight gain → worsens obesity; use if seizure type warrants; metabolic monitoring"},
    {"name": "Dietary Management / Low-GI diet",         "level": "B",  "target": "Truncal obesity / T2DM prevention",     "notes": "Not as severe as PWS; food locking NOT required; standard dietary counselling sufficient"},
    {"name": "Melatonin",                                "level": "B",  "target": "Sleep disturbance",                     "notes": "Hypotonia → fragmented sleep architecture; 2-5 mg nocte"},
    {"name": "SSRI (sertraline / fluoxetine)",           "level": "C",  "target": "Anxiety / behaviour",                   "notes": "30-40% response; ADHD features in 40-50%; consider if anxiety/OCD-like prominent"},
    {"name": "Metformin",                                "level": "C",  "target": "Insulin resistance / T2DM risk",        "notes": "Adults with truncal obesity + insulin resistance; weight-neutral; renal monitoring"},
]

# ── EEG patterns ──────────────────────────────────────────────────────────────
EEG_PATTERNS = [
    {"pattern": "Focal slowing (temporal/frontal)",       "pct": 40, "notes": "Most common; mild; non-pathognomonic"},
    {"pattern": "Focal spikes / sharp waves",             "pct": 25, "notes": "Focal epileptiform; temporal or frontal predominance"},
    {"pattern": "Generalised spike-wave (2.5-3.5 Hz)",   "pct": 15, "notes": "Absence-type seizures; respond to LTG/VPA"},
    {"pattern": "Normal EEG",                             "pct": 15, "notes": "30-40% of TS14 patients with epilepsy — normal inter-ictal EEG"},
    {"pattern": "Generalised slowing (diffuse theta)",    "pct": 5,  "notes": "Non-specific background slowing during or after seizures"},
]


def _make_patients(n=40):
    patients = []
    mechanism_pool = []
    for m in MECHANISMS:
        mechanism_pool.extend([m["mechanism"]] * m["freq"])
    phenotype_pool = []
    for p in PHENOTYPES:
        phenotype_pool.extend([p["group"]] * p["pct"])

    for i in range(n):
        sex = random.choice(["Male", "Female", "Female"])  # slight female excess due to CPP detection
        mech = random.choice(mechanism_pool)
        pheno = random.choice(phenotype_pool)
        has_epilepsy = random.random() < 0.25   # 20-30% epilepsy rate

        birth_wt = round(random.gauss(-2.2, 0.6), 1)  # SDS, SGA ~-2 to -3
        height_sds = round(random.gauss(-2.8, 0.7), 1)
        has_cpp = (sex == "Female" and random.random() < 0.55) or (sex == "Male" and random.random() < 0.20)
        age_puberty_onset = round(random.uniform(5.5, 8.5), 1) if has_cpp else None
        bmi_sds = round(random.gauss(1.6, 0.8), 1)  # truncal obesity; milder than PWS

        age_dx = round(random.uniform(0.1, 6.0), 1)
        iq = int(random.gauss(72, 12))
        iq = max(45, min(95, iq))

        seizure_type = None
        seizure_age = None
        aed = None
        eeg_pattern = None
        if has_epilepsy:
            seizure_type = random.choice(["Focal with impaired awareness", "GTCS", "Absence", "Febrile seizure", "Focal with secondary generalisation"])
            seizure_age = round(random.uniform(1.5, 9.0), 1)
            aed = random.choice(["LEV", "LTG", "OXC", "VPA", "LEV + LTG"])
            eeg_pattern = random.choice([ep["pattern"] for ep in EEG_PATTERNS])

        gnrh_therapy = has_cpp and random.random() < 0.85
        gh_therapy = (height_sds < -3.0) and random.random() < 0.50

        dlk1_protein = round(random.gauss(35, 12), 0) if "del" in mech else round(random.gauss(30, 10), 0)
        # DLK1 serum protein: LOW in TS14 (reference >80% of adult mean)
        dlk1_pct = max(5, int(dlk1_protein))

        patients.append({
            "patient_id": f"TS14-{i + 1:03d}",
            "sex": sex,
            "mechanism": mech,
            "phenotype_group": pheno,
            "birth_weight_sds": birth_wt,
            "height_sds": height_sds,
            "bmi_sds": bmi_sds,
            "iq": iq,
            "age_at_diagnosis_y": age_dx,
            "has_cpp": has_cpp,
            "age_puberty_onset_y": age_puberty_onset,
            "on_gnrh_analog": gnrh_therapy,
            "on_gh_therapy": gh_therapy,
            "has_epilepsy": has_epilepsy,
            "seizure_type": seizure_type,
            "seizure_onset_y": seizure_age,
            "current_aed": aed,
            "eeg_pattern": eeg_pattern,
            "dlk1_serum_pct": dlk1_pct,
            "scoliosis": random.random() < 0.25,
            "hypotonia_neonatal": random.random() < 0.80,
            "polyhydramnios": random.random() < 0.35,
        })
    return patients


_PATIENTS = _make_patients(40)


def get_overview():
    pts = _PATIENTS
    n = len(pts)
    epilepsy_n = sum(1 for p in pts if p["has_epilepsy"])
    cpp_n = sum(1 for p in pts if p["has_cpp"])
    gnrh_n = sum(1 for p in pts if p["on_gnrh_analog"])
    gh_n = sum(1 for p in pts if p["on_gh_therapy"])
    hypotonia_n = sum(1 for p in pts if p["hypotonia_neonatal"])
    scoliosis_n = sum(1 for p in pts if p["scoliosis"])
    polyhydramnios_n = sum(1 for p in pts if p["polyhydramnios"])
    avg_height_sds = round(sum(p["height_sds"] for p in pts) / n, 2)
    avg_iq = round(sum(p["iq"] for p in pts) / n, 1)
    avg_dlk1 = round(sum(p["dlk1_serum_pct"] for p in pts) / n, 1)
    avg_bmi_sds = round(sum(p["bmi_sds"] for p in pts) / n, 2)

    mechanism_counts = {}
    for p in pts:
        mechanism_counts[p["mechanism"]] = mechanism_counts.get(p["mechanism"], 0) + 1
    phenotype_counts = {}
    for p in pts:
        phenotype_counts[p["phenotype_group"]] = phenotype_counts.get(p["phenotype_group"], 0) + 1

    seizure_types = {}
    for p in pts:
        if p["seizure_type"]:
            seizure_types[p["seizure_type"]] = seizure_types.get(p["seizure_type"], 0) + 1

    aed_dist = {}
    for p in pts:
        if p["current_aed"]:
            aed_dist[p["current_aed"]] = aed_dist.get(p["current_aed"], 0) + 1

    return {
        "disease": "Temple Syndrome (TS14)",
        "gene": "MEG3 / DLK1",
        "locus": "14q32.3",
        "inheritance": "Genomic Imprinting — Paternal LOF",
        "mechanism": "Maternal UPD14 (70%) / Paternal deletion 14q32.3 (15%) / Epimutation (12%) / Maternal duplication (3%)",
        "omim_gene_meg3": "601626",
        "omim_gene_dlk1": "176290",
        "omim_disease": "616222",
        "prevalence": "~1:50,000–100,000 (likely underdiagnosed due to neonatal PWS misdiagnosis)",
        "cohort_n": n,
        "seed": SEED,
        "kpis": {
            "epilepsy_pct": round(100 * epilepsy_n / n, 1),
            "cpp_pct": round(100 * cpp_n / n, 1),
            "gnrh_analog_pct": round(100 * gnrh_n / n, 1),
            "gh_therapy_pct": round(100 * gh_n / n, 1),
            "neonatal_hypotonia_pct": round(100 * hypotonia_n / n, 1),
            "scoliosis_pct": round(100 * scoliosis_n / n, 1),
            "polyhydramnios_pct": round(100 * polyhydramnios_n / n, 1),
            "avg_height_sds": avg_height_sds,
            "avg_bmi_sds": avg_bmi_sds,
            "avg_iq": avg_iq,
            "avg_dlk1_serum_pct": avg_dlk1,
        },
        "mechanism_distribution": mechanism_counts,
        "phenotype_distribution": phenotype_counts,
        "seizure_types": seizure_types,
        "aed_distribution": aed_dist,
        "treatments": TREATMENTS,
        "eeg_patterns": EEG_PATTERNS,
        "key_facts": [
            "Temple Syndrome = PATERNAL LOF at 14q32.3 → loss of DLK1 (paternally expressed only)",
            "Most common mechanism: maternal UPD14 (~70%) — TWO maternal chr14, NO paternal",
            "DLK1 deficiency → loss of adipogenesis inhibition + GnRH pulse generator dysregulation",
            "Central precocious puberty (CPP) in ~50-60% females — treat with GnRH analog",
            "GH usually NORMAL (contrast PWS where GH deficient >90%) — GH therapy is off-label",
            "Neonatal hypotonia + SGA = 'PWS-like' presentation — methylation test at 14q32.3 differentiates",
            "Epilepsy in ~20-30%; focal type; treatment-responsive; NO absolute AED contraindications",
            "Methylation analysis (IG-DMR) is FIRST TEST — SNP array alone misses 70% (UPD = no CN change)",
            "Paternal UPD14 (Kagami-Ogata) = OPPOSITE PARENT → coat-hanger ribs + severe phenotype",
            "VPA moderate risk (weight gain) — NOT absolute CI as in Angelman",
        ],
    }


def get_breakdown():
    pts = _PATIENTS
    return {
        "patients": pts,
        "mechanisms": MECHANISMS,
        "phenotypes": PHENOTYPES,
        "variants": VARIANTS,
        "treatments": TREATMENTS,
        "eeg_patterns": EEG_PATTERNS,
        "summary": {
            "total": len(pts),
            "with_epilepsy": sum(1 for p in pts if p["has_epilepsy"]),
            "with_cpp": sum(1 for p in pts if p["has_cpp"]),
            "on_gnrh_analog": sum(1 for p in pts if p["on_gnrh_analog"]),
            "on_gh_therapy": sum(1 for p in pts if p["on_gh_therapy"]),
            "neonatal_hypotonia": sum(1 for p in pts if p["hypotonia_neonatal"]),
            "scoliosis": sum(1 for p in pts if p["scoliosis"]),
        },
    }


def get_definitions():
    return {
        "disease_overview": {
            "Temple_Syndrome_TS14": (
                "Temple Syndrome (TS14, OMIM #616222) is a genomic imprinting disorder caused by "
                "paternal loss-of-function at the 14q32.3 imprinting domain. "
                "The principal effector gene is DLK1 (Delta-Like 1 Homolog, OMIM *176290), "
                "a paternally expressed transmembrane protein that inhibits adipogenesis via Notch "
                "pathway and regulates the GnRH pulse generator. "
                "Loss of DLK1 (due to absent paternal allele) causes: truncal obesity, "
                "central precocious puberty, short stature, neonatal hypotonia, and "
                "mild-moderate intellectual disability. "
                "MEG3 (Maternally Expressed Gene 3, OMIM *601626), the paternally imprinted "
                "lncRNA at the same locus, becomes biallelically expressed in upd(14)mat. "
                "Prevalence: ~1:50,000-100,000 (likely underdiagnosed)."
            ),
        },
        "imprinting_mechanism": {
            "14q32_Imprinting_Map": (
                "The 14q32.3 imprinting domain contains paternally expressed genes (DLK1, RTL1) "
                "and maternally expressed non-coding RNAs (MEG3, MEG8, MEG9). "
                "The intergenic DMR (IG-DMR) between DLK1 and MEG3 acts as the primary imprinting "
                "control region: METHYLATED on the PATERNAL allele (silences MEG3), "
                "UNMETHYLATED on the MATERNAL allele (allows MEG3 expression). "
                "In upd(14)mat: patient has TWO maternal chr14 → IG-DMR is completely unmethylated "
                "(0% paternal methylation) → DLK1 absent (both maternal copies silenced) + "
                "MEG3 biallelically expressed."
            ),
            "DLK1_Function": (
                "DLK1 (383 aa, type I transmembrane protein) contains multiple EGF-like repeats "
                "homologous to Notch ligands. DLK1 inhibits pre-adipocyte differentiation by "
                "blocking the Notch signalling cascade required for adipogenesis. "
                "DLK1 also inhibits the hypothalamic GnRH pulse generator — normally, declining "
                "DLK1 during childhood triggers puberty onset. "
                "In TS14: DLK1 absent from birth → GnRH inhibition lost from birth → "
                "premature puberty initiation (CPP in ~50-60% females). "
                "Animal model: DLK1 null sheep and mice → obesity + early puberty = TS14 phenocopy."
            ),
            "MEG3_Function": (
                "MEG3 (OMIM *601626) encodes a long non-coding RNA expressed from the maternal "
                "allele. MEG3 activates the p53 tumour suppressor pathway and represses TGF-β "
                "signalling. In upd(14)mat: MEG3 biallelically expressed → enhanced p53 activity "
                "in GnRH neurons and hypothalamic cells → may contribute to TS14 neurological "
                "phenotype including CPP and intellectual disability (mechanism under study)."
            ),
        },
        "genetic_mechanisms": {
            "upd14mat_UPD": (
                "Maternal UPD14 (upd(14)mat) accounts for ~70% of Temple Syndrome cases. "
                "The patient has TWO maternal chromosome 14 copies and NO paternal chromosome 14. "
                "Two subtypes: (1) isodisomy — both chr14 from the same maternal homolog "
                "(identical copies, detected by LOH on SNP array across entire chr14); "
                "(2) heterodisomy — both maternal homologs present (LOH only at centromeric regions). "
                "Isodisomy creates risk of autosomal recessive disease (homozygosity) if a "
                "pathogenic variant is present on that chromosome. "
                "Recurrence risk: <1% (non-disjunction event, typically maternal meiosis II error)."
            ),
            "Paternal_deletion_14q32": (
                "Paternal deletion of 14q32.3 (del(14)(q32.3)pat) accounts for ~15% of TS14. "
                "The deletion removes DLK1 (and usually IG-DMR) from the paternal chromosome. "
                "Detected by chromosomal microarray (CMA) as a copy number loss at 14q32.3. "
                "Critical point: if the deletion is de novo → recurrence risk ~1%; "
                "if inherited from the FATHER → recurrence risk 50% (children of father may inherit "
                "the deleted paternal chr14). "
                "Maternal inheritance of same deletion → NO TS14 (DLK1 is not expressed from "
                "maternal allele anyway → deletion has no phenotypic effect maternally). "
                "This parent-of-origin effect is hallmark of imprinting."
            ),
            "IG_DMR_Epimutation": (
                "Epimutation at the IG-DMR accounts for ~12% of TS14. "
                "The paternal chromosome loses its normal CpG methylation at the IG-DMR, "
                "acquiring the maternal unmethylated pattern. "
                "Result: IG-DMR on paternal chr14 now behaves like maternal allele → "
                "DLK1 is silenced on both alleles → TS14 phenotype. "
                "Chromosome copy number is NORMAL; UPD is ABSENT; only methylation is abnormal. "
                "SNP array will be NORMAL → methylation analysis is mandatory first test. "
                "Cause: post-fertilisation methylation maintenance error (DNMT1/DNMT3A dysregulation)."
            ),
        },
        "diagnosis": {
            "First_Test_Methylation_IG_DMR": (
                "MS-MLPA (methylation-specific multiplex ligation-dependent probe amplification) "
                "at the 14q32.3 IG-DMR and MEG3-DMR is the FIRST and most sensitive diagnostic test. "
                "Normal result: ~50% methylation at IG-DMR (one methylated paternal + one unmethylated "
                "maternal allele → 50% methylated). "
                "Abnormal TS14 result: 0-10% methylation (no paternal methylated allele = no paternal chr14). "
                "Sensitivity: >98% for upd(14)mat + deletion + epimutation. "
                "SNP array alone will MISS ~70% of TS14 cases (UPD has no copy number change)."
            ),
            "Second_Test_SNP_Array": (
                "After abnormal methylation: SNP array identifies the mechanism. "
                "upd(14)mat isodisomy: LOH across entire chr14 without copy number change. "
                "upd(14)mat heterodisomy: LOH in centromeric region, normal distal CN. "
                "Paternal deletion: copy number loss at 14q32.3 (single copy). "
                "Parental studies (SNP array of both parents) confirm inheritance pattern. "
                "DLK1 protein level in serum: reduced in TS14 (emerging biomarker, not routine 2026)."
            ),
            "PWS_Misdiagnosis_Risk": (
                "Temple Syndrome is frequently misdiagnosed as Prader-Willi Syndrome (PWS) in "
                "the neonatal period. Both present with: severe hypotonia, NG tube feeding, SGA, "
                "genital hypoplasia in males, developmental delay. "
                "Initial DNA methylation test for 15q11-q13 (standard PWS test) will be NORMAL. "
                "If PWS methylation is normal + neonatal hypotonia + SGA: PROCEED to 14q32.3 "
                "methylation testing. "
                "Mean diagnostic delay in TS14: 2-5 years (PWS misdiagnosis is the most common "
                "cause of delay)."
            ),
        },
        "treatment_details": {
            "GnRH_Analog_CPP": (
                "Central precocious puberty (CPP) occurs in ~50-60% of female TS14 patients "
                "(onset 5.5-8.5 years). GnRH analogs (leuprolide acetate or triptorelin) suppress "
                "the premature GnRH pulse generator activation. "
                "Evidence level A: improves adult predicted height by 4-7 cm in TS14. "
                "Mechanism: DLK1 normally inhibits GnRH neurons; DLK1 absence → premature activation. "
                "Duration: typically until chronological age 10-11 (females) or bone age reaches "
                "appropriate stage. Monitor bone density during treatment."
            ),
            "GH_Therapy_Rationale": (
                "GH therapy in TS14 is Level C (off-label, not FDA approved for TS14 specifically). "
                "CRITICAL DISTINCTION FROM PWS: GH SECRETION IS USUALLY NORMAL IN TS14 "
                "(growth hormone provocation tests often normal). Short stature in TS14 is due to "
                "prenatal growth restriction + intrinsic skeletal limitation, NOT GH deficiency. "
                "If considering GH: confirm height SDS <-3.0 AND bone age not advanced. "
                "Reported benefit: 4-7 cm SDS improvement in case series. "
                "Monitoring: annual bone age X-ray, scoliosis surveillance, IGF-1 levels."
            ),
            "VPA_Moderate_Risk": (
                "Valproate (VPA) is NOT absolutely contraindicated in Temple Syndrome "
                "(contrast Angelman where CBZ/OXC = absolute CI, and PWS where VPA = HIGH RISK). "
                "VPA risk in TS14: MODERATE — weight gain worsens truncal obesity; "
                "hyperammonemia at high doses; teratogenicity for females of childbearing age. "
                "Use VPA if seizure type specifically warrants (e.g. generalised absence, myoclonic) "
                "and other AEDs have failed. "
                "If VPA used: monthly weight monitoring, quarterly BMI, liver enzymes annually; "
                "consider alternative (LTG, LEV) if weight gain exceeds 1 BMI-SDS."
            ),
        },
        "key_comparisons": {
            "TS14_vs_PWS": (
                "Temple Syndrome vs Prader-Willi Syndrome — KEY DISCRIMINATORS:\n"
                "SAME: neonatal hypotonia, SGA, NG tube feeding, failure to thrive, small hands/feet, "
                "truncal obesity, intellectual disability, short stature.\n"
                "DIFFERENT:\n"
                "  Locus: TS14 = 14q32.3 (DLK1/MEG3); PWS = 15q11.2-q13 (SNORD116/SNRPN/MKRN3)\n"
                "  Hyperphagia: PATHOLOGICAL in PWS (food locking mandatory); MILD in TS14\n"
                "  GH: DEFICIENT >90% in PWS (Level A FDA); USUALLY NORMAL in TS14 (off-label)\n"
                "  CPP: RARE in PWS; COMMON 50-60% in TS14 females\n"
                "  First methylation test: 15q11-q13 for PWS; 14q32.3 for TS14\n"
                "  Epilepsy: 10-20% PWS; 20-30% TS14 (both mild-moderate)\n"
                "  VPA risk: HIGH RISK in PWS (obesity + hepatosteatosis); MODERATE in TS14\n"
                "  Psychosis: 20-30% mUPD15 in PWS; NOT reported in TS14"
            ),
            "TS14_vs_Kagami_Ogata": (
                "Temple Syndrome vs Kagami-Ogata Syndrome — OPPOSITE PARENT, SAME LOCUS:\n"
                "TS14 (upd(14)mat / paternal LOF): DLK1 absent → CPP, obesity, mild IDD, epilepsy 20-30%\n"
                "Kagami-Ogata (upd(14)pat / maternal LOF): DLK1 overexpressed + MEG3 absent\n"
                "  → coat-hanger rib shape (thoracic dysplasia), abdominal wall defects,\n"
                "  → placentomegaly, large tongue, neonatal respiratory failure, severe/lethal\n"
                "  → NO obesity (opposite of TS14); NO CPP\n"
                "KEY: same locus, opposite phenotypes — genomic imprinting demonstrated by parent-of-origin"
            ),
            "TS14_vs_SRS": (
                "Temple Syndrome vs Silver-Russell Syndrome (SRS) — similar SGA presentation:\n"
                "SRS (OMIM #180860): SGA + relative macrocephaly + body asymmetry + limb discrepancy\n"
                "Mechanism: 11p15.5 IGF2/H19 locus (60%) or upd(7)mat (7-10%)\n"
                "TS14: truncal obesity develops; CPP common; hypotonia more severe\n"
                "SRS: NO obesity; NO CPP; limb asymmetry (hemihypotrophy) distinguishes from TS14\n"
                "Both: SGA, short stature, failure to thrive, methylation testing required"
            ),
            "TS14_vs_Angelman": (
                "Temple Syndrome vs Angelman Syndrome — different imprinting loci, both neonatal hypotonia:\n"
                "Angelman (UBE3A, 15q11-q13, MATERNAL LOF): severe epilepsy 85%, absent speech,\n"
                "  happy demeanor, ataxic gait, hand-flapping, CBZ/OXC absolute CI\n"
                "TS14 (14q32.3, PATERNAL LOF): epilepsy 20-30% mild/focal, speech develops (with delay),\n"
                "  no characteristic behavioral demeanor, NO absolute AED CI\n"
                "Key: epilepsy burden dramatically LOWER in TS14 vs AS; AED management very different"
            ),
        },
    }
