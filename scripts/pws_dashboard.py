#!/usr/bin/env python3
"""Prader-Willi Syndrome (PWS) Dashboard.

PWS involves PATERNALLY expressed genes at 15q11.2-q13 — the SAME locus as Angelman but
the OPPOSITE parent. Loss of paternal 15q11-q13 expression causes PWS:
  Principal genes: SNORD116 cluster (snoRNA), SNRPN, MKRN3, MAGEL2, NDN, PWAR5
  SNORD116: Critical minimal region — paternal SNORD116 deletion alone sufficient for PWS core features
  SNRPN: 465 aa; SmN protein; pre-mRNA splicing; host gene for SNORD cluster
  MKRN3: Makorin RING finger protein 3; E3 ubiquitin ligase; inhibits GnRH pulse generator
  MAGEL2: MAGE family member L2; protein homeostasis; circadian rhythm; hypothalamic function
  NDN: Necdin; 325 aa; MAGE family; promotes neuronal differentiation; hypothalamic neurons
  Locus: 15q11.2-q13 (PATERNAL expression required; maternally imprinted / silenced)
  OMIM Gene: *182279 (SNRPN)
  OMIM Disease: #176270 (Prader-Willi Syndrome)
  Prevalence: ~1:10,000–30,000 live births

IMPRINTING MECHANISM — WHY PATERNAL LOF CAUSES DISEASE:
  Genes in the 15q11-q13 PWS domain are PATERNALLY expressed only.
  The MATERNAL copies are silenced by imprinting (maternal imprinting centre directs silencing).
  Loss of paternal 15q11-q13 → loss of ALL paternally expressed genes in this domain.
  Only the maternal copies remain, but they are epigenetically silenced in relevant tissues.
  Result: zero expression of SNORD116, SNRPN, MKRN3, MAGEL2, NDN → Prader-Willi Syndrome.

FOUR GENETIC MECHANISMS (by frequency):
  1. Paternal deletion 15q11.2-q13 (60-70%): de novo; detected by FISH/CMA
     Type 1 (bp1-bp3, ~6 Mb, 60% of deletions): larger; OCA2 and P gene included; slightly more severe
     Type 2 (bp2-bp3, ~5 Mb, 40% of deletions): smaller; most common single subtype
     Both types: deletion of SNORD116, SNRPN, MKRN3, MAGEL2, NDN + proximal neighbour genes
  2. Maternal uniparental disomy (mUPD15) (25-30%): two maternal chromosome 15 copies
     Maternal copies silenced by imprinting → effectively no paternal-expressed genes
     Mildly MILDER hypotonia; HIGHER risk of psychosis (20-30%) vs deletion class
     Higher ASD features; may have slightly higher IQ; NBS undetectable by standard MS/MS
  3. Imprinting centre (IC) defect (~1-3%): paternal IC epimutation or microdeletion
     Paternal chromosome acquires maternal methylation pattern → paternal genes silenced
     Inherited IC defect: up to 50% recurrence; sporadic: <1% recurrence
  4. Chromosomal translocation (rare, <1%): unbalanced translocation; CMA/karyotype

SNORD116 — PRINCIPAL DRIVER:
  SNORD116 encodes a cluster of ~30 C/D box small nucleolar RNAs.
  Isolated SNORD116 microdeletion → FULL PWS metabolic phenotype (hyperphagia, obesity, GH deficiency).
  Mechanism: SNORD116 regulates pre-mRNA splicing and circadian rhythm in hypothalamic neurons.
  Loss → disrupted satiety signalling, disrupted circadian feeding drive, GH axis dysregulation.
  Downstream: SNORD116 targets include mRNA of NHLH2 (transcription factor regulating pro-opiomelanocortin).

CLINICAL PHASES:
  Phase 1 (birth–2 years): Severe HYPOTONIA; feeding failure; NG tube often needed;
    failure to thrive; temperature instability; genital hypoplasia
  Phase 2 (2–8 years): HYPERPHAGIA onset; behavioral food-seeking; weight gain begins;
    developmental delay; emotional lability
  Phase 3 (childhood–adult): OBESITY if uncontrolled; short stature (GH deficiency);
    hypogonadism; intellectual disability (IQ 60-70 modal); skin-picking; scoliosis

CHARACTERISTIC FEATURES:
  1. HYPERPHAGIA (pathological): insatiable hunger; food-seeking; no satiety signalling;
     hyperphagia onset age 2-6 years; if unchecked → morbid obesity + T2DM + cardiovascular death
  2. HYPOTONIA (neonatal severe): improves with age; persists throughout life
  3. SHORT STATURE: GH deficiency in >90%; responds dramatically to GH therapy
  4. HYPOGONADISM: cryptorchidism 100% males; female amenorrhea/oligomenorrhea; HGH-LH axis
  5. INTELLECTUAL DISABILITY: mild-moderate; IQ 60-70 typical; learning disability; adaptive deficits
  6. BEHAVIORAL: temper tantrums (50-70%); skin picking (70-80%); OCD-like rigidity; food obsession
  7. SLEEP: central hypoventilation; OSA; excessive daytime sleepiness; REM abnormalities
  8. CHARACTERISTIC FACIES: narrow bifrontal diameter; almond-shaped eyes; thin upper lip;
     downturned corners of mouth; small hands and feet; light complexion (relative, not pathognomonic)

EEG / EPILEPSY (much less common than Angelman — key contrast):
  Epilepsy: 10-20% (vs Angelman 85%; MAJOR CONTRAST)
  Seizure types: Focal > generalized; GTCS, absence; typically mild and treatment-responsive
  EEG: Nonspecific; focal slowing; occasional spike-wave; diffuse slowing
  Gelastic seizures: RARE; associated with hypothalamic hamartoma (not from PWS gene mechanism per se)
  Response: Most respond to standard AED monotherapy (LEV, LTG, VPA); DRE very uncommon
  No specific pathognomonic EEG pattern (contrast with Angelman high-amplitude delta)
  Psychosis risk: mUPD15 class — higher risk (bipolar-like, psychosis in 20-30%)

KEY BIOMARKERS / DIAGNOSIS:
  First test: DNA methylation study at SNRPN locus (15q11-q13) — detects deletion, mUPD, IC defect
  Abnormal methylation (paternal-only pattern): 97-99% sensitive for PWS
  Normal methylation: consider other diagnoses (or rare translocation — chromosome karyotype)
  After methylation: CMA/FISH (deletion? type 1 vs 2); SNP array (mUPD15 → LOH, normal CN);
    IC sequencing (IC defect)
  NBS: NOT on standard MS/MS panel (no acylcarnitine/amino acid marker); DNA-based NBS piloted

TREATMENT:
  1. GH THERAPY: Level A — FDA approved 2000 (Genotropin); start in infancy (even before hyperphagia);
     Benefits: stature (+10-12 cm), body composition (muscle:fat ratio), energy, cognition;
     Dose: 0.24 mg/kg/week subcutaneous; monitor for scoliosis, glucose; avoid in severe obesity/resp failure
  2. DIETARY MANAGEMENT + FOOD SECURITY: Level A — essential; structured calories (1,000-1,200 kcal/day);
     locked refrigerators and pantries; no unsupervised food access; permanent necessity
  3. CARBETOCIN (oxytocin agonist): Level B (Phase III trials CARE-PWS) — targets hyperphagia and behavior;
     investigational 2026; reduces food-related distress; intranasal
  4. MELATONIN: Level B — central hypoventilation + sleep disturbance; 2-6 mg nocte
  5. SSRI (fluoxetine/sertraline): Level B — skin picking, OCD-like, emotional lability; 30-40% response
  6. RISPERIDONE/ARIPIPRAZOLE: Level B — aggressive behavior, self-injury; BMI monitoring mandatory
  7. GnRH ANALOG (leuprolide): Level A — precocious puberty subset (MKRN3 LOF); timing critical
  8. SEX HORMONE REPLACEMENT: Level B — hypogonadism (testosterone boys; estrogen/progesterone girls)
  9. METFORMIN: Level B — insulin resistance, pre-T2DM; weight-neutral; renal monitoring
  10. TOPIRAMATE: Level C — epilepsy + weight-neutral effect; CAUTION — cognitive effects in already
      intellectually impaired patients; not first-line
  11. LEV (levetiracetam): Level B — first-line AED for epilepsy; broad-spectrum; safe
  12. LTG (lamotrigine): Level B — generalised seizures; safe; weight-neutral
  13. VPA (valproate): Level B for epilepsy BUT HIGH RISK — obesity → hepatic steatosis → VPA hepatotoxicity;
      weight gain (worsens PWS core problem); hyperammonemia at high doses; use with caution if at all

DRUG RISKS IN PWS:
  VPA: HIGH RISK — weight gain + hepatic steatosis risk; metabolic monitoring mandatory; avoid if possible
  Topiramate: CAUTION — cognitive impairment in IDD; weight-neutral but narrow therapeutic index
  Antipsychotics (typical): HIGH RISK — weight gain; extrapyramidal; metabolic syndrome;
     prefer atypicals (aripiprazole → lower metabolic risk)
  Benzodiazepines: CAUTION — respiratory depression risk in central hypoventilation/OSA
  GH: AVOID in severe untreated obesity + respiratory failure (risk of sudden death); screen before start
  No ABSOLUTE CI AED (unlike AS where CBZ/OXC are absolute CI): CBZ usable for focal if needed

KEY VARIANTS:
  del1 (Type 1, bp1-bp3, ~6 Mb): 60% of deletions; more severe behaviorally; slightly lower IQ
  del2 (Type 2, bp2-bp3, ~5 Mb): 40% of deletions; most common single deletion type
  mUPD15: maternal isodisomy or heterodisomy; psychosis risk 20-30%; more ASD; ≈IQ
  IC defect: methylation abnormal; IC sequencing confirms; recurrence risk 50% if inherited
  SNORD116 microdeletion: rare; isolated SNORD116 → pure PWS metabolic phenotype; proof of principal region

KEY EXAM TRAPS:
  PWS vs AS — SAME LOCUS, OPPOSITE PARENT:
    PATERNAL deletion 15q11-q13 = PWS (hypotonia → hyperphagia → obesity; minimal epilepsy)
    MATERNAL deletion 15q11-q13 = AS (severe epilepsy + ataxia + absent speech + happy demeanor)
  mUPD15 PWS: psychosis risk — do NOT miss in adults presenting with psychosis + obesity + hypotonia
  GH therapy is SAFE and EFFECTIVE even before hyperphagia onset — start in infancy
  GH therapy can MASK scoliosis progression — annual spine imaging mandatory
  VPA is NOT absolutely contraindicated in PWS (unlike AS), BUT very high metabolic risk — weigh carefully
  Neonatal hypotonia + NG tube feeding in male → check methylation immediately (50% diagnostic delay)

KEY DIFFERENTIALS:
  Angelman Syndrome (same locus, maternal): severe epilepsy + ataxia + happy demeanor; NO obesity
  Fragile X (FMR1, Xq27.3): more ASD; macroorchidism; CGG expansion; different locus
  Bardet-Biedl Syndrome (BBS; ciliopathy): obesity + polydactyly + retinal dystrophy + renal anomalies
  Alström (ALMS1): obesity + deafness + retinal dystrophy + cardiomyopathy; no hypotonia
  Schaaf-Yang (MAGEL2 LOF/truncation, 15q11-q13): PWS-like + congenital contractures + more severe autism
  Kleefstra (EHMT1, 9q34.3): obesity + intellectual disability + behavioral; different locus
  Cohen syndrome (VPS13B, 8q22): neutropenia + retinal dystrophy + facial features; not obese
"""

import random

SEED = 289
random.seed(SEED)

# ── Genetic mechanisms ────────────────────────────────────────────────────────
MECHANISMS = [
    {
        "mechanism": "Paternal deletion 15q11.2-q13 (Type 1, bp1-bp3, ~6 Mb)",
        "freq": 40,
        "detection": "CMA (chromosomal microarray) / FISH",
        "phenotype": "Classic PWS — slightly more severe behavioral; slightly lower IQ (~5 pts)",
        "note": (
            "~40% of all PWS cases (60% of deletion class × 65% type 1 proportion). "
            "Larger ~6 Mb deletion; includes NIPA1/NIPA2 genes not in type 2. "
            "Same core features as Type 2 but marginally more severe behavioral phenotype. "
            "OCA2 gene is within deleted region but maternal OCA2 intact → no hypopigmentation "
            "(OCA2 is biallelically expressed; only one copy needed). "
            "Contrast with AS: same physical region but maternal deletion causes AS."
        ),
    },
    {
        "mechanism": "Paternal deletion 15q11.2-q13 (Type 2, bp2-bp3, ~5 Mb)",
        "freq": 27,
        "detection": "CMA (chromosomal microarray) / FISH",
        "phenotype": "Classic PWS — most common single deletion type; full PWS features",
        "note": (
            "~27% of all PWS cases (60% deletion × 45% type 2 proportion). "
            "Smaller ~5 Mb deletion; NIPA1/NIPA2 retained. Full PWS features including "
            "hypotonia, hyperphagia, GH deficiency, hypogonadism, ID. "
            "De novo in >99% — extremely low recurrence risk. "
            "Detected by CMA showing hemizygous deletion of 15q11.2-q13."
        ),
    },
    {
        "mechanism": "Maternal uniparental disomy 15 (mUPD15)",
        "freq": 28,
        "detection": "Methylation study (abnormal) + SNP array (LOH without CN change)",
        "phenotype": "PWS — slightly milder hypotonia; HIGHER psychosis risk (20-30%); more ASD",
        "note": (
            "25-30% of PWS. Two MATERNAL chromosome 15 copies (isodisomy or heterodisomy). "
            "Paternal genes (SNORD116, SNRPN, MKRN3, MAGEL2, NDN) absent. "
            "CRITICAL: mUPD15 class has 20-30% lifetime risk of psychosis (bipolar-like or "
            "schizophrenia spectrum) — highest among all genetic UPD syndromes. "
            "More ASD features. May have slightly milder neonatal hypotonia. "
            "NOT detectable by FISH/CMA — methylation study ABNORMAL; SNP array shows "
            "loss of heterozygosity at 15q11 without copy number change."
        ),
    },
    {
        "mechanism": "Imprinting centre (IC) defect",
        "freq": 3,
        "detection": "Methylation study (abnormal) → IC sequencing",
        "phenotype": "PWS — similar to mUPD15 phenotype; high recurrence if inherited IC deletion",
        "note": (
            "~1-3% of PWS. Paternal chromosome acquires MATERNAL methylation pattern → "
            "paternal SNORD116/SNRPN/MKRN3/MAGEL2/NDN silenced. "
            "Sporadic epimutations: <1% recurrence. IC microdeletion: up to 50% recurrence "
            "(autosomal dominant imprinting centre deletion). "
            "Phenotype similar to mUPD15. Key for genetic counselling in families planning "
            "future pregnancies."
        ),
    },
    {
        "mechanism": "Chromosomal translocation / other",
        "freq": 2,
        "detection": "Karyotype + CMA / FISH",
        "phenotype": "Variable; depends on size and extent of unbalanced translocation",
        "note": (
            "<1% of PWS. Unbalanced chromosomal translocation involving 15q11-q13. "
            "May have additional features from gain/loss of other chromosomal material. "
            "Parents require karyotype to assess balanced carrier status and recurrence risk."
        ),
    },
]

# ── Distribution counts for 40 patients ─────────────────────────────────────
MECHANISM_DIST = {
    "Deletion 15q11.2-q13 Type 1 (bp1-bp3)": 16,
    "Deletion 15q11.2-q13 Type 2 (bp2-bp3)": 11,
    "Maternal UPD15 (mUPD15)": 11,
    "Imprinting centre defect": 1,
    "Chromosomal translocation": 1,
}

# ── Seizure types (much lower rates than AS) ──────────────────────────────────
SEIZURE_TYPES = [
    "Focal (unknown onset)", "GTCS", "Absence", "Myoclonic (rare)",
    "Gelastic (rare)", "Status epilepticus (rare)",
]
SEIZURE_PROBS = [0.08, 0.07, 0.05, 0.02, 0.01, 0.01]   # low rates; total ~10-15% epilepsy

# ── EEG patterns (nonspecific in PWS) ────────────────────────────────────────
EEG_PATTERNS = [
    "Normal EEG",
    "Focal slowing (frontotemporal/temporal)",
    "Occasional spike-wave (generalised)",
    "Diffuse slowing",
    "Normal background with rare IED",
]


def _make_patient(i):
    """Synthetic PWS patient record (seed=289, deterministic)."""
    rng = random.Random(SEED + i * 67)

    # Genetic mechanism and derived phenotype
    if i < 16:
        mechanism = "Deletion 15q11.2-q13 Type 1 (bp1-bp3)"
        severity_id = "Classic"
        hypotonia_score = round(rng.uniform(7.0, 10.0), 1)   # Severe neonatal
        gh_deficiency = True
        iq_est = rng.randint(52, 72)
        psychosis_risk = False
        asd_features = rng.random() < 0.15
        del_type = "bp1-bp3 (~6 Mb), Type 1"
    elif i < 27:
        mechanism = "Deletion 15q11.2-q13 Type 2 (bp2-bp3)"
        severity_id = "Classic"
        hypotonia_score = round(rng.uniform(7.0, 10.0), 1)
        gh_deficiency = True
        iq_est = rng.randint(55, 75)
        psychosis_risk = False
        asd_features = rng.random() < 0.15
        del_type = "bp2-bp3 (~5 Mb), Type 2"
    elif i < 38:
        mechanism = "Maternal UPD15 (mUPD15)"
        severity_id = "Moderate"
        hypotonia_score = round(rng.uniform(5.5, 8.5), 1)   # Slightly milder
        gh_deficiency = rng.random() < 0.85
        iq_est = rng.randint(58, 78)
        psychosis_risk = rng.random() < 0.25   # 20-30% lifetime risk
        asd_features = rng.random() < 0.40   # More ASD in mUPD15
        del_type = f"mUPD15 ({'isodisomy' if rng.random()<0.5 else 'heterodisomy'})"
    elif i == 38:
        mechanism = "Imprinting centre defect"
        severity_id = "Moderate"
        hypotonia_score = round(rng.uniform(5.0, 8.0), 1)
        gh_deficiency = rng.random() < 0.85
        iq_est = rng.randint(58, 76)
        psychosis_risk = rng.random() < 0.20
        asd_features = rng.random() < 0.35
        del_type = "IC epimutation / microdeletion (paternal)"
    else:
        mechanism = "Chromosomal translocation"
        severity_id = "Variable"
        hypotonia_score = round(rng.uniform(6.0, 9.5), 1)
        gh_deficiency = rng.random() < 0.90
        iq_est = rng.randint(45, 72)
        psychosis_risk = False
        asd_features = rng.random() < 0.25
        del_type = "Unbalanced translocation involving 15q11-q13"

    # BMI at last visit (uncontrolled hyperphagia → obesity in many)
    bmi = round(rng.uniform(22.0, 45.0), 1)
    obesity = bmi >= 30
    t2dm = bmi >= 35 and rng.random() < 0.40

    # GH therapy status
    gh_therapy = gh_deficiency and rng.random() < 0.88   # Most get GH if diagnosed
    gh_started_age_months = round(rng.uniform(6, 36), 1) if gh_therapy else None

    # Hyperphagia onset (age months)
    hyperphagia_onset_months = round(rng.uniform(18, 72), 1)   # Age 1.5-6 years

    # Seizure history
    has_epilepsy = rng.random() < 0.15   # ~15% PWS epilepsy
    seizures = []
    if has_epilepsy:
        for s, p in zip(SEIZURE_TYPES, SEIZURE_PROBS):
            if rng.random() < p * 6:   # Scale up within epilepsy group
                seizures.append(s)
        if not seizures:
            seizures = ["Focal (unknown onset)"]
    aed_used = rng.choice(["LEV", "LTG", "VPA", "None"]) if has_epilepsy else "None"
    seizure_controlled = rng.random() < 0.85 if has_epilepsy else True   # Good response

    # EEG
    eeg_pattern = rng.choice(EEG_PATTERNS)

    # Sleep
    sleep_apnea = rng.random() < 0.50   # OSA common
    eds = rng.random() < 0.60   # Excessive daytime sleepiness

    # Behavioral
    skin_picking = rng.random() < 0.75
    temper_tantrums = rng.random() < 0.65
    ocd_like = rng.random() < 0.55

    # Cryptorchidism (males)
    is_male = rng.random() < 0.50
    cryptorchidism = is_male and rng.random() < 0.95   # ~95% males

    # Scoliosis
    scoliosis = rng.random() < 0.30

    # Carbetocin trial
    carbetocin_trial = rng.random() < 0.08   # Trial participant

    # Onset/diagnosis
    onset_months = round(rng.uniform(0, 3), 1)   # Neonatal hypotonia
    diagnosis_age_months = round(onset_months + rng.uniform(2, 24), 1)

    return {
        "id":                        f"PWS-{SEED}-{i + 1:02d}",
        "mechanism":                 mechanism,
        "severity_class":            severity_id,
        "deletion_detail":           del_type,
        "iq_estimate":               iq_est,
        "hypotonia_score_0_10":      hypotonia_score,
        "bmi":                       bmi,
        "obesity":                   obesity,
        "t2dm":                      t2dm,
        "gh_deficiency":             gh_deficiency,
        "gh_therapy":                gh_therapy,
        "gh_started_age_months":     gh_started_age_months,
        "hyperphagia_onset_months":  hyperphagia_onset_months,
        "has_epilepsy":              has_epilepsy,
        "seizure_types":             seizures,
        "aed_used":                  aed_used,
        "seizure_controlled":        seizure_controlled,
        "eeg_pattern":               eeg_pattern,
        "sleep_apnea":               sleep_apnea,
        "eds":                       eds,
        "skin_picking":              skin_picking,
        "temper_tantrums":           temper_tantrums,
        "ocd_like":                  ocd_like,
        "asd_features":              asd_features,
        "psychosis_risk":            psychosis_risk,
        "is_male":                   is_male,
        "cryptorchidism":            cryptorchidism,
        "scoliosis":                 scoliosis,
        "carbetocin_trial":          carbetocin_trial,
        "onset_months":              onset_months,
        "diagnosis_age_months":      diagnosis_age_months,
    }


PATIENTS = [_make_patient(i) for i in range(40)]


def get_overview():
    n = len(PATIENTS)

    # Mechanism counts
    del1_n  = sum(1 for p in PATIENTS if "Type 1" in p["mechanism"])
    del2_n  = sum(1 for p in PATIENTS if "Type 2" in p["mechanism"])
    upd_n   = sum(1 for p in PATIENTS if "UPD" in p["mechanism"])
    ic_n    = sum(1 for p in PATIENTS if "Imprinting" in p["mechanism"])
    transl_n = sum(1 for p in PATIENTS if "translocation" in p["mechanism"])

    # Clinical
    epi_n      = sum(1 for p in PATIENTS if p["has_epilepsy"])
    gh_n       = sum(1 for p in PATIENTS if p["gh_therapy"])
    obesity_n  = sum(1 for p in PATIENTS if p["obesity"])
    t2dm_n     = sum(1 for p in PATIENTS if p["t2dm"])
    apnea_n    = sum(1 for p in PATIENTS if p["sleep_apnea"])
    eds_n      = sum(1 for p in PATIENTS if p["eds"])
    skin_n     = sum(1 for p in PATIENTS if p["skin_picking"])
    tantrums_n = sum(1 for p in PATIENTS if p["temper_tantrums"])
    ocd_n      = sum(1 for p in PATIENTS if p["ocd_like"])
    asd_n      = sum(1 for p in PATIENTS if p["asd_features"])
    psychosis_n = sum(1 for p in PATIENTS if p["psychosis_risk"])
    scoliosis_n = sum(1 for p in PATIENTS if p["scoliosis"])
    crypt_n    = sum(1 for p in PATIENTS if p["cryptorchidism"])
    carbetocin_n = sum(1 for p in PATIENTS if p["carbetocin_trial"])

    avg_bmi    = round(sum(p["bmi"] for p in PATIENTS) / n, 1)
    avg_iq     = round(sum(p["iq_estimate"] for p in PATIENTS) / n, 0)
    avg_hypo   = round(sum(p["hypotonia_score_0_10"] for p in PATIENTS) / n, 1)
    diag_delay = round(sum(p["diagnosis_age_months"] - p["onset_months"] for p in PATIENTS) / n, 1)

    return {
        "n_patients":     n,
        "seed":           SEED,
        "disease":        "Prader-Willi Syndrome (PWS)",
        "key_genes":      "SNORD116 (principal), SNRPN, MKRN3, MAGEL2, NDN",
        "locus":          "15q11.2-q13",
        "mechanism":      "Paternal imprinting — PATERNAL expression required; maternal copies silenced",
        "omim_gene":      "*182279 (SNRPN)",
        "omim_disease":   "#176270 (Prader-Willi Syndrome)",
        "prevalence":     "~1:10,000–30,000 live births",
        "inheritance":    "Genomic imprinting — paternal LOF (maternal copies silenced by IC)",
        "mechanism_distribution": {
            "Deletion 15q11.2-q13 Type 1 (bp1-bp3, ~6 Mb)": del1_n,
            "Deletion 15q11.2-q13 Type 2 (bp2-bp3, ~5 Mb)": del2_n,
            "Maternal UPD15 (mUPD15)": upd_n,
            "Imprinting centre defect": ic_n,
            "Chromosomal translocation": transl_n,
        },
        "epilepsy_summary": {
            "any_epilepsy_n":      epi_n,
            "epilepsy_pct":        round(epi_n / n * 100, 1),
            "vs_angelman_pct":     85,
            "note": "Epilepsy rare in PWS (~15%) vs Angelman (~85%) — KEY CONTRAST at same 15q11 locus",
        },
        "gh_therapy": {
            "on_gh_n":             gh_n,
            "gh_pct":              round(gh_n / n * 100, 1),
            "level":               "Level A (FDA approved 2000)",
        },
        "metabolic_features": {
            "obesity_n":           obesity_n,
            "obesity_pct":         round(obesity_n / n * 100, 1),
            "t2dm_n":              t2dm_n,
            "avg_bmi":             avg_bmi,
        },
        "behavioral_features": {
            "skin_picking_n":      skin_n,
            "temper_tantrums_n":   tantrums_n,
            "ocd_like_n":          ocd_n,
            "asd_features_n":      asd_n,
            "psychosis_risk_n":    psychosis_n,
        },
        "clinical_features": {
            "sleep_apnea_n":       apnea_n,
            "eds_n":               eds_n,
            "scoliosis_n":         scoliosis_n,
            "cryptorchidism_n":    crypt_n,
            "carbetocin_trial_n":  carbetocin_n,
        },
        "avg_bmi":                 avg_bmi,
        "avg_iq_estimate":         avg_iq,
        "avg_hypotonia_score_0_10": avg_hypo,
        "avg_diagnosis_delay_months": diag_delay,
        "key_exam_facts": [
            "PWS IMPRINTING: PATERNAL 15q11-q13 expression required; maternal copies silenced by IC — OPPOSITE OF ANGELMAN",
            "SAME LOCUS AS ANGELMAN (15q11-q13): paternal loss = PWS (hyperphagia + obesity); maternal loss = Angelman (severe epilepsy + ataxia)",
            "FOUR MECHANISMS: deletion (60-70%) > mUPD15 (25-30%) > IC defect (1-3%) > translocation (<1%)",
            "SNORD116 MINIMAL REGION: isolated SNORD116 microdeletion reproduces full PWS metabolic phenotype — principal driver",
            "CLINICAL PHASES: (1) neonatal hypotonia + NG tube → (2) hyperphagia onset age 2-6y → (3) morbid obesity if uncontrolled",
            "EPILEPSY ~10-15% (vs Angelman 85%) — mostly focal or generalized; well-controlled; NO pathognomonic EEG pattern",
            "GH THERAPY Level A (FDA 2000): start in infancy even before hyperphagia — improves stature, body composition, cognition",
            "mUPD15 CLASS: 20-30% lifetime PSYCHOSIS risk — highest genetic UPD syndrome risk; more ASD; do NOT miss",
            "VPA HIGH RISK in PWS: obesity → hepatic steatosis → VPA hepatotoxicity risk; weight gain worsens core disease",
            "FOOD SECURITY mandatory: locked refrigerators + pantries; hyperphagia cannot be controlled by willpower alone",
            "CARBETOCIN (oxytocin agonist) Phase III trials 2026: targets hypothalamic satiety pathway; investigational",
            "GH MONITORING: scoliosis progression risk + glucose — annual spine imaging + glucose tolerance test",
            "METHYLATION STUDY (SNRPN locus) FIRST TEST — abnormal methylation (paternal-only pattern) 97-99% sensitive for PWS",
            "mUPD15 vs deletion: SNP array distinguishes — LOH at 15q11 without copy number change = mUPD15",
            "NO HYPOPIGMENTATION in PWS (unlike AS deletion class): maternal OCA2 allele remains active — biallelically expressed",
        ],
    }


def get_breakdown():
    patients_out = []
    for p in PATIENTS:
        patients_out.append({
            "id":                       p["id"],
            "mechanism":                p["mechanism"],
            "severity_class":           p["severity_class"],
            "deletion_detail":          p["deletion_detail"],
            "iq_estimate":              p["iq_estimate"],
            "hypotonia_score_0_10":     p["hypotonia_score_0_10"],
            "bmi":                      p["bmi"],
            "obesity":                  p["obesity"],
            "t2dm":                     p["t2dm"],
            "gh_therapy":               p["gh_therapy"],
            "gh_started_age_months":    p["gh_started_age_months"],
            "hyperphagia_onset_months": p["hyperphagia_onset_months"],
            "has_epilepsy":             p["has_epilepsy"],
            "seizure_types":            p["seizure_types"],
            "aed_used":                 p["aed_used"],
            "seizure_controlled":       p["seizure_controlled"],
            "eeg_pattern":              p["eeg_pattern"],
            "sleep_apnea":              p["sleep_apnea"],
            "eds":                      p["eds"],
            "skin_picking":             p["skin_picking"],
            "temper_tantrums":          p["temper_tantrums"],
            "ocd_like":                 p["ocd_like"],
            "asd_features":             p["asd_features"],
            "psychosis_risk":           p["psychosis_risk"],
            "is_male":                  p["is_male"],
            "cryptorchidism":           p["cryptorchidism"],
            "scoliosis":                p["scoliosis"],
            "carbetocin_trial":         p["carbetocin_trial"],
            "onset_months":             p["onset_months"],
            "diagnosis_age_months":     p["diagnosis_age_months"],
        })

    # Group by mechanism
    mech_groups = {}
    for p in PATIENTS:
        mech_groups.setdefault(p["mechanism"], []).append(p)

    by_mechanism = {}
    for mech, pts in mech_groups.items():
        n_m = len(pts)
        by_mechanism[mech] = {
            "n":                 n_m,
            "epilepsy_pct":      round(sum(1 for x in pts if x["has_epilepsy"]) / n_m * 100, 1),
            "obesity_pct":       round(sum(1 for x in pts if x["obesity"]) / n_m * 100, 1),
            "psychosis_risk_pct": round(sum(1 for x in pts if x["psychosis_risk"]) / n_m * 100, 1),
            "asd_pct":           round(sum(1 for x in pts if x["asd_features"]) / n_m * 100, 1),
            "gh_therapy_pct":    round(sum(1 for x in pts if x["gh_therapy"]) / n_m * 100, 1),
            "avg_bmi":           round(sum(x["bmi"] for x in pts) / n_m, 1),
            "avg_iq":            round(sum(x["iq_estimate"] for x in pts) / n_m, 0),
            "avg_hypotonia":     round(sum(x["hypotonia_score_0_10"] for x in pts) / n_m, 1),
            "avg_diag_delay":    round(sum(x["diagnosis_age_months"] - x["onset_months"] for x in pts) / n_m, 1),
        }

    # AED distribution
    aed_counts = {}
    for p in PATIENTS:
        if p["aed_used"] != "None":
            aed_counts[p["aed_used"]] = aed_counts.get(p["aed_used"], 0) + 1

    # EEG pattern distribution
    eeg_counts = {}
    for p in PATIENTS:
        eeg_counts[p["eeg_pattern"]] = eeg_counts.get(p["eeg_pattern"], 0) + 1

    n = len(PATIENTS)
    clinical_summary = {
        "pct_epilepsy":          round(sum(1 for p in PATIENTS if p["has_epilepsy"]) / n * 100, 1),
        "pct_obesity":           round(sum(1 for p in PATIENTS if p["obesity"]) / n * 100, 1),
        "pct_gh_therapy":        round(sum(1 for p in PATIENTS if p["gh_therapy"]) / n * 100, 1),
        "pct_sleep_apnea":       round(sum(1 for p in PATIENTS if p["sleep_apnea"]) / n * 100, 1),
        "pct_skin_picking":      round(sum(1 for p in PATIENTS if p["skin_picking"]) / n * 100, 1),
        "pct_temper_tantrums":   round(sum(1 for p in PATIENTS if p["temper_tantrums"]) / n * 100, 1),
        "pct_psychosis_risk":    round(sum(1 for p in PATIENTS if p["psychosis_risk"]) / n * 100, 1),
        "pct_asd_features":      round(sum(1 for p in PATIENTS if p["asd_features"]) / n * 100, 1),
        "pct_scoliosis":         round(sum(1 for p in PATIENTS if p["scoliosis"]) / n * 100, 1),
        "avg_bmi":               round(sum(p["bmi"] for p in PATIENTS) / n, 1),
        "avg_iq":                round(sum(p["iq_estimate"] for p in PATIENTS) / n, 0),
    }

    return {
        "patients":          patients_out,
        "by_mechanism":      by_mechanism,
        "aed_counts":        aed_counts,
        "eeg_counts":        eeg_counts,
        "clinical_summary":  clinical_summary,
    }


def get_definitions():
    return {
        "disease_name": "Prader-Willi Syndrome (PWS)",
        "key_genes":    "SNORD116 cluster, SNRPN, MKRN3, MAGEL2, NDN (all paternally expressed, 15q11.2-q13)",
        "locus":        "15q11.2-q13 (PATERNAL expression required)",
        "omim_gene":    "SNRPN *182279",
        "omim_disease": "#176270 (Prader-Willi Syndrome)",
        "inheritance":  "Genomic imprinting — paternal LOF; maternal copies epigenetically silenced",
        "terms": {
            "PWS_imprinting_mechanism": (
                "Prader-Willi Syndrome arises from loss of PATERNALLY expressed genes at 15q11.2-q13. "
                "In normal development, maternal copies of these genes are SILENCED by genomic imprinting "
                "(maternal imprinting centre at 15q11 directs methylation-based silencing of maternal alleles). "
                "CRITICAL: only the PATERNAL copies of SNORD116, SNRPN, MKRN3, MAGEL2, NDN are expressed. "
                "Loss of paternal 15q11-q13 → all these genes absent → Prader-Willi Syndrome. "
                "Contrast: in Angelman Syndrome, MATERNAL 15q11-q13 is lost → UBE3A absent → very different phenotype."
            ),
            "SNORD116_principal_driver": (
                "SNORD116 is a cluster of ~30 C/D box small nucleolar RNAs (snoRNAs) encoded within the "
                "SNORD116 host gene (SNHG14/UBE3A-ATS antisense transcript region) at 15q11.2-q13. "
                "PATERNALLY expressed only. "
                "KEY EVIDENCE: isolated SNORD116 microdeletion (without other PWS genes) → FULL PWS "
                "metabolic phenotype (hyperphagia, obesity, GH deficiency, hypotonia). "
                "Mechanism: SNORD116 snoRNAs regulate pre-mRNA splicing in hypothalamic neurons; "
                "targets include NHLH2 (transcription factor regulating POMC/melanocortin signalling "
                "pathway — critical for satiety). Loss → disrupted satiety circuits."
            ),
            "PWS_vs_Angelman_same_locus": (
                "Prader-Willi Syndrome (PWS) and Angelman Syndrome (AS) arise from the SAME 15q11-q13 locus "
                "but affect the OPPOSITE PARENTAL chromosome: "
                "PWS: PATERNAL 15q11-q13 loss → hypotonia, hyperphagia, obesity, GH deficiency, "
                "hypogonadism, mild-moderate ID; MINIMAL epilepsy (~10-15%). "
                "AS: MATERNAL 15q11-q13 loss (specifically UBE3A) → severe epilepsy (~85%), absent speech, "
                "ataxia, happy demeanor; NO obesity. "
                "EXAM TRAP: the SAME chromosomal region deleted from the FATHER = PWS; from the MOTHER = AS. "
                "No other genetic locus has this property to this degree in clinical genetics."
            ),
            "GH_therapy_PWS": (
                "Growth hormone (GH) therapy is Level A (FDA-approved 2000) for PWS and is transformative: "
                "Indications: GH deficiency confirmed (IGF-1 low/low-normal) in >90% of PWS. "
                "Benefits: (1) Stature: +10-12 cm final height advantage; (2) Body composition: "
                "increases lean mass, reduces fat mass even before hyperphagia dominates; "
                "(3) Energy: improved daytime alertness; (4) Cognitive: modestly improves IQ and adaptive. "
                "Start timing: INFANCY (as early as 2-3 months) before hyperphagia establishes. "
                "Contraindications: severe uncorrected obesity + respiratory failure (sudden death risk); "
                "active malignancy. Monitoring: annual scoliosis (GH accelerates growth → may worsen); "
                "glucose tolerance (GH → insulin resistance); polysomnography."
            ),
            "Hyperphagia_management": (
                "Hyperphagia in PWS is pathological and lifelong — it is a NEUROLOGICAL problem (hypothalamic "
                "dysfunction) not a behavioral choice. Key management principles: "
                "(1) FOOD SECURITY is non-negotiable: locked refrigerators, pantries, bins — patients will "
                "consume anything accessible including frozen/uncooked food, animal food. "
                "(2) Structured meals: 1,000-1,200 kcal/day; 3 meals + 1 snack; no spontaneous access. "
                "(3) CARBETOCIN (oxytocin agonist) Phase III CARE-PWS trial (2022-2026): reduces "
                "food-related distress, hyperphagia scores; intranasal; investigational 2026. "
                "(4) Exercise: mandatory component; reduces BMI by ~2-3 points with structured program. "
                "(5) No pharmacological agent fully suppresses hyperphagia (GLP-1 agonists under study)."
            ),
            "mUPD15_psychosis_risk": (
                "Maternal UPD15 (mUPD15) in PWS — the mUPD15 class carries a 20-30% lifetime risk of "
                "psychosis (schizophrenia spectrum or bipolar-like with psychotic features). "
                "This is the HIGHEST genetic UPD-related psychiatric risk of any imprinting syndrome. "
                "Mechanism: hypothesised loss of paternal gene dosage in limbic-hypothalamic circuits "
                "combined with increased expression of maternally-imprinted genes (UBE3A, etc.) in mUPD15. "
                "Also: mUPD15 class has more ASD features, reduced social motivation. "
                "Clinical implication: MONITOR mUPD15 patients for psychiatric symptoms from adolescence; "
                "low threshold for psychiatric referral. Atypical antipsychotics (aripiprazole) preferred "
                "over typical APs (weight/metabolic concerns)."
            ),
            "VPA_risk_PWS": (
                "Valproate (VPA) in PWS carries HIGH RISK (not absolute contraindication, unlike Angelman). "
                "Mechanism of concern: (1) WEIGHT GAIN — VPA directly promotes weight gain via appetite "
                "stimulation and energy metabolism disruption; in PWS where obesity is the primary lethal "
                "comorbidity, additional weight gain is dangerous; "
                "(2) HEPATOTOXICITY — obesity in PWS → hepatic steatosis/NASH → VPA hepatotoxicity risk "
                "significantly amplified; "
                "(3) HYPERAMMONEMIA — at higher VPA doses, especially with polypharmacy. "
                "Alternative AEDs: LEV or LTG preferred for PWS epilepsy — weight-neutral, hepatically safe. "
                "If VPA truly needed: lowest effective dose + LFT monitoring + ammonia monitoring."
            ),
            "Methylation_study_PWS": (
                "DNA methylation study at SNRPN locus (15q11-q13) — FIRST-LINE test for PWS. "
                "Normal methylation: both methylated (paternal) and unmethylated (maternal) bands. "
                "Abnormal PWS methylation: ONLY unmethylated (maternal-only) band — paternal band absent. "
                "Detects deletion, mUPD15, IC defect in one test (97-99% sensitivity). "
                "Does NOT detect: chromosomal translocation without deletion of 15q11. "
                "After abnormal methylation: "
                "  CMA/FISH → deletion? Type 1 (bp1-bp3) vs Type 2 (bp2-bp3); "
                "  SNP array → LOH at 15q11 without CN change → mUPD15; "
                "  IC sequencing → IC epimutation or microdeletion."
            ),
            "Four_mechanisms_PWS": (
                "Four genetic mechanisms, all resulting in absent paternal SNORD116/SNRPN/MKRN3/MAGEL2/NDN: "
                "(1) Paternal deletion 15q11-q13 (60-70%): de novo; CMA detects; Type 1 (bp1-bp3) vs Type 2 (bp2-bp3). "
                "(2) Maternal UPD15 (25-30%): two maternal copies; methylation abnormal + SNP array LOH. "
                "(3) IC defect (~1-3%): paternal chromosome acquires maternal methylation; IC sequencing. "
                "(4) Chromosomal translocation (<1%): karyotype + CMA. "
                "Recurrence: deletion <1%; mUPD15 <1%; sporadic IC <1%; inherited IC deletion up to 50%."
            ),
            "Diagnosis_workup_PWS": (
                "Step 1: Clinical suspicion — neonatal hypotonia + feeding difficulty + cryptorchidism (males). "
                "Step 2: DNA methylation study (SNRPN locus) — FIRST test; 97-99% sensitivity. "
                "Step 3a (abnormal methylation): CMA/FISH (deletion? type 1 vs 2); SNP array (mUPD15?); "
                "  IC sequencing (IC defect?). "
                "Step 3b (normal methylation): chromosomal karyotype (translocation?); "
                "  consider alternative diagnoses (Angelman-like NDD, Bardet-Biedl, etc.). "
                "Mean diagnosis delay: ~2 years in some series; ideal = neonatal (first weeks of life). "
                "NBS: no standard MS/MS marker; DNA methylation NBS piloted in some programs."
            ),
            "MKRN3_GnRH_puberty": (
                "MKRN3 (Makorin RING finger protein 3) at 15q11.2 — maternally imprinted, paternally expressed. "
                "Normal function: MKRN3 inhibits the GnRH pulse generator in the hypothalamus; "
                "declining MKRN3 expression triggers puberty onset. "
                "In PWS: paternal MKRN3 absent → GnRH pulse generator may be dysregulated. "
                "Clinical: central precocious puberty (CPP) occurs in a subset of PWS patients. "
                "Isolated MKRN3 LOF mutations (outside PWS) cause familial CPP. "
                "Treatment of CPP in PWS: GnRH analog (leuprolide) — Level A for CPP."
            ),
            "Carbetocin_investigational": (
                "Carbetocin (RG7314) is a selective oxytocin receptor agonist under investigation for "
                "hyperphagia and behavioral symptoms in PWS. "
                "Rationale: Oxytocinergic neurons in the hypothalamic paraventricular nucleus (PVN) — "
                "regulated by MAGEL2 and NDN (both absent in PWS) — are critical for satiety signalling. "
                "PWS → reduced hypothalamic oxytocin neurons → impaired satiety → hyperphagia. "
                "CARE-PWS Phase III trial (Roche 2022-2026): intranasal carbetocin; primary endpoint "
                "Hyperphagia Questionnaire for Clinical Trials (HQ-CT) score. "
                "Status: completed enrollment; results expected 2026; NOT yet approved."
            ),
            "Scoliosis_GH_monitoring": (
                "Scoliosis occurs in ~30% of PWS patients and is important for several reasons: "
                "(1) GH therapy accelerates linear growth → rapid spinal growth → may worsen existing "
                "scoliosis or unmask subclinical curves. "
                "(2) Hypotonia → reduced paraspinal muscle tone → scoliosis risk independent of GH. "
                "(3) Obesity → increased mechanical load on already vulnerable spine. "
                "Monitoring: annual spine X-ray (Cobb angle) during GH therapy; "
                "orthopaedic referral if Cobb angle >20° or progressive. "
                "GH is NOT contraindicated in stable scoliosis — benefit outweighs risk; "
                "contraindicated only in progressive severe scoliosis requiring imminent surgery."
            ),
        },
    }
