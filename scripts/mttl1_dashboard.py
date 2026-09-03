#!/usr/bin/env python3
"""MT-TL1 — Mitochondrially Encoded tRNA-Leu (UUR) — MELAS Syndrome
(Mitochondrial Encephalomyopathy, Lactic Acidosis, Stroke-like Episodes) —
MATERNALLY INHERITED DIABETES AND DEAFNESS (MIDD) — CPEO / Exercise Intolerance —
m.3243A>G the MOST COMMON pathogenic mtDNA mutation worldwide.

MT-TL1 (OMIM *590050) encodes the mitochondrial tRNA for leucine (UUR anticodon),
rCRS H-strand positions 3230–3304 (74 bp). This tRNA is ESSENTIAL for translating
ALL 13 mtDNA-encoded OXPHOS subunits — mutations impair mt-translation globally,
causing pan-OXPHOS deficiency (CI + CIII + CIV all depressed; CII—nuclear—often
spared), unlike single-complex defects from protein-coding gene mutations.

m.3243A>G (posi 3243 in TΨC loop of tRNA-Leu) is THE most prevalent disease-causing
mtDNA variant: accounts for ~80% of MELAS syndrome and is found in ~1 in 400 adults
in the general population (including many undiagnosed MIDD carriers).

  MT-TL1 gene            OMIM *590050
  Primary disease        MELAS Syndrome (OMIM #540000)
                         MIDD — Maternally Inherited Diabetes + Deafness
                         CPEO / PEO / Exercise Intolerance
                         Overlap: Leigh-like (rare, very high heteroplasmy)
  Protein product        tRNA-Leu(UUR) — 74 nucleotides; no protein; RNA gene
  Genome                 Mitochondrial DNA (mtDNA), H-strand, rCRS 3230–3304
  Inheritance            MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Key mutation           m.3243A>G — TΨC loop disruption → impaired aminoacylation
                           + mt-translation → pan-OXPHOS deficiency

HETEROPLASMY THRESHOLD (m.3243A>G — blood underestimates by 20-30%):
  <30% blood (≈ <40% tissue):   Asymptomatic carrier / MIDD risk only
  30-60% blood (≈ 40-70% tissue): MIDD (diabetes + SNHL), mild exercise intolerance
  60-80% blood (≈ 70-90% tissue): Partial MELAS / MELAS-like / stroke-like episodes
  >80% blood (≈ >90% tissue):   Classic full MELAS + Leigh overlap risk
  Urinary epithelial cells preferred > blood (20-30% higher heteroplasmy, less drift)

PAN-OXPHOS DEFICIENCY — KEY DISTINGUISHER:
  MT-TL1 mutations impair mt-ribosome aminoacylation of Leu → defective translation
  of ALL 13 mtDNA-encoded OXPHOS subunits → CI + CIII + CIV reduction (CII/SDH
  nuclear-encoded: NORMAL or near-normal → CII/SDH NORMAL is the biochemical
  fingerprint of mt-translation defects including MT-TL1).
  BN-PAGE: CI reduced ± CIII/CIV reduced; CII normal; CV variable.

STROKE-LIKE EPISODES (SLE) — NOT THROMBOTIC:
  SLE in MELAS ≠ ischemic stroke:
  — MRA / MRV NORMAL (no vessel occlusion) — critical DDx from thrombotic stroke
  — DWI: cortical/subcortical T2/FLAIR lesions crossing vascular territories
  — Mechanism: energy failure of neurons → cytotoxic edema + cortical spreading
    depression (NOT vascular occlusion)
  — IV tPA ABSOLUTELY CONTRAINDICATED (no thrombus; bleeding risk)
  — IV L-Arginine: ONLY acute SLE treatment with Level B evidence
    (Koga 2010, Stroke: IV L-Arg 0.5 g/kg in acute SLE → reduces stroke-like damage)
  — Oral L-Arg: Level B maintenance (reduces SLE frequency and severity)

ABSOLUTE CONTRAINDICATIONS:
  Metformin:    Complex I inhibitor → additive with CI deficiency → fatal lactic acidosis
  VPA/Valproate: mt-ribosome inhibition + CoA sequestration + POLG interaction
  IV tPA:       SLE are NOT thrombotic — tPA causes hemorrhagic transformation without benefit
  Linezolid:    mt-23S rRNA inhibitor → directly blocks mt-translation of ALL 13 subunits
  Chloramphenicol: mt-ribosome inhibitor → same mechanism as linezolid
  Propofol:     PRIS + direct ETC inhibition → fatal in MELAS/pan-OXPHOS crisis
  Fasting / prolonged NPO: GIR 6-8 mg/kg/min MANDATORY — NEVER fast in MELAS/Leigh crisis

Key Published References:
  Goto Y et al. (1990) A mutation in the tRNA(Leu)(UUR) gene associated with the
    MELAS subgroup of mitochondrial encephalomyopathies. Nature 348(6302):651-653.
    (m.3243A>G FIRST DESCRIPTION — seminal paper defining MELAS-tRNA link)
  Koga Y et al. (2010) L-Arginine improves the symptoms of strokelike episodes in
    MELAS. Neurology 64(4):710-712; and Stroke 41(7):1579-1586.
    (L-Arginine IV acute SLE Level B; oral maintenance Level B)
  El-Hattab AW et al. (2012) Restoration of impaired nitric oxide production in MELAS
    syndrome with citrulline and arginine supplementation. Mol Genet Metab 105(4):607-614.
  Gorman GS et al. (2016) Prevalence of nuclear and mitochondrial DNA mutations related
    to adult mitochondrial disease. Ann Neurol 79(4):589-591.
    (m.3243A>G ~1 in 400 adults carry; population prevalence overview)
  Manwaring N et al. (2007) Population prevalence of the MELAS A3243G mutation.
    Mitochondrion 7(3):230-233.
    (Urinary epithelial cell testing preferred; blood heteroplasmy underestimates)
"""

import random

SEED = 783
N_PATIENTS = 40

# ── Pathogenic / likely-pathogenic variants in MT-TL1 ─────────────────────────
VARIANTS = [
    {
        "hgvs_mtdna": "m.3243A>G",
        "domain": "TΨC loop of tRNA-Leu(UUR) / mt-ribosome aminoacylation site",
        "type": "tRNA point mutation — TΨC loop A→G; impairs Leu-tRNA aminoacylation efficiency; reduces mt-translation of ALL 13 OXPHOS subunits",
        "severity": "Variable — heteroplasmy-dependent: <40% tissue = MIDD/mild; >70% tissue = MELAS",
        "phenotype": "MELAS / MIDD / Exercise Intolerance / CPEO (heteroplasmy-dependent)",
        "penetrance_pct": 90,
        "frequency_pct": 80,
        "notes": "~80% of MELAS cohort; ~1 in 400 adults carry m.3243A>G (population prevalence); THE most prevalent disease-causing mtDNA variant; TΨC loop A3243G disrupts the aminoacylation recognition site for mitochondrial leucyl-tRNA synthetase (LARS2); impairs Leu incorporation → premature termination of OXPHOS subunit translation; pan-OXPHOS defect (CI+CIII+CIV); CII/SDH NORMAL (nuclear, not affected); heteroplasmy threshold (blood): <30% = MIDD risk/asymptomatic; 30-60% = MIDD+SNHL; >60% = MELAS; CRITICAL: urinary epithelial cells show 20-30% HIGHER heteroplasmy than blood — must use urine for accurate threshold assessment; blood heteroplasmy UNDERESTIMATES due to clonal haematopoiesis; maternal family cascade testing mandatory; WES MISSES (mtDNA); dedicated mtDNA panel required.",
    },
    {
        "hgvs_mtdna": "m.3271T>C",
        "domain": "Anticodon stem of tRNA-Leu(UUR) / stem-loop structure",
        "type": "tRNA point mutation — anticodon stem T→C; disrupts tRNA secondary structure; moderate mt-translation impairment",
        "severity": "Moderate-Severe — MELAS-like / stroke-like episodes (smaller cohort than m.3243A>G)",
        "phenotype": "MELAS-like / stroke-like episodes / exercise intolerance",
        "penetrance_pct": 75,
        "frequency_pct": 7,
        "notes": "~7% of MELAS cohort; m.3271T>C in the anticodon stem of tRNA-Leu(UUR) — disrupts tRNA secondary structure stability; moderate pan-OXPHOS deficiency (less severe than m.3243A>G but similar phenotypic spectrum); MELAS-like with stroke-like episodes, lactic acidosis; SNHL common; diabetes less frequent than m.3243A>G; muscle biopsy: RRF + COX-negative fibres; heteroplasmy threshold similar to m.3243A>G (>70% tissue for MELAS); maternal inheritance; WES misses; dedicted mtDNA panel required.",
    },
    {
        "hgvs_mtdna": "m.3251A>G",
        "domain": "Variable loop region of tRNA-Leu(UUR)",
        "type": "tRNA point mutation — variable loop A→G; mild structural perturbation; partial mt-translation reduction",
        "severity": "Mild-Moderate — exercise intolerance / myopathy / mild MELAS",
        "phenotype": "Exercise Intolerance / Mild MELAS / Myopathy",
        "penetrance_pct": 60,
        "frequency_pct": 4,
        "notes": "~4% of MT-TL1 disease cohort; m.3251A>G in the variable loop — mildest structural impact of the common MT-TL1 variants; partial pan-OXPHOS deficiency; predominantly exercise intolerance + myopathy; full MELAS rare; stroke-like episodes uncommon; SNHL in ~40%; diabetes in ~25%; adult onset common; RRF on Gomori; isolated resting lactic acidosis mild to moderate; useful for tracking disease: lower heteroplasmy typical; may be found on MELAS screening panels; maternal inheritance; WES misses; mtDNA panel required.",
    },
    {
        "hgvs_mtdna": "m.3243A>G (low heteroplasmy — MIDD phenotype)",
        "domain": "TΨC loop — same locus as MELAS, different heteroplasmy tier",
        "type": "Same m.3243A>G locus — low blood heteroplasmy (20-45%) → MIDD not MELAS phenotype",
        "severity": "Mild — MIDD: maternally inherited diabetes + sensorineural deafness; no stroke-like episodes",
        "phenotype": "MIDD — Maternally Inherited Diabetes and Deafness",
        "penetrance_pct": 65,
        "frequency_pct": 9,
        "notes": "~9% of clinical MT-TL1 presentations are MIDD phenotype (low-heteroplasmy m.3243A>G carriers); blood heteroplasmy typically 15-40% in MIDD vs >60% in MELAS; MIDD: type 2 diabetes-like presentation (onset 20-40y) + progressive SNHL; NO stroke-like episodes; NO encephalopathy; RRF may be absent; often misdiagnosed as T2DM + age-related hearing loss; maternal pedigree of diabetes + deafness is the key clue; urine heteroplasmy typically 30-65% (higher than blood); MUST exclude MIDD before starting metformin for 'T2DM' — Metformin ABSOLUTE CI in any m.3243A>G carrier regardless of phenotype (lactic acidosis risk even without MELAS); maternal cascade testing MANDATORY; MIDD with MELAS risk increases as heteroplasmy rises with age (heteroplasmy can increase in some tissues).",
    },
    {
        "hgvs_mtdna": "Large mtDNA deletion (MT-TL1-spanning)",
        "domain": "Partial or complete MT-TL1 locus (rCRS 3230-3304) ± adjacent MT-ND1/MT-RNR2",
        "type": "Large deletion — disrupts tRNA-Leu gene; pan-OXPHOS (deletion removes multiple OXPHOS subunit genes)",
        "severity": "Variable",
        "phenotype": "KSS / CPEO / Pearson Syndrome (large mtDNA deletion syndrome)",
        "penetrance_pct": 55,
        "frequency_pct": 5,
        "notes": "~5% of MT-TL1 disease spectrum; large-scale mtDNA deletions spanning the MT-TL1 locus (rCRS 3230-3304) — typically also remove adjacent MT-ND1 (3307-4262), MT-RNR2 (16S rRNA), and/or other genes; KSS: CPEO + pigmentary retinopathy + cardiomyopathy onset <20y; Pearson: infantile sideroblastic anaemia + exocrine pancreatic insufficiency; CPEO: isolated progressive external ophthalmoplegia; large deletions cause multi-complex OXPHOS deficiency (broader than single tRNA mutation); WES misses; long-read sequencing or Southern blot required; KSS annual Holter MANDATORY — cardiac conduction block risk; pacemaker threshold: PR >240 ms or Mobitz II.",
    },
]

VARIANT_WEIGHTS = [0.80, 0.07, 0.04, 0.09, 0.05]
# Note: MIDD (idx=3) and deletion (idx=4) share the last 14%, renormalized below
# Actually let's just assign 40% to MELAS, etc. and let weights sum >1 be renormalized
# Correcting: 0.80 + 0.07 + 0.04 + 0.09 + 0.05 = 1.05 — need to fix
# But m.3243A>G is the main mutation; MIDD is same variant at low heteroplasmsy level
# Let's use: 0.60 m.3243A>G MELAS, 0.07 m.3271T>C, 0.04 m.3251A>G, 0.19 MIDD-low, 0.10 deletion
VARIANT_WEIGHTS = [0.60, 0.07, 0.04, 0.19, 0.10]


# ── Patient-cohort generator ──────────────────────────────────────────────────
def _make_patients(n=N_PATIENTS, seed=SEED):
    rng = random.Random(seed)
    patients = []

    outcomes = [
        ("Deceased < 10 years (MELAS severe, refractory SLE + lactic crisis)", 0.08),
        ("Deceased 10–30 years (MELAS, recurrent SLE, respiratory failure)", 0.10),
        ("Deceased adult (MELAS late complications — cardiomyopathy/aspiration)", 0.06),
        ("Alive — MELAS severe, wheelchair, significant disability", 0.12),
        ("Alive — MELAS moderate, recurrent SLE managed, partial independence", 0.16),
        ("Alive — MELAS mild/partial, infrequent SLE, working", 0.14),
        ("Alive — MIDD only, diabetes + SNHL managed, no SLE", 0.14),
        ("Alive — Exercise intolerance / myopathy dominant, working", 0.10),
        ("Alive — CPEO/KSS, pacemaker, managed", 0.07),
        ("Alive — Asymptomatic carrier (low heteroplasmy, monitoring)", 0.03),
    ]

    for i in range(n):
        v_idx = rng.choices(range(len(VARIANTS)), weights=VARIANT_WEIGHTS)[0]
        variant = VARIANTS[v_idx]

        is_melas       = variant["hgvs_mtdna"] == "m.3243A>G"
        is_m3271       = variant["hgvs_mtdna"] == "m.3271T>C"
        is_m3251       = variant["hgvs_mtdna"] == "m.3251A>G"
        is_midd        = "MIDD phenotype" in variant["hgvs_mtdna"]   # index-3 MIDD entry only
        is_deletion    = "deletion" in variant["hgvs_mtdna"]

        # Heteroplasmy (urine epithelial — preferred test)
        if is_midd:
            heteroplasmy_urine_pct = round(rng.uniform(25, 58), 1)
            heteroplasmy_blood_pct = round(rng.uniform(15, 42), 1)
        elif is_melas:
            heteroplasmy_urine_pct = round(rng.uniform(65, 99), 1)
            heteroplasmy_blood_pct = round(rng.uniform(45, 92), 1)
        elif is_m3271:
            heteroplasmy_urine_pct = round(rng.uniform(60, 95), 1)
            heteroplasmy_blood_pct = round(rng.uniform(42, 88), 1)
        elif is_m3251:
            heteroplasmy_urine_pct = round(rng.uniform(40, 80), 1)
            heteroplasmy_blood_pct = round(rng.uniform(28, 68), 1)
        elif is_deletion:
            heteroplasmy_urine_pct = round(rng.uniform(15, 55), 1)
            heteroplasmy_blood_pct = round(rng.uniform(10, 45), 1)
        else:
            heteroplasmy_urine_pct = round(rng.uniform(50, 90), 1)
            heteroplasmy_blood_pct = round(rng.uniform(35, 78), 1)

        # Phenotype label
        if is_deletion:
            pheno_label = rng.choices(
                ["Pearson Syndrome (infantile deletion)",
                 "Kearns-Sayre Syndrome (KSS)",
                 "CPEO (adult onset)"],
                weights=[0.15, 0.55, 0.30],
            )[0]
        elif is_midd:
            pheno_label = "MIDD — Maternally Inherited Diabetes and Deafness"
        elif is_m3251:
            pheno_label = rng.choices(
                ["Exercise Intolerance / Myopathy",
                 "Mild MELAS / partial MELAS phenotype"],
                weights=[0.70, 0.30],
            )[0]
        elif is_melas or is_m3271:
            pheno_label = rng.choices(
                ["Classic MELAS (stroke-like episodes + encephalopathy + LA)",
                 "Partial MELAS / MELAS-like (SLE without full syndrome)",
                 "MELAS + MIDD overlap (diabetes + SLE + SNHL)"],
                weights=[0.55, 0.30, 0.15],
            )[0]
        else:
            pheno_label = "MELAS / partial"

        # CI activity (pan-OXPHOS)
        if "Classic MELAS" in pheno_label:
            ci_activity_pct = round(rng.uniform(8, 30), 1)
        elif "Partial MELAS" in pheno_label or "MELAS-like" in pheno_label:
            ci_activity_pct = round(rng.uniform(18, 45), 1)
        elif "MIDD" in pheno_label:
            ci_activity_pct = round(rng.uniform(38, 72), 1)
        elif "Exercise Intolerance" in pheno_label:
            ci_activity_pct = round(rng.uniform(30, 65), 1)
        elif "CPEO" in pheno_label or "KSS" in pheno_label:
            ci_activity_pct = round(rng.uniform(15, 48), 1)
        elif "Pearson" in pheno_label:
            ci_activity_pct = round(rng.uniform(8, 28), 1)
        else:
            ci_activity_pct = round(rng.uniform(20, 55), 1)

        # Lactic acid
        if "Classic MELAS" in pheno_label:
            lactic_acid = round(rng.uniform(4.5, 18.0), 1)
        elif "Partial MELAS" in pheno_label:
            lactic_acid = round(rng.uniform(2.5, 8.0), 1)
        elif "MIDD" in pheno_label:
            lactic_acid = round(rng.uniform(1.2, 3.5), 1)
        elif "Exercise Intolerance" in pheno_label:
            lactic_acid = round(rng.uniform(1.8, 5.0), 1)
        elif "KSS" in pheno_label or "CPEO" in pheno_label:
            lactic_acid = round(rng.uniform(2.0, 6.0), 1)
        elif "Pearson" in pheno_label:
            lactic_acid = round(rng.uniform(3.5, 12.0), 1)
        else:
            lactic_acid = round(rng.uniform(2.0, 7.0), 1)

        # Clinical features
        stroke_like_episode = rng.random() < (
            0.90 if "Classic MELAS" in pheno_label else
            0.65 if "Partial MELAS" in pheno_label else
            0.12 if "MELAS + MIDD" in pheno_label else
            0.03
        )
        encephalopathy = rng.random() < (
            0.85 if "Classic MELAS" in pheno_label else
            0.50 if "Partial MELAS" in pheno_label else
            0.08
        )
        seizures = rng.random() < (
            0.80 if "Classic MELAS" in pheno_label else
            0.45 if "Partial MELAS" in pheno_label else
            0.10
        )
        sensorineural_hearing_loss = rng.random() < (
            0.80 if "Classic MELAS" in pheno_label else
            0.65 if "Partial MELAS" in pheno_label else
            0.85 if "MIDD" in pheno_label else
            0.42 if "CPEO" in pheno_label or "KSS" in pheno_label else
            0.45 if "Exercise Intolerance" in pheno_label else
            0.35
        )
        diabetes_mellitus = rng.random() < (
            0.55 if "Classic MELAS" in pheno_label else
            0.38 if "Partial MELAS" in pheno_label else
            0.88 if "MIDD" in pheno_label else
            0.12
        )
        lactic_acidosis = rng.random() < (
            0.92 if "Classic MELAS" in pheno_label else
            0.62 if "Partial MELAS" in pheno_label else
            0.20 if "MIDD" in pheno_label else
            0.45 if "Exercise Intolerance" in pheno_label else
            0.50 if "KSS" in pheno_label else 0.22
        )
        exercise_intolerance = rng.random() < (
            0.85 if "Classic MELAS" in pheno_label else
            0.78 if "Partial MELAS" in pheno_label else
            0.55 if "MIDD" in pheno_label else
            0.95 if "Exercise Intolerance" in pheno_label else
            0.60 if "KSS" in pheno_label else 0.30
        )
        myopathy = rng.random() < (
            0.80 if "Classic MELAS" in pheno_label else
            0.65 if "Partial MELAS" in pheno_label else
            0.25 if "MIDD" in pheno_label else
            0.92 if "Exercise Intolerance" in pheno_label else 0.35
        )
        cardiomyopathy = rng.random() < (
            0.30 if "Classic MELAS" in pheno_label else
            0.18 if "Partial MELAS" in pheno_label else
            0.08 if "MIDD" in pheno_label else
            0.40 if "KSS" in pheno_label else 0.05
        )
        cardiac_conduction_defect = rng.random() < (
            0.12 if "Classic MELAS" in pheno_label else
            0.40 if "KSS" in pheno_label else 0.05
        )
        ophthalmoplegia = rng.random() < (
            0.22 if "Classic MELAS" in pheno_label else
            0.75 if "CPEO" in pheno_label or "KSS" in pheno_label else 0.08
        )
        cpeo = rng.random() < (
            0.22 if "Classic MELAS" in pheno_label else
            0.72 if "CPEO" in pheno_label or "KSS" in pheno_label else 0.05
        )
        retinitis_pigmentosa = rng.random() < (
            0.35 if "KSS" in pheno_label else 0.04
        )
        ptosis = rng.random() < (
            0.20 if "Classic MELAS" in pheno_label else
            0.65 if "CPEO" in pheno_label or "KSS" in pheno_label else 0.08
        )
        cerebellar_ataxia = rng.random() < (
            0.55 if "Classic MELAS" in pheno_label else
            0.35 if "Partial MELAS" in pheno_label else 0.12
        )
        dementia_cognitive_decline = rng.random() < (
            0.45 if "Classic MELAS" in pheno_label else
            0.22 if "Partial MELAS" in pheno_label else 0.05
        )
        melas_mri_cortical = rng.random() < (
            0.88 if "Classic MELAS" in pheno_label else
            0.55 if "Partial MELAS" in pheno_label else 0.05
        )
        leigh_mri = rng.random() < (
            0.12 if "Classic MELAS" in pheno_label and rng.random() < 0.15 else 0.04
        )
        ragged_red_fibres = rng.random() < (
            0.75 if "Classic MELAS" in pheno_label else
            0.55 if "Partial MELAS" in pheno_label else
            0.30 if "MIDD" in pheno_label else
            0.65 if "Exercise Intolerance" in pheno_label else
            0.58 if "KSS" in pheno_label else 0.20
        )
        respiratory_failure = rng.random() < (
            0.38 if "Classic MELAS" in pheno_label else
            0.08 if "Partial MELAS" in pheno_label else 0.03
        )

        male_sex = rng.random() < 0.50  # maternal mtDNA; no sex predominance in MELAS

        outcome_label = rng.choices(
            [o[0] for o in outcomes], weights=[o[1] for o in outcomes]
        )[0]

        # Onset age
        if "Pearson" in pheno_label:
            onset_weeks = rng.randint(1, 12)
        elif "Classic MELAS" in pheno_label:
            onset_weeks = rng.randint(104, 936)    # 2-18 years typical
        elif "Partial MELAS" in pheno_label:
            onset_weeks = rng.randint(208, 1560)   # 4-30 years
        elif "MIDD" in pheno_label:
            onset_weeks = rng.randint(780, 2600)   # 15-50 years
        elif "Exercise Intolerance" in pheno_label:
            onset_weeks = rng.randint(520, 2600)   # 10-50 years
        elif "KSS" in pheno_label:
            onset_weeks = rng.randint(52, 1040)
        elif "CPEO" in pheno_label:
            onset_weeks = rng.randint(520, 2600)
        else:
            onset_weeks = rng.randint(104, 1560)

        patients.append({
            "patient_id": f"MTTL1-{i+1:03d}",
            "phenotype": pheno_label,
            "variant": variant["hgvs_mtdna"],
            "heteroplasmy_urine_pct": heteroplasmy_urine_pct,
            "heteroplasmy_blood_pct": heteroplasmy_blood_pct,
            "ci_activity_pct": ci_activity_pct,
            "lactic_acid_mmolL": lactic_acid,
            "onset_weeks": onset_weeks,
            "male_sex": male_sex,
            "stroke_like_episode": stroke_like_episode,
            "encephalopathy": encephalopathy,
            "seizures": seizures,
            "sensorineural_hearing_loss": sensorineural_hearing_loss,
            "diabetes_mellitus": diabetes_mellitus,
            "lactic_acidosis": lactic_acidosis,
            "exercise_intolerance": exercise_intolerance,
            "myopathy": myopathy,
            "cardiomyopathy": cardiomyopathy,
            "cardiac_conduction_defect": cardiac_conduction_defect,
            "ophthalmoplegia": ophthalmoplegia,
            "cpeo": cpeo,
            "retinitis_pigmentosa": retinitis_pigmentosa,
            "ptosis": ptosis,
            "cerebellar_ataxia": cerebellar_ataxia,
            "dementia_cognitive_decline": dementia_cognitive_decline,
            "melas_mri_cortical": melas_mri_cortical,
            "leigh_mri": leigh_mri,
            "ragged_red_fibres": ragged_red_fibres,
            "respiratory_failure": respiratory_failure,
            "outcome": outcome_label,
        })

    return patients


# ── Cohort statistics ─────────────────────────────────────────────────────────
def _cohort_stats(patients):
    n = len(patients)
    pct = lambda key: round(100 * sum(1 for p in patients if p.get(key, False)) / n, 1)
    avg = lambda key: round(sum(p[key] for p in patients) / n, 1)
    return {
        "stroke_like_episode_pct":          pct("stroke_like_episode"),
        "encephalopathy_pct":               pct("encephalopathy"),
        "seizures_pct":                     pct("seizures"),
        "sensorineural_hearing_loss_pct":   pct("sensorineural_hearing_loss"),
        "diabetes_mellitus_pct":            pct("diabetes_mellitus"),
        "lactic_acidosis_pct":              pct("lactic_acidosis"),
        "exercise_intolerance_pct":         pct("exercise_intolerance"),
        "myopathy_pct":                     pct("myopathy"),
        "cardiomyopathy_pct":               pct("cardiomyopathy"),
        "cerebellar_ataxia_pct":            pct("cerebellar_ataxia"),
        "melas_mri_cortical_pct":           pct("melas_mri_cortical"),
        "ragged_red_fibres_pct":            pct("ragged_red_fibres"),
        "respiratory_failure_pct":          pct("respiratory_failure"),
        "cpeo_pct":                         pct("cpeo"),
        "ophthalmoplegia_pct":              pct("ophthalmoplegia"),
        "retinitis_pigmentosa_pct":         pct("retinitis_pigmentosa"),
        "avg_ci_activity_pct":              avg("ci_activity_pct"),
        "avg_lactic_acid_mmolL":            avg("lactic_acid_mmolL"),
        "avg_heteroplasmy_urine_pct":       avg("heteroplasmy_urine_pct"),
        "avg_heteroplasmy_blood_pct":       avg("heteroplasmy_blood_pct"),
        "deceased_pct": round(
            100 * sum(1 for p in patients if "Deceased" in p["outcome"]) / n, 1
        ),
    }


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    patients = _make_patients()
    stats = _cohort_stats(patients)
    n = len(patients)

    feat_order = [
        ("Stroke-like episodes (SLE — NOT thrombotic, IV tPA CI)", "stroke_like_episode"),
        ("Exercise intolerance", "exercise_intolerance"),
        ("Myopathy", "myopathy"),
        ("Sensorineural hearing loss (SNHL)", "sensorineural_hearing_loss"),
        ("Lactic acidosis", "lactic_acidosis"),
        ("Seizures", "seizures"),
        ("Encephalopathy", "encephalopathy"),
        ("Diabetes mellitus (MIDD component)", "diabetes_mellitus"),
        ("Cerebellar ataxia", "cerebellar_ataxia"),
        ("Ragged red fibres (RRF) on Gomori", "ragged_red_fibres"),
        ("Cardiomyopathy", "cardiomyopathy"),
        ("MELAS cortical MRI lesions (crossing vascular territories)", "melas_mri_cortical"),
        ("CPEO (progressive external ophthalmoplegia)", "cpeo"),
        ("Respiratory failure", "respiratory_failure"),
        ("Ptosis", "ptosis"),
        ("Cognitive decline / dementia", "dementia_cognitive_decline"),
        ("Retinitis Pigmentosa (deletion/KSS only — absent in m.3243A>G)", "retinitis_pigmentosa"),
    ]
    features = [
        {
            "feature": label,
            "pct": round(100 * sum(1 for p in patients if p.get(key, False)) / n, 1),
        }
        for label, key in feat_order
    ]

    from collections import Counter
    v_counter = Counter(p["variant"] for p in patients)
    top_variants = [{"variant": k, "count": v} for k, v in v_counter.most_common(5)]

    alerts = [
        "🔴 IV tPA ABSOLUTE CI IN SLE: MELAS stroke-like episodes are NOT thrombotic — MRA/MRV shows NO vessel occlusion; mechanism is cytotoxic energy failure (cortical spreading depression + metabolic crisis), NOT vascular occlusion; IV tPA causes hemorrhagic transformation without benefit; FATAL if given; always check MELAS diagnosis before any stroke workup",
        "🔴 METFORMIN ABSOLUTE CI: ALL m.3243A>G carriers (MELAS + MIDD) — Complex I inhibitor → additive with CI deficiency → fatal lactic acidosis; MIDD patients are commonly prescribed metformin for 'T2DM' — this is a critical medication error; insulin is the SAFE diabetes treatment in MIDD/m.3243A>G",
        "🔴 VPA ABSOLUTE CI: Valproate in MELAS/MT-TL1 — mt-ribosome inhibition + CoA sequestration + POLG interaction; worsens pan-OXPHOS; hepatotoxicity risk; use LEV (levetiracetam) for seizures",
        "⚠️ L-ARGININE IV IN ACUTE SLE — Level B (Koga 2010): IV L-Arginine 0.5 g/kg (max 10g) over 30-60 min at SLE onset improves cerebral blood flow via NO synthesis; reduces ischaemic damage of SLE; oral L-Arg maintenance reduces SLE frequency; ONLY evidence-based treatment for acute SLE",
        "⚠️ URINE > BLOOD for heteroplasmy: m.3243A>G blood heteroplasmy UNDERESTIMATES by 20-30% due to clonal haematopoiesis (affected cells cleared); ALWAYS use urinary epithelial cells for accurate heteroplasmy threshold assessment; urine test: spot urine for urinary epithelial cell pellet mtDNA PCR",
        "⚠️ PAN-OXPHOS BIOCHEMICAL FINGERPRINT: MT-TL1 mutations impair mt-ribosome → CI+CIII+CIV all reduced; CII (SDH, nuclear-encoded) NORMAL; this distinguishes from single-complex defects (MT-ND1-6=CI only, MT-CO1-3=CIV only, MT-ATP6/8=CV only); BN-PAGE shows multi-complex reduction; SDH normal is the hallmark of mt-translation defects",
        "⚠️ MELAS MRI vs ISCHEMIC STROKE: MELAS cortical lesions cross arterial vascular territories; involve cortex + subcortex; may appear/resolve with metabolic state; MRA/MRV normal (no vessel occlusion); MRS shows lactate peak in lesion; DWI: cytotoxic pattern; contrast to Leigh (bilateral BG/brainstem) and to ischemic stroke (territory-specific, MRA abnormal)",
        "🔵 ABSOLUTE CI: Metformin (CI+lactic acidosis) · VPA (mt-ribosome+CoA) · IV tPA (SLE not thrombotic) · Linezolid (mt-23S rRNA→impairs ALL OXPHOS synthesis) · Chloramphenicol (mt-ribosome) · Propofol (PRIS+ETC) · Fasting (GIR 6-8 MANDATORY in crisis)",
        "🔵 THIAMINE B1 + BIOTIN + CoQ10 MANDATORY: Empiric thiamine B1 IV in metabolic crisis · Biotin pending BTBGD exclusion · CoQ10 ubiquinol Level C · Riboflavin B2 Level C · L-Carnitine Level C · L-Arginine oral Level B maintenance",
        "🔵 WES MISSES MT-TL1: MT-TL1 is a mitochondrial tRNA gene (rCRS H-strand 3230-3304); WES misses all mtDNA variants and tRNA-structural mutations by design; dedicated mtDNA sequencing required; urinary epithelial cell PCR for m.3243A>G quantitation; muscle biopsy for histochemistry (RRF) + respiratory chain enzymology (pan-OXPHOS vs single complex)",
    ]

    pheno_dist = Counter(p["phenotype"] for p in patients)
    phenotype_distribution = [
        {"phenotype": k, "count": v}
        for k, v in pheno_dist.most_common()
    ]

    onset_dist = {
        "neonatal_0_3m": 0,
        "infantile_3_24m": 0,
        "childhood_2_18y": 0,
        "adult_18y_plus": 0,
    }
    for p in patients:
        w = p["onset_weeks"]
        if w < 13:
            onset_dist["neonatal_0_3m"] += 1
        elif w < 104:
            onset_dist["infantile_3_24m"] += 1
        elif w < 936:
            onset_dist["childhood_2_18y"] += 1
        else:
            onset_dist["adult_18y_plus"] += 1

    key_molecular_features = [
        {
            "feature": "tRNA-Leu(UUR) — 74 nt RNA gene — encodes all 13 OXPHOS subunit leucine residues → pan-OXPHOS when disrupted",
            "value": "74 nt / tRNA",
            "significance": "MT-TL1 is a NON-PROTEIN-CODING gene; it encodes the mitochondrial tRNA for leucine (UUR anticodon); unlike protein-coding genes, a single tRNA mutation impairs translation of ALL 13 mtDNA-encoded OXPHOS subunits simultaneously → pan-OXPHOS deficiency (CI+CIII+CIV); CII/SDH (nuclear-encoded) spared — CII NORMAL is the biochemical fingerprint of mt-translation defects vs structural single-subunit mutations"
        },
        {
            "feature": "m.3243A>G — most common pathogenic mtDNA variant — ~1 in 400 adults carry it (population prevalence, Gorman 2016)",
            "value": "1 in 400 prevalence",
            "significance": "m.3243A>G is the single most prevalent disease-causing mtDNA mutation worldwide; ~80% of MELAS syndrome; ~1 in 400 adults carry it in the general population (many undiagnosed MIDD); clinical spectrum is entirely determined by heteroplasmy — the same nucleotide change causes full MELAS at >70% tissue heteroplasmy, MIDD at 30-60%, and silent carrier at <30%"
        },
        {
            "feature": "Heteroplasmy threshold — urine > blood — MIDD vs MELAS continuum (same m.3243A>G locus)",
            "value": "Urine >20-30% vs blood",
            "significance": "Blood heteroplasmy for m.3243A>G underestimates by 20-30% due to clonal selection (mitotically-active blood cells clear mutant mtDNA over time); urinary epithelial cells maintain stable heteroplasmy and are the preferred test; threshold: >70% tissue → MELAS; 30-60% tissue → MIDD (diabetes+deafness, NO stroke-like episodes); <30% tissue → often asymptomatic; blood can falsely appear 'low' in adults with MELAS — always confirm with urine"
        },
        {
            "feature": "SLE (stroke-like episodes) — NOT thrombotic — IV tPA ABSOLUTE CI — IV L-Arginine Level B",
            "value": "SLE ≠ ischemic stroke",
            "significance": "MELAS stroke-like episodes are caused by energy failure of neurons → cytotoxic oedema + cortical spreading depression; MRA/MRV shows NO vessel occlusion; lesions cross arterial territories; IV tPA is ABSOLUTELY CONTRAINDICATED (no thrombus; haemorrhagic transformation risk); IV L-Arginine (0.5 g/kg over 30-60 min) is the ONLY evidence-based acute treatment (Koga 2010, Stroke — Level B); NO calcium-channel blockers in acute SLE (worsen vascular NO)"
        },
        {
            "feature": "Pan-OXPHOS (CI+CIII+CIV reduced; CII NORMAL) — BN-PAGE fingerprint — distinguishes from single-complex defects",
            "value": "Pan-OXPHOS / CII normal",
            "significance": "BN-PAGE: Complex I, III, and IV all reduced in proportion to heteroplasmy; Complex II (succinate dehydrogenase, SDH) is nuclear-encoded and NOT affected → CII NORMAL is the biochemical fingerprint distinguishing mt-translation defects (MT-TL1, MT-TK, MTFMT) from structural subunit mutations (MT-ND1-6 = CI only; MT-CO1-3 = CIV only; MT-ATP6/8 = CV only); SDH staining NORMAL on histochemistry even when COX-negative fibres present"
        },
    ]

    heteroplasmy_clinical_map = [
        {"tier": "<30% blood (<40% tissue)", "phenotype": "Asymptomatic carrier / MIDD risk only", "sle_risk": "None", "metformin_ci": "YES"},
        {"tier": "30-45% blood (40-55% tissue)", "phenotype": "MIDD — diabetes + SNHL; no stroke-like episodes", "sle_risk": "None/minimal", "metformin_ci": "YES"},
        {"tier": "45-60% blood (55-70% tissue)", "phenotype": "MIDD+ / partial MELAS; exercise intolerance", "sle_risk": "Low", "metformin_ci": "YES"},
        {"tier": "60-80% blood (70-85% tissue)", "phenotype": "Partial MELAS / MELAS-like; SLE possible", "sle_risk": "Moderate", "metformin_ci": "YES"},
        {"tier": ">80% blood (>90% tissue)", "phenotype": "Classic full MELAS; recurrent SLE + encephalopathy + LA", "sle_risk": "High", "metformin_ci": "YES"},
    ]

    return {
        "gene": "MT-TL1",
        "omim_gene": "OMIM *590050",
        "protein": "tRNA-Leu(UUR) — 74 nucleotides; RNA gene (no protein); encodes mitochondrial tRNA for Leucine (UUR anticodon)",
        "module": "mt-translation — tRNA (no OXPHOS complex; impairs synthesis of ALL 13 mtDNA-encoded OXPHOS subunits when mutated)",
        "inheritance": "MATERNAL (mtDNA heteroplasmic — clinical phenotype entirely determined by heteroplasmy level; urine > blood for accurate measurement)",
        "primary_disease": "MELAS Syndrome (OMIM #540000) / MIDD (Maternally Inherited Diabetes and Deafness) / CPEO / Exercise Intolerance",
        "key_mutation": {
            "hgvs": "m.3243A>G",
            "frequency": "~80% of MELAS; ~1 in 400 adults (population carrier frequency)",
            "location": "TΨC loop of tRNA-Leu(UUR) rCRS position 3243",
            "mechanism": "Disrupts aminoacylation of Leu-tRNA → defective mt-ribosome translation → pan-OXPHOS deficiency",
        },
        "rcrs_positions": "3230-3304",
        "strand": "H-strand",
        "trna_type": "tRNA-Leu (UUR anticodon) — 74 nucleotides",
        "pan_oxphos": "CI + CIII + CIV all reduced (CII/SDH NORMAL — nuclear-encoded, not translated by mt-ribosomes)",
        "n_patients": n,
        "seed": SEED,
        "cohort_statistics": stats,
        "cohort_summary_features": features,
        "phenotype_distribution": phenotype_distribution,
        "onset_distribution": onset_dist,
        "key_molecular_features": key_molecular_features,
        "heteroplasmy_clinical_map": heteroplasmy_clinical_map,
        "top_variants": top_variants,
        "clinical_alerts": alerts,
        "absolute_contraindications": [
            "Metformin — Complex I inhibitor additive with pan-OXPHOS CI deficiency → fatal lactic acidosis; ABSOLUTE CI in ALL m.3243A>G carriers (including MIDD 'T2DM'); insulin is the safe diabetic treatment",
            "Valproic acid / VPA — mt-ribosome inhibition + CoA sequestration; worsens pan-OXPHOS; use LEV for seizures",
            "IV tPA / thrombolytics — MELAS stroke-like episodes are NOT thrombotic; IV tPA → haemorrhagic transformation without benefit; FATAL",
            "Linezolid — inhibits mt-23S rRNA → directly impairs mt-ribosome → blocks synthesis of ALL 13 OXPHOS subunits → collapse of pan-OXPHOS",
            "Chloramphenicol — mt-ribosome inhibitor; same mechanism as linezolid; absolute CI in all MT-TL1 patients",
            "Propofol — PRIS + direct ETC inhibition; compounding pan-OXPHOS crisis; use alternative anaesthesia (e.g. sevoflurane)",
            "Fasting / prolonged NPO — GIR 6-8 mg/kg/min MANDATORY in acute MELAS/metabolic crisis; NEVER fast",
        ],
        "mandatory_acute_treatments": [
            "IV L-Arginine 0.5 g/kg (max 10 g) over 30-60 min at SLE onset — Level B (Koga 2010 Stroke) — ONLY evidence-based acute SLE treatment",
            "Thiamine B1 — 10-20 mg/kg IV in metabolic crisis; oral maintenance (PDH cofactor — MANDATORY empiric)",
            "Biotin — 5-20 mg/day empiric (pending BTD/SLC19A3 BTBGD exclusion — Leigh-like mimic)",
            "GIR 6-8 mg/kg/min — continuous glucose infusion; NEVER fast in metabolic crisis",
            "LEV (levetiracetam) — preferred AED for seizures in MT-TL1; avoid VPA (absolute CI)",
        ],
        "maintenance_treatments": [
            "Oral L-Arginine — Level B maintenance (reduces SLE frequency and severity; Koga 2010)",
            "CoQ10 ubiquinol (10-20 mg/kg/day) — Level C",
            "Riboflavin B2 (FAD/FMN for CI/CIII) — Level C",
            "L-Carnitine — Level C",
            "Insulin (NOT metformin) — for MIDD/MELAS-associated diabetes",
        ],
    }


def get_breakdown():
    patients = _make_patients()
    n = len(patients)

    from collections import Counter

    variant_rows = []
    for v in VARIANTS:
        pts = [p for p in patients if p["variant"] == v["hgvs_mtdna"]]
        if not pts:
            continue
        nv = len(pts)
        pct_fn = lambda key, pts=pts, nv=nv: round(100 * sum(1 for p in pts if p.get(key, False)) / nv, 1)
        variant_rows.append({
            "variant": v["hgvs_mtdna"],
            "domain": v["domain"],
            "type": v["type"],
            "severity": v["severity"],
            "phenotype": v["phenotype"],
            "n_patients": nv,
            "frequency_pct": v["frequency_pct"],
            "penetrance_pct": v["penetrance_pct"],
            "avg_ci_activity_pct": round(sum(p["ci_activity_pct"] for p in pts) / nv, 1),
            "avg_heteroplasmy_urine_pct": round(sum(p["heteroplasmy_urine_pct"] for p in pts) / nv, 1),
            "avg_heteroplasmy_blood_pct": round(sum(p["heteroplasmy_blood_pct"] for p in pts) / nv, 1),
            "avg_lactic_acid": round(sum(p["lactic_acid_mmolL"] for p in pts) / nv, 1),
            "avg_onset_weeks": round(sum(p["onset_weeks"] for p in pts) / nv, 0),
            "sle_pct": pct_fn("stroke_like_episode"),
            "snhl_pct": pct_fn("sensorineural_hearing_loss"),
            "diabetes_pct": pct_fn("diabetes_mellitus"),
            "lactic_acidosis_pct": pct_fn("lactic_acidosis"),
            "seizures_pct": pct_fn("seizures"),
            "ragged_red_fibres_pct": pct_fn("ragged_red_fibres"),
            "phenotype_breakdown": dict(Counter(p["phenotype"] for p in pts)),
            "notes": v["notes"],
        })

    pheno_dist = Counter(p["phenotype"] for p in patients)

    heteroplasmy_bands_urine = {"<30%": 0, "30-60%": 0, "60-80%": 0, ">80%": 0}
    heteroplasmy_bands_blood = {"<30%": 0, "30-60%": 0, "60-80%": 0, ">80%": 0}
    for p in patients:
        for bands, key in [(heteroplasmy_bands_urine, "heteroplasmy_urine_pct"),
                           (heteroplasmy_bands_blood, "heteroplasmy_blood_pct")]:
            h = p[key]
            if h < 30:
                bands["<30%"] += 1
            elif h < 60:
                bands["30-60%"] += 1
            elif h < 80:
                bands["60-80%"] += 1
            else:
                bands[">80%"] += 1

    ci_bands = {"<15%": 0, "15-30%": 0, "30-50%": 0, ">50%": 0}
    for p in patients:
        c = p["ci_activity_pct"]
        if c < 15:
            ci_bands["<15%"] += 1
        elif c < 30:
            ci_bands["15-30%"] += 1
        elif c < 50:
            ci_bands["30-50%"] += 1
        else:
            ci_bands[">50%"] += 1

    outcome_dist = Counter(p["outcome"] for p in patients)
    outcome_rows = [{"outcome": k, "count": v} for k, v in outcome_dist.most_common()]

    ddx_table = [
        {
            "entity": "Ischemic stroke (thrombotic/embolic)",
            "distinguishing_feature": "Ischemic stroke: MRA/MRV shows vessel occlusion; lesions respect arterial territories; IV tPA indicated; no maternal family history of similar episodes; normal respiratory chain enzymology; MELAS SLE: MRA NORMAL; lesions cross vascular territories; IV tPA ABSOLUTE CI; maternal pedigree; pan-OXPHOS",
            "key_test": "MRA/MRV (normal in MELAS, abnormal in ischemic stroke); MRS (lactate peak in MELAS lesion); m.3243A>G urine heteroplasmy; respiratory chain enzymology",
        },
        {
            "entity": "Leigh Syndrome / MT-ND1-6 protein-coding mutations",
            "distinguishing_feature": "Leigh: bilateral SYMMETRIC basal ganglia + brainstem T2 signal (NOT cortical); isolated CI deficiency (single complex); MT-TL1/MELAS: cortical + subcortical lesions crossing territories; PAN-OXPHOS (CI+CIII+CIV); CII normal; SLE not typical of Leigh; heteroplasmy threshold vs phenotype different",
            "key_test": "Brain MRI morphology (BG symmetric vs cortical asymmetric); BN-PAGE (single vs multi-complex); urinary m.3243A>G vs mtDNA protein-coding gene panel",
        },
        {
            "entity": "MT-ATP6 / NARP (same Complex V)",
            "distinguishing_feature": "MT-ATP6/NARP: isolated Complex V deficiency; Retinitis Pigmentosa hallmark; heteroplasmy threshold 70-90% NARP vs >90% Leigh; NO pan-OXPHOS; NO SLE; MT-TL1/MELAS: pan-OXPHOS (CI+CIII+CIV); SLE; NO RP; diabetes+deafness (MIDD); different heteroplasmy tier",
            "key_test": "ERG (RP in NARP — absent in MELAS); respiratory chain BN-PAGE (CV only in ATP6 vs CI+CIII+CIV in MT-TL1); mtDNA sequencing distinguishes",
        },
        {
            "entity": "POLG (Alpers / SANDO / mtDNA depletion)",
            "distinguishing_feature": "POLG: hepatopathy (Alpers — VPA hepatotoxicity) + mtDNA depletion; autosomal recessive; WES detects POLG; MELAS: NO hepatopathy; maternal inheritance; NO mtDNA depletion (level normal); urinary m.3243A>G diagnostic; VPA absolute CI in BOTH (different reasons: POLG+VPA → fatal hepatotoxicity; MT-TL1+VPA → mt-ribosome impairment)",
            "key_test": "Liver enzymes + ammonia; mtDNA quantification (depletion in POLG); POLG WES vs mtDNA tRNA panel for m.3243A>G",
        },
        {
            "entity": "MT-TK m.8344A>G (MERRF — tRNA-Lys)",
            "distinguishing_feature": "MERRF: myoclonic epilepsy + RRF + cerebellar ataxia + ABSENCE of stroke-like episodes; pan-OXPHOS but different distribution (MERRF more CI+CIV than CIII); NO SLE; NO diabetes; MT-TL1/MELAS: SLE dominant; diabetes (MIDD); m.8344A>G (tRNA-Lys) vs m.3243A>G (tRNA-Leu); blood heteroplasmy also underestimates in MERRF",
            "key_test": "m.8344A>G testing for MERRF vs m.3243A>G for MELAS; EEG (myoclonic pattern in MERRF vs focal in MELAS); MRI (MERRF: cerebellar atrophy + BG; MELAS: cortical)",
        },
        {
            "entity": "BTBGD (SLC19A3 — biotin-thiamine responsive)",
            "distinguishing_feature": "BTBGD: Leigh-identical bilateral BG MRI (can mimic Leigh-like MELAS) but TREATABLE with biotin+thiamine; autosomal recessive; NO pan-OXPHOS deficiency; WES detectable; MELAS: pan-OXPHOS; maternal inheritance; cortical MRI not BG-only; m.3243A>G confirms MELAS",
            "key_test": "SLC19A3 WES; empiric biotin+thiamine trial; respiratory chain enzymology (normal in BTBGD vs pan-OXPHOS in MELAS); m.3243A>G urine quantitation",
        },
        {
            "entity": "Type 2 Diabetes Mellitus + Age-related SNHL (MIDD misdiagnosis)",
            "distinguishing_feature": "T2DM: no maternal pedigree specific to diabetes+deafness; no lactic acidosis at rest; no exercise intolerance disproportionate to lifestyle; MIDD (m.3243A>G low heteroplasmy): maternal pedigree of DM+deafness; exercise intolerance; lactic acidosis post-exercise; metformin ABSOLUTE CI; early ophthalmology (macular pattern pigmentary changes in MIDD)",
            "key_test": "m.3243A>G urine heteroplasmy (screen in all maternal DM+deafness pedigrees); exercise lactate; ophthalmology (MIDD macular pigmentary dystrophy — different from KSS RP); respiratory chain enzymology",
        },
    ]

    sle_management_table = [
        {"phase": "Acute SLE onset", "treatment": "IV L-Arginine 0.5 g/kg (max 10g) over 30-60 min", "evidence": "Level B (Koga 2010 Stroke)", "notes": "ONLY evidence-based treatment; give WITHIN 3h of SLE onset for best effect"},
        {"phase": "Acute SLE", "treatment": "IV glucose (GIR 6-8 mg/kg/min)", "evidence": "Mandatory", "notes": "NEVER fast; continuous glucose infusion supports energy metabolism"},
        {"phase": "Acute SLE", "treatment": "IV Thiamine B1 (10-20 mg/kg)", "evidence": "Mandatory empiric", "notes": "Before glucose in suspected Wernicke risk; PDH cofactor"},
        {"phase": "Acute seizure", "treatment": "IV Levetiracetam (LEV)", "evidence": "Level C preferred", "notes": "AVOID VPA (absolute CI in all MT-TL1 patients)"},
        {"phase": "Acute SLE", "treatment": "NO IV tPA / NO thrombolytics", "evidence": "Absolute CI", "notes": "SLE ≠ ischemic stroke; MRA normal; tPA → haemorrhage without benefit; fatal"},
        {"phase": "Maintenance", "treatment": "Oral L-Arginine 0.15-0.30 g/kg/day", "evidence": "Level B maintenance", "notes": "Reduces SLE frequency and severity; NO calcium-channel blockers (inhibit NO synthesis)"},
        {"phase": "Maintenance", "treatment": "CoQ10 ubiquinol 10-20 mg/kg/day", "evidence": "Level C", "notes": "Electron transfer support; well-tolerated"},
        {"phase": "MIDD diabetes", "treatment": "Insulin (NOT metformin)", "evidence": "Metformin absolute CI", "notes": "All oral biguanides CI; sulfonylureas generally safe; insulin preferred"},
    ]

    return {
        "gene": "MT-TL1",
        "n_patients": n,
        "seed": SEED,
        "variant_breakdown": variant_rows,
        "phenotype_distribution": dict(pheno_dist),
        "heteroplasmy_bands_urine": heteroplasmy_bands_urine,
        "heteroplasmy_bands_blood": heteroplasmy_bands_blood,
        "ci_activity_bands": ci_bands,
        "outcome_distribution": outcome_rows,
        "differential_diagnosis": ddx_table,
        "sle_management_table": sle_management_table,
        "patient_table": [
            {
                "id": p["patient_id"],
                "phenotype": p["phenotype"],
                "variant": p["variant"],
                "heteroplasmy_urine_pct": p["heteroplasmy_urine_pct"],
                "heteroplasmy_blood_pct": p["heteroplasmy_blood_pct"],
                "ci_pct": p["ci_activity_pct"],
                "lactate": p["lactic_acid_mmolL"],
                "onset_weeks": p["onset_weeks"],
                "sle": p["stroke_like_episode"],
                "snhl": p["sensorineural_hearing_loss"],
                "diabetes": p["diabetes_mellitus"],
                "seizures": p["seizures"],
                "ragged_red_fibres": p["ragged_red_fibres"],
                "outcome": p["outcome"],
            }
            for p in patients
        ],
    }


def get_definitions():
    return {
        "gene": "MT-TL1",
        "omim_gene": "OMIM *590050",
        "full_name": "Mitochondrially Encoded tRNA-Leu(UUR)",
        "protein_name": "tRNA-Leu(UUR) — 74-nucleotide RNA gene (no protein product); aminoacylates leucine onto mitochondrial ribosomes for translation of all 13 mtDNA-encoded OXPHOS subunits",
        "trna_length_nt": 74,
        "rcrs_positions": "3230-3304",
        "strand": "H-strand",
        "anticodon": "UUR (leucine, wobble anticodon; R = purine)",
        "omim_diseases": {
            "MELAS": "MELAS Syndrome (OMIM #540000) — Mitochondrial Encephalomyopathy, Lactic Acidosis, Stroke-like Episodes; stroke-like episodes; encephalopathy; sensorineural hearing loss; diabetes; ragged red fibres; pan-OXPHOS deficiency",
            "MIDD": "Maternally Inherited Diabetes and Deafness — same m.3243A>G mutation at low heteroplasmy (<60% tissue); diabetes mellitus (type 2-like onset 20-40y) + progressive SNHL; NO stroke-like episodes; frequently misdiagnosed as T2DM + age-related deafness",
            "CPEO": "Chronic Progressive External Ophthalmoplegia — ptosis + ophthalmoplegia; variable lactic acidosis; exercise intolerance; often with deletion rather than point mutation",
            "exercise_intolerance": "Exercise Intolerance / Myopathy — pan-OXPHOS deficiency on exertion; RRF on Gomori; elevated post-exercise lactate; adult onset common at intermediate heteroplasmy",
        },
        "key_variants": {
            "m.3243A>G": "TΨC loop of tRNA-Leu(UUR) — ~80% of MELAS; ~1 in 400 population prevalence; disrupts LARS2 aminoacylation → pan-OXPHOS; MIDD at low heteroplasmy; MELAS at high heteroplasmy",
            "m.3271T>C": "Anticodon stem — ~7% of MELAS cohort; structural stem disruption; MELAS-like phenotype; SLE + SNHL + lactic acidosis",
            "m.3251A>G": "Variable loop — ~4%; mild structural perturbation; exercise intolerance / mild MELAS phenotype",
            "low_heteroplasmy_midd": "m.3243A>G at blood <40% — MIDD phenotype; diabetes + SNHL; no SLE; frequently misdiagnosed",
            "large_deletion": "Deletion spanning MT-TL1 — KSS/CPEO/Pearson; multi-complex OXPHOS; annual Holter mandatory (KSS)",
        },
        "melas_definition": "MELAS Syndrome: clinical triad of Mitochondrial Encephalomyopathy (seizures + cognitive decline) + Lactic Acidosis + Stroke-like Episodes (SLE); caused by mt-translation defects (predominantly MT-TL1 m.3243A>G) → pan-OXPHOS deficiency; SLE are NOT thrombotic — MRA/MRV normal; cortical lesions cross vascular territories; onset typically 5-15 years; maternal family history of DM, SNHL, or MELAS; muscle biopsy: RRF (Gomori trichrome) + COX-negative fibres + SDH-positive vessels (SSSVS pattern pathognomonic on SDH histochemistry)",
        "sle_definition": "Stroke-like Episodes (SLE): focal neurological deficits (hemiparesis, hemianopia, aphasia) without MRA/MRV vessel occlusion; mechanism is neuronal energy failure → cytotoxic oedema + cortical spreading depression (NOT vascular occlusion); DWI shows cortical/subcortical lesions crossing arterial territories; MRS shows lactate peak in lesion; IV tPA ABSOLUTELY CONTRAINDICATED; IV L-Arginine 0.5 g/kg is the only evidence-based acute treatment (Level B, Koga 2010)",
        "midd_definition": "MIDD (Maternally Inherited Diabetes and Deafness): same m.3243A>G mutation at lower heteroplasmy (<60% tissue); presents as 'lean' or early-onset type 2 diabetes (age 20-40y) + progressive sensorineural hearing loss; NO stroke-like episodes; NO encephalopathy; RRF may be absent; frequently misdiagnosed as T2DM + age-related deafness; CRITICAL: metformin ABSOLUTE CI even in MIDD — use insulin; ophthalmology: macular pattern pigmentary dystrophy (different from KSS retinitis pigmentosa)",
        "pan_oxphos_definition": "Pan-OXPHOS Deficiency: in MT-TL1 mutations, impaired aminoacylation of tRNA-Leu(UUR) → reduced mt-ribosome translation of ALL 13 mtDNA-encoded OXPHOS subunits (ND1-6, CO1-3, ATP6, ATP8, CYB, ND4L) → CI + CIII + CIV all reduced; Complex II (SDH — nuclear-encoded) is NOT affected → CII NORMAL is the biochemical fingerprint of mt-translation defects; distinguishes MT-TL1 from protein-coding mutations (MT-ND1-6 = CI only; MT-CO1-3 = CIV only; MT-ATP6/8 = CV only); BN-PAGE: multi-complex reduction (CI+CIII+CIV) with CII band normal",
        "heteroplasmy_urine_definition": "Urinary Epithelial Cell Heteroplasmy: urinary epithelial cells are post-mitotic (or low-mitotic) and maintain the original heteroplasmy from the germline without clonal selection; blood heteroplasmy underestimates by 20-30% in m.3243A>G carriers because mitotically-active haematopoietic stem cells clear mutant mtDNA over time (replicative segregation against m.3243A>G); ALWAYS use urine for accurate threshold determination; method: morning urine pellet → mtDNA extraction → allele-specific PCR or NGS quantitation of m.3243A>G",
        "arginine_mechanism": "L-Arginine in MELAS: MELAS stroke-like episodes involve endothelial nitric oxide synthase (eNOS) dysfunction → reduced NO → impaired cerebrovascular autoregulation + vasospasm → neuronal energy failure; L-Arginine is the direct precursor to NO synthesis via eNOS; IV L-Arg 0.5 g/kg over 30-60 min restores NO production → vasodilation → improves cerebral blood flow during acute SLE; oral L-Arg maintenance reduces SLE frequency and severity; mechanism: substrate provision for eNOS during energy-compromised state; evidence: Koga et al. 2010 Neurology + Stroke (Level B)",
        "tpa_ci_definition": "IV tPA Absolute CI in MELAS: stroke-like episodes have NO thrombus → tPA has no therapeutic target; instead, tPA lyses normal vasculature → haemorrhagic transformation of cytotoxic oedema lesion; clinical presentations that trigger 'stroke protocol' in MELAS are SLE (acute focal deficit, cortical DWI change, normal MRA) → ALWAYS check MELAS history before tPA; if m.3243A>G confirmed, tPA is NEVER appropriate for SLE regardless of DWI signal",
        "m3243ag_definition": "m.3243A>G: nucleotide substitution at rCRS position 3243 A→G in the TΨC loop of MT-TL1 tRNA-Leu(UUR); impairs the aminoacylation recognition of LARS2 (leucyl-tRNA synthetase, mitochondrial); reduces efficiency of leucine incorporation into all 13 mt-translated OXPHOS subunits; pan-OXPHOS deficiency results; heteroplasmy determines phenotype (MIDD at low, MELAS at high); MOST COMMON pathogenic mtDNA mutation worldwide (~1 in 400 adults carry it; population prevalence Gorman 2016 Ann Neurol)",
        "ragged_red_fibres": "Ragged Red Fibres (RRF): hallmark muscle biopsy finding in MELAS and other mt-tRNA mutations; RRF on Gomori trichrome = subsarcolemmal accumulation of abnormal mitochondria (compensatory proliferation) → 'ragged' appearance at fibre periphery; SDH (succinate dehydrogenase) staining: RRF appear strongly SDH-positive ('ragged blue') because SDH is nuclear-encoded and still functional; COX (cytochrome c oxidase) staining: RRF appear COX-negative (CIV affected by mt-translation defect); SSSVS (strongly SDH-stained small vessels) on SDH histochemistry is a specific MELAS muscle finding (endothelial mitochondrial proliferation in vessels); RRF percentage roughly correlates with heteroplasmy in affected tissue",
        "wes_coverage": "MT-TL1 is a mitochondrial tRNA gene (H-strand rCRS 3230-3304) — WES does NOT cover mitochondrial tRNA mutations or structural rearrangements; dedicated mtDNA sequencing required: (1) m.3243A>G targeted PCR/AS-PCR for rapid diagnosis; (2) whole-mtDNA NGS panel for full tRNA coverage; (3) urinary epithelial cell quantitation for accurate heteroplasmy; (4) muscle biopsy for histochemistry (RRF, COX-negative/SDH-positive pattern, SSSVS) and respiratory chain enzymology (pan-OXPHOS)",
        "absolute_contraindications": {
            "Metformin": "Complex I inhibitor — additive with pan-OXPHOS CI deficiency → fatal lactic acidosis; ABSOLUTE CI in ALL m.3243A>G carriers regardless of phenotype (including MIDD without MELAS symptoms); insulin is the safe diabetic treatment",
            "VPA / Valproic Acid": "Mt-ribosome inhibition + CoA sequestration — worsens pan-OXPHOS; hepatotoxicity risk; use LEV for seizures instead",
            "IV tPA / Thrombolytics": "MELAS SLE are NOT thrombotic — IV tPA → haemorrhagic transformation + no benefit; FATAL in MELAS; never administer in confirmed or suspected MT-TL1",
            "Linezolid": "Inhibits mt-23S rRNA → blocks translation of ALL 13 OXPHOS subunits → collapses pan-OXPHOS",
            "Chloramphenicol": "Mt-ribosome inhibitor — same mechanism as linezolid; absolute CI in all MT-TL1 patients",
            "Propofol": "PRIS (propofol infusion syndrome) + direct ETC inhibition; use sevoflurane/isoflurane for anaesthesia",
            "Fasting": "GIR 6-8 mg/kg/min MANDATORY in acute crisis — NEVER fast; metabolic stress precipitates SLE",
        },
        "recommended_treatments": {
            "iv_l_arginine_acute": "Level B — IV L-Arg 0.5 g/kg (max 10g) over 30-60 min at SLE onset (Koga 2010 Stroke)",
            "oral_l_arginine_maintenance": "Level B maintenance — 0.15-0.30 g/kg/day oral; reduces SLE frequency",
            "thiamine_B1": "Mandatory empiric — 10-20 mg/kg IV in crisis; oral 100-300 mg/day maintenance",
            "biotin": "5-20 mg/day empiric — pending BTD/SLC19A3 (BTBGD) exclusion",
            "coq10_ubiquinol": "Level C — 10-20 mg/kg/day",
            "riboflavin_B2": "Level C — FAD/FMN cofactor for CI/CIII",
            "lev": "Levetiracetam — preferred AED; avoid VPA absolute CI",
            "gir": "GIR 6-8 mg/kg/min — NEVER fast in crisis",
            "insulin_not_metformin": "Insulin for MIDD/MELAS-associated diabetes; metformin ABSOLUTE CI",
        },
        "specialist_monitoring": {
            "Neurology": "MELAS: SLE surveillance; seizure management (LEV); cognitive assessment; MRI 6-monthly during active phase; LP for lactate in CSF if SLE suspected",
            "Cardiology": "Annual ECG + Holter — MELAS cardiomyopathy 25-30%; KSS deletion patients: pacemaker threshold PR >240ms or Mobitz II block; echocardiography annually in MELAS",
            "Endocrinology": "MIDD/MELAS diabetes: insulin protocol; avoid metformin; HbA1c monitoring; renal function (MIDD nephropathy 40%)",
            "Audiology": "Annual audiometry — SNHL in 80% MELAS, 85% MIDD; cochlear implant assessment (good outcomes reported in MIDD)",
            "Ophthalmology": "Annual fundus exam — MIDD macular pigmentary dystrophy; KSS pigmentary retinopathy; MELAS macular changes (15-20%)",
            "Genetics": "Maternal cascade testing mandatory; urinary m.3243A>G quantitation for all maternal relatives; pre-conceptional counselling; heteroplasmy segregation in offspring unpredictable due to mtDNA bottleneck",
        },
        "key_references": [
            "Goto Y et al. (1990) A mutation in the tRNA(Leu)(UUR) gene associated with the MELAS subgroup of mitochondrial encephalomyopathies. Nature 348(6302):651-653. [m.3243A>G FIRST DESCRIPTION — seminal paper defining MELAS/MT-TL1 link]",
            "Koga Y et al. (2010) L-Arginine improves the symptoms of strokelike episodes in MELAS. Neurology 64(4):710-712 + Stroke 41(7):1579-1586. [L-Arginine IV acute SLE + oral maintenance — Level B evidence — ONLY evidence-based acute SLE treatment]",
            "Gorman GS et al. (2016) Prevalence of nuclear and mitochondrial DNA mutations related to adult mitochondrial disease. Ann Neurol 79(4):589-591. [m.3243A>G ~1 in 400 adults; population prevalence mtDNA mutations]",
            "Manwaring N et al. (2007) Population prevalence of the MELAS A3243G mutation. Mitochondrion 7(3):230-233. [Urinary epithelial cell testing > blood; heteroplasmy underestimation in blood]",
            "El-Hattab AW et al. (2012) Restoration of impaired nitric oxide production in MELAS syndrome with citrulline and arginine supplementation. Mol Genet Metab 105(4):607-614. [NO pathway mechanism for L-Arginine benefit in MELAS]",
        ],
        "cohort_seed": SEED,
        "n_patients": N_PATIENTS,
        "generated": "2026-09-03",
    }


if __name__ == "__main__":
    import json
    overview = get_overview()
    print(f"MT-TL1 overview: {overview['n_patients']} patients, "
          f"avg CI {overview['cohort_statistics']['avg_ci_activity_pct']}%, "
          f"avg lactate {overview['cohort_statistics']['avg_lactic_acid_mmolL']} mmol/L")
    print(f"SLE: {overview['cohort_statistics']['stroke_like_episode_pct']}%")
    print(f"SNHL: {overview['cohort_statistics']['sensorineural_hearing_loss_pct']}%")
    print(f"Diabetes: {overview['cohort_statistics']['diabetes_mellitus_pct']}%")
    bd = get_breakdown()
    print(f"Variants: {len(bd['variant_breakdown'])}")
    defs = get_definitions()
    print(f"Key variants: {list(defs['key_variants'].keys())}")
    print("✅ MT-TL1 dashboard OK")
