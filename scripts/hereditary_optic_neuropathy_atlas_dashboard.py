#!/usr/bin/env python3
"""Hereditary-Optic-Neuropathy-Atlas — Complete 8-Gene Hereditary Optic Neuropathy Atlas
OPA1    (Dynamin-like GTPase 1; 3304 aa; 3q29; AD;
          Autosomal Dominant Optic Atrophy (ADOA / Kjer disease) — most common hereditary optic neuropathy;
          Centrocecal scotoma, temporal disc pallor, blue-yellow colour axis first;
          OPA1-plus (GTPase domain missense): ptosis + CPEO + myopathy + ataxia + low-frequency SNHL;
          Haploinsufficiency (frameshift/nonsense) → classic ADOA; GTPase missense → syndromic;
          Mitochondrial fusion GTPase — OPA1 LOF → fragmented mitochondria → RGC apoptosis) ·
OPA3    (OPA3 protein; 179 aa; 19q13.32; AD / AR;
          AD: Optic Atrophy + Cataract ± subclinical chorea/spastic paraparesis (3-MGC mild);
          AR (Costeff syndrome, Iraqi-Jewish founder c.143-1G>A): Optic Atrophy + Chorea + Spastic Paraplegia + elevated 3-methylglutaconic aciduria;
          Urine 3-MGC assay diagnostic for AR Costeff; MINOS complex (MIC19 partner);
          Mitochondrial inner membrane organising system — OPA3 variants disrupt mitochondrial morphology and cristae) ·
MT-ND4  (NADH:Ubiquinone Oxidoreductase Core Subunit ND4; mtDNA; Maternal;
          Leber Hereditary Optic Neuropathy (LHON) — m.11778G>A most common primary mutation (~70% worldwide);
          Subacute painless central visual loss, sequential eye involvement weeks-months apart;
          Male predominance: ~80% penetrance in hemizygous males vs ~25% females;
          Worst visual prognosis of three primary LHON variants — only ~20% partial recovery;
          Idebenone (Raxone) — approved Europe 2015; lenadogene nolparvovec (Lumevoq) gene therapy — intravitreal AAV2-ND4) ·
MT-ND1  (NADH:Ubiquinone Oxidoreductase Core Subunit ND1; mtDNA; Maternal;
          LHON — m.3460G>A second primary mutation (~13% of LHON cases worldwide);
          Poor visual prognosis; complex I electron transport impaired → excess ROS → RGC death;
          LHON-Plus: subset develop MS-like CNS white matter lesions + Leigh-like encephalopathy;
          Heteroplasmy threshold: >60% mutant load typically required for expression) ·
MT-ND6  (NADH:Ubiquinone Oxidoreductase Core Subunit ND6; mtDNA; Maternal;
          LHON — m.14484T>C third primary mutation (~14% worldwide);
          BEST visual prognosis of three primary mutations: ~70% spontaneous partial recovery;
          Youngest typical age of onset (teens); incomplete penetrance — most carriers never symptomatic;
          Same treatment as ND4: idebenone first, lenadogene nolparvovec for eligible eyes) ·
WFS1    (Wolframin; 890 aa; 4p16.1; AR / AD;
          AR: Wolfram syndrome (DIDMOAD) — Diabetes Insipidus + Diabetes Mellitus + Optic Atrophy + Deafness;
          Onset sequence: DM (1st decade) → OA (2nd) → DI + SNHL (3rd) → neurodegeneration (4th);
          Neurogenic bladder, psychiatric disorder, brainstem atrophy — progressive;
          Wolframin is an ER transmembrane protein — WFS1 LOF → ER calcium dysregulation → UPR → RGC loss;
          AD: heterozygous WFS1 → isolated SNHL (DFNA6/14/38) or mild OA — far milder phenotype) ·
TMEM126A (Transmembrane Protein 126A; 149 aa; 11q14.1; AR;
          Autosomal Recessive Optic Atrophy (AROA) — slowly progressive pure optic neuropathy;
          North African / Moroccan founder: p.Arg55* (c.163C>T) nonsense variant enriched;
          Mitochondrial complex I assembly factor — TMEM126A LOF impairs complex I biogenesis;
          Mild SNHL in subset; no systemic features (DDx Wolfram);
          Blue-yellow colour axis impaired first; centrocecal scotoma; slowly progressive) ·
ACO2    (Aconitase 2 / Mitochondrial Aconitase; 780 aa; 22q13.2; AR;
          Infantile optic atrophy + cerebellar ataxia + hypotonia + intellectual disability;
          TCA cycle enzyme: ACO2 catalyses citrate → isocitrate; LOF → mitochondrial energy failure;
          MRI: cerebellar atrophy ± periventricular white matter; elevated lactate in CSF/blood in some;
          Severe early-onset syndromic OA; visual impairment from infancy;
          Compound heterozygotes most common; homozygous founder variants in consanguineous families)
320-patient aggregate cohort (8 × 40, seeds 1414–1421)
"""

import random

SEED_BASE = 1414

HON_GENES = [
    # ── OPA1 — Autosomal Dominant Optic Atrophy ──
    {
        "gene": "OPA1",
        "protein": "Dynamin-like GTPase OPA1 (Mitochondrial Fusion GTPase)",
        "alias": (
            "OPA1; OMIM gene 605290; ADOA #165500; 3q29; 3304 aa; ~112 kDa; "
            "Most common hereditary optic neuropathy (~1:10 000–1:30 000); "
            "Haploinsufficiency (frameshift, nonsense, splice, large deletion) → classic ADOA — slowly progressive bilateral symmetric centrocecal scotoma; "
            "GTPase domain missense (e.g. R445H, c.1334G>A; D603H) → OPA1-plus: ptosis + CPEO + myopathy + ataxia + low-frequency SNHL; "
            "Temporal disc pallor + blue-yellow > red-green colour axis first; "
            "Onset typically 1st decade (4–6 yr) but wide range; "
            "Mitochondrial inner membrane GTPase essential for inner membrane fusion — LOF → mitochondrial fragmentation → RGC apoptosis; "
            "No proven disease-modifying treatment; idebenone anecdotally used; "
            "OPA1-plus: additional mtDNA deletions accumulate in muscle (secondary multiple mtDNA deletions)"
        ),
        "aa": "3304 aa",
        "kDa": "~112 kDa",
        "locus": "3q29",
        "omim_gene": 605290,
        "omim_disease": 165500,
        "inheritance": "AD — haploinsufficiency (frameshift/nonsense/splice) or dominant-negative (GTPase missense = OPA1-plus); de novo ~15%",
        "gene_class": (
            "OPA1 encodes the major mitochondrial inner membrane fusion GTPase. Haploinsufficiency "
            "from truncating or splice variants causes classic ADOA through reduced mitochondrial "
            "fusion in retinal ganglion cells (RGCs). Missense variants within the GTPase domain "
            "produce OPA1-plus syndrome by dominant-negative disruption of the dynamin ring, "
            "causing additional multi-tissue mitochondrial dysfunction with secondary accumulation "
            "of somatic mtDNA deletions. Temporal disc pallor and centrocecal scotoma are the "
            "clinical hallmarks. Colour discrimination testing (Ishihara + FM-100 hue) typically "
            "shows a tritan (blue-yellow) axis defect earlier than protan/deutan axes, distinguishing "
            "OPA1-related optic neuropathy from other causes."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("Haploinsufficiency: frameshift / nonsense / splice — classic ADOA", 0.65),
            ("GTPase domain missense (R445H, D603H, etc.) — OPA1-plus syndromic", 0.25),
            ("Large intragenic deletion (MLPA-detectable)", 0.10),
        ],
        "age_onset_years_range": (4, 20),
        "sex_ratio_M": 0.50,
        "rates": {
            "centrocecal_scotoma":              0.95,
            "temporal_disc_pallor":             0.93,
            "blue_yellow_colour_axis_first":    0.88,
            "bilateral_symmetric":              0.92,
            "low_frequency_snhl":               0.20,
            "opa1_plus_ptosis_cpeo_myopathy":   0.22,
            "ataxia_in_opa1_plus":              0.18,
            "de_novo_variant":                  0.15,
            "secondary_mtdna_deletions_muscle": 0.20,
            "visual_acuity_worse_than_0p5":     0.70,
            "nystagmus":                        0.05,
            "family_history_optic_atrophy":     0.85,
        },
        "hallmarks": [
            "Centrocecal scotoma: involves fixation and the blind spot — not peripheral loss",
            "Temporal disc pallor (superior and inferior pallor follows in OPA1-plus)",
            "Blue-yellow (tritan) colour axis impaired first — FM-100 hue test shows tritan error",
            "Slowly progressive — usually detected in first decade but may not be noticed until teens",
            "OPA1-plus: GTPase domain missense → ptosis + CPEO + proximal myopathy + ataxia",
            "Secondary multiple mtDNA deletions accumulate in muscle in OPA1-plus",
            "Low-frequency (250–1000 Hz) SNHL — audiogram pattern distinguishes from age-related SNHL",
        ],
        "treatment_alerts": [
            "No proven disease-modifying treatment for classic ADOA",
            "Idebenone (antioxidant) used off-label — evidence weak; consider trial in early progressive disease",
            "Avoid smoking and excessive alcohol — both accelerate RGC loss",
            "Avoid mtDNA-toxic drugs (linezolid, aminoglycosides, chloramphenicol, NRTI antiretrovirals) in OPA1-plus",
            "OPA1-plus: monitor cardiac function (cardiomyopathy reported) and respiratory muscle",
            "Low vision aids: eccentric viewing training, contrast-enhanced magnification",
            "Genetic counselling: 50% transmission risk; variable expressivity — mild to severe",
            "Ophthalmology 6–12 monthly: OCT RNFL (nerve fibre layer), VEP, colour vision",
        ],
        "primary_treatment": (
            "No FDA/EMA-approved treatment for OPA1-ADOA. Idebenone 900 mg/day in 3 divided doses "
            "used off-label based on LHON data; avoid mitochondrial-toxic substances; low vision "
            "rehabilitation; OPA1-plus: multidisciplinary (neurology, ophthalmology, cardiology, "
            "respiratory); genetic cascade testing of at-risk family members."
        ),
    },

    # ── OPA3 — Optic Atrophy type 3 / Costeff syndrome ──
    {
        "gene": "OPA3",
        "protein": "Optic Atrophy Protein 3 (MINOS complex component)",
        "alias": (
            "OPA3; OMIM gene 606580; AD OA3 #165300; AR Costeff/3-MGC III #258501; 19q13.32; 179 aa; ~20 kDa; "
            "AD (French Canadian): optic atrophy onset 1st decade + subclinical extrapyramidal features + early cataract; "
            "AR (Iraqi-Jewish founder c.143-1G>A splice): Costeff syndrome = optic atrophy + chorea + spastic paraplegia; "
            "Elevated 3-methylglutaconic acid (3-MGC) in urine — diagnostic marker for AR form; "
            "OPA3 is part of the MINOS (mitochondrial inner membrane organizing system) complex via MIC19; "
            "LOF → disrupted cristae morphology + mitochondrial fragmentation"
        ),
        "aa": "179 aa",
        "kDa": "~20 kDa",
        "locus": "19q13.32",
        "omim_gene": 606580,
        "omim_disease": 258501,
        "inheritance": "AD (mild OA + cataract ± chorea) or AR = Costeff syndrome (OA + chorea + spastic paraplegia + 3-MGC)",
        "gene_class": (
            "OPA3 encodes a small MINOS-complex protein on the mitochondrial inner membrane. "
            "Heterozygous dominant variants cause a milder phenotype (optic atrophy, early cataract, "
            "subclinical extrapyramidal features). Homozygous or compound heterozygous recessive "
            "variants — particularly the Iraqi-Jewish founder c.143-1G>A splice variant — cause "
            "Costeff syndrome (3-methylglutaconic aciduria type III): optic atrophy beginning in "
            "infancy, later chorea, and spastic paraplegia. Elevated 3-methylglutaconic acid in "
            "urine is a cheap, non-invasive diagnostic biomarker that should be checked before "
            "costly gene panels when Costeff is suspected."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("AR Costeff (c.143-1G>A Iraqi-Jewish founder homozygous)", 0.45),
            ("AR compound heterozygous — non-founder alleles", 0.20),
            ("AD dominant — OA + early cataract (French-Canadian / European)", 0.35),
        ],
        "age_onset_years_range": (1, 15),
        "sex_ratio_M": 0.50,
        "rates": {
            "optic_atrophy_bilateral":          0.98,
            "elevated_urine_3mgc_ar_form":      0.85,
            "chorea_ar_form":                   0.75,
            "spastic_paraplegia_ar_form":       0.70,
            "early_cataract":                   0.40,
            "early_infantile_onset_ar":         0.60,
            "mild_extrapyramidal_ad_form":      0.30,
            "intellectual_disability":          0.25,
            "family_history":                   0.80,
            "urine_3mgc_sent_as_first_test":    0.50,
        },
        "hallmarks": [
            "Costeff (AR): optic atrophy + chorea + spastic paraplegia — the triad",
            "Urine 3-methylglutaconic acid elevated in AR form — cheap, fast, non-invasive",
            "Iraqi-Jewish ancestry: high index of suspicion; c.143-1G>A splice homozygous",
            "AD form: optic atrophy + early-onset cataract ± subclinical extrapyramidal features",
            "Chorea onset after optic atrophy — variable timing (1st–3rd decade)",
            "No treatment for chorea — tetrabenazine occasionally used symptomatically",
        ],
        "treatment_alerts": [
            "Send urine 3-MGC (organic acids) as first-line test when Costeff suspected",
            "No disease-modifying therapy for OPA3 (AR or AD)",
            "Chorea: tetrabenazine or deutetrabenazine for disabling cases",
            "Spastic paraplegia: baclofen, physiotherapy, orthotics",
            "Iraqi-Jewish patients: screen all first-degree relatives (carrier frequency ~1/10 in community)",
            "Cataract surgery if visually significant — does not alter optic atrophy",
            "Avoid agents worsening chorea (dopamine antagonists that increase receptor sensitivity)",
        ],
        "primary_treatment": (
            "No curative treatment. Urine organic acids (3-MGC) to confirm AR. "
            "Chorea management: tetrabenazine (start 12.5 mg BD, titrate). "
            "Spasticity: baclofen 5–20 mg TDS. Low vision aids. "
            "Genetic cascade testing; Iraqi-Jewish families: community screening."
        ),
    },

    # ── MT-ND4 — LHON m.11778G>A ──
    {
        "gene": "MT-ND4",
        "protein": "NADH:Ubiquinone Oxidoreductase Core Subunit ND4 (Complex I)",
        "alias": (
            "MT-ND4; OMIM 516003; LHON #535000; mtDNA; Maternal inheritance; "
            "m.11778G>A most common primary LHON mutation (~70% of all LHON cases worldwide); "
            "Subacute painless central visual loss — sequential eye involvement (2nd eye typically 6–8 wk later); "
            "Male predominance: ~80% penetrance in male carriers vs ~25% in female carriers; "
            "WORST visual prognosis of three primary mutations — only ~20% experience partial spontaneous recovery; "
            "Complex I ND4 subunit — impairs electron transport chain → excess superoxide → RGC death; "
            "Idebenone (Raxone) EMA-approved 2015 (early stage); lenadogene nolparvovec (Lumevoq) — intravitreal AAV2-ND4 gene therapy"
        ),
        "aa": "459 aa (mtDNA encoded)",
        "kDa": "~52 kDa",
        "locus": "mtDNA (position 11778)",
        "omim_gene": 516003,
        "omim_disease": 535000,
        "inheritance": "Maternal (mitochondrial) — 100% matrilineal transmission; heteroplasmy in subset; penetrance sex-biased (~80% male, ~25% female)",
        "gene_class": (
            "MT-ND4 encodes the ND4 subunit of mitochondrial complex I, the first enzyme of the "
            "respiratory chain. The m.11778G>A pathogenic variant (p.Arg340His) disrupts the "
            "proton-pumping mechanism and increases reactive oxygen species production. RGCs — with "
            "their high metabolic demand and unmyelinated intraretinal axons — are selectively "
            "vulnerable. All maternally-related family members share the same mtDNA haplotype; "
            "penetrance is strongly sex-biased (testosterone appears protective in females; "
            "exogenous testosterone risks unmasking disease in female carriers). m.11778G>A "
            "carries the worst visual prognosis; only ~20% of affected individuals recover "
            "meaningful vision, compared with ~70% for MT-ND6 (m.14484T>C)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("m.11778G>A homoplasmic — classic LHON presentation", 0.82),
            ("m.11778G>A heteroplasmic (>60% mutant load) — similar phenotype", 0.12),
            ("m.11778G>A with secondary mtDNA background variant (haplogroup J) — increased penetrance", 0.06),
        ],
        "age_onset_years_range": (15, 45),
        "sex_ratio_M": 0.82,
        "rates": {
            "bilateral_sequential_vision_loss": 0.95,
            "central_scotoma_at_presentation":  0.97,
            "male_sex":                         0.82,
            "visual_acuity_worse_than_6_60":    0.85,
            "spontaneous_partial_recovery":     0.20,
            "idebenone_initiated":              0.65,
            "tobacco_use_at_onset":             0.55,
            "haplogroup_j_background":          0.25,
            "family_history_maternal":          0.75,
            "disc_pseudooedema_acute_phase":    0.60,
            "colour_vision_severely_impaired":  0.92,
            "contrast_sensitivity_loss":        0.93,
        },
        "hallmarks": [
            "Subacute painless central visual loss — acute phase: peripapillary telangiectatic microangiopathy + disc pseudooedema",
            "Sequential eye involvement: first eye then second ~2–8 weeks later",
            "Male carriers: ~80% lifetime penetrance; female carriers: ~25%",
            "WORST prognosis of three primary mutations — only ~20% partial recovery",
            "Haplogroup J background (haplogroup J1c, J2b) → increased penetrance",
            "Tobacco and alcohol use are environmental triggers — strongly avoid",
            "Idebenone must be started within 1 year of onset in the better-seeing eye to benefit",
        ],
        "treatment_alerts": [
            "IDEBENONE (Raxone) 900 mg/day in 3 doses — initiate within 1 year of onset; start as soon as LHON confirmed",
            "LENADOGENE NOLPARVOVEC (Lumevoq): AAV2-ND4 intravitreal injection — first eye only; EUGENIA trial shows contralateral benefit (bystander effect)",
            "AVOID tobacco absolutely — smoking is the strongest environmental modifier; nicotine compounds mitochondrial dysfunction",
            "AVOID alcohol — synergistic RGC toxicity",
            "AVOID linezolid, ethambutol, amiodarone, chloramphenicol — all mitochondrial toxic, worsen LHON",
            "FEMALE CARRIERS: do NOT prescribe exogenous testosterone — risk of unmasking LHON",
            "Genetic counselling: ALL maternally-related relatives at risk; test mtDNA not nuclear DNA",
            "OCT: peripapillary RNFL — use to monitor disease stage and recovery",
        ],
        "primary_treatment": (
            "Idebenone 900 mg/day in 3 divided doses — start immediately, continue ≥2 years. "
            "If within eligibility criteria: lenadogene nolparvovec (Lumevoq) intravitreal. "
            "Absolute: stop smoking, stop alcohol. Avoid mitochondrial-toxic drugs. "
            "Low vision rehabilitation early. Annual ophthalmology monitoring (OCT RNFL, VEP, fundus)."
        ),
    },

    # ── MT-ND1 — LHON m.3460G>A ──
    {
        "gene": "MT-ND1",
        "protein": "NADH:Ubiquinone Oxidoreductase Core Subunit ND1 (Complex I)",
        "alias": (
            "MT-ND1; OMIM 516000; LHON #535000; mtDNA; Maternal; "
            "m.3460G>A second most common primary LHON mutation (~13% worldwide); "
            "Poor visual prognosis — similar to ND4; "
            "LHON-Plus: subset of MT-ND1 (and MT-ND6) develop MS-like CNS demyelinating lesions (LHON/MS overlap); "
            "Leigh-like encephalopathy in rare severe MT-ND1 variants; "
            "Heteroplasmy threshold: >60% mutant load typically required for phenotypic expression"
        ),
        "aa": "318 aa (mtDNA encoded)",
        "kDa": "~36 kDa",
        "locus": "mtDNA (position 3460)",
        "omim_gene": 516000,
        "omim_disease": 535000,
        "inheritance": "Maternal (mitochondrial) — m.3460G>A; usually homoplasmic; penetrance sex-biased (~75% male, ~20% female)",
        "gene_class": (
            "MT-ND1 encodes complex I's ND1 core subunit, integral to the proton-pumping arm. "
            "The m.3460G>A variant (p.Ala52Thr) impairs complex I assembly and increases "
            "superoxide generation in RGC axons. The clinical phenotype is classic LHON, but "
            "MT-ND1 carriers have a higher association with LHON-Plus — an overlap phenotype "
            "with MS-like CNS white matter lesions, where affected individuals develop both "
            "optic neuropathy and relapsing CNS demyelination. This variant is the least "
            "common of the three primary LHON mutations and is disproportionately represented "
            "in European patients. Treatment parallels MT-ND4 (idebenone; gene therapy trials ongoing)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("m.3460G>A homoplasmic — classic LHON", 0.80),
            ("m.3460G>A + LHON-Plus (MS-like white matter lesions)", 0.15),
            ("m.3460G>A heteroplasmic with Leigh-like features (rare)", 0.05),
        ],
        "age_onset_years_range": (15, 45),
        "sex_ratio_M": 0.78,
        "rates": {
            "bilateral_sequential_vision_loss": 0.94,
            "central_scotoma":                  0.96,
            "male_sex":                         0.78,
            "poor_visual_prognosis":            0.80,
            "lhon_plus_ms_like_lesions":        0.15,
            "idebenone_initiated":              0.60,
            "tobacco_use_at_onset":             0.52,
            "spontaneous_partial_recovery":     0.22,
            "disc_pseudooedema_acute":          0.55,
            "mri_cns_white_matter_lesions":     0.15,
        },
        "hallmarks": [
            "Classic LHON phenotype — subacute painless central visual loss, sequential eyes",
            "LHON-Plus (MT-ND1/ND6 association): MS-like CNS lesions + optic neuropathy",
            "If brain MRI shows white matter lesions + LHON → test MT-ND1 AND MT-ND6 specifically",
            "Heteroplasmy threshold: >60% mutant load needed for expression — heteroplasmy testing essential",
            "Poor visual prognosis — similar to MT-ND4; only ~22% partial recovery",
            "European carrier enrichment — relative to MT-ND6/ND4 which are more global",
        ],
        "treatment_alerts": [
            "Idebenone 900 mg/day — same protocol as MT-ND4",
            "LHON-Plus: distinguish from primary MS before starting immunosuppressants — DMTs NOT indicated for LHON component",
            "Brain MRI if any CNS symptoms — white matter lesions change management",
            "Avoid mitochondrial-toxic drugs — same list as MT-ND4",
            "Stop smoking, stop alcohol",
            "Maternal relatives: mtDNA testing; both m.11778, m.3460, m.14484 should be tested as primary screen",
        ],
        "primary_treatment": (
            "Idebenone 900 mg/day. Avoid tobacco and alcohol. "
            "LHON-Plus: neurology co-management; brain MRI to characterise lesions; "
            "DMTs not indicated for optic neuropathy component. Gene therapy trials (ND1) ongoing."
        ),
    },

    # ── MT-ND6 — LHON m.14484T>C ──
    {
        "gene": "MT-ND6",
        "protein": "NADH:Ubiquinone Oxidoreductase Core Subunit ND6 (Complex I)",
        "alias": (
            "MT-ND6; OMIM 516006; LHON #535000; mtDNA; Maternal; "
            "m.14484T>C third primary LHON mutation (~14% worldwide); "
            "BEST visual prognosis of three primary mutations — ~70% spontaneous partial recovery; "
            "Youngest typical age of onset (teens) among LHON primary mutations; "
            "Incomplete penetrance — most maternal carriers never develop visual symptoms; "
            "LHON-Plus association: MS-like white matter lesions (shared with MT-ND1)"
        ),
        "aa": "174 aa (mtDNA encoded)",
        "kDa": "~19 kDa",
        "locus": "mtDNA (position 14484)",
        "omim_gene": 516006,
        "omim_disease": 535000,
        "inheritance": "Maternal (mitochondrial) — m.14484T>C; usually homoplasmic; incomplete penetrance; males ~60%, females ~8%",
        "gene_class": (
            "MT-ND6 encodes complex I's ND6 transmembrane subunit on the light strand of mtDNA. "
            "The m.14484T>C variant (p.Met64Val) causes the mildest of the three primary LHON "
            "phenotypes, with the youngest onset and by far the best prognosis (~70% partial "
            "spontaneous recovery). The mechanism involves impaired complex I assembly and "
            "reduced electron transfer. MT-ND6 also associates with LHON-Plus (MS-like overlap) "
            "more frequently than MT-ND4. Most maternal carriers are entirely asymptomatic — "
            "penetrance is the lowest of the three primary mutations."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("m.14484T>C homoplasmic — classic LHON, best prognosis", 0.85),
            ("m.14484T>C + LHON-Plus (MS-like white matter lesions)", 0.12),
            ("m.14484T>C heteroplasmic", 0.03),
        ],
        "age_onset_years_range": (10, 35),
        "sex_ratio_M": 0.72,
        "rates": {
            "bilateral_sequential_vision_loss": 0.92,
            "central_scotoma":                  0.95,
            "male_sex":                         0.72,
            "best_spontaneous_recovery":        0.70,
            "lhon_plus_ms_like_lesions":        0.12,
            "idebenone_initiated":              0.65,
            "tobacco_use_at_onset":             0.50,
            "youngest_onset_vs_nd4_nd1":        0.75,
            "disc_pseudooedema_acute":          0.58,
            "family_history_asymptomatic_carriers": 0.70,
        },
        "hallmarks": [
            "BEST prognosis of three primary LHON mutations — ~70% recover useful central vision spontaneously",
            "Youngest typical onset — teens; onset range 10–35 yr",
            "Lowest penetrance of primary mutations — most maternal carriers never affected",
            "LHON-Plus (MS-like) association shared with MT-ND1",
            "Same acute-phase disc pseudooedema as other LHON variants",
            "Recovery usually begins 4–12 months after nadir vision loss",
        ],
        "treatment_alerts": [
            "Idebenone — still recommended even though spontaneous recovery common; start early",
            "Prognosis counselling critical: '70% recover' applies specifically to m.14484T>C, NOT ND4 or ND1",
            "Do NOT conflate prognosis of different mutations — quote mutation-specific recovery rates",
            "Avoid tobacco and alcohol — modify natural history",
            "Brain MRI if any CNS symptoms — LHON-Plus overlap",
            "Low vision services early — recovery takes months; don't wait to refer",
            "Maternal relatives: test mtDNA; asymptomatic carriers: avoid triggers (smoking, extreme exercise, hypoxia)",
        ],
        "primary_treatment": (
            "Idebenone 900 mg/day. Stop smoking. Avoid alcohol. "
            "Prognosis: quote m.14484T>C-specific ~70% recovery rate. "
            "Low vision aids early (recovery may still take 4–12 months). "
            "Brain MRI if CNS symptoms. Asymptomatic carrier advice: avoid triggers."
        ),
    },

    # ── WFS1 — Wolfram Syndrome / DIDMOAD ──
    {
        "gene": "WFS1",
        "protein": "Wolframin (ER Transmembrane Glycoprotein)",
        "alias": (
            "WFS1; OMIM gene 606201; Wolfram syndrome #222300; DFNA6/14/38 (AD SNHL) #600510; 4p16.1; 890 aa; ~100 kDa; "
            "AR: Wolfram syndrome (DIDMOAD): Diabetes Insipidus + Diabetes Mellitus + Optic Atrophy + Deafness; "
            "Onset sequence: type 1 DM (1st decade) → OA (2nd) → DI + SNHL (3rd) → neurodegeneration (4th); "
            "Wolframin is an ER transmembrane protein — WFS1 LOF → ER calcium dysregulation → UPR activation → neuronal and beta-cell death; "
            "AD: heterozygous WFS1 → isolated SNHL (DFNA6/14/38) — far milder than AR Wolfram; "
            "No disease-modifying treatment for AR Wolfram; clinical trials: GLP-1 receptor agonists, sodium valproate, dantrolene"
        ),
        "aa": "890 aa",
        "kDa": "~100 kDa",
        "locus": "4p16.1",
        "omim_gene": 606201,
        "omim_disease": 222300,
        "inheritance": "AR (Wolfram syndrome, DIDMOAD) or AD (isolated low-frequency SNHL = DFNA6/14/38, or mild OA)",
        "gene_class": (
            "WFS1 encodes wolframin, a nine-transmembrane ER protein critical for ER calcium "
            "homeostasis and the unfolded protein response (UPR). Homozygous or compound "
            "heterozygous LOF variants produce Wolfram syndrome — a progressive neurodegenerative "
            "condition with the DIDMOAD tetrad. Optic atrophy appears in the second decade, "
            "typically after type 1 diabetes has already been diagnosed. The progressive "
            "neurodegeneration is inexorable: brainstem atrophy, neurogenic bladder, psychiatric "
            "disorder, and peripheral neuropathy develop over the third and fourth decades. "
            "Heterozygous carriers can develop isolated low-frequency SNHL (DFNA6/14/38), "
            "occasionally with mild optic atrophy."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("AR Wolfram (DIDMOAD) — compound heterozygous WFS1 LOF", 0.55),
            ("AR Wolfram — homozygous WFS1 LOF (consanguineous)", 0.30),
            ("AD — DFNA6/14/38 heterozygous (isolated SNHL ± mild OA)", 0.15),
        ],
        "age_onset_years_range": (1, 20),
        "sex_ratio_M": 0.52,
        "rates": {
            "type1_diabetes_mellitus":          0.98,
            "optic_atrophy":                    0.95,
            "diabetes_insipidus":               0.80,
            "sensorineural_hearing_loss":       0.75,
            "neurogenic_bladder":               0.60,
            "brainstem_atrophy_on_mri":         0.55,
            "psychiatric_disorder":             0.45,
            "peripheral_neuropathy":            0.50,
            "ataxia_or_cerebellar_features":    0.40,
            "anosmia":                          0.20,
            "dm_precedes_optic_atrophy":        0.90,
            "consanguinity_ar_form":            0.35,
        },
        "hallmarks": [
            "DIDMOAD sequence: DM (1st decade) → OA (2nd decade) → DI + SNHL (3rd decade) → neurodegeneration",
            "Type 1 DM presenting in childhood — often diagnosed before Wolfram suspected",
            "Optic atrophy appears AFTER DM — triad of DM + OA in child = screen for WFS1",
            "Brainstem and cerebellar atrophy on MRI — pontine atrophy prominent",
            "Neurogenic bladder: hydronephrosis risk — check upper urinary tract annually",
            "Psychiatric disorder (depression, anxiety, psychosis) in up to 60%",
        ],
        "treatment_alerts": [
            "No disease-modifying treatment approved for Wolfram syndrome",
            "Insulin therapy for type 1 DM (C-peptide absent — true autoimmune + ER-stress beta-cell death)",
            "DDAVP (desmopressin) nasal/oral for central diabetes insipidus",
            "Annual urological review: neurogenic bladder → hydronephrosis → CKD risk; intermittent catheterisation if needed",
            "Annual brain MRI from age 15 — brainstem atrophy pattern tracks neurodegeneration",
            "Psychiatric support proactively — depression is common and undertreated",
            "Clinical trials: GLP-1 agonists (semaglutide, liraglutide), sodium valproate (reduce ER stress), dantrolene",
            "Genetic counselling: AR — 25% recurrence; both parents carriers; test siblings",
        ],
        "primary_treatment": (
            "Insulin (DM management); DDAVP for DI; hearing aids for SNHL. "
            "Urological: bladder US + urodynamics annually. Brain MRI annually. "
            "Clinical trial referral — Wolfram Syndrome International Registry. "
            "Psychiatric support. Multidisciplinary (endocrine, ophthalmology, urology, neurology, psychiatry). "
            "Genetic cascade testing: both parents carriers; screen siblings."
        ),
    },

    # ── TMEM126A — Autosomal Recessive Optic Atrophy ──
    {
        "gene": "TMEM126A",
        "protein": "Transmembrane Protein 126A (Mitochondrial Complex I Assembly Factor)",
        "alias": (
            "TMEM126A; OMIM gene 612988; AROA #612988; 11q14.1; 149 aa; ~17 kDa; "
            "Autosomal Recessive Optic Atrophy (AROA) — pure, slowly progressive optic neuropathy; "
            "North African / Moroccan founder variant: p.Arg55* (c.163C>T) nonsense — enriched in this population; "
            "TMEM126A is a mitochondrial complex I assembly factor — LOF impairs complex I biogenesis; "
            "Mild SNHL in subset (up to 30%); pure optic neuropathy in majority; "
            "No systemic neurological features (DDx Wolfram syndrome, Behr syndrome); "
            "Slowly progressive; blue-yellow colour axis impaired first; centrocecal scotoma"
        ),
        "aa": "149 aa",
        "kDa": "~17 kDa",
        "locus": "11q14.1",
        "omim_gene": 612988,
        "omim_disease": 612988,
        "inheritance": "AR — biallelic LOF; p.Arg55* founder in North African/Moroccan populations; compound het in European patients",
        "gene_class": (
            "TMEM126A encodes a small mitochondrial inner membrane protein required for early "
            "steps in complex I assembly. Biallelic loss-of-function produces a pure optic "
            "neuropathy phenotype that is often slowly progressive and milder than OPA1-ADOA. "
            "The p.Arg55* (c.163C>T) nonsense variant is a founder allele highly enriched in "
            "Moroccan and North African Jewish communities. In most patients the phenotype is "
            "limited to optic atrophy; mild SNHL is an occasional additional feature but "
            "systemic neurological manifestations are absent, distinguishing TMEM126A-AROA "
            "from Wolfram syndrome and Behr syndrome."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("p.Arg55* (c.163C>T) founder homozygous — North African/Moroccan", 0.55),
            ("Compound heterozygous TMEM126A LOF — European patients", 0.35),
            ("Novel homozygous missense in consanguineous non-founder family", 0.10),
        ],
        "age_onset_years_range": (2, 20),
        "sex_ratio_M": 0.50,
        "rates": {
            "optic_atrophy_bilateral":              0.98,
            "slowly_progressive_course":            0.88,
            "centrocecal_scotoma":                  0.85,
            "blue_yellow_colour_axis_first":        0.80,
            "mild_snhl_in_subset":                  0.30,
            "no_systemic_neurological_features":    0.92,
            "north_african_moroccan_ancestry":      0.55,
            "consanguinity":                        0.45,
            "family_history_optic_atrophy":         0.72,
            "visual_acuity_worse_than_6_18":        0.65,
        },
        "hallmarks": [
            "Pure recessive optic atrophy — no multisystem neurological disease (key DDx from Wolfram)",
            "North African/Moroccan founder: p.Arg55* — screen this variant first in population",
            "Slowly progressive — milder than OPA1-ADOA in many but bilateral",
            "Blue-yellow (tritan) colour axis impaired first — similar to OPA1",
            "Mild SNHL in ~30% — but no DM, no DI, no ataxia (distinguishes from Wolfram)",
            "Temporal optic disc pallor; OCT RNFL: temporal sector thinning",
        ],
        "treatment_alerts": [
            "No disease-modifying treatment",
            "North African/Moroccan ancestry: test p.Arg55* first — cheap, rapid Sanger confirmation",
            "Screen for SNHL: audiogram at diagnosis and every 2–3 years",
            "Genetic counselling: AR — 25% recurrence; both parents are obligate carriers",
            "Low vision rehabilitation; tinted lenses if photophobia",
            "Annual OCT RNFL to track progression rate (most patients slowly progressive)",
        ],
        "primary_treatment": (
            "No approved treatment. Low vision aids. Annual OCT RNFL monitoring. "
            "Audiogram at diagnosis. Genetic counselling: AR inheritance. "
            "North African ancestry: p.Arg55* targeted testing as first line."
        ),
    },

    # ── ACO2 — Infantile Optic Atrophy + Cerebellar Ataxia ──
    {
        "gene": "ACO2",
        "protein": "Aconitase 2 (Mitochondrial Aconitase / TCA Cycle Enzyme)",
        "alias": (
            "ACO2; OMIM gene 100850; Infantile OA + cerebellar ataxia #616147; 22q13.2; 780 aa; ~83 kDa; "
            "Severe early-onset syndromic optic atrophy + cerebellar ataxia + hypotonia + intellectual disability; "
            "TCA cycle enzyme: catalyses citrate → isocitrate via cis-aconitate — ACO2 LOF → mitochondrial energy failure; "
            "MRI: cerebellar atrophy ± periventricular white matter signal; elevated lactate in subset; "
            "Visual impairment from infancy — often presenting feature before ataxia apparent; "
            "Compound heterozygotes most common; homozygous founder variants in consanguineous families"
        ),
        "aa": "780 aa",
        "kDa": "~83 kDa",
        "locus": "22q13.2",
        "omim_gene": 100850,
        "omim_disease": 616147,
        "inheritance": "AR — compound heterozygous or homozygous; de novo dominant ACO2 variants cause isolated infantile optic atrophy in rare cases",
        "gene_class": (
            "ACO2 encodes mitochondrial aconitase, the second enzyme of the TCA cycle, which "
            "catalyses the stereospecific conversion of citrate to isocitrate via cis-aconitate. "
            "ACO2 contains a [4Fe-4S] iron-sulfur cluster essential for catalysis. Biallelic "
            "LOF abolishes TCA cycle flux, severely impairing NADH production and ATP synthesis. "
            "RGCs, cerebellar Purkinje cells, and cortical neurons — all with exceptionally high "
            "mitochondrial energy demands — are preferentially vulnerable. The clinical triad "
            "of infantile optic atrophy + cerebellar ataxia + hypotonia is characteristic. "
            "Cerebral MRI typically shows cerebellar atrophy; elevated lactate on MRS or CSF "
            "analysis supports mitochondrial dysfunction. No disease-modifying treatment exists."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("Compound heterozygous ACO2 missense/LOF — most common", 0.60),
            ("Homozygous ACO2 LOF in consanguineous family", 0.30),
            ("De novo dominant ACO2 (isolated infantile OA variant)", 0.10),
        ],
        "age_onset_years_range": (0, 3),
        "sex_ratio_M": 0.52,
        "rates": {
            "optic_atrophy_infantile":          0.98,
            "cerebellar_ataxia":                0.88,
            "hypotonia":                        0.82,
            "intellectual_disability":          0.78,
            "mri_cerebellar_atrophy":           0.85,
            "elevated_lactate_csf_blood":       0.45,
            "periventricular_white_matter":     0.35,
            "nystagmus":                        0.55,
            "seizures":                         0.30,
            "consanguinity":                    0.40,
            "visual_impairment_from_birth":     0.80,
            "no_effective_treatment":           0.99,
        },
        "hallmarks": [
            "Infantile-onset visual impairment — often first presenting feature before ataxia apparent",
            "Cerebellar ataxia develops after 1st year — MRI: cerebellar cortical atrophy",
            "Hypotonia in neonatal period preceding ataxia",
            "Elevated lactate (blood/CSF/MRS) — mitochondrial metabolic marker",
            "Severe: no meaningful treatment; supportive care dominates",
            "[4Fe-4S] cluster in ACO2 — iron-sulfur cluster assembly variants rarely co-present",
        ],
        "treatment_alerts": [
            "No disease-modifying treatment — supportive care only",
            "Metabolic supplementation trials: riboflavin, CoQ10, thiamine — no proven benefit but often tried",
            "Seizure management: sodium valproate AVOID (inhibits mitochondrial function further); use levetiracetam, lamotrigine",
            "Nystagmus: prism glasses, botulinum toxin to rectus muscles in selected cases",
            "MRI: baseline + 2-yearly to track cerebellar and cerebral atrophy progression",
            "Consanguinity counselling: 25% recurrence; prenatal diagnosis / PGT available",
            "Multidisciplinary: paediatric neurology, ophthalmology, metabolic, speech therapy, physiotherapy, OT",
        ],
        "primary_treatment": (
            "Supportive: physiotherapy, OT, speech therapy. Seizures: levetiracetam or lamotrigine "
            "(avoid valproate). Nystagmus: prism glasses. Metabolic: riboflavin + CoQ10 trial. "
            "Annual MRI. Genetic counselling: 25% recurrence risk; PGT/prenatal diagnosis. "
            "Low vision services from infancy. Metabolic team review."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Patient Simulation
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_patients(gene_def: dict) -> list:
    rng = random.Random(gene_def["seed"])
    patients = []
    ages = list(range(gene_def["age_onset_years_range"][0], gene_def["age_onset_years_range"][1] + 1))
    n = gene_def["n_patients"]

    for i in range(n):
        age_onset = rng.choice(ages)

        # Etiology
        r = rng.random()
        cum = 0.0
        etiology = gene_def["etiologies"][-1][0]
        for label, frac in gene_def["etiologies"]:
            cum += frac
            if r < cum:
                etiology = label
                break

        # Clinical features from rates
        features = {}
        for feat, rate in gene_def["rates"].items():
            features[feat] = rng.random() < rate

        sex = "M" if rng.random() < gene_def["sex_ratio_M"] else "F"

        patients.append({
            "id": i + 1,
            "gene": gene_def["gene"],
            "age_onset": age_onset,
            "sex": sex,
            "etiology": etiology,
            "features": features,
        })
    return patients


def _aggregate_stats(patients: list, rates: dict) -> dict:
    if not patients:
        return {}
    n = len(patients)
    return {k: round(sum(p["features"].get(k, False) for p in patients) / n * 100, 1) for k in rates}


# ─────────────────────────────────────────────────────────────────────────────
# Build all cohorts once
# ─────────────────────────────────────────────────────────────────────────────
_ALL_PATIENTS: dict = {}
_ALL_STATS: dict = {}

for _gd in HON_GENES:
    _pts = _simulate_patients(_gd)
    _ALL_PATIENTS[_gd["gene"]] = _pts
    _ALL_STATS[_gd["gene"]] = _aggregate_stats(_pts, _gd["rates"])


# ─────────────────────────────────────────────────────────────────────────────
# API Data Functions
# ─────────────────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """Overview — aggregate stats across all 320 patients."""
    all_pts = [p for pts in _ALL_PATIENTS.values() for p in pts]
    n = len(all_pts)

    # Cross-gene aggregate stats
    def _pct(key: str) -> float:
        return round(sum(p["features"].get(key, False) for p in all_pts) / n * 100, 1)

    genes = [g["gene"] for g in HON_GENES]

    top_alerts = [
        "OPA1-HAPLOINSUFFICIENCY: classic ADOA; GTPase-missense → OPA1-plus (ptosis+CPEO+myopathy)",
        "OPA3-AR-COSTEFF: 3-MGC urine assay FIRST — cheap diagnostic; Iraqi-Jewish founder c.143-1G>A",
        "LHON-AVOID-TOBACCO: strongest environmental modifier — ABSOLUTE contraindication in all LHON",
        "MT-ND4-WORST-PROGNOSIS: only 20% recover; MT-ND6-BEST: 70% recover — quote mutation-specific",
        "WFS1-DIDMOAD: DM precedes OA by years — triad DM+OA in child = WFS1 first",
        "WFS1-BLADDER: annual urological review — neurogenic bladder → hydronephrosis → CKD",
        "TMEM126A-FOUNDER: North African p.Arg55* — targeted Sanger before panel in population",
        "ACO2-VALPROATE-AVOID: inhibits mitochondrial function — use levetiracetam/lamotrigine for seizures",
        "LHON-IDEBENONE: start within 1 year of onset (better-seeing eye), 900 mg/day in 3 doses",
        "LHON-MATERNAL-GENETICS: test mtDNA not nuclear DNA for maternal relatives",
        "OPA1-PLUS-mtDNA-DELETIONS: secondary multiple mtDNA deletions in muscle — muscle biopsy if atypical",
        "MT-ND1-ND6-LHON-PLUS: MS-like white matter lesions — brain MRI if any CNS symptoms",
    ]

    diseases = {g["gene"]: g["alias"].split(";")[3].strip() + " — " + g["alias"].split(";")[4].strip()
                if ";" in g["alias"] else g["alias"][:120]
                for g in HON_GENES}

    return {
        "total_patients": n,
        "genes": genes,
        "seed_range": "1414–1421",
        "aggregate_stats": {
            "optic_atrophy_any": round(sum(
                any(p["features"].get(k, False) for k in [
                    "optic_atrophy_bilateral", "optic_atrophy_infantile",
                    "bilateral_sequential_vision_loss", "centrocecal_scotoma"
                ]) for p in all_pts) / n * 100, 1),
            "male_predominance_lhon": round(sum(
                p["sex"] == "M" and p["gene"] in ("MT-ND4", "MT-ND1", "MT-ND6")
                for p in all_pts) /
                max(sum(p["gene"] in ("MT-ND4", "MT-ND1", "MT-ND6") for p in all_pts), 1) * 100, 1),
            "lhon_tobacco_use": _pct("tobacco_use_at_onset"),
            "wolfram_dm_at_presentation": _pct("type1_diabetes_mellitus"),
            "didmoad_optic_atrophy": _pct("optic_atrophy"),
            "wolfram_neurogenic_bladder": _pct("neurogenic_bladder"),
            "opa1_plus_syndromic": _pct("opa1_plus_ptosis_cpeo_myopathy"),
            "costeff_3mgc_elevated": _pct("elevated_urine_3mgc_ar_form"),
            "lhon_best_prognosis_nd6": _pct("best_spontaneous_recovery"),
            "aco2_cerebellar_atrophy": _pct("mri_cerebellar_atrophy"),
            "tmem126a_north_african": _pct("north_african_moroccan_ancestry"),
            "lhon_plus_cns_lesions": round(
                sum(_pct(k) for k in ["lhon_plus_ms_like_lesions"]) / 1, 1),
        },
        "top_alerts": top_alerts,
        "diseases": diseases,
    }


def get_breakdown() -> dict:
    """Per-gene breakdown for Gene Table and Clinical Atlas tabs."""
    result = {}
    for gd in HON_GENES:
        gene = gd["gene"]
        pts = _ALL_PATIENTS[gene]
        stats = _ALL_STATS[gene]

        etiology_distribution = [
            {"etiology": label, "fraction": round(frac, 3)}
            for label, frac in gd["etiologies"]
        ]

        result[gene] = {
            "gene":                 gene,
            "protein":              gd["protein"],
            "aa":                   gd["aa"],
            "locus":                gd["locus"],
            "omim_gene":            gd["omim_gene"],
            "omim_disease":         gd["omim_disease"],
            "inheritance":          gd["inheritance"],
            "organ_system":         "Optic nerve / Retinal ganglion cells / Mitochondria",
            "n_patients":           gd["n_patients"],
            "seed":                 gd["seed"],
            "gene_class":           gd["gene_class"],
            "hallmarks":            gd["hallmarks"],
            "treatment_alerts":     gd["treatment_alerts"],
            "primary_treatment":    gd["primary_treatment"],
            "stats":                stats,
            "etiology_distribution": etiology_distribution,
        }
    return result


def get_definitions() -> dict:
    """Disease classification, diagnostic rules, and treatment hierarchies."""
    return {
        "classification": {
            "mitochondrial_fusion_GTPases": {
                "OPA1_ADOA": "AD optic atrophy — haploinsufficiency or GTPase domain dominant-negative (OPA1-plus)",
                "OPA3_Costeff": "AR Costeff syndrome — OA + chorea + spastic paraplegia + 3-MGC; AD: OA + cataract",
            },
            "lhon_primary_mutations": {
                "MT_ND4_m11778GA": "Most common LHON (~70%) — worst prognosis, only 20% recovery; male 80% penetrance",
                "MT_ND1_m3460GA": "Second primary LHON (~13%) — poor prognosis; LHON-Plus association (MS-like CNS)",
                "MT_ND6_m14484TC": "Third primary LHON (~14%) — BEST prognosis (~70% recovery); youngest onset",
            },
            "systemic_optic_neuropathies": {
                "WFS1_Wolfram": "AR DIDMOAD — DM→OA→DI→SNHL sequence; progressive neurodegeneration; ER calcium dysregulation",
                "TMEM126A_AROA": "AR pure OA — complex I assembly factor; North African/Moroccan founder p.Arg55*; mild SNHL possible",
                "ACO2_infantile": "AR infantile OA + cerebellar ataxia + hypotonia; TCA cycle aconitase; elevated lactate",
            },
        },
        "key_diagnostic_rules": {
            "OPA1_HAPLOINSUFFICIENCY_vs_OPA1_PLUS": (
                "GTPase domain missense → OPA1-plus (syndromic: ptosis + CPEO + myopathy + ataxia + SNHL + secondary mtDNA deletions). "
                "Truncating / splice / large deletion → classic haploinsufficiency ADOA (pure optic neuropathy). "
                "Variant location predicts phenotype — always characterise the variant class."
            ),
            "OPA3_URINE_3MGC_FIRST": (
                "Send urine organic acids (3-methylglutaconic acid) BEFORE gene panel in suspected Costeff. "
                "3-MGC elevated in AR OPA3 — cheap, fast non-invasive biomarker. "
                "Iraqi-Jewish ancestry + OA + chorea = OPA3 c.143-1G>A targeted testing first."
            ),
            "LHON_TEST_mtDNA_NOT_NUCLEAR": (
                "LHON is mitochondrial — test mtDNA (m.11778, m.3460, m.14484) as primary screen, not whole-exome sequencing. "
                "Exome/genome sequencing routinely MISSES mtDNA point variants — request dedicated mtDNA analysis. "
                "Blood leukocyte mtDNA is sufficient; muscle biopsy not needed for primary LHON mutations."
            ),
            "LHON_PROGNOSIS_IS_MUTATION_SPECIFIC": (
                "NEVER apply a single prognosis figure to all LHON. "
                "MT-ND4 (m.11778): only 20% partial recovery. "
                "MT-ND1 (m.3460): similar poor prognosis ~22%. "
                "MT-ND6 (m.14484): 70% partial recovery — must specify mutation when counselling."
            ),
            "WFS1_DM_PRECEDES_OA": (
                "In Wolfram syndrome, type 1 DM is always the first feature — typically diagnosed in the 1st decade. "
                "OA appears in the 2nd decade. Child with DM1 + progressive OA: WFS1 first. "
                "WFS1 diabetes is C-peptide negative (true beta-cell loss, NOT immune/autoimmune DM1 antibody positive)."
            ),
            "WFS1_ANNUAL_BLADDER_SURVEILLANCE": (
                "Neurogenic bladder in Wolfram syndrome leads to hydronephrosis → chronic kidney disease. "
                "Annual upper urinary tract ultrasound + urodynamics from diagnosis. "
                "Intermittent self-catheterisation when post-void residual >150 mL. "
                "Renal impairment is a major morbidity driver — do NOT miss bladder surveillance."
            ),
            "TMEM126A_NORTH_AFRICAN_FOUNDER": (
                "p.Arg55* (c.163C>T) is a founder allele highly enriched in Moroccan and North African Jewish communities. "
                "In these populations: targeted Sanger for p.Arg55* is faster and cheaper than full panel. "
                "Phenotype: pure slowly progressive OA with occasional SNHL — NO DM, NO DI, NO ataxia (DDx Wolfram)."
            ),
            "ACO2_AVOID_VALPROATE": (
                "Sodium valproate inhibits mitochondrial beta-oxidation and impairs the ETC — contraindicated in ACO2 deficiency. "
                "For seizures: use levetiracetam or lamotrigine as first-line. "
                "Never start valproate empirically in a child with optic atrophy + cerebellar ataxia — investigate mitochondrial cause first."
            ),
            "LHON_PLUS_BRAIN_MRI": (
                "LHON-Plus is MT-ND1 and MT-ND6 predominant. If LHON patient develops any CNS symptoms → brain MRI. "
                "MS-like white matter lesions on MRI + LHON = LHON-Plus; DMTs NOT indicated for optic neuropathy component. "
                "Neurology co-management essential; distinguish carefully from co-occurring MS."
            ),
        },
        "treatment_hierarchy": {
            "OPA1_ADOA": [
                "1. Low vision aids and eccentric viewing training from diagnosis",
                "2. Idebenone 900 mg/day off-label — consider in early progressive disease",
                "3. Avoid mitochondrial toxins (tobacco, alcohol, linezolid, aminoglycosides, NRTIs)",
                "4. OPA1-plus: multidisciplinary (ophthalmology, neurology, cardiology, respiratory)",
                "5. Genetic counselling: 50% transmission; de novo ~15%; variable expressivity",
                "6. OCT RNFL + VEP + colour vision annually",
            ],
            "MT_ND4_LHON": [
                "1. Idebenone (Raxone) 900 mg/day (300 mg TDS) — start within 1 year of onset",
                "2. Lenadogene nolparvovec (Lumevoq) intravitreal — if eligible (within 1 year, better-seeing eye first)",
                "3. ABSOLUTE: stop smoking; stop alcohol",
                "4. Avoid mitochondrial toxic drugs (linezolid, ethambutol, amiodarone, chloramphenicol)",
                "5. Low vision rehabilitation — referral early (recovery takes months)",
                "6. mtDNA testing of all maternal relatives — penetrance counselling by sex",
            ],
            "WFS1_Wolfram": [
                "1. Insulin for DM (C-peptide absent; treat as absolute insulin deficiency)",
                "2. DDAVP (desmopressin) for central DI",
                "3. Annual bladder ultrasound + urodynamics → ISC if PVR >150 mL",
                "4. Annual brain MRI from age 15 (brainstem atrophy tracking)",
                "5. Psychiatric support proactively — depression common",
                "6. Clinical trial referral (Wolfram Syndrome International Registry)",
                "7. Genetic counselling: AR, 25% recurrence; sibling testing",
            ],
            "TMEM126A_AROA": [
                "1. Targeted p.Arg55* testing first in North African/Moroccan ancestry",
                "2. No disease-modifying treatment — low vision aids",
                "3. Annual OCT RNFL to track progression",
                "4. Audiogram at diagnosis and every 2–3 years",
                "5. Genetic counselling: AR, 25% recurrence",
            ],
            "ACO2_Infantile_OA": [
                "1. Avoid sodium valproate — use levetiracetam/lamotrigine for seizures",
                "2. Riboflavin + CoQ10 metabolic supplementation trial",
                "3. Physiotherapy, OT, speech therapy from infancy",
                "4. Annual brain MRI",
                "5. Multidisciplinary team: paediatric neurology, metabolic, ophthalmology",
                "6. Consanguinity: 25% recurrence; PGT / prenatal diagnosis available",
            ],
        },
    }
