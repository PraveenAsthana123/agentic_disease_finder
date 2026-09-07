#!/usr/bin/env python3
"""Hereditary-Purine-Atlas — Complete 8-Gene Hereditary Purine & Pyrimidine Metabolism Disorder Atlas
HPRT1   (hypoxanthine-guanine phosphoribosyltransferase 1; 217 aa; Xq26.2; XLR;
         Lesch-Nyhan syndrome (complete) — self-injurious behaviour + hyperuricaemia + choreoathetosis + dystonia;
         Kelley-Seegmiller (partial) — gout + mild/no neurological features;
         allopurinol/febuxostat controls uric acid BUT NOT neurological symptoms;
         orange grit in nappies (uric acid crystals) — early clinical clue;
         uric acid >600 µmol/L first biochemical flag; no curative treatment; gene therapy investigational;
         seed SEED_BASE+0) .
ADSL    (adenylosuccinate lyase; 484 aa; 22q13.1; AR;
         adenylosuccinate lyase deficiency — SAICAR and SAdo elevated in urine and CSF PATHOGNOMONIC;
         3 phenotypes: neonatal lethal / severe (psychomotor retardation + seizures + autistic features) / mild;
         SAICAR/SAdo ratio in urine best biomarker;
         no specific curative treatment; allopurinol used but benefit unproven;
         seed SEED_BASE+1) .
ADA     (adenosine deaminase; 363 aa; 20q13.12; AR;
         ADA-SCID — toxic dATP accumulation kills T/B/NK cells (all 3 lineages);
         NBS detected via lymphocyte count + TREC assay;
         Strimvelis gene therapy (EMA 2016) ex vivo HSC gene therapy;
         PEG-ADA (elapegademase) enzyme replacement bridges to definitive therapy;
         HSCT curative; matched sibling donor optimal;
         seed SEED_BASE+2) .
PNP     (purine nucleoside phosphorylase; 289 aa; 14q11.2; AR;
         PNP deficiency — T-cell selective immunodeficiency (contrast ADA-SCID);
         dGTP accumulates (toxic to T cells specifically);
         autoimmune complications: haemolytic anaemia, thrombocytopenia, SLE-like;
         spastic diplegia and neurological features;
         HSCT curative but less successful than ADA-SCID;
         seed SEED_BASE+3) .
XDH     (xanthine dehydrogenase/oxidase; 1333 aa; 2p23.1; AR;
         xanthinuria type I — xanthine kidney stones (radiolucent on plain X-ray KEY DDx calcium stones);
         uric acid very low (<100 µmol/L) KEY DDx from gout;
         allopurinol CONTRAINDICATED (inhibits absent XDH; ineffective);
         high fluid intake main treatment;
         seed SEED_BASE+4) .
APRT    (adenine phosphoribosyltransferase; 180 aa; 16q24.3; AR;
         APRT deficiency / 2,8-DHA urolithiasis;
         2,8-dihydroxyadenine (2,8-DHA) crystals in urine — brownish crystals PATHOGNOMONIC;
         progressive renal failure if untreated;
         allopurinol CURATIVE — blocks xanthine oxidase upstream, reduces 2,8-DHA;
         Japanese founder effect (high prevalence ~1:250,000);
         seed SEED_BASE+5) .
AMPD1   (AMP deaminase 1; 747 aa; 1p13.3; AR;
         myoadenylate deaminase deficiency — exercise intolerance + cramps (significance debated);
         forearm exercise test: lactate RISES normally but ammonia does NOT RISE PATHOGNOMONIC;
         common polymorphism p.Gln12Ter in ~2% of European population homozygous;
         primary vs secondary forms important to distinguish;
         seed SEED_BASE+6) .
UMPS    (UMP synthase, bifunctional; 480 aa; 3q13.33; AR;
         hereditary orotic aciduria type I — megaloblastic anaemia + orotic acid crystalluria + growth retardation;
         NOT responsive to B12 or folate (key DDx from nutritional megaloblastic anaemia);
         uridine replacement CURATIVE (oral uridine/uridine triacetate);
         DDx OTC deficiency (both orotic aciduria — OTC has hyperammonaemia, UMPS does NOT);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1814-1821)
"""

import random

SEED_BASE = 1814

PURINE_GENES = [
    # -- HPRT1 -- Lesch-Nyhan / Kelley-Seegmiller ----------------------------
    {
        "gene": "HPRT1",
        "protein": (
            "HPRT1 -- Xq26.2 XLR -- Hypoxanthine-Guanine-Phosphoribosyltransferase-1-217aa -- "
            "Lesch-Nyhan-Syndrome-Complete-SIB-Hyperuricaemia-Choreoathetosis-Dystonia -- "
            "Kelley-Seegmiller-Partial-Gout-Mild-Neurological-Or-None -- "
            "Allopurinol-Controls-Uric-Acid-NOT-Neurological-Symptoms -- "
            "Orange-Grit-Nappies-Uric-Acid-Crystals-First-Clue -- "
            "No-Curative-Treatment-Gene-Therapy-Investigational"
        ),
        "alias": (
            "HPRT1 (hypoxanthine-guanine phosphoribosyltransferase 1); OMIM gene 308000; "
            "Lesch-Nyhan syndrome OMIM 300322. "
            "Xq26.2; 217 aa; ~25 kDa; cytoplasmic enzyme; X-linked recessive. "
            "FUNCTION: HPRT1 catalyses the purine salvage pathway — it rescues hypoxanthine and guanine "
            "from nucleotide degradation, converting them back to nucleoside monophosphates: "
            "hypoxanthine + PRPP → inosine monophosphate (IMP); "
            "guanine + PRPP → guanosine monophosphate (GMP). "
            "PRPP (phosphoribosyl pyrophosphate) is the co-substrate. "
            "Without HPRT1, hypoxanthine and guanine cannot be salvaged and are instead converted "
            "to uric acid via xanthine oxidase → hyperuricaemia and gout. "
            "Simultaneously, PRPP accumulates (not consumed by HPRT1) → drives de novo purine synthesis "
            "→ further uric acid overproduction. "
            "The neurological consequences result from failure of purine salvage in dopaminergic neurons "
            "of the basal ganglia — exact mechanism debated but involves dopamine depletion/dysfunction. "
            "CLINICAL SPECTRUM (genotype-phenotype correlates): "
            "1. COMPLETE HPRT1 DEFICIENCY — LESCH-NYHAN SYNDROME: "
            "Neurological: motor delay from 3-6 months; choreoathetosis, dystonia, spasticity emerging by 12 months; "
            "intellectual disability (moderate to severe); dysarthria/dysphagia; "
            "compulsive self-injurious behaviour (SIB) — biting of lips/fingers/hands PATHOGNOMONIC; "
            "SIB begins at 12-24 months; patient appears distressed and may request restraints; "
            "renal: uric acid stones, haematuria, renal failure if untreated; "
            "gout: tophi, arthritis (if hyperuricaemia not treated). "
            "2. PARTIAL HPRT1 DEFICIENCY — KELLEY-SEEGMILLER SYNDROME: "
            "Variable residual HPRT1 activity (1-10%); "
            "gout (often severe, early onset, before age 30); "
            "hyperuricaemia; kidney stones; "
            "minimal or no neurological involvement; "
            "some patients have mild cerebellar features or cognitive effects; "
            "diagnosed when young adult presents with early severe gout — HPRT1 screening warranted "
            "if gout onset <30 years, especially in males. "
            "BIOCHEMISTRY: "
            "Uric acid: >600 µmol/L (often 700-1000 µmol/L); urine uric acid/creatinine ratio elevated; "
            "orange grit in nappies: uric acid crystals (birefringent under polarised light) — "
            "early presenting sign in male neonates/infants; "
            "plasma oxypurines elevated (hypoxanthine, xanthine); "
            "PRPP levels elevated. "
            "DIAGNOSIS: "
            "HPRT1 enzyme activity in erythrocytes or fibroblasts (gold standard); "
            "gene sequencing: >500 pathogenic HPRT1 variants; "
            "carrier testing in females (X-linked — females usually asymptomatic carriers); "
            "prenatal diagnosis by chorionic villus sampling. "
            "TREATMENT: "
            "URIC ACID CONTROL: allopurinol (xanthine oxidase inhibitor) lowers uric acid efficiently; "
            "febuxostat: alternative XO inhibitor if allopurinol intolerant; "
            "urate target: <300 µmol/L; oral hydration + alkalinisation; "
            "CRITICAL: allopurinol/febuxostat controls uric acid and prevents gout/stones "
            "but has NO effect on neurological symptoms — SIB, choreoathetosis, dystonia unchanged; "
            "NEUROLOGICAL MANAGEMENT: "
            "restraints: padded restraints often requested by patients to prevent SIB — ethically appropriate; "
            "gabapentin, benzodiazepines: limited benefit for dystonia/SIB; "
            "baclofen intrathecal: modest benefit for spasticity in some; "
            "deep brain stimulation (DBS): investigational; "
            "GENE THERAPY: HPRT1 gene replacement — investigational phase I/II trials in progress (2026). "
        ),
        "locus": "Xq26.2",
        "aa": 217,
        "kDa": 25,
        "omim_gene": 308000,
        "omim_disease": 300322,
        "inheritance": "XLR",
        "gene_class": "cytoplasmic purine salvage enzyme — hypoxanthine/guanine → IMP/GMP",
        "key_alerts": [
            "HPRT1-ALLOPURINOL-CONTROLS-URIC-ACID-NOT-NEUROLOGY: Allopurinol and febuxostat effectively normalise uric acid and prevent gout/stones in HPRT1 deficiency, but have ZERO effect on the neurological phenotype (SIB, choreoathetosis, dystonia, intellectual disability) — do not conflate metabolic control with neurological improvement; set family expectations accordingly",
            "HPRT1-SIB-BEGINS-12-24-MONTHS: Self-injurious behaviour (biting of lips, fingers, hands) in Lesch-Nyhan syndrome begins between 12-24 months of age — it is compulsive, not intentional, and the patient appears distressed; padded restraints requested by patients are ethically appropriate and prevent injury",
            "HPRT1-GOUT-IN-CHILDREN-SCREEN-HPRT1: Gout in a child or young male (<30 years) with hyperuricaemia should prompt HPRT1 enzyme assay and gene testing — Kelley-Seegmiller partial HPRT1 deficiency presents as early-onset severe gout without classic Lesch-Nyhan neurology; commonly missed for years",
            "HPRT1-NO-CURATIVE-TREATMENT-GENE-THERAPY-INVESTIGATIONAL: There is currently no curative treatment for Lesch-Nyhan syndrome — gene therapy (HPRT1 AAV-based and ex vivo HSC approaches) remains investigational in Phase I/II trials (2026); do not promise curative gene therapy to families outside a clinical trial setting",
            "HPRT1-GENE-PANEL-PARTIAL-KELLEY-SEEGMILLER: HPRT1 sequencing identifies >500 pathogenic variants; partial HPRT1 deficiency (Kelley-Seegmiller) can be missed if only uric acid is checked without enzyme assay — enzyme activity in erythrocytes is the gold standard for diagnosing partial forms",
        ],
        "etiologies": {
            "classic_lesch_nyhan_complete": 55,
            "kelley_seegmiller_partial_gout_mild_neuro": 30,
            "intermediate_neurological_features": 10,
            "family_screening_newborn": 5,
        },
        "stats": {
            "disease": "Lesch-Nyhan syndrome / Kelley-Seegmiller syndrome",
            "incidence": "1:380,000 live births (Lesch-Nyhan complete)",
            "sib_onset_months": "12-24",
            "uric_acid_umol_L_typical": ">600",
            "allopurinol_uric_control_pct": 98,
            "allopurinol_neuro_benefit_pct": 0,
            "early_gout_kelley_seegmiller_pct": 30,
            "orange_nappy_grit_pct": 60,
        },
        "dx_delay_distribution": {
            "diagnosed_within_6m": 15,
            "diagnosed_6_24m": 50,
            "diagnosed_2_5y": 25,
            "diagnosed_after_5y": 10,
        },
    },

    # -- ADSL -- adenylosuccinate lyase deficiency ----------------------------
    {
        "gene": "ADSL",
        "protein": (
            "ADSL -- 22q13.1 AR -- Adenylosuccinate-Lyase-484aa -- "
            "SAICAR-SAdo-Urine-CSF-PATHOGNOMONIC-Biomarkers -- "
            "3-Phenotype-Spectrum-Neonatal-Lethal-Severe-Mild -- "
            "Autism-Features-Common-Seizures-Psychomotor-Retardation -- "
            "No-Specific-Curative-Treatment -- "
            "Gene-Panel-All-Purine-Disorders"
        ),
        "alias": (
            "ADSL (adenylosuccinate lyase); OMIM gene 608222; "
            "adenylosuccinate lyase deficiency OMIM 103050. "
            "22q13.1; 484 aa; ~54 kDa; homotetrameric cytoplasmic enzyme; autosomal recessive. "
            "FUNCTION: ADSL catalyses two reactions in purine de novo synthesis and the purine nucleotide cycle: "
            "Reaction 1 (de novo synthesis): SAICAR → AICAR + fumarate; "
            "Reaction 2 (AMP regeneration): adenylosuccinate (SAdo) → AMP + fumarate. "
            "Without ADSL, SAICAR (succinyl-AICA-ribotide) and SAdo (succinyladenosine) accumulate "
            "in cells, blood, urine, and cerebrospinal fluid. "
            "Both metabolites are toxic to neurons — exact mechanisms include inhibition of fumarase "
            "and disruption of the purine nucleotide cycle in the CNS. "
            "CLINICAL PHENOTYPES (3 main forms): "
            "1. NEONATAL LETHAL FORM: "
            "Neonatal encephalopathy, respiratory failure, refractory seizures; "
            "death within days to weeks; very high SAICAR/SAdo accumulation; "
            "null variants (complete loss of function). "
            "2. SEVERE FORM (MOST COMMON): "
            "Psychomotor retardation (severe to profound intellectual disability); "
            "seizures (various types — infantile spasms, tonic-clonic); "
            "autistic features (stereotypies, poor social contact, absent/limited speech); "
            "growth retardation; feeding difficulties; "
            "onset in infancy; hypotonia followed by spasticity; "
            "brain MRI: cerebral and cerebellar atrophy, white matter changes; "
            "EEG: diffuse slowing, multifocal epileptiform discharges. "
            "3. MILD FORM: "
            "Intellectual disability (mild to moderate); "
            "autistic features prominent; seizures may be absent or controlled; "
            "better neuromotor function; "
            "residual ADSL activity present (some missense variants). "
            "BIOMARKERS — PATHOGNOMONIC: "
            "SAICAR in urine: elevated — detectable by urine purines/succinylpurines (HPLC); "
            "SAdo (succinyladenosine) in urine and CSF: elevated — more specific; "
            "SAICAR:SAdo ratio: correlates with phenotype severity (high SAICAR relative to SAdo = more severe); "
            "plasma amino acids: normal (ADSL deficiency does not affect amino acids); "
            "urine organic acids: usually normal (ADSL metabolites not detected on standard OA screen); "
            "SPECIFIC TEST REQUIRED: urine succinylpurines (not routine NBS or metabolic screen). "
            "DIAGNOSIS: "
            "Urine succinylpurines (SAICAR + SAdo) — first-line screening; "
            "CSF SAdo confirms CNS involvement; "
            "ADSL enzyme assay in erythrocytes (may be normal if erythrocyte-specific isoform absent); "
            "ADSL gene sequencing (22q13.1; >100 pathogenic variants described). "
            "TREATMENT: "
            "No specific curative treatment available; "
            "allopurinol: used empirically to reduce purine overproduction; clinical benefit unproven in randomised trials; "
            "ribose supplementation: attempted to provide alternative purine precursors — unproven; "
            "seizure management: standard anticonvulsants (levetiracetam, valproate, clobazam); "
            "neurodevelopmental support: physiotherapy, speech therapy, special education; "
            "gene therapy and enzyme replacement: pre-clinical stage only (2026). "
        ),
        "locus": "22q13.1",
        "aa": 484,
        "kDa": 54,
        "omim_gene": 608222,
        "omim_disease": 103050,
        "inheritance": "AR",
        "gene_class": "cytoplasmic purine de novo synthesis and nucleotide cycle enzyme — SAICAR/SAdo lyase",
        "key_alerts": [
            "ADSL-SAICAR-SADO-URINE-PATHOGNOMONIC: SAICAR (succinyl-AICA-ribotide) and SAdo (succinyladenosine) in urine are pathognomonic for ADSL deficiency — routine metabolic screens (amino acids, standard organic acids) will MISS this diagnosis; request specific urine succinylpurines (HPLC) in any child with unexplained psychomotor retardation + seizures + autistic features",
            "ADSL-3-PHENOTYPE-SPECTRUM: ADSL deficiency spans neonatal lethal (null variants) → severe (psychomotor retardation + seizures + autism + infantile onset) → mild (mild-moderate ID + autism, controlled epilepsy) — same gene, same enzyme, different residual activity; do not tell families 'ADSL always means severe outcome'",
            "ADSL-AUTISM-FEATURES-COMMON: Autistic features (social withdrawal, stereotypies, communication deficit) are a prominent and consistent feature of ADSL deficiency across all phenotypes — ADSL gene panel should be included in ASD with intellectual disability workup, especially with comorbid epilepsy",
            "ADSL-CSF-SAICAR-DIAGNOSTIC: CSF succinyladenosine (SAdo) is a highly specific CNS biomarker for ADSL deficiency when urine results are borderline; lumbar puncture is warranted in high-suspicion cases with negative or borderline urine succinylpurines",
            "ADSL-GENE-PANEL-ALL-PURINE-DISORDERS: ADSL deficiency is not detected by standard NBS; comprehensive purine disorder gene panel is essential in children with unexplained epileptic encephalopathy + autistic regression — panel should include ADSL, HPRT1, ADA, PNP, UMPS, and related genes",
        ],
        "etiologies": {
            "severe_psychomotor_retardation_seizures": 60,
            "mild_intellectual_disability_autism": 25,
            "neonatal_lethal_encephalopathy": 10,
            "family_screening": 5,
        },
        "stats": {
            "disease": "Adenylosuccinate lyase deficiency",
            "incidence": "rare: ~200 reported cases worldwide",
            "saicar_sado_elevated_pct": 100,
            "seizure_prevalence_pct": 85,
            "autistic_features_pct": 75,
            "severe_phenotype_pct": 60,
            "allopurinol_use_pct": 50,
        },
        "dx_delay_distribution": {
            "diagnosed_within_1y": 20,
            "diagnosed_1_3y": 40,
            "diagnosed_3_10y": 30,
            "diagnosed_after_10y": 10,
        },
    },

    # -- ADA -- ADA-SCID -------------------------------------------------------
    {
        "gene": "ADA",
        "protein": (
            "ADA -- 20q13.12 AR -- Adenosine-Deaminase-363aa -- "
            "ADA-SCID-Most-Common-Enzyme-Deficiency-SCID -- "
            "Toxic-dATP-Kills-T-B-NK-All-3-Lymphocyte-Lineages -- "
            "NBS-TREC-Assay-Detects -- "
            "Strimvelis-EMA2016-Ex-Vivo-HSC-Gene-Therapy -- "
            "PEG-ADA-Elapegademase-Bridge-To-Definitive-Therapy -- "
            "HSCT-Curative-Matched-Sibling-Optimal"
        ),
        "alias": (
            "ADA (adenosine deaminase); OMIM gene 608958; "
            "ADA-SCID OMIM 102700. "
            "20q13.12; 363 aa; ~41 kDa; homodimeric cytoplasmic enzyme; autosomal recessive. "
            "FUNCTION: ADA catalyses the irreversible deamination of adenosine and 2'-deoxyadenosine "
            "to inosine and 2'-deoxyinosine respectively, releasing ammonia. "
            "This is the principal route for disposing of adenosine and deoxyadenosine from nucleotide catabolism. "
            "Without ADA, deoxyadenosine accumulates → phosphorylated to dATP by deoxycytidine kinase → "
            "dATP pool expands massively (especially in lymphocytes which cannot export dATP efficiently). "
            "Toxic dATP accumulation: "
            "inhibits ribonucleotide reductase → blocks DNA synthesis; "
            "triggers apoptosis in lymphocytes; "
            "destroys T cells, B cells, AND NK cells — ALL 3 lineages (distinguishes from PNP deficiency). "
            "ADA-SCID is the most common enzyme deficiency causing SCID (~15% of SCID). "
            "CLINICAL PRESENTATION: "
            "Profound lymphopenia (ALC often <200/µL at birth); "
            "absent thymic shadow on chest X-ray; "
            "failure to thrive, recurrent infections within first weeks to months; "
            "opportunistic infections: Pneumocystis jirovecii pneumonia, CMV, EBV, adenovirus, "
            "Candida, Cryptosporidium; "
            "BCG vaccine — CONTRAINDICATED (severe BCG-osis in immunocompromised); "
            "live vaccines all contraindicated; "
            "protective isolation mandatory until immune reconstitution. "
            "EXTRA-IMMUNOLOGICAL FEATURES: "
            "Skeletal dysplasia (chondro-osseous dysplasia): characteristic bone changes on X-ray "
            "(flared costochondral junctions, rib cupping, platyspondyly); "
            "hearing loss (neurosensory) — adenosine important for cochlear development; "
            "neurological: developmental delay, autism spectrum, seizures (some patients); "
            "pulmonary alveolar proteinosis (rare, in older patients on PEG-ADA therapy). "
            "NEWBORN SCREENING: "
            "T-cell receptor excision circles (TREC) assay: detects absent thymic T-cell output; "
            "NBS TREC is the primary screen for all SCID including ADA-SCID; "
            "lymphocyte count: <2,000/µL at birth is suspicious; "
            "ADA enzyme activity in erythrocytes: confirmatory. "
            "TREATMENT: "
            "1. PEG-ADA (ELAPEGADEMASE-LVLR): "
            "Polyethylene glycol-conjugated bovine ADA; "
            "provides exogenous ADA activity, reduces toxic dATP accumulation; "
            "partial immune reconstitution (T cells >B cells); "
            "weekly/biweekly subcutaneous injection; "
            "used as BRIDGE to HSCT or gene therapy; long-term use in patients not eligible for definitive therapy; "
            "does not fully normalise immune function; "
            "2. HSCT: "
            "Curative; matched sibling donor (MSD) preferred — best outcomes; "
            "matched unrelated donor (MUD) acceptable; "
            "haploidentical (T-cell depleted) also performed; "
            "outcomes better when treated early (before infections); "
            "conditioning: reduced-intensity preferred in ADA-SCID; "
            "3. GENE THERAPY — STRIMVELIS (EMA APPROVED 2016): "
            "Ex vivo autologous HSC gene therapy; "
            "patient's own HSCs transduced with retroviral vector carrying ADA cDNA; "
            "available at specialist centres (Milan, Italy; GSK → Orchard Therapeutics); "
            "excellent T-cell reconstitution; B-cell reconstitution variable; "
            "avoids GVHD (autologous product); no need for MSD; "
            "lentiviral vector approaches also in trials (2026). "
        ),
        "locus": "20q13.12",
        "aa": 363,
        "kDa": 41,
        "omim_gene": 608958,
        "omim_disease": 102700,
        "inheritance": "AR",
        "gene_class": "cytoplasmic purine catabolism enzyme — adenosine/deoxyadenosine deaminase",
        "key_alerts": [
            "ADA-DATP-ACCUMULATION-KILLS-T-B-NK-CELLS: In ADA-SCID, toxic dATP accumulation destroys ALL 3 lymphocyte lineages (T, B, and NK cells) — this pan-lymphocyte destruction distinguishes ADA-SCID from PNP deficiency (T-cell selective); absent T, B, and NK cells on lymphocyte subset analysis is the hallmark",
            "ADA-NBS-TREC-DETECTS: ADA-SCID is detectable by TREC-based newborn screening (absent T-cell thymic output) — NBS programmes with TREC detect ADA-SCID before symptomatic infections occur; early diagnosis before infections dramatically improves gene therapy/HSCT outcomes",
            "ADA-STRIMVELIS-EMA2016-GENE-THERAPY: Strimvelis (ex vivo autologous HSC gene therapy, EMA-approved 2016) is the world's first approved gene therapy for ADA-SCID — it uses retroviral ADA-transduced autologous HSCs, avoids GVHD, and achieves excellent T-cell reconstitution; refer to specialist centre for eligibility assessment",
            "ADA-PEG-ADA-BRIDGE-TO-DEFINITIVE: PEG-ADA (elapegademase) is enzyme replacement therapy that partially reduces toxic dATP — it is a BRIDGE to HSCT or gene therapy, not a definitive treatment; do not continue PEG-ADA indefinitely without reassessing eligibility for curative therapy",
            "ADA-HSCT-CURATIVE-MATCHED-SIBLING-OPTIMAL: HSCT is curative for ADA-SCID; matched sibling donor (MSD) gives the best outcomes; begin HLA typing of family members immediately at diagnosis; if no MSD, proceed to gene therapy or matched unrelated donor HSCT at an experienced centre",
        ],
        "etiologies": {
            "classical_neonatal_scid_profound_lymphopenia": 70,
            "delayed_onset_partial_ada_deficiency": 15,
            "late_onset_residual_activity": 10,
            "family_screening_nbs": 5,
        },
        "stats": {
            "disease": "ADA-SCID",
            "incidence": "1:200,000-500,000 live births",
            "fraction_all_scid_pct": 15,
            "alc_at_diagnosis_per_ul": "<200",
            "nbs_trec_detection_pct": 95,
            "strimvelis_t_cell_reconstitution_pct": 90,
            "hsct_msd_survival_pct": 90,
            "peg_ada_partial_reconstitution_pct": 70,
        },
        "dx_delay_distribution": {
            "nbs_diagnosed_presymptomatic": 40,
            "diagnosed_within_3m_symptomatic": 40,
            "diagnosed_3_12m": 15,
            "diagnosed_after_1y": 5,
        },
    },

    # -- PNP -- Purine nucleoside phosphorylase deficiency --------------------
    {
        "gene": "PNP",
        "protein": (
            "PNP -- 14q11.2 AR -- Purine-Nucleoside-Phosphorylase-289aa -- "
            "T-Cell-Selective-Immunodeficiency-Contrast-ADA-SCID -- "
            "dGTP-Accumulation-T-Cell-Toxic -- "
            "Autoimmune-Haemolytic-Anaemia-Thrombocytopenia-SLE-Like -- "
            "Spastic-Diplegia-Neurological-Features -- "
            "HSCT-Curative-Less-Successful-Than-ADA"
        ),
        "alias": (
            "PNP (purine nucleoside phosphorylase); OMIM gene 164050; "
            "PNP deficiency OMIM 613179. "
            "14q11.2; 289 aa; ~32 kDa; homotrimeric cytoplasmic enzyme; autosomal recessive. "
            "FUNCTION: PNP catalyses the phosphorolytic cleavage of purine nucleosides: "
            "inosine + phosphate → hypoxanthine + ribose-1-phosphate; "
            "guanosine + phosphate → guanine + ribose-1-phosphate; "
            "deoxyguanosine + phosphate → guanine + deoxyribose-1-phosphate. "
            "Without PNP, deoxyguanosine (dGuo) accumulates → phosphorylated to dGTP; "
            "dGTP is specifically toxic to T lymphocytes, which have high deoxycytidine kinase activity "
            "and cannot efficiently export dGTP — leading to selective T-cell apoptosis. "
            "B cells and NK cells are relatively spared (they can export dGTP or have lower affinity kinases). "
            "CLINICAL FEATURES: "
            "T-CELL SELECTIVE IMMUNODEFICIENCY: "
            "T cells profoundly reduced (CD4+ and CD8+ both affected); "
            "B cells relatively preserved but functionally impaired (T-cell help absent); "
            "NK cells variable; "
            "onset of infections typically at 6-18 months (after maternal antibody wanes); "
            "less severe early course than ADA-SCID (B cells partially intact → some antibody production); "
            "recurrent bacterial, viral, and opportunistic infections; "
            "PCP, CMV, EBV, Cryptosporidium. "
            "AUTOIMMUNE COMPLICATIONS (unique to PNP, not typical in ADA-SCID): "
            "Autoimmune haemolytic anaemia (AIHA): ~50% of patients; "
            "immune thrombocytopenia (ITP): 25%; "
            "SLE-like disease: 15-20% (antinuclear antibodies, arthritis); "
            "inflammatory bowel disease-like; "
            "autoimmune disease may precede the immunodeficiency diagnosis by years; "
            "paradoxically, immune dysregulation occurs despite T-cell lymphopenia. "
            "NEUROLOGICAL FEATURES: "
            "Spastic diplegia: progressive upper motor neuron signs; "
            "ataxia, tremor; "
            "intellectual disability (mild to moderate in ~50%); "
            "mechanism: dGTP toxicity in CNS neurons/oligodendrocytes; "
            "neurological features may be the presenting complaint before diagnosis. "
            "BIOCHEMISTRY: "
            "Uric acid: very low (<100 µmol/L) — PNP provides hypoxanthine and guanine for XO → uric acid; "
            "when PNP absent, uric acid synthesis reduced (DDx from PNP: low uric acid); "
            "plasma inosine and guanosine elevated; "
            "urine inosine, guanosine, deoxyguanosine elevated; "
            "PNP enzyme activity in erythrocytes: absent. "
            "TREATMENT: "
            "HSCT: curative intent; "
            "outcomes less favourable than ADA-SCID due to: "
            "prior infections, pre-existing neurological damage, and autoimmune complications; "
            "less experience (rarer than ADA-SCID); "
            "gene therapy: not yet approved; preclinical and early-phase trials; "
            "Forodesine (BCX-1777): purine nucleoside phosphorylase inhibitor — paradoxically used "
            "to INHIBIT PNP in T-cell lymphoma; investigational as potential substrate-reduction approach; "
            "IVIG: for hypogammaglobulinaemia; infection prophylaxis. "
        ),
        "locus": "14q11.2",
        "aa": 289,
        "kDa": 32,
        "omim_gene": 164050,
        "omim_disease": 613179,
        "inheritance": "AR",
        "gene_class": "cytoplasmic purine salvage/catabolism enzyme — purine nucleoside phosphorolysis",
        "key_alerts": [
            "PNP-T-CELL-SELECTIVE-CONTRAST-ADA-SCID: PNP deficiency causes T-cell selective immunodeficiency (B cells and NK cells relatively intact) — in contrast to ADA-SCID where all 3 lymphocyte lineages are destroyed; this distinction changes the clinical presentation (less severe early course) and laboratory profile",
            "PNP-AUTOIMMUNE-HAEMOLYTIC-ANAEMIA-50pct: Autoimmune haemolytic anaemia occurs in ~50% of PNP deficiency patients — the autoimmune complications (AIHA, ITP, SLE-like) are paradoxically present despite T-cell lymphopenia and may be the presenting feature before immunodeficiency is diagnosed",
            "PNP-SPASTIC-DIPLEGIA-NEUROLOGICAL: Spastic diplegia and neurological features (ataxia, intellectual disability) are characteristic of PNP deficiency — dGTP toxicity damages CNS neurons; neurological symptoms may present before recurrent infections; do not dismiss neurological features as unrelated",
            "PNP-DGTP-ACCUMULATION-T-CELL-TOXIC: dGTP (from deoxyguanosine phosphorylation) is the toxic metabolite in PNP deficiency — it accumulates specifically in T cells (which cannot export it) → T-cell apoptosis; measuring plasma deoxyguanosine and erythrocyte dGTP supports the diagnosis",
            "PNP-HSCT-LESS-SUCCESSFUL-THAN-ADA: HSCT is curative intent for PNP deficiency but outcomes are less favourable than ADA-SCID — prior infections, neurological damage, and autoimmune complications worsen transplant outcomes; early referral before complications is critical; gene therapy in clinical trials",
        ],
        "etiologies": {
            "t_cell_immunodeficiency_infections": 55,
            "autoimmune_complications_first": 25,
            "neurological_spastic_diplegia": 15,
            "family_screening": 5,
        },
        "stats": {
            "disease": "Purine nucleoside phosphorylase deficiency",
            "incidence": "very rare: ~100 reported cases worldwide",
            "t_cell_lymphopenia_pct": 100,
            "b_cell_preserved_pct": 70,
            "autoimmune_aiha_pct": 50,
            "spastic_diplegia_pct": 60,
            "uric_acid_low_pct": 95,
            "hsct_performed_pct": 60,
        },
        "dx_delay_distribution": {
            "diagnosed_within_1y": 30,
            "diagnosed_1_3y": 40,
            "diagnosed_3_10y": 20,
            "diagnosed_after_10y": 10,
        },
    },

    # -- XDH -- Xanthinuria type I --------------------------------------------
    {
        "gene": "XDH",
        "protein": (
            "XDH -- 2p23.1 AR -- Xanthine-Dehydrogenase-Oxidase-1333aa -- "
            "Xanthinuria-Type-I-XDH-Only -- "
            "Xanthine-Kidney-Stones-Radiolucent-Plain-X-Ray-KEY-DDx-Calcium -- "
            "Uric-Acid-Very-Low-<100-µmol-L-KEY-DDx-Gout -- "
            "Allopurinol-CONTRAINDICATED-XDH-Already-Absent -- "
            "High-Fluid-Intake-Main-Treatment"
        ),
        "alias": (
            "XDH (xanthine dehydrogenase/xanthine oxidase); OMIM gene 607633; "
            "xanthinuria type I OMIM 278300. "
            "2p23.1; 1333 aa; ~147 kDa; molybdo-flavoprotein homodimer; autosomal recessive. "
            "FUNCTION: XDH is a molybdoflavoprotein that catalyses the final two steps of purine catabolism: "
            "hypoxanthine → xanthine (via xanthine dehydrogenase); "
            "xanthine → uric acid (via xanthine oxidase). "
            "XDH can operate in two interconvertible forms: "
            "dehydrogenase form (XDH, physiological, uses NAD+); "
            "oxidase form (XO, generated by oxidative stress, uses O2 and generates superoxide/ROS). "
            "Without XDH, xanthine cannot be converted to uric acid: "
            "xanthine accumulates (relatively insoluble, especially at neutral/alkaline pH); "
            "hypoxanthine also accumulates; "
            "uric acid is very low or absent. "
            "XDH REQUIRES TWO COFACTORS: "
            "FAD (flavin adenine dinucleotide) — at the flavin domain; "
            "Molybdenum cofactor (MoCo) — at the molybdopterin domain; "
            "MoCo deficiency (separate disease, MOCS1/MOCS2/GPHN mutations) causes combined XDH + AO deficiency "
            "= type II xanthinuria (more severe, neurological involvement). "
            "CLINICAL PRESENTATION: "
            "Xanthinuria type I is often ASYMPTOMATIC (incidental finding in up to 2/3); "
            "SYMPTOMATIC presentations: "
            "Urolithiasis: xanthine kidney stones — radiolucent on plain X-ray (unlike calcium stones); "
            "haematuria (macroscopic or microscopic); "
            "renal colic; "
            "xanthine stones do NOT show on plain abdominal film — CT scan or ultrasound needed; "
            "struvite-like deposits if secondary infection; "
            "Myopathy (rare): crystalline xanthine deposits in muscle — myalgia, cramps, elevated CK. "
            "BIOCHEMISTRY — KEY DIAGNOSTIC CLUES: "
            "Uric acid: VERY LOW (<100 µmol/L; normal 180-420 µmol/L); "
            "this is the cardinal diagnostic clue — unexplained very low uric acid; "
            "Plasma xanthine: elevated (normally <15 µmol/L; in XDH deficiency 200-600 µmol/L); "
            "Plasma hypoxanthine: elevated; "
            "Urine xanthine: elevated (xanthine crystalluria, brown/orange urine); "
            "Uric acid/creatinine ratio: markedly reduced. "
            "DIFFERENTIAL FROM GOUT: "
            "Gout: very HIGH uric acid; XDH deficiency: very LOW uric acid — opposite; "
            "A patient prescribed allopurinol by mistake for 'gout' with very low uric acid "
            "should raise XDH deficiency immediately. "
            "ALLOPURINOL — CONTRAINDICATED: "
            "Allopurinol inhibits XDH — which is already absent in xanthinuria type I; "
            "allopurinol has no substrate to act on and is ineffective; "
            "moreover, allopurinol itself is partially metabolised by XDH to oxypurinol — "
            "this conversion cannot occur, allopurinol accumulates; "
            "allopurinol is therefore both ineffective AND potentially toxic in XDH deficiency; "
            "it is explicitly contraindicated. "
            "TYPE I vs TYPE II XANTHINURIA: "
            "Type I (XDH only deficient): classical presentation as above; no neurological involvement; "
            "Type II (XDH + aldehyde oxidase [AO] both deficient): due to MoCo deficiency "
            "(MOCS1/MOCS2/GPHN genes) — much more severe, neurological involvement, seizures, "
            "intellectual disability; differentiated by sulphite test (MoCo deficiency shows positive). "
            "TREATMENT: "
            "High fluid intake (2-3 L/day): main treatment to prevent xanthine stone formation; "
            "alkalinisation of urine (sodium bicarbonate or potassium citrate): modest solubility benefit; "
            "low purine diet: reduces xanthine substrate; "
            "NO allopurinol; NO uricosuric agents (useless with undetectable uric acid). "
        ),
        "locus": "2p23.1",
        "aa": 1333,
        "kDa": 147,
        "omim_gene": 607633,
        "omim_disease": 278300,
        "inheritance": "AR",
        "gene_class": "molybdo-flavoprotein purine catabolism — xanthine oxidoreductase, final uric acid synthesis steps",
        "key_alerts": [
            "XDH-RADIOLUCENT-XANTHINE-STONES: Xanthine kidney stones are RADIOLUCENT on plain X-ray (unlike calcium oxalate or calcium phosphate stones which are radio-opaque) — plain abdominal film will NOT show xanthine stones; use renal ultrasound or CT KUB for diagnosis; suspect xanthinuria when 'no stone seen' on X-ray despite renal colic",
            "XDH-URIC-ACID-VERY-LOW-KEY-DDX-GOUT: Very low plasma uric acid (<100 µmol/L) is the cardinal biochemical clue for XDH deficiency — this is the OPPOSITE of gout (high uric acid); unexplained very low uric acid on a routine biochemistry panel should trigger xanthinuria workup with plasma/urine xanthine levels",
            "XDH-ALLOPURINOL-CONTRAINDICATED-XDH-ABSENT: Allopurinol is CONTRAINDICATED in XDH deficiency (type I xanthinuria) — it inhibits XDH which is already absent; allopurinol is ineffective (no substrate) and potentially toxic as it cannot be metabolised to oxypurinol; high fluid intake and low purine diet are the correct management",
            "XDH-HIGH-FLUID-INTAKE-MAIN-TREATMENT: The primary treatment for symptomatic xanthinuria is high fluid intake (2-3 L/day) to maintain dilute urine, preventing xanthine crystallisation and stone formation; alkalinisation (potassium citrate) provides modest additional benefit; no specific drug therapy",
            "XDH-TYPE-I-VS-TYPE-II-CRITICAL-DISTINCTION: Type I xanthinuria (XDH only) is benign in most patients; Type II (XDH + aldehyde oxidase deficiency from MoCo deficiency, MOCS1/MOCS2 genes) is much more severe with neurological involvement and seizures; distinguish using sulphite urine test (positive in MoCo deficiency) and AO activity assay",
        ],
        "etiologies": {
            "asymptomatic_incidental": 65,
            "xanthine_urolithiasis_renal_colic": 25,
            "haematuria_workup": 8,
            "myopathy_xanthine_deposits": 2,
        },
        "stats": {
            "disease": "Xanthinuria type I",
            "incidence": "1:69,000 (estimated; underdiagnosed)",
            "asymptomatic_pct": 65,
            "uric_acid_umol_L_typical": "<100",
            "plasma_xanthine_umol_L_typical": "200-600",
            "radiolucent_stone_plain_xray": 100,
            "allopurinol_prescribing_error_pct": 15,
            "type_ii_moco_deficiency_pct": 20,
        },
        "dx_delay_distribution": {
            "diagnosed_incidentally": 30,
            "diagnosed_first_stone_episode": 40,
            "diagnosed_after_recurrent_stones": 20,
            "diagnosed_after_10y_symptoms": 10,
        },
    },

    # -- APRT -- APRT deficiency / 2,8-DHA urolithiasis -----------------------
    {
        "gene": "APRT",
        "protein": (
            "APRT -- 16q24.3 AR -- Adenine-Phosphoribosyltransferase-180aa -- "
            "2-8-DHA-Crystals-Brownish-Urine-PATHOGNOMONIC -- "
            "Progressive-Renal-Failure-If-Untreated -- "
            "Allopurinol-CURATIVE-Blocks-XO-Upstream -- "
            "Low-Adenine-Diet-Adjunct -- "
            "Japanese-Founder-Effect-High-Prevalence"
        ),
        "alias": (
            "APRT (adenine phosphoribosyltransferase); OMIM gene 102600; "
            "APRT deficiency / 2,8-dihydroxyadenine urolithiasis OMIM 614723. "
            "16q24.3; 180 aa; ~20 kDa; homodimeric cytoplasmic enzyme; autosomal recessive. "
            "FUNCTION: APRT catalyses the purine salvage of adenine: "
            "adenine + PRPP → AMP + PPi. "
            "Without APRT, adenine cannot be salvaged and is instead catabolised by xanthine oxidase (XO): "
            "adenine → 8-hydroxyadenine → 2,8-dihydroxyadenine (2,8-DHA) — via XO. "
            "2,8-DHA is extremely insoluble in urine (solubility ~3 mg/L, compared to uric acid 65 mg/L): "
            "2,8-DHA precipitates in renal tubules and collecting system → "
            "tubular obstruction → interstitial nephritis → progressive renal failure. "
            "KEY BIOCHEMISTRY: "
            "Source of adenine: dietary (meat, yeast) → absorbed in gut; "
            "without APRT, dietary adenine → 2,8-DHA → excreted in urine → crystals; "
            "key observation: reducing dietary adenine (low purine diet) reduces 2,8-DHA burden. "
            "CLINICAL PRESENTATION: "
            "Urolithiasis: 2,8-DHA kidney stones and crystalluria; "
            "urine: brownish/dark discolouration; brown deposits in nappy (infants); "
            "radiolucent stones (2,8-DHA not radio-opaque); "
            "microscopy: round, birefringent, brown crystals (Maltese cross pattern under polarised light); "
            "renal failure: progressive if untreated — from infancy to adulthood; "
            "tubulo-interstitial nephropathy: can present as CKD without obvious stones; "
            "end-stage renal disease (ESRD) reported in untreated adults; "
            "recurrent haematuria; "
            "renal transplant: successful for ESRD but 2,8-DHA nephropathy recurs in graft if not treated with allopurinol. "
            "JAPANESE FOUNDER EFFECT: "
            "Prevalence in Japan: ~1:250,000 (10× higher than other populations); "
            "Japanese founder variant: p.Met136Thr (Type II APRT deficiency — Japan-specific); "
            "Type I APRT deficiency: null variants (all populations); "
            "Type II: Japanese founder, 15% residual activity; "
            "in Japan, APRT deficiency is relatively common → established screening programme. "
            "DIAGNOSIS: "
            "Urine microscopy: brown round 2,8-DHA crystals (birefringent, Maltese cross) — PATHOGNOMONIC; "
            "urinary adenine and 2,8-DHA by HPLC; "
            "APRT enzyme activity in erythrocytes (gold standard); "
            "gene sequencing: APRT on chromosome 16q24.3; "
            "plasma 2,8-DHA measurable by LC-MS/MS. "
            "TREATMENT: "
            "ALLOPURINOL — CURATIVE: "
            "Allopurinol inhibits XO (xanthine oxidase) — blocks the conversion of adenine → 2,8-DHA; "
            "adenine diverted to alternative pathways or excreted unchanged; "
            "dramatically reduces 2,8-DHA production and urinary crystallisation; "
            "prevents progressive renal damage and dissolves soft 2,8-DHA deposits; "
            "dose: 100-300 mg/day (standard allopurinol dosing); "
            "CONTRAST WITH XDH DEFICIENCY: in XDH, allopurinol is CI; in APRT, allopurinol is CURATIVE; "
            "DIET: low adenine (low purine) diet as adjunct to allopurinol; "
            "HIGH FLUID INTAKE: dilute urine reduces crystallisation; "
            "FEBUXOSTAT: alternative XO inhibitor if allopurinol intolerant. "
        ),
        "locus": "16q24.3",
        "aa": 180,
        "kDa": 20,
        "omim_gene": 102600,
        "omim_disease": 614723,
        "inheritance": "AR",
        "gene_class": "cytoplasmic purine salvage enzyme — adenine → AMP phosphoribosyltransferase",
        "key_alerts": [
            "APRT-2-8-DHA-CRYSTALS-BROWNISH-PATHOGNOMONIC: 2,8-dihydroxyadenine (2,8-DHA) crystals in urine are PATHOGNOMONIC for APRT deficiency — they appear brownish, round, and birefringent (Maltese cross pattern under polarised light); brown nappy deposits in infants or brownish urine should trigger immediate APRT workup; crystals can be mistaken for uric acid on routine microscopy",
            "APRT-RENAL-FAILURE-IF-UNTREATED: 2,8-DHA deposits progressively damage renal tubules → interstitial nephritis → CKD → ESRD if untreated; APRT deficiency can present as unexplained CKD in adults without obvious stones; any unexplained renal failure with crystalluria should prompt APRT enzyme assay",
            "APRT-ALLOPURINOL-CURATIVE: Allopurinol is CURATIVE for APRT deficiency — it blocks xanthine oxidase, preventing adenine → 2,8-DHA conversion; this is the opposite of XDH deficiency (where allopurinol is contraindicated); start allopurinol immediately at diagnosis and continue lifelong; it prevents further renal damage and may partially reverse early nephropathy",
            "APRT-JAPANESE-FOUNDER-EFFECT: APRT deficiency is ~10× more prevalent in Japan (~1:250,000) due to the Type II founder variant p.Met136Thr; Japan has established APRT screening programmes; outside Japan (Type I), APRT deficiency is underdiagnosed — include it in the differential of any unexplained 2,8-DHA crystalluria or urolithiasis globally",
            "APRT-LOW-ADENINE-DIET-ADJUNCT: A low adenine (low purine) diet reduces the substrate load reaching XO, complementing allopurinol therapy; dietary adenine comes primarily from meat and yeast extracts; dietary restriction alone is insufficient — allopurinol is always required; post-renal-transplant patients must continue allopurinol to prevent graft recurrence",
        ],
        "etiologies": {
            "urolithiasis_2_8_dha_stones": 50,
            "ckd_tubular_nephropathy_no_obvious_stones": 25,
            "haematuria_crystalluria_workup": 15,
            "family_screening_japan": 10,
        },
        "stats": {
            "disease": "APRT deficiency / 2,8-DHA urolithiasis",
            "incidence": "~1:50,000 to 1:250,000 (higher in Japan)",
            "japan_prevalence": "~1:250,000",
            "dha_crystals_pathognomonic_pct": 100,
            "renal_failure_untreated_pct": 60,
            "allopurinol_response_pct": 97,
            "type_ii_japan_pct": 80,
        },
        "dx_delay_distribution": {
            "diagnosed_within_1y_symptomatic": 20,
            "diagnosed_1_5y": 30,
            "diagnosed_5_20y": 30,
            "diagnosed_after_20y_ckd_only": 20,
        },
    },

    # -- AMPD1 -- Myoadenylate deaminase deficiency ---------------------------
    {
        "gene": "AMPD1",
        "protein": (
            "AMPD1 -- 1p13.3 AR -- AMP-Deaminase-1-Muscle-Isoform-747aa -- "
            "Exercise-Intolerance-Cramps-Clinical-Significance-Debated -- "
            "Forearm-Exercise-Test-Ammonia-ABSENT-PATHOGNOMONIC-Lactate-Rises-Normally -- "
            "Opposite-Of-McArdle-PYGM-Where-Lactate-Flat -- "
            "Common-Polymorphism-p-Gln12Ter-2pct-European-Homozygous -- "
            "Distinguish-Primary-vs-Secondary-Forms"
        ),
        "alias": (
            "AMPD1 (AMP deaminase 1, muscle isoform); OMIM gene 102770; "
            "myoadenylate deaminase deficiency OMIM 615511. "
            "1p13.3; 747 aa; ~87 kDa; skeletal muscle-specific isoform; autosomal recessive. "
            "FUNCTION: AMPD1 catalyses the deamination of AMP to IMP + NH3 in skeletal muscle: "
            "AMP + H2O → IMP + NH3. "
            "This reaction is part of the purine nucleotide cycle (PNC) in muscle, which operates during exercise: "
            "AMP → IMP (AMPD1) → adenylosuccinate (ADSS) → AMP (ADSL); "
            "the PNC regenerates AMP from aspartate, maintaining adenine nucleotide levels during intense exercise; "
            "a byproduct is ammonia (NH3) — detectable in forearm exercise testing. "
            "THE FOREARM EXERCISE TEST IN AMPD1 DEFICIENCY: "
            "Normal: during exercise, AMP → IMP via AMPD1 → NH3 released into blood; "
            "ammonia rises 2-3× baseline after forearm exercise; "
            "In AMPD1 deficiency: AMP cannot be deaminated; "
            "ammonia does NOT rise after exercise — PATHOGNOMONIC; "
            "lactate rises NORMALLY (glycolysis is intact — CONTRAST with McArdle/PYGM where lactate flat); "
            "therefore: AMPD1 = lactate rises, ammonia absent; McArdle/PYGM = lactate flat, ammonia rises. "
            "THE COMMON POLYMORPHISM CONTROVERSY: "
            "p.Gln12Ter (c.34C>T): nonsense variant causing AMPD1 deficiency; "
            "allele frequency: ~2% in European populations; "
            "homozygous frequency: ~1:2,500 in Europeans; "
            "CLINICAL SIGNIFICANCE DEBATED: many homozygous individuals are COMPLETELY ASYMPTOMATIC; "
            "some investigators consider AMPD1 deficiency a benign polymorphism rather than a true disease; "
            "primary AMPD1 deficiency: symptoms (exercise intolerance, cramps) with AMPD1 null variants; "
            "secondary AMPD1 deficiency: reduced AMPD1 activity in the context of another muscle disease "
            "(limb-girdle muscular dystrophy, polymyositis, McArdle, mitochondrial myopathy); "
            "secondary form is not a separate diagnosis — AMPD1 activity falls in diseased muscle generally; "
            "careful distinction between primary and secondary is essential before attributing symptoms to AMPD1. "
            "CLINICAL FEATURES OF PRIMARY AMPD1 DEFICIENCY: "
            "Exercise-induced muscle cramps and myalgia; "
            "post-exercise fatigue disproportionate to exertion; "
            "resting CK may be normal or mildly elevated; "
            "no myoglobinuria (unlike McArdle); "
            "no fixed weakness; "
            "symptoms typically begin in adulthood; "
            "many patients have alternative explanations (deconditioned, comorbid diagnosis). "
            "DIAGNOSIS: "
            "Forearm exercise test: lactate rises, ammonia absent (pathognomonic); "
            "muscle biopsy: absent AMPD1 histochemical staining (menadione-NBT method); "
            "gene sequencing: p.Gln12Ter most common; "
            "exclude secondary causes: full neuromuscular workup. "
            "TREATMENT: "
            "No specific treatment; "
            "aerobic exercise training improves exercise tolerance; "
            "ribose supplementation: attempted (poor evidence); "
            "manage expectations: many patients need reassurance that symptoms are benign. "
        ),
        "locus": "1p13.3",
        "aa": 747,
        "kDa": 87,
        "omim_gene": 102770,
        "omim_disease": 615511,
        "inheritance": "AR",
        "gene_class": "skeletal muscle-specific purine nucleotide cycle enzyme — AMP → IMP deaminase",
        "key_alerts": [
            "AMPD1-AMMONIA-ABSENT-FOREARM-TEST-PATHOGNOMONIC: In AMPD1 deficiency, the forearm exercise test shows ammonia does NOT rise (AMP deaminase absent → no NH3 production) while lactate RISES normally (glycolysis intact) — this combination is PATHOGNOMONIC for AMPD1 deficiency; contrast with McArdle (PYGM) where lactate is FLAT and ammonia RISES",
            "AMPD1-LACTATE-RISES-CONTRAST-McARDLE: AMPD1 deficiency and McArdle disease (PYGM) both cause exercise intolerance — but the forearm exercise test result is OPPOSITE: AMPD1 has normal lactate rise + absent ammonia; McArdle has absent lactate rise + normal ammonia; this distinction directs correct genetic testing",
            "AMPD1-COMMON-POLYMORPHISM-2pct-EUROPEAN: The p.Gln12Ter variant is found in ~2% of European alleles, making homozygosity (~1:2,500) the most common cause of AMPD1 deficiency — but many homozygous individuals are ASYMPTOMATIC; this questions whether AMPD1 deficiency is a true disease entity or a benign polymorphism; do not overdiagnose",
            "AMPD1-CLINICAL-SIGNIFICANCE-DEBATED: Primary AMPD1 deficiency has debated clinical significance — the exercise intolerance and cramps are real in symptomatic patients, but many genetically confirmed homozygous individuals have no symptoms; before attributing symptoms to AMPD1 deficiency, exclude all other causes of exercise intolerance",
            "AMPD1-DISTINGUISH-PRIMARY-VS-SECONDARY: Secondary AMPD1 deficiency (reduced enzyme activity in the context of another muscle disease — LGMD, polymyositis, McArdle, mitochondrial myopathy) is more common than primary; treating the underlying condition reverses AMPD1 activity; do not ascribe the diagnosis to AMPD1 when another disease is present",
        ],
        "etiologies": {
            "primary_exercise_intolerance_cramps": 45,
            "secondary_ampd1_underlying_muscle_disease": 35,
            "asymptomatic_homozygous_incidental": 15,
            "family_screening": 5,
        },
        "stats": {
            "disease": "Myoadenylate deaminase deficiency",
            "incidence": "~1:2,500 homozygous (Europeans, p.Gln12Ter)",
            "p_gln12ter_allele_freq_european_pct": 2,
            "asymptomatic_homozygous_pct": 50,
            "ammonia_absent_forearm_test_pct": 100,
            "lactate_rises_normally_pct": 99,
            "secondary_ampd1_pct": 35,
        },
        "dx_delay_distribution": {
            "diagnosed_within_1y_symptoms": 15,
            "diagnosed_1_5y": 35,
            "diagnosed_5_20y": 35,
            "diagnosed_after_20y": 15,
        },
    },

    # -- UMPS -- Hereditary orotic aciduria type I ----------------------------
    {
        "gene": "UMPS",
        "protein": (
            "UMPS -- 3q13.33 AR -- UMP-Synthase-Bifunctional-480aa -- "
            "Hereditary-Orotic-Aciduria-Type-I -- "
            "Megaloblastic-Anaemia-Orotic-Acid-Crystalluria-Growth-Retardation -- "
            "NOT-B12-Folate-Responsive-KEY-DDx -- "
            "Uridine-Replacement-CURATIVE -- "
            "DDx-OTC-Deficiency-Ammonia-Normal-In-UMPS -- "
            "Bifunctional-OPRT-Plus-OMP-Decarboxylase"
        ),
        "alias": (
            "UMPS (UMP synthase, bifunctional); OMIM gene 613891; "
            "hereditary orotic aciduria type I OMIM 258900. "
            "3q13.33; 480 aa; ~52 kDa; bifunctional enzyme; autosomal recessive. "
            "FUNCTION: UMPS is a bifunctional enzyme carrying two sequential catalytic activities "
            "in the de novo pyrimidine synthesis pathway: "
            "1. Orotate phosphoribosyltransferase (OPRT): orotate + PRPP → orotidine-5'-monophosphate (OMP); "
            "2. OMP decarboxylase (OMPDC): OMP → UMP + CO2. "
            "UMP is the precursor for all pyrimidines (UTP, CTP, dTTP) required for DNA and RNA synthesis. "
            "Without UMPS, orotate accumulates (upstream of both blocked steps) → "
            "orotic acid (crystalline form of orotate) excreted massively in urine. "
            "Without UMP, pyrimidines cannot be synthesised de novo → pyrimidine starvation → "
            "failure of DNA replication → megaloblastic haematopoiesis. "
            "CLINICAL PRESENTATION: "
            "MEGALOBLASTIC ANAEMIA: "
            "Presents in infancy — pallor, fatigue, failure to thrive; "
            "macrocytic anaemia on blood count; "
            "megaloblasts on bone marrow biopsy; "
            "KEY POINT: NOT responsive to vitamin B12 or folate (unlike nutritional/B12-folate-deficient megaloblastic anaemia); "
            "B12/folate levels are normal; "
            "OROTIC ACID CRYSTALLURIA: "
            "Dense orotic acid crystals in urine — white/yellow precipitate; "
            "can cause obstructive uropathy in infants; "
            "urine: white/yellowish sediment, turbid appearance; "
            "GROWTH RETARDATION: "
            "Failure to thrive; growth delay; "
            "developmental delay in untreated or late-diagnosed cases. "
            "DIFFERENTIAL DIAGNOSIS — CRITICAL: "
            "OTC deficiency (ornithine transcarbamylase deficiency): "
            "ALSO causes elevated orotic acid in urine (OTC block → carbamoyl phosphate → pyrimidine pathway overflow); "
            "KEY DISTINGUISHING FEATURE: OTC deficiency causes HYPERAMMONAEMIA; UMPS does NOT; "
            "blood ammonia is NORMAL in UMPS orotic aciduria; elevated in OTC deficiency; "
            "this single test — blood ammonia — separates the two diagnoses; "
            "gender: OTC is X-linked (males predominantly affected); UMPS is AR. "
            "OTHER DDx: "
            "B12 or folate deficiency: megaloblastic anaemia but orotic acid NOT elevated; B12/folate low; "
            "Diamond-Blackfan anaemia: macrocytic but no orotic aciduria; "
            "mitochondrial disorders: some have orotic aciduria but other features. "
            "TREATMENT — URIDINE REPLACEMENT: "
            "Uridine (oral) bypasses the UMPS block: exogenous uridine → UMP (via uridine kinase, separate enzyme); "
            "uridine triacetate (Xuriden® / Vistogard®): lipophilic prodrug, better absorption; "
            "result: pyrimidine synthesis restored → megaloblastic anaemia corrects rapidly; "
            "orotic acid levels fall (PRPP no longer diverted to orotate pathway once UMP available); "
            "dramatic clinical response: anaemia corrects within weeks; growth resumes; "
            "treatment is lifelong; "
            "dose: 100-200 mg/kg/day uridine (or equivalent uridine triacetate dose). "
            "PROGNOSIS: "
            "Excellent with uridine replacement — normal growth, normal neurodevelopment if treated early; "
            "untreated: severe anaemia, growth failure, cognitive impairment. "
        ),
        "locus": "3q13.33",
        "aa": 480,
        "kDa": 52,
        "omim_gene": 613891,
        "omim_disease": 258900,
        "inheritance": "AR",
        "gene_class": "bifunctional pyrimidine de novo synthesis enzyme — OPRT + OMP decarboxylase (UMP synthase)",
        "key_alerts": [
            "UMPS-URIDINE-REPLACEMENT-CURATIVE: Oral uridine (or uridine triacetate/Xuriden) is CURATIVE for hereditary orotic aciduria — it bypasses the UMPS block by providing exogenous pyrimidines; megaloblastic anaemia corrects within weeks; orotic acid excretion falls; start immediately at diagnosis and continue lifelong",
            "UMPS-NOT-B12-FOLATE-RESPONSIVE-KEY-DDX: Megaloblastic anaemia in UMPS orotic aciduria does NOT respond to vitamin B12 or folate — this is a critical diagnostic clue; B12 and folate levels are NORMAL; any megaloblastic anaemia not responding to B12/folate therapy should prompt urine orotic acid measurement",
            "UMPS-OROTIC-ACID-URINE-BIOMARKER: Orotic acid in urine is the key biomarker — easily measured by urine organic acids (orotic acid peak on OA chromatogram) or dedicated urine purine/pyrimidine quantitation; white/yellowish crystalline urine sediment in an infant with anaemia is a direct clinical sign",
            "UMPS-DDX-OTC-AMMONIA-NORMAL: Both UMPS orotic aciduria and OTC deficiency (X-linked) produce elevated urinary orotic acid — the critical distinguishing test is blood ammonia: NORMAL in UMPS orotic aciduria, ELEVATED (sometimes >200 µmol/L) in OTC deficiency; ammonia is the single most important differentiating test",
            "UMPS-BIFUNCTIONAL-ENZYME-OPRT-PLUS-OMP-DECARBOXYLASE: UMPS encodes a bifunctional protein with two sequential enzyme activities (OPRT and OMPDC) — both are deficient; orotic acid accumulates before both steps; PRPP is consumed by orotate but cannot progress to UMP; understanding this explains why orotic acid rises AND anaemia occurs simultaneously",
        ],
        "etiologies": {
            "megaloblastic_anaemia_infancy": 60,
            "failure_to_thrive_growth_retardation": 20,
            "crystalluria_obstructive_uropathy": 15,
            "family_screening": 5,
        },
        "stats": {
            "disease": "Hereditary orotic aciduria type I",
            "incidence": "very rare: ~25 reported cases worldwide",
            "b12_folate_responsive_pct": 0,
            "uridine_response_anaemia_pct": 99,
            "ammonia_elevated_pct": 0,
            "orotic_acid_urine_elevated_pct": 100,
            "otc_misdiagnosis_risk_pct": 25,
        },
        "dx_delay_distribution": {
            "diagnosed_within_6m": 30,
            "diagnosed_6_24m": 40,
            "diagnosed_2_10y": 20,
            "diagnosed_after_10y": 10,
        },
    },
]


def _generate_patients():
    """Generate 40 synthetic patients per gene (8 × 40 = 320 total)."""
    for idx, gene_data in enumerate(PURINE_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        patients = []
        gene = gene_data["gene"]

        for i in range(40):
            pid = f"{gene}-{seed}-P{i+1:02d}"
            age = rng.randint(0, 55)

            if gene == "HPRT1":
                complete_deficiency = rng.random() < 0.55
                sib_self_injury = complete_deficiency and rng.random() < 0.97
                choreoathetosis = complete_deficiency and rng.random() < 0.90
                dystonia = complete_deficiency and rng.random() < 0.85
                intellectual_disability = complete_deficiency and rng.random() < 0.95
                uric_acid = rng.randint(600, 950) if complete_deficiency else rng.randint(450, 750)
                orange_nappy_grit = rng.random() < (0.60 if age < 3 else 0.05)
                allopurinol = rng.random() < 0.90
                allopurinol_neuro_improvement = False  # never improves neurology
                gout = rng.random() < (0.85 if not complete_deficiency else 0.50)
                renal_stones = rng.random() < 0.45
                dx_delay_months = rng.randint(0, 48)
                restraints_used = sib_self_injury and rng.random() < 0.80
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "complete_deficiency_lesch_nyhan": complete_deficiency,
                    "self_injurious_behaviour": sib_self_injury,
                    "choreoathetosis": choreoathetosis,
                    "dystonia": dystonia,
                    "intellectual_disability": intellectual_disability,
                    "uric_acid_umol_L": uric_acid,
                    "orange_nappy_grit": orange_nappy_grit,
                    "on_allopurinol": allopurinol,
                    "allopurinol_neuro_improvement": allopurinol_neuro_improvement,
                    "gout": gout,
                    "renal_stones": renal_stones,
                    "restraints_used": restraints_used,
                    "dx_delay_months": dx_delay_months,
                    "outcome": "stable_uric_acid_uncontrolled_neuro" if complete_deficiency else "stable",
                })

            elif gene == "ADSL":
                phenotype = rng.choices(
                    ["severe", "mild", "neonatal_lethal"],
                    weights=[60, 25, 10]
                )[0]
                if phenotype == "neonatal_lethal":
                    phenotype = "neonatal_lethal"
                saicar_elevated = True  # always
                seizures = rng.random() < (0.90 if phenotype == "severe" else 0.30 if phenotype == "mild" else 1.0)
                autistic_features = rng.random() < (0.75 if phenotype in ("severe", "mild") else 0.50)
                psychomotor_retardation = rng.random() < (0.95 if phenotype == "severe" else 0.50 if phenotype == "mild" else 1.0)
                csf_sado_measured = rng.random() < 0.55
                allopurinol_tried = rng.random() < 0.50
                dx_delay_months = rng.randint(3, 120)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "phenotype": phenotype,
                    "saicar_sado_elevated": saicar_elevated,
                    "seizures": seizures,
                    "autistic_features": autistic_features,
                    "psychomotor_retardation": psychomotor_retardation,
                    "csf_sado_measured": csf_sado_measured,
                    "allopurinol_tried": allopurinol_tried,
                    "dx_delay_months": dx_delay_months,
                    "outcome": "deceased" if phenotype == "neonatal_lethal" else ("severe_disability" if phenotype == "severe" else "moderate_disability"),
                })

            elif gene == "ADA":
                nbs_detected = rng.random() < 0.40
                alc_at_dx = rng.randint(20, 180) if not nbs_detected else rng.randint(80, 300)
                dATP_elevated = True
                t_cell_absent = rng.random() < 0.95
                b_cell_absent = rng.random() < 0.90
                nk_cell_absent = rng.random() < 0.85
                pcp_infection = not nbs_detected and rng.random() < 0.45
                on_peg_ada = rng.random() < 0.60
                hsct_performed = rng.random() < 0.50
                strimvelis_gt = not hsct_performed and rng.random() < 0.30
                skeletal_dysplasia = rng.random() < 0.50
                hearing_loss = rng.random() < 0.35
                dx_delay_months = 0 if nbs_detected else rng.randint(1, 18)
                outcome = "immune_reconstituted" if (hsct_performed or strimvelis_gt) else ("partial_reconstitution" if on_peg_ada else "at_risk")
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "nbs_trec_detected": nbs_detected,
                    "alc_at_diagnosis": alc_at_dx,
                    "datp_accumulation": dATP_elevated,
                    "t_cell_absent": t_cell_absent,
                    "b_cell_absent": b_cell_absent,
                    "nk_cell_absent": nk_cell_absent,
                    "pcp_pneumonia": pcp_infection,
                    "on_peg_ada": on_peg_ada,
                    "hsct_performed": hsct_performed,
                    "strimvelis_gene_therapy": strimvelis_gt,
                    "skeletal_dysplasia": skeletal_dysplasia,
                    "hearing_loss": hearing_loss,
                    "dx_delay_months": dx_delay_months,
                    "outcome": outcome,
                })

            elif gene == "PNP":
                t_cell_lymphopenia = rng.random() < 1.0
                t_cell_count = rng.randint(5, 150)
                b_cell_preserved = rng.random() < 0.70
                autoimmune_aiha = rng.random() < 0.50
                itp = rng.random() < 0.25
                sle_like = rng.random() < 0.18
                spastic_diplegia = rng.random() < 0.60
                intellectual_disability = rng.random() < 0.50
                uric_acid_low = rng.randint(20, 90)
                deoxyguanosine_elevated = True
                hsct_performed = rng.random() < 0.60
                infections = rng.random() < (0.70 if not hsct_performed else 0.20)
                dx_delay_months = rng.randint(3, 60)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "t_cell_lymphopenia": t_cell_lymphopenia,
                    "t_cell_count_per_ul": t_cell_count,
                    "b_cell_relatively_preserved": b_cell_preserved,
                    "autoimmune_haemolytic_anaemia": autoimmune_aiha,
                    "immune_thrombocytopenia": itp,
                    "sle_like_disease": sle_like,
                    "spastic_diplegia": spastic_diplegia,
                    "intellectual_disability": intellectual_disability,
                    "uric_acid_umol_L": uric_acid_low,
                    "deoxyguanosine_elevated": deoxyguanosine_elevated,
                    "hsct_performed": hsct_performed,
                    "recurrent_infections": infections,
                    "dx_delay_months": dx_delay_months,
                    "outcome": "partial_immune_reconstitution" if hsct_performed else "progressive_immunodeficiency",
                })

            elif gene == "XDH":
                asymptomatic = rng.random() < 0.65
                xanthine_stones = not asymptomatic and rng.random() < 0.90
                radiolucent_on_xray = xanthine_stones  # always radiolucent
                uric_acid_low = rng.randint(10, 90)
                plasma_xanthine_high = rng.randint(200, 580)
                renal_colic = xanthine_stones and rng.random() < 0.70
                haematuria = rng.random() < (0.40 if xanthine_stones else 0.05)
                myopathy = rng.random() < 0.05
                allopurinol_error = rng.random() < 0.15  # prescribing error
                type_ii_moco = rng.random() < 0.20
                high_fluid_intake = rng.random() < 0.75
                dx_delay_months = rng.randint(0, 120)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "asymptomatic": asymptomatic,
                    "xanthine_kidney_stones": xanthine_stones,
                    "radiolucent_on_plain_xray": radiolucent_on_xray,
                    "uric_acid_umol_L": uric_acid_low,
                    "plasma_xanthine_umol_L": plasma_xanthine_high,
                    "renal_colic": renal_colic,
                    "haematuria": haematuria,
                    "myopathy_xanthine_deposits": myopathy,
                    "allopurinol_prescribed_error": allopurinol_error,
                    "type_ii_moco_deficiency": type_ii_moco,
                    "high_fluid_intake_treatment": high_fluid_intake,
                    "dx_delay_months": dx_delay_months,
                    "outcome": "stable" if not type_ii_moco else rng.choice(["neurological_disability", "stable"]),
                })

            elif gene == "APRT":
                dha_crystals = rng.random() < 0.90
                urolithiasis = rng.random() < 0.70
                ckd_stage = rng.choices([0, 1, 2, 3, 4, 5], weights=[20, 20, 25, 20, 10, 5])[0]
                renal_failure = ckd_stage >= 3
                allopurinol_prescribed = rng.random() < 0.80
                allopurinol_started_late = allopurinol_prescribed and rng.random() < 0.40
                japanese_founder = rng.random() < 0.30
                low_adenine_diet = rng.random() < 0.60
                transplant = ckd_stage == 5 and rng.random() < 0.70
                dx_delay_years = rng.randint(0, 30)
                haematuria = rng.random() < 0.55
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "dha_crystals_urine": dha_crystals,
                    "urolithiasis": urolithiasis,
                    "ckd_stage": ckd_stage,
                    "renal_failure": renal_failure,
                    "on_allopurinol": allopurinol_prescribed,
                    "allopurinol_started_late": allopurinol_started_late,
                    "japanese_founder_type_ii": japanese_founder,
                    "low_adenine_diet": low_adenine_diet,
                    "renal_transplant": transplant,
                    "haematuria": haematuria,
                    "dx_delay_years": dx_delay_years,
                    "outcome": "stable" if allopurinol_prescribed else ("esrd" if renal_failure else "progressive_nephropathy"),
                })

            elif gene == "AMPD1":
                primary_ampd1 = rng.random() < 0.45
                secondary_ampd1 = not primary_ampd1 and rng.random() < 0.75
                asymptomatic_homozygous = not primary_ampd1 and not secondary_ampd1
                exercise_intolerance = primary_ampd1 and rng.random() < 0.90
                muscle_cramps = primary_ampd1 and rng.random() < 0.85
                myoglobinuria = False  # not typical in AMPD1
                forearm_test_ammonia_absent = rng.random() < 1.0  # always absent in AMPD1
                forearm_test_lactate_rises = rng.random() < 0.99  # always rises (glycolysis intact)
                p_gln12ter_homozygous = rng.random() < 0.70
                ck_mildly_elevated = primary_ampd1 and rng.random() < 0.40
                alternative_dx = secondary_ampd1 and rng.random() < 0.80
                dx_delay_years = rng.randint(0, 20)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "primary_ampd1_deficiency": primary_ampd1,
                    "secondary_ampd1_deficiency": secondary_ampd1,
                    "asymptomatic_homozygous": asymptomatic_homozygous,
                    "exercise_intolerance": exercise_intolerance,
                    "muscle_cramps_post_exercise": muscle_cramps,
                    "myoglobinuria": myoglobinuria,
                    "forearm_test_ammonia_absent": forearm_test_ammonia_absent,
                    "forearm_test_lactate_rises_normally": forearm_test_lactate_rises,
                    "p_gln12ter_homozygous": p_gln12ter_homozygous,
                    "ck_mildly_elevated": ck_mildly_elevated,
                    "alternative_diagnosis_found": alternative_dx,
                    "dx_delay_years": dx_delay_years,
                    "outcome": "benign_exercise_limitation" if primary_ampd1 else ("underlying_disease" if secondary_ampd1 else "asymptomatic"),
                })

            elif gene == "UMPS":
                megaloblastic_anaemia = rng.random() < 0.95
                orotic_acid_crystalluria = rng.random() < 0.90
                growth_retardation = rng.random() < 0.85
                b12_folate_tried_error = rng.random() < 0.60  # common initial error
                b12_folate_responded = False  # never responds
                uridine_prescribed = rng.random() < 0.75
                uridine_response = uridine_prescribed and rng.random() < 0.99
                ammonia_elevated = False  # never in UMPS (DDx from OTC)
                otc_initially_suspected = rng.random() < 0.25
                anaemia_hb = round(rng.uniform(4.5, 9.0) if megaloblastic_anaemia else rng.uniform(9.0, 13.0), 1)
                dx_delay_months = rng.randint(1, 60)
                obstructive_uropathy = orotic_acid_crystalluria and rng.random() < 0.20
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "megaloblastic_anaemia": megaloblastic_anaemia,
                    "orotic_acid_crystalluria": orotic_acid_crystalluria,
                    "growth_retardation": growth_retardation,
                    "b12_folate_tried_error": b12_folate_tried_error,
                    "b12_folate_responded": b12_folate_responded,
                    "on_uridine_replacement": uridine_prescribed,
                    "uridine_response_anaemia_corrected": uridine_response,
                    "ammonia_elevated": ammonia_elevated,
                    "otc_deficiency_initially_suspected": otc_initially_suspected,
                    "haemoglobin_g_dL": anaemia_hb,
                    "obstructive_uropathy": obstructive_uropathy,
                    "dx_delay_months": dx_delay_months,
                    "outcome": "normal_growth_and_development" if uridine_response else "untreated_failure_to_thrive",
                })

        gene_data["patients"] = patients


_generate_patients()


def _pct_true(lst, key):
    if not lst:
        return 0
    return round(100 * sum(1 for p in lst if p.get(key)) / len(lst), 1)


def overview():
    all_genes_info = [
        {
            "gene": g["gene"],
            "locus": g["locus"],
            "aa": g["aa"],
            "n_patients": len(g["patients"]),
            "inheritance": g["inheritance"],
        }
        for g in PURINE_GENES
    ]
    total = sum(len(g["patients"]) for g in PURINE_GENES)
    pts = {g["gene"]: g["patients"] for g in PURINE_GENES}

    return {
        "atlas": "Hereditary Purine & Pyrimidine Atlas — Complete 8-Gene Purine and Pyrimidine Metabolism Disorder Atlas",
        "subtitle": (
            "HPRT1 (Xq26.2-XLR-Lesch-Nyhan-SIB-Hyperuricaemia-Choreoathetosis-Allopurinol-Uric-Acid-NOT-Neurology) . "
            "ADSL (22q13.1-AR-SAICAR-SAdo-Urine-PATHOGNOMONIC-3-Phenotype-Spectrum-Autism-Features-Common) . "
            "ADA (20q13.12-AR-ADA-SCID-dATP-Kills-T-B-NK-NBS-TREC-Strimvelis-EMA2016-Gene-Therapy) . "
            "PNP (14q11.2-AR-T-Cell-Selective-dGTP-Toxic-Autoimmune-AIHA-Spastic-Diplegia-HSCT) . "
            "XDH (2p23.1-AR-Xanthinuria-Type-I-Radiolucent-Stones-Uric-Acid-Very-Low-Allopurinol-CI) . "
            "APRT (16q24.3-AR-2-8-DHA-Crystals-Brownish-Pathognomonic-Renal-Failure-Allopurinol-Curative) . "
            "AMPD1 (1p13.3-AR-Ammonia-Absent-Forearm-Test-Lactate-Rises-Common-Polymorphism-Debated) . "
            "UMPS (3q13.33-AR-Orotic-Aciduria-Megaloblastic-Not-B12-Folate-Uridine-Curative-DDx-OTC) -- "
            "320 Patients (8x40, Seeds 1814-1821)"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "genes_covered": 8,
            "patients_per_gene": 40,
            "x_linked_genes": 1,
            "ar_genes": 7,
            # HPRT1
            "hprt1_sib_pct": _pct_true(pts["HPRT1"], "self_injurious_behaviour"),
            "hprt1_complete_lesch_nyhan_pct": _pct_true(pts["HPRT1"], "complete_deficiency_lesch_nyhan"),
            "hprt1_on_allopurinol_pct": _pct_true(pts["HPRT1"], "on_allopurinol"),
            "hprt1_neuro_improvement_allopurinol_pct": _pct_true(pts["HPRT1"], "allopurinol_neuro_improvement"),
            # ADSL
            "adsl_saicar_sado_elevated_pct": _pct_true(pts["ADSL"], "saicar_sado_elevated"),
            "adsl_seizures_pct": _pct_true(pts["ADSL"], "seizures"),
            "adsl_autistic_features_pct": _pct_true(pts["ADSL"], "autistic_features"),
            "adsl_csf_sado_measured_pct": _pct_true(pts["ADSL"], "csf_sado_measured"),
            # ADA
            "ada_nbs_detected_pct": _pct_true(pts["ADA"], "nbs_trec_detected"),
            "ada_t_cell_absent_pct": _pct_true(pts["ADA"], "t_cell_absent"),
            "ada_b_cell_absent_pct": _pct_true(pts["ADA"], "b_cell_absent"),
            "ada_strimvelis_gt_pct": _pct_true(pts["ADA"], "strimvelis_gene_therapy"),
            "ada_hsct_pct": _pct_true(pts["ADA"], "hsct_performed"),
            # PNP
            "pnp_t_cell_lymphopenia_pct": _pct_true(pts["PNP"], "t_cell_lymphopenia"),
            "pnp_b_cell_preserved_pct": _pct_true(pts["PNP"], "b_cell_relatively_preserved"),
            "pnp_aiha_pct": _pct_true(pts["PNP"], "autoimmune_haemolytic_anaemia"),
            "pnp_spastic_diplegia_pct": _pct_true(pts["PNP"], "spastic_diplegia"),
            "pnp_hsct_pct": _pct_true(pts["PNP"], "hsct_performed"),
            # XDH
            "xdh_asymptomatic_pct": _pct_true(pts["XDH"], "asymptomatic"),
            "xdh_xanthine_stones_pct": _pct_true(pts["XDH"], "xanthine_kidney_stones"),
            "xdh_allopurinol_error_pct": _pct_true(pts["XDH"], "allopurinol_prescribed_error"),
            "xdh_type_ii_moco_pct": _pct_true(pts["XDH"], "type_ii_moco_deficiency"),
            # APRT
            "aprt_dha_crystals_pct": _pct_true(pts["APRT"], "dha_crystals_urine"),
            "aprt_renal_failure_pct": _pct_true(pts["APRT"], "renal_failure"),
            "aprt_allopurinol_pct": _pct_true(pts["APRT"], "on_allopurinol"),
            "aprt_japanese_founder_pct": _pct_true(pts["APRT"], "japanese_founder_type_ii"),
            # AMPD1
            "ampd1_ammonia_absent_pct": _pct_true(pts["AMPD1"], "forearm_test_ammonia_absent"),
            "ampd1_lactate_rises_pct": _pct_true(pts["AMPD1"], "forearm_test_lactate_rises_normally"),
            "ampd1_asymptomatic_pct": _pct_true(pts["AMPD1"], "asymptomatic_homozygous"),
            "ampd1_p_gln12ter_pct": _pct_true(pts["AMPD1"], "p_gln12ter_homozygous"),
            # UMPS
            "umps_megaloblastic_anaemia_pct": _pct_true(pts["UMPS"], "megaloblastic_anaemia"),
            "umps_orotic_crystalluria_pct": _pct_true(pts["UMPS"], "orotic_acid_crystalluria"),
            "umps_b12_error_pct": _pct_true(pts["UMPS"], "b12_folate_tried_error"),
            "umps_b12_responded_pct": _pct_true(pts["UMPS"], "b12_folate_responded"),
            "umps_uridine_response_pct": _pct_true(pts["UMPS"], "uridine_response_anaemia_corrected"),
            "umps_ammonia_elevated_pct": _pct_true(pts["UMPS"], "ammonia_elevated"),
        },
        "genes": all_genes_info,
        "top_alerts": [
            "HPRT1-ALLOPURINOL-CONTROLS-URIC-ACID-NOT-NEUROLOGY: Allopurinol and febuxostat normalise uric acid and prevent gout in HPRT1 deficiency but have ZERO effect on neurological symptoms (SIB, choreoathetosis, dystonia, intellectual disability) — never conflate metabolic control with neurological improvement in Lesch-Nyhan syndrome; set family expectations explicitly",
            "ADA-STRIMVELIS-EMA2016-GENE-THERAPY: Strimvelis (ex vivo autologous HSC gene therapy, EMA-approved 2016) is curative for ADA-SCID — dATP accumulation destroys all 3 lymphocyte lineages (T, B, NK); NBS TREC detects absent thymic output before symptomatic infections; PEG-ADA bridges to definitive therapy",
            "PNP-T-CELL-SELECTIVE-CONTRAST-ADA-SCID: PNP deficiency causes T-cell selective immunodeficiency (B cells/NK relatively preserved) — distinct from ADA-SCID (pan-lymphopenia); autoimmune haemolytic anaemia in ~50% and spastic diplegia may predate the immunodeficiency diagnosis; dGTP accumulates specifically in T cells",
            "XDH-ALLOPURINOL-CONTRAINDICATED-RADIOLUCENT-STONES: Xanthinuria type I presents with radiolucent xanthine stones (missed on plain X-ray) and very low uric acid (<100 µmol/L) — allopurinol is CONTRAINDICATED (inhibits absent XDH; also cannot be metabolised); high fluid intake is the main treatment",
            "APRT-ALLOPURINOL-CURATIVE-2-8-DHA-BROWNISH-CRYSTALS: 2,8-DHA brownish crystals in urine are PATHOGNOMONIC for APRT deficiency — progressive renal failure ensues untreated; allopurinol is CURATIVE (blocks XO, reduces 2,8-DHA production); Japanese founder effect makes APRT the most common inherited renal stone disease in Japan",
            "UMPS-URIDINE-CURATIVE-NOT-B12-FOLATE: Hereditary orotic aciduria (UMPS) causes megaloblastic anaemia that does NOT respond to B12 or folate — uridine replacement (uridine triacetate) is curative; blood ammonia is NORMAL (distinguishes UMPS from OTC deficiency where ammonia is elevated; both cause orotic aciduria)",
            "AMPD1-AMMONIA-ABSENT-FOREARM-TEST-PATHOGNOMONIC-LACTATE-RISES: AMPD1 deficiency forearm exercise test: ammonia does NOT rise, lactate RISES normally — OPPOSITE of McArdle disease (PYGM: lactate flat, ammonia rises); the p.Gln12Ter polymorphism is common in Europeans (~2% allele frequency) but many homozygous individuals are asymptomatic",
            "ADSL-SAICAR-SADO-PATHOGNOMONIC-ROUTINE-SCREENS-MISS: SAICAR and SAdo (succinylpurines) in urine/CSF are pathognomonic for ADSL deficiency — routine amino acids and standard organic acid screens MISS this; request specific urine succinylpurines in any child with unexplained psychomotor retardation + seizures + autistic features",
            "PURINE-DISORDER-GENE-PANEL-MANDATORY-NBS-MISSES-MOST: Standard NBS detects ADA-SCID (TREC) but misses HPRT1, ADSL, PNP, XDH, APRT, AMPD1, and UMPS — comprehensive purine/pyrimidine gene panel is essential for any unexplained immunodeficiency, self-injurious behaviour in children, unexplained crystalluria, or megaloblastic anaemia not responding to B12/folate",
            "HPRT1-SIB-BEGINS-12-24-MONTHS-GOUT-CHILDREN-SCREEN: Self-injurious behaviour in a toddler (12-24 months) should trigger HPRT1 enzyme assay immediately; conversely, gout in any male <30 years or child warrants HPRT1 screening for Kelley-Seegmiller partial deficiency — early-onset gout in young males is the most common missed presentation",
        ],
    }


def breakdown():
    result = []
    for idx, g in enumerate(PURINE_GENES):
        pts = g["patients"]
        ec = {}
        for p in pts:
            et = (
                p.get("phenotype") or
                p.get("clinical_form") or
                ("symptomatic" if p.get("exercise_intolerance") else None) or
                ("megaloblastic_anaemia" if p.get("megaloblastic_anaemia") else None) or
                ("urolithiasis" if p.get("xanthine_kidney_stones") or p.get("urolithiasis") else None) or
                ("asymptomatic" if p.get("asymptomatic") or p.get("asymptomatic_homozygous") else None) or
                "other"
            )
            ec[et] = ec.get(et, 0) + 1
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "etiology_counts": ec,
            "computed": {
                "n_patients": len(pts),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": pts[:10],
        })
    return result


def definitions():
    return {
        "concepts": {
            "Purine & Pyrimidine Metabolism — Biochemistry and Clinical Classification": (
                "Purines (adenine and guanine) and pyrimidines (cytosine, thymine, uracil) are the nitrogenous bases "
                "of nucleotides, which serve as structural components of DNA and RNA, energy currency (ATP, GTP), "
                "and signalling molecules (cAMP, cGMP). Humans synthesise these via two pathways: "
                "DE NOVO SYNTHESIS: builds purine/pyrimidine rings from small precursors (amino acids, CO2, one-carbon units); "
                "energetically expensive; highly regulated; "
                "SALVAGE PATHWAY: recycles free bases and nucleosides from nucleotide degradation back to nucleoside monophosphates; "
                "economical; critical in tissues with limited de novo capacity (especially brain, erythrocytes). "
                "PURINE CATABOLISM: "
                "AMP → IMP → hypoxanthine → xanthine → uric acid (via xanthine oxidase/XDH); "
                "GMP → guanosine → guanine → xanthine → uric acid; "
                "uric acid is the end product in humans (unlike rodents which have uricase); "
                "uric acid excreted by kidneys; excess → gout, tophi, nephrolithiasis. "
                "PURINE SALVAGE: "
                "HPRT1: hypoxanthine + PRPP → IMP; guanine + PRPP → GMP; "
                "APRT: adenine + PRPP → AMP; "
                "ADA: adenosine → inosine (deamination); deoxyadenosine → deoxyinosine; "
                "PNP: inosine/guanosine/deoxyguanosine → hypoxanthine/guanine + ribose-1-P. "
                "PYRIMIDINE DE NOVO SYNTHESIS (UMPS): "
                "Carbamoyl phosphate + aspartate → dihydroorotate → orotate (via DHODH) → OMP (UMPS/OPRT) → UMP (UMPS/OMPDC); "
                "UMP → UDP → UTP → CTP; UMP → dTMP → dTTP. "
                "CLINICAL CLASSIFICATION OF PURINE/PYRIMIDINE DISORDERS: "
                "GROUP 1 — IMMUNODEFICIENCY: ADA (pan-SCID), PNP (T-cell selective); "
                "toxic dNTP accumulation → lymphocyte apoptosis. "
                "GROUP 2 — HYPERURICAEMIA/GOUT: HPRT1 (Lesch-Nyhan/Kelley-Seegmiller); "
                "failure of salvage → uric acid overproduction + PRPP accumulation. "
                "GROUP 3 — UROLITHIASIS/RENAL: XDH (xanthine stones), APRT (2,8-DHA stones); "
                "insoluble metabolite accumulation → crystal nephropathy. "
                "GROUP 4 — NEURODEVELOPMENTAL: ADSL (SAICAR/SAdo accumulation), HPRT1 (basal ganglia dopamine); "
                "de novo synthesis failure or toxic metabolite accumulation in CNS. "
                "GROUP 5 — MUSCLE: AMPD1 (exercise intolerance, purine nucleotide cycle failure). "
                "GROUP 6 — HAEMATOPOIETIC/PYRIMIDINE: UMPS (orotic aciduria, megaloblastic anaemia); "
                "pyrimidine starvation → defective DNA replication in proliferating cells. "
                "DIAGNOSTIC APPROACH: "
                "Plasma uric acid (high in HPRT1; very low in PNP, XDH); "
                "urine purine quantitation (HPLC): hypoxanthine, xanthine, uric acid, succinylpurines, orotic acid; "
                "lymphocyte subsets (ADA, PNP); "
                "enzyme assays in erythrocytes (HPRT1, APRT, ADA, PNP); "
                "gene panel: all 8 genes above plus ADSL, MOCS1/MOCS2 (MoCo deficiency). "
            ),
            "Forearm Exercise Test — Ammonia vs Lactate Response (AMPD1 vs PYGM)": (
                "The forearm exercise test (ischaemic or non-ischaemic modification) is the key "
                "bedside diagnostic test for two distinct enzyme deficiencies causing exercise intolerance: "
                "AMPD1 (myoadenylate deaminase) and PYGM (muscle phosphorylase, McArdle disease). "
                "The two conditions produce OPPOSITE results on the test — a critical clinical pearl. "
                "TEST PROTOCOL (non-ischaemic version, safer): "
                "Patient fasted 4 hours; IV cannula in antecubital fossa; "
                "baseline samples: venous lactate + ammonia; "
                "vigorous forearm exercise (squeeze dynamometer) for 60-90 seconds at near-maximum effort; "
                "samples at 1, 3, 5, 10 minutes post-exercise. "
                "NORMAL RESPONSE: "
                "Lactate: rises 2-3× baseline (glycolysis: glucose → pyruvate → lactate); "
                "Ammonia: rises 2-3× baseline (purine nucleotide cycle: AMP → IMP via AMPD1 → NH3). "
                "AMPD1 DEFICIENCY RESULT: "
                "Lactate: RISES NORMALLY (glycolysis intact, glycogen → G6P → pyruvate → lactate); "
                "Ammonia: does NOT RISE (AMPD1 absent, AMP cannot be deaminated to IMP, no NH3 produced); "
                "pattern: normal lactate, absent ammonia = PATHOGNOMONIC for AMPD1 deficiency. "
                "PYGM DEFICIENCY (McARDLE DISEASE) RESULT: "
                "Lactate: does NOT RISE (glycogen phosphorylase absent, muscle glycogen cannot be mobilised, "
                "no glycolytic substrate for lactate production); "
                "Ammonia: RISES NORMALLY (purine nucleotide cycle is intact, AMPD1 functional, NH3 produced); "
                "pattern: absent lactate, normal ammonia = PATHOGNOMONIC for McArdle disease (PYGM). "
                "CLINICAL UTILITY: "
                "The forearm exercise test differentiates AMPD1 from PYGM without biopsy; "
                "if BOTH lactate and ammonia fail to rise: poor effort or incorrect ischaemic protocol — repeat; "
                "follow-up after abnormal test: muscle biopsy (AMPD1 histochemistry; PAS staining for glycogen in PYGM); "
                "gene sequencing: AMPD1 p.Gln12Ter (most common); PYGM p.Arg50Ter (most common, Europeans). "
                "COMMON ERROR: confusing the two conditions based on symptoms alone (both cause exercise cramps) "
                "without performing the forearm exercise test first. "
            ),
            "Urine Metabolite Biomarkers in Purine Disorders — SAICAR/SAdo/Orotic Acid/2,8-DHA": (
                "Specific urine metabolites are pathognomonic biomarkers for individual purine and pyrimidine disorders. "
                "These are NOT detected by standard urine amino acid screens or routine organic acid screens "
                "— dedicated testing is required. "
                "1. SAICAR (succinyl-AICA-ribotide) and SAdo (succinyladenosine) — ADSL DEFICIENCY: "
                "Both metabolites accumulate in urine and CSF; "
                "detected by urine succinylpurines (HPLC/LC-MS); "
                "SAICAR:SAdo ratio correlates with phenotype severity; "
                "also detectable in CSF — CSF SAdo is highly specific; "
                "NOT on standard organic acid chromatography; "
                "clinical suspicion required to request the specific test. "
                "2. OROTIC ACID — UMPS DEFICIENCY (and OTC, CAD, other UCD): "
                "Orotic acid elevation is detectable on urine organic acids (orotic acid peak); "
                "or by dedicated urine pyrimidines quantitation; "
                "UMPS: orotic acid elevated, ammonia NORMAL; "
                "OTC deficiency: orotic acid elevated, ammonia ELEVATED — KEY DISTINCTION; "
                "clinical presentation (megaloblastic anaemia, crystalluria) guides UMPS workup. "
                "3. 2,8-DIHYDROXYADENINE (2,8-DHA) — APRT DEFICIENCY: "
                "2,8-DHA is detectable in urine by HPLC or LC-MS/MS; "
                "urine microscopy: brown, round, birefringent crystals (Maltese cross under polarised light); "
                "plasma 2,8-DHA measurable in symptomatic patients; "
                "can be mistaken for uric acid crystals on routine microscopy — specific assay required; "
                "2,8-DHA deposits in renal tubules cause progressive nephropathy. "
                "4. XANTHINE/HYPOXANTHINE — XDH DEFICIENCY: "
                "Urine xanthine elevated (normal <15 µmol/mmol creatinine); "
                "plasma xanthine elevated (normal <15 µmol/L; XDH deficiency: 200-600 µmol/L); "
                "uric acid very low (<100 µmol/L) — the primary diagnostic clue; "
                "xanthine/oxypurines quantitation by HPLC. "
                "PRACTICAL APPROACH: "
                "Any unexplained urolithiasis: check uric acid level (very low → XDH; very high → HPRT1 check); "
                "any child with crystalluria: urine microscopy + APRT assay + succinylpurines; "
                "megaloblastic anaemia not responding to B12/folate: urine organic acids for orotic acid; "
                "unexplained epileptic encephalopathy + autism: request urine succinylpurines (ADSL); "
                "targeted testing saves diagnostic years. "
            ),
            "Immunodeficiency in Purine Disorders — ADA-SCID vs PNP Deficiency": (
                "Two purine metabolism enzymes — adenosine deaminase (ADA) and purine nucleoside phosphorylase (PNP) "
                "— when deficient, cause distinct forms of primary immunodeficiency through the accumulation of "
                "toxic deoxyribonucleoside triphosphates in lymphocytes. "
                "ADA-SCID — MECHANISM: "
                "ADA deficiency → deoxyadenosine accumulates → phosphorylated to dATP by dCK (deoxycytidine kinase); "
                "dATP inhibits ribonucleotide reductase → blocks DNA synthesis → apoptosis in ALL lymphocytes; "
                "T cells, B cells, AND NK cells all destroyed (pan-lymphocyte destruction); "
                "deoxyadenosine also directly toxic (promotes apoptosis via DNA strand breaks). "
                "PNP DEFICIENCY — MECHANISM: "
                "PNP deficiency → deoxyguanosine accumulates → phosphorylated to dGTP by dGK (deoxyguanosine kinase); "
                "dGTP inhibits ribonucleotide reductase → specifically toxic to T cells "
                "(T cells have high dGK activity and limited dGTP export capacity); "
                "B cells and NK cells have lower dGK activity → relatively spared. "
                "KEY CLINICAL DISTINCTIONS: "
                "ADA-SCID: T, B, NK all absent; presents at birth/early infancy; severe infections immediately; "
                "no autoimmune features; NBS TREC detected; Strimvelis gene therapy available. "
                "PNP deficiency: T cells absent, B cells/NK relatively preserved; "
                "onset later (6-18 months, after maternal antibody wanes); "
                "AUTOIMMUNE complications in ~50% (AIHA, ITP, SLE-like) — unique to PNP; "
                "spastic diplegia and intellectual disability (neurological features) in PNP; "
                "uric acid very low in PNP (PNP provides substrate for XO → uric acid); "
                "HSCT less successful in PNP than ADA (due to neurological damage, autoimmunity, infections pre-HSCT). "
                "TREATMENT COMPARISON: "
                "ADA-SCID: PEG-ADA (bridge) → Strimvelis gene therapy (EMA 2016) or HSCT; gene therapy preferred; "
                "PNP: HSCT (curative intent); gene therapy in trials; no approved ERT equivalent; "
                "early treatment before infections and neurological damage improves outcomes in both. "
                "LABORATORY DIFFERENTIATION: "
                "ADA: lymphocyte subsets (T, B, NK all low); ADA enzyme activity (erythrocytes); "
                "dATP pool (plasma); "
                "PNP: T-cell lymphopenia + preserved B cells; PNP enzyme activity; uric acid very low; "
                "plasma deoxyguanosine elevated; "
                "both: gene sequencing for confirmation and prenatal/carrier testing. "
            ),
        }
    }
