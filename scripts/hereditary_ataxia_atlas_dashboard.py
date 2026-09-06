#!/usr/bin/env python3
"""Hereditary-Ataxia-Atlas — Complete 8-Gene Hereditary Recessive Ataxia Atlas
FXN      (Frataxin; 210 aa; 9q21.11; AR;
          Friedreich Ataxia (FRDA) — most common inherited ataxia (1/50,000);
          GAA TRINUCLEOTIDE REPEAT EXPANSION (intron 1) — standard gene panels MISS IT;
          Request SPECIFIC FXN REPEAT ASSAY (PCR/Southern blot);
          Normal <34 GAA repeats; pathogenic >66; borderline 34-66;
          Dorsal root ganglion (DRG) degeneration → sensory loss + areflexia;
          Hypertrophic cardiomyopathy in 80% — ECG + echocardiogram MANDATORY at diagnosis;
          Idebenone 5 mg/kg/day — cardiac benefit, modest neurological effect;
          Frataxin protein replacement (omaveloxolone FDA-approved 2023 ≥16 years);
          Diabetes mellitus 20-30%; scoliosis 80%; foot deformity (pes cavus) 90%) ·
APTX     (Aprataxin; 342 aa; 9p21.1; AR;
          Ataxia with Oculomotor Apraxia type 1 (AOA1) — second most common recessive ataxia in Japan;
          OCULOMOTOR APRAXIA: difficulty initiating horizontal saccades — head thrust compensates;
          HYPOALBUMINAEMIA (<3.5 g/dL) + HYPERCHOLESTEROLAEMIA → metabolic signature (no telangiectasia);
          Chorea in early stages, replaced by ataxia as disease progresses;
          Onset age 2-18 years; rapidly progressive over first decade;
          Peripheral neuropathy (axonal, sensorimotor) + cerebellar atrophy;
          Aprataxin repairs abortive ligations of DNA strand breaks) ·
SETX     (Senataxin; 2677 aa; 9q34.13; AR;
          Ataxia with Oculomotor Apraxia type 2 (AOA2) — most common recessive ataxia in Europe;
          AFP (alpha-fetoprotein) ELEVATED >10 mcg/L in >80% — diagnostic hallmark;
          AFP elevated in serum even when disease clinically mild — check AFP first;
          Onset late teen (15-25 years); slower progression than AOA1;
          Axonal sensorimotor neuropathy — absent lower-limb reflexes early;
          NO telangiectasia, NO immunodeficiency (DDx from A-T);
          Senataxin is an RNA-DNA helicase resolving R-loops during transcription) ·
ATM      (ATM Ser/Thr Kinase; 3056 aa; 11q22.3; AR;
          Ataxia-Telangiectasia (A-T) — combined immunodeficiency + cerebellar ataxia;
          TELANGIECTASIA appear 5-8 years (AFTER ataxia onset at 1-2 years);
          IgA deficiency in 40-80%; total IgG/IgM also low; recurrent sinopulmonary infections;
          CANCER SURVEILLANCE: 35% lifetime cancer risk (T-cell ALL in childhood; breast in ATM carriers);
          RADIOSENSITIVITY: avoid ionising radiation when possible (fragmented DSB repair);
          AFP elevated (useful DDx marker even in childhood A-T);
          Wheelchair by age 10-15; swallowing difficulty + aspiration pneumonia in adolescence;
          ATM heterozygous carriers (parents): ~2-fold breast cancer risk — mammography surveillance) ·
SACS     (Sacsin; 4579 aa; 13q12.12; AR;
          ARSACS — Autosomal Recessive Spastic Ataxia of Charlevoix-Saguenay;
          SPASTIC ATAXIA (cerebellar + UMN signs): early spasticity distinguishes ARSACS from pure cerebellar;
          Charlevoix-Saguenay, Quebec FOUNDER: p.Asp2521Asn (>90% Quebec ARSACS alleles);
          Retinal myelination of nerve-fibre layer — ophthalmoscopy finding (>80%);
          Onset age 1-2 years (gait instability); spasticity present from early childhood;
          Peripheral neuropathy (demyelinating); mitochondrial dysfunction mechanism;
          Sacsin maintains mitochondrial morphology via DNAJ chaperone interaction) ·
ANO10    (Anoctamin 10 / TMEM16K; 660 aa; 3p22.3; AR;
          SCAR10 — Slowly Progressive Cerebellar Ataxia;
          Adult onset (2nd-4th decade) — median age 25 years;
          Slowly progressive, often not wheelchair-dependent for decades;
          Pure cerebellar ataxia without neuropathy or oculomotor apraxia (early disease);
          Cognitive decline and seizures in subset (10-20%);
          Anoctamin-10 is a Ca2+-activated chloride channel expressed in cerebellum;
          MRI: cerebellar atrophy, often vermis predominant) ·
ADCK3    (AarF Domain Containing Kinase 3 / COQ8A; 669 aa; 1q42.13; AR;
          ARCA2 — Autosomal Recessive Cerebellar Ataxia type 2 / CoQ10-deficiency ataxia;
          COENZYME Q10 SUPPLEMENTATION trial mandatory: 5-10 mg/kg/day in 2 divided doses;
          ~20-40% show meaningful benefit from CoQ10 (reduction of ataxia, fatigue, exercise intolerance);
          ELEVATED CK + EXERCISE INTOLERANCE: frequent clue to mitochondrial aetiology;
          Muscle biopsy: reduced CoQ10 content + mitochondrial respiratory chain deficiency;
          SEIZURES in 30-40%; cerebellar atrophy on MRI;
          ADCK3 phosphorylates CoQ biosynthesis enzymes — LOF → reduced CoQ10 in muscle/brain) ·
ABHD12   (Abhydrolase Domain Containing 12; 398 aa; 20p11.21; AR;
          PHARC — Polyneuropathy, Hearing loss, Ataxia, Retinitis Pigmentosa, Cataract;
          PHARC ACRONYM SEQUENCE: neuropathy + SNHL appear first (teenage), then ataxia, then RP, then cataract;
          Retinitis pigmentosa (RP): night blindness → peripheral visual field loss → central loss;
          Sensorineural hearing loss (SNHL) often first symptom — audiogram in all early-onset neuropathy;
          ABHD12 hydrolyses lyso-phosphatidylserine and 2-arachidonoylglycerol (2-AG endocannabinoid);
          2-AG accumulates in CNS → microglial activation → demyelination;
          Iran / Palestine founder mutations (frameshift c.846+1G>T, 20p11.21 deletion);
          No disease-modifying therapy; management symptomatic for each PHARC component)
320-patient aggregate cohort (8 × 40, seeds 1390-1397)
"""

import random

SEED_BASE = 1390

ATAXIA_GENES = [
    # ── FXN — Friedreich Ataxia ──
    {
        "gene": "FXN",
        "protein": "Frataxin",
        "alias": (
            "FXN; OMIM gene 606829; FRDA #229300; "
            "9q21.11; 210 aa; ~23 kDa mitochondrial iron chaperone; "
            "Frataxin imports iron into mitochondria for Fe-S cluster biogenesis and haem synthesis; "
            "LOF → mitochondrial iron accumulation → oxidative stress → neurodegeneration (DRG, spinocerebellar tracts); "
            "GAA·TTC REPEAT EXPANSION in intron 1 — biallelic; normal ≤33 repeats; "
            "Pre-mutation 34-65; full expansion ≥66 GAA; most patients have 600-1000 GAA on each allele; "
            "STANDARD GENE PANELS MISS FXN: repeat expansion NOT detected by exome/WES — request SPECIFIC FXN REPEAT ASSAY; "
            "Omaveloxolone (Nrf2 activator, Skyclarys): FDA-approved Feb 2023 for ≥16 years; slows neurological decline"
        ),
        "aa": "210 aa",
        "kDa": "~23 kDa",
        "locus": "9q21.11",
        "omim_gene": 606829,
        "omim_disease": 229300,
        "inheritance": "AR — biallelic GAA repeat expansion (intron 1); haploinsufficiency of frataxin",
        "gene_class": (
            "FXN encodes Frataxin, a mitochondrial protein critical for iron-sulphur (Fe-S) cluster "
            "assembly. Fe-S clusters are required for mitochondrial respiratory chain complexes I, II, III "
            "and aconitase. LOF leads to mitochondrial iron accumulation, free radical generation, and "
            "selective neurodegeneration of dorsal root ganglia (DRG) and spinocerebellar tracts. "
            "The GAA expansion suppresses transcription via heterochromatin silencing of the FXN gene."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("Biallelic GAA repeat expansion (both alleles ≥66 repeats) — classic FRDA", 0.96),
            ("GAA expansion + point mutation / compound heterozygote — atypical FRDA", 0.04),
        ],
        "age_onset_years_range": (5, 25),
        "sex_ratio_M": 0.50,
        "rates": {
            "gait_ataxia":                    0.99,
            "areflexia":                      0.97,
            "sensory_loss":                   0.90,
            "hypertrophic_cardiomyopathy":    0.80,
            "scoliosis":                      0.80,
            "pes_cavus":                      0.90,
            "diabetes_mellitus":              0.25,
            "optic_atrophy":                  0.25,
            "dysarthria":                     0.90,
            "dysphagia":                      0.55,
            "wheelchair_dependent_by_25y":    0.75,
        },
        "hallmarks": [
            "GAA REPEAT EXPANSION: standard panels miss — request FXN-specific repeat PCR/Southern blot",
            "CARDIOMYOPATHY 80%: ECG + echocardiogram at diagnosis; repeat annually; cause of death in 50%",
            "DRG DEGENERATION: sensory loss + absent reflexes BEFORE cerebellar signs (distinguishes FRDA)",
            "PES CAVUS + SCOLIOSIS: foot deformity 90%; scoliosis 80% — early orthopaedic referral",
            "OMAVELOXOLONE (Skyclarys): FDA-approved 2023 ≥16y; Nrf2 activator; slows neurological decline",
            "DIABETES 25%: annual HbA1c from age 15; insulin dependence in subset",
            "OPTIC ATROPHY 25%: annual visual acuity + colour vision; OCT if available",
            "WHEELCHAIR by 25y in 75%: physiotherapy + occupational therapy from diagnosis",
        ],
        "treatment_alerts": [
            "FXN REPEAT ASSAY MANDATORY: exome/WES misses GAA expansion — specific PCR/Southern blot required",
            "OMAVELOXOLONE: Skyclarys 150 mg once daily — approved ≥16 years; watch hepatotoxicity (monthly LFTs x6 months)",
            "IDEBENONE 5 mg/kg/day: antioxidant; cardiac benefit; modest neurological effect; reasonable adjunct",
            "ECHOCARDIOGRAM ANNUALLY: hypertrophic cardiomyopathy; refer cardiology if LV outflow gradient or arrhythmia",
            "DIABETES SCREEN: annual HbA1c from diagnosis; SGLT2 inhibitors avoid in very low BMI patients",
        ],
        "organ_system": "cerebellum + DRG + spinocerebellar tracts + heart (cardiomyopathy) + pancreas",
        "primary_treatment": "Omaveloxolone 150 mg/day (≥16y); idebenone 5 mg/kg/day; cardiology surveillance; physio",
    },
    # ── APTX — AOA1 ──
    {
        "gene": "APTX",
        "protein": "Aprataxin",
        "alias": (
            "APTX; OMIM gene 606350; AOA1 #208920; "
            "9p21.1; 342 aa; ~40 kDa histidine triad nucleotide-binding (HIT) protein; "
            "Aprataxin resolves abortive DNA ligation intermediates (5'-AMP-DNA adducts) during BER; "
            "LOF → accumulation of strand-break intermediates → DNA damage → cerebellar neurodegeneration; "
            "Most common recessive ataxia in Japan and Portugal (founder mutations in each population); "
            "Japan: W279* (p.Trp279Ter) most common; Portugal: separate cluster of founder alleles; "
            "Oculomotor apraxia type 1: saccade initiation failure → compensatory head thrust (DDx gaze palsy); "
            "Metabolic signature: HYPOALBUMINAEMIA + HYPERCHOLESTEROLAEMIA (no telangiectasia, DDx from A-T)"
        ),
        "aa": "342 aa",
        "kDa": "~40 kDa",
        "locus": "9p21.1",
        "omim_gene": 606350,
        "omim_disease": 208920,
        "inheritance": "AR — biallelic LOF; Japanese founder p.Trp279Ter; Portuguese founder alleles",
        "gene_class": (
            "APTX encodes Aprataxin, a DNA-repair enzyme that removes dead-end ligation intermediates — "
            "specifically the 5'-adenylate (AMP) moiety covalently attached to a 5'-phosphate after "
            "abortive DNA ligation. Without Aprataxin, AMP-DNA adducts accumulate during base-excision "
            "repair, blocking further repair and causing persistent SSBs, which are especially toxic to "
            "post-mitotic neurons (cerebellar Purkinje cells and DRG neurons)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("Nonsense / frameshift → haploinsufficiency (includes p.Trp279* Japan founder)", 0.55),
            ("Missense (HIT domain disruption) → aprataxin loss of function", 0.30),
            ("Compound heterozygote (nonsense + missense)", 0.15),
        ],
        "age_onset_years_range": (2, 18),
        "sex_ratio_M": 0.50,
        "rates": {
            "cerebellar_ataxia":          0.99,
            "oculomotor_apraxia":         0.95,
            "hypoalbuminaemia":           0.75,
            "hypercholesterolaemia":      0.70,
            "chorea_early_disease":       0.50,
            "axonal_sensorimotor_neuropathy": 0.85,
            "cerebellar_atrophy_mri":     0.95,
            "absent_lower_limb_reflexes": 0.90,
            "dysarthria":                 0.85,
            "wheelchair_dependent_10y":   0.60,
        },
        "hallmarks": [
            "OCULOMOTOR APRAXIA: delayed saccade initiation + head thrust — NOT gaze palsy (VOR intact)",
            "HYPOALBUMINAEMIA (<3.5 g/dL) + HYPERCHOLESTEROLAEMIA: metabolic signature of AOA1",
            "NO TELANGIECTASIA: key DDx from Ataxia-Telangiectasia (ATM)",
            "CHOREA IN EARLY DISEASE: involuntary movements precede ataxia in some; resolves as ataxia worsens",
            "JAPAN / PORTUGAL POPULATIONS: high carrier frequency; test APTX first in these ancestries",
            "RAPIDLY PROGRESSIVE: wheelchair-dependent within 10-15 years of onset",
            "AXONAL NEUROPATHY: NCS/EMG shows reduced SNAP amplitudes; absent reflexes early",
            "CEREBELLAR ATROPHY: MRI shows vermis > hemispheres atrophy",
        ],
        "treatment_alerts": [
            "CHECK ALBUMIN + CHOLESTEROL: hypoalbuminaemia <3.5 g/dL is a diagnostic clue and may affect drug dosing",
            "NO SPECIFIC NEUROPROTECTIVE THERAPY: physiotherapy + occupational therapy; speech for dysarthria",
            "DISTINGUISH FROM A-T: no telangiectasia, no immunodeficiency, no AFP elevation (early stage), lower cancer risk",
            "FOUNDER TESTING: Japanese ancestry → p.Trp279Ter first; Portuguese → local founder panel",
            "CHOREA MANAGEMENT: if troublesome, low-dose benzodiazepine or clonazepam; avoid antipsychotics long-term",
        ],
        "organ_system": "cerebellum + DRG + peripheral nerve (axonal neuropathy); no heart/immune involvement",
        "primary_treatment": "Supportive — physiotherapy, OT, speech therapy; check albumin/cholesterol; no disease-modifying therapy",
    },
    # ── SETX — AOA2 ──
    {
        "gene": "SETX",
        "protein": "Senataxin",
        "alias": (
            "SETX; OMIM gene 608465; AOA2 #606002; "
            "9q34.13; 2677 aa; ~303 kDa RNA-DNA helicase; "
            "Senataxin resolves R-loops (RNA-DNA hybrids) during transcription termination; "
            "LOF → R-loop accumulation → transcription-replication conflicts → DNA DSBs → neurodegeneration; "
            "Most common recessive ataxia in Europe (France, North Africa, Middle East); "
            "AFP ELEVATION: serum AFP >10 mcg/L in >80% patients — best diagnostic biomarker; "
            "No telangiectasia, no immunodeficiency (distinguishes AOA2 from A-T); "
            "Dominant gain-of-function SETX mutations cause juvenile ALS4 — opposite phenotype"
        ),
        "aa": "2677 aa",
        "kDa": "~303 kDa",
        "locus": "9q34.13",
        "omim_gene": 608465,
        "omim_disease": 606002,
        "inheritance": "AR — biallelic LOF (dominant GOF → ALS4, distinct from AOA2); European + North African founder alleles",
        "gene_class": (
            "SETX encodes Senataxin, a superfamily 1 helicase that unwinds RNA-DNA hybrid structures "
            "(R-loops) formed when nascent RNA re-anneals to the DNA template during transcription. "
            "R-loop resolution is critical at transcription termination sites and during replication. "
            "In senataxin-deficient cells, R-loops persist, causing transcription-replication collisions, "
            "DNA double-strand breaks, and activation of the ATM-dependent DNA damage response. "
            "Purkinje cells and DRG neurons are selectively vulnerable."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("Truncating variant (nonsense/frameshift/splice) → senataxin LOF", 0.60),
            ("Missense in helicase domain → loss of unwinding activity", 0.30),
            ("Large intragenic deletion (detected by MLPA)", 0.10),
        ],
        "age_onset_years_range": (15, 25),
        "sex_ratio_M": 0.45,
        "rates": {
            "cerebellar_ataxia":                  0.99,
            "elevated_afp":                       0.85,
            "oculomotor_apraxia":                 0.75,
            "axonal_sensorimotor_neuropathy":      0.90,
            "cerebellar_atrophy_mri":              0.95,
            "tremor":                              0.50,
            "absent_lower_limb_reflexes":          0.88,
            "dysarthria":                          0.80,
            "scoliosis":                           0.30,
            "elevated_creatine_kinase_mild":       0.30,
        },
        "hallmarks": [
            "AFP >10 mcg/L: check SERUM AFP first in any recessive ataxia — simplest diagnostic step",
            "NO TELANGIECTASIA: AFP elevated but NO telangiectasia → AOA2 not A-T",
            "LATE TEEN ONSET: median age 15-25 years (later than AOA1 or FRDA)",
            "SLOWLY PROGRESSIVE: neurological plateau in some; life expectancy near-normal",
            "OCULOMOTOR APRAXIA 75%: less severe than AOA1; may be absent early",
            "SENSORIMOTOR AXONAL NEUROPATHY: NCS shows reduced SNAP, CMAPs; areflexia in lower limbs",
            "EUROPEAN / NORTH AFRICAN: highest prevalence; specific alleles in French, Moroccan, Tunisian",
            "DOMINANT SETX → ALS4 (juvenile ALS): opposite phenotype — same gene, opposite mechanism",
        ],
        "treatment_alerts": [
            "AFP AS DIAGNOSTIC BIOMARKER: check serum AFP in all recessive ataxias — rapid, cheap, high sensitivity",
            "NO DISEASE-MODIFYING THERAPY: physiotherapy; occupational therapy; speech therapy",
            "DDx FROM A-T: AFP elevated in both — check for telangiectasia + immunoglobulins to distinguish",
            "SLOW PROGRESSION: counsel patients that AOA2 is slower than FRDA or A-T; realistic functional goals",
            "GENE PANEL TESTING: SETX LOF biallelic = AOA2; dominant missense = ALS4 — clarify zygosity",
        ],
        "organ_system": "cerebellum + DRG + peripheral nerve (axonal neuropathy); no heart/immune involvement",
        "primary_treatment": "Supportive — physiotherapy, OT, speech; AFP as diagnostic/monitoring marker; no disease-modifying therapy",
    },
    # ── ATM — Ataxia-Telangiectasia ──
    {
        "gene": "ATM",
        "protein": "ATM Serine/Threonine Kinase",
        "alias": (
            "ATM; OMIM gene 607585; Ataxia-Telangiectasia #208900; "
            "11q22.3; 3056 aa; ~350 kDa PIKK family kinase; "
            "ATM is the master sensor-kinase of DNA double-strand breaks (DSBs); "
            "LOF → failure to halt cell cycle at DSBs → chromosomal instability → cancer + neurodegeneration; "
            "Ataxia onset age 1-2 years; TELANGIECTASIA on bulbar conjunctivae / ears / sun-exposed areas by 5-8y; "
            "Combined immunodeficiency: IgA deficiency 40-80%; lymphopenia; recurrent sinopulmonary infections; "
            "CANCER: 35% lifetime risk (T-ALL childhood; NHL; breast cancer in ATM heterozygous carriers ~2× risk); "
            "RADIOSENSITIVITY: ATM-null cells hypersensitive to ionising radiation — avoid X-rays when possible; "
            "AFP elevated from birth (unlike AOA2 where AFP rises with age)"
        ),
        "aa": "3056 aa",
        "kDa": "~350 kDa",
        "locus": "11q22.3",
        "omim_gene": 607585,
        "omim_disease": 208900,
        "inheritance": "AR — biallelic LOF; ATM heterozygous carriers have 2× breast cancer risk (important for parents)",
        "gene_class": (
            "ATM (Ataxia-Telangiectasia Mutated) encodes a PI3K-like kinase that is the central "
            "coordinator of the DNA double-strand break (DSB) response. On DSB detection, ATM autophosphorylates "
            "and phosphorylates hundreds of substrates including H2AX, CHK2, p53, BRCA1, and MDM2, "
            "coordinating cell cycle arrest, DNA repair (NHEJ/HR), and apoptosis. "
            "Biallelic LOF → chromosomal instability at TCR/IgH loci → lymphoid malignancy + "
            "progressive Purkinje cell death + thymic hypoplasia → T-cell immunodeficiency."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("Truncating variant (nonsense/frameshift/splice) → null allele, classic A-T", 0.70),
            ("Missense (kinase domain) → partial LOF, attenuated A-T phenotype", 0.20),
            ("Compound heterozygote (null + missense) — variable phenotype", 0.10),
        ],
        "age_onset_years_range": (1, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "progressive_cerebellar_ataxia":     0.99,
            "oculocutaneous_telangiectasia":     0.95,
            "iga_deficiency":                    0.70,
            "recurrent_sinopulmonary_infections": 0.70,
            "elevated_afp":                      0.90,
            "lymphopenia":                       0.80,
            "wheelchair_by_15y":                 0.85,
            "malignancy_by_30y":                 0.35,
            "oculomotor_apraxia":                0.85,
            "insulin_resistant_diabetes":         0.25,
            "elevated_creatine_kinase":           0.40,
        },
        "hallmarks": [
            "TELANGIECTASIA AFTER ATAXIA: ataxia onset 1-2y; telangiectasia on conjunctivae by 5-8y (NOT at birth)",
            "AFP ELEVATED FROM BIRTH: unlike AOA2 (which rises over years) — AFP high in A-T from infancy",
            "IgA DEFICIENCY: 40-80%; recurrent sinopulmonary infections → bronchiectasis by adulthood",
            "CANCER SURVEILLANCE: T-cell ALL (childhood); NHL; avoid radiation — use MRI over CT when possible",
            "RADIOSENSITIVITY: even diagnostic X-rays may increase mutation burden — minimise ionising radiation",
            "WHEELCHAIR BY 15y: 85% non-ambulatory by mid-teen years; preserve bulbar function with SLT",
            "ATM HETEROZYGOUS CARRIERS (parents): 2× breast cancer risk — mammography screening from age 40",
            "IMMUNOGLOBULIN REPLACEMENT: IgG<4 g/L → consider IVIG or SCIG therapy",
        ],
        "treatment_alerts": [
            "AVOID IONISING RADIATION: diagnostic X-rays/CT only when essential; MRI preferred for surveillance",
            "CANCER SURVEILLANCE: annual blood count; low threshold for lymph node biopsy; breast screen for carriers",
            "IVIG/SCIG: if IgG low + recurrent infections; protects lung from bronchiectasis",
            "IMMUNISATION: live attenuated vaccines CONTRAINDICATED (varicella, MMR) due to T-cell deficiency",
            "CARRIER PARENTS: counsel about 2× breast cancer risk; offer mammography + MRI from age 40",
        ],
        "organ_system": "cerebellum + immune system (thymus/B-cell) + haematopoietic (lymphoid malignancy) + vasculature (telangiectasia)",
        "primary_treatment": "IVIG (if IgG low); infection prophylaxis; cancer surveillance; avoid radiation; physiotherapy; speech therapy",
    },
    # ── SACS — ARSACS ──
    {
        "gene": "SACS",
        "protein": "Sacsin",
        "alias": (
            "SACS; OMIM gene 604490; ARSACS #270550; "
            "13q12.12; 4579 aa; ~520 kDa; largest known single-exon-encoded human protein; "
            "Sacsin maintains mitochondrial morphology and quality control via its DNAJ co-chaperone domain; "
            "LOF → mitochondrial fragmentation → energy failure → Purkinje cell + spinal neuron degeneration; "
            "SPASTIC ATAXIA: UMN signs (spasticity, hyperreflexia, extensor plantar) + cerebellar ataxia; "
            "Charlevoix-Saguenay region Quebec FOUNDER: p.Asp2521Asn (c.7563G>C) in >90% Quebec ARSACS; "
            "RETINAL MYELINATION: hypermyelination of retinal nerve-fibre layer on fundoscopy (>80%); "
            "Onset age 1-2 years; wheelchair usually by 4th-5th decade"
        ),
        "aa": "4579 aa",
        "kDa": "~520 kDa",
        "locus": "13q12.12",
        "omim_gene": 604490,
        "omim_disease": 270550,
        "inheritance": "AR — Quebec founder p.Asp2521Asn; other truncating/missense alleles worldwide",
        "gene_class": (
            "SACS encodes Sacsin, a giant scaffold protein (~520 kDa) with multiple functional domains: "
            "three Sacsin Repeat Regions (SRRs) mediating protein-protein interaction, a DNAJ domain "
            "that recruits Hsp70 for mitochondrial quality control, a UBL (ubiquitin-like) domain, "
            "and a XPCB domain. Sacsin's primary role is maintaining mitochondrial morphology by "
            "facilitating mitochondrial fission (DRP1 recruitment) and tethering to the ER. "
            "LOF → elongated, poorly functioning mitochondria → bioenergetic failure in Purkinje cells."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("Homozygous Quebec founder p.Asp2521Asn — classic ARSACS (Quebec ancestry)", 0.45),
            ("Compound heterozygote truncating + missense (non-Quebec ancestry)", 0.35),
            ("Homozygous truncating (non-Quebec founder)", 0.20),
        ],
        "age_onset_years_range": (1, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "spastic_ataxia":                        0.99,
            "hyperreflexia_lower_limbs":             0.95,
            "retinal_nerve_fibre_hypermyelination":  0.85,
            "peripheral_neuropathy":                 0.80,
            "dysarthria":                            0.95,
            "scoliosis":                             0.70,
            "nystagmus":                             0.75,
            "distal_muscle_wasting":                 0.60,
            "cerebellar_atrophy_mri":                0.90,
            "spinal_cord_atrophy_mri":               0.75,
        },
        "hallmarks": [
            "SPASTIC ATAXIA: UMN (hyperreflexia, spasticity, extensor plantar) + cerebellar — key distinction from FRDA",
            "RETINAL MYELINATION: yellowish hypermyelination of NFL on fundoscopy — pathognomonic (>80%)",
            "QUEBEC FOUNDER: p.Asp2521Asn — test SACS first in Charlevoix-Saguenay or French-Canadian ancestry",
            "EARLY ONSET: gait instability age 1-2y; walking impaired but preserved until 4th-5th decade",
            "DEMYELINATING NEUROPATHY: NCS shows slowed conduction velocity (unlike most recessive ataxias = axonal)",
            "MRI: superior cerebellar peduncle atrophy (linear pontine T2 hypointensity) + spinal atrophy",
            "MITOCHONDRIAL DYSFUNCTION: muscle biopsy may show complex I/III reduction; SACS LOF primary aetiology",
            "LARGEST SINGLE-GENE: SACS gene spans 220 kb with one major coding exon — targeted sequencing essential",
        ],
        "treatment_alerts": [
            "OPHTHALMOSCOPY: retinal NFL hypermyelination — diagnose ARSACS at eye exam if spastic ataxia present",
            "SPASTICITY MANAGEMENT: baclofen (oral or intrathecal); physiotherapy for stretching; avoid over-sedation",
            "SCOLIOSIS: spinal X-ray from diagnosis; orthopaedic follow-up; bracing if Cobb angle >25°",
            "QUEBEC PANEL: p.Asp2521Asn targeted assay before full gene sequencing in Quebec/French-Canadian",
            "NO DISEASE-MODIFYING THERAPY: mitochondrial supplements (CoQ10, riboflavin) anecdotally tried; no evidence",
        ],
        "organ_system": "cerebellum + spinal cord (corticospinal/spinocerebellar) + peripheral nerve + retina",
        "primary_treatment": "Spasticity management (baclofen); physiotherapy; ophthalmoscopy surveillance; scoliosis monitoring",
    },
    # ── ANO10 — SCAR10 ──
    {
        "gene": "ANO10",
        "protein": "Anoctamin 10 (TMEM16K)",
        "alias": (
            "ANO10; OMIM gene 613726; SCAR10 #613728; "
            "3p22.3; 660 aa; ~73 kDa Ca2+-activated phospholipid scramblase / chloride channel; "
            "Anoctamin-10 belongs to the TMEM16 family — mediates Ca2+-activated phospholipid scrambling; "
            "Highly expressed in cerebellar Purkinje cells — LOF → progressive Purkinje cell dysfunction; "
            "SLOWLY PROGRESSIVE adult-onset cerebellar ataxia (onset 2nd-4th decade, median age 25y); "
            "Rare condition: published cases predominantly European + Middle Eastern ancestry; "
            "MRI: cerebellar atrophy (vermis > hemispheres); no supratentorial abnormality; "
            "Cognitive decline and epilepsy in 10-20% of patients"
        ),
        "aa": "660 aa",
        "kDa": "~73 kDa",
        "locus": "3p22.3",
        "omim_gene": 613726,
        "omim_disease": 613728,
        "inheritance": "AR — biallelic LOF; adult onset distinguishes ANO10 from most other recessive ataxias",
        "gene_class": (
            "ANO10 (Anoctamin 10 / TMEM16K) is a Ca2+-activated ion channel and phospholipid scramblase "
            "belonging to the TMEM16 family. It is highly expressed in the cerebellum, particularly in "
            "Purkinje cells. Phospholipid scramblase activity of ANO10 regulates plasma membrane "
            "phosphatidylserine exposure during Ca2+ signalling. LOF → impaired Purkinje cell membrane "
            "homeostasis and Ca2+ signalling → progressive cerebellar degeneration. "
            "The endoplasmic reticulum-resident function may also play a role in IP3R-mediated Ca2+ release."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("Missense (transmembrane domain or ion channel pore) → LOF", 0.55),
            ("Truncating (nonsense/frameshift) → haploinsufficiency/NMD", 0.35),
            ("Homozygous deep intronic / large deletion (MLPA)", 0.10),
        ],
        "age_onset_years_range": (15, 45),
        "sex_ratio_M": 0.50,
        "rates": {
            "slowly_progressive_ataxia":           0.99,
            "cerebellar_atrophy_mri":              0.90,
            "dysarthria":                          0.75,
            "nystagmus":                           0.65,
            "epilepsy":                            0.15,
            "cognitive_decline":                   0.20,
            "peripheral_neuropathy_mild":          0.35,
            "absent_lower_limb_reflexes":          0.45,
            "retained_ambulation_at_10y_followup": 0.70,
        },
        "hallmarks": [
            "ADULT ONSET: 2nd-4th decade distinguishes ANO10 from most recessive ataxias (FRDA, APTX, SETX)",
            "SLOWLY PROGRESSIVE: majority ambulatory at 10 years follow-up — important for prognosis counselling",
            "PURE CEREBELLAR: no telangiectasia, no oculomotor apraxia, no cardiomyopathy, no immunodeficiency",
            "EPILEPSY 15%: EEG if seizures; levetiracetam or valproate depending on seizure type",
            "COGNITIVE DECLINE 20%: neuropsychological testing annually if concern; occupational therapy",
            "MRI: vermis atrophy early; hemisphere atrophy later; no brainstem lesion",
            "DIAGNOSIS BY EXCLUSION: after ruling out FRDA (AFP/FXN repeat), AOA2 (AFP), A-T (AFP/telangiectasia)",
            "ANCESTRY: predominantly European (Scandinavian, Dutch, French) and Middle Eastern (Lebanese, Iraqi)",
        ],
        "treatment_alerts": [
            "NO DISEASE-MODIFYING THERAPY: physiotherapy; OT; gait aids; fall prevention",
            "SLOWLY PROGRESSIVE: do not overestimate disability at diagnosis; prognosis better than FRDA/A-T",
            "EPILEPSY: standard AED management; levetiracetam preferred (no enzyme induction)",
            "DRIVING ASSESSMENT: refer to occupational therapist / driving centre when ataxia progresses",
            "GENETIC COUNSELLING: 25% recurrence per pregnancy; prenatal diagnosis available",
        ],
        "organ_system": "cerebellum (Purkinje cells); peripheral nerve (mild); brain (epilepsy/cognition in subset)",
        "primary_treatment": "Supportive — physiotherapy, OT, gait aids; AED if epilepsy; annual neurological review",
    },
    # ── ADCK3/COQ8A — ARCA2 ──
    {
        "gene": "ADCK3",
        "protein": "AarF Domain Containing Kinase 3 (COQ8A)",
        "alias": (
            "ADCK3; also COQ8A; OMIM gene 612399; ARCA2 #612936; "
            "1q42.13; 669 aa; ~76 kDa atypical kinase in Coenzyme Q biosynthesis complex; "
            "COQ8A phosphorylates and activates other CoQ biosynthesis enzymes — essential regulatory kinase; "
            "LOF → reduced CoQ10 in muscle, brain, and fibroblasts; "
            "COENZYME Q10 SUPPLEMENTATION: 5-10 mg/kg/day in 2 divided doses — ~20-40% respond meaningfully; "
            "ELEVATED CK + EXERCISE INTOLERANCE: mitochondrial clue in any young-onset ataxia; "
            "Cerebellar atrophy on MRI; seizures in 30-40%; onset childhood-adolescence; "
            "Muscle biopsy: reduced CoQ10 content + respiratory chain deficiency (CI+III combined most common)"
        ),
        "aa": "669 aa",
        "kDa": "~76 kDa",
        "locus": "1q42.13",
        "omim_gene": 612399,
        "omim_disease": 612936,
        "inheritance": "AR — biallelic LOF; variable CoQ10 deficiency severity",
        "gene_class": (
            "ADCK3 (also named COQ8A) encodes an atypical kinase that is a structural and functional "
            "component of the mitochondrial CoQ biosynthesis complex ('CoQ synthome'). Unlike canonical "
            "protein kinases, ADCK3 may phosphorylate small-molecule intermediates in the CoQ10 "
            "biosynthesis pathway rather than proteins. LOF impairs the entire CoQ biosynthesis complex, "
            "reducing CoQ10 levels in mitochondria. CoQ10 is essential for electron shuttling in the "
            "respiratory chain (Complex I/II → CoQ → Complex III) and as a lipid antioxidant in "
            "mitochondrial membranes. Reduced CoQ10 causes combined respiratory chain deficiency, "
            "increased ROS, and neurodegeneration (particularly Purkinje cells)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("Missense (kinase domain or AarF domain disruption)", 0.50),
            ("Truncating (nonsense/frameshift → null allele)", 0.35),
            ("Compound heterozygote missense + truncating", 0.15),
        ],
        "age_onset_years_range": (3, 25),
        "sex_ratio_M": 0.50,
        "rates": {
            "progressive_cerebellar_ataxia":       0.99,
            "elevated_creatine_kinase":            0.65,
            "exercise_intolerance":                0.70,
            "cerebellar_atrophy_mri":              0.90,
            "seizures":                            0.35,
            "coq10_deficiency_muscle_biopsy":      0.80,
            "fatigue":                             0.75,
            "dysarthria":                          0.80,
            "peripheral_neuropathy_mild":          0.35,
            "coq10_supplementation_response":      0.30,
        },
        "hallmarks": [
            "COQ10 DEFICIENCY: muscle CoQ10 measurement mandatory — diagnosis AND treatment eligibility",
            "ELEVATED CK: 65% — mitochondrial aetiology clue in ataxia (not expected in FRDA/AOA)",
            "EXERCISE INTOLERANCE: fatigue disproportionate to weakness — key mitochondrial symptom",
            "COQ10 SUPPLEMENTATION: 5-10 mg/kg/day in 2 doses; trial minimum 6 months; ~20-40% respond",
            "SEIZURES 35%: EEG at diagnosis; levetiracetam preferred; valproate CAUTION (mitochondrial toxicity)",
            "MUSCLE BIOPSY: reduced CoQ10 content + combined Complex I+III deficiency; ragged-red fibres rare",
            "FIBROBLAST CoQ10: if muscle biopsy not feasible, skin fibroblast CoQ10 assay diagnostic",
            "MITOCHONDRIAL SUPPLEMENTS: riboflavin, carnitine adjuncts considered alongside CoQ10",
        ],
        "treatment_alerts": [
            "COQ10 TRIAL: 5-10 mg/kg/day (max 600 mg/day adult) — MANDATORY before declaring no treatment available",
            "VALPROATE CAUTION: mitochondrial toxicity risk; use levetiracetam or lamotrigine for seizures",
            "COENZYME Q10 ASSAY: request specifically in muscle (not serum) — serum CoQ10 unreliable for deficiency",
            "6-MONTH TRIAL MINIMUM: CoQ10 response may be delayed; assess ataxia, fatigue, CK at 3 and 6 months",
            "MITOCHONDRIAL GENETIC PANEL: if ADCK3 biallelic LOF confirmed, check family for carrier status",
        ],
        "organ_system": "cerebellum + muscle (mitochondrial myopathy) + peripheral nerve + brain (seizures)",
        "primary_treatment": "CoQ10 5-10 mg/kg/day (mandatory trial); AED if seizures (avoid valproate); physio; mitochondrial supplements",
    },
    # ── ABHD12 — PHARC ──
    {
        "gene": "ABHD12",
        "protein": "Abhydrolase Domain Containing 12",
        "alias": (
            "ABHD12; OMIM gene 613599; PHARC #612674; "
            "20p11.21; 398 aa; ~45 kDa serine hydrolase (endocannabinoid system); "
            "ABHD12 hydrolyses 2-arachidonoylglycerol (2-AG) — the most abundant endocannabinoid in brain; "
            "Also degrades lyso-phosphatidylserine (LPS) in microglia — PS signalling for apoptotic cell clearance; "
            "LOF → 2-AG accumulation + LPS accumulation → chronic microglial activation → demyelination; "
            "PHARC: Polyneuropathy, Hearing loss, Ataxia, Retinitis Pigmentosa, Cataract; "
            "SEQUENCE OF ONSET: neuropathy + SNHL (teenage) → ataxia → RP → cataract (2nd-3rd decade); "
            "Iran / Palestine FOUNDER: c.846+1G>T splice-site + 20p11.21 large deletion in Middle Eastern ancestry"
        ),
        "aa": "398 aa",
        "kDa": "~45 kDa",
        "locus": "20p11.21",
        "omim_gene": 613599,
        "omim_disease": 612674,
        "inheritance": "AR — biallelic LOF; Iranian/Palestinian founder alleles; rare worldwide",
        "gene_class": (
            "ABHD12 (Abhydrolase Domain Containing 12) is a serine hydrolase that degrades two substrates: "
            "(1) 2-arachidonoylglycerol (2-AG), the most abundant endocannabinoid and a CB1/CB2 agonist; "
            "(2) lyso-phosphatidylserine (LPS), a lipid mediator that signals apoptotic cell recognition to microglia. "
            "LOF → 2-AG accumulation triggers CB2-dependent and CB2-independent chronic microglial activation, "
            "producing a demyelinating polyneuropathy and progressive multisystem degeneration. "
            "LPS accumulation also impairs apoptotic-cell engulfment (efferocytosis) → further inflammatory signalling. "
            "The PHARC acronym captures all affected organ systems."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("Iranian/Palestinian founder allele (c.846+1G>T splice or 20p11.21 deletion)", 0.45),
            ("Frameshift/nonsense → null allele (non-founder)", 0.35),
            ("Missense (hydrolase catalytic triad disruption)", 0.20),
        ],
        "age_onset_years_range": (10, 30),
        "sex_ratio_M": 0.48,
        "rates": {
            "polyneuropathy":                      0.98,
            "sensorineural_hearing_loss":          0.95,
            "cerebellar_ataxia":                   0.90,
            "retinitis_pigmentosa":                0.85,
            "cataract":                            0.70,
            "absent_lower_limb_reflexes":          0.95,
            "night_blindness_early_rp":            0.85,
            "vestibular_dysfunction":              0.50,
            "dysarthria":                          0.55,
            "cerebellar_atrophy_mri":              0.75,
        },
        "hallmarks": [
            "PHARC ACRONYM: Polyneuropathy + Hearing loss + Ataxia + RP + Cataract — all 5 in full syndrome",
            "NEUROPATHY + SNHL FIRST: teenage onset; audiogram + NCS/EMG the first investigations",
            "AUDIOGRAM MANDATORY: SNHL in all PHARC patients — cochlear implants effective for SNHL",
            "RETINITIS PIGMENTOSA: night blindness + visual field constriction; ERG reduced; ophthalmology annually",
            "FOUNDER POPULATIONS: Iran/Palestine — test ABHD12 first in Middle Eastern ataxia + neuropathy + SNHL",
            "ENDOCANNABINOID MECHANISM: 2-AG accumulation → microglial activation → demyelination (no 2-AG drug target yet)",
            "CATARACT 70%: slit-lamp annually from diagnosis; early extraction if visual acuity affected",
            "NO DISEASE-MODIFYING THERAPY: manage each component (hearing aid/CI; RP rehabilitation; physio; glasses)",
        ],
        "treatment_alerts": [
            "AUDIOGRAM FIRST: SNHL often predates ataxia — hearing aids from diagnosis; cochlear implant if profound",
            "OPHTHALMOLOGY ANNUALLY: ERG + VF + OCT for RP monitoring; cataract extraction if VA <6/18",
            "COCHLEAR IMPLANT: effective in PHARC SNHL — consider early before auditory cortex deprivation",
            "NEUROPATHY MANAGEMENT: neuropathic pain (duloxetine/gabapentin); foot care; orthotics for foot drop",
            "FOUNDER SCREENING: if Iranian/Palestinian heritage + ataxia + SNHL + neuropathy → ABHD12 first",
        ],
        "organ_system": "peripheral nerve + inner ear + cerebellum + retina + lens (cataract)",
        "primary_treatment": "Hearing aids / cochlear implant; ophthalmology (ERG/VF/cataract); neuropathy management; physio; no disease-modifying therapy",
    },
]


def _make_patients(gene_data: dict) -> list[dict]:
    rng = random.Random(gene_data["seed"])
    gene = gene_data["gene"]
    rates = gene_data["rates"]
    ptx = []

    for pid in range(1, gene_data["n_patients"] + 1):
        age_lo, age_hi = gene_data["age_onset_years_range"]
        sex = "M" if rng.random() < gene_data["sex_ratio_M"] else "F"
        onset_age = rng.randint(age_lo, age_hi)

        etio_choices = [e[0] for e in gene_data["etiologies"]]
        etio_probs = [e[1] for e in gene_data["etiologies"]]
        etiology = rng.choices(etio_choices, weights=etio_probs, k=1)[0]

        clinical = {k: (rng.random() < v) for k, v in rates.items()}

        ptx.append({
            "id": f"{gene}-{pid:03d}",
            "gene": gene,
            "sex": sex,
            "onset_age_years": onset_age,
            "etiology": etiology,
            **clinical,
        })
    return ptx


def _aggregate(patients: list[dict], rate_keys: list[str]) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    return {k: round(sum(1 for p in patients if p.get(k, False)) / n * 100, 1)
            for k in rate_keys}


ALL_PATIENTS = []
for _gd in ATAXIA_GENES:
    ALL_PATIENTS.extend(_make_patients(_gd))


def get_overview() -> dict:
    n = len(ALL_PATIENTS)
    rate_keys = list(ATAXIA_GENES[0]["rates"].keys())
    agg = _aggregate(ALL_PATIENTS, rate_keys)
    return {
        "atlas": "Hereditary Ataxia Atlas",
        "subtitle": "Complete 8-Gene Hereditary Recessive Ataxia Reference",
        "total_patients": n,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes": [g["gene"] for g in ATAXIA_GENES],
        "diseases": {
            "FXN":   "Friedreich Ataxia (FRDA) — GAA Repeat Expansion; Standard Panels MISS; Cardiomyopathy 80%; Omaveloxolone",
            "APTX":  "AOA1 — Oculomotor Apraxia; Hypoalbuminaemia; Hypercholesterolaemia; Japan/Portugal Founder",
            "SETX":  "AOA2 — AFP >10 mcg/L Hallmark; No Telangiectasia; Slow Progression; European/N.African",
            "ATM":   "Ataxia-Telangiectasia — IgA Deficiency; Cancer 35%; Radiosensitivity; Carrier Breast Risk",
            "SACS":  "ARSACS — Spastic Ataxia; Retinal NFL Hypermyelination; Quebec Founder p.Asp2521Asn",
            "ANO10": "SCAR10 — Adult Onset 25y; Slowly Progressive; Pure Cerebellar; Vermis Atrophy",
            "ADCK3": "ARCA2 — CoQ10 Deficiency; Elevated CK; Exercise Intolerance; CoQ10 Supplementation Trial",
            "ABHD12": "PHARC — Polyneuropathy + SNHL + Ataxia + RP + Cataract; Endocannabinoid; Iran/Palestine Founder",
        },
        "aggregate_stats": agg,
        "top_alerts": [
            "FXN: GAA REPEAT EXPANSION — standard exome/WES MISSES; request specific FXN repeat PCR/Southern blot",
            "FXN: CARDIOMYOPATHY 80% — ECG + echocardiogram MANDATORY at diagnosis; annual review",
            "ATM: RADIOSENSITIVITY — use MRI over CT; minimise ionising radiation in A-T patients",
            "ATM: CARRIER PARENTS (heterozygous ATM) — 2× breast cancer risk; mammography from age 40",
            "ADCK3: CoQ10 SUPPLEMENTATION TRIAL MANDATORY — 5-10 mg/kg/day minimum 6 months before declaring failure",
            "SETX: AFP >10 mcg/L — check serum AFP first in any recessive ataxia (cheap, fast, specific for AOA2+A-T)",
            "SACS: RETINAL NFL HYPERMYELINATION — fundoscopy diagnosis; Quebec founder p.Asp2521Asn test first",
            "ABHD12: AUDIOGRAM FIRST — SNHL precedes ataxia in PHARC; cochlear implant effective",
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for gd in ATAXIA_GENES:
        pts = [p for p in ALL_PATIENTS if p["gene"] == gd["gene"]]
        rate_keys = list(gd["rates"].keys())
        agg = _aggregate(pts, rate_keys)
        result[gd["gene"]] = {
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "organ_system": gd["organ_system"],
            "primary_treatment": gd["primary_treatment"],
            "stats": agg,
            "hallmarks": gd["hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "etiology_distribution": [
                {"etiology": e[0], "fraction": e[1]} for e in gd["etiologies"]
            ],
        }
    return result


def get_definitions() -> dict:
    return {
        "atlas": "Hereditary Ataxia Atlas",
        "classification": {
            "DNA_Repair_Ataxias": {
                "FRDA_FXN": "Friedreich Ataxia — frataxin; Fe-S cluster biogenesis; GAA repeat expansion",
                "AOA1_APTX": "AOA type 1 — aprataxin; abortive DNA ligation repair",
                "AOA2_SETX": "AOA type 2 — senataxin; R-loop resolution during transcription",
                "AT_ATM":   "Ataxia-Telangiectasia — ATM kinase; DSB sensing; combined immunodeficiency + cancer",
            },
            "Structural_Mitochondrial_Ataxias": {
                "ARSACS_SACS": "ARSACS — sacsin; mitochondrial morphology + spastic cerebellar ataxia; Quebec founder",
                "ARCA2_ADCK3": "ARCA2 / CoQ10 deficiency — ADCK3/COQ8A; CoQ biosynthesis regulatory kinase",
            },
            "Ion_Channel_Lipid_Ataxias": {
                "SCAR10_ANO10": "SCAR10 — anoctamin-10; Ca2+-activated Cl- channel; adult-onset slowly progressive",
                "PHARC_ABHD12": "PHARC — ABHD12; endocannabinoid (2-AG) + lyso-PS hydrolase; Iran/Palestine founder",
            },
        },
        "key_diagnostic_rules": {
            "FXN_REPEAT_ASSAY_MANDATORY": (
                "Friedreich Ataxia CANNOT be excluded by standard exome/WES/gene panel — "
                "GAA trinucleotide repeat expansion (intron 1) is invisible to short-read sequencing. "
                "Request FXN-specific repeat PCR (normal ≤33; pathogenic ≥66 GAA repeats) for any "
                "recessive ataxia with onset <25y + sensory loss + areflexia + cardiomyopathy."
            ),
            "AFP_FIRST_IN_RECESSIVE_ATAXIA": (
                "Serum AFP >10 mcg/L narrows the differential to AOA2 (SETX) or Ataxia-Telangiectasia (ATM). "
                "AFP is elevated in both conditions — distinguish by: telangiectasia (ATM only), "
                "IgA deficiency (ATM), radiosensitivity (ATM). AFP is cheap, fast, and highly sensitive."
            ),
            "ATM_RADIOSENSITIVITY": (
                "ATM-null cells cannot properly repair DSBs induced by ionising radiation. "
                "Even standard diagnostic X-rays incrementally increase genomic instability. "
                "Prefer MRI over CT whenever feasible in confirmed A-T patients; if CT essential, "
                "document indication and minimise dose."
            ),
            "ATM_CARRIER_BREAST_CANCER": (
                "Heterozygous ATM carriers (parents, siblings of A-T probands) have approximately "
                "2-fold increased lifetime breast cancer risk. Counsel carrier parents to commence "
                "breast cancer screening (mammography ± MRI) from age 40, or per local guidelines."
            ),
            "ADCK3_COQ10_TRIAL": (
                "Any patient with biallelic ADCK3 LOF MUST undergo a CoQ10 supplementation trial: "
                "5-10 mg/kg/day (split into 2 doses, with fatty meal for absorption), minimum 6 months. "
                "Assess at 3 and 6 months: ataxia rating scale, fatigue, CK, exercise tolerance. "
                "~20-40% show meaningful benefit. Ubiquinol formulation (reduced CoQ10) may be superior."
            ),
            "SACS_RETINAL_HYPERMYELINATION": (
                "ARSACS has a near-pathognomonic fundoscopic sign: hypermyelination of retinal nerve-fibre "
                "layer, visible as yellowish opaque streaks extending from the optic disc. Present in >80%. "
                "If found in combination with spastic ataxia, test SACS before broader panel."
            ),
            "ABHD12_PHARC_SEQUENCE": (
                "PHARC manifests in a characteristic temporal sequence: "
                "(1) Polyneuropathy + SNHL in teenage years (often misdiagnosed as CMT + age-related hearing loss); "
                "(2) Cerebellar ataxia in 2nd-3rd decade; "
                "(3) Retinitis Pigmentosa (night blindness → visual field loss); "
                "(4) Cataract. "
                "Test ABHD12 in any Middle Eastern patient with neuropathy + SNHL + ataxia."
            ),
            "CASCADE_TESTING": (
                "All 8 genes: biallelic AR inheritance — 25% risk per pregnancy of affected sib; "
                "50% carrier risk per sib. Offer gene-specific carrier testing to parents + siblings. "
                "Prenatal testing (CVS/amnio) and PGT-M available for all 8 conditions."
            ),
        },
        "treatment_hierarchy": {
            "FXN_FRDA": [
                "1. Omaveloxolone (Skyclarys) 150 mg/day — FDA-approved ≥16y; Nrf2 activator; monitor LFTs",
                "2. Idebenone 5 mg/kg/day — cardiac benefit; start early; adjunct to omaveloxolone",
                "3. Cardiology: annual echo + ECG; beta-blocker or ACEi if cardiomyopathy symptomatic",
                "4. Scoliosis: spinal X-ray annually; surgical fusion if Cobb angle >40°",
                "5. Diabetes: annual HbA1c; insulin if required; avoid metformin (mitochondrial concern)",
            ],
            "APTX_AOA1": [
                "1. No disease-modifying therapy; physiotherapy + OT + speech therapy",
                "2. Cholesterol: statin if significantly elevated (check drug interactions)",
                "3. Albumin: nutritional optimisation; dietitian input if hypoalbuminaemic",
                "4. Chorea: low-dose clonazepam if troublesome; avoid antipsychotics long-term",
                "5. Oculomotor apraxia: low vision aids; vestibular rehabilitation",
            ],
            "ATM_AT": [
                "1. IVIG/SCIG if IgG low + recurrent infections (target trough IgG >7 g/L)",
                "2. Cancer surveillance: annual lymph node check + CBC; low threshold for biopsy",
                "3. Avoid ionising radiation; MRI preferred for all imaging where possible",
                "4. No live vaccines (MMR, varicella) — T-cell deficiency",
                "5. Carrier parents: breast cancer screening (mammography ± MRI) from age 40",
            ],
            "ADCK3_ARCA2": [
                "1. CoQ10 (ubiquinol preferred) 5-10 mg/kg/day in 2 doses with fat-containing meal",
                "2. Seizures: levetiracetam (avoid valproate — mitochondrial toxicity)",
                "3. Riboflavin 100-200 mg/day adjunct (cofactor for Complex I/II)",
                "4. Physiotherapy + occupational therapy; fatigue management",
                "5. Muscle biopsy CoQ10 assay to confirm deficiency + quantify pre/post-treatment",
            ],
            "ABHD12_PHARC": [
                "1. SNHL: hearing aids early; cochlear implant if profound loss (effective in PHARC)",
                "2. RP: ophthalmology annually (ERG + VF + OCT); low-vision rehabilitation",
                "3. Cataract: slit-lamp annually; extraction when VA <6/18",
                "4. Neuropathy: duloxetine/gabapentin for pain; orthotics for foot drop",
                "5. Ataxia: physiotherapy; gait aids; fall prevention programme",
            ],
        },
        "genes_summary": {g["gene"]: g["alias"] for g in ATAXIA_GENES},
    }


# ── Convenience wrappers (called by api_backend.py) ──
def overview():
    return get_overview()


def breakdown():
    return get_breakdown()


def definitions():
    return get_definitions()
