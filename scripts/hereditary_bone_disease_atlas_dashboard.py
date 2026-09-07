#!/usr/bin/env python3
"""Hereditary-Metabolic-Bone-Disease-Atlas — Complete 8-Gene Hereditary Bone Disease Atlas
COL1A1  (collagen type I alpha-1; 1454 aa; 17q21.33; AD;
         Osteogenesis Imperfecta type 1/4 — haploinsufficiency; most common OI;
         blue sclerae pathognomonic; bisphosphonates first-line; hearing loss;
         dentinogenesis imperfecta; ambulatory prognosis; seed SEED_BASE+0) ·
COL1A2  (collagen type I alpha-2; 1366 aa; 7q21.3; AD;
         Osteogenesis Imperfecta type 3/4 — dominant-negative; most severe non-lethal OI;
         in-utero fractures; progressive deformity; Fassier-Duval intramedullary rodding;
         wheelchair-dependent majority; seed SEED_BASE+1) ·
ALPL    (alkaline phosphatase tissue-nonspecific isozyme; 524 aa; 1p36.12; AD/AR;
         Hypophosphatasia — PPi accumulation → impaired mineralisation;
         premature loss of deciduous teeth <5 y PATHOGNOMONIC;
         B6-responsive neonatal seizures; asfotase alfa FDA 2015 ERT;
         seed SEED_BASE+2) ·
PHEX    (phosphate-regulating endopeptidase homolog X-linked; 749 aa; Xp22.11; XLR;
         X-linked hypophosphatemia — most common hereditary rickets;
         dental abscesses 75% (without caries) PATHOGNOMONIC;
         burosumab (anti-FGF23) FDA 2018; enthesopathy adults;
         seed SEED_BASE+3) ·
FGF23   (fibroblast growth factor 23; 251 aa; 12p13.32; AD;
         Autosomal dominant hypophosphatemic rickets — GOF prevents cleavage;
         iron deficiency triggers disease flares — KEY DDx vs XLH;
         burosumab effective; elevated intact FGF23 distinguishes from nutritional;
         seed SEED_BASE+4) ·
ACVR1   (activin A receptor type 1 / ALK2; 509 aa; 2q24.1; AD;
         Fibrodysplasia ossificans progressiva — R206H in 97%;
         NO biopsies, NO IM injections, NO surgery (catastrophic HO flares);
         palovarotene FDA 2023; garetosmab Phase 2; progressive soft tissue ossification;
         seed SEED_BASE+5) ·
RUNX2   (runt-related transcription factor 2; 521 aa; 6p21.1; AD;
         Cleidocranial dysplasia — absent/hypoplastic clavicles PATHOGNOMONIC;
         supernumerary teeth → dental surgery mandatory; patent fontanelle;
         intelligence NORMAL; haploinsufficiency; seed SEED_BASE+6) ·
LRP5    (LDL receptor-related protein 5; 1615 aa; 11q13.2; AD-GOF / AR-LOF;
         Osteoporosis-pseudoglioma (OPPG) — AR biallelic LOF: blindness + severe osteoporosis;
         High bone mass (HBM) — AD GOF G171V: very high bone density, benign;
         WNT/β-catenin pathway; romosozumab mechanism; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1750–1757)
"""

import random

SEED_BASE = 1750

BONE_DISEASE_GENES = [
    # ── COL1A1 — Osteogenesis Imperfecta type 1/4 ────────────────────────────
    {
        "gene": "COL1A1",
        "protein": (
            "COL1A1 — 17q21.33 AD — Collagen-Type-I-Alpha-1-1454aa — "
            "Osteogenesis-Imperfecta-Type-1-Haploinsufficiency — "
            "Blue-Sclerae-PATHOGNOMONIC — Bisphosphonates-First-Line — "
            "Hearing-Loss-40pct-Adults — Dentinogenesis-Imperfecta-30pct — "
            "Ambulatory-Prognosis-Good"
        ),
        "alias": (
            "COL1A1 (collagen type I alpha-1 chain); OMIM gene 120150; "
            "Osteogenesis Imperfecta type 1 OMIM 166200 (and contributes to OI type 4 619795). "
            "17q21.33; 1454 aa; ~139 kDa (pro-alpha 1(I) chain); autosomal dominant. "
            "FUNCTION: COL1A1 encodes the alpha-1 chain of type I procollagen. "
            "Two COL1A1 chains + one COL1A2 chain form the fibrillar type I collagen heterotrimer, "
            "the principal structural protein of bone, tendons, skin, and sclera. "
            "In OI type 1, a COL1A1 null allele (frameshift/nonsense → NMD) produces "
            "haploinsufficiency: only 50% normal collagen I is made, but its quality is intact. "
            "MUTATION SPECTRUM: predominantly null alleles (frameshift, nonsense, splice-disrupting "
            "→ mRNA instability by NMD); rare promoter mutations; occasionally exon-skipping; "
            "missense mutations in COL1A1 tend to cause more severe phenotypes (type 2/3/4). "
            "CLINICAL FEATURES: "
            "Blue/grey sclerae (thinner sclera transmits choroidal pigment) — present virtually 100% "
            "in childhood, may fade with age; PATHOGNOMONIC in appropriate clinical context; "
            "Fracture history: variable, typically present from toddling through early adulthood; "
            "may diminish significantly after puberty (especially females — worsens post-menopause); "
            "Short stature: mild (within ~2SD of normal); NOT typically severely stunted as in OI type 3; "
            "Dentinogenesis imperfecta (DI): ~30–40% of COL1A1-OI; "
            "teeth: yellow-brown, translucent, chip and fracture easily; affects primary and permanent; "
            "Hearing loss: sensorineural + conductive (stapes ankylosis); "
            "onset typically 20–40 y; 40–50% of adults affected; check annually from age 20; "
            "Hyperlaxity of joints: mild in OI type 1; "
            "BONE MINERAL DENSITY: ↓ BMD (DXA Z-score typically -1 to -2.5); "
            "Wormian bones on skull X-ray: not pathognomonic but suggestive; "
            "FRACTURE HISTORY: any bone; long bones (femur, humerus, tibia), vertebral compression; "
            "rarely rib fractures spontaneously (more in OI type 3); "
            "DIAGNOSIS: COL1A1 molecular genetic testing (gene panel or trio WES/WGS); "
            "skin biopsy collagen biochemistry: ↓ collagen type I quantity (normal migration on SDS-PAGE); "
            "clinical criteria (Sillence classification OI type 1); "
            "MANAGEMENT: "
            "Bisphosphonates (IV pamidronate OR zoledronic acid OR oral alendronate/risedronate): "
            "standard of care — ↑ BMD, ↓ fracture rate, improve vertebral collapse; "
            "pamidronate IV cycles 3 days every 4–6 months in children (standard paediatric protocol); "
            "zoledronic acid IV once yearly (adults and adolescents — preferred for adherence); "
            "Denosumab: RANK-L inhibitor; option in patients intolerant to bisphosphonates; "
            "rebound hypercalcaemia on discontinuation — MUST NOT ABRUPTLY STOP without bridging; "
            "Romosozumab (anti-sclerostin): anabolic, under investigation in OI; "
            "Physiotherapy: muscle strengthening, fall prevention; "
            "Genetic counselling: 50% transmission risk; PGT-A available; "
            "Hearing aids when indicated; annual audiogram from age 20; "
            "Dental referral: caries prevention, composite/crown restoration in DI."
        ),
        "locus": "17q21.33",
        "aa": 1454,
        "kDa": 139,
        "omim_gene": "120150",
        "omim_disease": "166200",
        "inheritance": "AD; haploinsufficiency (null alleles); de novo ~25–30% familial OI1",
        "gene_class": "Fibrillar Collagen Type I Alpha-1 — Bone ECM Structural Component",
        "key_alerts": [
            "COL1A1-BLUE-SCLERAE-PATHOGNOMONIC: blue/grey sclerae in OI type 1 (thin sclera → choroidal pigment transmitted) are virtually universal in childhood; distinguish from benign blue sclerae (no fractures, no dentinogenesis imperfecta, normal DEXA) — if fractures + blue sclerae: COL1A1/2 panel IMMEDIATELY",
            "COL1A1-BISPHOSPHONATES-PAMIDRONATE-STANDARD-PAEDIATRIC: IV pamidronate (3 days, 1 mg/kg/day, cycles every 3–4 months in severe / 6 months in mild OI) is the paediatric gold standard — do NOT substitute with oral bisphosphonates in young children with active fractures; assess by DXA Z-score response",
            "COL1A1-DENOSUMAB-REBOUND: if denosumab is used (RANK-L inhibitor), NEVER abruptly discontinue — rebound hypercalcaemia and vertebral fracture surge documented; transition to bisphosphonate before stopping; coordinate with metabolic bone specialist",
            "COL1A1-HEARING-ANNUAL-AUDIOGRAM-FROM-20: conductive + sensorineural hearing loss develops in 40–50% of adult OI type 1 patients; begin annual audiogram at age 20; stapes surgery for conductive component; hearing aids when indicated; do NOT wait for symptom complaint",
        ],
        "etiologies": {
            "OI_Type_1_Classic": {"pct": 70, "phenotype": "null allele haploinsufficiency — mild, ambulatory, blue sclerae, fractures decrease post-puberty"},
            "OI_Type_4_COL1A1": {"pct": 20, "phenotype": "missense in non-collagenous domain — moderate, variable sclerae, DXA -2 to -3"},
            "DI_Positive_OI1": {"pct": 30, "phenotype": "dentinogenesis imperfecta — yellow-brown teeth, chipping, ↑ caries"},
            "Sporadic_De_Novo": {"pct": 28, "phenotype": "no family history — de novo COL1A1 null variant; one-parent mosaic possible"},
        },
        "stats": {
            "mean_onset_age_y": 2.1,
            "mean_dx_delay_months": 8.5,
            "fracture_reduction_biphosphonate_pct": 38,
            "adult_hearing_loss_pct": 44,
        },
        "dx_delay_distribution": {"<6mo": 42, "6-24mo": 38, ">24mo": 20},
        "patients": [],
    },
    # ── COL1A2 — Osteogenesis Imperfecta type 3 ──────────────────────────────
    {
        "gene": "COL1A2",
        "protein": (
            "COL1A2 — 7q21.3 AD — Collagen-Type-I-Alpha-2-1366aa — "
            "Osteogenesis-Imperfecta-Type-3-Dominant-Negative-Severe — "
            "In-Utero-Fractures-PATHOGNOMONIC-Severe — "
            "Fassier-Duval-Intramedullary-Rod-Mandatory — "
            "Wheelchair-Dependent-Majority — "
            "Bisphosphonates-IV-Pamidronate-Cycling"
        ),
        "alias": (
            "COL1A2 (collagen type I alpha-2 chain); OMIM gene 120160; "
            "Osteogenesis Imperfecta type 3 OMIM 259420 (most severe non-lethal OI). "
            "7q21.3; 1366 aa; ~129 kDa (pro-alpha 2(I) chain); predominantly autosomal dominant "
            "(dominant-negative glycine substitutions); rare AR forms (Bruck syndrome overlap). "
            "FUNCTION: COL1A2 encodes the alpha-2 chain (one per collagen I heterotrimer). "
            "Glycine (Gly-X-Y repeat) substitutions in COL1A2 cause a dominant-negative effect: "
            "the abnormal chain incorporates into collagen trimers → destabilised triple helix → "
            "↓ collagen secretion, ↑ post-translational over-modification, ↓ fibril quality. "
            "This is qualitatively different from COL1A1 haploinsufficiency: quality is impaired, "
            "not just quantity. "
            "MUTATION SPECTRUM: "
            "Glycine substitutions in the Gly-X-Y repeats of the triple helix domain — most pathogenic; "
            "severity correlates with position (α2 chain substitutions generally less severe than α1); "
            "splice mutations; occasional large deletions; "
            "exon 52 splice — special: COL1A2 exon 52 deletion → Bruck syndrome (AR) — OI + contractures; "
            "CLINICAL FEATURES — OI TYPE 3: "
            "Multiple intrauterine fractures — present at birth; bowing of all long bones at birth; "
            "CHARACTERISTIC RADIOGRAPH: 'crumpled' femora; old fractures with callus in neonate; "
            "Progressive skeletal deformity: kyphoscoliosis (often severe), thoracic cage deformity; "
            "Short stature: typically <3rd centile, often adult height <100 cm; "
            "Triangular facies: frontal bossing, triangular face; "
            "Basilar invagination: dens migrates upward → brainstem compression risk (MRI cervical mandatory); "
            "Cardiopulmonary complications: restrictive lung disease from chest wall; "
            "Progressive hearing loss similar to OI type 1 but may present younger; "
            "Blue sclerae in infancy, may normalise later; "
            "Sclerae lighter in OI type 4 (COL1A2 missense in less critical positions); "
            "Mobility: majority wheelchair-dependent in adulthood; "
            "MANAGEMENT — SPECIALISED MULTIDISCIPLINARY: "
            "IV pamidronate (standard OI type 3 protocol): 3-day cycles every 4 months in childhood; "
            "DRAMATIC effect on bone density and fracture rate; cycle scars on DXA (zebra lines) are normal; "
            "Intramedullary rodding: Fassier-Duval telescoping rods (elongate with growth); "
            "performed by OI-specialist orthopaedic surgeon; prevents deformity + improves ambulation; "
            "Timing: typically age 2–3 for femora, earlier if severe bowing impedes walking; "
            "Spinal surgery: complex; avoid unless cord compression imminent; "
            "Basilar invagination screening: annual cervical MRI from age 5; "
            "surgical decompression if symptomatic; "
            "Physiotherapy: pool therapy ideal; prone positioning in infants (pillow under chest); "
            "NON-IMPACT ACTIVITIES: swimming, cycling — impact sports prohibited; "
            "RESPIRATORY MONITORING: spirometry annually from age 6; NIV when FVC <50% predicted; "
            "GENETICS: 50% risk if parent affected; new dominant variants ~30% de novo in OI type 3."
        ),
        "locus": "7q21.3",
        "aa": 1366,
        "kDa": 129,
        "omim_gene": "120160",
        "omim_disease": "259420",
        "inheritance": "AD dominant-negative; rare AR (Bruck syndrome); de novo ~30%",
        "gene_class": "Fibrillar Collagen Type I Alpha-2 — Dominant-Negative Trimer Destabilisation",
        "key_alerts": [
            "COL1A2-IN-UTERO-FRACTURES-NEONATAL-EMERGENCY: OI type 3 presents at birth with multiple fractures and bone deformity — neonatal OI requires immediate specialist involvement; do NOT perform any procedures without OI-trained team; routine neonatal handling must be modified (fracture precautions); contact national OI centre",
            "COL1A2-BASILAR-INVAGINATION-ANNUAL-CERVICAL-MRI: basilar invagination (upward migration of dens → brainstem compression) occurs in ~25% of severe OI type 3 — MANDATORY annual cervical spine MRI from age 5; symptoms: new-onset headache, dysphagia, upper limb weakness, sleep apnoea → URGENT neurosurgical review",
            "COL1A2-FASSIER-DUVAL-RODDING-SPECIALIST: intramedullary rodding should ONLY be performed at OI-specialist orthopaedic centres; Fassier-Duval telescoping rods preferred in growing children; Bailey-Dubow rods for larger diameter; non-specialist fracture fixation can result in catastrophic complications",
            "COL1A2-RESTRICTIVE-LUNG-ANNUAL-SPIROMETRY: chest wall deformity → restrictive lung disease in OI type 3; annual spirometry from age 6; NIV (BiPAP) initiation when FVC <50% predicted or desaturation on oximetry; respiratory failure is a leading cause of mortality in severe OI",
        ],
        "etiologies": {
            "OI_Type_3_Glycine_Subst": {"pct": 72, "phenotype": "dominant-negative Gly→X substitution — severe, progressive, wheelchair"},
            "OI_Type_4_COL1A2_Mod": {"pct": 18, "phenotype": "less critical position — moderate phenotype, variable ambulation"},
            "Bruck_Syndrome_AR_COL1A2": {"pct": 5, "phenotype": "AR exon 52 splice — OI + congenital joint contractures (pterygium)"},
            "De_Novo_Severe": {"pct": 30, "phenotype": "de novo glycine substitution — no family history; parental mosaicism 5-7%"},
        },
        "stats": {
            "mean_onset_age_y": 0.1,
            "mean_dx_delay_months": 0.5,
            "wheelchair_dependence_pct": 65,
            "basilar_invagination_pct": 25,
        },
        "dx_delay_distribution": {"<1mo": 70, "1-6mo": 22, ">6mo": 8},
        "patients": [],
    },
    # ── ALPL — Hypophosphatasia ───────────────────────────────────────────────
    {
        "gene": "ALPL",
        "protein": (
            "ALPL — 1p36.12 AD/AR — Tissue-Nonspecific-Alkaline-Phosphatase-524aa — "
            "Hypophosphatasia-PPi-Accumulation — "
            "Premature-Deciduous-Teeth-Loss-<5y-PATHOGNOMONIC — "
            "B6-Responsive-Neonatal-Seizures — "
            "Asfotase-Alfa-FDA-2015-ERT — "
            "CPPD-Pseudogout-Adults"
        ),
        "alias": (
            "ALPL (alkaline phosphatase, tissue-nonspecific); OMIM gene 171760; "
            "Hypophosphatasia (HPP) OMIM 241500 (perinatal lethal), 241510 (infantile), "
            "146300 (childhood), 146300 (adult). "
            "1p36.12; 524 aa; ~57 kDa; autosomal dominant (mild adult forms) or recessive (severe). "
            "FUNCTION: ALPL encodes tissue-nonspecific alkaline phosphatase (TNSALP) — "
            "GPI-anchored ectoenzyme on osteoblasts, hepatocytes, kidney, and chondrocytes. "
            "TNSALP hydrolyses pyrophosphate (PPi), phosphoethanolamine (PEA), and "
            "pyridoxal-5-phosphate (PLP). "
            "PPi is a potent inhibitor of mineralisation; TNSALP deficiency → PPi accumulates → "
            "impaired hydroxyapatite crystal deposition → rickets/osteomalacia. "
            "PLP (active vitamin B6): TNSALP-deficient neurons accumulate PLP extracellularly → "
            "cannot be converted to pyridoxal (which crosses BBB) → intracellular PLP deficiency "
            "in CNS → B6-responsive seizures (neonatal). "
            "MUTATION SPECTRUM: >400 pathogenic variants; mostly missense; "
            "genotype-phenotype correlation: compound heterozygous or homozygous severe variants → "
            "perinatal/infantile severe forms; heterozygous mild variants → adult or odonto-HPP; "
            "CLINICAL SPECTRUM (6 forms): "
            "(1) Perinatal lethal — profound hypomineralisation; membraneous skull ossification only; "
            "hypoplastic lungs; respiratory failure at birth; "
            "(2) Perinatal benign (hypophosphatasaemia) — initial severe → spontaneous improvement; "
            "(3) Infantile — onset <6 months; rachitic changes, failure to thrive, hypercalcaemia, "
            "nephrocalcinosis, craniosynostosis, vitamin B6-responsive seizures; 50% mortality if untreated; "
            "(4) Childhood — PREMATURE LOSS OF DECIDUOUS TEETH BEFORE AGE 5 (root intact) PATHOGNOMONIC; "
            "rachitic-like bowing, stress fractures, delayed walking; "
            "(5) Adult — stress fractures (metatarsal 'pseudofractures'); CPPD/chondrocalcinosis; "
            "low ALP; may be missed for years; "
            "(6) Odonto-HPP — isolated dental manifestations; normal ALP sometimes; "
            "BIOCHEMISTRY: "
            "Serum ALP very LOW (below age-/sex-specific normal) — KEY diagnostic; "
            "plasma PLP ELEVATED (>5× ULN confirms HPP); "
            "urinary phosphoethanolamine ELEVATED; urinary PPi ELEVATED; "
            "IMPORTANT: standard labs 'normal ALP' in adults may still represent low-for-age; "
            "use paediatric reference ranges in children — adult ALP 'low-normal' may be pathological; "
            "DIAGNOSTIC TRAP: serum ALP may be 'artificially normal' during active fracture healing; "
            "TREATMENT — ASFOTASE ALFA (STRENSIQ, Alexion): "
            "FIRST enzyme replacement therapy for a primary bone disease (FDA/EMA 2015); "
            "recombinant TNSALP fused to bone-targeting deca-aspartate domain + IgG Fc; "
            "targets bone mineral surface directly; hydrolyses PPi locally; "
            "given subcutaneously 3×/week or 6×/week; "
            "INDICATIONS: perinatal/infantile/juvenile-onset HPP (FDA-labelled); "
            "adult: significant morbidity (stress fractures, respiratory compromise); "
            "RESPONSE MARKERS: improvement in rachitic changes (X-ray), respiratory improvement, "
            "mineralisation of skull, ↑ height velocity, ↑ mobility; "
            "ALP levels normalise or rise (from near-zero to low-normal); "
            "CONTRAINDICATION TO B6 supplementation in infantile HPP: "
            "high-dose B6 is NOT therapeutic and may worsen systemic PLP imbalance; "
            "CPPD IN ADULTS: calcium pyrophosphate deposition (pseudo-gout) in HPP adults — "
            "distinguish from gout (uric acid normal in HPP); synovial fluid analysis mandatory."
        ),
        "locus": "1p36.12",
        "aa": 524,
        "kDa": 57,
        "omim_gene": "171760",
        "omim_disease": "241500",
        "inheritance": "AD (mild adult/odonto-HPP); AR (severe infantile/perinatal); de novo severe",
        "gene_class": "Tissue-Nonspecific Alkaline Phosphatase — PPi Hydrolysis — Bone Mineralisation",
        "key_alerts": [
            "ALPL-PREMATURE-DECIDUOUS-TEETH-<5y-PATHOGNOMONIC: loss of baby teeth with INTACT ROOT before age 5 (i.e. not normal exfoliation, not caries-related) is PATHOGNOMONIC for childhood HPP — measure serum ALP IMMEDIATELY; refer to metabolic bone specialist; asfotase alfa treatment prevents progressive skeletal disease if started early",
            "ALPL-ASFOTASE-ALFA-PERINATAL-INFANTILE: asfotase alfa (STRENSIQ) is FDA-approved for perinatal/infantile/juvenile-onset HPP — the ONLY disease-modifying therapy; start immediately in all confirmed infantile HPP (survival benefit proven: 76% survival with treatment vs 27% historical untreated); enrol in registry (PEALS/MINERALS)",
            "ALPL-B6-SEIZURES-PYRIDOXINE-MECHANISM: neonatal seizures in infantile HPP are B6-DEPENDENT but NOT responsive to standard anticonvulsants (phenobarbital, benzodiazepines) — give IV pyridoxine 100 mg trial empirically in any neonate with refractory seizures + low ALP; asfotase alfa corrects underlying PLP accumulation",
            "ALPL-LOW-ALP-DIAGNOSTIC: serum ALP persistently BELOW the lower limit of normal (by age-sex reference range) is the biochemical hallmark — in paediatric practice, 'low-normal' ALP must be checked against paediatric norms (ALP is normally very high in children); do NOT dismiss low ALP as clinically insignificant without measuring PLP and PEA",
        ],
        "etiologies": {
            "Perinatal_Lethal": {"pct": 8, "phenotype": "biallelic severe — unmineralised bones, respiratory failure at birth"},
            "Infantile_HPP": {"pct": 22, "phenotype": "AR compound het — rachitis, hypercalcaemia, B6-seizures; 50% untreated mortality"},
            "Childhood_HPP": {"pct": 30, "phenotype": "AR or AD — premature tooth loss, stress fractures, delayed walking"},
            "Adult_HPP": {"pct": 30, "phenotype": "AD or mild AR — CPPD, metatarsal stress fractures, low serum ALP"},
            "Odonto_HPP": {"pct": 10, "phenotype": "mild heterozygous — isolated dental (premature tooth loss) only"},
        },
        "stats": {
            "mean_onset_age_y": 5.4,
            "mean_dx_delay_months": 28.0,
            "asfotase_alfa_survival_improvement_pct": 49,
            "adult_cppd_pct": 38,
        },
        "dx_delay_distribution": {"<6mo": 18, "6-24mo": 32, ">24mo": 50},
        "patients": [],
    },
    # ── PHEX — X-linked Hypophosphatemia ──────────────────────────────────────
    {
        "gene": "PHEX",
        "protein": (
            "PHEX — Xp22.11 XLR — Phosphate-Regulating-Endopeptidase-Homolog-749aa — "
            "X-Linked-Hypophosphatemia-Most-Common-Hereditary-Rickets — "
            "Dental-Abscesses-75pct-Without-Caries-PATHOGNOMONIC — "
            "Burosumab-Anti-FGF23-FDA-2018 — "
            "Enthesopathy-Adults — "
            "Monitor-TmP-GFR"
        ),
        "alias": (
            "PHEX (phosphate-regulating endopeptidase homolog, X-linked); OMIM gene 300550; "
            "X-linked hypophosphatemia (XLH) OMIM 307800. "
            "Xp22.11; 749 aa; ~86 kDa; X-linked recessive (fully penetrant in hemizygous males; "
            "variable penetrance in heterozygous females — mild to full phenotype). "
            "FUNCTION: PHEX is a zinc endopeptidase expressed on osteoblast/osteocyte surfaces. "
            "PHEX normally inactivates (cleaves) small integrin-binding ligand N-linked glycoproteins "
            "(SIBLINGs: DMP1, ASARM peptides) that stimulate FGF23 expression. "
            "PHEX loss-of-function → accumulation of ASARM peptides → excess FGF23 production "
            "by osteocytes → FGF23 acts on kidney proximal tubule → "
            "↓ NaPi-IIa/IIc cotransporter expression → phosphaturia → hypophosphataemia → "
            "impaired mineralisation of bone matrix (osteoid accumulation → rickets/osteomalacia). "
            "FGF23 simultaneously suppresses 1α-hydroxylase → ↓ 1,25-VitD synthesis → "
            "inappropriately low 1,25-VitD despite hypophosphataemia. "
            "GENETICS: "
            "Hemizygous males: fully affected; heterozygous females: may have mild, moderate, "
            "or rarely full phenotype; X-inactivation pattern influences female severity; "
            "de novo variants: ~30% of XLH cases (no family history); "
            "CLINICAL FEATURES: "
            "Rickets: bowing of legs in early childhood (weight-bearing → bilateral genu varum); "
            "delayed walking, waddling gait; growth failure; "
            "Dental abscesses (periapical): occur WITHOUT dental caries, WITHOUT trauma — "
            "DUE TO LARGE PULP CHAMBERS + poor mineralisation of circumpulpal dentine → "
            "bacterial microinfiltration → spontaneous abscess; "
            "75% of XLH children will have dental abscess at least once — PATHOGNOMONIC in context; "
            "Dental abscesses: begin with DECIDUOUS teeth; continue in permanent teeth; "
            "ENTHESOPATHY (calcification of tendon/ligament insertions): "
            "radiologically detected in >50% adults; causes joint pain, spinal stiffness; "
            "may mimic ankylosing spondylitis on imaging; "
            "Chiari malformation type I: increased risk (5–10× baseline); monitor with MRI in headaches; "
            "Craniosynostosis: premature fusion of skull sutures in infancy (~15%) → headache, raised ICP; "
            "BIOCHEMISTRY (key pattern): "
            "Serum phosphate LOW (< age-specific lower reference); "
            "TmP/GFR LOW (reduced tubular reabsorption of phosphate per unit GFR); "
            "Serum 1,25-VitD inappropriately normal or LOW (expected HIGH in hypophosphataemia); "
            "Serum FGF23 ELEVATED (intact FGF23 >100 RU/mL; distinguishes from nutritional rickets); "
            "Serum calcium: NORMAL; PTH: NORMAL or mildly elevated; "
            "ALP: ELEVATED (reflects bone turnover, not ALPL deficiency); "
            "TREATMENT: "
            "CONVENTIONAL (pre-burosumab): oral phosphate supplementation (multiple daily doses — "
            "4–6 doses/day to prevent renal washout) + calcitriol (1,25-VitD); "
            "MAJOR SIDE EFFECTS: nephrocalcinosis (urinary calcium × phosphate product high), "
            "hyperparathyroidism, GI intolerance; "
            "BUROSUMAB (CRYSVITA, Ultragenyx/Kyowa Kirin): anti-FGF23 monoclonal antibody (IgG1); "
            "FDA 2018 (children/adults); subcutaneous every 2 weeks; "
            "SUPERIORITY vs conventional: better phosphate correction, less nephrocalcinosis risk, "
            "once every 2 weeks vs multiple daily doses; "
            "SWITCH FROM CONVENTIONAL TO BUROSUMAB: "
            "stop phosphate and calcitriol BEFORE burosumab (risk of hyperphosphataemia); "
            "24h phosphate clearance gap mandatory; "
            "MONITORING ON BUROSUMAB: serum phosphate, TmP/GFR, ALP, PTH, 1,25-VitD, urinary calcium; "
            "DENTAL MANAGEMENT: "
            "preventive sealants, fluoride varnish, prompt antibiotic treatment for abscesses; "
            "burosumab may reduce new abscess incidence (improved dentine mineralisation)."
        ),
        "locus": "Xp22.11",
        "aa": 749,
        "kDa": 86,
        "omim_gene": "300550",
        "omim_disease": "307800",
        "inheritance": "XLR; hemizygous males fully affected; heterozygous females variable",
        "gene_class": "Phosphate-Regulating Endopeptidase — FGF23 Excess — Renal Phosphate Wasting",
        "key_alerts": [
            "PHEX-DENTAL-ABSCESS-75pct-WITHOUT-CARIES-PATHOGNOMONIC: spontaneous periapical dental abscesses WITHOUT identifiable caries in a child with bowed legs/rickets = PHEX/XLH until proven otherwise; measure fasting serum phosphate, FGF23, TmP/GFR immediately; dentist must be informed of XLH diagnosis to avoid unnecessary tooth extractions — abscess is not due to poor dental hygiene",
            "PHEX-BUROSUMAB-SWITCH-PHOSPHATE-STOP-FIRST: switching from conventional therapy (phosphate + calcitriol) to burosumab requires a 24-hour washout of both medications before first burosumab dose — concurrent use causes severe hyperphosphataemia and soft tissue calcification; coordinate carefully",
            "PHEX-ENTHESOPATHY-ADULTS-NOT-AS: calcification of entheses (tendon/ligament insertions) develops in >50% of adults with XLH — may look identical to ankylosing spondylitis on spinal X-ray; distinguish by: HLAB27 negative in XLH, sacroiliac joints spared in XLH, low phosphate + high FGF23; treat XLH not AS",
            "PHEX-CHIARI-MALFORMATION-HEADACHE-MRI: XLH carries ~5–10× increased risk of Chiari malformation type I (posterior fossa overcrowding due to skull base abnormality); any new-onset headache especially on Valsalva, neck pain, or upper limb symptoms → cervical/brain MRI; craniosynostosis in infants: monitor head circumference and fontanelle",
        ],
        "etiologies": {
            "Classic_XLH_Hemizygous_Male": {"pct": 48, "phenotype": "full phenotype — bowing, dental abscesses, short stature, enthesopathy adults"},
            "Heterozygous_Female_Moderate": {"pct": 32, "phenotype": "variable expression — mild to moderate bowing, dental, short stature"},
            "Heterozygous_Female_Mild": {"pct": 12, "phenotype": "biochemical only — hypophosphataemia without clinical rickets"},
            "De_Novo_XLH": {"pct": 30, "phenotype": "no family history — de novo PHEX variant; ~30% of all XLH"},
        },
        "stats": {
            "mean_onset_age_y": 1.5,
            "mean_dx_delay_months": 18.0,
            "dental_abscess_lifetime_pct": 75,
            "adult_enthesopathy_pct": 52,
        },
        "dx_delay_distribution": {"<6mo": 15, "6-24mo": 50, ">24mo": 35},
        "patients": [],
    },
    # ── FGF23 — Autosomal Dominant Hypophosphatemic Rickets ──────────────────
    {
        "gene": "FGF23",
        "protein": (
            "FGF23 — 12p13.32 AD — Fibroblast-Growth-Factor-23-251aa — "
            "Autosomal-Dominant-Hypophosphatemic-Rickets-ADHR — "
            "GOF-Prevents-PHEX-Cleavage-Elevated-Intact-FGF23 — "
            "Iron-Deficiency-TRIGGERS-Disease-Flares-KEY-DDx-XLH — "
            "Burosumab-Effective — "
            "Iron-Repletion-Alone-May-Resolve-Flare"
        ),
        "alias": (
            "FGF23 (fibroblast growth factor 23); OMIM gene 605380; "
            "Autosomal dominant hypophosphatemic rickets 1 (ADHR) OMIM 193100. "
            "12p13.32; 251 aa; ~32 kDa (mature protein after signal peptide cleavage); autosomal dominant. "
            "FUNCTION: FGF23 is an osteokine secreted by osteocytes/osteoblasts; "
            "acts on kidney proximal tubule (via FGFR1/Klotho co-receptor): "
            "→ ↓ NaPi-IIa/IIc expression → phosphaturia → hypophosphataemia; "
            "→ ↓ CYP27B1 (1α-hydroxylase) → ↓ 1,25-VitD synthesis. "
            "PHEX normally cleaves FGF23 at an RXXR motif (between R176-S177 or R179-Y180), "
            "degrading and inactivating it. "
            "ADHR MECHANISM: gain-of-function missense variants at R176 or R179 "
            "(e.g. R176Q, R179Q, R179W) → mutant FGF23 is RESISTANT TO CLEAVAGE by PHEX → "
            "excess intact FGF23 circulates → same downstream phosphate-wasting as XLH. "
            "KEY DISTINCTION FROM XLH: "
            "ADHR shows variable, episodic severity correlated with IRON STATUS. "
            "Iron deficiency → transcriptional upregulation of FGF23 AND impaired proprotein "
            "convertase cleavage → further accumulation of intact FGF23 → worsening phenotype. "
            "Iron repletion in iron-deficient ADHR patients can NORMALISE FGF23 and resolve rickets. "
            "This iron dependence is not seen in XLH. "
            "CLINICAL FEATURES — ADHR: "
            "Onset variable: some present in childhood (childhood ADHR); "
            "others present in adulthood during pregnancy/lactation or iron deficiency (adult-onset ADHR); "
            "Bone pain, fractures, muscle weakness similar to XLH; "
            "Dental abscesses: present (same mechanism — abnormal dentine mineralisation); "
            "Incomplete penetrance: some obligate carriers unaffected (notable in ADHR vs XLH); "
            "BIOCHEMISTRY: "
            "Hypophosphataemia + ↓ TmP/GFR + elevated intact FGF23 + ↓/N 1,25-VitD; "
            "KEY: measure SERUM FERRITIN / IRON STUDIES in all cases; "
            "iron deficiency → exacerbate or even precipitate first episode; "
            "TREATMENT: "
            "Burosumab (anti-FGF23): effective (same target as in XLH); "
            "IV iron supplementation (if iron-deficient): may alone normalise FGF23 and resolve disease; "
            "Conventional: oral phosphate + calcitriol (same as XLH, similar concerns); "
            "DIFFERENTIAL DIAGNOSIS FROM XLH: "
            "ADHR: FGF23 GOF variant (R176/179); episodic/variable severity; iron-dependent; AD; "
            "XLH: PHEX LOF variant; constant hypophosphataemia; not iron-dependent; XLR; "
            "TIO (tumour-induced osteomalacia): acquired; elevated FGF23 from tumour; "
            "find and excise tumour (Ga-68 DOTATATE PET/CT); "
            "ARHR1 (DMP1 LOF): AR; similar biochemistry; DMP1 panel; "
            "Nutritional rickets: low 25-OH-VitD; elevated 1,25-VitD (not low); FGF23 normal."
        ),
        "locus": "12p13.32",
        "aa": 251,
        "kDa": 32,
        "omim_gene": "605380",
        "omim_disease": "193100",
        "inheritance": "AD; gain-of-function at R176/R179; incomplete penetrance",
        "gene_class": "Fibroblast Growth Factor 23 — Osteokine — Renal Phosphate-Wasting Hormone",
        "key_alerts": [
            "FGF23-IRON-DEFICIENCY-TRIGGERS-ADHR-FLARES: iron deficiency is the KEY environmental trigger for ADHR severity — during pregnancy, lactation, or menorrhagia iron depletion raises FGF23 and precipitates disease flares or first presentation; CHECK FERRITIN in all hypophosphataemia of unknown cause; IV iron repletion alone can resolve an ADHR flare without burosumab",
            "FGF23-INTACT-FGF23-ASSAY-MANDATORY: intact FGF23 must be measured to confirm elevated; total FGF23 (intact + fragments) is NOT useful in this context; Kainos immunotopometric intact FGF23 assay: >30 pg/mL is above normal; collect in EDTA tube, immediately on ice; elevated FGF23 + hypophosphataemia → FGF23-mediated phosphate-wasting syndrome",
            "FGF23-ADHR-vs-XLH-GENETIC-TEST-MANDATORY: ADHR (FGF23 GOF) and XLH (PHEX LOF) have identical biochemistry — GENETIC TESTING is required to distinguish; XLH is XLR (PHEX), ADHR is AD (FGF23) — determines inheritance pattern, risk to offspring, and response to iron; same treatment (burosumab) works for both",
            "FGF23-TUMOUR-INDUCED-OSTEOMALACIA-EXCLUDE: elevated FGF23 + hypophosphataemia in an ADULT without family history → exclude TIO (Ga-68 DOTATATE PET/CT to find occult mesenchymal tumour); TIO is cured by tumour excision; FGF23 normalises within days; do NOT start lifelong burosumab before imaging",
        ],
        "etiologies": {
            "Childhood_ADHR_Classic": {"pct": 55, "phenotype": "childhood onset — rickets, bowing, short stature; variable severity"},
            "Adult_Onset_ADHR_Iron": {"pct": 30, "phenotype": "adult onset triggered by iron deficiency (pregnancy/lactation/menorrhagia)"},
            "Incomplete_Penetrance_Carrier": {"pct": 15, "phenotype": "obligate carrier — biochemically normal or minimally affected"},
        },
        "stats": {
            "mean_onset_age_y": 7.2,
            "mean_dx_delay_months": 34.0,
            "iron_deficiency_trigger_pct": 42,
            "burosumab_response_pct": 88,
        },
        "dx_delay_distribution": {"<6mo": 8, "6-24mo": 28, ">24mo": 64},
        "patients": [],
    },
    # ── ACVR1 — Fibrodysplasia Ossificans Progressiva ─────────────────────────
    {
        "gene": "ACVR1",
        "protein": (
            "ACVR1 — 2q24.1 AD — Activin-A-Receptor-Type-1-ALK2-509aa — "
            "Fibrodysplasia-Ossificans-Progressiva-Stone-Man-Syndrome — "
            "R206H-Founder-Mutation-97pct — "
            "NO-Biopsies-NO-IM-Injections-NO-Surgery-ABSOLUTE — "
            "Palovarotene-FDA-2023-First-Approved — "
            "Garetosmab-Phase2-LUMINA1"
        ),
        "alias": (
            "ACVR1 (activin A receptor type 1; also ALK2); OMIM gene 102576; "
            "Fibrodysplasia ossificans progressiva (FOP) OMIM 135100. "
            "2q24.1; 509 aa; ~57 kDa; autosomal dominant; "
            "R206H accounts for ~97% of all FOP cases; most are de novo (no family history). "
            "FUNCTION: ACVR1 (ALK2) is a type I bone morphogenetic protein (BMP) receptor — "
            "serine-threonine kinase that activates SMAD1/5/9 signalling. "
            "Wild-type: activated by BMP ligands (BMP2/4/7) → osteogenic differentiation; "
            "inhibited by FKBP12 in the unactivated state (GS domain interaction). "
            "R206H FOP MUTATION: disrupts FKBP12 inhibition → constitutively active ACVR1 → "
            "chronic low-level BMP signalling in muscles and soft tissues → susceptibility to "
            "aberrant ACVR1 activation by activin A → HO (heterotopic ossification). "
            "UNIQUE FOP MECHANISM: wild-type ACVR1 is INHIBITED by activin A (ligand trap); "
            "R206H ACVR1 is ACTIVATED by activin A → explains why garetosmab (anti-activin A) "
            "is therapeutically rational — blocking activin A prevents aberrant ACVR1 activation. "
            "CLINICAL FEATURES — FOP: "
            "CONGENITAL MALFORMATION: bilateral short first toes (hallux valgus + short great toe) "
            "are PRESENT AT BIRTH → most specific early diagnostic sign; "
            "almost 100% of FOP patients have this malformation; "
            "Swelling/flare-ups: childhood onset (median age ~5 years); "
            "soft-tissue swellings (often misdiagnosed as sarcoma → biopsy catastrophic); "
            "flares triggered by: trauma, intramuscular injections, viral illness, falls, "
            "fatigue, iatrogenic needle procedures, dental blocks by IM injection; "
            "after each flare: new HO forms at that site, permanently restricting movement; "
            "PROGRESSION: episodic, cumulative, irreversible; "
            "upper limbs and axial skeleton involved early (shoulders, spine); "
            "'stone man' progression: trunk → shoulders → elbows → hips → knees → jaw; "
            "Jaw involvement: trismus (mouth opening restricted) → "
            "dental treatment extremely restricted; dental GA must use nasotracheal/fibreoptic; "
            "Thoracic restriction: diaphragm-sparing initially; intercostal muscle HO → "
            "restrictive lung disease → respiratory failure (cause of death); "
            "Intelligence: NORMAL; cognitive function preserved; "
            "MEDICAL CATASTROPHES TO ABSOLUTELY AVOID: "
            "1. BIOPSY — triggers massive HO at biopsy site; "
            "2. IM INJECTIONS — triggers HO (all vaccinations must be subcutaneous or intradermal); "
            "3. SURGERY for existing HO — triggers more HO; HO should NEVER be excised; "
            "4. DENTAL INJECTIONS by standard inferior alveolar nerve block — "
            "mandibular injections permissible; posterior superior alveolar: risk; "
            "use only infiltration or mental nerve block; "
            "ANAESTHETIC PRECAUTIONS: "
            "Cervical spine: ankylosis in older FOP → limited neck extension → "
            "ALWAYS plan for DIFFICULT AIRWAY; videolaryngoscopy or fibreoptic mandatory; "
            "TREATMENT: "
            "Palovarotene (SOHONOS, Ipsen): oral retinoid receptor γ agonist; FDA 2023 "
            "(first approved therapy for FOP); reduces volume of new HO during flares; "
            "NOT disease-reversing — does not remove existing HO; "
            "DOSING: baseline (ongoing) + higher flare dose; "
            "ADVERSE EFFECTS: mucocutaneous (dry skin, lips), teratogenic (women of childbearing age: "
            "mandatory contraception); premature epiphyseal fusion in growing children; "
            "Garetosmab (anti-activin A monoclonal, Regeneron): Phase 2 LUMINA-1 — "
            "anti-activin A blocks aberrant ACVR1 activation → reduced new HO volume; "
            "Rapifudin (ACVR1 kinase inhibitor, Blueprint Medicines): Phase 2/3 ongoing."
        ),
        "locus": "2q24.1",
        "aa": 509,
        "kDa": 57,
        "omim_gene": "102576",
        "omim_disease": "135100",
        "inheritance": "AD; de novo in >85%; R206H founder ~97%; rare atypical ACVR1 variants",
        "gene_class": "Type I BMP Receptor / ALK2 — SMAD1/5/9 Kinase — Constitutively Active GOF",
        "key_alerts": [
            "ACVR1-NO-BIOPSY-NO-IM-INJECTION-NO-HO-SURGERY-ABSOLUTE: in any child with bilateral short great toes + soft-tissue swellings, FOP MUST be excluded BEFORE ANY PROCEDURE — biopsy, IM injection, or surgical excision of HO triggers catastrophic irreversible heterotopic ossification at the procedure site; all vaccinations must be SUBCUTANEOUS in confirmed or suspected FOP",
            "ACVR1-SHORT-GREAT-TOES-AT-BIRTH-DIAGNOSTIC-CLUE: bilateral hallux valgus with short first metatarsal/phalanx is present AT BIRTH in virtually all FOP patients — this is the KEY pre-flare diagnostic sign; any child with this foot abnormality + ANY soft tissue swelling: urgent FOP genetic testing before any procedure",
            "ACVR1-PALOVAROTENE-FDA-2023-FLARE-DOSE: palovarotene (SOHONOS) is the first FDA-approved FOP therapy; baseline 5 mg/day + flare dose 20 mg/day × 4 weeks then 10 mg/day × 8 weeks when flare begins; teratogenic → mandatory contraception in females; premature growth plate closure in children → monitor height velocity",
            "ACVR1-ANAESTHETIC-DIFFICULT-AIRWAY-MANDATORY-PLAN: FOP patients with jaw trismus and cervical ankylosis have PREDICTED DIFFICULT AIRWAY — anaesthetic team must be informed of FOP diagnosis at ANY procedure; plan for awake fibreoptic intubation or nasotracheal intubation; NEVER attempt standard laryngoscopy without backup; carry FOP emergency card",
        ],
        "etiologies": {
            "Classic_FOP_R206H": {"pct": 97, "phenotype": "R206H de novo — bilateral short toes, episodic HO flares, progressive restriction"},
            "Atypical_FOP_ACVR1_Variant": {"pct": 3, "phenotype": "atypical ACVR1 variant — variable phenotype, sometimes more severe"},
        },
        "stats": {
            "mean_onset_age_y": 4.8,
            "mean_dx_delay_months": 48.0,
            "de_novo_pct": 85,
            "respiratory_mortality_pct": 80,
        },
        "dx_delay_distribution": {"<12mo": 12, "12-60mo": 48, ">60mo": 40},
        "patients": [],
    },
    # ── RUNX2 — Cleidocranial Dysplasia ──────────────────────────────────────
    {
        "gene": "RUNX2",
        "protein": (
            "RUNX2 — 6p21.1 AD — Runt-Related-Transcription-Factor-2-521aa — "
            "Cleidocranial-Dysplasia-Haploinsufficiency — "
            "Absent-Hypoplastic-Clavicles-PATHOGNOMONIC-Shoulders-Midline — "
            "Supernumerary-Teeth-Dental-Surgery-Mandatory — "
            "Patent-Fontanelle-Adults — "
            "Intelligence-NORMAL"
        ),
        "alias": (
            "RUNX2 (runt-related transcription factor 2; also CBFA1, AML3, PEBP2aA); "
            "OMIM gene 600211; Cleidocranial dysplasia (CCD) OMIM 119600. "
            "6p21.1; 521 aa; ~57 kDa; autosomal dominant; haploinsufficiency mechanism. "
            "FUNCTION: RUNX2 is the master transcription factor for osteoblast differentiation; "
            "required for osteoblast commitment from mesenchymal stem cells; "
            "also essential for tooth eruption (controls FST, EGF, and dentoalveolar development). "
            "HAPLOINSUFFICIENCY: one functional RUNX2 allele is insufficient for normal bone/tooth development. "
            "MUTATION SPECTRUM: "
            "All types of LOF: nonsense, frameshift, splice, missense (in Runt domain or PST domain); "
            "large deletions encompassing RUNX2 → contiguous gene syndrome if large; "
            "~30% de novo variants; "
            "CLINICAL FEATURES — CLEIDOCRANIAL DYSPLASIA: "
            "CLAVICLES: absent or markedly hypoplastic (may be vestigial lateral stubs only); "
            "PATHOGNOMONIC MANOEUVRE: patient can approximate shoulders in midline of chest "
            "(or very close) — demonstrate at diagnosis; "
            "variable: complete absence vs. fibrous remnant vs. partial aplasia; "
            "shoulder instability, winging of scapula; "
            "SKULL: "
            "Widely patent fontanelles and sutures (frontal, parietal, sagittal persist well into adulthood); "
            "Wormian bones: multiple small irregular bones within sutures; "
            "Frontal and parietal bossing: large prominent forehead; "
            "Patent metopic suture; "
            "Midface hypoplasia: flat midface, depressed nasal bridge, hypertelorism; "
            "Delayed closure of fontanelle: anterior fontanelle open until 3rd–5th decade in some; "
            "TEETH — MAJOR CLINICAL PROBLEM: "
            "Supernumerary teeth: extra (unerupted) permanent teeth — typically 10–40 supernumerary; "
            "Retention of primary teeth: primary teeth NOT shed at normal age because "
            "RUNX2 deficiency impairs eruption pathway (alveolar remodelling fails); "
            "Retained primary teeth BLOCK supernumerary and permanent teeth from erupting; "
            "NET RESULT: overcrowded, impacted dental arches, risk of dental cyst formation; "
            "DENTAL MANAGEMENT (MANDATORY): "
            "Orthopantomogram (OPG) every 2 years from age 4–5; "
            "Surgical dental extraction of primary teeth and supernumerary teeth ~age 7–9 "
            "(allows space for permanent eruption); "
            "Orthodontic treatment ± further surgical exposure; "
            "Multiple surgeries often required into adulthood; "
            "SKELETAL: "
            "Short stature: typically -2 to -3 SD; "
            "Short hands with tapered fingers; brachydactyly; "
            "Genu valgum or varum; joint hypermobility; "
            "Scoliosis in some; narrowed chest; "
            "Delayed ossification of pubic symphysis; "
            "Intelligence: ENTIRELY NORMAL — important for counselling; "
            "Hearing: conductive loss from external auditory canal abnormalities; audiogram recommended; "
            "MANAGEMENT: "
            "Multidisciplinary: orthodontics/oral surgery (most intensive), physiotherapy, "
            "orthopaedics (shoulder stabilisation), audiology; "
            "No medical treatment for bone density (CCD is not an osteoporosis syndrome); "
            "Clavicle reconstruction: generally not attempted; shoulder physiotherapy for stability."
        ),
        "locus": "6p21.1",
        "aa": 521,
        "kDa": 57,
        "omim_gene": "600211",
        "omim_disease": "119600",
        "inheritance": "AD; haploinsufficiency; de novo ~30%; variable expressivity within families",
        "gene_class": "Master Osteoblast Transcription Factor — Runt Domain — Tooth Eruption Regulator",
        "key_alerts": [
            "RUNX2-SUPERNUMERARY-TEETH-DENTAL-SURGERY-MANDATORY: cleidocranial dysplasia causes 10–40 supernumerary unerupted teeth that will NOT erupt spontaneously — dental neglect leads to cyst formation, dental abscess, and permanent crowding; OPG from age 4–5, surgical extraction of primary + supernumerary teeth ~age 7–9; refer to oral-maxillofacial surgeon at diagnosis",
            "RUNX2-SHOULDERS-MIDLINE-MANOEUVRE-DIAGNOSTIC: pathognomonic clinical sign — patient approximates both shoulders toward midline in front of the chest (bilateral absent/hypoplastic clavicles permit this); demonstrate and photograph at first clinic visit; also X-ray clavicles; absent clavicle fragment = confirm CCD diagnosis",
            "RUNX2-INTELLIGENCE-NORMAL-COUNSELLING: RUNX2 haploinsufficiency causes skeletal/dental abnormalities ONLY — intelligence is entirely normal; do not imply or assume neurodevelopmental delay; this is critical for early counselling and educational planning; some families have been inappropriately placed in special education based on physical appearance",
            "RUNX2-PATENT-FONTANELLE-ADULTS-NOT-CONCERNING: adult CCD patients may have widely open anterior fontanelle — this is NOT a sign of raised intracranial pressure; no intervention required; document and inform treating physicians to prevent unnecessary skull surgery; Wormian bones on skull X-ray are expected findings in CCD",
        ],
        "etiologies": {
            "Classic_CCD_Full": {"pct": 65, "phenotype": "absent/vestigial clavicles + full supernumerary + patent fontanelle + short stature"},
            "CCD_Partial_Clavicle": {"pct": 25, "phenotype": "partial clavicle + supernumerary teeth — milder skeletal; dental mandatory"},
            "De_Novo_CCD": {"pct": 30, "phenotype": "de novo RUNX2 LOF — no family history; parental germline mosaic 3-5%"},
            "Intrafamilial_Variable": {"pct": 25, "phenotype": "highly variable within same family — same variant, different phenotype"},
        },
        "stats": {
            "mean_onset_age_y": 0.3,
            "mean_dx_delay_months": 12.0,
            "supernumerary_teeth_mean_count": 18,
            "dental_surgery_required_pct": 95,
        },
        "dx_delay_distribution": {"<6mo": 55, "6-24mo": 30, ">24mo": 15},
        "patients": [],
    },
    # ── LRP5 — Osteoporosis-Pseudoglioma / High Bone Mass ─────────────────────
    {
        "gene": "LRP5",
        "protein": (
            "LRP5 — 11q13.2 AD-GOF/AR-LOF — LDL-Receptor-Related-Protein-5-1615aa — "
            "Osteoporosis-Pseudoglioma-Syndrome-OPPG-AR-LOF-Blindness-Severe-Osteoporosis — "
            "High-Bone-Mass-Syndrome-AD-GOF-G171V-Benign — "
            "WNT-Beta-Catenin-Co-Receptor — "
            "Romosozumab-Mechanism-Exploits-LRP5-Pathway — "
            "Vitreous-Fibrovascular-Remnants-PATHOGNOMONIC-OPPG"
        ),
        "alias": (
            "LRP5 (low-density lipoprotein receptor-related protein 5); OMIM gene 603506; "
            "Osteoporosis-pseudoglioma syndrome (OPPG) OMIM 259770; "
            "High bone mass syndrome (HBM) OMIM 601884. "
            "11q13.2; 1615 aa; ~180 kDa; autosomal dominant (GOF-HBM) or recessive (LOF-OPPG). "
            "FUNCTION: LRP5 (and its paralogue LRP6) are WNT co-receptors — "
            "they form a signalling complex with Frizzled at the cell surface. "
            "WNT ligand binds Frizzled + LRP5/6 → intracellular phosphorylation of LRP5/6 → "
            "inhibition of β-catenin destruction complex (GSK3β/CK1α/Axin/APC) → "
            "cytoplasmic β-catenin accumulates → translocates to nucleus → "
            "activates TCF/LEF target genes → OSTEOBLAST PROLIFERATION + SURVIVAL. "
            "SCLEROSTIN (SOST protein) is an endogenous WNT antagonist: "
            "sclerostin binds LRP5/6 → inhibits WNT signalling → decreases bone formation; "
            "ROMOSOZUMAB (anti-sclerostin antibody) blocks sclerostin → "
            "LRP5/6 activation restored → WNT signalling → bone formation; "
            "This mechanism exploits LRP5 biology. "
            "LOF MECHANISM (OPPG): biallelic LOF → absent WNT/β-catenin in osteoblasts AND "
            "in vitreous vasculature → "
            "BONE: severe early-onset osteoporosis (DXA Z-score < -4 common); "
            "EYES: failure of vitreous vasculature regression in neonatal period → "
            "persistent fibrovascular remnants → retinal detachment → blindness; "
            "GOF MECHANISM (HBM): heterozygous G171V or similar → "
            "resistance to sclerostin-mediated inhibition → constitutive WNT signalling → "
            "very high bone density → sclerotic vertebral bodies; very strong bones; "
            "USUALLY BENIGN but may present as 'dense bones' on incidental imaging. "
            "CLINICAL FEATURES — OPPG (AR): "
            "Visual loss: bilateral; vitreous fibrovascular remnants; "
            "PATHOGNOMONIC ON OPHTHALMOLOGICAL EXAM: posterior persistent hyperplastic primary vitreous (PHPV) "
            "or vitreous fibrovascular remnants; typically detected in neonatal/infant period; "
            "may progress to retinal detachment and blindness in first years if untreated; "
            "Ophthalmology: emergency review within weeks of birth; vitrectomy may preserve vision; "
            "Severe osteoporosis: vertebral compression fractures; long-bone fractures (toddling age); "
            "DXA Z-score severely reduced; "
            "Mental development: variable mild ID in some patients (limbic LRP5 expression); "
            "NOT always intellectually normal — contrast with CCD where intelligence always normal; "
            "CLINICAL FEATURES — HBM (GOF, AD): "
            "Dense bones on X-ray or DXA: very high BMD (T-score or Z-score highly positive); "
            "Sclerotic vertebral bodies ('bone within bone' appearance on spinal X-ray); "
            "Usually asymptomatic and benign; "
            "Mandible may also be dense/expanded; "
            "Torus palatinus (bony palatal protuberance) — more common in HBM kindreds; "
            "MANAGEMENT — OPPG: "
            "Ophthalmology: urgent referral in neonatal period; "
            "vitrectomy if fibrovascular remnants present (preserve visual axis); "
            "Bisphosphonates: IV pamidronate/zoledronic acid — standard for OPPG osteoporosis; "
            "Recombinant human PTH (teriparatide): anabolic option in adults with OPPG; "
            "Romosozumab: studied in OPPG (anti-sclerostin = restore LRP5 downstream signalling); "
            "MANAGEMENT — HBM: "
            "Usually NO treatment required; reassure about bone density; "
            "Monitor for jaw/mandible effects (can complicate dental procedures)."
        ),
        "locus": "11q13.2",
        "aa": 1615,
        "kDa": 180,
        "omim_gene": "603506",
        "omim_disease": "259770",
        "inheritance": "AR (OPPG — biallelic LOF); AD (HBM — heterozygous GOF); carrier parents biochemically normal",
        "gene_class": "WNT Co-Receptor LRP5 — β-Catenin Signalling — Osteoblast / Vitreous Vasculature",
        "key_alerts": [
            "LRP5-OPPG-VITREOUS-FIBROVASCULAR-REMNANTS-NEONATAL-OPHTHALMOLOGY-URGENT: OPPG causes bilateral vitreous fibrovascular remnants pathognomonic on fundoscopy in infancy — ALL neonates with confirmed biallelic LRP5 LOF must have urgent ophthalmological review within first weeks of life; vitrectomy can preserve sight if performed before retinal detachment; missed diagnosis = irreversible blindness",
            "LRP5-OPPG-SEVERE-OSTEOPOROSIS-DXA-Z-BELOW-4: OPPG causes DXA Z-score < -4 in most patients — DO NOT dismiss as 'mild osteoporosis'; IV bisphosphonates (pamidronate cycles) mandatory; vertebral X-ray annually for compression fractures; fracture risk comparable to OI type 3; severity often under-recognised in absence of skeletal deformity",
            "LRP5-HBM-BENIGN-REASSURE-NO-TREATMENT: high bone mass (G171V and related GOF variants) presents as very high DXA and sclerotic vertebrae on X-ray — this is BENIGN and protective against osteoporosis; reassure patient; no treatment; IMPORTANT: do not confuse with malignant bone diseases (e.g. metastases or Paget disease) — family history of dense bones and no symptoms clinches HBM",
            "LRP5-ROMOSOZUMAB-MECHANISM-EXPLOITS-LRP5-PATHWAY: romosozumab (anti-sclerostin antibody) works by blocking sclerostin binding to LRP5/LRP6, allowing WNT signalling to proceed; this is the SAME pathway as LRP5 GOF-HBM; understanding LRP5 biology is the scientific basis for romosozumab in osteoporosis; anti-sclerostin approach under investigation in OPPG (to restore residual WNT signalling)",
        ],
        "etiologies": {
            "OPPG_Biallelic_LOF": {"pct": 60, "phenotype": "AR — vitreous fibrovascular remnants + severe osteoporosis; blindness if untreated"},
            "HBM_GOF_G171V_Classic": {"pct": 28, "phenotype": "AD GOF G171V — very high BMD, sclerotic vertebrae, torus palatinus; benign"},
            "HBM_Other_GOF": {"pct": 12, "phenotype": "AD other GOF variants — high BMD, variable phenotype"},
        },
        "stats": {
            "mean_onset_age_y": 1.2,
            "mean_dx_delay_months": 22.0,
            "oppg_blindness_if_untreated_pct": 70,
            "hbm_symptomatic_pct": 5,
        },
        "dx_delay_distribution": {"<6mo": 38, "6-24mo": 42, ">24mo": 20},
        "patients": [],
    },
]


def _generate_patients():
    for idx, gene_data in enumerate(BONE_DISEASE_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        patients = []
        gene = gene_data["gene"]
        for i in range(40):
            if gene == "COL1A1":
                oi_subtype = rng.choice(["OI1_null", "OI1_null", "OI4_missense"])
                has_di = rng.random() < 0.35
                has_hearing_loss = rng.random() < 0.45
                onset_age = rng.randint(0, 4)
                age_at_dx = onset_age + rng.randint(0, 2)
                dx_delay = max(1, (age_at_dx - onset_age) * 12 + rng.randint(0, 12))
                fracture_count = rng.randint(2, 18)
                patients.append({
                    "patient_id": f"COL1A1-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": oi_subtype,
                    "dentinogenesis_imperfecta": has_di,
                    "hearing_loss": has_hearing_loss,
                    "fracture_count": fracture_count,
                    "gene": gene, "seed": seed,
                })
            elif gene == "COL1A2":
                oi_subtype = rng.choice(["OI3_dominant_neg", "OI3_dominant_neg", "OI4_moderate"])
                in_utero_fractures = oi_subtype == "OI3_dominant_neg" and rng.random() < 0.7
                wheelchair = oi_subtype == "OI3_dominant_neg" and rng.random() < 0.65
                onset_age = 0
                age_at_dx = rng.randint(0, 1)
                dx_delay = rng.randint(0, 3)
                patients.append({
                    "patient_id": f"COL1A2-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": oi_subtype,
                    "in_utero_fractures": in_utero_fractures,
                    "wheelchair_dependent": wheelchair,
                    "gene": gene, "seed": seed,
                })
            elif gene == "ALPL":
                form = rng.choice(["infantile", "childhood", "childhood", "adult", "odonto"])
                onset_age = (
                    rng.uniform(0, 0.5) if form == "infantile"
                    else rng.randint(1, 5) if form == "childhood"
                    else rng.randint(20, 50) if form == "adult"
                    else rng.randint(2, 8)
                )
                age_at_dx = onset_age + rng.uniform(0, 3)
                dx_delay = max(0, (age_at_dx - onset_age) * 12 + rng.randint(0, 24))
                premature_teeth_loss = form in ("childhood", "odonto") and rng.random() < 0.8
                asfotase_alfa = form in ("infantile", "childhood") and rng.random() < 0.55
                patients.append({
                    "patient_id": f"ALPL-{i+1:03d}",
                    "onset_age": round(onset_age, 1),
                    "age_at_dx": round(age_at_dx, 1),
                    "dx_delay_months": round(dx_delay),
                    "phenotype": form,
                    "premature_deciduous_tooth_loss": premature_teeth_loss,
                    "asfotase_alfa_treated": asfotase_alfa,
                    "gene": gene, "seed": seed,
                })
            elif gene == "PHEX":
                sex = rng.choice(["M", "M", "F"])
                severity = "full" if sex == "M" else rng.choice(["moderate", "mild", "full"])
                onset_age = rng.randint(1, 3)
                age_at_dx = onset_age + rng.randint(0, 3)
                dx_delay = max(3, (age_at_dx - onset_age) * 12 + rng.randint(0, 18))
                dental_abscess = rng.random() < 0.75
                burosumab = rng.random() < 0.6
                patients.append({
                    "patient_id": f"PHEX-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "sex": sex,
                    "phenotype": severity,
                    "dental_abscess_history": dental_abscess,
                    "burosumab_treated": burosumab,
                    "gene": gene, "seed": seed,
                })
            elif gene == "FGF23":
                variant_type = rng.choice(["R176Q", "R179Q", "R179W"])
                onset_type = rng.choice(["childhood", "childhood", "adult_iron_def"])
                onset_age = rng.randint(2, 8) if onset_type == "childhood" else rng.randint(20, 42)
                age_at_dx = onset_age + rng.randint(1, 5)
                dx_delay = max(6, (age_at_dx - onset_age) * 12 + rng.randint(0, 24))
                iron_deficient = onset_type == "adult_iron_def" or rng.random() < 0.35
                patients.append({
                    "patient_id": f"FGF23-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "variant": variant_type,
                    "phenotype": onset_type,
                    "iron_deficient_at_presentation": iron_deficient,
                    "gene": gene, "seed": seed,
                })
            elif gene == "ACVR1":
                variant = rng.choice(["R206H"] * 19 + ["other_ACVR1"])
                onset_age = rng.randint(3, 8)
                age_at_dx = onset_age + rng.randint(0, 6)
                dx_delay = max(6, (age_at_dx - onset_age) * 12 + rng.randint(0, 48))
                biopsy_catastrophe = rng.random() < 0.35
                jaw_trismus = rng.random() < 0.5
                palovarotene_treated = rng.random() < 0.45
                patients.append({
                    "patient_id": f"ACVR1-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "variant": variant,
                    "phenotype": "FOP_classic" if variant == "R206H" else "FOP_atypical",
                    "prior_biopsy_catastrophe": biopsy_catastrophe,
                    "jaw_trismus": jaw_trismus,
                    "palovarotene_treated": palovarotene_treated,
                    "gene": gene, "seed": seed,
                })
            elif gene == "RUNX2":
                clavicle_severity = rng.choice(["absent_bilateral", "absent_bilateral", "vestigial", "partial"])
                supernumerary_count = rng.randint(8, 40)
                onset_age = 0
                age_at_dx = rng.randint(0, 2)
                dx_delay = age_at_dx * 12 + rng.randint(0, 12)
                hearing_loss = rng.random() < 0.35
                de_novo = rng.random() < 0.30
                patients.append({
                    "patient_id": f"RUNX2-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": clavicle_severity,
                    "supernumerary_teeth_count": supernumerary_count,
                    "hearing_loss": hearing_loss,
                    "de_novo": de_novo,
                    "gene": gene, "seed": seed,
                })
            else:  # LRP5
                variant_type = rng.choice(["OPPG_biallelic_LOF", "OPPG_biallelic_LOF", "HBM_G171V", "HBM_other_GOF"])
                oppg = variant_type == "OPPG_biallelic_LOF"
                onset_age = rng.randint(0, 2) if oppg else rng.randint(20, 50)
                age_at_dx = onset_age + rng.randint(0, 3)
                dx_delay = max(0, (age_at_dx - onset_age) * 12 + rng.randint(0, 18))
                visual_impaired = oppg and rng.random() < 0.65
                patients.append({
                    "patient_id": f"LRP5-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": variant_type,
                    "visual_impairment": visual_impaired,
                    "bisphosphonate_treated": oppg and rng.random() < 0.7,
                    "gene": gene, "seed": seed,
                })
        gene_data["patients"] = patients


_generate_patients()


def overview():
    all_delays = [
        p.get("dx_delay_months", 0)
        for g in BONE_DISEASE_GENES for p in g["patients"]
    ]
    all_ages = [
        p.get("onset_age", 5)
        for g in BONE_DISEASE_GENES for p in g["patients"]
    ]
    genes = []
    for idx, g in enumerate(BONE_DISEASE_GENES):
        delays = [p.get("dx_delay_months", 0) for p in g["patients"]]
        ages = [p.get("onset_age", 5) for p in g["patients"]]
        genes.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "key_alerts": g["key_alerts"],
            "n_patients": len(g["patients"]),
        })
    return {
        "atlas": "Hereditary-Metabolic-Bone-Disease-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Metabolic Bone Disease Atlas — "
            "COL1A1 / COL1A2 / ALPL / PHEX / FGF23 / ACVR1 / RUNX2 / LRP5 — "
            "320 Patients (8×40, Seeds 1750–1757)"
        ),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": sum(len(g["patients"]) for g in BONE_DISEASE_GENES),
        "aggregate_stats": {
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "genes_covered": len(BONE_DISEASE_GENES),
            "patients_per_gene": 40,
        },
        "genes": genes,
        "top_alerts": [
            "COL1A1-BISPHOSPHONATES-GOLD-STANDARD-OI: pamidronate IV cycling is the gold standard for paediatric OI type 1/4 — blue sclerae + fracture history = COL1A1/COL1A2 panel urgently; dental care and annual audiogram from age 20 are mandatory monitoring",
            "COL1A2-IN-UTERO-FRACTURES-NEONATAL-EMERGENCY: OI type 3 presents at birth — neonatal team MUST know fracture precautions; basilar invagination annual MRI from age 5; Fassier-Duval rodding at specialist OI centre only",
            "ALPL-PREMATURE-TEETH-LOSS-<5y-PATHOGNOMONIC: early deciduous tooth loss (root intact, before age 5) = HPP until proven otherwise; measure ALP immediately; asfotase alfa is FDA-approved ERT with proven survival benefit in infantile HPP",
            "PHEX-DENTAL-ABSCESS-WITHOUT-CARIES-XLH: dental abscess WITHOUT caries in child with rickets = XLH (PHEX); burosumab anti-FGF23 FDA 2018 is now first-line over conventional phosphate + calcitriol (fewer side effects, better adherence)",
            "FGF23-IRON-DEFICIENCY-TRIGGERS-ADHR: ADHR severity is iron-dependent; iron repletion alone can resolve a flare; measure ferritin in all hypophosphataemic rickets of uncertain cause; adult-onset ADHR often triggered by pregnancy/lactation iron depletion",
            "ACVR1-NO-BIOPSY-NO-IM-INJECTION-ABSOLUTE: FOP R206H (97%) — any biopsy or IM injection triggers catastrophic irreversible HO; palovarotene FDA 2023 is first approved therapy; short great toes at birth = FOP diagnosis before first flare",
            "RUNX2-DENTAL-SURGERY-MANDATORY-CCD: 10–40 supernumerary teeth in CCD will NOT erupt spontaneously; surgical extraction at age 7–9 mandatory to prevent cysts and permanent crowding; intelligence is ALWAYS normal in CCD",
            "LRP5-OPPG-VITREOUS-REMNANTS-NEONATAL-URGENT: OPPG (biallelic LRP5 LOF) causes vitreous fibrovascular remnants → retinal detachment → blindness if untreated; neonatal ophthalmology review mandatory within weeks of confirmed diagnosis; romosozumab mechanism exploits LRP5 WNT pathway",
        ],
    }


def breakdown():
    result = []
    for idx, g in enumerate(BONE_DISEASE_GENES):
        delays = [p.get("dx_delay_months", 0) for p in g["patients"]]
        ages = [p.get("onset_age", 5) for p in g["patients"]]
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
            "computed": {
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def definitions():
    return {
        "concepts": {
            "FGF23-Mediated Phosphate-Wasting — The Unified Pathway": (
                "The hereditary hypophosphataemic disorders (XLH, ADHR, ARHR) all converge on "
                "excess bioactive FGF23, which acts on kidney proximal tubule via FGFR1/αKlotho. "
                "NORMAL AXIS: "
                "Dietary phosphate load → osteocytes/osteoblasts secrete FGF23 → "
                "renal NaPi-IIa/IIc downregulation → ↑ urinary phosphate excretion → "
                "↓ CYP27B1 (1α-hydroxylase) → ↓ 1,25-VitD → ↓ intestinal phosphate absorption. "
                "PATHOLOGY: "
                "XLH (PHEX LOF): impaired ASARM/DMP1 inactivation → FGF23 OVERPRODUCED by osteocytes; "
                "ADHR (FGF23 R176/179 GOF): FGF23 RESISTANT TO CLEAVAGE → intact FGF23 accumulates; "
                "ARHR1 (DMP1 LOF): DMP1 normally reduces FGF23 production → LOF → excess FGF23; "
                "TIO: tumour secretes excess FGF23 → identical biochemistry, acquired. "
                "BIOCHEMICAL SIGNATURE (all): "
                "↓ serum Pi + ↓ TmP/GFR + ↑ intact FGF23 + ↓/N 1,25-VitD + ↑/N 25-OH-VitD "
                "(note: 25-VitD is NOT low in FGF23-disorders — this distinguishes from nutritional). "
                "TREATMENT RATIONALE: "
                "Burosumab blocks FGF23 at the ligand level → normalises renal phosphate handling; "
                "works regardless of upstream cause (XLH, ADHR, TIO); "
                "conventional (phosphate + calcitriol) bypasses FGF23 but causes nephrocalcinosis. "
                "DIAGNOSTIC TRAP: serum phosphate may appear low-normal in adults on conventional therapy; "
                "always check TmP/GFR as functional marker (tubular phosphate threshold); "
                "intact FGF23 (Kainos immunotopometric assay) >30 pg/mL confirms FGF23 excess."
            ),
            "Hypophosphatasia — ALPL Enzyme Biochemistry and Substrate Accumulation": (
                "Tissue-nonspecific alkaline phosphatase (TNSALP) hydrolyses three natural substrates: "
                "(1) Inorganic pyrophosphate (PPi): PPi is a potent mineralisation inhibitor; "
                "TNSALP deficiency → PPi accumulates extracellularly at bone mineralisation front → "
                "hydroxyapatite crystal deposition is blocked → osteomalacia/rickets; "
                "PPi also accumulates in joint spaces → calcium pyrophosphate deposition (CPPD, pseudo-gout) "
                "in adult HPP — seronegative arthritis picture; "
                "(2) Phosphoethanolamine (PEA): accumulates in urine of HPP patients; "
                "elevated urine PEA is a biochemical marker of HPP (not universally measured but specific); "
                "(3) Pyridoxal-5-phosphate (PLP — active form of vitamin B6): "
                "PLP is dephosphorylated by TNSALP at the cell surface → "
                "TNSALP deficiency → extracellular PLP accumulates → but pyridoxal (the dephosphorylated "
                "form) cannot be made → pyridoxal cannot cross BBB → CNS PLP deficiency → "
                "GABA synthesis impaired (PLP is cofactor for glutamic acid decarboxylase) → "
                "neonatal seizures clinically resembling B6-dependent epilepsy; "
                "IV pyridoxine treats seizures symptomatically; asfotase alfa resolves by normalising PLP. "
                "SERUM ALP INTERPRETATION: "
                "ALP is normally very high in children (bone ALP from growth); "
                "'low-normal' in a child on standard laboratory ranges may be pathologically low; "
                "always use paediatric ALP reference intervals; "
                "adult reference ALP 40–140 IU/L — any value <30 IU/L in adult = investigate HPP; "
                "ALP may transiently rise during fracture healing — do not dismiss HPP on a single normal. "
                "ASFOTASE ALFA: "
                "Recombinant human TNSALP fused to deca-aspartate (hydroxyapatite-binding) + Fc region; "
                "targets bone mineral surface → local PPi hydrolysis → mineralisation restored; "
                "given 3×/week or 6×/week SC; clinical trials demonstrated >70 percentage point improvement "
                "in survival for infantile HPP; dramatically improves radiological mineralisation."
            ),
            "Osteogenesis Imperfecta — COL1A1 vs COL1A2 Pathomechanism": (
                "Type I collagen (the main structural protein of bone, skin, tendon) consists of "
                "two α1(I) chains (encoded by COL1A1) + one α2(I) chain (encoded by COL1A2). "
                "PATHOMECHANISTIC DIFFERENCE: "
                "COL1A1 NULL ALLELE (haploinsufficiency, OI type 1): "
                "Nonsense/frameshift → mRNA NMD; only 1 functional COL1A1 allele → "
                "50% of normal collagen I quantity, but QUALITY IS NORMAL; "
                "bones are undermineralised and fragile due to reduced matrix, but "
                "collagen fibres that are present are normal in structure; "
                "CLINICAL CORRELATE: mild-moderate phenotype, good prognosis, ambulatory; "
                "COL1A1/2 GLYCINE SUBSTITUTION (dominant-negative, OI type 2/3/4): "
                "Glycine is the smallest amino acid — ESSENTIAL for the tight Gly-X-Y triple helix repeat; "
                "any substitution → disrupts triple helix folding → "
                "the abnormal chain is synthesised but POISONS the heterotrimer; "
                "collagen containing the abnormal chain is over-post-translationally modified "
                "(excess hydroxylation/glycosylation as helix folding is delayed) → "
                "secreted collagen is structurally abnormal → impaired fibril assembly → "
                "CLINICAL CORRELATE: severe-to-lethal phenotype, progressive deformity; "
                "SEVERITY GRADIENT BY CHAIN + POSITION: "
                "α1(I) Gly substitutions generally more severe than α2(I) substitutions; "
                "C-terminal substitutions more severe than N-terminal (C-terminal initiates helix folding); "
                "TREATMENT IMPLICATION: bisphosphonate response is similar regardless of COL1A1 vs COL1A2; "
                "emerging: setrusumab (anti-sclerostin) Phase 2 in OI types 1/3/4 — increases bone formation; "
                "gene therapy: OI type 1 (replace null allele) conceptually simpler than dominant-negative."
            ),
            "Fibrodysplasia Ossificans Progressiva — ACVR1 GOF and Activin A Neomorphism": (
                "FOP is caused by gain-of-function variants in ACVR1 (ALK2), primarily R206H (~97%). "
                "NORMAL ACVR1: "
                "ACVR1 is a type I BMP receptor (serine-threonine kinase); activated by BMP2/4/7; "
                "in the resting state, FKBP12 binds the GS domain and inhibits kinase; "
                "BMP ligand binding releases FKBP12 and activates SMAD1/5/9 → osteogenesis; "
                "Activin A (INHBA) INHIBITS wild-type ACVR1 (acts as a trap, preventing BMP signalling); "
                "FOP R206H MUTATION: "
                "R206H disrupts FKBP12 binding → partial constitutive activation; "
                "CRITICALLY: R206H ACVR1 is ACTIVATED (not inhibited) by activin A → "
                "activin A, normally a suppressor, now DRIVES osteogenic signalling via mutant ACVR1 → "
                "heterotopic ossification in soft tissue; "
                "TRIGGER MECHANISM: "
                "Soft tissue trauma (including IM injection, biopsy, viral illness-related inflammation) → "
                "local activin A release (immune cells) → activates R206H ACVR1 on muscle progenitors → "
                "SMAD1/5/9 → ectopic bone formation at the site; "
                "GARETOSMAB (anti-activin A, Regeneron): blocks activin A → "
                "prevents aberrant ACVR1 activation → reduces HO volume (Phase 2 LUMINA-1 positive); "
                "PALOVAROTENE (RARγ agonist): retinoic acid receptor γ activation inhibits chondrogenic "
                "condensation step in the HO pathway → reduces new HO volume; FDA approved 2023; "
                "PRACTICAL RULE: in any patient with known FOP (or suspected — short great toes + flare): "
                "ZERO IM INJECTIONS, ZERO BIOPSIES, ZERO HO SURGERY → "
                "these universally trigger catastrophic new HO at the procedure site."
            ),
            "WNT/LRP5 Pathway and Sclerostin Biology": (
                "The WNT/β-catenin (canonical WNT) pathway is the principal anabolic signalling system "
                "in bone: osteoblast differentiation, proliferation, and survival. "
                "PATHWAY: "
                "WNT ligand binds Frizzled + LRP5/LRP6 co-receptor complex → "
                "phosphorylation of LRP5/6 intracellular tail → "
                "Dishevelled activation → GSK-3β/CK1/Axin/APC destruction complex INHIBITED → "
                "β-catenin NOT phosphorylated → not ubiquitinated → not degraded → "
                "β-catenin accumulates → nuclear translocation → TCF/LEF transcription factor complex → "
                "target genes: bone formation (RUNX2, SP7/osterix, osteocalcin). "
                "SCLEROSTIN (SOST protein): "
                "Expressed exclusively by osteocytes; Wnt signalling antagonist; "
                "binds LRP5/LRP6 co-receptors → blocks Frizzled/LRP assembly → no WNT signal → "
                "osteoblast activity suppressed; "
                "Sclerostin production is suppressed by mechanical loading → "
                "explains why exercise increases bone density via WNT pathway. "
                "ROMOSOZUMAB (Evenity, Amgen/UCB): "
                "Anti-sclerostin monoclonal antibody (IgG2); FDA approved 2019 (postmenopausal OP); "
                "blocks sclerostin binding to LRP5/LRP6 → WNT signalling restored → "
                "anabolic AND antiresorptive effect; "
                "12 months treatment (cardiovascular box warning — higher CV events vs alendronate); "
                "LRP5 GOF (HBM): these patients have constitutively active LRP5 → "
                "sclerostin-resistance → naturally elevated bone mass → "
                "studying HBM families led to sclerostin as therapeutic target; "
                "LRP5 LOF (OPPG): WNT signalling absent → severe childhood osteoporosis + vitreous vascular "
                "regression failure; anti-sclerostin is under investigation (may restore residual signalling)."
            ),
        },
        "pharmacological_distinctions": [
            "Pamidronate IV (standard OI protocol) — COL1A1/COL1A2 OI types 1/3/4: 3-day IV cycle every 4–6 months in children; 60 mg/day × 3 days (weight-adjusted); cycle-related fever day 1; measurable BMD improvement and fracture reduction; first-dose acute phase reaction (fever, myalgia) — premedicate with paracetamol; DXA 'zebra lines' confirm cycling and are normal",
            "Zoledronic acid IV (adult OI / XLH-enthesopathy) — 5 mg IV once yearly; preferred over pamidronate in adolescents/adults for adherence; osteonecrosis of jaw risk (rare in OI — lower risk than oncology doses); dental check before starting; renal function check (hold if eGFR <35 mL/min)",
            "Asfotase alfa (STRENSIQ, Alexion) — HPP perinatal/infantile/juvenile: first ERT for a metabolic bone disease (FDA/EMA 2015); SC 3×/week (1 mg/kg) or 6×/week (0.5 mg/kg per dose); injection site reactions; monitor serum ALP (should rise from near-zero but not overshoot); NOT approved for adult-only HPP (off-label in adults); enrol in MINERALS registry",
            "Burosumab (CRYSVITA, Ultragenyx/Kyowa Kirin) — XLH (PHEX) and ADHR (FGF23): anti-FGF23 IgG1; SC every 2 weeks; monitor TmP/GFR, serum Pi, 1,25-VitD, PTH, urinary calcium; STOP phosphate salts and calcitriol 24h before first dose; superior to conventional therapy for phosphate normalisation and reduced nephrocalcinosis",
            "Palovarotene (SOHONOS, Ipsen) — FOP (ACVR1): retinoid receptor γ (RARγ) agonist; oral; FDA 2023 first approved FOP therapy; baseline 5 mg/day + flare 20 mg/day × 4 weeks then 10 mg/day × 8 weeks when flare begins; teratogenic (FDA category X for pregnancy) — mandatory contraception females ≥12 y; premature epiphyseal closure monitoring in growing children",
            "Garetosmab (anti-activin A, Regeneron) — FOP investigational: Phase 2 LUMINA-1 showed significant reduction in new HO volume; anti-activin A IgG4; IV monthly; not yet approved; enrol eligible FOP patients in trials; complementary to palovarotene (different mechanism)",
            "Denosumab (Prolia/XGEVA) — OI (off-label, second-line): RANK-L inhibitor; anti-resorptive; SC 60 mg every 6 months; CRITICAL: NEVER abruptly discontinue — rebound hypercalcaemia + multiple vertebral fractures documented in OI children; must bridge to bisphosphonate before stopping; not first-line in OI",
            "Romosozumab (Evenity, Amgen) — OPPG (investigational) / postmenopausal osteoporosis (approved): anti-sclerostin IgG2; 210 mg SC monthly × 12 months; FDA 2019 for postmenopausal OP; CV box warning (do not use if prior MI/stroke in last year); exploits LRP5 pathway biology; being studied in OPPG and OI",
            "Oral phosphate salts + calcitriol — XLH/ADHR (conventional, now largely superseded by burosumab): requires 4–6 daily doses of phosphate (GI intolerance high); calcitriol 20–30 ng/kg/day in 2 divided doses; nephrocalcinosis risk increases with combined therapy; hyperparathyroidism complication; SWITCH to burosumab preferred for all new paediatric patients with XLH",
        ],
        "key_standards": [
            "OI Care Standards (OI Foundation / OI European Consortium): bisphosphonate cycling per weight-based protocol; annual DXA from diagnosis; multidisciplinary team (metabolic bone, orthopaedics, physiotherapy, audiology, ophthalmology, dentistry, genetics); Fassier-Duval rodding at certified OI orthopaedic centre; basilar invagination MRI annually in OI type 3/4",
            "XLH Treatment Guidelines (Endocrine Society 2022 / Pediatric Endocrine Society): burosumab preferred first-line for all children/adolescents; measure intact FGF23 (Kainos assay); TmP/GFR as monitoring marker; dental abscess → antibiotic + dental referral (NOT extraction without cause); transition from conventional to burosumab: 24h phosphate/calcitriol washout; nephrocalcinosis follow-up annual renal ultrasound",
            "HPP Registry (Strensiq PASS / PEALS / MINERALS): all patients on asfotase alfa enrolled; safety and efficacy monitoring; ALP normalisation tracking; injection site reaction reporting; paediatric patients prioritised for access; adult HPP access variable by jurisdiction",
            "FOP Clinical Care Standards (IFOPA / FOP Collaborative): FOP emergency card to be carried by patient at all times; all vaccinations subcutaneous; jaw trismus protocol for dental emergencies; annual pulmonary function tests (restrictive disease monitoring); emergency anaesthetic protocol (difficult airway); no dental inferior alveolar nerve block without specialist guidance",
            "Cleidocranial Dysplasia Dental Protocol: OPG from age 4–5 years (identify supernumerary teeth); staged dental extractions by age 7–9 (primary + supernumerary) to allow permanent tooth eruption space; orthodontic treatment concurrent; CBCT (cone-beam CT) for surgical planning; annual OPG during treatment phase; refer to oral-maxillofacial surgeon at diagnosis",
            "OPPG Ophthalmology Standard: all biallelic LRP5 LOF neonates — ophthalmology review within 4–6 weeks of birth; fundoscopy ± RetCam imaging for vitreous fibrovascular remnants; vitrectomy if remnants present and visual axis threatened; slit-lamp and fundoscopy annually thereafter; genetic counselling for siblings (25% recurrence risk AR)",
            "HPP Adult Diagnosis Standard: measure serum ALP with age-sex specific reference interval; plasma PLP if ALP borderline; 24h urine phosphoethanolamine in specialist centres; any adult with unexplained stress fractures + low/low-normal ALP + elevated plasma PLP → HPP diagnosis; asfotase alfa access in adults with significant morbidity (fractures, respiratory, mobility)",
            "ADHR (FGF23 GOF) Iron Protocol: measure serum ferritin, serum iron, transferrin saturation in all hypophosphataemia of uncertain or variable severity; IV iron (ferric carboxymaltose or iron sucrose) for iron-deficient ADHR — may normalise FGF23 without burosumab; recheck intact FGF23 4–6 weeks post-iron repletion; pregnancy/lactation monitoring for ADHR flares",
        ],
    }
