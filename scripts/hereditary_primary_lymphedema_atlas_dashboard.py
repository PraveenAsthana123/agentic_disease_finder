#!/usr/bin/env python3
"""Hereditary-Primary-Lymphedema-Atlas — Complete 8-Gene Hereditary Primary Lymphedema Atlas
FLT4    (VEGFR3; 1363 aa; 5q35.3; AD;
          Milroy disease / primary congenital lymphedema type 1 (PCL-1); onset at birth;
          VEGFR3 tyrosine kinase domain missense mutations — dominant negative on VEGF-C signalling;
          Lower limb edema bilateral; chylothorax; hydrops fetalis (prenatal);
          Genotype confirms Milroy — no distichiasis distinguishes from FOXC2) ·
FOXC2   (Forkhead Box C2; 501 aa; 16q24.1; AD;
          Lymphedema-distichiasis syndrome (LDS); DISTICHIASIS pathognomonic — extra eyelashes
          from meibomian glands causing corneal irritation; lymphedema pubertal onset (NOT neonatal);
          Congenital heart defects in 7%; ptosis; varicose veins; ophthalmology MANDATORY) ·
PROX1   (Prospero Homeobox 1; 737 aa; 1q32.3; AD;
          Hypotrichosis-lymphedema-telangiectasia syndrome (HLTS); master regulator of lymphatic
          endothelial cell identity; triad: sparse/absent hair + lower limb lymphedema +
          skin/mucous membrane telangiectasia; variable expressivity; haploinsufficiency) ·
GJC2    (Connexin 47; 436 aa; 1q42.13; AR;
          Late-onset primary lymphedema type 1C (AR biallelic); gap junction channel;
          M283T and R260C hotspot variants; peripheral neuropathy overlap (SPG44 AR);
          Lymphedema onset late teens to 20s; CNS involvement if AR compound heterozygous) ·
SOX18   (SRY-Box 18; 384 aa; 20q13.33; AD / AR;
          Hypotrichosis-lymphedema-telangiectasia-renal defect (HLTRS) AD; severe HLTS AR;
          Ragged HMG-box mutations dominant negative on VEGF-C/VEGFR3 transcription;
          Dry/sparse hair from birth; lymphedema progressive; varicose veins;
          Renal anomalies in some; AR form: combined severe phenotype) ·
CCBE1   (Collagen and Calcium Binding EGF Domains 1; 429 aa; 18q21.32; AR;
          Hennekam lymphangiectasia-lymphedema syndrome type 1 (HLS1); CCBE1 activates VEGF-C
          via ADAMTS3 protease — loss → impaired VEGFR3 signalling → generalised lymphatic dysplasia;
          Intestinal lymphangiectasia → protein-losing enteropathy → hypoalbuminaemia;
          Intellectual disability ~50%; facial dysmorphism; lymphedema from birth; low albumin/lymphocytes) ·
KIF11   (Kinesin Family Member 11 / Eg5; 1056 aa; 10q23.33; AD;
          Microcephaly with or without chorioretinopathy, lymphedema, or intellectual disability (MCLMR);
          Kinesin spindle protein — bipolar spindle assembly; de novo / AD missense;
          MICROCEPHALY + CHORIORETINOPATHY + LYMPHEDEMA triad; ophthalmology mandatory;
          Lymphedema mild/intermittent; macular hypoplasia / absence on fundoscopy) ·
ADAMTS3 (A Disintegrin-Like and Metalloproteinase with Thrombospondin Motifs 3; 1232 aa; 4q13.3; AR;
          Hennekam lymphangiectasia-lymphedema syndrome type 3 (HLS3); VEGF-C processing enzyme;
          Biallelic LOF → impaired VEGF-C activation → VEGFR3 hyposignalling;
          Phenocopies CCBE1 — intestinal lymphangiectasia + protein-losing enteropathy + ID;
          Test CCBE1 AND ADAMTS3 simultaneously for Hennekam phenotype; MCT diet mandatory)
320-patient aggregate cohort (8 × 40, seeds 1454–1461)
"""

import random

SEED_BASE = 1454

LYMPHEDEMA_GENES = [
    # ── FLT4 (VEGFR3) — Primary congenital lymphedema / Milroy disease ──
    {
        "gene": "FLT4",
        "protein": "Vascular Endothelial Growth Factor Receptor 3 (VEGFR3) Lymphatic Tyrosine Kinase",
        "alias": (
            "FLT4; VEGFR3; OMIM gene 136352; Milroy disease OMIM 153100; 5q35.3; 1363 aa; ~152 kDa; "
            "Primary congenital lymphedema type 1 (PCL-1; Milroy disease); autosomal dominant; "
            "VEGFR3 tyrosine kinase domain missense mutations — dominant negative on VEGF-C/D signalling; "
            "Onset at birth or in utero — bilateral lower limb pitting oedema, often dorsal foot; "
            "Hydrops fetalis in severely affected foetuses; chylothorax rare; pleural effusions; "
            "Hydrocele in affected males (65%); papillomatosis; foot verrucae; upslanting toenails; "
            "NO distichiasis (distinguishes from FOXC2); NO intellectual disability; NO intestinal lymphangiectasia; "
            "Lymphoscintigraphy — absent lower limb lymphatics is characteristic; "
            "Compression hosiery lifelong; no curative therapy; annual limb volume assessment"
        ),
        "aa": "1363 aa",
        "kDa": "~152 kDa",
        "locus": "5q35.3",
        "omim_gene": 136352,
        "omim_disease": 153100,
        "inheritance": "AD — dominant negative missense in TK domain; congenital onset",
        "gene_class": (
            "FLT4 encodes VEGFR3, a receptor tyrosine kinase (RTK) of the VEGF receptor family expressed "
            "selectively on lymphatic endothelial cells (LECs) in adults. VEGFR3 mediates the primary "
            "growth, survival, and proliferation signal for LECs in response to its ligands VEGF-C and "
            "VEGF-D. Milroy disease pathogenic variants cluster in the kinase domain (exons 17–26) and "
            "act as dominant negatives — heterodimers with wild-type VEGFR3 are kinase-dead, abrogating "
            "VEGF-C signalling in developing lymphatics. Loss of VEGFR3 signal → failure of subcutaneous "
            "lymphatic capillary formation → primary congenital lymphoedema. Disease is present at birth "
            "(or in utero: hydrops fetalis). Classic phenotype: bilateral dorsal foot/lower limb pitting "
            "oedema, hydrocele in males, verrucous changes. Lymphoscintigraphy shows absent or grossly "
            "hypoplastic lower limb lymphatics. Penetrance ~85–90%; some carriers asymptomatic. >50 "
            "pathogenic FLT4 variants reported."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("FLT4 TK-domain missense (exon 17–26 hotspot) — dominant negative, most common", 0.55),
            ("FLT4 kinase-domain missense — rare variant, same mechanism", 0.20),
            ("FLT4 extracellular domain missense — reduced VEGF-C binding", 0.10),
            ("FLT4 frameshift — haploinsufficiency, milder phenotype", 0.08),
            ("FLT4 splice-site — partial kinase domain deletion", 0.07),
        ],
        "age_onset_years_range": (0, 1),
        "sex_ratio_M": 0.55,
        "rates": {
            "bilateral_lower_limb_oedema_congenital":   0.97,
            "dorsal_foot_oedema_predominant":           0.90,
            "hydrocele_males":                          0.65,
            "upslanting_toenails":                      0.50,
            "verrucous_papillomatosis_feet":            0.30,
            "hydrops_fetalis_severe_prenatal":          0.08,
            "chylothorax_pleural_effusion":             0.05,
            "no_distichiasis":                          1.00,
            "lymphoscintigraphy_absent_lymphatics":     0.88,
            "compression_hosiery_lifelong":             0.92,
            "annual_limb_volume_assessment":            0.85,
            "positive_family_history_ad":               0.80,
        },
        "critical_alerts": [
            "FLT4-NO-DISTICHIASIS: Absence of distichiasis distinguishes Milroy (FLT4) from Lymphedema-Distichiasis (FOXC2) — examine eyelids in ALL primary lymphoedema",
            "FLT4-CONGENITAL-ONSET: Present at birth or in utero — neonatal bilateral lower limb oedema should trigger FLT4 sequencing (after excluding hydrops aetiology)",
            "FLT4-HYDROCELE-MALES: Hydrocele in 65% of affected males — surgical drainage rarely needed; diagnosis lymphoedema-related, NOT idiopathic",
            "FLT4-LYMPHOSCINTIGRAPHY: Absent or grossly hypoplastic lower limb lymphatics on lymphoscintigraphy — characteristic finding, confirms diagnosis when genetic testing delayed",
            "FLT4-COMPRESSION-LIFELONG: Compression hosiery (20–40 mmHg class II) lifelong — no curative therapy; annual limb volume + skin integrity assessment mandatory",
        ],
        "key_ddx_rules": [
            "Bilateral lower limb oedema at birth + no distichiasis → FLT4/VEGFR3 (Milroy) first (most common hereditary congenital lymphedema)",
            "Congenital lymphedema + hydrocele in male + upslanting toenails → FLT4 virtually diagnostic",
            "FLT4 negative congenital lymphoedema → consider CCBE1 / ADAMTS3 (Hennekam) if intestinal lymphangiectasia; PROX1 if hypotrichosis",
        ],
    },

    # ── FOXC2 — Lymphedema-distichiasis syndrome ──
    {
        "gene": "FOXC2",
        "protein": "Forkhead Box Protein C2 Lymphatic and Venous Valve Transcription Factor",
        "alias": (
            "FOXC2; MFH1; OMIM gene 602402; Lymphedema-distichiasis OMIM 153400; 16q24.1; 501 aa; ~55 kDa; "
            "Lymphedema-distichiasis syndrome (LDS); autosomal dominant; high penetrance; "
            "DISTICHIASIS (accessory eyelashes from metaplastic meibomian glands) pathognomonic; "
            "Distichiasis present in virtually all mutation carriers — corneal irritation, photophobia, epiphora; "
            "Lymphedema predominantly pubertal onset (NOT congenital — key DDx from Milroy); "
            "Lower limb lymphoedema + varicose veins; congenital heart defects in ~7% (ASD/VSD/tetralogy); "
            "Ptosis in 30%; cleft palate rare; spinal extradural lymphatic cysts; "
            "Ophthalmology referral MANDATORY — corneal abrasion, erosion, secondary scarring risk; "
            "FOXC2 also controls venous valve development — varicose veins in 50%"
        ),
        "aa": "501 aa",
        "kDa": "~55 kDa",
        "locus": "16q24.1",
        "omim_gene": 602402,
        "omim_disease": 153400,
        "inheritance": "AD — haploinsufficiency; near-complete penetrance for distichiasis",
        "gene_class": (
            "FOXC2 encodes a forkhead box transcription factor essential for the development and "
            "maintenance of lymphatic valves and venous valves. FOXC2 binds the PROX1-driven lymphatic "
            "endothelial cell gene programme and is critical for the late stages of lymphatic valve "
            "morphogenesis. Haploinsufficiency results in valve hypoplasia → lymphatic reflux and "
            "progressive lower limb lymphoedema. FOXC2 also regulates meibomian gland identity; "
            "virtually all heterozygotes develop distichiasis (accessory eyelashes) — this is the most "
            "diagnostically useful feature, present at birth or early childhood before lymphoedema appears. "
            "Lymphoedema classically begins at puberty (oestrogen and gravity trigger), unlike FLT4 "
            "(congenital). Variant spectrum: truncating (frameshift/nonsense) ~55%, missense ~40% — all "
            "haploinsufficiency. Varicose veins are nearly universal (venous valve failure). Ophthalmology "
            "mandatory: untreated distichiasis → corneal scarring → vision loss."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("FOXC2 frameshift — haploinsufficiency, premature stop, most common", 0.35),
            ("FOXC2 nonsense — premature termination, NMD", 0.20),
            ("FOXC2 missense (forkhead domain) — DNA binding loss", 0.25),
            ("FOXC2 splice-site — exon skipping", 0.10),
            ("FOXC2 large deletion — MLPA detected", 0.10),
        ],
        "age_onset_years_range": (10, 30),
        "sex_ratio_M": 0.42,
        "rates": {
            "distichiasis_pathognomonic":            0.97,
            "corneal_irritation_epiphora":           0.75,
            "lymphedema_lower_limb":                 0.94,
            "pubertal_onset_lymphedema":             0.80,
            "varicose_veins":                        0.49,
            "ptosis":                                0.30,
            "congenital_heart_defect":               0.07,
            "spinal_extradural_lymphatic_cyst":      0.12,
            "cleft_palate_rare":                     0.04,
            "ophthalmology_referral_required":       0.97,
            "compression_hosiery":                   0.88,
            "family_history_ad":                     0.82,
        },
        "critical_alerts": [
            "FOXC2-DISTICHIASIS-PATHOGNOMONIC: Extra eyelashes from meibomian glands virtually UNIVERSAL in FOXC2 carriers — slit-lamp exam MANDATORY in all primary lymphoedema; diagnoses condition before lymphoedema onset",
            "FOXC2-CORNEAL-RISK: Untreated distichiasis → corneal abrasion → erosion → scarring → vision loss — ophthalmology referral at diagnosis; electrolysis/cryotherapy/laser ablation of lashes",
            "FOXC2-NOT-CONGENITAL: Lymphedema onset pubertal (NOT at birth) — key DDx from Milroy (FLT4); neonatal lymphedema with distichiasis = FOXC2 atypical variant",
            "FOXC2-CARDIAC-SCREEN: Congenital heart defects (ASD/VSD/tetralogy) in 7% — echo at diagnosis in paediatric cases",
            "FOXC2-VARICOSE-VEINS: Venous valve failure → varicose veins ~50% — compression essential; sclerotherapy/surgery not contraindicated but discuss lymphatic impairment",
        ],
        "key_ddx_rules": [
            "Lymphedema + distichiasis → FOXC2 — virtually diagnostic (distichiasis is absent in all other primary lymphedema genes)",
            "Young adult onset lower limb lymphedema + varicose veins + family history → FOXC2 before other genes",
            "FOXC2 negative but distichiasis present → re-check FOXC2 (large deletions, MLPA); extremely rare: FOXC2 non-coding variants",
        ],
    },

    # ── PROX1 — Hypotrichosis-lymphedema-telangiectasia ──
    {
        "gene": "PROX1",
        "protein": "Prospero Homeobox 1 Master Lymphatic Endothelial Cell Fate Regulator",
        "alias": (
            "PROX1; OMIM gene 601546; Hypotrichosis-lymphedema-telangiectasia OMIM 211900; 1q32.3; 737 aa; ~84 kDa; "
            "Hypotrichosis-lymphedema-telangiectasia syndrome (HLTS); autosomal dominant; "
            "PROX1 is the master transcription factor for lymphatic endothelial cell identity — "
            "reprogrammes venous ECs to lymphatic fate during embryogenesis; "
            "Triad: hypotrichosis (sparse/absent scalp/body hair) + lower limb lymphoedema + "
            "cutaneous and mucosal telangiectasia; variable expressivity within families; "
            "Haploinsufficiency → inadequate lymphatic specification → lymphatic hypoplasia"
        ),
        "aa": "737 aa",
        "kDa": "~84 kDa",
        "locus": "1q32.3",
        "omim_gene": 601546,
        "omim_disease": 211900,
        "inheritance": "AD — haploinsufficiency; variable expressivity; autosomal recessive (biallelic) severe",
        "gene_class": (
            "PROX1 (Prospero-related homeobox 1) is the master regulator of lymphatic endothelial cell "
            "(LEC) identity. During embryogenesis, PROX1 expression in venous endothelial cells of the "
            "cardinal vein reprogrammes them to LECs — initiating lymphatic sac formation. Without PROX1, "
            "no LECs form and the entire lymphatic system fails to develop. Haploinsufficiency in humans "
            "causes HLTS — a triad of hypotrichosis (sparse or absent scalp and body hair), lower limb "
            "lymphoedema (onset childhood to early adult), and mucocutaneous telangiectasia. The expressivity "
            "is highly variable: some carriers have all three features; others show only one. Telangiectasia "
            "may be subtle — check buccal mucosa, lips, conjunctiva, and skin. PROX1 also regulates hepatic "
            "progenitor fate and retinal development. Biallelic PROX1 loss is presumed lethal (complete "
            "lymphatic agenesis in mice). Diagnosis: sequencing ± del/dup."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("PROX1 missense (homeodomain) — lymphatic specification loss", 0.35),
            ("PROX1 frameshift — haploinsufficiency, NMD", 0.25),
            ("PROX1 nonsense — premature stop", 0.18),
            ("PROX1 splice-site — exon skipping, partial LOF", 0.12),
            ("PROX1 large deletion — rare, MLPA or chromosomal microarray", 0.10),
        ],
        "age_onset_years_range": (5, 25),
        "sex_ratio_M": 0.50,
        "rates": {
            "hypotrichosis_scalp_body":              0.85,
            "lower_limb_lymphedema":                 0.88,
            "mucocutaneous_telangiectasia":          0.75,
            "all_three_triad_features":              0.60,
            "childhood_or_early_adult_onset":        0.80,
            "variable_expressivity_intrafamilial":   0.70,
            "buccal_conjunctival_telangiectasia":    0.55,
            "compression_hosiery":                   0.80,
            "family_history_ad":                     0.75,
        },
        "critical_alerts": [
            "PROX1-TRIAD-INCOMPLETE: All three features (hypotrichosis + lymphedema + telangiectasia) present in only ~60% — examine hair, limbs, AND mucosae before ruling out PROX1",
            "PROX1-TELANGIECTASIA-SUBTLE: Check buccal mucosa, lips, conjunctiva, and fingernail beds — cutaneous telangiectasia may be missed on casual inspection",
            "PROX1-HYPOTRICHOSIS-FIRST: Sparse/absent hair often precedes lymphoedema — childhood hypotrichosis + family history → gene panel including PROX1",
            "PROX1-VARIABLE-EXPRESSIVITY: Intrafamilial variability extreme — a carrier parent may have only hypotrichosis while the proband has full triad; cascade test all at-risk relatives",
        ],
        "key_ddx_rules": [
            "Hypotrichosis + lymphedema + telangiectasia triad → PROX1 (first) / SOX18 (second) — both cause HLTS phenotype",
            "Lymphedema + sparse hair but NO telangiectasia → SOX18 more likely (variant phenotype); still test PROX1",
            "PROX1 negative HLTS → SOX18 sequencing mandatory (phenocopy); consider GJC2 if no hair/telangiectasia",
        ],
    },

    # ── GJC2 — Late-onset primary lymphedema (AR) / Hereditary spastic paraplegia SPG44 ──
    {
        "gene": "GJC2",
        "protein": "Gap Junction Gamma-2 Protein Connexin 47 Lymphatic Endothelial Gap Junction",
        "alias": (
            "GJC2; Cx47; OMIM gene 608803; Lymphedema late-onset OMIM 613480; SPG44 OMIM 613206; 1q42.13; 436 aa; ~47 kDa; "
            "Late-onset primary lymphedema type 1C (AR biallelic); autosomal recessive; "
            "GJC2/Connexin 47 forms gap junction channels between lymphatic endothelial cells; "
            "M283T and R260C hotspot variants in lymphedema; onset late teens to 30s; "
            "DUAL PHENOTYPE: biallelic LOF → leukodystrophy + spastic paraplegia (SPG44); "
            "Monoallelic missense → lymphedema only; screen for CNS in AR patients"
        ),
        "aa": "436 aa",
        "kDa": "~47 kDa",
        "locus": "1q42.13",
        "omim_gene": 608803,
        "omim_disease": 613480,
        "inheritance": "AR (lymphedema) biallelic missense; AD monoallelic missense (late-onset lymphedema); biallelic LOF = SPG44 leukodystrophy",
        "gene_class": (
            "GJC2 encodes Connexin 47 (Cx47), a gap junction protein expressed in lymphatic endothelial "
            "cells and oligodendrocytes. In lymphatics, GJC2 channels coordinate intercellular "
            "communication for lymphatic valve formation and endothelial barrier integrity. Biallelic "
            "pathogenic missense variants (especially M283T and R260C) cause late-onset primary "
            "lymphedema — typically bilateral lower limb oedema emerging in teenage or young adult years. "
            "Biallelic loss-of-function (truncating) variants cause the neurological disorder SPG44: "
            "leukodystrophy with hereditary spastic paraplegia, reflecting Cx47's role in "
            "oligodendrocyte gap junctions (panglial coupling). Heterozygous missense carriers may "
            "develop milder/late-onset lymphedema. Brain MRI is required in biallelic cases (especially "
            "compound heterozygotes with one LOF allele) to exclude leukodystrophy. NGS panel for "
            "primary lymphedema should include GJC2 with biallelic analysis."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("GJC2 M283T biallelic — commonest AR lymphedema hotspot", 0.30),
            ("GJC2 R260C biallelic — second hotspot AR lymphedema", 0.22),
            ("GJC2 compound heterozygous missense/missense — AR lymphedema", 0.20),
            ("GJC2 heterozygous missense — late-onset AD lymphedema", 0.15),
            ("GJC2 compound heterozygous missense/LOF — lymphedema ± CNS", 0.13),
        ],
        "age_onset_years_range": (14, 35),
        "sex_ratio_M": 0.45,
        "rates": {
            "bilateral_lower_limb_lymphedema":       0.95,
            "late_onset_teen_young_adult":           0.85,
            "ar_biallelic_confirmed":                0.72,
            "spg44_leukodystrophy_if_biallelic_lof": 0.15,
            "brain_mri_required_biallelic":          0.28,
            "heterozygous_mild_late_onset":          0.28,
            "consanguinity":                         0.25,
            "compression_hosiery":                   0.82,
        },
        "critical_alerts": [
            "GJC2-BIALLELIC-CNS-RISK: Compound heterozygotes with ≥1 truncating allele may develop SPG44 leukodystrophy — brain MRI MANDATORY in all biallelic GJC2 cases",
            "GJC2-M283T-HOTSPOT: M283T is the most common AR lymphedema variant — targeted testing cost-effective before full sequencing in late-onset bilateral lower limb lymphedema",
            "GJC2-DUAL-PHENOTYPE: GJC2 causes both primary lymphedema (AR missense) AND hereditary spastic paraplegia SPG44 (AR LOF) — clinical phenotype guides interpretation",
            "GJC2-AR-ANALYSIS: Late-onset primary lymphedema panel must include biallelic GJC2 analysis — heterozygous variants alone may not be pathogenic; confirm segregation",
        ],
        "key_ddx_rules": [
            "Late-onset bilateral lower limb lymphedema (teens–20s) + AR family history → GJC2 biallelic sequencing (M283T, R260C hotspots)",
            "Lymphedema + spastic paraplegia/leukodystrophy → biallelic GJC2 LOF (SPG44) — MRI white matter changes",
            "GJC2 negative late-onset lymphedema → FOXC2 (if distichiasis), PROX1/SOX18 (if hair/telangiectasia), FLT4 (if congenital onset)",
        ],
    },

    # ── SOX18 — Hypotrichosis-lymphedema-telangiectasia-renal defect (AD) / Severe form (AR) ──
    {
        "gene": "SOX18",
        "protein": "SRY-Box Transcription Factor 18 Lymphatic and Vascular Development Regulator",
        "alias": (
            "SOX18; OMIM gene 601618; HLTRS OMIM 607361; Severe AR form OMIM 614819; 20q13.33; 384 aa; ~44 kDa; "
            "Hypotrichosis-lymphedema-telangiectasia-renal defect syndrome (HLTRS); autosomal dominant; "
            "Ragged dominant negative mutations in HMG-box DNA-binding domain; "
            "Hypotrichosis (dry sparse hair from birth) + lymphedema + telangiectasia + renal anomalies; "
            "SOX18 drives PROX1 expression — loss impairs lymphatic specification; "
            "AR biallelic LOF = severe combined phenotype; varicose veins common; "
            "Renal anomalies in ~30% of AD HLTRS; annual renal ultrasound"
        ),
        "aa": "384 aa",
        "kDa": "~44 kDa",
        "locus": "20q13.33",
        "omim_gene": 601618,
        "omim_disease": 607361,
        "inheritance": "AD — ragged dominant negative HMG-box missense; AR biallelic LOF = severe HLTS",
        "gene_class": (
            "SOX18 encodes a high-mobility group (HMG) box transcription factor that acts upstream of "
            "PROX1 to initiate lymphatic endothelial cell specification from venous endothelium. SOX18 "
            "binds the PROX1 promoter and also drives expression of VEGFR3 (FLT4). Dominant missense "
            "variants in the HMG box (so-called 'ragged' variants — named after the ragged mouse model) "
            "act as dominant negatives: mutant SOX18 protein competes with wild-type for DNA binding but "
            "fails to transactivate target genes, and also sequesters wild-type SOX18 in non-productive "
            "complexes. The phenotype is HLTRS: hypotrichosis (dry, sparse, lusterless hair from birth "
            "or early infancy), lymphoedema (childhood to adult onset), telangiectasia (mucocutaneous), "
            "and renal anomalies (~30%) such as renal hypoplasia or vesico-ureteral reflux. AR biallelic "
            "LOF variants cause a more severe syndrome with all features pronounced. Varicose veins are "
            "common (venous valve contribution). Annual renal ultrasound is mandatory for all SOX18 "
            "mutation carriers. SOX18 and PROX1 phenocopy each other — test both in HLTS."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("SOX18 HMG-box ragged missense — dominant negative, most common AD", 0.45),
            ("SOX18 other missense (activation domain) — haploinsufficiency", 0.20),
            ("SOX18 frameshift — AR compound heterozygous, severe", 0.15),
            ("SOX18 nonsense — AR biallelic LOF severe / AD haploinsufficiency mild", 0.12),
            ("SOX18 splice-site — partial exon skipping", 0.08),
        ],
        "age_onset_years_range": (2, 20),
        "sex_ratio_M": 0.48,
        "rates": {
            "hypotrichosis_dry_sparse_hair":         0.88,
            "lower_limb_lymphedema":                 0.85,
            "mucocutaneous_telangiectasia":          0.72,
            "renal_anomalies":                       0.30,
            "varicose_veins":                        0.45,
            "ar_severe_combined_phenotype":          0.15,
            "annual_renal_ultrasound_required":      0.92,
            "family_history_ad":                     0.70,
            "compression_hosiery":                   0.78,
        },
        "critical_alerts": [
            "SOX18-RENAL-ANOMALIES: Renal hypoplasia / vesico-ureteral reflux in ~30% — renal ultrasound MANDATORY at diagnosis and annually; nephrology referral if abnormal",
            "SOX18-RAGGED-DOMINANT-NEGATIVE: HMG-box missense acts as dominant negative (not simple haploinsufficiency) — explains variable severity within families",
            "SOX18-PROX1-PHENOCOPY: SOX18 and PROX1 cause overlapping HLTS — test BOTH simultaneously in any hypotrichosis-lymphedema-telangiectasia phenotype",
            "SOX18-AR-SEVERE: Biallelic LOF causes severe combined phenotype — check parental carrier status; recurrence risk 25% per pregnancy for AR families",
            "SOX18-HAIR-FIRST: Dry/sparse/lusterless hair present from birth — in a child with 'hair problem' + family history of lymphedema, consider SOX18 before other workup",
        ],
        "key_ddx_rules": [
            "Hypotrichosis + lymphedema + telangiectasia + renal anomalies → SOX18 (HLTRS) — renal involvement increases SOX18 probability over PROX1",
            "HLTS phenotype + PROX1 negative → SOX18 sequencing (second most common HLTS gene)",
            "Sparse hair from birth + father with lymphedema + mother with telangiectasia → SOX18 dominant negative (ragged variant), request HMG-box targeted sequencing",
        ],
    },

    # ── CCBE1 — Hennekam lymphangiectasia-lymphedema syndrome type 1 ──
    {
        "gene": "CCBE1",
        "protein": "Collagen and Calcium Binding EGF Domains 1 VEGF-C Activator",
        "alias": (
            "CCBE1; OMIM gene 612753; Hennekam syndrome HLS1 OMIM 235510; 18q21.32; 429 aa; ~48 kDa; "
            "Hennekam lymphangiectasia-lymphedema syndrome type 1 (HLS1); autosomal recessive; "
            "CCBE1 is cofactor for ADAMTS3 protease — together they activate VEGF-C by N-terminal cleavage; "
            "Loss → pro-VEGF-C accumulates uncleaved → VEGFR3 hyposignalling → generalised lymphatic dysplasia; "
            "INTESTINAL LYMPHANGIECTASIA → protein-losing enteropathy → hypoalbuminaemia → secondary oedema; "
            "Intellectual disability ~50%; facial dysmorphism; lymphoedema from birth; "
            "Low plasma albumin + lymphocytes mandatory monitoring; MCT-enriched diet; "
            "Test CCBE1 AND ADAMTS3 simultaneously for Hennekam phenotype"
        ),
        "aa": "429 aa",
        "kDa": "~48 kDa",
        "locus": "18q21.32",
        "omim_gene": 612753,
        "omim_disease": 235510,
        "inheritance": "AR — biallelic LOF (frameshift/nonsense/missense); consanguinity common",
        "gene_class": (
            "CCBE1 encodes a secreted extracellular matrix protein with collagen repeats and EGF-like "
            "domains. CCBE1 functions as an essential cofactor for the metalloprotease ADAMTS3: the "
            "CCBE1–ADAMTS3 complex cleaves the N-terminal propeptide of pro-VEGF-C to generate the "
            "mature, VEGFR3-activating form of VEGF-C. Without CCBE1 (or ADAMTS3), pro-VEGF-C "
            "accumulates in the extracellular matrix but fails to activate VEGFR3, mimicking the "
            "phenotype of FLT4/VEGFR3 loss. The resulting generalised lymphatic dysplasia produces "
            "the Hennekam syndrome: lymphoedema and lymphangiectasia in multiple systems. Intestinal "
            "lymphangiectasia causes protein-losing enteropathy — hypoalbuminaemia, lymphopaenia, "
            "hypogammaglobulinaemia — and is the most life-threatening manifestation. Intellectual "
            "disability occurs in ~50% and is progressive. Distinctive facial dysmorphism (flat face, "
            "hypertelorism, epicanthal folds) aids clinical recognition. Consanguinity in >50% of "
            "reported families. Treatment: MCT-enriched diet, albumin supplementation; octreotide for "
            "severe intestinal lymphangiectasia; LMWH prophylaxis if immobile."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("CCBE1 frameshift — biallelic LOF, severe generalised lymphatic dysplasia", 0.35),
            ("CCBE1 nonsense — premature stop, NMD", 0.22),
            ("CCBE1 missense (EGF domain) — impaired ADAMTS3 interaction", 0.25),
            ("CCBE1 splice-site — exon skipping", 0.12),
            ("CCBE1 homozygous missense — consanguineous family", 0.06),
        ],
        "age_onset_years_range": (0, 2),
        "sex_ratio_M": 0.50,
        "rates": {
            "lymphedema_from_birth":                    0.95,
            "intestinal_lymphangiectasia":              0.85,
            "protein_losing_enteropathy":               0.80,
            "hypoalbuminaemia_low_albumin":             0.82,
            "lymphopaenia":                             0.75,
            "hypogammaglobulinaemia":                   0.60,
            "intellectual_disability":                  0.50,
            "facial_dysmorphism":                       0.80,
            "pulmonary_lymphangiectasia":               0.30,
            "pericardial_effusion":                     0.15,
            "consanguinity":                            0.55,
            "mct_diet_required":                        0.88,
            "albumin_supplementation":                  0.70,
            "annual_albumin_lymphocyte_monitoring":     0.90,
        },
        "critical_alerts": [
            "CCBE1-INTESTINAL-LYMPHANGIECTASIA: Protein-losing enteropathy → hypoalbuminaemia → secondary oedema — measure plasma albumin + lymphocyte count at diagnosis and every 6 months",
            "CCBE1-MCT-DIET-MANDATORY: Medium-chain triglyceride (MCT)-enriched diet bypasses intestinal lacteal absorption — start at diagnosis; dietitian referral mandatory",
            "CCBE1-TEST-ADAMTS3-SIMULTANEOUSLY: CCBE1 and ADAMTS3 are functional partners in VEGF-C activation — Hennekam phenotype → test BOTH genes at same time (panel sequencing)",
            "CCBE1-INTELLECTUAL-DISABILITY: ~50% have intellectual disability — neuropsychological assessment at diagnosis; early intervention programme from infancy",
            "CCBE1-OCTREOTIDE: Octreotide (somatostatin analogue) reduces lymphatic flow — use in severe intestinal lymphangiectasia unresponsive to MCT diet; specialist gastroenterology input",
        ],
        "key_ddx_rules": [
            "Neonatal generalised lymphedema + intestinal lymphangiectasia + facial dysmorphism ± intellectual disability → CCBE1/ADAMTS3 (Hennekam) panel",
            "Hypoalbuminaemia + lymphopaenia in a child with oedema → intestinal lymphangiectasia workup; if congenital → CCBE1/ADAMTS3",
            "CCBE1 negative Hennekam phenotype → ADAMTS3 (HLS3); if only lymphedema without intestinal features → FLT4 or other primary lymphedema genes",
        ],
    },

    # ── KIF11 — MCLMR: Microcephaly-Chorioretinopathy-Lymphedema-Mental Retardation ──
    {
        "gene": "KIF11",
        "protein": "Kinesin Family Member 11 Eg5 Bipolar Spindle Kinesin Motor",
        "alias": (
            "KIF11; Eg5; OMIM gene 148760; MCLMR OMIM 152950; 10q23.33; 1056 aa; ~120 kDa; "
            "Microcephaly with or without chorioretinopathy, lymphedema, or intellectual disability (MCLMR); "
            "Autosomal dominant (de novo or familial); "
            "KIF11 is a homotetrameric kinesin motor required for bipolar spindle assembly in mitosis; "
            "Loss → monopolar spindle → metaphase arrest → apoptosis of neuroprogenitors; "
            "MICROCEPHALY (primary, present at birth) + CHORIORETINOPATHY (macular hypoplasia/aplasia) + "
            "LYMPHEDEMA (mild, intermittent lower limb) + INTELLECTUAL DISABILITY; "
            "Ophthalmology MANDATORY — fundoscopy and OCT for macular structure; "
            "Lymphedema mild relative to other features; microcephaly most consistent finding"
        ),
        "aa": "1056 aa",
        "kDa": "~120 kDa",
        "locus": "10q23.33",
        "omim_gene": 148760,
        "omim_disease": 152950,
        "inheritance": "AD — de novo dominant negative or haploinsufficiency; familial AD also reported",
        "gene_class": (
            "KIF11 encodes Eg5, a homotetrameric plus-end-directed kinesin motor essential for "
            "bipolar mitotic spindle assembly. Eg5 crosslinks and slides antiparallel microtubules "
            "apart to establish and maintain the bipolar spindle. KIF11 haploinsufficiency or "
            "dominant-negative variants impair spindle bipolarity → monopolar spindle formation → "
            "mitotic arrest → apoptosis of neural progenitor cells in the developing neocortex → "
            "primary microcephaly. The MCLMR syndrome includes: (1) microcephaly — primary, present "
            "at birth, OFC typically −2 to −5 SD; (2) chorioretinopathy — macular hypoplasia to "
            "aplasia and retinal/RPE pigment changes, visible on fundoscopy and OCT; (3) lymphedema — "
            "lower limb, bilateral, mild, intermittent, often unnoticed; (4) intellectual disability — "
            "variable, mild to severe. Not all four features are present in every patient. Lymphedema "
            "is the least consistent feature. Ophthalmology is mandatory: chorioretinopathy discovered "
            "on exam even when not clinically apparent. De novo variants are most common (trio "
            "sequencing recommended). KIF11 is also a target of the anti-cancer drug ispinesib "
            "(clinical trials); this is not therapeutically relevant to MCLMR currently."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("KIF11 de novo missense (motor domain) — dominant negative, most common", 0.40),
            ("KIF11 frameshift — haploinsufficiency, familial AD", 0.22),
            ("KIF11 nonsense — premature stop, NMD", 0.18),
            ("KIF11 splice-site — exon skipping", 0.12),
            ("KIF11 de novo missense (stalk/tail domain) — variable phenotype", 0.08),
        ],
        "age_onset_years_range": (0, 1),
        "sex_ratio_M": 0.50,
        "rates": {
            "primary_microcephaly":                  0.97,
            "chorioretinopathy_macular_hypoplasia":  0.75,
            "intellectual_disability":               0.72,
            "lower_limb_lymphedema":                 0.55,
            "lymphedema_mild_intermittent":          0.50,
            "de_novo_variant_confirmed":             0.65,
            "ophthalmology_mandatory":               0.97,
            "oct_macular_assessment":                0.80,
            "family_history_ad":                     0.35,
            "trio_sequencing_recommended":           0.65,
        },
        "critical_alerts": [
            "KIF11-OPHTHALMOLOGY-MANDATORY: Chorioretinopathy (macular hypoplasia/aplasia + RPE changes) present in 75% — fundoscopy + OCT MANDATORY at diagnosis even if no visual complaint",
            "KIF11-MICROCEPHALY-PRIMARY: Primary microcephaly (present at birth, not acquired) — head circumference at birth is diagnostic; acquired microcephaly suggests different aetiology",
            "KIF11-LYMPHEDEMA-SUBTLE: Lymphedema in only 55% — mild, intermittent lower limb oedema often not the presenting complaint; active limb volume assessment needed",
            "KIF11-TRIO-SEQUENCING: Most cases de novo — trio exome/genome (proband + parents) recommended; familial AD recurrence risk 50%",
            "KIF11-INTELLECTUAL-DISABILITY-VARIABLE: Range mild to severe — neuropsychological assessment, early intervention programme; educational planning from diagnosis",
        ],
        "key_ddx_rules": [
            "Microcephaly + chorioretinopathy + lymphedema ± intellectual disability → KIF11 (MCLMR) — this specific combination is pathognomonic",
            "Microcephaly + macular hypoplasia on fundoscopy → KIF11 even without lymphedema (lymphedema absent in 45%)",
            "KIF11 negative MCLMR → second-hit KIF11 allele (mosaic); consider other microcephaly-retinopathy genes (CENPJ, CDK5RAP2, CEP152) — lymphedema helps specificity",
        ],
    },

    # ── ADAMTS3 — Hennekam lymphangiectasia-lymphedema syndrome type 3 ──
    {
        "gene": "ADAMTS3",
        "protein": "A Disintegrin-Like and Metalloproteinase with Thrombospondin Motifs 3 VEGF-C Activating Protease",
        "alias": (
            "ADAMTS3; OMIM gene 605011; Hennekam syndrome HLS3 OMIM 618154; 4q13.3; 1232 aa; ~140 kDa; "
            "Hennekam lymphangiectasia-lymphedema syndrome type 3 (HLS3); autosomal recessive; "
            "ADAMTS3 is a secreted metalloprotease that cleaves pro-VEGF-C N-terminal propeptide → "
            "generates mature VEGF-C → activates VEGFR3 signalling in developing lymphatics; "
            "CCBE1 is the obligate cofactor for ADAMTS3 activity — loss of EITHER gene blocks VEGF-C maturation; "
            "Phenocopies CCBE1 (HLS1) — intestinal lymphangiectasia + protein-losing enteropathy + ID; "
            "Test CCBE1 AND ADAMTS3 simultaneously (same pathway, same phenotype); MCT diet mandatory"
        ),
        "aa": "1232 aa",
        "kDa": "~140 kDa",
        "locus": "4q13.3",
        "omim_gene": 605011,
        "omim_disease": 618154,
        "inheritance": "AR — biallelic LOF (frameshift/nonsense/missense); consanguinity common",
        "gene_class": (
            "ADAMTS3 (A Disintegrin-like and Metalloproteinase with Thrombospondin motifs 3) is a "
            "secreted zinc-dependent metalloprotease of the ADAMTS family. ADAMTS3 cleaves the "
            "N-terminal CUB domain propeptide from pro-VEGF-C, generating the mature processed "
            "form that binds and activates VEGFR3 with high affinity. This processing step is "
            "essential for VEGFR3-driven lymphangiogenesis. CCBE1 is the obligate cofactor: it "
            "recruits ADAMTS3 to heparan sulfate proteoglycans on the lymphatic endothelial cell "
            "surface and enhances its proteolytic efficiency by ~100-fold. Without either CCBE1 or "
            "ADAMTS3, pro-VEGF-C accumulates but cannot signal — explaining why CCBE1 (HLS1) and "
            "ADAMTS3 (HLS3) cause identical Hennekam syndrome phenotypes. Biallelic ADAMTS3 "
            "loss-of-function variants → generalised lymphatic dysplasia: lymphoedema from birth, "
            "intestinal lymphangiectasia, protein-losing enteropathy (hypoalbuminaemia, "
            "lymphopaenia), intellectual disability (variable ~40%), and distinctive facial "
            "dysmorphism. Management identical to HLS1/CCBE1: MCT diet, albumin monitoring, "
            "octreotide for severe intestinal disease, and early developmental support. Genetic "
            "diagnosis: sequence CCBE1 and ADAMTS3 in parallel (panel)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("ADAMTS3 frameshift — biallelic LOF, most common", 0.32),
            ("ADAMTS3 nonsense — premature termination, NMD", 0.25),
            ("ADAMTS3 missense (metalloprotease domain) — impaired VEGF-C cleavage", 0.22),
            ("ADAMTS3 splice-site — exon skipping, partial LOF", 0.12),
            ("ADAMTS3 homozygous missense — consanguineous", 0.09),
        ],
        "age_onset_years_range": (0, 2),
        "sex_ratio_M": 0.50,
        "rates": {
            "lymphedema_from_birth":                    0.93,
            "intestinal_lymphangiectasia":              0.82,
            "protein_losing_enteropathy":               0.78,
            "hypoalbuminaemia":                         0.80,
            "lymphopaenia":                             0.70,
            "intellectual_disability":                  0.42,
            "facial_dysmorphism":                       0.75,
            "pulmonary_lymphangiectasia":               0.25,
            "consanguinity":                            0.52,
            "mct_diet_required":                        0.86,
            "albumin_supplementation_required":         0.68,
            "annual_albumin_lymphocyte_monitoring":     0.88,
            "ccbe1_tested_simultaneously":              0.92,
        },
        "critical_alerts": [
            "ADAMTS3-TEST-WITH-CCBE1: ADAMTS3 (HLS3) and CCBE1 (HLS1) cause identical Hennekam phenotype — ALWAYS test BOTH simultaneously (panel); sequential testing wastes diagnostic time",
            "ADAMTS3-PROTEIN-LOSING-ENTEROPATHY: Hypoalbuminaemia + lymphopaenia is the biochemical signature — measure at diagnosis, every 6 months; secondary immunodeficiency from hypogammaglobulinaemia",
            "ADAMTS3-MCT-DIET: Medium-chain triglyceride diet mandatory from diagnosis — bypasses impaired long-chain fat absorption via intestinal lacteals; reduces protein loss",
            "ADAMTS3-INTELLECTUAL-DISABILITY: ~40% affected — early developmental assessment and intervention; prognosis for ID better than CCBE1 in some series but variable",
            "ADAMTS3-VEGFC-PATHWAY: ADAMTS3 + CCBE1 form obligate complex for VEGF-C activation → VEGFR3 signalling — future VEGF-C replacement or VEGFR3 agonist therapy may target this pathway",
        ],
        "key_ddx_rules": [
            "Generalised neonatal lymphedema + intestinal lymphangiectasia + facial dysmorphism → request CCBE1 + ADAMTS3 panel simultaneously (phenotype identical)",
            "CCBE1 negative Hennekam phenotype → ADAMTS3 (HLS3) most likely next gene",
            "ADAMTS3/CCBE1 both negative Hennekam phenotype → PIK3CA mosaic lymphatic overgrowth (PROS), FAT4 (Van Maldergem); re-review phenotype",
        ],
    },
]


def _make_patients(gene_entry):
    rng = random.Random(gene_entry["seed"])
    patients = []
    for i in range(gene_entry["n_patients"]):
        etiol_labels, etiol_weights = zip(*gene_entry["etiologies"])
        etiology = rng.choices(etiol_labels, weights=etiol_weights)[0]
        a_lo, a_hi = gene_entry["age_onset_years_range"]
        age_onset = round(rng.uniform(a_lo, a_hi), 1)
        sex = "M" if rng.random() < gene_entry["sex_ratio_M"] else "F"
        flags = {}
        for key, rate in gene_entry["rates"].items():
            flags[key] = rng.random() < rate
        patients.append({
            "patient_id": f"{gene_entry['gene']}-{i+1:03d}",
            "gene": gene_entry["gene"],
            "etiology": etiology,
            "age_onset_years": age_onset,
            "sex": sex,
            "phenotype_flags": flags,
        })
    return patients


def _agg(patients, key):
    vals = [p["phenotype_flags"].get(key, False) for p in patients]
    return round(100 * sum(vals) / len(vals), 1) if vals else 0.0


def get_overview():
    all_patients = []
    for g in LYMPHEDEMA_GENES:
        all_patients.extend(_make_patients(g))

    agg = {
        "total_patients":                           len(all_patients),
        "n_genes":                                  len(LYMPHEDEMA_GENES),
        "seeds":                                    f"{SEED_BASE}–{SEED_BASE + len(LYMPHEDEMA_GENES) - 1}",
        "lower_limb_lymphedema_pct":               round(
            100 * sum(
                1 for p in all_patients
                if p["phenotype_flags"].get("lower_limb_lymphedema")
                or p["phenotype_flags"].get("bilateral_lower_limb_lymphedema")
                or p["phenotype_flags"].get("bilateral_lower_limb_oedema_congenital")
                or p["phenotype_flags"].get("lymphedema_from_birth")
            ) / len(all_patients), 1),
        "congenital_onset_pct":                    round(
            100 * sum(
                1 for p in all_patients
                if p["phenotype_flags"].get("bilateral_lower_limb_oedema_congenital")
                or p["phenotype_flags"].get("lymphedema_from_birth")
            ) / len(all_patients), 1),
        "intestinal_lymphangiectasia_pct":         _agg(all_patients, "intestinal_lymphangiectasia"),
        "protein_losing_enteropathy_pct":          _agg(all_patients, "protein_losing_enteropathy"),
        "intellectual_disability_pct":             _agg(all_patients, "intellectual_disability"),
        "distichiasis_foxc2_pct":                  _agg(all_patients, "distichiasis_pathognomonic"),
        "hypotrichosis_pct":                       round(
            100 * sum(
                1 for p in all_patients
                if p["phenotype_flags"].get("hypotrichosis_scalp_body")
                or p["phenotype_flags"].get("hypotrichosis_dry_sparse_hair")
            ) / len(all_patients), 1),
        "microcephaly_kif11_pct":                  _agg(all_patients, "primary_microcephaly"),
        "chorioretinopathy_pct":                   _agg(all_patients, "chorioretinopathy_macular_hypoplasia"),
        "mct_diet_required_pct":                   round(
            100 * sum(
                1 for p in all_patients
                if p["phenotype_flags"].get("mct_diet_required")
            ) / len(all_patients), 1),
    }

    top_alerts = [
        "FLT4-NO-DISTICHIASIS: Examine eyelids in ALL primary lymphoedema — absence of distichiasis favours Milroy (FLT4); presence pathognomonic for FOXC2",
        "FOXC2-DISTICHIASIS-PATHOGNOMONIC: Extra eyelashes virtually UNIVERSAL in FOXC2 carriers — ophthalmology MANDATORY; untreated → corneal scarring → vision loss",
        "CCBE1-ADAMTS3-TEST-SIMULTANEOUSLY: CCBE1 (HLS1) and ADAMTS3 (HLS3) phenocopies — ALWAYS test BOTH in Hennekam phenotype (intestinal lymphangiectasia + ID)",
        "CCBE1-ADAMTS3-MCT-DIET: Intestinal lymphangiectasia → protein-losing enteropathy → hypoalbuminaemia → dietitian + MCT diet at diagnosis; monitor albumin every 6 months",
        "KIF11-OPHTHALMOLOGY-MANDATORY: Chorioretinopathy in 75% of MCLMR — fundoscopy + OCT MANDATORY; microcephaly + chorioretinopathy combination virtually diagnostic",
        "GJC2-BIALLELIC-CNS: Compound heterozygotes with ≥1 LOF allele → SPG44 leukodystrophy risk — brain MRI mandatory in biallelic GJC2",
        "SOX18-RENAL-ULTRASOUND: Renal anomalies in 30% of HLTRS — annual renal ultrasound from diagnosis",
        "PROX1-SOX18-TRIAD: Both cause HLTS — test simultaneously; examine hair, limbs, AND mucosae for full triad",
    ]

    return {
        "atlas": "Hereditary-Primary-Lymphedema-Atlas",
        "title": "Complete 8-Gene Hereditary Primary Lymphedema Atlas",
        "genes": [g["gene"] for g in LYMPHEDEMA_GENES],
        "diseases": [g["omim_disease"] for g in LYMPHEDEMA_GENES],
        "aggregate_stats": agg,
        "top_alerts": top_alerts,
        "registered": "2026-09-05",
    }


def get_breakdown():
    result = {}
    for g in LYMPHEDEMA_GENES:
        patients = _make_patients(g)
        result[g["gene"]] = {
            "gene":            g["gene"],
            "protein":         g["protein"],
            "locus":           g["locus"],
            "aa":              g["aa"],
            "inheritance":     g["inheritance"],
            "omim_gene":       g["omim_gene"],
            "omim_disease":    g["omim_disease"],
            "n_patients":      g["n_patients"],
            "seed":            g["seed"],
            "alias":           g["alias"],
            "gene_class":      g["gene_class"],
            "critical_alerts": g["critical_alerts"],
            "key_ddx_rules":   g["key_ddx_rules"],
            "phenotype_rates": {k: round(v * 100, 1) for k, v in g["rates"].items()},
            "etiologies":      [{"label": l, "pct": round(w * 100, 1)} for l, w in g["etiologies"]],
            "aggregate":       {k: _agg(patients, k) for k in g["rates"]},
        }
    return result


def get_definitions():
    return {
        "atlas": "Hereditary-Primary-Lymphedema-Atlas",
        "definitions": [
            {
                "term": "Primary Lymphedema — Definition and Classification",
                "definition": (
                    "Primary lymphedema is oedema resulting from a structural or functional defect "
                    "intrinsic to the lymphatic system, not caused by an external insult (infection, "
                    "surgery, malignancy — those are secondary lymphedema). Classified by onset: "
                    "congenital (present at birth; includes Milroy disease/FLT4 and Hennekam/CCBE1-"
                    "ADAMTS3), praecox (onset puberty to age 35 — most common; FOXC2, GJC2, PROX1, "
                    "SOX18), and tarda (onset >35 years). Classified by distribution: unilateral, "
                    "bilateral, localised (lower limb most common), or generalised. Genetic aetiology "
                    "identified in ~30–40% of familial cases; panels including FLT4, FOXC2, PROX1, "
                    "GJC2, SOX18, CCBE1, KIF11, ADAMTS3, GATA2, FAT4, PIEZO1 recommended for "
                    "familial/syndromic presentations."
                )
            },
            {
                "term": "VEGF-C/VEGFR3 Signalling Axis — Core Lymphangiogenesis Pathway",
                "definition": (
                    "Vascular endothelial growth factor C (VEGF-C) is the primary lymphangiogenic "
                    "cytokine; it binds VEGFR3 (encoded by FLT4) on lymphatic endothelial cells (LECs) "
                    "to drive LEC proliferation, migration, and survival. VEGF-C is secreted as a "
                    "propeptide (pro-VEGF-C) that must be proteolytically activated by cleavage of its "
                    "N-terminal CUB domain. The CCBE1–ADAMTS3 complex performs this cleavage: CCBE1 "
                    "recruits ADAMTS3 to the LEC surface; ADAMTS3 cleaves pro-VEGF-C to the mature "
                    "active form. Loss of FLT4, CCBE1, or ADAMTS3 all produce overlapping lymphatic "
                    "hypoplasia phenotypes by disrupting this same signalling axis. SOX18 and PROX1 "
                    "regulate FLT4 transcription in developing LECs."
                )
            },
            {
                "term": "Distichiasis — FOXC2 Pathognomonic Sign",
                "definition": (
                    "Distichiasis is the presence of an accessory row of eyelashes arising from the "
                    "orifices of the meibomian glands (modified sebaceous glands) of the tarsal plates "
                    "of the eyelids. In lymphedema-distichiasis syndrome (FOXC2 mutations), FOXC2 "
                    "haploinsufficiency causes meibomian gland cells to adopt an eyelash follicle fate. "
                    "Distichiasis is present in virtually all FOXC2 mutation carriers (>97%) and can be "
                    "detected by slit-lamp biomicroscopy even before lymphoedema appears. The aberrant "
                    "lashes may be directed inward (trichiasis) → corneal irritation, recurrent erosions, "
                    "corneal scarring, and eventually vision loss. Management: electrolysis, cryotherapy, "
                    "or laser ablation under ophthalmological supervision. Distichiasis is absent in ALL "
                    "other hereditary primary lymphedema genes — its presence is pathognomonic for FOXC2."
                )
            },
            {
                "term": "Intestinal Lymphangiectasia — Hennekam Syndrome (CCBE1/ADAMTS3)",
                "definition": (
                    "Intestinal lymphangiectasia is dilation of the intestinal lacteals (lymphatic "
                    "capillaries of the small bowel villi) due to lymphatic obstruction or hypoplasia. "
                    "In Hennekam syndrome (CCBE1 or ADAMTS3 biallelic mutations), global lymphatic "
                    "dysplasia causes intestinal lymphatic failure → lacteal engorgement → rupture of "
                    "lacteals into the gut lumen → protein-losing enteropathy. Clinical consequences: "
                    "hypoalbuminaemia (primary oedema from low oncotic pressure, compounding the "
                    "lymphatic oedema), lymphopaenia (lymphocytes lost into gut), and "
                    "hypogammaglobulinaemia (secondary immune deficiency). Treatment: medium-chain "
                    "triglyceride (MCT)-rich diet (absorbed directly into portal blood, bypassing "
                    "lacteals), albumin supplementation when severe, octreotide for refractory cases. "
                    "Monitor plasma albumin + lymphocytes every 6 months."
                )
            },
            {
                "term": "Milroy Disease — FLT4/VEGFR3 Dominant Negative Mechanism",
                "definition": (
                    "Milroy disease (primary congenital lymphoedema type 1) is caused by heterozygous "
                    "missense variants in the kinase domain of FLT4/VEGFR3. Mutant VEGFR3 protein "
                    "forms dimers with wild-type VEGFR3 on LEC surfaces; these heterodimers are "
                    "catalytically inactive (dominant negative) — even when VEGF-C binds, the kinase "
                    "cannot phosphorylate downstream substrates. This dominant negative effect "
                    "completely abrogates VEGF-C signalling in heterozygous cells. As a result, "
                    "lower limb lymphatics fail to form subcutaneous capillary networks, and only "
                    "rudimentary lymphatics develop → congenital bilateral lower limb oedema. "
                    "Penetrance is ~85–90%; ~10–15% of FLT4 heterozygotes are asymptomatic. "
                    "Lymphoscintigraphy shows absent or grossly hypoplastic lower limb lymphatics — "
                    "characteristic and useful when genetic diagnosis is delayed."
                )
            },
            {
                "term": "GJC2/Connexin 47 Dual Phenotype — Lymphedema and SPG44",
                "definition": (
                    "GJC2 encodes Connexin 47, a gap junction protein expressed in both lymphatic "
                    "endothelial cells and oligodendrocytes. Biallelic missense variants (especially "
                    "M283T, R260C) impair LEC gap junction communication → defective lymphatic valve "
                    "development → late-onset primary lymphoedema. Biallelic loss-of-function (truncating) "
                    "variants cause SPG44: leukodystrophy with progressive spastic paraplegia, reflecting "
                    "Cx47's role in oligodendrocyte panglial coupling (GJC2 and GJB1/Cx32 form "
                    "oligodendrocyte–astrocyte coupling channels essential for myelin maintenance). "
                    "Compound heterozygotes (one missense + one truncating allele) may have intermediate "
                    "or combined phenotypes. Brain MRI is mandatory in all biallelic GJC2 carriers to "
                    "exclude leukodystrophy. Monoallelic heterozygous missense variants may cause mild "
                    "late-onset lymphoedema."
                )
            },
            {
                "term": "MCLMR Syndrome — KIF11 Microcephaly-Chorioretinopathy-Lymphedema",
                "definition": (
                    "MCLMR (Microcephaly with or without Chorioretinopathy, Lymphedema, or intellectual "
                    "disability Retardation) is caused by heterozygous KIF11 variants. KIF11/Eg5 drives "
                    "bipolar spindle assembly: haploinsufficiency or dominant-negative variants → "
                    "monopolar spindle formation → mitotic arrest → apoptosis of cortical neuroprogenitors "
                    "→ primary microcephaly (head circumference −2 to −5 SD at birth). Chorioretinopathy "
                    "is the second most consistent feature (75%): macular hypoplasia or aplasia, plus "
                    "RPE mottling/pigment changes, visible on fundoscopy and OCT. Lymphedema (lower limb, "
                    "bilateral, mild, intermittent) present in ~55%. Intellectual disability in ~72%, "
                    "variable severity. Ophthalmology with OCT is mandatory at diagnosis. Most cases are "
                    "de novo — trio exome/genome is the optimal first-tier genetic test."
                )
            },
            {
                "term": "MCT Diet — Management of Intestinal Lymphangiectasia",
                "definition": (
                    "Medium-chain triglycerides (MCTs; carbon chain length C8–C12) are unique among "
                    "dietary fats: they are absorbed directly from enterocytes into the portal "
                    "circulation as free fatty acids, bypassing the intestinal lymphatic system "
                    "(lacteals) entirely. Long-chain triglycerides (LCTs) require packaging into "
                    "chylomicrons, which enter lacteals — in intestinal lymphangiectasia, this flux "
                    "exacerbates lacteal engorgement, rupture, and protein loss. An MCT-enriched diet "
                    "(replacing LCT with MCT-rich oils such as coconut oil or specialised MCT oil "
                    "products) substantially reduces chylomicron formation, decreases lymphatic "
                    "pressure, and diminishes protein loss. Fat-soluble vitamins (A, D, E, K) require "
                    "supplementation (they are lost with lacteals rupture and cannot be absorbed on "
                    "strict MCT diet). Regular monitoring: plasma albumin, lymphocytes, "
                    "immunoglobulins, and fat-soluble vitamin levels every 6 months."
                )
            },
            {
                "term": "PROX1 Master Regulator — Lymphatic Endothelial Cell Identity",
                "definition": (
                    "PROX1 (Prospero-related homeobox 1) is the master transcription factor for lymphatic "
                    "endothelial cell (LEC) fate. During embryogenesis, PROX1 expression is initiated by "
                    "SOX18 in a subpopulation of venous endothelial cells (ECs) of the cardinal vein. "
                    "PROX1 then reprogrammes venous ECs to LECs by suppressing the blood-vascular EC "
                    "programme and activating the lymphatic-specific gene programme — including LYVE1, "
                    "PDPN (podoplanin), VEGFR3 (FLT4), and FOXC2. This reprogramming initiates lymphatic "
                    "sac formation. In PROX1 heterozygotes, incomplete LEC specification → fewer LECs → "
                    "lymphatic hypoplasia → HLTS (hypotrichosis, lymphedema, telangiectasia). PROX1 also "
                    "regulates hepatic progenitor fate (PROX1 is expressed in hepatocytes) and "
                    "cardiovascular development; extra-lymphatic phenotypes may emerge with age."
                )
            },
            {
                "term": "Compression Therapy — Primary Lymphedema Management",
                "definition": (
                    "Compression is the cornerstone of symptomatic management for all forms of primary "
                    "lymphedema. Graduated compression hosiery (20–40 mmHg, class II–III) reduces "
                    "ultrafiltration from capillaries, supports lymphatic pump function by intermittent "
                    "compression during gait, and prevents progression of oedema. For severe or "
                    "fluctuating cases: pneumatic intermittent compression devices (ICD) augment lymph "
                    "flow. Complete Decongestive Therapy (CDT) — combining manual lymphatic drainage "
                    "(MLD), compression bandaging, skin care, and exercises — is first-line for "
                    "significant lymphoedema. No curative therapy exists for any of the 8 genetic forms "
                    "in this atlas; treatment is symptomatic and lifelong. Annual limb volume "
                    "measurement, skin integrity assessment (cellulitis risk), and physiotherapist "
                    "review are mandatory."
                )
            },
            {
                "term": "Cascade Genetic Testing — Hereditary Primary Lymphedema",
                "definition": (
                    "When a pathogenic variant is identified in any of the 8 primary lymphedema genes, "
                    "predictive cascade genetic testing is recommended for all first-degree relatives at "
                    "50% prior risk (AD genes: FLT4, FOXC2, PROX1, SOX18, KIF11) or 25% prior risk "
                    "(AR genes: GJC2 biallelic, CCBE1, ADAMTS3 — carrier frequency ~50% in parents). "
                    "For AD genes: cascade testing identifies presymptomatic carriers who can commence "
                    "prophylactic compression and surveillance (FOXC2 — arrange ophthalmology; SOX18 — "
                    "arrange renal ultrasound) before oedema develops. For AR genes (Hennekam): "
                    "carrier parents have a 25% recurrence risk per pregnancy — refer to reproductive "
                    "genetics for pre-implantation genetic testing (PGT) or prenatal diagnosis options."
                )
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== BREAKDOWN (FLT4) ===")
    bd = get_breakdown()
    print(json.dumps(bd["FLT4"], indent=2))
    print("\n=== DEFINITIONS (first 2) ===")
    defs = get_definitions()
    for d in defs["definitions"][:2]:
        print(f"\n{d['term']}:\n{d['definition'][:200]}...")
