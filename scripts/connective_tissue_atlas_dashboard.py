#!/usr/bin/env python3
"""Connective-Tissue-Atlas — Complete 8-Gene Hereditary Connective Tissue Disorders Atlas
COL5A1  (Pro-alpha-1 chain of Collagen V; 2899 aa; 9q34.3; AD;
          Classical EDS (cEDS) type 1 — most common EDS subtype (~50%);
          HAPLOINSUFFICIENCY: PTC/NMD alleles → milder; dominant-negative missense → more severe;
          Beighton score ≥5/9 + wide atrophic scarring + skin hyperextensibility DIAGNOSTIC;
          Subluxations/dislocations; chronic musculoskeletal pain; proprioception impaired;
          Normal life expectancy; aortic involvement rare; vascular surveillance only if FH) ·
COL3A1  (Pro-alpha-1 chain of Collagen III; 1466 aa; 2q32.2; AD;
          Vascular EDS (vEDS) — LIFE-THREATENING; arterial rupture, hollow organ perforation;
          AVOID: colonoscopy, diagnostic angiography, unnecessary surgery — perforation risk;
          Celiprolol 400 mg/day (beta1-selective with beta2-agonist) — Level B evidence;
          Median survival 48 years (arterial event); first event median age 29 years;
          Skin translucency + acrogeria + easy bruising: classic vEDS triad) ·
TNXB    (Tenascin XB; 4108 aa; 6p21.32; AR;
          Classical-like EDS (clEDS) — haploinsufficiency also causes TNXB-related hypermobility;
          LOW SERUM TENASCIN-X (<4 mcg/mL) DIAGNOSTIC in biallelic clEDS;
          NO atrophic scars: key DDx from cEDS (COL5A1);
          Generalised joint hypermobility + hyperextensible skin + easy bruising without scarring;
          Skin healing normal: distinguishes from COL5A1/COL3A1;
          Some patients have myopathy and mild myasthenia features) ·
PLOD1   (Procollagen-Lysine 2-Oxoglutarate 5-Dioxygenase 1; 727 aa; 1p36.22; AR;
          Kyphoscoliotic EDS type 1 (kEDS-PLOD1 / EDS VIA) — progressive neonatal kyphoscoliosis;
          OCULAR FRAGILITY: corneal globe rupture risk — eye protection MANDATORY, avoid contact sports;
          URINE LYSYL-PYRIDINOLINE RATIO >0.09 DIAGNOSTIC (cheap, non-invasive);
          Pyridoxine (B6) 5 mg/kg/day: ~30% show meaningful benefit — trial MANDATORY;
          Hypotonia at birth; muscular hypotonia; marfanoid habitus;
          PLOD1 is lysyl hydroxylase 1 — hydroxylates telopeptide lysines for collagen crosslinking) ·
ADAMTS2 (ADAM Metallopeptidase with TS type 1 motif 2; 1218 aa; 5q35.3; AR;
          Dermatosparaxis EDS (dEDS) — most extreme skin fragility of all EDS subtypes;
          LOOSE DROOPING EXCESS SKIN: characteristic lax folds on face/limbs from birth;
          Umbilical hernia 100%; easy bruising; blue sclerae;
          Short fingers + broad forehead + micrognathia in some;
          ADAMTS2 cleaves N-propeptide of procollagen I, II, III — deficiency → retention of N-propeptide → abnormal fibril packing;
          Skin EM shows hieroglyphic/stellate collagen fibrils PATHOGNOMONIC) ·
FBN1    (Fibrillin-1; 2871 aa; 15q21.1; AD;
          Marfan Syndrome — aortic root dilatation + ocular + skeletal;
          ECTOPIA LENTIS: upward displacement (vs downward in homocystinuria) — slit-lamp MANDATORY;
          AORTIC ROOT SURGERY threshold 4.5 cm (or 4.0 cm if FH dissection / rapid growth >3 mm/year);
          LOSARTAN: reduces aortic root growth rate — Level A; Beta-blocker (bisoprolol/atenolol) adjunct;
          Arm-span:height ratio >1.05; reduced upper-to-lower segment; scoliosis; pes planus;
          FBN1 encodes fibrillin-1 in microfibrils — scaffolds TGF-beta and elastic fibres) ·
TGFBR2  (TGF-beta Receptor Type 2; 567 aa; 3p24.1; AD;
          Loeys-Dietz Syndrome type 2 (LDS2) — aortic aneurysm + bifid uvula/palate + arterial tortuosity;
          BIFID UVULA / CLEFT PALATE: mucosal finding — examine oropharynx in all connective tissue disorders;
          MORE AGGRESSIVE SURGERY THRESHOLD than Marfan: 4.0-4.2 cm (not 4.5 cm) — dissection at smaller diameters;
          Losartan MANDATORY: TGF-beta signalling overactivated in LDS — angiotensin II blockade blunts it;
          Arterial tortuosity: carotid, vertebral, intracranial — imaging entire aorta + cerebral vessels;
          Club foot + cervical spine instability + hypertelorism in some) ·
ACTA2   (Actin Alpha 2, Smooth Muscle; 377 aa; 10q23.31; AD;
          Multisystemic Smooth Muscle Dysfunction Syndrome (MSMD) / Familial TAAD type 6;
          IRIS FLOCCULI (iris transillumination defects / iridescent pupil margin fibres) PATHOGNOMONIC;
          MOYA MOYA DISEASE-LIKE: intracranial occlusive vasculopathy — stroke risk in young;
          Aortic aneurysm + pulmonary arterial hypertension + intestinal hypoperistalsis + bladder dysfunction;
          ACTA2 Arg179His: most severe MSMD (congenital heart defects + stroke); other missense = milder FTAAD;
          Surgery threshold 4.0 cm in MSMD; aspirin + statin if Moya Moya features detected by MRA)
320-patient aggregate cohort (8 × 40, seeds 1398-1405)
"""

import random

SEED_BASE = 1398

CT_GENES = [
    # ── COL5A1 — Classical EDS ──
    {
        "gene": "COL5A1",
        "protein": "Collagen Type V Alpha-1 Chain",
        "alias": (
            "COL5A1; OMIM gene 120215; cEDS #130000; "
            "9q34.3; 2899 aa; ~300 kDa (pro-alpha chain); "
            "Collagen V is a quantitatively minor fibrillar collagen (~5-10% of total skin collagen) "
            "that regulates collagen I fibril diameter — LOF → abnormally thick fibrils → impaired tensile strength; "
            "HAPLOINSUFFICIENCY (PTC → NMD): milder phenotype — quantitative deficit; "
            "DOMINANT-NEGATIVE MISSENSE: severe phenotype — abnormal chains poison native collagen V assembly; "
            "BEIGHTON SCORE ≥5/9 + WIDE ATROPHIC SCARS + SKIN HYPEREXTENSIBILITY = cEDS diagnostic triad; "
            "Subluxations and dislocations; chronic musculoskeletal pain; proprioceptive deficits; "
            "Normal cardiac surveillance (aortic involvement rare in COL5A1); life expectancy normal"
        ),
        "aa": "2899 aa",
        "kDa": "~300 kDa",
        "locus": "9q34.3",
        "omim_gene": 120215,
        "omim_disease": 130000,
        "inheritance": "AD — haploinsufficiency (PTC/NMD) or dominant-negative missense; de novo ~50%",
        "gene_class": (
            "COL5A1 encodes the pro-alpha-1 chain of Collagen V, a quantitatively minor fibrillar collagen "
            "that co-assembles with Collagen I to regulate fibril diameter and surface properties. "
            "In normal skin, Collagen V (α1α1α2 heterotrimer) forms the nucleating core of mixed fibrils; "
            "loss of one allele halves Collagen V output, leading to thicker, disorganised Collagen I fibrils "
            "with reduced tensile strength. Dominant-negative Gly-X-Y missense variants introduce "
            "structural disruption into the triple helix, producing a more severe phenotype than haploinsufficiency."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("Haploinsufficiency — PTC/splice/frameshift → NMD → quantitative deficit", 0.55),
            ("Dominant-negative Gly-X-Y missense → triple helix disruption → more severe", 0.35),
            ("Large intragenic deletion (MLPA-detectable)", 0.10),
        ],
        "age_onset_years_range": (3, 25),
        "sex_ratio_M": 0.35,
        "rates": {
            "skin_hyperextensibility":          0.95,
            "wide_atrophic_scarring":           0.90,
            "joint_hypermobility_beighton_5":   0.95,
            "recurrent_subluxations":           0.75,
            "chronic_musculoskeletal_pain":     0.80,
            "proprioceptive_deficit":           0.65,
            "easy_bruising":                    0.60,
            "mitral_valve_prolapse":            0.20,
            "scoliosis":                        0.35,
            "pelvic_floor_prolapse":            0.25,
        },
        "hallmarks": [
            "DIAGNOSTIC TRIAD: skin hyperextensibility + wide atrophic scars + Beighton score ≥5/9",
            "ATROPHIC SCARS: papyraceous / cigarette-paper scars — pathognomonic vs clEDS (TNXB) which has NO scars",
            "HAPLOINSUFFICIENCY MILDER: PTC/NMD alleles → fewer and milder scars than dominant-negative Gly-X-Y",
            "JOINT HYPERMOBILITY: Beighton ≥5/9 required for diagnosis; proprioception deficits increase fall risk",
            "CHRONIC PAIN: commonest disabling feature — multidisciplinary pain management from diagnosis",
            "AORTIC SURVEILLANCE: routine echocardiogram at diagnosis; repeat only if MVP or borderline aorta",
            "FEMALE PREDOMINANCE: F:M ~2:1 (joint laxity more symptomatic in females + hormonal effects on laxity)",
            "SKIN BIOPSY + EM: thick collagen fibrils visible on electron microscopy if diagnosis uncertain",
        ],
        "treatment_alerts": [
            "NO CURATIVE THERAPY: physiotherapy (joint stability), bracing (acute subluxations), pain management",
            "AVOID HIGH-IMPACT SPORT: joint protection; swimming and cycling preferred over running/contact sports",
            "SKIN WOUND CARE: reinforce wound edges; avoid staples (skin tears); steri-strips ± tension sutures",
            "OBSTETRIC RISK: uterine/cervical fragility + risk of preterm PROM — refer to maternal-fetal medicine",
            "PELVIC FLOOR: physiotherapy from young adulthood; prolapse risk — pelvic floor exercises MANDATORY",
        ],
        "organ_system": "skin + joints + fascia; mild cardiac (MVP); musculoskeletal (chronic pain)",
        "primary_treatment": "Physiotherapy (joint stabilisation); pain management; wound care; avoid high-impact sport; obstetric surveillance",
    },
    # ── COL3A1 — Vascular EDS ──
    {
        "gene": "COL3A1",
        "protein": "Collagen Type III Alpha-1 Chain",
        "alias": (
            "COL3A1; OMIM gene 120180; vEDS #130050; "
            "2q32.2; 1466 aa; ~139 kDa (pro-alpha chain); "
            "Collagen III is the major structural collagen of hollow viscera (bowel, uterus) and blood vessels; "
            "LOF → reduced vascular wall integrity → arterial dissection/rupture, bowel perforation, uterine rupture; "
            "MOST DANGEROUS EDS: median survival 48 years; first arterial event median age 29 years; "
            "AVOID: colonoscopy, diagnostic angiography, elective surgery — perforation/dissection risk; "
            "Celiprolol 400 mg/day (beta1-selective + beta2-agonist): Level B evidence — reduces events; "
            "Skin: thin/translucent, acrogeria (aged hands), easy bruising (NOT hyperextensible!)"
        ),
        "aa": "1466 aa",
        "kDa": "~139 kDa",
        "locus": "2q32.2",
        "omim_gene": 120180,
        "omim_disease": 130050,
        "inheritance": "AD — dominant-negative Gly-X-Y missense (~90%) or haploinsufficiency; de novo 50%",
        "gene_class": (
            "COL3A1 encodes the pro-alpha-1(III) chain, which homotrimerises to form Collagen III — "
            "the principal fibrillar collagen of hollow organs (bowel, uterus, bladder) and large blood vessels. "
            "Dominant-negative Gly-X-Y substitutions disrupt triple-helix folding of the homotrimer, "
            "reducing secretion and causing intracellular ER stress. Haploinsufficiency is less common but "
            "leads to quantitative reduction in vascular Collagen III, weakening arterial walls. "
            "The medium-sized arteries (celiac, renal, iliac, mesenteric) are most vulnerable to spontaneous dissection."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("Dominant-negative Gly-X-Y missense → triple-helix disruption (classic vEDS)", 0.90),
            ("Haploinsufficiency (PTC/splice → NMD) → milder phenotype", 0.10),
        ],
        "age_onset_years_range": (15, 40),
        "sex_ratio_M": 0.45,
        "rates": {
            "arterial_dissection_rupture":   0.70,
            "bowel_perforation":             0.25,
            "uterine_rupture_obstetric":     0.15,
            "skin_translucency":             0.80,
            "acrogeria":                     0.55,
            "easy_bruising":                 0.90,
            "skin_NOT_hyperextensible":      0.85,
            "pneumothorax":                  0.15,
            "carotid_artery_involvement":    0.40,
            "celiac_renal_iliac_artery":     0.60,
        },
        "hallmarks": [
            "SKIN NOT HYPEREXTENSIBLE: thin + translucent; acrogeria (prematurely aged hands) — NOT like cEDS",
            "ARTERIAL RUPTURE MEDIAN AGE 29: spontaneous — no trauma required; celiac/renal/iliac most common",
            "BOWEL PERFORATION: sigmoid colon most common; spontaneous without obstruction",
            "UTERINE RUPTURE: obstetric emergency — vEDS women: caesarean preferred; antenatal vascular surveillance",
            "AVOID COLONOSCOPY AND ANGIOGRAPHY: perforation/dissection risk; CT colography/CTA preferred",
            "CELIPROLOL 400 mg/day: Level B evidence reduces arterial events — start at diagnosis; titrate up",
            "ER VASCULAR SURGERY AWARENESS: inform treating vascular surgeons; carry emergency card",
            "FAMILY SCREENING: 50% risk per child; cascade testing of first-degree relatives MANDATORY",
        ],
        "treatment_alerts": [
            "CELIPROLOL 400 mg/day: beta1-blocker + beta2-agonist — reduces arterial events; start from diagnosis",
            "AVOID INVASIVE PROCEDURES: colonoscopy, angiography, elective surgery → perforation/dissection risk",
            "EMERGENCY CARD: carry vEDS alert card; inform ER of diagnosis before any procedure",
            "OBSTETRIC MANAGEMENT: high-risk obstetrics centre; caesarean preferred; uterine rupture risk post-delivery",
            "PSYCHOLOGICAL SUPPORT: high anxiety burden; anticipatory grief for shortened life expectancy",
        ],
        "organ_system": "blood vessels (medium arteries) + hollow viscera (bowel, uterus) + skin (fragility, translucency)",
        "primary_treatment": "Celiprolol 400 mg/day; avoid invasive procedures; vascular/surgical team awareness; cascade family testing",
    },
    # ── TNXB — Classical-like EDS ──
    {
        "gene": "TNXB",
        "protein": "Tenascin XB",
        "alias": (
            "TNXB; OMIM gene 600985; clEDS #606408; "
            "6p21.32; 4108 aa; ~450 kDa extracellular matrix glycoprotein; "
            "Tenascin-X (TNX) stabilises collagen fibril architecture in skin and connective tissue; "
            "LOF → reduced collagen fibril diameter regulation → joint laxity + skin hyperextensibility; "
            "BIALLELIC LOF (AR clEDS): LOW SERUM TENASCIN-X <4 mcg/mL diagnostic; "
            "MONOALLELIC (AD haploinsufficiency): hypermobility spectrum disorder; "
            "KEY DDx FROM cEDS: NO atrophic scars (skin heals normally in clEDS); "
            "6p21.32 proximity to C4 — TNXB/C4 deletions can cause congenital adrenal hyperplasia overlap"
        ),
        "aa": "4108 aa",
        "kDa": "~450 kDa",
        "locus": "6p21.32",
        "omim_gene": 600985,
        "omim_disease": 606408,
        "inheritance": "AR — biallelic LOF (clEDS); AD haploinsufficiency → hypermobility-spectrum; TNXB/C4 deletion overlap",
        "gene_class": (
            "TNXB encodes Tenascin-X (TNX), a large multimeric extracellular matrix glycoprotein expressed "
            "in skin dermis, fascia, and tendons. TNX binds decorin, fibronectin, and collagen fibrils, "
            "stabilising fibril organisation and regulating fibril diameter. "
            "Biallelic LOF → absent serum TNX (measurable by ELISA: <4 mcg/mL diagnostic) → "
            "disorganised collagen fibrils → skin laxity and joint hypermobility without scarring defects "
            "(skin healing enzymes remain intact). "
            "TNXB is adjacent to the C4A/C4B loci in the MHC III region; "
            "large 30 kb TNXB/C4 deletions can cause combined clEDS + CAH21."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("Homozygous frameshift/nonsense → null allele (classic AR clEDS)", 0.50),
            ("Compound heterozygote (two truncating/splice alleles)", 0.30),
            ("Large TNXB/C4 deletion (includes CAH21 overlap)", 0.20),
        ],
        "age_onset_years_range": (2, 20),
        "sex_ratio_M": 0.40,
        "rates": {
            "generalised_joint_hypermobility":  0.98,
            "skin_hyperextensibility":          0.90,
            "easy_bruising":                    0.75,
            "no_atrophic_scars":                0.95,
            "low_serum_tenascin_x":             0.98,
            "chronic_musculoskeletal_pain":     0.70,
            "proprioceptive_deficit":           0.60,
            "mild_myopathy":                    0.30,
            "fatigability":                     0.55,
            "skin_normal_healing":              0.95,
        },
        "hallmarks": [
            "LOW SERUM TENASCIN-X (<4 mcg/mL): ELISA — single most diagnostic test; cheap and non-invasive",
            "NO ATROPHIC SCARS: skin heals normally — key DDx from cEDS (COL5A1) which has wide atrophic scars",
            "GENERALISED JHM: Beighton ≥5/9 in almost all; instability more severe than COL5A1 in some",
            "TNXB/C4 DELETION: check 21-hydroxylase (17-OHP) — congenital adrenal hyperplasia may coexist",
            "MYOPATHY/FATIGABILITY: mild proximal weakness in some biallelic patients; CK usually normal",
            "HAPLOINSUFFICIENCY (monoallelic): hypermobility spectrum disorder — less severe than biallelic",
            "SKIN BIOPSY EM: collagen fibril diameter normal or mildly irregular (vs COL5A1/COL3A1)",
            "MHC III REGION: TNXB 6p21.32 — test complement (C4) levels if large deletion suspected",
        ],
        "treatment_alerts": [
            "SERUM TNX ASSAY: request ELISA for tenascin-X serum level — diagnosis before full gene sequencing",
            "HYDROCORTISONE SCREENING: if TNXB/C4 deletion — check 17-OHP (CAH21 may coexist and be missed)",
            "PHYSIOTHERAPY: joint stabilisation programme; proprioception training; fatigue management",
            "NO DISEASE-MODIFYING THERAPY: TNX replacement theoretical but not clinical; supportive only",
            "DISTINGUISH FROM hEDS: serum TNX normal in hEDS (non-genetic hypermobility); TNXB gene panel mandatory",
        ],
        "organ_system": "skin + joints + fascia + muscles (mild); no major cardiac/vascular involvement",
        "primary_treatment": "Serum TNX assay; joint stabilisation physiotherapy; screen for CAH if TNXB/C4 deletion; supportive",
    },
    # ── PLOD1 — Kyphoscoliotic EDS ──
    {
        "gene": "PLOD1",
        "protein": "Procollagen-Lysine 2-Oxoglutarate 5-Dioxygenase 1 (Lysyl Hydroxylase 1)",
        "alias": (
            "PLOD1; OMIM gene 153454; kEDS-PLOD1 / EDS VIA #225400; "
            "1p36.22; 727 aa; ~83 kDa mitochondria-associated ER enzyme; "
            "Lysyl hydroxylase 1 hydroxylates specific lysine residues in the collagen telopeptides; "
            "Hydroxylysines are essential for stable pyridinoline crosslinks between collagen fibrils; "
            "LOF → reduced telopeptide hydroxylysine → weak crosslinks → fragile connective tissue; "
            "NEONATAL KYPHOSCOLIOSIS: congenital or rapidly progressive — early sign; "
            "OCULAR FRAGILITY: corneal globe rupture even from minor trauma — eye protection MANDATORY; "
            "PYRIDOXINE (B6) 5 mg/kg/day: ~30% meaningful response — trial mandatory; "
            "Urine LP ratio (lysyl-pyridinoline:hydroxylysyl-pyridinoline) >0.09 diagnostic"
        ),
        "aa": "727 aa",
        "kDa": "~83 kDa",
        "locus": "1p36.22",
        "omim_gene": 153454,
        "omim_disease": 225400,
        "inheritance": "AR — biallelic LOF; exon 6 duplication in Turkish/Middle Eastern populations",
        "gene_class": (
            "PLOD1 encodes Lysyl Hydroxylase 1 (LH1), a collagen-modifying enzyme in the rough ER "
            "that hydroxylates specific lysine residues in collagen telopeptide regions (Gly-X-Lys sequences). "
            "Hydroxylysine residues in telopeptides are the obligatory substrates for lysyl oxidase, "
            "which generates pyridinoline crosslinks between adjacent collagen fibrils — critical for "
            "tensile strength. LOF → reduced LP crosslinks → unravelling fibrils under mechanical stress. "
            "LH1 uses 2-oxoglutarate and O2 as substrates and requires iron, ascorbate, and vitamin B6 — "
            "pyridoxine supplementation increases residual LH1 activity in some hypomorphic alleles."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("Exon 6 duplication (Turkish/Middle Eastern founder)", 0.40),
            ("Truncating (nonsense/frameshift) → null allele", 0.40),
            ("Missense (catalytic triad or cofactor binding) → partial LOF", 0.20),
        ],
        "age_onset_years_range": (0, 3),
        "sex_ratio_M": 0.50,
        "rates": {
            "neonatal_kyphoscoliosis":              0.95,
            "ocular_fragility_globe_rupture_risk":  0.90,
            "generalised_joint_hypermobility":      0.95,
            "hypotonia_neonatal":                   0.90,
            "marfanoid_habitus":                    0.65,
            "skin_hyperextensibility":              0.85,
            "easy_bruising":                        0.70,
            "abnormal_urine_lp_ratio":              0.98,
            "pyridoxine_response_30pct":            0.30,
            "scleral_thinning":                     0.80,
        },
        "hallmarks": [
            "NEONATAL KYPHOSCOLIOSIS: congenital or rapidly progressive — spine X-ray at birth; orthopaedic referral",
            "OCULAR FRAGILITY: corneal rupture from minor impact — protective glasses MANDATORY from diagnosis",
            "URINE LP RATIO >0.09: lysyl-pyridinoline/hydroxylysyl-pyridinoline — cheap diagnostic urine test",
            "PYRIDOXINE B6 TRIAL: 5 mg/kg/day mandatory — ~30% show meaningful scoliosis/strength benefit",
            "HYPOTONIA: neonatal + early childhood — distinguish from CMD/metabolic causes (LP ratio diagnostic)",
            "MARFANOID HABITUS: tall, arachnodactyly, pectus — exclude FBN1 Marfan (different mechanism)",
            "EXON 6 DUPLICATION: 45 kb founder duplication in Turkish/Moroccan/Middle Eastern populations",
            "SCLERAL THINNING: blue sclera (80%) + thin cornea — ophthalmic examination mandatory annually",
        ],
        "treatment_alerts": [
            "PYRIDOXINE B6 5 mg/kg/day: mandatory 6-month trial; monitor scoliosis progression + strength before/after",
            "EYE PROTECTION: polycarbonate protective glasses at all times — globe rupture risk from minor trauma",
            "SCOLIOSIS SURGERY: spinal fusion if Cobb angle >50° — avoid further progression + cardiopulmonary compromise",
            "AVOID CONTACT SPORTS: high-impact activities risk corneal/globe injury + worsening scoliosis",
            "URINE LP RATIO: can monitor treatment response (pyridoxine should reduce LP ratio toward normal)",
        ],
        "organ_system": "spine (kyphoscoliosis) + eyes (fragility) + joints + skin + muscle (hypotonia)",
        "primary_treatment": "Pyridoxine B6 5 mg/kg/day (trial mandatory); ocular protection; scoliosis surveillance + surgery; physio",
    },
    # ── ADAMTS2 — Dermatosparaxis EDS ──
    {
        "gene": "ADAMTS2",
        "protein": "ADAM Metallopeptidase with Thrombospondin Type-1 Motif 2 (Procollagen N-proteinase)",
        "alias": (
            "ADAMTS2; OMIM gene 604539; dEDS #225410; "
            "5q35.3; 1218 aa; ~134 kDa Zn-metalloprotease; "
            "ADAMTS2 cleaves the N-propeptide from procollagens I, II, and III — essential processing step; "
            "LOF → retention of N-propeptide on mature collagen fibrils → abnormal fibril packing → skin fragility; "
            "MOST EXTREME CUTANEOUS FRAGILITY: loose drooping skin folds (neck, axilla, groins) from birth; "
            "STELLATE / HIEROGLYPHIC COLLAGEN FIBRILS on electron microscopy — PATHOGNOMONIC for dEDS; "
            "Umbilical hernia (100%); easy bruising (100%); blue sclerae; short fingers; "
            "Facial features: broad forehead, epicanthal folds, micrognathia"
        ),
        "aa": "1218 aa",
        "kDa": "~134 kDa",
        "locus": "5q35.3",
        "omim_gene": 604539,
        "omim_disease": 225410,
        "inheritance": "AR — biallelic LOF; consanguineous Belgian/Turkish/Saudi founder alleles",
        "gene_class": (
            "ADAMTS2 (Procollagen I/II/III N-proteinase) is a secreted zinc metalloprotease that cleaves "
            "the N-propeptide from procollagens I, II, and III. This cleavage is required for self-assembly "
            "of mature collagen fibrils into the quarter-staggered, banded arrangement visible on EM. "
            "LOF → pN-collagen (N-propeptide-retaining) → abnormal lateral packing → thin, irregular fibrils "
            "with bizarre hieroglyphic shapes on EM. Skin tensile strength is severely reduced; "
            "fibrils cannot interdigitate normally, explaining the drooping, excess skin folds. "
            "The phenotype is the human homologue of 'dermatosparaxis' in cattle and sheep."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("Truncating (nonsense/frameshift/splice) → null allele", 0.65),
            ("Missense (Zn-binding or signal peptide disruption) → LOF", 0.25),
            ("Founder deletion (Belgian/Turkish/Saudi consanguineous)", 0.10),
        ],
        "age_onset_years_range": (0, 2),
        "sex_ratio_M": 0.50,
        "rates": {
            "loose_drooping_excess_skin":        0.98,
            "umbilical_hernia":                  0.98,
            "easy_bruising":                     0.98,
            "blue_sclerae":                      0.80,
            "joint_hypermobility_mild_moderate": 0.70,
            "short_fingers_brachydactyly":       0.55,
            "broad_forehead_epicanthal":         0.60,
            "skin_fragility_tearing":            0.95,
            "inguinal_hernia":                   0.60,
            "hieroglyphic_em_fibrils":           0.98,
        },
        "hallmarks": [
            "DROOPING EXCESS SKIN: lax skin folds at neck/axilla/groins from birth — not just hyperextensibility",
            "SKIN EM PATHOGNOMONIC: stellate/hieroglyphic collagen fibrils — biopsy essential if diagnosis uncertain",
            "UMBILICAL HERNIA 100%: present at birth — surgical repair after age 2-3 years",
            "SKIN FRAGILITY: tearing with minor trauma; wounds heal poorly with wide scars",
            "BLUE SCLERAE 80%: scleral thinning → underlying choroid visible; NOT globe rupture risk (unlike kEDS)",
            "JOINT HYPERMOBILITY: moderate — less severe than cEDS or clEDS as skin laxity dominates",
            "CONSANGUINITY: Turkish, Saudi, Belgian founder alleles — test ADAMTS2 first in consanguineous dEDS",
            "pN-COLLAGEN TESTING: biochemistry labs can quantify N-propeptide retention in fibroblast supernatant",
        ],
        "treatment_alerts": [
            "WOUND CARE: reinforce wound edges; slow dissolving deep sutures; extra-wide steri-strips; avoid tension",
            "HERNIA REPAIR: umbilical and inguinal hernias common; elective repair ≥2-3 years; generous tissue handling",
            "NO DISEASE-MODIFYING THERAPY: ADAMTS2 replacement theoretical; gene therapy in development",
            "SKIN PROTECTION: protective clothing; avoid sharp objects; padding; UV protection (skin more fragile with sun)",
            "OBSTETRIC ANAESTHESIA: spinal/epidural preferred; skin fragility complicates wound closure after caesarean",
        ],
        "organ_system": "skin (extreme fragility + laxity) + abdominal wall (hernias) + sclerae + joints",
        "primary_treatment": "Wound care; hernia repair; skin protection; supportive — no disease-modifying therapy available",
    },
    # ── FBN1 — Marfan Syndrome ──
    {
        "gene": "FBN1",
        "protein": "Fibrillin-1",
        "alias": (
            "FBN1; OMIM gene 134797; Marfan Syndrome #154700; "
            "15q21.1; 2871 aa; ~350 kDa extracellular matrix glycoprotein; "
            "Fibrillin-1 forms the backbone of microfibrils in elastic and non-elastic tissues; "
            "Microfibrils scaffold elastin in aortic wall + sequester latent TGF-beta (LTBP complexes); "
            "LOF → reduced microfibrils → dysregulated TGF-beta signalling (liberated) → aortic root dilatation; "
            "ECTOPIA LENTIS UPWARD: zonular fibre laxity → lens displaces superiorly (DDx homocystinuria = downward); "
            "AORTIC ROOT: surgery threshold 4.5 cm (or 4.0 cm if FH dissection or growth >3 mm/year); "
            "LOSARTAN: blunts AT1R→TGF-beta axis — Level A evidence; beta-blocker adjunct (bisoprolol/atenolol)"
        ),
        "aa": "2871 aa",
        "kDa": "~350 kDa",
        "locus": "15q21.1",
        "omim_gene": 134797,
        "omim_disease": 154700,
        "inheritance": "AD — 25-30% de novo; dominant-negative or haploinsufficiency; genotype-phenotype overlap",
        "gene_class": (
            "FBN1 encodes Fibrillin-1, a cysteine-rich glycoprotein that polymerises into 10-12 nm diameter "
            "extracellular microfibrils in the matrix of aortic media, skin, lung, periosteum, and ciliary zonules. "
            "Microfibrils serve two roles: (1) structural scaffolding for elastic fibres; "
            "(2) sequestration of latent TGF-beta via LTBP binding. LOF → reduced microfibril density → "
            "excess free active TGF-beta → smooth muscle cell apoptosis + matrix metalloproteinase activation → "
            "aortic root dilatation and aneurysm. The zonular fibres of the lens are composed entirely of "
            "fibrillin microfibrils — LOF → ectopia lentis (upward displacement)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("Dominant-negative Cys substitution (disrupts disulphide bonds in EGF-like domain)", 0.45),
            ("Haploinsufficiency (PTC → NMD) → quantitative fibrillin-1 deficit", 0.35),
            ("De novo missense (TGF-beta binding domain disruption)", 0.20),
        ],
        "age_onset_years_range": (0, 30),
        "sex_ratio_M": 0.50,
        "rates": {
            "aortic_root_dilatation":           0.90,
            "ectopia_lentis":                   0.60,
            "arm_span_height_ratio_105":        0.80,
            "reduced_upper_lower_segment":      0.75,
            "pectus_excavatum_carinatum":        0.70,
            "scoliosis":                         0.60,
            "pes_planus":                        0.65,
            "mitral_valve_prolapse":             0.45,
            "spontaneous_pneumothorax":          0.10,
            "dural_ectasia":                     0.65,
        },
        "hallmarks": [
            "ECTOPIA LENTIS UPWARD: slit-lamp annually — upward zonular laxity; DDx homocystinuria (downward + MR)",
            "AORTIC ROOT DILATATION 90%: echo at diagnosis; annual if stable; threshold 4.5 cm → surgical referral",
            "LOSARTAN FIRST-LINE: angiotensin II receptor blocker blunts TGF-beta axis — Level A; start from diagnosis",
            "BETA-BLOCKER ADJUNCT: bisoprolol or atenolol (heart rate control); use WITH losartan not instead",
            "ARM-SPAN HEIGHT RATIO >1.05: screened in children with family history — skeletal diagnostics",
            "DURAL ECTASIA: low back pain + sciatica in Marfan — MRI lumbosacral region if symptomatic",
            "AVOID CONTACT SPORTS: aortic root dilatation + activity restriction; swimming/low-resistance cycling OK",
            "SCOLIOSIS 60%: spinal X-ray at diagnosis; bracing if >20°; surgery if Cobb >45°",
        ],
        "treatment_alerts": [
            "LOSARTAN: first-line — start at diagnosis regardless of aortic size; titrate to 1.4 mg/kg/day",
            "AORTIC SURVEILLANCE: annual echo if stable <4 cm; every 6 months if growing or 4.0-4.5 cm range",
            "SURGICAL THRESHOLD 4.5 cm: valve-sparing aortic root replacement; lower (4.0 cm) if FH of dissection",
            "SLIT-LAMP ANNUALLY: ectopia lentis — corrective lenses; aphakia surgery if cataract or severe lens displacement",
            "PREGNANCY: high-risk; aortic dissection risk increases if root >4.0 cm; pre-conception counselling",
        ],
        "organ_system": "aorta + heart (MVP) + eyes (ectopia lentis) + skeleton (arm span/scoliosis) + dura",
        "primary_treatment": "Losartan (Level A) + bisoprolol; annual echo; slit-lamp; aortic surgery at 4.5 cm; avoid contact sport",
    },
    # ── TGFBR2 — Loeys-Dietz Syndrome 2 ──
    {
        "gene": "TGFBR2",
        "protein": "TGF-beta Receptor Type 2",
        "alias": (
            "TGFBR2; OMIM gene 190182; LDS2 #610168; "
            "3p24.1; 567 aa; ~65 kDa type II TGF-beta receptor; "
            "TGFBR2 is the obligate ligand-binding subunit that complexes with TGFBR1 on TGF-beta binding; "
            "LOF → paradoxically INCREASED TGF-beta signalling (SMAD2/3 overactivation) despite receptor LOF; "
            "MORE AGGRESSIVE AORTIC DISEASE: dissection at smaller diameters than Marfan — surgery at 4.0-4.2 cm; "
            "BIFID UVULA / CLEFT PALATE: oropharyngeal mucosal finding — examine all suspected LDS; "
            "HYPERTELORISM + CRANIOSYNOSTOSIS + CLUB FOOT: skeletal/craniofacial DDx from Marfan; "
            "Extensive arterial tortuosity (carotid, vertebral, intracranial) — full-body vascular imaging"
        ),
        "aa": "567 aa",
        "kDa": "~65 kDa",
        "locus": "3p24.1",
        "omim_gene": 190182,
        "omim_disease": 610168,
        "inheritance": "AD — dominant-negative kinase domain missense; rare haploinsufficiency; de novo 50%",
        "gene_class": (
            "TGFBR2 (TGF-beta Receptor Type 2) is the constitutively active kinase receptor that "
            "captures TGF-beta and transphosphorylates TGFBR1 (ALK5). Counter-intuitively, "
            "LOF mutations in TGFBR2 lead to INCREASED downstream SMAD2/3 phosphorylation — "
            "the paradox is explained by compensatory non-canonical TGF-beta signalling pathways "
            "(JNK, p38 MAPK) and possible autocrine feedback loops. "
            "This TGF-beta signalling excess in aortic smooth muscle cells promotes matrix degradation "
            "(MMP2/MMP9 upregulation) and medial degeneration — explaining why aortic dissection "
            "occurs at smaller diameters in LDS2 than in Marfan syndrome."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("Kinase-domain dominant-negative missense → TGF-beta signalling paradox", 0.80),
            ("Truncating (rare haploinsufficiency allele) → milder LDS phenotype", 0.10),
            ("De novo missense (ectodomain) → ligand-binding disruption", 0.10),
        ],
        "age_onset_years_range": (0, 35),
        "sex_ratio_M": 0.50,
        "rates": {
            "aortic_aneurysm_dilatation":       0.92,
            "bifid_uvula_cleft_palate":         0.70,
            "arterial_tortuosity":              0.85,
            "hypertelorism":                    0.55,
            "club_foot":                        0.40,
            "cervical_spine_instability":       0.35,
            "intracranial_tortuosity":          0.55,
            "easy_bruising":                    0.50,
            "joint_hypermobility_mild":         0.45,
            "aortic_dissection_at_small_size":  0.40,
        },
        "hallmarks": [
            "BIFID UVULA: examine oropharynx in all suspected CT disorders — LDS hallmark sign; cleft palate also",
            "AGGRESSIVE SURGERY THRESHOLD 4.0-4.2 cm: dissection at smaller sizes than Marfan (not 4.5 cm!)",
            "FULL-BODY VASCULAR IMAGING: MRA head to pelvis — extensive arterial tortuosity/aneurysm in celiac/renal",
            "LOSARTAN MANDATORY: TGF-beta overactivation drives aortic disease — AT1R blockade effective",
            "HYPERTELORISM: facial DDx from Marfan (no hypertelorism); LDS has craniofacial + vascular + skeletal",
            "CRANIOSYNOSTOSIS: some LDS2 patients — early suture fusion; CT head in infancy if skull asymmetric",
            "CERVICAL SPINE: instability → neurological risk — spine imaging before any procedures needing neck positioning",
            "INTRACRANIAL TORTUOSITY: carotid/vertebral kinking — MRA before any neck surgery/intubation",
        ],
        "treatment_alerts": [
            "LOSARTAN: angiotensin II receptor blockade — start at diagnosis; uptitrate to 1.4 mg/kg/day target",
            "SURGERY THRESHOLD 4.0 cm: lower than Marfan — refer vascular surgery at 4.0-4.2 cm in LDS2",
            "FULL BODY MRA: at diagnosis then every 1-2 years — aorta + celiac + renal + intracranial vessels",
            "PHYSICAL ACTIVITY RESTRICTION: avoid isometric exercise, contact sports, competitive sports",
            "OBSTETRIC RISK: aortic dissection risk in pregnancy; aortic surgery before pregnancy if root >4.0 cm",
        ],
        "organ_system": "aorta + all arteries (tortuosity/aneurysm) + oropharynx + craniofacial + cervical spine",
        "primary_treatment": "Losartan; surgery at 4.0-4.2 cm aortic root; full-body MRA surveillance; avoid contact sport",
    },
    # ── ACTA2 — Multisystemic Smooth Muscle Dysfunction Syndrome ──
    {
        "gene": "ACTA2",
        "protein": "Actin Alpha 2, Smooth Muscle",
        "alias": (
            "ACTA2; OMIM gene 102620; MSMD / FTAAD6 #613834; "
            "10q23.31; 377 aa; ~42 kDa smooth muscle actin isoform; "
            "Alpha-smooth muscle actin (alpha-SMA) is the main contractile protein of smooth muscle cells; "
            "LOF/DN mutations → impaired smooth muscle cell contractility → aortic aneurysm + multisystem vasculopathy; "
            "ACTA2 Arg179His: MOST SEVERE — congenital heart defects + fixed dilated pupils + patent ductus + Moya Moya; "
            "Other ACTA2 mutations: FTAAD6 (aortic aneurysm without MSMD features); "
            "IRIS FLOCCULI (transillumination defects / iridescent pupil margin fibres) PATHOGNOMONIC; "
            "MOYA MOYA: intracranial occlusive vasculopathy — stroke risk in young adults; aspirin"
        ),
        "aa": "377 aa",
        "kDa": "~42 kDa",
        "locus": "10q23.31",
        "omim_gene": 102620,
        "omim_disease": 613834,
        "inheritance": "AD — dominant-negative; Arg179His most severe (MSMD); other missense → FTAAD6 (aortic only)",
        "gene_class": (
            "ACTA2 encodes smooth muscle alpha-2 actin (alpha-SMA), the major contractile protein "
            "of vascular smooth muscle cells and visceral smooth muscle. "
            "alpha-SMA monomers polymerise into thin filaments that interact with smooth muscle myosin "
            "to generate contractile force regulating vascular tone, peristalsis, and sphincter function. "
            "Dominant-negative ACTA2 mutations (particularly Arg179His) disrupt thin filament assembly, "
            "impairing smooth muscle contractility globally. Smooth muscle dysfunction affects: "
            "(1) aorta/arteries → aneurysm; (2) intracranial vessels → Moya Moya-like occlusion; "
            "(3) gut → hypoperistalsis/pseudo-obstruction; (4) bladder → hypotonic/dysfunctional; "
            "(5) iris → iris flocculi (malformed iris dilator smooth muscle) → pupillary irregularity."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("Arg179His (c.536G>A) — most severe MSMD: PDA + fixed pupils + Moya Moya + aortic", 0.35),
            ("Arg258Cys, Arg258His — FTAAD6: aortic aneurysm ± mild vascular features", 0.45),
            ("Other missense (actin-myosin interface) — variable severity FTAAD6 spectrum", 0.20),
        ],
        "age_onset_years_range": (0, 40),
        "sex_ratio_M": 0.50,
        "rates": {
            "aortic_aneurysm":                    0.90,
            "iris_flocculi":                      0.65,
            "moya_moya_intracranial_occlusion":   0.50,
            "patent_ductus_arteriosus":            0.55,
            "fixed_dilated_pupils":               0.45,
            "intestinal_hypoperistalsis":         0.45,
            "bladder_dysfunction":                0.35,
            "pulmonary_arterial_hypertension":    0.25,
            "stroke_in_young_adult":              0.30,
            "livedo_reticularis":                 0.30,
        },
        "hallmarks": [
            "IRIS FLOCCULI PATHOGNOMONIC: iridescent translucent fibres at pupil margin — slit-lamp or ophthalmoscope",
            "MOYA MOYA-LIKE: intracranial occlusive vasculopathy — MRA for stroke risk; aspirin + statin",
            "ACTA2 Arg179His: most severe — PDA + fixed dilated pupils + Moya Moya + aortic from birth",
            "INTESTINAL HYPOPERISTALSIS: constipation/pseudo-obstruction — laxatives; avoid opioids if possible",
            "BLADDER DYSFUNCTION: urinary retention/overflow — urology referral; intermittent catheterisation if needed",
            "SURGERY THRESHOLD 4.0 cm: more aggressive than Marfan — smooth muscle wall weaker",
            "FULL-BODY VASCULAR IMAGING: aorta + intracranial MRA + pulmonary vasculature at diagnosis",
            "ASPIRIN FOR MOYA MOYA: antiplatelet therapy reduces stroke risk in intracranial occlusive disease",
        ],
        "treatment_alerts": [
            "IRIS FLOCCULI: ask ophthalmologist specifically — often missed unless specifically looked for",
            "MRA HEAD MANDATORY: Moya Moya-like intracranial vasculopathy → stroke risk; annual if found",
            "ASPIRIN 75-100 mg/day: if intracranial occlusive disease (Moya Moya features) confirmed on MRA",
            "SURGERY THRESHOLD: aortic root 4.0 cm (lower than Marfan) — weaker smooth muscle wall",
            "BOWEL/BLADDER MANAGEMENT: high-fibre diet; osmotic laxatives; urology referral for bladder dysfunction",
        ],
        "organ_system": "aorta + intracranial vasculature (Moya Moya) + iris + gut + bladder + pulmonary artery",
        "primary_treatment": "Aortic surveillance (surgery at 4.0 cm); aspirin if Moya Moya; bowel/bladder management; ophthalmology",
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
for _gd in CT_GENES:
    ALL_PATIENTS.extend(_make_patients(_gd))


def get_overview() -> dict:
    n = len(ALL_PATIENTS)
    rate_keys = list(CT_GENES[0]["rates"].keys())
    agg = _aggregate(ALL_PATIENTS, rate_keys)
    return {
        "atlas": "Connective Tissue Atlas",
        "subtitle": "Complete 8-Gene Hereditary Connective Tissue Disorders Reference (EDS + Marfan Spectrum)",
        "total_patients": n,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes": [g["gene"] for g in CT_GENES],
        "diseases": {
            "COL5A1":  "Classical EDS (cEDS) — Most Common EDS; Haploinsufficiency; Wide Atrophic Scars; Beighton ≥5; Joint Hypermobility",
            "COL3A1":  "Vascular EDS (vEDS) — LIFE-THREATENING; Arterial Rupture; Bowel Perforation; Celiprolol Level B; Avoid Angioscopy",
            "TNXB":    "Classical-like EDS (clEDS) — Low Tenascin-X Serum <4 mcg/mL Diagnostic; No Atrophic Scars DDx cEDS; AR",
            "PLOD1":   "Kyphoscoliotic EDS type 1 (kEDS-PLOD1) — Neonatal Kyphoscoliosis; Ocular Fragility Globe Rupture; Pyridoxine B6 Trial",
            "ADAMTS2": "Dermatosparaxis EDS (dEDS) — Extreme Skin Fragility; Loose Drooping Folds; Hieroglyphic EM Fibrils Pathognomonic",
            "FBN1":    "Marfan Syndrome — Aortic Root Dilatation; Ectopia Lentis Upward; Losartan Level A; Surgery 4.5 cm",
            "TGFBR2":  "Loeys-Dietz Syndrome 2 (LDS2) — Bifid Uvula Pathognomonic; More Aggressive Surgery 4.0 cm; Losartan; Arterial Tortuosity",
            "ACTA2":   "MSMD/FTAAD6 — Iris Flocculi Pathognomonic; Moya Moya Stroke Risk; Arg179His Most Severe; Surgery 4.0 cm",
        },
        "aggregate_stats": agg,
        "top_alerts": [
            "COL3A1 vEDS: AVOID colonoscopy/angiography/unnecessary surgery — bowel/arterial perforation risk; celiprolol 400 mg/day",
            "COL3A1 vEDS: Median survival 48 years; first arterial event median age 29 — celiprolol starts at diagnosis",
            "PLOD1 kEDS: Ocular fragility → globe rupture from minor trauma — protective glasses MANDATORY from diagnosis",
            "PLOD1 kEDS: Pyridoxine B6 5 mg/kg/day MANDATORY trial — ~30% respond meaningfully; urine LP ratio diagnostic",
            "FBN1 Marfan: Losartan Level A evidence — start at diagnosis regardless of aortic size; beta-blocker adjunct",
            "TGFBR2 LDS2: Surgery threshold 4.0-4.2 cm (NOT 4.5 cm like Marfan) — dissection at smaller diameters",
            "ACTA2: Iris flocculi PATHOGNOMONIC — specifically request slit-lamp for pupillary margin fibres; MRA for Moya Moya",
            "TNXB clEDS: Low serum tenascin-X (<4 mcg/mL) DIAGNOSTIC — ELISA before gene sequencing; no atrophic scars",
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for gd in CT_GENES:
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
        "atlas": "Connective Tissue Atlas",
        "classification": {
            "EDS_Types": {
                "cEDS_COL5A1":   "Classical EDS — Collagen V fibril regulation; wide atrophic scars + Beighton ≥5 + skin hyperextensibility",
                "vEDS_COL3A1":   "Vascular EDS — Collagen III hollow organ/vessel integrity; LIFE-THREATENING arterial rupture",
                "clEDS_TNXB":    "Classical-like EDS — Tenascin-X matrix stabilisation; no atrophic scars; low serum TNX diagnostic",
                "kEDS_PLOD1":    "Kyphoscoliotic EDS type 1 — Lysyl Hydroxylase 1; telopeptide crosslinking; neonatal kyphoscoliosis + ocular fragility",
                "dEDS_ADAMTS2":  "Dermatosparaxis EDS — Procollagen N-proteinase; N-propeptide retention; extreme skin fragility + loose folds",
            },
            "Marfan_Spectrum": {
                "Marfan_FBN1":   "Marfan Syndrome — fibrillin-1 microfibrils; TGF-beta sequestration; aortic root + ectopia lentis (upward)",
                "LDS2_TGFBR2":   "Loeys-Dietz Syndrome 2 — TGF-beta receptor paradox; bifid uvula; aggressive surgery at 4.0 cm; arterial tortuosity",
            },
            "Smooth_Muscle_Aortopathy": {
                "MSMD_FTAAD6_ACTA2": "MSMD/FTAAD6 — alpha-smooth muscle actin; iris flocculi PATHOGNOMONIC; Moya Moya; multi-system smooth muscle dysfunction",
            },
        },
        "key_diagnostic_rules": {
            "COL3A1_AVOID_PROCEDURES": (
                "Vascular EDS (vEDS) patients have COL3A1-deficient bowel wall and arterial media. "
                "Colonoscopy can cause spontaneous colonic perforation; diagnostic angiography can cause "
                "arterial dissection or rupture at puncture sites. Use CT colography and CTA instead. "
                "Elective surgery should be avoided unless life-threatening — consult vascular surgery."
            ),
            "PLOD1_PYRIDOXINE_TRIAL": (
                "Any patient with biallelic PLOD1 LOF MUST receive a pyridoxine (vitamin B6) trial: "
                "5 mg/kg/day, assessing scoliosis progression and muscle strength at 3 and 6 months. "
                "Approximately 30% show meaningful benefit. LH1 requires pyridoxal-5-phosphate as a "
                "cofactor — supplementation may rescue residual enzyme activity in hypomorphic alleles. "
                "Urine LP ratio (lysyl-pyridinoline/hydroxylysyl-pyridinoline) >0.09 is diagnostic and "
                "may normalise in pyridoxine-responsive patients."
            ),
            "FBN1_LOSARTAN_LEVEL_A": (
                "Losartan (an AT1R blocker) reduces TGF-beta signalling in fibrillin-1-deficient aortic "
                "smooth muscle. Level A evidence from the Marfan Syndrome Trial shows reduction in aortic "
                "root growth rate compared to placebo. Start from diagnosis regardless of aortic size. "
                "Target dose 1.4 mg/kg/day. Add beta-blocker (bisoprolol or atenolol) for heart rate control. "
                "Monitor potassium and creatinine (AT1R blockade effect)."
            ),
            "TGFBR2_SURGERY_THRESHOLD": (
                "In LDS2 (TGFBR2 mutations), aortic dissection occurs at SMALLER diameters than Marfan syndrome. "
                "The surgical threshold is 4.0-4.2 cm (not 4.5 cm as for FBN1 Marfan). "
                "Full-body MRA (head-to-pelvis) is required at diagnosis to detect arterial tortuosity "
                "and aneurysms at all levels — celiac, renal, iliac, intracranial vessels are all at risk."
            ),
            "ACTA2_IRIS_FLOCCULI": (
                "Iris flocculi (also termed iridescent pupil margin fibres or iris transillumination defects) "
                "are translucent fibrous processes at the pupillary margin, visible on slit-lamp examination. "
                "They represent malformed iris dilator smooth muscle. Iris flocculi are PATHOGNOMONIC for ACTA2 "
                "mutations (particularly Arg179His) and are not seen in Marfan/LDS or other CTDs. "
                "Specifically request slit-lamp exam for this finding in any young patient with aortic disease + "
                "pupillary irregularity."
            ),
            "TNXB_LOW_SERUM_TNX": (
                "Biallelic TNXB LOF (classical-like EDS) results in absent or severely reduced serum Tenascin-X. "
                "Serum TNX ELISA (<4 mcg/mL diagnostic) is a rapid, non-invasive, and inexpensive test "
                "that can guide gene panel sequencing in clEDS. The absence of atrophic scars "
                "is the key clinical DDx from cEDS (COL5A1) — if joint hypermobility + skin laxity WITHOUT "
                "atrophic scars, test serum TNX before ordering a skin biopsy."
            ),
            "CASCADE_TESTING": (
                "EDS types with AR inheritance (TNXB, PLOD1, ADAMTS2): 25% recurrence per pregnancy; "
                "carrier parents should receive counselling. "
                "AD conditions (COL5A1, COL3A1, FBN1, TGFBR2, ACTA2): 50% risk per child; "
                "de novo mutations (~25-50% in COL3A1, FBN1) reduce but do not eliminate cascade testing need. "
                "First-degree relatives of all 8 conditions should be offered gene-specific cascade testing."
            ),
        },
        "treatment_hierarchy": {
            "COL3A1_vEDS": [
                "1. Celiprolol 400 mg/day — beta1-antagonist/beta2-agonist; reduces arterial events; Level B",
                "2. Avoid invasive procedures (colonoscopy, angiography, elective surgery) — perforation risk",
                "3. Emergency card + vascular surgical team awareness at all acute presentations",
                "4. Obstetric: caesarean preferred; high-risk obstetric centre; post-delivery uterine rupture risk",
                "5. Psychological support + genetic counselling; cascade testing of all first-degree relatives",
            ],
            "PLOD1_kEDS": [
                "1. Pyridoxine B6 5 mg/kg/day — mandatory trial; monitor urine LP ratio + scoliosis at 3/6 months",
                "2. Ocular protection: polycarbonate protective glasses at all times from diagnosis",
                "3. Scoliosis surveillance: spine X-ray every 6 months; bracing if Cobb 20-50°; surgery if >50°",
                "4. Physiotherapy: core strengthening + joint stabilisation; swimming preferred",
                "5. Annual ophthalmology: slit-lamp for scleral thinning + globe integrity",
            ],
            "FBN1_Marfan": [
                "1. Losartan 1.4 mg/kg/day — start from diagnosis; Level A; AT1R blockade blunts TGF-beta",
                "2. Bisoprolol/atenolol — beta-blocker adjunct; heart rate control reduces aortic wall stress",
                "3. Annual echocardiogram — aortic root size; every 6 months if growing or >4.0 cm",
                "4. Surgery referral: 4.5 cm (or 4.0 cm if FH dissection); valve-sparing root replacement",
                "5. Annual slit-lamp — ectopia lentis; annual spine X-ray — scoliosis; activity restriction",
            ],
            "TGFBR2_LDS2": [
                "1. Losartan — TGF-beta blockade; start from diagnosis; same as Marfan regimen",
                "2. Full-body MRA at diagnosis: aorta + celiac/renal/intracranial vessels",
                "3. Surgery at 4.0-4.2 cm (lower threshold than Marfan) — smaller diameter dissection risk",
                "4. Cervical spine MRI — instability; inform anaesthetist before any procedure requiring neck flexion",
                "5. Activity restriction: no contact sport; no isometric exercise; low-impact cardio only",
            ],
            "ACTA2_MSMD": [
                "1. Aspirin 75-100 mg/day — if intracranial Moya Moya confirmed on MRA (antiplatelet protection)",
                "2. Full-body vascular MRA + MRI brain — aorta + intracranial surveillance",
                "3. Surgery at 4.0 cm aortic root — weaker smooth muscle wall than FBN1",
                "4. Bowel management: osmotic laxatives + high fibre; urology referral for bladder dysfunction",
                "5. Ophthalmology — iris flocculi (slit-lamp); annual optic nerve assessment",
            ],
        },
        "genes_summary": {g["gene"]: g["alias"] for g in CT_GENES},
    }


# ── Convenience wrappers (called by api_backend.py) ──
def overview():
    return get_overview()


def breakdown():
    return get_breakdown()


def definitions():
    return get_definitions()
