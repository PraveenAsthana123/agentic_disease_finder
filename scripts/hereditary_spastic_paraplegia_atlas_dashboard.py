#!/usr/bin/env python3
"""Hereditary-Spastic-Paraplegia-Atlas — Complete 8-Gene Hereditary Spastic Paraplegia Atlas
SPAST   (Spastin; 616 aa; 2p22.3; AD;
          Most common HSP gene — ~40% of all AD-HSP families; AAA+ ATPase microtubule-severing enzyme;
          Haploinsufficiency → corticospinal tract axon tip degeneration → progressive spastic paraplegia;
          Incomplete penetrance — 40% of carriers asymptomatic at age 60; onset 1–76 years;
          Pure HSP with bladder involvement common; ankle-foot orthosis + physiotherapy standard) ·
ATL1    (Atlastin-1 GTPase; 558 aa; 14q22.1; AD;
          Second most common AD-HSP gene — ~10% of AD-HSP; childhood onset (before age 10 typical);
          GTPase mediates ER membrane fusion and tubular ER network formation;
          Pure HSP — mild slow-progression phenotype despite early onset;
          Early onset does NOT imply severity — counselling mandatory) ·
REEP1   (Receptor Expression-Enhancing Protein 1; 201 aa; 2p11.2; AD;
          ~6% of AD-HSP; ER membrane curvature and axonal ER morphogenesis;
          Pure HSP — mild to moderate; axonal CMT overlap (CMT2B5) occurs;
          Haploinsufficiency → ER tubule shaping defect → distal axon degeneration) ·
SPG11   (Spatacsin; 2443 aa; 15q21.1; AR;
          Most common AR-HSP gene — ~20–25% of AR-HSP; thin corpus callosum (TCC) in >90%;
          Spatacsin–spastizin complex required for lysosomal axonal autophagy;
          Complicated HSP: intellectual disability (progressive), peripheral neuropathy, TCC;
          Brain MRI mandatory — TCC on T1 virtually obligatory in SPG11) ·
ZFYVE26 (Spastizin; 2539 aa; 14q24.1; AR;
          Kjellin syndrome: complicated HSP + macular degeneration + intellectual disability;
          FYVE-domain protein; pairs with spatacsin (SPG11) in autophagosome maturation complex;
          Annual fundoscopy + OCT mandatory — progressive macular degeneration → blindness 3rd–4th decade;
          TCC also present; cognitive decline progressive) ·
CYP7B1  (Oxysterol 7α-Hydroxylase; 506 aa; 8q12.3; AR;
          SPG5A — pure HSP; bile acid/oxysterol metabolism enzyme;
          Loss → accumulation of 25-hydroxycholesterol + 27-hydroxycholesterol — neurotoxic to spinal axons;
          Plasma oxysterols (25-HC, 27-HC) measurable — diagnostic biomarker + treatment monitor;
          Chenodeoxycholic acid (CDCA) replacement → normalises bile acids — neuroprotective benefit) ·
KIF1A   (KIF1A Kinesin Motor; 1826 aa; 2q37.3; AR (SPG30) / AD (KAND);
          Kinesin-3 motor; anterograde axonal transport of dense-core vesicles and synaptic vesicle precursors;
          AR biallelic LOF → SPG30 — adult onset, milder pure HSP;
          AD de novo gain-of-function/dominant-negative → KAND — severe: intellectual disability, optic atrophy, cerebellar atrophy;
          AR vs AD prognosis dramatically different — genotype class MANDATORY before counselling) ·
DDHD2   (DDHD Domain-Containing Protein 2; 711 aa; 8p11.23; AR;
          SPG54 — complicated HSP with intellectual disability and thin corpus callosum;
          DDHD2 is a phospholipase A1 acting on phosphatidic acid in brain lipid metabolism;
          Brain MR spectroscopy lipid peak at 1.3 ppm in basal ganglia/white matter PATHOGNOMONIC for SPG54;
          Annual cognitive assessment mandatory — intellectual disability progressive)
320-patient aggregate cohort (8 × 40, seeds 1446–1453)
"""

import random

SEED_BASE = 1446

HSP_GENES = [
    # ── SPAST — Most common AD-HSP ──
    {
        "gene": "SPAST",
        "protein": "Spastin AAA+ ATPase Microtubule-Severing Enzyme",
        "alias": (
            "SPAST; SPG4; OMIM gene 604277; SPG4 OMIM 182601; 2p22.3; 616 aa; ~60 kDa; "
            "Most common HSP gene — ~40% of all autosomal dominant HSP families; "
            "Spastin severs microtubules using ATPase activity — required for axon tip remodelling; "
            "Haploinsufficiency mechanism — microtubule accumulation at axon tips → corticospinal tract degeneration; "
            "Incomplete penetrance — ~40% of obligate SPAST heterozygotes asymptomatic at age 60; "
            "Onset highly variable: 1–76 years (mean ~30); pure HSP most common; "
            "Bladder detrusor hyper-reflexia in 50–70% — urgency, urge incontinence; "
            "Physiotherapy + AFO mainstay; baclofen/tizanidine for spasticity; "
            "No disease-modifying therapy — annual neurological review standard; "
            "Presymptomatic testing: genetic counselling mandatory — incomplete penetrance must be explained"
        ),
        "aa": "616 aa",
        "kDa": "~60 kDa",
        "locus": "2p22.3",
        "omim_gene": 604277,
        "omim_disease": 182601,
        "inheritance": "AD — haploinsufficiency; incomplete penetrance ~40% asymptomatic at age 60",
        "gene_class": (
            "SPAST encodes spastin, a microtubule-severing AAA+ ATPase. Spastin severs microtubules at "
            "axon tips to allow remodelling and branching; haploinsufficiency → excessive stable "
            "microtubule accumulation → impaired axon tip trafficking → length-dependent corticospinal "
            "tract degeneration (longest axons most vulnerable). SPG4 is the most common hereditary "
            "spastic paraplegia gene worldwide, accounting for ~40% of AD-HSP families. Phenotype is "
            "pure HSP in the majority: slowly progressive leg stiffness, spastic gait, and bladder "
            "urgency. Penetrance is incomplete (~60% by age 60), making presymptomatic testing complex "
            "and genetic counselling essential. >500 distinct SPAST variants described; missense, "
            "nonsense, frameshift, and large deletions (MLPA required for copy-number variants)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("SPAST missense (exon 7–15 hotspot) — prevalent pan-ethnic", 0.35),
            ("SPAST frameshift — premature stop, haploinsufficiency", 0.25),
            ("SPAST large exon deletion — MLPA required", 0.15),
            ("SPAST nonsense (exon 1–6) — juvenile onset", 0.12),
            ("SPAST splice-site variant — reduced expression", 0.08),
            ("SPAST 5′-UTR regulatory variant — reduced transcription", 0.05),
        ],
        "age_onset_years_range": (10, 55),
        "sex_ratio_M": 0.52,
        "rates": {
            "progressive_spastic_paraplegia":   0.98,
            "bilateral_lower_limb_spasticity":  0.98,
            "bladder_detrusor_hyperreflexia":   0.60,
            "hyperreflexia_brisk_deep_tendon":  0.95,
            "extensor_plantar_response":        0.70,
            "gait_scissors_pattern":            0.65,
            "ankle_clonus":                     0.55,
            "mild_distal_weakness_legs":        0.45,
            "foot_deformity_pes_cavus":         0.30,
            "family_history_positive":          0.75,
            "asymptomatic_carrier_family":      0.35,
            "physiotherapy_required":           0.90,
            "afo_ankle_foot_orthosis":          0.50,
            "baclofen_or_tizanidine":           0.55,
        },
        "critical_alerts": [
            "SPAST-INCOMPLETE-PENETRANCE: ~40% of SPAST heterozygotes asymptomatic by age 60 — genetic counselling MANDATORY before presymptomatic testing",
            "SPAST-MLPA-REQUIRED: Large exonic deletions (15% of pathogenic alleles) missed by standard sequencing — MLPA mandatory in SPAST-negative AD-HSP",
            "SPAST-BLADDER-ASSESSMENT: Bladder urgency in 50–70% — urodynamics if incontinence; anticholinergics worsen cognition — prefer mirabegron",
            "SPAST-ANNUAL-REVIEW: Annual neurological review — track progression, adjust PT and AFO, monitor bladder; disease severity scores (SPRS) documented",
            "SPAST-PRESYMPTOMATIC-COUNSELLING: Incomplete penetrance means a test-positive family member may never develop disease — do not imply certainty",
        ],
        "key_ddx_rules": [
            "Progressive spastic paraplegia + positive family history (AD) → SPAST panel first (40% yield)",
            "AD-HSP with bladder symptoms + no other features → SPAST most likely",
            "AD-HSP panel negative → check MLPA for SPAST deletions + consider ATL1/REEP1",
        ],
    },

    # ── ATL1 — Most common childhood-onset AD-HSP ──
    {
        "gene": "ATL1",
        "protein": "Atlastin-1 GTPase Endoplasmic Reticulum Fusion Enzyme",
        "alias": (
            "ATL1; SPG3A; OMIM gene 606439; SPG3A OMIM 182600; 14q22.1; 558 aa; ~63 kDa; "
            "Second most common AD-HSP — ~10% of AD-HSP families; "
            "Atlastin-1 GTPase mediates homotypic ER membrane fusion — essential for tubular ER network; "
            "Childhood onset (typically before age 10, often infancy/toddler); "
            "Pure HSP — mild slow-progressive phenotype despite childhood onset; "
            "Early onset does NOT predict severity — do not over-reassure or alarm; "
            "Dominant-negative and haploinsufficiency mechanisms both described; "
            "Physiotherapy from diagnosis — gait training, stretching; SPG3A among most stable AD-HSPs"
        ),
        "aa": "558 aa",
        "kDa": "~63 kDa",
        "locus": "14q22.1",
        "omim_gene": 606439,
        "omim_disease": 182600,
        "inheritance": "AD — dominant negative + haploinsufficiency; childhood onset typical",
        "gene_class": (
            "ATL1 encodes atlastin-1, a dynamin-related GTPase that mediates homotypic fusion of ER "
            "membranes to form the interconnected tubular ER network. This tubular ER is essential for "
            "axonal ER morphology; ATL1 haploinsufficiency or dominant-negative mutations impair ER "
            "tubule branching in long corticospinal axons. SPG3A accounts for ~10% of AD-HSP and is "
            "the most common cause of childhood-onset hereditary spastic paraplegia (onset <10 years, "
            "frequently <3 years). Despite early onset, disease course is typically slow and mild — "
            "patients often remain ambulatory throughout adult life with physiotherapy. Three common "
            "founder variants: R239C (European), R495W (pan-ethnic), P342S. CMT2 overlap rare."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("ATL1 R239C missense — European founder, most common", 0.30),
            ("ATL1 R495W missense — pan-ethnic common", 0.22),
            ("ATL1 P342S missense — moderate phenotype", 0.15),
            ("ATL1 other missense (GTPase domain) — variable", 0.20),
            ("ATL1 frameshift — haploinsufficiency, adult onset", 0.08),
            ("ATL1 de novo missense — sporadic childhood onset", 0.05),
        ],
        "age_onset_years_range": (1, 15),
        "sex_ratio_M": 0.50,
        "rates": {
            "progressive_spastic_paraplegia":   0.98,
            "childhood_onset_before_10":        0.85,
            "mild_progression":                 0.80,
            "pure_hsp_phenotype":               0.92,
            "bladder_mild_urgency":             0.30,
            "bilateral_leg_spasticity":         0.98,
            "ankle_clonus":                     0.60,
            "brisk_knee_ankle_jerks":           0.95,
            "pes_cavus":                        0.25,
            "family_history_ad":                0.80,
            "ambulatory_all_life":              0.85,
            "physiotherapy_required":           0.88,
        },
        "critical_alerts": [
            "ATL1-EARLY-ONSET-MILD: Childhood onset (infancy to age 10) does NOT predict severe disability — most patients remain ambulatory lifelong",
            "ATL1-DO-NOT-OVER-TREAT: Pure mild phenotype — avoid unnecessary procedures; physiotherapy + AFO adequate in most",
            "ATL1-FOUNDER-VARIANTS: R239C and R495W account for ~52% of ATL1 alleles — targeted testing cost-effective in European families",
            "ATL1-ANNUAL-REVIEW: Annual physiotherapy review mandatory — stretching, gait training, botulinum toxin if severe focal spasticity",
        ],
        "key_ddx_rules": [
            "HSP onset <10 years + AD family history → ATL1 (SPG3A) first (10% of AD-HSP, highest yield childhood-onset)",
            "Pure HSP + childhood onset + slow progression + ambulatory → ATL1 most likely (SPAST second)",
            "ATL1 negative childhood-onset AD-HSP → consider REEP1, SPAST (later onset family members)",
        ],
    },

    # ── REEP1 — AD-HSP ER tubule shaping ──
    {
        "gene": "REEP1",
        "protein": "Receptor Expression-Enhancing Protein 1 ER Membrane Curvature",
        "alias": (
            "REEP1; SPG31; OMIM gene 609139; SPG31 OMIM 610250; 2p11.2; 201 aa; ~22 kDa; "
            "~6% of AD-HSP; ER membrane curvature protein — shapes tubular ER in axons; "
            "REEP1 oligomerises to generate high-curvature ER membrane — essential for axonal ER architecture; "
            "Haploinsufficiency → ER tubule morphogenesis defect → distal axon degeneration; "
            "Pure HSP — mild to moderate; adult onset typical (20–40 years); "
            "Axonal CMT2 overlap (CMT2B5 phenotype) occurs in some families; "
            "Physiotherapy + AFO mainstay; bladder involvement in 30%"
        ),
        "aa": "201 aa",
        "kDa": "~22 kDa",
        "locus": "2p11.2",
        "omim_gene": 609139,
        "omim_disease": 610250,
        "inheritance": "AD — haploinsufficiency; adult onset 20–40 years typical",
        "gene_class": (
            "REEP1 encodes Receptor Expression-Enhancing Protein 1, a transmembrane ER-shaping protein. "
            "REEP1 uses hairpin turns in its transmembrane helices to generate high-membrane curvature, "
            "stabilising the thin tubular ER characteristic of distal axons. REEP1 interacts with "
            "spastin (SPAST) and atlastin-1 (ATL1) — all three are core components of the ER morphogenesis "
            "machinery in axons. Haploinsufficiency → ER tubule shaping defect → length-dependent "
            "corticospinal tract axon degeneration → progressive spastic paraplegia. SPG31 accounts "
            "for ~6% of AD-HSP. Most patients have pure HSP; a CMT2B5 axonal neuropathy overlap "
            "phenotype occurs in some families. Onset is adult (typically 20–40 years). MLPA recommended "
            "for copy-number variants if sequencing negative."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("REEP1 missense (transmembrane domain) — dominant negative + LOF", 0.35),
            ("REEP1 frameshift — haploinsufficiency", 0.25),
            ("REEP1 large deletion — MLPA detected", 0.15),
            ("REEP1 nonsense — premature stop, NMD", 0.15),
            ("REEP1 splice-site — partial exon skipping", 0.10),
        ],
        "age_onset_years_range": (18, 50),
        "sex_ratio_M": 0.51,
        "rates": {
            "progressive_spastic_paraplegia":   0.97,
            "adult_onset_20_40":                0.75,
            "pure_hsp_phenotype":               0.85,
            "cmt2_axonal_neuropathy_overlap":   0.15,
            "bilateral_leg_spasticity":         0.97,
            "bladder_urgency":                  0.30,
            "hyperreflexia_lower_limbs":        0.92,
            "distal_weakness_mild":             0.35,
            "pes_cavus":                        0.20,
            "family_history_ad":                0.78,
            "physiotherapy_required":           0.85,
            "afo_required":                     0.40,
        },
        "critical_alerts": [
            "REEP1-MLPA: Copy-number deletions (~15% of REEP1 alleles) missed by sequencing — MLPA mandatory in REEP1-negative AD-HSP suspicion",
            "REEP1-CMT2-OVERLAP: CMT2B5 axonal neuropathy occurs — NCS/EMG if distal weakness or sensory loss present",
            "REEP1-SPASTIN-ATLASTIN-PATHWAY: Same ER morphogenesis pathway as SPAST and ATL1 — pathway-based therapy research active",
            "REEP1-ANNUAL-REVIEW: Annual neurology + physiotherapy review; SPRS score annually to document trajectory",
        ],
        "key_ddx_rules": [
            "AD-HSP adult onset + SPAST/ATL1 negative → REEP1 panel (6% of AD-HSP)",
            "AD-HSP + axonal peripheral neuropathy features → REEP1 CMT2B5 overlap; also consider BSCL2",
            "REEP1 + SPAST + ATL1 all negative in AD-HSP → rare AD genes (ALDH18A1, KIF1A dominant, SPG8)",
        ],
    },

    # ── SPG11 — Most common AR-HSP with thin corpus callosum ──
    {
        "gene": "SPG11",
        "protein": "Spatacsin Lysosomal Axonal Autophagy Scaffold",
        "alias": (
            "SPG11; KIAA1840; OMIM gene 610844; SPG11 OMIM 604360; 15q21.1; 2443 aa; ~280 kDa; "
            "Most common AR-HSP gene — ~20–25% of all AR-HSP; thin corpus callosum (TCC) >90%; "
            "Spatacsin–spastizin (ZFYVE26) complex required for initiation of lysosomal axonal autophagy; "
            "Loss → autophagosome–lysosome fusion defect → axonal cargo accumulation → neurodegeneration; "
            "Complicated HSP: progressive intellectual disability + peripheral neuropathy + TCC; "
            "Brain MRI T1 MANDATORY in AR-HSP — TCC virtually obligatory in SPG11; "
            "Early rehabilitation input; occupational therapy; cognitive support from diagnosis"
        ),
        "aa": "2443 aa",
        "kDa": "~280 kDa",
        "locus": "15q21.1",
        "omim_gene": 610844,
        "omim_disease": 604360,
        "inheritance": "AR — biallelic LOF; predominantly frameshift/nonsense/splice",
        "gene_class": (
            "SPG11 encodes spatacsin, a large scaffold protein that forms a heteromeric complex with "
            "spastizin (ZFYVE26/SPG15). The spatacsin-spastizin complex is required on the lysosomal "
            "membrane for the biogenesis of autophagic lysosome reformation (ALR) — a recycling pathway "
            "that regenerates functional lysosomes after autophagy. Loss of SPG11 → ALR failure → "
            "lysosomal tubule and cargo accumulation in axons → progressive axonal degeneration. "
            "SPG11 is the most common AR-HSP gene worldwide, causing a complicated phenotype: "
            "progressive spastic paraplegia, mild-to-moderate intellectual disability (progressive), "
            "thin corpus callosum (TCC) visible on MRI T1 in >90%, and axonal peripheral neuropathy. "
            "Loss of ambulation typically occurs in the 3rd–4th decade. Truncating variants predominate. "
            "Surveillance: annual cognitive assessment, NCS/EMG every 2 years."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("SPG11 biallelic frameshift (compound heterozygous) — most common pan-ethnic", 0.30),
            ("SPG11 nonsense + frameshift (compound het) — North African/Middle East common", 0.22),
            ("SPG11 homozygous frameshift — consanguineous families", 0.20),
            ("SPG11 splice-site + frameshift — loss of spatacsin complex", 0.15),
            ("SPG11 large deletion (MLPA) + frameshift", 0.08),
            ("SPG11 missense (severe LOF effect) — rare", 0.05),
        ],
        "age_onset_years_range": (8, 25),
        "sex_ratio_M": 0.50,
        "rates": {
            "thin_corpus_callosum_mri":         0.92,
            "progressive_intellectual_disability": 0.80,
            "spastic_paraplegia_bilateral":     0.98,
            "axonal_peripheral_neuropathy":     0.75,
            "loss_of_ambulation_3rd_4th_decade": 0.60,
            "cerebellar_atrophy_mild":          0.30,
            "parkinsonism_late":                0.15,
            "bladder_neurogenic":               0.55,
            "dysarthria":                       0.40,
            "cognitive_decline_progressive":    0.78,
            "consanguinity_in_family":          0.35,
            "physiotherapy_required":           0.95,
            "wheelchair_by_age_40":             0.45,
        },
        "critical_alerts": [
            "SPG11-TCC-MANDATORY-MRI: Brain MRI T1 MANDATORY in all AR-HSP — thin corpus callosum (TCC) present >90% of SPG11; absence makes SPG11 unlikely",
            "SPG11-COGNITIVE-PROGRESSIVE: Intellectual disability is progressive — early neuropsychological assessment and cognitive support from diagnosis",
            "SPG11-AMBULATION-LOSS: Loss of ambulation expected 3rd–4th decade — early PT + OT planning, wheelchair assessment by age 30–35",
            "SPG11-ANNUAL-COGNITION: Annual cognitive assessment (MMSE/MoCA) + NCS/EMG every 2 years — track neuropathy progression",
            "SPG11-SPG15-COMPLEX: Spatacsin and spastizin form a complex — if SPG11 negative, test ZFYVE26 (SPG15) for same pathway/phenotype",
        ],
        "key_ddx_rules": [
            "AR-HSP + thin corpus callosum + cognitive decline → SPG11 first (most common AR-HSP worldwide ~20%)",
            "SPG11 negative + TCC + macular degeneration → ZFYVE26 (SPG15, Kjellin syndrome)",
            "AR-HSP + TCC + no cognitive decline → consider KIAA0196 (SPG8), CYP7B1, other AR-HSP",
        ],
    },

    # ── ZFYVE26 — Kjellin syndrome (SPG15) ──
    {
        "gene": "ZFYVE26",
        "protein": "Spastizin FYVE-Domain Autophagosome Maturation Protein",
        "alias": (
            "ZFYVE26; SPG15; Spastizin; OMIM gene 612015; Kjellin OMIM 270700; 14q24.1; 2539 aa; ~285 kDa; "
            "SPG15 / Kjellin syndrome: complicated HSP + macular degeneration + intellectual disability; "
            "Spastizin FYVE domain binds PI3P on autophagosomes — required for autophagosome maturation; "
            "Spastizin–spatacsin (SPG11) heteromeric complex — same ALR recycling pathway; "
            "Annual fundoscopy + OCT mandatory — macular degeneration progressive → blindness 3rd–4th decade; "
            "TCC on brain MRI (similar to SPG11); cognitive decline progressive; "
            "Kjellin Finnish founder: p.Arg2051Gln — first described in Swedish families"
        ),
        "aa": "2539 aa",
        "kDa": "~285 kDa",
        "locus": "14q24.1",
        "omim_gene": 612015,
        "omim_disease": 270700,
        "inheritance": "AR — biallelic LOF; Kjellin syndrome; Finnish-Swedish founder",
        "gene_class": (
            "ZFYVE26 encodes spastizin, a FYVE-domain protein that binds to phosphatidylinositol 3-phosphate "
            "(PI3P) on autophagosomal membranes and partners with spatacsin (SPG11) in the spatacsin-spastizin "
            "complex required for autophagic lysosome reformation (ALR). Loss of ZFYVE26 → ALR failure → "
            "lysosomal accumulation in axons → complicated spastic paraplegia identical in pathomechanism "
            "to SPG11. SPG15 (Kjellin syndrome) is clinically distinguished from SPG11 by the addition of "
            "progressive central retinal/macular degeneration causing visual loss in the 3rd–4th decade. "
            "Intellectual disability and thin corpus callosum are shared with SPG11. Finnish-Swedish "
            "Kjellin syndrome founder mutation p.Arg2051Gln is prevalent in Scandinavia. Annual fundoscopy "
            "with OCT and visual fields mandatory to track macular degeneration and plan visual rehabilitation."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("ZFYVE26 p.Arg2051Gln — Finnish-Swedish Kjellin founder", 0.25),
            ("ZFYVE26 biallelic frameshift — pan-ethnic", 0.25),
            ("ZFYVE26 nonsense + frameshift (compound het)", 0.20),
            ("ZFYVE26 homozygous nonsense — consanguineous", 0.15),
            ("ZFYVE26 splice-site + frameshift", 0.10),
            ("ZFYVE26 large deletion (MLPA)", 0.05),
        ],
        "age_onset_years_range": (10, 30),
        "sex_ratio_M": 0.50,
        "rates": {
            "progressive_macular_degeneration": 0.90,
            "visual_loss_3rd_4th_decade":       0.70,
            "thin_corpus_callosum_mri":         0.88,
            "intellectual_disability":          0.80,
            "spastic_paraplegia_bilateral":     0.98,
            "axonal_peripheral_neuropathy":     0.65,
            "retinal_pigment_epithelium_change": 0.75,
            "cognitive_decline_progressive":    0.78,
            "bladder_neurogenic":               0.50,
            "loss_ambulation_4th_decade":       0.50,
            "consanguinity":                    0.30,
            "low_vision_aids_required":         0.60,
        },
        "critical_alerts": [
            "ZFYVE26-ANNUAL-FUNDUS: Annual fundoscopy + OCT + visual fields MANDATORY — macular degeneration progressive → blindness 3rd–4th decade; early low-vision referral",
            "ZFYVE26-TCC-MRI: Brain MRI T1 mandatory — TCC in ~88%; essentially indistinguishable from SPG11 on MRI alone",
            "ZFYVE26-KJELLIN-FOUNDER: p.Arg2051Gln Scandinavian founder — targeted testing cost-effective in Finnish/Swedish families before full sequencing",
            "ZFYVE26-OPHTHALMOLOGY-REFERRAL: ERG + pattern VEP at diagnosis to establish retinal baseline; low-vision rehabilitation when acuity <0.3",
            "ZFYVE26-COGNITIVE-SUPPORT: Progressive intellectual disability — early neuropsychological testing, educational support, cognitive aids",
        ],
        "key_ddx_rules": [
            "AR-HSP + TCC + macular degeneration + intellectual disability → ZFYVE26 (SPG15/Kjellin) first",
            "SPG11 negative + TCC + visual loss → ZFYVE26 — same pathway, different gene (spastizin vs spatacsin)",
            "Kjellin syndrome in Scandinavian family → targeted ZFYVE26 p.Arg2051Gln before full sequencing",
        ],
    },

    # ── CYP7B1 — SPG5A, oxysterol accumulation ──
    {
        "gene": "CYP7B1",
        "protein": "Oxysterol 7α-Hydroxylase Bile Acid Synthesis Cytochrome P450",
        "alias": (
            "CYP7B1; SPG5A; OMIM gene 603711; SPG5A OMIM 270800; 8q12.3; 506 aa; ~57 kDa; "
            "Pure HSP — adult onset; bile acid/oxysterol metabolism cytochrome P450; "
            "CYP7B1 loss → failure to hydroxylate 25-hydroxycholesterol (25-HC) and 27-hydroxycholesterol (27-HC); "
            "Plasma oxysterol accumulation (25-HC, 27-HC) is neurotoxic to spinal cord axons; "
            "Plasma oxysterols measurable — diagnostic biomarker AND treatment monitor; "
            "Chenodeoxycholic acid (CDCA) replacement → normalises bile acid synthesis → lowers oxysterols; "
            "CDCA is the only potential disease-modifying therapy in any hereditary spastic paraplegia; "
            "Pure HSP phenotype — no cognitive or retinal involvement"
        ),
        "aa": "506 aa",
        "kDa": "~57 kDa",
        "locus": "8q12.3",
        "omim_gene": 603711,
        "omim_disease": 270800,
        "inheritance": "AR — biallelic LOF; oxysterol accumulation neurotoxic",
        "gene_class": (
            "CYP7B1 encodes oxysterol 7α-hydroxylase, a cytochrome P450 enzyme in the 'acidic' bile acid "
            "synthesis pathway. CYP7B1 hydroxylates 25-hydroxycholesterol and 27-hydroxycholesterol at the "
            "7α position, converting them to primary bile acid precursors. Biallelic loss-of-function → "
            "oxysterol 25-HC and 27-HC accumulate in plasma and brain to neurotoxic levels. These "
            "oxysterols are potent inducers of apoptosis in spinal cord neurons, causing length-dependent "
            "corticospinal tract and spinocerebellar tract degeneration. Unlike most HSP genes, CYP7B1 "
            "offers a measurable plasma biomarker (oxysterols) and a potential therapeutic target: "
            "chenodeoxycholic acid (CDCA) supplementation suppresses the alternative oxysterol pathway, "
            "lowering 25-HC/27-HC plasma levels. Early treatment may be neuroprotective, though clinical "
            "trial data remain limited. CYP7B1 also causes neonatal cholestatic liver disease (separate, "
            "severe alleles) — liver function should be monitored in biallelic carriers."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("CYP7B1 missense (substrate-binding domain LOF) — Mediterranean/European", 0.30),
            ("CYP7B1 biallelic frameshift — oxysterol accumulation confirmed", 0.25),
            ("CYP7B1 homozygous nonsense — consanguineous, severe", 0.20),
            ("CYP7B1 compound het (missense + splice-site) — pan-ethnic", 0.15),
            ("CYP7B1 large deletion — MLPA required", 0.10),
        ],
        "age_onset_years_range": (15, 50),
        "sex_ratio_M": 0.52,
        "rates": {
            "pure_hsp_phenotype":               0.95,
            "progressive_spastic_paraplegia":   0.97,
            "elevated_plasma_25hc":             0.95,
            "elevated_plasma_27hc":             0.93,
            "normal_brain_mri":                 0.90,
            "no_thin_corpus_callosum":          0.90,
            "no_cognitive_impairment":          0.90,
            "no_retinal_involvement":           0.95,
            "cdca_therapy_initiated":           0.40,
            "oxysterol_reduction_on_cdca":      0.70,
            "bladder_urgency_mild":             0.25,
            "consanguinity":                    0.30,
            "liver_function_monitored":         0.60,
        },
        "critical_alerts": [
            "CYP7B1-PLASMA-OXYSTEROLS: Measure plasma 25-HC and 27-HC — diagnostic biomarker AND monitoring tool for CDCA therapy response",
            "CYP7B1-CDCA-THERAPY: Chenodeoxycholic acid (CDCA) 750 mg/day lowers oxysterol levels — only potential disease-modifying therapy in any HSP gene",
            "CYP7B1-LIVER-MONITORING: CYP7B1 also causes neonatal cholestatic liver disease — liver function tests annually in all biallelic SPG5A patients",
            "CYP7B1-PURE-HSP: No cognitive or retinal involvement — complicated phenotype (TCC, cognitive decline) → consider SPG11/SPG15 instead",
            "CYP7B1-OXYSTEROL-SPECIALIST: Oxysterol quantification requires specialist metabolic lab — not a standard lipid panel",
        ],
        "key_ddx_rules": [
            "Pure AR-HSP + elevated plasma 25-HC/27-HC → CYP7B1 (SPG5A) confirmed biomarker",
            "Pure AR-HSP + normal oxysterols + normal MRI → consider other AR-HSP (SPG11 unlikely without TCC)",
            "CDCA therapy initiated → repeat oxysterol levels 3 months later to confirm suppression",
        ],
    },

    # ── KIF1A — SPG30/KAND, kinesin anterograde transport ──
    {
        "gene": "KIF1A",
        "protein": "KIF1A Kinesin-3 Anterograde Axonal Transport Motor Protein",
        "alias": (
            "KIF1A; SPG30; OMIM gene 601255; SPG30 OMIM 607259 (AR); KAND OMIM 614255 (AD); 2q37.3; 1826 aa; ~207 kDa; "
            "AR biallelic LOF → SPG30 — adult-onset mild pure/complicated HSP; "
            "AD de novo variants → KIF1A-associated neurological disorder (KAND) — severe: intellectual disability + optic atrophy + cerebellar atrophy; "
            "Kinesin-3 motor drives anterograde transport of dense-core vesicles and synaptic vesicle precursors; "
            "AR vs AD have dramatically different prognoses — genotype class MANDATORY before counselling; "
            "Exome/genome sequencing required — de novo variants in KAND missed by family history alone"
        ),
        "aa": "1826 aa",
        "kDa": "~207 kDa",
        "locus": "2q37.3",
        "omim_gene": 601255,
        "omim_disease": 607259,
        "inheritance": "AR (SPG30, biallelic LOF, mild) / AD de novo (KAND, severe intellectual disability + optic atrophy)",
        "gene_class": (
            "KIF1A encodes kinesin-3, the primary axonal motor for anterograde transport of dense-core "
            "vesicles (containing BDNF, neuropeptides) and synaptic vesicle precursors. KIF1A also "
            "transports Rab3 effectors required for synaptic vesicle exocytosis. Biallelic loss-of-function "
            "(AR, SPG30): adult-onset mildly progressive spastic paraplegia; cognitive function typically "
            "preserved or mildly affected; phenotype relatively mild. De novo heterozygous variants — "
            "often missense in the motor domain or stalk region — act via dominant negative or gain-of-"
            "function mechanisms, causing KIF1A-associated neurological disorder (KAND): severe "
            "intellectual disability, optic atrophy, cerebellar hypoplasia/atrophy, and spastic paraplegia "
            "presenting in early childhood. The AR and AD phenotypes are not interchangeable — never "
            "counsel a KAND de novo family using AR-SPG30 data. Exome/trio sequencing for de novo "
            "detection is mandatory when no family history exists."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("KIF1A biallelic LOF (AR-SPG30) — frameshift/nonsense compound het", 0.25),
            ("KIF1A de novo motor-domain missense (KAND) — R18C/R18H hotspot", 0.20),
            ("KIF1A de novo stalk-domain missense (KAND) — dominant negative", 0.20),
            ("KIF1A homozygous splice (AR-SPG30) — consanguineous", 0.15),
            ("KIF1A de novo nonsense (KAND, severe early-onset)", 0.10),
            ("KIF1A compound het missense + frameshift (AR, variable)", 0.10),
        ],
        "age_onset_years_range": (1, 35),
        "sex_ratio_M": 0.48,
        "rates": {
            "spastic_paraplegia":               0.98,
            "ar_spg30_mild_adult_onset":        0.45,
            "kand_de_novo_severe":              0.45,
            "intellectual_disability_kand":     0.42,
            "optic_atrophy_kand":               0.38,
            "cerebellar_atrophy_kand":          0.35,
            "epilepsy_kand":                    0.25,
            "peripheral_neuropathy_ar":         0.30,
            "de_novo_confirmed":                0.45,
            "family_history_negative_de_novo":  0.45,
            "ambulatory_ar_spg30":              0.80,
            "wheelchair_kand_by_10":            0.25,
        },
        "critical_alerts": [
            "KIF1A-AR-vs-AD-DIFFERENT-PROGNOSIS: AR-SPG30 (biallelic LOF) = mild adult HSP; AD-KAND (de novo) = severe childhood disorder — NEVER counsel AR family using KAND data or vice versa",
            "KIF1A-DE-NOVO-EXOME: KAND presents without family history — de novo KIF1A variants require exome/trio sequencing — missed by gene panel without exome",
            "KIF1A-OPTIC-ATROPHY-KAND: Annual ophthalmology in KAND — optic atrophy progressive; VEP + OCT from diagnosis",
            "KIF1A-EPILEPSY-KAND: Epilepsy in 25% of KAND — EEG at diagnosis; anti-seizure medication if confirmed",
            "KIF1A-GENOTYPE-FIRST: Motor domain vs stalk vs LOF variants have different mechanisms — genotype mandatory before prediction of trajectory",
        ],
        "key_ddx_rules": [
            "Severe childhood spastic paraplegia + intellectual disability + optic atrophy + de novo → KIF1A KAND",
            "Adult-onset mild AR-HSP + family history + biallelic LOF → KIF1A SPG30 (milder spectrum)",
            "KAND: de novo variant in KIF1A motor domain → gain-of-function/dominant-negative — distinguish from AR haploinsufficiency",
        ],
    },

    # ── DDHD2 — SPG54, brain lipid droplets ──
    {
        "gene": "DDHD2",
        "protein": "DDHD Domain-Containing Phospholipase A1 Brain Lipid Metabolism",
        "alias": (
            "DDHD2; SPG54; OMIM gene 615003; SPG54 OMIM 615033; 8p11.23; 711 aa; ~79 kDa; "
            "AR complicated HSP with intellectual disability and thin corpus callosum; "
            "DDHD2 phospholipase A1 cleaves phosphatidic acid → brain lipid metabolism; "
            "Loss → brain lipid droplet accumulation in astrocytes and oligodendrocytes; "
            "Brain MR spectroscopy lipid peak at 1.3 ppm in basal ganglia/cerebral white matter PATHOGNOMONIC; "
            "This brain MRS lipid peak distinguishes SPG54 from other AR-HSP genes with TCC; "
            "Annual cognitive assessment mandatory — intellectual disability progressive"
        ),
        "aa": "711 aa",
        "kDa": "~79 kDa",
        "locus": "8p11.23",
        "omim_gene": 615003,
        "omim_disease": 615033,
        "inheritance": "AR — biallelic LOF; brain lipid droplets on MR spectroscopy",
        "gene_class": (
            "DDHD2 encodes DDHD domain-containing protein 2, a phospholipase A1 that hydrolyses "
            "phosphatidic acid (PA) and lysophosphatidic acid at the sn-1 position in brain cell "
            "membranes and organelles. Biallelic loss-of-function → PA accumulation → aberrant lipid "
            "droplet formation in brain astrocytes and oligodendrocytes. This brain lipid accumulation "
            "is uniquely detectable by MR spectroscopy as a prominent lipid peak at 1.3 ppm (CH₂ of "
            "fatty acid chains) in basal ganglia and cerebral white matter — a finding not reported in "
            "other HSP genes. SPG54 presents as complicated HSP with progressive intellectual disability, "
            "thin corpus callosum (similar to SPG11/SPG15), and childhood-onset spastic paraplegia. "
            "MR spectroscopy should be added to MRI protocol in all AR-HSP with TCC to detect this "
            "pathognomonic finding. Cognitive support and occupational therapy from diagnosis; disease-"
            "modifying therapy under investigation (lipid pathway modulation)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("DDHD2 biallelic frameshift — most common pan-ethnic", 0.30),
            ("DDHD2 homozygous nonsense — consanguineous Arab/Turkish families", 0.25),
            ("DDHD2 compound het missense + frameshift — DDHD domain disruption", 0.20),
            ("DDHD2 splice-site + frameshift — partial loss of PA hydrolysis", 0.15),
            ("DDHD2 large deletion (MLPA) + frameshift", 0.10),
        ],
        "age_onset_years_range": (1, 10),
        "sex_ratio_M": 0.50,
        "rates": {
            "brain_lipid_peak_mrs_pathognomonic": 0.92,
            "thin_corpus_callosum_mri":           0.85,
            "intellectual_disability_progressive": 0.88,
            "spastic_paraplegia_bilateral":        0.97,
            "childhood_onset_before_5":            0.80,
            "cerebellar_atrophy_mild":             0.30,
            "epilepsy":                            0.20,
            "dysarthria":                          0.45,
            "loss_ambulation_by_adulthood":        0.55,
            "bladder_neurogenic":                  0.40,
            "consanguinity":                       0.50,
            "cognitive_support_required":          0.88,
        },
        "critical_alerts": [
            "DDHD2-MRS-PATHOGNOMONIC: Brain MR spectroscopy lipid peak at 1.3 ppm (basal ganglia/white matter) PATHOGNOMONIC for SPG54 — request MRS in all AR-HSP with TCC",
            "DDHD2-MRS-DISTINGUISHES: SPG54 lipid peak distinguishes from SPG11/SPG15 (same TCC+ID phenotype, NO lipid peak) — MRS mandatory differential step",
            "DDHD2-COGNITIVE-SUPPORT: Intellectual disability progressive from childhood — early neuropsychological assessment, educational planning, and occupational therapy",
            "DDHD2-ANNUAL-COGNITION: Annual cognitive assessment + physiotherapy review; escalating wheelchair support as ambulation declines",
            "DDHD2-CONSANGUINITY: 50% consanguineous families — full extended family cascade testing (carrier detection for reproductive planning)",
        ],
        "key_ddx_rules": [
            "AR-HSP + TCC + intellectual disability + brain MRS lipid peak 1.3 ppm → DDHD2 (SPG54) — pathognomonic finding",
            "AR-HSP + TCC + intellectual disability + NO lipid peak on MRS → SPG11 or ZFYVE26 (Kjellin) first",
            "AR-HSP + consanguinity + childhood onset + TCC → request MRS before sequencing to narrow differential",
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
    for g in HSP_GENES:
        all_patients.extend(_make_patients(g))

    agg = {
        "total_patients":                    len(all_patients),
        "n_genes":                           len(HSP_GENES),
        "seeds":                             f"{SEED_BASE}–{SEED_BASE + len(HSP_GENES) - 1}",
        "progressive_spastic_paraplegia_pct": _agg(all_patients, "progressive_spastic_paraplegia"),
        "bilateral_leg_spasticity_pct":       _agg(all_patients, "bilateral_lower_limb_spasticity")
                                              or _agg(all_patients, "bilateral_leg_spasticity"),
        "thin_corpus_callosum_pct":           _agg(all_patients, "thin_corpus_callosum_mri"),
        "intellectual_disability_pct":        _agg(all_patients, "intellectual_disability")
                                              or _agg(all_patients, "intellectual_disability_progressive")
                                              or _agg(all_patients, "intellectual_disability_kand"),
        "macular_degeneration_pct":           _agg(all_patients, "progressive_macular_degeneration"),
        "elevated_oxysterols_pct":            _agg(all_patients, "elevated_plasma_25hc"),
        "brain_lipid_peak_mrs_pct":           _agg(all_patients, "brain_lipid_peak_mrs_pathognomonic"),
        "bladder_involvement_pct":            _agg(all_patients, "bladder_detrusor_hyperreflexia")
                                              or _agg(all_patients, "bladder_urgency"),
        "de_novo_kif1a_pct":                  _agg(all_patients, "de_novo_confirmed"),
        "physiotherapy_required_pct":         _agg(all_patients, "physiotherapy_required"),
    }

    top_alerts = [
        "SPAST-INCOMPLETE-PENETRANCE: ~40% of SPAST carriers asymptomatic at 60 — genetic counselling MANDATORY before presymptomatic testing",
        "SPG11-TCC-MANDATORY-MRI: Brain MRI T1 mandatory in ALL AR-HSP — thin corpus callosum virtually obligatory in SPG11; absence makes SPG11 unlikely",
        "CYP7B1-OXYSTEROLS-BIOMARKER: Plasma 25-HC + 27-HC measurable — diagnostic AND treatment monitor for CDCA therapy in SPG5A",
        "KIF1A-AR-vs-AD-DIFFERENT-PROGNOSIS: AR-SPG30 mild adult HSP vs AD-KAND severe childhood disorder — never conflate; genotype class MANDATORY",
        "DDHD2-MRS-PATHOGNOMONIC: Brain MR spectroscopy lipid peak 1.3 ppm in basal ganglia pathognomonic for SPG54 — request MRS in AR-HSP+TCC",
        "ZFYVE26-ANNUAL-FUNDUS: Annual fundoscopy + OCT mandatory in SPG15 — macular degeneration → blindness 3rd–4th decade",
        "SPG11-SPATACSIN-SPG15-COMPLEX: SPG11 and SPG15 form heteromeric complex — negative SPG11 → test ZFYVE26 for same ALR pathway",
        "REEP1-SPAST-ATL1-PATHWAY: REEP1, SPAST, ATL1 all ER morphogenesis — pure AD-HSP negative SPAST → test ATL1 + REEP1 systematically",
    ]

    return {
        "atlas": "Hereditary-Spastic-Paraplegia-Atlas",
        "title": "Complete 8-Gene Hereditary Spastic Paraplegia Atlas",
        "genes": [g["gene"] for g in HSP_GENES],
        "diseases": [g["omim_disease"] for g in HSP_GENES],
        "aggregate_stats": agg,
        "top_alerts": top_alerts,
        "registered": "2026-09-05",
    }


def get_breakdown():
    result = {}
    for g in HSP_GENES:
        patients = _make_patients(g)
        result[g["gene"]] = {
            "gene":          g["gene"],
            "protein":       g["protein"],
            "locus":         g["locus"],
            "aa":            g["aa"],
            "inheritance":   g["inheritance"],
            "omim_gene":     g["omim_gene"],
            "omim_disease":  g["omim_disease"],
            "n_patients":    g["n_patients"],
            "seed":          g["seed"],
            "alias":         g["alias"],
            "gene_class":    g["gene_class"],
            "critical_alerts": g["critical_alerts"],
            "key_ddx_rules":   g["key_ddx_rules"],
            "phenotype_rates": {k: round(v * 100, 1) for k, v in g["rates"].items()},
            "etiologies":    [{"label": l, "pct": round(w * 100, 1)} for l, w in g["etiologies"]],
            "aggregate":     {k: _agg(patients, k) for k in g["rates"]},
        }
    return result


def get_definitions():
    return {
        "atlas": "Hereditary-Spastic-Paraplegia-Atlas",
        "definitions": [
            {
                "term": "Hereditary Spastic Paraplegia (HSP)",
                "definition": (
                    "Clinically and genetically heterogeneous group of inherited neurological disorders "
                    "characterised by progressive spasticity and weakness of the lower limbs due to "
                    "length-dependent corticospinal tract degeneration. Classified as 'pure' (spastic "
                    "paraplegia ± bladder) or 'complicated' (additional features: intellectual disability, "
                    "cerebellar ataxia, peripheral neuropathy, optic atrophy, macular degeneration, thin "
                    "corpus callosum). >80 SPG loci identified. Inheritance: AD (SPAST/ATL1/REEP1 most "
                    "common), AR (SPG11/ZFYVE26/CYP7B1 most common), X-linked, de novo."
                )
            },
            {
                "term": "Thin Corpus Callosum (TCC) in AR-HSP",
                "definition": (
                    "Thinning of the corpus callosum, most prominent in the body and genu, visible on T1 "
                    "brain MRI. TCC is present in >90% of SPG11 (spatacsin) and ~88% of SPG15 (spastizin "
                    "/ZFYVE26) patients. TCC + complicated AR-HSP (intellectual disability + peripheral "
                    "neuropathy) is highly characteristic and should trigger SPG11 sequencing first, "
                    "followed by ZFYVE26 if negative. Brain MRI T1 is MANDATORY in all AR-HSP."
                )
            },
            {
                "term": "Incomplete Penetrance — SPAST (SPG4)",
                "definition": (
                    "Approximately 40% of obligate SPAST heterozygotes (individuals who must carry the "
                    "variant based on family structure) remain clinically asymptomatic at age 60. This "
                    "incomplete penetrance has major implications for presymptomatic genetic testing: a "
                    "test-positive result does not mean the person will develop disease, and a currently "
                    "asymptomatic carrier may never develop HSP. Genetic counsellors must communicate this "
                    "uncertainty explicitly before presymptomatic testing is offered."
                )
            },
            {
                "term": "SPAST MLPA — Copy-Number Variant Detection",
                "definition": (
                    "Multiplex ligation-dependent probe amplification (MLPA) detects large exonic deletions "
                    "and duplications in SPAST. Approximately 15% of pathogenic SPAST alleles are copy-number "
                    "variants (CNVs) that standard sequencing (Sanger or NGS) cannot reliably detect. MLPA "
                    "is mandatory in any AD-HSP family where SPAST sequencing is negative but SPAST remains "
                    "the most likely diagnosis clinically."
                )
            },
            {
                "term": "Spatacsin–Spastizin Complex (SPG11 + SPG15 Shared Pathway)",
                "definition": (
                    "Spatacsin (encoded by SPG11) and spastizin (encoded by ZFYVE26/SPG15) form a "
                    "heteromeric scaffold complex on lysosomal membranes. This complex is required for "
                    "autophagic lysosome reformation (ALR) — the recycling of lysosomal membrane tubules "
                    "back into functional lysosomes after autophagy. Loss of either partner → ALR failure → "
                    "lysosome and autophagosome cargo accumulation in long axons → neurodegeneration. "
                    "SPG11 and SPG15 share the same pathomechanism and overlapping phenotype (TCC, "
                    "intellectual disability, peripheral neuropathy) distinguished clinically by the "
                    "presence of macular degeneration in SPG15 (Kjellin syndrome). Testing one partner "
                    "negative should prompt testing the other."
                )
            },
            {
                "term": "Plasma Oxysterols — CYP7B1 (SPG5A) Biomarker",
                "definition": (
                    "In CYP7B1 (SPG5A) deficiency, plasma 25-hydroxycholesterol (25-HC) and "
                    "27-hydroxycholesterol (27-HC) accumulate due to deficient 7α-hydroxylation. "
                    "These oxysterols are measurable by LC-MS/MS in specialist metabolic laboratories "
                    "and serve as: (1) a diagnostic biomarker — elevated in virtually all biallelic "
                    "CYP7B1 loss-of-function patients; and (2) a treatment response monitor — plasma "
                    "oxysterols decline with chenodeoxycholic acid (CDCA) replacement therapy. "
                    "Oxysterol testing should be ordered alongside genetic sequencing in pure AR-HSP."
                )
            },
            {
                "term": "Chenodeoxycholic Acid (CDCA) Therapy — CYP7B1",
                "definition": (
                    "Chenodeoxycholic acid (CDCA) is a primary bile acid that, when administered exogenously, "
                    "suppresses the alternative oxysterol 25-hydroxycholesterol and 27-hydroxycholesterol "
                    "synthesis pathway via negative feedback on CYP27A1. In CYP7B1 deficiency (SPG5A), CDCA "
                    "supplementation (750 mg/day in adults) reduces plasma oxysterol levels by 40–70%, "
                    "potentially reducing neurotoxic oxysterol burden. CDCA is the only disease-mechanism-"
                    "targeted therapy available for any hereditary spastic paraplegia gene. Clinical trials "
                    "are ongoing; monitoring: plasma oxysterols every 3 months, liver function tests "
                    "(CDCA can cause hepatotoxicity at high doses)."
                )
            },
            {
                "term": "KIF1A-Associated Neurological Disorder (KAND)",
                "definition": (
                    "De novo heterozygous pathogenic variants in KIF1A (typically missense in the motor "
                    "domain or stalk region) cause KIF1A-Associated Neurological Disorder (KAND) — a "
                    "severe neurodevelopmental and neurodegenerative condition distinct from AR-SPG30. "
                    "KAND features: spastic paraplegia (childhood onset), intellectual disability (often "
                    "severe), optic atrophy, cerebellar atrophy, and epilepsy in 25%. The severity of KAND "
                    "is far greater than biallelic AR-SPG30 (mild adult-onset HSP). De novo variants are "
                    "identified by trio exome/genome sequencing — family history is negative. Prognosis, "
                    "counselling, and management must not be derived from AR-SPG30 literature."
                )
            },
            {
                "term": "Brain MR Spectroscopy Lipid Peak — DDHD2 (SPG54)",
                "definition": (
                    "In DDHD2 (SPG54) deficiency, failure to hydrolyse phosphatidic acid leads to "
                    "intracellular lipid droplet accumulation in brain astrocytes and oligodendrocytes. "
                    "These lipid droplets are detectable on proton MR spectroscopy as a prominent lipid "
                    "peak at 1.3 ppm (–CH₂– resonance of saturated fatty acid chains) in basal ganglia "
                    "and cerebral white matter. This MRS finding is pathognomonic for SPG54 and distinguishes "
                    "it from SPG11 and SPG15 (which also show TCC + intellectual disability but lack the "
                    "lipid peak). MR spectroscopy should be requested alongside standard T1/T2 brain MRI "
                    "in all AR-HSP with thin corpus callosum."
                )
            },
            {
                "term": "Autophagic Lysosome Reformation (ALR) — SPG11/SPG15 Mechanism",
                "definition": (
                    "Autophagic lysosome reformation (ALR) is a membrane recycling pathway that generates "
                    "new proto-lysosomes from the limiting membrane of autolysosomes by membrane tubulation, "
                    "scission, and re-acidification. The spatacsin-spastizin complex is required on the "
                    "autolysosomal membrane to initiate tubule formation. Loss of either spatacsin (SPG11) "
                    "or spastizin (SPG15/ZFYVE26) → ALR failure → autolysosome swelling → failure of cargo "
                    "degradation → accumulation of undegraded autophagic substrates in long axons → "
                    "neurodegeneration. This shared mechanism explains the clinically similar complicated "
                    "HSP phenotype of SPG11 and SPG15."
                )
            },
            {
                "term": "ER Tubule Morphogenesis — SPAST, ATL1, REEP1 Shared Pathway",
                "definition": (
                    "SPAST (spastin), ATL1 (atlastin-1), and REEP1 are core components of the tubular ER "
                    "network morphogenesis machinery in axons. REEP1 generates ER membrane curvature; "
                    "ATL1 fuses ER tubule tips to create the network; SPAST severs and remodels "
                    "microtubules at ER–microtubule contact sites. All three proteins interact physically, "
                    "and mutations in any one cause AD pure HSP by impairing distal axon ER morphology "
                    "in long corticospinal neurons. Pure AD-HSP negative for all three → rare AD-HSP "
                    "genes (ALDH18A1 SPG9, KIAA0196 SPG8, SPG42 SLC33A1)."
                )
            },
            {
                "term": "Spastic Paraplegia Rating Scale (SPRS)",
                "definition": (
                    "The Spastic Paraplegia Rating Scale (SPRS) is a 13-item validated clinical rating "
                    "scale for HSP severity. It measures walking ability, stair climbing, lower limb "
                    "spasticity, weakness, and sensory function. Total score 0–52 (0 = normal). Annual "
                    "SPRS documentation is mandatory in all HSP patients to objectively track progression, "
                    "inform physiotherapy targets, guide AFO/wheelchair planning, and stratify trial "
                    "eligibility. SPRS should complement, not replace, timed walking tests (10-metre "
                    "walk, 6-minute walk) and functional ADL assessment."
                )
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== BREAKDOWN (SPAST) ===")
    bd = get_breakdown()
    print(json.dumps(bd["SPAST"], indent=2))
    print("\n=== DEFINITIONS (first 2) ===")
    defs = get_definitions()
    print(json.dumps(defs["definitions"][:2], indent=2))
