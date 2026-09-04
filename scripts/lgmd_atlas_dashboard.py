#!/usr/bin/env python3
"""LGMD-Atlas — Complete 8-Gene Limb-Girdle Muscular Dystrophy Atlas
CAPN3   (Calpain-3; AR; 821 aa; 15q15.1; LGMD R1/2A; most common AR-LGMD worldwide; calpain protease) ·
DYSF    (Dysferlin; AR; 2080 aa; 2p13.2; LGMD R2/2B; membrane repair; VERY HIGH CK; Miyoshi allelic; STEROIDS WORSEN) ·
SGCA    (α-Sarcoglycan/Adhalin; AR; 387 aa; 17q21.33; LGMD R3/2D; sarcoglycan complex; mimic DMD) ·
SGCB    (β-Sarcoglycan; AR; 318 aa; 4q12; LGMD R4/2E; sarcoglycan complex; mimic DMD) ·
FKRP    (Fukutin-Related Protein; AR; 495 aa; 19q13.32; LGMD R9/2I; dystroglycanopathy; CARDIAC MANDATORY 60-80%) ·
ANO5    (Anoctamin-5; AR; 913 aa; 11p14.3; LGMD R12/2L; chloride channel/membrane repair; asymmetric calf; Miyoshi-3) ·
LMNA    (Lamin A/C; AD; 664 aa; 1q22; LGMD D1/1B; laminopathy; LETHAL ARRHYTHMIA/DCM; ICD MANDATORY) ·
HNRNPDL (hnRNP D-like; AD; 420 aa; 4q21.22; LGMD D3; RNA stress granule; Brazilian founder p.Asp378Asn)
320-patient aggregate cohort (8 × 40, seeds 1030–1037)

Limb-Girdle Muscular Dystrophy — Key Neurological Principles:
  - LGMD DEFINITION: Clinically and genetically heterogeneous group of inherited myopathies characterised
    by progressive weakness and wasting of proximal shoulder and pelvic girdle muscles. Dystrophin
    immunostaining NORMAL or near-normal (excluding dystrophinopathies). Currently ≥30 genetic subtypes.
    NEW NOMENCLATURE (ENMC 2017): LGMD Rx (AR, numbered R1, R2 ...) and LGMD Dx (AD, D1, D2 ...).
  - CLASSIFICATION:
    AR (recessive): CAPN3 R1, DYSF R2, SGCA R3, SGCB R4, FKRP R9, ANO5 R12 (most common worldwide).
    AD (dominant): LMNA D1, HNRNPDL D3 (rarer; often de novo or founder allele).
  - CRITICAL TREATMENT RULES:
    (1) DYSF-LGMD: Corticosteroids WORSEN outcome — reduce macrophage-mediated muscle membrane repair;
        avoid unless co-existing inflammatory myopathy confirmed by biopsy + molecular confirmation absent.
    (2) LMNA-LGMD D1: Cardiac management FIRST — lethal ventricular arrhythmias/complete AV block occur
        BEFORE limb weakness is severe; ICD implantation and antiarrhythmics mandatory; sudden cardiac death
        is the most common cause of premature mortality.
    (3) FKRP-LGMD R9: Annual cardiac screening (echo + 24 h Holter) MANDATORY — dilated cardiomyopathy
        (DCM) in 60-80% by 4th decade; respiratory surveillance mandatory.
    (4) Sarcoglycanopathies (SGCA, SGCB): mimic Duchenne/Becker (DMD/BMD) clinically, but dystrophin
        normal — distinguish by sarcoglycan immunohistochemistry (IHC) panel on biopsy; WES confirmatory.
  - DYSF HALLMARK: very high creatine kinase (CK typically 5,000–30,000 IU/L), often misdiagnosed as
    inflammatory myopathy (polymyositis). Inflammatory infiltrate on biopsy can mislead — molecular
    diagnosis mandatory before starting immunosuppression.
  - LMNA HALLMARK: cardiac involvement (AV block, AF, VT, DCM) often presenting BEFORE muscle weakness;
    rigid spine and prominent neck flexor weakness; EMERY-DREIFUSS phenotypic overlap possible.
  - CAPN3 HALLMARK: asymmetric early scapular winging; peroneal weakness + calf hypertrophy possible;
    CK 500-5,000; missense pArg788Trp most common Basque founder allele.
  - ANO5 HALLMARK: asymmetric calf muscle wasting; exercise-induced rhabdomyolysis; Miyoshi-3 (distal)
    and LGMD overlap; predominantly posterior lower-limb on MRI.

COHORT: 8 × 40 = 320 patient slots (seeds 1030–1037; gene-specific seeds)
"""

import random

SEED_BASE = 1030

LGMD_GENES = [
    # ── CAPN3 — Calpain-3 (most common AR-LGMD) ──────────────────────────────
    {
        "gene": "CAPN3", "protein": "Calpain-3 (p94)",
        "alias": "p94; LGMD R1/2A; OMIM #253600; AR; 15q15.1; most common AR-LGMD worldwide; calpain calcium-activated protease",
        "aa": "821 aa", "kDa": "94 kDa",
        "gene_class": (
            "CAPN3 encodes calpain-3 (p94), a skeletal-muscle-specific calcium-activated cysteine protease. "
            "Calpain-3 is tightly associated with titin (TTN) at the N2A and PGM/IS2 regions; "
            "this interaction is essential for calpain-3 stability and auto-proteolytic activation. "
            "MECHANISM OF DISEASE: biallelic LOF mutations → calpain-3 deficiency → impaired sarcomere "
            "remodelling, disrupted IκB kinase/NF-κB pathway (normally calpain-3 activates this cascade "
            "to suppress nuclear factor activity), failure of myonuclear apoptosis homeostasis, "
            "and defective calcium-mediated protein turnover at the thin/thick filament interface. "
            "MUTATION SPECTRUM: >300 pathogenic variants; missense most common (60%). "
            "pArg788Trp (R788W): Basque founder allele (common in northern Spain, southern France). "
            "CAPN3 mutations may cause mis-splicing — full-length cDNA analysis sometimes required. "
            "SECONDARY CAPN3 DEFICIENCY is common (many LGMD subtypes reduce calpain-3 on biopsy) — "
            "absent calpain-3 IHC does NOT equal CAPN3 primary mutation. WES/panel mandatory. "
            "CAPN3 15q15.1; OMIM gene 114240; disease OMIM 253600."
        ),
        "lgmd_group": "AR Myofibrillar/Sarcomeric — Protease (Calpain)",
        "lgmd_type": "LGMD R1/2A — Calpain-3 Protease Deficiency",
        "locus": "15q15.1", "omim_gene": 114240, "omim_disease": 253600,
        "inheritance": (
            "Autosomal Recessive (AR). Biallelic LOF. Compound heterozygous frequent. "
            "Basque, Reunion Island, and other founder populations have higher prevalence. "
            "Secondary calpain-3 deficiency on IHC is common in DYSF, SGCA, SGCB — do NOT diagnose CAPN3 from IHC alone."
        ),
        "phenotype": (
            "ONSET: typically 2–40 years (most 8–30 y). Pelvic girdle weakness first (hip girdle, glutei, "
            "hamstrings, then quadriceps). SCAPULAR WINGING early — important clinical sign; "
            "test with wall push-up. Calf pseudo-hypertrophy in ~30%. Toe walking if peronei involved. "
            "Facial muscles and cardiac muscle SPARED (key DDx from FSH and LMNA). "
            "Respiratory: mild reduction in FVC in advanced cases; rarely ventilator-dependent. "
            "CK: typically 500–5,000 IU/L (5–20× ULN). Highly variable clinical severity (even within families)."
        ),
        "disease": (
            "CAPN3-LGMD R1/2A — Calpainopathy. Most common AR-LGMD worldwide (30-35% of all AR-LGMD). "
            "No disease-modifying therapy approved (gene therapy trials in progress). "
            "Management: physiotherapy (eccentric exercise may worsen — avoid), AFOs for foot drop, "
            "scapular fixation surgery for severe winging (temporary benefit). "
            "Annual cardiac echo (cardiac involvement rare but not excluded — rule out). "
            "Genetic counselling: 25% recurrence risk (AR); carrier testing available."
        ),
        "treatment_options": [
            "No approved disease-modifying therapy (gene therapy trial NCT04240314 in progress)",
            "Physiotherapy: maintenance of strength, avoid aggressive eccentric exercise (may worsen in advanced)",
            "AFOs (ankle-foot orthoses): for foot drop and peroneal weakness",
            "Scapular fixation surgery: temporary benefit for significant winging affecting arm elevation",
            "Annual cardiac echo + ECG: rule out subclinical cardiomyopathy",
            "Annual pulmonary function tests: monitor FVC for respiratory decline in advanced disease",
            "Genetic counselling: AR 25% recurrence risk; cascade carrier testing recommended",
        ],
        "key_ddx": [
            "DYSF-LGMD R2/2B: CK much higher (>10,000), posterior lower leg wasting, distal weakness, IHC/WES",
            "FKRP-LGMD R9/2I: similar proximal pattern; α-dystroglycan IHC reduced; cardiac mandatory screen",
            "DMD/BMD: dystrophin reduced/absent on IHC; X-linked; CAPN3 has normal dystrophin",
            "Sarcoglycanopathies (SGCA, SGCB): sarcoglycan IHC panel distinguishes; CAPN3 biopsy can show secondary SGCA reduction",
            "Inflammatory myopathy (PM/DM): high CK, inflammatory infiltrate (CAPN3 biopsy can look inflamed) — molecular panel to exclude",
        ],
        "onset_range_y": (2, 40),
        "cardiac_risk": False,
        "respiratory_risk": False,
        "contractures": False,
        "facial_weakness": False,
        "very_high_ck": False,
        "scapular_winging": True,
        "calf_pseudohypertrophy": True,
        "asymmetric": False,
        "steroids_safe": True,
        "ad_inheritance": False,
    },

    # ── DYSF — Dysferlin (LGMD R2/2B + Miyoshi Myopathy) ─────────────────────
    {
        "gene": "DYSF", "protein": "Dysferlin",
        "alias": "Dysferlin; LGMD R2/2B; Miyoshi myopathy 1; OMIM #253601; AR; 2p13.2; STEROIDS WORSEN; very high CK; membrane repair",
        "aa": "2080 aa", "kDa": "237 kDa",
        "gene_class": (
            "DYSF encodes dysferlin (2080 aa), a large C2-domain-containing transmembrane protein essential "
            "for calcium-triggered sarcolemmal membrane repair. "
            "MECHANISM: dysferlin accumulates at sites of sarcolemmal injury → calcium influx → "
            "dysferlin vesicle fusion and membrane patching (lysosome exocytosis pathway). "
            "LOF: membrane holes cannot be resealed → repeated injury → inflammatory cascade → myopathy. "
            "ALLELIC CONDITIONS: same DYSF mutations cause: (a) LGMD R2/2B (proximal-predominant) and "
            "(b) Miyoshi myopathy 1 (distal-predominant, gastrocnemius/soleus wasted, toe-standing impossible). "
            "Many patients have mixed proximal+distal phenotype. SAME GENE — phenotype determined by unknown modifiers. "
            "INFLAMMATORY INFILTRATE on biopsy (macrophage-predominant) → commonly misdiagnosed as "
            "polymyositis; steroids given → worsens membrane repair → accelerated decline. "
            "DYSF 2p13.2; OMIM gene 603009; disease OMIM 253601 (LGMD), 254130 (Miyoshi)."
        ),
        "lgmd_group": "AR Membrane Repair — Dysferlin",
        "lgmd_type": "LGMD R2/2B — Dysferlinopathy (Membrane Repair Defect)",
        "locus": "2p13.2", "omim_gene": 603009, "omim_disease": 253601,
        "inheritance": (
            "Autosomal Recessive (AR). Biallelic LOF. Carrier frequency ~1:100 in some populations. "
            "Libyan-Jewish founder allele (pArg796∗). High allelic heterogeneity (>400 pathogenic variants)."
        ),
        "phenotype": (
            "ONSET: 15–35 years (rarely childhood). VERY HIGH CK: typically 5,000–30,000 IU/L (hallmark). "
            "Initial weakness: posterior lower legs (gastrocnemius/soleus) in Miyoshi phenotype → "
            "unable to stand on tiptoes; OR proximal pelvic girdle first (LGMD phenotype). "
            "Progressive to all limb muscles over 15–20 years. Facial and cardiac muscles SPARED. "
            "MRI: posterior lower leg (gastrocnemius, soleus, peroneus) predominant early; "
            "thigh biceps femoris, semimembranosus early selective involvement. "
            "IMPORTANT: inflammatory infiltrate on biopsy → misdiagnosis as PM → steroids given → WORSENS."
        ),
        "disease": (
            "DYSF-Dysferlinopathy (LGMD R2/2B + Miyoshi myopathy 1). "
            "CRITICAL: STEROIDS WORSEN — macrophage membrane repair pathway suppressed; avoid. "
            "Confirm molecular diagnosis BEFORE any immunosuppression. "
            "No approved therapy; exon-skipping trials ongoing (DYSF-exon 32 skipping). "
            "Dysferlin blood monocyte assay useful screening (monocytes express dysferlin); "
            "absent dysferlin on IHC (sarcolemmal) confirms deficiency."
        ),
        "treatment_options": [
            "STEROIDS: ABSOLUTELY AVOID — reduce macrophage membrane repair; worsen course; confirm molecular diagnosis first",
            "No approved disease-modifying therapy (exon-skipping/gene therapy trials in progress)",
            "Monocyte dysferlin assay: screening blood test (monocytes express dysferlin); absent = highly suggestive",
            "Physiotherapy: low-resistance, endurance-based; avoid eccentric overload and high-intensity exercise",
            "Cardiac echo + ECG: cardiac spared but confirm annually",
            "Annual pulmonary function tests: respiratory sparing common but monitor in advanced disease",
            "Genetic counselling: AR; 25% recurrence; Miyoshi vs LGMD phenotype in siblings unpredictable",
        ],
        "key_ddx": [
            "Polymyositis/Inflammatory myopathy: inflammatory infiltrate on biopsy — molecular panel MANDATORY before steroids",
            "CAPN3-LGMD R1/2A: lower CK (<5,000 usually), proximal dominant, no distal/posterior leg; IHC/WES",
            "Becker MD (BMD): dystrophin reduced; X-linked; often cardiac; DYSF has normal dystrophin",
            "FKRP-LGMD R9/2I: α-dystroglycan IHC reduced; cardiac mandatory; CK usually lower than DYSF",
            "Miyoshi myopathy alleles: same DYSF gene — Miyoshi phenotype (distal gastrocnemius) vs LGMD (proximal) from same gene",
        ],
        "onset_range_y": (15, 35),
        "cardiac_risk": False,
        "respiratory_risk": False,
        "contractures": False,
        "facial_weakness": False,
        "very_high_ck": True,
        "scapular_winging": False,
        "calf_pseudohypertrophy": False,
        "asymmetric": True,
        "steroids_safe": False,
        "ad_inheritance": False,
    },

    # ── SGCA — Alpha-Sarcoglycan (LGMD R3/2D) ─────────────────────────────────
    {
        "gene": "SGCA", "protein": "Alpha-Sarcoglycan (Adhalin)",
        "alias": "Adhalin; α-SG; LGMD R3/2D; OMIM #600119; AR; 17q21.33; sarcoglycan complex; dystrophin-associated glycoprotein",
        "aa": "387 aa", "kDa": "50 kDa",
        "gene_class": (
            "SGCA encodes alpha-sarcoglycan (α-SG, adhalin; 387 aa), a transmembrane glycoprotein component "
            "of the sarcoglycan subcomplex (α, β, γ, δ subunits). The sarcoglycan complex is part of the "
            "dystrophin-associated protein complex (DAPC), linking the intracellular actin cytoskeleton "
            "(via dystrophin) to the extracellular matrix (via dystroglycan). "
            "MECHANISM: biallelic SGCA LOF → α-SG absent → destabilises entire sarcoglycan complex "
            "(secondary reduction of β, γ, δ sarcoglycans on IHC) → DAPC disruption → "
            "membrane fragility → myofibre necrosis → progressive myopathy. "
            "CLINICAL MIMIC: very similar phenotype to DMD/BMD → dystrophin IHC NORMAL/NEAR-NORMAL "
            "is critical distinguisher; sarcoglycan IHC panel (α, β, γ, δ) shows primary α-SG loss. "
            "pArg77Cys (R77C): most common SGCA allele in North Africa and European populations. "
            "SGCA 17q21.33; OMIM gene 600119; disease OMIM 600119."
        ),
        "lgmd_group": "AR Sarcoglycan Complex — Alpha-Sarcoglycan",
        "lgmd_type": "LGMD R3/2D — Alpha-Sarcoglycanopathy",
        "locus": "17q21.33", "omim_gene": 600119, "omim_disease": 600119,
        "inheritance": "Autosomal Recessive (AR). Biallelic LOF. Consanguinity increases prevalence in North Africa, Turkey.",
        "phenotype": (
            "ONSET: 3–15 years (may be as early as 2 y). DUCHENNE-LIKE PHENOTYPE: "
            "Gowers sign, waddling gait, calf pseudo-hypertrophy, difficulty climbing stairs. "
            "Proximal pelvic girdle → shoulder girdle progression. CK: 1,000–10,000 IU/L. "
            "Cardiac: cardiomyopathy in ~30% (less than β-SG or FKRP). ECG ± echo monitoring. "
            "Facial weakness ABSENT (key DDx from FSHD). "
            "Scoliosis in wheelchair-bound patients. Variable severity — severity correlates "
            "poorly with mutation class (missense vs nonsense)."
        ),
        "disease": (
            "SGCA-LGMD R3/2D — Alpha-Sarcoglycanopathy. "
            "KEY: Dystrophin IHC NORMAL → must test sarcoglycan IHC panel to identify α-SG absence. "
            "No approved disease-modifying therapy. Cardiac echo annually. "
            "Gene therapy (AAVrh74.MHCK7.SGCA — Solgenia/Sarepta) in Phase 2/3 trials — most advanced "
            "gene therapy programme in sarcoglycanopathies (2025). Corticosteroids sometimes used "
            "(limited evidence, extrapolated from DMD protocols) — not standard of care."
        ),
        "treatment_options": [
            "Sarcoglycan IHC panel (α, β, γ, δ): essential biopsy test — primary α loss, secondary β/γ/δ loss",
            "Cardiac echo + 24h Holter: annually from diagnosis (cardiomyopathy risk ~30%)",
            "Corticosteroids: limited evidence in LGMD (off-label, extrapolated from DMD); individual decision",
            "Gene therapy trial: AAVrh74.MHCK7.SGCA (Phase 2/3 — most advanced LGMD gene therapy 2025)",
            "Physiotherapy: ROM maintenance, respiratory exercises, hydrotherapy",
            "Scoliosis monitoring: Cobb angle; surgical threshold Cobb >40°",
            "Genetic counselling: AR 25% recurrence; dystrophin NORMAL distinguishes from BMD",
        ],
        "key_ddx": [
            "DMD/BMD: dystrophin absent/reduced on IHC; X-linked — SGCA is AR with normal dystrophin",
            "SGCB-LGMD R4/2E: β-SG absent primarily; clinical phenotype similar; IHC + WES distinguish",
            "CAPN3-LGMD R1/2A: secondary α-SG reduction possible; calpain-3 primary on WES",
            "FKRP-LGMD R9/2I: α-dystroglycan reduced; sarcoglycan usually normal; cardiac more common",
            "FSHD: facial weakness prominent; no scapular winging without facial; SMCHD1/D4Z4 test",
        ],
        "onset_range_y": (3, 15),
        "cardiac_risk": True,
        "respiratory_risk": True,
        "contractures": True,
        "facial_weakness": False,
        "very_high_ck": False,
        "scapular_winging": False,
        "calf_pseudohypertrophy": True,
        "asymmetric": False,
        "steroids_safe": True,
        "ad_inheritance": False,
    },

    # ── SGCB — Beta-Sarcoglycan (LGMD R4/2E) ──────────────────────────────────
    {
        "gene": "SGCB", "protein": "Beta-Sarcoglycan",
        "alias": "β-SG; LGMD R4/2E; OMIM #604286; AR; 4q12; sarcoglycan complex; DMD mimic; cardiomyopathy more common than SGCA",
        "aa": "318 aa", "kDa": "43 kDa",
        "gene_class": (
            "SGCB encodes beta-sarcoglycan (β-SG; 318 aa), a type II transmembrane glycoprotein and "
            "component of the sarcoglycan tetramer (α, β, γ, δ). "
            "β-SG serves as a scaffolding hub — direct binding partner of α, γ, and δ sarcoglycans. "
            "MECHANISM: biallelic SGCB LOF → β-SG absent → entire sarcoglycan complex destabilised "
            "→ secondary loss of α, γ, δ on IHC → DAPC disruption → sarcolemmal fragility → myonecrosis. "
            "SGCB PRIMARY LOSS vs SECONDARY: distinguish with sequential IHC — β-SG lost first, "
            "then secondary loss of others; in SGCA primary loss, β-SG may be relatively preserved. "
            "Cardiomyopathy risk in SGCB is HIGHER than SGCA (~50-60% vs ~30%). "
            "pLeu31Pro: founder allele in Brazilian population. "
            "SGCB 4q12; OMIM gene 600900; disease OMIM 604286."
        ),
        "lgmd_group": "AR Sarcoglycan Complex — Beta-Sarcoglycan",
        "lgmd_type": "LGMD R4/2E — Beta-Sarcoglycanopathy",
        "locus": "4q12", "omim_gene": 600900, "omim_disease": 604286,
        "inheritance": (
            "Autosomal Recessive (AR). Biallelic LOF. Brazilian population: pLeu31Pro founder allele. "
            "Consanguineous families: North Africa, Middle East."
        ),
        "phenotype": (
            "ONSET: 3–15 years; often indistinguishable from SGCA clinically. "
            "DUCHENNE-LIKE: proximal muscle weakness, calf hypertrophy, Gowers sign, waddling gait. "
            "CK: 1,000–15,000 IU/L. "
            "CARDIAC: 50-60% DCM or cardiomyopathy — MORE than SGCA; ECG changes early. "
            "CRITICAL: annual cardiac echo + Holter mandatory (may develop DCM before limb weakness is severe). "
            "Respiratory compromise: >30% by 3rd decade; spirometry annually. "
            "Scoliosis in non-ambulant. Facial muscles SPARED."
        ),
        "disease": (
            "SGCB-LGMD R4/2E — Beta-Sarcoglycanopathy. "
            "KEY: Cardiomyopathy risk 50-60% — higher than SGCA; cardiac monitoring MANDATORY. "
            "β-SG IHC primary absent (distinguishes from SGCA primary absent α-SG). "
            "Gene therapy trials: investigational (same vector technology as SGCA). "
            "Dystrophin IHC NORMAL — rule out BMD."
        ),
        "treatment_options": [
            "Cardiac echo + 24h Holter: annually from diagnosis — MANDATORY (DCM risk 50-60%)",
            "Annual spirometry + peak cough flow: respiratory muscle decline monitoring",
            "Sarcoglycan IHC panel: β-SG primary absent, α/γ/δ secondary reduced on biopsy",
            "Corticosteroids: limited evidence; individual decision (as for SGCA)",
            "Gene therapy trials: investigational (no approved therapy 2025)",
            "Physiotherapy, AFOs, scapular fixation: as for other LGMD",
            "ACE inhibitor/beta-blocker: if DCM confirmed (standard heart failure management)",
        ],
        "key_ddx": [
            "SGCA-LGMD R3/2D: α-SG primary absent vs β-SG primary absent — IHC + WES distinguish; similar clinical",
            "DMD/BMD: dystrophin absent/reduced; X-linked; dystrophin IHC distinguishes",
            "FKRP-LGMD R9/2I: dystroglycan IHC reduced; cardiac common; sarcoglycan usually preserved",
            "CAPN3-LGMD R1/2A: calpain-3 IHC absent (primary or secondary); CK typically lower",
            "Inflammatory myopathy: biopsy can show inflammation — molecular confirmation essential",
        ],
        "onset_range_y": (3, 15),
        "cardiac_risk": True,
        "respiratory_risk": True,
        "contractures": True,
        "facial_weakness": False,
        "very_high_ck": False,
        "scapular_winging": False,
        "calf_pseudohypertrophy": True,
        "asymmetric": False,
        "steroids_safe": True,
        "ad_inheritance": False,
    },

    # ── FKRP — Fukutin-Related Protein (LGMD R9/2I) ────────────────────────────
    {
        "gene": "FKRP", "protein": "Fukutin-Related Protein",
        "alias": "FKRP; LGMD R9/2I; OMIM #607155; AR; 19q13.32; dystroglycanopathy; CARDIAC MANDATORY; α-dystroglycan IHC reduced",
        "aa": "495 aa", "kDa": "55 kDa",
        "gene_class": (
            "FKRP encodes fukutin-related protein (495 aa), a ribitol-5-phosphate transferase in the "
            "Golgi apparatus involved in O-mannosyl glycosylation of α-dystroglycan (α-DG). "
            "MECHANISM: FKRP transfers CDP-ribitol to α-DG in a pathway: POMGNT2→B3GALNT2→POMK→LARGE1→FKTN→FKRP; "
            "biallelic LOF → hypoglycosylated α-DG → cannot bind extracellular matrix ligands "
            "(laminin, agrin, perlecan, neurexin) → sarcolemmal instability → myonecrosis. "
            "ALLELIC CONDITIONS: same FKRP mutations cause a wide spectrum: "
            "Mild end: LGMD R9/2I (pLeu276Ile in 90%+ of Northern European alleles). "
            "Severe end: Walker-Warburg syndrome / muscle-eye-brain disease (WWS/MEB — structural brain + eye). "
            "pLeu276Ile homozygous → almost always LGMD R9 (mild dystroglycanopathy). "
            "FKRP 19q13.32; OMIM gene 606596; disease OMIM 607155 (LGMD R9)."
        ),
        "lgmd_group": "AR Dystroglycanopathy — O-Mannosyl Glycosylation (FKRP)",
        "lgmd_type": "LGMD R9/2I — FKRP Dystroglycanopathy",
        "locus": "19q13.32", "omim_gene": 606596, "omim_disease": 607155,
        "inheritance": (
            "Autosomal Recessive (AR). Biallelic LOF. "
            "pLeu276Ile: most common allele in Northern Europe (UK, Scandinavia) — 90% of alleles in LGMD R9 cohorts. "
            "Most severe alleles (WWS-spectrum) typically compound heterozygous with severe truncating allele."
        ),
        "phenotype": (
            "ONSET: childhood to adulthood (typically 1–30 y in LGMD R9/pLeu276Ile). "
            "Proximal weakness: pelvic girdle > shoulder girdle. Calf hypertrophy common (70%). "
            "CK: 500–10,000 IU/L (very wide range). "
            "CARDIAC: 60-80% develop DCM by 4th–5th decade — ANNUAL ECHO + HOLTER MANDATORY. "
            "RESPIRATORY: significant respiratory decline; FVC monitoring ± NIV mandatory. "
            "Microcephaly, intellectual disability, structural brain/eye abnormalities: "
            "ABSENT in pLeu276Ile LGMD R9; PRESENT in WWS/MEB severe alleles. "
            "α-DG IHC: reduced/absent (use monoclonal antibody IIH6 or VIA4-1) — "
            "reduced α-DG suggests dystroglycanopathy but NOT specific to FKRP (test whole panel)."
        ),
        "disease": (
            "FKRP-LGMD R9/2I — Fukutin-Related Protein Dystroglycanopathy. "
            "CARDIAC SCREENING MANDATORY annually: DCM 60-80% prevalence; "
            "cardiac complications can be presenting feature or cause premature death. "
            "Respiratory surveillance essential. Gene therapy (AAV-FKRP) in preclinical trials. "
            "Riboflavin supplementation not established. Ribitol precursor trials exploratory."
        ),
        "treatment_options": [
            "Annual cardiac echo + 24h Holter: MANDATORY — DCM in 60-80%; early ACEi/beta-blocker if DCM",
            "Annual spirometry + sleep study: respiratory compromise common; NIV if FVC <50%",
            "α-Dystroglycan IHC (IIH6 antibody): primary screening biopsy marker for dystroglycanopathy",
            "No approved disease-modifying therapy (gene therapy trials in progress 2025)",
            "Physiotherapy, hydrotherapy, AFOs: standard LGMD management",
            "Corticosteroids: minimal evidence; cardiac co-morbidity complicates use",
            "Scoliosis monitoring: Cobb >40° → surgical consideration",
        ],
        "key_ddx": [
            "Other dystroglycanopathies (POMGNT1, FKTN, LARGE1, B3GALNT2, POMK): same α-DG IHC; gene panel required",
            "CAPN3-LGMD R1/2A: similar proximal weakness; α-DG IHC normal in CAPN3; calpain-3 IHC absent",
            "SGCA/SGCB sarcoglycanopathies: sarcoglycan IHC lost; α-DG preserved; gene panel distinguishes",
            "Walker-Warburg syndrome / MEB (severe FKRP alleles): brain malformations, eye anomalies — ABSENT in pLeu276Ile",
            "Becker MD: dystrophin reduced; dystrophin IHC key; FKRP has normal dystrophin",
        ],
        "onset_range_y": (1, 30),
        "cardiac_risk": True,
        "respiratory_risk": True,
        "contractures": False,
        "facial_weakness": False,
        "very_high_ck": False,
        "scapular_winging": False,
        "calf_pseudohypertrophy": True,
        "asymmetric": False,
        "steroids_safe": True,
        "ad_inheritance": False,
    },

    # ── ANO5 — Anoctamin-5 (LGMD R12/2L) ──────────────────────────────────────
    {
        "gene": "ANO5", "protein": "Anoctamin-5 (Calcium-Activated Chloride Channel)",
        "alias": "ANO5; LGMD R12/2L; Miyoshi myopathy 3; OMIM #611307; AR; 11p14.3; asymmetric calf wasting; membrane repair",
        "aa": "913 aa", "kDa": "104 kDa",
        "gene_class": (
            "ANO5 encodes anoctamin-5 (913 aa), a member of the anoctamin (TMEM16) family of calcium-activated "
            "chloride channels. Despite its ion channel annotation, ANO5 in skeletal muscle appears to function "
            "primarily in membrane repair (like DYSF) — the precise biochemical role remains investigational. "
            "MECHANISM: biallelic ANO5 LOF → impaired calcium-triggered membrane repair at sarcolemmal injury "
            "sites → repeated micro-injuries → myofibre degeneration. "
            "ALLELIC CONDITIONS: ANO5 mutations cause both: "
            "(a) LGMD R12/2L — proximal pelvic girdle weakness; "
            "(b) Miyoshi myopathy 3 (MM3) — distal calf predominant weakness. "
            "CHARACTERISTIC FINDING: asymmetric and posterior lower-limb involvement on muscle MRI; "
            "gastrocnemius medial head and soleus commonly affected. "
            "pTrp236Arg (W236R): founder allele in Northern Europe (common in UK, Scandinavian patients). "
            "ANO5 11p14.3; OMIM gene 608662; disease OMIM 611307."
        ),
        "lgmd_group": "AR Membrane Repair — Anoctamin-5",
        "lgmd_type": "LGMD R12/2L — Anoctamin-5 Myopathy",
        "locus": "11p14.3", "omim_gene": 608662, "omim_disease": 611307,
        "inheritance": (
            "Autosomal Recessive (AR). Biallelic LOF. "
            "pTrp236Arg: Northern European founder allele. "
            "High allelic heterogeneity otherwise."
        ),
        "phenotype": (
            "ONSET: adolescence to 4th decade (often 15–40 y). ASYMMETRIC CALF WASTING — "
            "characteristic and often the first finding; patients notice one calf smaller than the other. "
            "Posterior lower limb weakness (gastrocnemius/soleus) predominant; "
            "proximal pelvic weakness follows. Upper limb/shoulder involvement later/mild. "
            "EXERCISE-INDUCED RHABDOMYOLYSIS: CK spikes >10,000 IU/L after vigorous exercise; "
            "myalgias and dark urine episodes reported. Resting CK: 500–5,000 IU/L. "
            "Facial, cardiac, and respiratory involvement: SPARED in most patients. "
            "Slow progression compared to other AR-LGMDs. "
            "Males more severely affected than females (unclear reason — modifier genes)."
        ),
        "disease": (
            "ANO5-LGMD R12/2L / Miyoshi myopathy 3. "
            "ASYMMETRIC CALF WASTING is the phenotypic hallmark — examine both calves carefully. "
            "Warn against high-intensity exercise (rhabdomyolysis risk). "
            "No disease-modifying therapy. Cardiac and respiratory generally spared. "
            "Muscle MRI: posterior lower leg + thigh biceps femoris early; "
            "highly characteristic pattern aids diagnosis."
        ),
        "treatment_options": [
            "No approved disease-modifying therapy",
            "Exercise: graded low-intensity; AVOID vigorous exercise → rhabdomyolysis risk (CK spikes >10,000)",
            "Rhabdomyolysis management: IV fluids if CK very high; monitor renal function",
            "Cardiac echo + ECG: cardiac spared but confirm baseline and monitor",
            "Physiotherapy: lower limb focus; AFOs for ankle dorsiflexion weakness",
            "Annual CK monitoring: baseline; spikes after exercise are expected in ANO5",
            "Genetic counselling: AR; 25% recurrence; phenotypic variability within families",
        ],
        "key_ddx": [
            "DYSF-LGMD R2/2B: very high CK, posterior lower leg, inflammatory biopsy; dysferlin IHC absent",
            "Miyoshi myopathy 1 (DYSF): same clinical overlap — ANO5 vs DYSF; dysferlin IHC/WES distinguish",
            "CAPN3-LGMD R1/2A: more symmetric proximal; less lower-leg focus; calpain-3 IHC absent",
            "Sporadic exercise-induced myopathy: ANTI-HMGCR/SRP myopathy; CK pattern + antibody screen",
            "LGMD from HNRNPDL D3: AD; RNA inclusions on biopsy; HNRNPDL panel distinguishes",
        ],
        "onset_range_y": (15, 40),
        "cardiac_risk": False,
        "respiratory_risk": False,
        "contractures": False,
        "facial_weakness": False,
        "very_high_ck": True,
        "scapular_winging": False,
        "calf_pseudohypertrophy": False,
        "asymmetric": True,
        "steroids_safe": True,
        "ad_inheritance": False,
    },

    # ── LMNA — Lamin A/C (LGMD D1/1B) ─────────────────────────────────────────
    {
        "gene": "LMNA", "protein": "Lamin A/C",
        "alias": "Lamin A/C; LGMD D1/1B; Emery-Dreifuss MD; OMIM #159001; AD; 1q22; LETHAL ARRHYTHMIA/DCM; ICD MANDATORY",
        "aa": "664 aa", "kDa": "74/65 kDa",
        "gene_class": (
            "LMNA encodes lamin A (664 aa) and lamin C (572 aa) via alternative splicing — both are Type V "
            "intermediate filaments forming the nuclear lamina at the inner nuclear membrane. "
            "ALLELIC CONDITIONS (>400 LMNA mutations → >15 phenotypes): "
            "LGMD D1/1B, Emery-Dreifuss MD (EDMD2), dilated cardiomyopathy-1A (DCM1A, CMD1A), "
            "Charcot-Marie-Tooth 2B1 (CMT2B1), familial partial lipodystrophy 2 (FPLD2/Dunnigan), "
            "Hutchinson-Gilford progeria, restrictive dermopathy. "
            "MECHANISM: LOF → nuclear lamina instability → impaired mechanotransduction → "
            "myonuclear fragility → transcription factor dysregulation (LMNA binds and sequesters "
            "transcription factors including emerin, LINC complex, SUN proteins). "
            "CARDIAC IS THE LEADING CAUSE OF PREMATURE DEATH — ventricular arrhythmia, "
            "complete AV block, DCM → sudden cardiac death BEFORE muscle weakness becomes severe. "
            "LMNA 1q22; OMIM gene 150330; disease OMIM 159001."
        ),
        "lgmd_group": "AD Nuclear Envelope — Laminopathy (LGMD D1/1B)",
        "lgmd_type": "LGMD D1/1B — LMNA Laminopathy (Autosomal Dominant)",
        "locus": "1q22", "omim_gene": 150330, "omim_disease": 159001,
        "inheritance": (
            "Autosomal Dominant (AD). Heterozygous LOF/dominant-negative. De novo mutations common. "
            "Some missense alleles (pArg249Gln, pGlu161Lys) highly penetrant for arrhythmia. "
            "25% LGMD phenotype; EDMD2 (humeroperoneal + contractures) and cardiomyopathy phenotypes overlap."
        ),
        "phenotype": (
            "ONSET: childhood to adulthood (variable). "
            "MUSCLE: LGMD D1 pattern — proximal pelvic girdle predominant; RIGID SPINE early; "
            "NECK FLEXOR WEAKNESS prominent (chin-on-chest posture); "
            "Humeroperoneal pattern in EDMD overlap (biceps, triceps, peroneal muscles). "
            "CARDIAC: LEADS THE PROGNOSIS. DCM + arrhythmias (AV block, AF, VT, VF) often BEFORE "
            "limb weakness is severe. Sudden cardiac death in 15-40% without ICD. "
            "CONTRACTURES: elbow, Achilles tendon, and cervical spine contractures — "
            "PATHOGNOMONIC for EDMD2 overlap; may be present in LGMD D1. "
            "CK: often mildly elevated 200–1,500 IU/L (lower than other LGMDs)."
        ),
        "disease": (
            "LMNA-LGMD D1/1B — Laminopathy. "
            "CARDIAC IS PRIORITY OVER MUSCLE: ICD implantation mandatory when AV block/VT present; "
            "consider prophylactic ICD early in high-risk genotypes (pArg249Gln). "
            "Annual cardiac MRI/echo + 24h Holter MANDATORY from diagnosis. "
            "SUDDEN CARDIAC DEATH risk 15-40% without ICD. "
            "Antiarrhythmics: carvedilol, bisoprolol for DCM; amiodarone for VT bridge to ICD. "
            "Muscle management secondary to cardiac management."
        ),
        "treatment_options": [
            "CARDIAC FIRST: ICD implantation when AV block grade 2-3 or sustained VT — prevents sudden death",
            "Annual cardiac MRI + 24h Holter: MANDATORY from diagnosis; LMNA = highest SCD risk of all LGMD",
            "ACE inhibitor + beta-blocker: if DCM confirmed (LVEF <45%) — standard HF therapy",
            "Amiodarone: bridge to ICD for ventricular arrhythmia management",
            "LMNA-specific SCD risk score (2019): genotype + LNHB + atrial fibrillation + non-sustained VT + LVEF",
            "Contracture management: physiotherapy, splinting; elbow/Achilles; avoid forced passive stretching",
            "Genetic counselling: AD; 50% offspring risk; pre-symptomatic cardiac screening in first-degree relatives",
        ],
        "key_ddx": [
            "EDMD1 (X-linked EMD/Emerin): X-linked; emerin IHC absent on myonuclear envelope; same clinical triad",
            "EDMD3 (FHL1): X-linked; FHL1 IHC; contractures + cardiomyopathy; similar phenotype",
            "CAPN3-LGMD R1/2A: no cardiac arrhythmia; no contractures; AR; high CK; calpain-3 absent",
            "DCM-1A (LMNA isolated cardiomyopathy): same gene, cardiac dominant, minimal myopathy",
            "Rigid-spine muscular dystrophy (SEPN1/SELENON): rigid spine + respiratory; no cardiac arrhythmia",
        ],
        "onset_range_y": (5, 40),
        "cardiac_risk": True,
        "respiratory_risk": True,
        "contractures": True,
        "facial_weakness": False,
        "very_high_ck": False,
        "scapular_winging": False,
        "calf_pseudohypertrophy": False,
        "asymmetric": False,
        "steroids_safe": True,
        "ad_inheritance": True,
    },

    # ── HNRNPDL — hnRNP D-like (LGMD D3) ──────────────────────────────────────
    {
        "gene": "HNRNPDL", "protein": "Heterogeneous Nuclear Ribonucleoprotein D-Like (hnRNP D-like)",
        "alias": "hnRNP D-like; JKTBP; LGMD D3; OMIM #615366; AD; 4q21.22; Brazilian founder p.Asp378Asn; RNA stress granule; low-complexity domain",
        "aa": "420 aa", "kDa": "46 kDa",
        "gene_class": (
            "HNRNPDL encodes heterogeneous nuclear ribonucleoprotein D-like (hnRNP D-like / JKTBP; 420 aa), "
            "an RNA-binding protein with two RNA recognition motifs (RRMs) and a C-terminal low-complexity "
            "glycine-rich domain (GRD) — structurally similar to FUS, TDP-43, and hnRNPA2B1, "
            "which also cause myopathy via aggregation of low-complexity domains. "
            "MECHANISM: pathogenic AD mutations cluster in the GRD region → promote protein aggregation "
            "into cytoplasmic stress granules/rimmed vacuoles in muscle fibres → RNA processing dysfunction "
            "→ myopathy with rimmed vacuoles on biopsy (Engel-type). "
            "RELATIONSHIP: HNRNPDL joins a class of AD myopathies caused by low-complexity domain aggregation: "
            "FUS-myopathy, TDP-43 inclusions, hnRNPA1/A2 myopathy, hnRNPDL. "
            "pAsp378Asn (D378N): founder allele in Brazilian families (most described cases). "
            "pAsp378His (D378H): in Chinese families. Both in GRD. "
            "HNRNPDL 4q21.22; OMIM gene 607137; disease OMIM 615366."
        ),
        "lgmd_group": "AD RNA-Binding Protein — hnRNP D-like (Low-Complexity Domain)",
        "lgmd_type": "LGMD D3 — HNRNPDL Myopathy (AD, Brazilian Founder)",
        "locus": "4q21.22", "omim_gene": 607137, "omim_disease": 615366,
        "inheritance": (
            "Autosomal Dominant (AD). Heterozygous dominant-negative or gain-of-function aggregation. "
            "pAsp378Asn: Brazilian founder allele; highly penetrant. "
            "pAsp378His: Chinese families. "
            "Most reported cases familial; de novo cases described."
        ),
        "phenotype": (
            "ONSET: adulthood (3rd–6th decade); highly variable age-of-onset even within families. "
            "MUSCLE: proximal limb-girdle weakness; slow progression. "
            "BIOPSY HALLMARK: RIMMED VACUOLES (autophagic vacuoles with basophilic granules on H&E; "
            "highlighted by modified Gomori trichrome — characteristic of protein aggregate myopathies). "
            "HNRNPDL protein inclusions on IHC. "
            "CK: mildly elevated 200–1,500 IU/L; "
            "EMG: myopathic ± irritative features (fibrillation potentials in vacuolar myopathy). "
            "Cardiac and respiratory involvement: MILD to ABSENT in most reported cases. "
            "RARE disease: limited to Brazilian and Chinese cohorts primarily; "
            "fewer than 50 families described globally (2025)."
        ),
        "disease": (
            "HNRNPDL-LGMD D3 — hnRNP D-like Myopathy. "
            "Rimmed vacuoles on biopsy + LGMD phenotype + AD inheritance + Brazilian/Chinese ancestry → "
            "test HNRNPDL. Rarest of the 8 LGMD subtypes in this atlas. "
            "No approved disease-modifying therapy. Management: physiotherapy. "
            "Understand mechanistically as an RNA-binding protein aggregopathy (FUS/TDP-43 analogy). "
            "Future therapeutic direction: stress granule dissolution, phase-separation inhibitors."
        ),
        "treatment_options": [
            "No approved disease-modifying therapy",
            "Physiotherapy: ROM + strength maintenance; generally slow progression",
            "Cardiac echo + ECG: cardiac mild in most but confirm at baseline",
            "Annual pulmonary function: respiratory mild; monitor",
            "Biopsy: rimmed vacuoles on modified Gomori trichrome; HNRNPDL IHC inclusions confirming",
            "Genetic counselling: AD; 50% offspring risk; anticipation not documented",
            "WES/gene panel: HNRNPDL GRD-domain mutations in Brazilian/Chinese ancestry + rimmed vacuoles",
        ],
        "key_ddx": [
            "Other rimmed vacuole myopathies: GNE myopathy (sporadic IBM-like, AR), VCP myopathy, hnRNPA1/A2 myopathy",
            "Inclusion body myositis (IBM): sporadic; older onset; finger flexor+quadriceps pattern; CD8 T-cells",
            "FUS-myopathy: similar RNA-aggregation; FUS mutations; IHC FUS inclusions",
            "CAPN3-LGMD R1/2A: no rimmed vacuoles; calpain-3 IHC absent; AR not AD",
            "LMNA-LGMD D1: AD; cardiac arrhythmia; no rimmed vacuoles; LMNA distinct on WES",
        ],
        "onset_range_y": (20, 55),
        "cardiac_risk": False,
        "respiratory_risk": False,
        "contractures": False,
        "facial_weakness": False,
        "very_high_ck": False,
        "scapular_winging": False,
        "calf_pseudohypertrophy": False,
        "asymmetric": False,
        "steroids_safe": True,
        "ad_inheritance": True,
    },
]


def _gen_patients(gene_data: dict, seed: int) -> list:
    """Generate 40 synthetic LGMD patients for a single gene."""
    rng = random.Random(seed)
    gene = gene_data["gene"]
    onset_lo, onset_hi = gene_data["onset_range_y"]
    patients = []
    for i in range(40):
        onset = round(rng.uniform(onset_lo, max(onset_lo + 1.0, onset_hi)), 1)
        # severity weighting
        r = rng.random()
        if gene in ("SGCA", "SGCB"):
            sev = "Severe" if r < 0.40 else ("Moderate" if r < 0.72 else "Mild")
        elif gene in ("LMNA",):
            sev = "Severe" if r < 0.35 else ("Moderate" if r < 0.70 else "Mild")
        elif gene in ("DYSF", "FKRP"):
            sev = "Severe" if r < 0.28 else ("Moderate" if r < 0.65 else "Mild")
        elif gene in ("HNRNPDL",):
            sev = "Severe" if r < 0.10 else ("Moderate" if r < 0.50 else "Mild")
        else:
            sev = "Severe" if r < 0.22 else ("Moderate" if r < 0.60 else "Mild")

        # Clinical features
        cardiac_event = rng.random() < (
            0.70 if gene == "LMNA" else
            0.65 if gene == "FKRP" else
            0.50 if gene == "SGCB" else
            0.28 if gene == "SGCA" else 0.05
        )
        respiratory_decline = rng.random() < (
            0.60 if gene == "LMNA" else
            0.55 if gene in ("FKRP", "SGCB") else
            0.30 if gene == "SGCA" else 0.10
        )
        scapular_winging = rng.random() < (0.65 if gene == "CAPN3" else 0.12)
        calf_pseudohy = rng.random() < (
            0.65 if gene in ("SGCA", "SGCB", "FKRP") else
            0.30 if gene == "CAPN3" else 0.05
        )
        asymmetric_calf = rng.random() < (
            0.85 if gene == "ANO5" else
            0.55 if gene == "DYSF" else 0.08
        )
        contractures_obs = rng.random() < (
            0.75 if gene == "LMNA" else
            0.40 if gene in ("SGCA", "SGCB") else 0.08
        )
        rimmed_vacuoles = rng.random() < (0.80 if gene == "HNRNPDL" else 0.05)
        very_high_ck_obs = rng.random() < (0.85 if gene == "DYSF" else
                                            0.60 if gene == "ANO5" else 0.10)
        arrhythmia_on_holter = rng.random() < (0.65 if gene == "LMNA" else
                                                0.25 if gene in ("FKRP", "SGCB") else 0.03)

        # CK range
        if gene == "DYSF":
            ck_val = round(rng.uniform(5000, 30000))
        elif gene == "ANO5" and rng.random() < 0.5:
            ck_val = round(rng.uniform(2000, 15000))
        elif gene in ("SGCA", "SGCB", "FKRP"):
            ck_val = round(rng.uniform(500, 10000))
        elif gene == "CAPN3":
            ck_val = round(rng.uniform(300, 5000))
        elif gene in ("LMNA", "HNRNPDL"):
            ck_val = round(rng.uniform(150, 1500))
        else:
            ck_val = round(rng.uniform(200, 3000))

        # Treatment
        if gene == "DYSF":
            tx = "No steroids; supportive physiotherapy"
        elif gene == "LMNA":
            tx = "ICD ± ACEi/beta-blocker; cardiac monitoring"
        elif gene in ("SGCA", "SGCB"):
            tx = "Cardiac monitoring; physiotherapy; gene therapy trial consideration"
        elif gene == "FKRP":
            tx = "Annual echo/Holter; respiratory monitoring; physiotherapy"
        elif gene == "ANO5":
            tx = "Low-intensity exercise; avoid rhabdomyolysis triggers; physiotherapy"
        elif gene == "HNRNPDL":
            tx = "Physiotherapy; biopsy confirmation of rimmed vacuoles"
        else:
            tx = "Physiotherapy; AFOs; scapular fixation if winging severe"

        pid = f"LGMD-{gene}-{seed}-{i+1:03d}"
        sex = rng.choice(["M", "F"])
        patients.append({
            "id": pid, "gene": gene, "sex": sex,
            "onset_age_y": onset, "severity": sev,
            "cardiac_event": cardiac_event,
            "respiratory_decline": respiratory_decline,
            "scapular_winging": scapular_winging,
            "calf_pseudohypertrophy": calf_pseudohy,
            "asymmetric_calf_wasting": asymmetric_calf,
            "contractures": contractures_obs,
            "rimmed_vacuoles_biopsy": rimmed_vacuoles,
            "very_high_ck": very_high_ck_obs,
            "arrhythmia_on_holter": arrhythmia_on_holter,
            "ck_iu_l": ck_val,
            "current_treatment": tx,
            "inheritance": gene_data["inheritance"].split(".")[0],
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gene_data in enumerate(LGMD_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients(gene_data, seed))
    return all_pts


def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)
    gene_counts = {}
    for p in patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    cardiac_n = sum(1 for p in patients if p["cardiac_event"])
    resp_n = sum(1 for p in patients if p["respiratory_decline"])
    scap_n = sum(1 for p in patients if p["scapular_winging"])
    calf_hyp_n = sum(1 for p in patients if p["calf_pseudohypertrophy"])
    asym_n = sum(1 for p in patients if p["asymmetric_calf_wasting"])
    contract_n = sum(1 for p in patients if p["contractures"])
    rimmed_n = sum(1 for p in patients if p["rimmed_vacuoles_biopsy"])
    arrhyth_n = sum(1 for p in patients if p["arrhythmia_on_holter"])
    very_high_ck_n = sum(1 for p in patients if p["very_high_ck"])
    for p in patients:
        sev[p["severity"]] += 1

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 1)
    mean_ck = round(sum(p["ck_iu_l"] for p in patients) / n)

    return {
        "atlas": "LGMD-Atlas",
        "full_name": "Complete 8-Gene Limb-Girdle Muscular Dystrophy (LGMD) Atlas",
        "subtitle": "CAPN3·DYSF·SGCA·SGCB·FKRP·ANO5·LMNA·HNRNPDL — 320 patients (8×40, seeds 1030–1037)",
        "description": (
            "Comprehensive atlas of the 8 most clinically important Limb-Girdle Muscular Dystrophy (LGMD) genes. "
            "Covers AR subtypes (CAPN3 R1, DYSF R2, SGCA R3, SGCB R4, FKRP R9, ANO5 R12) and "
            "AD subtypes (LMNA D1, HNRNPDL D3). "
            "CRITICAL TREATMENT DISTINCTIONS: DYSF = corticosteroids WORSEN (membrane repair suppressed); "
            "LMNA = cardiac arrhythmia/DCM LEADING cause of death (ICD mandatory); "
            "FKRP = cardiac screening MANDATORY (DCM 60-80%); "
            "SGCA/SGCB = dystrophin NORMAL (exclude BMD before sarcoglycan panel)."
        ),
        "total_patients": n,
        "genes_covered": len(LGMD_GENES),
        "patients_per_gene": 40,
        "seed_range": "1030–1037",
        "gene_list": [g["gene"] for g in LGMD_GENES],
        "lgmd_category_breakdown": {
            "AR Calpain-3 Protease (Most Common AR-LGMD)": ["CAPN3"],
            "AR Membrane Repair (Dysferlin)": ["DYSF"],
            "AR Sarcoglycan Complex (Alpha)": ["SGCA"],
            "AR Sarcoglycan Complex (Beta)": ["SGCB"],
            "AR Dystroglycanopathy (FKRP)": ["FKRP"],
            "AR Membrane Repair (Anoctamin-5)": ["ANO5"],
            "AD Nuclear Envelope (Laminopathy)": ["LMNA"],
            "AD RNA-Binding Protein (hnRNP D-like)": ["HNRNPDL"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "mean_ck_iu_l": mean_ck,
        "clinical_features_prevalence": {
            "cardiac_event_pct": round(100 * cardiac_n / n, 1),
            "respiratory_decline_pct": round(100 * resp_n / n, 1),
            "scapular_winging_pct": round(100 * scap_n / n, 1),
            "calf_pseudohypertrophy_pct": round(100 * calf_hyp_n / n, 1),
            "asymmetric_calf_wasting_pct": round(100 * asym_n / n, 1),
            "contractures_pct": round(100 * contract_n / n, 1),
            "rimmed_vacuoles_biopsy_pct": round(100 * rimmed_n / n, 1),
            "arrhythmia_on_holter_pct": round(100 * arrhyth_n / n, 1),
            "very_high_ck_pct": round(100 * very_high_ck_n / n, 1),
        },
        "key_teaching_points": [
            "DYSF-LGMD R2: STEROIDS WORSEN — inflammatory biopsy mimics polymyositis; molecular confirmation BEFORE immunosuppression",
            "LMNA-LGMD D1: CARDIAC FIRST — lethal arrhythmia/DCM before muscle weakness severe; ICD mandatory; annual Holter+echo",
            "FKRP-LGMD R9: CARDIAC MANDATORY — DCM 60-80% by 4th decade; annual echo+Holter even if asymptomatic",
            "SGCA/SGCB: DYSTROPHIN NORMAL — sarcoglycan panel on biopsy distinguishes from DMD/BMD; gene therapy trials (SGCA Phase 3)",
            "CAPN3-LGMD R1: MOST COMMON AR-LGMD — scapular winging early; secondary calpain-3 loss common in other LGMDs (WES mandatory)",
            "ANO5-LGMD R12: ASYMMETRIC CALF WASTING — characteristic MRI pattern; rhabdomyolysis with vigorous exercise; Miyoshi-3 allelic",
            "HNRNPDL-LGMD D3: RIMMED VACUOLES + AD — Brazilian/Chinese founder pAsp378Asn; RNA stress granule aggregopathy (FUS/TDP-43 analogy)",
            "ALL LGMD: Statins caution (CK monitoring); aminoglycosides no NMJ concern (unlike CMS); exercise graded except LMNA high-intensity CI",
        ],
        "drug_alerts": [
            "DYSF-LGMD R2: Corticosteroids ABSOLUTELY AVOID — reduce macrophage membrane repair; accelerate decline",
            "LMNA-LGMD D1: Amiodarone + ICD for arrhythmia; avoid antiarrhythmics that prolong QTc without ICD backup",
            "ALL LGMD: Statins — may worsen (CK monitoring mandatory; switch to alternative if CK >5× ULN on statins)",
            "LMNA/FKRP/SGCB: General anaesthesia risk — cardiac arrhythmia trigger; cardiology pre-operative assessment mandatory",
            "FKRP-LGMD R9: ACE inhibitor + beta-blocker if DCM confirmed — do not delay",
            "ANO5: High-intensity exercise → rhabdomyolysis → acute kidney injury; warn patient and sports team",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    gene_profiles = []
    for gene_data in LGMD_GENES:
        gene_pts = [p for p in patients if p["gene"] == gene_data["gene"]]
        n = len(gene_pts)
        sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for p in gene_pts:
            sev[p["severity"]] += 1
        mean_ck_g = round(sum(p["ck_iu_l"] for p in gene_pts) / n)
        gene_profiles.append({
            "gene": gene_data["gene"],
            "protein": gene_data["protein"],
            "alias": gene_data["alias"],
            "locus": gene_data["locus"],
            "omim_gene": gene_data["omim_gene"],
            "omim_disease": gene_data["omim_disease"],
            "inheritance": gene_data["inheritance"],
            "lgmd_group": gene_data["lgmd_group"],
            "lgmd_type": gene_data["lgmd_type"],
            "aa": gene_data["aa"],
            "kDa": gene_data["kDa"],
            "gene_class": gene_data["gene_class"],
            "phenotype": gene_data["phenotype"],
            "disease": gene_data["disease"],
            "treatment_options": gene_data["treatment_options"],
            "key_ddx": gene_data["key_ddx"],
            "onset_range_y": list(gene_data["onset_range_y"]),
            "n_patients": n,
            "cardiac_risk": gene_data["cardiac_risk"],
            "respiratory_risk": gene_data["respiratory_risk"],
            "contractures": gene_data["contractures"],
            "facial_weakness": gene_data["facial_weakness"],
            "very_high_ck": gene_data["very_high_ck"],
            "scapular_winging": gene_data["scapular_winging"],
            "calf_pseudohypertrophy": gene_data["calf_pseudohypertrophy"],
            "asymmetric": gene_data["asymmetric"],
            "steroids_safe": gene_data["steroids_safe"],
            "ad_inheritance": gene_data["ad_inheritance"],
            "mean_ck_iu_l": mean_ck_g,
            "severity_distribution": {
                "mild_pct": round(100 * sev["Mild"] / n, 1),
                "moderate_pct": round(100 * sev["Moderate"] / n, 1),
                "severe_pct": round(100 * sev["Severe"] / n, 1),
            },
            "clinical_features": {
                "cardiac_event_pct": round(100 * sum(1 for p in gene_pts if p["cardiac_event"]) / n, 1),
                "respiratory_decline_pct": round(100 * sum(1 for p in gene_pts if p["respiratory_decline"]) / n, 1),
                "scapular_winging_pct": round(100 * sum(1 for p in gene_pts if p["scapular_winging"]) / n, 1),
                "calf_pseudohypertrophy_pct": round(100 * sum(1 for p in gene_pts if p["calf_pseudohypertrophy"]) / n, 1),
                "asymmetric_calf_pct": round(100 * sum(1 for p in gene_pts if p["asymmetric_calf_wasting"]) / n, 1),
                "contractures_pct": round(100 * sum(1 for p in gene_pts if p["contractures"]) / n, 1),
                "rimmed_vacuoles_pct": round(100 * sum(1 for p in gene_pts if p["rimmed_vacuoles_biopsy"]) / n, 1),
                "arrhythmia_pct": round(100 * sum(1 for p in gene_pts if p["arrhythmia_on_holter"]) / n, 1),
                "very_high_ck_pct": round(100 * sum(1 for p in gene_pts if p["very_high_ck"]) / n, 1),
            },
            "sample_patients": gene_pts[:3],
        })
    return {
        "atlas": "LGMD-Atlas",
        "genes": gene_profiles,
        "total_patients": len(patients),
    }


def get_definitions() -> dict:
    return {
        "atlas": "LGMD-Atlas",
        "definitions": [
            {
                "term": "Limb-Girdle Muscular Dystrophy (LGMD)",
                "definition": (
                    "Clinically and genetically heterogeneous group of inherited muscle diseases characterised "
                    "by progressive proximal shoulder-girdle and pelvic-girdle weakness. "
                    "Dystrophin immunohistochemistry NORMAL or near-normal (excluding dystrophinopathies). "
                    "Currently ≥30 genetic subtypes. New ENMC 2017 nomenclature: "
                    "LGMD Rx (AR, numbered R1, R2...) and LGMD Dx (AD, D1, D2...). "
                    "Diagnosis: muscle biopsy IHC + WES/gene panel + EMG + muscle MRI."
                ),
            },
            {
                "term": "Sarcoglycan Complex",
                "definition": (
                    "Tetrameric transmembrane glycoprotein complex (α, β, γ, δ sarcoglycans) forming part of "
                    "the dystrophin-associated protein complex (DAPC). Links intracellular actin (via dystrophin) "
                    "to extracellular matrix (via dystroglycan). Loss of any one sarcoglycan → secondary "
                    "reduction of others on IHC. Mutations: SGCA (α-SG), SGCB (β-SG), SGCG (γ-SG), SGCD (δ-SG). "
                    "Clinical phenotype Duchenne-like but dystrophin NORMAL — critical IHC distinction."
                ),
            },
            {
                "term": "Dysferlin (Membrane Repair)",
                "definition": (
                    "Large C2-domain protein (2080 aa) essential for calcium-triggered sarcolemmal repair. "
                    "Accumulates at injury sites → membrane patching via lysosome exocytosis. "
                    "LOF → repeated micro-injuries → macrophage infiltration → progressive myopathy. "
                    "Inflammatory biopsy mimics polymyositis → STEROIDS WORSEN (reduce macrophage repair). "
                    "Monocyte assay (blood test) screens for dysferlin deficiency. "
                    "Same gene causes LGMD R2 (proximal) and Miyoshi myopathy 1 (distal calf)."
                ),
            },
            {
                "term": "Calpain-3 (CAPN3)",
                "definition": (
                    "Skeletal-muscle-specific calcium-activated cysteine protease (821 aa). "
                    "Associates with titin at N2A/IS2 regions; required for sarcomere remodelling. "
                    "Most common AR-LGMD gene worldwide (~30-35% of AR-LGMD). "
                    "SECONDARY CALPAIN-3 REDUCTION: common in DYSF, SGCA, SGCB, FKRP myopathies — "
                    "absent calpain-3 IHC does NOT equal primary CAPN3 mutation. WES required."
                ),
            },
            {
                "term": "Dystroglycanopathy (FKRP)",
                "definition": (
                    "Group of muscular dystrophies caused by defective O-mannosyl glycosylation of "
                    "α-dystroglycan (α-DG). α-DG glycosylation is required for extracellular matrix binding. "
                    "FKRP transfers CDP-ribitol in the glycosylation pathway. "
                    "Reduced glycosylated α-DG on IHC (IIH6 antibody) is the diagnostic marker. "
                    "FKRP mutations cause a spectrum: mild LGMD R9 (pLeu276Ile homozygous) to "
                    "severe Walker-Warburg syndrome (structural brain/eye)."
                ),
            },
            {
                "term": "Laminopathy (LMNA)",
                "definition": (
                    "Group of diseases caused by LMNA mutations affecting lamin A/C — nuclear lamina "
                    "intermediate filaments. ALLELIC: >15 distinct phenotypes from same gene. "
                    "CARDIAC RISK: Ventricular arrhythmia + DCM → sudden cardiac death in 15-40% WITHOUT ICD. "
                    "Cardiac events often PRECEDE limb weakness. "
                    "LGMD D1/1B risk score (2019 ESC): non-sustained VT + AV block + AF + LVEF + LNHB → "
                    "guides ICD timing. LMNA = highest SCD risk among all LGMD subtypes."
                ),
            },
            {
                "term": "α-Dystroglycan IHC (IIH6 antibody)",
                "definition": (
                    "Immunohistochemistry using IIH6 (or VIA4-1) monoclonal antibody against glycosylated "
                    "α-dystroglycan epitope. REDUCED or absent staining = suggests dystroglycanopathy. "
                    "NOT subtype-specific: any of the >20 dystroglycanopathy genes may reduce IIH6 staining. "
                    "NORMAL staining does NOT exclude mild dystroglycanopathy. "
                    "Must be combined with gene panel for diagnosis."
                ),
            },
            {
                "term": "Rimmed Vacuoles",
                "definition": (
                    "Autophagic vacuoles in myofibres visible on muscle biopsy as basophilic granules "
                    "surrounding a vacuole on haematoxylin & eosin (H&E); best seen on modified Gomori "
                    "trichrome stain (red-rimmed). Indicate protein aggregate accumulation and autophagy. "
                    "In LGMD context: HNRNPDL myopathy (D3), GNE myopathy, VCP myopathy, IBM. "
                    "NOT specific to HNRNPDL — requires IHC for hnRNP D-like protein inclusions + WES."
                ),
            },
            {
                "term": "Corticosteroids in LGMD",
                "definition": (
                    "Deflazacort/prednisolone: beneficial in DMD (delays loss of ambulation). "
                    "In most LGMD: evidence limited; off-label extrapolated from DMD. "
                    "DYSF-LGMD R2: CONTRAINDICATED — reduce macrophage-mediated membrane repair; worsen course. "
                    "Confirm molecular diagnosis BEFORE starting steroids (DYSF biopsy mimics polymyositis). "
                    "Other LGMD: use cautiously with regular CK monitoring; benefit unclear."
                ),
            },
            {
                "term": "ICD in LMNA Myopathy",
                "definition": (
                    "Implantable cardioverter-defibrillator — mandatory consideration in LMNA-LGMD D1. "
                    "Indications: AV block grade 2-3; sustained VT; non-sustained VT with LVEF <45%; "
                    "high-risk genotype (pArg249Gln, missense in filament-forming domain). "
                    "2019 ESC HCM guidelines provide LMNA-specific sudden death risk calculator. "
                    "ICD protects against VT/VF; cardiac resynchronisation if LBBB + reduced EF. "
                    "Cardiac transplant: final option in refractory DCM."
                ),
            },
            {
                "term": "Muscle MRI in LGMD",
                "definition": (
                    "T1-weighted MRI shows fatty replacement (bright signal); STIR shows oedema (acute/active). "
                    "Selective muscle involvement patterns are diagnostically useful: "
                    "CAPN3: posterior thigh, adductors → late shoulder. "
                    "DYSF: gastrocnemius medial head, soleus, biceps femoris early. "
                    "FKRP: adductor magnus, posterior thigh; hamstrings predominant. "
                    "LMNA: humeroperoneal early. ANO5: gastrocnemius medial head + posterior thigh asymmetric. "
                    "MRI pattern combined with biopsy and WES provides highest diagnostic yield."
                ),
            },
            {
                "term": "Gene Therapy in Sarcoglycanopathies",
                "definition": (
                    "Recombinant adeno-associated virus (AAV) delivery of functional sarcoglycan cDNA. "
                    "Most advanced: AAVrh74.MHCK7.SGCA (Sarepta) for SGCA — Phase 3 (IGNITE-LGMD2D) 2025. "
                    "MHCK7 promoter: muscle-specific (cardiac + skeletal); rh74 capsid: high muscle tropism. "
                    "FKRP gene therapy: AAV-FKRP preclinical trials ongoing. "
                    "DYSF: minidysferlin constructs (2080 aa → ~1000 aa) — packaging size challenge. "
                    "CAPN3: AAV1.MHCK7.CAPN3 (Encoded Therapeutics) — Phase 1/2 ongoing."
                ),
            },
            {
                "term": "Dystrophin IHC in LGMD Diagnosis",
                "definition": (
                    "Dystrophin immunohistochemistry (rod domain + N-term + C-term antibodies) is the "
                    "FIRST biopsy test to distinguish LGMD from dystrophinopathy (DMD/BMD). "
                    "In all 8 LGMD subtypes covered here: dystrophin NORMAL or near-normal. "
                    "Exceptions to be aware of: dystrophin can be mildly reduced secondary in some LGMD. "
                    "If dystrophin IHC normal → proceed to sarcoglycan panel + dystroglycan panel + WES."
                ),
            },
        ],
    }
