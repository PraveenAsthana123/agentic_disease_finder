#!/usr/bin/env python3
"""Hereditary-Type-I-Interferonopathy-Atlas — Complete 8-Gene Hereditary Type I Interferonopathy Atlas
(TREX1 · RNASEH2B · RNASEH2A · RNASEH2C · SAMHD1 · ADAR1 · IFIH1 · STING1).

TREX1    (Three Prime Repair Exonuclease 1; 314 aa; 3p21.31; AD/AR;
          Aicardi-Goutières Syndrome Type 1 (AGS1) / Familial Chilblain Lupus (FCL) / SLE;
          MOST COMMON AGS GENE — accounts for ~25% of all AGS;
          AD: FCL + milder AGS; AR: severe early-onset Aicardi-Goutières;
          INTRACEREBRAL CALCIFICATIONS (basal ganglia+white matter) + ISG SCORE ELEVATED PATHOGNOMONIC;
          seed SEED_BASE+0).
RNASEH2B (Ribonuclease H2 Subunit B; 312 aa; 13q14.3; AR;
          Aicardi-Goutières Syndrome Type 3 (AGS3);
          MOST COMMON AR AGS GENE IN EUROPEAN POPULATION — mildest AGS phenotype;
          PROMINENT SPASTIC PARAPLEGIA + PRESERVED LANGUAGE = KEY DDx from other AGS;
          p.Ala177Thr founder in Northern Europe;
          seed SEED_BASE+1).
RNASEH2A (Ribonuclease H2 Subunit A (catalytic); 299 aa; 19p13.13; AR;
          Aicardi-Goutières Syndrome Type 4 (AGS4);
          CATALYTIC SUBUNIT — all enzymatic activity resides here;
          severe early-onset with microcephaly + cerebral atrophy;
          CEREBROVASCULAR COMPLICATIONS including stenosis rare but described;
          seed SEED_BASE+2).
RNASEH2C (Ribonuclease H2 Subunit C; 164 aa; 11q13.1; AR;
          Aicardi-Goutières Syndrome Type 2 (AGS2);
          AGS2 historically first-described AGS gene (RNase H2 complex);
          SMALLEST SUBUNIT — structural role anchoring catalytic RNASEH2A;
          severe phenotype like RNASEH2A;
          seed SEED_BASE+3).
SAMHD1   (SAM Domain and HD Domain 1 dNTPase; 626 aa; 20q11.23; AR;
          Aicardi-Goutières Syndrome Type 5 (AGS5);
          CEREBROVASCULAR DISEASE + ISCHAEMIC STROKE PATHOGNOMONIC (AGS5 unique);
          also causes FAMILIAL CHILBLAIN LUPUS (like TREX1-AD);
          dNTPase restricts retroviral replication (HIV-1 restriction factor);
          seed SEED_BASE+4).
ADAR1    (Adenosine Deaminase RNA-specific; 1226 aa; 1q21.3; AD/AR;
          Aicardi-Goutières Syndrome Type 6 (AGS6) + Dyschromatosis Symmetrica Hereditaria (DSH);
          BILATERAL STRIATAL NECROSIS on MRI PATHOGNOMONIC for ADAR1-AGS;
          MIXED HYPO/HYPERPIGMENTED MACULES (DSH) on extremities = skin phenotype;
          biallelic = severe AGS; heterozygous p.Gly1007Arg = DSH + milder AGS;
          seed SEED_BASE+5).
IFIH1    (Interferon Induced with Helicase C Domain 1 / MDA5; 1025 aa; 2q24.2; AD GOF;
          Aicardi-Goutières Syndrome Type 7 (AGS7);
          MDA5 CYTOSOLIC RNA HELICASE — constitutive activation (GOF) drives ISG elevation;
          MILDER AGS PHENOTYPE than TREX1/RNASEH2 — often presents with Singleton de novo;
          ALSO: IFIH1 GOF associated with Type 1 Diabetes protection (loss-of-function = T1DM risk);
          seed SEED_BASE+6).
STING1   (Stimulator of Interferon Genes / TMEM173; 379 aa; 5q31.2; AD GOF;
          STING-Associated Vasculopathy with onset in Infancy (SAVI);
          NOT classical AGS — DISTINCT PHENOTYPE: early-onset VASCULITIS;
          CUTANEOUS VASCULITIS (necrotic ulcers nose/ears/cheeks/fingers) PATHOGNOMONIC;
          INTERSTITIAL LUNG DISEASE (ILD) — major cause of mortality;
          JAK INHIBITORS (ruxolitinib/baricitinib) — HIGHLY EFFECTIVE for SAVI;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2318-2325).
"""

import random

SEED_BASE = 2318

IFN_GENES = [
    # -- TREX1 — AGS1 / FCL / SLE --------------------------------------------------------
    {
        "gene": "TREX1",
        "alt_name": (
            "TREX1 (TREX1-314aa-3p21.31 / AD-AR — AGS1-FCL-SLE — "
            "MOST-COMMON-AGS-GENE-25pct-All-AGS — "
            "INTRACEREBRAL-CALCIFICATIONS-PATHOGNOMONIC — "
            "ISG-SCORE-ELEVATED-Janus-Kinase-Inhibitor-Emerging)"
        ),
        "protein": (
            "TREX1 -- 3p21.31 AD/AR -- TREX1-314aa -- "
            "Three-Prime-Repair-Exonuclease-1-36kDa-ER-Membrane-Anchored-3'-5'-DNA-Exonuclease -- "
            "Degrades-Cytosolic-ssDNA-Prevents-cGAS-STING-Pathway-Activation -- "
            "TREX1-Deficiency-Accumulation-Self-DNA-Fragments-Triggers-Type-I-IFN-Production -- "
            "AD-FCL-Familial-Chilblain-Lupus-Cold-Triggered-Acral-Ulcers-40pct-TREX1-AD -- "
            "AR-Biallelic-AGS1-Severe-Neonatal-Onset-Progressive-Encephalopathy -- "
            "INTRACEREBRAL-CALCIFICATIONS-BASAL-GANGLIA-WHITE-MATTER-CT-SCAN-PATHOGNOMONIC -- "
            "MOST-COMMON-AGS-GENE-25pct-All-AGS-Cases-Worldwide -- "
            "ISG-SCORE-Interferon-Stimulated-Gene-Score-ELEVATED-HALLMARK-ALL-AGS -- "
            "cGAS-STING-PATHWAY-TARGET-Baricitinib-Ruxolitinib-JAK-Inhibitors-Emerging -- "
            "OMIM-Gene-606609-Disease-AGS1-225750-FCL-610448-SLE-152700"
        ),
        "locus": "3p21.31",
        "protein_size": "314 aa / 36 kDa",
        "inheritance": (
            "AD (FCL/milder AGS) or AR (severe AGS1); biallelic LOF = severe AGS1 (worst outcome); "
            "heterozygous LOF = Familial Chilblain Lupus (FCL) = milder skin-dominant phenotype; "
            "heterozygous gain-of-function mutations (p.Asp18His) = severe AD-AGS1; "
            "TREX1 mutations also increase risk of systemic lupus erythematosus (SLE); "
            "most common AGS gene: ~25% of all AGS; de novo mutations account for ~10% AD cases"
        ),
        "interferonopathy_category": "AGS type 1 (TREX1; cytosolic DNA sensing — cGAS-STING pathway; most common AGS gene ~25%)",
        "pathognomonic": (
            "INTRACEREBRAL CALCIFICATIONS — bilateral basal ganglia (putamen > globus pallidus) + periventricular white matter calcifications on CT PATHOGNOMONIC for AGS; "
            "ELEVATED ISG (INTERFERON-STIMULATED GENE) SCORE in blood — >2 SD above control mean = positive IFN signature; "
            "CSF INTERFERON-ALPHA ELEVATION in acute phase (>2 IU/mL); "
            "FCL (Familial Chilblain Lupus) in AD cases: cold-triggered acral ulcerating skin lesions on fingers/toes/ears; "
            "PROGRESSIVE ENCEPHALOPATHY in first year of life (neonatal/infantile onset for AR cases)"
        ),
        "treatment": (
            "JAK INHIBITORS — baricitinib (Olumiant) or ruxolitinib: "
            "block JAK1/JAK2 downstream of IFNAR (IFN receptor) → reduce ISG score + clinical stabilisation; "
            "EVIDENCE: small case series + one prospective trial (TOFA-SAVI/AGS); "
            "TARGET ISG SCORE normalisation + clinical stability (reduced spasticity/cognitive plateau); "
            "FCL MANAGEMENT: cold avoidance + topical steroids + hydroxychloroquine; "
            "SUPPORTIVE: physiotherapy (spasticity) + anticonvulsants (seizures 30-40%) + "
            "speech therapy + feeding support (NG/PEG if dysphagia); "
            "IVIG: not indicated (no humoral immunodeficiency); "
            "HYDROXYCHLOROQUINE: useful in FCL/SLE manifestations; "
            "CORTICOSTEROIDS: limited utility; anecdotal benefit in acute deterioration; "
            "GENETIC COUNSELLING: AR (biallelic) → 25% recurrence; AD (FCL) → 50% recurrence"
        ),
        "key_features": [
            "MOST COMMON AGS GENE: ~25% of all AGS cases worldwide (largest single-gene contribution)",
            "INTRACEREBRAL CALCIFICATIONS (bilateral basal ganglia + periventricular white matter) on plain CT PATHOGNOMONIC for AGS",
            "ELEVATED ISG SCORE: interferon-stimulated gene score >2 SD above control mean = positive IFN signature = hallmark of all AGS",
            "FCL (Familial Chilblain Lupus): AD heterozygous mutations → cold-triggered acral skin ulcers (fingers/toes/ears) without encephalopathy",
            "cGAS-STING PATHWAY: TREX1 degrades cytosolic ssDNA → TREX1 deficiency → ssDNA accumulates → cGAS activated → STING → IRF3/IRF7 → type I IFN",
            "NEONATAL ONSET (AR biallelic): fever + irritability + feeding difficulty + progressive neurological decline in first 3-6 months of life",
        ],
        "monitoring": [
            "ISG score (interferon-stimulated gene score): 6-gene or extended panel at diagnosis, 3-monthly on JAK inhibitor therapy (target <2 SD)",
            "MRI brain: at diagnosis + annual; track white matter volume, calcification extent, cerebral atrophy progression",
            "Neurological assessment: spasticity (Ashworth scale), cognitive, seizure diary — quarterly",
            "Skin surveillance: FCL cases — photograph acral lesions + cold provocation history at each visit",
            "Autoimmune panel: ANA, dsDNA, complement (C3/C4), CBC — 6-monthly (SLE risk in heterozygotes)",
            "JAK inhibitor monitoring: CBC + LFTs + lipid profile monthly × 3 months, then quarterly; watch for infections/herpes reactivation",
        ],
        "key_ddx": [
            "RNASEH2B-AGS3 (mildest AGS; spastic paraplegia > encephalopathy; normal cognition possible; TREC assay normal — not immunodeficient)",
            "SAMHD1-AGS5 (cerebrovascular disease + ischaemic stroke DISTINCTIVE; dNTPase also an HIV restriction factor)",
            "Congenital CMV (calcifications but periventricular ring-enhancement; CMV PCR on DBS/CSF; check hearing)",
            "Non-ketotic hyperglycinaemia (elevated CSF/plasma glycine; no calcifications; different age of onset)",
        ],
        "ifn_pathway": "cGAS-STING (cytosolic DNA sensor pathway)",
        "onset_age": "neonatal_to_infant",
        "cerebrovascular_risk": False,
        "skin_phenotype": True,
        "lung_disease": False,
        "fever_episodes": True,
        "scid_type": False,
        "severity_options": [
            "Severe AR-AGS1 (biallelic TREX1; neonatal onset; profound disability; <5yr survival in severe)",
            "Moderate AD-AGS1 (heterozygous GOF; progressive encephalopathy; survival into adulthood possible)",
            "FCL (Familial Chilblain Lupus; AD LOF; skin-dominant; no encephalopathy; lupus features)",
        ],
        "complication_options": ["Progressive spastic quadriplegia", "Epilepsy", "Calcification progression", "Feeding difficulties", "Chilblain lesions (FCL)", "Lupus nephritis"],
        "isg_level_options": ["markedly_elevated", "elevated", "borderline"],
        "treatments_used": ["Baricitinib (JAK inhibitor)", "Ruxolitinib", "Hydroxychloroquine (FCL/SLE)", "Physiotherapy", "Anticonvulsants", "PEG feeding support"],
    },
    # -- RNASEH2B — AGS3 ----------------------------------------------------------------
    {
        "gene": "RNASEH2B",
        "alt_name": (
            "RNASEH2B (RNASEH2B-312aa-13q14.3 / AR — AGS3 — "
            "MILDEST-AGS-PHENOTYPE-Most-Common-AR-AGS-Europe — "
            "SPASTIC-PARAPLEGIA-PRESERVED-LANGUAGE-KEY-DDx — "
            "pAla177Thr-Northern-European-Founder)"
        ),
        "protein": (
            "RNASEH2B -- 13q14.3 AR -- RNASEH2B-312aa -- "
            "RNase-H2-Subunit-B-36kDa-Non-Catalytic-Scaffold-Subunit-Connects-PIP-Box-PCNA -- "
            "RNase-H2-Heterotrimeric-Complex-RNASEH2A-Catalytic-RNASEH2B-RNASEH2C-Structural -- "
            "Cleaves-RNA-Strand-RNA-DNA-Hybrid-Removes-Ribonucleotides-Incorporated-Genome -- "
            "RNASEH2B-Deficiency-Accumulation-Ribonucleotides-Genome-R-Loops-Triggers-IFN -- "
            "MILDEST-AGS-PHENOTYPE-Milder-Than-TREX1-RNASEH2A-SAMHD1-ADAR -- "
            "SPASTIC-PARAPLEGIA-PROMINENT-PRESERVED-LANGUAGE-FUNCTION-KEY-DDx-Other-AGS -- "
            "p.Ala177Thr-Hypomorphic-Allele-Northern-European-Founder-Mutation -- "
            "MOST-COMMON-AR-AGS-GENE-EUROPEAN-POPULATION-35pct-AR-AGS -- "
            "OMIM-Gene-610326-Disease-AGS3-610329"
        ),
        "locus": "13q14.3",
        "protein_size": "312 aa / 36 kDa",
        "inheritance": (
            "AR; biallelic RNASEH2B mutations; "
            "most common AR-AGS gene in European population (~35% of AR AGS); "
            "p.Ala177Thr: Northern European founder allele — hypomorphic, explains mildest AGS phenotype; "
            "compound heterozygotes (p.Ala177Thr + null) → intermediate severity; "
            "null/null biallelic → more severe than p.Ala177Thr/null; "
            "no sex predilection; prenatal diagnosis available by molecular analysis"
        ),
        "interferonopathy_category": "AGS type 3 (RNASEH2B; ribonucleotide excision repair; mildest AGS; most common AR-AGS in Europe)",
        "pathognomonic": (
            "MILDEST AGS PHENOTYPE — SPASTIC PARAPLEGIA WITH PRESERVED COGNITION/LANGUAGE = KEY DDx from other AGS genes; "
            "INTRACEREBRAL CALCIFICATIONS (basal ganglia + white matter) — present but may be subtle; "
            "ISG SCORE ELEVATED but often lower than TREX1/RNASEH2A cases; "
            "p.Ala177Thr NORTHERN EUROPEAN FOUNDER: hypomorphic allele → residual RNase H2 activity → milder IFN response; "
            "MOTOR PREDOMINANT — walking difficulties, spastic gait, lower limb spasticity without global regression"
        ),
        "treatment": (
            "JAK INHIBITORS (baricitinib/ruxolitinib): emerging evidence; "
            "MILDEST AGS phenotype — motor stabilisation goal; "
            "PHYSIOTHERAPY: intensive for spastic paraplegia (stretching, hydrotherapy, orthoses); "
            "BACLOFEN: oral or intrathecal for severe spasticity; "
            "BOTULINUM TOXIN: for focal spasticity (equinovarus deformity); "
            "ANTICONVULSANTS: seizures less common in AGS3 vs AGS1/AGS4; "
            "SURVEILLANCE: annual ophthalmology (nystagmus, optic atrophy rare but described); "
            "GENETIC COUNSELLING: AR — 25% recurrence; "
            "PROGNOSIS: relatively good in p.Ala177Thr cases — majority reach adulthood with preserved communication; "
            "MOLECULAR DIAGNOSIS: WES or targeted AGS gene panel (test all 7 AGS genes simultaneously)"
        ),
        "key_features": [
            "MILDEST AGS PHENOTYPE: spastic paraplegia with preserved language and cognition — KEY DDx from TREX1/RNASEH2A/SAMHD1 (which cause global regression)",
            "MOST COMMON AR-AGS IN EUROPE: ~35% of all AR-AGS cases; p.Ala177Thr Northern European founder allele",
            "RNase H2 complex: RNASEH2A (catalytic) + RNASEH2B (scaffold, PCNA-binding) + RNASEH2C (structural) — all three subunits required for function",
            "RIBONUCLEOTIDE ACCUMULATION: RNase H2 deficiency → mis-incorporated ribonucleotides retained in genomic DNA → R-loops → cytosolic nucleic acid sensing → type I IFN",
            "p.Ala177Thr FOUNDER: hypomorphic — retains partial RNase H2 activity → milder IFN response → milder encephalopathy vs null mutations",
            "MOTOR-PREDOMINANT PRESENTATION: walking difficulties first, spastic gait — often misdiagnosed as hereditary spastic paraplegia (HSP) before calcifications found",
        ],
        "monitoring": [
            "ISG score at diagnosis and 6-monthly (lower baseline than other AGS subtypes)",
            "MRI brain annually: track calcification extent, white matter signal, any cerebral atrophy",
            "Motor function: 6-minute walk test / Gross Motor Function Classification — 6-monthly (track stability)",
            "Neuropsychological assessment annually: cognitive trajectory (preserved in p.Ala177Thr cases)",
            "Seizure diary if epileptic (less common in AGS3 vs AGS1)",
            "JAK inhibitor monitoring if initiated: CBC + LFTs quarterly",
        ],
        "key_ddx": [
            "Hereditary Spastic Paraplegia (HSP) — ATL1/SPG4/SPG3A: no calcifications; normal ISG score; different gene panel",
            "TREX1-AGS1 (more severe; global regression + neonatal onset; stronger IFN signature; FCL in AD cases)",
            "RNASEH2A/C-AGS4/2 (same gene complex but catalytic subunit deficiency → more severe phenotype)",
            "Pelizaeus-Merzbacher disease (leukodystrophy; no calcifications; PLP1 mutations; X-linked)",
        ],
        "ifn_pathway": "RNase H2 / ribonucleotide excision repair — genome-embedded ribonucleotides → cGAS-independent IFN",
        "onset_age": "infant_to_toddler",
        "cerebrovascular_risk": False,
        "skin_phenotype": False,
        "lung_disease": False,
        "fever_episodes": True,
        "scid_type": False,
        "severity_options": [
            "Mild (p.Ala177Thr homozygous or compound het; preserved cognition; ambulatory with support)",
            "Moderate (p.Ala177Thr + null; spastic paraplegia + mild cognitive involvement)",
            "Severe (biallelic null; global AGS phenotype with calcifications + regression)",
        ],
        "complication_options": ["Spastic paraplegia", "Seizures", "Feeding difficulties", "Nystagmus", "Scoliosis", "Joint contractures"],
        "isg_level_options": ["elevated", "mildly_elevated", "borderline"],
        "treatments_used": ["Physiotherapy", "Baclofen (spasticity)", "Botulinum toxin", "Baricitinib (emerging)", "Anticonvulsants if needed", "Orthopaedic interventions"],
    },
    # -- RNASEH2A — AGS4 ---------------------------------------------------------------
    {
        "gene": "RNASEH2A",
        "alt_name": (
            "RNASEH2A (RNASEH2A-299aa-19p13.13 / AR — AGS4 — "
            "CATALYTIC-SUBUNIT-RNase-H2-Complex — "
            "SEVERE-AGS-Microcephaly-Cerebral-Atrophy-ISG-High — "
            "Cerebrovascular-Stenosis-Rare-But-Described)"
        ),
        "protein": (
            "RNASEH2A -- 19p13.13 AR -- RNASEH2A-299aa -- "
            "RNase-H2-Subunit-A-Catalytic-33kDa-Endonuclease-Cleaves-5-End-Ribonucleotide-RNA-DNA-Hybrid -- "
            "ALL-RNase-H2-Enzymatic-Activity-Resides-In-RNASEH2A-Subunit-B-C-Only-Structural -- "
            "RNase-H2-Cleaves-Single-Ribonucleotides-Embedded-Genomic-DNA-Prevents-R-Loop-Accumulation -- "
            "RNASEH2A-Null-No-Residual-Enzymatic-Activity-Most-Severe-RNase-H2-Subunit-Defect -- "
            "SEVERE-EARLY-ONSET-AGS-Microcephaly-Progressive-Cerebral-Atrophy-Calcifications -- "
            "HIGHEST-ISG-SCORE-Of-All-RNase-H2-Subunits-Null-Activity-Maximum-IFN-Induction -- "
            "OMIM-Gene-606034-Disease-AGS4-610333"
        ),
        "locus": "19p13.13",
        "protein_size": "299 aa / 33 kDa",
        "inheritance": (
            "AR; biallelic RNASEH2A mutations; "
            "rarest of the three RNase H2 subunit AGS genes; "
            "all catalytic activity resides in RNASEH2A — null mutations → complete RNase H2 loss; "
            "no founder mutations identified; "
            "severe phenotype expected with biallelic null mutations; "
            "compound heterozygotes with hypomorphic alleles → intermediate severity"
        ),
        "interferonopathy_category": "AGS type 4 (RNASEH2A; catalytic RNase H2 subunit; complete enzymatic loss → severe IFN induction)",
        "pathognomonic": (
            "SEVERE EARLY-ONSET AGS with MICROCEPHALY and PROGRESSIVE CEREBRAL ATROPHY — distinguishes from RNASEH2B (mildest); "
            "INTRACEREBRAL CALCIFICATIONS (basal ganglia + periventricular white matter); "
            "HIGHEST ISG SCORE among RNase H2 subunits (complete loss of catalytic activity → maximum genomic ribonucleotide accumulation); "
            "PROGRESSIVE ENCEPHALOPATHY with psychomotor regression in first year; "
            "CEREBROVASCULAR COMPLICATIONS including rare cerebral vasculopathy described"
        ),
        "treatment": (
            "JAK INHIBITORS (baricitinib/ruxolitinib): most evidence-base in TREX1; extrapolated to all AGS including RNASEH2A; "
            "SUPPORTIVE CARE: physiotherapy (tone management) + speech therapy + feeding support; "
            "ANTICONVULSANTS: high seizure burden — levetiracetam/lamotrigine; "
            "ANTISPASTIC AGENTS: baclofen (oral or intrathecal) + benzodiazepines; "
            "GASTROSTOMY (PEG): required in majority due to severe feeding difficulties; "
            "OPHTHALMOLOGY: nystagmus + optic atrophy monitoring; "
            "PROGNOSIS: severe — most patients require full-time care; "
            "PALLIATIVE CARE: early integration appropriate given severe natural history"
        ),
        "key_features": [
            "CATALYTIC SUBUNIT: all RNase H2 enzymatic activity resides in RNASEH2A — biallelic null → complete enzyme loss → maximum ribonucleotide accumulation",
            "HIGHEST ISG SCORE of all RNase H2 subunit AGS genes (RNASEH2A > RNASEH2C > RNASEH2B)",
            "SEVERE PHENOTYPE: microcephaly + progressive cerebral atrophy + calcifications — more severe than RNASEH2B (mildest AGS)",
            "RNase H2 COMPLEX: three subunits (A + B + C) — any one subunit lost → whole complex non-functional; RNASEH2A loss has greatest enzymatic consequence",
            "RIBONUCLEOTIDE INCORPORATION RATE: ~1 ribonucleotide per 7,600 base pairs in normal DNA — RNase H2 essential for removal",
            "RARE AGS GENE: less common than TREX1 or RNASEH2B; often diagnosed through AGS gene panel on unexplained childhood neurological deterioration",
        ],
        "monitoring": [
            "ISG score at diagnosis + 3-monthly (high expected baseline — track response to JAK inhibition)",
            "MRI brain 6-monthly in first 2 years (rapid progression phase); annually thereafter",
            "Head circumference plotted at every visit (microcephaly progression)",
            "Seizure quantification (frequency/type) diary + EEG annually",
            "Nutritional status: weight/height/BMI + gastrostomy tube function if in situ",
            "Ophthalmology 6-monthly: nystagmus, optic atrophy, visual function",
        ],
        "key_ddx": [
            "RNASEH2B-AGS3 (mildest AGS, spastic paraplegia, preserved language — distinguished by milder phenotype)",
            "TREX1-AGS1 (most common AGS; FCL skin features in AD; cGAS-STING pathway vs RNase H2 pathway)",
            "Congenital cytomegalovirus (calcifications periventricular; CMV PCR; SNHL; hepatosplenomegaly at birth)",
            "Non-infectious causes of infantile calcifications: Krabbe disease, GM2 gangliosidosis — no IFN elevation",
        ],
        "ifn_pathway": "RNase H2 / ribonucleotide excision repair — complete catalytic loss → maximal genomic ribonucleotide accumulation → IFN",
        "onset_age": "neonatal_to_infant",
        "cerebrovascular_risk": True,
        "skin_phenotype": False,
        "lung_disease": False,
        "fever_episodes": True,
        "scid_type": False,
        "severity_options": [
            "Severe (biallelic null; microcephaly + cerebral atrophy; full-time care dependency)",
            "Moderate-severe (compound het with hypomorphic; slower progression)",
        ],
        "complication_options": ["Severe epilepsy", "Microcephaly", "Progressive cerebral atrophy", "Spastic quadriplegia", "Feeding failure (PEG)", "Optic atrophy"],
        "isg_level_options": ["markedly_elevated", "elevated"],
        "treatments_used": ["Baricitinib (JAK inhibitor)", "Anticonvulsants (levetiracetam)", "Baclofen (antispastic)", "PEG feeding", "Physiotherapy", "Palliative care integration"],
    },
    # -- RNASEH2C — AGS2 --------------------------------------------------------------
    {
        "gene": "RNASEH2C",
        "alt_name": (
            "RNASEH2C (RNASEH2C-164aa-11q13.1 / AR — AGS2 — "
            "SMALLEST-RNase-H2-Subunit-Structural-Anchoring-Role — "
            "SEVERE-AGS-Like-RNASEH2A — "
            "Historically-First-RNase-H2-Gene-Described-In-AGS)"
        ),
        "protein": (
            "RNASEH2C -- 11q13.1 AR -- RNASEH2C-164aa -- "
            "RNase-H2-Subunit-C-18kDa-Smallest-Subunit-Structural-Anchor-Between-A-and-B -- "
            "RNASEH2C-Bridges-Catalytic-RNASEH2A-To-PCNA-Binding-RNASEH2B-In-Heterotrimer -- "
            "Loss-Of-RNASEH2C-Destabilises-Entire-RNase-H2-Complex-At-Replication-Fork -- "
            "RNASEH2C-Deficiency-Prevents-Ribonucleotide-Removal-From-Newly-Synthesised-DNA -- "
            "SEVERE-AGS-Phenotype-Similar-To-RNASEH2A-Complete-Loss-Enzymatic-Activity-Consequence -- "
            "HISTORICALLY-FIRST-RNase-H2-GENE-DESCRIBED-IN-AGS-Crow-2006-Nature-Genetics -- "
            "OMIM-Gene-610330-Disease-AGS2-610181"
        ),
        "locus": "11q13.1",
        "protein_size": "164 aa / 18 kDa",
        "inheritance": (
            "AR; biallelic RNASEH2C mutations; "
            "historically the first RNase H2 gene described in AGS (Crow et al. 2006); "
            "RNASEH2C is the smallest subunit of the RNase H2 heterotrimer; "
            "structural bridge between catalytic RNASEH2A and PCNA-binding RNASEH2B; "
            "loss of RNASEH2C destabilises entire complex → functional null phenotype; "
            "severe AGS phenotype similar to RNASEH2A (complete enzymatic loss)"
        ),
        "interferonopathy_category": "AGS type 2 (RNASEH2C; structural RNase H2 subunit; complex destabilisation → complete enzymatic loss)",
        "pathognomonic": (
            "SEVERE AGS PHENOTYPE (similar to RNASEH2A) — complex destabilisation equivalent to catalytic null; "
            "INTRACEREBRAL CALCIFICATIONS (basal ganglia + white matter) on CT; "
            "ELEVATED ISG SCORE — comparable to RNASEH2A (complete RNase H2 loss); "
            "EARLY-ONSET PROGRESSIVE ENCEPHALOPATHY — psychomotor regression in first year; "
            "HISTORICAL: first RNase H2 gene identified in AGS families — landmark Crow 2006 paper"
        ),
        "treatment": (
            "JAK INHIBITORS (baricitinib/ruxolitinib): indicated given complete RNase H2 loss + high IFN; "
            "SUPPORTIVE: as for RNASEH2A (physiotherapy, anticonvulsants, baclofen, gastrostomy); "
            "SEIZURE MANAGEMENT: levetiracetam / valproate / lamotrigine per EEG pattern; "
            "SPASTICITY: baclofen + physiotherapy; intrathecal pump in severe cases; "
            "FEEDING: NG → PEG when persistent swallowing difficulty; "
            "OPHTHALMOLOGY: nystagmus monitoring; "
            "PALLIATIVE CARE: early integration appropriate for biallelic null"
        ),
        "key_features": [
            "SMALLEST RNase H2 subunit (164 aa / 18 kDa): structural bridge role anchoring RNASEH2A (catalytic) to RNASEH2B (PCNA-binding) in the heterotrimer",
            "HISTORICALLY FIRST RNase H2 gene described in AGS (Crow et al. 2006, Nature Genetics) — landmark paper defining molecular basis of AGS",
            "COMPLETE COMPLEX DESTABILISATION: loss of RNASEH2C → whole heterotrimer unstable → functional null → ribonucleotide accumulation → IFN",
            "SEVERE PHENOTYPE: comparable to RNASEH2A (not RNASEH2B mildest); early-onset regression + calcifications + high ISG",
            "REPLICATION FORK FUNCTION: RNase H2 complex recruited to replication forks via RNASEH2B-PCNA interaction — RNASEH2C essential for this localisation",
            "PART OF AGS PANEL: all three RNase H2 subunit genes (RNASEH2A, B, C) tested together as RNase H2 deficiency panel; also part of broader AGS 7-gene panel",
        ],
        "monitoring": [
            "ISG score at diagnosis and 3-monthly on JAK inhibitor",
            "MRI brain 6-monthly in first 2 years (progression), annually thereafter",
            "EEG at diagnosis + annually (seizure burden evaluation)",
            "Nutritional monitoring (weight, feeding tolerance, gastrostomy function)",
            "Developmental milestone tracking: Bayley/GMFCS quarterly in first 3 years",
            "Ophthalmology annually: nystagmus, optic atrophy",
        ],
        "key_ddx": [
            "RNASEH2B-AGS3 (milder — spastic paraplegia, preserved language; distinguished by clinical course)",
            "RNASEH2A-AGS4 (same pathway, similar severity; distinguish by molecular testing only)",
            "ADAR1-AGS6 (bilateral striatal necrosis DISTINCTIVE; mixed pigmentation skin phenotype DSH)",
            "Congenital rubella (calcifications; IUGR; cataracts; PDA; maternal serology/newborn PCR)",
        ],
        "ifn_pathway": "RNase H2 / ribonucleotide excision repair — structural subunit loss → complex destabilisation → complete enzymatic loss",
        "onset_age": "neonatal_to_infant",
        "cerebrovascular_risk": False,
        "skin_phenotype": False,
        "lung_disease": False,
        "fever_episodes": True,
        "scid_type": False,
        "severity_options": [
            "Severe (biallelic null; early global regression; full care dependency)",
            "Moderate (compound het with hypomorphic; slower progression)",
        ],
        "complication_options": ["Progressive spastic quadriplegia", "Drug-resistant epilepsy", "Microcephaly", "Feeding failure (PEG)", "Optic atrophy", "Severe cognitive impairment"],
        "isg_level_options": ["markedly_elevated", "elevated"],
        "treatments_used": ["Baricitinib (JAK inhibitor)", "Levetiracetam/Valproate", "Baclofen (antispastic)", "PEG gastrostomy", "Physiotherapy", "Early palliative integration"],
    },
    # -- SAMHD1 — AGS5 / FCL ----------------------------------------------------------
    {
        "gene": "SAMHD1",
        "alt_name": (
            "SAMHD1 (SAMHD1-626aa-20q11.23 / AR — AGS5-FCL — "
            "CEREBROVASCULAR-DISEASE-ISCHAEMIC-STROKE-PATHOGNOMONIC-AGS5-UNIQUE — "
            "dNTPase-HIV-1-Restriction-Factor-CD4+-T-Cell-Depletes-dNTP-Pool — "
            "Familial-Chilblain-Lupus-Overlap-Like-TREX1-AD)"
        ),
        "protein": (
            "SAMHD1 -- 20q11.23 AR -- SAMHD1-626aa -- "
            "SAM-Domain-HD-Domain-1-72kDa-dNTP-Triphosphohydrolase-dNTPase-Restricts-Retroviral-Replication -- "
            "Depletes-Cytosolic-dNTP-Pool-Required-For-Retroviral-Reverse-Transcription -- "
            "HIV-1-RESTRICTION-FACTOR-In-Non-Dividing-Cells-Macrophages-Dendritic-Cells -- "
            "Vpx-(HIV-2-SIV)-Counteracts-SAMHD1-By-Ubiquitin-Proteasome-Degradation -- "
            "SAMHD1-Deficiency-Accumulation-Nucleotide-Metabolites-Triggers-cGAS-Pathway -- "
            "CEREBROVASCULAR-DISEASE-ISCHAEMIC-STROKE-IN-CHILDHOOD-PATHOGNOMONIC-AGS5 -- "
            "FAMILIAL-CHILBLAIN-LUPUS-FCL-Overlap-Like-TREX1-AD-Cold-Triggered-Acral-Lesions -- "
            "OMIM-Gene-606754-Disease-AGS5-612952-FCL2-610448"
        ),
        "locus": "20q11.23",
        "protein_size": "626 aa / 72 kDa",
        "inheritance": (
            "AR; biallelic SAMHD1 mutations; "
            "also causes Familial Chilblain Lupus (FCL2) in heterozygotes (like TREX1-FCL); "
            "SAMHD1: dNTPase — depletes cytosolic dNTP pool essential for retroviruses; "
            "HIV-1 restriction factor in non-dividing cells; Vpx counteracts SAMHD1; "
            "loss of dNTPase function → dNTP accumulation → cGAS pathway activation → type I IFN; "
            "unique AGS5 feature: cerebrovascular complications including ischaemic stroke"
        ),
        "interferonopathy_category": "AGS type 5 (SAMHD1; dNTPase / HIV restriction factor; cerebrovascular disease distinctive)",
        "pathognomonic": (
            "CEREBROVASCULAR DISEASE + ISCHAEMIC STROKE IN CHILDHOOD PATHOGNOMONIC FOR AGS5 — unique among AGS subtypes; "
            "INTRACEREBRAL CALCIFICATIONS (basal ganglia + white matter) as in other AGS; "
            "ELEVATED ISG SCORE with type I IFN signature; "
            "FCL (Familial Chilblain Lupus) in heterozygous carriers: cold-triggered acral ulcers (like TREX1-FCL); "
            "dNTPase deficiency → dNTP pool dysregulation → cytosolic nucleic acid sensing"
        ),
        "treatment": (
            "JAK INHIBITORS (baricitinib/ruxolitinib): for IFN-mediated disease + FCL; "
            "ANTIPLATELET THERAPY (aspirin): for cerebrovascular disease / stroke prevention; "
            "ANTICOAGULATION: considered if recurrent cerebrovascular events; "
            "FCL MANAGEMENT: hydroxychloroquine + cold avoidance; "
            "STROKE REHABILITATION: early physiotherapy + occupational therapy + speech therapy; "
            "ANTICONVULSANTS: if post-stroke epilepsy; "
            "NEUROIMAGING SURVEILLANCE: MRI brain + MRA vessels annually (cerebrovascular risk); "
            "HAEMATOLOGY: CBC watch for thrombocytopenia (lupus component in some); "
            "GENETIC COUNSELLING: AR — 25% recurrence"
        ),
        "key_features": [
            "CEREBROVASCULAR DISEASE + ISCHAEMIC STROKE IN CHILDHOOD PATHOGNOMONIC FOR AGS5 — unique among all AGS subtypes; other AGS genes do not typically cause stroke",
            "HIV-1 RESTRICTION FACTOR: SAMHD1 depletes cytosolic dNTP pool in non-dividing cells (macrophages/dendritic cells) — HIV requires dNTPs for reverse transcription; Vpx (HIV-2/SIV) degrades SAMHD1 to overcome restriction",
            "FCL OVERLAP: heterozygous SAMHD1 mutations → Familial Chilblain Lupus (FCL2) — same cold-triggered acral skin ulcer phenotype as TREX1-FCL1",
            "dNTPase MECHANISM: SAMHD1 hydrolyses dNTPs to deoxyribonucleoside + triphosphate; deficiency → dNTP accumulation → triggers cGAS-STING pathway",
            "DUAL ROLE: restriction factor (innate antiviral) AND genome maintenance (dNTP pool regulation) — explains both IFN signature and cerebrovascular complications",
            "CEREBRAL VASCULOPATHY: MRI shows white matter lesions + basal ganglia calcifications + cerebral artery narrowing in some — MRA mandatory in AGS5",
        ],
        "monitoring": [
            "ISG score at diagnosis and 3-monthly on JAK inhibitor",
            "MRI brain + MR angiography (MRA) annually — cerebrovascular risk unique to AGS5",
            "Neurological assessment: stroke sequelae, spasticity, cognitive trajectory — quarterly",
            "CBC + coagulation profile 6-monthly (thrombocytopenia, lupus anticoagulant)",
            "ANA, dsDNA, complement (C3/C4) — 6-monthly (FCL/lupus risk even in AR cases)",
            "Ophthalmology annually (retinal vasculopathy described in rare cases)",
        ],
        "key_ddx": [
            "TREX1-AGS1 (most common AGS; FCL similar phenotype in AD; cGAS-STING pathway; no cerebrovascular disease)",
            "Childhood arterial ischaemic stroke (CADASIL — NOTCH3; Fabry disease — GLA; sickle cell — HBB; antiphospholipid syndrome)",
            "Moyamoya disease (cerebrovascular stenosis but no calcifications, no IFN signature, different MRI pattern)",
            "Non-AGS lupus (SLE — ANA/dsDNA positive, complement low, normal ISG score in most, no calcifications)",
        ],
        "ifn_pathway": "dNTPase / nucleotide metabolism — cGAS-STING pathway (dNTP pool dysregulation)",
        "onset_age": "infant_to_toddler",
        "cerebrovascular_risk": True,
        "skin_phenotype": True,
        "lung_disease": False,
        "fever_episodes": True,
        "scid_type": False,
        "severity_options": [
            "Severe AR-AGS5 (biallelic null; stroke + encephalopathy + severe disability)",
            "Moderate AR-AGS5 (compound het; encephalopathy without stroke)",
            "FCL2 (heterozygous; skin-dominant; no encephalopathy)",
        ],
        "complication_options": ["Ischaemic stroke", "Spastic hemiplegia post-stroke", "Chilblain lesions (FCL)", "Progressive calcifications", "Post-stroke epilepsy", "Thrombocytopenia"],
        "isg_level_options": ["markedly_elevated", "elevated"],
        "treatments_used": ["Baricitinib (JAK inhibitor)", "Aspirin (stroke prevention)", "Hydroxychloroquine (FCL)", "Stroke rehabilitation", "Anticonvulsants", "Physiotherapy"],
    },
    # -- ADAR1 — AGS6 / DSH ------------------------------------------------------------
    {
        "gene": "ADAR1",
        "alt_name": (
            "ADAR1 (ADAR1-1226aa-1q21.3 / AD-AR — AGS6-DSH — "
            "BILATERAL-STRIATAL-NECROSIS-MRI-PATHOGNOMONIC-ADAR1-AGS — "
            "Dyschromatosis-Symmetrica-Hereditaria-DSH-Mixed-Pigmentation-Skin-Phenotype — "
            "A-to-I-RNA-Editing-Prevents-MDA5-Self-RNA-Recognition)"
        ),
        "protein": (
            "ADAR1 -- 1q21.3 AD/AR -- ADAR1-1226aa -- "
            "Adenosine-Deaminase-RNA-Specific-Enzyme-139kDa-Converts-Adenosine-to-Inosine-A-to-I-RNA-Editing -- "
            "Edits-Endogenous-dsRNA-Self-Structures-Prevents-MDA5-IFIH1-Innate-Immune-Activation -- "
            "ADAR1-Deficiency-Unedited-dsRNA-Recognised-By-MDA5-As-Foreign-Triggers-IFN -- "
            "BILATERAL-STRIATAL-NECROSIS-MRI-Bilateral-Caudate-Putamen-DWI-Restriction-PATHOGNOMONIC -- "
            "DYSCHROMATOSIS-SYMMETRICA-HEREDITARIA-DSH-Mixed-Hypo-Hyperpigmented-Macules-Extremities-AD -- "
            "BIALLELIC-LOF-Severe-AGS6-Encephalopathy-High-IFN; HETEROZYGOUS-p.Gly1007Arg-DSH-Skin-Milder -- "
            "ADAR1-p100-Cytoplasmic-Interferon-Inducible-ADAR1-p150-KEY-Isoform-For-AGS -- "
            "OMIM-Gene-146920-Disease-AGS6-615010-DSH-127400"
        ),
        "locus": "1q21.3",
        "protein_size": "1226 aa / 139 kDa",
        "inheritance": (
            "AD (DSH + milder AGS) or AR (severe AGS6); "
            "ADAR1 p.Gly1007Arg: most common AD mutation → DSH + elevated ISG ± AGS features; "
            "biallelic LOF mutations → severe AGS6 with bilateral striatal necrosis; "
            "ADAR1 has two promoter-driven isoforms: p150 (cytoplasmic, IFN-inducible, Z-alpha domain) and p110 (nuclear); "
            "AGS mutations affect both isoforms; Z-alpha domain mutations affect dsRNA recognition; "
            "compound heterozygotes (p.Gly1007Arg + LOF) → intermediate severity"
        ),
        "interferonopathy_category": "AGS type 6 (ADAR1; A-to-I RNA editing; MDA5-driven IFN; bilateral striatal necrosis distinctive)",
        "pathognomonic": (
            "BILATERAL STRIATAL NECROSIS on MRI (bilateral caudate nucleus + putamen DWI restriction/T2 hyperintensity) PATHOGNOMONIC FOR ADAR1-AGS — unique among all interferonopathies; "
            "INTRACEREBRAL CALCIFICATIONS (basal ganglia + white matter); "
            "DSH (Dyschromatosis Symmetrica Hereditaria): mixed hypo- and hyper-pigmented macules on extremities (PATHOGNOMONIC skin phenotype); "
            "ELEVATED ISG SCORE with strong IFN signature; "
            "A-TO-I RNA EDITING: ADAR1 edits endogenous dsRNA → prevents MDA5 (IFIH1) recognition; deficiency → unedited dsRNA → MDA5 activation → IFN"
        ),
        "treatment": (
            "JAK INHIBITORS (baricitinib/ruxolitinib): most evidence in ADAR1-AGS (pathway: MDA5 → MAVS → IRF3 → IFN → JAK-STAT → ISG); "
            "ACUTE STRIATAL NECROSIS: steroids considered acutely (no RCT); "
            "ANTICONVULSANTS: epilepsy management; "
            "DSH SKIN: no specific treatment; sun protection; cosmetic camouflage; "
            "PHYSIOTHERAPY: spasticity + movement disorder management; "
            "DYSTONIA: trihexyphenidyl / tetrabenazine for extrapyramidal features; "
            "PROGNOSIS ADAR1-AGS biallelic: severe; bilateral striatal necrosis → fixed neurological deficit; "
            "PROGNOSIS DSH/AD p.Gly1007Arg: skin phenotype ± mild systemic features; survival to adulthood"
        ),
        "key_features": [
            "BILATERAL STRIATAL NECROSIS (bilateral caudate + putamen DWI restriction on MRI) PATHOGNOMONIC FOR ADAR1-AGS — unique among all interferonopathies",
            "A-TO-I RNA EDITING MECHANISM: ADAR1 converts adenosine → inosine in dsRNA; prevents endogenous dsRNA from activating MDA5 (IFIH1); ADAR1 deficiency → unedited dsRNA → MDA5-MAVS-IRF3 → IFN-β cascade",
            "DSH (Dyschromatosis Symmetrica Hereditaria): mixed hypo- and hyper-pigmented macules on dorsa of hands/feet — AD heterozygous p.Gly1007Arg pathognomonic skin phenotype",
            "TWO ISOFORMS: p150 (cytoplasmic, IFN-inducible, Z-alpha domain for Z-form dsRNA) and p110 (nuclear); both relevant to AGS pathology",
            "MDA5 DOWNSTREAM TARGET: ADAR1 suppresses MDA5 activation; IFIH1 (MDA5) GOF mutations cause AGS7 — both converge on same pathway",
            "AD-AR DUALITY: p.Gly1007Arg AD → DSH with/without mild AGS; biallelic LOF → severe AGS6 with striatal necrosis — same gene, drastically different phenotypes",
        ],
        "monitoring": [
            "ISG score at diagnosis + 3-monthly on JAK inhibitor (MDA5-pathway score should normalise)",
            "MRI brain at diagnosis + 6-monthly (striatal lesion evolution — may stabilise or progress)",
            "Skin examination at each visit: DSH macule distribution + photo-documentation",
            "Movement disorder assessment (dystonia scale) + neurological exam quarterly",
            "EEG annually + seizure diary",
            "Ophthalmology annually (optic atrophy described in biallelic severe cases)",
        ],
        "key_ddx": [
            "Leigh syndrome (bilateral striatal necrosis on MRI BUT: elevated lactate, mitochondrial enzyme deficiency, no IFN signature, TREC normal, different mutations)",
            "Wilson disease (basal ganglia T2 hyperintensity BUT: liver disease, Kayser-Fleischer rings, low ceruloplasmin, copper on MRI)",
            "TREX1-AGS1 (most common AGS; no striatal necrosis pattern; cGAS-STING not MDA5 pathway)",
            "IFIH1-AGS7 (same pathway — ADAR1 suppresses MDA5; IFIH1 GOF activates MDA5; distinguish by gene panel)",
        ],
        "ifn_pathway": "MDA5 (IFIH1) / RNA editing — A-to-I editing failure → unedited dsRNA → MDA5-MAVS-IRF3 cascade",
        "onset_age": "infant_to_childhood",
        "cerebrovascular_risk": True,
        "skin_phenotype": True,
        "lung_disease": False,
        "fever_episodes": True,
        "scid_type": False,
        "severity_options": [
            "Severe AR-AGS6 (biallelic LOF; bilateral striatal necrosis; severe disability)",
            "Moderate AD-AGS6 (compound het p.Gly1007Arg + LOF; encephalopathy ± DSH)",
            "Mild DSH (heterozygous p.Gly1007Arg; skin phenotype; no/mild encephalopathy)",
        ],
        "complication_options": ["Bilateral striatal necrosis", "Dystonia + movement disorder", "DSH skin lesions", "Seizures", "Spastic quadriplegia", "Progressive encephalopathy"],
        "isg_level_options": ["markedly_elevated", "elevated", "mildly_elevated"],
        "treatments_used": ["Baricitinib (JAK inhibitor)", "Trihexyphenidyl (dystonia)", "Anticonvulsants", "Physiotherapy", "Sun protection (DSH)", "Acute steroids (striatal necrosis)"],
    },
    # -- IFIH1/MDA5 — AGS7 ------------------------------------------------------------
    {
        "gene": "IFIH1",
        "alt_name": (
            "IFIH1 (IFIH1-1025aa-2q24.2 / AD-GOF — AGS7 — "
            "MDA5-Cytosolic-RNA-Helicase-Constitutive-Activation-GOF-Drives-IFN — "
            "MILDER-AGS-Phenotype-Singleton-De-Novo-Dominant — "
            "Paradox-IFIH1-LOF-Protects-Against-Type-1-Diabetes)"
        ),
        "protein": (
            "IFIH1 -- 2q24.2 AD-GOF -- IFIH1-1025aa -- "
            "Interferon-Induced-With-Helicase-C-Domain-1-MDA5-Melanoma-Differentiation-Associated-5-117kDa -- "
            "Cytosolic-RNA-Helicase-Recognises-Long-dsRNA-Viral-RNA-IRES-Sequences -- "
            "GOF-Mutations-Lower-Threshold-dsRNA-Sensing-Constitutive-MDA5-Activation-Without-Virus -- "
            "MDA5-MAVS-TBK1-IRF3-IRF7-Type-I-IFN-Production-Cascade -- "
            "ADAR1-p150-Suppresses-MDA5-Activation-By-Editing-Endogenous-dsRNA -- "
            "IFIH1-LOF-PROTECTIVE-Against-Type-1-Diabetes-T1DM-GWAS-Signal-2q24 -- "
            "MILDER-AGS-PHENOTYPE-Than-TREX1-RNASEH2A-Neonatal-SCID-Not-Typically-Seen -- "
            "SINGLETON-DE-NOVO-AD-GOF-Mutations-Common-Presentation -- "
            "OMIM-Gene-606951-Disease-AGS7-615846"
        ),
        "locus": "2q24.2",
        "protein_size": "1025 aa / 117 kDa",
        "inheritance": (
            "AD (GOF); gain-of-function mutations lower threshold for dsRNA sensing → constitutive MDA5 activation; "
            "singleton de novo mutations common (often not inherited from affected parent); "
            "autosomal dominant with variable penetrance; "
            "PARADOX: IFIH1 LOF (loss-of-function) common variants PROTECT against Type 1 Diabetes — "
            "reduced MDA5 sensing of enteroviral dsRNA → less beta-cell destruction; "
            "GOF (opposite) → constitutive IFN → AGS7; "
            "no known founder mutation"
        ),
        "interferonopathy_category": "AGS type 7 (IFIH1/MDA5; cytosolic RNA helicase GOF; milder AGS; de novo dominant)",
        "pathognomonic": (
            "MILDER AGS PHENOTYPE — less severe neurological involvement than TREX1/RNASEH2A/SAMHD1; "
            "ELEVATED ISG SCORE (constitutive MDA5 activation → persistent IFN production); "
            "INTRACEREBRAL CALCIFICATIONS may be subtle or absent in milder cases; "
            "SINGLETON DE NOVO AD-GOF MUTATIONS — often no family history; "
            "SINGLE GENE DRIVES CONSTITUTIVE IFN via MDA5-MAVS cascade without viral trigger"
        ),
        "treatment": (
            "JAK INHIBITORS (baricitinib/ruxolitinib): IFIH1-AGS7 responds well (JAK1/2 block downstream IFN signalling); "
            "CLINICAL TRIAL DATA: baricitinib shown effective in IFIH1-AGS7 (TOFA-AGS trial); "
            "SUPPORTIVE CARE: physiotherapy + anticonvulsants + feeding support; "
            "PROGNOSIS BETTER THAN TREX1/RNASEH2A: many patients retain communication + mobility; "
            "HYDROXYCHLOROQUINE: if lupus features emerge; "
            "ANNUAL REVIEW: ISG score + MRI brain + neurodevelopmental assessment; "
            "GENETIC COUNSELLING: AD GOF — 50% transmission risk if parent affected; de novo — very low recurrence"
        ),
        "key_features": [
            "MDA5 CONSTITUTIVE ACTIVATION (GOF): IFIH1 mutations lower sensing threshold → MDA5 activated by self-RNA without viral trigger → MAVS → TBK1 → IRF3/IRF7 → IFN-β",
            "MILDER AGS PHENOTYPE: less severe than TREX1/RNASEH2A/ADAR1-biallelic; many children maintain communication; calcifications may be subtle",
            "SINGLETON DE NOVO DOMINANT: AD-GOF mutations often appear de novo — no family history; parents typically unaffected; sporadic presentation",
            "IFIH1 LOF PARADOX: common LOF variants (rs1990760) PROTECT against Type 1 Diabetes by reducing MDA5-mediated beta-cell inflammation — opposite of GOF (AGS7)",
            "ADAR1 COUNTER-REGULATORY: ADAR1 edits endogenous dsRNA to prevent MDA5 activation; ADAR1 deficiency (AGS6) and IFIH1 GOF (AGS7) both activate MDA5 — converge on same pathway from opposite directions",
            "BEST RESPONSE TO JAK INHIBITORS among AGS subtypes: baricitinib clinical trial evidence specifically for IFIH1-AGS7 (direct pathway inhibition downstream of MDA5)",
        ],
        "monitoring": [
            "ISG score at diagnosis + 3-monthly on baricitinib (best JAK inhibitor evidence in IFIH1-AGS7)",
            "MRI brain annually (calcification progression; milder expected vs TREX1)",
            "Neurodevelopmental assessment: language (often preserved), cognition, motor function — 6-monthly",
            "EEG annually + seizure diary (epilepsy less common in AGS7 vs AGS1)",
            "CBC + LFTs on baricitinib quarterly (standard JAK inhibitor monitoring)",
            "Annual autoimmune panel: ANA, dsDNA (IFIH1 GOF increases lupus risk)",
        ],
        "key_ddx": [
            "ADAR1-AGS6 (MDA5 pathway also; ADAR1 regulates MDA5; distinguish: bilateral striatal necrosis unique to ADAR1; DSH skin; gene panel)",
            "TREX1-AGS1 (most common AGS; cGAS-STING not MDA5 pathway; FCL; more severe typical course)",
            "RNASEH2B-AGS3 (mildest AGS; spastic paraplegia; RNase H2 ribonucleotide pathway; different gene panel)",
            "SAVI-STING1 (vasculitis + ILD dominant; no classical AGS calcifications; different IFN-pathway branch)",
        ],
        "ifn_pathway": "MDA5 (IFIH1) / cytosolic RNA sensing — constitutive GOF MDA5-MAVS-TBK1-IRF3 cascade",
        "onset_age": "infant_to_childhood",
        "cerebrovascular_risk": False,
        "skin_phenotype": False,
        "lung_disease": False,
        "fever_episodes": True,
        "scid_type": False,
        "severity_options": [
            "Moderate-mild (de novo GOF; preserved cognition; calcifications variable; AGS milder end)",
            "Moderate (recurrent GOF with higher penetrance; more prominent calcifications)",
        ],
        "complication_options": ["Progressive spastic diplegia", "Mild intellectual disability", "Seizures (less common)", "ISG-driven lupus-like features", "Subtle white matter changes", "Language delay"],
        "isg_level_options": ["elevated", "markedly_elevated", "mildly_elevated"],
        "treatments_used": ["Baricitinib (best AGS7 evidence)", "Ruxolitinib", "Physiotherapy", "Anticonvulsants (if needed)", "Hydroxychloroquine (if lupus features)", "Speech therapy"],
    },
    # -- STING1/TMEM173 — SAVI --------------------------------------------------------
    {
        "gene": "STING1",
        "alt_name": (
            "STING1 (STING1-379aa-5q31.2 / AD-GOF — SAVI — "
            "CUTANEOUS-VASCULITIS-NECROTIC-ULCERS-NOSE-EARS-EXTREMITIES-PATHOGNOMONIC — "
            "INTERSTITIAL-LUNG-DISEASE-ILD-Major-Mortality-Cause — "
            "JAK-Inhibitors-Ruxolitinib-Baricitinib-HIGHLY-EFFECTIVE)"
        ),
        "protein": (
            "STING1 -- 5q31.2 AD-GOF -- STING1-379aa -- "
            "Stimulator-Of-Interferon-Genes-TMEM173-42kDa-ER-Transmembrane-Protein-cGAMP-Sensor -- "
            "cGAS-Produces-2-3-cGAMP-From-Cytosolic-dsDNA-STING1-Binds-cGAMP-As-Second-Messenger -- "
            "STING1-GOF-Constitutive-Activation-Without-cGAMP-Binding-Constitutive-IFN -- "
            "ER-GOLGI-STING1-Palmitoylated-Traffics-To-Golgi-Activates-TBK1-IRF3-IFN -- "
            "SAVI-STING-Associated-Vasculopathy-Onset-Infancy-Distinct-From-Classical-AGS -- "
            "CUTANEOUS-VASCULITIS-Necrotic-Ulcers-Nose-Ears-Cheeks-Digits-PATHOGNOMONIC -- "
            "INTERSTITIAL-LUNG-DISEASE-ILD-Progressive-Major-Mortality-Cause-SAVI -- "
            "JAK-INHIBITORS-Ruxolitinib-Baricitinib-HIGHLY-EFFECTIVE-Cutaneous-ILD-Improvement -- "
            "OMIM-Gene-612374-Disease-SAVI-615934"
        ),
        "locus": "5q31.2",
        "protein_size": "379 aa / 42 kDa",
        "inheritance": (
            "AD (GOF); gain-of-function mutations constitutively activate STING1 without cGAMP; "
            "most common mutations: p.Val155Met, p.Asn154Ser, p.Val147Leu, p.Arg284Ser; "
            "de novo mutations very common (sporadic SAVI); "
            "STING1: ER transmembrane cGAMP sensor — cGAS produces 2'3'-cGAMP from cytosolic dsDNA → "
            "STING1 binds cGAMP → palmitoylation → ER-to-Golgi trafficking → TBK1 → IRF3 → IFN-β; "
            "GOF mutations bypass cGAMP requirement → constitutive activation"
        ),
        "interferonopathy_category": "SAVI (STING1/TMEM173; cGAS-STING pathway GOF; vasculitis + ILD phenotype — DISTINCT from classical AGS)",
        "pathognomonic": (
            "CUTANEOUS VASCULITIS with NECROTIC ULCERS — nose, ears, cheeks, digits, lower extremities PATHOGNOMONIC for SAVI; "
            "INTERSTITIAL LUNG DISEASE (ILD) — fibrotic/inflammatory; main cause of mortality; "
            "onset in INFANCY (within first year); "
            "ELEVATED ISG SCORE (highest among STING pathway disorders); "
            "NO CLASSICAL AGS ENCEPHALOPATHY/CALCIFICATIONS — DISTINCT PHENOTYPE from TREX1/RNASEH2/SAMHD1/ADAR1/IFIH1"
        ),
        "treatment": (
            "JAK INHIBITORS — ruxolitinib (Jakafi) or baricitinib (Olumiant) HIGHLY EFFECTIVE: "
            "most dramatic clinical response among all interferonopathies; "
            "cutaneous vasculitis lesions heal + ILD stabilises on JAK inhibition; "
            "RUXOLITINIB (JAK1/JAK2): published case series showing healing of necrotic lesions; "
            "BARICITINIB (JAK1/JAK2): also effective; FDA-approved for other IFN-driven disorders; "
            "WOUND CARE: necrotic ulcer management — debridement + antimicrobial dressings + "
            "vascular surgery consultation for digital ischaemia; "
            "PULMONOLOGY JOINT CARE: ILD monitoring + high-resolution CT chest annually; "
            "CORTICOSTEROIDS: partial benefit only; do not use long-term without JAK inhibitor; "
            "MYCOPHENOLATE MOFETIL: second-line immunosuppression if JAK inhibitor insufficient; "
            "LUNG TRANSPLANT: considered in end-stage ILD non-responsive to immunosuppression"
        ),
        "key_features": [
            "CUTANEOUS VASCULITIS + NECROTIC ULCERS (nose/ears/cheeks/digits) PATHOGNOMONIC — no other interferonopathy presents with this vasculitic skin phenotype",
            "INTERSTITIAL LUNG DISEASE (ILD): fibrotic/inflammatory; progressive; main cause of mortality in SAVI; requires annual HRCT chest",
            "DISTINCT FROM AGS: no classical intracerebral calcifications + encephalopathy; vasculitis + ILD dominant — separates SAVI clinically from all AGS subtypes",
            "cGAS-STING PATHWAY GOF: STING1 constitutively active without cGAMP → palmitoylation → ER-to-Golgi trafficking → TBK1 → IRF3 → IFN-β; highest ISG scores among interferonopathies",
            "JAK INHIBITORS HIGHLY EFFECTIVE: most dramatic treatment response among all interferonopathies — wound healing of necrotic vasculitic ulcers reported on ruxolitinib/baricitinib",
            "ONSET IN INFANCY: vasculitis lesions present in first year of life; early JAK inhibitor initiation prevents ILD progression and digit loss",
        ],
        "monitoring": [
            "ISG score at diagnosis and 3-monthly on JAK inhibitor (target normalisation)",
            "HRCT chest at diagnosis + 6-monthly (ILD progression — main mortality risk)",
            "Pulmonary function tests (spirometry + DLCO) 6-monthly if age-appropriate",
            "Skin photography of vasculitic lesions at every visit; dermatology joint assessment",
            "Vascular surgery referral if digital ischaemia threatens digit viability",
            "Echocardiography annually (pulmonary hypertension as ILD complication)",
        ],
        "key_ddx": [
            "TREX1-AGS1 (cGAS-STING pathway also, but encephalopathy + calcifications; no vasculitic ulcers; no ILD)",
            "Polyarteritis nodosa (PAN) (vasculitis but ANCA-negative; older age; different vessel calibre; no IFN signature)",
            "Granulomatosis with polyangiitis (GPA/Wegener's) (ANCA c-ANCA PR3; granulomas; lung + kidney; no ISG elevation)",
            "Juvenile dermatomyositis (JDM) (muscle weakness + heliotrope; myositis enzymes elevated; no necrotic ulcers on nose/ears)",
        ],
        "ifn_pathway": "cGAS-STING (cytosolic DNA sensing — constitutive GOF palmitoylation → TBK1-IRF3 cascade)",
        "onset_age": "neonatal_to_infant",
        "cerebrovascular_risk": False,
        "skin_phenotype": True,
        "lung_disease": True,
        "fever_episodes": True,
        "scid_type": False,
        "severity_options": [
            "Severe SAVI (early-onset ILD + extensive vasculitis; digit/nasal necrosis; high mortality pre-JAK inhibitor)",
            "Moderate SAVI (later-onset vasculitis ± mild ILD; responds well to JAK inhibition)",
        ],
        "complication_options": ["Progressive ILD", "Necrotic vasculitic ulcers (nose/digits/ears)", "Digit amputation (ischaemia)", "Pulmonary hypertension", "Fever episodes", "Failure to thrive"],
        "isg_level_options": ["markedly_elevated", "elevated"],
        "treatments_used": ["Ruxolitinib (JAK1/2 inhibitor)", "Baricitinib (JAK1/2 inhibitor)", "Wound care (necrotic ulcers)", "Pulmonary function monitoring", "Mycophenolate mofetil (second-line)", "Lung transplant (end-stage ILD)"],
    },
]


# ---------------------------------------------------------------------------
# Patient cohort generator
# ---------------------------------------------------------------------------
def _build_cohort() -> list:
    cohort = []
    for idx, entry in enumerate(IFN_GENES):
        rng = random.Random(SEED_BASE + idx)
        gene = entry["gene"]
        n = 40
        for i in range(n):
            sev = rng.choice(entry["severity_options"])
            comp = rng.sample(entry["complication_options"], k=min(rng.randint(1, 3), len(entry["complication_options"])))
            isg = rng.choice(entry["isg_level_options"])
            treat = rng.sample(entry["treatments_used"], k=min(rng.randint(2, 4), len(entry["treatments_used"])))
            age_yr = round(rng.uniform(0.2, 18.0), 1)
            onset_yr = round(rng.uniform(0.0, min(age_yr, 3.0)), 2)
            delay_mo = max(0, round((age_yr - onset_yr) * 12 - rng.uniform(6, 36), 1))
            isg_score = round(rng.uniform(3.5, 12.0) if isg == "markedly_elevated" else
                              rng.uniform(2.0, 5.0) if isg == "elevated" else
                              rng.uniform(1.5, 2.5), 2)
            patient = {
                "patient_id": f"{gene}-{SEED_BASE + idx}-{i + 1:03d}",
                "gene": gene,
                "age_at_assessment_yr": age_yr,
                "age_at_symptom_onset_yr": onset_yr,
                "diagnosis_delay_months": delay_mo,
                "severity_label": sev.split("(")[0].strip(),
                "complications": comp,
                "isg_level": isg,
                "isg_score_sd_above_mean": isg_score,
                "treatments_used": treat,
                "cerebrovascular_event": entry["cerebrovascular_risk"] and rng.random() < 0.25,
                "skin_phenotype_present": entry["skin_phenotype"] and rng.random() < 0.55,
                "lung_disease_present": entry["lung_disease"] and rng.random() < 0.70,
                "seizures_present": rng.random() < 0.35,
                "ifn_pathway": entry["ifn_pathway"],
            }
            cohort.append(patient)
    return cohort


# ---------------------------------------------------------------------------
# API data generators
# ---------------------------------------------------------------------------
def generate_overview() -> dict:
    cohort = _build_cohort()
    gene_counts = {}
    pathway_counts = {}
    ags_genes = []
    savi_genes = []
    skin_genes = []
    lung_genes = []
    cerebrovascular_genes = []
    gene_summary = {}

    for entry in IFN_GENES:
        g = entry["gene"]
        pts = [p for p in cohort if p["gene"] == g]
        gene_counts[g] = len(pts)

        pathway = entry["ifn_pathway"].split(" /")[0].strip()
        pathway_counts[pathway] = pathway_counts.get(pathway, 0) + len(pts)

        if g != "STING1":
            ags_genes.append(g)
        else:
            savi_genes.append(g)
        if entry["skin_phenotype"]:
            skin_genes.append(g)
        if entry["lung_disease"]:
            lung_genes.append(g)
        if entry["cerebrovascular_risk"]:
            cerebrovascular_genes.append(g)

        gene_summary[g] = {
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "interferonopathy_category": entry["interferonopathy_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "ifn_pathway": entry["ifn_pathway"],
            "skin_phenotype": entry["skin_phenotype"],
            "lung_disease": entry["lung_disease"],
            "cerebrovascular_risk": entry["cerebrovascular_risk"],
            "onset_age": entry["onset_age"],
        }

    return {
        "title": "Hereditary-Type-I-Interferonopathy-Atlas — Complete 8-Gene Hereditary Type I Interferonopathy Atlas",
        "n_genes": len(IFN_GENES),
        "n_patients": len(cohort),
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "interferonopathy_categories": {
            "AGS (Aicardi-Goutières Syndrome)": ["TREX1", "RNASEH2B", "RNASEH2A", "RNASEH2C", "SAMHD1", "ADAR1", "IFIH1"],
            "SAVI (STING-Associated Vasculopathy)": ["STING1"],
        },
        "inheritance_map": {
            "TREX1": "AD/AR", "RNASEH2B": "AR", "RNASEH2A": "AR", "RNASEH2C": "AR",
            "SAMHD1": "AR/FCL-AD", "ADAR1": "AD/AR", "IFIH1": "AD-GOF", "STING1": "AD-GOF",
        },
        "ifn_pathway_map": {
            "TREX1": "cGAS-STING (cytosolic DNA)",
            "RNASEH2B": "RNase H2 / ribonucleotide excision",
            "RNASEH2A": "RNase H2 / ribonucleotide excision",
            "RNASEH2C": "RNase H2 / ribonucleotide excision",
            "SAMHD1": "dNTPase / cGAS-STING (nucleotide metabolism)",
            "ADAR1": "MDA5 (IFIH1) / A-to-I RNA editing failure",
            "IFIH1": "MDA5 (IFIH1) / cytosolic RNA sensing GOF",
            "STING1": "cGAS-STING (constitutive GOF)",
        },
        "key_clinical_pearls": [
            "UNIVERSAL BIOMARKER — ISG SCORE (Interferon-Stimulated Gene Score): all type I interferonopathies share an elevated ISG score in peripheral blood (>2 SD above mean) regardless of gene; this is the single most important diagnostic screening test; CSF IFN-alpha also elevated in AGS acute phase; normalisation of ISG on JAK inhibitor = treatment response marker",
            "INTRACEREBRAL CALCIFICATIONS (AGS1-7): bilateral basal ganglia (putamen > globus pallidus) + periventricular white matter calcifications on plain CT scan PATHOGNOMONIC for AGS — must order CT (not just MRI which underdetects calcifications); SAMHD1 + cerebrovascular disease (ischaemic stroke in childhood) = AGS5 unique; ADAR1 + bilateral striatal necrosis = AGS6 unique; SAVI (STING1) has NO calcifications",
            "JAK INHIBITORS — TREATMENT FOR ALL: baricitinib (Olumiant) or ruxolitinib (Jakafi) target JAK1/JAK2 downstream of IFNAR → block ISG induction; IFIH1-AGS7 has best clinical trial evidence; STING1-SAVI has most dramatic response (necrotic ulcers heal); all AGS subtypes benefit; monitoring: CBC + LFTs quarterly; herpes reactivation risk (aciclovir prophylaxis); live vaccines contraindicated on JAK inhibitors",
            "TREX1 MOST COMMON AGS GENE (~25%): AD = FCL (Familial Chilblain Lupus — cold-triggered acral ulcers) + SLE risk; AR biallelic = severe AGS1; TREX1 degrades cytosolic ssDNA → cGAS-STING activation when absent; RNASEH2B most common AR-AGS in Europe (~35% of AR cases); p.Ala177Thr Northern European founder = mildest AGS phenotype (preserved language/cognition)",
            "ADAR1-AGS6 UNIQUE FEATURE — BILATERAL STRIATAL NECROSIS: MRI shows bilateral caudate + putamen DWI restriction/T2 hyperintensity PATHOGNOMONIC — only interferonopathy with this finding; also DSH (Dyschromatosis Symmetrica Hereditaria) skin phenotype (mixed pigmentation macules, extremities); ADAR1 normally suppresses MDA5 by A-to-I editing of endogenous dsRNA — deficiency mimics IFIH1/MDA5 GOF",
            "SAVI (STING1) DISTINCT FROM AGS: NO calcifications, NO encephalopathy; instead CUTANEOUS VASCULITIS (necrotic ulcers — nose/ears/cheeks/digits) PATHOGNOMONIC + INTERSTITIAL LUNG DISEASE (main mortality cause); onset in infancy; JAK inhibitors MOST EFFECTIVE for SAVI (dramatic wound healing reported); cGAS-STING pathway GOF — STING1 constitutively active without cGAMP; STING1 palmitoylation → ER-to-Golgi trafficking → TBK1 → IRF3 → IFN-β",
            "FCL (FAMILIAL CHILBLAIN LUPUS) — TREX1 AD + SAMHD1 AD: heterozygous TREX1 or SAMHD1 mutations → cold-triggered painful ulcerating acral skin lesions (fingers/toes/ears); often misdiagnosed as chilblains or vasculitis; anti-IFN therapy (baricitinib) + hydroxychloroquine + cold avoidance; FCL may be the only presentation for years before lupus/AGS features emerge; check ISG score in any chilblain patient with family history",
        ],
        "clinical_emergency_flags": [
            "ACUTE NEUROLOGICAL DETERIORATION IN AGS: any AGS patient with new fever + acute neurological worsening (seizures, loss of milestones, acute hemiplegia) → rule out SAMHD1-AGS5 ISCHAEMIC STROKE (urgent CT/MRI + MRA); for all AGS — exclude intercurrent infection triggering IFN flare; consider acute dose of corticosteroids + adjust JAK inhibitor; ICU if airway/respiratory compromise",
            "ISCHAEMIC STROKE IN AGS5 (SAMHD1): childhood ischaemic stroke in a known or suspected interferonopathy patient = AGS5 until proven otherwise; URGENT CT/MRI + MRA; aspirin + antiplatelet immediately; haematology + vascular neurology consult; check coagulation (thrombotic tendency in vasculitis-driven stroke); rehabilitation immediately post-stroke",
            "SAVI RESPIRATORY CRISIS: SAVI patient with acute respiratory deterioration (dyspnoea + hypoxia) → progressive ILD exacerbation or pneumonia superimposed on ILD; HRCT chest urgently; IV methylprednisolone pulse if ILD flare; optimise JAK inhibitor dose; intensive care if respiratory failure; lung transplant team involved early in refractory ILD",
            "DIGITAL ISCHAEMIA IN SAVI: SAVI patient with cold/white/blue digits progressing to demarcation → impending digit loss; vascular surgery + dermatology URGENT; iloprost IV infusion (prostacyclin analogue) for digital vasospasm; JAK inhibitor dose optimisation; wound care specialist; cold avoidance; amputations can be avoided with early effective JAK inhibitor therapy",
            "JAK INHIBITOR INFECTION RISK: all interferonopathy patients on baricitinib/ruxolitinib have increased infection risk (JAK2 inhibition → impaired cytokine signalling); serious bacterial infections (pneumonia, sepsis) + herpes reactivation (VZV dermatomal zoster); prophylactic aciclovir (for VZV) and TMP-SMX (for PCP) in high-dose JAK inhibitor patients; hold JAK inhibitor during severe infections; do not give live vaccines",
        ],
        "gene_summary": gene_summary,
        "ags_genes": ags_genes,
        "savi_genes": savi_genes,
        "skin_phenotype_genes": skin_genes,
        "lung_disease_genes": lung_genes,
        "cerebrovascular_risk_genes": cerebrovascular_genes,
        "diagnostic_algorithm": [
            "Step 1 — Suspect Type I Interferonopathy: infant/child with unexplained progressive encephalopathy + calcifications (AGS) OR neonatal vasculitis + ILD (SAVI) OR chilblain lupus → measure ISG score (6-gene or extended panel in peripheral blood)",
            "Step 2 — ISG score elevated (>2 SD): confirms interferonopathy; order plain CT brain (calcifications — basal ganglia + white matter); MRI brain (white matter, striatal necrosis); CSF IFN-alpha if LP available; autoimmune panel (ANA/dsDNA — FCL/lupus overlap)",
            "Step 3 — Classify phenotype: AGS (calcifications + encephalopathy) vs SAVI (vasculitis + ILD, no calcifications); within AGS: most severe (TREX1/RNASEH2A/RNASEH2C/SAMHD1) vs mildest (RNASEH2B — spastic paraplegia preserved cognition); distinctive features: bilateral striatal necrosis = ADAR1; cerebrovascular disease = SAMHD1; FCL skin = TREX1/SAMHD1; DSH skin = ADAR1",
            "Step 4 — Molecular confirmation: targeted AGS 7-gene panel (TREX1/RNASEH2A/B/C/SAMHD1/ADAR1/IFIH1) + STING1 for SAVI; WES if panel negative + strong clinical suspicion; prenatal diagnosis available once index case identified",
            "Step 5 — Initiate JAK inhibitor therapy: baricitinib or ruxolitinib; start once interferonopathy confirmed (gene result not needed before treatment); monitor ISG score + clinical response; SAVI = most dramatic response (ulcer healing); IFIH1-AGS7 = clinical trial-proven response; all AGS subtypes benefit",
            "Step 6 — Supportive multidisciplinary care: neurology + physiotherapy + speech therapy + gastrostomy (if dysphagia); pulmonology (SAVI-ILD); dermatology (FCL/vasculitis); ophthalmology; haematology (FCL/lupus thrombocytopenia); vascular surgery (SAVI digits); regular ISG + MRI surveillance",
        ],
        "pathway_distribution": dict(sorted(pathway_counts.items(), key=lambda x: -x[1])),
        "gene_counts": gene_counts,
    }


def generate_breakdown() -> dict:
    cohort = _build_cohort()
    gene_breakdown = {}
    for entry in IFN_GENES:
        g = entry["gene"]
        pts = [p for p in cohort if p["gene"] == g]

        complication_counts = {}
        for p in pts:
            for c in p["complications"]:
                complication_counts[c] = complication_counts.get(c, 0) + 1

        isg_distribution = {}
        for p in pts:
            isg_distribution[p["isg_level"]] = isg_distribution.get(p["isg_level"], 0) + 1

        treatment_counts = {}
        for p in pts:
            for t in p["treatments_used"]:
                treatment_counts[t] = treatment_counts.get(t, 0) + 1

        sev_counts = {}
        for p in pts:
            sev_counts[p["severity_label"]] = sev_counts.get(p["severity_label"], 0) + 1

        avg_delay = round(sum(p["diagnosis_delay_months"] for p in pts) / len(pts), 1) if pts else 0
        avg_isg = round(sum(p["isg_score_sd_above_mean"] for p in pts) / len(pts), 2) if pts else 0

        gene_breakdown[g] = {
            "gene": g,
            "n_patients": len(pts),
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0],
            "interferonopathy_category": entry["interferonopathy_category"],
            "ifn_pathway": entry["ifn_pathway"],
            "pathognomonic": entry["pathognomonic"][:200],
            "onset_age": entry["onset_age"],
            "complication_distribution": dict(sorted(complication_counts.items(), key=lambda x: -x[1])),
            "isg_distribution": isg_distribution,
            "treatment_distribution": dict(sorted(treatment_counts.items(), key=lambda x: -x[1])[:6]),
            "severity_distribution": sev_counts,
            "avg_diagnosis_delay_months": avg_delay,
            "avg_isg_score_sd": avg_isg,
            "pct_seizures": round(sum(1 for p in pts if p["seizures_present"]) / len(pts) * 100, 1) if pts else 0,
            "pct_cerebrovascular": round(sum(1 for p in pts if p["cerebrovascular_event"]) / len(pts) * 100, 1) if pts else 0,
            "pct_skin_phenotype": round(sum(1 for p in pts if p["skin_phenotype_present"]) / len(pts) * 100, 1) if pts else 0,
            "pct_lung_disease": round(sum(1 for p in pts if p["lung_disease_present"]) / len(pts) * 100, 1) if pts else 0,
            "cerebrovascular_risk": entry["cerebrovascular_risk"],
            "skin_phenotype": entry["skin_phenotype"],
            "lung_disease": entry["lung_disease"],
        }
    return {"gene_breakdown": gene_breakdown, "n_genes": len(IFN_GENES), "n_patients": len(cohort)}


def generate_definitions() -> dict:
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["interferonopathy_category"],
                "ifn_pathway": entry["ifn_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:400],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "skin_phenotype": entry["skin_phenotype"],
                "lung_disease": entry["lung_disease"],
                "cerebrovascular_risk": entry["cerebrovascular_risk"],
                "onset_age": entry["onset_age"],
            }
            for entry in IFN_GENES
        },
        "interferonopathy_glossary": {
            "Type I Interferonopathy": (
                "A group of Mendelian disorders characterised by constitutive upregulation of type I interferon "
                "(IFN-alpha/beta) signalling in the absence of infection. Caused by mutations in genes encoding "
                "nucleic acid sensing/processing pathways (cGAS-STING, RNase H2, MDA5/IFIH1, ADAR1). "
                "Unified by an elevated ISG (interferon-stimulated gene) score in peripheral blood. "
                "Clinically heterogeneous: AGS (Aicardi-Goutières syndrome — brain), FCL (skin), SAVI (vessels + lung). "
                "JAK inhibitors (baricitinib, ruxolitinib) are the emerging treatment of choice."
            ),
            "ISG Score (Interferon-Stimulated Gene Score)": (
                "Quantitative measure of type I interferon activity in peripheral blood. "
                "Measured by RT-qPCR of 6 or more interferon-stimulated genes (IFIT1, IFIT2, IFI44L, IFI27, RSAD2, SIGLEC1). "
                "Result expressed as standard deviations (SD) above healthy control mean. "
                ">2 SD = positive IFN signature (abnormal). "
                "DIAGNOSTIC: all type I interferonopathies have elevated ISG score. "
                "MONITORING: ISG score normalisation on JAK inhibitor = treatment response. "
                "NOT ELEVATED in immunodeficiencies (SCID, XLA, APECED) — discriminates interferonopathies from PIDs."
            ),
            "Aicardi-Goutières Syndrome (AGS)": (
                "Mendelian type I interferonopathy with progressive encephalopathy, intracerebral calcifications "
                "(basal ganglia + white matter), and elevated ISG score. "
                "7 causative genes: TREX1 (AGS1), RNASEH2C (AGS2), RNASEH2B (AGS3), RNASEH2A (AGS4), "
                "SAMHD1 (AGS5), ADAR1 (AGS6), IFIH1 (AGS7). "
                "Clinical: onset in infancy, fever episodes, progressive spastic quadriplegia, intellectual disability. "
                "MIMICS CONGENITAL VIRAL INFECTION (hence 'pseudo-TORCH' syndrome). "
                "Diagnosis: CT calcifications + elevated ISG score + gene panel. "
                "Treatment: JAK inhibitors (baricitinib/ruxolitinib) — stabilise not cure."
            ),
            "SAVI (STING-Associated Vasculopathy with Infancy Onset)": (
                "Type I interferonopathy caused by STING1 (TMEM173) AD gain-of-function mutations. "
                "DISTINCT from AGS: vasculitis + ILD, NOT encephalopathy + calcifications. "
                "Clinical: early-onset cutaneous vasculitis (necrotic ulcers — nose, ears, digits), "
                "interstitial lung disease (main mortality cause), elevated ISG score. "
                "STING1 GOF: constitutive ER-to-Golgi trafficking + TBK1-IRF3-IFN-β cascade without cGAMP. "
                "Most dramatic JAK inhibitor response: ruxolitinib/baricitinib heals vasculitic ulcers + stabilises ILD. "
                "Early diagnosis and JAK inhibitor initiation prevents digit loss and ILD progression."
            ),
            "cGAS-STING Pathway": (
                "Cytosolic DNA sensing pathway. "
                "cGAS (cyclic GMP-AMP synthase): recognises cytosolic dsDNA → produces 2'3'-cGAMP. "
                "STING1 (TMEM173): binds cGAMP as second messenger → palmitoylated → ER-to-Golgi trafficking → "
                "TBK1 phosphorylation → IRF3 phosphorylation/dimerisation → nucleus → IFN-β transcription. "
                "AGS RELEVANCE: TREX1 (degrades cytosolic ssDNA — prevents cGAS activation) + SAMHD1 (dNTPase) "
                "act upstream; STING1 GOF acts at the sensor itself. "
                "THERAPEUTIC TARGET: baricitinib/ruxolitinib block JAK1/2 downstream of IFNAR; "
                "direct STING1 inhibitors (H-151, C-176) in preclinical/early clinical development."
            ),
            "RNase H2 Complex": (
                "Heterotrimeric nuclear enzyme: RNASEH2A (catalytic) + RNASEH2B (structural, PCNA-binding) + RNASEH2C (structural). "
                "Function: cleaves RNA strand of RNA-DNA hybrids; removes single ribonucleotides mis-incorporated into genomic DNA "
                "(ribonucleotide excision repair); resolves R-loops at transcription-replication conflicts. "
                "~1 ribonucleotide mis-incorporated per 7,600 base pairs in normal replication. "
                "RNASEH2 DEFICIENCY: ribonucleotide accumulation → genome instability + R-loops → "
                "cytosolic nucleic acid sensing (cGAS-independent pathway) → type I IFN. "
                "Severity hierarchy: RNASEH2A = RNASEH2C > RNASEH2B (p.Ala177Thr hypomorphic = mildest AGS)."
            ),
            "FCL (Familial Chilblain Lupus)": (
                "AD heterozygous mutations in TREX1 or SAMHD1 cause Familial Chilblain Lupus (FCL). "
                "Clinical: recurrent cold-triggered painful erythematous/ulcerating lesions on acral sites "
                "(fingers, toes, ears, nose) — identical to idiopathic chilblains/pernio but with autoimmune basis. "
                "PATHOGNOMONIC: lesions appear/worsen on cold exposure; heal in warm conditions. "
                "Associated features: elevated ISG score; anti-nuclear antibodies; risk of lupus nephritis. "
                "TREATMENT: cold avoidance + hydroxychloroquine + baricitinib (most effective); "
                "NOT simple chilblains — test ISG score + TREX1/SAMHD1 sequencing in familial or early-onset cases."
            ),
            "JAK Inhibitors in Interferonopathies": (
                "JAK1/JAK2 inhibitors (baricitinib = Olumiant, ruxolitinib = Jakafi) block JAK-STAT pathway "
                "downstream of type I IFN receptor (IFNAR1/IFNAR2). "
                "Mechanism: IFN binds IFNAR → JAK1/TYK2 phosphorylation → STAT1/STAT2 → ISG transcription; "
                "baricitinib/ruxolitinib block JAK1/JAK2 → prevent STAT phosphorylation → reduce ISG score. "
                "EVIDENCE: baricitinib clinical trial in IFIH1-AGS7 (primary endpoint: ISG normalisation); "
                "SAVI case series: dramatic ruxolitinib response (ulcer healing, ILD stabilisation). "
                "MONITORING: CBC + LFTs quarterly; herpes reactivation (aciclovir prophylaxis); "
                "NO live vaccines on JAK inhibitors; TB screening before initiation."
            ),
        },
        "treatment_glossary": {
            "Baricitinib (Olumiant)": (
                "Selective JAK1/JAK2 inhibitor (Eli Lilly; Incyte). FDA approved for rheumatoid arthritis, "
                "alopecia areata, and COVID-19 severe disease. Off-label (expanding) use in all type I interferonopathies. "
                "DOSING: 2-4mg once daily oral (adult RA dosing); interferonopathy dosing being established in trials. "
                "MECHANISM: JAK1/JAK2 blockade → prevents IFN-STAT1/STAT2 signalling → ISG score reduction. "
                "EVIDENCE: clinical trial-proven in IFIH1-AGS7; case series evidence in all AGS subtypes + SAVI. "
                "KEY SIDE EFFECTS: infections (particularly herpes zoster reactivation); cytopenias; hepatotoxicity; "
                "lipid elevation; DVT/PE risk (lower with baricitinib vs tofacitinib). "
                "MONITORING: CBC, LFTs, lipid profile, creatinine quarterly; VZV prophylaxis (aciclovir 400mg BD)."
            ),
            "Ruxolitinib (Jakafi)": (
                "JAK1/JAK2 inhibitor (Incyte/Novartis). FDA approved for myeloproliferative disorders (PV, MF) "
                "and graft-versus-host disease (GVHD). Off-label for interferonopathies including SAVI. "
                "SAVI EVIDENCE: most dramatic response; published cases of complete healing of necrotic vasculitic "
                "ulcers (nose, ears, digits) and stabilisation/improvement of ILD within weeks of starting. "
                "DOSING: 5-20mg twice daily (adult myeloproliferative dosing); interferonopathy dosing empirical. "
                "MECHANISM: same as baricitinib (JAK1/JAK2) but slight pharmacological differences. "
                "COMBINATION: may be used with mycophenolate mofetil for severe SAVI with ILD."
            ),
            "Hydroxychloroquine": (
                "Antimalarial / immunomodulatory drug used in FCL (Familial Chilblain Lupus) and lupus overlap in AGS. "
                "MECHANISM: lysosome alkalinisation → inhibits endosomal TLR signalling (TLR7/9) → reduces IFN-alpha "
                "production from plasmacytoid dendritic cells; also anti-inflammatory. "
                "DOSING: 5mg/kg/day oral (max 400mg/day adult; <6.5mg/kg/day ideal body weight to avoid retinopathy). "
                "FCL USE: reduces frequency + severity of chilblain episodes; combined with cold avoidance. "
                "MONITORING: ophthalmology annually (hydroxychloroquine retinopathy screening from 5yr of use)."
            ),
        },
        "diagnostic_tests": {
            "ISG Score (Interferon-Stimulated Gene Score)": (
                "The single most important screening test for all type I interferonopathies. "
                "Method: RT-qPCR of interferon-stimulated genes (IFI27, IFI44L, IFIT1, IFIT2, RSAD2, SIGLEC1) "
                "in peripheral blood; score expressed as SD above healthy control median. "
                "RESULT >2 SD = elevated ISG score = positive IFN signature. "
                "SENSITIVITY: ~95% for AGS; ~100% for SAVI; ~60-70% for FCL (intermittent IFN elevation). "
                "FALSE NEGATIVE: measure during active disease period (not during remission on JAK inhibitor); "
                "collect sample in the morning (diurnal variation); acute infections transiently elevate ISG. "
                "MONITORING USE: 3-monthly on JAK inhibitor to assess treatment response."
            ),
            "CT Brain (Intracerebral Calcification Detection)": (
                "Non-contrast CT brain is MANDATORY in suspected AGS — MRI is INSUFFICIENT for calcification detection. "
                "Calcifications appear as hyperdense foci on CT in: basal ganglia (putamen > globus pallidus), "
                "white matter (periventricular), cerebellum (rare). "
                "MRI often misses fine calcifications that CT detects clearly. "
                "INTERPRETATION: bilateral symmetrical basal ganglia + periventricular white matter calcifications "
                "in an infant with progressive encephalopathy = AGS until proven otherwise. "
                "DIFFERENTIAL: congenital CMV, congenital toxoplasma, Krabbe, GM2 gangliosidosis — exclude by workup."
            ),
            "HRCT Chest (SAVI-ILD Monitoring)": (
                "High-Resolution CT (HRCT) chest is mandatory at SAVI diagnosis and 6-monthly thereafter. "
                "ILD patterns in SAVI: ground-glass opacity, reticulation, bronchiectasis, traction bronchiectasis. "
                "ILD SEVERITY SCORING: Fleischner Society criteria; extent of involvement (% lung field); "
                "honeycomb change = advanced fibrosis. "
                "CORRELATION WITH OUTCOME: rate of ILD progression predicts mortality; "
                "JAK inhibitor response: stabilisation (no progression) = treatment success; improvement possible. "
                "COMPANION TESTS: pulmonary function tests (spirometry FVC, FEV1, DLCO) 6-monthly; "
                "bronchoscopy with BAL if diagnostic uncertainty or infection excluded."
            ),
            "AGS Gene Panel (TREX1/RNASEH2A/B/C/SAMHD1/ADAR1/IFIH1 + STING1)": (
                "Targeted next-generation sequencing (NGS) panel covering all 7 AGS genes + STING1 for SAVI. "
                "COVERAGE: coding regions + exon-intron boundaries of all 8 genes; deletion/duplication analysis "
                "(MLPA or CNV from NGS) for RNASEH2B (deletion common in p.Ala177Thr deletion carriers). "
                "INTERPRETATION: biallelic variants for AR genes (RNASEH2A/B/C, SAMHD1); "
                "heterozygous LOF or GOF for AD genes (TREX1-FCL/AD-AGS1, ADAR1-DSH, IFIH1-AGS7, STING1-SAVI); "
                "pathogenicity classified per ACMG/AMP 2015 guidelines. "
                "If panel NEGATIVE but clinical suspicion high: WES; then consider novel interferonopathy genes."
            ),
        },
    }


# Aliases for api_backend.py compatibility
def overview() -> dict:
    return generate_overview()


def breakdown() -> dict:
    return generate_breakdown()


def definitions() -> dict:
    return generate_definitions()
