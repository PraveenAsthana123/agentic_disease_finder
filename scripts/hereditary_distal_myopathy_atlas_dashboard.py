#!/usr/bin/env python3
"""Hereditary-Distal-Myopathy-Atlas — Complete 8-Gene Distal Myopathy Spectrum Atlas
(DYSF · GNE · MYH7 · TTN · VCP · MATR3 · HNRNPA2B1 · LDB3/ZASP).

DYSF    (Dysferlin; 2080 aa; 2p13.2; AR;
         LGMD-R2 / Miyoshi Myopathy 1 — most common AR distal/LGMD myopathy;
         POSTERIOR CALF + TIBIALIS ANTERIOR BOTH — Miyoshi (posterior) vs LGMD (proximal) allelic;
         CK 10-150× ULN PATHOGNOMONIC; dysferlin membrane repair protein;
         seed SEED_BASE+0).
GNE     (GNE myopathy / Nonaka / IBM2; 722 aa; 9p13.3; AR;
         QUADRICEPS SPARED LATE PATHOGNOMONIC — late-stage only;
         rimmed vacuoles + TDP-43 / p62 inclusions on biopsy;
         M712T Japanese/Korean founder; V572L Middle Eastern founder;
         seed SEED_BASE+1).
MYH7    (Beta-myosin heavy chain; 1935 aa; 14q11.2; AD;
         Laing distal myopathy — CONGENITAL FOOT DROP PATHOGNOMONIC;
         tibialis anterior > finger extensors; onset infancy-childhood;
         NO cardiac in distal phenotype (unlike HCM alleles K207Q etc.);
         seed SEED_BASE+2).
TTN     (Titin; 34350 aa; 2q31.2; AR/AD;
         Udd/tibial muscular dystrophy (AR-hom) — ANTERIOR TIBIAL ONSET;
         Finnish founder IVS340-1G>A FINmaj; late adult onset >35yr;
         AD heterozygous truncating = titinopathy DCM overlap;
         seed SEED_BASE+3).
VCP     (Valosin-containing protein; 806 aa; 17p13.3; AD;
         IBMPFD — MYOPATHY+PAGET+FTD TRIAD PATHOGNOMONIC;
         R155H most common (50% of cases); ubiquitin-rimmed vacuoles;
         multisystem proteinopathy MSP1; amyloid + TDP-43;
         seed SEED_BASE+4).
MATR3   (Matrin-3; 847 aa; 5q31.2; AD;
         VCPDM — VOCAL CORD + PHARYNGEAL + DISTAL LIMB triad PATHOGNOMONIC;
         hearing loss in many; S85C founder;
         also causes ALS-spectrum (MATR3-ALS); rimmed vacuoles;
         seed SEED_BASE+5).
HNRNPA2B1 (hnRNP A2/B1; 353 aa; 7p15.2; AD;
         Multisystem Proteinopathy MSP2 — D290V founder;
         IBM-like + BONE PAGET + BRAIN FTD/ALS;
         stress granule assembly; cytoplasmic aggregation;
         prion-like LCD domain mutations; autosomal dominant;
         seed SEED_BASE+6).
LDB3    (LIM Domain Binding 3 / ZASP; 727 aa; 10q22.2; AD;
         Markesbery-Griggs / Zaspopathy — LATE-ONSET ANTERIOR TIBIAL ONSET;
         Z-disc structural protein; cardiomyopathy co-segregation;
         A165V founder; onset 40-60yr; vacuolar myopathy;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2190-2197).
"""

import random

SEED_BASE = 2190

DISTAL_GENES = [
    # -- DYSF — Dysferlin, LGMD-R2 / Miyoshi Myopathy 1 ---------------------------------
    {
        "gene": "DYSF",
        "alt_name": (
            "DYSF (DYSF-2080aa-2p13.2 / AR — LGMD-R2/Miyoshi-Myopathy-1 — "
            "POSTERIOR-CALF-Miyoshi-OR-PROXIMAL-LGMD-ALLELIC-PATHOGNOMONIC — "
            "CK-10-150x-ULN-Mean-50x-PATHOGNOMONIC-Myonecrosis — "
            "Dysferlin-Membrane-Repair-Protein-C2-Domain-Calcium-Sensitive)"
        ),
        "protein": (
            "DYSF -- 2p13.2 AR -- DYSF-2080aa -- "
            "Dysferlin-237kDa-Ferlin-Family-C2-Domain-Calcium-Sensitive-Membrane-Repair-Protein -- "
            "LGMD-R2-Formerly-LGMD2B-OMIM-603009-Disease-OMIM-253601 -- "
            "Miyoshi-Myopathy-1-OMIM-254130-Posterior-Calf-Onset-Same-Gene -- "
            "CK-10-150x-ULN-Mean-50x-PATHOGNOMONIC-Markedly-Elevated-All-Cases -- "
            "POSTERIOR-CALF-ATROPHY-Miyoshi-Phenotype-Difficulty-Tiptoe-Walking -- "
            "LGMD-Proximal-Phenotype-Pelvifemoral-Weakness-Also-Same-Gene -- "
            "ANO5-KEY-DDx-Posterior-Calf-NO-Quads-Spared-ANO5-Quads-Spared -- "
            "Dysferlin-IHC-ABSENT-Western-Blot-50-60kDa-Absent -- "
            "Monocyte-Dysferlin-PBMC-Test-Blood-Based-Screening -- "
            "No-Cardiac-No-Respiratory-Primarily-Skeletal-Muscle -- "
            "Inflammatory-Infiltrate-Biopsy-Common-MIMICS-INFLAMMATORY-MYOPATHY -- "
            "IMMUNOSUPPRESSION-CONTRAINDICATED-Worsens-Not-Improves -- "
            "Onset-15-30yr-Young-Adult-Typical -- "
            "Founder-Mutations-Population-Specific-No-Single-Universal-Founder -- "
            "Libyan-Jewish-R1905* -- Spanish-del-Exon32 -- Japanese-Various -- "
            "OMIM-Gene-DYSF-603009-Disease-LGMD-R2-253601-Miyoshi-254130"
        ),
        "locus": "2p13.2",
        "protein_size": "2080 aa / 237 kDa",
        "inheritance": (
            "AR (autosomal recessive) biallelic; both sexes equally; "
            "PHENOTYPE: Miyoshi (posterior calf, tiptoe difficulty) OR LGMD-R2 (proximal pelvifemoral) — "
            "SAME GENE, ALLELE-DEPENDENT; Scapulo-peroneal intermediate phenotype; "
            "Carrier frequency ~1/200 in some populations"
        ),
        "key_features": [
            "POSTERIOR CALF ATROPHY — Miyoshi phenotype pathognomonic",
            "CK 10-150x ULN PATHOGNOMONIC — markedly elevated all cases",
            "ANO5 KEY DDx: DYSF quads involved, ANO5 quads spared",
            "Dysferlin IHC absent on muscle biopsy — diagnostic",
            "PBMC monocyte dysferlin test — blood-based screening",
            "Inflammatory infiltrate mimics PM/DM — IMMUNOSUPPRESSION CI",
            "No cardiac, no respiratory involvement",
            "Onset 15-30yr young adult typical",
        ],
        "treatment": (
            "No disease-modifying therapy; physical therapy/AFO for foot drop; "
            "AVOID corticosteroids (paradoxically harmful); "
            "gene therapy (exon-skipping) in early trials"
        ),
        "contraindications": [
            "IMMUNOSUPPRESSION CONTRAINDICATED — worsens, does not improve DYSF myopathy",
            "Corticosteroids — avoid; biopsy inflammatory infiltrate does not indicate PM/DM",
        ],
        "critical_pearls": [
            "DYSF inflammatory infiltrate mimics polymyositis — always IHC/WB before immunosuppression",
            "Monocyte dysferlin test: PBMC dysferlin absent = DYSF mutation highly likely",
            "Tiptoe walking difficulty = Miyoshi onset; LGMD-R2 = proximal onset — same gene",
            "ANO5 DDx: DYSF posterior calf + quads involved; ANO5 quads spared",
            "CK 50x mean: if CK normal/low — reconsider DYSF diagnosis",
        ],
        "mean_age_onset": 22,
        "mean_age_dx": 28,
        "sex_ratio_m_f": "1:1",
    },
    # -- GNE — GNE Myopathy / Nonaka / IBM2 -------------------------------------------
    {
        "gene": "GNE",
        "alt_name": (
            "GNE (GNE-722aa-9p13.3 / AR — GNE-Myopathy-Nonaka-IBM2 — "
            "QUADRICEPS-SPARED-LATE-PATHOGNOMONIC-Critical-DDx-IBMPFD — "
            "Rimmed-Vacuoles-TDP-43-p62-Inclusions-Biopsy — "
            "M712T-Japanese-Korean-Founder-V572L-Middle-Eastern-Founder)"
        ),
        "protein": (
            "GNE -- 9p13.3 AR -- GNE-722aa -- "
            "UDP-N-acetylglucosamine-2-epimerase-N-acetylmannosamine-kinase-Bifunctional-Enzyme -- "
            "GNE-Myopathy-OMIM-605820-Disease-OMIM-605820 -- "
            "Nonaka-Myopathy-IBM2-Hereditary-Inclusion-Body-Myopathy -- "
            "QUADRICEPS-SPARED-LATE-in-Disease-PATHOGNOMONIC-DDx-sIBM-IBMPFD -- "
            "Anterior-Tibial-Onset-Foot-Drop-Early-Presentation -- "
            "CK-1-10x-ULN-Mildly-Elevated-or-Normal -- "
            "Rimmed-Vacuoles-LIGHT-MICROSCOPY-PATHOGNOMONIC -- "
            "TDP-43-p62-SQSTM1-Cytoplasmic-Inclusions-EM -- "
            "M712T-Japanese-Korean-Founder-Most-Common-Asia-Pacific -- "
            "V572L-Founder-Iranian-Jewish-Ashkenazi-Jewish-Middle-Eastern -- "
            "ManNAc-Sialic-Acid-Pathway-Supplementation-Clinical-Trials -- "
            "NO-INFLAMMATION-DDx-Sporadic-IBM-Inflammatory -- "
            "Onset-15-40yr-Typically-20s-30s -- "
            "OMIM-Gene-GNE-603824-Disease-GNE-Myopathy-605820"
        ),
        "locus": "9p13.3",
        "protein_size": "722 aa / 79 kDa",
        "inheritance": (
            "AR (autosomal recessive) biallelic; compound heterozygote or homozygous; "
            "M712T homozygous = classic Japanese/Korean GNE myopathy; "
            "V572L compound = Middle Eastern (esp. Iranian Jewish) common; "
            "Onset 15-40yr; slow progression; wheelchair 2nd-4th decade"
        ),
        "key_features": [
            "QUADRICEPS SPARED LATE — pathognomonic DDx from sIBM (quads early)",
            "Anterior tibial onset — foot drop early presentation",
            "CK 1-10x mildly elevated or normal",
            "Rimmed vacuoles on light microscopy PATHOGNOMONIC",
            "TDP-43 and p62 cytoplasmic inclusions on EM",
            "M712T Japanese/Korean founder — commonest GNE mutation globally",
            "V572L Iranian/Middle Eastern Jewish founder",
            "ManNAc/sialic acid supplementation in clinical trials",
        ],
        "treatment": (
            "No approved disease-modifying therapy; ManNAc/sialic acid supplementation "
            "(clinical trials, not standard); supportive care; AFO for foot drop; "
            "intravenous immunoglobulin NOT indicated"
        ),
        "contraindications": [
            "IMMUNOSUPPRESSION NOT INDICATED — no inflammation; misdiagnosis as IBM leads to harm",
            "IVIg — not indicated in GNE myopathy",
        ],
        "critical_pearls": [
            "QUADRICEPS SPARED LATE: sIBM quads early (quadriceps weakness hallmark sIBM); GNE quads late",
            "Rimmed vacuoles + NO inflammation + AR inheritance = GNE myopathy until proved otherwise",
            "M712T homozygous: must check GNE panel in ANY young Asian patient with foot drop + rimmed vacuoles",
            "V572L: Iranian/Jewish cohorts — GNE myopathy underdiagnosed in these populations",
            "ManNAc trial: sialic acid precursor; disease modified in mouse model; human results pending",
        ],
        "mean_age_onset": 27,
        "mean_age_dx": 33,
        "sex_ratio_m_f": "1:1",
    },
    # -- MYH7 — Beta-myosin heavy chain, Laing distal myopathy --------------------------
    {
        "gene": "MYH7",
        "alt_name": (
            "MYH7 (MYH7-1935aa-14q11.2 / AD — Laing-Distal-Myopathy-MPD1 — "
            "CONGENITAL-FOOT-DROP-PATHOGNOMONIC-Infancy-Childhood-Onset — "
            "Tibialis-Anterior-FIRST-Finger-Extensors-SECOND — "
            "NO-Cardiac-Distal-Phenotype-Unlike-HCM-Alleles)"
        ),
        "protein": (
            "MYH7 -- 14q11.2 AD -- MYH7-1935aa -- "
            "Beta-Myosin-Heavy-Chain-Sarcomeric-Thick-Filament-Slow-Twitch-Cardiac -- "
            "Laing-Distal-Myopathy-MPD1-OMIM-160500-Disease-OMIM-160500 -- "
            "CONGENITAL-FOOT-DROP-Tibialis-Anterior-PATHOGNOMONIC-Onset-Infancy-to-Childhood -- "
            "Tibialis-Anterior-FIRST-Affected-Foot-Drop-Toe-Walking -- "
            "Finger-Extensors-SECOND-Affected-Progression-Proximal-Slow -- "
            "ALLELIC-Hypertrophic-Cardiomyopathy-HCM-AD-OMIM-192600 -- "
            "ALLELIC-Dilated-Cardiomyopathy-CMD1S-OMIM-613426 -- "
            "ALLELIC-Myosin-Storage-Myopathy-OMIM-608358 -- "
            "NO-CARDIAC-IN-LAING-PHENOTYPE-CRITICAL-DDx-HCM-Alleles -- "
            "CK-Normal-to-Mildly-Elevated-1-5x-ULN -- "
            "Biopsy-Type-1-Fibre-Predominance-Core-Like-Lesions -- "
            "Distal-Myopathy-1-MPD1-AD-Autosomal-Dominant -- "
            "OMIM-Gene-MYH7-160760-Disease-Laing-160500"
        ),
        "locus": "14q11.2",
        "protein_size": "1935 aa / 223 kDa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance; de novo mutations documented; "
            "Laing distal myopathy phenotype: CONGENITAL foot drop, tibialis anterior; "
            "Allelic: HCM (different mutations K207Q etc.) — check cardiac phenotype per allele; "
            "NO cardiac involvement in Laing distal alleles — critical distinction"
        ),
        "key_features": [
            "CONGENITAL FOOT DROP — tibialis anterior first, onset infancy/childhood PATHOGNOMONIC",
            "Finger extensors second — characteristic progression",
            "CK normal to mildly elevated (1-5x)",
            "Type 1 fibre predominance + core-like lesions on biopsy",
            "NO cardiac — critical DDx from HCM-associated MYH7 alleles",
            "Allelic: HCM (K207Q etc.), DCM (CMD1S), myosin storage myopathy",
            "AD inheritance — de novo mutations documented",
            "Slow proximal progression; many remain ambulant decades",
        ],
        "treatment": (
            "Supportive: AFO for foot drop; physiotherapy; "
            "no disease-modifying therapy available; "
            "cardiac screen unnecessary in pure Laing phenotype alleles"
        ),
        "contraindications": [
            "DO NOT assume cardiac risk in Laing alleles — cardiac involvement only in HCM/DCM alleles",
            "Allele-specific cardiac risk assessment mandatory before reassurance",
        ],
        "critical_pearls": [
            "Congenital foot drop in infant/child = Laing distal myopathy until WES proves otherwise",
            "MYH7 gene: allele determines phenotype — Laing (distal, no cardiac) vs HCM (cardiac, often no weakness) vs DCM",
            "Tibialis anterior FIRST: if quads/hip flexors weaken first, reconsider Laing diagnosis",
            "Type 1 fibre predominance: MYH7 myopathy biopsy hallmark — slow myosin storage myopathy overlap",
            "NO cardiac screening needed for Laing-phenotype alleles — do NOT reflexively echo all MYH7 patients",
        ],
        "mean_age_onset": 2,
        "mean_age_dx": 8,
        "sex_ratio_m_f": "1:1",
    },
    # -- TTN — Titin, Udd/Tibial Muscular Dystrophy -------------------------------------
    {
        "gene": "TTN",
        "alt_name": (
            "TTN (TTN-34350aa-2q31.2 / AR-AD — Udd-Tibial-Muscular-Dystrophy-FINmaj — "
            "ANTERIOR-TIBIAL-ONSET-LATE-ADULT->35yr-PATHOGNOMONIC — "
            "Finnish-Founder-IVS340-1G>A-FINmaj — "
            "AD-Heterozygous-Truncating-Titinopathy-DCM-Overlap)"
        ),
        "protein": (
            "TTN -- 2q31.2 AR/AD -- TTN-34350aa -- "
            "Titin-Connectin-Largest-Human-Protein-3-6MDa-Sarcomere-Z-to-M-Line -- "
            "Udd-Myopathy-Tibial-Muscular-Dystrophy-TMD-OMIM-600334 -- "
            "ANTERIOR-TIBIAL-ONSET-PATHOGNOMONIC-Foot-Drop-Late-Adult-Onset->35yr -- "
            "Finnish-Founder-IVS340-1G>A-FINmaj-AR-Homozygous-TMD -- "
            "AR-Homozygous-FINmaj/FINmaj-or-FINmaj-compound-hetero-Classic-Udd -- "
            "AD-Heterozygous-Truncating-Variants-M-Line-C-Term-DCM-Overlap-OMIM-604145 -- "
            "Titinopathy-DCM-CMD1G-Titinopathy-Limb-Girdle-Tibial-Distal-Spectrum -- "
            "CK-Normal-to-3x-ULN-Mildly-Elevated -- "
            "Biopsy-Rimmed-Vacuoles-TMD -- "
            "EMG-Myopathic-distal -- "
            "Onset-Late-Adult->35yr-FINmaj-Homozygous -- "
            "OMIM-Gene-TTN-188840-Disease-TMD-600334-CMD1G-604145"
        ),
        "locus": "2q31.2",
        "protein_size": "34350 aa / ~3-6 MDa (isoform-dependent)",
        "inheritance": (
            "AR (TMD/Udd — homozygous or compound heterozygous FINmaj-containing alleles) "
            "OR AD (titinopathy — heterozygous truncating C-terminal M-line variants → DCM+LGMD); "
            "FINmaj = Finnish founder IVS340-1G>A; carrier frequency 1-2% in Finland; "
            "Late onset >35yr in TMD; DCM may precede/co-exist in AD form"
        ),
        "key_features": [
            "ANTERIOR TIBIAL ONSET >35yr — late adult foot drop PATHOGNOMONIC (Udd/TMD)",
            "Finnish founder IVS340-1G>A (FINmaj) — homozygous = classic TMD",
            "CK normal to mildly elevated (1-3x)",
            "Rimmed vacuoles on biopsy in TMD",
            "AD truncating variants (M-line) → titinopathy with DCM + LGMD overlap",
            "Largest human protein — WES/WGS bioinformatics challenging (repeat regions)",
            "Onset >35yr distinguishes from MYH7 (childhood) and DYSF (young adult)",
            "Slow progression; many ambulant lifelong in TMD",
        ],
        "treatment": (
            "Supportive: AFO for anterior tibial weakness; "
            "cardiac monitoring mandatory in AD titinopathy (DCM risk); "
            "no disease-modifying therapy; ICD for DCM with arrhythmia per cardiology"
        ),
        "contraindications": [
            "Do not reassure on cardiac risk in AD titinopathy without echo — DCM mandatory screen",
            "WES TTN variant interpretation requires specific M-line exon weighting — generic VUS classification unreliable",
        ],
        "critical_pearls": [
            "FINmaj homozygous: anterior tibial >35yr in Finnish patient = TMD until WGS proves otherwise",
            "AD TTN truncating (C-terminal M-line): check DCM — cardiac involvement mandatory echo",
            "TTN is hardest gene to sequence: largest human gene, repeat Z-disk regions, many VUS",
            "TMD vs FSHD DDx: FSHD asymmetric facial weakness, D4Z4 contraction; TMD symmetric no facial",
            "Onset >35yr + anterior tibial + Finland ancestry = do TTN FINmaj founder test first",
        ],
        "mean_age_onset": 42,
        "mean_age_dx": 48,
        "sex_ratio_m_f": "1:1",
    },
    # -- VCP — Valosin-Containing Protein, IBMPFD/MSP1 ----------------------------------
    {
        "gene": "VCP",
        "alt_name": (
            "VCP (VCP-806aa-17p13.3 / AD — IBMPFD-Multisystem-Proteinopathy-MSP1 — "
            "MYOPATHY+PAGET-Bone+FTD-TRIAD-PATHOGNOMONIC — "
            "R155H-Most-Common-50pct — "
            "Ubiquitin-Rimmed-Vacuoles-TDP-43-Inclusions-Biopsy)"
        ),
        "protein": (
            "VCP -- 17p13.3 AD -- VCP-806aa -- "
            "Valosin-Containing-Protein-p97-AAA-ATPase-D1-D2-Hexameric-Unfoldase -- "
            "IBMPFD-Inclusion-Body-Myopathy-Paget-Frontotemporal-Dementia-OMIM-167320 -- "
            "Multisystem-Proteinopathy-MSP1-OMIM-615422 -- "
            "MYOPATHY-PAGET-FTD-TRIAD-PATHOGNOMONIC-Any-2-of-3-Makes-Diagnosis-Likely -- "
            "R155H-Most-Common-Mutation->50pct-All-VCP-Disease -- "
            "Ubiquitin-Positive-Rimmed-Vacuoles-HALLMARK-Biopsy -- "
            "TDP-43-Cytoplasmic-Inclusions-Redistributed-From-Nucleus -- "
            "ALS-VCP-Overlap-FUS-TDP-43-Proteinopathy-Spectrum -- "
            "ALS-Like-Motor-Neuron-Phenotype-Some-VCP-Families -- "
            "Paget-Bone-Disease-PDB-Elevated-ALP-Bone-Scan-Lytic-Lesions -- "
            "FTD-Frontotemporal-Dementia-Behavioural-Variant-Most-Common -- "
            "Cardiac-Involvement-Cardiomyopathy-Some-Families -- "
            "Proteasome-Autophagy-UPS-Dysfunction-Core-Pathomechanism -- "
            "OMIM-Gene-VCP-601023-Disease-IBMPFD-167320"
        ),
        "locus": "17p13.3",
        "protein_size": "806 aa / 97 kDa",
        "inheritance": (
            "AD (autosomal dominant) missense mutations in ATPase domains; "
            "R155H most common (~50%); R155C, R191Q, R159H also common; "
            "Penetrance: myopathy 90%, Paget 51%, FTD 30%; "
            "Triad not always complete — any 2/3 triggers VCP testing"
        ),
        "key_features": [
            "IBMPFD TRIAD: MYOPATHY + PAGET BONE DISEASE + FTD — PATHOGNOMONIC any 2/3",
            "R155H most common mutation (>50% VCP disease worldwide)",
            "Ubiquitin-positive rimmed vacuoles HALLMARK on biopsy",
            "TDP-43 cytoplasmic redistribution (nucleus → cytoplasm) on IHC",
            "Serum ALP elevated in Paget component — bone scan lytic lesions",
            "ALS-like motor neuron phenotype in some families (MSP spectrum)",
            "Cardiac cardiomyopathy in some VCP families — screen",
            "Onset 40-60yr typically; proximal or distal or mixed myopathy",
        ],
        "treatment": (
            "Myopathy: supportive/physiotherapy; "
            "Paget: bisphosphonate (zoledronic acid first-line) — effective; "
            "FTD: symptomatic; cognitive reserve; "
            "Cardiac: per cardiology; "
            "VCP inhibitor CB-5083 early trials; autophagy enhancement strategies"
        ),
        "contraindications": [
            "Do not treat Paget with etidronate (first-gen bisphosphonate) — zoledronic acid preferred",
            "Immunosuppression not indicated for myopathy component",
        ],
        "critical_pearls": [
            "Any distal myopathy + rimmed vacuoles in family with bone disease or dementia = VCP until WES",
            "R155H: >50% of VCP cases — test R155H first in suspected IBMPFD before full sequencing",
            "TDP-43 redistribution: nucleus → cytoplasm; seen also in sIBM but VCP is AD familial",
            "Paget + myopathy without FTD still warrants VCP test — triad incomplete in 70% cases",
            "ALS-VCP overlap: TDP-43 is the link; VCP mutations in 1-2% familial ALS",
        ],
        "mean_age_onset": 46,
        "mean_age_dx": 52,
        "sex_ratio_m_f": "1:1",
    },
    # -- MATR3 — Matrin-3, VCPDM --------------------------------------------------------
    {
        "gene": "MATR3",
        "alt_name": (
            "MATR3 (MATR3-847aa-5q31.2 / AD — VCPDM-Vocal-Cord-Pharyngeal-Weakness-Distal-Myopathy — "
            "VOCAL-CORD-PHARYNGEAL-WEAKNESS-PATHOGNOMONIC — "
            "Hearing-Loss-SNHL-Associated — "
            "S85C-Founder-Also-ALS-Spectrum)"
        ),
        "protein": (
            "MATR3 -- 5q31.2 AD -- MATR3-847aa -- "
            "Matrin-3-Nuclear-Matrix-Protein-Two-RRM-RNA-Binding-Domains-Two-Zinc-Fingers -- "
            "VCPDM-Vocal-Cord-and-Pharyngeal-Distal-Myopathy-OMIM-606210 -- "
            "VOCAL-CORD-WEAKNESS-Hoarseness-PATHOGNOMONIC-Early-Bulbar-Feature -- "
            "PHARYNGEAL-WEAKNESS-Dysphagia-Early-Nasogastric-Risk -- "
            "Distal-Limb-Weakness-Anterior-Tibial-Finger-Flexors -- "
            "Sensorineural-Hearing-Loss-SNHL-Many-Patients -- "
            "Rimmed-Vacuoles-Biopsy-Similar-GNE-VCP -- "
            "ALS-MATR3-Spectrum-Motor-Neuron-Disease-Some-Families -- "
            "S85C-Variant-Founder-Multiple-Families -- "
            "CK-Normal-to-Mildly-Elevated -- "
            "Nuclear-Matrix-RNA-Processing-Stress-Granule-Assembly -- "
            "Onset-40-70yr-Late-Adult -- "
            "OMIM-Gene-MATR3-164995-Disease-VCPDM-606210"
        ),
        "locus": "5q31.2",
        "protein_size": "847 aa / 95 kDa",
        "inheritance": (
            "AD (autosomal dominant); S85C is the most recognized founder; "
            "Variable penetrance; onset 40-70yr; "
            "MATR3-ALS spectrum: some mutations cause pure ALS; "
            "Hearing loss co-segregates in many families — often precedes myopathy"
        ),
        "key_features": [
            "VOCAL CORD WEAKNESS — hoarseness early PATHOGNOMONIC distal myopathy feature",
            "PHARYNGEAL WEAKNESS — dysphagia early, nasogastric risk",
            "Distal limb weakness: anterior tibial + finger flexors",
            "Sensorineural hearing loss (SNHL) — common early feature",
            "Rimmed vacuoles on biopsy",
            "ALS-MATR3 spectrum — motor neuron overlap in some families",
            "S85C founder variant — test early in VCPDM phenotype",
            "Late adult onset 40-70yr",
        ],
        "treatment": (
            "Vocal cord: ENT review, voice therapy, laryngoplasty if severe; "
            "dysphagia: speech therapy, thickened feeds, PEG consideration; "
            "hearing loss: hearing aids early; "
            "limb: AFO, physiotherapy; "
            "no disease-modifying therapy"
        ),
        "contraindications": [
            "Do not defer speech therapy — early vocal cord management prevents aspiration",
            "PEG timing: plan early before respiratory compromise limits anaesthesia",
        ],
        "critical_pearls": [
            "VOCAL CORD hoarseness in distal myopathy = MATR3 first (and MATR3-ALS) until proved otherwise",
            "SNHL + distal myopathy + hoarseness triad: highest prior probability for MATR3 VCPDM",
            "S85C: commonest MATR3 mutation; test early in VCPDM phenotype without full sequencing delay",
            "MATR3-ALS: same gene, different domain — motor neuron disease spectrum; RNA-processing pathomechanism",
            "Vocal cord paralysis in distal myopathy = MATR3 or RYR1; MATR3 also causes bulbar",
        ],
        "mean_age_onset": 52,
        "mean_age_dx": 58,
        "sex_ratio_m_f": "1:1",
    },
    # -- HNRNPA2B1 — hnRNP A2/B1, Multisystem Proteinopathy MSP2 -----------------------
    {
        "gene": "HNRNPA2B1",
        "alt_name": (
            "HNRNPA2B1 (HNRNPA2B1-353aa-7p15.2 / AD — Multisystem-Proteinopathy-MSP2 — "
            "IBM-Like-PLUS-BONE-Paget-PLUS-BRAIN-FTD-ALS — "
            "D290V-Founder-Prion-Like-LCD-Domain — "
            "Stress-Granule-Assembly-Cytoplasmic-Aggregation)"
        ),
        "protein": (
            "HNRNPA2B1 -- 7p15.2 AD -- HNRNPA2B1-353aa -- "
            "Heterogeneous-Nuclear-Ribonucleoprotein-A2-B1-RNA-Binding-RRM-LCD-Domain -- "
            "Multisystem-Proteinopathy-MSP2-OMIM-615422 -- "
            "IBM-Like-Inclusion-Body-Myopathy-Rimmed-Vacuoles-TDP-43-p62 -- "
            "Bone-Paget-Disease-Lytic-Lesions-ALP-Elevated -- "
            "Brain-FTD-ALS-Frontotemporal-Dementia-Motor-Neuron-Disease -- "
            "D290V-Most-Common-Mutation-Prion-Like-LCD-Domain -- "
            "Stress-Granule-Assembly-Dysregulation-Core-Mechanism -- "
            "Cytoplasmic-Aggregation-Nuclear-Clearance-TDP-43-Like -- "
            "HNRNPA1-Allelic-Similar-Phenotype-MSP3 -- "
            "FUS-TDP-43-HNRNPA2B1-Convergent-RNA-Proteinopathy-Pathway -- "
            "CK-Mildly-Elevated-1-5x-ULN -- "
            "Late-Onset-40-60yr-AD -- "
            "OMIM-Gene-HNRNPA2B1-600124-Disease-MSP2-615422"
        ),
        "locus": "7p15.2",
        "protein_size": "353 aa / 37 kDa",
        "inheritance": (
            "AD (autosomal dominant); D290V founder mutation in LCD prion-like domain; "
            "de novo mutations documented; "
            "Penetrance: myopathy >90%, Paget 50%, FTD/ALS 30%; "
            "Triad overlap with VCP IBMPFD — GENOpanel mandatory to distinguish"
        ),
        "key_features": [
            "IBM-LIKE MYOPATHY — rimmed vacuoles + TDP-43 + p62 inclusions",
            "Paget bone disease — ALP elevated, lytic lesions",
            "FTD/ALS — frontotemporal dementia or motor neuron disease",
            "D290V most common mutation in LCD prion-like domain",
            "Stress granule dysregulation — cytoplasmic RNA-binding protein aggregation",
            "HNRNPA1 allelic — MSP3, similar phenotype",
            "VCP IBMPFD DDx — gene panel mandatory (phenotype overlap)",
            "CK mildly elevated 1-5x; onset 40-60yr",
        ],
        "treatment": (
            "Myopathy: supportive; "
            "Paget: bisphosphonate (zoledronic acid); "
            "FTD/ALS: symptomatic; riluzole for ALS component; "
            "stress granule biology — therapeutic target in development; "
            "no approved disease-modifying therapy"
        ),
        "contraindications": [
            "Do not distinguish from VCP IBMPFD clinically — gene panel required",
            "Immunosuppression not indicated for myopathy component",
        ],
        "critical_pearls": [
            "MSP2 vs MSP1 (VCP): phenotype nearly identical — only gene panel distinguishes",
            "D290V in HNRNPA2B1 LCD = canonical MSP2 — test if VCP R155H negative in IBMPFD phenotype",
            "Prion-like LCD mutations: seeding/aggregation in vitro; LLPS (liquid-liquid phase separation) biology",
            "HNRNPA1 MSP3: same pathway, D262V analogous mutation — combined HNRNPA1/A2B1 panel",
            "ALS + Paget + IBM family: consider HNRNPA2B1/A1/VCP/TBK1 multisystem proteinopathy panel",
        ],
        "mean_age_onset": 48,
        "mean_age_dx": 54,
        "sex_ratio_m_f": "1:1",
    },
    # -- LDB3/ZASP — LIM Domain Binding 3, Markesbery-Griggs/Zaspopathy ----------------
    {
        "gene": "LDB3",
        "alt_name": (
            "LDB3 (LDB3-ZASP-727aa-10q22.2 / AD — Markesbery-Griggs-Zaspopathy — "
            "LATE-ONSET-ANTERIOR-TIBIAL-ONSET-40-60yr-PATHOGNOMONIC — "
            "Z-Disc-Structural-Protein-Alpha-Actinin-Binding — "
            "A165V-Founder-Cardiomyopathy-Co-Segregation)"
        ),
        "protein": (
            "LDB3 -- 10q22.2 AD -- LDB3-727aa -- "
            "LIM-Domain-Binding-Protein-3-ZASP-Z-Band-Alternatively-Spliced-PDZ-Motif -- "
            "Markesbery-Griggs-Distal-Myopathy-OMIM-609452 -- "
            "Zaspopathy-LDB3-Distal-Myopathy-AD -- "
            "ANTERIOR-TIBIAL-ONSET-LATE-40-60yr-Foot-Drop -- "
            "Z-Disc-Structural-Role-Alpha-Actinin-2-Binding-Partner -- "
            "A165V-Most-Common-Mutation-Founder-Multiple-Families -- "
            "Cardiomyopathy-Dilated-Or-Hypertrophic-Co-Segregation-Families -- "
            "Vacuolar-Myopathy-Biopsy-Rimmed-Vacuoles-Protein-Aggregates -- "
            "Myofibrillar-Myopathy-Overlap-Z-Disc-Pathology -- "
            "CK-Normal-to-Mild-1-5x-ULN -- "
            "Slow-Progression-Many-Ambulant->10yr-After-Onset -- "
            "OMIM-Gene-LDB3-605906-Disease-Markesbery-609452"
        ),
        "locus": "10q22.2",
        "protein_size": "727 aa / 77 kDa",
        "inheritance": (
            "AD (autosomal dominant); A165V most common; "
            "Onset 40-60yr late adult; penetrance high; "
            "Cardiomyopathy (DCM or HCM) co-segregates in families — echo mandatory; "
            "Vacuolar myopathy on biopsy; Z-disc pathology = myofibrillar myopathy overlap"
        ),
        "key_features": [
            "LATE-ONSET ANTERIOR TIBIAL — foot drop 40-60yr PATHOGNOMONIC",
            "A165V most common founder mutation",
            "Cardiomyopathy (DCM/HCM) co-segregates — echo mandatory for all LDB3",
            "Vacuolar myopathy + protein aggregates on biopsy",
            "Z-disc structural protein — myofibrillar myopathy overlap",
            "Slow progression — many ambulant >10yr after onset",
            "CK normal to mildly elevated",
            "TTN FINmaj DDx: both anterior tibial late adult; TTN Finland, LDB3 all populations",
        ],
        "treatment": (
            "Supportive: AFO for anterior tibial; physiotherapy; "
            "cardiac: echo every 2-3yr or guided by symptoms; "
            "no disease-modifying therapy; "
            "ICD per cardiology for DCM with arrhythmia"
        ),
        "contraindications": [
            "Do not omit cardiac echo — cardiomyopathy risk in LDB3 families is real",
            "Do not confuse with TTN FINmaj — geography and gene panel distinguish",
        ],
        "critical_pearls": [
            "Anterior tibial late adult + cardiac family history = LDB3 (any population) or TTN (Finland) — gene panel first",
            "A165V: test early if Markesbery-Griggs phenotype suspected — 50-60% of LDB3 mutations",
            "Echo all LDB3: DCM/HCM co-segregation rate significant — cardiac involvement may precede myopathy",
            "Myofibrillar myopathy panel should include LDB3, FLNC, DES, MYOT, BAG3 — Z-disc proteinopathy group",
            "Vacuolar myopathy biopsy in late-adult distal + slow progression = LDB3/ZASP, GNE, TTN — gene panel mandatory",
        ],
        "mean_age_onset": 50,
        "mean_age_dx": 56,
        "sex_ratio_m_f": "1:1",
    },
]


def _make_patients(gene_data, seed):
    """Generate 40 realistic synthetic patients for a given distal myopathy gene."""
    rng = random.Random(seed)
    gene = gene_data["gene"]
    mean_onset = gene_data["mean_age_onset"]
    mean_dx_delay = gene_data["mean_age_dx"] - mean_onset

    outcomes = {
        "DYSF":     {"ambulant_10yr": 0.55, "alive_20yr": 0.90, "treated_pct": 0.30},
        "GNE":      {"ambulant_10yr": 0.65, "alive_20yr": 0.88, "treated_pct": 0.20},
        "MYH7":     {"ambulant_10yr": 0.82, "alive_20yr": 0.95, "treated_pct": 0.15},
        "TTN":      {"ambulant_10yr": 0.78, "alive_20yr": 0.88, "treated_pct": 0.12},
        "VCP":      {"ambulant_10yr": 0.60, "alive_20yr": 0.75, "treated_pct": 0.55},
        "MATR3":    {"ambulant_10yr": 0.58, "alive_20yr": 0.78, "treated_pct": 0.45},
        "HNRNPA2B1":{"ambulant_10yr": 0.62, "alive_20yr": 0.80, "treated_pct": 0.50},
        "LDB3":     {"ambulant_10yr": 0.75, "alive_20yr": 0.88, "treated_pct": 0.20},
    }
    out = outcomes.get(gene, {"ambulant_10yr": 0.65, "alive_20yr": 0.85, "treated_pct": 0.30})

    patients = []
    for i in range(40):
        age_onset = max(0, int(rng.gauss(mean_onset, max(mean_onset * 0.25, 3))))
        dx_delay = max(0, int(rng.gauss(mean_dx_delay, 3)))
        age_dx = age_onset + dx_delay
        current_age = age_dx + rng.randint(1, 20)
        ck_mult = rng.uniform(
            {"DYSF": 10, "GNE": 1, "MYH7": 1, "TTN": 1, "VCP": 1, "MATR3": 1, "HNRNPA2B1": 1, "LDB3": 1}.get(gene, 1),
            {"DYSF": 150, "GNE": 10, "MYH7": 5, "TTN": 3, "VCP": 5, "MATR3": 3, "HNRNPA2B1": 5, "LDB3": 3}.get(gene, 5),
        )
        ck_val = round(ck_mult * rng.uniform(0.8, 1.2) * 200, 0)  # ULN~200 IU/L approximate
        ambulant = rng.random() < out["ambulant_10yr"]
        alive = rng.random() < out["alive_20yr"]
        treated = rng.random() < out["treated_pct"]
        attacks_py = round(rng.uniform(0, 2), 1) if gene in ("MYH7", "TTN") else round(rng.uniform(0, 1), 1)
        patients.append({
            "id": f"{gene}_{seed}_{i:02d}",
            "gene": gene,
            "age_onset": age_onset,
            "age_dx": age_dx,
            "dx_delay_yr": dx_delay,
            "current_age": current_age,
            "ck_iu_l": int(ck_val),
            "ambulant_at_10yr": ambulant,
            "alive": alive,
            "treated": treated,
            "attacks_per_year": attacks_py,
            "sex": rng.choice(["M", "F"]),
        })
    return patients


def overview():
    """Aggregate overview across all 8 genes (320 patients)."""
    all_patients = []
    for i, gd in enumerate(DISTAL_GENES):
        all_patients += _make_patients(gd, SEED_BASE + i)

    n = len(all_patients)
    alive = sum(1 for p in all_patients if p["alive"])
    treated = sum(1 for p in all_patients if p["treated"])
    ambulant = sum(1 for p in all_patients if p["ambulant_at_10yr"])
    mean_onset = round(sum(p["age_onset"] for p in all_patients) / n, 1)
    mean_dx_delay = round(sum(p["dx_delay_yr"] for p in all_patients) / n, 1)
    mean_ck = round(sum(p["ck_iu_l"] for p in all_patients) / n, 0)

    gene_summaries = []
    for i, gd in enumerate(DISTAL_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        gene_summaries.append({
            "gene": gd["gene"],
            "n_patients": len(pts),
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance": gd["inheritance"].split(";")[0].strip(),
            "mean_onset_yr": round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "mean_ck": round(sum(p["ck_iu_l"] for p in pts) / len(pts), 0),
            "alive_pct": round(sum(1 for p in pts if p["alive"]) / len(pts) * 100, 1),
            "ambulant_10yr_pct": round(sum(1 for p in pts if p["ambulant_at_10yr"]) / len(pts) * 100, 1),
        })

    return {
        "atlas": "Hereditary-Distal-Myopathy-Atlas",
        "subtitle": "Complete 8-Gene Distal Myopathy Spectrum (DYSF · GNE · MYH7 · TTN · VCP · MATR3 · HNRNPA2B1 · LDB3)",
        "genes_covered": [gd["gene"] for gd in DISTAL_GENES],
        "n_genes": 8,
        "n_patients": n,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "alive_pct": round(alive / n * 100, 1),
        "treated_pct": round(treated / n * 100, 1),
        "ambulant_10yr_pct": round(ambulant / n * 100, 1),
        "mean_age_onset_yr": mean_onset,
        "mean_dx_delay_yr": mean_dx_delay,
        "mean_ck_iu_l": int(mean_ck),
        "key_spectrum_facts": [
            "DYSF: most common AR distal myopathy; CK 10-150x; posterior calf OR proximal LGMD-R2; IMMUNOSUPPRESSION CI",
            "GNE: QUADRICEPS SPARED LATE PATHOGNOMONIC; rimmed vacuoles; M712T Japanese founder; V572L Middle Eastern",
            "MYH7: CONGENITAL FOOT DROP; tibialis anterior onset infancy; NO cardiac in Laing alleles; allelic HCM",
            "TTN: ANTERIOR TIBIAL >35yr; FINmaj Finnish founder; AD truncating = titinopathy DCM ECHO MANDATORY",
            "VCP: IBMPFD TRIAD — MYOPATHY+PAGET+FTD; R155H >50%; TDP-43 redistribution; bisphosphonate for Paget",
            "MATR3: VOCAL CORD + PHARYNGEAL + DISTAL LIMB TRIAD; SNHL; S85C founder; ALS-MATR3 spectrum",
            "HNRNPA2B1: MSP2 — IBM-like + PAGET + FTD/ALS; D290V LCD domain; VCP DDx by gene panel only",
            "LDB3/ZASP: LATE 40-60yr ANTERIOR TIBIAL; A165V; CARDIAC echo mandatory; Z-disc myofibrillar myopathy",
        ],
        "critical_ddx": {
            "DYSF_vs_ANO5": "DYSF: quads involved, posterior calf + proximal possible; ANO5: quads spared, no cardiac",
            "GNE_vs_sIBM": "GNE: AR biallelic, quads spared LATE, no inflammation; sIBM: sporadic, quads EARLY, CD8+ inflammation",
            "MYH7_vs_SMA": "MYH7 Laing: congenital foot drop, no tongue/bulbar, normal EMG distal myopathic; SMA: denervation EMG, SMN1",
            "TTN_vs_FSHD": "TTN TMD: symmetric anterior tibial, no facial, no D4Z4; FSHD: asymmetric, facial weakness, D4Z4 contraction",
            "VCP_vs_HNRNPA2B1": "VCP (MSP1) vs HNRNPA2B1 (MSP2): identical phenotype IBMPFD triad — gene panel MANDATORY",
            "MATR3_vs_OPMD": "MATR3: distal limb prominent, AD; OPMD (PABPN1): ptosis + dysphagia dominant, trinucleotide expansion",
            "LDB3_vs_TTN_distal": "LDB3: any population, cardiac co-segregation, A165V; TTN FINmaj: Finland, anterior tibial, FINmaj test",
        },
        "gene_summaries": gene_summaries,
    }


def breakdown():
    """Per-gene clinical breakdown with key metrics."""
    result = {}
    for i, gd in enumerate(DISTAL_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        result[gd["gene"]] = {
            "gene": gd["gene"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "n_patients": len(pts),
            "alive_pct": round(sum(1 for p in pts if p["alive"]) / len(pts) * 100, 1),
            "treated_pct": round(sum(1 for p in pts if p["treated"]) / len(pts) * 100, 1),
            "ambulant_10yr_pct": round(sum(1 for p in pts if p["ambulant_at_10yr"]) / len(pts) * 100, 1),
            "mean_age_onset": round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "mean_dx_delay_yr": round(sum(p["dx_delay_yr"] for p in pts) / len(pts), 1),
            "mean_ck_iu_l": int(sum(p["ck_iu_l"] for p in pts) / len(pts)),
            "mean_attacks_per_year": round(sum(p["attacks_per_year"] for p in pts) / len(pts), 1),
            "key_features": gd["key_features"],
            "treatment": gd["treatment"],
            "contraindications": gd["contraindications"],
            "critical_pearls": gd["critical_pearls"],
            "inheritance": gd["inheritance"][:160],
        }
    return result


def definitions():
    """Gene definitions, biopsy patterns, DDx tables, founder mutations."""
    gene_defs = {}
    for gd in DISTAL_GENES:
        gene_defs[gd["gene"]] = {
            "full_name": gd["protein"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance_detail": gd["inheritance"],
            "alt_name": gd["alt_name"],
        }

    return {
        "gene_definitions": gene_defs,
        "onset_age_spectrum": {
            "MYH7_Laing": "Infancy-childhood (congenital foot drop)",
            "DYSF":        "Young adult 15-30yr (posterior calf or proximal)",
            "GNE":         "Young to mid adult 15-40yr (anterior tibial foot drop)",
            "TTN_TMD":     "Late adult >35yr (anterior tibial, Finnish founder)",
            "VCP":         "Mid-late adult 40-60yr (IBMPFD triad)",
            "HNRNPA2B1":   "Mid-late adult 40-60yr (MSP2 triad)",
            "MATR3":       "Late adult 40-70yr (VCPDM triad)",
            "LDB3":        "Late adult 40-60yr (anterior tibial, Z-disc myopathy)",
        },
        "biopsy_pattern_table": {
            "DYSF":        "Necrosis, inflammation (mimics PM/DM) — IHC dysferlin ABSENT; no rimmed vacuoles",
            "GNE":         "Rimmed vacuoles + TDP-43/p62 cytoplasmic inclusions — hallmark; no inflammation",
            "MYH7":        "Type 1 fibre predominance + core-like lesions; hyaline inclusions (myosin storage myopathy alleles)",
            "TTN_TMD":     "Rimmed vacuoles (in TMD); non-specific myopathic changes; myofibrillar disorganisation",
            "VCP":         "Ubiquitin-positive rimmed vacuoles + TDP-43 redistribution + p62 inclusions",
            "MATR3":       "Rimmed vacuoles; non-specific myopathic; nuclear inclusions some cases",
            "HNRNPA2B1":   "Rimmed vacuoles + TDP-43 + p62 inclusions — similar VCP/GNE pattern",
            "LDB3":        "Vacuolar myopathy + protein aggregates (desmin, αB-crystallin); Z-disc streaming on EM",
        },
        "immunosuppression_table": {
            "DYSF":        "CONTRAINDICATED — inflammatory infiltrate mimics PM but IS NOT PM; corticosteroids cause harm",
            "GNE":         "NOT INDICATED — no inflammatory component; misdiagnosis as IBM leads to harm",
            "MYH7":        "Not applicable — no inflammatory infiltrate",
            "TTN":         "Not applicable — no inflammatory infiltrate",
            "VCP":         "Not indicated for myopathy; treat Paget with bisphosphonate",
            "MATR3":       "Not indicated for myopathy",
            "HNRNPA2B1":   "Not indicated for myopathy; treat Paget with bisphosphonate",
            "LDB3":        "Not applicable",
        },
        "cardiac_surveillance_table": {
            "DYSF":        "No cardiac surveillance required — no cardiac involvement in DYSF",
            "GNE":         "No cardiac surveillance required — no cardiac involvement in GNE",
            "MYH7_Laing":  "NO cardiac screening for Laing-alleles — cardiac only in HCM/DCM alleles; allele-specific",
            "TTN_AD":      "Echo + Holter every 1-2yr for AD titinopathy — DCM risk; arrhythmia risk",
            "VCP":         "Echo if cardiac symptoms; cardiomyopathy in some families",
            "MATR3":       "Echo if cardiac symptoms; not systematically elevated risk",
            "HNRNPA2B1":   "Echo if cardiac symptoms; cardiomyopathy in some MSP2 families",
            "LDB3":        "Echo every 2-3yr MANDATORY — DCM/HCM co-segregates in LDB3 families",
        },
        "ddx_table": {
            "DYSF_vs_ANO5":         "DYSF: quads involved, CK 10-150x; ANO5: quads SPARED, no cardiac — key distinguisher",
            "GNE_vs_sIBM":          "GNE: AR biallelic, quads spared late, no CD8+ inflammation; sIBM: sporadic, quads early weakness, endomysial inflammation",
            "GNE_vs_VCP":           "GNE: AR, no Paget/FTD; VCP: AD, Paget+FTD+myopathy IBMPFD triad",
            "MYH7_vs_RYR1":         "MYH7 Laing: foot drop onset infancy, Type1-fibre, no cores; RYR1-cores: cores on NADH, malignant hyperthermia risk",
            "VCP_vs_HNRNPA2B1":     "VCP MSP1 vs HNRNPA2B1 MSP2: identical IBMPFD phenotype — distinguish ONLY by gene panel; R155H VCP vs D290V HNRNPA2B1",
            "TTN_vs_FSHD":          "TTN TMD: symmetric, no facial, anterior tibial; FSHD: asymmetric, facial, humeral, D4Z4 contraction",
            "MATR3_vs_OPMD":        "MATR3: distal limb + vocal cord; OPMD (PABPN1): ptosis + dysphagia dominant, GCN repeat expansion",
            "LDB3_vs_Myofibrillar":  "LDB3 overlaps myofibrillar myopathy panel (FLNC, DES, MYOT, BAG3); all Z-disc proteinopathies — full panel mandatory",
        },
        "founder_mutations": {
            "GNE_M712T":          "GNE p.Met712Thr — Japanese/Korean founder; homozygous = classic GNE myopathy East Asia",
            "GNE_V572L":          "GNE p.Val572Leu — Iranian/Ashkenazi Jewish/Middle Eastern founder",
            "DYSF_R1905*_Libyan":  "DYSF p.Arg1905* — Libyan Jewish community",
            "TTN_FINmaj":         "TTN IVS340-1G>A — Finnish founder; carrier frequency 1-2% Finland; homozygous = Udd TMD",
            "VCP_R155H":          "VCP p.Arg155His — most common VCP mutation globally; >50% all IBMPFD/MSP1",
            "MATR3_S85C":         "MATR3 p.Ser85Cys — most recognised MATR3 VCPDM mutation",
            "HNRNPA2B1_D290V":    "HNRNPA2B1 p.Asp290Val — LCD domain founder; commonest MSP2 mutation",
            "LDB3_A165V":         "LDB3/ZASP p.Ala165Val — Markesbery-Griggs founder; 50-60% LDB3 disease",
        },
        "multisystem_proteinopathy_panel": {
            "MSP1":  "VCP — IBMPFD (myopathy + Paget + FTD); R155H dominant",
            "MSP2":  "HNRNPA2B1 — IBM-like + Paget + FTD/ALS; D290V dominant",
            "MSP3":  "HNRNPA1 — similar to MSP2; D262V analogous mutation",
            "MSP4":  "SQSTM1/p62 — Paget + ALS; phenotype overlap",
            "common_pathway": "UPS/autophagy failure; stress granule dysregulation; TDP-43 redistribution; RNA-binding protein aggregation",
            "gene_panel_recommendation": "If IBMPFD phenotype: test VCP R155H first → full VCP → HNRNPA2B1 D290V → HNRNPA1 → SQSTM1",
        },
        "myofibrillar_z_disc_panel": {
            "genes": ["LDB3/ZASP", "DES (desmin)", "MYOT (myotilin)", "FLNC (filamin-C)", "BAG3", "CRYAB (αB-crystallin)"],
            "shared_biopsy": "Protein aggregates + rimmed vacuoles + Z-disc streaming on EM",
            "cardiac_co-involvement": "DES, FLNC, BAG3 — cardiac involvement common; LDB3, MYOT — variable",
            "panel_note": "All Z-disc proteinopathies; overlap substantial; full myofibrillar myopathy panel recommended",
        },
        "glossary": {
            "Rimmed_vacuoles": "Vacuoles with basophilic rims on H&E; autophagic vacuoles containing ubiquitinated proteins; hallmark GNE, VCP, HNRNPA2B1, TTN-TMD",
            "TDP-43_redistribution": "TDP-43 normally nuclear; in MSP/VCP/GNE clears from nucleus → cytoplasmic aggregates; identical ALS/FTD finding",
            "Dysferlin_membrane_repair": "Dysferlin: ferlin-family C2-domain protein; calcium-triggered membrane repair at sarcolemmal tears; absent → necrosis cascade",
            "Prion_like_LCD": "Low-complexity domain (LCD) in hnRNP proteins; phase-transitions (LLPS) → stress granules; mutations stabilise aggregates → pathological",
            "IBMPFD_triad": "Inclusion Body Myopathy + Paget Bone Disease + Frontotemporal Dementia — any 2/3 warrants VCP gene testing",
            "VCPDM": "Vocal Cord and Pharyngeal Distal Myopathy — MATR3-specific triad; hoarseness + dysphagia + distal limb",
            "PBMC_dysferlin": "Peripheral blood mononuclear cell dysferlin assay: monocytes express dysferlin; WB of PBMC = blood-based DYSF screening",
            "FINmaj": "Finnish major founder allele TTN IVS340-1G>A splice-site; homozygous = tibial muscular dystrophy (Udd); carrier 1-2% Finland",
            "Zaspopathy": "Distal myopathy caused by LDB3/ZASP mutations; Z-disc structural protein; overlap myofibrillar myopathy",
            "GNE_pathway": "GNE enzyme: bifunctional; rate-limiting step sialic acid biosynthesis; GNE myopathy = hyposialylation glycoproteins → therapeutic target",
            "Titinopathy": "TTN gene disease spectrum: Udd/TMD (AR homozygous, distal) → LGMD (AR) → DCM-titinopathy (AD heterozygous C-terminal truncating)",
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = overview()
    print(json.dumps({k: v for k, v in ov.items() if k != "gene_summaries"}, indent=2))
    print("\n=== BREAKDOWN keys ===")
    br = breakdown()
    for gene, data in br.items():
        print(f"  {gene}: {data['n_patients']} patients, alive={data['alive_pct']}%, ambulant_10yr={data['ambulant_10yr_pct']}%")
    print("\n=== DEFINITIONS keys ===")
    defs = definitions()
    print(list(defs.keys()))
