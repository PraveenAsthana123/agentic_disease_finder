#!/usr/bin/env python3
"""Hereditary-Leukodystrophy-Atlas — Complete 8-Gene Atlas (Hereditary White Matter Disorders)
ARSA    (Arylsulfatase A; 507 aa; 22q13.33; AR;
         Metachromatic Leukodystrophy (MLD); sulfatide accumulation;
         Libmeldy / OTL-200 (EMA2020) — first approved gene therapy for MLD;
         pseudodeficiency pitfall: N350S + I179S polymorphism → low enzyme but no disease;
         seed SEED_BASE+0) .
GALC    (Galactocerebrosidase; 669 aa; 14q31.3; AR;
         Krabbe Disease / Globoid Cell Leukodystrophy;
         psychosine TOXIC at nanomolar concentrations → oligodendrocyte death;
         HSCT ONLY beneficial if pre-symptomatic — NBS mandatory;
         seed SEED_BASE+1) .
PLP1    (Proteolipid Protein 1; 276 aa; Xq22.2; X-linked;
         Pelizaeus-Merzbacher Disease (PMD) — DUPLICATION most common 70%;
         diffuse hypomyelination; nystagmus at birth = earliest clue;
         MLPA mandatory (detects duplications/deletions);
         seed SEED_BASE+2) .
ABCD1   (ATP-binding cassette D1 / ALDP; 745 aa; Xq28; X-linked;
         X-linked Adrenoleukodystrophy (X-ALD); VLCFA accumulation;
         CCALD: Skysona (elivaldogene, FDA Aug 2022) gene therapy; HSCT Level A early CCALD;
         adrenal insufficiency 71% males — adrenal crisis = emergency;
         genotype does NOT predict phenotype;
         seed SEED_BASE+3) .
ASPA    (Aspartoacylase; 313 aa; 17p13.2; AR;
         Canavan Disease; NAA elevated — most specific MRS biomarker in leukodystrophy;
         U-fibres involved; Ashkenazi Jewish founder mutations p.Glu285Ala + p.Tyr231X;
         seed SEED_BASE+4) .
GFAP    (Glial Fibrillary Acidic Protein; 432 aa; 17q21.31; AD;
         Alexander Disease — ALL mutations DOMINANT GAIN-OF-FUNCTION, NOT LOF;
         Rosenthal fibres (perivascular eosinophilic aggregates) PATHOGNOMONIC;
         GFAP protein elevated in CSF/blood — astrocytic injury marker;
         seed SEED_BASE+5) .
EIF2B5  (eIF2B epsilon subunit; 712 aa; 3q27.1; AR;
         Vanishing White Matter Disease (VWM); stress-triggered episodes EMERGENCY;
         ISR (integrated stress response) hypersensitivity; ISRIB most promising trial drug;
         ovarioleukodystrophy — premature ovarian failure in females;
         seed SEED_BASE+6) .
POLR3A  (RNA Polymerase III Subunit A; 1390 aa; 10q22.3; AR;
         POLR3-Related Leukodystrophy / HLD7; hypomyelination + cerebellar atrophy;
         dental abnormalities + leukodystrophy = POLR3 first;
         RNA Pol III → tRNA synthesis impairment → hypomyelination;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1934–1941)
"""

import random

SEED_BASE = 1934

LEUKODYSTROPHY_GENES = [
    # -- ARSA — Arylsulfatase A / Metachromatic Leukodystrophy (MLD) -----------------
    {
        "gene": "ARSA",
        "alt_name": "Arylsulfatase A (MLD)",
        "protein": (
            "ARSA -- 22q13.33 AR -- AryIsulfataseA-507aa -- "
            "MLD-Metachromatic-Leukodystrophy-Sulfatide-Accumulation -- "
            "Libmeldy-OTL200-EMA2020-First-Approved-Gene-Therapy-MLD -- "
            "Pseudodeficiency-N350S-I179S-Low-Enzyme-Normal-Sulfatide -- "
            "Tigroid-Leopard-Skin-MRI-Periventricular-T2-PATHOGNOMONIC"
        ),
        "locus": "22q13.33",
        "protein_size": "507 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Late-infantile (most common, ~50%): onset 1–4 yr, walking regression, hypotonia then spasticity; "
            "Juvenile: onset 4–16 yr, cognitive/behavioural first, then motor decline; "
            "Adult: onset >16 yr, psychiatric symptoms first (misdiagnosed as schizophrenia), then motor"
        ),
        "key_biomarker": (
            "Leukocyte ARSA enzyme activity <5–10% of normal (confirm with two independent assays); "
            "urine sulfatides elevated (metachromatic granules on urine microscopy — brown-red with crystal violet); "
            "MRI: confluent symmetric periventricular WM T2 hyperintensity with 'tigroid' or 'leopard-skin' sparing of perivascular regions (U-fibres spared early); "
            "CSF protein mildly elevated; NCV: demyelinating polyneuropathy; "
            "molecular: ARSA biallelic mutations; EXCLUDE pseudodeficiency (sulfatide urine normal in pseudodeficiency)"
        ),
        "pathognomonic": (
            "Late-infantile: walking regression at 1–4 yr + hypotonia progressing to spasticity + "
            "MRI tigroid/leopard-skin periventricular T2 signal + low ARSA enzyme + elevated urine sulfatides; "
            "Pseudodeficiency pitfall: N350S (c.1049A>G) + I179S (c.536T>G) = low enzyme BUT sulfatide urine NORMAL and NO disease; "
            "ALWAYS confirm low ARSA enzyme with urine/plasma sulfatide measurement + genetic testing before diagnosing MLD"
        ),
        "treatment": (
            "Libmeldy (OTL-200; eMA2020) — ex vivo lentiviral gene therapy; CD34+ HSCs transduced; "
            "INDICATED: pre-symptomatic late-infantile or early-symptomatic late-infantile (walking), OR pre-symptomatic/early-symptomatic juvenile; "
            "NOT for: late-symptomatic, adult MLD, or patients who have lost walking; "
            "HSCT: some benefit in pre-symptomatic juvenile; less effective than in Krabbe; "
            "Symptomatic: antiepileptics (LEV, VPA), physiotherapy, intrathecal enzyme replacement (investigational); "
            "Substrate reduction therapy (OA-519, lucerastat) in trials; "
            "ACE-i/ARB NOT relevant (neurological disease); genetic counselling"
        ),
        "critical_flags": [
            "ARSA-PSEUDODEFICIENCY-PITFALL: N350S + I179S polymorphism → ARSA enzyme LOW but NO disease; sulfatide urine NORMAL; carrier frequency 1-2% Europeans; ALWAYS check sulfatide urine before diagnosing MLD",
            "ARSA-LIBMELDY-EMA2020: first approved gene therapy for MLD (Europe 2020); pre-symptomatic or early-symptomatic only; walking must be preserved at treatment; TIMING IS CRITICAL",
            "ARSA-TIGROID-MRI: periventricular T2 hyperintensity with small spared areas around vessels ('tigroid'/'leopard-skin') — PATHOGNOMONIC for MLD; U-fibres spared early",
            "ARSA-ADULT-PSYCHIATRIC: adult MLD presents as schizophrenia/bipolar; white matter signal on MRI in 'psychiatric' patient = always check ARSA enzyme",
            "ARSA-NBS-GAP: MLD not universally on NBS panels; sibling diagnosis pathway critical for pre-symptomatic treatment window",
            "ARSA-SULFATIDE-URINE: urine sulfatides (metachromatic granules) confirm MLD diagnosis when ARSA enzyme low; normal sulfatide in low-enzyme patient = pseudodeficiency",
            "ARSA-HSCT-LIMITED: HSCT less effective for MLD than Krabbe; gene therapy (Libmeldy) preferred when available for eligible patients",
            "ARSA-JUVENILE-COGNITIVE-FIRST: juvenile MLD presents with cognitive and behavioural decline before motor — school failure, personality change, cognitive regression = MRI + ARSA enzyme",
        ],
        "alias": (
            "ARSA (Arylsulfatase A; 507 aa; 22q13.33) encodes a lysosomal enzyme that cleaves "
            "3-O-sulphogalactosylceramide (sulfatide) into galactosylceramide + sulphate. "
            "Biallelic LOF mutations cause Metachromatic Leukodystrophy (MLD; OMIM #250100) — "
            "sulfatide accumulates in lysosomes of oligodendrocytes, Schwann cells, and kidney → "
            "demyelination of CNS and PNS. Three clinical forms: late-infantile (onset 1–4 yr, most severe, "
            "walking regression, hypotonia, polyneuropathy), juvenile (4–16 yr, cognitive first), "
            "adult (>16 yr, psychiatric onset — frequently misdiagnosed). "
            "MRI: confluent periventricular T2 WM signal with tigroid/leopard-skin pattern (perivascular sparing) — pathognomonic. "
            "Pseudodeficiency pitfall: N350S + I179S polymorphisms in trans give low ARSA enzyme activity "
            "WITHOUT disease; urine sulfatide measurement is mandatory to distinguish. "
            "Libmeldy (OTL-200; EMA 2020) — ex vivo lentiviral CD34+ gene therapy — is the first approved "
            "treatment; eligible only for pre-symptomatic/early-symptomatic patients who still walk. "
            "HSCT has limited evidence in MLD (less effective than Krabbe). "
            "Molecular: biallelic ARSA mutations; pseudodeficiency alleles must be identified."
        ),
    },

    # -- GALC — Galactocerebrosidase / Krabbe Disease --------------------------------
    {
        "gene": "GALC",
        "alt_name": "Galactocerebrosidase (Krabbe)",
        "protein": (
            "GALC -- 14q31.3 AR -- Galactocerebrosidase-669aa -- "
            "Krabbe-Disease-Globoid-Cell-Leukodystrophy-Psychosine-Toxic -- "
            "HSCT-ONLY-Pre-Symptomatic-NBS-Essential -- "
            "Globoid-Cells-Multinucleated-Macrophages-PATHOGNOMONIC -- "
            "Psychosine-Nanomolar-Toxic-Oligodendrocytes-NOT-GalCer"
        ),
        "locus": "14q31.3",
        "protein_size": "669 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Infantile (85%): onset <6 months — hypersensitivity, irritability, hypertonicity, fever, rapidly progressive; "
            "Late-infantile: 6 months–3 yr; "
            "Juvenile: 3–16 yr; "
            "Adult: >16 yr — spastic paraplegia, peripheral neuropathy"
        ),
        "key_biomarker": (
            "Leukocyte GALC enzyme activity near-zero in infantile (<0.5 nmol/hr/mg); "
            "psychosine (galactosylsphingosine) levels elevated — most specific biomarker; "
            "CSF protein markedly elevated; NCV: severe demyelinating polyneuropathy; "
            "MRI: deep cerebellar WM, posterior cerebral WM, corticospinal tracts T2 signal early; "
            "NBS: DBS GALC enzyme + confirmatory psychosine; "
            "molecular: biallelic GALC mutations (30kb deletion common in European Krabbe)"
        ),
        "pathognomonic": (
            "Infantile: extreme irritability + hypertonicity + fever spells + rapid neurological decline <6 months + "
            "near-zero leukocyte GALC enzyme + psychosine elevated + globoid cells on brain biopsy; "
            "Globoid cells (multinucleated CD68+ macrophages with galactosylceramide inclusions) in WM — PATHOGNOMONIC on brain biopsy; "
            "psychosine hypothesis: psychosine (NOT galactosylceramide) is the toxic metabolite — nanomolar levels kill oligodendrocytes; "
            "GALC enzyme in DBS near-zero + psychosine >1 nM (NBS threshold) → immediate HSCT referral pre-symptomatic"
        ),
        "treatment": (
            "HSCT (haematopoietic stem cell transplantation) — ONLY effective if PRE-SYMPTOMATIC; "
            "NBS identification → HSCT within weeks → partially prevents neurological progression; "
            "SYMPTOMATIC patients: HSCT does NOT benefit — palliative care; "
            "infantile Krabbe post-symptom = NO HSCT benefit; "
            "Gene therapy (AAV9-GALC): Phase I/II trials (GECT01); combined GALC + psychosine-lowering approaches; "
            "Substrate reduction: L-cycloserine (reduces psychosine in animal models — trials); "
            "NBS programs (New York, other US states, Japan) have identified pre-symptomatic infants → HSCT → improved outcomes; "
            "palliative: seizure management (LEV, VPA), nutrition, pain"
        ),
        "critical_flags": [
            "GALC-HSCT-PRESYMPTOMATIC-ONLY: HSCT ONLY works pre-symptomatic; symptomatic infantile Krabbe = HSCT gives NO neurological benefit; NBS is the only way to identify pre-symptomatic",
            "GALC-PSYCHOSINE-TOXIC: psychosine (galactosylsphingosine) — NOT galactosylceramide — is the direct cytotoxin; nanomolar concentrations kill oligodendrocytes; psychosine level is the best biomarker",
            "GALC-NBS-MANDATORY: newborn screening (DBS GALC enzyme + psychosine) is MANDATORY for HSCT to be feasible; without NBS, infantile Krabbe diagnosed symptomatically = too late for HSCT",
            "GALC-GLOBOID-CELLS-PATHOGNOMONIC: multinucleated macrophages (globoid cells) in WM on brain biopsy — pathognomonic for Krabbe; seen on biopsy or autopsy",
            "GALC-INFANTILE-FATAL-2yr: untreated infantile Krabbe fatal by 2 yr; irritability + hypertonicity + fever in young infant = Krabbe until proven otherwise",
            "GALC-30KB-DELETION: common European deletion (30 kb, also called 502T/del) — MLPA detects; accounts for ~35-45% European Krabbe alleles; complex deletions require long-read sequencing",
            "GALC-ADULT-SPASTIC-PARAPLEGIA: adult Krabbe = progressive spastic paraplegia + peripheral neuropathy + cerebellar signs; much milder than infantile; often misdiagnosed as CMT",
            "GALC-GENE-THERAPY-TRIALS: AAV9-GALC Phase I/II active; psychosine reduction combined approach most promising; watch trial results",
        ],
        "alias": (
            "GALC (Galactocerebrosidase; 669 aa; 14q31.3) encodes a lysosomal enzyme that cleaves "
            "galactosylceramide and psychosine (galactosylsphingosine). "
            "Biallelic LOF mutations cause Krabbe Disease / Globoid Cell Leukodystrophy (OMIM #245200). "
            "The pathological mechanism is primarily driven by psychosine accumulation — "
            "psychosine (NOT galactosylceramide) is directly cytotoxic to oligodendrocytes "
            "at nanomolar concentrations ('psychosine hypothesis'). "
            "Globoid cells (multinucleated macrophages with undegraded galactosylceramide inclusions) "
            "are pathognomonic in WM on brain biopsy. "
            "Infantile Krabbe (85%) presents at <6 months with extreme irritability, hypertonicity, "
            "fever spells, and rapid neurological decline; death by 2 yr without treatment. "
            "HSCT is the only proven intervention, and is ONLY effective if initiated pre-symptomatically; "
            "NBS (DBS GALC enzyme + confirmatory psychosine) is the only route to pre-symptomatic identification. "
            "AAV9-GALC gene therapy is in Phase I/II trials. "
            "Late-onset forms (juvenile, adult) are milder, presenting with spastic paraplegia + peripheral neuropathy."
        ),
    },

    # -- PLP1 — Proteolipid Protein 1 / Pelizaeus-Merzbacher Disease -----------------
    {
        "gene": "PLP1",
        "alt_name": "Proteolipid Protein 1 (PMD / SPG2)",
        "protein": (
            "PLP1 -- Xq22.2 X-linked -- ProteolipidProtein1-276aa -- "
            "PMD-Pelizaeus-Merzbacher-DUPLICATION-70pct-Most-Common -- "
            "Diffuse-Hypomyelination-MRI-T2-Bright-Everywhere -- "
            "MLPA-Mandatory-Detects-Duplications-Deletions -- "
            "Nystagmus-Birth-EARLIEST-Clinical-Clue-PMD"
        ),
        "locus": "Xq22.2",
        "protein_size": "276 aa",
        "inheritance": "X-linked",
        "age_of_onset": (
            "Classic PMD (duplication): neonatal nystagmus, hypotonia; motor milestones severely delayed; "
            "Connatal PMD (null mutations): most severe, neonatal, stridor, absent motor development; "
            "Spastic Paraplegia 2 (SPG2, deletion/mild mutations): childhood/adult progressive spasticity; "
            "Males always severely affected; female carriers may have mild spasticity"
        ),
        "key_biomarker": (
            "MRI: diffuse T2 hyperintensity throughout WM (including U-fibres) from birth — absent normal WM; "
            "T2 signal essentially like CSF throughout white matter (unmyelinated); "
            "MLPA: detects PLP1 duplication (70-80% of PMD) or deletion (SPG2); "
            "NCV: normal (central not peripheral); "
            "molecular: full PLP1 gene sequencing + MLPA + copy number analysis"
        ),
        "pathognomonic": (
            "Male infant with nystagmus at/shortly after birth + generalized hypotonia + "
            "MRI showing near-absent myelination (diffuse T2 WM signal, no normal WM) + "
            "MLPA confirmation of PLP1 duplication; "
            "nystagmus onset in first weeks/months of life is the EARLIEST clinical clue to PMD; "
            "nystagmus + hypomyelination on MRI + X-linked family history = PLP1 MLPA first"
        ),
        "treatment": (
            "No approved disease-modifying therapy; "
            "gene therapy approach: gene silencing / antisense oligonucleotides to reduce PLP1 overexpression (duplication model); "
            "allele-specific siRNA and ASO approaches in preclinical/early trials; "
            "stem cell therapy (NSC-03, Phase I/II past): oligodendrocyte precursor transplantation — limited benefit shown; "
            "symptomatic: antispasticity (baclofen, tizanidine), antiepileptics (LEV for seizures), "
            "respiratory support (may need NIV/ventilation in severe cases); "
            "physiotherapy, gastrostomy for feeding difficulties; "
            "genetic counselling: X-linked — female carriers screen + reproductive options"
        ),
        "critical_flags": [
            "PLP1-DUPLICATION-70pct: DUPLICATION of PLP1 is the most common PMD mutation (70-80%); overexpression → protein overload → UPR → oligodendrocyte death; NOT LOF",
            "PLP1-MLPA-MANDATORY: standard sequencing alone MISSES duplications/deletions; MLPA is mandatory first-tier test for PMD (after MRI diagnosis)",
            "PLP1-NYSTAGMUS-EARLIEST: nystagmus in first weeks/months = earliest clinical sign of PMD; pendular nystagmus + male infant + hypotonia = MLPA immediately",
            "PLP1-DELETION-SPG2: PLP1 deletion → Spastic Paraplegia 2 (SPG2), milder than duplication PMD; same gene, opposite molecular mechanism (haploinsufficiency vs. overexpression)",
            "PLP1-NULL-CONNATAL: null PLP1 mutations (frameshift) = most severe (Connatal PMD); paradoxically, complete absence worse than partial — DM20 isoform also absent",
            "PLP1-NO-THERAPY: no approved disease-modifying therapy; gene silencing (reduce overexpression) is the rational approach for duplication PMD; in trials",
            "PLP1-FEMALE-CARRIERS: female carriers (X-linked) usually asymptomatic but ~30% develop mild spasticity or cognitive issues; brain MRI may show subtle WM signal",
            "PLP1-ALLELIC-SPG2: PMD and SPG2 are allelic disorders (same PLP1 gene); duplication/null = PMD; deletion/mild hypomorphic = SPG2 — severity inversely relates to residual PLP1 function",
        ],
        "alias": (
            "PLP1 (Proteolipid Protein 1; 276 aa; Xq22.2) encodes the major structural protein of CNS myelin "
            "(comprising ~50% of myelin protein by mass) and its DM20 isoform (alternative splicing of exon 3B). "
            "PLP1 is X-linked, so males are primarily affected. "
            "PLP1 DUPLICATION (70–80% of PMD) → overexpression → endoplasmic reticulum stress + UPR → oligodendrocyte apoptosis → "
            "diffuse hypomyelination (Pelizaeus-Merzbacher Disease, PMD; OMIM #312080). "
            "PLP1 DELETION → loss of function → Spastic Paraplegia Type 2 (SPG2, OMIM #312920). "
            "NULL mutations → Connatal PMD (most severe). "
            "Classic PMD: neonatal nystagmus (earliest sign), hypotonia, absent/severely delayed myelination on MRI "
            "(near-total T2 WM signal — unmyelinated brain). "
            "MLPA is mandatory (standard sequencing misses duplications/deletions). "
            "No approved disease-modifying therapy; gene silencing (ASO/siRNA to reduce overexpression in duplication) in trials. "
            "Symptomatic management: antispasticity, antiepileptics, respiratory support."
        ),
    },

    # -- ABCD1 — ATP-Binding Cassette D1 / X-linked Adrenoleukodystrophy --------------
    {
        "gene": "ABCD1",
        "alt_name": "ALDP (X-linked Adrenoleukodystrophy)",
        "protein": (
            "ABCD1 -- Xq28 X-linked -- ALDP-745aa -- "
            "X-ALD-VLCFA-C26-Accumulation-Peroxisomal-Half-Transporter -- "
            "CCALD-Childhood-Cerebral-35pct-Parieto-Occipital-Gd-Enhancement -- "
            "Skysona-Elivaldogene-FDA2022-Gene-Therapy-CCALD -- "
            "Adrenal-Insufficiency-71pct-Males-Crisis-EMERGENCY-Hydrocortisone"
        ),
        "locus": "Xq28",
        "protein_size": "745 aa",
        "inheritance": "X-linked",
        "age_of_onset": (
            "CCALD (childhood cerebral ALD, 35-40%): age 4–8 yr, rapid inflammatory demyelination; "
            "AMN (adrenomyeloneuropathy, 40-45%): from 20yr, progressive myelopathy, axonal; "
            "Addison only (~20%): adrenal insufficiency without neurological disease; "
            "Adrenal insufficiency: 71% of all males — may precede neurological presentation"
        ),
        "key_biomarker": (
            "Plasma VLCFA (very-long-chain fatty acids): C26:0 elevated; C26:0/C22:0 ratio elevated; "
            "MOST IMPORTANT FIRST-LINE TEST; "
            "NBS: C26-lyso-PC on DBS (30+ US states); "
            "MRI: CCALD — parieto-occipital WM T2 signal with Gd enhancement at leading edge (active demyelination); "
            "Loes score (MRI severity, 0–34); NRS (neurological rating scale 0–25); "
            "adrenal function: morning cortisol, ACTH stimulation test; "
            "adrenal insufficiency screen MANDATORY in ALL males with ABCD1 mutation"
        ),
        "pathognomonic": (
            "Boy 4–8 yr with behavioural change + school failure + posterior cerebral + parieto-occipital MRI lesion "
            "with Gd enhancement at leading edge + plasma VLCFA elevated = CCALD EMERGENCY (treatment window is NARROW); "
            "Loes ≤9 + NRS ≤1 + Gd enhancement = HSCT/Skysona WINDOW — act within weeks; "
            "AMN: young man 20yr+ with progressive spastic paraplegia + sensory ataxia + peripheral neuropathy + elevated VLCFA"
        ),
        "treatment": (
            "CCALD (early, Loes ≤9, NRS ≤1, Gd+): "
            "  HSCT (allogeneic, if HLA-matched donor) — Level A; arrests inflammatory demyelination; "
            "  Skysona (elivaldogene autotemcel, FDA Aug 2022) — ex vivo lentiviral gene therapy; for CCALD boys 4–17 yr WITH Gd enhancement, when no HLA-matched donor available; "
            "  REMS program; close monitoring for insertional mutagenesis (haematological malignancy risk); "
            "LATE CCALD (Loes >9) or AMN: NO HSCT benefit; "
            "Lorenzo's Oil (VLCFA-lowering diet + erucic acid + oleic acid): REDUCES plasma VLCFA but NO proven neurological benefit; "
            "  evidence only in pre-symptomatic males to delay CCALD onset (controversial); "
            "Adrenal insufficiency: hydrocortisone + fludrocortisone MANDATORY; stress doses (3× normal) for illness/surgery; "
            "SURVEILLANCE: ALL boys with ABCD1 — annual/6-monthly brain MRI from age 4–12yr; annual adrenal function; "
            "DRUG WARNING: PHT/fosphenytoin ABSOLUTE CI — CYP3A4 induction → cortisol metabolism → adrenal crisis; CBZ/OXC relative CI; LEV first-line AED"
        ),
        "critical_flags": [
            "ABCD1-GENOTYPE-NO-PHENOTYPE: SAME ABCD1 mutation can cause CCALD, AMN, or Addison only in different males (even within same family); cannot predict phenotype from genotype",
            "ABCD1-ANNUAL-MRI-ALL-BOYS: ALL ABCD1 males need annual/6-monthly brain MRI age 4–12yr to detect early CCALD; MRI identifies inflammatory CCALD before symptoms appear",
            "ABCD1-CCALD-NARROW-WINDOW: CCALD treatment window is NARROW — Loes ≤9 + NRS ≤1 + Gd enhancement; once Loes >9 or NRS >1, transplant does NOT help; EMERGENCY referral within weeks",
            "ABCD1-ADRENAL-INSUFFICIENCY-71pct: adrenal insufficiency in 71% of males; adrenal crisis = life-threatening emergency; ALL ABCD1 males need adrenal function testing; stress-dose hydrocortisone for illness",
            "ABCD1-PHT-ABSOLUTE-CI: phenytoin/fosphenytoin ABSOLUTE CONTRAINDICATED in X-ALD — CYP3A4 induction → increased cortisol catabolism → acute adrenal crisis; use LEV as first-line AED",
            "ABCD1-SKYSONA-FDA2022: elivaldogene autotemcel (Skysona) FDA approved Aug 2022 for CCALD boys 4–17yr with active cerebral disease; REMS program due to malignancy risk; used when no HLA match",
            "ABCD1-LORENZOS-OIL-NOT-NEUROLOGICAL: Lorenzo's Oil lowers plasma VLCFA but does NOT improve or halt neurological progression in symptomatic patients; evidence only for pre-symptomatic delay",
            "ABCD1-AMN-NOT-HSCT: AMN (adrenomyeloneuropathy) is axonal degeneration NOT inflammatory; HSCT/Skysona NOT indicated for AMN; supportive management (antispasticity, physiotherapy)",
        ],
        "alias": (
            "ABCD1 (ATP-Binding Cassette Subfamily D Member 1 / ALDP; 745 aa; Xq28) encodes a peroxisomal "
            "membrane half-transporter that imports very-long-chain fatty acids (VLCFAs, ≥C22) into the peroxisome for β-oxidation. "
            "ABCD1 LOF → VLCFA accumulation (especially C26:0) in plasma, adrenal cortex, nervous system. "
            "X-linked: hemizygous males severely affected; heterozygous females may develop mild AMN-like features. "
            "Clinical phenotypes in males: "
            "(1) CCALD (35–40%): childhood 4–8yr, rapidly progressive inflammatory demyelination of posterior cerebral WM, "
            "Gd enhancement at active edge — EMERGENCY requiring HSCT or Skysona within narrow window; "
            "(2) AMN (40–45%): adult progressive axonal myelopathy + peripheral neuropathy; not HSCT-responsive; "
            "(3) Addison-only (~20%): adrenal insufficiency without neurological disease. "
            "Adrenal insufficiency: 71% of males; adrenal crisis is a life-threatening emergency. "
            "Genotype DOES NOT predict phenotype. "
            "CCALD treatment: HSCT (Level A; HLA-matched) or Skysona (FDA2022; no HLA match); window Loes ≤9, NRS ≤1. "
            "Phenytoin/fosphenytoin are ABSOLUTELY contraindicated (CYP3A4 → adrenal crisis). "
            "NBS: C26-lyso-PC on DBS in 30+ US states."
        ),
    },

    # -- ASPA — Aspartoacylase / Canavan Disease ------------------------------------
    {
        "gene": "ASPA",
        "alt_name": "Aspartoacylase (Canavan Disease)",
        "protein": (
            "ASPA -- 17p13.2 AR -- Aspartoacylase-313aa -- "
            "Canavan-Disease-NAA-N-Acetylaspartate-Elevated-MOST-SPECIFIC-MRS-Biomarker -- "
            "U-Fibres-Involved-EARLY-Unlike-Most-Leukodystrophies -- "
            "Spongy-Degeneration-WM-Vacuolization -- "
            "Ashkenazi-Jewish-Founder-pGlu285Ala-pTyr231X"
        ),
        "locus": "17p13.2",
        "protein_size": "313 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Neonatal-infantile form: onset 3–6 months, macrocephaly at birth, hypotonia then spasticity; "
            "Mild/juvenile form: some residual ASPA activity, less severe; "
            "Macrocephaly: usually present at birth or develops within first months"
        ),
        "key_biomarker": (
            "NAA (N-acetylaspartate) ELEVATED: urine organic acids (NAA peak); plasma NAA; "
            "MRS (magnetic resonance spectroscopy): NAA peak markedly elevated = MOST SPECIFIC MRS BIOMARKER in all leukodystrophies; "
            "leukocyte/fibroblast ASPA enzyme near-zero; "
            "MRI: diffuse T2 WM signal with U-fibre involvement + globus pallidus signal; "
            "macrocephaly (OFC >97th centile); "
            "molecular: ARSA biallelic mutations (Ashkenazi: p.Glu285Ala, p.Tyr231X; non-Ashkenazi: p.Ala305Glu)"
        ),
        "pathognomonic": (
            "Macrocephalic infant 3–6 months with hypotonia + head lag + "
            "MRI diffuse T2 WM signal involving U-fibres (not spared) + globus pallidus + "
            "MRS: NAA peak elevated (2–3× normal) = Canavan until proven otherwise; "
            "NAA elevated in urine organic acids is highly specific; "
            "U-fibre involvement early distinguishes Canavan from many leukodystrophies where U-fibres are spared; "
            "macrocephaly at birth + leukodystrophy MRI = Canavan OR Alexander (different MRI pattern)"
        ),
        "treatment": (
            "No approved disease-modifying therapy; "
            "Gene therapy (AAV-based ASPA): intracranial/IV delivery; Phase I/II trials (Aspa-101 rAAV9); "
            "glyceryl triacetate (GTA): oral acetate supplementation to partially restore myelin synthesis — "
            "biochemical improvement, limited clinical data; "
            "lithium (reduces NAA production by inhibiting NAA synthetase NAT8L) — Phase I/II trial data; "
            "symptomatic: antiepileptics (LEV), antispasticity (baclofen), gastrostomy, physiotherapy; "
            "genetic counselling: 1/40 Ashkenazi carrier frequency; "
            "Ashkenazi Jewish population NBS/carrier screening programs"
        ),
        "critical_flags": [
            "ASPA-NAA-MRS-PATHOGNOMONIC: elevated NAA on MRS is the MOST SPECIFIC MRS biomarker in all leukodystrophies; Canavan = dramatically elevated NAA; no other common leukodystrophy has this",
            "ASPA-U-FIBRES-EARLY: U-fibres (subcortical arcuate fibres) involved EARLY in Canavan; most leukodystrophies spare U-fibres early — U-fibre involvement narrows DDx to Canavan + Alexander",
            "ASPA-MACROCEPHALY-BIRTH: macrocephaly at/before birth + leukodystrophy = Canavan or Alexander; both have distinct MRI patterns and molecular diagnosis",
            "ASPA-ASHKENAZI-FOUNDER: p.Glu285Ala (c.854A>C) + p.Tyr231X (c.693C>A) = 97% Ashkenazi alleles; carrier frequency 1/40 Ashkenazi; targeted panel highly efficient in this population",
            "ASPA-NAA-URINE: urine organic acids show elevated NAA peak — easily detected on standard organic acid screen; always check organic acids in leukodystrophy",
            "ASPA-GLOBUS-PALLIDUS-SIGNAL: globus pallidus T2 signal (bilateral) on MRI in Canavan — unusual feature; combined with U-fibre involvement distinguishes from other leukodystrophies",
            "ASPA-GENE-THERAPY-TRIALS: rAAV9-ASPA Phase I/II active; lithium (reduces NAA via ASPA substrate reduction) and glyceryl triacetate (acetate supplement) also in trials",
            "ASPA-SPONGY-DEGENERATION: neuropathology = spongy myelinopathy (vacuolization of WM) due to NAA-driven osmotic water influx into oligodendrocytes; gross-pathologic description",
        ],
        "alias": (
            "ASPA (Aspartoacylase; 313 aa; 17p13.2) encodes a cytosolic enzyme predominantly expressed in "
            "oligodendrocytes that hydrolyses N-acetylaspartate (NAA) into aspartate + acetate. "
            "The acetate product is essential for myelin lipid synthesis in oligodendrocytes. "
            "Biallelic ASPA LOF → NAA accumulates in brain, CSF, urine (Canavan Disease; OMIM #271900). "
            "NAA accumulation → osmotic water influx into oligodendrocytes → vacuolization (spongy degeneration) → demyelination. "
            "U-fibres are involved early (distinctive from many leukodystrophies where U-fibres are initially spared). "
            "Macrocephaly is present at/before birth. "
            "MRS: dramatically elevated NAA peak — the most specific MRS biomarker in all leukodystrophies. "
            "Ashkenazi Jewish founder alleles: p.Glu285Ala + p.Tyr231X (carrier frequency 1/40). "
            "No approved therapy; AAV9-ASPA gene therapy in Phase I/II; lithium (reduces NAA via ASPA pathway) in trials. "
            "Symptomatic management: antiepileptics, antispasticity, gastrostomy."
        ),
    },

    # -- GFAP — Glial Fibrillary Acidic Protein / Alexander Disease -----------------
    {
        "gene": "GFAP",
        "alt_name": "Glial Fibrillary Acidic Protein (Alexander Disease)",
        "protein": (
            "GFAP -- 17q21.31 AD -- GlialFibAcidProtein-432aa -- "
            "Alexander-Disease-ALL-Mutations-DOMINANT-GOF-NOT-LOF-CRITICAL -- "
            "Rosenthal-Fibres-Perivascular-Subpial-PATHOGNOMONIC -- "
            "GFAP-Protein-CSF-Blood-Elevated-Astrocytic-Injury-Marker -- "
            "Frontal-Dominant-WM-Plus-Basal-Ganglia-Brainstem-MRI"
        ),
        "locus": "17q21.31",
        "protein_size": "432 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Infantile (most common): onset 0–2 yr, macrocephaly, seizures (often early spasms), psychomotor delay; "
            "Juvenile: onset 4–14 yr, progressive neurological decline, bulbar symptoms begin; "
            "Adult (Type II): progressive bulbar dysfunction, palatal myoclonus, cerebellar/spinal involvement; "
            "Most mutations are DE NOVO"
        ),
        "key_biomarker": (
            "GFAP protein elevated in CSF (most sensitive) and blood — astrocytic injury marker; "
            "MRI: frontal-dominant WM T2 signal + enhancement (infantile) + "
            "basal ganglia + thalamus + brainstem involvement; "
            "'garland' pattern of enhancement in infantile; "
            "Rosenthal fibres: eosinophilic, perivascular/subpial aggregates on brain biopsy/autopsy (PATHOGNOMONIC); "
            "molecular: heterozygous GFAP missense mutation (or rarely small in-frame); "
            "IMPORTANT: NO ENZYME DEFICIENCY — GFAP is a structural protein, no metabolic test"
        ),
        "pathognomonic": (
            "Infantile: macrocephaly + seizures + psychomotor delay + "
            "MRI frontal-dominant T2 WM signal + basal ganglia/thalamic signal + gadolinium enhancement + "
            "GFAP protein elevated in CSF + heterozygous GFAP mutation; "
            "Rosenthal fibres on brain biopsy (perivascular eosinophilic inclusions); "
            "Adult Type II: palatal myoclonus + progressive bulbar dysfunction + medullary atrophy = "
            "Alexander Type II until proven otherwise; "
            "ALL Alexander disease mutations are GOF (dominant negative or toxic gain) — LOF has NO disease"
        ),
        "treatment": (
            "No approved disease-modifying therapy; "
            "GFAP-lowering strategies (antisense oligonucleotides / siRNA targeting GFAP mRNA) in preclinical trials; "
            "rationale: reducing mutant GFAP burden reduces Rosenthal fibre load; "
            "ceftriaxone (upregulates GLT-1 glutamate transporter) — anecdotal/open-label data; "
            "seizures: LEV, VPA (frontal seizures; spasms in infantile — vigabatrin/ACTH); "
            "anti-spasticity: baclofen; "
            "feeding: gastrostomy for bulbar dysfunction; "
            "palatal myoclonus: clonazepam or sodium valproate; "
            "genetic counselling: most mutations de novo; empirical recurrence risk ~1% (germline mosaicism)"
        ),
        "critical_flags": [
            "GFAP-GOF-NOT-LOF: ALL Alexander disease mutations are DOMINANT GAIN-OF-FUNCTION (toxic missense/in-frame); GFAP LOF would NOT cause Alexander disease; critical distinction from lysosomal leukodystrophies",
            "GFAP-DE-NOVO-MOST: most GFAP mutations are de novo; family history often negative; do NOT exclude Alexander because parents are unaffected",
            "GFAP-ROSENTHAL-PATHOGNOMONIC: Rosenthal fibres (eosinophilic perivascular/subpial aggregates) on brain biopsy/autopsy = PATHOGNOMONIC; seen by H&E or GFAP immunostain",
            "GFAP-CSF-BLOOD-BIOMARKER: CSF GFAP protein markedly elevated (astrocytic injury); serum GFAP also elevated; can track disease activity; GFAP is THE Alexander disease biomarker",
            "GFAP-FRONTAL-MRI: frontal WM predominance in infantile Alexander (anterior > posterior — OPPOSITE of X-ALD and Krabbe which are posterior-predominant); useful MRI clue for DDx",
            "GFAP-PALATAL-MYOCLONUS-TYPE2: palatal myoclonus + progressive bulbar dysfunction in adult = Alexander Type II hallmark; often misdiagnosed as ALS/bulbar palsy; MRI + GFAP CSF",
            "GFAP-MACROCEPHALY: macrocephaly in infantile Alexander (also in Canavan); combined with FRONTAL (not posterior) WM signal distinguishes from Canavan (posterior U-fibre + NAA elevated)",
            "GFAP-NO-ENZYME-TEST: Alexander disease has NO enzymatic/metabolic diagnostic test; diagnosis = MRI pattern + CSF GFAP + GFAP genetic sequencing",
        ],
        "alias": (
            "GFAP (Glial Fibrillary Acidic Protein; 432 aa; 17q21.31) encodes the principal intermediate filament "
            "of mature astrocytes. Heterozygous missense mutations cause Alexander Disease (OMIM #203450) "
            "by a DOMINANT GAIN-OF-FUNCTION mechanism — all known pathogenic GFAP mutations are dominant (missense or small in-frame); "
            "GFAP haploinsufficiency (LOF) does NOT cause disease. "
            "Mutant GFAP misfolds and aggregates, forming Rosenthal fibres "
            "(eosinophilic, electron-dense inclusions perivascular and subpial) — pathognomonic on brain biopsy. "
            "Infantile Alexander (most common): macrocephaly, frontal-dominant T2 WM signal with enhancement, "
            "seizures from infancy; most mutations de novo. "
            "Adult Type II: progressive bulbar dysfunction + palatal myoclonus + medullary/cerebellar atrophy. "
            "CSF GFAP protein markedly elevated — the disease biomarker. "
            "No approved therapy; GFAP ASO/siRNA (reduce mutant protein load) in preclinical development."
        ),
    },

    # -- EIF2B5 — eIF2B epsilon subunit / Vanishing White Matter Disease (VWM) -------
    {
        "gene": "EIF2B5",
        "alt_name": "eIF2B epsilon (Vanishing White Matter)",
        "protein": (
            "EIF2B5 -- 3q27.1 AR -- eIF2Bepsilon-712aa -- "
            "VWM-Vanishing-White-Matter-Stress-Triggered-Episodes-EMERGENCY -- "
            "ISR-Integrated-Stress-Response-Hypersensitivity -- "
            "ISRIB-ISR-Inhibitor-Most-Promising-Experimental-Therapy -- "
            "Ovarioleukodystrophy-Premature-Ovarian-Failure-Females"
        ),
        "locus": "3q27.1",
        "protein_size": "712 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Congenital: neonatal severe, multi-organ, null mutations; "
            "Infantile: 1–3 yr most common, episodic ataxia; "
            "Juvenile: 3–16 yr; "
            "Adult: >16 yr, dementia + psychiatric; "
            "Stress triggers: febrile illness → MOST COMMON trigger; minor head trauma; emotional shock; "
            "EIF2B1-5 mutations; EIF2B5 (epsilon, largest subunit) most commonly mutated"
        ),
        "key_biomarker": (
            "MRI: diffuse WM T2 signal with CSF-like (fluid-equivalent) signal on FLAIR in WM = rarefaction/'vanishing'; "
            "FLAIR WM signal isointense to CSF = pathognomonic VWM; "
            "proton MRS: lactate peak in WM (anaerobic glycolysis); NAA reduced in WM; "
            "CSF: may show elevated oligoclonal bands in some; "
            "molecular: biallelic EIF2B1-5 mutations; EIF2B5 most common; "
            "FSH/LH elevated in females (premature ovarian failure); "
            "genetic testing essential — NO metabolic/enzyme biomarker for VWM"
        ),
        "pathognomonic": (
            "Child with previously normal/near-normal development who has acute neurological DETERIORATION "
            "triggered by febrile illness or minor head trauma → "
            "MRI shows CSF-equivalent T2 signal in WM (FLAIR WM = CSF signal) = VWM until proven otherwise; "
            "episodes may partially recover; "
            "female with young-adult premature ovarian failure + WM MRI signal = ovarioleukodystrophy (EIF2B); "
            "congenital VWM: neonatal onset, death weeks-months, most severe biallelic null mutations"
        ),
        "treatment": (
            "NO approved disease-modifying therapy; "
            "ISRIB (integrated stress response inhibitor): MOST PROMISING trial drug; "
            "reduces ISR hypersensitivity by stabilising eIF2B; Phase I/II trials recruiting; "
            "PREVENTION OF STRESS TRIGGERS: "
            "  AGGRESSIVELY treat fever (antipyretics at onset of any fever — paracetamol + ibuprofen alternating); "
            "  NO contact sports (prevent head trauma); "
            "  emotional stress management; "
            "  written emergency protocol for fever management given to all caregivers + schools; "
            "  immunisations on schedule (vaccinations may trigger mild fever — antipyretics around time of vaccination); "
            "corticosteroids in acute deterioration episode: dexamethasone (some centres — limited evidence, anti-inflammatory); "
            "seizure management: LEV or VPA; "
            "premature ovarian failure: HRT (oestrogen + progesterone) if POF confirmed; "
            "genetic counselling: AR; 25% sibling recurrence"
        ),
        "critical_flags": [
            "EIF2B5-STRESS-TRIGGERS-EMERGENCY: fever, minor head trauma, emotional shock → ACUTE NEUROLOGICAL DETERIORATION in VWM; fever management is LIFE-SAVING; written emergency plan for carers/schools",
            "EIF2B5-ANTIPYRETICS-AGGRESSIVELY: start antipyretics (paracetamol/ibuprofen) at FIRST SIGN of fever; do NOT wait for high temperature; aim to keep temperature <37.8°C; this is primary prevention",
            "EIF2B5-FLAIR-CSF-SIGNAL: WM signal isointense to CSF on FLAIR MRI = rarefaction/vanishing; this pattern is essentially pathognomonic for VWM; WM has become fluid-filled cavities",
            "EIF2B5-OVARIOLEUKODYSTROPHY: premature ovarian failure (POF) in EIF2B mutations; female with WM disease + POF = EIF2B panel; also seen with ADAR1 but different MRI",
            "EIF2B5-ISRIB-TRIALS: ISRIB stabilises eIF2B complex → reduces ISR hypersensitivity; most rationally targeted VWM therapy; Phase I/II trials; watch results",
            "EIF2B5-PARTIAL-RECOVERY: VWM episodes may partially reverse after acute trigger resolved; partial recovery is diagnostic — suggests ongoing brain plasticity in mild mutations",
            "EIF2B5-NO-CONTACT-SPORTS: minor head trauma can trigger devastating deterioration; NO contact sports/activities with head trauma risk; helmets for cycling/skateboarding; school medical plan",
            "EIF2B5-EIF2B1-4-ALSO: VWM caused by biallelic mutations in EIF2B1 (alpha), EIF2B2 (beta), EIF2B3 (gamma), EIF2B4 (delta), EIF2B5 (epsilon); sequence all 5 if VWM suspected; EIF2B5 most common",
        ],
        "alias": (
            "EIF2B5 (eukaryotic initiation factor 2B epsilon subunit; 712 aa; 3q27.1) encodes the catalytic "
            "epsilon subunit of the eIF2B guanine-nucleotide exchange factor (GEF) complex. "
            "eIF2B regenerates active eIF2-GTP from eIF2-GDP, enabling translation initiation. "
            "Under stress, eIF2α is phosphorylated → inhibits eIF2B → reduced translation → integrated stress response (ISR). "
            "Biallelic EIF2B5 LOF → constitutively impaired eIF2B → hypersensitised ISR → "
            "oligodendrocyte and astrocyte fragility → "
            "Vanishing White Matter Disease (VWM; OMIM #603896). "
            "VWM episodes: triggered by fever, minor head trauma, or emotional shock → "
            "acute neurological deterioration ± partial recovery. "
            "MRI: FLAIR WM signal equivalent to CSF (WM rarefaction/vanishing) — pathognomonic. "
            "Ovarioleukodystrophy: premature ovarian failure in females — a key associated feature. "
            "VWM caused by biallelic mutations in any of EIF2B1-5 (epsilon most common). "
            "Fever prevention (aggressive antipyretics) is the cornerstone of management; no head trauma. "
            "ISRIB (ISR inhibitor) in Phase I/II trials — the most promising targeted therapy."
        ),
    },

    # -- POLR3A — RNA Polymerase III Subunit A / POLR3-Related Leukodystrophy / HLD7 --
    {
        "gene": "POLR3A",
        "alt_name": "RNA Pol III Subunit A (POLR3-Related / HLD7)",
        "protein": (
            "POLR3A -- 10q22.3 AR -- RNApolIIIsubunitA-1390aa -- "
            "POLR3-Related-Leukodystrophy-HLD7-Hypomyelinating -- "
            "Dental-Abnormalities-Hypodontia-LEUKODYSTROPHY-TRIAD-PATHOGNOMONIC -- "
            "RNA-Pol-III-tRNA-Synthesis-Impaired-Oligodendrocyte-Hypomyelination -- "
            "Cerebellar-Atrophy-Thinned-Corpus-Callosum-Myopia-Hypogonadism"
        ),
        "locus": "10q22.3",
        "protein_size": "1390 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Onset: early childhood 1–6 yr; "
            "Presentation: motor delay (late walking) + cerebellar ataxia + dental abnormalities; "
            "Slower progression than most leukodystrophies — patients may live to adulthood; "
            "Myopia common (often severe); "
            "Hypogonadotropic hypogonadism in some (delayed/absent puberty)"
        ),
        "key_biomarker": (
            "MRI: diffuse T2 WM signal (hypomyelination) + cerebellar atrophy + thin corpus callosum; "
            "T2 hypointensity of globus pallidus, ventral pons, dentate nuclei (unusual feature — iron/mineralisation?); "
            "NCV: normal or mildly abnormal (central not peripheral); "
            "ophthalmic assessment: myopia (often severe); "
            "dental X-ray: delayed dentition, hypodontia, oligodontia; "
            "hormonal: FSH/LH/oestrogen/testosterone (hypogonadism); "
            "molecular: biallelic POLR3A mutations (most common: p.Gly672Glu + intronic splice c.1909+22G>A compound heterozygous)"
        ),
        "pathognomonic": (
            "Child with motor delay (late walking age 2–4yr) + cerebellar ataxia + "
            "MRI hypomyelination (diffuse T2 WM) + cerebellar atrophy + thin corpus callosum + "
            "DENTAL ABNORMALITIES (hypodontia, delayed dentition) = POLR3-related leukodystrophy; "
            "dental abnormalities in a child with leukodystrophy is the strongest clinical clue; "
            "T2 hypointensity of globus pallidus + ventral pons + dentate = characteristic POLR3 signal; "
            "additional features: myopia (ophthalmologist) + hypogonadism (delayed puberty)"
        ),
        "treatment": (
            "No approved disease-modifying therapy; "
            "RNA Pol III pathway targeting in research: "
            "  tRNA supplement approaches; "
            "  ASO/small molecule approaches targeting POLR3A intronic splice mutation (c.1909+22G>A) in development; "
            "  modafinil (wakefulness agent) used empirically for fatigue/cognitive symptoms; "
            "symptomatic: physiotherapy + occupational therapy (cerebellar ataxia management); "
            "myopia: early ophthalmic correction (glasses/contact lenses); "
            "hypogonadism: hormone replacement therapy (oestrogen/testosterone) if confirmed; "
            "dental: orthodontic consultation early; dental implants/prosthetics for hypodontia; "
            "antiepileptics: seizures uncommon but LEV if needed; "
            "genetic counselling: AR; 25% sibling recurrence; "
            "allelic: POLR3B (HLD8), POLR3-related panel includes POLR3A, POLR3B, POLR1C, POLR3K"
        ),
        "critical_flags": [
            "POLR3A-DENTAL-PATHOGNOMONIC: dental abnormalities (hypodontia, oligodontia, delayed dentition) in a leukodystrophy patient = POLR3-related leukodystrophy FIRST; request dental X-ray in every unexplained leukodystrophy",
            "POLR3A-HYPOMYELINATION-NOT-DEMYELINATION: POLR3 leukodystrophy is HYPOMYELINATION (myelin never formed properly), not demyelination (myelin formed then lost); different from ARSA/GALC which are demyelinating",
            "POLR3A-SLOWER-PROGRESSION: POLR3-related leukodystrophy progresses slowly; patients survive to adulthood (unlike infantile ARSA/GALC); functional level depends on mutation severity",
            "POLR3A-MYOPIA: severe myopia is a common associated feature; ophthalmologist referral mandatory; myopia + cerebellar ataxia + WM signal = POLR3 panel",
            "POLR3A-INTRONIC-SPLICE: c.1909+22G>A deep intronic splice mutation (creates pseudoexon) — COMMON POLR3A allele; missed by exome sequencing; genome sequencing or RNA studies required for full mutation detection",
            "POLR3A-GLOBUS-PALLIDUS-T2-LOW: T2 hypointensity of globus pallidus + ventral pons + dentate nuclei is a characteristic POLR3 MRI pattern; most leukodystrophies have T2 HIGH signal in these structures",
            "POLR3A-POLR3-PANEL: POLR3-related leukodystrophy caused by POLR3A (HLD7), POLR3B (HLD8), POLR1C (HLD3), POLR3K; always sequence full POLR3 gene panel if one gene negative",
            "POLR3A-HYPOGONADISM: hypogonadotropic hypogonadism (delayed/absent puberty) in both sexes — FSH/LH/sex hormones; HRT for delayed puberty; a distinguishing feature from other hypomyelinating leukodystrophies",
        ],
        "alias": (
            "POLR3A (RNA Polymerase III Subunit A; 1390 aa; 10q22.3) encodes the largest subunit of RNA Polymerase III, "
            "which transcribes small non-coding RNAs essential for translation: 5S rRNA, all cytoplasmic tRNAs, "
            "7SL RNA, and U6 snRNA. "
            "Biallelic POLR3A LOF mutations impair tRNA synthesis → global translational slowdown → "
            "hypomyelination (myelin never properly formed) — POLR3-Related Leukodystrophy / HLD7 (OMIM #607694). "
            "Clinical triad: (1) hypomyelinating leukodystrophy (diffuse T2 WM + thin corpus callosum + cerebellar atrophy); "
            "(2) dental abnormalities (hypodontia, delayed dentition) — most specific clinical clue; "
            "(3) myopia. Additional: hypogonadotropic hypogonadism. "
            "T2 hypointensity of globus pallidus + ventral pons + dentate nuclei is characteristic (opposite to most leukodystrophies). "
            "Deep intronic splice mutation c.1909+22G>A is the most common POLR3A allele and requires genome sequencing to detect. "
            "Slow progression; patients survive to adulthood. "
            "Allelic disorders: POLR3B (HLD8), POLR1C (HLD3), POLR3K — sequence all if clinical suspicion. "
            "No approved therapy; ASO targeting intronic splice mutation in development."
        ),
    },
]


def _make_cohort(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    cohort = []
    for i in range(40):
        age = rng.randint(1, 35)
        severity = rng.choice(["mild", "moderate", "severe", "severe"])

        if gene == "ARSA":
            form = rng.choice(["late-infantile", "juvenile", "adult"])
            feature = rng.choice([
                "periventricular tigroid T2 signal", "urine sulfatides elevated",
                "ARSA enzyme <5%", "walking regression", "psychiatric onset (adult)"
            ])
            therapy = "Libmeldy (pre-symptomatic)" if form == "late-infantile" else rng.choice(["symptomatic", "HSCT (juvenile)"])
        elif gene == "GALC":
            form = rng.choice(["infantile", "late-infantile", "adult"])
            age = rng.randint(0, 2) if form == "infantile" else rng.randint(3, 40)
            feature = rng.choice([
                "irritability + hypertonicity", "near-zero GALC enzyme", "psychosine elevated",
                "cerebellar WM T2 signal", "progressive spastic paraplegia (adult)"
            ])
            therapy = "HSCT (pre-symptomatic)" if form == "infantile" else "palliative/supportive"
        elif gene == "PLP1":
            age = rng.randint(0, 5)
            feature = rng.choice([
                "nystagmus at birth", "diffuse hypomyelination MRI", "PLP1 duplication MLPA",
                "absent myelination on MRI", "severe motor delay"
            ])
            therapy = "symptomatic (no approved therapy)"
        elif gene == "ABCD1":
            form = rng.choice(["CCALD", "AMN", "Addison-only"])
            age = rng.randint(4, 45)
            if form == "CCALD":
                age = rng.randint(4, 12)
            feature = rng.choice([
                "posterior WM Gd enhancement", "VLCFA C26:0 elevated", "adrenal insufficiency",
                "Loes score ≤9 → HSCT window", "progressive myelopathy (AMN)"
            ])
            therapy = rng.choice(["HSCT (early CCALD)", "Skysona (no HLA match)", "hydrocortisone (Addison)", "supportive AMN"])
        elif gene == "ASPA":
            age = rng.randint(0, 3)
            feature = rng.choice([
                "macrocephaly at birth", "NAA elevated MRS", "urine NAA (organic acids)",
                "U-fibre T2 signal", "diffuse WM + globus pallidus"
            ])
            therapy = "supportive (gene therapy trials)"
        elif gene == "GFAP":
            form = rng.choice(["infantile", "adult-type2"])
            age = rng.randint(0, 2) if form == "infantile" else rng.randint(20, 60)
            feature = rng.choice([
                "frontal WM T2 + enhancement", "GFAP protein CSF elevated",
                "de novo GFAP mutation", "Rosenthal fibres biopsy", "palatal myoclonus (adult)"
            ])
            therapy = "symptomatic (ASO trials)"
        elif gene == "EIF2B5":
            feature = rng.choice([
                "WM FLAIR = CSF signal", "fever-triggered acute deterioration",
                "head trauma episode", "premature ovarian failure (female)", "ISR hypersensitivity"
            ])
            therapy = rng.choice(["fever protocol + antipyretics", "ISRIB (trial)", "supportive + HRT (POF)"])
        elif gene == "POLR3A":
            age = rng.randint(1, 20)
            feature = rng.choice([
                "dental hypodontia + leukodystrophy", "hypomyelination + cerebellar atrophy",
                "severe myopia", "hypogonadism + WM disease", "T2 globus pallidus hypointensity"
            ])
            therapy = "symptomatic (hormone replacement + ophthalmic + physiotherapy)"
        else:
            feature = "hypomyelination WM"
            therapy = "supportive"

        cohort.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "age": age,
            "gene": gene,
            "severity": severity,
            "key_feature": feature,
            "current_therapy": therapy,
        })
    return cohort


# ---------- API endpoint functions -------------------------------------------

def overview() -> dict:
    total = 0
    severe_count = 0
    avg_age_sum = 0
    gene_summary = []
    for idx, g in enumerate(LEUKODYSTROPHY_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        total += len(cohort)
        severe_count += sum(1 for p in cohort if p["severity"] == "severe")
        avg_age_sum += sum(p["age"] for p in cohort)
        gene_summary.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "n_patients": len(cohort),
        })
    avg_age = round(avg_age_sum / total, 1)
    return {
        "atlas": "Hereditary-Leukodystrophy-Atlas",
        "aggregate_stats": {
            "total_patients": total,
            "genes_covered": len(LEUKODYSTROPHY_GENES),
            "avg_age_at_diagnosis_yr": avg_age,
            "severe_cases_pct": round(100 * severe_count / total, 1),
            "seed_range": f"{SEED_BASE}–{SEED_BASE + len(LEUKODYSTROPHY_GENES) - 1}",
        },
        "gene_summary": gene_summary,
        "key_clinical_distinctions": [
            "ARSA-PSEUDODEFICIENCY: low ARSA enzyme + NORMAL sulfatide urine = pseudodeficiency (N350S+I179S), NOT MLD; always confirm with sulfatide urine BEFORE diagnosing MLD",
            "GALC-HSCT-PRESYMPTOMATIC-ONLY: HSCT ONLY works PRE-SYMPTOMATIC in Krabbe; NBS is mandatory to identify pre-symptomatic infantile; symptomatic infantile Krabbe = palliative",
            "PLP1-MLPA-MANDATORY: standard sequencing misses PLP1 duplication (70% PMD); MLPA first-tier test; nystagmus at birth in male = PLP1 MLPA immediately",
            "ABCD1-CCALD-NARROW-WINDOW: CCALD treatment window is NARROW (Loes ≤9, NRS ≤1, Gd+); every week matters; annual MRI surveillance mandatory for all ABCD1 boys age 4–12yr",
            "ABCD1-PHT-ABSOLUTE-CI: phenytoin/fosphenytoin ABSOLUTELY CONTRAINDICATED in X-ALD (CYP3A4 → cortisol catabolism → adrenal crisis); use LEV as AED",
            "ASPA-NAA-MRS-MOST-SPECIFIC: elevated NAA on MRS = Canavan until proven otherwise; most specific MRS biomarker in all leukodystrophies",
            "GFAP-GOF-NOT-LOF: ALL Alexander disease mutations are DOMINANT GOF; GFAP LOF does NOT cause Alexander; de novo most common; no enzyme test",
            "EIF2B5-FEVER-EMERGENCY: fever in VWM patient = EMERGENCY; aggressive antipyretics (target <37.8°C) prevents devastating deterioration episodes; written emergency protocol mandatory",
            "POLR3A-DENTAL-KEY-CLUE: dental abnormalities (hypodontia, delayed dentition) in leukodystrophy = POLR3-related first; intronic splice c.1909+22G>A requires genome sequencing to detect",
            "ABCD1-ADRENAL-MANDATORY: adrenal insufficiency in 71% ABCD1 males; screen ALL males; stress-dose hydrocortisone for illness/surgery; adrenal crisis is preventable death",
        ],
    }


def breakdown() -> dict:
    result = []
    for idx, g in enumerate(LEUKODYSTROPHY_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        severities = {}
        for p in cohort:
            severities[p["severity"]] = severities.get(p["severity"], 0) + 1
        result.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "severity_distribution": severities,
            "n_patients": len(cohort),
            "patients": cohort[:5],
        })
    return {"genes": result, "total_genes": len(LEUKODYSTROPHY_GENES)}


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Leukodystrophy-Atlas",
        "genes": [
            {
                "gene": g["gene"],
                "alt_name": g.get("alt_name", ""),
                "definition": g["alias"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "age_of_onset": g["age_of_onset"],
                "critical_flags": g["critical_flags"],
            }
            for g in LEUKODYSTROPHY_GENES
        ],
        "glossary": {
            "Leukodystrophy": (
                "Inherited disorders of myelin formation or maintenance; classified as hypomyelinating "
                "(myelin never formed — PLP1, POLR3A) vs demyelinating (myelin formed then lost — ARSA, GALC) "
                "vs spongiform (Canavan) vs astrocytopathy (Alexander); MRI is the primary diagnostic tool; "
                "always check U-fibres (subcortical arcuate fibres), corpus callosum, cerebellum, and peripheral nerves"
            ),
            "Metachromatic Leukodystrophy (MLD)": (
                "ARSA deficiency; sulfatide accumulation; lysosomal demyelinating leukodystrophy; "
                "periventricular tigroid T2 on MRI; urine metachromatic granules; "
                "Libmeldy gene therapy (EMA2020) for pre-symptomatic/early-symptomatic; "
                "pseudodeficiency pitfall (N350S+I179S) must be excluded"
            ),
            "Krabbe Disease (Globoid Cell Leukodystrophy)": (
                "GALC deficiency; psychosine toxic at nanomolar concentrations; globoid cells pathognomonic; "
                "infantile: fatal by 2yr untreated; HSCT ONLY pre-symptomatic; NBS mandatory; "
                "deep cerebellar/posterior cerebral WM early on MRI"
            ),
            "Pelizaeus-Merzbacher Disease (PMD)": (
                "PLP1 duplication (70%) / deletion (SPG2) / null (connatal); X-linked; "
                "diffuse hypomyelination from birth; nystagmus earliest sign; MLPA mandatory; "
                "no approved therapy; gene silencing approaches in development"
            ),
            "X-linked Adrenoleukodystrophy (X-ALD)": (
                "ABCD1 LOF; VLCFA accumulation; CCALD (35-40%): childhood parieto-occipital WM Gd enhancement; "
                "AMN (40-45%): adult progressive myelopathy; adrenal insufficiency 71%; "
                "Skysona (FDA2022) + HSCT for early CCALD; PHT absolute CI (adrenal crisis)"
            ),
            "Canavan Disease": (
                "ASPA deficiency; NAA accumulation; MRS NAA elevated = most specific leukodystrophy biomarker; "
                "U-fibre involvement early; macrocephaly; Ashkenazi founder (p.Glu285Ala + p.Tyr231X); "
                "no approved therapy; AAV9-ASPA trials"
            ),
            "Alexander Disease": (
                "GFAP dominant GOF (NOT LOF); all mutations heterozygous missense; de novo most common; "
                "Rosenthal fibres pathognomonic (perivascular/subpial eosinophilic aggregates); "
                "frontal WM + basal ganglia MRI; CSF GFAP elevated; no enzyme test; "
                "adult Type II: palatal myoclonus + bulbar dysfunction"
            ),
            "Vanishing White Matter Disease (VWM)": (
                "EIF2B1-5 biallelic LOF; ISR hypersensitivity; FLAIR WM = CSF signal (pathognomonic); "
                "stress triggers (fever, head trauma) → acute deterioration EMERGENCY; "
                "aggressive antipyretics mandatory; ovarioleukodystrophy in females; "
                "ISRIB in trials"
            ),
            "POLR3-Related Leukodystrophy (HLD7)": (
                "POLR3A biallelic LOF; RNA Pol III → tRNA impairment; hypomyelination (not demyelination); "
                "dental abnormalities = most specific clinical clue; myopia; hypogonadism; "
                "T2 hypointensity globus pallidus/ventral pons/dentate; slow progression; "
                "intronic splice c.1909+22G>A requires genome sequencing"
            ),
            "Pseudodeficiency (ARSA)": (
                "N350S + I179S ARSA polymorphisms in trans → low enzyme activity WITHOUT disease; "
                "carrier frequency ~1-2% Europeans; urine sulfatide NORMAL (in true MLD: elevated); "
                "CRITICAL: always confirm low ARSA with sulfatide urine + molecular testing before diagnosing MLD; "
                "treating pseudodeficiency as MLD = wrong diagnosis with treatment harm"
            ),
            "Psychosine Hypothesis (Krabbe)": (
                "Psychosine (galactosylsphingosine, not galactosylceramide) is the direct cytotoxin in Krabbe; "
                "accumulates because GALC also cleaves psychosine; nanomolar concentrations kill oligodendrocytes; "
                "psychosine plasma/DBS levels are the most sensitive Krabbe biomarker; "
                "NBS: DBS GALC enzyme + confirmatory psychosine"
            ),
            "Integrated Stress Response (ISR) — VWM": (
                "eIF2α phosphorylation → eIF2B inhibition → reduced translation → ISR; "
                "in VWM (EIF2B LOF): ISR is constitutively hypersensitive → oligodendrocytes/astrocytes fragile; "
                "any stressor (fever, trauma) → acute ISR → acute VWM deterioration; "
                "ISRIB stabilises eIF2B → reduces ISR hypersensitivity — most rational VWM therapy"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== HEREDITARY-LEUKODYSTROPHY-ATLAS — OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN (GFAP — GOF distinction) ===")
    bd = breakdown()
    gfap = next(g for g in bd["genes"] if g["gene"] == "GFAP")
    print(json.dumps(gfap, indent=2)[:2000])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps({"pseudodeficiency": df["glossary"]["Pseudodeficiency (ARSA)"]}, indent=2))
