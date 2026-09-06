#!/usr/bin/env python3
"""Leukodystrophy-Atlas — Complete 8-Gene Leukodystrophy Atlas
ARSA (MLD — Metachromatic Leukodystrophy, Libmeldy gene therapy) ·
GALC (Krabbe / Globoid Cell Leukodystrophy, HSCT pre-symptomatic) ·
PLP1 (Pelizaeus-Merzbacher Disease, X-linked) ·
ABCD1 (X-ALD / X-linked Adrenoleukodystrophy, Skysona gene therapy) ·
ASPA (Canavan Disease, NAA accumulation, spongy degeneration) ·
GFAP (Alexander Disease, gain-of-function dominant, Rosenthal fibres) ·
MLC1 (Megalencephalic Leukodystrophy with Subcortical Cysts) ·
EIF2B5 (Vanishing White Matter / CACH syndrome, stress-induced)
320-patient aggregate cohort (8 × 40, seeds 958–965)

Leukodystrophy facts:
  - Leukodystrophies are heritable monogenic disorders of white-matter myelin formation
    or maintenance. Primary target: oligodendrocytes and/or myelin sheath proteins.
  - MRI is the PRIMARY diagnostic tool: pattern recognition drives gene-panel selection.
    Posterior-predominant (MLD, Krabbe); anterior-predominant (VWM, MLC, Alexander);
    diffuse (PMD); adrenal + posterior (X-ALD).
  - TREATMENT BREAKTHROUGHS:
      MLD (ARSA): Libmeldy (atidarsagene autotemcel, OTL-200) — EMA-approved 2020 for
        pre-symptomatic early-juvenile and early-juvenile MLD. First approved gene therapy
        for MLD. Autologous HSC transduced ex-vivo with lentiviral vector encoding ARSA.
      Krabbe (GALC): HSCT in pre-symptomatic newborns (NBS screen essential); effective
        only before symptom onset — makes NBS life-saving.
      X-ALD (ABCD1): Skysona (elivaldogene autotemcel, Lenti-D) — FDA-approved 2022
        for boys 4-17 with early cerebral ALD (CALD) and ≤2 Loes score gadolinium
        enhancement; prevents neurological progression in ~90% at 2 years.
        Lorenzo's Oil (VLCFA normalization) does NOT halt neurological progression but
        reduces cerebral ALD risk in asymptomatic boys.
  - KEY TEACHING POINTS:
      ARSA SULFATIDE ACCUMULATION: demyelination from sulfatide deposition in oligos
        and Schwann cells → PNS + CNS both affected; metachromatic granules.
      GALC PSYCHOSINE (GALACTOSYLSPHINGOSINE) TOXIC: extreme cytotoxicity → globoid cells
        (multinucleated macrophages around vessels); spasticity + irritability rapid onset.
      PLP1 DOSE-MATTERS: DUPLICATION causes PMD (severe); DELETION causes SPG2 (mild, AD);
        null mutation causes connatal PMD (most severe — no protein at all).
      ABCD1 X-LINKED BUT FEMALE HETEROZYGOTES SYMPTOMATIC: 65% of female carriers
        develop AMN (adrenomyeloneuropathy) by age 60; adrenal insufficiency in 1% females.
      ASPA N-ACETYLASPARTATE (NAA): ASPA catabolises NAA in oligodendrocytes; deficiency
        → NAA accumulates → osmotic stress → spongy degeneration; NAA is a UNIQUE biomarker.
      GFAP GAIN-OF-FUNCTION: all GFAP mutations are dominant gain-of-function (NOT LOF);
        Rosenthal fibres (GFAP aggregates) are PATHOGNOMONIC on brain biopsy.
      MLC1 MEGALENCEPHALY AT BIRTH: head circumference above 98th centile at birth,
        then stabilises; white-matter cysts visible on MRI from infancy.
      EIF2B5 (VWM) STRESS-TRIGGERED EPISODIC CRISES: fever, minor head trauma, fright
        → acute neurological deterioration — parents must be STRICTLY counselled on
        avoiding febrile illness (antipyretics promptly) and head trauma.

COHORT: 8 × 40 = 320 patient slots (seeds 958–965; gene-specific seeds)
"""

import random

SEED_BASE = 958

LD_GENES = [
    # ── ARSA — Metachromatic Leukodystrophy ──────────────────────────────────────────
    {
        "gene": "ARSA", "protein": "Arylsulfatase A",
        "alias": "ARSA — Metachromatic Leukodystrophy (MLD); most common leukodystrophy (OMIM #250100)",
        "aa": "507 aa", "kDa": "59 kDa",
        "gene_class": (
            "Arylsulfatase A: lysosomal enzyme; requires saposin-B cofactor; "
            "catalyses sulfatide (3-O-sulfo-galactosylceramide) → galactosylceramide + sulfate; "
            "sulfatide is an essential myelin lipid; ARSA deficiency → sulfatide accumulates → "
            "demyelination; homodimeric at neutral pH, octameric at lysosomal pH; "
            "pseudodeficiency alleles (N350S + c.444+1G>A) exist — give low enzyme activity "
            "WITHOUT clinical disease; CRITICAL to distinguish by molecular testing"
        ),
        "ld_subgroup": "Lysosomal Enzyme Deficiency — Sphingolipidosis",
        "locus": "22q13.33", "omim_gene": 607574,
        "inheritance": "AR. 22q13.33. Both sexes equally. Incidence ~1/40,000-100,000; late-infantile most common form.",
        "onset_range_y": (1.0, 40.0),
        "phenotype": (
            "Three clinical forms: (1) LATE-INFANTILE (70%): onset 1-2 years; gait ataxia, loss of "
            "walking, peripheral neuropathy (reduced NCV), hypotonia, then spasticity; cognitive regression; "
            "death usually by age 5-10 without treatment. (2) JUVENILE (20%): onset 4-16 years; "
            "cognitive/behavioural changes first, then motor; school failure is often the presenting sign. "
            "(3) ADULT (10%): onset 16+ years; psychiatric symptoms (schizophrenia-like), personality change, "
            "progressive dementia + motor signs; often misdiagnosed as psychiatric disorder for years. "
            "ALL FORMS: peripheral neuropathy (ARSA affects Schwann cells too); fundus normal (vs Krabbe: "
            "optic atrophy common); MRI: posterior T2 hyperintensity, butterfly pattern sparing subcortical "
            "U-fibres until late; tigroid/leopard-skin pattern (periventricular)."
        ),
        "disease": (
            "ARSA encodes arylsulfatase A (507 aa, 59 kDa), a lysosomal hydrolase requiring saposin-B "
            "cofactor. Catalyses sulfatide hydrolysis: sulfatide + H2O → galactosylceramide + sulfate. "
            "Sulfatide is a major myelin constituent (~4% by weight); ARSA deficiency → metachromatic "
            "sulfatide deposits in white matter (CNS) and peripheral nerves (PNS) → progressive demyelination.\n\n"
            "BIOCHEMISTRY: low leukocyte ARSA enzyme activity (CRITICAL: pseudodeficiency alleles "
            "c.1049A>G/N350S + c.444+1G>A give low activity WITHOUT clinical MLD — always confirm with "
            "urine sulfatides elevated AND molecular testing). Urine metachromatic granules (under polarised "
            "light). NCV reduced (peripheral neuropathy before CNS signs in late-infantile).\n\n"
            "PSEUDODEFICIENCY: ~2% of Europeans carry pseudodeficiency alleles; enzyme activity 5-15% "
            "of normal; sulfatide excretion NORMAL; DO NOT treat as MLD. This is the #1 pitfall.\n\n"
            "TREATMENT: Libmeldy (atidarsagene autotemcel, OTL-200) — EMA-approved Nov 2020. "
            "Autologous HSC collected, transduced ex-vivo with lentiviral ARSA vector, reinfused after "
            "myeloablative conditioning. Effective ONLY pre-symptomatic (early-juvenile) or very early "
            "symptomatic. >90% stabilisation in pre-symptomatic early-juvenile at 3-year follow-up. "
            "HSCT from allogeneic donor is inferior to gene therapy. No effective treatment for "
            "symptomatic late-infantile (disease too advanced at gene therapy eligibility). "
            "Substrate reduction (patisiran analogue) in trials. Intrathecal enzyme replacement "
            "(HGT-1110) showed limited efficacy. Enzyme replacement IV does NOT cross BBB.\n\n"
            "p.Pro426Leu — most common pathogenic allele in European late-infantile. "
            "p.Ile179Ser — severe late-infantile. c.465+1G>A — splice allele late-infantile."
        ),
        "hallmark": (
            "ARSA HALLMARKS: "
            "(1) PSEUDODEFICIENCY ALLELES — CRITICAL PITFALL: N350S + c.444+1G>A give low enzyme WITHOUT MLD; "
            "confirm with urine sulfatides + sequencing. "
            "(2) SAPOSIN-B DEFICIENCY MIMICS MLD: same phenotype, normal ARSA enzyme; sulfatide elevated. "
            "(3) PNS + CNS BOTH AFFECTED: peripheral neuropathy (reduced NCV) precedes CNS in late-infantile. "
            "(4) LIBMELDY (OTL-200) FIRST APPROVED GENE THERAPY FOR MLD: EMA 2020, pre-symptomatic only. "
            "(5) MRI BUTTERFLY PATTERN: bilateral posterior periventricular T2 hyperintensity; U-fibres spared early. "
            "(6) METACHROMATIC GRANULES: sulfatide deposits stain metachromatically (brown-red with toluidine blue). "
            "(7) PSYCHIATRIC ONSET IN ADULTS: adult MLD often presents as schizophrenia — MRI is the key."
        ),
        "nbs_marker": "Not on standard NBS; pilot NBS programs use enzyme activity + biomarker sulfatide",
        "key_biomarker": "Low leukocyte ARSA enzyme + elevated urine sulfatides; peripheral nerve NCV reduced",
        "severity_spectrum": "Late-infantile most severe (early death); juvenile intermediate; adult slowest",
        "seed_offset": 0,
    },
    # ── GALC — Krabbe Disease ─────────────────────────────────────────────────────────
    {
        "gene": "GALC", "protein": "Galactocerebrosidase",
        "alias": "GALC — Krabbe Disease / Globoid Cell Leukodystrophy (GCL) (OMIM #245200)",
        "aa": "669 aa", "kDa": "76 kDa",
        "gene_class": (
            "Galactocerebrosidase: lysosomal enzyme requiring saposin-A cofactor; "
            "catalyses galactosylceramide → ceramide + galactose AND psychosine (galactosylsphingosine) "
            "→ sphingosine + galactose; PSYCHOSINE IS THE TOXIC METABOLITE — extremely cytotoxic "
            "to oligodendrocytes at low concentrations (nM range); GALC is expressed mainly in myelin-forming cells; "
            "14 kb deletion at 30q (del30) is the most common mutation in European Krabbe (45%)"
        ),
        "ld_subgroup": "Lysosomal Enzyme Deficiency — Galactosphingolipidosis",
        "locus": "14q31.3", "omim_gene": 606890,
        "inheritance": "AR. 14q31.3. Both sexes equally. Incidence ~1/100,000. Jewish: ~1/250.",
        "onset_range_y": (0.0, 40.0),
        "phenotype": (
            "EARLY-INFANTILE (85-90%): onset 3-6 months; EXTREME IRRITABILITY (hyperalgesia to touch/sound), "
            "hypertonia, opisthotonos, peripheral neuropathy, optic atrophy, rapid neurological decline, "
            "death usually 2-3 years. Classic presentation: previously normal infant becomes excessively "
            "irritable and develops spasticity. LATE-ONSET (10-15%): onset >6 months to adulthood; "
            "spastic paraparesis, ataxia, visual failure; much slower course. "
            "MRI: T2 hyperintensity deep white matter, cerebellar white matter, corticospinal tracts; "
            "posterior > anterior; early MRI normal in NBS-detected pre-symptomatic."
        ),
        "disease": (
            "GALC encodes galactocerebrosidase (669 aa, 76 kDa). Deficiency → two substrates accumulate: "
            "galactosylceramide (major myelin glycolipid) AND psychosine (galactosylsphingosine). "
            "PSYCHOSINE THEORY: psychosine is the primary toxic metabolite — selectively destroys "
            "oligodendrocytes and Schwann cells at nanomolar concentrations; 'suicide' hypothesis: "
            "GALC-deficient oligodendrocytes produce psychosine as they make myelin, triggering their "
            "own apoptosis. Galactosylceramide is phagocytosed by macrophages → globoid cells "
            "(multinucleated macrophages, 'globoid' appearance — PATHOGNOMONIC).\n\n"
            "BIOCHEMISTRY: low leukocyte GALC enzyme activity; psychosine (galactosylsphingosine) "
            "elevated in plasma/DBS — psychosine is now the primary NBS biomarker for Krabbe. "
            "CSF protein elevated (peripheral neuropathy). NCV markedly reduced.\n\n"
            "NBS: HSCT is effective ONLY if started pre-symptomatic (identified by NBS). "
            "New York was the first state to mandate NBS for Krabbe (2006). "
            "TREATMENT: HSCT (allogeneic) — stabilises neurological function in pre-symptomatic "
            "infants treated before 30 days of age. Umbilical cord blood transplant preferred "
            "(lower graft-versus-host, higher cord blood GALC activity). Late-onset Krabbe also "
            "benefits from HSCT if non-ambulatory disease not yet present. "
            "Gene therapy (AAV-GALC) in clinical trials. NO approved ERT (GALC crosses BBB poorly).\n\n"
            "del30 (14 kb deletion, c.857_861del) — 45% of European alleles. "
            "p.Thr513Met — late-onset. p.Gly270Asp — severe infantile."
        ),
        "hallmark": (
            "GALC HALLMARKS: "
            "(1) PSYCHOSINE TOXIC — primary killing mechanism; globoid cells are secondary; "
            "psychosine is now the preferred NBS biomarker. "
            "(2) EXTREME IRRITABILITY: hyperalgesia to touch/sound at 3-6 months = PATHOGNOMONIC early sign. "
            "(3) GLOBOID CELLS = PATHOGNOMONIC on brain biopsy (multinucleated macrophages). "
            "(4) NBS + HSCT ONLY WINDOW: treatment effective ONLY pre-symptomatic; NBS is life-saving. "
            "(5) PNS + CNS BOTH: peripheral neuropathy (NCV reduced) + central demyelination. "
            "(6) del30 (14 kb deletion) = most common European allele (45%); not detected by sequencing alone. "
            "(7) OPTIC ATROPHY: common (vs MLD where fundus usually normal until late)."
        ),
        "nbs_marker": "Low leukocyte GALC enzyme; psychosine elevated in DBS — primary NBS biomarker",
        "key_biomarker": "Leukocyte GALC enzyme low + psychosine elevated; NCV reduced; CSF protein high",
        "severity_spectrum": "Early-infantile most severe (die by 2-3y); late-onset much milder, slower",
        "seed_offset": 1,
    },
    # ── PLP1 — Pelizaeus-Merzbacher Disease ──────────────────────────────────────────
    {
        "gene": "PLP1", "protein": "Proteolipid Protein 1",
        "alias": "PLP1 — Pelizaeus-Merzbacher Disease (PMD) / SPG2 (OMIM #312080)",
        "aa": "276 aa", "kDa": "30 kDa",
        "gene_class": (
            "Proteolipid protein 1: most abundant CNS myelin protein (~50% of myelin protein); "
            "integral membrane protein with 4 TM helices; essential for myelin compaction and "
            "oligodendrocyte survival; X-linked; DOSAGE-SENSITIVE: "
            "DUPLICATION → PMD (most common cause, 70%); "
            "POINT MUTATION → classic PMD or SPG2 depending on severity; "
            "DELETION (null) → connatal PMD (most severe) or SPG2 (mild if minor allele); "
            "DM20 is a shorter isoform (alternative splicing, lacks exon 3B); "
            "PLP1 required for survival of myelinating oligodendrocytes"
        ),
        "ld_subgroup": "Myelin Structural Protein — X-Linked",
        "locus": "Xq22.2", "omim_gene": 300401,
        "inheritance": "X-linked. Xq22.2. Males affected; females usually asymptomatic carriers (some mild). Incidence ~1/200,000-500,000 males.",
        "onset_range_y": (0.0, 10.0),
        "phenotype": (
            "CONNATAL PMD (null/hypomorphic mutations, most severe): onset at birth; absent myelination; "
            "severe hypotonia → spasticity; nystagmus from birth; no head control; death in infancy/childhood. "
            "CLASSIC PMD (duplication, most common): nystagmus onset 1-3 months (PATHOGNOMONIC first sign); "
            "delayed motor milestones; ataxia; head titubation; slow progression; many survive to adulthood "
            "with quadriplegia; intelligence often relatively preserved early. "
            "TRANSITIONAL PMD: intermediate. SPG2 (spastic paraplegia type 2, mild point mutations): "
            "pure or complicated spastic paraplegia; ambulatory; less cognitive involvement. "
            "MRI: NEAR-COMPLETE ABSENCE OF MYELIN — white matter appears like newborn brain (hypomyelination), "
            "T2 uniformly hyperintense, T1 low; U-fibres also involved; NO inflammatory change."
        ),
        "disease": (
            "PLP1 encodes proteolipid protein 1 (276 aa, 30 kDa, with DM20 isoform lacking exon 3B). "
            "PLP1/DM20 is the most abundant CNS myelin protein. DOSAGE EFFECT is the key mechanism:\n\n"
            "DUPLICATION (Xq22.2 dup, 70% of PMD): excess PLP1 → retained in ER (ER stress) → "
            "oligodendrocyte apoptosis → hypomyelination. Classic PMD phenotype.\n\n"
            "POINT MUTATION (15-20%): misfolded PLP1 → ER stress → oligodendrocyte death. "
            "p.Ala242Val, p.Pro14Leu — classic PMD. p.Gly73Ala, p.Ile186Thr — connatal.\n\n"
            "NULL (DELETION, 10-15%): no PLP1 protein → oligodendrocytes initially myelinate "
            "BUT cannot maintain myelin → axonal degeneration develops over time. "
            "Paradoxically LESS SEVERE initially than duplication (less ER stress) but late axon loss.\n\n"
            "FEMALE CARRIERS: random X-inactivation. If predominantly expressing mutant X → mild PMD "
            "(spastic paraplegia). If wild-type skewed → asymptomatic. X-inactivation testing useful.\n\n"
            "DIAGNOSIS: MRI hypomyelination + X-linked family history + MLPA (duplication detection, "
            "NOT detected by sequencing alone!) + PLP1 sequencing.\n\n"
            "TREATMENT: No disease-modifying therapy approved. Thyroid hormone (T3) promotes myelination "
            "in animal models; human trials ongoing. Stem cell/ASO strategies in trials. "
            "Symptomatic: baclofen/tizanidine for spasticity; anti-epileptics if seizures."
        ),
        "hallmark": (
            "PLP1 HALLMARKS: "
            "(1) NYSTAGMUS ONSET 1-3 MONTHS = EARLIEST CLINICAL SIGN in classic PMD (duplication). "
            "(2) DOSAGE-SENSITIVE: duplication = PMD; null/deletion = connatal PMD OR SPG2 (milder); "
            "point mutation severity = allele-specific. "
            "(3) MLPA MANDATORY: duplication is 70% of cases — NOT detected by sequencing; always do MLPA/CNV. "
            "(4) X-LINKED: males severely affected; female carriers usually normal or mild SPG2. "
            "(5) MRI HYPOMYELINATION = NEAR-COMPLETE: white matter like newborn brain — no myelin signal. "
            "(6) NO DISEASE-MODIFYING TREATMENT APPROVED: symptomatic management only currently. "
            "(7) DM20 ISOFORM PARTIALLY SPARES MYELINATION: PLP1-null retains DM20 → initially less severe."
        ),
        "nbs_marker": "Not on standard NBS; suspected from nystagmus + family history",
        "key_biomarker": "MRI hypomyelination + MLPA for Xq22.2 duplication + PLP1 sequencing; no serum biomarker",
        "severity_spectrum": "Connatal (null mutations) most severe; classic PMD (duplication) intermediate; SPG2 mild",
        "seed_offset": 2,
    },
    # ── ABCD1 — X-linked Adrenoleukodystrophy ────────────────────────────────────────
    {
        "gene": "ABCD1", "protein": "ATP-Binding Cassette Subfamily D Member 1 (ALDP)",
        "alias": "ABCD1 — X-ALD / X-linked Adrenoleukodystrophy (OMIM #300100); most common peroxisomal disorder",
        "aa": "745 aa", "kDa": "84 kDa",
        "gene_class": (
            "ALDP (adrenoleukodystrophy protein): peroxisomal ABC half-transporter (homodimer in membrane); "
            "imports very-long-chain fatty acid (VLCFA, C22+)-CoA esters into peroxisomes for beta-oxidation; "
            "ABCD1 deficiency → VLCFA accumulate in plasma, brain white matter, adrenal cortex, testes; "
            "peroxisomal membrane protein; ABCD2 (ALDRP) is a homologue that can partially compensate; "
            "no genotype-phenotype correlation (same mutation → different phenotypes in same family)"
        ),
        "ld_subgroup": "Peroxisomal Membrane Transport — X-Linked",
        "locus": "Xq28", "omim_gene": 300371,
        "inheritance": "X-linked. Xq28. Males affected (cerebral ALD, AMN, adrenal insufficiency); 65% of female carriers symptomatic by age 60 (AMN). Incidence ~1/17,000 (all phenotypes combined).",
        "onset_range_y": (3.0, 60.0),
        "phenotype": (
            "MALES — four phenotypes (no genotype-phenotype correlation): "
            "(1) CHILDHOOD CEREBRAL ALD (CALD, 35-40%): onset 4-10 years; ADHD-like, school failure, "
            "then rapid neurological decline (spasticity, visual/hearing loss, seizures, vegetative state) "
            "within 2-3 years if untreated; MRI: posterior-predominant gadolinium-enhancing inflammatory "
            "demyelination (Loes score). (2) AMN (adrenomyeloneuropathy, 40-45%): onset 20-40 years; "
            "spastic paraparesis + peripheral neuropathy; slow progression; 50% develop cerebral involvement. "
            "(3) ADDISON-ONLY (10-15%): adrenal insufficiency only; no neurological disease initially "
            "(may develop AMN later). (4) ASYMPTOMATIC (<10%): incidental VLCFA elevation. "
            "FEMALES: 65% develop AMN-like syndrome by age 60; adrenal insufficiency rare (1%)."
        ),
        "disease": (
            "ABCD1 encodes ALDP (745 aa, 84 kDa), a peroxisomal half-transporter. Deficiency → "
            "C24:0 and C26:0 VLCFA accumulate in plasma, brain white matter, adrenal cortex, testes.\n\n"
            "PATHOGENESIS: VLCFA accumulate in myelin (structural instability) and adrenal cells "
            "(functional impairment → adrenal insufficiency). In childhood cerebral ALD, an "
            "inflammatory cascade amplifies demyelination — adrenal macrophages + lymphocytes "
            "infiltrate at gadolinium-enhancing lesion edge. This inflammatory component is "
            "targetable by HSCT/gene therapy.\n\n"
            "NO GENOTYPE-PHENOTYPE CORRELATION: brothers with the same mutation may have childhood "
            "CALD vs AMN vs Addison-only. Annual MRI surveillance mandatory in all affected boys.\n\n"
            "TREATMENT: (1) Skysona (elivaldogene autotemcel, Lenti-D) — FDA-approved Jul 2022 for "
            "boys 4-17 with early CALD (Loes score ≤4, gadolinium enhancement present). "
            "90% major functional disability-free survival at 2 years. Requires HLA-matched donor "
            "unavailability or preference for autologous. Risk of haematological malignancy "
            "(1 confirmed case in trials — boxed warning). (2) HSCT: alternative to gene therapy; "
            "effective only in early CALD. (3) Lorenzo's Oil (4:1 GTO:GTE mixture): normalises "
            "plasma VLCFA in ~90% of patients but does NOT halt or reverse neurological decline; "
            "may reduce risk of CALD onset in asymptomatic boys (not proven). "
            "(4) Adrenal insufficiency: mandatory hydrocortisone replacement — can be life-threatening "
            "crisis; ALL males must have adrenal function tested at diagnosis.\n\n"
            "BIOMARKER: plasma VLCFA (C26:0, C24:0, C26:0/C22:0 ratio) — elevated in >99% of males, "
            "~85% of female carriers. NBS: plasma C26:0-lysophosphatidylcholine (C26:0-LysoPC) on DBS."
        ),
        "hallmark": (
            "ABCD1 HALLMARKS: "
            "(1) NO GENOTYPE-PHENOTYPE CORRELATION: same mutation → CALD in one brother, AMN in another. "
            "(2) ANNUAL MRI MANDATORY IN ALL BOYS: CALD transforms pre-symptomatically; MRI surveillance "
            "is the only way to catch the treatable window. "
            "(3) ADRENAL INSUFFICIENCY ALWAYS TEST: adrenal crisis is life-threatening; not all boys have "
            "obvious neurological signs — adrenal must be assessed at diagnosis. "
            "(4) SKYSONA (FDA 2022): gene therapy for early CALD (Loes ≤4, gadolinium enhancement); "
            "boxed warning for haematological malignancy. "
            "(5) LORENZO'S OIL NORMALISES VLCFA BUT DOES NOT HALT NEUROLOGY: "
            "used for asymptomatic boys to reduce CALD risk (evidence weak). "
            "(6) NBS VLCFA-LysoPC: now in newborn screening in many states for early detection. "
            "(7) FEMALE CARRIERS SYMPTOMATIC IN 65% BY AGE 60: AMN; adrenal failure rare (1%)."
        ),
        "nbs_marker": "C26:0-lysophosphatidylcholine (LysoPC C26:0) elevated on DBS; NBS implemented in many US states",
        "key_biomarker": "Plasma VLCFA: C26:0 elevated, C24:0/C22:0 and C26:0/C22:0 ratios elevated; cortisol low if adrenal failure",
        "severity_spectrum": "CALD most severe/rapid (untreated); AMN slow progressive; Addison-only if no neurology; female carriers often develop AMN",
        "seed_offset": 3,
    },
    # ── ASPA — Canavan Disease ────────────────────────────────────────────────────────
    {
        "gene": "ASPA", "protein": "Aspartoacylase",
        "alias": "ASPA — Canavan Disease (OMIM #271900); spongy degeneration of the brain",
        "aa": "313 aa", "kDa": "36 kDa",
        "gene_class": (
            "Aspartoacylase: cytosolic enzyme in oligodendrocytes; "
            "catalyses N-acetylaspartate (NAA) → aspartate + acetate; "
            "NAA is the most abundant amino acid in the brain after glutamate; "
            "acetate released by ASPA is required for myelin lipid synthesis (acetyl-CoA); "
            "ASPA deficiency → NAA accumulates → osmotic vacuolation of myelin (spongy degeneration); "
            "NAA measured by MRS (proton MR spectroscopy) is pathognomonic: NAA peak MARKEDLY elevated"
        ),
        "ld_subgroup": "Metabolic Enzyme Deficiency — CNS-Specific",
        "locus": "17p13.2", "omim_gene": 608034,
        "inheritance": "AR. 17p13.2. Both sexes equally. Incidence ~1/6,400-13,500 in Ashkenazi Jewish; ~1/300,000-500,000 worldwide.",
        "onset_range_y": (0.0, 1.0),
        "phenotype": (
            "Onset: 2-4 months of age. Classic triad: (1) MACROCEPHALY — head circumference > 97th "
            "centile, progressive postnatal (born normal-sized); (2) HYPOTONIA — severe axial hypotonia "
            "with progressive spasticity developing later; (3) DEVELOPMENTAL REGRESSION — loss of "
            "milestones after initial partial acquisition. Additional: seizures (50%); optic atrophy "
            "(60%); irritability; poor head control. SEVERE form (most): death usually before age 10, "
            "often respiratory complications. MILD/JUVENILE form (<5% of cases): later onset, slower "
            "progression, some motor function retained. MRI: DIFFUSE cerebral white-matter T2 "
            "hyperintensity (subcortical U-fibres INVOLVED — differentiates from MLD/Krabbe); "
            "MR SPECTROSCOPY: MARKEDLY ELEVATED NAA PEAK — single most specific MRS finding in any "
            "leukodystrophy (NAA normally the tallest peak; in Canavan it is dramatically increased)."
        ),
        "disease": (
            "ASPA encodes aspartoacylase (313 aa, 36 kDa). NAA is synthesised in neurons by NAT8L "
            "(NAT8 like 8), then exported across the synapse; oligodendrocytes take up NAA and "
            "catabolise it via ASPA. ASPA deficiency → NAA accumulates → osmotic stress → "
            "intramyelinic vacuole formation → spongy degeneration (spongiform appearance on "
            "histology — vacuoles in astrocytes and oligodendrocytes within white matter).\n\n"
            "DUAL CONSEQUENCE: (1) NAA toxic accumulation → spongy vacuolation; (2) acetate "
            "deficiency → myelin lipid synthesis impaired (acetyl-CoA substrate reduced).\n\n"
            "ASHKENAZI JEWISH FOUNDER MUTATIONS: p.Glu285Ala (c.854A>C) — 82-86% of Ashkenazi alleles; "
            "p.Tyr231Ter (c.693C>A) — 14-18% of Ashkenazi alleles. Carrier frequency ~1/40 in Ashkenazi. "
            "Prenatal diagnosis available; carrier screening offered to Ashkenazi couples.\n\n"
            "URINE NAA: dramatically elevated (>1000 mmol/mol creatinine; normal <30). "
            "Plasma NAA elevated. MRS NAA peak elevated.\n\n"
            "TREATMENT: No disease-modifying therapy approved. Glyceryl triacetate (GTA) — "
            "acetate supplementation to restore oligodendrocyte substrate; animal studies positive; "
            "human pilot studies ongoing. AAV-ASPA gene therapy (IT/IC delivery) in Phase 1-2 trials "
            "(Aspa-null mouse fully rescued). rAAV9-ASPA ongoing (2023-2024). "
            "Lithium: inhibits INOSITOL pathway (possibly reduces NAA synthesis); small benefit."
        ),
        "hallmark": (
            "ASPA HALLMARKS: "
            "(1) MRS NAA MARKEDLY ELEVATED = PATHOGNOMONIC: highest NAA peak of any brain disorder; "
            "confirms Canavan when all other white-matter disorders are in differential. "
            "(2) MACROCEPHALY POSTNATAL-ONSET: born with normal head, macrocephaly develops 2-4 months. "
            "(3) U-FIBRES INVOLVED EARLY: MRI subcortical U-fibres abnormal (unlike MLD/Krabbe). "
            "(4) ASHKENAZI JEWISH FOUNDER: p.Glu285Ala (82-86%) + p.Tyr231Ter (14-18%); carrier 1/40. "
            "(5) URINE NAA >1000 mmol/mol creatinine: simple urine organic acid test detects it. "
            "(6) SPONGY DEGENERATION on pathology: vacuoles in myelin, astrocytes, oligodendrocytes. "
            "(7) AAV GENE THERAPY IN TRIALS: first leukodystrophy to show proof-of-concept in AAV preclinical."
        ),
        "nbs_marker": "Not on standard NBS; Ashkenazi carrier screening available (two founder mutations)",
        "key_biomarker": "Urine NAA dramatically elevated; MRS NAA peak elevated; ASPA enzyme assay; ASPA sequencing",
        "severity_spectrum": "Classic (most): severe, death before 10y; mild/juvenile (<5%): later onset, ambulatory",
        "seed_offset": 4,
    },
    # ── GFAP — Alexander Disease ──────────────────────────────────────────────────────
    {
        "gene": "GFAP", "protein": "Glial Fibrillary Acidic Protein",
        "alias": "GFAP — Alexander Disease (AxD) (OMIM #203450); ONLY dominant leukodystrophy (gain-of-function)",
        "aa": "432 aa", "kDa": "49.9 kDa",
        "gene_class": (
            "Glial Fibrillary Acidic Protein: intermediate filament protein; "
            "major structural component of astrocytes; forms heteropolymers with vimentin, nestin; "
            "ALL pathogenic GFAP mutations are dominant gain-of-function (NEVER haploinsufficiency); "
            "mutant GFAP aggregates form Rosenthal fibres (PATHOGNOMONIC); "
            "Rosenthal fibres are electron-dense inclusions in astrocytes containing GFAP + alphaB-crystallin + "
            "HSP27; astrocyte dysfunction → secondary myelin loss + neurodegeneration; "
            "p.Arg79 is hotspot (14-17% of all mutations)"
        ),
        "ld_subgroup": "Astrocyte Structural Protein — Dominant Gain-of-Function",
        "locus": "17q21.31", "omim_gene": 137780,
        "inheritance": "AD (gain-of-function, usually de novo). 17q21.31. Both sexes equally. Incidence ~1/2,700,000 (rare). De novo mutations account for >95% of infantile-onset cases.",
        "onset_range_y": (0.0, 70.0),
        "phenotype": (
            "Three forms (age-of-onset-based): (1) TYPE I (infantile, 75-80%): onset <4 years; "
            "macrocephaly (99%, birth or soon after — LARGEST HEAD in leukodystrophies); "
            "seizures (75%); progressive ataxia; spasticity; frontal-predominant white matter loss; "
            "death often 2nd-3rd decade. (2) TYPE II (juvenile, 10-15%): onset 4-14 years; "
            "bulbar dysfunction (dysarthria, dysphagia) + ataxia; palatal myoclonus; "
            "slower progression. (3) TYPE III (adult, 10-15%): onset >14 years (range to 70s); "
            "bulbar symptoms dominant; episodic deterioration; can survive decades. "
            "MRI TYPE I: bilateral FRONTAL predominant T2 hyperintensity (ANTERIOR vs MLD/Krabbe posterior); "
            "periventricular rim enhancement; basal ganglia T2 hyperintense; brain-stem involved; "
            "PATTERN: frontal subcortical > periventricular > cortical atrophy. "
            "TYPE II-III: brain stem + cerebellum dominant; white matter changes less prominent."
        ),
        "disease": (
            "GFAP encodes glial fibrillary acidic protein (432 aa, 49.9 kDa). ALL pathogenic GFAP "
            "mutations are heterozygous missense gain-of-function — they do NOT cause haploinsufficiency. "
            "Mutant GFAP misfolds and aggregates in astrocytes → Rosenthal fibres (PATHOGNOMONIC "
            "electron-dense inclusions: GFAP + alphaB-crystallin + HSP27 + ubiquitin). "
            "Astrocyte dysfunction → impaired K+ buffering, glutamate handling, water homeostasis "
            "→ secondary white matter damage + axonal loss.\n\n"
            "MUTATION HOTSPOTS: p.Arg79His/Cys (exon 1, ~14-17% of all cases — most common single "
            "mutation); p.Arg239His/Cys (exon 4); p.Arg416Trp (exon 8). >70 different mutations known.\n\n"
            "GENOTYPE-PHENOTYPE: p.Arg79 mutations predominate in infantile-onset; p.Arg239 and "
            "p.Arg416 more in juvenile/adult. However, same mutation can give different onset ages.\n\n"
            "BIOMARKER: GFAP protein in CSF and serum MARKEDLY ELEVATED in Alexander disease — "
            "can be used for monitoring; higher levels correlate with disease severity/activity. "
            "GFAP immunostaining in brain biopsy shows Rosenthal fibres.\n\n"
            "TREATMENT: No disease-modifying therapy approved. GFAP ASO (antisense oligonucleotide) — "
            "preclinical data shows reducing GFAP levels reduces Rosenthal fibres; clinical trial "
            "ongoing (GFAPsynergy — intrathecal ASO). Heat shock protein inducers (arimoclomol) in trials. "
            "Supportive: anti-epileptics; PEG tube for bulbar dysfunction."
        ),
        "hallmark": (
            "GFAP HALLMARKS: "
            "(1) ONLY DOMINANT LEUKODYSTROPHY (gain-of-function): ALL mutations are dominant gain-of-function; "
            "no recessive form; usually de novo. "
            "(2) ROSENTHAL FIBRES = PATHOGNOMONIC: GFAP aggregates in astrocytes; diagnose on brain biopsy "
            "OR GFAP sequencing. "
            "(3) MACROCEPHALY MOST SEVERE: largest head of any leukodystrophy in infantile form (>99%). "
            "(4) FRONTAL WHITE MATTER PREDOMINANT (TYPE I): anterior-predominant T2 on MRI — "
            "opposite of MLD/Krabbe (posterior). "
            "(5) BULBAR SYMPTOMS TYPE II/III: palatal myoclonus + dysarthria + dysphagia = "
            "Type II-III Alexander hallmark; often misdiagnosed as multiple sclerosis. "
            "(6) SERUM/CSF GFAP MARKEDLY ELEVATED: useful monitoring biomarker. "
            "(7) ASO THERAPY IN TRIALS: GFAP ASO reduces Rosenthal fibres in preclinical — "
            "first credible disease-modifying therapy candidate."
        ),
        "nbs_marker": "Not on NBS; macrocephaly + frontal MRI changes prompt GFAP sequencing",
        "key_biomarker": "CSF/serum GFAP markedly elevated; GFAP sequencing (usually de novo); Rosenthal fibres on biopsy",
        "severity_spectrum": "Type I (infantile) severe, childhood-adolescent death; Type II juvenile moderate; Type III adult slowest",
        "seed_offset": 5,
    },
    # ── MLC1 — Megalencephalic Leukodystrophy with Subcortical Cysts ─────────────────
    {
        "gene": "MLC1", "protein": "MLC1 Ion-Channel-like Protein (Vacuolating Megalencephalic "
                "Leukoencephalopathy Protein 1)",
        "alias": "MLC1 — Megalencephalic Leukodystrophy with Subcortical Cysts Type 1 (MLC1; OMIM #604004)",
        "aa": "377 aa", "kDa": "42 kDa",
        "gene_class": (
            "MLC1 protein: integral membrane protein (8 TM helices); expressed exclusively in astrocytic "
            "endfeet at blood-brain barrier and glia limitans; forms complex with GlialCAM (HEPACAM); "
            "MLC1 + GlialCAM regulate astrocytic volume and chloride channels (ClC-2 regulation); "
            "MLC1 mutations → impaired astrocytic volume regulation → vacuolation within myelin "
            "(intramyelinic oedema); MLC1 is PURE WHITE MATTER DISEASE with VERY SLOW PROGRESSION"
        ),
        "ld_subgroup": "Astrocyte Ion-Channel Protein — Vacuolating Leukodystrophy",
        "locus": "22q13.33", "omim_gene": 605908,
        "inheritance": "AR. 22q13.33. Both sexes equally. Incidence unknown (rare). Founder mutations in Agarwal (Indian) and Turkish communities.",
        "onset_range_y": (0.0, 2.0),
        "phenotype": (
            "Distinct clinical phenotype: (1) MEGALENCEPHALY AT OR AFTER BIRTH: head circumference "
            ">98th centile at birth or first year; stabilises in early childhood. (2) MRI "
            "SUBCORTICAL CYSTS: temporal-anterior + frontoparietal subcortical cysts visible from "
            "infancy — these cysts are PATHOGNOMONIC; diffuse white-matter swelling and T2 "
            "hyperintensity from birth. (3) VERY SLOW CLINICAL PROGRESSION: patients remain "
            "ambulatory for decades; mild ataxia, spasticity, seizures (40-60%); "
            "MILD COGNITIVE DELAY. (4) LATER DETERIORATION: after 3rd-4th decade, some lose "
            "ambulation; rare patients deteriorate rapidly (especially after head trauma or febrile illness). "
            "MLC2 (HEPACAM mutations): MLC2A (same phenotype as MLC1 — misroutes MLC1 protein) "
            "and MLC2B (HEPACAM mutation, same MRI but REMITTING — patients improve! MLC2B is benign). "
            "MLC1 vs MLC2B: MLC2B patients RECOVER white matter signal; MLC1 patients do NOT recover."
        ),
        "disease": (
            "MLC1 encodes MLC1 protein (377 aa, 42 kDa), expressed exclusively in astrocytic endfeet. "
            "MLC1 + GlialCAM (HEPACAM) form a complex at blood-brain barrier and glia limitans that "
            "regulates ClC-2 chloride channels. MLC1 deficiency → impaired astrocytic volume regulation "
            "→ intramyelinic vacuolation (fluid accumulates within the myelin sheath itself, not in "
            "the extracellular space) → white-matter oedema → megaloencephaly.\n\n"
            "VERY SLOW NATURAL HISTORY: oligodendrocytes continue to myelinate (unlike most "
            "leukodystrophies); vacuolation is the primary pathology; axons relatively spared for years. "
            "Explains why patients remain ambulatory for decades despite dramatic MRI changes.\n\n"
            "AGARWAL FOUNDER MUTATION: p.Ser93Leu (c.278C>T) — present in Agarwal caste (India) at "
            "carrier frequency ~1/40. Turkish founder: p.Ile107del.\n\n"
            "MLC2 (HEPACAM) distinction: MLC2A (missense → mis-routes MLC1 to lysosome) = same as MLC1 phenotype. "
            "MLC2B (dominant negative HEPACAM) = REMITTING benign form — some patients normalise MRI. "
            "HEPACAM testing important for prognosis (MLC2B patients improve; MLC1 patients do not).\n\n"
            "TREATMENT: No disease-modifying therapy. Seizure control (anti-epileptics). "
            "Strict head protection (head trauma → acute deterioration). Avoid febrile illness "
            "(similar to VWM). Acetazolamide (reduces CSF pressure/white-matter oedema) in pilot trials. "
            "Prognosis much better than MLD/Krabbe — most patients survive to adulthood."
        ),
        "hallmark": (
            "MLC1 HALLMARKS: "
            "(1) SUBCORTICAL CYSTS TEMPORAL + FRONTOPARIETAL = PATHOGNOMONIC: visible from infancy on MRI. "
            "(2) MEGALENCEPHALY AT BIRTH: head circumference >98th centile — not postnatal like Canavan. "
            "(3) VERY SLOW PROGRESSION: ambulatory for decades despite severe MRI changes — "
            "better prognosis than all other leukodystrophies. "
            "(4) MLC2B (HEPACAM dominant negative) IS REMITTING-BENIGN: white matter normalises — "
            "distinguish MLC1 from MLC2B for accurate prognosis. "
            "(5) AGARWAL FOUNDER (INDIA): p.Ser93Leu carrier 1/40 in Agarwal caste — "
            "most common in that community. "
            "(6) HEAD TRAUMA + FEVER = ACUTE DETERIORATION TRIGGER: counsel families strictly. "
            "(7) ASTROCYTE ENDFEET EXCLUSIVE: MLC1 protein only in astrocyte endfeet — explains why "
            "non-astrocyte brain cells are initially spared."
        ),
        "nbs_marker": "Not on NBS; megalencephaly at birth prompts MRI + MLC1 testing",
        "key_biomarker": "MRI subcortical temporal cysts + white-matter swelling; MLC1 + HEPACAM sequencing; no specific serum biomarker",
        "severity_spectrum": "MLC1: slow progressive, ambulatory decades; MLC2B (HEPACAM): benign remitting (completely different prognosis)",
        "seed_offset": 6,
    },
    # ── EIF2B5 — Vanishing White Matter / CACH ───────────────────────────────────────
    {
        "gene": "EIF2B5", "protein": "Eukaryotic Translation Initiation Factor 2B Subunit Epsilon",
        "alias": "EIF2B5 — Vanishing White Matter (VWM) / CACH Syndrome (OMIM #603896); most common EIF2B gene",
        "aa": "721 aa", "kDa": "82 kDa",
        "gene_class": (
            "eIF2B epsilon subunit: catalytic subunit of the eIF2B guanine nucleotide exchange factor (GEF) complex; "
            "eIF2B (5-subunit complex: EIF2B1-5) converts eIF2-GDP → eIF2-GTP for global translation initiation; "
            "STRESS RESPONSE: phosphorylated eIF2alpha (by PERK, HRI, GCN2, PKR) INHIBITS eIF2B → "
            "stress-induced translation arrest (integrated stress response, ISR); "
            "VWM mutations → eIF2B hypoactive → HYPERSENSITIVE to ISR → stress triggers acute decompensation; "
            "EIF2B5 mutations account for 65% of VWM alleles (most common of 5 subunit genes)"
        ),
        "ld_subgroup": "Translation Initiation — Integrated Stress Response",
        "locus": "3q27.1", "omim_gene": 603945,
        "inheritance": "AR. 3q27.1. Both sexes equally (but FEMALES MORE SEVERELY AFFECTED: ovarian failure in Ovarioleukodystrophy sub-phenotype). Incidence ~1/100,000-250,000.",
        "onset_range_y": (0.0, 60.0),
        "phenotype": (
            "EPISODIC STRESS-TRIGGERED DETERIORATION is the hallmark: infection, minor head trauma, "
            "or even fright → acute neurological deterioration (sometimes coma within hours) → "
            "partial recovery but always with residual deficit. Forms: (1) CLASSIC CHILDHOOD (2-5y): "
            "cerebellar ataxia, spasticity; episodic crises; episodic coma; slow deterioration; "
            "death in 2nd-4th decade. (2) INFANTILE (severe, onset <1y): rapid severe course; "
            "seizures; death before age 5. (3) JUVENILE/ADULT (onset >5y): milder; longer survival. "
            "(4) OVARIOLEUKODYSTROPHY (OL): FEMALES only; ovarian failure (premature menopause, "
            "amenorrhoea) + mild neurological disease; milder brain phenotype. "
            "MRI: WHITE MATTER DISAPPEARS — T2 hyperintense, T1 hypointense; eventually white matter "
            "becomes ISOINTENSE TO CSF (literally vanishes on MRI); U-fibres spared until late; "
            "frontal and temporal lobes; periventricular. MRS: lactate in white matter; NAA LOW "
            "(opposite of Canavan); lipid peaks."
        ),
        "disease": (
            "EIF2B5 encodes eIF2B epsilon (721 aa, 82 kDa), the catalytic subunit of the eIF2B GEF "
            "complex. eIF2B recycles eIF2-GDP → eIF2-GTP, enabling global translation initiation. "
            "When cells encounter stress (infection, ER stress, oxidative stress, amino acid deprivation), "
            "kinases (PERK, GCN2, HRI, PKR) phosphorylate eIF2alpha → INHIBIT eIF2B → "
            "translation arrest + selective ISR activation. VWM mutations → eIF2B hypoactive "
            "at baseline → HYPERSENSITIVE to eIF2alpha phosphorylation → severe translation "
            "arrest on stress → acute oligodendrocyte/astrocyte degeneration.\n\n"
            "WHY FEVER/TRAUMA TRIGGERS CRISIS: PERK (ER stress kinase) and HRI (haem-regulated) "
            "activated by fever; GCN2 by nutrient deprivation during illness; PKR by viral dsRNA. "
            "All converge on eIF2alpha phosphorylation → eIF2B inhibition → cell death in VWM.\n\n"
            "GENETICS: Any of 5 EIF2B subunit genes (EIF2B1-5) can cause VWM; EIF2B5 = 65% of alleles. "
            "p.Arg113His (EIF2B5) — founder in Cree and other populations. Arg195His — severe infantile. "
            "Most mutations are in the catalytic domain of EIF2B5.\n\n"
            "TREATMENT: INTEGRATED STRESS RESPONSE INHIBITOR (ISRIB) is the most promising: "
            "reverses eIF2alpha phosphorylation effects; rescues VWM mouse models completely. "
            "ISRIB in clinical trials (2023-2025). Guanabenz and sephin1 also target ISR. "
            "PREVENTION OF CRISES: antipyretics PROMPTLY at first fever (paracetamol/ibuprofen); "
            "strict head protection; avoid febrile illnesses; vaccinations important but monitor. "
            "VWM crisis is a MEDICAL EMERGENCY — hospital admission, IV hydration, seizure control."
        ),
        "hallmark": (
            "EIF2B5 HALLMARKS: "
            "(1) STRESS-TRIGGERED EPISODIC CRISES: fever, head trauma, fright → acute coma/deterioration "
            "within hours; parents MUST be educated to treat fever immediately. "
            "(2) WHITE MATTER VANISHES ON MRI: eventually becomes CSF-isointense (liquefied white matter). "
            "(3) ISRIB CLINICAL TRIALS: ISR inhibitor reverses VWM in animal models; most promising "
            "disease-modifying therapy in any leukodystrophy currently. "
            "(4) OVARIOLEUKODYSTROPHY: FEMALES with VWM can have premature ovarian failure (POF) as "
            "first/only symptom — genital-neurological association. "
            "(5) EIF2B5 MOST COMMON SUBUNIT (65%): but any EIF2B1-5 mutation causes same phenotype. "
            "(6) MRS LACTATE IN WHITE MATTER: raised lactate despite NO mitochondrial pathology — "
            "anaerobic glycolysis in dying cells. "
            "(7) FEVER = EMERGENCY: VWM crisis is life-threatening; fever management is a primary "
            "life-saving intervention — more important than most medications."
        ),
        "nbs_marker": "Not on standard NBS; episodic crises + MRI white-matter vanishing = clinical diagnosis",
        "key_biomarker": "MRI white-matter vanishing + MRS lactate; EIF2B1-5 panel sequencing; no reliable serum biomarker",
        "severity_spectrum": "Infantile most severe; classic childhood intermediate; juvenile/adult mild; ovarioleukodystrophy mildest",
        "seed_offset": 7,
    },
]


def _make_patients(gene_data, n=40):
    rng = random.Random(SEED_BASE + gene_data["seed_offset"])
    gene = gene_data["gene"]
    onset_lo, onset_hi = gene_data["onset_range_y"]
    patients = []
    for i in range(n):
        age_onset = round(rng.uniform(onset_lo, onset_hi), 1)
        age_dx    = round(age_onset + rng.uniform(0.5, 5.0), 1)
        age_now   = round(age_dx + rng.uniform(0.5, 10.0), 1)
        dx_delay  = round(age_dx - age_onset, 1)
        sex       = rng.choice(["M", "F", "M", "M"]) if gene in ("ABCD1", "PLP1") else rng.choice(["M", "F"])
        severity  = rng.choices(
            ["Severe", "Moderate", "Mild"],
            weights=[55, 30, 15] if gene in ("GALC", "ASPA") else
                    [40, 40, 20] if gene in ("ARSA", "MLC1") else
                    [35, 35, 30],
            k=1
        )[0]
        mri_pattern = {
            "ARSA":  "Posterior T2 hyperintensity + tigroid pattern",
            "GALC":  "Deep WM + cerebellar + corticospinal tracts T2",
            "PLP1":  "Diffuse hypomyelination (near-complete)",
            "ABCD1": "Posterior-parieto-occipital T2 + gadolinium enhancement",
            "ASPA":  "Diffuse WM + U-fibres + no sparing",
            "GFAP":  "Frontal-periventricular T2 + BG + brainstem",
            "MLC1":  "Diffuse WM swelling + temporal subcortical cysts",
            "EIF2B5":"WM vanishing (CSF-isointense) + U-fibre sparing early",
        }[gene]
        treatment = {
            "ARSA":  rng.choice(["Libmeldy (OTL-200)", "Supportive only", "HSCT (allogeneic)"]),
            "GALC":  rng.choice(["HSCT pre-symptomatic", "Supportive only", "Cord-blood HSCT"]),
            "PLP1":  rng.choice(["Supportive only", "Baclofen + physiotherapy", "Anti-epileptics + PT"]),
            "ABCD1": rng.choice(["Skysona gene therapy", "HSCT (early CALD)", "Lorenzo's Oil + surveillance", "Hydrocortisone (adrenal only)"]),
            "ASPA":  rng.choice(["Supportive only", "GTA (acetate supplement, trial)", "AAV gene therapy (trial)"]),
            "GFAP":  rng.choice(["Supportive only", "Anti-epileptics", "ASO trial (GFAPsynergy)"]),
            "MLC1":  rng.choice(["Supportive only", "Acetazolamide (pilot)", "Anti-epileptics"]),
            "EIF2B5":rng.choice(["ISRIB trial", "Fever prevention + supportive", "Anti-epileptics + ISRIB"]),
        }[gene]
        alive = rng.random() > (0.25 if gene in ("GALC", "ASPA") else 0.15)
        patients.append({
            "patient_id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "age_onset_y": age_onset,
            "age_dx_y": age_dx,
            "age_now_y": age_now,
            "dx_delay_y": dx_delay,
            "sex": sex,
            "severity": severity,
            "mri_pattern": mri_pattern,
            "treatment": treatment,
            "alive": alive,
        })
    return patients


def _all_patients():
    all_pts = []
    for g in LD_GENES:
        all_pts.extend(_make_patients(g))
    return all_pts


# ─── Public API ───────────────────────────────────────────────────────────────

def get_overview():
    pts = _all_patients()
    n = len(pts)
    alive = sum(1 for p in pts if p["alive"])
    sev_counts = {"Severe": 0, "Moderate": 0, "Mild": 0}
    for p in pts:
        sev_counts[p["severity"]] += 1
    avg_delay = round(sum(p["dx_delay_y"] for p in pts) / n, 2)
    avg_onset = round(sum(p["age_onset_y"] for p in pts) / n, 2)
    gene_counts = {}
    for p in pts:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1
    approved_therapies = [
        "Libmeldy (ARSA/MLD) — EMA 2020",
        "Skysona/elivaldogene (ABCD1/X-ALD) — FDA 2022",
        "HSCT pre-symptomatic (GALC/Krabbe)",
        "Hydrocortisone (ABCD1 adrenal insufficiency)",
    ]
    return {
        "atlas": "Leukodystrophy Atlas",
        "subtitle": "Complete 8-Gene Leukodystrophy Reference",
        "description": (
            "The Leukodystrophy Atlas covers 8 major heritable white-matter disorders spanning "
            "lysosomal enzymes (ARSA/MLD, GALC/Krabbe), myelin proteins (PLP1/PMD), peroxisomal "
            "transporters (ABCD1/X-ALD), metabolic enzymes (ASPA/Canavan), structural astrocyte "
            "proteins (GFAP/Alexander), astrocyte ion channels (MLC1), and translation initiation "
            "(EIF2B5/VWM). MRI pattern is the primary diagnostic guide; two have approved gene therapies."
        ),
        "genes": [g["gene"] for g in LD_GENES],
        "n_genes": len(LD_GENES),
        "total_patients": n,
        "patients_alive": alive,
        "survival_pct": round(100 * alive / n, 1),
        "avg_onset_y": avg_onset,
        "avg_dx_delay_y": avg_delay,
        "severity_distribution": sev_counts,
        "gene_counts": gene_counts,
        "approved_therapies": approved_therapies,
        "therapies_in_trials": [
            "ISRIB (EIF2B5/VWM) — most advanced ISR inhibitor",
            "AAV-ASPA gene therapy (Canavan) — Phase 1-2",
            "GFAP ASO / GFAPsynergy (Alexander) — Phase 1-2",
            "AAV-GALC (Krabbe) — pre-clinical to Phase 1",
            "Thyroid hormone T3 (PLP1/PMD) — pilot",
        ],
        "key_teaching": [
            "ARSA PSEUDODEFICIENCY: low enzyme ≠ MLD — confirm with urine sulfatides + sequencing",
            "GALC PSYCHOSINE is the toxic metabolite (not galactosylceramide)",
            "PLP1 DUPLICATION (70%) causes PMD — MLPA mandatory, not just sequencing",
            "ABCD1 no genotype-phenotype: annual MRI for all boys; adrenal function mandatory",
            "ASPA MRS NAA markedly elevated = pathognomonic for Canavan disease",
            "GFAP all mutations dominant gain-of-function (NOT haploinsufficiency)",
            "MLC1 very slow; MLC2B (HEPACAM) is REMITTING benign — dramatically different prognosis",
            "EIF2B5 fever = emergency; ISRIB is the most promising therapy in any leukodystrophy",
        ],
        "mri_patterns": {
            "ARSA": "Posterior-predominant bilateral T2; tigroid/leopard-skin; U-fibres spared early",
            "GALC": "Deep WM + cerebellar + corticospinal; T2 high; early MRI can be normal in NBS",
            "PLP1": "Diffuse hypomyelination — near-complete white-matter T2 hyperintensity like newborn",
            "ABCD1": "Parieto-occipital posterior T2 + gadolinium enhancement at active edge (CALD)",
            "ASPA": "Diffuse WM including U-fibres; MRS NAA markedly elevated",
            "GFAP": "Frontal subcortical > periventricular; BG T2 high; brainstem involved; type I",
            "MLC1": "Diffuse WM swelling + subcortical temporal + frontoparietal cysts from infancy",
            "EIF2B5": "WM vanishes (CSF-isointense); U-fibres spared early; MRS lactate",
        },
        "seeds_used": f"{SEED_BASE}–{SEED_BASE + len(LD_GENES) - 1}",
    }


def get_breakdown():
    result = []
    for gd in LD_GENES:
        pts = _make_patients(gd)
        sev = {"Severe": 0, "Moderate": 0, "Mild": 0}
        for p in pts:
            sev[p["severity"]] += 1
        avg_delay = round(sum(p["dx_delay_y"] for p in pts) / len(pts), 1)
        avg_onset = round(sum(p["age_onset_y"] for p in pts) / len(pts), 1)
        survival = round(100 * sum(1 for p in pts if p["alive"]) / len(pts), 1)
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        top_tx = sorted(treatments.items(), key=lambda x: -x[1])[:3]
        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "inheritance": gd["inheritance"],
            "ld_subgroup": gd["ld_subgroup"],
            "n_patients": len(pts),
            "severity_distribution": sev,
            "avg_onset_y": avg_onset,
            "avg_dx_delay_y": avg_delay,
            "survival_pct": survival,
            "top_treatments": [{"treatment": t, "n": n} for t, n in top_tx],
            "hallmark": gd["hallmark"],
            "nbs_marker": gd["nbs_marker"],
            "key_biomarker": gd["key_biomarker"],
            "severity_spectrum": gd["severity_spectrum"],
            "disease_summary": gd["disease"][:500] + "…",
            "phenotype_summary": gd["phenotype"][:400] + "…",
            "gene_class": gd["gene_class"][:400] + "…",
        })
    return {"breakdown": result, "total_genes": len(result)}


def get_definitions():
    return {
        "definitions": [
            {"term": "Leukodystrophy", "definition": "Heritable disorder of myelin formation or maintenance; primary target is oligodendrocytes and/or myelin sheath proteins; MRI is primary diagnostic tool."},
            {"term": "Hypomyelination", "definition": "Reduced or absent myelin formation; MRI T2 hyperintense (bright) throughout white matter; PMD/PLP1 is the classic example."},
            {"term": "Demyelination", "definition": "Destruction of previously formed normal myelin; MRI shows progressive white-matter loss; MLD and Krabbe are demyelinating."},
            {"term": "Vacuolating Leukodystrophy", "definition": "Leukodystrophy where fluid vacuoles accumulate within the myelin sheath (intramyelinic oedema); MLC1 and VWM/EIF2B5 are examples."},
            {"term": "Rosenthal Fibres", "definition": "Electron-dense astrocytic inclusions containing GFAP + alphaB-crystallin + HSP27 + ubiquitin; PATHOGNOMONIC for Alexander disease (GFAP mutations)."},
            {"term": "Psychosine", "definition": "Galactosylsphingosine; toxic lipid accumulating in Krabbe disease (GALC deficiency); selectively destroys oligodendrocytes and Schwann cells at nanomolar concentrations; primary pathogenic metabolite."},
            {"term": "Sulfatide", "definition": "3-O-sulfo-galactosylceramide; major myelin glycolipid catabolised by ARSA; accumulates in MLD causing metachromatic deposits (stain brown-red with toluidine blue under polarised light)."},
            {"term": "ARSA Pseudodeficiency", "definition": "c.1049A>G (N350S) + c.444+1G>A alleles give low ARSA enzyme activity WITHOUT clinical MLD; sulfatide excretion is NORMAL; carrier frequency ~2% in Europeans; must exclude by molecular testing."},
            {"term": "N-Acetylaspartate (NAA)", "definition": "Most abundant amino acid in brain after glutamate; catabolised by ASPA in oligodendrocytes; dramatically elevated in Canavan disease; the MRS NAA peak is the specific diagnostic biomarker."},
            {"term": "Integrated Stress Response (ISR)", "definition": "Cellular stress signalling pathway: kinases (PERK, GCN2, HRI, PKR) phosphorylate eIF2alpha → eIF2B inhibition → global translation arrest + selective ATF4/CHOP activation; hypersensitive in VWM/EIF2B5."},
            {"term": "ISRIB", "definition": "Integrated Stress Response InhiBitor; small molecule that bypasses eIF2alpha phosphorylation to restore eIF2B GEF activity; fully rescues VWM mouse models; in clinical trials for VWM (EIF2B5 mutations)."},
            {"term": "Globoid Cells", "definition": "Multinucleated macrophages (phagocytes) clustered around blood vessels, containing engulfed galactosylceramide; PATHOGNOMONIC for Krabbe disease (GALC deficiency) on brain biopsy or histology."},
            {"term": "Loes Score", "definition": "Radiological severity scale (0-34) for X-ALD (ABCD1/CALD) measuring gadolinium-enhancing lesion extent; score ≤4 + gadolinium enhancement = eligibility for Skysona gene therapy or HSCT."},
            {"term": "Libmeldy (OTL-200)", "definition": "Atidarsagene autotemcel; ex-vivo lentiviral ARSA gene therapy for MLD; EMA-approved November 2020 for pre-symptomatic early-juvenile or early-symptomatic early-juvenile MLD; >90% stabilisation at 3 years."},
            {"term": "Skysona (elivaldogene autotemcel)", "definition": "Lenti-D ex-vivo lentiviral ABCD1 gene therapy for early CALD; FDA-approved July 2022 for boys 4-17 with early cerebral ALD (Loes ≤4 + gadolinium enhancement); 90% MFD-free at 2 years."},
            {"term": "VLCFA", "definition": "Very-Long-Chain Fatty Acids (C22+, especially C24:0 and C26:0); accumulate in X-ALD (ABCD1 deficiency); elevated in plasma of >99% of affected males and ~85% of female carriers; primary diagnostic biomarker."},
            {"term": "Ovarioleukodystrophy", "definition": "Sub-phenotype of VWM (EIF2B mutations) in females; premature ovarian failure (amenorrhoea, infertility) + mild or late neurological disease; ovarian failure may precede neurological symptoms by years."},
            {"term": "MLC2B", "definition": "Dominant negative HEPACAM mutations causing benign remitting megalencephalic leukodystrophy; MRI abnormalities normalise spontaneously; completely different prognosis from MLC1 (MLC1 gene) which is progressive — critical distinction."},
            {"term": "Lorenzo's Oil", "definition": "4:1 mixture of glyceryl trioleate + glyceryl trierucate; normalises plasma VLCFA in ~90% of X-ALD patients within 4 weeks; does NOT halt or reverse neurological progression in symptomatic CALD; may reduce CALD onset risk in asymptomatic boys (unproven level I evidence)."},
            {"term": "Spongy Degeneration", "definition": "Histological finding in Canavan disease; fluid-filled vacuoles within myelin sheaths and astrocytes (not extracellular) giving sponge-like appearance; caused by NAA osmotic accumulation + acetate deficiency for myelin lipid synthesis."},
        ]
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print("=== LEUKODYSTROPHY ATLAS OVERVIEW ===")
    print(f"Genes: {ov['genes']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Survival: {ov['survival_pct']}%")
    print(f"Avg onset: {ov['avg_onset_y']} y")
    print(f"Avg dx delay: {ov['avg_dx_delay_y']} y")
    print(f"Approved therapies: {ov['approved_therapies']}")
    bd = get_breakdown()
    print(f"\n=== BREAKDOWN ({bd['total_genes']} genes) ===")
    for g in bd["breakdown"]:
        print(f"  {g['gene']}: {g['n_patients']} pts, onset {g['avg_onset_y']}y, delay {g['avg_dx_delay_y']}y, survival {g['survival_pct']}%")
    defs = get_definitions()
    print(f"\n=== DEFINITIONS: {len(defs['definitions'])} terms ===")
