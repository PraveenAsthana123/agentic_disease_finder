"""Hereditary Leukodystrophy Atlas — 8-Gene Reference
ABCD1-ARSA-GALC-PLP1-GJC2-POLR3A-EIF2B5-ADAR
320 patients (8 x 40), seeds 2614-2621.
Endpoints: /api/hereditary-leukodystrophy-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "ABCD1",
        "protein": (
            "ABCD1 -- Xq28 XLR -- 745aa -- ALD-Protein-ALDP-84kDa-"
            "Peroxisomal-VLCFA-ABC-Transporter-X-ALD-AMN-XLR -- OMIM-Gene-300371-Disease-ALD-300100"
        ),
        "locus": "Xq28",
        "protein_size": "745 aa / 84 kDa",
        "inheritance": (
            "XLR (X-linked recessive); males severely affected; "
            "X-linked adrenoleukodystrophy (X-ALD); most common X-linked leukodystrophy (1:20,000 males); "
            "Phenotypes: cerebral ALD (CALD) ~35% boys before 12yr — fatal inflammatory demyelination; "
            "adrenomyeloneuropathy (AMN) ~40-45% adult males — progressive spastic paraplegia/neuropathy; "
            "Addison-only ~10%; asymptomatic males; "
            "Carrier females: ~50% develop mild AMN-like myeloneuropathy by age 50; "
            "Genotype-phenotype correlation: POOR — same mutation gives CALD in one brother, AMN in another; "
            "X-ALD added to USA RUSP (NBS) 2016 — C26:0-lysophosphatidylcholine on DBS"
        ),
        "disease_category": (
            "X-linked Adrenoleukodystrophy (X-ALD); Peroxisomal VLCFA storage disorder; "
            "ABCD1 encodes ALDP — peroxisomal half-ABC transporter for very-long-chain fatty acids (VLCFA C22:0-C26:0); "
            "Loss of ABCD1 → VLCFA accumulate in plasma, adrenal cortex, CNS white matter; "
            "CALD: posterior inflammatory cerebral demyelination — MRI advancing anterior from parieto-occipital; "
            "AMN: non-inflammatory axonal degeneration — slowly progressive spastic paraplegia + neuropathy; "
            "Adrenal insufficiency (primary Addison) in ~70% males — life-threatening if unrecognised; "
            "HSCT curative for CALD (Loes score ≤9 + gadolinium enhancement); "
            "Skysona (elivaldogene autotemcel) HSC gene therapy FDA 2022 for early active CALD"
        ),
        "disease_pathway": (
            "ABCD1 encodes ALDP (ALD protein), a peroxisomal half-ABC transporter forming homodimers "
            "and heterodimers with ABCD2/3. ALDP transports very-long-chain fatty acid (VLCFA) CoA esters "
            "into peroxisomes for beta-oxidation. Loss of ABCD1 → VLCFA fail to enter peroxisomes → "
            "VLCFA (C22:0, C24:0, C26:0) accumulate in plasma, adrenal cortex, brain white matter, testes. "
            "CALD mechanism: VLCFA incorporate into phospholipid bilayers → membrane destabilisation → "
            "CD4+ T-cell and macrophage infiltration → perivascular cuffing → rapid inflammatory demyelination. "
            "Gadolinium enhancement = active blood-brain barrier breakdown = treatment window. "
            "AMN: non-inflammatory axonopathy of posterior columns and corticospinal tracts — slowly progressive; "
            "CALD trigger: UNKNOWN — trauma, infection, adolescent growth proposed as precipitants."
        ),
        "pathognomonic": (
            "CALD MRI POSTERIOR-ADVANCING PATTERN: Bilateral parieto-occipital T2/FLAIR hyperintensity "
            "advancing anterior — gadolinium-enhancing LEADING EDGE = ACTIVE INFLAMMATION = TREATMENT WINDOW; "
            "LOES SCORE (0-34): monitor cerebral involvement — Loes >9 = advanced, HSCT contraindicated; "
            "VLCFA PLASMA: C26:0 elevated; C26:0/C22:0 ratio elevated — confirms peroxisomal VLCFA disorder; "
            "Adrenal cortex: primary failure — Synacthen stimulation test (ACTH stimulation): low cortisol response; "
            "AMN: posterior column + corticospinal tract atrophy on spinal MRI; normal brain MRI initially; "
            "ADRENAL INSUFFICIENCY: hyperpigmentation (ACTH excess), hyponatraemia, hypoglycaemia; "
            "Carrier female: occasional MRI white matter change; mild lower-limb spasticity; VLCFA mildly elevated"
        ),
        "treatment": (
            "CEREBRAL ALD (CALD): "
            "ALLOGENEIC HSCT — curative for early CALD (Loes score ≤9 + gadolinium enhancement active); "
            "SKYSONA (elivaldogene autotemcel, LentiGlobin-ALD) — FDA 2022: autologous HSC gene therapy; "
            "avoids allogeneic GvHD; efficacy comparable to HSCT; "
            "Lorenzo's oil (VLC-FA-restricted diet + erucic/oleic acid): normalises plasma VLCFA; "
            "does NOT arrest established CALD; possible delay of onset in asymptomatic boys (unproven); "
            "ADRENAL INSUFFICIENCY: cortisol + fludrocortisone replacement MANDATORY; life-threatening crisis risk; "
            "AMN: antispastic (baclofen/tizanidine), neuropathic pain (gabapentin), physiotherapy; "
            "NBS PROTOCOL: annual brain MRI from age 3-12yr in ABCD1+ males; Synacthen test annually; "
            "GENETIC COUNSELLING: X-linked; carrier females; PGT/prenatal available"
        ),
        "key_features": [
            "ABCD1 (X-ALD): XLR; most common X-linked leukodystrophy (1:20,000 males); VLCFA peroxisomal storage",
            "CALD: boys 3-12yr; posterior T2 lesions advancing anterior; gadolinium enhancement = treatment window",
            "HSCT curative if Loes score ≤9 + gadolinium enhancement (active phase) — MRI TIMING CRITICAL",
            "Skysona (elivaldogene) autologous HSC gene therapy FDA 2022 — avoids allogeneic GvHD",
            "AMN: adult males; slowly progressive spastic paraplegia + neuropathy; non-inflammatory",
            "Adrenal insufficiency (Addison) ~70% males — cortisol replacement MANDATORY; life-threatening if missed",
            "Genotype-phenotype: POOR — same ABCD1 variant → CALD in one brother, AMN in another",
            "Annual MRI surveillance from age 3-12yr in all ABCD1+ males (USA NBS RUSP 2016)",
        ],
        "key_ddx": [
            "Multiple sclerosis: posterior lesions + enhancement — VLCFA normal in MS; ABCD1 sequencing discriminates",
            "Alexander disease (GFAP): frontal-predominant leukodystrophy; VLCFA normal; GFAP mutation",
            "Other peroxisomal disorders (PBD): multiorgan; phytanic + pipecolic + VLCFA elevated; more severe",
            "AMN vs HSP: spinal cord atrophy on MRI; VLCFA elevated in AMN; SPG4 SPAST normal VLCFA",
        ],
        "onset_age": 7.0,
        "wm_lesion_pct": 95,
        "adrenal_pct": 70,
        "spastic_pct": 88,
        "seed": 2614,
    },
    {
        "gene": "ARSA",
        "protein": (
            "ARSA -- 22q13.33 AR -- 507aa -- Arylsulfatase-A-62kDa-"
            "Lysosomal-Sulfatide-Sulfatase-MLD-AR -- OMIM-Gene-607574-Disease-MLD-250100"
        ),
        "locus": "22q13.33",
        "protein_size": "507 aa / 62 kDa",
        "inheritance": (
            "AR (biallelic loss of function); Metachromatic Leukodystrophy (MLD); "
            "Prevalence: ~1:40,000-1:160,000 live births; "
            "Late-infantile MLD: onset 1-2yr (most common ~50%, most severe, rapidly fatal); "
            "Juvenile MLD: onset 3-16yr (cognitive regression → motor decline); "
            "Adult MLD: onset >16yr (psychiatric/cognitive first — misdiagnosed schizophrenia); "
            "ARSA PSEUDODEFICIENCY: 10% general population have low ARSA enzyme activity with NORMAL sulfatide — NOT disease; "
            "Saposin B deficiency (PSAP gene): same MLD phenotype; normal ARSA enzyme; sulfatide assay required"
        ),
        "disease_category": (
            "Metachromatic Leukodystrophy (MLD); lysosomal storage disorder; "
            "ARSA cleaves sulfate from sulfatides (galactosylceramide-3-sulfate) in lysosomes — "
            "requires saposin B co-factor for substrate presentation; "
            "Loss of ARSA → sulfatide accumulates in oligodendrocytes, Schwann cells, visceral organs; "
            "Progressive demyelination CNS + PNS; "
            "METACHROMATIC GRANULES IN NERVE BIOPSY = PATHOGNOMONIC: "
            "sulfatide-laden Schwann cell lysosomes stain brown-red under polarised toluidine blue light; "
            "LIBMELDY (atidarsagene autotemcel) EMA 2020: HSC gene therapy — transformative if pre-symptomatic"
        ),
        "disease_pathway": (
            "ARSA (arylsulfatase A) is a lysosomal acid hydrolase that cleaves the sulfate ester bond "
            "from galactosylceramide-3-sulfate (sulfatide), producing galactosylceramide + sulfate. "
            "Saposin B acts as co-factor presenting sulfatide to ARSA in the lysosomal lumen. "
            "Loss of ARSA → lysosomal sulfatide accumulation in myelinating cells: "
            "oligodendrocytes (CNS) and Schwann cells (PNS) accumulate → undergo apoptosis → demyelination. "
            "Residual enzyme activity correlates with severity: <1% → late-infantile; 1-5% → juvenile; >5% → adult. "
            "METACHROMASIA mechanism: sulfatide-laden membrane fragments in Schwann cell lysosomes; "
            "toluidine blue forms metachromatic dye complex with sulfatide → shifts absorption peak → "
            "appears brown-red (not blue) under polarised light. "
            "CSF: elevated protein (demyelinating neuropathy contribution); NCV: markedly slow."
        ),
        "pathognomonic": (
            "METACHROMATIC GRANULES IN SURAL NERVE BIOPSY = PATHOGNOMONIC (historically; now rarely needed): "
            "Sulfatide deposits in Schwann cell lysosomes → toluidine blue under polarised light → brown-red METACHROMASIA; "
            "MRI: confluent periventricular T2/FLAIR hyperintensity — TIGROID PATTERN (sparing arcuate fibres) early; "
            "URINE SULFATIDE: elevated (quantitative test) — KEY to exclude ARSA pseudodeficiency; "
            "ARSA ENZYME ACTIVITY: low in leukocytes — BUT pseudodeficiency (activity low, sulfatide NORMAL) must be excluded; "
            "NCV: markedly slow (demyelinating neuropathy) — BOTH CNS + PNS involved; "
            "Late-infantile: regression of milestones at 1-2yr (walk→crawl→hypotonic); "
            "Adult MLD: frontal lobe syndrome / schizophrenia-like — psychiatric referral common → MRI key"
        ),
        "treatment": (
            "LIBMELDY (atidarsagene autotemcel, OTL-200) — EMA approved 2020: "
            "Autologous HSC gene therapy; lentiviral ARSA cDNA; "
            "Efficacy: pre-symptomatic late-infantile and early juvenile — near-normal motor/cognitive outcomes; "
            "Treated pre-symptomatically: 80%+ retain ambulation vs untreated (never walk); "
            "ALLOGENEIC HSCT: slows but does NOT stop late-infantile MLD; beneficial pre-symptomatic juvenile/adult; "
            "INTRATHECAL ERT (recombinant ARSA): limited CNS penetration; Phase II; "
            "NEWBORN SCREENING: not yet universal; NBS + Libmeldy is ideal pathway for late-infantile; "
            "Supportive: anti-epileptic (seizures 30%), neuropathic pain, PEG if dysphagia, physiotherapy; "
            "ARSA PSEUDODEFICIENCY: no treatment — not a disease; "
            "GENETIC COUNSELLING: AR; 25% recurrence; prenatal/PGT; saposin B excluded by urine sulfatide test"
        ),
        "key_features": [
            "ARSA (MLD): AR lysosomal; sulfatide accumulates; 1:40,000-1:160,000; oligodendrocyte + Schwann cell death",
            "Metachromatic granules nerve biopsy PATHOGNOMONIC (toluidine blue polarised light — brown-red)",
            "Libmeldy (atidarsagene) HSC gene therapy EMA 2020 — transformative if pre-symptomatic",
            "Late-infantile (1-2yr onset, 50%): most severe; rapidly fatal without treatment",
            "Adult MLD: schizophrenia-like onset — MRI white matter leukodystrophy KEY to diagnosis",
            "ARSA pseudodeficiency: 10% population have low enzyme but NORMAL sulfatide — NOT disease; CRITICAL DDx",
            "Urine sulfatide quantification MANDATORY before diagnosis — excludes pseudodeficiency",
            "NCV: markedly slow demyelinating neuropathy; both CNS + PNS involved; distinguishes from PMD (hypomyelination)",
        ],
        "key_ddx": [
            "ARSA pseudodeficiency: low enzyme, NORMAL urine sulfatides, no MRI — NOT MLD; critical to exclude",
            "Saposin B deficiency (PSAP): same MLD phenotype; ARSA activity NORMAL; elevated sulfatide confirms",
            "Krabbe (GALC): similar infantile; globoid cells (not metachromatic); less prominent PNS involvement early",
            "Adult MLD vs schizophrenia: MRI white matter + slow NCV + sulfatide assay discriminates",
        ],
        "onset_age": 2.5,
        "wm_lesion_pct": 98,
        "neuropathy_pct": 92,
        "seizure_pct": 30,
        "spastic_pct": 78,
        "seed": 2615,
    },
    {
        "gene": "GALC",
        "protein": (
            "GALC -- 14q31.3 AR -- 669aa -- Galactocerebrosidase-74kDa-"
            "Lysosomal-Psychosine-Galactosylceramide-Hydrolase-Krabbe-GLD-AR -- OMIM-Gene-606890-Disease-Krabbe-245200"
        ),
        "locus": "14q31.3",
        "protein_size": "669 aa / 74 kDa",
        "inheritance": (
            "AR (biallelic loss of function); Krabbe Disease (Globoid Cell Leukodystrophy, GLD); "
            "Prevalence: ~1:100,000; "
            "Classic infantile: onset <6m — most common 85-90%, most severe, rapidly fatal; "
            "Late-onset Krabbe: onset 6m-3yr, 3-8yr, or adult — residual GALC activity; "
            "GALC 30-kb deletion allele common in Northern European populations (~45% of alleles); "
            "Genotype-phenotype: missense with some residual activity → late-onset; null/deletion → infantile"
        ),
        "disease_category": (
            "Krabbe Disease (Globoid Cell Leukodystrophy, GLD); lysosomal storage disorder; "
            "GALC cleaves galactose from galactosylceramide AND psychosine (galactosylsphingosine); "
            "Loss of GALC → PSYCHOSINE accumulates — unique cytotoxin; kills oligodendrocytes + Schwann cells; "
            "GLOBOID CELLS: multinucleated macrophages (2-20 nuclei) with PAS-positive inclusions in white matter = PATHOGNOMONIC; "
            "Classic infantile: EXTREME IRRITABILITY (hyperalgesia — touch → screaming) PATHOGNOMONIC; "
            "HSCT pre-symptomatic: substantial benefit for late-onset; very limited for infantile; "
            "Krabbe added to USA RUSP NBS 2016 — GALC enzyme + psychosine second-tier"
        ),
        "disease_pathway": (
            "GALC (galactocerebrosidase) cleaves galactose from two key lysosomal substrates: "
            "1) Galactosylceramide: major myelin glycolipid — turnover generates ceramide; "
            "2) Psychosine (galactosylsphingosine): highly cytotoxic lysolipid. "
            "Loss of GALC → psychosine accumulates — it inserts into cellular membranes → "
            "destabilises lipid bilayers → activates caspase-3 apoptosis in oligodendrocytes and Schwann cells. "
            "Psychosine hypothesis: psychosine (not galactosylceramide) is the primary cytotoxic driver. "
            "Macrophages phagocytose myelin debris → overloaded with galactosylceramide → "
            "fuse into GLOBOID CELLS (multinucleated, PAS+, perivascular in white matter). "
            "Progressive: infantile course measured in months; late-onset in years. "
            "Psychosine in plasma/DBS: emerging biomarker for NBS second-tier and treatment monitoring."
        ),
        "pathognomonic": (
            "GLOBOID CELLS IN WHITE MATTER BIOPSY = PATHOGNOMONIC: "
            "Multinucleated macrophages (2-20 nuclei) with PAS-positive cytoplasmic inclusions; "
            "perivascular aggregates in demyelinated white matter (now rarely needed for diagnosis); "
            "EXTREME IRRITABILITY (hyperalgesia): PATHOGNOMONIC infantile presentation — "
            "touch or sound → paroxysmal screaming/stiffening; can be mistaken for colic; "
            "MRI: T2 hyperintensity cerebellum, corona radiata, posterior limb of internal capsule; "
            "progressive global white matter involvement; "
            "CSF PROTEIN: markedly elevated (>100 mg/dL) — demyelinating PNS + CNS component; "
            "GALC enzyme activity: markedly low (DBS or leukocytes); "
            "PSYCHOSINE (DBS/plasma): elevated — second-tier NBS marker + monitoring biomarker"
        ),
        "treatment": (
            "INFANTILE KRABBE: "
            "ALLOGENEIC HSCT PRE-SYMPTOMATIC (NBS-identified) — ONLY treatment; "
            "Pre-symptomatic HSCT mitigates disease but does NOT cure infantile Krabbe; "
            "NBS + HSCT improves outcomes compared to symptomatic presentation; "
            "Symptomatic infantile: HSCT does NOT reverse established damage; "
            "LATE-ONSET KRABBE (3-8yr pre-symptomatic): "
            "ALLOGENEIC HSCT — SUBSTANTIAL benefit; stabilises CNS demyelination; "
            "GENE THERAPY (investigational): AAV-GALC intrathecal + systemic; Phase I/II; "
            "Supportive: anti-convulsants, antispasticity, PEG, pain management (neuropathic/hyperalgesia); "
            "GENETIC COUNSELLING: AR; 25% recurrence; PGT/prenatal; GALC enzyme + psychosine NBS; "
            "NBS BENEFIT: identifies late-onset candidates (substantial HSCT benefit); "
            "infantile NBS benefit real but limited vs late-onset"
        ),
        "key_features": [
            "GALC (Krabbe/GLD): AR lysosomal; psychosine accumulation kills oligodendrocytes + Schwann cells",
            "Globoid cells (multinucleated PAS+ macrophages perivascular) in white matter PATHOGNOMONIC",
            "Extreme irritability/hyperalgesia (touch → screaming) PATHOGNOMONIC in infantile",
            "Classic infantile (<6m onset, 85-90%): rapidly fatal; HSCT pre-symptomatic — limited but real benefit",
            "Late-onset Krabbe (3-8yr): HSCT pre-symptomatic = substantial benefit",
            "Psychosine (plasma/DBS): emerging NBS second-tier marker + treatment monitoring biomarker",
            "CSF protein markedly elevated (>100 mg/dL) — combined CNS + PNS demyelination",
            "USA RUSP NBS 2016: GALC enzyme DBS + psychosine second-tier confirms late-onset candidates",
        ],
        "key_ddx": [
            "MLD (ARSA): metachromatic granules (not globoid cells); sulfatide elevated; peripheral NCV also slow",
            "GM1/GM2 gangliosidosis: cherry-red spot; different enzyme; no globoid cells; organomegaly",
            "Infantile GM2 (Tay-Sachs): cherry-red + hyperacusis; hexosaminidase A; no globoid cells",
            "Alexander disease: frontal predominance; Rosenthal fibres; GFAP; no globoid cells; macrocephaly",
        ],
        "onset_age": 3.0,
        "wm_lesion_pct": 97,
        "irritability_pct": 92,
        "neuropathy_pct": 85,
        "spastic_pct": 80,
        "seizure_pct": 35,
        "seed": 2616,
    },
    {
        "gene": "PLP1",
        "protein": (
            "PLP1 -- Xq22.2 XLR -- 276aa -- Proteolipid-Protein-1-PLP-DM20-30kDa-"
            "Major-CNS-Myelin-Structural-Protein-PMD-SPG2-XLR -- OMIM-Gene-300401-Disease-PMD-312080"
        ),
        "locus": "Xq22.2",
        "protein_size": "276 aa / 30 kDa",
        "inheritance": (
            "XLR (X-linked recessive); males severely affected; females mild-moderately affected carriers; "
            "Pelizaeus-Merzbacher Disease (PMD) / Spastic Paraplegia type 2 (SPG2); "
            "Most common X-linked hypomyelinating leukodystrophy; "
            "Duplications most common (60-70%): excess PLP1 → ER stress → oligodendrocyte death; "
            "Point mutations + deletions + null alleles account for remainder; "
            "Null PLP1 alleles → SPG2 (milder, adult-onset spastic paraplegia + PNS involvement); "
            "Connatal PMD: most severe (certain missense); classic PMD: intermediate (duplications); "
            "NYSTAGMUS AT BIRTH PATHOGNOMONIC"
        ),
        "disease_category": (
            "Pelizaeus-Merzbacher Disease (PMD) / Spastic Paraplegia type 2 (SPG2); "
            "PLP1 encodes proteolipid protein 1 (PLP/DM20) — major structural CNS myelin protein (~50% myelin protein mass); "
            "Mechanism: HYPOMYELINATION — myelin never forms adequately (not demyelination); "
            "Oligodendrocytes fail to produce/maintain compact myelin; "
            "Duplication → excess PLP → ER retention + UPR → oligodendrocyte apoptosis → hypomyelination; "
            "NYSTAGMUS AT BIRTH PATHOGNOMONIC for PMD; "
            "NO approved disease-modifying therapy; ASO/gene therapy in Phase I clinical trials"
        ),
        "disease_pathway": (
            "PLP1 encodes PLP (proteolipid protein) and its alternatively spliced isoform DM20. "
            "PLP is the most abundant CNS myelin protein, forming the hydrophobic core of compacted myelin. "
            "Mechanism by mutation class: "
            "1) DUPLICATION (most common): excess PLP protein → overloads ER folding capacity → "
            "ER stress + unfolded protein response (UPR) → oligodendrocyte apoptosis → hypomyelination; "
            "2) MISSENSE GOF: PLP misfolding → ER retention → UPR → same apoptotic pathway; "
            "3) NULL/DELETION (SPG2): loss of PLP → oligodendrocytes present but cannot properly compact myelin → "
            "axonal degeneration over years (paradox: milder than duplication clinically = less UPR stress); "
            "STATIC hypomyelination: myelin never formed; not progressive demyelination (mostly stable on MRI)."
        ),
        "pathognomonic": (
            "NYSTAGMUS AT BIRTH = PATHOGNOMONIC FOR PMD: "
            "Pendular or rotatory nystagmus within first weeks of life; improves somewhat but persists; "
            "Nystagmus + hypotonia in infant + leukodystrophy on MRI = PMD until proven otherwise; "
            "MRI: DIFFUSE T2/FLAIR HYPERINTENSITY throughout white matter (hypomyelination); "
            "Cerebellum, brainstem, internal capsule — all white matter involved; "
            "TIGROID PATTERN: patchy myelin islands (spared islands → tiger-stripe appearance on T2); "
            "ARRAY CGH/MLPA: detects PLP1 duplication (60-70%); "
            "SEQUENCING: detects point mutations; ALWAYS do both (duplication NOT detected by sequencing); "
            "MRS: markedly reduced NAA in white matter; "
            "Carrier females: may show mild MRI white matter changes; rarely symptomatic"
        ),
        "treatment": (
            "NO APPROVED DISEASE-MODIFYING THERAPY: "
            "ANTISENSE OLIGONUCLEOTIDE (ASO) — Phase I for PLP1 duplication: reduces PLP1 mRNA overexpression; "
            "AAV-based PLP1 gene replacement/silencing — preclinical; "
            "Allogeneic HSCT: very limited evidence; some MRI white matter improvement but minimal clinical benefit; "
            "Symptomatic management: "
            "ANTISPASTICITY: baclofen oral/intrathecal, tizanidine, botulinum toxin focal spasticity; "
            "Anti-epileptic: seizures ~30%; LEV or VPA; "
            "COMMUNICATION: AAC devices (most PMD patients non-verbal); "
            "Physiotherapy, occupational therapy, gastrostomy if dysphagia; "
            "GENETIC COUNSELLING: XLR; carrier testing; prenatal/PGT; ARRAY CGH + sequencing both mandatory"
        ),
        "key_features": [
            "PLP1 (PMD/SPG2): XLR; most common X-linked hypomyelinating leukodystrophy; males severely affected",
            "Nystagmus at birth PATHOGNOMONIC — pendular/rotatory; present from first weeks of life",
            "Hypomyelination (never forms adequately) NOT demyelination — mostly static pattern on MRI",
            "Duplication most common (60-70%): excess PLP → ER stress → oligodendrocyte apoptosis",
            "Null PLP1 → SPG2: milder adult-onset spastic paraplegia + PNS involvement",
            "NO approved disease-modifying therapy; ASO (PLP1 mRNA silencing) in Phase I for duplication",
            "ARRAY CGH/MLPA for duplication + sequencing for point mutations — both MANDATORY (not detected by one alone)",
            "Non-verbal in 65%+ — AAC communication devices central to management",
        ],
        "key_ddx": [
            "GJC2 PMLD: AR; no nystagmus at birth; connexin 47; milder hypomyelination; affects both sexes",
            "POLR3A 4H: hypomyelination + hypodontia + hypogonadism TRIAD; cerebellar atrophy; AR",
            "MLD (ARSA): progressive demyelination (not static); sulfatide; peripheral NCV very slow",
            "Oculomotor apraxia (ataxia-oculomotor): eye movement abnormality; different MRI; not hypomyelination",
        ],
        "onset_age": 0.2,
        "wm_lesion_pct": 99,
        "nystagmus_pct": 94,
        "spastic_pct": 88,
        "seizure_pct": 30,
        "seed": 2617,
    },
    {
        "gene": "GJC2",
        "protein": (
            "GJC2 -- 1q42.13 AR/AD -- 436aa -- Connexin-47-Cx47-46kDa-"
            "Oligodendrocyte-Astrocyte-Gap-Junction-PMLD-SPG44-AR-AD -- OMIM-Gene-608803-Disease-PMLD-608804"
        ),
        "locus": "1q42.13",
        "protein_size": "436 aa / 46 kDa",
        "inheritance": (
            "AR (biallelic loss of function) → PMLD (Pelizaeus-Merzbacher-Like Disease) — more severe; "
            "AD (heterozygous hypomorphic) → SPG44 (Spastic Paraplegia type 44) — milder, adult; "
            "Most common cause of PMLD after PLP1 exclusion; "
            "Prevalence PMLD (all causes): ~1:90,000; GJC2 accounts for significant fraction; "
            "Both sexes equally affected (AR) — vs PLP1 (XLR males); "
            "Nystagmus less prominent than PLP1-PMD (or absent)"
        ),
        "disease_category": (
            "Pelizaeus-Merzbacher-Like Disease type 1 (PMLD1) / Spastic Paraplegia type 44 (SPG44); "
            "GJC2 encodes Connexin 47 (Cx47) — gap junction channel protein in oligodendrocytes; "
            "Cx47 forms oligodendrocyte-astrocyte heterotypic channels (Cx47-Cx43); "
            "These channels essential for K+ spatial buffering + metabolic coupling in CNS myelin maintenance; "
            "Loss of Cx47 → gap junction coupling failure → oligodendrocyte metabolic vulnerability → hypomyelination; "
            "MILDER HYPOMYELINATION than PLP1-PMD; partial myelin formation present; "
            "SPG44: adult-onset pure spastic paraplegia; subtle white matter MRI changes"
        ),
        "disease_pathway": (
            "GJC2 encodes Connexin 47 (Cx47), expressed exclusively in oligodendrocytes. "
            "Cx47 forms heterotypic gap junction channels with astrocytic Connexin 43 (Cx43) — "
            "oligodendrocyte-astrocyte gap junctions buffer K+ in periaxonal space. "
            "During high-frequency axonal firing: K+ efflux into periaxonal space → rapidly cleared via "
            "oligodendrocyte Cx47-Cx43 channels → astrocytes → spatial buffering throughout glial syncytium. "
            "Loss of Cx47 → K+ accumulates → periaxonal hyperexcitability + oligodendrocyte metabolic stress "
            "→ impaired myelination. "
            "Cx47 also pairs with Cx32 (GJB1) in oligodendrocyte-oligodendrocyte channels. "
            "AR PMLD: both alleles non-functional → severe K+ buffering failure → hypomyelination. "
            "AD SPG44: one functional allele → partial K+ buffering → adult-onset axonopathy only."
        ),
        "pathognomonic": (
            "HYPOMYELINATION ON MRI (T2 diffuse white matter hyperintensity) — MILDER THAN PLP1-PMD; "
            "Partial myelin present (less severe than complete hypomyelination of PMD); "
            "AUTOSOMAL RECESSIVE pattern — both sexes; vs XLR for PLP1; "
            "NYSTAGMUS LESS PROMINENT or ABSENT — key distinction from PLP1-PMD; "
            "Cerebellar atrophy in subset; cerebellar signs (ataxia + dysmetria) alongside spasticity; "
            "NCV: mild slowing (less dramatic than MLD demyelinating neuropathy); "
            "GJC2 sequencing: biallelic variants confirm PMLD; monoallelic → SPG44; "
            "MRI: periventricular + subcortical T2 hyperintensity; internal capsule may be partially myelinated; "
            "After PLP1 exclusion in AR hypomyelination: GJC2 most likely gene → sequence first"
        ),
        "treatment": (
            "NO APPROVED DISEASE-MODIFYING THERAPY; "
            "ANTISPASTICITY: baclofen oral/intrathecal; tizanidine; "
            "Physiotherapy + occupational therapy — functional rehabilitation; "
            "Communication support (less severely affected than PLP1-PMD; many retain some speech); "
            "Anti-epileptic if seizures (less common than in PMD); "
            "SPG44: antispasticity + physiotherapy; milder course; many remain ambulant; "
            "GENE THERAPY: no clinical trials for GJC2; GJB1 (Cx32) CMT trials may inform; "
            "GENETIC COUNSELLING: AR for PMLD (25% recurrence); AD for SPG44 (50% dominant); "
            "MRI surveillance: static hypomyelination — monitor for any progression"
        ),
        "key_features": [
            "GJC2 (PMLD/SPG44): AR biallelic → PMLD; AD monoallelic → SPG44 adult spastic paraplegia",
            "Connexin 47 oligodendrocyte-astrocyte gap junctions — K+ buffering failure → hypomyelination",
            "Most common cause of PMLD after PLP1 exclusion (~1:90,000 PMLD prevalence)",
            "MILDER hypomyelination than PLP1-PMD; nystagmus less prominent or ABSENT",
            "AR inheritance — both sexes affected equally (vs XLR for PLP1)",
            "NO approved disease-modifying therapy; symptomatic management as for PMD",
            "SPG44: adult pure spastic paraplegia; subtle MRI white matter changes; monoallelic GJC2",
            "GJC2 sequencing after PLP1 exclusion as first AR hypomyelination step",
        ],
        "key_ddx": [
            "PLP1-PMD (XLR): nystagmus at birth; duplication most common; males only; more severe",
            "POLR3A 4H: AR; hypomyelination + hypodontia + hypogonadism TRIAD; cerebellar atrophy",
            "EIF2B5 VWM: stress-triggered episodes; white matter vanishes (fluid signal); EPISODIC",
            "Hereditary spastic paraplegias (SPG4/SPG7): MRI often normal; pure spastic paraplegia; no white matter T2",
        ],
        "onset_age": 1.5,
        "wm_lesion_pct": 95,
        "nystagmus_pct": 45,
        "spastic_pct": 92,
        "seizure_pct": 20,
        "cerebellar_pct": 55,
        "seed": 2618,
    },
    {
        "gene": "POLR3A",
        "protein": (
            "POLR3A -- 10q22.3 AR -- 1390aa -- RNA-Polymerase-III-Subunit-A-RPC1-155kDa-"
            "Largest-Pol-III-Catalytic-Subunit-4H-POLR3-HLD-AR -- OMIM-Gene-614258-Disease-POLR3-HLD-607694"
        ),
        "locus": "10q22.3",
        "protein_size": "1390 aa / 155 kDa",
        "inheritance": (
            "AR (biallelic hypomorphic/loss of function); "
            "POLR3-Related Leukodystrophy (POLR3-HLD) / 4H Syndrome; "
            "4H = Hypomyelination + Hypodontia + Hypogonadotropic Hypogonadism; "
            "Also caused by POLR3B (most common), POLR1C, POLR3K — panel testing required; "
            "POLR3A variants: hypomorphic (complete loss embryo-lethal — Pol III essential for viability); "
            "Compound heterozygotes (one splice + one missense) most common; "
            "~50% of hypomyelinating leukodystrophy after PLP1/GJC2 exclusion"
        ),
        "disease_category": (
            "POLR3-Related Leukodystrophy (POLR3-HLD) / 4H Syndrome; "
            "POLR3A encodes RPC1, largest catalytic subunit of RNA Polymerase III (Pol III); "
            "Pol III transcribes: 5S rRNA, tRNAs, U6 snRNA, 7SL RNA — all non-coding RNAs critical for translation; "
            "4H TRIAD = PATHOGNOMONIC: Hypomyelination + Hypodontia + Hypogonadotropic Hypogonadism; "
            "Cerebellar atrophy common (dentate nucleus + cerebellar white matter); "
            "Dental X-ray mandatory: oligodontia, peg teeth, delayed eruption confirms; "
            "Endocrine: central hypogonadism (FSH/LH low/normal + low sex steroids); "
            "Sex hormone replacement MANDATORY for bone health + cardiovascular protection"
        ),
        "disease_pathway": (
            "RNA Polymerase III (Pol III) is the nuclear enzyme transcribing short non-coding RNAs: "
            "5S rRNA (ribosome component), transfer RNAs (all 45 cytoplasmic tRNA species), "
            "U6 snRNA (spliceosome), 7SL RNA (signal recognition particle). "
            "POLR3A encodes RPC1, the catalytic subunit carrying the active site. "
            "Hypomorphic POLR3A → reduced Pol III transcriptional output selectively in highest-demand tissues: "
            "Oligodendrocytes: require massive tRNA output for myelin protein (MBP/PLP) translation → "
            "reduced tRNA → insufficient myelin protein synthesis → hypomyelination. "
            "Hypothalamic GnRH neurons: Pol III-dependent → GnRH deficiency → hypogonadotropic hypogonadism. "
            "Dental follicle cells: Pol III required for enamel/dentin matrix protein synthesis → hypodontia. "
            "Cerebellar Purkinje cells + dentate nucleus: Pol III sensitivity → cerebellar atrophy. "
            "The 4H triad maps to three tissues most sensitive to Pol III reduction."
        ),
        "pathognomonic": (
            "4H TRIAD PATHOGNOMONIC FOR POLR3-HLD: "
            "1. HYPOMYELINATION: MRI T2 diffuse white matter hyperintensity (static/slowly progressive); "
            "2. HYPODONTIA: Dental X-ray — oligodontia, peg teeth, delayed eruption, missing permanent teeth; "
            "3. HYPOGONADOTROPIC HYPOGONADISM: pubertal failure; low FSH/LH; low sex steroids = CENTRAL origin; "
            "CEREBELLAR ATROPHY: dentate nucleus and cerebellar white matter on MRI; "
            "Cerebellar signs: ataxia + dysmetria + intention tremor (alongside spasticity); "
            "DENTAL X-RAY MANDATORY: oligodontia present even before eruption age (tooth buds absent on OPG); "
            "ENDOCRINE: GnRH stimulation test distinguishes central (POLR3) from primary gonadal failure; "
            "POLR3 gene panel (POLR3A + POLR3B + POLR1C + POLR3K): biallelic variants confirm"
        ),
        "treatment": (
            "NO APPROVED DISEASE-MODIFYING THERAPY; "
            "SEX HORMONE REPLACEMENT MANDATORY: "
            "Females: oestrogen + progesterone replacement (puberty induction; bone mineral density; cardiovascular); "
            "Males: testosterone replacement; GnRH pulsatile pump for fertility; "
            "HYPODONTIA: dental implants when jaw growth complete; dentures interim; orthodontic management; "
            "ANTISPASTICITY: baclofen oral/intrathecal; tizanidine; "
            "CEREBELLAR ATAXIA: occupational therapy; aids; speech therapy for dysarthria; "
            "Anti-epileptic: seizures in 15-20%; LEV preferred; "
            "COGNITIVE SUPPORT: intellectual disability mild-moderate in some; education support; "
            "GENETIC COUNSELLING: AR; 25% recurrence; POLR3A + POLR3B panel; prenatal; "
            "MRI: annual (hypomyelination typically static); bone mineral density monitoring"
        ),
        "key_features": [
            "POLR3A (4H/POLR3-HLD): AR RNA Pol III; 4H triad PATHOGNOMONIC",
            "4H TRIAD: Hypomyelination + Hypodontia + Hypogonadotropic Hypogonadism — all three = POLR3",
            "Cerebellar atrophy (dentate nucleus) on MRI; cerebellar signs alongside spasticity",
            "Pol III transcribes tRNA/5S-rRNA: oligodendrocyte myelin protein synthesis impaired → hypomyelination",
            "Dental OPG X-ray mandatory: oligodontia confirms even before eruption age",
            "Central hypogonadism: low FSH/LH + sex steroids; HRT mandatory for bone + cardiovascular health",
            "Compound heterozygote (splice + missense) most common POLR3A genotype",
            "POLR3A + POLR3B + POLR1C panel required — same 4H phenotype from multiple genes",
        ],
        "key_ddx": [
            "PLP1-PMD: nystagmus at birth; XLR males; no dental/endocrine features; duplication most common",
            "Kallmann syndrome: hypogonadism + ANOSMIA; no leukodystrophy; no hypodontia; ANOS1/FGFR1 gene",
            "Septo-optic dysplasia (SOD): optic nerve hypoplasia; absent septum pellucidum; no hypomyelination",
            "POLR3B: same 4H phenotype as POLR3A; must panel-test both genes",
        ],
        "onset_age": 2.0,
        "wm_lesion_pct": 98,
        "hypodontia_pct": 85,
        "hypogonadism_pct": 78,
        "cerebellar_pct": 72,
        "spastic_pct": 80,
        "seizure_pct": 18,
        "seed": 2619,
    },
    {
        "gene": "EIF2B5",
        "protein": (
            "EIF2B5 -- 3q27.1 AR -- 721aa -- eIF2B-Epsilon-Subunit-80kDa-"
            "eIF2B-GEF-Catalytic-Integrated-Stress-Response-VWM-CACH-AR -- OMIM-Gene-603945-Disease-VWM-603896"
        ),
        "locus": "3q27.1",
        "protein_size": "721 aa / 80 kDa",
        "inheritance": (
            "AR (biallelic, often compound heterozygous); Vanishing White Matter Disease (VWM) / "
            "CACH (Childhood Ataxia with Central CNS Hypomyelination); "
            "EIF2B complex: 5 subunits (EIF2B1-5); mutations in any subunit cause VWM; "
            "EIF2B5 epsilon subunit: catalytic GEF domain — most mutations here; "
            "Prevalence: ~1:35,000; most common autosomal recessive leukodystrophy in children; "
            "Wide range: severe infantile to mild adult forms based on residual eIF2B GEF activity; "
            "Ovarioleukodystrophy: ovarian failure BEFORE neurological in some adult females"
        ),
        "disease_category": (
            "Vanishing White Matter Disease (VWM) / CACH; "
            "EIF2B5 is the catalytic epsilon subunit of eIF2B — the GDP→GTP exchange factor (GEF) for eIF2; "
            "eIF2-GTP allows Met-tRNA binding to 43S ribosomal complex for translational initiation; "
            "INTEGRATED STRESS RESPONSE (ISR): stress → eIF2alpha phosphorylation → ↓eIF2B GEF activity → ↓global translation; "
            "VWM: eIF2B5 mutations impair GEF activity → oligodendrocytes uniquely sensitive to ISR stress; "
            "STRESS-TRIGGERED ACUTE NEUROLOGICAL CRISES = PATHOGNOMONIC: febrile illness or minor head trauma → deterioration; "
            "White matter literally vanishes on MRI — replaced by CSF-signal fluid (vacuolation); "
            "ISRIB (ISR inhibitor): reverses VWM in mouse models — most promising therapeutic"
        ),
        "disease_pathway": (
            "EIF2B5 encodes the epsilon (catalytic) subunit of eIF2B, the guanine nucleotide exchange factor "
            "that converts eIF2-GDP to eIF2-GTP, enabling translational initiation. "
            "Integrated Stress Response (ISR): four stress kinases (HRI/PKR/PERK/GCN2) phosphorylate eIF2alpha-Ser51 "
            "→ phospho-eIF2alpha becomes competitive inhibitor of eIF2B → ↓global translation + ↑ATF4. "
            "VWM mutations reduce eIF2B GEF catalytic activity: under basal conditions, residual activity sufficient. "
            "Under ISR (fever, minor trauma): phospho-eIF2alpha accumulates → cannot be overcome by reduced eIF2B → "
            "translation fails specifically in cells with highest demand = oligodendrocytes → "
            "fail to maintain myelin proteins → white matter vacuolation + degeneration → "
            "MRI: white matter replaces with fluid (CSF-intensity). "
            "ISRIB: allosteric eIF2B activator → stabilises decameric eIF2B complex → overcomes ISR → "
            "restores translation → reverses VWM in mouse models (preclinical)."
        ),
        "pathognomonic": (
            "STRESS-TRIGGERED ACUTE NEUROLOGICAL DETERIORATION = PATHOGNOMONIC: "
            "FEBRILE ILLNESS or MINOR HEAD TRAUMA → sudden neurological worsening within hours-days; "
            "After crisis: PARTIAL recovery (baseline function lost progressively with each crisis); "
            "EPISODIC + STEPWISE neurological decline over years; "
            "MRI VANISHING WHITE MATTER: "
            "T2 hyperintensity with CSF-like signal in white matter (vacuolation/cystic change); "
            "Progressive replacement of white matter by fluid — white matter literally disappears; "
            "Corticospinal tracts relatively spared initially; diffuse eventually; "
            "OVARIAN FAILURE (POI) IN FEMALES: premature ovarian insufficiency — may present before neurological; "
            "CSF: oligoclonal bands absent; protein mild/normal; "
            "EIF2B1-5 PANEL: biallelic variants in any subunit confirm VWM"
        ),
        "treatment": (
            "NO APPROVED DISEASE-MODIFYING THERAPY: "
            "ISRIB (eIF2B GEF activator) — PRECLINICAL ONLY: reverses VWM in mouse models; "
            "multiple ISRIB analogues in pipeline; Phase I trials emerging; "
            "CRISIS PREVENTION — MOST CRITICAL MANAGEMENT: "
            "Early antipyretics: paracetamol/ibuprofen at FIRST sign of fever (before temperature rises); "
            "Influenza + COVID vaccination annually; "
            "CONTACT SPORTS PROHIBITED: minor head trauma → life-threatening crisis; "
            "Minor trauma crisis: IV dexamethasone may attenuate severity (anecdotal); "
            "OVARIAN FAILURE: oestrogen + progesterone replacement; fertility counselling; "
            "Anti-epileptic for seizures; antispastic for spasticity; "
            "WRITTEN EMERGENCY PROTOCOL for families: fever management + hospital threshold; "
            "GENETIC COUNSELLING: AR; 25% recurrence; EIF2B1-5 panel; prenatal"
        ),
        "key_features": [
            "EIF2B5 (VWM/CACH): AR; most common autosomal recessive leukodystrophy in children (1:35,000)",
            "Stress-triggered acute neurological crises (fever / minor head trauma) PATHOGNOMONIC",
            "White matter literally vanishes on MRI — replaced by CSF-signal fluid over years",
            "ISR (eIF2B GEF failure under stress) — oligodendrocytes most translation-dependent cells",
            "ISRIB (ISR inhibitor, eIF2B activator): reverses VWM in mice — most promising therapeutic target",
            "Ovarian failure (POI) in females — may precede neurological onset",
            "Contact sports PROHIBITED — minor head trauma triggers life-threatening neurological crisis",
            "Written emergency fever protocol mandatory for all VWM families",
        ],
        "key_ddx": [
            "MLD (ARSA): metachromatic granules; peripheral neuropathy; no episodic stress-triggered crises",
            "Alexander disease (GFAP): frontal predominance; Rosenthal fibres; macrocephaly; no episodic crises",
            "Megalencephalic leukoencephalopathy (MLC1): macrocephaly; vacuolating; MLC1/HEPACAM gene",
            "ADAR AGS6: calcifications + interferonopathy; elevated IFN-alpha CSF; no episodic fever crises",
        ],
        "onset_age": 3.0,
        "wm_lesion_pct": 99,
        "stress_trigger_pct": 88,
        "ovarian_failure_pct": 60,
        "seizure_pct": 35,
        "spastic_pct": 75,
        "seed": 2620,
    },
    {
        "gene": "ADAR",
        "protein": (
            "ADAR -- 1q21.3 AD-GOF/AR -- 1226aa -- Adenosine-Deaminase-RNA-Specific-ADAR1-136kDa-"
            "A-to-I-dsRNA-Editor-Interferonopathy-AGS6-AD-AR -- OMIM-Gene-146920-Disease-AGS6-615010"
        ),
        "locus": "1q21.3",
        "protein_size": "1226 aa / 136 kDa",
        "inheritance": (
            "AD heterozygous gain-of-function → AGS6 (most AGS6); "
            "AR biallelic → AGS6 (less common); "
            "Same gene (ADAR) AD LOF → Dyschromatosis Symmetrica Hereditaria (DSH — skin pigmentation only, NO brain); "
            "ADAR1 is most common AGS gene (~25-30% of all AGS cases); "
            "AGS spectrum: 7 genes (TREX1, RNASEH2A/2B/2C, SAMHD1, ADAR, IFIH1); "
            "De novo AD mutations occur; family history may be absent"
        ),
        "disease_category": (
            "Aicardi-Goutières Syndrome type 6 (AGS6); Type I Interferonopathy; Leukodystrophy with calcifications; "
            "ADAR1 edits adenosine → inosine (A→I) in endogenous Alu-repeat dsRNA — marks it as SELF; "
            "Loss/gain-of-function → unedited Alu-dsRNA → MDA5 (IFIH1) senses as NON-SELF → IFN-alpha cascade; "
            "PSEUDO-TORCH SYNDROME PATHOGNOMONIC: calcifications + leukodystrophy + microcephaly with NEGATIVE TORCH serology; "
            "CT BRAIN: basal ganglia + white matter + cerebellar calcifications — CT SUPERIOR TO MRI FOR CALCIUM; "
            "IFN-alpha CSF (>2 IU/mL) + Interferon Score (ISG15 blood): diagnostic; "
            "JAK inhibitors (baricitinib/ruxolitinib): most promising emerging treatment"
        ),
        "disease_pathway": (
            "ADAR1 (adenosine deaminase acting on RNA 1) catalyses adenosine-to-inosine (A→I) editing "
            "in cytoplasmic double-stranded RNA — primarily Alu retroelement repeat sequences that form "
            "fold-back dsRNA structures. "
            "Inosine-containing dsRNA is NOT recognised by MDA5 (IFIH1) pattern recognition receptor — "
            "ADAR1 editing marks endogenous Alu-dsRNA as SELF. "
            "Loss or gain-of-function mutations → insufficient Alu-dsRNA editing → unedited dsRNA accumulates → "
            "MDA5 detects as non-self foreign dsRNA → MAVS → IRF3/IRF7 → type I IFN (IFN-alpha/beta) transcription → "
            "JAK1/TYK2 → STAT1/STAT2 → ISG15, IFIT1, CXCL10 upregulation (Interferon Score). "
            "CNS: IFN-alpha toxic to oligodendrocytes → leukodystrophy; perivascular calcification (inflamed vessels → calcium). "
            "JAK inhibitors (baricitinib: JAK1/2; ruxolitinib: JAK1/2) → block IFN signalling downstream."
        ),
        "pathognomonic": (
            "PSEUDO-TORCH SYNDROME = PATHOGNOMONIC FOR AGS: "
            "Neonatal/early-infantile: TORCH-like features (microcephaly, calcifications, leukodystrophy) "
            "with NEGATIVE TORCH serology (CMV, toxoplasma, rubella, herpes) = PURSUE AGS DIAGNOSIS; "
            "CT BRAIN (NOT MRI ALONE): bilateral symmetric basal ganglia calcifications (putamen + caudate); "
            "white matter + cerebellar calcifications — CT > MRI for calcification detection; "
            "IFN-ALPHA IN CSF: >2 IU/mL (normal <2) = DIAGNOSTIC FOR AGS; "
            "INTERFERON SCORE: ISG15/IFIT1/CXCL10 gene expression in blood >2SD above control = AGS; "
            "CSF LYMPHOCYTOSIS (sterile): pleocytosis mimicking viral meningitis — misdiagnosis common; "
            "CHILBLAINS (acral cyanosis): cold-triggered skin lesions in ~40% of AGS — nifedipine helpful; "
            "ADAR sequencing: heterozygous GOF (most common) or biallelic LOF confirms AGS6"
        ),
        "treatment": (
            "JAK INHIBITORS — EMERGING (NOT YET APPROVED FOR AGS): "
            "BARICITINIB (JAK1/JAK2): most evidence; case series + small trials — stabilises/improves outcomes early; "
            "off-label (approved RA/alopecia); monitoring: neutropenia, LFTs, lipids, VZV reactivation; "
            "RUXOLITINIB: alternative JAK1/2 inhibitor; similar mechanism; "
            "REVERSE TRANSCRIPTASE INHIBITORS (RTIs): antiretrovirals (tenofovir/lamivudine/abacavir); "
            "rationale — LINE-1 retroelement activation as additional IFN trigger; "
            "SUPPORTIVE: anti-epileptic (seizures ~50%); antispasticity (baclofen); physiotherapy; "
            "CHILBLAINS: nifedipine (calcium channel blocker) — peripheral vasodilation; "
            "IFN-alpha CSF + ISG monitoring: response biomarkers for JAK inhibitor; "
            "GENETIC COUNSELLING: AD (50% from affected parent) vs AR (25%); de novo surveillance; "
            "CT brain every 1-2yr: monitor calcification progression"
        ),
        "key_features": [
            "ADAR (AGS6): AD GOF or AR; most common AGS gene (~25-30% of all AGS); Type I interferonopathy",
            "Pseudo-TORCH PATHOGNOMONIC: calcifications + leukodystrophy + microcephaly + NEGATIVE TORCH serology",
            "CT brain MANDATORY: basal ganglia + white matter calcifications (CT superior to MRI for calcium)",
            "IFN-alpha CSF (>2 IU/mL) + Interferon Score (ISG15 blood): diagnostic for AGS",
            "ADAR1 edits Alu-dsRNA as self: loss/gain of editing → MDA5 activation → IFN-alpha cascade",
            "JAK inhibitors (baricitinib/ruxolitinib): blocking JAK-STAT IFN signalling — most promising treatment",
            "Chilblains (acral cyanosis) in 40% — cold-triggered; nifedipine vasodilation",
            "CSF lymphocytosis sterile (mimics viral meningitis) — IFN-alpha CSF assay discriminates",
        ],
        "key_ddx": [
            "Congenital TORCH infection: calcifications + leukodystrophy BUT POSITIVE serology (CMV/toxo/rubella)",
            "Other AGS genes (TREX1/RNASEH2B/SAMHD1/IFIH1): same interferonopathy; ADAR most common; panel required",
            "Aicardi syndrome: females; absent corpus callosum + intracranial cysts; NOT interferonopathy; no IFN elevation",
            "GFAP Alexander disease: frontal + Rosenthal fibres; normal IFN score; macrocephaly",
        ],
        "onset_age": 0.3,
        "wm_lesion_pct": 95,
        "calcification_pct": 90,
        "ifn_elevated_pct": 95,
        "seizure_pct": 50,
        "chilblains_pct": 40,
        "spastic_pct": 72,
        "seed": 2621,
    },
]

SEEDS = [g["seed"] for g in ATLAS_GENES]


def _simulate_cohort(gene: dict, seed: int) -> list:
    rng = random.Random(seed)
    pts = []
    n = 40
    for i in range(n):
        age_onset = gene.get("onset_age", 2.0) + rng.gauss(0, 1.5)
        age_onset = max(0.1, age_onset)
        wm_lesion = int(rng.random() < gene.get("wm_lesion_pct", 90) / 100)
        seizure = int(rng.random() < gene.get("seizure_pct", 25) / 100)
        spastic = int(rng.random() < gene.get("spastic_pct", 70) / 100)
        pts.append({
            "gene": gene["gene"],
            "patient_id": f"{gene['gene']}-{seed}-{i+1:03d}",
            "age_onset": round(age_onset, 1),
            "wm_lesion": wm_lesion,
            "seizure": seizure,
            "spastic": spastic,
            "adrenal_insufficiency": int(gene["gene"] == "ABCD1" and rng.random() < 0.70),
            "calcifications": int(gene["gene"] == "ADAR" and rng.random() < 0.90),
            "stress_trigger": int(gene["gene"] == "EIF2B5" and rng.random() < 0.88),
            "nystagmus_birth": int(gene["gene"] == "PLP1" and rng.random() < 0.94),
            "hypodontia": int(gene["gene"] == "POLR3A" and rng.random() < 0.85),
            "ifn_elevated": int(gene["gene"] == "ADAR" and rng.random() < 0.95),
            "globoid_cells": int(gene["gene"] == "GALC" and rng.random() < 0.88),
            "metachromatic_granules": int(gene["gene"] == "ARSA" and rng.random() < 0.80),
            "seed": seed,
        })
    return pts


def generate_overview() -> dict:
    summary_by_gene = []
    all_pts = []
    for gene in ATLAS_GENES:
        pts = _simulate_cohort(gene, gene["seed"])
        all_pts.extend(pts)
        n = len(pts)
        summary_by_gene.append({
            "gene": gene["gene"],
            "locus": gene["locus"],
            "n_patients": n,
            "avg_onset_age": round(sum(p["age_onset"] for p in pts) / n, 1),
            "wm_lesion_pct": round(sum(p["wm_lesion"] for p in pts) / n * 100, 1),
            "seizure_pct": round(sum(p["seizure"] for p in pts) / n * 100, 1),
            "spastic_pct": round(sum(p["spastic"] for p in pts) / n * 100, 1),
        })

    total = len(all_pts)
    return {
        "atlas": "Hereditary-Leukodystrophy-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": len(ATLAS_GENES),
        "total_patients": total,
        "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "gene_summaries": summary_by_gene,
        "aggregate_stats": {
            "overall_wm_lesion_pct": round(sum(p["wm_lesion"] for p in all_pts) / total * 100, 1),
            "overall_seizure_pct": round(sum(p["seizure"] for p in all_pts) / total * 100, 1),
            "overall_spastic_pct": round(sum(p["spastic"] for p in all_pts) / total * 100, 1),
        },
        "disease_classes": [
            f"{g['gene']} — {g['disease_category'].split(';')[0].strip()}"
            for g in ATLAS_GENES
        ],
        "key_clinical_distinctions": [
            "ABCD1 X-ALD: VLCFA accumulate; CALD posterior advancing MRI; HSCT/Skysona-FDA2022 curative if Loes≤9+gadolinium; AMN adults; Addison 70% males",
            "ARSA MLD: sulfatide accumulates; metachromatic granules nerve biopsy PATHOGNOMONIC; Libmeldy gene therapy EMA2020; adult MLD = schizophrenia mimicry",
            "GALC Krabbe: psychosine cytotoxic; globoid cells PATHOGNOMONIC; extreme irritability infantile; HSCT pre-symptomatic late-onset",
            "PLP1 PMD: XLR; nystagmus at birth PATHOGNOMONIC; hypomyelination static; duplication most common (60-70%); NO disease-modifying therapy",
            "GJC2 PMLD: AR connexin 47; milder hypomyelination than PLP1; nystagmus less prominent/absent; SPG44 adult AD monoallelic",
            "POLR3A 4H: AR Pol-III; 4H TRIAD PATHOGNOMONIC (Hypomyelination+Hypodontia+Hypogonadotropic Hypogonadism); cerebellar atrophy; HRT mandatory",
            "EIF2B5 VWM: AR ISR GEF failure; stress-triggered crises (fever/minor trauma) PATHOGNOMONIC; white matter vanishes; ISRIB preclinical",
            "ADAR AGS6: AD GOF / AR interferonopathy; pseudo-TORCH PATHOGNOMONIC; CT calcifications; IFN-alpha CSF elevated; JAK inhibitors baricitinib emerging",
        ],
    }


def generate_breakdown() -> dict:
    gene_breakdowns = []
    for gene in ATLAS_GENES:
        pts = _simulate_cohort(gene, gene["seed"])
        n = len(pts)
        entry = {
            "gene": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"],
            "disease_category": gene["disease_category"],
            "pathognomonic": gene["pathognomonic"],
            "treatment": gene["treatment"],
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
            "n_patients": n,
            "avg_onset_age": round(sum(p["age_onset"] for p in pts) / n, 1),
            "wm_lesion_pct": round(sum(p["wm_lesion"] for p in pts) / n * 100, 1),
            "seizure_pct": round(sum(p["seizure"] for p in pts) / n * 100, 1),
            "spastic_pct": round(sum(p["spastic"] for p in pts) / n * 100, 1),
        }
        gene_breakdowns.append(entry)
    return {"gene_breakdowns": gene_breakdowns}


def generate_definitions() -> dict:
    gene_entries = {}
    for gene in ATLAS_GENES:
        gene_entries[gene["gene"]] = {
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"].split(";")[0].strip(),
            "disease_name": gene["disease_category"].split(";")[0].strip(),
            "disease_pathway": gene["disease_pathway"],
            "pathognomonic": gene["pathognomonic"],
            "treatment_summary": gene["treatment"].split(";")[0].strip() + "...",
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
        }
    return {
        "gene_entries": gene_entries,
        "leukodystrophy_glossary": {
            "Leukodystrophy Classification": (
                "Leukodystrophies are inherited disorders of white matter (myelin), classified by mechanism: "
                "1) Hypomyelinating: myelin never forms adequately (PLP1-PMD, GJC2-PMLD, POLR3A-4H); "
                "2) Demyelinating: myelin forms then breaks down (ARSA-MLD, GALC-Krabbe); "
                "3) Vacuolating: white matter becomes cystic/fluid-filled (EIF2B5-VWM); "
                "4) Neuroinflammatory/interferonopathy: immune-driven (ADAR-AGS6); "
                "5) Metabolic VLCFA storage (ABCD1-X-ALD). "
                "MRI pattern + distribution + progression rate narrows differential before genetic testing. "
                "Key patterns: posterior advancing (X-ALD), tigroid (MLD), vanishing (VWM), "
                "calcifications + leukodystrophy (AGS), static hypomyelination (PMD/4H/PMLD)."
            ),
            "Integrated Stress Response (ISR) and VWM Therapy": (
                "The Integrated Stress Response (ISR) is activated by four kinases (HRI/PKR/PERK/GCN2) "
                "phosphorylating eIF2alpha-Ser51 → inhibiting eIF2B GEF → ↓global translation + ↑ATF4. "
                "EIF2B5 mutations reduce eIF2B catalytic (GEF) activity. Basal: sufficient. "
                "Under ISR (fever/trauma): phospho-eIF2alpha cannot be overcome → translation fails in "
                "oligodendrocytes (highest demand) → white matter vacuolation. "
                "ISRIB (Integrated Stress Response InhiBitor): allosteric eIF2B activator — stabilises "
                "decameric eIF2B complex → maintains GEF activity even with phospho-eIF2alpha → "
                "reverses VWM in mouse models completely. Phase I trials of ISRIB analogues ongoing."
            ),
            "Type I Interferonopathy (AGS)": (
                "Aicardi-Goutières Syndrome (AGS) is the prototype type I interferonopathy: "
                "constitutive overactivation of innate immune type I IFN pathway → progressive neurodegeneration. "
                "Seven AGS genes (TREX1, RNASEH2A/2B/2C, SAMHD1, ADAR, IFIH1) all converge on "
                "preventing cytoplasmic nucleic acid sensing. "
                "ADAR1 normally edits Alu-dsRNA (self-mark as inosine); loss → unedited dsRNA → "
                "MDA5 activation → IFN-alpha cascade. "
                "Diagnosis: IFN-alpha CSF >2 IU/mL + Interferon Score (ISG15/IFIT1/CXCL10 blood upregulation). "
                "Treatment: JAK inhibitors (baricitinib/ruxolitinib) block JAK1/2 downstream of IFN receptor."
            ),
            "HSCT and Gene Therapy Windows in Leukodystrophies": (
                "Key principle: HSCT/gene therapy arrests disease but does NOT reverse established neurological damage. "
                "Treat BEFORE symptoms or in earliest stage for maximum benefit. "
                "X-ALD CALD: HSCT if Loes score ≤9 + gadolinium enhancement (active); "
                "Skysona gene therapy FDA 2022 (autologous — avoids GvHD); "
                "Krabbe late-onset: HSCT pre-symptomatic (NBS-identified); "
                "MLD: Libmeldy gene therapy EMA 2020 (autologous HSC; pre-symptomatic late-infantile/early juvenile); "
                "VWM/POLR3A-4H/PLP1-PMD: HSCT NOT effective (intrinsic oligodendrocyte defect — not enzyme deficiency). "
                "NBS enables pre-symptomatic identification — critical for these narrow treatment windows."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first entry) ===")
    defs = generate_definitions()
    first_gene = list(defs["gene_entries"].keys())[0]
    print(json.dumps(defs["gene_entries"][first_gene], indent=2)[:1000])
