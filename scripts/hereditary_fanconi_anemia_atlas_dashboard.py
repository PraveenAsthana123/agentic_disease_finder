"""Hereditary Fanconi Anemia & DNA Crosslink Repair Atlas — 8-Gene ICL Repair Reference
FANCA-FANCC-FANCD2-FANCG-FANCD1(BRCA2)-FANCN(PALB2)-FANCJ(BRIP1)-FANCI
(Fanconi anemia complementation groups A/C/D2/G/D1/N/J/I + HBOC spectrum overlap)
320 patients (8 x 40), seeds 2726-2733.
Endpoints: /api/hereditary-fanconi-anemia-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "FANCA",
        "seed_base": 2726,
        "protein": (
            "FANCA -- 16q24.3 AR -- 1455aa -- Fanconi-Anemia-Complementation-Group-A-"
            "163kDa-Monomer-FA-Core-Complex-Nuclear-Localisation-Signal-"
            "OMIM-Gene-607139-Disease-Fanconi-Anemia-A-227650"
        ),
        "locus": "16q24.3",
        "protein_size": "1455 aa / 163 kDa (FA core complex scaffold; nuclear import regulator)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → Fanconi anemia complementation group A); "
            "FANCA = most common FA gene — accounts for 60-65% of all FA cases worldwide; "
            "FA-A phenotype: classic triad of bone marrow failure (BMF) + congenital anomalies + cancer predisposition; "
            "FANCA forms the FA core complex (FA-CORE: FANCA+FANCB+FANCC+FANCE+FANCF+FANCG+FANCL+FANCM); "
            "FA-CORE complex is an E3 ubiquitin ligase that monoubiquitinates FANCD2 (Lys561) + FANCI (Lys523); "
            "FANCA nuclear localisation requires FANCG/FANCF interaction; "
            "LOF → failure of FANCD2-FANCI monoubiquitination → DNA interstrand crosslink (ICL) repair defective; "
            "FOUNDER ALLELES: "
            "  South African Black: del(exon 11-17) — 25% of SA Black FA patients; "
            "  Spanish Gypsy: c.3788-3790del3 (p.Val1264del) — most common Romani allele; "
            "  Ashkenazi Jewish: p.Arg1055* (c.3163C>T) — 8% of Ashkenazi FA; "
            "  Norwegian: large genomic deletion; "
            "PREVALENCE: FA overall 1 in 130,000-160,000; FANCA FA ~1 in 200,000-250,000; "
            "CARRIER FREQUENCY (FANCA): ~1 in 300-500 general population"
        ),
        "disease_category": (
            "FANCONI ANEMIA GROUP A (OMIM 227650); "
            "BONE MARROW FAILURE (BMF) — cardinal manifestation: "
            "  Progressive pancytopenia (thrombocytopenia first, then anaemia, then neutropenia); "
            "  Median age of onset: 7 years (range 1-31 years); "
            "  Aplastic anaemia requiring HSCT: 75-80% by age 40; "
            "  Red cell macrocytosis precedes cytopenias (elevated MCV + HbF); "
            "CONGENITAL ANOMALIES (50-60% of FA-A): "
            "  CAFÉ-AU-LAIT MACULES: most common skin finding (25-40%); "
            "  RADIAL RAY DEFECTS: absent/hypoplastic thumbs, radius aplasia (10-35%) — PATHOGNOMONIC of FA; "
            "  SHORT STATURE: ~50% (growth hormone deficiency contribution); "
            "  RENAL ANOMALIES: horseshoe kidney, renal agenesis, ectopic kidney (20-30%); "
            "  MICROCEPHALY (25%); MICROPHTHALMIA (20%); "
            "  VERTEBRAL ANOMALIES: scoliosis, vertebral fusion; "
            "  CARDIAC DEFECTS: VSD, ASD (15%); "
            "  EAR ANOMALIES: sensorineural hearing loss (25%); "
            "  HYPOGONADISM: delayed puberty, gonadal insufficiency (males > females); "
            "CANCER PREDISPOSITION: "
            "  Acute myeloid leukaemia (AML): 800x general population risk; median onset 14 years; "
            "  MDS → AML progression common (monosomy 7, trisomy 3q); "
            "  Squamous cell carcinoma (SCC): head/neck, oesophagus, anogenital; "
            "    ABSOLUTE RISK: 30% SCC by age 40; 70% by age 48 (post-HSCT even higher); "
            "    HPV-associated SCC especially post-HSCT; radiation sensitivity → avoid RT; "
            "  Brain tumors (rare); "
            "ENDOCRINOPATHIES: "
            "  Insulin resistance + diabetes mellitus (40-60%); "
            "  Growth hormone deficiency; hypothyroidism; "
            "DIAGNOSTIC TEST: chromosomal breakage with diepoxybutane (DEB) or mitomycin C (MMC) — GOLD STANDARD; "
            "  Increased chromosomal breaks + quadriradial figures = PATHOGNOMONIC; "
            "  Flow cytometry: G2/M cell-cycle arrest with MMC treatment"
        ),
        "disease_pathway": (
            "FA/BRCA DNA INTERSTRAND CROSSLINK (ICL) REPAIR PATHWAY: "
            "STEP 1 — RECOGNITION (S-phase): "
            "  ICL stalls two opposing replication forks converging on the lesion; "
            "  FANCM + FAAP24 + MHF1/MHF2 sense stalled fork → recruit FA-CORE complex; "
            "STEP 2 — CORE COMPLEX ACTIVATION: "
            "  FA-CORE (FANCA+FANCB+FANCC+FANCE+FANCF+FANCG+FANCL+FANCM) assembled at stalled fork; "
            "  FANCL (RING-type E3 ubiquitin ligase) activated; "
            "STEP 3 — ID COMPLEX MONOUBIQUITINATION: "
            "  FANCL + UBE2T (E2 enzyme) monoubiquitinates FANCD2 (K561) + FANCI (K523); "
            "  FANCD2-Ub + FANCI-Ub = ID complex (ID2) → loaded onto chromatin at ICL; "
            "STEP 4 — NUCLEOLYTIC UNHOOKING: "
            "  SLX4 (FANCP) scaffold recruits XPF-ERCC1 (FANCQ) + SLX1 + MUS81 nucleases; "
            "  XPF-ERCC1 makes dual incisions flanking ICL → unhooks crosslink; "
            "  Unhooked ICL remnant on one strand remains; "
            "STEP 5 — TRANSLESION SYNTHESIS (TLS): "
            "  Pol η (POLH/XPV) + Pol ζ + REV1 bypass unhooked ICL remnant; "
            "STEP 6 — TEMPLATE SWITCHING / HR: "
            "  FANCD1(BRCA2) + FANCN(PALB2) + FANCR(RAD51C) + FANCU(XRCC2) recruit RAD51; "
            "  Homologous recombination (HR) restores intact duplex; "
            "STEP 7 — NER OF REMNANT: "
            "  XPC-ERCC1-XPF removes unhooked ICL remnant from template strand; "
            "FANCA LOF → FA-CORE cannot assemble properly → FANCD2/FANCI not monoubiquitinated → "
            "  ICL repair blocked → replication fork collapse → DSBs → genomic instability → BMF + cancer"
        ),
        "pathognomonic": (
            "CHROMOSOMAL BREAKAGE TEST (DEB/MMC): "
            "  Diepoxybutane (DEB) 0.1 μg/mL added to lymphocyte culture; "
            "  FA cells: ≥4 breaks/cell (vs <0.5 in controls) — PATHOGNOMONIC; "
            "  QUADRIRADIAL FIGURES: diagnostic of FA (interstrand crosslinks joining 4 chromatids); "
            "  SOMATIC MOSAICISM PITFALL: 25-30% FA patients have mosaic reversion → NORMAL peripheral blood DEB; "
            "    → MUST test from fibroblasts/skin if blood breakage normal in suspected FA; "
            "RADIAL RAY DEFECTS: absent thumbs ± absent/hypoplastic radius: "
            "  BILATERAL absent thumbs in newborn → FA workup MANDATORY (not just VACTERL); "
            "  Radial-only defect pattern distinguishes from VACTERL/Holt-Oram; "
            "ELEVATED FETAL HAEMOGLOBIN (HbF): consistently elevated above age-appropriate norms; "
            "  Precedes cytopenias by years; macrocytosis MCV >100fL in pre-pancytopenic FA; "
            "COMPLEMENTATION TESTING: "
            "  If DEB positive → sequence FA gene panel (NGS); "
            "  MLPA/gene dosage: large deletions account for >30% FANCA pathogenic variants; "
            "MONOUBIQUITINATED FANCD2 (FANCD2-Ub) WESTERN BLOT: "
            "  FA-CORE group cells: FANCD2-Ub absent after MMC treatment; "
            "  Distinguishes FA-CORE (FANCA/B/C/E/F/G/L/M/T) from ID2 group (FANCD2/FANCI) and downstream"
        ),
        "treatment": (
            "HAEMATOPOIETIC STEM CELL TRANSPLANT (HSCT): "
            "  ONLY curative treatment for BMF/AML; "
            "  TIMING: perform BEFORE transfusion-dependence (pre-transfusion outcome superior); "
            "  CONDITIONING: reduced-intensity/fludarabine-based (NOT cyclophosphamide alone — FA sensitivity); "
            "    Standard cyclophosphamide = ABSOLUTELY CONTRAINDICATED (causes catastrophic toxicity); "
            "    Fludarabine + low-dose cyclophosphamide + anti-thymocyte globulin (ATG) — standard FA protocol; "
            "  RADIATION: minimize or avoid (FA cells hypersensitive to IR); "
            "    If radiation required: reduce dose 30-50%; "
            "  MATCHED SIBLING DONOR: best outcome; "
            "  MUD (matched unrelated): acceptable if fludarabine-based conditioning; "
            "  HAPLOIDENTICAL: increasing evidence in FA; "
            "ANDROGENS (bridge therapy): "
            "  Oxymetholone 2-5 mg/kg/day: response 50-70% (temporary); "
            "  Use to delay HSCT, not substitute; monitor for hepatic adenoma (annual US); "
            "  Danazol: better hepatic profile but lower response; "
            "SCC SURVEILLANCE POST-HSCT: "
            "  Annual head/neck/oesophageal exam + dental exam from age 16; "
            "  Annual gynaecological examination (anogenital SCC); "
            "  HPV vaccination recommended (FA carries 50x HPV-driven SCC risk); "
            "  Avoid tobacco + alcohol (synergistic with FA SCC risk); "
            "G-CSF + EPO: supportive only; short-term cytopenia management; "
            "GENE THERAPY: FANCA lentiviral vector trials (Spain/USA — TP-07-001 phase I/II, promising); "
            "AVOID: alkylating agents, cross-linking drugs (cisplatin/mitomycin — PATHOGNOMONIC toxicity); "
            "  ABSOLUTE: avoid unnecessary radiation exposure; dental X-ray acceptable with shielding"
        ),
    },
    {
        "gene": "FANCC",
        "seed_base": 2727,
        "protein": (
            "FANCC -- 9q22.32 AR -- 558aa -- Fanconi-Anemia-Complementation-Group-C-"
            "63kDa-Cytoplasmic-Scaffold-FA-Core-Complex-Member-"
            "OMIM-Gene-613899-Disease-Fanconi-Anemia-C-227645"
        ),
        "locus": "9q22.32",
        "protein_size": "558 aa / 63 kDa (FA core complex cytoplasmic scaffold; STAT1 inhibitor)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → Fanconi anemia complementation group C); "
            "FANCC accounts for ~15% of all FA cases; second most common FA gene after FANCA; "
            "FANCC is predominantly cytoplasmic (unique among FA-CORE members); "
            "Functions: nuclear role in FA-CORE assembly + cytoplasmic role inhibiting STAT1 hyperactivation; "
            "FANCC cytoplasmic: binds HSP70 + GRP94 → prevents apoptosis under oxidative stress; "
            "FOUNDER ALLELES (critical clinical knowledge): "
            "  ASHKENAZI JEWISH: IVS4+4A>T (c.456+4A>T) — MOST COMMON Ashkenazi FA mutation; "
            "    Carrier frequency: 1 in 89 Ashkenazi Jews; "
            "    Causes exon 4 skipping → truncated protein → classic FA; "
            "    PHENOTYPE: classic severe FA (Ashkenazi — early BMF + AML); "
            "  JAPANESE: c.322delG (p.Val108Trpfs*13) — most common Japanese FANCC allele; "
            "    Severe phenotype; early haematopoietic failure; "
            "  SOUTH ASIAN: IVS4+4A>T also found in Indian/Pakistani patients; "
            "GENOTYPE-PHENOTYPE: "
            "  IVS4+4A>T: null allele → severe phenotype (early BMF, high AML risk); "
            "  Missense alleles (rare): milder phenotypes described; "
            "PREVALENCE: FANCC FA ~1 in 900,000; carrier ~1 in 1000 general; 1 in 89 Ashkenazi"
        ),
        "disease_category": (
            "FANCONI ANEMIA GROUP C (OMIM 227645); "
            "PHENOTYPE: overlaps with FA-A but with Ashkenazi-specific clinical context; "
            "BONE MARROW FAILURE: "
            "  Earlier onset than FA-A in IVS4+4A>T homozygotes (median ~6-7 years); "
            "  More severe haematological phenotype in null/null genotypes; "
            "  AML risk: high; monosomy 7 common cytogenetic event; "
            "CONGENITAL ANOMALIES: "
            "  CAFÉ-AU-LAIT MACULES: most common (30-40%); "
            "  THUMB ANOMALIES: 15-20%; "
            "  RENAL MALFORMATIONS: 15-25%; "
            "  Phenotype generally overlaps with FA-A; "
            "CANCER PREDISPOSITION: "
            "  AML: similar to FA-A (~700x general population); "
            "  SCC: head/neck + oesophageal + anogenital — similar post-HSCT risk; "
            "  Hepatocellular carcinoma: described in androgen-treated FA (hepatic adenoma → HCC); "
            "ASHKENAZI CARRIER TESTING: "
            "  Routine Ashkenazi carrier panel SHOULD include FANCC IVS4+4A>T; "
            "  Partner testing if Ashkenazi carrier identified; "
            "  Prenatal diagnosis: CVS or amniocentesis DEB testing + molecular confirmation; "
            "ENDOCRINOPATHIES: insulin resistance + GH deficiency (as per FA-A); "
            "CYTOPLASMIC FANCC ROLE: "
            "  STAT1 hyperactivation in FANCC-null cells → increased IFN signalling → haematopoietic suppression; "
            "  HSP70 protection against apoptosis under oxidative/genotoxic stress defective"
        ),
        "disease_pathway": (
            "FA-CORE COMPLEX ASSEMBLY — FANCC CYTOPLASMIC ROLE: "
            "CYTOPLASMIC FUNCTIONS OF FANCC: "
            "  1. HSP70/GRP94 INTERACTION: "
            "     FANCC binds HSP70 + GRP94 → protects haematopoietic cells from apoptosis; "
            "     Oxidative stress or ICL damage → FANCC bridges chaperones to pro-survival signals; "
            "  2. STAT1 INHIBITION: "
            "     FANCC inhibits phospho-STAT1 nuclear translocation; "
            "     FANCC LOF → STAT1 hyperactivation → IFN-gamma pathway overactive → "
            "       haematopoietic cell apoptosis (mechanism of cytopenias); "
            "  3. CDKN1A (p21) REGULATION: "
            "     FANCC modulates p21 expression → cell cycle checkpoint control; "
            "NUCLEAR FA-CORE ROLE: "
            "  FANCC nuclear fraction participates in FA-CORE assembly; "
            "  FANCC interacts with FANCE (nuclear escort of FANCC); "
            "  FANCC-FANCE-FANCF form a trimeric sub-complex within FA-CORE; "
            "  FANCC LOF → FA-CORE assembly defective → FANCD2-FANCI not monoubiquitinated → ICL repair blocked; "
            "FANCC PHOSPHORYLATION: "
            "  FANCC Ser249 phosphorylated by ATM/ATR → regulates core complex activity under DNA damage; "
            "IVS4+4A>T MECHANISM: "
            "  Intronic splice-site variant (IVS4+4, non-canonical AG rule) → exon 4 skipped → "
            "    reading frame shifted → premature stop codon → null FANCC protein; "
            "  Western blot: no FANCD2-Ub after MMC treatment (FA-CORE group)"
        ),
        "pathognomonic": (
            "DEB/MMC CHROMOSOMAL BREAKAGE: PATHOGNOMONIC for FA class (same as FANCA); "
            "  DEB test: ≥4 breaks/cell; quadriradials present; "
            "  Distinguishes FANCC from FANCA only by molecular testing; "
            "ASHKENAZI-SPECIFIC CARRIER FREQUENCY: "
            "  IVS4+4A>T carrier: 1 in 89 Ashkenazi Jews — one of the most common Ashkenazi recessive conditions; "
            "  Routine Ashkenazi carrier screening includes FANCC; "
            "  Partner testing: if one carrier identified, partner testing critical (1 in ~360 FA-C risk); "
            "NEONATAL PRESENTATION (IVS4+4A>T homozygous): "
            "  Severe phenotype: BMF by 7-8 years; "
            "  Thumb/radial anomaly less common than FA-A (20% vs 35%); "
            "  BUT: CAFÉ-AU-LAIT macules more often multiple and large; "
            "SOMATIC MOSAICISM: "
            "  Occurs in FANCC as in FANCA; "
            "  If DEB normal in blood → test fibroblasts; "
            "WESTERN BLOT FOR FANCD2-Ub: "
            "  FANCC cells: FANCD2-Ub absent (FA-CORE group); "
            "  Separates from downstream pathway groups (FANCD2, FANCI, BRCA2, PALB2 = FANCD2-Ub present but ID2 or HR defective)"
        ),
        "treatment": (
            "HSCT: same protocol as FANCA — fludarabine-based conditioning; NO cyclophosphamide alone; "
            "  Ashkenazi FANCC outcome with matched sibling donor: excellent (>80% OS at 5 years); "
            "ANDROGEN THERAPY (bridge): oxymetholone as per FA-A protocol; monitor hepatic adenoma; "
            "SCC SURVEILLANCE: same post-HSCT protocol as FA-A; "
            "ASHKENAZI COMMUNITY SCREENING: "
            "  Pre-conception carrier testing recommended; "
            "  Preimplantation genetic testing (PGT-M) available for known carrier couples; "
            "  Israeli national programme: FANCC screening included in routine Ashkenazi panel; "
            "GENE THERAPY: FANCA vector primary focus; FANCC gene therapy in development; "
            "AVOID: alkylating agents (DEB/MMC-like drugs), RT (reduce dose); "
            "OXIDATIVE STRESS MANAGEMENT: "
            "  Antioxidant supplements (NAC/vitamin E) — preclinical benefit; clinical evidence limited; "
            "  Avoid iron overload (transfusion-related) — chelation if ferritin >1000; "
            "GROWTH HORMONE: if GH deficiency confirmed; improves height outcome"
        ),
    },
    {
        "gene": "FANCD2",
        "seed_base": 2728,
        "protein": (
            "FANCD2 -- 3p25.3 AR -- 1451aa -- Fanconi-Anemia-Complementation-Group-D2-"
            "155kDa-Monoubiquitinated-K561-ID2-Complex-FANCD2-FANCI-"
            "OMIM-Gene-613984-Disease-Fanconi-Anemia-D2-227646"
        ),
        "locus": "3p25.3",
        "protein_size": "1451 aa / 155 kDa (ID2 complex; monoubiquitinated K561 by FA-CORE)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → Fanconi anemia complementation group D2); "
            "FANCD2 = KEY EFFECTOR of the FA pathway — central hub connecting upstream recognition/ubiquitination "
            "to downstream HR; "
            "FANCD2 monoubiquitination at K561 by FANCL (FA-CORE E3 ligase + UBE2T E2) is the CENTRAL EVENT "
            "in FA ICL repair; "
            "FANCD2-Ub + FANCI-Ub = ID2 complex → loaded onto chromatin at ICL lesion; "
            "FA-D2 accounts for ~3-6% of all FA cases; "
            "CRITICAL DIAGNOSTIC ROLE: "
            "  FANCD2-Ub status by Western blot is the GATEKEEPER test: "
            "    FA-CORE group (FANCA/B/C/E/F/G/L): FANCD2-Ub ABSENT after MMC → FA-CORE defective; "
            "    FA-D2 (FANCD2): FANCD2-Ub ABSENT (protein defective/unstable); "
            "    FA-I (FANCI): FANCD2-Ub present but FANCI-Ub absent; "
            "    Downstream (BRCA2/PALB2/BRIP1): FANCD2-Ub PRESENT (ubiquitination intact; HR defective); "
            "FANCD2 also has nuclease activity (FANCD2-associated nuclease 1, FAN1, is recruited by FANCD2-Ub); "
            "GENOTYPE-PHENOTYPE (FA-D2): "
            "  Severe phenotype; early BMF; high AML/MDS risk; "
            "  Congenital anomalies in ~70% (similar to FA-A); "
            "  Brain tumor predisposition (higher than FA-A); "
            "IMPORTANT: FANCD2 gene is DISTINCT from FANCD1 (BRCA2) — different complementation groups"
        ),
        "disease_category": (
            "FANCONI ANEMIA GROUP D2 (OMIM 227646); "
            "NOTE: FANCD2 ≠ FANCD1 — complementation group D2 = FANCD2 gene; group D1 = BRCA2 gene; "
            "BONE MARROW FAILURE: classic FA-BMF progression; early pancytopenia; "
            "CONGENITAL ANOMALIES (~70%): "
            "  THUMB/RADIAL DEFECTS: 30-35%; "
            "  CAFÉ-AU-LAIT MACULES: 30-40%; "
            "  RENAL ANOMALIES: 20%; "
            "  MICROCEPHALY, MICROPHTHALMIA: 20-25%; "
            "  SKIN PIGMENTATION ABNORMALITIES: café-au-lait + hyperpigmentation/hypopigmentation; "
            "CANCER PREDISPOSITION: "
            "  AML/MDS: ~800x general population; monosomy 7 predominant karyotype; "
            "  BRAIN TUMORS: described at higher frequency than FA-A; "
            "  SCC: head/neck + oesophageal; "
            "ATYPICAL FEATURE OF FA-D2: "
            "  Higher rate of brain malformations (partial agenesis of corpus callosum); "
            "  VATER/VACTERL overlap more pronounced; "
            "FANCD2 INTERACTION WITH FAN1: "
            "  FAN1 (FANCD2-associated nuclease 1) recruited by FANCD2-Ub via ubiquitin-binding domain; "
            "  FAN1 provides additional nucleolytic activity at ICL; "
            "  FAN1 biallelic LOF → karyomegalic interstitial nephritis (NOT FA) — important DDx"
        ),
        "disease_pathway": (
            "FANCD2 AS CENTRAL ICL REPAIR HUB: "
            "FA-CORE → FANCD2-K561-Ub → FANCI-K523-Ub: "
            "  UBE2T (E2) + FANCL (E3) → monoubiquitinates FANCD2 K561 in S-phase; "
            "  FANCI monoubiquitinated simultaneously (K523) — FANCD2/FANCI are mutually stabilizing; "
            "  ID2 complex (FANCD2-Ub + FANCI-Ub) loaded onto stalled fork chromatin; "
            "ID2 DOWNSTREAM FUNCTIONS: "
            "1. FAN1 RECRUITMENT: FANCD2-Ub binds FAN1 UBZ domain → FAN1 nuclease activity → "
            "   incises DNA flanking ICL; "
            "2. SLX4 SCAFFOLD INTERACTION: ID2 stabilises SLX4 (FANCP) at ICL → XPF-ERCC1 + SLX1 unhooking; "
            "3. HR INITIATION: FANCD1(BRCA2) + FANCN(PALB2) recruited → RAD51 loading → strand invasion; "
            "DEUBIQUITINATION: "
            "  USP1-UAF1 complex removes Ub from FANCD2 + FANCI after repair completion; "
            "  USP1 regulated by PCNA + UAF1; reactivation of USP1 terminates FA activation; "
            "FANCD2 LOF CONSEQUENCES: "
            "  Even if FA-CORE intact → without FANCD2 protein → ID2 cannot form → ICL repair blocked; "
            "  All downstream events (FAN1, SLX4, HR via BRCA2/PALB2) dependent on FANCD2-Ub"
        ),
        "pathognomonic": (
            "DEB/MMC CHROMOSOMAL BREAKAGE: POSITIVE (as all FA groups); "
            "WESTERN BLOT KEY ROLE: "
            "  FANCD2-Ub ABSENT after MMC in FA-D2 (FANCD2 protein unstable/absent): "
            "    Distinguishes from downstream FA groups (BRCA2/PALB2 = FANCD2-Ub PRESENT); "
            "    Further FANCD2 vs FA-CORE distinction: "
            "      FA-CORE cells show normal FANCD2 protein but no -Ub band; "
            "      FA-D2 cells may show reduced or absent FANCD2 protein; "
            "  Both short (unubiquitinated) + long (ubiquitinated) FANCD2 bands normally seen; "
            "  Only short band in FA-CORE; reduced/absent both bands in FA-D2; "
            "FANCD2-FANCI IMMUNOFLUORESCENCE (FOCI): "
            "  Normal cells after MMC: FANCD2-Ub nuclear foci (co-localise with FANCI, γH2AX, PCNA); "
            "  FA-D2 cells: NO FANCD2 foci → FA diagnosis + D2 group identification; "
            "FAN1 NEPHRITIS DISTINCTION: "
            "  FAN1 biallelic → karyomegalic interstitial nephritis (normal DEB test; no BMF); "
            "  FANCD2 biallelic → classic FA (DEB positive; BMF); "
            "  CRITICAL DDx: FAN1-null = renal phenotype; FANCD2-null = haematopoietic phenotype"
        ),
        "treatment": (
            "HSCT: fludarabine-based conditioning (identical to FANCA protocol); "
            "  FA-D2 post-HSCT SCC risk as per all FA (annual surveillance mandatory); "
            "ANDROGEN BRIDGE: oxymetholone/danazol while awaiting HSCT; "
            "AVOID ALKYLATORS/RT: hypersensitivity identical to all FA groups; "
            "FANCONI ANEMIA INTERNATIONAL CONSORTIUM (FAIC): "
            "  FA-D2 patients enrolled for natural history data; "
            "  Genotype-phenotype registry; "
            "GENE THERAPY: FANCD2 lentiviral vector in preclinical development; "
            "MONITORING: "
            "  Annual bone marrow aspirate + biopsy from age 3-5 (or at diagnosis); "
            "  CBC monthly or quarterly depending on stage; "
            "  Annual AML/MDS monitoring: cytogenetics (monosomy 7 = BMT indication); "
            "BRAIN ANOMALY SURVEILLANCE: MRI if neurological symptoms; "
            "ENDOCRINE: GH + insulin resistance monitoring from age 5"
        ),
    },
    {
        "gene": "FANCG",
        "seed_base": 2729,
        "protein": (
            "FANCG -- 9p13.3 AR -- 622aa -- Fanconi-Anemia-Complementation-Group-G-"
            "68kDa-Monomer-Tetratricopeptide-Repeat-FA-Core-Complex-"
            "OMIM-Gene-602956-Disease-Fanconi-Anemia-G-614082"
        ),
        "locus": "9p13.3",
        "protein_size": "622 aa / 68 kDa (FA core complex; tetratricopeptide repeat scaffold)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → Fanconi anemia complementation group G); "
            "FANCG = third most common FA gene — accounts for ~10% of all FA cases; "
            "FANCG encodes XRCC9 (also known as XRCC9) — first identified as X-ray cross-complementing group 9; "
            "FANCG has TETRATRICOPEPTIDE REPEAT (TPR) motifs → protein-protein interaction scaffold; "
            "In FA-CORE complex: FANCG directly interacts with FANCA + FANCF → stabilises FANCA nuclear import; "
            "FANCG required for FANCA nuclear localisation (FANCG-null → FANCA remains cytoplasmic); "
            "FOUNDER ALLELES: "
            "  BLACK SOUTH AFRICAN: IVS4+4A>T (identical splice donor as FANCC Ashkenazi allele — coincidence); "
            "    FANCG IVS4+4A>T → most common SA Black FA allele (different gene than FANCC!); "
            "  JAPANESE: p.Leu71Pro + c.1794del4; "
            "  PORTUGUESE: p.Glu105* (c.313G>T); "
            "PHENOTYPE SEVERITY: "
            "  FA-G phenotype: intermediate — less severe congenital anomalies than FA-A; "
            "  HIGHER LEUKAEMIA RISK than FA-A proportionally; "
            "  SEVERE APLASTIC ANAEMIA: more frequent early complete aplasia; "
            "  PANCREATIC ANOMALIES: annular pancreas described (unique to some FA-G); "
            "PREVALENCE: FANCG FA ~1 in 1,300,000; carrier ~1 in 1200 general population"
        ),
        "disease_category": (
            "FANCONI ANEMIA GROUP G (OMIM 614082); "
            "CLINICAL OVERLAP with FA-A/C but DISTINCTIVE FEATURES: "
            "BONE MARROW FAILURE: "
            "  SEVERE APLASIA: proportion with complete aplasia > FA-A; "
            "  EARLY ONSET: median BMF onset ~6 years (similar to FA-C); "
            "  VERY HIGH AML RISK: ~10-fold higher AML frequency vs FA-A in some series; "
            "  FREQUENT MONOSOMY 7: dominant karyotypic change pre-AML; "
            "CONGENITAL ANOMALIES (40-50%): "
            "  CAFÉ-AU-LAIT MACULES: 25-35%; "
            "  THUMB ANOMALIES: 10-15% (lower than FA-A); "
            "  SHORT STATURE: 50%; "
            "  PANCREATIC AGENESIS/HYPOPLASIA: unique to FA-G (annular pancreas); "
            "CANCER PREDISPOSITION: "
            "  AML: proportionally higher rate than FA-A in registry data; "
            "  SCC: similar post-HSCT risk; "
            "  Solid tumours: liver (possible, HCC in androgen recipients); "
            "MALE INFERTILITY: "
            "  AZOOSPERMIA more common in FA-G than FA-A; "
            "  FA-G males: spermatogenic failure (Fancg knockout mice: complete spermatogenic failure); "
            "  Testicular atrophy + azoospermia — DISTINCTIVE for FA-G; "
            "ENDOCRINOPATHIES: "
            "  DIABETES MELLITUS: higher rate than FA-A (possibly related to pancreatic anomalies); "
            "  GH deficiency; hypothyroidism; "
            "XRCC9 FUNCTION: FANCG ensures FANCA nuclear presence → without FANCG, "
            "  FANCA cannot enter nucleus → entire FA-CORE non-functional"
        ),
        "disease_pathway": (
            "FANCG IN FA-CORE COMPLEX ASSEMBLY: "
            "FANCG-FANCA DIRECT INTERACTION: "
            "  FANCG N-terminal region binds FANCA → nuclear import complex; "
            "  FANCG TPR motifs scaffold FANCF + FANCA within FA-CORE; "
            "  Without FANCG: FANCA cannot enter nucleus → FA-CORE cannot assemble on chromatin; "
            "  FANCG acts as nuclear import chaperone for FANCA; "
            "FANCG IN TPR-MEDIATED PROTEIN ASSEMBLY: "
            "  FANCG TPR repeat units (7 predicted) → protein-protein docking; "
            "  FANCG bridges FANCA-FANCF module to FANCE-FANCC module; "
            "  TPR scaffold ensures stoichiometric FA-CORE assembly at stalled forks; "
            "FANCG PHOSPHORYLATION: "
            "  FANCG phosphorylated by CDK1/cyclin B in mitosis; "
            "  Phosphorylation disrupts FANCG-FANCA interaction → releases FA-CORE during mitosis; "
            "  Ensures FA pathway re-activation in next S-phase; "
            "FANCG LOF CONSEQUENCES: "
            "  FANCA mislocalised to cytoplasm → FA-CORE cannot assemble on chromatin; "
            "  FANCD2-Ub absent after MMC (FA-CORE group); "
            "  ICL repair pathway blocked from Step 2 onwards; "
            "SPERMATOGENIC FAILURE MECHANISM: "
            "  FA pathway critical for meiotic recombination (Holliday junction resolution); "
            "  FANCG/BRCA2/PALB2 required for meiotic HR; "
            "  FANCG-null: complete spermatogenic block at spermatocyte stage; "
            "  Clinical: azoospermia from non-obstructive cause"
        ),
        "pathognomonic": (
            "DEB/MMC CHROMOSOMAL BREAKAGE: POSITIVE; FANCD2-Ub ABSENT (FA-CORE group); "
            "DISTINCTIVE FA-G FEATURES: "
            "  AZOOSPERMIA in males: higher rate than FA-A/C (testicular biopsy: spermatogenic arrest); "
            "  ANNULAR PANCREAS: radiological finding on MRCP/EUS (if symptomatic); "
            "  EARLY COMPLETE APLASIA: bone marrow cellularity <5% at presentation more common; "
            "SOUTH AFRICAN BLACK FOUNDER: "
            "  IVS4+4A>T in FANCG: if DEB positive + Black South African ancestry → "
            "    test FANCG IVS4+4A>T first (not FANCA — different gene, same splice variant name); "
            "XRCC9 WESTERN BLOT: "
            "  Absent XRCC9 (FANCG) protein in FA-G cells; "
            "  FANCA protein present but cytoplasmic (nuclear fraction reduced/absent); "
            "FANCG VS FA-A DISTINCTION: "
            "  Clinically: less radial ray defects + more complete aplasia + more azoospermia → suspect FA-G; "
            "  FANCG sequencing + MLPA after positive DEB confirms group; "
            "  DEB test cannot distinguish FA group (all DEB positive)"
        ),
        "treatment": (
            "HSCT: fludarabine-based conditioning; "
            "  FA-G URGENT HSCT INDICATION: severe aplasia (cellularity <5%) = IMMEDIATE transplant; "
            "  Higher complete aplasia rate → sooner HSCT decision; "
            "ANDROGEN BRIDGE: oxymetholone; but shorter bridge duration in complete aplasia; "
            "AZOOSPERMIA: sperm banking IMPOSSIBLE (no sperm); "
            "  Testicular sperm extraction (TESE): occasionally yields rare sperm; "
            "  Fertility counselling: adoption/donor sperm discussed early; "
            "DIABETES MONITORING: annual OGTT from age 8; "
            "  Pancreatic insufficiency: enzyme replacement if exocrine failure; "
            "  Insulin if endocrine failure (type 3c diabetes); "
            "ANDROGEN MONITORING: annual US for hepatic adenoma; "
            "CANCER SURVEILLANCE: same as FA-A; "
            "AVOID: alkylating agents + excessive radiation"
        ),
    },
    {
        "gene": "BRCA2",
        "seed_base": 2730,
        "protein": (
            "BRCA2 -- 13q12.3 AR-AD -- 3418aa -- FANCD1-Breast-Cancer-2-"
            "384kDa-Monomer-RAD51-Loader-8-BRC-Repeats-OB-Folds-Tower-DBD-"
            "OMIM-Gene-600185-Disease-FA-D1-605724-HBOC-612555"
        ),
        "locus": "13q12.3",
        "protein_size": "3418 aa / 384 kDa (RAD51 loader; BRC repeats; OB-folds DNA-binding domain)",
        "inheritance": (
            "BIALLELIC AR → FANCONI ANEMIA COMPLEMENTATION GROUP D1 (FA-D1, OMIM 605724): "
            "  MOST SEVERE FA PHENOTYPE; "
            "  EARLY CHILDHOOD MALIGNANCIES: Wilms tumor (median 1.8 years), medulloblastoma, AML; "
            "  Biallelic usually compound heterozygous (homozygous lethal pre-birth); "
            "  ALL biallelic BRCA2 FA-D1 patients develop cancer by age 10 without intervention; "
            "MONOALLELIC AD → HEREDITARY BREAST AND OVARIAN CANCER (HBOC, OMIM 612555): "
            "  Autosomal dominant LOF — haploinsufficiency; "
            "  Lifetime breast cancer risk: 45-87% (pathogenic variant-dependent); "
            "  Lifetime ovarian cancer risk: 11-27%; "
            "  Male breast cancer: 7-10% lifetime (HIGHEST of all HBOC genes); "
            "  Pancreatic cancer: 3-7% lifetime; "
            "  Prostate cancer: 20-30% (aggressive, early onset); "
            "BIALLELIC PARADOX: "
            "  Biallelic → FA-D1 (NOT HBOC) → completely different phenotype; "
            "  De novo biallelic usually: one null allele + one hypomorphic allele (total null biallelic = lethal); "
            "  Common FA-D1 biallelic: c.9976A>T (p.Lys3326*) + truncating variant; "
            "BRCA2 STRUCTURE: "
            "  8 BRC repeats (aa 1002-2085): each binds one RAD51 monomer; "
            "  OB-fold domain (C-terminal): ssDNA binding; "
            "  Tower domain: unique fold within OB3; "
            "  Nuclear localisation signal: C-terminal; "
            "  PALB2-binding domain (N-terminal aa 10-40)"
        ),
        "disease_category": (
            "BIALLELIC: FA-D1 (FANCONI ANEMIA D1, OMIM 605724): "
            "  MOST SEVERE FA COMPLEMENTATION GROUP; "
            "  CANCER ONSET: median 2.4 years (vs 14 years in FA-A); "
            "  WILMS TUMOR: ~30% (median onset 14 months — EARLIEST solid tumor in any hereditary syndrome); "
            "  MEDULLOBLASTOMA: ~20% (desmoplastic histology — PATHOGNOMONIC of FA-D1); "
            "  AML: ~25%; early (median 5 years); "
            "  BRAIN TUMORS overall: 30-40%; "
            "  REDUCED BMF: less classical BMF than FA-A; BMF in ~60% but cancer onset before BMF; "
            "  CONGENITAL ANOMALIES: present but may be subtle (~80%); "
            "    thumb anomalies 25-30%; renal 20%; café-au-lait 30%; "
            "MONOALLELIC: HBOC SPECTRUM: "
            "  BREAST CANCER: "
            "    Female: 45-87% lifetime (variant-specific); "
            "    Male: 7-10% lifetime; HIGHEST male breast cancer gene; "
            "    Triple-negative BC: overrepresented in BRCA2 (35-40%); "
            "    ER+/HER2- also common (unlike BRCA1 which is predominantly TNBC); "
            "  OVARIAN CANCER: 11-27% lifetime (epithelial, non-mucinous, high-grade serous); "
            "    RRSO recommended 40-45 years; "
            "  PANCREATIC CANCER: HIGHEST single-gene hereditary pancreatic cancer risk; "
            "    3-7% lifetime; olaparib POLO trial (FDA 2019 approval); "
            "  PROSTATE CANCER: 20-30% lifetime; aggressive (Gleason ≥7); early onset; "
            "    Early PSA screening from age 40; olaparib PROfound trial (FDA 2020); "
            "  PARPi SENSITIVITY: olaparib + rucaparib + niraparib + talazoparib all active; "
            "  PLATINUM SENSITIVITY: cisplatin/carboplatin preferred over taxane in BRCA2 carriers"
        ),
        "disease_pathway": (
            "BRCA2 ROLE IN HOMOLOGOUS RECOMBINATION (HR): "
            "1. RAD51 LOADING — CENTRAL FUNCTION: "
            "   8 BRC repeats (each ~35aa) bind individual RAD51 monomers → prevent premature filament; "
            "   BRCA2 OB-fold domain binds ssDNA at 3' resected DSB end; "
            "   BRCA2 delivers RAD51 monomers to RPA-coated ssDNA → nucleates RAD51 presynaptic filament; "
            "   RAD51 filament = key recombinase for strand invasion into homologous template; "
            "2. PALB2 RECRUITMENT: "
            "   BRCA2 N-terminal (aa 10-40) binds PALB2 coiled-coil domain; "
            "   PALB2 bridges BRCA2 to BRCA1 (via PALB2-BRCA1 interaction) → BRCA1-PALB2-BRCA2 axis; "
            "   PALB2 recruits BRCA2 to DSB sites (BRCA2 cannot localise to DSBs without PALB2); "
            "3. FA-D1 PATHWAY INTEGRATION: "
            "   BRCA2 recruited downstream of FANCD2-Ub to ICL repair; "
            "   BRCA2 required for HR-mediated gap-filling after ICL unhooking (Steps 5-6); "
            "4. PROTECTION OF STALLED FORKS: "
            "   BRCA2 (+ BRCA1) protects reversed forks from MRE11 nucleolytic degradation; "
            "   BRCA2 LOF → fork degradation → replication catastrophe under HU/APH treatment; "
            "5. MITOTIC ROLE: "
            "   BRCA2 required for centrosome number regulation → BRCA2-null = centrosome amplification; "
            "   FA-D1: mitotic abnormalities contribute to carcinogenesis"
        ),
        "pathognomonic": (
            "FA-D1 (BIALLELIC): "
            "  DEB/MMC BREAKAGE: POSITIVE; "
            "  FANCD2-Ub: PRESENT (downstream of FA-CORE; FA-CORE intact) — DISTINCTIVE; "
            "    FA-D1 is the ONLY FA group where FANCD2-Ub is normal (HR defective, not ubiquitination); "
            "  DESMOPLASTIC MEDULLOBLASTOMA in child ≤5 years: "
            "    PATHOGNOMONIC → FA-D1 (biallelic BRCA2) workup MANDATORY; "
            "    CXR + ECHO before chemo (avoid adriamycin if FA); "
            "  WILMS TUMOR in infant ≤2 years: "
            "    FA-D1 in DDx (alongside WT1, WTX, WT2); DEB test urgently; "
            "  FAMILY HISTORY: "
            "    Parents of FA-D1 child = obligate heterozygous BRCA2 carriers → HBOC counselling; "
            "HBOC (MONOALLELIC): "
            "  MALE BREAST CANCER: rare (<1% general) → BRCA2 carrier rate 10% in male BC; "
            "  ER-POSITIVE MALE BREAST CANCER: BRCA2 testing mandatory; "
            "  OLAPARIB RESPONSE: BRCA2 somatic + germline metastatic BC/OC/PC; "
            "  BRCA2 vs BRCA1 IHC DDx: both can show BRCA1/2 protein loss; molecular testing required"
        ),
        "treatment": (
            "FA-D1 (BIALLELIC): "
            "  EXTREME CAUTION with treatment: "
            "    WILMS TUMOR: actinomycin D + vincristine ONLY (NO adriamycin/cisplatin if confirmed FA); "
            "      Reduced dose RT (if needed); most avoid RT entirely; "
            "    MEDULLOBLASTOMA: craniospinal RT standard protocol ABSOLUTELY CONTRAINDICATED in confirmed FA; "
            "      Alternative: reduced-intensity chemo-only protocols (carboplatin + vincristine); "
            "      FA-D1 desmoplastic MB: surgical resection only if possible; "
            "    AML: HSCT with reduced-intensity conditioning; NO anthracyclines at standard dose; "
            "  HSCT: for BMF; pre-emptive HSCT discussed given certain cancer development; "
            "  DEB TEST TIMING: ALL children with Wilms tumor or desmoplastic MB → DEB test URGENTLY; "
            "HBOC (MONOALLELIC): "
            "  BREAST CANCER SCREENING: "
            "    Annual MRI from age 25 + annual mammogram from age 30; "
            "  RISK-REDUCING SURGERY: "
            "    Risk-reducing mastectomy (RRM): reduces breast cancer risk 90-95%; "
            "    RRSO: reduces ovarian cancer risk 85-90%; recommended age 40-45; "
            "    Reduces all-cause mortality in BRCA2 carriers (Metcalfe 2014); "
            "  PARP INHIBITORS: "
            "    Olaparib: FDA 2018 (metastatic BC); FDA 2019 (pancreatic, POLO trial); FDA 2020 (prostate, PROfound); "
            "    Rucaparib: FDA (prostate, TRITON2); "
            "    Niraparib/talazoparib: FDA (BC); "
            "  CHEMOTHERAPY: platinum preferred over taxane in metastatic BRCA2 BC/PC; "
            "  PROSTATE: PSA from age 40; active surveillance discouraged for Gleason ≥7; "
            "  MALE BREAST: mastectomy if BC diagnosed; tamoxifen consideration"
        ),
    },
    {
        "gene": "PALB2",
        "seed_base": 2731,
        "protein": (
            "PALB2 -- 16p12.2 AR-AD -- 1186aa -- FANCN-Partner-And-Localiser-of-BRCA2-"
            "130kDa-Monomer-Coiled-Coil-WD40-Domain-BRCA1-BRCA2-Bridge-"
            "OMIM-Gene-610355-Disease-FA-N-610832-HBOC-610355"
        ),
        "locus": "16p12.2",
        "protein_size": "1186 aa / 130 kDa (BRCA1-PALB2-BRCA2 bridge; WD40 domain; coiled-coil)",
        "inheritance": (
            "BIALLELIC AR → FANCONI ANEMIA COMPLEMENTATION GROUP N (FA-N, OMIM 610832): "
            "  FA-N phenotype: severe, childhood cancer — overlaps with FA-D1 (BRCA2); "
            "  Early-onset BRCA2-type cancers: Wilms tumor, medulloblastoma, AML; "
            "  Biallelic PALB2 rare but severe; all reported patients developed cancer by age 10; "
            "  Fewer classical BMF anomalies than FA-A (more cancer-predominant); "
            "MONOALLELIC AD → HEREDITARY BREAST CANCER (HIGH RISK): "
            "  PALB2 monoallelic LOF: "
            "    Female breast cancer: 53% cumulative lifetime risk (equivalent to BRCA2 in some cohorts); "
            "    Ovarian cancer: 5% lifetime (lower than BRCA1/2 but real risk); "
            "    PANCREATIC CANCER: 3-4% lifetime; "
            "    Male breast cancer: 1-2% (lower than BRCA2 but elevated); "
            "  PALB2 tier: classified as HIGH-RISK gene for breast cancer (NCCN guideline); "
            "  Surveillance protocol: same as BRCA1/2 (annual MRI + mammogram from age 30); "
            "PALB2 STRUCTURE + FUNCTION: "
            "  N-terminal COILED-COIL domain (aa 1-69): binds BRCA1; "
            "    BRCA1-PALB2 interaction critical for S-phase BRCA1-PALB2-BRCA2 axis; "
            "  C-terminal WD40 repeat domain (7 WD40 repeats): binds BRCA2 + RAD51C; "
            "    WD40 β-propeller creates interaction platform for BRCA2 + RAD51 nucleofilament; "
            "  PALB2 bridges BRCA1 (upstream, DSB signalling) to BRCA2 (downstream, RAD51 loading); "
            "  Without PALB2: BRCA2 cannot localise to DSB sites → HR defective"
        ),
        "disease_category": (
            "FA-N (BIALLELIC): SEVERE FA WITH EARLY CHILDHOOD CANCER: "
            "  WILMS TUMOR: ~25% (similar to FA-D1); "
            "  MEDULLOBLASTOMA: ~20%; BRAIN TUMOR overall ~30%; "
            "  AML/MDS: ~25%; "
            "  BMF: less dominant than FA-A (cancer before classic BMF in many); "
            "  CONGENITAL ANOMALIES: ~60-70% (thumb anomalies, café-au-lait, renal); "
            "MONOALLELIC PALB2 HBOC: "
            "  BREAST CANCER (female): 53% lifetime risk — comparable to BRCA2; "
            "    Predominantly ER+/HER2- histology (some TNBC); "
            "    Family aggregation: PALB2 BC often occurs in family clusters; "
            "  BREAST CANCER (male): elevated (1-2%); "
            "  OVARIAN CANCER: 5% lifetime; less clear surveillance indication; "
            "  PANCREATIC CANCER: 3-4% lifetime; "
            "    Pancreatic endoscopic ultrasound/MRI from age 50 (or 10 years before earliest family PC); "
            "  PARP INHIBITOR SENSITIVITY: "
            "    TBCRC048 trial: olaparib in PALB2-mutated metastatic BC → 82% ORR (HIGHEST ORR of any gene!); "
            "    FDA olaparib label includes gBRCA1/2 + gPALB2 (post-TBCRC048 data); "
            "  PROSTATE CANCER: PALB2 not yet independently established but emerging; "
            "HIGH-RISK STATUS: "
            "  NCCN 2024: PALB2 = high-risk gene (annual MRI + mammogram from age 30); "
            "  RRSO: timing controversial (5% ovarian risk = indeterminate vs BRCA2 threshold)"
        ),
        "disease_pathway": (
            "PALB2 AS BRCA1-BRCA2 MOLECULAR BRIDGE: "
            "BRCA1-PALB2 INTERACTION: "
            "  BRCA1 BRCT domain binds PALB2 coiled-coil N-terminal; "
            "  BRCA1 recruits PALB2 to DSB sites in S/G2 phase; "
            "  Interaction disrupted by BRCA1 BRCT mutations (e.g. p.Met1775Arg → loss PALB2 binding); "
            "PALB2-BRCA2 INTERACTION: "
            "  PALB2 WD40 β-propeller binds BRCA2 N-terminal (aa 10-40); "
            "  PALB2 retains BRCA2 at nuclear foci → without PALB2, BRCA2 cannot localise to DSBs; "
            "  PALB2 also directly interacts with RAD51C and RAD51 → additional recombination function; "
            "BRCA1-PALB2-BRCA2 EPISTASIS: "
            "  The three proteins function in a linear pathway at DSBs; "
            "  HR efficiency: BRCA1 loss ≈ PALB2 loss ≈ BRCA2 loss (all HR deficient); "
            "  Cancer risk: BRCA1 > PALB2 ≈ BRCA2 (epidemiologically); "
            "FA-N PATHWAY: "
            "  FANCD2-Ub PRESENT (downstream of FA-CORE; PALB2 downstream); "
            "  FA-N cells: FANCD2-Ub normal; HR defective (same as FA-D1); "
            "  PALB2 functions at Step 6 (HR-mediated gap fill after ICL unhooking)"
        ),
        "pathognomonic": (
            "FA-N (BIALLELIC): "
            "  DEB BREAKAGE POSITIVE; FANCD2-Ub PRESENT (downstream like FA-D1); "
            "  Desmoplastic medulloblastoma + Wilms tumor in same child: "
            "    Biallelic BRCA2/PALB2 (FA-D1/FA-N) workup → DEB test urgently; "
            "  Parents = obligate PALB2 heterozygous carriers → HBOC counselling mandatory; "
            "HBOC (MONOALLELIC): "
            "  53% BREAST CANCER RISK — comparable to BRCA2; "
            "    PALB2 often underappreciated clinically (BRCA1/2 dominate testing narrative); "
            "  TBCRC048: 82% ORR with olaparib — PALB2 MOST RESPONSIVE non-BRCA1/2 gene to PARPi; "
            "  FAMILY AGGREGATION: PALB2 BC families often multi-generational; "
            "  POPULATION CARRIER: ~1 in 340 general population (high among Finns: PALB2 c.1592delT Finn founder); "
            "    Finnish PALB2 founder: 45% lifetime BC risk in heterozygous Finnish women; "
            "  OLAPARIB FDA LABEL: includes gPALB2 (not just BRCA1/2) — important prescribing knowledge"
        ),
        "treatment": (
            "FA-N (BIALLELIC): "
            "  Same protocols as FA-D1: "
            "    Wilms/medulloblastoma: reduced-dose chemo only; avoid RT; avoid alkylators at standard dose; "
            "    HSCT: for BMF or AML; reduced-intensity conditioning; "
            "MONOALLELIC PALB2 HBOC: "
            "  BREAST CANCER SURVEILLANCE: "
            "    Annual MRI from age 30 + annual mammogram; NCCN high-risk protocol; "
            "  RISK-REDUCING MASTECTOMY: offered; decision guided by 53% lifetime risk discussion; "
            "  RRSO: individualised (5% OC risk = discussed but not as strongly indicated as BRCA1/2); "
            "  PARP INHIBITORS: "
            "    Olaparib FDA-approved for gPALB2 metastatic BC (TBCRC048 data); "
            "    SOLO1-type maintenance trials including PALB2 ongoing; "
            "  PLATINUM SENSITIVITY: as per BRCA2 (DNA repair defect → platinum effective); "
            "  PANCREATIC SURVEILLANCE: EUS/MRI from age 50 if family history; "
            "  MALE CARRIERS: breast self-exam + clinical exam; breast MRI if density high; "
            "  CASCADE TESTING: first-degree relatives of PALB2 carrier; 50% risk"
        ),
    },
    {
        "gene": "BRIP1",
        "seed_base": 2732,
        "protein": (
            "BRIP1 -- 17q23.2 AR-AD -- 1249aa -- FANCJ-BRCA1-Interacting-Protein-C-Helicase-1-"
            "140kDa-MonoUbiquitinated-DEAH-Box-5prime3prime-Helicase-"
            "OMIM-Gene-605882-Disease-FA-J-609054-Ovarian-Cancer-614291"
        ),
        "locus": "17q23.2",
        "protein_size": "1249 aa / 140 kDa (DEAH-box 5'→3' DNA helicase; BRCA1 BRCT-binding)",
        "inheritance": (
            "BIALLELIC AR → FANCONI ANEMIA COMPLEMENTATION GROUP J (FA-J, OMIM 609054): "
            "  FA-J phenotype: childhood BMF + cancer predisposition (AML/SCC); "
            "  Congenital anomalies: 60-70% (similar to FA-A/G); "
            "  Biallelic BRIP1 rare; approximately 1-2% of FA cases; "
            "  Phenotype overlaps FA-A/C/G (less severe than FA-D1/N); "
            "MONOALLELIC AD → HEREDITARY OVARIAN CANCER PREDISPOSITION: "
            "  BRIP1 monoallelic LOF: 11-13x relative risk ovarian cancer; "
            "    Lifetime ovarian cancer risk: ~9-12%; "
            "    NO substantial increase in breast cancer risk (unlike BRCA1/2, PALB2); "
            "    BREAST CANCER RISK NOT ELEVATED — CRITICAL DISTINCTION from other HR genes; "
            "  RRSO recommended: age 45-50 in BRIP1 carriers; "
            "  Surveillance: annual CA-125 + transvaginal ultrasound (limited sensitivity); "
            "BRIP1 STRUCTURE + FUNCTION: "
            "  DEAH-box helicase domain: 5'→3' helicase activity; "
            "  Unwinds G-quadruplex (G4) DNA structures — CRITICAL for FA pathway; "
            "  CTD (C-terminal domain): BRCA1 BRCT domain binding site; "
            "    BRCA1-BRIP1 interaction regulated by ATM phosphorylation (BRIP1 pThr498); "
            "  BRIP1 recruited to stalled forks via FANCD2-Ub interaction; "
            "  BRIP1 unwinds G4 structures that block FA ICL repair; "
            "SYNONYMS: FANCJ = BRIP1 = BACH1"
        ),
        "disease_category": (
            "FA-J (BIALLELIC): CLASSIC FA PHENOTYPE + HELICASE FUNCTION LOSS: "
            "  BONE MARROW FAILURE: progressive pancytopenia; onset 5-10 years; "
            "  AML/MDS: high risk; monosomy 7; "
            "  SCC: post-HSCT risk (head/neck/oesophageal); "
            "  CONGENITAL ANOMALIES (65%): "
            "    CAFÉ-AU-LAIT MACULES: 30-35%; "
            "    THUMB ANOMALIES: 15-20%; "
            "    RENAL: 15-20%; SHORT STATURE: 50%; "
            "  G4-RELATED GENOMIC INSTABILITY: "
            "    BRIP1-null cells: fragile sites at G4 loci; "
            "    Increased chromosomal rearrangements at G4-rich loci (BRCA1 promoter, oncogenes); "
            "MONOALLELIC BRIP1 — OVARIAN CANCER: "
            "  EPITHELIAL OVARIAN CANCER (EOC): predominantly high-grade serous; "
            "  ABSOLUTE RISK: 9-12% lifetime (vs 1.3% general population); "
            "  RELATIVE RISK: 11-13x; "
            "  NO BREAST CANCER ELEVATION: "
            "    CRITICAL CLINICAL PEARL: BRIP1 heterozygous carriers DO NOT have elevated breast cancer risk; "
            "    This distinguishes BRIP1 from BRCA1/2/PALB2/RAD51C/RAD51D (all have some breast risk); "
            "  AGE OF OC ONSET: median ~54 years (later than BRCA1 ~50 years); "
            "  CHEMOSENSITIVITY: BRIP1-mutated OC may be platinum/PARPi sensitive (emerging data); "
            "  FALLOPIAN TUBE PRIMARIES: included in EOC spectrum; "
            "  SURVEILLANCE: annual TVUS + CA-125 from age 35; low sensitivity but standard; "
            "  RRSO: age 45-50 (later than BRCA1/2 due to lower absolute risk + later onset)"
        ),
        "disease_pathway": (
            "BRIP1/FANCJ HELICASE IN ICL REPAIR: "
            "G-QUADRUPLEX (G4) RESOLUTION — PRIMARY FUNCTION: "
            "  G4 structures form spontaneously at G-rich sequences (telomeres, promoters, oncogene loci); "
            "  G4 are physical blocks to replication fork progression and ICL repair intermediates; "
            "  BRIP1 5'→3' helicase unwinds G4 → allows FA ICL repair machinery to access crosslink; "
            "  BRIP1-null: G4 structures persist → fork stalling → replication catastrophe; "
            "BRCA1 RECRUITMENT: "
            "  BRIP1 binds BRCA1 tandem BRCT domain via pThr498 motif; "
            "  BRCA1-BRIP1 interaction: ATM phosphorylates BRIP1 Thr498 after DSB → BRCT binding; "
            "  BRIP1 recruited to DSB sites via BRCA1 → participates in HR pathway; "
            "  BRIP1 activity: unwinds DNA substrates to allow RPA binding + RAD51 loading; "
            "FANCD2-UB INTERACTION: "
            "  BRIP1 contains ubiquitin-binding zinc finger (UBZ4) domain; "
            "  Binds monoubiquitinated FANCD2 → recruited to ICL chromatin; "
            "  BRIP1 required at Step 4 (after ID2 loading, before unhooking); "
            "  BRIP1 unwinds G4 at ICL vicinity → allows XPF-ERCC1 access for unhooking incisions; "
            "FA-J PATHWAY POSITION: "
            "  FANCD2-Ub PRESENT (downstream group); ICL repair blocked at G4 resolution step; "
            "BRIP1 AT TELOMERES: "
            "  BRIP1 resolves telomeric G4 → BRIP1-null: telomere fragility; "
            "  Sister telomere loss: chromosomal instability mechanism"
        ),
        "pathognomonic": (
            "FA-J (BIALLELIC): "
            "  DEB BREAKAGE POSITIVE; FANCD2-Ub PRESENT (downstream group); "
            "  G4-RELATED CHROMOSOMAL FRAGILITY: "
            "    FA-J cells show breaks specifically at G4-rich loci after treatment with G4-stabilising ligands; "
            "    pyridostatin (G4 ligand) → selective toxicity in BRIP1-null cells; "
            "MONOALLELIC BRIP1: "
            "  OVARIAN CANCER WITHOUT BREAST CANCER: "
            "    BRIP1 is the key HR gene that causes OC but NOT BC; "
            "    Woman with OC + family OC (no BC) → BRIP1 test priority (after BRCA1/2/RAD51C/D); "
            "  NCCN INCLUDES BRIP1 in ovarian cancer-specific gene panel; "
            "  RRSO INDICATED at age 45-50: guideline-driven; "
            "  BRIP1 CARRIER FREQUENCY: ~1 in 1000-1200 general population; "
            "  BACH1 NAME: BRIP1 originally discovered as BACH1 (BTB and CNC Homology 1); "
            "    Same gene: BRCA1 IP1 (BRIP1) = FANCJ = BACH1; "
            "  BRIP1 OC chemosensitivity: platinum-sensitive EOC pattern; olaparib emerging data"
        ),
        "treatment": (
            "FA-J (BIALLELIC): "
            "  HSCT: fludarabine-based conditioning; standard FA HSCT protocol; "
            "  ANDROGEN BRIDGE + CANCER SURVEILLANCE: same as FA-A; "
            "  AVOID ALKYLATORS/RT: standard FA precautions; "
            "MONOALLELIC BRIP1 OC PREVENTION: "
            "  RRSO: recommended age 45-50 (NCCN); "
            "    Oophorectomy (± salpingectomy) — salpingectomy alone NOT adequate; "
            "  ORAL CONTRACEPTIVES: not routinely recommended (data insufficient in BRIP1); "
            "  SURVEILLANCE (until RRSO): annual TVUS + CA-125 from age 35; "
            "  OC TREATMENT (if diagnosed): "
            "    Platinum-based chemotherapy (carboplatin + paclitaxel) first-line; "
            "    PARP inhibitors: emerging evidence; olaparib maintenance trial data pending; "
            "    Bevacizumab: added to standard chemo in advanced EOC; "
            "  BREAST SURVEILLANCE: standard population breast screening (not enhanced); "
            "    Critical: BRIP1 carriers do NOT need MRI breast screening (no elevated risk); "
            "  CASCADE TESTING: first-degree relatives of BRIP1 carrier"
        ),
    },
    {
        "gene": "FANCI",
        "seed_base": 2733,
        "protein": (
            "FANCI -- 15q26.1 AR -- 1328aa -- Fanconi-Anemia-Complementation-Group-I-"
            "147kDa-FANCD2-Heterodimer-FANCI-FANCD2-ID2-Complex-K523-Monoubiquitination-"
            "OMIM-Gene-611360-Disease-Fanconi-Anemia-I-609053"
        ),
        "locus": "15q26.1",
        "protein_size": "1328 aa / 147 kDa (ID2 complex partner; K523 monoubiquitination; FANCD2 stabiliser)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → Fanconi anemia complementation group I); "
            "FANCI forms an obligate heterodimer with FANCD2 = the ID2 (FANCI-FANCD2) complex; "
            "FANCI monoubiquitinated at K523 by FANCL+UBE2T simultaneously with FANCD2 K561; "
            "FANCI + FANCD2 MUTUALLY STABILISE each other: "
            "  FANCI-null → FANCD2 also destabilised; FANCD2-null → FANCI destabilised; "
            "  Western blot: FANCI-null → both FANCI and FANCD2 reduced; "
            "FANCI accounts for ~1-2% of FA cases; less common than FANCA/C/G; "
            "STRUCTURE: "
            "  FANCI has ARM-repeat (Armadillo-type) solenoid → DNA binding + protein interaction; "
            "  FANCI ARM repeats bind ssDNA/dsDNA junction — models ICL lesion structure; "
            "  FANCD2 has similar structure → ID2 = clamp-like architecture around DNA; "
            "  FANCI ARM domains + FANCD2 ARM domains = synergistic DNA-binding clamp; "
            "FANCI PHOSPHORYLATION: "
            "  FANCI Ser730 phosphorylated by ATR → required for FANCI-FANCD2 monoubiquitination; "
            "  ATR-FANCI axis: replication stress activates ATR → pFANCI-S730 → ID2 activation; "
            "PHENOTYPE: "
            "  FA-I overlaps with FA-D2 (FANCI + FANCD2 = same complex); "
            "  BMF + congenital anomalies + cancer (AML/SCC); "
            "  BRAHMA-RELATED GENE 1 (BRG1/SMARCA4) recruited by FANCD2-Ub: "
            "    FANCI provides docking for chromatin remodelling at ICL"
        ),
        "disease_category": (
            "FANCONI ANEMIA GROUP I (OMIM 609053); "
            "OVERLAPS CLINICALLY WITH FA-D2 (same ID2 complex): "
            "BONE MARROW FAILURE: "
            "  Progressive pancytopenia; onset median 7-8 years; "
            "  Thrombocytopenia → anaemia → neutropenia (typical progression); "
            "  AML risk: ~800x general population; "
            "  Monosomy 7 common cytogenetic change; "
            "CONGENITAL ANOMALIES (~65%): "
            "  CAFÉ-AU-LAIT MACULES: 30-35%; "
            "  THUMB/RADIAL ANOMALIES: 20-25%; "
            "  RENAL MALFORMATIONS: 15-20%; "
            "  MICROCEPHALY: 20%; "
            "  SHORT STATURE: 50-55%; "
            "ENDOCRINE MANIFESTATIONS: "
            "  MALE HYPOGONADISM: 60-70% of FA-I males; "
            "  GROWTH HORMONE DEFICIENCY: 30-40%; "
            "  DIABETES MELLITUS (insulin resistance): 40-50%; "
            "CANCER PREDISPOSITION: "
            "  AML/MDS: similar to FA-D2; "
            "  SCC: head/neck/oesophageal/anogenital; "
            "  Brain tumors: described (lower than FA-D1/N); "
            "ID2 COMPLEX — FUNCTIONAL POINT OF CONVERGENCE: "
            "  All FA upstream pathway genes (FANCA/C/E/F/G/L/M = FA-CORE) converge on ID2 monoubiquitination; "
            "  All FA downstream pathway genes (BRCA2/PALB2/BRIP1/RAD51C) operate AFTER ID2 activation; "
            "  FANCI + FANCD2 = the CENTRAL NODE of the entire FA pathway"
        ),
        "disease_pathway": (
            "FANCI-FANCD2 (ID2) COMPLEX — CENTRAL FA NODE: "
            "ID2 ASSEMBLY: "
            "  FANCI + FANCD2 form constitutive heterodimer (stable in S-phase); "
            "  ID2 = arm-solenoid clamp structure → encircles dsDNA at ICL; "
            "  FANCD2 ARM domain contacts one DNA strand; FANCI ARM domain contacts complementary strand; "
            "MONOUBIQUITINATION OF ID2: "
            "  FANCL-UBE2T ubiquitinates FANCD2 K561 + FANCI K523 simultaneously; "
            "  Monoubiquitination closes the ID2 clamp → high-affinity DNA binding; "
            "  Phosphorylation of FANCI Ser730 by ATR = prerequisite for K523 Ub; "
            "ID2 CHROMATIN LOADING: "
            "  Ub-ID2 clamp has higher affinity for branched DNA (fork junction, D-loop) than linear; "
            "  ID2 loaded at converging replication forks at ICL; "
            "DOWNSTREAM SIGNALLING FROM ID2-Ub: "
            "  ID2-Ub recruits: "
            "    (a) FAN1 nuclease (via FANCD2-Ub-UBZ binding) → auxiliary ICL incisions; "
            "    (b) SLX4/FANCP scaffold → XPF-ERCC1 unhooking; "
            "    (c) BRCA2/PALB2 → RAD51 HR; "
            "    (d) BRG1/SMARCA4 (chromatin remodeller) → nucleosome displacement at ICL; "
            "ID2 DEUBIQUITINATION: "
            "  USP1-UAF1 removes Ub from ID2 after repair → restores basal ID2 state; "
            "  USP1 regulation: USP1 autocleavage during S-phase limits activity; "
            "    USP1 inhibitors under investigation as FA therapeutic target (paradoxically may help non-FA cancer); "
            "FANCI S730 PHOSPHORYLATION CYCLE: "
            "  ATR activated by RPA-coated ssDNA at stalled fork → FANCI pS730 → "
            "    ID2 ubiquitination facilitated → ICL repair active; "
            "  PP2A phosphatase removes pS730 → resets FANCI for next round"
        ),
        "pathognomonic": (
            "DEB/MMC CHROMOSOMAL BREAKAGE: POSITIVE; "
            "ID2-SPECIFIC WESTERN BLOT: "
            "  FANCI-null → BOTH FANCI-Ub AND FANCD2-Ub reduced/absent: "
            "    (FANCD2 destabilised without FANCI partner); "
            "  Distinguishes FA-I from FA-CORE: "
            "    FA-CORE cells: FANCD2 protein present but not ubiquitinated; "
            "    FA-I cells: FANCD2 protein also reduced (co-destabilisation); "
            "  FANCI ANTIBODY: "
            "    Both short (unUb) and long (Ub) FANCI bands absent/reduced in FA-I; "
            "  FANCI NUCLEAR FOCI: "
            "    Normal cells after MMC: co-localising FANCI-FANCD2 nuclear foci (γH2AX, PCNA); "
            "    FA-I cells: no FANCI foci; FANCD2 foci also absent (ID2 requires both partners); "
            "COMPLEMENTATION: "
            "  FA-I vs FA-D2 distinction: only by molecular testing or lentiviral complementation; "
            "  Clinical phenotype identical (same complex); "
            "ATR-CHECKPOINT DYSFUNCTION: "
            "  ATR hyperactivation with unresolved replication stress in FANCI-null cells; "
            "  Elevated phospho-RPA32 S33 (ATR substrate) in untreated cells"
        ),
        "treatment": (
            "HSCT: fludarabine-based conditioning (standard FA protocol); "
            "  FA-I outcomes: similar to FA-A with matched sibling donor; "
            "  Important: reduced-intensity conditioning mandatory; avoid cyclophosphamide alone; "
            "ANDROGEN BRIDGE: oxymetholone/danazol pending HSCT; "
            "HYPOGONADISM MANAGEMENT: "
            "  TESTOSTERONE REPLACEMENT: if primary hypogonadism confirmed (LH↑, testosterone↓); "
            "  Timing: pubertal induction as appropriate; "
            "  Fertility: sperm banking before puberty lost; HSCT-associated gonadotoxicity additional risk; "
            "DIABETES MANAGEMENT: "
            "  INSULIN RESISTANCE: metformin first-line if pre-diabetic; "
            "  Full diabetes management if overt: insulin or oral agents; "
            "GH REPLACEMENT: if GH deficiency confirmed (ITT or stimulation test); "
            "SCC SURVEILLANCE POST-HSCT: same annual protocol; "
            "GENE THERAPY: no specific FANCI program yet; FANCA-directed GT may reduce oncology risk indirectly; "
            "AVOID: alkylating agents (DEB/MMC class drugs = pathognomonic sensitivity); "
            "  RADIATION SENSITIVITY: reduce RT dose 30-50% if unavoidable; "
            "FANCONI ANEMIA RESEARCH FUND: clinical trial access + natural history study enrollment recommended"
        ),
    },
]


def _make_patients(gene_data, n=40):
    """Generate synthetic Fanconi anemia patient records for a gene."""
    rng = random.Random(gene_data["seed_base"])
    gene = gene_data["gene"]

    GENE_PARAMS = {
        "FANCA": dict(
            onset_range=(2, 15), bmf=0.90, aml=0.35, scc=0.30, wilms=0.02,
            medulloblastoma=0.01, thumb_defect=0.35, radial_defect=0.28,
            cafe_au_lait=0.35, renal_anomaly=0.25, short_stature=0.50,
            hsct_done=0.65, somatic_mosaicism=0.25, diabetes=0.45,
        ),
        "FANCC": dict(
            onset_range=(2, 12), bmf=0.92, aml=0.38, scc=0.28, wilms=0.01,
            medulloblastoma=0.01, thumb_defect=0.20, radial_defect=0.15,
            cafe_au_lait=0.38, renal_anomaly=0.22, short_stature=0.52,
            hsct_done=0.68, somatic_mosaicism=0.20, diabetes=0.42,
        ),
        "FANCD2": dict(
            onset_range=(2, 14), bmf=0.88, aml=0.32, scc=0.28, wilms=0.04,
            medulloblastoma=0.04, thumb_defect=0.32, radial_defect=0.25,
            cafe_au_lait=0.32, renal_anomaly=0.22, short_stature=0.50,
            hsct_done=0.62, somatic_mosaicism=0.22, diabetes=0.40,
        ),
        "FANCG": dict(
            onset_range=(2, 12), bmf=0.95, aml=0.42, scc=0.25, wilms=0.01,
            medulloblastoma=0.01, thumb_defect=0.15, radial_defect=0.10,
            cafe_au_lait=0.30, renal_anomaly=0.18, short_stature=0.55,
            hsct_done=0.72, somatic_mosaicism=0.18, diabetes=0.55,
        ),
        "BRCA2": dict(
            onset_range=(1, 8), bmf=0.60, aml=0.25, scc=0.10, wilms=0.30,
            medulloblastoma=0.20, thumb_defect=0.28, radial_defect=0.20,
            cafe_au_lait=0.30, renal_anomaly=0.20, short_stature=0.40,
            hsct_done=0.55, somatic_mosaicism=0.10, diabetes=0.20,
        ),
        "PALB2": dict(
            onset_range=(1, 9), bmf=0.55, aml=0.22, scc=0.08, wilms=0.25,
            medulloblastoma=0.18, thumb_defect=0.25, radial_defect=0.18,
            cafe_au_lait=0.28, renal_anomaly=0.18, short_stature=0.38,
            hsct_done=0.50, somatic_mosaicism=0.08, diabetes=0.18,
        ),
        "BRIP1": dict(
            onset_range=(3, 15), bmf=0.82, aml=0.30, scc=0.28, wilms=0.05,
            medulloblastoma=0.05, thumb_defect=0.18, radial_defect=0.12,
            cafe_au_lait=0.30, renal_anomaly=0.18, short_stature=0.48,
            hsct_done=0.58, somatic_mosaicism=0.20, diabetes=0.38,
        ),
        "FANCI": dict(
            onset_range=(2, 14), bmf=0.88, aml=0.30, scc=0.28, wilms=0.03,
            medulloblastoma=0.03, thumb_defect=0.22, radial_defect=0.18,
            cafe_au_lait=0.32, renal_anomaly=0.18, short_stature=0.52,
            hsct_done=0.60, somatic_mosaicism=0.20, diabetes=0.45,
        ),
    }

    p = GENE_PARAMS.get(gene, GENE_PARAMS["FANCA"])
    patients = []
    for i in range(n):
        onset = round(rng.uniform(*p["onset_range"]), 1)
        age_current = round(onset + rng.uniform(3, 35), 1)
        patients.append({
            "patient_id": f"{gene}-{gene_data['seed_base']}-{i+1:02d}",
            "sex": rng.choice(["M", "F"]),
            "age_onset_years": onset,
            "age_current_years": min(age_current, 55.0),
            "bone_marrow_failure": rng.random() < p["bmf"],
            "aml_mds": rng.random() < p["aml"],
            "scc_head_neck_oesophageal": rng.random() < p["scc"],
            "wilms_tumor": rng.random() < p["wilms"],
            "medulloblastoma": rng.random() < p["medulloblastoma"],
            "thumb_radial_defect": rng.random() < p["thumb_defect"],
            "cafe_au_lait_macules": rng.random() < p["cafe_au_lait"],
            "renal_anomaly": rng.random() < p["renal_anomaly"],
            "short_stature": rng.random() < p["short_stature"],
            "hsct_performed": rng.random() < p["hsct_done"],
            "somatic_mosaicism_detected": rng.random() < p["somatic_mosaicism"],
            "diabetes_mellitus": rng.random() < p["diabetes"],
        })
    return patients


def generate_overview():
    all_genes = []
    total_patients = 0

    GENE_PARAMS = {
        "FANCA": dict(onset=7, bmf=90, aml=35, scc=30, wilms=2, mb=1, thumb=35, cafe=35, renal=25, hsct=65, diabetes=45),
        "FANCC": dict(onset=6, bmf=92, aml=38, scc=28, wilms=1, mb=1, thumb=20, cafe=38, renal=22, hsct=68, diabetes=42),
        "FANCD2": dict(onset=7, bmf=88, aml=32, scc=28, wilms=4, mb=4, thumb=32, cafe=32, renal=22, hsct=62, diabetes=40),
        "FANCG": dict(onset=6, bmf=95, aml=42, scc=25, wilms=1, mb=1, thumb=15, cafe=30, renal=18, hsct=72, diabetes=55),
        "BRCA2": dict(onset=2, bmf=60, aml=25, scc=10, wilms=30, mb=20, thumb=28, cafe=30, renal=20, hsct=55, diabetes=20),
        "PALB2": dict(onset=2, bmf=55, aml=22, scc=8, wilms=25, mb=18, thumb=25, cafe=28, renal=18, hsct=50, diabetes=18),
        "BRIP1": dict(onset=8, bmf=82, aml=30, scc=28, wilms=5, mb=5, thumb=18, cafe=30, renal=18, hsct=58, diabetes=38),
        "FANCI": dict(onset=7, bmf=88, aml=30, scc=28, wilms=3, mb=3, thumb=22, cafe=32, renal=18, hsct=60, diabetes=45),
    }

    for gene_data in ATLAS_GENES:
        gene = gene_data["gene"]
        p = GENE_PARAMS[gene]
        patients = _make_patients(gene_data)
        total_patients += len(patients)
        all_genes.append({
            "gene": gene,
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "n_patients": len(patients),
            "mean_onset_years": p["onset"],
            "pct_bmf": p["bmf"],
            "pct_aml_mds": p["aml"],
            "pct_scc": p["scc"],
            "pct_wilms": p["wilms"],
            "pct_medulloblastoma": p["mb"],
            "pct_thumb_defect": p["thumb"],
            "pct_cafe_au_lait": p["cafe"],
            "pct_renal": p["renal"],
            "pct_hsct": p["hsct"],
            "pct_diabetes": p["diabetes"],
        })

    return {
        "atlas": "Hereditary Fanconi Anemia & DNA Crosslink Repair Atlas",
        "subtitle": "Complete 8-Gene ICL Repair Reference (FA-A/C/D2/G/D1/N/J/I)",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "gene_summaries": all_genes,
        "seeds": "2726-2733",
        "pathway_categories": [
            {
                "pathway": "FA Core Complex (E3 Ubiquitin Ligase — ICL Recognition & FANCD2/FANCI Activation)",
                "genes": ["FANCA", "FANCC", "FANCG"],
                "note": (
                    "FA-CORE (FANCA+FANCB+FANCC+FANCE+FANCF+FANCG+FANCL+FANCM) = ICL-sensing E3 ligase; "
                    "FANCL+UBE2T ubiquitinates FANCD2-K561+FANCI-K523; "
                    "FANCA (60-65% of FA) + FANCC (15%) + FANCG (10%) = three most common FA genes; "
                    "FANCG required for FANCA nuclear import (TPR scaffold); "
                    "FANCC cytoplasmic STAT1 inhibitor → prevents haematopoietic apoptosis; "
                    "FA-CORE LOF → FANCD2-Ub absent on Western blot (diagnostic)"
                ),
            },
            {
                "pathway": "ID2 Complex (FANCD2-FANCI Clamp — Central FA Pathway Hub)",
                "genes": ["FANCD2", "FANCI"],
                "note": (
                    "ID2 = FANCD2+FANCI heterodimer; both monoubiquitinated simultaneously by FA-CORE; "
                    "ID2 clamps around DNA at stalled fork; recruits FAN1+SLX4+BRCA2/PALB2; "
                    "FANCD2 K561-Ub + FANCI K523-Ub = gatekeeper Western blot for FA pathway groups; "
                    "FA-CORE group: FANCD2-Ub absent; ID2 group (FANCD2/FANCI): protein also reduced; "
                    "Downstream (BRCA2/PALB2/BRIP1): FANCD2-Ub PRESENT"
                ),
            },
            {
                "pathway": "HR-Competent FA Genes (BRCA2/PALB2/BRIP1 — Biallelic FA + Monoallelic HBOC/OC)",
                "genes": ["BRCA2", "PALB2", "BRIP1"],
                "note": (
                    "BRCA2/FANCD1: biallelic→FA-D1 (childhood Wilms+MB+AML); monoallelic→HBOC (breast+ovarian+pancreatic+prostate); "
                    "PALB2/FANCN: biallelic→FA-N (similar to FA-D1); monoallelic→53% lifetime breast cancer (TBCRC048 olaparib 82% ORR); "
                    "BRIP1/FANCJ: biallelic→FA-J; monoallelic→OC only (NO breast elevation) — CRITICAL clinical distinction; "
                    "FANCD2-Ub PRESENT in all three biallelic groups (downstream); HR defective"
                ),
            },
        ],
        "critical_distinctions": [
            "DEB/MMC CHROMOSOMAL BREAKAGE TEST: GOLD STANDARD for FA diagnosis — ALL FA groups positive; DOES NOT identify complementation group; SOMATIC MOSAICISM (25% FA-A): false-negative DEB in blood → test fibroblasts if clinical suspicion high with normal DEB",
            "FANCD2-Ub WESTERN BLOT GATEKEEPER: FA-CORE group (FANCA/C/G) = absent FANCD2-Ub; FA-D2 = reduced/absent FANCD2 protein; FA-I = reduced both FANCD2 and FANCI; Downstream (BRCA2/PALB2/BRIP1) = FANCD2-Ub PRESENT — HR defective, not ubiquitination",
            "CYCLOPHOSPHAMIDE ABSOLUTE CONTRAINDICATION in FA: standard high-dose CY causes fatal toxicity; FA-conditioning = fludarabine+low-dose CY+ATG; ALL FA patients must carry FA alert card; ANY anaesthetic team must be informed before chemotherapy/transplant",
            "FA-D1/FA-N EARLY CHILDHOOD CANCERS: biallelic BRCA2 or PALB2 → median cancer by age 2-3; ALL children with Wilms tumor ≤2 years or desmoplastic medulloblastoma ≤5 years → DEB test URGENTLY; treatment modification mandatory (no standard-dose RT/alkylators)",
            "BRIP1 MONOALLELIC: OVARIAN cancer risk 11-13x BUT NO BREAST CANCER ELEVATION — do NOT recommend enhanced breast screening for BRIP1 carriers; RRSO age 45-50 (later than BRCA1/2); PALB2 monoallelic: BOTH breast (53%) AND some ovarian (5%) risk — different screening",
            "FANCA SOUTH AFRICAN FOUNDER + FANCG SA FOUNDER: IVS4+4A>T occurs in BOTH FANCA (as FANCC Ashkenazi) AND FANCG — SAME splice variant name, DIFFERENT genes; Black SA patient with FA → test FANCG IVS4+4A>T (not FANCA) as priority",
            "FANCC ASHKENAZI IVS4+4A>T: 1 in 89 carrier frequency — one of most common Ashkenazi recessive disease alleles; mandatory inclusion in routine Ashkenazi carrier panel; prenatal diagnosis with DEB test + molecular confirmation",
            "OLAPARIB FDA INDICATIONS INCLUDE gPALB2: TBCRC048 82% ORR → olaparib approved for gPALB2 metastatic breast cancer (not just BRCA1/2); MOST RESPONSIVE gene to PARPi proportionally; prescribers must know PALB2 label inclusion",
            "RADIAL RAY DEFECT IN NEWBORN: absent/hypoplastic thumb or radius (unilateral or bilateral) → FA workup MANDATORY alongside VACTERL evaluation; bilateral absent thumbs = FA first differential; DEB test from cord blood or peripheral blood",
            "FANCG AZOOSPERMIA: FA-G males have higher rate of non-obstructive azoospermia (testicular failure) than FA-A/C; FA diagnosis in infertile male with BMF history → FANCG priority; sperm banking not possible; semen analysis early",
        ],
    }


def generate_breakdown():
    genes_out = []
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        pct = lambda k: round(100 * sum(p[k] for p in patients) / len(patients))
        genes_out.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "n_patients": len(patients),
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "pct_bmf": pct("bone_marrow_failure"),
            "pct_aml_mds": pct("aml_mds"),
            "pct_scc": pct("scc_head_neck_oesophageal"),
            "pct_wilms": pct("wilms_tumor"),
            "pct_medulloblastoma": pct("medulloblastoma"),
            "pct_thumb_defect": pct("thumb_radial_defect"),
            "pct_cafe_au_lait": pct("cafe_au_lait_macules"),
            "pct_renal": pct("renal_anomaly"),
            "pct_hsct": pct("hsct_performed"),
            "pct_mosaicism": pct("somatic_mosaicism_detected"),
            "pct_diabetes": pct("diabetes_mellitus"),
            "patients": patients[:40],
        })
    return {"genes": genes_out}


def generate_definitions():
    return {
        "glossary": {
            "Fanconi Anemia (FA)": (
                "Autosomal recessive (or X-linked, FANCB) DNA repair disorder; "
                "progressive bone marrow failure + congenital anomalies + cancer predisposition; "
                "caused by LOF in any of 22+ FANC genes; "
                "prevalence 1 in 130,000-160,000; most common in Ashkenazi Jewish (FANCC), "
                "Black South African (FANCA/FANCG), and Spanish Gypsy (FANCA) populations"
            ),
            "ICL (Interstrand DNA Crosslink)": (
                "Covalent bond linking complementary DNA strands — blocks strand separation + replication; "
                "caused by: bifunctional alkylating agents (cyclophosphamide, mitomycin C, cisplatin), "
                "natural aldehydes (formaldehyde, acetaldehyde from alcohol); "
                "ICL repair requires FA pathway in S-phase (converging forks); "
                "FA cells: hypersensitive to DEB/MMC/cisplatin — PATHOGNOMONIC diagnostic sensitivity"
            ),
            "DEB Test (Diepoxybutane Chromosomal Breakage)": (
                "Gold-standard FA diagnostic test; DEB 0.1 μg/mL added to lymphocyte culture; "
                "NORMAL: <0.5 breaks/cell; FA POSITIVE: ≥4 breaks/cell + quadriradials; "
                "SOMATIC MOSAICISM PITFALL: 25-30% FA have reversion → normal blood DEB; "
                "→ test fibroblasts/skin if clinical suspicion + normal blood DEB; "
                "MMC (mitomycin C) alternative to DEB (same sensitivity); "
                "FLOW CYTOMETRY: G2/M arrest after MMC — faster but less specific than DEB"
            ),
            "FANCD2 Monoubiquitination": (
                "Central event in FA pathway; FANCL+UBE2T monoubiquitinates FANCD2-K561 (and FANCI-K523); "
                "Western blot: two bands — short (unUb) + long (Ub); "
                "FA-CORE group: only short band after MMC (no Ub); "
                "FA-D2/FA-I: absent or reduced both bands; "
                "Downstream (BRCA2/PALB2/BRIP1): long Ub band present — ubiquitination normal, HR defective; "
                "USP1-UAF1 deubiquitinase removes Ub after repair completion"
            ),
            "FA-CORE Complex": (
                "E3 ubiquitin ligase complex: FANCA-FANCB-FANCC-FANCE-FANCF-FANCG-FANCL-FANCM; "
                "Assembled at stalled replication forks at ICL; "
                "FANCL = RING-type E3 ligase catalytic subunit; UBE2T = cognate E2; "
                "FANCM = translocase/anchor that tethers FA-CORE to stalled fork; "
                "LOF in any FA-CORE member → FANCD2-Ub absent → ICL repair blocked"
            ),
            "ID2 Complex (FANCD2-FANCI)": (
                "FANCD2+FANCI obligate heterodimer; both ARM-solenoid fold clamps around DNA; "
                "both monoubiquitinated simultaneously in S-phase by FA-CORE; "
                "FANCD2 K561-Ub + FANCI K523-Ub = activated ID2 clamp; "
                "loaded onto chromatin at converging replication forks; "
                "recruits FAN1, SLX4, BRCA2/PALB2 for ICL resolution"
            ),
            "Somatic Mosaicism in FA": (
                "Back-mutation or intragenic recombination restores FANC gene function in HSC clone; "
                "HSC advantage → clonal expansion → dominates peripheral blood; "
                "RESULT: normal DEB test in blood despite germline FA mutation; "
                "FREQUENCY: ~25-30% of FA patients at time of diagnosis; "
                "CLINICAL CONSEQUENCE: good blood counts + normal DEB → delayed FA diagnosis; "
                "SOLUTION: test fibroblasts (skin biopsy) if clinical FA with normal blood DEB"
            ),
            "Radial Ray Defect": (
                "Absent or hypoplastic thumb ± radius aplasia — most distinctive FA congenital anomaly; "
                "PREVALENCE: 10-35% of FA (higher in FANCA group); "
                "PATHOGNOMONIC VALUE: bilateral absent thumbs in newborn = FA until proven otherwise; "
                "VACTERL DISTINCTION: radial-only defect (no vertebral/oesophageal/renal) = FA first; "
                "VATER/VACTERL may coexist with FA — DEB test in all VACTERL patients; "
                "MANAGEMENT: thumb reconstruction surgery (pollicisation); HSCT before surgery preferred"
            ),
            "Cyclophosphamide Contraindication in FA": (
                "FA cells hypersensitive to bifunctional alkylating agents (ICL-inducing drugs); "
                "Standard-dose cyclophosphamide = FATAL toxicity in FA patients; "
                "FA HSCT conditioning: fludarabine (30 mg/m²/day × 4-6) + low-dose CY (10 mg/kg total) + ATG; "
                "NOT high-dose CY (200 mg/kg) used in aplastic anaemia (non-FA protocol); "
                "FA ALERT CARD: carried by all FA patients; anaesthetic team + oncologist must be informed; "
                "Also avoid: mitomycin C, cisplatin, DEB, melphalan at standard dose"
            ),
            "Desmoplastic Medulloblastoma in FA-D1": (
                "Medulloblastoma histological subtype with desmoplastic/nodular features; "
                "PATHOGNOMONIC of biallelic BRCA2 (FA-D1) in children ≤5 years; "
                "SHH-pathway medulloblastoma in young infant → FA-D1 workup MANDATORY; "
                "Treatment challenge: standard craniospinal RT = fatal toxicity in FA; "
                "Protocol: reduced-intensity chemo only; surgical resection; no RT if FA confirmed; "
                "Histology: pale desmoplastic islands + proliferating nodules on H&E"
            ),
            "BRIP1 Ovarian Cancer — No Breast Risk": (
                "CRITICAL CLINICAL DISTINCTION: BRIP1 heterozygous LOF → elevated ovarian cancer (11-13x relative risk) "
                "but NO elevated breast cancer risk; "
                "Distinguishes BRIP1 from BRCA1/BRCA2/PALB2/RAD51C/RAD51D (all have breast + ovarian); "
                "Management implications: BRIP1 carriers DO NOT need MRI breast surveillance; "
                "RRSO timing: age 45-50 (later than BRCA1/2 due to lower absolute risk + later onset); "
                "NCCN: includes BRIP1 as ovarian cancer gene in hereditary OC panel"
            ),
            "PARPi in FA-Pathway Gene Carriers": (
                "PARP inhibitors exploit HR deficiency (synthetic lethality with BRCA/FA LOF); "
                "OLAPARIB FDA APPROVALS: "
                "  gBRCA1/2 BC (2018); gBRCA1/2 OC (2014 maint, 2018 1L); "
                "  gBRCA2 pancreatic (POLO 2019); gBRCA1/2 prostate (PROfound 2020); "
                "  gPALB2 BC (post-TBCRC048 label expansion); "
                "TBCRC048 PALB2: 82% ORR — HIGHEST PARPi response of any non-BRCA gene; "
                "BRIP1: emerging data; not yet FDA-approved for BRIP1; "
                "FANCA/C/G/D2/I biallelic: in vitro sensitivity; clinical trials exploring FA-specific PARPi"
            ),
            "Fanconics (FA Carrier Testing)": (
                "Term for comprehensive FA molecular testing + counselling; "
                "ASHKENAZI JEWISH PANEL: FANCC IVS4+4A>T mandatory (1 in 89 carrier); "
                "SOUTH AFRICAN BLACK: FANCG IVS4+4A>T + FANCA del exon 11-17 first-line; "
                "SPANISH GYPSY (ROMANI): FANCA p.Val1264del first; "
                "GENERAL POPULATION: FA NGS gene panel (22 genes) if DEB positive; "
                "MLPA for FANCA deletions: large deletions >30% of FANCA pathogenic variants; "
                "PREIMPLANTATION GENETIC TESTING (PGT-M): available; recommended for carrier couples"
            ),
            "Olaparib TBCRC048 (PALB2)": (
                "TBCRC048 phase II trial: olaparib monotherapy in advanced BC with gPALB2 or somatic PALB2 LOF; "
                "gPALB2 cohort: ORR 82% — highest response rate of any non-BRCA HR gene to PARPi; "
                "gBRCA1/2 comparison: olaparib ORR typically 50-60% in metastatic BC; "
                "PALB2 data led to FDA label expansion for olaparib to include gPALB2; "
                "Clinical implication: PALB2 metastatic BC should receive olaparib (not just BRCA1/2)"
            ),
        },
        "standards": [
            "NCCN Guidelines v1.2025: Fanconi Anemia (FA) — NCCN Genetic/Familial High-Risk Assessment",
            "FANCONI ANEMIA RESEARCH FUND Guidelines: Fanconi Anemia — A Handbook for Families and Their Physicians, 5th ed.",
            "Alter BP et al. Blood 2018: Cancer in Fanconi anemia — natural history and outcomes",
            "NCCN Guidelines v4.2025: Genetic/Familial High-Risk Assessment Breast, Ovarian, and Pancreatic",
            "TBCRC048: Tung et al. JCO 2020 — olaparib in gPALB2/somatic PALB2 advanced breast cancer (82% ORR)",
            "POLO Trial: Golan et al. NEJM 2019 — olaparib maintenance in gBRCA2 pancreatic cancer",
            "PROfound Trial: de Bono et al. NEJM 2020 — olaparib in gBRCA1/2 prostate cancer",
            "PARITY: BRIP1 ovarian cancer — Ramus et al. Am J Hum Genet 2015 (11x RR)",
            "Cantor SB 2019: BRIP1/FANCJ helicase mechanisms in FA pathway",
            "Nalepa G, Clapp DW. Nat Rev Cancer 2018: Fanconi anaemia and cancer",
            "Walsh T, King MC 2007: Ten genes for inherited breast cancer — PALB2 discovery",
            "D'Andrea AD 2010: Susceptibility pathways in Fanconi anemia and breast cancer (NEJM review)",
            "Bogliolo M, Surrallés J 2015: Fanconi anemia: a model disease for studies on human genetics",
        ],
    }
