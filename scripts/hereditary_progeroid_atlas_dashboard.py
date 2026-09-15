"""Hereditary Progeroid & Premature-Aging Atlas — 8-Gene Segmental Progeroid Syndrome Reference
WRN-BLM-RECQL4-ERCC6-ERCC8-BANF1-TINF2-RTEL1
(Progeroid / RecQ helicase / Cockayne syndrome / Nuclear-lamina / Telomere-maintenance spectrum:
 Werner / Bloom / Rothmund-Thomson/Rapadilino / CS-B / CS-A / NGPS / Revesz / HHS)
320 patients (8 x 40), seeds 2758-2765.
Endpoints: /api/hereditary-progeroid-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "WRN",
        "seed_base": 2758,
        "protein": (
            "WRN -- 8p12 AR -- 1432aa -- Werner-Syndrome-Protein-"
            "162kDa-RecQ-Helicase-3prime5prime-Exonuclease-"
            "OMIM-Gene-604611-Disease-Werner-Syndrome-277700"
        ),
        "locus": "8p12",
        "protein_size": "1432 aa / 162 kDa (RecQ family helicase; 3'→5' DNA helicase + 3'→5' exonuclease; RQC + HRDC domains; nuclear localisation signal; interacts with p53/RPA/PCNA/BLM/telomere complex)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → Werner Syndrome — adult-onset segmental progeroid); "
            "BIALLELIC DISEASE (Werner Syndrome — WS / OMIM 277700): "
            "  Onset: late 2nd decade — bilateral cataracts FIRST SIGN most common; "
            "  Skin: premature wrinkling + scleroderma-like + ulcers over pressure points (especially Achilles region); "
            "  Metabolic: type 2 DM (insulin resistance) + dyslipidaemia + atherosclerosis (premature); "
            "  Gonadal atrophy: hypogonadism — infertility, irregular menstruation; "
            "  Short stature: truncal adiposity + 'bird-like' facies; "
            "  High-pitched hoarse voice: characteristic; "
            "  Cancer risk: MARKEDLY ELEVATED — mesenchymal tumours predominate: soft-tissue sarcoma, osteosarcoma, melanoma; "
            "  Also: thyroid carcinoma, meningioma, leukaemia; "
            "  Premature cardiovascular death: MI/stroke 3rd–4th decade; "
            "FOUNDER ALLELES: "
            "  Japanese: c.3139-1G>C (IVS25-1G>C) — most prevalent in Japan (founder); "
            "  Japanese: c.4197C>A (p.Cys1399Stop) — second most common Japan; "
            "  Sardinian: c.3546delA; "
            "MONOALLELIC RISK: "
            "  Heterozygous WRN variants: evidence for modestly elevated cancer risk — not guideline-level surveillance yet; "
            "EPIGENETIC ASPECT: WRN expression silencing by promoter methylation — acquired Werner-like accelerated aging in cancer"
        ),
        "disease_category": (
            "WERNER SYNDROME (WS) — ADULT-ONSET SEGMENTAL PROGERIA: "
            "DIAGNOSTIC CRITERIA (Oshima et al. / Japanese Ministry 2014): "
            "  CARDINAL (all 4 required for definite WS): "
            "    1. Cataracts (bilateral, premature) "
            "    2. Characteristic skin changes (scleroderma-like + ulcers + pigmentation) "
            "    3. Characteristic facies ('bird-like') + short stature "
            "    4. Premature greying / alopecia (2nd–3rd decade) "
            "  ADDITIONAL (≥2 support diagnosis): "
            "    T2DM; dyslipidaemia; hypogonadism; hoarse voice; family history; early-onset cancer; "
            "    Calcification of Achilles tendon / soft tissue; osteoporosis; "
            "HAEMATOLOGICAL MARKER: "
            "  SV40-immortalisation assay: WS fibroblasts immortalised at ~25% normal rate (WRN needed for telomere maintenance); "
            "  Chromosomal instability: variegated translocation mosaicism (VTM) — multiple translocations in different cells; "
            "    VTM = quasi-pathognomonic of WS + Bloom syndrome; "
            "CANCER SPECTRUM: "
            "  Mesenchymal > epithelial (inverted from general population) — KEY DDx marker; "
            "  Thyroid cancer: follicular carcinoma; meningioma; osteosarcoma; fibrosarcoma; acral lentiginous melanoma; "
            "  Leukaemia (ALL/AML) less common; "
            "  Total cancer risk: ~10x population (lifetime risk 40-60% depending on series); "
            "METABOLIC: "
            "  T2DM: pancreatic atrophy + peripheral insulin resistance; "
            "  Dyslipidaemia: TG elevated + HDL low → premature atherosclerosis; "
            "CARDIOVASCULAR: "
            "  Premature MI/stroke: median death age 48y (pre-treatment era); current management extending survival; "
            "MANAGEMENT: "
            "  Annual ophthalmic review (cataracts → cataract extraction); "
            "  DM surveillance + metformin; lipid management (statins); "
            "  Aggressive skin ulcer care (chronic non-healing ulcers); "
            "  Cancer surveillance: no established protocol — expert opinion varies; "
            "  Lonafarnib (farnesyltransferase inhibitor) — theoretical benefit; no RCT WS data; "
            "  Japanese WS Registry: most comprehensive data; national registry for clinical trials"
        ),
        "disease_pathway": (
            "WRN — RecQ HELICASE FUNCTIONS: "
            "REPLICATION: "
            "  WRN unwinds stalled replication forks (G4-quadruplexes, D-loops, holiday junctions); "
            "  WRN 3'→5' exonuclease degrades lagging-strand at stalled forks; "
            "  Interacts with PCNA/RFC/RPA at replication foci; "
            "  WRN LOF → replication fork collapse → DSBs → genomic instability → cancer + aging; "
            "TELOMERE MAINTENANCE: "
            "  WRN localises to telomeres during S-phase; "
            "  Unwinds telomere G-quadruplexes to allow lagging-strand synthesis; "
            "  WRN LOF → accelerated telomere shortening → replicative senescence → progeroid phenotype; "
            "  Telomere length inversely correlates with age of WS onset; "
            "RECOMBINATION: "
            "  WRN suppresses inappropriate RAD51-mediated recombination at stalled forks; "
            "  WRN LOF → variegated translocation mosaicism (VTM) — somatic translocations in multiple cells; "
            "  VTM explains cancer predisposition (chromosomal rearrangements); "
            "INTERACTION PARTNERS: "
            "  p53: WRN stabilises p53 → WRN LOF → reduced p53 checkpoint → mutagenic cells survive; "
            "  BLM: WRN+BLM cooperative at telomeres (both RecQ members); "
            "  TOPBP1/RPA: checkpoint mediators; "
            "  TRF2: shelterin component — WRN recruited to telomeres by TRF2; "
            "  Ku70/80: NHEJ interface (WRN may process DSBs before NHEJ ligation)"
        ),
        "pathognomonic": (
            "WERNER SYNDROME DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION: "
            "  Young adult (<40y) with bilateral cataracts + scleroderma-like skin + short stature + T2DM; "
            "  Achilles tendon calcification on X-ray — quasi-pathognomonic soft-tissue calcification site; "
            "CELLULAR TESTING: "
            "  Fibroblast growth: WS fibroblasts senesce at limited passage (passage 20 vs 60-80 normal); "
            "  SV40-immortalisation rate: ~25% that of normal fibroblasts; "
            "  VTM karyotype: multiple clonal chromosome translocations in different cells; "
            "MOLECULAR: "
            "  WRN gene sequencing (Sanger or NGS panel); "
            "  WRN protein expression (Western blot on fibroblasts/lymphoblasts); "
            "  Most pathogenic variants = truncating (frameshifts / nonsense); missense rare + require functional evidence; "
            "DDx FROM SIMILAR CONDITIONS: "
            "  Hutchinson-Gilford Progeria (HGPS/LMNA): childhood onset (1–2y), NO cataracts at onset, NO DM; "
            "  Rothmund-Thomson (RECQL4): childhood onset poikiloderma; no scleroderma; different cancer spectrum; "
            "  Mandibuloacral dysplasia: skeletal findings dominant; earlier onset; "
            "CRITICAL WARNINGS: "
            "  Radiation therapy: WRN fibroblasts RADIOSENSITIVE (less so than NHEJ-SCID but elevated); "
            "  Alkylating chemotherapy: WRN cells MORE SENSITIVE (impaired repair of crosslinks/DSBs); "
            "  Surgical procedures: poor wound healing — plan carefully; "
            "  Cancer: mesenchymal tumour = FIRST CONSIDERATION in WS; "
            "    Soft-tissue mass → biopsy; do not assume lipoma/benign"
        ),
        "clinical_variables": {
            "cataract_onset_age_y": (15, 30, "bilateral, premature — most common first sign of WS"),
            "cancer_lifetime_risk_pct": (40, 60, "mesenchymal > epithelial cancers"),
            "dm_prevalence_pct": (70, 80, "type 2 DM; insulin resistance + pancreatic atrophy"),
            "median_death_age_y": (46, 54, "premature cardiovascular death (MI/stroke); improving with management"),
            "achilles_calcification_pct": (60, 80, "soft-tissue calcification — key X-ray finding"),
            "vtm_karyotype_pct": (70, 90, "variegated translocation mosaicism — quasi-pathognomonic"),
        },
    },
    {
        "gene": "BLM",
        "seed_base": 2759,
        "protein": (
            "BLM -- 15q26.1 AR -- 1417aa -- Bloom-Syndrome-RecQ-Helicase-"
            "159kDa-DEAH-Box-3prime5prime-SCE-Suppressor-"
            "OMIM-Gene-604610-Disease-Bloom-Syndrome-210900"
        ),
        "locus": "15q26.1",
        "protein_size": "1417 aa / 159 kDa (RecQ family 3'→5' DNA helicase; DEAH box; RQC + HRDC domains; forms BTR complex with TOP3A-RMI1-RMI2; dissolves double Holliday junctions; suppresses SCE)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → Bloom Syndrome — early-onset growth retardation + cancer + sun sensitivity); "
            "BIALLELIC DISEASE (Bloom Syndrome — BS / OMIM 210900): "
            "  Growth: severe intrauterine + postnatal growth retardation — mean adult height 147cm male, 138cm female; "
            "  Skin: facial erythema (butterfly distribution) after sun exposure — telangiectatic erythema (NOT malar rash of lupus); "
            "  Immunodeficiency: mild to moderate — IgM+IgA reduced; recurrent otitis media; "
            "  Cancer risk: MASSIVELY ELEVATED — all cancer types; haematological + solid; "
            "  Fertility: female fertility retained (rare pregnancies reported); males: azoospermia/severe oligospermia; "
            "  Intelligence: normal in most; "
            "  Diabetes: T2DM develops in adulthood in significant proportion; "
            "ASHKENAZI JEWISH FOUNDER: "
            "  BLMAsh: c.2207_2212delATCTGAinsTAGATTC — 6-bp deletion-insertion in exon 10; "
            "  Carrier frequency: ~1 in 100 Ashkenazi Jewish; "
            "  If both partners Ashkenazi Jewish: recommend carrier testing pre-conception; "
            "DIAGNOSTIC GOLD STANDARD: "
            "  Sister Chromatid Exchange (SCE) rate: 10-fold elevation vs normal — PATHOGNOMONIC; "
            "  Normal SCE: 3–10/cell; BS: 40–100/cell (BrdU incorporation assay)"
        ),
        "disease_category": (
            "BLOOM SYNDROME — EARLY-ONSET GROWTH RETARDATION + EXTREME CANCER PREDISPOSITION: "
            "GROWTH PHENOTYPE: "
            "  Birth weight: markedly below normal — IUGR; "
            "  Length: severely reduced from birth; "
            "  Adult height: mean 147cm (M), 138cm (F) — proportionate short stature; "
            "  Weight: lean; no obesity; "
            "SKIN: "
            "  Facial erythema: butterfly pattern on cheeks/nose after sun exposure; onset 1st year of life; "
            "  Telangiectasias: in erythematous areas; "
            "  Café-au-lait spots: often multiple; "
            "  NOTE: DIFFERS from lupus — no systemic inflammation, no anti-dsDNA, no ANA; "
            "IMMUNODEFICIENCY: "
            "  Reduced serum IgM + IgA; IgG usually preserved; "
            "  Recurrent otitis media + respiratory tract infections; "
            "  Cellular immunity relatively preserved; "
            "CANCER: "
            "  All tumour types — haematological (ALL, AML, NHL, Hodgkin) + solid (CRC, breast, lung, etc.); "
            "  Onset: earlier than general population (childhood / 3rd–4th decade); "
            "  Lifetime cancer risk: approaches 100% if survival adequate; "
            "  Multiple primary cancers common; "
            "CELLULAR HALLMARK: "
            "  SCE elevation: 10x → 40–100 SCE/cell (key diagnostic biomarker + disease mechanism); "
            "  VTM: also present (like WS) — somatic chromosomal rearrangements in different lymphocyte clones; "
            "MANAGEMENT: "
            "  Sun avoidance + photoprotection (SPF50+); "
            "  Cancer surveillance: recommend annual full-body skin check + oncological review; "
            "  Chemotherapy: dose-reduce alkylating agents (elevated sensitivity); "
            "  Haematopoietic stem cell transplant: used for haematological malignancy — BUT BS cells radiosensitive → "
            "    avoid TBI conditioning; use busulfan-based or treosulfan-based RIC"
        ),
        "disease_pathway": (
            "BLM — BTR COMPLEX AND SCE SUPPRESSION: "
            "BTR COMPLEX FORMATION: "
            "  BLM + TOP3A + RMI1 + RMI2 = BTR (BLM-TOP3A-RMI1/2) complex; "
            "  BTR dissolves double Holliday junctions (dHJ) → non-crossover products; "
            "  Without BTR: dHJ resolved by resolvases (GEN1/MUS81) → crossover products → SCE; "
            "  BLM LOF → dHJ accumulation → excess crossover resolution → >10x SCE; "
            "REPLICATION FORK: "
            "  BLM unwinds stalled forks (G4 quadruplexes, D-loops); "
            "  BLM promotes fork restart after replication stress; "
            "  BLM LOF → fork collapse → DSBs → genomic instability; "
            "TELOMERE: "
            "  BLM localises to telomeres in S-phase; "
            "  BLM + WRN cooperate on telomere G-quadruplexes (both RecQ helicases); "
            "  BLM LOF → telomere fragility; "
            "RECOMBINATION REGULATION: "
            "  BLM disrupts RAD51 filaments at inappropriate sites → suppresses aberrant HR; "
            "  BLM-mediated dHJ dissolution = anti-crossover (pro-genome stability); "
            "CANCER MECHANISM: "
            "  SCE at oncogene/tumour suppressor loci → LOH → biallelic inactivation of TSG → tumorigenesis; "
            "  Excess crossovers at CpG islands → hypermutation (similar to defective MMR — microsatellite instability); "
            "  High base-substitution rate + elevated SCE = dual genomic instability drivers"
        ),
        "pathognomonic": (
            "BLOOM SYNDROME DIAGNOSTIC APPROACH: "
            "GOLD STANDARD TEST: "
            "  Sister chromatid exchange (SCE) rate in BrdU-pulsed lymphocytes/fibroblasts; "
            "  Method: grow cells in BrdU for 2 cell cycles → Hoechst-33258 + Giemsa staining → count SCE per cell; "
            "  Normal: 3–10 SCE/cell; "
            "  Bloom Syndrome: 40–100 SCE/cell (10-fold elevation) — PATHOGNOMONIC; "
            "  No other condition produces this level of SCE; "
            "CLINICAL CLUES: "
            "  Growth-retarded infant/child with facial erythema in butterfly pattern after sun; "
            "  Family history (consanguinity increases AR risk); "
            "  Ashkenazi ancestry: ALWAYS offer BLMAsh carrier testing to partner; "
            "MOLECULAR CONFIRMATION: "
            "  BLM gene sequencing (full gene); "
            "  BLMAsh c.2207_2212delATCTGAinsTAGATTC — targeted PCR/sequencing in Ashkenazi; "
            "DDx: "
            "  SLE: ANA/anti-dsDNA positive; systemic features; SCE normal; "
            "  Rothmund-Thomson (RECQL4): poikiloderma not butterfly; absent in early life; different cancer spectrum; "
            "  Xeroderma Pigmentosum (XP): sun sensitivity but neurological and skin tumours; no SCE elevation; "
            "DRUG SENSITIVITIES: "
            "  Alkylating agents: ELEVATED sensitivity (crosslink repair impaired); "
            "  Radiation: elevated sensitivity; "
            "  Mitomycin C: hypersensitivity (clastogenic) — use with caution in chemotherapy regimens"
        ),
        "clinical_variables": {
            "sce_per_cell": (40, 100, "sister chromatid exchanges — 10x normal — PATHOGNOMONIC"),
            "adult_height_male_cm": (140, 154, "severe proportionate short stature"),
            "cancer_lifetime_risk_pct": (85, 99, "all cancer types; multiple primaries"),
            "ashkenazi_carrier_freq": (0.8, 1.2, "percent; ~1 in 100 Ashkenazi Jewish"),
            "sun_erythema_onset_year": (1, 3, "years of age — butterfly facial erythema onset"),
            "azoospermia_male_pct": (80, 95, "severe male infertility (azoospermia)"),
        },
    },
    {
        "gene": "RECQL4",
        "seed_base": 2760,
        "protein": (
            "RECQL4 -- 8q24.12 AR -- 1208aa -- RecQ-Helicase-4-"
            "133kDa-DEAH-Box-Replication-Initiation-Okazaki-"
            "OMIM-Gene-603780-Disease-RTS2-268400-RAPADILINO-266280"
        ),
        "locus": "8q24.12",
        "protein_size": "1208 aa / 133 kDa (RecQ family helicase; N-terminal Sld2-homologous domain initiates replication; 3'→5' helicase activity; interacts with DNA Pol-alpha/MCM/p68; mitochondrial localisation signal)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — THREE ALLELIC DISORDERS depending on mutation type: "
            "1. ROTHMUND-THOMSON SYNDROME TYPE 2 (RTS2 / OMIM 268400): "
            "   Complete or near-complete LOF (truncating + nonsense): "
            "   Poikiloderma: congenital/infantile skin rash (erythema → atrophy + pigmentation + telangiectasia) — CARDINAL; "
            "   Skeletal: short stature + radial ray defects (aplasia/hypoplasia); "
            "   Cancer: OSTEOSARCOMA — very high risk (30-50% lifetime) — CRITICAL CLINICAL FLAG; "
            "   Hair: sparse/absent eyebrows and lashes; "
            "   Cataracts: juvenile bilateral — distinct from Werner; "
            "2. RAPADILINO SYNDROME (OMIM 266280): "
            "   Specific Finnish founder allele: c.1390+2T>C (IVS14+2T>C) in-frame splicing; "
            "   Radial ray + patella aplasia + cleft/arched palate + diarrhoea + dislocations + low birth weight + normal cognition; "
            "   RAPADILINO = RA-PA-DI-LI-NO: RAdialhypoplasia+Patella/palate-hypoplasia+DIarrhoea+LIttle size+NOrmal intelligence; "
            "   Osteosarcoma risk: also elevated (lower than RTS2 but present); "
            "3. BALLER-GEROLD SYNDROME (OMIM 218600): "
            "   Specific variant pattern; "
            "   Craniosynostosis + radial aplasia; "
            "   OVERLAPS with FGFR2/3 Baller-Gerold — molecular testing required; "
            "FOUNDER ALLELE RAPADILINO: "
            "  Finnish population enrichment; c.1390+2T>C splice-site"
        ),
        "disease_category": (
            "RECQL4 SPECTRUM — RTS2 / RAPADILINO / BALLER-GEROLD: "
            "POIKILODERMA (RTS2 CARDINAL): "
            "  Onset: 3–6 months of life — erythema (cheeks first) → blistering → atrophic skin + pigmentation + telangiectasia; "
            "  Distribution: face → extremities → spares trunk; "
            "  PRECEDES cancer by decades — essential screening marker; "
            "  Photosensitivity: present but variable; "
            "OSTEOSARCOMA — CRITICAL CANCER WARNING: "
            "  Risk: 30-50% lifetime in RTS2 (vs 0.03% general population); "
            "  Onset: childhood/adolescence (peak 5–20 years); "
            "  Location: extremity long bones (distal femur, proximal tibia); "
            "  Surveillance: annual X-ray of extremities from age 5; bone pain = URGENT evaluation; "
            "  Chemotherapy sensitivity: standard osteosarcoma regimens used; "
            "    BUT: cisplatin sensitivity may be elevated (RECQL4 involved in cisplatin repair); "
            "OTHER CANCER: "
            "  Non-melanoma skin cancer: elevated (UV-related); "
            "  MDS/leukaemia: lower risk than osteosarcoma; "
            "SKELETAL: "
            "  Radial ray defects (RTS2 + Baller-Gerold): absent/hypoplastic radius + thumb; "
            "  Short stature: variable; "
            "  Osteopenia: common; "
            "REPLICATION DEFECT: "
            "  RECQL4 N-terminus (Sld2-homologous) initiates DNA replication at origins; "
            "  RECQL4 LOF → impaired firing of replication origins → genomic instability; "
            "  Osteosarcoma may arise from replication stress in rapidly dividing osteoblasts; "
            "CLINICAL MANAGEMENT: "
            "  Osteosarcoma surveillance protocol: mandatory, annual imaging from age 5; "
            "  Skin protection: SPF50+ lifelong; annual dermatological review; "
            "  Ophthalmic review: cataracts (juvenile bilateral in RTS2); "
            "  Skeletal monitoring: radiology for scoliosis, osteoporosis"
        ),
        "disease_pathway": (
            "RECQL4 — REPLICATION INITIATION + GENOME STABILITY: "
            "REPLICATION INITIATION (N-TERMINAL Sld2-DOMAIN): "
            "  RECQL4 N-terminal domain (aa 1-240) is homologous to yeast Sld2/human RecQL4-NDT; "
            "  Sld2 (yeast) is essential for firing of replication origins in S-phase; "
            "  RECQL4 interacts with RPA/MCM complex at replication origins; "
            "  Recruited to origins via interaction with DNA Pol-alpha-primase; "
            "  RECQL4 LOF → impaired origin firing → replication stress → genomic instability; "
            "HELICASE DOMAIN (CENTRAL): "
            "  3'→5' RecQ-type helicase; "
            "  Unwinds D-loops, G4 structures, fork structures; "
            "  Less efficient than BLM/WRN at junction dissolution; "
            "  May collaborate with RPA to stabilise unwound DNA at replication forks; "
            "MITOCHONDRIAL FUNCTION: "
            "  RECQL4 has mitochondrial targeting sequence; localises to mitochondria; "
            "  Maintains mitochondrial DNA integrity; "
            "  RECQL4 LOF → mitochondrial genomic instability → contributes to aging phenotype; "
            "CANCER MECHANISM (OSTEOSARCOMA BIAS): "
            "  Osteoblasts require high replication → most sensitive to RECQL4 LOF; "
            "  Replication origin firing impaired → osteoblast genomic instability; "
            "  TP53 and RB1 LOH secondary events in RTS-associated osteosarcoma; "
            "  Cisplatin may be less effective if RECQL4 is part of cisplatin-repair machinery"
        ),
        "pathognomonic": (
            "RECQL4/RTS2 DIAGNOSTIC APPROACH: "
            "CLINICAL RECOGNITION: "
            "  Infant with poikiloderma (erythema → atrophy/pigmentation/telangiectasia) onset 3-6 months; "
            "  Cheeks → extremities distribution; "
            "  Radial ray defect + short stature: raises RTS2 immediately; "
            "DIFFERENTIAL DIAGNOSIS: "
            "  Dyskeratosis Congenita (DC): nail dystrophy triad + mucosal leucoplakia + BMF — ABSENT in RTS; "
            "  Fanconi Anaemia: DEB test + BMF — ABSENT in RTS; "
            "  Bloom Syndrome (BLM): butterfly facial erythema vs poikiloderma; SCE elevated in BLM not RTS; "
            "  IBIDS (Ichthyosis + BIDS): different distribution; TTD overlap? — gene sequencing resolves; "
            "MOLECULAR TESTING: "
            "  RECQL4 gene sequencing (NGS panel or full gene); "
            "  Target Finnish founder: c.1390+2T>C (RAPADILINO); "
            "  Protein: RECQL4 Western blot on fibroblasts; "
            "OSTEOSARCOMA SURVEILLANCE PROTOCOL: "
            "  Baseline: X-rays of extremities at diagnosis; "
            "  Annual: X-rays (tibia/fibula/femur/humerus) from age 5; "
            "  MRI if X-ray equivocal or bone pain; "
            "  Biopsy any periosteal reaction; "
            "  URGENT bone pain workup — osteosarcoma presents with localised bone pain"
        ),
        "clinical_variables": {
            "osteosarcoma_lifetime_risk_pct": (30, 50, "CRITICAL — very high osteosarcoma risk in RTS2"),
            "poikiloderma_onset_month": (3, 6, "months of life — congenital/infantile skin rash"),
            "radial_aplasia_pct": (40, 60, "radial ray defect (RTS2 + Baller-Gerold)"),
            "juvenile_cataract_pct": (10, 30, "bilateral juvenile cataracts in RTS2"),
            "skin_cancer_lifetime_pct": (15, 30, "non-melanoma skin cancer; UV-related"),
            "short_stature_pct": (80, 95, "proportionate short stature"),
        },
    },
    {
        "gene": "ERCC6",
        "seed_base": 2761,
        "protein": (
            "ERCC6 -- 10q11.23 AR -- 1493aa -- CSB-Cockayne-Syndrome-B-"
            "168kDa-SWI2-SNF2-Chromatin-Remodeller-TC-NER-Coupling-Factor-"
            "OMIM-Gene-609413-Disease-CS-B-133540"
        ),
        "locus": "10q11.23",
        "protein_size": "1493 aa / 168 kDa (SWI2/SNF2 ATPase/helicase; chromatin remodeller; TC-NER coupling factor; recruits XPA/TFIIH to stalled RNAPII; interacts with CSA/DDB1/p53/nucleosome; UVSS allelic with missense)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — MOST COMMON COCKAYNE SYNDROME GENE (~65% of CS): "
            "BIALLELIC DISEASE (Cockayne Syndrome B — CS-B / OMIM 133540): "
            "  MOST SEVERE classical CS group — earlier onset vs CS-A; "
            "  Neurological: progressive neurodegeneration + intellectual disability; "
            "  Growth: cachexia + growth failure + 'wizened old' appearance in children; "
            "  Photosensitivity: severe sun sensitivity WITHOUT skin cancer; "
            "  Facies: sunken eyes + large ears + beaked nose + microcephaly; "
            "  Hearing: progressive SNHL (cochlear); "
            "  Eyes: pigmentary retinopathy + cataracts + optic atrophy; "
            "  Demyelination: CNS white matter + PNS; "
            "  Basal ganglia: calcifications (CT head); "
            "  Dental: enamel hypoplasia; "
            "  Lifespan: SEVERELY reduced — mean death ~12–16 years in classic CS-B; "
            "ALLELIC MILDER FORM (UVSS — UV-Sensitive Syndrome): "
            "  Missense/hypomorphic ERCC6 alleles → UVSS: sun sensitivity ONLY — no neurodegeneration, no growth failure; "
            "  Survival normal; "
            "  UVSS-A (ERCC6) vs UVSS-B (ERCC8) — same clinical outcome; "
            "NEONATAL CS / CS TYPE III (CSNBF — Cerebro-Oculo-Facio-Skeletal): "
            "  Complete null ERCC6 → COFS syndrome: severe prenatal onset; "
            "  Microcephaly + cataracts + hip contractures + calcification — lethal before 2y; "
            "CRITICAL ABSENT FEATURE: "
            "  NO SKIN CANCER despite sun sensitivity — PATHOGNOMONIC distinction from XP; "
            "  XP has skin cancer (UDS-reduced); CS has premature aging without skin cancer"
        ),
        "disease_category": (
            "COCKAYNE SYNDROME B — TC-NER DEFECT WITHOUT CANCER: "
            "CLINICAL TRIAD (classic): "
            "  1. Sun sensitivity (severe) without skin cancer "
            "  2. Progressive neurological deterioration "
            "  3. Growth failure (cachexia + short stature) "
            "NEUROLOGICAL: "
            "  Progressive: starts with developmental regression 1–2y; "
            "  Cerebellar ataxia; spasticity; peripheral neuropathy; "
            "  MRI: hypomyelination + progressive atrophy (especially cerebellar); "
            "    Calcification of basal ganglia + periventricular regions (CT); "
            "  Hearing: cochlear SNHL (progressive); "
            "GROWTH: "
            "  Severe failure to thrive; cachectic habitus; "
            "  Weight disproportionately reduced vs height; "
            "  Feeding difficulties (poor suck, swallowing); gastrostomy often required; "
            "OPHTHALMOLOGICAL: "
            "  Cataracts: bilateral progressive; "
            "  Pigmentary retinopathy: salt-and-pepper; "
            "  Nystagmus; strabismus; "
            "  Reduced lacrimation; photophobia; "
            "SKIN: "
            "  Sun sensitivity: erythema + blistering after minimal UV; "
            "  NO skin cancer — absent (DDx from XP); "
            "  Lentigines occasionally; "
            "DENTAL: "
            "  Dental caries: severe + early; enamel hypoplasia; "
            "MANAGEMENT: "
            "  No disease-modifying therapy; supportive; "
            "  Sun protection: SPF50+ mandatory — prevents ulceration even without cancer risk; "
            "  Nutrition: gastrostomy early for weight gain; "
            "  Hearing aids; "
            "  Physiotherapy (spasticity management); "
            "  Ophthalmic: cataract surgery when vision-limiting; "
            "  Palliative/hospice as disease progresses (median death ~12–16y classic CS-B)"
        ),
        "disease_pathway": (
            "ERCC6/CSB — TC-NER COUPLING FACTOR: "
            "TC-NER INITIATION: "
            "  RNA Pol II stalls at DNA lesion (CPD, 6-4PP, oxidative damage) during transcription; "
            "  Stalled RNAPII → CSB recruited; "
            "  CSB ATPase remodels chromatin around stalled RNAPII; "
            "  CSB recruits CRL4(CSA) E3 ligase complex (ERCC8 = CSA subunit); "
            "  CRL4(CSA) ubiquitylates CSB → CSB degraded → RNAPII backtracked; "
            "  CSA recruits XPA, TFIIH, XPG → assembly of TC-NER pre-incision complex; "
            "  CSB LOF → RNAPII remains stalled → transcription block → apoptosis of post-mitotic neurons; "
            "GGR vs TC-NER: "
            "  GGR (global genome repair): DDB2/XPC sense damage anywhere → XPA/TFIIH; "
            "  TC-NER: CSB/CSA recognise transcription block → XPA/TFIIH; "
            "  CS patients: GGR INTACT (XPC/DDB2 normal) → cancer-protective; "
            "  CS patients: TC-NER deficient → transcription-arrested neurons → neurodegeneration; "
            "  XP: GGR ABSENT → skin cancer accumulates; "
            "  XP+CS overlap (ERCC2/3/5 null): both pathways impaired → XP skin cancer + CS neurodegeneration; "
            "OXIDATIVE DAMAGE REPAIR: "
            "  CSB also involved in base excision repair of oxidative lesions (8-oxoG) in actively transcribed genes; "
            "  CSB recruits OGG1/APE1 to oxidatively damaged transcribed strands; "
            "  This dual role (TC-NER + BER) explains why CS patients have multi-system oxidative damage features; "
            "PREMATURE AGING MECHANISM: "
            "  Persistent transcription-blocking lesions in non-dividing neurons → apoptosis; "
            "  Accumulation of unrepaired oxidative DNA damage → accelerated cellular aging; "
            "  Mitochondrial dysfunction: CSB localises to mitochondria; mitochondrial TC-NER role emerging"
        ),
        "pathognomonic": (
            "COCKAYNE SYNDROME B DIAGNOSTIC APPROACH: "
            "CLINICAL RECOGNITION: "
            "  Child with progressive neurodegeneration + cachexia + severe sun sensitivity WITHOUT skin cancer; "
            "  'Cachectic dwarfism' with large ears + beaked nose + sunken eyes; "
            "  Baseline CT head: calcifications in basal ganglia / periventricular (CS-B earlier than CS-A); "
            "CELLULAR ASSAYS: "
            "  UDS (Unscheduled DNA Synthesis) after UV: NORMAL in CS (unlike XP); "
            "    CS patients repair UVC-induced damage globally (GGR intact); "
            "  RNA synthesis recovery after UV (RRS): MARKEDLY REDUCED — GOLD STANDARD; "
            "    Normal: RNA synthesis recovers to ~80% within 24h after UV; "
            "    CS: <25% recovery — persistent transcription block; "
            "  Colony survival after UV: reduced vs control; "
            "MOLECULAR: "
            "  ERCC6 gene sequencing; Western blot CSB protein; "
            "  Note: ERCC6 is a LARGE gene (1493aa) — full sequencing required; "
            "DDx FROM XP: "
            "  SKIN CANCER ABSENT in CS → GGR intact → NER repairs non-transcribed strand normally; "
            "  XP: skin cancer present; UDS markedly reduced; no neurodegeneration in XP-C/E/V; "
            "  XP-A/D/F/G: neurodegeneration + skin cancer (both pathways impaired); "
            "    These have 'XP+CS' overlap if GGR and TC-NER both affected; "
            "DDx FROM RETT SYNDROME: "
            "  Rett: female predominance; MECP2 mutation; no sun sensitivity; "
            "ABSOLUTELY NO SKIN CANCER: "
            "  If CS patient develops skin cancer → review diagnosis → may be XP+CS overlap → re-sequence ERCC2/ERCC5"
        ),
        "clinical_variables": {
            "median_death_age_y": (12, 16, "severely reduced lifespan in classic CS-B"),
            "uvss_alleles_milder_pct": (10, 20, "UVSS (UV-Sensitive Syndrome) from mild alleles — normal lifespan"),
            "snhl_prevalence_pct": (70, 90, "progressive sensorineural hearing loss"),
            "cataract_prevalence_pct": (50, 70, "bilateral progressive cataracts"),
            "basal_ganglia_calcification_pct": (60, 80, "CT finding in CS-B"),
            "gastrostomy_requirement_pct": (40, 60, "due to severe feeding difficulties/cachexia"),
        },
    },
    {
        "gene": "ERCC8",
        "seed_base": 2762,
        "protein": (
            "ERCC8 -- 5q12.1 AR -- 396aa -- CSA-CRL4-WD40-Repeat-Substrate-Adapter-"
            "44kDa-CSB-Ubiquitylation-DCAF-TC-NER-"
            "OMIM-Gene-609412-Disease-CS-A-216400"
        ),
        "locus": "5q12.1",
        "protein_size": "396 aa / 44 kDa (WD40-repeat β-propeller; DCAF substrate-receptor of CRL4-DDB1 E3 ligase; ubiquitylates CSB/RNAPII to permit TC-NER assembly; p44 complex with DDB1/CUL4A; NAE-1 regulator)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — 2nd MOST COMMON COCKAYNE SYNDROME GENE (~35% of CS): "
            "BIALLELIC DISEASE (Cockayne Syndrome A — CS-A / OMIM 216400): "
            "  MILDER than CS-B overall — later onset + slower progression: "
            "  Same cardinal features as CS-B but variable severity: "
            "  Sun sensitivity without skin cancer; "
            "  Neurodegeneration (progressive): later onset vs CS-B (often 1st year vs 3–5y); "
            "  Growth failure: cachectic habitus; "
            "  Ophthalmological: cataracts + retinopathy + nystagmus; "
            "  SNHL: progressive; "
            "  Dental: caries + enamel hypoplasia; "
            "  Lifespan: mean ~16–25 years (range wide — some mild into 4th decade); "
            "UVSS (UV-SENSITIVE SYNDROME — ALLELIC): "
            "  Missense/mild LOF ERCC8 → UVSS-B: sun sensitivity ONLY; normal neurological/cognitive development; "
            "  Normal lifespan; "
            "  Clinically indistinguishable from UVSS-A (ERCC6) — molecular testing required; "
            "COMPARISON CS-A vs CS-B: "
            "  CS-B: more severe → ERCC6 null = earlier onset, worse neurodegeneration; "
            "  CS-A: relatively milder → some reach young adulthood; "
            "  Both: NO SKIN CANCER (GGR intact in both → they differ from XP); "
            "  Both: same cellular assay pattern (RRS reduced; UDS normal)"
        ),
        "disease_category": (
            "COCKAYNE SYNDROME A — TC-NER DEFECT: "
            "SEVERITY COMPARISON TO CS-B: "
            "  CS-A: milder overall; later onset; slower neurodegeneration vs CS-B; "
            "  CS-A vs CS-B cellular: both have reduced RRS, normal UDS — assays cannot distinguish; "
            "  CS-A vs CS-B molecular: ERCC8 vs ERCC6 gene sequencing required; "
            "CLINICAL FEATURES (same spectrum as CS-B, milder): "
            "  Sun sensitivity: present — avoid UV; SPF50+ mandatory; "
            "  Neurodegeneration: progressive cerebellar ataxia + spasticity; often slower than CS-B; "
            "  Growth: failure to thrive; cachectic; gastrostomy often required; "
            "  Eyes: cataracts + pigmentary retinopathy; "
            "  Hearing: SNHL; "
            "  Dental: severe caries; "
            "ERCC8 MOLECULAR SPECTRUM: "
            "  Classic CS-A: biallelic truncating/splice — severe; "
            "  Milder CS-A: at least one missense (residual protein) — delayed onset; "
            "  UVSS-B: specific missense maintaining DDB1 binding → sun sensitivity only; "
            "  Note: genotype–phenotype correlation imperfect — modifier genes influence outcome; "
            "MANAGEMENT (same principles as CS-B): "
            "  Sun protection (SPF50+ mandatory); "
            "  Nutrition (gastrostomy for severe feeding problems); "
            "  SNHL (hearing aids); "
            "  Cataract surgery (when vision-limiting); "
            "  No disease-modifying therapy approved; "
            "  NAD+ supplementation (preclinical evidence from CS mouse models — phase I pending); "
            "  Physiotherapy + spasticity management"
        ),
        "disease_pathway": (
            "ERCC8/CSA — CRL4 E3 UBIQUITIN LIGASE SUBSTRATE RECEPTOR: "
            "CRL4-CSA COMPLEX ASSEMBLY: "
            "  CSA (ERCC8) = WD40-repeat DCAF substrate receptor; "
            "  Assembles: CUL4A + RBX1 + DDB1 + CSA → CRL4(CSA) E3 ubiquitin ligase; "
            "  NAE-1/UBA3 neddylates CUL4A → activates CRL4(CSA); "
            "  DDB1 bridges CSA to CUL4A; "
            "TC-NER ROLE: "
            "  CSB stalls at transcription-blocking lesion → recruits CRL4(CSA); "
            "  CRL4(CSA) ubiquitylates CSB → CSB proteasomal degradation; "
            "  CSB removal allows RNAPII to backtrack and XPA/TFIIH/XPG/XPF to assemble; "
            "  CSA ALSO ubiquitylates RNAPII (largest subunit RPB1) → enables lesion access; "
            "  CSA LOF → CSB not ubiquitylated → CSB blocks NER complex assembly → TC-NER fails; "
            "SEPARATION OF GGR AND TC-NER: "
            "  GGR: DDB2(XPE) + XPC sense lesion in non-transcribed DNA → XPA/TFIIH → excision; "
            "    This pathway UNAFFECTED in CS-A/B (DDB2/XPC normal); "
            "    Therefore: CANCER ABSENT (GGR repairs mutagenic UV photoproducts in skin); "
            "  TC-NER: CSB+CSA required → absent in CS → persistent transcription blocks in neurons; "
            "PREMATURE AGING: "
            "  Post-mitotic neurons accumulate unrepaired transcription-blocking oxidative lesions; "
            "  Apoptosis of neurons → progressive atrophy; "
            "  Systemic features (growth failure): high metabolic demand tissues impaired by oxidative stress + apoptosis"
        ),
        "pathognomonic": (
            "COCKAYNE SYNDROME A DIAGNOSTIC APPROACH: "
            "CLINICAL RECOGNITION: "
            "  Child with sun sensitivity + progressive neurodegeneration + cachexia WITHOUT skin cancer; "
            "  Later onset than CS-B (onset often 1–3 years vs CS-B first year); "
            "  'Bird-like' facies; large ears; sunken eyes; "
            "CELLULAR ASSAYS: "
            "  UDS: NORMAL (GGR intact — unlike XP); "
            "  RRS: MARKEDLY REDUCED — GOLD STANDARD for CS diagnosis (both CS-A and CS-B); "
            "    Method: pulse-label RNA with 14C-uridine or BrdU after UV; measure recovery vs control; "
            "    CS: <25% recovery at 24h post-UV (normal >80%); "
            "  This assay distinguishes CS from XP (UDS + RRS) and from healthy (RRS reduced); "
            "MOLECULAR TESTING: "
            "  ERCC8 gene sequencing (panel or full gene); "
            "  Distinguish CS-A (ERCC8) from CS-B (ERCC6); "
            "  Both have identical cellular assay profile → gene sequencing mandatory for gene identification; "
            "DDx CS-A vs CS-B: "
            "  No reliable clinical distinction — both have sun sensitivity + neurodegeneration; "
            "  CS-A: milder, later onset — clinical impression only; "
            "  Molecular: ERCC8 (CS-A) vs ERCC6 (CS-B); "
            "UVSS-B (ERCC8 missense): "
            "  Sun sensitivity ONLY; no neurodegeneration; normal lifespan; "
            "  RRS: slightly reduced but better recovery than classic CS; "
            "  UDS: normal; "
            "  Molecular confirmation required to distinguish from UVSS-A (ERCC6)"
        ),
        "clinical_variables": {
            "median_death_age_y": (16, 25, "milder than CS-B; range wide (some into 4th decade)"),
            "uvss_b_alleles_pct": (10, 20, "UVSS-B from missense alleles — sun sensitivity only"),
            "snhl_prevalence_pct": (65, 85, "progressive sensorineural hearing loss"),
            "cataract_prevalence_pct": (45, 65, "bilateral progressive cataracts"),
            "rrs_recovery_pct": (10, 25, "RNA synthesis recovery after UV (gold standard — normal >80%)"),
            "uds_normal_vs_xp": (95, 100, "UDS normal — GGR intact (key DDx from XP)"),
        },
    },
    {
        "gene": "BANF1",
        "seed_base": 2763,
        "protein": (
            "BANF1 -- 11q13.1 AR -- 89aa -- BAF-Barrier-To-Autointegration-Factor-"
            "10kDa-Nuclear-Lamina-Adapter-LEM-Domain-Interactor-"
            "OMIM-Gene-603811-Disease-NGPS-614008"
        ),
        "locus": "11q13.1",
        "protein_size": "89 aa / 10 kDa (barrier-to-autointegration factor; homodimer; bridges LEM-domain proteins of nuclear lamina to chromatin via histone H3/H4 binding; phosphorylated by VRK1/PP2A; phosphorylation releases nuclear lamina)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → Nestor-Guillermo Progeria Syndrome — NGPS): "
            "BIALLELIC DISEASE (Nestor-Guillermo Progeria Syndrome / OMIM 614008): "
            "  ULTRA-RARE (fewer than 10 published patients globally 2024); "
            "  Clinically similar to Hutchinson-Gilford Progeria (HGPS/LMNA-G608G): "
            "  Progeroid features: loss of subcutaneous fat + wrinkled aged skin + short stature + alopecia; "
            "  Severe osteoporosis / osteolysis: distinctive skeletal feature vs HGPS; "
            "  Clavicular resorption: frequently; "
            "  Normal or mildly elevated cardiovascular risk (LESS than HGPS — atherosclerosis milder); "
            "  Normal intelligence; "
            "  Onset: 1st year of life — rapidly progressive; "
            "KEY DISTINCTION FROM HGPS (LMNA): "
            "  NGPS/BANF1: LMNA gene NORMAL; LMNA protein NORMAL; no progerin; "
            "  HGPS/LMNA: c.1824C>T (p.G608G) splicing → 50aa truncated progerin; "
            "  NGPS: cardiac disease MILDER than HGPS; skeletal changes MORE PROMINENT; "
            "  Both: alopecia + loss of subcutaneous fat; "
            "PATHOGENIC VARIANT: "
            "  p.Ala12Thr (c.34G>A): MOST COMMON / founding variant in NGPS patients; "
            "  Alanine 12 → polar Thr disrupts BAF homodimer interface; impairs LEM-domain binding; "
            "  Most NGPS patients are homozygous Ala12Thr; "
            "CELLULAR PHENOTYPE: "
            "  Nuclear blebbing similar to HGPS; "
            "  Abnormal heterochromatin organisation; "
            "  Reduced H3K27me3 / H3K9me3 (LAD heterochromatin lost)"
        ),
        "disease_category": (
            "NESTOR-GUILLERMO PROGERIA SYNDROME (NGPS) — NUCLEAR LAMINA BRIDGING DEFECT: "
            "CLINICAL COMPARISON WITH HGPS: "
            "  SIMILARITY: alopecia + loss of subcutaneous fat + aged skin appearance + short stature; "
            "  DIFFERENCE: NGPS has MORE skeletal pathology (severe osteoporosis, clavicular resorption, pathological fractures); "
            "  DIFFERENCE: NGPS cardiovascular disease LESS SEVERE than HGPS (no premature massive atherosclerosis); "
            "  DIFFERENCE: BANF1 mutation vs LMNA c.1824C>T; "
            "  DIFFERENCE: NGPS lifespan potentially slightly longer than HGPS (HGPS: mean 13.4y; NGPS: insufficient data but some survive into 2nd decade); "
            "SKELETAL PHENOTYPE (KEY): "
            "  Severe osteoporosis: bone mineral density markedly reduced from childhood; "
            "  Clavicular resorption: distinctive — may be partial or complete; "
            "  Pathological fractures: low-trauma fractures of long bones/vertebrae; "
            "  Acro-osteolysis: resorption of distal phalanges (similar to HGPS); "
            "  Scoliosis: common; "
            "  Craniofacial: micrognathia + prominent calvarium + delayed suture closure; "
            "SKIN: "
            "  Loss of subcutaneous fat: generalised; "
            "  Aged wrinkled skin; "
            "  Alopecia (complete/near-complete); "
            "  Scleroderma-like: skin tightening over digits/face; "
            "CARDIOVASCULAR: "
            "  Less severe than HGPS; "
            "  Some atherosclerosis reported but not the universal fatal course of HGPS at 13y; "
            "MOLECULAR BASIS: "
            "  BAF bridges LEM-domain nuclear lamina proteins (emerin, LAP2, MAN1) to chromatin; "
            "  BAF Ala12Thr → impaired homodimer → impaired LEM binding → nuclear lamina detachment from chromatin; "
            "  Consequence: lamina-associated domain (LAD) loss → heterochromatin disorganisation → premature senescence"
        ),
        "disease_pathway": (
            "BANF1/BAF — NUCLEAR LAMINA-CHROMATIN BRIDGING: "
            "BAF STRUCTURE: "
            "  89aa; homodimer in solution; beta-sheet core + C-terminal helix; "
            "  Two DNA-binding surfaces on homodimer — wraps around dsDNA; "
            "  VRK1/VRK2 phosphorylation (Ser4) → releases BAF from chromatin at mitosis onset; "
            "  PP2A dephosphorylation → BAF re-associates at telophase for nuclear lamina reassembly; "
            "LEM-DOMAIN BINDING: "
            "  BAF binds LEM domain (Lap2-Emerin-MAN1 shared domain) of nuclear envelope proteins; "
            "  LEM proteins: emerin (EMD), LAP2alpha/beta, MAN1 (LEMD3), ankle1/2; "
            "  BAF bridges: chromatin — BAF — LEM protein — nuclear lamina (lamin A/B); "
            "  This tethering maintains heterochromatin at nuclear periphery (LADs); "
            "BAF IN NUCLEAR ASSEMBLY: "
            "  At anaphase: BAF recruits LEM proteins to chromosomes → initiates nuclear membrane reassembly; "
            "  Without BAF: nuclear membrane fails to reassemble properly → chromosomal bridges; "
            "NGPS MECHANISM: "
            "  BAF Ala12Thr → impaired homodimer interface → reduced LEM binding affinity; "
            "  Nuclear lamina detachment from chromatin → LAD disorganisation; "
            "  H3K9me3/H3K27me3 (heterochromatin marks) lost at LADs → euchromatic shift; "
            "  Premature nuclear lamina destabilisation → phenotype mirroring lamin A/progerin accumulation; "
            "CONNECTION TO LMNA/PROGERIN: "
            "  HGPS progerin: farnesylated permanently → binds nuclear lamina → destabilises; "
            "  NGPS/BAF: reduced chromatin tethering → also destabilises nuclear architecture; "
            "  Both converge on LAD loss → heterochromatin reorganisation → premature aging"
        ),
        "pathognomonic": (
            "NGPS (BANF1) DIAGNOSTIC APPROACH: "
            "CLINICAL RECOGNITION: "
            "  Child with progeroid appearance (loss of subcutaneous fat + alopecia + aged skin) + prominent skeletal disease; "
            "  Clavicular resorption on CXR: DISTINCTIVE — raises NGPS above HGPS; "
            "  Osteoporosis with pathological fractures + acro-osteolysis; "
            "  LMNA gene: NORMAL (key DDx from HGPS — always sequence LMNA first); "
            "MOLECULAR TESTING: "
            "  BANF1 gene sequencing; target p.Ala12Thr (c.34G>A) first; "
            "  Progeroid gene panel: LMNA / BANF1 / ZMPSTE24 / WRN / RECQL4; "
            "CELLULAR ASSAYS: "
            "  Nuclear morphology: nuclear blebbing + nuclear envelope irregularity (similar to HGPS); "
            "  Heterochromatin markers: H3K9me3 / H3K27me3 reduced at nuclear periphery; "
            "DDx: "
            "  HGPS (LMNA G608G): cardiovascular much more severe; skeletal less prominent; LMNA abnormal; "
            "  Mandibuloacral Dysplasia (LMNA/ZMPSTE24): mandibular hypoplasia prominent; acral osteolysis; "
            "  Werner Syndrome (WRN): adult onset; DM+cataracts dominant; "
            "MANAGEMENT: "
            "  Bone health: calcium + vitamin D + bisphosphonates for osteoporosis; fracture prevention; "
            "  Cardiovascular monitoring: echo + ECG (less aggressive than HGPS protocol but still annual); "
            "  Lonafarnib: HGPS-approved (for LMNA/progerin); THEORETICAL role in NGPS (lamin-adjacent pathway) — off-label only; "
            "  Skin protection; physiotherapy; nutritional support"
        ),
        "clinical_variables": {
            "global_cases_total": (5, 12, "total published cases 2024 — ultra-rare"),
            "ala12thr_variant_pct": (80, 100, "p.Ala12Thr — most common/founding variant"),
            "clavicular_resorption_pct": (60, 90, "distinctive skeletal finding — DDx from HGPS"),
            "osteoporosis_severity": (70, 95, "severe early-onset osteoporosis + fractures"),
            "cardiovascular_severity_vs_hgps": (20, 40, "cardiovascular risk fraction compared to HGPS baseline"),
            "nuclear_blebbing_pct": (90, 100, "nuclear morphology abnormality in patient fibroblasts"),
        },
    },
    {
        "gene": "TINF2",
        "seed_base": 2764,
        "protein": (
            "TINF2 -- 14q12 AD-de-novo AR -- 354aa -- TIN2-Shelterin-Component-"
            "40kDa-TRF1-TRF2-TPP1-Bridge-OMIM-Gene-604319-"
            "Disease-DC2-613989-Revesz-616789"
        ),
        "locus": "14q12",
        "protein_size": "354 aa / 40 kDa (shelterin telomere-protection complex core component; bridges TRF1-TRF2 on telomere dsDNA to TPP1-POT1 on ssDNA overhang; recruits HP1gamma; interacts with PCNA via PIP-box; regulates telomere length)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (heterozygous de novo / familial): "
            "TINF2 DISEASE SPECTRUM: "
            "1. DYSKERATOSIS CONGENITA TYPE 3 (DC2 / OMIM 613989): "
            "   Moderate-severe DC triad + bone marrow failure + lung fibrosis; "
            "   Onset: childhood; "
            "   Classic DC triad: abnormal nail dystrophy + oral leucoplakia + skin reticulate pigmentation; "
            "   Bone marrow failure (BMF): progressive pancytopenia; "
            "   IPF / lung fibrosis: pulmonary complications; "
            "2. REVESZ SYNDROME (OMIM 268130 / 616789): "
            "   SEVERE end of spectrum — earlier onset + bilateral exudative retinopathy (PATHOGNOMONIC); "
            "   Bilateral exudative retinopathy: diagnostic hallmark of Revesz vs DC2; "
            "   Intracranial calcifications: present in Revesz (not DC2); "
            "   BMF: severe; cerebellar hypoplasia; sparse hair; "
            "   Most de novo heterozygous — parent unaffected; "
            "HOTSPOT VARIANTS: "
            "  Revesz: p.Arg282His / p.Arg282Cys — affects TRF2-binding interface; most de novo; "
            "  DC2: p.Lys280Glu / p.Lys280Asn / c.844_847del — variable severity; "
            "  Genotype-phenotype: Arg282 → severe Revesz; Lys280 → moderate DC2; "
            "TELOMERE SHORTENING: "
            "  Shortest telomeres of any DC genotype at equivalent age; "
            "  Flow-FISH telomere length: critical short → <1st percentile for age"
        ),
        "disease_category": (
            "DYSKERATOSIS CONGENITA DC2 / REVESZ SYNDROME — SHELTERIN-CORE DEFECT: "
            "DC TRIAD (CLASSIC): "
            "  1. Nail dystrophy: ridging + splitting + loss; "
            "  2. Oral leucoplakia: white patches on tongue/buccal mucosa; "
            "  3. Reticulate skin pigmentation: lacy net-like hyperpigmentation of neck/chest/upper arms; "
            "REVESZ-SPECIFIC FEATURES: "
            "  Bilateral exudative retinopathy: HALLMARK — Coats-like bilateral disease; "
            "    → Vision loss → retinal detachment if untreated; ophthalmological emergency; "
            "  Intracranial calcifications: basal ganglia/periventricular; "
            "  Cerebellar hypoplasia: variable; "
            "BONE MARROW FAILURE (BMF): "
            "  Aplastic anaemia: progressive pancytopenia; "
            "  MDS risk: elevated; "
            "  AML: secondary transformation possible; "
            "  Treatment: HSCT for severe BMF; "
            "    CRITICAL: pre-HSCT conditioning — myeloablative TBI CONTRAINDICATED (DC patients radiosensitive + poor lung reserve); "
            "    FLUDARABINE-BASED RIC MANDATORY; "
            "    IPF/Lung fibrosis: complicates HSCT — pulmonary function mandatory pre-HSCT; "
            "PULMONARY FIBROSIS (IPF): "
            "  Progressive interstitial lung disease; "
            "  May precede BMF; "
            "  Lung transplantation option in severe cases; "
            "  Danazol/androgens: may stabilise telomere loss + BMF (some benefit); "
            "CANCER: "
            "  Squamous cell carcinoma (SCC): oral + ano-genital + skin (from leucoplakia/reticulate skin); "
            "  AML/MDS: haematological; "
            "  Overall cancer risk substantially elevated; "
            "TELOMERE-BASED DIAGNOSIS: "
            "  Telomere length by Flow-FISH: <1st centile for age — critical short telomeres; "
            "  Most reliable objective test for DC spectrum; "
            "  Combined with gene panel sequencing"
        ),
        "disease_pathway": (
            "TINF2/TIN2 — SHELTERIN COMPLEX ARCHITECTURE: "
            "SHELTERIN COMPONENTS: "
            "  TRF1 (TERF1): binds telomere dsDNA (TTAGGG)n directly — homodimer; "
            "  TRF2 (TERF2): binds dsDNA — T-loop formation; protects 3' overhang; prevents ATM activation; "
            "  TIN2 (TINF2): bridges TRF1+TRF2 on dsDNA strand; anchors TPP1-POT1 on ssDNA strand; "
            "  TPP1 (ACD): recruits TERT to telomere; "
            "  POT1: binds ssDNA 3' overhang; prevents ATR activation at telomere; "
            "  RAP1 (TERF2IP): interacts with TRF2; NF-κB regulation; "
            "TIN2 BRIDGES dsDNA AND ssDNA SHELTERIN: "
            "  TIN2 N-terminus: binds TRF1 (directly at TRFH domain); "
            "  TIN2 central: binds TRF2 (at hinge); "
            "  TIN2 C-terminus: binds TPP1 (at TBM motif); "
            "  TIN2 = molecular glue connecting two major subcomplexes; "
            "TELOMERE PROTECTION: "
            "  TRF2-TIN2 complex maintains T-loop (3' overhang tucked into dsDNA strand) → hides end from DNA damage sensors; "
            "  POT1-TPP1 cap ssDNA 3' overhang → prevents ATR-ATRIP activation; "
            "  TINF2 mutation → disrupted TRF2/TRF1 bridging → uncapped telomere → DDR activation; "
            "TELOMERE ELONGATION: "
            "  TPP1 OB fold recruits TERT/TERC telomerase; "
            "  TIN2-TPP1 interaction critical for telomerase recruitment; "
            "  TINF2 LOF → TPP1 improperly positioned → reduced telomerase access → telomere shortening; "
            "REVESZ/DC2 MECHANISM: "
            "  Hotspot Arg282 is in TRF2-binding domain; Arg282 variants disrupt TRF2 binding → less stable shelterin; "
            "  Extreme telomere shortening → DNA damage response at telomere ends → cell senescence; "
            "  Rapidly dividing tissues (BMF, skin leucoplakia) most affected; "
            "  Retinal vasculature: sensitive to telomere dysfunction → exudative retinopathy"
        ),
        "pathognomonic": (
            "TINF2/DC2-REVESZ DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION: "
            "  DC triad (nail + leucoplakia + pigmentation) with: "
            "    Childhood BMF + "
            "    Bilateral exudative retinopathy (Revesz) or pulmonary fibrosis; "
            "  Revesz: bilateral exudative retinopathy in a child + BMF = TINF2 UNTIL PROVEN OTHERWISE; "
            "TELOMERE LENGTH TESTING (MANDATORY FIRST STEP): "
            "  Flow-FISH on peripheral blood lymphocytes (PBL); "
            "  TINF2: usually <1st percentile — shortest DC genotype; "
            "  Normal: 50th centile for age; DC: <10th centile; TINF2: <1st centile; "
            "MOLECULAR: "
            "  TINF2 gene sequencing; "
            "  Hotspot: p.Arg282His / p.Arg282Cys (Revesz) or p.Lys280Glu (DC2); "
            "  Full gene sequencing if hotspot negative; "
            "  Gene panel: TINF2 / DKC1 / TERT / TERC / RTEL1 / ACD / NHP2 / NOP10 / WRAP53; "
            "RETINAL EVALUATION: "
            "  Ophthalmology + RetCam/FFA: mandatory at diagnosis and every 6 months in Revesz; "
            "  Bilateral exudative retinopathy: anti-VEGF (bevacizumab) + laser photocoagulation; "
            "    Retinal detachment: vitrectomy; "
            "    Outcome: variable; some progress despite treatment; "
            "HSCT FOR BMF: "
            "  Telomere-based RIC conditioning MANDATORY (fludarabine-based); NO TBI; "
            "  Pulmonary function pre-HSCT (lung fibrosis risk); "
            "  Post-HSCT: haematological reconstituted but telomere-short non-haemopoietic tissues persist → "
            "    Ongoing IPF + retinopathy risk post-HSCT"
        ),
        "clinical_variables": {
            "telomere_length_percentile": (0.5, 2, "1st percentile — shortest in DC spectrum"),
            "revesz_retinopathy_pct": (60, 80, "bilateral exudative retinopathy in severe Revesz"),
            "bmf_by_age_10_pct": (50, 70, "bone marrow failure requiring HSCT"),
            "ipf_prevalence_pct": (20, 40, "interstitial pulmonary fibrosis complicating HSCT"),
            "de_novo_mutation_pct": (60, 80, "de novo variants — parent unaffected"),
            "cancer_scc_lifetime_pct": (20, 40, "squamous cell carcinoma from leucoplakia/skin sites"),
        },
    },
    {
        "gene": "RTEL1",
        "seed_base": 2765,
        "protein": (
            "RTEL1 -- 20q13.33 AR-AD -- 1219aa -- Regulator-Of-Telomere-Elongation-Helicase-1-"
            "128kDa-DEAH-D1-D2-Harmonin-Harmonin-Zinc-Finger-"
            "OMIM-Gene-608833-Disease-HH-616403-DC12-615190"
        ),
        "locus": "20q13.33",
        "protein_size": "1219 aa / 128 kDa (DEAH-box helicase; D1 + D2 tandem helicase domains; harmonin-N + PIP-box (PCNA-binding); zinc-finger; dismantles D-loops at telomeres during S-phase; disassembles G4 quadruplexes; suppresses T-loop excision; interacts with PCNA/RFC/shelterin TRF1)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic → severe: Hoyeraal-Hreidarsson / DC-12) AND "
            "AUTOSOMAL DOMINANT (monoallelic LOF → milder/adult: DC-12 / IPF / HCC predisposition): "
            "BIALLELIC SEVERE (Hoyeraal-Hreidarsson Syndrome — HHS / OMIM 616403): "
            "  Most severe end of dyskeratosis congenita spectrum; "
            "  Cerebellar hypoplasia: congenital/early — prominent; "
            "  Immunodeficiency: combined T + B + NK cell deficiency — severe; "
            "  Enteropathy: diarrhoea + failure to thrive; "
            "  DC triad: partial (nail dystrophy + leucoplakia ± pigmentation); "
            "  BMF: progressive + severe; "
            "  Short stature + microcephaly; "
            "  Onset: infancy–early childhood; "
            "  FATAL: median death without HSCT <10y; "
            "BIALLELIC MODERATE (DC-12 — Dyskeratosis Congenita type 12): "
            "  Standard DC with shorter telomeres; BMF; classic triad; no cerebellar involvement; "
            "MONOALLELIC RISK (AD Heterozygous): "
            "  Pulmonary fibrosis (IPF): RTEL1 c.2942A>G (p.Tyr981Cys) in IPF families; "
            "  Hepatocellular carcinoma (HCC): RTEL1 AD alleles in HCC predisposition pedigrees; "
            "  Telomere shortening: ~1st–5th centile; "
            "  Incomplete penetrance; family aggregation of IPF/liver disease; "
            "FOUNDER VARIANTS: "
            "  Ashkenazi Jewish: p.Arg995X (c.2983C>T) biallelic → HHS; "
            "  Finnish: p.Thr875Ala; Arab: p.Arg1264His"
        ),
        "disease_category": (
            "RTEL1 — HOYERAAL-HREIDARSSON (HHS) AND DC-12: "
            "HOYERAAL-HREIDARSSON (HHS) CLINICAL CRITERIA (OMIM modified): "
            "  Required major: cerebellar hypoplasia + immunodeficiency + "
            "    BMF + DC features (≥2 of triad); "
            "  Supporting: growth restriction; microcephaly; enteropathy; "
            "  HHS genes: RTEL1 most common; also DKC1 / TERT / TINF2 / PARN / ACD; "
            "NEUROLOGICAL: "
            "  Cerebellar hypoplasia: MRI finding — midline vermis most affected; "
            "  Developmental delay: variable; "
            "  Hypotonia: often; "
            "  Seizures: minority; "
            "IMMUNOLOGICAL: "
            "  Biallelic RTEL1 → profound combined immunodeficiency: "
            "    T cells: severely reduced; "
            "    B cells: reduced; "
            "    NK cells: reduced; "
            "    Immunoglobulins: panhypogammaglobulinaemia; "
            "  SCID-like presentation: Pneumocystis pneumonia; CMV; viral infections; "
            "BONE MARROW FAILURE: "
            "  Progressive pancytopenia; "
            "  Aplastic anaemia requiring transfusions + G-CSF; "
            "  HSCT: curative for BMF/immune — but cerebellar + lung disease not corrected; "
            "    Pre-HSCT: telomere-based RIC (NO TBI; fludarabine-based); "
            "    Short telomeres: RIC preferred to avoid telomere erosion from radiation; "
            "ENTEROPATHY: "
            "  Chronic diarrhoea; "
            "  Villous atrophy; "
            "  Partial response to parenteral nutrition; "
            "TELOMERE LENGTH: "
            "  Among shortest in all DC genotypes; "
            "  Flow-FISH <1st centile"
        ),
        "disease_pathway": (
            "RTEL1 — TELOMERE D-LOOP DISASSEMBLY + G4 RESOLUTION: "
            "TELOMERE D-LOOP DISASSEMBLY: "
            "  During S-phase: shelterin-bound telomere in T-loop configuration (3' overhang tucked into dsDNA); "
            "  RTEL1 (PIP-box) binds PCNA at replication fork → RTEL1 translocates to T-loop junction; "
            "  RTEL1 helicase unwinds the D-loop displacement strand → releases 3' overhang for replication; "
            "  Without RTEL1: T-loop cleavage by SLX4/MUS81 → terminal deletion → rapid telomere shortening; "
            "  RTEL1 LOF → excess T-loop excision → catastrophic telomere shortening → replicative crisis; "
            "G4 QUADRUPLEX RESOLUTION: "
            "  RTEL1 resolves G4 structures in telomeric ssDNA + at genome-wide G4 sites; "
            "  G4 resolution required to prevent replication fork stalling → prevents fragile site breakage; "
            "  RTEL1 LOF → G4 stalling → replication fork collapse → DSBs → genomic instability; "
            "PCNA INTERACTION (PIP-BOX): "
            "  RTEL1 C-terminal PIP-box binds PCNA ring; "
            "  PCNA recruits RTEL1 to replication forks; "
            "  RTEL1 helicase activity linked to fork-associated unwinding of secondary structures; "
            "ANTI-CROSSOVER FUNCTION: "
            "  RTEL1 dismantles D-loops in general genome → suppresses inappropriate homologous recombination (crossovers); "
            "  RTEL1 LOF → SCE elevation (less marked than BLM; moderate elevation); "
            "MONOALLELIC IPF MECHANISM: "
            "  Heterozygous LOF → reduced RTEL1 activity → accelerated telomere shortening in type 2 pneumocytes → "
            "    senescence → fibroblast activation → fibrosis; "
            "  Shared mechanism with TERT/TERC monoallelic IPF → telomere short → alveolar cell senescence"
        ),
        "pathognomonic": (
            "RTEL1 / HHS / DC-12 DIAGNOSTIC APPROACH: "
            "CLINICAL RECOGNITION: "
            "  Infant/young child with cerebellar hypoplasia + combined immunodeficiency + BMF + failure to thrive; "
            "  Partial DC triad (nail changes, oral white patches); "
            "  SCID-like infections: Pneumocystis carinii + viral → immunological work-up; "
            "TELOMERE LENGTH: "
            "  Flow-FISH: <1st centile (very short) in lymphocytes; "
            "  Alternative: qPCR telomere length in PBL (less precise but accessible); "
            "BRAIN MRI: "
            "  Cerebellar hypoplasia: midline vermis + bilateral; "
            "  White matter changes (variable); "
            "MOLECULAR: "
            "  RTEL1 gene sequencing; "
            "  Ashkenazi Jewish: target p.Arg995X first; "
            "  DC/HHS panel: RTEL1 / DKC1 / TERT / TERC / TINF2 / ACD / WRAP53 / NHP2 / PARN; "
            "HSCT APPROACH: "
            "  URGENCY: HHS with BMF + immunodeficiency → HSCT planning immediately; "
            "  Conditioning: FLUDARABINE-BASED RIC (NO TBI — telomere-short patients extremely sensitive); "
            "  HSCT corrects BMF + immunity; "
            "  Cerebellar + lung disease NOT corrected by HSCT; "
            "  Post-HSCT surveillance: pulmonary function + neurological; "
            "MONOALLELIC (AD) COUNSELLING: "
            "  Heterozygous parents of biallelic RTEL1 child: telomere screening; "
            "  IPF risk in adult heterozygotes: pulmonary function + HRCT every 2 years after 40y; "
            "  HCC: hepatology annual review if LFT abnormal; "
            "  Do NOT assume heterozygous parent is healthy — screen actively"
        ),
        "clinical_variables": {
            "telomere_length_percentile": (0.5, 1.5, "among shortest in DC spectrum — <1st centile"),
            "cerebellar_hypoplasia_pct": (75, 95, "MRI: cerebellar hypoplasia in HHS"),
            "immunodeficiency_severity_pct": (70, 90, "combined T+B+NK deficiency in biallelic RTEL1"),
            "bmf_requiring_hsct_pct": (70, 90, "progressive BMF requiring HSCT"),
            "ipf_monoallelic_risk_pct": (10, 25, "IPF risk in heterozygous carriers"),
            "enteropathy_prevalence_pct": (40, 60, "chronic diarrhoea / villous atrophy in HHS"),
        },
    },
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _patient_cohort(gene_entry: dict) -> list[dict]:
    """Generate 40 synthetic patients per gene following the progeroid/premature-aging phenotype."""
    gene = gene_entry["gene"]
    seed_base = gene_entry["seed_base"]
    clin = gene_entry["clinical_variables"]
    patients = []
    for i in range(40):
        r = _rng(seed_base * 1000 + i)

        # Age at first presentation
        if gene == "WRN":
            age = r.randint(16, 35)
            onset_label = "Adult-onset"
        elif gene == "BLM":
            age = r.randint(0, 5)
            onset_label = "Congenital/early-infantile"
        elif gene == "RECQL4":
            age = r.randint(0, 3)
            onset_label = "Infantile"
        elif gene in ("ERCC6", "ERCC8"):
            age = r.randint(0, 4)
            onset_label = "Infantile–early childhood"
        elif gene == "BANF1":
            age = r.randint(0, 2)
            onset_label = "Congenital/infantile"
        elif gene == "TINF2":
            age = r.randint(1, 10)
            onset_label = "Childhood"
        elif gene == "RTEL1":
            age = r.randint(0, 3)
            onset_label = "Infancy"
        else:
            age = r.randint(0, 15)
            onset_label = "Variable"

        sex = r.choice(["M", "F"])

        # Primary clinical variable value (gene-specific)
        primary_key = list(clin.keys())[0]
        lo, hi, _ = clin[primary_key]
        primary_val = round(r.uniform(lo, hi), 1)

        # Cancer risk (gene-specific)
        cancer_map = {
            "WRN": r.uniform(35, 60),
            "BLM": r.uniform(80, 99),
            "RECQL4": r.uniform(25, 55),
            "ERCC6": r.uniform(3, 8),   # No skin cancer; low overall
            "ERCC8": r.uniform(3, 8),   # No skin cancer; low overall
            "BANF1": r.uniform(5, 15),
            "TINF2": r.uniform(20, 40),
            "RTEL1": r.uniform(15, 35),
        }
        cancer_risk = round(cancer_map[gene], 1)

        # Inheritance
        inheritance_map = {
            "WRN": "AR", "BLM": "AR", "RECQL4": "AR",
            "ERCC6": "AR", "ERCC8": "AR", "BANF1": "AR",
            "TINF2": "AD (de novo)", "RTEL1": "AR",
        }

        # Treatment approach
        treatment_map = {
            "WRN": "Surveillance (cataracts/DM/cancer); statins; metformin; wound care",
            "BLM": "Sun avoidance; cancer surveillance; dose-reduced chemotherapy; avoid TBI",
            "RECQL4": "Osteosarcoma surveillance; sun protection; cataract extraction",
            "ERCC6": "Supportive (feeding/hearing); sun protection; palliative when advanced",
            "ERCC8": "Supportive; sun protection; hearing aids; nutrition",
            "BANF1": "Bone health (bisphosphonates); cardiac monitoring; lonafarnib (off-label)",
            "TINF2": "HSCT (fludarabine-RIC; no TBI); anti-VEGF/laser for retinopathy; androgens",
            "RTEL1": "HSCT (fludarabine-RIC; no TBI); pulmonary monitoring; immunoglobulin replacement",
        }

        patients.append({
            "id": f"{gene}-P{i+1:02d}",
            "gene": gene,
            "sex": sex,
            "age_at_presentation_y": age,
            "onset_label": onset_label,
            "inheritance": inheritance_map[gene],
            "primary_measure_label": primary_key,
            "primary_measure_value": primary_val,
            "cancer_risk_pct": cancer_risk,
            "treatment": treatment_map[gene],
            "locus": gene_entry["locus"],
        })
    return patients


def generate_overview() -> dict:
    """Overview: atlas-level summary across all 8 genes."""
    all_patients = []
    gene_summaries = []
    for g in ATLAS_GENES:
        pts = _patient_cohort(g)
        all_patients.extend(pts)
        avg_cancer = round(sum(p["cancer_risk_pct"] for p in pts) / len(pts), 1)
        gene_summaries.append({
            "gene": g["gene"],
            "locus": g["locus"],
            "protein_size": g["protein_size"].split("(")[0].strip(),
            "n_patients": len(pts),
            "avg_cancer_risk_pct": avg_cancer,
            "inheritance": pts[0]["inheritance"] if pts else "AR",
            "onset": pts[0]["onset_label"] if pts else "Variable",
            "disease_brief": g["protein"].split("OMIM")[0].strip(),
        })

    avg_cancer_all = round(sum(p["cancer_risk_pct"] for p in all_patients) / len(all_patients), 1)
    sex_dist = {"M": sum(1 for p in all_patients if p["sex"] == "M"),
                "F": sum(1 for p in all_patients if p["sex"] == "F")}

    return {
        "atlas": "Hereditary-Progeroid-Premature-Aging-Atlas",
        "subtitle": "Complete 8-Gene Segmental Progeroid Syndrome Reference",
        "genes": ["WRN", "BLM", "RECQL4", "ERCC6", "ERCC8", "BANF1", "TINF2", "RTEL1"],
        "n_genes": 8,
        "n_patients": len(all_patients),
        "seeds": "2758-2765",
        "avg_cancer_risk_pct": avg_cancer_all,
        "sex_distribution": sex_dist,
        "gene_summaries": gene_summaries,
        "pathway_groups": {
            "RecQ Helicases (premature aging + genomic instability)": ["WRN", "BLM", "RECQL4"],
            "Cockayne Syndrome / TC-NER (premature aging without cancer)": ["ERCC6", "ERCC8"],
            "Nuclear Lamina / Telomere Maintenance (progeroid + BMF)": ["BANF1", "TINF2", "RTEL1"],
        },
        "key_clinical_distinctions": [
            "WRN: adult-onset (2nd decade); bilateral cataracts FIRST; mesenchymal cancer predominant; DM; scleroderma",
            "BLM: SCE 10x elevated (PATHOGNOMONIC); Ashkenazi founder BLMAsh; butterfly facial erythema; azoospermia",
            "RECQL4: poikiloderma infantile; osteosarcoma 30-50% lifetime (CRITICAL); Rapadilino (Finnish) / Baller-Gerold",
            "ERCC6: CS-B (most severe CS); TC-NER defect; NO skin cancer; neurodegeneration + cachexia; RRS reduced/UDS normal",
            "ERCC8: CS-A (milder); CS-A vs CS-B = same cellular assay → gene sequencing required to distinguish",
            "BANF1: ultra-rare NGPS; progeroid like HGPS but LMNA normal; clavicular resorption distinctive; Ala12Thr founder",
            "TINF2: shelterin core; shortest telomeres; Revesz = bilateral exudative retinopathy (PATHOGNOMONIC); de novo",
            "RTEL1: HHS (most severe DC); cerebellar hypoplasia + SCID + BMF; D-loop disassembly; IPF monoallelic risk",
        ],
        "absolutely_contraindicated": [
            "TBI (total body irradiation): CONTRAINDICATED in TINF2/RTEL1 BMF — telomere-short + radiosensitive",
            "WRN / BLM: avoid standard myeloablative conditioning; use fludarabine-based RIC if HSCT required",
            "ERCC6/ERCC8 (CS): no standard drug CI but note increased sensitivity to alkylating agents (theoretical)",
            "RECQL4: cisplatin sensitivity may be elevated — note in osteosarcoma chemotherapy planning",
            "BLM: dose-reduce mitomycin C / alkylating agents (hypersensitivity to crosslinks)",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown: 40 patients/gene with progeroid clinical parameters."""
    breakdown = {}
    for g in ATLAS_GENES:
        pts = _patient_cohort(g)
        breakdown[g["gene"]] = {
            "gene": g["gene"],
            "locus": g["locus"],
            "protein": g["protein"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "n_patients": len(pts),
            "patients": pts,
            "clinical_variables": g["clinical_variables"],
            "pathognomonic_notes": g["pathognomonic"],
            "summary_stats": {
                "avg_age_at_presentation_y": round(sum(p["age_at_presentation_y"] for p in pts) / len(pts), 1),
                "avg_cancer_risk_pct": round(sum(p["cancer_risk_pct"] for p in pts) / len(pts), 1),
                "pct_male": round(100 * sum(1 for p in pts if p["sex"] == "M") / len(pts), 1),
            },
        }
    return breakdown


def generate_definitions() -> dict:
    """Glossary: progeroid syndromes, RecQ helicases, Cockayne syndrome, nuclear lamina, DC spectrum."""
    defs: dict[str, str] = {}
    for g in ATLAS_GENES:
        defs[g["gene"]] = g["protein"]

    defs.update({
        "Werner Syndrome (WS)": (
            "Autosomal recessive adult-onset segmental progeroid syndrome caused by biallelic WRN LOF; "
            "Bilateral cataracts: first sign (2nd decade); scleroderma-like skin + ulcers; T2DM; dyslipidaemia; "
            "Mesenchymal cancers predominate (sarcoma > carcinoma) — inverted vs general population; "
            "VTM (variegated translocation mosaicism) karyotype; "
            "Japanese IVS25-1G>C and c.4197C>A founders; "
            "Management: annual ophthalmology; metabolic control; aggressive wound care; cancer surveillance"
        ),
        "Bloom Syndrome (BS)": (
            "Autosomal recessive syndrome with severe growth retardation + butterfly facial erythema + cancer predisposition; "
            "SCE (sister chromatid exchange) 10-fold elevation — PATHOGNOMONIC; "
            "BLMAsh: c.2207_2212delATCTGAinsTAGATTC — Ashkenazi Jewish founder (~1/100 carrier); "
            "Cancer risk approaches 100% lifetime; haematological + solid tumours; multiple primaries; "
            "Male azoospermia (infertile); female fertility preserved; "
            "Sun avoidance; dose-reduced chemotherapy; avoid TBI conditioning"
        ),
        "Rothmund-Thomson Syndrome Type 2 (RTS2)": (
            "Autosomal recessive syndrome caused by biallelic RECQL4 LOF; "
            "Cardinal: poikiloderma (erythema → atrophy/pigmentation/telangiectasia) onset 3-6 months — cheeks → extremities; "
            "Osteosarcoma: 30-50% lifetime risk — annual surveillance X-rays from age 5; "
            "Also Rapadilino (Finnish c.1390+2T>C) and Baller-Gerold (craniosynostosis) allelic; "
            "Cisplatin sensitivity: possible → note in osteosarcoma planning"
        ),
        "Cockayne Syndrome B (CS-B)": (
            "Autosomal recessive segmental progeroid disorder caused by biallelic ERCC6 LOF; "
            "Most common CS gene (~65%); most severe CS form; "
            "Cardinal triad: sun sensitivity WITHOUT skin cancer + neurodegeneration + cachectic growth failure; "
            "TC-NER defect — GGR intact (explains absent skin cancer); "
            "RRS reduced / UDS normal — cellular gold standard; "
            "Median death ~12-16 years; UVSS-B (milder ERCC6 alleles) → sun sensitivity only; "
            "NAD+ supplementation: preclinical benefit in CS mouse models"
        ),
        "Cockayne Syndrome A (CS-A)": (
            "Autosomal recessive CS caused by biallelic ERCC8 LOF; "
            "2nd most common CS gene (~35%); milder than CS-B; "
            "Same phenotype as CS-B but later onset + slower progression; "
            "Cellular assay cannot distinguish CS-A from CS-B (both: UDS normal, RRS reduced) → gene sequencing required; "
            "UVSS-A (ERCC8 missense) → sun sensitivity only, normal lifespan"
        ),
        "Nestor-Guillermo Progeria Syndrome (NGPS)": (
            "Ultra-rare autosomal recessive progeroid syndrome caused by biallelic BANF1 LOF; "
            "Clinically similar to HGPS (alopecia + aged skin + subcutaneous fat loss) but LMNA NORMAL; "
            "Distinctive: severe osteoporosis + clavicular resorption + acro-osteolysis (more than HGPS); "
            "Cardiovascular less severe than HGPS; "
            "BANF1 p.Ala12Thr founding variant in most cases; "
            "Fewer than 12 published patients 2024; "
            "Management: bisphosphonates + cardiac monitoring + lonafarnib (off-label, HGPS-approved)"
        ),
        "Dyskeratosis Congenita (DC) / Revesz Syndrome (TINF2)": (
            "TINF2 (TIN2 shelterin) biallelic or dominant de novo → DC-2 or Revesz syndrome; "
            "DC triad: nail dystrophy + oral leucoplakia + reticulate skin pigmentation; "
            "Revesz: bilateral exudative retinopathy (PATHOGNOMONIC) + intracranial calcifications + severe BMF; "
            "Shortest telomeres in DC spectrum (Flow-FISH <1st centile); "
            "HSCT for BMF: fludarabine-RIC MANDATORY — NO TBI; "
            "Retinopathy: anti-VEGF + laser; retinal detachment risk"
        ),
        "Hoyeraal-Hreidarsson Syndrome (HHS) — RTEL1": (
            "Most severe DC spectrum phenotype; biallelic RTEL1 LOF; "
            "Features: cerebellar hypoplasia + combined immunodeficiency (SCID-like) + BMF + DC triad + failure to thrive; "
            "RTEL1 function: T-loop disassembly + G4 quadruplex resolution at telomeres; "
            "RTEL1 LOF → excess T-loop excision → catastrophic telomere shortening; "
            "HSCT: urgent (fludarabine-RIC; NO TBI); corrects BMF/immunity but NOT cerebellum; "
            "Monoallelic RTEL1: IPF + HCC risk in adult heterozygotes"
        ),
        "RecQ Helicase Family": (
            "Five human RecQ helicases: RECQL1 / BLM / WRN / RECQL4 / RECQL5; "
            "All 3'→5' DEAH-box helicases with RQC (RecQ C-terminal) + HRDC domains; "
            "BLM + WRN: suppress inappropriate crossover recombination (BTR complex / D-loop suppression); "
            "RECQL4: replication origin initiation (Sld2-homologous N-terminal domain); "
            "RecQ helicase LOF → segmental progeroid syndromes with variable cancer risk; "
            "All three (WRN/BLM/RECQL4) cause hereditary premature aging with distinct phenotypes"
        ),
        "Sister Chromatid Exchange (SCE)": (
            "Reciprocal exchange of DNA between sister chromatids during S-phase; "
            "Normal rate: 3-10 SCE/cell; "
            "Bloom Syndrome: 40-100 SCE/cell (10-fold elevation) — PATHOGNOMONIC; "
            "Assay: BrdU incorporation for 2 cell cycles + Hoechst-33258 + Giemsa staining; "
            "No other condition produces this SCE elevation — highly specific for BS"
        ),
        "TC-NER (Transcription-Coupled Nucleotide Excision Repair)": (
            "Sub-pathway of NER triggered by RNA Pol II stalling at DNA lesion; "
            "Requires CSB (ERCC6) + CSA (ERCC8) as coupling factors; "
            "CSB recruits CRL4(CSA); CSA ubiquitylates CSB + RNAPII; "
            "Shared final steps with GGR (XPA/TFIIH/XPG/XPF); "
            "TC-NER defect (CS): neurodegeneration + aging WITHOUT skin cancer (GGR intact repairs skin damage); "
            "GGR defect (XP): skin cancer from UV damage in non-transcribed DNA; "
            "Both defects (XP+CS overlap): cancer + neurodegeneration"
        ),
        "Variegated Translocation Mosaicism (VTM)": (
            "Multiple clonal chromosome translocations in different lymphocyte clones; "
            "Quasi-pathognomonic of Werner Syndrome + Bloom Syndrome; "
            "Mechanism: RecQ helicase LOF → impaired suppression of inappropriate recombination; "
            "Different cells in one individual acquire different somatic translocations; "
            "VTM karyotype: multiple aberrations per cell but different in each clone"
        ),
        "Shelterin Complex": (
            "Six-protein complex protecting telomere ends from DNA damage response; "
            "Components: TRF1 + TRF2 + TIN2 (TINF2) + TPP1 (ACD) + POT1 + RAP1 (TERF2IP); "
            "TRF1/TRF2 bind dsDNA TTAGGG repeats; POT1-TPP1 cap ssDNA 3' overhang; "
            "TIN2 bridges dsDNA and ssDNA subcomplexes; "
            "Loss: uncapped telomere → ATM/ATR activation → DDR → cell senescence/apoptosis"
        ),
        "Flow-FISH Telomere Length Assay": (
            "Gold-standard quantitative telomere length measurement in peripheral blood; "
            "Method: hybridise PNA-FITC probe to telomere repeats → flow cytometry; "
            "Reports: absolute telomere length (kb) vs normal centiles for age; "
            "DC spectrum: usually <10th centile; TINF2/RTEL1: <1st centile; "
            "Used to guide DC diagnosis, HSCT timing, and monitoring of telomere biology disorders"
        ),
        "Lonafarnib (Farnesyltransferase Inhibitor)": (
            "Inhibitor of farnesylation (post-translational modification); "
            "FDA-approved for HGPS (Hutchinson-Gilford Progeria — LMNA/progerin); "
            "Mechanism in HGPS: blocks farnesylation of progerin → progerin less tightly bound to lamina → reduced nuclear damage; "
            "In NGPS (BANF1): off-label only — theoretical basis (same downstream nuclear lamina pathway); "
            "In WS/Werner: no approved indication; farnesyl pathway less relevant; "
            "Side effects: nausea/vomiting, diarrhoea, bone marrow suppression (manageable)"
        ),
        "Cascade Testing in Progeroid Genes": (
            "WRN/BLM/RECQL4/ERCC6/ERCC8/BANF1: AUTOSOMAL RECESSIVE — carrier parents unaffected; "
            "  Sibling risk: 25% biallelic; carrier testing recommended for siblings; "
            "  Partner testing of carrier: important for reproductive planning; "
            "TINF2: AUTOSOMAL DOMINANT (de novo in majority — parent-to-child risk 50% if germline); "
            "  Gonadal mosaicism rare but documented — next sibling at low but non-zero risk; "
            "RTEL1: biallelic = AR (both parents carriers); monoallelic = AD risk in parent; "
            "  Screen heterozygous parents for IPF/liver disease (RTEL1 monoallelic risk); "
            "BLMAsh: pre-conception Ashkenazi Jewish carrier screening — standard of care in some centres; "
            "RECQL4/RTS2: osteosarcoma in first decade → cascade testing for siblings at 25% risk is urgent"
        ),
    })
    return {"definitions": defs, "n_entries": len(defs)}
