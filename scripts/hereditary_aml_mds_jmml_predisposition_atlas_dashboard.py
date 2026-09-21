"""Hereditary AML-MDS-JMML Predisposition Atlas — 8-Gene Reference
GATA2-DDX41-CEBPA-ETV6-SAMD9-SAMD9L-PTPN11-CBL
Germline Predisposition to AML / MDS / ALL / JMML / Monosomy-7 Syndromes
320 patients (8 x 40), seeds 2870-2877.
Endpoints: /api/hereditary-aml-mds-jmml-predisposition-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "GATA2",
        "protein": (
            "GATA2 -- 3q21.3 AD -- 480aa -- "
            "GATA-Binding-Protein-2-50kDa-Zinc-Finger-Transcription-Factor-"
            "Haematopoietic-Stem-Cell-Master-Regulator-"
            "GATA2-Deficiency-MonoMAC-Emberger-Familial-MDS-AML-"
            "OMIM-Gene-137295-Disease-MDS-614286-Emberger-614038"
        ),
        "locus": "3q21.3",
        "protein_size": (
            "480 aa / 50 kDa (GATA-binding protein 2; two zinc fingers ZnF-N / ZnF-C; "
            "ZnF-C mediates DNA binding to GATA motif (WGATAR); "
            "ZnF-N mediates cofactor interaction (FOG1/ZFPM1); "
            "GATA2 is master TF for haematopoietic stem cells (HSCs) and lymphoid progenitors; "
            "GATA2 deficiency → quantitative and qualitative defect in HSCs, "
            "dendritic cells (myeloid DCs, plasmacytoid DCs), NK cells, monocytes, B cells; "
            "LOF mutations: missense (ZnF, p.Thr354Met), nonsense, frameshift, splice, large deletions; "
            "R398W, C373R most recurrent; "
            "haploinsufficiency is disease mechanism — one functional allele insufficient"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — HAPLOINSUFFICIENCY: "
            "  De novo OR inherited; highly variable expressivity and penetrance; "
            "  Familial cases: affected parent may have mild phenotype (late-onset MDS/lymphedema only); "
            "  Penetrance: incomplete (~80% lifetime penetrance for some manifestation); "
            "CLINICAL PHENOTYPE SPECTRUM — GATA2 DEFICIENCY SYNDROME: "
            "  1. MonoMAC SYNDROME: MONOcytopenia, MAC (Mycobacterium avium complex) + other atypical infections; "
            "     Monocytes <0.01×10⁹/L (near-absent); NK cells markedly reduced; B cells absent/low; "
            "     Infections: NTM (MAC, M.kansasii), HPV (warts, cervical/anogenital dysplasia/cancer), CMV, EBV; "
            "  2. EMBERGER SYNDROME: primary lymphedema + MDS/AML + sensorineural hearing loss; "
            "     Lymphedema: unilateral lower limb, onset childhood-adolescence; "
            "  3. FAMILIAL MDS/AML: MDS (especially monosomy 7, trisomy 8) → AML transformation; "
            "     MDS transformation risk: ~75% by age 70; AML risk cumulative; "
            "  4. PULMONARY ALVEOLAR PROTEINOSIS (PAP): surfactant accumulation; GM-CSF signalling impaired; "
            "  5. DCML DEFICIENCY: Dendritic Cell, Monocyte, B, NK Lymphocyte deficiency"
        ),
        "disease_category": (
            "GATA2 DEFICIENCY SYNDROME — MULTIPLE HAEMATOLOGICAL MALIGNANCY PREDISPOSITION: "
            "MDS / MONOSOMY 7: most common malignant presentation; "
            "  Monosomy 7 or del(7q) in ~50-70% of MDS in GATA2 deficiency; "
            "  Trisomy 8 also common; "
            "AML: evolution from MDS; poor prognosis without HSCT; "
            "IMMUNODEFICIENCY: severe combined infections (NTM, fungal, viral); "
            "  NTM prophylaxis (azithromycin); antifungal (posaconazole); antiviral monitoring; "
            "HPV ONCOGENESIS: squamous cell carcinoma risk (cervical, anal, penile, oral) from persistent HPV; "
            "HSCT: only curative option for MDS/AML component; "
            "  Timing: before MDS → AML transformation; before severe infections compromise fitness; "
            "SURVEILLANCE: annual BM biopsy + flow cytometry for MDS in all GATA2 mutation carriers"
        ),
        "disease_pathway": (
            "GATA2 → HSC MAINTENANCE / LYMPHOID DIFFERENTIATION: "
            "NORMAL: GATA2 activates NOTCH signalling → definitive haematopoiesis; "
            "  GATA2 maintains HSC quiescence; "
            "  GATA2 required for myeloid DC / pDC / monocyte / NK / B lineage commitment; "
            "HAPLOINSUFFICIENCY CONSEQUENCES: "
            "  Reduced HSC self-renewal → compensatory proliferation → replicative stress; "
            "  Selective lymphoid lineage loss: pDC/NK/monocyte/B-cell near-absent; "
            "  Monocyte absence → macrophage-mediated mycobacterial killing fails (MonoMAC); "
            "  NK absence → viral (CMV, EBV) surveillance fails; "
            "  B-cell absence → antibody deficiency; "
            "CLONAL EVOLUTION: "
            "  GATA2 haploinsufficiency → genomic instability → monosomy 7 / trisomy 8 acquisition; "
            "  Secondary hits: ASXL1, SETBP1, KRAS, NRAS, CBL mutations; "
            "  GATA2 deficiency + monosomy 7 → MDS → AML pathway"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — GATA2 DEFICIENCY: "
            "  1. MONOCYTOPENIA (<0.01×10⁹/L near-absent) + NTM infection: MonoMAC PATHOGNOMONIC; "
            "     virtually no other condition causes monocyte near-absence with NTM; "
            "  2. DCML on flow cytometry: absent myeloid DCs + absent pDCs + absent NK cells + absent B cells; "
            "     all four absent simultaneously → near-pathognomonic GATA2 deficiency; "
            "  3. LYMPHEDEMA + MDS + SNHL triad: Emberger syndrome → GATA2 sequencing mandatory; "
            "  4. MONOSOMY 7 IN MDS in young adult with immunodeficiency history: GATA2 must be excluded; "
            "  5. RECURRENT HPV-RELATED ANOGENITAL DYSPLASIA in young immunocompetent-looking patient: "
            "     absent NK + B cells on flow → GATA2; "
            "DDx: WHIM (CXCR4 — neutrophil egress; myelokathexis; warts), "
            "XLA (BTK — B-cell absent but NK/monocytes normal), "
            "SCN (ELANE — neutropenia not monocytopenia)"
        ),
        "treatment": (
            "GATA2 DEFICIENCY TREATMENT: "
            "HSCT: only cure — corrects haematopoietic and immune defects; "
            "  Indicated before MDS→AML transformation or severe infectious complications; "
            "  Matched sibling or 10/10 MUD; conditioning per MDS protocols; "
            "  Post-HSCT: immune reconstitution; residual HPV disease may persist; "
            "ANTIMICROBIAL PROPHYLAXIS (pre-HSCT): "
            "  NTM: azithromycin prophylaxis; treat MAC with ≥3-drug regimen (clarithromycin + ethambutol + rifampicin); "
            "  Fungal: posaconazole prophylaxis; "
            "  Viral: aciclovir (HSV/VZV); CMV monitoring + ganciclovir; "
            "  PCP: co-trimoxazole; "
            "HPV SURVEILLANCE: cervical/anal smears annually; colposcopy if abnormal; "
            "PULMONARY PAP: whole-lung lavage or inhaled GM-CSF (off-label); "
            "LYMPHEDEMA: compression garments; manual lymphatic drainage; avoid cellulitis; "
            "SURVEILLANCE PROTOCOL: annual CBC + flow + BM biopsy in all mutation carriers"
        ),
        "seed": 2870,
        "n_patients": 40,
    },
    {
        "gene": "DDX41",
        "protein": (
            "DDX41 -- 5q35.3 AD -- 622aa -- "
            "DEAD-Box-Helicase-41-68kDa-RNA-Helicase-"
            "Innate-Immune-Sensing-dsDNA-Spliceosome-Component-"
            "Most-Common-Adult-AML-Germline-Predisposition-"
            "OMIM-Gene-608170-Disease-AML-Predisposition-616860"
        ),
        "locus": "5q35.3",
        "protein_size": (
            "622 aa / 68 kDa (DEAD-box RNA helicase 41; DEAD-box motif (Asp-Glu-Ala-Asp) — ATPase/helicase; "
            "functions: RNA splicing (spliceosome P-complex), "
            "innate immune sensing (cytosolic dsDNA via STING pathway), "
            "ribosome biogenesis; "
            "zinc-finger domain for protein-protein interaction; "
            "predominantly nuclear; "
            "germline LOF mutations: p.D140Gfs frameshift (most common in Europeans), "
            "p.M1I (start-loss, Northern European founder), p.Y259C; "
            "somatic second-hit mutations in tumour: p.R525H (most common somatic, impairs helicase activity)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — LOF — BIALLELIC HIT MECHANISM: "
            "  Germline LOF (first hit) + somatic second-hit in tumour (R525H most common); "
            "  Haploinsufficiency alone insufficient — second somatic hit required for AML; "
            "  ~4% of ALL AML carry germline DDX41 pathogenic variant (most common adult AML germline); "
            "  High penetrance in males (>85% lifetime AML risk); lower in females; "
            "  Median age of AML diagnosis: 60-70 years (late-onset); "
            "  Familial: parent + child cases; BOTH may develop AML decades apart; "
            "  Founder variants: p.D140Gfs*2 (European), p.M1I (Scandinavian); "
            "TUMOUR TYPE: predominantly AML (M1/M2), less commonly MDS → sAML; "
            "  DDX41-AML: often normal karyotype; biallelic DDX41 hits in tumour; "
            "  Somatic R525H impairs helicase + STING-pathway innate immune sensing"
        ),
        "disease_category": (
            "DDX41-AML — ADULT-ONSET MYELOID MALIGNANCY PREDISPOSITION: "
            "AML RISK: ~4-10% of AML with germline DDX41 LOF; "
            "  Male predominance (3:1); later onset than most hereditary AML syndromes; "
            "  Karyotype: often normal (favourable); "
            "  Morphology: AML-M1/M2 (myeloblastic without maturation or with maturation); "
            "  Response to standard induction (7+3): generally good first remission; "
            "  Relapse: common; HSCT in CR1 considered for germline DDX41 AML; "
            "MDS: MDS → AML transformation also described; "
            "IMPLICATIONS FOR FAMILY MEMBERS: "
            "  Siblings + children of DDX41 AML patient → germline testing mandatory; "
            "  Variant detected → clinical surveillance + HSCT-donor exclusion; "
            "SOMATIC SECOND HIT (R525H): if detected in tumour, confirms biallelic loss; "
            "CLINICAL ALERT: DDX41 germline = sibling cannot donate marrow (also carries variant risk)"
        ),
        "disease_pathway": (
            "DDX41 — RNA SPLICING + INNATE IMMUNE SENSING: "
            "NORMAL FUNCTIONS: "
            "  1. SPLICEOSOME P-COMPLEX: DDX41 resolves RNA-RNA secondary structures during splicing; "
            "     LOF → aberrant splicing of tumour suppressors (TP53, RB1 splicing defects); "
            "  2. INNATE IMMUNE: cytosolic dsDNA → DDX41 → STING → IRF3 → IFN-β; "
            "     LOF → impaired dsDNA sensing → immune evasion of pre-leukemic clone; "
            "  3. RIBOSOME BIOGENESIS: nucleolar rRNA processing; "
            "LOF MECHANISM: "
            "  Germline LOF (haploinsufficiency) → partial splicing defects + partial immune sensing loss; "
            "  Somatic R525H second hit → complete helicase inactivation + complete STING pathway loss; "
            "  Leukemic clone: evades innate immune surveillance + accumulates splicing errors; "
            "  R525H disrupts Walker A/B ATP-binding → no ATPase → no helicase activity"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — DDX41 AML: "
            "  1. AML IN MALE PATIENT AGED 60-75 WITH FAMILY HISTORY OF AML: "
            "     DDX41 germline probability high → germline testing mandatory; "
            "  2. NORMAL KARYOTYPE AML WITH SOMATIC R525H IN TUMOUR: "
            "     R525H is near-exclusive somatic second hit in DDX41 germline AML; "
            "     finding R525H in tumour → reflexly test germline for D140Gfs/M1I; "
            "  3. TWO GENERATIONS WITH AML (PARENT + CHILD): "
            "     familial AML pattern → DDX41 most common explanation; "
            "  4. AML + NO ANTECEDENT MPN/APLASIA: de novo; DDX41 AML typically de novo; "
            "DDx: RUNX1 FPD/AML (thrombocytopenia precedes), CEBPA (younger, biallelic bZIP), "
            "GATA2 (immunodeficiency + monocytopenia), ANKRD26 (thrombocytopenia + elevated TPO)"
        ),
        "treatment": (
            "DDX41 AML TREATMENT: "
            "INDUCTION: standard 7+3 (cytarabine + daunorubicin/idarubicin); "
            "  DDX41 AML: generally achieves CR1 with standard induction; "
            "CONSOLIDATION / HSCT: "
            "  HSCT in CR1 recommended given high relapse risk; "
            "  DONOR SELECTION CRITICAL: siblings must be DDX41-tested before donation; "
            "  ~50% of siblings carry same germline variant → cannot donate; "
            "  Prefer MUD over untested sibling; "
            "SURVEILLANCE IN GERMLINE CARRIERS (pre-AML): "
            "  Annual CBC; BM biopsy if cytopenias develop; "
            "  No proven chemoprevention; "
            "FAMILY CASCADING: all first-degree relatives offered germline DDX41 testing; "
            "GENETIC COUNSELLING: penetrance, donor implications, family planning discussed"
        ),
        "seed": 2871,
        "n_patients": 40,
    },
    {
        "gene": "CEBPA",
        "protein": (
            "CEBPA -- 19q13.11 AD -- 358aa -- "
            "CCAAT-Enhancer-Binding-Protein-Alpha-42kDa-"
            "Basic-Leucine-Zipper-bZIP-Myeloid-TF-"
            "Familial-AML-Biallelic-bZIP-Mutations-10pct-AML-Favourable-"
            "OMIM-Gene-116897-Disease-Familial-AML-601626"
        ),
        "locus": "19q13.11",
        "protein_size": (
            "358 aa / 42 kDa (CCAAT/enhancer-binding protein alpha; "
            "domains: N-terminal transactivation domains TA1 + TA2; "
            "central regulatory region; C-terminal basic leucine zipper (bZIP) — DNA binding + dimerisation; "
            "CEBPA drives myeloid differentiation: HSC → GMP → granulocyte; "
            "BIALLELIC MUTATION PATTERN IN FAMILIAL AML: "
            "  Germline mutation: N-terminal frameshift (e.g., p.Q72Pfs*11) — disrupts TA domains; "
            "  Somatic second hit: C-terminal bZIP mutation — dominant-negative, impairs DNA binding; "
            "  Both mutations required for AML (biallelic hit model); "
            "  Monoallelic germline N-terminal frameshift: obligate carrier, not yet AML; "
            "  p42/p30 isoform ratio altered by N-terminal frameshift (p30 dominant)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — BIALLELIC SOMATIC HIT MODEL: "
            "  Germline: N-terminal frameshift (monoallelic) = predisposition; "
            "  Somatic: C-terminal bZIP second hit in clone = AML onset; "
            "  ~10% of ALL AML harbour biallelic CEBPA mutations (mostly somatic-somatic); "
            "  ~5-10% of biallelic CEBPA AML have GERMLINE N-terminal frameshift; "
            "  FAMILIAL AML: parent + child AML; both carry germline N-terminal frameshift; "
            "  Penetrance: ~50-80% lifetime AML risk in N-terminal frameshift carriers; "
            "  Age of onset: younger than DDX41 (30-60 years); "
            "  Sex: equal M:F (unlike DDX41 male predominance); "
            "NO PRECEDING CYTOPENIAS: presents as de novo AML without antecedent MDS"
        ),
        "disease_category": (
            "CEBPA-bZIP AML — FAVOURABLE PROGNOSIS MYELOID MALIGNANCY: "
            "AML SUBTYPE: AML with biallelic CEBPA mutations; ELN 2022 FAVOURABLE risk; "
            "  Morphology: often M1/M2 (similar to DDX41); "
            "  Karyotype: predominantly normal; "
            "PROGNOSIS: biallelic CEBPA AML has BEST prognosis of favourable-risk AML; "
            "  CR1 rate: >90%; OS at 5y: ~60-70% (chemotherapy alone); "
            "  HSCT not routinely recommended in CR1 for favourable CEBPA AML (chemotherapy sufficient); "
            "  EXCEPTION: germline CEBPA AML → HSCT in CR1 considered (high relapse risk in germline cases); "
            "RELAPSE: ~30-40% relapse; relapse often has new C-terminal bZIP second hit; "
            "  Second induction achieves CR2 in many; HSCT in CR2; "
            "DISTINCTION FROM SOMATIC CEBPA: "
            "  Germline = N-terminal frameshift in all tissues (blood + buccal); "
            "  Somatic = only in tumour DNA"
        ),
        "disease_pathway": (
            "CEBPA → GRANULOCYTE DIFFERENTIATION BLOCK: "
            "NORMAL: CEBPA activates genes for GMP → promyelocyte → granulocyte: "
            "  CEBPA targets: GCSFR (CSF3R), MPO, lactoferrin, elastase; "
            "  CEBPA represses MYC (stops proliferation during differentiation); "
            "  CEBPA p42 isoform = differentiation; CEBPA p30 = mitogenic; "
            "N-TERMINAL FRAMESHIFT (GERMLINE): "
            "  Destroys TA1 domain → p30 dominant over p42 → MYC repression lost; "
            "  Progenitor over-proliferates but can still partially differentiate; "
            "  Carrier state (one functional allele): partial differentiation preserved; "
            "BIALLELIC (GERMLINE + SOMATIC bZIP): "
            "  bZIP mutation: dominant-negative heterodimerises with WT CEBPA → DNA binding abolished; "
            "  Complete myeloid differentiation block at GMP stage → myeloblast accumulation → AML; "
            "CO-MUTATIONS: GATA2 (20%), WT1 (10%), NPM1 rare (unlike NPM1 which is mutually exclusive with CEBPA)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — CEBPA FAMILIAL AML: "
            "  1. AML WITH BIALLELIC CEBPA MUTATIONS (N-terminal + C-terminal bZIP): "
            "     ~10% of all AML; ELN favourable risk; "
            "     N-terminal frameshift + C-terminal bZIP = classic biallelic hit; "
            "  2. SAME N-TERMINAL FRAMESHIFT IN BUCCAL DNA = GERMLINE: "
            "     germline testing should be reflexly done in ALL biallelic CEBPA AML; "
            "  3. YOUNG AML (30-50y) WITH FAMILY HISTORY OF EARLY-ONSET AML: "
            "     CEBPA germline must be excluded; "
            "  4. RELAPSED CEBPA AML WITH NEW C-TERMINAL bZIP: "
            "     second-hit evolution at relapse; bZIP clonal replacement common; "
            "DDx: DDX41 AML (older, male, normal karyotype, R525H somatic), "
            "RUNX1 FPD/AML (thrombocytopenia), NPM1 AML (NPM1 + FLT3 co-mutations common)"
        ),
        "treatment": (
            "CEBPA AML TREATMENT: "
            "INDUCTION: 7+3; high CR1 rate (>90%); "
            "CONSOLIDATION: "
            "  Somatic biallelic CEBPA: HiDAC × 3 cycles; no HSCT in CR1 (ELN favourable risk); "
            "  GERMLINE biallelic CEBPA: HSCT in CR1 (relapse risk higher in germline); "
            "RELAPSE: re-induction; HSCT if not done; venetoclax combinations under investigation; "
            "DONOR SELECTION: siblings → buccal N-terminal frameshift testing before donation; "
            "  ~50% of siblings carry germline N-terminal frameshift; "
            "FAMILY SURVEILLANCE: all first-degree relatives tested; annual CBC if carrier; "
            "GENETIC COUNSELLING: distinguish germline from somatic; "
            "  Lab: sequence buccal/skin fibroblast DNA alongside tumour"
        ),
        "seed": 2872,
        "n_patients": 40,
    },
    {
        "gene": "ETV6",
        "protein": (
            "ETV6 -- 12p13.2 AD -- 452aa -- "
            "ETS-Variant-Transcription-Factor-6-57kDa-"
            "PNT-Helix-Loop-Helix-ETS-Domain-Haematopoietic-Tumour-Suppressor-"
            "ETV6-Related-Thrombocytopenia-ALL-AML-Predisposition-"
            "OMIM-Gene-600618-Disease-Thrombocytopenia-5-616216"
        ),
        "locus": "12p13.2",
        "protein_size": (
            "452 aa / 57 kDa (ETS variant transcription factor 6; formerly TEL; "
            "domains: N-terminal PNT (pointed) domain — oligomerisation; "
            "central linker; C-terminal ETS domain — DNA binding to GGAA/T ETS motif; "
            "ETV6 acts as transcriptional repressor; "
            "haematopoietic tumour suppressor: ETV6-RUNX1 translocation = most common childhood ALL; "
            "germline LOF mutations: missense (ETS domain p.R369Q/H/C, p.P214L), nonsense, frameshift; "
            "ETS domain mutations: impair DNA binding → LOF + dominant-negative (heterodimerises WT ETV6); "
            "PNT domain mutations: impair oligomerisation; "
            "THROMBOCYTOPENIA: platelet count 30-150×10⁹/L; mild-moderate; lifelong"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — LOF + DOMINANT-NEGATIVE: "
            "  Germline ETS-domain mutations: dominant-negative (mutant heterodimerises with WT ETV6); "
            "  Complete LOF (nonsense/frameshift): haploinsufficiency; "
            "  Penetrance for thrombocytopenia: ~90%; "
            "  Penetrance for haematological malignancy: ~15-35% lifetime; "
            "HAEMATOLOGICAL MALIGNANCY SPECTRUM: "
            "  B-cell ALL (most common; childhood and adult); "
            "  AML; MDS; T-ALL; "
            "  MYELOPROLIFERATIVE syndromes (rare); "
            "THROMBOCYTOPENIA FEATURES: "
            "  Mild-moderate (typically 50-150×10⁹/L); "
            "  Often misdiagnosed as immune thrombocytopenia (ITP) — platelet antibodies absent; "
            "  Platelet function: abnormal dense granule release (similar to storage pool deficiency); "
            "  Large platelets (mild); BM: mild megakaryocyte abnormalities; "
            "FAMILY HISTORY: thrombocytopenia + haematological malignancy in multiple generations"
        ),
        "disease_category": (
            "ETV6-RELATED THROMBOCYTOPENIA AND HAEMATOLOGICAL MALIGNANCY PREDISPOSITION: "
            "THROMBOCYTOPENIA: "
            "  Lifelong mild-moderate; usually not severe; mucocutaneous bleeding episodes; "
            "  Often only diagnosed when AML/ALL occurs in family member → family cascade testing; "
            "B-CELL ALL PREDISPOSITION: "
            "  ALL risk: ~11% by 40 years; "
            "  ETV6-RUNX1 somatic translocation (NOT same as germline ETV6 LOF) = childhood ALL #1 somatic hit; "
            "  Germline ETV6 LOF → distinct biology: somatic second hits vary; "
            "AML: ~5-10% lifetime; "
            "MDS: also described; "
            "SOLID TUMOURS: increased risk suggested (colorectal, breast — limited data); "
            "CLINICAL ALERT: ETV6 germline = exclude as sibling HSCT donor; "
            "  Concomitant thrombocytopenia in donor candidate → ETV6 germline testing"
        ),
        "disease_pathway": (
            "ETV6 → HAEMATOPOIETIC TUMOUR SUPPRESSION: "
            "NORMAL: ETV6 represses target genes in haematopoietic progenitors: "
            "  ETV6 binds ETS motifs → recruits NCoR/HDAC co-repressors → transcriptional silencing; "
            "  Key targets: CCND1 (cyclin D1), MYC, FLT3; "
            "  ETV6 required for HSC maintenance in BM niche; "
            "  ETV6 co-represses FLI1 to allow erythroid/megakaryocyte balance; "
            "GERMLINE ETS MUTATION — DOMINANT-NEGATIVE: "
            "  Mutant ETV6 (ETS domain) dimerises via PNT domain with WT ETV6; "
            "  Dimer cannot bind DNA → WT ETV6 effectively lost → target gene derepression; "
            "  FLT3, CCND1 derepressed → enhanced progenitor proliferation; "
            "SOMATIC SECOND HIT (for malignant transformation): "
            "  del12p (LOH of remaining WT allele), JAK2 V617F, IKZF1 del, PAX5 mutations; "
            "MEGAKARYOCYTE DEFECT: "
            "  ETV6 LOF → megakaryocyte terminal differentiation impaired; "
            "  Proplatelet formation defective → thrombocytopenia"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — ETV6 GERMLINE: "
            "  1. FAMILIAL THROMBOCYTOPENIA + HAEMATOLOGICAL MALIGNANCY (ALL or AML): "
            "     multiple generations; thrombocytopenia precedes malignancy; "
            "     ETV6 germline most likely (alongside RUNX1, ANKRD26); "
            "  2. CHILDHOOD ALL IN PATIENT WITH LIFELONG MILD THROMBOCYTOPENIA: "
            "     ITP misdiagnosis common before ALL diagnosis; "
            "     ETV6 germline testing mandatory in this context; "
            "  3. THROMBOCYTOPENIA + ABNORMAL DENSE GRANULE RELEASE ON PLATELET AGGREGOMETRY: "
            "     dense granule storage pool deficiency-like pattern distinguishes ETV6 from ITP; "
            "  4. ETS-DOMAIN MISSENSE (p.R369Q/H/C) IN FAMILY WITH THROMBOCYTOPENIA + ALL: "
            "     hotspot mutations; "
            "DDx: RUNX1 FPD/AML (dense granule defect, absent second wave, AML > ALL), "
            "ANKRD26 THC2 (thrombocytopenia, AML risk, TPO elevated, RUNX1-FLI1 silencing), "
            "MYH9-RD (giant platelets, Döhle bodies — no malignancy predisposition)"
        ),
        "treatment": (
            "ETV6-RELATED THROMBOCYTOPENIA AND MALIGNANCY TREATMENT: "
            "THROMBOCYTOPENIA: "
            "  No treatment usually needed (mild platelet counts >50×10⁹/L); "
            "  Tranexamic acid / DDAVP for bleeding/surgery; "
            "  Platelet transfusion pre-surgery if <50×10⁹/L; "
            "  Avoid ITP-directed treatments (steroids, rituximab) — thrombocytopenia is not immune; "
            "ALL TREATMENT: standard paediatric/adult ALL protocols; "
            "  ETV6-germline ALL: similar outcome to sporadic ALL with standard therapy; "
            "AML TREATMENT: 7+3 standard induction; HSCT in CR1 for MDS/AML; "
            "HSCT DONOR EXCLUSION: all siblings tested before donation; "
            "SURVEILLANCE: "
            "  Annual CBC + review; BM biopsy if new cytopenias; "
            "  Haemato-oncology review annually from diagnosis; "
            "FAMILY CASCADE: first-degree relatives tested; children from age 5-10 years"
        ),
        "seed": 2873,
        "n_patients": 40,
    },
    {
        "gene": "SAMD9",
        "protein": (
            "SAMD9 -- 7q21.2 AD -- 1589aa -- "
            "Sterile-Alpha-Motif-Domain-9-170kDa-"
            "IFN-Inducible-Antiviral-Tumour-Suppressor-"
            "MIRAGE-Syndrome-Monosomy7-Pancytopenia-Growth-Retardation-"
            "OMIM-Gene-610456-Disease-MIRAGE-617053"
        ),
        "locus": "7q21.2",
        "protein_size": (
            "1589 aa / 170 kDa (sterile alpha motif domain-containing protein 9; "
            "SAM domain at N-terminus (protein-protein interaction); "
            "large central domain — function: inhibits cell proliferation + protein synthesis; "
            "localises to endosomes/multivesicular bodies; "
            "IFN-inducible antiviral protein (restricts poxvirus replication); "
            "GOF mutations (gain-of-function of ANTIPROLIFERATIVE activity): "
            "  Enhanced proliferation inhibition → severe growth retardation; "
            "  BM progenitors also inhibited → pancytopenia; "
            "  Monosomy 7 in BM = ADAPTATION (somatic rescue): "
            "    BM progenitor loses one SAMD9 GOF allele (on chr7) → selective advantage; "
            "    Monosomy 7 clone expands → apparent haematological improvement; "
            "    BUT monosomy 7 itself is pre-leukaemic (MDS risk)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — GAIN-OF-FUNCTION (ANTIPROLIFERATIVE GOF): "
            "  De novo predominantly (new mutation in proband); "
            "  GOF = enhanced antiproliferative activity → growth retardation + BM failure; "
            "  Somatic reversion (UPD, monosomy 7, intragenic revertants): "
            "    BM progenitors with somatic correction have SELECTIVE ADVANTAGE; "
            "    Monosomy 7 revertants expand → pancytopenia may IMPROVE over time; "
            "    BUT monosomy 7 = secondary MDS risk (5-10 year horizon); "
            "MIRAGE SYNDROME (full phenotype): "
            "  M = Myelodysplasia (BM failure, monosomy 7) "
            "  I = Infections (severe, recurrent, opportunistic) "
            "  R = Restriction of growth (IUGR + postnatal severe short stature) "
            "  A = Adrenal hypoplasia (adrenal insufficiency — life-threatening at diagnosis) "
            "  G = Genital abnormalities (XY → ambiguous/female external genitalia) "
            "  E = Enteropathy (malabsorption, diarrhoea)"
        ),
        "disease_category": (
            "MIRAGE SYNDROME — MULTI-SYSTEM SAMD9 GOF DISORDER: "
            "ADRENAL INSUFFICIENCY: often life-threatening first presentation; "
            "  Primary adrenal hypoplasia (aplastic/hypoplastic adrenal cortex); "
            "  Cortisol + aldosterone deficiency → Addisonian crisis; "
            "  Hydrocortisone + fludrocortisone replacement: life-saving; "
            "PANCYTOPENIA / BONE MARROW FAILURE: "
            "  BM: hypocellular + dysplastic; monosomy 7 clone emerges; "
            "  Monosomy 7 → MDS in 30-50% of survivors at 5-10 years; "
            "SEVERE GROWTH RETARDATION: IUGR + postnatal; GH insensitivity component; "
            "INFECTIONS: combined immunodeficiency (T/B/NK defects); NTM, candida, CMV; "
            "ENTEROPATHY: villous atrophy-like; TPN dependency common; "
            "PROGNOSIS: poor without HSCT (MDS risk + infections); "
            "  Early death (infancy) from adrenal crisis if unrecognised; "
            "  Survivors to mid-childhood: MDS surveillance critical"
        ),
        "disease_pathway": (
            "SAMD9 GOF → ANTIPROLIFERATIVE PATHWAY: "
            "NORMAL: SAMD9 restricts viral replication (poxvirus) and cell proliferation; "
            "  IFN-γ induces SAMD9; SAMD9 inhibits translation via eIF4A interaction; "
            "GOF MUTATION: enhanced antiproliferative + antitranslational activity; "
            "  ALL cells affected: soma (growth retardation, organ hypoplasia) + "
            "    BM progenitors (pancytopenia) + adrenal (hypoplasia) + gut epithelium (enteropathy); "
            "SOMATIC ADAPTATION IN BM: "
            "  Chr7 monosomy: loss of SAMD9-GOF allele (on chr7) → BM progenitor escapes growth inhibition; "
            "  Monosomy 7 clone outcompetes normal BM → cytopenias may improve transiently; "
            "  BUT monosomy 7 → NF1, KMT2C, other tumour suppressors lost → MDS/AML risk; "
            "INTRAGENIC REVERTANTS: "
            "  BM-restricted somatic intragenic mutations that neutralise GOF → escape; "
            "  Multiple revertant clones with different corrective mutations coexist"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — SAMD9 (MIRAGE): "
            "  1. NEWBORN: ambiguous genitalia (XY) + adrenal insufficiency + IUGR: "
            "     MIRAGE triad → SAMD9/SAMD9L sequencing mandatory (alongside SF1, NR0B1); "
            "  2. MONOSOMY 7 IN BM OF CHILD WITH PANCYTOPENIA + GROWTH RETARDATION: "
            "     monosomy 7 as somatic adaptation (not primary) → SAMD9/SAMD9L germline testing; "
            "  3. IMPROVING CYTOPENIAS + MONOSOMY 7 CLONAL EXPANSION: "
            "     paradoxical improvement (monosomy 7 revertants give BM advantage) → SAMD9; "
            "  4. MIRAGE ACRONYM: Myelodysplasia + Infections + Restriction + Adrenal + Genital + Enteropathy; "
            "DDx: SAMD9L (ataxia-pancytopenia — CEREBELLAR ATAXIA present, no adrenal hypoplasia), "
            "Fanconi anaemia (biallelic FANCA etc., DEB/MMC positive), "
            "46,XY DSD (SF1, WT1 — BM normal)"
        ),
        "treatment": (
            "MIRAGE SYNDROME / SAMD9 TREATMENT: "
            "ADRENAL INSUFFICIENCY (EMERGENCY): hydrocortisone IV stat + fludrocortisone; "
            "  Stress dosing critical; medic-alert bracelet mandatory; "
            "BONE MARROW FAILURE / MDS: "
            "  HSCT: only option for BM failure or MDS; "
            "  Timing: before MDS transformation; before severe infections; "
            "  Monosomy 7 alone does not mandate immediate HSCT — monitor; "
            "  MDS features (blasts >5%, dysplasia) → HSCT urgently; "
            "INFECTIONS: antimicrobial prophylaxis (bacterial + fungal + PCP + antiviral); "
            "NUTRITION: TPN / NG feeds for enteropathy; "
            "GROWTH: GH therapy (limited response due to GH insensitivity); "
            "SURVEILLANCE: "
            "  Quarterly CBC + differential; "
            "  BM biopsy ± karyotype every 6-12 months; "
            "  Monitor monosomy 7 clone size by FISH/SNP array"
        ),
        "seed": 2874,
        "n_patients": 40,
    },
    {
        "gene": "SAMD9L",
        "protein": (
            "SAMD9L -- 7q21.2 AD -- 1589aa -- "
            "Sterile-Alpha-Motif-Domain-9-Like-170kDa-"
            "Paralogue-SAMD9-Neurological-Haematopoietic-Tumour-Suppressor-"
            "Ataxia-Pancytopenia-Syndrome-Monosomy7-Cerebellar-Ataxia-"
            "OMIM-Gene-610495-Disease-Ataxia-Pancytopenia-159550"
        ),
        "locus": "7q21.2",
        "protein_size": (
            "1589 aa / 170 kDa (sterile alpha motif domain-containing protein 9-like; "
            "SAMD9L is paralogue of SAMD9 — 58% amino acid identity; "
            "tandemly duplicated on chr7q21.2 (SAMD9 then SAMD9L ~80kb apart); "
            "same SAM domain + large central antiproliferative domain; "
            "SAMD9L additionally expressed in cerebellum (Purkinje cells) — explains ataxia; "
            "SAMD9 not significantly expressed in cerebellum; "
            "GOF mutations: same gain-of-antiproliferative-function mechanism as SAMD9; "
            "KEY CLINICAL DISTINCTION FROM SAMD9: "
            "  SAMD9L → ATAXIA (cerebellar Purkinje cell dysfunction); "
            "  SAMD9 → ADRENAL HYPOPLASIA + genital abnormalities; "
            "  Both → pancytopenia + monosomy 7 somatic adaptation"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — GAIN-OF-FUNCTION (ANTIPROLIFERATIVE): "
            "  De novo OR inherited (variable expressivity within families); "
            "  Parents of proband may have mild ataxia only (no BM failure); "
            "  Penetrance: high for ataxia (~90%); variable for pancytopenia; "
            "ATAXIA-PANCYTOPENIA SYNDROME (APS): "
            "  Cerebellar ataxia: onset 1st-3rd decade; progressive; "
            "  Cerebellar vermis + hemispheric atrophy on MRI; "
            "  Pancytopenia: variable onset; episodic; monosomy 7 adaptation explains episodic improvement; "
            "  MDS: ~30-40% develop MDS within 5-10 years; "
            "SOMATIC ADAPTATION: same monosomy 7 / intragenic reversion as SAMD9; "
            "  Monosomy 7 = BM adaptation → partial correction of cytopenias; "
            "  Neurological features NOT corrected by monosomy 7 (CNS progenitors less able to adapt)"
        ),
        "disease_category": (
            "ATAXIA-PANCYTOPENIA SYNDROME — NEUROLOGICAL + HAEMATOLOGICAL: "
            "CEREBELLAR ATAXIA: "
            "  Gait ataxia → wheelchair in severe cases; dysarthria; nystagmus; "
            "  Cerebellar MRI: vermis + hemisphere atrophy (Purkinje cell loss); "
            "  Progressive — no disease-modifying neurological treatment; "
            "PANCYTOPENIA / MDS: "
            "  Pancytopenia variable (can be mild-moderate initially); "
            "  Monosomy 7 BM clone → cytopenias may fluctuate; "
            "  MDS (monosomy 7 + dysplasia) → AML risk; "
            "  HSCT corrects BM/haematological component; ataxia NOT improved by HSCT; "
            "IMMUNODEFICIENCY: infections (less severe than MIRAGE/SAMD9); "
            "CLINICAL ALERT: "
            "  Progressive ataxia + pancytopenia + monosomy 7 = SAMD9L until proven otherwise; "
            "  Most commonly misdiagnosed as Fanconi or hereditary ataxia (Friedreich etc.)"
        ),
        "disease_pathway": (
            "SAMD9L GOF — CNS + BM ANTIPROLIFERATIVE: "
            "SAMD9L EXPRESSION: ubiquitous + high in cerebellum (unlike SAMD9); "
            "GOF MECHANISM: same as SAMD9 — enhanced antiproliferative / antitranslational; "
            "CEREBELLAR SPECIFICITY: "
            "  Purkinje cells highly SAMD9L-expressing → most vulnerable to GOF; "
            "  Purkinje cell death (apoptosis) → cerebellar atrophy → ataxia; "
            "  CNS cells DO NOT undergo monosomy 7 reversion (post-mitotic neurons); "
            "  Hence ataxia is PROGRESSIVE and IRREVERSIBLE; "
            "BM ADAPTATION (same as SAMD9): "
            "  Monosomy 7 revertants expand → partial pancytopenia correction; "
            "  Secondary MDS from monosomy 7 (NF1 loss, KMT2C loss on chr7); "
            "CO-DELETION SAMD9+SAMD9L: "
            "  Some GOF mutations affect both paralogues (if large deletion at 7q21.2); "
            "  Combined MIRAGE + APS phenotype (adrenal + ataxia)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — SAMD9L (ATAXIA-PANCYTOPENIA): "
            "  1. CEREBELLAR ATAXIA + PANCYTOPENIA + MONOSOMY 7 IN BM: "
            "     TRIAD = SAMD9L PATHOGNOMONIC; essentially no other diagnosis; "
            "  2. MRI: CEREBELLAR VERMIS ATROPHY in child/young adult with pancytopenia: "
            "     ataxia in MDS/BM failure context → SAMD9L first; "
            "  3. MONOSOMY 7 AS SOMATIC ADAPTATION (improving cytopenias in ataxic patient): "
            "     paradoxical improvement while monosomy 7 expands → SAMD9L; "
            "  4. FAMILY: PARENT WITH MILD LATE-ONSET ATAXIA + CHILD WITH SEVERE PANCYTOPENIA: "
            "     variable expressivity within family → SAMD9L GOF; "
            "DDx: SAMD9 (MIRAGE — adrenal hypoplasia + ambiguous genitalia, no ataxia), "
            "Friedreich ataxia (AR, frataxin, no BM involvement), "
            "Fanconi anaemia (DEB/MMC positive, FANC gene biallelic, skeletal anomalies)"
        ),
        "treatment": (
            "ATAXIA-PANCYTOPENIA / SAMD9L TREATMENT: "
            "NEUROLOGICAL (ATAXIA): no disease-modifying treatment; "
            "  Physiotherapy + occupational therapy; speech therapy (dysarthria); "
            "  Ankle-foot orthoses; mobility aids; "
            "  HSCT does NOT improve ataxia; "
            "BONE MARROW FAILURE / MDS: "
            "  HSCT: corrects BM component (not neurological); "
            "  Timing: MDS with high-risk features or progressive pancytopenia; "
            "  Monosomy 7 alone: monitor; intervene if MDS blasts >5% or worsening cytopenias; "
            "INFECTIONS: standard antimicrobial prophylaxis; "
            "SURVEILLANCE: "
            "  Quarterly CBC; BM biopsy + karyotype every 6-12 months; "
            "  Neurological assessment annually; MRI brain every 2-3 years; "
            "  Monitor monosomy 7 clone dynamics by FISH/SNP array; "
            "COUNSELLING: explain that HSCT corrects BM but NOT ataxia — informed consent critical"
        ),
        "seed": 2875,
        "n_patients": 40,
    },
    {
        "gene": "PTPN11",
        "protein": (
            "PTPN11 -- 12q24.13 AD -- 593aa -- "
            "SHP2-Tyrosine-Phosphatase-Non-Receptor-Type-11-68kDa-"
            "RAS-MAPK-Signal-Amplifier-"
            "Noonan-Syndrome-JMML-GOF-Gain-Of-Function-"
            "OMIM-Gene-176876-Disease-Noonan-163950-JMML-607785"
        ),
        "locus": "12q24.13",
        "protein_size": (
            "593 aa / 68 kDa (SH2-containing protein tyrosine phosphatase 2 / SHP2; "
            "domains: N-SH2 (self-inhibitory); C-SH2 (substrate recruitment); PTP (phosphatase); "
            "autoinhibition: N-SH2 blocks PTP active site in basal state; "
            "activation: phosphotyrosine peptides bind N-SH2 → open conformation → PTP active; "
            "NORMAL FUNCTION: amplifies RAS-MAPK/PI3K signalling downstream of RTKs (FGFR, MET, EGFR); "
            "GOF MUTATIONS (Noonan/JMML): relieve autoinhibition → constitutive PTP activity → "
            "  hyperactivated RAS-ERK signalling; "
            "p.E76K (JMML hotspot — severe GOF); p.D61G (Noonan); p.Q79R (JMML + NS); "
            "JMML mutations: more activating than Noonan mutations (allele-specific severity)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — GAIN-OF-FUNCTION: "
            "NOONAN SYNDROME: "
            "  ~50% of all Noonan syndrome caused by PTPN11 GOF; "
            "  De novo OR inherited from mildly affected parent; "
            "  Features: short stature, craniofacial (hypertelorism, ptosis, low-set ears), "
            "    congenital heart (pulmonary valve stenosis 50-60%, HCM 20-30%), "
            "    bleeding tendency (low F11, VWD, thrombocytopenia), "
            "    cryptorchidism (males); "
            "JMML — JUVENILE MYELOMONOCYTIC LEUKEMIA: "
            "  25% of JMML caused by germline PTPN11 mutations (Noonan-JMML); "
            "  JMML = rare myeloproliferative/myelodysplastic neoplasm of childhood (median age 2y); "
            "  Noonan-JMML: paradoxically better prognosis than somatic PTPN11 JMML; "
            "    Some Noonan-JMML resolves spontaneously (transient JMML); "
            "  Somatic PTPN11 JMML (D61G, A72V): aggressive, requires HSCT"
        ),
        "disease_category": (
            "NOONAN SYNDROME — RASOPATHY WITH JMML PREDISPOSITION: "
            "NOONAN SYNDROME FEATURES: "
            "  CARDIAC: pulmonary valve stenosis (balloon valvuloplasty); HCM (beta-blocker/myectomy); "
            "  SHORT STATURE: GH therapy (FDA-approved for Noonan); "
            "  BLEEDING: factor replacement + DDAVP; avoid NSAIDs pre-procedure; "
            "  LEARNING: mild ID in 25%; "
            "JMML IN NOONAN: "
            "  Median age 6 months to 2 years; monocytosis + hepatosplenomegaly; "
            "  Noonan-JMML: ~20-30% resolve spontaneously (watch-and-wait justified); "
            "  Persistent or progressive Noonan-JMML: HSCT; "
            "  Somatic PTPN11 JMML: aggressive → HSCT mandatory; "
            "SOLID TUMOUR RISK: Noonan → neuroblastoma (rare); RAS GOF; "
            "CLINICAL ALERT: ALL Noonan patients with organomegaly + monocytosis: CBC + BM urgently"
        ),
        "disease_pathway": (
            "PTPN11 GOF → HYPERACTIVATED RAS-MAPK: "
            "NORMAL SHP2 CYCLE: "
            "  RTK activation → pTyr residues bind SHP2 SH2 domains → open conformation → PTP active; "
            "  SHP2 dephosphorylates Sprouty (RAS inhibitor) → RAS-GTP rises → ERK/MAPK activated; "
            "  Signal terminated by SHP2 returning to closed conformation; "
            "GOF MUTATIONS: "
            "  N-SH2 mutations (D61G, E76K) → cannot fold back → PTP constitutively open → "
            "    continuous Sprouty dephosphorylation → RAS-GTP high → ERK/MEK constitutive; "
            "  JMML mutations > Noonan mutations in PTP activity (explains severity spectrum); "
            "MYELOID PROGENITOR HYPERPROLIFERATION: "
            "  GM-CSF hypersensitivity (colony formation at zero GM-CSF): JMML diagnostic hallmark; "
            "  Myeloid progenitors: RAS-ERK → MYC → proliferation; "
            "  Monocyte overproduction → monocytosis + hepatosplenomegaly; "
            "TRAMETINIB / MEK INHIBITION: experimental (blocks ERK downstream of RAS)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — PTPN11 NOONAN + JMML: "
            "  1. NOONAN PHENOTYPE + MONOCYTOSIS + HEPATOSPLENOMEGALY IN INFANT: "
            "     NOONAN-JMML — diagnostic; PTPN11 germline testing + BM biopsy; "
            "  2. GM-CSF HYPERSENSITIVITY: spontaneous myeloid colony formation at zero GM-CSF: "
            "     JMML PATHOGNOMONIC — required for JMML diagnosis (WHO criteria); "
            "  3. HbF ELEVATED (>10%) IN CHILD WITH MONOCYTOSIS: "
            "     foetal haemoglobin switches back on in JMML — pathognomonic clue; "
            "  4. NOONAN CHILD WITH SPONTANEOUSLY RESOLVING JMML: "
            "     Noonan-JMML transient remission — unique to germline PTPN11 Noonan; "
            "DDx: CBL JMML (Noonan-like, CBL mutation — CBL protein null, UPD 11q23), "
            "KRAS/NRAS somatic JMML (no Noonan phenotype), "
            "NF1 JMML (café-au-lait, NF1 biallelic)"
        ),
        "treatment": (
            "NOONAN-PTPN11 + JMML TREATMENT: "
            "CARDIAC: "
            "  Pulmonary stenosis: balloon valvuloplasty; "
            "  HCM: beta-blocker; myectomy if LVOT obstruction; "
            "GH THERAPY: approved for Noonan short stature; "
            "BLEEDING: DDAVP pre-procedure + tranexamic acid; avoid aspirin; "
            "JMML — NOONAN: "
            "  Watch and wait (6-12 months) if mild and stable — spontaneous remission occurs; "
            "  Progressive JMML: HSCT (only cure); "
            "  Standard JMML conditioning: busulfan + cyclophosphamide + melphalan; "
            "  Post-HSCT relapse: ~35%; second HSCT; "
            "JMML — SOMATIC PTPN11 (D61G/A72V): "
            "  Aggressive; early HSCT; no watch-and-wait; "
            "MEK INHIBITOR (TRAMETINIB): investigational; clinical trials recruiting; "
            "  Early remission induction before HSCT — promising data; "
            "SURVEILLANCE: regular cardiac + growth + BM monitoring per RASopathy protocol"
        ),
        "seed": 2876,
        "n_patients": 40,
    },
    {
        "gene": "CBL",
        "protein": (
            "CBL -- 11q23.3 AD -- 906aa -- "
            "Casitas-B-Lineage-Lymphoma-Ubiquitin-E3-Ligase-"
            "RTK-Degradation-RAS-Regulation-"
            "Noonan-Like-Syndrome-JMML-Predisposition-UPD11q-Somatic-"
            "OMIM-Gene-165360-Disease-Noonan-Like-CBL-613563"
        ),
        "locus": "11q23.3",
        "protein_size": (
            "906 aa / 100 kDa (Casitas B-lineage lymphoma proto-oncogene; "
            "domains: TKB (tyrosine kinase binding) domain; "
            "RING finger domain (ubiquitin E3 ligase activity); "
            "C-terminal proline-rich domain + ubiquitin-associated domain; "
            "NORMAL FUNCTION: E3 ubiquitin ligase targeting activated RTKs (EGFR, KIT, FLT3) "
            "  for proteasomal degradation → terminates RTK signalling; "
            "ALSO: scaffolding for PI3K, SRC, GRB2 — both adaptor and ligase; "
            "GERMLINE LOF MUTATIONS: RING finger domain (splice, missense Y371H/C, W408C): "
            "  Loss of ubiquitin ligase → RTK not degraded → prolonged RAS-MAPK activation; "
            "BIALLELIC HIT IN TUMOUR: germline LOF (first hit) + UPD11q (acquired isodisomy): "
            "  UPD11q23 → homozygous LOF → complete CBL null → JMML"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — LOF GERMLINE + UPD SOMATIC BIALLELIC MECHANISM: "
            "GERMLINE CBL LOF (monoallelic): "
            "  Noonan-like syndrome features (mild): dysmorphic facies, short stature, PDA/ASD; "
            "  No JMML at germline stage (monoallelic CBL insufficient); "
            "SOMATIC SECOND HIT: "
            "  Acquired uniparental disomy (UPD) of 11q23 in myeloid progenitor: "
            "    LOH → homozygous CBL LOF → complete E3 ligase null → unrestrained RTK signalling; "
            "  UPD11q = essentially all JMML in germline CBL carriers; "
            "JMML RISK: ~20% of germline CBL LOF carriers develop JMML; "
            "  JMML onset: infancy to 3 years; "
            "SPONTANEOUS REMISSION: Noonan-CBL JMML also has high spontaneous remission rate (~30-50%); "
            "  Similar to Noonan-PTPN11; "
            "LATE VASCULITIS / CANCER: CBL germline carriers → vasculitis (inflammatory skin), "
            "  longer-term solid tumour risk (limited data)"
        ),
        "disease_category": (
            "NOONAN-LIKE SYNDROME WITH CBL JMML PREDISPOSITION: "
            "NOONAN-LIKE FEATURES: "
            "  Moderate dysmorphic facies (less severe than PTPN11 Noonan); "
            "  Cardiac (PDA, ASD — less frequent than pulmonary stenosis); "
            "  Short stature (mild); learning difficulties (mild); "
            "  Café-au-lait macules (sometimes — DDx with NF1); "
            "CBL JMML: "
            "  Similar to PTPN11-JMML: monocytosis, HbF elevation, hepatosplenomegaly; "
            "  GM-CSF hypersensitivity (like all JMML); "
            "  Spontaneous remission: ~30-50% (watch-and-wait in stable Noonan-CBL); "
            "VASCULITIS: cutaneous vasculitis in some CBL carriers (not seen in PTPN11 Noonan); "
            "  Leukocytoclastic vasculitis on skin biopsy; "
            "LATE-ONSET MALIGNANCY RISK: uncertain; surveillance recommended"
        ),
        "disease_pathway": (
            "CBL LOF → PROLONGED RTK / RAS-MAPK SIGNALLING: "
            "NORMAL CBL E3 LIGASE CYCLE: "
            "  Activated RTK (KIT, FLT3, EGFR) → autophosphorylation → CBL-TKB binds pTyr → "
            "    CBL-RING ubiquitinates RTK → proteasomal degradation → signal termination; "
            "CBL LOF: "
            "  Monoallelic: partial RTK degradation impairment; compensated; "
            "  Biallelic (UPD11q): complete CBL null → RTK not degraded → "
            "    KIT (mast cell/myeloid) constitutively signalling → RAS-GTP high → "
            "    MAPK/PI3K → myeloid progenitor hyperproliferation → JMML; "
            "KIT AXIS: "
            "  KIT (stem cell factor receptor) on myeloid progenitors: "
            "    CBL null → KIT not degraded → continuous myeloid growth signal; "
            "  SCF/KIT → CBL → KIT degradation: this axis most important in myeloid compartment; "
            "INTERACTION WITH PTPN11: "
            "  Both pathways converge on RAS-GTP; "
            "  CBL: RTK duration (upstream of RAS); "
            "  PTPN11/SHP2: Sprouty dephosphorylation (negative RAS regulator removed)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — CBL NOONAN-JMML: "
            "  1. NOONAN-LIKE PHENOTYPE + JMML + UPD11q23 IN TUMOUR: "
            "     UPD11q23 in JMML clone = virtually PATHOGNOMONIC for germline CBL; "
            "     SNP array or FISH showing LOH at 11q23 in BM → CBL germline must be tested; "
            "  2. CBL JMML + CUTANEOUS VASCULITIS: "
            "     vasculitis (leukocytoclastic) is unique to CBL — not seen in PTPN11/KRAS/NRAS/NF1 JMML; "
            "  3. CAFÉ-AU-LAIT MACULES + JMML (MIMICKING NF1-JMML): "
            "     CBL Noonan-like can have CALMs; NF1 JMML has NF1 biallelic LOF; "
            "     distinguish by NF1 sequencing + SPRED1 + CBL sequencing; "
            "  4. SPONTANEOUS JMML REMISSION IN NOONAN-LIKE CHILD: "
            "     spontaneous remission → germline RASopathy (PTPN11 or CBL) vs somatic JMML; "
            "DDx: PTPN11-JMML (no vasculitis, no UPD11q, pulmonary stenosis commoner), "
            "NF1-JMML (NF1 biallelic, café-au-lait prominent, no Noonan), "
            "KRAS/NRAS somatic (no Noonan phenotype)"
        ),
        "treatment": (
            "CBL NOONAN-JMML TREATMENT: "
            "JMML: "
            "  Watch and wait (6-12 months) in mild stable Noonan-CBL JMML: "
            "    Spontaneous remission in ~30-50%; monitor weekly CBC; "
            "  Progressive CBL JMML: HSCT; "
            "    Conditioning: busulfan + cyclophosphamide + melphalan (standard JMML); "
            "  Post-HSCT: ~30% relapse; second HSCT; "
            "MEK INHIBITORS (TRAMETINIB): investigational; may allow bridging to HSCT; "
            "CARDIAC: ASD/PDA repair as indicated; "
            "VASCULITIS: corticosteroids; "
            "GROWTH: GH (less evidence than for PTPN11 Noonan); "
            "SURVEILLANCE: "
            "  All germline CBL carriers: annual CBC from birth; "
            "  BM biopsy if monocytosis / splenomegaly; "
            "  SNP array BM to detect UPD11q emergence; "
            "  Annual dermatology review (vasculitis); "
            "GENETIC COUNSELLING: AD inheritance; UPD-mediated JMML biallelic model explained"
        ),
        "seed": 2877,
        "n_patients": 40,
    },
]


def _make_patients(gene_data: dict) -> list:
    """Generate synthetic patient records for a gene."""
    rng = random.Random(gene_data["seed"])
    gene = gene_data["gene"]
    n = gene_data["n_patients"]

    # Gene-specific clinical distributions
    GENE_PARAMS = {
        "GATA2": {
            "age_range": (15, 55), "f_pct": 0.50,
            "hb_range": (7.0, 12.5), "plt_range": (30, 180),
            "monocyte_range": (0.0, 0.02),  # near-absent in MonoMAC
            "primary_dx": ["MDS-monosomy7", "MonoMAC", "AML", "Emberger", "DCML-deficiency"],
            "treatment": ["HSCT", "NTM prophylaxis", "antifungal", "antiviral", "surveillance"],
        },
        "DDX41": {
            "age_range": (55, 80), "f_pct": 0.25,
            "hb_range": (6.0, 11.0), "plt_range": (20, 120),
            "monocyte_range": (0.2, 1.2),
            "primary_dx": ["AML-normal-karyotype", "MDS-sAML", "de-novo-AML"],
            "treatment": ["7+3 induction", "HSCT in CR1", "HiDAC consolidation", "surveillance"],
        },
        "CEBPA": {
            "age_range": (25, 65), "f_pct": 0.50,
            "hb_range": (6.5, 11.5), "plt_range": (25, 130),
            "monocyte_range": (0.2, 1.5),
            "primary_dx": ["AML-biallelic-CEBPA", "AML-M1", "AML-M2", "familial-AML"],
            "treatment": ["7+3 induction", "HiDAC x3", "HSCT-germline", "surveillance"],
        },
        "ETV6": {
            "age_range": (5, 65), "f_pct": 0.55,
            "hb_range": (9.0, 13.5), "plt_range": (30, 150),
            "monocyte_range": (0.2, 0.8),
            "primary_dx": ["thrombocytopenia-ETV6", "B-ALL", "AML", "MDS"],
            "treatment": ["surveillance", "tranexamic acid", "DDAVP", "ALL chemotherapy", "HSCT"],
        },
        "SAMD9": {
            "age_range": (0, 10), "f_pct": 0.50,
            "hb_range": (5.0, 10.0), "plt_range": (20, 100),
            "monocyte_range": (0.05, 0.5),
            "primary_dx": ["MIRAGE-syndrome", "MDS-monosomy7", "adrenal-insufficiency", "pancytopenia"],
            "treatment": ["hydrocortisone", "HSCT", "antimicrobial prophylaxis", "TPN", "surveillance"],
        },
        "SAMD9L": {
            "age_range": (2, 35), "f_pct": 0.50,
            "hb_range": (6.0, 11.0), "plt_range": (25, 110),
            "monocyte_range": (0.1, 0.6),
            "primary_dx": ["ataxia-pancytopenia", "MDS-monosomy7", "cerebellar-ataxia", "pancytopenia"],
            "treatment": ["HSCT (BM component)", "physiotherapy", "surveillance", "antimicrobial prophylaxis"],
        },
        "PTPN11": {
            "age_range": (0, 5), "f_pct": 0.50,
            "hb_range": (7.0, 12.0), "plt_range": (40, 200),
            "monocyte_range": (1.5, 8.0),  # monocytosis in JMML
            "primary_dx": ["Noonan-JMML", "Noonan-syndrome", "JMML", "transient-JMML"],
            "treatment": ["watch-and-wait", "HSCT", "trametinib (investigational)", "cardiac surgery"],
        },
        "CBL": {
            "age_range": (0, 4), "f_pct": 0.50,
            "hb_range": (7.5, 12.5), "plt_range": (50, 220),
            "monocyte_range": (1.0, 7.0),
            "primary_dx": ["CBL-JMML", "Noonan-like-CBL", "JMML-UPD11q", "vasculitis-CBL"],
            "treatment": ["watch-and-wait", "HSCT", "corticosteroids (vasculitis)", "surveillance"],
        },
    }

    params = GENE_PARAMS.get(gene, {
        "age_range": (10, 70), "f_pct": 0.50,
        "hb_range": (7.0, 12.0), "plt_range": (30, 200),
        "monocyte_range": (0.1, 1.0),
        "primary_dx": ["haematological malignancy"],
        "treatment": ["HSCT", "surveillance"],
    })

    patients = []
    for i in range(n):
        sex = "F" if rng.random() < params["f_pct"] else "M"
        age = rng.randint(*params["age_range"])
        hb = round(rng.uniform(*params["hb_range"]), 1)
        plt_k = rng.randint(*params["plt_range"])
        mono = round(rng.uniform(*params["monocyte_range"]), 3)
        dx = rng.choice(params["primary_dx"])
        tx = rng.choice(params["treatment"])
        hsct = "Yes" if "HSCT" in tx or rng.random() < 0.45 else "No"
        outcome_opts = ["remission", "stable-MDS", "alive-post-HSCT", "AML-transformed", "deceased", "partial-response"]
        outcome = rng.choice(outcome_opts[:4])
        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "age_at_dx": age,
            "sex": sex,
            "hb_gdl": hb,
            "platelets_k": plt_k,
            "monocytes_abs": mono,
            "primary_dx": dx,
            "treatment": tx,
            "hsct": hsct,
            "outcome": outcome,
        })
    return patients


def generate_overview() -> dict:
    all_patients = []
    gene_summaries = []

    for g in ATLAS_GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        n = len(pts)
        hsct_n = sum(1 for p in pts if p["hsct"] == "Yes")
        rem_n = sum(1 for p in pts if p["outcome"] == "remission")
        gene_summaries.append({
            "gene": g["gene"],
            "locus": g["locus"],
            "protein_summary": g["protein"].split("--")[2].strip() if "--" in g["protein"] else g["gene"],
            "n_patients": n,
            "median_age_dx": sorted([p["age_at_dx"] for p in pts])[n // 2],
            "pct_female": round(sum(1 for p in pts if p["sex"] == "F") / n * 100, 1),
            "mean_hb_gdl": round(sum(p["hb_gdl"] for p in pts) / n, 1),
            "mean_platelets_k": round(sum(p["platelets_k"] for p in pts) / n),
            "pct_hsct": round(hsct_n / n * 100, 1),
            "pct_remission": round(rem_n / n * 100, 1),
        })

    total = len(all_patients)
    return {
        "atlas": "Hereditary-AML-MDS-JMML-Predisposition-Atlas",
        "subtitle": (
            "Complete 8-Gene Germline AML / MDS / ALL / JMML Predisposition Reference — "
            "GATA2-DDX41-CEBPA-ETV6-SAMD9-SAMD9L-PTPN11-CBL"
        ),
        "total_patients": total,
        "seeds": "2870-2877",
        "n_genes": len(ATLAS_GENES),
        "gene_summaries": gene_summaries,
        "clinical_categories": {
            "AML_predisposition": ["GATA2", "DDX41", "CEBPA"],
            "ALL_predisposition": ["ETV6"],
            "MDS_monosomy7": ["GATA2", "SAMD9", "SAMD9L"],
            "JMML": ["PTPN11", "CBL"],
            "thrombocytopenia_plus_malignancy": ["ETV6"],
            "neurological_plus_haematological": ["SAMD9L"],
            "multi_system_syndromic": ["GATA2", "SAMD9", "SAMD9L", "PTPN11", "CBL"],
        },
        "key_clinical_alerts": [
            "DDX41 AML: exclude sibling as donor (50% also carry germline LOF)",
            "CEBPA AML: reflex germline testing (buccal) in ALL biallelic CEBPA AML",
            "GATA2 MonoMAC: near-absent monocytes + NTM = PATHOGNOMONIC — test germline",
            "SAMD9 MIRAGE: adrenal crisis at birth if missed — neonatal adrenal screen",
            "SAMD9L ataxia-pancytopenia: HSCT corrects BM NOT ataxia — critical counselling",
            "PTPN11/CBL Noonan-JMML: watch-and-wait justified (spontaneous remission 20-50%)",
            "Monosomy 7 in BM of child with growth failure = SAMD9/SAMD9L until proven otherwise",
        ],
        "pathway_summary": {
            "JAK-STAT": [],
            "RAS-MAPK": ["PTPN11", "CBL"],
            "Transcription_factor_LOF": ["GATA2", "CEBPA", "ETV6"],
            "RNA_splicing_innate_immune": ["DDX41"],
            "Antiproliferative_GOF": ["SAMD9", "SAMD9L"],
        },
    }


def generate_breakdown() -> dict:
    breakdown = {}
    for g in ATLAS_GENES:
        pts = _make_patients(g)
        n = len(pts)
        dx_counts = {}
        for p in pts:
            dx_counts[p["primary_dx"]] = dx_counts.get(p["primary_dx"], 0) + 1
        tx_counts = {}
        for p in pts:
            tx_counts[p["treatment"]] = tx_counts.get(p["treatment"], 0) + 1
        breakdown[g["gene"]] = {
            "gene": g["gene"],
            "locus": g["locus"],
            "protein": g["protein"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "n_patients": n,
            "dx_distribution": dict(sorted(dx_counts.items(), key=lambda x: -x[1])),
            "treatment_distribution": dict(sorted(tx_counts.items(), key=lambda x: -x[1])),
            "patients": pts,
        }
    return breakdown


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-AML-MDS-JMML-Predisposition-Atlas",
        "glossary": {
            "GATA2-DeficiencySyndrome": (
                "Spectrum disorder (MonoMAC + Emberger + familial MDS/AML + DCML deficiency) "
                "caused by germline GATA2 LOF. MonoMAC = monocytopenia + MAC infection; "
                "Emberger = lymphedema + MDS + SNHL. Monosomy 7 MDS → AML pathway. HSCT only cure."
            ),
            "DDX41-GermlineAML": (
                "Most common adult AML germline predisposition (~4% of all AML). "
                "Male predominance. Late onset (60-70y). Normal karyotype. "
                "Germline D140Gfs (first hit) + somatic R525H (second hit) = biallelic inactivation. "
                "CRITICAL: siblings cannot donate BM without prior germline testing."
            ),
            "CEBPA-BiallelicAML": (
                "AML with biallelic CEBPA mutations: ELN 2022 favourable risk. "
                "~10% of AML. CR1 >90%. Germline N-terminal frameshift = familial AML; "
                "buccal DNA sequencing distinguishes germline from somatic. "
                "Somatic: HiDAC × 3 (no HSCT); germline: HSCT in CR1."
            ),
            "ETV6-ThrombocytopeniaMALIGNANCY": (
                "Germline ETV6 LOF: mild-moderate thrombocytopenia (often ITP-misdiagnosed) "
                "+ 15-35% lifetime haematological malignancy (ALL, AML, MDS). "
                "ETS-domain dominant-negative mechanism. Dense granule release defect on aggregometry."
            ),
            "MIRAGE-SAMD9": (
                "SAMD9 GOF syndrome: Myelodysplasia + Infections + Restriction (growth) + "
                "Adrenal hypoplasia + Genital (XY-DSD) + Enteropathy. "
                "Monosomy 7 = somatic BM adaptation (escape from GOF antiproliferative); "
                "paradoxically improves cytopenias but creates MDS predisposition."
            ),
            "AtaxiaPancytopenia-SAMD9L": (
                "SAMD9L GOF: cerebellar ataxia (Purkinje cells) + pancytopenia + monosomy 7 BM adaptation. "
                "KEY DISTINCTION: HSCT corrects BM/haematological component but NOT ataxia. "
                "CNS neurons post-mitotic — cannot acquire somatic reversion."
            ),
            "Noonan-JMML-PTPN11": (
                "PTPN11 germline GOF → Noonan syndrome + JMML predisposition (~25% of JMML). "
                "JMML diagnostic: GM-CSF hypersensitivity (spontaneous colonies) + HbF >10%. "
                "Noonan-JMML: spontaneous remission in ~20-30% → watch-and-wait 6-12 months justified."
            ),
            "CBL-JMML-UPD11q": (
                "CBL germline LOF + acquired UPD11q23 (somatic isodisomy) → biallelic CBL null → JMML. "
                "UPD11q23 in JMML clone = near-pathognomonic for germline CBL. "
                "UNIQUE: cutaneous vasculitis seen in CBL carriers (not in PTPN11/KRAS/NRAS/NF1 JMML)."
            ),
            "GermlineTestingAlgorithm-MyeloidMalignancy": (
                "Reflex germline testing in AML/MDS: "
                "Young age + family history → DDX41, CEBPA, GATA2, RUNX1, ETV6; "
                "Biallelic CEBPA in tumour → buccal germline N-terminal frameshift; "
                "Normal karyotype AML + somatic R525H → DDX41 germline D140Gfs; "
                "Monosomy 7 in child with growth failure / multisystem → SAMD9/SAMD9L; "
                "Noonan phenotype + JMML → PTPN11/CBL; "
                "Monocytopenia + NTM → GATA2."
            ),
            "DonorExclusion-GermlineAML": (
                "CRITICAL FOR HSCT: siblings of germline AML patients (DDX41, CEBPA, GATA2, ETV6, RUNX1) "
                "must be germline-tested BEFORE being accepted as donors. "
                "~50% of siblings in AD conditions carry same pathogenic variant → cannot donate."
            ),
            "MonoMAC": (
                "Syndrome: MONOcytopenia + MAC (Mycobacterium avium complex) infections. "
                "Monocyte count near-zero (<0.01×10⁹/L). GATA2 germline LOF causative. "
                "Flow cytometry: absent myeloid DCs + absent pDCs + absent NK + absent B cells (DCML). "
                "ALL four absent together = virtually pathognomonic GATA2 deficiency."
            ),
            "JMML-WHO-Criteria": (
                "Juvenile myelomonocytic leukemia: "
                "Clinical: splenomegaly + monocytosis >1×10⁹/L + age <13 years; "
                "Genetic: RAS pathway mutation (PTPN11/CBL/KRAS/NRAS/NF1 biallelic); "
                "Biological: GM-CSF hypersensitivity (spontaneous myeloid colonies) OR HbF elevated for age; "
                "ALL criteria + genetic confirmation required."
            ),
            "Somatic-Reversion-Monosomy7": (
                "Somatic escape mechanism in SAMD9/SAMD9L GOF: "
                "BM progenitor loses one chr7 (containing GOF allele) → monosomy 7 → "
                "selective proliferative advantage over GOF cells → monosomy 7 clone expands. "
                "PARADOX: cytopenias improve as monosomy 7 expands BUT monosomy 7 itself = MDS risk. "
                "Monitor clone dynamics by FISH/SNP array every 6-12 months."
            ),
            "GM-CSF-Hypersensitivity": (
                "JMML hallmark: myeloid progenitor colony formation at ZERO GM-CSF (spontaneous). "
                "Normal progenitors require GM-CSF for colony growth. "
                "Pathogenesis: RAS-MAPK constitutively active → growth factor independence. "
                "Also seen at supra-physiological GM-CSF concentrations. "
                "WHO diagnostic criterion for JMML. Assay: CFU-GM on methylcellulose without GM-CSF."
            ),
        }
    }
