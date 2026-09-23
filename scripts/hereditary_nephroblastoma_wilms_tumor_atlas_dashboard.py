#!/usr/bin/env python3
"""Hereditary-Nephroblastoma-Wilms-Tumor-Predisposition-Atlas -- Complete 8-Gene Reference
WT1     (Wilms Tumor Suppressor 1; 383aa; 11p13; AD/de-novo LOF;
         WAGR syndrome + aniridia PATHOGNOMONIC;
         Denys-Drash syndrome: DMS PATHOGNOMONIC + 46XY PSD + Wilms 95%;
         Frasier syndrome: FSGS PATHOGNOMONIC + XY gonadal dysgenesis;
         US q3M until age 8; NO ionising screening;
         seed SEED_BASE+0) .
CDKN1C  (Cyclin-Dependent Kinase Inhibitor 1C; 316aa; 11p15.4; Maternal imprinted LOF;
         Beckwith-Wiedemann syndrome IC2;
         macroglossia + omphalocele + hemihypertrophy triad PATHOGNOMONIC;
         Wilms 7-10% BWS; hepatoblastoma 2-3%; neonatal hyperinsulinism;
         US q3M until age 8;
         seed SEED_BASE+1) .
SIX1    (Sine Oculis Homeobox 1; 284aa; 14q23.1; AD GOF Q177R hotspot;
         Wilms tumor 3-4% recurrent hotspot;
         blastemal-predominant favourable histology;
         germline/de-novo Q177R; familial clustering;
         seed SEED_BASE+2) .
SIX2    (Sine Oculis Homeobox 2; 313aa; 2p13.2; AD GOF Q177R hotspot;
         Wilms tumor 1-2% recurrent hotspot;
         same Q177R position as SIX1; overlap phenotype;
         seed SEED_BASE+3) .
DICER1  (DICER1 ribonuclease III; 1922aa; 14q32.13; AD LOF + RNase-IIIb hotspot second hit;
         DICER1 syndrome;
         cystic nephroma PATHOGNOMONIC -- can progress to anaplastic Wilms;
         pleuropulmonary blastoma PATHOGNOMONIC (sibling screen CT chest <8yr);
         AVOID radiation children;
         seed SEED_BASE+4) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         anaplastic Wilms >90% TP53 LOF PATHOGNOMONIC histology;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annual;
         seed SEED_BASE+5) .
WTX     (AMER1 / Wilms Tumor on X; 1135aa; Xq11.1; X-linked LOF;
         somatic 15-20% sporadic Wilms (hemizygous males);
         rare germline LOF; OSCS in females with gain-of-function;
         WNT/beta-catenin pathway negative regulator;
         seed SEED_BASE+6) .
BRCA2   (Breast cancer gene 2; 3418aa; 13q12.3; Biallelic AR LOF FA-D1 / AD LOF HBOC;
         FA complementation group D1: bilateral Wilms PATHOGNOMONIC;
         medulloblastoma + ALL + RMS + bilateral Wilms FA-D1;
         SIBLING DONOR EXCLUSION MANDATORY; HSCT most severe FA;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3262-3269)
"""
import random

SEED_BASE = 3262

ATLAS_GENES = [
    {
        "gene": "WT1",
        "protein": (
            "WT1 -- 11p13 Autosomal-Dominant-de-novo-LOF -- 383aa -- "
            "Wilms-Tumor-Suppressor-1-44kDa-ZnF-Transcription-Factor-"
            "WAGR-DDS-Frasier-Wilms-95pct-DDS-PATHOGNOMONIC-OMIM-194070"
        ),
        "locus": "11p13",
        "protein_size": (
            "383 aa / 44 kDa / 11p13 WT1 encodes Wilms tumour suppressor 1 (zinc finger transcription factor): "
            "STRUCTURE: "
            "  383 aa / 44 kDa; C2H2 zinc finger transcription factor; "
            "  N-terminal repression domain (aa 1-180): proline/glutamine rich; "
            "  Zinc fingers 1-4 (aa 307-399): DNA binding and RNA interaction; "
            "  KTS isoforms: +KTS (Lys-Thr-Ser insertion aa 317) and -KTS arising from alternative splicing of intron 9; "
            "  +KTS / -KTS ratio normally 2:1; critical developmental balance; "
            "  -KTS isoform: transcription factor (binds DNA) -- activates WT1 target genes; "
            "  +KTS isoform: RNA processing / splicing complex; co-localises with snRNPs; "
            "  WT1 activates: WT1-targets including WT1 itself, VIM, VEGF, nephrin; "
            "  WT1 represses: PAX2, EGR1, PDGF-A; role in kidney and gonad development; "
            "WAGR SYNDROME (WT1+PAX6 contiguous deletion): "
            "  OMIM 194072; 11p13 deletion (WT1 + PAX6 co-deletion); "
            "  Aniridia: PAX6 deletion PATHOGNOMONIC (always bilateral when germline); "
            "  Genitourinary anomalies (cryptorchidism, hypospadias in 46XY); "
            "  Intellectual disability (variable); "
            "  Wilms tumour: 45-60% WAGR patients; median age onset 1-2yr; "
            "  Nephroblastomatosis (precursor lesions) in WAGR kidneys; "
            "DENYS-DRASH SYNDROME (DDS): "
            "  OMIM 194080; WT1 missense zinc finger 2/3 (most common R394W, R362W); "
            "  Diffuse mesangial sclerosis (DMS): progressive GN -- ESRD by age 3-5yr PATHOGNOMONIC; "
            "  46XY pseudohermaphroditism (gonadal dysgenesis in 46XY) PATHOGNOMONIC; "
            "  Wilms tumour: ~95% DDS patients -- highest WT1-associated risk; "
            "  Bilateral Wilms in 20%; early onset <2yr typical; "
            "  DMS on biopsy PATHOGNOMONIC -- electron microscopy: mesangial matrix expansion; "
            "FRASIER SYNDROME (FS): "
            "  OMIM 136680; WT1 intron 9 donor splice (IVS9+4C>T, IVS9+5G>A most common); "
            "  +KTS isoform selectively absent -> gonadal dysgenesis cascade; "
            "  Focal segmental glomerulosclerosis (FSGS): progressive PATHOGNOMONIC; "
            "  46XY gonadal dysgenesis (streak gonads, female phenotype) PATHOGNOMONIC; "
            "  Gonadoblastoma in streak gonads 30-40% (WT1 + testicular GCTU predisposition); "
            "  Wilms: 5-10% FS (lower than DDS; later onset 5-10yr); "
            "  ISOLATED WT1 missense: Wilms 10-20% (without full DDS/FS phenotype); "
            "RENAL SURVEILLANCE (WT1): "
            "  Abdominal US every 3 months from diagnosis until age 8yr (high risk genes); "
            "  NO ionising radiation (X-ray/CT) for screening -- MRI if US equivocal; "
            "  DDS/WAGR: bilateral nephroblastomatosis monitoring; nephron-sparing surgery aim; "
            "  Renal: ESRD monitoring (DDS: dialysis 3-5yr; renal transplant ESRD endpoint); "
            "  Gonadoblastoma surveillance: pelvic US + MRI gonadal streak monitoring in FS/DDS 46XY; "
            "TREATMENT (WT1 Wilms): "
            "  SIOP protocol (Europe): neoadjuvant vincristine + actinomycin D x 4wk pre-nephrectomy; "
            "  COG protocol (North America): upfront nephrectomy then adjuvant chemotherapy; "
            "  Stage I/II favourable: VCR + AMD x 18 weeks; "
            "  Stage III/IV: VCR + AMD + doxorubicin x 24 weeks + RT flank/whole-abdomen; "
            "  Anaplastic: VCR + AMD + doxorubicin + carboplatin + etoposide (regimen DD-4A); "
            "  Bilateral (stage V): nephron-sparing bilateral, sequential chemotherapy; "
        ),
        "inheritance": "AD/de-novo LOF (missense or deletion); OMIM 194070; WAGR 11p13 deletion (WT1+PAX6); DDS WT1 missense ZnF 2/3; FS intron 9 splice; ~50% de novo in isolated WT1 mutations; Knudson two-hit for Wilms tumour",
        "cancer_risk": "Wilms tumour: DDS ~95% (highest), WAGR 45-60%, FS 5-10%, isolated WT1 ~20%; gonadoblastoma 30-40% in FS/DDS 46XY streak gonads; nephroblastomatosis (precursor) in WAGR/DDS kidneys",
        "pathognomonic": "Aniridia + Wilms PATHOGNOMONIC WAGR (11p13 deletion); diffuse mesangial sclerosis PATHOGNOMONIC DDS; 46XY gonadal dysgenesis + FSGS PATHOGNOMONIC Frasier; nephroblastomatosis bilateral WAGR/DDS",
        "surveillance_key": "US every 3 months until age 8yr; NO ionising radiation screening; bilateral nephroblastomatosis monitoring DDS/WAGR; DDS ESRD dialysis 3-5yr; gonadoblastoma US/MRI 46XY FS/DDS streak gonads; nephron-sparing surgery bilateral Wilms",
        "key_distinctions": [
            "DDS-DIFFUSE-MESANGIAL-SCLEROSIS-PATHOGNOMONIC",
            "WAGR-ANIRIDIA-PATHOGNOMONIC-11p13-DELETION",
            "FRASIER-FSGS-XY-GONADAL-DYSGENESIS-PATHOGNOMONIC",
            "WILMS-95PCT-DDS-HIGHEST-WT1-RISK",
            "US-Q3M-UNTIL-AGE-8-NO-CT-IONISING",
            "GONADOBLASTOMA-STREAK-GONADS-FS-DDS-XY",
        ],
    },
    {
        "gene": "CDKN1C",
        "protein": (
            "CDKN1C -- 11p15.4 Maternal-imprinted-LOF -- 316aa -- "
            "p57KIP2-36kDa-CKI-CDK2-CDK4-BWS-IC2-"
            "Macroglossia-Omphalocele-Hemihypertrophy-PATHOGNOMONIC-OMIM-130650"
        ),
        "locus": "11p15.4",
        "protein_size": (
            "316 aa / 36kDa / 11p15.4 CDKN1C encodes p57KIP2 (cyclin-dependent kinase inhibitor 1C): "
            "STRUCTURE: "
            "  316 aa / 36kDa; CIP/KIP family CDK inhibitor; "
            "  N-terminal CDK2-binding domain (aa 27-66): blocks CDK2/cyclin E kinase activity; "
            "  CDK4/6 inhibitory domain (aa 131-189); "
            "  PCNA-binding domain (aa 255-289); "
            "  QT/KH domain (aa 237-284): stability, nuclear export; "
            "  Maternally expressed, paternally imprinted (11p15 IC2 region); "
            "  CDKN1C LOF -> CDK2 active -> S-phase entry unchecked -> overgrowth; "
            "BECKWITH-WIEDEMANN SYNDROME (BWS): "
            "  OMIM 130650; 11p15 imprinting disorder; ~1/10,000-13,000 births; "
            "  BWS-IC2 (CDKN1C LOF + H19-IGF2 imprinting centre 2 region): ~40-50% BWS-CDKN1C; "
            "  BWS triad PATHOGNOMONIC: macroglossia (large tongue) + exomphalos/omphalocele + hemihypertrophy; "
            "  Neonatal hyperinsulinism (Nesidioblastosis): hypoglycaemia neonatal emergency; "
            "  Ear pits + ear creases: diagnostic feature (not pathognomonic alone); "
            "  Vascular malformations (facial naevus flammeus) common; "
            "  Overgrowth: large birth weight + length (macrosomia); "
            "  Cardiac defects: 10% BWS (structural); "
            "WILMS TUMOUR (BWS): "
            "  Wilms tumour 7-10% BWS overall (IC1 methylation subtype highest ~28%); "
            "  CDKN1C LOF subtype: Wilms ~3-5% (lower vs IC1 subtype); "
            "  Risk highest first 8yr; screening until 8yr mandatory; "
            "  Hepatoblastoma: 2-3% BWS (especially IC1/paternal UPD); "
            "  Other tumours: adrenal cortical carcinoma, rhabdomyosarcoma (rare); "
            "IMPRINTING MECHANISM: "
            "  Maternal LOF (inherited germline or de novo maternal): classic BWS; "
            "  Paternal UPD11p15 (uniparental disomy): ~20% BWS -- CDKN1C lost + IGF2 doubled; "
            "  IC1 hypermethylation (H19-IGF2 locus): highest tumour risk (distinct mechanism); "
            "  IC2 LOM (loss of methylation): most common BWS -- lower tumour risk; "
            "  Mosaic: milder phenotype; "
            "IMAAGE SYNDROME: "
            "  OMIM 614732; CDKN1C gain-of-function (AR) -- IMAGe = IUGR + metaphyseal dysplasia + "
            "  Adrenal hypoplasia + genital anomalies + elevated ACTH; distinct from BWS; "
            "NEONATAL HYPERINSULINISM MANAGEMENT: "
            "  Glucose infusion rate >10-15 mg/kg/min required; "
            "  Diazoxide first-line: KATP-channel blocker -- 50-70% respond; "
            "  Octreotide: somatostatin analogue second-line; "
            "  Pancreatectomy: focal vs diffuse on 18F-DOPA PET (mandatory distinction); "
            "  18F-DOPA PET: focal disease = curative 95% partial pancreatectomy; diffuse = near-total; "
            "SURVEILLANCE (BWS): "
            "  Abdominal US every 3 months until age 8yr; "
            "  Alpha-fetoprotein (AFP) q3M until age 4yr (hepatoblastoma screening); "
            "  Neurodevelopment monitoring (macroglossia -> speech therapy); "
        ),
        "inheritance": "Maternal LOF (imprinted; maternally expressed gene CDKN1C); inherited or de novo maternal; paternal UPD11p15 (~20% BWS); IC2 LOM most common BWS molecular subtype; IMAAGE syndrome AR GOF (rare; distinct clinical syndrome)",
        "cancer_risk": "Wilms tumour 7-10% BWS overall; hepatoblastoma 2-3% BWS (especially IC1/patUPD); adrenal cortical carcinoma rare; overall embryonal tumour risk ~10%; highest 8yr window",
        "pathognomonic": "Macroglossia + omphalocele + hemihypertrophy triad PATHOGNOMONIC BWS; neonatal hyperinsulinism + macroglossia PATHOGNOMONIC; 18F-DOPA PET focal vs diffuse hyperinsulinism PATHOGNOMONIC distinction; ear pits + creases in context",
        "surveillance_key": "US every 3 months until age 8yr; AFP q3M until age 4yr; 18F-DOPA PET for neonatal hyperinsulinism (focal vs diffuse mandatory); diazoxide first-line neonatal HI; partial pancreatectomy focal PATHOGNOMONIC curative",
        "key_distinctions": [
            "BWS-TRIAD-MACROGLOSSIA-OMPHALOCELE-HEMIHYPERTROPHY-PATHOGNOMONIC",
            "NEONATAL-HYPERINSULINISM-DIAZOXIDE-FIRST-LINE",
            "18F-DOPA-PET-FOCAL-VS-DIFFUSE-MANDATORY",
            "IC1-HIGHEST-WILMS-28PCT-IC2-LOM-LOWEST",
            "US-Q3M-UNTIL-8YR-AFP-Q3M-UNTIL-4YR",
            "IMAAGE-CDKN1C-GOF-AR-DISTINCT-FROM-BWS",
        ],
    },
    {
        "gene": "SIX1",
        "protein": (
            "SIX1 -- 14q23.1 Autosomal-Dominant-GOF-Q177R -- 284aa -- "
            "Sine-Oculis-Homeobox-1-32kDa-HD-SD-Transcription-Factor-"
            "Q177R-Wilms-3-4pct-Blastemal-Favourable-Hotspot-OMIM-601205"
        ),
        "locus": "14q23.1",
        "protein_size": (
            "284 aa / 32kDa / 14q23.1 SIX1 encodes Sine Oculis Homeobox 1 (SIX family transcription factor): "
            "STRUCTURE: "
            "  284 aa / 32kDa; SIX domain (SD, aa 76-142) + homeodomain (HD, aa 141-204); "
            "  SD: protein-protein interaction, binds co-activator EYA1/EYA2; "
            "  HD: DNA binding (consensus ATAATCA); "
            "  SIX1 activates: target genes in kidney development (WT1, PAX2, GDNF); "
            "  SIX1 maintains nephron progenitor pool (cap mesenchyme); "
            "  SIX1 Q177R (Gln177Arg): recurrent hotspot; located in homeodomain; "
            "  Q177R changes DNA binding specificity: neomorphic GOF transcription; "
            "  Q177R found in ~3-4% of Wilms tumours (somatic or germline de novo); "
            "Q177R HOTSPOT (SIX1 WILMS): "
            "  Recurrent Q177R mutation in SIX1: blastemal-predominant, epithelial-predominant histology; "
            "  Favourable histology (low risk relapse in appropriately treated); "
            "  SIX1 Q177R germline: rare, de novo; familial Wilms clustering reported; "
            "  Q177R disrupts SD/HD interaction: altered transcriptional target specificity; "
            "  RNA helicase DDX3X co-mutated in SIX1/SIX2 Q177R Wilms; "
            "  SIX1 Q177R Wilms: DROSHA/DGCR8 mutations mutually exclusive (different Wilms pathway); "
            "  Syndromic features: branchiootorenal spectrum (BOR) if other SIX1 variants (distinct); "
            "BRANCHIOOTORENAL SYNDROME (BOR1 -- distinct SIX1 variant): "
            "  OMIM 113650; SIX1 missense/truncation (non-Q177R): BOR spectrum; "
            "  Branchial cleft cysts + preauricular pits PATHOGNOMONIC + hearing loss + renal anomalies; "
            "  NOT Q177R -- Q177R = Wilms predisposition not BOR; "
            "  BOR1 renal: dysplasia, hypoplasia, duplicated collecting system; "
            "CLINICAL MANAGEMENT (SIX1 Q177R Wilms): "
            "  No established germline surveillance protocol (rare, evolving); "
            "  Emerging: if germline Q177R confirmed, consider Q3M US until age 8yr; "
            "  First-degree relatives: testing recommended (familial clustering cases); "
            "  Treatment: standard Wilms protocol (SIOP/COG per stage/histology); "
            "  SIX1 Q177R blastemal Wilms: generally good prognosis if stage I-II; "
        ),
        "inheritance": "AD GOF Q177R hotspot; often de novo germline or somatic; familial clustering documented; recurrent mutation in exon 2; BOR1 (different SIX1 non-Q177R variants): distinct clinical syndrome with branchial/otic/renal features",
        "cancer_risk": "Wilms tumour: ~3-4% recurrent Q177R hotspot contribution; blastemal-predominant; favourable histology; bilateral Wilms rare but reported; no clear secondary cancer risk established",
        "pathognomonic": "Q177R hotspot in SIX1 within Wilms blastemal pattern; BOR1 syndrome: branchial cleft cysts + preauricular pits PATHOGNOMONIC (different SIX1 variants); SIX1 Q177R + DDX3X co-mutation Wilms cluster",
        "surveillance_key": "Emerging: US q3M until age 8yr for germline Q177R; first-degree testing; standard Wilms treatment protocol; favourable histology blastemal Wilms generally good prognosis; BOR1 non-Q177R hearing assessment mandatory",
        "key_distinctions": [
            "Q177R-HOTSPOT-RECURRENT-WILMS-3-4PCT",
            "BLASTEMAL-PREDOMINANT-FAVOURABLE-HISTOLOGY",
            "GERMLINE-Q177R-DE-NOVO-FAMILIAL-CLUSTERING",
            "SIX1-BOR1-SEPARATE-SYNDROME-NON-Q177R",
            "DDX3X-CO-MUTATION-SIX1-Q177R-WILMS",
            "SIX1-Q177R-DROSHA-MUTUALLY-EXCLUSIVE-PATHWAY",
        ],
    },
    {
        "gene": "SIX2",
        "protein": (
            "SIX2 -- 2p13.2 Autosomal-Dominant-GOF-Q177R -- 313aa -- "
            "Sine-Oculis-Homeobox-2-36kDa-HD-SD-Nephron-Progenitor-"
            "Q177R-Wilms-1-2pct-Hotspot-SIX1-Overlap-OMIM-604994"
        ),
        "locus": "2p13.2",
        "protein_size": (
            "313 aa / 36kDa / 2p13.2 SIX2 encodes Sine Oculis Homeobox 2 (SIX family transcription factor): "
            "STRUCTURE: "
            "  313 aa / 36kDa; SIX domain (SD) + homeodomain (HD); "
            "  SIX2 critical for nephron progenitor self-renewal (cap mesenchyme maintenance); "
            "  SIX2 maintains UB-responsive progenitors: WT1 and CITED1 co-expressed; "
            "  SIX2 Q177R: same hotspot codon as SIX1 Q177R (Gln177Arg in HD); "
            "  SIX2 Q177R recurrent in Wilms tumours (~1-2%); "
            "  SIX2 activates: Wnt4, WT1, target genes in progenitor self-renewal; "
            "  SIX2 represses: MET, differentiation markers; "
            "  SIX2 Q177R changes DNA binding: neomorphic transcriptional activation; "
            "SIX2 Q177R WILMS: "
            "  Recurrent Q177R in SIX2: ~1-2% Wilms tumours; "
            "  Phenotypically similar to SIX1 Q177R: blastemal-type, favourable histology; "
            "  SIX1 and SIX2 Q177R co-mutated cases reported (compound effect); "
            "  SIX2 Q177R germline: rare de novo; no established hereditary syndrome yet; "
            "  SIX2 LOF variants: tubular dysgenesis, nephrotic syndrome (distinct; non-cancer); "
            "CLINICAL SIGNIFICANCE: "
            "  SIX2 Q177R should prompt germline consideration in bilateral/young Wilms; "
            "  SIX2 germline Q177R: similar surveillance as SIX1 Q177R (emerging guidance); "
            "  SIX2 non-Q177R germline: tubular dysgenesis spectrum (distinct phenotype); "
            "  Combined SIX1+SIX2 panel recommended for Wilms predisposition testing; "
        ),
        "inheritance": "AD GOF Q177R hotspot (same codon as SIX1); often somatic; germline reported in bilateral/young onset Wilms; SIX2 LOF different syndrome (tubular dysgenesis, not cancer-prone)",
        "cancer_risk": "Wilms tumour ~1-2%; blastemal-type favourable histology similar to SIX1; bilateral Wilms reported with germline; phenotypic overlap with SIX1 Q177R",
        "pathognomonic": "Q177R SIX2 hotspot in Wilms blastemal type; SIX2+SIX1 Q177R co-mutation reported; SIX2 germline + family history bilateral Wilms pathognomonic signal for genetic testing",
        "surveillance_key": "Emerging: US q3M until 8yr if germline Q177R confirmed; test first-degree relatives; standard SIOP/COG treatment protocol; SIX1+SIX2 panel testing simultaneously for Wilms predisposition",
        "key_distinctions": [
            "SIX2-Q177R-SAME-HOTSPOT-SIX1-1-2PCT-WILMS",
            "BLASTEMAL-TYPE-FAVOURABLE-HISTOLOGY-SIX2",
            "SIX2-LOF-TUBULAR-DYSGENESIS-NOT-CANCER",
            "SIX1-SIX2-PANEL-SIMULTANEOUSLY-WILMS",
            "Q177R-GERMLINE-BILATERAL-YOUNG-WILMS-SIGNAL",
            "SIX2-SIX1-CO-MUTATION-COMPOUND-EFFECT",
        ],
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF-RNaseIIIb-hotspot -- 1922aa -- "
            "DICER1-RNaseIII-219kDa-miRNA-Biogenesis-"
            "Cystic-Nephroma-PATHOGNOMONIC-PPB-PATHOGNOMONIC-Cervical-ERMS-PATHOGNOMONIC-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 219kDa / 14q32.13 DICER1 encodes DICER1 (dsRNA-specific endoribonuclease type III): "
            "STRUCTURE: "
            "  1922 aa / 219kDa; large multi-domain endonuclease; "
            "  DEAD helicase/ATPase domain (aa 70-440): ATP-dependent dsRNA unwinding; "
            "  Platform domain (aa 441-820): substrate tethering; "
            "  PAZ domain (aa 852-952): recognises 3' 2-nt overhang of pre-miRNA; "
            "  Connector helix (aa 953-1005): bridges PAZ to RNase IIIs; "
            "  RNase IIIa domain (aa 1006-1158): cleaves miRNA guide strand; "
            "  RNase IIIb domain (aa 1158-1345): cleaves miRNA passenger strand; "
            "  RNase IIIb hotspot residues (E1705/D1709/E1813): clustered metal-binding; "
            "  dsRBD (aa 1810-1922): C-terminal double-stranded RNA binding domain; "
            "  DICER1 LOF -> pre-miRNA not processed -> global miRNA downregulation; "
            "  RNase IIIb hotspot somatic second hit: type-specific defect (5p-arm miRNA loss); "
            "DICER1 SYNDROME (OMIM 601200): "
            "  AD LOF + somatic RNase IIIb hotspot second hit; ~50% de novo; "
            "  Pleuropulmonary blastoma type I/II/III PATHOGNOMONIC (hallmark tumour); "
            "  Cervical embryonal RMS (ERMS) PATHOGNOMONIC; "
            "  Sertoli-Leydig cell tumour (SLCT) ovary PATHOGNOMONIC (adolescent); "
            "  Multi-nodular goitre (MNG) ~75% penetrance; "
            "  Nasal chondromesenchymal hamartoma PATHOGNOMONIC; "
            "CYSTIC NEPHROMA (DICER1): "
            "  Cystic nephroma PATHOGNOMONIC DICER1 syndrome; "
            "  Renal cystic tumour: septate fluid-filled; benign but DICER1-associated; "
            "  Cystic nephroma can progress to anaplastic Wilms tumour (Wilms blastema within wall); "
            "  DICER1 Wilms ~1-2%: often anaplastic or mixed histology; more aggressive; "
            "  Cystic partially differentiated nephroblastoma: DICER1 overlap tumour; "
            "PPB (PLEUROPULMONARY BLASTOMA): "
            "  Type I (purely cystic): <2yr, lowest malignant potential, US/CT surveillance lungs; "
            "  Type II (mixed cystic+solid): 2-5yr, intermediate; "
            "  Type III (purely solid): >2yr, highest malignancy; chemotherapy required; "
            "  PPB PATHOGNOMONIC for DICER1 -- ALL siblings <8yr: chest CT mandatory; "
            "AVOID RADIATION CHILDREN (DICER1): "
            "  Developing organs in DICER1 children: radiation sensitivity increased; "
            "  RT contraindicated in young DICER1 children except life-saving scenario; "
            "  Fertility-sparing surgery: cervical ERMS -- try conservative if feasible adolescent; "
        ),
        "inheritance": "AD LOF germline (first hit); RNase IIIb hotspot somatic (second hit) in tumour; ~50% de novo; variable expressivity; full penetrance for surveillance-detectable lesions over lifetime",
        "cancer_risk": "PPB type I/II/III PATHOGNOMONIC (highest risk organ); cervical ERMS PATHOGNOMONIC; cystic nephroma PATHOGNOMONIC (Wilms risk ~1-2% from nephroma progression); SLCT ovary 10-15% females; nasal chondromesenchymal hamartoma PATHOGNOMONIC",
        "pathognomonic": "Cystic nephroma PATHOGNOMONIC DICER1; PPB type I (cystic lung <2yr) PATHOGNOMONIC; cervical ERMS PATHOGNOMONIC; nasal chondromesenchymal hamartoma PATHOGNOMONIC; RNase IIIb somatic hotspot in tumour PATHOGNOMONIC",
        "surveillance_key": "ALL siblings <8yr: chest CT (PPB risk); CT chest baseline + q2yr; US kidneys q3M until 8yr; pelvic US adolescent females (SLCT/cervical); MNG annual thyroid US; AVOID radiation children; cystic nephroma Wilms progression monitoring",
        "key_distinctions": [
            "CYSTIC-NEPHROMA-PATHOGNOMONIC-DICER1",
            "PPB-TYPE-I-PATHOGNOMONIC-SIBLINGS-CT-CHEST-MANDATORY",
            "CERVICAL-ERMS-PATHOGNOMONIC-ADOLESCENT",
            "RNASE-IIIB-HOTSPOT-SOMATIC-SECOND-HIT-PATHOGNOMONIC",
            "AVOID-RADIATION-DICER1-CHILDREN",
            "CYSTIC-NEPHROMA-WILMS-PROGRESSION-MONITOR",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-Anaplastic-Wilms-GT90pct-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43kDa / 17p13.1 TP53 encodes p53 (tumour protein p53; master transcriptional regulator): "
            "STRUCTURE: "
            "  393 aa / 43kDa; transcription factor with tetramerisation; "
            "  N-terminal transactivation domain 1+2 (aa 1-67): MDM2-binding; "
            "  DNA-binding domain (aa 102-292): 80% of cancer hotspot mutations; "
            "  Tetramerisation domain (aa 323-356): homo-tetramer required for function; "
            "  C-terminal regulatory domain (aa 357-393): acetylation, ubiquitination; "
            "  TP53 activates: CDKN1A (p21), MDM2, PUMA, NOXA, BAX (apoptosis/arrest); "
            "ANAPLASTIC WILMS TUMOUR (TP53): "
            "  Anaplastic histology Wilms: TP53 LOF somatic/germline >90% cases PATHOGNOMONIC; "
            "  Diffuse anaplasia Wilms: worst prognosis (5-yr OS 40-60% stage IV diffuse anaplasia); "
            "  Focal anaplasia: intermediate prognosis; "
            "  TP53 germline in diffuse anaplasia Wilms: evaluate for LFS (Chompret criteria); "
            "  Anaplastic Wilms + other LFS tumours in family: GERMLINE TP53 mandatory testing; "
            "  Chemotherapy resistance: anaplastic cells are p53-deficient -> limited apoptosis induction; "
            "LI-FRAUMENI SYNDROME (LFS) -- WILMS CONTEXT: "
            "  OMIM 151623; AD LOF; 1/3,000-1/5,000 births; ~20% de novo; "
            "  Wilms tumour: LFS tumour spectrum (<15yr); less frequent vs sarcoma/brain/breast; "
            "  TP53 R337H founder: southern Brazil 1/300 carrier; ACT paediatric elevated; "
            "  Bilateral Wilms in LFS: reported; requires nephron-sparing approach; "
            "RADIATION ABSOLUTE CONTRAINDICATION (LFS): "
            "  AVOID RADIATION ABSOLUTELY in LFS; "
            "  Wilms RT (flank/whole abdomen) absolutely contraindicated TP53 germline; "
            "  Chemotherapy-alone approach for LFS Wilms if feasible; "
            "  WBMRI Toronto Protocol: full-body MRI annually NOT CT/PET; "
            "TREATMENT (TP53 anaplastic Wilms): "
            "  Diffuse anaplasia stage II-IV: UH-1 regimen (VCR+AMD+doxorubicin+carboplatin+cyclophosphamide); "
            "  Stage IV diffuse anaplasia: highest intensity regimen + if no germline TP53: RT whole lung; "
            "  IF germline TP53: omit RT, maximise surgery (resect metastases), intensify chemo; "
        ),
        "inheritance": "AD LOF; OMIM 151623; 1/3,000-1/5,000; ~20% de novo; LFS classic + Chompret 2015 criteria; anaplastic Wilms: somatic TP53 >90% but test germline when LFS criteria met",
        "cancer_risk": "Anaplastic Wilms >90% TP53 LOF (somatic or germline; test germline in diffuse anaplasia); LFS: sarcoma 30-50% DOMINANT, premenopausal breast 30%, brain 15%, adrenocortical carcinoma (R337H), colorectal 3-5%",
        "pathognomonic": "Anaplastic Wilms histology PATHOGNOMONIC for TP53 LOF (>90%); diffuse anaplasia + LFS family history: GERMLINE TP53 mandatory; multiple LFS-spectrum primaries including Wilms PATHOGNOMONIC LFS",
        "surveillance_key": "AVOID RADIATION ABSOLUTELY in LFS; WBMRI Toronto annual NOT CT/PET; omit RT from Wilms protocol if TP53 germline; anaplastic Wilms always test germline TP53; UH-1 regimen diffuse anaplasia; annual breast MRI from 20yr LFS",
        "key_distinctions": [
            "ANAPLASTIC-WILMS-GT90PCT-TP53-LOF-PATHOGNOMONIC",
            "AVOID-RADIATION-ABSOLUTELY-LFS-WILMS-OMIT-RT",
            "WBMRI-TORONTO-ANNUAL-NOT-CT-LFS",
            "DIFFUSE-ANAPLASIA-GERMLINE-TP53-MANDATORY",
            "R337H-BRAZILIAN-FOUNDER-1IN300-ACT-ELEVATED",
            "UH-1-REGIMEN-DIFFUSE-ANAPLASIA-WILMS",
        ],
    },
    {
        "gene": "WTX",
        "protein": (
            "WTX/AMER1 -- Xq11.1 X-linked-LOF -- 1135aa -- "
            "AMER1-131kDa-APC-Membrane-Recruiter-WNT-Pathway-"
            "Somatic-15-20pct-Wilms-OSCS-Females-Germline-OMIM-300732"
        ),
        "locus": "Xq11.1",
        "protein_size": (
            "1135 aa / 131kDa / Xq11.1 WTX/AMER1 encodes AMER1 (APC membrane recruitment protein 1): "
            "STRUCTURE: "
            "  1135 aa / 131kDa; X-linked; males hemizygous; females heterozygous; "
            "  C-terminal APC-interaction domain (aa 800-1135): recruits APC to plasma membrane; "
            "  AMER1 stabilises APC at membrane: promotes beta-catenin degradation; "
            "  AMER1 LOF -> reduced membrane APC -> beta-catenin nuclear accumulation -> WNT activation; "
            "  X-linked: males require single hit (hemizygous); females need biallelic LOF; "
            "WTX SOMATIC IN SPORADIC WILMS: "
            "  WTX/AMER1 somatic LOF: 15-20% of sporadic Wilms tumours (males hemizygous); "
            "  Mechanism: deletion or mutation of Xq11.1 locus in Wilms; "
            "  WTX-mutant Wilms: often favourable histology, lower stage; "
            "  WTX pathway: independent of WT1/CTNNB1 pathway (complementary WNT activation); "
            "  WT1 + CTNNB1 co-mutated in some Wilms: WTX distinct molecular subtype; "
            "GERMLINE WTX/AMER1 (RARE): "
            "  Germline WTX LOF: rare reported in familial/bilateral Wilms; "
            "  Males hemizygous LOF: Wilms risk; "
            "  Females with heterozygous WTX: usually carriers; biallelic loss needed in tumour; "
            "OSTEOPATHIA STRIATA WITH CRANIAL SCLEROSIS (OSCS): "
            "  OMIM 300301; WTX/AMER1 gain-of-function (GOF) in females; "
            "  Females (heterozygous GOF): osteopathia striata (longitudinal bone sclerosis) PATHOGNOMONIC; "
            "    cranial sclerosis, macrocephaly, cleft palate, learning difficulties; "
            "  Males (hemizygous LOF of different allele): lethal in utero (hyperactivated WNT in embryo); "
            "  OSCS is distinct: GOF females survive, LOF males lethal -- paradoxical allele biology; "
            "  OSCS NOT a cancer predisposition syndrome; "
            "CLINICAL MANAGEMENT (WTX germline Wilms): "
            "  Suspected germline WTX/AMER1: US q3M until 8yr; "
            "  Bilateral Wilms in male + WTX LOF: test germline Xq11.1 deletion; "
            "  Standard Wilms treatment per stage/histology; "
        ),
        "inheritance": "X-linked LOF; males hemizygous (single hit for somatic WTX Wilms); females heterozygous (2 hits for tumour); OSCS: X-linked GOF (distinct syndrome -- not cancer predisposition); rare germline LOF in familial Wilms",
        "cancer_risk": "Wilms tumour: somatic WTX LOF 15-20% sporadic (not germline predisposition in most); germline WTX LOF rare in familial/bilateral Wilms; OSCS (GOF females) not cancer predisposition; favourable histology WTX-somatic Wilms",
        "pathognomonic": "Osteopathia striata (longitudinal striation metaphyses) PATHOGNOMONIC OSCS females; WTX GOF allele OSCS vs WTX LOF allele Wilms -- allele-specific distinction; X-linked hemizygous males single-hit somatic Wilms",
        "surveillance_key": "US q3M until 8yr if germline WTX LOF confirmed; WTX somatic in tumour: standard treatment protocol; OSCS (GOF): orthopaedic + craniofacial monitoring (distinct from cancer surveillance); males hemizygous LOF higher penetrance Wilms if germline",
        "key_distinctions": [
            "WTX-SOMATIC-15-20PCT-SPORADIC-WILMS-HEMIZYGOUS",
            "AMER1-APC-MEMBRANE-RECRUITER-WNT-PATHWAY",
            "OSCS-GOF-FEMALES-OSTEOPATHIA-STRIATA-PATHOGNOMONIC",
            "WTX-LOF-MALES-HEMIZYGOUS-WILMS",
            "WTX-GERMLINE-RARE-FAMILIAL-BILATERAL-WILMS",
            "WTX-PATHWAY-INDEPENDENT-WT1-CTNNB1-PATHWAY",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Biallelic-AR-LOF-FA-D1 / AD-LOF-HBOC -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-FAD1-Bilateral-Wilms-PATHOGNOMONIC-"
            "Sibling-Donor-Exclusion-MANDATORY-HSCT-Most-Severe-FA-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384kDa / 13q12.3 BRCA2 encodes BRCA2 (breast cancer susceptibility gene 2; HR repair scaffold): "
            "STRUCTURE: "
            "  3418 aa / 384kDa; large nuclear scaffold protein; "
            "  PALB2-binding domain (aa 10-40): nuclear localisation of BRCA2; "
            "  BRC repeats x8 (aa 1002-2085): RAD51 binding, displaces RPA on ssDNA; "
            "  DNA binding domain (DBD, aa 2402-3190): ssDNA and dsDNA binding; "
            "  BRCA2 loads RAD51 onto RPA-coated ssDNA at DSB -> homologous recombination; "
            "  BRCA2 also essential in Fanconi anemia pathway (FANCD1 complementation group); "
            "  Biallelic BRCA2 LOF: Fanconi anemia complementation group D1 (FA-D1); "
            "FANCONI ANEMIA COMPLEMENTATION GROUP D1 (FA-D1): "
            "  OMIM 605724; biallelic BRCA2 (compound heterozygous or homozygous); "
            "  Most severe Fanconi anemia subtype: median age cancer onset 2-5yr; "
            "  Bilateral Wilms tumour: FA-D1 PATHOGNOMONIC (50-60% risk bilateral); "
            "  Medulloblastoma (SHH subtype): 10-15% FA-D1 PATHOGNOMONIC; "
            "  Acute lymphoblastic leukaemia (ALL): 20-30% FA-D1; "
            "  Embryonal RMS: 5-10% FA-D1; "
            "  Classical FA: bone marrow failure (BMF) + VACTERL association + café-au-lait; "
            "  VACTERL: Vertebral, Anal, Cardiac, TEF, Oesophageal, Renal, Limb anomalies; "
            "  Chromosomal fragility: DEB/MMC test PATHOGNOMONIC (diagnostic for FA); "
            "  FA-D1 survival without HSCT: very poor (median survival ~5yr); "
            "SIBLING DONOR EXCLUSION (FA-D1 BILATERAL WILMS): "
            "  SIBLING DONOR EXCLUSION MANDATORY before HSCT for Wilms: "
            "    sibling carriers (25% risk each) cannot be HSCT donors; "
            "    ALL siblings must be tested before donor evaluation; "
            "    unaffected sibling FA-D1 = 25% risk (both parents heterozygous BRCA2); "
            "  DEB/MMC chromosomal fragility test: ALL potential donors mandatory pre-HSCT; "
            "BILATERAL WILMS FA-D1 MANAGEMENT: "
            "  Bilateral Wilms + FA-D1: nephron-sparing bilateral nephrectomy essential; "
            "  Chemotherapy: modified FA protocol (reduced alkylating agents -- BMF risk); "
            "  Avoid alkylating agents (cyclophosphamide, ifosfamide): worsen BMF in FA; "
            "  Carboplatin preferred over cisplatin for less marrow toxicity; "
            "  HSCT: only curative for BMF component (not for solid tumours); "
            "  Androgens: temporary BMF bridge prior to HSCT; "
            "MONOALLELIC BRCA2 (HBOC -- HEREDITARY BREAST AND OVARIAN CANCER): "
            "  Monoallelic AD LOF: HBOC; breast 47-69%, ovarian 11-17% lifetime; "
            "  Pancreatic 3-5x; prostate 6-9x; "
            "  Wilms: rare case reports monoallelic BRCA2 -- not established elevated risk; "
            "  PBSO by 40-45yr female BRCA2 (ovarian cancer prevention); "
        ),
        "inheritance": "Biallelic AR LOF (FA-D1): compound heterozygous BRCA2 -- both parents obligate heterozygous; Monoallelic AD LOF (HBOC): 50% transmission; FA-D1 prior probability: parents usually HBOC carriers (elevated cancer risk)",
        "cancer_risk": "FA-D1 biallelic: bilateral Wilms 50-60% PATHOGNOMONIC; medulloblastoma 10-15% PATHOGNOMONIC; ALL 20-30%; embryonal RMS 5-10%; BMF near-universal; Monoallelic HBOC: breast 47-69%, ovarian 11-17%",
        "pathognomonic": "Bilateral Wilms + DEB/MMC chromosomal fragility PATHOGNOMONIC FA-D1; DEB/MMC test PATHOGNOMONIC Fanconi anemia diagnosis; VACTERL + BMF + bilateral Wilms PATHOGNOMONIC FA-D1; sibling donor exclusion MANDATORY",
        "surveillance_key": "DEB/MMC test ALL potential donors MANDATORY; sibling donor exclusion before HSCT; modified FA chemo protocol (avoid alkylating agents in FA-D1); bilateral nephron-sparing surgery; androgens bridge BMF to HSCT; monoallelic BRCA2 PBSO by 40-45yr",
        "key_distinctions": [
            "FA-D1-BILATERAL-WILMS-50-60PCT-PATHOGNOMONIC",
            "SIBLING-DONOR-EXCLUSION-MANDATORY-HSCT",
            "DEB-MMC-CHROMOSOMAL-FRAGILITY-PATHOGNOMONIC-FA-DIAGNOSIS",
            "AVOID-ALKYLATING-AGENTS-FA-D1-BMF-WORSEN",
            "VACTERL-BMF-BILATERAL-WILMS-FA-D1-TRIAD",
            "BRCA2-MONOALLELIC-HBOC-PARENTS-OBLIGATE-CARRIERS",
        ],
    },
]

# Per-gene tumour types (key Wilms + associated tumours)
TUMOR_TYPES = {
    "WT1":    ["Wilms tumour (nephroblastoma)", "Gonadoblastoma (FS/DDS 46XY)", "Nephroblastomatosis", "Mesothelioma (rare WT1 somatic)"],
    "CDKN1C": ["Wilms tumour (7-10% BWS)", "Hepatoblastoma (2-3% BWS)", "Adrenal cortical carcinoma", "Rhabdomyosarcoma (rare)"],
    "SIX1":   ["Wilms tumour (3-4% Q177R)", "Blastemal-predominant Wilms", "Bilateral Wilms (rare germline)", "BOR1 renal anomalies (non-Q177R)"],
    "SIX2":   ["Wilms tumour (1-2% Q177R)", "Blastemal-type Wilms", "Bilateral Wilms (rare germline)", "Tubular dysgenesis (non-Q177R)"],
    "DICER1": ["Cystic nephroma PATHOGNOMONIC", "Wilms (anaplastic ~1-2%)", "PPB type I/II/III PATHOGNOMONIC", "Cervical ERMS PATHOGNOMONIC", "SLCT ovary", "MNG"],
    "TP53":   ["Anaplastic Wilms (diffuse/focal)", "Sarcoma (LFS 30-50%)", "Brain tumour (LFS 15%)", "Premenopausal breast (30%)", "Adrenocortical carcinoma"],
    "WTX":    ["Wilms tumour (somatic 15-20%)", "Bilateral Wilms (rare germline LOF)", "OSCS (GOF allele -- not cancer)", "Favourable histology Wilms"],
    "BRCA2":  ["Bilateral Wilms (FA-D1 PATHOGNOMONIC)", "Medulloblastoma (FA-D1 PATHOGNOMONIC)", "ALL (FA-D1 20-30%)", "Embryonal RMS (FA-D1)", "Breast (HBOC monoallelic)", "Ovarian (HBOC)"],
}

# Pathogenic variant examples
PATHOGENIC_VARIANTS = {
    "WT1":    ["R394W (DDS ZnF3 missense most common)", "R362W (DDS ZnF2 missense)", "IVS9+4C>T (Frasier +KTS splice)", "IVS9+5G>A (Frasier splice)", "11p13 deletion (WAGR+PAX6)", "R366H (DDS)", "Truncation/frameshift LOF"],
    "CDKN1C": ["Maternal missense (BWS-IC2)", "Maternal frameshift LOF", "IC2 LOM (methylation -- not sequencing variant)", "Paternal UPD11p15 (not variant -- epigenetic)", "GOF (IMAAGE: AR p.Arg279Pro)", "Promoter variants maternal"],
    "SIX1":   ["Q177R (Gln177Arg -- canonical hotspot)", "Other HD missense (BOR1 phenotype -- distinct)", "Germline Q177R de novo", "Somatic Q177R"],
    "SIX2":   ["Q177R (same hotspot as SIX1)", "Other HD missense (tubular dysgenesis -- distinct)", "Germline Q177R de novo", "Somatic Q177R"],
    "DICER1": ["E1705K (RNase IIIb hotspot somatic)", "D1709N (RNase IIIb hotspot somatic)", "E1813K (RNase IIIb hotspot somatic)", "Germline LOF: frameshift/truncation (first hit)", "Large deletions 14q32.13", "Splice variants"],
    "TP53":   ["R175H (structural GOF/dominant-negative)", "R248W (DNA contact GOF)", "R273H (DNA contact GOF)", "Intron/splice LOF (LFS)", "R337H (Brazil founder AD LOF)", "Large deletions LOF", "Frameshift/truncation LOF"],
    "WTX":    ["Xq11.1 deletion (hemizygous males)", "Frameshift LOF (hemizygous males)", "Large Xq11.1 deletion", "GOF missense (OSCS in females -- distinct)", "Exonic point mutations"],
    "BRCA2":  ["IVS7+2T>G (FA-D1 common)", "p.Trp31Ter (FA-D1)", "p.Thr3033Ile (BRCA2 pathogenic)", "Large exon deletions (HBOC monoallelic)", "6174delT (Ashkenazi Jewish founder)", "Compound heterozygous (FA-D1)"],
}

# Treatment protocols
TREATMENT_PROTOCOLS = {
    "WT1":    ["SIOP: neoadjuvant VCR+AMD x4wk pre-nephrectomy", "COG: upfront nephrectomy then adjuvant", "Stage I/II: VCR+AMD x18wk", "Stage III/IV: VCR+AMD+doxorubicin x24wk ± RT", "Bilateral (stage V): nephron-sparing sequential chemo", "DDS ESRD: renal transplant"],
    "CDKN1C": ["Wilms: standard SIOP/COG per stage", "Hepatoblastoma: cisplatin-based PLADO regimen", "Neonatal HI: glucose infusion + diazoxide + octreotide", "Focal HI: partial pancreatectomy (curative)", "Diffuse HI: near-total pancreatectomy"],
    "SIX1":   ["Standard SIOP/COG Wilms protocol per stage/histology", "SIX1 blastemal: VCR+AMD ± doxorubicin", "Favourable histology stage I: VCR+AMD x18wk", "No specific targeted therapy established"],
    "SIX2":   ["Standard SIOP/COG Wilms protocol", "SIX2 blastemal same as SIX1 treatment", "No specific targeted therapy"],
    "DICER1": ["PPB type I: surveillance (CT chest); type II/III: VCR+AMD+doxorubicin+etoposide", "Cystic nephroma: nephrectomy or nephron-sparing resection", "Cervical ERMS: fertility-sparing surgery if feasible + VAC", "SLCT ovary: salpingo-oophorectomy + BEP if malignant"],
    "TP53":   ["Anaplastic Wilms diffuse: UH-1 (VCR+AMD+doxo+carboplatin+cyclophosphamide)", "IF TP53 germline LFS: OMIT RT, maximise surgery", "Stage IV diffuse anaplasia: intensify chemo, resect metastases", "WBMRI annual surveillance (NOT CT)", "Bilateral Wilms: nephron-sparing + chemo-only LFS"],
    "WTX":    ["Standard SIOP/COG protocol WTX-mutant Wilms", "Favourable histology: VCR+AMD x18wk", "No specific targeted therapy for WTX LOF"],
    "BRCA2":  ["FA-D1 bilateral Wilms: AVOID alkylating agents (cyclophosphamide/ifosfamide)", "Carboplatin preferred (less BMF toxicity vs cisplatin)", "Sequential bilateral nephron-sparing nephrectomy", "HSCT: curative for BMF component (HLA-matched unrelated)", "Sibling donors: DEB/MMC test ALL siblings before evaluation", "Androgens (oxymetholone): BMF bridge to HSCT"],
}

# Surveillance protocols
SURVEILLANCE_PROTOCOLS = {
    "WT1":    ["US kidneys q3M until age 8yr (DDS/WAGR high risk)", "No CT/X-ray ionising screening", "Pelvic US + MRI gonadal streak (FS/DDS 46XY)", "Renal function q6M (DDS ESRD monitoring)", "Annual ophthal (WAGR aniridia)", "Cascade: first-degree relatives testing"],
    "CDKN1C": ["US kidneys q3M until age 8yr (Wilms)", "AFP q3M until age 4yr (hepatoblastoma)", "18F-DOPA PET: neonatal HI (focal vs diffuse MANDATORY)", "Macroglossia: speech therapy + orthopaedic", "Genetic testing: BWS methylation panel (IC1/IC2/UPD) mandatory"],
    "SIX1":   ["Emerging: US q3M until 8yr if germline Q177R confirmed", "First-degree relatives: genetic testing Q177R", "No established syndromic surveillance beyond Wilms", "BOR1 (non-Q177R): audiology + renal function"],
    "SIX2":   ["Emerging: US q3M until 8yr germline Q177R", "SIX1+SIX2 panel simultaneously for predisposition testing", "First-degree testing", "No syndromic surveillance beyond Wilms for Q177R"],
    "DICER1": ["CT chest baseline + q2yr ALL DICER1 (PPB)", "Siblings <8yr: CT chest MANDATORY (PPB risk)", "US kidneys q3M until 8yr (cystic nephroma/Wilms)", "Pelvic US adolescent females (SLCT/cervical ERMS)", "Annual thyroid US (MNG)", "No RT children"],
    "TP53":   ["WBMRI annually NOT CT/PET (LFS)", "Annual brain MRI", "Annual breast MRI from age 20yr", "US abdomen q6M (adrenal)", "Colonoscopy from 25yr q2yr", "Anaplastic Wilms: test germline TP53 mandatory"],
    "WTX":    ["US q3M until age 8yr if germline LOF confirmed", "Standard Wilms post-treatment surveillance (CT/US per COG/SIOP)", "No specific syndromic surveillance beyond Wilms", "OSCS (GOF): skeletal survey + cranial imaging"],
    "BRCA2":  ["DEB/MMC chromosomal fragility: ALL potential sibling donors pre-HSCT MANDATORY", "FA-D1: complete blood count monthly (BMF monitoring)", "US kidneys q3M (bilateral Wilms)", "Medulloblastoma: brain MRI surveillance (FA-D1)", "Monoallelic BRCA2 (parents): annual breast MRI, consider PBSO 40-45yr"],
}


def _make_cohort(gene_idx: int, seed: int) -> list:
    """Generate 40 deterministic synthetic Wilms tumour predisposition patients."""
    rng = random.Random(seed)

    gene = ATLAS_GENES[gene_idx]["gene"]
    age_ranges = {
        "WT1":    (0.2, 6.0),   # Wilms typically <5yr; DDS early
        "CDKN1C": (0.2, 7.0),   # BWS Wilms <8yr
        "SIX1":   (0.5, 7.0),   # Q177R blastemal Wilms
        "SIX2":   (0.5, 7.0),   # Same as SIX1
        "DICER1": (0.2, 10.0),  # Cystic nephroma/Wilms range
        "TP53":   (0.5, 8.0),   # Anaplastic Wilms paediatric
        "WTX":    (0.5, 6.0),   # Somatic/germline early
        "BRCA2":  (0.2, 5.0),   # FA-D1 bilateral very early
    }
    lo, hi = age_ranges.get(gene, (0.5, 7.0))

    histologies = {
        "WT1":    ["Favourable", "Favourable", "Anaplastic focal", "Favourable", "Mixed"],
        "CDKN1C": ["Favourable", "Favourable", "Favourable", "Mixed", "Blastemal"],
        "SIX1":   ["Blastemal", "Blastemal", "Blastemal", "Epithelial", "Favourable"],
        "SIX2":   ["Blastemal", "Blastemal", "Epithelial", "Favourable", "Mixed"],
        "DICER1": ["Anaplastic diffuse", "Mixed", "Favourable", "Anaplastic diffuse", "Mixed"],
        "TP53":   ["Anaplastic diffuse", "Anaplastic diffuse", "Anaplastic focal", "Anaplastic diffuse", "Mixed"],
        "WTX":    ["Favourable", "Favourable", "Blastemal", "Mixed", "Favourable"],
        "BRCA2":  ["Favourable bilateral", "Favourable bilateral", "Mixed bilateral", "Blastemal bilateral", "Anaplastic bilateral"],
    }

    stages = [1, 1, 2, 2, 3, 3, 4, 4, 5] if gene == "BRCA2" else [1, 1, 1, 2, 2, 3, 3, 4, 4]

    patients = []
    for pid in range(1, 41):
        age = round(rng.uniform(lo, hi), 1)
        stage = rng.choice(stages)
        hlist = histologies.get(gene, ["Favourable", "Mixed", "Blastemal"])
        hist = rng.choice(hlist)
        bilateral = (
            (gene == "BRCA2" and rng.random() < 0.55) or
            (gene == "WT1" and rng.random() < 0.20) or
            (rng.random() < 0.07)
        )
        cr = (
            rng.random() < (0.90 if stage <= 2 else (0.72 if stage == 3 else (0.52 if stage == 4 else 0.68)))
        )
        patients.append({
            "patient_id": f"{gene}-{pid:03d}",
            "age_dx_yr": age,
            "stage": stage,
            "histology": hist,
            "bilateral": bilateral,
            "cr_achieved": cr,
        })
    return patients


def generate_overview() -> dict:
    """Generate Hereditary-Nephroblastoma-Wilms-Tumor-Atlas overview."""
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(len(ATLAS_GENES))]
    total = sum(len(c) for c in cohorts)

    gene_summaries = []
    for i, (gdef, cohort) in enumerate(zip(ATLAS_GENES, cohorts)):
        ages = [p["age_dx_yr"] for p in cohort]
        cr_pct = round(100 * sum(p["cr_achieved"] for p in cohort) / len(cohort))
        bilateral_pct = round(100 * sum(p["bilateral"] for p in cohort) / len(cohort))
        gene_summaries.append({
            "gene": gdef["gene"],
            "locus": gdef["locus"],
            "protein": gdef["protein"],
            "inheritance": gdef["inheritance"],
            "n": len(cohort),
            "mean_age_dx": round(sum(ages) / len(ages), 1),
            "cr_pct": cr_pct,
            "bilateral_pct": bilateral_pct,
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
            "surveillance_key": gdef["surveillance_key"],
        })

    return _json_safe({
        "atlas": "Hereditary-Nephroblastoma-Wilms-Tumor-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference: WT1-CDKN1C-SIX1-SIX2-DICER1-TP53-WTX-BRCA2",
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes": gene_summaries,
        "key_pathognomonic": {
            "WT1_DDS": "Diffuse mesangial sclerosis (DMS) + 46XY PSD + Wilms 95% PATHOGNOMONIC DDS",
            "WT1_WAGR": "Aniridia + Wilms 45-60% PATHOGNOMONIC WAGR (11p13 WT1+PAX6 deletion)",
            "WT1_FRASIER": "FSGS + XY gonadal dysgenesis PATHOGNOMONIC Frasier syndrome",
            "CDKN1C_BWS": "Macroglossia + omphalocele + hemihypertrophy triad PATHOGNOMONIC BWS",
            "DICER1_NEPHROMA": "Cystic nephroma PATHOGNOMONIC DICER1 -- Wilms progression risk",
            "TP53_ANAPLASTIC": "Anaplastic Wilms histology (diffuse/focal) >90% TP53 LOF PATHOGNOMONIC",
            "BRCA2_FA_D1": "Bilateral Wilms PATHOGNOMONIC FA-D1 (biallelic BRCA2); DEB/MMC test PATHOGNOMONIC FA",
        },
        "key_surveillance": {
            "US_Q3M": "Abdominal US every 3 months until age 8yr (all high-risk genes)",
            "NO_IONISING": "No CT/X-ray ionising for Wilms screening (MRI preferred if US equivocal)",
            "TP53_NO_RT": "AVOID RADIATION ABSOLUTELY if TP53 germline (LFS) -- omit Wilms RT",
            "FA_D1_DONOR": "Sibling donor exclusion MANDATORY FA-D1 before HSCT (DEB/MMC test all siblings)",
            "DICER1_CT_CHEST": "ALL DICER1 siblings <8yr: chest CT (PPB risk)",
        },
    })


def generate_breakdown() -> dict:
    """Per-gene breakdown for Wilms-Tumor-Predisposition-Atlas."""
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(len(ATLAS_GENES))]

    breakdown = []
    for i, (gdef, cohort) in enumerate(zip(ATLAS_GENES, cohorts)):
        ages = [p["age_dx_yr"] for p in cohort]
        cr_pct = round(100 * sum(p["cr_achieved"] for p in cohort) / len(cohort))
        bilateral_pct = round(100 * sum(p["bilateral"] for p in cohort) / len(cohort))

        # Stage distribution
        stage_counts = {}
        for p in cohort:
            stage_counts[p["stage"]] = stage_counts.get(p["stage"], 0) + 1
        stage_dist = {f"stage_{k}": round(100 * v / len(cohort)) for k, v in sorted(stage_counts.items())}

        # Histology distribution
        hist_counts = {}
        for p in cohort:
            h = p["histology"]
            hist_counts[h] = hist_counts.get(h, 0) + 1
        hist_dist = {k: round(100 * v / len(cohort)) for k, v in sorted(hist_counts.items(), key=lambda x: -x[1])}

        breakdown.append({
            "gene": gdef["gene"],
            "locus": gdef["locus"],
            "protein_size": gdef["protein_size"],
            "n": len(cohort),
            "mean_age_dx": round(sum(ages) / len(ages), 1),
            "cr_pct": cr_pct,
            "bilateral_pct": bilateral_pct,
            "stage_distribution_pct": stage_dist,
            "histology_distribution_pct": hist_dist,
            "key_distinctions": gdef["key_distinctions"],
            "tumor_types": TUMOR_TYPES[gdef["gene"]],
            "pathogenic_variants": PATHOGENIC_VARIANTS[gdef["gene"]],
            "treatment_protocols": TREATMENT_PROTOCOLS[gdef["gene"]],
            "surveillance_protocols": SURVEILLANCE_PROTOCOLS[gdef["gene"]],
            "per_patient": cohort[:10],  # sample
        })

    return _json_safe({
        "atlas": "Hereditary-Nephroblastoma-Wilms-Tumor-Predisposition-Atlas",
        "breakdown": breakdown,
    })


def generate_definitions() -> dict:
    """Clinical definitions for Wilms-Tumor-Predisposition-Atlas."""
    definitions = {}
    for gdef in ATLAS_GENES:
        g = gdef["gene"]
        definitions[g] = {
            "gene": g,
            "locus": gdef["locus"],
            "protein_size": gdef["protein_size"],
            "inheritance": gdef["inheritance"],
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
            "surveillance_key": gdef["surveillance_key"],
            "key_distinctions": gdef["key_distinctions"],
            "pathogenic_variants": PATHOGENIC_VARIANTS[g],
            "tumor_types": TUMOR_TYPES[g],
            "treatment_protocols": TREATMENT_PROTOCOLS[g],
            "surveillance_protocols": SURVEILLANCE_PROTOCOLS[g],
        }

    return _json_safe({
        "atlas": "Hereditary-Nephroblastoma-Wilms-Tumor-Predisposition-Atlas",
        "definitions": definitions,
        "key_rules": {
            "WT1_DDS_DMS_PATHOGNOMONIC": (
                "Diffuse mesangial sclerosis (DMS) PATHOGNOMONIC Denys-Drash syndrome -- "
                "WT1 missense zinc finger 2/3 (R394W most common): DMS + 46XY PSD + Wilms 95%; "
                "DMS on kidney biopsy: electron microscopy mesangial matrix expansion PATHOGNOMONIC; "
                "DDS ESRD by age 3-5yr: renal transplant planning from diagnosis; "
                "US q3M until age 8yr DDS (highest risk) -- nephron-sparing surgery bilateral Wilms aim"
            ),
            "WAGR_ANIRIDIA_PATHOGNOMONIC": (
                "Aniridia PATHOGNOMONIC WAGR syndrome -- 11p13 deletion (WT1+PAX6 co-deletion); "
                "Bilateral aniridia + Wilms risk 45-60%: paediatric ophthalmology + US q3M mandatory; "
                "Intellectual disability variable (WT1/PAX6 deletion extent); "
                "WAGR nephroblastomatosis: bilateral precursor lesions -> Wilms bilateral risk; "
                "NO ionising radiation (X-ray/CT) screening -- MRI if US equivocal"
            ),
            "BWS_TRIAD_CDKN1C": (
                "BWS triad PATHOGNOMONIC: macroglossia + omphalocele/exomphalos + hemihypertrophy; "
                "CDKN1C LOF (maternal) = BWS-IC2 region: ~40-50% CDKN1C within BWS; "
                "Neonatal hyperinsulinism: glucose >10mg/kg/min + diazoxide; 18F-DOPA PET focal vs diffuse MANDATORY; "
                "Wilms 7-10% BWS (AFP q3M until 4yr + US q3M until 8yr); "
                "IC1 hypermethylation highest Wilms risk (~28%): molecular subtype mandatory (not sequencing alone)"
            ),
            "TP53_ANAPLASTIC_WILMS_NO_RT": (
                "Anaplastic Wilms histology (diffuse/focal) >90% TP53 LOF PATHOGNOMONIC -- "
                "germline TP53 testing MANDATORY for all diffuse anaplasia Wilms (LFS evaluation); "
                "IF germline TP53 confirmed: OMIT radiotherapy from Wilms protocol (LFS absolute CI RT); "
                "UH-1 regimen diffuse anaplasia: VCR+AMD+doxorubicin+carboplatin+cyclophosphamide; "
                "WBMRI Toronto Protocol annually NOT CT/PET (LFS); "
                "TP53 R337H southern Brazil 1/300 carrier frequency (population founder)"
            ),
            "FA_D1_BILATERAL_WILMS_DONOR": (
                "FA-D1 (biallelic BRCA2): bilateral Wilms PATHOGNOMONIC; most severe FA subtype; "
                "DEB/MMC chromosomal fragility test PATHOGNOMONIC for FA diagnosis -- ALL potential donors; "
                "SIBLING DONOR EXCLUSION MANDATORY: ALL siblings tested before HSCT evaluation; "
                "AVOID alkylating agents (cyclophosphamide/ifosfamide) in FA-D1: severe BMF; "
                "Carboplatin preferred; HSCT curative for BMF component (not solid tumour); "
                "Bilateral nephron-sparing nephrectomy required (renal preservation FA-D1)"
            ),
            "DICER1_CYSTIC_NEPHROMA_PPB": (
                "Cystic nephroma PATHOGNOMONIC DICER1 -- can progress to anaplastic Wilms (wall blastema); "
                "PPB type I (cystic lung <2yr) PATHOGNOMONIC DICER1: ALL siblings <8yr chest CT mandatory; "
                "DICER1 RNase IIIb hotspot (E1705/D1709/E1813): pathognomonic second somatic hit; "
                "AVOID radiation in DICER1 children (developing organs + radiation sensitivity); "
                "Cervical ERMS adolescent: fertility-sparing surgery if feasible"
            ),
            "CASCADE_WILMS_PREDISPOSITION": (
                "WT1/CDKN1C/DICER1/TP53: first-degree relatives 50% risk cascade testing. "
                "SIX1/SIX2 Q177R: first-degree testing (familial clustering documented). "
                "WTX germline: brothers 50% hemizygous LOF risk; sisters carriers. "
                "BRCA2 FA-D1: BOTH parents obligate heterozygous BRCA2 -- HBOC counselling parents; "
                "siblings 25% biallelic FA-D1 risk + 50% heterozygous HBOC risk; "
                "DEB/MMC test ALL siblings before HSCT donor evaluation MANDATORY."
            ),
        },
        "cascade_testing_rule": (
            "WT1/CDKN1C/DICER1/TP53: AD -- 50% risk first-degree relatives. "
            "SIX1/SIX2 Q177R: familial clustering -- test first-degree relatives. "
            "WTX: X-linked -- brothers 50% hemizygous; sisters obligate carrier if maternal. "
            "BRCA2 FA-D1: both parents BRCA2 het (HBOC risk parents); "
            "siblings 25% FA-D1 + 50% HBOC het; DEB/MMC ALL siblings MANDATORY pre-HSCT. "
            "BWS CDKN1C IC2 LOF: maternal germline -- test maternal relatives; "
            "paternal UPD not heritable; IC1 methylation recurrence risk varies (4-6% de novo families)."
        ),
    })


def _json_safe(obj):
    """Ensure all values are JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2)[:3000])
    print("\n--- breakdown (first gene) ---")
    bd = generate_breakdown()
    print(json.dumps(bd["breakdown"][0], indent=2)[:2000])
