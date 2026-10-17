#!/usr/bin/env python3
"""Hereditary-MDS-Predisposition-Atlas -- Complete 8-Gene Reference
Myelodysplastic Syndrome (MDS) germline predisposition.

GATA2    (GATA2; 480aa; 3q21.3; AD LOF;
          GATA2 Deficiency / MonoMAC Syndrome (OMIM 614172);
          MDS 30-70x HIGHEST; monosomy 7/del(7q) PATHOGNOMONIC;
          NK lymphopenia + DC aplasia PATHOGNOMONIC;
          HSCT curative; HPV/EBV/NTM Q3M mandatory surveillance;
          seed SEED_BASE+0).
DDX41    (DDX41; 622aa; 5q35.3; AD LOF;
          Familial MDS/AML (OMIM 616362);
          Late-onset MDS/AML 55-65yr HALLMARK;
          R525H most common; HMA+venetoclax preferred;
          donor DDX41 germline testing MANDATORY;
          seed SEED_BASE+1).
SAMD9L   (SAMD9L; 1589aa; 7q21.2; AD GOF;
          ATAD Syndrome (OMIM 616832);
          pediatric MDS; monosomy 7 + cerebellar ataxia PATHOGNOMONIC;
          reversion mosaicism — UPD7/del7q PATHOGNOMONIC;
          HSCT corrects MDS NOT ataxia;
          seed SEED_BASE+2).
RUNX1    (RUNX1; 453aa; 21q22.12; AD LOF;
          Familial Platelet Disorder/AML — FPD (OMIM 601399);
          thrombocytopenia PATHOGNOMONIC; MDS/AML 30-50%;
          sibling donors CI — 50% carry RUNX1; azacitidine preferred;
          seed SEED_BASE+3).
ETV6     (ETV6; 452aa; 12p13.2; AD LOF;
          Thrombocytopenia-5/THC5 (OMIM 616216);
          thrombocytopenia + MDS 10-15%; avoid NSAIDs; avoid high-dose MTX;
          ETV6-RUNX1 NOT expected with germline ETV6;
          seed SEED_BASE+4).
ANKRD26  (ANKRD26; 1710aa; 10p12.1; AD LOF;
          Thrombocytopenia-2/THC2 (OMIM 188000);
          MDS/AML 5-10%; platelet count monitoring;
          5'-UTR promoter mutation — standard exon sequencing MISSES;
          seed SEED_BASE+5).
TERT     (TERT; 1132aa; 5p15.33; AD LOF;
          Dyskeratosis Congenita/TBD (OMIM 127550);
          telomere biology disorder; MDS/aplastic anemia;
          Hoyeraal-Hreidarsson severe biallelic; danazol/eltrombopag;
          AVOID alkylating (very short telomeres); seed SEED_BASE+6).
TP53     (p53; 393aa; 17p13.1; AD LOF;
          Li-Fraumeni Syndrome (OMIM 151623);
          LFS MDS 10-20x complex karyotype; monosomal karyotype;
          AVOID RADIATION ABSOLUTELY; WB-MRI Toronto annually;
          venetoclax backbone; seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3638-3645)
"""
import random

SEED_BASE = 3638

ATLAS_GENES = [
    {
        "gene": "GATA2",
        "protein": (
            "GATA2 -- 3q21.3 AD LOF -- 480aa -- "
            "GATA2-50kDa-Zinc-Finger-Transcription-Factor-"
            "NF1-NF2-ZnF-TAD-"
            "MDS-30-70x-HIGHEST-MonoMAC-Emberger-NK-DC-Aplasia-PATHOGNOMONIC-"
            "GATA2-OMIM-614172"
        ),
        "locus": "3q21.3",
        "protein_size": (
            "480 aa / ~50 kDa / 3q21.3 GATA2 MDS molecular context: "
            "STRUCTURE: "
            "  480 aa / ~50 kDa; zinc-finger transcription factor; master regulator of haematopoiesis; "
            "  N-terminal transactivation domain (TAD, aa 1-200): transcriptional activation; "
            "  N-zinc finger (NF1, aa 201-230): protein-protein interactions; "
            "  C-zinc finger (NF2, aa 317-345): DNA binding GATA consensus (WGATAR); "
            "  Basic domain (aa 346-360): nuclear localisation; "
            "  C-terminal domain (aa 361-480): stability; "
            "  GATA2 LOF → haematopoietic stem cell (HSC) aplasia; NK/DC depletion; monocytopenia; "
            "  Two hotspot regions: NF2 zinc finger mutations + LOF truncating mutations; "
            "GATA2 MDS CONTEXT: "
            "  GATA2 Deficiency: MDS 30-70x HIGHEST — monosomy 7 and del(7q) PATHOGNOMONIC; "
            "  MonoMAC Syndrome: monocytopenia + MAC infection; "
            "  Emberger Syndrome: lymphoedema + MDS; "
            "  NK lymphopenia (<100/µL): PATHOGNOMONIC GATA2 deficiency; "
            "  DC aplasia (CD11c+ DC and pDC absent); "
            "  HPV-related warts refractory; EBV+MDS; NTM infection; "
            "  HSCT curative for MDS component (NOT lymphoedema); "
            "CLINICAL RULES FOR GATA2-MDS: "
            "  HSCT CURATIVE — do NOT wait for MDS progression to transplant; "
            "  Monosomy 7/del(7q): flag for GATA2 germline even without warts/lymphoedema; "
            "  NK lymphopenia (<100/µL): GATA2 mandatory testing; "
            "  HPV vaccination MANDATORY (4-valent minimum); "
            "  EBV/CMV/NTM Q3M monitoring MANDATORY; "
            "  Sibling donors: GATA2 testing MANDATORY (50% risk); "
            "  Lymphoedema: HSCT does NOT cure — manage independently; "
            "KEY VARIANTS: "
            "  NF2 zinc-finger: p.Thr354Met (most common), p.Arg398Trp, p.Cys373Arg; "
            "  Intronic: c.1018-1G>A; upstream: p.Arg307* (common); "
            "  LOF truncating: p.Arg396*, p.Leu321* (various truncating); "
            "SURVEILLANCE: "
            "  CBC + differential quarterly; NK count baseline + annual; "
            "  HPV vaccination; EBV/CMV/NTM Q3M; BM aspirate q12M from diagnosis; "
            "  Sibling donor GATA2 testing; lymphoedema referral vascular; "
            "TARGETED Rx: "
            "  HSCT curative — allogeneic SC transplant; "
            "  Azacitidine bridge pre-HSCT if MDS progresses; "
            "  HPV-related lesions: imiquimod topical; "
            "  NTM: rifampicin-based; MAC prophylaxis azithromycin if NK<100/µL; "
        ),
        "syndrome": "GATA2 Deficiency / MonoMAC Syndrome / Emberger Syndrome",
        "inheritance": "AD LOF heterozygous",
        "mds_risk": "MDS 30-70x HIGHEST; monosomy 7/del(7q) PATHOGNOMONIC; NK lymphopenia (<100/µL) PATHOGNOMONIC; HSCT curative",
        "pathognomonic": "NK lymphopenia + monocytopenia + warts/HPV + lymphoedema + monosomy 7 = GATA2 deficiency PATHOGNOMONIC",
        "key_rule": "HSCT curative — do NOT delay; monosomy 7 always test GATA2; NK<100/µL GATA2 mandatory; donor GATA2 screen",
        "key_avoid": "Waiting for MDS progression before HSCT (GATA2 deficiency HSCT curative early); unscreened sibling donors",
        "surveillance": "CBC + NK quarterly; EBV/CMV/NTM Q3M; BM q12M; HPV vaccine mandatory; sibling GATA2 germline testing",
        "targeted_rx": "Allogeneic HSCT curative; azacitidine bridge pre-HSCT; HPV vaccine; azithromycin MAC prophylaxis NK<100",
    },
    {
        "gene": "DDX41",
        "protein": (
            "DDX41 -- 5q35.3 AD LOF -- 622aa -- "
            "DDX41-70kDa-DEAD-Box-RNA-Helicase-"
            "DEAD-Box-SAT-Helicase-DEXDc-HelC-"
            "Familial-MDS-AML-Late-Onset-55-65yr-HALLMARK-R525H-50pct-"
            "DDX41-OMIM-616362"
        ),
        "locus": "5q35.3",
        "protein_size": (
            "622 aa / ~70 kDa / 5q35.3 DDX41 MDS molecular context: "
            "STRUCTURE: "
            "  622 aa / ~70 kDa; DEAD-box RNA helicase; nuclear; "
            "  DEXDc domain (aa 160-380): ATPase/helicase; "
            "  HelC domain (aa 390-500): RNA unwinding; "
            "  SPRY domain (aa 1-150): protein interactions; "
            "  R525H (most common germline): LOF of helicase activity; "
            "  DDX41 functions: R-loop resolution; innate immune sensing (cGAS-STING); splicing; "
            "  LOF → R-loop accumulation → replication stress → MDS; "
            "  Biallelic somatic: second hit R525H + p.Asp140Glyfs truncating; "
            "DDX41 MDS CONTEXT: "
            "  Familial MDS/AML — HALLMARK: late onset 55-65yr median; "
            "  R525H (p.Arg525His): ~50% of DDX41 germline variants; "
            "  MDS morphology: refractory anaemia; excess blasts; dysplastic megakaryocytes; "
            "  AML progression 30-40% from MDS; median age AML transformation 68yr; "
            "  Biallelic somatic second hit in >80% MDS/AML; "
            "  DDX41 MDS: typically MDS-EB-2 or higher risk; SF3B1/SRSF2 co-mutation common; "
            "CLINICAL RULES FOR DDX41-MDS: "
            "  LATE ONSET HALLMARK — new MDS in patients >55yr: DDX41 testing; "
            "  HMA (azacitidine/decitabine) + venetoclax preferred first-line; "
            "  HSCT: fit patients — curative intent; "
            "  Donor DDX41 germline testing MANDATORY — 50% sibling carrier risk; "
            "  Family cascade testing: siblings + children >40yr; "
            "  R525H founder variant: Ashkenazi Jewish population 1 in 3,000; "
            "KEY VARIANTS: "
            "  R525H (p.Arg525His): ~50% of cases; "
            "  p.Asp140Glyfs: truncating — somatic second hit; "
            "  p.Tyr259Cys, p.Ile396Thr: rare missense; "
            "SURVEILLANCE: "
            "  CBC + differential annual >40yr; BM aspirate if cytopenia; "
            "  Family cascade testing siblings + children; "
            "  Annual full blood panel; "
            "TARGETED Rx: "
            "  HMA (azacitidine/decitabine) + venetoclax BCL-2i: preferred DDX41-MDS; "
            "  Allogeneic HSCT: fit patients curative; "
            "  Enasidenib/ivosidenib if IDH2/IDH1 co-mutation; "
        ),
        "syndrome": "Familial MDS/AML — DDX41 Predisposition",
        "inheritance": "AD LOF heterozygous (common); biallelic somatic second hit in MDS/AML",
        "mds_risk": "MDS/AML late onset 55-65yr HALLMARK; R525H 50%; AML progression 30-40% from MDS; biallelic second hit >80%",
        "pathognomonic": "New MDS/AML in patient >55yr + family history = DDX41 germline testing MANDATORY; R525H founder",
        "key_rule": "Late onset HALLMARK; HMA+venetoclax preferred; donor DDX41 MANDATORY; family cascade >40yr; R525H Ashkenazi",
        "key_avoid": "Missing DDX41 in late-onset MDS (age >55yr); using unscreened sibling as donor; skipping family cascade",
        "surveillance": "CBC annual >40yr; BM aspirate if cytopenia; sibling + children cascade testing; annual full blood panel",
        "targeted_rx": "Azacitidine/decitabine + venetoclax DDX41-MDS preferred; HSCT curative fit; IDH inhibitor if IDH co-mutation",
    },
    {
        "gene": "SAMD9L",
        "protein": (
            "SAMD9L -- 7q21.2 AD GOF -- 1589aa -- "
            "SAMD9L-175kDa-Antiviral-Restriction-Factor-"
            "SAM-OB-fold-"
            "ATAD-Syndrome-Pediatric-MDS-Monosomy7-Cerebellar-Ataxia-PATHOGNOMONIC-Reversion-Mosaicism-"
            "SAMD9L-OMIM-616832"
        ),
        "locus": "7q21.2",
        "protein_size": (
            "1589 aa / ~175 kDa / 7q21.2 SAMD9L MDS molecular context: "
            "STRUCTURE: "
            "  1589 aa / ~175 kDa; SAMD9L (sterile alpha motif domain containing 9-like); "
            "  SAM domain (aa 1-65): protein-protein interactions; "
            "  OB fold (aa 200-350): RNA/DNA binding; "
            "  Central domain (aa 400-1200): antiproliferative; "
            "  C-terminal domain (aa 1200-1589): antiviral; "
            "  GOF mechanism: SAMD9L GOF → hyperactivation of mRNA translation inhibition → "
            "  HSC proliferation arrest → selective pressure for monosomy 7 (loses one GOF allele); "
            "  Reversion mosaicism: cells acquire UPD7q or del7q to escape SAMD9L GOF; "
            "  Monosomy 7 in SAMD9L is a REVERSION event (PROTECTIVE somatic escape); "
            "SAMD9L MDS CONTEXT: "
            "  ATAD Syndrome (Ataxia-Pancytopenia-Thrombocytopenia-Adrenal insufficiency-Deafness): "
            "    Cerebellar ataxia + MDS + monosomy 7 PATHOGNOMONIC; "
            "  Pediatric MDS hallmark — age <10yr; monosomy 7 in >80%; "
            "  Reversion mosaicism: UPD7q or del7q in normal haematopoiesis → SAMD9L GOF escaped; "
            "  SAMD9 (sister gene 7q21.2): overlapping phenotype — distinguish by sequencing; "
            "  HSCT corrects MDS but does NOT correct cerebellar ataxia; "
            "  Adrenal insufficiency: hydrocortisone replacement — stress dosing in illness; "
            "CLINICAL RULES FOR SAMD9L-MDS: "
            "  PEDIATRIC MDS + monosomy 7 + cerebellar ataxia = SAMD9L/SAMD9 first; "
            "  HSCT corrects MDS — does NOT correct ataxia (counsel family explicitly); "
            "  Reversion mosaicism in normal cells is NORMAL — do not misread as somatic rescue; "
            "  Adrenal insufficiency: hydrocortisone stress dosing (10x Sick Day Rule) MANDATORY; "
            "  SAMD9 vs SAMD9L: both on 7q21.2 — full sequencing distinguishes; "
            "KEY VARIANTS: "
            "  Gain-of-function: p.Arg986Cys, p.His880Leu, p.Thr936Ile — most reported; "
            "  Novel missense: assess functional impact with in vitro assay; "
            "SURVEILLANCE: "
            "  CBC + differential q3M; BM aspirate q12M; neurological exam q6M; "
            "  Adrenal function (cortisol/ACTH) annual; audiology annual; "
            "  Reversion mosaicism: expected in blood — do NOT misinterpret; "
            "TARGETED Rx: "
            "  Allogeneic HSCT: curative for MDS component; "
            "  Hydrocortisone replacement adrenal insufficiency (Sick Day Rule 10x); "
            "  Cerebellar ataxia: physiotherapy + OT (HSCT does not help); "
            "  G-CSF: use with caution (may accelerate monosomy 7 clone); "
        ),
        "syndrome": "ATAD Syndrome (Ataxia-Pancytopenia-Adrenal-Deafness) / SAMD9L-MDS Predisposition",
        "inheritance": "AD GOF heterozygous",
        "mds_risk": "Pediatric MDS monosomy 7 >80%; cerebellar ataxia PATHOGNOMONIC; HSCT corrects MDS NOT ataxia; adrenal insufficiency",
        "pathognomonic": "Pediatric MDS + monosomy 7 + cerebellar ataxia = SAMD9L/SAMD9 PATHOGNOMONIC triad; reversion mosaicism",
        "key_rule": "HSCT corrects MDS NOT ataxia (counsel explicitly); adrenal Sick Day Rule 10x hydrocortisone; G-CSF CAUTION",
        "key_avoid": "Misinterpreting reversion mosaicism as somatic rescue; missing adrenal insufficiency; G-CSF accelerating clone",
        "surveillance": "CBC q3M; BM q12M; neuro q6M; adrenal function annual; audiology annual; reversion mosaicism expected",
        "targeted_rx": "HSCT curative MDS; hydrocortisone Sick Day Rule; physio/OT ataxia; G-CSF CAUTION monosomy 7 acceleration",
    },
    {
        "gene": "RUNX1",
        "protein": (
            "RUNX1 -- 21q22.12 AD LOF -- 453aa -- "
            "RUNX1-51kDa-Runt-Domain-Transcription-Factor-"
            "Runt-VWRPY-Transactivation-"
            "FPD-Thrombocytopenia-PATHOGNOMONIC-MDS-AML-30-50pct-Sibling-Donors-CI-"
            "RUNX1-OMIM-601399"
        ),
        "locus": "21q22.12",
        "protein_size": (
            "453 aa / ~51 kDa / 21q22.12 RUNX1 MDS molecular context: "
            "STRUCTURE: "
            "  453 aa / ~51 kDa; RUNX1/AML1 transcription factor; "
            "  Runt homology domain (aa 50-177): DNA binding + CBFβ heterodimerisation; "
            "  VWRPY motif (aa 450-454): Groucho/TLE repressor interaction; "
            "  Transactivation domain (aa 291-371); "
            "  RUNX1:CBFβ heterodimer: master regulator of myeloid + lymphoid differentiation; "
            "  LOF → impaired megakaryocyte differentiation + haematopoietic stem cell exit; "
            "  Second-hit somatic LOH or dominant-negative: MDS/AML transformation; "
            "RUNX1 MDS CONTEXT: "
            "  Familial Platelet Disorder with AML (FPD/AML, FPD-MM): "
            "    Mild-moderate thrombocytopenia (plt 50-150k) PATHOGNOMONIC; "
            "    MDS/AML 30-50% lifetime; AML progression median 33yr; "
            "    Dense granule deficiency in platelets (aggregation defect); "
            "    MDS-EB or AML-NOS RUNX1 mutated; "
            "  Sibling donors: 50% carry RUNX1 — testing MANDATORY; "
            "  ASXL1/TET2/SRSF2 co-mutations: accelerate MDS progression in FPD; "
            "CLINICAL RULES FOR RUNX1-MDS: "
            "  THROMBOCYTOPENIA in family + MDS = RUNX1 germline testing MANDATORY; "
            "  Sibling donors ABSOLUTELY CI without RUNX1 germline testing; "
            "  Azacitidine preferred over intensive chemotherapy (preserve HSC function); "
            "  HSCT: fit patients curative; sibling donor MUST be RUNX1-negative; "
            "  Dense granule deficiency: bleeding risk — transfuse before invasive procedures; "
            "KEY VARIANTS: "
            "  Runt domain missense: p.Arg204Gln, p.Arg162Gln, p.Asp171Asn; "
            "  Frameshift/nonsense (haploinsufficiency): p.Lys83*, various exon 3-8; "
            "SURVEILLANCE: "
            "  CBC + platelet function q6M; BM aspirate if cytopenia or >MDS threshold; "
            "  Annual bone marrow for FPD patients >18yr; "
            "  Sibling donor testing BEFORE any haematopoietic stem cell donor evaluation; "
            "TARGETED Rx: "
            "  Azacitidine preferred first-line MDS; "
            "  Allogeneic HSCT curative (non-sibling or RUNX1-negative sibling); "
            "  Avoid aspirin/NSAIDs (dense granule deficiency); "
            "  Enasidenib/ivosidenib if IDH co-mutation; "
        ),
        "syndrome": "Familial Platelet Disorder with AML (FPD) — RUNX1 Predisposition",
        "inheritance": "AD LOF heterozygous",
        "mds_risk": "Thrombocytopenia PATHOGNOMONIC; MDS/AML 30-50% lifetime; sibling donors CI (50% carry RUNX1); AML median 33yr",
        "pathognomonic": "Family thrombocytopenia (plt 50-150k) + dense granule deficiency + MDS = FPD RUNX1 PATHOGNOMONIC",
        "key_rule": "Sibling donors ABSOLUTELY CI without RUNX1 testing; azacitidine preferred; avoid aspirin dense granule deficiency",
        "key_avoid": "Using untested sibling as stem cell donor; intensive chemotherapy before HSCT planning; aspirin/NSAIDs FPD",
        "surveillance": "CBC q6M; platelet function; BM aspirate annual >18yr; sibling donor RUNX1 testing before evaluation",
        "targeted_rx": "Azacitidine first-line; HSCT curative non-sibling or RUNX1-negative sibling; IDH inhibitor if co-mutation",
    },
    {
        "gene": "ETV6",
        "protein": (
            "ETV6 -- 12p13.2 AD LOF -- 452aa -- "
            "ETV6-57kDa-ETS-Transcription-Factor-"
            "PNT-DBD-"
            "THC5-Thrombocytopenia-MDS-10-15pct-Avoid-NSAIDs-Avoid-HighDose-MTX-"
            "ETV6-OMIM-616216"
        ),
        "locus": "12p13.2",
        "protein_size": (
            "452 aa / ~57 kDa / 12p13.2 ETV6 MDS molecular context: "
            "STRUCTURE: "
            "  452 aa / ~57 kDa; ETV6/TEL transcription factor; ETS family; "
            "  PNT domain (aa 43-125): pointed/SAM domain — protein oligomerisation; "
            "  DBD (aa 335-416): ETS DNA-binding domain (GGAA/T motif); "
            "  ETV6 is tumour suppressor: represses ETS targets involved in proliferation; "
            "  LOF → haematopoietic progenitor dysregulation; platelet biogenesis impaired; "
            "  Biallelic somatic: ETV6-RUNX1 t(12;21) in B-ALL — NOT expected with germline ETV6 LOF; "
            "ETV6 MDS CONTEXT: "
            "  THC5 (Thrombocytopenia-5): mild-moderate thrombocytopenia (plt 30-150k); "
            "  MDS/haematologic malignancy 10-15% lifetime (B-ALL, MDS, AML); "
            "  Note: ETV6-RUNX1 translocation found in sporadic childhood B-ALL — NOT expected when "
            "    ETV6 germline LOF present (one allele already lost); "
            "  High-dose methotrexate neurotoxicity risk (ETV6 LOF may sensitise to MTX); "
            "  Avoid NSAIDs: platelet aggregation further impaired; "
            "CLINICAL RULES FOR ETV6-MDS: "
            "  THROMBOCYTOPENIA + family history MDS/ALL = ETV6 testing; "
            "  ETV6-RUNX1 translocation: sporadic B-ALL — confirm NOT in germline ETV6 patients; "
            "  Avoid NSAIDs and high-dose MTX in ETV6 germline carriers; "
            "  Annual CBC surveillance; BM aspirate if count falls; "
            "KEY VARIANTS: "
            "  p.Arg359* (most common), p.Asp262Asn, p.Pro214Leu: thrombocytopenia; "
            "  DBD missense: reduce DNA binding → LOF; "
            "SURVEILLANCE: "
            "  CBC annual; platelet function assessment; BM if cytopenia; "
            "  Family cascade testing (AD — 50% child risk); "
            "TARGETED Rx: "
            "  MDS: azacitidine or HSCT depending on risk category; "
            "  Avoid NSAIDs + high-dose MTX germline ETV6 carriers; "
            "  Eltrombopag thrombocytopenia if count <20k; "
        ),
        "syndrome": "Thrombocytopenia-5 (THC5) / ETV6-MDS Predisposition",
        "inheritance": "AD LOF heterozygous",
        "mds_risk": "Thrombocytopenia (plt 30-150k); MDS/haematologic malignancy 10-15%; ETV6-RUNX1 NOT expected germline ETV6",
        "pathognomonic": "Inherited thrombocytopenia + family MDS/B-ALL = ETV6 THC5; ETV6-RUNX1 absent in germline ETV6 LOF",
        "key_rule": "Avoid NSAIDs + high-dose MTX; annual CBC; ETV6-RUNX1 absent with germline ETV6; HSCT for MDS progression",
        "key_avoid": "Prescribing NSAIDs ETV6 (platelet dysfunction); high-dose MTX; missing MDS surveillance annual CBC",
        "surveillance": "CBC annual; platelet function; BM if cytopenia; family cascade testing (AD 50% risk); annual surveillance",
        "targeted_rx": "Azacitidine MDS first-line; HSCT MDS progression; eltrombopag plt<20k; avoid NSAIDs + high-dose MTX",
    },
    {
        "gene": "ANKRD26",
        "protein": (
            "ANKRD26 -- 10p12.1 AD LOF -- 1710aa -- "
            "ANKRD26-190kDa-Ankyrin-Repeat-Domain-Protein-"
            "ANK-repeats-"
            "THC2-Thrombocytopenia2-MDS-AML-5-10pct-5UTR-Promoter-Standard-Sequencing-MISSES-"
            "ANKRD26-OMIM-188000"
        ),
        "locus": "10p12.1",
        "protein_size": (
            "1710 aa / ~190 kDa / 10p12.1 ANKRD26 MDS molecular context: "
            "STRUCTURE: "
            "  1710 aa / ~190 kDa; ankyrin repeat domain-containing protein 26; "
            "  Multiple ANK repeats (aa 1-500): protein-protein interactions; "
            "  C-terminal domain (aa 1200-1710): poorly characterised; "
            "  5'-UTR promoter region: ALL pathogenic variants cluster in 5'-UTR; "
            "  LOF mechanism: 5'-UTR mutations prevent RUNX1/FLI1 binding → ANKRD26 overexpression; "
            "  ANKRD26 overexpression → impaired megakaryocyte differentiation → thrombocytopenia; "
            "ANKRD26 MDS CONTEXT: "
            "  THC2 (Thrombocytopenia-2): moderate thrombocytopenia (plt 30-100k); "
            "  CRITICAL: ALL pathogenic variants in 5'-UTR promoter — standard exon sequencing MISSES; "
            "  MDS/AML 5-10% lifetime; CML reported; "
            "  Platelet function: normal aggregation (unlike RUNX1); platelet count is main indicator; "
            "  ANKRD26 5'-UTR panel: specifically required for diagnosis; "
            "CLINICAL RULES FOR ANKRD26-MDS: "
            "  THROMBOCYTOPENIA: order 5'-UTR ANKRD26 panel (NOT just exome/gene panel); "
            "  Standard WES/WGS may miss 5'-UTR variants — request targeted ANKRD26 5-UTR assay; "
            "  Annual CBC surveillance; BM aspirate if platelet count falls or cytopenia develops; "
            "  MDS development: azacitidine or HSCT by IPSS-R risk category; "
            "KEY VARIANTS: "
            "  c.-127A>T, c.-128A>T, c.-126A>G, c.-153C>T: 5'-UTR (ALL variants here); "
            "  ALL are promoter mutations — NOT exon coding variants; "
            "SURVEILLANCE: "
            "  CBC annual; BM if cytopenia or MDS suspected; "
            "  Family cascade (AD); 5'-UTR ANKRD26 confirmatory test; "
            "TARGETED Rx: "
            "  Azacitidine MDS first-line; HSCT MDS progression by IPSS-R; "
            "  Eltrombopag thrombocytopenia if count <20k; "
            "  Avoid routine bone marrow biopsies without clinical indication; "
        ),
        "syndrome": "Thrombocytopenia-2 (THC2) / ANKRD26-MDS Predisposition",
        "inheritance": "AD LOF — ALL variants in 5'-UTR promoter (NOT coding exons)",
        "mds_risk": "Thrombocytopenia (plt 30-100k); MDS/AML 5-10%; CML reported; 5'-UTR standard sequencing MISSES — 5-UTR panel needed",
        "pathognomonic": "Thrombocytopenia + normal platelet aggregation + family MDS = ANKRD26 5-UTR testing (NOT standard exome)",
        "key_rule": "5'-UTR ANKRD26 panel MANDATORY (standard exon sequencing misses all variants); annual CBC; IPSS-R for HSCT",
        "key_avoid": "Relying on standard WES/WGS (misses 5-UTR variants); missing MDS surveillance; delaying HSCT for IPSS-R high",
        "surveillance": "CBC annual; BM if cytopenia; family cascade AD; 5'-UTR ANKRD26 confirmatory assay mandatory",
        "targeted_rx": "Azacitidine MDS IPSS-R int/high; HSCT curative; eltrombopag plt<20k; 5-UTR panel first for diagnosis",
    },
    {
        "gene": "TERT",
        "protein": (
            "TERT -- 5p15.33 AD LOF -- 1132aa -- "
            "TERT-127kDa-Telomerase-Reverse-Transcriptase-"
            "TEN-TRBD-RT-CTE-"
            "TBD-DC-MDS-Aplastic-Anemia-Short-Telomeres-AVOID-ALKYLATING-Danazol-Eltrombopag-"
            "TERT-OMIM-127550"
        ),
        "locus": "5p15.33",
        "protein_size": (
            "1132 aa / ~127 kDa / 5p15.33 TERT MDS molecular context: "
            "STRUCTURE: "
            "  1132 aa / ~127 kDa; TERT — catalytic subunit of telomerase; "
            "  TEN domain (aa 1-325): RNA-template interaction; "
            "  TRBD (aa 325-567): TERC RNA binding; "
            "  RT (reverse transcriptase) domain (aa 567-870): catalytic; CX2CX1-8H metal cluster; "
            "  CTE (aa 870-1132): template alignment + processivity; "
            "  TERT + TERC (RNA) + dyskerin (DKC1): telomerase holoenzyme; "
            "  LOF → short telomeres → haematopoietic stem cell exhaustion → "
            "    aplastic anaemia, MDS, or pulmonary fibrosis (PF); "
            "  Anticipation: telomeres shorten with each generation (earlier/more severe); "
            "TERT MDS CONTEXT: "
            "  Telomere Biology Disorder (TBD) / Dyskeratosis Congenita (DC): "
            "    Mucocutaneous triad (nail dystrophy + leukoplakaía + reticulate skin pigmentation) — "
            "    present only in classic DC; TBD broader spectrum; "
            "  MDS/aplastic anaemia: haematopoietic failure dominant; "
            "  Pulmonary fibrosis (PF): 20-30%; may precede MDS; liver disease; "
            "  Hoyeraal-Hreidarsson: severe biallelic — cerebellar hypoplasia + IUGR; "
            "  Androgen therapy (danazol): telomerase activation; response 70%; "
            "  HSCT: curative for haematopoietic failure; lung/liver NOT cured by HSCT; "
            "  Alkylating agents: very short telomeres → extreme sensitivity → AVOID; "
            "  Fludarabine-based RIC conditioning preferred (NOT cyclophosphamide-based); "
            "CLINICAL RULES FOR TERT-MDS: "
            "  APLASTIC ANAEMIA + PF = TERT/TERC telomere biology testing MANDATORY; "
            "  Short telomeres by flow-FISH: <1st percentile in ≥3 cell types PATHOGNOMONIC TBD; "
            "  AVOID alkylating agents — ultra-short telomeres → fatal marrow aplasia; "
            "  Fludarabine-based RIC (NOT cyclophosphamide) for HSCT conditioning; "
            "  Androgen therapy (danazol 200mg TID): 70% response; liver enzyme monitoring; "
            "  Eltrombopag: TPO agonist; bridging to HSCT; "
            "KEY VARIANTS: "
            "  p.His412Tyr, p.Arg631Gln: common AD LOF; "
            "  Biallelic: Hoyeraal-Hreidarsson severe; "
            "SURVEILLANCE: "
            "  CBC + telomere length (flow-FISH) annual; PFTs q12M; LFTs if danazol; "
            "  Lung HRCT if PF symptoms; BM aspirate if cytopenia; "
            "  Family cascade: anticipation — each generation earlier onset; "
            "TARGETED Rx: "
            "  Danazol (androgen therapy) 200mg TID — 70% response; liver monitoring; "
            "  Eltrombopag TPO agonist; "
            "  Fludarabine-based RIC HSCT (AVOID cyclophosphamide); "
            "  AVOID alkylating agents absolutely; "
        ),
        "syndrome": "Telomere Biology Disorder (TBD) / Dyskeratosis Congenita (DC) / TERT-MDS Predisposition",
        "inheritance": "AD LOF heterozygous (TBD); AR biallelic = Hoyeraal-Hreidarsson severe",
        "mds_risk": "MDS/aplastic anemia; PF 20-30%; short telomeres <1st percentile PATHOGNOMONIC; AVOID alkylating absolutely",
        "pathognomonic": "Aplastic anemia + PF + reticulate pigmentation = TERT/TERC TBD; flow-FISH <1st percentile PATHOGNOMONIC",
        "key_rule": "AVOID alkylating ABSOLUTELY; fludarabine-based RIC conditioning; danazol 70% response; eltrombopag bridge",
        "key_avoid": "Cyclophosphamide-based HSCT conditioning (ultra-short telomeres → fatal aplasia); alkylating agents ABSOLUTELY",
        "surveillance": "CBC annual; telomere flow-FISH; PFTs q12M; lung HRCT PF; LFTs danazol; family cascade anticipation",
        "targeted_rx": "Danazol 200mg TID (70% response); eltrombopag; fludarabine RIC HSCT; AVOID alkylating absolutely",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "DBD-Tetramerisation-TAD-PrD-"
            "LFS-MDS-10-20x-Complex-Karyotype-AVOID-RADIATION-ABSOLUTELY-WB-MRI-Toronto-Venetoclax-"
            "TP53-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 MDS molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; tumour suppressor; tetramer transcription factor; "
            "  TAD1 (aa 1-40): p300/CBP; MDM2 binding; "
            "  TAD2 (aa 40-61): TFIID; "
            "  PRD (aa 62-91): protein-protein interaction; "
            "  DBD (aa 94-292): DNA-binding; hotspot mutations R175H/G245S/R248W/R248Q/R249S/R273H; "
            "  NLS (aa 316-325); "
            "  Tetramerisation (aa 323-356); "
            "  REG (aa 364-393): MDM2 interaction; "
            "  Wild-type p53: apoptosis, cell-cycle arrest, senescence; "
            "  LOF/GOF: tumour suppressor lost + dominant-negative over WT p53; "
            "LFS MDS CONTEXT: "
            "  LFS: germline TP53 LOF; MDS 10-20x; complex karyotype / monosomal karyotype; "
            "  TP53 biallelic MDS (second hit somatic LOH): ELN adverse, therapy-related; "
            "  Therapy-related MDS after RT/alkylating in LFS patients: highest risk; "
            "  Complex karyotype MDS: venetoclax + decitabine preferred; "
            "CLINICAL RULES FOR TP53-MDS: "
            "  AVOID RADIATION ABSOLUTELY — WB-MRI Toronto Protocol annually; "
            "  WB-MRI: annual from MDS diagnosis; excludes RT-related secondary tumours; "
            "  Venetoclax backbone: BCL-2 inhibition effective TP53-mutant MDS; "
            "  APR-246 (eprenetapopt): p53 reactivation; HMA combination Phase III; "
            "  Allogeneic HSCT: standard conditioning — AVOID RT-based conditioning; "
            "KEY VARIANTS: "
            "  Dominant hotspots: R175H, R248W, R248Q, R273H, R273C (DBD); "
            "  LOF: nonsense, frameshift, splice site throughout; "
            "SURVEILLANCE: "
            "  WB-MRI annually (Toronto Protocol); annual CBC + bone marrow; "
            "  First-degree family members: TP53 germline testing; "
            "TARGETED Rx: "
            "  Venetoclax + decitabine: TP53-mutant MDS preferred; "
            "  APR-246 + azacitidine: Phase III combination; "
            "  HSCT: AVOID RT-based conditioning (LFS); fludarabine/busulfan; "
            "  WB-MRI cancer surveillance (NOT CT/PET — radiation exposure); "
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) / TP53-MDS Predisposition",
        "inheritance": "AD LOF heterozygous",
        "mds_risk": "MDS 10-20x; complex/monosomal karyotype; therapy-related MDS HIGHEST after RT/alkylating; biallelic somatic second hit",
        "pathognomonic": "LFS + MDS complex karyotype = TP53 germline; biallelic TP53 MDS = ELN adverse; WB-MRI NOT CT annually",
        "key_rule": "AVOID RADIATION ABSOLUTELY; WB-MRI Toronto Protocol; venetoclax+decitabine preferred; HSCT avoid RT conditioning",
        "key_avoid": "Any radiation (RT in LFS = ABSOLUTE CI); CT surveillance (use WB-MRI); RT-based HSCT conditioning LFS",
        "surveillance": "WB-MRI annually (NOT CT/PET); annual CBC; BM annually; family TP53 germline cascade; paediatric surveillance",
        "targeted_rx": "Venetoclax+decitabine TP53-MDS preferred; APR-246+azacitidine Phase III; HSCT fludarabine/busulfan not RT",
    },
]

KEY_CLINICAL_DISTINCTIONS = [
    "GATA2-HSCT-CURATIVE-DO-NOT-DELAY: GATA2 deficiency MDS 30-70x HIGHEST — HSCT curative early; do NOT wait for progression; monosomy 7/NK lymphopenia = test GATA2; donor GATA2 MANDATORY.",
    "DDX41-LATE-ONSET-HALLMARK: New MDS in patient >55yr = DDX41 testing; R525H founder Ashkenazi; HMA+venetoclax preferred; donor DDX41 germline MANDATORY; family cascade >40yr.",
    "SAMD9L-HSCT-CORRECTS-MDS-NOT-ATAXIA: ATAD Syndrome pediatric MDS + monosomy 7 + cerebellar ataxia PATHOGNOMONIC; HSCT corrects MDS NOT ataxia — counsel explicitly; adrenal Sick Day Rule 10x hydrocortisone.",
    "RUNX1-SIBLING-DONORS-ABSOLUTELY-CI: FPD thrombocytopenia PATHOGNOMONIC; sibling donors CI without RUNX1 test (50% carry); azacitidine preferred; avoid aspirin/NSAIDs dense granule deficiency.",
    "ETV6-STANDARD-SEQUENCING-MISSES-AND-NO-ETV6-RUNX1: THC5 thrombocytopenia + MDS 10-15%; avoid NSAIDs + high-dose MTX; ETV6-RUNX1 translocation absent in germline ETV6 LOF carriers.",
    "ANKRD26-5UTR-PANEL-MANDATORY: All ANKRD26 variants in 5'-UTR promoter — standard WES/exome misses; MUST order targeted ANKRD26 5'-UTR assay; MDS/AML 5-10% lifetime surveillance.",
    "TERT-AVOID-ALKYLATING-ABSOLUTELY: TBD/DC MDS + aplastic anemia — AVOID alkylating ABSOLUTELY (ultra-short telomeres); fludarabine-based RIC conditioning (NOT cyclophosphamide); danazol 70% response.",
    "TP53-AVOID-RADIATION-ABSOLUTELY-MDS: LFS TP53 germline MDS 10-20x — ANY radiation ABSOLUTE CI; venetoclax+decitabine TP53-MDS; WB-MRI NOT CT annually; APR-246 Phase III combination.",
]


def _make_cohort(gene_def: dict, seed: int) -> dict:
    """Generate a 40-patient MDS cohort for one predisposition gene."""
    rng = random.Random(seed)

    gene  = gene_def["gene"]
    locus = gene_def["locus"]
    inh   = gene_def["inheritance"]

    n = 40

    # Average age at MDS diagnosis (per-gene, Gaussian σ≈12, clamp 2-80)
    avg_ages = {
        "GATA2":   30, "DDX41":  60, "SAMD9L":  8, "RUNX1":  38,
        "ETV6":    45, "ANKRD26": 50, "TERT":   42, "TP53":   48,
    }
    mu = avg_ages.get(gene, 40)
    sigma = 8 if gene in ("SAMD9L",) else 12
    ages = sorted([rng.gauss(mu, sigma) for _ in range(n)])
    ages = [max(2, min(80, round(a, 1))) for a in ages]

    # MDS subtype distribution per gene
    subtype_dist = {
        "GATA2":   [("MDS-MLD", 0.30), ("MDS-EB-1", 0.55), ("MDS-EB-2", 0.80), ("MDS/AML", 1.00)],
        "DDX41":   [("MDS-MLD", 0.20), ("MDS-EB-1", 0.40), ("MDS-EB-2", 0.65), ("MDS/AML", 1.00)],
        "SAMD9L":  [("MDS-SLD", 0.40), ("MDS-MLD", 0.70), ("MDS-EB-1", 0.90), ("MDS/AML", 1.00)],
        "RUNX1":   [("MDS-SLD", 0.25), ("MDS-MLD", 0.50), ("MDS-EB-1", 0.75), ("MDS/AML", 1.00)],
        "ETV6":    [("MDS-SLD", 0.40), ("MDS-MLD", 0.65), ("MDS-EB-1", 0.85), ("MDS/AML", 1.00)],
        "ANKRD26": [("MDS-SLD", 0.45), ("MDS-MLD", 0.70), ("MDS-EB-1", 0.85), ("MDS/AML", 1.00)],
        "TERT":    [("Aplastic-Anaemia", 0.35), ("MDS-MLD", 0.60), ("MDS-EB-1", 0.80), ("MDS/AML", 1.00)],
        "TP53":    [("MDS-EB-1", 0.25), ("MDS-EB-2", 0.50), ("MDS/AML", 0.75), ("tMDS-Complex-Karyotype", 1.00)],
    }
    def pick_subtype(gene_key):
        dist = subtype_dist.get(gene_key, [("MDS-MLD", 0.50), ("MDS-EB-1", 0.80), ("MDS/AML", 1.00)])
        r = rng.random()
        for label, threshold in dist:
            if r < threshold:
                return label
        return dist[-1][0]

    subtypes = [pick_subtype(gene) for _ in range(n)]
    mds_mld_n   = sum(1 for s in subtypes if "MLD" in s or "SLD" in s)
    mds_eb_n    = sum(1 for s in subtypes if "EB" in s)
    aml_prog_n  = sum(1 for s in subtypes if "AML" in s)
    aplastic_n  = sum(1 for s in subtypes if "Aplastic" in s)

    # Cytogenetics per gene
    cytogen_dist = {
        "GATA2":   [("monosomy7", 0.50), ("del7q", 0.75), ("trisomy8", 0.85), ("normal", 1.00)],
        "DDX41":   [("normal", 0.45), ("del5q", 0.65), ("complex", 0.80), ("monosomy7", 1.00)],
        "SAMD9L":  [("monosomy7", 0.65), ("del7q-UPD7", 0.85), ("normal", 1.00)],
        "RUNX1":   [("normal", 0.40), ("trisomy8", 0.60), ("del5q", 0.75), ("complex", 1.00)],
        "ETV6":    [("normal", 0.50), ("del12p", 0.70), ("trisomy8", 0.85), ("complex", 1.00)],
        "ANKRD26": [("normal", 0.55), ("trisomy8", 0.75), ("del5q", 0.88), ("complex", 1.00)],
        "TERT":    [("normal", 0.35), ("monosomy7", 0.55), ("complex", 0.75), ("MK", 1.00)],
        "TP53":    [("complex", 0.45), ("MK-monosomal", 0.70), ("del17p", 0.85), ("normal", 1.00)],
    }
    def pick_cytogen(gene_key):
        dist = cytogen_dist.get(gene_key, [("normal", 0.50), ("complex", 0.80), ("monosomy7", 1.00)])
        r = rng.random()
        for label, threshold in dist:
            if r < threshold:
                return label
        return dist[-1][0]

    cytogens = [pick_cytogen(gene) for _ in range(n)]
    monosomy7_n = sum(1 for c in cytogens if "monosomy7" in c or "del7" in c)
    complex_n   = sum(1 for c in cytogens if "complex" in c or "MK" in c)

    # IPSS-R risk category per gene
    ipss_dist = {
        "GATA2":   [("Low", 0.15), ("Int", 0.45), ("High", 0.75), ("VHigh", 1.00)],
        "DDX41":   [("Int", 0.20), ("High", 0.55), ("VHigh", 1.00)],
        "SAMD9L":  [("Low", 0.30), ("Int", 0.65), ("High", 0.90), ("VHigh", 1.00)],
        "RUNX1":   [("Low", 0.25), ("Int", 0.55), ("High", 0.80), ("VHigh", 1.00)],
        "ETV6":    [("VLow", 0.20), ("Low", 0.50), ("Int", 0.80), ("High", 1.00)],
        "ANKRD26": [("VLow", 0.25), ("Low", 0.55), ("Int", 0.80), ("High", 1.00)],
        "TERT":    [("Low", 0.20), ("Int", 0.45), ("High", 0.70), ("VHigh", 1.00)],
        "TP53":    [("High", 0.30), ("VHigh", 1.00)],
    }
    def pick_ipss(gene_key):
        dist = ipss_dist.get(gene_key, [("Int", 0.30), ("High", 0.65), ("VHigh", 1.00)])
        r = rng.random()
        for label, threshold in dist:
            if r < threshold:
                return label
        return dist[-1][0]

    ipss_cats = [pick_ipss(gene) for _ in range(n)]
    ipss_high_vh_n = sum(1 for i in ipss_cats if i in ("High", "VHigh"))

    # Treatment
    hma_n          = round(rng.gauss(0.55, 0.08) * n)
    venetoclax_n   = round(rng.gauss(0.38, 0.08) * n)
    hsct_n         = round(rng.gauss(0.32, 0.07) * n)
    targeted_n     = round(rng.gauss(0.22, 0.06) * n)
    danazol_n      = round(rng.gauss(0.30, 0.08) * n) if gene == "TERT" else 0
    cr_n           = round(rng.gauss(0.42, 0.07) * n)
    os2_n          = round(rng.gauss(0.55, 0.07) * n)
    cytopenia_n    = round(rng.gauss(0.78, 0.06) * n)
    thrombocytopenia_n = round(rng.gauss(0.72, 0.06) * n)
    pulmonary_n    = round(rng.gauss(0.22, 0.05) * n) if gene == "TERT" else round(rng.gauss(0.08, 0.03) * n)

    # Clamp
    def _clamp(v): return max(0, min(n, int(v)))
    hma_n             = _clamp(hma_n)
    venetoclax_n      = _clamp(venetoclax_n)
    hsct_n            = _clamp(hsct_n)
    targeted_n        = _clamp(targeted_n)
    danazol_n         = _clamp(danazol_n)
    cr_n              = _clamp(cr_n)
    os2_n             = _clamp(os2_n)
    cytopenia_n       = _clamp(cytopenia_n)
    thrombocytopenia_n = _clamp(thrombocytopenia_n)
    pulmonary_n       = _clamp(pulmonary_n)

    return {
        "gene":                gene,
        "locus":               locus,
        "inheritance":         inh,
        "n":                   n,
        "age_min":             ages[0],
        "age_max":             ages[-1],
        "age_median":          ages[n // 2],
        "mds_mld_n":           mds_mld_n,
        "mds_eb_n":            mds_eb_n,
        "aml_prog_n":          aml_prog_n,
        "aplastic_n":          aplastic_n,
        "monosomy7_n":         monosomy7_n,
        "complex_karyotype_n": complex_n,
        "ipss_high_vh_n":      ipss_high_vh_n,
        "cytopenia_n":         cytopenia_n,
        "thrombocytopenia_n":  thrombocytopenia_n,
        "pulmonary_n":         pulmonary_n,
        "hma_n":               hma_n,
        "venetoclax_n":        venetoclax_n,
        "hsct_n":              hsct_n,
        "targeted_n":          targeted_n,
        "danazol_n":           danazol_n,
        "cr_n":                cr_n,
        "os2_n":               os2_n,
        "mds_risk":            gene_def["mds_risk"],
        "key_avoid":           gene_def["key_avoid"],
        "key_rule":            gene_def["key_rule"],
        "targeted_rx":         gene_def["targeted_rx"],
        "pathognomonic":       gene_def["pathognomonic"],
        "surveillance":        gene_def["surveillance"],
    }


def generate_overview() -> dict:
    cohorts = {
        gd["gene"]: _make_cohort(gd, SEED_BASE + i)
        for i, gd in enumerate(ATLAS_GENES)
    }
    total = 320
    cr_total           = sum(c["cr_n"]              for c in cohorts.values())
    os2_total          = sum(c["os2_n"]             for c in cohorts.values())
    monosomy7_total    = sum(c["monosomy7_n"]        for c in cohorts.values())
    complex_total      = sum(c["complex_karyotype_n"] for c in cohorts.values())
    hsct_total         = sum(c["hsct_n"]             for c in cohorts.values())
    hma_total          = sum(c["hma_n"]              for c in cohorts.values())
    venetoclax_total   = sum(c["venetoclax_n"]       for c in cohorts.values())
    ipss_high_total    = sum(c["ipss_high_vh_n"]     for c in cohorts.values())
    aml_prog_total     = sum(c["aml_prog_n"]         for c in cohorts.values())
    thrombocytopenia_t = sum(c["thrombocytopenia_n"] for c in cohorts.values())

    return {
        "atlas":                  "Hereditary-MDS-Predisposition-Atlas",
        "seed_range":             f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":         total,
        "cr_n":                   cr_total,
        "cr_pct":                 round(cr_total / total * 100, 1),
        "os2_n":                  os2_total,
        "os2_pct":                round(os2_total / total * 100, 1),
        "monosomy7_n":            monosomy7_total,
        "monosomy7_pct":          round(monosomy7_total / total * 100, 1),
        "complex_karyotype_n":    complex_total,
        "complex_karyotype_pct":  round(complex_total / total * 100, 1),
        "hsct_n":                 hsct_total,
        "hsct_pct":               round(hsct_total / total * 100, 1),
        "hma_n":                  hma_total,
        "hma_pct":                round(hma_total / total * 100, 1),
        "venetoclax_n":           venetoclax_total,
        "venetoclax_pct":         round(venetoclax_total / total * 100, 1),
        "ipss_high_vh_n":         ipss_high_total,
        "ipss_high_vh_pct":       round(ipss_high_total / total * 100, 1),
        "aml_prog_n":             aml_prog_total,
        "aml_prog_pct":           round(aml_prog_total / total * 100, 1),
        "thrombocytopenia_n":     thrombocytopenia_t,
        "thrombocytopenia_pct":   round(thrombocytopenia_t / total * 100, 1),
        "gene_counts": {
            gene: {
                "n":                   c["n"],
                "cr_n":                c["cr_n"],
                "os2_n":               c["os2_n"],
                "monosomy7_n":         c["monosomy7_n"],
                "complex_karyotype_n": c["complex_karyotype_n"],
                "hsct_n":              c["hsct_n"],
                "hma_n":               c["hma_n"],
                "venetoclax_n":        c["venetoclax_n"],
                "ipss_high_vh_n":      c["ipss_high_vh_n"],
                "aml_prog_n":          c["aml_prog_n"],
                "mds_risk":            c["mds_risk"],
            }
            for gene, c in cohorts.items()
        },
    }


def generate_breakdown() -> dict:
    return {
        "atlas":     "Hereditary-MDS-Predisposition-Atlas",
        "gene_data": {
            gd["gene"]: _make_cohort(gd, SEED_BASE + i)
            for i, gd in enumerate(ATLAS_GENES)
        },
    }


def generate_definitions() -> dict:
    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"MDS RISK: {gd['mds_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-MDS-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 {gd["gene"]: gd["protein_size"] for gd in ATLAS_GENES},
        "key_clinical_distinctions": KEY_CLINICAL_DISTINCTIONS,
    }
