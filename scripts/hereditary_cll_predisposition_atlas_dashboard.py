#!/usr/bin/env python3
"""Hereditary-CLL-Predisposition-Atlas -- Complete 8-Gene Reference
Chronic Lymphocytic Leukaemia (CLL) germline predisposition.

ATM      (ATM; 3056aa; 11q22.3; AD LOF;
          Ataxia-Telangiectasia / monoallelic ATM (OMIM 208900);
          CLL 2-3x monoallelic; del11q somatic 20-25% worst prognosis;
          RADIOSENSITIVITY-ABSOLUTE; FCR inferior; ibrutinib/venetoclax;
          ceralasertib ATRi synthetic lethality; seed SEED_BASE+0).
TP53     (p53; 393aa; 17p13.1; AD LOF;
          Li-Fraumeni Syndrome (OMIM 151623);
          LFS CLL 2-4x; del17p 8-10% WORST-PROGNOSIS-CLL;
          AVOID RADIATION ABSOLUTELY; venetoclax backbone;
          FCR alkylating agents amplify genomic instability CI;
          seed SEED_BASE+1).
POT1     (POT1; 634aa; 7q31.33; AD LOF;
          Familial CLL / POT1-related shelterin (OMIM 606478);
          MOST-ESTABLISHED familial CLL gene ~8% familial CLL;
          OB-fold ssDNA binding; short telomeres flow-FISH PATHOGNOMONIC;
          ibrutinib/venetoclax preferred; FCR genomic instability inferior;
          seed SEED_BASE+2).
NBN      (NBS1; 754aa; 8q21.3; AD/AR LOF;
          Nijmegen Breakage Syndrome (OMIM 602667);
          657del5 Slavic founder 1in150; CLL/NHL 60x HIGHEST;
          cranial RT absolutely CI; MRN complex DSB repair;
          seed SEED_BASE+3).
BRCA2    (BRCA2; 3418aa; 13q12.3; AD LOF;
          HBOC / Fanconi Anaemia D1 (OMIM 600185);
          del13q somatic 55% CLL (most common CLL deletion);
          HRD 60%; cisplatin preferred; olaparib maintenance;
          FA-D1 biallelic: AVOID alkylating ABSOLUTELY; seed SEED_BASE+4).
CHEK2    (CHK2; 543aa; 22q12.1; AD LOF;
          CHEK2 syndrome (OMIM 604373);
          1100delC Western European 1in200; I157T Eastern European;
          CLL 2-3x moderate; NOT HRD; NO PARPi;
          ATM+CHEK2 additive 8-10x; seed SEED_BASE+5).
CDKN2A   (p16-INK4A; 156aa; 9p21.3; AD LOF;
          FAMM syndrome (OMIM 600160);
          del9p21 somatic 20% CLL; Richter transform 10-15%;
          CDK4/6i INACTIVE in CLL; ibrutinib/venetoclax standard;
          seed SEED_BASE+6).
RB1      (Rb1; 928aa; 13q14.2; AD LOF;
          Hereditary Retinoblastoma (OMIM 180200);
          del13q14 somatic 55% CLL MOST-COMMON sole=GOOD-prognosis;
          CDK4/6i INACTIVE in CLL unlike breast/MCL;
          venetoclax active BCL2 upregulated RB1-loss;
          secondary CLL elevated post-RT; seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3646-3653)
"""
import random

SEED_BASE = 3646

ATLAS_GENES = [
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD LOF -- 3056aa -- "
            "ATM-350kDa-PIK3-Related-Kinase-"
            "FATC-PI3K-FRAP-TRD-"
            "CLL-2-3x-Monoallelic-del11q-20-25pct-WORST-PROGNOSIS-"
            "ATM-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / ~350 kDa / 11q22.3 ATM CLL molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; PI3K-related kinase (PIKK family); master DSB signal transducer; "
            "  N-terminal HEAT repeats (aa 1-2716): substrate docking; "
            "  FAT domain (aa 1960-2566): regulatory; PIKK-specific; "
            "  Kinase domain (aa 2712-2962): catalytic; phosphorylates H2AX S139, BRCA1, CHEK2; "
            "  FATC domain (aa 2963-3056): essential for catalytic activity; "
            "  ATM activates CHEK2 → p53 → apoptosis at DSB; coordinates HR repair; "
            "MONOALLELIC ATM CLL CONTEXT: "
            "  Monoallelic germline ATM LOF: CLL risk 2-3x; del11q somatic in 20-25% sporadic CLL; "
            "  del11q (11q22-23): ATM + BIRC3 co-deletion → worst prognosis CLL second only to del17p/TP53; "
            "  FCR inferior in ATM-del11q CLL: fludarabine-based chemotherapy drives genomic instability; "
            "  Ibrutinib (BTKi) and venetoclax (BCL2i) preferred over FCR in del11q; "
            "  A-T biallelic: IgA deficiency PATHOGNOMONIC; cerebellar ataxia; telangiectasia; "
            "  RADIOSENSITIVITY-ABSOLUTE: even monoallelic ATM carriers show INCREASED RT sensitivity; "
            "CLINICAL RULES FOR ATM-CLL: "
            "  RADIOSENSITIVITY-ABSOLUTE — avoid RT in A-T; monoallelic: reduce RT dose 20-30%; "
            "  del11q testing MANDATORY at CLL diagnosis (FISH or CLL panel); "
            "  FCR inferior — ibrutinib/zanubrutinib or venetoclax preferred in del11q/ATM-mutant; "
            "  Ceralasertib (ATRi) synthetic lethality with ATM loss — Phase II trials; "
            "  IgA deficiency in A-T (biallelic): IVIG if recurrent sinopulmonary infections; "
            "  Cascade family testing: monoallelic ATM germline; "
            "KEY VARIANTS: "
            "  Pathogenic loss-of-function: truncating/frameshifts throughout; "
            "  Kinase domain: missense p.Arg2830Cys, p.Ser2909Leu; "
            "  Monoallelic founders: p.Leu2307Phe (Ashkenazi); "
            "SURVEILLANCE: "
            "  Annual CBC + smear from diagnosis of ATM LOF + lymphocytosis; "
            "  CLL FISH panel at diagnosis: del11q, del17p, del13q, +12; "
            "  IgA levels in A-T families; thyroid annually; skin annually; "
            "TARGETED Rx: "
            "  Ibrutinib/zanubrutinib (BTKi) preferred for del11q-ATM CLL; "
            "  Venetoclax (BCL2i) active regardless of del11q; "
            "  Ceralasertib (ATRi AZD6738) synthetic lethality — clinical trials; "
            "  AVOID fludarabine-based FCR if del11q/ATM-mutant"
        ),
        "syndrome": "Ataxia-Telangiectasia / monoallelic ATM",
        "inheritance": "AD LOF heterozygous",
        "cll_risk": "CLL 2-3x monoallelic; del11q somatic 20-25% worst-prognosis-second-only-del17p",
        "pathognomonic": "IgA-deficiency biallelic A-T PATHOGNOMONIC; cerebellar-ataxia-telangiectasia biallelic",
        "key_rule": "RADIOSENSITIVITY-ABSOLUTE; FCR-INFERIOR-ibrutinib-venetoclax-preferred-del11q",
        "key_avoid": "Avoid RT (absolute biallelic, reduce monoallelic); FCR inferior del11q/ATM-mutant",
        "surveillance": "Annual CBC; CLL FISH panel at diagnosis; IgA in A-T; thyroid/skin annual",
        "targeted_rx": "Ibrutinib/zanubrutinib BTKi preferred; venetoclax BCL2i; ceralasertib ATRi trials",
    },
    {
        "gene": "TP53",
        "protein": (
            "p53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "DBD-OD-TAD-"
            "LFS-CLL-2-4x-del17p-8-10pct-WORST-PROGNOSIS-"
            "TP53-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 CLL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; transcription factor / tumour suppressor; "
            "  N-terminal transactivation domain 1 (TAD1, aa 1-40); TAD2 (aa 40-67): MDM2 binding; "
            "  Proline-rich domain (aa 68-98): apoptosis signalling; "
            "  DNA-binding domain (DBD, aa 102-292): sequence-specific binding; "
            "  Nuclear localisation signal (aa 316-325); "
            "  Tetramerisation domain (OD, aa 323-356): functional tetramer; "
            "  C-terminal regulatory domain (aa 364-393): ubiquitination target; "
            "  TP53 LOF → failed apoptosis, G1 arrest bypass, genomic instability; "
            "TP53 LFS-CLL CONTEXT: "
            "  LFS germline TP53 LOF: CLL risk 2-4x; del17p in 8-10% sporadic CLL; "
            "  del17p CLL: WORST PROGNOSIS — median OS 2-3y without targeted Rx; "
            "  FCR: ABSOLUTELY CONTRAINDICATED with del17p/TP53-mutant CLL — alkylating drives clonal evolution; "
            "  Venetoclax (BCL2i) backbone — active despite p53 loss (BCL2-dependent apoptosis bypass); "
            "  Ibrutinib (BTKi) active — indirect pathway independent of p53; "
            "  APR-246 (eprenetapopt): mutant p53 refolding — Phase III trials; "
            "CLINICAL RULES FOR TP53-CLL: "
            "  AVOID RADIATION ABSOLUTELY in LFS; WB-MRI Toronto protocol annually; "
            "  del17p/TP53-mutant: FCR absolutely CI — venetoclax backbone mandatory; "
            "  del17p testing MANDATORY at CLL diagnosis (FISH); TP53 sequencing if del17p absent; "
            "  Venetoclax + obinutuzumab (CLL14 regimen) standard for del17p/TP53-mutant; "
            "  Ibrutinib/acalabrutinib active in del17p — TP53-independent mechanism; "
            "  WB-MRI annual: sarcoma, brain, breast surveillance (LFS cancer spectrum); "
            "KEY VARIANTS: "
            "  Hotspot DBD: p.Arg175His, p.Gly245Ser, p.Arg248Trp/Gln, p.Arg249Ser, p.Arg273His/Cys; "
            "  Dominant-negative effect — mut protein inhibits WT tetramer; "
            "SURVEILLANCE: "
            "  WB-MRI annually (Toronto protocol); TP53 + del17p FISH at CLL diagnosis; "
            "  Brain MRI annually; colonoscopy 25yr or 5yr before earliest family cancer; "
            "TARGETED Rx: "
            "  Venetoclax backbone (CLL14: Venet+Obi; MURANO: Venet+Ritux); "
            "  Ibrutinib/acalabrutinib/zanubrutinib (BTKi) all active; "
            "  Eprenetapopt (APR-246) mutant p53 refolding — trials; "
            "  AVOID FCR, chlorambucil, bendamustine-based chemotherapy in del17p"
        ),
        "syndrome": "Li-Fraumeni Syndrome",
        "inheritance": "AD LOF heterozygous",
        "cll_risk": "LFS CLL 2-4x; del17p somatic 8-10% WORST-PROGNOSIS-CLL",
        "pathognomonic": "LFS-diverse-cancer-spectrum; del17p CLL WORST-PROGNOSIS; FCR-CI-ABSOLUTELY",
        "key_rule": "AVOID-RADIATION-ABSOLUTELY; FCR-CI-del17p; venetoclax-backbone-MANDATORY",
        "key_avoid": "AVOID RT absolutely (LFS); FCR/alkylating CI in del17p/TP53-mutant CLL",
        "surveillance": "WB-MRI Toronto annually; del17p FISH + TP53 seq at CLL diagnosis; brain MRI annual",
        "targeted_rx": "Venetoclax backbone (CLL14/MURANO); ibrutinib/acalabrutinib BTKi; eprenetapopt trials",
    },
    {
        "gene": "POT1",
        "protein": (
            "POT1 -- 7q31.33 AD LOF -- 634aa -- "
            "POT1-71kDa-Shelterin-Telomere-OB-fold-"
            "OB1-OB2-TPPR-"
            "Familial-CLL-8pct-Short-Telomeres-PATHOGNOMONIC-"
            "POT1-OMIM-606478"
        ),
        "locus": "7q31.33",
        "protein_size": (
            "634 aa / ~71 kDa / 7q31.33 POT1 CLL molecular context: "
            "STRUCTURE: "
            "  634 aa / ~71 kDa; OB (oligonucleotide/oligosaccharide-binding) fold protein; "
            "  OB-fold 1 (aa 1-89): primary ssDNA binding; telomeric TTAGGG repeat recognition; "
            "  OB-fold 2 (aa 90-180): secondary ssDNA binding; "
            "  TPP1 binding domain (TPPR, aa 300-634): TPP1 interaction; shelterin complex assembly; "
            "  POT1 caps 3-prime telomere overhang → inhibits ATM/ATR activation; "
            "  POT1 LOF → unprotected telomere → DDR activation → chromosomal instability; "
            "  Short telomere phenotype: flow-FISH telomere length SHORT PATHOGNOMONIC; "
            "POT1 FAMILIAL CLL CONTEXT: "
            "  POT1 mutations: ~8% familial CLL (highest of any single gene); "
            "  OB-fold mutations → abnormal telomere capping → CLL oncogenesis; "
            "  POT1-CLL: early onset; short telomeres; genomic instability; "
            "  Short telomeres (flow-FISH): PATHOGNOMONIC POT1-shelterin defect; "
            "  FCR inferior: genomic instability → subclonal evolution under chemotherapy; "
            "  Ibrutinib/venetoclax preferred: independent of telomere dysfunction; "
            "CLINICAL RULES FOR POT1-CLL: "
            "  Flow-FISH telomere length: short telomeres (<10th percentile) — PATHOGNOMONIC; "
            "  CASCADE family testing: POT1 germline variants in familial CLL clusters; "
            "  Ibrutinib/venetoclax preferred — FCR inferior genomic instability; "
            "  Zanubrutinib (BTKi next-gen) preferred over ibrutinib (cardiac toxicity lower); "
            "  Somatic additional lesions: ATM loss, del17p may be acquired — monitor FISH evolution; "
            "  No established organ-specific surveillance beyond CLL — unlike BRCA/TP53; "
            "KEY VARIANTS: "
            "  OB1 mutations most common: p.Tyr89Cys (OB1 ssDNA contact), p.Phe62Ile; "
            "  OB2: p.Asp224Asn; "
            "  TPPR: p.Glu450Asp, p.Cys352Arg; "
            "  All missense — protein-destabilising loss-of-function mechanism; "
            "SURVEILLANCE: "
            "  Annual CBC + smear; lymphocyte doubling time from diagnosis; "
            "  Flow-FISH telomere length baseline; family cascade germline testing; "
            "  No additional organ surveillance beyond CLL spectrum established; "
            "TARGETED Rx: "
            "  Ibrutinib/zanubrutinib (BTKi) preferred — FCR inferior; "
            "  Venetoclax (BCL2i) active; "
            "  Combination BTKi+BCL2i (GLOW/CAPTIVATE) — emerging standard; "
            "  Pirtobrutinib (LOXO-305) non-covalent BTKi for C481S resistance"
        ),
        "syndrome": "Familial CLL — POT1-shelterin",
        "inheritance": "AD LOF heterozygous",
        "cll_risk": "Familial-CLL ~8% of familial CLL cohorts; MOST-ESTABLISHED familial CLL gene",
        "pathognomonic": "Short-telomeres-flow-FISH-PATHOGNOMONIC; familial-CLL-clustering",
        "key_rule": "SHORT-TELOMERES-PATHOGNOMONIC; FCR-INFERIOR; ibrutinib-venetoclax-preferred",
        "key_avoid": "FCR inferior (genomic instability); no established organ surveillance beyond CLL",
        "surveillance": "Annual CBC; flow-FISH telomere length baseline; cascade family germline testing",
        "targeted_rx": "Ibrutinib/zanubrutinib BTKi preferred; venetoclax BCL2i; BTKi+BCL2i combination",
    },
    {
        "gene": "NBN",
        "protein": (
            "NBS1 -- 8q21.3 AD/AR LOF -- 754aa -- "
            "NBS1-95kDa-MRN-Complex-"
            "FHA-BRCT-MRN-ATM-"
            "NBS-657del5-Slavic-1in150-CLL-NHL-60x-HIGHEST-"
            "NBN-OMIM-602667"
        ),
        "locus": "8q21.3",
        "protein_size": (
            "754 aa / ~95 kDa / 8q21.3 NBN CLL molecular context: "
            "STRUCTURE: "
            "  754 aa / ~95 kDa; adaptor subunit of MRN (MRE11-RAD50-NBS1) complex; "
            "  FHA domain (aa 1-110): pSer/pThr recognition; ATM recruitment; "
            "  BRCT1 domain (aa 111-196): phosphopeptide binding; "
            "  BRCT2 domain (aa 197-321): tandem with BRCT1; MDC1 binding; "
            "  Linker region (aa 322-665): flexible; NBS1-MRE11 interaction; "
            "  MRE11 binding domain (aa 665-693): MRN complex formation; "
            "  ATM binding domain (aa 734-754): ATM recruitment and activation at DSB; "
            "  NBS1 LOF → impaired ATM activation → defective DSB repair; "
            "NBN NBS-CLL CONTEXT: "
            "  NBS (Nijmegen Breakage Syndrome): microcephaly + immunodeficiency + BIRD-LIKE FACE PATHOGNOMONIC; "
            "  657del5 Slavic founder mutation (c.657_661del): most common NBS allele; "
            "  Population frequency 657del5: 1 in 150 Central/Eastern European (Czech, Slovak, Polish); "
            "  CLL/NHL in NBS patients: 60x HIGHEST RISK of any hereditary syndrome; "
            "  Heterozygous NBN (AD): moderate NHL/CLL 3-5x; "
            "  Cranial RT: ABSOLUTELY CI in NBS biallelic and should be minimised in heterozygotes; "
            "CLINICAL RULES FOR NBN-CLL: "
            "  Cranial RT ABSOLUTELY CI — extreme radiosensitivity, lethal doses with standard RT; "
            "  657del5 Slavic founder: population carrier testing in Central/Eastern Europeans with NHL/CLL; "
            "  Immune reconstitution: IVIG if IgG <3 g/L; "
            "  Heterozygous NBN: CLL 3-5x risk; minimise RT; ibrutinib preferred; "
            "  Biallelic NBS: HSCT for aplasia/severe immunodeficiency; "
            "KEY VARIANTS: "
            "  c.657_661delACAAA (657del5): most common ~90% of NBS alleles (Slavic founder); "
            "  c.698_701delAACA: second most common; "
            "  Missense: p.Ile171Val (hypomorphic — less severe); "
            "SURVEILLANCE: "
            "  CBC + lymphocyte subsets annually; IgG, IgA, IgM annually; "
            "  Brain MRI: biallelic NBS microcephaly baseline; "
            "  Annual skin exam; thyroid annually; "
            "  Cascade testing in Slavic families; "
            "TARGETED Rx: "
            "  Ibrutinib/zanubrutinib preferred (BTKi); "
            "  Venetoclax (BCL2i) active; "
            "  IVIG if IgG <3 g/L; "
            "  AVOID radiation-based conditioning in NBS biallelic HSCT — fludarabine RIC"
        ),
        "syndrome": "Nijmegen Breakage Syndrome",
        "inheritance": "AD/AR LOF; AR biallelic = NBS; AD heterozygous = moderate risk",
        "cll_risk": "NBS biallelic: CLL/NHL 60x HIGHEST; NBN heterozygous: 3-5x moderate",
        "pathognomonic": "Bird-like-face-microcephaly PATHOGNOMONIC NBS; 657del5-Slavic-founder 1in150",
        "key_rule": "CRANIAL-RT-ABSOLUTELY-CI; 657del5-Slavic-population-testing-MANDATORY",
        "key_avoid": "RT absolutely CI biallelic; minimise RT heterozygous; avoid alkylating in NBS HSCT",
        "surveillance": "Annual CBC + Ig subsets; brain MRI biallelic; cascade testing Slavic families",
        "targeted_rx": "Ibrutinib BTKi preferred; venetoclax BCL2i; IVIG IgG<3; fludarabine-RIC HSCT not TBI",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 AD LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Mediator-FANCD1-"
            "BRC-Repeats-OB-folds-"
            "HBOC-del13q-somatic-55pct-CLL-HRD-Cisplatin-"
            "BRCA2-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / ~384 kDa / 13q12.3 BRCA2 CLL molecular context: "
            "STRUCTURE: "
            "  3418 aa / ~384 kDa; HR DNA repair mediator; FANCD1; "
            "  N-terminal PALB2 binding (aa 10-40); "
            "  BRC repeats ×8 (aa 1002-2085): RAD51 filament assembly; "
            "  OB-fold domains (aa 2396-3102): ssDNA binding; "
            "  DBD (aa 2402-3190): RAD51 binding; "
            "  C-terminal: nuclear localisation; BRCA1 interaction; "
            "  BRCA2 LOF → impaired RAD51 loading → HR deficient; NHEJ-prone; genomic instability; "
            "BRCA2-CLL CONTEXT: "
            "  del13q14 SOMATIC: MOST COMMON cytogenetic alteration in CLL (55% of cases); "
            "  del13q14 sole abnormality: GOOD PROGNOSIS CLL (favourable compared to normal karyotype in some series); "
            "  del13q14 + other: poor prognosis; "
            "  Germline BRCA2 LOF: HBOC (breast/ovarian/pancreatic); CLL/NHL emerging 1.5-2x risk; "
            "  HRD (Homologous Recombination Deficiency) 60%: cisplatin preferred over carboplatin; "
            "  Olaparib (PARPi) maintenance active in BRCA2-deficient lymphoma; "
            "  FA-D1 biallelic: childhood medulloblastoma/Wilms + ALKYLATING ABSOLUTELY CI; "
            "CLINICAL RULES FOR BRCA2-CLL: "
            "  del13q14 sole: GOOD prognosis — watch and wait standard; "
            "  del13q14 + del17p/TP53 or +12: prognosis driven by the high-risk lesion; "
            "  Germline BRCA2: HRD 60% — cisplatin preferred over carboplatin in R/R CLL; "
            "  FA-D1 biallelic: AVOID alkylating ABSOLUTELY; MMC fragility test mandatory; "
            "  Olaparib: emerging in BRCA2-deficient CLL/lymphoma; "
            "  BRCA2 hereditary surveillance: annual mammography/MRI breast 25yr; RRSO 40yr; "
            "KEY VARIANTS: "
            "  Ashkenazi founder: p.Lys3326*, c.5946delT (6174delT); "
            "  Icelandic: c.771_775del5; "
            "  Pathogenic: truncating throughout coding region; "
            "SURVEILLANCE: "
            "  Annual breast MRI 25yr + mammogram 30yr; RRSO 40-45yr; "
            "  Pancreatic MRI/EUS 50yr or 10yr before earliest family pancreatic Ca; "
            "  del13q14 FISH at CLL diagnosis; "
            "TARGETED Rx: "
            "  Venetoclax BCL2i; ibrutinib BTKi; "
            "  Cisplatin-based Rx preferred in BRCA2-HRD R/R CLL; "
            "  Olaparib PARPi emerging; "
            "  FA-D1: AVOID alkylating; fludarabine non-alkylating preferred"
        ),
        "syndrome": "HBOC / Fanconi Anaemia D1",
        "inheritance": "AD LOF; AR biallelic = FA-D1",
        "cll_risk": "Germline BRCA2: CLL emerging 1.5-2x; del13q somatic 55% CLL MOST-COMMON",
        "pathognomonic": "del13q14-MOST-COMMON-CLL-cytogenetic; FA-D1-childhood-medulloblastoma-Wilms-PATHOGNOMONIC",
        "key_rule": "del13q-SOLE-GOOD-PROGNOSIS; FA-D1-AVOID-ALKYLATING-ABSOLUTELY; HRD-cisplatin-preferred",
        "key_avoid": "Alkylating agents in FA-D1 absolutely CI; MMC fragility test mandatory FA-D1",
        "surveillance": "Annual breast MRI 25yr; RRSO 40yr; pancreatic EUS 50yr; del13q FISH at CLL diagnosis",
        "targeted_rx": "Venetoclax BCL2i; ibrutinib BTKi; cisplatin preferred HRD; olaparib emerging",
    },
    {
        "gene": "CHEK2",
        "protein": (
            "CHK2 -- 22q12.1 AD LOF -- 543aa -- "
            "CHEK2-60kDa-ATM-Effector-Kinase-"
            "SQ-FHA-KD-"
            "1100delC-Western-European-1in200-I157T-Eastern-European-CLL-2-3x-"
            "CHEK2-OMIM-604373"
        ),
        "locus": "22q12.1",
        "protein_size": (
            "543 aa / ~60 kDa / 22q12.1 CHEK2 CLL molecular context: "
            "STRUCTURE: "
            "  543 aa / ~60 kDa; serine/threonine checkpoint kinase; ATM effector; "
            "  SCD/SQ cluster (aa 1-69): ATM phosphorylation targets T68; "
            "  FHA domain (aa 113-175): phosphopeptide recognition; dimerisation; "
            "  Kinase domain (KD, aa 210-486): catalytic; phosphorylates p53 S20, BRCA1 S988, CDC25; "
            "  Activation loop (aa 383-409): T383/T387 autophosphorylation; "
            "  CHEK2 LOF → impaired G1 checkpoint → p53 stabilisation failure → proliferation bypass; "
            "CHEK2-CLL CONTEXT: "
            "  1100delC Western European founder (c.1100delC): 1 in 200 population frequency; "
            "  I157T Eastern European founder (p.Ile157Thr): partial LOF; "
            "  Moderate CLL risk: 2-3x; heterozygous CHEK2 LOF; "
            "  NOT HRD: CHEK2 does NOT confer homologous recombination deficiency; "
            "  NO PARPi: olaparib/niraparib NOT indicated in CHEK2-only germline; "
            "  ATM + CHEK2 compound heterozygous: additive risk 8-10x CLL/NHL; "
            "CLINICAL RULES FOR CHEK2-CLL: "
            "  NOT HRD — cisplatin/PARPi NOT indicated (critical: BRCA-like language inapplicable); "
            "  ATM + CHEK2 co-carriers: 8-10x risk — intensive surveillance; "
            "  Standard CLL treatment: ibrutinib/venetoclax per del17p/del11q status; "
            "  Female carriers: breast cancer 2-3x — annual mammography 40yr; "
            "  Cascade testing recommended in families with multiple CLL/NHL + breast Ca; "
            "KEY VARIANTS: "
            "  c.1100delC (1100delC): most common pathogenic Western European; null allele; "
            "  p.Ile157Thr (I157T): Eastern European partial LOF; moderate risk; "
            "  p.Ser428Phe: FHA domain disruption; "
            "  c.444+1G>A: splice donor loss; "
            "SURVEILLANCE: "
            "  Annual CBC; mammography from 40yr (female); "
            "  Cascade family testing; "
            "  No additional CLL-specific surveillance beyond standard oncology; "
            "TARGETED Rx: "
            "  Standard CLL: ibrutinib/venetoclax per FISH-risk stratification; "
            "  AVOID PARPi: NOT HRD in CHEK2-only; "
            "  ATM+CHEK2 co-carrier: ibrutinib preferred over FCR"
        ),
        "syndrome": "CHEK2 syndrome — moderate multi-cancer",
        "inheritance": "AD LOF heterozygous",
        "cll_risk": "CLL 2-3x moderate; 1100delC 1in200 Western European; I157T Eastern European",
        "pathognomonic": "ATM+CHEK2-additive-8-10x CLL/NHL; 1100delC-1in200-MOST-COMMON-founder",
        "key_rule": "NOT-HRD-NO-PARPi; ATM+CHEK2-additive-8-10x; standard-CLL-Rx-per-FISH",
        "key_avoid": "PARPi NOT indicated; don't apply BRCA-like HRD language to CHEK2-only",
        "surveillance": "Annual CBC; mammography 40yr female; cascade family testing",
        "targeted_rx": "Standard CLL ibrutinib/venetoclax; ATM+CHEK2 co-carrier ibrutinib preferred; NO PARPi",
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "p16-INK4A -- 9p21.3 AD LOF -- 156aa -- "
            "p16-INK4A-17kDa-CDK4-6-Inhibitor-Dual-p14ARF-"
            "ANK-repeats-"
            "FAMM-del9p21-somatic-20pct-CLL-CDK4-6i-INACTIVE-Richter-10-15pct-"
            "CDKN2A-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa (p16-INK4A) / ~17 kDa / 9p21.3 CDKN2A CLL molecular context: "
            "STRUCTURE: "
            "  156 aa / ~17 kDa (p16-INK4A isoform); alternatively 132 aa / ~16 kDa (p14-ARF isoform); "
            "  p16-INK4A: 4 ankyrin (ANK) repeats (aa 1-148): CDK4/CDK6 binding and inhibition; "
            "  p14-ARF: alternative reading frame same locus: MDM2 binding → p53 stabilisation; "
            "  p16-INK4A LOF → CDK4/6 hyperactive → Rb phosphorylation → G1 bypass → proliferation; "
            "  Dual gene deletion: del9p21 destroys both p16-INK4A and p14-ARF simultaneously; "
            "CDKN2A-CLL CONTEXT: "
            "  del9p21 SOMATIC: ~20% of CLL (acquired during disease progression); "
            "  del9p21 in CLL: marker of disease progression and transformation risk; "
            "  Richter syndrome (diffuse large B-cell transformation): 10-15% CLL — del9p21 enriched; "
            "  CDK4/6i (palbociclib, ribociclib, abemaciclib): INACTIVE in CLL; "
            "  CDK4/6i active in: breast cancer, MCL (ibrutinib-resistant MCL) — NOT CLL; "
            "  FAMM germline: melanoma 28-67% lifetime, pancreatic 17-39% lifetime; "
            "CLINICAL RULES FOR CDKN2A-CLL: "
            "  CDK4/6i INACTIVE in CLL — do NOT use palbociclib/ribociclib for CLL component; "
            "  Richter transform surveillance: LDH + PET-CT if rapid lymph node enlargement; "
            "  del9p21 FISH or CLL panel at diagnosis — transformation risk marker; "
            "  FAMM: annual dermatology + dermoscopy from age 18; "
            "  Pancreatic EUS/MRI 40yr or 10yr before earliest family pancreatic Ca; "
            "  Melanoma: BRAF/MEK not relevant to CLL component; "
            "KEY VARIANTS: "
            "  Germline deletions: 9p21 large deletion (FAMM); "
            "  Missense p16-INK4A: p.Ala148Thr, p.Arg24Pro (ANK repeat disruption); "
            "  p.Gly101Trp (hotspot — disrupts CDK4 contact); "
            "  Promoter: c.-34G>T (methylation-prone); "
            "SURVEILLANCE: "
            "  Annual dermatology FAMM from age 18; dermoscopy; "
            "  Pancreatic MRI/EUS from 40yr; "
            "  PET-CT if Richter transform suspected (rapid LN growth, B symptoms, LDH rise); "
            "  del9p21 FISH at CLL diagnosis; "
            "TARGETED Rx: "
            "  Standard CLL: ibrutinib/venetoclax per risk stratification; "
            "  CDK4/6i NOT active in CLL; "
            "  Richter transform: R-CHOP or pola-R-CHP; "
            "  FAMM melanoma: BRAF/MEK inhibitor if BRAF V600E — separate from CLL Rx"
        ),
        "syndrome": "FAMM syndrome / dual p16-ARF locus",
        "inheritance": "AD LOF heterozygous",
        "cll_risk": "Germline CDKN2A: CLL association moderate; del9p21 somatic 20% CLL progression marker",
        "pathognomonic": "FAMM-melanoma-28-67pct; Richter-transform-del9p21-10-15pct; CDK4-6i-INACTIVE-CLL",
        "key_rule": "CDK4-6i-INACTIVE-CLL; Richter-transform-del9p21-surveillance; FAMM-dermatology-18yr",
        "key_avoid": "CDK4/6i NOT active in CLL (unlike MCL/BCa); don't confuse FAMM melanoma Rx with CLL Rx",
        "surveillance": "Annual dermatology 18yr FAMM; pancreatic EUS 40yr; PET-CT Richter suspect; del9p21 FISH",
        "targeted_rx": "Standard CLL ibrutinib/venetoclax; Richter: R-CHOP/pola-R-CHP; CDK4/6i NOT indicated",
    },
    {
        "gene": "RB1",
        "protein": (
            "Rb1 -- 13q14.2 AD LOF -- 928aa -- "
            "Rb1-110kDa-Cell-Cycle-Regulator-"
            "Pocket-A-B-spacer-"
            "del13q14-somatic-55pct-CLL-MOST-COMMON-GOOD-PROGNOSIS-sole-"
            "CDK4-6i-INACTIVE-BCL2-upregulated-Venetoclax-"
            "RB1-OMIM-180200"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / ~110 kDa / 13q14.2 RB1 CLL molecular context: "
            "STRUCTURE: "
            "  928 aa / ~110 kDa; tumour suppressor; master cell-cycle regulator; "
            "  N-terminal domain (aa 1-378): structural; "
            "  Pocket domain A (aa 379-572): E2F binding; "
            "  Spacer domain (aa 573-646): caspase-3 cleavage site; "
            "  Pocket domain B (aa 647-772): E2F binding; HDAC/LxCxE motif partners; "
            "  C-terminal domain (aa 773-928): CDK phosphorylation sites; nuclear localisation; "
            "  Hypophosphorylated Rb (active): binds E2F → G1 arrest; "
            "  CDK4/6 phosphorylates Rb → E2F release → S-phase entry; "
            "  Rb LOF → constitutive E2F activation → uncontrolled proliferation; "
            "RB1-CLL CONTEXT: "
            "  del13q14 SOMATIC: MOST COMMON cytogenetic alteration in CLL (55% of cases); "
            "  del13q14 SOLE abnormality: GOOD PROGNOSIS — best CLL cytogenetic group; "
            "  del13q14 + high-risk (del17p, del11q, +12): prognosis driven by high-risk lesion; "
            "  CDK4/6i (palbociclib, ribociclib): INACTIVE in CLL — RB1 already deleted; "
            "  CDK4/6i active in: breast cancer, MCL — NOT in CLL (Rb already lost — no target); "
            "  BCL2 upregulation with RB1 loss → venetoclax ACTIVE and often very effective; "
            "  Hereditary Rb (bilateral PATHOGNOMONIC): secondary CLL risk elevated post-RT; "
            "CLINICAL RULES FOR RB1-CLL: "
            "  del13q14 SOLE = GOOD prognosis — watch and wait standard; do NOT over-treat; "
            "  CDK4/6i ABSOLUTELY INACTIVE in CLL — RB1 loss means no drug target; "
            "  Venetoclax ACTIVE — BCL2 upregulated with RB1 loss; "
            "  Hereditary Rb bilateral PATHOGNOMONIC — germline RB1 at CLL diagnosis if bilateral Rb history; "
            "  Secondary CLL post-RT: elevated in hereditary Rb survivors — monitor CBC; "
            "  AVOID CDK4/6i — common error in breast oncologists treating secondary CLL; "
            "KEY VARIANTS: "
            "  Hereditary: nonsense/frameshift throughout 27 exons; "
            "  p.Arg552*: most common nonsense; "
            "  p.Trp563*: exon 17; "
            "  Large deletion/rearrangement: 10-15% of hereditary Rb; "
            "SURVEILLANCE: "
            "  Hereditary Rb: ophthalmology surveillance per RB1-specific protocol; "
            "  Annual CBC if secondary CLL risk (post-RT history); "
            "  del13q14 FISH at CLL diagnosis — GOOD prognosis if sole; "
            "  BONE scan: osteosarcoma surveillance in hereditary Rb survivors; "
            "TARGETED Rx: "
            "  Watch and wait: del13q14 sole — observation standard; "
            "  Venetoclax BCL2i ACTIVE; ibrutinib BTKi active; "
            "  CDK4/6i NOT active (RB1 loss = no drug target); "
            "  Richter rare in del13q14 sole — venetoclax if treatment required"
        ),
        "syndrome": "Hereditary Retinoblastoma",
        "inheritance": "AD LOF heterozygous",
        "cll_risk": "del13q somatic 55% CLL MOST-COMMON-sole=GOOD-PROGNOSIS; secondary CLL post-RT elevated",
        "pathognomonic": "Bilateral-retinoblastoma-PATHOGNOMONIC-hereditary-RB1; del13q-SOLE-GOOD-PROGNOSIS-CLL",
        "key_rule": "CDK4-6i-INACTIVE-CLL; del13q-SOLE-WATCH-WAIT; venetoclax-ACTIVE-BCL2-upregulated",
        "key_avoid": "CDK4/6i INACTIVE in CLL (RB1 already deleted); don't over-treat del13q14 sole",
        "surveillance": "Ophthalmology hereditary Rb protocol; annual CBC post-RT survivors; del13q FISH at diagnosis",
        "targeted_rx": "Venetoclax BCL2i ACTIVE; ibrutinib BTKi active; CDK4/6i NOT active; watch-wait del13q sole",
    },
]

KEY_CLINICAL_DISTINCTIONS = [
    "ATM: RADIOSENSITIVITY-ABSOLUTE; monoallelic CLL 2-3x; del11q 20-25% somatic worst-prognosis-second; FCR inferior; ibrutinib/venetoclax; ceralasertib ATRi",
    "TP53: AVOID-RADIATION-ABSOLUTELY; del17p 8-10% WORST-PROGNOSIS-CLL; FCR alkylating-CI-absolutely; venetoclax backbone mandatory; WB-MRI Toronto",
    "POT1: MOST-ESTABLISHED familial CLL gene ~8%; short telomeres flow-FISH PATHOGNOMONIC; FCR inferior; ibrutinib/venetoclax; BTKi+BCL2i combination",
    "NBN: 657del5-Slavic-founder-1in150; CLL/NHL-60x-HIGHEST; CRANIAL-RT-ABSOLUTELY-CI; fludarabine-RIC-not-TBI; IVIG-IgG<3",
    "BRCA2: del13q somatic 55% MOST-COMMON-CLL-cytogenetic; del13q-SOLE-GOOD-PROGNOSIS; HRD-cisplatin-preferred; FA-D1-ALKYLATING-CI-ABSOLUTELY",
    "CHEK2: NOT-HRD-NO-PARPi; 1100delC-1in200-Western-European; ATM+CHEK2-additive-8-10x; standard-CLL-Rx-per-FISH",
    "CDKN2A: CDK4-6i-INACTIVE-CLL; del9p21-somatic-20pct-progression; Richter-transform-10-15pct; FAMM-dermatology-18yr; pancreatic-EUS-40yr",
    "RB1: del13q-SOLE-GOOD-PROGNOSIS; CDK4-6i-INACTIVE-CLL-RB1-already-deleted; venetoclax-ACTIVE-BCL2-upregulated; bilateral-Rb-PATHOGNOMONIC",
]


def _make_cohort(gene_idx: int, seed: int) -> dict:
    """Generate a 40-patient CLL cohort for one predisposition gene."""
    rng = random.Random(seed)
    g = ATLAS_GENES[gene_idx]

    age_params = {
        0: (62, 8),   # ATM — monoallelic CLL; adult onset
        1: (58, 10),  # TP53 — LFS earlier onset
        2: (55, 9),   # POT1 — familial CLL earlier onset
        3: (52, 12),  # NBN — NBS variable onset
        4: (64, 8),   # BRCA2 — HBOC-associated CLL
        5: (65, 7),   # CHEK2 — moderate risk later onset
        6: (63, 9),   # CDKN2A — adult CLL
        7: (66, 8),   # RB1 — later onset
    }
    mu, sd = age_params[gene_idx]
    ages = [max(18, min(85, int(rng.gauss(mu, sd)))) for _ in range(40)]

    # CLL stage (Binet A/B/C)
    stage_dist = {
        0: [0.40, 0.35, 0.25],  # ATM: del11q — higher stage
        1: [0.25, 0.35, 0.40],  # TP53: del17p — highest stage
        2: [0.45, 0.35, 0.20],  # POT1: familial — lower stage
        3: [0.35, 0.35, 0.30],  # NBN: variable
        4: [0.45, 0.35, 0.20],  # BRCA2: del13q sole good
        5: [0.50, 0.30, 0.20],  # CHEK2: moderate risk
        6: [0.40, 0.35, 0.25],  # CDKN2A: del9p21 progression
        7: [0.55, 0.30, 0.15],  # RB1: del13q sole — good prognosis
    }
    bins = stage_dist[gene_idx]
    binet_a = int(round(40 * bins[0]))
    binet_b = int(round(40 * bins[1]))
    binet_c = 40 - binet_a - binet_b

    # IGHV mutational status
    ighv_unmut = {
        0: 18, 1: 22, 2: 14, 3: 16, 4: 12, 5: 14, 6: 18, 7: 10
    }[gene_idx]
    ighv_mut = 40 - ighv_unmut

    # Cytogenetic by FISH
    cyto_dist = {
        0: {"del11q": 18, "del13q": 8, "del17p": 2, "trisomy12": 4, "normal": 8},
        1: {"del17p": 22, "del11q": 5, "del13q": 5, "trisomy12": 3, "normal": 5},
        2: {"del13q": 12, "normal": 14, "del11q": 6, "del17p": 2, "trisomy12": 6},
        3: {"del13q": 10, "del11q": 8, "del17p": 6, "trisomy12": 6, "normal": 10},
        4: {"del13q": 24, "normal": 6, "del11q": 4, "del17p": 2, "trisomy12": 4},
        5: {"del13q": 14, "normal": 14, "del11q": 6, "del17p": 2, "trisomy12": 4},
        6: {"del9p21": 16, "del13q": 10, "del11q": 6, "del17p": 4, "normal": 4},
        7: {"del13q": 28, "normal": 6, "del11q": 2, "del17p": 2, "trisomy12": 2},
    }
    cyto = cyto_dist[gene_idx]

    # Treatment first-line
    tx_btki = {0: 22, 1: 20, 2: 18, 3: 16, 4: 14, 5: 12, 6: 14, 7: 10}[gene_idx]
    tx_venetoclax = {0: 12, 1: 16, 2: 10, 3: 10, 4: 14, 5: 10, 6: 12, 7: 8}[gene_idx]
    tx_watch_wait = 40 - tx_btki - tx_venetoclax - {0: 2, 1: 0, 2: 6, 3: 8, 4: 8, 5: 14, 6: 8, 7: 18}[gene_idx]
    tx_watch_wait = max(0, tx_watch_wait)

    # Outcomes
    cr_rate = {0: 0.48, 1: 0.32, 2: 0.55, 3: 0.45, 4: 0.58, 5: 0.55, 6: 0.46, 7: 0.62}[gene_idx]
    os_2yr = {0: 0.72, 1: 0.55, 2: 0.80, 3: 0.65, 4: 0.82, 5: 0.80, 6: 0.70, 7: 0.85}[gene_idx]

    richter_count = {0: 4, 1: 8, 2: 2, 3: 5, 4: 2, 5: 2, 6: 6, 7: 1}[gene_idx]
    radiosensitivity_flag = {0: 40, 1: 40, 2: 0, 3: 40, 4: 0, 5: 0, 6: 0, 7: 0}[gene_idx]

    return {
        "gene": g["gene"],
        "locus": g["locus"],
        "inheritance": g["inheritance"],
        "n": 40,
        "seed": seed,
        "ages": ages,
        "mean_age": round(sum(ages) / 40, 1),
        "binet_a": binet_a,
        "binet_b": binet_b,
        "binet_c": binet_c,
        "ighv_unmutated": ighv_unmut,
        "ighv_mutated": ighv_mut,
        "cytogenetics": cyto,
        "tx_btki": tx_btki,
        "tx_venetoclax": tx_venetoclax,
        "tx_watch_wait": tx_watch_wait,
        "cr_rate": cr_rate,
        "os_2yr": os_2yr,
        "richter_count": richter_count,
        "radiosensitivity_flag": radiosensitivity_flag,
        "cll_risk": g["cll_risk"],
        "pathognomonic": g["pathognomonic"],
        "key_rule": g["key_rule"],
        "key_avoid": g["key_avoid"],
        "targeted_rx": g["targeted_rx"],
    }


def generate_overview() -> dict:
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(8)]

    total_patients = sum(c["n"] for c in cohorts)
    mean_age_overall = round(sum(c["mean_age"] * c["n"] for c in cohorts) / total_patients, 1)

    total_binet_a = sum(c["binet_a"] for c in cohorts)
    total_binet_b = sum(c["binet_b"] for c in cohorts)
    total_binet_c = sum(c["binet_c"] for c in cohorts)
    total_btki = sum(c["tx_btki"] for c in cohorts)
    total_venetoclax = sum(c["tx_venetoclax"] for c in cohorts)
    total_watch_wait = sum(c["tx_watch_wait"] for c in cohorts)
    total_richter = sum(c["richter_count"] for c in cohorts)

    total_del13q = sum(c["cytogenetics"].get("del13q", 0) for c in cohorts)
    total_del17p = sum(c["cytogenetics"].get("del17p", 0) for c in cohorts)
    total_del11q = sum(c["cytogenetics"].get("del11q", 0) for c in cohorts)
    total_trisomy12 = sum(c["cytogenetics"].get("trisomy12", 0) for c in cohorts)
    total_del9p21 = sum(c["cytogenetics"].get("del9p21", 0) for c in cohorts)

    avg_cr = round(sum(c["cr_rate"] for c in cohorts) / 8 * 100, 1)
    avg_os2yr = round(sum(c["os_2yr"] for c in cohorts) / 8 * 100, 1)

    per_gene = [
        {
            "gene": c["gene"],
            "locus": c["locus"],
            "n": c["n"],
            "mean_age": c["mean_age"],
            "binet_a_pct": round(c["binet_a"] / 40 * 100),
            "binet_c_pct": round(c["binet_c"] / 40 * 100),
            "ighv_unmut_pct": round(c["ighv_unmutated"] / 40 * 100),
            "btki_pct": round(c["tx_btki"] / 40 * 100),
            "cr_pct": round(c["cr_rate"] * 100, 1),
            "os_2yr_pct": round(c["os_2yr"] * 100, 1),
            "richter_count": c["richter_count"],
            "cll_risk": c["cll_risk"],
            "key_rule": c["key_rule"],
            "targeted_rx": c["targeted_rx"],
        }
        for c in cohorts
    ]

    return {
        "atlas": "Hereditary-CLL-Predisposition-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": 8,
        "total_patients": total_patients,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_age_overall": mean_age_overall,
        "stage_distribution": {
            "binet_a": total_binet_a,
            "binet_b": total_binet_b,
            "binet_c": total_binet_c,
            "binet_a_pct": round(total_binet_a / total_patients * 100),
            "binet_c_pct": round(total_binet_c / total_patients * 100),
        },
        "cytogenetic_distribution": {
            "del13q": total_del13q,
            "del17p": total_del17p,
            "del11q": total_del11q,
            "trisomy12": total_trisomy12,
            "del9p21": total_del9p21,
        },
        "treatment_distribution": {
            "btki": total_btki,
            "venetoclax": total_venetoclax,
            "watch_wait": total_watch_wait,
        },
        "outcomes": {
            "avg_cr_pct": avg_cr,
            "avg_os_2yr_pct": avg_os2yr,
            "total_richter": total_richter,
            "richter_pct": round(total_richter / total_patients * 100, 1),
        },
        "key_clinical_distinctions": KEY_CLINICAL_DISTINCTIONS,
        "per_gene": per_gene,
    }


def generate_breakdown() -> dict:
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(8)]
    gene_data = {c["gene"]: c for c in cohorts}
    return {
        "atlas": "Hereditary-CLL-Predisposition-Atlas",
        "gene_data": gene_data,
        "atlas_genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein": g["protein"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "cll_risk": g["cll_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_rule": g["key_rule"],
                "key_avoid": g["key_avoid"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_definitions() -> dict:
    defs = []
    for g in ATLAS_GENES:
        defs.append({
            "term": f"{g['gene']} — {g['syndrome']}",
            "definition": (
                f"{g['locus']} {g['inheritance']} — "
                f"{g['cll_risk']} — "
                f"Pathognomonic: {g['pathognomonic']} — "
                f"Key rule: {g['key_rule']} — "
                f"Avoid: {g['key_avoid']} — "
                f"Surveillance: {g['surveillance']} — "
                f"Targeted Rx: {g['targeted_rx']}"
            ),
        })
        defs.append({
            "term": f"{g['gene']} molecular size",
            "definition": g["protein_size"],
        })
    defs.append({
        "term": "KEY CLINICAL DISTINCTIONS — Hereditary CLL Predisposition 8-Gene Atlas",
        "definition": " | ".join(KEY_CLINICAL_DISTINCTIONS),
    })
    return {"atlas": "Hereditary-CLL-Predisposition-Atlas", "definitions": defs}
