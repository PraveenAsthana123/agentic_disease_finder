#!/usr/bin/env python3
"""Hereditary-BL-Predisposition-Atlas -- Complete 8-Gene Reference
Burkitt Lymphoma (BL) germline predisposition.

ATM      (ATM kinase; 3056aa; 11q22.3; AD LOF;
          Ataxia-Telangiectasia (OMIM 208900);
          biallelic A-T: BL/B-cell lymphoma 5-15x HIGHEST;
          monoallelic BL 2-3x; RADIOSENSITIVITY ABSOLUTE;
          high-dose RT CI; cyclophosphamide DSB-repair impaired;
          ceralasertib ATRi synthetic lethality;
          seed SEED_BASE+0).
TP53     (p53; 393aa; 17p13.1; AD LOF;
          Li-Fraumeni Syndrome (OMIM 151623);
          LFS BL 2-4x; 17p deletion 20-30% BL;
          AVOID RADIATION ABSOLUTELY; WB-MRI Toronto annually;
          venetoclax backbone; R-CODOX-M/IVAC with dose-reduce RT;
          seed SEED_BASE+1).
BRCA2    (BRCA2; 3418aa; 13q12.3; AD LOF;
          HBOC/Fanconi D1 (OMIM 605724/600185);
          biallelic FA-D1: childhood BL PATHOGNOMONIC;
          monoallelic HBOC: BL/lymphoma 2-3x;
          AVOID alkylating ABSOLUTELY (FA-D1 biallelic);
          olaparib maintenance monoallelic; sibling donors EXCLUDE FA;
          seed SEED_BASE+2).
PIK3CD   (PI3K-delta; 1044aa; 1p36.22; AD GOF;
          APDS1/PASLI (OMIM 615513);
          PI3Kdelta GOF; EBV-driven BL 10-15%;
          leniolisib FDA2023; acyclovir MANDATORY prophylaxis;
          sirolimus bridge; live vaccines ABSOLUTELY CI;
          EBV/CMV viral load Q3M; seed SEED_BASE+3).
CARD11   (CARD11; 1154aa; 7p22.2; AD LOF/GOF;
          BENTA Disease GOF (OMIM 616452) / CID LOF (OMIM 615206);
          NF-kB dysregulation; B-cell lymphoma/BL predisposition;
          venetoclax preferred; MALT1 inhibitor JNJ-67856633;
          LOF-CID: HSCT urgent mandatory; seed SEED_BASE+4).
TNFRSF13B (TACI; 293aa; 17p11.2; AD LOF;
          CVID2 (OMIM 240500);
          NHL 10-12x HIGHEST; IVIG MANDATORY IgG threshold ≥7 g/L;
          C104R European founder 1 in 500; BL via impaired humoral immunity;
          rituximab + IVIG co-mandatory; seed SEED_BASE+5).
CD27     (CD27; 260aa; 12p13.31; AR LOF;
          CD27 Deficiency (OMIM 615122);
          EBV-driven B-cell LPD/BL 30-50x HIGHEST;
          HLH trigger; HSCT MANDATORY; EBV viral load Q3M;
          rituximab + IVIG; HLH-2004 etoposide; seed SEED_BASE+6).
NF1      (neurofibromin; 2839aa; 17q11.2; AD LOF;
          Neurofibromatosis type 1 (OMIM 162200);
          NHL/BL 2-3x; RAS-MAPK dysregulation;
          cafe-au-lait PATHOGNOMONIC (≥6 >5mm); Lisch nodules;
          MPNST 8-13%; selumetinib FDA2020; seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3630-3637)
"""
import random

SEED_BASE = 3630

ATLAS_GENES = [
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD LOF -- 3056aa -- "
            "ATM-350kDa-PIK3-Kinase-"
            "DDR-Sensor-MRN-Complex-H2AX-BRCA1-CHK2-p53-"
            "BL-2-3x-Monoallelic-5-15x-Biallelic-A-T-RADIOSENSITIVITY-ABSOLUTE-"
            "A-T-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / ~350 kDa / 11q22.3 ATM BL molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; PIKK-family Ser/Thr kinase; activated by DSBs via MRN complex; "
            "  HEAT repeats (aa 1-3000): scaffolding; FAT domain; kinase domain; FATC domain; "
            "  Key substrates: H2AX (γH2AX foci), p53, BRCA1, CHK2, 53BP1, SMC1; "
            "  Biallelic loss (A-T): combined immunodeficiency + cerebellar ataxia + telangiectasia; "
            "  Monoallelic heterozygous: BL 2-3x; NHL 4-5x; "
            "BL MECHANISM: "
            "  ATM LOF impairs BCR rearrangement surveillance → B-cell clonal expansion; "
            "  Biallelic A-T: B-cell lymphoma/BL 5-15x HIGHEST; "
            "  MYC translocation surveillance impaired in ATM-deficient B cells; "
            "  c-MYC/IgH t(8;14) — ATM co-mutation accelerates progression; "
            "CLINICAL RULES FOR ATM-BL: "
            "  RADIOSENSITIVITY ABSOLUTE — high-dose cranial RT CI; reduce RT fields; "
            "  R-CODOX-M/IVAC: standard BL; cyclophosphamide DSB repair impaired — dose-monitor; "
            "  Ceralasertib (AZD6738, ATRi): synthetic lethality with ATM LOF; "
            "  Intrathecal prophylaxis still required — NOT whole-brain RT; "
            "  EBV viral load monitoring Q3M (B-cell immune surveillance impaired); "
            "  Cascade ATM testing: first-degree family members; IgA level baseline; "
            "KEY VARIANTS: "
            "  Monoallelic: c.7271T>G (p.Val2424Gly), c.1009C>T (p.Arg337Cys), del11q22 large; "
            "  Biallelic: truncating + truncating (A-T full phenotype); "
            "SURVEILLANCE: "
            "  Annual lymph node exam + CBC; EBV Q3M; IgA level; "
            "  Biallelic: neurological surveillance; chest CT pulmonary atelectasis; "
            "TARGETED Rx: "
            "  Ceralasertib (ATRi): synthetic lethality; phase II BL relapsed; "
            "  R-CODOX-M/IVAC standard BL regimen with RT-dose reduction; "
            "  Rituximab-containing regimen mandatory B-cell CD20+; "
        ),
        "syndrome": "Ataxia-Telangiectasia (A-T) / ATM-BL Predisposition",
        "inheritance": "AD LOF (monoallelic); AR biallelic = A-T",
        "bl_risk": "Biallelic A-T BL 5-15x HIGHEST; monoallelic BL 2-3x; NHL 4-5x; MYC surveillance impaired",
        "pathognomonic": "A-T biallelic: cerebellar ataxia + telangiectasia + BL/lymphoma 5-15x — IgA low PATHOGNOMONIC A-T",
        "key_rule": "RADIOSENSITIVITY ABSOLUTE — cranial RT CI; ceralasertib ATRi synthetic lethality; intrathecal NOT whole-brain RT",
        "key_avoid": "High-dose RT cranial (ABSOLUTELY CI); unmonitored cyclophosphamide without DSB-repair assessment",
        "surveillance": "Annual CBC + lymph node exam; EBV Q3M; IgA baseline; biallelic: neuro + pulmonary surveillance",
        "targeted_rx": "Ceralasertib ATRi synthetic lethality; R-CODOX-M/IVAC with RT dose-reduction; rituximab mandatory",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "DBD-Tetramerisation-TAD-PrD-"
            "LFS-BL-2-4x-17p-del-20-30pct-AVOID-RADIATION-ABSOLUTELY-"
            "LFS-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 BL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; tumour suppressor; tetramer transcription factor; "
            "  TAD1 (aa 1-40): p300/CBP; MDM2 binding; "
            "  TAD2 (aa 40-61): TFIID; "
            "  PRD (aa 62-91): protein-protein interaction; "
            "  DBD (aa 94-292): DNA-binding; hotspot mutations R175H/G245S/R248W/R248Q/R249S/R273H/R273C/R282W; "
            "  NLS (aa 316-325); "
            "  Tetramerisation (aa 323-356): critical for oligomer function; "
            "  REG (aa 364-393): negative regulation; "
            "  Wild-type p53: apoptosis, cell-cycle arrest, senescence; "
            "  LOF/GOF: tumour suppressor lost + dominant-negative over WT p53; "
            "LFS BL CONTEXT: "
            "  LFS: germline TP53 LOF; BL 2-4x; 17p deletion 20-30% BL; "
            "  BL with p53 co-LOF: highly aggressive, complex karyotype; "
            "  Spontaneous remission NOT possible; MYC + TP53 co-loss = aggressive BL; "
            "CLINICAL RULES FOR TP53-BL: "
            "  AVOID RADIATION ABSOLUTELY — secondary sarcoma risk; "
            "  WB-MRI Toronto Protocol annually; "
            "  Venetoclax backbone: BCL-2 inhibition effective in TP53-mutant BL; "
            "  R-CODOX-M/IVAC ± venetoclax combination for TP53-BL; "
            "  No RT in CNS prophylaxis — intrathecal MTX only; "
            "KEY VARIANTS: "
            "  Dominant hotspots: R175H, R248W, R248Q, R273H, R273C; "
            "  LOF: nonsense, frameshift, splice site; "
            "SURVEILLANCE: "
            "  WB-MRI annually (Toronto Protocol); annual CBC; "
            "  First-degree family members: TP53 germline testing; "
            "TARGETED Rx: "
            "  Venetoclax BCL-2i: TP53-mutant BL backbone; "
            "  R-CODOX-M/IVAC standard + venetoclax combination; "
            "  APR-246 (eprenetapopt): p53 reactivation investigational; "
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) / TP53-BL Predisposition",
        "inheritance": "AD LOF heterozygous",
        "bl_risk": "LFS BL 2-4x; 17p deletion 20-30% BL; MYC+TP53 co-loss highly aggressive; complex karyotype",
        "pathognomonic": "LFS: BL + adrenocortical carcinoma/osteosarcoma in family — TP53 germline PATHOGNOMONIC",
        "key_rule": "AVOID RADIATION ABSOLUTELY; WB-MRI Toronto annually; venetoclax backbone TP53-mutant BL",
        "key_avoid": "ANY radiation field (ABSOLUTELY CI); intrathecal prophylaxis OK — NOT whole-brain RT",
        "surveillance": "WB-MRI annually Toronto; annual CBC; first-degree family TP53 testing; abdomen palpation",
        "targeted_rx": "Venetoclax BCL-2i backbone; R-CODOX-M/IVAC ± venetoclax; APR-246 investigational p53-reactivation",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 AD LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-DNA-Repair-Scaffold-"
            "OB-Folds-BRC-Repeats-RAD51-"
            "FA-D1-Biallelic-Childhood-BL-PATHOGNOMONIC-AVOID-ALKYLATING-"
            "HBOC-OMIM-605724-FA-D1-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / ~384 kDa / 13q12.3 BRCA2 BL molecular context: "
            "STRUCTURE: "
            "  3418 aa / ~384 kDa; key HR-pathway scaffold; "
            "  N-terminal PALB2-binding domain (aa 10-40); "
            "  BRC repeats (aa 1002-2085): 8 repeats, each binds RAD51 monomer; "
            "  OB-folds (aa 2402-3190): ssDNA binding + RPA displacement + RAD51 nucl loading; "
            "  C-terminal RAD51-binding (aa 3265-3330): meiotic recombination; "
            "  BRCA2 loads RAD51 onto ssDNA at DSBs; LOF → HR deficiency; "
            "FA-D1 BL CONTEXT: "
            "  Biallelic BRCA2 LOF = Fanconi D1 (FA-D1): childhood BL PATHOGNOMONIC; "
            "  FA-D1: AML by age 3yr; medulloblastoma; BL/lymphoma; "
            "  AVOID ALL ALKYLATING agents absolutely (bifunctional crosslinkers = FA trigger); "
            "  AVOID RT with concurrent alkylating; "
            "  Sibling donors: EXCLUDE from haematopoietic transplant without FA testing; "
            "  Monoallelic BRCA2 HBOC: BL/lymphoma 2-3x; HRD 60%; cisplatin preferred; "
            "CLINICAL RULES FOR BRCA2-BL: "
            "  BIALLELIC FA-D1: AVOID ALKYLATING ABSOLUTELY — MMC fragility test MANDATORY before treatment; "
            "  MONOALLELIC HBOC: HRD 60% — olaparib maintenance; cisplatin preferred; "
            "  FA-D1 HSCT: matched sibling excluded unless FA-negative; "
            "  No cyclophosphamide (alkylating) in biallelic FA-D1 BL; "
            "KEY VARIANTS: "
            "  Founder variants: c.5946delT (Israeli Arab), c.6174delT (Ashkenazi), c.1100delC (N/A); "
            "  LOF: frameshift, nonsense, splice across 27 exons; "
            "SURVEILLANCE: "
            "  Annual breast MRI (monoallelic BRCA2 females); "
            "  CBC annually; FA-D1: haematology surveillance from infancy; "
            "TARGETED Rx: "
            "  Olaparib PARP inhibitor (FDA approved): monoallelic HBOC-lymphoma maintenance; "
            "  Cisplatin-containing: HRD 60% cisplatin-sensitive; "
            "  Sibling donor exclusion protocol MANDATORY FA-D1; "
        ),
        "syndrome": "HBOC/Fanconi D1 (FA-D1) / BRCA2-BL Predisposition",
        "inheritance": "AD LOF monoallelic HBOC; AR biallelic = FA-D1 childhood BL",
        "bl_risk": "FA-D1 biallelic: childhood BL PATHOGNOMONIC; monoallelic HBOC: BL 2-3x; HRD 60%; cisplatin-sensitive",
        "pathognomonic": "FA-D1 biallelic: childhood BL + AML + medulloblastoma triad PATHOGNOMONIC; MMC fragility POSITIVE",
        "key_rule": "BIALLELIC: AVOID ALKYLATING ABSOLUTELY — MMC fragility MANDATORY; sibling donors EXCLUDE FA testing",
        "key_avoid": "Cyclophosphamide/alkylating in FA-D1 biallelic (ABSOLUTELY CI); sibling donor without FA screen",
        "surveillance": "MMC fragility test before treatment FA-D1; annual breast MRI monoallelic; CBC from infancy FA-D1",
        "targeted_rx": "Olaparib PARP inhibitor monoallelic; cisplatin-preferred HRD-60pct; FA-D1 HSCT with FA-negative sibling",
    },
    {
        "gene": "PIK3CD",
        "protein": (
            "PIK3CD -- 1p36.22 AD GOF -- 1044aa -- "
            "PI3K-delta-119kDa-Class-IA-PI3K-"
            "ABD-RBD-C2-Helical-Kinase-"
            "APDS1-EBV-BL-10-15pct-Leniolisib-FDA2023-"
            "APDS1-OMIM-615513"
        ),
        "locus": "1p36.22",
        "protein_size": (
            "1044 aa / ~119 kDa / 1p36.22 PIK3CD BL molecular context: "
            "STRUCTURE: "
            "  1044 aa / ~119 kDa; PI3K catalytic subunit delta (p110δ); Class IA PI3K; "
            "  ABD domain (aa 1-108): adaptor-binding (p85 regulatory subunit); "
            "  RBD domain (aa 190-291): RAS-binding; "
            "  C2 domain (aa 358-480): membrane binding; "
            "  Helical domain (aa 481-687): p85 nSH2 inhibition interface; "
            "  Kinase domain (aa 726-1044): catalytic; GOF variants E1021K/E525K; "
            "  PI3Kδ GOF → hyperactivation PIP3 → AKT/mTOR → B-cell senescence/apoptosis resistance; "
            "APDS1 BL CONTEXT: "
            "  APDS1 (Activated PI3K Delta Syndrome type 1): AD GOF PIK3CD; "
            "  EBV-driven B-cell lymphoma/BL 10-15%; "
            "  Nodular lymphoid hyperplasia → EBV+BL; "
            "  Recurrent respiratory infections, splenomegaly, lymphadenopathy; "
            "  EBV-driven MYC translocation in APDS1 B cells; "
            "CLINICAL RULES FOR PIK3CD-BL: "
            "  Leniolisib (PI3Kδ inhibitor, FDA Feb 2023): first targeted APDS1 therapy; "
            "  Acyclovir prophylaxis MANDATORY (EBV/VZV antiviral coverage); "
            "  CMV/EBV viral load Q3M MANDATORY; "
            "  Sirolimus (mTOR): bridge before leniolisib or if unavailable; "
            "  Live vaccines ABSOLUTELY CI (risk live-attenuated replication); "
            "  IVIG replacement if hypogammaglobulinaemia; "
            "KEY VARIANTS: "
            "  E1021K (c.3061G>A): most common APDS1 GOF (~65% European); "
            "  E525K (c.1573G>A): second most common; "
            "SURVEILLANCE: "
            "  EBV/CMV viral load Q3M MANDATORY; CBC quarterly; IgG levels q3M; "
            "  Annual CT scan (lymphadenopathy + splenomegaly); "
            "TARGETED Rx: "
            "  Leniolisib (PI3Kδi, FDA2023): first-line APDS1; "
            "  Sirolimus mTORi bridge; "
            "  Acyclovir MANDATORY prophylaxis; IVIG replacement; "
        ),
        "syndrome": "APDS1 / PASLI / Activated PI3Kδ Syndrome type 1 / PIK3CD-BL",
        "inheritance": "AD GOF heterozygous",
        "bl_risk": "EBV-driven BL 10-15%; APDS1 nodular lymphoid hyperplasia → EBV+BL; MYC translocation EBV-driven",
        "pathognomonic": "APDS1: recurrent infections + splenomegaly + nodular lymphoid hyperplasia + EBV+BL young onset",
        "key_rule": "Leniolisib FDA2023 first targeted therapy; acyclovir MANDATORY prophylaxis; EBV/CMV Q3M; live vaccines CI",
        "key_avoid": "Live vaccines ABSOLUTELY CI; delay leniolisib without sirolimus bridge if unavailable",
        "surveillance": "EBV/CMV Q3M MANDATORY; CBC quarterly; IgG q3M; annual CT; cascade PIK3CD testing first-degree",
        "targeted_rx": "Leniolisib PI3Kδi FDA2023; sirolimus mTORi bridge; acyclovir MANDATORY prophylaxis; IVIG replacement",
    },
    {
        "gene": "CARD11",
        "protein": (
            "CARD11 -- 7p22.2 AD LOF/GOF -- 1154aa -- "
            "CARD11-133kDa-CBM-Complex-"
            "CARD-Coiled-Coil-MAGUK-PDZ-SH3-GUK-"
            "BENTA-GOF-NF-kB-BL-CID-LOF-HSCT-Venetoclax-MALT1i-"
            "BENTA-OMIM-616452-CID-OMIM-615206"
        ),
        "locus": "7p22.2",
        "protein_size": (
            "1154 aa / ~133 kDa / 7p22.2 CARD11 BL molecular context: "
            "STRUCTURE: "
            "  1154 aa / ~133 kDa; scaffolding protein of CBM signalosome (CARD11-BCL10-MALT1); "
            "  CARD domain (aa 1-97): homotypic caspase recruitment domain; "
            "  Coiled-coil domain (aa 130-333): oligomerisation; GOF variants in coiled-coil; "
            "  MAGUK module (PDZ aa 520-600; SH3 aa 640-710; GUK aa 730-1000): scaffold; "
            "  CARD11 bridges TCR/BCR signal → IKK complex → NF-κB activation → B-cell survival; "
            "  GOF (constitutive NF-kB): BENTA disease → B-cell expansion → lymphoma; "
            "  LOF (no NF-kB signalling): CID → immunodeficiency → lymphoma susceptibility; "
            "CARD11 BL CONTEXT: "
            "  BENTA (B-cell Expansion with NF-κB and T-cell Anergy, GOF): B-cell lymphoma/BL predisposition; "
            "  CID (Combined Immunodeficiency, LOF): impaired lymphocyte survival → lymphoma susceptibility; "
            "  DLBCL somatic CARD11 mutations 10-20%; germline predisposition via same pathway; "
            "  MYC translocation + NF-kB co-activation in BL aggressive variant; "
            "CLINICAL RULES FOR CARD11-BL: "
            "  BENTA GOF: venetoclax preferred (BCL-2 upregulated via NF-kB); "
            "  MALT1 inhibitor JNJ-67856633: MALT1 protease inhibitor; phase II; "
            "  LOF-CID: HSCT URGENT MANDATORY before lymphoma; "
            "  Live vaccines: assess carefully — CID-LOF: live vaccine risk; "
            "KEY VARIANTS: "
            "  GOF coiled-coil: L232LI, E127G, G123S, G116S; "
            "  LOF: R35Q (CID), frameshift, truncating; "
            "SURVEILLANCE: "
            "  CBC 6-monthly; lymphocyte subsets annually; "
            "  Annual CT if BENTA GOF; HSCT evaluation at CID-LOF diagnosis; "
            "TARGETED Rx: "
            "  Venetoclax BCL-2i: BENTA/NF-kB GOF preferred; "
            "  MALT1 inhibitor JNJ-67856633: CBM signalosome; "
            "  Rapamycin mTOR: BENTA disease adjunct; "
            "  CID-LOF: HSCT urgent; "
        ),
        "syndrome": "BENTA Disease (GOF) / CID (LOF) / CARD11-BL Predisposition",
        "inheritance": "AD GOF (BENTA); AD LOF (CID) heterozygous",
        "bl_risk": "BENTA GOF: B-cell lymphoma/BL via constitutive NF-kB; CID LOF: lymphoma susceptibility via immunodeficiency",
        "pathognomonic": "BENTA: B-cell expansion + T-cell anergy + splenomegaly + lymphoma — coiled-coil GOF PATHOGNOMONIC",
        "key_rule": "GOF BENTA: venetoclax preferred + MALT1 inhibitor; LOF CID: HSCT URGENT MANDATORY before decompensation",
        "key_avoid": "Delaying HSCT in CID-LOF until lymphoma occurs; rituximab monotherapy without venetoclax backbone BENTA",
        "surveillance": "CBC 6-monthly; lymphocyte subsets annually; annual CT BENTA; HSCT evaluation at CID-LOF diagnosis",
        "targeted_rx": "Venetoclax BCL-2i BENTA GOF; MALT1 inhibitor JNJ-67856633; rapamycin adjunct BENTA; CID-LOF HSCT urgent",
    },
    {
        "gene": "TNFRSF13B",
        "protein": (
            "TNFRSF13B -- 17p11.2 AD LOF -- 293aa -- "
            "TACI-31kDa-TNF-Receptor-Superfamily-"
            "Cysteine-Rich-TM-ICD-TRAF-Binding-"
            "CVID2-NHL-10-12x-HIGHEST-IVIG-MANDATORY-BL-via-Humoral-Immunity-"
            "CVID2-OMIM-240500"
        ),
        "locus": "17p11.2",
        "protein_size": (
            "293 aa / ~31 kDa / 17p11.2 TNFRSF13B BL molecular context: "
            "STRUCTURE: "
            "  293 aa / ~31 kDa; TACI (Transmembrane Activator and CAML Interactor); "
            "  Signal peptide (aa 1-17); "
            "  Extracellular cysteine-rich domain (aa 18-68): BAFF/APRIL ligand binding; "
            "  Second CRD (aa 68-118): binding affinity modulation; "
            "  Transmembrane domain (aa 138-160): type I single-pass; "
            "  Intracellular domain (aa 161-293): TRAF2/5/6 binding motifs; "
            "  TACI: receptor for BAFF (B-cell survival) and APRIL (class switching, IgA/IgG production); "
            "  LOF → CVID2 (impaired B-cell humoral immunity) → NHL predisposition 10-12x; "
            "CVID2 BL CONTEXT: "
            "  CVID2: commonest cause of severe antibody deficiency in adults; "
            "  NHL/BL 10-12x HIGHEST — BL via EBV-driven lymphoproliferation in IVIG-insufficient patients; "
            "  IVIG non-compliance → EBV reactivation → BL; "
            "  C104R founder variant: European 1 in 500; "
            "  A181E founder: European; heterozygous LOF + modifier variants; "
            "CLINICAL RULES FOR TNFRSF13B-BL: "
            "  IVIG MANDATORY — NEVER STOP (IgG trough ≥7 g/L); "
            "  Rituximab + IVIG CO-MANDATORY (rituximab depletes B cells → IgG falls); "
            "  EBV viral load Q3M surveillance; "
            "  Prophylactic antibiotics (cotrimoxazole/azithromycin) pulmonary prophylaxis; "
            "KEY VARIANTS: "
            "  C104R (c.310T>C): European founder 1 in 500; common polymorphism → CVID when compound; "
            "  A181E (c.542C>A): dominant CVID2; "
            "SURVEILLANCE: "
            "  IgG trough monthly IVIG; EBV Q3M; annual CT lymph nodes; "
            "  PFT annual (pulmonary CVID complications); "
            "TARGETED Rx: "
            "  IVIG NEVER STOP; rituximab + IVIG co-mandatory in NHL; "
            "  R-CODOX-M/IVAC + IVIG for active BL; "
        ),
        "syndrome": "CVID type 2 (CVID2) / TACI Deficiency / TNFRSF13B-BL Predisposition",
        "inheritance": "AD LOF heterozygous (CVID2); compound heterozygous high penetrance",
        "bl_risk": "NHL 10-12x HIGHEST; BL via EBV in IVIG-insufficient CVID2; C104R European founder 1 in 500",
        "pathognomonic": "CVID2: recurrent sinopulmonary infections + low IgG + NHL/BL 10-12x — IgG <5g/L PATHOGNOMONIC CVID",
        "key_rule": "IVIG MANDATORY NEVER STOP (IgG ≥7g/L trough); rituximab + IVIG CO-MANDATORY; EBV Q3M surveillance",
        "key_avoid": "Stopping IVIG during rituximab therapy (IgG falls precipitously → life-threatening hypogammaglobulinaemia)",
        "surveillance": "IgG trough monthly IVIG; EBV Q3M; annual CT lymph nodes; PFT annual; C104R cascade testing family",
        "targeted_rx": "IVIG never stop; rituximab + IVIG co-mandatory NHL; R-CODOX-M/IVAC + IVIG for active BL treatment",
    },
    {
        "gene": "CD27",
        "protein": (
            "CD27 -- 12p13.31 AR LOF -- 260aa -- "
            "CD27-29kDa-TNF-Receptor-"
            "CRD1-CRD2-TM-ICD-TRAF2-5-"
            "CD27-Deficiency-EBV-B-cell-LPD-BL-30-50x-HIGHEST-HSCT-MANDATORY-"
            "CD27-Deficiency-OMIM-615122"
        ),
        "locus": "12p13.31",
        "protein_size": (
            "260 aa / ~29 kDa / 12p13.31 CD27 BL molecular context: "
            "STRUCTURE: "
            "  260 aa / ~29 kDa; TNF receptor superfamily member 7 (TNFRSF7); "
            "  Signal peptide (aa 1-19); "
            "  CRD1 (aa 20-60): TRAF2/5 binding; "
            "  CRD2 (aa 61-100): CD70 ligand interaction; "
            "  Stalk region (aa 101-180); "
            "  Transmembrane (aa 181-205): type I single-pass membrane; "
            "  Intracellular domain (aa 206-260): TRAF2/5 binding motifs → NF-kB/MAPK; "
            "  CD27-CD70 axis: B-cell memory, plasma cell differentiation, T-cell activation; "
            "  LOF → impaired EBV surveillance → EBV-driven B-cell lymphoproliferation → BL; "
            "CD27 BL CONTEXT: "
            "  CD27 Deficiency: AR biallelic LOF; "
            "  EBV-driven B-cell LPD/BL 30-50x HIGHEST (EBV not cleared without CD27 co-stimulation); "
            "  HLH trigger: fulminant EBV HLH in CD27-deficient patients; "
            "  Hypogammaglobulinaemia + lymphoproliferation; "
            "  B-cell lymphoma (including BL morphology) by age 5-20yr; "
            "CLINICAL RULES FOR CD27-BL: "
            "  HSCT MANDATORY curative (reconstitutes EBV surveillance); "
            "  EBV viral load Q3M MANDATORY; "
            "  Rituximab: applicable (B-cell CD20+) + IVIG; "
            "  HLH: etoposide + dexamethasone (HLH-2004) if HLH triggered; "
            "KEY VARIANTS: "
            "  Homozygous truncating: c.24+1G>A splice; p.Arg27Ter; "
            "  Compound heterozygous: missense in CRD + truncating; "
            "SURVEILLANCE: "
            "  EBV viral load Q3M MANDATORY; CBC quarterly; IgG levels; "
            "  Annual imaging (lymphadenopathy); HSCT evaluation at diagnosis; "
            "TARGETED Rx: "
            "  HSCT MANDATORY curative; "
            "  Rituximab + IVIG for EBV+BL; "
            "  HLH-2004: etoposide+dex if HLH trigger; "
        ),
        "syndrome": "CD27 Deficiency / TNFRSF7 Deficiency / CD27-BL Predisposition",
        "inheritance": "AR biallelic LOF",
        "bl_risk": "EBV-driven B-cell LPD/BL 30-50x HIGHEST; HLH trigger; hypogammaglobulinaemia; lymphoma age 5-20yr",
        "pathognomonic": "CD27 deficiency: EBV+ B-cell LPD + HLH + hypogammaglobulinaemia triad — biallelic PATHOGNOMONIC",
        "key_rule": "HSCT MANDATORY curative; EBV viral load Q3M MANDATORY; rituximab+IVIG; HLH-2004 etoposide if triggered",
        "key_avoid": "Delaying HSCT after B-cell LPD established (BL risk maximal); missing HLH trigger requiring etoposide",
        "surveillance": "EBV viral load Q3M MANDATORY; CBC quarterly; IgG q3M; annual imaging; HSCT evaluation at diagnosis",
        "targeted_rx": "HSCT mandatory curative; rituximab+IVIG EBV+BL; HLH-2004 etoposide+dex; IVIG replacement hypogamma",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 AD LOF -- 2839aa -- "
            "Neurofibromin-327kDa-RAS-GAP-"
            "CRAL-TRIO-PH-like-GRD-SEC14-"
            "NF1-NHL-BL-2-3x-Cafe-au-Lait-PATHOGNOMONIC-Selumetinib-FDA2020-"
            "NF1-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / ~327 kDa / 17q11.2 NF1 BL molecular context: "
            "STRUCTURE: "
            "  2839 aa / ~327 kDa; neurofibromin; major RAS GTPase-activating protein; "
            "  CRAL-TRIO domain (aa 1-100): lipid binding; "
            "  PH-like domain (aa 180-280); "
            "  GRD domain (aa 1172-1530): GAP-related domain; converts RAS-GTP → RAS-GDP; "
            "  Sec14 lipid-transfer domain (aa 1582-1738); "
            "  C-terminal PH-FERM-HEAT (aa 2000-2839): scaffold interactions; "
            "  NF1 LOF → reduced GAP activity → RAS-GTP accumulation → MAPK/ERK hyperactivation; "
            "  LOF mechanism: haploinsufficiency (monoallelic) + LOH somatic (second hit tumour); "
            "NF1 BL CONTEXT: "
            "  NF1 (Neurofibromatosis type 1): NHL/BL 2-3x; RAS-MAPK lymphoproliferation; "
            "  MPNST (malignant peripheral nerve sheath tumour): 8-13% NF1 lifetime; "
            "  Café-au-lait ≥6 (>5mm prepubertal / >15mm postpubertal): PATHOGNOMONIC NF1 diagnosis; "
            "  Lisch nodules (iris hamartomas): NF1 pathognomonic; "
            "  BL in NF1: RAS-MAPK driven + MYC translocation susceptibility; "
            "  Selumetinib: MEK1/2 inhibitor; FDA2020 paediatric plexiform neurofibromas; "
            "CLINICAL RULES FOR NF1-BL: "
            "  Selumetinib (MEKi, FDA2020): plexiform neurofibromas; MAPK pathway-targeted; "
            "  Standard R-CODOX-M/IVAC BL regimen: applicable NF1-BL; "
            "  MPNST: excision + radiation (not CI unlike TP53/ATM); "
            "  BRAF-V600E co-mutation: rare in NF1 — cobimetinib/trametinib combination; "
            "KEY VARIANTS: "
            "  NF1 exon 20-22 (microdelection ~5%): severe phenotype; "
            "  Pathogenic missense + truncating across 57 coding exons; "
            "SURVEILLANCE: "
            "  Annual neurological + MRI spine/brain; annual CBC; "
            "  Annual skin exam neurofibromas; BP monitoring; "
            "TARGETED Rx: "
            "  Selumetinib MEKi FDA2020 plexiform NF; "
            "  Standard BL R-CODOX-M/IVAC; "
            "  MPNST: surgery + chemoRT; "
        ),
        "syndrome": "Neurofibromatosis Type 1 (NF1) / NF1-BL Predisposition",
        "inheritance": "AD LOF heterozygous",
        "bl_risk": "NHL/BL 2-3x; RAS-MAPK lymphoproliferation; MPNST 8-13%; MYC translocation susceptibility NF1",
        "pathognomonic": "NF1: ≥6 cafe-au-lait + Lisch nodules + BL/NHL 2-3x — cafe-au-lait ≥6 PATHOGNOMONIC NF1 diagnosis",
        "key_rule": "Selumetinib MEKi FDA2020 plexiform neurofibromas; standard R-CODOX-M/IVAC for NF1-BL; RT not CI (unlike ATM/TP53)",
        "key_avoid": "Confusing MPNST AVOID with BL AVOID RT — NF1 RT is NOT CI (unlike ATM/TP53); delaying selumetinib plexiform",
        "surveillance": "Annual neuro + MRI spine/brain; annual CBC; annual skin exam; BP monitoring; cascade NF1 testing family",
        "targeted_rx": "Selumetinib MEKi FDA2020 plexiform NF; standard R-CODOX-M/IVAC BL; MPNST surgery + chemoRT NF1",
    },
]

KEY_CLINICAL_DISTINCTIONS = [
    "ATM-RADIOSENSITIVITY-ABSOLUTE-BL: Biallelic A-T BL 5-15x — cranial RT ABSOLUTELY CI; ceralasertib ATRi synthetic lethality; intrathecal NOT whole-brain RT; cyclophosphamide DSB-repair impaired.",
    "TP53-LFS-AVOID-RADIATION-ABSOLUTELY: LFS germline TP53 — ANY radiation causes secondary sarcoma; venetoclax backbone TP53-mutant BL; WB-MRI Toronto Protocol annually; APR-246 investigational.",
    "BRCA2-FA-D1-AVOID-ALKYLATING-ABSOLUTELY: FA-D1 biallelic childhood BL PATHOGNOMONIC — AVOID ALL ALKYLATING absolutely; MMC fragility test MANDATORY before treatment; sibling donors EXCLUDE FA testing.",
    "PIK3CD-APDS1-LIVE-VACCINES-CI: APDS1 live vaccines ABSOLUTELY CI; acyclovir MANDATORY prophylaxis; leniolisib FDA2023 first targeted therapy; EBV/CMV viral load Q3M.",
    "CARD11-LOF-CID-HSCT-URGENT: CARD11 CID-LOF HSCT URGENT MANDATORY; venetoclax preferred BENTA GOF; MALT1 inhibitor JNJ-67856633 CBM signalosome; NF-kB dysregulation BL.",
    "TNFRSF13B-IVIG-NEVER-STOP: CVID2 IVIG NEVER STOP (IgG ≥7g/L trough); rituximab + IVIG CO-MANDATORY (rituximab depletes B cells → IgG falls → fatal hypogammaglobulinaemia).",
    "CD27-HSCT-MANDATORY-EBV-BL: CD27 deficiency EBV-driven BL 30-50x HIGHEST — HSCT MANDATORY curative; EBV viral load Q3M; HLH-2004 etoposide if HLH triggered.",
    "NF1-RT-NOT-CI-UNLIKE-ATM-TP53: NF1-BL: RT is NOT CI (unlike ATM/TP53); selumetinib MEKi FDA2020 plexiform neurofibromas; standard R-CODOX-M/IVAC; MPNST 8-13% lifetime surveillance.",
]


def _make_cohort(gene_def: dict, seed: int) -> dict:
    """Generate a 40-patient BL cohort for one predisposition gene."""
    rng = random.Random(seed)

    gene  = gene_def["gene"]
    locus = gene_def["locus"]
    inh   = gene_def["inheritance"]

    n = 40

    # Average age at BL diagnosis (per-gene, Gaussian σ≈12, clamp 2-75)
    avg_ages = {
        "ATM": 18, "TP53": 22, "BRCA2": 8, "PIK3CD": 16,
        "CARD11": 25, "TNFRSF13B": 38, "CD27": 12, "NF1": 28,
    }
    mu = avg_ages.get(gene, 20)
    ages = sorted([rng.gauss(mu, 12) for _ in range(n)])
    ages = [max(2, min(75, round(a, 1))) for a in ages]

    # BL subtype distribution per gene
    subtype_dist = {
        "ATM":       [("Sporadic-BL", 0.55), ("Endemic-BL", 0.75), ("Immunodeficiency-BL", 1.00)],
        "TP53":      [("Sporadic-BL", 0.50), ("Endemic-BL", 0.70), ("Immunodeficiency-BL", 1.00)],
        "BRCA2":     [("FA-D1-BL", 0.70), ("Sporadic-BL", 0.90), ("Endemic-BL", 1.00)],
        "PIK3CD":    [("Immunodeficiency-BL", 0.65), ("Sporadic-BL", 0.90), ("Endemic-BL", 1.00)],
        "CARD11":    [("Immunodeficiency-BL", 0.60), ("Sporadic-BL", 0.80), ("Endemic-BL", 1.00)],
        "TNFRSF13B": [("Immunodeficiency-BL", 0.70), ("Sporadic-BL", 0.90), ("Endemic-BL", 1.00)],
        "CD27":      [("Immunodeficiency-BL", 0.80), ("EBV+BL", 0.95), ("Sporadic-BL", 1.00)],
        "NF1":       [("Sporadic-BL", 0.60), ("Immunodeficiency-BL", 0.80), ("Endemic-BL", 1.00)],
    }
    def pick_subtype(gene_key):
        dist = subtype_dist.get(gene_key, [("Sporadic-BL", 0.60), ("Immunodeficiency-BL", 0.85), ("Endemic-BL", 1.00)])
        r = rng.random()
        for label, threshold in dist:
            if r < threshold:
                return label
        return dist[-1][0]

    subtypes = [pick_subtype(gene) for _ in range(n)]
    sporadic_n      = sum(1 for s in subtypes if s == "Sporadic-BL")
    endemic_n       = sum(1 for s in subtypes if s == "Endemic-BL")
    immunodef_n     = sum(1 for s in subtypes if "Immunodeficiency" in s or "EBV" in s or "FA-D1" in s)

    # EBV+ rate per gene
    ebv_rates = {
        "ATM": 0.20, "TP53": 0.15, "BRCA2": 0.25, "PIK3CD": 0.70,
        "CARD11": 0.50, "TNFRSF13B": 0.65, "CD27": 0.90, "NF1": 0.20,
    }
    ebv_pos = [rng.random() < ebv_rates.get(gene, 0.30) for _ in range(n)]

    # CNS involvement rate
    cns_rates = {
        "ATM": 0.30, "TP53": 0.35, "BRCA2": 0.20, "PIK3CD": 0.25,
        "CARD11": 0.30, "TNFRSF13B": 0.22, "CD27": 0.28, "NF1": 0.25,
    }
    cns_inv = [rng.random() < cns_rates.get(gene, 0.25) for _ in range(n)]

    # HLH trigger rate
    hlh_rates = {
        "ATM": 0.05, "TP53": 0.04, "BRCA2": 0.08, "PIK3CD": 0.20,
        "CARD11": 0.15, "TNFRSF13B": 0.12, "CD27": 0.45, "NF1": 0.04,
    }
    hlh_trigger = [rng.random() < hlh_rates.get(gene, 0.08) for _ in range(n)]

    # HSCT rate
    hsct_rates = {
        "ATM": 0.10, "TP53": 0.08, "BRCA2": 0.35, "PIK3CD": 0.30,
        "CARD11": 0.40, "TNFRSF13B": 0.12, "CD27": 0.85, "NF1": 0.08,
    }
    hsct = [rng.random() < hsct_rates.get(gene, 0.15) for _ in range(n)]

    # CR rate
    cr_rates = {
        "ATM": 0.70, "TP53": 0.62, "BRCA2": 0.58, "PIK3CD": 0.65,
        "CARD11": 0.60, "TNFRSF13B": 0.72, "CD27": 0.62, "NF1": 0.75,
    }
    cr = [rng.random() < cr_rates.get(gene, 0.65) for _ in range(n)]

    # 2yr OS rate (BL generally high cure rate with treatment)
    os2_rates = {
        "ATM": 0.65, "TP53": 0.55, "BRCA2": 0.48, "PIK3CD": 0.62,
        "CARD11": 0.58, "TNFRSF13B": 0.70, "CD27": 0.60, "NF1": 0.72,
    }
    os2 = [rng.random() < os2_rates.get(gene, 0.62) for _ in range(n)]

    # Treatment: R-CODOX-M/IVAC
    rcodox_rates = {
        "ATM": 0.75, "TP53": 0.70, "BRCA2": 0.50, "PIK3CD": 0.65,
        "CARD11": 0.60, "TNFRSF13B": 0.72, "CD27": 0.55, "NF1": 0.78,
    }
    tx_rcodox = [rng.random() < rcodox_rates.get(gene, 0.65) for _ in range(n)]

    # Treatment: venetoclax
    venetoclax_rates = {
        "ATM": 0.20, "TP53": 0.45, "BRCA2": 0.25, "PIK3CD": 0.30,
        "CARD11": 0.50, "TNFRSF13B": 0.20, "CD27": 0.25, "NF1": 0.15,
    }
    tx_venetoclax = [rng.random() < venetoclax_rates.get(gene, 0.25) for _ in range(n)]

    # Treatment: targeted agent
    targeted_rates = {
        "ATM": 0.25, "TP53": 0.35, "BRCA2": 0.30, "PIK3CD": 0.55,
        "CARD11": 0.45, "TNFRSF13B": 0.30, "CD27": 0.35, "NF1": 0.25,
    }
    tx_targeted = [rng.random() < targeted_rates.get(gene, 0.30) for _ in range(n)]

    return {
        "gene":             gene,
        "locus":            locus,
        "inheritance":      inh,
        "protein_size":     gene_def["protein_size"],
        "n":                n,
        "avg_age_dx":       round(sum(ages) / n, 1),
        "sporadic_n":       sporadic_n,
        "endemic_n":        endemic_n,
        "immunodef_n":      immunodef_n,
        "ebv_pos_n":        sum(ebv_pos),
        "cns_inv_n":        sum(cns_inv),
        "hlh_n":            sum(hlh_trigger),
        "hsct_n":           sum(hsct),
        "cr_n":             sum(cr),
        "os2_n":            sum(os2),
        "tx_rcodox":        sum(tx_rcodox),
        "tx_venetoclax":    sum(tx_venetoclax),
        "tx_targeted":      sum(tx_targeted),
        "key_avoid":        gene_def["key_avoid"],
        "key_rule":         gene_def["key_rule"],
        "targeted_rx":      gene_def["targeted_rx"],
        "bl_risk":          gene_def["bl_risk"],
        "pathognomonic":    gene_def["pathognomonic"],
        "surveillance":     gene_def["surveillance"],
    }


def generate_overview() -> dict:
    cohorts = {
        gd["gene"]: _make_cohort(gd, SEED_BASE + i)
        for i, gd in enumerate(ATLAS_GENES)
    }
    total = 320
    cr_total        = sum(c["cr_n"]        for c in cohorts.values())
    os2_total       = sum(c["os2_n"]       for c in cohorts.values())
    ebv_total       = sum(c["ebv_pos_n"]   for c in cohorts.values())
    hlh_total       = sum(c["hlh_n"]       for c in cohorts.values())
    hsct_total      = sum(c["hsct_n"]      for c in cohorts.values())
    cns_total       = sum(c["cns_inv_n"]   for c in cohorts.values())
    immunodef_t     = sum(c["immunodef_n"] for c in cohorts.values())
    rcodox_t        = sum(c["tx_rcodox"]   for c in cohorts.values())
    targeted_t      = sum(c["tx_targeted"] for c in cohorts.values())
    venetoclax_t    = sum(c["tx_venetoclax"] for c in cohorts.values())

    return {
        "atlas":                "Hereditary-BL-Predisposition-Atlas",
        "seed_range":           f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":       total,
        "cr_n":                 cr_total,
        "cr_pct":               round(cr_total / total * 100, 1),
        "os2_n":                os2_total,
        "os2_pct":              round(os2_total / total * 100, 1),
        "ebv_pos_n":            ebv_total,
        "ebv_pos_pct":          round(ebv_total / total * 100, 1),
        "hlh_n":                hlh_total,
        "hlh_pct":              round(hlh_total / total * 100, 1),
        "hsct_n":               hsct_total,
        "hsct_pct":             round(hsct_total / total * 100, 1),
        "cns_inv_n":            cns_total,
        "cns_inv_pct":          round(cns_total / total * 100, 1),
        "immunodef_n":          immunodef_t,
        "immunodef_pct":        round(immunodef_t / total * 100, 1),
        "rcodox_n":             rcodox_t,
        "rcodox_pct":           round(rcodox_t / total * 100, 1),
        "targeted_n":           targeted_t,
        "targeted_pct":         round(targeted_t / total * 100, 1),
        "venetoclax_n":         venetoclax_t,
        "venetoclax_pct":       round(venetoclax_t / total * 100, 1),
        "gene_counts": {
            gene: {
                "n":            c["n"],
                "cr_n":         c["cr_n"],
                "os2_n":        c["os2_n"],
                "ebv_pos_n":    c["ebv_pos_n"],
                "hlh_n":        c["hlh_n"],
                "hsct_n":       c["hsct_n"],
                "cns_inv_n":    c["cns_inv_n"],
                "immunodef_n":  c["immunodef_n"],
                "tx_rcodox":    c["tx_rcodox"],
                "tx_targeted":  c["tx_targeted"],
                "bl_risk":      c["bl_risk"],
            }
            for gene, c in cohorts.items()
        },
    }


def generate_breakdown() -> dict:
    return {
        "atlas":     "Hereditary-BL-Predisposition-Atlas",
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
            f"BL RISK: {gd['bl_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-BL-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 {gd["gene"]: gd["protein_size"] for gd in ATLAS_GENES},
        "key_clinical_distinctions": KEY_CLINICAL_DISTINCTIONS,
    }
