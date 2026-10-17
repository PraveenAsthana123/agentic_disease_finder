#!/usr/bin/env python3
"""Hereditary-PTCL-Predisposition-Atlas -- Complete 8-Gene Reference
Peripheral T-Cell Lymphoma (PTCL) germline predisposition.

ATM      (ATM kinase; 3056aa; 11q22.3; AD LOF;
          Ataxia-Telangiectasia (OMIM 208900);
          biallelic A-T: T-cell lymphoma 75x HIGHEST;
          monoallelic PTCL 3-5x; RADIOSENSITIVITY ABSOLUTE;
          TBI conditioning CI; BV-CHP preferred; ceralasertib ATRi;
          seed SEED_BASE+0).
TP53     (p53; 393aa; 17p13.1; AD LOF;
          Li-Fraumeni Syndrome (OMIM 151623);
          LFS PTCL 3-4x; complex karyotype 50-60%;
          AVOID RADIATION ABSOLUTELY; WB-MRI Toronto annually;
          venetoclax backbone; pralatrexate FOLR1;
          seed SEED_BASE+1).
CD27     (CD27; 260aa; 12p13.31; AR LOF;
          CD27 Deficiency (OMIM 615122);
          EBV+ T/NK lymphoproliferative disease HIGHEST 60-70x;
          HLH trigger; HSCT curative MANDATORY;
          Rituximab NOT appropriate (T-cell disease);
          seed SEED_BASE+2).
ITK      (ITK; 620aa; 5q33.3; AR LOF;
          ITK Deficiency (OMIM 613496);
          EBV-associated CD4+ T-cell LPD 90%+; biallelic LOF;
          HSCT before EBV exposure MANDATORY;
          EBV-IGIV bridge; Rituximab INEFFECTIVE;
          HLH 50%; seed SEED_BASE+3).
GATA2    (GATA2; 480aa; 3q21.3; AD LOF;
          GATA2 Deficiency/Emberger Syndrome (OMIM 137295);
          NK-cell lymphopenia → NK/T-cell lymphoma predisposition;
          lymphedema PATHOGNOMONIC Emberger; MonoMAC syndrome;
          HSCT curative; EBV viral load Q3M mandatory;
          seed SEED_BASE+4).
TET2     (TET2; 2002aa; 4q24; AD LOF;
          Germline TET2/CHIP;
          AITL 50-80% somatic co-mutation; germline PTCL 3-5x;
          hypomethylation; azacitidine/decitabine sensitive;
          IDH2 R172K co-mutation test mandatory; enasidenib if co-IDH2;
          seed SEED_BASE+5).
STAT3    (STAT3; 770aa; 17q21.2; AD GOF;
          STAT3 GOF Disease (OMIM 615952);
          T-LGL Leukemia; cytopenia HALLMARK;
          ruxolitinib JAKi FDA2023; cyclosporine secondline;
          PTCL via chronic JAK-STAT activation; methotrexate AVOID LGL;
          seed SEED_BASE+6).
PIK3CD   (PI3K-delta; 1044aa; 1p36.22; AD GOF;
          APDS1/PASLI (OMIM 615513);
          PI3Kδ GOF; EBV/CMV T-cell lymphoma 10-15%;
          leniolisib FDA2023; acyclovir MANDATORY prophylaxis;
          sirolimus bridge; live vaccines CI;
          CMV/EBV viral load Q3M; seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3622-3629)
"""
import random

SEED_BASE = 3622

ATLAS_GENES = [
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD LOF -- 3056aa -- "
            "ATM-350kDa-PIK3-Like-Kinase-"
            "DDR-Sensor-MRN-Complex-H2AX-BRCA1-CHK2-p53-"
            "PTCL-3-5x-Monoallelic-75x-Biallelic-A-T-RADIOSENSITIVITY-ABSOLUTE-"
            "A-T-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / ~350 kDa / 11q22.3 ATM PTCL molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; PIKK-family Ser/Thr kinase; activated by DSBs via MRN complex; "
            "  HEAT repeats (aa 1-3000): scaffolding; FAT domain; kinase domain; FATC domain; "
            "  Key substrates: H2AX (γH2AX foci), p53, BRCA1, CHK2, 53BP1, SMC1; "
            "  Biallelic loss (A-T): combined immunodeficiency + cerebellar ataxia + telangiectasia; "
            "  Monoallelic heterozygous: PTCL 3-5x; NHL 4-5x; "
            "PTCL MECHANISM: "
            "  ATM LOF impairs TCR rearrangement surveillance → T-cell clonal expansion; "
            "  Biallelic A-T: T-cell lymphoma 75x HIGHEST — first lymphoma often T-cell; "
            "  Monoallelic ATM: RADIOSENSITIVITY reduced (haploinsufficiency); "
            "  AITL 35%, PTCL-NOS 30%, NK/T-cell 15%, ALCL-ALK- 20% subtype distribution; "
            "CLINICAL RULES FOR ATM-PTCL: "
            "  RADIOSENSITIVITY ABSOLUTE — TBI conditioning ABSOLUTELY CI; "
            "  Use reduced-intensity conditioning (RIC) for HSCT; "
            "  BV-CHP (brentuximab+CHP) preferred over CHOP (omit vincristine neuropathy); "
            "  Ceralasertib (AZD6738, ATRi): synthetic lethality with ATM LOF; "
            "  EBV viral load monitoring Q3M (T-cell immune surveillance impaired); "
            "  Cascade ATM testing: first-degree family members annually CBC + IgA/IgG; "
            "KEY VARIANTS: "
            "  Monoallelic: c.7271T>G (p.Val2424Gly), c.1009C>T (p.Arg337Cys), del11q22 large; "
            "  Biallelic: truncating + truncating (A-T full phenotype); "
            "SURVEILLANCE: "
            "  Annual lymph node exam + CBC; EBV viral load Q3M; "
            "  Biallelic: neurological surveillance; IgA level; chest CT pulmonary atelectasis; "
            "TARGETED Rx: "
            "  Ceralasertib (ATRi, AstraZeneca): phase II PTCL relapsed; "
            "  BV-CHP: brentuximab vedotin+cyclophosphamide+doxorubicin+prednisone; "
            "  Romidepsin HDAC inhibitor (FDA approved PTCL); "
        ),
        "syndrome": "Ataxia-Telangiectasia (A-T) / ATM-PTCL",
        "inheritance": "AD LOF heterozygous (monoallelic); AR biallelic = A-T",
        "ptcl_risk": "Monoallelic: PTCL 3-5x; Biallelic A-T: T-cell lymphoma 75x HIGHEST; AITL 35%, PTCL-NOS 30%",
        "pathognomonic": "IgA deficiency PATHOGNOMONIC A-T biallelic; cerebellar ataxia + telangiectasia biallelic",
        "key_rule": "RADIOSENSITIVITY ABSOLUTE — TBI conditioning ABSOLUTELY CI; RIC-HSCT only; BV-CHP preferred; ceralasertib ATRi",
        "key_avoid": "TBI in HSCT conditioning ABSOLUTELY CI; full-dose RT biallelic; even monoallelic: dose-reduce RT",
        "surveillance": "Annual CBC + IgA; EBV viral load Q3M; lymph node exam; cascade family ATM testing",
        "targeted_rx": "Ceralasertib ATRi (synthetic lethality ATM LOF); BV-CHP over CHOP; romidepsin HDACi PTCL",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "Tetramer-DBD-OD-TAD-Pro-"
            "PTCL-3-4x-LFS-AVOID-RADIATION-ABSOLUTELY-WB-MRI-Toronto-"
            "LFS-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 PTCL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; transcription factor homotetrameric; "
            "  TAD1 (aa 1-40) + TAD2 (aa 40-67): transactivation; "
            "  Proline-rich (aa 67-98): apoptosis modulation; "
            "  DBD (aa 102-292): DNA binding; hotspot mutations R175H/G245S/R248W/R248Q/R249S/R273H/R282W; "
            "  OD (aa 323-356): tetramerisation; "
            "  Regulatory (aa 364-393): nuclear localisation; "
            "  LFS: AD germline TP53 LOF → childhood/young adult cancers (sarcoma, breast, brain, PTCL, adrenal); "
            "PTCL MECHANISM: "
            "  TP53 LOF → T-cell escape from apoptosis; complex karyotype 50-60% in TP53-PTCL; "
            "  PTCL 3-4x in LFS germline TP53; PTCL-NOS 45% most common subtype; "
            "  AITL 25% with epigenetic co-driver TET2/IDH2 somatic hits; "
            "CLINICAL RULES FOR TP53-PTCL: "
            "  AVOID RADIATION ABSOLUTELY (LFS): ANY radiation → secondary sarcoma/cancer; "
            "  WB-MRI Toronto Protocol ANNUALLY (whole-body MRI surveillance); "
            "  Venetoclax backbone: p53-independent apoptosis; "
            "  Pralatrexate (FOLR1-targeted): FDA-approved PTCL-NOS; "
            "  Cascade TP53 testing mandatory in LFS families; "
            "KEY VARIANTS: "
            "  R248W/R248Q (hotspot DBD); R175H, G245S, R249S, R273H, R282W; "
            "  Splice site IVS6+1G>A; del17p13.1 large deletion; "
            "SURVEILLANCE: "
            "  WB-MRI annually (Toronto Protocol); annual CBC; avoid all CT imaging (cumulative radiation LFS); "
            "TARGETED Rx: "
            "  Venetoclax (BCL-2i): p53-independent apoptosis; "
            "  Pralatrexate (FOLR1): FDA-approved relapsed PTCL; "
            "  Romidepsin HDACi: FDA-approved PTCL; "
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) / TP53-PTCL",
        "inheritance": "AD LOF heterozygous",
        "ptcl_risk": "PTCL 3-4x; LFS PTCL-NOS 45%, AITL 25%; complex karyotype 50-60%; transformation risk high",
        "pathognomonic": "LFS sarcoma+breast+brain+PTCL cluster; complex karyotype PTCL; secondary PTCL after prior RT",
        "key_rule": "AVOID RADIATION ABSOLUTELY; WB-MRI Toronto Protocol annually; venetoclax backbone; pralatrexate FOLR1",
        "key_avoid": "ANY radiation (even palliative RT); CT imaging (cumulative dose LFS); RT consolidation",
        "surveillance": "WB-MRI annually (Toronto Protocol); annual CBC; avoid CT; cascade TP53 testing",
        "targeted_rx": "Venetoclax BCL-2i backbone; pralatrexate FOLR1 FDA-approved; romidepsin HDACi FDA-approved PTCL",
    },
    {
        "gene": "CD27",
        "protein": (
            "CD27 -- 12p13.31 AR LOF -- 260aa -- "
            "CD27-29kDa-TNF-Receptor-TNFRSF7-"
            "T-B-NK-Costimulation-EBV-Control-"
            "EBV-T-NK-LPD-60-70x-HIGHEST-HLH-HSCT-MANDATORY-"
            "CD27-Deficiency-OMIM-615122"
        ),
        "locus": "12p13.31",
        "protein_size": (
            "260 aa / ~29 kDa / 12p13.31 CD27 PTCL molecular context: "
            "STRUCTURE: "
            "  260 aa / ~29 kDa; TNF receptor superfamily member 7 (TNFRSF7); "
            "  Cysteine-rich domain (CRD, aa 1-90): CD70 ligand binding; "
            "  Transmembrane domain (aa 191-211): type I membrane protein; "
            "  Cytoplasmic domain (aa 212-260): TRAF2/TRAF5 signalling; "
            "  CD27-CD70 axis: T-cell and NK-cell costimulation; critical for EBV immune control; "
            "  Biallelic LOF: loss of T/NK costimulation → EBV uncontrolled replication; "
            "CD27 DEFICIENCY PHENOTYPE: "
            "  Biallelic LOF: combined immunodeficiency — progressive hypogammaglobulinaemia; "
            "  EBV+ T/NK lymphoproliferative disease HIGHEST 60-70x; "
            "  HLH (haemophagocytic lymphohistiocytosis) trigger 45%; "
            "  NK/T-cell lymphoma 60%, EBV+PTCL 30%; "
            "CLINICAL RULES FOR CD27-PTCL: "
            "  RITUXIMAB NOT APPROPRIATE — this is T-cell/NK-cell disease (CD20-negative T-cells); "
            "  HSCT curative MANDATORY before progressive EBV lymphoproliferation; "
            "  EBV-IGIV (high-titre EBV immunoglobulin) bridge pre-HSCT; "
            "  HLH protocol: dexamethasone + etoposide (HLH-94/2004); "
            "  HSCT: use RIC (reduced intensity); no TBI biallelic; "
            "KEY VARIANTS: "
            "  Biallelic truncating LOF; missense in CRD impairing CD70 binding; "
            "  No single founder — private family-specific LOF variants; "
            "SURVEILLANCE: "
            "  EBV viral load Q3M MANDATORY; CBC quarterly; immunoglobulin levels monthly; "
            "  Lymph node PET/CT q6M from diagnosis; "
            "TARGETED Rx: "
            "  RIC-HSCT curative (biallelic CD27 deficiency); "
            "  EBV-IGIV bridge; etoposide HLH trigger; "
            "  Pembrolizumab AVOID (T-cell disease — risk immune exacerbation); "
        ),
        "syndrome": "CD27 Deficiency (Immunodeficiency) / CD27-PTCL",
        "inheritance": "AR biallelic LOF",
        "ptcl_risk": "EBV+ T/NK LPD HIGHEST 60-70x biallelic; NK/T-cell lymphoma 60%; HLH trigger 45%; HSCT mandatory",
        "pathognomonic": "EBV-driven T/NK lymphoproliferation PATHOGNOMONIC biallelic; HLH + T-cell lymphoma combination",
        "key_rule": "RITUXIMAB NOT APPROPRIATE (T-cell disease CD20-); HSCT MANDATORY before progression; EBV-IGIV bridge",
        "key_avoid": "Rituximab (T-cell CD20-negative disease — completely inappropriate); delay HSCT",
        "surveillance": "EBV viral load Q3M MANDATORY; CBC quarterly; Ig levels monthly; PET/CT q6M",
        "targeted_rx": "RIC-HSCT curative; EBV-IGIV bridge pre-HSCT; etoposide HLH; romidepsin PTCL salvage",
    },
    {
        "gene": "ITK",
        "protein": (
            "ITK -- 5q33.3 AR LOF -- 620aa -- "
            "ITK-72kDa-Tec-Kinase-"
            "PH-TH-SH3-SH2-Kinase-TCR-PLCg1-"
            "EBV-CD4-T-LPD-90pct-HSCT-MANDATORY-HLH-50pct-"
            "ITK-Deficiency-OMIM-613496"
        ),
        "locus": "5q33.3",
        "protein_size": (
            "620 aa / ~72 kDa / 5q33.3 ITK PTCL molecular context: "
            "STRUCTURE: "
            "  620 aa / ~72 kDa; Tec-family non-receptor tyrosine kinase; "
            "  PH domain (aa 1-100): membrane recruitment (PIP3 binding); "
            "  TH domain (aa 100-165): zinc-coordinated; "
            "  SH3 domain (aa 165-225): proline-rich motif binding; "
            "  SH2 domain (aa 225-330): phospho-Tyr binding; "
            "  Kinase domain (aa 330-620): catalytic; activates PLCγ1; "
            "  ITK activates PLCγ1 → TCR → IP3/DAG → NFAT → T-cell activation; "
            "  Biallelic ITK LOF: absent TCR-downstream signalling → CD4+ EBV uncontrolled; "
            "ITK DEFICIENCY PHENOTYPE: "
            "  Biallelic LOF: EBV-associated CD4+ T-cell lymphoproliferative disease 90%+; "
            "  HLH in 50% of affected patients; "
            "  EBV+CD4+T-LPD 70%, PTCL-NOS 20%, AITL 10%; "
            "CLINICAL RULES FOR ITK-PTCL: "
            "  HSCT MANDATORY — before EBV exposure or at earliest diagnosis; "
            "  Rituximab INEFFECTIVE — T-cell disease (CD20-); "
            "  EBV-IGIV bridge until HSCT; "
            "  HLH protocol: dexamethasone + etoposide; "
            "  Avoid calcineurin inhibitors (cyclosporine) — ITK downstream pathway; "
            "KEY VARIANTS: "
            "  Biallelic truncating ITK LOF; frameshift + missense; "
            "  PH domain mutations: impaired membrane recruitment; "
            "SURVEILLANCE: "
            "  EBV viral load Q3M MANDATORY; CBC quarterly; CD4+/CD8+ counts monthly; "
            "  Lymph node exam q3M; PET/CT at diagnosis and q6M; "
            "TARGETED Rx: "
            "  RIC-HSCT curative (biallelic ITK deficiency — MANDATORY before EBV); "
            "  EBV-IGIV bridge; etoposide HLH trigger; "
            "  Romidepsin HDACi salvage post-HSCT if relapse; "
        ),
        "syndrome": "ITK Deficiency / EBV-associated CD4+ T-cell LPD / ITK-PTCL",
        "inheritance": "AR biallelic LOF",
        "ptcl_risk": "EBV-associated CD4+ T-cell LPD 90%+; HLH 50%; EBV+T-LPD 70%, PTCL-NOS 20%; HSCT mandatory",
        "pathognomonic": "EBV-driven CD4+ T-LPD PATHOGNOMONIC biallelic ITK; HLH co-occurrence 50%",
        "key_rule": "HSCT MANDATORY (curative, do early); RITUXIMAB INEFFECTIVE (T-cell CD20-); EBV-IGIV bridge",
        "key_avoid": "Rituximab (CD20- T-cell — completely wrong); calcineurin inhibitors; delay HSCT",
        "surveillance": "EBV viral load Q3M MANDATORY; CBC quarterly; CD4+/CD8+ monthly; PET/CT at dx + q6M",
        "targeted_rx": "RIC-HSCT MANDATORY curative; EBV-IGIV bridge; etoposide HLH; romidepsin salvage",
    },
    {
        "gene": "GATA2",
        "protein": (
            "GATA2 -- 3q21.3 AD LOF -- 480aa -- "
            "GATA2-50kDa-Zinc-Finger-Transcription-Factor-"
            "NK-Lymphopenia-Emberger-MonoMAC-"
            "NK-T-Cell-Lymphoma-Predisposition-HSCT-Curative-EBV-Q3M-"
            "GATA2-Deficiency-OMIM-137295"
        ),
        "locus": "3q21.3",
        "protein_size": (
            "480 aa / ~50 kDa / 3q21.3 GATA2 PTCL molecular context: "
            "STRUCTURE: "
            "  480 aa / ~50 kDa; dual zinc-finger transcription factor; "
            "  N-terminal zinc finger (NZF, aa 200-240): protein-protein interaction; "
            "  C-terminal zinc finger (CZF, aa 317-358): DNA binding (WGATAR motif); "
            "  Transactivation domain (aa 1-200): cofactor recruitment; "
            "  GATA2 is master regulator of haematopoietic/vascular development; "
            "  LOF → NK-cell lymphopenia, monocytopenia, B-lymphopenia (MonoMAC syndrome); "
            "GATA2 DEFICIENCY PHENOTYPE (Emberger/MonoMAC): "
            "  NK-cell lymphopenia → defective EBV/CMV NK surveillance; "
            "  NK/T-cell lymphoma predisposition 50% NK/T, 30% EBV+PTCL; "
            "  Lymphedema (unilateral/bilateral) PATHOGNOMONIC — Emberger syndrome; "
            "  MonoMAC: monocytopenia + mycobacterial + HPV infections; "
            "  MDS/AML co-occurrence 75% (haematopoietic failure); "
            "CLINICAL RULES FOR GATA2-PTCL: "
            "  HSCT curative — should proceed before haematopoietic decompensation; "
            "  EBV viral load Q3M MANDATORY (NK surveillance absent); "
            "  CMV viral load Q3M monitoring; "
            "  Asparaginase-based NK/T regimen (SMILE/AspaMetDex); "
            "  Lymphedema management: compression; lymph node dissection risks; "
            "KEY VARIANTS: "
            "  p.Thr354Met (CZF DNA-binding): most common Emberger variant; "
            "  p.Arg396Gln; c.1061C>T; del3q21 large; "
            "SURVEILLANCE: "
            "  EBV/CMV viral load Q3M; CBC quarterly; NK-cell counts quarterly; "
            "  Lymphedema monitoring; MDS surveillance BM aspirate q2yr; "
            "TARGETED Rx: "
            "  RIC-HSCT curative (GATA2 deficiency — restores NK haematopoiesis); "
            "  SMILE (Steroid+Methotrexate+Ifosfamide+L-asparaginase+Etoposide) for NK/T; "
            "  AspaMetDex for NK/T-cell lymphoma; "
        ),
        "syndrome": "GATA2 Deficiency / Emberger Syndrome / MonoMAC / GATA2-PTCL",
        "inheritance": "AD LOF heterozygous",
        "ptcl_risk": "NK/T-cell lymphoma 50%; EBV+PTCL 30%; PTCL-NOS 20%; NK-cell lymphopenia predisposes; MDS/AML 75%",
        "pathognomonic": "Lymphedema PATHOGNOMONIC Emberger (unilateral); MonoMAC monocytopenia + mycobacteria + NK-lymphopenia",
        "key_rule": "HSCT curative (before decompensation); EBV/CMV viral load Q3M MANDATORY; SMILE/AspaMetDex NK/T regimen",
        "key_avoid": "Delay HSCT past haematopoietic decompensation; anthracycline-only regimen for NK/T (use asparaginase)",
        "surveillance": "EBV/CMV Q3M; CBC quarterly; NK-cell counts; BM aspirate q2yr MDS; lymphedema monitoring",
        "targeted_rx": "RIC-HSCT curative; SMILE NK/T lymphoma; AspaMetDex NK/T; romidepsin PTCL salvage",
    },
    {
        "gene": "TET2",
        "protein": (
            "TET2 -- 4q24 AD LOF -- 2002aa -- "
            "TET2-223kDa-5mC-Dioxygenase-"
            "CXXC-Catalytic-5hmC-5fC-5caC-"
            "AITL-50-80pct-Germline-PTCL-3-5x-Azacitidine-IDH2-Enasidenib-"
            "Germline-TET2-CHIP"
        ),
        "locus": "4q24",
        "protein_size": (
            "2002 aa / ~223 kDa / 4q24 TET2 PTCL molecular context: "
            "STRUCTURE: "
            "  2002 aa / ~223 kDa; Ten-eleven translocation (TET) family dioxygenase; "
            "  CXXC domain (aa 1130-1185): CpG-rich DNA binding; "
            "  Catalytic domain (aa 1290-1905): Fe(II)- and α-ketoglutarate-dependent dioxygenase; "
            "  DSBH (double-stranded β-helix) core; low-complexity N-terminus (aa 1-1100); "
            "  TET2 converts 5-methylcytosine (5mC) → 5hmC → 5fC → 5caC → DNA demethylation; "
            "  LOF → hypermethylation → silencing of tumour suppressors; "
            "TET2 PTCL CONTEXT: "
            "  AITL (angioimmunoblastic T-cell lymphoma): somatic TET2 mutation 50-80%; "
            "  Germline TET2 LOF (CHIP-associated): PTCL predisposition 3-5x; "
            "  IDH2 R172K somatic co-mutation in AITL 20-30%: test MANDATORY; "
            "  Azacitidine/decitabine (HMA): TET2-LOF sensitive (hypomethylation re-induction); "
            "CLINICAL RULES FOR TET2-PTCL: "
            "  IDH2 R172K co-mutation test MANDATORY — enasidenib (IDH2i FDA2017) if positive; "
            "  Azacitidine or decitabine (HMA): TET2-LOF sensitive; "
            "  AITL regimen: CHOP or CHOEP; romidepsin + azacitidine combination; "
            "  PI3K/mTOR pathway co-activated — consider idelalisib/copanlisib; "
            "KEY VARIANTS: "
            "  Somatic: frameshift/nonsense widely distributed; "
            "  Germline TET2 (CHIP-associated): hypomorphic missense + truncating; "
            "SURVEILLANCE: "
            "  CBC annually (CHIP surveillance); PTCL symptoms surveillance; "
            "  IDH2 status on diagnostic biopsy; "
            "TARGETED Rx: "
            "  Azacitidine HMA: TET2-sensitive demethylation; "
            "  Enasidenib (IDH2i): if co-IDH2 R172K positive; "
            "  Romidepsin + azacitidine (clinical trial); "
        ),
        "syndrome": "Germline TET2 / CHIP-PTCL / AITL Predisposition",
        "inheritance": "AD LOF heterozygous (CHIP-associated germline)",
        "ptcl_risk": "AITL 65% subtype; germline PTCL 3-5x; IDH2 R172K co-mutation 20-30% AITL; HMA sensitive",
        "pathognomonic": "AITL + TET2 + IDH2 co-mutation triple: PATHOGNOMONIC epigenetic PTCL; HMA-sensitive hypermethylation",
        "key_rule": "IDH2 R172K co-mutation test MANDATORY; enasidenib if co-IDH2+; azacitidine/decitabine HMA for TET2-LOF",
        "key_avoid": "Omitting IDH2 co-mutation test; missing enasidenib opportunity; CHOP alone without epigenetic agent",
        "surveillance": "Annual CBC (CHIP); IDH2 status on biopsy; TET2 germline cascade testing",
        "targeted_rx": "Azacitidine HMA TET2-sensitive; enasidenib IDH2i if co-IDH2; romidepsin+azacitidine combination",
    },
    {
        "gene": "STAT3",
        "protein": (
            "STAT3 -- 17q21.2 AD GOF -- 770aa -- "
            "STAT3-92kDa-JAK-STAT-Transcription-Factor-"
            "SH2-DBD-Coiled-Coil-"
            "T-LGL-Leukemia-Cytopenia-HALLMARK-Ruxolitinib-JAKi-FDA2023-"
            "STAT3-GOF-OMIM-615952"
        ),
        "locus": "17q21.2",
        "protein_size": (
            "770 aa / ~92 kDa / 17q21.2 STAT3 PTCL molecular context: "
            "STRUCTURE: "
            "  770 aa / ~92 kDa; Signal Transducer and Activator of Transcription 3; "
            "  N-terminal domain (aa 1-120): oligomerisation; "
            "  Coiled-coil domain (aa 120-315): ISGF3 assembly; "
            "  DNA-binding domain (DBD, aa 315-490): GAS element recognition; "
            "  Linker domain (aa 490-575); "
            "  SH2 domain (aa 575-680): JAK2/TYK2 pY-binding; dimerisation; "
            "  Transactivation domain (aa 680-770): Tyr705 phosphorylation site; "
            "  GOF variants: SH2 domain (Y640F/D661Y/D661V/N647I): constitutive STAT3 activation; "
            "STAT3 GOF DISEASE / T-LGL: "
            "  T-LGL (T-Large Granular Lymphocyte) Leukemia: somatic STAT3 GOF 40% T-LGL; "
            "  Germline STAT3 GOF: multi-organ autoimmunity + T-LGL + PTCL predisposition; "
            "  Cytopenia HALLMARK: neutropenia 65%, anaemia 50%, thrombocytopenia 40%; "
            "  Chronic JAK-STAT activation → clonal T-cell expansion → PTCL; "
            "  T-LGL 55%, PTCL-NOS 25%, AITL 20% subtype distribution; "
            "CLINICAL RULES FOR STAT3-PTCL: "
            "  Ruxolitinib (JAK1/2i, FDA2023 T-LGL label): first-line cytopenia treatment; "
            "  Cyclosporine: second-line immunosuppression T-LGL; "
            "  METHOTREXATE AVOID in T-LGL (cytopaenia worsens); "
            "  G-CSF for severe neutropenia (bridging therapy); "
            "  PTCL: CHOP/CHOEP + romidepsin; "
            "KEY VARIANTS: "
            "  SH2 GOF: Y640F (most common ~50% T-LGL); D661Y; D661V; N647I; "
            "  Germline GOF: autoimmunity + lymphoma penetrance; "
            "SURVEILLANCE: "
            "  CBC monthly (cytopenia monitoring); LGL count quarterly; "
            "  Annual CT/PET for PTCL surveillance in STAT3-GOF germline; "
            "TARGETED Rx: "
            "  Ruxolitinib JAKi (FDA 2023 T-LGL): first-line for cytopenia; "
            "  Cyclosporine second-line T-LGL; "
            "  Romidepsin HDACi: FDA-approved PTCL; "
        ),
        "syndrome": "STAT3 GOF Disease / T-LGL Leukemia / PTCL Predisposition",
        "inheritance": "AD GOF heterozygous",
        "ptcl_risk": "T-LGL 55%; PTCL-NOS 25%; AITL 20%; cytopenia HALLMARK (neutropenia 65%, anaemia 50%); JAK-STAT chronic",
        "pathognomonic": "T-LGL + cytopenia + STAT3 SH2 GOF Y640F PATHOGNOMONIC; neutropenia 65% + thrombocytopenia 40%",
        "key_rule": "Ruxolitinib JAKi FDA2023 first-line T-LGL; METHOTREXATE AVOID LGL (worsens cytopaenia); G-CSF bridge",
        "key_avoid": "Methotrexate in T-LGL (cytopaenia WORSENS); alkylating agents without cytopaenia recovery first",
        "surveillance": "CBC monthly cytopaenia; LGL count quarterly; annual CT/PET PTCL surveillance; cascade STAT3 testing",
        "targeted_rx": "Ruxolitinib JAKi FDA2023 T-LGL; cyclosporine second-line; romidepsin HDACi FDA-approved PTCL",
    },
    {
        "gene": "PIK3CD",
        "protein": (
            "PIK3CD -- 1p36.22 AD GOF -- 1044aa -- "
            "PI3K-delta-119kDa-Class-IA-PI3K-"
            "ABD-RBD-C2-Helical-Kinase-"
            "APDS1-EBV-CMV-T-Cell-Lymphoma-10-15pct-Leniolisib-FDA2023-"
            "APDS1-OMIM-615513"
        ),
        "locus": "1p36.22",
        "protein_size": (
            "1044 aa / ~119 kDa / 1p36.22 PIK3CD PTCL molecular context: "
            "STRUCTURE: "
            "  1044 aa / ~119 kDa; PI3K catalytic subunit delta (p110δ); Class IA PI3K; "
            "  ABD domain (aa 1-108): adaptor-binding (p85 regulatory subunit); "
            "  RBD domain (aa 190-291): RAS-binding; "
            "  C2 domain (aa 358-480): membrane binding; "
            "  Helical domain (aa 481-687): p85 nSH2 inhibition interface; "
            "  Kinase domain (aa 726-1044): catalytic; GOF variants E1021K/E525K; "
            "  PI3Kδ GOF → hyperactivation PIP3 → AKT/mTOR → T-cell senescence/apoptosis resistance; "
            "APDS1 PHENOTYPE: "
            "  APDS1 (Activated PI3K Delta Syndrome type 1): AD GOF PIK3CD; "
            "  EBV/CMV lymphoproliferative T-cell lymphoma 10-15%; "
            "  Nodular lymphoid hyperplasia → EBV+PTCL 50%, PTCL-NOS 30%, NK/T 20%; "
            "  Recurrent respiratory infections, splenomegaly, lymphadenopathy; "
            "CLINICAL RULES FOR PIK3CD-PTCL: "
            "  Leniolisib (PI3Kδ inhibitor, FDA Feb 2023): first targeted APDS1 therapy; "
            "  Acyclovir prophylaxis MANDATORY (EBV/VZV antiviral coverage); "
            "  CMV/EBV viral load Q3M MANDATORY; "
            "  Sirolimus (mTOR): bridge before leniolisib or if unavailable; "
            "  Live vaccines ABSOLUTELY CI (risk live-attenuated replication); "
            "  IVIG replacement if hypogammaglobulinaemia; "
            "KEY VARIANTS: "
            "  E1021K (c.3061G>A): most common APDS1 GOF (~65% European); "
            "  E525K (c.1573G>A): second most common; "
            "  Other kinase domain GOF missense variants; "
            "SURVEILLANCE: "
            "  EBV/CMV viral load Q3M MANDATORY; CBC quarterly; IgG levels q3M; "
            "  Annual CT scan (lymphadenopathy + splenomegaly); "
            "TARGETED Rx: "
            "  Leniolisib (PI3Kδi, FDA2023): first-line APDS1; "
            "  Sirolimus mTORi bridge; "
            "  Acyclovir MANDATORY prophylaxis; IVIG replacement; "
        ),
        "syndrome": "APDS1 / PASLI / Activated PI3Kδ Syndrome type 1 / PIK3CD-PTCL",
        "inheritance": "AD GOF heterozygous",
        "ptcl_risk": "EBV+T-LPD 50%; PTCL-NOS 30%; NK/T-cell 20%; EBV/CMV lymphoproliferative T-cell lymphoma 10-15%",
        "pathognomonic": "APDS1: recurrent infections + splenomegaly + nodular lymphoid hyperplasia + EBV+PTCL in young",
        "key_rule": "Leniolisib FDA2023 first targeted therapy; acyclovir MANDATORY prophylaxis; EBV/CMV Q3M; live vaccines CI",
        "key_avoid": "Live vaccines ABSOLUTELY CI (risk replication); delay leniolisib (use sirolimus bridge if unavailable)",
        "surveillance": "EBV/CMV Q3M MANDATORY; CBC quarterly; IgG q3M; annual CT lymphadenopathy; cascade PIK3CD testing",
        "targeted_rx": "Leniolisib PI3Kδi FDA2023; sirolimus mTORi bridge; acyclovir prophylaxis MANDATORY; IVIG replacement",
    },
]

KEY_CLINICAL_DISTINCTIONS = [
    "ATM-BIALLELIC-A-T-PTCL-75x-HIGHEST: Biallelic A-T = T-cell lymphoma 75x — TBI conditioning ABSOLUTELY CI; RIC-HSCT only; BV-CHP preferred; ceralasertib ATRi synthetic lethality.",
    "TP53-LFS-AVOID-RADIATION-ABSOLUTELY: LFS germline TP53 — ANY radiation causes secondary sarcoma; WB-MRI Toronto Protocol annually; venetoclax backbone; pralatrexate FOLR1 FDA-approved.",
    "CD27-RITUXIMAB-NOT-APPROPRIATE: CD27 deficiency = EBV+ T/NK LPD 60-70x — Rituximab COMPLETELY WRONG (T-cell CD20-); HSCT MANDATORY; EBV-IGIV bridge; HLH etoposide.",
    "ITK-HSCT-MANDATORY-BEFORE-EBV: ITK deficiency = EBV-associated CD4+ T-LPD 90%+ — HSCT MANDATORY before EBV exposure; Rituximab INEFFECTIVE (T-cell); HLH 50%.",
    "GATA2-EMBERGER-NK-LYMPHOPENIA: Lymphedema PATHOGNOMONIC Emberger; NK-cell lymphopenia → NK/T-cell lymphoma; EBV/CMV Q3M MANDATORY; HSCT curative before decompensation.",
    "TET2-IDH2-CO-MUTATION-MANDATORY-TEST: AITL 65% TET2-PTCL; IDH2 R172K co-mutation test MANDATORY — enasidenib FDA2017 if co-IDH2+; azacitidine HMA for TET2-LOF PTCL.",
    "STAT3-GOF-METHOTREXATE-AVOID-LGL: T-LGL cytopenia HALLMARK — METHOTREXATE AVOID (worsens cytopaenia); ruxolitinib JAKi FDA2023 first-line; cyclosporine second-line T-LGL.",
    "PIK3CD-APDS1-LIVE-VACCINES-CI: APDS1 live vaccines ABSOLUTELY CI; acyclovir MANDATORY prophylaxis; leniolisib FDA2023 first targeted therapy; EBV/CMV viral load Q3M.",
]


def _make_cohort(gene_def: dict, seed: int) -> dict:
    """Generate a 40-patient PTCL cohort for one predisposition gene."""
    rng = random.Random(seed)

    gene  = gene_def["gene"]
    locus = gene_def["locus"]
    inh   = gene_def["inheritance"]

    n = 40

    # Average age at PTCL diagnosis (per-gene, Gaussian σ≈12, clamp 8-82)
    avg_ages = {
        "ATM": 35, "TP53": 42, "CD27": 22, "ITK": 18,
        "GATA2": 38, "TET2": 55, "STAT3": 58, "PIK3CD": 28,
    }
    mu = avg_ages.get(gene, 40)
    ages = sorted([rng.gauss(mu, 12) for _ in range(n)])
    ages = [max(8, min(82, round(a, 1))) for a in ages]

    # PTCL subtype distribution per gene
    subtype_dist = {
        "ATM":    [("AITL", 0.35), ("PTCL-NOS", 0.65), ("NK/T-cell", 0.80), ("ALCL-ALK-", 1.00)],
        "TP53":   [("PTCL-NOS", 0.45), ("AITL", 0.70), ("ALCL-ALK-", 0.90), ("NK/T-cell", 1.00)],
        "CD27":   [("NK/T-cell", 0.60), ("EBV+PTCL", 0.90), ("AITL", 1.00)],
        "ITK":    [("EBV+CD4+T-LPD", 0.70), ("PTCL-NOS", 0.90), ("AITL", 1.00)],
        "GATA2":  [("NK/T-cell", 0.50), ("EBV+PTCL", 0.80), ("PTCL-NOS", 1.00)],
        "TET2":   [("AITL", 0.65), ("PTCL-NOS", 0.90), ("NK/T-cell", 1.00)],
        "STAT3":  [("T-LGL", 0.55), ("PTCL-NOS", 0.80), ("AITL", 1.00)],
        "PIK3CD": [("EBV+T-LPD", 0.50), ("PTCL-NOS", 0.80), ("NK/T-cell", 1.00)],
    }
    def pick_subtype(gene_key):
        dist = subtype_dist.get(gene_key, [("PTCL-NOS", 0.60), ("AITL", 0.85), ("NK/T-cell", 1.00)])
        r = rng.random()
        for label, threshold in dist:
            if r < threshold:
                return label
        return dist[-1][0]

    subtypes = [pick_subtype(gene) for _ in range(n)]

    # Count dominant subtypes
    aitl_n   = sum(1 for s in subtypes if s == "AITL")
    nkt_n    = sum(1 for s in subtypes if "NK/T" in s or "NK/T-cell" in s)
    ebv_sub_n = sum(1 for s in subtypes if "EBV" in s or "T-LPD" in s)
    tlgl_n   = sum(1 for s in subtypes if s == "T-LGL")

    # EBV+ rate per gene
    ebv_rates = {
        "ATM": 0.20, "TP53": 0.15, "CD27": 0.85, "ITK": 0.90,
        "GATA2": 0.65, "TET2": 0.30, "STAT3": 0.20, "PIK3CD": 0.70,
    }
    ebv_pos = [rng.random() < ebv_rates.get(gene, 0.30) for _ in range(n)]

    # HLH trigger rate
    hlh_rates = {
        "ATM": 0.08, "TP53": 0.05, "CD27": 0.45, "ITK": 0.50,
        "GATA2": 0.35, "TET2": 0.08, "STAT3": 0.10, "PIK3CD": 0.30,
    }
    hlh_trigger = [rng.random() < hlh_rates.get(gene, 0.10) for _ in range(n)]

    # HSCT rate
    hsct_rates = {
        "ATM": 0.15, "TP53": 0.10, "CD27": 0.88, "ITK": 0.90,
        "GATA2": 0.85, "TET2": 0.18, "STAT3": 0.12, "PIK3CD": 0.40,
    }
    hsct = [rng.random() < hsct_rates.get(gene, 0.20) for _ in range(n)]

    # CR rate
    cr_rates = {
        "ATM": 0.52, "TP53": 0.45, "CD27": 0.68, "ITK": 0.65,
        "GATA2": 0.60, "TET2": 0.55, "STAT3": 0.58, "PIK3CD": 0.50,
    }
    cr = [rng.random() < cr_rates.get(gene, 0.55) for _ in range(n)]

    # 5yr OS rate
    os5_rates = {
        "ATM": 0.38, "TP53": 0.30, "CD27": 0.55, "ITK": 0.52,
        "GATA2": 0.48, "TET2": 0.42, "STAT3": 0.62, "PIK3CD": 0.40,
    }
    os5 = [rng.random() < os5_rates.get(gene, 0.42) for _ in range(n)]

    # Treatment tracking
    # tx_chop_choep: CHOP/CHOEP front-line
    chop_rates = {
        "ATM": 0.65, "TP53": 0.60, "CD27": 0.10, "ITK": 0.08,
        "GATA2": 0.12, "TET2": 0.70, "STAT3": 0.55, "PIK3CD": 0.35,
    }
    tx_chop_choep = [rng.random() < chop_rates.get(gene, 0.50) for _ in range(n)]

    # tx_brentuximab: brentuximab vedotin (for ALCL/ALK+, ATM-PTCL mainly)
    bv_rates = {
        "ATM": 0.22, "TP53": 0.18, "CD27": 0.05, "ITK": 0.05,
        "GATA2": 0.08, "TET2": 0.10, "STAT3": 0.08, "PIK3CD": 0.10,
    }
    tx_brentuximab = [rng.random() < bv_rates.get(gene, 0.10) for _ in range(n)]

    # tx_targeted: targeted agent by gene
    targeted_rates = {
        "ATM": 0.35, "TP53": 0.40, "CD27": 0.25, "ITK": 0.20,
        "GATA2": 0.45, "TET2": 0.55, "STAT3": 0.60, "PIK3CD": 0.55,
    }
    tx_targeted = [rng.random() < targeted_rates.get(gene, 0.35) for _ in range(n)]

    return {
        "gene":             gene,
        "locus":            locus,
        "inheritance":      inh,
        "protein_size":     gene_def["protein_size"],
        "n":                n,
        "avg_age_dx":       round(sum(ages) / n, 1),
        "aitl_n":           aitl_n,
        "nkt_n":            nkt_n,
        "ebv_sub_n":        ebv_sub_n,
        "tlgl_n":           tlgl_n,
        "ebv_pos_n":        sum(ebv_pos),
        "hlh_n":            sum(hlh_trigger),
        "hsct_n":           sum(hsct),
        "cr_n":             sum(cr),
        "os5_n":            sum(os5),
        "tx_chop_choep":    sum(tx_chop_choep),
        "tx_brentuximab":   sum(tx_brentuximab),
        "tx_targeted":      sum(tx_targeted),
        "key_avoid":        gene_def["key_avoid"],
        "key_rule":         gene_def["key_rule"],
        "targeted_rx":      gene_def["targeted_rx"],
        "ptcl_risk":        gene_def["ptcl_risk"],
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
    os5_total       = sum(c["os5_n"]       for c in cohorts.values())
    ebv_total       = sum(c["ebv_pos_n"]   for c in cohorts.values())
    hlh_total       = sum(c["hlh_n"]       for c in cohorts.values())
    hsct_total      = sum(c["hsct_n"]      for c in cohorts.values())
    aitl_t          = sum(c["aitl_n"]      for c in cohorts.values())
    nkt_t           = sum(c["nkt_n"]       for c in cohorts.values())
    chop_t          = sum(c["tx_chop_choep"] for c in cohorts.values())
    targeted_t      = sum(c["tx_targeted"] for c in cohorts.values())
    bv_t            = sum(c["tx_brentuximab"] for c in cohorts.values())

    return {
        "atlas":            "Hereditary-PTCL-Predisposition-Atlas",
        "seed_range":       f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":   total,
        "cr_n":             cr_total,
        "cr_pct":           round(cr_total / total * 100, 1),
        "os5_n":            os5_total,
        "os5_pct":          round(os5_total / total * 100, 1),
        "ebv_pos_n":        ebv_total,
        "ebv_pos_pct":      round(ebv_total / total * 100, 1),
        "hlh_n":            hlh_total,
        "hlh_pct":          round(hlh_total / total * 100, 1),
        "hsct_n":           hsct_total,
        "hsct_pct":         round(hsct_total / total * 100, 1),
        "aitl_n":           aitl_t,
        "aitl_pct":         round(aitl_t / total * 100, 1),
        "nkt_n":            nkt_t,
        "nkt_pct":          round(nkt_t / total * 100, 1),
        "chop_n":           chop_t,
        "chop_pct":         round(chop_t / total * 100, 1),
        "targeted_n":       targeted_t,
        "targeted_pct":     round(targeted_t / total * 100, 1),
        "bv_n":             bv_t,
        "bv_pct":           round(bv_t / total * 100, 1),
        "gene_counts": {
            gene: {
                "n":            c["n"],
                "cr_n":         c["cr_n"],
                "os5_n":        c["os5_n"],
                "ebv_pos_n":    c["ebv_pos_n"],
                "hlh_n":        c["hlh_n"],
                "hsct_n":       c["hsct_n"],
                "aitl_n":       c["aitl_n"],
                "nkt_n":        c["nkt_n"],
                "tx_chop_choep": c["tx_chop_choep"],
                "tx_targeted":  c["tx_targeted"],
                "ptcl_risk":    c["ptcl_risk"],
            }
            for gene, c in cohorts.items()
        },
    }


def generate_breakdown() -> dict:
    return {
        "atlas":     "Hereditary-PTCL-Predisposition-Atlas",
        "gene_data": {
            gd["gene"]: _make_cohort(gd, SEED_BASE + i)
            for i, gd in enumerate(ATLAS_GENES)
        },
    }


def generate_definitions() -> dict:
    defs = {
        gd["gene"]: gd["protein_size"]
        for gd in ATLAS_GENES
    }

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"PTCL RISK: {gd['ptcl_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-PTCL-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": KEY_CLINICAL_DISTINCTIONS,
    }
