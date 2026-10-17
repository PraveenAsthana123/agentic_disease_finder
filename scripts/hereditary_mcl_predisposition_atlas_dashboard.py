#!/usr/bin/env python3
"""Hereditary-MCL-Predisposition-Atlas -- Complete 8-Gene Reference
ATM      (ATM kinase; 3056aa; 11q22.3; AD LOF;
          Ataxia-Telangiectasia heterozygous (OMIM 208900);
          MCL del11q somatic 40-75%; ibrutinib-active del11q; RADIOSENSITIVITY ABSOLUTE;
          ceralasertib ATRi; seed SEED_BASE+0).
TP53     (p53; 393aa; 17p13.1; AD LOF;
          Li-Fraumeni Syndrome (OMIM 151623);
          MCL blastoid/pleomorphic 30-35%; complex karyotype; MCL-IPI high;
          AVOID RADIATION ABSOLUTELY; venetoclax preferred; seed SEED_BASE+1).
CDKN2A   (p16INK4A; 156aa; 9p21.3; AD LOF;
          FAMM-Pancreatic Syndrome (OMIM 600160);
          del9p21 in 30-60% blastoid MCL; CDK4/6i palbociclib INACTIVE in MCL;
          pancreatic EUS from 40yr; seed SEED_BASE+2).
RB1      (Rb1; 928aa; 13q14.2; AD LOF;
          Hereditary Retinoblastoma (OMIM 180200);
          del13q14 somatic 25-30% MCL; secondary B-cell NHL post-RT;
          CDK4/6i INACTIVE; bilateral Rb PATHOGNOMONIC; seed SEED_BASE+3).
NBN      (NBS1; 754aa; 8q21.3; AD/AR LOF;
          Nijmegen Breakage Syndrome (OMIM 251260);
          NHL 60x; MCL 3-5x; 657del5 Slavic founder; RADIOSENSITIVITY ABSOLUTE biallelic;
          cranial RT ABSOLUTELY CI; seed SEED_BASE+4).
BRCA2    (BRCA2; 3418aa; 13q12.3; AD LOF;
          HBOC-Hereditary Breast-Ovarian Cancer (OMIM 612555);
          NHL/MCL 2-3x; HRD 60%; cisplatin preferred; olaparib PARP;
          FA-D1 biallelic HIGHEST; seed SEED_BASE+5).
CHEK2    (CHK2; 543aa; 22q12.1; AD LOF;
          Li-Fraumeni Syndrome-2 (OMIM 609309);
          1100delC European 1:200; NHL 2x; ATM+CHEK2 additive 8-10x;
          NOT HRD — no PARPi; seed SEED_BASE+6).
MAP2K1   (MEK1; 393aa; 15q22.31; AD GOF;
          Cardio-facio-cutaneous Syndrome (OMIM 115150);
          MCL relapsed MAPK activation 10-15%; cobimetinib/trametinib MEKi;
          BRAF V600E co-test mandatory; seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3598-3605)
"""
import random

SEED_BASE = 3598

ATLAS_GENES = [
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD LOF -- 3056aa -- "
            "ATM-350kDa-PIK3-Kinase-"
            "MRN-Recruitment-DSB-Sensor-"
            "MCL-del11q-40-75pct-Ibrutinib-Active-RADIOSENSITIVITY-ABSOLUTE-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / ~350 kDa / 11q22.3 ATM MCL molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; PI3K-like serine/threonine kinase; "
            "  FAT domain (aa 1-1970): scaffolding, DNA damage response; "
            "  Kinase domain (aa 2714-2899): phosphorylates H2AX, CHK2, BRCA1, p53; "
            "  FATC domain (aa 2900-3056): redox-sensitive activation; "
            "  MRN complex (MRE11-RAD50-NBS1) recruits ATM to DSBs; "
            "  Germline LOF heterozygous = A-T carrier (~1:100 general population); "
            "  Biallelic LOF = classic A-T (cerebellar ataxia, telangiectasia, IgA deficiency); "
            "MCL MECHANISM: "
            "  del11q22-q23 (ATM locus) somatic in 40-75% MCL — most common chromosomal loss in MCL; "
            "  ATM LOF → defective DSB repair → genomic instability → cyclin D1 amplification; "
            "  del11q correlates with blastoid transformation, complex karyotype, IGHV-unmutated; "
            "  Germline ATM LOF: MCL risk 5-10x in het carriers; familial MCL clusters reported; "
            "  ATM null status sensitises to PARP inhibitors, ATRi ceralasertib (AZD6738); "
            "CLINICAL RULES FOR ATM-MCL: "
            "  RADIOSENSITIVITY ABSOLUTE — NO involved-field RT, NO consolidation RT, NO TBI; "
            "  Ibrutinib/acalabrutinib: del11q ATM-LOF MCL responds well to BTKi; "
            "  Ceralasertib AZD6738 (ATRi): Phase 2 ATM-null MCL — synthetic lethality; "
            "  Olaparib PARP: Phase 2 ATM-deficient B-cell lymphoma cohort active; "
            "  RIC-HSCT only if HSCT needed (fludarabine-based, no TBI); "
            "  del11q22.3 FISH mandatory at MCL diagnosis; "
            "  Cascade testing: ATM germline panel for first-degree relatives of young MCL; "
            "KEY VARIANTS: "
            "  Splice-site c.7271-2A>G (IVS51), missense c.6769A>G (p.Arg2257Gly); "
            "  Truncating in kinase/FAT domain: most damaging; "
            "  Founder variants: Polish c.7630-2A>C, Ashkenazi p.Ile2626Asn; "
            "SURVEILLANCE: "
            "  Annual lymph node exam + CBC; no RT for surveillance (PET-CT acceptable); "
            "  Breast MRI females (NO mammography alone); "
            "  del11q22.3 FISH at MCL relapse/progression; "
        ),
        "syndrome": "AT-Ataxia-Telangiectasia-Heterozygous-Carrier-NHL-MCL-Risk",
        "inheritance": "AD LOF heterozygous (biallelic = A-T)",
        "mcl_risk": "MCL 5-10x; del11q somatic 40-75% MCL (most common loss); blastoid/complex karyotype enriched; ibrutinib-active del11q; radiosensitivity absolute; ceralasertib ATRi",
        "pathognomonic": (
            "MCL + del11q22.3 FISH + germline ATM = hereditary MCL predisposition; "
            "A-T carrier: IgA deficiency (30-40%) + cerebellar signs (subtle het carrier); "
            "del11q22-q23 somatic → test germline ATM in young MCL (<50yr) or familial MCL"
        ),
        "key_avoid": (
            "AVOID ALL RT ABSOLUTELY (involved-field, ISRT, consolidation RT, TBI) — radiosensitivity; "
            "AVOID conventional busulfan TBI-based conditioning for HSCT; "
            "Do NOT withhold BTKi for del11q ATM-LOF MCL (ibrutinib/acalabrutinib active)"
        ),
        "key_rule": (
            "del11q22.3 FISH MANDATORY at MCL diagnosis — predicts BTKi sensitivity + radiosensitivity; "
            "Ceralasertib (ATRi) synthetic lethality in ATM-null MCL — enroll in trial; "
            "Olaparib PARP active in ATM-deficient lymphoma — dual PARP+BTKi emerging; "
            "RIC-HSCT (fludarabine-cyclophosphamide): only conditioning allowed in ATM LOF"
        ),
        "surveillance": (
            "Annual CBC + lymph node exam; PET-CT at progression (not annual); "
            "Breast MRI annual (females, NOT mammography primary); "
            "del11q FISH at every biopsy; cascade testing all first-degree relatives"
        ),
        "targeted_rx": (
            "Ibrutinib 560 mg QD (del11q BTKi first-line); "
            "Acalabrutinib 100 mg BID (preferred over ibrutinib — less AF); "
            "Ceralasertib (AZD6738) 160-240 mg BID (ATRi — clinical trial); "
            "Olaparib 300 mg BID (PARP — ATM-deficient lymphoma trial); "
            "Venetoclax 400-800 mg QD (BCL-2 inhibitor — del11q MCL active)"
        ),
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "G1-S-Checkpoint-Apoptosis-Guardian-"
            "MCL-Blastoid-30-35pct-AVOID-RADIATION-ABSOLUTELY-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 MCL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; transcription factor; tetramer; "
            "  N-terminal transactivation domain (aa 1-67): MDM2 binding; "
            "  Proline-rich region (aa 68-98): apoptosis signalling; "
            "  DNA-binding domain (aa 94-292): sequence-specific DNA binding, 90% cancer mutations here; "
            "  Tetramerisation domain (aa 325-356): essential for transcriptional activation; "
            "  C-terminal regulatory domain (aa 363-393): ubiquitination, acetylation; "
            "  LOF: loss of G1/S and G2/M checkpoint → uncontrolled proliferation; "
            "MCL MECHANISM: "
            "  TP53 mutation/del17p in MCL: 5-12% primary MCL, 30-35% blastoid/pleomorphic; "
            "  TP53 LOF → refractory to genotoxic chemotherapy (CHOP, HyperCVAD); "
            "  Complex karyotype (≥3 chromosomal abnormalities) in 60-80% TP53-mutant MCL; "
            "  MCL-IPI high-risk score; median OS <2yr after standard therapy; "
            "  Germline TP53 (LFS): MCL 5-8x; presents at younger age; "
            "  SOX11 positive blastoid MCL with TP53: most aggressive subtype; "
            "CLINICAL RULES FOR TP53-MCL: "
            "  AVOID RADIATION ABSOLUTELY (LFS + radiotherapy → fatal sarcomas); "
            "  Venetoclax 400-800 mg QD: BCL-2 independent of TP53 — preferred backbone; "
            "  Ibrutinib + venetoclax doublet: TP53-mutant MCL (AIM trial data); "
            "  Brexucabtagene autoleucel CAR-T: approved rel/ref MCL — TP53-mutant responds; "
            "  APR-246 (eprenetapopt) + ibrutinib: restores TP53 folding (investigational); "
            "  Allo-SCT in CR1 preferred if TP53-mutant young-fit MCL; "
            "  WB-MRI Toronto protocol (NO PET-CT for LFS surveillance); "
            "KEY VARIANTS: "
            "  R175H, R248W, R248Q, R273H (GOF hotspots — often worse prognosis in MCL); "
            "  Y220C: treatable with p53 reactivator APR-246; "
            "  del17p13.1: biallelic loss → complete p53 absence; FISH mandatory; "
            "SURVEILLANCE: "
            "  WB-MRI Toronto protocol annually (LFS); colonoscopy from 25yr; "
            "  Annual brain MRI; breast MRI/MX from 20yr females; "
            "  Cascade testing ALL first-degree relatives MANDATORY; "
        ),
        "syndrome": "LFS-Li-Fraumeni-Syndrome-MCL-Blastoid-Aggressive",
        "inheritance": "AD LOF heterozygous",
        "mcl_risk": "MCL blastoid/pleomorphic 30-35% TP53-mutant; complex karyotype 60-80%; MCL-IPI high; refractory chemotherapy; venetoclax preferred; AVOID RADIATION ABSOLUTELY",
        "pathognomonic": (
            "MCL + TP53 mutation + complex karyotype + blastoid morphology = TP53-MCL aggressive; "
            "LFS: sarcoma/brain/breast/adrenal cluster + MCL in family; "
            "del17p13.1 FISH at MCL diagnosis — mandatory for treatment stratification"
        ),
        "key_avoid": (
            "AVOID ALL RADIATION ABSOLUTELY (LFS — radiation-induced sarcoma risk); "
            "AVOID standard genotoxic CHOP/HyperCVAD alone in TP53-mutant MCL (primary resistance); "
            "AVOID watchful waiting for TP53-mutant MCL (rapid blastoid transformation)"
        ),
        "key_rule": (
            "Venetoclax 400-800 mg QD preferred backbone — BCL-2 inhibition bypasses TP53 apoptosis; "
            "Ibrutinib + venetoclax doublet (AIM trial): ORR 83% TP53-unselected; "
            "Brexucabtagene autoleucel (ZUMA-2) CAR-T: approved rel/ref MCL — TP53-mutant included; "
            "WB-MRI Toronto annually (NO CT/PET-CT for LFS cancer surveillance); "
            "Allo-SCT CR1 consolidation preferred in TP53-mutant young-fit MCL"
        ),
        "surveillance": (
            "WB-MRI Toronto annually; brain MRI annually; colonoscopy from 25yr; "
            "Breast MRI+mammography from 20yr (females); adrenal imaging as indicated; "
            "Cascade testing ALL first-degree relatives — TP53 germline"
        ),
        "targeted_rx": (
            "Venetoclax 400-800 mg QD (BCL-2 — preferred TP53-mutant backbone); "
            "Ibrutinib 560 mg QD (BTKi — add to venetoclax doublet AIM trial); "
            "Brexucabtagene autoleucel (CAR-T, approved rel/ref MCL); "
            "APR-246 eprenetapopt + ibrutinib (p53 reactivator — investigational); "
            "Allo-SCT CR1 (preferred consolidation TP53-mutant fit MCL)"
        ),
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 AD LOF -- 156aa -- "
            "p16-INK4A-16kDa-CDK4-6-Inhibitor-"
            "p14-ARF-MDM2-p53-Stabiliser-"
            "MCL-del9p21-Blastoid-30-60pct-CDK4-6i-INACTIVE-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / ~16 kDa / 9p21.3 CDKN2A MCL molecular context: "
            "STRUCTURE: "
            "  CDKN2A encodes TWO proteins from alternate reading frames: "
            "  p16INK4A (156 aa / 16 kDa): CDK4/6 inhibitor → retains Rb in hypophosphorylated state; "
            "  p14ARF (132 aa / 14 kDa): sequesters MDM2 → stabilises p53; "
            "  LOF: Rb pathway activation lost → CDK4/6 unrestricted → G1/S passage; "
            "  LOF: p14ARF absent → MDM2 unchecked → p53 degradation; "
            "MCL MECHANISM: "
            "  CDKN2A del9p21 in 30-60% blastoid/pleomorphic MCL (versus <10% conventional MCL); "
            "  del9p21 strongly associated with blastoid transformation and Ki-67 >70%; "
            "  CDK4/6i palbociclib PARADOXICALLY INACTIVE in MCL — cyclin D1 drives the biology; "
            "  MCL uniquely resistant to CDK4/6i despite CDKN2A loss (tested MONARCL trial); "
            "  Pancreatic cancer EUS surveillance from 40yr (FAMM — pancreatic 17-39x risk); "
            "  FAMM (OMIM 600160): multiple atypical nevi + pancreatic cancer family history; "
            "CLINICAL RULES FOR CDKN2A-MCL: "
            "  DO NOT use palbociclib/ribociclib/abemaciclib in MCL — MONARCL trial failed; "
            "  del9p21 FISH = blastoid marker → escalate therapy (HyperCVAD-RA, allo-SCT CR1); "
            "  Pancreatic MRI/EUS from 40yr (FAMM — 17-39x lifetime risk); "
            "  Annual skin exam (FAMM — melanoma 28-67x lifetime risk); "
            "  Venetoclax + ibrutinib doublet: effective in CDKN2A-null blastoid MCL; "
            "KEY VARIANTS: "
            "  del9p21.3 homozygous: complete p16INK4A + p14ARF loss; "
            "  Methylation-silenced (somatic): not germline; germline = truncating/missense exon 1-3; "
            "  V126D, R87P: FAMM-associated missense; "
            "SURVEILLANCE: "
            "  Annual melanoma skin check from puberty; annual ophthalmology (uveal melanoma risk); "
            "  Pancreatic MRI/EUS from 40yr (FAMM); breast MRI (CDKN2A germline data emerging); "
        ),
        "syndrome": "FAMM-Pancreatic-Syndrome-Melanoma-Pancreatic-MCL-Risk",
        "inheritance": "AD LOF heterozygous",
        "mcl_risk": "del9p21 in 30-60% blastoid MCL; CDK4/6i palbociclib INACTIVE (MONARCL failed); pancreatic EUS from 40yr; melanoma 28-67x; blastoid marker → allo-SCT CR1",
        "pathognomonic": (
            "Blastoid MCL + del9p21 FISH = CDKN2A-null poor prognosis; "
            "FAMM: multiple atypical nevi + family history pancreatic cancer + MCL = CDKN2A germline; "
            "CDK4/6i are INACTIVE in MCL regardless of CDKN2A status (do not prescribe palbociclib)"
        ),
        "key_avoid": (
            "AVOID CDK4/6i (palbociclib, ribociclib, abemaciclib) in MCL — MONARCL trial failed; "
            "AVOID watchful waiting for del9p21 blastoid MCL (rapid progression); "
            "Do NOT confuse CDKN2A-DLBCL (where CDK4/6i may be active) with MCL (INACTIVE)"
        ),
        "key_rule": (
            "del9p21 FISH at MCL diagnosis — blastoid marker → escalate to HyperCVAD-RA or clinical trial; "
            "Pancreatic MRI/EUS from 40yr MANDATORY (FAMM — 17-39x pancreatic cancer); "
            "Annual skin exam + dermatology from puberty (melanoma 28-67x FAMM); "
            "Venetoclax + ibrutinib doublet: most active regimen in CDKN2A-null blastoid MCL"
        ),
        "surveillance": (
            "Annual dermatology skin check from puberty; pancreatic MRI/EUS from 40yr; "
            "Annual ophthalmology (uveal melanoma); cascade testing first-degree relatives"
        ),
        "targeted_rx": (
            "Ibrutinib 560 mg QD + venetoclax 400 mg QD (doublet — blastoid MCL); "
            "HyperCVAD-RA (consolidation in del9p21 blastoid MCL — aggressive regimen); "
            "Allo-SCT CR1 (preferred consolidation in CDKN2A-null blastoid young-fit MCL); "
            "Brexucabtagene autoleucel (CAR-T rel/ref MCL); "
            "NOT: palbociclib/ribociclib/abemaciclib — INACTIVE in MCL"
        ),
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 AD LOF -- 928aa -- "
            "Rb1-110kDa-Cell-Cycle-Regulator-"
            "E2F-Sequestration-G1-S-Checkpoint-"
            "MCL-del13q14-25-30pct-Secondary-NHL-Post-RT-CDK4-6i-INACTIVE-OMIM-180200"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / ~110 kDa / 13q14.2 RB1 MCL molecular context: "
            "STRUCTURE: "
            "  928 aa / ~110 kDa; 'pocket protein' family; "
            "  N-terminal domain (aa 1-378): cyclin D-CDK4/6 binding; "
            "  Pocket domain (aa 379-792): E2F transcription factor sequestration; "
            "  C-terminal domain (aa 793-928): CDK phosphorylation sites, LXCXE binding; "
            "  Hypophosphorylated Rb: sequesters E2F → G1 arrest; "
            "  CDK4/6-cyclin D phosphorylates Rb → E2F released → S-phase entry; "
            "  Germline LOF het = hereditary retinoblastoma (bilateral Rb PATHOGNOMONIC); "
            "MCL MECHANISM: "
            "  del13q14 (RB1 locus) somatic in 25-30% MCL; "
            "  RB1 LOF cooperates with cyclin D1 (t(11;14)) — both hit G1 checkpoint; "
            "  Secondary B-cell NHL (including MCL) in hereditary Rb survivors post-RT; "
            "  CDK4/6i palbociclib INACTIVE in MCL (cyclin D1 mechanism bypasses Rb pathway); "
            "  del13q14 → shortened PFS on ibrutinib monotherapy; "
            "  Hereditary Rb patients: secondary MCL risk from prior radiation; "
            "CLINICAL RULES FOR RB1-MCL: "
            "  CDK4/6i INACTIVE in MCL regardless of RB1 status (see CDKN2A rule); "
            "  Prior RT (for Rb treatment) → increased secondary MCL risk; "
            "  WB-MRI annual (hereditary Rb surveillance); NO additional RT for MCL treatment; "
            "  Ibrutinib + venetoclax doublet active in del13q MCL; "
            "  Allo-SCT: discuss if del13q14 + other high-risk features (blastoid, del17p); "
            "KEY VARIANTS: "
            "  Truncating (exon 1-27 nonsense/frameshift): LOF hereditary Rb; "
            "  Missense in pocket domain: LOF; "
            "  del13q14.3: somatic loss — FISH detects; "
            "SURVEILLANCE: "
            "  Annual dilated fundus exam (ophthalmology) for unaffected eye; "
            "  WB-MRI annually (secondary malignancy surveillance hereditary Rb); "
            "  CBC annual; minimize radiation (secondary sarcoma risk); "
        ),
        "syndrome": "Hereditary-Retinoblastoma-Secondary-MCL-Post-RT-CDK4-6i-INACTIVE",
        "inheritance": "AD LOF heterozygous",
        "mcl_risk": "del13q14 somatic 25-30% MCL; secondary B-cell NHL post-RT in hereditary Rb; CDK4/6i palbociclib INACTIVE; bilateral Rb PATHOGNOMONIC germline; shortened PFS ibrutinib monotherapy",
        "pathognomonic": (
            "Bilateral retinoblastoma (childhood) + MCL (adult) = hereditary RB1 LOF; "
            "del13q14.3 FISH in MCL: poor prognosis marker (shorter PFS BTKi monotherapy); "
            "Leukocoria (white pupil reflex) PATHOGNOMONIC bilateral Rb in children"
        ),
        "key_avoid": (
            "AVOID additional radiation in hereditary RB1 MCL (secondary sarcoma risk + prior RT); "
            "AVOID CDK4/6i (palbociclib, ribociclib) in MCL — INACTIVE regardless of RB1; "
            "AVOID single-agent ibrutinib in del13q14 MCL (shortened PFS — use doublet)"
        ),
        "key_rule": (
            "del13q14 FISH at MCL diagnosis — shortened PFS on ibrutinib monotherapy → prefer doublet; "
            "WB-MRI annual for hereditary Rb survivors (secondary malignancy surveillance); "
            "Ibrutinib 560 mg QD + venetoclax 400 mg QD doublet preferred in del13q MCL; "
            "Bilateral fundus exam annually (ophthalmology) for unaffected eye in hereditary Rb"
        ),
        "surveillance": (
            "Annual ophthalmology (dilated fundus, unaffected eye); WB-MRI annually; "
            "CBC annual; avoid further radiation; cascade testing first-degree relatives"
        ),
        "targeted_rx": (
            "Ibrutinib 560 mg QD + venetoclax 400 mg QD (doublet — del13q MCL); "
            "Acalabrutinib 100 mg BID + venetoclax (preferred doublet — less AF risk); "
            "Brexucabtagene autoleucel (CAR-T rel/ref MCL); "
            "Allo-SCT CR1 (if del13q + del17p or blastoid — very high risk); "
            "NOT: CDK4/6i — INACTIVE in MCL"
        ),
    },
    {
        "gene": "NBN",
        "protein": (
            "NBN -- 8q21.3 AD/AR LOF -- 754aa -- "
            "NBS1-95kDa-MRN-Complex-"
            "FHA-BRCT-MRE11-ATM-Scaffold-"
            "NHL-60x-MCL-3-5x-657del5-Slavic-RADIOSENSITIVITY-ABSOLUTE-Biallelic-OMIM-251260"
        ),
        "locus": "8q21.3",
        "protein_size": (
            "754 aa / ~95 kDa / 8q21.3 NBN MCL molecular context: "
            "STRUCTURE: "
            "  754 aa / ~95 kDa; scaffold protein of MRN (MRE11-RAD50-NBS1) complex; "
            "  FHA domain (aa 24-109): phosphoprotein binding; "
            "  BRCT1 (aa 111-196) + BRCT2 (aa 221-332): MDC1 interaction, H2AX-γ binding; "
            "  MRE11-binding domain (aa 735-754): MRN complex assembly; "
            "  MRN complex: senses DSBs → recruits ATM → signals CHK2, BRCA1, p53; "
            "  Germline biallelic = NBS (Nijmegen Breakage Syndrome): microcephaly + immunodeficiency; "
            "  Germline het (657del5 Slavic ~1:150): NHL 60x highest risk; "
            "MCL MECHANISM: "
            "  NBN het: NHL 60x (highest risk of any NHL predisposition gene); MCL 3-5x; "
            "  657del5 (c.657_661del5) Slavic founder — Poland, Czech, Slovakia: 1 in 150; "
            "  NBN LOF → defective ATM activation → impaired DSB repair; "
            "  MCL in NBN carriers: younger onset, aggressive features; "
            "  Biallelic NBS patients: AVOID ALL RT AND ALKYLATING AGENTS ABSOLUTELY; "
            "  Het carriers: RADIOSENSITIVITY (reduce RT dose 20-30%); "
            "CLINICAL RULES FOR NBN-MCL: "
            "  657del5 Slavic ancestry: test NBN germline in ALL MCL (1 in 150 carrier); "
            "  Biallelic NBS: RADIOSENSITIVITY ABSOLUTE — NO RT, NO alkylating, fludarabine-based HSCT; "
            "  Het NBN carriers: reduce RT dose 20-30% (radiosensitivity intermediate); "
            "  Ibrutinib + venetoclax: preferred BTKi doublet in NBN-LOF MCL; "
            "  PARP + ATRi combination trial for NBN-LOF MCL (MRN-deficient); "
            "KEY VARIANTS: "
            "  c.657_661del5 (657del5): Slavic founder — most common in Central Europe; "
            "  R215W: hypomorphic; "
            "  Splice: IVS10+1G>C; "
            "SURVEILLANCE: "
            "  Annual CBC + lymph node exam; no unnecessary CT (minimize radiation); "
            "  Immunoglobulin levels (biallelic: CVID-like immunodeficiency); "
            "  Cascade testing 657del5 Slavic ancestry; "
        ),
        "syndrome": "NBS-Nijmegen-Breakage-Syndrome-NHL-60x-MCL-3-5x-657del5-Slavic",
        "inheritance": "AD het LOF (NHL risk) / AR biallelic LOF (NBS)",
        "mcl_risk": "NHL 60x HIGHEST risk gene; MCL 3-5x; 657del5 Slavic 1:150; biallelic NBS RADIOSENSITIVITY ABSOLUTE (no RT, no alkylating); het carriers intermediate radiosensitivity (reduce RT 20-30%)",
        "pathognomonic": (
            "MCL + Slavic ancestry + microcephaly family history = NBN 657del5 biallelic/het; "
            "NHL 60x: HIGHEST known germline NHL predisposition; "
            "NBS biallelic: microcephaly + bird-like face + immunodeficiency = PATHOGNOMONIC"
        ),
        "key_avoid": (
            "Biallelic NBS: AVOID ALL RT ABSOLUTELY and ALL ALKYLATING agents; "
            "Het NBN carriers: AVOID full-dose RT (reduce 20-30%) — intermediate radiosensitivity; "
            "AVOID cranial RT in NBN (biallelic and het carriers)"
        ),
        "key_rule": (
            "657del5 Slavic (Central European) ancestry: test NBN germline in ALL MCL diagnoses; "
            "Biallelic NBS: fludarabine-based RIC-HSCT only (NO TBI, NO busulfan); "
            "Ibrutinib 560 mg QD + venetoclax 400 mg QD: preferred NBN-LOF MCL backbone; "
            "PARP inhibitor + ATRi emerging in MRN-deficient (NBN/MRE11/RAD50) lymphoma"
        ),
        "surveillance": (
            "Annual CBC + lymph node exam; minimize radiation surveillance (PET-CT: use sparingly); "
            "Ig levels (biallelic: check for CVID-like); "
            "Cascade testing: 657del5 Slavic — test all siblings/children"
        ),
        "targeted_rx": (
            "Ibrutinib 560 mg QD + venetoclax 400 mg QD (doublet); "
            "Acalabrutinib 100 mg BID + venetoclax (preferred — less AF); "
            "Olaparib 300 mg BID (PARP — MRN-deficient MCL trial); "
            "Fludarabine-based RIC-HSCT (biallelic NBS — NO TBI, NO busulfan); "
            "NOT: TBI/busulfan conditioning, NOT: full-dose RT"
        ),
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 AD LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-"
            "RAD51-Loading-HRD-"
            "NHL-MCL-2-3x-HRD-60pct-Cisplatin-Preferred-Olaparib-PARP-OMIM-612555"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / ~384 kDa / 13q12.3 BRCA2 MCL molecular context: "
            "STRUCTURE: "
            "  3418 aa / ~384 kDa; largest tumour suppressor; "
            "  N-terminal PALB2-binding domain (aa 1-40): BRCA1-PALB2-BRCA2 complex; "
            "  BRC repeats (aa 1002-2085): RAD51 loading onto ssDNA at DSBs; "
            "  OB-fold domains (aa 2402-3190): ssDNA binding, RPA displacement; "
            "  C-terminal FANCD1 region: Fanconi anaemia core complex; "
            "  LOF → HRD (homologous recombination deficiency) → NHEJ error-prone repair; "
            "  Biallelic = FA-D1 (Fanconi anaemia): highest lymphoma + solid tumour risk; "
            "MCL MECHANISM: "
            "  BRCA2 het: NHL/MCL 2-3x; HRD → PARP trapping sensitises to olaparib; "
            "  HRD score 60% in BRCA2-mut B-cell lymphoma (GIS/COSMIC HRD signatures); "
            "  Cisplatin/oxaliplatin DNA crosslinkers preferred (HRD exploits DSB repair deficit); "
            "  Olaparib maintenance after response: active in BRCA2-deficient lymphoma; "
            "  Biallelic FA-D1: AVOID ALKYLATING AGENTS ABSOLUTELY (MMC test for Fanconi); "
            "  BRCA2 germline + MCL: younger onset, concurrent breast/ovarian risk; "
            "CLINICAL RULES FOR BRCA2-MCL: "
            "  HRD testing MANDATORY at MCL diagnosis in BRCA2 carriers; "
            "  Cisplatin 70 mg/m² preferred over cyclophosphamide/bendamustine in BRCA2-MCL; "
            "  Olaparib 300 mg BID: maintenance after platinum-based response; "
            "  Biallelic FA-D1: AVOID ALKYLATING, DEB/MMC chromosomal fragility test; "
            "  Concurrent breast MRI annually (NO mammography primary in BRCA2); "
            "  Prophylactic salpingo-oophorectomy 35-40yr females BRCA2; "
            "KEY VARIANTS: "
            "  c.5946delT (6174delT) Ashkenazi founder; c.9999del11 (9396del11) Icelandic; "
            "  c.3175del4 (3tab): truncating variants in coding exons; "
            "  N991D: polymorphism — do not misclassify; "
            "SURVEILLANCE: "
            "  Annual breast MRI (NO mammography primary); transvaginal US + CA-125 annual; "
            "  RRSO 35-40yr females; prostate PSA + MRI males from 40yr; "
            "  MCL surveillance: CBC annual, lymph node exam; "
        ),
        "syndrome": "HBOC-Hereditary-Breast-Ovarian-Cancer-NHL-MCL-2-3x-HRD-Olaparib",
        "inheritance": "AD LOF heterozygous (biallelic = FA-D1)",
        "mcl_risk": "NHL/MCL 2-3x; HRD 60% (cisplatin preferred, olaparib maintenance); biallelic FA-D1 AVOID ALKYLATING ABSOLUTELY; concurrent breast 71%/ovarian 17% risk; younger onset MCL in BRCA2 carriers",
        "pathognomonic": (
            "MCL + breast/ovarian history (personal or family) = BRCA2 germline; "
            "HRD score 60% in BRCA2-MCL (GIS signature SBS3); "
            "FA-D1 biallelic: Fanconi features (short stature, thumb anomalies, DEB/MMC fragility) + MCL PATHOGNOMONIC"
        ),
        "key_avoid": (
            "Biallelic FA-D1: AVOID ALL ALKYLATING AGENTS ABSOLUTELY (bendamustine, cyclophosphamide, platinum); "
            "Het BRCA2: AVOID non-platinum DNA-damaging agents as first choice (cisplatin preferred over alkylating); "
            "AVOID olaparib without HRD confirmation (cisplatin response first)"
        ),
        "key_rule": (
            "HRD testing MANDATORY in BRCA2-MCL; cisplatin 70 mg/m² preferred over bendamustine/cyclophosphamide; "
            "Olaparib 300 mg BID maintenance after platinum response in BRCA2-MCL; "
            "Annual breast MRI (NO mammography primary in BRCA2 carriers); "
            "RRSO 35-40yr females BRCA2 (reduces ovarian cancer risk 80-90%)"
        ),
        "surveillance": (
            "Annual breast MRI (females); transvaginal US + CA-125 annual; "
            "Prostate PSA + MRI males from 40yr; RRSO discussion from 35yr females; "
            "MCL: CBC annual, lymph node exam; cascade testing first-degree relatives"
        ),
        "targeted_rx": (
            "Cisplatin 70 mg/m² IV (preferred platinum in BRCA2-MCL HRD); "
            "Olaparib 300 mg BID (PARP maintenance after platinum response); "
            "Ibrutinib + venetoclax doublet (BTKi backbone + BCL-2); "
            "Brexucabtagene autoleucel (CAR-T rel/ref MCL); "
            "NOT: alkylating agents as first choice in BRCA2 het (use platinum instead)"
        ),
    },
    {
        "gene": "CHEK2",
        "protein": (
            "CHEK2 -- 22q12.1 AD LOF -- 543aa -- "
            "CHK2-60kDa-ATM-Downstream-Effector-"
            "1100delC-European-1in200-"
            "NHL-2x-ATM-CHEK2-Additive-8-10x-NOT-HRD-NO-PARPi-OMIM-609309"
        ),
        "locus": "22q12.1",
        "protein_size": (
            "543 aa / ~60 kDa / 22q12.1 CHEK2 MCL molecular context: "
            "STRUCTURE: "
            "  543 aa / ~60 kDa; serine/threonine checkpoint kinase; "
            "  SQ/TQ domain (aa 1-69): ATM phosphorylation sites (T68 primary); "
            "  FHA domain (aa 92-175): phosphoprotein binding, dimerisation; "
            "  Kinase domain (aa 226-486): phosphorylates CDC25A/C, BRCA2, p53-Ser20; "
            "  ATM activates CHK2 (T68 phosphorylation) → CDC25C phosphorylation → G2/M arrest; "
            "  CHK2 → BRCA1-S988 phosphorylation → HR activation; "
            "  CHEK2 LOF: G2/M checkpoint defective + impaired HR; "
            "MCL MECHANISM: "
            "  CHEK2 LOF: NHL 2x overall; MCL moderate risk (~1.5-2x); "
            "  ATM + CHEK2 compound heterozygous: 8-10x MCL/NHL (synthetic additive); "
            "  1100delC (c.1100delC): European founder — 1:200 general European population; "
            "  IVS2+1G>A: Ashkenazi Jewish founder; "
            "  I157T: Eastern European — intermediate risk (reduces kinase but not null); "
            "  CHEK2 NOT HRD — olaparib/PARP do NOT work (different from BRCA2); "
            "CLINICAL RULES FOR CHEK2-MCL: "
            "  CHEK2 alone: moderate MCL risk — no specific altered treatment except surveillance; "
            "  ATM + CHEK2 compound het: treat as high-risk (8-10x) — aggressive surveillance; "
            "  DO NOT prescribe olaparib for CHEK2-MCL (NOT HRD, PARP inhibitors inactive); "
            "  Standard ibrutinib/venetoclax doublet appropriate for CHEK2-MCL; "
            "  Breast cancer risk: 1100delC → 25-30% lifetime breast cancer; "
            "  Annual breast MRI from 40yr females CHEK2 1100delC (moderate risk); "
            "KEY VARIANTS: "
            "  c.1100delC: European 1:200; breast + NHL; "
            "  IVS2+1G>A: Ashkenazi; "
            "  I157T: Eastern European, lower penetrance; "
            "  R145W, R180H: rare pathogenic; "
            "SURVEILLANCE: "
            "  Annual breast MRI from 40yr (females 1100delC); CBC annual; "
            "  Colonoscopy from 40yr (CHEK2 — colorectal 2x); "
            "  Cascade testing: 1100delC — test siblings/children (carrier 1:200); "
        ),
        "syndrome": "LFS2-Li-Fraumeni-Syndrome-2-NHL-2x-ATM-CHEK2-Additive-8-10x-NOT-HRD",
        "inheritance": "AD LOF heterozygous",
        "mcl_risk": "NHL 2x; ATM+CHEK2 additive 8-10x; 1100delC European 1:200; NOT HRD (olaparib INACTIVE); moderate risk alone; breast 25-30%; standard BTKi treatment appropriate",
        "pathognomonic": (
            "MCL + 1100delC European + breast cancer family history = CHEK2 germline; "
            "ATM + CHEK2 compound het (both pathogenic) = 8-10x MCL/NHL risk (synergistic); "
            "CHEK2 does NOT cause HRD — do NOT prescribe PARPi (ineffective)"
        ),
        "key_avoid": (
            "AVOID olaparib/rucaparib/niraparib in CHEK2-MCL — NOT HRD, PARP inhibitors inactive; "
            "Do NOT conflate CHEK2 with BRCA2 HRD pathway — different mechanism, different treatment; "
            "AVOID omitting breast surveillance in CHEK2 1100delC females (25-30% lifetime breast risk)"
        ),
        "key_rule": (
            "CHEK2 alone: standard ibrutinib + venetoclax doublet — no treatment escalation needed; "
            "ATM + CHEK2 compound het: high-risk 8-10x — aggressive surveillance + allo-SCT discussion; "
            "CHEK2 NOT HRD: cisplatin no advantage over standard treatment (unlike BRCA2); "
            "Annual breast MRI from 40yr females 1100delC; colonoscopy from 40yr"
        ),
        "surveillance": (
            "Annual breast MRI from 40yr (females 1100delC); CBC annual; "
            "Colonoscopy from 40yr; cascade testing 1100delC/IVS2 Ashkenazi carriers"
        ),
        "targeted_rx": (
            "Ibrutinib 560 mg QD + venetoclax 400 mg QD (standard doublet — CHEK2-MCL); "
            "Acalabrutinib 100 mg BID + venetoclax (preferred — less AF); "
            "Brexucabtagene autoleucel (CAR-T rel/ref MCL); "
            "NOT: olaparib/PARPi — INACTIVE in CHEK2 (not HRD); "
            "NOT: escalated platinum regimen (no HRD advantage in CHEK2 vs BRCA2)"
        ),
    },
    {
        "gene": "MAP2K1",
        "protein": (
            "MAP2K1 -- 15q22.31 AD GOF -- 393aa -- "
            "MEK1-44kDa-MAPKK-ERK-Activator-"
            "RAS-MAPK-Kinase-Cascade-"
            "MCL-Relapsed-MAPK-Activation-10-15pct-Cobimetinib-Trametinib-MEKi-OMIM-115150"
        ),
        "locus": "15q22.31",
        "protein_size": (
            "393 aa / ~44 kDa / 15q22.31 MAP2K1 MCL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~44 kDa; MAP kinase kinase 1 (MEK1); "
            "  N-terminal regulatory domain (aa 1-64): ERK-binding domain (EBD); "
            "  Kinase domain (aa 65-367): ATP binding, activation loop T292; "
            "  NRR (negative regulatory region, aa 98-104): autoinhibitory; "
            "  GOF: NRR deletion/point mutation → constitutive ERK1/2 phosphorylation; "
            "  GOF: BRAF V600E → RAF → MEK → ERK signalling cascade; "
            "  Germline GOF: CFC (Cardio-facio-cutaneous) syndrome — heart defect + face + hair; "
            "MCL MECHANISM: "
            "  MAP2K1 somatic GOF in 10-15% relapsed/transformed MCL; "
            "  NRR deletion (Δ98-104): most common MCL MAP2K1 GOF variant; "
            "  MAPK hyperactivation → increased BCL-2 translation + ibrutinib resistance; "
            "  MAP2K1 GOF often co-occurs with BTK C481S resistance mutation at ibrutinib relapse; "
            "  Germline MAP2K1 GOF (CFC): MCL rare but documented; "
            "  RAS/MAPK pathway: also includes BRAF V600E (co-test MANDATORY); "
            "CLINICAL RULES FOR MAP2K1-MCL: "
            "  Cobimetinib 60 mg OD (21 days on/7 days off): MEK1 inhibitor active in MAP2K1 GOF; "
            "  Trametinib 2 mg QD: alternative MEKi; MEK162 (binimetinib) emerging; "
            "  BRAF V600E co-test MANDATORY — add vemurafenib/dabrafenib if BRAF co-mutated; "
            "  p-ERK IHC on biopsy: confirms MAPK pathway activation (biomarker); "
            "  Ibrutinib + venetoclax backbone + cobimetinib triplet: investigated for MAP2K1 MCL; "
            "  CFC germline: heart defect (ASD/VSD/PS) — cardiology assessment MANDATORY; "
            "KEY VARIANTS: "
            "  NRR deletion Δ98-104 (p.E98_I103del): most common MCL GOF; "
            "  P124L, P124S: activating, CFC-associated; "
            "  K57N, Q56P: CFC germline hotspots; "
            "SURVEILLANCE: "
            "  BRAF V600E co-test at every MCL biopsy; p-ERK IHC biomarker; "
            "  CFC: cardiology echo (heart defects); skin exam (keratosis pilaris); "
            "  Repeat biopsy at BTKi progression (MAP2K1 GOF + BTK C481S co-mutation check); "
        ),
        "syndrome": "CFC-Cardio-Facio-Cutaneous-Syndrome-MAPK-MEK1-MCL-Relapsed-MEKi-Active",
        "inheritance": "AD GOF",
        "mcl_risk": "MCL relapsed/transformed MAPK activation 10-15%; cobimetinib/trametinib MEKi active; BRAF V600E co-test MANDATORY; CFC germline (heart defect + face + sparse hair); ibrutinib resistance mechanism",
        "pathognomonic": (
            "Relapsed MCL + MAP2K1 GOF + BTK C481S = ibrutinib resistance with MAPK escape; "
            "CFC syndrome: broad forehead + sparse curly hair + heart defect + MCL = germline MAP2K1 GOF; "
            "p-ERK IHC positive at relapse biopsy = MAPK pathway active → MEKi indicated"
        ),
        "key_avoid": (
            "AVOID ibrutinib monotherapy at MAP2K1 GOF MCL relapse (MAPK escape mechanism); "
            "Do NOT ignore BRAF V600E co-test — MAP2K1 + BRAF co-mutation changes treatment; "
            "AVOID standard ICE/DHAP salvage without adding MEKi in MAP2K1 GOF relapsed MCL"
        ),
        "key_rule": (
            "Cobimetinib 60 mg OD (21/7 schedule) or trametinib 2 mg QD: MEKi active in MAP2K1 GOF; "
            "BRAF V600E co-test MANDATORY at relapse biopsy (add vemurafenib if BRAF V600E); "
            "p-ERK IHC on biopsy: confirms MAPK activation — biomarker for MEKi response; "
            "CFC germline: cardiology echo (ASD/VSD/pulmonary stenosis 55-65%)"
        ),
        "surveillance": (
            "BRAF V600E co-test at every biopsy; p-ERK IHC; "
            "CFC: cardiology echo from diagnosis; skin/dermatology annually; "
            "Cascade testing CFC-family (MAP2K1 germline GOF)"
        ),
        "targeted_rx": (
            "Cobimetinib 60 mg OD (21 days on/7 off) — MEKi first choice MAP2K1 GOF; "
            "Trametinib 2 mg QD — alternative MEKi; "
            "Ibrutinib 560 mg QD + venetoclax 400 mg QD + cobimetinib triplet (MAP2K1 MCL); "
            "Vemurafenib 960 mg BID + cobimetinib (if BRAF V600E co-mutation); "
            "Brexucabtagene autoleucel (CAR-T rel/ref MCL — MAP2K1 included)"
        ),
    },
]


def _make_patients(gd: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    n = 40
    gene = gd["gene"]

    # MCL-specific distributions per gene
    age_range = {
        "ATM":    (48, 78),
        "TP53":   (45, 75),
        "CDKN2A": (50, 80),
        "RB1":    (40, 70),
        "NBN":    (38, 68),
        "BRCA2":  (46, 74),
        "CHEK2":  (52, 80),
        "MAP2K1": (50, 76),
    }.get(gene, (50, 75))

    blastoid_rate = {
        "ATM":    0.45,
        "TP53":   0.70,
        "CDKN2A": 0.65,
        "RB1":    0.35,
        "NBN":    0.30,
        "BRCA2":  0.25,
        "CHEK2":  0.20,
        "MAP2K1": 0.55,
    }.get(gene, 0.35)

    cr_rate = {
        "ATM":    0.72,
        "TP53":   0.48,
        "CDKN2A": 0.52,
        "RB1":    0.68,
        "NBN":    0.74,
        "BRCA2":  0.70,
        "CHEK2":  0.75,
        "MAP2K1": 0.50,
    }.get(gene, 0.65)

    del11q_rate = {
        "ATM": 0.70,
    }.get(gene, 0.12)

    del17p_rate = {
        "TP53": 0.65,
    }.get(gene, 0.08)

    del9p_rate = {
        "CDKN2A": 0.60,
        "TP53":   0.20,
    }.get(gene, 0.08)

    del13q_rate = {
        "RB1": 0.75,
    }.get(gene, 0.15)

    ibrutinib_rate = {
        "ATM":    0.82,
        "TP53":   0.45,
        "CDKN2A": 0.50,
        "RB1":    0.65,
        "NBN":    0.78,
        "BRCA2":  0.70,
        "CHEK2":  0.75,
        "MAP2K1": 0.38,
    }.get(gene, 0.65)

    venetoclax_rate = {
        "ATM":    0.68,
        "TP53":   0.72,
        "CDKN2A": 0.65,
        "RB1":    0.60,
        "NBN":    0.62,
        "BRCA2":  0.67,
        "CHEK2":  0.64,
        "MAP2K1": 0.58,
    }.get(gene, 0.63)

    ki67_high_rate = {     # Ki-67 ≥70% (blastoid)
        "ATM":    0.40,
        "TP53":   0.68,
        "CDKN2A": 0.60,
        "RB1":    0.30,
        "NBN":    0.25,
        "BRCA2":  0.22,
        "CHEK2":  0.18,
        "MAP2K1": 0.50,
    }.get(gene, 0.30)

    pts = []
    for i in range(n):
        age = rng.randint(*age_range)
        blastoid = rng.random() < blastoid_rate
        ki67_high = rng.random() < ki67_high_rate
        del11q = rng.random() < del11q_rate
        del17p = rng.random() < del17p_rate
        del9p = rng.random() < del9p_rate
        del13q = rng.random() < del13q_rate
        complex_karyotype = (blastoid and rng.random() < 0.72) or rng.random() < 0.20
        sox11_pos = rng.random() < (0.55 if blastoid else 0.85)
        ighv_unmut = rng.random() < (0.70 if blastoid else 0.50)
        stage4 = rng.random() < 0.75
        bm_involved = rng.random() < 0.65
        mcl_ipi_high = blastoid or (ki67_high and rng.random() < 0.65)
        ibrutinib_resp = rng.random() < ibrutinib_rate
        venetoclax_resp = rng.random() < venetoclax_rate
        cr = rng.random() < cr_rate
        alive_3yr = cr and rng.random() < (0.70 if not blastoid else 0.45)
        concurrent_breast = (gene == "BRCA2") and (rng.random() < 0.30)
        pts.append({
            "id":              f"{gene}-{i+1:03d}",
            "age_dx":         age,
            "blastoid":       blastoid,
            "ki67_high":      ki67_high,
            "del11q":         del11q,
            "del17p":         del17p,
            "del9p21":        del9p,
            "del13q14":       del13q,
            "complex_karyotype": complex_karyotype,
            "sox11_positive": sox11_pos,
            "ighv_unmutated": ighv_unmut,
            "stage4":         stage4,
            "bm_involved":    bm_involved,
            "mcl_ipi_high":   mcl_ipi_high,
            "ibrutinib_response": ibrutinib_resp,
            "venetoclax_response": venetoclax_resp,
            "cr":             cr,
            "alive_3yr":      alive_3yr,
            "concurrent_breast": concurrent_breast,
        })
    return pts


def generate_overview() -> dict:
    all_pts = []
    for i, gd in enumerate(ATLAS_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        all_pts.extend(pts)

    total = len(all_pts)
    blastoid_n = sum(1 for p in all_pts if p["blastoid"])
    ki67_high_n = sum(1 for p in all_pts if p["ki67_high"])
    del11q_n = sum(1 for p in all_pts if p["del11q"])
    del17p_n = sum(1 for p in all_pts if p["del17p"])
    del9p_n = sum(1 for p in all_pts if p["del9p21"])
    del13q_n = sum(1 for p in all_pts if p["del13q14"])
    complex_n = sum(1 for p in all_pts if p["complex_karyotype"])
    sox11_n = sum(1 for p in all_pts if p["sox11_positive"])
    ighv_unmut_n = sum(1 for p in all_pts if p["ighv_unmutated"])
    ipi_high_n = sum(1 for p in all_pts if p["mcl_ipi_high"])
    stage4_n = sum(1 for p in all_pts if p["stage4"])
    bm_n = sum(1 for p in all_pts if p["bm_involved"])
    ibru_resp_n = sum(1 for p in all_pts if p["ibrutinib_response"])
    vene_resp_n = sum(1 for p in all_pts if p["venetoclax_response"])
    cr_n = sum(1 for p in all_pts if p["cr"])
    alive_3yr_n = sum(1 for p in all_pts if p["alive_3yr"])
    concurrent_b_n = sum(1 for p in all_pts if p["concurrent_breast"])
    avg_age = round(sum(p["age_dx"] for p in all_pts) / total, 1)

    gene_counts = {}
    for i, gd in enumerate(ATLAS_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        gene_counts[gd["gene"]] = {
            "n":           len(pts),
            "blastoid_n":  sum(1 for p in pts if p["blastoid"]),
            "ki67_high_n": sum(1 for p in pts if p["ki67_high"]),
            "del11q_n":    sum(1 for p in pts if p["del11q"]),
            "del17p_n":    sum(1 for p in pts if p["del17p"]),
            "complex_n":   sum(1 for p in pts if p["complex_karyotype"]),
            "ipi_high_n":  sum(1 for p in pts if p["mcl_ipi_high"]),
            "cr_n":        sum(1 for p in pts if p["cr"]),
            "alive_3yr_n": sum(1 for p in pts if p["alive_3yr"]),
            "mcl_risk":    gd["mcl_risk"],
        }

    return {
        "atlas":            "Hereditary-MCL-Predisposition-Atlas",
        "seed_range":       f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":   total,
        "total_genes":      len(ATLAS_GENES),
        "avg_age_dx":       avg_age,
        "blastoid_n":       blastoid_n,
        "blastoid_pct":     round(blastoid_n / total * 100, 1),
        "ki67_high_n":      ki67_high_n,
        "ki67_high_pct":    round(ki67_high_n / total * 100, 1),
        "del11q_n":         del11q_n,
        "del11q_pct":       round(del11q_n / total * 100, 1),
        "del17p_n":         del17p_n,
        "del17p_pct":       round(del17p_n / total * 100, 1),
        "del9p21_n":        del9p_n,
        "del9p21_pct":      round(del9p_n / total * 100, 1),
        "del13q14_n":       del13q_n,
        "del13q14_pct":     round(del13q_n / total * 100, 1),
        "complex_karyotype_n":   complex_n,
        "complex_karyotype_pct": round(complex_n / total * 100, 1),
        "sox11_positive_n":      sox11_n,
        "sox11_positive_pct":    round(sox11_n / total * 100, 1),
        "ighv_unmutated_n":      ighv_unmut_n,
        "ighv_unmutated_pct":    round(ighv_unmut_n / total * 100, 1),
        "mcl_ipi_high_n":        ipi_high_n,
        "mcl_ipi_high_pct":      round(ipi_high_n / total * 100, 1),
        "stage4_n":         stage4_n,
        "stage4_pct":       round(stage4_n / total * 100, 1),
        "bm_involved_n":    bm_n,
        "bm_involved_pct":  round(bm_n / total * 100, 1),
        "ibrutinib_response_n":  ibru_resp_n,
        "ibrutinib_response_pct": round(ibru_resp_n / total * 100, 1),
        "venetoclax_response_n": vene_resp_n,
        "venetoclax_response_pct": round(vene_resp_n / total * 100, 1),
        "cr_n":             cr_n,
        "cr_pct":           round(cr_n / total * 100, 1),
        "alive_3yr_n":      alive_3yr_n,
        "alive_3yr_pct":    round(alive_3yr_n / total * 100, 1),
        "concurrent_breast_n":   concurrent_b_n,
        "gene_counts":      gene_counts,
    }


def generate_breakdown() -> dict:
    gene_data = {}
    for i, gd in enumerate(ATLAS_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        cr_n = sum(1 for p in pts if p["cr"])
        blastoid_n = sum(1 for p in pts if p["blastoid"])
        ki67_hi_n = sum(1 for p in pts if p["ki67_high"])
        del11q_n = sum(1 for p in pts if p["del11q"])
        del17p_n = sum(1 for p in pts if p["del17p"])
        complex_n = sum(1 for p in pts if p["complex_karyotype"])
        sox11_n = sum(1 for p in pts if p["sox11_positive"])
        ipi_hi_n = sum(1 for p in pts if p["mcl_ipi_high"])
        stage4_n = sum(1 for p in pts if p["stage4"])
        alive3_n = sum(1 for p in pts if p["alive_3yr"])
        concurrent_b = sum(1 for p in pts if p["concurrent_breast"])
        ibru_n = sum(1 for p in pts if p["ibrutinib_response"])
        vene_n = sum(1 for p in pts if p["venetoclax_response"])
        avg_age_dx = round(sum(p["age_dx"] for p in pts) / len(pts), 1)

        patients_list = [{
            "id":              p["id"],
            "age_dx":         p["age_dx"],
            "blastoid":       p["blastoid"],
            "ki67_high":      p["ki67_high"],
            "del11q":         p["del11q"],
            "del17p":         p["del17p"],
            "del9p21":        p["del9p21"],
            "del13q14":       p["del13q14"],
            "complex_karyotype": p["complex_karyotype"],
            "sox11_positive": p["sox11_positive"],
            "ighv_unmutated": p["ighv_unmutated"],
            "mcl_ipi_high":   p["mcl_ipi_high"],
            "ibrutinib_response": p["ibrutinib_response"],
            "venetoclax_response": p["venetoclax_response"],
            "cr":             p["cr"],
            "alive_3yr":      p["alive_3yr"],
        } for p in pts]

        gene_data[gd["gene"]] = {
            "gene":            gd["gene"],
            "syndrome":        gd["syndrome"],
            "locus":           gd["locus"],
            "inheritance":     gd["inheritance"],
            "n":               len(pts),
            "cr_n":            cr_n,
            "blastoid_n":      blastoid_n,
            "ki67_hi_n":       ki67_hi_n,
            "del11q_n":        del11q_n,
            "del17p_n":        del17p_n,
            "complex_n":       complex_n,
            "sox11_n":         sox11_n,
            "ipi_hi_n":        ipi_hi_n,
            "stage4_n":        stage4_n,
            "alive_3yr_n":     alive3_n,
            "concurrent_breast_n": concurrent_b,
            "ibrutinib_n":     ibru_n,
            "venetoclax_n":    vene_n,
            "avg_age_dx":      avg_age_dx,
            "mcl_risk":        gd["mcl_risk"],
            "pathognomonic":   gd["pathognomonic"],
            "key_rule":        gd["key_rule"],
            "key_avoid":       gd["key_avoid"],
            "targeted_rx":     gd["targeted_rx"],
            "patients":        patients_list[:20],
        }

    return {
        "atlas":      "Hereditary-MCL-Predisposition-Atlas",
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_data":  gene_data,
    }


def generate_definitions() -> dict:
    defs = {}
    for gd in ATLAS_GENES:
        defs[gd["gene"]] = {
            "gene":          gd["gene"],
            "locus":         gd["locus"],
            "protein_size":  gd["protein_size"],
            "syndrome":      gd["syndrome"],
            "inheritance":   gd["inheritance"],
            "mcl_risk":      gd["mcl_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_avoid":     gd["key_avoid"],
            "key_rule":      gd["key_rule"],
            "surveillance":  gd["surveillance"],
            "targeted_rx":   gd["targeted_rx"],
        }

    key_clinical_distinctions = [
        "ATM-MCL RADIOSENSITIVITY ABSOLUTE — del11q FISH MANDATORY: del11q somatic 40-75% MCL (most common loss); "
        "RADIOSENSITIVITY ABSOLUTE — NO RT ever (no involved-field, no consolidation, no TBI); "
        "ibrutinib/acalabrutinib active in del11q ATM-LOF MCL; "
        "ceralasertib (ATRi) + olaparib (PARPi): synthetic lethality trials in ATM-null MCL; "
        "RIC-HSCT (fludarabine-cyclophosphamide) only — NO TBI/busulfan conditioning",

        "TP53-MCL AVOID RADIATION ABSOLUTELY — venetoclax backbone preferred: "
        "blastoid/pleomorphic MCL 30-35% TP53-mutant; complex karyotype 60-80%; MCL-IPI high; "
        "AVOID genotoxic CHOP/HyperCVAD alone (primary resistance in TP53-mutant MCL); "
        "venetoclax 400-800 mg QD backbone (BCL-2 independent of TP53); "
        "ibrutinib + venetoclax AIM trial: ORR 83% TP53-unselected; "
        "brexucabtagene autoleucel CAR-T approved rel/ref MCL; WB-MRI Toronto (NO CT/PET LFS)",

        "CDKN2A-MCL del9p21 CDK4/6i INACTIVE (MONARCL failed) — pancreatic EUS from 40yr: "
        "del9p21 in 30-60% blastoid MCL; CDK4/6i palbociclib INACTIVE in MCL (MONARCL trial) — "
        "do NOT prescribe regardless of CDKN2A status; "
        "del9p21 = blastoid marker → HyperCVAD-RA or allo-SCT CR1; "
        "FAMM: pancreatic cancer 17-39x (EUS from 40yr); melanoma 28-67x (annual skin exam)",

        "RB1-MCL del13q14 shortened PFS BTKi monotherapy — CDK4/6i INACTIVE: "
        "del13q14 somatic 25-30% MCL; secondary B-cell NHL in hereditary Rb post-RT; "
        "CDK4/6i INACTIVE in MCL regardless of RB1 status; "
        "del13q = shortened PFS ibrutinib monotherapy → prefer BTKi+venetoclax doublet; "
        "WB-MRI annually (hereditary Rb secondary malignancy surveillance); NO additional RT",

        "NBN-MCL NHL 60x HIGHEST RISK — 657del5 Slavic 1:150 — RADIOSENSITIVITY biallelic: "
        "NHL 60x: highest known germline NHL risk gene; MCL 3-5x; "
        "657del5 Slavic founder (Polish/Czech/Slovak/Slovenian) — 1:150 carrier frequency; "
        "test NBN germline in ALL MCL with Slavic ancestry; "
        "biallelic NBS: RADIOSENSITIVITY ABSOLUTE — NO RT, NO alkylating; fludarabine RIC-HSCT only; "
        "het NBN: reduce RT dose 20-30% (intermediate radiosensitivity)",

        "BRCA2-MCL HRD 60% — cisplatin preferred over bendamustine — olaparib maintenance: "
        "NHL/MCL 2-3x; HRD score 60% (GIS signature SBS3); "
        "cisplatin 70 mg/m² preferred over bendamustine/cyclophosphamide in BRCA2-MCL; "
        "olaparib 300 mg BID maintenance after platinum response (PARP inhibitor); "
        "biallelic FA-D1: AVOID ALL ALKYLATING ABSOLUTELY; DEB/MMC chromosomal fragility test; "
        "annual breast MRI (NO mammography primary); RRSO 35-40yr females",

        "CHEK2-MCL NOT HRD — olaparib INACTIVE — ATM+CHEK2 additive 8-10x: "
        "CHEK2 NOT HRD — olaparib/PARPi INACTIVE (different from BRCA2); "
        "1100delC European 1:200; NHL 2x; ATM+CHEK2 compound het = 8-10x MCL risk (synergistic); "
        "CHEK2 alone: standard BTKi+venetoclax doublet appropriate (no escalation); "
        "1100delC females: annual breast MRI from 40yr (25-30% lifetime breast cancer); "
        "Do NOT conflate CHEK2 with BRCA2 HRD pathway",

        "MAP2K1-MCL GOF cobimetinib MEKi — BRAF V600E co-test MANDATORY — ibrutinib resistance: "
        "MAP2K1 GOF in 10-15% relapsed/transformed MCL; NRR deletion Δ98-104 most common; "
        "constitutive ERK1/2 phosphorylation → ibrutinib resistance mechanism; "
        "cobimetinib 60 mg OD (21/7) or trametinib 2 mg QD: MEKi active in MAP2K1 GOF; "
        "BRAF V600E co-test MANDATORY at relapse (vemurafenib if BRAF co-mutated); "
        "p-ERK IHC confirms MAPK activation; CFC germline: cardiology echo mandatory",
    ]

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"MCL RISK: {gd['mcl_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-MCL-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": key_clinical_distinctions,
    }
