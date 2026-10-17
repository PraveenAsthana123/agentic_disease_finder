#!/usr/bin/env python3
"""Hereditary-MZL-Predisposition-Atlas -- Complete 8-Gene Reference
Marginal Zone Lymphoma: Extranodal MALT, Nodal MZL, Splenic MZL.

MYD88    (MyD88; 296aa; 3p22.2; AD GOF/LOF;
          susceptibility to MZL/MALT; L265P somatic in 5-15% MALT;
          germline activating variants → constitutive TLR/NF-κB;
          Hyper-IgM phenotype; anti-H.pylori failure; seed SEED_BASE+0).
TNFRSF13B (TACI; 293aa; 17p11.2; AD CVID2;
           CVID2 (OMIM 240500); NHL 10-12x HIGHEST germline NHL risk;
           MZL/MALT risk; C104R 1:500 European; IVIG mandatory IgG <7 g/L;
           seed SEED_BASE+1).
ATM      (ATM kinase; 3056aa; 11q22.3; AD LOF;
          Ataxia-Telangiectasia (OMIM 208900);
          NHL 4-5x; MZL/MALT 3-4x; del11q somatic; RADIOSENSITIVITY ABSOLUTE;
          ceralasertib ATRi; seed SEED_BASE+2).
TP53     (p53; 393aa; 17p13.1; AD LOF;
          Li-Fraumeni Syndrome (OMIM 151623);
          MZL transformation 20-30%; complex karyotype; AVOID RADIATION ABSOLUTELY;
          venetoclax preferred; seed SEED_BASE+3).
CDKN2A   (p16INK4A; 156aa; 9p21.3; AD LOF;
          FAMM-Pancreatic Syndrome (OMIM 600160);
          gastric MALT 5-10x; H. pylori + CDKN2A → MALT;
          CDK4/6i palbociclib INACTIVE in MZL; seed SEED_BASE+4).
BRCA2    (BRCA2; 3418aa; 13q12.3; AD LOF;
          HBOC (OMIM 612555);
          NHL/MZL 1.5-2x; HRD 60%; cisplatin preferred; olaparib PARP;
          seed SEED_BASE+5).
CARD11   (CARD11; 1154aa; 7p22.2; AD LOF/GOF;
          BENTA Disease (OMIM 616098) — GOF; CARD11 LOF → CID;
          GOF constitutive NF-κB → MZL/MALT extranodal;
          venetoclax preferred; MALT1 inhibitor investigational;
          seed SEED_BASE+6).
PIK3CD   (PI3Kδ; 1044aa; 1p36.22; AD GOF;
          APDS1 (OMIM 615513);
          EBV+ lymphoma; MZL/MALT 10-15% in APDS1;
          leniolisib FDA2023; sirolimus bridge; seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3606-3613)
"""
import random

SEED_BASE = 3606

ATLAS_GENES = [
    {
        "gene": "MYD88",
        "protein": (
            "MYD88 -- 3p22.2 AD GOF/LOF -- 296aa -- "
            "MyD88-33kDa-TLR-IL1R-Adaptor-"
            "TIR-Death-Domain-"
            "MZL-MALT-L265P-5-15pct-NF-kB-Constitutive-OMIM-616641"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "296 aa / ~33 kDa / 3p22.2 MYD88 MZL molecular context: "
            "STRUCTURE: "
            "  296 aa / ~33 kDa; intracellular TLR/IL-1R signal adaptor; "
            "  Death domain (aa 1-110): self-oligomerisation, Myddosome assembly; "
            "  TIR domain (aa 159-296): TLR4/7/9/IL-1R interaction; "
            "  L265P (BB-loop TIR): constitutive Myddosome → NF-κB + JAK/STAT3; "
            "  Somatic L265P in 90% WM (see WM/LPL atlas), 5-15% MALT, 3-5% DLBCL; "
            "  Germline activating MYD88 variants (not L265P) → Hyper-IgM-like phenotype; "
            "MZL/MALT MECHANISM: "
            "  Somatic L265P in gastric/ocular/salivary MALT lymphoma (5-15%); "
            "  Confers H. pylori eradication resistance — MALT that is L265P+ = "
            "    antibiotic-refractory; requires rituximab ± chlorambucil upfront; "
            "  Germline activating variants: constitutive NF-κB → marginal zone B-cell expansion; "
            "  MYD88 LOF germline: increased susceptibility to Mycobacteria, Herpes, "
            "    Enterobacteriaceae → chronic antigen stimulation → secondary MALT; "
            "CLINICAL RULES FOR MYD88-MZL: "
            "  L265P somatic testing MANDATORY at MALT diagnosis (predicts H. pylori-eradication failure); "
            "  L265P+ gastric MALT: skip H. pylori–only approach; go to rituximab-based Rx; "
            "  Germline MYD88 GOF: ibrutinib BTKi + anti-inflammatory; monitor WM conversion; "
            "  MYD88 L265P in MALT + orbital/CNS involvement: rule out transformed DLBCL urgently; "
            "KEY VARIANTS: "
            "  L265P somatic (c.794T>C): most pathogenic; test from FFPE MALT biopsy; "
            "  Germline GOF: rare activating TIR-domain variants; "
            "  MYD88 LOF: homozygous (AR) = rare immunodeficiency; heterozygous = mild risk; "
            "SURVEILLANCE: "
            "  Annual CBC + paraprotein screen (WM conversion); "
            "  L265P in any MALT sample → repeat imaging if transformation suspected; "
            "  Ophthalmology annual (ocular adnexal MALT 20-25% of MZL); "
        ),
        "syndrome": "MYD88-GOF-MALT-L265P-HP-Eradication-Resistant",
        "inheritance": "AD GOF (somatic L265P 5-15% MALT; germline GOF rare; AR LOF rare)",
        "mzl_risk": "Somatic L265P → H. pylori eradication failure in MALT; germline activating → MZL/MALT 2-3x; NF-κB constitutive; ibrutinib active",
        "pathognomonic": (
            "MALT lymphoma + MYD88 L265P = H. pylori-eradication resistant; skip antibiotic monotherapy; "
            "Orbital/CNS MALT + MYD88 L265P = rule out DLBCL transformation URGENTLY; "
            "Germline MYD88 GOF: Hyper-IgM-like + recurrent infections + marginal zone expansion"
        ),
        "key_avoid": (
            "AVOID H. pylori–eradication monotherapy if L265P+ MALT (will fail — resistance); "
            "AVOID ibrutinib dose de-escalation in MYD88 GOF MZL (BTKi active, full dose mandatory); "
            "Do NOT miss CNS/orbital transformation in L265P+ MALT (MRI brain mandatory)"
        ),
        "key_rule": (
            "MYD88 L265P testing from FFPE MALT biopsy MANDATORY — guides H. pylori Rx decision; "
            "L265P+ → rituximab-based therapy; do NOT rely on antibiotics alone; "
            "WM conversion monitoring: annual IgM paraprotein + BM if rising IgM; "
            "BRAF V600E co-test if MALT orbital/thyroid (hairy cell leukemia DDx)"
        ),
        "surveillance": (
            "Annual: CBC + peripheral blood immunophenotyping; serum IgM paraprotein; "
            "ophthalmology (ocular adnexal MALT); "
            "MRI brain/orbit if orbital MALT (annual first 3yr); "
            "CECT chest/abdomen 12-monthly until remission confirmed ×2yr"
        ),
        "targeted_rx": (
            "L265P+ MALT: rituximab 375 mg/m² ×4-6 cycles ± chlorambucil (R-Chl); "
            "MYD88 GOF MZL: ibrutinib 420 mg OD (BTKi); zanubrutinib alternative; "
            "MALT1 inhibitor (JNJ-67856633) in NF-κB–active MZL: Phase 1-2; "
            "L265P- MALT gastric: H. pylori eradication (triple therapy) → 75-80% CR"
        ),
    },
    {
        "gene": "TNFRSF13B",
        "protein": (
            "TNFRSF13B -- 17p11.2 AD CVID2 -- 293aa -- "
            "TACI-31kDa-TNF-Receptor-"
            "BAFF-APRIL-Signalling-"
            "CVID2-NHL-10-12x-HIGHEST-MZL-MALT-IVIG-Mandatory-OMIM-240500"
        ),
        "locus": "17p11.2",
        "protein_size": (
            "293 aa / ~31 kDa / 17p11.2 TNFRSF13B (TACI) MZL context: "
            "STRUCTURE: "
            "  293 aa / ~31 kDa; transmembrane TNF receptor superfamily member 13B; "
            "  CRD domain (aa 28-84): BAFF binding; "
            "  CRD domain (aa 86-137): APRIL binding; "
            "  TRAF3-binding domain (aa 233-293): NF-κB2 (non-canonical) activation; "
            "  TACI signal: B-cell class-switch recombination (IgG, IgA), plasma-cell differentiation; "
            "  Loss-of-function variants → CVID (hypogammaglobulinaemia) → IVIG dependence; "
            "  C104R: most common CVID-associated variant, European 1:500 carrier; "
            "CVID2/NHL MECHANISM: "
            "  TNFRSF13B LOF → absent BAFF/APRIL-mediated class-switch → low IgG, IgA, IgM; "
            "  Impaired B-cell apoptosis clearance → surviving B-cells clonally expand; "
            "  NHL risk 10-12x (HIGHEST known germline NHL risk; higher than ATM); "
            "  MZL/MALT most common lymphoma subtype in CVID (marginal zone B-cell origin); "
            "  EBV-driven lymphoproliferation in CVID (TNFRSF13B LOF PATHOGNOMONIC risk); "
            "CLINICAL RULES FOR TNFRSF13B-MZL: "
            "  IVIG every 3-4wk MANDATORY (IgG trough >7 g/L); stop ONLY with haematology approval; "
            "  NHL surveillance: PET-CT or CECT every 12-18 months in CVID; "
            "  Lymph node biopsy if ANY new lymphadenopathy >2cm or B symptoms; "
            "  Leniolisib (PI3Kδ inhibitor) if APDS1 co-exists (rare digenic); "
            "  Monthly splenomegaly screen (splenic MZL most aggressive CVID-NHL); "
            "KEY VARIANTS: "
            "  C104R (c.310T>C): most common; heterozygous in CVID; 1:500 European; "
            "  A181E: severe LOF; "
            "  P251L: moderate; associated with autoimmunity + lymphoma; "
            "SURVEILLANCE: "
            "  Monthly IgG trough; 6-monthly IgM (rising = MZL/WM screen); "
            "  Annual PET-CT in confirmed CVID2 + splenomegaly; "
            "  EBV viral load annually; "
            "  Annual pulmonary function (CVID granulomatous-lymphocytic ILD)"
        ),
        "syndrome": "CVID2-TNFRSF13B-TACI-LOF-NHL-10-12x-HIGHEST-IVIG-Mandatory",
        "inheritance": "AD (heterozygous C104R common); AR (homozygous/compound het severe CVID)",
        "mzl_risk": "NHL 10-12x HIGHEST germline; MZL/MALT most common NHL subtype in CVID; EBV+ lymphoproliferation; splenic MZL aggressive; IVIG mandatory IgG >7g/L",
        "pathognomonic": (
            "CVID + TNFRSF13B C104R + new lymphadenopathy = MZL/MALT until proven otherwise; "
            "Splenomegaly + IgM spike in CVID2 = splenic MZL PATHOGNOMONIC risk pattern; "
            "EBV+ B-cell proliferation in CVID = TNFRSF13B LOF until excluded"
        ),
        "key_avoid": (
            "NEVER stop IVIG without specialist approval — IgG <7 g/L → uncontrolled EBV → lymphoma; "
            "AVOID rituximab monotherapy without IVIG support (IgG will collapse — life-threatening); "
            "Do NOT miss splenic MZL: splenomegaly in CVID2 needs BM biopsy + splenic MRI, not watchful waiting"
        ),
        "key_rule": (
            "IVIG every 3-4wk MANDATORY, IgG trough >7 g/L ALWAYS; "
            "NHL surveillance mandatory in ALL CVID2: PET-CT or CECT annually; "
            "Splenic MZL in CVID2: splenectomy + rituximab ± bendamustine (not watch-and-wait); "
            "Rituximab + IVIG co-administration MANDATORY (prevent IgG collapse)"
        ),
        "surveillance": (
            "Monthly: IgG trough (target >7 g/L); "
            "6-monthly: CBC, IgM paraprotein; EBV viral load; "
            "Annual: CECT chest/abdomen/pelvis; spleen size ultrasound; PFTs; "
            "BM biopsy if: unexplained cytopenia, splenomegaly >18cm, IgM >20 g/L, B symptoms"
        ),
        "targeted_rx": (
            "MALT/MZL in CVID2: rituximab 375 mg/m² + IVIG concurrent (MANDATORY); "
            "R-bendamustine (R-B) for splenic MZL: bendamustine 90 mg/m² d1-2 + R; "
            "R-CHOP if transformed DLBCL; "
            "Leniolisib 70 mg BID (FDA2023) if digenic APDS1; "
            "HSCT (allo-RIC) for relapsed/refractory MZL in CVID2 — curative for CVID + lymphoma"
        ),
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD LOF -- 3056aa -- "
            "ATM-350kDa-PIK3-Kinase-"
            "MRN-Recruitment-DSB-Sensor-"
            "MZL-NHL-4-5x-del11q-RADIOSENSITIVITY-ABSOLUTE-ceralasertib-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / ~350 kDa / 11q22.3 ATM MZL context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; PI3K-like serine/threonine kinase; "
            "  FAT domain (aa 1-1970): scaffolding; "
            "  Kinase domain (aa 2714-2899): phosphorylates H2AX, CHK2, BRCA1, p53; "
            "  FATC domain (aa 2900-3056): redox-sensitive activation; "
            "  Germline heterozygous LOF (~1:100 population); A-T biallelic = full syndrome; "
            "MZL MECHANISM: "
            "  ATM germline LOF het: NHL 4-5x; MZL/splenic MZL 3-4x; "
            "  del11q22-q23 somatic in 15-25% MZL (especially splenic MZL); "
            "  ATM null → defective DSB repair → genomic instability → MZL progression; "
            "  A-T biallelic patients: MZL/splenic MZL frequent in teens-early adulthood; "
            "  ATM LOF → RADIOSENSITIVITY ABSOLUTE (no RT safe threshold known); "
            "CLINICAL RULES FOR ATM-MZL: "
            "  RADIOSENSITIVITY ABSOLUTE — NO RT (ISRT, consolidation, TBI) — no exceptions; "
            "  Ceralasertib (AZD6738 ATRi): Phase 2 ATM-null lymphoma; synthetic lethality; "
            "  RIC-HSCT only (fludarabine-based, no TBI); "
            "  del11q22.3 FISH at MZL diagnosis in any patient <60yr or familial lymphoma; "
            "KEY VARIANTS: "
            "  IVS51 splice c.7271-2A>G; p.Arg2257Gly; truncating in kinase/FAT; "
            "  Founders: Polish c.7630-2A>C; Ashkenazi p.Ile2626Asn; "
            "SURVEILLANCE: "
            "  Annual CBC + lymph node exam; no RT for surveillance (PET-CT acceptable); "
            "  Breast MRI females (NO mammography primary); "
            "  del11q22.3 FISH at MZL relapse/progression"
        ),
        "syndrome": "AT-Ataxia-Telangiectasia-Heterozygous-NHL-MZL-RADIOSENSITIVITY-ABSOLUTE",
        "inheritance": "AD LOF heterozygous (biallelic = classical A-T)",
        "mzl_risk": "NHL 4-5x; MZL/splenic MZL 3-4x; del11q22.3 somatic 15-25% MZL; radiosensitivity absolute; ceralasertib ATRi; no RT ever",
        "pathognomonic": (
            "Splenic MZL + del11q22.3 FISH + age <55yr = test germline ATM; "
            "A-T biallelic: MZL/splenic lymphoma in teens-young adults; "
            "ATM + rituximab ± bendamustine: RT-free approach MANDATORY"
        ),
        "key_avoid": (
            "AVOID ALL RT ABSOLUTELY — involved-field, consolidation, TBI, stereotactic; "
            "AVOID TBI-based conditioning for HSCT; fludarabine-based RIC ONLY; "
            "Do NOT miss del11q22.3 in MZL: FISH mandatory in young/familial MZL"
        ),
        "key_rule": (
            "del11q22.3 FISH MANDATORY at MZL diagnosis — predicts radiosensitivity; "
            "Ceralasertib synthetic lethality in ATM-null lymphoma — enroll in trial; "
            "Cascade testing: ATM panel for first-degree relatives of MZL patient <55yr; "
            "Splenectomy for splenic MZL: safe (no RT needed); rituximab monotherapy effective"
        ),
        "surveillance": (
            "Annual: CBC; lymphadenopathy exam; splenic US (splenic MZL); "
            "Breast MRI females from 40yr (no mammography primary — radiation risk); "
            "del11q22.3 FISH at MZL relapse; "
            "PET-CT: preferred (no RT); CECT if PET unavailable"
        ),
        "targeted_rx": (
            "Splenic MZL: splenectomy (curative in 40-50%); rituximab 375 mg/m² ×4-8 if no surgery; "
            "Systemic MZL: R-bendamustine preferred (R-CHOP acceptable, no RT adjuvant); "
            "ATRi ceralasertib: enroll Phase 2 ATM-null lymphoma trial if relapsed; "
            "RIC-HSCT (fludarabine-based ONLY, no TBI) for eligible relapsed ATM-MZL"
        ),
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "Transcription-Factor-Tetramer-"
            "LFS-MZL-Transformation-20-30pct-AVOID-RADIATION-ABSOLUTELY-Venetoclax-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 MZL context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; sequence-specific transcription factor; "
            "  N-terminal TAD1/TAD2 (aa 1-67): MDM2 interaction; "
            "  Proline-rich (aa 68-98): apoptosis; "
            "  DBD core (aa 102-292): 95% of LFS variants — DNA contact R248W/Q, R273H; "
            "  Tetramerisation domain (aa 325-356); "
            "  C-terminal regulatory domain (aa 364-393); "
            "MZL MECHANISM: "
            "  TP53 germline LOF het = LFS; NHL in LFS families (ESMO 2% cumulative); "
            "  TP53 somatic loss at MZL transformation → high-grade diffuse large B-cell; "
            "  del17p or biallelic somatic TP53 in 20-30% transformed MZL; "
            "  RT in TP53 germline LOF: risk of therapy-related sarcoma / secondary leukaemia; "
            "  WB-MRI Toronto protocol: MANDATORY LFS surveillance (avoids ionising radiation); "
            "CLINICAL RULES FOR TP53-MZL: "
            "  AVOID RADIATION ABSOLUTELY — ionising risk for therapy-related malignancy in LFS; "
            "  Venetoclax (BCL2 inhibitor): effective in TP53-mutated B-cell lymphoma; preferred; "
            "  CAR-T (brexucabtagene autoleucel): active in TP53-mutated transformed MZL; "
            "  WB-MRI Toronto protocol (annual, no CT) for ongoing LFS surveillance; "
            "KEY VARIANTS: "
            "  Hotspots: R248W (c.742C>T), R248Q (c.743G>A), R175H (c.524G>A); "
            "  C176F — cysteine rich domain; G245D/S — most common in NHL; "
            "SURVEILLANCE: "
            "  WB-MRI annually (NO CT, NO PET — ionising avoided in LFS); "
            "  FAST protocol (Frankfurt Adaptations): brain MRI q6M if CNS lesion; "
            "  Monthly self-exam; annual gastroscopy from 25yr; "
            "  Breast MRI 6-monthly from 20-25yr or 5yr before earliest family cancer"
        ),
        "syndrome": "LFS-Li-Fraumeni-Syndrome-MZL-Transformation-AVOID-RADIATION-ABSOLUTELY",
        "inheritance": "AD LOF heterozygous",
        "mzl_risk": "LFS NHL rare (2% cumulative); MZL transformation to DLBCL 20-30%; del17p 20-30% transformed MZL; AVOID RADIATION ABSOLUTELY; venetoclax preferred; WB-MRI Toronto mandatory",
        "pathognomonic": (
            "MZL + del17p + complex karyotype = TP53 germline testing in <55yr; "
            "Transformed MZL + del17p = venetoclax backbone NOT anthracycline-first; "
            "LFS + B-cell mass = MZL/DLBCL: biopsy urgently (not watch and wait)"
        ),
        "key_avoid": (
            "AVOID ALL RT ABSOLUTELY — sarcoma/AML risk in LFS; "
            "AVOID CT for surveillance in LFS — WB-MRI ONLY; "
            "AVOID R-CHOP anthracycline escalation without venetoclax in del17p MZL (will fail)"
        ),
        "key_rule": (
            "WB-MRI Toronto annual MANDATORY — no CT, no PET/CT for LFS surveillance; "
            "del17p in MZL: venetoclax 400 mg OD (V) backbone; venetoclax + rituximab (VR); "
            "Transformed MZL del17p: CAR-T or allo-SCT early consideration; "
            "Cascade testing: TP53 germline panel for all first-degree relatives of MZL LFS patient"
        ),
        "surveillance": (
            "Annual: WB-MRI (brain + spine + abdomen/pelvis) — NO CT; "
            "6-monthly: lymph node exam; CBC; "
            "Breast MRI 6-monthly from 20yr (NO mammography); "
            "Brain MRI q6M if prior CNS involvement; "
            "Gastroscopy annual from 25yr (gastric MALT/DLBCL risk)"
        ),
        "targeted_rx": (
            "del17p MZL/transformed: venetoclax 400 mg OD + rituximab (VR); "
            "Transformed DLBCL TP53: CAR-T brexucabtagene autoleucel; "
            "R-bendamustine for non-transformed MZL (no anthracycline needed if early); "
            "Allo-SCT RIC early in del17p/complex karyotype MZL if eligible"
        ),
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 AD LOF -- 156aa -- "
            "p16-INK4A-16kDa-CDK4-6-Inhibitor-"
            "ARF-p14-Alternative-Reading-Frame-"
            "FAMM-Gastric-MALT-5-10x-CDK4-6i-INACTIVE-H-pylori-Synergy-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / ~16 kDa / 9p21.3 CDKN2A MZL context: "
            "STRUCTURE: "
            "  156 aa / ~16 kDa (p16INK4A isoform); "
            "  p14ARF encoded by alternate reading frame (same locus); "
            "  Ankyrin repeats: CDK4/6 binding and inhibition; "
            "  LOF → CDK4/6 hyperactivation → Rb hyperphosphorylation → uncontrolled S-phase entry; "
            "MZL MECHANISM: "
            "  CDKN2A germline LOF het = FAMM (Familial Atypical Multiple Mole Melanoma); "
            "  Gastric MALT: H. pylori + CDKN2A del → somatic CDKN2A loss in gastric MALT 20-40%; "
            "  Gastric MALT 5-10x risk in germline CDKN2A carriers + H. pylori co-exposure; "
            "  del9p21 somatic in 15-20% MZL transformation events; "
            "  CDK4/6i (palbociclib/ribociclib/abemaciclib): INACTIVE in MZL (no cell cycle arrest); "
            "  p14ARF loss also affects MDM2-p53 axis → TP53 degradation → dual pathway loss; "
            "CLINICAL RULES FOR CDKN2A-MZL: "
            "  H. pylori eradication MANDATORY in ALL CDKN2A carriers (co-exposure synergy); "
            "  CDK4/6i INACTIVE in MZL — do not use palbociclib/ribociclib for MZL; "
            "  EUS pancreatic screen from 40yr (FAMM-associated pancreatic cancer risk 17-38x); "
            "  Annual dermoscopy (melanoma 25-36x risk); "
            "KEY VARIANTS: "
            "  p.Gly101Trp (founder, Dutch/European); p.Arg24Pro; del9p21.3 large deletions; "
            "  p14ARF variants may co-occur; "
            "SURVEILLANCE: "
            "  Annual dermoscopy; annual H. pylori breath test (eradication monitoring); "
            "  EUS pancreatic from 40yr; "
            "  6-monthly gastroscopy if gastric MALT diagnosed (CDKN2A → recurrence risk); "
            "  del9p21.3 FISH at MZL progression (predicts transformation)"
        ),
        "syndrome": "FAMM-Pancreatic-Syndrome-Gastric-MALT-5-10x-CDK4-6i-INACTIVE",
        "inheritance": "AD LOF heterozygous",
        "mzl_risk": "Gastric MALT 5-10x; H. pylori synergy; del9p21.3 15-20% MZL transformation; CDK4/6i INACTIVE in MZL; pancreatic EUS 40yr",
        "pathognomonic": (
            "Gastric MALT + H. pylori + germline CDKN2A = highest gastric MALT risk; eradicate H. pylori IMMEDIATELY; "
            "del9p21.3 in MZL + FAMM = CDKN2A germline panel; "
            "CDKN2A + melanoma family + gastric MALT = FAMM until proven otherwise"
        ),
        "key_avoid": (
            "AVOID CDK4/6i (palbociclib/ribociclib/abemaciclib) in MZL — INACTIVE (mechanistic failure); "
            "AVOID delaying H. pylori eradication in CDKN2A carriers — synergistic MALT driver; "
            "Do NOT miss EUS pancreatic screen from 40yr (17-38x pancreatic cancer risk)"
        ),
        "key_rule": (
            "H. pylori eradication MANDATORY in ALL CDKN2A carriers (MZL AND non-MZL); "
            "EUS pancreatic annually from 40yr or 5yr before earliest family pancreatic cancer; "
            "Annual dermoscopy (melanoma 25-36x); "
            "del9p21.3 FISH at MZL progression — guide transformation decision"
        ),
        "surveillance": (
            "Annual: dermoscopy (dermatology); H. pylori breath test; "
            "From 40yr: annual EUS pancreas; "
            "6-monthly: gastroscopy if gastric MALT diagnosed; "
            "del9p21.3 FISH at MZL diagnosis (young/familial cases)"
        ),
        "targeted_rx": (
            "Gastric MALT + H. pylori: triple-therapy eradication FIRST (amoxicillin + clarithromycin + PPI); "
            "H. pylori– gastric MALT or eradication-refractory: rituximab monotherapy or R-chl; "
            "Transformed MZL del9p21: R-CHOP ± radiotherapy AVOIDED (CDKN2A + RT risk); "
            "Consider CDK4/6i ONLY in co-existing melanoma (active in melanoma, INACTIVE in MZL)"
        ),
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 AD LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-"
            "RAD51-Filament-Formation-"
            "HBOC-NHL-MZL-1-5-2x-HRD-60pct-Cisplatin-Preferred-Olaparib-PARP-OMIM-612555"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / ~384 kDa / 13q12.3 BRCA2 MZL context: "
            "STRUCTURE: "
            "  3418 aa / ~384 kDa; homologous recombination scaffold; "
            "  BRC repeats ×8 (aa 1002-2082): RAD51 binding; "
            "  OB-fold ssDNA binding; "
            "  C-terminal RAD51 binding site (aa 3265-3330); "
            "  LOF → RAD51 filament failure → HRD (relies on NHEJ/MMEJ — error-prone); "
            "MZL MECHANISM: "
            "  BRCA2 LOF → HRD (60% GIS scar signature SBS3) → MZL 1.5-2x; "
            "  NHL broadly 2-3x; MZL slightly lower than DLBCL or CLL in BRCA2 carriers; "
            "  cisplatin-sensitive HRD pathway: platinum-based chemotherapy preferred in BRCA2-MZL; "
            "  Olaparib (PARPi): maintenance after platinum response in BRCA2-deficient lymphoma; "
            "  FA-D1 biallelic: AVOID ALL ALKYLATING ABSOLUTELY (chlorambucil, melphalan forbidden); "
            "CLINICAL RULES FOR BRCA2-MZL: "
            "  Cisplatin 70 mg/m² preferred over bendamustine for BRCA2-MZL requiring chemotherapy; "
            "  Olaparib 300 mg BID maintenance after R-cisplatin CR; "
            "  Biallelic (FA-D1): DEB/MMC chromosomal fragility test; avoid ALL alkylating; "
            "  Annual breast MRI (NO mammography primary); RRSO 35-40yr females; "
            "KEY VARIANTS: "
            "  c.5946delT (6174delT) Ashkenazi; c.886delGT Portuguese; "
            "  Exon 11 large del; c.8537_8538del; "
            "SURVEILLANCE: "
            "  Annual breast MRI from 25-30yr; RRSO 35-40yr; "
            "  Annual CECT if NHL history; "
            "  Prostate PSA males from 40yr; pancreatic EUS from 50yr; "
            "  Cascade testing first-degree relatives"
        ),
        "syndrome": "HBOC-Hereditary-Breast-Ovarian-Cancer-NHL-MZL-1-5-2x-HRD-Cisplatin-Olaparib",
        "inheritance": "AD LOF heterozygous",
        "mzl_risk": "NHL/MZL 1.5-2x; HRD 60% SBS3; cisplatin preferred over alkylating; olaparib maintenance PARP; FA-D1 biallelic AVOID ALKYLATING ABSOLUTELY",
        "pathognomonic": (
            "MZL + BRCA2 LOF + HRD GIS score >42 = cisplatin preferred; olaparib maintenance; "
            "FA-D1 (biallelic BRCA2): AML + ALL HIGHEST; lymphoma; AVOID alkylating absolutely; "
            "BRCA2 + early-onset NHL/MZL = cascade testing for first-degree relatives MANDATORY"
        ),
        "key_avoid": (
            "AVOID bendamustine/chlorambucil as FIRST choice in BRCA2-MZL (cisplatin superior — HRD); "
            "FA-D1 biallelic: AVOID ALL ALKYLATING ABSOLUTELY (cyclophosphamide, chlorambucil, melphalan); "
            "Do NOT withhold PARPi maintenance after platinum CR in BRCA2-MZL (olaparib ACTIVE)"
        ),
        "key_rule": (
            "HRD GIS score MANDATORY in BRCA2-MZL requiring systemic Rx — guides platinum use; "
            "R-cisplatin (rituximab + cisplatin 70 mg/m²) over R-bendamustine in BRCA2-MZL; "
            "Olaparib 300 mg BID maintenance after platinum CR; "
            "Annual breast MRI from 25yr; RRSO 35-40yr females"
        ),
        "surveillance": (
            "Annual: breast MRI (from 25-30yr); prostate PSA (males from 40yr); "
            "Pancreatic EUS from 50yr; "
            "CECT chest/abdomen/pelvis annually if NHL history; "
            "RRSO 35-40yr females (BRCA2 ovarian cancer risk 10-20%)"
        ),
        "targeted_rx": (
            "BRCA2-MZL systemic: R-cisplatin (rituximab 375 + cisplatin 70 mg/m² ×6 cycles); "
            "Olaparib 300 mg BID maintenance post-platinum CR; "
            "R-bendamustine if cisplatin contraindicated (CKD/peripheral neuropathy); "
            "FA-D1 biallelic: fludarabine-based RIC-HSCT (NO alkylating/TBI); haematology specialist mandatory"
        ),
    },
    {
        "gene": "CARD11",
        "protein": (
            "CARD11 -- 7p22.2 AD LOF-GOF -- 1154aa -- "
            "CARD11-133kDa-CBM-Complex-"
            "BCL10-MALT1-NF-kB-Activation-"
            "BENTA-GOF-MZL-MALT-NF-kB-Venetoclax-LOF-CID-EBV-NHL-OMIM-616098"
        ),
        "locus": "7p22.2",
        "protein_size": (
            "1154 aa / ~133 kDa / 7p22.2 CARD11 MZL context: "
            "STRUCTURE: "
            "  1154 aa / ~133 kDa; signalling scaffold in BCR/TCR pathway; "
            "  CARD domain (aa 1-100): homotypic BCL10 interaction; "
            "  Coiled-coil domain (aa 109-272): GOF mutations here → BENTA; "
            "  MAGUK domain (PDZ/SH3/GUK): scaffold; "
            "  BCR → CARD11-BCL10-MALT1 (CBM) complex → canonical NF-κB → B-cell survival; "
            "  GOF (coiled-coil): constitutive NF-κB → B-cell expansion → BENTA disease; "
            "  LOF: blocked BCR NF-κB signalling → combined immunodeficiency (CID); "
            "BENTA/MZL MECHANISM: "
            "  BENTA (B-cell Expansion with NF-κB and T-cell Anergy): CARD11 GOF; "
            "  Constitutive NF-κB → marginal zone B-cell proliferation → extranodal MZL/MALT; "
            "  NF-κB overactivation → BCL2 upregulation → venetoclax sensitivity; "
            "  MALT1 paracaspase activity hyperactive in CARD11 GOF; "
            "  CARD11 LOF: EBV-driven B-cell lymphoproliferation (CID → NHL); "
            "  CARD11 somatic GOF: 10-15% DLBCL ABC subtype, 5-8% MZL (activating B1-domain); "
            "CLINICAL RULES FOR CARD11-MZL: "
            "  CARD11 GOF MZL: venetoclax 400 mg OD preferred (BCL2 upregulated by NF-κB); "
            "  MALT1 inhibitor (JNJ-67856633, MPT-0G211): Phase 1-2 in CARD11 GOF MZL/MALT; "
            "  Ibrutinib BTKi: effective upstream (BTK → CARD11 activation); "
            "  CARD11 LOF (CID): HSCT curative; NHL managed with IVIG + antivirals; "
            "KEY VARIANTS: "
            "  GOF BENTA: G116S, G116E, F123I, R179H, E127K (coiled-coil domain); "
            "  LOF CID: truncating/missense in CARD/MAGUK; "
            "  Somatic activating: L225LI, L225LV (B1 domain), L244P; "
            "SURVEILLANCE: "
            "  Annual CBC + B-cell flow cytometry (BENTA); "
            "  EBV viral load q6M (LOF); splenomegaly US annually; "
            "  Lymph node biopsy for any >2cm node >6 weeks"
        ),
        "syndrome": "BENTA-Disease-CARD11-GOF-NF-kB-MZL-MALT-Venetoclax-MALT1-Inhibitor",
        "inheritance": "AD GOF (BENTA); AD/AR LOF (CID); somatic GOF in MZL/DLBCL",
        "mzl_risk": "CARD11 GOF constitutive NF-κB → MZL/MALT extranodal; BCL2 upregulated → venetoclax preferred; MALT1 inhibitor investigational; LOF CID → EBV-NHL",
        "pathognomonic": (
            "BENTA + extranodal mass = MZL/MALT PATHOGNOMONIC; CBM complex overactivation; "
            "CARD11 GOF + BCL2 overexpression = venetoclax preferred over standard R-CHOP; "
            "CARD11 LOF + EBV+ B-cell mass = CID-NHL; HSCT referral URGENT"
        ),
        "key_avoid": (
            "AVOID R-CHOP anthracycline-first in CARD11 GOF MZL if venetoclax available (BCL2 driven); "
            "AVOID ibrutinib dose reduction in CARD11 GOF MZL (BTKi active upstream — full dose needed); "
            "Do NOT miss MALT1 inhibitor trial eligibility in CARD11 GOF relapsed MZL"
        ),
        "key_rule": (
            "CARD11 GOF MZL: venetoclax 400 mg OD ± rituximab (VR) as preferred upfront; "
            "MALT1 inhibitor enroll: JNJ-67856633 or MPT-0G211 if CARD11 GOF confirmed; "
            "Ibrutinib 420 mg OD (BTKi) active in CARD11 GOF MZL; "
            "CARD11 LOF CID-NHL: HSCT referral URGENT — curative"
        ),
        "surveillance": (
            "Annual: CBC + B-cell flow cytometry; EBV viral load (q6M in LOF); "
            "Splenomegaly US annually; lymph node exam; "
            "BM biopsy if: unexplained cytopenia, splenomegaly >18cm, B symptoms; "
            "BENTA: annual echo (cardiomyopathy risk in CARD11 GOF)"
        ),
        "targeted_rx": (
            "CARD11 GOF MZL: venetoclax 400 mg OD + rituximab (VR); "
            "Ibrutinib 420 mg OD (BTKi) ± venetoclax; "
            "MALT1 inhibitor JNJ-67856633 Phase 1-2 (CARD11 GOF, trial enroll); "
            "CARD11 LOF CID-NHL: HSCT allo (preferred) + antiviral prophylaxis (valganciclovir)"
        ),
    },
    {
        "gene": "PIK3CD",
        "protein": (
            "PIK3CD -- 1p36.22 AD GOF -- 1044aa -- "
            "PI3K-delta-119kDa-Catalytic-Subunit-"
            "PI3K-AKT-mTOR-FOXO1-"
            "APDS1-EBV-MZL-MALT-10-15pct-Leniolisib-FDA2023-Sirolimus-Bridge-OMIM-615513"
        ),
        "locus": "1p36.22",
        "protein_size": (
            "1044 aa / ~119 kDa / 1p36.22 PIK3CD MZL context: "
            "STRUCTURE: "
            "  1044 aa / ~119 kDa; catalytic subunit of PI3K class IA δ-isoform; "
            "  Adaptor-binding domain (ABD): p85 regulatory subunit interaction; "
            "  Ras-binding domain (RBD): Ras-GTPase activation; "
            "  C2 domain: membrane targeting; "
            "  Helical domain: regulatory; "
            "  Kinase domain: PIP2 → PIP3 conversion; "
            "  GOF variants → constitutive PIP3 → AKT/mTOR/FOXO1 hyperactivation; "
            "  APDS1 = Activated PI3K Delta Syndrome 1 — primary immunodeficiency; "
            "APDS1/MZL MECHANISM: "
            "  PIK3CD GOF → PI3Kδ constitutive → impaired B/T-cell terminal differentiation; "
            "  EBV+ B-cell lymphoproliferation (EBV viral load key monitor); "
            "  APDS1 NHL: predominantly MZL/MALT and Hodgkin lymphoma (EBV-driven); "
            "  MZL/MALT in 10-15% of APDS1 patients — especially extranodal (salivary/gastric/ocular); "
            "  mTORC1/FOXO1 hyperactivation → marginal zone B-cell survival; "
            "  Leniolisib (PI3Kδ selective inhibitor): FDA2023 for APDS1; "
            "  Sirolimus (mTOR inhibitor): bridge therapy; partial response in APDS1 lymphoproliferation; "
            "CLINICAL RULES FOR PIK3CD-MZL: "
            "  Leniolisib 70 mg BID: FDA-approved (2023) for APDS1 (≥12yr); preferred upfront; "
            "  Sirolimus 2-4 mg/day: mTOR bridge if leniolisib unavailable; "
            "  EBV viral load MANDATORY q3-6M (EBV-driven lymphoma risk); "
            "  Rituximab: effective in EBV+ MZL in APDS1; "
            "  HSCT allo-RIC: curative for APDS1 + refractory MZL; "
            "KEY VARIANTS: "
            "  E1021K (most common GOF, exon 24); C416R; N334K; E525K; "
            "  All GOF: helical or kinase domain; "
            "SURVEILLANCE: "
            "  EBV viral load q3M first 2yr, then q6M; "
            "  Annual CECT or PET-CT; IgG level q3M (hypogammaglobulinaemia); "
            "  Pulmonary: HRCT annual (bronchiectasis in APDS1); "
            "  CMV viral load q3M"
        ),
        "syndrome": "APDS1-Activated-PI3K-Delta-Syndrome-EBV-MZL-MALT-10-15pct-Leniolisib-FDA2023",
        "inheritance": "AD GOF heterozygous",
        "mzl_risk": "MZL/MALT 10-15% in APDS1; EBV-driven; extranodal salivary/gastric/ocular; leniolisib FDA2023; sirolimus bridge; HSCT curative for refractory",
        "pathognomonic": (
            "APDS1 + EBV+ mass = MZL/MALT PATHOGNOMONIC; leniolisib FIRST-LINE; "
            "PIK3CD GOF + salivary/parotid enlargement = extranodal MALT until proven otherwise; "
            "APDS1 + rising EBV viral load = lymphoma screen URGENT (MZL/HL)"
        ),
        "key_avoid": (
            "AVOID standard CHOP chemotherapy upfront in APDS1 MZL — leniolisib FIRST (PI3Kδ-driven); "
            "AVOID IVIG without IgG monitoring (PIK3CD GOF → hypo-gammaglobulinaemia — IVIG needed); "
            "Do NOT delay HSCT referral in relapsed PIK3CD MZL — curative window narrow"
        ),
        "key_rule": (
            "Leniolisib 70 mg BID FDA2023 FIRST-LINE for APDS1 MZL; "
            "EBV viral load MANDATORY q3M — rising viral load = lymphoma screen (PET-CT); "
            "Sirolimus 2-4 mg/day bridge if leniolisib delayed; "
            "HSCT allo-RIC for APDS1 + refractory MZL — curative intent"
        ),
        "surveillance": (
            "q3M: EBV viral load; IgG level; CMV viral load; "
            "Annual: CECT/PET-CT; HRCT chest (bronchiectasis); CBC; "
            "BM biopsy if: EBV >2000 IU/mL + splenomegaly; "
            "Ophthalmology annual (ocular adnexal MALT risk in APDS1)"
        ),
        "targeted_rx": (
            "Leniolisib 70 mg BID (FDA2023): first-line APDS1 MZL (PI3Kδ inhibitor); "
            "Sirolimus 2-4 mg/day: mTOR bridge or alternative; "
            "Rituximab 375 mg/m² ×4-6: EBV+ MZL in APDS1; "
            "HSCT allo-RIC (fludarabine-based): curative for APDS1 + refractory MZL"
        ),
    },
]

# ─── Key Clinical Distinctions ─────────────────────────────────────────────────
KEY_CLINICAL_DISTINCTIONS = [
    "MYD88 L265P MALT — H. pylori eradication FAILS — rituximab MANDATORY UPFRONT: "
    "L265P somatic in 5-15% MALT lymphoma; "
    "L265P+ gastric MALT does NOT respond to H. pylori antibiotics alone (resistance); "
    "Standard approach (H. pylori eradication 75-80% CR) FAILS in L265P+ MALT; "
    "L265P+ → rituximab-based Rx mandatory upfront (skip antibiotic monotherapy); "
    "BRAF V600E co-test mandatory in orbital/thyroid MALT (hairy cell DDx); "
    "CONTRAST with MYD88 L265P in WM/LPL (90% WM — different disease, ibrutinib preferred there).",

    "TNFRSF13B (TACI) CVID2 — NHL 10-12x HIGHEST — IVIG NEVER STOP — rituximab WITHOUT IVIG = DEATH: "
    "TNFRSF13B LOF = HIGHEST known germline NHL risk gene (10-12x; higher than ATM 4-5x); "
    "MZL/MALT most common lymphoma subtype in CVID2; "
    "IVIG MANDATORY every 3-4wk (IgG trough >7 g/L) — NEVER stop without specialist approval; "
    "Rituximab depletes B-cells → IgG will collapse to life-threatening levels if IVIG not co-given; "
    "Rituximab + IVIG co-administration MANDATORY in CVID2 lymphoma; "
    "CONTRAST with normal NHL: rituximab alone is standard — CVID2 is the life-threatening exception.",

    "ATM MZL — RADIOSENSITIVITY ABSOLUTE — NO RT EVER — ceralasertib ATRi synthetic lethality: "
    "ATM germline LOF het: NHL 4-5x; MZL/splenic MZL 3-4x; del11q22.3 somatic 15-25% MZL; "
    "RADIOSENSITIVITY ABSOLUTE — no safe RT dose threshold exists in ATM LOF; "
    "NO involved-field RT, NO consolidation RT, NO TBI for HSCT conditioning; "
    "Ceralasertib (AZD6738 ATRi) Phase 2 ATM-null lymphoma — synthetic lethality; "
    "CONTRAST with TP53 MZL: both RT-forbidden, but different mechanisms and Rx pathways.",

    "CDKN2A Gastric MALT — H. pylori SYNERGY — CDK4/6i INACTIVE in MZL — EUS from 40yr: "
    "CDKN2A LOF + H. pylori co-exposure = 5-10x gastric MALT risk (synergistic); "
    "H. pylori eradication MANDATORY in ALL CDKN2A carriers (prevent MALT); "
    "del9p21.3 somatic in 15-20% MZL transformation — FISH at progression; "
    "CDK4/6i (palbociclib/ribociclib/abemaciclib) INACTIVE in MZL — mechanistic failure; "
    "CONTRAST with melanoma (CDKN2A carrier): CDK4/6i ACTIVE in melanoma, INACTIVE in lymphoma — "
    "never extrapolate cancer-type agnostic CDK4/6i use.",

    "CARD11 GOF BENTA — NF-κB Constitutive — Venetoclax PREFERRED — MALT1 inhibitor Trial: "
    "CARD11 GOF = BENTA disease; constitutive NF-κB → BCL2 upregulation → MZL/MALT extranodal; "
    "Venetoclax preferred (400 mg OD + rituximab) over R-CHOP (BCL2-driven); "
    "MALT1 inhibitor (JNJ-67856633): Phase 1-2 in CARD11 GOF MZL — enroll eligible patients; "
    "CARD11 LOF (CID): EBV-NHL → HSCT URGENT (curative); "
    "CONTRAST with CARD11 in DLBCL (see DLBCL atlas): same gene, same GOF/LOF distinction, "
    "but MZL-specific consideration is MALT1 inhibitor eligibility (DLBCL uses venetoclax+R).",

    "PIK3CD APDS1 MZL — Leniolisib FDA2023 FIRST-LINE — EBV viral load MANDATORY q3M: "
    "PIK3CD GOF = APDS1; EBV-driven lymphoproliferation → MZL/MALT 10-15% in APDS1; "
    "Leniolisib 70 mg BID FDA2023 (first PI3Kδ-specific inhibitor approved) — FIRST-LINE; "
    "EBV viral load MANDATORY q3M — rising EBV = lymphoma screen URGENT (PET-CT); "
    "Sirolimus 2-4 mg/day bridge if leniolisib delayed; "
    "CONTRAST with MCL/DLBCL PIK3CD context: in APDS1 MZL, leniolisib is the targeted Rx; "
    "in non-APDS1 MZL, PI3K inhibition is not standard (copanlisib used in indolent NHL broadly).",

    "BRCA2 MZL — HRD 60% — Cisplatin Preferred OVER Bendamustine — Olaparib Maintenance: "
    "BRCA2 LOF: HRD score 60% (GIS SBS3); cisplatin-sensitive; "
    "R-cisplatin (rituximab + cisplatin 70 mg/m²) preferred over R-bendamustine for BRCA2-MZL; "
    "Olaparib 300 mg BID maintenance after platinum CR (PARP inhibitor); "
    "FA-D1 (biallelic BRCA2): AVOID ALL ALKYLATING ABSOLUTELY (chlorambucil/bendamustine/melphalan); "
    "CONTRAST with non-BRCA2 MZL: R-bendamustine is standard; only in BRCA2 does HRD guide "
    "cisplatin preference — most oncologists not aware of this intersection.",

    "TP53 LFS MZL — AVOID RADIATION ABSOLUTELY — WB-MRI Toronto — Venetoclax Backbone: "
    "TP53 germline LOF = LFS; NHL/MZL rare (2% cumulative) but transforms aggressively; "
    "AVOID RADIATION ABSOLUTELY — therapy-related sarcoma/AML risk in LFS; "
    "WB-MRI Toronto protocol MANDATORY annual (no CT, no PET/CT for LFS surveillance); "
    "del17p somatic at transformation: venetoclax 400 mg OD backbone; "
    "CONTRAST with ATM MZL: both RT-forbidden, but LFS surveillance is WB-MRI; "
    "ATM surveillance allows PET-CT (ionising acceptable for non-LFS carriers).",
]


def _make_cohort(gene_def: dict, seed: int) -> dict:
    """Generate a 40-patient MZL cohort for one gene."""
    rng = random.Random(seed)

    gene   = gene_def["gene"]
    locus  = gene_def["locus"]
    inh    = gene_def["inheritance"]
    protein_size = gene_def["protein_size"]

    n = 40
    ages = sorted([rng.gauss(54, 13) for _ in range(n)])
    ages = [max(22, min(85, round(a, 1))) for a in ages]

    # MZL subtype mix (MALT ~50%, Nodal ~25%, Splenic ~25%)
    subtypes = rng.choices(
        ["Extranodal MALT", "Nodal MZL", "Splenic MZL"],
        weights=[0.50, 0.25, 0.25],
        k=n,
    )

    # H. pylori status (mostly gastric MALT context)
    hp_pos = [rng.random() < 0.45 for _ in range(n)]

    # EBV status (relevant for PIK3CD, CARD11 LOF, TNFRSF13B)
    ebv_rates = {
        "MYD88": 0.08, "TNFRSF13B": 0.30, "ATM": 0.10,
        "TP53": 0.12, "CDKN2A": 0.08, "BRCA2": 0.08,
        "CARD11": 0.25, "PIK3CD": 0.40,
    }
    ebv_pos = [rng.random() < ebv_rates.get(gene, 0.10) for _ in range(n)]

    # MYD88 L265P somatic rate
    l265p_rates = {
        "MYD88": 0.55, "TNFRSF13B": 0.10, "ATM": 0.08,
        "TP53": 0.06, "CDKN2A": 0.06, "BRCA2": 0.06,
        "CARD11": 0.12, "PIK3CD": 0.08,
    }
    l265p = [rng.random() < l265p_rates.get(gene, 0.08) for _ in range(n)]

    # del11q somatic
    del11q = [rng.random() < 0.20 for _ in range(n)]

    # Transformation to DLBCL
    transform_rates = {
        "MYD88": 0.15, "TNFRSF13B": 0.20, "ATM": 0.18,
        "TP53": 0.28, "CDKN2A": 0.20, "BRCA2": 0.15,
        "CARD11": 0.22, "PIK3CD": 0.18,
    }
    transformed = [rng.random() < transform_rates.get(gene, 0.18) for _ in range(n)]

    # Remission (CR/PR)
    cr_rates = {
        "MYD88": 0.78, "TNFRSF13B": 0.65, "ATM": 0.70,
        "TP53": 0.60, "CDKN2A": 0.75, "BRCA2": 0.72,
        "CARD11": 0.68, "PIK3CD": 0.72,
    }
    cr = [rng.random() < cr_rates.get(gene, 0.70) for _ in range(n)]

    # 5yr OS
    os5_rates = {
        "MYD88": 0.72, "TNFRSF13B": 0.60, "ATM": 0.65,
        "TP53": 0.55, "CDKN2A": 0.70, "BRCA2": 0.68,
        "CARD11": 0.62, "PIK3CD": 0.65,
    }
    os5 = [rng.random() < os5_rates.get(gene, 0.65) for _ in range(n)]

    return {
        "gene":            gene,
        "locus":           locus,
        "inheritance":     inh,
        "protein_size":    protein_size,
        "n":               n,
        "avg_age_dx":      round(sum(ages) / n, 1),
        "subtype_malt_n":  sum(1 for s in subtypes if s == "Extranodal MALT"),
        "subtype_nodal_n": sum(1 for s in subtypes if s == "Nodal MZL"),
        "subtype_splenic_n": sum(1 for s in subtypes if s == "Splenic MZL"),
        "hp_pos_n":        sum(hp_pos),
        "ebv_pos_n":       sum(ebv_pos),
        "l265p_n":         sum(l265p),
        "del11q_n":        sum(del11q),
        "transformed_n":   sum(transformed),
        "cr_n":            sum(cr),
        "os5_n":           sum(os5),
        "key_avoid":       gene_def["key_avoid"],
        "key_rule":        gene_def["key_rule"],
        "targeted_rx":     gene_def["targeted_rx"],
        "mzl_risk":        gene_def["mzl_risk"],
        "pathognomonic":   gene_def["pathognomonic"],
        "surveillance":    gene_def["surveillance"],
    }


def generate_overview() -> dict:
    cohorts = {
        gd["gene"]: _make_cohort(gd, SEED_BASE + i)
        for i, gd in enumerate(ATLAS_GENES)
    }
    total = 320
    cr_total    = sum(c["cr_n"] for c in cohorts.values())
    os5_total   = sum(c["os5_n"] for c in cohorts.values())
    transform_t = sum(c["transformed_n"] for c in cohorts.values())
    hp_total    = sum(c["hp_pos_n"] for c in cohorts.values())
    ebv_total   = sum(c["ebv_pos_n"] for c in cohorts.values())
    l265p_total = sum(c["l265p_n"] for c in cohorts.values())
    del11q_t    = sum(c["del11q_n"] for c in cohorts.values())
    malt_t      = sum(c["subtype_malt_n"] for c in cohorts.values())
    nodal_t     = sum(c["subtype_nodal_n"] for c in cohorts.values())
    splenic_t   = sum(c["subtype_splenic_n"] for c in cohorts.values())

    return {
        "atlas":           "Hereditary-MZL-Predisposition-Atlas",
        "seed_range":      f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":  total,
        "cr_n":            cr_total,
        "cr_pct":          round(cr_total / total * 100, 1),
        "os5_n":           os5_total,
        "os5_pct":         round(os5_total / total * 100, 1),
        "transformed_n":   transform_t,
        "transformed_pct": round(transform_t / total * 100, 1),
        "hp_pos_n":        hp_total,
        "hp_pos_pct":      round(hp_total / total * 100, 1),
        "ebv_pos_n":       ebv_total,
        "ebv_pos_pct":     round(ebv_total / total * 100, 1),
        "l265p_n":         l265p_total,
        "l265p_pct":       round(l265p_total / total * 100, 1),
        "del11q_n":        del11q_t,
        "del11q_pct":      round(del11q_t / total * 100, 1),
        "malt_n":          malt_t,
        "malt_pct":        round(malt_t / total * 100, 1),
        "nodal_mzl_n":     nodal_t,
        "nodal_mzl_pct":   round(nodal_t / total * 100, 1),
        "splenic_mzl_n":   splenic_t,
        "splenic_mzl_pct": round(splenic_t / total * 100, 1),
        "gene_counts":     {
            gene: {
                "n":             c["n"],
                "cr_n":          c["cr_n"],
                "os5_n":         c["os5_n"],
                "transformed_n": c["transformed_n"],
                "hp_pos_n":      c["hp_pos_n"],
                "ebv_pos_n":     c["ebv_pos_n"],
                "l265p_n":       c["l265p_n"],
                "del11q_n":      c["del11q_n"],
                "malt_n":        c["subtype_malt_n"],
                "splenic_n":     c["subtype_splenic_n"],
                "mzl_risk":      c["mzl_risk"],
            }
            for gene, c in cohorts.items()
        },
    }


def generate_breakdown() -> dict:
    return {
        "atlas":     "Hereditary-MZL-Predisposition-Atlas",
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

    key_clinical_distinctions = KEY_CLINICAL_DISTINCTIONS

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"MZL RISK: {gd['mzl_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-MZL-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": key_clinical_distinctions,
    }
