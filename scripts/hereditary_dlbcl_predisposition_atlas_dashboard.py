#!/usr/bin/env python3
"""Hereditary-DLBCL-Predisposition-Atlas -- Complete 8-Gene Reference
TP53    (p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome (OMIM 151623);
         DLBCL 5-8x lifetime risk; complex karyotype 60%;
         AVOID RADIATION ABSOLUTELY; WB-MRI Toronto; APR-246/eprenetapopt;
         seed SEED_BASE+0).
ATM     (ATM kinase; 3056aa; 11q22.3; AD LOF;
         Ataxia-Telangiectasia heterozygous carrier (OMIM 208900);
         NHL/DLBCL 4-5x; del11q somatic 40%;
         RADIOSENSITIVITY ABSOLUTE — no RT ever; ceralasertib ATRi;
         seed SEED_BASE+1).
BRCA2   (BRCA2; 3418aa; 13q12.3; AD LOF;
         Hereditary Breast and Ovarian Cancer (OMIM 612555);
         NHL/DLBCL 2-3x; HRD positive 60%;
         PARP inhibitor sensitive; cisplatin preferred;
         seed SEED_BASE+2).
CHEK2   (CHK2; 543aa; 22q12.1; AD LOF;
         CHEK2-related cancer predisposition;
         1100delC European (1 in 200); IVS2+1G>A Ashkenazi;
         NHL 2x; additive with ATM; breast cancer concurrent 40%;
         seed SEED_BASE+3).
PIK3CD  (PI3K catalytic subunit delta; 1044aa; 1p36.22; AD GOF;
         Activated PI3K-delta Syndrome 1 / APDS1 (OMIM 615513);
         EBV-driven lymphoproliferation/DLBCL; Leniolisib FDA2023;
         EBV positivity 70%; PI3Kδ inhibitor curative;
         seed SEED_BASE+4).
KMT2D   (Lysine methyltransferase 2D / MLL4; 5537aa; 12q13.12; AD LOF;
         Kabuki Syndrome 1 (OMIM 147920);
         B-cell lymphoma predisposition; H3K4me3 epigenetic dysregulation;
         EZH2 inhibitor tazemetostat sensitivity;
         seed SEED_BASE+5).
CARD11  (CARD11; 1154aa; 7p22.2; AD LOF/GOF;
         BENTA disease (OMIM 616452);
         NF-κB constitutive → B-cell lymphoma; GOF → Venetoclax preferred;
         LOF → atopic disease; MALT1 inhibitor JNJ-67856633 GOF;
         seed SEED_BASE+6).
TNFRSF13B (TACI; 293aa; 17p11.2; AD;
         CVID-2 (OMIM 607892);
         NHL/DLBCL 10-12x lifetime risk; C104R/A181E founders;
         IVIG replacement; rituximab + lenalidomide for lymphoma;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3582-3589)
"""
import random

SEED_BASE = 3582

ATLAS_GENES = [
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "LFS-Li-Fraumeni-DLBCL-5-8x-"
            "Complex-Karyotype-60pct-"
            "AVOID-RADIATION-ABSOLUTELY-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 DLBCL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; tumour suppressor; tetramer; "
            "  N-terminal transactivation domain (aa 1-67); proline-rich region; "
            "  DNA-binding domain (aa 100-290) — 90% of all cancer mutations; "
            "  Linker + tetramerisation domain (aa 323-356); C-terminal reg domain; "
            "  p53 acts as sequence-specific TF → apoptosis, cell-cycle arrest, DNA repair, senescence; "
            "  LOF: dominant-negative effect (mut-TP53 tetramers sequester WT p53); "
            "LI-FRAUMENI DLBCL MECHANISM: "
            "  Germline TP53 LOF → impaired G1 checkpoint → accumulation of DNA damage; "
            "  DLBCL risk 5-8x lifetime; complex karyotype in 60% of TP53-germline DLBCL; "
            "  ELN Adverse equivalent: TP53 del17p/mut in sporadic DLBCL = worst prognosis; "
            "  Germline TP53 → DLBCL typically younger onset (mean 35-45 yr) vs sporadic (65-70 yr); "
            "  Secondary DLBCL after prior therapy: TP53-germline patients at 20-40% 10-yr risk; "
            "CLINICAL RULES FOR TP53-DLBCL: "
            "  AVOID RADIATION ABSOLUTELY — even low-dose RT triggers secondary malignancies in LFS; "
            "  WB-MRI surveillance Toronto protocol: Annual WB-MRI + brain MRI; "
            "  APR-246 (eprenetapopt; PRIMA-1MET): reactivates mut-TP53 → apoptosis; "
            "  Venetoclax: BCL-2 upregulated in TP53-DLBCL → venetoclax combination active; "
            "  R-CHOP alternative: R-ICE or polatuzumab-BR in TP53-DLBCL; "
            "  Allo-SCT in CR2: preferred consolidation to avoid RT; "
            "KEY VARIANTS: "
            "  Dominant-negative: R175H, G245S, R248W, R248Q, R249S, R273H, R282W; "
            "  LOF (truncating): splice-site + frameshift + stopgain; "
            "  R337H Brazilian founder: 0.3% south Brazil population; "
            "SURVEILLANCE: "
            "  Annual WB-MRI + brain MRI + abdominal US; CBC q6M; "
            "  No routine PET-CT (radiation exposure); MRI preferred; "
            "  Cascade testing ALL first-degree relatives MANDATORY; "
        ),
        "syndrome": "LFS-Li-Fraumeni-Syndrome",
        "inheritance": "AD LOF heterozygous",
        "dlbcl_risk": "DLBCL 5-8x lifetime; complex karyotype 60%; younger onset 35-45 yr",
        "pathognomonic": (
            "Complex karyotype DLBCL + young age + personal/family history soft tissue sarcoma "
            "or brain tumour or breast cancer = TP53 germline testing MANDATORY; "
            "p53 IHC: >50% nuclear staining OR complete absence (null pattern) = TP53 mutation proxy"
        ),
        "key_avoid": (
            "AVOID RADIATION ABSOLUTELY (including low-dose RT, involved-field RT); "
            "AVOID alkylating agents without careful benefit-risk (secondary malignancy risk); "
            "AVOID surveillance PET-CT (radiation exposure in LFS increases second-cancer risk)"
        ),
        "key_rule": (
            "AVOID RADIATION ABSOLUTELY TP53/LFS — even salvage RT contraindicated; "
            "WB-MRI Toronto annual surveillance protocol MANDATORY; "
            "Allo-SCT preferred over RT-based consolidation in TP53-germline DLBCL"
        ),
        "surveillance": (
            "Annual WB-MRI + brain MRI (Toronto protocol); abdominal ultrasound q6M; "
            "CBC + LDH q6M; NO PET-CT (radiation); "
            "Cascade genetic testing all 1st-degree relatives; "
            "Breast MRI annually from age 20 (females)"
        ),
        "targeted_rx": (
            "APR-246/Eprenetapopt (mut-TP53 reactivator; phase 3); "
            "Venetoclax (BCL-2 elevated in TP53-DLBCL; combo R-ICE+venetoclax); "
            "Polatuzumab-BR (CD79b-ADC + BR; TP53-DLBCL salvage); "
            "Allo-SCT CR1/CR2 preferred consolidation (avoids RT)"
        ),
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD LOF -- 3056aa -- "
            "ATM-350kDa-PIK3-Kinase-"
            "A-T-Monoallelic-NHL-DLBCL-4-5x-"
            "RADIOSENSITIVITY-ABSOLUTE-"
            "Ceralasertib-ATRi-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / ~350 kDa / 11q22.3 ATM DLBCL molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; serine/threonine kinase; PI3K-related kinase (PIKK) family; "
            "  HEAT repeats (N-terminal); FAT domain; kinase domain; FATC domain; "
            "  Activated by DSB (double-strand breaks): MRN complex (MRE11-RAD50-NBS1) recruits ATM; "
            "  Autophosphorylation Ser1981 → activation → phosphorylates H2AX, CHK2, BRCA1, p53; "
            "  ATM null → impaired DSB repair → genomic instability; "
            "MONOALLELIC ATM-DLBCL MECHANISM: "
            "  Heterozygous ATM LOF → NHL/DLBCL 4-5x lifetime risk; "
            "  del11q22 (somatic ATM deletion) in 40% DLBCL at diagnosis; "
            "  Biallelic loss (germline + somatic hit) in tumour → complete ATM null; "
            "  RADIOSENSITIVITY ABSOLUTE in monoallelic carriers: increased RT toxicity; "
            "  Ceralasertib (ATR inhibitor; AZD6738): synthetic lethality in ATM-null DLBCL; "
            "  ATM-deficient DLBCL: more sensitive to PARP inhibitors, DNA-PKcs inhibitors; "
            "CLINICAL RULES ATM-DLBCL: "
            "  RADIOSENSITIVITY ABSOLUTE — no involved-field RT, no TBI; "
            "  Avoid TBI conditioning for HSCT; use RIC (fludarabine-cyclophosphamide); "
            "  Test ATM kinase function (kinase assay) and protein expression (IHC) in tumour; "
            "  Ceralasertib (ATRi): clinical trials for ATM-null DLBCL (AZD6738); "
            "KEY VARIANTS: "
            "  Moderate-risk truncating: c.7271T>G, IVS-10 A>G; "
            "  European founder: c.3161C>A (p.Ser1054Ter); "
            "  Missense loss-of-function: c.7517A>G, c.5932G>T; "
        ),
        "syndrome": "A-T-Heterozygous-NHL-DLBCL-Predisposition",
        "inheritance": "AD LOF heterozygous",
        "dlbcl_risk": "NHL/DLBCL 4-5x; del11q somatic 40%; radiosensitivity absolute",
        "pathognomonic": (
            "del11q22 somatic + germline ATM LOF = biallelic ATM null in tumour; "
            "ATM protein absent on IHC + germline ATM sequencing; "
            "IgA deficiency in biallelic A-T (not monoallelic) — not a marker for DLBCL predisposition"
        ),
        "key_avoid": (
            "RADIOSENSITIVITY ABSOLUTE — no involved-field RT, no TBI, no WART; "
            "AVOID TBI-based conditioning for HSCT (use RIC fludarabine-cyclophosphamide); "
            "AVOID bleomycin (pulmonary toxicity additive with ATM deficiency)"
        ),
        "key_rule": (
            "RADIOSENSITIVITY ABSOLUTE ATM — no RT ever; "
            "Ceralasertib (ATRi) synthetic lethality in ATM-null tumour; "
            "del11q DLBCL: check germline ATM — biallelic loss common"
        ),
        "surveillance": (
            "Annual CT chest/abdomen/pelvis (if RT avoided); CBC + LDH q6M; "
            "Breast MRI annually females 30+ (4-5x breast cancer risk); "
            "Colonoscopy q5yr from age 40; "
            "Cascade genetic testing all 1st-degree relatives"
        ),
        "targeted_rx": (
            "Ceralasertib (AZD6738; ATRi; synthetic lethality ATM-null); "
            "M3814/Peposertib (DNA-PKcs inhibitor + R-CHOP); "
            "PARP inhibitors (olaparib; ATM-null synthetic lethality); "
            "R-CHOP standard; dose-reduce radiation — AVOID entirely"
        ),
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 AD LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-"
            "HBOC-NHL-DLBCL-2-3x-"
            "HRD-PARP-Sensitive-"
            "Cisplatin-Preferred-OMIM-612555"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / ~384 kDa / 13q12.3 BRCA2 DLBCL molecular context: "
            "STRUCTURE: "
            "  3418 aa / ~384 kDa; HR scaffold; "
            "  OB folds (BRCA2 BRC repeats aa 1002-2667): 8x RAD51 binding; "
            "  C-terminal DNA-binding domain (aa 2396-3186); "
            "  N-terminal BRCA1-interaction region; "
            "  BRCA2 loads RAD51 onto ssDNA at DSB → homologous recombination; "
            "  BRCA2 LOF → HRD (homologous recombination deficiency); "
            "BRCA2-NHL/DLBCL MECHANISM: "
            "  BRCA2 heterozygous LOF → NHL/DLBCL 2-3x lifetime risk; "
            "  HRD signature: SBS3 (C>T at NCN context) + INDEL ID6 in tumour; "
            "  BRCA2-DLBCL: cisplatin-based regimens more effective (HRD exploits); "
            "  PARP inhibitor sensitivity (olaparib, niraparib, talazoparib): traps PARP-DNA complex; "
            "  Biallelic BRCA2 = Fanconi Anaemia type D1 (FA-D1): highest childhood AML/ALL risk; "
            "CLINICAL RULES BRCA2-DLBCL: "
            "  R-CHOP + cisplatin (RCHOP-esc: R-DHAP alternating): HRD exploited; "
            "  Olaparib maintenance post-CR: emerging data for HRD+ DLBCL; "
            "  Avoid alkylating agents in FA-D1 (biallelic) — absolutely CI; "
            "  BRCA2 concurrent breast/ovarian cancer in 15-20% of BRCA2-DLBCL patients; "
            "KEY VARIANTS: "
            "  6174delT (Ashkenazi Jewish founder, 1%); c.9976A>T (p.Lys3326Ter pseudo-pathogenic); "
            "  c.5946delT Slavic founder; Spanish founder c.3036delAAAC; "
        ),
        "syndrome": "HBOC-BRCA2-NHL-Predisposition",
        "inheritance": "AD LOF heterozygous",
        "dlbcl_risk": "NHL/DLBCL 2-3x; HRD positive 60%; cisplatin-sensitive; concurrent breast/ovarian 15-20%",
        "pathognomonic": (
            "HRD signature SBS3 + ID6 in tumour + germline BRCA2 = BRCA2-DLBCL; "
            "HRD scar genomic score (Myriad myChoice / FoundationOne): ≥42 = HRD positive; "
            "BRCA2 DLBCL with concurrent HBOC history in 15-20%"
        ),
        "key_avoid": (
            "AVOID alkylating agents in FA-D1 (biallelic BRCA2) ABSOLUTELY; "
            "AVOID RT in FA-D1 ABSOLUTELY; "
            "CAUTION with cumulative anthracycline if concurrent breast cancer history"
        ),
        "key_rule": (
            "Cisplatin-based salvage preferred BRCA2-DLBCL (HRD exploited); "
            "Olaparib maintenance post-CR emerging for HRD+ DLBCL; "
            "Screen for concurrent HBOC (breast/ovarian) at DLBCL diagnosis"
        ),
        "surveillance": (
            "Annual breast MRI + mammography (females); "
            "Annual ovarian USS + CA125 (females 30+); "
            "Annual prostate PSA (males 40+); "
            "Pancreatic EUS + MRI q1yr from age 50 (BRCA2 pancreatic 3-5x)"
        ),
        "targeted_rx": (
            "Olaparib (PARP inhibitor; HRD+ DLBCL maintenance); "
            "Cisplatin-based salvage (R-DHAP/R-ICE with cisplatin); "
            "Talazoparib (potent PARP trapper; BRCA2 DLBCL); "
            "Niraparib (PARP inhibitor; whole-blood BRCA2)"
        ),
    },
    {
        "gene": "CHEK2",
        "protein": (
            "CHEK2 -- 22q12.1 AD LOF -- 543aa -- "
            "CHK2-61kDa-ATM-Downstream-Checkpoint-"
            "1100delC-European-1in200-"
            "IVS2-Ashkenazi-NHL-DLBCL-2x-"
            "ATM-CHEK2-Additive-OMIM-604373"
        ),
        "locus": "22q12.1",
        "protein_size": (
            "543 aa / ~61 kDa / 22q12.1 CHEK2 DLBCL molecular context: "
            "STRUCTURE: "
            "  543 aa / ~61 kDa; serine/threonine kinase; "
            "  SQ/TQ cluster domain (SCD; aa 19-69): ATM phosphorylation sites; "
            "  Forkhead-associated domain (FHA; aa 112-175): pThr-binding; "
            "  Kinase domain (aa 220-486); "
            "  ATM phosphorylates CHK2-T68 → CHK2 activation → phosphorylates CDC25A/B/C + BRCA1 + p53; "
            "  CHEK2 LOF → impaired G1/S and G2/M checkpoint → accumulated DNA damage; "
            "CHEK2-NHL/DLBCL MECHANISM: "
            "  CHEK2 1100delC (c.1100delC): European founder; allele frequency 1 in 200 general; "
            "  NHL risk 2-fold; DLBCL specifically 2x; additive with ATM (dual heterozygosity → 8-10x); "
            "  IVS2+1G>A: Ashkenazi Jewish founder; similar risk; "
            "  CHEK2 I157T missense: Central European; modest risk 1.5x (weaker allele); "
            "  Concurrent breast cancer risk 25-37% lifetime (CHEK2 1100delC females); "
            "CLINICAL RULES CHEK2-DLBCL: "
            "  Standard R-CHOP — no specific chemotherapy modification required; "
            "  Screen concurrent breast cancer: annual MRI + mammography; "
            "  Test for ATM co-inheritance: CHEK2+ATM dual heterozygosity = 8-10x NHL risk; "
            "KEY VARIANTS: "
            "  c.1100delC (p.Thr367MetfsTer15): most common pathogenic; "
            "  c.444+1G>A (IVS2+1G>A): splice-site, Ashkenazi; "
            "  c.470T>C (p.Ile157Thr): Central European moderate-risk; "
        ),
        "syndrome": "CHEK2-Related-Cancer-Predisposition",
        "inheritance": "AD LOF heterozygous",
        "dlbcl_risk": "NHL/DLBCL 2x; ATM+CHEK2 dual heterozygosity 8-10x; concurrent breast cancer 25-37%",
        "pathognomonic": (
            "European ancestry + NHL + breast cancer + family history = CHEK2 1100delC testing; "
            "Dual heterozygosity ATM+CHEK2 = markedly elevated NHL risk (8-10x); "
            "CHEK2 is NOT monogenic cancer predisposition — it is a moderate-risk allele"
        ),
        "key_avoid": (
            "No specific chemotherapy avoidance in CHEK2 heterozygous (unlike ATM/TP53); "
            "CAUTION: radiation sensitivity mild (not absolute like ATM); "
            "AVOID missing concurrent breast cancer at DLBCL diagnosis"
        ),
        "key_rule": (
            "CHEK2+ATM dual heterozygosity = 8-10x NHL — panel-test for ATM if CHEK2 positive; "
            "Annual breast MRI mandatory females; "
            "Standard R-CHOP — no chemotherapy modification needed"
        ),
        "surveillance": (
            "Annual breast MRI + mammography females from age 30; "
            "Colonoscopy q5yr age 40+ (CHEK2 CRC 2-3x); "
            "Annual CBC + LDH; "
            "ATM co-test panel MANDATORY if CHEK2 positive (dual risk)"
        ),
        "targeted_rx": (
            "R-CHOP standard (no CHEK2-specific regimen modification); "
            "Clinical trials: WEE1 inhibitor (adavosertib) in CHEK2-null tumours; "
            "Breast cancer: olaparib for concurrent HER2-neg metastatic (CHEK2 emerging data); "
            "ATM-CHEK2 dual: ceralasertib trials"
        ),
    },
    {
        "gene": "PIK3CD",
        "protein": (
            "PIK3CD -- 1p36.22 AD GOF -- 1044aa -- "
            "PI3K-catalytic-delta-119kDa-"
            "APDS1-EBV-Driven-DLBCL-"
            "Leniolisib-FDA2023-"
            "PI3Kdelta-Inhibitor-Curative-OMIM-615513"
        ),
        "locus": "1p36.22",
        "protein_size": (
            "1044 aa / ~119 kDa / 1p36.22 PIK3CD DLBCL molecular context: "
            "STRUCTURE: "
            "  1044 aa / ~119 kDa; PI3K catalytic p110δ subunit; class IA PI3K; "
            "  N-terminal adapter-binding domain (ABD); RAS-binding domain (RBD); "
            "  C2 domain; helical domain; kinase domain (aa 726-1044); "
            "  Forms heterodimer with p85 regulatory subunit (PIK3R1/PIK3R2); "
            "  PIK3CD GOF → constitutive PI3K-AKT-mTOR → B-cell hyperactivation; "
            "APDS1/PIK3CD-DLBCL MECHANISM: "
            "  Gain-of-function mutations (E1021K most common; >50% APDS1): "
            "    constitutive PI3Kδ → pAKT elevated → mTOR → B-cell terminal differentiation arrested; "
            "  EBV-driven lymphoproliferation: PIK3CD-GOF → impaired EBV-specific CTL surveillance; "
            "  DLBCL: EBV-positive DLBCL 70% in APDS1; younger age onset (paediatric-adult); "
            "  CMV co-reactivation common (40%); lymphadenopathy + splenomegaly preceding DLBCL; "
            "  Leniolisib (Joenja®; PI3Kδ inhibitor; Pharming Group; FDA2023): approved APDS; "
            "    oral; 70 mg twice daily; reduces lymphadenopathy + restores lymphocyte subset; "
            "    first PI3Kδ-specific inhibitor approved; selective p110δ inhibition; "
            "CLINICAL RULES PIK3CD-DLBCL: "
            "  EBV PCR quantification MANDATORY in all PIK3CD carriers; "
            "  Leniolisib FDA2023 for APDS1: reduces EBV-driven lymphoproliferation pre-DLBCL; "
            "  Sirolimus (mTOR inhibitor) for lymphoproliferation bridge; "
            "  HSCT: RIC-HSCT curative for recurrent EBV-DLBCL in APDS1; "
            "KEY VARIANTS: "
            "  E1021K (p.Glu1021Lys): most common GOF, >50% APDS1; "
            "  E525K; C416R; N334K: alternative GOF hotspots; "
        ),
        "syndrome": "APDS1-Activated-PI3K-Delta-Syndrome-1",
        "inheritance": "AD GOF heterozygous",
        "dlbcl_risk": "EBV-driven DLBCL; EBV positivity 70%; lymphoproliferation precursor; younger onset",
        "pathognomonic": (
            "EBV-positive DLBCL + young age + recurrent respiratory infections + lymphadenopathy = "
            "PIK3CD testing MANDATORY; "
            "Flow cytometry: elevated transitional B-cells, reduced naïve B-cells, inverted CD4/CD8 ratio"
        ),
        "key_avoid": (
            "AVOID prolonged immunosuppression (corticosteroids alone) without leniolisib; "
            "AVOID rituximab monotherapy (depletes normal B-cells, increases infection risk); "
            "AVOID live vaccines"
        ),
        "key_rule": (
            "Leniolisib FDA2023 (PI3Kδ inhibitor 70mg BID) APPROVED APDS1; "
            "EBV PCR quantification MANDATORY — pre-emptive antivirals (valganciclovir); "
            "RIC-HSCT curative for recurrent EBV-DLBCL; sirolimus bridge"
        ),
        "surveillance": (
            "EBV + CMV PCR monthly (active surveillance); "
            "Annual CT chest/abdomen/pelvis; CBC + immunoglobulins q3M; "
            "IVIG replacement if hypogammaglobulinaemia; "
            "Spirometry annual (bronchiectasis risk)"
        ),
        "targeted_rx": (
            "Leniolisib (Joenja® 70 mg BID oral; FDA2023 APDS1; PI3Kδ inhibitor); "
            "Sirolimus (mTOR inhibitor; lymphoproliferation bridge); "
            "Valganciclovir (EBV/CMV pre-emptive antiviral); "
            "RIC-HSCT (fludarabine-cyclophosphamide; curative for recurrent EBV-DLBCL)"
        ),
    },
    {
        "gene": "KMT2D",
        "protein": (
            "KMT2D -- 12q13.12 AD LOF -- 5537aa -- "
            "MLL4-ALR-500kDa-H3K4-Methyltransferase-"
            "Kabuki1-B-cell-Lymphoma-Predisposition-"
            "EZH2-Inhibitor-Tazemetostat-Sensitive-"
            "COMPASS-Complex-OMIM-147920"
        ),
        "locus": "12q13.12",
        "protein_size": (
            "5537 aa / ~500 kDa / 12q13.12 KMT2D DLBCL molecular context: "
            "STRUCTURE: "
            "  5537 aa / ~500 kDa; histone H3 lysine 4 (H3K4) methyltransferase; "
            "  SET domain (aa 5432-5530): catalytic H3K4 mono/dimethylation; "
            "  PHD fingers (x6), FYRN/FYRC domains, AT-hook domains; "
            "  Component of COMPASS-like complex (ASC-2 complex); "
            "  KMT2D deposits H3K4me1 at active enhancers → gene activation; "
            "  KMT2D LOF → global loss of H3K4me1 at enhancers → tumour suppressor silencing; "
            "KMT2D-DLBCL MECHANISM: "
            "  Germline KMT2D LOF → Kabuki Syndrome 1 (intellectual disability, facial anomalies); "
            "  KMT2D somatic mutations in 60-80% of follicular lymphoma + 20-30% DLBCL; "
            "  Germline KMT2D carriers: B-cell lymphoma predisposition (lymphoproliferative in Kabuki); "
            "  KMT2D loss → reduced H3K4me1 → polycomb repressive complex 2 (PRC2/EZH2) gain; "
            "  EZH2 inhibitor tazemetostat (Tazverik®; FDA2020): synthetic lethality in KMT2D-null; "
            "  Lymphoproliferative disease in 5-10% of Kabuki patients; DLBCL reported; "
            "CLINICAL RULES KMT2D-DLBCL: "
            "  EZH2 inhibitor tazemetostat: approved for EZH2-mutant FL; active KMT2D-null DLBCL; "
            "  Epigenetic therapy: azacitidine + tazemetostat combination; "
            "  Kabuki clinical features: hypertelorism, arched eyebrows, brachydactyly, short stature; "
            "KEY VARIANTS: "
            "  Truncating (>70% of mutations): frameshift, nonsense, splice-site; "
            "  Kabuki-associated: highly heterogeneous; no common founder; "
        ),
        "syndrome": "Kabuki-Syndrome-1-KMT2D-Lymphoma-Predisposition",
        "inheritance": "AD LOF heterozygous",
        "dlbcl_risk": "B-cell lymphoma/DLBCL; lymphoproliferative 5-10% Kabuki; EZH2 inhibitor sensitive",
        "pathognomonic": (
            "Kabuki facial features (hypertelorism, arched eyebrows, brachydactyly, short stature) + "
            "B-cell lymphoma = KMT2D germline testing; "
            "KMT2D IHC absent/reduced in tumour + EZH2 gain = synthetic lethality target"
        ),
        "key_avoid": (
            "AVOID missing Kabuki features at lymphoma diagnosis (clinical dysmorphology screen); "
            "AVOID EZH2 inhibitor in EZH2 WT tumours without KMT2D confirmation; "
            "AVOID delay in tazemetostat in KMT2D-null/EZH2-overexpressed DLBCL"
        ),
        "key_rule": (
            "Tazemetostat (EZH2 inhibitor FDA2020) active in KMT2D-null/EZH2-overexpressed DLBCL; "
            "Screen for Kabuki features at lymphoma presentation; "
            "Epigenetic combination therapy (azacitidine + tazemetostat) emerging"
        ),
        "surveillance": (
            "Annual CBC + LDH; abdominal USS for lymphadenopathy; "
            "Intellectual disability management (Kabuki specific); "
            "Cardiology review (congenital heart disease Kabuki 30-50%); "
            "Ophthalmology annual (Kabuki ptosis)"
        ),
        "targeted_rx": (
            "Tazemetostat (Tazverik® EZH2 inhibitor FDA2020; 800 mg BID oral); "
            "Azacitidine (demethylating; epigenetic combo); "
            "EZH2 inhibitor + R-CHOP combination (investigational); "
            "Allo-SCT for recurrent KMT2D-DLBCL"
        ),
    },
    {
        "gene": "CARD11",
        "protein": (
            "CARD11 -- 7p22.2 AD LOF/GOF -- 1154aa -- "
            "CARD11-133kDa-CBM-Complex-"
            "BENTA-NF-kB-B-cell-Lymphoma-"
            "GOF-Venetoclax-Preferred-"
            "LOF-Atopic-MALT1-Inhibitor-JNJ-OMIM-616452"
        ),
        "locus": "7p22.2",
        "protein_size": (
            "1154 aa / ~133 kDa / 7p22.2 CARD11 DLBCL molecular context: "
            "STRUCTURE: "
            "  1154 aa / ~133 kDa; scaffolding protein; CBM complex (CARD11-BCL10-MALT1); "
            "  CARD domain (aa 1-100): dimerisation; "
            "  Coiled-coil domain; PDZ domain; SH3 domain; GUK-like domain; "
            "  CBM complex → IKK activation → IκBα phosphorylation → NF-κB nuclear translocation; "
            "  CARD11 GOF: somatic in ABC-DLBCL (20-25%); germline GOF = BENTA disease; "
            "BENTA/CARD11-DLBCL MECHANISM: "
            "  GOF (BENTA = B-cell Expansion with NF-κB and T-cell Anergy): "
            "    constitutive NF-κB → massive B-cell polyclonal expansion → lymphoma; "
            "    BENTA features: atopic dermatitis, food allergy, eosinophilia, splenomegaly; "
            "    B-cell lymphoma (DLBCL, MALT-type) in 10-20% BENTA patients; "
            "    Venetoclax PREFERRED (NF-κB upregulates BCL-2 → venetoclax sensitivity); "
            "    MALT1 inhibitor JNJ-67856633: blocks CARD11-BCL10-MALT1 → NF-κB suppression; "
            "  LOF (CARD11 haploinsufficiency): "
            "    impaired BCR signalling → combined immunodeficiency + lymphoma predisposition; "
            "    Primary immunodeficiency: recurrent respiratory infections; "
            "    B-cell lymphoma via impaired tumour immunosurveillance; "
            "CLINICAL RULES CARD11-DLBCL: "
            "  GOF-BENTA: venetoclax + R-CHOP (BCL-2 elevated); MALT1 inhibitor JNJ clinical trial; "
            "  LOF: IVIG replacement + rituximab-sparing (immunodeficiency); "
            "  Ibrutinib: LESS effective in GOF-CARD11 (downstream NF-κB bypass); "
            "KEY VARIANTS: "
            "  GOF: L225LI, G116E, R179Q (BENTA dominant); "
            "  LOF: frameshift/nonsense/splice-site (combined immunodeficiency); "
        ),
        "syndrome": "BENTA-Disease-CARD11-GOF-NF-kB-Lymphoma",
        "inheritance": "AD LOF/GOF heterozygous",
        "dlbcl_risk": "B-cell lymphoma/DLBCL 10-20% BENTA (GOF); immunodeficiency lymphoma (LOF); venetoclax sensitive GOF",
        "pathognomonic": (
            "Atopic dermatitis + food allergy + eosinophilia + splenomegaly + DLBCL = BENTA/CARD11 GOF; "
            "NF-κB pathway IHC (p65 nuclear) + BCL-2 high in tumour = venetoclax target; "
            "Ibrutinib resistance: test CARD11 GOF before prescribing BTKi in DLBCL"
        ),
        "key_avoid": (
            "AVOID ibrutinib in CARD11 GOF-DLBCL (downstream NF-κB bypass reduces efficacy); "
            "AVOID high-dose steroids alone in BENTA (immunosuppression worsens B-cell expansion); "
            "AVOID rituximab monotherapy in CARD11 LOF (worsens immunodeficiency)"
        ),
        "key_rule": (
            "Venetoclax PREFERRED CARD11 GOF-DLBCL (BCL-2 upregulated by NF-κB); "
            "MALT1 inhibitor JNJ-67856633 for GOF (CARD11-BCL10-MALT1 blockade); "
            "Ibrutinib LESS effective GOF — confirm variant before BTKi"
        ),
        "surveillance": (
            "Annual CBC + immunoglobulins + LDH; abdominal USS for splenomegaly; "
            "Skin assessment (atopic BENTA); allergy/immunology co-management; "
            "IVIG if hypogammaglobulinaemia (LOF); EBV PCR q6M"
        ),
        "targeted_rx": (
            "Venetoclax + R-CHOP (GOF-BENTA; BCL-2 elevated; preferred); "
            "MALT1 inhibitor JNJ-67856633 (investigational GOF-CARD11); "
            "Sirolimus (mTOR bridge in BENTA lymphoproliferation); "
            "IVIG (LOF immunodeficiency replacement)"
        ),
    },
    {
        "gene": "TNFRSF13B",
        "protein": (
            "TNFRSF13B -- 17p11.2 AD -- 293aa -- "
            "TACI-31kDa-TNF-Receptor-Superfamily-"
            "CVID-2-NHL-DLBCL-10-12x-"
            "C104R-A181E-Founders-"
            "IVIG-Rituximab-Lenalidomide-OMIM-607892"
        ),
        "locus": "17p11.2",
        "protein_size": (
            "293 aa / ~31 kDa / 17p11.2 TNFRSF13B DLBCL molecular context: "
            "STRUCTURE: "
            "  293 aa / ~31 kDa; TACI (transmembrane activator and CAML interactor); "
            "  TNF receptor superfamily member 13B; "
            "  Cysteine-rich domains (CRD1 aa 1-38, CRD2 aa 40-71); "
            "  Transmembrane domain; CAML-binding domain; TRAF2/5/6-binding domain; "
            "  TACI ligands: APRIL (TNFSF13) and BAFF (TNFSF13B); "
            "  TACI signalling → NF-κB → IgA/IgG class switching; "
            "  TACI LOF/dominant-negative → impaired CSR → CVID; "
            "TNFRSF13B-CVID/DLBCL MECHANISM: "
            "  Heterozygous TNFRSF13B mutations → CVID-2 (common variable immunodeficiency); "
            "  CVID → NHL/DLBCL 10-12x lifetime risk (highest NHL risk among PID); "
            "  Mechanisms: impaired immune surveillance + chronic antigen stimulation + EBV; "
            "  Granulomatous disease (30%) + splenomegaly (40%) in CVID-2; "
            "  C104R (aa 104): European founder (allele frequency ~1 in 500); "
            "  A181E (aa 181): common pathogenic variant; "
            "  Both C104R and A181E impair TACI-APRIL/BAFF binding → reduced B-cell signalling; "
            "CLINICAL RULES TNFRSF13B-DLBCL: "
            "  IVIG replacement MANDATORY in CVID-2 (maintains serum IgG ≥7 g/L target); "
            "  Rituximab + lenalidomide + R-CHOP for CVID-DLBCL; "
            "  Reduce rituximab dose/duration (hypogammaglobulinaemia worsened); "
            "  Belimumab (anti-BAFF) may paradoxically worsen CVID (BAFF is survival signal); "
            "KEY VARIANTS: "
            "  C104R (c.310T>C): most common, European; haploinsufficiency; "
            "  A181E (c.542C>A): dominant negative; impairs TACI-TRAF interaction; "
            "  P251L; S144X; various rare truncating alleles; "
        ),
        "syndrome": "CVID-2-TNFRSF13B-NHL-DLBCL-Predisposition",
        "inheritance": "AD heterozygous (haploinsufficiency/dominant-negative)",
        "dlbcl_risk": "NHL/DLBCL 10-12x lifetime (HIGHEST among immune-dysregulation NHL predispositions); CVID context",
        "pathognomonic": (
            "CVID (serum IgG <7 g/L + poor vaccine response + absent isohaemagglutinins) + DLBCL = "
            "TNFRSF13B testing; "
            "Granulomatous disease 30% (non-caseating granulomas BM/liver/lung) in CVID-2; "
            "C104R allele frequency 1 in 500 Europeans — penetrance variable"
        ),
        "key_avoid": (
            "AVOID reducing/stopping IVIG replacement during DLBCL treatment; "
            "AVOID belimumab (anti-BAFF) — BAFF is survival signal for residual B-cells in CVID; "
            "AVOID full-dose rituximab without IVIG cover (worsens hypogammaglobulinaemia)"
        ),
        "key_rule": (
            "IVIG MANDATORY (target IgG ≥7 g/L) throughout DLBCL treatment in CVID-2; "
            "NHL/DLBCL 10-12x HIGHEST immune-dysregulation risk; "
            "Rituximab + lenalidomide + R-CHOP preferred; monitor IgG monthly during treatment"
        ),
        "surveillance": (
            "IVIG q3-4wk (maintain IgG ≥7 g/L); annual CT chest (bronchiectasis, lymphoma); "
            "Annual CBC + LDH; IgG/A/M q3M; "
            "Annual abdominal USS (splenomegaly, lymphadenopathy); "
            "Spirometry annual (bronchiectasis CVID 30%)"
        ),
        "targeted_rx": (
            "R-CHOP + lenalidomide (rituximab-lenalidomide-CHOP; DLBCL CVID); "
            "IVIG replacement q3-4wk (400-600 mg/kg; target IgG ≥7 g/L); "
            "Abatacept (CTLA4-Ig; granulomatous CVID; empirical); "
            "RIC-HSCT (for recurrent CVID-DLBCL; restores immune function)"
        ),
    },
]


# ---------------------------------------------------------------------------
# Synthetic patient cohort
# ---------------------------------------------------------------------------

def _make_patient(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)

    # Gene-specific DLBCL clinical parameters
    dlbcl_params = {
        "TP53":       dict(age_mu=38, age_sd=14,  male_prob=0.50, ebv_pos_prob=0.12,
                           hrd_pos_prob=0.10,  complex_karyotype_prob=0.60,
                           concurrent_breast_prob=0.15, ivig_required_prob=0.02,
                           cr_rate=0.68, five_yr_os=0.55),
        "ATM":        dict(age_mu=58, age_sd=12,  male_prob=0.50, ebv_pos_prob=0.10,
                           hrd_pos_prob=0.20,  complex_karyotype_prob=0.35,
                           concurrent_breast_prob=0.18, ivig_required_prob=0.02,
                           cr_rate=0.72, five_yr_os=0.62),
        "BRCA2":      dict(age_mu=58, age_sd=12,  male_prob=0.40, ebv_pos_prob=0.08,
                           hrd_pos_prob=0.65,  complex_karyotype_prob=0.22,
                           concurrent_breast_prob=0.20, ivig_required_prob=0.02,
                           cr_rate=0.74, five_yr_os=0.64),
        "CHEK2":      dict(age_mu=62, age_sd=10,  male_prob=0.45, ebv_pos_prob=0.08,
                           hrd_pos_prob=0.15,  complex_karyotype_prob=0.18,
                           concurrent_breast_prob=0.40, ivig_required_prob=0.02,
                           cr_rate=0.75, five_yr_os=0.65),
        "PIK3CD":     dict(age_mu=25, age_sd=12,  male_prob=0.52, ebv_pos_prob=0.72,
                           hrd_pos_prob=0.05,  complex_karyotype_prob=0.12,
                           concurrent_breast_prob=0.02, ivig_required_prob=0.45,
                           cr_rate=0.65, five_yr_os=0.58),
        "KMT2D":      dict(age_mu=38, age_sd=14,  male_prob=0.50, ebv_pos_prob=0.15,
                           hrd_pos_prob=0.05,  complex_karyotype_prob=0.15,
                           concurrent_breast_prob=0.08, ivig_required_prob=0.08,
                           cr_rate=0.70, five_yr_os=0.60),
        "CARD11":     dict(age_mu=22, age_sd=10,  male_prob=0.50, ebv_pos_prob=0.35,
                           hrd_pos_prob=0.04,  complex_karyotype_prob=0.12,
                           concurrent_breast_prob=0.04, ivig_required_prob=0.30,
                           cr_rate=0.72, five_yr_os=0.62),
        "TNFRSF13B":  dict(age_mu=48, age_sd=14,  male_prob=0.45, ebv_pos_prob=0.30,
                           hrd_pos_prob=0.05,  complex_karyotype_prob=0.14,
                           concurrent_breast_prob=0.06, ivig_required_prob=0.85,
                           cr_rate=0.68, five_yr_os=0.56),
    }
    p = dlbcl_params[gene]

    # Age at DLBCL diagnosis
    age_raw = rng.gauss(p["age_mu"], p["age_sd"])
    if gene in ("PIK3CD", "CARD11"):
        age = max(8, min(55, round(age_raw, 1)))
    elif gene == "KMT2D":
        age = max(15, min(65, round(age_raw, 1)))
    elif gene == "TP53":
        age = max(18, min(70, round(age_raw, 1)))
    else:
        age = max(25, min(80, round(age_raw, 1)))

    is_male              = rng.random() < p["male_prob"]
    ebv_positive         = rng.random() < p["ebv_pos_prob"]
    hrd_positive         = rng.random() < p["hrd_pos_prob"]
    complex_karyotype    = rng.random() < p["complex_karyotype_prob"]
    concurrent_breast    = (not is_male) and rng.random() < p["concurrent_breast_prob"]
    ivig_required        = rng.random() < p["ivig_required_prob"]

    # IPI score (International Prognostic Index)
    ipi_raw = rng.gauss(2.8 if complex_karyotype else 2.2, 1.2)
    ipi = min(5, max(0, round(ipi_raw)))

    # DLBCL subtype (ABC vs GCB)
    if gene in ("CARD11", "TNFRSF13B"):
        abc_prob = 0.68
    elif gene == "TP53":
        abc_prob = 0.42
    else:
        abc_prob = 0.45
    dlbcl_subtype = "ABC" if rng.random() < abc_prob else "GCB"

    # Ann Arbor stage
    stage_probs = [0.08, 0.18, 0.30, 0.44]
    stage_roll = rng.random()
    stage = 1
    cumulative = 0.0
    for s, sp in enumerate(stage_probs, 1):
        cumulative += sp
        if stage_roll < cumulative:
            stage = s
            break

    # Extranodal involvement
    extranodal = rng.random() < (0.55 if stage >= 3 else 0.25)

    # CNS involvement
    cns_involved = rng.random() < (0.15 if gene == "TP53" else 0.08)

    # LDH
    ldh_ratio = max(0.8, rng.gauss(2.1 if stage >= 3 else 1.4, 0.8))
    ldh_ratio = round(ldh_ratio, 1)

    # Response to first-line R-CHOP
    response_prob = p["cr_rate"] - (0.05 if complex_karyotype else 0) + (0.05 if ipi <= 2 else -0.05)
    response_prob = max(0.40, min(0.90, response_prob))
    response = rng.choices(
        ["CR", "PR", "SD", "PD"],
        weights=[response_prob * 100, 15, 8, (1 - response_prob) * 100 - 23],
        k=1
    )[0]
    if response == "CR":
        response = "CR"
    elif response == "PR":
        response = "PR"

    # 5-yr OS
    os_base = p["five_yr_os"]
    if complex_karyotype:
        os_base -= 0.08
    if ipi >= 4:
        os_base -= 0.10
    if cns_involved:
        os_base -= 0.12
    os_base = max(0.20, min(0.90, os_base))

    five_yr_os_achieved = rng.random() < os_base
    outcome = "Alive in remission" if (five_yr_os_achieved and response == "CR") else \
              "Alive with disease" if (five_yr_os_achieved and response != "CR") else \
              "Deceased" if not five_yr_os_achieved else "Alive in remission"

    return {
        "gene":                gene,
        "seed":                seed,
        "sex":                 "M" if is_male else "F",
        "age_at_diagnosis":    age,
        "dlbcl_subtype":       dlbcl_subtype,
        "stage":               stage,
        "ipi":                 ipi,
        "ebv_positive":        ebv_positive,
        "hrd_positive":        hrd_positive,
        "complex_karyotype":   complex_karyotype,
        "extranodal":          extranodal,
        "cns_involved":        cns_involved,
        "ldh_ratio":           ldh_ratio,
        "concurrent_breast":   concurrent_breast,
        "ivig_required":       ivig_required,
        "response":            response,
        "five_yr_os":          five_yr_os_achieved,
        "outcome":             outcome,
        "locus":               g["locus"],
        "inheritance":         g["inheritance"],
        "key_rule":            g["key_rule"],
    }


def generate_overview() -> dict:
    patients = []
    for i, gd in enumerate(ATLAS_GENES):
        for j in range(40):
            patients.append(_make_patient(gd["gene"], SEED_BASE + i + j * 8))

    gene_counts = {}
    for p in patients:
        gene_counts.setdefault(p["gene"], 0)
        gene_counts[p["gene"]] += 1

    total         = len(patients)
    cr_n          = sum(1 for p in patients if p["response"] == "CR")
    ebv_pos       = sum(1 for p in patients if p["ebv_positive"])
    hrd_pos       = sum(1 for p in patients if p["hrd_positive"])
    complex_k     = sum(1 for p in patients if p["complex_karyotype"])
    extranodal    = sum(1 for p in patients if p["extranodal"])
    cns_n         = sum(1 for p in patients if p["cns_involved"])
    ivig_req      = sum(1 for p in patients if p["ivig_required"])
    concurrent_b  = sum(1 for p in patients if p["concurrent_breast"])
    five_yr_alive = sum(1 for p in patients if p["five_yr_os"])
    males         = sum(1 for p in patients if p["sex"] == "M")
    abc_subtype   = sum(1 for p in patients if p["dlbcl_subtype"] == "ABC")

    avg_age    = round(sum(p["age_at_diagnosis"] for p in patients) / total, 1)
    avg_ipi    = round(sum(p["ipi"] for p in patients) / total, 1)
    avg_ldh    = round(sum(p["ldh_ratio"] for p in patients) / total, 1)

    gene_ages = {}
    for gd in ATLAS_GENES:
        pts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_ages[gd["gene"]] = round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1)

    gene_summary = []
    for gd in ATLAS_GENES:
        pts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_summary.append({
            "gene":           gd["gene"],
            "syndrome":       gd["syndrome"],
            "locus":          gd["locus"],
            "inheritance":    gd["inheritance"],
            "n":              len(pts),
            "mean_age_dx":    gene_ages[gd["gene"]],
            "dlbcl_risk":     gd["dlbcl_risk"],
            "pathognomonic":  gd["pathognomonic"],
            "key_rule":       gd["key_rule"],
            "key_avoid":      gd["key_avoid"],
            "targeted_rx":    gd["targeted_rx"],
        })

    return {
        "atlas":             "Hereditary-DLBCL-Predisposition-Atlas",
        "seed_range":        f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":    total,
        "total_genes":       8,
        "gene_counts":       gene_counts,
        "cr_n":              cr_n,
        "ebv_positive":      ebv_pos,
        "hrd_positive":      hrd_pos,
        "complex_karyotype": complex_k,
        "extranodal":        extranodal,
        "cns_involved":      cns_n,
        "ivig_required":     ivig_req,
        "concurrent_breast": concurrent_b,
        "five_yr_alive":     five_yr_alive,
        "males":             males,
        "abc_subtype":       abc_subtype,
        "avg_age_dx":        avg_age,
        "avg_ipi":           avg_ipi,
        "avg_ldh_ratio":     avg_ldh,
        "gene_summary":      gene_summary,
        "clinical_pearl": (
            "DLBCL-GERMLINE: germline predisposition alters BOTH risk stratification AND treatment choice; "
            "TP53-AVOID-RADIATION-ABSOLUTELY (LFS); ATM-RADIOSENSITIVITY-ABSOLUTE (no RT, no TBI); "
            "BRCA2-HRD-cisplatin-preferred; PIK3CD-Leniolisib-FDA2023-EBV-driven; "
            "CARD11-GOF-Venetoclax-preferred (ibrutinib less effective); "
            "TNFRSF13B-IVIG-MANDATORY-throughout-treatment (highest NHL risk PID 10-12x)"
        ),
    }


def generate_breakdown() -> dict:
    all_patients = []
    for i, gd in enumerate(ATLAS_GENES):
        for j in range(40):
            all_patients.append(_make_patient(gd["gene"], SEED_BASE + i + j * 8))

    gene_data = {}
    for gd in ATLAS_GENES:
        pts = [p for p in all_patients if p["gene"] == gd["gene"]]

        cr_n         = sum(1 for p in pts if p["response"] == "CR")
        ebv_n        = sum(1 for p in pts if p["ebv_positive"])
        hrd_n        = sum(1 for p in pts if p["hrd_positive"])
        complex_n    = sum(1 for p in pts if p["complex_karyotype"])
        abc_n        = sum(1 for p in pts if p["dlbcl_subtype"] == "ABC")
        cns_n        = sum(1 for p in pts if p["cns_involved"])
        ivig_n       = sum(1 for p in pts if p["ivig_required"])
        alive5_n     = sum(1 for p in pts if p["five_yr_os"])
        concurrent_b = sum(1 for p in pts if p["concurrent_breast"])
        stage34_n    = sum(1 for p in pts if p["stage"] >= 3)

        avg_age_dx   = round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1)
        avg_ipi      = round(sum(p["ipi"] for p in pts) / len(pts), 1)
        avg_ldh      = round(sum(p["ldh_ratio"] for p in pts) / len(pts), 1)

        patients_list = [{
            "seed":              p["seed"],
            "sex":               p["sex"],
            "age_at_diagnosis":  p["age_at_diagnosis"],
            "dlbcl_subtype":     p["dlbcl_subtype"],
            "stage":             p["stage"],
            "ipi":               p["ipi"],
            "ebv_positive":      p["ebv_positive"],
            "hrd_positive":      p["hrd_positive"],
            "complex_karyotype": p["complex_karyotype"],
            "cns_involved":      p["cns_involved"],
            "ldh_ratio":         p["ldh_ratio"],
            "ivig_required":     p["ivig_required"],
            "response":          p["response"],
            "outcome":           p["outcome"],
        } for p in pts]

        gene_data[gd["gene"]] = {
            "gene":             gd["gene"],
            "syndrome":         gd["syndrome"],
            "locus":            gd["locus"],
            "inheritance":      gd["inheritance"],
            "n":                len(pts),
            "cr_n":             cr_n,
            "ebv_positive_n":   ebv_n,
            "hrd_positive_n":   hrd_n,
            "complex_karyotype_n": complex_n,
            "abc_subtype_n":    abc_n,
            "cns_involved_n":   cns_n,
            "ivig_required_n":  ivig_n,
            "five_yr_alive_n":  alive5_n,
            "concurrent_breast_n": concurrent_b,
            "stage_3_4_n":      stage34_n,
            "avg_age_dx":       avg_age_dx,
            "avg_ipi":          avg_ipi,
            "avg_ldh_ratio":    avg_ldh,
            "dlbcl_risk":       gd["dlbcl_risk"],
            "pathognomonic":    gd["pathognomonic"],
            "key_rule":         gd["key_rule"],
            "key_avoid":        gd["key_avoid"],
            "targeted_rx":      gd["targeted_rx"],
            "patients":         patients_list[:20],
        }

    return {
        "atlas":      "Hereditary-DLBCL-Predisposition-Atlas",
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
            "dlbcl_risk":    gd["dlbcl_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_avoid":     gd["key_avoid"],
            "key_rule":      gd["key_rule"],
            "surveillance":  gd["surveillance"],
            "targeted_rx":   gd["targeted_rx"],
        }

    key_clinical_distinctions = [
        "TP53-DLBCL AVOID RADIATION ABSOLUTELY: Li-Fraumeni germline → even low-dose RT triggers second malignancies; "
        "use MRI-based surveillance (Toronto WB-MRI protocol); APR-246/eprenetapopt reactivates mut-p53; "
        "venetoclax + R-ICE preferred salvage; allo-SCT CR1/CR2 consolidation preferred over RT",
        "ATM-DLBCL RADIOSENSITIVITY ABSOLUTE — no RT, no TBI: del11q somatic in 40% ATM-DLBCL (biallelic tumour loss); "
        "ceralasertib (ATRi AZD6738) synthetic lethality in ATM-null; use RIC-HSCT (fludarabine-cyclophosphamide) "
        "not TBI-based; PARP inhibitors active in ATM-null DLBCL; screen female carriers for breast/ovarian cancer",
        "BRCA2-DLBCL HRD positive 60%: cisplatin-based salvage preferred (R-DHAP with cisplatin > oxaliplatin); "
        "olaparib maintenance post-CR for HRD+ DLBCL emerging; SBS3+ID6 signatures confirm HRD; "
        "concurrent breast/ovarian cancer 15-20% of BRCA2-DLBCL — screen at diagnosis; FA-D1 biallelic = AVOID alkylating + RT absolutely",
        "CHEK2+ATM dual heterozygosity 8-10x NHL risk: panel-test for ATM when CHEK2 positive; "
        "standard R-CHOP (no specific chemotherapy modification for CHEK2 alone); "
        "annual breast MRI mandatory females 30+; CHEK2 I157T is moderate-risk allele only (1.5x) — not equivalent to 1100delC",
        "PIK3CD-APDS1 Leniolisib FDA2023 APPROVED: first PI3Kδ-selective inhibitor; oral 70 mg BID; "
        "reduces EBV-driven lymphoproliferation pre-DLBCL; EBV PCR monitoring mandatory; "
        "sirolimus bridge; RIC-HSCT curative for recurrent EBV-DLBCL; AVOID rituximab monotherapy (worsens immunodeficiency)",
        "KMT2D-Kabuki DLBCL — tazemetostat EZH2 inhibitor FDA2020 active: KMT2D LOF → H3K4me1 loss → EZH2 overactivation; "
        "tazemetostat 800 mg BID for EZH2-mutant FL and KMT2D-null DLBCL; screen for Kabuki features at lymphoma presentation; "
        "azacitidine + tazemetostat epigenetic combination; KMT2D somatic in 20-30% all DLBCL (not just germline)",
        "CARD11 GOF vs LOF distinction CRITICAL for treatment choice: GOF (BENTA) → constitutive NF-κB → BCL-2 elevated → "
        "venetoclax preferred + MALT1 inhibitor JNJ-67856633; ibrutinib LESS effective GOF (downstream NF-κB bypass); "
        "LOF → immunodeficiency → IVIG replacement + rituximab-sparing regimens; confirm variant type before BTKi prescription",
        "TNFRSF13B-CVID-2 IVIG MANDATORY throughout DLBCL treatment: CVID → NHL/DLBCL 10-12x = HIGHEST among PIDs; "
        "maintain serum IgG ≥7 g/L during R-CHOP; C104R (1 in 500 Europeans) + A181E most common founders; "
        "rituximab + lenalidomide + R-CHOP preferred; AVOID belimumab (BAFF is B-cell survival signal in CVID); "
        "granulomatous disease 30% (non-caseating: BM/liver/lung) — diagnose before lymphoma attribution",
    ]

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"DLBCL RISK: {gd['dlbcl_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-DLBCL-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": key_clinical_distinctions,
    }
