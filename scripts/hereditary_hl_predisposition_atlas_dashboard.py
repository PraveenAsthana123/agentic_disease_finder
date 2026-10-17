#!/usr/bin/env python3
"""Hereditary-HL-Predisposition-Atlas -- Complete 8-Gene Reference
Classical Hodgkin Lymphoma (cHL) and Nodular Lymphocyte-Predominant HL (NLPHL).

ATM      (ATM kinase; 3056aa; 11q22.3; AD LOF;
          Ataxia-Telangiectasia (OMIM 208900);
          HL 3-4x monoallelic; biallelic 40-60x highest;
          RADIOSENSITIVITY ABSOLUTE — avoid RT even het; ceralasertib ATRi;
          seed SEED_BASE+0).
NBN      (NBS1/Nibrin; 754aa; 8q21.3; AD-AR LOF;
          Nijmegen Breakage Syndrome (OMIM 251260);
          HL/NHL 60x HIGHEST biallelic; 657del5 Slavic 1in150;
          RADIOSENSITIVITY ABSOLUTE biallelic; cranial RT absolutely CI biallelic;
          seed SEED_BASE+1).
BRCA2    (BRCA2; 3418aa; 13q12.3; AD LOF;
          HBOC (OMIM 612555);
          HL 2-3x; NHL 2-3x; HRD 60%; cisplatin preferred; olaparib PARP;
          FA-D1 biallelic — AVOID alkylating ABSOLUTELY;
          seed SEED_BASE+2).
TP53     (p53; 393aa; 17p13.1; AD LOF;
          Li-Fraumeni Syndrome (OMIM 151623);
          HL 3-5x; transformation HL→DLBCL 15-20%; AVOID RADIATION ABSOLUTELY;
          venetoclax preferred; WB-MRI Toronto annually; seed SEED_BASE+3).
CHEK2    (CHK2; 543aa; 22q12.1; AD LOF;
          1100delC European 1in200; IVS2+1G>A Ashkenazi;
          HL/NHL 2-3x; NOT HRD — NO PARPi; ATM+CHEK2 additive 8-10x;
          seed SEED_BASE+4).
BLM      (BLM helicase; 1417aa; 15q26.1; AR LOF;
          Bloom Syndrome (OMIM 210900);
          HL/NHL 150-300x biallelic (Bloom); monoallelic carrier 2-3x;
          2281del6ins7 Ashkenazi founder 1in107; SCE elevated PATHOGNOMONIC;
          AVOID alkylating/radiation biallelic; seed SEED_BASE+5).
KLHDC8B  (KLHDC8B; 560aa; 3q22.3; AD LOF;
          Nodular Lymphocyte-Predominant HL (NLPHL) specific;
          NLPHL young onset 20-40yr; EBV-negative NLPHL;
          anti-CD20 rituximab preferred; R-CHOP or R-CHOP-like;
          seed SEED_BASE+6).
RB1      (Rb1; 928aa; 13q14.2; AD LOF;
          Hereditary Retinoblastoma (OMIM 180200);
          secondary HL after Rb RT 5-10x; del13q14 somatic;
          CDK4/6i INACTIVE — pRb pathway bypassed; avoid RT in Rb survivors;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3614-3621)
"""
import random

SEED_BASE = 3614

ATLAS_GENES = [
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD LOF -- 3056aa -- "
            "ATM-350kDa-PIK3-Like-Kinase-"
            "DDR-Sensor-MRN-Complex-H2AX-BRCA1-CHK2-p53-"
            "HL-3-4x-Monoallelic-40-60x-Biallelic-RADIOSENSITIVITY-ABSOLUTE-"
            "A-T-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / ~350 kDa / 11q22.3 ATM HL molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; PIKK-family Ser/Thr kinase; activated by DSBs via MRN complex; "
            "  HEAT repeats (aa 1-3000): scaffolding; FAT domain; kinase domain; FATC domain; "
            "  Key substrates: H2AX (γH2AX foci), p53, BRCA1, CHK2, 53BP1, SMC1; "
            "  Biallelic loss (A-T): combined immunodeficiency + cerebellar ataxia + telangiectasia; "
            "  Monoallelic (heterozygous carriers): HL 3-4x; NHL 4-5x; breast 2x; "
            "HL MECHANISM: "
            "  Defective G2/M checkpoint → RS-cell precursors survive; "
            "  del11q22.3 somatic in 15-25% cHL tumours; "
            "  Monoallelic ATM: normal radiation response REDUCED (haploinsufficiency); "
            "  Biallelic ATM (A-T): HL 40-60x HIGHEST risk — first lymphoma often HL; "
            "CLINICAL RULES FOR ATM-HL: "
            "  RADIOSENSITIVITY ABSOLUTE — even heterozygous carriers: reduce RT dose 20-30% "
            "    OR substitute chemotherapy-only (ABVD/BV-AVD without RT); "
            "  Biallelic A-T: RT ABSOLUTELY CONTRAINDICATED; HSCT: no TBI; "
            "  Ceralasertib (AZD6738, ATRi): synthetic lethality with ATM LOF; "
            "  IgA deficiency PATHOGNOMONIC of A-T biallelic; "
            "  PET-adapted therapy: use PET after 2 cycles to guide consolidation RT decision; "
            "  Cascade ATM testing: first-degree family members annually CBC + IgA/IgG; "
            "KEY VARIANTS: "
            "  Monoallelic: c.7271T>G (p.Val2424Gly), c.1009C>T (p.Arg337Cys), del11q22 large; "
            "  Biallelic: truncating + truncating (A-T full phenotype); "
            "  c.5557G>A (p.Asp1853Asn) missense: reduced kinase — moderate risk; "
            "SURVEILLANCE: "
            "  Annual lymph node exam + CBC; PET/CT only if clinical suspicion; "
            "  Heterozygous: routine follow-up; no prophylactic HSCT; "
            "  Biallelic: neurological surveillance; IgA level; chest CT for pulmonary atelectasis; "
            "TARGETED Rx: "
            "  Ceralasertib (ATRi, AstraZeneca): phase II HL relapsed; "
            "  BV-AVD (brentuximab vedotin + AVD): preferred over ABVD+RT for monoallelic ATM; "
            "  Pembrolizumab (PD-1): salvage for relapsed/refractory cHL; "
        ),
        "syndrome": "Ataxia-Telangiectasia (A-T) / ATM-HL",
        "inheritance": "AD LOF heterozygous (monoallelic); AR biallelic = A-T",
        "hl_risk": "Monoallelic: HL 3-4x; Biallelic A-T: HL 40-60x HIGHEST; del11q somatic 15-25%",
        "pathognomonic": "IgA deficiency PATHOGNOMONIC A-T biallelic; cerebellar ataxia + telangiectasia biallelic",
        "key_rule": "RADIOSENSITIVITY ABSOLUTE — reduce RT 20-30% monoallelic; RT ABSOLUTELY CI biallelic; BV-AVD preferred",
        "key_avoid": "RT biallelic ABSOLUTELY CI; TBI in HSCT CI; even monoallelic: dose-reduce RT or omit",
        "surveillance": "Annual CBC + IgA (biallelic); lymph node exam; cascade family ATM testing",
        "targeted_rx": "Ceralasertib ATRi (synthetic lethality ATM LOF); BV-AVD over ABVD+RT; pembrolizumab PD-1 salvage",
    },
    {
        "gene": "NBN",
        "protein": (
            "NBN -- 8q21.3 AD-AR LOF -- 754aa -- "
            "NBS1-95kDa-MRN-Complex-Nibrin-"
            "BRCT-FHA-ATM-Recruitment-"
            "NHL-60x-HIGHEST-657del5-Slavic-1in150-RADIOSENSITIVITY-ABSOLUTE-"
            "NBS-OMIM-251260"
        ),
        "locus": "8q21.3",
        "protein_size": (
            "754 aa / ~95 kDa / 8q21.3 NBN NBS1 HL molecular context: "
            "STRUCTURE: "
            "  754 aa / ~95 kDa; MRN complex component (MRE11-RAD50-NBS1); "
            "  FHA domain (aa 24-108): phospho-Thr recognition (MDC1 docking); "
            "  BRCT1 (aa 111-196) + BRCT2 (aa 221-320): tandem BRCT pair; γH2AX binding; "
            "  ATM-binding domain (aa 734-754): C-terminal; recruits and activates ATM at DSBs; "
            "  NLS (nuclear localisation signals); interacts with PCNA; "
            "  657del5 (c.657_661del5): Slavic founder; 1:150 Ashkenazi/Slavic; truncation p.Lys219Asnfs; "
            "NBS BIALLELIC PHENOTYPE: "
            "  Microcephaly PATHOGNOMONIC (100%); bird-like face; combined ID (T + B cell); "
            "  NHL/HL 60x HIGHEST risk; lymphoma median onset 10 years; "
            "  Radiosensitivity ABSOLUTE (biallelic): cranial RT absolutely CI — fatal toxicity; "
            "MONOALLELIC (CARRIER): "
            "  657del5 heterozygous: HL/NHL 2-3x general; no immunodeficiency; "
            "  Annual surveillance recommended in NBS families; "
            "CLINICAL RULES FOR NBN-HL: "
            "  Biallelic NBS: RT ABSOLUTELY CI (cranial and mediastinal); HSCT no TBI; "
            "  Biallelic: reduce anthracycline cumulative dose (cardiotoxicity combined ID); "
            "  Biallelic: use non-alkylating HL protocols adapted from paediatric immunodeficiency HL; "
            "  Monoallelic 657del5: standard HL therapy; reduce RT to lowest effective dose; "
            "  Cascade 657del5 testing mandatory in Slavic/Ashkenazi families with HL; "
            "KEY VARIANTS: "
            "  657del5 (c.657_661delACAAA): Slavic founder 90% NBS; Czech/Polish/Ashkenazi; "
            "  c.835G>T (p.Glu279Ter): Czech; c.1089C>A (p.Cys363Ter): rare; "
            "SURVEILLANCE: "
            "  Biallelic: MRI brain q2yr (medulloblastoma risk); PET/CT q2yr lymphoma screening; "
            "  CBC quarterly biallelic; immunoglobulin levels (IVIG if low); "
            "TARGETED Rx: "
            "  Biallelic NBS-HL: gemcitabine-based salvage (NO alkylating); dose-modified OEPA/COPP; "
            "  HSCT curative for NBS-immunodeficiency — use reduced-intensity conditioning (RIC) NO TBI; "
            "  Monoallelic: standard BV-AVD; "
        ),
        "syndrome": "Nijmegen Breakage Syndrome (NBS) / NBN-HL",
        "inheritance": "AR biallelic = NBS; AD heterozygous carrier = mild HL risk",
        "hl_risk": "Biallelic NBS: HL/NHL 60x HIGHEST; Monoallelic 657del5 carrier: HL 2-3x",
        "pathognomonic": "Microcephaly PATHOGNOMONIC NBS biallelic; 657del5 Slavic founder 1in150",
        "key_rule": "RT ABSOLUTELY CI biallelic; no TBI HSCT; cranial RT fatal toxicity NBS; monoallelic: reduce RT dose",
        "key_avoid": "Cranial/mediastinal RT biallelic ABSOLUTELY CI; full-dose alkylating biallelic; TBI HSCT CI",
        "surveillance": "Biallelic: MRI brain q2yr; CBC quarterly; PET/CT q2yr; IVIG if IgG low",
        "targeted_rx": "RIC-HSCT no TBI (biallelic curative); dose-modified OEPA/COPP; gemcitabine salvage; NO alkylating biallelic",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 AD LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-"
            "RAD51-8-BRC-Repeats-ssDNA-"
            "HL-2-3x-HRD-60pct-Cisplatin-Preferred-Olaparib-PARP-"
            "HBOC-OMIM-612555-FA-D1-Biallelic"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / ~384 kDa / 13q12.3 BRCA2 HL molecular context: "
            "STRUCTURE: "
            "  3418 aa / ~384 kDa; HR scaffold/mediator protein; "
            "  8 × BRC repeats (aa 1002-2085): RAD51 binding; "
            "  OB-fold DBD (aa 2402-3190): ssDNA/RPA binding; "
            "  C-terminal RAD51-binding (aa 3196-3418): stabilises RAD51 nucleofilament; "
            "  PALB2-binding domain (aa 10-40): nuclear import; "
            "  Biallelic = Fanconi Anaemia subtype D1 (FA-D1): childhood cancer, AVOID alkylating; "
            "HL MECHANISM: "
            "  HRD (Homologous Recombination Deficiency) 60%: SBS3 signature; "
            "  HL 2-3x in BRCA2 carriers; mechanism — RS cell HRD impairs DDR → RS-cell survival; "
            "  Concurrent breast/ovarian cancer in BRCA2 HL patients (surveillance mandatory); "
            "CLINICAL RULES FOR BRCA2-HL: "
            "  HRD 60%: cisplatin preferred over carboplatin; cisplatin-based salvage (DHAP, ESHAP); "
            "  Olaparib maintenance after ASCT: phase II evidence; "
            "  FA-D1 biallelic: AVOID ALL ALKYLATING ABSOLUTELY; cyclophosphamide, chlorambucil CI; "
            "  FA-D1 biallelic: ABVD very high toxicity — dacarbazine only; no alkylating; "
            "  Annual breast MRI from age 25; ovarian screening transvaginal USS from 30; "
            "  PD-1/PDL1 checkpoint inhibitors (pembrolizumab) active in BRCA2-HL; "
            "KEY VARIANTS: "
            "  c.5946delT (p.Ser1982Asnfs): Ashkenazi founder; c.9976A>T (p.Lys3326Ter): low-penetrance; "
            "  c.6174delT: Ashkenazi; Norwegian c.1832delA; "
            "SURVEILLANCE: "
            "  Annual breast MRI age 25+; annual ovarian CA-125 + USS from 30; "
            "  BRCA2-HL: CBC annual; lymph node exam q6M; "
            "TARGETED Rx: "
            "  Olaparib (PARPi) maintenance post-ASCT; "
            "  Cisplatin-based salvage (DHAP, ESHAP); "
            "  Pembrolizumab PD-1 (FDA 2017 cHL relapsed/refractory); "
        ),
        "syndrome": "HBOC (Hereditary Breast-Ovarian Cancer) / BRCA2-HL",
        "inheritance": "AD LOF heterozygous (HBOC); AR biallelic = FA-D1",
        "hl_risk": "HL 2-3x monoallelic; FA-D1 biallelic: HL/childhood cancer very high; HRD 60%",
        "pathognomonic": "SBS3 mutational signature (HRD 60%); cisplatin hypersensitivity; concurrent breast/ovarian Ca",
        "key_rule": "Cisplatin preferred (HRD); FA-D1 biallelic: AVOID ALKYLATING ABSOLUTELY; olaparib PARP maintenance",
        "key_avoid": "Alkylating agents biallelic FA-D1 ABSOLUTELY CI; carboplatin over cisplatin (reduce HRD benefit)",
        "surveillance": "Annual breast MRI 25+; ovarian USS+CA-125 30+; cascade BRCA2 testing",
        "targeted_rx": "Olaparib PARPi maintenance post-ASCT; cisplatin-based salvage DHAP/ESHAP; pembrolizumab PD-1",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "Tetramer-DBD-OD-TAD-Pro-"
            "HL-3-5x-LFS-AVOID-RADIATION-ABSOLUTELY-WB-MRI-Toronto-"
            "LFS-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 HL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; transcription factor homotetrameric; "
            "  TAD1 (aa 1-40) + TAD2 (aa 40-67): transactivation; "
            "  Proline-rich (aa 67-98): apoptosis modulation; "
            "  DBD (aa 102-292): DNA binding; hotspot mutations R175H/G245S/R248W/R248Q/R249S/R273H/R282W; "
            "  OD (aa 323-356): tetramerisation; "
            "  Regulatory (aa 364-393): nuclear localisation; "
            "  LFS: AD germline TP53 LOF → childhood cancers (sarcoma, breast, brain, HL, adrenal); "
            "HL MECHANISM: "
            "  TP53 LOF → RS-cell escape from apoptosis; del17p somatic 10-15% cHL; "
            "  HL 3-5x in LFS germline TP53; median age HL in LFS: 25-35 years; "
            "  Transformation risk HL→DLBCL 15-20% in TP53 germline; "
            "CLINICAL RULES FOR TP53-HL: "
            "  AVOID RADIATION ABSOLUTELY (LFS): ANY radiation → secondary sarcoma/cancer; "
            "  WB-MRI Toronto Protocol ANNUALLY (whole-body MRI surveillance); "
            "  Venetoclax backbone: p53-independent apoptosis pathway; "
            "  No ABVD + mediastinal RT — chemotherapy-only ABVD or BV-AVD; "
            "  Secondary HL after prior RT in LFS: confirm TP53 germline before re-irradiating; "
            "  Cascade TP53 testing mandatory in LFS families (all first-degree relatives); "
            "KEY VARIANTS: "
            "  R248W/R248Q (hotspot DBD); R175H, G245S, R249S, R273H, R282W; "
            "  Splice site IVS6+1G>A; intron 7 mutations; del17p13.1 large deletion; "
            "SURVEILLANCE: "
            "  WB-MRI annually (Toronto Protocol); annual CBC; breast MRI 20+ women; "
            "  Annual abdominal USS; avoid all CT imaging (cumulative radiation LFS); "
            "TARGETED Rx: "
            "  Venetoclax (BCL-2i): p53-independent apoptosis; "
            "  Pembrolizumab PD-1 salvage; "
            "  CAR-T therapy (axi-cel/tisagenlecleucel) avoid RT consolidation; "
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) / TP53-HL",
        "inheritance": "AD LOF heterozygous",
        "hl_risk": "HL 3-5x; LFS childhood/young adult HL; del17p somatic 10-15%; transformation 15-20%",
        "pathognomonic": "LFS sarcoma+breast+brain+HL cluster; secondary HL after prior RT treatment",
        "key_rule": "AVOID RADIATION ABSOLUTELY; WB-MRI Toronto Protocol annually; venetoclax backbone; chemotherapy-only",
        "key_avoid": "ANY radiation (even palliative RT); consolidation mediastinal RT; CT imaging (cumulative dose)",
        "surveillance": "WB-MRI annually (Toronto Protocol); annual CBC; breast MRI 20+; avoid CT; cascade TP53 testing",
        "targeted_rx": "Venetoclax BCL-2i backbone; pembrolizumab PD-1 salvage; CAR-T avoid RT consolidation",
    },
    {
        "gene": "CHEK2",
        "protein": (
            "CHEK2 -- 22q12.1 AD LOF -- 543aa -- "
            "CHK2-60kDa-ATM-Effector-Kinase-"
            "FHA-SQ-Kinase-"
            "HL-2-3x-1100delC-European-1in200-NOT-HRD-NO-PARPi-"
            "OMIM-604373"
        ),
        "locus": "22q12.1",
        "protein_size": (
            "543 aa / ~60 kDa / 22q12.1 CHEK2 HL molecular context: "
            "STRUCTURE: "
            "  543 aa / ~60 kDa; serine/threonine kinase; ATM effector; "
            "  SQ/TQ motif cluster domain (SCD, aa 19-69): ATM phosphorylation targets (Thr68); "
            "  FHA domain (aa 92-175): phospho-Thr binding; dimer interface; "
            "  Kinase domain (aa 210-486): catalytic; substrates BRCA1/CDC25A/CDC25C/p53; "
            "  Homodimerisation: trans-autophosphorylation Thr383/Thr387; "
            "  1100delC (c.1100del): frameshift at codon 367 — MOST COMMON European LoF; "
            "  IVS2+1G>A: Ashkenazi founder splice variant; "
            "CHEK2 HL CONTEXT: "
            "  CHEK2 LOF: HL 2-3x; NHL 2-3x; breast 2x (women); prostate 2-3x; "
            "  1100delC: 1:200 European general population; ~1:400 carrier; "
            "  ATM + CHEK2 co-occurrence: ADDITIVE 8-10x HL/NHL risk; "
            "  CHEK2 is NOT an HRD gene (not in HR pathway): NO PARPi benefit; "
            "CLINICAL RULES FOR CHEK2-HL: "
            "  NOT HRD — PARPi (olaparib, niraparib) NO benefit; do NOT use; "
            "  ATM+CHEK2 co-carriers: 8-10x HL risk — annual lymphoma surveillance; "
            "  Standard HL therapy (ABVD/BV-AVD/ABVD-escBEACOPP): no dose modification; "
            "  Radiation: standard — CHEK2 does NOT cause radiosensitivity; "
            "  Checkpoint inhibitors pembrolizumab active in CHEK2-HL (checkpoint pathway context); "
            "  Cascade 1100delC testing in European families with HL under 45 years; "
            "KEY VARIANTS: "
            "  1100delC (c.1100delC): European 1in200; most common LOF; "
            "  IVS2+1G>A (c.444+1G>A): Ashkenazi founder; "
            "  c.1283C>T (p.Thr394Met): missense reduced kinase activity; "
            "  del22q12: large deletion — extremely rare; "
            "SURVEILLANCE: "
            "  Annual lymph node exam; annual breast exam + MRI (women CHEK2+BRCA); "
            "  ATM+CHEK2 co-carrier: PET/CT every 2 years from age 30; "
            "TARGETED Rx: "
            "  BV-AVD (brentuximab + AVD): standard first-line for high-risk cHL; "
            "  Pembrolizumab PD-1: salvage relapsed/refractory; "
            "  ASCT consolidation: standard; "
        ),
        "syndrome": "CHEK2-HL (elevated HL/NHL risk; 1100delC European 1in200)",
        "inheritance": "AD LOF heterozygous",
        "hl_risk": "HL 2-3x; NHL 2-3x; ATM+CHEK2 co-carrier 8-10x additive; 1100delC European 1in200",
        "pathognomonic": "No pathognomonic syndromic feature; 1100delC founder mutation in European HL family clustering",
        "key_rule": "NOT HRD — NO PARPi; ATM+CHEK2 co-carrier 8-10x HL risk — annual surveillance from 30yr",
        "key_avoid": "PARPi (olaparib, niraparib) — CHEK2 NOT HRD, no benefit; no dose modification needed",
        "surveillance": "Annual lymph node exam; breast MRI (women); ATM+CHEK2 PET/CT q2yr from 30",
        "targeted_rx": "BV-AVD first-line high-risk cHL; pembrolizumab PD-1 salvage; ASCT consolidation standard",
    },
    {
        "gene": "BLM",
        "protein": (
            "BLM -- 15q26.1 AR LOF -- 1417aa -- "
            "BLM-159kDa-RecQ-Helicase-"
            "SCE-Elevated-PATHOGNOMONIC-"
            "HL-NHL-150-300x-Biallelic-2-3x-Monoallelic-Ashkenazi-1in107-"
            "BS-OMIM-210900"
        ),
        "locus": "15q26.1",
        "protein_size": (
            "1417 aa / ~159 kDa / 15q26.1 BLM RecQ-helicase HL molecular context: "
            "STRUCTURE: "
            "  1417 aa / ~159 kDa; RecQ-family DNA helicase (RECQL3); "
            "  HRDC domain (aa 1212-1292): Holliday junction dissolution; "
            "  Helicase domain (aa 649-1001): 3'→5' unwinding; "
            "  RQC domain (aa 1002-1212): winged-helix; zinc-binding; "
            "  N-terminal (aa 1-648): interaction with MLH1/FANCJ/TopoIIIα; "
            "  BLM + TopoIIIα + RMI1 + RMI2 = BTR complex: dHJ dissolution (prevents sister chromatid exchange); "
            "  2281del6ins7 (c.2207_2212delinsTAGATTC): Ashkenazi Jewish founder; 1:107 carrier; "
            "BLOOM SYNDROME (BS) BIALLELIC PHENOTYPE: "
            "  Proportionate growth retardation; sun-sensitive facial erythema PATHOGNOMONIC; "
            "  SCE (sister chromatid exchanges): ELEVATED 10x PATHOGNOMONIC — diagnostic BN cytogenetics; "
            "  HL/NHL 150-300x HIGHEST risk biallelic; median age lymphoma 20-24 years; "
            "  Diabetes mellitus in 50%; male infertility 100%; "
            "MONOALLELIC CARRIER (ASHKENAZI 1:107): "
            "  Mild elevated HL/NHL 2-3x; no syndromic features; "
            "  2281del6ins7 heterozygote: carrier testing in Ashkenazi HL families; "
            "CLINICAL RULES FOR BLM-HL: "
            "  Biallelic BS: AVOID ALKYLATING ABSOLUTELY (cyclophosphamide, chlorambucil, DTIC); "
            "  Biallelic BS: AVOID RT — extreme radiosensitivity; DNA repair severely impaired; "
            "  Use gemcitabine-based non-DNA-damaging protocols for BS-HL; "
            "  HSCT curative for BS-associated lymphoma: RIC (no myeloablative alkylating/TBI); "
            "  Monoallelic 2281del6ins7: standard HL therapy; "
            "  SCE cytogenetics DIAGNOSTIC if biallelic suspected; "
            "KEY VARIANTS: "
            "  2281del6ins7 (c.2207_2212delinsTAGATTC): Ashkenazi founder 90% BS alleles; "
            "  c.3415C>T (p.Arg1139Ter): European; c.1642C>T (p.Arg548Ter): truncating; "
            "SURVEILLANCE: "
            "  Biallelic BS: annual PET/CT from age 12 (lymphoma); quarterly CBC; "
            "  Annual abdominal USS (Wilms tumour); skin sun-protection mandatory; "
            "  Diabetes screening from age 10 (biallelic); "
            "TARGETED Rx: "
            "  Biallelic: gemcitabine-based adapted protocol; RIC-HSCT curative; "
            "  Monoallelic: BV-AVD standard; pembrolizumab salvage; "
        ),
        "syndrome": "Bloom Syndrome (BS) / BLM-HL",
        "inheritance": "AR biallelic = Bloom syndrome; AD heterozygous carrier = mild HL risk",
        "hl_risk": "Biallelic BS: HL/NHL 150-300x HIGHEST; Monoallelic Ashkenazi 2281del6ins7 carrier: HL 2-3x",
        "pathognomonic": "SCE elevated 10x PATHOGNOMONIC biallelic; facial erythema sun-sensitive; 2281del6ins7 Ashkenazi 1in107",
        "key_rule": "Biallelic: AVOID ALKYLATING ABSOLUTELY; AVOID RT; SCE cytogenetics diagnostic; gemcitabine-based protocol",
        "key_avoid": "Alkylating agents biallelic ABSOLUTELY CI; RT biallelic ABSOLUTELY CI; TBI in HSCT CI",
        "surveillance": "Biallelic: annual PET/CT from 12yr; quarterly CBC; diabetes screening; sun-protection mandatory",
        "targeted_rx": "RIC-HSCT curative (biallelic); gemcitabine-based HL protocol; BV-AVD monoallelic standard",
    },
    {
        "gene": "KLHDC8B",
        "protein": (
            "KLHDC8B -- 3q22.3 AD LOF -- 560aa -- "
            "KLHDC8B-63kDa-Kelch-Domain-Ubiquitin-Adaptor-"
            "NLPHL-Specific-EBV-Negative-Young-Onset-20-40yr-"
            "Anti-CD20-Rituximab-Preferred-R-CHOP-"
            "NLPHL-OMIM-None-Familial"
        ),
        "locus": "3q22.3",
        "protein_size": (
            "560 aa / ~63 kDa / 3q22.3 KLHDC8B NLPHL HL molecular context: "
            "STRUCTURE: "
            "  560 aa / ~63 kDa; Kelch domain-containing protein 8B; "
            "  BTB domain: Cullin3-RING ubiquitin ligase adaptor (CRL3); "
            "  Kelch repeats (6x): substrate recognition domain; "
            "  Function: targeted ubiquitination of centrosome/mitosis proteins; "
            "  KLHDC8B LOF: impaired mitotic checkpoint → aberrant B-cell proliferation; "
            "  Familial NLPHL: rare autosomal dominant; loss-of-function variants segregate with NLPHL; "
            "NLPHL CONTEXT: "
            "  NLPHL = nodular lymphocyte-predominant HL (separate from cHL); "
            "  'Popcorn cells' (lymphocyte-predominant / LP cells): CD20+ CD15- CD30- EBV-; "
            "  KLHDC8B familial NLPHL: young onset 20-40 years; "
            "  EBV-NEGATIVE in NLPHL (contrast cHL EBV+ 30-50%); "
            "  NLPHL transformation to DLBCL 2-5% (lower than cHL but LPHL→DLBCL harder to predict); "
            "CLINICAL RULES FOR KLHDC8B-NLPHL: "
            "  Anti-CD20 rituximab preferred: LP cells CD20+ — rituximab monotherapy effective early stage; "
            "  R-CHOP or R-CVP: rituximab-containing chemotherapy for advanced NLPHL; "
            "  AVOID ABVD for NLPHL (ABVD is cHL-specific; NLPHL is distinct entity); "
            "  PET-CT staging MANDATORY to distinguish cHL from NLPHL; "
            "  LP cell biopsy: CD20+ CD15- CD30- EBV-EBER- PD-L1-; "
            "  Cascade KLHDC8B family testing when familial NLPHL pattern; "
            "  Transformation surveillance: repeat biopsy for clinical change; "
            "KEY VARIANTS: "
            "  KLHDC8B LOF variants in familial NLPHL: frameshift + missense segregating; "
            "  No single dominant founder — private family-specific variants; "
            "  BTB domain variants impairing CRL3 adaptor function; "
            "SURVEILLANCE: "
            "  Annual lymph node examination; PET-CT every 2yr for surveillance; "
            "  Biopsy suspicious nodes (do NOT assume reactive); "
            "  Family members: annual lymph node exam from age 18; "
            "TARGETED Rx: "
            "  Rituximab monotherapy (early NLPHL stage IA/IIA); "
            "  R-CHOP or R-CVP (advanced NLPHL stage III/IV); "
            "  Anti-CD19 CAR-T if NLPHL→DLBCL transformation; "
        ),
        "syndrome": "Familial NLPHL (Nodular Lymphocyte-Predominant HL) / KLHDC8B-NLPHL",
        "inheritance": "AD LOF heterozygous (familial NLPHL)",
        "hl_risk": "NLPHL-specific; young onset 20-40yr; EBV-negative; LP cells CD20+ CD15- CD30-; transformation 2-5%",
        "pathognomonic": "LP (popcorn) cells CD20+ CD15- CD30- EBV-EBER-; EBV-negative NLPHL in family cluster",
        "key_rule": "Anti-CD20 rituximab preferred (LP cells CD20+); AVOID ABVD (cHL-specific, not NLPHL); PET-CT staging mandatory",
        "key_avoid": "ABVD (classical HL regimen — inappropriate for NLPHL); EBV-based risk stratification (NLPHL is EBV-)",
        "surveillance": "Annual lymph node exam; PET-CT q2yr; biopsy suspicious nodes; family: annual exam from 18yr",
        "targeted_rx": "Rituximab monotherapy early NLPHL; R-CHOP/R-CVP advanced; anti-CD19 CAR-T if transformation DLBCL",
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 AD LOF -- 928aa -- "
            "Rb1-110kDa-Cell-Cycle-Regulator-"
            "Pocket-Protein-E2F-Repressor-"
            "Secondary-HL-After-Rb-RT-5-10x-CDK4-6i-INACTIVE-"
            "Hereditary-Retinoblastoma-OMIM-180200"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / ~110 kDa / 13q14.2 RB1 HL molecular context: "
            "STRUCTURE: "
            "  928 aa / ~110 kDa; pocket protein family (pRb/p107/p130); "
            "  Pocket A (aa 379-572) + Pocket B (aa 646-772): 'Pocket domain'; "
            "    Binds E2F transactivation domain; LxCxE (LXCXE) motif binding; "
            "  Spacer region (aa 572-646): LXCXE interface for cyclins, viral oncoproteins; "
            "  C-terminal (aa 772-928): cyclin binding; "
            "  Hypophosphorylated: binds E2F/DP → G1 cell cycle arrest; "
            "  Hyperphosphorylated (CDK4/6-cyclinD): releases E2F → S-phase; "
            "  Germline RB1 LOF: hereditary retinoblastoma (AD); bilateral Rb PATHOGNOMONIC; "
            "  del13q14: somatic in 25-30% MCL (not hereditary HL mechanism directly); "
            "HL MECHANISM: "
            "  Retinoblastoma survivors treated with RT: secondary HL 5-10x at mediastinal/cervical fields; "
            "  RB1 germline: pRb loss allows G1→S bypass → RS-cell survival; "
            "  Secondary HL most common 10-30 years after Rb RT; "
            "  Somatic del13q14 in HL tumour: worse outcome (shorter PFS BV-AVD); "
            "CLINICAL RULES FOR RB1-HL: "
            "  CDK4/6i (palbociclib, ribociclib, abemaciclib) INACTIVE — pRb pathway bypassed; "
            "  Avoid RT in Rb survivors: secondary HL risk 5-10x; if RT needed: lowest dose; "
            "  ABVD or BV-AVD: standard HL therapy (chemotherapy approach preferred over ABVD+RT); "
            "  Secondary HL after Rb RT: confirm histology (cHL or NLPHL); "
            "  Biallelic RB1 (Rb tumour + germline): HSCT considered for aggressive secondary HL; "
            "  Surveillance Rb survivors: annual chest/neck exam + CXR from age 5+ (mediastinal fields); "
            "KEY VARIANTS: "
            "  del13q14 large deletion (bilateral Rb PATHOGNOMONIC); "
            "  c.2T>A (p.Met1?) start loss; c.1981C>T (p.Arg661Ter); "
            "  Intron 17 splice variants; mosaic mutations (lower penetrance); "
            "SURVEILLANCE: "
            "  Rb survivors: annual CXR + lymph node exam; "
            "  PET-CT if clinical suspicion secondary HL; "
            "  Ophthalmology q6M biallelic; bone marrow annual if secondary leukaemia concern; "
            "TARGETED Rx: "
            "  BV-AVD preferred (avoid RT consolidation in Rb survivors); "
            "  Pembrolizumab PD-1 salvage; "
            "  CDK4/6i INACTIVE (pRb pathway lost); venetoclax active (BCL-2 upregulated); "
        ),
        "syndrome": "Hereditary Retinoblastoma (RB) / Secondary HL after Rb RT",
        "inheritance": "AD LOF heterozygous (hereditary Rb)",
        "hl_risk": "Secondary HL 5-10x after Rb RT; del13q14 somatic shorter PFS; CDK4/6i INACTIVE pRb bypassed",
        "pathognomonic": "Bilateral retinoblastoma PATHOGNOMONIC germline RB1; secondary HL 10-30yr post-RT at irradiated field",
        "key_rule": "CDK4/6i INACTIVE (pRb bypassed); avoid RT in Rb survivors (secondary HL risk); BV-AVD preferred",
        "key_avoid": "CDK4/6i (palbociclib, ribociclib, abemaciclib) — INACTIVE; RT consolidation in Rb survivors",
        "surveillance": "Rb survivors: annual CXR + lymph node exam from 5yr; PET-CT clinical suspicion; ophthalmology q6M",
        "targeted_rx": "BV-AVD preferred (avoid RT); venetoclax (BCL-2 active); pembrolizumab PD-1 salvage; CDK4/6i INACTIVE",
    },
]

KEY_CLINICAL_DISTINCTIONS = [
    "ATM-RADIOSENSITIVITY-ABSOLUTE: Even monoallelic ATM reduces RT tolerance — dose-reduce or omit RT; biallelic A-T: RT ABSOLUTELY CI; substitute BV-AVD chemotherapy-only.",
    "NBN-657DEL5-SLAVIC-1IN150: Biallelic NBS = cranial RT absolutely fatal; no TBI in HSCT; RIC-HSCT curative; monoallelic: standard RT dose reduction.",
    "BLM-SCE-PATHOGNOMONIC: Bloom syndrome biallelic: SCE 10x elevated diagnostic; AVOID alkylating ABSOLUTELY; AVOID RT absolutely; gemcitabine-based protocol only.",
    "TP53-AVOID-RADIATION-ABSOLUTELY: LFS germline TP53 — ANY radiation causes secondary sarcoma; WB-MRI Toronto Protocol annually; venetoclax backbone.",
    "KLHDC8B-NLPHL-NOT-CLASSICAL-HL: Familial NLPHL LP cells CD20+ CD15- CD30- EBV-EBER-; ABVD is WRONG for NLPHL; rituximab monotherapy or R-CHOP correct.",
    "BRCA2-HRD-60PCT: HRD signature SBS3 in BRCA2-HL; cisplatin preferred over carboplatin; olaparib PARP maintenance post-ASCT; FA-D1 biallelic: NO alkylating ABSOLUTELY.",
    "CHEK2-NOT-HRD-NO-PARPi: CHEK2 is checkpoint kinase (cell cycle), NOT HR pathway; olaparib/niraparib NO benefit; ATM+CHEK2 co-carrier 8-10x additive HL risk.",
    "RB1-CDK4-6I-INACTIVE: pRb pathway bypassed in RB1 LOF; CDK4/6i (palbociclib, ribociclib, abemaciclib) INACTIVE; venetoclax (BCL-2 upregulated) active; avoid RT in Rb survivors.",
]


def _make_cohort(gene_def: dict, seed: int) -> dict:
    """Generate a 40-patient cHL/NLPHL cohort for one predisposition gene."""
    rng = random.Random(seed)

    gene  = gene_def["gene"]
    locus = gene_def["locus"]
    inh   = gene_def["inheritance"]

    n = 40
    # Age at HL diagnosis
    ages = sorted([rng.gauss(36, 14) for _ in range(n)])
    ages = [max(8, min(82, round(a, 1))) for a in ages]

    # HL subtype (cHL vs NLPHL)
    nlphl_rates = {
        "ATM": 0.15, "NBN": 0.18, "BRCA2": 0.12, "TP53": 0.12,
        "CHEK2": 0.10, "BLM": 0.14, "KLHDC8B": 0.90, "RB1": 0.12,
    }
    subtypes = [
        "NLPHL" if rng.random() < nlphl_rates.get(gene, 0.15) else "cHL"
        for _ in range(n)
    ]

    # EBV status (cHL EBV+ 30-50%; NLPHL EBV-)
    ebv_rates = {
        "ATM": 0.35, "NBN": 0.38, "BRCA2": 0.32, "TP53": 0.30,
        "CHEK2": 0.32, "BLM": 0.28, "KLHDC8B": 0.02, "RB1": 0.32,
    }
    ebv_pos = [
        rng.random() < (0.02 if subtypes[i] == "NLPHL" else ebv_rates.get(gene, 0.35))
        for i in range(n)
    ]

    # Reed-Sternberg / LP cell type
    rs_or_lp = ["LP cells (NLPHL)" if subtypes[i] == "NLPHL" else "RS cells (cHL)" for i in range(n)]

    # del11q somatic (ATM locus involvement in HL)
    del11q_rates = {
        "ATM": 0.30, "NBN": 0.12, "BRCA2": 0.10, "TP53": 0.10,
        "CHEK2": 0.10, "BLM": 0.10, "KLHDC8B": 0.08, "RB1": 0.10,
    }
    del11q = [rng.random() < del11q_rates.get(gene, 0.10) for _ in range(n)]

    # del13q14 (RB1 locus)
    del13q_rates = {
        "ATM": 0.08, "NBN": 0.08, "BRCA2": 0.08, "TP53": 0.10,
        "CHEK2": 0.08, "BLM": 0.08, "KLHDC8B": 0.06, "RB1": 0.30,
    }
    del13q = [rng.random() < del13q_rates.get(gene, 0.08) for _ in range(n)]

    # Transformation to DLBCL (HL → DLBCL in NLPHL, or RS → DLBCL relapse)
    transform_rates = {
        "ATM": 0.12, "NBN": 0.14, "BRCA2": 0.10, "TP53": 0.18,
        "CHEK2": 0.10, "BLM": 0.16, "KLHDC8B": 0.04, "RB1": 0.12,
    }
    transformed = [rng.random() < transform_rates.get(gene, 0.12) for _ in range(n)]

    # Complete response (CR) rates
    cr_rates = {
        "ATM": 0.74, "NBN": 0.68, "BRCA2": 0.76, "TP53": 0.65,
        "CHEK2": 0.78, "BLM": 0.62, "KLHDC8B": 0.88, "RB1": 0.72,
    }
    cr = [rng.random() < cr_rates.get(gene, 0.73) for _ in range(n)]

    # 5yr OS rates
    os5_rates = {
        "ATM": 0.72, "NBN": 0.64, "BRCA2": 0.74, "TP53": 0.62,
        "CHEK2": 0.76, "BLM": 0.58, "KLHDC8B": 0.88, "RB1": 0.70,
    }
    os5 = [rng.random() < os5_rates.get(gene, 0.70) for _ in range(n)]

    # Treatment: ABVD, BV-AVD, NLPHL-R, adapted
    tx_options = {
        "ATM":     ["BV-AVD", "ABVD-noRT", "escBEACOPP"],
        "NBN":     ["BV-AVD-adapted", "ABVD-noRT", "Gemcitabine-based"],
        "BRCA2":   ["BV-AVD", "ABVD", "Cisplatin-DHAP-salvage"],
        "TP53":    ["BV-AVD-noRT", "ABVD-noRT", "Venetoclax-based"],
        "CHEK2":   ["BV-AVD", "ABVD", "escBEACOPP"],
        "BLM":     ["Gemcitabine-based-noRT", "BV-AVD-adapted", "RIC-HSCT"],
        "KLHDC8B": ["Rituximab-mono", "R-CHOP", "R-CVP"],
        "RB1":     ["BV-AVD-noRT", "ABVD-noRT", "Pembrolizumab-salvage"],
    }
    treatments = [rng.choice(tx_options.get(gene, ["ABVD", "BV-AVD", "escBEACOPP"])) for _ in range(n)]

    chl_n  = sum(1 for s in subtypes if s == "cHL")
    nlphl_n = sum(1 for s in subtypes if s == "NLPHL")

    return {
        "gene":           gene,
        "locus":          locus,
        "inheritance":    inh,
        "protein_size":   gene_def["protein_size"],
        "n":              n,
        "avg_age_dx":     round(sum(ages) / n, 1),
        "chl_n":          chl_n,
        "nlphl_n":        nlphl_n,
        "ebv_pos_n":      sum(ebv_pos),
        "del11q_n":       sum(del11q),
        "del13q_n":       sum(del13q),
        "transformed_n":  sum(transformed),
        "cr_n":           sum(cr),
        "os5_n":          sum(os5),
        "tx_bv_avd":      sum(1 for t in treatments if "BV-AVD" in t),
        "tx_abvd":        sum(1 for t in treatments if "ABVD" in t and "BV" not in t),
        "tx_rituximab":   sum(1 for t in treatments if "Rituximab" in t or "R-CHOP" in t or "R-CVP" in t),
        "key_avoid":      gene_def["key_avoid"],
        "key_rule":       gene_def["key_rule"],
        "targeted_rx":    gene_def["targeted_rx"],
        "hl_risk":        gene_def["hl_risk"],
        "pathognomonic":  gene_def["pathognomonic"],
        "surveillance":   gene_def["surveillance"],
    }


def generate_overview() -> dict:
    cohorts = {
        gd["gene"]: _make_cohort(gd, SEED_BASE + i)
        for i, gd in enumerate(ATLAS_GENES)
    }
    total = 320
    cr_total        = sum(c["cr_n"]        for c in cohorts.values())
    os5_total       = sum(c["os5_n"]       for c in cohorts.values())
    transform_t     = sum(c["transformed_n"] for c in cohorts.values())
    ebv_total       = sum(c["ebv_pos_n"]   for c in cohorts.values())
    del11q_t        = sum(c["del11q_n"]    for c in cohorts.values())
    del13q_t        = sum(c["del13q_n"]    for c in cohorts.values())
    chl_t           = sum(c["chl_n"]       for c in cohorts.values())
    nlphl_t         = sum(c["nlphl_n"]     for c in cohorts.values())
    bv_avd_t        = sum(c["tx_bv_avd"]   for c in cohorts.values())
    rituximab_t     = sum(c["tx_rituximab"] for c in cohorts.values())

    return {
        "atlas":            "Hereditary-HL-Predisposition-Atlas",
        "seed_range":       f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":   total,
        "cr_n":             cr_total,
        "cr_pct":           round(cr_total / total * 100, 1),
        "os5_n":            os5_total,
        "os5_pct":          round(os5_total / total * 100, 1),
        "transformed_n":    transform_t,
        "transformed_pct":  round(transform_t / total * 100, 1),
        "ebv_pos_n":        ebv_total,
        "ebv_pos_pct":      round(ebv_total / total * 100, 1),
        "del11q_n":         del11q_t,
        "del11q_pct":       round(del11q_t / total * 100, 1),
        "del13q_n":         del13q_t,
        "del13q_pct":       round(del13q_t / total * 100, 1),
        "chl_n":            chl_t,
        "chl_pct":          round(chl_t / total * 100, 1),
        "nlphl_n":          nlphl_t,
        "nlphl_pct":        round(nlphl_t / total * 100, 1),
        "bv_avd_n":         bv_avd_t,
        "bv_avd_pct":       round(bv_avd_t / total * 100, 1),
        "rituximab_n":      rituximab_t,
        "rituximab_pct":    round(rituximab_t / total * 100, 1),
        "gene_counts": {
            gene: {
                "n":             c["n"],
                "cr_n":          c["cr_n"],
                "os5_n":         c["os5_n"],
                "transformed_n": c["transformed_n"],
                "ebv_pos_n":     c["ebv_pos_n"],
                "del11q_n":      c["del11q_n"],
                "del13q_n":      c["del13q_n"],
                "chl_n":         c["chl_n"],
                "nlphl_n":       c["nlphl_n"],
                "hl_risk":       c["hl_risk"],
            }
            for gene, c in cohorts.items()
        },
    }


def generate_breakdown() -> dict:
    return {
        "atlas":     "Hereditary-HL-Predisposition-Atlas",
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
            f"HL RISK: {gd['hl_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-HL-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": KEY_CLINICAL_DISTINCTIONS,
    }
