#!/usr/bin/env python3
"""Hereditary-Multiple-Myeloma-Predisposition-Atlas -- Complete 8-Gene Reference
FAM46C  (Family with sequence similarity 46 member C; 155aa; 1p12; LOF;
         Familial Multiple Myeloma — first hereditary MM gene (2016); RNA polyadenylase;
         Ashkenazi founder p.S103N; ~10-20% familial MM pedigrees; germline AR/AD;
         seed SEED_BASE+0).
BRCA2   (Breast cancer gene 2; 3418aa; 13q12.3; AD LOF;
         HBOC — MM 2-3x elevated risk; HRD → olaparib/niraparib sensitivity;
         FA-D1 biallelic = AML/childhood PATHOGNOMONIC; POLO paradigm maintenance;
         seed SEED_BASE+1).
ATM     (Ataxia telangiectasia mutated; 3056aa; 11q22.3; AD/AR LOF;
         MM 2x elevated monoallelic; B-cell malignancy CLL 4-5x, NHL 3-5x;
         biallelic A-T = ALL/lymphoma 100x PATHOGNOMONIC; RADIOSENSITIVITY ABSOLUTE;
         seed SEED_BASE+2).
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         LFS — plasma cell leukemia (PCL) and MM elevated; del17p somatic 10-15% MM;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         seed SEED_BASE+3).
TRAF3   (TNF receptor-associated factor 3; 568aa; 14q32.32; AD LOF;
         Hereditary MM — NF-κB non-canonical pathway suppressor; familial MM pedigrees;
         del14q32 somatic ~5-8% sporadic MM; constitutive NF-κB → plasma cell survival;
         seed SEED_BASE+4).
CDKN2A  (Cyclin-dependent kinase inhibitor 2A; 156aa; 9p21.3; AD LOF;
         FAMM — hematologic malignancy including MM; del9p21 somatic 30-40% MM;
         p16/INK4a + p14/ARF dual suppressors; CDK4/6i INACTIVE p16-null;
         seed SEED_BASE+5).
NBN     (Nibrin/NBS1; 754aa; 8q21.3; AD LOF;
         Nijmegen Breakage Syndrome (NBS); 657del5 Slavic founder;
         B-cell NHL 60x HIGHEST; MM 3x elevated; RADIOSENSITIVITY biallelic ABSOLUTE CI;
         seed SEED_BASE+6).
PALB2   (Partner and localiser of BRCA2; 1186aa; 16p12.2; AD LOF;
         HBOC-2 / FANCN; MM in PALB2 families described; breast 53%; ovarian 5-7%;
         FA-N biallelic = pancytopenia + AML; TBCRC048 olaparib 82% ORR;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3526-3533)
"""
import random

SEED_BASE = 3526

ATLAS_GENES = [
    {
        "gene": "FAM46C",
        "protein": (
            "FAM46C -- 1p12 LOF -- 155aa -- "
            "Family-With-Sequence-Similarity-46-Member-C-17kDa-"
            "Noncanonical-PolyA-Polymerase-RNA-Polyadenylase-"
            "Familial-Multiple-Myeloma-FIRST-GENE-Ashkenazi-S103N-Founder-"
            "10-20pct-Familial-MM-Pedigrees-Bortezomib-Sensitivity-OMIM-611652"
        ),
        "locus": "1p12",
        "protein_size": (
            "155 aa / 17 kDa / 1p12 FAM46C multiple myeloma molecular context: "
            "STRUCTURE: "
            "  155 aa / ~17 kDa; nucleotidyl transferase fold (Pol-β superfamily); "
            "  Noncanonical poly(A) polymerase — elongates mRNA poly(A) tails in cytoplasm; "
            "  Unrelated to canonical nuclear PAP (PAPOLA); mRNA stability/translation regulator; "
            "  LOF → impaired cytoplasmic polyadenylation → reduced plasma-cell gene expression; "
            "FAMILIAL MULTIPLE MYELOMA (FMM): "
            "  FAM46C = FIRST identified hereditary MM susceptibility gene (2016, Broderick+Pathak); "
            "  Germline LOF variants in ~10-20% of MM families with ≥2 first-degree relatives; "
            "  Also: somatic FAM46C LOF in ~4-13% sporadic MM (MMRF CoMMpass); "
            "  Ashkenazi Jewish founder variant: p.S103N — carrier frequency ~1% in AJ population; "
            "  Penetrance incompletely defined; MM risk markedly elevated in FAM46C LOF families; "
            "MOLECULAR MECHANISM: "
            "  FAM46C loss → impaired mRNA polyadenylation of key plasma cell regulators; "
            "  Downstream: reduced IRF4, BLIMP1 mRNA stability (plasma-cell differentiation disruption); "
            "  May sensitise myeloma cells to proteasome inhibitors (bortezomib, carfilzomib): "
            "  FAM46C LOF tumours show elevated UPR — proteasome inhibitor vulnerability; "
            "CLINICAL FEATURES: "
            "  Presentation: MM (IgG/IgA kappa/lambda); may overlap Waldenström/LPL in some families; "
            "  No extra-hematologic phenotype described in heterozygous carriers to date; "
            "  Autosomal recessive (biallelic) vs. dominant (haploinsufficient) — still debated; "
            "  CASCADE TESTING: first-degree relatives of FAM46C LOF MM patients — germline panel; "
            "SURVEILLANCE: "
            "  Serum protein electrophoresis (SPEP) + immunofixation + free light chains annually >40yr; "
            "  If MGUS detected: IMWG risk stratification + annual SPEP/FLC; "
            "  MRI spine/pelvis if bone pain or CRAB criteria suspected; "
            "  Family testing: FMM panel (FAM46C + BRCA2 + ATM + TP53 + TRAF3 + CDKN2A + NBN + PALB2)"
        ),
        "syndrome": "Familial Multiple Myeloma (FMM) — FAM46C",
        "inheritance": "LOF (AR biallelic or AD haploinsufficient — debated; somatic also common)",
        "myeloma_risk": "MM 10-20% familial pedigrees; first identified FMM gene (2016); Ashkenazi founder p.S103N ~1% carrier",
        "pathognomonic": "No pathognomonic extramedullary features; FAM46C LOF + MM in ≥2 first-degree relatives = Familial MM",
        "key_avoid": "Do NOT miss FAM46C in Ashkenazi Jewish MM families — p.S103N founder; test on FMM panel, not just BRCA2; somatic FAM46C LOF in sporadic MM does NOT confirm germline — germline testing required separately",
        "key_rule": "FAM46C LOF in a MM patient with ≥1 first-degree relative with MM = FAMILIAL MM. Cascade test all first-degree relatives. Annual SPEP/FLC >40yr. Bortezomib/carfilzomib preferred (FAM46C LOF → UPR → proteasome inhibitor sensitivity hypothesis).",
        "surveillance": "Annual SPEP + immunofixation + FLC chain >40yr in carriers; IMWG MGUS criteria if M-protein detected; MRI spine/pelvis for bone disease; genetic counselling + first-degree relative cascade testing",
        "targeted_rx": "Proteasome inhibitors (bortezomib, carfilzomib) — FAM46C LOF sensitises via UPR; lenalidomide+bortezomib+dexamethasone (VRd) standard induction; ASCT for eligible patients; daratumumab + VRd if high-risk features",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "HR-Scaffold-384kDa-RAD51-Loader-FANCD1-"
            "HBOC-MM-2-3x-Elevated-HRD-Olaparib-Niraparib-Sensitivity-"
            "FA-D1-Biallelic-AML-Childhood-PATHOGNOMONIC-POLO-Maintenance-"
            "Breast-69-72pct-Ovarian-11-17pct-DOMINANT-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 multiple myeloma molecular context: "
            "STRUCTURE: "
            "  384 kDa nuclear protein; 8 BRC repeats (aa 1002-2081): RAD51 binding domain; "
            "  OB fold domain: DNA single-strand binding; C-terminal RAD51 interaction (aa 3266-3330); "
            "  BRCA2 loads RAD51 nucleoprotein filament onto ssDNA at resected DSBs; "
            "  HR scaffold — FANCD1 complementation group; "
            "BRCA2 AND MULTIPLE MYELOMA: "
            "  MM risk: 2-3x elevated in BRCA2 carriers (retrospective cohort data); "
            "  HRD-positive myeloma (SBS3 mutational signature): PARP inhibitor sensitivity; "
            "  Niraparib + pomalidomide: Phase 1/2 activity in relapsed/refractory MM with HRD; "
            "  Olaparib: exploratory activity in BRCA2-mutated myeloma (case reports + basket trials); "
            "  Del13q (includes RB1, BRCA2 region): somatic in ~50% MM — not same as germline BRCA2; "
            "FA-D1 BIALLELIC: "
            "  Biallelic BRCA2 (Fanconi Anemia complementation group D1): "
            "  AML/ALL in childhood = PATHOGNOMONIC; Wilms tumour; medulloblastoma; "
            "  AVOID ALKYLATING AGENTS ABSOLUTELY in FA-D1 (melphalan in MM = contraindicated); "
            "  Sibling exclusion from bone marrow donation — siblings may also be biallelic carriers; "
            "POLO PARADIGM: "
            "  Olaparib maintenance post-platinum in BRCA1/2-mutated pancreatic (POLO trial); "
            "  Analogous maintenance strategy being explored in BRCA2 MM; "
            "BREAST/OVARIAN DOMINANT: "
            "  Breast 69-72%; ovarian 11-17%; pancreatic 5-7%; prostate 15-20%; MM 2-3x; "
            "  RRSO recommended 40-45yr (women); annual breast MRI from 25yr; PSA from 40yr (men)"
        ),
        "syndrome": "HBOC (Hereditary Breast and Ovarian Cancer) / FANCD1 (Fanconi Anemia group D1)",
        "inheritance": "AD LOF monoallelic (HBOC); AR biallelic = FA-D1 (Fanconi Anemia)",
        "myeloma_risk": "MM 2-3x elevated; HRD-positive myeloma → PARP inhibitor sensitivity; del13q somatic ≠ germline BRCA2",
        "pathognomonic": "FA-D1 biallelic: AML/ALL childhood PATHOGNOMONIC; Wilms tumour + medulloblastoma in FA-D1 family = BRCA2 biallelic testing MANDATORY",
        "key_avoid": "AVOID ALKYLATING AGENTS in FA-D1 biallelic (melphalan, cyclophosphamide = CONTRAINDICATED); del13q somatic in 50% MM does NOT confirm germline BRCA2 — germline testing required; sibling exclusion from BMT in FA-D1 families",
        "key_rule": "BRCA2 germline LOF + MM: HRD test on tumour (SBS3/HRDetect) → niraparib/olaparib eligibility. Annual PSA from 40yr (men). RRSO 40-45yr (women). FA-D1: AVOID melphalan ABSOLUTELY — use carfilzomib-based regimens without alkylating.",
        "surveillance": "Annual breast MRI from 25yr; annual mammography from 30yr; RRSO 40-45yr (women); annual PSA from 40yr (men); annual pancreatic MRI/EUS from 50yr (or 10yr before youngest affected); SPEP/FLC annually >40yr",
        "targeted_rx": "Niraparib + pomalidomide (Phase 1/2 HRD myeloma); olaparib (basket trial BRCA2 MM); carfilzomib-based induction (avoid melphalan in BRCA2 HRD); daratumumab + carfilzomib + dexamethasone (KdD) for HRD-positive MM",
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant/AR-LOF -- 3056aa -- "
            "ATM-Kinase-350kDa-PIK3-Like-DNA-DSB-Sensor-"
            "MM-2x-Elevated-Monoallelic-B-Cell-CLL-4-5x-NHL-3-5x-"
            "Biallelic-AT-ALL-Lymphoma-100x-PATHOGNOMONIC-"
            "RADIOSENSITIVITY-ABSOLUTE-Ceralasertib-ATRi-Synthetic-Lethality-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM multiple myeloma molecular context: "
            "STRUCTURE: "
            "  350 kDa PI3-kinase-like kinase (PIKK family); HEAT repeats (N-terminal, 1-1600); "
            "  FAT domain (aa 1960-2566); PI3K catalytic domain (aa 2712-2962); FATC domain (C-terminal); "
            "  Activated by DSBs (autophosphorylation S1981); recruits MRN complex at DSBs; "
            "  Phosphorylates: H2AX (γH2AX), CHK2 (cell cycle arrest), p53, BRCA1, 53BP1; "
            "ATM AND MULTIPLE MYELOMA: "
            "  Monoallelic ATM LOF: MM risk ~2x elevated; B-cell malignancies broadly elevated; "
            "  CLL: 4-5x elevated (strongest monoallelic ATM signal in hematologic malignancy); "
            "  NHL (including DLBCL, mantle cell, follicular): 3-5x elevated; "
            "  MM: del11q22 somatic in ~10% sporadic MM — includes ATM locus; "
            "  Biallelic A-T: ALL/lymphoma 100x PATHOGNOMONIC — confirmed diagnostic criteria; "
            "RADIOSENSITIVITY — CRITICAL: "
            "  Monoallelic heterozygote: intermediate radiosensitivity; "
            "  RT REDUCE 20-30% dose even in monoallelic ATM carriers (myeloma context: avoid TBI); "
            "  Biallelic A-T: RT ABSOLUTELY CONTRAINDICATED — T-cell ALL deaths after RT documented; "
            "  Total body irradiation (TBI) in ATM carriers → catastrophic toxicity; "
            "ATM A-T BIALLELIC: "
            "  Ataxia telangiectasia (A-T): progressive cerebellar ataxia from 2-3yr; "
            "  Oculocutaneous telangiectasias (conjunctival) = PATHOGNOMONIC age 3-6yr; "
            "  IgA deficiency; sinopulmonary infections; immune deficiency; "
            "  ALL/lymphoma 100x elevated lifetime PATHOGNOMONIC; "
            "CERALASERTIB (ATR INHIBITOR): "
            "  ATM-deficient tumours rely on ATR for replication stress response; "
            "  Ceralasertib (AZ6738) + olaparib: synthetic lethality in ATM-deficient myeloma; "
            "  Phase 1/2 ceralasertib combinations in relapsed/refractory myeloma (ongoing)"
        ),
        "syndrome": "Ataxia-Telangiectasia (A-T) — biallelic; ATM heterozygote = B-cell malignancy predisposition",
        "inheritance": "AD LOF monoallelic (MM/NHL/CLL elevated); AR biallelic = A-T syndrome",
        "myeloma_risk": "MM 2x elevated monoallelic; CLL 4-5x, NHL 3-5x; biallelic A-T: ALL/lymphoma 100x PATHOGNOMONIC",
        "pathognomonic": "Oculocutaneous telangiectasias + cerebellar ataxia + ALL/lymphoma = A-T PATHOGNOMONIC; biallelic ATM; del11q22 somatic in MM does NOT confirm germline ATM",
        "key_avoid": "AVOID RT/TBI in ATM biallelic (ABSOLUTELY CONTRAINDICATED); REDUCE RT 20-30% in monoallelic ATM heterozygote; do NOT use TBI conditioning in ATM carriers; del11q22 somatic ≠ germline ATM — test germline separately",
        "key_rule": "ATM germline LOF: avoid TBI in transplant conditioning — carfilzomib-based non-myeloablative preferred. RT reduce 20-30% even in monoallelic. Ceralasertib ATRi + olaparib synthetic lethality in ATM-deficient myeloma. Annual SPEP/FLC >40yr.",
        "surveillance": "Annual SPEP + FLC >40yr; annual CBC for cytopenias; lymph node exam annually; brain MRI if neurological symptoms; avoid radiation procedures where alternative exists (PET-CT vs. CT/MRI)",
        "targeted_rx": "Carfilzomib-based induction (avoid TBI); ceralasertib (ATRi) + olaparib (synthetic lethality ATM-null MM, Phase 1/2); lenalidomide maintenance post-ASCT; avoid RT/TBI in conditioning; daratumumab + KRd for high-risk",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-Tetramer-"
            "LFS-MM-Elevated-Plasma-Cell-Leukemia-PCL-"
            "del17p-Somatic-10-15pct-MM-WORST-Prognosis-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 multiple myeloma molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; tetrameric transcription factor (4×43 kDa); "
            "  N-terminal transactivation domain (TAD1: aa 1-40, TAD2: aa 40-67); "
            "  Proline-rich region (aa 67-98); DNA-binding domain (aa 100-290): most variants; "
            "  Tetramerization domain (aa 323-356); C-terminal regulatory domain; "
            "  MDM2-binding (aa 19-26): MDM2 promotes TP53 ubiquitination and degradation; "
            "TP53 AND MULTIPLE MYELOMA: "
            "  del17p13 (TP53 locus) somatic in 10-15% de novo MM — WORST prognosis cytogenetic feature; "
            "  Biallelic TP53 loss in MM (germline + somatic del17p): plasma cell leukemia (PCL) risk; "
            "  Germline TP53 LOF (LFS): MM risk elevated; plasma cell dyscrasias reported; "
            "  PCL = secondary plasma cell leukemia if biallelic TP53 loss during progression; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  Classic LFS tumours: sarcoma (30%), breast (30%), brain (10%), adrenal (5%), leukemia; "
            "  MM: less common than classic LFS tumours but documented in LFS families; "
            "  WBMRI Toronto Protocol: annual whole-body MRI (standard LFS surveillance); "
            "  Annual breast MRI from 20yr (women in LFS); brain MRI annually; colonoscopy from 25yr; "
            "AVOID RADIATION ABSOLUTELY: "
            "  TP53 LOF → impaired G1/S checkpoint → ionising radiation → accelerated second primaries; "
            "  TBI (total body irradiation) in TP53 germline carriers: secondary sarcoma in RT field; "
            "  Myeloma conditioning: non-myeloablative or melphalan 140mg/m² (not full dose TBI); "
            "  Document TP53 germline status before ANY radiation decision in MM; "
            "del17p CLINICAL SIGNIFICANCE IN MM: "
            "  del17p in MM = high-risk cytogenetics (ISS Stage III equivalence); "
            "  Poor response to lenalidomide maintenance; consider bortezomib-based maintenance; "
            "  Daratumumab improves outcomes in del17p MM: CASTOR/CANDOR/MAIA trial data"
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS)",
        "inheritance": "AD LOF (de novo ~30%; high penetrance; heterozygous sufficient for LFS)",
        "myeloma_risk": "MM elevated in LFS; del17p somatic 10-15% MM = worst cytogenetic prognosis; PCL risk when biallelic TP53 loss",
        "pathognomonic": "del17p somatic MM = high-risk cytogenetics (not germline marker); CPC <5yr + sarcoma + adrenal ACC = LFS PATHOGNOMONIC; R337H Brazilian founder (1 in 300) = population ACC/LFS screening",
        "key_avoid": "AVOID RADIATION ABSOLUTELY in germline TP53 LOF (TBI for ASCT conditioning: CONTRAINDICATED); del17p somatic in MM does NOT confirm germline TP53 — germline test separately; R337H in Brazilian families = population-level screening priority",
        "key_rule": "TP53 germline LOF + MM: WBMRI Toronto protocol annually. AVOID TBI/RT ABSOLUTELY. Melphalan 140mg/m² non-TBI conditioning for ASCT. del17p somatic = daratumumab-based induction + bortezomib maintenance. PCL: intensive induction + tandem transplant strategy.",
        "surveillance": "WBMRI annually (LFS protocol); annual breast MRI from 20yr (women); annual brain MRI; colonoscopy from 25yr; SPEP + FLC annually >40yr; annual adrenal imaging; CBC + differential 6-monthly",
        "targeted_rx": "Daratumumab + VRd for del17p MM (MAIA/CASTOR/CANDOR); bortezomib maintenance (overcomes TP53 resistance better than lenalidomide alone); melphalan 140 non-TBI ASCT conditioning; venetoclax for t(11;14) co-mutation MM; avoid TBI absolutely in germline TP53",
    },
    {
        "gene": "TRAF3",
        "protein": (
            "TRAF3 -- 14q32.32 Autosomal-Dominant-LOF -- 568aa -- "
            "TRAF3-64kDa-NF-κB-Non-Canonical-Suppressor-"
            "Hereditary-MM-NF-κB-Pathway-Familial-Pedigrees-"
            "del14q32-Somatic-5-8pct-Sporadic-MM-"
            "Constitutive-NIK-NF-κB2-Plasma-Cell-Survival-OMIM-601896"
        ),
        "locus": "14q32.32",
        "protein_size": (
            "568 aa / ~64 kDa / 14q32.32 TRAF3 multiple myeloma molecular context: "
            "STRUCTURE: "
            "  568 aa; N-terminal RING domain (E3 ubiquitin ligase); Zinc finger domain; "
            "  TRAF domain (C-terminal): receptor binding + TRAF homodimerisation; "
            "  TRAF3 = adaptor and negative regulator of multiple signalling pathways; "
            "NF-κB NON-CANONICAL PATHWAY: "
            "  Non-canonical NF-κB: NIK (NF-κB-Inducing Kinase) → IKKα → p100 processing → p52/RelB; "
            "  TRAF3 normally suppresses NIK by recruiting cIAP1/2 E3 ligases → NIK ubiquitination → degradation; "
            "  TRAF3 LOF → stabilised NIK → constitutive non-canonical NF-κB → p52/RelB nuclear; "
            "  Plasma cell survival signals: BCMA, APRIL, BAFF → TRAF3 node; "
            "  Del14q32 somatic in ~5-8% sporadic MM (MMRF CoMMpass): TRAF3 locus deletion; "
            "HEREDITARY MM (HMM): "
            "  Germline TRAF3 LOF variants described in familial MM pedigrees (Keats et al.); "
            "  Mechanism: constitutive NIK → NF-κB non-canonical → BCMA upregulation → plasma cell immortalisation; "
            "  TRAF3-null MM cells highly dependent on NF-κB for survival; "
            "  IKKα inhibitor + bortezomib: synthetic lethality hypothesis in TRAF3-null MM; "
            "IMMUNOLOGICAL CONTEXT: "
            "  TRAF3 also regulates: type I interferon signalling (IRF3/IRF7 pathway); "
            "  Toll-like receptor (TLR) signalling; CD40 signalling in B cells; "
            "  TRAF3 LOF → impaired TLR/CD40 signalling → B-cell differentiation dysregulation; "
            "TARGETED VULNERABILITY: "
            "  TRAF3-null MM: constitutive NF-κB → depends on IKK complex → proteasome inhibitor sensitivity; "
            "  Bortezomib/carfilzomib: established sensitivity in NF-κB-activated MM (del14q32 class); "
            "  Anti-BCMA agents (teclistamab, belantamab): BCMA upregulated in TRAF3-null MM; "
            "  Daratumumab (CD38): standard backbone effective independent of TRAF3 status; "
            "  Selinexor (XPO1 inhibitor): activity in NF-κB-activated myeloma"
        ),
        "syndrome": "Hereditary Multiple Myeloma (HMM) — TRAF3 / NF-κB Non-Canonical Pathway",
        "inheritance": "AD LOF (germline TRAF3 variants in familial MM; also somatic del14q32 in sporadic MM)",
        "myeloma_risk": "MM in familial pedigrees (TRAF3 LOF); del14q32 somatic 5-8% sporadic MM; NF-κB non-canonical constitutively activated",
        "pathognomonic": "No unique extramedullary phenotype; TRAF3 LOF + MM in ≥2 relatives = HMM evaluation; del14q32 on FISH in sporadic MM (not germline marker)",
        "key_avoid": "Del14q32 somatic on FISH in MM does NOT confirm germline TRAF3 — germline test required; do NOT miss BCMA upregulation in TRAF3-null MM — teclistamab/belantamab highly relevant; do NOT overlook TRAF3 panel in IgG-kappa MM families",
        "key_rule": "TRAF3 germline LOF + MM: NF-κB-directed therapy (bortezomib/carfilzomib proteasome inhibitors — constitutive NF-κB sensitivity). Anti-BCMA therapy (teclistamab, belantamab, elranatamab) highly relevant in TRAF3-null MM. Selinexor for NF-κB-activated relapsed disease.",
        "surveillance": "Annual SPEP + immunofixation + FLC >40yr; annual CBC; cascade genetic testing first-degree relatives; FISH panel at diagnosis (del14q32 somatic confirms NF-κB activation class); PET-CT for staging at diagnosis",
        "targeted_rx": "Proteasome inhibitors (bortezomib/carfilzomib) — NF-κB constitutive sensitivity; anti-BCMA bispecifics (teclistamab/elranatamab) — BCMA upregulated TRAF3-null; belantamab mafodotin (BCMA-ADC); selinexor + dexamethasone (NF-κB nuclear export inhibition); daratumumab backbone standard",
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4A-p14-ARF-Dual-Tumour-Suppressor-17kDa-"
            "FAMM-Hematologic-Malignancy-Including-MM-"
            "del9p21-Somatic-30-40pct-MM-CDK4-6i-INACTIVE-p16-Null-"
            "Pancreatic-17-39pct-MANDATORY-EUS-40yr-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 17 kDa / 9p21.3 CDKN2A multiple myeloma molecular context: "
            "STRUCTURE: "
            "  Locus encodes two distinct tumour suppressors via alternative reading frames: "
            "  p16/INK4a (156 aa): inhibits CDK4 and CDK6 → Rb remains hypophosphorylated → G1 arrest; "
            "  p14/ARF (132 aa): binds MDM2 → prevents MDM2-mediated p53 degradation → TP53 stabilisation; "
            "  LOF at CDKN2A disables BOTH p16-Rb AND p14-MDM2-p53 axes simultaneously; "
            "CDKN2A AND MULTIPLE MYELOMA: "
            "  del9p21 somatic in 30-40% of MM (MMRF CoMMpass, IFM studies): intermediate frequency; "
            "  Germline CDKN2A LOF (FAMM): hematologic malignancies including MM described; "
            "  CDK4/6 inhibitor RESISTANCE: p16/INK4a null → CDK4/6 constitutively active → palbociclib INEFFECTIVE; "
            "  KEY PITFALL: palbociclib (CDK4/6i) ineffective in CDKN2A-null MM — do NOT use; "
            "FAMM SYNDROME (Familial Atypical Multiple Mole Melanoma): "
            "  Melanoma: 25-36% lifetime; pancreatic: 17-39% MANDATORY EUS surveillance; "
            "  Multiple dysplastic naevi (atypical moles) >50: PATHOGNOMONIC phenotype; "
            "  Annual total-body skin exam from teenage years; annual pancreatic MRI/EUS from 40yr; "
            "  Hematologic malignancy: lymphoma, MM, CLL elevated in FAMM families; "
            "PANCREATIC CANCER KEY RULE: "
            "  Annual MRI/EUS pancreas from 40yr (or 10yr before youngest FAMM-related PDAC in family); "
            "  CDKN2A pancreatic risk 17-39% HIGHEST hereditary PDAC syndrome; "
            "  Sun protection ABSOLUTE — UV avoidance (melanoma primary prevention); "
            "CDK4/6i RESISTANCE CLINICAL IMPLICATIONS: "
            "  Palbociclib, ribociclib, abemaciclib: ALL ineffective in p16-null tumours; "
            "  CDKN2A somatic in MM → do NOT add CDK4/6i to MM backbone; "
            "  t(11;14) MM: cyclin D1 overexpression → CDK4/6 pathway activated → venetoclax ± PI preferred"
        ),
        "syndrome": "FAMM (Familial Atypical Multiple Mole Melanoma) / P16-Leiden syndrome",
        "inheritance": "AD LOF (heterozygous sufficient; high penetrance for melanoma and pancreatic cancer by 70yr)",
        "myeloma_risk": "MM/hematologic malignancy in FAMM families; del9p21 somatic 30-40% sporadic MM; CDK4/6i INACTIVE when p16-null",
        "pathognomonic": ">50 dysplastic/atypical naevi + melanoma in ≥2 relatives = FAMM PATHOGNOMONIC; annual total-body skin exam from adolescence",
        "key_avoid": "CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib) INEFFECTIVE in CDKN2A-null MM — do NOT use; del9p21 somatic does NOT confirm germline — germline test separately; do NOT miss annual pancreatic EUS/MRI from 40yr in FAMM",
        "key_rule": "CDKN2A germline LOF + MM: Annual pancreatic MRI/EUS from 40yr (MANDATORY). Total-body skin exam annually. CDK4/6i INEFFECTIVE — use PI/IMiD/daratumumab backbone. t(11;14) co-mutation: venetoclax + bortezomib highly active (cyclin D1 overexpression).",
        "surveillance": "Annual total-body skin exam + dermatology from adolescence; annual pancreatic MRI/EUS from 40yr; annual SPEP/FLC >40yr; ophthalmology for uveal melanoma screening; sun protection ABSOLUTE",
        "targeted_rx": "Venetoclax + bortezomib (t(11;14) co-mutation CDKN2A MM — cyclin D1 pathway); daratumumab backbone (standard); carfilzomib-based for high-risk cytogenetics; AVOID CDK4/6i (CDKN2A-null); ASCT for eligible; immunotherapy checkpoint AVOID (melanoma surveillance concurrent)",
    },
    {
        "gene": "NBN",
        "protein": (
            "NBN -- 8q21.3 Autosomal-Dominant-LOF -- 754aa -- "
            "Nibrin-NBS1-85kDa-MRN-Complex-MRE11-RAD50-NBS1-"
            "NBS-Nijmegen-Breakage-Syndrome-657del5-Slavic-Founder-"
            "B-cell-NHL-60x-HIGHEST-MM-3x-Elevated-"
            "RADIOSENSITIVITY-Biallelic-ABSOLUTE-Monoallelic-Moderate-OMIM-602667"
        ),
        "locus": "8q21.3",
        "protein_size": (
            "754 aa / 85 kDa / 8q21.3 NBN multiple myeloma molecular context: "
            "STRUCTURE: "
            "  754 aa / 85 kDa; Forkhead-associated (FHA) domain (aa 24-108): phosphopeptide binding; "
            "  BRCA1 C-terminus (BRCT) domain (aa 114-196): phosphopeptide binding; "
            "  MRE11-binding domain (aa 734-754): anchors NBS1 to MRE11-RAD50 complex; "
            "  MRN complex (MRE11-RAD50-NBS1): initial DSB sensor → ATM recruitment → γH2AX signalling; "
            "NBN AND B-CELL MALIGNANCY: "
            "  Monoallelic NBN LOF: B-cell NHL 60x HIGHEST — strongest haploinsufficiency signal; "
            "  Multiple myeloma: ~3x elevated (monoallelic NBN); "
            "  CLL: 2-4x elevated; lymphoma (DLBCL, mantle cell): 10-20x; "
            "  Biallelic NBS: B-cell lymphoma near-universal by 20yr — essentially PATHOGNOMONIC; "
            "NBS SYNDROME (BIALLELIC): "
            "  Microcephaly at birth (PATHOGNOMONIC in NBS); bird-like face; growth retardation; "
            "  Primary immunodeficiency: recurrent sinopulmonary infections (IgA/IgG deficiency); "
            "  Ovarian failure; café-au-lait macules; "
            "  B-cell NHL by 20yr: near-universal in biallelic NBS; "
            "657del5 SLAVIC FOUNDER: "
            "  NBS1 c.657_661del5 (p.K219fs) — founder variant in Czech, Slovak, Polish, Ukrainian; "
            "  Carrier frequency: ~1 in 150 in Czech/Slovak population; "
            "  Monoallelic 657del5: NHL/lymphoma 2-5x; MM 3x elevated; prostate 3-4x elevated; "
            "  Biallelic: NBS syndrome with near-universal lymphoma; "
            "RADIOSENSITIVITY: "
            "  Biallelic NBS: RT ABSOLUTELY CONTRAINDICATED — deaths from standard RT dose documented; "
            "  Monoallelic NBN heterozygote: moderate radiosensitivity — RT reduce 20-30%; "
            "  TBI (total body irradiation) in NBN biallelic: ABSOLUTELY CONTRAINDICATED; "
            "  Recommend non-TBI conditioning for any ASCT in NBN LOF patients; "
            "MRN COMPLEX TARGETED THERAPY: "
            "  MRN-deficient cells (NBN LOF): ATM pathway impaired → PARP inhibitor sensitivity; "
            "  Olaparib/niraparib: MRN-deficient myeloma (exploratory); "
            "  Ceralasertib (ATRi): ATR compensates in MRN-deficient cells → synthetic lethality"
        ),
        "syndrome": "Nijmegen Breakage Syndrome (NBS) — biallelic; NBN heterozygote = B-cell malignancy predisposition",
        "inheritance": "AD LOF monoallelic (NHL/MM elevated); AR biallelic = NBS syndrome",
        "myeloma_risk": "MM 3x elevated monoallelic; B-cell NHL 60x HIGHEST monoallelic; biallelic NBS: lymphoma near-universal <20yr",
        "pathognomonic": "Microcephaly + bird-face + immunodeficiency + B-cell NHL = NBS PATHOGNOMONIC; 657del5 Slavic founder in Czech/Slovak/Polish families; biallelic NBS lymphoma near-universal",
        "key_avoid": "RT/TBI ABSOLUTELY CONTRAINDICATED in biallelic NBS; REDUCE RT 20-30% in monoallelic NBN heterozygote; avoid TBI conditioning in NBN carriers; 657del5 monoallelic ≠ NBS syndrome — risk elevated but NOT biallelic features",
        "key_rule": "NBN 657del5 (Slavic founder) + MM: avoid TBI conditioning. RT reduce 20-30% monoallelic. PARP inhibitor sensitivity (MRN complex defect — olaparib/niraparib). B-cell NHL surveillance: annual exam + CBC. Cascade test in Czech/Slovak/Polish MM families.",
        "surveillance": "Annual CBC + differential; annual lymph node exam; annual SPEP + FLC >40yr; annual prostate PSA (men >40yr); avoid chest CT/PET unless clinical need (minimise radiation exposure); cascade test 657del5 in Slavic MM families",
        "targeted_rx": "PARP inhibitors (olaparib/niraparib) — MRN complex defect sensitivity; ceralasertib (ATRi) synthetic lethality in NBN-null MM; carfilzomib-based induction (avoid TBI); daratumumab standard backbone; B-cell lymphoma: R-CHOP (reduce RT component); consider venetoclax if t(11;14) co-mutation",
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- "
            "PALB2-131kDa-WD40-BRCA2-Bridge-FANCN-"
            "HBOC-2-Breast-53pct-Ovarian-5-7pct-Pancreatic-2-4x-"
            "MM-Described-FA-N-Biallelic-Pancytopenia-AML-"
            "TBCRC048-Olaparib-82pct-ORR-HIGHEST-OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa / 16p12.2 PALB2 multiple myeloma molecular context: "
            "STRUCTURE: "
            "  1186 aa / 131 kDa; N-terminal coiled-coil domain (aa 1-40): BRCA1 binding; "
            "  Central region: chromatin-association motif; nuclear localisation; "
            "  WD40 repeat domain (aa 853-1186): BRCA2 interaction; "
            "  PALB2 bridges BRCA1 (N-terminal) with BRCA2 (C-terminal) in HR repair complex; "
            "  FANCN complementation group; "
            "PALB2 AND MULTIPLE MYELOMA: "
            "  MM risk: elevated in PALB2 families — case series and familial aggregation data; "
            "  Mechanism: HRD (homologous recombination deficiency) → genomic instability → MM; "
            "  SBS3 signature (HRD): PALB2 LOF tumours carry SBS3 — PARP inhibitor sensitivity; "
            "  Niraparib/olaparib: active in PALB2-BRCA1/2 germline cancers (TBCRC048); "
            "FA-N BIALLELIC: "
            "  Fanconi Anemia complementation group N (FA-N): biallelic PALB2; "
            "  Aplastic anaemia, pancytopenia from early childhood; AML risk; "
            "  AVOID ALKYLATING AGENTS ABSOLUTELY in FA-N (melphalan, cyclophosphamide); "
            "  Sibling BMT exclusion (siblings may also be FA-N biallelic); "
            "TBCRC048 DATA: "
            "  TBCRC048 trial: olaparib monotherapy in PALB2 germline (+ BRCA1/2) HER2-negative BC; "
            "  Overall response rate 82% — HIGHEST among non-BRCA2 HRD genes; "
            "  Extrapolation to PALB2 MM: olaparib/niraparib highly likely active (HRD basis); "
            "  Phase 1/2 PALB2 MM olaparib/niraparib: ongoing evaluation; "
            "DOMINANT CANCER RISKS: "
            "  Breast: 53% lifetime (women) — HIGHEST non-BRCA2 HRD gene; "
            "  Ovarian: 5-7% lifetime; pancreatic: 2-4x elevated; prostate: 2-3x; "
            "  Annual breast MRI from 25-30yr; RRSO 45-50yr (women); annual pancreatic MRI/EUS from 50yr; "
            "  Annual PSA from 40yr (men)"
        ),
        "syndrome": "HBOC-2 (PALB2-associated Hereditary Breast/Ovarian Cancer) / FANCN (Fanconi Anemia N)",
        "inheritance": "AD LOF monoallelic (HBOC-2); AR biallelic = FA-N (Fanconi Anemia)",
        "myeloma_risk": "MM described in PALB2 families (elevated); HRD → olaparib/niraparib sensitivity (TBCRC048 82% ORR extrapolation); FA-N biallelic = AML + pancytopenia PATHOGNOMONIC",
        "pathognomonic": "FA-N biallelic: childhood aplastic anaemia + AML = PATHOGNOMONIC; TBCRC048 82% ORR olaparib = HIGHEST HRD PARP response rate; annual breast MRI from 25-30yr (dominant risk)",
        "key_avoid": "AVOID ALKYLATING AGENTS (melphalan, cyclophosphamide) in FA-N biallelic — ABSOLUTELY CONTRAINDICATED; sibling BMT exclusion in FA-N; do NOT miss annual breast MRI (53% lifetime breast risk dominates over MM risk)",
        "key_rule": "PALB2 germline LOF + MM: HRD test (SBS3/HRDetect) → niraparib/olaparib eligibility (TBCRC048 basis). Annual breast MRI from 25-30yr (women). RRSO 45-50yr. Avoid melphalan in FA-N biallelic — use carfilzomib-based regimen. Cascade germline test.",
        "surveillance": "Annual breast MRI from 25-30yr (women); mammography from 30yr; RRSO 45-50yr; annual PSA from 40yr (men); annual pancreatic MRI/EUS from 50yr; annual SPEP + FLC >40yr; CBC annually for marrow reserve monitoring",
        "targeted_rx": "Olaparib/niraparib (PARP inhibitor — HRD SBS3 PALB2; TBCRC048 extrapolation for MM); carfilzomib-based induction (avoid melphalan in FA-N); daratumumab + KRd (high-risk MM); ASCT eligible with carfilzomib conditioning (non-alkylating if FA-N); anti-BCMA therapy for relapsed disease",
    },
]


def _make_patient(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)

    # Gene-specific MM characteristics
    mm_params = {
        "FAM46C": dict(age_mu=60, age_sd=8,  mm_prob=0.80, ighv_somatic=True,  iss3_prob=0.30, del17p_prob=0.10, t_11_14_prob=0.15, carfilzomib_benefit=True),
        "BRCA2":  dict(age_mu=63, age_sd=9,  mm_prob=0.75, ighv_somatic=True,  iss3_prob=0.35, del17p_prob=0.12, t_11_14_prob=0.12, carfilzomib_benefit=True),
        "ATM":    dict(age_mu=61, age_sd=10, mm_prob=0.70, ighv_somatic=True,  iss3_prob=0.30, del17p_prob=0.08, t_11_14_prob=0.10, carfilzomib_benefit=True),
        "TP53":   dict(age_mu=58, age_sd=11, mm_prob=0.65, ighv_somatic=True,  iss3_prob=0.50, del17p_prob=0.60, t_11_14_prob=0.08, carfilzomib_benefit=True),
        "TRAF3":  dict(age_mu=62, age_sd=9,  mm_prob=0.78, ighv_somatic=True,  iss3_prob=0.32, del17p_prob=0.06, t_11_14_prob=0.20, carfilzomib_benefit=True),
        "CDKN2A": dict(age_mu=64, age_sd=10, mm_prob=0.68, ighv_somatic=True,  iss3_prob=0.28, del17p_prob=0.08, t_11_14_prob=0.40, carfilzomib_benefit=False),
        "NBN":    dict(age_mu=55, age_sd=12, mm_prob=0.72, ighv_somatic=True,  iss3_prob=0.35, del17p_prob=0.10, t_11_14_prob=0.12, carfilzomib_benefit=True),
        "PALB2":  dict(age_mu=62, age_sd=9,  mm_prob=0.70, ighv_somatic=True,  iss3_prob=0.30, del17p_prob=0.10, t_11_14_prob=0.14, carfilzomib_benefit=True),
    }
    p = mm_params[gene]

    age = max(28, min(88, round(rng.gauss(p["age_mu"], p["age_sd"]), 1)))
    has_mm        = rng.random() < p["mm_prob"]
    is_iss3       = rng.random() < p["iss3_prob"]
    del17p        = rng.random() < p["del17p_prob"]
    t_11_14       = rng.random() < p["t_11_14_prob"]
    hrd_positive  = gene in ("BRCA2", "ATM", "NBN", "PALB2") and rng.random() < 0.60
    parp_eligible = hrd_positive
    vrd_response  = rng.random() < (0.85 if not del17p else 0.55)

    tumour_types = ["Multiple Myeloma", "MGUS progressed to MM", "Smouldering MM progressed",
                    "Plasma Cell Leukemia", "Waldenström/LPL", "Plasmacytoma"]
    tumour_probs  = [0.68, 0.10, 0.08, 0.05, 0.05, 0.04]
    cumulative = 0.0; r = rng.random()
    tumour_type = tumour_types[-1]
    for tt, tp in zip(tumour_types, tumour_probs):
        cumulative += tp
        if r < cumulative:
            tumour_type = tt; break

    return {
        "gene":           gene,
        "seed":           seed,
        "age_at_dx":      age,
        "tumour_type":    tumour_type,
        "has_mm":         has_mm,
        "iss3":           is_iss3,
        "del17p":         del17p,
        "t_11_14":        t_11_14,
        "hrd_positive":   hrd_positive,
        "parp_eligible":  parp_eligible,
        "vrd_response":   vrd_response,
    }


def _build_cohorts() -> dict:
    cohorts = {}
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        cohorts[gene] = [_make_patient(gene, SEED_BASE + i + j * 8) for j in range(40)]
    return cohorts


def generate_overview() -> dict:
    cohorts = _build_cohorts()
    all_pts = [p for pts in cohorts.values() for p in pts]

    mm_n        = sum(1 for p in all_pts if p["has_mm"])
    del17p_n    = sum(1 for p in all_pts if p["del17p"])
    t_11_14_n   = sum(1 for p in all_pts if p["t_11_14"])
    hrd_n       = sum(1 for p in all_pts if p["hrd_positive"])
    parp_elig_n = sum(1 for p in all_pts if p["parp_eligible"])
    mean_age    = round(sum(p["age_at_dx"] for p in all_pts) / len(all_pts), 1)

    gene_summary = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        mm_gene_n = sum(1 for p in pts if p["has_mm"])
        gene_summary.append({
            "gene":           gene,
            "locus":          gd["locus"],
            "n":              len(pts),
            "mm_n":           mm_gene_n,
            "mm_pct":         round(100 * mm_gene_n / len(pts), 1),
            "mean_age_onset": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "del17p_n":       sum(1 for p in pts if p["del17p"]),
            "hrd_n":          sum(1 for p in pts if p["hrd_positive"]),
            "inheritance":    gd["inheritance"],
            "key_distinctions": [
                gd["myeloma_risk"][:80],
                gd["key_rule"][:80],
            ],
        })

    genes_detail = []
    for gd in ATLAS_GENES:
        genes_detail.append({
            "gene":           gd["gene"],
            "protein":        gd["protein"],
            "locus":          gd["locus"],
            "protein_size":   gd["protein_size"],
            "inheritance":    gd["inheritance"],
            "cancer_risk":    gd["myeloma_risk"],
            "pathognomonic":  gd["pathognomonic"],
            "surveillance_key": gd["surveillance"][:160],
            "key_distinctions": [
                gd["myeloma_risk"][:80],
                gd["key_avoid"][:80],
                gd["key_rule"][:80],
            ],
        })

    return {
        "atlas":                 "Hereditary-Multiple-Myeloma-Predisposition-Atlas",
        "seed_range":            f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":        len(all_pts),
        "genes_n":               8,
        "mm_total_n":            mm_n,
        "mm_total_pct":          round(100 * mm_n / len(all_pts), 1),
        "mean_age_onset":        mean_age,
        "del17p_n":              del17p_n,
        "del17p_pct":            round(100 * del17p_n / len(all_pts), 1),
        "t_11_14_n":             t_11_14_n,
        "t_11_14_pct":           round(100 * t_11_14_n / len(all_pts), 1),
        "hrd_n":                 hrd_n,
        "hrd_pct":               round(100 * hrd_n / len(all_pts), 1),
        "parp_eligible_n":       parp_elig_n,
        "parp_eligible_pct":     round(100 * parp_elig_n / len(all_pts), 1),
        "gene_summary":          gene_summary,
        "genes_detail":          genes_detail,
    }


def _is_mm(tumour_type: str) -> bool:
    return "Myeloma" in tumour_type or "Myeloma" in tumour_type or "MM" in tumour_type or "Plasmacytoma" in tumour_type or "Waldenström" in tumour_type or "Plasma Cell" in tumour_type


def generate_breakdown() -> dict:
    cohorts = _build_cohorts()
    _gene_pathog = {gd["gene"]: gd["pathognomonic"] for gd in ATLAS_GENES}
    _gene_surv   = {gd["gene"]: gd["surveillance"][:120] for gd in ATLAS_GENES}
    _gene_dist   = {gd["gene"]: [gd["myeloma_risk"][:80], gd["key_rule"][:80]] for gd in ATLAS_GENES}

    breakdown_array = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        mm_n = sum(1 for p in pts if p["has_mm"])
        breakdown_array.append({
            "gene":             gene,
            "locus":            gd["locus"],
            "n":                len(pts),
            "mean_age_onset":   round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "mm_n":             mm_n,
            "mm_pct":           round(100 * mm_n / len(pts), 1),
            "del17p_n":         sum(1 for p in pts if p["del17p"]),
            "t_11_14_n":        sum(1 for p in pts if p["t_11_14"]),
            "hrd_n":            sum(1 for p in pts if p["hrd_positive"]),
            "parp_eligible_n":  sum(1 for p in pts if p["parp_eligible"]),
            "seed":             SEED_BASE + i,
            "key_distinctions": _gene_dist[gene],
            "pathognomonic":    _gene_pathog[gene],
            "surveillance_key": _gene_surv[gene],
        })
    return {"breakdown": breakdown_array, "total_patients": sum(len(v) for v in cohorts.values())}


def generate_definitions() -> dict:
    defs = {}
    for gd in ATLAS_GENES:
        defs[gd["gene"]] = {
            "gene":          gd["gene"],
            "locus":         gd["locus"],
            "protein_size":  gd["protein_size"],
            "syndrome":      gd["syndrome"],
            "inheritance":   gd["inheritance"],
            "myeloma_risk":  gd["myeloma_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_avoid":     gd["key_avoid"],
            "key_rule":      gd["key_rule"],
            "surveillance":  gd["surveillance"],
            "targeted_rx":   gd["targeted_rx"],
        }

    key_distinctions = [
        "FAM46C vs BRCA2 familial MM: FAM46C=first dedicated FMM gene, Ashkenazi S103N founder; BRCA2=dominant HRD signal with olaparib/niraparib evidence (POLO paradigm)",
        "BRCA2 vs PALB2 HRD myeloma: BRCA2=del13q somatic 50% MM does NOT confirm germline; PALB2=TBCRC048 82% ORR olaparib HIGHEST HRD response; both avoid alkylating FA biallelic",
        "ATM vs NBN radiosensitivity: both = AVOID TBI; ATM monoallelic RT reduce 20-30%; NBN 657del5 Slavic founder B-cell NHL 60x HIGHEST — strongest monoallelic lymphoma signal",
        "TP53 del17p somatic vs germline: del17p somatic in 10-15% MM ≠ germline LFS; germline TP53=AVOID RADIATION ABSOLUTELY + WBMRI Toronto; somatic del17p=daratumumab+bortezomib maintenance preferred",
        "TRAF3 NF-κB pathway: TRAF3 LOF → constitutive NIK → NF-κB non-canonical → BCMA upregulated → teclistamab/belantamab highly relevant; proteasome inhibitor sensitivity (PI-based induction preferred)",
        "CDKN2A CDK4/6i resistance: p16/INK4a null → CDK4/6 pathway bypassed → palbociclib INEFFECTIVE; t(11;14) co-mutation → venetoclax + bortezomib highly active (cyclin D1 overexpression)",
        "NBN 657del5 Slavic founder: carrier freq 1/150 Czech/Slovak/Polish; test NBN specifically in Slavic MM families; B-cell NHL 60x monoallelic = HIGHEST haploinsufficiency lymphoma signal",
        "Universal HRD testing in hereditary MM: BRCA2+ATM+NBN+PALB2 all contribute to HRD signature; SBS3/HRDetect on tumour → niraparib/olaparib eligibility criteria",
        "Radiosensitivity cascade in hereditary MM: TP53(AVOID RT absolutely)+ATM(reduce 20-30%)+NBN biallelic(ABSOLUTELY CI)+ATM biallelic(ABSOLUTELY CI) — TBI in ASCT must be reviewed against germline status",
        "Anti-BCMA therapy applicability: TRAF3 LOF → BCMA upregulated; FAM46C LOF → UPR activation; teclistamab/elranatamab/belantamab highly relevant in hereditary MM with TRAF3/FAM46C LOF",
    ]

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"MYELOMA RISK: {gd['myeloma_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": key_distinctions,
        "seed_range":                f"{SEED_BASE}–{SEED_BASE + 7}",
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(json.dumps({k: v for k, v in ov.items() if k != "genes_detail"}, indent=2))
    print("\n=== BREAKDOWN (first 2 genes) ===")
    br = generate_breakdown()
    print(json.dumps({"breakdown": br["breakdown"][:2], "total_patients": br["total_patients"]}, indent=2))
    print("\n=== DEFINITIONS (first key distinction) ===")
    df = generate_definitions()
    print(df["key_clinical_distinctions"][0])
