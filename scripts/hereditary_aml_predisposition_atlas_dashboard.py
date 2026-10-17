#!/usr/bin/env python3
"""Hereditary-AML-Predisposition-Atlas -- Complete 8-Gene Reference
CEBPA   (CCAAT/enhancer-binding protein alpha; 358aa; 19q13.11; AD biallelic GOF/LOF;
         Familial AML CEBPA (OMIM 601626); biallelic CEBPA AML 10-15% de novo AML;
         N-terminal (bZIP inhibitor) LOF + C-terminal (bZIP DNA-binding) GOF biallelic;
         COMPLETE REMISSION 90% biallelic CEBPA; RELAPSE RISK 30-50% lifetime;
         seed SEED_BASE+0).
RUNX1   (Runt-related transcription factor 1; 453aa; 21q22.12; AD LOF;
         FPD/AML RUNX1 (OMIM 601399); Familial Platelet Disorder + myeloid malignancy;
         AML/MDS lifetime 30-50%; thrombocytopenia PATHOGNOMONIC FPD;
         seed SEED_BASE+1).
GATA2   (GATA binding protein 2; 480aa; 3q21.3; AD LOF;
         GATA2 deficiency (OMIM 614286); MonoMAC syndrome; AML/MDS 30-70x;
         lymphedema PATHOGNOMONIC; MDS/AML median age 30yr;
         seed SEED_BASE+2).
DDX41   (DEAD-box helicase 41; 622aa; 5q35.3; AD LOF;
         Familial MDS/AML DDX41 (OMIM 608170); late onset 55-65yr HALLMARK;
         R525H splice site most common germline; biallelic somatic second-hit 10-20%;
         seed SEED_BASE+3).
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         LFS (OMIM 151623); AML 10-20x; AVOID RADIATION ABSOLUTELY;
         WBMRI Toronto annually; alkylating agent CI LFS;
         seed SEED_BASE+4).
BRCA2   (Breast cancer susceptibility protein 2; 3418aa; 13q12.3; AD LOF;
         HBOC-2 / FA-D1 biallelic (OMIM 600185); AML/ALL HIGHEST biallelic FA-D1;
         AVOID ALKYLATING AGENTS ABSOLUTELY biallelic; PARP inhibitor olaparib HRD;
         seed SEED_BASE+5).
FANCA   (Fanconi anemia complementation group A; 1455aa; 16q24.3; AR LOF;
         Fanconi Anemia A (OMIM 227650); most common FA ~65%; AML/MDS 50x biallelic;
         AVOID ALKYLATING AGENTS ABSOLUTELY biallelic; bone marrow failure CARDINAL;
         seed SEED_BASE+6).
NF1     (Neurofibromin 1; 2839aa; 17q11.2; AD LOF;
         NF1 syndrome (OMIM 162200); JMML 200-500x HIGHEST paediatric / AML 5-10x;
         Ras/MAPK constitutive; café-au-lait macules PATHOGNOMONIC;
         selumetinib FDA2020; MEK inhibitor active JMML;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3566-3573)
"""
import random

SEED_BASE = 3566

ATLAS_GENES = [
    {
        "gene": "CEBPA",
        "protein": (
            "CEBPA -- 19q13.11 AD biallelic GOF/LOF -- 358aa -- "
            "CEBPA-42kDa-bZIP-TF-"
            "Myeloid-Differentiation-Master-"
            "Familial-AML-CEBPA-Biallelic-10-15pct-"
            "N-Terminal-LOF-C-Terminal-GOF-Biallelic-"
            "CR-90pct-Relapse-Clone-Same-Germline-OMIM-601626"
        ),
        "locus": "19q13.11",
        "protein_size": (
            "358 aa / ~42 kDa / 19q13.11 CEBPA AML molecular context: "
            "STRUCTURE: "
            "  358 aa / ~42 kDa; bZIP transcription factor; "
            "  N-terminal transactivation domain (aa 1-120) — p42 isoform; "
            "  Regulatory region (aa 121-250) — CEBPA uORF translational control; "
            "  C-terminal bZIP domain (aa 250-358) — DNA binding + leucine zipper dimerisation; "
            "  p42 (full-length) and p30 (N-truncated from downstream start codon) isoforms; "
            "  p42 promotes myeloid differentiation; p30 inhibits p42 and promotes proliferation; "
            "  CEBPA activates neutrophil differentiation genes (G-CSFR, MPO, lactoferrin); "
            "FAMILIAL AML CEBPA AND MOLECULAR MECHANISM: "
            "  Germline CEBPA mutation: N-terminal frameshift/nonsense → LOF of p42 (anti-proliferative lost); "
            "  Biallelic CEBPA AML: germline N-terminal LOF + somatic C-terminal GOF (in-frame bZIP mutation); "
            "  Biallelic CEBPA = ~10-15% of de novo AML (ELN favourable category — best prognosis); "
            "  Germline CEBPA families: ALL or SOME AML cases biallelic (somatic second hit occurs independently); "
            "  The germline mutation is the N-terminal LOF; somatic second hit is C-terminal GOF — different event; "
            "  Monoallelic germline CEBPA: does NOT have same high AML risk as biallelic (AML requires biallelic); "
            "BIALLELIC CEBPA AML — PROGNOSIS EXCELLENT BUT RELAPSE DISTINCT: "
            "  Complete remission rate 90%+ biallelic CEBPA AML (ELN favourable); "
            "  Relapse in germline CEBPA families: relapse clone carries NEW C-terminal GOF + SAME germline N-LOF; "
            "  CRITICAL: relapse ≠ same clone recurrence → re-induction high response rate (new clone salvageable); "
            "  Standard 7+3 induction (cytarabine + daunorubicin/idarubicin) — biallelic CEBPA responds well; "
            "  Allogeneic SCT NOT always required first CR biallelic CEBPA (consolidation with HiDAC acceptable); "
            "  ELN 2022: biallelic CEBPA = favourable risk — HSCT not mandatory CR1 biallelic CEBPA; "
            "GERMLINE TESTING AND SURVEILLANCE: "
            "  ALL AML patients with CEBPA biallelic must have germline testing (blood/fibroblast — not marrow); "
            "  AML CR: germline confirmed → annual CBC + marrow biopsy surveillance in remission; "
            "  Family cascade testing: first-degree relatives for germline N-terminal CEBPA LOF; "
            "  Genetic counselling: offspring 50% risk N-terminal LOF → monitoring from birth; "
            "  Germline CEBPA in MDS: also predisposes (MDS → AML pathway common in older carriers)"
        ),
        "syndrome": "Familial AML CEBPA (OMIM 601626) / biallelic CEBPA AML ELN-favourable / N-terminal LOF germline + C-terminal GOF somatic biallelic / CR 90%+ biallelic / relapse = new clone (same germline) re-salvageable",
        "inheritance": "AD biallelic GOF/LOF",
        "all_risk": "CEBPA biallelic AML: 10-15% de novo AML ELN-favourable; germline N-terminal LOF → somatic C-terminal GOF = biallelic; CR 90%+ biallelic; RELAPSE 30-50% lifetime (new clone — same germline + NEW somatic second hit — re-salvageable high rate); HiDAC consolidation acceptable CR1 (HSCT not mandatory ELN-favourable); monoallelic germline: lower AML risk (requires biallelic for full transformation); germline testing MANDATORY all biallelic CEBPA AML; MDS also predisposed germline CEBPA carriers",
        "pathognomonic": "Biallelic CEBPA AML + family history AML = PATHOGNOMONIC germline CEBPA LOF; N-terminal CEBPA frameshift in blood/fibroblast (not marrow) = germline PATHOGNOMONIC; AML relapse with NEW C-terminal GOF + SAME germline N-LOF = PATHOGNOMONIC germline CEBPA (relapse new clone, not recurrence); young AML (<40yr) biallelic CEBPA ELN-favourable = germline testing MANDATORY; CEBPA biallelic in ≥2 family members = hereditary PATHOGNOMONIC; p30 isoform excess on Western blot = CEBPA disruption marker",
        "key_avoid": "Do NOT classify biallelic CEBPA AML relapse as treatment failure without germline testing — relapse clone is NEW (different C-terminal GOF + same germline N-LOF) = re-salvageable; do NOT mandate allogeneic SCT in CR1 ELN-favourable biallelic CEBPA without evidence of high-risk features (ELN 2022: HiDAC consolidation acceptable); do NOT use marrow DNA for germline testing in active AML (contaminated with blasts — use blood/fibroblast); do NOT skip cascade testing first-degree relatives germline CEBPA (50% inheritance risk + surveillance opportunity pre-AML); do NOT test CEBPA only on somatic AML panel (misses germline N-terminal mutations if panel non-germline); do NOT delay remission marrow surveillance in germline CEBPA carriers",
        "key_rule": "CEBPA AML key rules: (1) biallelic CEBPA = ELN-favourable — CR 90%+; HSCT not mandatory CR1 ELN-favourable biallelic; (2) germline = N-terminal LOF (blood/fibroblast) + somatic C-terminal GOF (marrow blasts); (3) RELAPSE = NEW clone (new C-terminal GOF + SAME germline N-LOF) — high re-salvage rate; (4) germline testing MANDATORY ALL biallelic CEBPA AML (blood or fibroblast — NOT marrow); (5) annual CBC + marrow surveillance CR carriers; (6) cascade testing first-degree relatives MANDATORY; (7) MDS also predisposed in germline carriers; (8) HiDAC consolidation acceptable ELN-favourable biallelic CEBPA CR1.",
        "surveillance": "Germline CEBPA carriers: annual CBC + differential from diagnosis; bone marrow biopsy if CBC concerning; remission surveillance: annual marrow biopsy in CR to detect new AML clone (germline risk persists); first-degree relatives: cascade testing + annual CBC from birth/diagnosis; genetic counselling MANDATORY; marrow biopsy at suspicious cytopenias; AML relapse: complete re-staging (bone marrow aspirate + biopsy + cytogenetics + molecular); somatic CEBPA C-terminal GOF in relapse clone confirms new event",
        "targeted_rx": "Standard 7+3 induction (cytarabine 100-200mg/m²/d × 7d + daunorubicin/idarubicin); HiDAC consolidation (cytarabine 3g/m² × 6 doses) ELN-favourable CR1 biallelic CEBPA; allogeneic SCT: reserved for relapsed/refractory or high-risk features biallelic CEBPA; gemtuzumab ozogamicin (CD33+ AML — check CD33 expression); FLT3 inhibitor (midostaurin/gilteritinib) if FLT3-ITD co-mutation; venetoclax + azacitidine (older unfit patients — VIALE-A); cascade testing MANDATORY germline CEBPA",
    },
    {
        "gene": "RUNX1",
        "protein": (
            "RUNX1 -- 21q22.12 AD LOF -- 453aa -- "
            "RUNX1-51kDa-Runt-Domain-TF-"
            "Haematopoiesis-Master-"
            "FPD-AML-Familial-Platelet-Disorder-30-50pct-"
            "Thrombocytopenia-PATHOGNOMONIC-FPD-"
            "ELN-2022-Germline-Mandatory-OMIM-601399"
        ),
        "locus": "21q22.12",
        "protein_size": (
            "453 aa / ~51 kDa / 21q22.12 RUNX1 AML molecular context: "
            "STRUCTURE: "
            "  453 aa / ~51 kDa; Runt domain (aa 50-177) — DNA binding + CBFβ heterodimerisation; "
            "  Transcriptional activation domain (aa 291-371); "
            "  Inhibitory domain (aa 371-411); "
            "  VWRPY motif (aa 449-453) — Groucho/TLE corepressor interaction; "
            "  RUNX1:CBFβ heterodimer = core binding factor (CBF) complex; "
            "  RUNX1 activates haematopoietic genes: PU.1, CEBPA, MPO, GM-CSFR; "
            "FAMILIAL PLATELET DISORDER WITH MYELOID MALIGNANCY (FPD): "
            "  Germline RUNX1 LOF → FPD: thrombocytopenia (platelet 50-150 × 10^9/L) + platelet function defect; "
            "  Platelet aggregation defect: reduced response to ADP, collagen, arachidonic acid; "
            "  MK differentiation impaired: megakaryocyte (MK) maturation arrest → platelet release defect; "
            "  Dense granule deficiency: storage pool disorder in FPD; "
            "  THROMBOCYTOPENIA IS PATHOGNOMONIC FPD — lifelong, moderate, present from childhood; "
            "AML AND MDS RISK FPD: "
            "  AML/MDS lifetime risk 30-50% (estimated) in germline RUNX1 LOF carriers; "
            "  Median AML onset: 30-40yr (much younger than sporadic AML ~68yr); "
            "  Second somatic hit common: RUNX1 second allele LOF, WT1, TET2, DNMT3A, del7q; "
            "  T-cell ALL also elevated in RUNX1 germline LOF families (RUNX1 haematopoietic master); "
            "  RUNX1::RUNX1T1 t(8;21) somatic fusion = DISTINCT from germline RUNX1 LOF; "
            "  t(8;21) somatic = AML M2, good prognosis, CBF-AML — NOT hereditary RUNX1; "
            "ELN 2022 GERMLINE RECOMMENDATION: "
            "  ELN 2022 AML guidelines: test for germline RUNX1 LOF in AML ≤ 40yr + thrombocytopenia; "
            "  FPD family history + AML = MANDATORY germline RUNX1 sequencing; "
            "  Germline detected → sibling donor for HSCT must be tested (sibling may carry same mutation); "
            "TREATMENT: "
            "  AML: standard 7+3 induction; allogeneic SCT generally recommended given 30-50% AML risk; "
            "  Avoid RUNX1-mutant sibling donor for HSCT (germline → haematopoietic engraftment → same predisposition); "
            "  Use unrelated matched donor or haploidentical donor for HSCT in RUNX1 germline AML; "
            "  MDS: HMA (azacitidine/decitabine) → HSCT; watch for AML evolution; "
            "  Platelet transfusion for procedures (FPD platelet dysfunction); AVOID NSAIDs/aspirin FPD"
        ),
        "syndrome": "FPD/AML RUNX1 (OMIM 601399) / Familial Platelet Disorder with myeloid malignancy predisposition / thrombocytopenia PATHOGNOMONIC / AML/MDS lifetime 30-50% / RUNX1 sibling donor CI for HSCT",
        "inheritance": "AD LOF",
        "all_risk": "RUNX1 germline LOF: AML/MDS 30-50% lifetime risk; thrombocytopenia PATHOGNOMONIC FPD (platelet 50-150 × 10^9/L, platelet function defect); median AML onset 30-40yr (vs 68yr sporadic); second somatic hit (RUNX1 second allele LOF, WT1, TET2, del7q) common; T-ALL also elevated; RUNX1::RUNX1T1 somatic t(8;21) DISTINCT from germline RUNX1 LOF (CBF-AML somatic good prognosis); AVOID RUNX1-mutant sibling donor HSCT (germline → engraftment risk); cascade testing MANDATORY",
        "pathognomonic": "Thrombocytopenia (platelet 50-150 × 10^9/L) + platelet function defect (reduced ADP/collagen/AA aggregation) from childhood + AML = PATHOGNOMONIC germline RUNX1 LOF FPD; FPD + AML in ≥2 family members = PATHOGNOMONIC hereditary RUNX1; dense granule deficiency on electron microscopy + thrombocytopenia + AML = FPD PATHOGNOMONIC; RUNX1 frameshift/nonsense in blood = germline; Runt domain RUNX1 missense = functional null allele; RUNX1::RUNX1T1 somatic t(8;21) DISTINCT (somatic CBF-AML — NOT germline FPD)",
        "key_avoid": "AVOID RUNX1-mutant sibling donor for HSCT in germline RUNX1 AML (sibling may carry same germline LOF → haematopoietic engraftment from donor with same predisposition — use unrelated donor or haploidentical); do NOT diagnose FPD thrombocytopenia as ITP without RUNX1 germline testing in AML families (ITP treatment — steroids/IVIG — misses AML surveillance opportunity); do NOT classify RUNX1::RUNX1T1 t(8;21) somatic as hereditary RUNX1 (COMPLETELY DIFFERENT entity — somatic CBF-AML good prognosis); do NOT skip RUNX1 germline testing in AML ≤40yr + thrombocytopenia (ELN 2022 mandatory); do NOT use NSAIDs/aspirin in FPD thrombocytopenia (platelet function worsened)",
        "key_rule": "RUNX1 AML key rules: (1) thrombocytopenia (platelet 50-150 × 10^9/L) + platelet function defect = FPD PATHOGNOMONIC → germline RUNX1 testing; (2) AML/MDS 30-50% lifetime — HSCT generally recommended; (3) AVOID RUNX1-mutant sibling donor HSCT (use unrelated/haplo); (4) RUNX1::RUNX1T1 somatic t(8;21) = DISTINCT CBF-AML — NOT hereditary RUNX1; (5) germline testing MANDATORY AML ≤40yr + thrombocytopenia (ELN 2022); (6) cascade testing first-degree relatives; (7) AVOID NSAIDs/aspirin FPD; (8) MDS evolution: HMA → HSCT surveillance MANDATORY.",
        "surveillance": "Annual CBC + peripheral blood smear from childhood FPD; platelet function testing (aggregometry) at diagnosis; bone marrow biopsy if CBC concerning (cytopenias, blasts); annual marrow aspirate in germline RUNX1 carriers from age 30yr; cascade testing MANDATORY first-degree relatives (FPD thrombocytopenia may be sole phenotype); sibling donor testing before HSCT (exclude same germline mutation); flow cytometry for MDS markers (CD34+CD117+) at surveillance marrow; clonal evolution monitoring (somatic RUNX1, WT1, TET2); genetic counselling MANDATORY",
        "targeted_rx": "Standard 7+3 induction; allogeneic SCT recommended (high lifetime AML/MDS risk — RUNX1 germline); unrelated matched donor or haploidentical preferred (AVOID RUNX1-carrier sibling donor); HMA (azacitidine/decitabine) for MDS → HSCT transition; gemtuzumab ozogamicin (CD33+ AML); FLT3 inhibitor (midostaurin/gilteritinib) if FLT3 co-mutation; venetoclax + azacitidine (unfit AML/MDS); AVOID NSAIDs/aspirin FPD; platelet transfusion for invasive procedures FPD; cascade testing MANDATORY",
    },
    {
        "gene": "GATA2",
        "protein": (
            "GATA2 -- 3q21.3 AD LOF -- 480aa -- "
            "GATA2-50kDa-Zinc-Finger-TF-"
            "Haematopoietic-Lymphatic-Progenitor-"
            "GATA2-Deficiency-MonoMAC-MDS-AML-30-70x-"
            "Lymphedema-PATHOGNOMONIC-"
            "Infection-Opportunistic-NTM-HPV-OMIM-614286"
        ),
        "locus": "3q21.3",
        "protein_size": (
            "480 aa / ~50 kDa / 3q21.3 GATA2 AML molecular context: "
            "STRUCTURE: "
            "  480 aa / ~50 kDa; GATA-family zinc finger transcription factor; "
            "  N-finger (aa 217-247) — cofactor interaction; "
            "  C-finger (aa 317-347) — DNA binding (GATA motif: (T/A)GATA(A/G)); "
            "  N-terminal domain — transactivation; "
            "  GATA2 critical for HSC maintenance, lymphatic endothelium, megakaryocyte, NK, dendritic cell; "
            "GATA2 DEFICIENCY SYNDROME — PLEIOTROPY: "
            "  GATA2 deficiency = myelodysplasia + lymphoedema + deafness + susceptibility (MonoMAC syndrome); "
            "  MonoMAC: monocytopenia + NK deficiency + B-lymphocytopenia → opportunistic infections; "
            "  Classic infections: NTM (Mycobacterium avium complex), HPV-related warts/dysplasia, CMV, EBV; "
            "  Lymphedema: primary lymphoedema congenital or early onset (lower limb) PATHOGNOMONIC; "
            "  Deafness (sensorineural hearing loss) in subset; "
            "MDS/AML RISK GATA2: "
            "  MDS: 70-90% penetrance by age 60yr in GATA2 deficiency; "
            "  AML: 30-70x elevated risk; median MDS/AML onset age 20-30yr (can present in adolescence); "
            "  Haematological features: cytopenia (mono/NK/B-lymphocytopenia), hypocellular marrow, del(7q), monosomy 7; "
            "  Del7q/monosomy 7 cytogenetics HIGHLY CHARACTERISTIC GATA2 MDS; "
            "  RUNX1 somatic mutations frequent second hit; "
            "LYMPHEDEMA — PATHOGNOMONIC: "
            "  Primary lymphedema (congenital/early) in setting of recurrent opportunistic infection + MDS = PATHOGNOMONIC GATA2; "
            "  Lymphedema may precede haematological disease by years; "
            "  Lymphatic vessel hypoplasia (PROX1 target downstream GATA2); "
            "TREATMENT AND HSCT: "
            "  Allogeneic HSCT = curative for haematological manifestations (MDS/AML/immune deficiency); "
            "  HSCT early (before AML evolution) strongly preferred — pre-emptive HSCT in MDS stage; "
            "  Antimicrobial prophylaxis: azithromycin (NTM), acyclovir (HSV), antifungal (Aspergillus); "
            "  HPV vaccine (Gardasil-9) MANDATORY — HPV-related dysplasia/cancer prevention; "
            "  Avoid live vaccines (immunodeficiency); IgG replacement if hypogammaglobulinaemia"
        ),
        "syndrome": "GATA2 deficiency (OMIM 614286) / MonoMAC syndrome / lymphoedema PATHOGNOMONIC / MDS/AML 30-70x / del7q/monosomy7 CHARACTERISTIC / opportunistic infections NTM-HPV-CMV / HSCT curative",
        "inheritance": "AD LOF",
        "all_risk": "GATA2 germline LOF: MDS 70-90% penetrance by 60yr; AML 30-70x elevated; median onset 20-30yr; del7q/monosomy 7 characteristic cytogenetics; lymphoedema PATHOGNOMONIC (congenital/early primary); MonoMAC syndrome (monocytopenia + NK deficiency + B-lymphopenia → NTM/HPV/CMV opportunistic); HSCT curative for haematological disease + immune deficiency; pre-emptive HSCT in MDS stage preferred; HPV vaccine MANDATORY; NTM prophylaxis azithromycin; RUNX1 somatic second hit common",
        "pathognomonic": "Primary lymphoedema (lower limb) + monocytopenia + NK deficiency + B-lymphopenia + recurrent NTM/HPV/CMV + MDS/AML = GATA2 deficiency PATHOGNOMONIC (MonoMAC); del7q/monosomy7 on karyotype + GATA2 germline LOF = PATHOGNOMONIC haematological GATA2 deficiency; HPV-related extensive warts/condyloma/CIN in young patient + MDS + lymphoedema = GATA2 PATHOGNOMONIC; monocyte count < 0.05 × 10^9/L + NK < 0.01 × 10^9/L + cytopenias from early adulthood = GATA2 deficiency PATHOGNOMONIC",
        "key_avoid": "Do NOT delay HSCT until AML evolution in GATA2 deficiency — MDS → AML transition rapid + transplant outcome worse in AML stage (pre-emptive MDS-stage HSCT preferred); do NOT give live vaccines to GATA2 deficiency patients (severe immunodeficiency — fatal live attenuated infection risk); do NOT miss HPV vaccination in GATA2 deficiency (Gardasil-9 MANDATORY — HPV-related cancer risk dramatically elevated); do NOT miss NTM surveillance/prophylaxis (Mycobacterium avium complex — azithromycin prophylaxis MANDATORY); do NOT use peripheral blood for germline GATA2 testing if cytopenias present (use fibroblast/buccal); do NOT overlook lymphoedema as an early GATA2 sign in young MDS patients",
        "key_rule": "GATA2 AML key rules: (1) MDS 70-90% penetrance — pre-emptive HSCT in MDS stage preferred (before AML); (2) lymphoedema PATHOGNOMONIC + MonoMAC triad = GATA2 MANDATORY testing; (3) del7q/monosomy 7 characteristic GATA2 MDS karyotype; (4) HSCT curative (haematological + immune manifestations); (5) HPV vaccine MANDATORY (Gardasil-9 — HPV cancer risk); (6) NTM prophylaxis azithromycin; (7) AVOID live vaccines; (8) cascade testing MANDATORY first-degree relatives.",
        "surveillance": "CBC + differential monthly/quarterly (monocyte + NK count critical); bone marrow biopsy + cytogenetics annually from diagnosis; HPV screening: annual anogenital exam + Pap/ThinPrep from age 21yr; NTM screening: chest CT + sputum culture annually; CMV/EBV/HHV6 monitoring; hearing test annually (sensorineural deafness); lymphoedema management; ophthalmology (retinal lesions GATA2); genetic counselling MANDATORY; cascade testing first-degree relatives; GATA2 germline panel in all MDS ≤ 40yr + cytopenias + opportunistic infections",
        "targeted_rx": "Allogeneic HSCT (curative — MDS/AML + immune deficiency; preferred pre-AML stage); myeloablative conditioning (preferred younger patients); RIC (reduced-intensity conditioning) older/frail patients; azithromycin NTM prophylaxis; acyclovir HSV prophylaxis; antifungal prophylaxis (azole); Gardasil-9 MANDATORY (HPV-related dysplasia/cancer prevention); IgG supplementation (hypogammaglobulinaemia); HMA (azacitidine/decitabine) MDS bridge to HSCT; venetoclax (MDS/AML bridge HSCT); cascade testing MANDATORY GATA2 families",
    },
    {
        "gene": "DDX41",
        "protein": (
            "DDX41 -- 5q35.3 AD LOF -- 622aa -- "
            "DDX41-70kDa-DEAD-Box-Helicase-"
            "RNA-Splicing-Innate-Immune-Sensor-"
            "Familial-MDS-AML-Late-Onset-55-65yr-HALLMARK-"
            "R525H-Most-Common-Germline-"
            "Biallelic-Somatic-Second-Hit-10-20pct-OMIM-608170"
        ),
        "locus": "5q35.3",
        "protein_size": (
            "622 aa / ~70 kDa / 5q35.3 DDX41 AML molecular context: "
            "STRUCTURE: "
            "  622 aa / ~70 kDa; DEAD-box RNA helicase superfamily; "
            "  N-terminal unstructured domain (aa 1-89) — NLS, protein interactions; "
            "  RecA1 domain (aa 89-305) — ATP binding (Walker A/B motifs); "
            "  RecA2 domain (aa 305-460) — helicase activity (DEAD motif: D342, E343, A344, D345); "
            "  C-terminal SPRY domain (aa 460-622) — protein-protein interaction; "
            "  DDX41 functions: pre-mRNA splicing, rRNA processing, innate immune cGAS/STING dsDNA sensing; "
            "FAMILIAL MDS/AML DDX41 — KEY FEATURES: "
            "  Germline DDX41 LOF: most common hereditary myeloid malignancy gene in adults (2026); "
            "  LATE ONSET HALLMARK: median onset 55-65yr (strikingly different from CEBPA/RUNX1/GATA2 younger onset); "
            "  Predominantly MDS (53%) > AML (35%) > CMML/other (12%); "
            "  Germline R525H (c.1574G>A) most common European founder mutation; "
            "  Germline D140fs most common in Asian populations; "
            "BIALLELIC SOMATIC SECOND HIT: "
            "  10-20% AML cases with germline DDX41 also acquire somatic DDX41 second hit (biallelic); "
            "  Biallelic DDX41: R525H somatic + germline frameshift OR germline + somatic R525H; "
            "  Biallelic DDX41 AML: slightly higher blast count + more aggressive; "
            "  R525H somatic second hit also found as sole somatic event in 5% sporadic AML (not germline); "
            "IMMUNOLOGICAL FUNCTION: "
            "  DDX41 senses cytosolic dsDNA → activates cGAS/STING → type I interferon production; "
            "  DDX41 LOF → impaired innate immune dsDNA sensing → possibly MDS permissive immune environment; "
            "  Also: pre-spliceosomal role → U2AF1/SF3B1-independent splicing factor context (SF-AML); "
            "TREATMENT: "
            "  AML/MDS treatment: standard 7+3 or HMA (azacitidine); "
            "  Allogeneic HSCT recommended for AML (high-risk transformation); "
            "  HMA + venetoclax: emerging PREFERRED in older unfit DDX41 AML/MDS patients; "
            "  DDX41 AML: IDH1/2 co-mutations reported (enasidenib/ivosidenib potentially active); "
            "  Germline DDX41 detected → sibling donor testing before HSCT"
        ),
        "syndrome": "Familial MDS/AML DDX41 (OMIM 608170) / LATE ONSET 55-65yr HALLMARK / R525H most common germline / biallelic somatic second hit 10-20% / MDS (53%) > AML (35%) / innate immune cGAS-STING dsDNA sensor",
        "inheritance": "AD LOF",
        "all_risk": "DDX41 germline LOF: MDS 53% / AML 35% / CMML 12%; LATE ONSET HALLMARK (55-65yr); R525H c.1574G>A most common European germline; biallelic somatic second-hit R525H in 10-20% AML (more aggressive); IDH1/2 co-mutations reported; HMA + venetoclax preferred older unfit; HSCT recommended AML DDX41; sibling testing before donor; innate immune cGAS/STING impairment mechanism; cascade testing MANDATORY",
        "pathognomonic": "MDS/AML onset age 55-65yr + family history + R525H germline = DDX41 PATHOGNOMONIC; DDX41 R525H or D140fs in blood/fibroblast = germline PATHOGNOMONIC; biallelic DDX41 AML (germline + somatic R525H) = PATHOGNOMONIC hereditary DDX41; late-onset MDS in ≥2 family members (55-65yr onset) = DDX41 germline testing MANDATORY; DDX41 somatic R525H on myeloid panel + age >55yr + MDS = germline R525H testing MANDATORY (somatic may mask germline)",
        "key_avoid": "Do NOT dismiss DDX41 somatic R525H in older MDS patient as purely somatic without germline testing (somatic R525H frequently represents second hit on germline allele — missing germline diagnosis = missed cascade testing); do NOT apply CEBPA/RUNX1/GATA2 early-onset surveillance framework to DDX41 (LATE ONSET — surveillance starts 40-50yr, not childhood); do NOT skip sibling donor testing before HSCT (DDX41 germline — sibling risk); do NOT overlook IDH1/2 co-mutation in DDX41 AML (enasidenib/ivosidenib potentially active); do NOT over-treat low-risk MDS DDX41 (HMA + venetoclax or HMA alone preferred older patients before HSCT consideration)",
        "key_rule": "DDX41 AML key rules: (1) LATE ONSET 55-65yr HALLMARK — most common adult hereditary myeloid malignancy gene; (2) R525H germline most common European (test germline if somatic R525H found); (3) biallelic DDX41 somatic second-hit in 10-20% AML — more aggressive; (4) HMA + venetoclax preferred older unfit DDX41 MDS/AML; (5) HSCT recommended AML DDX41; (6) sibling donor testing MANDATORY before HSCT; (7) IDH1/2 co-mutations reported (enasidenib/ivosidenib); (8) cascade testing first-degree relatives MANDATORY.",
        "surveillance": "CBC + differential annually from age 45yr for germline DDX41 carriers; bone marrow biopsy if cytopenias; bone marrow aspirate + cytogenetics at MDS diagnosis; somatic R525H on myeloid panel in older MDS patient → germline testing MANDATORY; sibling testing before HSCT; IDH1/2 mutation testing at AML diagnosis (targeted therapy eligibility); cascade testing first-degree relatives; genetic counselling (late-onset family members — 50% inheritance risk); marrow surveillance annually from 50yr in confirmed carriers",
        "targeted_rx": "HMA (azacitidine/decitabine) — MDS DDX41; HMA + venetoclax (PREFERRED older/unfit AML/MDS DDX41 — VIALE-A equivalent); standard 7+3 induction (fit younger AML); enasidenib (IDH2 inhibitor) if IDH2 co-mutation; ivosidenib (IDH1 inhibitor) if IDH1 co-mutation; allogeneic HSCT (AML DDX41 — unrelated/haplo preferred; sibling tested first); luspatercept (anaemia MDS); venetoclax mono or combo; cascade testing MANDATORY DDX41 families",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "Transcription-Factor-G1-Arrest-"
            "LFS-AML-10-20x-"
            "AVOID-RADIATION-ABSOLUTELY-LFS-"
            "WBMRI-Toronto-Annually-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 AML molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; N-terminal transactivation domain (aa 1-40); "
            "  Proline-rich region (aa 40-92); DNA-binding domain (aa 100-290) — MOST mutations; "
            "  Tetramerisation domain (aa 320-356) — p53 homotetramers; "
            "  C-terminal regulatory domain (aa 363-393); "
            "  p53 activates: CDKN1A (p21 G1 arrest), BAX (apoptosis), PUMA, GADD45; "
            "LFS AND AML RISK: "
            "  LFS (OMIM 151623): germline TP53 LOF → tumour spectrum: sarcomas (MOST common), breast, brain, AML/ALL, adrenocortical; "
            "  AML: 10-20x elevated in LFS; acute leukaemia component of core LFS tumour spectrum; "
            "  Monosomy 17p / del17p / TP53 somatic: 5-10% de novo AML; complex karyotype AML; "
            "  Complex karyotype AML (≥3 chromosomal abnormalities) = TP53 somatic most common; "
            "  Complex karyotype + TP53 = worst AML prognosis (ELN adverse); "
            "RADIOSENSITIVITY AND TREATMENT IMPLICATIONS: "
            "  AVOID RADIATION ABSOLUTELY in LFS — secondary malignancies catastrophic; "
            "  Whole-body MRI (WBMRI) Toronto protocol: annual surveillance (replaces CT/PET for LFS); "
            "  Alkylating agents — CI in LFS (DNA-damaging agents → secondary malignancy explosion in LFS); "
            "  AML treatment in LFS: AVOID radiotherapy; minimise alkylating agents; "
            "  Decitabine/azacitidine HMA: preferred TP53-mutant AML (TP53 demethylation mechanism); "
            "  APR-246 (eprenetapopt) + azacitidine: TP53 refolding mutant TP53 — clinical trials; "
            "  Venetoclax: LESS active in TP53-mutant AML (TP53 mutations bypass BCL2 dependence); "
            "  Etoposide: CI in LFS (topoisomerase II — secondary malignancy risk); "
            "PROGNOSIS: "
            "  Complex karyotype + TP53 AML: median OS 5-8 months with standard therapy; "
            "  Biallelic TP53 AML (both alleles hit): extremely poor prognosis (median OS 3-5 months); "
            "  Decitabine/azacitidine achieves transient responses but durable CR rare; "
            "  Eprenetapopt (APR-246) trials ongoing for TP53-mutant AML/MDS"
        ),
        "syndrome": "LFS TP53 (OMIM 151623) / AML 10-20x / complex karyotype AML worst prognosis / AVOID RADIATION ABSOLUTELY / alkylating CI LFS / WBMRI Toronto annually / decitabine/azacitidine preferred TP53-mutant AML",
        "inheritance": "AD LOF",
        "all_risk": "TP53 germline LOF: AML 10-20x LFS; complex karyotype + TP53 somatic = ELN adverse worst prognosis; AVOID RADIATION ABSOLUTELY (secondary malignancies catastrophic LFS); alkylating CI LFS; WBMRI Toronto annually (replaces CT/PET — radiation avoidance); decitabine/azacitidine preferred TP53-mutant AML; APR-246 eprenetapopt + azacitidine trials; venetoclax LESS active TP53-mutant; biallelic TP53 AML extremely poor prognosis; cascade testing MANDATORY",
        "pathognomonic": "LFS spectrum: osteosarcoma + breast cancer <45yr + brain tumour + AML + adrenocortical carcinoma in family = LFS PATHOGNOMONIC; TP53 germline in blood = LFS PATHOGNOMONIC; AML + complex karyotype + TP53 bi-allelic = PATHOGNOMONIC worst prognosis AML; adrenocortical carcinoma in child = PATHOGNOMONIC LFS (TP53 mandatory testing); bilateral breast cancer <35yr + AML + sarcoma in family = LFS PATHOGNOMONIC",
        "key_avoid": "AVOID RADIATION ABSOLUTELY in LFS TP53 (secondary malignancy catastrophic — sarcoma, contralateral breast cancer, secondary AML); AVOID alkylating agents in LFS (cyclophosphamide, melphalan, busulfan — secondary malignancy risk); do NOT use CT/PET for LFS surveillance (radiation — use WBMRI Toronto protocol); do NOT miss biallelic TP53 AML (both alleles = extremely poor prognosis — consider clinical trial); do NOT over-rely on venetoclax TP53-mutant AML (LESS active — TP53 mutations bypass BCL2 dependence); do NOT skip cascade testing in LFS families",
        "key_rule": "TP53 AML key rules: (1) AVOID RADIATION ABSOLUTELY LFS (secondary malignancies catastrophic); (2) WBMRI Toronto annually (replaces CT/PET for LFS surveillance); (3) complex karyotype AML + TP53 = ELN adverse worst prognosis; (4) decitabine/azacitidine preferred TP53-mutant AML; (5) APR-246 eprenetapopt + azacitidine clinical trials; (6) venetoclax LESS active TP53-mutant AML; (7) alkylating CI LFS; (8) biallelic TP53 AML = extremely poor prognosis — clinical trial preferred.",
        "surveillance": "WBMRI Toronto annually (whole-body MRI — no radiation LFS); breast MRI annually from age 20-25yr; annual physical exam + CBC; abdominal/pelvic ultrasound 6-monthly (LFS sarcoma/adrenocortical); brain MRI annually; colonoscopy from age 25yr; adrenocortical carcinoma screening ultrasound childhood; CBC + differential annually; cascade testing MANDATORY first-degree relatives; genetic counselling — reproductive planning (preimplantation genetic testing); bone marrow biopsy if cytopenias + blast concern",
        "targeted_rx": "Decitabine (azacitidine alternative) — PREFERRED TP53-mutant AML; APR-246 eprenetapopt + azacitidine (Phase 3 AGILE trial — TP53 refolding mutant p53); magrolimab + azacitidine (CD47 antibody + HMA — TP53-mutant AML trials); venetoclax (LESS active TP53-mutant — use cautiously); AVOID radiotherapy (LFS ABSOLUTELY); AVOID alkylating agents (LFS); allogeneic HSCT (high-risk setting — non-TBI conditioning; AVOID TBI LFS); cascade testing MANDATORY TP53/LFS families",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 AD LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-"
            "RAD51-Filament-ssDNA-Repair-"
            "FA-D1-Biallelic-AML-ALL-HIGHEST-"
            "AVOID-ALKYLATING-ABSOLUTELY-Biallelic-"
            "PARP-Olaparib-HRD-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / ~384 kDa / 13q12.3 BRCA2 AML molecular context: "
            "STRUCTURE: "
            "  3418 aa / ~384 kDa; N-terminal transactivation domain (aa 10-40) — PALB2 interaction; "
            "  BRC repeats ×8 (aa 1002-2082) — RAD51 monomer binding; "
            "  DNA-binding domain (DBD; aa 2402-3190) — ssDNA/RPA interaction; "
            "  C-terminal BRCA2 (aa 3190-3418) — RAD51C/XRCC3 interaction; "
            "  BRCA2 loads RAD51 onto RPA-coated ssDNA at DSBs → homologous recombination; "
            "FA-D1 BIALLELIC AND AML/ALL: "
            "  Biallelic BRCA2 LOF = Fanconi Anemia complementation group D1 (FA-D1; OMIM 605724); "
            "  FA-D1: AML/ALL HIGHEST risk among all FA subtypes (AML + ALL both very high); "
            "  FA-D1 AML typically myeloid (AML-M4/M7 most common); FA-D1 ALL (B-ALL and T-ALL); "
            "  AVOID ALKYLATING AGENTS ABSOLUTELY biallelic FA-D1 (DNA crosslinks catastrophic — aplasia); "
            "  AVOID radiation biallelic FA-D1 (RADIOSENSITIVITY ABSOLUTE FA); "
            "  Bone marrow failure (BMF) cardinal FA-D1 feature (progressive pancytopenia); "
            "  Solid tumours (medulloblastoma, Wilms, bilateral Wilms PATHOGNOMONIC FA-D1); "
            "MONOALLELIC BRCA2 AND AML: "
            "  Monoallelic BRCA2 germline LOF: AML 2-3x elevated; breast/ovarian HBOC primary risk; "
            "  AML in BRCA2 monoallelic: especially therapy-related AML (alkylating + RT prior exposure); "
            "  PARP inhibitors (olaparib, rucaparib, niraparib) active in BRCA2 germline LOF tumours; "
            "  HRD (homologous recombination deficiency) = olaparib FDA2020 sensitive; "
            "  Platinum-based chemotherapy highly active in BRCA2 LOF (cisplatin/carboplatin); "
            "HRD AND TARGETED THERAPY: "
            "  Olaparib (PARP1/2 inhibitor) FDA2020 approved BRCA2-mutant AML/solid tumours; "
            "  Mechanism: PARP trapping on SSBs → replication fork collapse → DSBs → lethal in BRCA2-LOF; "
            "  Rucaparib, niraparib, talazoparib: PARP inhibitor alternatives; "
            "  HRD score (Myriad myChoice) ≥42 = HRD positive = PARP sensitive; "
            "  Cisplatin: preferred platinum in BRCA2-LOF (DNA crosslink + HRD = synthetic lethality)"
        ),
        "syndrome": "HBOC-2 monoallelic / FA-D1 biallelic (OMIM 605724) / AML/ALL HIGHEST biallelic FA-D1 / AVOID ALKYLATING ABSOLUTELY biallelic / PARP inhibitor olaparib HRD / bilateral Wilms PATHOGNOMONIC FA-D1 / bone marrow failure CARDINAL FA",
        "inheritance": "AD LOF / AR biallelic FA-D1",
        "all_risk": "BRCA2 biallelic FA-D1: AML/ALL HIGHEST among all FA groups; AVOID ALKYLATING AGENTS ABSOLUTELY biallelic (DNA crosslinks catastrophic aplasia); RADIOSENSITIVITY ABSOLUTE biallelic; bone marrow failure cardinal FA-D1; bilateral Wilms tumour PATHOGNOMONIC FA-D1; solid tumours (medulloblastoma); monoallelic BRCA2: AML 2-3x (especially therapy-related); PARP inhibitor olaparib FDA2020 HRD active; cisplatin/platinum preferred; HRD score ≥42 = PARP sensitive; cascade testing MANDATORY",
        "pathognomonic": "Bilateral Wilms tumour in child = PATHOGNOMONIC FA-D1 biallelic BRCA2; bone marrow failure (pancytopenia) + medulloblastoma + bilateral Wilms + AML = FA-D1 PATHOGNOMONIC; chromosomal fragility test (diepoxybutane DEB or mitomycin C MMC breakage assay) positive = Fanconi Anemia (confirms biallelic FA); BRCA2 in blood/fibroblast (biallelic) = FA-D1 confirmed; HRD score ≥42 + BRCA2 germline = olaparib-sensitive PATHOGNOMONIC; olaparib response in BRCA2-LOF tumour confirms HRD",
        "key_avoid": "AVOID ALKYLATING AGENTS ABSOLUTELY biallelic BRCA2 FA-D1 (cyclophosphamide, melphalan, busulfan, temozolomide — DNA crosslinks catastrophic aplasia in FA); AVOID radiation biallelic BRCA2 FA-D1 (RADIOSENSITIVITY ABSOLUTE — fatal RT toxicity); do NOT use standard AML induction in FA-D1 biallelic BRCA2 without dose reduction (cytarabine OK dose-reduced; anthracyclines reduced; NO alkylating agents); do NOT miss bilateral Wilms tumour as PATHOGNOMONIC FA-D1 in child with pancytopenia; do NOT skip cascade testing siblings of FA-D1 BRCA2 (25% biallelic risk in consanguineous families)",
        "key_rule": "BRCA2 AML key rules: (1) BIALLELIC FA-D1: AML/ALL HIGHEST — AVOID ALKYLATING ABSOLUTELY; (2) RADIOSENSITIVITY ABSOLUTE biallelic FA-D1 — no RT, reduced-intensity chemotherapy only; (3) bilateral Wilms PATHOGNOMONIC FA-D1; (4) DEB/MMC chromosomal fragility test = FA diagnosis; (5) MONOALLELIC: AML 2-3x (especially therapy-related); (6) olaparib PARP inhibitor FDA2020 active BRCA2 LOF (HRD score ≥42); (7) cisplatin/platinum preferred monoallelic BRCA2 AML; (8) cascade testing MANDATORY both biallelic and monoallelic.",
        "surveillance": "Biallelic FA-D1: CBC + marrow biopsy every 3-6 months (bone marrow failure surveillance); abdominal ultrasound every 3-6 months (Wilms); ophthalmology; renal ultrasound (bilateral Wilms); DEB/MMC test at diagnosis FA confirmation; allogeneic HSCT consideration early (FA-D1 BMF → AML prevention); monoallelic BRCA2: annual mammography + breast MRI from 25yr; annual clinical breast exam; HRD testing at AML diagnosis; cascade testing first-degree relatives MANDATORY; olaparib eligibility testing (HRD assay); platinum-sensitivity monitoring",
        "targeted_rx": "Biallelic FA-D1 AML: reduced-intensity chemotherapy (cytarabine dose-reduced; NO alkylating); HSCT early (FA-D1 — non-TBI, non-alkylating conditioning: fludarabine-based); AVOID alkylating agents; AVOID RT; monoallelic BRCA2: olaparib (PARP inhibitor FDA2020 — HRD AML/solid tumours); rucaparib, niraparib, talazoparib alternatives; cisplatin/carboplatin (platinum — HRD sensitive); DEB/MMC chromosomal fragility test at FA diagnosis; cascade testing MANDATORY BRCA2 families",
    },
    {
        "gene": "FANCA",
        "protein": (
            "FANCA -- 16q24.3 AR LOF -- 1455aa -- "
            "FANCA-163kDa-FA-Core-Complex-"
            "FANCD2-FANCI-Monoubiquitination-"
            "Most-Common-FA-65pct-"
            "AML-MDS-50x-Biallelic-HIGHEST-"
            "AVOID-ALKYLATING-ABSOLUTELY-Biallelic-OMIM-227650"
        ),
        "locus": "16q24.3",
        "protein_size": (
            "1455 aa / ~163 kDa / 16q24.3 FANCA AML molecular context: "
            "STRUCTURE: "
            "  1455 aa / ~163 kDa; alpha-helical solenoid scaffold; "
            "  N-terminal nuclear localisation signal (aa 1-30); "
            "  NLS clusters (aa 17-31, 1369-1382) — nuclear import; "
            "  FANCB/FANCC/FANCL/FANCM binding regions — core complex assembly; "
            "  FANCA forms FA core E3 ligase complex with FANCB/FANCC/FANCE/FANCF/FANCG/FANCL/FANCM; "
            "  Core complex monoubiquitinates FANCD2/FANCI → ID2 complex → interstrand crosslink (ICL) repair; "
            "FANCONI ANEMIA TYPE A — MOST COMMON (65% OF ALL FA): "
            "  Biallelic FANCA LOF = Fanconi Anemia type A (FA-A); most common FA subtype (65%); "
            "  Diagnosis criteria: DEB/MMC chromosomal fragility test + biallelic FA gene sequencing; "
            "  Cardinal features: bone marrow failure (progressive pancytopenia), congenital anomalies, AML; "
            "  Congenital anomalies: short stature, café-au-lait macules, thumb/radius anomalies, renal; "
            "  CAFÉ-AU-LAIT MACULES CARDINAL FA-A: ≥3 cafe-au-lait macules, irregular borders; "
            "AML/MDS RISK FANCA BIALLELIC: "
            "  AML/MDS: 50x elevated biallelic FANCA (FA-A); predominantly myeloid (AML-M4/M7 commonest); "
            "  Median AML onset in FA: 13-16yr (adolescent/young adult — much younger than sporadic AML); "
            "  Bone marrow failure typically precedes AML by years: aplasia → MDS → AML progression; "
            "  Clonal evolution: RUNX1 somatic, TP53 somatic, complex karyotype in FA AML; "
            "ALKYLATING AGENTS — ABSOLUTELY CONTRAINDICATED BIALLELIC FANCA: "
            "  AVOID ALKYLATING AGENTS ABSOLUTELY biallelic FA-A (cyclophosphamide, melphalan, busulfan); "
            "  Mechanism: alkylating agents → DNA interstrand crosslinks → FA pathway overwhelmed → fatal aplasia; "
            "  AVOID radiation biallelic FA-A (RADIOSENSITIVITY ABSOLUTE biallelic FA); "
            "  Reduced-intensity conditioning (RIC) for HSCT: fludarabine-based — NO alkylating, NO TBI; "
            "  Low-dose cyclophosphamide ONLY in carefully dosed HSCT conditioning if unavoidable (historical); "
            "HSCT AND TREATMENT: "
            "  Allogeneic HSCT curative for haematological manifestations (BMF/MDS/AML); "
            "  HSCT from HLA-matched sibling: check sibling for FA (DEB test MANDATORY — 25% carrier risk); "
            "  Unrelated/haplo donor if sibling affected; "
            "  Oxymetholone (androgens): temporise BMF (increase platelet/neutrophil temporarily — not curative); "
            "  G-CSF (filgrastim): neutropenia management bridge to HSCT"
        ),
        "syndrome": "Fanconi Anemia Type A (OMIM 227650) / most common FA 65% / AML/MDS 50x biallelic HIGHEST / AVOID ALKYLATING ABSOLUTELY biallelic / RADIOSENSITIVITY ABSOLUTE biallelic / bone marrow failure CARDINAL / café-au-lait macules + thumb anomalies PATHOGNOMONIC / HSCT curative",
        "inheritance": "AR LOF",
        "all_risk": "FANCA biallelic FA-A: AML/MDS 50x HIGHEST; median AML onset 13-16yr; bone marrow failure cardinal (aplasia → MDS → AML); AVOID ALKYLATING AGENTS ABSOLUTELY biallelic (DNA crosslinks fatal aplasia); RADIOSENSITIVITY ABSOLUTE biallelic; HSCT curative (fludarabine-based non-TBI, non-alkylating RIC); sibling donor DEB test MANDATORY (25% biallelic risk); oxymetholone (androgens) temporise BMF; café-au-lait macules + thumb/radius anomalies PATHOGNOMONIC FA; cascade testing MANDATORY",
        "pathognomonic": "Bone marrow failure (pancytopenia) + café-au-lait macules (≥3, irregular) + thumb/radius anomalies (radial ray defect, absent thumb, bifid thumb) + AML = FA-A PATHOGNOMONIC; DEB/MMC chromosomal fragility test PATHOGNOMONIC FA (excessive breakage, triradials, quadriradials); biallelic FANCA LOF = FA-A confirmed; short stature + café-au-lait + aplastic anemia in child = FA PATHOGNOMONIC triad; microcephaly + renal anomaly + thumb defect + cytopenias = FA-A PATHOGNOMONIC constellation",
        "key_avoid": "AVOID ALKYLATING AGENTS ABSOLUTELY biallelic FA-A (cyclophosphamide, melphalan, busulfan, temozolomide — DNA ICL catastrophic aplasia + fatal toxicity); AVOID radiation biallelic FA-A (RADIOSENSITIVITY ABSOLUTE — use fludarabine-based non-TBI conditioning); do NOT use HLA-matched sibling as HSCT donor without DEB/MMC testing the sibling (25% biallelic FA risk — affected sibling donor = transfer same disease); do NOT diagnose as acquired aplastic anemia without FA workup in child with cytopenias + congenital anomalies (DEB test FIRST); do NOT delay HSCT until frank AML (HSCT in BMF/MDS stage — better outcomes than AML stage)",
        "key_rule": "FANCA AML key rules: (1) AVOID ALKYLATING AGENTS ABSOLUTELY biallelic — DNA crosslinks fatal; (2) RADIOSENSITIVITY ABSOLUTE biallelic — no TBI; fludarabine-based RIC conditioning for HSCT; (3) DEB/MMC test MANDATORY FA diagnosis; (4) sibling donor DEB testing MANDATORY (25% biallelic risk); (5) HSCT curative for BMF/MDS/AML (fludarabine-based non-alkylating non-TBI); (6) café-au-lait macules + thumb/radius defects + pancytopenia = FA PATHOGNOMONIC; (7) median AML onset 13-16yr (adolescent); (8) cascade testing MANDATORY FA-A families.",
        "surveillance": "FA-A: CBC + reticulocyte count monthly; bone marrow biopsy + cytogenetics every 6 months (BMF surveillance); head/neck exam + head/neck MRI (SCC surveillance from age 10yr); gynaecological exam (vulval/cervical SCC from age 13yr); DEB/MMC test at diagnosis; sibling DEB testing; HSCT consideration at aplasia/MDS onset; annual audiology (hearing); renal ultrasound (renal anomalies); growth monitoring; genetic counselling MANDATORY; cascade testing first-degree relatives; androgen therapy monitoring (virilisation); G-CSF monitoring (neutropenia management pre-HSCT)",
        "targeted_rx": "Allogeneic HSCT (curative FA-A — fludarabine-based RIC; NO alkylating; NO TBI); oxymetholone androgens (temporise BMF — not curative); G-CSF (filgrastim — neutropenia bridge to HSCT); eltrombopag (thrombopoietin receptor agonist — thrombocytopenia FA bridge); AVOID alkylating agents (cyclophosphamide, melphalan, busulfan); AVOID TBI; reduced-dose chemotherapy (cytarabine dose-reduced) if AML treatment required pre-HSCT; DEB/MMC test at diagnosis; sibling DEB testing MANDATORY; cascade testing MANDATORY FANCA families",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 AD LOF -- 2839aa -- "
            "Neurofibromin-327kDa-Ras-GAP-"
            "Ras-GTP-Hydrolysis-Accelerator-"
            "JMML-200-500x-HIGHEST-Paediatric-"
            "AML-5-10x-Ras-MAPK-"
            "Cafe-au-Lait-PATHOGNOMONIC-Selumetinib-FDA2020-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / ~327 kDa / 17q11.2 NF1 AML molecular context: "
            "STRUCTURE: "
            "  2839 aa / ~327 kDa; Ras-GAP (GTPase-activating protein); "
            "  N-terminal coiled-coil (aa 1-560) — membrane association; "
            "  SEC14 domain (aa 1560-1726) — lipid binding; "
            "  GRD/Ras-GAP related domain (aa 1200-1530) — catalytic Ras-GTP hydrolysis (CRITICAL); "
            "  CSRD (aa 1580-1600) — cysteine-serine rich domain; "
            "  Syndecan-binding domain (aa 2597-2796) — cytoskeletal interaction; "
            "  Neurofibromin catalytic domain accelerates intrinsic Ras GTPase activity ×1000; "
            "  NF1 LOF → Ras-GTP constitutive → RAF/MEK/ERK constitutive → cell proliferation/survival; "
            "JMML — HIGHEST PAEDIATRIC MYELOID MALIGNANCY RISK NF1: "
            "  Juvenile Myelomonocytic Leukaemia (JMML): NF1 germline LOF + somatic NF1 second hit; "
            "  JMML NF1: 200-500x HIGHEST risk — NF1 germline is the HIGHEST risk gene for JMML; "
            "  JMML: monocytosis >1 × 10^9/L + peripheral blasts < 20% + splenomegaly + BCR-ABL1 negative; "
            "  NF1-JMML: median onset 2-4yr (EARLY childhood); spontaneous remission RARE in NF1-JMML; "
            "  JMML WHO 2022: NF1/PTPN11/KRAS/NRAS/CBL mutations = 5 canonical JMML driver genes; "
            "  NF1-JMML prognosis: without HSCT median OS < 2yr; HSCT curative (alloSCT — best NF1-JMML Rx); "
            "AML RISK NF1: "
            "  AML (not JMML): 5-10x elevated NF1 germline LOF; predominantly adult onset (30-50yr); "
            "  NF1 AML: Ras activating co-mutations common (NRAS, KRAS, PTPN11 co-mutations); "
            "  Complex karyotype AML with NF1: Ras pathway hyperactivation + genomic instability; "
            "SELUMETINIB (MEK INHIBITOR) AND NF1: "
            "  Selumetinib (MEK inhibitor, FDA 2020): approved NF1 plexiform neurofibromas ≥2yr; "
            "  Selumetinib in JMML: MEK inhibition reduces monocytes, splenomegaly (clinical trials); "
            "  Trametinib (MEK inhibitor): also investigational NF1-JMML + AML RAS-mutant; "
            "  RAS/MEK targeting: potential in NF1-AML (Ras constitutive — MEKi active pathway);"
        ),
        "syndrome": "NF1 (OMIM 162200) / JMML 200-500x HIGHEST paediatric / AML 5-10x adult / Ras/MAPK constitutive / café-au-lait macules ≥6 PATHOGNOMONIC / selumetinib FDA2020 / JMML HSCT curative / cascade testing MANDATORY",
        "inheritance": "AD LOF",
        "all_risk": "NF1 germline LOF: JMML 200-500x HIGHEST paediatric (median onset 2-4yr; NF1 second somatic hit required); AML 5-10x adult (30-50yr); Ras/MAPK constitutive; café-au-lait macules ≥6 PATHOGNOMONIC; selumetinib FDA2020 NF1 plexiform neurofibromas; trametinib MEK inhibitor investigational JMML/AML; HSCT curative JMML (without HSCT < 2yr OS); MPNST 8-13% lifetime; cascade testing MANDATORY",
        "pathognomonic": "≥6 café-au-lait macules (>5mm prepubertal, >15mm postpubertal) = PATHOGNOMONIC NF1; Lisch nodules (iris hamartomas, slit-lamp) = PATHOGNOMONIC NF1 (80% adults); plexiform neurofibroma + JMML in child = NF1 PATHOGNOMONIC; axillary/inguinal freckling (Crowe sign) = PATHOGNOMONIC NF1 (70%); optic pathway glioma (OPG) in child = NF1 PATHOGNOMONIC; JMML + monosomy 7 + NF1 germline = PATHOGNOMONIC NF1-JMML; NF1 germline + somatic NF1 LOF in JMML blasts = PATHOGNOMONIC biallelic NF1",
        "key_avoid": "Do NOT delay HSCT in NF1-JMML (spontaneous remission RARE in NF1-JMML — without HSCT median OS < 2yr); do NOT classify NF1-JMML as reactive monocytosis without JMML diagnostic criteria (NF1 + monocytosis > 1 × 10^9/L + splenomegaly + BCR-ABL1 negative = JMML diagnostic); do NOT miss selumetinib eligibility in NF1 plexiform neurofibromas (FDA2020 ≥2yr — MPNST risk reduction); do NOT skip cascade testing in NF1 families (JMML in child + café-au-lait = MANDATORY NF1 germline + first-degree cascade); do NOT overlook MPNST surveillance (MPNST 8-13% NF1 — MRI nodal/plexiform monitoring)",
        "key_rule": "NF1 AML key rules: (1) JMML 200-500x HIGHEST — NF1 MOST common germline gene JMML; (2) JMML: HSCT curative (without HSCT < 2yr OS); (3) ≥6 café-au-lait macules PATHOGNOMONIC NF1; (4) selumetinib FDA2020 NF1 plexiform neurofibromas; (5) trametinib MEK inhibitor investigational NF1-JMML; (6) AML 5-10x adult NF1 (Ras/MAPK); (7) AVOID delay in JMML HSCT (spontaneous remission RARE NF1-JMML); (8) cascade testing first-degree relatives MANDATORY.",
        "surveillance": "NF1: annual physical exam (neurofibromas, Lisch nodules slit-lamp, BP, growth); annual ophthalmology (optic pathway glioma) in childhood; brain MRI (OPG — NF1 specific): annually ages 0-7yr, every 2-3yr thereafter; CBC + monocyte count from birth (JMML surveillance); bone marrow biopsy if monocytosis > 1 × 10^9/L or splenomegaly (JMML); skin surveillance (MPNST — progressive pain/growth in plexiform = urgent MRI); NF1 cognitive surveillance; cascade testing first-degree relatives MANDATORY; annual CBC adult NF1 carriers (AML surveillance); selumetinib eligibility assessment if plexiform neurofibromas",
        "targeted_rx": "Selumetinib (AstraZeneca — FDA2020 — NF1 plexiform neurofibromas ≥2yr; MEK1/2 inhibitor); trametinib (MEK inhibitor — investigational NF1-JMML/RAS-mutant AML trials); allogeneic HSCT (JMML — curative; myeloablative conditioning preferred younger patients; busulfan-based NF1 OK — not RADIOSENSITIVE); standard 7+3 AML induction + HSCT (NF1-AML adult); cobimetinib (MEK inhibitor — investigational); navitoclax (BCL-2/BCL-XL — JMML investigational); 13-cis-retinoic acid (JMML historically — replaced by HSCT/MEKi); cascade testing MANDATORY NF1 families",
    },
]


def _make_patient(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)

    # Gene-specific AML/myeloid characteristics
    aml_params = {
        "CEBPA": dict(age_mu=35, age_sd=12, aml_prob=0.70, mds_prob=0.20, jmml_prob=0.00,
                      eln_fav_prob=0.88, complex_karyotype_prob=0.05, hrd_prob=0.05,
                      biallelic_hit_prob=0.90, hsct_needed_prob=0.45),
        "RUNX1": dict(age_mu=35, age_sd=10, aml_prob=0.55, mds_prob=0.30, jmml_prob=0.00,
                      eln_fav_prob=0.20, complex_karyotype_prob=0.18, hrd_prob=0.04,
                      biallelic_hit_prob=0.45, hsct_needed_prob=0.75),
        "GATA2": dict(age_mu=28, age_sd=10, aml_prob=0.50, mds_prob=0.40, jmml_prob=0.00,
                      eln_fav_prob=0.15, complex_karyotype_prob=0.55, hrd_prob=0.03,
                      biallelic_hit_prob=0.55, hsct_needed_prob=0.90),
        "DDX41": dict(age_mu=60, age_sd=8,  aml_prob=0.45, mds_prob=0.50, jmml_prob=0.00,
                      eln_fav_prob=0.15, complex_karyotype_prob=0.25, hrd_prob=0.03,
                      biallelic_hit_prob=0.18, hsct_needed_prob=0.65),
        "TP53":  dict(age_mu=45, age_sd=15, aml_prob=0.60, mds_prob=0.30, jmml_prob=0.00,
                      eln_fav_prob=0.02, complex_karyotype_prob=0.78, hrd_prob=0.04,
                      biallelic_hit_prob=0.40, hsct_needed_prob=0.55),
        "BRCA2": dict(age_mu=40, age_sd=15, aml_prob=0.45, mds_prob=0.25, jmml_prob=0.00,
                      eln_fav_prob=0.12, complex_karyotype_prob=0.30, hrd_prob=0.82,
                      biallelic_hit_prob=0.25, hsct_needed_prob=0.60),
        "FANCA": dict(age_mu=14, age_sd=5,  aml_prob=0.60, mds_prob=0.30, jmml_prob=0.00,
                      eln_fav_prob=0.10, complex_karyotype_prob=0.45, hrd_prob=0.78,
                      biallelic_hit_prob=0.95, hsct_needed_prob=0.92),
        "NF1":   dict(age_mu=8,  age_sd=6,  aml_prob=0.45, mds_prob=0.15, jmml_prob=0.55,
                      eln_fav_prob=0.10, complex_karyotype_prob=0.35, hrd_prob=0.04,
                      biallelic_hit_prob=0.70, hsct_needed_prob=0.80),
    }
    p = aml_params[gene]

    # Age with gene-specific ranges
    age_raw = rng.gauss(p["age_mu"], p["age_sd"])
    if gene == "FANCA":
        age = max(2, min(35, round(age_raw, 1)))
    elif gene == "NF1":
        age = max(1, min(30, round(age_raw, 1)))
    elif gene in ("CEBPA", "RUNX1", "GATA2"):
        age = max(18, min(60, round(age_raw, 1)))
    elif gene == "DDX41":
        age = max(40, min(80, round(age_raw, 1)))
    else:
        age = max(18, min(75, round(age_raw, 1)))

    has_aml          = rng.random() < p["aml_prob"]
    has_mds          = rng.random() < p["mds_prob"]
    has_jmml         = rng.random() < p["jmml_prob"]
    eln_favourable   = rng.random() < p["eln_fav_prob"]
    complex_karyotype= rng.random() < p["complex_karyotype_prob"]
    hrd_positive     = rng.random() < p["hrd_prob"]
    biallelic_hit    = rng.random() < p["biallelic_hit_prob"]
    hsct_needed      = rng.random() < p["hsct_needed_prob"]

    # AML FAB subtype distribution
    aml_fab_types = ["AML-M0", "AML-M1", "AML-M2", "AML-M3", "AML-M4", "AML-M5", "AML-M6", "AML-M7"]
    if gene == "CEBPA":
        fab_probs = [0.05, 0.20, 0.30, 0.02, 0.25, 0.10, 0.05, 0.03]
    elif gene in ("FANCA", "BRCA2"):
        fab_probs = [0.05, 0.10, 0.15, 0.02, 0.30, 0.18, 0.05, 0.15]
    elif gene == "NF1":
        fab_probs = [0.05, 0.10, 0.15, 0.02, 0.30, 0.25, 0.05, 0.08]
    elif gene == "GATA2":
        fab_probs = [0.10, 0.15, 0.20, 0.02, 0.25, 0.20, 0.05, 0.03]
    else:
        fab_probs = [0.08, 0.15, 0.22, 0.03, 0.25, 0.15, 0.06, 0.06]

    cumulative = 0.0
    r = rng.random()
    fab = aml_fab_types[-1]
    for ft, fp in zip(aml_fab_types, fab_probs):
        cumulative += fp
        if r < cumulative:
            fab = ft
            break

    return {
        "gene":              gene,
        "seed":              seed,
        "age_at_dx":         age,
        "has_aml":           has_aml,
        "has_mds":           has_mds,
        "has_jmml":          has_jmml,
        "fab_subtype":       fab,
        "eln_favourable":    eln_favourable,
        "complex_karyotype": complex_karyotype,
        "hrd_positive":      hrd_positive,
        "biallelic_hit":     biallelic_hit,
        "hsct_needed":       hsct_needed,
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

    aml_n          = sum(1 for p in all_pts if p["has_aml"])
    mds_n          = sum(1 for p in all_pts if p["has_mds"])
    jmml_n         = sum(1 for p in all_pts if p["has_jmml"])
    eln_fav_n      = sum(1 for p in all_pts if p["eln_favourable"])
    complex_kar_n  = sum(1 for p in all_pts if p["complex_karyotype"])
    hrd_n          = sum(1 for p in all_pts if p["hrd_positive"])
    biallelic_n    = sum(1 for p in all_pts if p["biallelic_hit"])
    hsct_n         = sum(1 for p in all_pts if p["hsct_needed"])
    mean_age       = round(sum(p["age_at_dx"] for p in all_pts) / len(all_pts), 1)

    gene_summary = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        aml_gene_n = sum(1 for p in pts if p["has_aml"])
        mds_gene_n = sum(1 for p in pts if p["has_mds"])
        jmml_gene_n = sum(1 for p in pts if p["has_jmml"])
        gene_summary.append({
            "gene":              gene,
            "locus":             gd["locus"],
            "n":                 len(pts),
            "aml_n":             aml_gene_n,
            "aml_pct":           round(100 * aml_gene_n / len(pts), 1),
            "mds_n":             mds_gene_n,
            "mds_pct":           round(100 * mds_gene_n / len(pts), 1),
            "jmml_n":            jmml_gene_n,
            "jmml_pct":          round(100 * jmml_gene_n / len(pts), 1),
            "mean_age_onset":    round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "complex_karyotype_n": sum(1 for p in pts if p["complex_karyotype"]),
            "hrd_positive_n":    sum(1 for p in pts if p["hrd_positive"]),
            "biallelic_hit_n":   sum(1 for p in pts if p["biallelic_hit"]),
            "hsct_needed_n":     sum(1 for p in pts if p["hsct_needed"]),
            "inheritance":       gd["inheritance"],
            "key_distinctions":  [
                gd["all_risk"][:80],
                gd["key_rule"][:80],
            ],
        })

    genes_detail = []
    for gd in ATLAS_GENES:
        genes_detail.append({
            "gene":             gd["gene"],
            "protein":          gd["protein"],
            "locus":            gd["locus"],
            "protein_size":     gd["protein_size"],
            "inheritance":      gd["inheritance"],
            "cancer_risk":      gd["all_risk"],
            "syndrome":         gd["syndrome"],
            "pathognomonic":    gd["pathognomonic"],
            "surveillance_key": gd["surveillance"][:160],
            "key_rule":         gd["key_rule"],
            "key_avoid":        gd["key_avoid"],
        })

    pathway_categories = [
        {"pathway": "Myeloid Transcription Factor / Differentiation Arrest", "genes": ["CEBPA", "RUNX1", "GATA2"], "mechanism": "Transcription factor LOF → myeloid differentiation arrest → AML (CEBPA biallelic ELN-favourable; RUNX1 FPD 30-50% AML; GATA2 70-90% MDS penetrance)"},
        {"pathway": "RNA Splicing / Innate Immune Sensing", "genes": ["DDX41"], "mechanism": "DDX41 LOF → impaired cGAS/STING innate immune dsDNA sensing + pre-mRNA splicing defect → late-onset MDS/AML (hallmark 55-65yr)"},
        {"pathway": "TP53 / Cell Cycle G1 Checkpoint", "genes": ["TP53"], "mechanism": "p53 LOF → G1 arrest failure → complex karyotype AML; ELN adverse worst prognosis; AVOID RADIATION ABSOLUTELY LFS; alkylating CI"},
        {"pathway": "Homologous Recombination (HR) / Fanconi Anemia Pathway", "genes": ["BRCA2", "FANCA"], "mechanism": "BRCA2/FANCA LOF → HR defect → ICL repair failure → AML/MDS; AVOID ALKYLATING ABSOLUTELY biallelic; PARP inhibitor HRD sensitivity; RADIOSENSITIVITY ABSOLUTE biallelic FA"},
        {"pathway": "Ras/MAPK / GTPase Signalling", "genes": ["NF1"], "mechanism": "Neurofibromin Ras-GAP LOF → constitutive Ras-GTP → MEK/ERK → JMML 200-500x (paediatric) + AML 5-10x (adult); MEK inhibitor selumetinib/trametinib active"},
    ]

    critical_distinctions = [
        "CEBPA biallelic AML RELAPSE = NEW CLONE (not recurrence): germline N-terminal LOF persists; somatic C-terminal GOF = new independent second hit at relapse; CRITICAL clinical implication — relapse is re-inducible at high rate (NOT treatment failure); standard relapse = new clone = re-induction + HSCT consideration; do NOT interpret relapse as chemotherapy resistance without germline CEBPA testing (relapse new clone can achieve CR2 or CR3); all biallelic CEBPA AML must have germline testing (blood/fibroblast — not marrow)",
        "DDX41 LATE ONSET HALLMARK vs other hereditary AML genes: DDX41 germline MDS/AML onset 55-65yr (UNIQUE — all other main hereditary AML genes have onset 15-40yr); late-onset MDS in family + R525H somatic on panel = germline test MANDATORY; surveillance framework DDX41 starts age 45-50yr (not childhood); DDX41 most common adult hereditary myeloid malignancy gene (2026); HMA + venetoclax preferred older unfit DDX41 AML/MDS",
        "GATA2 deficiency MonoMAC vs acquired aplastic anemia: monocytopenia < 0.05 × 10^9/L + NK deficiency < 0.01 × 10^9/L + B-lymphopenia in MDS patient = PATHOGNOMONIC GATA2 (MonoMAC) — NOT acquired aplastic anemia; GATA2 MDS: hypocellular marrow + del7q/monosomy7 characteristic (NOT aplastic anemia pattern); HSCT is curative for GATA2 deficiency haematological + immunological manifestations; pre-emptive HSCT at MDS stage preferred (before AML evolution); HPV vaccination MANDATORY GATA2",
        "FA pathway (FANCA/BRCA2): AVOID ALKYLATING AGENTS ABSOLUTELY biallelic — CRITICAL alkylating-agent contraindication: cyclophosphamide, melphalan, busulfan, temozolomide = DNA ICL catastrophic aplasia in biallelic Fanconi Anemia; RADIOSENSITIVITY ABSOLUTE biallelic; reduced-intensity fludarabine-based conditioning (non-alkylating non-TBI) MANDATORY for HSCT biallelic FA; DEB/MMC chromosomal fragility test = diagnostic gold standard FA; SIBLING DONOR TESTING MANDATORY (25% biallelic risk)",
        "NF1-JMML vs reactive monocytosis: NF1 germline LOF + monocytosis > 1 × 10^9/L + splenomegaly + BCR-ABL1 negative + peripheral blasts < 20% = JMML (WHO 2022 criteria); reactive monocytosis (infection, CMV, inflammatory) does NOT meet JMML criteria; spontaneous remission RARE in NF1-JMML (distinct from KRAS/NRAS JMML where spontaneous remission more common); HSCT curative NF1-JMML (without HSCT median OS < 2yr); selumetinib investigational NF1-JMML (MEK inhibition); trametinib emerging NF1-JMML/AML trials",
        "RUNX1 sibling donor contraindication for HSCT: germline RUNX1 LOF transmitted with 50% probability → sibling has 50% chance of same germline mutation; if sibling donor also carries germline RUNX1 LOF → HSCT donates haematopoietic stem cells with same predisposition → future leukaemia risk from donor-origin cells; MANDATORY sibling DEB test (FANCA/BRCA2) or RUNX1 germline test (RUNX1) before HSC donation; prefer unrelated matched or haploidentical donor for germline RUNX1/FANCA/BRCA2 AML HSCT",
        "HRD (Homologous Recombination Deficiency) and PARP inhibitors in AML: BRCA2 monoallelic germline LOF → HRD → olaparib FDA2020 active (PARP trapping synthetic lethality); FANCA biallelic → HRD → PARP sensitivity; HRD score (Myriad myChoice ≥42) = PARP inhibitor eligible; cisplatin/carboplatin preferred platinum (HRD sensitive); venetoclax active in BRCA2-LOF AML regardless HRD; PARP inhibitor + HMA emerging combination HRD-AML (clinical trials ongoing 2026)",
        "TP53 complex karyotype AML — worst prognosis management: complex karyotype (≥3 chromosomal abnormalities) + TP53 = ELN adverse; biallelic TP53 AML = extremely poor prognosis (median OS 3-5 months); decitabine/azacitidine HMA preferred TP53-mutant AML (demethylation mechanism); APR-246 eprenetapopt (p53 refolding mutant) + azacitidine Phase 3 AGILE trial; venetoclax LESS active TP53-mutant AML; magrolimab (CD47) + azacitidine trials TP53-mutant AML; AVOID radiation + alkylating agents LFS TP53 germline",
        "Hereditary AML germline panel: AML + thrombocytopenia ≤40yr → RUNX1 FIRST; biallelic CEBPA AML any age → germline testing MANDATORY; late-onset MDS/AML 55-65yr + family → DDX41 R525H; young MDS + lymphoedema + opportunistic infections → GATA2; pancytopenia + thumb anomalies + café-au-lait in child → FANCA DEB test; JMML in NF1 child → NF1 cascade; AML in LFS family → TP53; AML + HBOC/FA-D1 family → BRCA2; cascade testing MANDATORY all hereditary AML genes; germline testing from blood/fibroblast (NOT marrow in active AML)",
    ]

    return {
        "atlas":                "Hereditary-AML-Predisposition-Atlas",
        "subtitle":             "8-Gene AML/MDS/JMML Predisposition Reference — 320-Patient Aggregate",
        "genes":                [gd["gene"] for gd in ATLAS_GENES],
        "seed_range":           f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":       len(all_pts),
        "genes_n":              8,
        "aml_total_n":          aml_n,
        "aml_total_pct":        round(100 * aml_n / len(all_pts), 1),
        "mds_total_n":          mds_n,
        "mds_total_pct":        round(100 * mds_n / len(all_pts), 1),
        "jmml_total_n":         jmml_n,
        "jmml_total_pct":       round(100 * jmml_n / len(all_pts), 1),
        "mean_age_onset":       mean_age,
        "eln_fav_n":            eln_fav_n,
        "eln_fav_pct":          round(100 * eln_fav_n / len(all_pts), 1),
        "complex_karyotype_n":  complex_kar_n,
        "complex_karyotype_pct":round(100 * complex_kar_n / len(all_pts), 1),
        "hrd_positive_n":       hrd_n,
        "hrd_positive_pct":     round(100 * hrd_n / len(all_pts), 1),
        "biallelic_hit_n":      biallelic_n,
        "biallelic_hit_pct":    round(100 * biallelic_n / len(all_pts), 1),
        "hsct_needed_n":        hsct_n,
        "hsct_needed_pct":      round(100 * hsct_n / len(all_pts), 1),
        "gene_summary":         gene_summary,
        "genes_detail":         genes_detail,
        "pathway_categories":   pathway_categories,
        "critical_distinctions": critical_distinctions,
        "gene_n":               {gd["gene"]: 40 for gd in ATLAS_GENES},
    }


def generate_breakdown() -> dict:
    cohorts = _build_cohorts()
    _gene_pathog = {gd["gene"]: gd["pathognomonic"] for gd in ATLAS_GENES}
    _gene_surv   = {gd["gene"]: gd["surveillance"][:120] for gd in ATLAS_GENES}
    _gene_rule   = {gd["gene"]: gd["key_rule"] for gd in ATLAS_GENES}

    breakdown_array = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        aml_n  = sum(1 for p in pts if p["has_aml"])
        mds_n  = sum(1 for p in pts if p["has_mds"])
        jmml_n = sum(1 for p in pts if p["has_jmml"])
        breakdown_array.append({
            "gene":              gene,
            "locus":             gd["locus"],
            "n":                 len(pts),
            "mean_age_onset":    round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "aml_pct":           round(100 * aml_n / len(pts), 1),
            "mds_pct":           round(100 * mds_n / len(pts), 1),
            "jmml_pct":          round(100 * jmml_n / len(pts), 1),
            "complex_karyotype_n": sum(1 for p in pts if p["complex_karyotype"]),
            "hrd_positive_n":    sum(1 for p in pts if p["hrd_positive"]),
            "biallelic_hit_n":   sum(1 for p in pts if p["biallelic_hit"]),
            "hsct_needed_n":     sum(1 for p in pts if p["hsct_needed"]),
            "seed":              SEED_BASE + i,
            "pathognomonic":     _gene_pathog[gene],
            "surveillance_key":  _gene_surv[gene],
            "key_rule":          _gene_rule[gene],
        })
    return {
        "breakdown":      breakdown_array,
        "total_patients": sum(len(v) for v in cohorts.values()),
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
            "all_risk":      gd["all_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_avoid":     gd["key_avoid"],
            "key_rule":      gd["key_rule"],
            "surveillance":  gd["surveillance"],
            "targeted_rx":   gd["targeted_rx"],
        }

    key_clinical_distinctions = [
        "CEBPA biallelic AML relapse = NEW clone: germline N-terminal LOF persists; somatic C-terminal GOF = new event at relapse; re-inducible at high rate (NOT treatment failure); germline testing MANDATORY all biallelic CEBPA AML from blood/fibroblast (NOT marrow); ELN-favourable biallelic CEBPA — HSCT not mandatory in CR1; HiDAC consolidation acceptable ELN-favourable biallelic CEBPA",
        "DDX41 late-onset hallmark (55-65yr) vs all other hereditary AML genes (15-40yr): most common adult hereditary myeloid malignancy gene; R525H germline most common European; biallelic somatic second-hit in 10-20% AML; HMA + venetoclax preferred older unfit; surveillance starts age 45-50yr; sibling testing before HSCT",
        "FA pathway AVOID ALKYLATING ABSOLUTELY biallelic (FANCA/BRCA2): alkylating agents → DNA ICL → fatal aplasia biallelic FA; fludarabine-based non-TBI non-alkylating conditioning MANDATORY HSCT; RADIOSENSITIVITY ABSOLUTE biallelic FA; DEB/MMC chromosomal fragility test diagnostic; sibling DEB/BRCA2 germline testing MANDATORY before donor (25% biallelic risk)",
        "NF1-JMML vs reactive monocytosis: NF1 germline + monocytosis > 1 × 10^9/L + splenomegaly + BCR-ABL1 negative + blasts < 20% = JMML; spontaneous remission RARE NF1-JMML; HSCT curative NF1-JMML (median OS < 2yr without HSCT); selumetinib/trametinib MEK inhibition investigational; café-au-lait macules ≥6 PATHOGNOMONIC NF1",
        "GATA2 deficiency MonoMAC: monocytopenia + NK deficiency + B-lymphopenia + lymphoedema + opportunistic NTM/HPV/CMV = PATHOGNOMONIC; del7q/monosomy 7 characteristic GATA2 MDS; pre-emptive HSCT at MDS stage preferred (before AML evolution — HSCT curative); HPV vaccination MANDATORY (Gardasil-9); AVOID live vaccines GATA2 (immunodeficiency)",
        "RUNX1 FPD thrombocytopenia PATHOGNOMONIC: platelet 50-150 × 10^9/L + platelet function defect from childhood + AML family = FPD PATHOGNOMONIC germline RUNX1; AVOID RUNX1-mutant sibling donor HSCT; RUNX1::RUNX1T1 somatic t(8;21) CBF-AML DISTINCT from germline RUNX1 LOF FPD (different entities); ELN 2022 MANDATORY germline testing AML ≤40yr + thrombocytopenia",
        "HRD and PARP inhibitors: BRCA2 germline LOF + HRD score ≥42 = olaparib FDA2020 active (PARP trapping synthetic lethality); FANCA biallelic HRD-positive → PARP sensitivity; cisplatin/carboplatin preferred platinum HRD-AML; venetoclax active regardless BRCA2/FANCA HRD status; PARP inhibitor + HMA emerging combination HRD-AML clinical trials",
        "TP53 complex karyotype AML worst prognosis: ELN adverse biallelic TP53; complex karyotype ≥3 abnormalities + TP53 = worst OS; decitabine/azacitidine preferred TP53-mutant; APR-246 eprenetapopt + azacitidine Phase 3 AGILE; venetoclax LESS active TP53-mutant; AVOID radiation ABSOLUTELY LFS TP53 germline; WBMRI Toronto annually LFS",
        "Germline testing framework hereditary AML: blood/fibroblast ONLY (NOT marrow in active AML — blast contamination); R525H on somatic panel → germline test (may be second hit); biallelic CEBPA → germline MANDATORY; thrombocytopenia ≤40yr + AML → RUNX1; lymphoedema + opportunistic + MDS → GATA2; thumb/radius defects + pancytopenia → DEB test FA; JMML in NF1 child → NF1; LFS history + AML → TP53; cascade testing MANDATORY all hereditary AML genes",
    ]

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"AML/MDS RISK: {gd['all_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-AML-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": key_clinical_distinctions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"Total patients: {ov['total_patients']}, AML n: {ov['aml_total_n']} ({ov['aml_total_pct']}%)")
    print(f"MDS: {ov['mds_total_n']} ({ov['mds_total_pct']}%), JMML: {ov['jmml_total_n']} ({ov['jmml_total_pct']}%)")
    print(f"ELN-fav: {ov['eln_fav_n']} ({ov['eln_fav_pct']}%), complex karyotype: {ov['complex_karyotype_n']} ({ov['complex_karyotype_pct']}%)")
    print(f"HRD-positive: {ov['hrd_positive_n']} ({ov['hrd_positive_pct']}%), HSCT needed: {ov['hsct_needed_n']} ({ov['hsct_needed_pct']}%)")
    print("=== BREAKDOWN ===")
    br = generate_breakdown()
    for b in br["breakdown"]:
        print(f"  {b['gene']}: AML {b['aml_pct']}%, MDS {b['mds_pct']}%, JMML {b['jmml_pct']}%, age {b['mean_age_onset']}yr, HRD {b['hrd_positive_n']}, HSCT {b['hsct_needed_n']}")
    print("=== DEFINITIONS ===")
    df = generate_definitions()
    print(f"Genes: {list(df['gene_defs'].keys())}")
    print(f"Key distinctions: {len(df['key_clinical_distinctions'])}")
