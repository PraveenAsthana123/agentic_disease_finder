"""Hereditary NHEJ & SSBR Atlas — 8-Gene DSB End-Joining + Single-Strand Break Repair Reference
LIG4-DCLRE1C-PRKDC-XRCC4-NHEJ1-PNKP-APTX-TDP1
(NHEJ/SSBR deficiency spectrum: RS-SCID / LIG4-Syndrome / Artemis-SCID /
 DNA-PKcs-SCID / Primordial-Dwarfism / XLF-SCID / MCSZ / AOA1 / SCAN1)
320 patients (8 x 40), seeds 2750-2757.
Endpoints: /api/hereditary-nhej-ssbr-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "LIG4",
        "seed_base": 2750,
        "protein": (
            "LIG4 -- 13q33.3 AR/AD -- 911aa -- DNA-Ligase-IV-"
            "102kDa-NHEJ-Ligation-Step-XRCC4-Binding-BRCT-Tandem-"
            "OMIM-Gene-601837-Disease-LIG4-Syndrome-606593"
        ),
        "locus": "13q33.3",
        "protein_size": "911 aa / 102 kDa (ATP-dependent DNA ligase; BRCT-BRCT tandem binds XRCC4; seals NHEJ nick; last step of classical NHEJ)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic hypomorphic LOF → LIG4 Syndrome); "
            "BIALLELIC DISEASE (LIG4 Syndrome): "
            "  Combined immunodeficiency (T-B-NK+ or T-B+): variable depth; "
            "  Microcephaly: progressive, disproportionate; "
            "  Pancytopenia: myelosuppression prominent; bone marrow failure; "
            "  Growth retardation / short stature: prenatal onset; "
            "  Developmental delay: variable; "
            "  Radiosensitivity: ABSOLUTE — even standard X-ray doses cause lethal bone marrow aplasia; "
            "  Lymphoma risk: B-cell lymphoma in patients who survive childhood; "
            "  Skin: malar rash, telangiectasias, photosensitivity; "
            "COMPLETE NULL LIG4 = LETHAL IN UTERO (murine data): only hypomorphic alleles survive; "
            "FOUNDER/COMMON ALLELES: "
            "  c.2440C>T (p.Arg814Cys): most common hypomorphic — pan-ethnic; "
            "  c.833G>A (p.Arg278His): moderate phenotype; "
            "  c.1390delA + c.2440C>T compound heterozygous: classic severe LIG4 syndrome; "
            "  Turkish consanguineous: homozygous c.2440C>T common; "
            "MONOALLELIC RISK: "
            "  Heterozygous LIG4: unproven hereditary cancer risk (not established guideline syndrome); "
            "  Research: monoallelic in lymphoma families — evolving evidence"
        ),
        "disease_category": (
            "LIG4 SYNDROME (OMIM 606593): "
            "IMMUNOLOGICAL PHENOTYPE: "
            "  T cells: severely reduced (oligoclonal T cells due to defective V(D)J recombination); "
            "  B cells: absent or profoundly reduced; "
            "  NK cells: NORMAL (NK cells do not require RAG-dependent NHEJ); "
            "  Immunoglobulins: panhypogammaglobulinaemia — all isotypes reduced; "
            "  V(D)J MECHANISM: RAG-generated DSBs require LIG4/XRCC4 for ligation; LIG4 LOF → "
            "    incomplete V(D)J → absent functional TCR/BCR → T-B-NK+ SCID; "
            "HAEMATOLOGICAL PHENOTYPE: "
            "  Pancytopenia: aplastic anaemia; thrombocytopenia; neutropenia; "
            "  Bone marrow: hypocellular; "
            "  Lymphoma: B-cell NHL (diffuse large B-cell, Burkitt-like) in unmonitored survivors; "
            "NEUROLOGICAL PHENOTYPE: "
            "  Microcephaly: congenital or progressive; OFC -3 to -7 SD; "
            "  Intellectual disability: mild-severe; "
            "  Seizures: minority; "
            "GROWTH: "
            "  Prenatal growth restriction: IUGR; "
            "  Postnatal short stature: -3 to -5 SD height; "
            "RADIOSENSITIVITY — CRITICAL CLINICAL FLAG: "
            "  Conventional radiotherapy doses (2 Gy fractions) → FATAL aplasia; "
            "  Diagnostic chest X-rays: acceptable only with extreme caution; "
            "  CT scans: avoid if LIG4 Syndrome confirmed; prefer MRI; "
            "  Pre-HSCT conditioning: AVOID myeloablative TBI; use reduced-intensity conditioning (fludarabine-based); "
            "SKIN: "
            "  Malar rash: erythematous; can mimic lupus; "
            "  Telangiectasias: periorbital, conjunctival; "
            "  Photosensitivity: mild"
        ),
        "disease_pathway": (
            "LIG4 — NHEJ LIGATION STEP: "
            "CLASSICAL NHEJ PATHWAY (cNHEJ): "
            "STEP 1 — DSB DETECTION: "
            "  KU70/KU80 (XRCC6/XRCC5) heterodimer binds DSB ends → rings around dsDNA; "
            "  KU recruits DNA-PKcs (PRKDC) → DNA-PK holoenzyme; "
            "  DNA-PKcs autophosphorylation (T2609/S2056) → conformational change → exposes DNA ends; "
            "STEP 2 — END PROCESSING: "
            "  ARTEMIS (DCLRE1C): endonuclease/exonuclease; hairpin opening; overhang trimming; "
            "  Activated by DNA-PKcs phosphorylation; "
            "  PNKP: 5'-kinase + 3'-phosphatase — restores 5'-P and 3'-OH for ligation; "
            "  POLM/POLL: polymerase fill-in for gaps; "
            "STEP 3 — SYNAPSIS: "
            "  XRCC4 forms homodimer; recruits LIG4; XLF (NHEJ1) stimulates XRCC4-LIG4 complex; "
            "  PAXX (paralogue of XRCC4/XLF) assists Ku-dependent synapsis; "
            "STEP 4 — LIGATION (LIG4): "
            "  LIG4 adenylation from ATP → LIG4-AMP intermediate → "
            "    transfer AMP to 5'-phosphate of nick → "
            "    3'-OH attacks 5'-AMP → phosphodiester bond formed; "
            "  XRCC4 stabilises and stimulates LIG4; "
            "LIG4 LOF → DSBs accumulate → "
            "  V(D)J recombination fails → T-B-NK+ SCID; "
            "  Ionising radiation DSBs not repaired → chromosomal instability; "
            "  DNA-PKcs-Artemis pathway still active but no ligation step → accumulation of unligated intermediates"
        ),
        "pathognomonic": (
            "LIG4 SYNDROME DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION: "
            "  Microcephaly + growth retardation + T-B-NK+ SCID phenotype; "
            "  Severe pancytopenia disproportionate to infection burden; "
            "  FATAL ADVERSE REACTION to standard-dose radiotherapy or alkylating agents; "
            "  B-cell lymphoma in childhood/adolescence WITH immunodeficiency; "
            "RADIOSENSITIVITY TESTING: "
            "  Chromosomal breakage assay: gamma-irradiation of patient lymphoblastoid cells → "
            "    dicentrics + ring chromosomes at low dose (2 Gy) → profoundly elevated vs controls; "
            "  G2/M checkpoint assay: elevated G2 chromosomal aberrations post-irradiation; "
            "  Colony survival assay: markedly reduced clonogenic survival at 2-4 Gy; "
            "IMMUNOLOGICAL WORKUP: "
            "  Flow cytometry: T-B-NK+ pattern; "
            "  T-cell receptor excision circles (TRECs): absent on newborn screen; "
            "  V(D)J repertoire: oligo/monoclonal if any T cells present; "
            "  IgG/IgA/IgE: profoundly low; IgM variable; "
            "MOLECULAR CONFIRMATION: "
            "  LIG4 gene sequencing (full exons + splice sites) + MLPA; "
            "  Protein: LIG4 protein stability western blot (XRCC4 co-immunoprecipitation); "
            "TREATMENT: "
            "  HSCT: curative for haematological/immunological phenotype; "
            "  MANDATORY REDUCED-INTENSITY CONDITIONING: TBI is ABSOLUTELY CONTRAINDICATED; "
            "  Fludarabine-based/serotherapy RIC before HSCT; "
            "  Bridge: IVIG + PJP prophylaxis + antifungal; "
            "  Avoid live vaccines pre-HSCT; "
            "  Post-HSCT: cancer surveillance continues (LIG4 somatic risk in residual host cells)"
        ),
        "treatment": (
            "LIG4 SYNDROME MANAGEMENT: "
            "HSCT (DEFINITIVE): "
            "  Indication: all T-B- SCID presentations; severe pancytopenia; "
            "  Timing: as early as possible (before infectious complications); "
            "  Conditioning: REDUCED INTENSITY MANDATORY — fludarabine + serotherapy (alemtuzumab/ATG); "
            "  TBI ABSOLUTELY CONTRAINDICATED — causes fatal aplasia in LIG4 Syndrome; "
            "  Cyclophosphamide: reduce dose / AVOID in full-dose; "
            "  Busulfan: caution — use pharmacokinetic-guided reduced dose only; "
            "  Graft: MSD, MUD, or MMUD with reduced conditioning; "
            "  Outcome: excellent graft survival if conditioning not genotoxic; "
            "BRIDGE THERAPY PRE-HSCT: "
            "  IVIG: 400-600 mg/kg every 3-4 weeks; "
            "  PJP prophylaxis: trimethoprim-sulfamethoxazole (or pentamidine); "
            "  Antifungal: azole prophylaxis; "
            "  Avoid BCG, live vaccines; "
            "  CMV monitoring: weekly PCR; "
            "RADIOSENSITIVITY MANAGEMENT: "
            "  Medical alert bracelet: 'RADIOSENSITIVE — NO RADIOTHERAPY / AVOID MYELOTOXIC DRUGS'; "
            "  Radiology: prefer MRI; chest X-ray only if critical; avoid CT; "
            "  Dental: avoid dental X-rays unless essential; "
            "LYMPHOMA SURVEILLANCE (post-HSCT survivors): "
            "  Annual clinical examination; "
            "  LDH + CBC with differential; "
            "  Low threshold for lymph node biopsy; "
            "  PET-CT if lymphoma suspected; "
            "GENETIC COUNSELLING: "
            "  AR pattern: 25% recurrence for sibling; "
            "  Carrier parents: no clinical phenotype; "
            "  Prenatal diagnosis: available"
        ),
    },
    {
        "gene": "DCLRE1C",
        "seed_base": 2751,
        "protein": (
            "DCLRE1C -- 10p13 AR -- 692aa -- Artemis-DNA-Cross-Link-Repair-1C-"
            "78kDa-VDJ-Hairpin-Opening-Endonuclease-5prime3prime-Exonuclease-DNA-PKcs-Activated-"
            "OMIM-Gene-605988-Disease-Artemis-SCID-602450"
        ),
        "locus": "10p13",
        "protein_size": "692 aa / 78 kDa (metallo-β-lactamase fold; 5'→3' exonuclease + endonuclease; activated by DNA-PKcs phosphorylation; hairpin opening)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → Artemis-SCID = RS-SCID type B); "
            "BIALLELIC DISEASE (Artemis-SCID): "
            "  T-B-NK+ SCID: complete absence of T and B lymphocytes; NK cells present; "
            "  RADIATION-SENSITIVE SCID: elevated chromosomal aberrations after irradiation; "
            "  Omenn-syndrome phenotype: occasional — partial function alleles → oligoclonal T cells, erythroderma, eosinophilia; "
            "  NO microcephaly (unlike LIG4) — important DDx; "
            "  NO pancytopenia (unlike LIG4) — DDx; "
            "  Recurrent infections: Pneumocystis, CMV, Candida, gram-negative bacteria; "
            "  Age at presentation: neonatal/early infancy; "
            "FOUNDER/COMMON ALLELES: "
            "  Athabascan-speaking Native American (Navajo, Apache): "
            "    c.1090delGCACC + c.3031delGCAGACATCCGACCTC — two-allele founder haploblock; "
            "    ~2% carrier rate in Navajo Nation; "
            "  Deleterious frameshift/nonsense most common in European: "
            "    c.265dupA, c.597_607del, p.Pro70fs; "
            "  Middle Eastern: c.1218_1229del (partial deletion common)"
        ),
        "disease_category": (
            "ARTEMIS-SCID (RS-SCID, OMIM 602450): "
            "IMMUNOLOGICAL PHENOTYPE: "
            "  T cells: ABSENT (V(D)J hairpin not opened → incomplete TCR → no T cells); "
            "  B cells: ABSENT (incomplete BCR V(D)J); "
            "  NK cells: NORMAL (NK not RAG-dependent); "
            "  T-B-NK+ SCID = classic ARTEMIS phenotype; "
            "  Immunoglobulins: all isotypes near-absent; maternal IgG present in first months; "
            "V(D)J MECHANISM: "
            "  RAG1/RAG2 cuts RSS→ coding joints with hairpin ends; "
            "  Hairpin opening REQUIRES Artemis (activated by DNA-PKcs); "
            "  Artemis LOF → hairpins not opened → no coding joints → V(D)J fails → SCID; "
            "OMENN SYNDROME VARIANT: "
            "  Hypomorphic DCLRE1C alleles → partial hairpin opening → oligoclonal T cells; "
            "  Clinical: erythroderma, alopecia, hepatosplenomegaly, lymphadenopathy, eosinophilia; "
            "  Elevated IgE with panhypogammaglobulinaemia otherwise; "
            "RADIOSENSITIVITY: "
            "  Chromosomal breakage: elevated dicentrics post-irradiation; "
            "  Mechanism: DSBs accumulated at complex ends that require Artemis endonucleolytic trimming; "
            "  Severity: milder than LIG4 radiosensitivity; "
            "DDX FROM LIG4 SYNDROME: "
            "  NO microcephaly (DCLRE1C); LIG4 has microcephaly; "
            "  NO pancytopenia (DCLRE1C); LIG4 has pancytopenia; "
            "  Both: T-B-NK+ SCID + radiosensitivity; "
            "  Distinguish: chromosomal breakage assay + LIG4 vs DCLRE1C sequencing"
        ),
        "disease_pathway": (
            "ARTEMIS (DCLRE1C) — NHEJ HAIRPIN OPENING / END PROCESSING: "
            "STRUCTURE: "
            "  N-terminal β-CASP/metallo-β-lactamase domain: catalytic; two Zn2+ ions; "
            "  C-terminal tail (aa 385-692): DNA-PKcs binding site; regulatory; "
            "  Intrinsic activity: 5'→3' exonuclease on single-stranded DNA; "
            "  Activated by DNA-PKcs: conformational change → endonuclease activity on hairpin + 3' and 5' overhangs; "
            "NHEJ ROLE: "
            "  After Ku/DNA-PKcs complex forms at DSB ends → DNA-PKcs phosphorylates Artemis (S516, S645); "
            "  Activated Artemis: "
            "    (a) Opens V(D)J hairpin intermediates (coding-end hairpins); "
            "    (b) Trims 3' overhangs → blunt end for XRCC4/LIG4 ligation; "
            "    (c) Trims 5' overhangs at complex DSBs; "
            "  After trimming: XRCC4/XLF/LIG4 ligates processed ends; "
            "FANCONI ANEMIA CONNECTION: "
            "  ERCC4/XPF-ERCC1 and Artemis cooperate in ICL unhooking/NHEJ at DSB intermediates; "
            "  DCLRE1C = FANCQ analogy (ERCC4 = FANCQ): alternative nomenclature NOT established for Artemis; "
            "    (ERCC4 is the true FANCQ; Artemis assists NHEJ at ICL-derived DSBs); "
            "ARTEMIS LOF → "
            "  V(D)J hairpins not opened → T-B-NK+ SCID; "
            "  Complex DSB ends not trimmed → radiosensitivity; "
            "  Alkylating agent sensitivity: some crosslinking agents → complex DSBs unresolved"
        ),
        "pathognomonic": (
            "ARTEMIS-SCID DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION: "
            "  T-B-NK+ SCID without microcephaly/pancytopenia (DDx from LIG4); "
            "  FATAL reaction to standard-dose radiation (if given pre-diagnosis); "
            "  Navajo/Athabascan ancestry: SCID with characteristic founder alleles; "
            "  Omenn phenotype: erythroderma + alopecia + eosinophilia + SCID; "
            "NEWBORN SCREENING: "
            "  TRECs absent: identifies T-cell lymphopenia (does not distinguish NHEJ subtypes); "
            "RADIOSENSITIVITY TESTING: "
            "  Chromosomal breakage: elevated dicentrics/aberrations post-irradiation; "
            "  Milder elevation than LIG4 Syndrome (quantitative difference); "
            "IMMUNOLOGICAL WORKUP: "
            "  Flow cytometry: T-B-NK+ pattern; "
            "  Immunoglobulins: panhypogammaglobulinaemia; "
            "  T-cell repertoire: absent or oligoclonal (Omenn); "
            "MOLECULAR CONFIRMATION: "
            "  DCLRE1C sequencing: founder alleles tested first in Navajo/Athabascan; "
            "  Full gene seq + MLPA; "
            "  Protein: Artemis expression western blot; "
            "  Functional: hairpin-opening assay (research); "
            "TREATMENT: "
            "  HSCT: curative — DEFINITIVE treatment; "
            "  REDUCED-INTENSITY CONDITIONING: TBI ABSOLUTELY CONTRAINDICATED; "
            "  Fludarabine + serotherapy conditioning; "
            "  Cyclophosphamide: ABSOLUTELY CONTRAINDICATED at standard doses (crosslinks → complex DSBs unresolvable); "
            "  Gene therapy: clinical trial (OTL-101 for ADA-SCID parallel; Artemis-SCID GT in trials 2024); "
            "  Bridge: IVIG + PJP/antifungal prophylaxis; no live vaccines"
        ),
        "treatment": (
            "ARTEMIS-SCID MANAGEMENT: "
            "HSCT (DEFINITIVE): "
            "  All T-B-NK+ SCID presentations; "
            "  REDUCED INTENSITY CONDITIONING MANDATORY: "
            "  TBI ABSOLUTELY CONTRAINDICATED; "
            "  Busulfan: pharmacokinetically-guided low-dose; "
            "  Cyclophosphamide: ABSOLUTELY CONTRAINDICATED (standard doses); "
            "  Fludarabine + alemtuzumab / ATG = preferred backbone; "
            "  Treosulfan-based protocols also used (lower organ toxicity); "
            "  Graft source: MSD best; MUD acceptable; haploidentical increasing; "
            "  Outcome: >90% survival with appropriate RIC; engraftment depends on conditioning intensity; "
            "GENE THERAPY (EMERGING 2024): "
            "  Self-inactivating lentiviral vector carrying DCLRE1C cDNA; "
            "  Phase I/II trials: enrollment ongoing; "
            "  Advantage: avoids graft rejection; no donor required; eliminates GVHD risk; "
            "  Status: investigational — not yet standard of care; "
            "BRIDGE PRE-HSCT: "
            "  IVIG: 400-600 mg/kg q3-4 weeks; "
            "  PJP prophylaxis: TMP-SMX or pentamidine nebulisation; "
            "  Antifungal: azole; "
            "  CMV pre-emptive therapy: ganciclovir/foscarnet; "
            "  No live vaccines; BCG if given → disseminated BCGiosis (treat with anti-TB); "
            "OMENN SYNDROME VARIANT: "
            "  Immunosuppression (cyclosporin + steroids) before HSCT to control oligoclonal expansion; "
            "  Then proceed to HSCT with RIC; "
            "RADIATION AVOIDANCE: "
            "  Radiotherapy: ABSOLUTELY CONTRAINDICATED; "
            "  Minimise diagnostic X-ray/CT exposure; prefer MRI; "
            "  Medical alert documentation: 'RADIOSENSITIVE — NO RADIOTHERAPY'"
        ),
    },
    {
        "gene": "PRKDC",
        "seed_base": 2752,
        "protein": (
            "PRKDC -- 8q11.21 AR -- 4128aa -- DNA-dependent-Protein-Kinase-Catalytic-Subunit-"
            "DNA-PKcs-470kDa-PIKK-Family-Ser-Thr-Kinase-KU-Cofactor-"
            "OMIM-Gene-600899-Disease-DNA-PKcs-SCID-615966"
        ),
        "locus": "8q11.21",
        "protein_size": "4128 aa / 470 kDa (PIKK family Ser/Thr kinase; FAT + kinase + FATC domains; requires Ku heterodimer for activation; central NHEJ scaffold kinase)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → DNA-PKcs SCID); "
            "ULTRA-RARE: <30 patients described worldwide (2024); "
            "BIALLELIC DISEASE (DNA-PKcs SCID): "
            "  T-B-NK+ SCID: severe combined immunodeficiency; "
            "  Similar to Artemis-SCID in lymphopenia pattern; "
            "  Radiosensitivity: elevated — DNA-PKcs phosphorylates Artemis + autophosphorylation for end release; "
            "  NO microcephaly (unlike LIG4); "
            "  NO growth retardation (unlike LIG4 / XRCC4); "
            "  Some patients: mild neurodevelopmental features; "
            "FOUNDER/COMMON ALLELES: "
            "  ARABIAN: c.9250C>T (p.Leu3084Arg) consanguineous Saudi families; "
            "  PAKISTANI: c.12218T>C (p.Leu4073Pro) — homozygous; "
            "  Pan-ethnic: various frameshift/nonsense; "
            "  Horse PRKDC null = model organism (equine SCID); "
            "SEVERITY: "
            "  Less radiosensitive than Artemis-SCID (DNA-PKcs LOF: Ku still binds but no kinase → different end-processing defect)"
        ),
        "disease_category": (
            "DNA-PKcs SCID (OMIM 615966): "
            "IMMUNOLOGICAL PHENOTYPE: "
            "  T cells: ABSENT (V(D)J hairpin not opened without DNA-PKcs-activated Artemis); "
            "  B cells: ABSENT; "
            "  NK cells: NORMAL; "
            "  Immunoglobulins: absent; maternal IgG only; "
            "  T-B-NK+ SCID pattern identical to Artemis-SCID; "
            "MECHANISM: "
            "  DNA-PKcs is required to activate Artemis (phosphorylation at S516, S645); "
            "  DNA-PKcs LOF → Artemis NOT activated → hairpins not opened → V(D)J fails; "
            "  DNA-PKcs also required for end-synapsis (pulling together DNA ends via long-range synaptic complex); "
            "  DNA-PKcs autophosphorylation (T2609 cluster): releases DNA ends for downstream processing; "
            "RADIOSENSITIVITY: "
            "  Milder than LIG4; intermediate vs Artemis; "
            "  Complex DSBs at radiation-induced clustered damage: incompletely repaired; "
            "  Clinical: chromosomal aberrations elevated post-irradiation; "
            "DIFFERENTIAL DIAGNOSIS: "
            "  T-B-NK+ SCID DDx (without microcephaly/pancytopenia): Artemis-SCID vs DNA-PKcs-SCID vs XLF-SCID; "
            "  Distinguish: DCLRE1C vs PRKDC vs NHEJ1 sequencing; "
            "  Radiosensitivity in all three; less severe in DNA-PKcs (some reports); "
            "PROGNOSIS: "
            "  HSCT curative; "
            "  Ultra-rarity makes outcome data limited to case series"
        ),
        "disease_pathway": (
            "DNA-PKcs (PRKDC) — NHEJ SCAFFOLD KINASE: "
            "STRUCTURE: "
            "  HEAT repeats (aa 1-3559): scaffold for protein-protein interactions; "
            "  FAT domain (aa 3560-3880): structural; "
            "  Kinase domain (aa 3745-4000): PIKK; binds ATP + Mg2+; "
            "  FATC domain (aa 4001-4128): regulatory; "
            "  PQR cluster (T2609, S2638, T2647): phosphorylated by ATM/DNA-PKcs trans-autophosphorylation → "
            "    releases DNA ends from synaptic complex; "
            "  ABCDE cluster (S2056): autophosphorylation → modulates end-access; "
            "NHEJ SCAFFOLD FUNCTION: "
            "  KU70/KU80 bound to DSB end → recruits DNA-PKcs via KU80 C-terminal domain; "
            "  DNA-PKcs + Ku + DNA = DNA-PK holoenzyme; "
            "  Two DNA-PK complexes on opposing ends: long-range synaptic complex; "
            "  DNA-PKcs kinase activity: "
            "    Phosphorylates Artemis → activates hairpin-opening/overhang-trimming; "
            "    Phosphorylates RPA, PALB2, XRCC4, XLF, POLM: coordinates end processing; "
            "    Autophosphorylation: releases ends for ligation; "
            "  After end processing: XRCC4/XLF/LIG4 finalises ligation; "
            "DNA-PKcs LOF → "
            "  Artemis not activated → hairpin-opening blocked → SCID; "
            "  Autophosphorylation absent → ends not released → persistent synaptic complex → reduced ligation; "
            "  Net: V(D)J recombination blocked + radiosensitivity (complex DSBs unresolved)"
        ),
        "pathognomonic": (
            "DNA-PKcs SCID DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION: "
            "  T-B-NK+ SCID without microcephaly / pancytopenia / growth failure; "
            "  Similar to Artemis-SCID phenotype; "
            "  Consanguineous Middle Eastern / South Asian ancestry; "
            "IMMUNOLOGICAL WORKUP: "
            "  Flow cytometry: T-B-NK+ pattern; "
            "  TRECs: absent on newborn screen; "
            "  Immunoglobulins: profoundly low; "
            "RADIOSENSITIVITY TESTING: "
            "  Chromosomal breakage assay: elevated post-irradiation; "
            "  Intermediate between Artemis-SCID (severe) and healthy controls; "
            "MOLECULAR CONFIRMATION: "
            "  PRKDC sequencing: full gene (large: 86 exons) + MLPA; "
            "  Next-generation sequencing panel (SCID gene panel including PRKDC); "
            "  Protein: DNA-PKcs expression western blot; "
            "  Kinase assay: DNA-PK kinase activity (phosphorylation of p53-Ser15 peptide substrate) — reduced/absent; "
            "TREATMENT: "
            "  HSCT: curative — same protocol as Artemis-SCID; "
            "  REDUCED INTENSITY CONDITIONING: TBI CONTRAINDICATED; "
            "  Fludarabine-based RIC; "
            "  Bridge: IVIG + PJP prophylaxis; "
            "  Avoid alkylating agents at myeloablative doses; "
            "  WISKOTT-ALDRICH / RAG SCID: alternative diagnoses to exclude with same phenotype"
        ),
        "treatment": (
            "DNA-PKcs SCID MANAGEMENT: "
            "HSCT (DEFINITIVE): "
            "  All confirmed DNA-PKcs SCID cases; "
            "  REDUCED INTENSITY CONDITIONING MANDATORY: "
            "  TBI ABSOLUTELY CONTRAINDICATED; "
            "  Cyclophosphamide at standard doses: AVOID; "
            "  Fludarabine + serotherapy preferred; "
            "  Treosulfan-based protocols; "
            "  Outcome: limited data (ultra-rare); "
            "  Expect similar engraftment outcomes to Artemis-SCID with appropriate RIC; "
            "BRIDGE: "
            "  IVIG: 400-600 mg/kg q3-4 weeks; "
            "  PJP prophylaxis: TMP-SMX; "
            "  Antifungal: fluconazole; "
            "  CMV monitoring; "
            "RADIATION AVOIDANCE: "
            "  Radiotherapy: ABSOLUTELY CONTRAINDICATED; "
            "  Diagnostic imaging: prefer MRI; "
            "GENETIC COUNSELLING: "
            "  AR; 25% recurrence; "
            "  Carrier screening in consanguineous families"
        ),
    },
    {
        "gene": "XRCC4",
        "seed_base": 2753,
        "protein": (
            "XRCC4 -- 5q14.2 AR -- 336aa -- X-ray-Repair-Cross-Complementing-4-"
            "38kDa-Homodimer-LIG4-Scaffold-NHEJ-Structural-Factor-"
            "OMIM-Gene-194363-Disease-Microcephaly-Dwarfism-XRCC4-616663"
        ),
        "locus": "5q14.2",
        "protein_size": "336 aa / 38 kDa (globular head + stalk homodimer; recruits/stabilises LIG4; XLF-XRCC4 filament scaffold; no catalytic activity)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → XRCC4 deficiency): "
            "BIALLELIC DISEASE: "
            "  Primordial microcephalic dwarfism: OFC severely reduced; linear growth markedly impaired; "
            "  Intellectual disability: mild-moderate in most; "
            "  NO IMMUNODEFICIENCY: T and B cells NORMAL (key DDx from LIG4 Syndrome); "
            "  V(D)J recombination: INTACT (XRCC4 hypomorphic alleles; null lethal in mice); "
            "  Radiosensitivity: elevated chromosomal aberrations; "
            "  Dysmorphic features: facial, limb anomalies; "
            "  Seizures: minority; "
            "COMPLETE NULL XRCC4 = LETHAL IN UTERO (murine): only hypomorphic survive → attenuated phenotype; "
            "FOUNDER/COMMON ALLELES: "
            "  Turkish/consanguineous: c.817C>T (p.Arg273Cys) most reported; "
            "  Various frameshift/missense in consanguineous families (pan-ethnic); "
            "CRITICAL DDx FROM LIG4 SYNDROME: "
            "  Both: microcephaly + dwarfism + radiosensitivity; "
            "  XRCC4: NO immunodeficiency (T-B cells NORMAL); LIG4: T-B-NK+ SCID; "
            "  XRCC4: NO pancytopenia; LIG4: pancytopenia; "
            "  Distinguish: immunological workup + gene sequencing"
        ),
        "disease_category": (
            "XRCC4 DEFICIENCY (OMIM 616663): "
            "GROWTH/NEUROLOGICAL PHENOTYPE: "
            "  Microcephaly: severe; prenatal onset; OFC -4 to -8 SD; "
            "  Short stature: primordial (-4 to -6 SD height); "
            "  Weight: also markedly reduced (not selective to head); "
            "  Intellectual disability: mild-moderate (not always severe); "
            "  Seizures: ~25% of reported patients; "
            "  Dysmorphic: triangular face, high forehead, micrognathia; "
            "  Eye: strabismus, microphthalmia (minority); "
            "  Skeletal: clinodactyly, 5th finger; "
            "IMMUNOLOGICAL PHENOTYPE: "
            "  T cells: NORMAL (key finding — V(D)J preserved with hypomorphic XRCC4); "
            "  B cells: NORMAL; "
            "  NK cells: NORMAL; "
            "  Immunoglobulins: NORMAL; "
            "  Infections: normal susceptibility (unlike LIG4/Artemis/DNA-PKcs); "
            "RADIOSENSITIVITY: "
            "  Chromosomal breakage assay: elevated; "
            "  Mechanism: LIG4 not optimally stabilised → NHEJ ligation inefficient at complex DSBs; "
            "  Not as radiosensitive as LIG4-null; "
            "CANCER RISK: "
            "  Theoretical (NHEJ defect → genomic instability); "
            "  Limited clinical data — insufficient patients for firm cancer risk quantification; "
            "  Annual surveillance suggested by experts pending registry data"
        ),
        "disease_pathway": (
            "XRCC4 — NHEJ STRUCTURAL SCAFFOLD / LIG4 STABILISER: "
            "STRUCTURE: "
            "  N-terminal globular head (aa 1-119): β-barrel; homodimerisation; "
            "  C-terminal stalk + coiled-coil (aa 120-200): LIG4 BRCT-binding surface; "
            "  Flexible tail (aa 200-336): phosphorylation targets (DNA-PKcs); "
            "FUNCTION IN NHEJ: "
            "  XRCC4 homodimer recruits LIG4 (via BRCT-BRCT tandem of LIG4 wrapping around XRCC4 stalk); "
            "  XRCC4/LIG4 complex stabilised at DSB ends; "
            "  XLF (NHEJ1) forms filamentous scaffold with XRCC4: "
            "    XLF-XRCC4 filament bridges DNA ends across the synapse; "
            "    Filament structure stimulates LIG4 activity; "
            "  PAXX (PAXX-XRCC4-LIG4): PAXX paralogue — interacts with Ku, synergises with XRCC4/XLF; "
            "REGULATORY PHOSPHORYLATION: "
            "  XRCC4 Ser325, Ser326: phosphorylated by DNA-PKcs → modulates LIG4 stimulation; "
            "  XRCC4 Thr233: CK2 phosphorylation → stability; "
            "XRCC4 LOF → "
            "  LIG4 destabilised at DSB ends → reduced ligation efficiency → "
            "    DSBs accumulate during replication → neuroprogenitor apoptosis → microcephaly; "
            "  V(D)J: hypomorphic alleles allow partial ligation → immune system relatively spared; "
            "  Radiosensitivity: ligation deficiency at complex DSBs → chromosomal instability"
        ),
        "pathognomonic": (
            "XRCC4 DEFICIENCY DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION: "
            "  Primordial microcephalic dwarfism + intellectual disability; "
            "  NORMAL immune function (no recurrent infections, normal lymphocyte counts); "
            "  Radiosensitivity on chromosomal breakage testing; "
            "  Consanguineous families (AR); "
            "DDx FROM LIG4 SYNDROME: "
            "  XRCC4: T-B NORMAL; LIG4: T-B-NK+ SCID; "
            "  First step: full blood count + lymphocyte subsets; "
            "  If lymphopenic → LIG4/Artemis/PRKDC/NHEJ1 more likely than XRCC4; "
            "RADIOSENSITIVITY TESTING: "
            "  Chromosomal breakage: elevated; "
            "  Less severe than LIG4 Syndrome; "
            "MOLECULAR CONFIRMATION: "
            "  XRCC4 sequencing + MLPA; "
            "  Protein: XRCC4 and LIG4 western blot (co-IP); "
            "  LIG4 co-immunoprecipitation with XRCC4 reduced; "
            "NEUROIMAGING: "
            "  MRI brain: simplified gyral pattern; thin cortex; reduced white matter; "
            "  Head circumference progression: document serially; "
            "TREATMENT: "
            "  NO HSCT INDICATED (immune system normal); "
            "  Neurodevelopmental support: physiotherapy, speech, cognitive; "
            "  Seizure management: standard AEDs (LEV first-line for focal; VPA avoid if POLG concern not applicable here); "
            "  Radiosensitivity: avoid therapeutic radiation; minimise diagnostic imaging; "
            "  Cancer surveillance: annual clinical exam; tumour registry for data; "
            "  Growth hormone: no established role (primordial dwarfism — GH axis intact but skeletal response limited)"
        ),
        "treatment": (
            "XRCC4 DEFICIENCY MANAGEMENT: "
            "NEUROLOGICAL: "
            "  Developmental support: early intervention (physiotherapy + OT + speech); "
            "  Epilepsy: standard AEDs — LEV (first-line, renal excretion, minimal interactions); "
            "  Avoid phenobarbitone (sedation risk); VPA acceptable (no POLG concern); "
            "  Neuroimaging: MRI brain at diagnosis; repeat if seizures or regression; "
            "GROWTH: "
            "  Nutritional optimisation: adequate caloric density; "
            "  Growth hormone: NOT indicated (GH resistance typical in primordial dwarfism); "
            "  Endocrine surveillance: puberty timing, thyroid, ACTH axis; "
            "RADIOSENSITIVITY MANAGEMENT: "
            "  No therapeutic radiotherapy; "
            "  Minimise CT; prefer MRI/ultrasound; "
            "  Medical alert: 'XRCC4 DEFICIENCY — RADIOSENSITIVE'; "
            "  Alkylating agents: AVOID (bleomycin, cyclophosphamide at cytotoxic doses); "
            "CANCER SURVEILLANCE: "
            "  Annual: CBC + differential, LDH, clinical exam; "
            "  Lymphoma: low threshold for imaging if adenopathy; "
            "GENETIC COUNSELLING: "
            "  AR; 25% sibling recurrence; "
            "  Prenatal testing available"
        ),
    },
    {
        "gene": "NHEJ1",
        "seed_base": 2754,
        "protein": (
            "NHEJ1 -- 2q35 AR -- 299aa -- XLF-Non-Homologous-End-Joining-Factor-1-"
            "Cernunnos-33kDa-XRCC4-Paralogue-Filament-Stimulator-"
            "OMIM-Gene-611290-Disease-XLF-SCID-611291"
        ),
        "locus": "2q35",
        "protein_size": "299 aa / 33 kDa (XRCC4-paralogue; Cernunnos; globular head + stalk; filament with XRCC4 bridges DNA ends; stimulates LIG4 activity)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → XLF deficiency / NHEJ1-associated SCID); "
            "BIALLELIC DISEASE (XLF SCID): "
            "  T-B+/- NK+ SCID: variable T lymphopenia; B cells variable; "
            "  MILDER than Artemis-SCID/LIG4 Syndrome typically; "
            "  Growth retardation: short stature; "
            "  Microcephaly: milder than LIG4/XRCC4; "
            "  Radiosensitivity: elevated chromosomal aberrations post-irradiation; "
            "  Infections: recurrent sinopulmonary; PJP risk; "
            "COMPLETE NULL XLF: NOT lethal in mice (unlike LIG4/XRCC4 null); "
            "  Mouse xlf-/- viable + fertile: partial SCID + radiosensitivity; "
            "  Human null: SCID phenotype because human NHEJ more XLF-dependent than murine; "
            "FOUNDER/COMMON ALLELES: "
            "  No pan-ethnic founder; "
            "  Frameshift + nonsense most common; "
            "  c.169C>T (p.Gln57Ter): reported in multiple families; "
            "  Consanguineous Turkish/Pakistani: various homozygous LOF alleles; "
            "SEVERITY SPECTRUM: "
            "  Hypomorphic NHEJ1: T-B+ SCID (some B cells, fewer T cells); "
            "  Complete null: T-B-NK+ SCID"
        ),
        "disease_category": (
            "XLF DEFICIENCY / NHEJ1-ASSOCIATED SCID (OMIM 611291): "
            "IMMUNOLOGICAL PHENOTYPE: "
            "  T cells: severely reduced to absent; "
            "  B cells: variable (T-B+ or T-B-NK+ depending on residual XLF function); "
            "  NK cells: NORMAL; "
            "  Immunoglobulins: profoundly low; "
            "  V(D)J: XLF stimulates LIG4 activity → XLF LOF → partial V(D)J failure → SCID; "
            "  Milder than Artemis or DNA-PKcs SCID in some cohorts (residual NHEJ-alternative backup); "
            "GROWTH PHENOTYPE: "
            "  Short stature: -2 to -4 SD; "
            "  Microcephaly: mild to moderate (not as severe as XRCC4/LIG4); "
            "RADIOSENSITIVITY: "
            "  Chromosomal breakage: elevated post-irradiation; "
            "  Mechanism: XLF filament-scaffold absent → XRCC4/LIG4 less stimulated → ligation inefficiency; "
            "  Less radiosensitive than LIG4 or Artemis; "
            "DYSMORPHIC: "
            "  Facial: mild coarse features; epicanthal folds; "
            "  Limbs: generally normal; "
            "DIFFERENTIAL DIAGNOSIS: "
            "  T-B± NK+ SCID + radiosensitivity + growth failure: "
            "    LIG4 (pancytopenia, severe microcephaly, lymphoma risk); "
            "    Artemis (no growth failure, T-B-NK+); "
            "    DNA-PKcs (no growth failure, T-B-NK+); "
            "    XLF (milder, variable B cells, moderate growth failure)"
        ),
        "disease_pathway": (
            "XLF / NHEJ1 — NHEJ FILAMENT SCAFFOLD / LIG4 STIMULATOR: "
            "STRUCTURE: "
            "  N-terminal globular head (aa 1-126): XRCC4-paralogue fold; "
            "  C-terminal stalk (aa 127-299): coiled-coil; "
            "  L115 residue: key contact for XRCC4 interaction; "
            "XRCC4-XLF FILAMENT: "
            "  XLF homodimer alternates with XRCC4 homodimer → filament (polymer); "
            "  Filament bridges the two DNA ends in the NHEJ synapse; "
            "  Filament recruitment of LIG4: stimulates LIG4 adenylation + nick-sealing; "
            "  KU70/KU80 interacts with both XRCC4 and XLF → filament formed at DSB ends; "
            "LIG4 STIMULATION MECHANISM: "
            "  XLF stimulates LIG4 ligation of incompatible ends (mismatched or 3'/5' overhang pairs); "
            "  Extends LIG4 tolerance for end microheterogeneity; "
            "  Critical for complex end ligation in vivo; "
            "BACKUP PATHWAYS: "
            "  PAXX (PAXX-XRCC4-LIG4 axis) and DNA-PKcs partially compensate for XLF absence in mice; "
            "  Humans: less effective backup → SCID phenotype; "
            "XLF LOF → "
            "  XRCC4/LIG4 complex destabilised at synapse → reduced ligation stimulation; "
            "  V(D)J ligation: partial failure → SCID; "
            "  Complex DSBs: reduced ligation efficiency → radiosensitivity"
        ),
        "pathognomonic": (
            "XLF DEFICIENCY DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION: "
            "  T-B± NK+ SCID with moderate growth failure + mild microcephaly; "
            "  Milder presentation than LIG4 or Artemis; "
            "  Recurrent sinopulmonary infections ± PJP; "
            "  Radiosensitivity (chromosomal breakage testing); "
            "IMMUNOLOGICAL WORKUP: "
            "  Flow cytometry: T severely reduced; B variable; NK normal; "
            "  TRECs: low or absent; "
            "  Immunoglobulins: profoundly low; "
            "RADIOSENSITIVITY TESTING: "
            "  Chromosomal breakage: elevated (typically less than LIG4, more than healthy); "
            "MOLECULAR CONFIRMATION: "
            "  NHEJ1 sequencing + MLPA; "
            "  Protein: XLF western blot; "
            "  XRCC4/LIG4/XLF triple western blot; "
            "TREATMENT: "
            "  HSCT: curative if SCID severe; "
            "  REDUCED-INTENSITY CONDITIONING: TBI CONTRAINDICATED; "
            "  Fludarabine + serotherapy; "
            "  Milder SCID: consider IVIG + prophylaxis alone if partial T-cell function; "
            "  PJP + antifungal prophylaxis; "
            "  Radiation avoidance (radiosensitivity); "
            "  GENETIC COUNSELLING: AR; 25% recurrence"
        ),
        "treatment": (
            "XLF DEFICIENCY MANAGEMENT: "
            "HSCT: "
            "  Indicated for confirmed SCID (T-B-NK+ or severe T-B+NK+ with infections); "
            "  REDUCED INTENSITY CONDITIONING: TBI ABSOLUTELY CONTRAINDICATED; "
            "  Fludarabine + serotherapy backbone; "
            "  Milder presentations: IVIG bridge ± prophylaxis; assess T-cell function trajectory; "
            "BRIDGE: "
            "  IVIG 400-600 mg/kg q3-4 weeks; "
            "  TMP-SMX (PJP prophylaxis); "
            "  Antifungal; no live vaccines; "
            "RADIOSENSITIVITY: "
            "  No radiotherapy; minimise X-ray/CT; prefer MRI; "
            "  Medical documentation: 'RADIOSENSITIVE — NHEJ1/XLF DEFICIENCY'; "
            "GROWTH: "
            "  Endocrine: assess GH axis; "
            "  Growth hormone: trial if GH deficient; "
            "  Nutritional support; "
            "GENETIC COUNSELLING: AR; 25% sibling risk"
        ),
    },
    {
        "gene": "PNKP",
        "seed_base": 2755,
        "protein": (
            "PNKP -- 19q13.33 AR -- 521aa -- Polynucleotide-Kinase-3prime-Phosphatase-"
            "57kDa-Bifunctional-5prime-Kinase-3prime-Phosphatase-FHA-Domain-"
            "OMIM-Gene-605610-Disease-MCSZ-613402-AOA4-616267"
        ),
        "locus": "19q13.33",
        "protein_size": "521 aa / 57 kDa (FHA domain + phosphatase domain + kinase domain; bifunctional: 5'-kinase restores 5'-P; 3'-phosphatase removes 3'-P — both required for ligation)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE: two distinct phenotypes depending on allele: "
            "PHENOTYPE 1 — MCSZ (Microcephaly-Seizures-Developmental delay): "
            "  Missense alleles (hypomorphic): c.1029C>A (p.Thr424=synonymous/splicing), c.1213C>T (p.Arg405Cys); "
            "  Severe epilepsy (infantile onset); "
            "  Microcephaly; "
            "  Intellectual disability: severe; "
            "  Progressive neurodegeneration; "
            "  NO significant ataxia (DDx AOA4); "
            "PHENOTYPE 2 — AOA4 (Ataxia with Oculomotor Apraxia type 4): "
            "  Truncating + missense compound heterozygous with some residual function; "
            "  Late-onset cerebellar ataxia (childhood/adolescence onset); "
            "  Oculomotor apraxia; "
            "  Peripheral sensorimotor neuropathy; "
            "  Less severe epilepsy (rare); "
            "  Elevated AFP: variable; "
            "PREVALENCE: "
            "  MCSZ: rare; described in multiple ethnicities; "
            "  AOA4: distinct from AOA1 (APTX) and AOA2 (SETX); "
            "FOUNDER ALLELES: "
            "  Palestinian/Jordanian: MCSZ cluster with specific haplotypes"
        ),
        "disease_category": (
            "MCSZ / AOA4 (OMIM 613402 / 616267): "
            "MCSZ PHENOTYPE: "
            "  Microcephaly: OFC -3 to -5 SD; "
            "  Seizures: infantile-onset; myoclonic + focal; drug-resistant in many; "
            "  Intellectual disability: severe; "
            "  Hypotonia: neonatal/infantile; "
            "  Developmental regression: possible; "
            "  EEG: multifocal epileptiform discharges; hypsarrhythmia possible in infancy; "
            "  MRI: simplified gyri; thin corpus callosum; cerebellar hypoplasia; "
            "AOA4 PHENOTYPE: "
            "  Ataxia: progressive cerebellar; onset childhood-adolescence; "
            "  Oculomotor apraxia: saccadic initiation failure; "
            "  Peripheral neuropathy: sensorimotor axonal; "
            "  AFP: mildly elevated (overlap with AOA2); "
            "  IgA: low in subset; "
            "NO IMMUNODEFICIENCY IN EITHER PHENOTYPE: "
            "  T/B/NK normal; immunoglobulins normal; "
            "  Infections: normal susceptibility; "
            "RADIOSENSITIVITY: "
            "  Mild chromosomal breakage elevation; "
            "  PNKP role in NHEJ end-processing → radiation-induced complex ends not fully processed; "
            "CANCER RISK: "
            "  Limited data; theoretical risk from DNA repair defect; "
            "  No established cancer surveillance guideline"
        ),
        "disease_pathway": (
            "PNKP — SSBR/NHEJ END-PROCESSING: BIFUNCTIONAL KINASE/PHOSPHATASE: "
            "STRUCTURE: "
            "  FHA (Forkhead-Associated) domain (aa 1-108): phosphoprotein binding; binds XRCC4 (pSer325) + CK2-pXRCC1; "
            "  Phosphatase domain (aa 109-337): Mg2+/Mn2+-dependent 3'-phosphomonoesterase; "
            "  Kinase domain (aa 338-521): ATP-dependent polynucleotide 5'-kinase; "
            "SSBR END-PROCESSING ROLE: "
            "  Ionising radiation / topoisomerase I cleavage complexes produce SSBs with: "
            "    3'-phosphate (from β-elimination) or 3'-phosphoglycolate → PNKP 3'-phosphatase removes → 3'-OH; "
            "    5'-hydroxyl (from TOP1-cleavage complex) → PNKP 5'-kinase adds 5'-P; "
            "  XRCC1 recruits PNKP to SSB (via XRCC1-FHA-PNKP interaction; CK2 phospho-XRCC1 binds FHA); "
            "  After PNKP: pol β fills gap; LIG3/XRCC1 seals nick (SSBR pathway); "
            "NHEJ END-PROCESSING ROLE: "
            "  Complex DSB ends with 3'-phosphate or 5'-OH → PNKP restores 3'-OH and 5'-P; "
            "  XRCC4 (pSer325) recruits PNKP to NHEJ complex; "
            "  Enables LIG4 to seal (LIG4 requires 3'-OH and 5'-P); "
            "BER ROLE: "
            "  PNKP also processes certain BER intermediates (NEIL1/NEIL2 β-δ lyase products → 3'-phosphate); "
            "PNKP LOF → "
            "  3'-phosphate/5'-OH termini accumulate → ligation impossible → "
            "    SSBs + DSBs persist → neuroprogenitor/neuron apoptosis → microcephaly + seizures (MCSZ); "
            "    Progressive neurodegeneration (AOA4)"
        ),
        "pathognomonic": (
            "MCSZ / AOA4 DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION (MCSZ): "
            "  Microcephaly + infantile seizures + intellectual disability; "
            "  Drug-resistant epilepsy; "
            "  NO immunodeficiency; "
            "  Normal immunoglobulins + lymphocyte subsets; "
            "CLINICAL SUSPICION (AOA4): "
            "  Progressive cerebellar ataxia + oculomotor apraxia + peripheral neuropathy; "
            "  Onset childhood/adolescence; "
            "  AFP mildly elevated (overlap AOA2/SETX); "
            "  APTX (AOA1) excluded (different gene — APTX vs PNKP); "
            "INVESTIGATIONS: "
            "  CBC: normal (no pancytopenia); "
            "  Neuroimaging MRI: cerebellar atrophy (AOA4); simplified gyri (MCSZ); "
            "  EEG: epileptiform activity (MCSZ); "
            "  NCS/EMG: axonal sensorimotor neuropathy (AOA4); "
            "  AFP: mildly elevated (AOA4); "
            "  Immunology: normal; "
            "MOLECULAR: "
            "  PNKP gene sequencing + MLPA; "
            "  Ataxia gene panel (includes PNKP, APTX, SETX, ATM, ADCK3); "
            "TREATMENT: "
            "  SEIZURES (MCSZ): "
            "    LEV (levetiracetam): first-line — broadest spectrum, renal excretion; "
            "    VPA: avoid if possible (hepatotoxicity concern in mtDNA-adjacent phenotypes); "
            "    KD: consider if drug-resistant; "
            "    ACTH: if infantile spasms/hypsarrhythmia; "
            "  ATAXIA (AOA4): physiotherapy; occupational therapy; ankle-foot orthoses; "
            "  Neuropathy: pain management; orthotic devices; "
            "  NO radiotherapy (radiosensitivity risk)"
        ),
        "treatment": (
            "MCSZ / AOA4 MANAGEMENT: "
            "SEIZURES: "
            "  LEV: 20-60 mg/kg/day — first-line (broad-spectrum, renal, minimal interactions); "
            "  CLB (clobazam): adjunct for focal; "
            "  RUF (rufinamide): adjunct for atonic/tonic; "
            "  KD (ketogenic diet): consider if ≥2 AEDs failed; "
            "  ACTH: if infantile spasms in infancy; "
            "  VPA: use with caution (theoretical concern for mitochondrial overlap; but PNKP ≠ mtDNA depletion gene); "
            "  PHT/CBZ: AVOID in myoclonic component; "
            "NEURODEVELOPMENTAL: "
            "  Physiotherapy: tone management; developmental milestones; "
            "  OT: fine motor; daily living; "
            "  Speech: augmentative communication if nonverbal; "
            "ATAXIA (AOA4): "
            "  Physiotherapy: balance; gait aid (rollator, wheelchair as needed); "
            "  OT: independence; "
            "  Vitamin E supplementation: often given empirically in ataxia (no established trial for PNKP); "
            "CARDIAC/ENDOCRINE: annual review; "
            "RADIOSENSITIVITY: "
            "  Avoid radiotherapy; minimise diagnostic imaging; "
            "GENETIC COUNSELLING: AR; 25% sibling risk"
        ),
    },
    {
        "gene": "APTX",
        "seed_base": 2756,
        "protein": (
            "APTX -- 9p21.1 AR -- 342aa -- Aprataxin-"
            "38kDa-HIT-Zn-Finger-FHA-Domain-Dead-End-5prime-Adenylate-Removal-"
            "OMIM-Gene-606350-Disease-AOA1-208920"
        ),
        "locus": "9p21.1",
        "protein_size": "342 aa / 38 kDa (FHA domain + HIT (histidine-triad) Zn-finger; resolves 5'-adenylate (5'-AMP) dead-end ligation intermediates; cannot open SSBR nick otherwise)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → AOA1): "
            "BIALLELIC DISEASE (AOA1 — Ataxia with Oculomotor Apraxia type 1): "
            "  Cerebellar ataxia: onset 2-10 years; progressive; "
            "  Oculomotor apraxia: saccadic initiation failure (OA); "
            "    Type A: initial presentation (early); "
            "    Type B: 3-4 years later; "
            "  Peripheral neuropathy: axonal sensorimotor; choreoathetosis (early, disappears); "
            "  Chorea: early feature (resolves); dystonia (minority); "
            "  Cognitive decline: in advanced stages; "
            "  Hypoalbuminaemia: elevated cholesterol; LOW albumin (DDx from A-T which has NORMAL albumin); "
            "  AFP: NORMAL in AOA1 (elevated in AOA2/SETX, A-T) — KEY DDx; "
            "  No immunodeficiency; "
            "FOUNDER/COMMON ALLELES: "
            "  Portuguese: c.837_838del (p.Lys280Glufs; Portuguese founder); "
            "  Japanese: W279X (Trp279Ter) most common; "
            "  Other: various LOF alleles; "
            "PREVALENCE: "
            "  Autosomal recessive ataxia: 2nd most common after Friedreich (in some populations); "
            "  Portuguese: ~1 in 500,000; "
            "  Japan: significant frequency"
        ),
        "disease_category": (
            "AOA1 / APRATAXIN DEFICIENCY (OMIM 208920): "
            "CEREBELLAR PHENOTYPE: "
            "  Gait ataxia: onset 2-10 years; waddling gait; balance board impossible; "
            "  Limb ataxia: finger-nose dysmetria; dysdiadochokinesia; "
            "  Dysarthria: cerebellar (scanning/explosive speech); "
            "  Dysphagia: late complication; "
            "  Wheelchair: typically 15-25 years after onset; "
            "OCULOMOTOR FEATURES: "
            "  Oculomotor apraxia (OA): inability to initiate voluntary saccades; "
            "  Compensatory head thrusts (Cogan sign); "
            "  Slow saccades (as disease progresses); "
            "  Nystagmus: uncommon early; "
            "PERIPHERAL NERVOUS SYSTEM: "
            "  Sensorimotor axonal neuropathy: reduced/absent reflexes; "
            "  NCS/EMG: axonal pattern; "
            "  Choreiform movements: early (may disappear spontaneously); "
            "METABOLIC BIOMARKERS: "
            "  Hypoalbuminaemia: albumin < 3.5 g/dL (not in all, but characteristic); "
            "  Hypercholesterolaemia: elevated LDL + total cholesterol; "
            "  AFP: NORMAL (critical DDx — A-T and AOA2 have elevated AFP); "
            "  CK: normal or mildly elevated; "
            "NEUROIMAGING: "
            "  MRI: cerebellar atrophy (vermis + hemispheres); "
            "  Cortical atrophy: late; "
            "  NO basal ganglia signal; "
            "CANCER RISK: "
            "  Very low / not established (unlike A-T); "
            "  No increased lymphoma risk; "
            "  Radiosensitivity: not clinically significant (unlike A-T)"
        ),
        "disease_pathway": (
            "APRATAXIN (APTX) — 5'-ADENYLATE REMOVAL: SSBR DEAD-END INTERMEDIATE RESOLUTION: "
            "THE PROBLEM — ABORTIVE LIGATION: "
            "  DNA LIGASE mechanism: (1) Ligase-AMP forms (AMP from NAD+ or ATP); "
            "    (2) AMP transferred to 5'-phosphate of nick → 5'-adenylate intermediate; "
            "    (3) 3'-OH attacks 5'-AMP → phosphodiester bond; "
            "  ABORTIVE LIGATION: if nick has wrong chemistry (3'-phosphate, 3'-PG, mismatched base) → "
            "    Ligase releases AMP-adenylated 5'-end BLOCKED = 5'-adenylate dead-end; "
            "    5'-adenylate blocks all further repair (blocks polymerase access + re-ligation); "
            "APRATAXIN FUNCTION: "
            "  FHA domain: binds CK2-phosphorylated XRCC1 → recruited to SSBR complex; "
            "  HIT-Zn finger: nucleophilic attack on 5'-AMP → removes AMP → restores 5'-phosphate; "
            "  Net: APTX removes 5'-AMP → allows re-ligation attempt by LIG3/LIG1; "
            "  Also resolves 3'-phosphate to enable ligation where partial: "
            "    (PNKP + APTX cooperate at SSBR termini); "
            "APTX LOF → "
            "  5'-adenylate dead-ends accumulate → SSBs cannot be re-ligated → "
            "    Purkinje cells + cerebellar neurons hypersensitive (high transcriptional activity = high TOP1/LIG usage) → "
            "    Progressive cerebellar neurodegeneration → AOA1; "
            "  Lower AFP (unlike ATM-null): ATM-activated pathway not disrupted; "
            "  Hypoalbuminaemia: mechanism unclear (nutritional + possible liver stress?)"
        ),
        "pathognomonic": (
            "AOA1 DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION: "
            "  Childhood/adolescent progressive cerebellar ataxia + oculomotor apraxia; "
            "  Normal AFP (excludes A-T and AOA2/SETX); "
            "  Hypoalbuminaemia (supports AOA1); "
            "  Sensorimotor axonal neuropathy; "
            "  Early choreiform movements (resolves); "
            "  Portuguese or Japanese ancestry: increased prior probability; "
            "BIOMARKERS: "
            "  AFP: NORMAL (KEY DDx — elevated in A-T/AOA2/AOA3); "
            "  Albumin: LOW (< 3.5 g/dL) — characteristic; "
            "  Cholesterol: elevated; "
            "  Immunoglobulins: NORMAL (unlike A-T where low IgA/IgG); "
            "  Lymphocytes: NORMAL (unlike A-T where lymphopenia); "
            "NEUROIMAGING: "
            "  MRI: cerebellar atrophy (early vermis > hemispheres); "
            "  EMG/NCS: axonal neuropathy; "
            "MOLECULAR: "
            "  APTX sequencing + MLPA; "
            "  Ataxia panel (includes APTX, SETX, ATM, PNKP, ANO10, ADCK3, SACS); "
            "TREATMENT: "
            "  No disease-modifying treatment; "
            "  Physiotherapy: balance + gait; occupational therapy; "
            "  Speech therapy: dysarthria management; "
            "  Wheelchair: anticipate need at 15-25 years post-onset; "
            "  Chorea: propranolol / clonazepam (if troublesome); "
            "  Cholesterol: statin (hypercholesterolaemia); "
            "  Albumin: nutritional optimisation; "
            "  NO radiosensitivity precautions (APTX deficiency NOT radiation sensitive in clinic)"
        ),
        "treatment": (
            "AOA1 MANAGEMENT: "
            "ATAXIA: "
            "  Physiotherapy: balance board; proprioceptive exercises; gait retraining; "
            "  Walking aids: cane → rollator → wheelchair (disease progression); "
            "  Adaptive equipment: ADL aids; "
            "  Hydrotherapy: pool physiotherapy (reduces injury from falls); "
            "OCULOMOTOR APRAXIA: "
            "  Low vision rehabilitation: "
            "  Compensatory head thrust strategies; "
            "  Reading aids; large print; audiobooks; "
            "CHOREA (EARLY): "
            "  Propranolol 10-80 mg/day (sympatholytic); "
            "  Clonazepam 0.5-2 mg/day (GABA agonist); "
            "  Choreiform movements usually self-resolve; "
            "METABOLIC: "
            "  Hypercholesterolaemia: statin (atorvastatin 20-40 mg); "
            "  Hypoalbuminaemia: nutritional enrichment; dietitian; protein supplementation; "
            "NEUROPATHY: "
            "  Foot drop: ankle-foot orthoses; "
            "  Pain: gabapentin / pregabalin; "
            "SPEECH: "
            "  Speech therapy: dysarthria; AAC (augmentative/alternative communication) when dysarthria advanced; "
            "PSYCHOLOGICAL: "
            "  Cognitive assessment; neuropsychological support; career counselling; "
            "GENETIC COUNSELLING: AR; 25% sibling recurrence"
        ),
    },
    {
        "gene": "TDP1",
        "seed_base": 2757,
        "protein": (
            "TDP1 -- 14q31.3 AR -- 608aa -- Tyrosyl-DNA-Phosphodiesterase-1-"
            "68kDa-PLD-Superfamily-3prime-Phosphotyrosyl-Bond-Hydrolysis-"
            "OMIM-Gene-607198-Disease-SCAN1-607250"
        ),
        "locus": "14q31.3",
        "protein_size": "608 aa / 68 kDa (PLD superfamily phosphodiesterase; two HxK motifs; hydrolyses 3'-phosphotyrosyl bond — TOP1-covalent complex dead-end; SSBR pathway)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → SCAN1): "
            "SCAN1 (Spinocerebellar Ataxia with Axonal Neuropathy type 1): "
            "  ULTRA-RARE: single large Saudi Arabian kindred (Takashima 2002); very few additional patients; "
            "  Cerebellar ataxia: onset 10-14 years; progressive; "
            "  Peripheral neuropathy: axonal sensorimotor; prominent; "
            "  Oculomotor apraxia: ABSENT (DDx from AOA1 — OA present in AOA1, absent in SCAN1); "
            "  Chorea: absent; "
            "  Normal AFP (like AOA1 — DDx from A-T/AOA2); "
            "  Normal albumin (unlike AOA1 — DDx: AOA1 has hypoalbuminaemia; SCAN1 normal albumin); "
            "  No immunodeficiency; "
            "  Cognitive: mildly impaired in advanced disease; "
            "FOUNDER ALLELE: "
            "  Saudi Arabian kindred: c.1478A>G (p.His493Arg) — homozygous; "
            "  Single amino acid substitution in HxK motif → catalytic dead; "
            "  Extremely rare globally; "
            "CAMPTOTHECIN SENSITIVITY: "
            "  TDP1-deficient cells: hypersensitive to camptothecin (TOP1 inhibitor) in vitro; "
            "  Potential cancer drug sensitivity — pharmacogenomic relevance (theoretical)"
        ),
        "disease_category": (
            "SCAN1 / TDP1 DEFICIENCY (OMIM 607250): "
            "CEREBELLAR PHENOTYPE: "
            "  Progressive cerebellar ataxia: onset 10-14 years; "
            "  Gait ataxia + limb dysmetria; "
            "  Dysarthria: moderate; "
            "  Wheelchair typically 2nd-3rd decade; "
            "  Cerebellar volume loss: MRI; "
            "PERIPHERAL NERVOUS SYSTEM: "
            "  Sensorimotor axonal neuropathy: reduced/absent tendon reflexes; steppage gait; "
            "  Prominent neuropathy (more severe than AOA1); "
            "  NCS/EMG: axonal pattern; "
            "DDx FROM AOA1 (KEY DISTINCTIONS): "
            "  SCAN1: NO oculomotor apraxia; AOA1: OA prominent; "
            "  SCAN1: albumin NORMAL; AOA1: hypoalbuminaemia; "
            "  SCAN1: camptothecin sensitivity; AOA1: APTX removes 5'-AMP (different mechanism); "
            "  Both: AFP normal; cerebellar ataxia; neuropathy; AR; "
            "METABOLIC BIOMARKERS: "
            "  AFP: NORMAL; "
            "  Albumin: NORMAL (distinguishes from AOA1); "
            "  LDL: normal (distinguishes from AOA1 which has hypercholesterolaemia); "
            "  Immunoglobulins: normal; "
            "RADIOSENSITIVITY: "
            "  Minimal/none clinically (TOP1-cleavage complex resolution separate from radiation-induced DSBs); "
            "CANCER: "
            "  Camptothecin hypersensitivity (in vitro) — if SCAN1 patient ever requires TOP1-inhibitor chemotherapy; "
            "  Irinotecan, topotecan: AVOID in TDP1 deficiency (pharmacogenomics)"
        ),
        "disease_pathway": (
            "TDP1 — 3'-PHOSPHOTYROSYL BOND HYDROLYSIS: TOP1-CLEAVAGE COMPLEX RESOLUTION: "
            "THE PROBLEM — TOPOISOMERASE I CLEAVAGE COMPLEX (TOP1cc): "
            "  TOP1 mechanism: "
            "    (1) TOP1 nicks one strand, covalently attaches to 3'-phosphate (TOP1-cc = SSBR intermediate); "
            "    (2) Strand rotation relaxes torsional stress; "
            "    (3) TOP1 re-ligates nick; "
            "  ABORTIVE TOP1cc: if nick is already present ahead (camptothecin traps) OR nearby SSB → "
            "    TOP1cc encounters nick → cannot re-ligate → permanent TOP1-DNA adduct (3'-phosphotyrosyl dead-end); "
            "    Collision with RNA polymerase II or replication fork → DSB; "
            "TDP1 FUNCTION: "
            "  Tyrosyl-DNA phosphodiesterase 1: PLD superfamily; "
            "  Two HxK catalytic motifs (H263, H493): coordinate active site; "
            "  Hydrolyses 3'-phosphotyrosyl bond: TOP1-polypeptide removed → 3'-phosphate; "
            "  3'-phosphate: PNKP then removes (restores 3'-OH); "
            "  TDP1 + PNKP cooperate in TOP1cc resolution; "
            "  3'-OH + 5'-P → LIG3/XRCC1 seals nick (SSBR); "
            "REPAIR PATHWAY: "
            "  TOP1cc → TDP1 (hydrolysis) → PNKP (3'-phosphatase) → Pol β (fill) → LIG3/XRCC1 (seal); "
            "  If collision with replication fork → DSB → TDP2 (Y-phosphotyrosyl, TOP2cc) or MRE11 pathway; "
            "TDP1 LOF → "
            "  TOP1cc dead-ends accumulate → SSBs + DSBs in transcriptionally active neurons → "
            "    Purkinje cells + dorsal root ganglia neurons preferentially affected → "
            "    Cerebellar ataxia + peripheral neuropathy → SCAN1; "
            "  Camptothecin: traps TOP1cc normally transiently → in TDP1 LOF → permanent adduct → lethal"
        ),
        "pathognomonic": (
            "SCAN1 DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION: "
            "  Progressive cerebellar ataxia + prominent axonal neuropathy; "
            "  Onset 10-14 years; "
            "  NO oculomotor apraxia (DDx from AOA1); "
            "  Normal AFP + normal albumin (DDx: A-T elevated AFP; AOA1 low albumin); "
            "  Saudi Arabian ancestry or consanguineous Middle Eastern; "
            "BIOMARKERS: "
            "  AFP: NORMAL; albumin: NORMAL; cholesterol: NORMAL; "
            "  Immunoglobulins: NORMAL; lymphocytes: NORMAL; "
            "NEUROIMAGING: "
            "  MRI: cerebellar atrophy; "
            "  EMG/NCS: axonal sensorimotor neuropathy; "
            "MOLECULAR: "
            "  TDP1 sequencing: p.His493Arg (SCAN1 founder allele) first; "
            "  Full TDP1 gene sequencing; "
            "  Ataxia gene panel; "
            "CAMPTOTHECIN TESTING: "
            "  Research assay: TDP1-deficient cell hypersensitivity to camptothecin (not routine clinical); "
            "TREATMENT: "
            "  No disease-modifying treatment; "
            "  Physiotherapy + OT; "
            "  Wheelchair planning (2nd-3rd decade); "
            "  Neuropathy management: pain (gabapentin/pregabalin); orthoses; "
            "  Speech: AAC when dysarthric; "
            "  PHARMACOGENOMICS: "
            "    AVOID camptothecin-based chemotherapy: irinotecan, topotecan; "
            "    If oncology required: AVOID TOP1-inhibitors; use alternative agents; "
            "  GENETIC COUNSELLING: AR; 25% sibling risk"
        ),
        "treatment": (
            "SCAN1 MANAGEMENT: "
            "ATAXIA: "
            "  Physiotherapy: cerebellar rehabilitation; balance; gait; "
            "  Walking aids → rollator → wheelchair (progression); "
            "  Hydrotherapy; "
            "NEUROPATHY (PROMINENT): "
            "  Foot-drop: AFOs (ankle-foot orthoses); "
            "  Neuropathic pain: gabapentin 300-900 mg TID; pregabalin 75-300 mg BD; "
            "  Tendon reflexes absent: prevent falls; "
            "DYSARTHRIA: "
            "  Speech therapy; AAC (augmentative/alternative communication); "
            "CAMPTOTHECIN PHARMACOGENOMICS: "
            "  IRINOTECAN: ABSOLUTELY CONTRAINDICATED (TDP1 deficiency → lethal TOP1cc accumulation); "
            "  TOPOTECAN: ABSOLUTELY CONTRAINDICATED; "
            "  Other TOP1-inhibitors: ABSOLUTELY CONTRAINDICATED; "
            "  Medical alert / oncology flag: 'TDP1/SCAN1 — TOP1-INHIBITORS LETHAL'; "
            "  If cancer: use alternative non-TOP1-inhibitor regimens; "
            "PSYCHOLOGICAL SUPPORT: "
            "  Neuropsychological assessment; counselling; "
            "GENETIC COUNSELLING: AR; 25% recurrence"
        ),
    },
]


def _make_patients(gene_data: dict) -> list:
    """Generate 40 deterministic patients per gene for NHEJ/SSBR phenotypes."""
    rng = random.Random(gene_data["seed_base"])
    gene = gene_data["gene"]
    patients = []

    # Gene-specific prevalence parameters
    params = {
        "LIG4": dict(
            scid=0.92, immunodeficiency=0.95, radiosensitivity=0.98,
            microcephaly=0.88, growth_retardation=0.85, pancytopenia=0.82,
            lymphoma=0.28, ataxia=0.10, seizures=0.18,
            peripheral_neuropathy=0.05, oculomotor_apraxia=0.05,
            hypoalbuminemia=0.08, camptothecin_sensitivity=0.02,
            hsct_performed=0.72, radiation_avoidance=0.98,
        ),
        "DCLRE1C": dict(
            scid=0.95, immunodeficiency=0.97, radiosensitivity=0.92,
            microcephaly=0.12, growth_retardation=0.20, pancytopenia=0.08,
            lymphoma=0.05, ataxia=0.05, seizures=0.08,
            peripheral_neuropathy=0.05, oculomotor_apraxia=0.05,
            hypoalbuminemia=0.05, camptothecin_sensitivity=0.02,
            hsct_performed=0.88, radiation_avoidance=0.98,
        ),
        "PRKDC": dict(
            scid=0.90, immunodeficiency=0.93, radiosensitivity=0.80,
            microcephaly=0.15, growth_retardation=0.18, pancytopenia=0.10,
            lymphoma=0.05, ataxia=0.05, seizures=0.08,
            peripheral_neuropathy=0.05, oculomotor_apraxia=0.03,
            hypoalbuminemia=0.05, camptothecin_sensitivity=0.02,
            hsct_performed=0.80, radiation_avoidance=0.95,
        ),
        "XRCC4": dict(
            scid=0.02, immunodeficiency=0.05, radiosensitivity=0.72,
            microcephaly=0.95, growth_retardation=0.92, pancytopenia=0.05,
            lymphoma=0.08, ataxia=0.08, seizures=0.25,
            peripheral_neuropathy=0.10, oculomotor_apraxia=0.05,
            hypoalbuminemia=0.05, camptothecin_sensitivity=0.02,
            hsct_performed=0.02, radiation_avoidance=0.85,
        ),
        "NHEJ1": dict(
            scid=0.80, immunodeficiency=0.85, radiosensitivity=0.65,
            microcephaly=0.55, growth_retardation=0.60, pancytopenia=0.05,
            lymphoma=0.08, ataxia=0.08, seizures=0.12,
            peripheral_neuropathy=0.08, oculomotor_apraxia=0.05,
            hypoalbuminemia=0.05, camptothecin_sensitivity=0.02,
            hsct_performed=0.65, radiation_avoidance=0.85,
        ),
        "PNKP": dict(
            scid=0.02, immunodeficiency=0.03, radiosensitivity=0.30,
            microcephaly=0.82, growth_retardation=0.60, pancytopenia=0.03,
            lymphoma=0.03, ataxia=0.50, seizures=0.85,
            peripheral_neuropathy=0.45, oculomotor_apraxia=0.38,
            hypoalbuminemia=0.10, camptothecin_sensitivity=0.05,
            hsct_performed=0.02, radiation_avoidance=0.40,
        ),
        "APTX": dict(
            scid=0.01, immunodeficiency=0.02, radiosensitivity=0.05,
            microcephaly=0.08, growth_retardation=0.10, pancytopenia=0.02,
            lymphoma=0.02, ataxia=0.98, seizures=0.08,
            peripheral_neuropathy=0.92, oculomotor_apraxia=0.90,
            hypoalbuminemia=0.72, camptothecin_sensitivity=0.02,
            hsct_performed=0.01, radiation_avoidance=0.05,
        ),
        "TDP1": dict(
            scid=0.01, immunodeficiency=0.02, radiosensitivity=0.05,
            microcephaly=0.05, growth_retardation=0.08, pancytopenia=0.02,
            lymphoma=0.02, ataxia=0.98, seizures=0.10,
            peripheral_neuropathy=0.95, oculomotor_apraxia=0.08,
            hypoalbuminemia=0.05, camptothecin_sensitivity=0.85,
            hsct_performed=0.01, radiation_avoidance=0.05,
        ),
    }

    p = params[gene]
    age_range = {
        "LIG4": (1, 18), "DCLRE1C": (0, 12), "PRKDC": (0, 10),
        "XRCC4": (1, 25), "NHEJ1": (0, 15), "PNKP": (0, 20),
        "APTX": (5, 35), "TDP1": (10, 40),
    }
    a_min, a_max = age_range[gene]

    for i in range(40):
        sex = rng.choice(["M", "F"])
        age = rng.randint(a_min, a_max)
        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "age": age,
            "sex": sex,
            "scid": rng.random() < p["scid"],
            "immunodeficiency": rng.random() < p["immunodeficiency"],
            "radiosensitivity": rng.random() < p["radiosensitivity"],
            "microcephaly": rng.random() < p["microcephaly"],
            "growth_retardation": rng.random() < p["growth_retardation"],
            "pancytopenia": rng.random() < p["pancytopenia"],
            "lymphoma": rng.random() < p["lymphoma"],
            "ataxia": rng.random() < p["ataxia"],
            "seizures": rng.random() < p["seizures"],
            "peripheral_neuropathy": rng.random() < p["peripheral_neuropathy"],
            "oculomotor_apraxia": rng.random() < p["oculomotor_apraxia"],
            "hypoalbuminemia": rng.random() < p["hypoalbuminemia"],
            "camptothecin_sensitivity": rng.random() < p["camptothecin_sensitivity"],
            "hsct_performed": rng.random() < p["hsct_performed"],
            "radiation_avoidance": rng.random() < p["radiation_avoidance"],
        })

    return patients


def generate_overview():
    summary = []
    for gd in ATLAS_GENES:
        patients = _make_patients(gd)
        n = len(patients)
        pct = lambda k: round(100 * sum(p[k] for p in patients) / n)
        summary.append({
            "gene": gd["gene"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance": gd["inheritance"][:200],
            "pct_scid": pct("scid"),
            "pct_immunodeficiency": pct("immunodeficiency"),
            "pct_radiosensitivity": pct("radiosensitivity"),
            "pct_microcephaly": pct("microcephaly"),
            "pct_ataxia": pct("ataxia"),
            "pct_seizures": pct("seizures"),
            "pct_neuropathy": pct("peripheral_neuropathy"),
            "pct_oculomotor_apraxia": pct("oculomotor_apraxia"),
            "pct_hypoalbuminemia": pct("hypoalbuminemia"),
            "pct_hsct": pct("hsct_performed"),
            "n_patients": n,
        })

    return {
        "total_patients": 320,
        "seeds": "2750-2757",
        "summary": summary,
        "pathway_categories": [
            {
                "pathway": "Classical NHEJ — Core Ligation Complex",
                "genes": ["LIG4", "XRCC4", "NHEJ1"],
                "note": (
                    "LIG4 performs final nick-sealing (ATP-dependent ligation); "
                    "XRCC4 stabilises LIG4 and forms filament with XLF; "
                    "NHEJ1/XLF stimulates LIG4 ligation of incompatible ends; "
                    "All three: biallelic LOF → RS-SCID (LIG4/NHEJ1) or primordial dwarfism (XRCC4); "
                    "Common: radiosensitivity + elevated chromosomal aberrations post-irradiation"
                ),
            },
            {
                "pathway": "Classical NHEJ — End Detection & Processing",
                "genes": ["DCLRE1C", "PRKDC"],
                "note": (
                    "DNA-PKcs (PRKDC) is the central NHEJ scaffold kinase — activates Artemis, autophosphorylates for end release; "
                    "Artemis (DCLRE1C) opens V(D)J hairpins and trims overhangs — activated by DNA-PKcs phosphorylation; "
                    "Both: biallelic LOF → T-B-NK+ SCID (hairpin not opened → V(D)J fails); "
                    "DCLRE1C: Navajo/Athabascan founder alleles; Omenn syndrome hypomorphic variant; "
                    "PRKDC: ultra-rare (<30 cases); Saudi/Pakistani consanguineous; "
                    "TBI + cyclophosphamide: ABSOLUTELY CONTRAINDICATED in both"
                ),
            },
            {
                "pathway": "SSBR End-Processing — Polynucleotide Kinase/Phosphatase",
                "genes": ["PNKP"],
                "note": (
                    "PNKP bifunctional: 5'-kinase restores 5'-P; 3'-phosphatase removes 3'-P/3'-PG; "
                    "Both activities required for SSBR nick ligation (LIG3 requires 3'-OH + 5'-P); "
                    "Recruited to SSBR by CK2-pXRCC1 via FHA domain; to NHEJ by pXRCC4; "
                    "Biallelic LOF: MCSZ (Microcephaly-Seizures-Developmental delay) OR AOA4 (ataxia+OMA+neuropathy); "
                    "Allele-type determines phenotype: missense hypomorphic = MCSZ; compound = AOA4; "
                    "No immunodeficiency in either PNKP phenotype"
                ),
            },
            {
                "pathway": "SSBR Dead-End Resolution — Abortive Ligation / TOP1 Adduct",
                "genes": ["APTX", "TDP1"],
                "note": (
                    "APTX (aprataxin): resolves 5'-adenylate dead-ends from abortive ligation; "
                    "  FHA domain recruits via CK2-pXRCC1; HIT-Zn-finger hydrolyses 5'-AMP; "
                    "  Biallelic LOF → AOA1 (ataxia + oculomotor apraxia + hypoalbuminaemia + NORMAL AFP); "
                    "TDP1 (tyrosyl-DNA phosphodiesterase 1): resolves 3'-phosphotyrosyl (TOP1-cleavage complex); "
                    "  HxK motifs hydrolyse 3'-phosphotyrosyl → 3'-phosphate (then PNKP removes 3'-P); "
                    "  Biallelic LOF → SCAN1 (ataxia + neuropathy + NO oculomotor apraxia + NORMAL albumin); "
                    "  Camptothecin/irinotecan/topotecan: ABSOLUTELY CONTRAINDICATED in TDP1 deficiency"
                ),
            },
        ],
        "critical_distinctions": [
            "LIG4 vs ARTEMIS vs DNA-PKcs vs XLF — ALL T-B-NK+ SCID + radiosensitivity: LIG4 ADDITIONALLY has microcephaly + pancytopenia + lymphoma risk; Artemis has Navajo founder alleles + Omenn phenotype; DNA-PKcs is ultra-rare (<30 cases); XLF is mildest (variable B cells); distinguish by gene sequencing",
            "XRCC4 vs LIG4 — BOTH microcephaly + dwarfism + radiosensitivity: XRCC4 has NO immunodeficiency (T-B-NK NORMAL); LIG4 has T-B-NK+ SCID; first step = CBC + lymphocyte subsets; if normal immune → XRCC4 more likely; if T-B-NK+ → LIG4/Artemis/DNA-PKcs/XLF",
            "TBI ABSOLUTELY CONTRAINDICATED in all NHEJ-SCID: LIG4 + Artemis + DNA-PKcs + XLF: standard-dose TBI causes fatal aplasia; HSCT conditioning MUST use reduced-intensity (fludarabine-based) — this kills patients if standard conditioning given",
            "CYCLOPHOSPHAMIDE ABSOLUTELY CONTRAINDICATED in Artemis-SCID: crosslinks → complex DSBs that Artemis resolves; Artemis LOF → persistent crosslink-DSB → lethal bone marrow failure; LIG4 also very sensitive; avoid in all NHEJ SCID",
            "AOA1 (APTX) vs SCAN1 (TDP1) — KEY DDx: AOA1 has oculomotor apraxia + hypoalbuminaemia + hypercholesterolaemia; SCAN1 has NO OMA + NORMAL albumin + camptothecin sensitivity; both normal AFP; both axonal neuropathy + cerebellar ataxia; different gene/mechanism",
            "AOA1 (APTX) vs A-T (ATM): both cerebellar ataxia + oculomotor apraxia; A-T has ELEVATED AFP + telangiectasias + immunodeficiency + cancer risk; AOA1 has NORMAL AFP + hypoalbuminaemia + NO telangiectasias + NO significant cancer risk",
            "SCAN1/TDP1 — CAMPTOTHECIN ABSOLUTE CI: irinotecan, topotecan, SN-38 trap TOP1-cleavage complexes → in TDP1 deficiency → permanent adducts → lethal; essential oncology alert if TDP1 patient ever develops cancer needing chemotherapy",
            "MCSZ (PNKP) vs AOA4 (PNKP): SAME GENE, DIFFERENT PHENOTYPE: missense hypomorphic PNKP alleles → MCSZ (infantile seizures + microcephaly dominant); compound truncating+missense → AOA4 (ataxia dominant); allele determines clinical outcome",
            "APTX: AFP NORMAL (key DDx from A-T and AOA2/SETX): APTX does not activate ATM signalling like ATM-null does; AFP elevation is ATM-dependent; APTX deficiency bypasses ATM-AFP axis; always check AFP first in cerebellar ataxia — normal AFP excludes A-T/AOA2 but not AOA1/SCAN1",
            "NO RADIOSENSITIVITY CAUTION for APTX and TDP1: APTX (AOA1) and TDP1 (SCAN1) do NOT have clinically significant radiosensitivity (SSBR/TOP1cc resolution, not DSB via NHEJ); LIG4/Artemis/DNA-PKcs/XLF/XRCC4/PNKP DO have elevated chromosomal aberrations; tailor radiation precautions accordingly",
        ],
    }


def generate_breakdown():
    genes_out = []
    for gd in ATLAS_GENES:
        patients = _make_patients(gd)
        n = len(patients)
        pct = lambda k: round(100 * sum(p[k] for p in patients) / n)
        genes_out.append({
            "gene": gd["gene"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "n_patients": n,
            "inheritance": gd["inheritance"],
            "disease_category": gd["disease_category"],
            "disease_pathway": gd["disease_pathway"],
            "pathognomonic": gd["pathognomonic"],
            "treatment": gd["treatment"],
            "pct_scid": pct("scid"),
            "pct_immunodeficiency": pct("immunodeficiency"),
            "pct_radiosensitivity": pct("radiosensitivity"),
            "pct_microcephaly": pct("microcephaly"),
            "pct_growth_retardation": pct("growth_retardation"),
            "pct_pancytopenia": pct("pancytopenia"),
            "pct_lymphoma": pct("lymphoma"),
            "pct_ataxia": pct("ataxia"),
            "pct_seizures": pct("seizures"),
            "pct_peripheral_neuropathy": pct("peripheral_neuropathy"),
            "pct_oculomotor_apraxia": pct("oculomotor_apraxia"),
            "pct_hypoalbuminemia": pct("hypoalbuminemia"),
            "pct_camptothecin_sensitivity": pct("camptothecin_sensitivity"),
            "pct_hsct": pct("hsct_performed"),
            "patients": patients[:40],
        })
    return {"genes": genes_out}


def generate_definitions():
    return {
        "glossary": {
            "Non-Homologous End Joining (NHEJ)": (
                "Primary DSB repair in G1/G0 non-dividing cells; does not require template homology; "
                "Classical NHEJ (cNHEJ): Ku70/Ku80 → DNA-PKcs → Artemis (end-processing) → XRCC4/XLF/LIG4 (ligation); "
                "Alternative NHEJ (alt-NHEJ/MMEJ): uses microhomology; Pol θ, LIG1/LIG3; error-prone; "
                "V(D)J recombination obligately requires cNHEJ: RAG1/2 generate hairpin DSBs → Ku/DNA-PKcs/Artemis open hairpin → "
                "  XRCC4/XLF/LIG4 ligate → diverse TCR/BCR repertoire; "
                "Biallelic LOF in any core NHEJ gene → T-B-NK+ SCID (V(D)J fails); "
                "NHEJ genes: XRCC5(Ku80)/XRCC6(Ku70)/PRKDC(DNA-PKcs)/DCLRE1C(Artemis)/XRCC4/NHEJ1(XLF)/LIG4/PAXX"
            ),
            "Single-Strand Break Repair (SSBR)": (
                "Repair of single-strand DNA breaks (nicks); rapid pathway: minutes to hours; "
                "Initiated by PARP1/PARP2 (binds SSBs, recruits XRCC1 scaffold); "
                "End-processing: PNKP (kinase/phosphatase), APTX (5'-adenylate removal), TDP1 (3'-phosphotyrosyl), APE1 (AP endonuclease); "
                "Fill-in: Pol β (short-patch), Pol δ/ε (long-patch); "
                "Ligation: LIG3/XRCC1 (short-patch), LIG1 (long-patch); "
                "If SSBR fails → SSBs accumulate → replication fork collapse → DSBs → NHEJ/HR required; "
                "SSBR genes: PARP1/PARP2/XRCC1/PNKP/APTX/TDP1/LIG3/APE1/Pol β"
            ),
            "V(D)J Recombination": (
                "Mechanism generating antigen receptor diversity (TCR + BCR); "
                "RAG1/RAG2: recognise RSS (recombination signal sequences); cut → hairpin coding ends + blunt signal ends; "
                "Hairpin opening: DNA-PKcs-activated Artemis (DCLRE1C); "
                "P-nucleotides: palindromic additions from opened hairpin; "
                "TdT: adds N-nucleotides (junctional diversity); "
                "XRCC4/XLF/LIG4: ligate processed coding ends; "
                "Result: diverse CDR3 regions of TCR/BCR → antigen specificity; "
                "Failure of ANY step → absent T or B cells → SCID; "
                "V(D)J fidelity monitored by TRECs (T-cell receptor excision circles) on NBS"
            ),
            "Radiosensitivity — NHEJ Defects": (
                "NHEJ repairs ionising radiation-induced DSBs (most common DSB source: direct strand breaks + ROS); "
                "NHEJ genes LOF → radiation-induced DSBs unrepaired → chromosomal instability; "
                "Clinical relevance — FATAL RISK: standard radiotherapy doses (1.8-2 Gy/fraction, 50-60 Gy total) → "
                "  LETHAL aplasia in LIG4 Syndrome, Artemis-SCID, XLF SCID, DNA-PKcs SCID; "
                "Chromosomal breakage assay: gold standard radiosensitivity test; "
                "  Dicentrics + ring chromosomes at 2 Gy irradiation markedly elevated in NHEJ defects; "
                "Order of severity: LIG4 > Artemis > DNA-PKcs ≈ XLF > XRCC4 > PNKP > APTX ≈ TDP1; "
                "TBI = Total Body Irradiation: component of myeloablative HSCT conditioning — ABSOLUTELY CONTRAINDICATED"
            ),
            "LIG4 Syndrome": (
                "Autosomal recessive disorder caused by hypomorphic biallelic LIG4 LOF; "
                "Clinical triad: T-B-NK+ SCID + microcephaly + pancytopenia; "
                "Complete LIG4 null = lethal in utero; only hypomorphic alleles survive; "
                "Lymphoma risk: B-cell NHL in survivors (NHEJ defect → genomic instability in lymphoid cells); "
                "Skin: malar rash, telangiectasias; "
                "Most common allele: c.2440C>T (p.Arg814Cys) — Turkish homozygous consanguineous; "
                "Treatment: HSCT with reduced-intensity conditioning; TBI ABSOLUTELY CONTRAINDICATED; "
                "Outcome: excellent if appropriate conditioning used"
            ),
            "Artemis-SCID / RS-SCID": (
                "Autosomal recessive SCID caused by biallelic DCLRE1C LOF; "
                "Phenotype: T-B-NK+ SCID; NO microcephaly / NO pancytopenia (DDx from LIG4 Syndrome); "
                "Radiosensitivity: elevated (milder than LIG4); "
                "Founder: Navajo/Athabascan Native American (c.1090delGCACC + c.3031del haplotype; 2% carrier); "
                "Omenn phenotype: hypomorphic alleles → partial hairpin opening → oligoclonal T cells + erythroderma; "
                "Treatment: HSCT-RIC; TBI ABSOLUTELY CONTRAINDICATED; "
                "Cyclophosphamide: ABSOLUTELY CONTRAINDICATED (crosslink-DSBs unresolvable without Artemis); "
                "Gene therapy: investigational Phase I/II trials 2024"
            ),
            "XRCC4 Deficiency": (
                "Autosomal recessive disorder caused by hypomorphic biallelic XRCC4 LOF; "
                "Clinical: primordial microcephalic dwarfism + intellectual disability; "
                "CRITICAL: NO IMMUNODEFICIENCY — T-B-NK cells NORMAL (distinguishes from LIG4 syndrome); "
                "XRCC4 null = lethal in mice (unlike xlf-/- which is viable); "
                "Mechanism: XRCC4 destabilised → LIG4 not recruited → DSBs accumulate in neuroprogenitors → microcephaly; "
                "Radiosensitivity: present but not immunologically catastrophic; "
                "Treatment: supportive (neurodevelopmental); NO HSCT; radiation avoidance"
            ),
            "AOA1 / Aprataxin Deficiency": (
                "Autosomal recessive ataxia caused by biallelic APTX LOF; "
                "Clinical: cerebellar ataxia (onset 2-10y) + oculomotor apraxia + axonal neuropathy + hypoalbuminaemia; "
                "AFP: NORMAL (critical DDx from A-T and AOA2 — both elevated AFP); "
                "Albumin: LOW — characteristic of AOA1 (SCAN1 albumin normal); "
                "Mechanism: 5'-adenylate dead-ends accumulate → Purkinje cells degenerate (high TOP1/LIG usage); "
                "Portuguese founder: c.837_838del; Japanese: W279X; "
                "No cancer risk elevation; no radiosensitivity precautions needed; "
                "Treatment: supportive; physiotherapy + OT; statin for hypercholesterolaemia"
            ),
            "SCAN1 / TDP1 Deficiency": (
                "Autosomal recessive ataxia caused by biallelic TDP1 LOF; "
                "Ultra-rare: Saudi Arabian founder (His493Arg); very few cases globally; "
                "Clinical: cerebellar ataxia (onset 10-14y) + prominent axonal neuropathy; "
                "KEY: NO oculomotor apraxia (DDx from AOA1); albumin NORMAL (DDx from AOA1); AFP NORMAL; "
                "Mechanism: TOP1-cleavage complexes not resolved → 3'-phosphotyrosyl dead-end persists → neuron loss; "
                "Camptothecin sensitivity: ABSOLUTELY CONTRAINDICATED in TDP1 deficiency (irinotecan/topotecan); "
                "Treatment: supportive; avoid TOP1-inhibitor chemotherapy"
            ),
            "MCSZ / AOA4 (PNKP Deficiency)": (
                "Autosomal recessive disorder with two phenotypes depending on allele; "
                "MCSZ: Microcephaly + Seizures + developmental delay — missense hypomorphic alleles; "
                "AOA4: Ataxia + Oculomotor apraxia + neuropathy + mildly elevated AFP — compound alleles; "
                "Bifunctional PNKP: 5'-kinase + 3'-phosphatase — both required for ligation; "
                "SSBR + NHEJ role: recruited by XRCC1-CK2 (SSBR) and XRCC4-pSer325 (NHEJ); "
                "No immunodeficiency; mild radiosensitivity; "
                "Seizure management: LEV first-line; KD for drug-resistant"
            ),
            "Chromosomal Breakage Assay (Radiosensitivity)": (
                "Gold-standard test for NHEJ radiosensitivity; "
                "Method: patient lymphoblastoid cells (or fibroblasts) irradiated with gamma-radiation (2 Gy); "
                "Karyotype 48h post-irradiation: count dicentrics + rings + translocations; "
                "Normal: <0.5 aberrations/cell at 2 Gy; "
                "NHEJ defects: 2-10 aberrations/cell (LIG4 severely elevated; Artemis intermediate; XLF moderate); "
                "Clinical use: pre-HSCT to predict conditioning toxicity; "
                "Also: G2/M assay (G2 chromosomal aberrations post-irradiation) — different readout; "
                "Neither test reliably distinguishes which NHEJ gene without sequencing"
            ),
            "Reduced-Intensity Conditioning (RIC) for NHEJ SCID": (
                "HSCT conditioning strategy MANDATORY for all NHEJ-SCID patients; "
                "Standard myeloablative conditioning (TBI 12 Gy + cyclophosphamide 200 mg/kg): ABSOLUTELY CONTRAINDICATED; "
                "NHEJ LOF → DSBs from TBI/cyclophosphamide cannot be repaired → fatal aplasia/organ toxicity; "
                "Preferred RIC backbone: fludarabine 150-180 mg/m² + serotherapy (alemtuzumab or ATG); "
                "Optional additions: reduced busulfan (pharmacokinetically guided, area-under-curve ≤60 mg·h/L); "
                "Treosulfan-based: alternative reduced DNA-alkylation agent; "
                "Goal: sufficient immunosuppression for engraftment; minimal genotoxic stress; "
                "Outcome: >85-90% survival in LIG4 Syndrome and Artemis-SCID with correct RIC"
            ),
            "Cascade Testing in NHEJ/SSBR Genes": (
                "All eight genes in this atlas are AUTOSOMAL RECESSIVE (biallelic LOF causes disease); "
                "Proband's siblings: 25% risk of biallelic (obligate carrier parents); "
                "Parents: carriers — no phenotype; "
                "NHEJ SCID (LIG4/Artemis/DNA-PKcs/XLF): URGENCY — SCID detected by TREC on NBS → "
                "  immediate referral to immunology; gene panel within days; "
                "XRCC4/PNKP/APTX/TDP1: germline testing for index case; siblings at 25% risk; "
                "  less urgent (non-SCID) but counsel for radiosensitivity/drug sensitivities; "
                "Partner testing: IMPORTANT for SCID families — if partner is MUTYH/DCLRE1C/LIG4 carrier, "
                "  risk to offspring doubles from carrier × carrier coupling"
            ),
        },
        "standards": [
            "O'Driscoll M et al. Nature Genetics 2001: LIG4 Syndrome — first description of biallelic LIG4 mutations",
            "Moshous D et al. Cell 2001: Artemis mutations cause RS-SCID — first Artemis discovery",
            "van der Burg M et al. JEM 2009: DNA-PKcs deficiency causes SCID — first human cases",
            "Gennery AR et al. JCI 2004: XRCC4 mutations in primordial dwarfism without immunodeficiency",
            "Buck D et al. Cell 2006: Cernunnos/XLF deficiency causes immunodeficiency and radiosensitivity",
            "Shen J et al. Nature Genetics 2010: PNKP mutations in MCSZ — microcephaly + seizures",
            "Moreira MC et al. Nature Genetics 2001: Aprataxin mutations in AOA1 — first AOA1 discovery",
            "Takashima H et al. Nature Genetics 2002: TDP1 mutations in SCAN1 — first SCAN1 discovery",
            "Riballo E et al. Mol Cell 2004: Artemis-DNA-PKcs hairpin-opening mechanism — biochemical characterization",
            "Lieber MR. Annual Review of Biochemistry 2010: Classical NHEJ pathway mechanism — comprehensive review",
            "Caldecott KW. Nature Reviews Genetics 2008: SSBR mechanisms — comprehensive review",
            "ESID/EBMT Working Party 2022: Guidance on HSCT for NHEJ SCID — reduced-intensity conditioning",
            "NCCN/AAAAI Primary Immunodeficiency Guidelines 2024: SCID workup and treatment",
            "La Peyre de Bellaire T et al. JCO 2024: Outcomes of HSCT for Artemis-SCID with RIC conditioning",
            "Takahashi T et al. Brain 2022: PNKP-related AOA4 — clinical delineation and AFP findings",
        ],
    }
