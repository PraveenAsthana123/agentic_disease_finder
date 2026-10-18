#!/usr/bin/env python3
"""Hereditary-SCID-RAG-IL2RG-ADA-Atlas -- Complete 8-Gene Reference
Classic Severe Combined Immunodeficiency (SCID) Genes
(IL2RG/X-SCID-Most-Common-XL-Gamma-c-Chain-T-B-NK--SCID-PCP-Infant-Boy-PATHOGNOMONIC,
 JAK3/T-B-NK--SCID-AR-Phenocopy-XL-SCID-JAK3-Western-Blot-Distinguishes,
 RAG1/T-B-NK+-SCID-or-Omenn-RAG1-Null-Complete-SCID-Missense-Omenn-Spectrum,
 RAG2/T-B-NK+-SCID-or-Omenn-RAG2-Null-Complete-SCID-11p13-Cotesting-RAG1,
 DCLRE1C/Artemis-T-B-NK+-Radiosensitive-SCID-TBI-ABSOLUTELY-CI-Reduced-Intensity-Conditioning,
 ADA/Metabolic-SCID-dATP-Toxic-PEG-ADA-Enzyme-Replacement-LIFE-SAVING-Gene-Therapy-Approved,
 IL7R/T-B+NK+-SCID-IL7-Receptor-Alpha-AR-T-Cell-Development-Block-Thymus-Empty,
 AK2/Reticular-Dysgenesis-Most-Severe-SCID-SCID+Agranulocytosis-AK2-ABSOLUTE-HSCT-Emergency):

IL2RG  (gamma-c chain; 256aa; Xq13.1; XLR;
         Common gamma chain shared by IL-2, IL-4, IL-7, IL-9, IL-15, IL-21 receptors;
         IL2RG LOF → absent γc signalling → block at every lymphocyte developmental stage;
         Immunophenotype: T-B-NK- SCID (T absent, B present but non-functional, NK absent);
         Most common SCID: ~50% of all SCID in males; 1:50,000–75,000 live births;
         X-linked recessive: virtually all affected patients are male;
         PATHOGNOMONIC presentation: PCP in an infant boy in first 6 months of life = X-SCID until proven otherwise;
         Treatment: HSCT curative; gene therapy approved (Strimvelis® EU 2016, gene-edited T cells USA 2024);
         BCG absolutely contraindicated;
         seed SEED_BASE+0).
JAK3   (Janus kinase 3; 1124aa; 19p13.11; AR;
         JAK3 is the intracellular kinase that directly binds γc chain and transduces IL-2/-4/-7/-9/-15/-21 signals;
         JAK3 LOF → phenotypically IDENTICAL to IL2RG deficiency: T-B-NK- SCID;
         KEY DISTINCTION: JAK3 is AUTOSOMAL RECESSIVE — both sexes equally affected (vs IL2RG X-linked males only);
         JAK3 absent by Western blot; JAK3 immunostaining on T cells = 0 (flow-based assay);
         Affects females → X-SCID excluded → JAK3 sequencing next in AR T-B-NK- SCID;
         Treatment: HSCT (same approach as IL2RG); no approved gene therapy yet;
         BCG absolutely contraindicated;
         seed SEED_BASE+1).
RAG1   (Recombination-activating gene 1; 1043aa; 11p13; AR;
         RAG1 forms the RAG complex with RAG2 — catalyses V(D)J recombination of TCR and immunoglobulin genes;
         RAG1 LOF (null) → absent V(D)J recombination → T-B-NK+ SCID (NK cells present, T/B absent);
         RAG1 hypomorphic → Omenn syndrome: oligoclonal T cells, hypereosinophilia, erythroderma, hepatosplenomegaly;
         PATHOGNOMONIC distinction: NK CELLS PRESENT (unlike IL2RG/JAK3 where NK also absent);
         Omenn syndrome (hypomorphic) can be misdiagnosed as GvHD or histiocytosis → RAG panel mandatory;
         RAG1/RAG2 cotesting mandatory (adjacent genes at 11p13 — same sequencing panel);
         HSCT curative; Omenn requires immunosuppression (steroid/CSA) before HSCT;
         seed SEED_BASE+2).
RAG2   (Recombination-activating gene 2; 527aa; 11p13; AR;
         RAG2 is the catalytic partner of RAG1 — they form the RAG1/2 complex for V(D)J recombination;
         RAG2 LOF (null) → T-B-NK+ SCID (identical phenotype to RAG1 null);
         RAG2 hypomorphic → Omenn syndrome (as with RAG1 hypomorphic);
         Adjacent to RAG1 on 11p13: RAG1+RAG2 are ALWAYS tested together (same chromosomal locus, same panel);
         Distinguishing RAG1 vs RAG2: only by sequencing, not phenotype;
         RAG1/RAG2 mutations account for ~30-35% of T-B-NK+ SCID;
         Late-onset hypomorphic variants: combined immunodeficiency with granulomas reported;
         HSCT curative;
         seed SEED_BASE+3).
DCLRE1C (DNA cross-link repair 1C (Artemis); 692aa; 10p13; AR;
         Artemis is a V(D)J recombination-specific endonuclease that opens hairpin intermediates;
         DCLRE1C LOF → T-B-NK+ SCID with DNA DOUBLE-STRAND BREAK REPAIR DEFECT;
         RADIOSENSITIVITY ABSOLUTE: Artemis required for NHEJ DNA repair → radiation kills cells with DCLRE1C LOF;
         TOTAL BODY IRRADIATION (TBI) ABSOLUTELY CONTRAINDICATED as HSCT conditioning;
         Reduced-intensity conditioning (RIC) WITHOUT TBI mandatory;
         Sensitive to alkylating agents (busulfan): lower doses required;
         Native American Athabascan SCID (NAID): p.Thr347Ile founder in Navajo/Apache populations;
         ~10% of T-B-NK+ SCID cases;
         HSCT (RIC without TBI) curative;
         seed SEED_BASE+4).
ADA    (Adenosine deaminase; 363aa; 20q13.11; AR;
         ADA catalyses deamination of adenosine → inosine and 2'-deoxyadenosine → 2'-deoxyinosine;
         ADA LOF → accumulation of dATP (toxic metabolite): dATP kills lymphocytes → combined SCID;
         Immunophenotype: T-B-NK- SCID (all 3 lineages affected by dATP toxicity);
         KEY FEATURE: toxic metabolite SCID — enzyme replacement possible (unlike other SCID genes);
         PEG-ADA enzyme replacement therapy (ERT): bridges patients to HSCT or gene therapy;
         Gene therapy APPROVED (Strimvelis® EU 2016 for ADA-SCID); first approved gene therapy ever;
         Calcifications: periarticular, renal, pulmonary — unique to ADA-SCID (skeletal survey at diagnosis);
         Neurological comorbidities (dATP neurotoxicity): sensorineural hearing loss 50%, cognitive effects;
         Bilateral sensorineural hearing loss in 50% = PATHOGNOMONIC clue in T-B-NK- SCID;
         seed SEED_BASE+5).
IL7R   (Interleukin-7 receptor subunit alpha; 459aa; 5p13.2; AR;
         IL-7Rα pairs with γc chain (IL2RG) to form the IL-7 receptor;
         IL-7 signalling is ESSENTIAL for T cell development in the thymus (not B or NK cells);
         IL7R LOF → block in T cell development → T-B+NK+ SCID (T cells absent; B cells PRESENT; NK cells PRESENT);
         KEY IMMUNOPHENOTYPE DISTINCTION: T-B+NK+ SCID (B and NK normal) vs IL2RG/JAK3 (T-B-NK-);
         Thymus: absent T cells but thymic shadow can appear normal (unlike IL2RG/JAK3 thymic atrophy);
         IL7R T-B+NK+ SCID: same immunophenotype as CD3 complex defects (CD3D/CD3E) BUT different mechanism;
         Flow cytometry: absent CD3 T cells BUT B cells and NK cells NORMAL (key distinction from γc chain defects);
         HSCT curative;
         seed SEED_BASE+6).
AK2    (Adenylate kinase 2; 239aa; 1p34.1; AR;
         AK2 is a mitochondrial enzyme that catalyses reversible transfer of phosphate between adenine nucleotides;
         AK2 LOF → Reticular Dysgenesis (RD): MOST SEVERE AND RAREST SCID;
         UNIQUE PHENOTYPE: SCID + AGRANULOCYTOSIS (profound neutropenia from birth);
         Neutropenia + T/B/NK lymphopenia = PATHOGNOMONIC for Reticular Dysgenesis;
         Hearing loss (bilateral sensorineural) in 60%: AK2 expressed in inner ear (stria vascularis);
         Death within weeks without HSCT — the only SCID where sepsis kills before opportunistic infection;
         Bone marrow: myeloid arrest (no neutrophils from myeloid side) + lymphoid arrest;
         URGENT HSCT in first weeks of life: bacterial sepsis is the acute threat (not PCP);
         AK2 patients need broad-spectrum antibiotics AND antifungals from birth;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3958-3965)
"""
import random

SEED_BASE = 3958

ATLAS_GENES = [
    {
        "gene": "IL2RG",
        "protein": (
            "IL2RG -- Xq13.1 X-Linked-Recessive-LOF -- 256aa -- "
            "Common-Gamma-Chain-IL-2-4-7-9-15-21-Receptor-29kDa-"
            "T-B-NK--SCID-Most-Common-50pct-Males-XL-PCP-Infant-Boy-PATHOGNOMONIC-"
            "HSCT-Gene-Therapy-Strimvelis-Approved-BCG-ABSOLUTELY-CI-OMIM-308380"
        ),
        "locus": "Xq13.1",
        "protein_size": (
            "256 aa / 29 kDa / IL2RG molecular context: "
            "STRUCTURE: "
            "  256 aa / 29 kDa; type I cytokine receptor; "
            "  Extracellular domain: cytokine-binding module; "
            "  Single TM helix; cytoplasmic tail: binds JAK3 (Box1/Box2 motifs); "
            "  γc is the COMMON signalling subunit shared by 6 cytokine receptors: "
            "    IL-2R (γc+βc+α), IL-4R (γc+α), IL-7R (γc+α=IL-7Rα), "
            "    IL-9R (γc+α), IL-15R (γc+βc+α), IL-21R (γc+α); "
            "FUNCTION: "
            "  γc chain assembles with cytokine-specific α chains → JAK3 (on γc) + JAK1 (on α) transphosphorylate; "
            "  → STAT5/3/6 activation → transcription; "
            "  IL-7 (via γc) = T cell development; IL-15 (via γc) = NK cell development; "
            "IMMUNOPHENOTYPE: "
            "  T cells: ABSENT (IL-7/γc block in thymic T cell development); "
            "  B cells: PRESENT but non-functional (B development IL-7-independent in mice, partial in humans); "
            "  NK cells: ABSENT (IL-15/γc required for NK development); "
            "  Phenotype: T-B+NK- SCID (B cells present but non-functional); "
            "  Note: often called T-B-NK- by clinical convention as B cells non-functional; "
            "EPIDEMIOLOGY: "
            "  Most common SCID: ~45-50% of all SCID; X-linked → virtually all males; "
            "  Incidence: 1:50,000–75,000 live births; "
            "  PATHOGNOMONIC: PCP pneumonia in an infant boy in first 6 months of life; "
            "TREATMENT: "
            "  HSCT: gold standard (genoidentical sibling > haploidentical > matched unrelated); "
            "  Gene therapy: APPROVED (Strimvelis® EU EMA 2016 for ADA-SCID; "
            "    IL2RG lentiviral gene therapy USA approved 2024 — Lyfgenia-related); "
            "    ex vivo autologous HSC gene correction; curative; avoids GvHD; "
            "  BCG absolutely contraindicated; "
        ),
        "syndrome": "X-linked SCID (X-SCID) — T-B+NK- SCID",
        "inheritance": "X-linked recessive (XLR)",
        "immunophenotype": "T-B+NK- SCID: T cells absent, B cells present (non-functional), NK absent",
        "t_cell_pct_mean": 0.3,
        "pathognomonic": "PCP pneumonia in infant BOY aged 2-6 months = X-SCID until proven otherwise. T-B+NK- phenotype (NK absent distinguishes from CD3 complex defects T-B+NK+).",
        "key_avoid": "NEVER give BCG; NEVER give live vaccines; NEVER give non-irradiated blood products; NEVER use TBI conditioning (prefer Busulfan-based RIC for IL2RG gene therapy); NEVER give IL-2 in GOF context",
        "key_rule": "X-SCID: PCP in infant boy = emergency. T-B+NK- SCID. MALES ONLY (X-linked). NK absent = key DDx from CD3 complex defects (T-B+NK+). Gene therapy APPROVED — consider before haploidentical HSCT if available.",
        "surveillance": [
            "TREC NBS at birth — TREC=0",
            "T/B/NK flow cytometry monthly (T absent, NK absent — distinguishes from IL7R T-B+NK+)",
            "CMV/EBV/RSV/PCP surveillance biweekly",
            "Immunoglobulin levels (B cells present but non-functional → low Ig)",
            "IL2RG sequencing; maternal carrier testing; prenatal diagnosis",
        ],
        "targeted_rx": "HSCT (genoidentical/haploidentical) curative; gene therapy approved (ex vivo lentiviral IL2RG); IVIG bridging until HSCT; PCP prophylaxis (co-trimoxazole); CMV prophylaxis (ganciclovir)",
    },
    {
        "gene": "JAK3",
        "protein": (
            "JAK3 -- 19p13.11 Autosomal-Recessive-LOF -- 1124aa -- "
            "Janus-Kinase-3-125kDa-FERM-SH2-JH2-JH1-Gamma-c-Chain-Signal-Transducer-"
            "T-B-NK--SCID-AR-Phenocopy-XL-SCID-Both-Sexes-JAK3-Western-Blot-Diagnostic-"
            "HSCT-Curative-BCG-ABSOLUTELY-CI-OMIM-600173"
        ),
        "locus": "19p13.11",
        "protein_size": (
            "1124 aa / 125 kDa / JAK3 molecular context: "
            "STRUCTURE: "
            "  1124 aa / 125 kDa; non-receptor tyrosine kinase; Janus kinase family; "
            "  FERM domain (N-terminal, aa 1-350): membrane association; "
            "  SH2-like domain; "
            "  Pseudokinase domain JH2 (aa 589-800): regulatory; "
            "  Kinase domain JH1 (aa 812-1124): catalytic; activation loop Tyr980/Tyr981; "
            "JAK3 BINDS EXCLUSIVELY TO γc CHAIN (IL2RG): "
            "  JAK3 is uniquely expressed in lymphocytes and binds only γc cytoplasmic tail; "
            "  JAK1 binds the cytokine-specific α chains; JAK3+JAK1 transphosphorylate → STAT5; "
            "IMMUNOPHENOTYPE: "
            "  T-B+NK- SCID: PHENOTYPICALLY IDENTICAL to IL2RG deficiency; "
            "  AUTOSOMAL RECESSIVE: affects BOTH MALES AND FEMALES equally; "
            "KEY DISTINCTION: "
            "  T-B+NK- SCID in a FEMALE = JAK3 (not IL2RG) until proven otherwise; "
            "  T-B+NK- SCID in male: first exclude IL2RG (sequence Xq13.1), then JAK3 (19p13.11); "
            "DIAGNOSIS: "
            "  JAK3 protein absent/reduced by Western blot (if antibody available); "
            "  Flow-based JAK3 intracellular staining in T/NK cells; "
            "  Genetic sequencing confirms; "
            "TREATMENT: "
            "  HSCT curative; no approved gene therapy yet for JAK3; "
            "  JAK inhibitors (tofacitinib/ruxolitinib) investigated as bridge (GOF context); "
        ),
        "syndrome": "JAK3 deficiency — T-B+NK- SCID (autosomal recessive IL2RG phenocopy)",
        "inheritance": "Autosomal recessive (AR)",
        "immunophenotype": "T-B+NK- SCID: T cells absent, B present (non-functional), NK absent",
        "t_cell_pct_mean": 0.4,
        "pathognomonic": "T-B+NK- SCID in a FEMALE = JAK3 (autosomal) NOT IL2RG (X-linked). Phenotype identical to X-SCID. JAK3 Western blot absent. Both sexes affected equally.",
        "key_avoid": "NEVER give BCG; NEVER give live vaccines; NEVER give non-irradiated blood; NEVER assume X-SCID without genetics in female T-B+NK- SCID; NEVER give JAK inhibitors without genetic confirmation of LOF vs GOF",
        "key_rule": "JAK3 = AR phenocopy of X-SCID. Female with T-B+NK- SCID → sequence JAK3 before IL2RG. JAK3 Western blot absent = diagnostic. Same clinical urgency as X-SCID. HSCT same approach.",
        "surveillance": [
            "TREC NBS = 0",
            "T/B/NK flow (T absent, NK absent; distinguish from IL7R T-B+NK+)",
            "JAK3 Western blot + flow intracellular staining",
            "CMV/EBV/RSV surveillance biweekly",
            "JAK3 genetic sequencing; sibling carrier testing",
        ],
        "targeted_rx": "HSCT curative; IVIG bridging; PCP prophylaxis; CMV prophylaxis; no approved gene therapy (investigational lentiviral JAK3)",
    },
    {
        "gene": "RAG1",
        "protein": (
            "RAG1 -- 11p13 Autosomal-Recessive-LOF -- 1043aa -- "
            "Recombination-Activating-Gene-1-119kDa-V-D-J-Recombination-Endonuclease-"
            "T-B-NK+-SCID-Null-or-Omenn-Syndrome-Hypomorphic-NK-PRESENT-PATHOGNOMONIC-"
            "RAG1-RAG2-Cotesting-Mandatory-11p13-BCG-ABSOLUTELY-CI-OMIM-179615"
        ),
        "locus": "11p13",
        "protein_size": (
            "1043 aa / 119 kDa / RAG1 molecular context: "
            "STRUCTURE: "
            "  1043 aa / 119 kDa; RING-finger domain (E3 ubiquitin ligase activity, N-terminal); "
            "  Non-core region (aa 1-383): regulatory, not required for recombination; "
            "  Core RAG1 (aa 384-1040): catalytically active; "
            "  RING domain; NBD (nonamer binding domain); central domain; CTD (C-terminal domain) with DDE triad; "
            "  DDE triad (Asp600, Asp708, Glu962): catalytic active site for DNA cleavage; "
            "  Pairs with RAG2 to form the RAG complex (2×RAG1 + 2×RAG2 tetramer); "
            "V(D)J RECOMBINATION MECHANISM: "
            "  RAG complex recognises RSS (recombination signal sequences) flanking V, D, J gene segments; "
            "  RAG1 introduces double-strand breaks → hairpin intermediates; "
            "  Artemis (DCLRE1C) opens hairpins → NHEJ ligates → diverse receptor genes; "
            "IMMUNOPHENOTYPE: "
            "  NULL alleles: T-B-NK+ SCID (NK cells present — V(D)J-independent); "
            "  HYPOMORPHIC alleles: Omenn syndrome (oligoclonal T cells, eosinophilia, erythroderma, HSM); "
            "  LATE-ONSET hypomorphic: combined ID with granulomata, lymphocytopenia; "
            "KEY RULE: "
            "  RAG1/RAG2 always tested together (adjacent genes at 11p13); "
            "  NK PRESENT = V(D)J defect (RAG1/RAG2/DCLRE1C); NK ABSENT = γc chain defect (IL2RG/JAK3); "
        ),
        "syndrome": "RAG1 deficiency — T-B-NK+ SCID or Omenn syndrome (hypomorphic)",
        "inheritance": "Autosomal recessive (AR)",
        "immunophenotype": "T-B-NK+ SCID: T absent, B absent, NK PRESENT; or Omenn (hypomorphic: oligoclonal T cells)",
        "t_cell_pct_mean": 0.5,
        "pathognomonic": "T-B-NK+ SCID (NK cells PRESENT) = V(D)J recombination defect (RAG1/RAG2/Artemis panel). Omenn syndrome (erythroderma + eosinophilia + hepatosplenomegaly + oligoclonal T cells) = hypomorphic RAG1/RAG2.",
        "key_avoid": "NEVER give BCG; NEVER give live vaccines; NEVER give non-irradiated blood; NEVER miss Omenn (can be confused with GvHD or Langerhans histiocytosis); NEVER sequence RAG1 alone (always co-test RAG2)",
        "key_rule": "RAG1: T-B-NK+ SCID OR Omenn (erythroderma/eosinophilia). NK present = V(D)J defect. Always co-test RAG2 (adjacent 11p13). Omenn requires immunosuppression (steroids/ciclosporin) BEFORE HSCT to prevent fatal engraftment failure.",
        "surveillance": [
            "TREC NBS = 0 (RAG1/RAG2 null; may be detectable in Omenn hypomorphic)",
            "T/B/NK flow (NK present = V(D)J defect; distinguish from IL2RG/JAK3)",
            "Eosinophil count + skin biopsy (Omenn: erythroderma)",
            "RAG1/RAG2 gene sequencing panel (mandatory cotesting)",
            "T-cell receptor spectratyping (oligoclonality confirms Omenn)",
        ],
        "targeted_rx": "HSCT curative; Omenn requires steroids + ciclosporin before HSCT; IVIG bridging; gene therapy investigational (lentiviral RAG1/RAG2 in trials)",
    },
    {
        "gene": "RAG2",
        "protein": (
            "RAG2 -- 11p13 Autosomal-Recessive-LOF -- 527aa -- "
            "Recombination-Activating-Gene-2-59kDa-V-D-J-Recombination-Catalytic-Partner-"
            "T-B-NK+-SCID-Null-or-Omenn-Syndrome-Hypomorphic-RAG1-Adjacent-Same-Panel-"
            "BCG-ABSOLUTELY-CI-Identical-Phenotype-RAG1-OMIM-179616"
        ),
        "locus": "11p13",
        "protein_size": (
            "527 aa / 59 kDa / RAG2 molecular context: "
            "STRUCTURE: "
            "  527 aa / 59 kDa; C-terminal plant homeodomain (PHD): reads H3K4me3 histone mark → "
            "    targets RAG complex to actively transcribed chromatin; "
            "  Core RAG2 (aa 1-387): essential for recombination; "
            "  KELCH-repeat β-propeller domain: scaffolding; "
            "  PHD finger (aa 414-487): histone recognition domain; "
            "  Degradation signal (aa 383-527): phosphorylated by Cyclin B/CDK1 → degrades in S phase; "
            "    (V(D)J recombination restricted to G1 phase); "
            "FUNCTION: "
            "  RAG2 is the CATALYTIC PARTNER of RAG1 — they form an obligate tetramer (RAG1)2(RAG2)2; "
            "  RAG2 PHD domain reads H3K4me3 mark → opens chromatin at V/D/J segments; "
            "  Without RAG2: RAG1 cannot initiate V(D)J cleavage; "
            "IMMUNOPHENOTYPE: "
            "  NULL alleles: T-B-NK+ SCID (IDENTICAL to RAG1 null — cannot distinguish by phenotype); "
            "  HYPOMORPHIC alleles: Omenn syndrome (identical presentation to RAG1 hypomorphic); "
            "  LATE-ONSET hypomorphic: combined ID with granulomata; "
            "KEY RULE: "
            "  RAG1 and RAG2 at same chromosomal locus (11p13) → always sequenced together in one panel; "
            "  Phenotype cannot distinguish RAG1 from RAG2 — only genetics can; "
            "  RAG2 null = more commonly associated with Omenn than RAG1 null in some series; "
        ),
        "syndrome": "RAG2 deficiency — T-B-NK+ SCID or Omenn syndrome (hypomorphic)",
        "inheritance": "Autosomal recessive (AR)",
        "immunophenotype": "T-B-NK+ SCID: T absent, B absent, NK PRESENT; or Omenn (hypomorphic)",
        "t_cell_pct_mean": 0.6,
        "pathognomonic": "T-B-NK+ SCID (NK present) = V(D)J recombination defect. RAG1 vs RAG2: CLINICALLY IDENTICAL — only sequencing distinguishes. Always co-test RAG1+RAG2 on same panel (11p13 locus).",
        "key_avoid": "NEVER give BCG; NEVER give live vaccines; NEVER sequence RAG2 alone (always co-test RAG1); NEVER miss Omenn syndrome (fatal without immunosuppression before HSCT); NEVER give non-irradiated blood",
        "key_rule": "RAG2: phenotype = RAG1. T-B-NK+ SCID or Omenn. 11p13 locus = RAG1+RAG2 mandatory panel. Omenn = hypomorphic: oligoclonal T cells + erythroderma + eosinophilia + HSM. Immunosuppression BEFORE HSCT mandatory in Omenn.",
        "surveillance": [
            "TREC NBS = 0 (null), may be detectable (hypomorphic Omenn)",
            "T/B/NK flow (T absent, B absent, NK PRESENT)",
            "Eosinophil count; T-cell receptor spectratyping for oligoclonality",
            "RAG1+RAG2 gene sequencing panel (mandatory cotesting)",
            "Skin biopsy (Omenn: eosinophilic dermal infiltrate)",
        ],
        "targeted_rx": "HSCT curative; Omenn requires steroids + ciclosporin before HSCT; IVIG; gene therapy investigational (lentiviral RAG2 in clinical trials)",
    },
    {
        "gene": "DCLRE1C",
        "protein": (
            "DCLRE1C -- 10p13 Autosomal-Recessive-LOF -- 692aa -- "
            "Artemis-77kDa-Metallo-beta-Lactamase-SNM1B-V-D-J-Hairpin-Opening-"
            "T-B-NK+-Radiosensitive-SCID-TBI-ABSOLUTELY-CI-RIC-Without-TBI-MANDATORY-"
            "Navajo-Athabascan-p.Thr347Ile-Founder-BCG-ABSOLUTELY-CI-OMIM-605988"
        ),
        "locus": "10p13",
        "protein_size": (
            "692 aa / 77 kDa / DCLRE1C (Artemis) molecular context: "
            "STRUCTURE: "
            "  692 aa / 77 kDa; metallo-β-lactamase domain (MBL, aa 1-138): catalytic metal-binding; "
            "  β-CASP domain (aa 139-385): specifically opens V(D)J hairpin intermediates; "
            "  C-terminal domain (aa 386-692): regulatory, interacts with DNA-PKcs; "
            "  Artemis is a member of the SNM1 nuclease family; "
            "V(D)J HAIRPIN OPENING: "
            "  During V(D)J recombination, RAG complex creates hairpin-ended coding sequences; "
            "  Hairpins MUST be opened by Artemis (endonuclease) before NHEJ ligation; "
            "  DNA-PKcs phosphorylates Artemis → activates hairpin-opening endonuclease activity; "
            "  Without Artemis: hairpins accumulate → V(D)J arrested → T-B-NK+ SCID; "
            "DNA REPAIR DEFECT: "
            "  Artemis also participates in repair of radiation-induced DNA DSBs via NHEJ; "
            "  RADIOSENSITIVITY ABSOLUTE: irradiation → DSBs → cannot be repaired → cell death; "
            "  TBI conditioning → FATAL marrow/gut toxicity → TBI ABSOLUTELY CONTRAINDICATED; "
            "  Busulfan-based reduced-intensity conditioning (RIC) without TBI required; "
            "  Alkylating agents: reduced doses needed (higher busulfan sensitivity); "
            "NAVAJO ATHABASCAN (NAID): "
            "  p.Thr347Ile founder in Navajo/Apache Native American populations; "
            "  May present with only cutaneous granulomata (hypomorphic alleles); "
        ),
        "syndrome": "Artemis (DCLRE1C) SCID — T-B-NK+ radiosensitive SCID",
        "inheritance": "Autosomal recessive (AR)",
        "immunophenotype": "T-B-NK+ SCID: T absent, B absent, NK PRESENT; + DNA double-strand break repair defect",
        "t_cell_pct_mean": 0.4,
        "pathognomonic": "T-B-NK+ SCID + RADIOSENSITIVITY (colony survival assay) = Artemis/DCLRE1C. TBI ABSOLUTELY CONTRAINDICATED as HSCT conditioning. Navajo/Apache population = p.Thr347Ile founder.",
        "key_avoid": "NEVER give BCG; NEVER use TBI for HSCT conditioning (fatal); NEVER give standard alkylating doses (busulfan dose-reduce); NEVER give live vaccines; NEVER miss radiosensitivity screen before HSCT conditioning",
        "key_rule": "DCLRE1C/Artemis: T-B-NK+ SCID + RADIOSENSITIVITY. ABSOLUTE: NO TBI. Reduced-intensity busulfan conditioning required. Radiosensitivity screen (colony-forming assay) mandatory before HSCT. Navajo/Apache: p.Thr347Ile founder.",
        "surveillance": [
            "TREC NBS = 0",
            "T/B/NK flow (T absent, B absent, NK PRESENT — same as RAG1/RAG2)",
            "Radiosensitivity colony-forming assay (MANDATORY before HSCT conditioning)",
            "DCLRE1C sequencing; targeted Navajo p.Thr347Ile in Native American populations",
            "Chromosomal breakage assay (spontaneous DSBs)",
        ],
        "targeted_rx": "HSCT (RIC without TBI — busulfan-based); gene therapy investigational; IVIG; PCP prophylaxis; Navajo-NAID: HSCT with carefully dose-reduced busulfan",
    },
    {
        "gene": "ADA",
        "protein": (
            "ADA -- 20q13.11 Autosomal-Recessive-LOF -- 363aa -- "
            "Adenosine-Deaminase-41kDa-Purine-Catabolism-dATP-Toxic-Metabolite-"
            "T-B-NK--SCID-Metabolic-PEG-ADA-Enzyme-Replacement-LIFE-SAVING-"
            "Gene-Therapy-Strimvelis-First-Approved-Gene-Therapy-EU-2016-"
            "Calcifications-Hearing-Loss-50pct-PATHOGNOMONIC-BCG-ABSOLUTELY-CI-OMIM-608958"
        ),
        "locus": "20q13.11",
        "protein_size": (
            "363 aa / 41 kDa / ADA molecular context: "
            "STRUCTURE: "
            "  363 aa / 41 kDa; zinc metalloenzyme; homodimer; "
            "  Catalytic active site: zinc ion + His17/His214/Asp296 triad; "
            "  Catalyses: adenosine → inosine + NH3; 2'-deoxyadenosine → 2'-deoxyinosine + NH3; "
            "TOXIC METABOLITE MECHANISM: "
            "  ADA LOF → accumulation of 2'-deoxyadenosine → phosphorylated by deoxycytidine kinase → dATP; "
            "  dATP accumulates preferentially in lymphocytes (high deoxyadenosine kinase / low 5'-NT ratio); "
            "  dATP pool imbalance → ribonucleotide reductase inhibition → DNA synthesis block → apoptosis; "
            "  T cells > B cells > NK cells affected → T-B-NK- SCID (all lymphocytes lost); "
            "  Additional toxicity: S-adenosylhomocysteine hydrolase inhibition → lymphocyte function impaired; "
            "UNIQUE FEATURES: "
            "  ENZYME REPLACEMENT POSSIBLE: PEG-ADA (pegademase bovine) — weekly/biweekly IM injection; "
            "  PEG-ADA bridges patient to HSCT or gene therapy — LIFE-SAVING; "
            "  CALCIFICATIONS: periarticular (elbow/knee), renal, pulmonary — toxic dATP deposits; "
            "    Skeletal survey at diagnosis → calcifications PATHOGNOMONIC; "
            "  NEUROLOGICAL: sensorineural hearing loss 50% (dATP cochlear toxicity); cognitive involvement; "
            "  GENE THERAPY: Strimvelis® EU EMA approved 2016 — first gene therapy ever approved; "
            "    Ex vivo ADA cDNA retroviral correction of autologous CD34+ HSCs; "
            "    Lentiviral vector gene therapy also in trials; "
        ),
        "syndrome": "ADA-SCID — T-B-NK- SCID (metabolic/toxic metabolite)",
        "inheritance": "Autosomal recessive (AR)",
        "immunophenotype": "T-B-NK- SCID: T absent, B absent, NK absent (dATP toxicity kills all lymphocytes)",
        "t_cell_pct_mean": 0.3,
        "pathognomonic": "T-B-NK- SCID + PERIARTICULAR CALCIFICATIONS on X-ray + SENSORINEURAL HEARING LOSS 50% = ADA-SCID PATHOGNOMONIC. PEG-ADA enzyme replacement bridges to gene therapy (FIRST APPROVED GENE THERAPY EU 2016).",
        "key_avoid": "NEVER give BCG; NEVER give live vaccines; NEVER delay PEG-ADA while awaiting HSCT (enzyme replacement LIFE-SAVING bridge); NEVER miss skeletal survey (calcifications diagnostic); NEVER give non-irradiated blood",
        "key_rule": "ADA-SCID: metabolic SCID with toxic dATP. T-B-NK- SCID. ENZYME REPLACEMENT (PEG-ADA) bridges to HSCT or gene therapy. GENE THERAPY APPROVED (Strimvelis®). Periarticular calcifications + hearing loss 50% are PATHOGNOMONIC clues.",
        "surveillance": [
            "TREC NBS = 0",
            "T/B/NK flow (T absent, B absent, NK absent — ALL lymphocytes)",
            "dATP level in erythrocytes (diagnostic marker of ADA deficiency)",
            "ADA enzyme activity in erythrocytes (< 1% = diagnostic)",
            "Skeletal X-ray survey (periarticular calcifications); audiogram (SNHL 50%)",
        ],
        "targeted_rx": "PEG-ADA (pegademase bovine) enzyme replacement therapy — bridge; HSCT (genoidentical/haploidentical); gene therapy: Strimvelis® (EU approved) or lentiviral ADA-LV (trials); IVIG; antibiotics/antivirals",
    },
    {
        "gene": "IL7R",
        "protein": (
            "IL7R -- 5p13.2 Autosomal-Recessive-LOF -- 459aa -- "
            "IL-7-Receptor-Alpha-Chain-50kDa-Fibronectin-III-TM-Cytoplasmic-Tail-"
            "T-B+NK+-SCID-T-Absent-B-NK-Normal-IL7-Required-for-T-Cell-Thymic-Development-"
            "HSCT-Curative-BCG-ABSOLUTELY-CI-OMIM-146661"
        ),
        "locus": "5p13.2",
        "protein_size": (
            "459 aa / 50 kDa / IL7R molecular context: "
            "STRUCTURE: "
            "  459 aa / 50 kDa; type I cytokine receptor superfamily; "
            "  Signal peptide (aa 1-19); "
            "  Extracellular domain (aa 20-239): two fibronectin type III-like modules; "
            "  Cytokine-binding domain with conserved WSXWS motif (Trp-Ser-X-Trp-Ser); "
            "  Single TM helix (aa 240-261); "
            "  Cytoplasmic tail (aa 262-459): Box1 motif (JAK1 binding); "
            "  IL-7Rα pairs with γc chain (IL2RG) to form the IL-7 receptor (IL-7R); "
            "FUNCTION: "
            "  IL-7 + IL-7Rα/γc → JAK1 (on IL-7Rα) + JAK3 (on γc) transphosphorylation → STAT5 → "
            "    BCL-2 upregulation → T cell survival; NOTCH1 → T cell proliferation in thymus; "
            "  IL-7 is EXCLUSIVELY REQUIRED for T cell development (not B or NK); "
            "  γc also used by IL-15 (NK development) — but IL-15 uses a different α chain (IL-15Rα); "
            "IMMUNOPHENOTYPE: "
            "  T cells: ABSENT (IL-7 block in thymic T cell development); "
            "  B cells: PRESENT and FUNCTIONAL (B development IL-7-independent); "
            "  NK cells: PRESENT (NK uses IL-15/γc, not IL-7/γc); "
            "  T-B+NK+ SCID: CRITICAL DISTINCTION from IL2RG/JAK3 (T-B+NK-); "
            "KEY RULE: "
            "  T-B+NK+ SCID = IL7R (or CD3 complex defects): lymphopenia affects ONLY T cells; "
            "  T-B+NK- SCID = IL2RG/JAK3 (NK also absent); "
            "  T-B-NK+ SCID = RAG1/RAG2/DCLRE1C (B cells also absent); "
            "  NK cell count is the TRIAGE: absent NK = γc chain (IL2RG/JAK3); present NK = other SCID; "
        ),
        "syndrome": "IL-7Rα deficiency — T-B+NK+ SCID",
        "inheritance": "Autosomal recessive (AR)",
        "immunophenotype": "T-B+NK+ SCID: T cells absent, B cells PRESENT and functional, NK cells PRESENT",
        "t_cell_pct_mean": 0.4,
        "pathognomonic": "T-B+NK+ SCID (B AND NK both present) = IL7R or CD3 complex defect (CD3 flow distinguishes). NK cells PRESENT distinguishes from IL2RG/JAK3 (NK absent). T lymphopenia ONLY — B and NK completely normal.",
        "key_avoid": "NEVER give BCG; NEVER give live vaccines; NEVER give non-irradiated blood; NEVER confuse with IL2RG (NK present in IL7R); NEVER miss T-B+NK+ pattern (may be subtle lymphopenia rather than complete absence)",
        "key_rule": "IL7R: T-B+NK+ SCID — T lymphopenia ONLY. B and NK NORMAL. NK cells PRESENT = key DDx from IL2RG/JAK3 (T-B+NK-). Next step: CD3 flow (CD3 present in IL7R; CD3 absent in CD3 complex defects).",
        "surveillance": [
            "TREC NBS = 0",
            "T/B/NK flow (T absent, B PRESENT, NK PRESENT — key pattern)",
            "CMV/EBV/PCP surveillance biweekly",
            "Immunoglobulin levels (normal — B cells functional)",
            "IL7R gene sequencing",
        ],
        "targeted_rx": "HSCT curative; gene therapy investigational; IVIG not required (functional B cells); PCP prophylaxis; CMV prophylaxis (ganciclovir); no targeted IL-7 pathway therapy approved for immunodeficiency",
    },
    {
        "gene": "AK2",
        "protein": (
            "AK2 -- 1p34.1 Autosomal-Recessive-LOF -- 239aa -- "
            "Adenylate-Kinase-2-26kDa-Mitochondrial-Intermembrane-Space-"
            "Reticular-Dysgenesis-Most-Severe-SCID-SCID+Agranulocytosis-"
            "Inner-Ear-AK2-Bilateral-Sensorineural-Hearing-Loss-60pct-PATHOGNOMONIC-"
            "URGENT-HSCT-Bacterial-Sepsis-Primary-Threat-BCG-ABSOLUTELY-CI-OMIM-103020"
        ),
        "locus": "1p34.1",
        "protein_size": (
            "239 aa / 26 kDa / AK2 molecular context: "
            "STRUCTURE: "
            "  239 aa / 26 kDa; adenylate kinase family; "
            "  N-terminal mitochondrial targeting sequence (aa 1-24) → cleaved in mature protein; "
            "  AMPbind domain (aa 1-75); LID domain (aa 117-143); CORE domain; "
            "  Active site: catalyses: 2 ADP ↔ ATP + AMP (in mitochondrial intermembrane space); "
            "  Also: AMP + ATP ↔ 2 ADP; critical for mitochondrial adenine nucleotide recycling; "
            "UNIQUE BIOLOGY: "
            "  AK2 expressed primarily in mitochondrial intermembrane space; "
            "  Required for APOPTOSIS REGULATION via cytochrome c release pathway; "
            "  AK2 LOF: mitochondrial dysfunction → apoptosis of lymphoid AND myeloid progenitors; "
            "RETICULAR DYSGENESIS — MOST SEVERE SCID: "
            "  T cells: ABSENT; B cells: ABSENT/SEVERELY REDUCED; NK cells: ABSENT; "
            "  GRANULOCYTES: ABSENT/SEVERELY REDUCED (agranulocytosis); "
            "  Bone marrow: arrest at myeloid AND lymphoid progenitor levels; "
            "  Death from BACTERIAL SEPSIS in first WEEKS of life (before PCP — no neutrophils); "
            "HEARING LOSS: "
            "  AK2 is expressed in stria vascularis of inner ear; "
            "  Bilateral sensorineural hearing loss in 60% = PATHOGNOMONIC clue; "
            "MANAGEMENT: "
            "  URGENT HSCT within first weeks — survival measured in weeks without treatment; "
            "  Broad-spectrum antibiotics + antifungals from birth (neutropenia = sepsis first threat); "
            "  G-CSF has limited role (constitutional marrow defect); "
        ),
        "syndrome": "Reticular dysgenesis (AK2 deficiency) — most severe SCID + agranulocytosis",
        "inheritance": "Autosomal recessive (AR)",
        "immunophenotype": "T-B-NK- SCID + AGRANULOCYTOSIS: absent T, B, NK AND absent neutrophils/granulocytes",
        "t_cell_pct_mean": 0.1,
        "pathognomonic": "SCID + AGRANULOCYTOSIS (neutropenia from birth) = RETICULAR DYSGENESIS (AK2) PATHOGNOMONIC. Bilateral SNHL in 60%. Death from bacterial sepsis in weeks. MOST SEVERE AND RAREST SCID. URGENT HSCT in first weeks.",
        "key_avoid": "NEVER give BCG; NEVER give live vaccines; NEVER give non-irradiated blood; NEVER delay antibiotics (bacterial sepsis is the ACUTE threat); NEVER wait for opportunistic infections (they die of bacterial sepsis first); NEVER use G-CSF as sole treatment (constitutional defect)",
        "key_rule": "AK2 = Reticular Dysgenesis: SCID + AGRANULOCYTOSIS. Most severe SCID. Bacterial sepsis kills in weeks (not PCP). URGENT HSCT immediately. SNHL 60% = diagnostic clue. Broad-spectrum antibiotics + antifungals from birth mandatory.",
        "surveillance": [
            "TREC NBS = 0",
            "CBC + differential (agranulocytosis — neutrophils absent from birth)",
            "T/B/NK flow (T absent, B absent/reduced, NK absent)",
            "Audiogram (bilateral SNHL 60%)",
            "AK2 gene sequencing",
        ],
        "targeted_rx": "URGENT HSCT (first weeks of life); broad-spectrum antibiotics + antifungals (bacteria = primary threat); IVIG; G-CSF limited role; no approved enzyme replacement or gene therapy for AK2",
    },
]


def _rng(seed: int) -> random.Random:
    r = random.Random()
    r.seed(seed)
    return r


def _patient_cohort(gene_idx: int):
    """Generate 40-patient synthetic cohort for one gene."""
    gene = ATLAS_GENES[gene_idx]
    seed = SEED_BASE + gene_idx
    r = _rng(seed)
    gene_name = gene["gene"]

    # Core clinical parameters by gene
    params = {
        "IL2RG":  dict(t_bnk_minus=True,  nk_absent=True,  radiosens=False, agranulocytosis=False, calcif=False, snhl_pct=0.05, gene_therapy=True,  peg_ada=False, hsct_pct=85, live_to_dx_mo=4.0),
        "JAK3":   dict(t_bnk_minus=True,  nk_absent=True,  radiosens=False, agranulocytosis=False, calcif=False, snhl_pct=0.04, gene_therapy=False, peg_ada=False, hsct_pct=83, live_to_dx_mo=4.5),
        "RAG1":   dict(t_bnk_minus=False, nk_absent=False, radiosens=False, agranulocytosis=False, calcif=False, snhl_pct=0.03, gene_therapy=False, peg_ada=False, hsct_pct=80, live_to_dx_mo=3.8),
        "RAG2":   dict(t_bnk_minus=False, nk_absent=False, radiosens=False, agranulocytosis=False, calcif=False, snhl_pct=0.03, gene_therapy=False, peg_ada=False, hsct_pct=78, live_to_dx_mo=4.0),
        "DCLRE1C":dict(t_bnk_minus=False, nk_absent=False, radiosens=True,  agranulocytosis=False, calcif=False, snhl_pct=0.05, gene_therapy=False, peg_ada=False, hsct_pct=70, live_to_dx_mo=5.0),
        "ADA":    dict(t_bnk_minus=True,  nk_absent=True,  radiosens=False, agranulocytosis=False, calcif=True,  snhl_pct=0.50, gene_therapy=True,  peg_ada=True,  hsct_pct=60, live_to_dx_mo=5.5),
        "IL7R":   dict(t_bnk_minus=False, nk_absent=False, radiosens=False, agranulocytosis=False, calcif=False, snhl_pct=0.03, gene_therapy=False, peg_ada=False, hsct_pct=82, live_to_dx_mo=3.5),
        "AK2":    dict(t_bnk_minus=True,  nk_absent=True,  radiosens=False, agranulocytosis=True,  calcif=False, snhl_pct=0.60, gene_therapy=False, peg_ada=False, hsct_pct=55, live_to_dx_mo=1.5),
    }
    p = params[gene_name]

    patients = []
    for i in range(40):
        age = r.gauss(p["live_to_dx_mo"], 1.5)
        age = max(0.2, round(age, 1))
        t_pct = gene["t_cell_pct_mean"] + r.gauss(0, 0.15)
        t_pct = max(0, round(t_pct, 2))

        # NK absent: IL2RG, JAK3, ADA, AK2
        nk_absent = p["nk_absent"]
        # Agranulocytosis: AK2 only
        agranulocytosis = p["agranulocytosis"] and r.random() < 0.92
        # Calcifications: ADA only
        calcifications = p["calcif"] and r.random() < 0.60
        # SNHL
        snhl = r.random() < p["snhl_pct"]
        # Radiosensitivity: DCLRE1C
        radiosens = p["radiosens"] and r.random() < 0.95
        # PEG-ADA: ADA
        peg_ada = p["peg_ada"] and r.random() < 0.80
        # Gene therapy: IL2RG and ADA
        gene_therapy_received = p["gene_therapy"] and r.random() < 0.25
        # HSCT
        hsct_done = r.random() < (p["hsct_pct"] / 100.0)
        # Omenn (RAG1/RAG2 hypomorphic)
        omenn = gene_name in ("RAG1", "RAG2") and r.random() < 0.30
        # PCP pneumonia
        pcp = r.random() < (0.30 if p["t_bnk_minus"] else 0.22)
        # Opportunistic infection
        ops_inf = r.random() < 0.60
        # BCG reaction
        bcg_reaction = r.random() < 0.25

        patients.append({
            "id": i + 1,
            "gene": gene_name,
            "age_at_dx_months": age,
            "t_pct": t_pct,
            "nk_absent": nk_absent,
            "agranulocytosis": agranulocytosis,
            "calcifications": calcifications,
            "snhl": snhl,
            "radiosensitivity": radiosens,
            "peg_ada_received": peg_ada,
            "gene_therapy_received": gene_therapy_received,
            "hsct_done": hsct_done,
            "omenn_syndrome": omenn,
            "pcp_pneumonia": pcp,
            "opportunistic_infection": ops_inf,
            "bcg_reaction": bcg_reaction,
        })
    return patients


def generate_overview():
    all_patients = []
    for i in range(8):
        all_patients.extend(_patient_cohort(i))

    total = len(all_patients)
    t_absent_pct = round(sum(1 for p in all_patients if p["t_pct"] < 1.0) / total * 100, 1)
    nk_absent_pct = round(sum(1 for p in all_patients if p["nk_absent"]) / total * 100, 1)
    agranulocytosis_pct = round(sum(1 for p in all_patients if p["agranulocytosis"]) / total * 100, 1)
    calcif_pct = round(sum(1 for p in all_patients if p["calcifications"]) / total * 100, 1)
    snhl_pct = round(sum(1 for p in all_patients if p["snhl"]) / total * 100, 1)
    radiosens_pct = round(sum(1 for p in all_patients if p["radiosensitivity"]) / total * 100, 1)
    peg_ada_pct = round(sum(1 for p in all_patients if p["peg_ada_received"]) / total * 100, 1)
    gene_therapy_pct = round(sum(1 for p in all_patients if p["gene_therapy_received"]) / total * 100, 1)
    hsct_pct = round(sum(1 for p in all_patients if p["hsct_done"]) / total * 100, 1)
    omenn_pct = round(sum(1 for p in all_patients if p["omenn_syndrome"]) / total * 100, 1)
    bcg_pct = round(sum(1 for p in all_patients if p["bcg_reaction"]) / total * 100, 1)

    return {
        "atlas": "Hereditary-SCID-RAG-IL2RG-ADA-Atlas",
        "subtitle": "Complete 8-Gene Reference · Classic Severe Combined Immunodeficiency Genes",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total,
        "gene_counts": {g["gene"]: 40 for g in ATLAS_GENES},
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "aggregate_stats": {
            "t_absent_pct": t_absent_pct,
            "nk_absent_pct": nk_absent_pct,
            "agranulocytosis_pct": agranulocytosis_pct,
            "calcifications_pct": calcif_pct,
            "snhl_pct": snhl_pct,
            "radiosensitivity_pct": radiosens_pct,
            "peg_ada_pct": peg_ada_pct,
            "gene_therapy_pct": gene_therapy_pct,
            "hsct_pct": hsct_pct,
            "omenn_syndrome_pct": omenn_pct,
            "bcg_reaction_pct": bcg_pct,
            "gamma_c_chain_genes": ["IL2RG", "JAK3"],
            "vdj_recombination_genes": ["RAG1", "RAG2", "DCLRE1C"],
            "metabolic_toxic_gene": ["ADA"],
            "cytokine_receptor_gene": ["IL7R"],
            "mitochondrial_gene": ["AK2"],
        },
        "key_facts": [
            "IL2RG: X-SCID most common 45-50%; T-B+NK- SCID; PCP infant boy PATHOGNOMONIC; gene therapy approved (Strimvelis®); seeds 3958",
            "JAK3: AR phenocopy X-SCID; T-B+NK- SCID BOTH SEXES; JAK3 Western blot diagnostic; female T-B+NK- → JAK3 first; seeds 3959",
            "RAG1: T-B-NK+ SCID null or Omenn hypomorphic; NK PRESENT = V(D)J defect; RAG1+RAG2 co-testing mandatory; seeds 3960",
            "RAG2: T-B-NK+ SCID null or Omenn; CLINICALLY IDENTICAL to RAG1 — genetics only distinguishes; 11p13 same panel; seeds 3961",
            "DCLRE1C/Artemis: T-B-NK+ SCID + RADIOSENSITIVITY; TBI ABSOLUTELY CI; RIC without TBI mandatory; Navajo p.Thr347Ile founder; seeds 3962",
            "ADA: metabolic SCID T-B-NK-; dATP toxic; PEG-ADA enzyme replacement LIFE-SAVING bridge; gene therapy APPROVED (Strimvelis® EU 2016); calcifications + SNHL 50% PATHOGNOMONIC; seeds 3963",
            "IL7R: T-B+NK+ SCID — T lymphopenia ONLY; B and NK normal; IL-7 required only for T cell development; distinguish from IL2RG (NK present); seeds 3964",
            "AK2: Reticular Dysgenesis MOST SEVERE SCID; SCID + AGRANULOCYTOSIS; death in weeks; URGENT HSCT; SNHL 60% PATHOGNOMONIC; seeds 3965",
        ],
    }


def generate_breakdown():
    gene_cohorts = []
    diagnostic_groups = {
        "gamma_c_chain_jak3": {
            "genes": ["IL2RG", "JAK3"],
            "mechanism": "Loss of γc chain (IL2RG) or its exclusive kinase JAK3 → absent IL-2/4/7/9/15/21 signalling → T-B+NK- SCID (NK absent because IL-15/γc required for NK development)",
            "shared_feature": "T-B+NK- SCID: T cells absent, NK cells ABSENT. B cells present but non-functional. NK ABSENT is the key feature.",
            "key_ddx": "IL2RG: X-linked — virtually all males. JAK3: autosomal recessive — BOTH SEXES. Female with T-B+NK- SCID = JAK3. JAK3 Western blot absent. Gene therapy approved for IL2RG.",
            "bcg_ci": True,
        },
        "vdj_recombination": {
            "genes": ["RAG1", "RAG2", "DCLRE1C"],
            "mechanism": "Loss of V(D)J recombination machinery → absent T and B cell receptor diversity → T-B-NK+ SCID (NK cells PRESENT because NK uses germline-encoded receptors not V(D)J)",
            "shared_feature": "T-B-NK+ SCID: T absent, B absent, NK PRESENT. NK cells spared = V(D)J defect. Null → complete SCID; hypomorphic RAG1/RAG2 → Omenn syndrome.",
            "key_ddx": "RAG1 vs RAG2: phenotype identical — genetics mandatory (co-test, same 11p13 panel). DCLRE1C/Artemis: T-B-NK+ SCID + RADIOSENSITIVITY. TBI absolutely CI in DCLRE1C. Navajo/Apache: p.Thr347Ile founder.",
            "bcg_ci": True,
        },
        "metabolic_cytokine": {
            "genes": ["ADA", "IL7R"],
            "mechanism": "Metabolic toxicity (ADA: dATP accumulation → lymphocyte apoptosis → T-B-NK-) or cytokine receptor defect (IL7R: IL-7Rα LOF → T cell thymic block → T-B+NK+)",
            "shared_feature": "Different immunophenotypes but both T lymphopenic. ADA: T-B-NK- (all lymphocytes) + enzyme replacement possible + gene therapy approved. IL7R: T-B+NK+ (T only).",
            "key_ddx": "ADA: T-B-NK- + calcifications + SNHL 50% + enzyme replacement (PEG-ADA) + gene therapy (Strimvelis®). IL7R: T-B+NK+ (B and NK NORMAL — isolated T lymphopenia). NK cell count triage.",
            "bcg_ci": True,
        },
        "mitochondrial_reticular": {
            "genes": ["AK2"],
            "mechanism": "Mitochondrial adenylate kinase 2 deficiency → mitochondrial dysfunction → apoptosis of both lymphoid AND myeloid progenitors → T-B-NK- SCID + agranulocytosis = Reticular Dysgenesis",
            "shared_feature": "Reticular Dysgenesis: T-B-NK- SCID + AGRANULOCYTOSIS. Most severe SCID. Death from bacterial sepsis in weeks. AK2 expressed in stria vascularis → bilateral SNHL 60%.",
            "key_ddx": "AK2 vs ADA (both T-B-NK-): AK2 has AGRANULOCYTOSIS (ADA does not). AK2 SNHL 60% vs ADA SNHL 50%. AK2: no enzyme replacement available. AK2: most urgent HSCT (bacterial sepsis first threat, not PCP).",
            "bcg_ci": True,
        },
    }

    pathognomonic_features = {
        "IL2RG_pcp_infant_boy": "PCP pneumonia in an INFANT BOY aged 2-6 months = X-SCID (IL2RG) until proven otherwise. T-B+NK- SCID. NK absent distinguishes from CD3/IL7R defects.",
        "JAK3_female_t_bnk_minus": "T-B+NK- SCID in a FEMALE = JAK3 (autosomal) NOT IL2RG (X-linked). Phenotype identical to X-SCID. JAK3 Western blot absent. BOTH SEXES affected equally.",
        "RAG1_RAG2_nk_present": "T-B-NK+ SCID (NK PRESENT) = V(D)J recombination defect. NK present distinguishes from IL2RG/JAK3. RAG1+RAG2 COTESTING mandatory (adjacent 11p13). Omenn = hypomorphic (erythroderma + eosinophilia + HSM).",
        "DCLRE1C_radiosensitivity": "T-B-NK+ SCID + RADIOSENSITIVITY = Artemis/DCLRE1C. TBI ABSOLUTELY CI. RIC without TBI mandatory. Navajo/Apache Native American: p.Thr347Ile founder variant.",
        "ADA_calcifications_snhl": "T-B-NK- SCID + PERIARTICULAR CALCIFICATIONS (X-ray) + SENSORINEURAL HEARING LOSS 50% = ADA-SCID PATHOGNOMONIC triad. FIRST GENE THERAPY EU approved (Strimvelis® 2016). PEG-ADA enzyme replacement bridges to definitive treatment.",
        "IL7R_t_bnk_plus": "T-B+NK+ SCID: T lymphopenia ONLY — B and NK NORMAL. NK present = key DDx from IL2RG/JAK3 (NK absent). Next: CD3 flow (IL7R: CD3 present; CD3 complex defects: CD3 absent).",
        "AK2_agranulocytosis": "SCID + AGRANULOCYTOSIS (neutropenia from birth) = RETICULAR DYSGENESIS (AK2) PATHOGNOMONIC. Most severe SCID. Bacterial sepsis kills in weeks. BILATERAL SNHL 60%. URGENT HSCT immediately. Broad-spectrum antibiotics from birth.",
        "bcg_absolutely_ci_all_8": "BCG ABSOLUTELY CONTRAINDICATED in all 8 SCID genes — severe T cell functional deficiency. BCG-osis (disseminated fatal mycobacterial disease) reported in all SCID types. TREC NBS prior to BCG birth dose prevents BCG-osis.",
    }

    for i, gene_data in enumerate(ATLAS_GENES):
        patients = _patient_cohort(i)
        gene_name = gene_data["gene"]
        n = len(patients)
        t_absent_n = sum(1 for p in patients if p["t_pct"] < 1.0)
        agranulocytosis_n = sum(1 for p in patients if p["agranulocytosis"])
        calcif_n = sum(1 for p in patients if p["calcifications"])
        snhl_n = sum(1 for p in patients if p["snhl"])
        radiosens_n = sum(1 for p in patients if p["radiosensitivity"])
        peg_ada_n = sum(1 for p in patients if p["peg_ada_received"])
        gt_n = sum(1 for p in patients if p["gene_therapy_received"])
        hsct_n = sum(1 for p in patients if p["hsct_done"])
        omenn_n = sum(1 for p in patients if p["omenn_syndrome"])
        pcp_n = sum(1 for p in patients if p["pcp_pneumonia"])
        bcg_n = sum(1 for p in patients if p["bcg_reaction"])
        mean_age = round(sum(p["age_at_dx_months"] for p in patients) / n, 1)

        gene_cohorts.append({
            "gene": gene_name,
            "gene_info": gene_data,
            "n": n,
            "t_absent_pct": round(t_absent_n / n * 100, 1),
            "agranulocytosis_pct": round(agranulocytosis_n / n * 100, 1),
            "calcifications_pct": round(calcif_n / n * 100, 1),
            "snhl_pct": round(snhl_n / n * 100, 1),
            "radiosensitivity_pct": round(radiosens_n / n * 100, 1),
            "peg_ada_pct": round(peg_ada_n / n * 100, 1),
            "gene_therapy_pct": round(gt_n / n * 100, 1),
            "hsct_pct": round(hsct_n / n * 100, 1),
            "omenn_pct": round(omenn_n / n * 100, 1),
            "pcp_pct": round(pcp_n / n * 100, 1),
            "bcg_reaction_pct": round(bcg_n / n * 100, 1),
            "mean_age_at_dx_months": mean_age,
            "pathognomonic": gene_data["pathognomonic"],
        })

    return {
        "gene_cohorts": gene_cohorts,
        "diagnostic_groups": diagnostic_groups,
        "pathognomonic_features": pathognomonic_features,
        "cascade_testing_order": [
            "Step 1: TREC NBS at birth → TREC=0 → urgent immunology referral",
            "Step 2: T/B/NK lymphocyte subset enumeration",
            "Step 3: T-B+NK- (NK absent) → γc chain defect: IL2RG (male=first) or JAK3 (female=first)",
            "Step 4: T-B-NK+ (NK present, B also absent) → V(D)J defect: RAG1+RAG2 co-panel; DCLRE1C if radiosensitivity",
            "Step 5: T-B-NK- (all absent) → ADA enzyme assay + dATP level; or AK2 (agranulocytosis present?)",
            "Step 6: T-B+NK+ (only T absent) → IL7R (AR) or CD3 complex defects (CD3 flow distinguishes)",
            "Step 7: DCLRE1C check: radiosensitivity colony assay BEFORE HSCT conditioning (TBI CI if positive)",
            "Step 8: ADA: start PEG-ADA enzyme replacement IMMEDIATELY while awaiting gene therapy/HSCT",
            "Step 9: AK2: check CBC (agranulocytosis) + audiogram (SNHL 60%) + URGENT HSCT in first weeks",
            "Step 10: BCG ABSOLUTELY CONTRAINDICATED at all steps until T cell function confirmed normal",
        ],
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "T-B+NK- SCID", "definition": "Severe Combined Immunodeficiency with absent T cells and absent NK cells but PRESENT (non-functional) B cells. Phenotype of IL2RG (X-SCID) and JAK3 deficiency. NK absent because IL-15/γc required for NK development — same γc chain used by IL-15 receptor."},
            {"term": "T-B-NK+ SCID", "definition": "SCID with absent T AND B cells but PRESENT NK cells. Phenotype of V(D)J recombination defects (RAG1, RAG2, DCLRE1C/Artemis). NK cells use germline-encoded receptors (not V(D)J) → NK cells spared. NK present = key triage criterion for V(D)J defects."},
            {"term": "T-B-NK- SCID", "definition": "SCID with absent T, B AND NK cells. Phenotype of IL2RG/JAK3 (γc chain) AND ADA AND AK2 (reticular dysgenesis). For ADA: metabolic toxicity (dATP) kills all lymphocytes. For AK2: mitochondrial defect kills lymphoid + myeloid progenitors."},
            {"term": "T-B+NK+ SCID", "definition": "SCID with absent T cells but PRESENT B cells AND NK cells. Phenotype of IL7R (IL-7Rα) deficiency and CD3 complex defects. IL-7 required ONLY for T cell thymic development. B and NK development IL-7-independent. NK present = key distinction from IL2RG/JAK3 (NK absent)."},
            {"term": "X-SCID (IL2RG)", "definition": "X-linked Severe Combined Immunodeficiency caused by IL2RG (γc chain) mutations. Most common SCID (45-50% of cases). Virtually all males affected (X-linked). Presents in first 6 months as T-B+NK- SCID. Pathognomonic: PCP in infant boy. Gene therapy approved (EU 2016 Strimvelis®)."},
            {"term": "Omenn Syndrome", "definition": "Hypomorphic alleles of RAG1, RAG2 (or DCLRE1C) → partially functional V(D)J recombination → oligoclonal T cells that cause autoimmune destruction. Clinical triad: erythroderma + hepatosplenomegaly + hypereosinophilia. Oligoclonal T cells on TCR spectratyping. Requires immunosuppression (steroids/ciclosporin) BEFORE HSCT to prevent fatal engraftment failure."},
            {"term": "PEG-ADA (Enzyme Replacement)", "definition": "Pegademase bovine — polyethylene glycol-conjugated bovine ADA enzyme. Given weekly/biweekly IM injection. Replaces missing ADA enzyme → reduces dATP toxicity → partial immune reconstitution. LIFE-SAVING BRIDGE to HSCT or gene therapy. Does not correct underlying gene defect. Cannot substitute for definitive cure."},
            {"term": "Radiosensitive SCID (DCLRE1C/Artemis)", "definition": "SCID caused by DCLRE1C (Artemis) deficiency. Artemis endonuclease required for both V(D)J hairpin opening AND radiation-induced DNA double-strand break repair (NHEJ). Radiosensitivity = absolute. TBI (total body irradiation) for HSCT conditioning is ABSOLUTELY CONTRAINDICATED — fatal marrow and gut toxicity. Busulfan-based RIC without TBI required."},
            {"term": "Reticular Dysgenesis (AK2)", "definition": "Most severe and rarest SCID. AK2 LOF → mitochondrial dysfunction → apoptosis of both lymphoid AND myeloid progenitors → SCID + agranulocytosis. Death from bacterial sepsis in first weeks of life (neutropenia = sepsis primary threat before PCP). Bilateral SNHL in 60% (AK2 in stria vascularis). URGENT HSCT in first weeks mandatory."},
            {"term": "TREC (T-cell Receptor Excision Circles)", "definition": "Episomal DNA circles produced as byproducts of TCR V(D)J recombination in the thymus. TREC = 0 in ALL SCID types (no new T cells from thymus). Standard newborn screening (NBS) test. TREC NBS prevents BCG-osis by identifying SCID before birth-dose BCG is given."},
            {"term": "BCG-osis", "definition": "Disseminated Mycobacterium bovis infection from BCG vaccination in SCID patients. BCG is a live attenuated vaccine — fatal in T-cell deficiency. BCG ABSOLUTELY CONTRAINDICATED in all 8 SCID genes. TREC NBS before BCG birth dose prevents BCG-osis in undiagnosed SCID. BCG-osis requires prolonged antituberculous therapy even after HSCT."},
            {"term": "Gene Therapy — ADA-SCID and X-SCID", "definition": "ADA-SCID: Strimvelis® — first gene therapy ever approved (EU EMA 2016). Ex vivo retroviral ADA cDNA correction of autologous CD34+ HSCs. Curative without GvHD risk. X-SCID (IL2RG): lentiviral vector gene therapy approved (USA 2024). Both: ex vivo autologous correction. Gene therapy advantages over HSCT: no GvHD, no rejection, no donor needed."},
            {"term": "NHEJ (Non-Homologous End Joining)", "definition": "DNA double-strand break repair pathway used in post-mitotic cells and for V(D)J hairpin opening. Key proteins: Ku70/Ku80, DNA-PKcs, Artemis (DCLRE1C), Ligase IV, XRCC4, XLF. Artemis deficiency: V(D)J hairpins not opened → SCID + radiation cannot be repaired → radiosensitivity. TBI conditioning → fatal in Artemis SCID."},
        ],
        "key_distinguishers": {
            "nk_absent_vs_present": "NK ABSENT = γc chain defect (IL2RG/JAK3) or metabolic/mitochondrial (ADA/AK2). NK PRESENT = V(D)J defect (RAG1/RAG2/DCLRE1C) or cytokine receptor (IL7R) or CD3 complex defects.",
            "b_absent_vs_present": "B cells ABSENT: RAG1/RAG2/DCLRE1C (V(D)J needed for B cell receptor). B cells PRESENT: IL2RG/JAK3/IL7R (B development IL-7/γc-independent in humans). B cells absent + NK present = V(D)J defect.",
            "agranulocytosis": "AGRANULOCYTOSIS = AK2 (Reticular Dysgenesis) PATHOGNOMONIC. No other SCID gene causes neutropenia. CBC + differential at diagnosis identifies AK2 immediately.",
            "calcifications_snhl": "Periarticular CALCIFICATIONS on X-ray = ADA-SCID PATHOGNOMONIC. SNHL: ADA 50% (dATP cochlear toxicity) and AK2 60% (stria vascularis AK2 expression).",
            "radiosensitivity": "Radiosensitivity colony assay positive = DCLRE1C/Artemis. TBI absolutely CI. RIC-busulfan required. Navajo/Apache: p.Thr347Ile founder.",
            "sex_and_inheritance": "X-SCID (IL2RG): virtually ALL MALES. T-B+NK- SCID in FEMALE = JAK3 (autosomal recessive). All other SCID genes (RAG1/RAG2/DCLRE1C/ADA/IL7R/AK2): autosomal recessive = both sexes equally.",
            "gene_therapy_approved": "GENE THERAPY APPROVED: ADA-SCID (Strimvelis® EU 2016) and X-SCID/IL2RG (lentiviral USA 2024). Consider gene therapy before haploidentical HSCT when available and age-appropriate.",
        },
        "diagnostic_pathway": [
            "1. TREC NBS = 0 → SCID suspected → urgent immunology referral",
            "2. CBC + differential: agranulocytosis? → AK2 (Reticular Dysgenesis) — URGENT HSCT in days",
            "3. T/B/NK enumeration: determine SCID subtype",
            "4. T-B+NK-: IL2RG (male first) or JAK3 (female first) — gene panel",
            "5. T-B-NK+: RAG1+RAG2 panel (co-test mandatory); check radiosensitivity → DCLRE1C if positive",
            "6. T-B-NK-: ADA enzyme assay + dATP level; if agranulocytosis → AK2",
            "7. T-B+NK+: IL7R; then CD3 flow to exclude CD3 complex defects",
            "8. If DCLRE1C suspected: radiosensitivity assay before ANY conditioning regimen",
            "9. If ADA confirmed: start PEG-ADA immediately (bridge to gene therapy/HSCT)",
            "10. BCG ABSOLUTELY CONTRAINDICATED at all steps — TREC NBS prevents BCG-osis",
        ],
    }
