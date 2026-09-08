#!/usr/bin/env python3
"""Hereditary-Haematological-Malignancy-Predisposition-Atlas — Complete 8-Gene Inherited Haematological
Cancer Predisposition Atlas.

RUNX1   (RUNX family transcription factor 1; 453 aa; 21q22.12; AD;
         Familial Platelet Disorder / AML predisposition (FPD-AML) — most common RUNX1-AML;
         30-44% lifetime AML/MDS risk; thrombocytopenia + platelet function defect;
         SIBLING DONOR TESTING MANDATORY before allogeneic HSCT;
         seed SEED_BASE+0).
CEBPA   (CCAAT/enhancer binding protein alpha; 358 aa; 19q13.11; AD/biallelic;
         Familial AML — germline N-terminal bZIP frame-shift → biallelic somatic second hit at C-terminal;
         ELN favourable prognosis; HSCT in CR1 controversial (good prognosis without);
         seed SEED_BASE+1).
DDX41   (DEAD-box helicase 41; 622 aa; 5q35.3; AD;
         Most prevalent germline MDS/AML predisposition in adults; median onset 65 yr;
         Splice variants predominant (c.1574+1G>A); somatic R525H acquired at transformation;
         seed SEED_BASE+2).
TP53    (tumour protein p53; 393 aa; 17p13.1; AD;
         Li-Fraumeni syndrome → therapy-related AML / complex karyotype AML;
         Venetoclax POOR response; AVOID radiation in Li-Fraumeni; APR-246 investigational;
         seed SEED_BASE+3).
ETV6    (ETS variant transcription factor 6; 452 aa; 12p13.2; AD;
         Thrombocytopenia 5 (THRO5) + childhood ALL predisposition (30% lifetime);
         May mimic ITP — germline ETV6 often misdiagnosed as autoimmune thrombocytopenia;
         seed SEED_BASE+4).
ANKRD26 (ankyrin repeat domain 26; 1710 aa; 10p12.1; AD;
         Thrombocytopenia 2 (THRO2) — 5'UTR variants MISSED by standard WES;
         5-8% lifetime MDS/AML risk; specific 5'UTR sequencing MANDATORY;
         seed SEED_BASE+5).
SAMD9L  (sterile alpha motif domain containing 9-like; 1589 aa; 7q21.2; AD — GOF;
         Ataxia-Pancytopenia Syndrome (ATXPC) — cerebellar ataxia + bone marrow failure;
         Monosomy 7 PARADOXICALLY FAVOURABLE (somatic LOH removes GOF allele = reversion);
         Revertant mosaicism common — clinical improvement over time;
         seed SEED_BASE+6).
NF1     (neurofibromin 1; 2839 aa; 17q11.2; AD;
         NF1 — Neurofibromatosis type 1; JMML predisposition (30% of JMML has germline NF1);
         RAS-GAP loss-of-function → constitutive RAS-MAPK; trametinib investigational in JMML;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2022-2029).
"""

import random

SEED_BASE = 2022

HM_GENES = [
    # -- RUNX1 — Familial Platelet Disorder / AML predisposition (AD) ------------------
    {
        "gene": "RUNX1",
        "alt_name": "RUNX1 (RUNX Family TF-1 / AD — FPD-AML — Most Common Inherited RUNX1-AML — Sibling Donor Testing MANDATORY)",
        "protein": (
            "RUNX1 -- 21q22.12 AD -- RUNX1-453aa -- "
            "Familial-Platelet-Disorder-AML-FPD-AML -- "
            "30-44pct-Lifetime-AML-MDS-Risk -- "
            "Thrombocytopenia-Plus-Platelet-Function-Defect-Dense-Granule -- "
            "Sibling-Donor-RUNX1-Testing-MANDATORY-Before-HSCT"
        ),
        "locus": "21q22.12",
        "protein_size": "453 aa",
        "inheritance": "AD (autosomal dominant) — haploinsufficiency or dominant negative (missense in Runt domain)",
        "age_of_onset": (
            "Variable: thrombocytopenia present from birth; AML/MDS onset adolescence to adult (median 33 yr); "
            "Platelet count: 50-150 × 10⁹/L (mild-moderate thrombocytopenia); "
            "Platelet function: dense granule secretion defect → prolonged bleeding time despite adequate count; "
            "Bleeding: mucocutaneous (epistaxis, menorrhagia, bruising) proportionally worse than count suggests; "
            "AML transformation: 30-44% lifetime risk — often normal/intermediate karyotype; "
            "MDS: precursor state before AML in many cases; "
            "Cutaneous: eczematous lesions (in some kindreds); "
            "Misdiagnosis risk: immune thrombocytopenia (ITP) — steroid unresponsive thrombocytopenia = RUNX1 flag"
        ),
        "key_biomarker": (
            "CBC: mild-moderate thrombocytopenia (50-150 × 10⁹/L); normal platelet size (MPV normal); "
            "Platelet aggregation: reduced response to ADP, epinephrine; dense granule deficiency on electron microscopy; "
            "Bone marrow biopsy: megakaryocyte dysplasia; "
            "Flow cytometry: normal CD41/CD42 expression (distinguishes from Bernard-Soulier); "
            "WES/gene panel: RUNX1 germline sequencing — missense (Runt domain), nonsense, frameshift, deletion (MLPA); "
            "CBC trend: watch for falling count (MDS transformation signal); "
            "BM cytogenetics: if MDS suspected — monitor for -7, +8, +21"
        ),
        "pathognomonic": (
            "Familial thrombocytopenia + platelet function defect + AML in family = RUNX1-FPD; "
            "Dense granule deficiency on platelet EM — PATHOGNOMONIC for RUNX1 among hereditary thrombocytopenias; "
            "DISTINGUISH: MYH9 (giant platelets + Döhle bodies); Bernard-Soulier (low GPIb/CD42); ETV6 (ALL risk not AML); "
            "Steroid-unresponsive 'ITP' in multiple family members = RUNX1 germline until proven otherwise; "
            "Sibling donor MUST be tested for RUNX1 before HSCT donation — germline variant donor → HSCT failure"
        ),
        "treatment": (
            "Thrombocytopenia: avoid antiplatelet drugs (aspirin, NSAIDs, clopidogrel); DDAVP before procedures; "
            "No treatment needed if asymptomatic thrombocytopenia with adequate function; "
            "AML/MDS transformation: standard induction chemotherapy ± allogeneic HSCT; "
            "CRITICAL: sibling donor RUNX1 germline testing BEFORE donation — affected sibling → graft failure risk; "
            "Surveillance: annual CBC; bone marrow assessment if CBC declining; "
            "No proven chemoprevention for AML transformation; "
            "AVOID: chronic immunosuppression for misdiagnosed ITP — thrombocytopenia will not respond; "
            "Genetic counselling: AD — 50% risk; family cascade testing"
        ),
        "critical_flags": [
            "RUNX1-SIBLING-DONOR-TESTING-MANDATORY-BEFORE-HSCT",
            "RUNX1-30-44pct-LIFETIME-AML-RISK",
            "RUNX1-DENSE-GRANULE-DEFECT-PLATELET-FUNCTION",
            "RUNX1-MIMIC-ITP-STEROID-UNRESPONSIVE",
            "RUNX1-AVOID-ANTIPLATELET-ASPIRIN-NSAIDs",
            "RUNX1-ANNUAL-CBC-SURVEILLANCE",
            "RUNX1-MISDIAGNOSED-AS-IMMUNE-THROMBOCYTOPENIA",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- CEBPA — Familial AML (AD + biallelic somatic second hit) ----------------------
    {
        "gene": "CEBPA",
        "alt_name": "CEBPA (C/EBPalpha / AD Germline N-terminal + Somatic C-terminal — Familial AML — ELN Favourable — HSCT in CR1 Controversial)",
        "protein": (
            "CEBPA -- 19q13.11 AD -- CEBPA-358aa -- "
            "Familial-AML-Germline-N-terminal-bZIP-Frameshift -- "
            "Biallelic-CEBPA-AML-ELN-Favourable-Prognosis -- "
            "CD19-Positive-Distinctive-Immunophenotype -- "
            "Somatic-C-terminal-In-Frame-Mutation-Second-Hit-At-Transformation"
        ),
        "locus": "19q13.11",
        "protein_size": "358 aa",
        "inheritance": "AD (autosomal dominant) — germline N-terminal frameshift; biallelic AML = germline + somatic C-terminal second hit",
        "age_of_onset": (
            "AML onset: young adults (median ~20-25 yr for familial CEBPA); earlier than sporadic AML; "
            "Germline CEBPA: N-terminal frameshift → truncated form that acts as dominant-negative of p42-CEBPA; "
            "Somatic second hit at C-terminal bZIP domain acquired at AML transformation; "
            "Blood count: normal until AML; no thrombocytopenia prodrome (unlike RUNX1, ETV6); "
            "AML immunophenotype: CD13/CD33/MPO+ with CD19 POSITIVE — distinctive for biallelic CEBPA; "
            "Karyotype: normal or +8 (favourable); complex karyotype RARE in biallelic CEBPA; "
            "Prognosis: ELN favourable — 60-70% 5-year OS; better than most AML; "
            "Second AML (relapse or de novo): biallelic CEBPA can develop new AML (clonally distinct from relapse)"
        ),
        "key_biomarker": (
            "CBC: normal until AML onset (no prodrome); "
            "Bone marrow: AML with CD19-positive blasts — DISTINCTIVE immunophenotype; "
            "Cytogenetics: normal karyotype or +8 (favourable); "
            "CEBPA mutation analysis: distinguish N-terminal (germline candidate) vs C-terminal (somatic); "
            "Sequencing blood/skin for germline confirmation vs somatic-only biallelic; "
            "ELN 2022 risk: Favourable — biallelic CEBPA regardless of co-mutations (except KMT2A-PTD); "
            "VAF analysis: N-terminal germline clone present in all cells at ~50% VAF; "
            "Family history: multiple members with young-onset AML = familial CEBPA flag"
        ),
        "pathognomonic": (
            "Young-onset AML (< 30 yr) + CD19-positive blasts + biallelic CEBPA mutations = familial CEBPA AML; "
            "DISTINGUISH: sporadic biallelic CEBPA (both somatic, no family history) vs familial (germline N-terminal); "
            "CD19 expression in AML is unusual — biallelic CEBPA is one of the few AML subtypes with CD19+; "
            "N-terminal frameshift confirmed in germline (blood DNA when in CR = all AML blasts cleared); "
            "Family history of young AML with ELN favourable karyotype = CEBPA testing priority"
        ),
        "treatment": (
            "Induction: standard 7+3 (cytarabine + anthracycline); CR rate ~80-90%; "
            "Post-remission: HIGH-dose cytarabine consolidation (HDAC) × 3-4 cycles; "
            "HSCT in CR1: CONTROVERSIAL — biallelic CEBPA prognosis sufficiently good that consolidation alone may suffice; "
            "ELN recommendation: HSCT in CR1 NOT routinely recommended for biallelic CEBPA without adverse features; "
            "Second AML event: clonally distinct second primary — treat as new AML (not relapse); "
            "Surveillance: annual CBC post-remission; germline family members annual CBC + BM if CBC changes; "
            "Genetic counselling: AD germline — 50% risk; pre-natal testing available"
        ),
        "critical_flags": [
            "CEBPA-CD19-POSITIVE-BLASTS-DISTINCTIVE-BIALLELIC",
            "CEBPA-ELN-FAVOURABLE-BIALLELIC",
            "CEBPA-HSCT-CR1-CONTROVERSIAL-NOT-ROUTINELY-NEEDED",
            "CEBPA-GERMLINE-N-TERMINAL-FRAMESHIFT",
            "CEBPA-SECOND-AML-CLONALLY-DISTINCT-NOT-RELAPSE",
            "CEBPA-CONFIRM-GERMLINE-IN-CR-BLOOD-DNA",
            "CEBPA-YOUNG-ONSET-FAMILY-AML-SCREENING",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- DDX41 — Most prevalent germline AML in adults ---------------------------------
    {
        "gene": "DDX41",
        "alt_name": "DDX41 (DEAD-Box Helicase 41 / AD — Most Prevalent Germline MDS/AML Adults — Splice Variants — Late Onset 65yr — R525H Somatic Second Hit)",
        "protein": (
            "DDX41 -- 5q35.3 AD -- DDX41-622aa -- "
            "Most-Prevalent-Germline-MDS-AML-Adults-3-4pct-De-Novo-AML -- "
            "Median-Onset-65yr-Splice-Variants-c.1574plus1GA-Most-Common -- "
            "Somatic-pArg525His-Second-Hit-At-Transformation-Pathognomonic -- "
            "Nordic-European-Founder-Effect"
        ),
        "locus": "5q35.3",
        "protein_size": "622 aa",
        "inheritance": "AD (autosomal dominant) — loss-of-function splice site or truncating variants",
        "age_of_onset": (
            "Late adult onset: median age 65 years at diagnosis (oldest of all hereditary AML predisposition genes); "
            "Disease spectrum: MDS → AML; aplastic anaemia-like cytopenias in some; "
            "Cytopenias: insidious onset anaemia, thrombocytopenia, neutropenia — precede AML by months-years; "
            "AML transformation: ELN intermediate-adverse risk; complex karyotype less common than TP53; "
            "Splice variants: intronic splice site variants (c.1574+1G>A most common); may be missed by exon-only WES; "
            "Family history: multiple adults with AML/MDS — often attributed to age until DDX41 identified; "
            "Nordic/European ancestry: founder effect with c.1574+1G>A and p.Asp140Glyfs; "
            "Somatic second hit: p.Arg525His (R525H) acquired in AML blasts — not in germline"
        ),
        "key_biomarker": (
            "CBC: insidious cytopenias — anaemia first, then thrombocytopenia/neutropenia; "
            "BM biopsy: hypocellular or dysplastic — may mimic aplastic anaemia; "
            "Cytogenetics: normal or intermediate risk (unlike TP53 complex); "
            "DDX41 sequencing: INTRONIC splice variants — ENSURE panel covers splice sites; "
            "Somatic R525H: present in AML blasts, absent in germline — acquired second hit pathognomonic; "
            "BM cytogenetics at transformation: -7, del(5q) possible; "
            "Family history: obtain detailed haematological history of siblings/parents with MDS/AML; "
            "NGS panel: ensure 5q35.3 coverage including splice regions"
        ),
        "pathognomonic": (
            "Elderly adult with AML/MDS + DDX41 germline splice variant + somatic R525H second hit = DDX41 AML; "
            "Somatic p.Arg525His (R525H) in AML blasts confirms activated germline predisposition; "
            "DISTINGUISH from age-related clonal haematopoiesis (CHIP): DDX41 germline — family history, younger relatives; "
            "Aplastic anaemia-like presentation: DDX41 should be tested in aplastic anaemia not responding to IST; "
            "European founder variants: c.1574+1G>A and Asp140Glyfs in Nordic/European ancestry"
        ),
        "treatment": (
            "MDS treatment: based on IPSS-R risk; supportive care + azacitidine for higher-risk; "
            "AML induction: standard 7+3; response rates comparable to non-predisposition AML; "
            "Allogeneic HSCT: recommended for fit patients with MDS/AML; consider early if MDS progressing; "
            "Sibling donor: test for DDX41 germline — affected sibling unsuitable; "
            "Venetoclax + azacitidine: reasonable for unfit patients; response data limited in DDX41; "
            "Surveillance: annual CBC for germline carriers; low threshold for BM biopsy if cytopenias; "
            "Aplastic anaemia workup: include DDX41 if IST not responding; "
            "Genetic counselling: AD — 50% risk; offer cascade testing to adult family members"
        ),
        "critical_flags": [
            "DDX41-MOST-PREVALENT-GERMLINE-AML-ADULTS",
            "DDX41-SPLICE-VARIANTS-ENSURE-INTRONIC-COVERAGE",
            "DDX41-SOMATIC-R525H-PATHOGNOMONIC-SECOND-HIT",
            "DDX41-LATE-ONSET-65yr-MEDIAN",
            "DDX41-SIBLING-DONOR-TEST-BEFORE-HSCT",
            "DDX41-APLASTIC-ANAEMIA-PHENOTYPE-TEST-IF-IST-FAILS",
            "DDX41-NORDIC-EUROPEAN-FOUNDER",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- TP53 — Li-Fraumeni + therapy-related AML / complex karyotype ------------------
    {
        "gene": "TP53",
        "alt_name": "TP53 (Tumour Protein p53 / AD — Li-Fraumeni — Therapy-Related AML — Complex Karyotype — Venetoclax POOR — AVOID Radiation)",
        "protein": (
            "TP53 -- 17p13.1 AD -- TP53-393aa -- "
            "Li-Fraumeni-Syndrome-LFS-Multiple-Cancer-Predisposition -- "
            "Therapy-Related-AML-MDS-Complex-Karyotype-tAML -- "
            "Venetoclax-Response-POOR-TP53-AML -- "
            "APR-246-Eprenetapopt-Investigational-Refolding-Mutant-p53"
        ),
        "locus": "17p13.1",
        "protein_size": "393 aa",
        "inheritance": "AD (autosomal dominant) — Li-Fraumeni; AR (biallelic) — Li-Fraumeni 2 (very rare)",
        "age_of_onset": (
            "Li-Fraumeni syndrome: broad cancer spectrum — sarcoma, breast, brain, adrenocortical carcinoma, leukaemia; "
            "Leukaemia in LFS: 3-4% of childhood AML has germline TP53; therapy-related AML after prior cancer treatment; "
            "Therapy-related AML/MDS: del17p, complex karyotype (≥3 abnormalities) — TP53 signature; "
            "Monosomal karyotype: -17/del17p — TP53 biallelic loss = worst AML prognosis; "
            "Radiation sensitivity: Li-Fraumeni patients — AVOID therapeutic radiation → field cancers; "
            "AVOID annual whole-body CT in LFS — cumulative radiation risk; use MRI surveillance protocol; "
            "Multiple primary cancers: sequential malignancies over lifetime — each should prompt TP53 germline test"
        ),
        "key_biomarker": (
            "CBC: cytopenias at therapy-related AML onset; "
            "BM cytogenetics: del17p / complex karyotype (≥3 abnormalities) — TP53 monosomal karyotype; "
            "TP53 IHC: nuclear overexpression (missense) or absence (null) on BM trephine; "
            "NGS: TP53 sequencing — hotspot residues (R175H, R248W, R248Q, R273H, R282W); "
            "VAF: TP53 variant in AML — distinguish 1 hit (heterozygous) vs 2 hit (biallelic LOH or compound het); "
            "Germline confirmation: skin biopsy / constitutional DNA outside tumour cells; "
            "Prior cancer history: sarcoma, breast cancer, CNS tumour, adrenocortical carcinoma + AML → LFS flag; "
            "Therapy-related flag: AML after alkylating agents / topoisomerase II inhibitors / radiation"
        ),
        "pathognomonic": (
            "Therapy-related AML + del17p/complex karyotype = TP53 until proven otherwise; "
            "Li-Fraumeni: young adult sarcoma + family with multiple cancers at young ages = TP53 germline; "
            "Monosomal karyotype (-17 + one or more nullisomic chromosomes) — ELN adverse — TP53 biallelic; "
            "DISTINGUISH: TP53 somatic (acquired in AML) vs germline (LFS) — germline present in ALL cells; "
            "Adrenocortical carcinoma in child + family cancer history = LFS — TP53 highest priority"
        ),
        "treatment": (
            "AML induction: standard 7+3 or decitabine — CR rates reduced in TP53 AML vs other AML; "
            "Venetoclax (BCL2 inhibitor): POOR response in TP53 AML — TP53 confers venetoclax resistance; "
            "APR-246 (eprenetapopt): investigational — refolds mutant p53 → EMERALD-1 trial; "
            "Decitabine + APR-246: Phase III data in TP53 MDS/AML — some benefit; "
            "HSCT: modest benefit in TP53 AML — relapse rate high even after allogeneic HSCT; "
            "Li-Fraumeni surveillance: Toronto Protocol — biannual whole-body MRI (NOT CT), annual brain MRI, skin; "
            "AVOID radiation therapy in LFS where alternatives exist — radiation-induced secondary cancers; "
            "Genetic counselling: AD — 50% risk; cascade family testing; pre-natal option"
        ),
        "critical_flags": [
            "TP53-VENETOCLAX-POOR-RESPONSE-RESISTANCE",
            "TP53-AVOID-RADIATION-LI-FRAUMENI",
            "TP53-COMPLEX-KARYOTYPE-THERAPY-RELATED-AML",
            "TP53-APR-246-EPRENETAPOPT-INVESTIGATIONAL",
            "TP53-MONOSOMAL-KARYOTYPE-ELN-ADVERSE",
            "TP53-LFS-SURVEILLANCE-MRI-NOT-CT",
            "TP53-GERMLINE-IN-ALL-CELLS-SKIN-BIOPSY-CONFIRM",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- ETV6 — Thrombocytopenia 5 + childhood ALL predisposition ----------------------
    {
        "gene": "ETV6",
        "alt_name": "ETV6 (ETS Variant TF-6 / AD — Thrombocytopenia-5 — 30pct Childhood ALL Risk — ITP Mimic — ETS Domain Hotspot)",
        "protein": (
            "ETV6 -- 12p13.2 AD -- ETV6-452aa -- "
            "Thrombocytopenia-5-THRO5-Plus-ALL-Predisposition -- "
            "30pct-Lifetime-ALL-Risk-Predominantly-B-Lineage -- "
            "Misdiagnosed-As-Immune-Thrombocytopenia-ITP -- "
            "ETS-Domain-Hotspot-Mutations-p.P214L-p.R358X"
        ),
        "locus": "12p13.2",
        "protein_size": "452 aa",
        "inheritance": "AD (autosomal dominant) — ETS domain missense or truncating variants",
        "age_of_onset": (
            "Thrombocytopenia: congenital — present from birth, often discovered incidentally; "
            "Platelet count: 50-150 × 10⁹/L (mild thrombocytopenia); normal platelet size; "
            "Bleeding: usually mild — may present as prolonged epistaxis, surgical bleeding; "
            "ALL onset: childhood (peak 2-15 yr); B-lineage ALL; ~30% lifetime risk of ALL; "
            "ITP misdiagnosis: thrombocytopenia alone initially; autoimmune markers negative; steroid non-response; "
            "Platelet morphology: normal (no giant platelets — distinguishes from MYH9-RD or BSS); "
            "ETV6-RUNX1 translocation in sporadic ALL is SOMATIC (t(12;21)) — DIFFERENT from germline ETV6; "
            "Note: somatic ETV6-RUNX1 fusion is the MOST COMMON somatic alteration in childhood ALL but is not related to germline ETV6"
        ),
        "key_biomarker": (
            "CBC: mild thrombocytopenia (50-150 × 10⁹/L); normal platelet volume (MPV); "
            "Platelet aggregation: mild dysfunction in some; "
            "Autoimmune markers: ANA, anti-platelet antibodies — NEGATIVE (distinguishes from ITP); "
            "Flow cytometry: normal CD41/CD42 on platelets; "
            "ETV6 sequencing: germline — ETS domain (codons 206-269) hotspot; "
            "Family history: multiple family members with thrombocytopenia ± childhood ALL; "
            "ALL diagnosis: standard BM aspirate + cerebrospinal fluid; ETV6-RUNX1 PCR in ALL blasts (somatic); "
            "No correlation between platelet count and ALL risk"
        ),
        "pathognomonic": (
            "Familial thrombocytopenia + childhood ALL in family = ETV6 germline until proven otherwise; "
            "DISTINGUISH from ITP: ETV6 thrombocytopenia = steroid-unresponsive; normal platelet antibody tests; "
            "DISTINGUISH germline ETV6 LOF from somatic ETV6-RUNX1 fusion in childhood ALL (somatic, not predisposition); "
            "Autosomal dominant thrombocytopenia with normal platelet size + ALL family history = ETV6 flag; "
            "ETV6 ETS domain variants most pathogenic: P214L, R358X recurrent hotspots"
        ),
        "treatment": (
            "Thrombocytopenia: usually no treatment needed if mild; avoid antiplatelet drugs; "
            "IVIg/steroids: INEFFECTIVE for ETV6 thrombocytopenia (not immune-mediated); "
            "Do NOT administer long-term steroids for presumed ITP — causes harm without benefit; "
            "ALL treatment: standard paediatric ALL protocol (BFM/COG); ETV6 germline does not alter treatment; "
            "After ALL remission: continued surveillance for relapse; germline ETV6 does not increase relapse risk; "
            "Surveillance: annual CBC for platelet trend; low threshold for BM if counts dropping; "
            "Siblings: cascade testing — offer haematology review + ETV6 germline testing; "
            "Genetic counselling: AD — 50% risk; discuss ALL surveillance plan with family"
        ),
        "critical_flags": [
            "ETV6-30pct-LIFETIME-ALL-RISK-B-LINEAGE",
            "ETV6-ITP-MIMIC-STEROID-UNRESPONSIVE",
            "ETV6-DO-NOT-GIVE-STEROIDS-FOR-THROMBOCYTOPENIA",
            "ETV6-GERMLINE-NOT-SOMATIC-ETV6-RUNX1-FUSION",
            "ETV6-ETS-DOMAIN-P214L-R358X-HOTSPOT",
            "ETV6-ANNUAL-CBC-SURVEILLANCE",
            "ETV6-CHILDHOOD-ALL-SURVEILLANCE-PLAN-MANDATORY",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- ANKRD26 — Thrombocytopenia 2 + MDS risk (5'UTR — WES MISSES) -----------------
    {
        "gene": "ANKRD26",
        "alt_name": "ANKRD26 (Ankyrin Repeat Domain-26 / AD — Thrombocytopenia-2 — 5'UTR Variants WES MISSES — Targeted Sequencing MANDATORY — 5-8pct MDS Risk)",
        "protein": (
            "ANKRD26 -- 10p12.1 AD -- ANKRD26-1710aa -- "
            "Thrombocytopenia-2-THRO2 -- "
            "5prime-UTR-Variants-c-127AT-c-128GA-RUNX1-Binding-Site -- "
            "Standard-WES-Exon-Capture-MISSES-5prime-UTR-TARGETED-SEQUENCING-MANDATORY -- "
            "5-8pct-Lifetime-MDS-AML-Risk"
        ),
        "locus": "10p12.1",
        "protein_size": "1710 aa",
        "inheritance": "AD (autosomal dominant) — 5'UTR promoter region point variants (not coding exon variants)",
        "age_of_onset": (
            "Thrombocytopenia: congenital; mild — platelet count 50-150 × 10⁹/L; "
            "Normal platelet size (distinguishes from MYH9, Bernard-Soulier); "
            "Bleeding: mild mucocutaneous; often asymptomatic; "
            "MDS/AML risk: 5-8% lifetime — moderate; onset adult; "
            "KEY DIAGNOSTIC TRAP: ANKRD26 5'UTR variants are in the PROMOTER region, "
            "NOT in coding exons → standard WES which captures exons will MISS these variants; "
            "Mechanism: 5'UTR variants disrupt RUNX1 binding site → de-repression of ANKRD26 in megakaryocytes → "
            "impaired terminal megakaryocyte differentiation → thrombocytopenia; "
            "Normal platelet function (aggregation normal — distinguishes from RUNX1)"
        ),
        "key_biomarker": (
            "CBC: mild stable thrombocytopenia (50-150 × 10⁹/L); normal MPV; "
            "Platelet function tests: NORMAL (aggregation intact — unlike RUNX1); "
            "ANKRD26 TARGETED sequencing of 5'UTR region: c.-127A>T or c.-128G>A; "
            "Standard WES WILL NOT detect: must specifically request 5'UTR sequencing or targeted panel; "
            "Family history: multigenerational mild thrombocytopenia; adult MDS/AML in affected members; "
            "BM biopsy if MDS suspected: megakaryocyte hyperplasia + dysplasia; "
            "No specific laboratory marker for MDS risk beyond CBC trend"
        ),
        "pathognomonic": (
            "Familial mild thrombocytopenia + NORMAL platelet function + NORMAL platelet size = ANKRD26 (not RUNX1); "
            "5'UTR c.-127A>T or c.-128G>A variants — only detectable with targeted 5'UTR sequencing; "
            "CRITICAL TRAP: patient told 'WES is negative' but ANKRD26 5'UTR was NEVER sequenced; "
            "DISTINGUISH from RUNX1: platelet function NORMAL in ANKRD26; impaired in RUNX1; "
            "DISTINGUISH from ETV6: ETV6 has ALL risk; ANKRD26 has MDS/AML risk; platelet function normal in both"
        ),
        "treatment": (
            "Thrombocytopenia: usually no treatment — mild and stable; avoid antiplatelet agents; "
            "Platelet transfusion for surgical procedures if count <50 × 10⁹/L; "
            "Eltrombopag (TPO-RA): may temporarily raise count pre-procedure — use cautiously; "
            "MDS/AML surveillance: annual CBC; BM biopsy if CBC declining; "
            "MDS treatment: standard MDS protocol if transformation occurs; "
            "HSCT eligibility: based on IPSS-R risk at MDS diagnosis; "
            "Genetic counselling: AD — 50% risk; emphasise need for TARGETED 5'UTR sequencing not standard WES; "
            "Family cascade: all first-degree relatives — targeted ANKRD26 5'UTR sequencing"
        ),
        "critical_flags": [
            "ANKRD26-5PRIME-UTR-STANDARD-WES-MISSES-TARGETED-SEQUENCING-MANDATORY",
            "ANKRD26-5-8pct-MDS-AML-RISK",
            "ANKRD26-NORMAL-PLATELET-FUNCTION-UNLIKE-RUNX1",
            "ANKRD26-c-127AT-c-128GA-RUNX1-BINDING-SITE",
            "ANKRD26-ANNUAL-CBC-SURVEILLANCE",
            "ANKRD26-DO-NOT-RELY-ON-WES-ALONE",
            "ANKRD26-ELTROMBOPAG-CAUTIOUS-SHORT-TERM-ONLY",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- SAMD9L — Ataxia-Pancytopenia + Monosomy 7 Reversion --------------------------
    {
        "gene": "SAMD9L",
        "alt_name": "SAMD9L (Sterile Alpha Motif Domain 9-Like / AD GOF — Ataxia-Pancytopathy — Monosomy-7 PARADOXICALLY FAVOURABLE Reversion — Revertant Mosaicism Common)",
        "protein": (
            "SAMD9L -- 7q21.2 AD-GOF -- SAMD9L-1589aa -- "
            "Ataxia-Pancytopenia-Syndrome-ATXPC -- "
            "Cerebellar-Ataxia-Plus-Bone-Marrow-Failure -- "
            "Monosomy-7-PARADOXICALLY-FAVOURABLE-Somatic-LOH-Removes-GOF-Allele -- "
            "Revertant-Mosaicism-Clinical-Improvement-Over-Time"
        ),
        "locus": "7q21.2",
        "protein_size": "1589 aa",
        "inheritance": "AD (autosomal dominant) — gain-of-function (GOF) → hyperactivated antiproliferative signalling",
        "age_of_onset": (
            "Cerebellar ataxia: childhood onset (1-5 yr); progressive gait ataxia, dysarthria, nystagmus; "
            "Pancytopenia: concomitant or may lag behind neurological features; anaemia, neutropenia, thrombocytopenia; "
            "MDS risk: ~30% develop MDS with monosomy 7 as most common cytogenetic finding; "
            "MDS severity: variable — some resolve spontaneously (revertant mosaicism); "
            "Monosomy 7 paradox: normally poor prognosis → in SAMD9L, monosomy 7 represents LOH of mutant allele → "
            "haematopoietic clones lose GOF SAMD9L → grow selectively (reversion = 'escape from disease'); "
            "Revertant mosaicism: acquired somatic mutations in SAMD9L that neutralise GOF effect → BM recovery; "
            "SAMD9 (not SAMD9L): same locus area, but SAMD9 → MIRAGE syndrome (no ataxia, more severe systemic features)"
        ),
        "key_biomarker": (
            "CBC: pancytopenia — anaemia + neutropenia + thrombocytopenia; variable severity; "
            "BM biopsy: hypocellular; may show MDS features; "
            "BM cytogenetics: monosomy 7 (-7) — PARADOXICALLY FAVOURABLE in SAMD9L (mechanism = reversion); "
            "Neurological: MRI cerebellum — atrophy progressive; "
            "SAMD9L sequencing: GOF variants predominantly in exons encoding STAND domain; "
            "Revertant mosaicism marker: blood VAF < skin/buccal VAF (reversion enriched in blood); "
            "SAMD9 vs SAMD9L: both at 7q21.2, both test in same panel — different phenotypes; "
            "Inflammatory markers: CRP, ferritin (HLH-like features in some patients)"
        ),
        "pathognomonic": (
            "Childhood cerebellar ataxia + pancytopenia + monosomy 7 = SAMD9L until proven otherwise; "
            "Monosomy 7 that improves over time (reversion) + ataxia + pancytopenia = SAMD9L pathognomonic; "
            "DISTINGUISH from SAMD9: SAMD9L has cerebellar ataxia (SAMD9 does NOT); "
            "DISTINGUISH from GATA2 deficiency: GATA2 → monocytopenia + mycobacteria + lymphedema (not ataxia); "
            "Revertant mosaicism: blood-derived cells show lower VAF than constitutional cells → IMPROVEMENT over time"
        ),
        "treatment": (
            "Pancytopenia/MDS: supportive care (G-CSF, transfusions, EPO if isolated anaemia); "
            "HSCT: considered for severe MDS or pancytopenia not improving — monosomy 7 alone NOT mandatory HSCT trigger; "
            "Monosomy 7: WAIT before HSCT if clinical improvement (reversion possible); "
            "OBSERVE monosomy 7 clones — can spontaneously remit via revertant mosaicism (unique to SAMD9L); "
            "Cerebellar ataxia: no disease-modifying therapy; physiotherapy + speech therapy; "
            "Sibling donor: SAMD9L germline testing mandatory before donation; "
            "HLH-like episodes: consider HLH protocol (dexamethasone + etoposide) if meets HLH criteria; "
            "Genetic counselling: AD GOF — 50% risk; germline testing of family"
        ),
        "critical_flags": [
            "SAMD9L-MONOSOMY-7-PARADOXICALLY-FAVOURABLE-REVERSION",
            "SAMD9L-REVERTANT-MOSAICISM-CLINICAL-IMPROVEMENT",
            "SAMD9L-CEREBELLAR-ATAXIA-DISTINGUISHES-FROM-SAMD9",
            "SAMD9L-DO-NOT-RUSH-HSCT-MONOSOMY-7-MAY-REVERT",
            "SAMD9L-SIBLING-DONOR-TESTING-MANDATORY",
            "SAMD9L-DISTINGUISH-FROM-GATA2-NO-MONOCYTOPENIA",
            "SAMD9L-GOF-ANTIPROLIFERATIVE-HAEMATOPOIESIS-SUPPRESSED",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- NF1 — JMML predisposition (AD, RAS-GAP, RAS-MAPK) ----------------------------
    {
        "gene": "NF1",
        "alt_name": "NF1 (Neurofibromin-1 / AD — NF1 — JMML-Predisposition-30pct-Germline — RAS-GAP-Loss — Trametinib-Investigational — Monosomy-7-Most-Common-Cytogenetics)",
        "protein": (
            "NF1 -- 17q11.2 AD -- NF1-2839aa -- "
            "Neurofibromatosis-Type-1 -- "
            "JMML-Juvenile-Myelomonocytic-Leukaemia-30pct-Has-Germline-NF1 -- "
            "RAS-GAP-Loss-Constitutive-RAS-MAPK-Myeloid-Proliferation -- "
            "Trametinib-MEK-Inhibitor-Investigational-JMML-NF1"
        ),
        "locus": "17q11.2",
        "protein_size": "2839 aa",
        "inheritance": "AD (autosomal dominant) — haploinsufficiency; NF1-JMML requires biallelic somatic LOH of second allele",
        "age_of_onset": (
            "NF1 features: café-au-lait macules ≥6 (≥5mm pre-pubertal, ≥15mm post-pubertal) — from birth; "
            "Lisch nodules: iris hamartomas — seen from ~6 yr on slit lamp; "
            "Neurofibromas: plexiform (childhood), cutaneous (puberty onward); "
            "JMML onset: < 5 years (median 1.8 yr); "
            "JMML features: splenomegaly + monocytosis + circulating myeloid precursors + skin infiltrates; "
            "JMML diagnostic criteria: monocytes >1 × 10⁹/L + splenomegaly + BCR::ABL1 negative + somatic/germline RAS pathway; "
            "Monosomy 7: most common cytogenetic finding in NF1-JMML (~25%); "
            "Spontaneous resolution: VERY RARE — JMML in NF1 generally requires HSCT; "
            "Distinguishes from other RASopathy-JMML (PTPN11, KRAS, NRAS — sporadic JMML)"
        ),
        "key_biomarker": (
            "CBC: monocytosis (>1 × 10⁹/L) + thrombocytopenia + anaemia; "
            "BM biopsy: hypercellular with monocytic/myeloid proliferation; "
            "HbF: elevated (>10%) — JMML diagnostic criterion (foetal haemoglobin >10%); "
            "Cytogenetics: monosomy 7 (most common in NF1-JMML); normal or other; "
            "BCR::ABL1 PCR: NEGATIVE (mandatory exclusion — CML excluded by this); "
            "NF1 germline sequencing: large gene — ensure full gene sequencing (MLPA for deletion); "
            "Skin: 6+ café-au-lait macules; axillary/inguinal freckling; "
            "Ophthalmology: slit-lamp for Lisch nodules from age 6"
        ),
        "pathognomonic": (
            "Child <5 yr + ≥6 café-au-lait macules + monocytosis + splenomegaly = NF1-JMML; "
            "JMML + NF1 diagnosis = immediate referral to paediatric haematology/oncology; "
            "HbF >10% in child with NF1 + splenomegaly = JMML until proven otherwise; "
            "DISTINGUISH from CML: BCR::ABL1 negative; monocytosis predominates (not neutrophilia); "
            "DISTINGUISH from sporadic JMML (PTPN11/KRAS/NRAS somatic): no NF1 features in sporadic"
        ),
        "treatment": (
            "JMML definitive: allogeneic HSCT — only curative option; urgency depends on disease pace; "
            "Pre-HSCT: MEK inhibitor trametinib INVESTIGATIONAL (CoALL JMML-2015 protocol approach); "
            "13-cis-retinoic acid: post-HSCT maintenance — reduces relapse risk in JMML; "
            "Conditioning: myeloablative preferred; RIC (reduced-intensity) used in some centres for NF1-JMML; "
            "NF1 surveillance (non-JMML): annual ophthalmology, blood pressure, dermatology, neurology; "
            "Plexiform neurofibroma: selumetinib (MEK inhibitor) FDA-approved 2020 for symptomatic inoperable PNF in NF1; "
            "MPNST risk: lifetime 8-13% — surveillance imaging for deep plexiforms; "
            "Genetic counselling: AD NF1 — 50% risk; 50% de novo (no family history); pre-natal testing available"
        ),
        "critical_flags": [
            "NF1-JMML-HSCT-ONLY-CURATIVE-OPTION",
            "NF1-JMML-BCR-ABL1-NEGATIVE-MANDATORY-EXCLUSION",
            "NF1-HbF-ABOVE-10pct-JMML-CRITERION",
            "NF1-TRAMETINIB-INVESTIGATIONAL-JMML",
            "NF1-SELUMETINIB-FDA-2020-PLEXIFORM-NEUROFIBROMA",
            "NF1-MPNST-8-13pct-LIFETIME-RISK",
            "NF1-50pct-DE-NOVO-NO-FAMILY-HISTORY",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _make_cohort(gene_entry: dict, seed: int) -> list:
    rng = random.Random(seed)
    cohort = []
    gene = gene_entry["gene"]
    for i in range(40):
        age = rng.randint(1, 80)
        sex = rng.choice(["M", "F"])

        if gene == "RUNX1":
            thrombocytopenia     = True
            platelet_dysfunction = True   # dense granule defect always
            aml_mds              = rng.random() < 0.38   # 30-44% lifetime risk
            all_risk             = False
            jmml                 = False
            five_utr_missed      = False
            monosomy7            = rng.random() < 0.20
            revertant_mosaicism  = False
            ataxia               = False
            li_fraumeni          = False
            nf1_features         = False
            venetoclax_poor      = False
            sibling_donor_risk   = True
            hsct_required        = aml_mds and rng.random() < 0.50

        elif gene == "CEBPA":
            thrombocytopenia     = False
            platelet_dysfunction = False
            aml_mds              = True   # AML is defining
            all_risk             = False
            jmml                 = False
            five_utr_missed      = False
            monosomy7            = False
            revertant_mosaicism  = False
            ataxia               = False
            li_fraumeni          = False
            nf1_features         = False
            venetoclax_poor      = False
            sibling_donor_risk   = rng.random() < 0.50
            hsct_required        = rng.random() < 0.30   # controversial

        elif gene == "DDX41":
            thrombocytopenia     = rng.random() < 0.40
            platelet_dysfunction = False
            aml_mds              = rng.random() < 0.65   # most develop MDS/AML
            all_risk             = False
            jmml                 = False
            five_utr_missed      = False
            monosomy7            = rng.random() < 0.20
            revertant_mosaicism  = False
            ataxia               = False
            li_fraumeni          = False
            nf1_features         = False
            venetoclax_poor      = False
            sibling_donor_risk   = True
            hsct_required        = aml_mds and rng.random() < 0.55

        elif gene == "TP53":
            thrombocytopenia     = rng.random() < 0.30
            platelet_dysfunction = False
            aml_mds              = rng.random() < 0.55   # therapy-related
            all_risk             = False
            jmml                 = False
            five_utr_missed      = False
            monosomy7            = False
            revertant_mosaicism  = False
            ataxia               = False
            li_fraumeni          = True   # all have LFS
            nf1_features         = False
            venetoclax_poor      = aml_mds   # all TP53-AML → venetoclax poor
            sibling_donor_risk   = rng.random() < 0.50
            hsct_required        = aml_mds and rng.random() < 0.40

        elif gene == "ETV6":
            thrombocytopenia     = True
            platelet_dysfunction = False   # normal function
            aml_mds              = False
            all_risk             = rng.random() < 0.30
            jmml                 = False
            five_utr_missed      = False
            monosomy7            = False
            revertant_mosaicism  = False
            ataxia               = False
            li_fraumeni          = False
            nf1_features         = False
            venetoclax_poor      = False
            sibling_donor_risk   = False
            hsct_required        = all_risk and rng.random() < 0.40

        elif gene == "ANKRD26":
            thrombocytopenia     = True
            platelet_dysfunction = False   # normal function
            aml_mds              = rng.random() < 0.07   # 5-8% lifetime
            all_risk             = False
            jmml                 = False
            five_utr_missed      = True   # always — standard WES misses
            monosomy7            = False
            revertant_mosaicism  = False
            ataxia               = False
            li_fraumeni          = False
            nf1_features         = False
            venetoclax_poor      = False
            sibling_donor_risk   = False
            hsct_required        = aml_mds and rng.random() < 0.40

        elif gene == "SAMD9L":
            thrombocytopenia     = rng.random() < 0.80   # pancytopenia
            platelet_dysfunction = False
            aml_mds              = rng.random() < 0.30
            all_risk             = False
            jmml                 = False
            five_utr_missed      = False
            monosomy7            = rng.random() < 0.60   # common but paradoxically good
            revertant_mosaicism  = rng.random() < 0.45
            ataxia               = True   # cerebellar ataxia always
            li_fraumeni          = False
            nf1_features         = False
            venetoclax_poor      = False
            sibling_donor_risk   = rng.random() < 0.50
            hsct_required        = aml_mds and not revertant_mosaicism and rng.random() < 0.60

        else:  # NF1
            thrombocytopenia     = rng.random() < 0.60   # JMML
            platelet_dysfunction = False
            aml_mds              = False
            all_risk             = False
            jmml                 = rng.random() < 0.05   # 5% of NF1 children
            five_utr_missed      = False
            monosomy7            = jmml and rng.random() < 0.25
            revertant_mosaicism  = False
            ataxia               = False
            li_fraumeni          = False
            nf1_features         = True   # always
            venetoclax_poor      = False
            sibling_donor_risk   = rng.random() < 0.50
            hsct_required        = jmml   # JMML → HSCT mandatory

        cohort.append({
            "patient_id":            f"{gene}-{i+1:03d}",
            "age":                   age,
            "sex":                   sex,
            "gene":                  gene,
            "thrombocytopenia":      thrombocytopenia,
            "platelet_dysfunction":  platelet_dysfunction,
            "aml_mds":               aml_mds,
            "all_risk":              all_risk,
            "jmml":                  jmml,
            "five_utr_missed":       five_utr_missed,
            "monosomy7":             monosomy7,
            "revertant_mosaicism":   revertant_mosaicism,
            "ataxia":                ataxia,
            "li_fraumeni":           li_fraumeni,
            "nf1_features":          nf1_features,
            "venetoclax_poor":       venetoclax_poor,
            "sibling_donor_risk":    sibling_donor_risk,
            "hsct_required":         hsct_required,
        })
    return cohort


def _generate_cohort(gene_entry: dict) -> list:
    return _make_cohort(gene_entry, gene_entry["seed"])


def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in HM_GENES]
    all_pts = [p for c in all_cohorts for p in c]
    total = len(all_pts)

    def N(key): return sum(1 for p in all_pts if p[key])

    return {
        "atlas": "Hereditary-Haematological-Malignancy-Predisposition-Atlas",
        "subtitle": (
            "Complete 8-Gene Inherited Haematological Cancer Predisposition Atlas: "
            "RUNX1 (FPD-AML — sibling donor testing mandatory) + "
            "CEBPA (Familial AML — ELN favourable — HSCT in CR1 controversial) + "
            "DDX41 (Most prevalent germline MDS/AML adults — splice variants — R525H second hit) + "
            "TP53 (Li-Fraumeni — therapy-related AML — venetoclax POOR — avoid radiation) + "
            "ETV6 (Thrombocytopenia-5 — 30% childhood ALL risk — ITP mimic) + "
            "ANKRD26 (THRO2 — 5'UTR WES MISSES — targeted sequencing mandatory) + "
            "SAMD9L (ATXPC — monosomy 7 paradoxically favourable — revertant mosaicism) + "
            "NF1 (JMML — 30% germline NF1 — trametinib investigational — selumetinib plexiform)"
        ),
        "genes": [g["gene"] for g in HM_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}–{SEED_BASE + len(HM_GENES) - 1}",
        "thrombocytopenia_patients":     N("thrombocytopenia"),
        "platelet_dysfunction_patients": N("platelet_dysfunction"),
        "aml_mds_patients":              N("aml_mds"),
        "all_risk_patients":             N("all_risk"),
        "jmml_patients":                 N("jmml"),
        "five_utr_missed_patients":      N("five_utr_missed"),
        "monosomy7_patients":            N("monosomy7"),
        "revertant_mosaicism_patients":  N("revertant_mosaicism"),
        "ataxia_patients":               N("ataxia"),
        "li_fraumeni_patients":          N("li_fraumeni"),
        "nf1_features_patients":         N("nf1_features"),
        "venetoclax_poor_patients":      N("venetoclax_poor"),
        "sibling_donor_risk_patients":   N("sibling_donor_risk"),
        "hsct_required_patients":        N("hsct_required"),
        "gene_patient_counts": {g["gene"]: 40 for g in HM_GENES},
        "pathway": (
            "Hereditary haematological malignancy predisposition — shared mechanism: "
            "germline variants in transcription factors (RUNX1, CEBPA, ETV6), tumour suppressors (TP53, NF1), "
            "RNA helicases (DDX41), antiproliferative regulators (SAMD9L), or scaffold proteins (ANKRD26) "
            "impair haematopoietic differentiation or tumour suppression, predisposing to leukaemia/MDS. "
            "RUNX1: master haematopoietic TF; haploinsufficiency → megakaryocyte/platelet defect + AML. "
            "CEBPA: myeloid differentiation TF; germline N-terminal + somatic C-terminal → biallelic AML. "
            "DDX41: RNA helicase/innate immunity; loss → impaired RNA surveillance → myeloid transformation. "
            "TP53: cell cycle checkpoint; loss → genome instability → complex karyotype AML. "
            "ETV6: ETS repressor; loss → transcriptional de-repression → B-ALL + megakaryocyte defect. "
            "ANKRD26: 5'UTR de-repression → megakaryocyte dysfunction → MDS risk. "
            "SAMD9L: GOF → antiproliferative → haematopoietic failure; monosomy 7 = reversion escape. "
            "NF1: RAS-GAP; loss → constitutive RAS-MAPK → myeloid proliferation → JMML."
        ),
        "key_clinical_insight": (
            "RUNX1: sibling donor testing MANDATORY before HSCT — affected sibling donor → graft failure. "
            "CEBPA: CD19-positive AML blasts distinctive; HSCT in CR1 controversial (ELN favourable). "
            "DDX41: most prevalent germline MDS/AML in adults; splice variants; R525H somatic second hit pathognomonic. "
            "TP53: venetoclax POOR response; AVOID radiation in Li-Fraumeni; APR-246 investigational. "
            "ETV6: steroid-unresponsive ITP in family = ETV6 germline; 30% childhood ALL risk. "
            "ANKRD26: 5'UTR variants MISSED by standard WES — targeted 5'UTR sequencing MANDATORY. "
            "SAMD9L: monosomy 7 PARADOXICALLY FAVOURABLE (reversion); do NOT rush HSCT for monosomy 7 alone. "
            "NF1: JMML HSCT mandatory; selumetinib FDA-2020 for plexiform neurofibroma; MPNST 8-13% lifetime."
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in HM_GENES:
        cohort = _generate_cohort(gene_entry)
        gene = gene_entry["gene"]

        def pct(key):
            return round(100 * sum(1 for p in cohort if p[key]) / len(cohort))

        result[gene] = {
            "gene":                  gene,
            "alt_name":              gene_entry["alt_name"],
            "locus":                 gene_entry["locus"],
            "protein_size":          gene_entry["protein_size"],
            "inheritance":           gene_entry["inheritance"],
            "n_patients":            len(cohort),
            "thrombocytopenia_pct":     pct("thrombocytopenia"),
            "platelet_dysfunction_pct": pct("platelet_dysfunction"),
            "aml_mds_pct":              pct("aml_mds"),
            "all_risk_pct":             pct("all_risk"),
            "jmml_pct":                 pct("jmml"),
            "five_utr_missed_pct":      pct("five_utr_missed"),
            "monosomy7_pct":            pct("monosomy7"),
            "revertant_mosaicism_pct":  pct("revertant_mosaicism"),
            "ataxia_pct":               pct("ataxia"),
            "li_fraumeni_pct":          pct("li_fraumeni"),
            "nf1_features_pct":         pct("nf1_features"),
            "venetoclax_poor_pct":      pct("venetoclax_poor"),
            "sibling_donor_risk_pct":   pct("sibling_donor_risk"),
            "hsct_required_pct":        pct("hsct_required"),
            "age_of_onset":   gene_entry["age_of_onset"],
            "key_biomarker":  gene_entry["key_biomarker"],
            "pathognomonic":  gene_entry["pathognomonic"],
            "treatment":      gene_entry["treatment"],
            "critical_flags": gene_entry["critical_flags"],
            "seed":           gene_entry["seed"],
            "cohort_preview": cohort[:5],
        }
    return result


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Haematological-Malignancy-Predisposition-Atlas",
        "pathway": "Haematopoietic Transcription Factors / Tumour Suppression / RAS-MAPK / RNA Surveillance / Megakaryopoiesis",
        "shared_mechanism": (
            "Hereditary haematological malignancy predisposition syndromes share a final common pathway: "
            "germline defects in haematopoietic transcription factors, tumour suppressors, or signalling regulators "
            "impair normal blood cell differentiation or genome integrity, creating a fertile ground for "
            "subsequent somatic mutations that complete malignant transformation. "
            "Most follow a 'two-hit' model: germline variant provides the first hit (haploinsufficiency or dominant negative); "
            "somatic LOH or second mutation at the same locus provides the second hit at leukaemia onset. "
            "Exceptions: SAMD9L (GOF → somatic loss paradoxically beneficial); CEBPA (biallelic = germline N + somatic C). "
            "Surveillance mandates annual CBC in all germline carriers; sibling donor testing is critical "
            "because affected siblings used as HSCT donors risk graft failure or donor-derived leukaemia."
        ),
        "genes": {
            g["gene"]: {
                "full_name": g["alt_name"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "critical_flags": g["critical_flags"],
                "pathognomonic": g["pathognomonic"],
                "treatment_summary": g["treatment"],
            }
            for g in HM_GENES
        },
        "glossary": {
            "RUNX1-FPD": "Familial Platelet Disorder with predisposition to AML; germline RUNX1 haploinsufficiency; thrombocytopenia + platelet function defect + 30-44% lifetime AML/MDS risk; sibling donor testing mandatory",
            "Dense granule deficiency": "Platelet storage pool defect; reduced ADP and serotonin in dense granules; impaired secondary aggregation; PATHOGNOMONIC for RUNX1-FPD among hereditary thrombocytopenias; detected by platelet electron microscopy",
            "Sibling donor testing": "Before using a sibling as HSCT donor, ALL siblings of a hereditary haematological predisposition patient must be tested for the family variant; affected sibling donation risks donor-derived leukaemia or graft failure",
            "Biallelic CEBPA AML": "AML with two CEBPA mutations; germline families have N-terminal frameshift + somatic C-terminal in-frame; ELN 2022 Favourable risk category; CD19-positive immunophenotype distinctive",
            "DDX41 somatic R525H": "Somatic p.Arg525His variant acquired in AML blasts but NOT in germline; its presence in AML of a DDX41 germline carrier = confirmation of two-hit mechanism; pathognomonic second hit",
            "Li-Fraumeni syndrome (LFS)": "Germline TP53 — predisposition to sarcoma, breast cancer, CNS tumours, adrenocortical carcinoma, leukaemia; radiation AVOIDED; Toronto Protocol surveillance (whole-body MRI biannual, NOT CT)",
            "APR-246 (eprenetapopt)": "Small molecule that refolds mutant p53 protein back to wild-type conformation; investigational in TP53 AML/MDS; Phase III data in combination with azacitidine",
            "ETV6-RUNX1 translocation": "t(12;21)(p13;q22) — SOMATIC fusion in childhood B-ALL, not related to germline ETV6 LOF; most common chromosomal abnormality in childhood ALL; do NOT confuse with germline ETV6 predisposition",
            "ANKRD26 5'UTR": "Variants c.-127A>T and c.-128G>A in the PROMOTER/5'UTR region of ANKRD26; disrupt RUNX1 binding site → de-repression of ANKRD26 in megakaryocytes; MISSED by standard WES (exon-only capture); targeted 5'UTR panel mandatory",
            "SAMD9L GOF": "Gain-of-function variants in SAMD9L → hyperactivated SAMD9L protein → antiproliferative signal → suppressed haematopoietic stem cell expansion → aplasia/MDS",
            "Revertant mosaicism": "Somatic acquisition of second mutation that neutralises the pathogenic germline variant effect; clones with reversion outcompete pathogenic clones; leads to clinical improvement over time; common in SAMD9L",
            "Monosomy 7 reversion (SAMD9L)": "In SAMD9L GOF, loss of chromosome 7 (monosomy 7) in HSC removes the GOF allele (located at 7q21.2) via somatic LOH; these clones have normal SAMD9L → selective growth advantage; PARADOXICALLY FAVOURABLE in SAMD9L unlike other MDS",
            "JMML": "Juvenile Myelomonocytic Leukaemia — rare childhood myeloid malignancy; monocytes >1 × 10⁹/L + splenomegaly + BCR::ABL1 negative + RAS pathway variant (PTPN11, KRAS, NRAS, NF1, CBL germline/somatic); HSCT only curative option",
            "NF1 (neurofibromin)": "RAS-GTPase activating protein; inactivation → constitutive active RAS-GTP → MEK-ERK signalling → myeloid proliferation; NF1 feature: café-au-lait macules, Lisch nodules, neurofibromas",
            "Selumetinib (Koselugo)": "MEK1/2 inhibitor; FDA approved 2020 for symptomatic, inoperable plexiform neurofibromas in NF1 patients ≥2 years; does NOT treat JMML (trametinib investigational for that)",
            "Venetoclax resistance in TP53 AML": "TP53 mutations confer BCL-2 independence through p53-mediated upregulation of anti-apoptotic proteins or downregulation of pro-apoptotic BH3-only proteins; venetoclax + azacitidine response rates markedly lower in TP53 AML",
            "ELN 2022 risk classification": "European LeukemiaNet 2022 AML risk categories: Favourable (biallelic CEBPA, NPM1 without FLT3-ITDhigh, etc.), Intermediate, Adverse (TP53, complex karyotype, etc.); guides post-remission strategy",
            "HbF in JMML": "Fetal haemoglobin (HbF) >10% in child with JMML is a WHO diagnostic criterion; JMML blasts aberrantly maintain foetal haematopoiesis programme; NF1-JMML typically has elevated HbF",
            "THRO2 (Thrombocytopenia type 2)": "ANKRD26-related thrombocytopenia; mild platelet count reduction; normal platelet function and size; 5-8% MDS/AML risk; 5'UTR variants not detected by standard WES",
            "THRO5 (Thrombocytopenia type 5)": "ETV6-related thrombocytopenia; mild platelet count reduction; B-ALL predisposition; often misdiagnosed as immune thrombocytopenia; steroid non-responsive",
            "ATXPC": "Ataxia-Pancytopenia Syndrome (SAMD9L GOF); cerebellar ataxia + bone marrow failure + MDS with monosomy 7; revertant mosaicism and clinical improvement distinguish from other aplasias",
            "13-cis-retinoic acid": "Post-HSCT maintenance in JMML; reduces relapse risk; given for 1 year post-HSCT in many JMML protocols",
        },
        "surveillance_protocols": {
            "RUNX1": "Annual: CBC + differential; platelet function if surgical procedure planned; BM biopsy if platelet count falling; genetic counselling on sibling donor implications; avoid chronic steroids for 'ITP'",
            "CEBPA": "Annual CBC (post-remission); BM biopsy if CBC declining; family testing — first-degree relatives; distinguish germline from somatic in remission blood; monitor for second clonally distinct AML event",
            "DDX41": "Annual CBC; low threshold for BM biopsy if cytopenias develop; ensure splice-site intronic coverage on panel; sibling donor testing before HSCT; consider BM biopsy at diagnosis baseline",
            "TP53": "Toronto Protocol (LFS): biannual whole-body MRI (avoid CT/radiation); annual brain MRI + dermatology; avoid radiation therapy where alternative exists; annual haematology review; BM biopsy if cytopenias",
            "ETV6": "Annual CBC; childhood ALL surveillance if family ALL history; diagnose thrombocytopenia as ETV6 (NOT ITP) — avoid steroids; BM biopsy if count declining; haematology follow-up through childhood",
            "ANKRD26": "Annual CBC; targeted 5'UTR ANKRD26 sequencing for family members (not WES); BM biopsy if MDS suspected; MDS IPSS-R risk stratification if MDS develops; HSCT discussion for higher-risk MDS",
            "SAMD9L": "Annual CBC; BM biopsy + cytogenetics if pancytopenia worsening; MRI cerebellum (ataxia progression); OBSERVE monosomy 7 clones — reversion possible; neurology/physiotherapy for ataxia; HSCT only for severe MDS not reverting",
            "NF1": "Annual: ophthalmology (Lisch nodules), BP, dermatology, neurology, MRI for deep plexiforms; JMML surveillance: CBC + spleen size in children <5 yr with NF1; refer to paediatric haematology immediately if monocytosis + splenomegaly",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"AML/MDS patients: {ov['aml_mds_patients']}")
    print(f"Thrombocytopenia: {ov['thrombocytopenia_patients']}")
    print(f"HSCT required: {ov['hsct_required_patients']}")
    print(f"Sibling donor risk: {ov['sibling_donor_risk_patients']}")
    print(f"Venetoclax poor: {ov['venetoclax_poor_patients']}")
    print("Breakdown keys:", list(breakdown().keys()))
