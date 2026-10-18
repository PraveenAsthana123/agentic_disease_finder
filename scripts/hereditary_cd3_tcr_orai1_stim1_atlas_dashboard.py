#!/usr/bin/env python3
"""Hereditary-CD3-TCR-ORAI1-STIM1-Atlas -- Complete 8-Gene Reference
TCR-Complex-Assembly and Calcium-Signaling T-Cell Immunodeficiency
(CD3D/T-B+NK+-SCID-CD3delta-Complete-TCR-Absent-Surface-Flow-DIAGNOSTIC,
 CD3E/T-B+NK+-SCID-CD3epsilon-Morocco-Cys68Tyr-Founder-Identical-Phenotype-CD3D,
 CD3G/T-lymphopenia-PARTIAL-Milder-Residual-T-Cells-NOT-Complete-SCID,
 CD247/T-B+NK+-SCID-CD3zeta-Turkey-Caucasus-Founder-ULTRA-RARE,
 LCK/T-lymphopenia-Lympho-Proliferation-LOF-GOF-Same-Gene-OPPOSITE,
 LAT/T-B+NK+-SCID-Linker-Adapter-TCR-Signal-Relay-Scaffold-No-TCR-Downstream,
 ORAI1/CombinedID-Anhidrosis-Myopathy-SOCE-CRAC-ECTODERMAL-PATHOGNOMONIC,
 STIM1/CombinedID-Tubular-Myopathy-Anhidrosis-Cardiac-Arrhythmia-Ca2+sensor):

CD3D   (CD3 delta; 171aa; 11q23.3; AR;
         CD3δ is one of the four invariant subunits of the CD3 complex that associates with αβ TCR;
         CD3 complex: CD3δ + CD3ε heterodimer + CD3γ + CD3ε heterodimer + CD3ζ homodimer;
         CD3D LOF → absent surface CD3 complex → no TCR signalling → T cells absent from thymic selection;
         Immunophenotype: T-B+NK+ SCID (T cells absent, B and NK cells NORMAL — opposite of IL2RG);
         Diagnostic: absent CD3 on surface T-cell flow cytometry (CD3 flow is the KEY diagnostic test);
         BCG absolutely contraindicated from birth;
         HSCT curative; conditioning required;
         seed SEED_BASE+0).
CD3E   (CD3 epsilon; 207aa; 11q23.3; AR;
         CD3ε participates in TWO heterodimers: CD3δε and CD3γε — central to TCR complex assembly;
         CD3E LOF → absent surface CD3 complex → T-B+NK+ SCID (phenotypically identical to CD3D deficiency);
         Morocco founder variant: p.Cys68Tyr in North African Moroccan populations;
         Adjacent gene to CD3D on chromosome 11q23.3 — always sequence CD3D + CD3E + CD3G together (same locus);
         Surface CD3 absent on flow cytometry — diagnostic;
         Rare milder hypomorphic CD3E variants reported with partial T cells;
         HSCT curative;
         seed SEED_BASE+1).
CD3G   (CD3 gamma; 182aa; 11q23.3; AR;
         CD3γ is a component of the CD3γε heterodimer — part of the TCR signalling complex;
         CD3G LOF → MILDER phenotype than CD3D/CD3E — significant residual surface CD3 and T cells;
         Immunophenotype: T-lymphopenia (NOT complete SCID); partial T cells present but reduced CD4 cells;
         KEY DISTINCTION from CD3D/CD3E: NOT complete T-B+NK+ SCID — attenuated phenotype;
         CD3 present on flow (reduced, not absent) — critical DDx from CD3D/CD3E;
         Presentation may be later (childhood, not neonatal) due to partial T cell function;
         Adjacent to CD3D and CD3E on 11q23.3 — multiplex locus sequencing recommended;
         HSCT if severe; conservative management possible in mild cases;
         seed SEED_BASE+2).
CD247  (CD3 zeta; 164aa; 1q24.2; AR;
         CD3ζ (CD247) forms the homodimer (ζζ) within the TCR-CD3 complex — carries ITAM motifs;
         CD247 LOF → absent ζ-chain signalling → T-B+NK+ SCID (complete absent T cells);
         ULTRA-RARE: fewer than 30 families reported worldwide (2026);
         Turkey and Caucasus founder variants identified — p.Arg103Ter most common in Turkish kindreds;
         Surface CD3 absent on flow cytometry (same as CD3D/CD3E);
         PATHOGNOMONIC epidemiological context: T-B+NK+ SCID in consanguineous Turkish/Caucasian family;
         Isolated ζ-chain deficiency without CD3δ/ε/γ involvement confirmed by Western blot;
         HSCT curative; reduced-intensity conditioning acceptable;
         seed SEED_BASE+3).
LCK    (Lymphocyte-specific protein tyrosine kinase; 509aa; 1p35.1; AR/AD;
         Lck is the proximal tyrosine kinase that associates with CD4/CD8 co-receptors and initiates TCR signalling;
         Lck phosphorylates CD3 ITAMs → recruits ZAP-70 → activates TCR downstream pathway;
         LCK LOF → T-lymphopenia + reduced T-cell activation (MILDER phenotype, not complete SCID);
         LCK GOF (activating variants) → lymphoproliferation, EBV susceptibility, autoimmunity;
         OPPOSITE PHENOTYPES from same gene — LOF vs GOF requires sequencing to distinguish;
         CD4+ T cells disproportionately reduced (CD4 uses Lck more than CD8);
         T-cell activation assay (anti-CD3/anti-CD28 stimulation) reduced in LOF;
         Treatment: HSCT for severe LOF; JAK inhibitors under investigation for GOF;
         seed SEED_BASE+4).
LAT    (Linker for Activation of T cells; 262aa; 16q24.1; AR;
         LAT is a palmitoylated scaffold adapter protein that recruits PLCγ1, PI3K, GRB2, GADS after TCR stimulation;
         LAT is phosphorylated by ZAP-70 immediately downstream of CD3ζ → central relay node;
         LAT LOF → absent PLCγ1 activation → no calcium mobilisation → no NF-AT nuclear translocation → T cells fail to activate;
         Immunophenotype: T cells present in thymus but fail to signal → peripheral T-lymphopenia;
         ULTRA-RARE: fewer than 10 families reported; first described 2023;
         KEY CONCEPT: LAT deficiency mimics ZAP-70 deficiency (selective CD8 issue) in some hypomorphic alleles;
         Diagnosis: absent LAT protein by Western blot; absent TCR signalling on stimulation assay;
         HSCT likely curative based on murine models; gene therapy under investigation;
         seed SEED_BASE+5).
ORAI1  (ORAI1 calcium release-activated calcium channel protein 1; 301aa; 12q24.31; AR;
         ORAI1 is the pore-forming subunit of the CRAC (Ca2+ Release-Activated Ca2+) channel in T cells;
         Store-operated calcium entry (SOCE): ER calcium depletion → STIM1 aggregates → activates ORAI1 pore;
         ORAI1 LOF → absent calcium influx → absent NFAT activation → T cell anergy despite normal T-cell numbers;
         Immunophenotype: T present but NON-FUNCTIONAL (T lymphopenia + combined ID);
         ECTODERMAL DYSPLASIA COMPONENT PATHOGNOMONIC: anhidrosis (absent sweating) + hypoplastic nail + thin hair;
         Tubingen founder variant: p.Arg91Trp in German/Turkish populations;
         Myopathy: ORAI1 mutations cause myopathy — congenital myopathy in severely affected patients;
         BCG ABSOLUTELY CONTRAINDICATED;
         HSCT curative for immunodeficiency; myopathy partially ameliorated;
         seed SEED_BASE+6).
STIM1  (Stromal interaction molecule 1; 685aa; 11p15.5; AR;
         STIM1 is the ER-resident calcium sensor that activates ORAI1 upon ER calcium store depletion;
         STIM1 forms oligomers and migrates to ER-PM junctions to open ORAI1 pore → store-operated Ca2+ entry;
         STIM1 LOF → absent SOCE → T cell anergy (T cells present but cannot activate);
         Immunophenotype: combined ID (T cells present but non-functional);
         TUBULAR AGGREGATE MYOPATHY PATHOGNOMONIC: muscle biopsy shows tubular aggregates of sarcoplasmic reticulum;
         Anhidrosis: ectodermal involvement (absent sweating like ORAI1);
         CARDIAC ARRHYTHMIA RISK: STIM1 involved in cardiac calcium handling → QTc prolongation, arrhythmia;
         Annual ECG MANDATORY;
         STIM1 GOF → York platelet syndrome (thrombocytopenia + platelet aggregation defect);
         HSCT curative for immunodeficiency; tubular myopathy persists post-HSCT;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3950-3957)
"""
import random

SEED_BASE = 3950

ATLAS_GENES = [
    {
        "gene": "CD3D",
        "protein": (
            "CD3D -- 11q23.3 Autosomal-Recessive-LOF -- 171aa -- "
            "CD3delta-TCR-Complex-Subunit-19kDa-Disulfide-Bond-Cys23-Cys93-"
            "T-B+NK+-SCID-COMPLETE-T-Cell-Absent-Flow-CD3-Absent-DIAGNOSTIC-"
            "BCG-ABSOLUTELY-CI-HSCT-CURATIVE-OMIM-186790"
        ),
        "locus": "11q23.3",
        "protein_size": (
            "171 aa / 19 kDa / CD3D molecular context: "
            "STRUCTURE: "
            "  171 aa / 19 kDa; signal peptide aa 1-21; extracellular Ig-like domain; "
            "  transmembrane domain with charged residue (Asp80) for CD3 assembly; "
            "  cytoplasmic tail 9 aa (short); disulfide bond Cys23-Cys93 extracellular; "
            "  CD3δ pairs exclusively with CD3ε to form CD3δε heterodimer; "
            "IMMUNOPHENOTYPE (T-CELL FOCUS): "
            "  T cells: ABSENT (<1% CD3+ T cells); "
            "  B cells: NORMAL (present, functional); "
            "  NK cells: NORMAL (present); "
            "  Immunophenotype: T-B+NK+ SCID (opposite of γc-chain/JAK3 SCID); "
            "  Surface CD3 ABSENT by flow cytometry — DIAGNOSTIC TEST; "
            "  TREC = 0 on newborn screening; "
            "GENOTYPE-PHENOTYPE: "
            "  Complete null = T-B+NK+ SCID neonatal; "
            "  Hypomorphic = late-onset T-lymphopenia (childhood); "
            "KEY MANAGEMENT: "
            "  BCG absolutely contraindicated from birth (TREC 0 identifies pre-BCG); "
            "  PCP prophylaxis co-trimoxazole + CMV/RSV surveillance; "
            "  HSCT: curative; myeloablative conditioning required (unlike RAG/ADA); "
            "  Gene therapy under investigation (lentiviral CD3D); "
        ),
        "syndrome": "CD3δ deficiency — T-B+NK+ SCID",
        "inheritance": "AR LOF (autosomal recessive loss-of-function)",
        "immunophenotype": "T-B+NK+ SCID: T cells absent, B normal, NK normal",
        "t_cell_pct_mean": 0.4,
        "pathognomonic": "Surface CD3 ABSENT by flow cytometry in T-B+NK+ SCID infant — CD3D/CD3E/CD3G/CD247 panel mandatory",
        "key_avoid": "NEVER give BCG; NEVER give live vaccines; NEVER give non-irradiated blood products (GvHD); NEVER delay HSCT beyond 3.5 months of age",
        "key_rule": "T-B+NK+ SCID + absent surface CD3 (flow) = CD3 complex deficiency → sequence all 4 CD3 genes: CD3D-CD3E-CD3G-CD247. BCG absolutely contraindicated before TREC NBS result.",
        "surveillance": [
            "TREC NBS at birth — MANDATORY in all NBS programmes",
            "T/B/NK lymphocyte subset enumeration monthly until HSCT",
            "CMV/EBV/RSV surveillance biweekly",
            "CD3 surface expression flow cytometry at diagnosis",
            "Western blot CD3δ protein confirmation",
        ],
        "targeted_rx": "HSCT (myeloablative RIC); gene therapy investigational; no targeted small molecule therapy",
    },
    {
        "gene": "CD3E",
        "protein": (
            "CD3E -- 11q23.3 Autosomal-Recessive-LOF -- 207aa -- "
            "CD3epsilon-TCR-Complex-Two-Heterodimers-CD3deltaE-CD3gammaE-23kDa-"
            "T-B+NK+-SCID-Morocco-Cys68Tyr-Founder-North-African-"
            "BCG-ABSOLUTELY-CI-HSCT-CURATIVE-OMIM-186830"
        ),
        "locus": "11q23.3",
        "protein_size": (
            "207 aa / 23 kDa / CD3E molecular context: "
            "STRUCTURE: "
            "  207 aa / 23 kDa; participates in BOTH CD3δε and CD3γε heterodimers; "
            "  UNIQUE role: central to TCR complex assembly — linchpin of two heterodimers; "
            "  Extracellular Ig-like domain; charged TM residue Glu99 for assembly; "
            "  Cytoplasmic tail 5 aa (shortest CD3 subunit tail); "
            "IMMUNOPHENOTYPE: "
            "  T cells: ABSENT (<1% CD3+ T cells); B cells: NORMAL; NK cells: NORMAL; "
            "  T-B+NK+ SCID phenotypically identical to CD3D deficiency; "
            "  Surface CD3 ABSENT on flow cytometry; "
            "GENOTYPE-PHENOTYPE: "
            "  p.Cys68Tyr = Morocco/North African founder variant; "
            "  Null alleles = complete T-B+NK+ SCID; "
            "  Hypomorphic alleles = partial T cells + residual CD3 surface expression; "
            "KEY MANAGEMENT: "
            "  BCG absolutely CI; PCP prophylaxis mandatory; "
            "  Sequence CD3D+CD3E+CD3G+CD247 together (11q23.3 cluster); "
        ),
        "syndrome": "CD3ε deficiency — T-B+NK+ SCID",
        "inheritance": "AR LOF",
        "immunophenotype": "T-B+NK+ SCID: T cells absent, B normal, NK normal",
        "t_cell_pct_mean": 0.5,
        "pathognomonic": "Surface CD3 absent on flow + T-B+NK+ SCID + North African consanguineous family → p.Cys68Tyr Morocco founder",
        "key_avoid": "NEVER give BCG; NEVER give live vaccines; NEVER give non-irradiated blood; NEVER use CD3E alone without CD3D/G/Z panel",
        "key_rule": "CD3E and CD3D CLINICALLY IDENTICAL: phenotype cannot distinguish — gene panel mandatory. Morocco founder p.Cys68Tyr confirms CD3E in North African populations.",
        "surveillance": [
            "TREC NBS at birth",
            "T/B/NK flow monthly",
            "CMV/EBV/RSV PCR biweekly",
            "Genetic testing: CD3D-CD3E-CD3G-CD247 multiplex panel",
            "Western blot CD3ε protein",
        ],
        "targeted_rx": "HSCT (myeloablative RIC); gene therapy investigational",
    },
    {
        "gene": "CD3G",
        "protein": (
            "CD3G -- 11q23.3 Autosomal-Recessive-LOF -- 182aa -- "
            "CD3gamma-TCR-Complex-Partial-Defect-20kDa-MILDER-NOT-COMPLETE-SCID-"
            "T-lymphopenia-Residual-CD3-Surface-Flow-DIAGNOSTIC-DDx-CD3D-CD3E-"
            "BCG-CI-HSCT-IF-SEVERE-CONSERVATIVE-MILD-OMIM-186740"
        ),
        "locus": "11q23.3",
        "protein_size": (
            "182 aa / 20 kDa / CD3G molecular context: "
            "STRUCTURE: "
            "  182 aa / 20 kDa; forms CD3γε heterodimer; "
            "  PARTIALLY DISPENSABLE: γε heterodimer partially replaced by second δε — hence milder phenotype; "
            "  Charged TM residue Asp83 for CD3 complex assembly; "
            "  Cytoplasmic tail 44 aa with ITAM; "
            "IMMUNOPHENOTYPE: "
            "  T cells: REDUCED but NOT ABSENT (10-30% residual T cells); "
            "  B cells: NORMAL; NK cells: NORMAL; "
            "  CD3 surface REDUCED (not absent) — critical DDx from CD3D/CD3E; "
            "  Presentation LATER: childhood/adolescence rather than neonatal; "
            "GENOTYPE-PHENOTYPE: "
            "  MILDER than CD3D/CD3E: residual T cell function because γε partially replaced by second δε; "
            "KEY MANAGEMENT: "
            "  BCG contraindicated (T cell function reduced); "
            "  Conservative management possible in mild cases; "
            "  HSCT only if severe recurrent infections or T cells <10%; "
        ),
        "syndrome": "CD3γ deficiency — T-lymphopenia (partial, not complete SCID)",
        "inheritance": "AR LOF",
        "immunophenotype": "T-lymphopenia: T cells reduced (not absent), B normal, NK normal",
        "t_cell_pct_mean": 18.0,
        "pathognomonic": "T-lymphopenia with REDUCED (not absent) surface CD3 on flow — CD3G deficiency milder than CD3D/CD3E; adjacent locus 11q23.3",
        "key_avoid": "NEVER give BCG; NEVER give live vaccines; NEVER mistake reduced CD3 for CD3D/CD3E (absent CD3 = complete SCID); NEVER withhold HSCT if T cells <10%",
        "key_rule": "CD3G = MILDER: residual T cells + reduced (not absent) CD3 surface flow. Late presentation. HSCT threshold lower than CD3D/CD3E. Always sequence all 4 CD3 genes together.",
        "surveillance": [
            "Annual T/B/NK lymphocyte subset enumeration",
            "CD3 surface expression flow (reduced vs absent = key)",
            "T-cell mitogen response (PHA) annually",
            "CMV/EBV surveillance if T cells <20%",
            "Genetic panel: CD3D-CD3E-CD3G-CD247",
        ],
        "targeted_rx": "Conservative management if mild; HSCT if severe; no specific small molecule therapy",
    },
    {
        "gene": "CD247",
        "protein": (
            "CD247 -- 1q24.2 Autosomal-Recessive-LOF -- 164aa -- "
            "CD3zeta-ITAM-Homodimer-ZZ-18kDa-T-B+NK+-SCID-Turkey-Caucasus-Arg103Ter-Founder-"
            "ULTRA-RARE-LT30-Families-Worldwide-ABSOLUTELY-BCG-CI-HSCT-CURATIVE-OMIM-186780"
        ),
        "locus": "1q24.2",
        "protein_size": (
            "164 aa / 18 kDa / CD247 molecular context: "
            "STRUCTURE: "
            "  164 aa / 18 kDa; forms ζζ homodimer in TCR complex (most ITAM-rich component); "
            "  3 ITAMs in cytoplasmic tail (aa 112-164) — all 6 ITAM tyrosines are ZAP-70 substrates; "
            "  Transmembrane acidic residue (Asp9) for CD3 complex anchoring; "
            "  CD247 encodes ζζ and is also expressed on NK cells (FcεRIγ can partially substitute); "
            "IMMUNOPHENOTYPE: "
            "  T cells: ABSENT; B cells: NORMAL; NK cells: NORMAL/MILDLY REDUCED; "
            "  Surface CD3 ABSENT by flow; "
            "EPIDEMIOLOGY: "
            "  ULTRA-RARE: <30 families worldwide (2026); "
            "  Turkey/Caucasus: p.Arg103Ter founder variant most common in Turkish kindreds; "
            "  Consanguinity strongly associated; "
            "KEY MANAGEMENT: "
            "  BCG absolutely CI; "
            "  HSCT curative; reduced-intensity conditioning acceptable in some reports; "
        ),
        "syndrome": "CD3ζ deficiency — T-B+NK+ SCID (ultra-rare)",
        "inheritance": "AR LOF",
        "immunophenotype": "T-B+NK+ SCID: T cells absent, B normal, NK normal/reduced",
        "t_cell_pct_mean": 0.3,
        "pathognomonic": "T-B+NK+ SCID + absent surface CD3 + Turkish/Caucasian consanguineous family → p.Arg103Ter founder variant (CD247)",
        "key_avoid": "NEVER give BCG; NEVER rely on NK sparing to distinguish from other CD3 defects; NEVER skip Western blot to confirm ζ chain absence",
        "key_rule": "CD247 = ζ-chain SCID: T-B+NK+ SCID like CD3D/E but ULTRA-RARE. Turkish/Caucasian consanguinity + absent CD3 surface flow = sequence CD247 first. Always panel all 4 CD3 genes.",
        "surveillance": [
            "TREC NBS",
            "T/B/NK flow at diagnosis",
            "CMV/EBV/RSV surveillance biweekly",
            "Western blot CD3ζ (CD247) protein",
            "CD247 genetic sequencing (targeted p.Arg103Ter in Turkish populations)",
        ],
        "targeted_rx": "HSCT (reduced-intensity conditioning acceptable); gene therapy investigational",
    },
    {
        "gene": "LCK",
        "protein": (
            "LCK -- 1p35.1 Autosomal-Recessive-LOF-or-AD-GOF -- 509aa -- "
            "Lymphocyte-Specific-Protein-Tyrosine-Kinase-56kDa-SH2-SH3-SH4-Kinase-Domains-"
            "CD4-CD8-Coreceptor-Proximal-TCR-Kinase-LOF-T-Lymphopenia-GOF-Lymphoproliferation-"
            "OPPOSITE-PHENOTYPES-SAME-GENE-CRITICAL-DDx-BCG-CI-LOF-OMIM-153390"
        ),
        "locus": "1p35.1",
        "protein_size": (
            "509 aa / 56 kDa / LCK molecular context: "
            "STRUCTURE: "
            "  509 aa / 56 kDa; SH4 domain (N-terminal myristoylation + palmitoylation, plasma membrane); "
            "  Unique domain (aa 50-88) for CD4/CD8 cytoplasmic tail binding; "
            "  SH3 domain (Pro-rich ligand binding); SH2 domain (pTyr binding); "
            "  Kinase domain (aa 251-500) — activation loop Tyr394 (activating), Tyr505 (inhibitory, CSK phosphorylates); "
            "PHENOTYPE DICHOTOMY: "
            "  LOF (biallelic): T-lymphopenia; reduced T-cell activation; CD4 disproportionately low; "
            "  GOF (monoallelic activating): T-cell hyperactivation + lymphoproliferation + EBV susceptibility; "
            "  SAME GENE — OPPOSITE PHENOTYPES: sequencing mandatory (LOF vs GOF cannot be clinically distinguished without genetics); "
            "KEY MANAGEMENT: "
            "  LOF: conservative management; HSCT if severe; "
            "  GOF: JAK inhibitors (ruxolitinib) under investigation; rituximab for EBV; "
        ),
        "syndrome": "Lck deficiency (LOF) or Lck GOF lymphoproliferation",
        "inheritance": "AR LOF (biallelic) or AD/de novo GOF",
        "immunophenotype": "LOF: T-lymphopenia (CD4>CD8 disproportion); GOF: lymphoproliferation + EBV",
        "t_cell_pct_mean": 12.0,
        "pathognomonic": "LOF: T-lymphopenia with CD4+ disproportionately reduced + absent T-cell activation (anti-CD3 stimulation). GOF: lymphoproliferation + elevated CD4:CD8 ratio.",
        "key_avoid": "NEVER give BCG (LOF); NEVER treat LOF with HSCT for GOF; NEVER miss GOF variants (could worsen with HSCT); NEVER skip TCR activation assay (anti-CD3/anti-CD28)",
        "key_rule": "LCK LOF vs GOF = OPPOSITE phenotypes, same gene. LOF = T-lymphopenia, GOF = lymphoproliferation. T-cell activation assay (PHA/anti-CD3) MANDATORY to classify before treatment.",
        "surveillance": [
            "T/B/NK flow cytometry with CD4:CD8 ratio",
            "T-cell mitogen (PHA) and anti-CD3 proliferation assay",
            "Lck protein Western blot and phosphorylation assay",
            "EBV PCR (LOF and GOF both susceptible)",
            "Annual imaging (GOF: lymphoma surveillance)",
        ],
        "targeted_rx": "LOF: HSCT if severe; GOF: ruxolitinib (investigational) + rituximab (EBV); no FDA-approved targeted therapy",
    },
    {
        "gene": "LAT",
        "protein": (
            "LAT -- 16q24.1 Autosomal-Recessive-LOF -- 262aa -- "
            "Linker-for-Activation-of-T-cells-27kDa-Palmitoylated-Scaffold-Adapter-"
            "ZAP70-Substrate-PLCgamma1-PI3K-GRB2-GADS-Recruiter-"
            "T-B+NK+-SCID-Like-TCR-Downstream-Signalling-Absent-ULTRA-RARE-"
            "BCG-ABSOLUTELY-CI-HSCT-LIKELY-CURATIVE-OMIM-602354"
        ),
        "locus": "16q24.1",
        "protein_size": (
            "262 aa / 27 kDa / LAT molecular context: "
            "STRUCTURE: "
            "  262 aa / 27 kDa; N-terminal palmitoylation (Cys26, Cys29) → lipid raft targeting; "
            "  Short extracellular ectodomain; TM domain; cytoplasmic C-terminal signalling scaffold; "
            "  KEY PHOSPHORYLATION SITES: Tyr132 (PLCγ1), Tyr171/Tyr191 (GRB2/GADS), Tyr226 (GRB2); "
            "  ZAP-70 is upstream of LAT: ZAP-70 phosphorylates LAT Tyr residues; "
            "MECHANISM: "
            "  LAT is the central relay node of TCR proximal signalling; "
            "  pTyr132-LAT recruits PLCγ1 → IP3 + DAG → calcium mobilisation + PKC activation; "
            "  Without LAT: no IP3, no calcium, no NFAT, no T-cell activation; "
            "IMMUNOPHENOTYPE: "
            "  T cells present in thymus but fail peripheral signalling → T-lymphopenia or SCID-like; "
            "  ULTRA-RARE: first human cases described 2023; "
            "KEY MANAGEMENT: "
            "  HSCT likely curative (murine LAT-/- fully corrected by HSCT); "
        ),
        "syndrome": "LAT deficiency — T-SCID-like with absent TCR downstream signalling",
        "inheritance": "AR LOF",
        "immunophenotype": "T-lymphopenia/T-SCID: T cells absent or severely reduced; B and NK normal",
        "t_cell_pct_mean": 1.5,
        "pathognomonic": "T-SCID + absent TCR signalling on stimulation assay (anti-CD3/PHA) + normal ZAP-70 expression → LAT deficiency (first rule out CD3 complex defects by flow)",
        "key_avoid": "NEVER give BCG; NEVER miss LAT in T-SCID with normal CD3 surface flow (surface CD3 may be partially present); NEVER skip TCR stimulation assay",
        "key_rule": "LAT deficiency: surface CD3 may be PRESENT (unlike CD3D/E/G/Z) but TCR downstream signalling ABSENT. Activation assay (anti-CD3) with flow for Ca2+ influx or proliferation distinguishes LAT from CD3 complex defects.",
        "surveillance": [
            "TREC NBS",
            "T/B/NK flow (CD3 surface present in LAT — key DDx from CD3 complex defects)",
            "TCR activation assay (anti-CD3/PHA) — absent proliferation",
            "Calcium influx assay (Fura-2) — absent in LAT/ORAI1/STIM1",
            "LAT phosphorylation Western blot (pTyr132-LAT)",
        ],
        "targeted_rx": "HSCT curative (based on murine data); gene therapy under investigation",
    },
    {
        "gene": "ORAI1",
        "protein": (
            "ORAI1 -- 12q24.31 Autosomal-Recessive-LOF -- 301aa -- "
            "CRAC-Channel-Pore-Forming-Subunit-33kDa-4-TM-Helices-SOCE-"
            "CombinedID-Anhidrosis-Myopathy-ECTODERMAL-PATHOGNOMONIC-"
            "Tubingen-Arg91Trp-Founder-BCG-ABSOLUTELY-CI-HSCT-CURATIVE-OMIM-610277"
        ),
        "locus": "12q24.31",
        "protein_size": (
            "301 aa / 33 kDa / ORAI1 molecular context: "
            "STRUCTURE: "
            "  301 aa / 33 kDa; 4 TM helices (TM1-TM4); TM1 lines the Ca2+ pore; "
            "  Glu106 in TM1: Ca2+-selective filter (mutation here abolishes Ca2+ selectivity); "
            "  N-terminal STIM1-activating region; C-terminal STIM1 docking site (ETON region aa 272-292); "
            "  6 ORAI1 subunits form the hexameric CRAC channel pore; "
            "SOCE MECHANISM: "
            "  TCR activation → IP3 → ER calcium depletion → STIM1 unfolds → STIM1 oligomerises → "
            "  STIM1 migrates to ER-PM junction → activates ORAI1 pore → Ca2+ influx → NFAT dephosphorylation → "
            "  NFAT enters nucleus → cytokine gene transcription (IL-2, IFN-γ, etc.); "
            "PHENOTYPE TRIAD (PATHOGNOMONIC): "
            "  1. Combined immunodeficiency (T present but non-functional); "
            "  2. Anhidrosis (absent sweating — ectodermal dysplasia); "
            "  3. Myopathy (skeletal muscle ORAI1-dependent calcium handling); "
            "  Tubingen founder: p.Arg91Trp in German/Turkish populations; "
            "KEY MANAGEMENT: "
            "  HSCT curative for immunodeficiency; myopathy partially ameliorated; "
            "  Anhidrosis: cooling vests mandatory in hot environments; "
        ),
        "syndrome": "ORAI1 deficiency — combined immunodeficiency + anhidrosis + myopathy (SOCE defect)",
        "inheritance": "AR LOF",
        "immunophenotype": "Combined ID: T cells present but non-functional; B cells present; NK reduced",
        "t_cell_pct_mean": 22.0,
        "pathognomonic": "Combined ID + ANHIDROSIS (absent sweating) + MYOPATHY = ORAI1 or STIM1 PATHOGNOMONIC triad. Anhidrosis distinguishes from other T-SCIDs. Sweat test = 0 mg/m2 NaCl.",
        "key_avoid": "NEVER give BCG; NEVER miss anhidrosis (heat stroke risk); NEVER mistake for hypo-Ig syndrome (T cells are PRESENT); NEVER skip calcium flux assay (Ca2+ entry absent despite normal T-cell numbers)",
        "key_rule": "ORAI1 deficiency: T cells PRESENT but ANERGIC (cannot activate). KEY DDx: combined ID with anhidrosis + myopathy is PATHOGNOMONIC for SOCE defect (ORAI1/STIM1). Anhidrosis = sweat test zero.",
        "surveillance": [
            "Calcium influx assay (Fura-2) — ABSENT in ORAI1/STIM1",
            "T/B/NK flow (T cells present — key DDx from CD3 complex defects)",
            "T-cell activation assay (anti-CD3/PHA) — severely impaired",
            "Sweat test: pilocarpine iontophoresis (NaCl 0 = anhidrosis)",
            "Muscle biopsy: tubular aggregates in severe cases",
        ],
        "targeted_rx": "HSCT curative for immunodeficiency; cooling vests for anhidrosis; physiotherapy for myopathy; no FDA-approved CRAC channel modulator for immunodeficiency indication",
    },
    {
        "gene": "STIM1",
        "protein": (
            "STIM1 -- 11p15.5 Autosomal-Recessive-LOF -- 685aa -- "
            "ER-Calcium-Sensor-SOCE-Activator-77kDa-EF-Hand-SAM-TM-CRAC-Activation-Domain-"
            "CombinedID-Tubular-Aggregate-Myopathy-PATHOGNOMONIC-Anhidrosis-Cardiac-Arrhythmia-"
            "Annual-ECG-MANDATORY-BCG-ABSOLUTELY-CI-HSCT-CURATIVE-OMIM-605921"
        ),
        "locus": "11p15.5",
        "protein_size": (
            "685 aa / 77 kDa / STIM1 molecular context: "
            "STRUCTURE: "
            "  685 aa / 77 kDa; N-terminal signal peptide (ER lumen); "
            "  EF-hand domain (aa 58-201): Ca2+-binding; Ca2+ dissociation triggers unfolding; "
            "  SAM domain (aa 201-337): protein-protein interactions; "
            "  TM domain (aa 446-474): ER membrane anchor; "
            "  CRAC Activation Domain (CAD, aa 342-448): binds and activates ORAI1; "
            "  Serine-proline rich region; EB1-binding microtubule-tracking domain; "
            "SOCE MECHANISM: "
            "  ER Ca2+ depletion → STIM1 EF-hand loses Ca2+ → STIM1 unfolds → oligomerises (SOAR/CAD); "
            "  STIM1 oligomers migrate to ER-PM junctions → directly binds ORAI1 N- and C-termini → "
            "  Opens ORAI1 pore → Ca2+ influx → NFAT activation; "
            "PATHOGNOMONIC FEATURES: "
            "  1. Combined immunodeficiency (T cells present but non-functional); "
            "  2. TUBULAR AGGREGATE MYOPATHY (muscle biopsy: sarcoplasmic reticulum tubular aggregates); "
            "  3. Anhidrosis (ectodermal); "
            "  4. CARDIAC ARRHYTHMIA (QTc prolongation, SVT) — annual ECG MANDATORY; "
            "  STIM1 GOF → York platelet syndrome (AD, thrombocytopenia); "
            "KEY MANAGEMENT: "
            "  Annual ECG + Holter monitoring (cardiac); "
            "  HSCT curative for immune deficiency; tubular myopathy PERSISTS post-HSCT; "
        ),
        "syndrome": "STIM1 deficiency — combined immunodeficiency + tubular aggregate myopathy + cardiac arrhythmia (SOCE defect)",
        "inheritance": "AR LOF (biallelic) or AD GOF (York platelet syndrome)",
        "immunophenotype": "Combined ID: T cells present but non-functional; B cells present; NK reduced",
        "t_cell_pct_mean": 20.0,
        "pathognomonic": "Combined ID + TUBULAR AGGREGATE MYOPATHY (muscle biopsy) + CARDIAC ARRHYTHMIA = STIM1 PATHOGNOMONIC. Annual ECG mandatory (QTc arrhythmia risk). York platelet syndrome = AD GOF.",
        "key_avoid": "NEVER give BCG; NEVER skip annual ECG (life-threatening arrhythmia); NEVER mistake for ORAI1 without muscle biopsy; NEVER overlook York platelet syndrome (STIM1 GOF, AD, distinct phenotype)",
        "key_rule": "STIM1 LOF: combined ID + tubular aggregate myopathy (muscle biopsy MANDATORY) + anhidrosis + CARDIAC ARRHYTHMIA (ECG annual). Myopathy PERSISTS after HSCT. GOF = York platelet syndrome (thrombocytopenia, AD).",
        "surveillance": [
            "Annual ECG + Holter monitoring (arrhythmia surveillance MANDATORY)",
            "Calcium influx assay (Fura-2) — absent",
            "Muscle biopsy: tubular aggregates (sarcoplasmic reticulum)",
            "T/B/NK flow cytometry (T cells present)",
            "T-cell activation assay (anti-CD3/PHA) — severely impaired",
        ],
        "targeted_rx": "HSCT curative for immunodeficiency (myopathy persists); annual cardiac monitoring; antiarrhythmic therapy as needed; gene therapy investigational; no FDA-approved SOCE modulator",
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

    # Core immunophenotype parameters by gene
    params = {
        "CD3D":  dict(t_absent=True,  ca_impaired=False, eda=False, myo=False, cardiac=False, hsct_pct=82, live_to_dx_mo=3.5),
        "CD3E":  dict(t_absent=True,  ca_impaired=False, eda=False, myo=False, cardiac=False, hsct_pct=80, live_to_dx_mo=3.8),
        "CD3G":  dict(t_absent=False, ca_impaired=False, eda=False, myo=False, cardiac=False, hsct_pct=28, live_to_dx_mo=14.0),
        "CD247": dict(t_absent=True,  ca_impaired=False, eda=False, myo=False, cardiac=False, hsct_pct=78, live_to_dx_mo=4.0),
        "LCK":   dict(t_absent=False, ca_impaired=True,  eda=False, myo=False, cardiac=False, hsct_pct=35, live_to_dx_mo=24.0),
        "LAT":   dict(t_absent=True,  ca_impaired=True,  eda=False, myo=False, cardiac=False, hsct_pct=70, live_to_dx_mo=5.0),
        "ORAI1": dict(t_absent=False, ca_impaired=True,  eda=True,  myo=True,  cardiac=False, hsct_pct=65, live_to_dx_mo=8.0),
        "STIM1": dict(t_absent=False, ca_impaired=True,  eda=True,  myo=True,  cardiac=True,  hsct_pct=60, live_to_dx_mo=9.0),
    }
    p = params[gene_name]

    patients = []
    for i in range(40):
        age = r.gauss(p["live_to_dx_mo"], 2.5)
        age = max(0.5, round(age, 1))
        t_pct = gene["t_cell_pct_mean"] + r.gauss(0, gene["t_cell_pct_mean"] * 0.3 + 1)
        t_pct = max(0, round(t_pct, 1))

        bcg_reaction = r.random() < (0.35 if p["t_absent"] else 0.18)
        pcp = r.random() < (0.28 if p["t_absent"] else 0.08)
        ops_infection = r.random() < (0.65 if p["t_absent"] else 0.40)
        anhidrosis = p["eda"] and r.random() < 0.85
        myopathy = p["myo"] and r.random() < 0.78
        cardiac = p["cardiac"] and r.random() < 0.55
        hsct_done = r.random() < (p["hsct_pct"] / 100.0)
        ca_absent = p["ca_impaired"] and r.random() < 0.90

        patients.append({
            "id": i + 1,
            "gene": gene_name,
            "age_at_dx_months": age,
            "t_pct": t_pct,
            "bcg_reaction": bcg_reaction,
            "pcp_pneumonia": pcp,
            "opportunistic_infection": ops_infection,
            "anhidrosis": anhidrosis,
            "myopathy": myopathy,
            "cardiac_arrhythmia": cardiac,
            "calcium_influx_absent": ca_absent,
            "hsct_done": hsct_done,
        })
    return patients


def generate_overview():
    all_patients = []
    for i in range(8):
        all_patients.extend(_patient_cohort(i))

    total = len(all_patients)
    t_absent_pct = round(sum(1 for p in all_patients if p["t_pct"] < 2.0) / total * 100, 1)
    bcg_pct = round(sum(1 for p in all_patients if p["bcg_reaction"]) / total * 100, 1)
    hsct_pct = round(sum(1 for p in all_patients if p["hsct_done"]) / total * 100, 1)
    eda_pct = round(sum(1 for p in all_patients if p["anhidrosis"]) / total * 100, 1)
    ca_pct = round(sum(1 for p in all_patients if p["calcium_influx_absent"]) / total * 100, 1)
    cardiac_pct = round(sum(1 for p in all_patients if p["cardiac_arrhythmia"]) / total * 100, 1)
    opp_pct = round(sum(1 for p in all_patients if p["opportunistic_infection"]) / total * 100, 1)

    return {
        "atlas": "Hereditary-CD3-TCR-ORAI1-STIM1-Atlas",
        "subtitle": "Complete 8-Gene Reference · CD3-Complex · LCK · LAT · SOCE-ORAI1-STIM1",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total,
        "gene_counts": {g["gene"]: 40 for g in ATLAS_GENES},
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "aggregate_stats": {
            "t_absent_pct": t_absent_pct,
            "bcg_reaction_pct": bcg_pct,
            "hsct_pct": hsct_pct,
            "ectodermal_dysplasia_pct": eda_pct,
            "calcium_influx_absent_pct": ca_pct,
            "cardiac_arrhythmia_pct": cardiac_pct,
            "opportunistic_infection_pct": opp_pct,
            "cd3_complex_genes": ["CD3D", "CD3E", "CD3G", "CD247"],
            "tcr_kinase_adapter_genes": ["LCK", "LAT"],
            "soce_calcium_genes": ["ORAI1", "STIM1"],
        },
        "key_facts": [
            "CD3D: T-B+NK+ SCID — surface CD3 ABSENT by flow; BCG absolutely CI; HSCT myeloablative; seeds 3950",
            "CD3E: T-B+NK+ SCID — phenotype IDENTICAL to CD3D; Morocco p.Cys68Tyr founder; sequence all 4 CD3 genes; seeds 3951",
            "CD3G: T-lymphopenia — MILDER (NOT complete SCID); residual T cells 10-30%; CD3 reduced not absent; seeds 3952",
            "CD247: T-B+NK+ SCID — CD3ζ ULTRA-RARE (<30 families); Turkey/Caucasus p.Arg103Ter founder; seeds 3953",
            "LCK: LOF = T-lymphopenia CD4>CD8; GOF = lymphoproliferation + EBV — OPPOSITE PHENOTYPES SAME GENE; seeds 3954",
            "LAT: T-SCID-like — surface CD3 PRESENT but TCR downstream ABSENT; activation assay distinguishes; seeds 3955",
            "ORAI1: combined ID + ANHIDROSIS + MYOPATHY PATHOGNOMONIC TRIAD — Tübingen p.Arg91Trp founder; seeds 3956",
            "STIM1: combined ID + TUBULAR AGGREGATE MYOPATHY + CARDIAC ARRHYTHMIA — annual ECG MANDATORY; seeds 3957",
        ],
    }


def generate_breakdown():
    gene_cohorts = []
    diagnostic_groups = {
        "cd3_complex_assembly": {
            "genes": ["CD3D", "CD3E", "CD3G", "CD247"],
            "mechanism": "Loss of TCR-CD3 surface complex assembly → absent surface CD3 → complete (CD3D/E/Z) or partial (CD3G) T cell development failure",
            "shared_feature": "Surface CD3 absent or reduced by flow cytometry; T-B+NK+ phenotype (complete) or T-lymphopenia (CD3G partial)",
            "key_ddx": "CD3D vs CD3E: phenotypically identical — genetics required. CD3G: milder (reduced not absent CD3 flow). CD247: ultra-rare, Turkish founder.",
            "bcg_ci": True,
        },
        "tcr_proximal_kinase_adapter": {
            "genes": ["LCK", "LAT"],
            "mechanism": "Normal TCR-CD3 surface expression but absent/impaired TCR signal transduction: Lck (upstream kinase) and LAT (downstream scaffold) deficiency",
            "shared_feature": "T cells present with NORMAL surface CD3 flow BUT absent/impaired T-cell activation on stimulation assay",
            "key_ddx": "LCK LOF vs GOF: OPPOSITE phenotypes — sequencing mandatory. LAT: absent pTyr-LAT Western blot. Both: TCR stimulation assay (anti-CD3/PHA) critical.",
            "bcg_ci": True,
        },
        "soce_calcium_entry": {
            "genes": ["ORAI1", "STIM1"],
            "mechanism": "Absent store-operated calcium entry (SOCE): ORAI1 pore (LOF) or STIM1 sensor (LOF) → no Ca2+ influx → absent NFAT → T-cell anergy despite normal T-cell numbers",
            "shared_feature": "Combined ID with absent calcium influx assay (Fura-2); T cells PRESENT but non-functional; anhidrosis (ectodermal); myopathy",
            "key_ddx": "ORAI1 vs STIM1: STIM1 additionally has tubular aggregate myopathy (muscle biopsy) and CARDIAC ARRHYTHMIA (annual ECG mandatory). ORAI1: Tübingen founder p.Arg91Trp.",
            "bcg_ci": True,
        },
    }

    pathognomonic_features = {
        "CD3D_complete_scid": "T-B+NK+ SCID + absent surface CD3 by flow + TREC=0 on NBS = CD3 complex deficiency (CD3D/CD3E/CD3G/CD247 panel mandatory)",
        "CD3G_partial_milder": "T-lymphopenia with REDUCED (not absent) surface CD3 = CD3G — milder, later onset; HSCT threshold higher than CD3D/E/Z",
        "LCK_opposite_phenotypes": "LOF = T-lymphopenia (CD4 disproportionate) vs GOF = lymphoproliferation + EBV: SAME GENE, OPPOSITE PHENOTYPES — sequencing mandatory before HSCT",
        "LAT_normal_cd3_absent_signalling": "Surface CD3 PRESENT but TCR downstream signalling ABSENT (anti-CD3 assay) = LAT/LCK proximal defect (DDx from CD3 complex defects by flow)",
        "ORAI1_anhidrosis_triad": "Combined ID + ANHIDROSIS + MYOPATHY = ORAI1 or STIM1 PATHOGNOMONIC triad. Sweat test (pilocarpine) = 0 mg/m2 NaCl. Ca2+ flux absent (Fura-2).",
        "STIM1_tubular_aggregates": "Combined ID + TUBULAR AGGREGATE MYOPATHY (muscle biopsy) + CARDIAC ARRHYTHMIA (ECG: QTc prolongation) = STIM1 PATHOGNOMONIC. Annual ECG MANDATORY.",
        "bcg_absolutely_ci_all_8": "BCG ABSOLUTELY CONTRAINDICATED in all 8 genes — T cell functional deficiency (even if T cells present in ORAI1/STIM1/LCK/LAT) — BCG-osis fatal",
    }

    for i, gene_data in enumerate(ATLAS_GENES):
        patients = _patient_cohort(i)
        gene_name = gene_data["gene"]
        n = len(patients)
        t_absent_n = sum(1 for p in patients if p["t_pct"] < 2.0)
        bcg_n = sum(1 for p in patients if p["bcg_reaction"])
        pcp_n = sum(1 for p in patients if p["pcp_pneumonia"])
        eda_n = sum(1 for p in patients if p["anhidrosis"])
        myo_n = sum(1 for p in patients if p["myopathy"])
        cardiac_n = sum(1 for p in patients if p["cardiac_arrhythmia"])
        hsct_n = sum(1 for p in patients if p["hsct_done"])
        ca_n = sum(1 for p in patients if p["calcium_influx_absent"])
        opp_n = sum(1 for p in patients if p["opportunistic_infection"])
        mean_age = round(sum(p["age_at_dx_months"] for p in patients) / n, 1)
        mean_t = round(sum(p["t_pct"] for p in patients) / n, 1)

        gene_cohorts.append({
            "gene": gene_name,
            "gene_info": gene_data,
            "n": n,
            "t_absent_pct": round(t_absent_n / n * 100, 1),
            "bcg_reaction_pct": round(bcg_n / n * 100, 1),
            "pcp_pct": round(pcp_n / n * 100, 1),
            "ectodermal_dysplasia_pct": round(eda_n / n * 100, 1),
            "myopathy_pct": round(myo_n / n * 100, 1),
            "cardiac_arrhythmia_pct": round(cardiac_n / n * 100, 1),
            "hsct_pct": round(hsct_n / n * 100, 1),
            "calcium_influx_absent_pct": round(ca_n / n * 100, 1),
            "opportunistic_infection_pct": round(opp_n / n * 100, 1),
            "mean_age_at_dx_months": mean_age,
            "mean_t_cell_pct": mean_t,
            "pathognomonic": gene_data["pathognomonic"],
        })

    return {
        "gene_cohorts": gene_cohorts,
        "diagnostic_groups": diagnostic_groups,
        "pathognomonic_features": pathognomonic_features,
        "cascade_testing_order": [
            "Step 1: T/B/NK lymphocyte subset enumeration + TREC on NBS",
            "Step 2: Surface CD3 flow cytometry (absent = CD3D/E/G/Z; present = LCK/LAT/ORAI1/STIM1)",
            "Step 3: If CD3 absent → gene panel CD3D+CD3E+CD3G+CD247 (all on 11q23.3 except CD247 1q24.2)",
            "Step 4: If CD3 present + T-lymphopenia → TCR stimulation assay (anti-CD3/PHA proliferation)",
            "Step 5: If stimulation impaired → calcium influx assay (Fura-2): absent = ORAI1/STIM1; partial = LCK/LAT",
            "Step 6: STIM1 vs ORAI1: muscle biopsy (tubular aggregates = STIM1) + ECG (arrhythmia = STIM1)",
            "Step 7: LCK vs LAT: Lck Western blot + pTyr132-LAT phosphorylation assay",
            "Step 8: BCG ABSOLUTELY CONTRAINDICATED at all steps until T cell function confirmed normal",
        ],
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "T-B+NK+ SCID", "definition": "Severe Combined Immunodeficiency with absent T cells but NORMAL B cells and NK cells. Phenotype of CD3D, CD3E, CD247, and similar gene defects. Opposite of γc-chain/JAK3 SCID (T-B+NK-). T-cell receptor complex failure."},
            {"term": "TCR-CD3 Complex", "definition": "The T-cell antigen receptor complex: TCRαβ heterodimer non-covalently associated with CD3γε, CD3δε heterodimers and CD3ζζ homodimer. Six signalling subunits (CD3D, CD3E×2, CD3G, CD247×2). Absent surface CD3 by flow = CD3 complex defect."},
            {"term": "TREC (T-cell Receptor Excision Circle)", "definition": "Episomal DNA circles generated as byproducts of TCR V(D)J recombination in the thymus. TREC = 0 on newborn screening = SCID (T cells absent). Standard NBS test for T-cell deficiency. TREC = 0 for all T-B+NK+ SCID genes."},
            {"term": "SOCE (Store-Operated Calcium Entry)", "definition": "Calcium influx mechanism triggered by ER calcium store depletion. STIM1 (ER sensor) → ORAI1 (plasma membrane pore). SOCE = primary calcium influx route in T cells. Absent SOCE → no NFAT → no cytokine gene transcription → T-cell anergy."},
            {"term": "CRAC Channel", "definition": "Calcium Release-Activated Calcium channel. Formed by ORAI1 hexamer. Activated by STIM1 oligomers at ER-PM junctions. Highly selective for Ca2+. Hallmark: inwardly-rectifying, Ca2+-selective current (ICRAC) measurable by patch clamp. Absent ICRAC = ORAI1 deficiency."},
            {"term": "Anhidrosis", "definition": "Absent or severely impaired sweating. Pathognomonic of ORAI1 and STIM1 deficiency (SOCE defect in eccrine sweat glands). Diagnosed by pilocarpine sweat test = 0 mg/m2 NaCl. Life-threatening: hyperthermia in hot environments. Requires cooling vests."},
            {"term": "Tubular Aggregate Myopathy", "definition": "Muscle biopsy finding of aggregated sarcoplasmic reticulum tubules (seen on electron microscopy or Gomori trichrome stain). Pathognomonic of STIM1 deficiency. Represents accumulated, non-functional STIM1-activated SR membranes. Myopathy PERSISTS after HSCT."},
            {"term": "Calcium Flux Assay (Fura-2)", "definition": "Fluorometric assay using calcium-sensitive dye Fura-2. Measures intracellular calcium rise after TCR stimulation (anti-CD3) or thapsigargin (ER Ca2+ depletion). ABSENT in ORAI1 and STIM1 deficiency (SOCE absent). REDUCED in LCK/LAT deficiency. NORMAL in CD3 complex defects (entry mechanism intact)."},
            {"term": "Lck LOF vs GOF", "definition": "LCK gene: BIALLELIC LOF = T-lymphopenia (T cells reduced, CD4 > CD8 disproportion). AD/de novo GOF = lymphoproliferation, EBV susceptibility, autoimmunity. OPPOSITE PHENOTYPES from same gene. Sequencing MANDATORY before HSCT decision (GOF may worsen with allogenic HSCT-mediated immune activation)."},
            {"term": "LAT phosphorylation (pTyr132)", "definition": "Key diagnostic test for LAT deficiency. After TCR stimulation: Western blot for pTyr132-LAT. Absent pTyr132-LAT = LAT LOF or absent ZAP-70 activation. LAT has surface CD3 PRESENT (unlike CD3 complex defects) but no downstream signalling. pTyr132 = PLCγ1 recruitment site."},
            {"term": "BCG Absolutely Contraindicated (all 8 genes)", "definition": "BCG is a live attenuated Mycobacterium bovis vaccine. ALL 8 genes cause T-cell functional deficiency → BCG-osis: disseminated fatal mycobacterial infection. BCG must NEVER be given before T-cell functional assessment is complete. TREC NBS prior to BCG birth dose prevents BCG-osis in undiagnosed SCID."},
            {"term": "York Platelet Syndrome", "definition": "STIM1 GOF (gain-of-function) heterozygous variant. Distinct from STIM1 LOF immunodeficiency. Features: thrombocytopenia + giant platelets + platelet aggregation defect. Autosomal DOMINANT. Does NOT cause immunodeficiency. No HSCT. Management: platelet transfusion support, thrombopoietin-receptor agonists."},
        ],
        "key_distinguishers": {
            "absent_vs_reduced_cd3": "ABSENT CD3 (flow) = CD3D/CD3E/CD247 (complete T-B+NK+ SCID). REDUCED CD3 = CD3G (partial T-lymphopenia). CD3 NORMAL = LCK/LAT/ORAI1/STIM1 (post-receptor defects).",
            "complete_vs_partial_scid": "Complete T-B+NK+ SCID (neonatal onset, TREC=0) = CD3D/CD3E/CD247/LAT. Partial T-lymphopenia (childhood onset, residual T cells) = CD3G/LCK/ORAI1/STIM1.",
            "soce_vs_cd3_defect": "SOCE defect (ORAI1/STIM1): T cells PRESENT but anergic + calcium flux ABSENT + anhidrosis + myopathy. CD3 complex defect: T cells ABSENT + CD3 absent on flow + no ectodermal/muscle features.",
            "orai1_vs_stim1": "ORAI1 vs STIM1: both cause combined ID + anhidrosis + myopathy. STIM1 UNIQUE: tubular aggregate myopathy (muscle biopsy) + CARDIAC ARRHYTHMIA (ECG: QTc prolongation). Tübingen p.Arg91Trp = ORAI1.",
            "lck_lof_vs_gof": "LCK LOF: T-lymphopenia, CD4 disproportionate, normal B/NK. LCK GOF: lymphoproliferation, EBV, autoimmunity. SAME GENE: sequencing ALWAYS required. GOF = AD/de novo. LOF = AR biallelic.",
            "lat_ddx": "LAT: surface CD3 PRESENT (key distinction from CD3D/E/Z). T cell activation assay ABSENT (anti-CD3). pTyr132-LAT absent on Western blot. ZAP-70 normal. Ultra-rare (first cases 2023).",
        },
        "diagnostic_pathway": [
            "1. TREC NBS at birth → T cell absent = refer immunology URGENTLY",
            "2. T/B/NK enumeration → T-B+NK+ = TCR complex defect (CD3 flow next)",
            "3. Surface CD3 flow: ABSENT = CD3D/CD3E/CD3G/CD247 panel; PRESENT = post-receptor defect (step 4)",
            "4. TCR stimulation assay (anti-CD3/PHA): impaired = LCK/LAT/ORAI1/STIM1",
            "5. Calcium influx (Fura-2): absent = ORAI1/STIM1; partial = LCK/LAT",
            "6. ORAI1 vs STIM1: muscle biopsy (tubular aggregates = STIM1) + ECG (arrhythmia = STIM1) + sweat test (=0 both)",
            "7. LCK vs LAT: Lck Western blot + pTyr132-LAT phospho-blot + genetics",
            "8. BCG ABSOLUTELY CONTRAINDICATED until T cell function confirmed normal across all steps",
        ],
    }
