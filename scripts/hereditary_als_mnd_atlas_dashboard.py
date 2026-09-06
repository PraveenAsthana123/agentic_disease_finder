#!/usr/bin/env python3
"""Hereditary-ALS-MND-Atlas — Complete 8-Gene Hereditary ALS/Motor Neuron Disease Atlas
SOD1   (Cu/Zn superoxide dismutase 1; 153 aa; 21q22.11; AD/AR;
         ALS1 — first familial ALS gene (1993); A4V most common North America; rapid progression <1y;
         Tofersen (ASO) FDA 2023; seed SEED_BASE+0) ·
TARDBP (TDP-43; 414 aa; 1p36.22; AD;
         ALS10 — TDP-43 cytoplasmic inclusions PATHOGNOMONIC 97% sporadic ALS;
         nuclear clearance + cytoplasmic aggregation; ubiquitinated inclusions;
         seed SEED_BASE+1) ·
FUS    (fused in sarcoma; 526 aa; 16p11.2; AD/AR;
         ALS6 — young onset <40y; nuclear localisation signal mutations most severe;
         P525L juvenile ALS; basophilic inclusions; seed SEED_BASE+2) ·
C9ORF72 (chromosome 9 open reading frame 72; 481 aa; 9p21.2; AD;
          GGGGCC hexanucleotide repeat expansion — MOST COMMON familial ALS/FTD ~40%;
          repeat-primed PCR required; RNA foci + DPR proteins; seed SEED_BASE+3) ·
UBQLN2 (ubiquilin 2; 624 aa; Xp11.21; X-linked dominant;
         ALS15 — first X-linked ALS; ubiquitin receptor; proteasomal pathway;
         ubiquilin-2 inclusions; seed SEED_BASE+4) ·
VCP    (valosin-containing protein; 806 aa; 9p13.3; AD;
         ALS14/IBMPFD — multisystem: IBM + Paget bone disease + FTD + ALS;
         TDP-43 pathology; R155H/G/C most common; seed SEED_BASE+5) ·
OPTN   (optineurin; 577 aa; 10p13; AD/AR;
         ALS12 — optineurin inclusions; autophagy receptor; p62 co-immunoreactivity;
         haploinsufficiency mechanism; E478G most common; seed SEED_BASE+6) ·
TBK1   (TANK-binding kinase 1; 729 aa; 12q14.2; AD;
         ALS with FTD — haploinsufficiency; phosphorylates OPTN/p62; autophagy-NF-κB;
         FTD common (~40%); seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1710–1717)
"""

import random

SEED_BASE = 1710

ALS_GENES = [
    # ── SOD1 — ALS1 ──────────────────────────────────────────────────────────
    {
        "gene": "SOD1",
        "protein": "SOD1 — 21q22.11 AD/AR — Cu/Zn-Superoxide-Dismutase-153aa — ALS1-First-Familial-ALS-Gene-1993 — A4V-Rapid-<1y — Tofersen-FDA2023-ASO",
        "alias": (
            "SOD1 (superoxide dismutase 1, soluble); OMIM gene 147450; "
            "ALS1 (amyotrophic lateral sclerosis type 1) OMIM 105400. "
            "21q22.11; 153 aa; ~16 kDa; autosomal dominant (most), autosomal recessive (some variants). "
            "FUNCTION: SOD1 is a homodimeric copper- and zinc-binding metalloprotein that catalyses the "
            "dismutation of superoxide radical (O2•−) to oxygen (O2) and hydrogen peroxide (H2O2). "
            "It is the primary cytoplasmic antioxidant enzyme. "
            "ALS MECHANISM: SOD1 mutations do NOT cause disease through loss of enzyme activity — "
            "they cause a toxic gain of function (misfolded SOD1 aggregates). "
            "Misfolded SOD1 aggregates in motor neurons → mitochondrial dysfunction, ER stress, "
            "axonal transport failure, neuroinflammation → selective motor neuron death. "
            "EPIDEMIOLOGY: ~20% of all familial ALS; ~2% of sporadic ALS. "
            "~200+ pathogenic variants identified. "
            "A4V (p.Ala5Val, North America): most common SOD1 variant in North America; "
            "VERY RAPID progression — median survival 12 months from symptom onset; "
            "limb onset pattern; no bulbar predominance unlike D90A. "
            "D90A (p.Asp91Asn, Scandinavia): most common SOD1 variant worldwide; "
            "homozygous (AR) in Scandinavian population → slowly progressive ALS; "
            "heterozygous → rapid ALS (like other AD SOD1); "
            "phenotypic discordance highlights importance of homozygosity testing. "
            "G93A: used in the canonical SOD1-G93A transgenic mouse model (seminal ALS research tool). "
            "I113T: intermediate progression; psychiatric features occasionally. "
            "TOFERSEN (Qalsody, Biogen): antisense oligonucleotide (ASO) targeting SOD1 mRNA; "
            "FDA approved April 2023 for adults with ALS due to SOD1 mutation; "
            "reduces SOD1 protein in CSF/plasma; slows neurofilament light chain (NfL) rise; "
            "intrathecal injection every 28 days (3 loading doses then monthly); "
            "PRISM trial showed NfL reduction; open-label ATLAS pre-symptomatic trial ongoing. "
            "CLINICAL FEATURES: predominantly limb-onset; relatively spared bulbar in early phases; "
            "pure LMN variant (flail arm) seen with some variants; "
            "UMN + LMN combined typical; autonomic spared; cognition preserved (unlike TDP-43 related). "
            "GENETIC COUNSELLING: AD (most); AR (D90A homozygous, some others); "
            "penetrance variable (~80-90% lifetime for A4V); de novo rare; "
            "pre-symptomatic testing with genetic counselling critical (tofersen trial eligibility). "
            "BIOMARKER: plasma/CSF NfL strongly elevated; SOD1 protein reduced by tofersen treatment; "
            "SOD1 enzyme activity NOT useful (toxic GOF not LOF)."
        ),
        "locus": "21q22.11",
        "aa": 153,
        "kDa": 16,
        "omim_gene": "147450",
        "omim_disease": "105400",
        "inheritance": "AD (most variants) / AR (D90A homozygous Scandinavian, a2V homozygous) — toxic gain-of-function misfolded aggregates; NOT loss of enzyme activity; penetrance ~80-90% lifetime",
        "gene_class": "Antioxidant enzyme (Cu/Zn superoxide dismutase); cytoplasmic ROS scavenger; toxic GOF aggregation mechanism in ALS (not LOF); SOD1 aggregates motor neuron selective",
        "key_alerts": [
            "SOD1-TOFERSEN-FDA2023-MANDATORY-REFERRAL: tofersen (Qalsody) FDA-approved April 2023 for SOD1-ALS — intrathecal ASO reduces SOD1 protein and slows NfL rise; all SOD1 mutation carriers (symptomatic) should be referred to ALS specialist for tofersen assessment; pre-symptomatic ATLAS trial ongoing",
            "SOD1-A4V-RAPID-PROGRESSION-<1Y: A4V is most common SOD1 variant in North America; median survival 12 months from onset — one of fastest progressing ALS variants; early gastrostomy (PEG) and NIV discussion at diagnosis essential; palliative care integration from day 1",
            "SOD1-D90A-HOMOZYGOUS-AR-SLOWLY-PROGRESSIVE: D90A homozygous (autosomal recessive, Scandinavian) has remarkably slow progression (decades); D90A heterozygous = typical rapid AD SOD1-ALS; always check homozygosity — changes prognosis and counselling completely",
            "SOD1-PRESYMPTOMATIC-TESTING-ATLAS-TRIAL: genetic testing of at-risk relatives enables pre-symptomatic tofersen trial participation (ATLAS); genetic counselling MANDATORY before testing; positive pre-symptomatic result requires specialist ALS centre follow-up",
            "SOD1-COGNITION-PRESERVED: SOD1-ALS typically spares cognition and frontotemporal function (unlike C9ORF72/TBK1/VCP) — pure motor neuron disease; this distinction important for prognosis discussion and trial eligibility (some cognitive exclusion criteria)",
            "SOD1-NfL-BIOMARKER-PROGRESSION: plasma/CSF neurofilament light chain (NfL) is strong progression biomarker in SOD1-ALS; used to monitor tofersen response; NfL rise predates clinical deterioration; baseline NfL helps prognosticate survival",
        ],
        "etiologies": [
            "A4V (p.Ala5Val) — North American founder; most common SOD1 variant NA; median survival ~12 months; limb onset; very rapid bulbar involvement late",
            "D90A (p.Asp91Asn) — Scandinavian founder; homozygous (AR) = slow decades-long ALS; heterozygous (AD) = rapid ALS; most common SOD1 variant worldwide",
            "G93A — canonical research variant (transgenic mouse model); intermediate-rapid progression; diverse phenotype",
            "I113T — intermediate progression; higher incomplete penetrance in females; occasional psychiatric features",
            "E100G/K — rapid limb-onset; significant NfL elevation",
            "N139D/S — reported in Italian/European cohorts; progression variable",
            "Homozygous AR variants (non-D90A) — very rare; slowly progressive; important not to assume AD without pedigree analysis",
        ],
        "stats": {
            "mean_dx_age": 52,
            "mean_dx_delay_months": 14,
            "pct_familial": 20,
            "pct_NfL_elevated": 95,
            "mean_survival_months_A4V": 12,
        },
        "dx_delay_distribution": "Gaussian(mean=14, sd=6, min=4, max=36) months",
    },
    # ── TARDBP — ALS10 ───────────────────────────────────────────────────────
    {
        "gene": "TARDBP",
        "protein": "TARDBP — 1p36.22 AD — TDP-43-414aa — ALS10 — TDP-43-Cytoplasmic-Inclusions-PATHOGNOMONIC-97pct-Sporadic-ALS — Nuclear-Clearance",
        "alias": (
            "TARDBP (TAR DNA-binding protein 43, TDP-43); OMIM gene 605078; "
            "ALS10 (amyotrophic lateral sclerosis type 10 with or without FTD) OMIM 612069. "
            "1p36.22; 414 aa; ~43 kDa; autosomal dominant. "
            "FUNCTION: TDP-43 is an RNA-binding protein (RBP) that shuttles between nucleus and cytoplasm. "
            "It contains two RNA recognition motifs (RRM1, RRM2) and a glycine-rich C-terminal domain "
            "(site of most ALS mutations). "
            "Nuclear functions: regulation of mRNA splicing (thousands of transcripts), "
            "pre-mRNA processing, microRNA biogenesis, transcriptional repression. "
            "ALS/FTD PATHOMECHANISM — TDP-43 PROTEINOPATHY: "
            "In ALS-TDP, TDP-43 mislocalises from nucleus → cytoplasm; "
            "forms ubiquitinated, phosphorylated, truncated cytoplasmic inclusions (hallmark); "
            "nuclear TDP-43 is DEPLETED → loss of nuclear function (spliceopathy of hundreds of exons); "
            "cytoplasmic inclusions → toxic gain of function. "
            "KEY NEUROPATHOLOGICAL FACT: TDP-43 cytoplasmic inclusions are present in "
            "~97% of ALL sporadic ALS cases and ~50% of all FTD cases (FTD-TDP subtype) "
            "regardless of whether TARDBP is mutated — TDP-43 proteinopathy = unifying ALS pathology. "
            "TARDBP mutations: ~4% familial ALS, ~1.5% sporadic ALS. "
            "Common mutations: A315T (most frequent, worldwide); G294A; M337V; N352S; A382T (Sardinian founder). "
            "A382T: Sardinian founder mutation; higher FTD penetrance than other TARDBP variants. "
            "CLINICAL FEATURES: predominantly limb-onset ALS; "
            "FTD co-occurrence higher than SOD1-ALS but lower than C9ORF72; "
            "bulbar onset possible; variable progression (slower than A4V-SOD1); "
            "TREATMENT: no approved TDP-43-targeted therapy; riluzole + edaravone standard; "
            "antisense oligonucleotide and TDP-43 aggregate clearance trials ongoing; "
            "TDP-43 clearance strategies: autophagy enhancers (rapamycin), proteasome activators; "
            "BIOMARKER: phospho-TDP-43 in CSF (emerging); NfL elevated; "
            "DIAGNOSIS: genetic panel including TARDBP; skin biopsy TDP-43 pathology (research); "
            "CSF phospho-TDP-43 seed amplification assay (SEED, research)."
        ),
        "locus": "1p36.22",
        "aa": 414,
        "kDa": 43,
        "omim_gene": "605078",
        "omim_disease": "612069",
        "inheritance": "AD — autosomal dominant; most mutations in glycine-rich C-terminal domain; A382T Sardinian founder AR-like penetrance pattern; de novo rare",
        "gene_class": "RNA-binding protein (RRM1+RRM2 + glycine-rich CTD); nuclear-cytoplasmic shuttle; master splicing regulator; TDP-43 cytoplasmic inclusions = universal ALS hallmark regardless of mutation",
        "key_alerts": [
            "TARDBP-TDP43-PATHOGNOMONIC-97PCT-SPORADIC: TDP-43 cytoplasmic inclusions occur in ~97% of ALL sporadic ALS — TARDBP mutation is NOT required for TDP-43 pathology; pathology is the unifying hallmark of ALS regardless of genetics",
            "TARDBP-A382T-SARDINIAN-FOUNDER-FTD-RISK: A382T is a Sardinian founder mutation with disproportionately high FTD co-occurrence; screen all Sardinian/Italian ALS patients; FTD genetic counselling important for families",
            "TARDBP-NO-TARGETED-THERAPY-YET: unlike SOD1 (tofersen), TARDBP has no FDA-approved targeted therapy; trials ongoing for TDP-43 aggregate clearance and nuclear restoration; standard of care = riluzole + edaravone + MDT",
            "TARDBP-NUCLEAR-DEPLETION-SPLICEOPATHY: TDP-43 nuclear loss → mis-splicing of >1000 transcripts (including STMN2 and UNC13A) — these cryptic exon inclusions are therapeutic targets; STMN2 ASO restoration in trials",
            "TARDBP-CSF-PHOSPHO-TDP43-EMERGING-BIOMARKER: phosphorylated TDP-43 (pTDP-43 S409/410) in CSF is an emerging diagnostic and monitoring biomarker; seed amplification assay (SAA) can detect TDP-43 pathology antemortem",
            "TARDBP-FTD-SURVEILLANCE: all TARDBP mutation carriers need baseline cognitive assessment (ACE-III, MoCA); A382T carriers especially; FTD may precede motor symptoms; behavioural variant FTD can mask ALS diagnosis",
        ],
        "etiologies": [
            "A315T — most frequent TARDBP mutation worldwide; mixed limb/bulbar; intermediate progression",
            "G294A — European cohorts; predominantly limb onset; slower progression than SOD1-A4V",
            "M337V — significant bulbar involvement; faster progression; European/North American",
            "N352S — North American; limb onset; relatively preserved cognition",
            "A382T — Sardinian founder; high FTD co-occurrence; slower motor progression but cognitive decline prominent",
            "D169G — rare; severe early onset; nuclear localisation disrupted",
            "Q331K — affects RNA binding; limb onset; intermediate survival",
        ],
        "stats": {
            "mean_dx_age": 57,
            "mean_dx_delay_months": 16,
            "pct_familial": 4,
            "pct_sporadic_TDP43_pathology": 97,
            "pct_FTD_cooccurrence": 15,
        },
        "dx_delay_distribution": "Gaussian(mean=16, sd=8, min=4, max=48) months",
    },
    # ── FUS — ALS6 ───────────────────────────────────────────────────────────
    {
        "gene": "FUS",
        "protein": "FUS — 16p11.2 AD/AR — Fused-in-Sarcoma-526aa — ALS6 — Young-Onset-<40y — P525L-Juvenile-ALS — NLS-Mutations-Most-Severe",
        "alias": (
            "FUS (fused in sarcoma; also TLS — translocated in liposarcoma); OMIM gene 137070; "
            "ALS6 (amyotrophic lateral sclerosis type 6 with or without FTD) OMIM 608030. "
            "16p11.2; 526 aa; ~53 kDa; autosomal dominant (most), rare AR. "
            "FUNCTION: FUS is an RNA-binding protein (FET family with EWS and TAF15). "
            "It contains an N-terminal QGSY-rich low-complexity domain (LCD/prion-like domain), "
            "an RRM (RNA recognition motif), multiple RGG repeats, and a C-terminal "
            "nuclear localisation signal (NLS) recognised by transportin 1 (TRN1/TNPO1). "
            "FUNCTIONS: transcriptional regulation, pre-mRNA splicing, mRNA transport, "
            "DNA damage response, stress granule formation. "
            "ALS PATHOMECHANISM: ALS mutations cluster in the NLS (C-terminal) — "
            "most common: R521C, R521H, R521G, R522G (NLS mutations); "
            "NLS mutations → impair TRN1 binding → FUS mislocalises to cytoplasm; "
            "cytoplasmic FUS aggregates (basophilic inclusions — unlike TDP-43 which are ubiquitinated); "
            "SEVERITY GRADIENT: "
            "NLS mutations (R521C, R521H) → moderately severe; "
            "P525L → very severe early onset (juvenile, teens-20s); "
            "H517Q → rapidly fatal neonatal ALS; "
            "Truncating mutations (R495X) → severe with FTD; "
            "JUVENILE ALS — P525L: "
            "TARDBP and FUS mutations are the MOST COMMON cause of juvenile ALS (<25y); "
            "FUS P525L: disease onset <20y; aggressive; rapidly fatal within 1-3 years; "
            "clinically mimics juvenile spinal muscular atrophy initially. "
            "FUS-ALS CLINICAL FEATURES: "
            "Mean age of onset: 40-50y (adult onset NLS mutations) or <25y (P525L); "
            "predominant UMN features early (spasticity prominent); "
            "FTD less common than C9ORF72; "
            "basophilic FUS/p62 inclusions (distinct from TDP-43 ubiquitin inclusions). "
            "TREATMENT: no FUS-targeted therapy approved; "
            "transportin-mediated nuclear import restoration strategies (research); "
            "antisense oligonucleotides targeting FUS mRNA (preclinical). "
            "GENETIC COUNSELLING: AD; most mutations de novo in juvenile cases; "
            "parental mosaicism described; genetic testing of parents essential for recurrence risk."
        ),
        "locus": "16p11.2",
        "aa": 526,
        "kDa": 53,
        "omim_gene": "137070",
        "omim_disease": "608030",
        "inheritance": "AD (most) / AR (rare) — NLS mutations (C-terminal) impair transportin-1 binding → cytoplasmic FUS mislocalisation; juvenile cases often de novo; parental mosaicism documented",
        "gene_class": "RNA-binding protein (FET family; LCD/prion-like domain + RRM + RGG + NLS); nuclear-cytoplasmic shuttle via TRN1/TNPO1; basophilic FUS inclusions (distinct from TDP-43 ubiquitin pattern)",
        "key_alerts": [
            "FUS-P525L-JUVENILE-ALS-EXTREME-SEVERITY: P525L is a C-terminal NLS mutation causing juvenile ALS (onset <20y); one of most aggressive ALS variants; fatal within 1-3 years; requires immediate aggressive MDT, early NIV/tracheostomy discussion, and paediatric palliative care integration",
            "FUS-YOUNG-ONSET-TRIGGER-GENETIC-TESTING: FUS and TARDBP mutations are the most common cause of juvenile ALS (<25y); any young-onset ALS or rapidly progressive MND should have comprehensive panel including FUS sequencing before diagnosis confirmed as sporadic",
            "FUS-BASOPHILIC-INCLUSIONS-DISTINCT-TDP43: FUS inclusions are basophilic (not ubiquitinated like TDP-43); FUS-ALS cases are TDP-43 NEGATIVE on neuropathology — important for biomarker and trial eligibility; phospho-TDP-43 CSF assay will be NEGATIVE in FUS-ALS",
            "FUS-NLS-SEVERITY-GRADIENT: NLS mutations vary in severity; H517Q = neonatal fatal; P525L = juvenile lethal; R521C/H = aggressive adult; R521G = moderately aggressive; severity correlates with degree of cytoplasmic mislocalisation — genotype guides prognosis",
            "FUS-DENOVO-COUNSELLING: juvenile FUS-ALS mutations are frequently de novo — parental mosaicism possible; both parents should be tested even if mutation not found by routine sequencing; recurrence risk counselling requires full pedigree",
            "FUS-NO-APPROVED-TARGETED-THERAPY: no FDA-approved FUS-targeted therapy; transportin-restoration and FUS-ASO strategies preclinical; standard care = riluzole + edaravone + aggressive respiratory/nutritional MDT given rapid progression",
        ],
        "etiologies": [
            "R521C — most common adult FUS mutation; NLS disruption; moderate-severe progression",
            "R521H — similar to R521C; European cohorts; prominent UMN features",
            "P525L — juvenile ALS (<20y); extreme NLS disruption; fatal 1-3y; most severe adult-onset FUS",
            "H517Q — ultra-severe; neonatal/infantile onset; extremely rare; fatal within months",
            "R495X — truncating; NLS absent; very severe; FTD co-occurrence higher",
            "G156E — N-terminal LCD mutation; atypical; milder phenotype",
            "R244C — RGG2 region; intermediate severity; limb-onset predominantly",
        ],
        "stats": {
            "mean_dx_age": 44,
            "mean_dx_delay_months": 12,
            "pct_familial": 3,
            "pct_juvenile_onset": 20,
            "pct_UMN_predominant": 65,
        },
        "dx_delay_distribution": "Gaussian(mean=12, sd=5, min=3, max=30) months",
    },
    # ── C9ORF72 — ALS-FTD ────────────────────────────────────────────────────
    {
        "gene": "C9ORF72",
        "protein": "C9ORF72 — 9p21.2 AD — 481aa — GGGGCC-Hexanucleotide-Repeat-Expansion — MOST-COMMON-Familial-ALS-FTD-40pct — Repeat-Primed-PCR-Required",
        "alias": (
            "C9ORF72 (chromosome 9 open reading frame 72); OMIM gene 614260; "
            "ALS-FTD (ALS-frontotemporal dementia, chromosome 9p-linked) OMIM 105550. "
            "9p21.2; 481 aa; ~54 kDa; autosomal dominant. "
            "MUTATION: GGGGCC hexanucleotide repeat expansion (G4C2) in intron 1 of C9ORF72. "
            "Normal alleles: <30 repeats. "
            "Pathogenic expansions: typically >30 repeats, often hundreds to thousands. "
            "DETECTION: STANDARD PCR FAILS to detect large expansions — repeat-primed PCR (RP-PCR) REQUIRED "
            "as initial screen; Southern blot or long-read sequencing for precise repeat count. "
            "EPIDEMIOLOGY: "
            "~40% of ALL familial ALS; ~8% of ALL sporadic ALS; "
            "~25% of all familial FTD; most common single genetic cause of BOTH ALS and FTD; "
            "Finnish/Northern European populations: highest frequency; "
            "founder haplotype identified (single Finnish founder ~1500 years ago). "
            "PATHOMECHANISMS (THREE parallel mechanisms): "
            "1. RNA FOCI: G-quadruplex-forming RNA foci sequester RNA-binding proteins (hnRNPs) → "
            "global spliceopathy; "
            "2. DIPEPTIDE REPEAT PROTEINS (DPRs): repeat-associated non-ATG translation (RAN translation) "
            "produces 5 DPR species: poly-GA, poly-GR, poly-PR (most toxic), poly-PA, poly-GP; "
            "poly-PR and poly-GR especially neurotoxic (nucleolar stress, nuclear transport impairment); "
            "3. C9ORF72 HAPLOINSUFFICIENCY: reduced C9ORF72 protein → impaired autophagy, "
            "lysosomal trafficking, and microglial activation. "
            "CLINICAL SPECTRUM — C9ORF72 ALS-FTD CONTINUUM: "
            "Pure ALS (no cognitive symptoms): ~50%; "
            "ALS+FTD: ~15%; pure FTD: ~35%; "
            "psychosis/hallucinations: rare but reported (~5%); "
            "parkinsonism: ~5%; "
            "WITHIN FAMILY VARIABILITY: same C9ORF72 expansion → ALS in one family member, FTD in another, "
            "ALS-FTD in a third (phenotypic heterogeneity); "
            "GENETIC ANTICIPATION: not typical but repeat length can vary between generations; "
            "PENETRANCE: ~50% by age 60, ~80% by age 80 (age-dependent incomplete penetrance); "
            "TREATMENT: no approved C9ORF72-targeted therapy; "
            "antisense oligonucleotides targeting repeat RNA (BIIB078, WVE-004) — Phase I/II trials; "
            "riluzole + edaravone standard; "
            "BIOMARKER: CSF/plasma NfL elevated; DPR proteins (poly-GP) in CSF measurable; "
            "cortical atrophy on MRI (frontal/temporal)."
        ),
        "locus": "9p21.2",
        "aa": 481,
        "kDa": 54,
        "omim_gene": "614260",
        "omim_disease": "105550",
        "inheritance": "AD — GGGGCC hexanucleotide repeat expansion in intron 1; >30 repeats pathogenic; incomplete age-dependent penetrance (~50% by 60y, ~80% by 80y); intrafamily phenotypic heterogeneity (ALS vs FTD vs ALS-FTD)",
        "gene_class": "DENN-domain GEF (guanine nucleotide exchange factor); autophagy regulator; lysosomal trafficking; haploinsufficiency + toxic RNA foci + dipeptide repeat proteins (DPR) — triple pathomechanism",
        "key_alerts": [
            "C9ORF72-STANDARD-PCR-FAILS-REPEAT-PRIMED-PCR-MANDATORY: standard PCR cannot detect large C9ORF72 expansions — REPEAT-PRIMED PCR (RP-PCR) is mandatory as first-line screen; if positive RP-PCR: confirm with Southern blot or long-read sequencing; never report C9ORF72 negative based on standard PCR alone",
            "C9ORF72-MOST-COMMON-FAMILIAL-ALS-FTD: ~40% familial ALS, ~8% sporadic ALS, ~25% familial FTD — single most common genetic cause of both diseases; test C9ORF72 FIRST in all familial ALS/FTD panels before other genes",
            "C9ORF72-FTD-PSYCHIATRIC-SAME-FAMILY: within the same family, C9ORF72 expansion can cause pure ALS in one member, pure FTD (behavioural variant) in another, and ALS-FTD in a third; psychiatric symptoms (psychosis, hallucinations) can be the presenting feature — important for genetic counselling",
            "C9ORF72-INCOMPLETE-PENETRANCE-PRE-SYMPTOMATIC: penetrance ~50% by 60y and ~80% by 80y — pre-symptomatic carriers may never develop disease; genetic counselling must explain this; pre-symptomatic testing requires specialist neurogenetics counselling and psychological support",
            "C9ORF72-ASO-TRIALS-ACTIVE: BIIB078, WVE-004, and other antisense oligonucleotides targeting C9ORF72 repeat RNA in Phase I/II trials; eligible patients should be referred to ALS specialist for trial screening; disease-modifying therapy may emerge",
            "C9ORF72-COGNITIVE-SCREENING-MANDATORY: all C9ORF72-ALS patients need cognitive screening (FTD assessment: ACE-III, FAB, Ekman faces); FTD in ALS affects decision-making capacity — critical for ventilation/tracheostomy decisions and driving safety",
        ],
        "etiologies": [
            "G4C2 repeat expansion >30 (typically hundreds-thousands) — founder haplotype; Finnish/Northern European highest frequency",
            "ALS phenotype — pure motor neuron disease; ~50% of C9ORF72-ALS",
            "FTD phenotype (bvFTD) — behavioural/executive dysfunction predominant; ~35% of C9ORF72 expansion carriers",
            "ALS-FTD — concurrent motor and frontotemporal syndrome; ~15%; most severe prognosis",
            "Psychiatric phenotype — psychosis, hallucinations, depression (rare, ~5%); often misdiagnosed initially",
            "Parkinsonism — atypical parkinsonism, sometimes CBS or PSP-like; ~5%; cortical atrophy on MRI",
        ],
        "stats": {
            "mean_dx_age": 58,
            "mean_dx_delay_months": 13,
            "pct_familial": 40,
            "pct_FTD_cooccurrence": 15,
            "pct_pure_ALS": 50,
        },
        "dx_delay_distribution": "Gaussian(mean=13, sd=6, min=4, max=36) months",
    },
    # ── UBQLN2 — ALS15 ───────────────────────────────────────────────────────
    {
        "gene": "UBQLN2",
        "protein": "UBQLN2 — Xp11.21 X-linked-Dominant — Ubiquilin-2-624aa — ALS15-First-X-Linked-ALS — Proteasomal-Pathway — Ubiquilin-2-Inclusions",
        "alias": (
            "UBQLN2 (ubiquilin 2); OMIM gene 300264; "
            "ALS15 (amyotrophic lateral sclerosis type 15 with or without FTD) OMIM 300857. "
            "Xp11.21; 624 aa; ~66 kDa; X-linked dominant. "
            "FUNCTION: Ubiquilin 2 (UBQLN2) is a ubiquitin receptor/shuttle protein that delivers "
            "polyubiquitinated proteins to the 26S proteasome for degradation. "
            "It contains an N-terminal ubiquitin-like (UBL) domain (binds proteasome) "
            "and a C-terminal ubiquitin-associated (UBA) domain (binds polyubiquitin chains). "
            "It also contains a PXX repeat region (proline-rich domain) where most ALS mutations cluster. "
            "ADDITIONAL FUNCTIONS: autophagy-proteasome cross-talk; ER-associated protein degradation; "
            "stress granule regulation. "
            "ALS PATHOMECHANISM: UBQLN2 mutations → impaired proteasomal delivery → "
            "accumulation of polyubiquitinated proteins → motor neuron toxicity. "
            "UBQLN2 inclusions (positive for ubiquilin-2, ubiquitin, p62) in spinal motor neurons "
            "are PATHOGNOMONIC for UBQLN2-ALS. "
            "X-LINKED INHERITANCE FEATURES: "
            "ALS15 is X-linked dominant — males are hemizygous and typically MORE severely affected "
            "(earlier onset, faster progression); "
            "carrier females: variable expression (X-inactivation skewing); "
            "~70% penetrance in heterozygous females; "
            "males: ~100% penetrance; "
            "CLINICAL FEATURES: "
            "Males: mean onset 40-50y, rapid progression; "
            "Females: mean onset 50-60y, more variable; "
            "FTD co-occurrence in ~20% of UBQLN2 families; "
            "dementia component can be prominent (PXX domain mutations especially); "
            "MUTATIONS: P497H, P497S, P500S, P506T, P509S, P516S (PXX domain); "
            "T487I (outside PXX); T152M (UBL domain — affects proteasome binding). "
            "GENETIC COUNSELLING: X-linked dominant — sons of carrier females: 50% risk; "
            "daughters: 50% carrier (may manifest); all daughters of affected males are carriers; "
            "sons of affected males: unaffected (Y chromosome from father); "
            "TREATMENT: no UBQLN2-targeted therapy; proteasome enhancement (research); "
            "standard riluzole + edaravone; aggressive respiratory monitoring."
        ),
        "locus": "Xp11.21",
        "aa": 624,
        "kDa": 66,
        "omim_gene": "300264",
        "omim_disease": "300857",
        "inheritance": "X-linked dominant — males hemizygous (more severe, ~100% penetrance); females heterozygous (~70% penetrance, variable expression); sons of carrier females 50% risk; daughters of carrier females 50% carrier",
        "gene_class": "Ubiquitin receptor/shuttle protein (UBL + UBA + PXX proline-rich domain); delivers polyubiquitinated proteins to 26S proteasome; mutations impair proteasomal clearance → protein aggregate accumulation",
        "key_alerts": [
            "UBQLN2-X-LINKED-INHERITANCE-PATTERN: ALS15 is X-linked dominant — affected males hemizygous (most severe); carrier females variably affected; sons of affected males NEVER inherit (Y from father); daughters of affected males ALL carriers; pedigree analysis mandatory before genetic counselling",
            "UBQLN2-MALES-MORE-SEVERELY-AFFECTED: hemizygous males have earlier onset (40-50y vs 50-60y in females) and faster progression; male family members of female UBQLN2 carriers warrant urgent genetic testing and neurological surveillance",
            "UBQLN2-FTD-COGNITIVE-COMPONENT: FTD co-occurs in ~20% of UBQLN2 families especially with PXX domain mutations; cognitive and behavioural assessment mandatory at diagnosis; FTD affects ventilation consent capacity",
            "UBQLN2-INCLUSIONS-PATHOGNOMONIC: ubiquilin-2 positive inclusions in spinal motor neurons are pathognomonic for UBQLN2-ALS; neuropathological diagnosis requires ubiquilin-2 IHC (not just ubiquitin/p62 which are shared with TDP-43 pathology)",
            "UBQLN2-NO-TARGETED-THERAPY: no approved UBQLN2-directed treatment; proteasome activation and autophagy enhancement are preclinical strategies; standard of care = riluzole + edaravone + MDT",
            "UBQLN2-PXX-DOMAIN-HOTSPOT: most ALS mutations cluster in proline-rich PXX repeat domain (P497-P516); PXX mutations most strongly impair proteasomal delivery; T152M (UBL domain) is rarer but affects proteasome binding directly",
        ],
        "etiologies": [
            "P497H — most commonly reported PXX domain mutation; male-severe early onset",
            "P497S — PXX domain; X-linked dominant; variable female expression",
            "P500S — PXX; hemizygous males rapidly progressive",
            "P506T — PXX domain; FTD co-occurrence reported",
            "P509S, P516S — PXX domain; functional studies show proteasome impairment",
            "T487I — outside PXX; similar severity to PXX mutations",
            "T152M — UBL domain; impairs proteasome binding directly; rare",
        ],
        "stats": {
            "mean_dx_age": 49,
            "mean_dx_delay_months": 15,
            "pct_familial": 2,
            "pct_FTD_cooccurrence": 20,
            "pct_male_more_severe": 100,
        },
        "dx_delay_distribution": "Gaussian(mean=15, sd=7, min=4, max=42) months",
    },
    # ── VCP — ALS14/IBMPFD ────────────────────────────────────────────────────
    {
        "gene": "VCP",
        "protein": "VCP — 9p13.3 AD — Valosin-Containing-Protein-806aa — ALS14/IBMPFD — Multisystem-IBM+Paget+FTD+ALS — TDP-43-Pathology — R155H-Most-Common",
        "alias": (
            "VCP (valosin-containing protein; also p97, CDC48); OMIM gene 601023; "
            "IBMPFD1 (inclusion body myopathy with early-onset Paget disease of bone and/or frontotemporal dementia 1) / "
            "ALS14 OMIM 167320. "
            "9p13.3; 806 aa; ~97 kDa; autosomal dominant. "
            "FUNCTION: VCP/p97 is an abundant AAA+ ATPase that forms a hexameric ring structure. "
            "It functions as a molecular 'segregase' — unfolds and extracts ubiquitinated proteins "
            "from various complexes for proteasomal degradation (ERAD, NF-κB pathway, cell cycle). "
            "Key cofactors: UFD1-NPLOC4 (ER-associated degradation); "
            "p47/UBXN (membrane fusion); ATXN3 (deubiquitylation). "
            "Functions: ER-associated degradation (ERAD), Golgi membrane fusion, "
            "DNA damage response, autophagosome maturation, lysosomal pathway. "
            "VCP-ALS/IBMPFD PATHOMECHANISM: "
            "VCP mutations → impaired protein homeostasis → "
            "accumulation of poly-ubiquitinated proteins and TDP-43 aggregates. "
            "VCP-ALS has TDP-43 pathology (cytoplasmic TDP-43 inclusions) — "
            "links VCP to TDP-43 proteinopathy pathway. "
            "MULTISYSTEM DISEASE — IBMPFD: "
            "IBM (Inclusion Body Myopathy): ~90% of VCP mutation carriers by age 45; "
            "proximal > distal weakness; vacuolated muscle fibres with TDP-43 inclusions; "
            "PAGET DISEASE OF BONE: ~50% of carriers; elevated ALP; lytic lesions skull/spine; "
            "FTD: ~30% of carriers; behavioural variant FTD; "
            "ALS: ~10% of carriers; often occurs in IBM background; "
            "Cardiomyopathy: rare but reported; "
            "VARIABLE PENETRANCE AND EXPRESSIVITY: same mutation can cause IBM-only, IBM+Paget, "
            "IBM+FTD, or full IBMPFD+ALS in different family members. "
            "MUTATIONS: R155H — most common worldwide (~50% of VCP-IBMPFD); "
            "R191Q, R155C, A232E, N387S, A439S — others. "
            "TREATMENT: no VCP-targeted therapy; "
            "IBM: manage with physiotherapy; no steroids (worsen); "
            "Paget: bisphosphonates (zoledronic acid); ALP monitoring; "
            "FTD: behavioural management; "
            "ALS: riluzole + edaravone; respiratory monitoring; "
            "VCP inhibitors in preclinical ALS models (research)."
        ),
        "locus": "9p13.3",
        "aa": 806,
        "kDa": 97,
        "omim_gene": "601023",
        "omim_disease": "167320",
        "inheritance": "AD — autosomal dominant; most mutations in N-terminal D1 ATPase domain; R155H most common (~50%); high penetrance for IBM component (~90%); variable penetrance for Paget/FTD/ALS components",
        "gene_class": "AAA+ ATPase (hexameric segregase); molecular unfoldase — extracts ubiquitinated proteins for proteasomal degradation; ERAD, autophagy, DNA damage response; VCP-ALS has TDP-43 pathology",
        "key_alerts": [
            "VCP-IBMPFD-MULTISYSTEM-SCREEN-ALL-COMPONENTS: VCP mutations cause IBMPFD — screen ALL components: IBM (90%, muscle biopsy with TDP-43 IHC), Paget (ALP, bone scan, skeletal survey), FTD (neuropsychology), ALS (EMG/NCS); missing one component leads to misdiagnosis and treatment delays",
            "VCP-IBM-NO-STEROIDS-WORSEN: VCP inclusion body myopathy (IBM) should NOT be treated with corticosteroids — steroids worsen IBM and are harmful; biopsy TDP-43/p62/ubiquitin IHC distinguishes VCP-IBM from inflammatory myopathies that DO respond to steroids",
            "VCP-PAGET-BISPHOSPHONATES-MANDATORY: Paget component requires bisphosphonate treatment (zoledronic acid first-line); ALP must be monitored 6-monthly; untreated Paget → bone pain, deformity, deafness, secondary osteosarcoma (rare but preventable)",
            "VCP-TDP43-PATHOLOGY-LINKS-ALS-PATHWAY: VCP-ALS has TDP-43 cytoplasmic inclusions — VCP haploinsufficiency impairs TDP-43 clearance; important for trial eligibility (some TDP-43-targeted trials may include VCP-ALS)",
            "VCP-R155H-MOST-COMMON-GENETIC-COUNSELLING: R155H (~50% of VCP-IBMPFD) is highly penetrant for IBM; siblings of R155H carriers should be offered genetic testing; predictive testing before symptom onset allows proactive Paget surveillance and cardiac monitoring",
            "VCP-ALS-WITHIN-IBM-BACKGROUND: ALS develops in ~10% of VCP carriers, typically in patients already known to have IBM; worsening weakness + new UMN signs in IBM patient = EMG for concurrent ALS; VCP-ALS may progress faster than other familial ALS",
        ],
        "etiologies": [
            "R155H — most common worldwide (~50%); full IBMPFD spectrum; high IBM penetrance",
            "R191Q — IBM predominant; Paget less common; ALS rare",
            "R155C — similar to R155H; predominantly European cohorts",
            "A232E — FTD and IBM; Paget variable",
            "N387S — IBM + Paget; ALS rare; D2 ATPase domain",
            "A439S — IBM predominant; slower progression; D2 ATPase domain",
            "D592N — D2 ATPase; rare; muscle and bone involvement",
        ],
        "stats": {
            "mean_dx_age": 45,
            "mean_dx_delay_months": 36,
            "pct_familial": 1,
            "pct_IBM_component": 90,
            "pct_Paget_component": 50,
            "pct_FTD_component": 30,
            "pct_ALS_component": 10,
        },
        "dx_delay_distribution": "Gaussian(mean=36, sd=18, min=6, max=120) months",
    },
    # ── OPTN — ALS12 ─────────────────────────────────────────────────────────
    {
        "gene": "OPTN",
        "protein": "OPTN — 10p13 AD/AR — Optineurin-577aa — ALS12 — Autophagy-Receptor — p62-Co-immunoreactive — Haploinsufficiency — E478G-Most-Common",
        "alias": (
            "OPTN (optineurin); OMIM gene 602432; "
            "ALS12 (amyotrophic lateral sclerosis type 12) OMIM 613435. "
            "10p13; 577 aa; ~66 kDa; autosomal dominant (most), autosomal recessive (some). "
            "FUNCTION: Optineurin is a multifunctional adaptor protein with roles in: "
            "(1) NF-κB regulation (negative regulator via NEMO-related domain); "
            "(2) AUTOPHAGY RECEPTOR: optineurin targets ubiquitinated cargo for selective autophagy "
            "(xenophagy of bacteria; mitophagy; aggrephagy of protein aggregates); "
            "binds ubiquitin chains via C-terminal UBAN domain and autophagosome via LIR motif; "
            "(3) Golgi ribbon maintenance; "
            "(4) myosin VI interaction (vesicular trafficking). "
            "TBK1 phosphorylates OPTN at S177 → activates autophagy; "
            "OPTN and TBK1 form a functional pair in selective autophagy. "
            "ALS PATHOMECHANISM: "
            "OPTN loss of function → impaired selective autophagy of protein aggregates "
            "(including TDP-43, ubiquitinated proteins) → accumulation → motor neuron death. "
            "OPTN inclusions in ALS spinal neurons co-immunoreactive with p62, ubiquitin. "
            "MUTATIONS: "
            "E478G — most common OPTN-ALS mutation; loss of UBAN ubiquitin-binding → impaired aggrephagy; "
            "Q398X — truncating; AR in Japanese cohorts; slowly progressive; "
            "M98K — NF-κB regulatory domain; variable phenotype; "
            "Deletions — haploinsufficiency mechanism in AD cases. "
            "CLINICAL FEATURES: "
            "Mean onset 50-60y; predominantly limb onset; variable rate of progression; "
            "Some AR cases (Q398X) with remarkably slow progression (decades, 'ALS with very slow progression'); "
            "FTD: uncommon in OPTN-ALS; "
            "Normal cognition typical; "
            "GENETIC COUNSELLING: AD (most deletions, E478G); AR (Q398X, Japanese); "
            "AR-OPTN is an important cause of slowly progressive familial ALS in Japan; "
            "TREATMENT: no approved OPTN-targeted therapy; "
            "TBK1-OPTN pathway understanding led to clinical trials of autophagy enhancers. "
            "GLAUCOMA CONNECTION: OPTN also causes familial primary open-angle glaucoma (POAG); "
            "E50K OPTN mutation → glaucoma; E478G → ALS; different domains, different diseases."
        ),
        "locus": "10p13",
        "aa": 577,
        "kDa": 66,
        "omim_gene": "602432",
        "omim_disease": "613435",
        "inheritance": "AD (most deletions, E478G haploinsufficiency) / AR (Q398X Japanese slowly-progressive) — both mechanisms impair selective autophagy; AR-OPTN slower progression than AD",
        "gene_class": "Autophagy receptor (aggrephagy/mitophagy/xenophagy); NF-κB negative regulator; TBK1 substrate (S177); UBAN ubiquitin-binding domain; LIR autophagosome-binding; functionally paired with TBK1",
        "key_alerts": [
            "OPTN-TBK1-FUNCTIONAL-PAIR: OPTN and TBK1 function together in selective autophagy — TBK1 phosphorylates OPTN-S177 to activate cargo clearance; OPTN-ALS and TBK1-ALS share the same autophagy pathway impairment; both genes should be sequenced in familial ALS panels",
            "OPTN-AR-Q398X-SLOWLY-PROGRESSIVE: AR OPTN mutations (Q398X) in Japanese cohorts cause remarkably slowly progressive ALS (decades-long survival); contrast with rapidly fatal classic ALS; important prognostic counselling difference — do not apply standard ALS prognosis to AR-OPTN",
            "OPTN-GLAUCOMA-DIFFERENT-DOMAIN: E50K OPTN mutation causes familial glaucoma (NOT ALS); E478G OPTN mutation causes ALS (NOT glaucoma); different domains, different mechanisms; ophthalmological findings should NOT be expected in OPTN-ALS patients",
            "OPTN-AUTOPHAGY-IMPAIRMENT-TRIAL-TARGET: OPTN haploinsufficiency impairs selective autophagy of TDP-43 and ubiquitinated aggregates; autophagy-enhancing strategies (rapamycin analogues, TFEB activators) are rational therapeutic targets under investigation",
            "OPTN-INCLUSIONS-P62-CO-IMMUNOREACTIVE: optineurin inclusions in spinal motor neurons are p62-positive and ubiquitin-positive; neuropathology requires OPTN IHC specifically (not just p62/ubiquitin which are shared by TDP-43 and other pathologies)",
            "OPTN-HAPLOINSUFFICIENCY-MECHANISM: AD-OPTN ALS is primarily haploinsufficiency (one functional copy insufficient for motor neuron survival); deletions of one OPTN allele sufficient for disease — genomic copy number analysis (MLPA/array-CGH) needed alongside sequencing",
        ],
        "etiologies": [
            "E478G — most common; UBAN ubiquitin-binding impaired; AD; moderate progression",
            "Q398X — truncating; AR in Japanese; slowly progressive (decades); haploinsufficiency",
            "M98K — NF-κB regulatory domain; variable AD/AR; glaucoma link reported",
            "Exon deletions (various) — haploinsufficiency; AD; intermediate severity",
            "A100V — UBAN region; AD; rare; European cohorts",
            "H486R — near UBAN; AD; limb onset predominant",
        ],
        "stats": {
            "mean_dx_age": 55,
            "mean_dx_delay_months": 18,
            "pct_familial": 2,
            "pct_AR_slowly_progressive": 30,
            "pct_FTD_cooccurrence": 5,
        },
        "dx_delay_distribution": "Gaussian(mean=18, sd=9, min=4, max=60) months",
    },
    # ── TBK1 — ALS with FTD ──────────────────────────────────────────────────
    {
        "gene": "TBK1",
        "protein": "TBK1 — 12q14.2 AD — TANK-Binding-Kinase-1-729aa — ALS-FTD — Haploinsufficiency — Phosphorylates-OPTN-p62 — Autophagy-NF-kB-Pathway — FTD-40pct",
        "alias": (
            "TBK1 (TANK-binding kinase 1; also NAK, T2K); OMIM gene 604834; "
            "ALS with FTD (amyotrophic lateral sclerosis with frontotemporal dementia, TBK1-linked). "
            "12q14.2; 729 aa; ~84 kDa; autosomal dominant (haploinsufficiency). "
            "FUNCTION: TBK1 is a serine/threonine kinase of the IKK (I-kappa-B kinase) family. "
            "It is a multifunctional signalling hub: "
            "(1) INNATE IMMUNITY / INTERFERON SIGNALLING: "
            "TBK1 phosphorylates IRF3 and IRF7 → type I interferon (IFNα/β) production; "
            "downstream of pattern recognition receptors (cGAS-STING, RIG-I-MAVS, TLR3-TRIF); "
            "critical for antiviral defence. "
            "(2) SELECTIVE AUTOPHAGY: "
            "TBK1 phosphorylates OPTN (S177) → activates OPTN autophagy receptor function; "
            "TBK1 phosphorylates p62/SQSTM1 (S403) → increases ubiquitin binding affinity → "
            "enhanced cargo recruitment to autophagosome; "
            "TBK1 phosphorylates NDP52/CALCOCO2 (another autophagy receptor). "
            "(3) NF-κB PATHWAY: TBK1 → IKKβ axis → NF-κB inflammatory gene activation. "
            "ALS-FTD PATHOMECHANISM: "
            "TBK1 haploinsufficiency (LOF mutations, deletions) → "
            "impaired selective autophagy of TDP-43 and ubiquitinated aggregates; "
            "Neuroinflammation (impaired interferon regulatory signalling in glial cells). "
            "FTD is particularly common in TBK1-ALS (~40%) compared to other ALS genes. "
            "MUTATIONS: "
            "LOF mutations: p.E696K (impairs OPTN binding); frameshift/nonsense; "
            "exon deletions; "
            "Missense affecting kinase domain or OPTN/p62 interaction surfaces. "
            "CLINICAL FEATURES: "
            "Mean onset ~55y; predominantly limb onset; "
            "FTD in ~40% — highest FTD rate among common ALS genes (after C9ORF72); "
            "ALS + FTD prognosis: worse than pure ALS; "
            "Rapid progression in some; others slower; "
            "OVERLAP WITH FTD-TBK1: TBK1 mutations identified in pure FTD cohorts. "
            "TREATMENT: no TBK1-targeted therapy; "
            "TBK1 kinase inhibitors (for hyperactive TBK1 in other conditions) NOT appropriate for "
            "TBK1-ALS (which is haploinsufficiency — further inhibition would worsen disease); "
            "Autophagy enhancement strategies rational; riluzole + edaravone standard. "
            "IMPORTANT DISTINCTION: TBK1 kinase inhibitors (amlexanox, GSK8612) are studied for "
            "other diseases — they would be contraindicated in TBK1-ALS LOF mutations."
        ),
        "locus": "12q14.2",
        "aa": 729,
        "kDa": 84,
        "omim_gene": "604834",
        "omim_disease": "620535",
        "inheritance": "AD — haploinsufficiency (LOF mutations, deletions); TBK1 kinase inhibitors CONTRAINDICATED in TBK1-ALS (LOF disease); interferon signalling and selective autophagy pathways both impaired",
        "gene_class": "IKK-family serine/threonine kinase; IRF3/IRF7 phosphorylation (interferon); OPTN-S177 and p62-S403 phosphorylation (autophagy activation); NF-κB pathway; multisystem signalling hub",
        "key_alerts": [
            "TBK1-KINASE-INHIBITORS-CONTRAINDICATED-IN-ALS: TBK1 LOF haploinsufficiency causes ALS — TBK1 kinase inhibitors (amlexanox, GSK8612, momelotinib) used in other conditions are ABSOLUTELY CONTRAINDICATED in TBK1-ALS; further kinase inhibition would worsen autophagy impairment and accelerate neurodegeneration",
            "TBK1-FTD-40PCT-HIGHEST-AFTER-C9: FTD co-occurrence in TBK1-ALS is ~40% — second only to C9ORF72 among common ALS genes; ALL TBK1-ALS patients need full FTD cognitive assessment; FTD may dominate the clinical picture and mask the ALS diagnosis",
            "TBK1-OPTN-FUNCTIONAL-PAIR-BOTH-TEST: TBK1 and OPTN form a functional autophagy pair; if TBK1 is identified on ALS panel, OPTN should also be specifically checked (and vice versa); compound heterozygosity of TBK1+OPTN reported to modify severity",
            "TBK1-HAPLOINSUFFICIENCY-COPY-NUMBER: TBK1 deletions (whole exon or larger) cause ALS through haploinsufficiency; standard sequencing misses deletions — MLPA or array-CGH required in TBK1-ALS workup if sequencing is negative in suggestive pedigree",
            "TBK1-INNATE-IMMUNITY-NEUROINFLAMMATION: TBK1 haploinsufficiency → impaired IRF3/IFN-β signalling → dysregulated microglial interferon response → neuroinflammation component; this opens potential for interferon pathway modulation as therapy (research)",
            "TBK1-CONSENT-CAPACITY-FTD: with ~40% FTD co-occurrence, TBK1-ALS patients need early capacity assessment for ventilation, tracheostomy, and trial participation decisions; advanced care planning should begin at diagnosis, not when FTD becomes overt",
        ],
        "etiologies": [
            "E696K — impairs OPTN binding surface; most functionally studied TBK1 mutation; ALS ± FTD",
            "Frameshift/nonsense mutations (haploinsufficiency) — variable exon location; ALS-FTD spectrum",
            "Exon deletions (MLPA required) — haploinsufficiency; FTD predominant in some deletion families",
            "Kinase domain missense (LOF) — impaired OPTN/p62 phosphorylation; ALS with FTD common",
            "D135G — kinase domain; severe; young onset; FTD prominent",
            "R357Q — kinase domain; intermediate; ALS with cognitive changes",
        ],
        "stats": {
            "mean_dx_age": 55,
            "mean_dx_delay_months": 15,
            "pct_familial": 2,
            "pct_FTD_cooccurrence": 40,
            "pct_MLPA_detects_deletions": 15,
        },
        "dx_delay_distribution": "Gaussian(mean=15, sd=8, min=3, max=48) months",
    },
]


def _generate_patients():
    for idx, gene_data in enumerate(ALS_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        gene = gene_data["gene"]
        patients = []
        for i in range(40):
            stats = gene_data["stats"]
            mean_age = stats.get("mean_dx_age", 52)
            dx_delay_mean = stats.get("mean_dx_delay_months", 15)

            onset_age = max(8, int(rng.gauss(mean_age, mean_age * 0.15)))
            dx_delay = max(2, int(rng.gauss(dx_delay_mean, dx_delay_mean * 0.4)))
            age_at_dx = onset_age + round(dx_delay / 12)

            if gene == "SOD1":
                variant = rng.choice(["A4V", "D90A-het", "D90A-hom", "G93A", "I113T", "other"])
                survival_months = (
                    rng.randint(8, 16) if variant == "A4V"
                    else rng.randint(60, 480) if variant == "D90A-hom"
                    else rng.randint(18, 60)
                )
                tofersen_eligible = rng.random() > 0.3
                bulbar_onset = variant not in ("A4V",) and rng.random() > 0.75
                ftd = False
                nfl_elevated = True
                patients.append({
                    "patient_id": f"SOD1-{i+1:03d}",
                    "onset_age": onset_age, "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "variant": variant, "survival_months": survival_months,
                    "tofersen_eligible": tofersen_eligible,
                    "bulbar_onset": bulbar_onset, "ftd": ftd,
                    "nfl_elevated": nfl_elevated,
                    "gene": gene, "seed": seed,
                })

            elif gene == "TARDBP":
                variant = rng.choice(["A315T", "G294A", "M337V", "A382T", "N352S", "other"])
                ftd = variant == "A382T" and rng.random() > 0.55
                sardinian = variant == "A382T" and rng.random() > 0.5
                bulbar_onset = rng.random() > 0.7
                survival_months = rng.randint(18, 72)
                phospho_tdp43_csf = rng.random() > 0.35
                patients.append({
                    "patient_id": f"TARDBP-{i+1:03d}",
                    "onset_age": onset_age, "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "variant": variant, "ftd": ftd,
                    "sardinian": sardinian, "bulbar_onset": bulbar_onset,
                    "survival_months": survival_months,
                    "phospho_tdp43_csf": phospho_tdp43_csf,
                    "gene": gene, "seed": seed,
                })

            elif gene == "FUS":
                variant = rng.choice(["R521C", "R521H", "P525L", "R495X", "G156E", "other"])
                juvenile = variant == "P525L" or (onset_age < 30 and rng.random() > 0.6)
                if juvenile:
                    onset_age = max(12, rng.randint(12, 28))
                    dx_delay = max(3, rng.randint(3, 18))
                    age_at_dx = onset_age + round(dx_delay / 12)
                de_novo = juvenile and rng.random() > 0.55
                uMN_predominant = rng.random() > 0.35
                bulbar_onset = rng.random() > 0.8
                survival_months = (
                    rng.randint(8, 30) if variant == "P525L"
                    else rng.randint(18, 72)
                )
                ftd = variant == "R495X" and rng.random() > 0.6
                patients.append({
                    "patient_id": f"FUS-{i+1:03d}",
                    "onset_age": onset_age, "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "variant": variant, "juvenile": juvenile,
                    "de_novo": de_novo, "uMN_predominant": uMN_predominant,
                    "bulbar_onset": bulbar_onset,
                    "survival_months": survival_months, "ftd": ftd,
                    "gene": gene, "seed": seed,
                })

            elif gene == "C9ORF72":
                repeat_count = rng.choice([
                    rng.randint(31, 100),
                    rng.randint(100, 800),
                    rng.randint(800, 3000),
                ])
                phenotype = rng.choices(
                    ["ALS", "FTD", "ALS-FTD", "ALS+psychiatric"],
                    weights=[50, 35, 12, 3]
                )[0]
                ftd = "FTD" in phenotype
                bulbar_onset = rng.random() > 0.6
                rp_pcr_detected = True
                survival_months = (
                    rng.randint(8, 24) if phenotype == "ALS-FTD"
                    else rng.randint(18, 72)
                )
                cognitive_impaired = ftd or rng.random() > 0.8
                patients.append({
                    "patient_id": f"C9ORF72-{i+1:03d}",
                    "onset_age": onset_age, "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "repeat_count": repeat_count, "phenotype": phenotype,
                    "ftd": ftd, "bulbar_onset": bulbar_onset,
                    "rp_pcr_detected": rp_pcr_detected,
                    "survival_months": survival_months,
                    "cognitive_impaired": cognitive_impaired,
                    "gene": gene, "seed": seed,
                })

            elif gene == "UBQLN2":
                sex = rng.choice(["M", "M", "M", "F"])  # more males hemizygous
                variant = rng.choice(["P497H", "P497S", "P500S", "P506T", "T487I", "other"])
                ftd = rng.random() > 0.8
                hemizygous_male = sex == "M"
                onset_age = max(30, rng.randint(35, 55)) if hemizygous_male else max(40, rng.randint(45, 65))
                dx_delay = max(3, rng.randint(8, 30))
                age_at_dx = onset_age + round(dx_delay / 12)
                ubqln2_inclusions_path = rng.random() > 0.05
                survival_months = rng.randint(18, 60) if hemizygous_male else rng.randint(24, 96)
                patients.append({
                    "patient_id": f"UBQLN2-{i+1:03d}",
                    "onset_age": onset_age, "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "sex": sex, "variant": variant, "ftd": ftd,
                    "hemizygous_male": hemizygous_male,
                    "ubqln2_inclusions_path": ubqln2_inclusions_path,
                    "survival_months": survival_months,
                    "gene": gene, "seed": seed,
                })

            elif gene == "VCP":
                variant = rng.choice(["R155H", "R191Q", "R155C", "A232E", "other"])
                ibm_component = rng.random() > 0.1
                paget_component = rng.random() > 0.5
                ftd_component = rng.random() > 0.7
                als_component = rng.random() > 0.9
                alp_elevated = paget_component
                muscle_biopsy_vacuolated = ibm_component and rng.random() > 0.15
                steroids_given_incorrectly = ibm_component and rng.random() > 0.6
                dx_delay = max(6, rng.randint(18, 96))  # VCP often very delayed dx
                onset_age = max(30, rng.randint(35, 60))
                age_at_dx = onset_age + round(dx_delay / 12)
                patients.append({
                    "patient_id": f"VCP-{i+1:03d}",
                    "onset_age": onset_age, "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "variant": variant,
                    "ibm_component": ibm_component,
                    "paget_component": paget_component,
                    "ftd_component": ftd_component,
                    "als_component": als_component,
                    "alp_elevated": alp_elevated,
                    "muscle_biopsy_vacuolated": muscle_biopsy_vacuolated,
                    "steroids_given_incorrectly": steroids_given_incorrectly,
                    "gene": gene, "seed": seed,
                })

            elif gene == "OPTN":
                variant = rng.choice(["E478G", "Q398X-AR", "M98K", "deletion", "other"])
                ar_slowly_progressive = variant == "Q398X-AR"
                ftd = rng.random() > 0.95
                autophagy_impaired = True
                p62_positive_inclusions = True
                survival_months = (
                    rng.randint(180, 600) if ar_slowly_progressive
                    else rng.randint(24, 72)
                )
                glaucoma = rng.random() > 0.95  # rare coincidental
                optn_ihc_positive = rng.random() > 0.05
                patients.append({
                    "patient_id": f"OPTN-{i+1:03d}",
                    "onset_age": onset_age, "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "variant": variant,
                    "ar_slowly_progressive": ar_slowly_progressive,
                    "ftd": ftd, "autophagy_impaired": autophagy_impaired,
                    "p62_positive_inclusions": p62_positive_inclusions,
                    "survival_months": survival_months,
                    "glaucoma": glaucoma,
                    "optn_ihc_positive": optn_ihc_positive,
                    "gene": gene, "seed": seed,
                })

            else:  # TBK1
                variant = rng.choice(["E696K", "frameshift", "deletion", "D135G", "R357Q", "other"])
                ftd = rng.random() > 0.6  # 40% FTD
                deletion_requires_mlpa = variant == "deletion"
                autophagy_impaired = True
                kinase_inhibitor_risk = rng.random() > 0.85  # risk of being given kinase inhibitor
                capacity_impaired = ftd and rng.random() > 0.5
                survival_months = (
                    rng.randint(8, 24) if ftd
                    else rng.randint(18, 60)
                )
                patients.append({
                    "patient_id": f"TBK1-{i+1:03d}",
                    "onset_age": onset_age, "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "variant": variant, "ftd": ftd,
                    "deletion_requires_mlpa": deletion_requires_mlpa,
                    "autophagy_impaired": autophagy_impaired,
                    "kinase_inhibitor_risk": kinase_inhibitor_risk,
                    "capacity_impaired": capacity_impaired,
                    "survival_months": survival_months,
                    "gene": gene, "seed": seed,
                })

        gene_data["patients"] = patients


_generate_patients()


def get_overview():
    all_delays = [p.get("dx_delay_months", 0) for g in ALS_GENES for p in g["patients"]]
    all_ages = [p.get("onset_age", 50) for g in ALS_GENES for p in g["patients"]]
    genes = []
    for idx, g in enumerate(ALS_GENES):
        delays = [p.get("dx_delay_months", 0) for p in g["patients"]]
        ages = [p.get("onset_age", 50) for p in g["patients"]]
        genes.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "key_alerts": g["key_alerts"],
            "n_patients": len(g["patients"]),
        })
    return {
        "atlas": "Hereditary-ALS-MND-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary ALS/Motor Neuron Disease Atlas — "
            "SOD1 / TARDBP / FUS / C9ORF72 / UBQLN2 / VCP / OPTN / TBK1 — "
            "320 Patients (8×40, Seeds 1710–1717)"
        ),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": sum(len(g["patients"]) for g in ALS_GENES),
        "aggregate_stats": {
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "genes_covered": len(ALS_GENES),
            "patients_per_gene": 40,
        },
        "genes": genes,
        "top_alerts": [
            "SOD1-TOFERSEN-FDA2023-ALL-SOD1-ALS: tofersen (Qalsody) is the first approved disease-modifying therapy targeting a specific ALS gene — all SOD1-ALS patients must be urgently referred to ALS specialist for tofersen assessment; pre-symptomatic ATLAS trial open for mutation carriers; genetic diagnosis of SOD1-ALS is now a treatment-triggering finding",
            "C9ORF72-REPEAT-PRIMED-PCR-MANDATORY: standard PCR FAILS to detect C9ORF72 expansion — GGGGCC repeat-primed PCR (RP-PCR) is mandatory; C9ORF72 is the single most common familial ALS gene (~40%) and most common FTD gene (~25%); never exclude C9ORF72 based on standard PCR",
            "FUS-P525L-JUVENILE-LETHAL-IMMEDIATE-MDT: P525L FUS mutation causes juvenile ALS with onset <20y and survival 1-3y — immediately activate full MDT (ALS specialist, respiratory, nutrition, palliative care, genetics, paediatrics); parental mosaicism testing mandatory in apparently de novo juvenile cases",
            "TBK1-KINASE-INHIBITORS-ABSOLUTE-CI: TBK1-ALS is caused by haploinsufficiency (LOF) — TBK1 kinase inhibitors (amlexanox, GSK8612) used for other conditions are ABSOLUTELY CONTRAINDICATED and would accelerate neurodegeneration; medication review at every clinical contact",
            "VCP-IBM-STEROIDS-HARMFUL: VCP inclusion body myopathy is commonly misdiagnosed as inflammatory myopathy and treated with steroids — steroids worsen VCP-IBM; any myopathy patient on steroids with vacuolated biopsy + TDP-43 inclusions + bone involvement = stop steroids immediately, request VCP sequencing",
            "OPTN-TBK1-AUTOPHAGY-PAIR-TEST-BOTH: OPTN and TBK1 are functional autophagy partners — when either is found pathogenic, the other should be specifically evaluated; compound heterozygosity may modify severity; deletions in TBK1 require MLPA (sequencing misses them)",
            "C9ORF72-FTD-CAPACITY-ASSESSMENT: C9ORF72-ALS carries ~15% ALS-FTD and ~35% pure FTD risk within families — cognitive capacity assessment mandatory at diagnosis for ALL C9ORF72 carriers; FTD impairs consent for tracheostomy, ventilation, clinical trial participation, and driving decisions",
            "UBQLN2-X-LINKED-MALE-SEVERITY: UBQLN2-ALS is X-linked dominant — hemizygous males more severely affected (earlier onset, faster progression) than heterozygous female carriers; male relatives of carrier females urgently need genetic testing; X-linkage must be explained in family counselling",
        ],
    }


def get_breakdown():
    result = []
    for idx, g in enumerate(ALS_GENES):
        delays = [p.get("dx_delay_months", 0) for p in g["patients"]]
        ages = [p.get("onset_age", 50) for p in g["patients"]]
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "computed": {
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def get_definitions():
    return {
        "concepts": {
            "ALS Genetics — The Four-Pathway Framework": (
                "Familial ALS genes cluster into four overlapping pathogenic pathways: "
                "(1) PROTEIN HOMEOSTASIS (UPS/autophagy): SOD1 (toxic aggregates), UBQLN2 (proteasomal shuttle), "
                "VCP (AAA+ segregase), OPTN (autophagy receptor), TBK1 (autophagy kinase) — all impair "
                "clearance of misfolded proteins; "
                "(2) RNA METABOLISM: TARDBP (TDP-43 splicing master), FUS (RNA transport/splicing), "
                "C9ORF72 (RNA foci sequester RBPs) — all disrupt RNA processing in motor neurons; "
                "(3) NEUROINFLAMMATION: C9ORF72 (dipeptide repeats activate innate immunity), TBK1 (IRF3/IFN-β), "
                "OPTN (NF-κB regulation) — glial and microglial dysfunction amplify motor neuron death; "
                "(4) CYTOSKELETAL/AXONAL TRANSPORT: SOD1 aggregates, TDP-43 inclusions impair neurofilament "
                "assembly and axonal transport — motor axons are selectively vulnerable. "
                "THERAPEUTIC IMPLICATION: identifying which pathway is dominant guides trial eligibility. "
                "TDP-43 UNIFYING PATHOLOGY: TDP-43 (TARDBP) cytoplasmic inclusions are present in ~97% of "
                "ALL sporadic ALS (and most familial ALS except SOD1 and FUS) — TDP-43 is the 'final common "
                "pathway' downstream of multiple upstream gene defects. "
                "GENETIC TESTING HIERARCHY: "
                "Step 1: C9ORF72 repeat-primed PCR (most common); "
                "Step 2: SOD1 sequencing (tofersen-eligible); "
                "Step 3: TARDBP, FUS (especially young onset); "
                "Step 4: UBQLN2 (X-linked pedigrees), VCP (IBMPFD multisystem), OPTN, TBK1; "
                "Step 5: Comprehensive panel (NEK1, SETX, DCTN1, SQSTM1 etc). "
                "~65% of familial ALS and ~10% of sporadic ALS explained by known genes."
            ),
            "C9ORF72 Repeat Expansion Detection — Why Standard PCR Fails": (
                "The GGGGCC hexanucleotide repeat in C9ORF72 forms stable G-quadruplex structures that "
                "standard PCR cannot amplify efficiently beyond ~30 repeats. Pathogenic expansions are "
                "typically hundreds to thousands of repeats — standard PCR produces a single normal-size "
                "band even in fully affected patients, falsely appearing 'normal.' "
                "REPEAT-PRIMED PCR (RP-PCR): uses a primer that anneals within the repeat + flanking primer "
                "→ produces a characteristic stutter ladder extending into the repeat; a positive stutter "
                "pattern = expansion detected (not quantified); confirms expansion present. "
                "SOUTHERN BLOT: gold standard for size estimation; labour-intensive; confirms RP-PCR positive. "
                "LONG-READ SEQUENCING (Oxford Nanopore / PacBio): emerging; can characterise repeat length "
                "and methylation status simultaneously; increasingly first-line in specialist labs. "
                "REPORTING REQUIREMENT: any ALS/FTD genetic report that says 'C9ORF72 negative' must state "
                "the detection method — if only standard PCR was used, the test is INVALID for C9ORF72. "
                "REPEAT LENGTH CORRELATION: some correlation between repeat length and age of onset but "
                "weak — not used clinically; epigenetic modification (methylation) of the expansion locus "
                "modifies penetrance and phenotype."
            ),
            "SOD1-ALS and Tofersen — Precision Medicine in ALS": (
                "Tofersen (Qalsody, Biogen) is an antisense oligonucleotide (ASO) that targets SOD1 mRNA "
                "via RNase H1-mediated degradation → reduces SOD1 protein in CSF and plasma. "
                "FDA approval: April 2023 for adults with ALS caused by SOD1 gene mutation (any variant). "
                "MECHANISM: 20-mer ASO complementary to SOD1 mRNA; RNase H1 cleaves RNA:DNA duplex; "
                "reduces both mutant and wild-type SOD1 protein — important as wild-type reduction is safe. "
                "ADMINISTRATION: intrathecal injection (lumbar puncture); 3 loading doses (weeks 0, 2, 4) "
                "then maintenance every 28 days (monthly). "
                "PRISM TRIAL: Phase 3; 108 patients SOD1-ALS; primary endpoint (ALSFRS-R slope) missed "
                "at 28 weeks (SOD1 protein reduction takes time); post-hoc and biomarker analyses showed "
                "strong NfL reduction and slower ALSFRS decline in longer follow-up; FDA approved based on "
                "totality of evidence including biomarker data. "
                "BIOMARKER MONITORING: CSF SOD1 protein (direct target engagement); plasma NfL (progression "
                "rate); ALSFRS-R every 4 weeks. "
                "ATLAS TRIAL: open-label, randomised; pre-symptomatic SOD1 mutation carriers identified "
                "by rising NfL; tofersen started before symptom onset; primary endpoint = time to diagnosis; "
                "groundbreaking — first ALS presymptomatic intervention trial. "
                "ELIGIBILITY REQUIREMENTS: confirmed pathogenic SOD1 variant (not just variant of uncertain "
                "significance); adult (≥18y); ALS diagnosis (El Escorial criteria or equivalent); "
                "accessible lumbar puncture; hepatic function monitoring (AST/ALT quarterly). "
                "SIDE EFFECTS: lumbar puncture complications; papilloedema (rare, monitor); "
                "radiculopathy (inflammatory); hepatotoxicity (monitor)."
            ),
            "FTD-ALS Continuum — The Cognitive Imperative in ALS Care": (
                "ALS and frontotemporal dementia (FTD) exist on a continuum: "
                "Pure ALS (no cognitive symptoms): ~80%; ALS with cognitive impairment (ALSci): ~10%; "
                "ALS-FTD: ~5%; ALS with behavioural impairment (ALSbi): ~5%. "
                "GENETIC RISK BY GENE: "
                "C9ORF72: ~15% ALS-FTD, ~35% pure FTD; TBK1: ~40% FTD; "
                "VCP: ~30% FTD; FUS (R495X): ~20% FTD; TARDBP A382T: ~20% FTD. "
                "CLINICAL IMPORTANCE: "
                "FTD in ALS causes IMPAIRED DECISION-MAKING CAPACITY for: "
                "ventilation consent, tracheostomy decision, clinical trial participation, driving safety, "
                "financial decisions, resuscitation orders. "
                "MANDATORY SCREENING TOOLS: Addenbrooke's Cognitive Examination-III (ACE-III); "
                "Edinburgh Cognitive and Behavioural ALS Screen (ECAS) — designed for ALS limitations; "
                "Frontal Assessment Battery (FAB); Ekman emotional recognition. "
                "TIMING: screen at diagnosis, 6-monthly, and when clinical concern arises. "
                "ADVANCE CARE PLANNING: all ALS patients should complete advance care plans EARLY — "
                "FTD may supervene and remove capacity; document wishes before capacity is lost. "
                "NEUROIMAGING: frontal/temporal atrophy on MRI; FDG-PET hypometabolism; "
                "CSF: elevated phospho-TDP-43, NfL, GFAp (astrocytic activation)."
            ),
        },
        "pharmacological_distinctions": [
            "Tofersen (Qalsody) — SOD1-ALS ONLY; ASO intrathecal; FDA 2023; reduces SOD1 protein; NfL biomarker monitoring; NOT for other ALS genes; ATLAS pre-symptomatic trial ongoing",
            "Riluzole — ALL ALS; oral glutamate antagonist (blocks persistent Na+ currents + glutamate release); modest ~3 months survival benefit; nausea, elevated LFTs common; LFT monitoring q3 months",
            "Edaravone (Radicava) — ALL ALS; IV or oral suspension; free radical scavenger; FDA 2017 (IV) / 2022 (oral); modest benefit in early rapidly-progressing ALS; not TDP-43 specific",
            "AMX0035 (sodium phenylbutyrate + TUDCA, Relyvrio) — ALL ALS; dual anti-ER-stress + mitochondrial protection; FDA 2022; withdrew from market 2024 after Phase 3 failed; not currently available",
            "TBK1 kinase inhibitors (amlexanox, GSK8612) — CONTRAINDICATED in TBK1-ALS LOF; would worsen haploinsufficiency disease; used in other conditions (NASH, lupus etc) — critically important drug-disease interaction",
            "Steroids (any) — CONTRAINDICATED in VCP-IBM; also contraindicated in DYSF (dysferlinopathy) and GNE myopathy; inflammatory muscle biopsy in IBM/VCP misled many prescribers; genetic diagnosis prevents steroid harm",
            "Zoledronic acid (Zometa) — for VCP-Paget component; IV bisphosphonate; first-line Paget treatment; suppress ALP to normal; repeat imaging after 6 months; dental review before administration (jaw osteonecrosis risk)",
            "Nusinersen / risdiplam / onasemnogene — SMA treatments NOT for hereditary ALS; SMN1 testing must rule out SMA in young-onset lower motor neuron predominant disease before ALS panel",
        ],
        "key_standards": [
            "Gold Standards of Practice — ALS Society of America (ALSA), European ALS Consortium (ENCALS), EFNS/ENS ALS guidelines: comprehensive MDT care, regular respiratory monitoring (FVC q3m), early NIV when FVC <50%, PEG when oral intake inadequate, specialist genetic counselling for all familial ALS",
            "El Escorial Revised Criteria (Brooks 2000) / Gold Coast Criteria (2020) — ALS diagnosis requires upper AND lower motor neuron signs in ≥2 body regions; mandatory to exclude mimics before genetic testing; EMG essential",
            "ATLAS Pre-symptomatic Trial (Biogen): tofersen for SOD1 mutation carriers with rising NfL but pre-symptom — first ALS prevention trial; refer ALL pre-symptomatic SOD1 carriers to ALS specialist immediately",
            "C9ORF72 Detection Standard: repeat-primed PCR as first-line screen; Southern blot or long-read sequencing for confirmation and approximate sizing; NEVER report C9ORF72 negative based on standard PCR alone (laboratory protocol critical)",
            "PRISM Trial (tofersen): Phase 3 SOD1-ALS; 28-week primary endpoint missed; totality of evidence (NfL, SOD1 protein, 12-month ALSFRS-R analyses) supported FDA approval April 2023; now standard of care for SOD1-ALS",
            "FOCUS-C9 Trial (afinersen/ISIS-C9Rx): C9ORF72 ASO; Phase 1/2; reduces DPR poly-GP in CSF; further trials ongoing — C9ORF72 precision therapy approaching",
            "VCP-IBMPFD Management: ALP every 6 months (Paget surveillance); bone scan if ALP elevated; bisphosphonate for Paget; AVOID steroids in IBM; annual cardiac echo; neuropsychological assessment annually; IBM physiotherapy (resistance training has evidence)",
            "ALS Genetic Testing Guidelines (EFNS 2012 + updates): genetic counselling BEFORE testing in familial ALS; panel recommended (not sequential single-gene); cascade testing of at-risk relatives after index case identified; pre-symptomatic testing only with specialist counselling",
        ],
    }
