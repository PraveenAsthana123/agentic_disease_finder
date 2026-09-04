#!/usr/bin/env python3
"""MND-Atlas — Complete 8-Gene Motor Neuron Disease / ALS Atlas
SOD1     (Superoxide Dismutase 1; AD/AR; 154 aa; 21q22.11; ALS1; tofersen ASO FIRST APPROVED gene-specific ALS treatment) ·
C9orf72  (Chromosome 9 open reading frame 72; AD; intronic hexanucleotide repeat; 9p21.2; ALS-FTD; most common fALS gene) ·
TARDBP   (TDP-43; AD; 414 aa; 1p36.22; ALS10; pathological TDP-43 in 97% ALL ALS regardless of gene) ·
FUS      (Fused in Sarcoma; AD/AR; 526 aa; 16p11.2; ALS6; juvenile onset, most aggressive; cytoplasmic mis-localisation) ·
VCP      (Valosin-Containing Protein; AD; 806 aa; 9p13.3; ALS14/MSP; IBM + FTD + ALS + Paget = VCP tetrad) ·
SETX     (Senataxin; AD; 2677 aa; 9q34.13; ALS4 Juvenile; DNA-RNA helicase; slow progression, allelic AOA2) ·
NEK1     (NIMA-Related Kinase 1; AD; 1258 aa; 4q33; ALS22; ~3% fALS; DNA damage repair) ·
UBQLN2   (Ubiquilin 2; XLD; 624 aa; Xp11.21; ALS15; X-linked dominant; ubiquitin-proteasome pathway; rapid juvenile course)
320-patient aggregate cohort (8 × 40, seeds 1046–1053)

Motor Neuron Disease / ALS — Key Neurological Principles:
  - DEFINITION: ALS (amyotrophic lateral sclerosis) is a progressive, fatal neurodegenerative disease
    affecting both upper motor neurons (UMN: Betz cells in motor cortex) and lower motor neurons
    (LMN: anterior horn cells + brainstem nuclei). Combined UMN + LMN signs are pathognomonic.
    Median survival 2–5 years from symptom onset (bulbar onset worse; respiratory failure = death).
    ~10% of ALS is familial (fALS); ~90% sporadic (sALS). >40 ALS genes identified.
  - CRITICAL TREATMENT RULES:
    (1) SOD1: TOFERSEN (QALSODY) — first gene-specific ALS treatment approved by FDA (April 2023).
        Intrathecal ASO that reduces SOD1 protein in CSF. Eligibility: SOD1 variant confirmed + ALS.
        Dramatically reduces neurofilament light chain (NfL) — a surrogate marker of neurodegeneration.
        Must start early; patients with presymptomatic SOD1 variants should be monitored.
    (2) C9orf72: REPEAT EXPANSION — do NOT use standard sequencing (misses the repeat); MUST use
        repeat-primed PCR + Southern blot for accurate sizing. >30 GGGGCC repeats = pathogenic.
        Up to 50% develop FTD — screen ALL C9orf72 patients + families for cognitive/behavioural change.
        Emotional lability (pseudobulbar affect, PBA) managed with dextromethorphan/quinidine.
    (3) TDP-43 PATHOLOGY: UNIVERSAL IN ALS — pathological TDP-43 aggregates found in 97% of ALL ALS
        (sporadic + familial, except SOD1 and FUS which have distinct pathology).
        This makes TDP-43 (TARDBP) THE biomarker of ALS regardless of causative gene.
    (4) ALL ALS: Riluzole MANDATORY first-line (prolongs survival ~3 months; inhibits glutamate release).
        Edaravone IV approved for some patients (free radical scavenger, benefit modest).
        RESPIRATORY: FVC monitoring mandatory; NIV significantly extends survival (+~7 months);
        start when FVC <50% predicted or significant orthopnoea. PEG when dysphagia severe.

COHORT: 8 × 40 = 320 patient slots (seeds 1046–1053; gene-specific seeds)
"""

import random

SEED_BASE = 1046

MND_GENES = [
    # ── SOD1 — Superoxide Dismutase 1 ─────────────────────────────────────
    {
        "gene": "SOD1", "protein": "Cu/Zn Superoxide Dismutase",
        "alias": "SOD1; OMIM gene 147450; OMIM disease #105400 ALS1; 21q22.11; ~20% fALS, 2% all ALS; tofersen (QALSODY) FDA approved 2023",
        "aa": "154 aa", "kDa": "16 kDa (homodimer 32 kDa)",
        "gene_class": (
            "SOD1 encodes Cu/Zn superoxide dismutase, a ubiquitously expressed homodimeric cytoplasmic "
            "enzyme (154 aa per subunit) that catalyses dismutation of superoxide radicals to H2O2 + O2, "
            "protecting cells from oxidative stress. MECHANISM OF DISEASE: >200 pathogenic variants; "
            "virtually all pathogenic through GAIN OF TOXIC FUNCTION (not loss of enzymatic activity). "
            "Misfolded SOD1 protein forms cytoplasmic aggregates → motor neuron death. "
            "TOFERSEN MECHANISM: intrathecal antisense oligonucleotide (ASO) complementary to SOD1 mRNA "
            "→ RNase H-mediated SOD1 mRNA cleavage → reduced SOD1 protein in CSF/spinal cord → "
            "slows neurodegeneration (measured by NfL). FDA-approved April 2023 (QALSODY). "
            "SOD1-ALS has DISTINCT pathology from TDP-43 ALS: SOD1 protein aggregates (NOT TDP-43 inclusions). "
            "Penetrance: most AD mutations near-complete; p.Ala4Val (A4V) most common US variant — "
            "extremely aggressive (mean survival 1.2 y from symptom onset)."
        ),
        "mnd_group": "Oxidative Stress / Toxic Protein Aggregation (SOD1 GOF)",
        "mnd_type": "ALS1 — Familial ALS / SOD1-ALS",
        "locus": "21q22.11", "omim_gene": 147450, "omim_disease": 105400,
        "inheritance": (
            "Predominantly AD (autosomal dominant, most variants); rare true AR (e.g., p.Asp90Ala/"
            "D90A — homozygous AR in Scandinavian families, slow progression; heterozygous D90A = "
            "reduced penetrance AD in these families). XL excluded. "
            "Penetrance: most AD SOD1 variants >90% lifetime penetrance. "
            "~20% of all familial ALS (fALS) globally. ~2% of all ALS (sporadic + familial combined). "
            "p.Ala4Val (A4V): ~50% of SOD1-ALS in North America; most aggressive (12–18 months survival). "
            "p.Glu100Gly (E100G): similar severity. p.Asp90Ala (D90A): uniquely slow (>10 y possible)."
        ),
        "phenotype": (
            "ONSET: typically 50s–60s (AD); younger (30s–40s) in aggressive variants. "
            "CLASSIC: mixed UMN + LMN signs. Limb-onset in most SOD1. "
            "A4V: rapidly ascending spinal ALS; predominantly LMN signs; bulbar involvement late; "
            "survival typically 12–18 months from symptom onset. "
            "D90A (homozygous): exceptionally slow — leg cramps and weakness beginning in 30s–40s; "
            "preserved bulbar function; survival >10 years common. "
            "BULBAR ONSET: ~10–15% SOD1 (less than idiopathic ALS); dysarthria, dysphagia, emotional lability. "
            "COGNITIVE IMPAIRMENT: usually ABSENT in SOD1-ALS (TDP-43 pathology required for cognitive involvement). "
            "RESPIRATORY: FVC decline mandatory monitoring; NIV survival benefit. "
            "CK: mildly elevated in many (~300–800 IU/L from denervation). "
            "SOD1 aggregates in CSF/plasma: new biomarker (not TDP-43, unlike most ALS)."
        ),
        "disease": (
            "ALS1 — SOD1-ALS (familial and de novo). Diagnosis: gene panel + EMG (fibrillations + PSW "
            "in ≥3 regions for El Escorial criteria) + MRI (excluding mimics). "
            "SPECIFIC THERAPY: Tofersen 100 mg intrathecal every 4 weeks (after 3 loading doses); "
            "monitor CSF NfL (target reduction >50%). "
            "GENERAL ALS: Riluzole 50 mg BD; edaravone IV (selected patients); MDT care; "
            "NIV; PEG; communication aids; palliative planning."
        ),
        "treatment_options": [
            "TOFERSEN (QALSODY) 100 mg IT: loading doses weeks 1, 3, 5 → maintenance q28 days; "
            "dramatically reduces CSF SOD1 protein + NfL; approved FDA 2023; FIRST gene-specific ALS treatment",
            "Riluzole 50 mg BD (first-line ALL ALS; modest survival benefit ~3 months; monitor LFTs at 3–6 months)",
            "Edaravone IV 60 mg/day × 14 days, rest 14 days; 6-cycle courses (free radical scavenger; "
            "benefit mainly in early rapidly declining patients — ALSFRS-R criteria for eligibility)",
            "NIV (BiPAP): start when FVC <50% predicted or orthopnoea; survival benefit ~7 months",
            "PEG/RIG: when dysphagia prevents adequate nutrition or aspiration risk; best placed before FVC <50%",
            "Riluzole + tofersen combination: standard in eligible SOD1-ALS",
            "MDT: neurology + respiratory + gastroenterology + speech + OT + palliative care",
            "Presymptomatic SOD1 variant carriers: neurological monitoring + NfL tracking; "
            "tofersen in presymptomatic SOD1 at risk of ALS — clinical trial (ATLAS study)",
        ],
        "key_ddx": [
            "Sporadic ALS (TDP-43 pathology vs SOD1 aggregates — pathologically distinct)",
            "Kennedy Disease / SBMA (AR, all LMN, sensory neuropathy, gynecomastia — check AR CAG repeat)",
            "Hereditary Spastic Paraplegia + amyotrophy (SPAST, SPG7 — UMN dominant)",
            "Multi-focal motor neuropathy with conduction block (MMN — treatable, anti-GM1 Ab)",
            "Cervical myelopathy + radiculopathy (MRI spine essential)",
            "Post-polio syndrome (prior poliovirus infection history)",
        ],
        "onset_range_y": (30, 75),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "ftd_risk": False,  # SOD1 typically NO cognitive impairment
        "bulbar_onset": True,
        "juvenile_onset": False,
        "xlinked": False,
        "very_slow_progression": True,  # D90A subtype
        "gene_therapy_available": True,  # tofersen
        "ck_range": (200, 800),
        "survival_y_range": (1.0, 12.0),
    },
    # ── C9orf72 — hexanucleotide repeat expansion ──────────────────────────
    {
        "gene": "C9orf72", "protein": "C9orf72 (function uncertain; GEF-like, autophagy regulatory)",
        "alias": "C9orf72; OMIM disease #105400 (ALS/FTD1); 9p21.2; GGGGCC repeat >30 copies = pathogenic; 30-40% fALS, 5-10% sALS; most common ALS gene",
        "aa": "481 aa (isoform 2)", "kDa": "54 kDa",
        "gene_class": (
            "C9orf72 harbours an intronic GGGGCC (G4C2) hexanucleotide repeat expansion (HRE) in intron 1. "
            "Normal: <10 repeats. Pathogenic: >30 repeats (most disease-associated >100; some >1000 repeats). "
            "TRIPLE MECHANISM OF TOXICITY: (1) Loss of C9orf72 protein (haploinsufficiency → impaired autophagy); "
            "(2) RNA gain-of-function — sense and antisense GGGGCC repeat RNA foci sequester RNA-binding proteins; "
            "(3) Dipeptide repeat proteins (DPRs: poly-GA, poly-GR, poly-PR, poly-PA, poly-GP) synthesised by "
            "repeat-associated non-ATG (RAN) translation → cytoplasmic inclusions → toxicity. "
            "TDP-43 pathology: present in C9orf72-ALS (same as sporadic ALS). "
            "FTD OVERLAP: ~50% of C9orf72-ALS develop FTD or cognitive/behavioural changes. "
            "C9orf72 is also the most common cause of familial FTD (~25% fFTD). "
            "DIAGNOSTIC PITFALL: Standard Sanger sequencing and even NGS SHORT-READ panels MISS the repeat — "
            "MUST use repeat-primed PCR ± Southern blot for sizing. "
            "Fragment analysis: confirms expansion; does NOT size accurately; Southern blot = gold standard for sizing."
        ),
        "mnd_group": "RNA Toxicity / Dipeptide Repeat Protein / TDP-43 Pathology",
        "mnd_type": "ALS-FTD1 — C9orf72 repeat expansion (most common familial ALS/FTD)",
        "locus": "9p21.2", "omim_gene": 614260, "omim_disease": 105400,
        "inheritance": (
            "AD with reduced penetrance (age-dependent). Penetrance ~50% by age 60, >90% by age 80. "
            "De novo expansions occur but rare. "
            "~30-40% of all familial ALS globally (highest in European populations). "
            "~5-10% of sporadic ALS (sALS). ~25% of familial FTD. "
            "Family history often absent (reduced penetrance + FTD presenting in different family members). "
            "ETHNIC VARIATION: HRE frequency higher in European-ancestry populations; "
            "lower in Asian populations (different haplotype background)."
        ),
        "phenotype": (
            "ONSET: typically 50s (range 35–80). "
            "ALS PHENOTYPE: mixed UMN + LMN; bulbar onset more frequent than SOD1 (~30%). "
            "FTD FEATURES: up to 50% develop behavioural-variant FTD (disinhibition, apathy, compulsions), "
            "executive dysfunction, language impairment (non-fluent PPA). "
            "COGNITIVE IMPAIRMENT: even ALS patients without frank FTD show executive deficits. "
            "PSYCHOSIS: rare; schizophrenia-like presentations reported in C9orf72 families. "
            "EMOTIONAL LABILITY (PBA): common; crying/laughing uncontrolled; dextromethorphan/quinidine (Nuedexta). "
            "SEIZURES: rare but reported. "
            "CEREBELLAR FEATURES: in some individuals. "
            "PROGNOSIS: median survival similar to sporadic ALS (25–36 months from symptom onset). "
            "CK: mildly elevated (denervation)."
        ),
        "disease": (
            "C9orf72-ALS/FTD — most common familial ALS gene. Diagnosis: repeat-primed PCR + Southern blot "
            "(NOT short-read NGS alone). Must screen ALL C9orf72 patients AND families for FTD/cognitive symptoms. "
            "TREATMENT: Riluzole + edaravone (no gene-specific therapy yet approved; antisense trials ongoing). "
            "FTD management: SSRI (apathy), carbamazepine/levetiracetam (disinhibition), behavioural therapy. "
            "PBA: dextromethorphan/quinidine (Nuedexta) FDA-approved."
        ),
        "treatment_options": [
            "Riluzole 50 mg BD (first-line ALL ALS; modest survival benefit; LFT monitoring)",
            "Edaravone IV (free radical scavenger; selected patients with early rapidly declining ALS)",
            "Dextromethorphan/quinidine (Nuedexta) for pseudobulbar affect (PBA) — laughing/crying uncontrolled",
            "SSRI (e.g., sertraline) for behavioural FTD — apathy, disinhibition, depression",
            "Cognitive assessment mandatory — ALL C9orf72 patients + family members neuropsychology referral",
            "NIV (BiPAP) when FVC <50%; PEG when dysphagia severe",
            "Antisense oligonucleotide (ASO) trials targeting C9orf72 repeat RNA — in clinical trials",
            "Genetic counselling: reduced penetrance; full family screening recommended; "
            "presymptomatic carriers: annual neurological + neuropsychological monitoring",
        ],
        "key_ddx": [
            "Sporadic ALS-FTD (C9orf72 screen mandatory in ALL ALS/FTD overlap)",
            "Behavioural-variant FTD without ALS (C9orf72 most common genetic cause of bvFTD)",
            "Primary progressive aphasia (PPA) — C9orf72 can present here",
            "Huntington disease (misdiagnosed if psychiatric onset) — check C9orf72 HRE before HD genetic screen",
            "SOD1-ALS (no cognitive impairment — if FTD present, think C9orf72)",
            "GRN mutations (frontotemporal lobar degeneration with TDP-43 — clinically similar bvFTD)",
        ],
        "onset_range_y": (35, 80),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "ftd_risk": True,
        "bulbar_onset": True,
        "juvenile_onset": False,
        "xlinked": False,
        "very_slow_progression": False,
        "gene_therapy_available": False,
        "ck_range": (150, 600),
        "survival_y_range": (1.5, 4.0),
    },
    # ── TARDBP — TDP-43 ────────────────────────────────────────────────────
    {
        "gene": "TARDBP", "protein": "TAR DNA-Binding Protein (TDP-43)",
        "alias": "TARDBP; OMIM gene 605078; OMIM disease #612069 ALS10; 1p36.22; 414 aa; TDP-43 pathology in 97% ALL ALS",
        "aa": "414 aa", "kDa": "43 kDa",
        "gene_class": (
            "TARDBP encodes TDP-43 (Transactive Response DNA Binding Protein 43 kDa), a nuclear RNA-binding "
            "protein with 2 RNA Recognition Motifs (RRM1, RRM2) and a C-terminal glycine-rich prion-like domain "
            "(LCD) critical for phase separation and stress granule formation. "
            "NORMAL FUNCTION: pre-mRNA splicing, miRNA biogenesis, RNA transport, stress granule dynamics. "
            "MECHANISM OF DISEASE: TDP-43 pathological hallmarks = nuclear clearance + cytoplasmic inclusions "
            "(ubiquitinated, phosphorylated, truncated fragments). Seen in: 97% sporadic ALS, 97% familial ALS "
            "(except SOD1 and FUS which have their own distinct inclusions), ~50% of all FTD. "
            "ALS10-causative TARDBP mutations: >60 identified; predominantly in the prion-like LCD; AD. "
            "Mutations CAUSE ALS and simultaneously CAUSE TDP-43 pathology. "
            "p.Ala382Thr (A382T): most common pathogenic variant in Sardinian population (~50% of Sardinian fALS). "
            "p.Gly298Ser, p.Met337Val, p.Glu384Lys also well-characterised."
        ),
        "mnd_group": "RNA-Binding Protein / Universal ALS Biomarker (TDP-43)",
        "mnd_type": "ALS10 — TARDBP-ALS / TDP-43 Proteinopathy",
        "locus": "1p36.22", "omim_gene": 605078, "omim_disease": 612069,
        "inheritance": (
            "AD (autosomal dominant). High penetrance for most variants. "
            "~4% of fALS; <1% of sALS (but pathological TDP-43 in >97% sALS regardless). "
            "De novo mutations reported. "
            "IMPORTANT DISTINCTION: having TARDBP pathological variant (causing ALS10) is separate from "
            "having TDP-43 pathology in your motor neurons — the latter occurs in nearly ALL ALS regardless "
            "of causative gene. TARDBP mutations = CAUSE of some ALS + PATHOLOGICAL MARKER for virtually all ALS."
        ),
        "phenotype": (
            "ONSET: typically 50s–70s. "
            "PHENOTYPE: classic ALS (mixed UMN + LMN); limb-onset or bulbar-onset; "
            "clinically overlaps with sporadic ALS — often indistinguishable without genetic testing. "
            "FTD overlap: ~30% TARDBP-ALS develop cognitive/behavioural features (less than C9orf72). "
            "PROGRESSION: similar rate to sporadic ALS (median survival ~2–3 years). "
            "p.Ala382Thr: Sardinian founder; may have slightly slower course. "
            "TDP-43 cytoplasmic inclusions visible on pathology: "
            "phospho-TDP-43 immunostaining (pTDP-43) is gold standard diagnostic neuropathological marker. "
            "CK: mildly elevated (denervation). "
            "NfL (neurofilament light chain): elevated in CSF and blood — "
            "biomarker of neurodegeneration, not specific to TARDBP."
        ),
        "disease": (
            "ALS10 — TARDBP-ALS. Diagnosis: gene panel + EMG + MRI. "
            "Treatment: riluzole + edaravone (no gene-specific therapy approved yet). "
            "TDP-43-targeted approaches in trials (antisense oligonucleotides, protein degraders). "
            "GENETIC COUNSELLING: high penetrance AD — first-degree relatives 50% risk."
        ),
        "treatment_options": [
            "Riluzole 50 mg BD (first-line ALL ALS)",
            "Edaravone IV (selected patients; early rapidly declining ALS)",
            "NIV (BiPAP) when FVC <50%; significantly extends survival",
            "PEG/RIG for dysphagia (before FVC <50% if possible)",
            "TDP-43-ASO clinical trials (TARDBP-specific and TDP-43 pathology-targeting)",
            "MDT care: neurology + respiratory + gastroenterology + speech-language + OT + palliative",
            "NfL monitoring (CSF or serum) for disease activity",
            "Genetic counselling: 50% familial risk for AD variants; predictive testing available",
        ],
        "key_ddx": [
            "Sporadic ALS (clinically indistinguishable; gene panel differentiates)",
            "C9orf72-ALS/FTD (FTD overlap higher in C9orf72; repeat-primed PCR for C9orf72 first)",
            "FTLD-TDP (frontotemporal lobar degeneration with TDP-43 inclusions — overlap spectrum)",
            "Multifocal motor neuropathy (MMN — LMN only, conduction block, anti-GM1 Ab, treatable)",
            "Kennedy disease SBMA (AR, all LMN, sensory NCS abnormal, gynecomastia, androgen receptor CAG)",
            "Cervical myelopathy + radiculopathy (MRI spine essential; no LMN tongue wasting in myelopathy)",
        ],
        "onset_range_y": (40, 78),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "ftd_risk": True,
        "bulbar_onset": True,
        "juvenile_onset": False,
        "xlinked": False,
        "very_slow_progression": False,
        "gene_therapy_available": False,
        "ck_range": (150, 650),
        "survival_y_range": (1.5, 4.0),
    },
    # ── FUS — Fused in Sarcoma ─────────────────────────────────────────────
    {
        "gene": "FUS", "protein": "Fused in Sarcoma (FUS/TLS)",
        "alias": "FUS; TLS; OMIM gene 137070; OMIM disease #608030 ALS6; 16p11.2; 526 aa; JUVENILE onset; most aggressive ALS; cytoplasmic mis-localisation hallmark",
        "aa": "526 aa", "kDa": "53 kDa (includes NLS; nuclear → cytoplasmic shift in disease)",
        "gene_class": (
            "FUS (Fused in Sarcoma / TLS) is a ubiquitously expressed nuclear RNA-binding protein belonging "
            "to the FET family (FUS, EWSR1, TAF15), sharing with TDP-43 a C-terminal prion-like domain (LCD). "
            "Contains: N-terminal SYGQ-rich prion-like LCD + RRM + RGG boxes + zinc finger + C-terminal NLS. "
            "NORMAL: predominantly nuclear; RNA processing, DNA damage response, transcription regulation. "
            "MECHANISM: ALS6 mutations predominantly in C-terminal NLS (nuclear localisation signal) → "
            "nuclear export or retention failure → cytoplasmic FUS mis-localisation → FUS-positive "
            "cytoplasmic inclusions (distinct from TDP-43 inclusions — FUS-ALS is TDP-43 NEGATIVE). "
            "FUS AND TARDBP ARE MUTUALLY EXCLUSIVE pathologically: confirming FUS pathology rules out TDP-43 "
            "as the inclusions. "
            "IMPORTANT: p.Pro525Leu (P525L) and p.Arg521Cys (R521C) = classic aggressive variants. "
            "p.His517Gln: milder AD adult-onset. "
            "TRUE AR FUS VARIANTS: very rare; extremely aggressive neonatal ALS (lethal in infancy)."
        ),
        "mnd_group": "RNA-Binding Protein / FET Family / FUS Mis-localisation (TDP-43 NEGATIVE)",
        "mnd_type": "ALS6 — FUS-ALS (juvenile/young adult onset, aggressive)",
        "locus": "16p11.2", "omim_gene": 137070, "omim_disease": 608030,
        "inheritance": (
            "Predominantly AD (dominant-negative). Most NLS mutations AD. "
            "Rare true AR (biallelic) → neonatal/infantile lethal ALS (reported in consanguineous families). "
            "De novo AD mutations: common, especially aggressive juvenile variants (p.Pro525Leu). "
            "~4–5% of fALS; ~1% of sALS. "
            "PENETRANCE: high for aggressive NLS mutations. Variable for milder variants."
        ),
        "phenotype": (
            "ONSET: hallmark of FUS-ALS is EARLY ONSET — most common cause of juvenile ALS (<25 years). "
            "Range: 12–60 years (NLS mutations younger; C-terminal non-NLS variants older). "
            "p.Pro525Leu: aggressive; onset 15–25 years; survival <12 months; frequently respiratory. "
            "p.Arg521Cys: onset 30–45 years; moderately aggressive (survival ~2–3 y). "
            "PHENOTYPE: Mixed UMN + LMN; often LIMB-ONSET with prominent UMN signs early; "
            "cognitive and behavioural features LESS COMMON than C9orf72 or TDP-43 proteinopathies. "
            "FUS-positive inclusions on pathology: cytoplasmic FUS inclusions; TDP-43 NEGATIVE — "
            "important diagnostic neuropathological distinction. "
            "ESSENTIAL DIAGNOSIS: ANY ALS in patient <40 years = MUST CHECK FUS. "
            "NfL markedly elevated in aggressive FUS variants. "
            "CK: mildly elevated."
        ),
        "disease": (
            "ALS6 — FUS-ALS. Most common cause of juvenile ALS. Diagnosis: gene panel (FUS sequencing) "
            "+ EMG + brain/spine MRI. Pathology: FUS cytoplasmic inclusions, TDP-43 negative. "
            "No gene-specific therapy approved; riluzole + edaravone + aggressive MDT support. "
            "Juvenile FUS-ALS: early NIV planning critical; progression faster than adult-onset ALS."
        ),
        "treatment_options": [
            "Riluzole 50 mg BD (first-line ALL ALS; modest survival benefit; LFT monitoring)",
            "Edaravone IV (selected patients; may benefit aggressive early FUS-ALS phenotype)",
            "EARLY NIV planning: juvenile FUS-ALS progresses rapidly; pre-emptive NIV discussion essential",
            "PEG/RIG for dysphagia — consider early given rapid progression in p.P525L",
            "FUS-targeted ASO or small molecule trials (preclinical/early phase trials)",
            "MDT: early introduction of palliative care in aggressive juvenile variants",
            "Genetic counselling: de novo mutations in juvenile FUS-ALS; sibling risk low if de novo confirmed",
        ],
        "key_ddx": [
            "Juvenile ALS other causes: ALS4/SETX (slow progression distinguishes), ALS15/UBQLN2",
            "Kennedy disease SBMA (all LMN only, sensory neuropathy on NCS, CAG repeat AR gene — not in juvenile)",
            "SMA type 4 (LMN only; no UMN signs; SMN1 deletion)",
            "ALS-FTD with FUS pathology vs C9orf72 (FUS pathology TDP-43 negative; C9orf72 is TDP-43 positive)",
            "Spinal muscular atrophy with respiratory distress (SMARD1/IGHMBP2 — neonatal; diaphragm paralysis first)",
            "Post-infectious myelitis / ADEM mimicking MND in young patients",
        ],
        "onset_range_y": (12, 60),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "ftd_risk": False,
        "bulbar_onset": False,
        "juvenile_onset": True,  # hallmark
        "xlinked": False,
        "very_slow_progression": False,
        "gene_therapy_available": False,
        "ck_range": (100, 500),
        "survival_y_range": (0.7, 4.0),
    },
    # ── VCP — Valosin-Containing Protein ──────────────────────────────────
    {
        "gene": "VCP", "protein": "Valosin-Containing Protein (VCP/p97)",
        "alias": "VCP; p97; OMIM gene 601023; OMIM disease #167320 IBMPFD1/MSP; 9p13.3; 806 aa; Multisystem Proteinopathy: IBM + FTD + ALS + Paget = tetrad",
        "aa": "806 aa", "kDa": "97 kDa (hexamer ~582 kDa)",
        "gene_class": (
            "VCP (Valosin-Containing Protein, also called p97 or CDC48) is a AAA-ATPase (ATPases Associated "
            "with diverse Activities) that forms a hexameric ring. Critical role in ubiquitin-proteasome "
            "pathway (UPS): extracts ubiquitinated proteins from membranes/chromatin for proteasomal "
            "degradation. Also functions in ER-associated degradation (ERAD), autophagy, DNA damage response. "
            "MECHANISM OF DISEASE: AD VCP mutations → aberrant protein quality control → ubiquitinated "
            "protein accumulation → multi-tissue proteinopathy. TDP-43 inclusions present in VCP disease "
            "(VCP = 'master regulator' of TDP-43 clearance). "
            "VCP TETRAD (Multisystem Proteinopathy / IBMPFD): "
            "(1) Inclusion Body Myopathy (IBM) — most common presentation; proximal + distal weakness; "
            "rimmed vacuoles on biopsy; TDP-43 inclusions; NOT inflammatory (no response to steroids); "
            "(2) Paget Disease of Bone — lytic bone lesions, alkaline phosphatase elevated; "
            "(3) Frontotemporal Dementia (FTD) — cognitive + behavioural; TDP-43 pathology; "
            "(4) ALS — less common; present in ~10% of VCP kindreds. "
            "Not all patients develop all four — IBM most penetrant (~90%); Paget ~40%; FTD ~30%; ALS ~10%."
        ),
        "mnd_group": "Ubiquitin-Proteasome Pathway / Multisystem Proteinopathy (VCP-p97)",
        "mnd_type": "ALS14 / MSP-IBMPFD — VCP-related Multisystem Proteinopathy with ALS",
        "locus": "9p13.3", "omim_gene": 601023, "omim_disease": 167320,
        "inheritance": (
            "AD (autosomal dominant). High penetrance for IBM (>90% by age 50). "
            "ALS component: ~10% of VCP kindreds develop ALS (lowest penetrance of the tetrad). "
            "Paget disease: ~40% penetrance. FTD: ~30% penetrance. "
            "Not all features manifest in every affected individual — variable expressivity within families. "
            "p.Arg155His (R155H): most common pathogenic variant (~50% of VCP families). "
            "p.Arg155Cys, p.Arg191Gln, p.Ala232Glu: other established pathogenic variants. "
            "Mutations cluster in the N-domain (ubiquitin-binding interface with UFD1–NPL4 adapter)."
        ),
        "phenotype": (
            "ONSET: IBM typically 30s–50s; ALS onset if develops usually 50s–60s. "
            "IBM: slowly progressive proximal + distal weakness (tibialis anterior → foot drop early); "
            "CK markedly elevated (5–10× normal); rimmed vacuoles on muscle biopsy; "
            "TDP-43 inclusions in muscle (IBM biopsy pathognomonic); steroid-resistant. "
            "PAGET: bone pain, fracture risk; alkaline phosphatase markedly elevated; "
            "bisphosphonate responsive. "
            "FTD: behavioural-variant FTD; executive dysfunction; loss of empathy. "
            "ALS: mixed UMN + LMN features; may coexist with IBM (ALS + IBM in same patient). "
            "VCP DIAGNOSTIC CLUE: IBM + paget + FTD family history → always check VCP. "
            "CK: markedly elevated in IBM component (1000–5000+ IU/L). "
            "NfL elevated if ALS/FTD component active."
        ),
        "disease": (
            "VCP-related Multisystem Proteinopathy (MSP/IBMPFD). Diagnosis: VCP gene panel + "
            "muscle biopsy (rimmed vacuoles + TDP-43 inclusions) + bone scan/ALP for Paget + "
            "neuropsychological assessment for FTD + EMG for ALS component. "
            "TREATMENT: IBM — no disease-modifying therapy (steroids INEFFECTIVE); physio. "
            "Paget — bisphosphonates (zoledronate). FTD — symptomatic (SSRI, behavioural management). "
            "ALS component — riluzole standard. "
            "Therapeutic target: VCP inhibitor (NMS-873), HSP70 induction, rapamycin (autophagy) — all trials."
        ),
        "treatment_options": [
            "IBM: physiotherapy (resistance training); NO steroids (waste time, cause harm — steroid-resistant)",
            "Paget disease: bisphosphonates (zoledronate 5 mg IV single dose most effective; monitor ALP)",
            "FTD: SSRI (sertraline/citalopram for apathy + disinhibition); memantine; behavioural management",
            "ALS component: riluzole 50 mg BD; edaravone if eligible",
            "Rapamycin (mTOR inhibitor → autophagy induction): clinical trial for VCP-IBM — UKMND-VCP trial",
            "VCP inhibitors (NMS-873, CB-5083): preclinical; reduce protein aggregation in cell/animal models",
            "MDT: neuromuscular specialist + neurologist (FTD/ALS) + metabolic bone specialist + orthopaedics",
            "Genetic counselling: AD; 50% familial risk; VCP tetrad — screen all first-degree relatives",
        ],
        "key_ddx": [
            "Sporadic IBM (s-IBM — much more common; no VCP mutations; no Paget; no FTD; older onset)",
            "Inflammatory myopathy (DM/PM — TDP-43 NOT a feature; responds to steroids — critical distinction)",
            "Becker muscular dystrophy (DMD mutations; no rimmed vacuoles; no TDP-43; dystrophin IHC negative)",
            "Hereditary IBM2 (IBMFD2) — GNE-myopathy (AR; no UMN signs; no Paget; no FTD)",
            "Oculopharyngeal muscular dystrophy (OPMD — PABPN1; ptosis + dysphagia; different biopsy)",
            "FTLD-TDP other causes (GRN, C9orf72 — check VCP when FTD + IBM in family)",
        ],
        "onset_range_y": (30, 70),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "ftd_risk": True,
        "bulbar_onset": False,
        "juvenile_onset": False,
        "xlinked": False,
        "very_slow_progression": True,  # IBM component slow
        "gene_therapy_available": False,
        "ck_range": (800, 8000),  # IBM component
        "survival_y_range": (2.0, 10.0),
    },
    # ── SETX — Senataxin ──────────────────────────────────────────────────
    {
        "gene": "SETX", "protein": "Senataxin (DNA-RNA Helicase)",
        "alias": "SETX; OMIM gene 608465; OMIM disease #205100 ALS4 juvenile; 9q34.13; 2677 aa; SLOW progression; allelic AOA2 (Ataxia Oculomotor Apraxia 2, AR)",
        "aa": "2677 aa", "kDa": "302 kDa",
        "gene_class": (
            "Senataxin (SETX) is a large superfamily 1 helicase (2677 aa, 302 kDa) that resolves "
            "RNA-DNA hybrid structures (R-loops), which form transiently during transcription. "
            "R-loop resolution is critical for transcription termination, DNA replication fidelity, "
            "and genome stability at sites of collision between RNA Pol II and replication forks. "
            "SETX also plays roles in DNA damage response (DDR), particularly at double-strand breaks "
            "associated with transcription, and in spermatogenesis (pachytene checkpoint). "
            "MECHANISM OF DISEASE (ALS4, AD): dominant-negative or GOF mutations → motor neuron-specific "
            "R-loop accumulation → transcription-replication conflicts → DNA damage → motor neuron death. "
            "ALLELIC DISEASE (AOA2, AR): biallelic LOF mutations → Ataxia Oculomotor Apraxia type 2 — "
            "COMPLETELY DIFFERENT PHENOTYPE: cerebellar ataxia + oculomotor apraxia + peripheral neuropathy "
            "+ elevated AFP (Alpha-Fetoprotein) — no ALS. "
            "CRITICAL DISTINCTION: SETX AD gain-of-function → ALS4 (motor neuron); "
            "SETX AR loss-of-function → AOA2 (cerebellar ataxia — entirely different disease)."
        ),
        "mnd_group": "DNA-RNA Helicase / R-Loop Resolution / Genomic Stability",
        "mnd_type": "ALS4 — Juvenile Autosomal Dominant ALS (slow progression, favorable prognosis)",
        "locus": "9q34.13", "omim_gene": 608465, "omim_disease": 205100,
        "inheritance": (
            "AD (dominant) for ALS4: gain-of-function or dominant-negative AD mutations. "
            "AR for AOA2: biallelic loss-of-function — completely different phenotype (cerebellar ataxia, not ALS). "
            "ALS4: RARE (~<1% fALS); important because JUVENILE onset + SLOW progression. "
            "Penetrance: high for established ALS4 variants. "
            "De novo mutations possible. "
            "Family history important — may span multiple generations with slowly progressive weakness."
        ),
        "phenotype": (
            "ONSET: ALS4 — JUVENILE; typically 10–25 years (range 5–30 years); "
            "earliest onset of any heritable ALS form (except neonatal FUS-AR). "
            "HALLMARK: SLOW PROGRESSION — patients walk and live for decades (unlike aggressive FUS); "
            "some ALS4 patients survive >30 years from onset. "
            "PHENOTYPE: UMN + LMN signs; distal > proximal weakness; lower limb > upper limb onset; "
            "Bilateral foot drop is a common early feature. "
            "PRESERVED: bulbar function often preserved for many years; "
            "respiratory function decline late and slow. "
            "COGNITIVE: normal cognition (no FTD in ALS4 — clear distinction from C9orf72). "
            "KEY TEACHING: ALS4 = best prognosis of all ALS forms — important to identify to avoid "
            "prognostic nihilism; patients need long-term support planning, not short-term palliative focus. "
            "NfL: elevated but less dramatically than aggressive ALS. "
            "CK: mildly elevated (denervation)."
        ),
        "disease": (
            "ALS4 — SETX-related Juvenile ALS. Diagnosis: gene panel (SETX sequencing) + EMG + MRI. "
            "Critical: distinguish AD ALS4 variants from AR AOA2 variants — entirely different prognosis/management. "
            "TREATMENT: Riluzole standard (evidence extrapolated; survival benefit modest). "
            "Long-term physiotherapy, orthotics (AFOs for foot drop). "
            "Multidisciplinary follow-up every 6–12 months given slow progression. "
            "Employment/education planning — most ALS4 patients remain employed/active for many years."
        ),
        "treatment_options": [
            "Riluzole 50 mg BD (standard ALS first-line; evidence from adult ALS trials extrapolated to ALS4)",
            "Long-term physiotherapy (strength maintenance; functional preservation; hydrotherapy)",
            "AFOs (ankle-foot orthoses) for bilateral foot drop — early fitting improves function + safety",
            "Annual FVC monitoring (respiratory decline slow but mandatory monitoring)",
            "Annual swallowing assessment (dysphagia late; early surveillance important)",
            "Employment/education planning: most ALS4 patients retain full independence for years–decades",
            "MDT: neuromuscular specialist long-term follow-up (NOT acute palliative team as in aggressive ALS)",
            "Genetic counselling: AD inheritance; 50% offspring risk; exclude AOA2 (AR SETX) in siblings",
        ],
        "key_ddx": [
            "Other juvenile ALS: FUS-ALS6 (MUCH more aggressive — survival <1 year vs years-decades for ALS4)",
            "Hereditary Spastic Paraplegia (HSP) — UMN predominant; no LMN; SPG4 most common",
            "SMA type 3/4 (Kugelberg-Welander — all LMN; SMN1 deletion; no UMN signs)",
            "AOA2 (AR SETX) — same gene but DIFFERENT phenotype: cerebellar ataxia + OMA + AFP elevated (NO ALS)",
            "Friedreich Ataxia (FRDA — AR; ataxia + cardiomyopathy; no UMN; FXN GAA repeat)",
            "Distal hereditary motor neuropathy (dHMN) — all LMN; no UMN; NCS: pure motor",
        ],
        "onset_range_y": (5, 30),
        "cardiac_risk": False,
        "respiratory_risk": False,  # slow, late
        "ftd_risk": False,
        "bulbar_onset": False,
        "juvenile_onset": True,
        "xlinked": False,
        "very_slow_progression": True,  # HALLMARK
        "gene_therapy_available": False,
        "ck_range": (80, 400),
        "survival_y_range": (15.0, 40.0),  # slow!
    },
    # ── NEK1 — NIMA-Related Kinase 1 ──────────────────────────────────────
    {
        "gene": "NEK1", "protein": "NIMA-Related Kinase 1 (NEK1)",
        "alias": "NEK1; OMIM gene 604588; OMIM disease #617435 ALS22; 4q33; 1258 aa; ~3% fALS; DNA damage repair + ciliogenesis + centrosome",
        "aa": "1258 aa", "kDa": "141 kDa",
        "gene_class": (
            "NEK1 (NIMA-Related Kinase 1) is a dual-specificity kinase involved in multiple cellular processes: "
            "(1) DNA damage response (DDR): phosphorylates and activates VDAC1 (mitochondrial), BRCA1, "
            "KIF1B; critical for proper G2/M checkpoint after DNA damage; "
            "(2) Ciliogenesis: required for primary cilia formation; NEK1 LOF = short/absent cilia; "
            "(3) Centrosome function: NEK1 localises to centrosome + kinetochore; "
            "(4) Mitochondrial homeostasis: NEK1 regulates VDAC1 phosphorylation status → "
            "mitochondrial permeability transition (MPT) regulation. "
            "MECHANISM OF ALS: haploinsufficiency (AD LOF); NEK1 heterozygous variants found in ~3% fALS "
            "(Project MinE consortium, 2016). Exact mechanism: DNA damage accumulation in motor neurons "
            "(long-lived post-mitotic cells with high transcriptional demand) → accelerated neurodegeneration. "
            "HETEROZYGOUS LOF: majority are truncating/frameshift variants — consistent with haploinsufficiency. "
            "NEK1 was identified as an ALS risk gene through large-scale exome sequencing consortia."
        ),
        "mnd_group": "DNA Damage Response / Kinase / Ciliogenesis",
        "mnd_type": "ALS22 — NEK1-ALS (familial; ~3% fALS; typical adult-onset ALS phenotype)",
        "locus": "4q33", "omim_gene": 604588, "omim_disease": 617435,
        "inheritance": (
            "AD (haploinsufficiency — most truncating/frameshift variants). "
            "~3% of familial ALS (Project MinE exome sequencing; n=1,000+ fALS). "
            "<1% of sporadic ALS. "
            "Penetrance: uncertain; likely incomplete (some variant carriers asymptomatic in families). "
            "No specific founder variants identified (unlike SOD1 A4V or C9orf72 repeat). "
            "Gene first identified as ALS risk through large ALS exome sequencing consortia 2016."
        ),
        "phenotype": (
            "ONSET: typically 40s–70s — adult onset similar to sporadic ALS. "
            "PHENOTYPE: clinically typical ALS; mixed UMN + LMN signs; limb or bulbar onset. "
            "DISTINGUISHING FEATURES: none identified — NEK1-ALS is clinically indistinguishable from "
            "sporadic ALS without genetic testing. "
            "PROGRESSION: moderate — similar to sporadic ALS (median survival 2–4 years). "
            "FTD: not prominently associated (unlike C9orf72). "
            "COGNITIVE: usually normal. "
            "CK: mildly elevated (denervation). "
            "RESEARCH IMPORTANCE: NEK1 identified through large genomic discovery efforts; "
            "validates DDR pathway as ALS pathomechanism alongside TDP-43/FUS RNA-binding pathways."
        ),
        "disease": (
            "ALS22 — NEK1-related ALS. Diagnosis: gene panel + EMG + MRI. "
            "No gene-specific therapy. Riluzole standard. "
            "NEK1 identified as risk gene (not rare Mendelian variant) → "
            "interpretation of individual variants requires careful evidence review. "
            "CAUTION: not all NEK1 variants are pathogenic — only truncating/frameshift/established "
            "missense variants in known functional domains should be reported as likely pathogenic."
        ),
        "treatment_options": [
            "Riluzole 50 mg BD (standard ALS first-line)",
            "Edaravone IV (selected patients meeting eligibility criteria)",
            "NIV when FVC <50%; PEG for dysphagia",
            "MDT standard ALS care",
            "DDR pathway targeting (pre-clinical: CDK5 inhibitors, PARP inhibitors — ALS models); no approved agents",
            "NfL monitoring (serum or CSF)",
            "Genetic counselling: AD; variant interpretation essential — pathogenic vs VUS distinction critical",
        ],
        "key_ddx": [
            "Sporadic ALS (clinically identical; NEK1 diagnosis requires genetic testing)",
            "SOD1-ALS (specific therapy available — tofersen; SOD1 must be checked)",
            "C9orf72-ALS (repeat-primed PCR mandatory; FTD features help distinguish)",
            "Multi-focal motor neuropathy (treatable — anti-GM1 Ab, IVIG)",
            "Inclusion body myositis (IBM) — if LMN only phenotype with high CK (VCP more likely)",
        ],
        "onset_range_y": (40, 72),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "ftd_risk": False,
        "bulbar_onset": True,
        "juvenile_onset": False,
        "xlinked": False,
        "very_slow_progression": False,
        "gene_therapy_available": False,
        "ck_range": (150, 600),
        "survival_y_range": (1.5, 4.0),
    },
    # ── UBQLN2 — Ubiquilin 2 ──────────────────────────────────────────────
    {
        "gene": "UBQLN2", "protein": "Ubiquilin 2",
        "alias": "UBQLN2; OMIM gene 300264; OMIM disease #300857 ALS15; Xp11.21; 624 aa; X-linked DOMINANT; ubiquitin-proteasome shuttle; ALS + dementia; rapid progression",
        "aa": "624 aa", "kDa": "66 kDa",
        "gene_class": (
            "Ubiquilin 2 (UBQLN2) is a ubiquitin-like modifier-activating enzyme scaffold protein that "
            "shuttles ubiquitinated proteins to the proteasome for degradation. Contains: "
            "N-terminal UBL (ubiquitin-like) domain + C-terminal UBA (ubiquitin-associated) domain + "
            "proline-rich PXX repeat domain (site of ALS15 mutations). "
            "FUNCTION: proteasomal protein quality control; stress granule regulation; "
            "aggresome clearance; parkin-mediated mitophagy. "
            "MECHANISM OF DISEASE: ALS15 mutations cluster in the PXX repeat domain (aa 450–505); "
            "dominant mutations → ubiquitin-proteasome pathway dysfunction → TDP-43 + FUS inclusions "
            "(UBQLN2-ALS has TDP-43 AND FUS pathology in some inclusions — unique overlap). "
            "UBQLN2 inclusions found in sporadic ALS brains (like TDP-43) → UBQLN2 dysfunction may "
            "be a COMMON DOWNSTREAM PATHWAY in ALS regardless of causative gene. "
            "X-LINKED DOMINANT: males fully affected; females ALSO affected (unlike X-linked recessive) — "
            "female carriers develop disease (incomplete penetrance but significant). "
            "IMPORTANT: NO SKIPPING IN MALES — X-linked dominant; hemizygous males always affected."
        ),
        "mnd_group": "Ubiquitin-Proteasome System (UPS) / X-Linked Dominant / Protein Quality Control",
        "mnd_type": "ALS15 — UBQLN2-ALS (X-linked dominant; ALS + dementia; rapid juvenile-onset possible)",
        "locus": "Xp11.21", "omim_gene": 300264, "omim_disease": 300857,
        "inheritance": (
            "X-LINKED DOMINANT (XLD): both males (hemizygous) and females (heterozygous) are affected. "
            "Males: typically more severe / earlier onset (hemizygous). "
            "Females: heterozygous — affected with variable expressivity; some have later/milder disease; "
            "does NOT follow X-linked recessive pattern (not carrier-only in females). "
            "De novo mutations documented (important in apparent sporadic early-onset). "
            "~1–2% of all ALS families. Mutations cluster in PXX domain (exon 2, aa 450-505). "
            "p.Pro497His (P497H): most common pathogenic variant; associated with ALS + dementia. "
            "p.Pro506Ser, p.Pro509Ser, p.Pro525Ser: other established pathogenic alleles."
        ),
        "phenotype": (
            "ONSET: wide range — juvenile (10–20 years in some males) to adult (50s–60s). "
            "HALLMARK: ALS + DEMENTIA coexistence — more common than other ALS genes. "
            "FTD-type dementia: executive dysfunction + behavioural change; in ~50% of families. "
            "PROGRESSION: typically FASTER than sporadic ALS; aggressive in many families. "
            "MALES: earlier onset + more aggressive. "
            "FEMALES: later onset + milder (but still affected — X-linked DOMINANT). "
            "PHENOTYPE VARIATION: some families = pure ALS; some = ALS + dementia; rare = pure dementia. "
            "KEY TEACHING: X-linked ALS in FEMALES is not carrier status — female carriers are PATIENTS. "
            "UMN + LMN mixed signs; bulbar involvement common. "
            "CK: mildly elevated (denervation). "
            "Inclusions: TDP-43 + ubiquilin-2-positive inclusions in neurons."
        ),
        "disease": (
            "ALS15 — UBQLN2-related ALS/ALS-dementia (X-linked dominant). "
            "Diagnosis: UBQLN2 gene sequencing (X-linked — sequence in both males and females separately). "
            "Treatment: riluzole standard; edaravone; NIV + PEG. "
            "Dementia management: SSRI + memantine (limited evidence). "
            "GENETIC COUNSELLING: X-linked DOMINANT — daughters of affected males: ALL at 50% risk "
            "(X from father); sons of affected males: UNAFFECTED (Y from father). "
            "Daughters of affected females: 50% risk. Sons of affected females: 50% risk. "
            "Pedigree is CRITICAL to interpret correctly."
        ),
        "treatment_options": [
            "Riluzole 50 mg BD (standard ALS first-line; LFT monitoring)",
            "Edaravone IV (selected patients; may be considered in rapidly progressive juvenile males)",
            "Early NIV planning (aggressive males may decline rapidly)",
            "PEG/RIG for dysphagia — early consideration given rapid progression",
            "Dementia management: SSRI (sertraline) for behavioural FTD; memantine; behavioural therapy",
            "UBQLN2 protein quality control pathway therapies (pre-clinical: proteasome enhancers, HSP induction)",
            "MDT: neurology + neuropsychology (FTD component) + respiratory + palliative",
            "Genetic counselling: X-LINKED DOMINANT (NOT recessive); affected females are patients, not carriers; "
            "full pedigree analysis essential; X-chromosome inheritance pattern must be explained clearly",
        ],
        "key_ddx": [
            "FUS-ALS6 (juvenile males — both can present young; FUS = TDP-43 negative; UBQLN2 = TDP-43 positive)",
            "C9orf72 ALS-FTD (FTD + ALS — C9orf72 is autosomal; check repeat first; more FTD than UBQLN2 on average)",
            "X-linked bulbo-spinal muscular atrophy / Kennedy disease (SBMA — XLR, NOT dominant; all LMN; "
            "sensory NCS abnormal; androgen receptor CAG >38)",
            "X-linked adrenoleukodystrophy (ALD — X-linked recessive; white matter + adrenal; VLCFA elevated)",
            "VCP-MSP (IBM + FTD + ALS — TDP-43 positive, autosomal dominant, NOT X-linked)",
            "Sporadic ALS-FTD (clinically similar; UBQLN2 test needed to differentiate)",
        ],
        "onset_range_y": (10, 65),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "ftd_risk": True,
        "bulbar_onset": True,
        "juvenile_onset": True,
        "xlinked": True,  # X-linked DOMINANT
        "very_slow_progression": False,
        "gene_therapy_available": False,
        "ck_range": (150, 700),
        "survival_y_range": (1.0, 4.0),
    },
]


def _gen_patients(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    patients = []
    ck_lo, ck_hi = gene_data["ck_range"]
    surv_lo, surv_hi = gene_data["survival_y_range"]
    onset_lo, onset_hi = gene_data["onset_range_y"]

    for i in range(40):
        onset = round(rng.uniform(onset_lo, onset_hi), 1)
        ck_val = round(rng.uniform(ck_lo, ck_hi))

        # Severity based on gene characteristics
        r = rng.random()
        if gene_data["very_slow_progression"]:
            sev = "Mild" if r < 0.55 else ("Moderate" if r < 0.85 else "Severe")
        elif gene in ("FUS", "UBQLN2"):
            sev = "Severe" if r < 0.55 else ("Moderate" if r < 0.85 else "Mild")
        else:
            sev = "Moderate" if r < 0.50 else ("Mild" if r < 0.75 else "Severe")

        resp = rng.random() < (0.25 if gene_data["very_slow_progression"] else 0.65)
        ftd = rng.random() < (0.50 if gene_data["ftd_risk"] else 0.08)
        bulbar = rng.random() < (0.35 if gene_data["bulbar_onset"] else 0.15)
        pba = rng.random() < (0.40 if gene == "C9orf72" else (0.20 if bulbar else 0.10))
        cognitive = ftd or rng.random() < 0.15
        gene_therapy_offered = gene_data["gene_therapy_available"] and rng.random() < 0.60
        riluzole = rng.random() < 0.92  # almost universal
        niv = resp and rng.random() < 0.80
        peg = bulbar and rng.random() < 0.65
        survival_y = round(rng.uniform(surv_lo, surv_hi), 1)

        # Treatment
        if gene == "SOD1":
            tx = "Tofersen IT + riluzole" if gene_therapy_offered else "Riluzole + edaravone; NIV as needed"
        elif gene == "C9orf72":
            tx = "Riluzole + PBA management (Nuedexta); FTD MDT" if ftd or pba else "Riluzole; MDT; NIV planning"
        elif gene == "SETX":
            tx = "Riluzole long-term; AFOs; physiotherapy; MDT annual review"
        elif gene == "VCP":
            tx = "Riluzole (ALS); physio (IBM); bisphosphonate (Paget); SSRI (FTD)"
        elif gene == "FUS":
            tx = "Riluzole + aggressive NIV planning (rapid juvenile course)"
        elif gene == "UBQLN2":
            tx = "Riluzole + early NIV; dementia management SSRI" if ftd else "Riluzole + NIV planning"
        elif gene == "NEK1":
            tx = "Riluzole; standard ALS MDT"
        else:
            tx = "Riluzole; edaravone if eligible; NIV + PEG as needed"

        pid = f"MND-{gene}-{seed}-{i+1:03d}"
        # X-linked dominant UBQLN2: both sexes affected; males tend to be younger
        if gene == "UBQLN2":
            sex = "M" if (i < 24) else "F"  # ~60% male (hemizygous = always affected; het females variable)
        else:
            sex = rng.choice(["M", "F"])

        patients.append({
            "id": pid, "gene": gene, "sex": sex,
            "onset_age_y": onset, "severity": sev,
            "respiratory_decline": resp,
            "ftd_features": ftd,
            "pba": pba,
            "bulbar_onset": bulbar,
            "cognitive_impairment": cognitive,
            "gene_therapy_offered": gene_therapy_offered,
            "riluzole": riluzole,
            "niv": niv,
            "peg": peg,
            "survival_y_projected": survival_y,
            "juvenile_onset": onset < 30,
            "ck_iu_l": ck_val,
            "current_treatment": tx,
            "inheritance": gene_data["inheritance"].split(".")[0],
            "xlinked": gene_data["xlinked"],
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gene_data in enumerate(MND_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients(gene_data, seed))
    return all_pts


def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)
    gene_counts = {}
    for p in patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] += 1

    resp_n   = sum(1 for p in patients if p["respiratory_decline"])
    ftd_n    = sum(1 for p in patients if p["ftd_features"])
    bulbar_n = sum(1 for p in patients if p["bulbar_onset"])
    pba_n    = sum(1 for p in patients if p["pba"])
    juv_n    = sum(1 for p in patients if p["juvenile_onset"])
    niv_n    = sum(1 for p in patients if p["niv"])
    peg_n    = sum(1 for p in patients if p["peg"])
    gt_n     = sum(1 for p in patients if p["gene_therapy_offered"])
    cognitive_n = sum(1 for p in patients if p["cognitive_impairment"])
    xlinked_n   = sum(1 for p in patients if p["xlinked"])

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 1)
    mean_ck = round(sum(p["ck_iu_l"] for p in patients) / n)
    mean_surv = round(sum(p["survival_y_projected"] for p in patients) / n, 1)

    return {
        "atlas": "MND-Atlas",
        "full_name": "Complete 8-Gene Motor Neuron Disease / ALS Atlas",
        "subtitle": "SOD1·C9orf72·TARDBP·FUS·VCP·SETX·NEK1·UBQLN2 — 320 patients (8×40, seeds 1046–1053)",
        "description": (
            "Comprehensive atlas of the 8 most clinically and scientifically important Motor Neuron Disease "
            "(MND/ALS) genes. Covers: SOD1 (tofersen FDA-approved 2023 — FIRST gene-specific ALS therapy); "
            "C9orf72 (most common familial ALS; GGGGCC repeat expansion; ALS/FTD overlap); "
            "TARDBP (TDP-43 — universal ALS pathological marker); FUS (juvenile ALS, most aggressive); "
            "VCP (Multisystem Proteinopathy: IBM + FTD + ALS + Paget); SETX (juvenile ALS4, best prognosis); "
            "NEK1 (~3% fALS, DNA damage repair); UBQLN2 (X-linked dominant ALS + dementia). "
            "CRITICAL DISTINCTIONS: Only SOD1 has approved gene-specific therapy (tofersen IT); "
            "C9orf72 requires repeat-primed PCR (not standard NGS); TDP-43 pathology in 97% ALL ALS "
            "(except SOD1 and FUS which have distinct inclusions); FUS and SETX are the two main "
            "causes of juvenile ALS (FUS = aggressive; SETX = slow); UBQLN2 is X-linked DOMINANT "
            "(females are patients, not carriers). Riluzole is standard first-line for ALL ALS."
        ),
        "total_patients": n,
        "genes_covered": len(MND_GENES),
        "patients_per_gene": 40,
        "seed_range": "1046–1053",
        "gene_list": [g["gene"] for g in MND_GENES],
        "mnd_category_breakdown": {
            "Oxidative Stress / SOD1 GOF (Gene Therapy Available)": ["SOD1"],
            "RNA Toxicity / Repeat Expansion / ALS-FTD (Most Common fALS)": ["C9orf72"],
            "RNA-Binding Protein / Universal TDP-43 Marker": ["TARDBP"],
            "RNA-Binding Protein / FUS Mis-localisation (Juvenile ALS)": ["FUS"],
            "Ubiquitin-Proteasome / Multisystem Proteinopathy (IBM+FTD+ALS+Paget)": ["VCP"],
            "DNA-RNA Helicase / Juvenile ALS (Slow — Best Prognosis)": ["SETX"],
            "DNA Damage Response / Kinase (~3% fALS)": ["NEK1"],
            "UPS Shuttle / X-Linked Dominant / ALS + Dementia": ["UBQLN2"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "mean_ck_iu_l": mean_ck,
        "mean_projected_survival_y": mean_surv,
        "clinical_features_prevalence": {
            "respiratory_decline_pct": round(100 * resp_n / n, 1),
            "ftd_features_pct":        round(100 * ftd_n / n, 1),
            "bulbar_onset_pct":        round(100 * bulbar_n / n, 1),
            "pseudobulbar_affect_pct": round(100 * pba_n / n, 1),
            "juvenile_onset_pct":      round(100 * juv_n / n, 1),
            "niv_required_pct":        round(100 * niv_n / n, 1),
            "peg_required_pct":        round(100 * peg_n / n, 1),
            "gene_therapy_offered_pct": round(100 * gt_n / n, 1),
            "cognitive_impairment_pct": round(100 * cognitive_n / n, 1),
            "xlinked_dominant_pct":    round(100 * xlinked_n / n, 1),
        },
        "key_teaching_points": [
            "SOD1: TOFERSEN (QALSODY) — FIRST FDA-approved gene-specific ALS treatment (April 2023); "
            "intrathecal ASO; reduces SOD1 protein + NfL; presymptomatic SOD1 carriers now monitored "
            "for early treatment (ATLAS presymptomatic trial)",
            "C9orf72: REPEAT-PRIMED PCR MANDATORY — standard NGS/Sanger MISSES the GGGGCC repeat; "
            "check ALL ALS/FTD patients; >30 repeats = pathogenic; FTD in ~50%; PBA managed with Nuedexta",
            "TDP-43 (TARDBP): pathological TDP-43 inclusions in 97% ALL ALS regardless of gene — "
            "except SOD1 (SOD1 aggregates) and FUS (FUS inclusions, TDP-43 NEGATIVE); "
            "TDP-43 is THE universal ALS pathological biomarker",
            "FUS: MOST COMMON CAUSE OF JUVENILE ALS (<25 years) — aggressive; survival <12–18 months "
            "in NLS mutations (p.P525L); DISTINCT from SETX (also juvenile ALS but slow progression); "
            "FUS inclusions are TDP-43 NEGATIVE — pathologically distinct",
            "VCP: TETRAD = IBM + Paget + FTD + ALS — IBM most penetrant (90%); "
            "IBM biopsy shows rimmed vacuoles + TDP-43 inclusions; STEROID-RESISTANT (steroids USELESS, HARMFUL); "
            "Paget: bisphosphonate (zoledronate); VCP inhibitors in trials",
            "SETX (ALS4): JUVENILE ALS with BEST PROGNOSIS — onset 5–25 years; patients survive DECADES; "
            "critical to distinguish from FUS (aggressive juvenile ALS) to avoid prognostic nihilism; "
            "allelic AOA2 (AR LOF SETX) = cerebellar ataxia, NOT ALS — entirely different phenotype",
            "UBQLN2: X-LINKED DOMINANT — affected FEMALES are PATIENTS (NOT just carriers); "
            "hemizygous males + heterozygous females both develop ALS±dementia; ALS15 dementia common; "
            "X-linked pedigree must be interpreted as XLD not XLR",
            "RILUZOLE: first-line ALL ALS (modest ~3 months survival benefit); monitor LFTs at 3–6 months; "
            "do NOT withhold based on severity — universal first-line therapy",
        ],
        "drug_alerts": [
            "SOD1: TOFERSEN — first approved gene-specific ALS therapy; intrathecal; confirm SOD1 pathogenic "
            "variant before prescribing; dramatically reduces NfL (surrogate neurodegeneration marker)",
            "C9orf72: PBA — dextromethorphan/quinidine (Nuedexta) for pseudobulbar affect (crying/laughing "
            "uncontrolled); do NOT confuse with depression (different treatment)",
            "VCP-IBM: STEROIDS ABSOLUTELY INEFFECTIVE AND HARMFUL — IBM is NOT inflammatory myositis; "
            "steroids accelerate weakness + cause side effects; confirm molecular diagnosis before any "
            "immunosuppression",
            "ALL ALS: RILUZOLE first-line (prolongs survival ~3 months; mechanism: glutamate release inhibition); "
            "monitor LFTs monthly × 3 months then q3 months; stop if ALT/AST >5× ULN",
            "FUS JUVENILE: EARLY ADVANCE CARE PLANNING — p.P525L survival <12 months; NIV + PEG "
            "discussions must begin at diagnosis, not at crisis",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    gene_profiles = []
    for gene_data in MND_GENES:
        gene_pts = [p for p in patients if p["gene"] == gene_data["gene"]]
        n = len(gene_pts)
        sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for p in gene_pts:
            sev[p["severity"]] += 1
        mean_ck_g = round(sum(p["ck_iu_l"] for p in gene_pts) / n)
        mean_surv_g = round(sum(p["survival_y_projected"] for p in gene_pts) / n, 1)
        gene_profiles.append({
            "gene": gene_data["gene"],
            "protein": gene_data["protein"],
            "alias": gene_data["alias"],
            "locus": gene_data["locus"],
            "omim_gene": gene_data["omim_gene"],
            "omim_disease": gene_data["omim_disease"],
            "inheritance": gene_data["inheritance"],
            "mnd_group": gene_data["mnd_group"],
            "mnd_type": gene_data["mnd_type"],
            "aa": gene_data["aa"],
            "kDa": gene_data["kDa"],
            "gene_class": gene_data["gene_class"],
            "phenotype": gene_data["phenotype"],
            "disease": gene_data["disease"],
            "treatment_options": gene_data["treatment_options"],
            "key_ddx": gene_data["key_ddx"],
            "onset_range_y": list(gene_data["onset_range_y"]),
            "n_patients": n,
            "cardiac_risk": gene_data["cardiac_risk"],
            "respiratory_risk": gene_data["respiratory_risk"],
            "ftd_risk": gene_data["ftd_risk"],
            "bulbar_onset": gene_data["bulbar_onset"],
            "juvenile_onset": gene_data["juvenile_onset"],
            "xlinked": gene_data["xlinked"],
            "very_slow_progression": gene_data["very_slow_progression"],
            "gene_therapy_available": gene_data["gene_therapy_available"],
            "mean_ck_iu_l": mean_ck_g,
            "mean_projected_survival_y": mean_surv_g,
            "severity_distribution": {
                "mild_pct": round(100 * sev["Mild"] / n, 1),
                "moderate_pct": round(100 * sev["Moderate"] / n, 1),
                "severe_pct": round(100 * sev["Severe"] / n, 1),
            },
            "clinical_features": {
                "respiratory_decline_pct": round(100 * sum(1 for p in gene_pts if p["respiratory_decline"]) / n, 1),
                "ftd_features_pct":        round(100 * sum(1 for p in gene_pts if p["ftd_features"]) / n, 1),
                "bulbar_onset_pct":        round(100 * sum(1 for p in gene_pts if p["bulbar_onset"]) / n, 1),
                "pba_pct":                 round(100 * sum(1 for p in gene_pts if p["pba"]) / n, 1),
                "juvenile_onset_pct":      round(100 * sum(1 for p in gene_pts if p["juvenile_onset"]) / n, 1),
                "niv_pct":                 round(100 * sum(1 for p in gene_pts if p["niv"]) / n, 1),
                "peg_pct":                 round(100 * sum(1 for p in gene_pts if p["peg"]) / n, 1),
                "gene_therapy_offered_pct": round(100 * sum(1 for p in gene_pts if p["gene_therapy_offered"]) / n, 1),
            },
            "sample_patients": gene_pts[:3],
        })
    return {
        "atlas": "MND-Atlas",
        "genes": gene_profiles,
        "total_patients": len(patients),
    }


def get_definitions() -> dict:
    return {
        "atlas": "MND-Atlas",
        "definitions": [
            {
                "term": "Amyotrophic Lateral Sclerosis (ALS) — El Escorial Criteria",
                "definition": (
                    "ALS is a progressive neurodegenerative disease affecting both upper motor neurons (UMN: "
                    "corticospinal tract, Betz cells) and lower motor neurons (LMN: anterior horn cells, "
                    "brainstem nuclei). El Escorial diagnostic criteria require UMN + LMN signs in at least "
                    "one region (El Escorial Definite = ≥3 regions). Revised ALS criteria (Gold Coast 2020) "
                    "require: (1) progressive motor dysfunction; (2) UMN + LMN signs in same or different "
                    "body region; (3) exclusion of ALS mimics. EMG is essential: "
                    "fibrillations + PSW (positive sharp waves) in ≥3 regions support LMN diagnosis. "
                    "MRI brain/spine excludes structural mimics (cervical myelopathy, MS, brain tumour). "
                    "CSF: typically normal (used to exclude infection/inflammatory). "
                    "Median survival: 2–5 years from symptom onset; respiratory failure = terminal event in most."
                ),
            },
            {
                "term": "Tofersen (QALSODY) — SOD1-Specific ASO",
                "definition": (
                    "Tofersen is an intrathecal antisense oligonucleotide (ASO) that targets SOD1 mRNA via "
                    "RNase H-mediated cleavage → reduces SOD1 protein production in the CSF/spinal cord. "
                    "FDA approved April 2023 (Accelerated Approval) for adults with SOD1-variant ALS — "
                    "FIRST gene-specific ALS treatment EVER approved. "
                    "DOSE: 100 mg intrathecal; loading doses at weeks 1, 3, 5; maintenance every 28 days. "
                    "BIOMARKER: significantly reduces NfL (neurofilament light chain) — surrogate marker "
                    "of neurodegeneration. Clinical benefit: slowed ALSFRS-R decline in responders; "
                    "ATLAS trial: testing presymptomatic SOD1 carriers. "
                    "SIDE EFFECTS: myelitis, aseptic meningitis, lumbar puncture complications. "
                    "ELIGIBILITY: SOD1 pathogenic variant confirmed + ALS diagnosis; NOT for other ALS genes."
                ),
            },
            {
                "term": "TDP-43 Pathology — Universal ALS Biomarker",
                "definition": (
                    "TDP-43 (TARDBP gene product) is a nuclear RNA-binding protein that forms pathological "
                    "cytoplasmic inclusions in the motor neurons of ~97% of ALL ALS patients, regardless of "
                    "the causative gene. TDP-43 pathological hallmarks: nuclear clearance + cytoplasmic "
                    "aggregates that are phosphorylated (pTDP-43), ubiquitinated, and truncated (25 kDa "
                    "C-terminal fragment). "
                    "EXCEPTIONS where TDP-43 pathology is ABSENT: (1) SOD1-ALS: SOD1 aggregates instead; "
                    "(2) FUS-ALS: FUS cytoplasmic inclusions instead (FUS-positive, TDP-43-negative). "
                    "TDP-43 pathology also present in ~50% of FTD (FTLD-TDP subtypes A, B, C). "
                    "CLINICAL APPLICATION: phospho-TDP-43 immunostaining (pTDP-43) on post-mortem or "
                    "biopsy tissue is the gold-standard neuropathological confirmation of ALS. "
                    "NB: detecting TDP-43 pathology does NOT identify the causative gene — separate genetic testing needed."
                ),
            },
            {
                "term": "C9orf72 Hexanucleotide Repeat Expansion (HRE) — Diagnostic Pitfall",
                "definition": (
                    "The C9orf72 GGGGCC (G4C2) repeat expansion is the most common genetic cause of both "
                    "familial ALS and familial FTD. Normal: <10 repeats. Pathogenic: >30 repeats "
                    "(most disease-causing expansions have >100 repeats; some >1000). "
                    "DIAGNOSTIC PITFALL: Standard short-read NGS (whole exome, gene panels) and Sanger "
                    "sequencing CANNOT detect this expansion — the repeat is too large and GC-rich. "
                    "MANDATORY TESTING: Repeat-primed PCR (RP-PCR) → confirms expansion PRESENT. "
                    "Southern blot → accurately SIZES the expansion. "
                    "Fragment analysis → confirms but does NOT size. "
                    "CLINICAL IMPLICATION: Any ALS or FTD patient with negative gene panel but family history "
                    "of ALS/FTD/dementia → C9orf72 repeat-primed PCR is mandatory. "
                    "A normal comprehensive gene panel does NOT exclude C9orf72 expansion."
                ),
            },
            {
                "term": "Riluzole — First-Line ALS Treatment",
                "definition": (
                    "Riluzole (Rilutek, Tiglutik) is the only oral ALS disease-modifying treatment with "
                    "established survival benefit in randomised trials. DOSE: 50 mg BD (100 mg/day). "
                    "MECHANISM: inhibits presynaptic glutamate release → reduces glutamate excitotoxicity "
                    "(a key pathomechanism in ALS — motor neurons are hypersensitive to glutamate-mediated "
                    "calcium influx and oxidative damage). "
                    "SURVIVAL BENEFIT: modest — approximately 3 months extension of median survival. "
                    "Meta-analysis: HR for survival ~0.84 (16% reduction in mortality rate). "
                    "MONITORING: LFTs monthly × 3 months, then every 3 months for 1 year; "
                    "stop if ALT/AST >5× ULN. Neutropenia: rare but check FBC if infection. "
                    "INDICATION: ALL ALS patients regardless of gene, stage, or severity — universal first-line. "
                    "LIQUID FORMULATION: Tiglutik (oral suspension) for patients with dysphagia."
                ),
            },
            {
                "term": "FUS vs TDP-43 — Mutually Exclusive Inclusions",
                "definition": (
                    "FUS (Fused in Sarcoma) and TDP-43 (TARDBP gene) are both hnRNP nuclear RNA-binding "
                    "proteins with prion-like C-terminal domains, and both cause ALS when mutated. "
                    "However, their pathological inclusions are MUTUALLY EXCLUSIVE: "
                    "(1) FUS-ALS: cytoplasmic FUS-positive inclusions; TDP-43 remains nuclear (NEGATIVE); "
                    "(2) All other ALS (SOD1 excluded): TDP-43 pathological inclusions; FUS remains nuclear. "
                    "This mutual exclusivity suggests FUS and TDP-43 compete for the same pathological "
                    "stress granule incorporation process. "
                    "CLINICAL RELEVANCE: if post-mortem/biopsy shows FUS-positive, TDP-43-negative "
                    "inclusions → FUS-ALS (ALS6); if TDP-43-positive → virtually all other ALS types. "
                    "Biomarker research: CSF FUS levels elevated in FUS-ALS; may enable ante-mortem "
                    "distinction from TDP-43 proteinopathies."
                ),
            },
            {
                "term": "VCP Multisystem Proteinopathy (IBMPFD) — IBM vs Inflammatory Myositis",
                "definition": (
                    "VCP-related inclusion body myopathy (IBM) is FUNDAMENTALLY DIFFERENT from "
                    "sporadic inclusion body myositis (s-IBM) and inflammatory myopathies (DM/PM): "
                    "(1) VCP-IBM: HEREDITARY (AD VCP mutation); younger onset; part of VCP tetrad; "
                    "biopsy: rimmed vacuoles + TDP-43 inclusions + ubiquitin inclusions; NO inflammation. "
                    "(2) s-IBM: sporadic; older onset; inflammatory infiltrates (CD8+ endomysial); "
                    "rimmed vacuoles + TDP-43 inclusions (pathologically similar to VCP-IBM); "
                    "associated with anti-NT5C1A antibody. "
                    "CRITICAL DISTINCTION — STEROIDS: "
                    "VCP-IBM: steroid-RESISTANT — DO NOT TREAT WITH STEROIDS (causes side effects without benefit). "
                    "Inflammatory myositis (DM/PM): steroid-RESPONSIVE (first-line treatment). "
                    "BEFORE STARTING STEROIDS FOR 'MYOSITIS': check for VCP mutations if rimmed vacuoles "
                    "on biopsy + family history of Paget/FTD/ALS — molecular diagnosis FIRST."
                ),
            },
            {
                "term": "SETX ALS4 vs FUS ALS6 — Juvenile ALS Differential",
                "definition": (
                    "Both ALS4 (SETX) and ALS6 (FUS) cause juvenile ALS. They are clinically distinct: "
                    "ALS4/SETX: "
                    "  - Onset 5–25 years; typically teen years. "
                    "  - SLOW progression — decades of survival; most patients remain active for 15–30+ years. "
                    "  - UMN + LMN signs; distal > proximal; foot drop early. "
                    "  - No FTD. Normal cognition. "
                    "  - Respiratory function preserved for many years. "
                    "  - Prognosis: best of all ALS subtypes. "
                    "  - ALLELIC: AR SETX → AOA2 (ataxia, OMA, AFP elevated) — NOT ALS. "
                    "ALS6/FUS: "
                    "  - Onset 12–35 years (NLS mutations); most aggressive in p.P525L. "
                    "  - RAPID progression — survival <12–24 months from symptom onset in aggressive variants. "
                    "  - FUS cytoplasmic inclusions; TDP-43 NEGATIVE. "
                    "  - NIV + PEG must be planned at DIAGNOSIS given speed of progression. "
                    "  - De novo mutations common (p.P525L often de novo). "
                    "CLINICAL RULE: juvenile ALS + SLOW progression = think SETX; "
                    "juvenile ALS + RAPID progression = think FUS."
                ),
            },
            {
                "term": "Neurofilament Light Chain (NfL) — ALS Biomarker",
                "definition": (
                    "Neurofilament light chain (NfL) is a structural protein of neuronal axons, released "
                    "into CSF and blood upon neuronal injury/death. In ALS: "
                    "CSF NfL: markedly elevated (often >4000 pg/mL vs normal <400 pg/mL); "
                    "Serum/plasma NfL: elevated (~3–8× normal); correlates with CSF NfL. "
                    "CLINICAL USES: (1) DIAGNOSIS: distinguishes ALS from mimic disorders; "
                    "(2) PROGNOSIS: baseline NfL inversely correlates with survival; "
                    "(3) TREATMENT MONITORING: tofersen in SOD1-ALS → dramatically reduces NfL within "
                    "weeks (>50% reduction used as surrogate efficacy endpoint; FDA accelerated approval basis). "
                    "LIMITATIONS: NfL elevated in many neurological diseases (MS, traumatic brain injury, "
                    "dementia) — NOT ALS-specific; diagnostic utility is in conjunction with clinical context. "
                    "TESTING: Lumipulse or Simoa platforms; serum NfL increasingly replacing CSF NfL in trials."
                ),
            },
        ],
    }
