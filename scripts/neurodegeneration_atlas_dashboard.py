#!/usr/bin/env python3
"""Hereditary Neurodegeneration Atlas — Complete 8-Gene Adult-Onset Neurodegenerative Disease Atlas
PSEN1   (Familial Alzheimer Disease 3 — 467 aa; 14q24.2; presenilin-1 γ-secretase catalytic;
         AD dominant; most common FAD gene; >300 pathogenic variants; onset 30–60s;
         lecanemab/donanemab eligibility: early AD regardless of gene; NEVER start early
         before comprehensive genetics counselling — positive test has 100% penetrance) ·
PSEN2   (Familial Alzheimer Disease 4 — 448 aa; 1q42.13; presenilin-2 γ-secretase;
         AD dominant; LOWER penetrance ~95%; later onset 50–70s; Volga German founder
         Asn141Ile; milder phenotype than PSEN1) ·
APP     (FAD2 / Cerebral Amyloid Angiopathy — 770 aa; 21q21.3; amyloid precursor protein;
         AD dominant; duplications → onset 50s; CAA: Ala692Gly (Dutch) + Arg693Gln (Italian)
         → lobar ICH — NEVER anticoagulate; trisomy 21 mechanism explains Down AD) ·
MAPT    (FTD-Parkinsonism Linked to Chr17 FTDP-17 — 758 aa; 17q21.31; tau microtubule;
         AD dominant; bvFTD, CBS, PSP-like; 17q21 H1/H2 haplotype — H1 PSP/CBD risk;
         Ile10Thr splice shifts 4R/3R; no approved DMT; avoid antipsychotics for agitation) ·
GRN     (FTLD-TDP type A / FTD-GRN — 593 aa; 17q21.32; progranulin growth factor;
         AD haploinsufficiency; plasma PGRN < 130 ng/mL diagnostic; lysosomal dysfunction;
         latozinemab Phase III; NCE-progranulin AL001 targets sortilin) ·
C9orf72 (ALS/FTD spectrum — hexanucleotide G4C2 repeat; 9p21.2; AD incomplete penetrance;
         >30 repeats pathogenic; >200 "full expansion" (ALS/FTD); <30 = normal;
         DPR (poly-GA/GR/PR/PA/GP) + RNA foci co-pathology; tofersen class ASO trials) ·
LRRK2   (Autosomal Dominant Parkinson Disease — 2527 aa; 12q12; LRRK2 leucine-rich repeat
         kinase; p.Gly2019Ser most common (~1–2% sporadic PD, 15–40% Ashkenazi Jewish PD,
         30–40% N.African Arabic PD); ROC-kinase domain GOF; DNL151/BIIB122 Phase II/III;
         > 5 years prodromal hyposmia + constipation + RBD before tremor) ·
SNCA    (PD / DLB / MSA spectrum — 140 aa; 4q22.1; α-synuclein presynaptic chaperone;
         AD point mutations Ala53Thr/Ala30Pro/Glu46Lys; CNV: duplication → mid-severity,
         triplication → rapid severe onset ≤40 yr; RBD 80% prodromal; Lewy body = PATHOGNOMONIC
         eosinophilic inclusion; prasinezumab Phase II anti-SNCA immunotherapy)
320-patient aggregate cohort (8 × 40, seeds 1238–1245)
"""

import random

SEED_BASE = 1238

NEURODEGEN_GENES = [
    # ── PSEN1 — Familial Alzheimer Disease 3 ──────────────────────────────────
    {
        "gene": "PSEN1",
        "protein": "Presenilin-1 (γ-Secretase Catalytic Subunit)",
        "alias": (
            "PSEN1; OMIM gene 104311; Alzheimer Disease-3 (FAD3) #607822; 14q24.2; 467 aa; ~52 kDa; "
            "AD dominant near-100% penetrance; most common FAD gene (>300 pathogenic variants); "
            "γ-secretase complex: PSEN1 (catalytic) + nicastrin + APH1 + PEN2; "
            "cleaves APP at γ-site generating Aβ peptides; GOF shifts ratio Aβ42/Aβ40 → ↑Aβ42 aggregation; "
            "also cleaves NOTCH ICD — explaining embryonic lethality of homozygous knockout; "
            "PSEN1 also regulates lysosomal pH, calcium homeostasis, autophagy"
        ),
        "aa": "467 aa",
        "kDa": "~52 kDa",
        "locus": "14q24.2",
        "omim_gene": 104311,
        "omim_disease": 607822,
        "inheritance": "AD dominant (penetrance ~100%)",
        "gene_class": (
            "9-pass transmembrane protein; forms catalytic core of γ-secretase; "
            "Asp257 + Asp385 = catalytic aspartyl protease dyad (in TM6 + TM7); "
            "PSEN1 pathogenic variants → impaired γ-secretase processivity → ↑Aβ42:Aβ40 ratio; "
            "Aβ42 is more aggregation-prone, seeds amyloid plaques earlier than Aβ40; "
            "mutation classes: missense (loss of processivity), splice (exon skipping), early stop (NMD); "
            "PSEN1 variant database: >300 pathogenic, curated at ADMutations.org and ClinVar; "
            "hotspot exons: 4 (residues 100–120), 5 (135–165), 8 (250–290), 12 (383–416 near catalytic Asp385); "
            "Leu170Phe: classic European founder; Met146Leu: Colombian kindred (earliest onset 45 yr); "
            "PSEN1 is NOT a risk gene — it is a fully penetrant, dominant causal gene; "
            "heterozygous pathogenic variant = Alzheimer disease develops with near certainty"
        ),
        "phenotype": (
            "Onset: mean 45 years (range 28–65 depending on variant); "
            "Core syndrome: rapidly progressive amnestic dementia; episodic memory loss earliest; "
            "language, executive, visuospatial deficits follow; "
            "Atypical features unique to PSEN1: spastic paraparesis (18–20%), paraparesis before/during dementia; "
            "cerebellar ataxia (10%); seizures (30%) — earlier than sporadic AD; "
            "Cotton wool plaques: neuropathological hallmark of PSEN1 deletions + early stop; "
            "Biomarkers: CSF Aβ42 ↓↓, p-tau ↑↑, t-tau ↑↑; Amyloid PET positive 100% in symptomatic; "
            "can be positive 15–20 years before symptoms; plasma pTau181/pTau217 ↑ before symptoms"
        ),
        "hallmark": (
            "EARLY ONSET (before 65) + FAMILY HISTORY of AD → PSEN1 first in panel; "
            "SPASTIC PARAPARESIS in a young AD patient — PATHOGNOMONIC for some PSEN1 deletions/early-stop; "
            "COTTON WOOL PLAQUES: pale diffuse Aβ plaques without neuritic halo — specific to PSEN1 null-type; "
            "SEIZURES in young AD patient: PSEN1 > PSEN2 > APP in frequency; "
            "GENETICS COUNSELLING MANDATORY before testing: positive result = near-certain disease; "
            "avoid genetic testing in minors unless clinically indicated; "
            "HOMOLOGOUS gene PSEN2 has lower penetrance — distinguish before counselling"
        ),
        "treatment_alert": (
            "LECANEMAB (LEQEMBI): FDA 2023 approved for early AD (MCI/mild AD); "
            "eligibility: amyloid-confirmed early-stage regardless of PSEN1 genotype; "
            "ARIA monitoring: MRI at baseline, 5th, 7th infusions; APOE ε4 status affects ARIA risk; "
            "ARIA-E (oedema) + ARIA-H (microhaemorrhage/siderosis): withhold if symptomatic; "
            "Severe ARIA: stop lecanemab; anticoagulants + lecanemab = HIGH ARIA-H risk — review benefit/risk; "
            "DONANEMAB: Phase III TRAILBLAZER-ALZ2 — similar ARIA profile; "
            "SEIZURES: levetiracetam first line (PSEN1 seizures); avoid carbamazepine (CYP3A4 inducer); "
            "SPASTIC PARAPARESIS: baclofen, physiotherapy, fall prevention; "
            "GENETIC COUNSELLING: pre-test + post-test MANDATORY; first-degree relatives 50% risk; "
            "presymptomatic testing protocol (GINA-like protections vary by jurisdiction — Canada/EU protections); "
            "DIAN trial: Dominantly Inherited Alzheimer Network — eligible patients should be enrolled"
        ),
        "key_ddx": (
            "PSEN2-FAD4: same phenotype, lower penetrance, later onset (50–70); Asn141Ile Volga German founder; "
            "APP-FAD2: overlapping; APP duplication → younger onset; Dutch/Italian APP → CAA/lobar ICH; "
            "Sporadic AD (APOE ε4/ε4): onset 60s–70s; no PSEN1 mutation; APOE is risk not deterministic; "
            "DLB: Parkinsonism + fluctuating cognition + visual hallucinations + RBD before dementia; "
            "FTD (MAPT/GRN/C9): behavioural > memory; younger onset; frontal atrophy not parieto-temporal; "
            "Prion (PRNP): rapid progression <1 year; periodic EEG complexes; DWI cortical ribboning"
        ),
        "onset_pattern": "Mean 45 years (range 28–65); 15–20 yr earlier than sporadic AD",
        "biomarker_pattern": "CSF Aβ42 ↓ + p-tau181 ↑ + t-tau ↑; amyloid PET 100% positive symptomatic",
        "motor_pattern": "Spastic paraparesis 18–20%; cerebellar 10%; seizures 30%; extrapyramidal rare",
    },
    # ── PSEN2 — Familial Alzheimer Disease 4 ──────────────────────────────────
    {
        "gene": "PSEN2",
        "protein": "Presenilin-2 (γ-Secretase Catalytic Subunit 2)",
        "alias": (
            "PSEN2; OMIM gene 600759; Alzheimer Disease-4 (FAD4) #606889; 1q42.13; 448 aa; ~50 kDa; "
            "AD dominant; LOWER penetrance ~95% (not 100%); latest onset of FAD genes (50–70s); "
            "Volga German kindred: Asn141Ile founder variant — many families from Volga River German settlers; "
            "also found in Italian pedigrees (Ile143Thr); "
            "functions identically to PSEN1 in γ-secretase but with lower endogenous expression in neurons"
        ),
        "aa": "448 aa",
        "kDa": "~50 kDa",
        "locus": "1q42.13",
        "omim_gene": 600759,
        "omim_disease": 606889,
        "inheritance": "AD dominant (penetrance ~95%, lower than PSEN1)",
        "gene_class": (
            "Paralogue of PSEN1; 67% amino acid identity; same 9-TM topology; same aspartyl dyad; "
            "forms γ-secretase complex with nicastrin/APH1b/PEN2 (different APH1 isoform than PSEN1 complex); "
            "PSEN2 variants shift Aβ42/Aβ40 ratio upward similarly to PSEN1 but often less severely; "
            "lower neuronal expression explains later onset and lower penetrance; "
            "obligate carriers of Asn141Ile over age 82 who remain unaffected are documented — key counselling point; "
            "PSEN2 Asn141Ile: also found in sporadic AD cohorts at low frequency — misinterpreted as risk allele; "
            "distinguish from PSEN1 before counselling because penetrance discussion differs critically"
        ),
        "phenotype": (
            "Onset: mean 63 years (range 40–88); "
            "Core syndrome: similar to PSEN1 — amnestic dementia; episodic memory earliest; "
            "Milder than PSEN1: less spastic paraparesis (<5%), fewer seizures (<15%); "
            "Biomarkers: CSF Aβ42 ↓, p-tau ↑ — similar pattern to PSEN1; Amyloid PET positive; "
            "Variability WITHIN families: some Asn141Ile carriers develop AD at 55, others asymptomatic at 82; "
            "APOE ε4 co-inheritance accelerates onset in PSEN2 carriers (gene–gene interaction)"
        ),
        "hallmark": (
            "LOWER PENETRANCE: ~5% of PSEN2 pathogenic variant carriers do NOT develop AD; "
            "CRITICAL for genetic counselling: NOT as deterministic as PSEN1; "
            "VOLGA GERMAN ANCESTRY: Asn141Ile — ask ancestry questions in all early-onset AD; "
            "APOE ε4 CO-INHERITANCE: significantly accelerates onset in PSEN2 carriers; "
            "LATE-ONSET END of FAD spectrum: PSEN2 can present in 70s — overlaps with sporadic AD age range; "
            "GENE PANEL ORDER: PSEN1 → PSEN2 → APP in early-onset FAD hierarchy; "
            "PSEN2 variants of uncertain significance (VUS): common challenge; segregation data crucial"
        ),
        "treatment_alert": (
            "LECANEMAB/DONANEMAB: same eligibility as PSEN1 if amyloid-confirmed early AD; "
            "ARIA risk: APOE ε4 homozygotes on anti-amyloid antibodies = HIGHEST ARIA risk; "
            "PSEN2 + APOE ε4/ε4: discuss ARIA risk carefully before initiating; "
            "GENETIC COUNSELLING: explicitly communicate incomplete penetrance (~95%, not 100%); "
            "Presymptomatic testing: some carriers will never develop AD — this complicates insurance/employment implications; "
            "APOE ε4 testing: test concurrently for ARIA risk stratification if anti-amyloid therapy planned; "
            "FAMILY COUNSELLING: different message from PSEN1 — 'high but not certain risk'; "
            "DIAN: PSEN2 families may be eligible for prevention trials (lower enrollment than PSEN1)"
        ),
        "key_ddx": (
            "PSEN1-FAD3: earlier onset, higher penetrance, spastic paraparesis, seizures more common; "
            "Sporadic AD APOE ε4/ε4: onset 60s–70s; similar age range to PSEN2; no pathogenic variant; "
            "VCI (vascular cognitive impairment): stepwise progression; white matter lesions; no amyloid PET; "
            "DLB: parkinsonism + fluctuation + visual hallucinations + RBD; alpha-synuclein pathology; "
            "Normal pressure hydrocephalus: gait ataxia + incontinence + dementia triad; ventricular dilatation"
        ),
        "onset_pattern": "Mean 63 years (range 40–88); latest onset of all FAD genes; overlaps sporadic AD",
        "biomarker_pattern": "CSF Aβ42 ↓ + p-tau ↑; amyloid PET positive; APOE ε4 co-inheritance accelerates onset",
        "motor_pattern": "Spastic paraparesis <5%; seizures <15%; generally milder than PSEN1",
    },
    # ── APP — FAD type 2 / Cerebral Amyloid Angiopathy ───────────────────────
    {
        "gene": "APP",
        "protein": "Amyloid Precursor Protein (APP770)",
        "alias": (
            "APP; OMIM gene 104760; Alzheimer Disease-2 (FAD2) #104300 + Hereditary Cerebral Haemorrhage with "
            "Amyloidosis Dutch type (HCHWA-D) + CAA type 1; 21q21.3; 770 aa (APP770 isoform); ~87 kDa; "
            "AD dominant; single gene — multiple distinct mutations producing AD or CAA; "
            "chromosome 21 trisomy (Down syndrome) → 3 copies APP → universal AD by 40s; "
            "APP duplications: all Down-like onset 50s; V717I (London), V717F (Indiana), A673V (recessive) "
        ),
        "aa": "770 aa",
        "kDa": "~87 kDa",
        "locus": "21q21.3",
        "omim_gene": 104760,
        "omim_disease": 104300,
        "inheritance": "AD dominant (FAD2/CAA); rare AR (A673V Italian recessive AD)",
        "gene_class": (
            "Type I transmembrane protein; physiological roles: synaptic plasticity, neuronal differentiation, "
            "copper binding, vesicular transport; "
            "sequential cleavage: α (ADAM10) + γ → non-amyloidogenic (sCAPPα); "
            "β (BACE1, β-secretase) + γ → Aβ40/Aβ42 amyloidogenic pathway; "
            "Swedish mutation (Lys670Asn/Met671Leu, KM→NL): BACE1 cleavage ↑ 100×; overproduction model; "
            "London mutation (Val717Ile in APP770 = Val710Ile in APP695): γ-secretase cut shifts toward Aβ42; "
            "Dutch mutation (Glu693Gln = Glu22Gln in Aβ): within Aβ sequence → CAA NOT neuritic plaques; "
            "CAA mutations (Dutch, Italian, Arctic, Iowa, Flemish): Aβ deposits in vessel walls → lobar ICH; "
            "A673T (Iceland, Jonsson 2012): PROTECTIVE — reduces BACE1 cleavage 40%; opposite of Swedish; "
            "A673V (Italian, Di Fede 2009): recessive pathogenic — heterozygous = protective-like; "
            "APP duplications: all 21q21 CNV duplications → overproduction of all Aβ species (dose effect)"
        ),
        "phenotype": (
            "FAD2 (V717I/V717F): onset mean 52 years; amnestic dementia similar to PSEN1; "
            "APP duplication: onset 50s; often accompanies early CAA features; "
            "CAA-type (Dutch Glu22Gln / Glu693Gln): recurrent LOBAR cerebral haemorrhage from age 40–60; "
            "Transient focal neurological episodes (TFNE); progressive cognitive decline AFTER haemorrhages; "
            "CAA-related inflammation: headache + seizures + white matter signal; "
            "Down syndrome: ALL patients with Trisomy 21 develop AD neuropathology by age 40–45; "
            "Biomarkers: amyloid PET positive; CAA-type may show cortical superficial siderosis on MRI"
        ),
        "hallmark": (
            "LOBAR ICH in young patient (40–60) + family history → APP CAA mutation (Dutch/Italian); "
            "NEVER ANTICOAGULATE APP-CAA lobar ICH — dramatically increases recurrent haemorrhage; "
            "NEVER start anti-amyloid antibodies (lecanemab/donanemab) in established CAA — ARIA-H catastrophic; "
            "DOWN SYNDROME + dementia in 40s: APP trisomy mechanism (expect it, not a diagnostic surprise); "
            "SWEDISH APP: Alzheimer's research positive control — biomarkers most robust; "
            "PROTECTIVE A673T (Icelandic): found in carriers who do NOT develop AD despite age — research interest; "
            "BACE1 inhibitor rationale strongest for Swedish APP (direct substrate)"
        ),
        "treatment_alert": (
            "APP-CAA LOBAR ICH — ANTICOAGULATION ABSOLUTELY CONTRAINDICATED: "
            "warfarin, NOACs, heparin all increase re-bleeding risk catastrophically in established CAA; "
            "use antiplatelet monotherapy ONLY if prosthetic valve etc. (unavoidable — haematology referral); "
            "ANTI-AMYLOID ANTIBODIES + CAA: ARIA-H microhaemorrhage risk very high — "
            "lecanemab/donanemab contraindicated in known CAA (heavy cortical siderosis on MRI); "
            "FAD2 (V717I) WITHOUT CAA: same lecanemab/donanemab eligibility as PSEN1; "
            "BACE1 inhibitors (verubecestat, atabecestat, lanabecestat): failed trials — 2018–2021 discontinued; "
            "APOE ε4 genotyping mandatory before anti-amyloid therapy — affects ARIA risk stratification; "
            "DOWN SYNDROME-AD: lecanemab trial evidence emerging; anti-amyloid not yet standard for DS-AD; "
            "SEIZURE management: levetiracetam first line; avoid enzyme-inducing AEDs"
        ),
        "key_ddx": (
            "PSEN1 FAD3: same amnestic profile; APP-CAA vs PSEN1 — lobar ICH vs spastic paraparesis; "
            "Sporadic CAA (APOE ε4 related): older onset (70s+); no APP pathogenic variant; lobar ICH; "
            "CADASIL (NOTCH3): subcortical infarcts + WML; migraine with aura; not lobar ICH; "
            "Amyloid-related imaging abnormalities (ARIA): therapy-related vs spontaneous CAA-related; "
            "CARASAL (CTSA): catechol-O-methyltransferase related; rare; "
            "Haemorrhagic transformation of ischaemic stroke: localisation differs from lobar CAA ICH"
        ),
        "onset_pattern": "FAD2 (V717I): mean 52 yr; CAA (Dutch): lobar ICH 40–60; duplication: 50s",
        "biomarker_pattern": "Amyloid PET positive; CAA: cortical superficial siderosis + microhaemorrhages on MRI",
        "motor_pattern": "CAA TFNE (transient focal neurological episodes); seizures 20%; no parkinsonism",
    },
    # ── MAPT — FTD-Parkinsonism linked to chromosome 17 ─────────────────────
    {
        "gene": "MAPT",
        "protein": "Microtubule-Associated Protein Tau (Tau 2+3+10 isoform)",
        "alias": (
            "MAPT; OMIM gene 157140; Frontotemporal Dementia and Parkinsonism Linked to Chr17 (FTDP-17) "
            "#600274; 17q21.31; 758 aa (longest isoform 2+3+10); ~79 kDa (longest); "
            "AD dominant; 6 brain isoforms (3R: 0N3R/1N3R/2N3R; 4R: 0N4R/1N4R/2N4R) by alternative splicing "
            "of exons 2,3,10; MAPT mutations: missense + splice site; splice shifts 3R/4R ratio; "
            "MAPT H1 haplotype: PSP + CBD risk (normal variation); H2: protective; H1/H1 homozygosity ↑ PSP risk"
        ),
        "aa": "758 aa",
        "kDa": "~79 kDa",
        "locus": "17q21.31",
        "omim_gene": 157140,
        "omim_disease": 600274,
        "inheritance": "AD dominant; MAPT H1 haplotype = PSP/CBD risk variant (NOT causal gene)",
        "gene_class": (
            "Tau: intrinsically disordered microtubule-stabilising protein; predominantly axonal; "
            "4 microtubule-binding repeat domains (in 4R isoforms) stabilise tubulin polymerisation; "
            "hyperphosphorylation (T181, S202, T205, T231, S262, S396, S422) → microtubule detachment → "
            "neurofibrillary tangles (NFTs) = paired helical filaments (PHFs); "
            "MAPT mutations group: (1) splice site exon 10 (IVS10+16, +3, +14) → 4R tauopathy; "
            "(2) missense (Pro301Leu, Pro301Ser, Arg406Trp, Gly272Val, Val337Met): misfolded tau aggregation; "
            "Pro301Leu: most common, many families worldwide; bvFTD or PSP-like; "
            "Arg406Trp: late onset, progressive aphasia presentation; "
            "3R tauopathy (Pick's disease): sporadic mostly; 4R (FTDP-17 MAPT): genetic; "
            "CSF p-tau181: ELEVATED in AD tauopathy; NORMAL or LOW in some MAPT mutations (paradox); "
            "plasma pTau217 correlates better with tangles than pTau181 in MAPT-FTDP"
        ),
        "phenotype": (
            "Onset: mean 57 years (range 40–75); "
            "Behavioral variant FTD (bvFTD): DISINHIBITION (most prominent early sign), apathy, compulsions, "
            "hyperphagia (sweet foods), loss of empathy; executive dysfunction before memory loss; "
            "Nonfluent/Agrammatic PPA: expressive aphasia, motor speech errors, apraxia of speech; "
            "PSP-like (MAPT Pro301Leu): vertical supranuclear gaze palsy, falls backward, 'surprised facies'; "
            "Corticobasal Syndrome (CBS): asymmetric apraxia, alien limb, cortical sensory loss; "
            "Parkinsonism present in 60% — levodopa usually poorly responsive; "
            "MRI: FRONTAL + TEMPORAL atrophy (asymmetric); MIDBRAIN atrophy in PSP-like (hummingbird sign); "
            "NO amyloid PET signal (tauopathy without amyloidosis); tau PET (flortaucipir) positive"
        ),
        "hallmark": (
            "DISINHIBITION + FRONTAL DEMENTIA in 50s + FAMILY HISTORY → MAPT or GRN first in FTD panel; "
            "VERTICAL GAZE PALSY + FALLS BACKWARD + SURPRISED LOOK → PSP-like MAPT variant; "
            "AMYLOID PET NEGATIVE in FTDP-17 MAPT: distinguishes from AD (both can present with dementia); "
            "HUMMINGBIRD SIGN: midbrain atrophy on sagittal MRI = PSP/MAPT Pro301Leu; "
            "ALIEN LIMB PHENOMENON: pathognomonic for CBS spectrum (MAPT, PSP, CBD); "
            "SWEET FOOD CRAVING + DISINHIBITION in 50s: PATHOGNOMONIC FTD cluster; "
            "C9orf72 repeat expansion test FIRST before MAPT if ALS features co-exist"
        ),
        "treatment_alert": (
            "NO APPROVED DISEASE-MODIFYING THERAPY for MAPT-FTDP-17 (2026); "
            "tau immunotherapies (semorinemab, gosuranemab, tilavonemab): all failed Phase II/III in sporadic PSP/AD; "
            "investigational: antisense oligonucleotides targeting MAPT exon 10 splice (WVE-003, ION-464 in trials); "
            "ANTIPSYCHOTICS: USE LOWEST DOSE / AVOID if possible — bvFTD patients hypersensitive to EPS; "
            "quetiapine or clozapine preferred if antipsychotic unavoidable (lowest D2 blockade); "
            "SSRI/SNRI: first-line for disinhibition, compulsions, hyperphagia — trazodone or sertraline; "
            "LEVODOPA: trial for parkinsonism — poor response expected but worth testing (10–20% benefit); "
            "CAREGIVER EDUCATION: disinhibition and impulsivity behaviours = neurological symptoms, not intentional; "
            "SAFETY: driving assessment mandatory (disinhibition + executive dysfunction); "
            "GENETIC COUNSELLING: first-degree relatives 50% risk; cascade panel testing recommended"
        ),
        "key_ddx": (
            "GRN-FTD: clinically indistinguishable; plasma PGRN <130 ng/mL = GRN; MAPT = normal PGRN; "
            "C9orf72 ALS/FTD: motor neuron signs (fasciculations, wasting) + FTD; C9 repeat test first; "
            "Sporadic PSP: amyloid PET negative; tau PET positive; NO MAPT pathogenic variant; H1 haplotype risk; "
            "Sporadic bvFTD (TDP-43 type B without GRN): no identified genetic cause in ~60%; "
            "DLB: visual hallucinations + fluctuation + parkinsonism + RBD; amyloid PET often positive; "
            "NPH: gait disorder + incontinence + cognitive triad; treatable — LP trial before FTD workup"
        ),
        "onset_pattern": "Mean 57 years (range 40–75); bvFTD or PSP-like or CBS; parkinsonism in 60%",
        "biomarker_pattern": "Amyloid PET NEGATIVE; tau PET positive; CSF p-tau181 low/normal (paradox); plasma pTau217 elevated",
        "motor_pattern": "Parkinsonism 60% (levodopa-poor); PSP-like gaze palsy; CBS alien limb; vertical gaze palsy",
    },
    # ── GRN — FTLD-TDP type A / FTD-GRN ──────────────────────────────────────
    {
        "gene": "GRN",
        "protein": "Progranulin (PGRN Growth Factor)",
        "alias": (
            "GRN; OMIM gene 138945; Frontotemporal Dementia type with TDP-43 (FTLD-TDP type A) #607485; "
            "17q21.32 (adjacent to MAPT but distinct); 593 aa; ~63 kDa (full-length); "
            "AD haploinsufficiency — loss-of-function mechanism (unlike most dominant diseases); "
            ">70 pathogenic null variants (frameshift, nonsense, splice, large deletion); "
            "plasma progranulin < 130 ng/mL: highly sensitive and specific diagnostic test; "
            "PGRN processed into granulins within lysosomes — lysosomal biogenesis regulator; "
            "GBA2 (TMEM106B) is a major genetic modifier of GRN-FTD penetrance and onset"
        ),
        "aa": "593 aa",
        "kDa": "~63 kDa",
        "locus": "17q21.32",
        "omim_gene": 138945,
        "omim_disease": 607485,
        "inheritance": "AD haploinsufficiency (LOF); NB: ~5% with pathogenic variant do not develop FTD",
        "gene_class": (
            "Secreted cysteine-rich growth factor; contains 7.5 granulin (GRN) repeat domains; "
            "lysosomal function: processed intralysosomally → granulins (GRN-A to -G); "
            "PGRN regulates lysosomal biogenesis via TFEB pathway and cathepsin activation; "
            "TDP-43 pathology: FTLD-TDP type A histology (short dystrophic neurites + compact inclusions); "
            "TDP-43 (TARDBP): RNA binding protein; nuclear depletion + cytoplasmic inclusions = TDP-43 proteinopathy; "
            "GRN haploinsufficiency → lysosomal dysfunction → TDP-43 mislocalisation → neurodegeneration; "
            "plasma PGRN: simple ELISA; GRN heterozygous carriers <130 ng/mL; homozygous = Neuronal Ceroid "
            "Lipofuscinosis type 11 (NCL11) — completely distinct childhood-onset lysosomal storage disease; "
            "AL-001/latozinemab: anti-sortilin antibody (blocks lysosomal sorting of PGRN) → ↑ plasma PGRN; "
            "AAV-GRN gene replacement: preclinical promising"
        ),
        "phenotype": (
            "Onset: mean 60 years (range 45–80); wide variability even within families; "
            "Behavioral variant FTD (bvFTD): disinhibition, apathy, compulsions, hyperphagia; "
            "Nonfluent/agrammatic PPA: expressive aphasia, motor speech apraxia (commoner in GRN vs MAPT); "
            "Corticobasal Syndrome: asymmetric parkinsonism + apraxia + alien limb; "
            "DISTINCTIVE GRN features: parietal atrophy (visuospatial deficits) earlier than MAPT; "
            "asymmetric atrophy on MRI more pronounced than other FTDs; "
            "Parkinsonism: present in 30–40% (less than MAPT); PGRN plasma < 130 ng/mL in ALL GRN carriers; "
            "NCL11: biallelic null GRN → ceroid lipofuscinosis, seizures, retinal dystrophy (childhood — NOT the FTD)"
        ),
        "hallmark": (
            "PLASMA PGRN < 130 ng/mL: DIAGNOSTIC for GRN heterozygous pathogenic variant; "
            "order plasma PGRN as FIRST TEST in all FTD-spectrum patients before genetic panel; "
            "COST-EFFECTIVE screening: plasma PGRN ELISA < $100 vs full panel; "
            "PARIETAL ATROPHY early: GRN-FTD presents with more visuospatial symptoms than typical bvFTD; "
            "ASYMMETRIC CORTICAL ATROPHY on MRI more pronounced in GRN than MAPT or C9; "
            "BIALLELIC GRN = NCL11 (childhood lysosomal disease) — NOT adult FTD; "
            "TMEM106B (T185S) as MODIFIER: homozygous T/T = earlier onset/worse GRN-FTD; "
            "GRN on chr17q21 ADJACENT TO MAPT: both FTD genes on same chromosome → check both"
        ),
        "treatment_alert": (
            "LATOZINEMAB (AL-001, Alector): Phase III INFRONT-3 — anti-sortilin monoclonal antibody; "
            "mechanism: sortilin routes PGRN to lysosomes for degradation; block sortilin → ↑ PGRN 10–30%; "
            "INFRONT-2 showed biological target engagement (↑ CSF PGRN); INFRONT-3 clinical outcomes ongoing (2026); "
            "DO NOT WAIT for results to refer eligible patients to trials; "
            "NCE-PGRN (progranulin replacement): EFF1 phase completed; recombinant PGRN IV infusion; "
            "BEHAVIOURAL MANAGEMENT (same as MAPT): SSRIs (sertraline/trazodone) for disinhibition/compulsions; "
            "CAREGIVER EDUCATION: behaviours are neurological, not volitional; "
            "PLASMA PGRN MONITORING: longitudinal PGRN levels track disease progression in trials; "
            "GENETIC COUNSELLING: first-degree relatives 50%; cascade plasma PGRN test easy first step; "
            "NCL11 (biallelic GRN): ophthalmology (ERG), neurological monitoring, enzyme replacement investigational"
        ),
        "key_ddx": (
            "MAPT-FTDP-17: clinically identical; distinguish by plasma PGRN (low = GRN, normal = MAPT/other); "
            "C9orf72 ALS/FTD: motor neuron signs; C9 repeat expansion test (distinct from GRN/MAPT); "
            "Sporadic FTD TDP-43: no GRN mutation; normal plasma PGRN; environmental/other genetic cause; "
            "PSP/CBD sporadic: tau PET positive; normal PGRN; H1 haplotype risk; "
            "Semantic variant PPA (svPPA): temporal pole atrophy; TDP-43 type C (NOT type A); mostly sporadic; "
            "Lewy body disease: visual hallucinations, RBD, parkinsonism — different protein signature"
        ),
        "onset_pattern": "Mean 60 years (range 45–80); wide intra-family variability; asymmetric atrophy prominent",
        "biomarker_pattern": "Plasma PGRN < 130 ng/mL DIAGNOSTIC; amyloid PET negative; tau PET variable; TDP-43 PET emerging",
        "motor_pattern": "Parkinsonism 30–40%; CBS 15%; nonfluent PPA (apraxia of speech); less severe than MAPT",
    },
    # ── C9orf72 — ALS/FTD spectrum ────────────────────────────────────────────
    {
        "gene": "C9orf72",
        "protein": "C9orf72 protein (DENN-domain protein; hexanucleotide G4C2 repeat expansion)",
        "alias": (
            "C9orf72; OMIM gene 614260; ALS-FTD spectrum / FTD type 1 / ALS14 #105400; 9p21.2; "
            "481 aa (transcript variant 2, full-length protein); ~54 kDa; "
            "AD incomplete penetrance; most common genetic cause of BOTH familial ALS (~40%) and familial FTD (~25%); "
            "G4C2 repeat expansion in intron 1: <30 repeats = normal; 30–200 = intermediate/uncertain; "
            ">200 (usually 400–2500) = pathogenic full expansion; "
            "pure ALS, pure FTD, ALS+FTD COMBINATION in same patient = pathognomonic for C9orf72"
        ),
        "aa": "481 aa",
        "kDa": "~54 kDa",
        "locus": "9p21.2",
        "omim_gene": 614260,
        "omim_disease": 105400,
        "inheritance": "AD incomplete penetrance (~50% by age 65; ~80% by age 80; some asymptomatic at 90)",
        "gene_class": (
            "DENN (differentially expressed in normal and neoplastic cells) domain protein; "
            "C9orf72 protein functions: RAB GEF activity, autophagy-lysosomal pathway regulation, "
            "microglial innate immune regulation (C9orf72 KO = systemic autoimmunity in mice); "
            "PATHOMECHANISMS of expansion — THREE simultaneous mechanisms: "
            "(1) C9orf72 haploinsufficiency: reduced protein, lysosomal/autophagy dysfunction; "
            "(2) RNA foci: sense (GGGGCC) + antisense (CCCCGG) G-quadruplex RNA sequesters RBPs "
            "(hnRNP-H, ALYREF, SF2/ASF etc.) → splicing defects; "
            "(3) Repeat-associated non-ATG (RAN) translation: 6 reading frames → 5 DPRs (dipeptide repeat proteins): "
            "poly-GA, poly-GP, poly-GR (most toxic to nucleolus/nuclear pore), poly-PA, poly-PR; "
            "DPR poly-GR + poly-PR: insert into nuclear pore complex (FG-Nups) → nucleocytoplasmic transport failure; "
            "TDP-43 pathology: C9orf72 + C9orf72-ALS = TDP-43 type B inclusions (long dystrophic neurites); "
            "FUS pathology: RARE with C9 (usually FUS mutations); "
            "SOD1 variants are a distinct ALS cause — NOT C9; distinguish before trial enrollment (SOD1-ASOs)"
        ),
        "phenotype": (
            "Onset: mean 58 years (range 35–80); "
            "ALS-ONLY (50%): UMN + LMN signs; weakness, fasciculations, bulbar involvement; median survival 2–5 yr; "
            "FTD-ONLY (30%): bvFTD with disinhibition, apathy; cognitive decline; "
            "ALS+FTD COMBINATION (20%): both motor neuron disease AND frontal-behaviour syndrome IN SAME PATIENT; "
            "ALS+FTD COMBINATION is PATHOGNOMONIC for C9orf72 until proven otherwise; "
            "Psychosis: 4–8% C9orf72 carriers develop schizophrenia-like psychosis (rare but characteristic); "
            "CEREBELLAR features: ~10% ataxia; distinct from typical ALS; "
            "MRI: bilateral frontal + temporal atrophy (FTD); 'streaking' of corticospinal tract (ALS); "
            "EMG: denervation + fasciculations in ALS component"
        ),
        "hallmark": (
            "ALS + DEMENTIA (FTD) IN SAME PATIENT = C9orf72 until proven otherwise; "
            "C9 REPEAT EXPANSION TEST: separate PCR/Southern blot test — NOT covered by standard exome/WGS; "
            "ORDER C9 EXPANSION FIRST in any ALS or FTD patient — before full gene panel; "
            "PSYCHOSIS + ALS/FTD FAMILY HISTORY: check C9 (rare but distinctive); "
            "CEREBELLAR ATAXIA in ALS: C9 expansion (atypical ALS presentation); "
            "ANTICIPATION: repeat expansions can increase across generations; "
            "SOMATIC MOSAICISM: expansion length varies between tissues — blood vs brain may differ; "
            "FTD-ALS FAMILIES: ~50% of families positive for C9 in FTD-ALS linkage studies"
        ),
        "treatment_alert": (
            "RILUZOLE: approved for ALS; modest survival benefit (~3 months); "
            "EDARAVONE: approved ALS (rapidly progressive); IV/oral; moderate functional benefit; "
            "TOFERSEN (QALSODY): antisense oligonucleotide for SOD1-ALS ONLY — NOT C9orf72; "
            "C9orf72-specific ASOs: RO-7262786 (Roche), BIIB078, WVE-004 — Phase I/II; refer to trials; "
            "PHENOBARBITAL/CLOBAZAM: seizures if present (rare in ALS; more in FTD component); "
            "FTD COMPONENT: SSRIs for disinhibition; avoid antipsychotics if possible; "
            "RILUZOLE: equally applicable to C9-ALS as SOD1-ALS (non-specific glutamate modulator); "
            "MULTIDISCIPLINARY TEAM: MND clinic (neurologist + respiratory + dietitian + OT + PT + SLT + SW); "
            "PEG/RIG: nutritional support before weight loss >10%; "
            "NIV (non-invasive ventilation): when FVC <50% or orthopnoea — extends survival 6–19 months; "
            "GENETIC COUNSELLING: repeat testing in at-risk family (50% risk; expansion inheritance unpredictable)"
        ),
        "key_ddx": (
            "SOD1-ALS: pure LMN-dominant ALS; no FTD; SOD1 mutations NOT C9; tofersen applicable to SOD1 not C9; "
            "TARDBP (TDP-43) ALS: rare ALS gene; no FTD typically; no C9 expansion; "
            "FUS-ALS: younger onset (<40); no FTD; different pathology (FUS inclusions, not TDP-43); "
            "Sporadic ALS: no C9; C9 accounts for ~5–10% sporadic ALS; most sporadic = unknown; "
            "GRN-FTD: FTD only (no ALS); low plasma PGRN; "
            "Kennedy disease (SBMA X-linked bulbospinal): X-linked; AR repeat; only males; no FTD; slow progression"
        ),
        "onset_pattern": "Mean 58 years; ALS-only (50%), FTD-only (30%), ALS+FTD combination (20%)",
        "biomarker_pattern": "EMG denervation for ALS; bilateral frontal/temporal atrophy MRI for FTD; C9 repeat expansion (PCR/Southern)",
        "motor_pattern": "UMN+LMN signs (ALS); disinhibition/apathy (FTD); ALS+FTD combination is pathognomonic C9",
    },
    # ── LRRK2 — Autosomal Dominant Parkinson Disease ─────────────────────────
    {
        "gene": "LRRK2",
        "protein": "Leucine-Rich Repeat Kinase 2 (LRRK2, ROCO protein)",
        "alias": (
            "LRRK2; OMIM gene 609007; Parkinson Disease type 8 (PARK8) #607060; 12q12; 2527 aa; ~280 kDa; "
            "AD dominant; p.Gly2019Ser most common — frequency: ~1–2% sporadic PD (worldwide); "
            "15–40% of familial PD in Ashkenazi Jewish; 30–42% of familial PD in North African Arab; "
            "12% of sporadic PD in Ashkenazi Jewish; "
            "penetrance: age-dependent (30% by 60 yr; 75% by 80 yr; NOT 100%); "
            "LRRK2 inhibitors (DNL151/BIIB122) in Phase II/III clinical trials 2025–2026"
        ),
        "aa": "2527 aa",
        "kDa": "~280 kDa",
        "locus": "12q12",
        "omim_gene": 609007,
        "omim_disease": 607060,
        "inheritance": "AD dominant (age-dependent penetrance, NOT 100%)",
        "gene_class": (
            "Multidomain kinase belonging to ROCO protein family: "
            "N-terminal HEAT repeats → ARM repeat → ANK repeats → LRR (leucine-rich repeats) → "
            "ROC (Ras of complex proteins, GTPase) → COR (C-terminal of ROC) → kinase → WD40; "
            "ROC domain: GTPase activity (GTP hydrolysis to GDP); COR: GTPase regulatory; "
            "Kinase domain: serine/threonine kinase; substrates include Rab GTPases (Rab8A/10 at Thr72); "
            "p.Gly2019Ser: kinase domain activation loop → ↑ LRRK2 kinase activity 2–3× (GOF); "
            "hyper-phosphorylated Rab8A/10 → lysosomal/autophagy dysfunction; Rab-phosphorylation is LRRK2 "
            "biomarker in blood (pRab10 Thr72 assay for target engagement); "
            "Other variants: p.Arg1441Cys/Gly/His (ROC domain — Basque founder), p.Tyr1699Cys (COR domain); "
            "LRRK2 inhibitors: MLi-2 → DNL151 → BIIB122; Phase II/III LIGHTHOUSE, LUMA trials (2025–2026); "
            "LRRK2 overactivation: Parkinson; LRRK2 KO: lung pathology (lysosome defect in pneumocytes)"
        ),
        "phenotype": (
            "Onset: mean 63 years (range 40–90); "
            "Phenotype: clinically identical to idiopathic PD: "
            "rest tremor (60–70%), bradykinesia, rigidity, postural instability (later); "
            "Asymmetric onset: PATHOGNOMONIC for PD vs MSA/PSP (which are symmetric); "
            "GOOD levodopa response (85–90%) — characteristic LRRK2 feature; "
            "Slower progression than idiopathic PD; longer time to motor complications; "
            "Neuropathology: variable — Lewy bodies (synuclein, most common), NFTs (tau), TDP-43, or "
            "pure nigrostriatal degeneration WITHOUT inclusions (unique to LRRK2 among PD genes); "
            "Hyposmia: 70% prodromal marker (5–15 yr before motor); "
            "RBD (REM sleep behaviour disorder): 50% prodromal; constipation 60%; "
            "Cognitive decline: slower than idiopathic PD — dementia in <20% by 10 yr disease duration"
        ),
        "hallmark": (
            "ASHKENAZI JEWISH ANCESTRY + PD: LRRK2 p.Gly2019Ser testing MANDATORY; "
            "N.AFRICAN ARAB ANCESTRY + PD: LRRK2 testing mandatory (30–40%); "
            "SLOW PD PROGRESSION: LRRK2 Gly2019Ser = benign course relative to GBA/SNCA; "
            "NEUROPATHOLOGICAL VARIABILITY: LRRK2 can have tau, TDP-43, or NO inclusions — unique; "
            "pRab10 Thr72 BLOOD BIOMARKER: confirmed target engagement for LRRK2 inhibitors; "
            "PRODROMAL TRIAD: hyposmia + constipation + RBD beginning 10+ years before tremor; "
            "LRRK2 + GBA2 MODIFIER: Asn409Ser GBA2 co-variant (NOT GBA1) accelerates LRRK2-PD; "
            "GENETIC COUNSELLING: 50% offspring risk; penetrance ~75% by 80 yr (not certain disease)"
        ),
        "treatment_alert": (
            "LEVODOPA + CARBIDOPA: first-line PD treatment; LRRK2 patients EXCELLENT responders (85–90%); "
            "dopamine agonists (pramipexole, ropinirole): young-onset LRRK2 to delay levodopa dyskinesia; "
            "MAO-B inhibitors (rasagiline, selegiline, safinamide): adjunct therapy; "
            "DNL151/BIIB122 (denali/BIOGEN LRRK2 kinase inhibitor): Phase II/III; refer Gly2019Ser carriers to trial; "
            "pRab10 monitoring: target engagement biomarker for LRRK2 kinase inhibitors; "
            "RBD: clonazepam 0.5–2 mg OR melatonin 3–12 mg at night; "
            "HYPOSMIA: no treatment; document as prodromal marker for screening; "
            "CONSTIPATION: macrogol/lactulose; fibre; hydration; start early (prodromal = years before motor); "
            "DBS (deep brain stimulation): STN-DBS effective for LRRK2-PD as for idiopathic PD; "
            "GENETIC TESTING DECISION: discuss penetrance (not 100%) — some prefer not to know; "
            "GINA (Genetic Information Nondiscrimination Act) USA / PHIPA Canada: document genetic counselling"
        ),
        "key_ddx": (
            "SNCA-PD: earlier onset, rapid progression, dementia common; SNCA point mutation or triplication; "
            "GBA1-PD (not covered here): most common genetic PD risk (not dominant); earlier onset; DLB overlap; "
            "Idiopathic PD: no LRRK2 or SNCA mutation; sporadic; same phenotype; "
            "MSA (Multiple System Atrophy): symmetric parkinsonism + autonomic failure + cerebellar; POOR levodopa; "
            "PSP: axial rigidity + gaze palsy + falls backward; POOR levodopa; tau pathology; "
            "Drug-induced parkinsonism: antipsychotics, metoclopramide; symmetric; reversible on withdrawal"
        ),
        "onset_pattern": "Mean 63 years; clinically identical to idiopathic PD; slow progression; 85–90% levodopa responders",
        "biomarker_pattern": "DaTscan: dopaminergic deficit; pRab10 Thr72 blood biomarker for LRRK2 kinase activity; DAT SPECT reduced",
        "motor_pattern": "Asymmetric tremor + rigidity + bradykinesia; GOOD levodopa response (85–90%); slower progression",
    },
    # ── SNCA — PD / DLB / MSA spectrum ───────────────────────────────────────
    {
        "gene": "SNCA",
        "protein": "Alpha-Synuclein (SNCA, NACP, α-syn)",
        "alias": (
            "SNCA; OMIM gene 163890; Parkinson Disease type 1 (PARK1/4) #168601; 4q22.1; 140 aa; ~14 kDa; "
            "AD dominant: point mutations Ala53Thr (Greek founder), Ala30Pro (German), Glu46Lys (Spanish Basque), "
            "His50Gln, Gly51Asp (very severe, infantile), Ala53Glu; "
            "CNV: SNCA duplication → mid-severity PD; SNCA triplication → severe early-onset PD+DLB ≤40 yr; "
            "RBD (REM sleep behaviour disorder) 80% BEFORE motor symptoms — PATHOGNOMONIC prodrome; "
            "alpha-synuclein = PRINCIPAL COMPONENT of Lewy bodies and Lewy neurites in ALL PD/DLB"
        ),
        "aa": "140 aa",
        "kDa": "~14 kDa",
        "locus": "4q22.1",
        "omim_gene": 163890,
        "omim_disease": 168601,
        "inheritance": "AD dominant (point mutations + CNV duplications/triplications)",
        "gene_class": (
            "Intrinsically disordered small protein; three structural regions: "
            "N-terminal amphipathic lipid-binding helix (residues 1–60), NAC central hydrophobic domain "
            "(61–95, drives aggregation), C-terminal acidic tail (96–140); "
            "physiological role: presynaptic vesicle trafficking, SNARE complex assembly, dopamine release; "
            "pathological aggregation pathway: monomer → oligomers (soluble toxic) → protofibrils → "
            "mature amyloid fibrils → Lewy body inclusion; "
            "prion-like propagation: cell-to-cell spread via exosomes/endocytosis (Braak staging hypothesis); "
            "SNCA fibrils: ≥5 distinct strains (MSA fibril vs PD fibril — different PMCA fingerprint); "
            "SNCA triplication (PARK4): 4 copies → 2× protein → very early PD+DLB+autonomic failure; "
            "Gly51Asp: most severe point mutation — parkinsonian + dementia in teens/twenties; "
            "PRASINEZUMAB (anti-SNCA): Phase IIb PADOVA trial — SNCA-specific monoclonal antibody (Roche/Prothena); "
            "SNL3/4 (LIXISENATIDE) GLP-1R: neuroprotective signal in 1 PD trial — mechanism via SNCA clearance"
        ),
        "phenotype": (
            "Onset: SNCA triplication mean 38 yr; point mutations (Ala53Thr) mean 46 yr; "
            "PD + Dementia WITH LEWY BODIES (DLB) combined spectrum in SNCA; "
            "Ala53Thr: classic PD but with RAPID COGNITIVE DECLINE, early autonomic failure; "
            "Triplication: severe — early PD + DLB + autonomic failure (orthostatic hypotension + urinary); "
            "Gly51Asp: pediatric/young adult onset; rapidly progressive; "
            "RBD 80%: PREM sleep-related movement disorder (acting out dreams) > 5–15 yr BEFORE motor; "
            "POLYSOMNOGRAPHY: abnormal REM atonia = prodromal SNCA-PD; "
            "DLB features: fluctuating cognition, complex visual hallucinations, parkinsonism, RBD; "
            "Autonomic: orthostatic hypotension (30–50%), constipation, urinary retention; "
            "Lewy bodies in CARDIAC sympathetic nerves: MIBG scintigraphy reduced = sympathetic denervation"
        ),
        "hallmark": (
            "RBD (REM SLEEP BEHAVIOUR DISORDER) = EARLIEST MARKER: 80% of SNCA-PD have RBD 5–15 yr before tremor; "
            "COMPLEX VISUAL HALLUCINATIONS (CVH) in PD = Lewy body disease; SNCA pathology; "
            "FLUCTUATING COGNITION: waxing-waning alertness = DLB (not AD); "
            "LEWY BODY = EOSINOPHILIC CYTOPLASMIC INCLUSION: contains alpha-synuclein (SNCA), ubiquitin, p62; "
            "MIBG SCINTIGRAPHY REDUCED: cardiac sympathetic denervation in SNCA-Lewy body disease; "
            "TRIPLICATION PATIENT: young PD ≤40 yr + family history = SNCA CNV test; "
            "GLP-1R AGONIST SIGNAL: lixisenatide single trial → SNCA clearance hypothesis; "
            "PRASINEZUMAB Phase IIb ongoing: anti-SNCA antibody for PD (PADOVA trial)"
        ),
        "treatment_alert": (
            "LEVODOPA/CARBIDOPA: effective for parkinsonism (60–70% responders in SNCA — less than LRRK2); "
            "RIVASTIGMINE (cholinesterase inhibitor): approved for PARKINSONS DISEASE DEMENTIA (PDD); "
            "also used for DLB dementia (off-label but evidence-based); "
            "DONEPEZIL: second choice for PD dementia; "
            "ANTIPSYCHOTICS: HALOPERIDOL + RISPERIDONE ABSOLUTELY CONTRAINDICATED in DLB/SNCA-PD; "
            "DLB/SNCA patients are exquisitely sensitive to D2 blockers → severe neuroleptic sensitivity: "
            "acute rigidity, confusion, fever, respiratory failure, death; "
            "QUETIAPINE (low dose, 12.5–25 mg): preferred if antipsychotic absolutely needed; "
            "PIMAVANSERIN (5-HT2A inverse agonist, NUPLAZID): approved for PD psychosis; no EPS; "
            "CLONAZEPAM/MELATONIN: RBD treatment; "
            "PRASINEZUMAB trial: refer early-stage SNCA-PD to PADOVA or extension trials; "
            "DBS: less predictable benefit in SNCA-PD than LRRK2 (cognitive co-morbidity limits candidacy); "
            "MIDODRINE + FLUDROCORTISONE: orthostatic hypotension; "
            "SKIN BIOPSY for SNCA: phosphorylated SNCA in dermal nerves = prodromal biomarker (Syn-One test)"
        ),
        "key_ddx": (
            "LRRK2-PD: clinically identical; LRRK2 = GOOD levodopa response + SLOW progression; "
            "SNCA-PD: faster, more dementia, more autonomic; "
            "Idiopathic DLB: Lewy body pathology without SNCA point mutation/CNV; sporadic; "
            "MSA-P / MSA-C: MSA also has SNCA inclusions (glial cytoplasmic inclusions GCI); POOR levodopa; "
            "MULTIPLE SYSTEM ATROPHY-parkinsonism: severe autonomic failure dominant; "
            "Idiopathic PD: no SNCA mutation; SNCA accounts for <1% PD overall; "
            "DLB vs AD dementia: fluctuating cognition + visual hallucinations + parkinsonism = DLB; "
            "Haloperidol sensitivity test (AVOID): neuroleptic challenge is DANGEROUS — not a diagnostic manoeuvre"
        ),
        "onset_pattern": "Triplication ≤38 yr; Ala53Thr ~46 yr; duplication ~55 yr; RBD precedes motor by 5–15 yr",
        "biomarker_pattern": "DaTscan positive; MIBG scintigraphy reduced (cardiac sympathetic); skin biopsy pSNCA; RBD polysomnography",
        "motor_pattern": "Asymmetric tremor (less dominant than LRRK2); rapid dementia (triplication); visual hallucinations; RBD prodrome",
    },
]


def _simulate_gene(gene_data: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    patients = []

    for pid in range(1, n + 1):
        # ── onset age varies by gene
        if gene == "PSEN1":
            onset_age = rng.gauss(45, 8)
        elif gene == "PSEN2":
            onset_age = rng.gauss(63, 9)
        elif gene == "APP":
            onset_age = rng.gauss(52, 10)
        elif gene == "MAPT":
            onset_age = rng.gauss(57, 9)
        elif gene == "GRN":
            onset_age = rng.gauss(60, 10)
        elif gene == "C9orf72":
            onset_age = rng.gauss(58, 10)
        elif gene == "LRRK2":
            onset_age = rng.gauss(63, 10)
        else:  # SNCA
            onset_age = rng.gauss(48, 10)
        onset_age = max(28, min(88, onset_age))

        # ── amyloid PET positive
        if gene in ("PSEN1", "PSEN2", "APP"):
            amyloid_pet = rng.random() < 0.97
        elif gene in ("MAPT", "GRN", "C9orf72"):
            amyloid_pet = rng.random() < 0.06
        else:  # LRRK2, SNCA — occasionally co-pathology
            amyloid_pet = rng.random() < 0.18

        # ── lobar ICH (APP-CAA only significantly)
        lobar_ich = rng.random() < (0.30 if gene == "APP" else 0.02)

        # ── parkinsonism
        if gene in ("LRRK2", "SNCA"):
            parkinsonism = rng.random() < 0.95
        elif gene in ("MAPT",):
            parkinsonism = rng.random() < 0.60
        elif gene in ("GRN",):
            parkinsonism = rng.random() < 0.38
        elif gene == "C9orf72":
            parkinsonism = rng.random() < 0.12
        else:
            parkinsonism = rng.random() < 0.08

        # ── RBD
        if gene == "SNCA":
            rbd = rng.random() < 0.80
        elif gene == "LRRK2":
            rbd = rng.random() < 0.50
        else:
            rbd = rng.random() < 0.10

        # ── levodopa response
        if gene in ("LRRK2",):
            levodopa_response = rng.random() < 0.87 if parkinsonism else False
        elif gene in ("SNCA",):
            levodopa_response = rng.random() < 0.62 if parkinsonism else False
        elif gene in ("MAPT",):
            levodopa_response = rng.random() < 0.18 if parkinsonism else False
        else:
            levodopa_response = rng.random() < 0.50 if parkinsonism else False

        # ── behavioural features (FTD genes)
        if gene in ("MAPT", "GRN"):
            behavioural = rng.random() < 0.78
        elif gene == "C9orf72":
            behavioural = rng.random() < 0.52
        else:
            behavioural = rng.random() < 0.15

        # ── ALS motor neuron signs (C9 only significantly)
        als_signs = rng.random() < (0.68 if gene == "C9orf72" else 0.02)

        # ── spastic paraparesis (PSEN1 specific)
        spastic_paraparesis = rng.random() < (0.19 if gene == "PSEN1" else 0.02)

        # ── seizures
        if gene == "PSEN1":
            seizures = rng.random() < 0.30
        elif gene == "APP":
            seizures = rng.random() < 0.20
        elif gene == "PSEN2":
            seizures = rng.random() < 0.14
        else:
            seizures = rng.random() < 0.07

        # ── plasma PGRN low (GRN carriers)
        pgrn_low = rng.random() < (0.98 if gene == "GRN" else 0.02)

        # ── severity
        # Triplication SNCA and PSEN1 = more severe
        sev_score = rng.gauss(0, 1)
        if gene in ("PSEN1", "SNCA"):
            sev_score += 0.5
        elif gene in ("C9orf72",):
            sev_score += 0.3
        elif gene in ("LRRK2",):
            sev_score -= 0.4

        severity = "mild" if sev_score < -0.3 else ("severe" if sev_score > 0.8 else "moderate")

        # ── cognitive score (MMSE equivalent at diagnosis)
        if gene in ("PSEN1", "SNCA"):
            mmse = max(5, min(28, round(rng.gauss(20, 5))))
        elif gene in ("MAPT", "GRN", "C9orf72"):
            mmse = max(5, min(28, round(rng.gauss(19, 5))))
        elif gene in ("LRRK2",):
            mmse = max(18, min(30, round(rng.gauss(27, 2))))
        else:
            mmse = max(10, min(28, round(rng.gauss(22, 4))))

        # ── family history positive
        fam_hx = rng.random() < (0.70 if gene in ("PSEN1","LRRK2","SNCA") else 0.60)

        patients.append({
            "pid": f"{gene}-{pid:03d}",
            "gene": gene,
            "seed": seed,
            "onset_age": round(onset_age, 1),
            "amyloid_pet_positive": amyloid_pet,
            "lobar_ich": lobar_ich,
            "parkinsonism": parkinsonism,
            "rbd": rbd,
            "levodopa_response": levodopa_response,
            "behavioural_features": behavioural,
            "als_signs": als_signs,
            "spastic_paraparesis": spastic_paraparesis,
            "seizures": seizures,
            "pgrn_low": pgrn_low,
            "mmse_at_diagnosis": mmse,
            "family_history_positive": fam_hx,
            "severity": severity,
        })

    return patients


def _cohort_stats(pts: list) -> dict:
    n = len(pts)
    if n == 0:
        return {}

    def pct(key, val=True):
        return round(100 * sum(1 for p in pts if p.get(key) == val) / n, 1)

    def pct_bool(key):
        return round(100 * sum(1 for p in pts if p.get(key)) / n, 1)

    ages = [p["onset_age"] for p in pts]
    mmse_vals = [p["mmse_at_diagnosis"] for p in pts]
    mean_onset = round(sum(ages) / n, 1)
    mean_mmse = round(sum(mmse_vals) / n, 1)

    return {
        "n": n,
        "mean_onset_age": mean_onset,
        "mean_mmse_at_diagnosis": mean_mmse,
        "amyloid_pet_positive_pct": pct_bool("amyloid_pet_positive"),
        "lobar_ich_pct": pct_bool("lobar_ich"),
        "parkinsonism_pct": pct_bool("parkinsonism"),
        "rbd_pct": pct_bool("rbd"),
        "levodopa_response_pct": pct_bool("levodopa_response"),
        "behavioural_pct": pct_bool("behavioural_features"),
        "als_signs_pct": pct_bool("als_signs"),
        "spastic_paraparesis_pct": pct_bool("spastic_paraparesis"),
        "seizures_pct": pct_bool("seizures"),
        "pgrn_low_pct": pct_bool("pgrn_low"),
        "family_history_pct": pct_bool("family_history_positive"),
        "severity_mild_pct": pct("severity", "mild"),
        "severity_moderate_pct": pct("severity", "moderate"),
        "severity_severe_pct": pct("severity", "severe"),
    }


def _all_patients() -> list:
    all_pts = []
    for i, ge in enumerate(NEURODEGEN_GENES):
        seed = SEED_BASE + i
        pts = _simulate_gene(ge, seed, 40)
        all_pts.extend(pts)
    return all_pts


# ─── Public API functions ──────────────────────────────────────────────────────

def get_overview() -> dict:
    all_pts = _all_patients()
    agg = _cohort_stats(all_pts)
    return {
        "atlas_name": "Hereditary Neurodegeneration Atlas",
        "atlas_subtitle": "Complete 8-Gene Adult-Onset Neurodegenerative Disease Atlas",
        "n_genes": 8,
        "n_patients": len(all_pts),
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
        "genes": [g["gene"] for g in NEURODEGEN_GENES],
        "description": (
            "The Hereditary Neurodegeneration Atlas covers 8 clinically actionable genes across the full spectrum "
            "of hereditary adult-onset neurodegeneration: "
            "PSEN1 (the most common FAD gene, >300 variants, near-100% penetrance, earliest onset 30–60s), "
            "PSEN2 (FAD4 — Volga German founder Asn141Ile, lower penetrance ~95%, later onset 50–70s), "
            "APP (FAD2/CAA — Swedish overproduction, London γ-secretase shift, Dutch/Italian CAA with lobar ICH; "
            "anticoagulants ABSOLUTELY CONTRAINDICATED in established CAA), "
            "MAPT (FTDP-17 — bvFTD, PSP-like, CBS; amyloid PET NEGATIVE; no approved DMT), "
            "GRN (FTD-TDP type A — progranulin haploinsufficiency; plasma PGRN <130 ng/mL DIAGNOSTIC; "
            "latozinemab Phase III), "
            "C9orf72 (ALS/FTD spectrum — G4C2 hexanucleotide repeat; ALS+FTD combination PATHOGNOMONIC; "
            "most common genetic ALS/FTD cause), "
            "LRRK2 (AD Parkinson disease — Gly2019Ser most common; 85–90% levodopa response; "
            "DNL151/BIIB122 Phase II/III), "
            "and SNCA (α-synuclein PD/DLB — RBD 80% prodromal; Lewy body = eosinophilic synuclein inclusion; "
            "antipsychotics CONTRAINDICATED in DLB; prasinezumab Phase IIb). "
            "320 patients (8 × 40, seeds 1238–1245)."
        ),
        "aggregate_clinical": agg,
        "drug_alerts": [
            {
                "title": "APP-CAA LOBAR ICH: ANTICOAGULATION ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "Cerebral Amyloid Angiopathy (APP Dutch/Italian/Iowa/Flemish mutations) → "
                    "Aβ deposits in cortical vessel walls → fragile vessels → recurrent lobar cerebral haemorrhage. "
                    "Warfarin, NOACs, heparin all dramatically increase re-bleeding risk. "
                    "Use antiplatelet monotherapy ONLY if prosthetic valve etc. (unavoidable — haematology co-management). "
                    "Anti-amyloid antibodies (lecanemab/donanemab) also CONTRAINDICATED in established CAA with "
                    "heavy cortical siderosis on MRI — catastrophic ARIA-H risk."
                ),
            },
            {
                "title": "DLB/SNCA-PD: HALOPERIDOL + RISPERIDONE ABSOLUTELY CONTRAINDICATED — fatal neuroleptic sensitivity",
                "body": (
                    "DLB and SNCA-Parkinson patients have EXQUISITE sensitivity to dopamine D2 blockers. "
                    "Haloperidol, risperidone, olanzapine (and all typical antipsychotics) can cause "
                    "acute life-threatening neuroleptic sensitivity: severe rigidity, confusion, fever, "
                    "respiratory failure, death. "
                    "If antipsychotic needed: quetiapine 12.5–25 mg lowest dose; pimavanserin (PD psychosis approved). "
                    "CLOZAPINE: effective but requires REMS monitoring (agranulocytosis risk)."
                ),
            },
            {
                "title": "LECANEMAB/DONANEMAB + APOE ε4 + anticoagulants: HIGHEST ARIA-H RISK",
                "body": (
                    "Anti-amyloid antibodies approved for early AD (MCI/mild AD). "
                    "ARIA-H (microhaemorrhage/siderosis) risk: greatest in APOE ε4 homozygotes. "
                    "Concurrent anticoagulation + anti-amyloid therapy = DO NOT CO-PRESCRIBE without "
                    "specialist haematology/neurology review. "
                    "Mandatory MRI monitoring: baseline, after 5th dose, after 7th dose. "
                    "Symptomatic ARIA: withhold therapy; severe ARIA: discontinue permanently."
                ),
            },
            {
                "title": "MAPT-FTDP-17: AVOID ANTIPSYCHOTICS — behavioural FTD neuroleptic sensitivity",
                "body": (
                    "FTD (MAPT/GRN/C9) patients are hypersensitive to neuroleptic-related extrapyramidal side-effects. "
                    "Antipsychotics for FTD agitation/disinhibition: USE LOWEST DOSE, SHORTEST DURATION. "
                    "Preferred: SSRIs (sertraline 50–200 mg, trazodone 50–100 mg) FIRST LINE for behavioural symptoms; "
                    "mirtazapine for hyperphagia + sleep; quetiapine if SSRI insufficient. "
                    "NEVER haloperidol/risperidone in FTD with parkinsonism (MAPT-PSP phenotype)."
                ),
            },
            {
                "title": "C9orf72: REPEAT EXPANSION TEST REQUIRED — NOT detected by standard exome/WGS",
                "body": (
                    "C9orf72 G4C2 hexanucleotide repeat expansion is a structural variant in non-coding intron 1. "
                    "Standard exome sequencing and whole-genome short-read NGS DO NOT reliably detect it. "
                    "Dedicated repeat-primed PCR (RP-PCR) + Southern blot IS REQUIRED. "
                    "Order C9 expansion test AS FIRST TEST in all ALS and FTD patients — before full gene panel. "
                    "Most laboratory panels include it — verify explicitly with the laboratory report."
                ),
            },
            {
                "title": "PSEN1/FAD: GENETIC COUNSELLING MANDATORY — near 100% penetrance, asymptomatic testing protocol required",
                "body": (
                    "PSEN1 pathogenic variants confer near-100% lifetime risk of Alzheimer disease. "
                    "Pre-test genetic counselling is MANDATORY before result disclosure. "
                    "Do NOT test minors unless clinically symptomatic (ethical/legal standard). "
                    "Post-test counselling MANDATORY for positive result. "
                    "Consider DIAN (Dominantly Inherited Alzheimer Network) trial referral — eligible PSEN1 carriers "
                    "may access prevention trials. "
                    "Distinguish from PSEN2 before counselling — PSEN2 penetrance ~95% (not 100%)."
                ),
            },
        ],
        "clinical_pearls": [
            "FAD hierarchy (early onset): PSEN1 (most common, >300 variants) → APP (duplications+V717I) → PSEN2 (latest onset, Volga German Asn141Ile)",
            "FTD hierarchy: C9orf72 (most common familial FTD ~25%) → GRN (plasma PGRN <130 ng/mL diagnostic) → MAPT (PSP/CBS/bvFTD)",
            "ALS hierarchy: C9orf72 (most common genetic ALS ~40% familial) → SOD1 → TARDBP/FUS (rare)",
            "ALS+FTD combination in SAME patient = C9orf72 until proven otherwise",
            "Amyloid PET: POSITIVE in PSEN1/PSEN2/APP; NEGATIVE in MAPT/GRN/C9/LRRK2/SNCA",
            "Plasma PGRN <130 ng/mL: screen ALL FTD patients — cheap, sensitive, specific for GRN",
            "SNCA RBD prodrome 80%: earliest marker of synucleinopathy (5–15 yr before motor onset)",
            "LRRK2 Gly2019Ser: highest frequency in Ashkenazi Jewish (15–40% familial PD) and North African Arab (30–42%)",
            "DLB/SNCA: haloperidol/risperidone ABSOLUTELY CONTRAINDICATED; fatal neuroleptic sensitivity",
            "APP-CAA: anticoagulants ABSOLUTELY CONTRAINDICATED in lobar ICH; anti-amyloid antibodies also CI with heavy CAA",
            "LEVODOPA RESPONSE RANKING: LRRK2 ~87% > SNCA ~62% > MAPT <20% (levodopa-resistant parkinsonism)",
            "All 8 genes: genetic counselling + cascade family testing mandatory before and after result disclosure",
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for i, ge in enumerate(NEURODEGEN_GENES):
        seed = SEED_BASE + i
        pts = _simulate_gene(ge, seed, 40)
        stats = _cohort_stats(pts)
        result[ge["gene"]] = {
            "gene": ge["gene"],
            "protein": ge["protein"],
            "alias": ge["alias"],
            "aa": ge["aa"],
            "kDa": ge["kDa"],
            "locus": ge["locus"],
            "omim_gene": ge["omim_gene"],
            "omim_disease": ge["omim_disease"],
            "inheritance": ge["inheritance"],
            "gene_class": ge["gene_class"],
            "phenotype": ge["phenotype"],
            "hallmark": ge["hallmark"],
            "treatment_alert": ge["treatment_alert"],
            "key_ddx": ge["key_ddx"],
            "onset_pattern": ge["onset_pattern"],
            "biomarker_pattern": ge["biomarker_pattern"],
            "motor_pattern": ge["motor_pattern"],
            "cohort_n": len(pts),
            "seed": seed,
            "stats": stats,
            "patients": pts,
        }
    return result


def get_definitions() -> dict:
    return {
        "atlas_name": "Hereditary Neurodegeneration Atlas",
        "terms": [
            {
                "term": "Amyloid Cascade Hypothesis (ACH)",
                "definition": (
                    "The dominant hypothesis in Alzheimer's disease pathogenesis: "
                    "Amyloid-β (Aβ) overproduction or impaired clearance → Aβ42 aggregation into oligomers → "
                    "plaques → tau hyperphosphorylation → NFTs → neuronal loss → dementia. "
                    "Genetic evidence: PSEN1/PSEN2 mutations ↑ Aβ42:Aβ40 ratio; APP duplications ↑ total Aβ; "
                    "APP A673T (Icelandic protective) reduces BACE1 cleavage; Down syndrome (trisomy 21) → 3× APP. "
                    "Clinical validation: anti-amyloid antibodies (lecanemab, donanemab) slow progression 30–35% "
                    "in early AD — first successful trials (CLARITY-AD 2023, TRAILBLAZER-ALZ2 2023)."
                ),
            },
            {
                "term": "γ-Secretase (PSEN1/PSEN2)",
                "definition": (
                    "Intramembrane aspartyl protease complex cleaving >100 substrates including APP and NOTCH. "
                    "Composition: presenilin-1 or -2 (catalytic) + nicastrin + APH1 + PEN2. "
                    "APP γ-cleavage: ε-cleavage → ζ-cleavage → γ-cleavage → Aβ40 or Aβ42 (processivity matters). "
                    "PSEN1/PSEN2 GOF mutations: impair processivity → incomplete trimming → ↑ Aβ42 (long toxic form). "
                    "NOTCH cleavage: releases NICD (NOTCH intracellular domain) for transcription — "
                    "explains why γ-secretase inhibitors (semagacestat) caused gastrointestinal toxicity (NOTCH off-target)."
                ),
            },
            {
                "term": "ARIA (Amyloid-Related Imaging Abnormalities)",
                "definition": (
                    "MRI signal changes caused by anti-amyloid immunotherapy (lecanemab, donanemab, aducanumab). "
                    "ARIA-E: vasogenic oedema + sulcal effusions (T2/FLAIR hyperintensity); "
                    "ARIA-H: microhaemorrhages + superficial siderosis (T2*/SWI hypointensity). "
                    "Risk factors: APOE ε4 homozygous, higher antibody dose, prior microhaemorrhages, concurrent anticoagulants. "
                    "Management: symptomatic ARIA → hold therapy; severe ARIA → discontinue. "
                    "Mandatory MRI schedule: baseline, after infusion 5, after infusion 7."
                ),
            },
            {
                "term": "Tau — NFT (Neurofibrillary Tangle)",
                "definition": (
                    "Tau: microtubule-stabilising protein; 6 isoforms in adult brain (3R + 4R). "
                    "Pathological phosphorylation (pTau) at T181, S202/T205, T231, S396: detaches from microtubules. "
                    "Aggregates as PHF (paired helical filaments) → NFTs (Braak staging I–VI). "
                    "MAPT mutations shift 3R/4R ratio (splice) or destabilise tau fold (missense) → tauopathy. "
                    "CSF pTau181 ↑: AD signature; LOW in MAPT-FTDP (paradox — mutation affects phosphorylation sites). "
                    "Tau PET (flortaucipir/MK-6240): positive in AD + MAPT tauopathy; negative in α-synucleinopathy."
                ),
            },
            {
                "term": "TDP-43 (TARDBP) Proteinopathy",
                "definition": (
                    "TDP-43 (TAR DNA binding protein 43): nuclear RNA-binding protein regulating splicing. "
                    "In FTLD-TDP (FTD with TDP-43 inclusions): nuclear depletion + cytoplasmic phospho-TDP-43 inclusions. "
                    "Subtypes by histology: Type A (GRN — short dystrophic neurites + compact inclusions), "
                    "Type B (C9orf72 — long dystrophic neurites), Type C (svPPA), Type D (VCP mutations). "
                    "TARDBP mutations (rare): cause ALS-FTLD with TDP-43 pathology directly (distinct from GRN/C9). "
                    "TDP-43 PET tracers: in development (not yet clinical); CSF TDP-43: research stage."
                ),
            },
            {
                "term": "C9orf72 RAN Translation + DPR Proteins",
                "definition": (
                    "Repeat-associated non-ATG (RAN) translation: ribosomes translate G4C2 repeat expansion "
                    "in all 6 reading frames without ATG start codon → 5 dipeptide repeat proteins (DPRs): "
                    "poly-GA (most abundant), poly-GR (most toxic), poly-PR (toxic), poly-PA, poly-GP (least toxic). "
                    "Poly-GR + poly-PR: insert into FG-nucleoporins of nuclear pore complex → "
                    "nucleocytoplasmic transport failure → TDP-43 mislocalisation → ALS/FTD pathology. "
                    "RNA foci: G-quadruplex GGGGCC + CCCCGG RNA sequesters hnRNP-H and other splicing factors → "
                    "splicing defects. Three simultaneous pathomechanisms in C9orf72."
                ),
            },
            {
                "term": "LRRK2 Kinase / pRab10 Biomarker",
                "definition": (
                    "LRRK2: leucine-rich repeat kinase 2; 2527 aa; phosphorylates Rab GTPases (Rab8A Thr72, Rab10 Thr72) "
                    "regulating lysosomal/endolysosomal trafficking. "
                    "p.Gly2019Ser: activation loop mutation → kinase activity ↑ 2–3× (GOF). "
                    "pRab10 T72 (phospho-Rab10): blood biomarker for LRRK2 kinase activity — measured by immunoassay; "
                    "elevated in Gly2019Ser carriers; normalised by LRRK2 kinase inhibitors → target engagement confirmed. "
                    "DNL151/BIIB122: CNS-penetrant LRRK2 kinase inhibitor; Phase II LIGHTHOUSE trial (LRRK2-PD) 2025–2026."
                ),
            },
            {
                "term": "Alpha-Synuclein — Lewy Body — Prion-like Spread",
                "definition": (
                    "α-Synuclein (SNCA): 140 aa presynaptic protein; intrinsically disordered → misfolding → "
                    "amyloid fibril aggregation → Lewy body (eosinophilic cytoplasmic inclusion with SNCA + ubiquitin + p62). "
                    "Prion-like propagation: misfolded SNCA templates native protein → cell-to-cell exosomal spread "
                    "→ Braak PD staging (olfactory/gut → brainstem → nigra → cortex) hypothesis. "
                    "SNCA fibril strains: PD strain vs MSA strain (glial cytoplasmic inclusions GCIs) biochemically distinct. "
                    "Skin biopsy (Syn-One test): phospho-SNCA in dermal nerve fibres — prodromal biomarker."
                ),
            },
            {
                "term": "REM Sleep Behaviour Disorder (RBD) — Prodromal Synucleinopathy",
                "definition": (
                    "RBD: loss of normal REM sleep muscle atonia → patient acts out vivid dreams: "
                    "vocalisation, punching, kicking, falling out of bed. "
                    "Diagnosis: polysomnography (PSG) showing REM without atonia (RSWA). "
                    "Prodromal synuclein marker: 80–90% of idiopathic RBD patients develop PD/DLB/MSA within 10–15 yr. "
                    "SNCA-PD: 80% have RBD; LRRK2-PD: ~50% have RBD. "
                    "Treatment: clonazepam 0.5–2 mg at bedtime OR melatonin 3–12 mg. "
                    "Safety modification: bed rails, mattress on floor, remove sharp objects near bed."
                ),
            },
            {
                "term": "DLB (Dementia with Lewy Bodies) — Neuroleptic Sensitivity",
                "definition": (
                    "DLB diagnostic criteria (McKeith 2017): core features: fluctuating cognition, complex visual "
                    "hallucinations, RBD, Parkinsonism (1–2 of 4 = probable DLB). "
                    "Supportive biomarkers: reduced DAT (DaTscan), reduced MIBG cardiac scintigraphy, REM without atonia PSG. "
                    "NEUROLEPTIC SENSITIVITY: haloperidol/risperidone/olanzapine → severe EPS + confusion + fever + "
                    "respiratory failure + death in 30–40% DLB patients. "
                    "MECHANISM: dopaminergic + cholinergic depletion in DLB → exquisite D2 sensitivity. "
                    "Safe alternatives: quetiapine 12.5–25 mg; pimavanserin (NUPLAZID — 5-HT2Ai, approved PD psychosis)."
                ),
            },
            {
                "term": "Progranulin (PGRN) / Latozinemab",
                "definition": (
                    "Progranulin (GRN): 593 aa secreted growth factor; cleaved intralysosomally → granulins; "
                    "regulates lysosomal biogenesis via TFEB pathway + cathepsin activation. "
                    "GRN haploinsufficiency → reduced PGRN → lysosomal dysfunction → TDP-43 mislocalisation → FTD. "
                    "Plasma PGRN ELISA: GRN heterozygous < 130 ng/mL; sensitivity ~95%, specificity ~98% vs controls. "
                    "Latozinemab (AL-001): anti-sortilin monoclonal antibody; sortilin routes PGRN to lysosomal "
                    "degradation; blocking sortilin → ↑ extracellular PGRN 10–30%. "
                    "INFRONT-3: Phase III GRN-FTD trial ongoing 2024–2026."
                ),
            },
            {
                "term": "DIAN (Dominantly Inherited Alzheimer Network)",
                "definition": (
                    "DIAN: international observational and interventional study of PSEN1, PSEN2, APP mutation carriers. "
                    "DIAN-OBS: longitudinal biomarker study — amyloid PET, tau PET, CSF, plasma, cognitive testing "
                    "in presymptomatic and symptomatic carriers. Biomarker changes begin 15–20 yr before symptoms. "
                    "DIAN-TU-001: prevention trials — gantenerumab + solanezumab (both failed endpoint 2023); "
                    "secondary analyses suggest benefit in some biomarkers; lecanemab-based prevention trial in planning. "
                    "Referral: any PSEN1/PSEN2/APP pathogenic variant carrier should be offered DIAN registration."
                ),
            },
            {
                "term": "CSF Biomarkers — Alzheimer vs FTD",
                "definition": (
                    "Alzheimer CSF signature: Aβ42 ↓ (aggregating in plaques) + pTau181 ↑ + tTau ↑. "
                    "Aβ42/Aβ40 ratio: more robust than Aβ42 alone (controls for pre-analytical variability). "
                    "pTau181/pTau217: elevated in AD; LOW/NORMAL paradox in MAPT-FTDP (mutation affects pTau sites). "
                    "FTD (GRN/C9): normal CSF Aβ; mildly ↑ tTau; neurofilament light (NfL) markedly ↑ (neurodegeneration marker). "
                    "Plasma biomarkers (2024): plasma pTau217 (PSEN1 > sporadic AD > FTD); plasma Aβ42/40; plasma NfL. "
                    "GFAP plasma ↑: reactive astrogliosis — elevated in ALL neurodegenerative diseases."
                ),
            },
        ]
    }
