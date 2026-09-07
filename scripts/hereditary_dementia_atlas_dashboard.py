#!/usr/bin/env python3
"""Hereditary-Dementia-Atlas — Complete 8-Gene Hereditary Dementia Atlas
PSEN1   (presenilin-1; 467 aa; 14q24.2; AD;
         Alzheimer's Disease-3 [EOAD] — most common cause of familial EOAD (50–70% EOFAD families);
         >300 pathogenic variants; E280A Paisas Colombian founder; virtually 100% penetrance;
         γ-secretase catalytic subunit; APP-CTF → Aβ42 overproduction; amyloid cascade;
         lecanemab/donanemab ARIA-highest-risk; seed SEED_BASE+0) ·
PSEN2   (presenilin-2; 448 aa; 1q42.13; AD;
         Alzheimer's Disease-4 [EOAD] — less common EOFAD; milder penetrance (~95%);
         N141I = Volga German founder; later onset than PSEN1 (range 40–75 y);
         γ-secretase complex component (non-catalytic); Aβ42:Aβ40 ratio ↑;
         seed SEED_BASE+1) ·
APP     (amyloid precursor protein; 770 aa; 21q21.3; AD;
         Alzheimer's Disease-1 / Cerebral Amyloid Angiopathy — duplication → EOAD (trisomy 21);
         V717I London most common; E693Q Dutch → CAA-dominant stroke phenotype;
         lecanemab/donanemab ARIA risk highest; 3× proteolytic cleavage;
         seed SEED_BASE+2) ·
GRN     (progranulin; 593 aa; 17q21.31; AD haploinsufficiency;
         Frontotemporal Dementia-GRN — plasma progranulin <100 ng/mL DIAGNOSTIC of pathogenic variant;
         TDP-43 type A inclusions; latozinemab (anti-sortilin) Phase 3 INFRONT-3;
         ubiquitin-TDP-43 neuropathology; seed SEED_BASE+3) ·
MAPT    (microtubule-associated protein tau; 758 aa; 17q21.31; AD;
         FTDP-17 — intronic splicing variants ↑ 4R:3R tau; exonic mutations direct tau dysfunction;
         PSP-like / CBS-like / PiD-like phenotypes; tau PET diagnostic;
         17q21 inversion haplotype H1/H2; seed SEED_BASE+4) ·
C9ORF72 (chromosome 9 open reading frame 72; 481 aa; 9p21.2; AD;
         ALS-FTD — GGGGCC hexanucleotide repeat; most common genetic cause of BOTH ALS AND FTD;
         repeat-primed PCR MANDATORY — standard PCR FAILS; ~40% familial ALS, ~25% familial FTD;
         incomplete penetrance ~50% by 65 y; RNA foci + DPR proteins;
         seed SEED_BASE+5) ·
PRNP    (prion protein; 253 aa; 20p13; AD/AR;
         Prion diseases — GSS (P102L most common; slowly progressive cerebellar ataxia then dementia);
         Fatal Familial Insomnia (FFI; D178N+M129V SAME allele); familial CJD (D178N+M129M);
         universally fatal; NO organ donation; biosafety level 2+; no treatment;
         seed SEED_BASE+6) ·
TREM2   (triggering receptor expressed on myeloid cells 2; 230 aa; 6p21.1; AD risk / AR Nasu-Hakola;
         R47H increases AD risk ~2–4×; biallelic LOF → Nasu-Hakola disease (PLOSL);
         microglial biology; AL001 (anti-TREM2 agonist) Phase 2 INVOKE-2;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1742–1749)
"""

import random

SEED_BASE = 1742

DEMENTIA_GENES = [
    # ── PSEN1 — Alzheimer's Disease-3 (EOAD) ──────────────────────────────────
    {
        "gene": "PSEN1",
        "protein": (
            "PSEN1 — 14q24.2 AD — γ-Secretase-Catalytic-467aa — "
            "Alzheimer-Disease-3-EOAD-Most-Common-EOFAD-50-70pct — "
            "E280A-Paisas-Colombian-Founder — "
            "Virtually-100pct-Penetrance — Aβ42-Overproduction-Amyloid-Cascade — "
            "Lecanemab-Donanemab-ARIA-Highest-Risk — >300-Pathogenic-Variants"
        ),
        "alias": (
            "PSEN1 (presenilin-1); OMIM gene 104311; "
            "Alzheimer's Disease-3 (EOAD) OMIM 607822. "
            "14q24.2; 467 aa; ~52 kDa; autosomal dominant; >300 pathogenic variants catalogued. "
            "FUNCTION: PSEN1 encodes the catalytic subunit (aspartyl protease) of the γ-secretase complex "
            "(with nicastrin, APH-1, PEN-2). γ-Secretase cleaves APP within its transmembrane domain → "
            "produces Aβ peptides of varying lengths. PSEN1 mutations shift the Aβ42:Aβ40 ratio upward "
            "(↑ Aβ42, or ↑ longer Aβ43/45), increasing amyloidogenic potential and fibril formation rate. "
            "PSEN1 also cleaves Notch, E-cadherin, and other substrates. "
            "MUTATION SPECTRUM: >300 missense variants throughout the 9 transmembrane domains; "
            "most are gain-of-toxic-function (altered γ-secretase cleavage products) rather than simple LOF. "
            "CLINICAL FEATURES: "
            "Onset age 30–60 y (mean ~45 y, earlier than PSEN2 or APP variants); "
            "rapidly progressive amnestic syndrome; hippocampal atrophy on MRI; "
            "amyloid PET strongly positive; CSF: ↓ Aβ42, ↑ tau, ↑ phospho-tau (181/217); "
            "plasma: ↑ p-tau217, ↑ p-tau181 (excellent screening in DOMINANTLY predictive mutation carriers); "
            "SPECIAL PRESENTATIONS: "
            "Posterior cortical atrophy (PCA) phenotype in some variants; "
            "spastic paraparesis + dementia (M連 phenotype, rare); "
            "seizures in 10–20%; myoclonus; visual/occipital features; "
            "FOUNDER MUTATION E280A (Glu280Ala): 5,000+ carriers in Antioquia, Colombia — Paisas study; "
            "world's largest EOFAD kindred; lecanemab prevention trial (AHEAD 3-45 enrolled carrier arm); "
            "onset ~44 y in E280A homozygotes; APOE3 Christchurch variant delays onset; "
            "DIAGNOSTIC APPROACH: PSEN1 genetic test positive → clinical diagnosis confirmed; "
            "no amyloid PET required for diagnosis in proven pathogenic variant carriers (but done in trials); "
            "family cascade testing mandatory; predictive testing requires genetic counselling; "
            "MANAGEMENT: no disease-modifying therapy approved for PSEN1-EOAD; "
            "symptomatic: acetylcholinesterase inhibitors (donepezil, rivastigmine, galantamine) — "
            "standard but modest benefit; memantine (moderate-severe stage); "
            "lecanemab / donanemab: FDA approved for early AD (amyloid-confirmed), but ARIA risk "
            "significantly HIGHER in PSEN1/PSEN2/APP carriers → specialist assessment required; "
            "APOE4 carriers AND genetic AD variants: maximum ARIA risk — some centres exclude or "
            "require monthly MRI monitoring; "
            "ADVANCE CARE PLANNING: early (20s–30s if family history known); genetic counselling; "
            "predictive testing protocol (18 y minimum; psychological support mandatory)."
        ),
        "locus": "14q24.2",
        "aa": 467,
        "kDa": 52,
        "omim_gene": "104311",
        "omim_disease": "607822",
        "inheritance": "AD; >300 pathogenic variants; virtually 100% penetrance",
        "gene_class": "γ-Secretase Catalytic Subunit — APP Cleavage — Aβ42 Overproduction",
        "key_alerts": [
            "PSEN1-LECANEMAB-DONANEMAB-ARIA-RISK: anti-amyloid immunotherapies (lecanemab, donanemab) cause ARIA (amyloid-related imaging abnormalities) at significantly HIGHER rates in PSEN1/PSEN2/APP variant carriers — monthly MRI monitoring mandatory; APOE4 compound heterozygotes (PSEN1 + APOE4/4) at maximum risk; neurology specialist referral before prescribing",
            "PSEN1-VIRTUALLY-100pct-PENETRANCE: most pathogenic PSEN1 variants show near-100% penetrance — positive predictive test result means very high lifetime risk of EOAD; genetic counselling and advance care planning must be initiated immediately in variant carriers",
            "PSEN1-ONSET-30-60Y-MEAN-45Y: PSEN1-EOAD begins 20–30 years earlier than sporadic AD; young patients in their 30s–40s with amnestic/behavioural symptoms and positive family history require URGENT genetic dementia evaluation, not reassurance",
            "PSEN1-CASCADE-TESTING: first-degree relatives have 50% risk; cascade testing should be offered to all at-risk relatives; children of affected parents may have reproductive decisions depending on carrier status",
        ],
        "etiologies": {
            "E280A_Paisas_Colombian": {
                "pct": 15,
                "phenotype": "Antioquia Colombia founder ~5000 carriers; onset ~44y; world's largest EOFAD kindred",
                "notes": "AHEAD 3-45 prevention trial enrolled; APOE3 Christchurch modifies penetrance",
            },
            "Other_missense_variants": {
                "pct": 70,
                "phenotype": "Global; >300 variants; onset 30-60y; amnestic or PCA phenotype",
                "notes": "M139T, L286P, G209V and hundreds more; variable phenotypic nuance by position",
            },
            "Spastic_Paraparesis_Phenotype": {
                "pct": 5,
                "phenotype": "L435F, C410Y, A431E — spastic paraparesis + cognitive decline",
                "notes": "Misdiagnosed as HSP; brain MRI shows white matter + amyloid; PSEN1 gene test resolves",
            },
            "PCA_Phenotype": {
                "pct": 10,
                "phenotype": "Posterior cortical atrophy; visual symptoms dominate early",
                "notes": "Misdiagnosed as eye disease; neuroimaging reveals parieto-occipital atrophy + amyloid",
            },
        },
        "stats": {
            "mean_onset_age": 44.5,
            "mean_dx_age": 46.2,
            "mean_dx_delay_months": 20,
            "prevalence_pct_of_EOFAD": 50,
        },
        "dx_delay_distribution": {
            "0–12 mo": 30,
            "12–24 mo": 35,
            "24–36 mo": 20,
            "36+ mo": 15,
        },
        "patients": [],
    },
    # ── PSEN2 — Alzheimer's Disease-4 (EOAD, Volga German) ────────────────────
    {
        "gene": "PSEN2",
        "protein": (
            "PSEN2 — 1q42.13 AD — γ-Secretase-Complex-448aa — "
            "Alzheimer-Disease-4-EOAD-Less-Common-EOFAD — "
            "N141I-Volga-German-Founder — Milder-Penetrance-95pct — "
            "Later-Onset-40-75y — Aβ42:Aβ40-Ratio-Increased"
        ),
        "alias": (
            "PSEN2 (presenilin-2); OMIM gene 600759; "
            "Alzheimer's Disease-4 (EOAD) OMIM 606889. "
            "1q42.13; 448 aa; ~51 kDa; autosomal dominant; ~14 known pathogenic variants. "
            "FUNCTION: PSEN2 is a non-catalytic presenilin paralog that can substitute for PSEN1 "
            "in the γ-secretase complex; shares 67% sequence identity with PSEN1. "
            "PSEN2 mutations also shift Aβ42:Aβ40 ratio upward but mechanism slightly different — "
            "PSEN2-containing γ-secretase generates longer Aβ species more readily. "
            "CLINICAL FEATURES: "
            "Onset age 40–75 y (mean ~55 y) — significantly later than PSEN1; "
            "penetrance ~95% — lower than PSEN1, allowing some variant carriers to remain unaffected past 75 y; "
            "clinical syndrome essentially identical to PSEN1-EOAD: amnestic, hippocampal atrophy, "
            "amyloid PET positive, CSF biomarkers consistent; "
            "FOUNDER MUTATION N141I (Asn141Ile): Volga German families; "
            "emigrated from Germany to Russia (Volga region) 18th century, then to USA; "
            "onset ~50–60 y in N141I; larger kindreds described in North America (Bird, 1988); "
            "DIAGNOSTIC NOTE: PSEN2 sequencing in an AD gene panel; milder penetrance means "
            "some variant-positive relatives may not develop clinical disease — counselling nuanced; "
            "plasma p-tau217 and CSF biomarkers positive before symptoms (EOAD biomarker biology same); "
            "MANAGEMENT: identical to PSEN1 (AChEI + memantine symptomatic; lecanemab/donanemab ARIA risk "
            "still elevated vs sporadic AD; advance care planning with family counselling)."
        ),
        "locus": "1q42.13",
        "aa": 448,
        "kDa": 51,
        "omim_gene": "600759",
        "omim_disease": "606889",
        "inheritance": "AD; ~14 known pathogenic variants; penetrance ~95%",
        "gene_class": "γ-Secretase Complex Component — PSEN1 Paralog — Aβ42 Overproduction",
        "key_alerts": [
            "PSEN2-PENETRANCE-95pct-NOT-100pct: unlike PSEN1, PSEN2 penetrance is ~95% — some variant carriers aged >75 y may remain unaffected; genetic counselling must explain reduced penetrance to avoid fatalism or unnecessary distress in older mutation-positive relatives",
            "PSEN2-LATER-ONSET-THAN-PSEN1: mean onset ~55 y vs ~45 y for PSEN1; clinicians should not exclude PSEN2 diagnosis in patients presenting in their 50s–60s with positive family history — this is the expected age range",
            "PSEN2-N141I-VOLGA-GERMAN-FOUNDER: ask about Volga German ancestry in EOAD families; N141I is the most common PSEN2 pathogenic variant; founder population with increased prevalence in descendants of Volga German immigrant communities",
            "PSEN2-LECANEMAB-ARIA-RISK: same elevated ARIA risk as PSEN1 with anti-amyloid therapy — specialist neurologist assessment required before initiating lecanemab or donanemab in any PSEN2 variant carrier",
        ],
        "etiologies": {
            "N141I_Volga_German": {
                "pct": 40,
                "phenotype": "Volga German ancestry; onset 50-65y; most common PSEN2 pathogenic variant",
                "notes": "First described Bird 1988; North American kindreds; penetrance ~95% by 75y",
            },
            "Other_missense_variants": {
                "pct": 60,
                "phenotype": "Global; various; onset 40-75y; amnestic syndrome predominantly",
                "notes": "M239V, T122P, I213T, V393M and others; milder than PSEN1 average",
            },
        },
        "stats": {
            "mean_onset_age": 55.2,
            "mean_dx_age": 57.5,
            "mean_dx_delay_months": 26,
            "prevalence_pct_of_EOFAD": 5,
        },
        "dx_delay_distribution": {
            "0–12 mo": 25,
            "12–24 mo": 30,
            "24–36 mo": 25,
            "36+ mo": 20,
        },
        "patients": [],
    },
    # ── APP — Alzheimer's Disease-1 / Cerebral Amyloid Angiopathy ─────────────
    {
        "gene": "APP",
        "protein": (
            "APP — 21q21.3 AD — Amyloid-Precursor-Protein-770aa — "
            "Alzheimer-Disease-1-EOAD-Duplication-Trisomy21-Mechanism — "
            "V717I-London-Most-Common-Missense — E693Q-Dutch-CAA-Stroke-Dominant — "
            "Lecanemab-Donanemab-ARIA-Highest-Risk-APP-Duplication — "
            "Triple-Proteolytic-Cleavage-α-β-γ-Secretase"
        ),
        "alias": (
            "APP (amyloid precursor protein); OMIM gene 104760; "
            "Alzheimer's Disease-1 (EOAD) OMIM 104300; Cerebral Amyloid Angiopathy (CAA) OMIM 605714. "
            "21q21.3; 770 aa; ~87 kDa; autosomal dominant (most EOAD variants); "
            "mutations at or near β-/γ-secretase cleavage sites. "
            "FUNCTION: APP is a type I transmembrane protein expressed in neurons, astrocytes, and other cells. "
            "Three proteolytic pathways: "
            "(1) Non-amyloidogenic: α-secretase (ADAM10/17) cleaves within Aβ domain → precludes amyloid formation → produces sAPPα (neuroprotective); "
            "(2) Amyloidogenic: β-secretase (BACE1) → C99 fragment; γ-secretase (PSEN1/2) → Aβ peptides (mainly Aβ40 normal, Aβ42 pathogenic); "
            "(3) Alternative γ-secretase cleavages produce Aβ43, Aβ45 (more aggregogenic). "
            "MUTATION CLASSES: "
            "Duplication: whole-gene duplication → doubled APP dose → EOAD (same mechanism as trisomy 21/Down syndrome); "
            "V717I (London): most common pathogenic missense; near γ-secretase cleavage site; ↑ Aβ42; onset 45–60 y; "
            "V717L, V717F (Indiana): similar to V717I; "
            "K670N/M671L (Swedish): near BACE1 site; ↑ total Aβ; used in transgenic mouse models; "
            "E693Q (Dutch): within Aβ domain; dominant CAA with cortical haemorrhage + strokes; "
            "Aβ self-associates abnormally with vessels; anticoagulation difficult; "
            "A673T (Icelandic protective variant): reduces BACE1 cleavage; protective against sporadic AD; "
            "LECANEMAB / DONANEMAB IN APP VARIANTS: "
            "APP duplication carriers: highest ARIA risk group (~50% ARIA in Phase 3 duplication subgroup); "
            "V717I and other missense: elevated ARIA risk compared to sporadic AD; "
            "E693Q (Dutch): theoretical concern — dense vascular amyloid; specialist evaluation mandatory."
        ),
        "locus": "21q21.3",
        "aa": 770,
        "kDa": 87,
        "omim_gene": "104760",
        "omim_disease": "104300",
        "inheritance": "AD (most); APP duplication (dominant-dosage); E693Q Dutch CAA (AD)",
        "gene_class": "Amyloid Precursor — β-/γ-Secretase Substrate — Aβ Source — CAA Risk",
        "key_alerts": [
            "APP-DUPLICATION-ARIA-RISK-50pct: APP duplication carriers have ~50% ARIA rate with lecanemab/donanemab (Phase 3 data) — the highest risk group; these patients should NOT receive anti-amyloid immunotherapy outside specialist centres with monthly MRI surveillance",
            "APP-E693Q-DUTCH-ANTICOAGULATION-DILEMMA: E693Q (Dutch-type CAA) causes lobar haemorrhages from amyloid angiopathy — anticoagulation for AF is DANGEROUS; specialist stroke/genetics joint clinic mandatory; aspirin may be preferred over anticoagulation; each AF treatment decision requires individual risk-benefit assessment",
            "APP-TRISOMY-21-MECHANISM: APP duplication mimics trisomy 21 AD — Down syndrome adults almost universally develop AD by their 50s due to gene dosage; clinicians evaluating EOAD families should enquire about Down syndrome relatives as indirect evidence of APP region dosage effect",
            "APP-A673T-PROTECTIVE-DO-NOT-REPORT-AS-PATHOGENIC: the Icelandic A673T variant reduces AD risk by ~40% — it is PROTECTIVE not pathogenic; misclassification as 'variant of uncertain significance' is a known error in some labs; specialist interpretation required for APP variants",
        ],
        "etiologies": {
            "APP_Duplication": {
                "pct": 20,
                "phenotype": "EOAD onset 40-65y; aggressive course; highest ARIA risk with immunotherapy",
                "notes": "Dosage effect = doubled Aβ production; same mechanism as trisomy 21 AD",
            },
            "V717I_London": {
                "pct": 35,
                "phenotype": "EOAD onset 45-60y; amnestic; most common pathogenic missense",
                "notes": "γ-secretase cleavage site; ↑ Aβ42:Aβ40 ratio; international distribution",
            },
            "Swedish_K670N_M671L": {
                "pct": 10,
                "phenotype": "EOAD onset 50-60y; used in transgenic models; Swedish kindreds",
                "notes": "BACE1 cleavage site; ↑ total Aβ; first mutation enabling transgenic AD mouse",
            },
            "E693Q_Dutch_CAA": {
                "pct": 25,
                "phenotype": "CAA-dominant; lobar haemorrhage strokes 40-65y; Dutch/Flemish ancestry",
                "notes": "Aβ domain mutation; vascular amyloid deposits; anticoagulation dangerous",
            },
            "Other_γ_site_variants": {
                "pct": 10,
                "phenotype": "V717L/F Indiana; rare variants near γ-secretase site",
                "notes": "Similar to V717I; ↑ Aβ42 confirmed biochemically",
            },
        },
        "stats": {
            "mean_onset_age": 52.0,
            "mean_dx_age": 54.5,
            "mean_dx_delay_months": 28,
            "prevalence_pct_of_EOFAD": 10,
        },
        "dx_delay_distribution": {
            "0–12 mo": 20,
            "12–24 mo": 30,
            "24–36 mo": 30,
            "36+ mo": 20,
        },
        "patients": [],
    },
    # ── GRN — Frontotemporal Dementia-GRN (TDP-43 Type A) ─────────────────────
    {
        "gene": "GRN",
        "protein": (
            "GRN — 17q21.31 AD-Haploinsufficiency — Progranulin-593aa — "
            "Frontotemporal-Dementia-GRN-TDP43-TypeA — "
            "Plasma-Progranulin-<100ngmL-DIAGNOSTIC — "
            "Latozinemab-Anti-Sortilin-Phase3-INFRONT3 — "
            "Ubiquitin-TDP43-Inclusions-Neuropathology"
        ),
        "alias": (
            "GRN (progranulin); OMIM gene 138945; "
            "Frontotemporal Dementia-GRN OMIM 607485. "
            "17q21.31; 593 aa; ~68 kDa (glycoprotein, secreted); autosomal dominant haploinsufficiency; "
            "~70 known pathogenic variants (mostly nonsense, frameshift, splice). "
            "FUNCTION: Progranulin (PGRN) is a secreted pleiotropic growth factor with roles in: "
            "lysosomal biogenesis and function (trafficked via sortilin/SORT1 receptor); "
            "microglial function (lysosomal degradation of synaptic material); "
            "neurotrophic support; anti-inflammatory regulation; "
            "TDP-43 nuclear clearance (indirectly, via lysosomal health). "
            "GRN haploinsufficiency → ↓ progranulin → lysosomal dysfunction → "
            "TDP-43 becomes phosphorylated, ubiquitinated, and forms cytoplasmic inclusions "
            "(characteristic TDP-43 type A, with short dystrophic neurites, NCI + NII). "
            "CLINICAL FEATURES: "
            "FRONTOTEMPORAL DEMENTIA (FTD) — predominantly behavioural variant FTD (bvFTD) or "
            "language variants (progressive non-fluent aphasia PNFA; semantic variant less common); "
            "onset 45–70 y (mean ~60 y); "
            "Parkinsonism in 10–20% (complicates Parkinson's disease DDx); "
            "rarely corticobasal syndrome (CBS) or PSP-like; "
            "PLASMA PROGRANULIN ASSAY — DIAGNOSTIC: "
            "plasma PGRN <100 ng/mL = PATHOGNOMONIC of GRN pathogenic variant (100% specificity "
            "if measured correctly with validated ELISA); no other FTD gene causes haploinsufficiency "
            "of secreted progranulin; "
            "PGRN 100–180 ng/mL = intermediate/borderline; check GRN sequencing; "
            "PGRN >180 ng/mL = normal; GRN LOF essentially excluded; "
            "TEST INTERPRETATION: PGRN assay should precede or accompany GRN gene panel; "
            "provides immediate categorical result; sample handling important (avoid freeze-thaw); "
            "LATOZINEMAB (AL001): anti-sortilin monoclonal antibody; blocks SORT1-mediated PGRN "
            "degradation → raises circulating PGRN; Phase 3 INFRONT-3 trial in GRN-FTD; "
            "NEUROPATHOLOGY: FTLD-TDP type A (long dystrophic neurites in cortex layer 2 "
            "+ NCIs [neuronal cytoplasmic inclusions] + NIIs [neuronal intranuclear inclusions]); "
            "ubiquitin/p62/TDP-43 IHC positive; tau negative."
        ),
        "locus": "17q21.31",
        "aa": 593,
        "kDa": 68,
        "omim_gene": "138945",
        "omim_disease": "607485",
        "inheritance": "AD haploinsufficiency; ~70 known pathogenic LOF variants",
        "gene_class": "Progranulin Haploinsufficiency — Lysosomal Biology — TDP-43 Type A",
        "key_alerts": [
            "GRN-PLASMA-PROGRANULIN-<100-DIAGNOSTIC: plasma progranulin <100 ng/mL is PATHOGNOMONIC for GRN pathogenic variant — this simple blood test provides near-certain diagnosis before genetic sequencing; all FTD patients should have plasma progranulin measured; validated ELISA only (commercial kits available)",
            "GRN-LATOZINEMAB-INFRONT3-CLINICAL-TRIAL: latozinemab (AL001, Alector) raises progranulin by blocking sortilin-mediated clearance — Phase 3 INFRONT-3 trial is ongoing in symptomatic GRN-FTD; refer all GRN-FTD patients to trial sites; this is the most advanced disease-modifying approach in FTD",
            "GRN-17q21-LOCUS-SAME-AS-MAPT: GRN and MAPT are both on chromosome 17q21.31 and can be confused on reports — confirm which gene is pathogenic; GRN-FTD shows TDP-43 type A neuropathology (NOT tau); MAPT-FTD shows tau pathology; biopsy/autopsy distinguishes definitively",
            "GRN-PARKINSONISM-MISDIAGNOSIS: 10–20% of GRN-FTD presents with prominent parkinsonism — misdiagnosed as Parkinson's disease or CBD; young-onset Parkinson's with FTD features or positive family history warrants GRN sequencing + plasma progranulin",
        ],
        "etiologies": {
            "Nonsense_Frameshift_Loss_of_Function": {
                "pct": 70,
                "phenotype": "bvFTD or PNFA; onset 55-70y; plasma PGRN <100 ng/mL",
                "notes": "Most are PTC or frameshift → NMD → haploinsufficiency; >70 different variants",
            },
            "Splice_Site_Variants": {
                "pct": 20,
                "phenotype": "Same FTD phenotype; plasma PGRN consistently low",
                "notes": "Intronic variants abolish splicing → loss of functional mRNA → haploinsufficiency",
            },
            "Missense_LOF": {
                "pct": 10,
                "phenotype": "bvFTD or CBS-like; plasma PGRN <100 ng/mL confirming LOF mechanism",
                "notes": "Less common; must confirm haploinsufficiency via plasma PGRN to establish LOF",
            },
        },
        "stats": {
            "mean_onset_age": 60.5,
            "mean_dx_age": 63.0,
            "mean_dx_delay_months": 30,
            "prevalence_pct_of_familial_FTD": 20,
        },
        "dx_delay_distribution": {
            "0–12 mo": 15,
            "12–24 mo": 25,
            "24–36 mo": 35,
            "36+ mo": 25,
        },
        "patients": [],
    },
    # ── MAPT — FTDP-17 (Tau) ──────────────────────────────────────────────────
    {
        "gene": "MAPT",
        "protein": (
            "MAPT — 17q21.31 AD — Microtubule-Associated-Protein-Tau-758aa — "
            "FTDP-17-FTD-Parkinsonism-Chromosome17 — "
            "Intronic-Splicing-4R:3R-Ratio-Increase — Exonic-Tau-Dysfunction-Direct — "
            "PSP-like-CBS-like-PiD-like-Phenotypes — "
            "Tau-PET-Diagnostic — H1-H2-Inversion-Haplotype-17q21"
        ),
        "alias": (
            "MAPT (microtubule-associated protein tau); OMIM gene 157140; "
            "Frontotemporal Dementia and Parkinsonism Linked to Chromosome 17 (FTDP-17) OMIM 600274; "
            "Pick's Disease (PiD) OMIM 172700. "
            "17q21.31; 758 aa (longest isoform 2N4R); ~79 kDa; autosomal dominant; "
            "~50+ pathogenic variants (intronic and exonic). "
            "FUNCTION: Tau is a microtubule-associated protein (MAP) expressed predominantly in neurons "
            "(axons); stabilises microtubules and promotes axonal transport. "
            "Six isoforms generated by alternative splicing of exons 2, 3, and 10: "
            "Exon 10 inclusion: 4-repeat tau (4R) — binds microtubules with higher affinity; "
            "Exon 10 exclusion: 3-repeat tau (3R) — normal adult brain contains ~equal 4R:3R; "
            "PATHOGENIC MECHANISM: "
            "(A) Intronic mutations (IVS10+3, IVS10+12, IVS10+16): disrupt RNA stem-loop at "
            "exon10/intron boundary → ↑ exon 10 inclusion → ↑ 4R:3R ratio → hyperphosphorylated 4R tau; "
            "(B) Exonic mutations (P301L, P301S, R406W, V337M): directly impair tau-microtubule binding "
            "→ tau detaches → aggregates (both 3R + 4R or mixed); "
            "P301L: most common MAPT pathogenic variant; "
            "P301S: aggressive; onset 40–50 y; used in transgenic models; "
            "R406W: later onset ~60 y; AD-like phenotype with amyloid and tau on PET; "
            "V337M: dementia-parkinsonism; 50s onset; familial clustering; "
            "CLINICAL FEATURES: "
            "Wide phenotypic spectrum: bvFTD > PSP-like (parkinsonism-predominant) > CBS > PiD-like; "
            "onset 45–65 y (mean ~55 y); "
            "TAU PET (flortaucipir, MK-6240, PI-2620) now available as diagnostic tool — "
            "patterns differ by 3R/4R ratio; "
            "MRI: variable frontal/temporal ± parietal atrophy; "
            "CSF: tau ↑ (total and phospho), Aβ42 normal (tau tauopathy not amyloid)."
        ),
        "locus": "17q21.31",
        "aa": 758,
        "kDa": 79,
        "omim_gene": "157140",
        "omim_disease": "600274",
        "inheritance": "AD; ~50 pathogenic variants (intronic + exonic)",
        "gene_class": "Tau Microtubule Stabiliser — 4R:3R Splicing — Neurofibrillary Tangle Formation",
        "key_alerts": [
            "MAPT-17q21-SAME-LOCUS-AS-GRN: MAPT and GRN are in the same chromosomal region (17q21.31) and both cause FTD — critically distinguish: GRN-FTD = TDP-43 type A pathology + plasma PGRN <100 ng/mL; MAPT-FTD = tau pathology + tau PET positive; amyloid PET NEGATIVE in both",
            "MAPT-P301L-MOST-COMMON-PSP-DDx: P301L (Pro301Leu) is the most common MAPT pathogenic variant and frequently presents as PSP-like syndrome with vertical gaze palsy + parkinsonism + dementia — MAPT sequencing in all young-onset PSP (<65y) or PSP with positive family history is mandatory",
            "MAPT-TAU-PET-POSITIVE-AMYLOID-PET-NEGATIVE: MAPT-FTDP-17 has positive tau PET and NEGATIVE amyloid PET — contrast with AD where both are positive; misclassification risk if only amyloid biomarkers are checked; tau PET pattern varies by mutation (3R vs 4R ratio)",
            "MAPT-NO-APPROVED-THERAPY: no disease-modifying therapy approved for MAPT-FTDP-17; anti-tau therapeutic trials ongoing (gosuranemab, semorinemab, bepranemab); do NOT give anti-amyloid therapy (lecanemab/donanemab) — no amyloid pathology, no benefit, ARIA risk without indication",
        ],
        "etiologies": {
            "P301L_Most_Common": {
                "pct": 30,
                "phenotype": "bvFTD or PSP-like; onset 45-60y; 4R tau predominant",
                "notes": "Impairs tau-microtubule binding; detached tau aggregates; transgenic PS19 mouse model",
            },
            "Intronic_4R_Splicing": {
                "pct": 25,
                "phenotype": "PSP-like or bvFTD; onset 50-65y; 4R tau overrepresentation",
                "notes": "IVS10+3, +12, +16 disrupt stem-loop → exon 10 inclusion ↑ → 4R:3R ratio ↑",
            },
            "V337M_Dementia_Parkinsonism": {
                "pct": 15,
                "phenotype": "FTD-parkinsonism; onset 40-60y; familial clustering USA/Europe",
                "notes": "Reduces microtubule binding; mixed 3R+4R tau aggregates",
            },
            "R406W_AD_like": {
                "pct": 15,
                "phenotype": "Later onset ~60y; AD-like amnestic; tau + amyloid both elevated on PET",
                "notes": "Unusual: both biomarkers positive; requires careful interpretation",
            },
            "P301S_Aggressive": {
                "pct": 15,
                "phenotype": "Aggressive; onset 40-50y; rapid progression; CBS or bvFTD",
                "notes": "Used in PS19 transgenic model; shorter survival than P301L",
            },
        },
        "stats": {
            "mean_onset_age": 54.8,
            "mean_dx_age": 57.5,
            "mean_dx_delay_months": 32,
            "prevalence_pct_of_familial_FTD": 15,
        },
        "dx_delay_distribution": {
            "0–12 mo": 10,
            "12–24 mo": 25,
            "24–36 mo": 35,
            "36+ mo": 30,
        },
        "patients": [],
    },
    # ── C9ORF72 — ALS-FTD (Repeat Expansion) ──────────────────────────────────
    {
        "gene": "C9ORF72",
        "protein": (
            "C9ORF72 — 9p21.2 AD — DENN-GEF-Autophagy-481aa — "
            "ALS-FTD-GGGGCC-Hexanucleotide-Repeat — "
            "Most-Common-Genetic-Cause-Both-ALS-AND-FTD — "
            "Repeat-Primed-PCR-MANDATORY-Standard-PCR-FAILS — "
            "40pct-Familial-ALS-25pct-Familial-FTD — "
            "Incomplete-Penetrance-50pct-By-65y — RNA-Foci-DPR-Proteins"
        ),
        "alias": (
            "C9ORF72 (chromosome 9 open reading frame 72); OMIM gene 614260; "
            "ALS-FTD-1 OMIM 105550; FTD OMIM 600274 (shared with MAPT). "
            "9p21.2; 481 aa; ~54 kDa; autosomal dominant; single pathogenic mechanism "
            "(GGGGCC hexanucleotide repeat expansion in intron 1). "
            "FUNCTION: C9ORF72 protein functions as a DENN-domain GTPase effector / autophagy regulator; "
            "normally maintains lysosomal-autophagy pathway; "
            "PATHOGENIC MECHANISM OF EXPANSION: three converging mechanisms: "
            "(1) HAPLOINSUFFICIENCY: expanded allele → ↓ C9ORF72 mRNA → reduced autophagy; "
            "(2) RNA FOCI: GGGGCC repeat RNA sequesters RNA-binding proteins (hnRNPs, etc.) "
            "→ widespread spliceopathy in affected neurons and glia; "
            "(3) DIPEPTIDE REPEAT PROTEINS (DPRs): repeat-associated non-ATG (RAN) translation "
            "produces poly-GA, poly-GR, poly-GP, poly-PR, poly-PA; "
            "poly-GR and poly-PR are most toxic (disrupt nucleocytoplasmic transport, "
            "stress granule dynamics, nucleolus); "
            "CLINICAL SPECTRUM: "
            "ALS (40% of familial ALS, 7% sporadic ALS); "
            "bvFTD (25% of familial FTD, 6% sporadic FTD); "
            "ALS-FTD (15% of C9ORF72 ALS families); "
            "psychosis (rare but recognised — first psychiatric presentation possible); "
            "PENETRANCE: incomplete — ~50% penetrance by age 65 y; ~90% by age 80 y; "
            "explains unaffected carriers in family pedigrees; "
            "DETECTION — REPEAT-PRIMED PCR MANDATORY: "
            "Standard PCR FAILS: G-quadruplex structures prevent amplification beyond ~30 repeats; "
            "pathogenic expansions = hundreds–thousands of repeats (most common ~800 repeats); "
            "standard PCR shows one normal-size band in fully affected patients; "
            "RP-PCR: primer within repeat + flanking primer → stutter ladder = positive; "
            "Southern blot: gold standard for size; "
            "long-read sequencing (Nanopore/PacBio): emerging, characterises length + methylation; "
            "CRITICAL: any report saying 'C9ORF72 NEGATIVE' based on standard PCR is INVALID; "
            "laboratory must confirm detection method."
        ),
        "locus": "9p21.2",
        "aa": 481,
        "kDa": 54,
        "omim_gene": "614260",
        "omim_disease": "105550",
        "inheritance": "AD; GGGGCC hexanucleotide repeat expansion; incomplete penetrance",
        "gene_class": "DENN-GEF Autophagy Regulator — Repeat Expansion — RNA Foci + DPR — ALS-FTD",
        "key_alerts": [
            "C9ORF72-REPEAT-PRIMED-PCR-MANDATORY-STANDARD-PCR-FAILS: C9ORF72 GGGGCC expansions form G-quadruplexes that standard PCR cannot traverse — standard PCR produces a FALSELY NORMAL result in affected patients; repeat-primed PCR (RP-PCR) is the ONLY valid first-line screening method; any negative C9ORF72 result from standard PCR must be repeated with RP-PCR",
            "C9ORF72-MOST-COMMON-GENETIC-CAUSE-ALS-AND-FTD: C9ORF72 accounts for ~40% of familial ALS AND ~25% of familial FTD — it must be the FIRST gene tested in any familial neurodegenerative dementia or motor neuron disease panel; testing C9ORF72 last is a diagnostic error",
            "C9ORF72-FTD-CAPACITY-ASSESSMENT: C9ORF72 carriers who develop FTD have profoundly impaired decision-making capacity (executive function, social cognition, insight) — advance care planning, power of attorney, and dementia care plans must be established EARLY before cognitive deterioration",
            "C9ORF72-INCOMPLETE-PENETRANCE-50pct-BY-65y: ~50% of C9ORF72 expansion carriers are unaffected at age 65 — unaffected family members should not be falsely reassured; penetrance ~90% by 80y; age-appropriate predictive testing discussion mandatory for at-risk relatives",
        ],
        "etiologies": {
            "ALS_Predominant": {
                "pct": 45,
                "phenotype": "Classical ALS; UMN + LMN signs ≥2 regions; onset 50-65y",
                "notes": "~40% familial ALS, ~7% sporadic ALS; riluzole + edaravone standard; tofersen NOT applicable",
            },
            "FTD_Predominant_bvFTD": {
                "pct": 30,
                "phenotype": "Behavioural variant FTD; executive dysfunction, disinhibition, apathy; onset 55-70y",
                "notes": "~25% familial FTD; TDP-43 type B inclusions; no approved FTD therapy",
            },
            "ALS_FTD_Combined": {
                "pct": 15,
                "phenotype": "Both motor neuron disease and frontotemporal syndrome; 55-65y",
                "notes": "Capacity assessment critical; DNR/ventilation decisions may be impaired early",
            },
            "Pure_FTD_Psychosis_Rare": {
                "pct": 10,
                "phenotype": "Psychosis without motor features; rare early presentation; misdiagnosed psychiatric",
                "notes": "C9ORF72 should be in differential for atypical early-onset psychosis with family history",
            },
        },
        "stats": {
            "mean_onset_age": 57.5,
            "mean_dx_age": 59.5,
            "mean_dx_delay_months": 24,
            "prevalence_pct_of_familial_FTD_ALS": 33,
        },
        "dx_delay_distribution": {
            "0–12 mo": 25,
            "12–24 mo": 35,
            "24–36 mo": 25,
            "36+ mo": 15,
        },
        "patients": [],
    },
    # ── PRNP — Prion Diseases (GSS / FFI / Familial CJD) ──────────────────────
    {
        "gene": "PRNP",
        "protein": (
            "PRNP — 20p13 AD/AR — Prion-Protein-253aa — "
            "GSS-Gerstmann-Straussler-Scheinker-P102L-Most-Common — "
            "FFI-Fatal-Familial-Insomnia-D178N-M129V-Same-Allele — "
            "Familial-CJD-D178N-M129M-Same-Allele — "
            "Universally-Fatal-NO-Treatment — "
            "NO-Organ-Donation-ABSOLUTE-Biosafety-Level-2-Plus"
        ),
        "alias": (
            "PRNP (prion protein); OMIM gene 176640; "
            "Gerstmann-Sträussler-Scheinker syndrome (GSS) OMIM 137440; "
            "Fatal Familial Insomnia (FFI) OMIM 600072; "
            "Familial Creutzfeldt-Jakob Disease (fCJD) OMIM 123400. "
            "20p13; 253 aa; ~35 kDa; autosomal dominant (GSS, FFI, fCJD); "
            "codon 129 polymorphism (Met/Val) is a critical disease modifier; "
            "~30 pathogenic PRNP variants described. "
            "FUNCTION: Cellular prion protein (PrPC) is a GPI-anchored glycoprotein expressed ubiquitously "
            "(highest in neurons). Normal function incompletely understood: copper binding, "
            "neuroprotection, cell signalling, synaptic plasticity. "
            "PATHOGENIC MECHANISM: PRNP mutations and/or codon 129 modulate the propensity of PrPC "
            "to misfold into the pathogenic isoform PrPSc (protease-resistant, beta-sheet-rich). "
            "PrPSc acts as a template — it converts normal PrPC to PrPSc: exponential self-propagating "
            "'prion replication.' PrPSc accumulates → neuronal vacuolation (spongiform change) → "
            "gliosis → neuronal death. Infectious: PrPSc propagates by contact. "
            "GENETIC FORMS: "
            "(1) GSS (Gerstmann-Sträussler-Scheinker): P102L (most common) — onset 40–60 y; "
            "slowly progressive cerebellar ataxia PRECEDING dementia (contrast with CJD where dementia first); "
            "survival 2–10 y; histology: multicentric PrP amyloid plaques; "
            "also A117V, F198S, Q217R; "
            "(2) FFI (Fatal Familial Insomnia): D178N-129V (asparagine → aspartate at 178 on SAME allele "
            "as valine at codon 129); progressive insomnia → autonomic failure → motor signs → dementia → death; "
            "thalamic degeneration on MRI/PET; survival 7–36 months; rare (~40 families worldwide); "
            "(3) fCJD: D178N-129M (same codon 178 variant as FFI but different codon 129 on SAME allele "
            "→ completely different disease); rapidly progressive dementia; EEG: periodic sharp waves; "
            "CSF: 14-3-3 positive, rt-QuIC positive; survival 6–18 months; "
            "E200K: most common worldwide familial CJD variant; Libyan Jews/Slovak/Chilean founders; "
            "BIOSAFETY AND INFECTION CONTROL: "
            "PrPSc resists standard sterilisation (autoclaving, formalin); "
            "neurosurgical instruments must be destroyed or subjected to extended decontamination; "
            "ORGAN DONATION: ABSOLUTELY CONTRAINDICATED in known/suspected prion disease carriers; "
            "corneal transplant from undiagnosed CJD has transmitted disease; "
            "MANDATORY REPORTING: CJD is a notifiable disease in most countries."
        ),
        "locus": "20p13",
        "aa": 253,
        "kDa": 35,
        "omim_gene": "176640",
        "omim_disease": "137440",
        "inheritance": "AD (GSS, FFI, fCJD); rare AR; codon 129 Met/Val polymorphism modifies phenotype",
        "gene_class": "Prion Protein — Conformational Disease — PrPSc Propagation — Universally Fatal",
        "key_alerts": [
            "PRNP-NO-ORGAN-DONATION-ABSOLUTE-CONTRAINDICATION: any patient with confirmed or suspected prion disease (including asymptomatic PRNP variant carriers) must be flagged in medical records as ineligible for organ/tissue donation — corneal transplant from an undiagnosed CJD patient has previously transmitted disease; inform patient and document prominently",
            "PRNP-BIOSAFETY-NEUROSURGICAL-INSTRUMENTS: PrPSc resists standard autoclaving and formalin fixation — neurosurgical instruments used on prion patients must be quarantined, destroyed, or subjected to extended decontamination protocol (NaOH + autoclaving); biopsy tissue requires biosafety level 2+ precautions; inform pathologist and surgical team BEFORE any procedure",
            "PRNP-D178N-CODON-129-DETERMINES-DISEASE: D178N (Asp178Asn) causes EITHER FFI OR familial CJD depending on the codon 129 polymorphism on THE SAME allele (cis-acting): D178N + Val129 (same allele) = FFI; D178N + Met129 (same allele) = familial CJD — COMPLETELY DIFFERENT DISEASES from one amino acid context; laboratory must report codon 129 haplotype with D178N result",
            "PRNP-GSS-CEREBELLAR-ATAXIA-FIRST-DEMENTIA-SECOND: GSS (P102L most common) causes cerebellar ataxia BEFORE dementia — misdiagnosed as spinocerebellar ataxia for years; when ataxia precedes dementia in 40s–60s with positive family history, PRNP sequencing is mandatory",
        ],
        "etiologies": {
            "GSS_P102L_Cerebellar": {
                "pct": 40,
                "phenotype": "Ataxia-predominant then dementia; onset 40-60y; survival 2-10y",
                "notes": "Most common GSS mutation; multicentric PrP amyloid plaques at autopsy; slow progression",
            },
            "FFI_D178N_V129": {
                "pct": 20,
                "phenotype": "Insomnia → autonomic failure → motor/dementia; rare ~40 families; survival 7-36mo",
                "notes": "D178N on Val129 haplotype; thalamic degeneration; EEG: no periodic complexes (unlike CJD)",
            },
            "fCJD_E200K": {
                "pct": 25,
                "phenotype": "Rapidly progressive dementia; EEG periodic waves; 14-3-3 CSF positive; survival 6-18mo",
                "notes": "Most common fCJD worldwide; Libyan Jewish/Slovak/Chilean founder effect; PRNP E200K",
            },
            "fCJD_D178N_M129": {
                "pct": 10,
                "phenotype": "Rapidly progressive dementia (not insomnia); same gene as FFI different haplotype",
                "notes": "D178N on Met129 allele; classic CJD neuropathology; MRI DWI cortical ribboning",
            },
            "Other_GSS_Variants": {
                "pct": 5,
                "phenotype": "A117V, F198S, Q217R — GSS variants with varying cerebellar vs dementia preponderance",
                "notes": "Rare; PrP amyloid plaques; longer survival than CJD; PRNP panel needed for distinction",
            },
        },
        "stats": {
            "mean_onset_age": 52.0,
            "mean_dx_age": 53.5,
            "mean_dx_delay_months": 18,
            "survival_months_mean": 28,
        },
        "dx_delay_distribution": {
            "0–12 mo": 35,
            "12–24 mo": 40,
            "24–36 mo": 20,
            "36+ mo": 5,
        },
        "patients": [],
    },
    # ── TREM2 — Nasu-Hakola Disease (AR) / AD Risk Modifier (AD) ──────────────
    {
        "gene": "TREM2",
        "protein": (
            "TREM2 — 6p21.1 AR-Nasu-Hakola-Disease / AD-Risk-Modifier — "
            "Microglial-DAP12-Receptor-230aa — "
            "R47H-Increases-AD-Risk-2-4x-Like-APOE4 — "
            "Biallelic-LOF-Nasu-Hakola-PLOSL — "
            "AL001-Anti-TREM2-Agonist-Phase2-INVOKE2 — "
            "Microglial-Lysosomal-Biology-Lipid-Sensing"
        ),
        "alias": (
            "TREM2 (triggering receptor expressed on myeloid cells 2); OMIM gene 605086; "
            "Nasu-Hakola disease / PLOSL OMIM 221770; "
            "Alzheimer's Disease risk modifier (heterozygous R47H). "
            "6p21.1; 230 aa; ~26 kDa; type I transmembrane protein; "
            "expressed on microglia, osteoclasts, macrophages, dendritic cells. "
            "FUNCTION: TREM2 is a pattern recognition receptor on microglia that signals through DAP12 "
            "(TYROBP) co-receptor → activates downstream PI3K, ERK, PLCγ pathways → "
            "(1) microglial survival, proliferation, and activation; "
            "(2) phagocytosis of dead neurons, myelin debris, lipid particles, amyloid plaques; "
            "(3) lysosomal function and lipid metabolism; "
            "(4) transition to disease-associated microglia (DAM) phenotype around amyloid plaques; "
            "TREM2 on microglia — the 'glial guardian': senses lipid-rich debris and amyloid → "
            "drives microglial clustering around plaques (protective, limiting spread). "
            "HETEROZYGOUS R47H VARIANT: "
            "Common population variant (~0.3% Europeans); "
            "R47H reduces TREM2 ligand binding affinity (lipid-sensing impaired); "
            "heterozygous carriers: AD risk ~2–4× higher (equivalent to ~1 APOE4 allele effect); "
            "single-gene AD testing panels should include TREM2 R47H; "
            "NOT sufficient alone for familial EOAD — consider as risk modifier; "
            "BIALLELIC LOF — NASU-HAKOLA DISEASE (PLOSL): "
            "Polycystic lipomembranous osteodysplasia with sclerosing leukoencephalopathy; "
            "rare (few hundred cases worldwide); "
            "bone component: lipomembranous cysts in small bones → pathological fractures (teens–20s); "
            "brain component: frontal leukoencephalopathy → frontotemporal dementia (30s–40s); "
            "bone biopsy: cysts lined by lipomembranous material; "
            "NMJ is spared (contrast with TREM2-pathway microglial diseases); "
            "radiology: diffuse white matter signal abnormality frontal > posterior; "
            "THERAPEUTIC APPROACHES: "
            "AL001 (latozinemab, Alector — different from GRN latozinemab; this is anti-TREM2 agonist): "
            "Phase 2 INVOKE-2 trial in early AD; promotes microglial activation; "
            "mCSF (macrophage colony-stimulating factor) — preclinical microglial support; "
            "APOE/TREM2 interaction: risk is multiplicative (APOE4 + R47H); highest-risk group."
        ),
        "locus": "6p21.1",
        "aa": 230,
        "kDa": 26,
        "omim_gene": "605086",
        "omim_disease": "221770",
        "inheritance": "AR biallelic LOF → Nasu-Hakola disease; AD heterozygous R47H → AD risk modifier",
        "gene_class": "Microglial DAP12-Receptor — Lipid Sensing — DAM Phenotype — AD Risk Modifier",
        "key_alerts": [
            "TREM2-R47H-AD-RISK-2-4x-NOT-MONOGENIC-CAUSE: heterozygous TREM2 R47H increases AD risk ~2–4× (similar magnitude to one APOE4 allele) — it is a RISK MODIFIER not a monogenic EOAD cause; should be included in comprehensive dementia genetic risk panels; do not use alone to diagnose familial AD",
            "TREM2-BIALLELIC-NASU-HAKOLA-BONE-BRAIN-COMBINATION: homozygous or compound heterozygous TREM2 LOF causes Nasu-Hakola disease (PLOSL) — pathological bone fractures in teens + frontal leukoencephalopathy + FTD in 30s; always investigate BOTH bone AND brain in young patients with either component",
            "TREM2-AL001-INVOKE-2-CLINICAL-TRIAL: AL001 (anti-TREM2 agonist antibody) is in Phase 2 INVOKE-2 trial for early AD — promotes microglial activation around amyloid plaques; TREM2 pathway is a major therapeutic frontier; refer TREM2-associated dementia patients to specialist centres for trial eligibility assessment",
            "TREM2-APOE4-COMBINED-RISK: TREM2 R47H + APOE4 heterozygosity multiply dementia risk; combined genetic risk assessment should include both genes; counselling must address risk probability, uncertainty, and non-deterministic nature of these variants",
        ],
        "etiologies": {
            "R47H_AD_Risk_Modifier": {
                "pct": 60,
                "phenotype": "Late-onset AD (65-80y); risk ~2-4x baseline; amyloid and tau biomarkers positive",
                "notes": "Most common hereditary TREM2-associated presentation; not monogenic EOAD",
            },
            "Nasu_Hakola_PLOSL_Biallelic": {
                "pct": 25,
                "phenotype": "Bone fractures teens-20s + FTD 30s-40s; rare; global distribution",
                "notes": "Homozygous or compound heterozygous LOF; bone biopsy diagnostic; no treatment",
            },
            "Other_R47H_Independent_Variants": {
                "pct": 15,
                "phenotype": "T96K, R62H — reduced TREM2 function; AD risk elevation confirmed in some",
                "notes": "Less characterised than R47H; significance varies by population",
            },
        },
        "stats": {
            "mean_onset_age": 68.0,
            "mean_dx_age": 70.5,
            "mean_dx_delay_months": 30,
        },
        "dx_delay_distribution": {
            "0–12 mo": 15,
            "12–24 mo": 30,
            "24–36 mo": 35,
            "36+ mo": 20,
        },
        "patients": [],
    },
]


def _generate_patients():
    for idx, gene_data in enumerate(DEMENTIA_GENES):
        gene = gene_data["gene"]
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        patients = []
        for i in range(40):
            if gene == "PSEN1":
                variant = rng.choice([
                    "E280A", "M139T", "L286P", "G384A", "A431E",
                    "C410Y", "L435F", "G206A", "A79V", "other",
                ])
                onset_age = rng.randint(35, 58)
                age_at_dx = onset_age + rng.randint(1, 3)
                dx_delay = (age_at_dx - onset_age) * 12 + rng.randint(-6, 12)
                pca_phenotype = variant in ("C410Y", "A79V")
                spastic_paraparesis = variant in ("L435F", "C410Y", "A431E")
                aria_risk_high = True
                amyloid_pet_positive = True
                csf_abeta42_low = True
                patients.append({
                    "patient_id": f"PSEN1-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": max(6, dx_delay),
                    "variant": variant,
                    "pca_phenotype": pca_phenotype,
                    "spastic_paraparesis": spastic_paraparesis,
                    "aria_risk_high": aria_risk_high,
                    "amyloid_pet_positive": amyloid_pet_positive,
                    "csf_abeta42_low": csf_abeta42_low,
                    "gene": gene, "seed": seed,
                })

            elif gene == "PSEN2":
                variant = rng.choice(["N141I", "M239V", "T122P", "I213T", "V393M", "other"])
                onset_age = rng.randint(45, 72)
                age_at_dx = onset_age + rng.randint(1, 3)
                dx_delay = (age_at_dx - onset_age) * 12 + rng.randint(0, 18)
                volga_german = variant == "N141I"
                penetrance_95 = True
                aria_risk_elevated = True
                patients.append({
                    "patient_id": f"PSEN2-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": max(6, dx_delay),
                    "variant": variant,
                    "volga_german_founder": volga_german,
                    "penetrance_95": penetrance_95,
                    "aria_risk_elevated": aria_risk_elevated,
                    "gene": gene, "seed": seed,
                })

            elif gene == "APP":
                variant = rng.choice([
                    "Duplication", "V717I_London", "K670N_M671L_Swedish",
                    "E693Q_Dutch_CAA", "V717L_Indiana", "other",
                ])
                onset_age = rng.randint(42, 65)
                age_at_dx = onset_age + rng.randint(1, 4)
                dx_delay = (age_at_dx - onset_age) * 12 + rng.randint(0, 24)
                aria_risk = "HIGHEST" if variant in ("Duplication", "E693Q_Dutch_CAA") else "ELEVATED"
                caa_dominant = variant == "E693Q_Dutch_CAA"
                anticoagulation_risk = caa_dominant
                patients.append({
                    "patient_id": f"APP-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": max(6, dx_delay),
                    "variant": variant,
                    "aria_risk": aria_risk,
                    "caa_dominant": caa_dominant,
                    "anticoagulation_dangerous": anticoagulation_risk,
                    "gene": gene, "seed": seed,
                })

            elif gene == "GRN":
                variant = rng.choice([
                    "nonsense_NMD", "frameshift_NMD", "splice_site_LOF",
                    "missense_LOF_confirmed", "other_LOF",
                ])
                onset_age = rng.randint(50, 72)
                age_at_dx = onset_age + rng.randint(2, 5)
                dx_delay = (age_at_dx - onset_age) * 12 + rng.randint(6, 30)
                plasma_pgrn_ngml = rng.randint(40, 95)  # <100 = diagnostic
                tdp43_type_a = True
                parkinsonism = rng.random() > 0.85
                phenotype = rng.choice(["bvFTD", "PNFA", "CBS_like"])
                patients.append({
                    "patient_id": f"GRN-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": max(12, dx_delay),
                    "variant": variant,
                    "plasma_pgrn_ngml": plasma_pgrn_ngml,
                    "tdp43_type_a": tdp43_type_a,
                    "parkinsonism": parkinsonism,
                    "phenotype": phenotype,
                    "gene": gene, "seed": seed,
                })

            elif gene == "MAPT":
                variant = rng.choice([
                    "P301L", "intronic_4R_splicing", "V337M",
                    "R406W", "P301S_aggressive", "other",
                ])
                onset_age = rng.randint(42, 68)
                age_at_dx = onset_age + rng.randint(2, 5)
                dx_delay = (age_at_dx - onset_age) * 12 + rng.randint(12, 36)
                tau_pet_positive = True
                amyloid_pet_negative = variant != "R406W"
                phenotype = rng.choice(["bvFTD", "PSP_like", "CBS_like", "PiD_like"])
                anti_tau_trial_eligible = rng.random() > 0.4
                patients.append({
                    "patient_id": f"MAPT-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": max(12, dx_delay),
                    "variant": variant,
                    "tau_pet_positive": tau_pet_positive,
                    "amyloid_pet_negative": amyloid_pet_negative,
                    "phenotype": phenotype,
                    "anti_tau_trial_eligible": anti_tau_trial_eligible,
                    "gene": gene, "seed": seed,
                })

            elif gene == "C9ORF72":
                phenotype = rng.choice(["ALS", "bvFTD", "ALS_FTD", "other_FTD"])
                onset_age = rng.randint(48, 68)
                age_at_dx = onset_age + rng.randint(1, 3)
                dx_delay = (age_at_dx - onset_age) * 12 + rng.randint(3, 24)
                rp_pcr_required = True
                standard_pcr_fails = True
                penetrance_incomplete = True
                ftd_component = phenotype in ("bvFTD", "ALS_FTD", "other_FTD")
                capacity_assessment_needed = ftd_component
                patients.append({
                    "patient_id": f"C9ORF72-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": max(6, dx_delay),
                    "phenotype": phenotype,
                    "rp_pcr_required": rp_pcr_required,
                    "standard_pcr_fails": standard_pcr_fails,
                    "penetrance_incomplete": penetrance_incomplete,
                    "capacity_assessment_needed": capacity_assessment_needed,
                    "gene": gene, "seed": seed,
                })

            elif gene == "PRNP":
                disease_type = rng.choice(["GSS_P102L", "fCJD_E200K", "FFI_D178N_V129", "fCJD_D178N_M129", "other_GSS"])
                onset_age = rng.randint(38, 65)
                age_at_dx = onset_age + rng.randint(0, 2)
                dx_delay = max(6, (age_at_dx - onset_age) * 12 + rng.randint(0, 18))
                gss_phenotype = disease_type in ("GSS_P102L", "other_GSS")
                universally_fatal = True
                organ_donation_contraindicated = True
                biosafety_required = True
                survival_months = (
                    rng.randint(24, 120) if gss_phenotype
                    else rng.randint(6, 18)
                )
                patients.append({
                    "patient_id": f"PRNP-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "disease_type": disease_type,
                    "gss_phenotype": gss_phenotype,
                    "universally_fatal": universally_fatal,
                    "organ_donation_contraindicated": organ_donation_contraindicated,
                    "biosafety_required": biosafety_required,
                    "survival_months": survival_months,
                    "gene": gene, "seed": seed,
                })

            else:  # TREM2
                variant = rng.choice(["R47H_heterozygous", "biallelic_LOF_Nasu-Hakola", "T96K", "other"])
                nasu_hakola = variant == "biallelic_LOF_Nasu-Hakola"
                onset_age = rng.randint(33, 55) if nasu_hakola else rng.randint(60, 80)
                age_at_dx = onset_age + rng.randint(1, 5)
                dx_delay = (age_at_dx - onset_age) * 12 + rng.randint(12, 36)
                bone_fractures = nasu_hakola
                ftd_component = nasu_hakola
                apoe4_combined = not nasu_hakola and rng.random() > 0.65
                patients.append({
                    "patient_id": f"TREM2-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": max(12, dx_delay),
                    "variant": variant,
                    "nasu_hakola": nasu_hakola,
                    "bone_fractures_present": bone_fractures,
                    "ftd_component": ftd_component,
                    "apoe4_combined_risk": apoe4_combined,
                    "gene": gene, "seed": seed,
                })

        gene_data["patients"] = patients


_generate_patients()


def overview():
    all_delays = [
        p.get("dx_delay_months", 0)
        for g in DEMENTIA_GENES for p in g["patients"]
    ]
    all_ages = [
        p.get("onset_age", 55)
        for g in DEMENTIA_GENES for p in g["patients"]
    ]
    genes = []
    for idx, g in enumerate(DEMENTIA_GENES):
        delays = [p.get("dx_delay_months", 0) for p in g["patients"]]
        ages = [p.get("onset_age", 55) for p in g["patients"]]
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
        "atlas": "Hereditary-Dementia-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Dementia Atlas — "
            "PSEN1 / PSEN2 / APP / GRN / MAPT / C9ORF72 / PRNP / TREM2 — "
            "320 Patients (8×40, Seeds 1742–1749)"
        ),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": sum(len(g["patients"]) for g in DEMENTIA_GENES),
        "aggregate_stats": {
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "genes_covered": len(DEMENTIA_GENES),
            "patients_per_gene": 40,
        },
        "genes": genes,
        "top_alerts": [
            "PSEN1-VIRTUALLY-100pct-PENETRANCE-LECANEMAB-ARIA-RISK: PSEN1 mutations carry near-100% lifetime EOAD penetrance AND significantly elevated ARIA risk with lecanemab/donanemab — anti-amyloid immunotherapy requires specialist monthly MRI surveillance in all PSEN1/PSEN2/APP variant carriers",
            "C9ORF72-REPEAT-PRIMED-PCR-MANDATORY-STANDARD-PCR-FAILS: C9ORF72 is the MOST COMMON familial dementia/ALS gene (~40% familial ALS, ~25% familial FTD) — standard PCR FAILS to detect expansion; repeat-primed PCR is the ONLY valid test; test C9ORF72 FIRST in any familial neurodegenerative panel",
            "GRN-PLASMA-PROGRANULIN-<100-DIAGNOSTIC: plasma progranulin <100 ng/mL is PATHOGNOMONIC for GRN pathogenic variant — this blood test must precede or accompany GRN genetic testing in all FTD evaluations; latozinemab INFRONT-3 trial open for GRN-FTD patients",
            "PRNP-NO-ORGAN-DONATION-BIOSAFETY-MANDATORY: prion disease patients CANNOT donate organs or tissue; neurosurgical instruments require extended decontamination; D178N variant phenotype (FFI vs CJD) is determined by cis-acting codon 129 polymorphism on SAME allele — laboratory MUST report both",
            "MAPT-17q21-GRN-17q21-DISTINGUISH-CRITICALLY: MAPT and GRN are both at 17q21.31 — tau PET + amyloid PET negative + plasma PGRN normal = MAPT; tau PET + plasma PGRN <100 = GRN; NEVER give anti-amyloid therapy (lecanemab/donanemab) in MAPT-FTD — no amyloid, no indication, ARIA risk",
            "APP-E693Q-DUTCH-CAA-ANTICOAGULATION-DANGEROUS: APP E693Q (Dutch-type) causes cerebral amyloid angiopathy with lobar haemorrhages — anticoagulation for AF is dangerous; each case requires neurology-cardiology-genetics joint decision; aspirin may be preferred",
            "PSEN2-PENETRANCE-95pct-COUNSELLING-NUANCED: PSEN2 penetrance is ~95% by age 80 — some carriers remain unaffected; counselling must explain both the high (but not absolute) risk and the later average onset (mean ~55y) compared to PSEN1; predictive testing requires full protocol",
            "TREM2-R47H-RISK-MODIFIER-NOT-MONOGENIC: TREM2 R47H increases AD risk ~2–4× (similar to APOE4 heterozygosity) — it is a RISK MODIFIER not a monogenic cause; combined TREM2 R47H + APOE4 confers multiplicative risk; AL001 (TREM2 agonist antibody) in Phase 2 INVOKE-2 trial",
        ],
    }


def breakdown():
    result = []
    for idx, g in enumerate(DEMENTIA_GENES):
        delays = [p.get("dx_delay_months", 0) for p in g["patients"]]
        ages = [p.get("onset_age", 55) for p in g["patients"]]
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


def definitions():
    return {
        "concepts": {
            "Hereditary Dementia — The Three-Pathway Framework": (
                "Hereditary dementia genes cluster into three overlapping pathogenic pathways: "
                "(1) AMYLOID CASCADE: PSEN1, PSEN2, APP — all ↑ Aβ42 production or Aβ42:Aβ40 ratio; "
                "amyloid PET positive; CSF Aβ42 ↓, tau ↑; target of lecanemab/donanemab; "
                "ARIA risk highest in this group; "
                "(2) TAU TAUOPATHY: MAPT — direct tau dysfunction; tau PET positive; "
                "amyloid PET NEGATIVE (except R406W); TDP-43 NEGATIVE; "
                "multiple phenotypes (bvFTD, PSP-like, CBS, PiD); "
                "no approved disease-modifying therapy; anti-tau trials ongoing; "
                "(3) TDP-43 / LYSOSOMAL / MICROGLIAL: GRN, C9ORF72, TREM2 — "
                "lysosomal dysfunction → TDP-43 pathology (GRN type A; C9ORF72 type B); "
                "microglial lipid-sensing failure (TREM2); "
                "C9ORF72 additionally produces toxic DPR proteins from repeat expansion; "
                "DIAGNOSTIC IMPLICATION: biomarker profile reveals the pathway — "
                "amyloid+tau biomarkers: pathway 1 (amyloid cascade); "
                "tau PET+ amyloid PET-: pathway 2 (MAPT tauopathy); "
                "plasma PGRN <100: pathway 3 GRN; RP-PCR positive: pathway 3 C9ORF72; "
                "PRNP fits separately as conformational prion disease — unique epidemiology. "
                "THERAPEUTIC TARGETING: "
                "Pathway 1 → anti-amyloid immunotherapy (with ARIA monitoring); "
                "Pathway 2 → anti-tau trials (gosuranemab, semorinemab, bepranemab in progress); "
                "Pathway 3 → latozinemab (GRN, raises progranulin); AL001 (TREM2 agonist); "
                "C9ORF72 ASO trials (afinersen — reduces DPR poly-GP) ongoing."
            ),
            "ARIA — Amyloid-Related Imaging Abnormalities in Genetic AD": (
                "ARIA (amyloid-related imaging abnormalities) are the main safety concern with "
                "anti-amyloid immunotherapies (lecanemab, donanemab). "
                "ARIA-E (oedema/effusions on FLAIR MRI); ARIA-H (haemosiderin deposits on SWI MRI). "
                "RISK STRATIFICATION FOR GENETIC VARIANTS: "
                "HIGHEST RISK: APP duplication (~50% ARIA), APOE4/4 homozygotes; "
                "HIGH RISK: PSEN1, PSEN2, APP missense variants (~2–3× sporadic risk), APOE4/3; "
                "MODERATE RISK: APOE4/4-negative sporadic AD, TREM2 R47H; "
                "LOWER RISK: APOE3/3 sporadic AD, APOE2 carriers; "
                "MONITORING PROTOCOL: monthly brain MRI for 1 year in high-risk cases; "
                "any new neurological symptoms during treatment → IMMEDIATE brain MRI; "
                "MANAGEMENT: hold immunotherapy if ARIA detected; resume after resolution on MRI; "
                "Grade 3–4 ARIA: PERMANENTLY discontinue lecanemab/donanemab; "
                "EXCLUSIONS MANDATED: prior intracerebral haemorrhage; anticoagulated patients; "
                "E693Q Dutch APP (CAA-dominant) — absolute contraindication to anti-amyloid therapy; "
                "SPECIALIST CENTRE REQUIREMENT: PSEN1/PSEN2/APP variant carriers should only receive "
                "anti-amyloid therapy at centres with monthly MRI capacity and neuroradiology expertise."
            ),
            "GRN Progranulin Biology and the Sortilin Pathway": (
                "Progranulin (PGRN) is secreted by neurons and microglia; it functions as a "
                "lysosomal homeostasis factor critical for TDP-43 nuclear retention and clearance. "
                "SORTILIN (SORT1) receptor: PGRN is endocytosed via sortilin → lysosomal trafficking; "
                "excessive SORT1-mediated endocytosis reduces circulating PGRN levels; "
                "LATOZINEMAB (AL001): anti-sortilin antibody; blocks SORT1 on hepatocytes and neurons "
                "→ reduces PGRN clearance → raises plasma/CSF PGRN levels; "
                "Phase 3 INFRONT-3: symptomatic GRN-FTD patients; endpoint NfL, clinical scales; "
                "mCSF + other microglial support agents in preclinical testing. "
                "PLASMA PGRN DIAGNOSTIC PROTOCOL: "
                "Draw: EDTA plasma; single freeze-thaw acceptable; validated commercial ELISA only; "
                "<100 ng/mL: pathognomonic GRN LOF; 100–180 ng/mL: borderline → GRN sequencing; "
                ">180 ng/mL: GRN haploinsufficiency essentially excluded; "
                "False normal: inflammatory states, PGRN precursor forms — check validated kit; "
                "False low: severe systemic illness; "
                "GENETIC TESTING: confirms specific variant after abnormal plasma PGRN; "
                "variant classification (pathogenic vs VUS) uses plasma PGRN as functional evidence."
            ),
            "C9ORF72 Repeat Expansion and Detection — Why Standard PCR Fails": (
                "The GGGGCC hexanucleotide repeat in intron 1 of C9ORF72 forms stable G-quadruplex "
                "structures and hairpin secondary structures that standard PCR polymerases cannot "
                "efficiently traverse beyond ~30 repeats. Pathogenic expansions are typically "
                "hundreds to thousands of repeats (most ~800 in ALS, variable in FTD). "
                "Standard PCR: products a single normal-size band even in fully affected patients "
                "— FALSELY NORMAL; the test is INVALID for C9ORF72 exclusion. "
                "REPEAT-PRIMED PCR (RP-PCR): primer anchored within the repeat + flanking primer "
                "→ stutter ladder extending into the expanded repeat; "
                "positive stutter pattern = expansion present (not quantified); "
                "first-line screening test for C9ORF72. "
                "SOUTHERN BLOT: hybridisation-based; gold standard for expansion confirmation and "
                "approximate sizing; labour-intensive; 2–4 week turnaround; "
                "confirms RP-PCR positive and provides size estimate. "
                "LONG-READ SEQUENCING (Oxford Nanopore / PacBio): emerging gold standard; "
                "simultaneously characterises repeat length, methylation status (epigenetic modifier), "
                "and flanking sequence; increasingly first-line in specialist labs. "
                "LABORATORY REPORTING STANDARD: any report stating 'C9ORF72 NEGATIVE' must specify "
                "detection method; standard PCR negative = INVALID for exclusion; "
                "repeat-primed PCR negative = valid exclusion (99%+ sensitivity). "
                "PENETRANCE: ~50% by age 65 y; ~90% by age 80 y — incomplete; unaffected carriers exist."
            ),
            "Prion Disease Biosafety and Codon 129 Genotype Principle": (
                "Prion diseases require special biosafety considerations unique among hereditary dementias. "
                "PrPSc (pathogenic prion isoform) resists: autoclaving at 121°C, formalin fixation, "
                "UV irradiation, ethanol — standard laboratory decontamination protocols are INSUFFICIENT. "
                "DECONTAMINATION: NaOH 1N 60 min + autoclaving 134°C 60 min for reusable instruments; "
                "or single-use instruments (preferred for neurosurgical); "
                "TISSUE HANDLING: fresh/fixed brain tissue requires BSL2+ containment; "
                "neuropathology must be informed BEFORE specimen arrives; "
                "ORGAN DONATION: ABSOLUTE CONTRAINDICATION — corneal, dura, and growth hormone "
                "transplants from undiagnosed prion patients have caused iatrogenic transmission; "
                "flag in medical records, blood bank, tissue bank, and hospital ID systems; "
                "CODON 129 GENOTYPE PRINCIPLE (D178N EXAMPLE): "
                "Codon 129 encodes methionine (M) or valine (V): "
                "D178N on the SAME allele as VAL-129 → FATAL FAMILIAL INSOMNIA (FFI); "
                "D178N on the SAME allele as MET-129 → FAMILIAL CREUTZFELDT-JAKOB DISEASE (fCJD); "
                "This cis-acting modifier completely changes the phenotype from a sleep/autonomic disorder "
                "(FFI) to a rapidly progressive dementia (fCJD) — the SAME mutation, DIFFERENT diseases; "
                "Laboratories MUST report PRNP codon 129 genotype alongside any PRNP pathogenic variant; "
                "failure to phase D178N with codon 129 is a diagnostic error; "
                "codon 129 MM homozygotes: highest CJD risk (sporadic and variant CJD)."
            ),
        },
        "pharmacological_distinctions": [
            "Lecanemab (Leqembi, Biogen/Eisai) — ALL AMYLOID-CONFIRMED EARLY AD; anti-Aβ protofibrils IgG1; FDA accelerated approval 2023, full approval 2023; biweekly IV infusion; ARIA risk significantly higher in PSEN1/PSEN2/APP carriers and APOE4/4 → monthly MRI first year; NOT indicated in MAPT/GRN/PRNP/C9ORF72-FTD (no amyloid pathology, no benefit, ARIA risk)",
            "Donanemab (Kisunla, Eli Lilly) — ALL AMYLOID-CONFIRMED EARLY AD; anti-Aβ plaque IgG1; FDA approval 2024; monthly IV until plaque clearance; ARIA risk elevated in PSEN1/PSEN2/APP; same ARIA monitoring requirements; tapering approach unique (treatment stops when amyloid cleared on PET)",
            "Donepezil (Aricept) — ALL AD STAGES (approved); acetylcholinesterase inhibitor; once daily; GI tolerability issues (nausea, diarrhoea); HR bradycardia; QTc check if cardiac disease; same indication regardless of genetic variant; NOT disease-modifying",
            "Rivastigmine (Exelon) — MILD-MODERATE AD AND PD-DEMENTIA; AChEI + butyrylcholinesterase inhibitor; patch formulation preferred (better GI tolerance); especially used in Lewy body dementia; indicated in PD-dementia (unlike donepezil which is not specifically approved for PDD)",
            "Memantine (Namenda) — MODERATE-SEVERE AD; NMDA receptor antagonist; blocks excessive glutamate excitotoxicity; can combine with AChEI; modest benefit on ADAS-cog and ADL scales; contraindicated in severe renal failure",
            "Latozinemab (AL001, Alector) — GRN-FTD ONLY; anti-sortilin antibody; raises progranulin by blocking SORT1-mediated clearance; Phase 3 INFRONT-3 ongoing (symptomatic GRN-FTD); NOT approved; NOT applicable to MAPT/C9ORF72/PSEN1 dementia; enrol eligible patients in trial",
            "AL001 (Alector, anti-TREM2 agonist antibody) — EARLY AD with TREM2 biology; Phase 2 INVOKE-2; promotes microglial DAM phenotype around amyloid plaques; NOT same compound as latozinemab; NOT approved; distinct from anti-amyloid therapy",
            "Anti-tau therapies (gosuranemab, semorinemab, bepranemab, E2814) — MAPT-FTD and PROGRESSIVE SUPRANUCLEAR PALSY in trials; Phase 2 in tauopathies; targeting extracellular tau, tau aggregation, or tau seeding; NONE approved yet; refer MAPT-FTDP-17 patients to trial centres",
            "No treatment for prion disease (PRNP) — GSS/FFI/fCJD: supportive care only; quinacrine and doxycycline trials negative; branaplam (PTC Therapeutics) evaluated in CJD; all universally fatal; palliative care specialist involvement mandatory early",
        ],
        "key_standards": [
            "International Society for Frontotemporal Dementias (ISFTD) — GRN/MAPT/C9ORF72: specialist referral, plasma progranulin in all FTD, trial enrolment (INFRONT-3 for GRN-FTD, ALLFTD for all genetic FTDs), advance care planning mandatory at diagnosis",
            "Dominantly Inherited Alzheimer Network (DIAN) — PSEN1/PSEN2/APP: international registry for familial AD; DIAN observational study enrolment for all EOFAD variant carriers; DIAN-TU treatment trials ongoing; longitudinal biomarker tracking beginning before symptom onset",
            "Alzheimer's Association Genetic Testing Guidance (2022): comprehensive panel recommended for EOAD (<65y) with positive family history; include PSEN1, PSEN2, APP, GRN, MAPT, C9ORF72, PRNP; pre/post-test genetic counselling mandatory; predictive testing requires full protocol (minimum age 18y)",
            "C9ORF72 Detection Standard (ALS/FTD Consortium): repeat-primed PCR as first-line screen; Southern blot confirmation; long-read sequencing for research/precision; NEVER report C9ORF72 negative based on standard PCR; test C9ORF72 FIRST before other familial dementia genes",
            "GRN Plasma Progranulin Standard: ELISA-based plasma PGRN measurement in ALL FTD patients; <100 ng/mL = pathognomonic LOF; validated kit required; PGRN functional evidence counted toward GRN variant classification",
            "Prion Disease CJD national surveillance (national CJD Units — UK, Germany, France, USA): all suspected prion disease must be notified; standardised diagnostic workup (rt-QuIC CSF, 14-3-3, MRI DWI cortical ribboning, EEG); international registry enrolment; organ donation flag mandatory at registration",
            "ARIA Monitoring Protocol — Lecanemab/Donanemab (FDA prescribing information): baseline MRI before starting; 5th, 7th, 14th infusion MRIs (weeks 1, 3, 6); monthly for year 1 in APOE4/4 and genetic variant carriers; symptom MRI immediately for any new neurological change; hold/discontinue protocol by ARIA grade",
            "MAPT/Tau Trial Network (4RT consortium — PSP+CBD+FTDP-17): refer all MAPT-FTDP-17 patients; tau PET as biomarker; trial eligibility for gosuranemab, bepranemab, ABBV-CLS-7262; no approved tau-modifying therapy yet",
        ],
    }
