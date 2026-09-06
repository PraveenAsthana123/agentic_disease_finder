#!/usr/bin/env python3
"""Hereditary-Prostate-Cancer-Atlas — Complete 8-Gene Hereditary Prostate Cancer Atlas
BRCA2   (Breast cancer type 2 susceptibility; 3418 aa; 13q12.3; AD;
         HPCA-1 — 15–20× prostate cancer RR; lethal/metastatic phenotype dominant;
         PROfound trial: Olaparib FDA 2020; PSMA-PET mandatory;
         seed SEED_BASE+0) ·
ATM     (Ataxia-telangiectasia mutated; 3056 aa; 11q22.3; AD heterozygous;
         HPCA-2 — 2–4× prostate RR; biallelic = A-T syndrome;
         PROfound: Olaparib HR 0.72 (modest); radiation sensitivity heterozygous;
         seed SEED_BASE+1) ·
CHEK2   (Checkpoint kinase 2; 543 aa; 22q12.1; AD;
         HPCA-3 — 2–3× prostate RR; c.1100delC + I157T European founders;
         no approved PARPi for prostate; standard androgen-deprivation therapy;
         seed SEED_BASE+2) ·
HOXB13  (Homeobox protein Hox-B13; 284 aa; 17q21.32; AD;
         HPCA-4 — G84E founder Scandinavian/Northern European; 4–8× prostate RR;
         prostate-specific gene; no approved targeted therapy;
         seed SEED_BASE+3) ·
MSH2    (MutS protein homologue 2; 934 aa; 2p21; AD;
         HPCA-5 — Lynch syndrome prostate 5–10× RR; dMMR/MSI-H → Pembrolizumab;
         mPRIMO registry; urothelial + endometrial also elevated;
         seed SEED_BASE+4) ·
BRCA1   (Breast cancer type 1 susceptibility; 1863 aa; 17q21.31; AD;
         HPCA-6 — 2–3× prostate RR (weaker than BRCA2); female carriers: RRSO needed;
         PARPi less effective vs BRCA2 in prostate (PROfound HR 0.82 subset);
         seed SEED_BASE+5) ·
PALB2   (Partner and localiser of BRCA2; 1186 aa; 16p12.2; AD;
         HPCA-7 — emerging 2–4× prostate RR; PARPi-sensitive DNA repair gene;
         no dedicated prostate-specific PARPi label yet; expanding trial data;
         seed SEED_BASE+6) ·
NBN     (Nibrin / Nibrin NBS1; 754 aa; 8q21.3; AR biallelic = Nijmegen; AD heterozygous;
         HPCA-8 — 657del5 Slavic founder; heterozygous 3–4× prostate RR;
         NBN is the Nijmegen Breakage Syndrome gene; MRN complex for DSB repair;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1638–1645)
"""

import random

SEED_BASE = 1638

PROSTATE_GENES = [
    # ── BRCA2 — HPCA-1 ──────────────────────────────────────────────────────
    {
        "gene": "BRCA2",
        "protein": "BRCA2 — HPCA-1 AD — RAD51-Loader BRC-Repeat Scaffold — 15–20× Prostate RR — PROfound Olaparib FDA 2020 — PSMA-PET Mandatory",
        "alias": (
            "BRCA2; OMIM gene 600185; Hereditary Breast and Ovarian Cancer Syndrome 2 OMIM 612555; "
            "also: Hereditary Prostate Cancer 1B; 13q12.3; 3418 aa; ~384 kDa; AD haploinsufficiency. "
            "BRCA2 encodes the largest protein in the homologous recombination (HR) pathway: "
            "eight BRC repeat motifs (residues 1002–2085) each bind a RAD51 monomer → "
            "regulates RAD51 nucleofilament assembly on resected ssDNA; "
            "OB-fold domain at C-terminus (BRCA2-DBD) binds ssDNA directly; "
            "N-terminal transactivation domain; "
            "C-terminal interaction motif with RAD51 (residues 3268–3305) — "
            "essential for RAD51 localisation to DSB. "
            "FUNCTION: RAD51 loader — master regulator of high-fidelity HR DSB repair. "
            "At double-strand break (DSB): MRN-ATM → CtIP-mediated 5' resection → "
            "RPA-coated 3' ssDNA overhang → PALB2 bridges BRCA1-BRCA2 → "
            "BRCA2 displaces RPA with RAD51 → RAD51 nucleofilament → strand invasion → "
            "D-loop → error-free template repair. "
            "LOSS OF FUNCTION: HR deficiency (HRD) → NHEJ-mediated DSB repair → "
            "signature SBS3/SBS8 (del at microhomology repeats, large deletions) → "
            "chromosomal instability → high-grade, aggressive tumour phenotype. "
            "PROSTATE CANCER RISK (BRCA2 heterozygotes): "
            "Relative risk (RR): 5–8× before age 65; 15–20× overall lifetime. "
            "Absolute lifetime risk: ~25–35% (versus ~12% general population). "
            "Phenotype: lethal/metastatic prostate cancer enriched (HR 2.1 for castration-resistant "
            "and lethal disease in IMPACT study; BRCA2 carriers are over-represented in "
            "metastatic cohorts by 3–4× vs localised cohorts). "
            "Gleason grade: predominantly GS 7–9 at diagnosis; grade group ≥3 more common. "
            "PROSTATE CANCER SCREENING (BRCA2 male carriers): "
            "Annual PSA (prostate-specific antigen) from age 40 (not 50); "
            "PSA threshold for biopsy: ≥3.0 ng/mL (not 4.0 — carrier-specific threshold); "
            "Multi-parametric MRI (mpMRI) prostate annually or on PSA rise; "
            "PSMA-PET/CT recommended at biochemical recurrence and at staging for high-risk localised disease "
            "(superior sensitivity over CT+bone scan for micrometastases in BRCA2 carriers). "
            "TARGETED THERAPY — PROfound trial (de Bono 2020, NEJM): "
            "Olaparib vs enzalutamide/abiraterone in mCRPC with HRR gene mutations. "
            "Cohort A (BRCA1/2 + ATM): Olaparib HR 0.34 (rPFS); BRCA2 subgroup HR 0.22 — "
            "most dramatic improvement; FDA approved olaparib for BRCA1/2-mutated mCRPC 2020. "
            "TALAPRO-2: talazoparib + enzalutamide vs placebo + enzalutamide; "
            "rPFS benefit in HRR-altered patients. "
            "ELM-PC4: niraparib + abiraterone + prednisone; FDA approved 2023. "
            "PATHOGENIC VARIANTS IN BRCA2: "
            "BRCA2 c.6174delT (Ashkenazi Jewish founder) and c.886delGT (Danish/Icelandic) "
            "enriched in European hereditary prostate cancer cohorts; "
            "c.10141_10142delCT (Slavic), c.8765delAG; "
            "large rearrangements in 5% — MLPA mandatory if sequencing negative in high-risk families. "
            "CASCADE TESTING: All male first-degree relatives of BRCA2 carrier men with prostate cancer "
            "require genetic counselling; female relatives need full HBOC cascade. "
            "LIQUID BIOPSY: ctDNA BRCA2 reversion mutations detected in ~20% of mCRPC post-PARPi — "
            "mechanism of PARPi acquired resistance; serial ctDNA monitoring recommended."
        ),
        "locus": "13q12.3",
        "aa": 3418,
        "kDa": 384,
        "omim_gene": "600185",
        "omim_disease": "Hereditary Breast and Ovarian Cancer Syndrome 2 (OMIM 612555); Hereditary Prostate Cancer 1B",
        "inheritance": "AD; haploinsufficiency; biallelic lethal (Fanconi anaemia complementation group D1 — FANCD1)",
        "gene_class": "High-risk HPCA gene; DNA repair (HR); PARPi-actionable; BRCA2 prostate is lethal-phenotype enriched",
        "key_alerts": [
            "BRCA2-PROSTATE-CANCER-15-20X-RR: BRCA2 heterozygotes have 15–20× prostate cancer risk; annual PSA from age 40 (not 50); mpMRI annually or on PSA rise; biopsy threshold PSA ≥3.0 ng/mL (not 4.0)",
            "BRCA2-OLAPARIB-FDA-2020-PROfound: Olaparib approved mCRPC BRCA2-mutated (PROfound HR 0.22 rPFS subgroup); most dramatic PARPi benefit of any prostate HRR gene; offer at mCRPC transition",
            "BRCA2-PSMA-PET-MANDATORY: PSMA-PET/CT recommended staging + biochemical recurrence in BRCA2 carriers; superior to CT+bone scan for detecting micrometastases",
            "BRCA2-LETHAL-PHENOTYPE-ENRICHED: BRCA2 prostate cancers are 3–4× over-represented in metastatic vs localised cohorts; higher-grade (GS ≥7) at diagnosis; aggressive multimodality treatment warranted",
            "BRCA2-CASCADE-BOTH-SEXES: Male relatives need prostate screening cascade; female relatives need full HBOC (breast + ovarian) cascade — same BRCA2 variant; do not omit female relatives",
            "BRCA2-PARP-REVERSION-RESISTANCE: ctDNA BRCA2 reversion mutations in ~20% post-PARPi mCRPC; liquid biopsy for reversion variants before switching therapy",
        ],
        "etiologies": {
            "BRCA2 heterozygous germline": 78,
            "Somatic BRCA2 (tumour-only)": 12,
            "BRCA2 mosaic germline": 4,
            "BRCA2 VUS (high-risk family)": 6,
        },
        "stats": {
            "mean_dx_age": 62,
            "mean_dx_delay_months": 9,
            "n_patients": 40,
        },
        "dx_delay_distribution": {"<6 mo": 45, "6–12 mo": 32, "12–24 mo": 16, ">24 mo": 7},
    },
    # ── ATM — HPCA-2 ──────────────────────────────────────────────────────────
    {
        "gene": "ATM",
        "protein": "ATM — HPCA-2 AD-Heterozygous — PI3K DSB Kinase — 2–4× Prostate RR — PROfound Modest Benefit — Radiation Sensitivity",
        "alias": (
            "ATM; OMIM gene 607585; Ataxia-Telangiectasia OMIM 208900 (biallelic); "
            "Hereditary Prostate Cancer (heterozygous) — no dedicated OMIM entry; "
            "11q22.3; 3056 aa; ~351 kDa; AD heterozygous (one allele); "
            "biallelic = classic Ataxia-Telangiectasia (AR). "
            "ATM is the master kinase of the DNA damage response (DDR): "
            "HEAT repeat domain (N-terminal) for protein-protein interactions; "
            "FRAP-ATM-TRRAP (FAT) domain; central catalytic PI3K-like kinase domain; "
            "FATC C-terminal regulatory domain (essential for kinase activation). "
            "ACTIVATION: ATM is normally inactive as a homodimer; "
            "DSB → MRN complex (MRE11-RAD50-NBN) recruited → ATM monomerised → "
            "autophosphorylation Ser1981 → active monomer phosphorylates H2AX (γH2AX), "
            "CHK2 (Thr68), BRCA1 (Ser1524), p53 (Ser15), MDM2 → "
            "G1, S, G2 cell cycle checkpoints + apoptosis. "
            "PROSTATE CANCER RISK (ATM heterozygotes): "
            "Meta-analysis (Hickey 2021, JCO): RR 2–4× for prostate cancer. "
            "Risk is intermediate; predominantly localised or locally advanced disease "
            "(less lethal than BRCA2, but metastatic risk elevated vs general population). "
            "Spectrum of variants: missense (kinase domain), frameshift, nonsense; "
            "c.7271T>G (p.Val2424Gly) and c.3161C>A (p.Ser1054*) associated with higher penetrance. "
            "PROSTATE CANCER SCREENING (ATM heterozygotes): "
            "Annual PSA from age 40; biopsy threshold PSA ≥3.0 ng/mL (same as BRCA2); "
            "mpMRI; consider PSMA-PET in high-grade or biochemically recurrent disease. "
            "TARGETED THERAPY: "
            "PROfound Cohort A included ATM; Olaparib HR 0.72 (rPFS) — modest benefit. "
            "ATM-mutated tumours have intermediate HRD score (not as strongly HRD-positive as BRCA2); "
            "PARPi approved (FDA label includes ATM in mCRPC Olaparib label); "
            "ATM-mutated mCRPC: Olaparib reasonable at mCRPC transition even with modest HR. "
            "RADIATION SENSITIVITY — CLINICAL IMPLICATIONS: "
            "ATM HETEROZYGOTES: intermediate radiation sensitivity — standard fractionated RT generally safe; "
            "hypofractionated or SBRT: some concern for increased late normal-tissue toxicity; "
            "inform radiation oncologist of ATM status before planning; "
            "radical prostatectomy may be preferred over definitive RT for localised ATM-PCa. "
            "BIALLELIC ATM (A-T SYNDROME): absolute radiation contraindication — "
            "even diagnostic fluoroscopy carries increased mortality risk; "
            "biallelic ATM carriers have >100× elevated cancer risk; "
            "heterozygous relatives (parents, children) of A-T patients are at increased PCa risk. "
            "PATHOGENIC VARIANTS: frameshift/nonsense 60%, missense kinase domain 30%, "
            "large rearrangements 10% (MLPA if negative sequencing in A-T families)."
        ),
        "locus": "11q22.3",
        "aa": 3056,
        "kDa": 351,
        "omim_gene": "607585",
        "omim_disease": "Ataxia-Telangiectasia OMIM 208900 (biallelic); Hereditary Prostate Cancer (heterozygous)",
        "inheritance": "AD heterozygous for PCa risk; AR biallelic = Ataxia-Telangiectasia",
        "gene_class": "Intermediate-risk HPCA gene; DNA damage response kinase; PARPi-sensitive (modest); radiation-sensitivity heterozygous intermediate",
        "key_alerts": [
            "ATM-PROSTATE-2-4X-RR: ATM heterozygotes have 2–4× prostate cancer risk; annual PSA from age 40; same biopsy threshold as BRCA2 (≥3.0 ng/mL)",
            "ATM-OLAPARIB-MODEST-PROfound: Olaparib PROfound HR 0.72 for ATM (vs 0.22 BRCA2); still offer at mCRPC; benefit real but weaker; do not conflate with BRCA2-equivalent PARPi response",
            "ATM-RADIATION-SENSITIVITY-INTERMEDIATE: Inform radiation oncologist of ATM status; standard RT generally safe for heterozygotes; hypofractionation may increase late toxicity; radical prostatectomy preferred by some centres",
            "ATM-BIALLELIC-ABSOLUTE-RADIATION-CI: Biallelic A-T patients — even diagnostic fluoroscopy contraindicated; standard RT can be fatal; relatives of A-T patients need ATM carrier testing",
            "ATM-MLPA-IF-NEGATIVE-SEQUENCING: Large ATM rearrangements in 10% of A-T families; if sequencing negative in A-T family, add MLPA",
        ],
        "etiologies": {
            "ATM missense (kinase domain)": 42,
            "ATM frameshift/nonsense": 36,
            "ATM large rearrangement": 12,
            "ATM VUS (high-risk family)": 10,
        },
        "stats": {
            "mean_dx_age": 64,
            "mean_dx_delay_months": 11,
            "n_patients": 40,
        },
        "dx_delay_distribution": {"<6 mo": 38, "6–12 mo": 35, "12–24 mo": 19, ">24 mo": 8},
    },
    # ── CHEK2 — HPCA-3 ──────────────────────────────────────────────────────
    {
        "gene": "CHEK2",
        "protein": "CHEK2 — HPCA-3 AD — CHK2 Kinase FHA Domain — 2–3× Prostate RR — c.1100delC I157T Founders — NO PARPi Approved — Standard ADT",
        "alias": (
            "CHEK2; OMIM gene 604373; no dedicated OMIM disease entry (moderate-risk susceptibility); "
            "22q12.1; 543 aa; ~61 kDa; AD haploinsufficiency. "
            "CHEK2 encodes CHK2 kinase — a key transducer of the ATM-initiated DNA damage checkpoint: "
            "N-terminal SQ/TQ cluster domain (ATM phosphorylation sites); "
            "FHA (forkhead-associated) domain — phosphothreonine recognition for protein interactions; "
            "C-terminal serine/threonine kinase domain. "
            "CHECKPOINT FUNCTION: ATM-activated CHK2 → phosphorylates CDC25A (degradation → "
            "S-phase arrest), CDC25C (cytoplasmic sequestration → G2 arrest), BRCA1 (Ser988), "
            "p53 (Ser20 → MDM2 dissociation → p53 stabilisation → apoptosis/arrest). "
            "PROSTATE CANCER RISK (CHEK2 heterozygotes): "
            "Meta-analysis (Maia 2021): RR 2–3× overall; RR ~3× for c.1100delC; "
            "RR ~2× for p.I157T (missense — kinase domain, less penetrant); "
            "predominantly localised to locally advanced prostate cancer (rarely drives lethal metastatic disease). "
            "FOUNDER VARIANTS: "
            "c.1100delC — frameshift, 1 bp deletion; prevalent Northern/Central European; "
            "carrier frequency: Finland 1.1%, Netherlands 0.7%, Germany 0.5%, UK 0.3%; "
            "p.I157T (c.470T>C) — common missense, lower risk; prevalent Eastern Europe; "
            "c.444+1G>A — splice site; Germany/Netherlands. "
            "PROSTATE CANCER SCREENING (CHEK2 carriers): "
            "Annual PSA from age 40–45; biopsy threshold ≥3.0 ng/mL; "
            "mpMRI for equivocal PSA; "
            "no dedicated mpMRI surveillance protocol for CHEK2-PCa (insufficient evidence); "
            "PSMA-PET not routinely recommended (localised disease predominant). "
            "TREATMENT — NO PARPi APPROVED: "
            "CHEK2-mutated prostate cancers are NOT consistently HRD-positive; "
            "PARPi are NOT approved for CHEK2-only mCRPC; "
            "standard ADT + docetaxel + enzalutamide/abiraterone; "
            "PARPi off-label trials ongoing but no PROfound cohort benefit demonstrated for CHEK2 alone. "
            "IMPORTANT DISTINCTION: CHEK2 c.1100delC breast cancer has ER+ phenotype (not HRD-driven); "
            "same applies to prostate cancer — treating CHEK2-PCa as BRCA2-equivalent (PARPi) is a management error. "
            "OTHER CHEK2 CANCER RISKS: "
            "Male breast cancer: ~9× (one of the highest male breast cancer risk genes); "
            "Colon/colorectal: 2–3×; Thyroid: 3–4×; "
            "Female breast: 25–30% lifetime (same range as ATM); "
            "Kidney: 1.5–2×."
        ),
        "locus": "22q12.1",
        "aa": 543,
        "kDa": 61,
        "omim_gene": "604373",
        "omim_disease": "Hereditary Prostate Cancer susceptibility (moderate-risk); also Hereditary Breast Cancer (moderate-risk)",
        "inheritance": "AD; haploinsufficiency; biallelic = Li-Fraumeni-like (very rare)",
        "gene_class": "Moderate-risk HPCA gene; checkpoint kinase; NOT PARPi-actionable; c.1100delC + I157T founders",
        "key_alerts": [
            "CHEK2-PROSTATE-2-3X-RR: CHEK2 heterozygotes have 2–3× prostate cancer risk; c.1100delC higher risk than I157T; annual PSA from age 40–45",
            "CHEK2-NO-PARP-INHIBITOR-PROSTATE: CHEK2-mutated mCRPC — PARPi NOT approved; CHEK2 prostate is not HRD-driven; standard ADT/docetaxel/enzalutamide/abiraterone only",
            "CHEK2-DO-NOT-CONFLATE-WITH-BRCA2: Treating CHEK2-PCa with PARPi by analogy with BRCA2 is a management error; wait for trial data before off-label use",
            "CHEK2-MALE-BREAST-CANCER-9X: CHEK2 male breast cancer risk ~9×; annual mammogram ± MRI from age 40 for male CHEK2 carriers (often missed)",
            "CHEK2-MULTI-CANCER: Colon 2–3×; thyroid 3–4×; kidney 1.5–2×; multi-organ surveillance beyond prostate",
        ],
        "etiologies": {
            "CHEK2 c.1100delC (frameshift)": 48,
            "CHEK2 p.I157T (missense)": 29,
            "CHEK2 splice variant (c.444+1G>A)": 14,
            "CHEK2 other truncating": 9,
        },
        "stats": {
            "mean_dx_age": 66,
            "mean_dx_delay_months": 13,
            "n_patients": 40,
        },
        "dx_delay_distribution": {"<6 mo": 35, "6–12 mo": 38, "12–24 mo": 18, ">24 mo": 9},
    },
    # ── HOXB13 — HPCA-4 ──────────────────────────────────────────────────────
    {
        "gene": "HOXB13",
        "protein": "HOXB13 — HPCA-4 AD — G84E Founder Scandinavian/Northern European — 4–8× Prostate-Specific — No Approved Targeted Therapy — Androgen-Regulated Homeobox",
        "alias": (
            "HOXB13; OMIM gene 604607; Prostate Cancer Susceptibility OMIM 614261; "
            "17q21.32; 284 aa; ~31 kDa; AD (G84E — dominant gain-of-function for prostate susceptibility). "
            "HOXB13 encodes Homeobox protein Hox-B13 — a transcription factor essential for "
            "prostate development and androgen receptor (AR) co-regulatory activity: "
            "N-terminal transactivation domain; "
            "homeodomain (HD) — 60 aa helix-turn-helix DNA binding; "
            "HOXB13 is expressed exclusively in the prostate among adult male tissues "
            "(prostate-specific expression maintained by androgen-responsive elements). "
            "PROTEIN FUNCTION: "
            "HOXB13 controls prostate epithelial differentiation; "
            "regulates expression of prostate-specific antigen (KLK3/PSA) gene; "
            "interacts with the androgen receptor (AR) → "
            "modulates AR target gene expression in prostate cells; "
            "G84E variant (p.Gly84Glu; rs138213197): located in conserved MEIS-binding interface of homeodomain → "
            "altered protein-protein interaction → dominant susceptibility mechanism (not haploinsufficiency). "
            "PROSTATE CANCER RISK (HOXB13 G84E carriers): "
            "OR ~4–8× for prostate cancer (Ewing 2012, NEJM; confirmed in 8 independent cohorts). "
            "G84E FOUNDER VARIANT: "
            "Carrier frequency: Sweden 2.5%, Finland 2.0%, Norway 1.8%, Denmark 1.5%, "
            "Netherlands 0.7%, Germany 0.4%, UK 0.2%, Ireland 0.3%; "
            "Essentially absent in Asian and African populations. "
            "Accounts for ~5–10% of hereditary prostate cancer in Scandinavian populations. "
            "PHENOTYPE: "
            "Early-onset (mean age at diagnosis ~59 years vs 65 general population); "
            "familial clustering — penetrance increases with positive family history; "
            "predominantly localised at diagnosis (unlike BRCA2 metastatic enrichment); "
            "Gleason distribution similar to general population. "
            "PROSTATE CANCER SCREENING (HOXB13 G84E carriers): "
            "Annual PSA from age 40; "
            "biopsy threshold ≥3.0 ng/mL (carrier-specific threshold); "
            "mpMRI for elevated PSA or abnormal digital rectal exam; "
            "PSMA-PET not specifically indicated over standard practice (localised phenotype). "
            "TARGETED THERAPY: "
            "HOXB13 is a transcription factor — NOT a DNA repair gene; "
            "NO approved targeted therapy (no PARPi, no immunotherapy label); "
            "HOXB13 tumours are NOT HRD-positive; "
            "standard prostatectomy/radiotherapy/ADT per Gleason/risk grade; "
            "active surveillance may be appropriate for low-risk localised G84E PCa; "
            "AR-directed therapy (enzalutamide/abiraterone) rational for metastatic disease "
            "(HOXB13 co-regulates AR; ADT sensitivity expected). "
            "TESTING: "
            "G84E is the only established pathogenic variant in HOXB13; "
            "hotspot/targeted testing sufficient for Northern European males; "
            "full HOXB13 sequencing for non-European populations (different variant distribution rare). "
            "CLINICAL CAVEAT: "
            "HOXB13 G84E is prostate-specific — NOT associated with breast, ovarian, or other cancers; "
            "do NOT expand surveillance to other organs beyond prostate in male G84E carriers."
        ),
        "locus": "17q21.32",
        "aa": 284,
        "kDa": 31,
        "omim_gene": "604607",
        "omim_disease": "Prostate Cancer Susceptibility (OMIM 614261); HOXB13 G84E Scandinavian Founder",
        "inheritance": "AD; G84E dominant susceptibility mechanism (not haploinsufficiency)",
        "gene_class": "High-risk prostate-specific HPCA gene; AR co-regulator; NOT DNA repair; NOT PARPi-actionable; G84E founder Northern European",
        "key_alerts": [
            "HOXB13-G84E-PROSTATE-SPECIFIC: HOXB13 G84E is a prostate-cancer-only gene; do NOT screen for breast/ovarian/colon in G84E male carriers — not indicated",
            "HOXB13-4-8X-PROSTATE-RR: G84E carriers have 4–8× prostate cancer risk; onset earlier (~59 yr); annual PSA from age 40",
            "HOXB13-NO-TARGETED-THERAPY: HOXB13 is NOT a DNA repair gene; NO PARPi, NO immunotherapy label; standard prostatectomy/RT/ADT; do not misclassify as HRD",
            "HOXB13-SCANDINAVIAN-FOUNDER-2-5PCT-CARRIER: G84E carrier frequency 2.5% in Sweden/Finland; test Northern European males with family history of prostate cancer; rare in Asian/African populations",
            "HOXB13-ACTIVE-SURVEILLANCE-APPROPRIATE: Low-risk localised G84E PCa may be appropriately managed with active surveillance (localised phenotype predominant); do not over-treat",
        ],
        "etiologies": {
            "HOXB13 G84E (founder)": 96,
            "HOXB13 other variant (rare)": 4,
        },
        "stats": {
            "mean_dx_age": 59,
            "mean_dx_delay_months": 10,
            "n_patients": 40,
        },
        "dx_delay_distribution": {"<6 mo": 42, "6–12 mo": 33, "12–24 mo": 17, ">24 mo": 8},
    },
    # ── MSH2 — HPCA-5 ──────────────────────────────────────────────────────────
    {
        "gene": "MSH2",
        "protein": "MSH2 — HPCA-5 AD — Lynch Syndrome — 5–10× Prostate RR — dMMR/MSI-H → Pembrolizumab — mPRIMO Registry — Urothelial + Endometrial Also Elevated",
        "alias": (
            "MSH2; OMIM gene 609309; Lynch Syndrome 1 OMIM 120435; "
            "2p21; 934 aa; ~105 kDa; AD haploinsufficiency (Lynch syndrome). "
            "MSH2 encodes MutS protein homologue 2 — the core subunit of the MutSα heterodimer "
            "(MSH2-MSH6) that recognises single-base mismatches and 1–2 base insertion-deletion loops (IDLs) "
            "in the mismatch repair (MMR) pathway: "
            "Mismatch-binding domain (MBD); lever domain; clamp domain; ATPase domain; "
            "MSH2 dimerises with MSH6 (MutSα — single base mismatch recognition) OR "
            "with MSH3 (MutSβ — larger IDL recognition). "
            "MMR FUNCTION: MSH2-MSH6 (MutSα) scans nascent DNA post-replication → "
            "recognises mismatch → recruits MLH1-PMS2 (MutLα) → signals to PCNA-loaded EXO1 → "
            "excision + re-synthesis of error-containing strand. "
            "LOSS OF MMR → microsatellite instability (MSI-H) → accumulation of frameshift mutations "
            "in microsatellite regions → immune-hot tumour microenvironment → "
            "neoantigen burden + T-cell infiltration → sensitivity to checkpoint blockade. "
            "PROSTATE CANCER RISK (MSH2 Lynch carriers): "
            "Lynch syndrome prostate RR: 5–10× overall (Ericson 2008, Dominguez-Valentin 2021). "
            "Absolute lifetime risk: ~15–25% in MSH2 carriers (vs ~12% general population); "
            "MSH2 highest Lynch prostate risk (vs MLH1 3–5×, MSH6 2–3×, PMS2 marginal). "
            "Mean age at prostate cancer diagnosis in Lynch: ~60 years (earlier than general population). "
            "TARGETED THERAPY — PEMBROLIZUMAB: "
            "FDA 2017 (tumor-agnostic): pembrolizumab approved for dMMR/MSI-H solid tumours — "
            "first-ever tumor-agnostic approval; applies to MSH2-deficient prostate cancer. "
            "ORR in dMMR prostate cancer (KEYNOTE-158): ~33% (modest vs colorectal); "
            "mPRIMO registry (multicentre) collects MSI-H prostate outcomes; "
            "consider pembrolizumab for mCRPC with confirmed MSI-H or dMMR (IHC loss of MSH2/MSH6). "
            "LYNCH SYNDROME PROSTATE SURVEILLANCE: "
            "Annual PSA from age 40; biopsy threshold ≥3.0 ng/mL; "
            "mpMRI if PSA elevated. "
            "MULTI-ORGAN SURVEILLANCE IN MSH2: "
            "Colorectal cancer: 50–60% lifetime; colonoscopy every 1–2 years from age 25. "
            "Endometrial: 40–60% lifetime; annual gynaecological surveillance from 30–35. "
            "Urothelial (upper tract): 14% lifetime in MSH2 (HIGHEST Lynch urothelial risk); "
            "annual urine cytology ± imaging from age 30–35. "
            "Ovarian: 10–15%; annual TVU + CA-125. "
            "Gastric: 5–10%; gastroscopy every 2–3 years from age 30–35. "
            "Lynch syndrome surveillance is comprehensive — prostate is one of many. "
            "EPCAM UPSTREAM DELETION: MSH2 can be silenced by EPCAM deletion "
            "(EPCAM exon 8–9 deletion → antisense read-through → MSH2 promoter methylation); "
            "if MSH2 sequencing negative but IHC shows MSH2 loss → EPCAM MLPA mandatory."
        ),
        "locus": "2p21",
        "aa": 934,
        "kDa": 105,
        "omim_gene": "609309",
        "omim_disease": "Lynch Syndrome 1 (OMIM 120435); Hereditary Non-Polyposis Colorectal Cancer HNPCC1",
        "inheritance": "AD; haploinsufficiency; Lynch syndrome (MSH2 highest Lynch urothelial + prostate risk)",
        "gene_class": "High-risk Lynch-HPCA gene; mismatch repair; dMMR/MSI-H → pembrolizumab; multi-organ Lynch surveillance mandatory",
        "key_alerts": [
            "MSH2-PROSTATE-5-10X-RR: MSH2 Lynch carriers have 5–10× prostate cancer risk (highest Lynch gene for prostate); annual PSA from age 40",
            "MSH2-PEMBROLIZUMAB-dMMR: Confirmed MSI-H/dMMR prostate cancer → pembrolizumab (FDA 2017 tumor-agnostic); IHC MSH2/MSH6 loss in tumour confirms dMMR; offer at mCRPC",
            "MSH2-UROTHELIAL-14PCT-HIGHEST: MSH2 has 14% urothelial lifetime risk (highest Lynch gene for upper tract); annual urine cytology + upper tract imaging from age 30–35",
            "MSH2-EPCAM-MLPA-MANDATORY: If MSH2 sequencing negative but IHC shows MSH2 loss → EPCAM MLPA for upstream deletion (3' EPCAM deletions silence MSH2 by read-through methylation)",
            "MSH2-MULTI-ORGAN-LYNCH: Colorectal 50–60%; endometrial 40–60%; urothelial 14%; ovarian 10–15%; gastric 5–10%; comprehensive Lynch surveillance — prostate is one of many organs at risk",
        ],
        "etiologies": {
            "MSH2 frameshift/nonsense": 52,
            "MSH2 missense (MBD/ATPase)": 26,
            "EPCAM upstream deletion (MSH2 silencing)": 13,
            "MSH2 large deletion/rearrangement": 9,
        },
        "stats": {
            "mean_dx_age": 60,
            "mean_dx_delay_months": 12,
            "n_patients": 40,
        },
        "dx_delay_distribution": {"<6 mo": 40, "6–12 mo": 34, "12–24 mo": 18, ">24 mo": 8},
    },
    # ── BRCA1 — HPCA-6 ──────────────────────────────────────────────────────────
    {
        "gene": "BRCA1",
        "protein": "BRCA1 — HPCA-6 AD — RING-BRCT HR Scaffold — 2–3× Prostate RR (Weaker than BRCA2) — PARPi Less Effective — Female Carriers Need Full HBOC Cascade",
        "alias": (
            "BRCA1; OMIM gene 113705; Hereditary Breast and Ovarian Cancer Syndrome 1 OMIM 604370; "
            "Hereditary Prostate Cancer (moderate association); "
            "17q21.31; 1863 aa; ~220 kDa; AD haploinsufficiency. "
            "BRCA1 encodes the master scaffold of the HR DSB repair pathway: "
            "N-terminal RING domain (BARD1-heterodimerisation; E3 ubiquitin ligase → H2AK119ub); "
            "central NLS/SQ-TQ phosphorylation domain (ATM/ATR targets; interaction with PALB2, FANCJ/BRIP1); "
            "C-terminal tandem BRCT repeats (phosphopeptide recognition; BACH1, Abraxas, CtIP binding). "
            "PROSTATE CANCER RISK (BRCA1 heterozygotes): "
            "RR 2–3× for prostate cancer (IMPACT Consortium, Leongamornlert 2014). "
            "This is LOWER than BRCA2 (15–20×) — the difference is clinically significant. "
            "Phenotype: less metastatic enrichment than BRCA2; localised to locally advanced predominant. "
            "PROfound BRCA1 subgroup: Olaparib HR 0.82 (rPFS) — much weaker than BRCA2 (HR 0.22); "
            "some oncologists question whether Olaparib benefit in BRCA1-mCRPC justifies use "
            "(FDA label includes BRCA1 in Olaparib approval, but BRCA1-specific RCT evidence is thin). "
            "PROSTATE CANCER SCREENING (BRCA1 males): "
            "Annual PSA from age 40–45 (BRCA1 male prostate risk is lower than BRCA2, "
            "but still double the general population); "
            "biopsy threshold ≥3.0 ng/mL; "
            "mpMRI for elevated PSA. "
            "IMPORTANT: BRCA1-SPECIFIC MALE PROSTATE RISK ≠ BRCA2 LEVEL: "
            "Some guidelines recommend BRCA1 male prostate surveillance start at 40–45 (slightly later than BRCA2); "
            "PSMA-PET not specifically mandated for BRCA1 localised PCa; "
            "PARPi at mCRPC: offer per FDA label but counsel on weaker evidence vs BRCA2. "
            "FEMALE BRCA1 CARRIERS — FULL HBOC CASCADE MANDATORY: "
            "Female relatives of BRCA1 prostate cancer patients need complete HBOC evaluation: "
            "breast: 70% lifetime risk; RRSO: age 35–40; PARPi (olaparib/talazoparib) if breast/ovarian cancer; "
            "do not limit cascade to prostate-related counselling. "
            "BRCA1 PROSTATE HISTOLOGY: "
            "Unlike breast (75% triple-negative), BRCA1 prostate cancers do not have a defined immunohistochemical "
            "footprint distinct from sporadic prostate cancer; "
            "grade/stage distributions intermediate between general population and BRCA2 carriers. "
            "PATHOGENIC VARIANTS: "
            "Ashkenazi Jewish founders: c.68_69delAG (185delAG) + c.5266dupC (5382insC) — "
            "carrier frequency 1/40 Ashkenazi; "
            "Portuguese c.2037del4; Dutch c.2985_2986del; "
            "MLPA mandatory if sequencing negative (large rearrangements 10%)."
        ),
        "locus": "17q21.31",
        "aa": 1863,
        "kDa": 220,
        "omim_gene": "113705",
        "omim_disease": "Hereditary Breast and Ovarian Cancer Syndrome 1 (OMIM 604370); Hereditary Prostate Cancer (moderate)",
        "inheritance": "AD; haploinsufficiency; biallelic = Fanconi anaemia complementation group S (FANCS)",
        "gene_class": "Moderate-risk HPCA gene; DNA repair (HR); PARPi-sensitive (weaker than BRCA2 for prostate); female relatives need full HBOC cascade",
        "key_alerts": [
            "BRCA1-PROSTATE-2-3X-RR: BRCA1 has 2–3× prostate risk (lower than BRCA2 15–20×); do not conflate BRCA1 and BRCA2 prostate risk; management differs",
            "BRCA1-PARP-INHIBITOR-WEAKER-PROSTATE: PROfound BRCA1 subgroup HR 0.82 (vs BRCA2 HR 0.22); Olaparib is FDA-approved for BRCA1 mCRPC but evidence weaker; counsel accordingly",
            "BRCA1-FEMALE-RELATIVES-FULL-HBOC: Female relatives of BRCA1 PCa patients need complete HBOC cascade (breast 70% + ovarian 46%); do not limit counselling to prostate-only discussion",
            "BRCA1-MLPA-MANDATORY: Large BRCA1 rearrangements in 10%; if NGS negative in high-risk family → MLPA mandatory",
            "BRCA1-ASHKENAZI-FOUNDER: c.68_69delAG + c.5266dupC carrier frequency 1/40 Ashkenazi Jewish; targeted founder testing in eligible populations before full panel",
        ],
        "etiologies": {
            "BRCA1 frameshift/nonsense": 48,
            "BRCA1 large rearrangement (MLPA)": 11,
            "BRCA1 missense (RING/BRCT)": 24,
            "BRCA1 VUS (high-risk family)": 17,
        },
        "stats": {
            "mean_dx_age": 65,
            "mean_dx_delay_months": 10,
            "n_patients": 40,
        },
        "dx_delay_distribution": {"<6 mo": 43, "6–12 mo": 33, "12–24 mo": 17, ">24 mo": 7},
    },
    # ── PALB2 — HPCA-7 ──────────────────────────────────────────────────────────
    {
        "gene": "PALB2",
        "protein": "PALB2 — HPCA-7 AD — BRCA1-BRCA2 Bridge — Emerging 2–4× Prostate RR — PARPi-Sensitive DNA Repair — No Dedicated Prostate PARPi Label Yet",
        "alias": (
            "PALB2; OMIM gene 610355; Hereditary Breast and Ovarian Cancer PALB2 OMIM 610355; "
            "Fanconi Anaemia Complementation Group N (FANCN) biallelic; "
            "16p12.2; 1186 aa; ~131 kDa; AD haploinsufficiency. "
            "PALB2 (Partner And Localiser of BRCA2) encodes the essential molecular bridge "
            "connecting BRCA1 and BRCA2 in the HR pathway: "
            "N-terminal coiled-coil domain → binds BRCA1 (Leu21 critical); "
            "central WD40 repeat domain (7 WD40 repeats → β-propeller scaffold) → "
            "RAD51 interaction; "
            "C-terminal BRCA2-binding domain → recruits BRCA2 to DSB site; "
            "Without PALB2: BRCA1 cannot recruit BRCA2 to the DSB → HR failure. "
            "PROSTATE CANCER RISK (PALB2 heterozygotes): "
            "Risk data are emerging and less mature than BRCA2: "
            "OR 2–4× from case-control studies (Yang 2020 NCCN Update; Silvestri 2021). "
            "Risk is likely intermediate between CHEK2 and BRCA2. "
            "PALB2 prostate cancer phenotype: early data suggest lethal/metastatic enrichment "
            "similar to BRCA2 (PALB2 is also an HR gene → HRD tumours); "
            "insufficient cohort size for definitive estimates. "
            "TARGETED THERAPY: "
            "PALB2 tumours ARE HRD-positive (same HR pathway as BRCA1/BRCA2); "
            "PARPi sensitivity expected: off-label trials show PALB2 mCRPC responds to olaparib/rucaparib; "
            "no dedicated mCRPC prostate-specific PARPi label for PALB2 yet (FDA label focuses on BRCA1/2); "
            "some centres include PALB2 in their HRR panel for mCRPC PARPi eligibility; "
            "TALAPRO-2 and ELM-PC4 included PALB2 in HRR-altered subgroups. "
            "PROSTATE CANCER SCREENING (PALB2 males): "
            "Annual PSA from age 40–45; biopsy threshold ≥3.0 ng/mL; "
            "mpMRI for elevated PSA; PSMA-PET for high-risk/metastatic. "
            "OTHER PALB2 CANCER RISKS: "
            "Breast (female): 35–58% lifetime (BRCA2-comparable at upper end — PALB2 International Consortium); "
            "BRCA1 bridge: if PALB2 fails, BRCA1-BRCA2 cannot couple → complete HR failure; "
            "Ovarian: ~5% (uncertain); pancreatic: 2–4× risk; "
            "Biallelic PALB2 = Fanconi Anaemia type N (FANCN) — bone marrow failure + solid tumours. "
            "CASCADE TESTING: female relatives need PALB2 breast/ovarian risk assessment; "
            "TBCRC048 (olaparib for PALB2 metastatic breast): 82% ORR — strong PARPi response data "
            "supports PARPi sensitivity across PALB2-mutated solid tumours."
        ),
        "locus": "16p12.2",
        "aa": 1186,
        "kDa": 131,
        "omim_gene": "610355",
        "omim_disease": "Hereditary Breast and Ovarian Cancer (PALB2); Fanconi Anaemia FANCN (biallelic)",
        "inheritance": "AD; haploinsufficiency; biallelic = Fanconi Anaemia type N (FANCN)",
        "gene_class": "Intermediate-high-risk HPCA gene (emerging data); BRCA1-BRCA2 bridge; HR repair; PARPi-sensitive; prostate label expanding",
        "key_alerts": [
            "PALB2-PROSTATE-EMERGING-2-4X: PALB2 prostate cancer risk 2–4× (less mature data than BRCA2); annual PSA from age 40–45; data evolving — check updated guidelines",
            "PALB2-PARPi-SENSITIVE-NO-LABEL: PALB2 mCRPC tumours are HRD-positive; PARPi off-label use growing; no dedicated prostate-specific PARPi label yet; TBCRC048 ORR 82% in breast supports sensitivity",
            "PALB2-FEMALE-RELATIVES-BREAST-58PCT: Female relatives of PALB2 PCa patients need breast surveillance (35–58% lifetime risk); RRSO consideration age 45–50 for ovarian (~5%)",
            "PALB2-BRCA1-BRCA2-BRIDGE: PALB2 bridges BRCA1 to BRCA2; loss = complete HR failure; tumour biology similar to BRCA2; expect lethal-phenotype enrichment in prostate (data emerging)",
        ],
        "etiologies": {
            "PALB2 frameshift/nonsense": 55,
            "PALB2 missense (WD40/coiled-coil)": 25,
            "PALB2 splice variant": 13,
            "PALB2 large deletion": 7,
        },
        "stats": {
            "mean_dx_age": 63,
            "mean_dx_delay_months": 11,
            "n_patients": 40,
        },
        "dx_delay_distribution": {"<6 mo": 40, "6–12 mo": 35, "12–24 mo": 17, ">24 mo": 8},
    },
    # ── NBN — HPCA-8 ──────────────────────────────────────────────────────────────
    {
        "gene": "NBN",
        "protein": "NBN — HPCA-8 AR-Biallelic-Nijmegen AR / AD-Heterozygous Prostate — 657del5 Slavic Founder — 3–4× Prostate RR Heterozygous — MRN DSB Complex",
        "alias": (
            "NBN (also NBS1, p95); OMIM gene 602667; Nijmegen Breakage Syndrome OMIM 251260 (biallelic AR); "
            "Hereditary Prostate Cancer susceptibility (AD heterozygous); "
            "8q21.3; 754 aa; ~85 kDa; "
            "biallelic: AR (Nijmegen Breakage Syndrome); heterozygous: AD moderate prostate/breast risk. "
            "NBN encodes Nibrin (NBS1/p95) — a scaffold/regulatory subunit of the MRN complex "
            "(MRE11-RAD50-NBN): "
            "N-terminal FHA (forkhead-associated) domain → phosphothreonine binding (MDC1-BRCT interaction); "
            "BRCT1-BRCT2 tandem domains → γH2AX recognition at DSB; "
            "C-terminal MRE11-binding domain (residues 665–700) → MRN complex assembly. "
            "MRN COMPLEX FUNCTION: "
            "First sensor of DSB → NBN recruits MRN to DSB via FHA/BRCT → "
            "MRE11 nuclease performs initial 5' end resection → "
            "RAD50 ATP-dependent bridging of DSB ends → "
            "NBS1 C-terminus recruits and activates ATM → "
            "γH2AX foci → MDC1-53BP1-BRCA1 cascade → HR or NHEJ. "
            "NBN 657del5 FOUNDER MUTATION: "
            "c.657_661del5 (p.Lys219Glufs*17); frameshift → C-terminal truncation → "
            "hypomorphic MRN complex (partial ATM activation); "
            "NOT complete loss-of-function (complete loss is embryonic lethal in homozygotes — "
            "Nijmegen patients have hypomorphic alleles, not null). "
            "CARRIER FREQUENCY: "
            "Poland 0.5–1.0%; Czech Republic/Slovakia 0.8%; Ukraine 0.5%; Russia 0.3%; "
            "Belarus 0.6%; essentially absent in non-Slavic populations. "
            "BIALLELIC NBN — NIJMEGEN BREAKAGE SYNDROME: "
            "Bird-like facial features (microcephaly, beak-like nose, micrognathia); "
            "severe combined immunodeficiency (recurrent respiratory infections); "
            "chromosomal instability (Sister chromatid exchanges, radiosensitivity); "
            "profound cancer predisposition: lymphoma (~50%), medulloblastoma, breast cancer. "
            "PROSTATE CANCER RISK (NBN 657del5 heterozygotes): "
            "OR 3–4× for prostate cancer (Cybulski 2004, 2013 — Polish studies); "
            "absolute prostate cancer lifetime risk ~25–30% (Polish male heterozygous carrier data); "
            "predominantly localised to locally advanced prostate cancer; "
            "OR for breast cancer (female NBN carriers): ~2–3× (moderate breast risk). "
            "PROSTATE CANCER SCREENING (NBN 657del5 heterozygotes): "
            "Annual PSA from age 40; biopsy threshold ≥3.0 ng/mL; "
            "mpMRI for elevated PSA. "
            "TARGETED THERAPY: "
            "NBN participates in ATM activation — NBN-mutated tumours may have intermediate HRD; "
            "PARPi data for NBN-mutated prostate cancer sparse; "
            "some mCRPC trials include NBN in HRR panel eligibility; "
            "no approved targeted therapy specific to NBN-mutated prostate cancer. "
            "RADIATION SENSITIVITY: "
            "NBS patients: profound radiosensitivity (MRN is ATM sensor — loss → ATM cannot activate); "
            "heterozygous NBN carriers: intermediate radiation sensitivity (similar to ATM heterozygotes); "
            "inform radiation oncologist of NBN 657del5 carrier status. "
            "TESTING NOTE: "
            "Slavic heritage: targeted 657del5 hotspot testing appropriate; "
            "non-Slavic: full NBN sequencing if clinical suspicion; "
            "MLPA for large rearrangements in NBS families."
        ),
        "locus": "8q21.3",
        "aa": 754,
        "kDa": 85,
        "omim_gene": "602667",
        "omim_disease": "Nijmegen Breakage Syndrome OMIM 251260 (biallelic); Hereditary Prostate Cancer (heterozygous 657del5)",
        "inheritance": "AR biallelic = Nijmegen Breakage Syndrome; AD heterozygous = moderate prostate/breast risk",
        "gene_class": "Moderate-risk HPCA gene; MRN DSB complex (NBN subunit); 657del5 Slavic founder; intermediate radiation sensitivity heterozygous; PARPi data sparse",
        "key_alerts": [
            "NBN-657del5-SLAVIC-FOUNDER: NBN 657del5 carrier frequency ~0.5–1% in Polish/Czech/Slovak/Ukrainian populations; targeted hotspot testing in Slavic males with prostate cancer family history",
            "NBN-PROSTATE-3-4X-RR: NBN 657del5 heterozygotes have 3–4× prostate cancer risk; annual PSA from age 40; predominantly localised phenotype",
            "NBN-NIJMEGEN-BIALLELIC-NBS: Biallelic NBN = Nijmegen Breakage Syndrome (microcephaly, immunodeficiency, lymphoma); if NBS patient diagnosed → all relatives need NBN carrier testing for prostate/breast risk",
            "NBN-RADIATION-SENSITIVITY-INTERMEDIATE: NBN heterozygotes have intermediate radiation sensitivity (MRN-ATM axis); inform radiation oncologist; standard RT generally tolerated but monitor",
            "NBN-PARPi-SPARSE-DATA: No approved PARPi for NBN-mutated prostate cancer; some HRR panel-based trials include NBN; do not infer equivalent benefit to BRCA2",
        ],
        "etiologies": {
            "NBN 657del5 (Slavic founder)": 88,
            "NBN other truncating variant": 8,
            "NBN missense (FHA/BRCT)": 4,
        },
        "stats": {
            "mean_dx_age": 64,
            "mean_dx_delay_months": 12,
            "n_patients": 40,
        },
        "dx_delay_distribution": {"<6 mo": 37, "6–12 mo": 36, "12–24 mo": 19, ">24 mo": 8},
    },
]


def _make_patients(gene_info: dict, seed: int) -> list:
    rng = random.Random(seed)
    mean_age = gene_info["stats"]["mean_dx_age"]
    mean_delay = gene_info["stats"]["mean_dx_delay_months"]
    patients = []
    for i in range(40):
        age = max(38, min(82, int(rng.gauss(mean_age, 6))))
        delay = max(1, int(rng.gauss(mean_delay, 4)))
        patients.append({
            "patient_id": f"HPCA-{gene_info['gene']}-{seed}-{i+1:03d}",
            "age_at_diagnosis": age,
            "diagnosis_delay_months": delay,
            "seed": seed,
        })
    return patients


def get_overview() -> dict:
    all_pts = []
    gene_summaries = []
    for idx, g in enumerate(PROSTATE_GENES):
        seed = SEED_BASE + idx
        pts = _make_patients(g, seed)
        all_pts.extend(pts)
        gene_summaries.append({
            "gene": g["gene"],
            "locus": g["locus"],
            "inheritance": g["inheritance"].split(";")[0].strip(),
            "omim_disease": g["omim_disease"],
            "mean_dx_age": g["stats"]["mean_dx_age"],
            "n_patients": 40,
        })

    rng = random.Random(SEED_BASE)
    total = len(all_pts)
    mean_age = round(sum(p["age_at_diagnosis"] for p in all_pts) / total, 1)
    mean_delay = round(sum(p["diagnosis_delay_months"] for p in all_pts) / total, 1)

    return {
        "atlas": "Hereditary-Prostate-Cancer-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Prostate Cancer Reference · "
            "BRCA2 · ATM · CHEK2 · HOXB13 · MSH2 · BRCA1 · PALB2 · NBN · "
            "320 patients (8×40, seeds 1638–1645)"
        ),
        "aggregate_stats": {
            "total_patients": total,
            "mean_dx_age_years": mean_age,
            "mean_dx_delay_months": mean_delay,
            "brca2_lethal_metastatic_enriched": True,
            "brca2_psa_screening_age": 40,
            "brca2_psa_biopsy_threshold_ng_ml": 3.0,
            "brca2_olaparib_fda_approved": True,
            "brca2_profond_hr_rPFS": 0.22,
            "atm_profond_hr_rPFS": 0.72,
            "hoxb13_g84e_carrier_pct_sweden": 2.5,
            "hoxb13_prostate_only_gene": True,
            "msh2_lynch_prostate_rr": "5–10×",
            "msh2_pembrolizumab_approved_dMMR": True,
            "nbn_657del5_slavic_founder": True,
            "cascade_tested_pct": round(rng.uniform(64, 78), 1),
            "psma_pet_performed_pct": round(rng.uniform(42, 55), 1),
        },
        "genes": gene_summaries,
        "top_alerts": [
            "BRCA2-PROSTATE-15-20X-RR: Annual PSA age 40; mpMRI; biopsy threshold ≥3.0 ng/mL; PSMA-PET mandatory metastatic",
            "BRCA2-OLAPARIB-PROfound-HR-0.22: Most effective PARPi in mCRPC; offer at castration-resistant transition; PROfound FDA approved 2020",
            "HOXB13-G84E-PROSTATE-ONLY: No PARPi, no immunotherapy label; transcription factor not DNA repair gene; active surveillance appropriate for localised low-risk",
            "MSH2-LYNCH-PEMBROLIZUMAB: Confirm MSI-H/dMMR by tumour IHC → pembrolizumab (FDA 2017 tumor-agnostic); mPRIMO registry",
            "NBN-657del5-SLAVIC-RADIATION-SENSITIVITY: Slavic heritage → targeted hotspot testing; intermediate radiation sensitivity; inform radiation oncologist",
            "CHEK2-NO-PARP-INHIBITOR: CHEK2 mCRPC NOT PARPi-approved; avoid BRCA2-equivalent treatment; standard ADT pathway",
            "ATM-RADIATION-INTERMEDIATE: Heterozygous ATM — standard RT generally safe; inform radiation oncologist; radical prostatectomy preferred some centres",
            "BRCA1-WEAKER-PROSTATE-PARP: BRCA1 prostate PARPi HR 0.82 (vs BRCA2 0.22); included in Olaparib FDA label but evidence weaker; counsel accordingly",
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for idx, g in enumerate(PROSTATE_GENES):
        seed = SEED_BASE + idx
        pts = _make_patients(g, seed)
        info = g["stats"]
        result[g["gene"]] = {
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
            "stats": info,
            "dx_delay_distribution": g["dx_delay_distribution"],
            "patients": pts[:10],
        }
    return result


def get_definitions() -> dict:
    return {
        "atlas": "Hereditary-Prostate-Cancer-Atlas",
        "concepts": {
            "HPCA Gene Tiers — Risk-Stratified Clinical Management": (
                "Hereditary Prostate Cancer (HPCA) genes are divided into risk tiers for clinical management: "
                "HIGH-RISK DNA REPAIR GENES (BRCA2, MSH2): "
                "BRCA2: 15–20× RR; annual PSA from age 40; biopsy ≥3.0 ng/mL; PARPi (olaparib) FDA-approved mCRPC; PSMA-PET mandatory; "
                "MSH2: 5–10× RR (Lynch); annual PSA from age 40; dMMR/MSI-H → pembrolizumab; multi-organ Lynch surveillance. "
                "INTERMEDIATE-RISK DNA REPAIR GENES (ATM, BRCA1, PALB2, NBN): "
                "ATM: 2–4× RR; olaparib PROfound modest benefit (HR 0.72); radiation sensitivity intermediate; "
                "BRCA1: 2–3× RR; olaparib PROfound weak benefit (HR 0.82); female relatives need full HBOC cascade; "
                "PALB2: 2–4× RR emerging; PARPi sensitive (no dedicated prostate label); BRCA1-BRCA2 bridge; "
                "NBN: 3–4× RR; 657del5 Slavic founder; PARPi data sparse. "
                "MODERATE-RISK NON-DNA-REPAIR GENES (CHEK2, HOXB13): "
                "CHEK2: 2–3× RR; c.1100delC + I157T founders; NO PARPi approved for prostate; standard ADT; "
                "HOXB13: 4–8× RR (G84E); prostate-specific transcription factor; NO PARPi; NO targeted therapy; "
                "active surveillance appropriate for low-risk localised HOXB13 prostate cancer. "
                "ALL CARRIERS: Annual PSA from age 40; biopsy threshold ≥3.0 ng/mL (not 4.0 — carrier-specific); "
                "mpMRI for elevated PSA; PSMA-PET for high-grade or metastatic disease (particularly BRCA2); "
                "cascade testing of first-degree male AND female relatives."
            ),
            "PROfound Trial — Olaparib in Homologous Recombination-Altered mCRPC": (
                "PROfound (de Bono 2020, NEJM) is the pivotal phase III trial that established PARPi in prostate cancer: "
                "DESIGN: mCRPC patients with HRR gene alteration (progressed on enzalutamide/abiraterone) → "
                "randomised Olaparib 300 mg BD vs enzalutamide/abiraterone (physician choice). "
                "COHORT A (BRCA1/2 + ATM): primary endpoint: "
                "Olaparib rPFS HR 0.34 (95% CI 0.25–0.47); OS HR 0.64. "
                "BRCA2 subgroup: rPFS HR 0.22 — most dramatic HRR gene benefit. "
                "ATM subgroup: rPFS HR 0.72 — modest but positive. "
                "BRCA1 subgroup: rPFS HR 0.82 — weakest benefit; small number. "
                "COHORT B (non-BRCA1/2/ATM HRR genes including CDK12, CHEK2, FANCA, etc.): "
                "rPFS HR 0.49; included in secondary endpoint only; "
                "CHEK2-alone: minimal benefit — NOT a primary approved indication. "
                "FDA APPROVAL 2020: Olaparib approved for gBRCA1/2 or sBRCA1/2 mCRPC. "
                "Label expanded to include ATM; practice varies for PALB2/NBN (off-label). "
                "CLINICAL TAKE: "
                "BRCA2 = offer Olaparib immediately at mCRPC; "
                "ATM = offer Olaparib at mCRPC (benefit modest but real); "
                "CHEK2 = do NOT offer Olaparib as standard (not approved; no PROfound signal); "
                "HOXB13 = NOT HRD gene; PARPi completely contraindicated as targeted therapy."
            ),
            "MSI-H/dMMR in Prostate Cancer — Pembrolizumab FDA 2017": (
                "Mismatch repair deficiency (dMMR) in prostate cancer creates an immune-hot tumour "
                "microenvironment amenable to checkpoint blockade: "
                "MECHANISM: Loss of MSH2 (or other MMR genes) → MSI-H → thousands of frameshift neoantigens → "
                "T-cell infiltration → immune checkpoint (PD-1/PD-L1) upregulation → "
                "pembrolizumab anti-PD-1 antibody releases immune suppression → cytotoxic T-cell killing. "
                "FREQUENCY: dMMR in 3–5% of all prostate cancers; enriched in mCRPC (~12%); "
                "Lynch syndrome prostate cancers: 100% dMMR (germline MMR loss). "
                "PEMBROLIZUMAB FDA 2017 (tumor-agnostic): "
                "First-ever tumor-agnostic approval; for ALL solid tumours with MSI-H or dMMR. "
                "KEYNOTE-158 prostate cohort: ORR 33% (lower than colorectal 40%; colorectal benefits from "
                "MMR-deficiency T-cell priming from gut microbiome). "
                "DIAGNOSTICS: "
                "IHC MMR protein panel (MSH2/MSH6/MLH1/PMS2) on tumour biopsy; loss of MSH2 = dMMR; "
                "MSI PCR or NGS-based MSI; NGS tumour mutational burden (TMB) ≥10 mut/Mb (alternative FDA label). "
                "mPRIMO REGISTRY: multicentre registry collecting MSI-H prostate outcomes; "
                "long-term responders (>2 years) documented; complete responses rare but durable. "
                "MANAGEMENT: Confirm dMMR in tumour before pembrolizumab; "
                "germline MMR testing (Lynch syndrome cascade) in dMMR prostate patients regardless of age."
            ),
            "PSMA-PET/CT in Hereditary Prostate Cancer — When and Why": (
                "Prostate-specific membrane antigen (PSMA) PET/CT is the most sensitive imaging modality "
                "for detecting prostate cancer metastases: "
                "PSMA = FOLH1; transmembrane glutamate carboxypeptidase; overexpressed 100–1000× on "
                "prostate cancer cells including micrometastases. "
                "PSMA-PET TRACERS: Ga-68-PSMA-11 (FDA 2020 — biochemical recurrence); "
                "F-18-DCFPyL (Pylarify; FDA 2021); F-18-PSMA-1007 (EU licensed). "
                "INDICATIONS IN HPCA: "
                "BRCA2/ATM/PALB2 carriers at initial staging (high-risk localised or locally advanced): "
                "PSMA-PET detects occult nodal/bone metastases superior to CT+bone scan "
                "(sensitivity 40% higher for N1 disease); "
                "changes staging in ~27% of high-risk cases (ProPSMA trial). "
                "Biochemical recurrence after curative intent therapy: "
                "PSMA-PET detects recurrence site at PSA 0.5–1.0 ng/mL vs PSA >10 for CT+bone scan. "
                "For all BRCA2 carriers with mCRPC: PSMA-PET for response monitoring and lesion progression. "
                "CLINICAL NOTE: PSMA expression can be LOST in aggressive neuroendocrine differentiation "
                "(common in mCRPC after ADT) → FDG-PET may be superior in neuroendocrine-transformed PCa. "
                "NOT ROUTINELY INDICATED: CHEK2, HOXB13, NBN localised prostate cancer — "
                "standard CT+bone scan appropriate for initial staging in localised disease."
            ),
            "HOXB13 G84E — Unique Mechanism: Transcription Factor, Not DNA Repair": (
                "HOXB13 G84E is unique among hereditary prostate cancer genes because it is NOT a DNA repair gene: "
                "MECHANISM OF SUSCEPTIBILITY: "
                "G84E (p.Gly84Glu; rs138213197) is located in the conserved MEIS-binding interface of the HOXB13 "
                "homeodomain — Gly84 is structurally critical for normal MEIS protein interaction. "
                "G84E disrupts HOXB13-MEIS complex → altered prostate-specific transcriptional programme → "
                "enhanced AR co-stimulatory activity → dysregulated prostate epithelial differentiation. "
                "NOT a haploinsufficiency/LOF gene — G84E is a dominant susceptibility variant, "
                "not a tumour suppressor mutation. "
                "THERAPEUTIC IMPLICATIONS: "
                "HOXB13 tumours are NOT HRD-positive (no HR repair defect); "
                "PARPi mechanism requires HRD — HOXB13 prostate cancers lack HRD; "
                "PARPi therapy is contraindicated as targeted (mechanistically unjustified) therapy; "
                "offering PARPi based on hereditary prostate cancer history without HRR testing is an error. "
                "ANDROGEN RECEPTOR PATHWAY: "
                "HOXB13 co-regulates AR → ADT-directed therapy rational for metastatic HOXB13-PCa; "
                "enzalutamide/abiraterone sensitivity expected (same AR-driven biology). "
                "TESTING IN CLINICAL PRACTICE: "
                "G84E is the only validated pathogenic variant; hotspot testing sufficient; "
                "no MLPA or large-rearrangement testing indicated (point mutation only); "
                "population screening in Scandinavian males with prostate cancer family history is rational "
                "(2.5% carrier frequency in Sweden)."
            ),
            "NBN 657del5 — Nijmegen Breakage Syndrome and Hereditary Prostate Cancer": (
                "NBN encodes Nibrin (NBS1), the sensor/scaffold subunit of the MRN (MRE11-RAD50-NBN) "
                "complex — the first sensor of double-strand breaks (DSBs): "
                "MRN FUNCTION AT DSB: NBN FHA/BRCT1-2 domains recognise γH2AX-MDC1 → "
                "MRE11 nuclease performs 5' end resection → RAD50 bridges DSB ends → "
                "NBN C-terminus directly activates ATM kinase → ATM phosphorylation cascade. "
                "NBN 657del5 MUTATION: c.657_661delACAAA → p.Lys219GlufsX17 → "
                "C-terminal truncation eliminates MRE11-binding domain → "
                "hypomorphic MRN complex (partial — not complete — ATM activation retained); "
                "complete null is embryonic lethal (homozygous). "
                "NIJMEGEN BREAKAGE SYNDROME (BIALLELIC): "
                "Autosomal recessive; typical features: microcephaly at birth (progressive); "
                "bird-like face (beak-like nose; prominent mid-face; receding forehead + chin); "
                "severe combined immunodeficiency (recurrent sinopulmonary infections); "
                "chromosomal radiosensitivity (spontaneous SCEs; G2 checkpoint failure); "
                "cancer: B-cell lymphoma ~50% by age 20; medulloblastoma; breast cancer. "
                "HETEROZYGOUS NBN 657del5 CARRIERS: "
                "Prostate cancer: OR 3–4× (Polish case-control series, Cybulski 2004/2013); "
                "Breast cancer (female): OR ~2–3× (moderate risk); "
                "Carrier frequency: ~0.5–1% in Slavic European populations. "
                "CLINICAL MANAGEMENT OF HETEROZYGOUS CARRIERS: "
                "Annual PSA from age 40; "
                "Radiation sensitivity: intermediate (similar to ATM heterozygotes — impaired ATM activation); "
                "inform radiation oncologist before prostate RT planning; "
                "No approved targeted therapy; PARPi data extremely sparse for NBN-only prostate cancer. "
                "CASCADE TESTING: If NBS patient (biallelic) diagnosed → ALL relatives (parents = obligate carriers) "
                "need prostate + breast cancer counselling per carrier risk data."
            ),
        },
        "pharmacological_distinctions": [
            "BRCA2 vs BRCA1 olaparib response in prostate cancer: PROfound rPFS HR 0.22 (BRCA2) vs 0.82 (BRCA1) — clinically significant difference; counsel BRCA1 prostate patients that PARPi benefit is real but much weaker; do not conflate with BRCA2-equivalent response",
            "Olaparib vs niraparib in prostate cancer: Olaparib (PROfound) vs niraparib+abiraterone (AMPLITUDE/ELM-PC4) — different combination partners; niraparib is FDA approved in combination with abiraterone for HRR-altered mCRPC; olaparib as monotherapy post-enzalutamide/abiraterone; selection based on prior therapy",
            "PARPi-sensitive (BRCA2, BRCA1, ATM, PALB2) vs PARPi-NOT-approved (CHEK2, HOXB13): Do not extrapolate PARPi to non-HRD genes; CHEK2 and HOXB13 are NOT DNA repair genes with HRD; treating CHEK2/HOXB13-PCa with PARPi off-label without trial context is unjustified",
            "MSH2 dMMR pembrolizumab vs BRCA2 olaparib — mutually exclusive HRR/MMR pathways: Test for BOTH MMR deficiency (IHC) AND HRR status (NGS) in mCRPC; some tumours are dMMR AND BRCA2-mutated (both pathways altered); pembrolizumab first if MSI-H; PARPi if HRR only; sequential or combination trials ongoing",
            "PSMA-PET mandatory (BRCA2) vs optional (others): BRCA2 metastatic enrichment mandates PSMA-PET at staging and biochemical recurrence; ATM/BRCA1/PALB2 PSMA-PET by standard high-risk criteria; CHEK2/HOXB13/NBN localised phenotype — PSMA-PET per standard risk stratification (not hereditary-gene mandated)",
        ],
        "key_standards": [
            "NCCN Prostate Cancer Early Detection (v2.2024): annual PSA from age 40 for BRCA2/ATM/CHEK2/HOXB13/Lynch carriers; biopsy threshold PSA ≥3.0 ng/mL for BRCA2 carriers (vs ≥4.0 general population)",
            "FDA Olaparib mCRPC approval 2020 (PROfound): BRCA1/2 and ATM-altered mCRPC; label does not include CHEK2, HOXB13, or NBN as standalone indications",
            "FDA Pembrolizumab 2017 tumor-agnostic dMMR/MSI-H: MSH2-deficient prostate cancer confirmed dMMR → pembrolizumab regardless of line of therapy",
            "EAU Hereditary Prostate Cancer Guidelines 2023: BRCA2/ATM/HOXB13/CHEK2/MSH2 as established hereditary PCa genes; CASCADE testing of all first-degree relatives; multi-gene panel preferred over sequential testing",
            "IMPACT Study (Prostate Cancer UK): prospective study of BRCA1/2 carrier PSA surveillance; confirmed earlier diagnosis at higher proportion of clinically significant PCa vs unaffected controls; validates PSA screening from age 40",
        ],
    }
