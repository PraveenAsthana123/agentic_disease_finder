#!/usr/bin/env python3
"""Hereditary-HBOC-Atlas — Complete 8-Gene Hereditary Breast & Ovarian Cancer Atlas
BRCA1    (Breast cancer type 1 susceptibility; 1863 aa; 17q21.31; AD;
          HBOC-1 — 70% lifetime breast, 46% ovarian; RING/BRCT domain;
          olaparib/talazoparib PARP inhibitors; prophylactic mastectomy/BSO;
          seed SEED_BASE+0) ·
BRCA2    (Breast cancer type 2 susceptibility; 3418 aa; 13q12.3; AD;
          HBOC-2 — 69% lifetime breast, 25% ovarian; BRC repeats/OB-folds;
          olaparib/niraparib; pancreatic + prostate cancer risk;
          seed SEED_BASE+1) ·
PALB2    (Partner and localiser of BRCA2; 1186 aa; 16p12.2; AD;
          HBOC-PALB2 — 35–58% breast, ~5% ovarian; coiled-coil + WD40;
          PARP inhibitor sensitive; BRCA2-bridge function;
          seed SEED_BASE+2) ·
CHEK2    (Checkpoint kinase 2; 543 aa; 22q12.1; AD;
          Moderate-risk HBOC — 25–35% breast; c.1100delC European founder;
          no PARP inhibitor approved; intensive surveillance;
          seed SEED_BASE+3) ·
ATM      (Ataxia-telangiectasia mutated; 3056 aa; 11q22.3; AD heterozygous;
          Moderate-risk HBOC — 25–33% breast; biallelic = A-T syndrome;
          olaparib compassionate data; radiation sensitivity;
          seed SEED_BASE+4) ·
RAD51C   (RAD51 paralogue C; 376 aa; 17q22; AD;
          Ovarian-dominant HBOC — 10–14% ovarian, ~3% breast;
          PARP inhibitor + platinum sensitive; Fanconi pathway (FANCO);
          seed SEED_BASE+5) ·
RAD51D   (RAD51 paralogue D; 328 aa; 17q12; AD;
          Ovarian-dominant HBOC — 10–13% ovarian, ~3% breast;
          PARP inhibitor + platinum sensitive; Fanconi pathway (FANCS);
          seed SEED_BASE+6) ·
BRIP1    (BRCA1-interacting protein C-terminal helicase 1; 1249 aa; 17q23.2; AD;
          Ovarian-moderate HBOC — 5–10% ovarian, ~1–2% breast;
          cisplatin/platinum sensitivity; FANCJ; BRCA1-BRCT binding;
          seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1630–1637)
"""

import random

SEED_BASE = 1630

HBOC_GENES = [
    # ── BRCA1 — HBOC type 1 ──────────────────────────────────────────────────
    {
        "gene": "BRCA1",
        "protein": "BRCA1 — HBOC-1 AD — RING-BRCT Double-Strand Break Repair — 70pct Breast 46pct Ovarian — Olaparib-Talazoparib-PARPi — Prophylactic-Mastectomy-BSO",
        "alias": (
            "BRCA1; OMIM gene 113705; Hereditary Breast and Ovarian Cancer Syndrome 1 OMIM 604370; "
            "17q21.31; 1863 aa; ~220 kDa; AD haploinsufficiency. "
            "BRCA1 encodes a nuclear scaffold protein with multiple functional domains: "
            "N-terminal RING domain (E3 ubiquitin ligase; heterodimerises with BARD1) → "
            "ubiquitinates H2AK119 at DNA damage sites; "
            "central domain (SQ/TQ motifs phosphorylated by ATM/ATR; "
            "NLS for nuclear localisation; BACH1/FANCJ/BLM/PALB2 interaction); "
            "C-terminal tandem BRCT domains (phosphopeptide recognition; binds CtIP, BACH1, "
            "Abraxas; essential for cell cycle checkpoint signalling and HR repair). "
            "FUNCTION: Master regulator of homologous recombination (HR) DNA repair. "
            "At DSB: ATM/ATR → phospho-H2AX → MRN/53BP1 focal recruitment → "
            "BRCA1-PALB2-BRCA2-RAD51 axis → RAD51 nucleofilament on ssDNA → HR repair. "
            "G2/M checkpoint: BRCA1-CtIP promotes DSB end resection → "
            "RPA → ATRIP/ATR → CHK1 → CDC25C degradation → G2 arrest. "
            "LOSS OF FUNCTION: HR deficiency (HRD) → cells reliant on error-prone NHEJ → "
            "accumulation of deletions, translocations, chromosomal instability → "
            "BRCA1-signature mutational landscape (SBS3, ID6/ID8 insertions-deletions). "
            "LIFETIME CANCER RISKS (meta-analysis 2021): "
            "Breast: 72% (95% CI 65–79%) by age 80 (without intervention). "
            "Ovarian: 46% (95% CI 39–54%). "
            "Contralateral breast (after first breast cancer): 20% at 10 years, 40% at 20 years. "
            "Pancreatic: 2–3× relative risk (absolute lifetime ~3–5%). "
            "Cervical/uterine: no significant increase. "
            "Male breast cancer: <1% lifetime (versus BRCA2 where male breast cancer is 5–7%). "
            "Prostate: modest increase vs. BRCA2; rarely triggers dedicated prostate screening. "
            "PATHOGENIC VARIANTS: >4000 unique pathogenic/likely pathogenic variants in ClinVar. "
            "Founder populations: Ashkenazi Jewish c.68_69delAG (185delAG) + c.5266dupC (5382insC) — "
            "carrier frequency 1/40 Ashkenazi Jewish women; 1/400 general population. "
            "Portuguese c.2037del4 + c.3331_3334del; "
            "Dutch c.2985_2986del. "
            "Spectrum: frameshift (45%), nonsense (25%), large genomic rearrangements (10% — "
            "MLPA mandatory if sequencing negative in high-risk families), missense (20% — "
            "VUS problem is significant; functional assay (MAVE) being implemented). "
            "DIAGNOSTICS: "
            "Germline sequencing BRCA1+2 (NGS panel); "
            "MLPA for large rearrangements if NGS negative; "
            "somatic tumour testing identifies acquired BRCA1/2 loss in non-carriers; "
            "HRD score (genomic scarring: LOH + TAI + LST) — independent predictor of PARPi benefit. "
            "SURVEILLANCE (BRCA1 carriers without prophylactic surgery): "
            "Annual breast MRI from age 25 (mammography from 30); "
            "Annual transvaginal ultrasound + CA-125 (limited sensitivity for OC, not guideline-endorsed "
            "for surveillance — prophylactic BSO preferred); "
            "No proven ovarian cancer screening protocol — BSO is the recommended risk-reduction strategy. "
            "RISK-REDUCTION SURGERY: "
            "Risk-reducing bilateral salpingo-oophorectomy (RRBSO): "
            "Recommended at age 35–40 or completion of childbearing; "
            "reduces ovarian cancer risk by >96%; "
            "reduces breast cancer risk by 50% (pre-menopausal BSO). "
            "Prophylactic bilateral mastectomy (PBM): "
            "Reduces breast cancer risk by >95%; "
            "patient decision: must weigh reconstruction, psychological impact; "
            "nipple-sparing mastectomy oncologically safe in low-risk areas. "
            "CHEMOPREVENTION: "
            "Tamoxifen (5 years): 30% breast cancer risk reduction in ER+ tumours; "
            "limited benefit in BRCA1 (typically ER-negative tumours); more useful in BRCA2. "
            "Raloxifene, aromatase inhibitors: limited data specifically for BRCA1. "
            "TREATMENT OF BRCA1-ASSOCIATED BREAST/OVARIAN CANCER: "
            "PARP inhibitors (HRD exploited via synthetic lethality): "
            "Olaparib (Lynparza): approved gBRCA1/2 HER2-negative breast cancer (OlympiAD); "
            "approved gBRCA1/2 ovarian cancer (maintenance, SOLO-1/SOLO-2); "
            "300 mg oral BD; AEs: nausea, fatigue, anaemia, MDS/AML (rare). "
            "Talazoparib (Talzenna): approved gBRCA1/2 HER2-negative breast cancer (EMBRACA); "
            "1 mg oral OD; most potent PARP trapper. "
            "Niraparib (Zejula): approved gBRCA1/2 ovarian; also HRD-positive tumours. "
            "Rucaparib (Rubraca): approved gBRCA1/2 ovarian; "
            "PLATINUM SENSITIVITY: BRCA1/2 tumours are platinum-sensitive (HR-deficient); "
            "carboplatin/cisplatin often preferred in ovarian BRCA1 cancers; "
            "PARPi maintenance after platinum response is standard of care."
        ),
        "gene_class": "RING-BRCT scaffold / HR repair / tumour suppressor",
        "locus": "17q21.31",
        "aa": 1863,
        "kDa": 220,
        "omim_gene": "113705",
        "omim_disease": "HBOC OMIM 604370",
        "inheritance": "AD; haploinsufficiency; second hit (somatic LOH) in tumour",
        "n_patients": 40,
        "key_alerts": [
            "BRCA1-BREAST-TRIPLE-NEGATIVE: 70–80% of BRCA1-associated breast cancers are triple-negative (ER-/PR-/HER2-) — platinum/PARPi most relevant; endocrine therapy of limited benefit for the primary tumour",
            "BRCA1-BSO-AGE-35-40: Risk-reducing bilateral salpingo-oophorectomy by age 35–40 reduces ovarian cancer risk >96% AND reduces breast cancer risk 50% pre-menopausally; no proven OC surveillance alternative",
            "BRCA1-MLPA-MANDATORY: Large genomic rearrangements (exon deletions/duplications) account for 10% of BRCA1 pathogenic variants — MLPA is mandatory if sequencing is negative in a high-risk pedigree",
            "BRCA1-PARP-INHIBITOR-APPROVED: Olaparib and talazoparib are FDA/EMA approved for gBRCA1/2 metastatic HER2-negative breast cancer; olaparib/niraparib approved for gBRCA1/2 ovarian maintenance",
        ],
        "etiologies": {
            "Frameshift — haploinsufficiency (deletions/insertions)": 45,
            "Nonsense — haploinsufficiency": 25,
            "Large genomic rearrangement — MLPA-detected": 10,
            "Missense — RING/BRCT functional null": 12,
            "Splice site — exon skipping": 8,
        },
        "stats": {
            "pct_triple_negative_breast": 75,
            "mean_breast_dx_age_y": 42,
            "pct_ovarian_high_grade_serous": 95,
            "pct_prophylactic_bso_completed": 55,
            "pct_prophylactic_mastectomy_completed": 38,
            "pct_parp_inhibitor_received": 60,
            "mean_dx_age_y": 42,
            "mean_dx_delay_months": 8,
        },
        "dx_delay_distribution": {"<6 months": 42, "6–18 months": 35, "18–36 months": 15, ">36 months": 8},
    },
    # ── BRCA2 — HBOC type 2 ──────────────────────────────────────────────────
    {
        "gene": "BRCA2",
        "protein": "BRCA2 — HBOC-2 AD — BRC-Repeats-RAD51-Loading — 69pct Breast 25pct Ovarian — Pancreatic-Prostate-Male-Breast — Olaparib-Niraparib",
        "alias": (
            "BRCA2; OMIM gene 600185; Hereditary Breast and Ovarian Cancer Syndrome 2 OMIM 612555; "
            "13q12.3; 3418 aa; ~390 kDa; AD haploinsufficiency. "
            "BRCA2 is the largest human tumour suppressor. "
            "DOMAIN ARCHITECTURE: N-terminal transactivation domain (PALB2 binding via N-terminal coiled-coil); "
            "central BRC repeats (8 × BRC motifs, aa 1002–2667) — each BRC repeat directly binds "
            "one RAD51 monomer; C-terminal DNA-binding domain (OB-folds 1-3 and tower domain — "
            "ssDNA/dsDNA junction binding) + CTE (C-terminal extension — BRCA2 nuclear export regulation). "
            "FUNCTION: RAD51 recombinase loader at DNA double-strand breaks. "
            "MECHANISM: DSB → ATM → BRCA1-PALB2 → BRCA2 recruited → "
            "BRCA2 displaces RPA from ssDNA overhang → loads RAD51 monomers onto ssDNA → "
            "RAD51 nucleofilament forms → searches homologous template → strand invasion → HR repair. "
            "WITHOUT BRCA2: RAD51 loading fails → ssDNA vulnerable to NHEJ, SSA, BIR → "
            "chromosomal instability; BRCA2-signature (SBS3) accumulates. "
            "CANCER RISKS: "
            "Breast (female): 69% lifetime (95% CI 61–77%); peak incidence 40s–50s. "
            "Ovarian: 25% lifetime (95% CI 19–31%); later onset than BRCA1 (mean 54 vs 50 years). "
            "Male breast cancer: 5–7% lifetime (vs 0.1% general male population) — "
            "MANDATORY screening: annual MRI + mammogram from age 40. "
            "Pancreatic cancer: 4–7% lifetime (5–8× relative risk) — "
            "EUS/MRI surveillance from age 45 or 10 years before earliest family pancreatic cancer. "
            "Prostate cancer: 8× relative risk for lethal prostate cancer — "
            "annual PSA + DRE from age 40; olaparib approved (PROfound trial) for gBRCA2 mCRPC. "
            "Melanoma: 2× relative risk. "
            "PATHOGENIC VARIANTS: c.5946delT (6174delT) most common Ashkenazi Jewish founder "
            "(carrier frequency 1/100 Ashkenazi Jewish; 1/800 general); "
            "Icelandic 999del5 founder (carrier frequency 0.6% in Iceland). "
            "Large deletions: 5–8% of BRCA2 PVs → MLPA mandatory. "
            "VUS problem: large gene (27 coding exons, 10.3 kb CDS) → "
            "many missense VUS; functional classification via SatMutagenesis. "
            "EXON 11: largest exon (3.3 kb); contains most BRC repeats; "
            "many truncating PVs in exon 11 (moderate-risk truncations vs. high-risk based on position). "
            "CLINICAL DISTINCTIONS FROM BRCA1: "
            "Tumour phenotype: BRCA2 breast cancers are more often ER+ (55–60%) vs. BRCA1 (20%); "
            "endocrine therapy + PARPi combination rational for BRCA2. "
            "Ovarian cancer: BRCA2 ovarian cancers are EXCLUSIVELY high-grade serous (HGSC); "
            "onset 10 years later than BRCA1 (age 54 vs. 45). "
            "Male breast cancer: primarily BRCA2 (BRCA1 male breast cancer risk minimal). "
            "Pancreatic/prostate: BRCA2 >>> BRCA1 for these extra-mammary/ovarian risks. "
            "TREATMENT: "
            "PARPi: olaparib (OlympiAD breast; SOLO-1/SOLO-2 ovarian); "
            "niraparib (PRIMA/NOVA ovarian); "
            "rucaparib (ovarian); talazoparib (breast). "
            "Prostate: olaparib approved mCRPC gBRCA1/2 (PROfound 2020) — "
            "significant OS benefit vs. enzalutamide/abiraterone in BRCA2 subgroup; "
            "rucaparib (TRITON2/3) + niraparib + talazoparib — emerging data. "
            "Pancreatic: olaparib maintenance after platinum (POLO trial) — "
            "gBRCA1/2 pancreatic cancer after 16+ weeks platinum → olaparib maintenance "
            "doubles progression-free survival. "
            "SURVEILLANCE: identical breast MRI protocol to BRCA1; "
            "BSO recommended at 40–45 (later than BRCA1 given lower ovarian risk + ER+ tumours); "
            "annual PSA/DRE (male carriers, from age 40); "
            "pancreatic: EUS alternating with MRI/MRCP annually from age 45."
        ),
        "gene_class": "BRC-repeat RAD51 loader / HR repair / tumour suppressor",
        "locus": "13q12.3",
        "aa": 3418,
        "kDa": 390,
        "omim_gene": "600185",
        "omim_disease": "HBOC OMIM 612555",
        "inheritance": "AD; haploinsufficiency; biallelic = Fanconi anemia type D1 (FANCD1)",
        "n_patients": 40,
        "key_alerts": [
            "BRCA2-MALE-BREAST-CANCER-5-7pct: BRCA2 male breast cancer risk is 5–7% lifetime; annual breast MRI + mammogram mandatory for male carriers from age 40 — this risk is often missed in male carriers",
            "BRCA2-PANCREATIC-POLO-TRIAL: Olaparib maintenance after platinum in gBRCA1/2 pancreatic cancer (POLO 2019) doubles PFS — platinum + maintenance PARPi is standard of care for gBRCA2 pancreatic cancer",
            "BRCA2-PROSTATE-PROFOUND: Olaparib FDA-approved for mCRPC gBRCA1/2 (PROfound 2020); greatest benefit in BRCA2 subgroup — PSA monitoring + DRE from age 40 in male BRCA2 carriers",
            "BRCA2-ER-POSITIVE: 55–60% of BRCA2-associated breast cancers are ER+ — endocrine therapy is relevant (unlike BRCA1 triple-negative); olaparib + endocrine therapy combination trials ongoing",
        ],
        "etiologies": {
            "Frameshift — haploinsufficiency": 50,
            "Nonsense — haploinsufficiency": 22,
            "Large genomic rearrangement — MLPA-detected": 7,
            "Missense — BRC-repeat or OB-fold null": 14,
            "Splice site": 7,
        },
        "stats": {
            "pct_er_positive_breast": 58,
            "mean_breast_dx_age_y": 47,
            "pct_male_breast_cancer": 8,
            "pct_pancreatic_cancer": 5,
            "pct_prophylactic_bso_completed": 48,
            "pct_parp_inhibitor_received": 65,
            "mean_dx_age_y": 47,
            "mean_dx_delay_months": 10,
        },
        "dx_delay_distribution": {"<6 months": 38, "6–18 months": 37, "18–36 months": 17, ">36 months": 8},
    },
    # ── PALB2 — HBOC moderate-high ───────────────────────────────────────────
    {
        "gene": "PALB2",
        "protein": "PALB2 — HBOC-PALB2 AD — BRCA1-BRCA2 Bridge — 35-58pct Breast 5pct Ovarian — PARP-Inhibitor-Sensitive",
        "alias": (
            "PALB2; OMIM gene 610355; HBOC susceptibility OMIM 610355 (gene); "
            "16p12.2; 1186 aa; ~130 kDa; AD haploinsufficiency. "
            "PALB2 (Partner And Localiser of BRCA2) is the critical bridge between BRCA1 and BRCA2. "
            "DOMAIN ARCHITECTURE: N-terminal coiled-coil (CC) domain (aa 9–44) — binds BRCA1 coiled-coil; "
            "central region — MRG15 chromatin-remodeller interaction; "
            "WD40 repeat domain (aa 853–1186) — binds BRCA2 N-terminus and RAD51. "
            "FUNCTION: Without PALB2, BRCA2 cannot localise to nuclear DNA damage foci; "
            "PALB2 is the molecular bridge that ensures BRCA1-directed end resection is "
            "immediately channelled into BRCA2/RAD51-mediated HR. "
            "LOF: BRCA2 mislocalised → HR deficiency → identical HRD signature to BRCA1/2. "
            "Biallelic PALB2 = Fanconi anaemia type N (FANCN) — pancytopenia, congenital anomalies. "
            "CANCER RISKS (King 2014 + Antoniou 2014 + updated EMBRACE/CIMBA 2020): "
            "Breast: 35% (low-risk families) to 58% (high-risk families) lifetime; "
            "mean onset ~40–45 years; hazard ratio vs. general population 7.18. "
            "Ovarian: ~5% lifetime risk; much lower than BRCA1/2; "
            "clinical management guidelines still evolving (RRSO not uniformly recommended — "
            "some centres recommend at age 45–50 given cumulative ovarian risk). "
            "Pancreatic: 2–3× relative risk; same order as BRCA2. "
            "Male breast: data limited; ~1% lifetime (vs. 5–7% BRCA2). "
            "PATHOGENIC VARIANTS: c.1592delT (European founder); "
            "c.3113+1G>T (splice site); "
            "PALB2 truncating PVs account for ~1–2% of familial breast cancer. "
            "Large deletions: rare but possible — MLPA available. "
            "CLINICAL MANAGEMENT DIFFERENCES vs. BRCA1/2: "
            "Breast surveillance: same intensity as BRCA1 (annual MRI from age 25–30); "
            "RRSO timing: guideline bodies differ — NICE recommends discussion but not mandated until data mature; "
            "NCCN recommends consider RRSO at 45–50 for PALB2 carriers; "
            "Chemoprevention: tamoxifen data in PALB2 extrapolated from BRCA2-like ER+ phenotype. "
            "PARP INHIBITOR DATA: "
            "PALB2-associated tumours show HRD and platinum/PARPi sensitivity; "
            "OlympiAD/EMBRACA did NOT include PALB2 (non-BRCA1/2) — no specific label; "
            "ongoing basket trials (TBCRC048) show 82% ORR to olaparib in PALB2 metastatic breast — "
            "off-label use increasing; EMA/FDA indication extensions pending."
        ),
        "gene_class": "BRCA1-BRCA2 nuclear bridge / HR localisation / tumour suppressor",
        "locus": "16p12.2",
        "aa": 1186,
        "kDa": 130,
        "omim_gene": "610355",
        "omim_disease": "HBOC susceptibility OMIM 610355",
        "inheritance": "AD; haploinsufficiency; biallelic = Fanconi anaemia type N (FANCN)",
        "n_patients": 40,
        "key_alerts": [
            "PALB2-RISK-COMPARABLE-BRCA2: Breast cancer risk in high-risk PALB2 families (58%) is comparable to BRCA2 — PALB2 should be managed with BRCA1/2-level breast surveillance (annual MRI from age 25)",
            "PALB2-PARP-INHIBITOR-SENSITIVITY: PALB2-mutant tumours show HRD and olaparib sensitivity (TBCRC048: 82% ORR) — off-label PARPi appropriate in metastatic setting while awaiting formal approval",
            "PALB2-OVARIAN-UNCERTAIN: Ovarian cancer risk ~5% — RRSO not universally mandated; discuss individually at age 45–50; ongoing registry data accumulating; annual ultrasound + CA125 less effective than surgery",
            "PALB2-BRCA2-BRIDGE: Without PALB2, BRCA2 cannot localise to DNA damage — PALB2 LOF produces identical HRD genomic signature to BRCA1/2; platinum sensitivity expected",
        ],
        "etiologies": {
            "Frameshift — haploinsufficiency": 55,
            "Nonsense — haploinsufficiency": 28,
            "Splice site": 10,
            "Missense — CC/WD40 null": 7,
        },
        "stats": {
            "pct_breast_er_positive": 55,
            "mean_breast_dx_age_y": 43,
            "pct_parp_inhibitor_received": 40,
            "pct_rrso_completed": 32,
            "pct_bilateral_mastectomy_completed": 30,
            "mean_dx_age_y": 43,
            "mean_dx_delay_months": 12,
        },
        "dx_delay_distribution": {"<6 months": 35, "6–18 months": 38, "18–36 months": 18, ">36 months": 9},
    },
    # ── CHEK2 — moderate-risk HBOC ───────────────────────────────────────────
    {
        "gene": "CHEK2",
        "protein": "CHEK2 — Moderate-Risk-HBOC AD — CHK2-Kinase — 25-35pct Breast — c.1100delC-European-Founder — No-PARPi-Approved",
        "alias": (
            "CHEK2; OMIM gene 604373; Breast cancer susceptibility OMIM 604373; "
            "22q12.1; 543 aa; ~61 kDa; AD dominant-negative or haploinsufficiency. "
            "CHEK2 encodes Checkpoint Kinase 2 (CHK2), a serine/threonine kinase central to "
            "the DNA damage checkpoint pathway. "
            "DOMAIN ARCHITECTURE: FHA domain (aa 92–175, phosphopeptide binding — BRCA1/CDC25A docking); "
            "kinase domain (aa 220–486, SQ/TQ phosphorylation). "
            "ACTIVATION: ATM → T68 phosphorylation → CHK2 dimerisation → full activation → "
            "p53-Ser20 phosphorylation (stabilisation) + BRCA1-Ser988 phosphorylation + "
            "CDC25A phosphorylation (degradation → S-phase arrest). "
            "CANCER RISKS: "
            "Breast (female): 25–35% lifetime (vs. 12.5% general population); "
            "relative risk 2.7× for c.1100delC; higher risk with family history. "
            "Male breast cancer: 9% lifetime (≈ BRCA2 or higher; often overlooked in males); "
            "CHEK2 is the most common cause of male breast cancer after BRCA2. "
            "Colon cancer: 2–3× relative risk → colonoscopy from age 40. "
            "Thyroid (papillary): 3× relative risk. "
            "Prostate: 1.5–2× relative risk. "
            "Ovarian: not significantly elevated (key difference from BRCA1/2). "
            "PATHOGENIC VARIANTS: "
            "c.1100delC (p.Thr367Metfs): frameshift; European founder; "
            "carrier frequency 1% northern European; 0.5% USA European ancestry; "
            "most studied CHEK2 variant; classic moderate-risk allele. "
            "p.I157T (c.470T>C): missense; 4× more common than c.1100delC; "
            "LOWER risk per allele (~1.5× breast cancer); Polish/central European. "
            "c.1283C>T (p.Ser428Phe): splice site; uncertain classification in some labs. "
            "Large deletions (exon 9–10 deletion): 'del5395' — associated with highest absolute risk. "
            "c.319+2T>A: splice site; null allele. "
            "CLINICAL DISTINCTIONS (CHEK2 vs. BRCA1/2): "
            "NO APPROVED PARP INHIBITOR: CHEK2 tumours do NOT consistently show HRD; "
            "PARPi not indicated; platinum not first choice; "
            "endocrine therapy for ER+ tumours as standard. "
            "RRSO NOT RECOMMENDED: ovarian risk not elevated — BSO not indicated; "
            "this is a critical distinction from BRCA1/2 counselling. "
            "SURVEILLANCE: annual breast MRI from age 30–40 (age depends on personal/family risk); "
            "mammography annually from 40; "
            "contralateral surveillance after breast cancer equally important. "
            "MALE CHEK2 CARRIERS: breast cancer risk 9% — "
            "annual clinical breast exam + mammogram from age 40 or 10 years before first male relative's diagnosis. "
            "CHEMOPREVENTION: tamoxifen/raloxifene may reduce ER+ breast risk; "
            "CHEK2 breast cancers are predominantly ER+ (similar to sporadic). "
            "NO RRSO: ovarian cancer risk not increased — NEVER recommend BSO for CHEK2 alone."
        ),
        "gene_class": "CHK2 serine/threonine kinase / DNA damage checkpoint / tumour suppressor",
        "locus": "22q12.1",
        "aa": 543,
        "kDa": 61,
        "omim_gene": "604373",
        "omim_disease": "Breast cancer susceptibility OMIM 604373",
        "inheritance": "AD; haploinsufficiency or dominant-negative",
        "n_patients": 40,
        "key_alerts": [
            "CHEK2-NO-RRSO: CHEK2 does NOT elevate ovarian cancer risk — BSO/RRSO is NOT recommended for CHEK2 alone; this is the most critical distinction from BRCA1/2 counselling",
            "CHEK2-NO-PARP-INHIBITOR: CHEK2-mutant tumours do NOT reliably show HRD; PARPi are NOT indicated; treat ER+ breast cancer with endocrine therapy + standard chemotherapy",
            "CHEK2-MALE-BREAST-CANCER-9pct: CHEK2 male breast cancer risk 9% lifetime — often missed; male carriers need annual mammogram + clinical exam from age 40",
            "CHEK2-COLON-CANCER-2-3x: Colonoscopy from age 40 recommended for CHEK2 carriers — moderate colorectal cancer risk elevation requires surveillance beyond breast",
        ],
        "etiologies": {
            "c.1100delC frameshift — founder": 55,
            "p.I157T missense": 22,
            "Large deletion (del5395, exon 9–10)": 8,
            "Other frameshift/nonsense": 10,
            "Splice site": 5,
        },
        "stats": {
            "pct_er_positive_breast": 80,
            "mean_breast_dx_age_y": 50,
            "pct_male_breast_cancer": 7,
            "pct_rrso_incorrectly_offered": 15,
            "pct_parp_inhibitor_incorrectly_given": 12,
            "pct_colonoscopy_surveillance_done": 40,
            "mean_dx_age_y": 50,
            "mean_dx_delay_months": 14,
        },
        "dx_delay_distribution": {"<6 months": 30, "6–18 months": 40, "18–36 months": 20, ">36 months": 10},
    },
    # ── ATM — moderate-risk HBOC / A-T ───────────────────────────────────────
    {
        "gene": "ATM",
        "protein": "ATM — Moderate-Risk-HBOC AD-Heterozygous — 25-33pct Breast — Biallelic-Ataxia-Telangiectasia — Radiation-Sensitivity-PATHOGNOMONIC — Olaparib-Emerging",
        "alias": (
            "ATM; OMIM gene 607585; Ataxia-telangiectasia (biallelic) OMIM 208900; "
            "HBOC susceptibility (heterozygous) OMIM 114480; "
            "11q22.3; 3056 aa; ~350 kDa; AD haploinsufficiency (heterozygous) or AR (biallelic A-T). "
            "ATM (Ataxia-Telangiectasia Mutated) is the master kinase of the DNA damage response. "
            "DOMAIN ARCHITECTURE: FAT domain (FRAP-ATM-TRRAP; structural) + "
            "FATC domain (C-terminal; required for kinase activity) + "
            "PI3K-like kinase domain (phosphorylates >700 substrates in response to DSBs): "
            "H2AX-Ser139 (γH2AX — DSB marker); CHK2-Thr68; p53-Ser15; "
            "BRCA1-Ser1524; KAP1-Ser824; NBS1-Ser343; SMC1-Ser966. "
            "ACTIVATION: DSB → MRN complex (MRE11-RAD50-NBS1) → ATM monomer → "
            "autophosphorylation (Ser1981) → dimer → active monomer → "
            "phosphorylates >700 substrates → cell cycle arrest + DNA repair + apoptosis. "
            "CANCER RISKS (heterozygous ATM PV carriers): "
            "Breast: 25–33% lifetime (2–3× relative risk); ER+ predominance. "
            "Ovarian: no significant elevation (unlike BRCA1/2). "
            "Prostate: ~2× relative risk for aggressive disease. "
            "Pancreatic: ~5× relative risk. "
            "Colorectal: modest increase. "
            "BIALLELIC ATM = ATAXIA-TELANGIECTASIA (A-T syndrome): "
            "Progressive cerebellar ataxia (onset age 1–3); "
            "telangiectasias (conjunctival, facial — PATHOGNOMONIC by age 5); "
            "immune deficiency (IgA, IgG2 deficiency → recurrent sinopulmonary infections); "
            "radiation hypersensitivity (must inform radiotherapy team — fatal outcomes with standard doses); "
            "markedly elevated AFP (>10 ng/mL — diagnostic clue); "
            "lymphoma/leukaemia risk 70–100× general population (ALL, NHL, T-cell). "
            "HETEROZYGOUS CARRIERS (parents of A-T children): "
            "2–3× breast cancer risk elevation; radiation sensitivity intermediate (NOT absolute CI); "
            "carrier children of A-T parents warrant genetic counselling. "
            "RADIATION SENSITIVITY IN HETEROZYGOUS CARRIERS: "
            "Radiotherapy NOT absolutely contraindicated in ATM heterozygotes "
            "(unlike homozygotes where it can be fatal); "
            "however, RELATIVE hypersensitivity is established — "
            "standard fractionation generally safe but hypofractionation may increase toxicity; "
            "inform radiation oncologist of ATM heterozygous status. "
            "DIAGNOSTICS: NGS panel — ATM is large (66 exons); sequencing + MLPA; "
            "variant interpretation challenging (many missense VUS in large gene). "
            "TREATMENT: "
            "Breast: standard protocols; ER+ tumours → endocrine therapy; "
            "PARPi (olaparib) — PATHFINDER trial shows activity in ATM breast cancer; "
            "not yet formally approved for ATM (non-BRCA1/2); off-label basis. "
            "Prostate: olaparib PROfound trial included ATM — modest benefit (HR 0.82 vs. HR 0.34 for BRCA2); "
            "FDA label includes ATM for mCRPC; clinical effect weaker than BRCA2."
        ),
        "gene_class": "PI3K-like serine/threonine kinase / master DSB signalling / tumour suppressor",
        "locus": "11q22.3",
        "aa": 3056,
        "kDa": 350,
        "omim_gene": "607585",
        "omim_disease": "A-T OMIM 208900 (biallelic); HBOC susceptibility (heterozygous)",
        "inheritance": "AD haploinsufficiency (heterozygous) / AR (biallelic = A-T)",
        "n_patients": 40,
        "key_alerts": [
            "ATM-RADIATION-SENSITIVITY: ATM heterozygous carriers have intermediate radiation sensitivity — inform radiation oncologist before radiotherapy; fractionated RT generally safe but hypofractionation risk elevated",
            "ATM-BIALLELIC-A-T: Biallelic ATM = Ataxia-Telangiectasia — cerebellar ataxia + conjunctival telangiectasias (PATHOGNOMONIC) + elevated AFP >10 ng/mL + radiation hypersensitivity; standard RT doses can be fatal in A-T",
            "ATM-PARP-INHIBITOR-WEAKER: ATM-mutant cancers show PARPi sensitivity but weaker than BRCA1/2 (PROfound HR 0.82 for ATM vs. 0.34 for BRCA2 overall); olaparib FDA-approved for mCRPC ATM but clinical effect modest",
            "ATM-NO-OVARIAN-RISK: ATM heterozygosity does NOT significantly elevate ovarian cancer risk — RRSO not recommended; this distinguishes ATM from BRCA1/2 management",
        ],
        "etiologies": {
            "Frameshift — haploinsufficiency": 40,
            "Nonsense — haploinsufficiency": 28,
            "Missense — kinase domain null (truncating-equivalent)": 18,
            "Splice site — exon skipping": 10,
            "Large genomic rearrangement": 4,
        },
        "stats": {
            "pct_er_positive_breast": 72,
            "mean_breast_dx_age_y": 51,
            "pct_prostate_surveillance": 35,
            "pct_rt_toxicity_increased": 18,
            "mean_dx_age_y": 51,
            "mean_dx_delay_months": 16,
        },
        "dx_delay_distribution": {"<6 months": 28, "6–18 months": 38, "18–36 months": 22, ">36 months": 12},
    },
    # ── RAD51C — ovarian-dominant HBOC ───────────────────────────────────────
    {
        "gene": "RAD51C",
        "protein": "RAD51C — Ovarian-Dominant-HBOC AD — FANCO-Fanconi — 10-14pct Ovarian 3pct Breast — PARP-Inhibitor-Platinum-Sensitive",
        "alias": (
            "RAD51C; OMIM gene 602774; Ovarian cancer susceptibility OMIM 613399; "
            "Fanconi anaemia type O (FANCO) OMIM 613390 (biallelic); "
            "17q22; 376 aa; ~42 kDa; AD haploinsufficiency. "
            "RAD51C is one of five RAD51 paralogues (RAD51B, RAD51C, RAD51D, XRCC2, XRCC3) "
            "that function in the BCDX2 (RAD51B-RAD51C-RAD51D-XRCC2) and CX3 (RAD51C-XRCC3) "
            "complexes. "
            "FUNCTION: "
            "BCDX2 complex promotes RAD51 nucleofilament assembly (analogous to BRCA2 function "
            "but structurally distinct); "
            "CX3 complex resolves Holliday junctions during HR repair completion; "
            "RAD51C specifically: phosphorylated at Thr32 by ATM/ATR after DSB; "
            "FANCL-mediated monoubiquitination of FANCD2 requires intact RAD51C (FANCO pathway); "
            "RAD51C connects HR and Fanconi anaemia interstrand crosslink repair. "
            "LOF: HR deficiency → SBS3 signature → high-grade serous ovarian cancer susceptibility. "
            "Biallelic RAD51C = Fanconi anaemia type O (FANCO): "
            "bone marrow failure + congenital anomalies + early childhood leukaemia. "
            "CANCER RISKS: "
            "Ovarian: 10–14% lifetime (vs. <2% general population); "
            "primarily HIGH-GRADE SEROUS ovarian cancer (HGSC) — same histotype as BRCA1/2; "
            "mean onset age 53–58; "
            "relative risk vs. general population: 5.8× (Thompson 2012, CIMBA). "
            "Breast: ~3% lifetime (marginal elevation; not clearly above 2× threshold "
            "used for intensive surveillance in some guidelines). "
            "Ovarian dominance: RAD51C and RAD51D are the 'ovarian cancer genes' among HR paralogues — "
            "breast risk is modest; ovarian risk drives clinical management. "
            "MANAGEMENT: "
            "RRSO: recommended at age 45–50 (similar to BRCA2 schedule); "
            "clear evidence of ovarian cancer risk reduction. "
            "Breast surveillance: annual MRI may be offered given uncertain breast elevation; "
            "NCCN/NICE guidance evolving — most recommend offering breast MRI from age 40. "
            "TREATMENT: "
            "PARPi: RAD51C-mutant ovarian cancers show HRD signature and PARPi sensitivity; "
            "olaparib/niraparib sensitivity demonstrated in vitro and case series; "
            "formal PARPi label does not yet include RAD51C (non-BRCA1/2); "
            "treatment in HGSC context: platinum-based + PARPi maintenance (same protocol as HGSC); "
            "HRD testing by tumour should include RAD51C somatic LOH. "
            "Platinum: high platinum sensitivity expected — HGSC standard protocol appropriate."
        ),
        "gene_class": "RAD51 paralogue / BCDX2+CX3 complex / Fanconi-HR axis / tumour suppressor",
        "locus": "17q22",
        "aa": 376,
        "kDa": 42,
        "omim_gene": "602774",
        "omim_disease": "Ovarian cancer susceptibility OMIM 613399; FANCO OMIM 613390",
        "inheritance": "AD haploinsufficiency (heterozygous) / AR (biallelic = Fanconi anaemia type O)",
        "n_patients": 40,
        "key_alerts": [
            "RAD51C-OVARIAN-DOMINANT: RAD51C elevates ovarian cancer risk 10–14% (vs. breast ~3%) — RRSO at age 45–50 is the primary risk-reduction intervention; breast surveillance is secondary in RAD51C management",
            "RAD51C-PARP-INHIBITOR-SENSITIVE: RAD51C-mutant HGSC shows HRD and PARPi sensitivity — treat HGSC with platinum + PARPi maintenance; formal non-BRCA label expanding",
            "RAD51C-FANCONI-BIALLELIC: Biallelic RAD51C = Fanconi anaemia type O (FANCO) — bone marrow failure in childhood; heterozygous carriers are parents of FANCO children",
            "RAD51C-GENETIC-PANEL-ONLY: RAD51C is only diagnosable via multi-gene panel sequencing; not included in historic single-gene BRCA1/2 testing — diagnostic lag common in pre-panel era",
        ],
        "etiologies": {
            "Frameshift — haploinsufficiency": 48,
            "Nonsense — haploinsufficiency": 30,
            "Splice site": 14,
            "Missense — HR complex null": 8,
        },
        "stats": {
            "pct_ovarian_hgsc": 98,
            "mean_ovarian_dx_age_y": 55,
            "pct_rrso_completed": 40,
            "pct_platinum_response": 85,
            "pct_missed_by_single_gene_brca_testing": 100,
            "mean_dx_age_y": 55,
            "mean_dx_delay_months": 9,
        },
        "dx_delay_distribution": {"<6 months": 40, "6–18 months": 35, "18–36 months": 18, ">36 months": 7},
    },
    # ── RAD51D — ovarian-dominant HBOC ───────────────────────────────────────
    {
        "gene": "RAD51D",
        "protein": "RAD51D — Ovarian-Dominant-HBOC AD — FANCS-Fanconi — 10-13pct Ovarian — Platinum-PARPi-Sensitive",
        "alias": (
            "RAD51D; OMIM gene 602954; Ovarian cancer susceptibility OMIM 614291; "
            "Fanconi anaemia type S (FANCS) OMIM 617883 (biallelic); "
            "17q12; 328 aa; ~37 kDa; AD haploinsufficiency. "
            "RAD51D is the smallest of the RAD51 paralogues. "
            "FUNCTION: "
            "Part of BCDX2 complex (RAD51B-RAD51C-RAD51D-XRCC2) — "
            "promotes RAD51 nucleofilament assembly at resected DSBs; "
            "BCDX2 complex may act upstream of BRCA2 in RAD51 loading; "
            "RAD51D specifically required for: "
            "1. Telomere maintenance (telomeric HR); "
            "2. Replication fork protection under hydroxyurea-induced stress; "
            "3. Fanconi anaemia type S pathway (biallelic: FANCS). "
            "LOF: HR deficiency → high-grade serous ovarian cancer susceptibility "
            "(same mechanistic basis as RAD51C). "
            "CANCER RISKS (Loveday 2011 Nature Genetics; CIMBA 2020 replication): "
            "Ovarian: 10–13% lifetime (6.3× relative risk); "
            "high-grade serous dominant; "
            "Breast: 2–3% lifetime (no clear above-population threshold in some analyses; "
            "breast surveillance offered but not mandated at BRCA1/2 intensity by all guidelines). "
            "PATHOGENIC VARIANTS: "
            "c.904_907del4 (frameshift — Tyr302 region); "
            "c.694C>T (p.Arg232Ter); "
            "RAD51D truncating PVs are rare in population (~1/1500 carrier frequency). "
            "GENOTYPE-PHENOTYPE: no clear variant-specific modifiers identified. "
            "CLINICAL MANAGEMENT (same approach as RAD51C): "
            "RRSO: recommended at 45–50; primary risk-reduction strategy for ovarian cancer. "
            "Breast: annual MRI considered; absolute risk elevation modest. "
            "TREATMENT: "
            "HGSC standard: platinum-based combination + bevacizumab in front-line; "
            "PARPi maintenance: RAD51D-mutant tumours show HRD and PARPi sensitivity; "
            "olaparib/niraparib — same mechanism as BRCA1/2; label extension trials pending; "
            "clinical practice: treat HGSC with BRCA1/2-equivalent PARPi protocol if HRD-positive. "
            "DISTINGUISHING RAD51C FROM RAD51D: "
            "Clinically and mechanistically nearly identical; "
            "slightly lower ovarian risk for RAD51D vs. RAD51C (10% vs. 14%); "
            "both are BCDX2 complex members; "
            "both associated with Fanconi anaemia (FANCO and FANCS respectively); "
            "CX3 complex membership is unique to RAD51C (not RAD51D) — "
            "Holliday junction resolution function is RAD51C-specific."
        ),
        "gene_class": "RAD51 paralogue / BCDX2 complex / telomere maintenance / tumour suppressor",
        "locus": "17q12",
        "aa": 328,
        "kDa": 37,
        "omim_gene": "602954",
        "omim_disease": "Ovarian cancer susceptibility OMIM 614291; FANCS OMIM 617883",
        "inheritance": "AD haploinsufficiency (heterozygous) / AR (biallelic = Fanconi anaemia type S)",
        "n_patients": 40,
        "key_alerts": [
            "RAD51D-OVARIAN-DOMINANT: RAD51D elevates ovarian cancer risk 10–13% — RRSO at age 45–50 is the primary management strategy; breast risk elevation modest (2–3%)",
            "RAD51D-RAD51C-NEAR-IDENTICAL-MANAGEMENT: RAD51D and RAD51C have near-identical clinical management; RAD51C has additional CX3/Holliday junction function but practical patient management differs minimally",
            "RAD51D-PARP-INHIBITOR-SENSITIVE: RAD51D-mutant HGSC shows HRD — platinum + PARPi maintenance protocol (identical to BRCA1/2 HGSC) is appropriate practice pending formal label",
            "RAD51D-PANEL-ONLY-DIAGNOSIS: RAD51D is not in historic BRCA1/2-only testing — multi-gene panel mandatory for complete HBOC evaluation",
        ],
        "etiologies": {
            "Frameshift — haploinsufficiency": 50,
            "Nonsense — haploinsufficiency": 32,
            "Splice site": 12,
            "Missense — BCDX2 null": 6,
        },
        "stats": {
            "pct_ovarian_hgsc": 97,
            "mean_ovarian_dx_age_y": 57,
            "pct_rrso_completed": 37,
            "pct_platinum_response": 83,
            "mean_dx_age_y": 57,
            "mean_dx_delay_months": 10,
        },
        "dx_delay_distribution": {"<6 months": 39, "6–18 months": 35, "18–36 months": 19, ">36 months": 7},
    },
    # ── BRIP1 (FANCJ) — ovarian-moderate HBOC ─────────────────────────────────
    {
        "gene": "BRIP1",
        "protein": "BRIP1 — Ovarian-Moderate-HBOC AD — FANCJ-Helicase — 5-10pct Ovarian 1-2pct Breast — Cisplatin-Sensitive — BRCA1-BRCT-Binding",
        "alias": (
            "BRIP1 (BACH1, FANCJ); OMIM gene 605882; "
            "Ovarian cancer susceptibility OMIM 614327; "
            "Fanconi anaemia type J (FANCJ) OMIM 609054 (biallelic); "
            "17q23.2; 1249 aa; ~140 kDa; AD haploinsufficiency. "
            "BRIP1 (BRCA1-Interacting Protein 1) encodes a DNA helicase "
            "(also known as BACH1 — BRCA1-Associated C-terminal Helicase 1; and FANCJ). "
            "DOMAIN ARCHITECTURE: "
            "DEAD-box helicase domain (aa 1–855; Fe-S cluster required for helicase activity); "
            "C-terminal BRCT-binding motif (pThr1133; binds BRCA1-BRCT in phospho-dependent manner); "
            "RAD51 interaction domain. "
            "FUNCTION: "
            "1. G-quadruplex (G4) DNA unwinding — BRIP1 is the primary G4 helicase; "
            "G4 structures form at gene-rich regions, telomeres, promoters; "
            "failure to resolve G4 → replication stalling → DSB → genomic instability; "
            "2. BRCA1-BRCT interaction — BRIP1 acts downstream of BRCA1 in HR; "
            "Thr1133 phosphorylation recruits BRCA1 to BRIP1; "
            "3. Fanconi ICL repair — BRIP1 unwinds ICL flanking DNA → "
            "FANCD2-FANCI incision + translesion synthesis; "
            "biallelic LOF = Fanconi anaemia type J. "
            "CANCER RISKS (Ramus 2015 Journal of Medical Genetics; CIMBA): "
            "Ovarian: 5–10% lifetime (2–3× relative risk); "
            "LOWER absolute risk vs. BRCA1/2/RAD51C/RAD51D; "
            "high-grade serous dominant. "
            "Breast: ~1–2% lifetime (marginal; below clearly actionable threshold); "
            "original reports of breast cancer association not consistently replicated. "
            "MANAGEMENT: "
            "RRSO: NCCN includes BRIP1 as a consideration for RRSO at age 45–50 "
            "(given 5–10% OC risk); not universally mandated; risk-based individualisation; "
            "NICE (UK): RRSO for BRIP1 carriers is recommended if ovarian risk judged clinically significant. "
            "Breast: annual breast MRI NOT uniformly recommended (risk elevation minimal); "
            "clinical risk modelling (Tyrer-Cuzick) guides individual decisions. "
            "TREATMENT: "
            "Platinum sensitivity: BRIP1-mutant tumours accumulate ICL-repair defects → "
            "platinum (cisplatin/carboplatin) highly effective; "
            "PARPi: BRIP1 tumours may show modest PARPi sensitivity but data less robust than BRCA1/2; "
            "HRD testing guides PARPi maintenance; "
            "G4 helicase loss → CDN pathway activation — emerging G4-targeted therapies. "
            "DIAGNOSTIC NOTE: "
            "Fe-S cluster variants (e.g., missense in helicase core) may disrupt G4 unwinding "
            "without obvious frameshift — functional testing important for missense VUS; "
            "biallelic BRIP1 = Fanconi anaemia type J — all Fanconi patients' parents are obligate carriers."
        ),
        "gene_class": "FANCJ G4 helicase / BRCA1-BRCT interactor / Fanconi ICL repair / tumour suppressor",
        "locus": "17q23.2",
        "aa": 1249,
        "kDa": 140,
        "omim_gene": "605882",
        "omim_disease": "Ovarian cancer susceptibility OMIM 614327; FANCJ OMIM 609054",
        "inheritance": "AD haploinsufficiency (heterozygous) / AR (biallelic = Fanconi anaemia type J)",
        "n_patients": 40,
        "key_alerts": [
            "BRIP1-OVARIAN-MODERATE-RISK: BRIP1 ovarian cancer risk 5–10% — lower than RAD51C/RAD51D/BRCA1/2; RRSO at age 45–50 is discussable but not as clearly mandated; individualised risk counselling required",
            "BRIP1-G4-HELICASE: BRIP1 is the primary G4-DNA helicase — G-quadruplex unwinding failure → replication fork collapse → genomic instability; G4-targeted therapies in preclinical development",
            "BRIP1-CISPLATIN-SENSITIVE: BRIP1-mutant HGSC shows platinum sensitivity (ICL repair defect) — carboplatin/cisplatin preferred; PARPi benefit less established than RAD51C/RAD51D",
            "BRIP1-BREAST-RISK-NOT-ACTIONABLE: BRIP1 breast cancer risk elevation is modest and inconsistently replicated — annual breast MRI is NOT uniformly recommended; treat breast risk based on personal/family history modelling",
        ],
        "etiologies": {
            "Frameshift — haploinsufficiency": 52,
            "Nonsense — haploinsufficiency": 28,
            "Fe-S cluster missense — helicase null": 10,
            "Splice site": 8,
            "Large deletion": 2,
        },
        "stats": {
            "pct_ovarian_hgsc": 90,
            "mean_ovarian_dx_age_y": 58,
            "pct_rrso_completed": 28,
            "pct_platinum_response": 80,
            "pct_breast_mri_offered_unnecessarily": 20,
            "mean_dx_age_y": 58,
            "mean_dx_delay_months": 11,
        },
        "dx_delay_distribution": {"<6 months": 38, "6–18 months": 36, "18–36 months": 18, ">36 months": 8},
    },
]


def _make_cohort():
    cohort = {}
    for i, gene_info in enumerate(HBOC_GENES):
        seed = SEED_BASE + i
        rng = random.Random(seed)
        gene = gene_info["gene"]
        n = gene_info["n_patients"]
        patients = []
        for p in range(n):
            age_dx = round(rng.gauss(gene_info["stats"].get("mean_dx_age_y", 48), 10), 1)
            age_dx = max(18, min(85, age_dx))
            dx_delay = round(rng.gauss(gene_info["stats"].get("mean_dx_delay_months", 11), 5), 1)
            dx_delay = max(1.0, min(60, dx_delay))
            patients.append({
                "patient_id": f"{gene}-{seed}-{p+1:03d}",
                "gene": gene,
                "age_at_diagnosis": age_dx,
                "diagnosis_delay_months": dx_delay,
                "seed": seed,
            })
        cohort[gene] = {
            **gene_info,
            "patients": patients,
        }
    return cohort


_COHORT = _make_cohort()


# ─── API functions ─────────────────────────────────────────────────────────────

def get_overview():
    total = sum(v["n_patients"] for v in _COHORT.values())
    mean_dx_age = round(
        sum(p["age_at_diagnosis"] for v in _COHORT.values() for p in v["patients"]) / total, 1
    )
    mean_dx_delay = round(
        sum(p["diagnosis_delay_months"] for v in _COHORT.values() for p in v["patients"]) / total, 1
    )

    top_alerts = []
    for v in _COHORT.values():
        top_alerts.extend(v["key_alerts"][:2])

    genes_summary = []
    for g, v in _COHORT.items():
        pts = v["patients"]
        mean_age = round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1)
        genes_summary.append({
            "gene": g,
            "protein_short": v["protein"][:80],
            "locus": v["locus"],
            "inheritance": v["inheritance"].split(";")[0],
            "omim_disease": v["omim_disease"],
            "mean_dx_age": mean_age,
            "n_patients": v["n_patients"],
        })

    brca1 = _COHORT["BRCA1"]["stats"]
    brca2 = _COHORT["BRCA2"]["stats"]
    chek2 = _COHORT["CHEK2"]["stats"]
    rad51c = _COHORT["RAD51C"]["stats"]

    return {
        "atlas": "Hereditary-HBOC-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Breast & Ovarian Cancer Reference",
        "genes": genes_summary,
        "aggregate_stats": {
            "total_patients": total,
            "mean_dx_age_years": mean_dx_age,
            "mean_dx_delay_months": mean_dx_delay,
            "brca1_triple_negative_pct": brca1["pct_triple_negative_breast"],
            "brca1_bso_completed_pct": brca1["pct_prophylactic_bso_completed"],
            "brca1_parp_inhibitor_pct": brca1["pct_parp_inhibitor_received"],
            "brca2_er_positive_pct": brca2["pct_er_positive_breast"],
            "brca2_male_breast_pct": brca2["pct_male_breast_cancer"],
            "chek2_no_rrso_warranted_pct": 100,
            "chek2_rrso_incorrectly_offered_pct": chek2["pct_rrso_incorrectly_offered"],
            "rad51c_ovarian_hgsc_pct": rad51c["pct_ovarian_hgsc"],
            "rad51c_platinum_response_pct": rad51c["pct_platinum_response"],
            "cascade_tested_pct": 58,
        },
        "top_alerts": top_alerts[:16],
        "seeds": list(range(SEED_BASE, SEED_BASE + 8)),
    }


def get_breakdown():
    result = {}
    for gene, info in _COHORT.items():
        pts = info["patients"]
        result[gene] = {
            "gene": gene,
            "n_patients": info["n_patients"],
            "alias": info["alias"],
            "gene_class": info["gene_class"],
            "locus": info["locus"],
            "aa": info["aa"],
            "kDa": info["kDa"],
            "omim_gene": info["omim_gene"],
            "omim_disease": info["omim_disease"],
            "inheritance": info["inheritance"],
            "key_alerts": info["key_alerts"],
            "etiologies": info["etiologies"],
            "stats": info["stats"],
            "dx_delay_distribution": info["dx_delay_distribution"],
            "patients": pts[:10],
        }
    return result


def get_definitions():
    return {
        "atlas": "Hereditary-HBOC-Atlas",
        "concepts": {
            "HBOC Gene Tiers — Risk-Stratified Clinical Management": (
                "Hereditary Breast and Ovarian Cancer (HBOC) genes are divided into risk tiers: "
                "HIGH-RISK GENES (BRCA1, BRCA2, PALB2): "
                "Breast risk ≥35–70% lifetime; annual breast MRI from age 25–30; "
                "prophylactic mastectomy reduces breast risk >95%; "
                "BRCA1/BRCA2: RRSO recommended age 35–45; PALB2: RRSO consideration age 45–50; "
                "BRCA1/BRCA2 PARP inhibitors (olaparib/talazoparib/niraparib) are FDA/EMA approved; "
                "PALB2 PARPi expanding (TBCRC048 data). "
                "MODERATE-RISK GENES (CHEK2, ATM): "
                "Breast risk 25–35% lifetime; annual MRI from age 30–40 (age depends on family history); "
                "CHEK2 and ATM do NOT significantly elevate ovarian cancer risk — RRSO is NOT indicated; "
                "NO approved PARPi for CHEK2; ATM has modest PARPi benefit (PROfound) but weaker than BRCA2; "
                "endocrine therapy for ER+ breast cancers (both CHEK2 and ATM tumours are predominantly ER+). "
                "OVARIAN-DOMINANT GENES (RAD51C, RAD51D, BRIP1): "
                "Ovarian risk 5–14% lifetime; breast risk modest (<3–5%); "
                "RRSO recommended at age 45–50 for RAD51C and RAD51D; "
                "BRIP1 RRSO individualised (risk 5–10% — discussable, not mandated); "
                "PARPi sensitivity expected (HRD) but formal label limited to BRCA1/2 currently; "
                "platinum sensitivity high for all three — HGSC standard protocol applies."
            ),
            "Homologous Recombination Deficiency (HRD) — PARP Inhibitor Mechanism": (
                "Homologous recombination (HR) is the high-fidelity DNA double-strand break repair pathway. "
                "HRD = loss of HR → tumour cells rely on error-prone NHEJ and PARP1-dependent repair. "
                "PARP INHIBITOR MECHANISM (synthetic lethality): "
                "PARP1/2 normally repairs single-strand breaks (SSBs); "
                "PARPi traps PARP on SSB → SSB cannot be ligated → replication fork encounters SSB → "
                "DSB generated during replication → cell attempts HR → HR is ABSENT (BRCA1/2/PALB2/RAD51C/D LOF) → "
                "DSB cannot be repaired → mitotic catastrophe → cell death. "
                "Normal (non-tumour) cells survive PARPi because one functional allele remains (heterozygous carrier). "
                "TUMOUR CELLS: second hit (somatic LOH, promoter methylation, second mutation) eliminates "
                "the second allele → complete HR loss → PARPi synthetic lethality. "
                "HRD SCORES (genomic scarring): "
                "LOH score: loss of heterozygosity segments; "
                "TAI score: telomeric allelic imbalance; "
                "LST score: large-scale state transitions; "
                "Myraid Genetics HRD score ≥42 (or BRCA status positive) = HRD-positive; "
                "Foundation Medicine HRD + FoundationOne CDx measures LOH; "
                "HRD-positive tumours (regardless of BRCA status) respond to PARPi. "
                "GENES WITH HRD TUMOUR PROFILE: BRCA1, BRCA2, PALB2, RAD51C, RAD51D, BRIP1, ATM (weaker). "
                "GENES WITHOUT CONSISTENT HRD: CHEK2 — treat as standard ER+ breast cancer."
            ),
            "BRCA1 vs. BRCA2 Clinical Distinctions — High-Yield Table": (
                "BREAST CANCER HISTOTYPE: "
                "BRCA1: 70–80% triple-negative (ER-/PR-/HER2-) → PARPi + platinum most relevant, "
                "endocrine therapy only for rare ER+ tumours; "
                "BRCA2: 55–60% ER-positive → endocrine therapy + PARPi combination trials; "
                "tamoxifen/aromatase inhibitor indicated; "
                "PALB2: 55% ER-positive (BRCA2-like). "
                "MALE BREAST CANCER: "
                "BRCA2: 5–7% lifetime; annual mammogram + MRI from age 40 mandatory; "
                "BRCA1: <1% (marginal); CHEK2: ~9% — underrecognised. "
                "OVARIAN CANCER: "
                "BRCA1: 46% lifetime, onset 45–50; BRCA2: 25% lifetime, onset 50–55; "
                "PALB2: ~5% (uncertain); CHEK2/ATM: NOT elevated. "
                "RRSO TIMING: BRCA1: 35–40; BRCA2/PALB2: 40–45; RAD51C/RAD51D: 45–50; "
                "CHEK2/ATM: NOT recommended. "
                "EXTRA-BREAST/OVARIAN RISKS: "
                "BRCA2: pancreatic 4–7%, prostate 8×; "
                "BRCA1: pancreatic 2–3× only; "
                "ATM: prostate 2×, pancreatic 5×; "
                "CHEK2: colon 2–3×, male breast 9%, thyroid 3×."
            ),
            "Risk-Reducing Bilateral Salpingo-Oophorectomy (RRBSO) — Gene-by-Gene Guide": (
                "RRBSO completely prevents ovarian/fallopian tube cancer and reduces breast cancer risk "
                "by ~50% if performed pre-menopausally (by removing the primary source of oestrogen). "
                "TIMING BY GENE: "
                "BRCA1: Age 35–40 (high OC risk with early onset — single most impactful intervention); "
                "BRCA2: Age 40–45 (OC onset 10 years later; pre-menopausal removal still reduces breast risk); "
                "PALB2: Age 45–50 (OC risk ~5%; individualised decision; NCCN recommendation); "
                "RAD51C: Age 45–50 (10–14% OC risk; RRSO evidence-based); "
                "RAD51D: Age 45–50 (10–13% OC risk; same rationale as RAD51C); "
                "BRIP1: Age 45–50 (5–10% OC risk; NICE recommends if clinically significant; NCCN discuss); "
                "CHEK2: NOT RECOMMENDED (OC risk not elevated); "
                "ATM: NOT RECOMMENDED (OC risk not elevated). "
                "PROCEDURE: laparoscopic bilateral salpingo-oophorectomy including the entire fallopian tube "
                "(to prevent primary peritoneal or fallopian tube cancer); "
                "hysterectomy not routinely added (only if other indication); "
                "pre-operative transvaginal ultrasound + CA-125 (to exclude occult malignancy); "
                "pathology: serial sectioning of fimbriated end of fallopian tube mandatory "
                "(SEE protocol — Sectioning and Extensively Examining fimbriated end); "
                "occult carcinoma in fallopian tube found in 2–4% of BRCA1/2 RRSO specimens. "
                "POST-RRSO: Hormone replacement therapy (HRT) to age 50 is safe and recommended "
                "(reverses surgical menopause symptoms; does not negate breast risk reduction from RRSO). "
                "ONGOING SURVEILLANCE AFTER RRSO: "
                "Ovarian: no further OC surveillance needed (risk dramatically reduced); "
                "Breast: continue annual MRI ± mammogram per gene-specific guidance."
            ),
        },
        "pharmacological_distinctions": [
            "Olaparib vs. Talazoparib (breast cancer): Both PARPi approved for gBRCA1/2 HER2-negative breast cancer; Talazoparib (EMBRACA) is the more potent PARP trapper; Olaparib (OlympiAD) allows combination with endocrine therapy; selection based on tolerability (talazoparib: haematologic toxicity; olaparib: GI dominant)",
            "BRCA1 triple-negative vs. BRCA2 ER-positive treatment: BRCA1 breast cancers are 75% triple-negative — endocrine therapy of limited benefit for the primary tumour; olaparib/talazoparib + chemotherapy; BRCA2 are 58% ER+ — add endocrine therapy (tamoxifen/AI) after PARPi; combination olaparib + endocrine therapy trials ongoing",
            "PARPi approved (BRCA1/2) vs. off-label (PALB2/RAD51C/RAD51D/BRIP1): Only BRCA1/2 carry full FDA/EMA PARPi label for breast + ovarian; PALB2 has strongest emerging data (TBCRC048 82% ORR); RAD51C/RAD51D/BRIP1 — treat HRD-positive HGSC with platinum + PARPi maintenance by analogy with BRCA; formal label expansion expected",
            "CHEK2 vs BRCA-like management error: CHEK2 tumours are predominantly ER+ without HRD — PARPi is NOT indicated; RRSO is NOT indicated; offering PARPi or RRSO to CHEK2 carriers (without additional BRCA mutation) is a management error arising from miscategorisation",
            "ATM radiation sensitivity — heterozygous vs. biallelic: Biallelic ATM (A-T syndrome): standard RT doses can be fatal — absolute contraindication; ATM heterozygotes: intermediate sensitivity — standard fractionated RT generally safe; inform radiation oncologist; hypofractionation may increase normal tissue toxicity; this is a spectrum, not a binary",
        ],
        "key_standards": [
            "NCCN Genetic/Familial High-Risk Assessment: Breast, Ovarian, Pancreatic (v2.2024): BRCA1/2 surveillance and surgical management guidelines; PALB2/CHEK2/ATM/RAD51C/RAD51D/BRIP1 included in multi-gene panel recommendations",
            "NICE (UK) Familial Breast Cancer CG164 + BRCAExchange: BRCA1/2 testing criteria, PALB2 inclusion, RAD51C/RAD51D/BRIP1 RRSO guidance; CHEK2 risk modification",
            "CASCADE TESTING: ALL first-degree relatives of any HBOC gene carrier require genetic counselling; multi-gene panel preferred over sequential single-gene testing to capture all variants in one test",
            "MULTI-GENE PANEL MANDATORY: Historic BRCA1/2-only testing misses PALB2, CHEK2, ATM, RAD51C, RAD51D, BRIP1 — all clinically actionable genes; combined panel testing is now standard of care per NCCN/ESMO/ASCO",
            "HRD TESTING IN TUMOUR: For all HGSC ovarian cancers and HER2-negative breast cancers, somatic HRD testing (Foundation Medicine/Myriad Genomic) guides PARPi maintenance regardless of germline result",
        ],
    }
