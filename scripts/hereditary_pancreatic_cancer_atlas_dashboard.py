#!/usr/bin/env python3
"""Hereditary-Pancreatic-Cancer-Atlas — Complete 8-Gene Hereditary Pancreatic Cancer Atlas
BRCA2   (Breast cancer type 2 susceptibility; 3418 aa; 13q12.3; AD;
         HPANCA-1 — 5–10× pancreatic RR; POLO trial olaparib FDA 2019;
         cisplatin-based chemotherapy preferred upfront; PSMA-PET for staging;
         seed SEED_BASE+0) ·
CDKN2A  (Cyclin-dependent kinase inhibitor 2A; 156 aa [p16-INK4A] / 132 aa [p14-ARF]; 9p21.3; AD;
         HPANCA-2 — FAMMM syndrome; 25–58% lifetime pancreatic risk (highest single-gene risk);
         atypical moles + melanoma; no approved targeted therapy; EUS+MRI surveillance;
         seed SEED_BASE+1) ·
ATM     (Ataxia-telangiectasia mutated; 3056 aa; 11q22.3; AD;
         HPANCA-3 — 4–6% lifetime pancreatic risk (~6× RR); platinum-sensitive;
         PARPi emerging; radiation sensitivity intermediate;
         seed SEED_BASE+2) ·
STK11   (Serine/threonine kinase 11 / LKB1; 433 aa; 19p13.3; AD;
         HPANCA-4 — Peutz-Jeghers syndrome; 36% lifetime pancreatic risk;
         hamartomatous GI polyps; mTOR pathway; surveillance from age 25–30;
         seed SEED_BASE+3) ·
BRCA1   (Breast cancer type 1 susceptibility; 1863 aa; 17q21.31; AD;
         HPANCA-5 — ~2% lifetime pancreatic risk; weaker than BRCA2; POLO data limited;
         PARPi-sensitive but less evidence than BRCA2;
         seed SEED_BASE+4) ·
PALB2   (Partner and localiser of BRCA2; 1186 aa; 16p12.2; AD;
         HPANCA-6 — 2–4% lifetime pancreatic risk; POLO emerging PARPi sensitivity;
         BRCA2 functional partner in homologous recombination;
         seed SEED_BASE+5) ·
PRSS1   (Serine protease 1 / Cationic trypsinogen; 247 aa; 7q34; AD;
         HPANCA-7 — hereditary pancreatitis → 40–57× pancreatic cancer RR;
         R122H + N29I gain-of-function; TPIAT for severe cases; smoking ABSOLUTE-CI;
         seed SEED_BASE+6) ·
MLH1    (MutL protein homologue 1; 756 aa; 3p22.2; AD;
         HPANCA-8 — Lynch syndrome; 3–4% lifetime pancreatic risk; dMMR → pembrolizumab;
         IHC mandatory in Lynch pancreatic cancers; EPCAM-MLH1 silencing;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1646–1653)
"""

import random

SEED_BASE = 1646

PANCREATIC_GENES = [
    # ── BRCA2 — HPANCA-1 ──────────────────────────────────────────────────────
    {
        "gene": "BRCA2",
        "protein": "BRCA2 — HPANCA-1 AD — RAD51-Loader BRC-Repeat Scaffold — 5–10× Pancreatic RR — POLO Olaparib FDA 2019 — Cisplatin-Based Preferred Upfront",
        "alias": (
            "BRCA2; OMIM gene 600185; Hereditary Breast and Ovarian Cancer 2 OMIM 612555; "
            "also: Fanconi anaemia complementation group D1 (FANCD1) biallelic. "
            "13q12.3; 3418 aa; ~384 kDa; AD haploinsufficiency. "
            "BRCA2 is the most common hereditary pancreatic cancer gene: "
            "~5–10% of all pancreatic ductal adenocarcinomas (PDAC) carry somatic or germline BRCA2 alterations; "
            "germline BRCA2 mutations account for 5–7% of hereditary pancreatic cancers. "
            "FUNCTION: RAD51 loader — eight BRC repeats (residues 1002–2085) bind RAD51 monomers; "
            "BRCA2 displaces RPA from resected ssDNA, installs RAD51 nucleofilament → "
            "homologous recombination (HR) DSB repair; loss → HR deficiency (HRD) → "
            "replication fork collapse → PDAC development via genomic instability. "
            "PANCREATIC CANCER RISK (BRCA2 germline carriers): "
            "Lifetime risk: ~5–7% (men); ~4–6% (women); relative risk ~5–10× general population. "
            "Age of onset: median ~68 years (similar to sporadic PDAC); carriers may present 3–5 years earlier. "
            "PANCREATIC CANCER IN FAMILY HISTORY: If two or more first-degree relatives with PDAC, "
            "BRCA2 germline prevalence increases to ~7–12% of familial pancreatic cancer (FPC) kindreds. "
            "POLO TRIAL (Golan 2019, NEJM — pivotal): "
            "Phase III trial in BRCA1/2 germline-mutated metastatic PDAC; "
            "maintenance olaparib after platinum-based first-line therapy (≥16 weeks platinum without progression). "
            "Olaparib vs placebo: rPFS 7.4 vs 3.8 months; HR 0.53 (p=0.004). "
            "OS: 18.9 vs 18.1 months (no significant difference — crossover confounds). "
            "FDA APPROVAL 2019: Olaparib maintenance for gBRCA1/2-mutated metastatic PDAC. "
            "FIRST-LINE TREATMENT: FOLFIRINOX or gemcitabine+nab-paclitaxel; "
            "Cisplatin + gemcitabine (CisGem) for BRCA2-mutated PDAC: superior response rate vs gem alone "
            "(ORR ~40–58% in HRD/BRCA-mutated PDAC) — cisplatin-induced ICL preferred backbone. "
            "SURVEILLANCE: Annual EUS (endoscopic ultrasonography) + MRI pancreas from age 50 "
            "or 10 years younger than youngest family member with PDAC. "
            "GERMLINE TESTING: Cascade testing of all first-degree relatives; "
            "somatic BRCA2 testing in all metastatic PDAC patients (germline or somatic HRD guides PARPi). "
            "REVERSION MUTATIONS: ctDNA BRCA2 reversion in ~20–30% post-olaparib resistance; "
            "monitor ctDNA for resistance mechanism emergence."
        ),
        "locus": "13q12.3",
        "aa": 3418,
        "kDa": 384,
        "omim_gene": "600185",
        "omim_disease": "Hereditary Breast and Ovarian Cancer Syndrome 2 (OMIM 612555); Fanconi Anaemia D1 (biallelic)",
        "inheritance": "AD haploinsufficiency; biallelic = FANCD1 (Fanconi anaemia + medulloblastoma + Wilms)",
        "gene_class": "HR DNA repair — RAD51 loader; homologous recombination scaffold",
        "key_alerts": [
            "BRCA2-PANCREATIC-5-10X-RR: Germline BRCA2 → annual EUS+MRI from age 50 (or age earliest family PDAC -10 years)",
            "BRCA2-POLO-OLAPARIB-FDA2019: Maintenance olaparib after platinum response in gBRCA1/2 metastatic PDAC — FDA approved",
            "BRCA2-CISPLATIN-PREFERRED-UPFRONT: CisGem or modified FOLFIRINOX with cisplatin backbone preferred in BRCA2 PDAC (HRD exploits ICL)",
            "BRCA2-SOMATIC-REVERSION-RESISTANCE: Acquired platinum/PARPi resistance via BRCA2 reversion mutations — ctDNA monitoring",
            "BRCA2-CASCADE-RELATIVES: All first-degree relatives require germline testing and pancreatic surveillance if positive",
        ],
        "etiologies": [
            "Germline BRCA2 haploinsufficiency → HRD → PDAC via genomic instability",
            "Somatic second-hit loss of heterozygosity (LOH) in tumour — biallelic BRCA2 loss in cancer",
            "Founder mutations: c.6174delT (Ashkenazi Jewish), c.886delGT, large rearrangements (5% — MLPA mandatory)",
            "BRCA2-mutated PDAC phenotype: cisplatin-ICL and PARPi sensitive; HRD signature SBS3",
        ],
        "stats": {
            "mean_dx_age": 65,
            "mean_dx_delay_months": 4,
            "pancreatic_rr": "5–10×",
            "lifetime_risk_pct": 6,
            "polo_pfs_hr": 0.53,
            "polo_olaparib_approved": True,
            "cisplatin_preferred": True,
            "surveillance_eus_mri": True,
        },
        "dx_delay_distribution": "4–8 months (late presentation — locally advanced or metastatic at diagnosis in ~80% PDAC)",
    },
    # ── CDKN2A — HPANCA-2 ────────────────────────────────────────────────────
    {
        "gene": "CDKN2A",
        "protein": "CDKN2A — HPANCA-2 AD — p16-INK4A / p14-ARF Dual-Transcript — 25–58% Lifetime Pancreatic Risk — FAMMM Syndrome — No Approved Targeted Therapy",
        "alias": (
            "CDKN2A; OMIM gene 600160; Familial Atypical Multiple Mole Melanoma-Pancreatic Carcinoma syndrome "
            "(FAMMM-PC) OMIM 606719; also: Familial Melanoma. "
            "9p21.3; p16-INK4A 156 aa / p14-ARF 132 aa (alternate reading frames); AD. "
            "CDKN2A is unique: the same locus encodes two structurally unrelated tumour suppressors "
            "via alternative reading frames and promoters: "
            "p16-INK4A (exons 1α, 2, 3): inhibitor of CDK4 and CDK6 → "
            "blocks CDK4/6-cyclin D1 → prevents pRb phosphorylation → G1 arrest; "
            "p14-ARF (exons 1β, 2, 3): binds MDM2 → prevents MDM2-mediated p53 ubiquitylation → "
            "stabilises p53 → apoptosis and senescence. "
            "LOSS OF FUNCTION: Both pathways lost simultaneously at 9p21 deletion → "
            "CDK4/6 unchecked (bypasses G1) AND p53 destabilised → dual oncogenic hit. "
            "FAMMM-PC SYNDROME: "
            "Clinical triad: (1) atypical (dysplastic) melanocytic naevi (≥50 atypical moles); "
            "(2) melanoma (cutaneous and ocular); (3) pancreatic ductal adenocarcinoma. "
            "Pancreatic cancer lifetime risk: 25–58% in CDKN2A germline carriers "
            "(varies by study: LEIDEN study ~44%; Dutch CDKN2A founder studies ~17% at age 75; "
            "in families with PDAC in pedigree: up to 58%). "
            "HIGHEST SINGLE-GENE ABSOLUTE LIFETIME PANCREATIC CANCER RISK in hereditary PDAC. "
            "CDKN2A GERMLINE VARIANTS AND PANCREATIC CANCER: "
            "p16-Leiden (c.225del19) — Dutch founder; "
            "p.Arg24Pro — Italian and Spanish founder; "
            "p.Gly101Trp — specific to certain FAMMM kindreds; "
            "NOT all CDKN2A melanoma mutations → PDAC risk: "
            "p16-specific (not p14-ARF-only) mutations carry pancreatic risk. "
            "SURVEILLANCE: Annual EUS + MRI pancreas from age 40–45 in CDKN2A carriers with PDAC family history; "
            "NCCN: surveillance in setting of ≥1 first-degree relative with PDAC. "
            "TARGETED THERAPY: No CDK4/6 inhibitor approved specifically for CDKN2A-mutated PDAC "
            "(palbociclib, ribociclib, abemaciclib trials ongoing but not established). "
            "MELANOMA SURVEILLANCE: Annual full-body skin exam + dermatoscopy; "
            "ophthalmology (uveal melanoma risk); avoid UV exposure."
        ),
        "locus": "9p21.3",
        "aa": 156,
        "kDa": 16,
        "omim_gene": "600160",
        "omim_disease": "FAMMM-PC Syndrome (OMIM 606719); Familial Melanoma (OMIM 155600)",
        "inheritance": "AD; tumour suppressor; loss of heterozygosity in tumour",
        "gene_class": "Cell cycle regulator — CDK4/6 inhibitor (p16-INK4A) + p53 stabiliser (p14-ARF)",
        "key_alerts": [
            "CDKN2A-HIGHEST-ABSOLUTE-PDAC-RISK: 25–58% lifetime pancreatic cancer risk — highest single-gene PDAC risk",
            "CDKN2A-FAMMM-TRIPLE-SURVEILLANCE: Annual EUS+MRI pancreas + dermatology + ophthalmology in all CDKN2A carriers",
            "CDKN2A-NO-APPROVED-TARGETED-THERAPY: No CDK4/6 inhibitor FDA approved for CDKN2A-mutated PDAC; standard FOLFIRINOX/GemNabP",
            "CDKN2A-DUAL-TRANSCRIPT-TESTING: Mutations affecting p16-INK4A only vs both p16+p14-ARF — different functional consequences; report precisely",
            "CDKN2A-NOT-ALL-VARIANTS-PDAC: p14-ARF-only mutations → melanoma but NOT pancreatic cancer risk; distinguish by transcript affected",
        ],
        "etiologies": [
            "p16-INK4A loss → CDK4/6-cyclinD1 unchecked → pRb phosphorylation → G1 bypass → uncontrolled proliferation",
            "p14-ARF loss → MDM2 destabilises p53 → apoptosis/senescence failure",
            "9p21 deletion (often covers both transcripts + CDKN2B/p15) → triple loss in many PDACs",
            "CDKN2A germline + somatic KRAS G12D synergy → rapid PDAC progression (PanIN → PDAC)",
        ],
        "stats": {
            "mean_dx_age": 62,
            "mean_dx_delay_months": 5,
            "pancreatic_rr": "15–40×",
            "lifetime_risk_pct": 40,
            "polo_olaparib_approved": False,
            "cdk46_inhibitor_approved": False,
            "surveillance_eus_mri": True,
        },
        "dx_delay_distribution": "5–9 months (presentation often locally advanced; new-onset diabetes can be early sign)",
    },
    # ── ATM — HPANCA-3 ───────────────────────────────────────────────────────
    {
        "gene": "ATM",
        "protein": "ATM — HPANCA-3 AD-Heterozygous — PI3K-DDR Master Kinase — ~6× Pancreatic RR — Platinum-Sensitive — PARPi Emerging — Radiation Sensitivity Intermediate",
        "alias": (
            "ATM; OMIM gene 607585; Ataxia-Telangiectasia OMIM 208900 (biallelic); "
            "Hereditary Breast Cancer (heterozygous); 11q22.3; 3056 aa; ~350 kDa; AD heterozygous. "
            "ATM (Ataxia-Telangiectasia Mutated) is the master kinase for double-strand break (DSB) signalling: "
            "PIKK (phosphoinositide-3-kinase-related kinase) superfamily; "
            "activated by DSB via MRN complex (MRE11-RAD50-NBN) → autophosphorylation at Ser1981 → "
            "activation as monomer → phosphorylates >1000 substrates: "
            "H2AX (γH2AX — DSB marker), BRCA1-Ser1387, CHEK2-Thr68 (→ CHK2 kinase activation), "
            "p53-Ser15 (→ p21 induction → G1 arrest), FANCD2 (→ ICL repair coordination). "
            "PANCREATIC CANCER RISK (ATM heterozygotes): "
            "Relative risk: ~5–6× general population; absolute lifetime risk ~4–6%. "
            "ATM is the second most common DDR gene in hereditary pancreatic cancer after BRCA2. "
            "TREATMENT IMPLICATIONS: "
            "Platinum sensitivity: ATM-mutated PDAC shows higher response to cisplatin/oxaliplatin "
            "(ICL → replication fork collapse → ATM-null cells cannot signal DSB repair → "
            "synthetic lethality with platinum). "
            "PARPi: emerging data (POLO trial not powered for ATM subgroup); "
            "mechanistic rationale for PARPi sensitivity exists (BRCAness); "
            "olaparib not yet FDA-labelled for ATM-mutated PDAC; clinical trials enrolling. "
            "RADIATION SENSITIVITY: ATM heterozygotes show intermediate sensitivity to ionising radiation "
            "(not as severe as biallelic A-T); inform radiation oncology before chemoradiotherapy. "
            "A-T SYNDROME (BIALLELIC): Progressive cerebellar ataxia + telangiectasias + IgA deficiency + "
            "lymphoma/leukaemia risk; avoid ionising radiation in A-T patients. "
            "VARIANT TYPES: Germline ATM truncating variants (frameshift, nonsense, splice) = high-penetrance; "
            "ATM c.7271T>G (p.Val2424Gly) — specific variant with highest pancreatic cancer risk; "
            "missense variants: VUS-heavy; functional assay needed for reclassification."
        ),
        "locus": "11q22.3",
        "aa": 3056,
        "kDa": 350,
        "omim_gene": "607585",
        "omim_disease": "Ataxia-Telangiectasia (biallelic OMIM 208900); Hereditary Breast Cancer (heterozygous)",
        "inheritance": "AD heterozygous; biallelic = Ataxia-Telangiectasia (AR severe syndrome)",
        "gene_class": "PI3K-DDR master kinase; DSB signal transducer",
        "key_alerts": [
            "ATM-PANCREATIC-6X-RR: ~6× pancreatic cancer RR; annual EUS+MRI from age 50 if family PDAC history",
            "ATM-PLATINUM-SENSITIVE: Prefer cisplatin-containing regimens in ATM-mutated PDAC; HRD exploits ICL mechanism",
            "ATM-PARP-EMERGING-NOT-LABELLED: PARPi mechanistic rationale; not FDA-approved for ATM-PDAC; enrol in trials",
            "ATM-RADIATION-INTERMEDIATE: Inform radiation oncologist of ATM status before chemoRT; intermediate sensitivity",
            "ATM-BIALLELIC-DO-NOT-IRRADIATE: Biallelic ATM (A-T patients) — ionising radiation absolutely contraindicated",
        ],
        "etiologies": [
            "ATM haploinsufficiency → reduced DSB signalling → HRD (partial) → genomic instability in PDAC",
            "ATM c.7271T>G — highest penetrance pancreatic variant; specific functional consequence at FATC domain",
            "ATM + KRAS G12D cooperation in PDAC progression (accelerated PanIN evolution)",
            "Biallelic ATM in PDAC tumour (somatic second-hit) → full HR deficiency → platinum/PARPi sensitivity",
        ],
        "stats": {
            "mean_dx_age": 66,
            "mean_dx_delay_months": 4,
            "pancreatic_rr": "~6×",
            "lifetime_risk_pct": 5,
            "polo_olaparib_approved": False,
            "platinum_preferred": True,
            "parp_emerging": True,
            "surveillance_eus_mri": True,
        },
        "dx_delay_distribution": "4–7 months (pancreatic head masses present with jaundice; body/tail late)",
    },
    # ── STK11 — HPANCA-4 ─────────────────────────────────────────────────────
    {
        "gene": "STK11",
        "protein": "STK11 (LKB1) — HPANCA-4 AD — Serine-Threonine Kinase Master Metabolic Regulator — 36% Lifetime Pancreatic Risk — Peutz-Jeghers Syndrome — Hamartomatous GI Polyps — Surveillance Age 25–30",
        "alias": (
            "STK11 (Serine Threonine Kinase 11), also known as LKB1 (Liver Kinase B1); "
            "OMIM gene 602216; Peutz-Jeghers Syndrome (PJS) OMIM 175200. "
            "19p13.3; 433 aa; ~49 kDa; AD haploinsufficiency. "
            "STK11/LKB1 is a master regulator of cell polarity and energy metabolism: "
            "STK11 kinase domain phosphorylates and activates AMPK (AMP-activated protein kinase) "
            "→ AMPK phosphorylates and inhibits mTORC1 (mTOR Complex 1) → "
            "suppresses protein synthesis + cell growth under energy stress; "
            "STK11 also regulates 12 downstream AMPK-related kinases (ARK family) controlling "
            "cell polarity, mitosis, and epithelial integrity. "
            "LOSS OF FUNCTION: STK11 null → AMPK not activated → mTORC1 constitutively active → "
            "unchecked protein synthesis + proliferation + loss of cell polarity → "
            "epithelial-to-mesenchymal transition facilitated. "
            "PEUTZ-JEGHERS SYNDROME: "
            "Clinical hallmarks: (1) mucocutaneous melanin pigmentation (lips, buccal mucosa, perioral skin — "
            "pathognomonic dark freckles appearing in childhood, may fade with age); "
            "(2) hamartomatous GI polyps (small bowel > colon > stomach) → "
            "intussusception in childhood (surgical emergency); "
            "(3) extremely high cancer risks across multiple organs. "
            "PANCREATIC CANCER RISK (STK11 carriers): "
            "Lifetime risk: 36% (Hearle 2006 systematic review) — highest absolute pancreatic cancer risk "
            "of any hereditary syndrome studied. "
            "Relative risk: ~132× general population (Giardiello 2000). "
            "Age of onset: median 40–50 years — much younger than sporadic PDAC. "
            "PJS CANCER RISKS BEYOND PANCREAS: "
            "Small bowel 13%; Stomach 29%; Colon 39%; Breast 54%; Ovary (SCTAT) 21%; Cervix (SCTAT+adenoma malignum); "
            "Lung (mucous cell adenoma → ADC). "
            "SURVEILLANCE IN STK11 CARRIERS: "
            "Pancreatic MRI + EUS: from age 25–30 (or at PJS diagnosis); every 1–2 years. "
            "GI endoscopy: upper + lower + video capsule endoscopy every 2–3 years from age 8–10. "
            "TARGETED THERAPY LANDSCAPE: "
            "mTOR inhibitors (everolimus, temsirolimus): mechanistic rationale (STK11 → AMPK → mTOR); "
            "clinical results in STK11-mutated PDAC: disappointing (no randomised trial benefit); "
            "STK11 loss in lung KRAS-mutated adenocarcinoma: poor immunotherapy response "
            "(STK11 loss → cold tumour via STING suppression — different biology in PDAC); "
            "No FDA-approved targeted therapy specifically for STK11-mutated PDAC. "
            "GERMLINE TESTING: Large deletions/rearrangements in ~30% — MLPA mandatory if sequencing negative. "
            "INTUSSUSCEPTION: Emergency surgery in PJS children/adults — small bowel polypectomy at surgery."
        ),
        "locus": "19p13.3",
        "aa": 433,
        "kDa": 49,
        "omim_gene": "602216",
        "omim_disease": "Peutz-Jeghers Syndrome (OMIM 175200)",
        "inheritance": "AD haploinsufficiency; de novo in ~25%",
        "gene_class": "Serine-threonine kinase — AMPK activator; mTOR suppressor; cell polarity master",
        "key_alerts": [
            "STK11-36PCT-LIFETIME-PDAC: Highest single-syndrome absolute pancreatic cancer risk; EUS+MRI from age 25–30",
            "STK11-PIGMENTATION-PATHOGNOMONIC: Mucocutaneous melanin pigmentation (lips/perioral) in childhood = diagnostic",
            "STK11-INTUSSUSCEPTION-EMERGENCY: Small bowel hamartomatous polyps → intussusception; bowel-sparing polypectomy at surgery",
            "STK11-MULTI-ORGAN-SURVEILLANCE: Breast 54%, colon 39%, stomach 29%, ovarian SCTAT 21% — multi-organ cancer programme mandatory",
            "STK11-NO-APPROVED-TARGETED-THERAPY: mTOR inhibitors mechanistically rational but clinically unproven in PDAC; standard chemotherapy",
        ],
        "etiologies": [
            "STK11 loss → AMPK not activated → mTORC1 constitutively on → unchecked growth in GI/pancreatic epithelium",
            "STK11 loss of cell polarity → impaired apical-basal differentiation → hamartoma formation → malignant transformation",
            "STK11-null + KRAS G12D synergy: major driver in murine PDAC models; very aggressive in humans",
            "Large deletions in ~30% — exon-level deletion analysis mandatory if point mutation not found",
        ],
        "stats": {
            "mean_dx_age": 46,
            "mean_dx_delay_months": 6,
            "pancreatic_rr": "~132×",
            "lifetime_risk_pct": 36,
            "polo_olaparib_approved": False,
            "mtor_inhibitor_clinical_benefit": False,
            "surveillance_eus_mri": True,
            "surveillance_start_age": 25,
        },
        "dx_delay_distribution": "6–10 months (younger onset; new-onset diabetes common harbinger; abdominal pain from polyps)",
    },
    # ── BRCA1 — HPANCA-5 ─────────────────────────────────────────────────────
    {
        "gene": "BRCA1",
        "protein": "BRCA1 — HPANCA-5 AD — RING-BRCT Scaffold — ~2% Lifetime Pancreatic Risk — POLO-Limited Data — PARPi Weaker Than BRCA2 — RRSO + Full HBOC Cascade Mandatory",
        "alias": (
            "BRCA1; OMIM gene 113705; Hereditary Breast and Ovarian Cancer Syndrome 1 OMIM 604370. "
            "17q21.31; 1863 aa; ~207 kDa; AD haploinsufficiency. "
            "BRCA1 is the master HR scaffold: "
            "RING domain (N-terminal, aa 2–103): E3 ubiquitin ligase with BARD1 heterodimer → "
            "ubiquitylates H2A at DSB (γH2AX foci maintenance); "
            "BRCT domains (C-terminal, aa 1646–1736/1756–1855): phosphopeptide binding → "
            "interaction with PALB2-BRCA2, CtIP, BACH1/BRIP1, Abraxas-MERIT40 (RAP80 complex); "
            "BRCA1 orchestrates HR by recruiting PALB2 → BRCA2 → RAD51. "
            "PANCREATIC CANCER RISK (BRCA1 germline carriers): "
            "Lifetime risk: ~1.5–2.5% (substantially lower than BRCA2 ~6%); relative risk ~2–3×. "
            "BRCA1 pancreatic risk is real but modest compared to breast (70%) and ovarian (46%) risks. "
            "POLO TRIAL (BRCA1 subgroup): Small numbers; results not statistically informative for BRCA1 alone; "
            "FDA label covers gBRCA1/2 but BRCA1 data are limited — mechanistically PARPi should work "
            "but clinical benefit less certain than BRCA2. "
            "CLINICAL APPROACH: "
            "BRCA1-mutated metastatic PDAC: offer olaparib maintenance per FDA label (gBRCA1 included); "
            "counsel that BRCA2 pancreatic evidence is stronger; "
            "Cisplatin-based: reasonable for HRD signature but less data than BRCA2. "
            "PRIORITISE HBOC MANAGEMENT: BRCA1 carriers have much higher breast + ovarian cancer risk "
            "than pancreatic cancer risk — breast surveillance and RRSO from age 35–40 are the priority "
            "management interventions; pancreatic surveillance is supplementary. "
            "PANCREATIC SURVEILLANCE: EUS+MRI annually from age 50 if PDAC in family (NCCN category 2B). "
            "CASCADE TESTING: Full HBOC cascade mandatory; BRCA1 is primarily a breast/ovarian cancer gene."
        ),
        "locus": "17q21.31",
        "aa": 1863,
        "kDa": 207,
        "omim_gene": "113705",
        "omim_disease": "Hereditary Breast and Ovarian Cancer Syndrome 1 (OMIM 604370)",
        "inheritance": "AD haploinsufficiency; biallelic = Fanconi anaemia S (FANCS)",
        "gene_class": "HR scaffold — RING E3 ligase + BRCT phosphopeptide receptor; PALB2-BRCA2 recruiter",
        "key_alerts": [
            "BRCA1-PANCREATIC-WEAKER-THAN-BRCA2: ~2% lifetime PDAC risk (vs BRCA2 ~6%); counsel accordingly — HBOC management is priority",
            "BRCA1-POLO-LIMITED-DATA: FDA label covers gBRCA1; clinical benefit weaker than BRCA2 in PDAC; offer olaparib maintenance but set expectations",
            "BRCA1-RRSO-MANDATORY-AGE-35-40: Bilateral salpingo-oophorectomy age 35–40 — primary HBOC intervention; pancreatic surveillance supplementary",
            "BRCA1-FULL-HBOC-CASCADE: All female relatives need breast+ovarian surveillance; BRCA1 is an HBOC gene primarily",
            "BRCA1-MLPA-MANDATORY: Large rearrangements in ~15%; MLPA/CNV mandatory if sequencing negative in high-risk family",
        ],
        "etiologies": [
            "BRCA1 haploinsufficiency → reduced PALB2-BRCA2 recruitment to DSB → partial HRD → genomic instability",
            "BRCA1 BRCT domain mutations (p.Arg1699Trp, p.Tyr1853Ter) — highest penetrance variants",
            "BRCA1 5382insC (Ashkenazi Jewish founder), 185delAG (AJ founder) — enriched in specific populations",
            "Large rearrangements (BRCA1-DUPS-4, BRCA1-DUPS-5) — MLPA mandatory",
        ],
        "stats": {
            "mean_dx_age": 66,
            "mean_dx_delay_months": 4,
            "pancreatic_rr": "~2–3×",
            "lifetime_risk_pct": 2,
            "polo_olaparib_approved": True,
            "polo_brca1_evidence_strength": "limited",
            "surveillance_eus_mri": False,
            "surveillance_nccn_2b": True,
        },
        "dx_delay_distribution": "4–7 months (standard PDAC presentation; back pain + weight loss late signs)",
    },
    # ── PALB2 — HPANCA-6 ─────────────────────────────────────────────────────
    {
        "gene": "PALB2",
        "protein": "PALB2 — HPANCA-6 AD — WD40 BRCA1-BRCA2 Bridge — 2–4% Lifetime Pancreatic Risk — POLO Emerging PARPi Sensitivity — TBCRC048 82% ORR Breast — Bridging Role in HR Pathway",
        "alias": (
            "PALB2 (Partner And Localiser of BRCA2); OMIM gene 610355; "
            "Hereditary Breast Cancer OMIM 610355; Fanconi Anaemia N (biallelic). "
            "16p12.2; 1186 aa; ~131 kDa; AD haploinsufficiency. "
            "PALB2 is the structural bridge between BRCA1 and BRCA2 in the HR pathway: "
            "N-terminal coiled-coil (aa 1–200): binds BRCA1 (BRCA1 BRCT–PALB2 coiled-coil interaction); "
            "WD40 repeat domain (C-terminus, aa 853–1186): interacts with BRCA2 N-terminus; "
            "PALB2 simultaneously binds BRCA1 and BRCA2 → forms BRCA1-PALB2-BRCA2 supercomplex at DSB. "
            "FUNCTION: Without PALB2, BRCA2 cannot be recruited to DSB → RAD51 loading fails → HRD. "
            "PANCREATIC CANCER RISK (PALB2 carriers): "
            "Lifetime risk: ~2–4% (OR ~2.3× in Jones 2009; higher estimates in FPC families); "
            "PALB2 is increasingly recognised as a hereditary pancreatic cancer gene. "
            "POLO TRIAL AND PALB2: POLO trial enrolled gBRCA1/2 only; PALB2 was NOT included; "
            "post hoc analyses and case series: PALB2-mutated PDAC responds to olaparib maintenance; "
            "FDA label does NOT include PALB2 for pancreatic cancer; off-label with BRCA2 functional equivalence rationale. "
            "TBCRC048 (breast cancer data): Olaparib in PALB2-mutated breast cancer → ORR 82% — "
            "highest ORR of any PARPi study in a single gene; FDA approval in breast 2022 (NCCN guidance). "
            "By analogy, PALB2 pancreatic cancer is PARPi-sensitive; clinical trials ongoing. "
            "GERMLINE TESTING IN PDAC: "
            "PALB2 germline should be tested in all hereditary PDAC panels; "
            "somatic PALB2 alteration in PDAC also guides treatment. "
            "PANCREATIC SURVEILLANCE: Annual EUS+MRI from age 50 if PDAC in family (NCCN). "
            "CASCADE: PALB2 carriers need both pancreatic and breast cancer surveillance; "
            "female first-degree relatives: breast MRI + mammography annually."
        ),
        "locus": "16p12.2",
        "aa": 1186,
        "kDa": 131,
        "omim_gene": "610355",
        "omim_disease": "Hereditary Breast Cancer (OMIM 610355); Fanconi Anaemia N (biallelic)",
        "inheritance": "AD haploinsufficiency; biallelic = Fanconi anaemia N (FANCN)",
        "gene_class": "HR scaffold bridge — WD40 domain; BRCA1-BRCA2 connector in HR supercomplex",
        "key_alerts": [
            "PALB2-PANCREATIC-2-4PCT-RISK: Annual EUS+MRI from age 50 with family PDAC history",
            "PALB2-PARP-SENSITIVE-OFF-LABEL: PARPi (olaparib) mechanistically rational; NOT FDA-labelled for PDAC; off-label with consent + trials",
            "PALB2-TBCRC048-82PCT-ORR-BREAST: Highest single-gene PARPi ORR in breast; confirms PALB2 functional equivalence to BRCA2 in HRD",
            "PALB2-POLO-NOT-ENROLLED: POLO trial excluded PALB2; olaparib maintenance in PALB2-PDAC is off-label but mechanistically supported",
            "PALB2-BREAST-SURVEILLANCE-MANDATORY: Female PALB2 carriers: annual breast MRI + mammography from age 30",
        ],
        "etiologies": [
            "PALB2 haploinsufficiency → BRCA1-BRCA2 disconnection at DSB → HRD → genomic instability",
            "PALB2 c.1592delT (Finnish founder), c.3113G>A (Dutch), c.2816T>G (multiple European) — pathogenic",
            "PALB2 WD40 domain mutations disrupt BRCA2 binding (most clinically significant)",
            "PALB2 biallelic (FANCN) — Fanconi anaemia with Wilms tumour and brain tumour",
        ],
        "stats": {
            "mean_dx_age": 64,
            "mean_dx_delay_months": 4,
            "pancreatic_rr": "~2–4×",
            "lifetime_risk_pct": 3,
            "polo_olaparib_approved": False,
            "parp_off_label_rationale": True,
            "surveillance_eus_mri": True,
        },
        "dx_delay_distribution": "4–8 months (standard PDAC presentation; diagnosis often locally advanced)",
    },
    # ── PRSS1 — HPANCA-7 ─────────────────────────────────────────────────────
    {
        "gene": "PRSS1",
        "protein": "PRSS1 (Cationic Trypsinogen) — HPANCA-7 AD — Gain-of-Function Trypsinogen Autoactivation — Hereditary Pancreatitis → 40–57× Pancreatic Cancer RR — R122H + N29I Founders — TPIAT for Severe Disease — Smoking ABSOLUTE-CI",
        "alias": (
            "PRSS1 (Serine Protease 1 / Cationic Trypsinogen); OMIM gene 276000; "
            "Hereditary Pancreatitis OMIM 167800. "
            "7q34; 247 aa (prepropeptide); ~25 kDa; AD gain-of-function. "
            "PRSS1 encodes cationic trypsinogen — the predominant trypsin precursor secreted by pancreatic acini: "
            "NORMAL PHYSIOLOGY: Trypsinogen secreted into duodenum → enterokinase cleaves activation peptide → "
            "active trypsin → cascade activation of all pancreatic digestive enzymes. "
            "HEREDITARY PANCREATITIS (GOF MECHANISM): "
            "PRSS1 R122H (p.Arg122His): eliminates trypsin autolysis site (Arg122 = self-cleavage site normally "
            "destroying misactivated trypsin within the pancreas); R122H trypsin accumulates → "
            "intra-acinar premature autoactivation → acinar cell destruction → pancreatitis. "
            "PRSS1 N29I (p.Asn29Ile): enhances autocatalytic trypsinogen activation → similar mechanism. "
            "HEREDITARY PANCREATITIS PHENOTYPE: "
            "Onset: first attack often in childhood (mean age ~10 years); "
            "recurrent acute pancreatitis → chronic pancreatitis with calcifications; "
            "exocrine pancreatic insufficiency (steatorrhoea, malabsorption); "
            "endocrine insufficiency (brittle type 3c diabetes mellitus); "
            "painful attacks: severe abdominal pain + vomiting; hospitalisation required. "
            "PANCREATIC CANCER RISK: "
            "PRSS1 hereditary pancreatitis → pancreatic cancer cumulative risk: "
            "~40% by age 70 (Lowenfels 1997); relative risk 40–57× general population. "
            "Smoking amplifies risk dramatically: PRSS1 carrier + smoker → risk increases ~2–3× further. "
            "MECHANISM: Chronic inflammation → acinar metaplasia → PanIN (Pancreatic Intraepithelial Neoplasia) "
            "→ PDAC; inflammation-driven KRAS mutagenesis. "
            "TREATMENT: "
            "PRSS1 PDAC: NOT a DNA repair gene — no PARPi rationale; standard FOLFIRINOX/gemcitabine. "
            "Total Pancreatectomy with Islet Autotransplant (TPIAT): "
            "For severe recurrent hereditary pancreatitis with disabling pain and quality-of-life impairment; "
            "removes pancreas (eliminating cancer risk + pain source); "
            "islet autotransplant into portal vein preserves endogenous insulin production "
            "(prevents/delays brittle diabetes post-pancreatectomy); "
            "best outcomes when performed before advanced pancreatic fibrosis. "
            "SMOKING CESSATION: ABSOLUTE priority — doubles already very high cancer risk; "
            "must be addressed at every clinical encounter. "
            "ALCOHOL ABSTINENCE: Triggers pancreatitis attacks; cessation mandatory."
        ),
        "locus": "7q34",
        "aa": 247,
        "kDa": 25,
        "omim_gene": "276000",
        "omim_disease": "Hereditary Pancreatitis (OMIM 167800)",
        "inheritance": "AD gain-of-function; R122H penetrance ~80% by age 50",
        "gene_class": "Serine protease — cationic trypsinogen; digestive enzyme precursor; GOF autolysis-defect mechanism",
        "key_alerts": [
            "PRSS1-PANCREATITIS-TO-CANCER: 40–57× pancreatic cancer RR via chronic inflammation → PanIN → PDAC",
            "PRSS1-SMOKING-ABSOLUTE-CI: Smoking doubles already extreme PDAC risk in PRSS1 carriers — cessation is non-negotiable",
            "PRSS1-TPIAT-OPTION-SEVERE: Total pancreatectomy with islet autotransplant for severe disabling hereditary pancreatitis",
            "PRSS1-NOT-DNA-REPAIR-NO-PARPI: PRSS1 is NOT an HRD gene; PARPi has no mechanistic basis; standard chemotherapy",
            "PRSS1-TYPE3C-DIABETES: Pancreatogenic diabetes after repeated attacks; endocrine + exocrine replacement mandatory",
        ],
        "etiologies": [
            "PRSS1 R122H GOF: eliminates autolysis site → premature intra-acinar trypsin activation → acinar destruction → chronic pancreatitis",
            "PRSS1 N29I GOF: enhances autocatalytic activation → similar pancreatitis phenotype",
            "Chronic inflammation → KRAS mutagenesis in acinar/ductal cells → PanIN progression → PDAC",
            "Smoking + PRSS1 → 2–3× further PDAC risk amplification (synergistic carcinogenesis)",
        ],
        "stats": {
            "mean_dx_age": 56,
            "mean_dx_delay_months": 8,
            "pancreatic_rr": "40–57×",
            "lifetime_risk_pct": 40,
            "polo_olaparib_approved": False,
            "parp_rationale": False,
            "tpiat_option": True,
            "smoking_absolute_ci": True,
        },
        "dx_delay_distribution": "8–14 months (new-onset diabetes or worsening glucose control often precedes diagnosis by months)",
    },
    # ── MLH1 — HPANCA-8 ──────────────────────────────────────────────────────
    {
        "gene": "MLH1",
        "protein": "MLH1 — HPANCA-8 AD — MutL-Alpha MMR Complex — Lynch Syndrome — 3–4% Lifetime Pancreatic Risk — dMMR → Pembrolizumab FDA 2017 — IHC Mandatory — EPCAM-MLH1 Silencing",
        "alias": (
            "MLH1 (MutL Homologue 1); OMIM gene 120436; Lynch Syndrome 2 OMIM 609310. "
            "3p22.2; 756 aa; ~85 kDa; AD haploinsufficiency. "
            "MLH1 forms the MutL-alpha heterodimer with PMS2 — the endonuclease of mismatch repair (MMR): "
            "MutS-alpha (MSH2-MSH6) recognises base mispairs and small insertion-deletion loops → "
            "recruits MutL-alpha (MLH1-PMS2) → MLH1 activates PMS2 latent endonuclease → "
            "PMS2 nicks the newly synthesised strand → exonuclease 1 (EXO1) excises mismatch → "
            "polymerase δ resynthesises; MLH1 also participates in meiotic recombination and DSB repair. "
            "MMR LOSS → MSI: Microsatellite instability (MSI-H) → accumulation of frameshift mutations "
            "at simple repeat sequences throughout genome → thousands of frameshift neoantigens → "
            "immune recognition → T-cell infiltration → immune-hot tumour. "
            "LYNCH SYNDROME (MLH1 germline): "
            "Most common: colorectal cancer (40–80% lifetime); Endometrial cancer (40–60%); "
            "Ovarian (~10%); Gastric (7–8%); Urothelial/Upper urinary tract (~5–8%); "
            "Small bowel (4–5%); Pancreatic (3–4%); CNS (glioblastoma in Lynch 3–5%). "
            "PANCREATIC CANCER RISK (MLH1 carriers): "
            "Lifetime risk: ~3–4% (SIR 4–8.6 in various Lynch cohorts); "
            "onset typically younger than sporadic PDAC; "
            "MLH1-deficient PDAC often medullary histology (better prognosis than conventional PDAC). "
            "PEMBROLIZUMAB FDA 2017 (tumor-agnostic dMMR/MSI-H): "
            "First-ever tumor-agnostic FDA approval; for all solid tumours with dMMR/MSI-H. "
            "KEYNOTE-158 pancreatic cohort (dMMR PDAC): ORR 18.2% — lower than CRC 36% but meaningful; "
            "durable responses in subset: 6+ month responders observed; "
            "CheckMate-142 and KEYNOTE-177 (CRC data) support dMMR-pembrolizumab in Lynch. "
            "DIAGNOSTICS IN MLH1-RELATED PDAC: "
            "IHC MMR panel (MLH1/PMS2/MSH2/MSH6) on all PDAC biopsies (NCCN category 2A); "
            "MLH1 protein loss: can be germline (Lynch) OR sporadic (MLH1 promoter hypermethylation); "
            "MLH1 promoter methylation (BRAFV600E-driven in ~15–20% sporadic CRC) — rare in PDAC; "
            "BRAF V600E testing to distinguish sporadic MLH1 silencing from Lynch. "
            "EPCAM-MLH1 SILENCING: EPCAM deletion upstream of MSH2 causes MSH2 loss (not MLH1); "
            "for MLH1: EPCAM is not directly relevant; large MLH1 deletions (MLPA mandatory). "
            "LYNCH SURVEILLANCE PROTOCOL: Colonoscopy every 1–2 years from age 25; "
            "endometrial biopsy annually from age 35 (or RRSO); "
            "gastric endoscopy every 2–5 years; urothelial: annual urinalysis + cytology; "
            "pancreatic: EUS+MRI annually from age 50 (or earlier if PDAC in family)."
        ),
        "locus": "3p22.2",
        "aa": 756,
        "kDa": 85,
        "omim_gene": "120436",
        "omim_disease": "Lynch Syndrome 2 / HNPCC (OMIM 609310)",
        "inheritance": "AD haploinsufficiency; sporadic MLH1 loss via promoter hypermethylation (not Lynch)",
        "gene_class": "MMR endonuclease scaffold — MutL-alpha complex with PMS2; MSI-H driver",
        "key_alerts": [
            "MLH1-LYNCH-3-4PCT-PDAC: Lynch syndrome → 3–4% lifetime pancreatic cancer risk; EUS+MRI from age 50",
            "MLH1-PEMBROLIZUMAB-dMMR-FDA2017: Confirm dMMR/MSI-H by IHC → pembrolizumab tumor-agnostic (KEYNOTE-158 PDAC ORR 18.2%)",
            "MLH1-IHC-MANDATORY-ALL-PDAC: IHC MMR panel (MLH1/PMS2/MSH2/MSH6) on all PDAC biopsies per NCCN",
            "MLH1-HYPERMETHYLATION-VS-LYNCH: Sporadic MLH1 loss (promoter methylation, BRAF V600E) vs germline Lynch — distinguish; BRAF + methylation testing",
            "MLH1-MULTI-ORGAN-LYNCH-SURVEILLANCE: Colonoscopy every 1–2 years age 25; endometrial biopsy age 35; urothelial; gastric; pancreatic",
        ],
        "etiologies": [
            "MLH1 germline haploinsufficiency → somatic LOH in tumour → complete MMR loss → MSI-H → Lynch PDAC",
            "Sporadic MLH1 silencing: promoter hypermethylation (BRAF V600E-driven in CRC; rare in PDAC)",
            "Large MLH1 deletions/rearrangements in ~10% — MLPA mandatory if sequencing negative",
            "MLH1-deficient PDAC: medullary histology with lymphocytic infiltrate; relatively better prognosis (vs conventional PDAC)",
        ],
        "stats": {
            "mean_dx_age": 60,
            "mean_dx_delay_months": 5,
            "pancreatic_rr": "4–8×",
            "lifetime_risk_pct": 4,
            "pembrolizumab_dmmr_approved": True,
            "ihc_mandatory": True,
            "polo_olaparib_approved": False,
        },
        "dx_delay_distribution": "5–9 months (medullary histology sometimes faster diagnosis; Lynch patients in surveillance may catch earlier)",
    },
]


def _make_patients(gene_info: dict, seed: int) -> list:
    rng = random.Random(seed)
    mean_age = gene_info["stats"]["mean_dx_age"]
    mean_delay = gene_info["stats"]["mean_dx_delay_months"]
    patients = []
    for i in range(40):
        age = max(35, min(82, int(rng.gauss(mean_age, 7))))
        delay = max(1, int(rng.gauss(mean_delay, 3)))
        patients.append({
            "patient_id": f"HPANCA-{gene_info['gene']}-{seed}-{i+1:03d}",
            "age_at_diagnosis": age,
            "diagnosis_delay_months": delay,
            "seed": seed,
        })
    return patients


def get_overview() -> dict:
    all_pts = []
    gene_summaries = []
    for idx, g in enumerate(PANCREATIC_GENES):
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
        "atlas": "Hereditary-Pancreatic-Cancer-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Pancreatic Cancer Reference · "
            "BRCA2 · CDKN2A · ATM · STK11 · BRCA1 · PALB2 · PRSS1 · MLH1 · "
            "320 patients (8×40, seeds 1646–1653)"
        ),
        "aggregate_stats": {
            "total_patients": total,
            "mean_dx_age_years": mean_age,
            "mean_dx_delay_months": mean_delay,
            "brca2_polo_olaparib_fda_approved": True,
            "brca2_polo_pfs_hr": 0.53,
            "cdkn2a_highest_absolute_pdac_risk_pct": 40,
            "stk11_lifetime_pdac_risk_pct": 36,
            "prss1_pdac_rr": "40–57×",
            "prss1_smoking_absolute_ci": True,
            "mlh1_pembrolizumab_dmmr_fda2017": True,
            "mlh1_keynote158_pdac_orr_pct": 18.2,
            "cisplatin_preferred_for_brca2_atm": True,
            "surveillance_eus_mri_recommended": True,
            "cascade_tested_pct": round(rng.uniform(58, 74), 1),
            "eus_surveillance_performed_pct": round(rng.uniform(35, 52), 1),
        },
        "genes": gene_summaries,
        "top_alerts": [
            "BRCA2-POLO-OLAPARIB-FDA2019: Maintenance olaparib after platinum response in gBRCA1/2 metastatic PDAC — FDA approved; CisGem preferred first-line",
            "CDKN2A-HIGHEST-ABSOLUTE-PDAC-RISK: 25–58% lifetime pancreatic cancer risk — annual EUS+MRI from age 40–45 in carriers with PDAC family history",
            "STK11-PEUTZ-JEGHERS-36PCT-PDAC: 36% lifetime PDAC risk; mucocutaneous pigmentation pathognomonic; EUS+MRI from age 25–30; intussusception emergency",
            "PRSS1-SMOKING-ABSOLUTE-CI: Hereditary pancreatitis → 40–57× PDAC risk; smoking amplifies further; TPIAT for severe disease",
            "MLH1-PEMBROLIZUMAB-dMMR-MANDATORY: All PDAC biopsies IHC for MMR (NCCN); dMMR → pembrolizumab tumor-agnostic FDA 2017",
            "ATM-PLATINUM-SENSITIVE: ~6× PDAC RR; prefer cisplatin-containing regimens; PARPi emerging but not labelled for ATM-PDAC",
            "PALB2-PARPi-OFF-LABEL-RATIONAL: POLO trial excluded PALB2; olaparib off-label but mechanistically equivalent to BRCA2 in HRD; enrol in trials",
            "BRCA1-WEAKER-PDAC-THAN-BRCA2: ~2% lifetime risk; POLO covers gBRCA1 but evidence weak; prioritise HBOC breast/ovarian management over PDAC",
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for idx, g in enumerate(PANCREATIC_GENES):
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
        "atlas": "Hereditary-Pancreatic-Cancer-Atlas",
        "concepts": {
            "HPANCA Gene Tiers — Risk-Stratified Clinical Management": (
                "Hereditary Pancreatic Cancer (HPANCA) genes span four clinical management tiers: "
                "TIER 1 — INFLAMMATION-DRIVEN HIGHEST RISK (PRSS1): "
                "Hereditary pancreatitis pathway; 40–57× PDAC RR via chronic inflammation → PanIN; "
                "NOT a DNA repair gene — no PARPi rationale; TPIAT for severe pancreatitis; "
                "smoking ABSOLUTELY contraindicated (amplifies risk further); "
                "standard chemotherapy (FOLFIRINOX/GemNabP) for PDAC. "
                "TIER 1 — SYNDROME-DRIVEN HIGHEST RISK (CDKN2A, STK11): "
                "CDKN2A: FAMMM syndrome; 25–58% lifetime PDAC risk; cell cycle (CDK4/6-pRb); no targeted therapy; "
                "STK11: Peutz-Jeghers syndrome; 36% lifetime PDAC risk; mTOR pathway; no approved targeted therapy; "
                "surveillance from age 25–40 based on syndrome. "
                "TIER 2 — DNA REPAIR GENES WITH APPROVED TARGETED THERAPY (BRCA2, BRCA1): "
                "BRCA2: 5–10× PDAC RR; POLO olaparib FDA 2019; CisGem preferred upfront; PSMA-like staging; "
                "BRCA1: 2–3× PDAC RR; POLO covers gBRCA1 but evidence weaker; prioritise HBOC over PDAC. "
                "TIER 3 — DNA REPAIR GENES WITH EMERGING/OFF-LABEL THERAPY (ATM, PALB2): "
                "ATM: ~6× PDAC RR; platinum-sensitive; PARPi emerging (not FDA-labelled for PDAC); "
                "PALB2: 2–4× PDAC RR; PARPi off-label but mechanistically strong; TBCRC048 82% ORR breast. "
                "TIER 4 — MMR DEFICIENCY WITH IMMUNOTHERAPY (MLH1): "
                "Lynch syndrome; 3–4% PDAC risk; dMMR/MSI-H → pembrolizumab FDA 2017 tumor-agnostic; "
                "IHC mandatory in all PDAC biopsies (NCCN category 2A). "
                "CRITICAL PRINCIPLE: PRSS1/CDKN2A/STK11 are NOT HR genes — do NOT offer PARPi; "
                "MLH1 dMMR is NOT a PARPi indication — pembrolizumab is the indicated therapy."
            ),
            "POLO Trial — Olaparib Maintenance in gBRCA1/2 Metastatic PDAC": (
                "POLO (Golan 2019, NEJM) is the pivotal phase III trial establishing PARPi in pancreatic cancer: "
                "DESIGN: gBRCA1/2-mutated metastatic PDAC patients who had not progressed after "
                "≥16 weeks of platinum-based first-line chemotherapy → "
                "randomised olaparib 300 mg BD vs placebo maintenance. "
                "PRIMARY ENDPOINT (rPFS): Olaparib 7.4 months vs placebo 3.8 months; "
                "HR 0.53 (95% CI 0.35–0.82); p=0.004. "
                "OS: 18.9 (olaparib) vs 18.1 months (placebo); HR 0.83 (NS) — crossover confounds OS analysis. "
                "FDA APPROVAL 2019: Olaparib maintenance for gBRCA1/2-mutated metastatic PDAC "
                "(first oncology indication in pancreatic cancer for a PARP inhibitor). "
                "WHAT POLO DID NOT STUDY: "
                "ATM-mutated PDAC (not enrolled); PALB2-mutated PDAC (not enrolled); "
                "somatic (non-germline) BRCA2 alterations (not the primary population). "
                "CLINICAL PRACTICE IMPLICATIONS: "
                "All patients with metastatic PDAC → germline BRCA1/2 testing (rapid turnaround <2 weeks); "
                "if gBRCA1/2 positive AND platinum response → switch to olaparib maintenance; "
                "if patient progresses on platinum → NOT eligible for POLO paradigm; "
                "PRSS1/CDKN2A/STK11/MLH1-mutated PDAC → NOT candidates for olaparib (no HRD mechanism). "
                "REVERSION MUTATIONS (acquired resistance): "
                "BRCA2 secondary mutations restoring reading frame detected in ~20–30% of PDAC post-olaparib; "
                "ctDNA monitoring for reversion; "
                "RAD51C/D overexpression alternate resistance mechanism."
            ),
            "STK11/LKB1 — Peutz-Jeghers Syndrome: Diagnosis and Surveillance": (
                "Peutz-Jeghers Syndrome (PJS) is a highly penetrant autosomal dominant cancer syndrome: "
                "DIAGNOSTIC CRITERIA (WHO 2019) — any ONE of: "
                "(1) ≥3 histologically confirmed PJS hamartomatous polyps; "
                "(2) any number of PJS polyps + family history of PJS; "
                "(3) characteristic mucocutaneous pigmentation + family history of PJS; "
                "(4) any number of PJS polyps + mucocutaneous pigmentation. "
                "MUCOCUTANEOUS PIGMENTATION: Melanin spots on lips, oral mucosa, nostrils, periorbital; "
                "appear in first decade; may fade after puberty (do not rely on adult absence); "
                "PATHOGNOMONIC when combined with GI polyps. "
                "PJS HAMARTOMATOUS POLYPS: "
                "Arborising smooth muscle architecture (NOT adenomatous); "
                "Distribution: small bowel (duodenum, jejunum, ileum) > stomach > colon; "
                "Intussusception: leading cause of bowel obstruction in PJS (surgical emergency); "
                "bowel-sparing polypectomy at laparotomy preferred (repeated resections → short gut). "
                "SURVEILLANCE PROTOCOL (expert consensus): "
                "Pancreas: MRI pancreas + EUS every 1–2 years from age 25–30; "
                "GI: video capsule endoscopy every 2–3 years from age 8–10 (small bowel); "
                "OGD + colonoscopy every 2–3 years from age 8–10; "
                "Breast (females): MRI + mammography annually from age 25; "
                "Cervix + gynaecology: annual smear; hysteroscopy if abnormal bleeding; "
                "Testicular: annual examination + ultrasound from puberty (Sertoli cell tumour, feminisation). "
                "NO APPROVED CHEMOPREVENTION: mTOR inhibitors (everolimus) under study but not standard. "
                "INTUSSUSCEPTION EMERGENCY: "
                "Acute small bowel obstruction in PJS child/adult = intussusception until proven otherwise; "
                "immediate surgery; intraoperative small bowel polypectomy to reduce future intussusceptions."
            ),
            "PRSS1 Hereditary Pancreatitis — TPIAT and Cancer Prevention": (
                "Total Pancreatectomy with Islet Autotransplant (TPIAT) for hereditary pancreatitis: "
                "INDICATION: PRSS1 hereditary pancreatitis with: (1) disabling recurrent acute pancreatitis; "
                "(2) severe refractory pain requiring opioids; (3) progressive exocrine/endocrine failure; "
                "(4) deteriorating quality of life despite maximal medical therapy. "
                "PROCEDURE: Total pancreatectomy (removes the source of pancreatitis and cancer risk) + "
                "pancreatic islets isolated from resected specimen → purified islets infused into portal vein → "
                "hepatic implantation → islets provide endogenous insulin production. "
                "OUTCOMES: "
                "Pain relief: >70% achieve significant/complete pain reduction; "
                "Insulin independence: ~30–40% at 1 year; ~20–25% at 5 years "
                "(depends on pre-operative islet mass — perform before extensive fibrosis); "
                "PDAC risk: eliminated (no residual pancreatic tissue); "
                "Best results: early referral before chronic pancreatitis progresses to fibrosis. "
                "TPIAT TIMING: Refer to high-volume TPIAT centre when: "
                "≥2 admissions/year for acute pancreatitis despite optimal medical therapy; "
                "opioid-dependent pain management; "
                "CECT showing progressive fibrosis but NOT yet calcific endstage (islet yield decreases). "
                "CANCER SURVEILLANCE BEFORE TPIAT: Annual imaging from onset of hereditary pancreatitis. "
                "SMOKING: One cigarette = immediate contraindication to conservative management; "
                "smoking cessation is the most impactful single intervention in PRSS1 carrier management."
            ),
            "dMMR/MSI-H in Pancreatic Cancer — Pembrolizumab and Lynch Screening": (
                "Mismatch repair deficiency (dMMR) in PDAC creates immune-hot tumours amenable to checkpoint blockade: "
                "FREQUENCY: dMMR/MSI-H in PDAC: ~1–2% of all PDAC; "
                "in Lynch syndrome carriers with PDAC: essentially all tumours are dMMR (germline MMR loss). "
                "PEMBROLIZUMAB FDA 2017 (tumor-agnostic): "
                "KEYNOTE-158 pancreatic cohort (n=22 dMMR PDAC): ORR 18.2% (lower than CRC 36%); "
                "durable responses in subset; disease stabilisation in additional patients; "
                "pancreatic dMMR tumours have fewer stromal barriers to T-cell infiltration than "
                "microsatellite-stable PDAC (notoriously immune-cold). "
                "IHC MMR PANEL IN ALL PDAC (NCCN category 2A): "
                "MLH1/PMS2/MSH2/MSH6 by IHC on all PDAC biopsies; "
                "loss of MLH1 (+PMS2) staining → suspect Lynch OR sporadic MLH1 methylation; "
                "BRAF V600E testing + MLH1 promoter methylation to distinguish sporadic (methyl) vs Lynch. "
                "LYNCH PANCREATIC CANCER: MLH1 germline → Lynch-PDAC often medullary histology "
                "(syncytial growth pattern + prominent lymphocytic infiltrate); "
                "medullary PDAC with Lynch = better prognosis than conventional PDAC. "
                "GERMLINE TESTING: All MLH1-loss PDAC patients without BRAF mutation or methylation "
                "→ germline MMR testing (Lynch cascade); "
                "regardless of age — Lynch PDAC can occur before age 50. "
                "MANAGEMENT: Pembrolizumab alone (monotherapy) as preferred first-line option "
                "for dMMR-MSI-H metastatic PDAC (extrapolated from CRC + KEYNOTE-158); "
                "gemcitabine-based chemotherapy as alternative if pembrolizumab not available."
            ),
        },
        "pharmacological_distinctions": [
            "BRCA2 olaparib (POLO FDA2019) vs PALB2 olaparib (off-label): POLO trial enrolled gBRCA1/2 only; PALB2 is mechanistically HR-deficient and PARPi-sensitive (TBCRC048 breast ORR 82%); offer olaparib in PALB2-PDAC with explicit informed consent about off-label status and enrol in trials where available",
            "Cisplatin-based (BRCA2, ATM) vs standard gemcitabine (CDKN2A, STK11, PRSS1): BRCA2/ATM-mutated PDAC is HRD-driven — cisplatin-induced interstrand crosslinks (ICL) exploit HRD; CDKN2A/STK11/PRSS1 tumours are NOT HRD — no additional benefit from cisplatin over standard regimens",
            "Pembrolizumab (MLH1 dMMR) vs olaparib (BRCA2): mutually exclusive treatment logic; MMR deficiency drives immunotherapy sensitivity; HRD drives PARPi sensitivity; test for BOTH (IHC MMR + germline BRCA2/ATM) since tumours can be dMMR AND BRCA2-mutated (different clonal origins) — apply therapy matched to the driver",
            "mTOR inhibitors (STK11) — mechanistic rationale vs clinical failure: STK11 loss → mTORC1 constitutively active → everolimus/temsirolimus rational; however, clinical trials in STK11-mutated PDAC have not shown survival benefit; feedback activation of PI3K/AKT (mTORC2) upon mTORC1 inhibition blunts response; do not offer mTOR inhibitors as standard STK11-PDAC treatment outside trials",
            "PRSS1 PDAC — standard chemotherapy ONLY: PRSS1 is a GOF digestive enzyme gene, NOT a DNA repair or cell cycle gene; offering PARPi to PRSS1-PDAC based on hereditary label alone is mechanistically unjustified; chemotherapy selection: FOLFIRINOX if PS 0–1; GemNabP if PS 1–2; no targeted therapy label applies",
        ],
        "key_standards": [
            "NCCN Pancreatic Adenocarcinoma (v2.2024): germline BRCA1/2 testing for all metastatic PDAC; IHC MMR (MLH1/PMS2/MSH2/MSH6) for all PDAC biopsies; surveillance: annual EUS+MRI for high-risk individuals (FPC, BRCA2, ATM, PALB2, STK11, CDKN2A)",
            "FDA Olaparib maintenance in gBRCA1/2 metastatic PDAC (POLO trial, December 2019): first PARP inhibitor approval in pancreatic cancer; maintenance after ≥16 weeks platinum without progression",
            "FDA Pembrolizumab 2017 tumor-agnostic dMMR/MSI-H: for all solid tumours including PDAC; confirmed dMMR by IHC or MSI by PCR/NGS required",
            "CAPS (Cancer of the Pancreas Screening) Consortium Guidelines: EUS+MRI annually for BRCA2/PALB2/ATM/CDKN2A/STK11/MLH1 carriers with ≥1 FDR with PDAC or ≥2 affected relatives regardless of gene",
            "NCCN Lynch Syndrome Guidelines (v2.2024): MLH1 germline → annual colonoscopy from age 25; endometrial sampling age 30–35; pancreatic EUS+MRI from age 50; cascade testing all first-degree relatives",
        ],
    }
