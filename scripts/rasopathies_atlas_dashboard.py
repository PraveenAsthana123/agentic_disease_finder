#!/usr/bin/env python3
"""Rasopathies Atlas — Complete 8-Gene RAS/MAPK Pathway Disorders Atlas
PTPN11  (Noonan Syndrome type 1 — 593 aa; 12q24.13; SHP2 phosphatase; AD GOF;
         most common RASopathy 50% of Noonan; pulmonary stenosis 50–80%;
         HCM 20%; MEKi selumetinib for inoperable PN; GH therapy approved) ·
RAF1    (Noonan Syndrome type 5 with HCM — 648 aa; 3p25.2; RAF1 kinase;
         AD GOF; HCM in 75% vs 20% typical NS; severe HCM; AHA cardiomyopathy Tx) ·
SOS1    (Noonan Syndrome type 4 — 1333 aa; 2p22.1; RAS GEF;
         AD GOF; mildest NS; near-normal intellect; tallest stature; LEAST HCM) ·
KRAS    (Noonan Syndrome type 3 / CFC spectrum — 189 aa; 12p12.1; KRAS GTPase;
         AD GOF; intermediate severity; café-au-lait ± JMML-like haematology) ·
BRAF    (CFC Syndrome / LEOPARD-BRAF — 766 aa; 7q34; BRAF kinase;
         AD GOF; severe cognitive + ichthyosis; ectodermal anomalies) ·
MAP2K1  (CFC Syndrome MEK1 — 393 aa; 15q22.31; MEK1 kinase;
         AD GOF; ± MAP2K2 (MEK2); CFC diagnosis when BRAF/KRAS/HRAS negative;
         vemurafenib / cobimetinib off-label investigational) ·
HRAS    (Costello Syndrome — 189 aa; 11p15.5; HRAS GTPase p.Gly12Ser founder;
         AD GOF de novo; nasal papillomata PATHOGNOMONIC; RMS 15%; BT 5%;
         AVOID rapid rehydration cardiac arrhythmia; GH investigational NOT standard) ·
CBL     (Noonan-like Syndrome with JMML — 906 aa; 11q23.3; E3 ubiquitin ligase;
         AD GOF germline + somatic second-hit; JMML 20–50%; spontaneous remission;
         HCM; vasculitis; hyperpigmentation; HIGH cancer surveillance mandatory)
320-patient aggregate cohort (8 × 40, seeds 1230–1237)
"""

import random

SEED_BASE = 1230

RASOPATHY_GENES = [
    # ── PTPN11 — Noonan Syndrome Type 1 ─────────────────────────────────────
    {
        "gene": "PTPN11",
        "protein": "SHP2 (Src Homology-2 Domain-Containing Protein Tyrosine Phosphatase 2)",
        "alias": (
            "PTPN11; OMIM gene 176876; Noonan Syndrome 1 (NS1) #163950; 12q24.13; 593 aa; ~68 kDa; "
            "AD GOF; accounts for ~50% of all Noonan syndrome; SHP2 is the archetypal RASopathy phosphatase; "
            "also mutated somatically in 35% of juvenile myelomonocytic leukaemia (JMML) and 3% of ALL"
        ),
        "aa": "593 aa",
        "kDa": "~68 kDa",
        "locus": "12q24.13",
        "omim_gene": 176876,
        "omim_disease": 163950,
        "inheritance": "AD GOF (de novo 60%; familial 40%)",
        "gene_class": (
            "Protein tyrosine phosphatase; N-SH2 + C-SH2 + PTP catalytic domain; "
            "normally maintained in CLOSED (auto-inhibited) state by intramolecular N-SH2:PTP interaction; "
            "GOF mutations (Asp61Asn, Asn308Asp — most common) disrupt auto-inhibition → constitutive OPEN state "
            "→ sustained RAS-ERK and PI3K-AKT signalling → proliferative, anti-apoptotic, pro-growth; "
            "LEOPARD syndrome alleles (Tyr279Cys, Thr468Met) are CATALYTIC-DEAD loss-of-function yet still "
            "dominant negative → clinically LEOPARD via dominant negative RAS attenuation; "
            "SHP2 is a cytoplasmic phosphatase that amplifies RAS-GTP loading upstream of RAF-MEK-ERK; "
            "downstream of receptor tyrosine kinases (FGFR, PDGFR, MET, EGFR, EPO-R); "
            "SHP2 inhibitor SHP099 / TNO155 (Novartis) in clinical trials for PTPN11-mutant cancers; "
            "germline GOF alleles are typically hypomorphic relative to somatic oncogenic alleles (e.g. JMML Glu76Lys)"
        ),
        "phenotype": (
            "Facial: hypertelorism, ptosis, low-set posteriorly-rotated ears, short neck with webbing (pterygium colli), "
            "depressed nasal root, wide nasal tip, low anterior hairline; "
            "Cardiac: PULMONARY VALVE STENOSIS (50–80% — most common CHD in NS), ASD (10–15%), HCM (20%); "
            "Short stature: mean adult height −2 SD; GH-IGF1 axis often intact but GH secretion reduced; "
            "Cognitive: mild intellectual disability 25% (IQ 80–90 typical); language delay; "
            "Coagulopathy: Factor XI deficiency (20–30%), Factor XII, von Willebrand disease Type I — pre-op check MANDATORY; "
            "Cryptorchidism (60–80% boys); lymphoedema (20%); thoracic cage deformity (pectus excavatum/carinatum)"
        ),
        "hallmark": (
            "PULMONARY VALVE STENOSIS in young patient with characteristic facies — think NS/PTPN11; "
            "PTERYGIUM COLLI (neck webbing) — also in Turner/Noonan; karyotype first to exclude Turner; "
            "FACTOR XI DEFICIENCY in a Noonan patient — MANDATORY pre-operative coagulation screen; "
            "COAGULOPATHY: 35% of NS have Factor VIII/IX/XI/vWF deficiency — haematology referral pre-op; "
            "LEOPARD ALLELES (Tyr279/Thr468): multiple lentigines + HCM + DEAFNESS (distinct from NS); "
            "JMML: germline PTPN11 GOF + somatic PTPN11 — spontaneous remission in germline-only; "
            "p.Asn308Asp associated with highest PS frequency; p.Asp61Asn with highest HCM frequency; "
            "Rasopathy panel: PTPN11 + RAF1 + SOS1 + KRAS + BRAF + MAP2K1/2 + HRAS + CBL + NF1 + SHOC2"
        ),
        "treatment_alert": (
            "PULMONARY VALVE STENOSIS: balloon pulmonary valvuloplasty if gradient >50 mmHg (first line); "
            "surgical if dysplastic valve; annual echo surveillance; "
            "HCM: propranolol/bisoprolol (beta-blockade first line); avoid digoxin in outflow obstruction; "
            "MEK INHIBITOR SELUMETINIB: approved FDA 2020 (KOSELUGO) for NF1 inoperable plexiform neurofibroma — "
            "investigational off-label for PTPN11-Noonan with HCM (BEACON trial); "
            "GH THERAPY: APPROVED for NS with short stature (not FH — GH is ON-label for NS); "
            "dose 0.033–0.067 mg/kg/day; annual IGF-1; cardiac echo BEFORE starting GH (HCM risk); "
            "COAGULOPATHY: pre-operative clotting screen (PT/APTT/Factor XI/XII/vWF) MANDATORY; "
            "FFP or factor concentrate if Factor XI <30%; tranexamic acid peri-operative; "
            "DEVELOPMENTAL: early speech + physio + learning support; "
            "LEOPARD: cardiomyopathy Tx (same as PTPN11-NS); sensorineural hearing aids; dermatology lentigines monitoring"
        ),
        "key_ddx": (
            "Turner syndrome: 45,X or mosaic; only females; no PTPN11 mutation; karyotype first; "
            "RAF1-NS5: same facial features + MORE severe HCM (75% vs 20%); gene panel needed; "
            "SOS1-NS4: same facies + LESS HCM + taller stature + preserved intellect; "
            "LEOPARD syndrome: lentigines + HCM + deafness; PTPN11 Tyr279/Thr468 alleles; "
            "KRAS-NS3: intermediate; ± café-au-lait; often more severe cognitive; "
            "NF1 (Neurofibromatosis type 1): Café-au-lait macules + Lisch nodules; NF1 gene; rarely facies"
        ),
        "cardiac_pattern": "Pulmonary stenosis (50–80%) > ASD (10–15%) > HCM (20%) — PS pathognomonic for NS among RASopathies",
        "height_pattern": "Adult mean −2 SD; GH on-label; mildest growth failure among major NS alleles",
        "cognitive_pattern": "Mild ID 25%; IQ 80–90 median; language delay commonest; learning support 40%",
        "severity_weights": [0.15, 0.60, 0.25],  # mild/moderate/severe
    },

    # ── RAF1 — Noonan Syndrome Type 5 with HCM ───────────────────────────────
    {
        "gene": "RAF1",
        "protein": "RAF1 (Rapidly Accelerated Fibrosarcoma 1 / CRAF)",
        "alias": (
            "RAF1; OMIM gene 164760; Noonan Syndrome 5 (NS5) #611553 / LEOPARD Syndrome 2 #611554; 3p25.2; 648 aa; ~73 kDa; "
            "AD GOF; accounts for 3–17% of Noonan syndrome; "
            "HCM present in 75% of RAF1-NS vs only 20% of PTPN11-NS — most severe HCM in RASopathies; "
            "RAF kinase links RAS-GTP to MEK1/2-ERK1/2 cascade"
        ),
        "aa": "648 aa",
        "kDa": "~73 kDa",
        "locus": "3p25.2",
        "omim_gene": 164760,
        "omim_disease": 611553,
        "inheritance": "AD GOF (predominantly de novo)",
        "gene_class": (
            "Serine/threonine-protein kinase; RAF family (ARAF/BRAF/RAF1); "
            "normal activation: RAS-GTP → recruits RAF1 to plasma membrane → phosphorylation of S338/Y341 → "
            "full RAF1 kinase activation → phosphorylates MEK1/2 → ERK1/2 activation → growth/survival/differentiation; "
            "GOF mutations cluster in the CR2 (14-3-3 binding) and kinase (CR3) domains: "
            "Ser257Leu and Leu613Val are the two most common RAF1 Noonan alleles; "
            "Ser257Leu: N-terminal CR2 → disrupts 14-3-3-mediated auto-inhibition → constitutive RAF1-MEK-ERK; "
            "Leu613Val: C-terminal CR3 → kinase domain → hyperactivation; "
            "RAF1 GOF → strongest MAPK activation among NS alleles → most severe HCM; "
            "LEOPARD RAF1 alleles: Thr543Met, Ser257Leu — HCM + lentigines overlap with PTPN11-LEOPARD; "
            "RAF1 also activates MEK independently of RAS via SRC-mediated phosphorylation at Y341"
        ),
        "phenotype": (
            "Facial: same NS facies as PTPN11 (hypertelorism, ptosis, low-set ears, pterygium colli); "
            "CARDIAC: HCM in 75% (highest among NS alleles); often SEVERE obstructive HCM; "
            "septal thickness may require surgical myectomy or alcohol ablation; "
            "pulmonary stenosis 20–30% (less than PTPN11-NS); "
            "Short stature: similar to PTPN11-NS; GH-responsive; "
            "Cognitive: mild-moderate ID similar to PTPN11-NS; "
            "LEOPARD RAF1 overlap: lentigines + severe HCM; distinguish by lentigines distribution; "
            "Lymphoedema, cryptorchidism, coagulopathy as PTPN11-NS"
        ),
        "hallmark": (
            "SEVERE HCM in a Noonan-phenotype patient — RAF1 mutation most likely; "
            "HCM BEFORE cardiac symptoms (75% RAF1 vs 20% PTPN11) — cardiac MRI/echo annual; "
            "RAF1 Ser257Leu (most common allele) — severe obstructive HCM; "
            "OBSTRUCTIVE HCM with LVEDP elevation — surgical myectomy/septal ablation may be required earlier than PTPN11-NS; "
            "DISTINGUISH from PTPN11: RAF1-HCM 75% vs PTPN11-HCM 20% — severity guides workup priority; "
            "PRE-OP COAG: same mandatory as PTPN11-NS; Factor XI/vWF screen before myectomy"
        ),
        "treatment_alert": (
            "HCM MANAGEMENT IS PRIORITY in RAF1-NS: "
            "beta-blocker (bisoprolol/metoprolol) first line for obstructive HCM; "
            "MAVACAMTEN (CAMZYOS): approved FDA 2022 for obstructive HCM — first myosin inhibitor; "
            "consider for severe obstructive RAF1-HCM failing beta-blockade; "
            "SURGICAL SEPTAL MYECTOMY: for severe LVOTO (gradient >50 mmHg) refractory to medical Rx; "
            "ALCOHOL SEPTAL ABLATION: alternative to surgery in adult RAF1-NS (not children — scar risk); "
            "AVOID DIGOXIN in obstructive HCM — worsens LVOTO; "
            "AVOID VIGOROUS EXERCISE: risk of arrhythmia + sudden death in severe HCM; "
            "SCD RISK: annual ILR/Holter if syncope; ICD for sustained VT/VF; "
            "GH THERAPY: on-label but CARDIAC ECHO MANDATORY before start — GH can worsen HCM transiently; "
            "MEKi TRAMETINIB: investigational for RAF1-HCM (reduces ERK activation); "
            "COAGULOPATHY: pre-op screen mandatory"
        ),
        "key_ddx": (
            "PTPN11-NS: HCM only 20% vs RAF1 75%; PTPN11 is more common (50% NS); gene panel needed; "
            "Sarcomere HCM (MYH7, MYBPC3): NOT a RASopathy; no facial features; family history different; "
            "Pompe disease: HCM + hypotonia + macroglossia; GAA enzyme assay; "
            "LEOPARD/PTPN11-Tyr279: lentigines + HCM; distinguish from RAF1-LEOPARD by allele; "
            "BRAF-CFC: severe cognitive + ichthyosis; less HCM"
        ),
        "cardiac_pattern": "HCM 75% (most severe among NS alleles); LVOTO common; PS less prominent than PTPN11-NS",
        "height_pattern": "Adult −2 SD; GH on-label; ECHO before GH start mandatory",
        "cognitive_pattern": "Mild-moderate ID similar to PTPN11-NS; IQ 75–90",
        "severity_weights": [0.10, 0.50, 0.40],  # mild/moderate/severe
    },

    # ── SOS1 — Noonan Syndrome Type 4 ────────────────────────────────────────
    {
        "gene": "SOS1",
        "protein": "SOS1 (Son of Sevenless 1 — RAS-Specific Guanine Nucleotide Exchange Factor)",
        "alias": (
            "SOS1; OMIM gene 182530; Noonan Syndrome 4 (NS4) #610733; 2p22.1; 1333 aa; ~152 kDa; "
            "AD GOF; accounts for 10–13% of Noonan syndrome; "
            "MILDEST NS allele: near-normal intellect, tallest adult height, LEAST HCM; "
            "SOS1 is a RAS GEF (guanine nucleotide exchange factor) that loads RAS with GTP"
        ),
        "aa": "1333 aa",
        "kDa": "~152 kDa",
        "locus": "2p22.1",
        "omim_gene": 182530,
        "omim_disease": 610733,
        "inheritance": "AD GOF (de novo and familial)",
        "gene_class": (
            "RAS-specific guanine nucleotide exchange factor; DH + PH + REM + CDC25 + PR + SH3 domains; "
            "CDC25 homology domain catalyses GDP→GTP exchange on RAS (activating step); "
            "allosteric regulation: inactive conformation maintained by DH-PH intramolecular contacts; "
            "SOS1 GOF mutations (Arg552Gly — most common; Met269Ile) → release of auto-inhibition → "
            "constitutive RAS-GTP loading → sustained RAS-RAF-MEK-ERK; "
            "SOS1 also activates RAC1 via DH domain → cytoskeletal remodelling; "
            "SOS1 is recruited to activated RTKs via GRB2 adaptor (GRB2-SH2 binds pTyr on RTKs); "
            "SOSi pan-RAS inhibitor BI-3406 targets SOS1-KRAS interface — clinical trials ongoing; "
            "uniquely among NS alleles, SOS1 GOF generates least sustained ERK activation → mild phenotype"
        ),
        "phenotype": (
            "Facial: NS facies (hypertelorism, ptosis, ear anomalies) — PRESENT but often MILD; "
            "CARDIAC: pulmonary stenosis 30–40% (lower than PTPN11-NS); HCM RARE (<5%); "
            "ASD 10%; "
            "Height: tallest among NS alleles — mean adult height only −1 SD; "
            "Cognitive: NEAR-NORMAL intellect (IQ 90–100 typical); school performance often age-appropriate; "
            "Macrocephaly (68%); ectodermal anomalies — ectropion eyelids, hyperpigmented nevi; "
            "Coagulopathy: Factor XI/XII deficiency same as PTPN11-NS — pre-op screen still MANDATORY; "
            "Lymphoedema lower frequency than PTPN11-NS"
        ),
        "hallmark": (
            "NOONAN PHENOTYPE + NEAR-NORMAL INTELLECT — think SOS1; "
            "TALLEST STATURE among NS alleles — GH may NOT be required; measure growth curve carefully; "
            "LEAST HCM among NS alleles (<5% vs 20% PTPN11 vs 75% RAF1); "
            "MACROCEPHALY (68%) — distinctive feature of SOS1-NS vs other alleles; "
            "ECTODERMAL: ectropion eyelids + hyperpigmented nevi — dermatology referral; "
            "FAMILIAL NS: SOS1 alleles more often FAMILIAL than PTPN11 (which is 60% de novo); "
            "PRE-OP COAG: still mandatory despite mild phenotype"
        ),
        "treatment_alert": (
            "PS: balloon valvuloplasty same threshold as PTPN11 (gradient >50 mmHg); "
            "HCM: RARE — if present, same beta-blockade as RAF1-NS; echo annual; "
            "GH: on-label for NS; however SOS1 patients often have near-normal height — reassess need annually; "
            "INTELLECTUAL SUPPORT: less needed than PTPN11/RAF1-NS; appropriate school assessment; "
            "COAGULOPATHY: pre-op Factor XI/XII/vWF screen MANDATORY even though phenotype mild; "
            "SKIN: dermatology for ectropion + nevi surveillance; "
            "SOSi BI-3406 (investigational): shows preclinical promise for SOS1-driven cancers but NOT used in NS yet"
        ),
        "key_ddx": (
            "PTPN11-NS: more common (50% NS); more HCM; more ID; shorter stature; gene panel; "
            "RAF1-NS: severe HCM; less common; gene panel; "
            "KRAS-NS3: more severe cognitive; café-au-lait possible; "
            "Normal variant short stature: no facial features; no cardiac; karyotype + panel negative"
        ),
        "cardiac_pattern": "Pulmonary stenosis 30–40%; HCM RARE (<5%); mildest cardiac involvement of NS alleles",
        "height_pattern": "Mildest growth failure (−1 SD); near-normal; GH less often needed",
        "cognitive_pattern": "Near-normal IQ 90–100; school performance often age-appropriate; least ID of NS alleles",
        "severity_weights": [0.40, 0.50, 0.10],  # mild/moderate/severe
    },

    # ── KRAS — Noonan Syndrome Type 3 / CFC Spectrum ─────────────────────────
    {
        "gene": "KRAS",
        "protein": "KRAS (Kirsten Rat Sarcoma Viral Proto-Oncogene GTPase)",
        "alias": (
            "KRAS; OMIM gene 190070; Noonan Syndrome 3 (NS3) #609942 / CFC1B #615278; 12p12.1; 189 aa (KRAS4B isoform); ~21 kDa; "
            "AD GOF (germline); accounts for 2–5% of Noonan syndrome; "
            "KRAS is the most commonly somatically mutated oncogene in human cancer (NSCLC, CRC, PDAC); "
            "germline KRAS alleles are HYPOMORPHIC relative to somatic cancer alleles"
        ),
        "aa": "189 aa (KRAS4B)",
        "kDa": "~21 kDa",
        "locus": "12p12.1",
        "omim_gene": 190070,
        "omim_disease": 609942,
        "inheritance": "AD GOF (de novo; virtually always)",
        "gene_class": (
            "Small GTPase; RAS superfamily; switch I (30–38) + switch II (59–67) dynamic regions; "
            "KRAS cycles between GTP (active) and GDP (inactive) states; "
            "GTPase-accelerating proteins (GAPs: NF1/RasGAP) hydrolyse GTP → GDP (inactivation); "
            "GEFs (SOS1) catalyse GDP→GTP exchange (activation); "
            "oncogenic mutations: Gly12/13/61 — abolish intrinsic GTPase activity → constitutive GTP-bound; "
            "GERMLINE NS alleles (Thr58Ile, Pro34Arg, Leu19Val): partial GOF → hypomorphic → viable; "
            "somatic cancer alleles (Gly12Val, Gly12Asp): stronger GOF → lethal if germline (no NS patients with Gly12 alleles); "
            "KRAS4B is the dominant isoform: C-terminal CAAX box → farnesylation → plasma membrane targeting; "
            "RAS → RAF1/BRAF → MEK1/2 → ERK1/2 (proliferation) AND "
            "RAS → PI3K → AKT → mTOR (survival); "
            "KRAS G12C (sotorasib/adagrasib) — first approved direct KRAS inhibitor — NOT applicable to germline NS alleles"
        ),
        "phenotype": (
            "Facial: NS facies (hypertelorism, ptosis, low ears); often MORE SEVERE than PTPN11-NS; "
            "Cognitive: MODERATE intellectual disability (IQ 60–80) more common than PTPN11-NS; "
            "CARDIAC: PS + ASD; HCM 15–20%; "
            "CFC overlap features: sparse curly hair, hyperkeratosis, ectropion, multiple nevi; "
            "Short stature: more severe; GH-responsive; "
            "Lymphoedema; cryptorchidism; "
            "HAEMATOLOGY: café-au-lait macules in some; transient myeloproliferative disorder (rare); "
            "KRAS-NS may overlap phenotypically with BRAF/MAP2K1-CFC — molecular diagnosis required"
        ),
        "hallmark": (
            "NOONAN PHENOTYPE + MORE SEVERE COGNITIVE + CFC OVERLAP FEATURES — think KRAS; "
            "CAFÉ-AU-LAU MACULES in a NS-phenotype patient — KRAS or NF1 overlap; "
            "CFC OVERLAP: sparse curly hair + ectropion + hyperkeratosis alongside NS facies; "
            "MOLECULAR DISTINCTION: KRAS NS3 vs BRAF/MAP2K1 CFC — phenotypic overlap; gene panel essential; "
            "GERMLINE VS SOMATIC KRAS: germline alleles (Thr58Ile etc.) are NOT the Gly12 cancer alleles — "
            "do NOT apply sotorasib/adagrasib (KRASG12C inhibitor) to NS3 patients"
        ),
        "treatment_alert": (
            "CARDIAC: PS balloon valvuloplasty + HCM beta-blockade same as PTPN11-NS; "
            "COGNITIVE: earlier intensive developmental support; "
            "educational placement assessment early (IQ 60–80 range); "
            "GH: on-label; monitor for worsening lymphoedema; "
            "MEKi TRAMETINIB / COBIMETINIB: investigational for CFC including KRAS-variant (Phase II ongoing); "
            "DO NOT USE SOTORASIB (KRASG12C-specific): NS3 alleles are NOT Gly12Cys; "
            "DO NOT USE VEMURAFENIB (BRAFi): activates paradoxical CRAF signalling in KRAS-driven cells; "
            "COAGULOPATHY: standard NS pre-op coag screen; "
            "CANCER SURVEILLANCE: KRAS germline — borderline increased risk; "
            "haematology review if cytopenias develop (JMML-like rare but reported)"
        ),
        "key_ddx": (
            "PTPN11-NS: more common; milder cognitive; less CFC overlap; panel needed; "
            "BRAF-CFC: more severe; ichthyosis; essentially no PS; "
            "MAP2K1-CFC: same as BRAF-CFC; "
            "NF1: café-au-lait + Lisch nodules; NF1 gene; "
            "Somatic KRAS cancer: Gly12 alleles; NOT germline"
        ),
        "cardiac_pattern": "PS 30–40%; HCM 15–20%; ASD 10%; intermediate between typical NS and CFC",
        "height_pattern": "−2 to −3 SD; more severe than PTPN11-NS; GH on-label",
        "cognitive_pattern": "Moderate ID IQ 60–80 more common; intermediate between NS and CFC",
        "severity_weights": [0.15, 0.50, 0.35],  # mild/moderate/severe
    },

    # ── BRAF — CFC Syndrome ───────────────────────────────────────────────────
    {
        "gene": "BRAF",
        "protein": "BRAF (B-Raf Proto-Oncogene Serine/Threonine Kinase)",
        "alias": (
            "BRAF; OMIM gene 164757; CFC Syndrome 1 #115150 / LEOPARD-like; 7q34; 766 aa; ~84 kDa; "
            "AD GOF; accounts for 50–75% of CFC syndrome; "
            "most commonly mutated gene in CFC; "
            "BRAF V600E is the archetypal oncogenic BRAF allele in melanoma — "
            "germline CFC alleles (Gln257Arg, Glu501Lys, Lys499Glu) are DIFFERENT from V600E"
        ),
        "aa": "766 aa",
        "kDa": "~84 kDa",
        "locus": "7q34",
        "omim_gene": 164757,
        "omim_disease": 115150,
        "inheritance": "AD GOF (virtually always de novo in CFC)",
        "gene_class": (
            "Serine/threonine protein kinase; RAF family; "
            "CR1 (RAS-binding RBD + CRD) + CR2 (regulatory) + CR3 (kinase) domains; "
            "BRAF is the predominant RAF isoform that phosphorylates MEK1/2 directly; "
            "canonical pathway: RAS-GTP → BRAF dimerisation + conformational activation → MEK1/2-Ser218/Ser222 phosphorylation → ERK1/2 → proliferation; "
            "V600E (somatic melanoma): activates BRAF as monomer → sensitive to vemurafenib/dabrafenib; "
            "CFC germline alleles (Gln257Arg, Glu501Lys): kinase-activating mutations in DFG motif region → "
            "hypermorphic but NOT V600E → NOT sensitive to vemurafenib (paradoxical ERK activation via CRAF); "
            "paradoxical activation: RAF inhibitors in RAS-active context → CRAF dimerisation + ERK elevation; "
            "BRAF GOF → strongest ERK activation in CFC family → severe phenotype with ectodermal features; "
            "also mutated in NS ≈ 5% (milder alleles), LEOPARD-like (CFC overlap)"
        ),
        "phenotype": (
            "Facial: CFC triad — frontal bossing + sparse/absent eyebrows + sparse curly/ectodermal hair; "
            "hypertelorism, ptosis, low ears as NS but FACE more coarse; "
            "ECTODERMAL: ichthyosis/hyperkeratosis (60%); eczema; absent/sparse eyebrows; hyperpigmented lentigines; "
            "COGNITIVE: MODERATE-SEVERE intellectual disability (IQ 40–70); "
            "seizures (50%); Chiari malformation; ventriculomegaly; "
            "CARDIAC: HCM (40–50%); PS (25–35%); dysrhythmia; essentially LESS PS vs NS but MORE HCM vs SOS1-NS; "
            "Short stature −3 SD; macrocephaly; feeding difficulties in infancy; "
            "NO effective cancer predisposition (unlike HRAS-Costello); "
            "NO JMML (unlike CBL or PTPN11 with specific alleles)"
        ),
        "hallmark": (
            "NOONAN PHENOTYPE + ICHTHYOSIS + ABSENT EYEBROWS + SEVERE COGNITIVE — think BRAF-CFC; "
            "FRONTAL BOSSING + SPARSE CURLY HAIR + HYPERKERATOSIS — CFC triad; "
            "SEIZURES in 50% of CFC (vs ~15% NS) — early EEG; "
            "DISTINGUISH from PTPN11-NS: CFC facies MORE COARSE + ichthyosis + absent eyebrows + severe cognitive; "
            "DO NOT USE VEMURAFENIB/DABRAFENIB: V600E-specific — NOT effective for CFC BRAF alleles; "
            "paradoxical ERK activation if used in RAS-active context; "
            "MEKi (trametinib) is the rational agent downstream of BRAF — investigational CFC trials"
        ),
        "treatment_alert": (
            "HCM: beta-blockade + mavacamten for obstructive HCM as per RAF1-NS; "
            "SEIZURES: standard AED management (LEV first line given POLG exclusion); "
            "ECTODERMAL: emollient + keratolytic (urea/lactic acid) for hyperkeratosis/ichthyosis; "
            "ophthalmology for ectropion + corneal exposure; "
            "COGNITIVE: intensive developmental programme; IEP at school; "
            "GH: on-label for CFC short stature; cardiac echo before GH start; "
            "MEKi TRAMETINIB: the most biologically rational agent for BRAF-CFC (MEK is downstream of BRAF GOF); "
            "AVOID VEMURAFENIB / DABRAFENIB (BRAF V600E-specific): "
            "CFC alleles are NOT V600E; vemurafenib paradoxically activates ERK via CRAF dimer; "
            "CANCER SURVEILLANCE: CFC has LOW cancer risk (unlike HRAS-Costello or CBL); no JMML; "
            "COAGULOPATHY: pre-op coag screen same as NS"
        ),
        "key_ddx": (
            "MAP2K1-CFC: phenotypically identical to BRAF-CFC; gene panel needed; "
            "KRAS-NS3: milder; less ectodermal; intermediate phenotype; "
            "Costello/HRAS: papillomata + RMS risk; different allele; HRAS gene; "
            "NS/PTPN11: less severe cognitively; less ectodermal; more PS; "
            "Cardiofaciocutaneous not fitting panel: FMR1 expansion; Angelman; chromosomal microarray needed"
        ),
        "cardiac_pattern": "HCM 40–50%; PS 25–35%; dysrhythmia 15%; more HCM than SOS1-NS but less PS than PTPN11-NS",
        "height_pattern": "−3 SD severe; GH on-label; cardiac echo before GH start mandatory",
        "cognitive_pattern": "Moderate-severe ID (IQ 40–70); seizures 50%; most severe cognitive of NS spectrum",
        "severity_weights": [0.05, 0.40, 0.55],  # mild/moderate/severe
    },

    # ── MAP2K1 — CFC Syndrome MEK1 ────────────────────────────────────────────
    {
        "gene": "MAP2K1",
        "protein": "MEK1 (Mitogen-Activated Protein Kinase Kinase 1 / MAP2K1)",
        "alias": (
            "MAP2K1; OMIM gene 176872; CFC Syndrome 3 #615279; 15q22.31; 393 aa; ~44 kDa; "
            "AD GOF; accounts for 15–25% of CFC syndrome together with MAP2K2; "
            "MEK1/2 are the immediate downstream effectors of BRAF and RAF1; "
            "trametinib (MEKINIST) and cobimetinib are approved MEK inhibitors for cancer (BRAF V600E+ melanoma)"
        ),
        "aa": "393 aa",
        "kDa": "~44 kDa",
        "locus": "15q22.31",
        "omim_gene": 176872,
        "omim_disease": 615279,
        "inheritance": "AD GOF (de novo)",
        "gene_class": (
            "Dual-specificity kinase (phosphorylates ERK1/2 on Thr/Tyr); "
            "MEK1 and MEK2 (MAP2K2) are functionally redundant MAPK kinases; "
            "activated by RAF1 or BRAF (Ser218/Ser222 phosphorylation of MEK1 activation loop); "
            "MEK1 → phospho-ERK1/2 (Thr202/Tyr204) → nucleus → transcription factors (Elk-1, Fos); "
            "negative feedback: phospho-ERK phosphorylates SOS1 → reduces GEF activity; "
            "CFC MAP2K1 GOF alleles (Phe53Leu, Tyr130Cys, Gln56Pro): "
            "activation loop mutations → constitutive MEK1 kinase activity without RAF phosphorylation input; "
            "MEK inhibitors block the ATP-binding cleft of MEK1/2 as non-competitive allosteric inhibitors; "
            "trametinib (0.5–2 mg/day oral) approved for BRAF V600 + melanoma/NSCLC/thyroid; "
            "investigational for RASopathy CFC in Phase II (LOVS1K trial, Novartis, ongoing)"
        ),
        "phenotype": (
            "IDENTICAL TO BRAF-CFC clinically: "
            "CFC triad — frontal bossing + absent/sparse eyebrows + ectodermal anomalies; "
            "ECTODERMAL: hyperkeratosis/ichthyosis + eczema + curly sparse hair; "
            "COGNITIVE: moderate-severe ID similar to BRAF-CFC (IQ 40–70); "
            "seizures 50%; "
            "CARDIAC: HCM 40–50%; PS 25–35%; "
            "Short stature −3 SD; macrocephaly; "
            "MOLECULAR DISTINCTION: MAP2K1 vs BRAF-CFC requires gene panel — phenotype identical; "
            "MAP2K1 accounts for CFC when BRAF negative — do not stop panel at BRAF"
        ),
        "hallmark": (
            "CFC DIAGNOSIS WHEN BRAF NEGATIVE — MAP2K1 (+ MAP2K2) next most common CFC gene; "
            "BRAF-NEGATIVE CFC: sequence MAP2K1 + MAP2K2; "
            "PHENOTYPE INDISTINGUISHABLE from BRAF-CFC: facial coarsening + ichthyosis + severe cognitive + HCM; "
            "MEKi TRAMETINIB IS THE MOST DIRECTLY RATIONAL AGENT: "
            "MAP2K1 GOF activates MEK constitutively → trametinib directly inhibits MAP2K1 product (MEK1 kinase); "
            "DISTINGUISH from BRAF-CFC only by gene panel — no clinical differentiator"
        ),
        "treatment_alert": (
            "MEKi TRAMETINIB: most rational agent for MAP2K1-CFC (trametinib directly inhibits GOF MAP2K1); "
            "LOVS1K Phase II trial ongoing — refer eligible CFC patients to trial sites; "
            "HCM: mavacamten + beta-blockade as per BRAF-CFC; "
            "ECTODERMAL: same emollient + keratolytic as BRAF-CFC; "
            "COGNITIVE + SEIZURES: same developmental programme + LEV-first AED; "
            "GH: on-label; cardiac echo before start; "
            "DO NOT USE VEMURAFENIB (BRAFi): does NOT target MAP2K1 and causes paradoxical ERK activation; "
            "COBIMETINIB: same MEK inhibitor class as trametinib; investigational for CFC; "
            "pre-op coag screen standard"
        ),
        "key_ddx": (
            "BRAF-CFC: phenotypically identical; BRAF gene; panel needed; "
            "MAP2K2-CFC: same gene family; CFC4 #615280; "
            "KRAS-NS3/CFC overlap: less ectodermal; panel; "
            "Normal-pressure hydrocephalus + dysmorphia: CMA + exome to exclude chromosomal"
        ),
        "cardiac_pattern": "HCM 40–50%; PS 25–35%; identical to BRAF-CFC; MEKi rationale highest for MAP2K1",
        "height_pattern": "−3 SD; GH on-label; cardiac before GH mandatory",
        "cognitive_pattern": "Moderate-severe ID IQ 40–70; seizures 50%; identical to BRAF-CFC",
        "severity_weights": [0.05, 0.40, 0.55],  # mild/moderate/severe
    },

    # ── HRAS — Costello Syndrome ──────────────────────────────────────────────
    {
        "gene": "HRAS",
        "protein": "HRAS (Harvey Rat Sarcoma Viral Proto-Oncogene GTPase)",
        "alias": (
            "HRAS; OMIM gene 190020; Costello Syndrome #218040; 11p15.5; 189 aa; ~21 kDa; "
            "AD GOF (de novo); accounts for virtually all Costello syndrome; "
            "p.Gly12Ser (c.34G>A) is the FOUNDER allele (>80% of Costello); "
            "HRAS is the smallest and most constitutively membrane-associated RAS isoform"
        ),
        "aa": "189 aa",
        "kDa": "~21 kDa",
        "locus": "11p15.5",
        "omim_gene": 190020,
        "omim_disease": 218040,
        "inheritance": "AD GOF (virtually always de novo; paternal age effect)",
        "gene_class": (
            "Small GTPase; RAS superfamily; 89% identical to KRAS4B and NRAS at protein level; "
            "HRAS-specific: constitutively palmitoylated at Cys181/184 → tighter plasma membrane attachment; "
            "less exchange between plasma membrane and endomembranes than KRAS (which uses electrostatic farnesyl only); "
            "Gly12Ser: glycine→serine at position 12 → steric clash with GAP arginine finger → GTPase abolished → "
            "constitutive HRAS-GTP → strongest sustained RAS-ERK + PI3K-AKT activation in the RASopathy family; "
            "Gly12Ala, Gly12Asp, Gly13Cys: other Costello alleles (milder); "
            "CANCER RISK: Gly12Ser generates highest RAS-ERK flux among germline RASopathy alleles → "
            "RHABDOMYOSARCOMA 15% lifetime; BLADDER TRANSITIONAL CELL CARCINOMA 5%; NEUROBLASTOMA rare; "
            "CARDIAC ARRHYTHMIA: Gly12Ser activates HCN4 via PI3K → prolonged QTc + SVT risk"
        ),
        "phenotype": (
            "DISTINCTIVE FEATURES distinguishing Costello from NS/CFC: "
            "NASAL / PERIANAL PAPILLOMATA (40–50%) — PATHOGNOMONIC for Costello; appear age 2–7 years; "
            "benign HPV-negative fibroepithelial polyps (do NOT confuse with viral warts); "
            "CARDIAC: HCM (60%); PS (40%); ATRIAL TACHYCARDIA/SVT (50%); ectopic atrial rhythm; QTc prolongation; "
            "Neonatal: severe failure to thrive + feeding difficulties + polyhydramnios + macrosomia; "
            "coarse facies: full cheeks, depressed nasal bridge, curly hair; "
            "deep palmar/plantar creases; loose skin; "
            "Cognitive: moderate ID (IQ 50–75); "
            "SHORT STATURE (severe); "
            "CANCER: RMS 15%, TCC bladder 5% — ongoing annual surveillance mandatory"
        ),
        "hallmark": (
            "NASAL PAPILLOMATA IN A CHILD WITH CARDIAC ANOMALIES — PATHOGNOMONIC for Costello/HRAS; "
            "PERIANAL PAPILLOMATA — PATHOGNOMONIC; HPV-negative fibroepithelial polyps; "
            "ATRIAL TACHYCARDIA + HCM + PAPILLOMATA — Costello triad; "
            "RMS SURVEILLANCE: annual abdominal/pelvic MRI + urine cytology from diagnosis to age 10; "
            "AVOID RAPID IV FLUID BOLUS: QTc prolongation + SVT risk → cardiac arrhythmia death reported with rapid rehydration; "
            "SVT 50%: Holter + cardiologist involvement; beta-blocker if SVT or QTc >500 ms; "
            "CANCER: HIGHEST cancer risk of all RASopathies; NOT a typical NS case — dedicated oncology surveillance"
        ),
        "treatment_alert": (
            "PAPILLOMATA: ENT for nasal polyp debulking (surgical; recurrence common); "
            "dermatology for perianal; surveillance for RCC/RMS transformation; "
            "CARDIAC: "
            "HCM → beta-blockade + mavacamten for obstruction; "
            "SVT → beta-blocker (propranolol) first; adenosine for acute SVT; "
            "QTc MONITORING: baseline + 3-monthly ECG; avoid QTc-prolonging drugs (ondansetron, domperidone, azithromycin); "
            "AVOID RAPID IV REHYDRATION: bolus 0.9% NaCl → triggered SVT/VF deaths reported; "
            "use slow iv infusion (max 10 mL/kg/hr) + telemetry monitoring during any IV fluids; "
            "CANCER SURVEILLANCE (MANDATORY): "
            "abdominal/pelvic MRI every 3–4 months until age 8–10 (RMS highest risk); "
            "urine cytology annually from age 10 (bladder TCC); "
            "GH: INVESTIGATIONAL (NOT standard) for Costello — concern that GH → increased RAS signalling → ↑RMS risk; "
            "MEKi TRAMETINIB: rationale for HCM + growth; investigational; discuss via trial only; "
            "COAGULOPATHY: standard pre-op screen"
        ),
        "key_ddx": (
            "BRAF/MAP2K1-CFC: no papillomata; less cardiac arrhythmia; no RMS; gene panel; "
            "PTPN11-NS: no papillomata; less HCM; milder; "
            "Neurofibromatosis type 1: Lisch nodules + café-au-lait; NF1 gene; no papillomata; "
            "Perianal warts (HPV): STI context; viral; PCR HPV positive; not Costello"
        ),
        "cardiac_pattern": "HCM 60%; PS 40%; SVT/ectopic atrial tachycardia 50%; QTc prolongation — most cardiac of NS family",
        "height_pattern": "Severe −3 SD; GH NOT standard (RMS risk concern); trametinib investigational",
        "cognitive_pattern": "Moderate ID IQ 50–75; intense developmental support; neonatal FTT common",
        "severity_weights": [0.05, 0.45, 0.50],  # mild/moderate/severe
    },

    # ── CBL — Noonan-like Syndrome with JMML ─────────────────────────────────
    {
        "gene": "CBL",
        "protein": "CBL (Casitas B-Lineage Lymphoma E3 Ubiquitin Ligase)",
        "alias": (
            "CBL; OMIM gene 165360; Noonan-like Syndrome with or without JMML #613563; 11q23.3; 906 aa; ~100 kDa; "
            "AD GOF (germline) + somatic second hit (uniparental disomy 11q) → JMML; "
            "accounts for ~3% of Noonan-like syndrome; "
            "CBL germline mutation + somatic loss of heterozygosity = JMML (20–50% of CBL-Noonan patients)"
        ),
        "aa": "906 aa",
        "kDa": "~100 kDa",
        "locus": "11q23.3",
        "omim_gene": 165360,
        "omim_disease": 613563,
        "inheritance": "AD GOF germline + somatic second-hit (UPD11q) for JMML",
        "gene_class": (
            "RING-type E3 ubiquitin ligase; TKB (tyrosine kinase binding) + RING finger + proline-rich tail domains; "
            "CBL negatively regulates RTKs (FLT3, KIT, PDGFR, EGFR) by ubiquitinating activated RTKs → "
            "receptor internalisation + lysosomal degradation → signal termination; "
            "RING domain mutations (Tyr371His, Cys381Arg — hot spot): abrogate E3 ligase activity → "
            "RTK ubiquitination fails → RAS-GTP sustained activation via SOS1/GRB2 without clearance; "
            "TWO-HIT MECHANISM: germline CBL GOF (loss of E3) + somatic UPD11q23 (uniparental disomy → "
            "second allele also RING-mutant) → clonal expansion → JMML; "
            "CBL-JMML: SPONTANEOUS REMISSION in some (unlike PTPN11-JMML which requires SCT); "
            "Noonan-like phenotype: CBL loss of E3 → increased RAS-ERK + PI3K flux similar to other RASopathies"
        ),
        "phenotype": (
            "NS-LIKE FACIES: hypertelorism, ptosis, low ears — similar to PTPN11-NS but often MILDER facies; "
            "CARDIAC: HCM (50%); PS (20%); ASD; often severe HCM; "
            "HAEMATOLOGICAL: JMML in 20–50% (spontaneous remission possible — OBSERVE FIRST); "
            "leukocytosis + monocytosis + thrombocytopenia + splenomegaly in JMML; "
            "Hyperpigmentation + café-au-lait macules; "
            "VASCULITIS: medium vessel vasculitis (10%); "
            "Cryptorchidism; lymphoedema; "
            "Short stature; GH-responsive; "
            "Cognitive: mild ID (IQ 75–90); "
            "CANCER: elevated RAS signalling → long-term haematological surveillance mandatory"
        ),
        "hallmark": (
            "NOONAN-LIKE + JMML IN CHILD — CBL mutation most likely (alongside PTPN11); "
            "CBL-JMML: SPONTANEOUS REMISSION in 40–60% — DO NOT RUSH TO SCT (unlike KRAS/PTPN11-JMML); "
            "ACTIVE WATCH-AND-WAIT for CBL-JMML: haematology + serial FBC/smear; "
            "VASCULITIS: medium vessel — MRA + skin biopsy; treat with steroids if active; "
            "HYPERPIGMENTATION + CAFÉ-AU-LAU MACULES in an NS-like patient — think CBL; "
            "HCM SEVERITY: 50% HCM; similar to RAF1-NS; echocardiography annually; "
            "TWO-HIT CBL: somatic UPD11q23 confirms JMML clonal origin — bone marrow FISH + array"
        ),
        "treatment_alert": (
            "JMML MANAGEMENT: "
            "CBL-JMML — OBSERVE FIRST (spontaneous remission 40–60% in CBL vs <5% in other JMML); "
            "repeat FBC/smear every 4 weeks + liver/spleen USS; "
            "STEM CELL TRANSPLANT (SCT): ONLY if progressive JMML (rising WBC >50, worsening splenomegaly); "
            "do NOT use azacitidine (used in KRAS/PTPN11-JMML) as first line — observe first; "
            "HCM: mavacamten + beta-blockade; echo annually; "
            "VASCULITIS: METHYLPREDNISOLONE 1mg/kg/day → taper; methotrexate for steroid-sparing; "
            "GH: on-label; cardiac echo before start; "
            "CANCER SURVEILLANCE: haematological (FBC annual lifelong); "
            "COAGULOPATHY: standard NS pre-op screen; "
            "MEKi TRAMETINIB: investigational for CBL-driven HCM/JMML (Phase I/II); "
            "FAMILY SCREENING: 1st-degree relatives of germline CBL; cascade genetic testing"
        ),
        "key_ddx": (
            "PTPN11-NS+JMML: PTPN11 JMML does NOT spontaneously remit — SCT standard; "
            "distinguish CBL vs PTPN11 JMML by gene panel — management differs critically; "
            "KRAS-NS3 JMML: similar; no spontaneous remission; "
            "NF1 + JMML: NF1 loss → JMML; NF1 gene; café-au-lait + Lisch nodules; "
            "CMML: adult; somatic CBL mutation in CMML — different from germline NS; "
            "NS without JMML: PTPN11/SOS1/RAF1 — no haematological malignancy"
        ),
        "cardiac_pattern": "HCM 50% (severe); PS 20%; ASD 15%; high HCM burden similar to RAF1-NS",
        "height_pattern": "−2 SD; GH on-label; cardiac echo before GH",
        "cognitive_pattern": "Mild ID IQ 75–90; vasculitis can affect CNS — neuroimaging if deficits",
        "severity_weights": [0.20, 0.50, 0.30],  # mild/moderate/severe
    },
]


# ─── Patient simulation ────────────────────────────────────────────────────────

def _simulate_gene(gene_def: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    patients = []
    sev_w = gene_def.get("severity_weights", [0.20, 0.60, 0.20])
    severities = rng.choices(["Mild", "Moderate", "Severe"], weights=sev_w, k=n)

    # Cardiac patterns per gene
    cardiac_map = {
        "PTPN11": {"PS": 65, "HCM": 20, "ASD": 12, "None": 3},
        "RAF1":   {"HCM": 75, "PS": 25, "ASD": 10, "None": 0},  # HCM dominant
        "SOS1":   {"PS": 35, "ASD": 12, "HCM": 4, "None": 49},
        "KRAS":   {"PS": 35, "HCM": 18, "ASD": 10, "None": 37},
        "BRAF":   {"HCM": 45, "PS": 28, "Arrhythmia": 12, "None": 15},
        "MAP2K1": {"HCM": 45, "PS": 27, "Arrhythmia": 11, "None": 17},
        "HRAS":   {"HCM": 60, "PS": 38, "SVT": 48, "None": 2},  # can have multiple
        "CBL":    {"HCM": 50, "PS": 20, "ASD": 15, "None": 15},
    }
    gene_name = gene_def["gene"]
    c_map = cardiac_map.get(gene_name, {"PS": 30, "HCM": 20, "None": 50})
    cardiac_options = list(c_map.keys())
    cardiac_weights = list(c_map.values())

    # Age at diagnosis (months — most present in infancy/childhood)
    for j in range(n):
        sev = severities[j]
        age_diag_mo = rng.randint(0, 72) if sev == "Severe" else rng.randint(6, 180)
        age_diag_yr = round(age_diag_mo / 12, 1)

        cardiac = rng.choices(cardiac_options, weights=cardiac_weights, k=1)[0]
        has_hcm = "HCM" in cardiac or (gene_name in ("RAF1", "HRAS", "CBL", "BRAF", "MAP2K1") and rng.random() < 0.2)

        jmml = gene_name == "CBL" and rng.random() < 0.30
        papillomata = gene_name == "HRAS" and rng.random() < 0.42
        vasculitis = gene_name == "CBL" and rng.random() < 0.12
        svt = gene_name == "HRAS" and rng.random() < 0.48
        ichthyosis = gene_name in ("BRAF", "MAP2K1") and rng.random() < 0.58
        coagulopathy = rng.random() < 0.28

        short_stature_sd = round(rng.uniform(-3.5, -1.0), 1)
        iq_est = max(25, int({
            "PTPN11": rng.gauss(85, 12),
            "RAF1": rng.gauss(80, 13),
            "SOS1": rng.gauss(95, 10),
            "KRAS": rng.gauss(72, 15),
            "BRAF": rng.gauss(58, 14),
            "MAP2K1": rng.gauss(58, 14),
            "HRAS": rng.gauss(65, 14),
            "CBL": rng.gauss(83, 11),
        }.get(gene_name, rng.gauss(80, 12))))

        cancer_risk = gene_name == "HRAS" and rng.random() < 0.20
        gh_started = sev in ("Moderate", "Severe") and rng.random() < 0.55

        patients.append({
            "id": f"{gene_name}-{seed}-{j+1:02d}",
            "gene": gene_name,
            "seed": seed,
            "age_diagnosis_yr": age_diag_yr,
            "severity": sev,
            "cardiac_primary": cardiac,
            "has_hcm": has_hcm,
            "jmml": jmml,
            "papillomata": papillomata,
            "vasculitis": vasculitis,
            "svt": svt,
            "ichthyosis": ichthyosis,
            "coagulopathy": coagulopathy,
            "short_stature_sd": short_stature_sd,
            "iq_estimate": iq_est,
            "cancer_surveillance": cancer_risk,
            "gh_started": gh_started,
        })
    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    hcm_n = sum(p["has_hcm"] for p in patients)
    jmml_n = sum(p["jmml"] for p in patients)
    papillomata_n = sum(p["papillomata"] for p in patients)
    svt_n = sum(p["svt"] for p in patients)
    ichthyosis_n = sum(p["ichthyosis"] for p in patients)
    coag_n = sum(p["coagulopathy"] for p in patients)
    gh_n = sum(p["gh_started"] for p in patients)
    mean_age = round(sum(p["age_diagnosis_yr"] for p in patients) / n, 1)
    mean_iq = round(sum(p["iq_estimate"] for p in patients) / n, 0)
    mean_ht_sd = round(sum(p["short_stature_sd"] for p in patients) / n, 2)
    sev_mild = sum(p["severity"] == "Mild" for p in patients)
    sev_mod = sum(p["severity"] == "Moderate" for p in patients)
    sev_sev = sum(p["severity"] == "Severe" for p in patients)
    return {
        "n": n,
        "hcm_pct": round(100 * hcm_n / n, 1),
        "jmml_pct": round(100 * jmml_n / n, 1),
        "papillomata_pct": round(100 * papillomata_n / n, 1),
        "svt_pct": round(100 * svt_n / n, 1),
        "ichthyosis_pct": round(100 * ichthyosis_n / n, 1),
        "coagulopathy_pct": round(100 * coag_n / n, 1),
        "gh_started_pct": round(100 * gh_n / n, 1),
        "mean_age_diagnosis_yr": mean_age,
        "mean_iq": int(mean_iq),
        "mean_short_stature_sd": mean_ht_sd,
        "severity_mild_pct": round(100 * sev_mild / n, 1),
        "severity_moderate_pct": round(100 * sev_mod / n, 1),
        "severity_severe_pct": round(100 * sev_sev / n, 1),
    }


def _all_patients() -> list:
    all_pts = []
    for i, ge in enumerate(RASOPATHY_GENES):
        seed = SEED_BASE + i
        pts = _simulate_gene(ge, seed, 40)
        all_pts.extend(pts)
    return all_pts


# ─── Public API functions ──────────────────────────────────────────────────────

def get_overview() -> dict:
    all_pts = _all_patients()
    agg = _cohort_stats(all_pts)
    return {
        "atlas_name": "Rasopathies Atlas",
        "atlas_subtitle": "Complete 8-Gene RAS/MAPK Pathway Disorders Atlas",
        "n_genes": 8,
        "n_patients": len(all_pts),
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
        "genes": [g["gene"] for g in RASOPATHY_GENES],
        "description": (
            "The Rasopathies Atlas covers 8 clinically actionable genes across the full RAS/MAPK pathway disorder spectrum: "
            "PTPN11 (the archetypal and most common Noonan syndrome gene, 50% of NS, SHP2 phosphatase upstream of RAS), "
            "RAF1 (Noonan NS5 with the most severe HCM, 75% HCM vs 20% for PTPN11), "
            "SOS1 (Noonan NS4, mildest allele — near-normal intellect and tallest stature), "
            "KRAS (Noonan NS3/CFC overlap — intermediate severity, moderate ID), "
            "BRAF (CFC syndrome — most common CFC gene, severe cognitive, ichthyosis, absent eyebrows), "
            "MAP2K1 (CFC MEK1 — phenotypically identical to BRAF-CFC; MEKi most directly rational), "
            "HRAS (Costello syndrome — nasal papillomata PATHOGNOMONIC, highest cancer risk of all RASopathies, "
            "rapid IV fluids DANGEROUS due to SVT/QTc), "
            "and CBL (Noonan-like + JMML — spontaneous JMML remission in 40–60% distinguishes it from PTPN11-JMML). "
            "RASopathies share a cascade: RTK → RAS → RAF → MEK → ERK, but each gene position produces a distinct "
            "clinical syndrome. Key pharmacological hierarchy: MEK inhibitors (trametinib, cobimetinib) act downstream "
            "of all upstream mutations — rationale is highest for MAP2K1 (direct GOF enzyme) and solid across CFC. "
            "320 patients (8 × 40, seeds 1230–1237)."
        ),
        "aggregate_clinical": agg,
        "drug_alerts": [
            {
                "title": "HRAS-Costello: AVOID RAPID IV FLUID BOLUS — arrhythmia and VF deaths reported",
                "body": (
                    "Gly12Ser HRAS activates HCN4-PI3K → prolonged QTc + ectopic atrial tachycardia. "
                    "Rapid rehydration (IV bolus) has triggered lethal arrhythmia in Costello patients. "
                    "Use slow infusion (max 10 mL/kg/hr) with cardiac telemetry monitoring for ANY IV fluids. "
                    "Baseline ECG + 3-monthly Holter. Beta-blocker if SVT or QTc >500 ms. "
                    "Avoid all QTc-prolonging co-medications (ondansetron, domperidone, azithromycin, haloperidol)."
                ),
            },
            {
                "title": "CBL-JMML: OBSERVE FIRST — spontaneous remission 40–60%; do NOT rush to SCT",
                "body": (
                    "CBL-JMML is fundamentally different from PTPN11/KRAS-JMML: the somatic second-hit (UPD11q23) "
                    "creates transient clonal dominance that remits spontaneously in 40–60%. "
                    "Management: serial FBC/smear every 4 weeks + USS spleen/liver; "
                    "SCT ONLY if progressive disease (rising WBC >50×10⁹/L, progressive splenomegaly, organ failure). "
                    "Do NOT use azacitidine as first-line for CBL-JMML. "
                    "Confirm CBL germline + UPD11q23 somatic by bone marrow FISH + array before treatment decisions."
                ),
            },
            {
                "title": "BRAF/MAP2K1-CFC: DO NOT USE VEMURAFENIB/DABRAFENIB — paradoxical ERK activation",
                "body": (
                    "BRAF V600E inhibitors (vemurafenib, dabrafenib) are approved for somatic V600E melanoma. "
                    "CFC germline BRAF alleles (Gln257Arg, Glu501Lys) are NOT V600E. "
                    "In RAS-active cells, RAF inhibitors paradoxically activate ERK via CRAF dimerisation. "
                    "MEK inhibitors (trametinib, cobimetinib) are the rational agents for BRAF/MAP2K1-CFC "
                    "— investigational via Phase II trials (LOVS1K). Refer eligible patients to trial sites."
                ),
            },
            {
                "title": "HRAS-Costello: GH NOT STANDARD — RMS risk concern; cancer surveillance MANDATORY",
                "body": (
                    "GH therapy is on-label for NS (PTPN11/RAF1/SOS1/KRAS/CBL) but is INVESTIGATIONAL (not standard) "
                    "for HRAS-Costello. Concern: GH amplifies RAS-IGF1R-ERK → may increase rhabdomyosarcoma risk. "
                    "Cancer surveillance MANDATORY from diagnosis: "
                    "abdominal/pelvic MRI every 3–4 months until age 8–10 (RMS window); "
                    "annual urine cytology from age 10 (bladder TCC). "
                    "MEKi trametinib via trial is preferred over GH for Costello growth failure."
                ),
            },
            {
                "title": "ALL NS alleles: PRE-OPERATIVE COAGULATION SCREEN MANDATORY",
                "body": (
                    "35% of Noonan syndrome patients have haematological coagulopathy: "
                    "Factor XI deficiency (20–30%), Factor XII deficiency, von Willebrand disease Type I, "
                    "platelet dysfunction. All NS variants (PTPN11, RAF1, SOS1, KRAS, CBL) — pre-op screen: "
                    "PT, APTT, Factor VIII/IX/XI/XII, vWF antigen + activity. "
                    "If Factor XI <30%: FFP or factor concentrate peri-operatively. "
                    "Tranexamic acid for dental/minor procedures. Never skip pre-op coag in NS."
                ),
            },
            {
                "title": "GH THERAPY: CARDIAC ECHO MANDATORY BEFORE STARTING in all NS alleles",
                "body": (
                    "GH is approved (on-label) for short stature in Noonan syndrome. "
                    "However GH can transiently increase HCM severity in RAF1-NS and other high-HCM alleles. "
                    "MANDATORY pre-GH cardiac workup: echocardiogram + 12-lead ECG. "
                    "If LVOTO gradient >50 mmHg at rest, optimise with beta-blockade/mavacamten BEFORE starting GH. "
                    "Annual echo on GH. Suspend GH if LVOTO worsens. "
                    "SOS1-NS often has near-normal height — reassess whether GH is needed annually."
                ),
            },
        ],
        "clinical_pearls": [
            "Rasopathy hierarchy: PTPN11 (50% NS) → RAF1 (3–17%) → SOS1 (10–13%) → BRAF-CFC (50% of CFC) → MAP2K1-CFC (15–25% CFC) → HRAS-Costello → KRAS-NS3 (2–5%) → CBL-JMML (3% NS-like)",
            "HCM frequency spectrum: RAF1 75% > HRAS 60% > CBL 50% > BRAF/MAP2K1 40–45% > KRAS 18% > PTPN11 20% > SOS1 <5%",
            "Cognitive severity spectrum (MOST to LEAST severe): BRAF/MAP2K1-CFC (IQ 40–70) > HRAS-Costello (IQ 50–75) > KRAS-NS3 (IQ 60–80) > PTPN11/RAF1-NS (IQ 75–90) > CBL-NS (IQ 75–90) > SOS1-NS (IQ 90–100)",
            "PS vs HCM dominance: PTPN11/SOS1 are PS-dominant; RAF1/CBL/HRAS/BRAF/MAP2K1 are HCM-dominant",
            "JMML: CBL (spontaneous remission 40–60%) vs PTPN11 (SCT required) — gene matters critically for management",
            "Papillomata PATHOGNOMONIC for Costello/HRAS: nasal + perianal; HPV-negative fibroepithelial polyps",
            "MEK inhibitor hierarchy: MAP2K1 > BRAF > RAF1 > PTPN11 > SOS1 (direct GOF to indirect GOF signal)",
            "KRAS germline alleles (Thr58Ile, Pro34Arg) are NOT G12 cancer alleles — sotorasib NOT applicable",
            "SOS1-NS: macrocephaly 68% + ectropion + hyperpigmented nevi — distinctive ectodermal cluster"
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for i, ge in enumerate(RASOPATHY_GENES):
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
            "cardiac_pattern": ge["cardiac_pattern"],
            "height_pattern": ge["height_pattern"],
            "cognitive_pattern": ge["cognitive_pattern"],
            "cohort_n": len(pts),
            "seed": seed,
            "stats": stats,
            "patients": pts,
        }
    return result


def get_definitions() -> dict:
    return {
        "atlas_name": "Rasopathies Atlas",
        "terms": [
            {
                "term": "RASopathy",
                "definition": (
                    "Group of developmental syndromes caused by germline mutations in genes encoding components "
                    "of the RAS/MAPK signalling cascade (RAS → RAF → MEK → ERK). "
                    "Shared features: characteristic facies (hypertelorism, ptosis, low ears), "
                    "short stature, cardiac defects (PS/HCM/ASD), variable cognitive impairment. "
                    "Individually rare, collectively ~1 in 1000 live births. "
                    "Unified by pathway overactivation (GOF mutations); unified treatment rationale: MEK inhibitors."
                ),
            },
            {
                "term": "PTPN11 / SHP2",
                "definition": (
                    "SHP2: cytoplasmic protein tyrosine phosphatase; amplifies RAS activation downstream of RTKs. "
                    "Normally auto-inhibited (N-SH2:PTP closed state). "
                    "GOF Noonan alleles (Asp61Asn, Asn308Asp): disrupt auto-inhibition → open state → "
                    "sustained RAS-ERK + PI3K-AKT. "
                    "LEOPARD alleles (Tyr279Cys): catalytic dead, dominant negative → LEOPARD (lentigines, HCM, deafness). "
                    "50% of Noonan syndrome (most common RASopathy gene)."
                ),
            },
            {
                "term": "RAF1 / CRAF",
                "definition": (
                    "RAF1 (CRAF): RAF family serine/threonine kinase; activated by RAS-GTP → phosphorylates MEK1/2. "
                    "NS5 alleles (Ser257Leu, Leu613Val): constitutive kinase activity. "
                    "DISTINCTIVE: HCM in 75% (highest of all NS alleles). "
                    "Management priority: HCM surveillance + mavacamten for obstructive HCM."
                ),
            },
            {
                "term": "SOS1",
                "definition": (
                    "SOS1: RAS guanine nucleotide exchange factor (GEF); recruited via GRB2 to activated RTKs; "
                    "catalyses GDP→GTP exchange on RAS (activation). "
                    "NS4 alleles (Arg552Gly): disrupts auto-inhibitory DH-PH contacts → constitutive RAS loading. "
                    "MILDEST NS: near-normal intellect (IQ 90–100), tallest stature (−1 SD), least HCM (<5%). "
                    "SOSi BI-3406 blocks SOS1-KRAS interface — clinical trials for cancer."
                ),
            },
            {
                "term": "KRAS4B",
                "definition": (
                    "KRAS4B: predominant KRAS isoform; constitutively farnesylated; plasma membrane–anchored. "
                    "Most commonly somatically mutated oncogene (NSCLC, CRC, PDAC) — Gly12Val/Asp. "
                    "GERMLINE NS3 alleles (Thr58Ile, Pro34Arg) are HYPOMORPHIC — NOT Gly12 cancer alleles. "
                    "Sotorasib/adagrasib (KRASG12Ci) NOT applicable to NS3 patients. "
                    "Intermediate phenotype — more cognitive than PTPN11, some CFC overlap."
                ),
            },
            {
                "term": "BRAF V600E vs CFC alleles",
                "definition": (
                    "V600E (somatic melanoma): activation loop Val600→Glu → constitutive BRAF monomer kinase → "
                    "sensitive to vemurafenib/dabrafenib (BRAFi). "
                    "CFC germline alleles (Gln257Arg, Glu501Lys): DFG motif region → NOT V600E. "
                    "BRAFi causes PARADOXICAL ERK activation in RAS-active context (CRAF dimerisation). "
                    "MEKi (trametinib) is rational for CFC — investigational LOVS1K trial."
                ),
            },
            {
                "term": "MEK1 (MAP2K1) inhibition — trametinib",
                "definition": (
                    "MEK1 (MAP2K1) and MEK2 (MAP2K2): dual-specificity kinases; phosphorylate ERK1/2 on Thr/Tyr; "
                    "central bottleneck of MAPK cascade. "
                    "Trametinib (MEKINIST, GSK1120212): allosteric non-competitive MEKi; approved for BRAF V600 + cancer. "
                    "For RASopathies: trametinib investigational (LOVS1K trial); MAP2K1-CFC has highest direct rationale. "
                    "Cobimetinib, binimetinib, selumetinib (approved NF1-PN) are same MEKi class."
                ),
            },
            {
                "term": "Costello Nasal Papillomata",
                "definition": (
                    "Nasal and perianal fibroepithelial papillomata: HPV-NEGATIVE benign polyps appearing age 2–7 years. "
                    "PATHOGNOMONIC for HRAS-Costello syndrome. "
                    "Management: ENT debulking for nasal polyps (surgical; recurrence expected); "
                    "dermatology for perianal; surveillance for malignant transformation (rare but reported in RMS context). "
                    "Do NOT confuse with HPV warts — no antiviral therapy indicated."
                ),
            },
            {
                "term": "CBL JMML — spontaneous remission",
                "definition": (
                    "CBL-JMML: two-hit — germline CBL RING mutation + somatic UPD11q23 (loss of wild-type allele). "
                    "Unique biology: spontaneous remission occurs in 40–60% without SCT (unlike KRAS/PTPN11-JMML). "
                    "Management: serial FBC every 4 weeks; splenomegaly USS; SCT only for progressive disease. "
                    "Confirm two-hit by bone marrow FISH (UPD11q23) before treatment escalation."
                ),
            },
            {
                "term": "Pulmonary Valve Stenosis (PS) in NS",
                "definition": (
                    "Most common CHD in PTPN11-NS (50–80%); dysplastic pulmonary valve with thickened cusps. "
                    "Gradient >50 mmHg at rest: indication for balloon pulmonary valvuloplasty (BPV — first line). "
                    "Surgical repair: for severely dysplastic valve unamenable to BPV. "
                    "Annual echo until stable post-intervention. "
                    "LESS prominent in RAF1/CBL (HCM dominant) and minimal in SOS1-NS."
                ),
            },
            {
                "term": "Noonan coagulopathy",
                "definition": (
                    "Haematological coagulopathy in 35% of NS: Factor XI deficiency (most common, 20–30%), "
                    "Factor XII, von Willebrand disease Type I, platelet dysfunction. "
                    "PRE-OPERATIVE SCREEN MANDATORY for ALL NS alleles: PT, APTT, FXI, FXII, vWF Ag + activity. "
                    "Management: FFP or factor concentrate if FXI <30%; tranexamic acid for minor procedures."
                ),
            },
            {
                "term": "LEOPARD Syndrome",
                "definition": (
                    "LEOPARD = Lentigines + ECG abnormalities + Ocular hypertelorism + Pulmonary stenosis + "
                    "Abnormal genitalia + Retardation of growth + Deafness (sensorineural). "
                    "Caused by PTPN11 catalytic-dead alleles (Tyr279Cys, Thr468Met) OR RAF1 (Thr543Met, Ser257Leu). "
                    "PARADOX: LEOPARD alleles LOSE phosphatase activity → dominant negative → REDUCES RAS signalling "
                    "yet causes NS-like phenotype + HCM + lentigines. "
                    "Treatment: same HCM management; dermatology for lentigines; audiology for SNHL."
                ),
            },
            {
                "term": "CFC triad",
                "definition": (
                    "Cardio-Facio-Cutaneous (CFC) syndrome triad: "
                    "(1) Frontal bossing + coarse facial features (more severe than NS); "
                    "(2) Ectodermal anomalies — ichthyosis/hyperkeratosis, absent/sparse eyebrows, curly/woolly hair; "
                    "(3) Significant cognitive impairment (IQ 40–70) + seizures (50%). "
                    "Caused by BRAF (50–75% CFC), MAP2K1 + MAP2K2 (15–25% CFC), KRAS (<5%). "
                    "NO papillomata (distinguishes from Costello); NO JMML (distinguishes from CBL)."
                ),
            },
            {
                "term": "Pterygium colli",
                "definition": (
                    "Neck webbing (pterygium colli): bilateral folds of skin from mastoid to acromion. "
                    "Seen in NS (PTPN11, RAF1, SOS1 alleles) and Turner syndrome (45,X). "
                    "DIFFERENTIAL: karyotype first to exclude Turner in any phenotypic female with webbed neck; "
                    "if chromosomes normal → RASopathy panel. "
                    "Surgical correction (Z-plasty) for functional or cosmetic reasons; "
                    "does NOT affect cardiac or cognitive management."
                ),
            },
            {
                "term": "Selumetinib (KOSELUGO) — NF1 MEKi",
                "definition": (
                    "Selumetinib: oral MEKi approved FDA 2020 for NF1 inoperable plexiform neurofibroma (PN) in children ≥2 years. "
                    "MOA: inhibits MEK1/2 → reduces ERK → PN tumour cell proliferation ↓ → PN shrinkage (30%+ in SPRINT). "
                    "NOT yet approved for Noonan/CFC — investigational in PTPN11-NS HCM and BRAF/MAP2K1-CFC (trials). "
                    "Compare: trametinib (approved melanoma) + cobimetinib (approved melanoma) — same MEKi class."
                ),
            },
        ]
    }
