#!/usr/bin/env python3
"""Hereditary-Ovarian-Cancer-Atlas — Complete 8-Gene Hereditary Ovarian Cancer Atlas.

BRCA1   (BRCA1 associated RING domain protein 1; 1863 aa; 17q21.31; AD;
          HBOC1 — 44% lifetime ovarian cancer risk; PARP inhibitors olaparib;
          RRSO 35-40 yr; Ashkenazi 185delAG 5382insC founders;
          seed SEED_BASE+0).
BRCA2   (BRCA2 / FANCD1; 3418 aa; 13q12.3; AD;
          HBOC2 — 17% lifetime ovarian cancer risk; olaparib / niraparib;
          RRSO 40-45 yr; Ashkenazi 6174delT founder;
          seed SEED_BASE+1).
BRIP1   (BRCA1-interacting protein C-terminal helicase 1 / FANCJ; 1249 aa; 17q23.2; AD;
          HOCA3 — 11-13x relative risk ovarian cancer; NO breast risk;
          RRSO 45-50 yr; Fanconi anemia FANCJ biallelic;
          seed SEED_BASE+2).
RAD51C  (RAD51 paralog C / FANCO; 376 aa; 17q22; AD;
          HOCA4 — 5-6x relative risk ovarian cancer; NO significant breast risk;
          RRSO 45-50 yr; Fanconi anemia FANCO biallelic;
          seed SEED_BASE+3).
RAD51D  (RAD51 paralog D; 328 aa; 17q12; AD;
          HOCA5 — 5-6x relative risk ovarian cancer; very low breast risk;
          RRSO 45-50 yr; olaparib emerging;
          seed SEED_BASE+4).
PALB2   (partner and localizer of BRCA2 / FANCN; 1186 aa; 16p12.2; AD;
          HOCA6 — 3-5% lifetime ovarian cancer; 53% lifetime breast cancer;
          Fanconi anemia FANCN biallelic; RRSO timing controversial (breast priority);
          seed SEED_BASE+5).
MLH1    (MutL homolog 1; 756 aa; 3p22.2; AD;
          Lynch syndrome 1 — 10-12% ovarian cancer; ENDOMETRIOID not HGSOC;
          dMMR/MSI-H; pembrolizumab FDA 2017; endometrial cancer 40-60%;
          seed SEED_BASE+6).
MSH2    (MutS homolog 2; 934 aa; 2p21; AD;
          Lynch syndrome 2 — 10-12% ovarian cancer; ENDOMETRIOID clear cell;
          dMMR/MSI-H; pembrolizumab; EPCAM 3-prime deletion silences MSH2;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2062-2069).
"""

import random

SEED_BASE = 2062

OC_GENES = [
    # -- BRCA1 — HBOC1 (AD) --------------------------------------------------------
    {
        "gene": "BRCA1",
        "alt_name": "BRCA1 (BRCA1-1863aa-17q21.31 / AD — HBOC1-Hereditary-Breast-Ovarian-Cancer — 44pct-Lifetime-Ovarian-Risk — RRSO-35-40yr-MANDATORY — Olaparib-PARP-Inhibitor-1st-Line-Maintenance — Ashkenazi-185delAG-5382insC-Founders)",
        "protein": (
            "BRCA1 -- 17q21.31 AD -- BRCA1-1863aa -- "
            "BRCA1-Associated-RING-Domain-Protein-1-BRCT-Domain-Tumor-Suppressor -- "
            "HBOC1-Hereditary-Breast-Ovarian-Cancer -- "
            "44pct-Lifetime-Ovarian-Cancer-Risk-10-40x-Relative-Risk -- "
            "High-Grade-Serous-Ovarian-Cancer-HGSOC-Fallopian-Tube-Origin -- "
            "Homologous-Recombination-Deficiency-HRD-DNA-Repair -- "
            "Olaparib-Niraparib-Rucaparib-PARP-Inhibitors-Maintenance -- "
            "RRSO-Risk-Reducing-Salpingo-Oophorectomy-35-40yr-Post-Childbearing"
        ),
        "locus": "17q21.31",
        "protein_size": "1863 aa",
        "inheritance": (
            "AD (autosomal dominant) — BRCA1 haploinsufficiency; "
            "Prevalence: 1 in 400 general population; 1 in 40 Ashkenazi Jewish; "
            "Ashkenazi Jewish founders: c.68_69delAG (185delAG), c.5266dupC (5382insC); "
            "Slavic founder: c.5266dupC (5382insC); "
            "Penetrance: ovarian cancer 44% (range 36-46%); breast cancer 72% lifetime; "
            "Phenotypic onset ovarian cancer: 40-60 yr (earlier than sporadic); "
            "Male carriers: increased prostate cancer risk (3-4x); male breast cancer <1%; "
            "Siblings: 50% risk; cascade testing mandatory"
        ),
        "age_of_onset": (
            "Ovarian cancer: typically 40-60 yr (peak 55-65 yr); "
            "Earlier than sporadic ovarian cancer (sporadic peak 63 yr); "
            "Fallopian tube primary carcinoma: BRCA1/2 most common cause; "
            "Peritoneal primary carcinoma: 3-5% even post-RRSO; "
            "Breast cancer: 30-70 yr (lifetime risk 72%); "
            "Triple negative breast cancer: BRCA1 strongly associated; "
            "Pancreatic cancer: slightly elevated risk; "
            "Bilateral risk: bilateral breast cancer risk — contralateral 20-30% lifetime"
        ),
        "key_biomarker": (
            "BRCA1 germline sequencing: pathogenic variant confirms; "
            "HRD (Homologous Recombination Deficiency) assay: MyChoice CDx (Myriad) — LOH+TAI+LST; "
            "CA-125: ovarian cancer surveillance (sensitivity limited for early detection); "
            "Transvaginal ultrasound: pre-RRSO surveillance (limited efficacy); "
            "BRCA1 somatic testing in tumor: sporadic BRCA1 mutations — PARP inhibitor eligibility; "
            "Platinum sensitivity: HGSOC BRCA1/2 — high platinum response rates; "
            "MMR status: distinguish Lynch (dMMR) — BRCA1 tumors are MMR-proficient; "
            "Oestrogen receptor: BRCA1 breast cancers predominantly ER-negative (triple negative)"
        ),
        "pathognomonic": (
            "HIGH-GRADE SEROUS OVARIAN CANCER under age 60 = test BRCA1/2 regardless of family history; "
            "FALLOPIAN TUBE PRIMARY CARCINOMA = BRCA1/2 pathogenic variant in 20-30%; "
            "TRIPLE NEGATIVE BREAST CANCER under 60 = BRCA1 testing mandatory (NICE/NCCN); "
            "BILATERAL BREAST CANCER especially early onset = BRCA1 strongly suspect; "
            "MALE BREAST CANCER (rare) = BRCA2 more likely than BRCA1 but test both; "
            "FAMILY HISTORY (1st-degree): ovarian + breast = HBOC syndrome — cascade test all relatives"
        ),
        "treatment": (
            "Acute ovarian cancer — systemic: platinum-taxane (carboplatin+paclitaxel) standard; "
            "PARP inhibitor 1st-line maintenance: olaparib (SOLO-1: 60% 5yr PFS improvement); "
            "niraparib (PRIMA: significant PFS benefit in HRD); "
            "bevacizumab + olaparib: PAOLA-1 (HRD enriched group greatest benefit); "
            "Platinum-sensitive relapse: olaparib (SOLO-2 PFS HR 0.30); niraparib; rucaparib; "
            "Risk reduction: RRSO at 35-40 yr (post-childbearing) — reduces ovarian risk 80-95%; "
            "REDUCES BREAST CANCER RISK 50% if pre-menopausal RRSO; "
            "OCP: 50% ovarian cancer risk reduction with ≥5yr use (acceptable short-term); "
            "Breast: surveillance (annual MRI) or risk-reducing mastectomy; "
            "MENOPAUSE: HRT until age 50 post-RRSO (does not negate breast risk reduction)"
        ),
        "critical_flags": [
            "RRSO-35-40yr-POST-CHILDBEARING-REDUCES-OVARIAN-RISK-80-95pct",
            "PARP-INHIBITOR-1st-LINE-MAINTENANCE-OLAPARIB-SOLO1",
            "HIGH-GRADE-SEROUS-OC-AGE<60-TEST-BRCA1-REGARDLESS-FAMILY-HISTORY",
            "OCP-50pct-OVARIAN-RISK-REDUCTION-ACCEPTABLE",
            "RRSO-REDUCES-BREAST-CANCER-RISK-50pct-PRE-MENOPAUSAL",
            "HRT-POST-RRSO-UNTIL-AGE-50-DOES-NOT-NEGATE-BREAST-BENEFIT",
            "CASCADE-TESTING-MANDATORY-50pct-SIBLING-RISK",
        ],
    },
    # -- BRCA2 — HBOC2 (AD) --------------------------------------------------------
    {
        "gene": "BRCA2",
        "alt_name": "BRCA2 (BRCA2-3418aa-13q12.3 / AD — HBOC2-FANCD1 — 17pct-Lifetime-Ovarian-Risk — RRSO-40-45yr — Olaparib-PROfound — Ashkenazi-6174delT-Founder — Male-Breast-Prostate-Risk)",
        "protein": (
            "BRCA2 -- 13q12.3 AD -- BRCA2-3418aa -- "
            "BRCA2-FANCD1-DNA-Repair-HR-RAD51-Mediator -- "
            "HBOC2-Hereditary-Breast-Ovarian-Cancer-Type-2 -- "
            "17pct-Lifetime-Ovarian-Cancer-Risk-5-10x-Relative-Risk -- "
            "RRSO-40-45yr-Post-Childbearing -- "
            "Olaparib-PROfound-FDA2020 -- "
            "Ashkenazi-Founder-c.5946delT-6174delT -- "
            "Male-Breast-Cancer-7x-Prostate-Cancer-4-7x"
        ),
        "locus": "13q12.3",
        "protein_size": "3418 aa",
        "inheritance": (
            "AD (autosomal dominant) — BRCA2 haploinsufficiency; "
            "Prevalence: 1 in 400 general population; 1 in 40 Ashkenazi Jewish; "
            "Ashkenazi founder: c.5946delT (6174delT); "
            "Penetrance: ovarian cancer 17% (range 11-17%); breast cancer 69% lifetime; "
            "Male carriers: prostate cancer 4-7x (aggressive); breast cancer 7x; "
            "Biallelic BRCA2: Fanconi anemia complementation group D1 (FANCD1) — severe; "
            "Pancreatic cancer: elevated risk (~5%); "
            "BRCA2 ovarian risk lower than BRCA1 (17% vs 44%) — RRSO timing adjusted accordingly"
        ),
        "age_of_onset": (
            "Ovarian cancer: typically 50-60 yr (slightly later than BRCA1); "
            "Breast cancer: 30-70 yr (lifetime 69%); "
            "Male breast cancer: 50-70 yr (7x risk); "
            "Prostate cancer: 4-7x — early-onset aggressive prostate cancer; "
            "Pancreatic cancer: 5x relative risk — BRCA2 most common hereditary pancreatic cancer gene; "
            "Fallopian tube primary: BRCA2 also causes fallopian tube carcinoma; "
            "BRCA2 ovarian cancer histology: high-grade serous (HGSOC) predominant, as with BRCA1"
        ),
        "key_biomarker": (
            "BRCA2 germline sequencing: pathogenic variant confirms; "
            "HRD assay: MyChoice CDx — HRD positive (LOH+TAI+LST); "
            "CA-125 + transvaginal USS: pre-RRSO surveillance (limited, not guideline-endorsed); "
            "PSA: annual from 40 yr male carriers (aggressive early prostate cancer); "
            "Somatic BRCA2: tumor testing for PARP inhibitor eligibility (non-germline carriers); "
            "Platinum sensitivity: BRCA2 HGSOC — very high platinum response; "
            "Pancreatic MRI: surveillance from 50 yr if FH (CAPS consortium guidelines); "
            "Male breast mammography: annual mammogram from 50 yr male carriers"
        ),
        "pathognomonic": (
            "HIGH-GRADE SEROUS OVARIAN CANCER = BRCA2 germline in 10-15%; "
            "MALE BREAST CANCER + ovarian cancer FH = BRCA2 most likely; "
            "AGGRESSIVE EARLY PROSTATE CANCER under 60 = BRCA2 testing mandatory; "
            "PANCREATIC CANCER + ovarian cancer FH = BRCA2 most common cause; "
            "FALLOPIAN TUBE CARCINOMA = BRCA2 in 10-15% cases; "
            "FANCONI ANEMIA (biallelic BRCA2/FANCD1): severe marrow failure + solid tumors in childhood — BRCA2 VUS in Fanconi context always escalate to specialist"
        ),
        "treatment": (
            "Ovarian cancer — systemic: platinum-taxane first-line; "
            "PARP inhibitor maintenance: olaparib (SOLO-1, SOLO-2); niraparib (PRIMA/ENGAGE-1); "
            "Prostate cancer: olaparib (PROfound: OS benefit in BRCA2 HR 0.42); rucaparib (TRITON2); "
            "Pancreatic cancer: olaparib maintenance (POLO trial: PFS benefit); "
            "RRSO: 40-45 yr post-childbearing (later than BRCA1 — lower ovarian risk); "
            "Male carriers: annual PSA from 40 yr; breast exam + annual mammogram from 50 yr; "
            "Breast: annual MRI from 25 yr (or 10 yr before earliest FH case); "
            "Risk-reducing mastectomy: option for high-risk carriers; "
            "OCP: 50% ovarian risk reduction; acceptable protective option"
        ),
        "critical_flags": [
            "RRSO-40-45yr-LATER-THAN-BRCA1-LOWER-OVARIAN-RISK",
            "PARP-INHIBITOR-OLAPARIB-PROFOUND-PROSTATE-CANCER-FDA2020",
            "MALE-CARRIERS-PSA-FROM-40yr-AGGRESSIVE-PROSTATE",
            "PANCREATIC-CANCER-POLO-OLAPARIB-MAINTENANCE",
            "BIALLELIC-BRCA2-FANCD1-FANCONI-ANEMIA-SEVERE",
            "MALE-BREAST-CANCER-7x-ANNUAL-MAMMOGRAM-FROM-50yr",
            "6174delT-ASHKENAZI-FOUNDER-TEST-FIRST",
        ],
    },
    # -- BRIP1 — HOCA3 / FANCJ (AD) -----------------------------------------------
    {
        "gene": "BRIP1",
        "alt_name": "BRIP1 (BRIP1-1249aa-17q23.2 / AD — FANCJ-Fanconi-Anemia-J — 11-13x-Ovarian-RR — NO-Breast-Risk — RRSO-45-50yr — High-Grade-Serous — Biallelic-FANCJ)",
        "protein": (
            "BRIP1 -- 17q23.2 AD -- BRIP1-1249aa -- "
            "BRCA1-Interacting-Protein-C-Terminal-Helicase-1-FANCJ-BACH1 -- "
            "Fanconi-Anemia-Complementation-Group-J-FANCJ -- "
            "HOCA3-Hereditary-Ovarian-Cancer-Association-3 -- "
            "11-13x-Relative-Risk-Ovarian-Cancer -- "
            "NO-Significant-Breast-Cancer-Risk-KEY-DDx-BRCA1-BRCA2 -- "
            "RRSO-45-50yr-Post-Childbearing -- "
            "High-Grade-Serous-HGSOC-Predominant"
        ),
        "locus": "17q23.2",
        "protein_size": "1249 aa",
        "inheritance": (
            "AD (autosomal dominant) — BRIP1 haploinsufficiency; "
            "Prevalence: ~1 in 500-1000 general population; "
            "Ovarian cancer relative risk: 11-13x (Rafnar 2011 Nat Genet); ~10-15% lifetime; "
            "NO significant breast cancer risk — critical distinction from BRCA1/2; "
            "Incomplete penetrance; variable expressivity; "
            "Biallelic BRIP1: Fanconi anemia complementation group J (FANCJ) — severe; "
            "FANCJ biallelic: marrow failure, AML predisposition, solid tumors; "
            "Monoallelic: ovarian cancer only — DO NOT counsel as high breast cancer risk"
        ),
        "age_of_onset": (
            "Ovarian cancer: typically 55-65 yr (later than BRCA1); "
            "High-grade serous predominant — same histology as BRCA1/2 carriers; "
            "No increased breast cancer onset — CRITICAL for counselling (no breast MRI escalation needed unless other risk factors); "
            "Fallopian tube primaries reported; "
            "Peritoneal primary: possible but less data than BRCA1/2; "
            "Biallelic FANCJ: childhood — bone marrow failure, AML, solid tumors (brain, Wilms); "
            "No increased colorectal or endometrial risk (unlike Lynch)"
        ),
        "key_biomarker": (
            "BRIP1 germline sequencing: pathogenic variant confirms; "
            "CA-125 + transvaginal USS: surveillance pre-RRSO (limited efficacy, not guideline-endorsed); "
            "HRD assay: BRIP1 tumors may show HRD — PARP inhibitor activity emerging (limited data); "
            "MMR status: MMR-proficient (distinguish Lynch); "
            "BRIP1 somatic: tumor testing may identify BRIP1 — but germline vs somatic must be distinguished; "
            "Fanconi anemia workup (biallelic): chromosomal breakage test (diepoxybutane/DEB test); "
            "No elevated PSA screening burden (unlike BRCA2); "
            "Pelvic MRI: no established role for surveillance (RRSO preferred)"
        ),
        "pathognomonic": (
            "BRIP1 PATHOGENIC VARIANT = ovarian cancer risk 11-13x WITHOUT breast cancer risk; "
            "INCORRECTLY COUNSELLING BRIP1 AS HIGH BREAST CANCER RISK = wrong — only ovarian; "
            "HIGH-GRADE SEROUS OC in woman with NEGATIVE BRCA1/2 = test BRIP1/RAD51C/RAD51D panel; "
            "FANCONI ANEMIA + known BRIP1 family history = test for biallelic BRIP1/FANCJ; "
            "FAMILY HISTORY of ovarian only (no breast): BRIP1/RAD51C/RAD51D more likely than BRCA1/2"
        ),
        "treatment": (
            "Ovarian cancer — systemic: platinum-taxane first-line; "
            "PARP inhibitor: limited data vs BRCA1/2; HRD-positive BRIP1 tumors may respond; "
            "RRSO: 45-50 yr post-childbearing (later than BRCA1/2 — moderate risk, later onset); "
            "Breast surveillance: standard population-based (NO escalation needed for monoallelic BRIP1); "
            "Cascading: first-degree relatives BRIP1 testing; "
            "OCP: 50% ovarian risk reduction — appropriate protective option; "
            "Annual pelvic exam: limited clinical utility pre-RRSO; "
            "Fanconi anemia (biallelic): allogenic HSCT for marrow failure; avoid radiation (DNA repair defect)"
        ),
        "critical_flags": [
            "BRIP1-MONOALLELIC-NO-BREAST-RISK-CRITICAL-DDx-BRCA1-BRCA2",
            "RRSO-45-50yr-LATER-THAN-BRCA1-2",
            "BRIP1-11-13x-OVARIAN-RR-HIGH-GRADE-SEROUS",
            "BIALLELIC-FANCJ-FANCONI-ANEMIA-DEB-TEST",
            "PARP-INHIBITOR-LIMITED-DATA-NOT-ESTABLISHED",
            "OCP-50pct-OVARIAN-RISK-REDUCTION-APPROPRIATE",
            "OVARIAN-ONLY-FAMILY-HISTORY-TEST-BRIP1-PANEL",
        ],
    },
    # -- RAD51C — HOCA4 / FANCO (AD) -----------------------------------------------
    {
        "gene": "RAD51C",
        "alt_name": "RAD51C (RAD51C-376aa-17q22 / AD — FANCO-Fanconi-Anemia-O — 5-6x-Ovarian-RR — NO-Significant-Breast-Risk — RRSO-45-50yr — Biallelic-FANCO)",
        "protein": (
            "RAD51C -- 17q22 AD -- RAD51C-376aa -- "
            "RAD51-Paralog-C-HR-Repair-RAD51B-RAD51C-RAD51D-XRCC2-XRCC3-Complex -- "
            "Fanconi-Anemia-Complementation-Group-O-FANCO -- "
            "HOCA4-5-6x-Ovarian-Cancer-Relative-Risk -- "
            "6-7pct-Lifetime-Ovarian-Cancer-Risk -- "
            "NO-Significant-Breast-Cancer-Risk -- "
            "RRSO-45-50yr -- "
            "High-Grade-Serous-HGSOC-Predominant"
        ),
        "locus": "17q22",
        "protein_size": "376 aa",
        "inheritance": (
            "AD (autosomal dominant) — RAD51C haploinsufficiency; "
            "Prevalence: ~1 in 800-1000 population; "
            "Ovarian cancer relative risk: 5-6x (Loveday 2012 Nat Genet); 6-7% lifetime risk; "
            "No significant breast cancer risk in monoallelic carriers (KEY distinction from BRCA1/2); "
            "Incomplete penetrance; "
            "Biallelic RAD51C: Fanconi anemia complementation group O (FANCO) — severe; "
            "FANCO: severe aplastic anemia; "
            "Population-specific founder variants: c.145+1G>A (Finnish?); regional variants; "
            "Part of HR repair complex with RAD51B, RAD51D, XRCC2, XRCC3"
        ),
        "age_of_onset": (
            "Ovarian cancer: 55-65 yr (moderate risk, late onset); "
            "High-grade serous predominant; "
            "No breast cancer age of onset concern for monoallelic carriers; "
            "Fallopian tube primary: possible; "
            "Biallelic FANCO: childhood marrow failure, solid tumors; "
            "No endometrial or colorectal cancer increase; "
            "Peritoneal primary: very limited data"
        ),
        "key_biomarker": (
            "RAD51C germline sequencing: pathogenic variant; "
            "HRD assay: RAD51C tumors may show HRD — emerging PARP inhibitor data; "
            "CA-125: pre-RRSO surveillance (limited); "
            "DEB test (biallelic Fanconi workup); "
            "MMR status: proficient; "
            "RAD51C protein: immunohistochemical expression; "
            "BRCA1/2 negative ovarian cancer panel: RAD51C routinely included"
        ),
        "pathognomonic": (
            "RAD51C PATHOGENIC VARIANT = ovarian risk 5-6x WITHOUT breast cancer risk escalation; "
            "HIGH-GRADE SEROUS OC + BRCA1/2 NEGATIVE = RAD51C/RAD51D/BRIP1 panel next; "
            "OVARIAN CANCER FAMILY HISTORY ONLY (no breast) = RAD51C more likely than BRCA1/2; "
            "BIALLELIC RAD51C = Fanconi anemia group O — severe aplastic anemia childhood; "
            "MONOALLELIC = only ovarian cancer risk; breast surveillance NOT escalated"
        ),
        "treatment": (
            "Ovarian cancer — systemic: platinum-taxane; "
            "PARP inhibitor: emerging data (olaparib extended cohort RAD51C/D patients); "
            "RRSO: 45-50 yr post-childbearing; "
            "Breast surveillance: standard population-based (no MRI escalation for monoallelic); "
            "OCP: ovarian cancer risk reduction; "
            "Fanconi anemia (biallelic): HSCT for severe aplastic anemia; "
            "Cascade testing: first-degree relatives; "
            "Genetic counselling: important to distinguish monoallelic (ovarian only) from biallelic (FA)"
        ),
        "critical_flags": [
            "RAD51C-MONOALLELIC-NO-BREAST-RISK",
            "5-6x-OVARIAN-RR-RRSO-45-50yr",
            "BIALLELIC-FANCO-APLASTIC-ANEMIA-CHILDHOOD",
            "BRCA1-2-NEGATIVE-HGSOC-TEST-RAD51C-PANEL",
            "PARP-INHIBITOR-EMERGING-NOT-ESTABLISHED",
            "OCP-OVARIAN-RISK-REDUCTION-APPROPRIATE",
            "OVARIAN-ONLY-FAMILY-HISTORY-NOT-BREAST-SUGGESTS-RAD51C",
        ],
    },
    # -- RAD51D — HOCA5 (AD) -------------------------------------------------------
    {
        "gene": "RAD51D",
        "alt_name": "RAD51D (RAD51D-328aa-17q12 / AD — 5-6x-Ovarian-RR — Very-Low-Breast-Risk — RRSO-45-50yr — Olaparib-Emerging-RAD51C-D — Part-HR-Complex)",
        "protein": (
            "RAD51D -- 17q12 AD -- RAD51D-328aa -- "
            "RAD51-Paralog-D-HR-Repair-BCDX2-Complex-RAD51B-RAD51C-RAD51D-XRCC2 -- "
            "HOCA5-5-6x-Relative-Risk-Ovarian-Cancer -- "
            "5-7pct-Lifetime-Ovarian-Cancer-Risk -- "
            "Very-Low-Breast-Cancer-Risk-KEY-DDx-BRCA1-BRCA2 -- "
            "RRSO-45-50yr -- "
            "Olaparib-Extended-Cohort-Emerging-Data"
        ),
        "locus": "17q12",
        "protein_size": "328 aa",
        "inheritance": (
            "AD (autosomal dominant) — RAD51D haploinsufficiency; "
            "Prevalence: ~1 in 800-1200 population; "
            "Ovarian cancer relative risk: 5-6x (Loveday 2012 Nat Genet); 5-7% lifetime; "
            "Very low breast cancer risk — even lower than RAD51C; "
            "Incomplete penetrance; "
            "No Fanconi anemia phenotype established for biallelic RAD51D (unlike RAD51C/BRIP1); "
            "Part of BCDX2 complex (RAD51B-RAD51C-RAD51D-XRCC2) in HR repair; "
            "Regional founder variants exist in various populations"
        ),
        "age_of_onset": (
            "Ovarian cancer: 55-65 yr; high-grade serous predominant; "
            "Very low breast cancer concern; "
            "No endometrial or colorectal cancer increase; "
            "Fallopian tube primary: possible; "
            "Peritoneal primary: very limited data; "
            "No male cancer risk established (unlike BRCA2)"
        ),
        "key_biomarker": (
            "RAD51D germline sequencing: pathogenic variant; "
            "HRD assay: emerging PARP inhibitor data (olaparib cohort includes RAD51D); "
            "CA-125: pre-RRSO; "
            "MMR status: proficient; "
            "BRCA1/2/BRIP1/RAD51C negative ovarian cancer panel: RAD51D included; "
            "Somatic RAD51D: tumor testing"
        ),
        "pathognomonic": (
            "RAD51D PATHOGENIC VARIANT = ovarian risk 5-6x WITHOUT meaningful breast risk; "
            "HIGH-GRADE SEROUS OC BRCA1/2 NEGATIVE = RAD51D/RAD51C/BRIP1 panel; "
            "OVARIAN CANCER ONLY FAMILY (no breast history) = RAD51D, RAD51C, BRIP1 most likely; "
            "RAD51D + RAD51C: similar risk, similar management, both target HR repair"
        ),
        "treatment": (
            "Ovarian cancer — systemic: platinum-taxane; "
            "PARP inhibitor: olaparib extended cohort RAD51C/D — emerging data (not yet standard); "
            "RRSO: 45-50 yr post-childbearing; "
            "Breast surveillance: standard population-based (no MRI escalation); "
            "OCP: ovarian risk reduction; "
            "Cascade testing: first-degree relatives; "
            "Genetic counselling: very low breast risk — important for reducing unnecessary interventions"
        ),
        "critical_flags": [
            "RAD51D-VERY-LOW-BREAST-RISK-NO-MRI-ESCALATION",
            "5-6x-OVARIAN-RR-RRSO-45-50yr",
            "PARP-INHIBITOR-EMERGING-OLAPARIB-COHORT",
            "BRCA1-2-NEGATIVE-HGSOC-TEST-RAD51D-PANEL",
            "OCP-PROTECTIVE-APPROPRIATE",
            "OVARIAN-ONLY-FAMILY-HISTORY-NOT-BREAST",
        ],
    },
    # -- PALB2 — HOCA6 / FANCN (AD) -----------------------------------------------
    {
        "gene": "PALB2",
        "alt_name": "PALB2 (PALB2-1186aa-16p12.2 / AD — FANCN-Fanconi-Anemia-N — 53pct-Lifetime-Breast-Risk — 3-5pct-Ovarian-Risk — RRSO-Timing-Controversial-Breast-Priority — Biallelic-FANCN)",
        "protein": (
            "PALB2 -- 16p12.2 AD -- PALB2-1186aa -- "
            "Partner-and-Localizer-of-BRCA2-WD40-Domain-BRCA1-PALB2-BRCA2-Complex -- "
            "Fanconi-Anemia-Complementation-Group-N-FANCN -- "
            "HOCA6-3-5pct-Lifetime-Ovarian-Cancer-Risk -- "
            "53pct-Lifetime-Breast-Cancer-Risk-Second-Highest-After-BRCA1-BRCA2 -- "
            "RRSO-Timing-Controversial-Breast-Cancer-Priority -- "
            "Biallelic-FANCN-Severe-Fanconi-Anemia"
        ),
        "locus": "16p12.2",
        "protein_size": "1186 aa",
        "inheritance": (
            "AD (autosomal dominant) — PALB2 haploinsufficiency; "
            "Prevalence: ~1 in 200-400 general population; "
            "Breast cancer: 53% lifetime (Antoniou 2014 NEJM — key paper); highest after BRCA1 (72%) / BRCA2 (69%); "
            "Ovarian cancer: 3-5% lifetime risk (moderate — lower than BRCA1/2); "
            "Biallelic PALB2: Fanconi anemia complementation group N (FANCN) — severe; "
            "Pancreatic cancer: 3-4x risk; "
            "Founder variants: c.3113G>A (W1038*) in Polish; c.1592delT (Finnish PALB2); "
            "PALB2 'bridges' BRCA1 and BRCA2 — essential for HR complex assembly"
        ),
        "age_of_onset": (
            "Breast cancer: 30-70 yr (peak 40-50 yr); often ER-positive (unlike BRCA1); "
            "Ovarian cancer: 55-65 yr (lower risk, later onset compared to BRCA1/2); "
            "Pancreatic cancer: 55-70 yr; "
            "Biallelic FANCN: childhood — severe marrow failure, leukemia, solid tumors; "
            "Male carriers: elevated breast cancer risk (8-9x); elevated pancreatic risk; "
            "Triple negative breast cancer: less common than BRCA1 (PALB2 more luminal B/ER+)"
        ),
        "key_biomarker": (
            "PALB2 germline sequencing: pathogenic variant; "
            "Breast MRI: annual from 25-30 yr (high breast risk — 53% lifetime); "
            "CA-125 + ultrasound: pre-RRSO surveillance; "
            "HRD assay: PALB2 tumors show HRD — PARP inhibitor activity; "
            "Pancreatic MRI: surveillance from 50 yr or 10 yr before earliest FH; "
            "DEB test (biallelic Fanconi): chromosomal breakage; "
            "MMR status: proficient; "
            "ER/PR receptor: PALB2 breast more often ER-positive (vs BRCA1 triple-negative)"
        ),
        "pathognomonic": (
            "PALB2 VARIANT = HIGH BREAST RISK (53% — near BRCA2 level) + moderate ovarian risk (3-5%); "
            "RRSO TIMING CONTROVERSIAL: moderate ovarian risk vs high breast risk (pre-menopausal RRSO causes menopause — weighing quality of life); "
            "BREAST MRI MANDATORY from 25-30 yr for PALB2 monoallelic carriers; "
            "BIALLELIC PALB2/FANCN: severe Fanconi anemia — DEB chromosomal breakage test diagnostic; "
            "PANCREATIC CANCER + BREAST CANCER FH = PALB2 testing alongside BRCA2"
        ),
        "treatment": (
            "Breast cancer: annual MRI from 25-30 yr; consider risk-reducing mastectomy; "
            "PARP inhibitor: emerging data (olaparib — HRD positive tumors respond); "
            "Ovarian cancer — systemic: platinum-taxane; PARP inhibitor emerging; "
            "RRSO: controversial timing — 45-50 yr post-childbearing (some guidelines later due to moderate ovarian risk but high breast risk priority); "
            "OCP: 50% ovarian risk reduction; short-term use acceptable; "
            "Pancreatic surveillance: MRI/MRCP from 50 yr (CAPS consortium); "
            "FANCN (biallelic): allogenic HSCT; avoid radiation (radiosensitivity); "
            "Genetic counselling: clearly separate breast risk (53%) from ovarian risk (3-5%)"
        ),
        "critical_flags": [
            "PALB2-53pct-LIFETIME-BREAST-RISK-ANNUAL-MRI-MANDATORY",
            "RRSO-TIMING-CONTROVERSIAL-MODERATE-OVARIAN-RISK-vs-HIGH-BREAST-RISK",
            "BIALLELIC-FANCN-SEVERE-FA-DEB-TEST",
            "PARP-INHIBITOR-EMERGING-HRD-POSITIVE",
            "PANCREATIC-SURVEILLANCE-FROM-50yr",
            "PALB2-BRIDGES-BRCA1-BRCA2-HR-COMPLEX",
            "3-5pct-OVARIAN-RISK-LOWER-THAN-BRCA1-2",
        ],
    },
    # -- MLH1 — Lynch Syndrome 1 (AD) ----------------------------------------------
    {
        "gene": "MLH1",
        "alt_name": "MLH1 (MLH1-756aa-3p22.2 / AD — Lynch-Syndrome-1 — 10-12pct-Ovarian-Risk-ENDOMETRIOID-NOT-HGSOC — Endometrial-40-60pct — dMMR-MSI-H — Pembrolizumab-FDA-2017 — Universal-MMR-IHC)",
        "protein": (
            "MLH1 -- 3p22.2 AD -- MLH1-756aa -- "
            "MutL-Homolog-1-MMR-Protein-MLH1-PMS2-Heterodimer -- "
            "Lynch-Syndrome-1-HNPCC-1 -- "
            "10-12pct-Lifetime-Ovarian-Cancer-Risk-Endometrioid-Clear-Cell-NOT-HGSOC -- "
            "Endometrial-Cancer-40-60pct-Most-Prominent-Extra-Colonic -- "
            "Colorectal-Cancer-40-80pct -- "
            "dMMR-MSI-H-Pembrolizumab-FDA2017 -- "
            "MLH1-Promoter-Methylation-Sporadic-Somatic-DISTINGUISH"
        ),
        "locus": "3p22.2",
        "protein_size": "756 aa",
        "inheritance": (
            "AD (autosomal dominant) — MLH1 haploinsufficiency (MMR); "
            "Prevalence: 1 in 300-1000 general population; "
            "Penetrance variable — modified by other genetic factors; "
            "Lynch ovarian cancer: endometrioid or clear cell (NOT high-grade serous — CRITICAL DDx vs BRCA1/2); "
            "MLH1 promoter methylation: SOMATIC sporadic event — MLH1 IHC loss ≠ Lynch (must distinguish from germline); "
            "Biallelic MLH1: constitutional MLH1 methylation or biallelic mutations → constitutional mismatch repair deficiency (CMMRD); "
            "CMMRD: severe phenotype — childhood brain tumors, lymphoma, colorectal cancer; "
            "Amsterdam II + Bethesda guidelines: define Lynch testing criteria"
        ),
        "age_of_onset": (
            "Colorectal cancer: 40-50 yr (20-30 yr earlier than sporadic); "
            "Endometrial cancer: 40-60 yr — often earlier presentation than general population; "
            "Ovarian cancer: 40-60 yr (earlier than sporadic 63 yr); "
            "Lynch-associated ovarian cancer: synchronous endometrial + ovarian in 15-20%; "
            "Gastric cancer: elevated risk (Lynch MLH1/MSH2); "
            "Urinary tract: renal pelvis, ureter, bladder — Lynch associated; "
            "Brain tumors (Turcot): MLH1/PMS2 Lynch association; "
            "CMMRD (biallelic): childhood brain tumors, lymphoma, colorectal cancer"
        ),
        "key_biomarker": (
            "MLH1 IHC: absent MLH1/PMS2 staining in tumor — NB: absent MLH1 IHC → check MLH1 promoter methylation first to exclude somatic sporadic; "
            "MLH1 promoter methylation: methylated = sporadic (NOT Lynch) in 70-80% of IHC-absent cases; "
            "BRAF V600E: if MLH1 methylation absent + BRAF mut → sporadic (MLH1 promoter methylated via CIMP); "
            "MSI-H: microsatellite instability high — tumor testing; "
            "Germline MLH1 sequencing: confirms Lynch syndrome 1; "
            "CA-125: ovarian cancer surveillance; "
            "Colonoscopy: 1-2 yr Lynch surveillance; "
            "Endometrial biopsy: symptomatic Lynch carriers annually from 35 yr"
        ),
        "pathognomonic": (
            "SYNCHRONOUS ENDOMETRIAL + OVARIAN CANCER in women under 50 = Lynch syndrome until proven otherwise; "
            "OVARIAN CANCER HISTOLOGY ENDOMETRIOID/CLEAR CELL (not HGSOC) = test MMR IHC; "
            "MLH1 IHC ABSENT → ALWAYS test promoter methylation first before calling Lynch — majority sporadic; "
            "dMMR/MSI-H + MLH1 GERMLINE = pembrolizumab FDA 2017 approved; "
            "COLORECTAL CANCER UNDER 50 + FH OVARIAN = Lynch mandatory testing; "
            "UNIVERSAL MMR IHC on ALL endometrial AND colorectal tumors (NCCN/NICE guideline)"
        ),
        "treatment": (
            "Ovarian cancer: systemic platinum-taxane; pembrolizumab (dMMR/MSI-H FDA 2017); "
            "Endometrial cancer: pembrolizumab; dostarlimab (FDA 2021 — dMMR); lenvatinib+pembrolizumab; "
            "Colorectal cancer: pembrolizumab (KEYNOTE-177: PFS HR 0.60 dMMR 1st line); "
            "Surveillance: colonoscopy 1-2 yr from 25 yr; endometrial biopsy annually from 35 yr; "
            "Risk-reducing hysterectomy + BSO (bilateral salpingo-oophorectomy): 40-45 yr post-childbearing; "
            "Urinary tract: urinalysis annually; "
            "Gastric: EGD from 30-35 yr (MLH1/MSH2 particularly); "
            "Aspirin: 600 mg daily (CAPP2 trial: 63% Lynch CRC risk reduction at 5yr use); "
            "FAMILY SCREENING: universal testing all first-degree relatives"
        ),
        "critical_flags": [
            "MLH1-IHC-ABSENT-CHECK-PROMOTER-METHYLATION-FIRST-MAJORITY-SPORADIC",
            "SYNCHRONOUS-ENDOMETRIAL-OVARIAN-UNDER-50-LYNCH-UNTIL-PROVEN-OTHERWISE",
            "ENDOMETRIOID-CLEAR-CELL-NOT-HGSOC-KEY-DDx-BRCA1-2",
            "PEMBROLIZUMAB-FDA-2017-dMMR-MSI-H",
            "UNIVERSAL-MMR-IHC-ALL-ENDOMETRIAL-COLORECTAL-TUMORS",
            "ASPIRIN-600mg-CAPP2-LYNCH-COLORECTAL-PREVENTION",
            "RISK-REDUCING-HYSTERECTOMY-BSO-40-45yr",
        ],
    },
    # -- MSH2 — Lynch Syndrome 2 (AD) ----------------------------------------------
    {
        "gene": "MSH2",
        "alt_name": "MSH2 (MSH2-934aa-2p21 / AD — Lynch-Syndrome-2 — 10-12pct-Ovarian-Risk — Endometrial-25-60pct — dMMR-MSI-H — EPCAM-3prime-Deletion-Silences-MSH2 — Muir-Torre-Variant-Sebaceous)",
        "protein": (
            "MSH2 -- 2p21 AD -- MSH2-934aa -- "
            "MutS-Homolog-2-MMR-MSH2-MSH6-MutSalpha-Heterodimer -- "
            "Lynch-Syndrome-2-HNPCC-2 -- "
            "10-12pct-Lifetime-Ovarian-Cancer-Risk-Endometrioid-Clear-Cell -- "
            "Endometrial-Cancer-25-60pct -- "
            "Colorectal-Cancer-40-80pct -- "
            "EPCAM-3prime-Deletion-Silences-MSH2-via-Read-Through-Transcription -- "
            "Muir-Torre-Variant-Sebaceous-Gland-Tumors-KEY"
        ),
        "locus": "2p21",
        "protein_size": "934 aa",
        "inheritance": (
            "AD (autosomal dominant) — MSH2 haploinsufficiency; "
            "Prevalence: 1 in 400-1000 general population; "
            "Penetrance: colorectal cancer 40-80%; endometrial 25-60%; ovarian 10-12%; "
            "EPCAM 3-prime end deletions → MSH2 silencing via read-through transcription: "
            "EPCAM deletion detected by MLPA — standard MSH2 sequencing MISSES this variant; "
            "Muir-Torre syndrome: MSH2/MSH6 Lynch variant — sebaceous adenoma/carcinoma + internal malignancies; "
            "MSH2 IHC loss: MSH2 and MSH6 both absent (MSH6 can be absent alone with MSH6 mutations); "
            "No promoter methylation mechanism (unlike MLH1) — MSH2 IHC loss = high probability Lynch"
        ),
        "age_of_onset": (
            "Colorectal cancer: 40-50 yr; right colon predominant (compared to sporadic left colon); "
            "Endometrial cancer: 40-60 yr — often presenting before colorectal diagnosis; "
            "Ovarian cancer: 40-60 yr; endometrioid/clear cell histology; "
            "Sebaceous gland tumors (Muir-Torre): 40-60 yr — multiple sebaceous adenomas/carcinomas; "
            "Urinary tract cancers: 15% (renal pelvis, ureter, bladder); "
            "Small bowel: 1-4%; "
            "Gastric cancer: 5% (MLH1/MSH2 both); "
            "Brain (Turcot variant): glioblastoma"
        ),
        "key_biomarker": (
            "MSH2 IHC: absent MSH2 AND MSH6 in tumor (MSH6 loss follows MSH2 loss); "
            "EPCAM deletion MLPA: MANDATORY if MSH2 IHC absent + sequencing negative — WES/standard seq MISSES EPCAM deletion; "
            "MSI-H: microsatellite instability high; "
            "Germline MSH2 sequencing: pathogenic variant; "
            "Sebaceous tumor: IHC MSH2/MSH6 — Muir-Torre screen; "
            "CA-125: ovarian surveillance; "
            "Endometrial biopsy; "
            "Colonoscopy: Lynch CRC surveillance 1-2 yr from 25 yr"
        ),
        "pathognomonic": (
            "MSH2 IHC ABSENT = high probability Lynch (no promoter methylation mechanism unlike MLH1); "
            "EPCAM 3-PRIME DELETION: standard sequencing misses this — MLPA MANDATORY if MSH2 IHC absent + seq negative; "
            "SEBACEOUS ADENOMA/CARCINOMA = Muir-Torre = MSH2/MSH6 Lynch — test MMR IHC on sebaceous tumors; "
            "SYNCHRONOUS ENDOMETRIAL + OVARIAN CANCER in women under 50 + MSH2/MSH6 absent IHC = Lynch; "
            "RIGHT-SIDED COLORECTAL CANCER under 50 = Universal MMR IHC (NCCN/NICE); "
            "MULTIPLE SEBACEOUS TUMORS in one patient = Muir-Torre = Lynch testing mandatory"
        ),
        "treatment": (
            "Ovarian cancer: platinum-taxane; pembrolizumab (dMMR/MSI-H FDA 2017); "
            "Endometrial cancer: pembrolizumab; dostarlimab (FDA 2021); lenvatinib+pembrolizumab; "
            "Colorectal cancer: pembrolizumab (KEYNOTE-177 dMMR 1st line); "
            "Surveillance: colonoscopy 1-2 yr from 25 yr; endometrial biopsy annually from 35 yr; "
            "Risk-reducing hysterectomy + BSO: 40-45 yr post-childbearing; "
            "Urinary tract: urinalysis annually from 25-30 yr; "
            "Gastric: EGD from 30-35 yr; "
            "Sebaceous: excision + dermatology; "
            "Aspirin: 600 mg daily (CAPP2); "
            "EPCAM deletion carriers: identical management to MSH2 germline mutation"
        ),
        "critical_flags": [
            "EPCAM-3PRIME-DELETION-MLPA-MANDATORY-STANDARD-SEQ-MISSES",
            "MSH2-IHC-ABSENT-NO-METHYLATION-MECHANISM-HIGH-PROBABILITY-LYNCH",
            "SEBACEOUS-ADENOMA-MUIR-TORRE-MSH2-MSH6-TEST-MMR-IHC",
            "SYNCHRONOUS-ENDOMETRIAL-OVARIAN-UNDER-50-LYNCH-TEST",
            "ENDOMETRIOID-CLEAR-CELL-NOT-HGSOC-KEY-DDx-BRCA1-2",
            "PEMBROLIZUMAB-FDA-2017-dMMR-MSI-H",
            "ASPIRIN-600mg-CAPP2-PREVENTION",
        ],
    },
]


def _generate_cohort():
    """Generate 320-patient cohort (40 per gene, seeds 2062-2069)."""
    all_patients = []
    for idx, gene_data in enumerate(OC_GENES):
        gene = gene_data["gene"]
        rng = random.Random(SEED_BASE + idx)
        for i in range(40):
            age = int(rng.gauss(55, 10))
            age = max(28, min(age, 78))
            if gene == "BRCA1":
                hgsoc = rng.random() < 0.90
                breast = rng.random() < 0.72
                rrso_done = rng.random() < 0.55
                parp_inhibitor = hgsoc and rng.random() < 0.65
                bilateral_breast = breast and rng.random() < 0.25
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "hgsoc": hgsoc,
                    "breast_cancer": breast,
                    "rrso_done": rrso_done,
                    "parp_inhibitor_use": parp_inhibitor,
                    "bilateral_breast": bilateral_breast,
                    "ovarian_cancer": hgsoc,
                })
            elif gene == "BRCA2":
                hgsoc = rng.random() < 0.85
                breast = rng.random() < 0.69
                prostate = rng.random() < 0.08
                pancreatic = rng.random() < 0.05
                rrso_done = rng.random() < 0.50
                parp_inhibitor = (hgsoc or prostate or pancreatic) and rng.random() < 0.55
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "hgsoc": hgsoc,
                    "breast_cancer": breast,
                    "prostate_cancer": prostate,
                    "pancreatic_cancer": pancreatic,
                    "rrso_done": rrso_done,
                    "parp_inhibitor_use": parp_inhibitor,
                    "ovarian_cancer": hgsoc,
                })
            elif gene == "BRIP1":
                hgsoc = rng.random() < 0.85
                breast = False  # NO breast risk
                rrso_done = rng.random() < 0.40
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": min(age + 5, 78),
                    "hgsoc": hgsoc,
                    "breast_cancer": breast,
                    "rrso_done": rrso_done,
                    "ovarian_cancer": hgsoc,
                })
            elif gene == "RAD51C":
                hgsoc = rng.random() < 0.82
                breast = rng.random() < 0.03  # very low
                rrso_done = rng.random() < 0.38
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": min(age + 5, 78),
                    "hgsoc": hgsoc,
                    "breast_cancer": breast,
                    "rrso_done": rrso_done,
                    "ovarian_cancer": hgsoc,
                })
            elif gene == "RAD51D":
                hgsoc = rng.random() < 0.82
                breast = rng.random() < 0.02  # very low
                rrso_done = rng.random() < 0.38
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": min(age + 5, 78),
                    "hgsoc": hgsoc,
                    "breast_cancer": breast,
                    "rrso_done": rrso_done,
                    "ovarian_cancer": hgsoc,
                })
            elif gene == "PALB2":
                breast = rng.random() < 0.53
                hgsoc = rng.random() < 0.04
                pancreatic = rng.random() < 0.04
                rrso_done = rng.random() < 0.30
                parp_inhibitor = breast and rng.random() < 0.30
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "hgsoc": hgsoc,
                    "breast_cancer": breast,
                    "pancreatic_cancer": pancreatic,
                    "rrso_done": rrso_done,
                    "parp_inhibitor_use": parp_inhibitor,
                    "ovarian_cancer": hgsoc,
                })
            elif gene == "MLH1":
                endometrioid = rng.random() < 0.75  # ovarian histology
                ovarian = rng.random() < 0.12  # 10-12% lifetime
                endometrial = rng.random() < 0.55
                colorectal = rng.random() < 0.65
                synchronous = endometrial and ovarian and rng.random() < 0.18
                pembrolizumab = (ovarian or endometrial or colorectal) and rng.random() < 0.45
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "ovarian_cancer": ovarian,
                    "endometrioid_histology": endometrioid if ovarian else False,
                    "endometrial_cancer": endometrial,
                    "colorectal_cancer": colorectal,
                    "synchronous_endometrial_ovarian": synchronous,
                    "pembrolizumab_use": pembrolizumab,
                    "dmmr_msi_h": True,
                })
            elif gene == "MSH2":
                ovarian = rng.random() < 0.11
                endometrioid_oc = ovarian and rng.random() < 0.80
                endometrial = rng.random() < 0.45
                colorectal = rng.random() < 0.65
                sebaceous = rng.random() < 0.15  # Muir-Torre
                urinary_tract = rng.random() < 0.12
                epcam_variant = rng.random() < 0.08  # ~8% of MSH2 Lynch due to EPCAM
                synchronous = endometrial and ovarian and rng.random() < 0.18
                pembrolizumab = (ovarian or endometrial or colorectal) and rng.random() < 0.45
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "ovarian_cancer": ovarian,
                    "endometrioid_oc": endometrioid_oc,
                    "endometrial_cancer": endometrial,
                    "colorectal_cancer": colorectal,
                    "sebaceous_tumor": sebaceous,
                    "urinary_tract_cancer": urinary_tract,
                    "epcam_deletion": epcam_variant,
                    "synchronous_endometrial_ovarian": synchronous,
                    "pembrolizumab_use": pembrolizumab,
                    "dmmr_msi_h": True,
                })
    return all_patients


def overview():
    """Return atlas-level aggregate overview."""
    patients = _generate_cohort()
    ovarian_patients = sum(1 for p in patients if p.get("ovarian_cancer"))
    breast_patients = sum(1 for p in patients if p.get("breast_cancer"))
    hgsoc_patients = sum(1 for p in patients if p.get("hgsoc"))
    endometrial_patients = sum(1 for p in patients if p.get("endometrial_cancer"))
    colorectal_patients = sum(1 for p in patients if p.get("colorectal_cancer"))
    rrso_patients = sum(1 for p in patients if p.get("rrso_done"))
    parp_patients = sum(1 for p in patients if p.get("parp_inhibitor_use"))
    pembrolizumab_patients = sum(1 for p in patients if p.get("pembrolizumab_use"))
    synchronous_patients = sum(1 for p in patients if p.get("synchronous_endometrial_ovarian"))
    dmmr_patients = sum(1 for p in patients if p.get("dmmr_msi_h"))
    return {
        "atlas": "Hereditary-Ovarian-Cancer-Atlas",
        "genes": [g["gene"] for g in OC_GENES],
        "total_patients": len(patients),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "ovarian_cancer_patients": ovarian_patients,
        "breast_cancer_patients": breast_patients,
        "hgsoc_patients": hgsoc_patients,
        "endometrial_cancer_patients": endometrial_patients,
        "colorectal_cancer_patients": colorectal_patients,
        "rrso_done_patients": rrso_patients,
        "parp_inhibitor_patients": parp_patients,
        "pembrolizumab_patients": pembrolizumab_patients,
        "synchronous_endometrial_ovarian": synchronous_patients,
        "dmmr_msi_h_patients": dmmr_patients,
    }


def breakdown():
    """Return per-gene breakdown with clinical details."""
    patients = _generate_cohort()
    result = {}
    for gene_data in OC_GENES:
        gene = gene_data["gene"]
        gene_patients = [p for p in patients if p["gene"] == gene]
        result[gene] = {
            "gene": gene,
            "alt_name": gene_data["alt_name"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "patient_count": len(gene_patients),
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "critical_flags": gene_data["critical_flags"],
            "age_of_onset": gene_data["age_of_onset"],
            "key_biomarker": gene_data["key_biomarker"],
        }
    return result


def definitions():
    """Return gene definitions, glossary and surveillance protocols."""
    return {
        "genes": {g["gene"]: g["protein"] for g in OC_GENES},
        "glossary": {
            "HBOC (Hereditary Breast Ovarian Cancer)": "Syndrome caused by germline BRCA1/BRCA2 pathogenic variants; BRCA1 — 44% ovarian, 72% breast lifetime risk; BRCA2 — 17% ovarian, 69% breast lifetime risk; managed with PARP inhibitors and RRSO",
            "HGSOC (High-Grade Serous Ovarian Cancer)": "Most common ovarian cancer subtype; arises from fallopian tube fimbriae; BRCA1/2/BRIP1/RAD51C/RAD51D carriers; platinum-sensitive; HRD positive; PARP inhibitor responsive",
            "RRSO (Risk-Reducing Salpingo-Oophorectomy)": "Bilateral surgical removal of fallopian tubes + ovaries; reduces ovarian cancer risk 80-95%; timing varies by gene: BRCA1 35-40yr, BRCA2 40-45yr, BRIP1/RAD51C/RAD51D 45-50yr; pre-menopausal RRSO also reduces breast risk 50%",
            "PARP Inhibitor": "Poly(ADP-ribose) polymerase inhibitor; exploits HRD (BRCA1/2 HR deficiency) via synthetic lethality; agents: olaparib, niraparib, rucaparib, veliparib; 1st line maintenance: olaparib (SOLO-1), niraparib (PRIMA); relapse: SOLO-2, TRITON3",
            "HRD (Homologous Recombination Deficiency)": "Loss of homologous recombination DNA repair; BRCA1/2 pathogenic variants → HRD; scored by MyChoice CDx (Myriad): LOH + telomeric allelic imbalance + large-scale transitions; HRD positive predicts PARP inhibitor response",
            "Lynch Syndrome": "Autosomal dominant mismatch repair gene deficiency (MLH1/MSH2/MSH6/PMS2/EPCAM); dMMR/MSI-H tumors; colorectal + endometrial + ovarian cancer; endometrioid/clear cell histology (NOT HGSOC); pembrolizumab FDA 2017; aspirin 600mg CAPP2 prevention",
            "dMMR/MSI-H": "Deficient mismatch repair / Microsatellite instability-high; result of MMR gene loss (Lynch germline) or MLH1 promoter methylation (sporadic); IHC: absent MLH1/PMS2 or MSH2/MSH6; pembrolizumab FDA 2017 pan-tumor approval; keytruda KEYNOTE-158/177",
            "Endometrioid Ovarian Cancer": "Ovarian cancer subtype associated with Lynch syndrome; distinct from HGSOC (which is BRCA1/2 related); endometrial-type glands; often synchronous with endometrial cancer (15-20%); dMMR testing mandatory",
            "EPCAM Deletion": "3-prime end deletions of EPCAM gene cause transcriptional read-through into MSH2 promoter → MSH2 silencing; detected by MLPA (NOT standard sequencing); Lynch syndrome phenotype identical to MSH2 germline mutations; ~8% of MSH2-deficient Lynch cases",
            "Muir-Torre Syndrome": "Variant of Lynch syndrome (MSH2/MSH6 most common); sebaceous gland neoplasms (adenoma, epithelioma, carcinoma) + internal malignancies (colorectal, endometrial, ovarian); sebaceous tumor = MMR IHC mandatory",
            "RAD51 Paralogs": "RAD51C and RAD51D: HR repair proteins; moderate ovarian cancer risk (5-6x relative risk) WITHOUT significant breast risk; critical distinction from BRCA1/2; RRSO recommended 45-50yr; PARP inhibitor data emerging; part of BCDX2/CX3 HR complexes",
            "PALB2 Bridge": "PALB2 physically bridges BRCA1 (via coiled-coil domain) and BRCA2 (via WD40); essential for HR complex assembly at DSBs; monoallelic: high breast cancer risk (53% — second only to BRCA1/2); biallelic: Fanconi anemia FANCN; moderate ovarian risk 3-5%",
            "Fanconi Anemia": "Rare AR syndrome from biallelic HR gene defects (BRIP1/FANCJ, RAD51C/FANCO, PALB2/FANCN, BRCA2/FANCD1); bone marrow failure + AML + solid tumors; DEB (diepoxybutane) chromosomal breakage test diagnostic; allogenic HSCT treatment",
            "Pembrolizumab (Keytruda)": "Anti-PD-1 immune checkpoint inhibitor; FDA 2017: dMMR/MSI-H solid tumors (pan-tumor approval); KEYNOTE-177: PFS HR 0.60 vs chemotherapy in dMMR colorectal 1st line; KEYNOTE-158: endometrial, ovarian dMMR response 50%+",
            "CA-125": "Cancer antigen 125; glycoprotein; elevated in ovarian cancer (especially HGSOC); pre-RRSO surveillance (CA-125 + transvaginal USS — limited for early detection in BRCA carriers); post-treatment monitoring; not sufficiently sensitive/specific for population screening",
        },
        "surveillance_protocols": {
            "BRCA1": "Annual breast MRI from 25-30 yr; mammogram from 30 yr; RRSO 35-40 yr post-childbearing; post-RRSO HRT until age 50; OCP option for ovarian protection; annual CA-125 + TVUSS pre-RRSO (limited utility, not guideline-standard); risk-reducing mastectomy option; family cascade testing; prostate cancer screening annual PSA from 40 yr male carriers",
            "BRCA2": "Annual breast MRI from 25-30 yr; mammogram from 30 yr; RRSO 40-45 yr post-childbearing; male carriers: annual PSA from 40 yr + annual breast exam + mammogram from 50 yr; pancreatic MRI from 50 yr (CAPS); annual CA-125 + TVUSS pre-RRSO; OCP option; family cascade testing",
            "BRIP1": "RRSO 45-50 yr post-childbearing; breast surveillance standard population-based (NO MRI escalation); annual pelvic exam pre-RRSO (limited utility); family cascade testing; OCP option; no gastric/colorectal surveillance indicated (no Lynch risk)",
            "RAD51C": "RRSO 45-50 yr post-childbearing; breast surveillance standard population-based; family cascade testing; OCP option; no Lynch-type surveillance indicated; Fanconi workup (DEB test) if biallelic suspected",
            "RAD51D": "RRSO 45-50 yr post-childbearing; breast surveillance standard population-based; family cascade testing; OCP option; no Lynch-type surveillance indicated",
            "PALB2": "Annual breast MRI from 25-30 yr (53% lifetime breast risk); mammogram from 30 yr; RRSO timing controversial (45-50 yr; discuss quality of life with pre-menopausal menopause vs moderate ovarian risk); pancreatic MRI from 50 yr; risk-reducing mastectomy option; family cascade testing; Fanconi workup (DEB) if biallelic",
            "MLH1": "Colonoscopy 1-2 yearly from 25 yr; endometrial biopsy annually from 35 yr; risk-reducing hysterectomy + BSO 40-45 yr post-childbearing; urinalysis from 25-30 yr; gastric EGD from 30-35 yr; annual CA-125 + TVUSS pre-BSO; aspirin 600mg daily (CAPP2); consider brain MRI (Turcot); family cascade testing",
            "MSH2": "Colonoscopy 1-2 yearly from 25 yr; endometrial biopsy annually from 35 yr; risk-reducing hysterectomy + BSO 40-45 yr post-childbearing; urinalysis from 25-30 yr; gastric EGD from 30-35 yr; skin (sebaceous) surveillance — Muir-Torre; EPCAM deletion MLPA if seq negative + IHC absent; aspirin 600mg daily; family cascade testing",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Ovarian cancer patients: {ov['ovarian_cancer_patients']}")
    print(f"Breast cancer patients: {ov['breast_cancer_patients']}")
    print(f"HGSOC patients: {ov['hgsoc_patients']}")
    print(f"Endometrial cancer patients: {ov['endometrial_cancer_patients']}")
    print(f"dMMR/MSI-H patients: {ov['dmmr_msi_h_patients']}")
    print(f"PARP inhibitor patients: {ov['parp_inhibitor_patients']}")
