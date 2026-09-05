#!/usr/bin/env python3
"""Neurocutaneous-Atlas — Complete 8-Gene Neurocutaneous Tumor Predisposition Syndromes Atlas
NF1     (Neurofibromin; ~2818 aa; 17q11.2; RasGAP; AD 50% de novo; 1:3000; café-au-lait + plexiform NF + MPNST; selumetinib FDA 2020) ·
NF2     (Merlin/Schwannomin; ~595 aa; 22q12.2; ERM tumor suppressor; AD 50% de novo; 1:25000; bilateral VS PATHOGNOMONIC; bevacizumab) ·
TSC1    (Hamartin; ~1164 aa; 9q34.13; mTORC1 suppressor; AD 33% de novo; 1:6000-10000; epilepsy 90%; infantile spasms; everolimus FDA 2017) ·
TSC2    (Tuberin; ~1807 aa; 16p13.3; GAP Rheb; AD 80% de novo; 1:6000-10000; more severe than TSC1; LAM women; renal AML hemorrhage) ·
VHL     (pVHL; ~213 aa; 3p25.3; E3 ubiquitin ligase HIF; AD 20% de novo; 1:36000; hemangioblastoma + RCC + pheo; belzutifan FDA 2021) ·
PTEN    (PTEN phosphatase; ~403 aa; 10q23.31; PIP3→PIP2; AD 10% de novo; 1:200000; Cowden/PHTS; macrocephaly + trichilemmoma + cancer surveillance) ·
PTCH1   (Patched-1; ~1447 aa; 9q22.32; Hedgehog/SMO; AD 25% de novo; 1:31000; Gorlin; BCC + OKC + falx calcification; radiation ABSOLUTELY CI) ·
SMARCB1 (INI1/BAF47; ~385 aa; 22q11.23; SWI/SNF; AD 25% de novo; rare; AT/RT + schwannomatosis; INI1 IHC PATHOGNOMONIC)
320-patient aggregate cohort (8 × 40, seeds 1134–1141)
"""

import random

SEED_BASE = 1134

NEUROCUTANEOUS_GENES = [
    # ── NF1 — Neurofibromatosis Type 1, Neurofibromin, RasGAP ──────────────────
    {
        "gene": "NF1",
        "protein": "Neurofibromin (NF1)",
        "alias": (
            "NF1; OMIM gene 613113; 17q11.2; ~2818 aa; Neurofibromatosis type 1 (OMIM #162200); "
            "AD 50% de novo; prevalence 1:3000 (most common single-gene neurocutaneous disorder); "
            "RasGAP — accelerates Ras-GTP → Ras-GDP hydrolysis; LOF → MAPK/ERK hyperactivation; "
            "selumetinib (MEK inhibitor) FDA 2020 for inoperable plexiform NF; "
            "radiation AVOID (MPNST induction risk)"
        ),
        "aa": "~2818 aa",
        "kDa": "~319 kDa",
        "gene_class": "RasGAP (Ras GTPase-activating protein); tumor suppressor; MAPK/PI3K pathway regulator",
        "locus": "17q11.2",
        "omim_gene": 613113,
        "omim_disease": 162200,
        "phenotype": "Neurofibromatosis Type 1 — most common neurocutaneous disorder; café-au-lait macules, neurofibromas, plexiform NF, MPNST, optic pathway glioma, Lisch nodules",
        "disease": (
            "NF1 encodes Neurofibromin, a large GTPase-activating protein (RasGAP) that is the "
            "major negative regulator of the Ras signalling cascade. "
            "NORMAL FUNCTION: Neurofibromin binds GTP-bound Ras (active form) and dramatically "
            "accelerates the intrinsic GTPase activity of Ras, converting active Ras-GTP to "
            "inactive Ras-GDP. This terminates MAPK/ERK and PI3K/mTOR downstream proliferative "
            "signalling. Neurofibromin is particularly critical in Schwann cells, melanocytes, "
            "neurons, and mast cells, which are the primary cell types affected in NF1. "
            "PATHOMECHANISM: Heterozygous LOF variants in NF1 → haploinsufficiency of neurofibromin "
            "→ insufficient Ras-GTP hydrolysis → sustained Ras-GTP and downstream MAPK/ERK "
            "and PI3K/mTOR hyperactivation → hyperproliferation of neural crest-derived cells "
            "→ neurofibromas, plexiform neurofibromas, MPNSTs, optic pathway gliomas. "
            "NIH DIAGNOSTIC CRITERIA (2+ required): "
            "(1) 6+ café-au-lait macules >5mm (prepubertal) or >15mm (postpubertal); "
            "(2) 2+ cutaneous/subcutaneous neurofibromas OR 1 plexiform neurofibroma; "
            "(3) Axillary or inguinal freckling (Crowe sign — PATHOGNOMONIC when present); "
            "(4) Optic pathway glioma (OPG) — often asymptomatic, detected on MRI; "
            "(5) 2+ Lisch nodules (iris hamartomas on slit-lamp examination); "
            "(6) Distinctive osseous lesion (sphenoid wing dysplasia, long bone pseudarthrosis, tibial bowing); "
            "(7) First-degree relative with NF1. "
            "PLEXIFORM NEUROFIBROMAS (PNF): deep nerve sheath tumors, present at birth (congenital), "
            "grow during childhood; risk of MPNST transformation ~10% lifetime. "
            "Selumetinib (MEK inhibitor): FDA approved April 2020 for inoperable/symptomatic PNF "
            "in children ≥2 years — first targeted therapy for NF1; shrinks PNF 20-70%. "
            "MPNST (Malignant Peripheral Nerve Sheath Tumor): most common cause of NF1-related "
            "mortality; risk factors: rapid PNF enlargement (>20%/year), new pain in PNF, "
            "new neurological deficit → URGENT MRI + biopsy. Lifetime risk ~10%; "
            "5-year survival ~20% — major unmet need. "
            "LEARNING DISABILITY: 80% have some learning difficulty; IQ in normal range but "
            "specific learning disability (attention, executive function, visuospatial). "
            "DRUG ALERT: Radiation therapy → ABSOLUTELY AVOID in NF1 unless no alternative "
            "(induces MPNST in radiation field in NF1; multiple secondary tumors; net harm). "
            "RAS pathway inhibitors (RAF/MEK) are first-line for MPNST."
        ),
        "inheritance": "Autosomal Dominant (AD); 50% de novo; 100% penetrance; variable expressivity; NF1 mutational spectrum: frameshift, nonsense, missense, splicing, large deletions (microdeletion syndrome: more severe)",
        "hallmark": "CALMs (café-au-lait macules) ≥6 at puberty; plexiform neurofibromas; Lisch nodules; axillary freckling (Crowe sign PATHOGNOMONIC); OPG; MPNST risk 10%",
        "key_ddx": "Legius syndrome (CALMs, no neurofibromas, no Lisch, SPRED1 gene), NF2 (bilateral VS, no CALMs), Schwannomatosis (SMARCB1/LZTR1, adults, no CALMs), Watson syndrome (NF1 allelic), constitutional mismatch repair deficiency (multiple CALMs in childhood cancer)",
        "treatment_alert": "Selumetinib (MEK inhibitor) FDA 2020 for symptomatic/inoperable PNF age ≥2y; AVOID radiation (MPNST induction); annual MRI brain/orbit for OPG; neurodevelopmental support",
        "seed": 1134,
        "cohort_n": 40,
        # Cohort generation rates
        "epilepsy_rate": 0.07,
        "cns_tumor_rate": 0.17,
        "mpnst_rate": 0.10,
        "learning_disability_rate": 0.80,
        "plexiform_rate": 0.40,
        "cafeau_lait_rate": 1.00,
        "de_novo_rate": 0.50,
        "drug_error_rate": 0.08,
        "severity_weights": {"Mild": 0.30, "Moderate": 0.50, "Severe": 0.20},
        "skin_lesion_type": "Café-au-lait macules + neurofibromas",
        "primary_malignancy": "MPNST",
    },

    # ── NF2 — Neurofibromatosis Type 2, Merlin (Schwannomin), ERM ──────────────
    {
        "gene": "NF2",
        "protein": "Merlin / Schwannomin (NF2)",
        "alias": (
            "NF2; OMIM gene 607379; 22q12.2; ~595 aa; Neurofibromatosis type 2 (OMIM #101000); "
            "AD 50% de novo; prevalence 1:25000; ERM family tumor suppressor; Hippo pathway; "
            "bilateral vestibular schwannomas PATHOGNOMONIC (Manchester criteria); "
            "bevacizumab (anti-VEGF) — FDA for growing schwannomas; annual audiological surveillance mandatory"
        ),
        "aa": "~595 aa",
        "kDa": "~67 kDa",
        "gene_class": "ERM (Ezrin-Radixin-Moesin) family tumor suppressor; links actin cytoskeleton to membrane; regulates Hippo pathway, contact inhibition, RTK signalling",
        "locus": "22q12.2",
        "omim_gene": 607379,
        "omim_disease": 101000,
        "phenotype": "Neurofibromatosis Type 2 — bilateral vestibular schwannomas (PATHOGNOMONIC) + meningiomas + ependymomas + cataracts",
        "disease": (
            "NF2 encodes Merlin (also called Schwannomin), an ERM (Ezrin-Radixin-Moesin) family "
            "protein that functions as a tumor suppressor at the interface of the cell membrane "
            "and actin cytoskeleton. "
            "NORMAL FUNCTION: Merlin links integral membrane proteins to the actin cytoskeleton, "
            "regulating: (1) Hippo pathway activation → YAP/TAZ phosphorylation and cytoplasmic "
            "retention → growth suppression; (2) Contact-dependent growth inhibition — merlin "
            "mediates the growth arrest response to cell-cell contact; "
            "(3) Receptor tyrosine kinase signalling attenuation at the cell membrane; "
            "(4) PI3K/mTOR and Ras/MAPK suppression. "
            "Merlin is particularly essential in Schwann cells, meningeal cells, and ependymal cells. "
            "PATHOMECHANISM: LOF variants in NF2 → absent/non-functional merlin → loss of Hippo "
            "pathway and contact-dependent growth inhibition → unregulated PI3K/mTOR and Ras/MAPK "
            "→ schwannoma, meningioma, ependymoma formation. "
            "NF2 follows a classic two-hit tumor suppressor model: germline LOF + somatic second hit "
            "in each tumor (LOH at 22q12). "
            "PATHOGNOMONIC FEATURE: Bilateral vestibular schwannomas (bilateral acoustic neuromas) "
            "on MRI/audiogram → diagnostic without family history. "
            "MANCHESTER DIAGNOSTIC CRITERIA: "
            "Definite: bilateral VS on MRI; OR unilateral VS age <30 + first-degree relative with NF2. "
            "BEVACIZUMAB (anti-VEGF): Reduces schwannoma volume 30-40% and improves hearing "
            "in growing schwannomas in NF2 — off-label but well-established; IV monthly infusions. "
            "HEARING LOSS: 95% develop significant hearing loss due to bilateral VS compression of "
            "cochlear nerves; auditory brainstem implants (ABI) offer alternative to CI in NF2 "
            "when cochlear nerve is damaged/resected. "
            "CATARACTS: subcapsular posterior lens opacities in 60% — slit-lamp examination "
            "annually from diagnosis; juvenile onset. "
            "MENINGIOMAS: multiple, often asymptomatic; intracranial and spinal; progesterone "
            "receptors → rapid growth during pregnancy (progesterone-driven). "
            "EPENDYMOMAS: spinal cord ependymoma in 30% — serial spinal MRI mandatory. "
            "PREGNANCY CAUTION: NF2 meningiomas and schwannomas can grow rapidly during pregnancy; "
            "multidisciplinary neurosurgical/obstetric planning essential."
        ),
        "inheritance": "Autosomal Dominant (AD); 50% de novo; constitutional mosaicism in 25-30% of apparent de novo cases (milder phenotype); two-hit tumor suppressor",
        "hallmark": "Bilateral vestibular schwannomas (PATHOGNOMONIC); cataracts (juvenile posterior); multiple meningiomas; spinal ependymomas; NO café-au-lait macules; hearing loss 95%",
        "key_ddx": "NF1 (CALMs, neurofibromas, no bilateral VS), Schwannomatosis-SMARCB1 (peripheral schwannomas, no bilateral VS, no meningiomas), sporadic unilateral VS (no second tumor, no family history), meningiomatosis",
        "treatment_alert": "Bevacizumab (anti-VEGF) for growing schwannomas; annual audiological surveillance from diagnosis; neurosurgery for VS with brainstem compression or rapid growth; ABI if cochlear nerve sacrificed",
        "seed": 1135,
        "cohort_n": 40,
        "epilepsy_rate": 0.12,
        "cns_tumor_rate": 1.00,  # bilateral schwannomas defining
        "mpnst_rate": 0.00,  # no MPNST in NF2
        "learning_disability_rate": 0.05,
        "plexiform_rate": 0.00,
        "cafeau_lait_rate": 0.10,  # small number, not diagnostic
        "de_novo_rate": 0.50,
        "drug_error_rate": 0.06,
        "severity_weights": {"Mild": 0.20, "Moderate": 0.50, "Severe": 0.30},
        "skin_lesion_type": "Subcutaneous schwannomas (few), NO CALMs",
        "primary_malignancy": None,
    },

    # ── TSC1 — Tuberous Sclerosis Complex Type 1, Hamartin ─────────────────────
    {
        "gene": "TSC1",
        "protein": "Hamartin (TSC1)",
        "alias": (
            "TSC1; OMIM gene 605284; 9q34.13; ~1164 aa; Tuberous sclerosis complex (OMIM #191100); "
            "AD 33% de novo; prevalence 1:6000-10000; hamartin binds tuberin (TSC2) → GAP complex "
            "suppressing Rheb → mTORC1 OFF; epilepsy 90%; infantile spasms; vigabatrin first-line; "
            "everolimus (mTOR inhibitor) FDA 2017 for drug-resistant TSC epilepsy and SEGA; TSC1 less severe than TSC2"
        ),
        "aa": "~1164 aa",
        "kDa": "~130 kDa",
        "gene_class": "TSC1/2 complex scaffolding protein; hamartin-tuberin GAP complex suppresses Rheb → inhibits mTORC1",
        "locus": "9q34.13",
        "omim_gene": 605284,
        "omim_disease": 191100,
        "phenotype": "Tuberous Sclerosis Complex Type 1 — epilepsy + cortical tubers + SEGA + cardiac rhabdomyoma + facial angiofibromas + renal AML + hypomelanotic macules + autism",
        "disease": (
            "TSC1 encodes Hamartin, a large scaffolding protein that forms an obligate heterodimer "
            "with Tuberin (TSC2), creating the TSC1/TSC2 GAP complex that is the primary negative "
            "regulator of mTORC1. "
            "NORMAL FUNCTION: The TSC1-TSC2 complex has GTPase-activating protein (GAP) activity "
            "toward the small GTPase Rheb. TSC2 (tuberin) catalyses GTP hydrolysis of Rheb → "
            "Rheb-GDP (inactive) → cannot activate mTORC1 → mTOR pathway OFF. TSC1 (hamartin) "
            "stabilises TSC2 and scaffolds the complex to lysosomes. Together, the complex "
            "integrates growth factor signals (PI3K/AKT), energy status (AMPK), and hypoxia "
            "signals upstream of mTOR. "
            "PATHOMECHANISM: LOF variants in TSC1 → absent/unstable hamartin → destabilisation of "
            "TSC2 → TSC1/TSC2 complex LOF → Rheb remains GTP-bound → mTORC1 constitutively active "
            "→ unregulated protein synthesis, cell growth, proliferation → hamartoma formation in "
            "brain, kidney, lung, skin, heart. "
            "EPILEPSY (90%): most common and debilitating manifestation. "
            "Infantile spasms: 50% of TSC develop infantile spasms in first year of life → "
            "PATHOGNOMONIC context in TSC; vigabatrin is first-line (reduces spasms by 95% in TSC "
            "vs ~50% in other causes). Cortical tubers → epileptogenic zones. "
            "mTOR INHIBITORS (everolimus): FDA September 2017 for TSC-associated partial-onset "
            "seizures (ages ≥2 years); reduces seizure frequency 50%+ vs placebo. "
            "SEGA (Subependymal Giant Cell Astrocytoma): benign but obstructs foramen of Monro "
            "→ obstructive hydrocephalus (12% TSC1, 17% TSC2); everolimus shrinks SEGA; "
            "serial MRI surveillance every 1-3 years mandatory. "
            "RENAL AML >3cm: high haemorrhage risk (Wunderlich syndrome — retroperitoneal bleed); "
            "everolimus or embolization when AML >3 cm. "
            "CARDIAC RHABDOMYOMA: present at birth in 60% TSC1; often regress spontaneously post-neonatal; "
            "fetal cardiac rhabdomyoma on prenatal US → screen for TSC. "
            "FACIAL ANGIOFIBROMAS: 85%; appear age 3-5; pathognomonic in TSC context. "
            "VIGABATRIN MONITORING: visual field constriction (irreversible retinal toxicity) in "
            "30-50% of long-term users → formal visual field testing every 6-12 months mandatory."
        ),
        "inheritance": "Autosomal Dominant (AD); 33% de novo in TSC1; two-hit model for tumour formation; TSC1 less severe overall than TSC2",
        "hallmark": "Cortical tubers + epilepsy 90% + infantile spasms (vigabatrin first-line) + hypomelanotic macules + facial angiofibromas + SEGA + cardiac rhabdomyoma + renal AML; TSC1 generally less severe than TSC2",
        "key_ddx": "TSC2 (more severe: more tubers, higher SEGA rate, more AML, more LAM in women, higher ID rate), focal cortical dysplasia (no systemic features), NF1 (CALMs not hypomelanotic macules), Cowden/PTEN (macrocephaly, different skin lesions)",
        "treatment_alert": "Vigabatrin first-line for infantile spasms in TSC; visual field monitoring MANDATORY; everolimus for drug-resistant epilepsy + SEGA + AML >3cm; mTOR inhibitors NOT anti-epileptic drugs",
        "seed": 1136,
        "cohort_n": 40,
        "epilepsy_rate": 0.90,
        "cns_tumor_rate": 0.90,  # cortical tubers
        "mpnst_rate": 0.00,
        "learning_disability_rate": 0.50,
        "plexiform_rate": 0.00,
        "cafeau_lait_rate": 0.00,
        "de_novo_rate": 0.33,
        "drug_error_rate": 0.10,
        "severity_weights": {"Mild": 0.15, "Moderate": 0.40, "Severe": 0.45},
        "skin_lesion_type": "Hypomelanotic macules + facial angiofibromas + shagreen patches",
        "primary_malignancy": None,
        "autism_rate": 0.35,
        "sega_rate": 0.12,
        "renal_aml_rate": 0.55,
    },

    # ── TSC2 — Tuberous Sclerosis Complex Type 2, Tuberin ──────────────────────
    {
        "gene": "TSC2",
        "protein": "Tuberin (TSC2)",
        "alias": (
            "TSC2; OMIM gene 191092; 16p13.3; ~1807 aa; Tuberous sclerosis complex (OMIM #613254); "
            "AD 80% de novo; prevalence 1:6000-10000; tuberin = GAP for Rheb → mTOR OFF; "
            "MORE SEVERE than TSC1: higher epilepsy rate, more cortical tubers, higher SEGA rate, "
            "more renal AML, LAM in women (pulmonary lymphangioleiomyomatosis); renal AML >3cm → embolize"
        ),
        "aa": "~1807 aa",
        "kDa": "~200 kDa",
        "gene_class": "RhebGAP; GAP domain catalyzes Rheb GTP hydrolysis → mTORC1 suppression; TSC2 is the catalytic subunit of TSC1/TSC2 complex",
        "locus": "16p13.3",
        "omim_gene": 191092,
        "omim_disease": 613254,
        "phenotype": "Tuberous Sclerosis Complex Type 2 — more severe than TSC1; epilepsy 92% + more cortical tubers + higher SEGA rate + LAM in women + higher renal AML + higher autism/ID rates",
        "disease": (
            "TSC2 encodes Tuberin, the catalytic subunit of the TSC1/TSC2 heterodimer GAP complex. "
            "NORMAL FUNCTION: Tuberin has the GAP activity in the complex — its GAP domain "
            "accelerates GTP hydrolysis by Rheb (Ras homolog enriched in brain) → Rheb-GDP "
            "(inactive) → mTORC1 pathway OFF. Rheb-GTP (when TSC2 is LOF) activates mTORC1 "
            "kinase → hyperphosphorylation of S6K1 and 4E-BP1 → unregulated translation, "
            "cell growth, proliferation, and survival. "
            "TSC1 (hamartin) stabilises TSC2 and prevents its ubiquitin-mediated degradation. "
            "PATHOMECHANISM: TSC2 LOF variants → absent tuberin → TSC1-TSC2 complex disruption "
            "→ Rheb-GTP accumulates → constitutive mTORC1 hyperactivation → hamartoma formation "
            "throughout the body. TSC2 mutations are the DIRECT cause of mTOR dysregulation "
            "(vs TSC1 acting indirectly via TSC2 stabilisation). "
            "TSC2 IS MORE SEVERE THAN TSC1: "
            "More cortical tubers (average 18 vs 7 tubers), higher cognitive impairment (65% ID vs 50%), "
            "higher autism rate (45% vs 35%), more SEGA (17% vs 12%), higher renal AML load (70% vs 55%), "
            "more severe epilepsy (92% vs 90%), earlier seizure onset. "
            "LAM (Lymphangioleiomyomatosis): TSC2 is especially important in TSC2-LAM — "
            "pulmonary LAM occurs in 15-30% of TSC2 females (vs rare in TSC1 females). "
            "LAM = cystic lung destruction by TSC2-null smooth muscle-like cells → progressive "
            "dyspnea, recurrent pneumothorax; sirolimus (mTOR inhibitor) reduces FEV1 decline. "
            "RENAL AML >3cm: significantly higher rate in TSC2 (70% vs 55% in TSC1); "
            "Wunderlich syndrome (life-threatening retroperitoneal haemorrhage) → embolization or "
            "everolimus when AML >3 cm; >4 cm → higher priority. "
            "INFANTILE SPASMS: vigabatrin is first-line; visual field constriction monitoring mandatory "
            "(every 6-12 months; irreversible retinal toxicity risk 30-50% long-term). "
            "mTOR INHIBITORS: everolimus FDA 2017 for TSC seizures; 2012 for SEGA; 2017 for AML. "
            "Sirolimus alternative, especially for LAM."
        ),
        "inheritance": "Autosomal Dominant (AD); 80% de novo (higher de novo rate than TSC1); two-hit tumor suppressor; sporadic cases more common than familial for TSC2",
        "hallmark": "More severe than TSC1; cortical tubers 95% + epilepsy 92% (infantile spasms) + SEGA 17% + renal AML 70% + LAM 15% (women) + autism 45% + ID 65%; vigabatrin visual field monitoring mandatory",
        "key_ddx": "TSC1 (less severe, lower de novo rate, less LAM, fewer tubers), focal cortical dysplasia, isolated LAM (no systemic TSC features — SOMATIC TSC2 mutation), isolated renal AML (may be TSC2 forme fruste)",
        "treatment_alert": "Vigabatrin (infantile spasms) visual field monitoring MANDATORY; everolimus for epilepsy + SEGA + AML; sirolimus for LAM; embolize AML >3cm BEFORE haemorrhage; TSC2 more severe than TSC1",
        "seed": 1137,
        "cohort_n": 40,
        "epilepsy_rate": 0.92,
        "cns_tumor_rate": 0.95,  # cortical tubers
        "mpnst_rate": 0.00,
        "learning_disability_rate": 0.65,
        "plexiform_rate": 0.00,
        "cafeau_lait_rate": 0.00,
        "de_novo_rate": 0.80,
        "drug_error_rate": 0.12,
        "severity_weights": {"Mild": 0.08, "Moderate": 0.32, "Severe": 0.60},
        "skin_lesion_type": "Hypomelanotic macules + facial angiofibromas + shagreen patches (more prominent than TSC1)",
        "primary_malignancy": None,
        "autism_rate": 0.45,
        "sega_rate": 0.17,
        "renal_aml_rate": 0.70,
        "lam_rate": 0.15,
    },

    # ── VHL — Von Hippel-Lindau Disease, pVHL, HIF-α ubiquitination ────────────
    {
        "gene": "VHL",
        "protein": "pVHL (VHL tumor suppressor)",
        "alias": (
            "VHL; OMIM gene 608537; 3p25.3; ~213 aa; Von Hippel-Lindau syndrome (OMIM #193300); "
            "AD 20% de novo; prevalence 1:36000; pVHL = E3 ubiquitin ligase substrate recognition → "
            "ubiquitinates HIF-1α/HIF-2α → proteasomal degradation under normoxia; LOF → HIF accumulation "
            "→ VEGF/PDGF/EPO upregulation → highly vascular tumors; belzutifan (HIF-2α inhibitor) FDA 2021; "
            "ELST → progressive SNHL; annual MRI surveillance mandatory"
        ),
        "aa": "~213 aa",
        "kDa": "~24 kDa",
        "gene_class": "Substrate recognition component of CRL2VHL E3 ubiquitin ligase complex; targets HIF-1α and HIF-2α for proteasomal degradation under normoxic conditions",
        "locus": "3p25.3",
        "omim_gene": 608537,
        "omim_disease": 193300,
        "phenotype": "Von Hippel-Lindau Syndrome — hemangioblastoma (CNS/retinal) + RCC (clear cell) + pheochromocytoma + ELST + pancreatic NET/cysts",
        "disease": (
            "VHL encodes pVHL, the substrate recognition subunit of the CRL2VHL E3 ubiquitin ligase "
            "complex (also contains Elongin B, Elongin C, Cullin-2, Rbx1). "
            "NORMAL FUNCTION: Under normoxia, pVHL binds hydroxylated proline residues in the "
            "oxygen-dependent degradation (ODD) domain of HIF-1α and HIF-2α (hydroxylated by "
            "PHD1/2/3 prolyl hydroxylases when oxygen is present). pVHL binding → ubiquitination "
            "of HIF-α → proteasomal degradation. This prevents pseudo-hypoxic gene expression "
            "under normal oxygen conditions. "
            "pVHL also has HIF-independent roles: microtubule stability, primary cilia assembly, "
            "and NF-κB attenuation. "
            "PATHOMECHANISM: LOF mutations in VHL → absent/non-functional pVHL → HIF-1α and HIF-2α "
            "accumulate despite normal oxygen levels → constitutive 'pseudo-hypoxic' transcriptional "
            "programme → upregulation of VEGF, PDGF-B, GLUT1, EPO, TGF-α → highly vascularised "
            "tumors (hemangioblastomas, RCC) and erythrocytosis. "
            "VHL TYPE CLASSIFICATION (genotype-phenotype): "
            "Type 1 (truncating/large deletion): hemangioblastoma + RCC, LOW pheochromocytoma risk; "
            "Type 2A (missense, surface): pheo + hemangioblastoma, LOW RCC risk; "
            "Type 2B (missense, buried hydrophobic): pheo + hemangioblastoma + RCC (high risk all three); "
            "Type 2C (missense, VDEP surface): pheo ONLY; no hemangioblastoma, no RCC. "
            "HEMANGIOBLASTOMA: benign but symptom-producing via mass effect; most common in cerebellum "
            "(70%), spinal cord (20%), brainstem (10%), retina (retinal angioma/capillary hemangioblastoma). "
            "BELZUTIFAN (HIF-2α inhibitor): FDA August 2021 for VHL-associated RCC, CNS hemangioblastoma, "
            "and pNET — first oral targeted therapy for VHL; once-daily pill; well-tolerated. "
            "ELST (Endolymphatic Sac Tumor): 12% VHL; locally aggressive; causes progressive unilateral "
            "SNHL, tinnitus, vertigo; CT/MRI temporal bone at baseline for all VHL patients; "
            "surgical resection before hearing loss is critical. "
            "SURVEILLANCE PROTOCOL: Annual MRI brain/spine (hemangioblastoma) + abdominal MRI "
            "(RCC/pNET/pheo/cysts) + ophthalmology + 24h urine catecholamines/metanephrines + "
            "blood pressure monitoring."
        ),
        "inheritance": "Autosomal Dominant (AD); 20% de novo; two-hit tumor suppressor; somatic second hit in each tumor; VHL type predicts phenotype",
        "hallmark": "Hemangioblastoma (CNS/retinal) + clear cell RCC + pheochromocytoma + ELST (progressive SNHL); belzutifan FDA 2021; annual MRI surveillance mandatory; VHL typing guides risk prediction",
        "key_ddx": "Sporadic hemangioblastoma (no RCC, no family history), sporadic pheochromocytoma (SDHx/RET/MAX mutation), PHEO/PGL syndromes (SDHB/SDHC/SDHD), Birt-Hogg-Dubé (renal tumors, skin fibrofolliculomas)",
        "treatment_alert": "Belzutifan (HIF-2α inhibitor) FDA 2021 for VHL-RCC/hemangioblastoma/pNET; annual MRI surveillance; ELST — baseline CT temporal bone + serial follow-up; alpha-blockade before pheochromocytoma surgery",
        "seed": 1138,
        "cohort_n": 40,
        "epilepsy_rate": 0.05,  # from CNS hemangioblastoma
        "cns_tumor_rate": 0.70,  # hemangioblastoma
        "mpnst_rate": 0.00,
        "learning_disability_rate": 0.03,
        "plexiform_rate": 0.00,
        "cafeau_lait_rate": 0.00,
        "de_novo_rate": 0.20,
        "drug_error_rate": 0.08,
        "severity_weights": {"Mild": 0.25, "Moderate": 0.50, "Severe": 0.25},
        "skin_lesion_type": "None (no skin lesions in VHL)",
        "primary_malignancy": "Clear cell RCC",
        "hemangioblastoma_rate": 0.70,
        "retinal_angioma_rate": 0.40,
        "rcc_rate": 0.45,
        "pheo_rate": 0.20,
        "elst_rate": 0.12,
    },

    # ── PTEN — PTEN Hamartoma Tumor Syndrome / Cowden Syndrome ─────────────────
    {
        "gene": "PTEN",
        "protein": "PTEN phosphatase (phosphatase and tensin homolog)",
        "alias": (
            "PTEN; OMIM gene 601728; 10q23.31; ~403 aa; PTEN Hamartoma Tumor Syndrome / Cowden syndrome "
            "(OMIM #158350) / BRR (OMIM #153480); AD 10% de novo; prevalence 1:200000; "
            "PTEN = dual-specificity phosphatase PIP3→PIP2 → opposes PI3K → inhibits AKT/mTOR; "
            "LOF → AKT/mTOR hyperactivation → hamartomas + cancer risk; "
            "macrocephaly + trichilemmoma (PATHOGNOMONIC) + thyroid + breast + uterine cancer surveillance"
        ),
        "aa": "~403 aa",
        "kDa": "~47 kDa",
        "gene_class": "Dual-specificity lipid/protein phosphatase; dephosphorylates PIP3 at the 3-position → PIP2; principal antagonist of PI3K/AKT/mTOR signalling",
        "locus": "10q23.31",
        "omim_gene": 601728,
        "omim_disease": 158350,
        "phenotype": "PTEN Hamartoma Tumor Syndrome (PHTS) — Cowden syndrome + Bannayan-Riley-Ruvalcaba syndrome; macrocephaly + mucocutaneous lesions + high cancer risks (breast 20-50%, thyroid 30%, uterine 10-30%)",
        "disease": (
            "PTEN encodes PTEN (phosphatase and tensin homolog), a dual-specificity phosphatase "
            "that is the primary negative regulator of the PI3K/AKT/mTOR signalling axis. "
            "NORMAL FUNCTION: PTEN dephosphorylates phosphatidylinositol-3,4,5-trisphosphate "
            "(PIP3) at the 3-position → producing PIP2. PIP3 is the second messenger generated "
            "by PI3K activation (by growth factor receptors) that recruits and activates AKT at "
            "the plasma membrane. By converting PIP3→PIP2, PTEN prevents AKT activation → "
            "mTORC1/mTORC2 remain suppressed → cell growth, survival, and proliferation are "
            "controlled. PTEN is also present in the nucleus where it maintains genomic stability. "
            "PATHOMECHANISM: LOF variants in PTEN → absent/non-functional PTEN phosphatase → "
            "PIP3 accumulates → constitutive AKT-Ser473 and AKT-Thr308 phosphorylation → "
            "mTORC1 and mTORC2 hyperactivation → unrestricted cell growth, proliferation, survival "
            "→ hamartoma formation and high cancer susceptibility. "
            "PTEN HAMARTOMA TUMOR SYNDROME (PHTS): umbrella term for Cowden, BRR, Proteus-like syndromes. "
            "COWDEN SYNDROME hallmarks: "
            "Multiple trichilemmomas (benign follicular infundibular tumors) on face — PATHOGNOMONIC; "
            "Oral mucosal papillomas (cobblestone tongue/gingiva); "
            "Acral keratoses; Thyroid follicular adenoma/cancer; "
            "Macrocephaly (OFC >98th centile in 95% — key first clue); "
            "Uterine leiomyoma/cancer; Colon hamartomatous polyps. "
            "BANNAYAN-RILEY-RUVALCABA SYNDROME (BRR): "
            "Young males predominant presentation; macrocephaly + lipomatosis + "
            "penile lentigines (PATHOGNOMONIC in males) + intracranial lipomas + autism/ID. "
            "CANCER SURVEILLANCE: Mandatory for all PHTS: "
            "Annual breast MRI (from age 30-35 or 10 years before youngest family case) + mammography; "
            "Annual thyroid ultrasound from age 18 (or from diagnosis if younger); "
            "Colonoscopy every 5 years from age 35; "
            "Annual pelvic ultrasound + endometrial sampling from age 30-35 (women); "
            "Annual dermatology skin check. "
            "mTOR inhibitors (everolimus) in clinical trials for PHTS. "
            "AKT inhibitors and PI3K inhibitors under investigation."
        ),
        "inheritance": "Autosomal Dominant (AD); 10% de novo; variable penetrance; Cowden and BRR are allelic (same PTEN gene); germline PTEN testing for ALL Cowden/BRR/macrocephaly + hamartoma presentations",
        "hallmark": "Macrocephaly (95%) + trichilemmoma (PATHOGNOMONIC) + oral papillomas + penile lentigines (BRR males) + HIGH cancer risks (breast/thyroid/uterine); PI3K/AKT/mTOR hyperactivation",
        "key_ddx": "TSC1/TSC2 (hypomelanotic macules not trichilemmomas, mTOR-specific features), Cowden-like (KLLN/PIK3CA/SDHB gene panel), Gorlin syndrome (PTCH1 — BCC/OKC not breast cancer), juvenile polyposis (SMAD4/BMPR1A — GI polyps)",
        "treatment_alert": "Cancer surveillance mandatory: annual breast MRI + thyroid US + colonoscopy 5-yearly + endometrial sampling; mTOR inhibitor trials; penile lentigines (BRR males) = PTEN diagnosis until proven otherwise",
        "seed": 1139,
        "cohort_n": 40,
        "epilepsy_rate": 0.22,
        "cns_tumor_rate": 0.05,
        "mpnst_rate": 0.00,
        "learning_disability_rate": 0.15,
        "plexiform_rate": 0.00,
        "cafeau_lait_rate": 0.00,
        "de_novo_rate": 0.10,
        "drug_error_rate": 0.10,
        "severity_weights": {"Mild": 0.35, "Moderate": 0.45, "Severe": 0.20},
        "skin_lesion_type": "Trichilemmomas + oral papillomas + acral keratoses",
        "primary_malignancy": "Breast cancer / Thyroid cancer",
        "macrocephaly_rate": 0.95,
        "thyroid_cancer_rate": 0.30,
        "breast_cancer_rate": 0.20,
        "autism_rate": 0.25,
    },

    # ── PTCH1 — Gorlin Syndrome / NBCCS, Hedgehog Pathway, Patched-1 ───────────
    {
        "gene": "PTCH1",
        "protein": "Patched-1 (PTCH1)",
        "alias": (
            "PTCH1; OMIM gene 601309; 9q22.32; ~1447 aa; Gorlin syndrome / NBCCS (OMIM #109400); "
            "AD 25% de novo; prevalence 1:31000; PTCH1 = transmembrane Hedgehog receptor; in absence of Hh → "
            "PTCH1 inhibits SMO → pathway OFF; Hh binds PTCH1 → SMO active → GLI transcription; "
            "BCC (hundreds in UV/radiation field) + OKC + calcified falx; "
            "CRITICAL: radiation ABSOLUTELY CI → field cancerization; vismodegib/sonidegib for BCC"
        ),
        "aa": "~1447 aa",
        "kDa": "~162 kDa",
        "gene_class": "12-transmembrane Hedgehog pathway receptor/repressor; inhibits Smoothened (SMO) in absence of Hedgehog ligand; tumor suppressor via GLI transcription factor suppression",
        "locus": "9q22.32",
        "omim_gene": 601309,
        "omim_disease": 109400,
        "phenotype": "Gorlin Syndrome (NBCCS) — multiple basal cell carcinomas + odontogenic keratocysts + calcified falx cerebri + bifid ribs + medulloblastoma (childhood); radiation ABSOLUTELY CONTRAINDICATED",
        "disease": (
            "PTCH1 encodes Patched-1, a 12-transmembrane protein that is the receptor for Hedgehog "
            "(Hh) ligands (Sonic Hh, Desert Hh, Indian Hh) and the key negative regulator of the "
            "Hedgehog signalling pathway. "
            "NORMAL FUNCTION: In the absence of Hh ligand, PTCH1 constitutively inhibits Smoothened "
            "(SMO), a G-protein coupled receptor-like protein, by preventing its translocation into "
            "the primary cilium. SMO inactive → no activation of GLI transcription factors "
            "(GLI1, GLI2, GLI3) → Hedgehog target genes (PTCH1 itself, GLI1, CCND1, Snail, "
            "Wnt ligands, VEGF) remain suppressed. When Hh ligand binds PTCH1 → SMO is released "
            "from inhibition → SMO activates GLI transcription factors → Hh target gene upregulation. "
            "PATHOMECHANISM: Heterozygous LOF variants in PTCH1 → haploinsufficiency → "
            "reduced SMO inhibition → SMO constitutively partially active → partial Hedgehog pathway "
            "activation even without ligand. Second hit (somatic LOH) in each BCC → complete SMO "
            "derepression → full Hedgehog pathway activation → BCC. "
            "CRITICAL DRUG CONTRAINDICATION — RADIATION THERAPY: "
            "In Gorlin syndrome, ionising radiation (X-ray, proton, photon) causes massive "
            "'field cancerization' — hundreds of BCCs appear within the radiation field over "
            "subsequent years. This is due to radiation-induced somatic mutations creating the "
            "second hit in PTCH1-haploinsufficient cells simultaneously across a wide field. "
            "Radiation to ANY site → BCCs in that radiation field. "
            "For medulloblastoma in Gorlin (childhood, peak age 2-3, desmoplastic subtype) → "
            "chemotherapy-ONLY protocols (carboplatin + vincristine + cyclophosphamide) — "
            "NO craniospinal radiation; alternative to RT MANDATORY. "
            "For BCC: no radiotherapy even for locally advanced BCC — vismodegib or sonidegib instead. "
            "PATHOGNOMONIC FEATURES: "
            "Calcified falx cerebri on X-ray/CT: present in 85-90% from teens; "
            "X-ray skull AP: lamellar/sheet calcification of falx — highly specific for Gorlin. "
            "OKCs (odontogenic keratocysts / keratocystic odontogenic tumors): "
            "70-80%; beginning in teens; jaw cysts on OPG; recur after curettage → need enucleation. "
            "VISMODEGIB/SONIDEGIB (Hedgehog pathway inhibitors, SMO inhibitors): "
            "FDA for locally advanced/metastatic BCCs; effective in Gorlin; teratogenic — "
            "mandatory pregnancy prevention; muscle cramps, ageusia (taste loss), hair loss common; "
            "drug holiday often needed for adverse effects. "
            "UV protection: lifelong rigorous sun avoidance mandatory; arsenic exposure absolute avoid."
        ),
        "inheritance": "Autosomal Dominant (AD); 25% de novo; two-hit tumor suppressor; SMO inhibitor pathway as therapeutic target",
        "hallmark": "Multiple BCCs (hundreds, especially UV/radiation fields) + OKCs (jaw, teens onset) + calcified falx cerebri (PATHOGNOMONIC on X-ray, 85%) + bifid ribs; RADIATION ABSOLUTELY CI → field cancerization → hundreds of BCCs",
        "key_ddx": "Sporadic BCC (single, not multiple, no OKC, no jaw cysts), Rombo syndrome, XP/xeroderma pigmentosum (UV-sensitive with ALL skin cancers, not just BCC), Bazex syndrome (follicular atrophoderma, BCCs), multiple self-healing squamous epitheliomata (TGFBR1 gene)",
        "treatment_alert": "RADIATION ABSOLUTELY CONTRAINDICATED in Gorlin — even for medulloblastoma; chemotherapy-only medulloblastoma protocol; vismodegib/sonidegib for BCC; teratogenic — contraception mandatory; OKC — surgical enucleation; jaw OPG from age 8",
        "seed": 1140,
        "cohort_n": 40,
        "epilepsy_rate": 0.03,
        "cns_tumor_rate": 0.05,  # medulloblastoma childhood
        "mpnst_rate": 0.00,
        "learning_disability_rate": 0.05,
        "plexiform_rate": 0.00,
        "cafeau_lait_rate": 0.00,
        "de_novo_rate": 0.25,
        "drug_error_rate": 0.12,  # radiation error is critical
        "severity_weights": {"Mild": 0.30, "Moderate": 0.45, "Severe": 0.25},
        "skin_lesion_type": "Multiple basal cell carcinomas (BCC) + jaw OKCs + milia",
        "primary_malignancy": "Basal cell carcinoma",
        "bcc_rate": 0.88,
        "okc_rate": 0.80,
        "calcified_falx_rate": 0.85,
        "bifid_ribs_rate": 0.60,
        "radiation_error_rate": 0.12,
    },

    # ── SMARCB1 — Rhabdoid Tumor Predisposition / Schwannomatosis 2 ────────────
    {
        "gene": "SMARCB1",
        "protein": "INI1 / BAF47 / SNF5 (SMARCB1)",
        "alias": (
            "SMARCB1; OMIM gene 601607; 22q11.23; ~385 aa; Rhabdoid tumor predisposition syndrome 1 "
            "(RTPS1, OMIM #609322) / Schwannomatosis 2 (OMIM #615670); AD 25% de novo; rare; "
            "INI1/BAF47 = core SWI/SNF chromatin remodeling complex subunit; LOF → chromatin inaccessibility "
            "→ tumor suppressor gene silencing → AT/RT + rhabdoid tumors; "
            "INI1 IHC PATHOGNOMONIC: complete loss of nuclear INI1 staining; "
            "germline → sibling surveillance MANDATORY"
        ),
        "aa": "~385 aa",
        "kDa": "~44 kDa",
        "gene_class": "Core SWI/SNF (BAF) chromatin remodeling complex subunit; regulates chromatin accessibility and transcription of tumor suppressor genes including CDKN2A (p16) and CDK4/6 pathway",
        "locus": "22q11.23",
        "omim_gene": 601607,
        "omim_disease": 609322,
        "phenotype": "SMARCB1-associated tumors — AT/RT (atypical teratoid/rhabdoid tumor, CNS pediatric) + extrarenal/renal rhabdoid tumors + schwannomatosis (adult peripheral schwannomas) + chronic pain",
        "disease": (
            "SMARCB1 encodes INI1 (also called BAF47, SNF5, hSNF5), a core structural subunit of "
            "the mammalian SWI/SNF (BAF — BRG1/BRM-associated factor) chromatin remodeling complex. "
            "NORMAL FUNCTION: The SWI/SNF complex uses ATP hydrolysis to reposition, eject, or "
            "restructure nucleosomes, creating chromatin accessibility at gene promoters. INI1/BAF47 "
            "is essential for complex stability and for directing the SWI/SNF complex to "
            "target loci. Key tumor suppressor targets of INI1-mediated SWI/SNF activity include: "
            "CDKN2A (p16/p14ARF) — cyclin-dependent kinase inhibitor activating RB pathway and p53; "
            "CDK4/6 → RB phosphorylation (cell cycle arrest); cyclin D — proliferation regulator. "
            "PATHOMECHANISM: LOF variants in SMARCB1 → absent INI1 → SWI/SNF complex destabilisation "
            "→ failure to activate CDKN2A/p16 and other tumor suppressor genes → RB pathway "
            "inactivation → unrestrained CDK4/6 activity → cell cycle progression → "
            "rhabdoid tumour formation. Rhabdoid tumours are among the most genetically simple cancers: "
            "biallelic SMARCB1 LOF (germline + somatic second hit) with NO other recurrent mutations "
            "in most AT/RTs. "
            "PATHOGNOMONIC IHC: Complete loss of INI1 nuclear staining by immunohistochemistry "
            "(using anti-INI1/BAF47 antibody) — 100% specific for SMARCB1-deficient tumors. "
            "Importantly, ALL other cells in the same section (endothelium, stroma, neurons, "
            "inflammatory cells) retain INI1 nuclear staining → provides internal positive control "
            "confirming antibody worked. INI1 IHC is MANDATORY in ALL infant CNS tumors. "
            "AT/RT (Atypical Teratoid/Rhabdoid Tumor): most common malignant CNS tumor under 3 years; "
            "histologically resembles embryonal tumors (ependymoma, PNET) but INI1 loss distinguishes it; "
            "prognosis poor (median survival 12-20 months); intensive chemotherapy + focal RT (non-germline). "
            "RHABDOID TUMOR PREDISPOSITION SYNDROME (RTPS1): germline SMARCB1 LOF → "
            "child diagnosed with AT/RT or rhabdoid tumor → siblings at 50% risk → "
            "URGENT brain MRI + abdominal/pelvic US every 3-4 months from birth until age 5 in siblings. "
            "SCHWANNOMATOSIS 2 (adult phenotype): multiple painful peripheral schwannomas WITHOUT "
            "bilateral vestibular schwannomas (distinguishes from NF2); chronic neuropathic pain "
            "(70%) often precedes schwannoma diagnosis; NCS/EMG defines affected nerves; "
            "nerve-sparing surgery; pain management (gabapentin, pregabalin, TCAs); no specific "
            "systemic therapy approved. "
            "EZH2 inhibitors (tazemetostat): FDA 2020 for INI1-negative epithelioid sarcoma; "
            "clinical trials for SMARCB1-deficient AT/RT and other INI1-null tumors."
        ),
        "inheritance": "Autosomal Dominant (AD); 25% de novo; two-hit tumor suppressor; germline in RTPS1/schwannomatosis; LZTR1 is the other schwannomatosis gene (Schwannomatosis 1)",
        "hallmark": "AT/RT in infants/young children (most common malignant CNS tumor <3y) + rhabdoid tumors + adult schwannomatosis (multiple peripheral schwannomas + chronic pain); INI1 IHC PATHOGNOMONIC (complete nuclear loss); germline → sibling surveillance MANDATORY",
        "key_ddx": "NF2 (bilateral VS, meningiomas, no AT/RT history, no germline INI1 loss — though NF2 is on chr22q12 vs SMARCB1 on 22q11), SMARCA4-rhabdoid tumors (SMARCA4 IHC uses different antibody), medulloblastoma (INI1 retained), ependymoma (INI1 retained), LZTR1-schwannomatosis (Schwannomatosis 1, chr22q11.21)",
        "treatment_alert": "INI1 IHC MANDATORY in ALL infant CNS tumors — complete loss = AT/RT; germline SMARCB1 → sibling brain MRI + abdominal US surveillance from birth; schwannomatosis — pain management; EZH2 inhibitor trials for INI1-null tumors",
        "seed": 1141,
        "cohort_n": 40,
        "epilepsy_rate": 0.15,
        "cns_tumor_rate": 0.40,  # AT/RT
        "mpnst_rate": 0.00,
        "learning_disability_rate": 0.20,
        "plexiform_rate": 0.00,
        "cafeau_lait_rate": 0.00,
        "de_novo_rate": 0.25,
        "drug_error_rate": 0.10,
        "severity_weights": {"Mild": 0.10, "Moderate": 0.35, "Severe": 0.55},
        "skin_lesion_type": "None (no pathognomonic skin lesion; peripheral schwannomas subcutaneous in schwannomatosis)",
        "primary_malignancy": "AT/RT / Rhabdoid tumor",
        "atrt_rate": 0.40,
        "schwannomatosis_rate": 0.35,
        "chronic_pain_rate": 0.70,
    },
]


# ── Cohort generation ──────────────────────────────────────────────────────────

def _gen_patients_for_gene(gene_data: dict, seed: int) -> list:
    """Generate 40 deterministic synthetic patients for one gene."""
    rng = random.Random(seed)
    patients = []
    for i in range(gene_data["cohort_n"]):
        # Severity
        sev_choices = list(gene_data["severity_weights"].keys())
        sev_weights = list(gene_data["severity_weights"].values())
        sev = rng.choices(sev_choices, weights=sev_weights, k=1)[0]

        sex = rng.choice(["M", "F"])

        # Clinical features
        epilepsy = rng.random() < gene_data["epilepsy_rate"]
        cns_tumor = rng.random() < gene_data["cns_tumor_rate"]
        mpnst = rng.random() < gene_data["mpnst_rate"]
        learning_disability = rng.random() < gene_data["learning_disability_rate"]
        malignancy = None
        if mpnst and gene_data.get("primary_malignancy") == "MPNST":
            malignancy = "MPNST"
        elif gene_data.get("primary_malignancy") == "Clear cell RCC" and rng.random() < gene_data.get("rcc_rate", 0):
            malignancy = "Clear cell RCC"
        elif gene_data.get("primary_malignancy") == "Basal cell carcinoma" and rng.random() < gene_data.get("bcc_rate", 0):
            malignancy = "BCC"
        elif gene_data.get("primary_malignancy") == "AT/RT / Rhabdoid tumor" and rng.random() < gene_data.get("atrt_rate", 0):
            malignancy = "AT/RT"
        elif gene_data.get("primary_malignancy") in ("Breast cancer / Thyroid cancer",):
            if rng.random() < gene_data.get("breast_cancer_rate", 0) + gene_data.get("thyroid_cancer_rate", 0) * 0.3:
                malignancy = "Breast/Thyroid cancer"

        de_novo_variant = rng.random() < gene_data["de_novo_rate"]
        drug_error = rng.random() < gene_data["drug_error_rate"]

        # Age at diagnosis
        age_at_dx_years = round(rng.uniform(0.5, 45.0), 1)

        # Radiation error (PTCH1 only)
        radiation_error = False
        if gene_data["gene"] == "PTCH1":
            radiation_error = rng.random() < gene_data.get("radiation_error_rate", 0)

        # mTOR treatment (TSC1/TSC2 if SEGA or renal AML)
        mtor_treatment = False
        if gene_data["gene"] in ("TSC1", "TSC2"):
            sega = rng.random() < gene_data.get("sega_rate", 0)
            aml = rng.random() < gene_data.get("renal_aml_rate", 0)
            mtor_treatment = sega or aml

        surveillance_adherent = rng.random() < 0.72  # general surveillance adherence

        patients.append({
            "patient_id": f"{gene_data['gene']}-{i+1:03d}",
            "gene": gene_data["gene"],
            "seed": seed,
            "severity": sev,
            "sex": sex,
            "epilepsy": epilepsy,
            "cns_tumor": "Present" if cns_tumor else None,
            "skin_lesion_type": gene_data["skin_lesion_type"],
            "learning_disability": learning_disability,
            "malignancy": malignancy,
            "de_novo_variant": de_novo_variant,
            "drug_error": drug_error,
            "age_at_dx_years": age_at_dx_years,
            "surveillance_adherent": surveillance_adherent,
            "radiation_error": radiation_error,
            "mtor_treatment": mtor_treatment,
        })
    return patients


def _gen_cohort() -> list:
    """Generate all 320 patients (8 genes × 40) deterministically."""
    all_pts = []
    for idx, gd in enumerate(NEUROCUTANEOUS_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients_for_gene(gd, seed))
    return all_pts


# ── API functions ──────────────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    # Severity breakdown
    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] = sev.get(p["severity"], 0) + 1

    # Aggregate clinical statistics
    epilepsy_n = sum(1 for p in patients if p["epilepsy"])
    cns_tumor_n = sum(1 for p in patients if p["cns_tumor"])
    malignancy_n = sum(1 for p in patients if p["malignancy"])
    learning_disability_n = sum(1 for p in patients if p["learning_disability"])
    de_novo_n = sum(1 for p in patients if p["de_novo_variant"])
    drug_error_n = sum(1 for p in patients if p["drug_error"])
    radiation_error_n = sum(1 for p in patients if p["radiation_error"])
    mtor_n = sum(1 for p in patients if p["mtor_treatment"])
    surveillance_n = sum(1 for p in patients if p["surveillance_adherent"])

    # Per-gene stats
    gene_stats = {}
    for gd in NEUROCUTANEOUS_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_stats[gd["gene"]] = {
            "epilepsy_pct": round(100 * sum(1 for p in gpts if p["epilepsy"]) / len(gpts), 1),
            "cns_tumor_pct": round(100 * sum(1 for p in gpts if p["cns_tumor"]) / len(gpts), 1),
            "malignancy_pct": round(100 * sum(1 for p in gpts if p["malignancy"]) / len(gpts), 1),
            "drug_error_pct": round(100 * sum(1 for p in gpts if p["drug_error"]) / len(gpts), 1),
        }

    disease_cat = {
        "Neurofibromatosis Type 1 (NF1)": round(100 * 40 / n, 1),
        "Neurofibromatosis Type 2 (NF2)": round(100 * 40 / n, 1),
        "Tuberous Sclerosis Complex Type 1 (TSC1)": round(100 * 40 / n, 1),
        "Tuberous Sclerosis Complex Type 2 (TSC2)": round(100 * 40 / n, 1),
        "Von Hippel-Lindau Syndrome (VHL)": round(100 * 40 / n, 1),
        "PTEN Hamartoma Tumor Syndrome (PTEN/Cowden)": round(100 * 40 / n, 1),
        "Gorlin Syndrome/NBCCS (PTCH1)": round(100 * 40 / n, 1),
        "SMARCB1/INI1 Rhabdoid Tumor Predisposition": round(100 * 40 / n, 1),
    }

    kpis = [
        {"label": "Total Patients", "value": str(n)},
        {"label": "Genes Covered", "value": "8"},
        {"label": "Epilepsy", "value": f"{round(100*epilepsy_n/n,1)}%"},
        {"label": "CNS Tumor", "value": f"{round(100*cns_tumor_n/n,1)}%"},
        {"label": "Malignancy", "value": f"{round(100*malignancy_n/n,1)}%"},
        {"label": "Learning Disability", "value": f"{round(100*learning_disability_n/n,1)}%"},
        {"label": "De Novo Variant", "value": f"{round(100*de_novo_n/n,1)}%"},
        {"label": "Drug Error", "value": f"{round(100*drug_error_n/n,1)}%"},
    ]

    return {
        "atlas_name": "Neurocutaneous-Atlas",
        "atlas_subtitle": (
            "NF1·NF2·TSC1·TSC2·VHL·PTEN·PTCH1·SMARCB1 — "
            "320 patients (8×40, seeds 1134–1141)"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1134–1141",
        "description": (
            "Comprehensive atlas of 8 major neurocutaneous tumor predisposition syndromes: "
            "NF1/Neurofibromatosis-1 (AD; RasGAP; café-au-lait + plexiform NF + MPNST risk 10%; "
            "selumetinib FDA 2020; AVOID radiation — MPNST induction risk; learning disability 80%); "
            "NF2 (AD; ERM tumor suppressor merlin; bilateral vestibular schwannomas PATHOGNOMONIC; "
            "bevacizumab for growing VS; annual audiological surveillance); "
            "TSC1 (AD; hamartin-mTOR suppressor; epilepsy 90%; infantile spasms — vigabatrin first-line; "
            "SEGA → everolimus; renal AML; less severe than TSC2); "
            "TSC2 (AD; tuberin RhebGAP; more severe than TSC1; epilepsy 92%; LAM in women; "
            "renal AML 70% → embolize if >3cm; everolimus FDA 2017); "
            "VHL (AD; E3-ligase HIF ubiquitination; hemangioblastoma + RCC + pheochromocytoma + ELST; "
            "belzutifan FDA 2021; annual MRI surveillance mandatory); "
            "PTEN/Cowden (AD; PIP3→PIP2 phosphatase; macrocephaly + trichilemmoma PATHOGNOMONIC; "
            "breast/thyroid/uterine cancer surveillance mandatory); "
            "PTCH1/Gorlin (AD; Hedgehog/SMO regulator; BCC hundreds + OKC + calcified falx; "
            "RADIATION ABSOLUTELY CONTRAINDICATED — field cancerization → hundreds BCCs; vismodegib); "
            "SMARCB1/INI1 (AD; SWI/SNF chromatin remodeler; AT/RT in infants + schwannomatosis adults; "
            "INI1 IHC PATHOGNOMONIC — complete nuclear loss; germline → sibling surveillance MANDATORY). "
            "Critical rules: PTCH1 radiation CI; TSC vigabatrin visual monitoring; VHL annual MRI; "
            "PTEN cancer surveillance; SMARCB1 INI1 IHC mandatory in infant CNS tumors."
        ),
        "aggregate_clinical": {
            "epilepsy_pct": round(100 * epilepsy_n / n, 1),
            "cns_tumor_pct": round(100 * cns_tumor_n / n, 1),
            "malignancy_pct": round(100 * malignancy_n / n, 1),
            "skin_lesion_pct": round(100 * sum(1 for p in patients if p["skin_lesion_type"] and "None" not in p["skin_lesion_type"]) / n, 1),
            "learning_disability_pct": round(100 * learning_disability_n / n, 1),
            "de_novo_pct": round(100 * de_novo_n / n, 1),
            "drug_error_pct": round(100 * drug_error_n / n, 1),
            "radiation_error_pct": round(100 * radiation_error_n / n, 1),
            "mtor_treatment_pct": round(100 * mtor_n / n, 1),
            "surveillance_adherent_pct": round(100 * surveillance_n / n, 1),
        },
        "drug_alerts": [
            {
                "type": "danger",
                "title": "PTCH1/GORLIN: RADIATION THERAPY ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "Ionising radiation in Gorlin syndrome (PTCH1 germline LOF) triggers field "
                    "cancerization: radiation-induced somatic mutations create simultaneous second "
                    "hits in PTCH1-haploinsufficient cells across the entire radiation field → "
                    "hundreds of basal cell carcinomas develop within 1-10 years in the irradiated "
                    "area. For medulloblastoma in Gorlin (desmoplastic subtype, peak age 2-3): "
                    "CHEMOTHERAPY-ONLY protocols (NO craniospinal radiation) — mandatory. "
                    "Vismodegib/sonidegib (Hedgehog inhibitors, SMO inhibitors) are the systemic "
                    "alternative for locally advanced/metastatic BCCs. Teratogenic — contraception mandatory."
                ),
            },
            {
                "type": "danger",
                "title": "TSC1/TSC2: VIGABATRIN VISUAL FIELD MONITORING MANDATORY",
                "body": (
                    "Vigabatrin (first-line for infantile spasms in TSC) causes irreversible peripheral "
                    "retinal toxicity (visual field constriction) in 30-50% of long-term users. "
                    "Monitoring: formal visual field testing (Goldmann/Humphrey perimetry + ERG) "
                    "every 6-12 months from initiation. Risk does NOT preclude vigabatrin use in TSC "
                    "(benefit > risk for infantile spasms) but monitoring is non-negotiable. "
                    "Failure to monitor = medico-legal liability. "
                    "Everolimus (mTOR inhibitor) reduces TSC seizure frequency but does NOT replace "
                    "vigabatrin for acute infantile spasms management."
                ),
            },
            {
                "type": "warning",
                "title": "NF1: RADIATION AVOID — MPNST INDUCTION IN RADIATION FIELD",
                "body": (
                    "NF1 patients receiving radiation therapy have significantly elevated risk of "
                    "radiation-induced secondary MPNST within the irradiated field (due to NF1 "
                    "haploinsufficiency of neurofibromin in neural crest cells in the radiation field). "
                    "Radiation should be avoided unless no oncologically equivalent alternative exists. "
                    "Optic pathway gliomas: chemotherapy-first (carboplatin/vincristine); "
                    "MEK inhibitors (selumetinib) emerging for OPG. "
                    "Plexiform neurofibroma: selumetinib (FDA 2020) preferred over surgical debulking "
                    "and radiation."
                ),
            },
            {
                "type": "warning",
                "title": "VHL: ANNUAL MRI SURVEILLANCE MANDATORY — Missing Surveillance = Preventable Death",
                "body": (
                    "VHL surveillance protocol: Annual MRI brain/spine (hemangioblastoma) + "
                    "abdominal MRI/CT (RCC + pheo + pancreatic NET/cysts) + "
                    "annual ophthalmology (retinal angioma + ELST) + "
                    "annual 24h urine catecholamines/metanephrines + blood pressure. "
                    "Belzutifan (HIF-2α inhibitor, FDA 2021): oral once-daily for VHL-RCC, "
                    "hemangioblastoma, and pNET — first systemic targeted therapy. "
                    "ELST: baseline CT temporal bone mandatory; early resection preserves hearing. "
                    "Pheochromocytoma: alpha-blockade (phenoxybenzamine) BEFORE surgical resection — "
                    "failure to alpha-block = hypertensive crisis."
                ),
            },
            {
                "type": "warning",
                "title": "SMARCB1/INI1 LOSS: SIBLING SURVEILLANCE MANDATORY FOR GERMLINE RTPS1",
                "body": (
                    "When a child (typically <3 years) is diagnosed with AT/RT: "
                    "INI1 IHC MANDATORY on tumor specimen — complete loss = SMARCB1-deficient tumor. "
                    "Germline SMARCB1 testing of the child and parents. "
                    "If germline SMARCB1 LOF confirmed: siblings at 50% risk → "
                    "brain MRI + abdominal/pelvic ultrasound every 3-4 months from birth to age 5 mandatory. "
                    "Failure to perform sibling surveillance = preventable childhood mortality from "
                    "undetected AT/RT or rhabdoid tumors. "
                    "Adult schwannomatosis: chronic neuropathic pain often precedes tumor detection; "
                    "pain management essential (gabapentin, pregabalin, TCA); NCS/EMG to map nerve involvement."
                ),
            },
        ],
        "critical_rules": [
            "PTCH1 RADIATION ABSOLUTELY CI: alternatives mandatory even for medulloblastoma — chemotherapy-only protocol",
            "TSC1/TSC2 infantile spasms: vigabatrin first-line; visual field monitoring MANDATORY every 6-12 months",
            "TSC2 renal AML >3cm: embolize (or everolimus) BEFORE Wunderlich hemorrhage — do not wait",
            "NF1 plexiform NF: selumetinib (MEK inhibitor) FDA 2020 for inoperable/symptomatic; avoid radiation",
            "NF2 bilateral VS: bevacizumab for growing schwannomas; annual audiological surveillance from diagnosis",
            "VHL: belzutifan FDA 2021; annual MRI mandatory; ELST → CT temporal bone at diagnosis; alpha-block before pheo surgery",
            "PTEN/Cowden: annual breast MRI + thyroid US + colonoscopy 5y + endometrial sampling from age 30-35",
            "SMARCB1: INI1 IHC mandatory in ALL infant CNS tumors; complete nuclear loss = AT/RT; germline → sibling surveillance",
        ],
        "pathway_targets": {
            "NF1_NF2": "RAS/MAPK + Hippo pathway — MEK inhibitors (selumetinib, trametinib); bevacizumab (NF2)",
            "TSC1_TSC2": "mTORC1 pathway — everolimus, sirolimus (mTOR inhibitors) — FDA approved multiple indications",
            "VHL": "HIF-2α pathway — belzutifan (HIF-2α inhibitor, FDA 2021)",
            "PTEN": "PI3K/AKT/mTOR pathway — mTOR inhibitors in trials; PI3K inhibitors in development",
            "PTCH1": "Hedgehog/SMO pathway — vismodegib, sonidegib (SMO inhibitors)",
            "SMARCB1": "SWI/SNF/EZH2 pathway — tazemetostat (EZH2 inhibitor, FDA 2020 for epithelioid sarcoma); CDK4/6 inhibitors",
        },
        "severity": sev,
        "disease_category_breakdown": disease_cat,
        "gene_stats": gene_stats,
        "kpis": kpis,
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    genes_out = []
    for gd in NEUROCUTANEOUS_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        n = len(gpts)

        epilepsy_pct = round(100 * sum(1 for p in gpts if p["epilepsy"]) / n, 1)
        cns_tumor_pct = round(100 * sum(1 for p in gpts if p["cns_tumor"]) / n, 1)
        malignancy_pct = round(100 * sum(1 for p in gpts if p["malignancy"]) / n, 1)
        learning_disability_pct = round(100 * sum(1 for p in gpts if p["learning_disability"]) / n, 1)
        de_novo_pct = round(100 * sum(1 for p in gpts if p["de_novo_variant"]) / n, 1)
        drug_error_pct = round(100 * sum(1 for p in gpts if p["drug_error"]) / n, 1)
        radiation_error_pct = round(100 * sum(1 for p in gpts if p["radiation_error"]) / n, 1)
        mtor_pct = round(100 * sum(1 for p in gpts if p["mtor_treatment"]) / n, 1)
        surveillance_pct = round(100 * sum(1 for p in gpts if p["surveillance_adherent"]) / n, 1)
        mean_age_dx = round(sum(p["age_at_dx_years"] for p in gpts) / n, 1)

        genes_out.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "gene_class": gd["gene_class"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "phenotype": gd["phenotype"],
            "disease": gd["disease"],
            "inheritance": gd["inheritance"],
            "hallmark": gd["hallmark"],
            "key_ddx": gd["key_ddx"],
            "treatment_alert": gd["treatment_alert"],
            "skin_lesion_type": gd["skin_lesion_type"],
            "primary_malignancy": gd["primary_malignancy"],
            "seed": gd["seed"],
            "cohort_n": n,
            "mean_age_at_dx_years": mean_age_dx,
            "epilepsy_pct": epilepsy_pct,
            "cns_tumor_pct": cns_tumor_pct,
            "malignancy_pct": malignancy_pct,
            "learning_disability_pct": learning_disability_pct,
            "de_novo_pct": de_novo_pct,
            "drug_error_pct": drug_error_pct,
            "radiation_error_pct": radiation_error_pct,
            "mtor_treatment_pct": mtor_pct,
            "surveillance_adherent_pct": surveillance_pct,
            "severity_weights": gd["severity_weights"],
        })

    return {"genes": genes_out}


def get_definitions() -> list:
    return [
        {
            "term": "Neurocutaneous Syndromes (Phakomatoses)",
            "definition": (
                "A group of hereditary disorders characterised by developmental abnormalities "
                "affecting both the nervous system (brain, spinal cord, peripheral nerves) and "
                "the skin (ectoderm-derived tissues), typically involving tumor predisposition. "
                "Also called phakomatoses (from Greek 'phakos' = lens-shaped spot). "
                "The major syndromes share: (1) germline LOF in tumor suppressor genes; "
                "(2) AD inheritance (mostly); (3) hamartoma/tumor formation in multiple tissues; "
                "(4) characteristic cutaneous findings enabling clinical diagnosis. "
                "Major syndromes: NF1 (RAS-MAPK), NF2 (Hippo), TSC1/2 (mTOR), "
                "VHL (HIF/VEGF), PTEN/Cowden (PI3K-AKT-mTOR), Gorlin/PTCH1 (Hedgehog), "
                "SMARCB1/AT/RT (SWI/SNF). Each represents a targetable signalling pathway."
            ),
        },
        {
            "term": "mTORC1 (mechanistic Target of Rapamycin Complex 1)",
            "definition": (
                "A serine-threonine kinase complex (mTOR + Raptor + mLST8 + PRAS40 + DEPTOR) "
                "that is the central integrator of growth factor, nutrient, energy, and oxygen "
                "signals controlling cell growth and protein synthesis. "
                "mTORC1 is activated by: Rheb-GTP (TSC1/TSC2 LOF → Rheb stays GTP-bound), "
                "amino acids (RAG GTPases), PI3K/AKT (RTK signalling via PTEN LOF). "
                "Activated mTORC1 phosphorylates: S6K1 (protein synthesis activation), "
                "4E-BP1 (cap-dependent translation initiation), ULK1 (autophagy suppression). "
                "In TSC1/TSC2 LOF: mTORC1 constitutively active → hamartoma formation in brain "
                "(cortical tubers, SEGA), kidney (AML), lung (LAM), skin (angiofibromas), heart (rhabdomyoma). "
                "mTOR inhibitors: everolimus (FDA for TSC seizures, SEGA, AML, and multiple other indications); "
                "sirolimus/rapamycin (for TSC-LAM); rapalogues = allosteric mTORC1 inhibitors."
            ),
        },
        {
            "term": "MPNST (Malignant Peripheral Nerve Sheath Tumor)",
            "definition": (
                "Aggressive malignant soft tissue sarcoma arising from nerve sheaths, "
                "representing the major cause of NF1-related mortality. "
                "Prevalence: ~10% NF1 lifetime risk (vs <0.001% general population). "
                "Most arise from pre-existing plexiform neurofibromas (PNF) — malignant transformation "
                "requires biallelic NF1 LOF + additional somatic alterations (CDKN2A, PTEN, RAS pathway). "
                "Red flags for PNF → MPNST transformation: "
                "(1) Rapid volume increase (>20% in 12 months); "
                "(2) New constant pain in PNF; "
                "(3) New neurological deficit; "
                "(4) FDG-PET SUVmax >4 (high sensitivity for MPNST). "
                "Treatment: surgery (wide excision) ± chemotherapy; prognosis poor (5-year OS ~20%). "
                "Selumetinib: reduces PNF volume (prevents MPNST precursor progression); "
                "NOT established for MPNST treatment. "
                "Radiation: increases MPNST risk in NF1 → AVOID unless no alternative."
            ),
        },
        {
            "term": "Infantile Spasms (West Syndrome) in TSC",
            "definition": (
                "Epileptic spasms in infants (typically <12 months) presenting as sudden flexion "
                "or extension of trunk and limbs, in clusters (series). "
                "In TSC: 50% of patients develop infantile spasms; EEG shows hypsarrhythmia "
                "(chaotic high-voltage slow waves + multifocal spikes). "
                "TSC-associated infantile spasms have a SPECIFIC treatment advantage: "
                "Vigabatrin (GABA transaminase inhibitor) achieves spasm freedom in 95% of TSC "
                "infants (vs ~50% in non-TSC IS). Vigabatrin is FIRST-LINE for TSC-IS. "
                "Visual field monitoring mandatory: vigabatrin causes irreversible peripheral retinal "
                "toxicity (visual field constriction) in 30-50% of long-term users → "
                "formal visual fields every 6-12 months. "
                "Everolimus (mTOR inhibitor): FDA 2017 for drug-resistant TSC partial seizures; "
                "reduces seizure frequency 50%+ vs placebo; adjunctive therapy."
            ),
        },
        {
            "term": "SEGA (Subependymal Giant Cell Astrocytoma)",
            "definition": (
                "Benign WHO grade 1 tumor arising at the foramen of Monro from subependymal nodules "
                "in tuberous sclerosis complex. "
                "Prevalence: ~12% TSC1, ~17% TSC2; typically diagnosed in childhood/adolescence. "
                "Clinical consequence: SEGA grows → obstructs foramen of Monro → "
                "obstructive hydrocephalus → raised intracranial pressure → "
                "papilloedema, headache, vomiting, visual loss, herniation (if untreated). "
                "Treatment: Everolimus (mTOR inhibitor) FDA 2012 for TSC-SEGA — shrinks SEGA "
                "volume by 50%+ in majority of patients; avoids surgical risks. "
                "Surgical resection: for large/symptomatic SEGA causing acute hydrocephalus. "
                "Surveillance: serial MRI brain every 1-3 years (more frequent in children/growth spurts)."
            ),
        },
        {
            "term": "VHL Disease Classification (Type 1, 2A, 2B, 2C)",
            "definition": (
                "Genotype-phenotype classification of Von Hippel-Lindau syndrome based on the "
                "type of VHL mutation and its impact on HIF-α binding vs overall protein function: "
                "Type 1 (truncating/large deletion): hemangioblastoma + RCC; LOW pheochromocytoma risk. "
                "Type 2A (missense, affects surface charge): pheo + hemangioblastoma; LOW RCC risk. "
                "Type 2B (missense, affects hydrophobic core — complete HIF function lost): "
                "pheo + RCC + hemangioblastoma — HIGHEST malignant risk. "
                "Type 2C (missense, affects VDEP surface — HIF binding preserved): pheo ONLY; "
                "no hemangioblastoma, no RCC. "
                "Mechanism: Type 2 mutations affect specific pVHL structural domains; "
                "Type 2C mutations preserve HIF-α ubiquitination but impair pVHL's HIF-independent "
                "functions → pheochromocytoma without RCC/HB. "
                "Belzutifan (HIF-2α inhibitor, FDA 2021) is particularly effective for Type 1 and 2B VHL."
            ),
        },
        {
            "term": "Trichilemmoma (Cowden Syndrome / PTEN)",
            "definition": (
                "Benign follicular infundibular hamartoma (skin appendage tumor) arising from the "
                "outer root sheath of the hair follicle. "
                "PATHOGNOMONIC for PTEN Hamartoma Tumor Syndrome (PHTS)/Cowden syndrome "
                "when multiple lesions present. "
                "Clinical appearance: multiple small (2-4mm), smooth, skin-colored or slightly "
                "pigmented papules on the face, especially around the nose, mouth, and ears. "
                "Often warty or verrucous. Begin appearing in the second-third decade. "
                "Histology: lobulated proliferation of pale, glycogen-rich cells of outer root sheath "
                "cells with basement membrane periphery. "
                "Differential: verruca vulgaris (HPV), milia, acne, trichoepithelioma. "
                "Multiple trichilemmomas + macrocephaly = Cowden until proven otherwise. "
                "PTEN germline testing mandatory in all suspected Cowden syndrome."
            ),
        },
        {
            "term": "OKC (Odontogenic Keratocyst) in Gorlin Syndrome",
            "definition": (
                "Keratocystic odontogenic tumor (KCOT/OKC) — jaw cyst lined by parakeratinized "
                "stratified squamous epithelium, arising from dental lamina remnants. "
                "In Gorlin syndrome (PTCH1 LOF): multiple OKCs begin in teen years; "
                "usually in the mandible; recurrence rate HIGH after curettage → "
                "enucleation with peripheral ostectomy preferred. "
                "First symptom in many Gorlin patients — OPG (orthopantomogram) "
                "from age 8 years mandatory for Gorlin surveillance. "
                "OKC expansion → root resorption, tooth displacement, pathological fracture. "
                "Vismodegib reduces OKC size (Hedgehog inhibition) in Gorlin — useful "
                "for extensive multi-cystic disease. "
                "2022 WHO reclassification: KCOT is the preferred term (vs OKC/KOT in older literature)."
            ),
        },
        {
            "term": "INI1 IHC (Immunohistochemistry) in AT/RT",
            "definition": (
                "Anti-INI1/BAF47/SMARCB1 immunohistochemistry is the gold-standard diagnostic "
                "test for SMARCB1-deficient tumors (AT/RT, renal/extrarenal rhabdoid tumors, "
                "epithelioid sarcoma, poorly differentiated chordoma). "
                "Result interpretation: "
                "POSITIVE (retained nuclear staining): SMARCB1 protein present → NOT AT/RT. "
                "NEGATIVE (complete loss of nuclear staining): SMARCB1 absent → AT/RT or other "
                "INI1-negative tumor. "
                "INTERNAL POSITIVE CONTROL: stromal cells, endothelium, inflammatory cells in "
                "the same section retain nuclear INI1 staining → confirms antibody worked; "
                "if internal controls are also negative, repeat IHC with fresh antibody. "
                "SENSITIVITY and SPECIFICITY: >99% specific for SMARCB1-deficient tumors "
                "when properly performed. "
                "MANDATORY in ALL infant CNS tumors <3 years (AT/RT is most common malignant CNS "
                "tumor <3y and is frequently misdiagnosed as ependymoma or PNET)."
            ),
        },
        {
            "term": "Belzutifan (PT2977) — HIF-2α Inhibitor for VHL",
            "definition": (
                "Belzutifan (Welireg) is an oral small molecule inhibitor of HIF-2α (EPAS1), "
                "approved by FDA August 2021 for VHL disease-associated: "
                "(1) Renal cell carcinoma (RCC), (2) CNS hemangioblastomas, (3) Pancreatic NETs. "
                "Mechanism: Belzutifan binds the PAS-B domain of HIF-2α, preventing its "
                "dimerisation with HIF-1β (ARNT) → HIF-2α transcriptional activity blocked → "
                "VEGF, EPO, GLUT1, and other HIF-2α target genes suppressed. "
                "Clinical efficacy: In LITESPARK-004 trial (VHL disease): "
                "RCC objective response rate 49%; HB ORR 30%; pNET ORR 91%. "
                "Adverse effects: anaemia (HIF-2α drives EPO → EPO drops when blocked; "
                "transfusion if haemoglobin <8 g/dL); dizziness; fatigue; weight gain. "
                "Teratogenic: contraception mandatory in women of childbearing potential. "
                "MONITORING: haemoglobin monthly first 8 months; oxygen saturation monitoring."
            ),
        },
        {
            "term": "Selumetinib (MEK Inhibitor) for NF1 Plexiform Neurofibromas",
            "definition": (
                "Selumetinib (Koselugo) is an oral MEK1/2 inhibitor (MAPK kinase inhibitor) "
                "approved by FDA April 2020 for neurofibromatosis type 1 (NF1) in pediatric "
                "patients aged ≥2 years with symptomatic, inoperable plexiform neurofibromas. "
                "First and only FDA-approved targeted therapy for NF1. "
                "Mechanism: NF1 LOF → unregulated Ras-GTP → sustained MEK1/2 activation → "
                "ERK1/2 phosphorylation → proliferation. Selumetinib inhibits MEK1/2 → "
                "ERK1/2 suppression → reduced Schwann cell proliferation. "
                "Clinical efficacy: SPRINT Stratum 1 trial: 70% partial response (≥20% volume "
                "reduction); improved pain, function, disfigurement. "
                "Dosing: 25 mg/m² orally twice daily continuously. "
                "Adverse effects: acneiform rash, paronychia, GI (nausea, diarrhoea), "
                "elevated CK, cardiomyopathy (monitor ECHO 8-weekly in first 12 months). "
                "Important: selumetinib is NOT indicated for MPNST treatment (established efficacy "
                "in benign PNF shrinkage only); MPNST requires surgery ± chemotherapy."
            ),
        },
        {
            "term": "Gorlin Syndrome Calcified Falx Cerebri",
            "definition": (
                "Lamellar (sheet-like) calcification of the falx cerebri (interhemispheric dural "
                "fold separating the cerebral hemispheres) visible on plain skull X-ray or CT. "
                "Present in 85-90% of Gorlin syndrome (PTCH1) patients, typically appearing "
                "in the teens or early twenties. "
                "PATHOGNOMONIC for Gorlin syndrome when bilateral/lamellar (vs physiological "
                "calcification which is unilateral, nodular, and occurs later in life). "
                "Identification: skull AP plain X-ray — vertical dense calcification in midline "
                "above corpus callosum; CT brain (non-contrast) shows calcification clearly. "
                "Clinical significance: No direct treatment required; marker for Gorlin diagnosis. "
                "Other calcifications in Gorlin: tentorium cerebelli, choroid plexus (earlier "
                "onset than general population), petroclinoid ligaments. "
                "Diagnostic implication: Any teenager with multiple BCCs + jaw OKCs + "
                "calcified falx on skull X-ray = Gorlin syndrome until proven otherwise."
            ),
        },
        {
            "term": "ELST (Endolymphatic Sac Tumor) in VHL",
            "definition": (
                "Low-grade papillary adenomatous tumor arising from the endolymphatic sac "
                "(in the posterior temporal bone, adjacent to the posterior wall of the petrous bone). "
                "Occurs in ~12% of VHL patients; bilateral in ~30% of VHL ELST cases. "
                "VHL pathomechanism: pVHL LOF in endolymphatic sac epithelium → HIF-2α "
                "accumulation → VEGF-driven vascular tumor locally invasive into temporal bone. "
                "Clinical presentation: unilateral progressive SNHL (first symptom), tinnitus, "
                "aural fullness, Meniere-like attacks, vertigo; ELST is a key cause of hearing "
                "loss in VHL disease. "
                "Diagnosis: CT temporal bone (irregular lytic lesion in posterior petrous bone "
                "with matrix calcification — PATHOGNOMONIC appearance) + MRI (T1-bright tumor "
                "if haemorrhagic). "
                "Management: surgical resection is curative if performed before significant SNHL. "
                "Surveillance: CT/MRI temporal bone at VHL diagnosis and every 1-2 years; "
                "audiometry annually from VHL diagnosis."
            ),
        },
    ]


# ── Module self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=== Neurocutaneous Atlas — Self-test ===")

    overview = get_overview()
    print(f"\nAtlas: {overview['atlas_name']}")
    print(f"Patients: {overview['n_patients']}")
    print(f"Seeds: {overview['seeds']}")
    print(f"Aggregate clinical:")
    for k, v in overview["aggregate_clinical"].items():
        print(f"  {k}: {v}")

    print(f"\nKPIs:")
    for kpi in overview["kpis"]:
        print(f"  {kpi['label']}: {kpi['value']}")

    breakdown = get_breakdown()
    print(f"\nBreakdown — {len(breakdown['genes'])} genes:")
    for g in breakdown["genes"]:
        print(
            f"  {g['gene']:8s} | n={g['cohort_n']} | epilepsy={g['epilepsy_pct']}% "
            f"| CNS_tumor={g['cns_tumor_pct']}% | malignancy={g['malignancy_pct']}% "
            f"| LD={g['learning_disability_pct']}% | de_novo={g['de_novo_pct']}%"
        )

    defs = get_definitions()
    print(f"\nDefinitions: {len(defs)} terms")
    for d in defs:
        print(f"  {d['term']}")

    print("\n=== Self-test PASSED ===")
