#!/usr/bin/env python3
"""Hereditary-Pheo-PGL-Atlas — Complete 8-Gene Hereditary Pheochromocytoma-Paraganglioma Atlas
SDHB   (succinate dehydrogenase iron-sulfur subunit B; 280 aa; 1p36.13; AD;
         SDH complex II subunit B; most aggressive SDH variant — highest malignant PGL rate ~40%;
         extraadrenal, multifocal; DOTATATE-PET mandatory; SDHB IHC loss diagnostic;
         seed SEED_BASE+0) ·
SDHD   (succinate dehydrogenase membrane anchor subunit D; 159 aa; 11q23.1; AD;
         PATERNAL IMPRINTING — maternal transmission = carrier only (no disease);
         predominantly head/neck PGL (parasympathetic, non-secreting);
         bilateral carotid body tumours pathognomonic;
         seed SEED_BASE+1) ·
VHL    (von Hippel-Lindau tumour suppressor; 213 aa; 3p25.3; AD;
         bilateral adrenal pheo (VHL type 2A/2B/2C); secreting adrenal pheo +
         clear cell RCC + cerebellar hemangioblastoma;
         belzutifan FDA-2021 for VHL-related tumours;
         seed SEED_BASE+2) ·
RET    (rearranged during transfection proto-oncogene; 1114 aa; 10q11.21; AD; GOF;
         MEN2A (C634R — medullary thyroid + pheo + primary hyperparathyroidism) and
         MEN2B (M918T — marfanoid + mucosal neuromas + medullary thyroid + pheo);
         prophylactic thyroidectomy age-dependent on variant;
         seed SEED_BASE+3) ·
NF1    (neurofibromin 1; 2818 aa; 17q11.2; AD;
         Neurofibromatosis type 1; café-au-lait macules + Lisch nodules + neurofibromas +
         pheo in 0.1–5.7%; hypertensive crisis presentation;
         NF1 pheos are EPINEPHRINE-secreting (adrenal);
         seed SEED_BASE+4) ·
MAX    (MYC-associated factor X; 151 aa; 14q23.3; AD;
         PATERNAL IMPRINTING (like SDHD); bilateral adrenal pheo, catecholamine-secreting;
         elevated norepinephrine+epinephrine; young onset; no head/neck PGL;
         seed SEED_BASE+5) ·
SDHA   (succinate dehydrogenase flavoprotein subunit A; 664 aa; 5p15.33; AR biallelic
         full expression / AD haploinsufficiency; SDH complex II catalytic subunit;
         SDHA IHC loss = SPECIFIC for SDHA mutation (vs pan-SDH loss in SDHB/C/D);
         pituitary adenoma association UNIQUE among SDH genes; rarest SDH pheo;
         seed SEED_BASE+6) ·
SDHC   (succinate dehydrogenase integral membrane subunit C; 169 aa; 1q23.3; AD;
         head/neck PGL (similar to SDHD but NO imprinting);
         gastric GIST (gastrointestinal stromal tumour) — UNIQUE to SDHC/SDHD among SDH pheo genes;
         SDH-deficient GIST on SDHC IHC;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1686–1693)
"""

import random

SEED_BASE = 1686

PHEO_PGL_GENES = [
    # ── SDHB — SDH Complex II Subunit B — Most Aggressive SDH PGL ───────────
    {
        "gene": "SDHB",
        "protein": "SDHB — SDH4 AD — Iron-Sulfur Cluster Complex II Subunit B — Most Aggressive SDH PGL — 40pct Malignant — DOTATATE-PET Mandatory — SDHB IHC Loss Diagnostic",
        "alias": (
            "SDHB (succinate dehydrogenase iron-sulfur subunit B); OMIM gene 185470; "
            "SDH-related pheochromocytoma-paraganglioma (SDH4 / SDHB-PGL) OMIM 115310. "
            "1p36.13; 280 aa; ~32 kDa; AD heterozygous LOF (second-hit required for tumorigenesis — Knudson two-hit). "
            "FUNCTION: SDHB encodes the iron-sulfur (Fe-S) subunit of mitochondrial succinate dehydrogenase (SDH, Complex II). "
            "SDH catalyses oxidation of succinate → fumarate in the TCA cycle while simultaneously "
            "transferring electrons to ubiquinone (CoQ10) in the respiratory chain. "
            "SDHB contains three Fe-S clusters (2Fe-2S, 4Fe-4S, 3Fe-4S) that form the electron transfer chain "
            "from SDHA (FAD) through SDHB to ubiquinone. "
            "SDH COMPLEX ASSEMBLY: SDHB associates with SDHA to form the catalytic subunit; "
            "SDHC and SDHD anchor the complex to the inner mitochondrial membrane. "
            "If SDHB is absent, the entire SDH complex is destabilised and degraded — "
            "this explains why SDHB IHC loss is DIAGNOSTIC for ANY SDH subunit mutation "
            "(SDHB protein is the most labile and disappears first even in SDHC/SDHD/SDHA mutations). "
            "ONCOGENIC MECHANISM: SDH LOF → succinate accumulates → "
            "succinate competitively inhibits α-ketoglutarate–dependent dioxygenases (TET methylcytosine dioxygenases, PHDs); "
            "PHD inhibition → HIF1α/2α stabilisation (pseudo-hypoxia) → VEGF upregulation → angiogenesis; "
            "TET inhibition → DNA and histone hypermethylation (CpG island methylator phenotype — CIMP) → "
            "silencing of tumour suppressor genes → oncogenesis. "
            "CLINICAL PHENOTYPE — MOST AGGRESSIVE SDH VARIANT: "
            "MALIGNANT PGL RATE ~40%: highest among all SDH subtypes; "
            "malignant defined as metastases to non-chromaffin tissue (lymph nodes, liver, lung, bone); "
            "no single histological feature reliably predicts malignancy — all SDHB tumours must be considered potentially malignant; "
            "TUMOUR LOCATIONS: predominantly extraadrenal paraganglia (abdominal/pelvic retroperitoneal — organ of Zuckerkandl, bladder wall); "
            "adrenal pheo occurs in ~30% of SDHB; head/neck PGL rare (contrast SDHD); "
            "MULTIFOCAL DISEASE: synchronous or metachronous tumours common; "
            "lifelong surveillance mandatory; "
            "BIOCHEMISTRY: plasma normetanephrine elevated (norepinephrine-producing tumours); "
            "plasma metanephrine may also be elevated in adrenal lesions; "
            "3-methoxytyramine elevated if dopaminergic component (rare in SDHB). "
            "IMAGING AND FUNCTIONAL IMAGING: "
            "DOTATATE-PET (68Ga-DOTATATE / 18F-FDOPA-PET) is MANDATORY for staging and surveillance — "
            "superior sensitivity to MIBG for SDH-related PGL; "
            "MIBG scan is INFERIOR for SDHB (many SDHB tumours are MIBG-negative despite functional); "
            "MRI abdomen + neck + chest annually; "
            "whole-body DOTATATE-PET every 2 years for surveillance. "
            "DIAGNOSIS: "
            "SDHB IHC on any pheo/PGL specimen: cytoplasmic staining LOST → SDHB/C/D/A mutation; "
            "SDHA IHC retained (positive) → NOT SDHA mutation → order SDHB/C/D germline; "
            "SDHB germline sequencing + MLPA (deletions ~10% of SDHB mutations). "
            "TREATMENT: "
            "Surgical resection for localised disease; "
            "sunitinib (VEGF pathway): FIRSTMAPPP trial ORR 36% in progressive SDHB malignant PGL; "
            "Lu-177-DOTATATE (PRRT) for DOTATATE-avid metastatic disease; "
            "temozolomide + capecitabine (TEMCAP) for chemotherapy option; "
            "CVD chemotherapy (cyclophosphamide/vincristine/dacarbazine) for rapidly progressive disease. "
            "SURVEILLANCE: annual biochemistry (plasma metanephrines/3-MT) + MRI from age 5–8 years for germline carriers; "
            "cascade testing: all first-degree relatives; biallelic SDHB = extremely rare embryonic lethal. "
            "CASCADE TESTING CRITICAL: SDHB penetrance ~25–50% lifetime; "
            "affected relatives should begin surveillance in childhood (tumours reported from age 5); "
            "prophylactic adrenalectomy NOT recommended for SDHB (extraadrenal predominance)."
        ),
        "locus": "1p36.13",
        "aa": 280,
        "kDa": 32,
        "omim_gene": "185470",
        "omim_disease": "SDH-related Pheochromocytoma-Paraganglioma (SDH4 / SDHB-PGL) OMIM 115310",
        "inheritance": "AD heterozygous LOF — two-hit Knudson mechanism for tumorigenesis; malignant PGL rate ~40%; biallelic germline essentially lethal",
        "gene_class": "Iron-sulfur cluster protein — SDH Complex II electron transfer chain — Fe-S clusters 2Fe-2S/4Fe-4S/3Fe-4S — SDHB IHC loss diagnostic for all SDH subunit mutations — succinate accumulation → pseudo-hypoxia HIF pathway",
        "key_alerts": [
            "SDHB-MALIGNANT-PGL-40PCT-HIGHEST-SDH: SDHB has the highest malignant PGL rate (~40%) of all SDH variants; all SDHB-associated pheo/PGL must be considered potentially malignant regardless of histology; metastases can appear 5–20 years after primary resection — lifelong annual biochemical surveillance is non-negotiable",
            "SDHB-DOTATATE-PET-MANDATORY-MIBG-INFERIOR: DOTATATE-PET (68Ga or 18F-FDOPA) is MANDATORY for staging — many SDHB tumours are MIBG-negative; using MIBG alone for SDHB staging will miss metastatic disease; whole-body DOTATATE-PET at diagnosis and every 2 years for malignant disease",
            "SDHB-IHC-LOSS-DIAGNOSTIC-ALL-SDH: SDHB IHC cytoplasmic loss on any pheo/PGL is diagnostic for SDH subunit mutation (SDHB/C/D/A) — not just SDHB mutation; SDHB protein is the most labile SDH subunit and disappears secondary to loss of any SDH subunit; SDHA IHC must be checked separately (SDHA IHC specific for SDHA mutation)",
            "SDHB-CHILDHOOD-ONSET-SURVEILLANCE-FROM-AGE-5: SDHB tumours can present from age 5; germline carriers should begin annual biochemistry (plasma metanephrines) from age 5–8 and annual MRI from age 8–10; do NOT wait for symptoms — most SDHB tumours are detected on surveillance, not clinical presentation",
            "SDHB-SUNITINIB-FIRSTMAPPP-36PCT-ORR: Progressive metastatic SDHB PGL: sunitinib (anti-VEGF TKI) demonstrated 36% ORR + 83% disease control rate in FIRSTMAPPP trial — currently standard for progressive disease; Lu-177-DOTATATE PRRT for DOTATATE-avid metastatic disease; TEMCAP chemotherapy for rapidly progressive",
        ],
        "etiologies": [
            "SDHB germline heterozygous LOF (missense/nonsense/frameshift) + somatic second-hit (LOH 1p or somatic missense) → absent SDH Complex II activity → succinate accumulation → PHD inhibition → HIF1α/2α stabilisation → pseudo-hypoxia → angiogenic transcriptome → paraganglioma formation",
            "SDHB MLPA deletion (10% of SDHB mutations) — large genomic deletions encompassing one or more SDHB exons; MLPA or aCGH required in SDHB IHC-loss tumours with negative sequencing; germline deletion → same SDH LOF mechanism → same malignant potential as point mutations",
            "SDHB de novo germline mutation — ~10% of SDHB-related PGL arise from de novo mutations not found in parents; de novo rate justifies testing affected individual even when family history is negative; de novo variants carry same malignant risk as inherited variants",
        ],
        "stats": {
            "mean_dx_age": 38,
            "mean_dx_delay_months": 24,
            "malignant_pgl_pct": 40,
            "extraadrenal_tumour_pct": 70,
            "dotatate_avid_pct": 85,
        },
        "dx_delay_distribution": "12–48 months (hypertension often labelled essential; extraadrenal retroperitoneal PGL not palpable; SDHB pheo awareness improves with SDH IHC reflex testing)",
    },

    # ── SDHD — SDH Complex II Membrane Anchor D — Paternal Imprinting ────────
    {
        "gene": "SDHD",
        "protein": "SDHD — SDH3 AD Paternal-Imprinting — Membrane Anchor D — Head/Neck PGL Parasympathetic Non-Secreting — Bilateral Carotid Body Pathognomonic — Maternal-Transmission Carrier-Only",
        "alias": (
            "SDHD (succinate dehydrogenase membrane anchor subunit D); OMIM gene 602690; "
            "Hereditary PGL/Pheo type 1 (PGL1 / SDHD-PGL) OMIM 168000. "
            "11q23.1; 159 aa; ~17 kDa; AD — STRICT PATERNAL IMPRINTING. "
            "FUNCTION: SDHD is the smaller membrane-anchoring subunit of SDH Complex II. "
            "Together with SDHC, SDHD anchors the SDHA-SDHB catalytic core to the inner mitochondrial membrane "
            "and coordinates ubiquinone binding at the Qp site. "
            "SDHD contains histidine residues that coordinate the b-heme, which stabilises the entire complex. "
            "PATERNAL IMPRINTING — CRITICAL CLINICAL PEARL: "
            "SDHD is subject to maternal genomic imprinting (SDHD maternal allele is epigenetically silenced). "
            "DISEASE OCCURS ONLY IF THE MUTATED ALLELE IS INHERITED FROM THE FATHER. "
            "Maternally inherited SDHD mutations → carrier only; no tumour development because the only expressed "
            "(paternal) allele is wild-type. "
            "Clinical consequence: family history on the MATERNAL side does NOT predict disease risk; "
            "a patient with an SDHD mutation inherited from their mother is NOT at increased tumour risk; "
            "cascade testing must account for parent-of-origin: "
            "children of affected SDHD fathers → 50% mutation inheritance, ALL will be at risk; "
            "children of SDHD-carrier mothers → 50% mutation inheritance, NONE at risk. "
            "CLINICAL PHENOTYPE: "
            "PREDOMINANTLY HEAD/NECK PGL (PARASYMPATHETIC GANGLIA): "
            "Carotid body tumour (CBT): pulsatile neck mass at carotid bifurcation — painless, slow-growing; "
            "bilateral CBT: PATHOGNOMONIC for SDHD (bilateral CBT in up to 50% of SDHD carriers with disease); "
            "glomus jugulare: pulsatile tinnitus + hearing loss; "
            "glomus tympanicum: 'red mass behind eardrum' — pulsatile tinnitus; "
            "glomus vagale: vagal neuropathy (hoarseness, dysphagia); "
            "PARASYMPATHETIC PGL = NON-SECRETING: head/neck PGL are usually biochemically silent "
            "(no excess catecholamines); elevated 3-methoxytyramine (3-MT) in ~30% due to dopaminergic secretion; "
            "hypertension NOT typical (contrast SDHB/VHL/RET/NF1 pheo); "
            "MULTIFOCAL: multiple synchronous/metachronous head/neck PGL in 30–50%; "
            "ADRENAL PHEO: in ~20% of SDHD carriers, adrenal pheo occurs — secreting. "
            "BIOCHEMISTRY: "
            "Plasma 3-methoxytyramine (3-MT) elevated → dopaminergic PGL; "
            "24h urine catecholamines often NORMAL or borderline; "
            "normal plasma metanephrines does NOT exclude SDHD PGL (non-secreting). "
            "IMAGING: "
            "MRI neck + base of skull annually; "
            "DOTATATE-PET excellent sensitivity for head/neck PGL; "
            "Colour Doppler ultrasound for carotid body screening. "
            "TREATMENT: "
            "Small stable head/neck PGL: watchful waiting (slow growth <2mm/year); "
            "surgery vs stereotactic radiosurgery (SRS/CyberKnife) for enlarging/symptomatic PGL; "
            "embolisation pre-surgery for vascular CBT; "
            "MALIGNANCY RATE LOW for SDHD head/neck PGL (~5%); much lower than SDHB. "
            "SURVEILLANCE: annual biochemistry (3-MT + chromogranin A) + MRI from age 5–10 for paternally inherited carriers."
        ),
        "locus": "11q23.1",
        "aa": 159,
        "kDa": 17,
        "omim_gene": "602690",
        "omim_disease": "Hereditary Pheochromocytoma-Paraganglioma type 1 (PGL1 / SDHD-PGL) OMIM 168000",
        "inheritance": "AD — STRICT PATERNAL IMPRINTING: only paternally inherited SDHD mutations cause disease; maternally inherited SDHD = carrier only (no tumour risk); maternal allele epigenetically silenced",
        "gene_class": "SDH Complex II small membrane-anchoring subunit — b-heme coordination — ubiquinone Qp-site formation — SDHD with SDHC anchors SDHA-SDHB to inner mitochondrial membrane — parasympathetic head/neck PGL predominance",
        "key_alerts": [
            "SDHD-PATERNAL-IMPRINTING-MANDATORY-PARENT-OF-ORIGIN: SDHD disease ONLY occurs from paternally inherited mutations; a child who inherits the SDHD mutation from their MOTHER is a carrier but has NO increased tumour risk; always establish parent-of-origin before advising on risk — maternal inheritance = reassure; paternal inheritance = full surveillance programme",
            "SDHD-BILATERAL-CAROTID-BODY-PATHOGNOMONIC: Bilateral carotid body tumours are PATHOGNOMONIC for SDHD (and SDHC to lesser extent); any bilateral CBT mandates immediate SDHD (and SDHB/C) germline testing; unilateral CBT also strongly prompts SDH testing — 30–40% of CBTs have hereditary SDH basis",
            "SDHD-HEAD-NECK-PGL-NON-SECRETING-PLASMA-METANEPHRINES-NORMAL: Head/neck PGL are PARASYMPATHETIC and typically NON-SECRETING; normal plasma metanephrines does NOT exclude SDHD PGL; check plasma 3-methoxytyramine (3-MT) and chromogranin A; MRI neck/skull base is the primary surveillance tool",
            "SDHD-LOW-MALIGNANCY-RISK-VS-SDHB: SDHD head/neck PGL malignancy rate ~5% — much lower than SDHB (~40%); management is therefore often watchful waiting for stable small tumours rather than aggressive surgery; patient counselling about low malignant risk for head/neck PGL is important",
            "SDHD-DOTATATE-PET-SURVEILLANCE: Annual MRI neck + skull base for all at-risk SDHD carriers from age 5–10; DOTATATE-PET every 2 years for known tumour surveillance; whole-body DOTATATE if abdominal/adrenal pheo suspected; cascade testing: all children of SDHD-affected FATHER are at risk — all children of SDHD-carrier MOTHER are NOT at risk",
        ],
        "etiologies": [
            "SDHD germline heterozygous missense/truncating LOF (paternally inherited) + somatic second-hit at 11q23 (LOH or somatic mutation) → SDHD absent → SDH Complex II destabilised → succinate accumulation → HIF pseudo-hypoxia → PGL — predominantly parasympathetic head/neck ganglia",
            "SDHD Arg38 hotspot mutations (Arg38Trp, Arg38Gln) — Arg38 is in the ubiquinone-binding Qp site; mutations here disrupt ubiquinone binding → electron transport impaired → succinate accumulates; these missense variants are among the most common SDHD pathogenic variants",
            "SDHD Dutch founder effect (Asp92Tyr — 11q23 haplotype) — large hereditary PGL families in Netherlands with SDHD Asp92Tyr; this variant has been traced to a common Dutch ancestor; imprinting behaviour identical to other SDHD variants — only paternal transmission causes disease",
        ],
        "stats": {
            "mean_dx_age": 40,
            "mean_dx_delay_months": 36,
            "head_neck_pgl_pct": 70,
            "bilateral_cbm_pct": 45,
            "malignant_pgl_pct": 5,
        },
        "dx_delay_distribution": "24–72 months (pulsatile tinnitus often dismissed; bilateral neck mass may be missed; non-secreting PGL means no hypertension to prompt workup)",
    },

    # ── VHL — von Hippel-Lindau — Bilateral Adrenal Pheo + RCC ──────────────
    {
        "gene": "VHL",
        "protein": "VHL — HRCC-VHL1 AD — E3-Ubiquitin-Ligase HIF2a Substrate — Bilateral Adrenal Pheo + ClearCell-RCC + Cerebellar-Hemangioblastoma — Belzutifan-FDA-2021 — 3cm-Active-Surveillance Threshold",
        "alias": (
            "VHL (von Hippel-Lindau tumour suppressor); OMIM gene 608537; "
            "Von Hippel-Lindau syndrome OMIM 193300. "
            "3p25.3; 213 aa; ~24 kDa; AD — two-hit LOF (Knudson). "
            "FUNCTION: pVHL is the substrate recognition component of a CRL2-VHL E3 ubiquitin ligase complex "
            "(pVHL-elongin B-elongin C-cullin 2-RBX1). "
            "Under normoxia: pVHL binds HIF1α and HIF2α that have been hydroxylated by PHD1/2/3 → "
            "ubiquitin-proteasome degradation of HIF → no hypoxia response. "
            "VHL LOF → HIF1α/2α constitutively stable → "
            "activation of HIF target genes: VEGF (angiogenesis), PDGF, EPO, GLUT1, carbonic anhydrase IX (CA-IX). "
            "HIF2α is the dominant oncogenic driver in VHL-related clear cell RCC (HIF2α → cyclin D1 + CCND1). "
            "VHL DISEASE SUBTYPES: "
            "Type 1: truncating/deletion mutations → predominantly RCC + hemangioblastoma; pheo RARE; "
            "Type 2A: missense (Y98H, V84L) → pheo + hemangioblastoma; RCC RARE; "
            "Type 2B: missense (R167W) → pheo + hemangioblastoma + RCC; "
            "Type 2C: missense (E70A) → PHEO ONLY; NO RCC; NO hemangioblastoma; rarest VHL subtype. "
            "CLINICAL MANIFESTATIONS (surveillance targets): "
            "CLEAR CELL RCC: 70% lifetime risk; bilateral/multifocal; 3CM THRESHOLD for surgery "
            "(tumours <3cm monitored by annual MRI — grows slowly; surgery avoids renal loss); "
            "CEREBELLAR HEMANGIOBLASTOMA: benign vascular tumour; Lindau triad: RCC + hemangioblastoma + epididymal cysts; "
            "cystic component + mural nodule on MRI; symptomatic lesions: surgery/radiosurgery; "
            "RETINAL HEMANGIOBLASTOMA: present early (adolescence); blindness if untreated; annual ophthalmology mandatory; "
            "PHEOCHROMOCYTOMA: ADRENAL, BILATERAL in up to 60% of Type 2A/2B VHL; "
            "SECRETING: norepinephrine-producing primarily; plasma normetanephrine elevated; "
            "usually discovered on surveillance; malignant rate LOW (~5% VHL pheo); "
            "ENDOLYMPHATIC SAC TUMOURS (ELST): hearing loss + tinnitus; MRI base of skull; "
            "PANCREATIC CYSTS/NEUROENDOCRINE TUMOURS (pNET): pancreatic lesions in 35–70%; pNET surgery if >3cm. "
            "BELZUTIFAN (HIF2α inhibitor, FDA 2021): "
            "FDA approved for VHL-related RCC, hemangioblastoma, and pNET not requiring immediate surgery; "
            "ORR 49% in VHL RCC (LITESPARK-004); reduces need for multiple surgeries; "
            "anaemia is class effect (belzutifan → reduced EPO → anaemia); monitor haemoglobin; "
            "CONTRAINDICATED in pregnancy (teratogenic). "
            "SURVEILLANCE SCHEDULE: "
            "Annual ophthalmology (from infancy/early childhood); "
            "annual audiometry (ELST); "
            "annual plasma metanephrines/normetanephrines; "
            "annual MRI brain/spine (hemangioblastoma); "
            "annual MRI abdomen (RCC/pNET); "
            "genetic testing of first-degree relatives."
        ),
        "locus": "3p25.3",
        "aa": 213,
        "kDa": 24,
        "omim_gene": "608537",
        "omim_disease": "Von Hippel-Lindau Syndrome OMIM 193300 — subtype 2A/2B/2C includes pheochromocytoma",
        "inheritance": "AD two-hit LOF (Knudson model); germline + somatic second-hit; de novo rate ~20%; VHL type determines pheo risk (Type 2 = pheo risk; Type 1 = RCC/hemangioblastoma dominant)",
        "gene_class": "E3 ubiquitin ligase substrate recognition — CRL2-VHL complex — HIF1α/HIF2α prolyl-hydroxylation-dependent degradation — VHL LOF → constitutive HIF activation → pseudo-hypoxia → VEGF/PDGF angiogenic transcriptome — CA-IX biomarker",
        "key_alerts": [
            "VHL-SUBTYPE-DETERMINES-PHEO-RISK: VHL TYPE predicts pheo risk; Type 1 (truncating/deletion): RCC+hemangioblastoma dominant, pheo RARE; Type 2A/2B missense: pheo + hemangioblastoma ± RCC; Type 2C: pheo ONLY; know the genotype before counselling pheo risk — Type 1 mutation carrier does NOT need annual metanephrine surveillance if confirmed Type 1",
            "VHL-BELZUTIFAN-FDA2021-HIF2a-INHIBITOR: Belzutifan is the first approved HIF2α inhibitor (FDA 2021) for VHL-related non-metastatic clear cell RCC, hemangioblastoma, and pNET not requiring immediate surgery; ORR 49%; reduces cumulative surgical burden; anaemia is expected class effect; avoid in pregnancy (teratogenic)",
            "VHL-3CM-THRESHOLD-ACTIVE-SURVEILLANCE: VHL RCC <3cm → annual MRI surveillance, NO surgery; VHL RCC ≥3cm → nephron-sparing surgery or ablation; 3cm threshold avoids unnecessary nephrectomies (multifocal bilateral disease); belzutifan can delay reaching surgical threshold",
            "VHL-ANNUAL-OPHTHALMOLOGY-RETINAL-HEMANGIOBLASTOMA: Retinal hemangioblastoma is the EARLIEST VHL manifestation (adolescence); blindness is preventable with annual ophthalmology from infancy; laser photocoagulation or anti-VEGF for small lesions; vitreoretinal surgery for advanced disease; do NOT wait for visual symptoms",
            "VHL-BILATERAL-ADRENAL-PHEO-SURVEILLANCE: Annual plasma metanephrines/normetanephrines for Type 2 VHL; bilateral adrenal pheo in up to 60%; plan cortical-sparing adrenalectomy to avoid bilateral adrenalectomy (Addison's) — unilateral adrenalectomy first, then cortex-sparing on contralateral side; pre-operative alpha-blockade (phenoxybenzamine) mandatory",
        ],
        "etiologies": [
            "VHL Type 2A/2B germline missense (Y98H, R167W, Y112H) → pVHL-elongin interaction preserved but HIF binding impaired → HIF1α/2α not efficiently ubiquitinated → partial pseudo-hypoxia → pheo + hemangioblastoma formation; missense allows some residual pVHL function (less severe than truncating → lower RCC risk in Type 2A)",
            "VHL Type 2C germline missense (E70A, F76del) → pheo ONLY — specific structural disruption allows HIF binding to be maintained for RCC suppression while failing for pheo suppression; Type 2C demonstrates that pheo and RCC tumorigenesis via VHL are separable functional domains",
            "VHL germline deletion/truncation (Type 1) → no pVHL protein → complete CRL2-VHL complex loss → HIF constitutively stable → maximal pseudo-hypoxia → RCC + cerebellar hemangioblastoma; pheo RARE because Type 1 pVHL lacks the specific structural integrity required for adrenal chromaffin cell tumour suppression",
        ],
        "stats": {
            "mean_dx_age": 32,
            "mean_dx_delay_months": 18,
            "bilateral_adrenal_pheo_pct": 60,
            "rcc_concurrent_pct": 50,
            "cerebellar_hemangioblastoma_pct": 35,
        },
        "dx_delay_distribution": "6–36 months (VHL often detected on surveillance once index case identified; de novo cases may present with RCC or hemangioblastoma first; pheo discovered incidentally on abdominal MRI)",
    },

    # ── RET — MEN2A/MEN2B Proto-Oncogene — Medullary Thyroid Cancer ─────────
    {
        "gene": "RET",
        "protein": "RET — MEN2A-MEN2B AD-GOF — Receptor-Tyrosine-Kinase GDNF-Receptor — C634R-MEN2A-MTC+Pheo+PHPT — M918T-MEN2B-Marfanoid+MucoNeuromas — Prophylactic-Thyroidectomy-Age-Variant-Dependent",
        "alias": (
            "RET (rearranged during transfection proto-oncogene / receptor tyrosine kinase); OMIM gene 164761; "
            "Multiple Endocrine Neoplasia type 2A (MEN2A) OMIM 171400; MEN2B OMIM 162300. "
            "10q11.21; 1114 aa; ~124 kDa; AD GOF (gain-of-function). "
            "FUNCTION: RET encodes a receptor tyrosine kinase (RTK) expressed in neural crest-derived cells. "
            "Normal RET activation: GDNF family ligands (GDNF, neurturin, artemin, persephin) bind GFRα co-receptors → "
            "RET extracellular domain dimerisation → intracellular kinase domain transphosphorylation → "
            "activation of RAS/ERK, PI3K/AKT, PLC-γ, JNK signalling → proliferation + survival + differentiation "
            "of neural crest derivatives (kidney, autonomic nervous system, enteric neurons). "
            "GOF MECHANISMS: "
            "MEN2A (cysteine mutations — exon 10/11 extracellular domain): "
            "cysteine residues normally form disulfide bonds holding RET extracellular domain in monomer configuration; "
            "cysteine mutation → free thiol → disulfide bond with adjacent RET molecule → constitutive dimerisation "
            "→ constitutive RET kinase activation WITHOUT ligand → uncontrolled proliferation; "
            "C634R (exon 11): most common MEN2A mutation; "
            "C620R, C618R, C611R (exon 10): lower pheo penetrance than C634R; "
            "MEN2B (M918T — exon 16 intracellular kinase domain): "
            "M918T alters substrate specificity of RET kinase → activates different downstream substrates → "
            "more aggressive biology; M918T accounts for 95% of MEN2B; "
            "de novo M918T in ~50% of MEN2B (new mutations). "
            "MEN2A PHENOTYPE (triad): "
            "1. MEDULLARY THYROID CANCER (MTC): 90–95% penetrance; bilateral + multicentric; "
            "C-cell hyperplasia → MTC; earliest manifestation; calcitonin elevated BEFORE tumour palpable; "
            "2. PHEOCHROMOCYTOMA: 40–50% lifetime risk (C634R highest risk); "
            "adrenal (not extraadrenal); bilateral in 50–80%; SECRETING (norepinephrine + epinephrine); "
            "ALWAYS operate pheo BEFORE thyroid surgery (avoid hypertensive crisis induction); "
            "3. PRIMARY HYPERPARATHYROIDISM (PHPT): 20–30% (4-gland hyperplasia, not adenoma); "
            "MEN2B PHENOTYPE (no PHPT, most severe): "
            "Medullary thyroid cancer: EARLIEST ONSET (age 1–2 years); most aggressive; "
            "Pheochromocytoma: 40–50%; "
            "Marfanoid habitus: tall, long limbs, high palatal arch — but no lens dislocation; "
            "Mucosal neuromas: tongue, lips, eyelids (everted) — PATHOGNOMONIC; ganglioneuromatosis; "
            "PROPHYLACTIC THYROIDECTOMY: "
            "Age-based on ATA risk categorisation: "
            "HIGHEST risk (M918T): within first 6 months of life; "
            "HIGH risk (C634F/G/R/S/W/Y, C618R, C620R): by age 5; "
            "MODERATE risk (other cysteine variants): by age 5–10. "
            "TREATMENT: "
            "MTC: total thyroidectomy + central neck dissection; vandetanib/cabozantinib for systemic disease; "
            "Pheo: laparoscopic adrenalectomy (cortex-sparing preferred); alpha-blockade pre-op; "
            "PHPT: parathyroidectomy at time of thyroid surgery."
        ),
        "locus": "10q11.21",
        "aa": 1114,
        "kDa": 124,
        "omim_gene": "164761",
        "omim_disease": "Multiple Endocrine Neoplasia type 2A (MEN2A) OMIM 171400; MEN2B OMIM 162300; FMTC OMIM 155240",
        "inheritance": "AD GOF — constitutive RET kinase activation; almost all RET MEN2 mutations are missense gain-of-function; de novo rate ~50% in MEN2B (M918T)",
        "gene_class": "Receptor tyrosine kinase — GDNF/GFRα signalling — neural crest cell survival/differentiation — RAS/ERK/PI3K/AKT/PLC-γ downstream — GOF via constitutive dimerisation (cysteine mutants) or altered kinase substrate specificity (M918T)",
        "key_alerts": [
            "RET-PHEO-BEFORE-THYROID-SURGERY-MANDATORY: In any MEN2 patient with both pheo and MTC, the PHEOCHROMOCYTOMA must be operated FIRST; induction of anaesthesia for thyroid surgery without pre-treating pheo → hypertensive crisis → mortality; screen plasma metanephrines before ALL thyroid surgery in RET mutation carriers",
            "RET-PROPHYLACTIC-THYROIDECTOMY-AGE-BY-VARIANT: Timing is variant-dependent: M918T (MEN2B) → thyroidectomy within 6 months of birth; C634 (highest risk MEN2A) → by age 5; moderate risk variants → by age 5–10; never delay thyroid surgery based on normal calcitonin in HIGH/HIGHEST risk variants — C-cell hyperplasia precedes elevation",
            "RET-MEN2B-MUCOSAL-NEUROMAS-PATHOGNOMONIC: Mucosal neuromas (tongue, lips, everted eyelids) are PATHOGNOMONIC for MEN2B; any child with mucosal neuromas must have URGENT RET M918T testing; MEN2B presents as early as age 1–2 with aggressive MTC — delay is life-threatening",
            "RET-BILATERAL-ADRENAL-PHEO-CORTEX-SPARING: RET pheo is bilateral in up to 80% of MEN2A; ALWAYS attempt cortex-sparing adrenalectomy to avoid bilateral adrenalectomy (Addisonian crisis, glucocorticoid dependence); if bilateral adrenalectomy unavoidable → hydrocortisone sick-day rules mandatory + medic-alert bracelet",
            "RET-VANDETANIB-CABOZANTINIB-SYSTEMIC-MTC: Vandetanib (VEGFR/EGFR/RET inhibitor) and cabozantinib (RET/MET/VEGFR2) FDA-approved for progressive metastatic MTC; both cause QT prolongation — ECG monitoring mandatory; diarrhoea, hand-foot syndrome, hypertension are common toxicities; pralsetinib/selpercatinib (selective RET inhibitors) approved for RET-altered MTC with superior selectivity",
        ],
        "etiologies": [
            "RET C634R (exon 11) — most common MEN2A mutation; Cys634 normally forms critical disulfide bond in extracellular cadherin-like domain; Arg substitution → free thiol on adjacent Cys → intermolecular disulfide bond → constitutive homodimerisation → continuous RAS/ERK activation → MTC + pheo + PHPT",
            "RET M918T (exon 16) — defines MEN2B; Met918 lines the catalytic cleft of RET kinase domain; Thr substitution changes substrate specificity from RET2A substrates to substrates resembling oncogenic EGFR/SRC → broader oncogenic signalling → more aggressive MTC (onset age 1–2) + marfanoid + mucosal neuromas + pheo",
            "RET C620R/C618R/C611R (exon 10) — lower pheo penetrance than C634; familial MTC (FMTC) risk higher with C620; these extracellular domain cysteines disrupt monomer disulfide architecture → constitutive dimerisation via intermolecular disulfide bonding; same mechanism as C634 but different structural position → different penetrance profiles",
        ],
        "stats": {
            "mean_dx_age": 35,
            "mean_dx_delay_months": 12,
            "medullary_thyroid_cancer_pct": 90,
            "bilateral_adrenal_pheo_pct": 55,
            "hyperparathyroidism_pct": 25,
        },
        "dx_delay_distribution": "6–24 months (MEN2 often identified by cascade testing after index case; pheo discovered on surveillance; MEN2B de novo may be delayed if marfanoid features not recognised)",
    },

    # ── NF1 — Neurofibromatosis Type 1 — Epinephrine-Secreting Adrenal Pheo ─
    {
        "gene": "NF1",
        "protein": "NF1 — Neurofibromatosis-Type-1 AD — Neurofibromin-1 RAS-GAP — Cafe-au-Lait Macules + Lisch Nodules + Neurofibromas — Pheo-in-0.1-5.7pct — Epinephrine-Secreting-ADRENAL — Hypertensive-Crisis-Presentation",
        "alias": (
            "NF1 (neurofibromin 1); OMIM gene 613113; "
            "Neurofibromatosis type 1 (NF1) OMIM 162200. "
            "17q11.2; 2818 aa; ~327 kDa; AD — LOF (neurofibromin is a tumour suppressor). "
            "FUNCTION: Neurofibromin is a GTPase-activating protein (GAP) — specifically, it contains a central "
            "RAS-GAP domain (RAS-GRD) that accelerates intrinsic GTPase activity of RAS proteins "
            "(H-RAS, K-RAS, N-RAS) → promotes conversion of active RAS-GTP → inactive RAS-GDP. "
            "Neurofibromin LOF → RAS-GTP accumulates → constitutive activation of RAS/RAF/MEK/ERK and PI3K/AKT/mTOR → "
            "increased proliferation, survival, and transformation in neural-crest derived cells. "
            "NF1 PHEO — UNIQUE CATECHOLAMINE BIOCHEMISTRY: "
            "NF1-associated pheo is located exclusively in the ADRENAL medulla (not extraadrenal); "
            "NF1 adrenal chromaffin cells predominantly express PNMT (phenylethanolamine N-methyltransferase), "
            "the enzyme that converts norepinephrine → EPINEPHRINE; "
            "therefore NF1 pheo secretes BOTH epinephrine AND norepinephrine, but often PREDOMINANTLY EPINEPHRINE; "
            "plasma METANEPHRINE (metabolite of epinephrine) is the primary biomarker — elevated in NF1 pheo; "
            "this contrasts with SDH pheos which primarily secrete norepinephrine (elevated normetanephrine). "
            "NF1 DIAGNOSIS (clinical — NIH criteria): "
            "6+ café-au-lait macules (≥5mm prepubertal; ≥15mm postpubertal); "
            "Lisch nodules (iris hamartomas: seen on slit-lamp); "
            "neurofibromas (cutaneous ≥2 or 1 plexiform); "
            "axillary/inguinal freckling; "
            "optic glioma; "
            "distinctive bony abnormality (sphenoid dysplasia, tibial pseudarthrosis); "
            "first-degree relative with NF1. "
            "NF1 PHEO PREVALENCE: 0.1–5.7% of NF1 patients; "
            "typically present in 30s–50s (older than SDH/RET); "
            "PRESENTATION: HYPERTENSIVE CRISIS most common presentation (catecholamine surge); "
            "paroxysmal hypertension + headache + sweating + palpitations — THE CLASSIC TETRAD; "
            "tumour is almost always unilateral; bilateral NF1 pheo uncommon (<5%); "
            "MALIGNANCY RATE: ~10% NF1 pheo — higher than VHL/RET, lower than SDHB. "
            "NF1 MANAGEMENT: "
            "Pheo: laparoscopic adrenalectomy with pre-operative alpha blockade; "
            "no routine surveillance for pheo in NF1 (low overall prevalence); "
            "target symptomatic patients + those with abdominal NF1 complications; "
            "MEK inhibitor (selumetinib, FDA 2020) for NF1-related plexiform neurofibroma in children ≥2 years; "
            "optic glioma: carboplatin/vincristine or MEK inhibitor; "
            "annual surveillance: blood pressure, ophthalmology, neurodevelopmental assessment; "
            "annual MRI NOT routinely recommended in asymptomatic adults."
        ),
        "locus": "17q11.2",
        "aa": 2818,
        "kDa": 327,
        "omim_gene": "613113",
        "omim_disease": "Neurofibromatosis type 1 (NF1) OMIM 162200 — pheo in 0.1–5.7%; adrenal only; epinephrine-secreting",
        "inheritance": "AD LOF — neurofibromin haploinsufficiency; second somatic hit required for tumour formation; de novo rate ~50% (high mutation rate at NF1 — largest tumour suppressor gene in genome)",
        "gene_class": "RAS GTPase-activating protein (GAP) — RAS-GRD domain — accelerates RAS-GTP → RAS-GDP hydrolysis — NF1 LOF → constitutive RAS/RAF/MEK/ERK + PI3K/AKT/mTOR — 2818 aa largest tumour suppressor — neural crest cell proliferation regulator",
        "key_alerts": [
            "NF1-PHEO-EPINEPHRINE-SECRETING-ADRENAL-ONLY: NF1 pheo is ADRENAL-ONLY (not extraadrenal) and predominantly EPINEPHRINE-secreting; plasma METANEPHRINE (not normetanephrine) is the primary biomarker; screen symptomatic NF1 patients (hypertension, paroxysms) with plasma metanephrine first — do not rely on normetanephrine alone for NF1 pheo screening",
            "NF1-HYPERTENSIVE-CRISIS-PRESENTATION: NF1 pheo classically presents as HYPERTENSIVE CRISIS — paroxysmal severe hypertension + headache + diaphoresis + palpitations; any NF1 patient with paroxysmal hypertension must have plasma metanephrines checked URGENTLY; intraoperative hypertensive crisis during unrelated surgery has been the first manifestation",
            "NF1-DIAGNOSIS-CLINICAL-NOT-GENETIC: NF1 is primarily a CLINICAL diagnosis (NIH criteria); genetic testing confirms but is not always required; the 2 most pathognomonic features are Lisch nodules (slit-lamp) and plexiform neurofibromas; NF1 gene sequencing detects ~95% but MLPA adds 5% (large deletions); mosaic NF1 requires tumour/affected tissue sequencing",
            "NF1-SELUMETINIB-MEK-INHIBITOR-PLEXIFORM: Selumetinib (MEK1/2 inhibitor) FDA 2020 for symptomatic inoperable plexiform neurofibromas in children ≥2 years; tumour volume reduction ~20% in SPRINT trial; not indicated for malignant peripheral nerve sheath tumour (MPNST) — MEK inhibitor resistance in MPNST; monitor for cardiomyopathy (left ventricular monitoring mandatory)",
            "NF1-NO-ROUTINE-PHEO-SURVEILLANCE: Pheo occurs in only 0.1–5.7% of NF1 — routine annual plasma metanephrine surveillance is NOT currently recommended for all NF1 patients; target surveillance to symptomatic patients (paroxysmal hypertension/headache/sweating); diagnosis of NF1 alone does NOT warrant annual pheo biochemistry unless symptomatic",
        ],
        "etiologies": [
            "NF1 de novo or inherited LOF + somatic second-hit in adrenal chromaffin cells → complete neurofibromin loss → RAS-GTP accumulates → constitutive MEK/ERK proliferation signal → adrenal pheo; PNMT activity in adrenal medulla → epinephrine-predominant catecholamine secretion distinguishing NF1 from extraadrenal SDH pheo",
            "NF1 large deletion (14-17q11.2 contiguous deletion) — NF1 microdeletion syndrome: deletion of NF1 + flanking genes (SPRED1, SUZ12); more severe phenotype — facial dysmorphia + intellectual disability + faster neurofibroma growth; large deletions detected by MLPA; pheo risk similar to point mutations",
            "NF1 segmental/mosaic NF1 — post-zygotic NF1 mutation; features limited to a body segment; pheo possible if adrenal soma carries the mutation; germline testing may be negative in mosaic NF1 — tumour tissue or skin biopsy from affected area required for molecular diagnosis",
        ],
        "stats": {
            "mean_dx_age": 45,
            "mean_dx_delay_months": 30,
            "adrenal_pheo_pct": 100,
            "epinephrine_secreting_pct": 85,
            "hypertensive_crisis_pct": 50,
        },
        "dx_delay_distribution": "12–60 months (NF1 pheo rare — often not considered; hypertension in NF1 attributed to renal artery stenosis or essential hypertension; NF1 diagnosis already known but pheo not screened routinely)",
    },

    # ── MAX — MYC-Associated Factor X — Paternal Imprinting Bilateral Pheo ───
    {
        "gene": "MAX",
        "protein": "MAX — MYC-Associated-Factor-X AD-Paternal-Imprinting — Bilateral-Adrenal-Pheo — Catecholamine-Secreting — Young-Onset — No-Head-Neck-PGL — SDHB-IHC-Retained",
        "alias": (
            "MAX (MYC-associated factor X); OMIM gene 154950; "
            "Hereditary Pheochromocytoma (MAX-related) OMIM 614537. "
            "14q23.3; 151 aa; ~17 kDa; AD — PATERNAL IMPRINTING (same imprinting as SDHD). "
            "FUNCTION: MAX is a basic helix-loop-helix leucine zipper (bHLH-LZ) transcription factor. "
            "MAX forms heterodimers with MYC family oncoproteins (c-MYC, N-MYC, L-MYC) → "
            "MYC-MAX heterodimer binds E-box sequences (CACGTG) → "
            "activates transcription of MYC target genes (cell cycle progression, ribosome biogenesis, metabolic reprogramming). "
            "MAX also heterodimerises with MAD/MXD family proteins → "
            "MAD-MAX represses MYC target genes → growth arrest and differentiation. "
            "Therefore MAX functions as a CENTRAL NODE connecting oncogenic MYC transcription to growth suppression: "
            "MAX LOF → MYC-MAX transactivation impaired; "
            "however, also MAD-MAX repression abolished → NET EFFECT in chromaffin cells: "
            "increased MYC oncogenic output (c-MYC freed from MAX competition) → proliferation. "
            "PATERNAL IMPRINTING — like SDHD: "
            "MAX maternal allele is epigenetically silenced; "
            "disease occurs ONLY from paternally inherited MAX mutations; "
            "maternally transmitted MAX mutations → carrier only; "
            "clinical cascade testing must establish parent-of-origin. "
            "CLINICAL PHENOTYPE: "
            "BILATERAL ADRENAL PHEOCHROMOCYTOMA: 60–80% of MAX patients develop bilateral adrenal pheo; "
            "NO head/neck PGL (contrast SDHD/SDHC); NO extraadrenal abdominal PGL (contrast SDHB); "
            "YOUNG ONSET: median age at diagnosis ~28 years (youngest of all hereditary pheo genes); "
            "CATECHOLAMINE-SECRETING: predominantly norepinephrine + epinephrine from bilateral adrenal; "
            "plasma metanephrines + normetanephrines both elevated; "
            "MALIGNANCY RATE: ~20–25% (intermediate between SDHD and SDHB); "
            "BIOCHEMISTRY: SDHB IHC RETAINED (unlike SDH mutations); MAX mutations do NOT disrupt SDH complex → "
            "SDHB protein is not lost → SDHB IHC remains positive — critical DDx from SDH mutations; "
            "DOTATATE-PET useful for staging. "
            "DIAGNOSIS: "
            "MAX germline sequencing in young-onset bilateral adrenal pheo with SDHB IHC retained; "
            "MAX mutations account for ~1–2% of hereditary pheo; "
            "MAX testing should follow SDHB/SDHD/SDHC/VHL/RET/NF1 negative workup in bilateral adrenal pheo. "
            "TREATMENT: "
            "Cortex-sparing bilateral adrenalectomy (staged: most dangerous pheo first); "
            "alpha-blockade mandatory pre-operatively; "
            "long-term surveillance for metachronous bilateral disease; "
            "DOTATATE-PET for metastatic disease evaluation."
        ),
        "locus": "14q23.3",
        "aa": 151,
        "kDa": 17,
        "omim_gene": "154950",
        "omim_disease": "Hereditary Pheochromocytoma MAX-related OMIM 614537 — bilateral adrenal, young onset, paternal imprinting",
        "inheritance": "AD — PATERNAL IMPRINTING: disease only from paternally inherited MAX mutations; maternal MAX mutations = carrier only; same imprinting mechanism as SDHD at 11q23",
        "gene_class": "bHLH-LZ transcription factor — MYC-MAX-MAD regulatory network hub — E-box binding — MYC oncogenic partner — MAD tumour suppressor dimerisation partner — MAX LOF disrupts both oncogenic and tumour-suppressive arms of MYC network",
        "key_alerts": [
            "MAX-PATERNAL-IMPRINTING-LIKE-SDHD: MAX disease occurs ONLY from paternally inherited mutations; a patient who inherits MAX mutation from their MOTHER is a carrier but has NO pheo risk; always establish parent-of-origin; paternal inheritance → full surveillance; maternal inheritance → no personal risk (but their children via paternal transmission would be at risk)",
            "MAX-BILATERAL-ADRENAL-YOUNG-ONSET-SDHB-IHC-RETAINED: MAX pheo is bilateral adrenal + young onset (~28 years) with SDHB IHC RETAINED (positive) — this distinguishes MAX from SDH mutations where SDHB IHC is LOST; any bilateral adrenal pheo with retained SDHB IHC in a young patient should trigger MAX sequencing after VHL/RET exclusion",
            "MAX-NO-HEAD-NECK-PGL-NO-EXTRAADRENAL-ABD: Unlike SDH mutations, MAX pheo is restricted to ADRENAL only; no head/neck PGL and no extraadrenal abdominal PGL; surveillance MRI is therefore focused on adrenal glands rather than whole-body MRI required for SDHB",
            "MAX-CORTEX-SPARING-ADRENALECTOMY-STAGED: Bilateral adrenal pheo in MAX requires staged CORTEX-SPARING adrenalectomy; bilateral simultaneous adrenalectomy → permanent Addison's disease → life-threatening steroid dependency; operate the larger/more dangerous pheo first; delay second side ≥6 months; monitor cortisol post-operatively",
            "MAX-MALIGNANT-RISK-INTERMEDIATE-20-25PCT: MAX malignancy rate ~20–25% (higher than VHL/RET/SDHD, lower than SDHB); DOTATATE-PET staging mandatory at diagnosis; annual biochemical surveillance post-resection; young-onset bilateral disease warrants lifelong imaging surveillance every 2–3 years",
        ],
        "etiologies": [
            "MAX germline heterozygous LOF (paternally inherited) + somatic second-hit → complete MAX loss in adrenal chromaffin cells → MYC-MAX transactivation axis disrupted + MAD-MAX tumour suppression lost → net MYC-driven proliferation → bilateral adrenal pheo at young age",
            "MAX missense disrupting MYC dimerisation interface — some MAX variants affect the bHLH domain required for MYC binding without eliminating MAX protein → dominant-negative effect competing with wild-type MAX for E-box binding → functional MAX pathway disruption equivalent to truncating mutation",
            "MAX germline truncating (nonsense/frameshift) → absent MAX protein → MYC freed from MAX-dependent regulation; c-MYC, N-MYC, L-MYC all unconstrained → maximal E-box transactivation → adrenal chromaffin hyperproliferation → pheo; N-MYC amplification seen as somatic event in aggressive MAX-mutant pheo",
        ],
        "stats": {
            "mean_dx_age": 28,
            "mean_dx_delay_months": 18,
            "bilateral_adrenal_pct": 70,
            "malignant_pct": 22,
            "no_head_neck_pgl_pct": 100,
        },
        "dx_delay_distribution": "6–36 months (bilateral adrenal pheo in young adult often prompts hereditary workup; SDH IHC retained may delay diagnosis if SDH panel ordered first; MAX not included in all hereditary pheo panels)",
    },

    # ── SDHA — SDH Complex II Catalytic A — Pituitary Adenoma Unique ─────────
    {
        "gene": "SDHA",
        "protein": "SDHA — SDH5 AR-Biallelic-Full/AD-Haploinsufficiency — FAD-Flavoprotein-Catalytic-Subunit — Pituitary-Adenoma-UNIQUE — SDHA-IHC-Specific-Dual-Staining — Rarest-SDH-Pheo",
        "alias": (
            "SDHA (succinate dehydrogenase flavoprotein subunit A); OMIM gene 600857; "
            "SDH-related PGL/pheo (SDH5 / SDHA-PGL) — SDHA-related pituitary adenoma OMIM 615830. "
            "5p15.33; 664 aa; ~73 kDa; AR for Leigh syndrome; AD haploinsufficiency for pheo/PGL risk. "
            "FUNCTION: SDHA is the largest and catalytic subunit of SDH Complex II. "
            "SDHA contains the FAD (flavin adenine dinucleotide) prosthetic group at its N-terminal succinate-binding domain. "
            "Catalytic mechanism: succinate enters the succinate-binding pocket of SDHA → "
            "succinate oxidised to fumarate → 2 electrons transferred to FAD → FAD → FADH2 → "
            "electrons pass via SDHB Fe-S clusters → to ubiquinone at the Qp site of SDHC/SDHD; "
            "net reaction: succinate + ubiquinone → fumarate + ubiquinol; "
            "SDHA is essential for TCA cycle flux and Complex II electron transport. "
            "INHERITANCE — DUAL: "
            "BIALLELIC (AR) SDHA LOF → Leigh syndrome (mitochondrial encephalopathy); "
            "HETEROZYGOUS (AD haploinsufficiency) SDHA LOF → susceptibility to pheo/PGL/GIST (low penetrance); "
            "penetrance of heterozygous SDHA for pheo is ~1–2% (EXTREMELY LOW); "
            "~1% of general population carries heterozygous SDHA variant — most are benign passengers; "
            "SDHA pheo is the RAREST SDH-related pheo. "
            "SDHA IHC — CRITICAL SPECIFICITY: "
            "SDHA IHC is SPECIFIC for SDHA mutation (unlike SDHB IHC which is SENSITIVE for all SDH mutations): "
            "SDHB IHC loss = any SDH subunit mutation (SDHB/C/D/A); "
            "SDHA IHC loss = ONLY SDHA mutation (not loss of other subunits); "
            "therefore DUAL IHC staining required: "
            "SDHB lost + SDHA retained → SDHB, SDHC, or SDHD mutation; "
            "SDHB lost + SDHA also lost → SDHA mutation. "
            "PITUITARY ADENOMA — UNIQUE ASSOCIATION: "
            "SDHA-associated pituitary adenoma is a UNIQUE association not seen with SDHB/C/D; "
            "GH-secreting (acromegaly) and non-secreting pituitary adenomas reported; "
            "pituitary MRI annually in SDHA heterozygous carriers; "
            "SDHA accounts for <5% of SDH-related pituitary adenomas. "
            "CLINICAL PHEO PHENOTYPE: "
            "Adult onset (older than SDHB/MAX); mean diagnosis age ~50 years; "
            "predominantly head/neck PGL or abdominal PGL; "
            "malignancy rate VERY LOW (~5%); "
            "GIST association: SDHA-deficient GIST (seen with SDHC/SDHD also). "
            "POPULATION CARRIER FREQUENCY: ~1 in 100 carry an SDHA heterozygous variant; "
            "VAST MAJORITY of SDHA heterozygotes will NEVER develop pheo; "
            "VUS management: SDHA IHC on any tumour specimen is the best functional assay. "
            "TREATMENT: same as other SDH pheo — surgical resection; surveillance."
        ),
        "locus": "5p15.33",
        "aa": 664,
        "kDa": 73,
        "omim_gene": "600857",
        "omim_disease": "SDHA-related PGL/Pheo (SDH5) — pituitary adenoma OMIM 615830; Leigh syndrome (biallelic) OMIM 256000",
        "inheritance": "AD haploinsufficiency (heterozygous) for pheo/PGL/pituitary adenoma — very low penetrance (~1–2%); AR biallelic LOF → Leigh syndrome (lethal in childhood); ~1% population carrier frequency for heterozygous SDHA variants",
        "gene_class": "FAD-containing flavoprotein catalytic subunit of SDH Complex II — succinate → fumarate oxidation — FADH2 electron donor to Fe-S chain — SDHA IHC loss SPECIFIC (not sensitive) for SDHA mutation — dual SDHB+SDHA IHC required to identify SDHA-specific loss",
        "key_alerts": [
            "SDHA-IHC-DUAL-STAINING-SPECIFIC: SDHA IHC loss is SPECIFIC for SDHA mutation (unlike SDHB IHC which is SENSITIVE for all SDH mutations); always perform DUAL staining: SDHB lost + SDHA retained → order SDHB/C/D germline; SDHB lost + SDHA also lost → order SDHA germline; never assume SDHA mutation from SDHB IHC loss alone",
            "SDHA-1PCT-POPULATION-CARRIER-LOW-PENETRANCE: ~1% of the general population carries a heterozygous SDHA variant; penetrance for pheo/PGL is only 1–2%; SDHA VUS must be interpreted cautiously — SDHA IHC on tumour tissue is the best functional VUS classifier; do NOT alarm patients with heterozygous SDHA variants of uncertain significance without tumour IHC data",
            "SDHA-PITUITARY-ADENOMA-UNIQUE-ANNUAL-MRI: Pituitary adenoma is UNIQUE to SDHA among SDH pheo genes; annual pituitary MRI in SDHA heterozygous germline carriers with confirmed pathogenic variant; GH-secreting pituitary adenoma → acromegaly may be the presenting feature; IGF-1 + prolactin annually",
            "SDHA-RAREST-SDH-PHEO-LOW-MALIGNANCY: SDHA pheo is the RAREST SDH-related pheo (lower penetrance, older onset); malignancy rate very low (~5%); management is conservative surveillance rather than aggressive intervention; adult onset (mean ~50 years) contrasts with SDHB (mean ~38 years)",
            "SDHA-LEIGH-SYNDROME-BIALLELIC-DISTINCTION: Biallelic SDHA LOF → Leigh syndrome (fatal childhood mitochondrial encephalopathy — NOT a pheo condition); heterozygous SDHA → pheo susceptibility; always clarify which inheritance pattern applies to the patient; Leigh syndrome parents are obligate SDHA heterozygotes with pheo surveillance obligation",
        ],
        "etiologies": [
            "SDHA germline heterozygous LOF (paternally or maternally inherited — no imprinting) + somatic second-hit in chromaffin cells → complete SDHA absence → FADH2 generation abolished → SDH Complex II non-functional → succinate accumulation → HIF pseudo-hypoxia → pheo (same final pathway as SDHB/C/D but via primary succinate oxidation block)",
            "SDHA germline heterozygous missense disrupting FAD binding pocket (Arg31 hotspot) → succinate cannot be oxidised → FADH2 not generated → succinate accumulates; SDHA protein may still be present (missense) but IHC may still show reduced staining due to conformational instability; functional SDHA IHC loss guides pathogenicity assignment",
            "SDHA biallelic (compound heterozygous or homozygous) in Leigh syndrome — two pathogenic SDHA alleles → absent SDHA → complete SDH Complex II loss → severe complex II deficiency → Leigh encephalopathy; parents are obligate heterozygotes with pheo/pituitary surveillance obligation",
        ],
        "stats": {
            "mean_dx_age": 50,
            "mean_dx_delay_months": 48,
            "pituitary_adenoma_pct": 25,
            "malignant_pct": 5,
            "sdha_ihc_loss_pct": 100,
        },
        "dx_delay_distribution": "24–96 months (rarest SDH pheo; SDHA not on all panels; low penetrance means few index cases to trigger cascade testing; pituitary adenoma may present first and prompt SDH IHC reflex testing)",
    },

    # ── SDHC — SDH Complex II Membrane Anchor C — Head/Neck + Gastric GIST ───
    {
        "gene": "SDHC",
        "protein": "SDHC — SDH2 AD-No-Imprinting — Integral-Membrane-Anchor-C — Head/Neck-PGL-Like-SDHD — Gastric-GIST-UNIQUE-SDH-Gene — SDH-Deficient-GIST-SDHC-IHC",
        "alias": (
            "SDHC (succinate dehydrogenase integral membrane subunit C); OMIM gene 602413; "
            "Hereditary PGL/Pheo type 3 (PGL3 / SDHC-PGL) OMIM 605373. "
            "1q23.3; 169 aa; ~17 kDa; AD — LOF; NO imprinting (contrast SDHD). "
            "FUNCTION: SDHC is the larger of the two membrane-anchoring subunits of SDH Complex II. "
            "Together with SDHD, SDHC embeds the SDHA-SDHB catalytic core into the inner mitochondrial membrane. "
            "SDHC contains two transmembrane helices and holds the heme b556 that participates in "
            "electron transfer from SDHB → ubiquinone at the Qp site. "
            "SDHC LOF → same downstream oncogenic cascade as SDHB/D: "
            "SDHB IHC loss (all SDH subunit mutations destabilise SDHB), "
            "succinate accumulation → PHD inhibition → HIF pseudo-hypoxia → VEGF → angiogenesis. "
            "NO GENOMIC IMPRINTING — KEY CONTRAST TO SDHD: "
            "SDHD (also a membrane anchor, 11q23.1) has strict maternal imprinting → maternal transmission = carrier only. "
            "SDHC has NO imprinting — mutations inherited from EITHER parent convey risk; "
            "clinical penetrance ~20–30% lifetime; lower than SDHB; higher than SDHA. "
            "HEAD/NECK PGL — SIMILAR TO SDHD BUT WITHOUT IMPRINTING: "
            "SDHC predominantly causes parasympathetic head/neck PGL: "
            "Carotid body tumour, glomus jugulare, glomus vagale; "
            "NON-SECRETING: parasympathetic PGL → no excess catecholamines; elevated 3-MT possible; "
            "BILATERAL head/neck PGL less common than SDHD (~20%); "
            "ADRENAL PHEO: uncommon in SDHC (~10–15%); "
            "MALIGNANCY RATE: low (~5%) for SDHC head/neck PGL — similar to SDHD. "
            "GASTRIC GIST — UNIQUE SDHC/SDHD ASSOCIATION: "
            "SDH-deficient GIST is a distinct subset of GIST (non-KIT/PDGFRA GIST); "
            "SDHC is MORE COMMONLY associated with GIST than any other SDH subunit; "
            "SDHD also causes SDH-deficient GIST; "
            "Carney triad (non-hereditary): gastric GIST + pulmonary chondroma + paraganglioma — "
            "SDHC promoter hypermethylation (epigenetic, not germline) responsible in Carney triad; "
            "Carney-Stratakis syndrome (hereditary): PGL + GIST — germline SDHB/C/D mutations; "
            "GIST resistance to imatinib: SDH-deficient GIST has KIT/PDGFRA wild-type → "
            "imatinib INEFFECTIVE; sunitinib/regorafenib have some activity; "
            "GIST surveillance: annual abdominal MRI for SDHC germline carriers. "
            "IHC: SDHB IHC loss identifies SDH-deficient GIST; "
            "SDHA IHC retained in SDHC mutation → distinguishes from SDHA mutation. "
            "TREATMENT: "
            "Head/neck PGL: watchful waiting for stable small lesions; surgery or SRS; "
            "GIST: imatinib INEFFECTIVE (KIT/PDGFRA WT); sunitinib or temsirolimus; "
            "surveillance: annual biochemistry + MRI neck + abdomen."
        ),
        "locus": "1q23.3",
        "aa": 169,
        "kDa": 17,
        "omim_gene": "602413",
        "omim_disease": "Hereditary Pheochromocytoma-Paraganglioma type 3 (PGL3 / SDHC-PGL) OMIM 605373; Carney-Stratakis Syndrome OMIM 606864",
        "inheritance": "AD LOF — NO maternal imprinting (contrast SDHD); mutations from either parent convey tumour risk; penetrance ~20–30%; de novo SDHC mutations uncommon",
        "gene_class": "SDH Complex II large integral membrane-anchoring subunit — two transmembrane helices — heme b556 coordination — ubiquinone Qp-site assembly partner with SDHD — SDHC promoter hypermethylation causes Carney triad (non-germline) — SDH-deficient GIST predominant among SDH subunit GISTs",
        "key_alerts": [
            "SDHC-NO-IMPRINTING-CONTRAST-SDHD: SDHC has NO maternal imprinting — any SDHC mutation (maternally or paternally inherited) conveys tumour risk; this is the critical clinical contrast with SDHD (where only paternal inheritance causes disease); all children of SDHC-affected individuals are at 50% risk regardless of which parent is affected",
            "SDHC-GASTRIC-GIST-UNIQUE-IMATINIB-INEFFECTIVE: SDHC-associated gastric GIST is SDH-deficient (KIT/PDGFRA wild-type) → IMATINIB IS INEFFECTIVE; always test SDHB IHC on GIST specimens; SDH-deficient GIST = refer for SDHC/D/B germline testing; sunitinib or temsirolimus for systemic therapy; annual abdominal MRI for GIST surveillance in SDHC carriers",
            "SDHC-HEAD-NECK-PGL-NON-SECRETING-SIMILAR-SDHD: SDHC head/neck PGL is clinically similar to SDHD — parasympathetic, non-secreting, normal plasma metanephrines; check 3-methoxytyramine and chromogranin A; MRI neck + skull base is the primary surveillance; malignancy rate low (~5%) — conservative management appropriate for stable small lesions",
            "SDHC-CARNEY-STRATAKIS-SYNDROME-GIST-PGL-TRIAD: Carney-Stratakis syndrome (hereditary SDHB/C/D mutation) = gastric GIST + paraganglioma; Carney triad (non-hereditary SDHC promoter methylation) = GIST + pulmonary chondroma + PGL; distinguishing germline from epigenetic SDHC alteration requires germline sequencing — Carney triad does NOT require cascade family testing",
            "SDHC-LOWER-MALIGNANCY-THAN-SDHB: SDHC malignancy rate ~5% for head/neck PGL — far lower than SDHB (~40%); patients with SDHC should be reassured of lower malignant potential; watchful waiting is appropriate for stable small tumours; full staging DOTATATE-PET recommended at diagnosis but annual DOTATATE surveillance not required for all stable SDHC carriers",
        ],
        "etiologies": [
            "SDHC germline heterozygous LOF (from either parent) + somatic second-hit at 1q23 → complete SDHC absence → SDHA-SDHB catalytic core cannot anchor to mitochondrial inner membrane → SDH Complex II assembly failure → succinate accumulation → HIF pseudo-hypoxia → parasympathetic PGL and/or gastric GIST formation",
            "SDHC promoter CpG methylation (epigenetic, NOT germline) — Carney triad mechanism: SDHC promoter hypermethylation silences SDHC in GIST + pulmonary chondroma + PGL without germline mutation; SDHB IHC loss identifies the SDH-deficient tumours; germline SDHC negative in Carney triad; cascade family testing NOT indicated; somatic epigenetic mechanism non-hereditary",
            "SDHC p.Arg31Trp missense — disrupts the heme b556 coordination in the transmembrane domain → impaired electron transfer from SDHB to ubiquinone → functional Complex II deficiency → succinate accumulation; Arg31 hotspot mutations identified in multiple unrelated SDHC PGL/GIST families",
        ],
        "stats": {
            "mean_dx_age": 42,
            "mean_dx_delay_months": 42,
            "head_neck_pgl_pct": 60,
            "gastric_gist_pct": 20,
            "malignant_pct": 5,
        },
        "dx_delay_distribution": "24–84 months (non-secreting PGL; GIST may be the first manifestation; SDH IHC reflexing from GIST to germline testing is a newer diagnostic pathway improving detection)",
    },
]


def _generate_patients():
    """Generate 40 patients per gene using deterministic RNG (seed = SEED_BASE + gene_idx)."""
    for idx, gene in enumerate(PHEO_PGL_GENES):
        rng = random.Random(SEED_BASE + idx)
        patients = []
        for i in range(40):
            age_at_dx = rng.randint(18, 70)
            dx_delay = rng.randint(
                int(gene["stats"].get("mean_dx_delay_months", 24) * 0.4),
                int(gene["stats"].get("mean_dx_delay_months", 24) * 1.8),
            )

            g = gene["gene"]

            # Gene-specific catecholamine and clinical parameters
            if g == "SDHB":
                plasma_normetanephrine_elevated = True
                plasma_metanephrine_elevated = rng.random() > 0.5
                tumour_location = rng.choice(["extraadrenal", "extraadrenal", "adrenal"])
                malignant_pgl = rng.random() > 0.6
                bilateral_tumour = rng.random() > 0.7
                rcc_present = False
                secondary_tumour = "metastatic_pgl" if malignant_pgl else None
                age_at_dx = rng.randint(20, 60)

            elif g == "SDHD":
                plasma_normetanephrine_elevated = rng.random() > 0.7  # often non-secreting
                plasma_metanephrine_elevated = rng.random() > 0.85
                tumour_location = rng.choice(["head_neck", "head_neck", "head_neck", "adrenal"])
                malignant_pgl = rng.random() > 0.95
                bilateral_tumour = rng.random() > 0.4  # bilateral carotid body
                rcc_present = False
                secondary_tumour = "bilateral_carotid_body" if bilateral_tumour else None
                age_at_dx = rng.randint(25, 60)

            elif g == "VHL":
                plasma_normetanephrine_elevated = True
                plasma_metanephrine_elevated = rng.random() > 0.4
                tumour_location = "adrenal"
                malignant_pgl = rng.random() > 0.95
                bilateral_tumour = rng.random() > 0.4  # bilateral adrenal pheo
                rcc_present = rng.random() > 0.5
                secondary_tumour = ("rcc_hemangioblastoma" if rng.random() > 0.35
                                    else ("rcc" if rcc_present else None))
                age_at_dx = rng.randint(18, 50)

            elif g == "RET":
                plasma_normetanephrine_elevated = True
                plasma_metanephrine_elevated = True
                tumour_location = "adrenal"
                malignant_pgl = rng.random() > 0.95
                bilateral_tumour = rng.random() > 0.5
                rcc_present = False
                secondary_tumour = ("medullary_thyroid_cancer" if rng.random() > 0.1
                                    else "medullary_thyroid_cancer_phpt")
                age_at_dx = rng.randint(20, 55)

            elif g == "NF1":
                plasma_normetanephrine_elevated = rng.random() > 0.3
                plasma_metanephrine_elevated = True  # epinephrine-secreting
                tumour_location = "adrenal"
                malignant_pgl = rng.random() > 0.9
                bilateral_tumour = rng.random() > 0.95  # bilateral rare in NF1
                rcc_present = False
                secondary_tumour = "plexiform_neurofibroma" if rng.random() > 0.4 else None
                age_at_dx = rng.randint(30, 65)

            elif g == "MAX":
                plasma_normetanephrine_elevated = True
                plasma_metanephrine_elevated = True
                tumour_location = "adrenal"
                malignant_pgl = rng.random() > 0.8
                bilateral_tumour = rng.random() > 0.3  # bilateral in 70%
                rcc_present = False
                secondary_tumour = None
                age_at_dx = rng.randint(18, 45)

            elif g == "SDHA":
                plasma_normetanephrine_elevated = rng.random() > 0.3
                plasma_metanephrine_elevated = rng.random() > 0.4
                tumour_location = rng.choice(["head_neck", "adrenal", "extraadrenal"])
                malignant_pgl = rng.random() > 0.95
                bilateral_tumour = rng.random() > 0.9
                rcc_present = False
                secondary_tumour = "pituitary_adenoma" if rng.random() > 0.75 else None
                age_at_dx = rng.randint(35, 70)

            else:  # SDHC
                plasma_normetanephrine_elevated = rng.random() > 0.7  # non-secreting
                plasma_metanephrine_elevated = rng.random() > 0.85
                tumour_location = rng.choice(["head_neck", "head_neck", "adrenal"])
                malignant_pgl = rng.random() > 0.95
                bilateral_tumour = rng.random() > 0.8
                rcc_present = False
                secondary_tumour = "gastric_gist" if rng.random() > 0.8 else None
                age_at_dx = rng.randint(25, 65)

            hypertensive_crisis = rng.random() > 0.7 if g in ["NF1", "RET"] else rng.random() > 0.6
            on_alpha_blockade = rng.random() > 0.2
            dotatate_avid = rng.random() > 0.15 if g != "NF1" else rng.random() > 0.3
            genetic_cascade_tested = rng.random() > 0.3

            patients.append({
                "patient_id": f"{g}-{i+1:03d}",
                "age_at_dx": age_at_dx,
                "dx_delay_months": dx_delay,
                "plasma_normetanephrine_elevated": plasma_normetanephrine_elevated,
                "plasma_metanephrine_elevated": plasma_metanephrine_elevated,
                "tumour_location": tumour_location,
                "malignant_pgl": malignant_pgl,
                "bilateral_tumour": bilateral_tumour,
                "rcc_present": rcc_present,
                "secondary_tumour": secondary_tumour,
                "hypertensive_crisis": hypertensive_crisis,
                "on_alpha_blockade": on_alpha_blockade,
                "dotatate_avid": dotatate_avid,
                "genetic_cascade_tested": genetic_cascade_tested,
                "gene": g,
                "seed": SEED_BASE + idx,
            })
        gene["patients"] = patients


_generate_patients()


def get_overview():
    all_ages = [p["age_at_dx"] for g in PHEO_PGL_GENES for p in g["patients"]]
    all_delays = [p["dx_delay_months"] for g in PHEO_PGL_GENES for p in g["patients"]]
    genes = []
    for idx, g in enumerate(PHEO_PGL_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        genes.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
            "key_alerts": g["key_alerts"],
            "n_patients": len(g["patients"]),
        })
    return {
        "atlas": "Hereditary-Pheo-PGL-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Pheochromocytoma-Paraganglioma Atlas — "
            "SDHB / SDHD / VHL / RET / NF1 / MAX / SDHA / SDHC — 320 Patients (8×40, Seeds 1686–1693)"
        ),
        "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
        "total_patients": sum(len(g["patients"]) for g in PHEO_PGL_GENES),
        "aggregate_stats": {
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "genes_covered": len(PHEO_PGL_GENES),
            "patients_per_gene": 40,
        },
        "genes": genes,
        "top_alerts": [
            "SDHB-40PCT-MALIGNANT-DOTATATE-MANDATORY: SDHB has the highest malignant PGL rate (~40%); MIBG is inferior for SDHB — DOTATATE-PET mandatory for staging and surveillance; lifelong annual biochemistry + MRI from age 5–8 for all germline SDHB carriers",
            "SDHD-MAX-PATERNAL-IMPRINTING-PARENT-OF-ORIGIN: SDHD and MAX both exhibit PATERNAL IMPRINTING — disease occurs ONLY from paternally inherited mutations; always establish parent-of-origin before advising risk; maternal inheritance = carrier only, no tumour risk",
            "RET-PHEO-BEFORE-THYROID-SURGERY-MEN2: In MEN2 patients with concurrent pheo and MTC, operate the PHEOCHROMOCYTOMA FIRST — induction of anaesthesia for thyroid surgery in unblocked pheo → hypertensive crisis → death; alpha-blockade pre-operative mandatory for all pheo surgery",
            "VHL-BELZUTIFAN-FDA2021-SUBTYPE-DETERMINES-PHEO-RISK: VHL type 2A/2B/2C causes pheo; Type 1 (truncating) predominantly RCC/hemangioblastoma with rare pheo; belzutifan (HIF2α inhibitor) FDA 2021 reduces surgical burden for VHL-related RCC/hemangioblastoma/pNET; annual ophthalmology mandatory from infancy for retinal hemangioblastoma",
            "NF1-PHEO-EPINEPHRINE-ADRENAL-ONLY: NF1 pheo is adrenal-only and EPINEPHRINE-secreting; plasma METANEPHRINE is the primary biomarker (not normetanephrine); hypertensive crisis is the presenting feature; no routine annual pheo surveillance for all NF1 — screen symptomatic patients",
            "SDHA-IHC-DUAL-STAINING-PITUITARY-UNIQUE: SDHA IHC loss is SPECIFIC for SDHA mutation (unlike SDHB IHC which indicates any SDH mutation); dual SDHB+SDHA IHC required; pituitary adenoma association is UNIQUE to SDHA among all SDH genes; annual pituitary MRI for confirmed SDHA pathogenic carriers",
            "SDHC-GASTRIC-GIST-IMATINIB-INEFFECTIVE: SDHC-associated gastric GIST is SDH-deficient (KIT/PDGFRA wild-type) — IMATINIB IS INEFFECTIVE; sunitinib/regorafenib have activity; SDHC has no imprinting (unlike SDHD) — both maternal and paternal inheritance convey risk",
            "MAX-BILATERAL-ADRENAL-YOUNG-SDHB-IHC-RETAINED: MAX pheo: bilateral adrenal, youngest onset (~28 years), SDHB IHC RETAINED (not an SDH gene); cortex-sparing staged adrenalectomy mandatory to avoid permanent Addison's; DOTATATE-PET for malignant staging (malignancy ~20–25%)",
        ],
    }


def get_breakdown():
    result = []
    for idx, g in enumerate(PHEO_PGL_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
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
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def get_definitions():
    return {
        "concepts": {
            "SDH Complex II — Succinate Dehydrogenase and the Pseudo-Hypoxia Oncogenic Axis": (
                "Succinate dehydrogenase (SDH, Complex II) is the only enzyme shared between the TCA cycle "
                "and the mitochondrial electron transport chain. "
                "SUBUNIT ORGANISATION: "
                "SDHA (FAD-flavoprotein, 664 aa) — catalytic subunit; oxidises succinate → fumarate; "
                "SDHB (iron-sulfur protein, 280 aa) — electron transfer via 3 Fe-S clusters; "
                "SDHC (integral membrane, 169 aa) — larger anchor subunit, ubiquinone Qp site; "
                "SDHD (integral membrane, 159 aa) — smaller anchor subunit, heme coordination. "
                "ONCOGENIC MECHANISM: "
                "SDH LOF → succinate cannot be metabolised → succinate accumulates in cytoplasm; "
                "succinate is a competitive inhibitor of α-ketoglutarate (αKG)-dependent dioxygenases: "
                "PHD1/2/3 (prolyl hydroxylases) — inhibited → HIF1α/2α not hydroxylated → "
                "not recognised by pVHL → not ubiquitinated → HIF stable → PSEUDO-HYPOXIA; "
                "TET1/2/3 (DNA demethylases) — inhibited → DNA hypermethylation (CIMP phenotype); "
                "KDM histone demethylases — inhibited → H3K27me3 accumulation → gene silencing. "
                "HIF TARGET GENES: VEGF (angiogenesis), PDGF, SLC2A1 (GLUT1), EPO, CA-IX; "
                "constitutive HIF activation drives paraganglioma and renal cell carcinoma formation. "
                "SDHB IHC AS SURROGATE MARKER: "
                "SDHB is the most labile Complex II subunit — any SDH subunit absence destabilises SDHB; "
                "therefore SDHB IHC loss = ANY SDH subunit mutation (SDHB/C/D/A); "
                "SDHA IHC loss = SPECIFIC for SDHA mutation only; "
                "dual SDHB + SDHA IHC is required to distinguish SDHA-specific from other SDH mutations."
            ),
            "Paternal Genomic Imprinting in Hereditary PGL — SDHD and MAX": (
                "Genomic imprinting is an epigenetic phenomenon where gene expression depends on parent-of-origin. "
                "SDHD (11q23.1) and MAX (14q23.3) are both subject to MATERNAL imprinting: "
                "the maternally inherited allele is epigenetically silenced by methylation of CpG islands "
                "in the gene promoter region during oogenesis; "
                "the paternally inherited allele is expressed normally. "
                "CLINICAL CONSEQUENCE: "
                "A person who inherits an SDHD/MAX mutation from their FATHER → "
                "only the mutated (paternally inherited) allele is expressed → second-hit somatic mutation "
                "→ complete SDHD/MAX loss → paraganglioma/pheo; "
                "A person who inherits an SDHD/MAX mutation from their MOTHER → "
                "the mutated allele is silenced (maternally imprinted) → "
                "only the wild-type paternal allele is expressed → NO tumour risk; "
                "GENOTYPING ALONE IS INSUFFICIENT — parent-of-origin must always be established. "
                "TRANSMISSION RULES: "
                "An SDHD/MAX-carrier father → 50% of children inherit the mutation → ALL at risk; "
                "An SDHD/MAX-carrier mother → 50% of children inherit the mutation → NONE at risk (imprinted); "
                "however, affected male carrier children will transmit to 50% of THEIR children → at risk. "
                "CONTRAST WITH SDHB AND SDHC: neither SDHB nor SDHC is subject to imprinting; "
                "mutations from EITHER parent convey risk."
            ),
            "Catecholamine Biochemistry — Tumour Location Predicts Biomarker": (
                "The catecholamine secretion profile of pheo/PGL is determined by tumour location and gene: "
                "ADRENAL PHEO (chromaffin cells): express PNMT → norepinephrine (NE) → EPINEPHRINE (E); "
                "plasma METANEPHRINE (methyl-NE metabolite) is elevated; "
                "NF1 pheo: EPINEPHRINE-predominant (adrenal, high PNMT); "
                "RET/VHL pheo: NE + E (adrenal); "
                "MAX pheo: NE + E (bilateral adrenal). "
                "EXTRAADRENAL PGL (sympathetic — abdominal/pelvic): no PNMT → only NE secreted; "
                "plasma NORMETANEPHRINE is the primary biomarker; "
                "SDHB retroperitoneal PGL: elevated normetanephrine. "
                "HEAD/NECK PGL (parasympathetic — carotid body/glomus): "
                "secretion usually ABSENT (non-secreting); "
                "3-METHOXYTYRAMINE (3-MT) elevated if dopamine secretion (20–30%); "
                "SDHD and SDHC head/neck PGL: normal plasma metanephrines + normetanephrines in most; "
                "SCREENING ALGORITHM: "
                "Adrenal pheo suspected: plasma metanephrines + normetanephrines; "
                "Extraadrenal PGL suspected: plasma normetanephrine + 3-MT; "
                "Head/neck PGL: plasma 3-MT + chromogranin A (normetanephrine may be normal); "
                "24h urine catecholamines: less sensitive than plasma fractionated metanephrines; "
                "preferred method: plasma fractionated metanephrines (seated 30 min, antecubital); "
                "interfering substances: acetaminophen, labetalol, decongestants → false positives."
            ),
            "DOTATATE-PET vs MIBG — Functional Imaging Selection by Genotype": (
                "Functional imaging for pheo/PGL uses somatostatin receptor (SSTR) or catecholamine uptake tracers. "
                "DOTATATE-PET (68Ga-DOTATATE or 18F-DOTA-peptides): "
                "targets SSTR2 (somatostatin receptor 2) — expressed on most pheo/PGL; "
                "SUPERIOR sensitivity for SDH-related PGL (SDHB/C/D head/neck and extraadrenal); "
                "PREFERRED for: SDHB (high SSTR2), SDHD head/neck PGL, SDHC GIST-PGL, VHL; "
                "detects head/neck PGL with >95% sensitivity; "
                "also identifies metastatic lesions not seen on CT/MRI. "
                "MIBG (123I-MIBG or 131I-MIBG): "
                "metaiodobenzylguanidine — analogue of norepinephrine; "
                "taken up by norepinephrine transporter (NET) expressed in chromaffin cells; "
                "INFERIOR for SDHB extraadrenal PGL (many SDHB tumours have low NET expression — MIBG negative); "
                "PREFERRED for: NF1 adrenal pheo, RET adrenal pheo (high NET expression); "
                "therapeutic 131I-MIBG available for MIBG-avid metastatic pheo (PRRT alternative). "
                "18F-FDOPA-PET: alternative for head/neck PGL (dopaminergic); "
                "FDG-PET: malignant/metastatic SDHB PGL — high FDG avidity in aggressive disease. "
                "GENOTYPE-GUIDED IMAGING SELECTION: "
                "SDHB → DOTATATE-PET (MIBG often false-negative); "
                "SDHD/SDHC head/neck → DOTATATE-PET + MRI skull base; "
                "VHL adrenal → MRI abdomen (structural) ± DOTATATE-PET for staging; "
                "RET/NF1 adrenal → MRI + MIBG or DOTATATE (both useful); "
                "Metastatic disease: DOTATATE-PET for Lu-177-DOTATATE patient selection."
            ),
            "Malignant Pheo/PGL — Definition, Risk Stratification, and Treatment": (
                "Malignant pheochromocytoma/paraganglioma: defined by metastases to non-chromaffin tissue "
                "(lymph nodes, liver, lung, bone) — histology CANNOT predict malignancy reliably. "
                "RISK BY GENE: "
                "SDHB ~40% (highest); MAX ~20–25%; NF1 ~10%; SDHC ~5%; SDHD ~5%; "
                "VHL ~5%; SDHA ~5%; RET ~5% (lowest). "
                "BIOCHEMICAL RISK: "
                "3-MT elevation + large tumour + extra-adrenal location → higher malignant risk; "
                "succinate accumulation (SDH) → CIMP → epigenetic silencing → more aggressive biology in SDHB. "
                "TREATMENT OF MALIGNANT PHEO: "
                "131I-MIBG therapy (Azedra — iobenguane I-131 FDA 2018): for MIBG-avid metastatic pheo; "
                "Lu-177-DOTATATE PRRT (Lutathera): for DOTATATE-avid metastatic PGL (SDHB priority); "
                "Sunitinib: FIRSTMAPPP trial — 36% ORR in progressive SDHB malignant PGL; "
                "TEMCAP (temozolomide + capecitabine): modest activity; "
                "CVD chemotherapy (cyclophosphamide/vincristine/dacarbazine): for rapid progression; "
                "Belzutifan: off-label but rationale for HIF2α inhibition in SDH malignant PGL (trials ongoing). "
                "ALPHA-BLOCKADE PRE-OPERATIVELY: "
                "Phenoxybenzamine (non-selective irreversible α-blocker) preferred: 7–14 days pre-op; "
                "alternative: doxazosin or prazosin (selective α1); "
                "beta-blocker ONLY AFTER alpha-blockade established (never beta-first → paradoxical hypertension); "
                "intraoperative management: phentolamine/nitroprusside for hypertensive crises."
            ),
        },
        "pharmacological_distinctions": [
            "SDHB-malignant-PGL: Sunitinib (VEGFR/PDGFR TKI) — FIRSTMAPPP Phase 2/3 ORR 36%; standard for progressive SDHB malignant PGL; side effects: hand-foot syndrome, hypertension, hypothyroidism, hepatotoxicity; QTc monitoring (sunitinib + pheo = QTc risk); Lu-177-DOTATATE PRRT for DOTATATE-avid disease",
            "RET-MEN2-MTC: Vandetanib (RET/VEGFR2/EGFR inhibitor) or Cabozantinib (RET/MET/VEGFR2) for progressive metastatic MTC; pralsetinib and selpercatinib (highly selective RET inhibitors) superior selectivity and tolerability; QT prolongation with vandetanib/cabozantinib — ECG monitoring mandatory",
            "VHL-RCC: Belzutifan (HIF2α inhibitor, Welireg) — oral, FDA 2021 for non-metastatic VHL-related RCC, hemangioblastoma, pNET; ORR 49%; anaemia universal class effect (EPO suppression); teratogenic (category X) — contraception mandatory; avoids multi-organ surgery burden in VHL",
            "NF1-plexiform-neurofibroma: Selumetinib (MEK1/2 inhibitor, Koselugo) FDA 2020 for symptomatic inoperable NF1 plexiform neurofibroma in children ≥2 years; tumour volume reduction ~20% (SPRINT trial); cardiotoxicity monitoring (ECHO annually); MPNST: selumetinib not effective — treat as sarcoma",
            "All-pheo-pre-op: Phenoxybenzamine (irreversible non-selective α-blocker) — drug of choice for pre-operative alpha-blockade; 7–14 days pre-op at escalating doses; NEVER start beta-blocker before alpha-blockade (β-blockade first → α-mediated vasoconstriction unopposed → hypertensive crisis); post-op hypotension expected from pre-operative alpha blockade — prepare IV fluids",
            "Metastatic-pheo-MIBG-avid: Iobenguane I-131 (Azedra) FDA 2018 for MIBG-avid unresectable/metastatic pheo/PGL; administer 131I-MIBG therapeutic dose in radiation isolation; myelosuppression is dose-limiting; thyroid protection with Lugol's iodine mandatory; eligibility: diagnostic MIBG scan must demonstrate uptake at all lesions",
        ],
        "key_standards": [
            "FIRSTMAPPP Trial (2021): Phase 2/3 RCT of sunitinib vs placebo in progressive malignant pheo/PGL; primary endpoint PFS HR 0.35 (p=0.0023); ORR 36% sunitinib vs 9% placebo; established sunitinib as first-line systemic therapy for progressive malignant PGL, especially SDHB",
            "LITESPARK-004 Trial (2022, NEJM): Phase 2 single-arm study of belzutifan in VHL disease; ORR 49% for RCC, 77% for hemangioblastoma, 91% for pNET; established belzutifan as standard for VHL-related non-metastatic tumours; FDA approved August 2021",
            "ENDOCRINE 2022 Guidelines (Endocrine Society) and ESMO 2023: Recommend plasma fractionated metanephrines as first-line biochemical test; DOTATATE-PET preferred over MIBG for SDH-related PGL; germline testing in ALL pheo/PGL patients <45 years or with known hereditary syndrome or bilateral/multifocal disease",
            "Kasperlik-Zaluska/AACE Guidelines — Prophylactic Thyroidectomy Timing in RET: HIGHEST risk (M918T/MEN2B) → thyroidectomy by 6 months of age; HIGH risk (C634 MEN2A variants) → by age 5; MODERATE risk → by age 5–10; calcitonin monitoring begins at age 3 months for all RET carriers; thyroidectomy before lymph node involvement = cure",
            "Benn et al. / ESE Guidance on SDH-Related Pheo (2021): Annual MRI from age 5–10 for all germline SDH carriers; DOTATATE-PET at diagnosis; SDHB: annual biochemistry from age 5; SDHD/MAX paternal-imprinting parent-of-origin mandatory before surveillance commitment; SDHA: pituitary MRI annually",
            "Lenders et al. Pheo/PGL Diagnosis European Network (ENSAT) 2014 — updated 2022: 24h urine catecholamines less sensitive than plasma metanephrines; seated 30-min rest before blood draw for plasma metanephrines; interfering drugs list: tricyclics, labetolol, acetaminophen, decongestants; clonidine suppression test for borderline normetanephrine; MN/NMN ratio guides diagnosis of secreting vs non-secreting",
            "Carney-Stratakis Syndrome / SDH-deficient GIST: Imatinib INEFFECTIVE for SDH-deficient GIST (KIT/PDGFRA WT); sunitinib modest activity; temsirolimus or regorafenib as alternatives; SDHB IHC recommended on ALL gastric GIST; SDHB IHC loss → germline SDH testing (SDHB/C/D); Carney triad (SDHC epigenetic — non-germline) → no cascade testing required",
        ],
    }
