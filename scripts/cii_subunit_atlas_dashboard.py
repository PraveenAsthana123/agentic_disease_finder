#!/usr/bin/env python3
"""CII-Subunit-Atlas — Complete 6-Gene Nuclear-Encoded Complex II (Succinate Dehydrogenase) Deficiency Atlas
SDHA · SDHB · SDHC · SDHD · SDHAF1 · SDHAF2
240-patient aggregate cohort (6 × 40, seeds 807–812)

Complex II (Succinate Dehydrogenase / SDH) facts:
  - CII catalyses: succinate → fumarate + FADH2 (TCA cycle step 4)
    AND transfers electrons: FADH2 → QH2 (ubiquinone reduction, OXPHOS chain)
  - CII is the ONLY OXPHOS complex with ALL subunits encoded by nuclear DNA (no mtDNA subunits)
  - CII links the TCA cycle and respiratory chain; smallest OXPHOS complex (~140 kDa)
  - CII structure: Fp subunit (SDHA, FAD-binding, catalytic) + Ip subunit (SDHB, 3 Fe-S clusters)
    + cytochrome b560 large anchor (SDHC) + cytochrome b560 small anchor (SDHD) + 2 assembly factors (SDHAF1, SDHAF2)
  - BIMODAL DISEASE SPECTRUM: Biallelic loss → early-onset neurological (Leigh syndrome / CII deficiency);
    Monoallelic haploinsufficiency → hereditary tumour predisposition (paraganglioma / pheochromocytoma)
  - Imprinting: SDHD and SDHAF2 are maternally imprinted (only paternal transmission causes disease)
  - Succinate accumulation in SDH-deficient tumours → HIF1α pseudo-hypoxia → oncogenesis (succinate = oncometabolite)
  - IHC: SDHB immunostaining loss distinguishes ALL SDH-deficient tumours (regardless of which gene mutated)

ATLAS SCOPE (6 nuclear-encoded CII/SDH genes):
  Catalytic dimer (matrix-facing):
    SDHA — Fp (flavoprotein, FAD-binding, 621aa, 5p15.33)
    SDHB — Ip (iron-sulfur protein, 3 Fe-S clusters, 280aa, 1p36.13)
  Membrane anchor dimer (transmembrane):
    SDHC — cytochrome b560 large subunit (169aa, 1q23.3)
    SDHD — cytochrome b560 small subunit (159aa, 11q23.1) — MATERNALLY IMPRINTED
  Assembly factors:
    SDHAF1 — lipoate-binding assembly factor (102aa, 19q13.12) → biallelic = Leigh / leukodystrophy
    SDHAF2 — FAD insertion chaperone (166aa, 11q13.1) — MATERNALLY IMPRINTED → monoallelic = PGL2

CRITICAL CLINICAL RULES:
  1. BIMODAL SPECTRUM — biallelic (homozygous/compound het) → metabolic/neurological disease;
     monoallelic (heterozygous) → hereditary tumour predisposition (paraganglioma, phaeochromocytoma, RCC)
  2. ONLY OXPHOS complex with ALL nuclear-encoded subunits — WES detects ALL 6 genes without mtDNA panel
  3. SDHB IHC staining loss = universal SDH-tumour marker regardless of which gene is mutated;
     SDHA IHC loss is SDHAF1/SDHAF2/SDHA-specific (SDHA protein also lost in SDHA-mutant tumours)
  4. SDHD and SDHAF2 are MATERNALLY IMPRINTED — maternal carriers do NOT develop tumours;
     disease only from PATERNAL transmission; test the father when any proband has PGL
  5. SDHB variants carry highest malignant risk (~20-40% metastatic PGL/PHEO); surveillance mandatory
  6. Metformin ABSOLUTE CI in biallelic CII deficiency (SDHA/SDHAF1) — mitochondrial toxin + CI inhibitor;
     also avoid in SDH-deficient tumour patients with concurrent mitochondrial dysfunction
  7. KD CONTRAINDICATED in biallelic CII deficiency (Leigh / SDHA biallelic / SDHAF1) —
     unlike PDC deficiency where KD is primary treatment; CII block means electrons from FADH2 cannot enter chain
  8. Succinate supplementation NOT proven beneficial despite theoretical rationale (upstream precursor)
  9. VPA AVOID in biallelic CII deficiency — complex I inhibitor + worsens OXPHOS impairment
  10. Propofol AVOID PRIS — propofol infusion syndrome involves respiratory chain dysfunction
  11. SDH-deficient tumour surveillance: annual biochemistry (catecholamines/metanephrines) + imaging (MRI/DOTATATE PET)
  12. BTBGD (SLC19A3) MANDATORY EXCLUSION in any Leigh-like CII presentation before full workup

COHORT: 6 × 40 = 240 patient slots (seeds 807–812; gene-specific seeds)
"""

import random

SEED_BASE = 807

# ── All 6 nuclear-encoded Complex II / SDH genes ─────────────────────────────
CII_GENES = [
    # ── SDHA — Fp subunit, FAD-binding, AR biallelic = Leigh, AD monoallelic = PGL5 ──
    {
        "gene": "SDHA", "alias": "Fp / Flavoprotein subunit (FAD-binding, catalytic)",
        "aa": "621 aa", "kDa": "72.7 kDa",
        "gene_class": "fp_subunit_ar_leigh_ad_pgl5",
        "locus": "5p15.33", "omim_gene": 600857,
        "phenotype": "CII Deficiency Leigh syndrome (biallelic AR) · PGL5 hereditary paraganglioma (monoallelic AD, rare)",
        "disease": "Biallelic SDHA → Complex II deficiency / Leigh syndrome (OMIM #252011) — SDHA encodes the flavoprotein (Fp) subunit; 621aa; contains the covalently-bound FAD cofactor (His99); catalytic site for succinate oxidation; biallelic loss → <20% CII residual activity → classic Leigh syndrome (bilateral basal ganglia + brainstem MRI, lactic acidosis, hypotonia, epilepsy, developmental regression); L/P ratio >25 (citric-acid OXPHOS pattern, UNLIKE PDC deficiency); HCM 55%; bimodal inheritance: biallelic AR → disease; monoallelic AD → PGL5 (rare paraganglioma, Baysal 2002; < SDHB/C/D in penetrance); Bourgeron 1995 first CII deficiency in humans (cytochrome b560 + Leigh).",
        "inheritance": "Biallelic AR (Leigh syndrome / CII deficiency) — all published Leigh-CII patients are biallelic SDHA; monoallelic AD (heterozygous) → PGL5 (paraganglioma syndrome type 5, rare). Full recessive for neurological disease; haploinsufficiency → rare tumour predisposition.",
        "hallmark": "SDHA = CATALYTIC HEART of CII — covalently-bound FAD (His99 attachment) + succinate-binding site; biallelic SDHA loss → NO CII catalytic function → Leigh syndrome with FULL OXPHOS pattern (L/P >25); distinguishes from PDC deficiency (L/P <25); CII enzyme activity <20% in fibroblasts/muscle confirms; HCM in 55% (myocardium most CII-dependent organ); SDHB protein ALSO lost by IHC in SDHA-mutant tumours (whole complex destabilised); SDHA IHC additionally lost; BTBGD exclusion mandatory before CII workup; thiamine trial NOT beneficial (FAD not TPP).",
        "key_ddx": "vs SDHAF1 (other biallelic Leigh CII gene): SDHAF1 earlier neonatal, leukodystrophy more prominent; vs OXPHOS Leigh (SURF1/ND genes): CII-Leigh: CII specifically reduced, CI/III/IV spared; vs PDC deficiency: L/P >25 in CII, L/P <25 in PDC; vs BTBGD (SLC19A3): biotin+thiamine responsive treatable mimic; vs SDHA PGL5: monoallelic only, no neurological disease in tumour patients.",
        "founder_variant": "p.Arg451Cys (European, CII deficiency), p.Arg408Cys (Leigh), p.Trp173Cys (Leigh severe); for PGL5: p.Arg31* (French, PGL5) — all private/small-population",
        "onset_pattern": "Biallelic: neonatal to infantile (0–12 months); acute decompensations triggered by febrile illness; chronic progressive encephalopathy; HCM may precede neurological presentation.",
        "mri_pattern": "Leigh syndrome: bilateral T2 hyperintensity basal ganglia (putamen, caudate) + brainstem (periaqueductal grey, dorsal pons); MRS: lactate peak in affected regions; cortical atrophy in severe; spares spinal cord (unlike DARS2/LBSL)",
        "hepatopathy_rate": 0.08, "myopathy_rate": 0.55, "encephalopathy_rate": 0.90,
        "cpeo_rate": 0.05, "epilepsy_rate": 0.75, "ataxia_rate": 0.60,
        "lactic_ac_rate": 0.68, "hcm_rate": 0.55, "snhl_rate": 0.12,
        "renal_rate": 0.06, "gi_rate": 0.08, "cognitive_rate": 0.88, "respiratory_rate": 0.50,
        "paraganglioma_rate": 0.10, "pheochromocytoma_rate": 0.05, "rcc_rate": 0.03,
        "seed": 807,
        "biallelic_leigh": True, "monoallelic_pgl": True,
        "imprinted_maternal": False,
        "malignant_risk_pct": 5,
        "leigh_syndrome": True,
        "pgl_type": "PGL5 (monoallelic only, rare)",
        "sdha_ihc_lost": True, "sdhb_ihc_lost": True,
        "vpa_ci": "AVOID — complex I inhibitor + worsens OXPHOS impairment in biallelic CII deficiency; prefer LEV/LCM",
        "kd_ci": True, "metformin_ci": True,
    },
    # ── SDHB — Ip subunit, Fe-S clusters, monoallelic = PGL4 (highest malignant risk) ──
    {
        "gene": "SDHB", "alias": "Ip / Iron-sulfur subunit ([2Fe-2S] + [3Fe-4S] + [4Fe-4S])",
        "aa": "280 aa", "kDa": "32.0 kDa",
        "gene_class": "ip_subunit_ad_pgl4",
        "locus": "1p36.13", "omim_gene": 185470,
        "phenotype": "PGL4 hereditary paraganglioma/phaeochromocytoma · HIGHEST malignant risk (~30%) · RCC · GIST",
        "disease": "Monoallelic SDHB → Paraganglioma Syndrome Type 4 / PGL4 (OMIM #115310) — SDHB encodes the iron-sulfur (Ip) subunit; 280aa; contains 3 Fe-S clusters ([2Fe-2S] cluster 1, [3Fe-4S] cluster 2, [4Fe-4S] cluster 3) that form the electron transfer chain from FAD (SDHA) to ubiquinone; SDHB loss → succinate accumulation → HIF1α pseudo-hypoxia → tumour initiation; PGL4 phenotype: sympathetic paraganglioma (abdominal/thoracic/pelvic) + phaeochromocytoma (adrenal) + rare head-neck PGL; HIGHEST MALIGNANT RISK of all SDH genes (30-40% metastatic disease); clear cell RCC (Linehan 2004); GIST in some; Astuti 2001 first SDHB-PGL4 report.",
        "inheritance": "Autosomal dominant, haploinsufficiency; penetrance ~25-35% by age 40; sex-equal (no imprinting); monoallelic → tumour; biallelic SDHB: not reported viable (likely lethal embryonically).",
        "hallmark": "SDHB = ELECTRON TRANSFER BRIDGE — 3 Fe-S clusters essential for QH2 production; SDHB IHC loss = universal SDH-tumour marker (ALL SDH-deficient tumours lose SDHB staining regardless of which gene mutated — most sensitive CII IHC test); SDHB ITSELF shows cytoplasmic granular IHC in normal; diffuse loss in SDH-deficient; HIGHEST MALIGNANT RISK (30-40% → metastatic paraganglioma/PHEO at extra-adrenal/retroperitoneal sites); surveillance: annual plasma/urine metanephrines + fractionated catecholamines + MRI abdomen/pelvis/chest; DOTATATE PET-CT for functional imaging.",
        "key_ddx": "vs SDHD (PGL1): SDHD multilocular head-neck, lower malignant risk; SDHB extra-adrenal, higher malignant risk; vs SDHC (PGL3): SDHC predominantly head-neck, lower risk; vs VHL (clear cell RCC): VHL no catecholamine excess; vs NF1 (PHEO): NF1 stigmata, no paraganglioma history; vs RET (MEN2): medullary thyroid cancer + parathyroid; vs MAX: rarer, young-onset bilateral pheo.",
        "founder_variant": "p.Arg46Gln (c.137GA, European founder, ~30% carriers with PGL/PHEO); p.Cys101Phe (North American); p.Ser163Pro (Italian); p.Leu139Pro (Spanish); p.Thr220Ile — many private variants",
        "onset_pattern": "Variable: 20-50 years typical presentation; younger onset (20s) raises malignant concern; median age at diagnosis ~35 years; earlier if surveillance detected.",
        "mri_pattern": "Not a neurological/Leigh MRI phenotype. Tumour imaging: abdominal/pelvic PGL as enhancing mass at para-aortic/organ of Zuckerkandl; adrenal PHEO; clear cell RCC at kidney; DOTATATE PET-CT for whole-body staging.",
        "hepatopathy_rate": 0.08, "myopathy_rate": 0.06, "encephalopathy_rate": 0.12,
        "cpeo_rate": 0.02, "epilepsy_rate": 0.06, "ataxia_rate": 0.06,
        "lactic_ac_rate": 0.12, "hcm_rate": 0.03, "snhl_rate": 0.05,
        "renal_rate": 0.35, "gi_rate": 0.18, "cognitive_rate": 0.20, "respiratory_rate": 0.10,
        "paraganglioma_rate": 0.55, "pheochromocytoma_rate": 0.28, "rcc_rate": 0.12,
        "seed": 808,
        "biallelic_leigh": False, "monoallelic_pgl": True,
        "imprinted_maternal": False,
        "malignant_risk_pct": 35,
        "leigh_syndrome": False,
        "pgl_type": "PGL4 — extra-adrenal sympathetic + PHEO (HIGHEST malignant risk ~30-40%)",
        "sdha_ihc_lost": False, "sdhb_ihc_lost": True,
        "vpa_ci": "NOT DIRECTLY CI for tumour-only phenotype; avoid in any SDH-deficient patient with concurrent mitochondrial dysfunction",
        "kd_ci": False, "metformin_ci": False,
    },
    # ── SDHC — large membrane anchor, monoallelic = PGL3 (head-neck, GIST) ──
    {
        "gene": "SDHC", "alias": "Cytochrome b560 large membrane anchor subunit",
        "aa": "169 aa", "kDa": "15.0 kDa",
        "gene_class": "cytb_large_ad_pgl3",
        "locus": "1q23.3", "omim_gene": 602413,
        "phenotype": "PGL3 hereditary paraganglioma · Head-neck predominant · GIST risk · Lower malignant risk",
        "disease": "Monoallelic SDHC → Paraganglioma Syndrome Type 3 / PGL3 (OMIM #605373) — SDHC encodes the large membrane anchor subunit of CII (cytochrome b560 large); 169aa; 3 transmembrane helices; anchors the catalytic SDHA-SDHB dimer to the IMM; coordinates ubiquinone-binding site with SDHD; SDHC loss → complex destabilised → SDHB IHC loss; PGL3 phenotype: predominantly HEAD-NECK paraganglioma (glomus jugulare, glomus tympanicum, carotid body tumours) at 60-70%; sympathetic PGL + PHEO rare (8%); GIST (gastric GIST, succinate-deficient) in 10-20%; lower malignant risk than SDHB; Niemann 2008 PGL3 characterisation.",
        "inheritance": "Autosomal dominant, haploinsufficiency; no maternal imprinting (unlike SDHD/SDHAF2); penetrance ~15-25% by age 50; sex-equal.",
        "hallmark": "SDHC = LARGE MEMBRANE ANCHOR — coordinates ubiquinone-binding with SDHD; predominantly head-neck PGL (head-neck paraganglia: carotid body, jugular foramen, middle ear, vagal body); GIST is a distinctive SDHC/SDHD/SDHB association (succinate-deficient GIST, Carney-Stratakis if SDHC/SDHD); cranial nerve involvement from locally invasive head-neck PGL (CN IX-XII, Horner syndrome); SDHB IHC loss positive; SDHA IHC retained (unlike SDHAF1/SDHAF2-mutant when SDHA may partially reduce); lower risk of malignant disease than SDHB.",
        "key_ddx": "vs SDHD (PGL1): both head-neck but SDHD imprinted (paternal only); SDHD higher penetrance; vs SDHB (PGL4): SDHB extra-adrenal/high-malignant; vs VHL (PHEO): no paraganglioma pattern; vs sporadic head-neck PGL: 30-40% of apparently sporadic HNPGLs carry SDH variants; vs Carney triad (non-hereditary): GIST + PGL + pulmonary chondroma (no SDH germline).",
        "founder_variant": "p.Arg6Trp (Italian), p.Pro31Leu (Swedish), p.Gly73Asp (US); many private variants given smaller gene size",
        "onset_pattern": "Head-neck PGL: 30-60 years typical; GIST: 20-50 years; often detected incidentally on imaging for other reasons; pulsatile tinnitus / cranial neuropathy / neck mass are presenting symptoms.",
        "mri_pattern": "Not Leigh-type. Head-neck MRI: 'salt and pepper' pattern within PGL on T1/T2 (flow voids from hypervascular tumour); carotid body tumour splays ICA/ECA (lyre sign on MRA); DOTATATE PET-CT for systemic staging.",
        "hepatopathy_rate": 0.05, "myopathy_rate": 0.04, "encephalopathy_rate": 0.06,
        "cpeo_rate": 0.02, "epilepsy_rate": 0.04, "ataxia_rate": 0.05,
        "lactic_ac_rate": 0.08, "hcm_rate": 0.02, "snhl_rate": 0.12,
        "renal_rate": 0.06, "gi_rate": 0.18, "cognitive_rate": 0.10, "respiratory_rate": 0.06,
        "paraganglioma_rate": 0.65, "pheochromocytoma_rate": 0.08, "rcc_rate": 0.04,
        "seed": 809,
        "biallelic_leigh": False, "monoallelic_pgl": True,
        "imprinted_maternal": False,
        "malignant_risk_pct": 8,
        "leigh_syndrome": False,
        "pgl_type": "PGL3 — head-neck paraganglioma predominant + GIST (lower malignant risk)",
        "sdha_ihc_lost": False, "sdhb_ihc_lost": True,
        "vpa_ci": "NOT directly CI for PGL3 tumour phenotype",
        "kd_ci": False, "metformin_ci": False,
    },
    # ── SDHD — small membrane anchor, MATERNALLY IMPRINTED, monoallelic = PGL1 ──
    {
        "gene": "SDHD", "alias": "Cytochrome b560 small membrane anchor subunit (maternally imprinted)",
        "aa": "159 aa", "kDa": "16.8 kDa",
        "gene_class": "cytb_small_ad_pgl1_imprinted",
        "locus": "11q23.1", "omim_gene": 602690,
        "phenotype": "PGL1 hereditary paraganglioma · MOST COMMON · Head-neck multilocular · MATERNAL IMPRINTING — paternal transmission only",
        "disease": "Monoallelic SDHD → Paraganglioma Syndrome Type 1 / PGL1 (OMIM #168000) — SDHD encodes the small membrane anchor subunit; 159aa; 4 transmembrane helices; forms cytochrome b560 with SDHC; coordinates ubiquinone-binding site; MATERNALLY IMPRINTED — only the paternal allele is expressed; maternal SDHD carriers do NOT develop PGL; SDHD loss (paternal) → succinate accumulation → HIF1α pseudo-hypoxia → PGL; PGL1 phenotype: MULTILOCULAR head-neck PGL (most common SDH syndrome); bilateral/multiple carotid body tumours + glomus jugulare; sympathetic PGL and PHEO in 10-15%; relatively LOW malignant risk (~5-10%); Baysal 2000 first SDHD-PGL1 discovery.",
        "inheritance": "Autosomal dominant with MATERNAL IMPRINTING — gene at 11q23.1 in an imprinted region. ONLY PATERNAL TRANSMISSION CAUSES DISEASE. Maternal carriers are protected (maternal allele silenced). Test the father when proband has PGL1/multilocular HN PGL. Maternal carrier sisters of patient are NOT at risk unless their father is also a carrier.",
        "hallmark": "SDHD = MATERNAL IMPRINTING — the single most important clinical fact; multilocular (multiple simultaneous) head-neck PGLs are the hallmark — carotid body + jugular + vagal body simultaneously; familial HNPGL with paternal family history; SDHB IHC loss; SDHA IHC retained; penetrance 50-80% in paternal transmitters; surveillance includes bilateral neck ultrasound/MRI + catecholamine biochemistry; surgery on benign head-neck PGL offers cure but risks CN IX-XII damage (glossopharyngeal, vagal, accessory, hypoglossal) — conservative observation acceptable for non-growing tumours.",
        "key_ddx": "vs SDHB (PGL4): SDHB extra-adrenal, high malignant risk, not imprinted; vs SDHC (PGL3): SDHC not imprinted, similar head-neck but monolocular; vs SDHAF2 (PGL2): SDHAF2 also imprinted (paternal only) but rarer; vs sporadic carotid body tumour: 20-30% carry SDH; vs VHL: pheo without head-neck PGL; vs multiple endocrine neoplasia: MEN2 has medullary thyroid cancer; vs NF1: NF stigmata.",
        "founder_variant": "p.Asp92Tyr (European/Iberian founder, PGL1, high penetrance); p.Leu95Pro (Dutch); p.His102Arg; c.14+1GT (splice); many private variants",
        "onset_pattern": "Typically 30-60 years; multilocular bilateral carotid body tumours are common presentation; pulsatile tinnitus (glomus jugulare); neck mass; Horner syndrome; cranial nerve palsies with large tumours.",
        "mri_pattern": "Multiple bilateral head-neck masses: carotid body (splaying ICA/ECA lyre sign), jugular foramen (glomus jugulare), middle ear (glomus tympanicum); 'salt and pepper' T1/T2 MRI; DOTATATE PET whole-body staging.",
        "hepatopathy_rate": 0.04, "myopathy_rate": 0.04, "encephalopathy_rate": 0.06,
        "cpeo_rate": 0.02, "epilepsy_rate": 0.04, "ataxia_rate": 0.05,
        "lactic_ac_rate": 0.07, "hcm_rate": 0.02, "snhl_rate": 0.18,
        "renal_rate": 0.05, "gi_rate": 0.10, "cognitive_rate": 0.10, "respiratory_rate": 0.06,
        "paraganglioma_rate": 0.75, "pheochromocytoma_rate": 0.12, "rcc_rate": 0.04,
        "seed": 810,
        "biallelic_leigh": False, "monoallelic_pgl": True,
        "imprinted_maternal": True,
        "malignant_risk_pct": 8,
        "leigh_syndrome": False,
        "pgl_type": "PGL1 — head-neck multilocular (MOST COMMON SDH syndrome; paternal imprinting only)",
        "sdha_ihc_lost": False, "sdhb_ihc_lost": True,
        "vpa_ci": "NOT directly CI for PGL1 tumour phenotype",
        "kd_ci": False, "metformin_ci": False,
    },
    # ── SDHAF1 — assembly factor (lipoate-binding), biallelic = Leigh / leukodystrophy ──
    {
        "gene": "SDHAF1", "alias": "SDH Assembly Factor 1 (lipoate-binding, biallelic = Leigh / infantile leukodystrophy)",
        "aa": "102 aa", "kDa": "11.0 kDa",
        "gene_class": "assembly_factor_ar_leigh",
        "locus": "19q13.12", "omim_gene": 612848,
        "phenotype": "CII Deficiency Leigh syndrome / infantile leukodystrophy (biallelic AR) — earliest-onset CII Leigh",
        "disease": "Biallelic SDHAF1 → Complex II deficiency / Leigh syndrome and leukodystrophy (OMIM #252011) — SDHAF1 (also known as LYRM8) is the CII-specific assembly factor; 102aa; interacts with the Ip subunit (SDHB) and is required for [Fe-S] cluster insertion/stabilisation during CII assembly; also has lipoate-binding capacity; biallelic loss → no assembled CII → profound CII deficiency; earlier-onset than SDHA biallelic — neonatal/early infantile lactic acidosis + hypotonia; LEUKODYSTROPHY prominent (white matter T2 signal in addition to BG/brainstem Leigh pattern) — distinguishes SDHAF1 from SDHA Leigh; Ghezzi 2009 discovery; LYR motif protein superfamily member.",
        "inheritance": "Autosomal recessive (biallelic), equal sex incidence; most reported patients from consanguineous families; small number reported globally.",
        "hallmark": "SDHAF1 = Fe-S CLUSTER CHAPERONE FOR CII — loss → CII-specific deficiency without affecting CI/III/IV; LEUKODYSTROPHY SIGNAL on brain MRI (periventricular and subcortical white matter T2 hyperintensity) IN ADDITION TO Leigh BG+brainstem lesions distinguishes SDHAF1 from SDHA (which is pure Leigh); CII enzyme activity specifically reduced in fibroblasts/muscle; SDHB and SDHA protein both reduced on Western blot (whole complex collapses); NEONATAL/EARLY INFANTILE onset most typical; lactic acidosis + hypotonia at birth; no adrenal or head-neck tumours (pure assembly factor — monoallelic carriers are healthy).",
        "key_ddx": "vs SDHA biallelic: SDHA earlier described, leukodystrophy less prominent than SDHAF1; vs OXPHOS multi-complex Leigh (SURF1/CI genes): CII-specific isolated CII deficiency in SDHAF1; vs BTBGD (SLC19A3): treatable mimic — exclude first; vs PC deficiency: L/P >25 and hyperammonemia in PC; vs other leukodystrophies (LBSL/DARS2, mtARS): SDHAF1 has simultaneous Leigh + WM pattern; vs SDHA: both biallelic CII Leigh but SDHAF1 earlier + leukodystrophy signal.",
        "founder_variant": "p.Arg20Gln (Arab/consanguineous Italian kindreds, Ghezzi 2009); p.Ala98Val; few reported private variants",
        "onset_pattern": "Neonatal to early infantile (days to 3 months); most severe early presentation of CII deficiency; acute neonatal crisis with lactic acidosis + hypotonia in neonatal period.",
        "mri_pattern": "BIMODAL: (1) Leigh lesions — bilateral BG + brainstem T2 hyperintensity; (2) LEUKODYSTROPHY — periventricular and subcortical WM T2 signal (distinguishes SDHAF1 from pure SDHA Leigh); MRS: lactate in BG + WM; progressive atrophy.",
        "hepatopathy_rate": 0.10, "myopathy_rate": 0.50, "encephalopathy_rate": 0.96,
        "cpeo_rate": 0.04, "epilepsy_rate": 0.80, "ataxia_rate": 0.55,
        "lactic_ac_rate": 0.65, "hcm_rate": 0.45, "snhl_rate": 0.08,
        "renal_rate": 0.05, "gi_rate": 0.08, "cognitive_rate": 0.92, "respiratory_rate": 0.55,
        "paraganglioma_rate": 0.0, "pheochromocytoma_rate": 0.0, "rcc_rate": 0.0,
        "seed": 811,
        "biallelic_leigh": True, "monoallelic_pgl": False,
        "imprinted_maternal": False,
        "malignant_risk_pct": 0,
        "leigh_syndrome": True,
        "pgl_type": "None — SDHAF1 biallelic carriers are asymptomatic",
        "sdha_ihc_lost": True, "sdhb_ihc_lost": True,
        "vpa_ci": "AVOID — complex I inhibitor + worsens OXPHOS impairment in biallelic CII deficiency; prefer LEV/LCM/ZNS",
        "kd_ci": True, "metformin_ci": True,
    },
    # ── SDHAF2 — assembly factor (FAD insertion), MATERNALLY IMPRINTED, monoallelic = PGL2 ──
    {
        "gene": "SDHAF2", "alias": "SDH Assembly Factor 2 (FAD insertion chaperone; maternally imprinted)",
        "aa": "166 aa", "kDa": "18.7 kDa",
        "gene_class": "assembly_factor_ad_pgl2_imprinted",
        "locus": "11q13.1", "omim_gene": 613019,
        "phenotype": "PGL2 hereditary paraganglioma · Head-neck predominant · MATERNAL IMPRINTING — paternal transmission only · Rare",
        "disease": "Monoallelic SDHAF2 → Paraganglioma Syndrome Type 2 / PGL2 (OMIM #601650) — SDHAF2 (also known as SDH5) encodes the FAD insertion chaperone for SDHA (Fp subunit); 166aa; required for covalent FAD attachment to His99 of SDHA before complex assembly; without SDHAF2, SDHA cannot bind FAD → no assembled functional CII → succinate accumulates → HIF1α pseudo-hypoxia → PGL; MATERNALLY IMPRINTED (like SDHD at 11q) — only paternal transmission causes disease; PGL2 phenotype: rare, head-neck PGL (similar to PGL1/SDHD), some sympathetic PGL; Hao 2009 discovery (SDH5/SDHAF2).",
        "inheritance": "Autosomal dominant with MATERNAL IMPRINTING — SDHAF2 at 11q13.1, same chromatin territory as SDHD (11q23.1); both are maternally imprinted. ONLY PATERNAL TRANSMISSION CAUSES DISEASE. Maternal carriers of SDHAF2 do NOT develop PGL. Very rare syndrome — fewer than 50 well-documented families worldwide.",
        "hallmark": "SDHAF2 = FAD INSERTION CHAPERONE — without SDHAF2, SDHA cannot acquire its covalently-bound FAD (His99) → entire CII fails to assemble; MATERNALLY IMPRINTED (paternal only — same principle as SDHD); very rare PGL2 syndrome; head-neck PGL predominantly; SDHB IHC loss (marker of all SDH-deficient tumours); SDHA IHC also reduced (SDHA itself non-functional without FAD → reduced on IHC); biallelic SDHAF2 not reported — likely lethal embryonically (FAD insertion is an obligate early assembly step).",
        "key_ddx": "vs SDHD (PGL1): both imprinted (paternal), both head-neck — distinguish by sequencing; SDHD more common; vs SDHB (PGL4): SDHB not imprinted, high malignant risk; vs SDHC (PGL3): not imprinted; vs sporadic head-neck PGL: exclude all SDH genes; vs VHL: no head-neck PGL.",
        "founder_variant": "p.Gly78Arg (Dutch founder, Hao 2009 original family); very few additional families described globally; WES often identifies novel private variants in new PGL2 families",
        "onset_pattern": "Variable onset: typically 30-70 years; head-neck PGL presenting as neck mass, pulsatile tinnitus, cranial nerve palsy, or incidental imaging finding.",
        "mri_pattern": "Head-neck MRI: paraganglioma at carotid body / jugular foramen / vagal body; 'salt and pepper' pattern; DOTATATE PET for functional staging. No Leigh/neurological MRI pattern.",
        "hepatopathy_rate": 0.05, "myopathy_rate": 0.04, "encephalopathy_rate": 0.06,
        "cpeo_rate": 0.02, "epilepsy_rate": 0.04, "ataxia_rate": 0.04,
        "lactic_ac_rate": 0.07, "hcm_rate": 0.02, "snhl_rate": 0.14,
        "renal_rate": 0.04, "gi_rate": 0.10, "cognitive_rate": 0.08, "respiratory_rate": 0.05,
        "paraganglioma_rate": 0.68, "pheochromocytoma_rate": 0.08, "rcc_rate": 0.04,
        "seed": 812,
        "biallelic_leigh": False, "monoallelic_pgl": True,
        "imprinted_maternal": True,
        "malignant_risk_pct": 5,
        "leigh_syndrome": False,
        "pgl_type": "PGL2 — head-neck PGL (rare; paternal imprinting only; Dutch founder p.Gly78Arg)",
        "sdha_ihc_lost": True, "sdhb_ihc_lost": True,
        "vpa_ci": "NOT directly CI for PGL2 tumour phenotype",
        "kd_ci": False, "metformin_ci": False,
    },
]


def _generate_cohort(g: dict, n: int = 40) -> list:
    rng = random.Random(g["seed"])
    cohort = []
    for _ in range(n):
        cohort.append({
            "hepatopathy":         rng.random() < g["hepatopathy_rate"],
            "myopathy":            rng.random() < g["myopathy_rate"],
            "encephalopathy":      rng.random() < g["encephalopathy_rate"],
            "cpeo":                rng.random() < g["cpeo_rate"],
            "epilepsy":            rng.random() < g["epilepsy_rate"],
            "ataxia":              rng.random() < g["ataxia_rate"],
            "lactic_acidosis":     rng.random() < g["lactic_ac_rate"],
            "hcm":                 rng.random() < g["hcm_rate"],
            "snhl":                rng.random() < g["snhl_rate"],
            "renal":               rng.random() < g["renal_rate"],
            "gi_dysmotility":      rng.random() < g["gi_rate"],
            "cognitive":           rng.random() < g["cognitive_rate"],
            "respiratory_failure": rng.random() < g["respiratory_rate"],
            "paraganglioma":       rng.random() < g["paraganglioma_rate"],
            "pheochromocytoma":    rng.random() < g["pheochromocytoma_rate"],
            "rcc":                 rng.random() < g["rcc_rate"],
        })
    return cohort


def _pct(cohort: list, key: str) -> float:
    return round(sum(1 for p in cohort if p.get(key)) / len(cohort) * 100, 1)


# ── Public API functions ──────────────────────────────────────────────────────

def get_overview():
    """Return atlas-wide overview and aggregate clinical statistics."""
    all_patients = []
    for g in CII_GENES:
        all_patients.extend(_generate_cohort(g))

    total = len(all_patients)
    leigh_patients = []
    pgl_patients = []
    for g in CII_GENES:
        cohort = _generate_cohort(g)
        if g["biallelic_leigh"]:
            leigh_patients.extend(cohort)
        else:
            pgl_patients.extend(cohort)

    def agg_pct(key):
        return round(sum(1 for p in all_patients if p.get(key)) / total * 100, 1)

    def leigh_pct(key):
        if not leigh_patients:
            return 0.0
        return round(sum(1 for p in leigh_patients if p.get(key)) / len(leigh_patients) * 100, 1)

    return {
        "atlas_name": "CII-Subunit-Atlas",
        "atlas_subtitle": "Complete 6-Gene Nuclear-Encoded Complex II (Succinate Dehydrogenase) Deficiency Reference",
        "n_genes": len(CII_GENES),
        "n_patients": total,
        "seeds": "807–812",
        "description": (
            "Complex II (Succinate Dehydrogenase / SDH) is the only OXPHOS complex with ALL subunits encoded "
            "by nuclear DNA. It catalyses succinate → fumarate (TCA step 4) and transfers electrons from FADH₂ "
            "to ubiquinone (QH₂) in the respiratory chain. Defects in the six CII/SDH nuclear genes produce a "
            "BIMODAL disease spectrum: biallelic loss of SDHA or SDHAF1 causes early-onset Leigh syndrome / CII "
            "deficiency (metabolic/neurological); monoallelic haploinsufficiency of SDHB, SDHC, SDHD, or SDHAF2 "
            "causes hereditary paraganglioma and phaeochromocytoma (PGL/PHEO) syndromes (neoplastic). "
            "SDHD and SDHAF2 are maternally imprinted — only paternal transmission causes disease. "
            "SDHB carries the highest malignant risk (~30-40% metastatic PGL/PHEO). "
            "SDHB immunostaining loss on IHC identifies all six genotypes in tumour tissue."
        ),
        "complex_architecture": {
            "catalytic_dimer": "SDHA (Fp, FAD-binding, catalytic, 621aa) + SDHB (Ip, 3 Fe-S clusters [2Fe-2S]+[3Fe-4S]+[4Fe-4S], 280aa) — matrix-facing, succinate oxidation and electron transfer",
            "membrane_anchor": "SDHC (cytochrome b560 large, 169aa) + SDHD (cytochrome b560 small, 159aa, MATERNALLY IMPRINTED) — IMM anchoring, ubiquinone binding site at SDHC-SDHD interface",
            "assembly_factors": "SDHAF1 (lipoate-binding, 102aa; required for SDHB Fe-S assembly) + SDHAF2/SDH5 (FAD insertion chaperone, 166aa, MATERNALLY IMPRINTED; required for SDHA His99-FAD covalent bond)",
            "electron_path": "Succinate → FAD (SDHA His99) → [2Fe-2S](SDHB) → [3Fe-4S](SDHB) → [4Fe-4S](SDHB) → ubiquinone (Q-site at SDHC-SDHD)",
            "coenzymes": "FAD (covalently bound to SDHA His99); ubiquinone/coenzyme Q as electron acceptor; no TPP, lipoate, or NAD+ required (unlike PDC, 2-OGDC)",
        },
        "bimodal_spectrum": {
            "leigh_genes": ["SDHA (biallelic AR)", "SDHAF1 (biallelic AR)"],
            "pgl_genes": ["SDHB (PGL4, highest malignant risk)", "SDHC (PGL3, head-neck, GIST)", "SDHD (PGL1, most common, imprinted)", "SDHAF2 (PGL2, rare, imprinted)"],
            "imprinted_genes": ["SDHD (PGL1, 11q23.1)", "SDHAF2 (PGL2, 11q13.1)"],
            "key_rule": "Biallelic loss → metabolic disease (Leigh); monoallelic haploinsufficiency → tumour predisposition (PGL/PHEO). Different genes cause entirely different disease categories.",
        },
        "aggregate_clinical": {
            "encephalopathy_pct": agg_pct("encephalopathy"),
            "lactic_acidosis_pct": agg_pct("lactic_acidosis"),
            "epilepsy_pct": agg_pct("epilepsy"),
            "hcm_pct": agg_pct("hcm"),
            "ataxia_pct": agg_pct("ataxia"),
            "myopathy_pct": agg_pct("myopathy"),
            "cognitive_pct": agg_pct("cognitive"),
            "snhl_pct": agg_pct("snhl"),
            "hepatopathy_pct": agg_pct("hepatopathy"),
            "respiratory_pct": agg_pct("respiratory_failure"),
            "paraganglioma_pct": agg_pct("paraganglioma"),
            "pheochromocytoma_pct": agg_pct("pheochromocytoma"),
            "renal_rcc_pct": agg_pct("renal"),
            "leigh_encephalopathy_pct": leigh_pct("encephalopathy"),
            "leigh_epilepsy_pct": leigh_pct("epilepsy"),
            "leigh_lactic_ac_pct": leigh_pct("lactic_acidosis"),
        },
        "drug_contraindications": {
            "metformin_ci": "ABSOLUTE CI in biallelic CII deficiency (SDHA/SDHAF1 Leigh) — mitochondrial toxin (CI inhibitor) worsens OXPHOS dysfunction; avoid in all confirmed biallelic CII Leigh patients",
            "vpa_ci": "AVOID in biallelic CII deficiency — Complex I inhibitor + worsens OXPHOS impairment; prefer LEV, LCM, or ZNS for seizure control",
            "kd_ci": "CONTRAINDICATED in biallelic CII deficiency Leigh (SDHA/SDHAF1) — unlike PDC deficiency where KD is treatment; CII block prevents FADH₂ electron entry into chain; increased fat catabolism does NOT bypass CII block",
            "propofol_avoid": "PROPOFOL AVOID / CAUTION — propofol infusion syndrome (PRIS) involves respiratory chain dysfunction; use with extreme caution in biallelic CII deficiency; anaesthetic team must be aware",
            "not_applicable_pgl": "PGL/PHEO tumour phenotype genes (SDHB/C/D/AF2): standard CI rules do not apply — these patients do not have mitochondrial metabolic disease",
        },
        "ihc_guide": {
            "sdhb_ihc": "SDHB IHC loss = universal SDH-tumour marker — ALL six SDH-gene-deficient tumours lose SDHB staining (most sensitive CII IHC test); normal = granular cytoplasmic staining",
            "sdha_ihc": "SDHA IHC loss = present in SDHA-mutant tumours AND SDHAF1/SDHAF2-mutant (SDHA non-functional without FAD/assembly); SDHB/C/D-mutant tumours: SDHA IHC RETAINED",
            "protocol": "Order BOTH SDHB + SDHA IHC on any suspected SDH-deficient tumour; SDHB-lost + SDHA-retained → SDHB/C/D mutation; SDHB-lost + SDHA-lost → SDHA/SDHAF1/SDHAF2 mutation; then confirm with WES/germline panel",
        },
        "imprinting_guide": {
            "sdhd_imprinting": "SDHD (11q23.1): maternally imprinted — maternal allele silenced; only paternal allele expressed; ONLY PATERNAL TRANSMISSION CAUSES PGL1. Maternal SDHD carriers (daughter of male patient) have an SDHD variant but will NOT develop tumours unless their own father transmitted it.",
            "sdhaf2_imprinting": "SDHAF2 (11q13.1): same chromatin territory as SDHD, same maternal imprinting rule; ONLY PATERNAL TRANSMISSION CAUSES PGL2. Very rare syndrome.",
            "clinical_rule": "When proband has PGL: test the FATHER first (not mother) for SDHD/SDHAF2 if these genes are implicated. Maternal carriers may share the variant but are at no increased tumour risk.",
        },
        "wes_utility": {
            "detects_all_6": True,
            "note": "WES detects ALL 6 nuclear-encoded CII/SDH genes — no mtDNA panel needed for CII (unique among OXPHOS complexes). Confirm pathogenicity with CII enzyme activity in fibroblasts/muscle (spectrophotometric; CII: succinate:DCPIP assay). SDHB Western blot confirms complex assembly failure for SDHAF1/SDHAF2.",
        },
        "btbgd_exclusion": "MANDATORY in any Leigh-like CII presentation — exclude SLC19A3 deficiency (biotin-thiamine-responsive basal ganglia disease) before CII workup. Empiric biotin + thiamine trial in any acute Leigh-like encephalopathy, including CII-suspected cases.",
        "key_rules": {
            "lp_ratio": "L/P >25 in CII deficiency Leigh (citric-acid / OXPHOS pattern) — UNLIKE PDC deficiency (L/P <25 pyruvate-type). Both cause Leigh syndrome; L/P ratio is the critical biochemical differentiator.",
            "cii_only_nuclear": "CII is the ONLY OXPHOS complex where ALL subunits are encoded by nuclear DNA — WES is fully diagnostic without separate mtDNA analysis for CII.",
            "bimodal_treatment": "Treatment is BIMODAL: Leigh (SDHA/SDHAF1): supportive (NO KD, avoid metformin/VPA); PGL/PHEO (SDHB/C/D/AF2): surgical resection + surveillance programme + catecholamine blockade pre-op (phenoxybenzamine/doxazosin → then beta-blocker).",
            "succinate_oncometabolite": "Succinate is an oncometabolite in SDH-deficient PGL/PHEO — SDH loss → succinate accumulates → inhibits α-KG-dependent dioxygenases (including HIF prolyl hydroxylases and epigenetic TET/JHDM demethylases) → HIF1α pseudo-hypoxia + epigenetic reprogramming → tumour initiation.",
        },
    }


def get_breakdown():
    """Return per-gene detailed breakdown for all 6 CII genes."""
    results = []
    for g in CII_GENES:
        cohort = _generate_cohort(g)
        results.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "gene_class": g["gene_class"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "phenotype": g["phenotype"],
            "disease": g["disease"],
            "inheritance": g["inheritance"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "founder_variant": g["founder_variant"],
            "onset_pattern": g["onset_pattern"],
            "mri_pattern": g["mri_pattern"],
            "seed": g["seed"],
            "cohort_n": len(cohort),
            # Aggregate phenotype rates from cohort
            "hepatopathy_pct": _pct(cohort, "hepatopathy"),
            "myopathy_pct": _pct(cohort, "myopathy"),
            "encephalopathy_pct": _pct(cohort, "encephalopathy"),
            "cpeo_pct": _pct(cohort, "cpeo"),
            "epilepsy_pct": _pct(cohort, "epilepsy"),
            "ataxia_pct": _pct(cohort, "ataxia"),
            "lactic_ac_pct": _pct(cohort, "lactic_acidosis"),
            "hcm_pct": _pct(cohort, "hcm"),
            "snhl_pct": _pct(cohort, "snhl"),
            "renal_pct": _pct(cohort, "renal"),
            "gi_pct": _pct(cohort, "gi_dysmotility"),
            "cognitive_pct": _pct(cohort, "cognitive"),
            "respiratory_pct": _pct(cohort, "respiratory_failure"),
            "paraganglioma_pct": _pct(cohort, "paraganglioma"),
            "pheochromocytoma_pct": _pct(cohort, "pheochromocytoma"),
            "rcc_pct": _pct(cohort, "rcc"),
            # CII-specific flags
            "biallelic_leigh": g["biallelic_leigh"],
            "monoallelic_pgl": g["monoallelic_pgl"],
            "imprinted_maternal": g["imprinted_maternal"],
            "malignant_risk_pct": g["malignant_risk_pct"],
            "leigh_syndrome": g["leigh_syndrome"],
            "pgl_type": g["pgl_type"],
            "sdha_ihc_lost": g["sdha_ihc_lost"],
            "sdhb_ihc_lost": g["sdhb_ihc_lost"],
            "vpa_ci": g["vpa_ci"],
            "kd_ci": g["kd_ci"],
            "metformin_ci": g["metformin_ci"],
        })
    return {"genes": results}


def get_definitions():
    """Return list of key CII/SDH-specific clinical terms and definitions."""
    return [
        {
            "term": "Complex II (Succinate Dehydrogenase / SDH)",
            "definition": "The smallest OXPHOS complex (~140 kDa) embedded in the inner mitochondrial membrane. Catalyses two functions: (1) TCA cycle: succinate → fumarate + FADH₂ (TCA step 4); (2) OXPHOS chain: FADH₂ → QH₂ (ubiquinol), feeding electrons into Complex III via ubiquinone. Unlike all other OXPHOS complexes, CII is composed entirely of nuclear-encoded subunits — no mtDNA subunits. Biochemical assay: succinate:DCPIP (2,6-dichlorophenolindophenol) oxidoreductase activity; CII-specific (DCPIP reduction). Normal CII activity: 30-60 nmol/min/mg protein in fibroblasts; <20% = CII deficiency."
        },
        {
            "term": "Bimodal Disease Spectrum of CII Genes",
            "definition": "CII genes cause two entirely distinct categories of disease depending on whether both alleles or only one allele are affected: BIALLELIC LOSS → Metabolic/neurological disease (Leigh syndrome, early-onset CII deficiency). Genes: SDHA, SDHAF1. MONOALLELIC HAPLOINSUFFICIENCY → Hereditary tumour predisposition (paraganglioma/phaeochromocytoma syndromes, PGL). Genes: SDHB (PGL4), SDHC (PGL3), SDHD (PGL1), SDHAF2 (PGL2). The biallelic and monoallelic phenotypes do not overlap clinically. SDHB heterozygous patients develop PGL/PHEO, not Leigh syndrome. SDHA biallelic patients develop CII deficiency, not paraganglioma."
        },
        {
            "term": "SDHB Immunohistochemistry (IHC) — Universal SDH-Tumour Marker",
            "definition": "Loss of SDHB protein staining by immunohistochemistry identifies SDH-deficient tumours regardless of which SDH gene is mutated. In normal tissue: granular cytoplasmic staining (mitochondrial pattern). In SDH-deficient tumours: diffuse cytoplasmic staining loss (whole-complex destabilisation secondary to any SDH subunit loss). SDHB IHC is the most sensitive single marker for all 6 genotypes. Protocol: SDHA IHC should be ordered alongside SDHB IHC — SDHA loss only in SDHA-mutant / SDHAF1-mutant / SDHAF2-mutant tumours; SDHA retained in SDHB/C/D-mutant. Combined IHC guides targeted sequencing."
        },
        {
            "term": "Maternal Imprinting (SDHD, SDHAF2)",
            "definition": "SDHD (11q23.1) and SDHAF2 (11q13.1) are located in genomic regions subject to maternal imprinting — epigenetic silencing of the maternal allele through DNA methylation and histone modifications established in the germline. Only the paternal allele is expressed. Consequence: a pathogenic variant is disease-causing ONLY if inherited from the FATHER. Maternal carriers of SDHD/SDHAF2 variants do NOT develop paraganglioma (their paternal wild-type allele is expressed; the maternal variant allele is silenced). Clinical rule: when SDHD or SDHAF2 is implicated, test the father for segregation. Daughters of male PGL1/PGL2 patients who inherit the variant ARE at risk (they received it paternally)."
        },
        {
            "term": "Succinate as Oncometabolite",
            "definition": "In SDH-deficient tumours, succinate accumulates in the mitochondrial matrix and cytosol. Succinate is an alpha-ketoglutarate (α-KG) analogue that competitively inhibits: (1) HIF prolyl hydroxylases (PHDs) → HIF1α not degraded → pseudo-hypoxia response → VEGF + angiogenesis upregulation; (2) α-KG-dependent epigenetic enzymes (TET methylcytosine dioxygenases + Jumonji histone demethylases) → CpG island hypermethylation + repressive histone methylation → epigenetic reprogramming and transcriptional silencing of tumour suppressors. This makes succinate a 'oncometabolite' analogous to 2-hydroxyglutarate in IDH1/2-mutant glioma — SDH-deficient PGL/PHEO are among the most epigenetically methylated tumours known."
        },
        {
            "term": "Paraganglioma Syndromes (PGL1-5)",
            "definition": "Five hereditary paraganglioma-phaeochromocytoma syndromes caused by SDH gene mutations: PGL1 (SDHD, 11q23.1, maternally imprinted, most common); PGL2 (SDHAF2, 11q13.1, maternally imprinted, very rare); PGL3 (SDHC, 1q23.3, head-neck + GIST); PGL4 (SDHB, 1p36.13, highest malignant risk ~30-40%); PGL5 (SDHA, 5p15.33, biallelic = Leigh; monoallelic = rare PGL, low penetrance). Paragangliomas arise from chromaffin/glomus cells of the paraganglia — head-neck (vagal, carotid body, jugulotympanic) or sympathetic (organ of Zuckerkandl, adrenal medulla/PHEO). Treatment: surgical resection; catecholamine blockade pre-op for secreting tumours (phenoxybenzamine first, then beta-blocker)."
        },
        {
            "term": "SDHB PGL4 — Highest Malignant Risk",
            "definition": "Of all SDH syndromes, SDHB (PGL4) carries the highest risk of malignant (metastatic) paraganglioma/phaeochromocytoma — approximately 30-40% lifetime risk of metastatic disease. Extra-adrenal sympathetic location (retroperitoneal/para-aortic/organ of Zuckerkandl) predominates, in contrast to head-neck PGL in PGL1/PGL3. Clear cell renal cell carcinoma (ccRCC) is an additional SDHB-associated tumour (12% risk; distinguish from VHL-related RCC by SDH IHC). Surveillance: annual plasma-free metanephrines / 24h urine fractionated catecholamines + full-body MRI (with DWI) annually; DOTATATE PET-CT for baseline and recurrence staging. Genetic testing of first-degree relatives mandatory."
        },
        {
            "term": "Leigh Syndrome in CII Deficiency (SDHA, SDHAF1)",
            "definition": "Biallelic SDHA or SDHAF1 → Complex II deficiency → Leigh syndrome phenotype. Distinguishing features from OXPHOS Leigh (CI/IV-Leigh): (1) Isolated CII deficiency on enzyme panel (CI, CIII, CIV normal); (2) L/P ratio >25 (citric-acid type — same as other OXPHOS Leigh); (3) HCM: present in 50-55% (SDHA) — particularly common relative to other CII presentations; (4) SDHAF1: leukodystrophy (white matter signal) in addition to basal ganglia + brainstem Leigh lesions. Treatment: supportive; NO KD (unlike PDC Leigh where KD is treatment); avoid Metformin (CI inhibitor) and VPA; succinate supplementation has theoretical rationale but no proven clinical benefit; BTBGD (SLC19A3) mandatory exclusion."
        },
        {
            "term": "SDHAF1 Leukodystrophy — Distinguishing MRI Feature",
            "definition": "SDHAF1 (19q13.12) biallelic deficiency causes a distinctive MRI pattern combining Leigh syndrome (basal ganglia + brainstem T2 hyperintensity) WITH periventricular and subcortical white matter T2 signal abnormalities (leukodystrophy signal). This white matter involvement is atypical for 'pure' SDHA biallelic Leigh (which shows mainly basal ganglia + brainstem without prominent WM signal). The leukodystrophy component of SDHAF1 reflects that CII activity is critical for myelin maintenance (oligodendrocytes are highly oxidative). MRS shows lactate in both BG/brainstem and white matter regions."
        },
        {
            "term": "L/P Ratio in CII vs. PDC Deficiency",
            "definition": "Critical diagnostic distinction: CII deficiency (SDHA/SDHAF1 Leigh): L/P ratio >25 — CITRIC-ACID TYPE (OXPHOS pattern). Rationale: CII block impairs TCA cycle → NADH accumulates from upstream TCA → drives pyruvate→lactate (LDH reaction) → L/P ratio elevated. PDC deficiency (PDHA1/PDHB etc.): L/P ratio NORMAL or <25 — PYRUVATE TYPE. Rationale: pyruvate cannot be converted to acetyl-CoA → pyruvate accumulates equally with lactate → L/P ratio normal/low. BOTH can cause Leigh syndrome on MRI. Measure paired blood (and ideally CSF) lactate + pyruvate simultaneously on ice — delayed or improperly handled samples lose pyruvate, falsely elevating the L/P ratio."
        },
        {
            "term": "CII-Only OXPHOS Deficiency on Enzyme Panel",
            "definition": "CII deficiency (biallelic SDHA or SDHAF1) produces an isolated Complex II deficiency on spectrophotometric enzyme analysis — CI, CIII, CIV activities are NORMAL. This pattern is the biochemical fingerprint: succinate:DCPIP activity reduced (<20% of normal mean); NADH:cytochrome c reductase (CI+III), cytochrome c oxidase (CIV) normal; citrate synthase normal (mitochondrial mass preserved). Multi-complex deficiency argues against isolated SDHA/SDHAF1 and suggests mtDNA depletion (MDDS) or other nuclear-encoded OXPHOS disorders. Enzyme analysis in fibroblasts or fresh muscle biopsy (avoid frozen tissue where possible for CII)."
        },
        {
            "term": "Surgical Pre-operative Catecholamine Blockade",
            "definition": "Secreting paragangliomas and phaeochromocytomas require mandatory pre-operative catecholamine blockade to prevent hypertensive crisis and cardiac arrhythmia during surgical manipulation. Protocol: (1) Alpha-adrenergic blockade FIRST — phenoxybenzamine (non-competitive, irreversible, standard) or doxazosin (competitive, reversible, shorter duration); (2) Beta-blocker ONLY AFTER adequate alpha blockade is established (typically 10-14 days) — adding beta-blocker before alpha-blockade causes paradoxical hypertension (unopposed alpha stimulation); (3) High-salt diet + fluid loading to reverse volume contraction from catecholamine vasoconstriction. Non-secreting head-neck PGL (SDHD/SDHC): catecholamine blockade not required; confirm biochemistry before surgery."
        },
        {
            "term": "DOTATATE PET-CT for SDH-Deficient Tumours",
            "definition": "⁶⁸Ga-DOTATATE PET-CT (somatostatin receptor scintigraphy) is the preferred functional imaging modality for paraganglioma and phaeochromocytoma surveillance in SDH syndrome patients. SDH-deficient PGL/PHEO overexpress somatostatin receptors (SSTR2 especially). DOTATATE PET is superior to ¹²³I-MIBG for most SDH PGL/PHEO, particularly head-neck and extra-adrenal. Baseline whole-body DOTATATE PET at diagnosis; repeat timing based on risk stratification. MIBG retained for therapeutic use (ⁱ³¹I-MIBG therapy for metastatic SDH-deficient PHEO/PGL). MRI abdomen-pelvis-chest for anatomical staging. Annual biochemistry (plasma-free metanephrines + chromogranin A) for monitoring."
        },
        {
            "term": "BTBGD (SLC19A3) — Mandatory Exclusion Before CII Leigh Workup",
            "definition": "Biotin-Thiamine-Responsive Basal Ganglia Disease (BTBGD, SLC19A3 deficiency) is a treatable mimic of Leigh syndrome that produces identical Leigh MRI (bilateral BG + brainstem T2 signal, lactate peak on MRS) to biallelic SDHA/SDHAF1 CII deficiency. Empiric biotin (5-10 mg/day) + thiamine (300 mg/day) must be given in any acute Leigh-like encephalopathy regardless of suspected aetiology — BEFORE completing a full metabolic/genetic workup. Missing BTBGD means missing a cure. SLC19A3 sequencing should accompany empiric treatment. BTBGD exclusion is mandatory for CII Leigh, OXPHOS Leigh, PDC Leigh — any Leigh-presenting patient."
        },
    ]


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(f"Atlas: {ov['atlas_name']}")
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}")
    cl = ov['aggregate_clinical']
    print(f"Aggregate encephalopathy: {cl['encephalopathy_pct']}%")
    print(f"Aggregate paraganglioma: {cl['paraganglioma_pct']}%")
    print(f"Leigh encephalopathy: {cl['leigh_encephalopathy_pct']}%")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    defs = get_definitions()
    print(f"Definitions: {len(defs)}")
    print("OK")
