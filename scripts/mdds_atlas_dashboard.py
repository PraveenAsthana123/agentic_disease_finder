#!/usr/bin/env python3
"""MDDS-Atlas — Complete 13-Gene mtDNA Depletion Syndrome Atlas
POLG · TWNK · DGUOK · MPV17 · SUCLA2 · SUCLG1 · RRM2B · SLC25A4
TK2 · TYMP · FBXL4 · RNASEH1 · DNA2
520-patient aggregate cohort (13 × 40, seeds 787–799)

mtDNA Depletion Syndrome (MDDS) facts:
  - MDDS = quantitative reduction of mtDNA copy number in affected tissues
  - Caused by nuclear-encoded proteins maintaining mtDNA replication, repair, or dNTP supply
  - mtDNA copy number: measured by quantitative PCR (qPCR) on muscle biopsy DNA
  - WES detects all 13 nuclear genes; mtDNA copy number requires SEPARATE qPCR assay
  - Inheritance: mostly AR; SLC25A4 (ANT1) predominantly AD; RRM2B AD or AR
  - Clinical clusters: hepatocerebral, myopathic, encephalomyopathic, MNGIE, PEO/adult-onset

ATLAS SCOPE (13 nuclear-encoded mtDNA maintenance genes):
  Hepatocerebral cluster (liver + brain):
    DGUOK, MPV17, SUCLG1
  POLG (wide spectrum, hepatocerebral when early-onset):
    POLG
  Myopathic cluster:
    TK2, SLC25A4
  Encephalomyopathic cluster:
    SUCLA2, FBXL4, RRM2B
  MNGIE (GI dysmotility + thymidine phosphorylase):
    TYMP
  PEO / adult-onset multiple deletions:
    TWNK, RNASEH1, DNA2

CRITICAL CLINICAL RULES:
  1. VPA ABSOLUTE CI in POLG — fatal Alpers hepatotoxicity (know POLG status before ANY VPA)
  2. VPA HIGH RISK in ALL other 12 MDDS genes — OXPHOS dysfunction + hepatotoxicity potential
  3. Metformin ABSOLUTE CI in all MDDS — Complex I inhibitor → lactate crisis
  4. Linezolid AVOID in all MDDS — mitochondrial ribosome inhibitor
  5. Chloramphenicol AVOID in all MDDS — mitochondrial ribosome
  6. Aminoglycosides AVOID in all MDDS — SNHL additive to mitochondrial dysfunction
  7. KD CONTRAINDICATED in POLG; CAUTION in other MDDS
  8. TYMP (MNGIE): Metronidazole AVOID (accumulates due to GI dysmotility)
  9. BTBGD (SLC19A3) MANDATORY EXCLUSION before MDDS workup
  10. WES detects all 13 nuclear genes; mtDNA copy number requires SEPARATE muscle biopsy qPCR

COHORT: 13 × 40 = 520 patient slots (seeds 787–799; gene-specific seeds)
"""

import random

SEED_BASE = 787

# ── All 13 nuclear-encoded mtDNA depletion/multiple deletion syndrome genes ───
MDDS_GENES = [
    # ── Hepatocerebral POLG (wide spectrum) ─────────────────────────────────
    {
        "gene": "POLG", "alias": "POLG1 / mtDNA Polymerase Gamma catalytic subunit",
        "aa": "1350 aa", "kDa": "140.0 kDa",
        "gene_class": "hepatocerebral_polg",
        "locus": "15q26.1", "omim_gene": 174763,
        "phenotype": "Alpers-Huttenlocher / CPEO / SANDO / MIRAS / SCAE — MOST COMMON MDDS gene",
        "disease": "Alpers-Huttenlocher (OMIM #203700), adPEO1 (OMIM #157640), SANDO (OMIM #607459), MIRAS (OMIM #609286), SCAE (OMIM #612073) — POLG; Most common nuclear cause of mtDNA depletion/multiple deletions; catalytic subunit of mtDNA polymerase gamma; loss → mtDNA depletion (early-onset) or multiple deletions (adult); VPA ABSOLUTE CI → fatal hepatotoxicity in Alpers; p.Ala467Thr most common European variant; p.Trp748Ser Finnish founder",
        "inheritance": "AR (Alpers, SANDO, MIRAS, SCAE) / AD (adPEO1)",
        "hallmark": "POLG = MOST COMMON MDDS GENE; VPA ABSOLUTE CONTRAINDICATION — POLG-Alpers Huttenlocher hepatotoxicity is fatal with VPA regardless of whether diagnosis is known at time of prescription; multiple clinical phenotypes determined by variant severity + modifier alleles; Alpers: hepatocerebral, cortical+occipital seizures, liver failure; adult forms (PEO, SANDO, SCAE): ophthalmoplegia + peripheral neuropathy + ataxia + parkinsonism ± psychiatric; p.Ala467Thr most common European allele (~40% of POLG disease alleles); p.Trp748Ser Finnish/Nordic; POLG MUST be excluded before VPA prescription in ANY pediatric encephalopathy + liver disease",
        "key_ddx": "vs TWNK (PEO1): multiple deletions, no hepatopathy; vs DGUOK: neonatal liver failure, no CPEO; vs valproate toxicity (idiosyncratic): exclude POLG first; vs mtDNA large deletions (Pearson/KSS): sporadic, not AR; vs GRACILE (BCS1L): iron overload, aminoaciduria; vs Alper-like due to FARS2/PARS2 (mtARS): no mtDNA depletion in mtARS",
        "founder_variant": "p.Ala467Thr (European, ~40% alleles); p.Trp748Ser (Finnish/Nordic); p.Gly848Ser (Turkish); p.Tyr831Cys",
        "onset_pattern": "BIMODAL: infantile (Alpers: 2–4y) OR adult (PEO/SANDO: 18–40y) — variant-severity dependent",
        "mri_pattern": "Alpers: occipital cortex signal loss + thalamic signal → cortical laminar necrosis; adult CPEO/SANDO: normal or cerebellar/posterior fossa atrophy",
        "hepatopathy_rate": 0.70, "myopathy_rate": 0.60, "encephalopathy_rate": 0.75,
        "cpeo_rate": 0.55, "epilepsy_rate": 0.65, "ataxia_rate": 0.60,
        "lactic_ac_rate": 0.75, "hcm_rate": 0.10, "snhl_rate": 0.15,
        "renal_rate": 0.10, "gi_rate": 0.15, "cognitive_rate": 0.65, "respiratory_rate": 0.30,
        "seed": 787,
        "vpa_ci": "ABSOLUTE CI — fatal Alpers hepatotoxicity; NEVER give VPA without excluding POLG",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": False, "git_dysmotility": False,
    },
    # ── PEO / Adult-onset multiple deletions ─────────────────────────────────
    {
        "gene": "TWNK", "alias": "PEO1 / C10orf2 / Twinkle helicase",
        "aa": "684 aa", "kDa": "77.0 kDa",
        "gene_class": "peo_adult",
        "locus": "10q24.31", "omim_gene": 606075,
        "phenotype": "adCPEO / Kearns-Sayre-like / Infantile-onset spinocerebellar ataxia (IOSCA)",
        "disease": "adPEO2 (OMIM #609286) / IOSCA (OMIM #271245) / MDDS7 (OMIM #271245) — TWNK/C10orf2; Twinkle is the mtDNA helicase; dominant gain-of-function stalling → multiple deletions; recessive LOF → IOSCA or hepatocerebral MDDS; adCPEO: progressive ophthalmoplegia + ptosis + myopathy; no systemic disease usually; IOSCA: Finnish founder, cerebellar ataxia, epilepsy, peripheral neuropathy in infancy",
        "inheritance": "AD (adPEO2) / AR (IOSCA, hepatocerebral MDDS7)",
        "hallmark": "TWNK = mtDNA helicase — essential for replication fork opening; dominant alleles (stalling variants) → multiple deletions (adult PEO); recessive null → complete mtDNA depletion (infantile/hepatocerebral); adCPEO: bilateral progressive ophthalmoplegia in adults; no cognitive decline unless KSS features develop; IOSCA Finnish founder (p.Tyr508Cys); multiple deletions in muscle documented by Southern blot or long-range PCR",
        "key_ddx": "vs POLG (adPEO1): clinically similar; POLG VPA CI; vs SLC25A4 (ANT1/adPEO1): similar CPEO but ANT1 more cardiac; vs mtDNA large deletion (sporadic KSS): not hereditary; vs RNASEH1/DNA2: adult PEO, slower",
        "founder_variant": "p.Tyr508Cys (Finnish IOSCA founder); p.Arg334Gln (European adPEO); p.Ala318Thr (Italian adPEO)",
        "onset_pattern": "Adult (adCPEO: 20–50y) / Infantile (IOSCA: 9–18 months for ataxia)",
        "mri_pattern": "Normal in adPEO; IOSCA: cerebellar atrophy + brainstem atrophy",
        "hepatopathy_rate": 0.20, "myopathy_rate": 0.65, "encephalopathy_rate": 0.30,
        "cpeo_rate": 0.80, "epilepsy_rate": 0.25, "ataxia_rate": 0.55,
        "lactic_ac_rate": 0.40, "hcm_rate": 0.10, "snhl_rate": 0.15,
        "renal_rate": 0.05, "gi_rate": 0.10, "cognitive_rate": 0.25, "respiratory_rate": 0.20,
        "seed": 788,
        "vpa_ci": "HIGH RISK — OXPHOS dysfunction + hepatotoxicity potential; prefer LEV/LCM",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": False, "git_dysmotility": False,
    },
    # ── Hepatocerebral cluster ────────────────────────────────────────────────
    {
        "gene": "DGUOK", "alias": "dGK / Deoxyguanosine kinase",
        "aa": "277 aa", "kDa": "30.3 kDa",
        "gene_class": "hepatocerebral",
        "locus": "2p13.1", "omim_gene": 601465,
        "phenotype": "Hepatocerebral MDDS — neonatal liver failure + hypotonia + neurodegeneration",
        "disease": "MDDS3 (OMIM #251880) — DGUOK; Hakonen 2007; mitochondrial deoxyguanosine kinase salvages dGuo/dAdo → dGTP/dATP for mtDNA synthesis; deficiency → hepatocerebral depletion; neonatal: jaundice + liver failure + hypotonia + nystagmus; Armenian/Navajo neurohepatopathy overlap phenotype; liver transplant improves hepatic but not cerebral disease; elevated AFP in neonatal form; no effective treatment beyond supportive; multisystem: liver + brain + sometimes heart",
        "inheritance": "AR",
        "hallmark": "DGUOK = HEPATOCEREBRAL MDDS, most severe neonatal form; hepatopathy is dominant (liver failure within weeks); nystagmus (90%) is a distinctive clinical sign among MDDS genes; elevated AFP marker of liver involvement; liver transplant corrects hepatic but brain disease continues (contraindication to transplant if significant neurological impairment); Armenian founder variant described",
        "key_ddx": "vs MPV17 (Navajo): Navajo founder p.Arg50Gln, similar hepatocerebral but MPV17 no MMA; vs SUCLG1: MMA prominent in SUCLG1, not DGUOK; vs POLG-Alpers: older age, cortical/occipital seizures; vs neonatal liver failure panel: GRACILE, NKH, tyrosinemia",
        "founder_variant": "p.Gly4Arg (Armenian neurohepatopathy); p.Ser65Ter (Caucasian null); p.Arg142His (compound)",
        "onset_pattern": "Neonatal (days–weeks): liver failure + hypotonia; death often <2y without transplant",
        "mri_pattern": "Cerebral atrophy + delayed myelination + BG signal; MRS lactate prominent",
        "hepatopathy_rate": 0.90, "myopathy_rate": 0.30, "encephalopathy_rate": 0.85,
        "cpeo_rate": 0.10, "epilepsy_rate": 0.55, "ataxia_rate": 0.40,
        "lactic_ac_rate": 0.90, "hcm_rate": 0.15, "snhl_rate": 0.15,
        "renal_rate": 0.10, "gi_rate": 0.15, "cognitive_rate": 0.80, "respiratory_rate": 0.50,
        "seed": 789,
        "vpa_ci": "HIGH RISK — severe hepatopathy baseline; VPA can precipitate acute liver failure; ABSOLUTE CONTRAINDICATION in liver disease context",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": True, "git_dysmotility": False,
    },
    {
        "gene": "MPV17", "alias": "MTDPS6 / Navajo Neurohepatopathy gene",
        "aa": "176 aa", "kDa": "20.3 kDa",
        "gene_class": "hepatocerebral",
        "locus": "2p23.3", "omim_gene": 137960,
        "phenotype": "Navajo Neurohepatopathy / Hepatocerebral MDDS — liver failure + peripheral neuropathy",
        "disease": "MDDS6 (OMIM #256810) — MPV17; Navajo Neurohepatopathy (NNH) is the Navajo founder phenotype (p.Arg50Gln); MPV17 is an inner mitochondrial membrane channel-like protein maintaining mitochondrial dNTP pools; hepatocerebral MDDS with liver failure + peripheral neuropathy + sensory ataxia; no MMA (distinguishes from SUCLA2/SUCLG1); liver transplant may be considered (similar caveat to DGUOK — neurological disease continues)",
        "inheritance": "AR",
        "hallmark": "MPV17 = Navajo Neurohepatopathy FOUNDER GENE (p.Arg50Gln homozygous in Navajo); inner membrane protein with unknown precise channel function; hepatocerebral with prominent peripheral neuropathy; sensory neuropathy + hepatopathy combination is distinctive; no MMA (vs SUCLA2 mild, SUCLG1 severe); liver cirrhosis develops; Navajo NNH prevalence ~1 in 1600 Navajo births",
        "key_ddx": "vs DGUOK: both hepatocerebral neonatal; DGUOK has nystagmus 90%; MPV17 has neuropathy; vs POLG-Alpers: cortical seizures, VPA CI; vs SUCLG1: MMA in SUCLG1; vs SUCLA2: MMA mild in SUCLA2, no hepatopathy",
        "founder_variant": "p.Arg50Gln (Navajo NNH founder — homozygous); p.Pro98Leu (European); p.Arg41Gln",
        "onset_pattern": "Neonatal or early infancy; progressive liver disease; neuropathy accumulates over months",
        "mri_pattern": "White matter changes + peripheral nerve involvement; cerebral atrophy",
        "hepatopathy_rate": 0.85, "myopathy_rate": 0.35, "encephalopathy_rate": 0.75,
        "cpeo_rate": 0.10, "epilepsy_rate": 0.45, "ataxia_rate": 0.50,
        "lactic_ac_rate": 0.85, "hcm_rate": 0.10, "snhl_rate": 0.20,
        "renal_rate": 0.15, "gi_rate": 0.20, "cognitive_rate": 0.70, "respiratory_rate": 0.40,
        "seed": 790,
        "vpa_ci": "HIGH RISK — severe baseline hepatopathy; ABSOLUTE CONTRAINDICATION in liver disease",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": True, "git_dysmotility": False,
    },
    {
        "gene": "SUCLG1", "alias": "SCS-G alpha / Succinyl-CoA Ligase GDP-forming alpha subunit (shared)",
        "aa": "346 aa", "kDa": "36.5 kDa",
        "gene_class": "hepatocerebral",
        "locus": "2p11.2", "omim_gene": 611224,
        "phenotype": "MDDS9 — Fatal neonatal encephalomyopathic + hepatopathy + SEVERE MMA",
        "disease": "MDDS9 (OMIM #612235) — SUCLG1; alpha subunit shared by both SCS-A (SUCLA2) and SCS-G (SCS-GTP); deficiency impairs BOTH TCA cycle arms + succinylcarnitine accumulation; SEVERE methylmalonic acidemia (MMA 500–3000 μmol/L) distinguishes from SUCLA2 (mild 10–100); hepatopathy prominent; fatal often neonatal/infantile; C4-DC elevated (succinylcarnitine); encephalomyopathic + multi-system; Ostergaard 2007",
        "inheritance": "AR",
        "hallmark": "SUCLG1 = FATAL NEONATAL MDDS with SEVERE MMA; shared alpha subunit means BOTH SCS-A and SCS-G deficient → more severe than SUCLA2 (beta only); hallmarks: MMA >500 (vs <100 in SUCLA2), hepatopathy (absent in SUCLA2), C4-DC elevated, fasting hypoglycemia (PEPCK impaired via SCS-G axis); NEVER FAST — continuous feeds mandatory; Leigh MRI in 60%",
        "key_ddx": "vs SUCLA2: SUCLG1 MMA severe + hepatopathy; SUCLA2 MMA mild, no hepatopathy; vs MUT/MMA enzyme deficiency: no mtDNA depletion; vs POLG-Alpers: cortical seizures, no MMA; vs DGUOK: no MMA in DGUOK",
        "founder_variant": "p.Arg59Trp (European); p.Ala8Pro (consanguineous Arabic); p.Phe234Leu",
        "onset_pattern": "Neonatal: lactic acidosis + MMA + hypotonia; fatal without intervention, often <1y",
        "mri_pattern": "Leigh-like (60%): bilateral BG + brainstem; cerebral atrophy",
        "hepatopathy_rate": 0.70, "myopathy_rate": 0.80, "encephalopathy_rate": 0.95,
        "cpeo_rate": 0.05, "epilepsy_rate": 0.65, "ataxia_rate": 0.40,
        "lactic_ac_rate": 0.95, "hcm_rate": 0.20, "snhl_rate": 0.40,
        "renal_rate": 0.25, "gi_rate": 0.20, "cognitive_rate": 0.90, "respiratory_rate": 0.70,
        "seed": 791,
        "vpa_ci": "HIGH RISK — severe hepatopathy + OXPHOS dysfunction; potential fatal hepatotoxicity",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": False, "git_dysmotility": False,
    },
    # ── Encephalomyopathic cluster ────────────────────────────────────────────
    {
        "gene": "SUCLA2", "alias": "SCS-A beta / Succinyl-CoA Synthetase ADP-forming beta subunit",
        "aa": "463 aa", "kDa": "50.1 kDa",
        "gene_class": "encephalomyopathic",
        "locus": "13q14.2", "omim_gene": 603921,
        "phenotype": "MDDS10 — Encephalomyopathic + SNHL + mild MMA; Faroe Islands founder",
        "disease": "MDDS10 (OMIM #612075) — SUCLA2; beta subunit of ADP-forming SCS-A only; SNHL prominent (75%) — cochlear implants effective; mild MMA (10–100 μmol/L vs SUCLG1 500–3000); NO hepatopathy (key DDx from DGUOK/MPV17/SUCLG1); Leigh MRI (80%); dystonia (70%); Faroe Islands founder p.Asp333Gly (1 in 1000 births); Ostergaard 2004",
        "inheritance": "AR",
        "hallmark": "SUCLA2 = MDDS10 ENCEPHALOMYOPATHIC; SNHL 75% (cochlear implants effective — important for quality of life); mild MMA distinguishes from MUT/MMACHC (no mtDNA depletion) and from SUCLG1 (severe MMA + hepatopathy); NO hepatopathy is critical DDx; Faroe Islands extremely high prevalence; Leigh MRI ± BG dystonia; C4-DC succinylcarnitine elevated",
        "key_ddx": "vs SUCLG1: SUCLG1 has hepatopathy + severe MMA; vs MUT/MMA enzyme: no mtDNA depletion, no SNHL; vs SUCLA2-like phenotype: C4-DC elevated in both SCS axis disorders; vs MMACHC (cblC): homocystinuria absent in SUCLA2",
        "founder_variant": "p.Asp333Gly (Faroe Islands, ~1 in 1000 births homozygous); p.Ile322Met (European); p.Ile290Phe",
        "onset_pattern": "Infancy (3–12 months): hypotonia + developmental delay + SNHL; slow progression",
        "mri_pattern": "Leigh-like BG/brainstem (80%); caudate + putamen + brainstem signal T2; lactate MRS",
        "hepatopathy_rate": 0.05, "myopathy_rate": 0.70, "encephalopathy_rate": 0.80,
        "cpeo_rate": 0.05, "epilepsy_rate": 0.55, "ataxia_rate": 0.45,
        "lactic_ac_rate": 0.80, "hcm_rate": 0.10, "snhl_rate": 0.75,
        "renal_rate": 0.10, "gi_rate": 0.10, "cognitive_rate": 0.80, "respiratory_rate": 0.45,
        "seed": 792,
        "vpa_ci": "HIGH RISK — OXPHOS dysfunction; prefer LEV; no hepatopathy but mitochondrial toxic",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": False, "git_dysmotility": False,
    },
    {
        "gene": "RRM2B", "alias": "p53R2 / Ribonucleoside diphosphate reductase subunit M2 B",
        "aa": "351 aa", "kDa": "39.9 kDa",
        "gene_class": "encephalomyopathic",
        "locus": "8q22.3", "omim_gene": 604712,
        "phenotype": "MDDS8A — Multiple deletions + encephalomyopathy + CPEO + renal tubular (AR) / PEO5 (AD)",
        "disease": "MDDS8A (OMIM #612075) / PEO5 (OMIM #613077) — RRM2B; p53-regulated small subunit of ribonucleotide reductase; provides dNTPs in post-mitotic cells (muscle, neurons); AR: severe infantile/childhood encephalomyopathy + CPEO + Fanconi syndrome (52%) + respiratory failure; AD: adult-onset PEO (PEO5) — multiple deletions only; Fanconi syndrome (renal tubular acidosis) is distinctive among MDDS genes (not in TK2, SUCLA2, SUCLG1); Bourdon 2007 NatGenet",
        "inheritance": "AR (MDDS8A encephalomyopathy) / AD (PEO5 adult)",
        "hallmark": "RRM2B = p53R2; p53-regulated dNTP supply in post-mitotic cells; AR form: ENCEPHALOMYOPATHY + FANCONI SYNDROME (renal tubular acidosis — distinctive, not in other MDDS); CPEO; respiratory failure 65%; hypotonia 100%; AD form: adult CPEO + multiple deletions only (mild); CK mildly elevated (vs TK2 very high); LEV preferred (renal excretion); propofol AVOID (PRIS); Bourdon 2007 discovery",
        "key_ddx": "vs TK2: TK2 very high CK + severe myopathy, no Fanconi; vs POLG (adPEO): VPA CI, no Fanconi; vs RTA from other causes: combine with CPEO + myopathy to suspect p53R2; vs DGUOK: hepatocerebral, no Fanconi",
        "founder_variant": "p.Arg41Trp (Bourdon 2007 original AR); p.Thr33Ala; splice c.472+2T>A (AR severe)",
        "onset_pattern": "AR: infancy–childhood; AD (PEO5): adult 20–50y",
        "mri_pattern": "Cerebral atrophy; ± white matter signal; Leigh-like in severe cases",
        "hepatopathy_rate": 0.10, "myopathy_rate": 0.80, "encephalopathy_rate": 0.75,
        "cpeo_rate": 0.60, "epilepsy_rate": 0.45, "ataxia_rate": 0.35,
        "lactic_ac_rate": 0.70, "hcm_rate": 0.10, "snhl_rate": 0.30,
        "renal_rate": 0.52, "gi_rate": 0.10, "cognitive_rate": 0.65, "respiratory_rate": 0.65,
        "seed": 793,
        "vpa_ci": "HIGH RISK — OXPHOS dysfunction; VPA hepatotoxicity risk; prefer LEV (renal excretion safe even in Fanconi with dose monitoring)",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": False, "git_dysmotility": False,
    },
    {
        "gene": "FBXL4", "alias": "F-box and leucine-rich repeat protein 4 / MDDS13",
        "aa": "783 aa", "kDa": "88.3 kDa",
        "gene_class": "encephalomyopathic",
        "locus": "6q16.1-q16.2", "omim_gene": 605654,
        "phenotype": "MDDS13 — Severe infantile encephalopathy + lactic acidosis + mtDNA depletion <20%",
        "disease": "MDDS13 (OMIM #615471) — FBXL4; Gai/Bonnen 2013 dual independent discovery; mitochondrial matrix SCF-E3 ligase adaptor regulating mitophagy; LOF → pan-OXPHOS deficiency (CI+CIII+CIV+CV all depressed); mtDNA in muscle below 20% of normal (severe depletion); lactic acidosis 100%; Leigh-like MRI 65%; no MMA (vs SUCLA2/SUCLG1); no Fanconi (vs RRM2B); consanguineous Turkish/Saudi/Egyptian families predominate",
        "inheritance": "AR",
        "hallmark": "FBXL4 = SEVERE INFANTILE MDDS13; pan-OXPHOS deficiency (CI+CIII+CIV+CV — all four respiratory complexes affected, not isolated); mtDNA muscle below 20% cardinal finding; lactic acidosis severe (100%) and persistent; no MMA (DDx from SCS axis); no Fanconi (DDx from RRM2B); no nystagmus (DDx from DGUOK); Leigh MRI in 65%; consanguineous backgrounds predominant; ACTH for infantile spasms; propofol AVOID (PRIS)",
        "key_ddx": "vs SUCLA2: MMA present in SUCLA2; vs RRM2B: Fanconi in RRM2B; vs DGUOK: nystagmus + hepatopathy in DGUOK; vs SUCLG1: severe MMA + hepatopathy in SUCLG1; vs CI-Leigh (nuclear): isolated CI, not pan-OXPHOS",
        "founder_variant": "p.Arg482Gln (Turkish/Arabic consanguineous); p.Glu500Lys; c.1315+1G>A splice",
        "onset_pattern": "Neonatal to 3 months: lactic acidosis + hypotonia + encephalopathy; fatal often <2y",
        "mri_pattern": "Leigh-like (65%): BG + brainstem bilateral symmetric; cerebral atrophy rapid",
        "hepatopathy_rate": 0.20, "myopathy_rate": 0.85, "encephalopathy_rate": 0.95,
        "cpeo_rate": 0.05, "epilepsy_rate": 0.58, "ataxia_rate": 0.30,
        "lactic_ac_rate": 1.00, "hcm_rate": 0.20, "snhl_rate": 0.15,
        "renal_rate": 0.10, "gi_rate": 0.10, "cognitive_rate": 0.90, "respiratory_rate": 0.70,
        "seed": 794,
        "vpa_ci": "HIGH RISK — pan-OXPHOS dysfunction; CoA sequestration additive; ABSOLUTE CONTRAINDICATION recommended",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": False, "git_dysmotility": False,
    },
    # ── Myopathic cluster ─────────────────────────────────────────────────────
    {
        "gene": "TK2", "alias": "mtTK / Thymidine kinase 2",
        "aa": "265 aa", "kDa": "29.4 kDa",
        "gene_class": "myopathic",
        "locus": "16q21", "omim_gene": 188250,
        "phenotype": "MDDS4 — Myopathic MDDS; severe childhood myopathy + respiratory failure; nucleoside therapy",
        "disease": "MDDS4 (OMIM #600462) — TK2; mitochondrial thymidine kinase phosphorylates dThd/dCyd for mtDNA synthesis in post-mitotic cells; LOF → myopathic depletion; severe childhood myopathy + respiratory failure; Copeland 2002; CPEO in 40%; no hepatopathy (key DDx); very high CK (distinguishes from RRM2B); emerging therapy: nucleoside supplementation (thymidine + deoxycytidine monophosphate oral) bypasses TK2 deficiency; clinical trial data positive (Garone 2014, ongoing compassionate use)",
        "inheritance": "AR",
        "hallmark": "TK2 = MYOPATHIC MDDS; very high CK (>10× ULN in most); rapid respiratory failure requiring NIV; nucleoside supplementation (dThd + dCyd or their monophosphates) is the emerging treatment — bypasses TK2 block via cytoplasmic kinase pathway; no hepatopathy; no MMA; CPEO in 40%; adult-onset milder form exists; respiratory failure = primary cause of death; proactive NIV assessment mandatory",
        "key_ddx": "vs RRM2B: CK mildly elevated in RRM2B, very high in TK2; Fanconi in RRM2B absent in TK2; vs POLG (myopathic): POLG has VPA CI; vs SMA: no OXPHOS deficiency; vs Duchenne: dystrophin, no mtDNA depletion",
        "founder_variant": "p.Ile212Asn (Copeland 2002 original); p.His90Asn; p.Ala8Val (adult-onset milder)",
        "onset_pattern": "Childhood (2–10y): proximal muscle weakness → respiratory failure; some adult-onset",
        "mri_pattern": "Normal brain; muscle MRI: diffuse signal changes in limb girdle muscles",
        "hepatopathy_rate": 0.05, "myopathy_rate": 0.95, "encephalopathy_rate": 0.25,
        "cpeo_rate": 0.40, "epilepsy_rate": 0.20, "ataxia_rate": 0.20,
        "lactic_ac_rate": 0.60, "hcm_rate": 0.15, "snhl_rate": 0.10,
        "renal_rate": 0.05, "gi_rate": 0.05, "cognitive_rate": 0.15, "respiratory_rate": 0.80,
        "seed": 795,
        "vpa_ci": "HIGH RISK — OXPHOS dysfunction; prefer LEV; avoid agents worsening respiratory failure",
        "nucleoside_therapy": True, "sct_curative": False,
        "liver_transplant": False, "git_dysmotility": False,
    },
    {
        "gene": "SLC25A4", "alias": "ANT1 / ADP/ATP translocase 1",
        "aa": "298 aa", "kDa": "32.9 kDa",
        "gene_class": "myopathic",
        "locus": "4q35.1", "omim_gene": 103220,
        "phenotype": "adPEO1 (AD) / AR cardiomyopathy-myopathy — multiple deletions; muscle + heart enriched",
        "disease": "adPEO1 (OMIM #157640) / Cardiomyopathy-myopathy (OMIM #615418) — SLC25A4/ANT1; ADP/ATP translocase 1 (ANT1) expressed at highest levels in muscle and heart; dominant gain-of-function → PEO + multiple deletions; rare AR loss-of-function → severe cardiomyopathy + myopathy; CPEO + proximal myopathy; heart involvement distinguishes from TWNK; dominant PEO: onset 20–40y; no cognitive decline; cardiac monitoring mandatory for dominant forms",
        "inheritance": "AD (adPEO1) predominantly / rare AR (cardiomyopathy-myopathy)",
        "hallmark": "SLC25A4/ANT1 = MUSCLE + HEART ENRICHED; dominant alleles → multiple deletions via dNTP imbalance (ANT1 exports ATP from mitochondria, loss disrupts nucleotide balance); cardiac involvement distinguishes ANT1 from POLG/TWNK PEO (cardiac monitoring mandatory); AR forms: severe neonatal cardiomyopathy + myopathy; Kaukonen 2000 dominant PEO discovery; p.Ala114Pro most common dominant variant",
        "key_ddx": "vs POLG (adPEO): POLG VPA CI; vs TWNK (adPEO2): less cardiac involvement in TWNK; vs MERRF (mtDNA): maternal inheritance; vs Barth syndrome (TAFAZZIN): X-linked, 3-MGA",
        "founder_variant": "p.Ala114Pro (dominant European adPEO1); p.Leu98Pro (dominant); p.Arg80His (AR severe)",
        "onset_pattern": "AD: adult 20–40y PEO; AR: neonatal/infantile cardiomyopathy",
        "mri_pattern": "Normal brain; cardiac MRI: LV hypertrophy/dilatation in cardiac forms",
        "hepatopathy_rate": 0.05, "myopathy_rate": 0.85, "encephalopathy_rate": 0.15,
        "cpeo_rate": 0.75, "epilepsy_rate": 0.10, "ataxia_rate": 0.20,
        "lactic_ac_rate": 0.45, "hcm_rate": 0.40, "snhl_rate": 0.10,
        "renal_rate": 0.05, "gi_rate": 0.05, "cognitive_rate": 0.10, "respiratory_rate": 0.30,
        "seed": 796,
        "vpa_ci": "HIGH RISK — OXPHOS dysfunction; cardiac monitoring; prefer LEV/LCM",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": False, "git_dysmotility": False,
    },
    # ── MNGIE cluster ─────────────────────────────────────────────────────────
    {
        "gene": "TYMP", "alias": "TP / Thymidine phosphorylase (MNGIE gene)",
        "aa": "482 aa", "kDa": "54.0 kDa",
        "gene_class": "mngie",
        "locus": "22q13.33", "omim_gene": 131222,
        "phenotype": "MNGIE — GI dysmotility + cachexia + CPEO + peripheral neuropathy + leukoencephalopathy",
        "disease": "MNGIE (OMIM #603041) — TYMP; Mitochondrial Neurogastrointestinal Encephalomyopathy; thymidine phosphorylase degrades thymidine/deoxyuridine; deficiency → dThd/dUrd accumulation → dNTP pool imbalance → mtDNA multiple deletions; GI dysmotility (pseudo-obstruction + borborygmi + abdominal pain) is the dominant/most disabling feature; cachexia; CPEO; peripheral neuropathy; allogeneic SCT potentially curative (restores TP enzyme in bone marrow macrophages); Metronidazole AVOID (drug accumulation via GI dysmotility impairs clearance); Nishino 1999",
        "inheritance": "AR",
        "hallmark": "TYMP/MNGIE = UNIQUE GI-DOMINANT MDDS; GI dysmotility (pseudo-obstruction) is the primary disability; cachexia results from malnutrition due to dysmotility; CPEO in most; leukoencephalopathy on brain MRI (peripheral predominant); plasma thymidine/deoxyuridine elevated (diagnostic biomarker); allogeneic hematopoietic SCT is the only potentially curative therapy (corrects enzyme in myeloid cells → normalizes plasma nucleoside levels); Metronidazole ABSOLUTE AVOID (GI dysmotility → accumulation → neurotoxicity)",
        "key_ddx": "vs POLG (GI features in SANDO): POLG no pseudo-obstruction pattern; vs MELAS: maternal inheritance, no GI dysmotility; vs GI pseudo-obstruction from other causes (Ogilvie): check plasma thymidine; vs IBD: not inflammatory; vs anorexia/malabsorption: plasma nucleosides diagnostic",
        "founder_variant": "p.Glu289Lys (Mediterranean); p.His126Tyr; p.Gln121Stop (null — severe)",
        "onset_pattern": "Young adult (10–30y): GI symptoms first → CPEO → neuropathy; progressive",
        "mri_pattern": "Leukoencephalopathy: diffuse WM signal T2; typically NOT Leigh pattern; peripheral nerve thickening on MRI nerve",
        "hepatopathy_rate": 0.10, "myopathy_rate": 0.55, "encephalopathy_rate": 0.45,
        "cpeo_rate": 0.70, "epilepsy_rate": 0.20, "ataxia_rate": 0.40,
        "lactic_ac_rate": 0.55, "hcm_rate": 0.05, "snhl_rate": 0.20,
        "renal_rate": 0.10, "gi_rate": 0.95, "cognitive_rate": 0.40, "respiratory_rate": 0.25,
        "seed": 797,
        "vpa_ci": "HIGH RISK — GI dysmotility + OXPHOS dysfunction + malabsorption complicates drug levels",
        "nucleoside_therapy": False, "sct_curative": True,
        "liver_transplant": False, "git_dysmotility": True,
    },
    # ── PEO / adult-onset multiple deletions ─────────────────────────────────
    {
        "gene": "RNASEH1", "alias": "RNase H1 / Ribonuclease H1",
        "aa": "299 aa", "kDa": "32.7 kDa",
        "gene_class": "peo_adult",
        "locus": "2p25.3", "omim_gene": 604123,
        "phenotype": "PEO + cerebellar ataxia + multiple deletions + heteroplasmy + Leigh-like",
        "disease": "RNASEH1-related PEO (OMIM #616479) — RNASEH1; RNase H1 removes RNA primers during mtDNA replication; LOF → multiple deletions + increased heteroplasmy of pathogenic mtDNA variants; clinical: CPEO + cerebellar ataxia + myopathy; Leigh-like features in severe cases; Reyes 2015; Akman 2016; adult-onset predominantly; heteroplasmy dysregulation means inherited mtDNA variants may become more apparent",
        "inheritance": "AR",
        "hallmark": "RNASEH1 = UNIQUE ROLE — removes RNA/DNA hybrids (R-loops) during mtDNA replication; deficiency → multiple deletions AND increased heteroplasmy of existing mtDNA point mutations; clinical: adult-onset CPEO + cerebellar ataxia; Leigh-like features rare but reported; brain MRI cerebellar atrophy pattern; Reyes 2015 NatGenet discovery; molecular: Southern blot multiple deletions + heteroplasmy amplification",
        "key_ddx": "vs POLG (CPEO + ataxia): POLG VPA CI; vs TWNK: IOSCA cerebellar ataxia in infancy; vs DNA2 (PEO): also adult + multiple deletions; vs mtDNA point mutations with amplification: RNASEH1 amplifies existing variants",
        "founder_variant": "p.Ala185Val (Reyes 2015 original); p.Trp47Ser; p.Val142Ile",
        "onset_pattern": "Adult (20–50y): CPEO + ataxia; slowly progressive over decades",
        "mri_pattern": "Cerebellar atrophy; ± cerebral atrophy; Leigh-like (rare, severe cases)",
        "hepatopathy_rate": 0.05, "myopathy_rate": 0.60, "encephalopathy_rate": 0.30,
        "cpeo_rate": 0.75, "epilepsy_rate": 0.20, "ataxia_rate": 0.70,
        "lactic_ac_rate": 0.40, "hcm_rate": 0.05, "snhl_rate": 0.15,
        "renal_rate": 0.05, "gi_rate": 0.05, "cognitive_rate": 0.25, "respiratory_rate": 0.25,
        "seed": 798,
        "vpa_ci": "HIGH RISK — OXPHOS dysfunction; prefer LEV/LCM for seizures",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": False, "git_dysmotility": False,
    },
    {
        "gene": "DNA2", "alias": "DNA2L / DNA replication helicase/nuclease 2",
        "aa": "1060 aa", "kDa": "119.8 kDa",
        "gene_class": "peo_adult",
        "locus": "10q21.3", "omim_gene": 601810,
        "phenotype": "PEO + progressive myopathy + multiple deletions; dual role: mtDNA repair + nuclear DNA repair",
        "disease": "DNA2-related PEO/MDDS (OMIM #615156) — DNA2; bifunctional helicase/nuclease with roles in BOTH mitochondrial (Okazaki fragment processing + mtDNA repair) AND nuclear DNA repair (DSB resection); LOF → multiple deletions; adult-onset CPEO + myopathy ± limb girdle weakness; Ronchi 2013; nuclear DNA repair role means some patients have additional nuclear instability phenotypes; largest MDDS gene (1060 aa)",
        "inheritance": "AD",
        "hallmark": "DNA2 = LARGEST MDDS GENE; DUAL NUCLEAR + MITOCHONDRIAL ROLES — mtDNA multiple deletions (mitochondrial) AND nuclear DSB resection (NHEJ/HR); dominant alleles → mtDNA multiple deletions → adult PEO + myopathy; Ronchi 2013 discovery in Italian families; adult-onset progressive proximal myopathy + CPEO; nuclear DNA repair role distinguishes DNA2 from all other MDDS genes conceptually; Southern blot shows multiple deletions",
        "key_ddx": "vs POLG (adPEO): POLG VPA CI; vs TWNK (adPEO): TWNK helicase only mtDNA; vs RNASEH1: cerebellar ataxia prominent in RNASEH1; vs SLC25A4 (ANT1): cardiac involvement in ANT1",
        "founder_variant": "p.Arg284His (Ronchi 2013 Italian); p.Leu1092Pro; p.Lys1080Glu",
        "onset_pattern": "Adult (20–50y): CPEO + proximal myopathy; slowly progressive",
        "mri_pattern": "Normal brain; muscle MRI: limb girdle muscle signal changes",
        "hepatopathy_rate": 0.05, "myopathy_rate": 0.80, "encephalopathy_rate": 0.15,
        "cpeo_rate": 0.75, "epilepsy_rate": 0.10, "ataxia_rate": 0.30,
        "lactic_ac_rate": 0.35, "hcm_rate": 0.05, "snhl_rate": 0.10,
        "renal_rate": 0.05, "gi_rate": 0.05, "cognitive_rate": 0.10, "respiratory_rate": 0.30,
        "seed": 799,
        "vpa_ci": "HIGH RISK — OXPHOS dysfunction; prefer LEV/LCM",
        "nucleoside_therapy": False, "sct_curative": False,
        "liver_transplant": False, "git_dysmotility": False,
    },
]


def _generate_cohort(gene_data):
    """Generate per-gene patient cohort deterministically using gene seed."""
    rng = random.Random(gene_data["seed"])
    n = 40
    cohort = []
    for i in range(n):
        age_onset = rng.gauss(
            5 if gene_data["gene_class"] in ("hepatocerebral", "hepatocerebral_polg", "encephalomyopathic") else
            (35 if gene_data["gene_class"] in ("peo_adult", "mngie") else 8),
            3
        )
        age_onset = max(0.1, age_onset)
        cohort.append({
            "patient_id": i + 1,
            "age_onset_y": round(age_onset, 1),
            "hepatopathy": rng.random() < gene_data["hepatopathy_rate"],
            "myopathy": rng.random() < gene_data["myopathy_rate"],
            "encephalopathy": rng.random() < gene_data["encephalopathy_rate"],
            "cpeo": rng.random() < gene_data["cpeo_rate"],
            "epilepsy": rng.random() < gene_data["epilepsy_rate"],
            "ataxia": rng.random() < gene_data["ataxia_rate"],
            "lactic_acidosis": rng.random() < gene_data["lactic_ac_rate"],
            "hcm": rng.random() < gene_data["hcm_rate"],
            "snhl": rng.random() < gene_data["snhl_rate"],
            "renal": rng.random() < gene_data["renal_rate"],
            "gi_dysmotility": rng.random() < gene_data["gi_rate"],
            "cognitive": rng.random() < gene_data["cognitive_rate"],
            "respiratory_failure": rng.random() < gene_data["respiratory_rate"],
        })
    return cohort


def _pct(cohort, field):
    """Compute integer percentage for a boolean field in a cohort."""
    return round(100 * sum(1 for p in cohort if p.get(field)) / len(cohort))


def get_overview():
    """Return aggregate overview of all 13 MDDS genes (520 patients)."""
    # Aggregate clinical rates across all genes
    all_cohorts = {g["gene"]: _generate_cohort(g) for g in MDDS_GENES}
    all_patients = [p for c in all_cohorts.values() for p in c]
    n = len(all_patients)

    def agg_pct(field):
        return round(100 * sum(1 for p in all_patients if p.get(field)) / n)

    genes_by_class = {}
    for g in MDDS_GENES:
        cls = g["gene_class"]
        genes_by_class.setdefault(cls, []).append(g["gene"])

    return {
        "atlas_name": "MDDS-Atlas — Complete 13-Gene mtDNA Depletion Syndrome Atlas",
        "n_genes": 13,
        "n_patients": 520,
        "cohort_formula": "13 genes × 40 patients = 520, seeds 787–799",
        "function": "mtDNA maintenance: replication (POLG, TWNK, DNA2, RNASEH1), dNTP supply (TK2, TYMP, DGUOK, RRM2B), nucleotide transport/metabolism (SLC25A4, MPV17), TCA cycle (SUCLA2, SUCLG1), mitophagy regulation (FBXL4)",
        "pathway": "mtDNA replication and repair — nuclear-encoded proteins maintain mitochondrial genome copy number in post-mitotic and dividing cells",
        "genes_by_class": genes_by_class,
        "aggregate_clinical": {
            "hepatopathy_pct": agg_pct("hepatopathy"),
            "myopathy_pct": agg_pct("myopathy"),
            "encephalopathy_pct": agg_pct("encephalopathy"),
            "cpeo_pct": agg_pct("cpeo"),
            "epilepsy_pct": agg_pct("epilepsy"),
            "ataxia_pct": agg_pct("ataxia"),
            "lactic_acidosis_pct": agg_pct("lactic_acidosis"),
            "snhl_pct": agg_pct("snhl"),
            "hcm_pct": agg_pct("hcm"),
            "renal_pct": agg_pct("renal"),
            "gi_dysmotility_pct": agg_pct("gi_dysmotility"),
            "respiratory_failure_pct": agg_pct("respiratory_failure"),
        },
        "drug_contraindications": {
            "vpa_polg_absolute_ci": {
                "rule": "VPA ABSOLUTE CONTRAINDICATION in POLG — fatal Alpers hepatotoxicity regardless of whether POLG diagnosis is known. POLG MUST be excluded before VPA in any pediatric encephalopathy + liver disease.",
                "alternative": "Levetiracetam (LEV), Lacosamide (LCM), Zonisamide (ZNS) — all non-mitochondrial-toxic AEDs"
            },
            "vpa_all_mdds_high_risk": "VPA HIGH RISK in ALL other 12 MDDS genes — OXPHOS dysfunction + hepatotoxicity potential; prefer non-VPA AEDs across all MDDS",
            "metformin_absolute_ci": {
                "rule": "Metformin ABSOLUTE CI in ALL 13 MDDS genes — Complex I inhibitor → impairs already-deficient OXPHOS → lactate crisis",
                "alternative": "Insulin for diabetes; GLP-1 agonists with caution; no biguanides"
            },
            "linezolid_avoid": "Linezolid AVOID in ALL 13 MDDS — inhibits mitochondrial 23S rRNA (prokaryotic origin) → worsens mt-ribosome function → additive OXPHOS failure",
            "chloramphenicol_avoid": "Chloramphenicol AVOID in ALL 13 MDDS — mt 50S ribosome inhibitor → blocks all 13 OXPHOS mt-encoded subunit synthesis",
            "aminoglycosides_avoid": {
                "rule": "Aminoglycosides AVOID in ALL 13 MDDS — mt ribosome inhibitors (prokaryotic origin); SNHL additive to mitochondrial dysfunction baseline",
                "alternative": "Beta-lactams or non-aminoglycoside antibiotics when possible"
            },
            "kd_polg_contraindicated": "Ketogenic Diet CONTRAINDICATED in POLG — worsens hepatic disease; pyruvate dehydrogenase issues; CAUTION in all other MDDS (forces beta-oxidation + OXPHOS dependence)",
            "metronidazole_tymp_avoid": "Metronidazole AVOID specifically in TYMP/MNGIE — GI dysmotility → drug accumulation → neurotoxicity",
            "propofol_avoid": "Propofol AVOID in all MDDS — PRIS (propofol infusion syndrome) risk; direct mitochondrial respiratory chain toxicity",
        },
        "special_therapies": {
            "nucleoside_supplementation_tk2": {
                "gene": "TK2",
                "therapy": "Thymidine + deoxycytidine (or TMP+dCMP) oral supplementation",
                "mechanism": "Bypasses TK2 block via cytoplasmic kinases; restores mtDNA copy number",
                "status": "Compassionate use / clinical trial positive data (Garone 2014)"
            },
            "sct_curative_mngie": {
                "gene": "TYMP",
                "therapy": "Allogeneic hematopoietic stem cell transplantation (SCT)",
                "mechanism": "Restores thymidine phosphorylase enzyme in myeloid cells → normalizes plasma thymidine/deoxyuridine → halts mtDNA damage",
                "status": "Potentially curative; best outcomes when undertaken before severe neurological involvement"
            },
            "liver_transplant_hepatocerebral": {
                "genes": ["DGUOK", "MPV17"],
                "therapy": "Liver transplantation",
                "caveat": "Corrects hepatic disease ONLY; brain/neurological disease continues — patient selection critical (exclude significant neurological impairment before transplant)",
                "status": "Available; improves hepatic survival; neurological prognosis unchanged"
            }
        },
        "key_rules": {
            "polg_vpa_absolute_ci": "POLG VPA ABSOLUTE CI — Know POLG status before prescribing VPA to ANY patient with epilepsy + liver disease + encephalopathy; fatal hepatotoxicity mechanism same as POLG-Alpers even if diagnosis not yet confirmed",
            "mtdna_qpcr_separate_assay": "WES finds the nuclear gene mutation; mtDNA copy number REQUIRES SEPARATE muscle biopsy qPCR — they are independent assays; both needed for complete diagnosis",
            "btbgd_mandatory_exclusion": "BTBGD (SLC19A3 — Biotin-thiamine-responsive basal ganglia disease) MANDATORY EXCLUSION before MDDS workup — treatable mimic; give biotin+thiamine empirically in any Leigh-like presentation",
            "wes_nuclear_all_13": "WES detects ALL 13 nuclear MDDS genes reliably; mtDNA large deletion syndromes (Pearson, KSS, single deletion) are NOT captured by WES — require dedicated mtDNA assay",
            "hepatocerebral_liver_tx_caveat": "Liver transplant for DGUOK/MPV17 DOES NOT treat brain disease — neurological disease continues; careful patient selection required",
            "tymp_plasma_biomarker": "TYMP/MNGIE: plasma thymidine + deoxyuridine elevated — specific biomarker; plasma TP enzyme activity measurable in buffy coat; DIAGNOSTIC without muscle biopsy in MNGIE",
            "tk2_nucleoside_therapy": "TK2: nucleoside supplementation (dThd+dCyd) emerging therapy — initiate early before respiratory failure; NIV assessment mandatory and proactive",
            "sucla2_snhl_cochlear": "SUCLA2/MDDS10: SNHL in 75% — cochlear implants highly effective in SUCLA2; refer audiologist early",
            "suclg1_never_fast": "SUCLG1: NEVER FAST — continuous feeds mandatory; fasting hypoglycemia from PEPCK failure can be fatal",
            "rrm2b_fanconi": "RRM2B: Fanconi syndrome (renal tubular acidosis) in 52% — unique among MDDS; monitor electrolytes, bicarbonate, phosphate; LEV safe even in renal tubular disease with dose monitoring",
        },
        "wes_utility": {
            "nuclear_genes_detected": "ALL 13 nuclear MDDS genes detected by WES/gene panel",
            "mtdna_copy_number_separate_assay": "mtDNA copy number measurement requires SEPARATE qPCR assay on muscle biopsy DNA — WES does NOT quantify copy number",
            "muscle_biopsy_preferred": "Muscle biopsy preferred tissue: OXPHOS enzymology + histochemistry (COX/SDH staining) + Southern blot (multiple deletions) + qPCR (copy number) — all required for complete characterization",
            "southern_blot_deletions": "Southern blot or long-range PCR required to detect multiple deletions in PEO-type MDDS (POLG adult, TWNK, SLC25A4, RRM2B AR, RNASEH1, DNA2)",
        },
        "polg_note": "POLG is the MOST COMMON MDDS gene. VPA ABSOLUTE CONTRAINDICATION. Multiple phenotypes: Alpers-Huttenlocher (infantile, fatal), SANDO, MIRAS, SCAE (adult). p.Ala467Thr most common European allele.",
        "tymp_note": "TYMP/MNGIE: GI dysmotility is the dominant disability. Plasma thymidine/deoxyuridine is diagnostic. Allogeneic SCT potentially curative. Metronidazole ABSOLUTE AVOID.",
        "tk2_note": "TK2 nucleoside supplementation (thymidine + deoxycytidine) bypasses the block. Initiate early. NIV is life-saving. Very high CK distinguishes from RRM2B.",
        "dguok_note": "DGUOK neonatal liver failure: nystagmus 90% is distinctive. Liver transplant corrects hepatic, not neurological. Exclude DGUOK before liver transplant if neurological signs.",
    }


def get_breakdown():
    """Return per-gene detailed breakdown for all 13 MDDS genes."""
    results = []
    for g in MDDS_GENES:
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
            # Therapy flags
            "vpa_ci": g["vpa_ci"],
            "nucleoside_therapy": g["nucleoside_therapy"],
            "sct_curative": g["sct_curative"],
            "liver_transplant": g["liver_transplant"],
            "git_dysmotility": g["git_dysmotility"],
        })
    return {"genes": results}


def get_definitions():
    """Return list of key MDDS-specific clinical terms and definitions."""
    return [
        {
            "term": "mtDNA Depletion Syndrome (MDDS)",
            "definition": "Group of disorders caused by quantitative reduction (depletion) of mitochondrial DNA copy number in affected tissues, caused by mutations in nuclear-encoded genes involved in mtDNA replication, repair, or mitochondrial dNTP supply. Distinct from mtDNA quality defects (multiple deletions) though some genes cause both."
        },
        {
            "term": "mtDNA Multiple Deletions",
            "definition": "Accumulation of multiple large-scale deletions in mitochondrial DNA, typically in post-mitotic tissues (muscle, brain). Caused by defects in mtDNA maintenance proteins (POLG, TWNK, SLC25A4, RRM2B, RNASEH1, DNA2). Distinguished from single sporadic deletion (KSS/Pearson) by inheritance pattern and Southern blot/long-read PCR pattern."
        },
        {
            "term": "Alpers-Huttenlocher Syndrome",
            "definition": "Severe infantile/early childhood manifestation of POLG deficiency: cortical + occipital seizures + liver failure + neurodegeneration + death. VPA ABSOLUTE CONTRAINDICATION — fatal hepatotoxicity. Hallmark MRI: occipital cortex signal loss progressing to cortical laminar necrosis. Most common pediatric MDDS presentation of POLG."
        },
        {
            "term": "VPA (Valproic Acid) in POLG — ABSOLUTE CONTRAINDICATION",
            "definition": "Valproic acid causes fatal hepatotoxicity in POLG-Alpers patients via a mechanism analogous to mtDNA depletion — POLG deficiency + VPA together → hepatocyte mitochondrial failure → acute liver failure. This contraindication applies WHETHER OR NOT POLG diagnosis is established. POLG must be excluded before VPA in any child with encephalopathy + liver disease + epilepsy."
        },
        {
            "term": "MNGIE (Mitochondrial Neurogastrointestinal Encephalomyopathy)",
            "definition": "Caused by TYMP (thymidine phosphorylase) deficiency. Defining features: GI dysmotility (pseudo-obstruction, borborygmi, abdominal pain, diarrhea) + cachexia + CPEO + peripheral neuropathy + leukoencephalopathy. Plasma thymidine + deoxyuridine elevated (diagnostic). Allogeneic SCT is potentially curative (restores TP enzyme via donor myeloid cells)."
        },
        {
            "term": "Nucleoside Supplementation (TK2 therapy)",
            "definition": "Emerging therapy for TK2 (thymidine kinase 2) deficiency: oral thymidine monophosphate + deoxycytidine monophosphate (or thymidine + deoxycytidine) bypasses the mitochondrial TK2 block via cytoplasmic kinases, restoring mtDNA copy number in muscle. Clinical trial data positive (Garone 2014). Best started before severe respiratory failure."
        },
        {
            "term": "mtDNA Copy Number (qPCR)",
            "definition": "Quantitative PCR measurement of mtDNA copies relative to nuclear DNA in a tissue sample. Muscle biopsy is the preferred tissue for MDDS (blood copy number is often normal or less informative). WES detects the causative nuclear gene mutation; mtDNA copy number requires a SEPARATE dedicated qPCR assay — these are independent diagnostic tests."
        },
        {
            "term": "Hepatocerebral MDDS",
            "definition": "Subgroup of MDDS with combined liver (hepatocellular failure, cholestasis, hepatomegaly) and brain (encephalopathy, seizures, neurodegeneration) involvement. Caused by DGUOK, MPV17, POLG (early-onset), SUCLG1. Liver transplant corrects hepatic disease only; neurological disease continues. Patient selection critical."
        },
        {
            "term": "CPEO (Chronic Progressive External Ophthalmoplegia)",
            "definition": "Slowly progressive bilateral ptosis and limitation of extraocular movements (ophthalmoplegia) without diplopia (due to symmetric involvement). Cardinal feature of adult-onset MDDS/multiple deletion syndromes: POLG, TWNK, SLC25A4, RRM2B, RNASEH1, DNA2. Also seen in mtDNA point mutations (MELAS, MERRF) and single deletions (KSS)."
        },
        {
            "term": "Navajo Neurohepatopathy (NNH)",
            "definition": "MPV17-related hepatocerebral MDDS with Navajo founder mutation p.Arg50Gln (homozygous). Prevalence ~1:1600 Navajo births. Features: neonatal/infant liver disease + peripheral neuropathy + sensory ataxia + leukodystrophy. No MMA (distinguishes from SUCLA2/SUCLG1). Liver transplant available but neurological disease continues."
        },
        {
            "term": "Metformin — ABSOLUTE CONTRAINDICATION in all MDDS",
            "definition": "Metformin inhibits Complex I of the mitochondrial respiratory chain (mechanism of its anti-hyperglycemic effect). In MDDS, where OXPHOS is already compromised, metformin → exacerbates Complex I deficiency → severe lactic acidosis → metabolic crisis. ABSOLUTE CONTRAINDICATION in all 13 MDDS genes. Use insulin or alternative agents for diabetes."
        },
        {
            "term": "BTBGD (Biotin-Thiamine-Responsive Basal Ganglia Disease) Mandatory Exclusion",
            "definition": "SLC19A3 (BTBGD) mutations cause a treatable Leigh-like syndrome that MIMICS MDDS clinically. Biotin + thiamine administration can be dramatically effective. BTBGD must be excluded (SLC19A3 sequencing + empiric biotin+thiamine) BEFORE pursuing invasive MDDS workup (muscle biopsy, mtDNA studies). Missing BTBGD = missed curative treatment."
        },
        {
            "term": "Fanconi Syndrome in RRM2B",
            "definition": "Renal proximal tubular dysfunction (Fanconi syndrome) in 52% of AR RRM2B patients — distinctive feature not seen in other MDDS genes. Results in aminoaciduria, glucosuria, phosphaturia, bicarbonate wasting, RTA. Manage with alkali supplementation + phosphate + electrolytes. LEV preferred AED (safe in renal disease with dose monitoring)."
        },
        {
            "term": "SUCLA2 / MDDS10 Faroe Islands Founder",
            "definition": "SUCLA2 p.Asp333Gly is a Faroe Islands founder variant with prevalence ~1 in 1000 births (heterozygous carrier ~1 in 16). Causes MDDS10: encephalomyopathic + SNHL (75%) + mild MMA (10–100 μmol/L) + NO hepatopathy. SNHL responds to cochlear implants. Distinguishes from SUCLG1 (severe MMA + hepatopathy). C4-DC (succinylcarnitine) elevated on NBS."
        },
        {
            "term": "Southern Blot / Long-Range PCR for Multiple Deletions",
            "definition": "Molecular technique to detect mtDNA multiple deletions in adult-onset MDDS (POLG adPEO, TWNK, SLC25A4, RRM2B AD, RNASEH1, DNA2). Southern blot shows smearing/additional bands below the full-length mtDNA band. Long-range PCR amplifies across deletion breakpoints. WES does NOT detect deletions — dedicated mtDNA structural assay required."
        },
    ]


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(f"Atlas: {ov['atlas_name']}")
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}")
    print(f"Aggregate hepatopathy: {ov['aggregate_clinical']['hepatopathy_pct']}%")
    print(f"Aggregate myopathy: {ov['aggregate_clinical']['myopathy_pct']}%")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    defs = get_definitions()
    print(f"Definitions: {len(defs)}")
    print("OK")
