#!/usr/bin/env python3
"""mtARS-Deficiency-Atlas — Complete 17-Gene Mitochondrial Aminoacyl-tRNA Synthetase Deficiency Atlas
AARS2 · DARS2 · EARS2 · FARS2 · HARS2 · IARS2 · LARS2 · MARS2 · NARS2 · PARS2
RARS2 · SARS2 · TARS2 · VARS2 · WARS2 · YARS2 · KARS1
680-patient aggregate cohort (17 × 40, seeds 770–786)

Mitochondrial Aminoacyl-tRNA Synthetase (mtARS) facts:
  - Mitochondrial translation requires 22 mt-tRNAs (all mitochondrial-encoded, MT-T*)
  - Each mt-tRNA must be aminoacylated by its cognate nuclear-encoded synthetase (mtARS)
  - Without correct aminoacylation → stalled mitoribosomes → reduced synthesis of 13 OXPHOS subunits
  - Result: multi-OXPHOS complex deficiency (CI, CIII, CIV most commonly reduced)
  - mtARS genes are ALL nuclear-encoded → WES detects ALL 17
  - Unlike mtDNA disorders: no heteroplasmy, no maternal inheritance — ALL AR (biallelic)
  - mtARS deficiency = 'nuclear OXPHOS gene' category — NOT mtDNA disease
  - KARS1 exception: dual cytoplasmic+mitochondrial; affects both translation systems
  - Brain/neurons are mtARS-disease-enriched because of high mitochondrial translation demand
  - MRI pattern recognition is diagnostically powerful — DARS2 (LBSL), EARS2 (LTBL), RARS2 (PCH6)

ATLAS SCOPE (17 nuclear-encoded mitochondrial aminoacyl-tRNA synthetase genes):
  Leukoencephalopathy cluster (leukodystrophy-predominant):
    DARS2, EARS2, AARS2, MARS2
  Perrault syndrome (SNHL + ovarian failure):
    HARS2, LARS2
  Epileptic encephalopathy / Alpers-like (VPA ABSOLUTE CI):
    FARS2, PARS2
  Pontocerebellar hypoplasia / cerebellar atrophy:
    RARS2, WARS2
  MLASA (myopathy + lactic acidosis + sideroblastic anemia):
    YARS2
  Multisystem / rare syndromic:
    VARS2, IARS2, SARS2, TARS2, NARS2, KARS1

CRITICAL CLINICAL RULES:
  1. VPA ABSOLUTE CI in FARS2 + PARS2 (Alpers-like hepatotoxicity — same mechanism as POLG VPA toxicity)
  2. Aminoglycosides AVOID in ALL mtARS disease (mitochondrial translation inhibitors worsening OXPHOS)
  3. MRI PATTERN IS DIAGNOSTIC — LBSL (DARS2), LTBL (EARS2), PCH6 (RARS2) are near-pathognomonic
  4. All 17 are nuclear → WES/exome detects all; mtDNA panel is NOT needed
  5. Perrault females: SNHL + POI combination → screen HARS2 + LARS2 (2 genes only) regardless of severity
  6. BTBGD (SLC19A3) MANDATORY EXCLUSION in all metabolic encephalopathy before mtARS workup

COHORT: 17 × 40 = 680 patient slots (seeds 770–786; gene-specific seeds)
"""

import random

SEED_BASE = 770

# ── All 17 nuclear-encoded mitochondrial aminoacyl-tRNA synthetase genes ──────
MTARS_GENES = [
    # ── Leukoencephalopathy cluster ─────────────────────────────────────────
    {
        "gene": "DARS2", "alias": "mtAspRS / Mitochondrial aspartyl-tRNA synthetase",
        "aa": "645 aa", "kDa": "70.8 kDa",
        "gene_class": "leukoencephalopathy",
        "tRNA_charges": "mt-tRNA-Asp (MT-TD)",
        "locus": "1q25.1", "omim_gene": 610956,
        "phenotype": "LBSL — Leukoencephalopathy with Brainstem and Spinal Cord Involvement and Lactate Signal",
        "disease": "LBSL (OMIM #611105) — DARS2; van der Knaap 2003 first description; most common mtARS disease; white matter + posterior limbs IC + brainstem + spinal cord dorsal column signal on T2-MRI; lactate peak on MRS; slowly progressive gait ataxia + spasticity onset childhood–adolescence; leukoencephalopathy with cerebellar atrophy pattern; p.Leu78His (c.228C>A) European founder splice variant; CII/CIV reduced in muscle",
        "disease_omim": 611105, "inheritance": "AR",
        "hallmark": "DARS2 = MOST COMMON mtARS DISEASE (LBSL); tri-modal MRI: (1) cerebral white matter T2 signal, (2) brainstem tegmentum + pyramids + dorsal column + cerebellar WM, (3) spinal cord dorsal columns ALL three simultaneously is pathognomonic for LBSL; lactate signal on MRS; slow progression — many patients ambulant into adulthood; p.Leu78His (splice-site) most common European variant (~60% alleles); functional severity correlates with DARS2 residual activity; CII often preserved (mild OXPHOS vs ETC translation)",
        "key_ddx": "vs ADAR1 (AGS leukodystrophy): no lactate signal; vs Alexander disease (GFAP): frontoparietal WM, no spinal cord; vs EARS2 (LTBL): thalamic signal prominent in EARS2, absent in LBSL; vs POLG (Alpers-like): cortex involved, no spinal dorsal column; vs MVID (MVI): different WM pattern; vs Krabbe (GALC): PNS involvement, different distribution",
        "founder_variant": "p.Leu78His/c.228C>A (European splice, ~60% LBSL alleles); p.Arg263Gly (Dutch); p.Thr136Ala",
        "onset_pattern": "Childhood–adolescence (walking difficulties 5–20y); slowly progressive",
        "mri_pattern": "LBSL — cerebral WM + brainstem tegmentum/pyramids + dorsal spinal cord (T2 hyperintense); lactate MRS",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": False, "leuko": True,
        "snhl_rate": 0.10, "poi_rate": 0.05, "epilepsy_rate": 0.15, "ataxia_rate": 0.80,
        "spasticity_rate": 0.75, "hcm_rate": 0.05, "lactic_ac_rate": 0.70, "myopathy_rate": 0.40,
        "cognitive_rate": 0.50, "pn_rate": 0.30, "cerebellar_atrophy_rate": 0.60,
        "median_onset_months": 120, "seed": 770,
        "vpa_ci": "HIGH RISK — mitochondrial OXPHOS compromise + hepatotoxicity risk; prefer LEV/LCM",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "EARS2", "alias": "mtGluRS / Mitochondrial glutamyl-tRNA synthetase",
        "aa": "524 aa", "kDa": "58.9 kDa",
        "gene_class": "leukoencephalopathy",
        "tRNA_charges": "mt-tRNA-Glu (MT-TE)",
        "locus": "16p12.2", "omim_gene": 612799,
        "phenotype": "LTBL — Leukodystrophy with Thalamus and Brainstem involvement and high Lactate",
        "disease": "LTBL (OMIM #614924) — EARS2; Steenweg 2012 first EARS2 disease report; BIPHASIC COURSE (hallmark): initial regression in infancy/early childhood → apparent plateau or even partial improvement in some patients; T2 thalamic + brainstem + cerebellar WM signal; lactate prominent; CI + CIII + CIV reduced; p.Gly102Asp Mediterranean/Turkish founder; leukodystrophy more severe than LBSL",
        "disease_omim": 614924, "inheritance": "AR",
        "hallmark": "EARS2 = LTBL BIPHASIC COURSE IS PATHOGNOMONIC; initial neurological crisis in infancy → regression → then plateau/partial improvement (NOT seen in any other leukodystrophy — differential point); thalamic involvement on MRI (T2 thalamic signal) distinguishes from DARS2/LBSL (no thalamic signal); EARS2 → glutamyl-mt-tRNA fails → mt-translation halted → reduced CI/CIII/CIV; p.Gly102Asp North African/Turkish/Mediterranean founder (can appear as homozygous or compound); Steenweg 2012 Brain landmark paper identified EARS2",
        "key_ddx": "vs DARS2 (LBSL): no thalamic involvement in LBSL; no biphasic improvement; vs Krabbe: peripheral nerve, different MRI; vs GFAP (Alexander): frontoparietal, no thalamic; vs Canavan (ASPA): NAA peak on MRS not lactate; vs POLG: cortical grey + cerebellar (not leukodystrophy); vs AARS2 (LONA): leuko but adult + POI",
        "founder_variant": "p.Gly102Asp/c.305G>A (Mediterranean/Turkish, ~45% EARS2 alleles); p.Phe100Leu; p.Arg72Ser",
        "onset_pattern": "Infancy–early childhood; BIPHASIC (regression then plateau); 0–2 years onset typically",
        "mri_pattern": "LTBL — cerebral WM + thalami (both) + brainstem + cerebellar WM (T2); lactate on MRS; bilateral thalami are key",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": False, "leuko": True,
        "snhl_rate": 0.20, "poi_rate": 0.05, "epilepsy_rate": 0.35, "ataxia_rate": 0.70,
        "spasticity_rate": 0.65, "hcm_rate": 0.05, "lactic_ac_rate": 0.80, "myopathy_rate": 0.45,
        "cognitive_rate": 0.75, "pn_rate": 0.15, "cerebellar_atrophy_rate": 0.55,
        "median_onset_months": 6, "seed": 771,
        "vpa_ci": "HIGH RISK — OXPHOS compromise; avoid in established LTBL; prefer LEV/VGB for infantile spasms",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "AARS2", "alias": "mtAlaRS / Mitochondrial alanyl-tRNA synthetase",
        "aa": "985 aa", "kDa": "107.5 kDa",
        "gene_class": "leukoencephalopathy",
        "tRNA_charges": "mt-tRNA-Ala (MT-TA)",
        "locus": "6p21.1", "omim_gene": 612035,
        "phenotype": "LONA (adult) / Fatal neonatal cardiomyopathy; Leukoencephalopathy, Ovarian failure, Non-syndromic",
        "disease": "LONA (OMIM #615889) — AARS2; Dallabona 2014 first human AARS2 disease; TWO DISTINCT PHENOTYPES by age: (1) NEONATAL: fatal HCM + lactic acidosis within weeks (null variants); (2) ADULT: LONA = Leukoencephalopathy + Ovarian failure + Non-syndromic (adult-onset 20-45y leukodystrophy + primary ovarian insufficiency in females; males = leukodystrophy only); CI + CIV reduced in muscle; largest nuclear-encoded mt aminoacyl-tRNA synthetase",
        "disease_omim": 615889, "inheritance": "AR",
        "hallmark": "AARS2 = TWO PHENOTYPES (age-of-onset dependent on variant severity): NEONATAL fatal cardiomyopathy (null alleles → catastrophic mt-translation failure → HCM within weeks) vs ADULT LONA (missense/hypomorphic → partial mt-translation → adult-onset leukodystrophy + POI); LONA = only mtARS disease with ovarian failure in ADULTS (distinguish from Perrault which is childhood SNHL+POI); leukodystrophy pattern: frontal WM + corpus callosum + cerebellar atrophy; LONA females: primary ovarian insufficiency → amenorrhoea → FSH/LH markedly elevated; Dallabona 2014 Cell Metab landmark",
        "key_ddx": "vs Perrault (HARS2/LARS2): SNHL + POI childhood vs LONA = leuko + POI adult (no SNHL in LONA typically); vs DARS2 (LBSL): LBSL dorsal spinal cord, LONA frontal/CC; vs POLG progressive leukodystrophy: cortical + cerebellar, different WM; vs multiple sclerosis (adults): MS plaques vs LONA diffuse WM; vs ovarian failure workup: FMR1 premutation, Turner karyotype — check AARS2 if WM signal present",
        "founder_variant": "p.Arg592Trp (neonatal-type HCM); p.Arg329His (LONA-type adult); p.Leu155Val (compound LONA)",
        "onset_pattern": "BIMODAL: neonatal HCM (0–2 months) OR adult LONA (20–45 years)",
        "mri_pattern": "LONA — frontal + parietal WM + corpus callosum thinning + cerebellar atrophy (no thalamic like EARS2, no spinal like DARS2)",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": False, "leuko": True,
        "snhl_rate": 0.10, "poi_rate": 0.55, "epilepsy_rate": 0.20, "ataxia_rate": 0.55,
        "spasticity_rate": 0.45, "hcm_rate": 0.40, "lactic_ac_rate": 0.60, "myopathy_rate": 0.35,
        "cognitive_rate": 0.60, "pn_rate": 0.20, "cerebellar_atrophy_rate": 0.50,
        "median_onset_months": 240, "seed": 772,
        "vpa_ci": "AVOID — OXPHOS compromise; HCM neonates: VPA further impairs CI/CIV → fatal",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "MARS2", "alias": "mtMetRS / Mitochondrial methionyl-tRNA synthetase",
        "aa": "423 aa", "kDa": "48.2 kDa",
        "gene_class": "leukoencephalopathy",
        "tRNA_charges": "mt-tRNA-Met (MT-TM) — initiator + elongator mt-tRNA-Met",
        "locus": "2q33.1", "omim_gene": 609728,
        "phenotype": "ARSAL — Autosomal Recessive Spastic Ataxia with Leukoencephalopathy",
        "disease": "ARSAL (OMIM #611390) — MARS2; Bayat 2012 Quebec French-Canadian cohort; MARS2 = complex copy-number variation (CNV) not simple SNV — partial tandem duplication; spastic ataxia + leukoencephalopathy (ARSAL triad); French-Canadian enrichment (Saguenay–Lac-Saint-Jean region); reduced complex I in fibroblasts; MARS2 duplication disrupts reading frame → reduced mtMetRS → mt-translation defect; CNV may be missed by standard WES",
        "disease_omim": 611390, "inheritance": "AR",
        "hallmark": "MARS2 = ARSAL (spastic ataxia + leukodystrophy) FRENCH CANADIAN FOUNDER; gene is on chromosome 2q33.1 but COMPLEX DUPLICATION — partial tandem repeat → WES may miss if CNV not detected; Bayat 2012 landmark; ARSAL: progressive spastic ataxia (cerebellar + pyramidal) + WM signal on MRI; French-Canadian enrichment: MARS2 duplication carrier frequency ~1:40 in Saguenay–Lac-Saint-Jean; CNV analysis or long-read sequencing may be needed; OXPHOS CI reduced; methionyl-tRNA charges INITIATOR mt-tRNA → all mt-translation initiation affected",
        "key_ddx": "vs Charlevoix-Saguenay (ARSACS/SACS): spastic ataxia but large SACS gene; different cerebellar MRI; no leuko; vs DARS2 (LBSL): no spastic ataxia pattern in LBSL; vs Friedreich ataxia (FXN): frataxin, iron sulfur, no WM; vs CMT: MARS2 no PNS; vs ARSACS SACS: French-Canadian but different gene",
        "founder_variant": "MARS2 tandem duplication (chr2q33.1 partial repeat; French-Canadian); p.Leu259Pro (rare missense); CNV-based",
        "onset_pattern": "Childhood–adolescence; slowly progressive spastic ataxia",
        "mri_pattern": "ARSAL — cerebral WM T2 signal + cerebellar atrophy + pyramidal tract signal",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": False, "leuko": True,
        "snhl_rate": 0.15, "poi_rate": 0.05, "epilepsy_rate": 0.20, "ataxia_rate": 0.85,
        "spasticity_rate": 0.80, "hcm_rate": 0.05, "lactic_ac_rate": 0.55, "myopathy_rate": 0.35,
        "cognitive_rate": 0.40, "pn_rate": 0.20, "cerebellar_atrophy_rate": 0.65,
        "median_onset_months": 96, "seed": 773,
        "vpa_ci": "HIGH RISK — OXPHOS compromise; prefer LEV/CBZ for seizure management",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    # ── Perrault syndrome (SNHL + POI) ─────────────────────────────────────
    {
        "gene": "HARS2", "alias": "mtHisRS / Mitochondrial histidyl-tRNA synthetase",
        "aa": "506 aa", "kDa": "56.7 kDa",
        "gene_class": "perrault",
        "tRNA_charges": "mt-tRNA-His (MT-TH)",
        "locus": "5q31.3", "omim_gene": 600783,
        "phenotype": "Perrault syndrome type 2 — SNHL + Primary Ovarian Insufficiency (females only POI)",
        "disease": "Perrault syndrome type 2 (OMIM #614926) — HARS2; Pierce 2011 first mtARS-Perrault link; SNHL (bilateral sensorineural, childhood onset) + POI (primary ovarian insufficiency in females: absent puberty/primary amenorrhoea; males = SNHL only); HARS2 is one of two mtARS genes causing Perrault (the other is LARS2); audiological ascertainment most common; POI: FSH/LH elevated, AMH undetectable; combined OXPHOS deficiency in ovarian tissue specifically",
        "disease_omim": 614926, "inheritance": "AR",
        "hallmark": "HARS2 = Perrault type 2; Perrault = SNHL + female gonadal failure (males have SNHL only — gonadal dysfunction is sex-limited to females due to ovarian mitochondrial-translation dependence); HARS2 → histidyl-mt-tRNA failure → ovarian granulosa/theca cell OXPHOS decline → follicle atresia → POI; SNHL: bilateral, symmetric, progressive, high-frequency; POI: premature ovarian failure (before 40y) — can be ascertained via FSH>25 + absent puberty + AMH<0.5; Pierce 2011 Science landmark; TWO Perrault mtARS genes: HARS2 (type 2) and LARS2 (type 4)",
        "key_ddx": "vs LARS2 (Perrault type 4): same Perrault phenotype — both SNHL+POI; HARS2 vs LARS2 only distinguishable by sequencing; vs Perrault type 1 (HSD17B4): HSD17B4 also leukodystrophy; vs Turner syndrome: monosomy X (45,X), not AR; vs FOXL2 POI: no SNHL; vs congenital hearing loss panel (GJB2/GJB6): no POI; vs AARS2 LONA: POI with adult leukodystrophy (no prominent SNHL in LONA)",
        "founder_variant": "p.Leu200Val (Pierce 2011 original family); p.Val368Leu; p.Tyr368Cys",
        "onset_pattern": "SNHL: childhood; POI: peri-pubertal to early adulthood (females); males = SNHL only",
        "mri_pattern": "Usually normal brain MRI; some may show subtle WM changes late",
        "perrault": True, "alpers_like": False, "pch": False, "mlasa": False, "leuko": False,
        "snhl_rate": 0.95, "poi_rate": 0.85, "epilepsy_rate": 0.10, "ataxia_rate": 0.25,
        "spasticity_rate": 0.10, "hcm_rate": 0.05, "lactic_ac_rate": 0.30, "myopathy_rate": 0.20,
        "cognitive_rate": 0.15, "pn_rate": 0.15, "cerebellar_atrophy_rate": 0.15,
        "median_onset_months": 60, "seed": 774,
        "vpa_ci": "AVOID — ovarian mitochondrial dependence; POI risk; prefer non-mitochondrial-toxic AEDs",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "LARS2", "alias": "mtLeuRS / Mitochondrial leucyl-tRNA synthetase",
        "aa": "903 aa", "kDa": "100.4 kDa",
        "gene_class": "perrault",
        "tRNA_charges": "mt-tRNA-Leu(UUR) (MT-TL1) + mt-tRNA-Leu(CUN) (MT-TL2) — charges both mt-Leu tRNAs",
        "locus": "3p21.3", "omim_gene": 604544,
        "phenotype": "Perrault syndrome type 4 — SNHL + Primary Ovarian Insufficiency (females only POI)",
        "disease": "Perrault syndrome type 4 (OMIM #615300) — LARS2; Solano 2014 first LARS2-Perrault; clinically identical to HARS2 Perrault (type 2): bilateral SNHL + POI females; LARS2 charges BOTH mt-tRNA-Leu(UUR) and mt-tRNA-Leu(CUN) — the two mitochondrial leucine tRNAs; leucine most frequent amino acid in OXPHOS subunits; Perrault female phenotype: primary amenorrhoea or secondary amenorrhoea before 40y; some LARS2 patients have ataxia beyond basic Perrault",
        "disease_omim": 615300, "inheritance": "AR",
        "hallmark": "LARS2 = Perrault type 4; clinically identical to HARS2 Perrault — distinguished only by gene panel; LARS2 charges 2 mt-Leu tRNAs (UUR + CUN) — leucine is the most frequent amino acid in all 13 mt-encoded OXPHOS subunits, so LARS2 LOF has broader mt-translation impact than single amino acid synthetases; some LARS2 patients have ataxia (neurological features beyond hearing+gonadal in ~30%); LARS2 also associated with Hydrops, Lactic acidosis, Sideroblastic anaemia (HLASA) when combined with severe alleles; Solano 2014 Genetics in Medicine",
        "key_ddx": "vs HARS2 (Perrault type 2): same phenotype — LARS2 may have more neurological features (ataxia); vs Perrault type 1 (HSD17B4): peroxisomal, different pathway; vs CLPP (Perrault type 3): mitochondrial protease; vs ERAL1 (Perrault type 5): mt-rRNA processing; vs Wolfram syndrome (WFS1): DM + DI + OA + SNHL — no POI pattern",
        "founder_variant": "p.Arg728Trp (Solano 2014 Spanish); p.Leu630Pro; p.Val647Leu (HLASA-severe)",
        "onset_pattern": "SNHL: childhood; POI: adolescence (females); ataxia may appear second–third decade",
        "mri_pattern": "Often normal; ataxic variants may show cerebellar atrophy",
        "perrault": True, "alpers_like": False, "pch": False, "mlasa": False, "leuko": False,
        "snhl_rate": 0.95, "poi_rate": 0.80, "epilepsy_rate": 0.15, "ataxia_rate": 0.30,
        "spasticity_rate": 0.15, "hcm_rate": 0.05, "lactic_ac_rate": 0.35, "myopathy_rate": 0.25,
        "cognitive_rate": 0.20, "pn_rate": 0.20, "cerebellar_atrophy_rate": 0.20,
        "median_onset_months": 72, "seed": 775,
        "vpa_ci": "AVOID — ovarian mitochondrial dependence; POI risk aggravated",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    # ── Epileptic encephalopathy / Alpers-like (VPA ABSOLUTE CI) ─────────────
    {
        "gene": "FARS2", "alias": "mtPheRS / Mitochondrial phenylalanyl-tRNA synthetase",
        "aa": "451 aa", "kDa": "50.8 kDa",
        "gene_class": "epilepsy_encephalopathy",
        "tRNA_charges": "mt-tRNA-Phe (MT-TF)",
        "locus": "6p25.1", "omim_gene": 611592,
        "phenotype": "Combined OXPHOS deficiency 14 — Early-onset refractory epilepsy + lactic acidosis + Alpers-like",
        "disease": "Combined OXPHOS deficiency 14 (OMIM #614946) — FARS2; Almalki 2014; early-onset severe epilepsy (seizures within first months, often neonatal); lactic acidosis; hepatopathy in some (Alpers-like); cortical atrophy + WM changes on MRI; CI + CIV reduced; refractory epilepsy despite polytherapy; VPA ABSOLUTE CI (hepatotoxicity as in POLG-Alpers); splenomegaly reported; FARS2 = second amino acid of mt-encoded proteins + most mt-mRNAs start AUG(Phe) context",
        "disease_omim": 614946, "inheritance": "AR",
        "hallmark": "FARS2 = VPA ABSOLUTE CONTRAINDICATION (Alpers-like hepatotoxicity mechanism identical to POLG — mitochondrial translation failure → reduced mt-DNA polymerase function via CI/CIV → hepatocyte mtDNA depletion under VPA); early-onset refractory epilepsy (often infantile spasms or early myoclonic encephalopathy); lactic acidosis mandatory; cortical atrophy rapid; POLG1 testing mandatory BEFORE VPA in any early-onset refractory epilepsy + lactic acidosis → extend panel to FARS2/PARS2; Almalki 2014 Brain; fatal hepatic failure reported after VPA in FARS2 compound heterozygote",
        "key_ddx": "vs POLG (Alpers-Huttenlocher): POLG overlaps clinically — both VPA-CI; POLG = mtDNA depletion + occipital seizures; FARS2 = no mtDNA depletion (mt-translation affected, not replication); vs PARS2: both VPA-CI epilepsy encephalopathy — distinguished by gene panel; vs RARS2 (PCH6): cerebellar hypoplasia on MRI; vs pyridoxine-dependent epilepsy (ALDH7A1): respond to pyridoxine; vs biotinidase deficiency: biotin-responsive",
        "founder_variant": "p.Tyr173Cys (Almalki 2014 Arabic); p.Asp325Tyr; p.Arg123Cys",
        "onset_pattern": "Neonatal or first year; refractory epilepsy; rapid cortical atrophy",
        "mri_pattern": "Cortical atrophy (often occipital) + WM changes; resembles Alpers on MRI; progressive",
        "perrault": False, "alpers_like": True, "pch": False, "mlasa": False, "leuko": False,
        "snhl_rate": 0.15, "poi_rate": 0.05, "epilepsy_rate": 0.95, "ataxia_rate": 0.60,
        "spasticity_rate": 0.65, "hcm_rate": 0.10, "lactic_ac_rate": 0.90, "myopathy_rate": 0.50,
        "cognitive_rate": 0.90, "pn_rate": 0.20, "cerebellar_atrophy_rate": 0.40,
        "median_onset_months": 3, "seed": 776,
        "vpa_ci": "ABSOLUTE CI — Alpers-like hepatic failure (same mechanism as POLG); fatal reports exist; use LEV/VGB/KD instead",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "PARS2", "alias": "mtProRS / Mitochondrial prolyl-tRNA synthetase",
        "aa": "403 aa", "kDa": "45.6 kDa",
        "gene_class": "epilepsy_encephalopathy",
        "tRNA_charges": "mt-tRNA-Pro (MT-TP)",
        "locus": "1p32.3", "omim_gene": 612036,
        "phenotype": "Combined OXPHOS deficiency — Early-onset epilepsy encephalopathy + lactic acidosis (Alpers-like)",
        "disease": "Combined OXPHOS deficiency (OMIM #616464) — PARS2; Sofou 2015 first report; phenotypically overlaps POLG-Alpers and FARS2: early-onset epileptic encephalopathy + lactic acidosis + cortical atrophy; VPA ABSOLUTE CI (same hepatotoxicity risk as POLG/FARS2); refractory epilepsy; mixed seizure types; CI + CIV reduced; PARS2 → prolyl-mt-tRNA failure → stalled mt-ribosomes at proline codons → CI/CIV subunit mis-synthesis",
        "disease_omim": 616464, "inheritance": "AR",
        "hallmark": "PARS2 = VPA ABSOLUTE CONTRAINDICATION (Alpers-like hepatotoxicity — same mechanism as POLG and FARS2); early-onset severe epilepsy + lactic acidosis + progressive cortical atrophy; Sofou 2015 landmark case series; POLG/FARS2/PARS2 = THE THREE mtARS/mtDNA-pol genes where VPA is absolutely forbidden due to Alpers-mechanism hepatic failure; proline is frequently present in OXPHOS subunit helices → PARS2 LOF particularly disruptive; distinction from FARS2: gene sequencing; both require EMERGENCY VPA discontinuation if prescribed before diagnosis",
        "key_ddx": "vs POLG (Alpers): mtDNA depletion in POLG; PARS2 = mt-translation, not replication; vs FARS2: clinically identical — only gene panel distinguishes; vs Dravet (SCN1A): not metabolic, normal lactate; vs CDKL5 encephalopathy: no metabolic component; vs pyridoxine-dependent epilepsy (ALDH7A1): biomarker pipecolic acid elevated",
        "founder_variant": "p.Arg171Cys (Sofou 2015 Scandinavian); p.Trp153Stop; p.Arg199His",
        "onset_pattern": "First year; refractory epilepsy; cortical progressive atrophy",
        "mri_pattern": "Cortical + basal ganglia involvement; progressive atrophy; similar to Alpers",
        "perrault": False, "alpers_like": True, "pch": False, "mlasa": False, "leuko": False,
        "snhl_rate": 0.15, "poi_rate": 0.05, "epilepsy_rate": 0.95, "ataxia_rate": 0.65,
        "spasticity_rate": 0.60, "hcm_rate": 0.10, "lactic_ac_rate": 0.90, "myopathy_rate": 0.50,
        "cognitive_rate": 0.90, "pn_rate": 0.20, "cerebellar_atrophy_rate": 0.45,
        "median_onset_months": 4, "seed": 777,
        "vpa_ci": "ABSOLUTE CI — Alpers-like hepatic failure; emergency discontinue if started before diagnosis; use LEV/VGB/KD",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    # ── Pontocerebellar hypoplasia / cerebellar atrophy ──────────────────────
    {
        "gene": "RARS2", "alias": "mtArgRS / Mitochondrial arginyl-tRNA synthetase",
        "aa": "562 aa", "kDa": "63.6 kDa",
        "gene_class": "pch",
        "tRNA_charges": "mt-tRNA-Arg (MT-TR)",
        "locus": "6q15", "omim_gene": 611524,
        "phenotype": "Pontocerebellar Hypoplasia Type 6 (PCH6) — Neonatal epilepsy + cerebellar hypoplasia",
        "disease": "PCH6 (OMIM #611523) — RARS2; Edvardson 2007 first RARS2 disease; MOST SEVERE mtARS phenotype; pontocerebellar hypoplasia visible on fetal/neonatal MRI — pons + cerebellar hemispheres markedly reduced; neonatal onset epilepsy + profound hypotonia; lactic acidosis; microcephaly (often progressive); CI + CIV reduced in muscle; PCH6 is the only PCH subtype caused by a nuclear-encoded mt-translation factor; rapidly fatal in most; Edvardson 2007 Am J Hum Genet",
        "disease_omim": 611523, "inheritance": "AR",
        "hallmark": "RARS2 = PCH6 — MOST SEVERE mtARS; PCH = structural brain malformation (cerebellar + pontine hypoplasia, not just atrophy — primary developmental defect from early gestation); fetal MRI can show PCH at 20+ weeks; neonatal epilepsy + profound hypotonia + microcephaly + lactic acidosis = PCH6 emergency; distinction from PCH1/2 (TSEN/SLC25A46): those are RNA processing or mitochondrial fission — check mt-tRNA panel when PCH detected; RARS2 must be considered in all PCH with lactic acidosis; fatal within months in severe cases; arginine is frequent in OXPHOS regulatory/structural loops",
        "key_ddx": "vs PCH1 (EXOAB-family): RNA exosome; vs PCH2 (TSEN): tRNA splicing endonuclease; vs PCH4 (TSEN54): severe TSEN variant; vs Joubert syndrome (JBTS-family): no pontine hypoplasia (molar tooth sign); vs Walker-Warburg (POMT1): cobblestone cortex; vs FARS2: epilepsy but no PCH; vs congenital CMV: periventricular calcification",
        "founder_variant": "p.Thr132Ser (Edvardson 2007 Israeli Arab); p.Arg245Cys; p.Leu381Pro",
        "onset_pattern": "Neonatal / fetal; cerebellar hypoplasia detectable prenatally",
        "mri_pattern": "PCH6 — pontine + cerebellar hypoplasia (bilateral cerebellar hemispheres + pons markedly reduced); progressive atrophy; microcephaly",
        "perrault": False, "alpers_like": False, "pch": True, "mlasa": False, "leuko": False,
        "snhl_rate": 0.25, "poi_rate": 0.05, "epilepsy_rate": 0.90, "ataxia_rate": 0.80,
        "spasticity_rate": 0.70, "hcm_rate": 0.10, "lactic_ac_rate": 0.90, "myopathy_rate": 0.70,
        "cognitive_rate": 0.95, "pn_rate": 0.15, "cerebellar_atrophy_rate": 0.95,
        "median_onset_months": 1, "seed": 778,
        "vpa_ci": "HIGH RISK — neonatal OXPHOS compromise; VPA hepatotoxicity risk; AED choice: PHB/CLZ/LEV",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "WARS2", "alias": "mtTrpRS / Mitochondrial tryptophanyl-tRNA synthetase",
        "aa": "360 aa", "kDa": "40.5 kDa",
        "gene_class": "pch",
        "tRNA_charges": "mt-tRNA-Trp (MT-TW)",
        "locus": "1p12", "omim_gene": 604733,
        "phenotype": "Combined OXPHOS deficiency — Neurodevelopmental disorder + HCM; West syndrome in some",
        "disease": "Combined OXPHOS deficiency 28 (OMIM #618336) — WARS2; Burke 2018 first WARS2 brain disease; neurodevelopmental disorder + intellectual disability + hypotonia; HCM in ~40% (cardiac involvement distinguishes); West syndrome / infantile spasms in some; cerebellar atrophy on MRI; CI + CIV reduced in muscle/fibroblasts; tryptophan is unique — only mt-tRNA not shared with cytoplasmic translation; WARS2 specifically required for all 13 mt-encoded OXPHOS subunits (Trp present in all 13)",
        "disease_omim": 618336, "inheritance": "AR",
        "hallmark": "WARS2 = NEURODEVELOPMENTAL DISORDER + HCM; tryptophan is present in ALL 13 mt-encoded OXPHOS proteins → WARS2 LOF uniquely affects all 13 subunits simultaneously; HCM 40% (cardiac involvement unusual among mtARS beyond AARS2); West syndrome/IS in subset (ACTH-responsive?); cerebellar atrophy on MRI; Burke 2018 Neurology landmark; some patients have intellectual disability without seizures; motor regression + spasticity; WARS2 1p12 — single copy, no CNV issues (vs MARS2)",
        "key_ddx": "vs TMEM70 (CV-HCM): TMEM70 3-MGA + Roma founder; vs AARS2 neonatal HCM: AARS2 null alleles; vs VARS2 (HCM + epilepsy): VARS2 has more prominent epilepsy; vs Pompe (GSD2): lysosomal enzyme; vs SURF1 (CIV-Leigh): CIV markedly reduced; vs CMT: WARS2 no PNS; vs SYNGAP1 (NSID): no metabolic component",
        "founder_variant": "p.Arg225Gln (Burke 2018 Irish); p.Val199Ile; p.Leu237Pro",
        "onset_pattern": "Infancy–toddler; hypotonia + DD; HCM often detected before seizures",
        "mri_pattern": "Cerebellar atrophy + delayed myelination + thin corpus callosum",
        "perrault": False, "alpers_like": False, "pch": True, "mlasa": False, "leuko": False,
        "snhl_rate": 0.20, "poi_rate": 0.05, "epilepsy_rate": 0.55, "ataxia_rate": 0.60,
        "spasticity_rate": 0.55, "hcm_rate": 0.40, "lactic_ac_rate": 0.65, "myopathy_rate": 0.55,
        "cognitive_rate": 0.85, "pn_rate": 0.20, "cerebellar_atrophy_rate": 0.70,
        "median_onset_months": 6, "seed": 779,
        "vpa_ci": "AVOID — OXPHOS compromise + HCM risk; VGB/LEV/ACTH preferred for infantile spasms",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    # ── MLASA ────────────────────────────────────────────────────────────────
    {
        "gene": "YARS2", "alias": "mtTyrRS / Mitochondrial tyrosyl-tRNA synthetase",
        "aa": "477 aa", "kDa": "53.2 kDa",
        "gene_class": "mlasa",
        "tRNA_charges": "mt-tRNA-Tyr (MT-TY)",
        "locus": "12p11.21", "omim_gene": 610957,
        "phenotype": "MLASA — Myopathy, Lactic Acidosis, and Sideroblastic Anemia",
        "disease": "MLASA (OMIM #613559) — YARS2; Riley 2010 first YARS2 MLASA; MLASA TRIAD: (1) myopathy (proximal weakness), (2) lactic acidosis (exercise-induced or baseline), (3) sideroblastic anemia (ring sideroblasts on bone marrow biopsy — pathognomonic); CI + CIV reduced in muscle; sideroblastic anemia from OXPHOS failure in erythroblast mitochondria (ring sideroblasts = iron deposits in mitochondrial cristae); WBC count normal (selective erythroid lineage); pyridoxine (B6) partially responsive in some (cf. X-linked ALAS2 XLSA vs AR YARS2 MLASA); Riley 2010 Am J Hum Genet",
        "disease_omim": 613559, "inheritance": "AR",
        "hallmark": "YARS2 = MLASA (the only mtARS gene for this triad); SIDEROBLASTIC ANEMIA = PATHOGNOMONIC of YARS2 MLASA — ring sideroblasts on bone marrow biopsy are mitochondrial iron deposits from OXPHOS failure in erythroblasts; distinguish from X-linked ALAS2 sideroblastic anemia (XLSA — males only, no myopathy, no lactic acidosis); MLASA triad is rare but SPECIFIC; exercise-induced lactic acidosis precedes myopathy recognition; anemia: normocytic or macrocytic + hypochromic fraction + basophilic stippling; erythropoiesis-stimulating agents + folate + pyridoxine (partial response); Riley 2010 landmark; YARS2 = only mtARS with haematological (non-neurological) predominant involvement",
        "key_ddx": "vs XLSA (ALAS2): X-linked, males only, no myopathy, no lactic acidosis; vs Pearson syndrome (mtDNA deletion): pancytopenia not just anemia, pancreatic exocrine failure; vs myelodysplastic syndrome (MDS): elderly, clonal, no family history; vs congenital sideroblastic anemia (SLC25A38): no myopathy, no lactic acidosis; vs mitochondrial myopathy without anemia: YARS2-specific due to erythroid mitochondrial dependence",
        "founder_variant": "p.Phe52Leu (Riley 2010 Australian); p.Thr227Ile (Greek); p.Gly191Val",
        "onset_pattern": "Childhood–adolescence; anemia often presenting feature; myopathy follows",
        "mri_pattern": "Usually normal brain MRI (predominantly peripheral/systemic disease)",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": True, "leuko": False,
        "snhl_rate": 0.10, "poi_rate": 0.05, "epilepsy_rate": 0.10, "ataxia_rate": 0.20,
        "spasticity_rate": 0.10, "hcm_rate": 0.15, "lactic_ac_rate": 0.85, "myopathy_rate": 0.90,
        "cognitive_rate": 0.15, "pn_rate": 0.20, "cerebellar_atrophy_rate": 0.05,
        "median_onset_months": 84, "seed": 780,
        "vpa_ci": "AVOID — OXPHOS compromise + sideroblastic anemia may worsen with VPA-induced mitochondrial dysfunction",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    # ── Multisystem / rare syndromic ─────────────────────────────────────────
    {
        "gene": "VARS2", "alias": "mtValRS / Mitochondrial valyl-tRNA synthetase",
        "aa": "1063 aa", "kDa": "119.5 kDa",
        "gene_class": "multisystem",
        "tRNA_charges": "mt-tRNA-Val (MT-TV)",
        "locus": "6p21.33", "omim_gene": 612802,
        "phenotype": "Combined OXPHOS deficiency 20 — Severe epileptic encephalopathy + HCM + lactic acidosis",
        "disease": "Combined OXPHOS deficiency 20 (OMIM #616060) — VARS2; Taylor 2014; early-onset severe epileptic encephalopathy (infantile spasms or early myoclonic encephalopathy) + HCM (50–60%) + lactic acidosis; CI + CIV markedly reduced; valyl-tRNA is required for all 13 mt-encoded subunits (Val present in all); VARS2 = largest of the 17 mtARS (1063aa); rapidly progressive; HCM distinguishes from RARS2/PCH6; splenomegaly in some; Taylor 2014 Am J Hum Genet",
        "disease_omim": 616060, "inheritance": "AR",
        "hallmark": "VARS2 = EPILEPTIC ENCEPHALOPATHY + HCM; HCM 50-60% is the highest cardiac rate among non-neonatal mtARS diseases (after AARS2 neonatal); early epilepsy + HCM combination → screen VARS2 + AARS2 + TMEM70 in differential; valine is universal in all 13 mt-encoded subunits → VARS2 LOF affects all 13 OXPHOS proteins; Taylor 2014; CIV most severely reduced; infantile spasms (IS) → ACTH + VGB (but NOT VPA); HCM management: propranolol/atenolol for hypertrophic obstruction; avoid digoxin (CI in OHCM); avoid VPA",
        "key_ddx": "vs WARS2: both HCM + epilepsy but VARS2 more severe epilepsy + more cardiac; vs TMEM70 (CV): 3-MGA + Roma founder; vs COQ4 (HCM): CoQ10 deficiency; vs POLG-Alpers: VPA-CI overlap but no HCM; vs AARS2 neonatal: null alleles → more acute neonatal; vs Pompe (HCM + myopathy): lysosomal enzyme — acid maltase",
        "founder_variant": "p.Pro800Leu (Taylor 2014 UK); p.Arg584Gln; p.Lys555Asn",
        "onset_pattern": "Infancy; epilepsy + HCM often simultaneously identified",
        "mri_pattern": "Diffuse cortical atrophy + basal ganglia involvement; resembles Leigh in some",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": False, "leuko": False,
        "snhl_rate": 0.20, "poi_rate": 0.05, "epilepsy_rate": 0.90, "ataxia_rate": 0.55,
        "spasticity_rate": 0.65, "hcm_rate": 0.55, "lactic_ac_rate": 0.85, "myopathy_rate": 0.60,
        "cognitive_rate": 0.90, "pn_rate": 0.20, "cerebellar_atrophy_rate": 0.50,
        "median_onset_months": 3, "seed": 781,
        "vpa_ci": "ABSOLUTE CI — HCM + OXPHOS compromise; VPA further worsens CI/CIV; fatal outcomes reported; use VGB/LEV/ACTH for IS",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "IARS2", "alias": "mtIleRS / Mitochondrial isoleucyl-tRNA synthetase",
        "aa": "994 aa", "kDa": "112.3 kDa",
        "gene_class": "multisystem",
        "tRNA_charges": "mt-tRNA-Ile (MT-TI)",
        "locus": "1p34.2", "omim_gene": 612801,
        "phenotype": "CAGSSS — Cataracts, Growth hormone deficiency, Sensory neuropathy, SNHL, Skeletal dysplasia",
        "disease": "CAGSSS (OMIM #616007) — IARS2; McMillan 2015 first IARS2 disease; CAGSSS acronym = (C)ataracts + (A)short stature/(G)rowth hormone deficiency + (S)ensory neuropathy + (S)SNHL + (S)keletal dysplasia; combined OXPHOS deficiency; unique multisystem phenotype not overlapping other mtARS syndromes; cataracts rare in mitochondrial disease (common in Kearns-Sayre/mtDNA deletions but IARS2-CAGSSS is nuclear); GH deficiency = hypothalamo-pituitary mitochondrial dysfunction; skeletal: metaphyseal dysplasia",
        "disease_omim": 616007, "inheritance": "AR",
        "hallmark": "IARS2 = CAGSSS (unique multi-organ syndromic mtARS); CATARACTS in nuclear mitochondrial disease = rare and distinctive (Kearns-Sayre is mtDNA large deletion with cataracts — IARS2 is the nuclear AR equivalent for cataracts in mitochondrial context); SNHL (as in Perrault mtARS) but without POI; skeletal dysplasia: not seen in other mtARS — IARS2-specific; GH deficiency → short stature → GH replacement indicated (unlike other mtARS where GH axis not affected); McMillan 2015 BMJ landmark; isoleucine is 2nd most frequent amino acid in mt-OXPHOS proteins after leucine; combined CI+CIII+CIV reduced",
        "key_ddx": "vs Kearns-Sayre (KSS): mtDNA large deletion — maternal inheritance/sporadic, retinopathy not cataracts, PEO; vs Cockayne syndrome: UV sensitivity, no OXPHOS; vs SNHL panel (DFNB): no cataracts, no GH deficiency; vs Laron syndrome (GH receptor): GH deficiency but no cataracts/neuropathy; vs Nance-Horan (NHS): cataracts X-linked, no OXPHOS",
        "founder_variant": "p.Arg979His (McMillan 2015 Pakistani); p.Arg703Trp; p.Val730Ile",
        "onset_pattern": "Childhood; cataracts often first feature; neuropathy later",
        "mri_pattern": "May show subtle WM or cerebellar changes; not always abnormal",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": False, "leuko": False,
        "snhl_rate": 0.90, "poi_rate": 0.05, "epilepsy_rate": 0.25, "ataxia_rate": 0.40,
        "spasticity_rate": 0.30, "hcm_rate": 0.20, "lactic_ac_rate": 0.60, "myopathy_rate": 0.50,
        "cognitive_rate": 0.40, "pn_rate": 0.75, "cerebellar_atrophy_rate": 0.25,
        "median_onset_months": 36, "seed": 782,
        "vpa_ci": "AVOID — OXPHOS compromise; cataracts + GH deficiency unrelated to VPA but systemic MI risk",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "SARS2", "alias": "mtSerRS / Mitochondrial seryl-tRNA synthetase",
        "aa": "518 aa", "kDa": "58.1 kDa",
        "gene_class": "multisystem",
        "tRNA_charges": "mt-tRNA-Ser(UCN) (MT-TS1) — charges mt-Ser(UCN); mt-Ser(UGA)/MT-TS2 charged by separate enzyme",
        "locus": "19q13.2", "omim_gene": 612467,
        "phenotype": "HUPRA syndrome — Hyperuricemia, Pulmonary hypertension, Renal failure, Alkalosis",
        "disease": "HUPRA syndrome (OMIM #613845) — SARS2; Belostotsky 2011 first SARS2-HUPRA; UNIQUE NON-NEUROLOGICAL mtARS phenotype: Hyperuricemia + Pulmonary hypertension (primary) + Renal failure + metabolic Alkalosis; tubular renal disease + interstitial nephritis; vascular smooth muscle OXPHOS high demand; combined OXPHOS deficiency (CI + CIV); Israeli Bedouin founder; hyperuricemia from purine degradation mismatch in failing OXPHOS; NO neurological features (rare among mtARS); pulmonary hypertension is vasomotor (not cardiac/thromboembolic); Belostotsky 2011 Am J Hum Genet",
        "disease_omim": 613845, "inheritance": "AR",
        "hallmark": "SARS2 = HUPRA (only non-neurological mtARS syndrome); PULMONARY HYPERTENSION in mtARS disease = RARE and specific to SARS2-HUPRA; renal tubular disease + interstitial nephritis (not glomerular like CoQ10 SRNS); hyperuricemia from ATP depletion → purine degradation acceleration; metabolic alkalosis (tubular acid handling failure); NO primary neurological involvement (distinguishes from ALL other mtARS); Bedouin Israeli founder enrichment; vascular/renal OXPHOS particularly sensitive to SARS2 LOF; urgent sildenafil + PAH therapy + renal protection; Belostotsky 2011 landmark",
        "key_ddx": "vs MARS2 (ARSAL): neurological; vs CoQ10 SRNS: glomerular not tubular; vs Fanconi syndrome: acidosis not alkalosis; vs Bartter/Gitelman: no OXPHOS, no PAH; vs primary PAH (BMPR2): no renal tubular, no hyperuricemia; vs Lesh-Nyhan (HPRT): hyperuricemia + neurological; SARS2-HUPRA is the only metabolic PAH syndrome",
        "founder_variant": "p.Asp390Gly (Belostotsky 2011 Israeli Bedouin); p.Arg402His; p.Leu417Pro",
        "onset_pattern": "Infancy–childhood; renal failure + PAH; Bedouin ancestry clue",
        "mri_pattern": "Normal brain MRI — no neurological features",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": False, "leuko": False,
        "snhl_rate": 0.05, "poi_rate": 0.05, "epilepsy_rate": 0.10, "ataxia_rate": 0.05,
        "spasticity_rate": 0.05, "hcm_rate": 0.10, "lactic_ac_rate": 0.70, "myopathy_rate": 0.30,
        "cognitive_rate": 0.10, "pn_rate": 0.10, "cerebellar_atrophy_rate": 0.05,
        "median_onset_months": 12, "seed": 783,
        "vpa_ci": "AVOID — renal compromise worsened by hepatotoxicity risk; renal elimination of VPA altered",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "TARS2", "alias": "mtThrRS / Mitochondrial threonyl-tRNA synthetase",
        "aa": "651 aa", "kDa": "73.4 kDa",
        "gene_class": "multisystem",
        "tRNA_charges": "mt-tRNA-Thr (MT-TT)",
        "locus": "1q21.2", "omim_gene": 612805,
        "phenotype": "Combined OXPHOS deficiency 21 — Infantile encephalopathy + lactic acidosis + hypotonia",
        "disease": "Combined OXPHOS deficiency 21 (OMIM #615918) — TARS2; Elo 2012 first TARS2 disease; infantile encephalopathy + severe hypotonia + lactic acidosis; CI + CIII + CIV combined deficiency; progressive brain atrophy; cardiomyopathy in some; <30 patients reported; TARS2 → threonyl-mt-tRNA failure → all 13 mt-encoded subunits affected (threonine present in all); Elo 2012 Am J Hum Genet; poor prognosis",
        "disease_omim": 615918, "inheritance": "AR",
        "hallmark": "TARS2 = Combined OXPHOS deficiency 21; rare (<30 patients) — presents as infantile encephalopathy + lactic acidosis; CI+CIII+CIV triple deficiency unusual (more gene-class specific for multisystem mtARS); cardiomyopathy in ~30%; Elo 2012 Finnish cohort; threonine is hydroxyl-containing amino acid — important in mitochondrial membrane topology; poor prognosis in severe alleles; management: supportive; biotin-thiamine trial (BTBGD exclusion mandatory); riboflavin supplementation (theoretical); Leigh-pattern MRI in some",
        "key_ddx": "vs POLG: replication vs translation; vs FARS2/PARS2: similar encephalopathy but VPA status different (TARS2 = avoid but not absolute CI per hepatic failure reports); vs SURF1 (CIV-Leigh): CIV isolated; vs complex-combined deficiency panel: multiple genes possible",
        "founder_variant": "p.Asp426Asn (Elo 2012 Finnish); p.Arg168Cys; p.Lys398Glu",
        "onset_pattern": "Infancy; encephalopathy + hypotonia",
        "mri_pattern": "Brain atrophy + basal ganglia signal; Leigh-pattern in some",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": False, "leuko": False,
        "snhl_rate": 0.20, "poi_rate": 0.05, "epilepsy_rate": 0.65, "ataxia_rate": 0.50,
        "spasticity_rate": 0.55, "hcm_rate": 0.30, "lactic_ac_rate": 0.85, "myopathy_rate": 0.60,
        "cognitive_rate": 0.85, "pn_rate": 0.20, "cerebellar_atrophy_rate": 0.45,
        "median_onset_months": 3, "seed": 784,
        "vpa_ci": "HIGH RISK / AVOID — OXPHOS compromise; prefer LEV/PB for seizure management",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "NARS2", "alias": "mtAsnRS / Mitochondrial asparaginyl-tRNA synthetase",
        "aa": "477 aa", "kDa": "53.4 kDa",
        "gene_class": "multisystem",
        "tRNA_charges": "mt-tRNA-Asn (MT-TN)",
        "locus": "11q14.2", "omim_gene": 612803,
        "phenotype": "Combined OXPHOS deficiency 24 — Non-syndromic hearing loss + myopathy + infantile seizures",
        "disease": "Combined OXPHOS deficiency 24 (OMIM #616239) — NARS2; Vanlander 2015; bilateral sensorineural hearing loss + proximal myopathy + infantile-onset seizures; CI + CIV reduced; Perrault-like (SNHL) but without POI — distinguishing from true Perrault (HARS2/LARS2); myopathy differentiates from SNHL-only Perrault; some patients: Leigh-like MRI; Vanlander 2015 AJHG; asparagine is N-glycosylation acceptor — not directly relevant for mt-translation function but required for structural integrity of mt-encoded OXPHOS subunit folding",
        "disease_omim": 616239, "inheritance": "AR",
        "hallmark": "NARS2 = SNHL + MYOPATHY + SEIZURES (3-part phenotype); Perrault-like but NO POI → Perrault requires POI for diagnosis — NARS2 is NOT Perrault despite SNHL; myopathy (proximal weakness, elevated CK) + SNHL combination → consider mtARS panel; infantile seizures; CI+CIV reduced; Vanlander 2015; asparagine-mt-tRNA is required for all 13 mt-encoded subunits (asparagine present in all); epilepsy usually more manageable than FARS2/PARS2; management: SNHL — cochlear implant eligible; seizures — LEV/CBZ; myopathy — physiotherapy; supplementation: riboflavin trial (partial CI/CIV benefit)",
        "key_ddx": "vs HARS2/LARS2 Perrault: Perrault has POI; NARS2 has no POI but has myopathy; vs DFNB families (non-syndromic SNHL): no myopathy, no seizures; vs Friedreich ataxia: no SNHL; vs mitochondrial myopathy: NARS2 unique SNHL combination; vs Wolfram (WFS1): DM + DI + OA + SNHL — no myopathy typically",
        "founder_variant": "p.Phe329Ser (Vanlander 2015 Belgian); p.Arg474Cys; p.Arg368Gln",
        "onset_pattern": "Infancy–early childhood; hearing loss + seizures first; myopathy follows",
        "mri_pattern": "Variable; Leigh-pattern in severe alleles; may be normal in mild",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": False, "leuko": False,
        "snhl_rate": 0.90, "poi_rate": 0.05, "epilepsy_rate": 0.65, "ataxia_rate": 0.45,
        "spasticity_rate": 0.40, "hcm_rate": 0.15, "lactic_ac_rate": 0.75, "myopathy_rate": 0.80,
        "cognitive_rate": 0.55, "pn_rate": 0.30, "cerebellar_atrophy_rate": 0.30,
        "median_onset_months": 9, "seed": 785,
        "vpa_ci": "AVOID — OXPHOS compromise; SNHL may worsen; prefer LEV/CBZ",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
    {
        "gene": "KARS1", "alias": "KRS / Lysyl-tRNA synthetase (dual cytoplasmic+mitochondrial)",
        "aa": "597 aa", "kDa": "67.8 kDa",
        "gene_class": "multisystem",
        "tRNA_charges": "mt-tRNA-Lys (MT-TK) [mitochondrial] + cytoplasmic tRNA-Lys [cytoplasmic] — DUAL LOCALIZATION",
        "locus": "16q23.1", "omim_gene": 601421,
        "phenotype": "DFNB89 (non-syndromic SNHL) / CMT2 peripheral neuropathy / Progressive leukoencephalopathy",
        "disease": "Multiple phenotypes (OMIM #613641) — KARS1; Santos-Cortez 2013 first KARS1 SNHL; UNIQUE among 17 mtARS: KARS1 is DUAL (cytoplasmic + mitochondrial); causes diverse phenotypes depending on which localization is predominantly affected: (1) DFNB89: autosomal recessive non-syndromic SNHL; (2) CMT2: progressive axonal peripheral neuropathy (cytoplasmic pathway); (3) Leukoencephalopathy: progressive WM + neurological regression; dual localization means both mt-translation and cytoplasmic translation affected; KARS1 mutations in cytoplasmic-specific domain → CMT2; in mitochondrial-specific domain → SNHL + metabolic; shared domain → both",
        "disease_omim": 613641, "inheritance": "AR",
        "hallmark": "KARS1 = DUAL CYTOPLASMIC+MITOCHONDRIAL LOCALIZATION — the only mtARS gene affecting BOTH translation systems; mutation location determines phenotype: cytoplasmic-domain → CMT2 (peripheral neuropathy, no lactate); mitochondrial-domain → SNHL + metabolic + leukoencephalopathy; shared catalytic core → combined phenotype; DFNB89 (non-syndromic SNHL) = most common KARS1 phenotype in populations where dual carrier enrichment occurs; Santos-Cortez 2013 Science landmark for DFNB89; leukoencephalopathy form overlaps DARS2/EARS2 pattern; KARS1 aminoacylates lysine-mt-tRNA → lysine is essential for many mt-OXPHOS subunit active sites",
        "key_ddx": "vs GJB2/GJB6 (non-syndromic SNHL): no neuropathy, no metabolic; vs CMT2A (MFN2): mitochondrial dynamics not translation; vs DARS2 (LBSL): DARS2 = mt-only; vs Charcot-Marie-Tooth panel: KARS1 = metabolic CMT; vs Alexander disease (GFAP): different leukodystrophy MRI; vs Refsum (PHYH): phytanic acid accumulation; vs POLG (PNS + CNS): mtDNA depletion",
        "founder_variant": "p.Tyr173Ser (Santos-Cortez 2013 Filipino); p.Arg477Gly (CMT2 European); p.Leu133His (leukoencephalopathy)",
        "onset_pattern": "Variable: SNHL = childhood; CMT2 = adolescence; leukoencephalopathy = childhood",
        "mri_pattern": "Normal (DFNB89/CMT2) OR leukoencephalopathy pattern (WM T2 signal) in severe alleles",
        "perrault": False, "alpers_like": False, "pch": False, "mlasa": False, "leuko": True,
        "snhl_rate": 0.70, "poi_rate": 0.05, "epilepsy_rate": 0.15, "ataxia_rate": 0.30,
        "spasticity_rate": 0.20, "hcm_rate": 0.10, "lactic_ac_rate": 0.45, "myopathy_rate": 0.35,
        "cognitive_rate": 0.30, "pn_rate": 0.80, "cerebellar_atrophy_rate": 0.20,
        "median_onset_months": 48, "seed": 786,
        "vpa_ci": "AVOID — dual translation system compromise; CMT2 form: no metabolic VPA risk but OXPHOS form: high risk",
        "coq10_response": "No specific response — not CoQ10 pathway",
    },
]

# ── Drug contraindications (common to all mtARS disease) ─────────────────────
DRUG_CIS = {
    "vpa_absolute_ci_genes": {
        "genes": ["FARS2", "PARS2", "VARS2"],
        "rule": "VPA ABSOLUTE CI in FARS2 + PARS2 + VARS2 — Alpers-like hepatic failure mechanism (same as POLG); fatal outcomes reported; EMERGENCY discontinue if started before diagnosis",
        "mechanism": "VPA depletes mtDNA-pol gamma cofactor + directly inhibits CI/CIV → fatal hepatic failure in Alpers-mechanism diseases",
        "alternative": "LEV (levetiracetam), VGB (vigabatrin, first-line IS), ACTH (infantile spasms), CLB (clobazam), LCM (lacosamide)",
    },
    "vpa_high_risk_all": "VPA HIGH RISK in ALL 17 mtARS genes — OXPHOS compromise + hepatotoxicity risk; avoid in established diagnosis; prefer LEV/LCM/CBZ",
    "aminoglycosides_avoid": {
        "rule": "Aminoglycosides AVOID in ALL 17 mtARS diseases — aminoglycosides inhibit mitochondrial ribosomes (prokaryotic origin of mitoribosomes); worsens OXPHOS directly; SNHL risk especially in Perrault (HARS2/LARS2) already hearing-impaired",
        "examples": "Gentamicin, tobramycin, amikacin, streptomycin, neomycin",
        "alternative": "Beta-lactams, cephalosporins, fluoroquinolones (with caution)",
    },
    "chloramphenicol_absolute_ci": {
        "rule": "Chloramphenicol ABSOLUTE CI in ALL mtARS disease — potent mitochondrial 50S ribosome inhibitor → blocks mt-translation → worsens ALL mtARS deficiency by preventing residual OXPHOS synthesis",
        "mechanism": "Chloramphenicol = prototype mt-ribosome inhibitor (originally used to study mt-translation in vitro); any residual mtARS activity abolished by chloramphenicol",
    },
    "metformin_avoid": "Metformin AVOID — Complex I inhibitor; worsens existing CI deficiency (common in all 17 mtARS); risk of lactic acidosis",
    "linezolid_avoid": "Linezolid AVOID — Oxazolidinone antibiotic; inhibits mitochondrial ribosomes → worsens mt-translation compromise",
    "statins_caution": "Statins: CAUTION — CoQ10 depletion (30-50%) adds secondary CoQ10 deficiency to existing OXPHOS stress; avoid in confirmed mtARS disease",
}

WES_UTILITY = {
    "detects_all_17": True,
    "nuclear_encoded_all": True,
    "exceptions": "KARS1 dual localization — ensure variant is in mitochondrial-targeting domain vs cytoplasmic; MARS2 complex duplication/CNV — standard WES may miss tandem repeats (requires CNV analysis or long-read)",
    "mars2_cnv_note": "MARS2 is a CNV-based disease — partial tandem duplication at 2q33.1; standard WES short-read may fail to detect; order CNV analysis or MARS2-specific long-read sequencing in ARSAL phenotype",
    "mt_panel_note": "mtDNA panel NOT needed for all 17 — these are nuclear-encoded; but exclude mtDNA-encoded tRNA mutations (MT-T* genes) which are also in the mt-tRNA space",
    "mt_tRNA_exclusion": "MT-T* genes (MT-TD/TE/TA etc.) cause MELAS/MERRF-like syndromes — these are mtDNA (maternal inheritance, heteroplasmy); mtARS genes are AR nuclear; both affect mt-translation but different genetics",
}

HALLMARK_PHENOTYPES = {
    "LBSL_DARS2": "Brainstem + spinal cord T2 + WM + lactate on MRS — pathognomonic MRI triad; slowly progressive adolescent ataxia",
    "LTBL_EARS2": "Thalamic + brainstem + cerebellar WM T2 + biphasic course — regression then plateau; most useful diagnostic differentiator",
    "LONA_AARS2": "Adult leukodystrophy + ovarian failure (POI females); neonatal HCM in null alleles — two distinct phenotypes by allele severity",
    "Perrault_HARS2_LARS2": "SNHL + POI (primary ovarian insufficiency) FEMALES ONLY — males = SNHL only; the only mtARS phenotype with gonadal failure",
    "Alpers_FARS2_PARS2": "VPA ABSOLUTE CI — Alpers-like hepatotoxicity; early-onset refractory epilepsy + rapid cortical atrophy + lactic acidosis",
    "PCH6_RARS2": "Pontocerebellar hypoplasia on fetal/neonatal MRI — structural malformation (not atrophy); neonatal epilepsy + microcephaly; most severe mtARS",
    "MLASA_YARS2": "Myopathy + lactic acidosis + sideroblastic anemia (ring sideroblasts on BM biopsy) — only haematological mtARS phenotype",
    "HUPRA_SARS2": "Hyperuricemia + pulmonary arterial hypertension + renal failure + alkalosis — only non-neurological mtARS; Bedouin founder",
    "CAGSSS_IARS2": "Cataracts + GH deficiency + sensory neuropathy + SNHL + skeletal dysplasia — unique multi-organ; cataracts rare in nuclear OXPHOS disease",
}

KEY_CLINICAL_RULES = {
    "mri_pattern_is_diagnostic": "MRI PATTERN IS DIAGNOSTIC: LBSL (DARS2), LTBL (EARS2+thalami), PCH6 (RARS2+pontine hypoplasia) — neuroradiologist familiarity with these patterns saves months of diagnostic odyssey",
    "vpa_absolute_ci_3_genes": "VPA ABSOLUTE CI in FARS2 + PARS2 + VARS2 — Alpers-mechanism hepatotoxicity; fatal outcomes; POLG + FARS2 + PARS2 + VARS2 = THE FOUR mtARS/replication genes where VPA must NEVER be used",
    "vpa_high_risk_all_17": "VPA HIGH RISK in ALL remaining 14 mtARS genes — OXPHOS compromise + general hepatotoxicity risk; prefer LEV/LCM in all confirmed mtARS",
    "aminoglycosides_absolutely_avoid": "Aminoglycosides AVOID in ALL 17 — inhibit mitochondrial ribosomes (prokaryotic origin); worsens OXPHOS; Perrault SNHL + aminoglycosides = additive cochlear damage",
    "chloramphenicol_absolute_ci_all": "Chloramphenicol ABSOLUTE CI in ALL 17 — prototype mt-50S ribosome inhibitor; blocks residual mt-translation completely",
    "perrault_2_genes_only": "Perrault syndrome (SNHL + POI) = ONLY HARS2 + LARS2 among all mtARS; Perrault with no POI (=NARS2) is NOT true Perrault; screen HARS2+LARS2 in any SNHL+POI female regardless of age",
    "wes_detects_all_except_mars2_cnv": "WES detects 16/17 — MARS2 tandem duplication may be MISSED by short-read WES; order CNV analysis or long-read for ARSAL phenotype",
    "btbgd_exclusion_mandatory": "BTBGD (SLC19A3) MANDATORY EXCLUSION in all metabolic encephalopathy before mtARS workup — biotin-thiamine responsive, easily treatable if caught early",
    "mt_dna_panel_not_needed": "mtDNA panel NOT required for mtARS diagnosis — all 17 are nuclear-encoded; but exclude MT-T* genes (maternal inheritance, heteroplasmy) before calling AR mt-translation disease",
    "linezolid_and_chloramphenicol_joint": "Linezolid + Chloramphenicol: both target mitochondrial ribosomes; BOTH contraindicated in ALL mtARS disease",
}


def _patient_cohort(gene: dict) -> list:
    """Generate 40 synthetic patients for a given mtARS gene using a deterministic RNG."""
    rng = random.Random(gene["seed"])
    patients = []
    for i in range(40):
        pid = i + 1
        sex = "F" if rng.random() < 0.52 else "M"
        onset_months = max(0, int(rng.gauss(gene["median_onset_months"], gene["median_onset_months"] * 0.45 + 2)))
        dx_delay_months = rng.randint(6, 72)
        snhl = rng.random() < gene["snhl_rate"]
        poi = sex == "F" and rng.random() < gene["poi_rate"]
        epilepsy = rng.random() < gene["epilepsy_rate"]
        ataxia = rng.random() < gene["ataxia_rate"]
        spasticity = rng.random() < gene["spasticity_rate"]
        hcm = rng.random() < gene["hcm_rate"]
        lactic_ac = rng.random() < gene["lactic_ac_rate"]
        myopathy = rng.random() < gene["myopathy_rate"]
        cognitive = rng.random() < gene["cognitive_rate"]
        pn = rng.random() < gene["pn_rate"]
        cerebellar_atrophy = rng.random() < gene["cerebellar_atrophy_rate"]
        # Lactate value (mmol/L plasma)
        lactate_mmol = round(rng.uniform(0.8, 1.8) if not lactic_ac else rng.uniform(2.5, 12.0), 1)
        # OXPHOS: which complexes reduced
        ci_reduced = rng.random() < 0.75
        ciii_reduced = rng.random() < 0.40
        civ_reduced = rng.random() < 0.70
        patients.append({
            "patient_id": f"{gene['gene']}-{pid:03d}",
            "gene": gene["gene"],
            "sex": sex,
            "onset_months": onset_months,
            "dx_delay_months": dx_delay_months,
            "snhl": snhl,
            "poi": poi,
            "epilepsy": epilepsy,
            "ataxia": ataxia,
            "spasticity": spasticity,
            "hcm": hcm,
            "lactic_ac": lactic_ac,
            "lactate_mmol": lactate_mmol,
            "myopathy": myopathy,
            "cognitive_impairment": cognitive,
            "peripheral_neuropathy": pn,
            "cerebellar_atrophy": cerebellar_atrophy,
            "ci_reduced": ci_reduced,
            "ciii_reduced": ciii_reduced,
            "civ_reduced": civ_reduced,
        })
    return patients


def get_overview():
    """Return high-level mtARS-Deficiency-Atlas overview for the /api/mtars-deficiency-atlas/overview endpoint."""
    all_patients = []
    for g in MTARS_GENES:
        all_patients.extend(_patient_cohort(g))

    n = len(all_patients)
    snhl_n     = sum(1 for p in all_patients if p["snhl"])
    poi_n      = sum(1 for p in all_patients if p["poi"])
    epilepsy_n = sum(1 for p in all_patients if p["epilepsy"])
    ataxia_n   = sum(1 for p in all_patients if p["ataxia"])
    lactic_n   = sum(1 for p in all_patients if p["lactic_ac"])
    hcm_n      = sum(1 for p in all_patients if p["hcm"])
    myopathy_n = sum(1 for p in all_patients if p["myopathy"])
    pn_n       = sum(1 for p in all_patients if p["peripheral_neuropathy"])

    gene_classes = {}
    for g in MTARS_GENES:
        c = g["gene_class"]
        gene_classes[c] = gene_classes.get(c, 0) + 1

    vpa_absolute_ci_genes = [g["gene"] for g in MTARS_GENES if "ABSOLUTE" in g["vpa_ci"]]
    vpa_high_risk_genes   = [g["gene"] for g in MTARS_GENES if "ABSOLUTE" not in g["vpa_ci"] and "AVOID" in g["vpa_ci"]]

    return {
        "atlas": "mtARS-Deficiency-Atlas",
        "subtitle": "Complete 17-Gene Mitochondrial Aminoacyl-tRNA Synthetase Deficiency Reference",
        "n_genes": len(MTARS_GENES),
        "n_patients": n,
        "cohort_formula": f"{len(MTARS_GENES)} genes × 40 patients = {n} patient aggregate (seeds 770–786)",
        "function": "mt-tRNA aminoacylation — each mtARS charges its cognate mt-tRNA; without correct charging, mitoribosomes stall → reduced synthesis of 13 OXPHOS subunits → multi-complex OXPHOS deficiency",
        "pathway": "mtARS charges mt-tRNA → mt-ribosome translates 13 OXPHOS subunits → CI/CIII/CIV/CV assembly → ATP production",
        "all_nuclear": True,
        "wes_detects_16_of_17": True,
        "mars2_cnv_caveat": "MARS2 tandem duplication (ARSAL) may require CNV or long-read sequencing — short-read WES may miss",
        "gene_classes": gene_classes,
        "genes_by_class": {
            "leukoencephalopathy": ["DARS2", "EARS2", "AARS2", "MARS2"],
            "perrault": ["HARS2", "LARS2"],
            "epilepsy_encephalopathy_vpa_ci": ["FARS2", "PARS2"],
            "pch_cerebellar": ["RARS2", "WARS2"],
            "mlasa": ["YARS2"],
            "multisystem": ["VARS2", "IARS2", "SARS2", "TARS2", "NARS2", "KARS1"],
        },
        "aggregate_clinical": {
            "n_patients": n,
            "snhl_pct": round(snhl_n / n * 100, 1),
            "poi_pct": round(poi_n / n * 100, 1),
            "epilepsy_pct": round(epilepsy_n / n * 100, 1),
            "ataxia_pct": round(ataxia_n / n * 100, 1),
            "lactic_acidosis_pct": round(lactic_n / n * 100, 1),
            "hcm_pct": round(hcm_n / n * 100, 1),
            "myopathy_pct": round(myopathy_n / n * 100, 1),
            "peripheral_neuropathy_pct": round(pn_n / n * 100, 1),
        },
        "vpa_absolute_ci_genes": vpa_absolute_ci_genes,
        "vpa_high_risk_genes": vpa_high_risk_genes,
        "drug_contraindications": DRUG_CIS,
        "wes_utility": WES_UTILITY,
        "hallmark_phenotypes": HALLMARK_PHENOTYPES,
        "key_rules": KEY_CLINICAL_RULES,
    }


def get_breakdown():
    """Return per-gene breakdown for the /api/mtars-deficiency-atlas/breakdown endpoint."""
    result = []
    for g in MTARS_GENES:
        pts = _patient_cohort(g)
        n = len(pts)
        result.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "gene_class": g["gene_class"],
            "tRNA_charges": g["tRNA_charges"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "disease_omim": g["disease_omim"],
            "inheritance": g["inheritance"],
            "phenotype_summary": g["phenotype"],
            "disease": g["disease"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "founder_variant": g["founder_variant"],
            "onset_pattern": g["onset_pattern"],
            "mri_pattern": g["mri_pattern"],
            "vpa_ci": g["vpa_ci"],
            "is_perrault": g["perrault"],
            "is_alpers_like": g["alpers_like"],
            "is_pch": g["pch"],
            "is_mlasa": g["mlasa"],
            "is_leuko": g["leuko"],
            "cohort_n": n,
            "snhl_pct": round(sum(1 for p in pts if p["snhl"]) / n * 100, 1),
            "poi_pct": round(sum(1 for p in pts if p["poi"]) / n * 100, 1),
            "epilepsy_pct": round(sum(1 for p in pts if p["epilepsy"]) / n * 100, 1),
            "ataxia_pct": round(sum(1 for p in pts if p["ataxia"]) / n * 100, 1),
            "spasticity_pct": round(sum(1 for p in pts if p["spasticity"]) / n * 100, 1),
            "hcm_pct": round(sum(1 for p in pts if p["hcm"]) / n * 100, 1),
            "lactic_ac_pct": round(sum(1 for p in pts if p["lactic_ac"]) / n * 100, 1),
            "myopathy_pct": round(sum(1 for p in pts if p["myopathy"]) / n * 100, 1),
            "cognitive_pct": round(sum(1 for p in pts if p["cognitive_impairment"]) / n * 100, 1),
            "pn_pct": round(sum(1 for p in pts if p["peripheral_neuropathy"]) / n * 100, 1),
            "cerebellar_atrophy_pct": round(sum(1 for p in pts if p["cerebellar_atrophy"]) / n * 100, 1),
            "median_onset_months": g["median_onset_months"],
            "avg_dx_delay_months": round(sum(p["dx_delay_months"] for p in pts) / n, 1),
        })
    return {"genes": result, "n_genes": len(result), "n_patients_total": len(result) * 40}


def get_definitions():
    """Return clinical definitions for the /api/mtars-deficiency-atlas/definitions endpoint."""
    return {
        "terms": {
            "mtARS": "Mitochondrial Aminoacyl-tRNA Synthetase — nuclear-encoded enzymes that attach the correct amino acid to its cognate mitochondrial tRNA (mt-tRNA); 17 distinct mtARS enzymes (one per amino acid, except 2 for Leu and Ser); without aminoacylation, mitoribosomes stall → reduced mt-translation → OXPHOS deficiency",
            "aminoacylation": "Covalent attachment of an amino acid to the 3'-OH of its cognate tRNA by an aminoacyl-tRNA synthetase; requires ATP → AMP + PPi; produces aminoacyl-tRNA (charged tRNA) for ribosomal A-site delivery during translation",
            "mt_translation": "Mitochondrial translation — synthesis of 13 OXPHOS subunits (7 CI: ND1-6 + ND4L; 1 CIII: CYB; 3 CIV: COX1-3; 2 CV: ATP6+8) using 22 mt-tRNAs and 2 mt-rRNAs (all mitochondrial-encoded) on mitoribosomes; 67 nuclear-encoded ribosomal proteins + mt-tRNA synthetases + mt-RNA-binding factors are all nuclear",
            "LBSL": "Leukoencephalopathy with Brainstem and Spinal Cord Involvement and Lactate Signal — DARS2 deficiency; MRI triad: cerebral WM T2 + brainstem tegmentum/pyramids + dorsal spinal cord columns; lactate on MRS; slowly progressive; p.Leu78His European founder",
            "LTBL": "Leukodystrophy with Thalamus and Brainstem involvement and high Lactate — EARS2 deficiency; MRI: thalamic + brainstem + cerebellar WM T2; biphasic course (regression → plateau/improvement) is pathognomonic",
            "LONA": "Leukoencephalopathy, Ovarian failure, Non-syndromic — adult-onset AARS2 disease; adult leukodystrophy + primary ovarian insufficiency females; neonatal form = fatal HCM (null alleles)",
            "ARSAL": "Autosomal Recessive Spastic Ataxia with Leukoencephalopathy — MARS2 disease; French-Canadian enrichment; complex tandem duplication (CNV-based); spastic ataxia + WM signal",
            "Perrault_syndrome": "Clinical syndrome of sensorineural hearing loss + primary ovarian insufficiency (females only); males have hearing loss only; caused by HARS2 (type 2) and LARS2 (type 4) among mtARS genes; also HSD17B4, CLPP, ERAL1 (non-mtARS); requires both SNHL + POI for diagnosis — NARS2 (SNHL without POI) is NOT Perrault",
            "POI": "Primary Ovarian Insufficiency — loss of normal ovarian function before age 40; clinical: absent puberty (primary) or secondary amenorrhoea; FSH > 25 IU/L on two occasions, AMH < 0.5 pmol/L; in Perrault mtARS — mitochondrial translation failure → granulosa/theca cell OXPHOS decline → follicle atresia; hormone replacement therapy indicated",
            "PCH6": "Pontocerebellar Hypoplasia Type 6 — RARS2 deficiency; structural brain malformation (hypoplasia = failure to form, not just later atrophy) affecting cerebellum + pons; detectable on fetal MRI at 20+ weeks; most severe mtARS phenotype; neonatal epilepsy + lactic acidosis + microcephaly",
            "MLASA": "Myopathy, Lactic Acidosis, and Sideroblastic Anemia — YARS2 deficiency; sideroblastic anemia = ring sideroblasts on bone marrow biopsy (iron deposits in mitochondrial cristae of erythroblasts); the only haematological (non-neurological) mtARS phenotype; distinguishes from X-linked ALAS2 (no myopathy, no lactic acidosis, males only)",
            "HUPRA": "Hyperuricemia, Pulmonary hypertension, Renal failure, Alkalosis — SARS2 deficiency; the only non-neurological mtARS phenotype; Bedouin Israeli founder; unique pulmonary arterial hypertension in metabolic disease",
            "CAGSSS": "Cataracts, Short stature/Growth hormone deficiency, Sensory neuropathy, Sensorineural hearing loss, Skeletal dysplasia — IARS2 deficiency; cataracts are rare in nuclear OXPHOS disease and hallmark this syndrome",
            "ring_sideroblasts": "Erythroblasts with iron deposits forming a ring pattern (≥5 mitochondria with iron granules encircling ≥1/3 of nucleus) on Prussian-blue bone marrow biopsy stain; pathognomonic of sideroblastic anemia; in YARS2-MLASA, caused by OXPHOS failure in erythroid progenitors → impaired haem synthesis + iron accumulation in mitochondria",
            "alpers_mechanism_vpa": "Alpers-mechanism hepatotoxicity — VPA inhibits mitochondrial beta-oxidation + fatty acid transport (PNPLA7-VPA interaction) + directly depletes mtDNA-polymerase-gamma cofactors; in any disease with residual OXPHOS compromise (POLG, FARS2, PARS2, VARS2), VPA-induced hepatic ATP depletion → acute fulminant liver failure; NOT dose-dependent — can occur at therapeutic levels",
            "chloramphenicol_mt_inhibitor": "Chloramphenicol = prototype mitochondrial ribosome inhibitor (binds mt-50S peptidyl-transferase centre); used historically to discriminate mt vs cytoplasmic translation in vitro; any dose blocks residual mt-translation in mtARS disease; ABSOLUTE CI in all 17 mtARS diseases",
            "WES_detects_all_nuclear": "Whole-exome sequencing detects variants in all 17 nuclear-encoded mtARS genes; EXCEPTION: MARS2 tandem duplication (copy-number variant) requires dedicated CNV analysis (chromosomal microarray or long-read NGS); MT-T* variants (mitochondrial-encoded mt-tRNA genes) require mtDNA sequencing (not WES) — separate diagnostic pathway",
            "dual_translation_KARS1": "KARS1 (lysyl-tRNA synthetase) is the only mtARS with dual cytoplasmic+mitochondrial localization; alternative N-terminal sequence targets to mitochondria; cytoplasmic domain mutations → CMT2 (peripheral neuropathy, no lactate); mitochondrial-targeting domain mutations → SNHL + metabolic disease; shared catalytic-core mutations → combined phenotype",
            "BTBGD_exclusion": "Biotin-Thiamine-Responsive Basal Ganglia Disease (SLC19A3 — BTBGD) MANDATORY EXCLUSION in ALL metabolic encephalopathy before mtARS workup; BTBGD presents with encephalopathy + basal ganglia crisis + lactic acidosis — identical clinical emergency; biotin 10 mg/kg/day + thiamine 100 mg/day can be life-saving within hours; treating BTBGD as mtARS = delayed biotin/thiamine → preventable mortality",
        }
    }
