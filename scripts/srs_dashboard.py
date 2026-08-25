#!/usr/bin/env python3
"""Silver-Russell Syndrome (SRS) Dashboard.

Silver-Russell Syndrome is the GROWTH-RESTRICTION opposite of Beckwith-Wiedemann Syndrome
at the SAME 11p15.5 locus — the canonical same-locus, opposite-phenotype imprinting pair.
  Principal genes: IGF2 (Insulin-Like Growth Factor 2, paternally expressed, growth driver)
                   H19 (lncRNA, maternally expressed, ICR1 methylation anchor)
                   CDKN1C (p57KIP2, maternally expressed, growth inhibitor)
  Mechanism: PATERNAL LOF at 11p15.5 → LOSS of paternal IGF2 expression → growth restriction
  Most common cause: H19-ICR1 hypomethylation on paternal allele (~45%) → IGF2 silenced biallelically
  OMIM Disease: #180860 · Genes: IGF2 *147470 · H19 *103280 · CDKN1C *600856
  Prevalence: ~1:30,000–1:100,000

IMPRINTING MECHANISM — WHY PATERNAL LOF CAUSES SRS:
  IGF2 (Insulin-Like Growth Factor 2, 7.5 kDa peptide, 11p15.5): PATERNALLY expressed only
    — Normally: ICR1 (H19/IGF2 Imprinting Control Region 1) = METHYLATED on paternal allele
    — ICR1 methylation on paternal allele silences H19 (paternally) → IGF2 enhancers can contact IGF2 → IGF2 expressed
    — On maternal allele: ICR1 UNMETHYLATED → CTCF binds → blocks IGF2 enhancers → IGF2 silenced maternally
    — Result: only PATERNAL IGF2 is expressed (maternal IGF2 silenced)
    — In SRS: ICR1 hypomethylated on PATERNAL allele → CTCF binds BOTH alleles → IGF2 silenced BIALLELICALLY
    — Consequence: ZERO IGF2 protein → profound prenatal and postnatal growth restriction
  H19 (lncRNA tumour suppressor, 11p15.5): MATERNALLY expressed
    — ICR1 controls H19 and IGF2 reciprocally: paternal methylation silences H19 / allows IGF2
    — In SRS: paternal ICR1 unmethylated → H19 expressed from both alleles (biallelic H19)
    — H19 overexpression may independently suppress growth (H19 is a growth repressor lncRNA)
  CDKN1C (p57KIP2, cyclin-dependent kinase inhibitor, 11p15.5 ICR2 region): MATERNALLY expressed
    — CDKN1C normally GROWTH-RESTRICTING (halts cell cycle)
    — In SRS: CDKN1C intact (ICR2 region unaffected in H19-ICR1 hypomethylation type)
    — In BWS: CDKN1C ABSENT / mutated → cell cycle unchecked → overgrowth
    — CDKN1C pathogenic variants cause SRS when maternal (gain of growth restriction)
      or BWS when paternal (haploinsufficiency from maternal allele)

FIVE GENETIC MECHANISMS (by frequency):
  1. H19-ICR1 hypomethylation (paternal allele, ~45%):
     — Paternal ICR1 loses methylation → biallelic IGF2 silencing
     — Methylation assay (MS-MLPA or SNaPshot) at 11p15.5: ICR1 methylation <50% (normal ~50%)
     — SNP array NORMAL (no copy number change, no UPD)
     — Usually de novo; recurrence <1% (assisted reproduction may increase risk)
  2. Maternal UPD7 (upd(7)mat, ~10%):
     — Two maternal chromosome 7 copies, no paternal chr7
     — Loss of paternally expressed genes on chr7 (GRB10, SGCE, others)
     — GRB10 (IGF1R signalling inhibitor): paternally expressed, loss → impaired IGF1R signalling
     — Milder SRS phenotype; less hemihypotrophy; café-au-lait spots common (ring chromosome 7 overlap)
     — SNP array: LOH chr7 (isodisomy = complete) or partial LOH; no ICR1 abnormality
  3. CDKN1C pathogenic variant (ICR2 region, ~5%):
     — Gain-of-function variant in CDKN1C → excessive growth restriction on MATERNAL allele
     — Autosomal dominant; ~50% recurrence from carrier mother
     — No ICR1 hypomethylation; ICR2 methylation test normal
     — NGS/Sanger required after negative methylation + SNP array
  4. Paternal deletion 11p15.5 (~3-5%):
     — Deletion of paternal IGF2 → loss of the sole expressed IGF2 allele
     — CMA: copy number loss at 11p15.5 on paternal allele
     — If inherited from father: 50% recurrence; de novo: <1%
  5. Maternal duplication 11p15.5 (~2%):
     — Extra maternal 11p15.5 → extra H19 / extra CDKN1C → additional growth suppression
     — CMA: copy number gain at 11p15.5 on maternal allele
  6. Unknown mechanism (~35%): no abnormality found on standard testing; likely ultra-rare copy number variants
     or imprinting defects beyond current panel resolution; clinical diagnosis by Netchine-Harbison criteria

CLINICAL CRITERIA — NETCHINE-HARBISON (NH) SCORE (diagnosis requires ≥4/6):
  1. SGA (birth weight or length ≤-2 SDS for gestational age)
  2. Postnatal growth restriction (current height ≤-2 SDS)
  3. Relative macrocephaly at birth (HC SDS - birth weight SDS ≥1.5)
  4. Body asymmetry (limb length discrepancy ≥0.5 cm, hemifacial asymmetry)
  5. Protruding forehead in early life (frontal bossing, triangular face)
  6. Feeding difficulties (tube feeding required, or BMI <-2 SDS at 24 months)

CLINICAL FEATURES:
  Growth:
    Severe SGA at birth: ~96%; mean birth weight ~-3.5 SDS (often <1500g term)
    Birth length: mean ~-3.0 SDS
    Relative macrocephaly: HC disproportionately large vs weight (~70%)
    Head sparing: brain growth preserved (contrast: weight and length severely restricted)
    Postnatal growth restriction: universal; without GH therapy, adult height -3 to -4 SDS
    GH therapy (rhGH, Genotropin®): LEVEL A evidence; FDA ODA-approved 2003; start ASAP ideally before age 4
    Mean adult height gain with GH: +7-10 cm (vs untreated ~141 cm male, 131 cm female)
    Target height calculation: essential (mid-parental height usually normal in de novo SRS)
  Body Asymmetry:
    Hemihypotrophy (limb/face/trunk asymmetry): ~50-65%
    Usually ipsilateral (same side leg + arm + face), can be isolated limb length discrepancy
    Leg length discrepancy ≥0.5 cm: requires orthopedic follow-up + shoe lifts
    NOT progressive in most (stabilises as growth slows)
    Body asymmetry is KEY differentiator from Temple Syndrome (Temple = NO asymmetry)
  Facies:
    Triangular face: narrow forehead, pointed chin (most distinctive)
    Frontal bossing: prominent forehead
    Micrognathia: small lower jaw
    Flat nasal bridge, upturned nose
    Thin upper lip, down-turned mouth
    Fifth finger clinodactyly: ~70%
    Brachydactyly: ~20%
    Ears: normally positioned, sometimes low-set
  Metabolic:
    Neonatal hypoglycemia: ~50-60% (insufficient IGF2 → reduced hepatic glycogen + impaired gluconeogenesis)
    Fasting hypoglycemia: HIGH RISK throughout childhood (prolonged fasting ABSOLUTE contraindication)
    Cornstarch / nocturnal feeds: Level A standard of care in SRS infancy
    Insulin: NORMAL or LOW (hypoglycemia is NOT hyperinsulinemic, unlike SCHAD/HH disorders)
    IGF2 serum level: LOW (direct measure of defect; not routinely tested clinically)
    IGF-1: often low-normal or low (GH-deficient subset)
    GH axis: ~40% have partial GH deficiency on provocation testing
  Endocrine:
    Central Precocious Puberty (CPP): ~25-30% (less than Temple's 50-60%)
    Premature adrenarche: ~35% (earlier pubic hair; androgen excess from thin body habitus / stress)
    GnRH analog (leuprolide, triptorelin): Level B (SRS) — less robust than Temple (Level A)
    GH therapy may accelerate bone age; dose monitoring required
    Adult fertility: generally preserved
  Cognitive:
    Intelligence usually NORMAL — KEY contrast with Temple Syndrome (80% mild-moderate ID)
    Borderline IQ (IQ 80-90): ~30%
    Motor delay in infancy: ~50% (resolves by school age)
    Speech delay: ~35% (often feeding/hypotonia related, resolves)
    ADHD/attention difficulties: ~25%
    Learning difficulties (dyslexia, dyscalculia): ~20%
  Musculoskeletal:
    Scoliosis: ~30-40%
    Low muscle mass / poor muscle tone: universal in infancy
    Café-au-lait spots: ~10% (mainly maternal UPD7 and ring chr7 cases)
  Feeding:
    Severe feeding difficulties: ~70% (major cause of morbidity, hospitalisation)
    Require NG tube or gastrostomy in ~30%
    Low appetite / poor caloric intake persists into childhood
    High-calorie supplementation: cornstarch, PediaSure, overnight feeds
    Extreme stress response to caloric restriction (must avoid fasting)

EPILEPSY (~10-15%):
  MECHANISM: NOT primary epilepsy gene disorder
    — Neonatal/infantile hypoglycemia → hypoglycemic seizures (most common, ~70% of SRS epilepsy)
    — HIE from neonatal complications (SGA birth, preterm)
    — Febrile convulsions (increased susceptibility, ~5%)
    — ADHD medication side effects (rare)
  Seizure types:
    Focal hypoglycemic seizures: most common in neonatal period
    Febrile generalised convulsions: common in early childhood
    Infantile spasms: rare (<5% of SRS epilepsy), associated with HIE
    GTCS: in HIE context
  EEG: often normal interictally; focal slowing if HIE-related
  AED management:
    No absolute AED contraindications in SRS (unlike Angelman's CBZ/OXC CI)
    Glucose correction FIRST — many seizures resolve with IV glucose
    LEV preferred if AED needed (weight-neutral, no metabolic concerns)
    VPA: MODERATE RISK (hepatic monitoring; weight gain risk in thin SRS patients; not absolute CI)
    Phenobarbital: avoid long-term (impairs feeding in infants already feeding-challenged)
  Prognosis: good if hypoglycemia prevented; recurrent severe hypoglycemia = risk factor for DRE + cognitive impairment

AED CONTRAINDICATION SUMMARY:
  CBZ/OXC: NOT absolutely contraindicated (unlike Angelman); use with monitoring
  VPA: MODERATE risk (weight gain; hepatic; avoid in first-line unless myoclonic)
  Phenobarbital: AVOID in neonatal SRS (impairs feeding further)
  FASTING: ABSOLUTE contraindication — hypoglycemia + seizure risk
  Ketogenic diet: CONTRAINDICATED — severe caloric restriction causes severe hypoglycemia in SRS
  Cornstarch: MANDATORY — prevents overnight fasting hypoglycemia

DIAGNOSTIC PATHWAY:
  Step 1: Clinical Netchine-Harbison score (≥4/6 = diagnostic probability HIGH → test)
  Step 2: Methylation analysis 11p15.5 (H19-ICR1 methylation by MS-MLPA):
    — detects H19-ICR1 hypomethylation (~45%) + maternal duplication/paternal deletion at 11p15.5
    — Sensitivity for molecular SRS: ~60% (ICR1 types)
    — FIRST-LINE test
  Step 3 (if negative): Chromosomal SNP array:
    — detects maternal UPD7 (LOH chr7) + structural anomalies at 11p15.5
    — adds ~10% (UPD7) + ~5% (copy number variants)
  Step 4 (if negative): CDKN1C sequencing (NGS panel):
    — detects CDKN1C variants (~5%)
  Note: ~35% remain molecularly unsolved; clinical diagnosis by NH criteria if ≥4/6 met

SAME-LOCUS OPPOSITE-PHENOTYPE PAIR (SRS vs BWS):
  Locus: 11p15.5 (TWO ICRs: ICR1 controls H19/IGF2; ICR2 controls KCNQ1OT1/CDKN1C)
  SRS: paternal LOF at ICR1 → IGF2 absent → GROWTH RESTRICTION
  BWS: maternal LOF at ICR1 (paternal ICR1 hypermethylation) → IGF2 excess → OVERGROWTH
  BWS: maternal LOF at ICR2 (KCNQ1OT1 gain / CDKN1C loss) → CDKN1C absent → OVERGROWTH
  SRS hemihypotrophy ↔ BWS hemihypertrophy (Wilms tumour risk)
  SRS fasting hypoglycemia (IGF2-absent, insulin normal) ↔ BWS hyperinsulinism (KATP or focal)
  SRS: NO tumour surveillance usually ↔ BWS: mandatory AFP + ultrasound (hepatoblastoma, Wilms, adrenocortical)
  SRS: H19 biallelic expressed ↔ BWS: H19 silenced (ICR1 hypermethylated both alleles in paternal gain)
  SAME LOCUS, OPPOSITE PARENT, OPPOSITE PHENOTYPE = proof of 11p15.5 imprinting

KEY DIFFERENTIALS:
  SRS vs Temple Syndrome (14q32.3):
    Shared: SGA, postnatal short stature, feeding difficulties, mild CPP
    SRS-specific: hemihypotrophy, NORMAL cognition, 11p15.5 ICR1 hypomethylation, café-au-lait (UPD7)
    Temple-specific: 14q32.3 DLK1 loss, 80% mild-moderate ID, CPP 50-60% (Level A GnRH analog), truncal obesity
    Test: methylation 11p15 vs methylation 14q32.3 — different panels
  SRS vs Noonan Syndrome:
    Shared: short stature, facial dysmorphism
    SRS: NO cardiac defects (Noonan: 80%); SRS: hemihypotrophy; Noonan: RAS/MAPK variant
  SRS vs Turner Syndrome:
    Both: short stature, SGA in some
    Turner: 45,X; gonadal dysgenesis; SRS: normal karyotype
  SRS vs IUGR without imprinting:
    IUGR: proportionate growth restriction (weight + length proportional)
    SRS: relative macrocephaly (brain sparing) + hemihypotrophy + facies — NOT just IUGR

GENETICS / RECURRENCE:
  H19-ICR1 hypomethylation: usually de novo; recurrence <1% (unless assisted reproduction, which may increase risk slightly)
  Maternal UPD7: always de novo; recurrence <1%
  CDKN1C variant (maternal): ~50% if mother is carrier; AUTOSOMAL DOMINANT from maternal allele
  Paternal deletion: if inherited from father: 50% recurrence; de novo: <1%
  Parental karyotype: mandatory for copy number variants
  Epigenetic counseling: ICR1 hypomethylation may recur in siblings (multilocus imprinting disturbance in ~5%)

INCIDENCE / EPIDEMIOLOGY:
  Prevalence: ~1:30,000–1:100,000 (SRS is underdiagnosed due to phenotypic variability)
  Most common imprinting growth disorder after SGA
  Female:Male ratio approximately 1:1 (no sex predilection)
  Ethnicity: no predilection; described worldwide
  ART (assisted reproduction): ~10x increased risk of ICR1 hypomethylation with IVF/ICSI
"""

import random
import json
from datetime import datetime, timedelta

SEED = 295
rng = random.Random(SEED)

MECHANISMS = [
    {"id": "icr1_hypo", "label": "H19-ICR1 Hypomethylation (Paternal)", "n": 18, "pct": 45,
     "methylation_pct": 15,  # mean ICR1 methylation (normal ~50%)
     "asymmetry_rate": 0.65, "cpp_rate": 0.30, "hypo_rate": 0.65, "id_rate": 0.05},
    {"id": "upd7mat", "label": "Maternal UPD7 (upd(7)mat)", "n": 4, "pct": 10,
     "methylation_pct": 50,  # ICR1 normal
     "asymmetry_rate": 0.40, "cpp_rate": 0.20, "hypo_rate": 0.45, "id_rate": 0.10},
    {"id": "cdkn1c", "label": "CDKN1C Variant (ICR2 Region)", "n": 4, "pct": 10,
     "methylation_pct": 50,  # ICR1 normal
     "asymmetry_rate": 0.50, "cpp_rate": 0.25, "hypo_rate": 0.55, "id_rate": 0.08},
    {"id": "pat_del", "label": "Paternal Deletion 11p15.5", "n": 4, "pct": 10,
     "methylation_pct": 10,  # severely low
     "asymmetry_rate": 0.70, "cpp_rate": 0.35, "hypo_rate": 0.70, "id_rate": 0.05},
    {"id": "mat_dup", "label": "Maternal Duplication 11p15.5", "n": 2, "pct": 5,
     "methylation_pct": 48,  # slightly low
     "asymmetry_rate": 0.55, "cpp_rate": 0.25, "hypo_rate": 0.50, "id_rate": 0.05},
    {"id": "unknown", "label": "Unknown Mechanism", "n": 8, "pct": 20,
     "methylation_pct": 45,  # borderline
     "asymmetry_rate": 0.45, "cpp_rate": 0.20, "hypo_rate": 0.40, "id_rate": 0.10},
]
assert sum(m["n"] for m in MECHANISMS) == 40

PHENOTYPE_GROUPS = [
    {"label": "Classic SRS", "desc": "NH score 5-6/6, severe SGA ≤-3 SDS, hemihypotrophy, relative macrocephaly", "n": 16, "nh_min": 5},
    {"label": "Moderate SRS", "desc": "NH score 4-5/6, SGA ≤-2.5 SDS, asymmetry present, typical facies", "n": 16, "nh_min": 4},
    {"label": "Mild/Atypical SRS", "desc": "NH score 4/6, SGA ≤-2 SDS, fewer features, often ICR1 partial", "n": 8, "nh_min": 4},
]

VARIANTS = [
    {"variant": "H19-ICR1 hypomethylation", "mechanism": "icr1_hypo", "freq_pct": 45,
     "description": "Paternal ICR1 loses methylation → CTCF binds both alleles → biallelic IGF2 silencing → zero IGF2 protein",
     "methylation": "ICR1 ~15% (normal 50%)", "severity": "Classic", "recurrence": "<1% (de novo)"},
    {"variant": "Maternal UPD7 (upd(7)mat)", "mechanism": "upd7mat", "freq_pct": 10,
     "description": "Two maternal chr7 → loss of paternal GRB10, SGCE → impaired IGF1R signalling; café-au-lait spots common",
     "methylation": "ICR1 normal (50%)", "severity": "Milder", "recurrence": "<1% (de novo)"},
    {"variant": "CDKN1C gain-of-function (maternal)", "mechanism": "cdkn1c", "freq_pct": 5,
     "description": "Maternal CDKN1C pathogenic variant → excess p57KIP2 → cell cycle arrest → growth restriction; AD from carrier mother",
     "methylation": "ICR1 normal (50%)", "severity": "Variable", "recurrence": "~50% if mother carrier"},
    {"variant": "Paternal deletion 11p15.5", "mechanism": "pat_del", "freq_pct": 4,
     "description": "Deletion removes paternal IGF2 → total loss of IGF2 (paternal-only gene deleted); methylation severely low",
     "methylation": "ICR1 <10% (paternal allele deleted)", "severity": "Classic", "recurrence": "50% if inherited; <1% de novo"},
    {"variant": "Maternal duplication 11p15.5", "mechanism": "mat_dup", "freq_pct": 2,
     "description": "Extra maternal 11p15.5 → extra H19 + extra CDKN1C → biallelic growth suppression",
     "methylation": "ICR1 slightly low (~48%)", "severity": "Mild", "recurrence": "50% if maternal; de novo <1%"},
    {"variant": "Unknown mechanism", "mechanism": "unknown", "freq_pct": 34,
     "description": "Clinical SRS by NH criteria (≥4/6); no abnormality found on MS-MLPA + SNP array + CDKN1C; likely ultra-rare/novel variants",
     "methylation": "Normal (50%)", "severity": "Variable", "recurrence": "Unknown"},
]

AEDS = [
    {"name": "Levetiracetam (LEV)", "level": "B", "role": "First-line if AED needed",
     "rationale": "Weight-neutral; no metabolic interference; safe in SRS feeding-challenged patients"},
    {"name": "Valproic acid (VPA)", "level": "B-moderate-risk",
     "role": "Use if myoclonic or GTCS; monitor carefully",
     "rationale": "Weight gain risk significant in thin SRS patients; hepatic monitoring needed; NOT absolute CI"},
    {"name": "Lamotrigine (LTG)", "level": "B", "role": "Focal seizures; good metabolic profile",
     "rationale": "Weight-neutral; no fasting/metabolic concerns; suitable for long-term SRS use"},
    {"name": "Phenobarbital (PB)", "level": "AVOID",
     "role": "Avoid in SRS neonates/infants",
     "rationale": "IMPAIRS FEEDING — catastrophic in SRS where feeding difficulties already severe and life-threatening"},
    {"name": "Carbamazepine / Oxcarbazepine", "level": "Caution",
     "role": "Not absolutely contraindicated (unlike Angelman)",
     "rationale": "No specific SRS contraindication; use with standard monitoring; prefer LEV/LTG as first-line"},
    {"name": "Cornstarch (Glycosade®)", "level": "A",
     "role": "MANDATORY — prevents overnight hypoglycemic seizures",
     "rationale": "Slow-release glucose; prevents dawn fasting hypoglycemia → seizure risk; ABSOLUTE standard of care in SRS infancy"},
    {"name": "Ketogenic diet", "level": "CONTRAINDICATED",
     "role": "NEVER use in SRS",
     "rationale": "KD imposes caloric restriction + carbohydrate exclusion → SEVERE hypoglycemia in SRS; life-threatening"},
]

MANAGEMENTS = [
    {"category": "GH Therapy (rhGH)", "intervention": "Recombinant GH (Genotropin/Norditropin/Humatrope)",
     "dose": "0.035 mg/kg/day SC", "evidence": "Level A (FDA ODA 2003)",
     "outcome": "+7-10 cm adult height; improved body composition; +lean mass; IGF-1 normalisation",
     "timing": "Start ASAP; ideally before age 4 (bone age critical window)", "monitoring": "IGF-1 levels 6-monthly; bone age annual"},
    {"category": "Fasting Prevention", "intervention": "Cornstarch + overnight feeds",
     "dose": "1-2 g/kg uncooked cornstarch at bedtime", "evidence": "Level A",
     "outcome": "Prevents fasting hypoglycemia overnight → prevents hypoglycemic seizures",
     "timing": "Begin in neonatal period; continue through childhood until GH therapy improves body composition",
     "monitoring": "Fasting glucose; blood glucose log"},
    {"category": "GnRH Analog (CPP)", "intervention": "Leuprolide or triptorelin",
     "dose": "Standard CPP dosing", "evidence": "Level B",
     "outcome": "+3-5 cm adult height (synergistic with GH); slows bone age advancement",
     "timing": "At CPP onset (if confirmed by GnRH stimulation test)",
     "monitoring": "Bone age 6-monthly; LH/FSH response"},
    {"category": "Nutritional Support", "intervention": "High-calorie feeds (NG/gastrostomy if needed)",
     "dose": "110-120% RDA for ideal weight", "evidence": "Level A",
     "outcome": "Prevents hypoglycemia; improves catch-up; reduces hospitalisation",
     "timing": "Neonatal; NG tube in 30%; G-tube in 10% for severe feeding failure",
     "monitoring": "Weight velocity; caloric intake log"},
    {"category": "Orthopaedic (Leg Length)", "intervention": "Shoe lifts → epiphysiodesis if >2 cm discrepancy",
     "dose": "Lift 50% of discrepancy initially", "evidence": "Level B",
     "outcome": "Prevents scoliosis; equalises gait biomechanics",
     "timing": "From walking age; surgical if discrepancy >2 cm and still growing",
     "monitoring": "Standing radiograph for leg lengths 6-monthly"},
    {"category": "Scoliosis", "intervention": "Physiotherapy + bracing if >20° Cobb",
     "dose": "GH monitoring (GH may worsen scoliosis in rapid growth phase)", "evidence": "Level B",
     "outcome": "Prevention of surgical intervention",
     "timing": "Spinal radiograph annual from age 5", "monitoring": "Cobb angle progression"},
    {"category": "Bone Density", "intervention": "DXA scan + calcium/vitamin D",
     "dose": "1000 mg calcium/day; 400-800 IU vitamin D", "evidence": "Level B",
     "outcome": "GH therapy improves BMD significantly in SRS",
     "timing": "DXA at GH start; repeat 2-yearly", "monitoring": "DXA Z-score"},
]

DIFFERENTIALS = [
    {"condition": "Temple Syndrome (14q32.3)", "shared": "SGA, short stature, feeding difficulties, CPP (~25-30% vs SRS)",
     "srs_unique": "Hemihypotrophy (asymmetry); NORMAL cognition; 11p15.5 ICR1 defect; café-au-lait (UPD7)",
     "temple_unique": "80% mild-moderate ID (IQ 65-80); CPP 50-60% Level-A GnRH; truncal obesity; 14q32.3 DLK1/MEG3",
     "key_discriminator": "Hemihypotrophy = SRS; Cognitive impairment = Temple; DIFFERENT methylation panels",
     "verdict": "Often misdiagnosed as each other — test BOTH loci when SGA + short stature"},
    {"condition": "BWS (Beckwith-Wiedemann Syndrome, 11p15.5)", "shared": "Same 11p15.5 locus; may have hemihypertrophy; hypoglycemia",
     "srs_unique": "GROWTH RESTRICTION (BWS = overgrowth); hemihypotrophy; no Wilms/hepatoblastoma surveillance needed",
     "bws_unique": "Macrosomia; omphalocele; macroglossia; HYPERINSULINISM (not fasting-type hypoglycemia); Wilms + hepatoblastoma risk",
     "key_discriminator": "SRS = growth RESTRICTION; BWS = growth EXCESS — SAME locus OPPOSITE phenotype OPPOSITE parent",
     "verdict": "Paradigm of 11p15.5 imprinting: paternal loss = SRS; maternal loss = BWS"},
    {"condition": "IUGR (non-imprinting)", "shared": "SGA, postnatal short stature, feeding difficulties",
     "srs_unique": "Relative macrocephaly (brain sparing disproportionate); hemihypotrophy; triangular face; ICR1 hypomethylation",
     "iugr_unique": "Proportionate growth restriction; normal facial features; molecular testing negative",
     "key_discriminator": "NH criteria ≥4/6 + molecular testing distinguishes SRS from idiopathic IUGR",
     "verdict": "SRS is specifically diagnosed; 'just SGA' requires NH exclusion"},
    {"condition": "Noonan Syndrome", "shared": "Short stature, facial dysmorphism, feeding difficulties in infancy",
     "srs_unique": "Hemihypotrophy; no cardiac defects; 11p15.5 molecular abnormality",
     "noonan_unique": "Cardiac defects 80% (pulmonary stenosis, HCM); RAS/MAPK variant (PTPN11 etc); webbed neck",
     "key_discriminator": "Cardiac exam + molecular panel (RAS/MAPK) rules out Noonan",
     "verdict": "Echocardiogram + genetics panel differentiates"},
    {"condition": "Turner Syndrome (45,X)", "shared": "Short stature, lymphedema in infancy, feeding difficulties",
     "srs_unique": "Males affected; hemihypotrophy; 11p15.5 molecular; no gonadal dysgenesis",
     "turner_unique": "Female only; 45,X karyotype; gonadal dysgenesis; cardiac/renal anomalies",
     "key_discriminator": "Karyotype (chromosome analysis) immediately differentiates",
     "verdict": "Karyotype is first-line in short-stature females"},
]

def _make_cohort():
    pts = []
    pid = 1
    mech_pool = []
    for m in MECHANISMS:
        mech_pool.extend([m] * m["n"])
    rng.shuffle(mech_pool)
    pheno_labels = []
    for pg in PHENOTYPE_GROUPS:
        pheno_labels.extend([pg["label"]] * pg["n"])
    rng.shuffle(pheno_labels)
    for i, (mech, pheno) in enumerate(zip(mech_pool, pheno_labels)):
        age_diag = round(rng.uniform(0.3, 5.0), 1)  # years
        birth_weight_sds = round(rng.uniform(-4.5, -2.0), 1)
        birth_length_sds = round(rng.uniform(-4.0, -2.0), 1)
        current_height_sds = round(rng.uniform(-3.5, -1.8), 1)
        icr1_meth = round(mech["methylation_pct"] + rng.uniform(-5, 5), 1)
        icr1_meth = max(5, min(50, icr1_meth))
        nh_score = rng.randint(4, 6) if pheno == "Classic SRS" else rng.randint(4, 5) if pheno == "Moderate SRS" else 4
        asymmetry = rng.random() < mech["asymmetry_rate"]
        cpp = rng.random() < mech["cpp_rate"]
        neonatal_hypo = rng.random() < mech["hypo_rate"]
        epilepsy = rng.random() < 0.12  # ~12% overall
        gh_therapy = rng.random() < 0.80  # most SRS get GH
        on_gnrh = cpp and rng.random() < 0.70
        pts.append({
            "id": f"SRS-{pid:03d}",
            "mechanism": mech["label"],
            "phenotype_group": pheno,
            "age_diagnosis_y": age_diag,
            "sex": rng.choice(["M", "F"]),
            "birth_weight_sds": birth_weight_sds,
            "birth_length_sds": birth_length_sds,
            "current_height_sds": current_height_sds,
            "nh_score": nh_score,
            "icr1_methylation_pct": icr1_meth if mech["id"] in ("icr1_hypo", "pat_del", "mat_dup", "unknown") else 50.0,
            "hemihypotrophy": asymmetry,
            "relative_macrocephaly": rng.random() < 0.70,
            "feeding_difficulties": rng.random() < 0.70,
            "neonatal_hypoglycemia": neonatal_hypo,
            "cpp": cpp,
            "gh_therapy": gh_therapy,
            "on_gnrh_analog": on_gnrh,
            "scoliosis": rng.random() < 0.30,
            "epilepsy": epilepsy,
            "aed_if_epilepsy": rng.choice(["LEV", "LTG", "VPA-low-dose"]) if epilepsy else None,
        })
        pid += 1
    return pts

_COHORT = _make_cohort()

def get_overview():
    cohort = _COHORT
    n = len(cohort)
    return {
        "disease": "Silver-Russell Syndrome (SRS)",
        "omim": "#180860",
        "genes": ["IGF2 (*147470)", "H19 (*103280)", "CDKN1C (*600856)"],
        "locus": "11p15.5 (ICR1: H19/IGF2 + ICR2: KCNQ1OT1/CDKN1C)",
        "mechanism": "Paternal LOF at 11p15.5 → Loss of paternal IGF2 → Biallelic IGF2 silencing → Profound growth restriction",
        "imprinting_class": "Genomic Imprinting — Paternal LOF (11p15.5)",
        "same_locus_opposite": "BWS (Beckwith-Wiedemann Syndrome) — Maternal LOF same locus → OVERGROWTH (OPPOSITE phenotype)",
        "prevalence": "~1:30,000–1:100,000 (underdiagnosed)",
        "cohort_size": n,
        "seed": SEED,
        "kpis": {
            "total_patients": n,
            "icr1_hypo_pct": 45,
            "upd7mat_pct": 10,
            "sga_universal_pct": 96,
            "hemihypotrophy_pct": round(sum(1 for p in cohort if p["hemihypotrophy"]) / n * 100),
            "neonatal_hypoglycemia_pct": round(sum(1 for p in cohort if p["neonatal_hypoglycemia"]) / n * 100),
            "epilepsy_pct": round(sum(1 for p in cohort if p["epilepsy"]) / n * 100),
            "gh_therapy_pct": round(sum(1 for p in cohort if p["gh_therapy"]) / n * 100),
            "cpp_pct": round(sum(1 for p in cohort if p["cpp"]) / n * 100),
            "mean_birth_weight_sds": round(sum(p["birth_weight_sds"] for p in cohort) / n, 1),
            "mean_icr1_meth": round(sum(p["icr1_methylation_pct"] for p in cohort) / n, 1),
            "mean_age_diagnosis_y": round(sum(p["age_diagnosis_y"] for p in cohort) / n, 1),
        },
        "mechanism_breakdown": {m["label"]: m["n"] for m in MECHANISMS},
        "phenotype_breakdown": {pg["label"]: pg["n"] for pg in PHENOTYPE_GROUPS},
        "nh_criteria": [
            {"criterion": "SGA (birth weight or length ≤-2 SDS)", "prevalence": "96%"},
            {"criterion": "Postnatal growth restriction (height ≤-2 SDS)", "prevalence": "100%"},
            {"criterion": "Relative macrocephaly at birth (HC SDS - weight SDS ≥1.5)", "prevalence": "70%"},
            {"criterion": "Body asymmetry (hemihypotrophy, ≥0.5 cm LLD)", "prevalence": "50-65%"},
            {"criterion": "Protruding forehead / frontal bossing (triangular face)", "prevalence": "75%"},
            {"criterion": "Feeding difficulties (NG tube or BMI <-2 SDS at 24 months)", "prevalence": "70%"},
        ],
        "key_alerts": [
            {"level": "DANGER", "msg": "FASTING ABSOLUTE CI — hypoglycemia risk → seizures → brain injury"},
            {"level": "DANGER", "msg": "KETOGENIC DIET CONTRAINDICATED — severe hypoglycemia in SRS"},
            {"level": "DANGER", "msg": "Phenobarbital AVOID in infants — worsens already severe feeding difficulties"},
            {"level": "WARN", "msg": "VPA MODERATE RISK — weight gain in thin SRS patients; hepatic monitoring"},
            {"level": "WARN", "msg": "GH therapy: start ASAP (before age 4); monitor IGF-1 + bone age"},
            {"level": "INFO", "msg": "Cornstarch overnight = MANDATORY from neonatal period (Level A)"},
            {"level": "INFO", "msg": "Cognition NORMAL — KEY contrast with Temple Syndrome (80% mild-moderate ID)"},
            {"level": "INFO", "msg": "Same locus as BWS — H19-ICR1 paternal LOF (SRS) vs maternal LOF (BWS)"},
        ],
        "diagnostic_pathway": [
            {"step": 1, "test": "Clinical: Netchine-Harbison score (NH)", "threshold": "≥4/6 = test", "detects": "Clinical suspicion"},
            {"step": 2, "test": "MS-MLPA / methylation-specific 11p15.5 (ICR1)", "yield": "~60% (ICR1 hypo + copy number)", "detects": "H19-ICR1 hypomethylation, paternal deletion, maternal duplication"},
            {"step": 3, "test": "Chromosomal SNP array", "yield": "+10-15%", "detects": "Maternal UPD7 (LOH chr7), structural variants"},
            {"step": 4, "test": "CDKN1C sequencing (NGS)", "yield": "+5%", "detects": "CDKN1C pathogenic variants"},
            {"step": 5, "test": "Clinical diagnosis (NH ≥4/6 with negative molecular)", "yield": "~35% remain unsolved", "detects": "Clinical SRS without molecular confirmation"},
        ],
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

def get_breakdown():
    cohort = _COHORT
    return {
        "patients": cohort,
        "mechanism_details": MECHANISMS,
        "phenotype_groups": PHENOTYPE_GROUPS,
        "variants": VARIANTS,
        "aed_guide": AEDS,
        "management_protocols": MANAGEMENTS,
        "differentials": DIFFERENTIALS,
        "biomarker_thresholds": {
            "birth_weight_sds_srs": "<= -2.0 SDS (often -3.5 SDS)",
            "birth_length_sds_srs": "<= -2.0 SDS",
            "icr1_methylation_normal": "~50% (paternal allele methylated)",
            "icr1_methylation_srs_threshold": "<40% (significant hypomethylation)",
            "igf2_serum_low": "<100 ng/mL (normal >150 ng/mL) — not routinely measured",
            "glucose_fasting_danger": "<3.0 mmol/L (50 mg/dL) — hypoglycemia threshold",
            "igf1_gh_deficient": "<-2 SDS for age — partial GH deficiency in 40%",
            "nh_score_diagnostic": "≥4/6 criteria",
            "leg_length_discrepancy_treat": ">0.5 cm (shoe lift); >2 cm (surgical consideration)",
        },
        "compared_with_bws": {
            "locus": "Same: 11p15.5",
            "icr_involved": "SRS: ICR1 paternal hypomethylation → IGF2 absent | BWS: ICR1 maternal hypermethylation → IGF2 excess",
            "igf2": "SRS: IGF2 silenced biallelically | BWS: IGF2 expressed biallelically",
            "growth": "SRS: profound restriction (-3.5 SDS) | BWS: overgrowth (macrosomia, LGA)",
            "hypoglycemia": "SRS: fasting-type (low IGF2; normal insulin) | BWS: hyperinsulinism (KATP or focal)",
            "asymmetry": "SRS: hemihypotrophy (one side smaller) | BWS: hemihypertrophy (one side larger)",
            "tumour_risk": "SRS: minimal | BWS: Wilms (7%), hepatoblastoma (1.3%), adrenocortical (1%)",
            "parent_of_origin": "SRS: paternal allele LOF | BWS: maternal allele LOF",
            "principle": "SAME LOCUS — OPPOSITE PARENT — OPPOSITE PHENOTYPE",
        },
        "compared_with_temple": {
            "locus": "SRS: 11p15.5 | Temple: 14q32.3",
            "gene": "SRS: IGF2/H19 | Temple: DLK1/MEG3",
            "mechanism": "SRS: ICR1 paternal hypo → IGF2 absent | Temple: IG-DMR paternal hypo → DLK1 absent",
            "growth": "Both: SGA + short stature (SRS more severe: -3.5 vs -2.5 SDS)",
            "asymmetry": "SRS: hemihypotrophy YES (~55%) | Temple: NO (symmetric)",
            "cognition": "SRS: NORMAL | Temple: 80% mild-moderate ID (IQ 65-80)",
            "cpp": "SRS: 25-30% | Temple: 50-60% (Level A GnRH analog)",
            "obesity": "SRS: thin (BMI low) | Temple: truncal obesity 60-70%",
            "test": "SRS: methylation 11p15.5 | Temple: methylation 14q32.3 — DIFFERENT PANELS",
        },
    }

def get_definitions():
    return {
        "disease": "Silver-Russell Syndrome (SRS)",
        "omim": "#180860",
        "gene_definitions": [
            {"gene": "IGF2", "omim": "*147470", "protein": "Insulin-Like Growth Factor 2 (70 aa precursor → 7.5 kDa mature)",
             "location": "11p15.5", "expression": "PATERNALLY expressed only (maternal IGF2 silenced by ICR1-CTCF)",
             "function": "Principal fetal growth factor; binds IGF1R and M6PR/IGF2R; stimulates cell proliferation, differentiation, organ growth",
             "in_srs": "ABSENT (biallelically silenced) — loss of the sole expressed IGF2 allele → profound growth restriction",
             "contrast_bws": "In BWS: OVEREXPRESSED biallelically (both alleles active) → macrosomia"},
            {"gene": "H19", "omim": "*103280", "protein": "H19 lncRNA (non-coding RNA, imprinted locus)",
             "location": "11p15.5 (adjacent to IGF2, reciprocal regulation)", "expression": "MATERNALLY expressed only (paternal H19 silenced by ICR1 methylation)",
             "function": "Growth repressor lncRNA; sequesters miR-675; regulates IGFBP3; tumour suppressor in some contexts",
             "in_srs": "BIALLELICALLY expressed (paternal H19 now active since ICR1 unmethylated) → excess H19 → additional growth suppression",
             "contrast_bws": "In BWS: H19 silenced biallelically (excess paternal ICR1 methylation) → loss of tumour suppressor"},
            {"gene": "CDKN1C", "omim": "*600856", "protein": "p57KIP2 (316 aa) — cyclin-dependent kinase inhibitor 1C",
             "location": "11p15.5 ICR2 region (KCNQ1 locus)", "expression": "MATERNALLY expressed (paternal CDKN1C silenced by KCNQ1OT1 lncRNA)",
             "function": "Cell cycle brake (G1 arrest); growth restricting; tumour suppressor",
             "in_srs": "ICR2 region INTACT in most SRS (not the primary defect); CDKN1C gain-of-function variants cause SRS when maternal",
             "contrast_bws": "In BWS: CDKN1C LOF (maternal deletion or hypermethylation silences CDKN1C) → cell cycle unchecked → overgrowth"},
        ],
        "imprinting_concepts": [
            {"term": "ICR1 (H19/IGF2 Imprinting Control Region 1)", "definition": "CpG island at 11p15.5 controlling H19 and IGF2 reciprocally. METHYLATED on paternal allele = IGF2 expressed (paternal). UNMETHYLATED on maternal allele = CTCF binds = IGF2 silenced maternally. In SRS: ICR1 unmethylated on BOTH alleles → biallelic IGF2 silencing"},
            {"term": "CTCF (CCCTC-binding factor)", "definition": "Insulator protein that binds unmethylated ICR1. Binding blocks IGF2 enhancers from activating IGF2. In SRS: CTCF binds both alleles (both unmethylated) → IGF2 silent on both"},
            {"term": "Genomic imprinting", "definition": "Epigenetic marking of genes based on parental origin. Imprinted genes expressed from ONLY one parental allele. ICR methylation determines which allele is active. Disruption causes disease depending on which parent's allele is lost"},
            {"term": "H19-ICR1 hypomethylation", "definition": "The most common SRS mechanism (~45%). Paternal ICR1 fails to acquire or maintain methylation → CTCF binds paternal allele (normally blocked by methylation) → paternal IGF2 silenced → total IGF2 loss"},
            {"term": "Maternal UPD7 (upd(7)mat)", "definition": "Two copies of maternal chromosome 7, no paternal chr7. Causes SRS-like phenotype by losing paternally expressed genes on chr7 (GRB10, SGCE/PEG10 region). Milder SRS; café-au-lait spots common"},
            {"term": "Netchine-Harbison (NH) score", "definition": "Clinical diagnostic tool for SRS. 6 criteria: SGA, postnatal growth restriction, relative macrocephaly, body asymmetry, frontal bossing, feeding difficulties. Score ≥4/6 = investigate for SRS. Score ≥4/6 with negative molecular = clinical SRS diagnosis"},
            {"term": "Hemihypotrophy", "definition": "One side of the body smaller than the other (one limb, one side of face, or entire hemi-body). Key SRS feature (~50-65%). Reflects somatic mosaicism in some; constitutional asymmetry in others. DISTINGUISHES SRS from Temple Syndrome (no asymmetry in Temple)"},
            {"term": "Relative macrocephaly", "definition": "Head circumference SDS disproportionately larger than weight SDS (HC SDS minus weight SDS ≥1.5). Brain sparing: brain growth preserved despite profound body growth restriction. NH criterion #3"},
        ],
        "drug_classes": [
            {"drug": "Recombinant GH (rhGH)", "examples": "Genotropin, Norditropin, Humatrope", "mechanism": "Replaces/supplements GH; activates IGF-1 axis; promotes linear growth; improves lean body mass",
             "evidence_srs": "Level A (FDA ODA 2003) — highest evidence; start before age 4; +7-10 cm adult height",
             "monitoring": "IGF-1 every 6 months; bone age annually; avoid IGF-1 >+2 SDS (cancer risk theoretical)"},
            {"drug": "GnRH analogs", "examples": "Leuprolide, triptorelin", "mechanism": "Suppress HPG axis; delay bone age advancement; extend GH window",
             "evidence_srs": "Level B (less robust than Temple Syndrome Level A)",
             "monitoring": "Bone age 6-monthly; LH/FSH on GnRH stimulation"},
            {"drug": "Cornstarch (uncooked)", "examples": "Glycosade, regular maize starch", "mechanism": "Slow-release glucose polymer; digested slowly → sustained blood glucose overnight",
             "evidence_srs": "Level A — MANDATORY in SRS; prevents overnight fasting hypoglycemia and hypoglycemic seizures",
             "monitoring": "Fasting glucose log; blood glucose monitor"},
        ],
        "key_facts_exam": [
            "SRS = H19-ICR1 HYPOMETHYLATION (paternal allele, ~45%) → BIALLELIC IGF2 SILENCING → GROWTH RESTRICTION",
            "BWS = H19-ICR1 HYPERMETHYLATION (maternal allele) → BIALLELIC IGF2 EXPRESSION → OVERGROWTH — SAME LOCUS, OPPOSITE PARENT",
            "HEMIHYPOTROPHY = SRS signature (one side smaller) vs HEMIHYPERTROPHY = BWS (one side larger, Wilms risk)",
            "COGNITION NORMAL in SRS — KEY contrast with Temple Syndrome (80% mild-moderate ID)",
            "FASTING ABSOLUTE CI — hypoglycemia → seizures → brain injury; cornstarch MANDATORY",
            "KD (Ketogenic diet) CONTRAINDICATED in SRS — carbohydrate restriction → severe hypoglycemia",
            "PHENOBARBITAL AVOID in SRS neonates — impairs feeding which is already severely compromised",
            "GH therapy Level A (FDA 2003) — must start ASAP, ideally before age 4 — +7-10 cm adult height",
            "NH score ≥4/6 → molecular testing (MS-MLPA first → SNP array → CDKN1C sequencing)",
            "~35% molecularly unsolved — clinical SRS if NH ≥4/6 after full negative workup",
            "CPP 25-30% (less than Temple 50-60%); GnRH analog Level B (vs Temple Level A)",
            "SRS vs Temple: ASYMMETRY = SRS; NO asymmetry + ID = Temple; DIFFERENT methylation panels (11p15.5 vs 14q32.3)",
            "ART (IVF/ICSI) increases ICR1 hypomethylation risk ~10x — take reproductive history",
            "Recurrence: ICR1 hypo de novo <1%; CDKN1C maternal variant = 50% from carrier mother (AD)",
        ],
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

if __name__ == "__main__":
    import json
    ov = get_overview()
    br = get_breakdown()
    df = get_definitions()
    print(f"Overview KPIs: {json.dumps(ov['kpis'], indent=2)}")
    print(f"Cohort size: {len(br['patients'])}")
    print(f"Definitions gene count: {len(df['gene_definitions'])}")
    print("SRS dashboard OK")
