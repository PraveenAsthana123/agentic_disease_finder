#!/usr/bin/env python3
"""Hereditary-Dystroglycanopathy-Atlas — Complete 8-Gene Atlas (Alpha-Dystroglycan O-Glycosylation Disorders)
FKRP    (Fukutin-Related Protein; 495 aa; 19q13.32; AR;
         LGMD2I / MDC1C / MEB / WWS spectrum;
         L276I Caucasian founder — most common AR LGMD Northern Europe;
         mandatory cardiac surveillance — DCM up to 30%;
         seed SEED_BASE+0) .
FKTN    (Fukutin; 461 aa; 9q31.2; AR;
         Fukuyama Congenital Muscular Dystrophy (FCMD) — most common CMD Japan;
         3 kb retrotransposon SVA insertion founder — 1 in 10 000 Japanese births;
         dilated cardiomyopathy onset 2nd–3rd decade — cardiac transplant required;
         LGMD2M outside Japan;
         seed SEED_BASE+1) .
POMT1   (Protein O-Mannosyltransferase 1; 747 aa; 9q34.13; AR;
         Walker-Warburg Syndrome type 1 (WWS1) — most severe dystroglycanopathy;
         lissencephaly + eye malformations + CMD — median survival < 3 yr;
         LGMD2K (milder alleles);
         seed SEED_BASE+2) .
POMT2   (Protein O-Mannosyltransferase 2; 750 aa; 14q24.3; AR;
         Walker-Warburg Syndrome type 2 (WWS2) + LGMD2N;
         cobblestone lissencephaly + Dandy-Walker + CMD;
         slightly less severe than POMT1 WWS;
         seed SEED_BASE+3) .
POMGNT1 (Protein O-Linked Mannose GlcNAc-Transferase 1; 740 aa; 1p34.1; AR;
         Muscle-Eye-Brain Disease (MEB) / Santavuori congenital muscular dystrophy;
         progressive myopia + retinal dysplasia + CMD + cerebellar cysts;
         DISTINGUISH from Fukuyama: progressive high myopia + ERG flat;
         seed SEED_BASE+4) .
LARGE1  (LARGE Xylosyl- and Glucuronyltransferase 1; 756 aa; 22q12.3; AR;
         MDC1D — most severe post-glycosylation modifier defect;
         profound intellectual disability + white matter abnormalities;
         cerebellar cysts; LARGE2 does NOT compensate;
         seed SEED_BASE+5) .
ISPD    (Isoprenoid Synthase Domain Containing / CRPPA; 352 aa; 7p21.2; AR;
         WWS subtype — cardiomyopathy + Leigh-like brain MRI + cobblestone cortex;
         p.Arg272Cys founder variant Northern European;
         CMD + brain malformations + cardiac;
         seed SEED_BASE+6) .
GMPPB   (GDP-Mannose Pyrophosphorylase B; 395 aa; 3p24.3; AR;
         LGMD2T / Myasthenic Dystroglycanopathy;
         UNIQUE: fluctuating weakness + abnormal decrement on EMG — mimics myasthenia;
         pyridostigmine RESPONSIVE (only dystroglycanopathy with neuromuscular junction component);
         CK elevated 5–50x; normal intelligence; no brain abnormalities;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1958–1965)
"""

import random

SEED_BASE = 1958

DYSTROGLYCANOPATHY_GENES = [
    # -- FKRP — Fukutin-Related Protein ------------------------------------------
    {
        "gene": "FKRP",
        "alt_name": "FKRP (LGMD2I / MDC1C / Fukutin-Related Protein)",
        "protein": (
            "FKRP -- 19q13.32 AR -- FKRP-495aa -- "
            "LGMD2I-Most-Common-AR-LGMD-Northern-Europe -- "
            "L276I-Caucasian-Founder-80pct-LGMD2I -- "
            "MDC1C-Severe-CMD-Brain-Abnormalities -- "
            "Cardiomyopathy-DCM-30pct-Mandatory-Cardiac-Surveillance -- "
            "Respiratory-Failure-50pct-NIV-Mandatory"
        ),
        "locus": "19q13.32",
        "protein_size": "495 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "LGMD2I: childhood to early adult onset; proximal girdle weakness; "
            "p.Leu276Ile (L276I) homozygotes — milder LGMD phenotype, ambulant into 4th decade; "
            "MDC1C: neonatal/congenital onset — hypotonia, feeding difficulty, respiratory failure from birth; "
            "Cardiomyopathy: insidious onset 2nd–3rd decade in LGMD2I; earlier and more severe in MDC1C; "
            "Respiratory: nocturnal hypoventilation develops before daytime symptoms"
        ),
        "key_biomarker": (
            "CK: markedly elevated 10–50× normal (range 1000–10000 U/L) even in mild cases; "
            "alpha-dystroglycan immunohistochemistry: reduced/absent glycosylation (VIA4-1 or IIH6 antibodies); "
            "MRI muscle: posterior compartment predominance (gastrocnemius, hamstrings, glutei); "
            "cardiac MRI/echo: DCM — ICD if EF < 35% or arrhythmia; "
            "spirometry: FVC decline (NIV threshold FVC < 50% predicted); "
            "molecular: FKRP p.Leu276Ile (c.826C>A) — confirm both alleles (founder + pathogenic second)"
        ),
        "pathognomonic": (
            "LGMD phenotype + markedly elevated CK + reduced alpha-DG glycosylation on biopsy = "
            "dystroglycanopathy until gene confirmed; "
            "FKRP L276I/L276I: mildest phenotype — preserved ambulation into 30–40s; "
            "FKRP compound het (L276I + truncating): more severe — MDC1C phenotype; "
            "DISTINGUISH from DMD: DMD X-linked; dystrophin absent on biopsy; no brain MRI abnormalities; "
            "DISTINGUISH from LGMD2C–2F sarcoglycanopathies: sarcoglycan IHC absent; FKRP normal sarcoglycan"
        ),
        "treatment": (
            "Respiratory: NIV (BiPAP) when FVC < 50% predicted or nocturnal desaturation documented; "
            "annual spirometry + overnight oximetry — mandatory; "
            "Cardiac: annual ECG + echo; ICD if EF < 35% or sustained arrhythmia; "
            "cardiac transplantation considered in isolated DCM with preserved skeletal muscle; "
            "Physiotherapy: aquatic therapy + stretching to delay contractures; "
            "AVOID: prolonged immobility (accelerates decline); vigorous eccentric exercise; "
            "Gene therapy: SRP-9003 (scAAV9-FKRP) — Phase 1/2 trials ongoing 2025; "
            "Ribose supplementation: Level C evidence only; not standard; "
            "Corticosteroids: NOT effective in FKRP — distinct from DMD mechanism; do NOT use"
        ),
        "critical_flags": [
            "FKRP-CARDIAC-MANDATORY: DCM in 20–30% LGMD2I; annual echo from diagnosis; ICD if EF <35% or Lp arrhythmia; cardiac death is leading cause of FKRP mortality — do NOT skip cardiac surveillance even in ambulant patients",
            "FKRP-L276I-FOUNDER: p.Leu276Ile (L276I) c.826C>A in 80% Northern European LGMD2I alleles; allele-specific assay or NGS confirmation required; heterozygous L276I with clinical LGMD = find second allele (full gene sequencing)",
            "FKRP-STEROIDS-NOT-EFFECTIVE: FKRP dystroglycanopathy is NOT a dystrophinopathy; corticosteroids do not slow progression and cause side effects; do NOT start prednisolone/deflazacort",
            "FKRP-ALPHA-DG-IHC-REQUIRED: molecular alone insufficient — confirm reduced alpha-DG glycosylation on muscle biopsy IHC; normal result does NOT exclude FKRP (some variants preserve partial glycosylation)",
            "FKRP-RESPIRATORY-FIRST: respiratory failure can precede significant limb weakness in MDC1C; ANY neonatal hypotonia + high CK + feeding difficulty = check FKRP; start NIV early before CO2 retention",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- FKTN — Fukutin / Fukuyama CMD -------------------------------------------
    {
        "gene": "FKTN",
        "alt_name": "FKTN (Fukuyama CMD / LGMD2M / Fukutin)",
        "protein": (
            "FKTN -- 9q31.2 AR -- FKTN-461aa -- "
            "Fukuyama-Congenital-Muscular-Dystrophy-FCMD-Japan-1in10000-Births -- "
            "3kb-SVA-Retrotransposon-Insertion-Founder-Allele-87pct-Japanese-FCMD -- "
            "Dilated-Cardiomyopathy-Onset-2nd-3rd-Decade-Cardiac-Transplant-Required -- "
            "Cobblestone-Lissencephaly-Pachygyria-Cortical-Dysplasia -- "
            "LGMD2M-Non-Founder-Outside-Japan"
        ),
        "locus": "9q31.2",
        "protein_size": "461 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "FCMD (founder): congenital onset — floppy infant; CMD from birth; "
            "ambulation: rarely achieved; if achieved, lost by age 10 yr; "
            "intellectual disability: mild to moderate — IQ 30–60; "
            "epilepsy: onset 1st year in ~50%; "
            "Cardiomyopathy: clinically overt by 2nd–3rd decade; "
            "LGMD2M (non-founder): adolescent/adult onset; milder; ambulation preserved longer"
        ),
        "key_biomarker": (
            "Muscle biopsy: reduced alpha-DG glycosylation (IIH6 antibody); dystrophic pattern + type 1 fibre predominance; "
            "Brain MRI: pachygyria-polymicrogyria + lissencephaly posterior > anterior; "
            "cerebellar cysts + white matter abnormalities — PATHOGNOMONIC FCMD; "
            "CK: severely elevated 10–50× (neonatal >1000 U/L); "
            "echo/cardiac MRI: DCM (assess annually from age 10); "
            "molecular: 3kb SVA insertion in 3'UTR — NOT detected by standard exome sequencing; "
            "requires MLPA or long-read sequencing for founder allele detection"
        ),
        "pathognomonic": (
            "Japanese child + CMD + cobblestone lissencephaly + cerebellar cysts = FCMD until proven; "
            "DISTINGUISH from Walker-Warburg (POMT1/POMT2): WWS more severe lissencephaly; earlier death; no Japanese founder; "
            "DISTINGUISH from Muscle-Eye-Brain (POMGNT1): MEB = progressive myopia + ERG changes; "
            "3kb SVA founder insertion + compound het = FCMD molecular confirmation; "
            "DISTINGUISH from LAMA2 CMD (merosin deficiency): LAMA2 — normal DG glycosylation; merosin absent on biopsy"
        ),
        "treatment": (
            "Cardiac: annual echo from age 10; ACE-i/ARB initiated for DCM; "
            "ICD if EF < 35% or sustained arrhythmia; cardiac transplantation in isolated DCM with stable skeletal; "
            "Epilepsy: valproate or levetiracetam; avoid enzyme-inducing AEDs; "
            "Respiratory: annual spirometry; NIV (BiPAP) when FVC < 50%; "
            "Orthopaedic: ankle-foot orthoses; Achilles tendon release if needed; "
            "AVOID: immobility (accelerates DCM); vigorous eccentric exercise; "
            "Gene therapy: exon-skipping of SVA insertion in preclinical studies; "
            "Corticosteroids: NOT standard in FCMD; limited evidence"
        ),
        "critical_flags": [
            "FKTN-SVA-EXOME-MISSED: 3kb SVA retrotransposon insertion in FKTN 3'UTR is NOT detected by standard WES/panel sequencing; ALL suspected FCMD require MLPA or Southern blot or long-read sequencing for founder allele; missed diagnosis in Japanese patients common",
            "FKTN-CARDIAC-TRANSPLANT: DCM in FCMD can be severe enough to require cardiac transplantation; isolated cardiomyopathy may dominate in adolescent/adult LGMD2M; annual echo mandatory from age 10 in ALL FKTN patients",
            "FKTN-COBBLESTONE-PATHOGNOMONIC: pachygyria + cerebellar cysts on MRI + Japanese ethnicity = FCMD until excluded; brain MRI mandatory in all suspected CMD",
            "FKTN-OUTSIDE-JAPAN-LGMD2M: non-founder FKTN mutations cause LGMD2M (milder, no brain malformations, ambulation preserved); do NOT apply FCMD prognosis to non-Japanese FKTN patients without confirming the founder allele",
            "FKTN-EPILEPSY-EARLY: seizures in first year of life in ~50% FCMD; EEG and neurology review at diagnosis; VPA is first-line but check carnitine levels (VPA-induced depletion in myopathic patients)",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- POMT1 — Protein O-Mannosyltransferase 1 ---------------------------------
    {
        "gene": "POMT1",
        "alt_name": "POMT1 (Walker-Warburg Syndrome type 1 / LGMD2K)",
        "protein": (
            "POMT1 -- 9q34.13 AR -- POMT1-747aa -- "
            "Walker-Warburg-Syndrome-WWS-Type1-Most-Severe-Dystroglycanopathy -- "
            "Lissencephaly-Type2-Cobblestone-Hydrocephalus-Encephalocele-Lethal -- "
            "Eye-Malformations-Microphthalmia-Cataracts-Retinal-Dysplasia -- "
            "Median-Survival-Less-3-Years -- "
            "LGMD2K-Mild-Alleles-Adolescent-Adult-Onset"
        ),
        "locus": "9q34.13",
        "protein_size": "747 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "WWS (severe alleles): congenital — stillbirth or early neonatal death; "
            "surviving infants: profound hypotonia + severe intellectual disability; "
            "median survival < 3 yr (respiratory failure); "
            "LGMD2K (milder missense alleles): adolescent/adult onset; ambulant; "
            "cognitive impairment mild to absent in LGMD2K; "
            "Eye: microphtalmia/cataracts present at birth in WWS"
        ),
        "key_biomarker": (
            "Brain MRI (prenatal US or neonatal MRI): cobblestone lissencephaly type II + "
            "hydrocephalus + posterior fossa abnormalities (Dandy-Walker variant); "
            "eye exam: microphthalmia + anterior segment dysgenesis + cataract + retinal dysplasia; "
            "CK: severely elevated (>5000 U/L) in CMD/WWS; elevated in LGMD2K (3–30×); "
            "muscle biopsy: dystrophic + severely reduced alpha-DG glycosylation; "
            "molecular: POMT1 biallelic pathogenic variants — truncating = WWS; missense = LGMD2K"
        ),
        "pathognomonic": (
            "Cobblestone lissencephaly + CMD + eye malformations (classic triad) = WWS; "
            "PATHOGNOMONIC MRI: cobblestone cortex (bumpy outer surface) + agyric inner surface = "
            "TYPE II lissencephaly (distinct from smooth type I of DCX/LIS1); "
            "DISTINGUISH from POMT2 WWS: clinically indistinguishable — gene testing required; "
            "DISTINGUISH from POMGNT1 MEB: MEB = more preserved cortex + progressive myopia dominant; "
            "DISTINGUISH from LAMA2 merosin-def CMD: LAMA2 = normal cortex; leukodystrophy not lissencephaly"
        ),
        "treatment": (
            "WWS: palliative — ventriculoperitoneal shunt for hydrocephalus; respiratory support; "
            "feeding: nasogastric or gastrostomy for nutritional support; "
            "AEDs for seizures (levetiracetam, valproate); "
            "Ophthalmology: cataracts — surgical extraction in first weeks if possible; "
            "LGMD2K: physiotherapy; respiratory surveillance; cardiac echo annually; "
            "Gene therapy: preclinical; AAV9-POMT1 in mouse models; human trials not yet open; "
            "Prenatal: amniocentesis / CVS for affected family — 25% recurrence risk; "
            "Genetic counselling: carrier testing of parents + siblings mandatory"
        ),
        "critical_flags": [
            "POMT1-LETHAL-MEDIAN-3YR: WWS (severe alleles) — median survival < 3 years; respiratory failure dominant; family counselling and palliative planning essential at diagnosis; do NOT delay comfort goals discussion",
            "POMT1-PRENATAL-ULTRASOUND: cobblestone lissencephaly visible on fetal ultrasound from 20 weeks; hydrocephalus + posterior fossa cysts early sign; refer to fetal MRI if US suggests brain malformation",
            "POMT1-LGMD2K-MILD-ALLELES: milder missense variants in POMT1 → LGMD2K (normal cognition; ambulant); do NOT apply WWS prognosis to LGMD2K; genotype-phenotype correlation critical",
            "POMT1-EYE-EXAM-MANDATORY: anterior segment dysgenesis + cataracts + retinal dysplasia in WWS; ophthalmology review within first week of life; early cataract surgery improves visual development",
            "POMT1-ALPHA-DG-PANEL-FIRST: alpha-DG IHC on muscle biopsy → if reduced glycosylation, sequence entire dystroglycanopathy panel (FKRP, FKTN, POMT1, POMT2, POMGNT1, LARGE1, ISPD, GMPPB); do not sequence single genes sequentially",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- POMT2 — Protein O-Mannosyltransferase 2 ---------------------------------
    {
        "gene": "POMT2",
        "alt_name": "POMT2 (Walker-Warburg Syndrome type 2 / LGMD2N)",
        "protein": (
            "POMT2 -- 14q24.3 AR -- POMT2-750aa -- "
            "Walker-Warburg-Syndrome-Type2-LGMD2N -- "
            "Cobblestone-Lissencephaly-Dandy-Walker-Malformation -- "
            "CMD-with-Brain-Eye-Involvement -- "
            "Slightly-Less-Severe-Than-POMT1-WWS -- "
            "Pontine-Hypoplasia-Cerebellar-Underdevelopment"
        ),
        "locus": "14q24.3",
        "protein_size": "750 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "WWS (severe alleles): congenital; "
            "survival slightly longer than POMT1 WWS (median 1–5 yr); "
            "LGMD2N (milder alleles): childhood to adult onset; proximal girdle weakness; "
            "intellectual disability: present in WWS (profound); variable in LGMD2N (mild); "
            "Eye: anterior segment dysgenesis + retinal dysplasia in WWS"
        ),
        "key_biomarker": (
            "MRI: cobblestone lissencephaly + Dandy-Walker malformation + "
            "pontine hypoplasia + cerebellar cysts; "
            "alpha-DG IHC: severely reduced glycosylation; "
            "CK: >5000 U/L in CMD; 3–20× in LGMD2N; "
            "eye: microphthalmia, cataract, retinal dysplasia on slit-lamp + ERG; "
            "molecular: POMT2 biallelic pathogenic variants; "
            "distinguish from POMT1 only by genetic testing — clinically overlapping"
        ),
        "pathognomonic": (
            "Cobblestone lissencephaly + CMD + eye malformations = WWS spectrum; "
            "POMT2 vs POMT1: clinically indistinguishable — molecular diagnosis essential; "
            "Dandy-Walker malformation as dominant posterior fossa finding favours POMT2 over POMT1 but NOT specific; "
            "DISTINGUISH from POMGNT1 MEB: MEB has less severe lissencephaly + progressive myopia dominant; "
            "LGMD2N: mild cognitive impairment + ambulant proximal weakness + markedly elevated CK"
        ),
        "treatment": (
            "WWS: palliative — hydrocephalus shunting; respiratory support; "
            "AEDs: levetiracetam or valproate for seizures; "
            "Ophthalmology: cataract surgery in first weeks; low-vision aids; "
            "LGMD2N: physiotherapy; annual cardiac echo; respiratory surveillance; "
            "Enzyme replacement: no approved therapy; "
            "Gene therapy: preclinical data for POMT2 rescue with AAV; "
            "Prenatal testing: CVS or amniocentesis in affected families"
        ),
        "critical_flags": [
            "POMT2-VS-POMT1-INDISTINGUISHABLE: WWS due to POMT2 is clinically indistinguishable from POMT1; sequencing panel testing required — do NOT limit to POMT1 if initial sequencing negative",
            "POMT2-DANDY-WALKER: Dandy-Walker malformation (enlarged posterior fossa cyst + cerebellar vermis agenesis) is a clue to POMT2 within WWS spectrum; brain MRI mandatory in all CMD",
            "POMT2-LGMD2N-UNDERDIAGNOSED: milder POMT2 alleles cause LGMD2N (adult-onset proximal weakness, elevated CK, no brain abnormalities); alpha-DG IHC on biopsy identifies candidate for panel testing",
            "POMT2-PONTINE-HYPOPLASIA: pontine underdevelopment causes swallowing and respiratory centre dysfunction; early assessment of bulbar function; gastrostomy if unsafe swallow; NIV threshold lower in CMD with brainstem involvement",
            "POMT2-HETEROZYGOUS-CARRIERS-NORMAL: carriers (1 pathogenic POMT2 allele) are phenotypically normal; carrier testing of parents + siblings essential for reproductive planning (25% recurrence per pregnancy)",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- POMGNT1 — Protein O-Linked Mannose GlcNAc-Transferase 1 ----------------
    {
        "gene": "POMGNT1",
        "alt_name": "POMGNT1 (Muscle-Eye-Brain Disease / MEB / Santavuori CMD)",
        "protein": (
            "POMGNT1 -- 1p34.1 AR -- POMGNT1-740aa -- "
            "Muscle-Eye-Brain-Disease-MEB-Santavuori-CMD -- "
            "Progressive-HIGH-Myopia-PATHOGNOMONIC-Clue -- "
            "Retinal-Dysplasia-Flat-ERG-Nystagmus -- "
            "Cerebellar-Cysts-Cerebellar-Dysplasia -- "
            "DISTINGUISH-from-Fukuyama-Progressive-Myopia-Key-Clue"
        ),
        "locus": "1p34.1",
        "protein_size": "740 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Congenital onset: hypotonia + nystagmus from birth; "
            "Visual: progressive myopia (very high, -10 to -25 D) develops in first years; "
            "nystagmus and visual impairment from birth/infancy; "
            "Intellectual disability: moderate to profound; "
            "Epilepsy: onset in first year in 60–70%; "
            "Ambulation: rarely achieved; "
            "Milder alleles: later onset LGMD; "
            "Survival: improved over WWS — majority survive into adulthood with support"
        ),
        "key_biomarker": (
            "Ophthalmology: progressive high myopia (> -8 D) + retinal dysplasia + flat ERG = "
            "key diagnostic clue separating MEB from FCMD/WWS; "
            "MRI brain: cerebellar cortical dysplasia + cysts (posterior fossa) + "
            "pachygyria (less severe than WWS); periventricular white matter signal; "
            "CK: markedly elevated (5–30× normal); "
            "alpha-DG IHC: reduced glycosylation; "
            "ERG: flat/severely reduced retinal function; "
            "molecular: POMGNT1 biallelic pathogenic variants"
        ),
        "pathognomonic": (
            "CMD + progressive HIGH MYOPIA + cerebellar cysts on MRI = MEB/POMGNT1 until proved; "
            "HIGH MYOPIA is the KEY DISTINGUISHING FEATURE: Fukuyama CMD (FKTN) rarely has high myopia; "
            "WWS (POMT1/POMT2) may have eye involvement but myopia rarely progressive and very high; "
            "flat ERG + nystagmus + CMD from birth = MEB triad; "
            "DISTINGUISH from LCA (Leber Congenital Amaurosis): LCA — no CMD, no high CK, brain MRI normal; "
            "DISTINGUISH from FCMD: FCMD — Japanese founder; cobblestone lissencephaly more anterior"
        ),
        "treatment": (
            "Eye: refraction correction (glasses/contact lenses) for myopia; "
            "ERG monitoring; retinal detachment surveillance (high myopia risk); "
            "Ophthalmology review every 6 months; "
            "AEDs: levetiracetam or valproate for seizures; avoid carbamazepine (myopathic CNS risk); "
            "Respiratory: NIV when FVC < 50%; annual spirometry; "
            "Physiotherapy + orthotics: AFOs; hydrotherapy; "
            "Communication: AAC devices for non-verbal patients; "
            "Cardiac: annual echo (DCM less common than FKRP/FKTN but monitor); "
            "Gene therapy: preclinical stage"
        ),
        "critical_flags": [
            "POMGNT1-MYOPIA-KEY-CLUE: progressive high myopia (> -8 D) in first year of life + CMD = POMGNT1/MEB until excluded; ophthalmology and gene panel testing mandatory; retinal detachment risk increases with age in high myopia",
            "POMGNT1-MEB-NOT-WWS: MEB has better survival than WWS; some patients survive into 3rd–4th decade with support; do NOT apply WWS prognosis to MEB patients",
            "POMGNT1-ERG-FLAT: retinal function is severely impaired from birth (flat ERG); formal ERG in first year of life; nystagmus + flat ERG + hypotonia = urgent ophthalmology + neurology + genetics referral",
            "POMGNT1-CEREBELLAR-CYSTS: posterior fossa cysts on MRI are characteristic of MEB; cysts may enlarge over time; serial neuroimaging recommended; hydrocephalus can develop",
            "POMGNT1-EPILEPSY-FREQUENT: 60–70% develop seizures in first year; EEG at diagnosis; levetiracetam preferred (minimal interactions, no liver enzyme induction in hepatically vulnerable myopathic patients)",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- LARGE1 — LARGE Xylosyl- and Glucuronyltransferase 1 ---------------------
    {
        "gene": "LARGE1",
        "alt_name": "LARGE1 (MDC1D / LGMD2L — Most Severe Post-Glycosylation Modifier Defect)",
        "protein": (
            "LARGE1 -- 22q12.3 AR -- LARGE1-756aa -- "
            "MDC1D-Severe-CMD-Profound-Intellectual-Disability -- "
            "White-Matter-Abnormalities-Periventricular-Leukodystrophy -- "
            "Cerebellar-Cysts-Cerebellar-Hypoplasia -- "
            "LARGE2-Does-NOT-Compensate-Despite-Shared-Function -- "
            "Extremely-Rare-Most-Severe-Post-Glycosylation-Modifying-Factor"
        ),
        "locus": "22q12.3",
        "protein_size": "756 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "CMD: neonatal/congenital onset; profound hypotonia; "
            "Intellectual disability: severe to profound (most severe among dystroglycanopathies); "
            "White matter: periventricular leukodystrophy visible from birth on MRI; "
            "Cerebellar: cerebellar hypoplasia + cysts; "
            "LGMD2L (milder alleles): adult onset; proximal weakness; cognition may be preserved; "
            "Respiratory: early failure requiring NIV in CMD form"
        ),
        "key_biomarker": (
            "MRI brain: extensive periventricular white matter signal + cerebellar cysts + "
            "pontine hypoplasia (more severe white matter involvement than FKTN/FKRP); "
            "alpha-DG IHC: severely reduced (LARGE1 adds terminal repeating units to matriglycan "
            "— CRITICAL for laminin binding; complete loss = complete loss of extracellular matrix attachment); "
            "CK: markedly elevated >5000 U/L (CMD); 3–20× in LGMD2L; "
            "molecular: LARGE1 biallelic pathogenic variants; "
            "LARGE2 expression: LARGE2 paralog does NOT compensate in muscle"
        ),
        "pathognomonic": (
            "CMD + profound intellectual disability + white matter leukodystrophy + cerebellar cysts = "
            "LARGE1/MDC1D most likely; "
            "SEVERITY: MDC1D is among the most severe CMD syndromes (more severe than FCMD); "
            "DISTINGUISH from FKTN FCMD: FCMD — Japanese founder; cobblestone lissencephaly not leukodystrophy; "
            "DISTINGUISH from POMT1/POMT2 WWS: WWS — cobblestone cortex (not white matter); earlier death; "
            "White matter abnormality + alpha-DG deficiency = LARGE1 top differential"
        ),
        "treatment": (
            "Palliative for CMD: gastrostomy; NIV (early); seizure management; "
            "Physiotherapy: range of motion; hydrotherapy; "
            "Communication: AAC devices; "
            "Cardiac: annual echo; "
            "Respiratory: NIV threshold lower in CMD with bulbar involvement; "
            "Steroids: NOT indicated; "
            "Gene therapy: LARGE1 gene large (>5kb) — AAV capacity challenges; biglycan as alternative substrate?"
        ),
        "critical_flags": [
            "LARGE1-LARGE2-NO-COMPENSATION: LARGE2 paralog is expressed in heart/kidney but NOT in skeletal muscle sufficiently to compensate for LARGE1 loss; do NOT reassure family that paralog will compensate",
            "LARGE1-WHITE-MATTER-PATHOGNOMONIC: periventricular white matter T2 signal + alpha-DG deficiency = LARGE1/MDC1D most likely diagnosis; must distinguish from LAMA2 (merosin) which also causes CMD + white matter signal",
            "LARGE1-MATRIGLYCAN-CRITICAL: LARGE1 adds xylose-glucuronate repeating units to matriglycan (alpha-DG glycan chain); without this, laminin (ECM) cannot bind; complete loss = loss of sarcolemmal stability; mechanism differs from POMT1/POMT2",
            "LARGE1-SEVERITY-WORST: MDC1D survival worse than FCMD; intensive palliative planning from diagnosis; early advance care planning discussion with family regarding respiratory failure end-of-life",
            "LARGE1-EXTREMELY-RARE: fewer than 30 reported cases worldwide; refer to specialist neuromuscular centre with dystroglycanopathy expertise before making management decisions",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- ISPD — Isoprenoid Synthase Domain Containing (CRPPA) ---------------------
    {
        "gene": "ISPD",
        "alt_name": "ISPD/CRPPA (CDP-Ribitol Pyrophosphorylase A — WWS Subtype with Cardiomyopathy)",
        "protein": (
            "ISPD -- 7p21.2 AR -- ISPD-352aa -- "
            "Walker-Warburg-Subtype-CDP-Ribitol-Pyrophosphorylase -- "
            "Cardiomyopathy-DCM-Distinctive-Among-Dystroglycanopathies -- "
            "Leigh-Like-Brain-MRI-Cobblestone-Cortex -- "
            "Arg272Cys-Northern-European-Founder-Variant -- "
            "CDP-Ribitol-Substrate-Enables-O-Mannosylation"
        ),
        "locus": "7p21.2",
        "protein_size": "352 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "CMD: neonatal/congenital onset — severe hypotonia + feeding difficulty; "
            "Brain: cobblestone lissencephaly + cerebellar/pontine hypoplasia at birth; "
            "Cardiomyopathy: DCM in subset — can be presenting feature in milder alleles; "
            "Leigh-like: mitochondrial-appearing brain MRI (basal ganglia + brainstem signal) — "
            "metabolic workup negative; MRI pattern alone suggests metabolic but is structural; "
            "LGMD (mild alleles): adult onset; proximal weakness; preserved cognition"
        ),
        "key_biomarker": (
            "MRI brain: cobblestone cortex + cerebellar/pontine hypoplasia + "
            "DISTINCTIVE basal ganglia signal mimicking Leigh syndrome; "
            "alpha-DG IHC: reduced glycosylation; "
            "CK: markedly elevated; "
            "Echo/cardiac MRI: DCM — evaluate in ALL ISPD patients at diagnosis; "
            "Metabolic screen: NORMAL (excludes true mitochondrial disease — Leigh-like MRI is structural NOT metabolic); "
            "molecular: ISPD biallelic pathogenic variants; p.Arg272Cys founder (Northern European)"
        ),
        "pathognomonic": (
            "CMD + cobblestone lissencephaly + DCM = ISPD top differential; "
            "LEIGH-LIKE MRI + CMD + alpha-DG deficiency = ISPD/CRPPA (rare but distinctive); "
            "DISTINGUISH from true mitochondrial Leigh disease: Leigh = normal CK; no CMD; lactate HIGH; "
            "DISTINGUISH from POMT1 WWS: POMT1 — no DCM; no basal ganglia signal; "
            "ISPD p.Arg272Cys homozygous: Northern European founder — may present in LGMD clinic"
        ),
        "treatment": (
            "Cardiac: echo at diagnosis + annually; ACE-i/ARB for DCM; ICD if EF < 35%; "
            "Respiratory: NIV early in CMD; gastrostomy if unsafe swallow; "
            "AEDs: levetiracetam for seizures; "
            "Physiotherapy: range of motion; AFOs; "
            "AVOID: diagnostic delay from Leigh syndrome assumption — metabolic screen will be NORMAL; "
            "Prenatal: CVS/amniocentesis; 25% recurrence; "
            "Gene therapy: preclinical; CDP-ribitol supplementation theoretical"
        ),
        "critical_flags": [
            "ISPD-LEIGH-LIKE-MRI-TRAP: basal ganglia + brainstem signal on MRI mimics Leigh syndrome; metabolic screen (lactate, pyruvate, ETC enzymes, mtDNA) will be NORMAL; alpha-DG IHC + muscle biopsy distinguishes ISPD from true Leigh",
            "ISPD-CARDIAC-DISTINCTIVE: DCM in ISPD is more common and earlier than in most other dystroglycanopathies; echo at diagnosis essential; cardiomyopathy can dominate in milder LGMD presentations of ISPD",
            "ISPD-ARG272CYS-FOUNDER: p.Arg272Cys (c.814C>T) is a Northern European founder allele; homozygous or compound het LGMD patients of this ancestry should have ISPD screened specifically",
            "ISPD-CDP-RIBITOL-MECHANISM: ISPD (CRPPA) produces CDP-ribitol, the substrate for FKTN and FKRP to add ribitol-5-phosphate to alpha-DG; loss of ISPD = loss of substrate for downstream enzymes; upstream block in same pathway",
            "ISPD-PANEL-NOT-SEQUENTIAL: as with all dystroglycanopathies — alpha-DG IHC positive → sequence full dystroglycanopathy panel simultaneously; sequential gene testing wastes months",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- GMPPB — GDP-Mannose Pyrophosphorylase B ----------------------------------
    {
        "gene": "GMPPB",
        "alt_name": "GMPPB (LGMD2T — Myasthenic Dystroglycanopathy; Pyridostigmine Responsive)",
        "protein": (
            "GMPPB -- 3p24.3 AR -- GMPPB-395aa -- "
            "LGMD2T-Myasthenic-Dystroglycanopathy -- "
            "Fluctuating-Weakness-Abnormal-Decrement-EMG-Mimics-Myasthenia -- "
            "Pyridostigmine-RESPONSIVE-ONLY-Dystroglycanopathy-with-NMJ-Component -- "
            "Normal-Intelligence-No-Brain-Abnormalities-KEY-DDx -- "
            "CK-Elevated-5-50x"
        ),
        "locus": "3p24.3",
        "protein_size": "395 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Childhood to adult onset; variable; "
            "Initial presentation: exercise-induced fatigable weakness — mimics myasthenia gravis; "
            "Fluctuating weakness: worse at end of day; ptosis in some; "
            "Intellect: NORMAL (key DDx from other dystroglycanopathies); "
            "Brain MRI: NORMAL (no cobblestone, no white matter, no cerebellar cysts); "
            "Progression: slow; ambulant for decades; "
            "Severe form: congenital onset with intellectual disability — rare"
        ),
        "key_biomarker": (
            "EMG: abnormal decrement on repetitive nerve stimulation (> 10%) — mimics myasthenic syndrome; "
            "CK: markedly elevated 5–50× (not expected in true myasthenia gravis — KEY CLUE); "
            "AChR and MuSK antibodies: NEGATIVE (seronegative myasthenia workup); "
            "alpha-DG IHC: reduced glycosylation; "
            "Single-fibre EMG: increased jitter — NMJ dysfunction confirmed; "
            "Brain MRI: NORMAL — key distinguishing feature from other dystroglycanopathies; "
            "molecular: GMPPB biallelic pathogenic variants"
        ),
        "pathognomonic": (
            "Fatigable proximal weakness + HIGH CK + negative MG antibodies + decrement on RNS = "
            "GMPPB/LGMD2T until proven; "
            "HIGH CK in 'myasthenia' = ALWAYS check alpha-DG IHC; seronegative myasthenia + elevated CK = GMPPB; "
            "DISTINGUISH from true myasthenia: myasthenia — CK NORMAL (or mildly elevated); brain MRI normal; "
            "DISTINGUISH from DOK7 CMS: DOK7 — worsens with pyridostigmine; GMPPB responds to pyridostigmine; "
            "DISTINGUISH from other dystroglycanopathies: GMPPB — NORMAL brain MRI; normal intellect; NMJ involvement"
        ),
        "treatment": (
            "Pyridostigmine (anticholinesterase): RESPONSIVE — improves fatigable weakness; "
            "start low-dose (30–60 mg TID); titrate to response; "
            "3,4-Diaminopyridine: may add benefit for NMJ component; "
            "Physical therapy: preserve strength; avoid overexertion; "
            "AVOID: neuromuscular blocking agents without anaesthetic review; "
            "AVOID: aminoglycosides (impair NMJ); "
            "Respiratory: annual FVC; NIV if FVC < 50%; "
            "Cardiac: echo annually — DCM rare but reported; "
            "Gene therapy: preclinical; CDP-mannose supplementation theoretical (GMPPB synthesizes GDP-mannose)"
        ),
        "critical_flags": [
            "GMPPB-PYRIDOSTIGMINE-RESPONSIVE: ONLY dystroglycanopathy with NMJ dysfunction responsive to pyridostigmine; do NOT withhold pyridostigmine in LGMD2T; failure to treat NMJ component leaves patient unnecessarily disabled",
            "GMPPB-CK-ELEVATED-IN-MYASTHENIA-CLUE: CK elevation in a 'myasthenic' patient = STOP — this is not myasthenia gravis; normal MG has normal or mildly elevated CK; markedly elevated CK = muscle disease + NMJ; order alpha-DG IHC + GMPPB sequencing",
            "GMPPB-NORMAL-BRAIN-MRI: unlike almost all other dystroglycanopathies, GMPPB shows NORMAL brain MRI and NORMAL cognition; do NOT apply dystroglycanopathy prognosis from FCMD/MEB/WWS to LGMD2T patients",
            "GMPPB-SERONEGATIVE-MYASTHENIA-TRAP: patients diagnosed with seronegative myasthenia gravis for years before GMPPB identified; high CK + negative antibodies + decrement = send GMPPB sequencing; review all seronegative MG patients with elevated CK",
            "GMPPB-GDP-MANNOSE-MECHANISM: GMPPB synthesizes GDP-mannose, the mannose donor for O-mannosylation of alpha-DG; without GDP-mannose, POMT1/POMT2 cannot glycosylate dystroglycan; upstream mannose metabolism block affects NMJ via synaptic alpha-DG",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _generate_cohort(gene_entry: dict) -> list:
    rng = random.Random(gene_entry["seed"])
    gene = gene_entry["gene"]
    pts = []
    for i in range(40):
        age_dx = rng.randint(0, 35)
        ck_mult = rng.uniform(5, 50)
        has_cm = rng.random() < (0.3 if gene in ("FKRP", "FKTN", "ISPD") else 0.1)
        has_brain = gene not in ("FKRP", "GMPPB")
        has_nmj = gene == "GMPPB"
        severity = rng.choice(["CMD", "CMD", "LGMD", "LGMD", "LGMD"]) if gene in ("FKRP", "FKTN", "ISPD") else (
            "CMD" if gene in ("POMT1", "POMT2", "POMGNT1", "LARGE1") else "LGMD"
        )
        pts.append({
            "patient_id": f"{gene}-{i+1:03d}",
            "age_at_diagnosis": age_dx,
            "sex": rng.choice(["M", "F"]),
            "phenotype_severity": severity,
            "ck_x_normal": round(ck_mult, 1),
            "cardiomyopathy": has_cm,
            "brain_abnormalities": has_brain,
            "nmj_component": has_nmj,
            "ambulation_preserved": rng.random() < (0.7 if severity == "LGMD" else 0.15),
            "respiratory_support": rng.random() < (0.6 if severity == "CMD" else 0.25),
        })
    return pts


def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in DYSTROGLYCANOPATHY_GENES]
    total = sum(len(c) for c in all_cohorts)
    all_pts = [p for c in all_cohorts for p in c]
    cmd_n = sum(1 for p in all_pts if p["phenotype_severity"] == "CMD")
    lgmd_n = total - cmd_n
    cardiac_n = sum(1 for p in all_pts if p["cardiomyopathy"])
    brain_n = sum(1 for p in all_pts if p["brain_abnormalities"])
    resp_n = sum(1 for p in all_pts if p["respiratory_support"])
    amb_n = sum(1 for p in all_pts if p["ambulation_preserved"])
    gene_counts = {g["gene"]: len(_generate_cohort(g)) for g in DYSTROGLYCANOPATHY_GENES}
    return {
        "atlas": "Hereditary-Dystroglycanopathy-Atlas",
        "subtitle": "Complete 8-Gene Alpha-Dystroglycan O-Glycosylation Disorder Atlas",
        "genes": [g["gene"] for g in DYSTROGLYCANOPATHY_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}–{SEED_BASE+7}",
        "cmd_patients": cmd_n,
        "lgmd_patients": lgmd_n,
        "cardiomyopathy_patients": cardiac_n,
        "brain_abnormality_patients": brain_n,
        "respiratory_support_patients": resp_n,
        "ambulation_preserved_patients": amb_n,
        "gene_patient_counts": gene_counts,
        "pathway": "O-Mannosylation of alpha-Dystroglycan — Matriglycan Synthesis",
        "key_clinical_insight": (
            "ALL dystroglycanopathies share reduced alpha-DG glycosylation; "
            "FKRP L276I is the most common AR LGMD in Northern Europe; "
            "GMPPB is the only dystroglycanopathy with NMJ component (pyridostigmine responsive); "
            "brain MRI differentiates subtypes (cobblestone = POMT1/POMT2; white matter = LARGE1; "
            "cerebellar cysts = POMGNT1; basal ganglia = ISPD; NORMAL = FKRP/GMPPB)"
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in DYSTROGLYCANOPATHY_GENES:
        cohort = _generate_cohort(gene_entry)
        cmd_pct = round(100 * sum(1 for p in cohort if p["phenotype_severity"] == "CMD") / len(cohort))
        cardiac_pct = round(100 * sum(1 for p in cohort if p["cardiomyopathy"]) / len(cohort))
        amb_pct = round(100 * sum(1 for p in cohort if p["ambulation_preserved"]) / len(cohort))
        result[gene_entry["gene"]] = {
            "gene": gene_entry["gene"],
            "alt_name": gene_entry["alt_name"],
            "locus": gene_entry["locus"],
            "protein_size": gene_entry["protein_size"],
            "inheritance": gene_entry["inheritance"],
            "n_patients": len(cohort),
            "cmd_pct": cmd_pct,
            "lgmd_pct": 100 - cmd_pct,
            "cardiomyopathy_pct": cardiac_pct,
            "ambulation_preserved_pct": amb_pct,
            "age_of_onset": gene_entry["age_of_onset"],
            "key_biomarker": gene_entry["key_biomarker"],
            "pathognomonic": gene_entry["pathognomonic"],
            "treatment": gene_entry["treatment"],
            "critical_flags": gene_entry["critical_flags"],
            "seed": gene_entry["seed"],
            "cohort_preview": cohort[:5],
        }
    return result


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Dystroglycanopathy-Atlas",
        "gene_definitions": {
            g["gene"]: {
                "protein": g["protein"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
            }
            for g in DYSTROGLYCANOPATHY_GENES
        },
        "glossary": {
            "alpha-dystroglycan (alpha-DG)": "Peripheral membrane glycoprotein; highly glycosylated extracellular domain binds laminin/agrin/neurexin; loss of glycosylation = sarcolemmal instability",
            "matriglycan": "Terminal repeating xylose-glucuronate chain added by LARGE1 to alpha-DG; required for high-affinity laminin binding",
            "O-mannosylation": "First step of alpha-DG glycosylation initiated by POMT1/POMT2 complex; prerequisite for all downstream glycan extensions",
            "cobblestone lissencephaly (type II)": "Cortical dysplasia from over-migration of neurons through gaps in glia limitans; bumpy/cobbled outer surface; DISTINCT from type I (smooth lissencephaly of LIS1/DCX)",
            "Walker-Warburg Syndrome (WWS)": "Most severe dystroglycanopathy: CMD + cobblestone lissencephaly + eye malformations; < 3 yr median survival",
            "Muscle-Eye-Brain Disease (MEB)": "POMGNT1-associated: CMD + progressive high myopia + cerebellar cysts + pachygyria; better survival than WWS",
            "Fukuyama CMD (FCMD)": "FKTN 3kb SVA founder: most common CMD in Japan; cobblestone cortex + cognitive impairment + cardiomyopathy",
            "LGMD2I": "FKRP-associated LGMD; L276I Caucasian founder; most common AR LGMD in Northern Europe; cardiomyopathy risk 20-30%",
            "MDC1D": "LARGE1-associated CMD; most severe cognitive impairment; periventricular white matter signal dominant",
            "LGMD2T": "GMPPB-associated; UNIQUE NMJ component (decrement on EMG); pyridostigmine responsive; normal brain MRI",
            "pyridostigmine responsiveness": "GMPPB is the ONLY dystroglycanopathy where NMJ dysfunction responds to AChE inhibition; all others do not benefit",
            "CDP-ribitol": "Substrate produced by ISPD/CRPPA; used by FKTN and FKRP to add ribitol-phosphate to alpha-DG; ISPD deficiency = no substrate for downstream enzymes",
            "VIA4-1 / IIH6 antibodies": "Monoclonal antibodies against glycosylated epitopes of alpha-DG; used in IHC to confirm reduced glycosylation",
            "SVA retrotransposon": "SINE/VNTR/Alu retrotransposon; 3kb insertion in FKTN 3'UTR is the FCMD founder allele; NOT detected by standard WES",
        },
        "clinical_pearls": [
            "Any CMD or LGMD patient with elevated CK: ORDER alpha-DG IHC on muscle biopsy; if reduced, sequence full dystroglycanopathy panel (not sequential single-gene testing)",
            "FKRP L276I Caucasian founder: homozygous L276I = mild LGMD2I; heterozygous L276I + pathogenic second allele = variable (LGMD to MDC1C); all need cardiac surveillance",
            "GMPPB elevated CK + 'myasthenia': any seronegative myasthenia with CK > 3× normal = rule out GMPPB before finalising myasthenia diagnosis",
            "Japanese CMD: FKTN/FCMD; screen 3kb SVA insertion FIRST (founder); standard sequencing MISSES it",
            "ISPD Leigh-like MRI: metabolic workup negative; alpha-DG deficiency identifies correct diagnosis; do NOT delay gene panel awaiting metabolic results",
            "Cardiac surveillance: MANDATORY in FKRP, FKTN, ISPD from diagnosis; annual echo; ICD threshold EF < 35%",
            "Pyridostigmine: ADD in all confirmed GMPPB/LGMD2T; AVOID in other dystroglycanopathies (no NMJ benefit; not harmful but not helpful)",
            "Prenatal: all dystroglycanopathies AR; 25% recurrence per pregnancy; CVS/amniocentesis in affected families",
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2))
