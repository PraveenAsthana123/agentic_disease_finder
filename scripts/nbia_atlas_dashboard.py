#!/usr/bin/env python3
"""NBIA-Atlas — Complete 8-Gene Neurodegeneration with Brain Iron Accumulation Atlas
PANK2 · PLA2G6 · C19orf12 · FA2H · WDR45 · COASY · ATP13A2 · DCAF17
320-patient aggregate cohort (8 × 40, seeds 926–933)

Neurodegeneration with Brain Iron Accumulation (NBIA) facts:
  - NBIA = heterogeneous group of inherited neurodegenerative diseases characterised by
    progressive iron accumulation in the basal ganglia (globus pallidus, substantia nigra)
    and neuronal degeneration.
  - Shared clinical features: extrapyramidal movement disorder (dystonia, parkinsonism,
    chorea), corticospinal tract signs (spasticity), and progressive neurological decline.
  - KEY TEACHING POINTS:
      PANK2 — Eye-of-Tiger sign on T2 MRI (central T2 hyperintensity + peripheral
        low signal in globus pallidus) is PATHOGNOMONIC for PKAN; most common NBIA (~50%).
      WDR45 — X-LINKED DOMINANT de novo; mainly females (4:1); childhood static
        phase then adult-onset parkinsonism-dementia is the hallmark biphasic course.
      PLA2G6 — Three distinct phenotypes: INAD (infantile; axonal spheroids), NBIA-PLA2G6
        (juvenile; atypical parkinsonism), PARK14 (adult; L-DOPA-responsive parkinsonism).
      FA2H — Leukodystrophy (T2 white matter changes) + spastic paraplegia (SPG35) +
        cerebellar ataxia; 16q23.1; also causes Hereditary Spastic Paraplegia SPG35.
      DCAF17 — ONLY NBIA gene with systemic features: hypogonadism (hypergonadotropic),
        alopecia, diabetes mellitus, sensorineural deafness (Woodhouse-Sakati syndrome).
      ATP13A2 — Juvenile parkinsonism + pyramidal signs + supranuclear gaze palsy;
        Kufor-Rakeb disease; lysosomal ATPase; partial L-DOPA response early.
      COASY — Pantothenate/CoA biosynthesis (same pathway as PANK2); CoPAN; rare.
      C19orf12 — MPAN (Mitochondria-associated neurodegeneration); optic atrophy;
        psychiatric onset common; 19q12; fatty acid synthesis component.
      Iron chelation (deferiprone) shows some signal in PKAN (PANK2); no other
        disease-modifying therapy approved for any NBIA gene.
      Deep brain stimulation (DBS) for palliation of dystonia, especially in PKAN.

COHORT: 8 × 40 = 320 patient slots (seeds 926–933; gene-specific seeds)
"""

import random

SEED_BASE = 926

# ── All 8 NBIA Genes ──────────────────────────────────────────────────────────────
NBIA_GENES = [
    # ── PANK2 — Pantothenate Kinase 2 (PKAN) ─────────────────────────────────────
    {
        "gene": "PANK2", "alias": "PANK2 — PKAN: Pantothenate Kinase-Associated Neurodegeneration (OMIM #234200)",
        "aa": "314 aa (isoform 1: 570aa; principal mitochondrial isoform)", "kDa": "56 kDa (isoform 2: 36 kDa)",
        "gene_class": "Pantothenate kinase: CoA biosynthesis step 1 — pantothenate + ATP → 4'-phosphopantothenate",
        "nbia_subgroup": "CoA biosynthesis pathway (PANK2 · COASY)",
        "locus": "20q13.33", "omim_gene": 606157,
        "phenotype": "PKAN: progressive dystonia (generalised), dysarthria, pigmentary retinopathy, parkinsonism; Classic (<20y, rapid) vs Atypical (>20y, slower); Eye-of-Tiger PATHOGNOMONIC on T2 MRI",
        "disease": (
            "PANK2 encodes pantothenate kinase 2 (PANK2, 314aa mitochondrial isoform; 570aa full-length contains "
            "N-terminal mitochondrial targeting sequence), the rate-limiting enzyme in mitochondrial CoA biosynthesis. "
            "PANK2 phosphorylates pantothenate (vitamin B5) → 4'-phosphopantothenate, the first committed step. "
            "PANK2 deficiency → CoA deficiency in mitochondria → impaired fatty acid β-oxidation + TCA cycle. "
            "CoA deficiency also impairs phospholipid synthesis in myelin, explaining white matter involvement. "
            "Cysteine (a PANK2 substrate in the second step of CoA synthesis) accumulates when PANK2 fails → "
            "cysteine auto-oxidation → iron chelation → iron-cysteine complex deposition in globus pallidus (GP). "
            "GP iron deposition is the pathological hallmark. PKAN is the most common NBIA, accounting for ~50% of cases. "
            "Two clinical forms: Classic PKAN (60%): onset <10y (typically 3-6y); severe progressive dystonia becoming "
            "generalised; dysarthria; pigmentary retinopathy in 70%; RAPID DECLINE with loss of ambulation by teenage years. "
            "Atypical PKAN (40%): onset >10y (often 13-30y, range to 60y); speech/psychiatric features predominate; "
            "SLOWER progression; retinopathy less common. "
            "Eye-of-Tiger sign: T2 MRI shows bilateral symmetric signal in GP: "
            "central hyperintensity (T2-bright = gliosis/cystic necrosis) within a rim of hypointensity (T2-dark = iron). "
            "This sign is PATHOGNOMONIC for PKAN — not seen in other NBIA genes. "
            "~5% of PKAN lack the eye-of-tiger sign (variant PKAN with predominantly iron-only GP changes). "
            "Incidence: ~1-3/1,000,000. AR inheritance. Both sexes equally affected."
        ),
        "inheritance": "AR. 20q13.33. Both sexes equally. Consanguinity in ~30%. Classic PKAN ~60%, Atypical PKAN ~40%.",
        "hallmark": (
            "PANK2 HALLMARKS: "
            "(1) EYE-OF-TIGER SIGN T2 MRI PATHOGNOMONIC: bilateral symmetric GP lesion; "
            "central T2 hyperintensity (gliosis/necrosis) surrounded by T2 hypointensity (iron); "
            "NO OTHER NBIA GENE has this sign; specific for PKAN; "
            "(2) DYSTONIA GENERALISED PROGRESSIVE: onset 3-6y classic; severe; generalised by adolescence; "
            "postural and action dystonia; oromandibular dystonia with dysarthria; "
            "(3) PIGMENTARY RETINOPATHY 70% OF PKAN: fundoscopic bone-spicule deposits; visual field constriction; "
            "ERG abnormal (rod-cone dystrophy pattern); ophthalmology mandatory; "
            "(4) PANK2 SAME PATHWAY AS COASY (CoPAN): both in CoA synthesis; "
            "CoA deficiency is the common mechanism; pantethine experimental (bypasses PANK2); "
            "(5) DEFERIPRONE (iron chelation) — PARTIAL BENEFIT IN PKAN ONLY: "
            "clinical trials show slower progression in PKAN (not other NBIA); "
            "NOT yet standard of care (conflicting trial results); "
            "(6) DBS FOR DYSTONIA PALLIATION: globus pallidus internus (GPi) DBS reduces dystonia severity; "
            "does not modify disease; symptom-modifying only; "
            "(7) CLASSIC vs ATYPICAL PKAN: classic = rapid, childhood, generalised dystonia; "
            "atypical = slower, psychiatric/speech, adult onset; "
            "(8) PANTETHINE EXPERIMENTAL: bypasses PANK2 deficiency by entering CoA pathway downstream; "
            "early-phase trials; not yet approved"
        ),
        "key_ddx": (
            "PANK2 DDx: "
            "(1) PLA2G6 INAD: onset younger (infancy); eye-of-tiger ABSENT; cerebellar atrophy; axonal spheroids; "
            "(2) WDR45 BPAN: X-linked dominant; biphasic (static childhood → adult PD); eye-of-tiger ABSENT; "
            "(3) FA2H FAHN: leukodystrophy T2 WM changes (not just GP); spastic paraplegia; eye-of-tiger ABSENT; "
            "(4) C19orf12 MPAN: optic atrophy; psychiatric onset; eye-of-tiger ABSENT; "
            "(5) Wilson disease: KF rings; hepatic disease; copper not iron; "
            "(6) Huntington: CAG repeat; chorea predominant; adults; "
            "(7) Idiopathic torsion dystonia (DYT1/TOR1A): NO brain iron; normal MRI signal"
        ),
        "diet_treatment": "No diet modification. Deferiprone (iron chelation) 25-30 mg/kg/day in 3 doses — shows some disease-slowing in PKAN specifically (TIRCON trial, FAIR-PARK I data); not yet standard of care. Pantethine (CoA precursor bypass) experimental. DBS (GPi) for dystonia palliation. Baclofen/trihexyphenidyl for spasticity/dystonia. Botulinum toxin for focal dystonia. Physical therapy, speech therapy, gastrostomy for dysphagia.",
        "gene_therapy_status": "No approved gene therapy. AAV-PANK2 preclinical in mouse models. Gene therapy challenge: targeting mitochondrial matrix delivery. PANK2 is a strong candidate due to well-defined CoA pathway defect; small gene size facilitates AAV packaging.",
        "critical_ci": (
            "CRITICAL: (1) Assuming eye-of-tiger = universal NBIA finding — it is PKAN-SPECIFIC; "
            "other NBIA genes do NOT show eye-of-tiger; "
            "(2) Misattributing eye-of-tiger to Wilson disease — Wilson has KF rings, copper, not iron; "
            "(3) Using deferiprone in non-PANK2 NBIA without evidence; "
            "(4) Missing retinopathy — always refer to ophthalmology in PKAN; "
            "(5) DBS without realistic expectations — DBS improves dystonia but does NOT stop progression; "
            "(6) Missing CoA pathway connection to COASY — useful for family counselling"
        ),
        "nbs_marker": "No NBS marker (PKAN is not screened on newborn metabolic panels). Diagnosis clinical + MRI eye-of-tiger + PANK2 sequencing. Acylcarnitine profile: some reports of low free carnitine (CoA pathway disruption affects carnitine metabolism) but not diagnostic. CSF: nonspecific. MRI is the cornerstone investigation.",
        "key_biomarker": "T2 MRI: Eye-of-Tiger sign in globus pallidus PATHOGNOMONIC. SWI/GRE sequences: iron deposition (dark signal). Retinal examination: pigmentary retinopathy (ERG). PANK2 sequencing (WES/gene panel). Pantothenate kinase enzyme activity in fibroblasts (reduced). CoA levels in cells (reduced). Acylcarnitine profile (mild CoA-related changes).",
        "severity_spectrum": "Classic PKAN (~60%): onset 3-6y; rapid generalised dystonia; lose ambulation by ~15y; typical survival 20-30y with complications. Atypical PKAN (~40%): onset 13-30y; slower; psychiatric/speech features first; ambulatory longer; survival longer. Genotype correlates with form: null mutations → classic; missense → atypical in many.",
        "founder_variant": "No single global founder. Regional variants: p.Thr418Met (pantothenate kinase domain; atypical PKAN); p.Gly411Arg (classic PKAN). Consanguinity-associated variants in Middle Eastern, South Asian populations. No single founder like MDDS genes.",
        "key_variants": ["p.Thr418Met (atypical PKAN)", "p.Gly411Arg (classic PKAN)", "c.1583C>T p.Ser528Phe", "c.215G>A p.Gly72Glu", "p.Ala278Val"],
        "seed": SEED_BASE + 0,
    },
    # ── PLA2G6 — Phospholipase A2 Group VI (PLAN/INAD/PARK14) ────────────────────
    {
        "gene": "PLA2G6", "alias": "PLA2G6 — PLAN/INAD/PARK14: iPLA2β deficiency (OMIM #256600 INAD, #610217 PLAN, #612953 PARK14)",
        "aa": "806 aa", "kDa": "88 kDa",
        "gene_class": "Phospholipase A2, calcium-independent group VI: sn-2 fatty acid cleavage from phospholipids — membrane remodelling, mitochondrial membrane integrity",
        "nbia_subgroup": "Phospholipid remodelling (PLA2G6 · FA2H)",
        "locus": "22q13.1", "omim_gene": 603604,
        "phenotype": "Three phenotypes: INAD (infantile; axonal spheroids; cerebellar atrophy); PLAN/NBIA-PLA2G6 (juvenile; atypical parkinsonism); PARK14 (adult; L-DOPA-responsive parkinsonism with psychiatric features)",
        "disease": (
            "PLA2G6 encodes iPLA2β (calcium-independent phospholipase A2β, 806aa, 88kDa), which cleaves sn-2 "
            "fatty acids from phospholipids. iPLA2β is critical for: membrane phospholipid remodelling "
            "(maintaining proper phospholipid composition of mitochondrial and ER membranes), "
            "mitochondrial membrane integrity (loss → mitochondrial membrane breakdown → "
            "axonal degeneration), and arachidonic acid release from membranes "
            "(signalling functions). "
            "PLA2G6 loss → defective membrane remodelling → AXONAL SPHEROIDS (swellings in axons, "
            "filled with mitochondrial debris and membranous whorls) — the pathological hallmark. "
            "Cerebellar cortex axons are disproportionately affected. "
            "Three distinct phenotypic spectra: "
            "1. INAD (Infantile Neuroaxonal Dystrophy, OMIM #256600): onset 6-18 months; hypotonia → "
            "spasticity; cerebellar atrophy on MRI (vermis > hemispheres); T2 hyperintensity cerebellar cortex; "
            "strabismus; nystagmus; rapid progression → tetraplegia + absent speech by 5y; death typically 5-10y. "
            "AXONAL SPHEROIDS on sural nerve biopsy DIAGNOSTIC (now largely replaced by PLA2G6 sequencing). "
            "2. Atypical NBIA / PLAN (PLA2G6-associated neurodegeneration, OMIM #610217): "
            "onset 1-8y; dystonia + cerebellar ataxia + cognitive regression; slower than INAD; "
            "brain iron + cerebellar atrophy on MRI; survives longer. "
            "3. PARK14 (adult-onset parkinsonism, OMIM #612953): onset 20-40y; L-DOPA-responsive parkinsonism "
            "with psychiatric features (anxiety, psychosis, cognitive decline); resembles idiopathic PD; "
            "iron may be detected on SWI MRI; slow progression. "
            "Incidence: INAD ~1/1,000,000 (most common). AR inheritance all three forms."
        ),
        "inheritance": "AR. 22q13.1. Both sexes equally. Three allelic phenotypes: INAD, PLAN/NBIA, PARK14.",
        "hallmark": (
            "PLA2G6 HALLMARKS: "
            "(1) THREE-PHENOTYPE SPECTRUM FROM ONE GENE: INAD (infantile, severe, rapid) → "
            "PLAN (juvenile, moderate) → PARK14 (adult, mild PD-like); "
            "genotype-phenotype correlation: severe truncating → INAD; missense → PARK14; "
            "(2) CEREBELLAR ATROPHY ON MRI: early prominent cerebellar vermis atrophy; "
            "T2 cerebellar cortex hyperintensity in INAD; eye-of-tiger ABSENT; "
            "GP iron on SWI in PLAN/PARK14 but NOT eye-of-tiger; "
            "(3) AXONAL SPHEROIDS ON BIOPSY DIAGNOSTIC (INAD): "
            "PAS-positive axonal swellings in peripheral and central nervous system; "
            "sural nerve biopsy shows spheroids; now largely replaced by molecular diagnosis; "
            "(4) PARK14 ADULT-ONSET L-DOPA-RESPONSIVE PARKINSONISM: "
            "resembles idiopathic PD; responds to L-DOPA early; develops dyskinesias; "
            "psychiatric symptoms (anxiety, psychosis) common — differentiates from typical PD; "
            "(5) INAD: NO BRAIN IRON IN EARLY STAGES — iron appears later; "
            "cerebellar atrophy is the early imaging hallmark, not iron; "
            "(6) ipla2β MEMBRANE REMODELLING: loss of phospholipid remodelling → mitochondrial "
            "membrane breaks down → axons swell with mitochondrial debris → spheroids"
        ),
        "key_ddx": (
            "PLA2G6 DDx: "
            "(1) PANK2 PKAN: eye-of-tiger PRESENT; dystonia predominant; no cerebellar atrophy early; "
            "(2) C19orf12 MPAN: optic atrophy (absent in PLA2G6); psychiatric onset similar in PARK14 form; "
            "(3) Ataxia telangiectasia: cerebellar ataxia + telangiectasias + immunodeficiency; "
            "(4) Canavan / Pelizaeus-Merzbacher: leukodystrophy (T2 WM changes; not cerebellar grey matter); "
            "(5) Friedreich ataxia: GAA repeat; cardiomyopathy; NO brain iron; "
            "(6) Idiopathic PD (for PARK14): age of onset; psychiatric features early; PLA2G6 sequencing if atypical"
        ),
        "diet_treatment": "No specific diet or approved disease-modifying therapy. Supportive: L-DOPA for PARK14 form (partial response, wanes). Antispasticity (baclofen, tizanidine) for INAD. Seizure management. Gastrostomy for INAD dysphagia. Physiotherapy. Assistive devices. Palliative care for INAD.",
        "gene_therapy_status": "No approved gene therapy. AAV9-PLA2G6 preclinical (mouse INAD models; 2015-2020s). Gene therapy for INAD is an active research focus given severe early phenotype and lack of treatment. Challenge: large gene (806aa); cerebellar targeting with CNS-tropic AAVs.",
        "critical_ci": (
            "CRITICAL: (1) Assuming all NBIA has eye-of-tiger — PLA2G6 does NOT have eye-of-tiger; "
            "cerebellar atrophy is PLA2G6's MRI hallmark; "
            "(2) Missing INAD diagnosis — any infant with hypotonia + cerebellar atrophy needs PLA2G6; "
            "(3) Biopsy-first approach — molecular diagnosis has superseded nerve biopsy for INAD; "
            "(4) Missing PARK14 in young-onset PD differential — PLA2G6 panel in atypical PD <40y; "
            "(5) Expecting L-DOPA to sustain benefit in PARK14 — response wanes; psychiatric meds needed"
        ),
        "nbs_marker": "No NBS marker. Diagnosis clinical + MRI + PLA2G6 sequencing. Nerve biopsy (sural) shows axonal spheroids in INAD (now replaced by molecular). Phospholipase activity in fibroblasts (reduced/absent). CSF: elevated neurofilament light chain (neurodegeneration marker; non-specific).",
        "key_biomarker": "MRI: cerebellar vermis atrophy + T2 cerebellar cortex hyperintensity (INAD); GP/SN iron on SWI (PLAN/PARK14). Sural nerve biopsy: axonal spheroids (PAS+, EM mitochondrial debris). PLA2G6 sequencing. Phospholipase A2 activity (fibroblasts). CSF neurofilament light chain (elevated).",
        "severity_spectrum": "INAD (most severe): onset 6-18m; death 5-10y. PLAN/NBIA (moderate): onset 1-8y; survival to 30s-40s. PARK14 (mildest): onset 20-40y; normal lifespan possible with symptom management.",
        "founder_variant": "p.Arg741Gln (PARK14; Japan, Korean cases); p.Asp331Asn (INAD; European); p.Arg543Trp (PLAN). No single global founder. Regional clustering in Middle East and East Asia.",
        "key_variants": ["p.Arg741Gln (PARK14, East Asian)", "p.Asp331Asn (INAD)", "p.Arg543Trp (PLAN)", "c.2239G>A p.Gly747Arg", "IVS7+2T>A splice"],
        "seed": SEED_BASE + 1,
    },
    # ── C19orf12 — MPAN (Mitochondria-Associated Neurodegeneration) ──────────────
    {
        "gene": "C19orf12", "alias": "C19orf12 — MPAN: Mitochondria-associated Neurodegeneration (OMIM #614297)",
        "aa": "142 aa", "kDa": "17 kDa",
        "gene_class": "Mitochondria-associated membrane (MAM) protein: fatty acid synthesis / uncertain function; myo-inositol metabolism",
        "nbia_subgroup": "Mitochondria-associated membrane / fatty acid synthesis (C19orf12)",
        "locus": "19q12", "omim_gene": 614297,
        "phenotype": "MPAN: progressive spastic paraplegia + optic atrophy + psychiatric features (depression, psychosis) + parkinsonism in adulthood; striatal iron; optic atrophy distinguishes from other NBIA",
        "disease": (
            "C19orf12 (chromosome 19 open reading frame 12) encodes a 142aa (17kDa) protein of uncertain function. "
            "It localises to mitochondria-associated membranes (MAMs — ER-mitochondria contact sites) and mitochondria. "
            "Evidence suggests roles in: fatty acid synthesis (in mitochondria), "
            "myo-inositol metabolism, and ER-mitochondria tethering. "
            "C19orf12 loss → progressive neurodegeneration with iron accumulation in the striatum "
            "(globus pallidus, substantia nigra), optic atrophy, and psychiatric symptoms. "
            "MPAN is one of the more common NBIA forms (~20% after PKAN). "
            "Clinical presentation of MPAN: onset typically 10-20y; "
            "psychiatric features early and prominent (depression, emotional lability, psychosis, OCD-like) — "
            "these psychiatric features lead to frequent misdiagnosis as primary psychiatric disorder for years; "
            "progressive spastic paraplegia; optic atrophy (optic nerve pallor ± visual acuity decline) — "
            "PRESENT IN ~50-70% OF MPAN, DISTINGUISHING FEATURE from PKAN and WDR45; "
            "parkinsonism develops in adulthood; motor neuron disease-like features (both UMN and LMN) in some; "
            "MRI: iron in GP (SWI dark signal) + T2 iron in SN; no eye-of-tiger. "
            "p.Thr11Met (c.32C>T) is a founder variant in European (especially Polish/Czech) populations. "
            "Incidence: ~1-5/1,000,000. AR inheritance."
        ),
        "inheritance": "AR. 19q12. Both sexes equally. p.Thr11Met founder in European populations (Polish, Czech, Slovak).",
        "hallmark": (
            "C19orf12 HALLMARKS: "
            "(1) PSYCHIATRIC ONSET PATHOGNOMONIC CONTEXT: "
            "MPAN typically begins with psychiatric symptoms (depression, psychosis, mood disorder) "
            "BEFORE movement disorder develops; patients misdiagnosed as primary psychiatric disorder; "
            "any early-onset psychiatric + movement disorder + optic atrophy combination = MPAN until proven otherwise; "
            "(2) OPTIC ATROPHY IN ~50-70% OF MPAN: "
            "fundoscopy shows optic disc pallor; VEP delayed/absent; visual acuity declines; "
            "PKAN and WDR45 DO NOT have optic atrophy as a feature — optic atrophy DDx key; "
            "(3) STRIATAL IRON SWI: GP + SN iron accumulation on SWI MRI; "
            "NO eye-of-tiger (GP T2 central bright spot — that's PKAN only); "
            "(4) SPASTIC PARAPLEGIA: corticospinal tract involvement; UMN signs (hyperreflexia, Babinski); "
            "may also have LMN signs → mixed UMN/LMN pattern resembling ALS; "
            "(5) EUROPEAN FOUNDER VARIANT p.Thr11Met: "
            "c.32C>T; concentrated in Slavic populations; "
            "allows targeted testing in high-risk groups; "
            "(6) ADULTHOOD PARKINSONISM: parkinsonism develops after the spastic/psychiatric phase; "
            "partial L-DOPA response; "
            "(7) MRI EVOLUTION: early imaging may be near-normal; "
            "iron appears over years; repeat MRI if index of suspicion high"
        ),
        "key_ddx": (
            "C19orf12 DDx: "
            "(1) PANK2 PKAN: eye-of-tiger present; NO optic atrophy; NO psychiatric onset; "
            "(2) WDR45 BPAN: X-linked dominant; biphasic; NO optic atrophy; "
            "(3) PLA2G6 PARK14: adult PD-like; NO optic atrophy; cerebellar involvement; "
            "(4) Primary psychiatric disorder: NO optic atrophy; NO MRI iron; "
            "(5) Hereditary spastic paraplegia (SPG): NO brain iron; NO optic atrophy in most SPG types; "
            "(6) Multiple sclerosis: demyelinating plaques; relapsing-remitting; NOT iron accumulation"
        ),
        "diet_treatment": "No disease-modifying therapy. Psychiatric medications for depression/psychosis. Baclofen/tizanidine for spasticity. L-DOPA for parkinsonism (partial response). Physiotherapy. Vision aids for optic atrophy. Surveillance: annual ophthalmology (VEP + fundoscopy), neurology, and psychiatry.",
        "gene_therapy_status": "No approved gene therapy. C19orf12 is a small gene (142aa) — highly suitable for AAV packaging. Preclinical studies limited due to poorly characterised protein function. Active research into MAM function.",
        "critical_ci": (
            "CRITICAL: (1) Treating MPAN as primary psychiatric disorder — delay in recognising the neurological component; "
            "(2) Missing optic atrophy — always examine optic fundi + VEP in NBIA workup; "
            "(3) Assuming eye-of-tiger in C19orf12 — it is ABSENT (PKAN only); "
            "(4) Not testing in Slavic patients with early-onset psychiatric + motor features; "
            "(5) Expecting full L-DOPA response — partial only in MPAN"
        ),
        "nbs_marker": "No NBS marker. Diagnosis clinical + MRI (SWI iron) + C19orf12 sequencing. Ophthalmology (optic atrophy, VEP). Psychiatric evaluation. MRI: iron accumulation on SWI. Founder variant p.Thr11Met can be tested specifically in high-risk populations.",
        "key_biomarker": "MRI SWI: iron in GP/SN (no eye-of-tiger). Fundoscopy + VEP: optic atrophy (50-70%). C19orf12 sequencing. Neuropsychiatric assessment. Myo-inositol levels (CSF/plasma; research only).",
        "severity_spectrum": "Moderate-severe: onset 10-20y; progressive over decades; lose ambulation in 30-40y; optic atrophy progresses; psychiatric features throughout; parkinsonism superimposed; survival to 40-60y (variable).",
        "founder_variant": "p.Thr11Met (c.32C>T) — European founder, especially Polish, Czech, Slovak populations. Also p.Gly69Arg (less common founder, Czech/Slovak). Targeted testing justified in Slavic patients.",
        "key_variants": ["p.Thr11Met (European founder)", "p.Gly69Arg", "c.204_214del (frameshift)", "p.Leu35Pro", "p.Glu108Lys"],
        "seed": SEED_BASE + 2,
    },
    # ── FA2H — Fatty Acid 2-Hydroxylase (FAHN / SPG35) ───────────────────────────
    {
        "gene": "FA2H", "alias": "FA2H — FAHN (Fatty Acid Hydroxylase-associated Neurodegeneration) / SPG35 (OMIM #612319)",
        "aa": "516 aa", "kDa": "58 kDa",
        "gene_class": "Fatty acid 2-hydroxylase: ER enzyme converting fatty acids to 2-hydroxy fatty acids — critical for myelin galactosphingolipid (HFA-GalCer) and GalCer synthesis",
        "nbia_subgroup": "Phospholipid remodelling / myelin (FA2H · PLA2G6)",
        "locus": "16q23.1", "omim_gene": 611026,
        "phenotype": "FAHN / SPG35: childhood-onset spastic paraplegia (SPG35) + cerebellar ataxia + dysarthria + dystonia + LEUKODYSTROPHY (T2 periventricular WM signal) + iron accumulation; progresses to dystonia + parkinsonism",
        "disease": (
            "FA2H encodes fatty acid 2-hydroxylase (516aa, 58kDa), an ER-resident enzyme that converts "
            "long-chain saturated and monounsaturated fatty acids to their 2-hydroxy forms. "
            "2-hydroxy fatty acids are essential for synthesising 2-hydroxy galactosylceramide (HFA-GalCer), "
            "a myelin-specific sphingolipid critical for compact myelin formation and myelin sheath stability. "
            "FA2H deficiency → lack of 2-hydroxy GalCer → myelin instability → demyelination/dysmyelination "
            "(LEUKODYSTROPHY) + progressive neurodegeneration. "
            "Axonal iron accumulation occurs secondary to demyelination. "
            "FA2H deficiency causes two overlapping phenotypes: "
            "SPG35 (Hereditary Spastic Paraplegia 35): childhood-onset spastic paraplegia as predominant feature; "
            "considered within the SPG/HSP classification; "
            "FAHN (Fatty acid hydroxylase-associated neurodegeneration): childhood onset (typically 4-12y); "
            "spastic paraplegia + CEREBELLAR ATAXIA (gait + limb ataxia) + DYSARTHRIA (cerebellar) + "
            "dystonia + parkinsonism; MRI = LEUKODYSTROPHY (T2 periventricular and subcortical white matter "
            "hyperintensities) + cerebellar atrophy + IRON in GP (SWI); "
            "vision loss (optic atrophy in ~30%); "
            "progression to wheelchair by 20-40y; cognitive decline later. "
            "The combination of LEUKODYSTROPHY + NBIA is characteristic of FA2H — "
            "leukodystrophy is not a feature of PANK2, WDR45, or C19orf12. "
            "Incidence: very rare, ~1/2,000,000+. AR inheritance."
        ),
        "inheritance": "AR. 16q23.1. Both sexes equally. Also classified as SPG35 (Hereditary Spastic Paraplegia 35).",
        "hallmark": (
            "FA2H HALLMARKS: "
            "(1) LEUKODYSTROPHY + NBIA COMBINATION = FA2H FINGERPRINT: "
            "T2 periventricular and subcortical WM hyperintensities (demyelination) "
            "PLUS T2/SWI iron in GP/SN; "
            "NO OTHER COMMON NBIA GENE has leukodystrophy as a defining feature; "
            "leukodystrophy on MRI in a child with dystonia/spasticity → FA2H first; "
            "(2) SPG35 OVERLAP: same gene (FA2H); "
            "SPG35 emphasises the spastic paraplegia; "
            "FAHN is the broader NBIA phenotype; "
            "these are the same condition on a spectrum; "
            "(3) CEREBELLAR ATROPHY + ATAXIA: prominent cerebellar component; "
            "gait ataxia + limb ataxia + dysarthria; "
            "cerebellar atrophy on MRI; distinguishes from pure spastic forms; "
            "(4) 2-HYDROXY GALACTOSYLCERAMIDE DEFICIENCY: "
            "myelin-specific sphingolipid; FA2H is the unique enzyme; "
            "skin fibroblasts show absent 2-hydroxy fatty acid synthesis (diagnostic); "
            "(5) OPTIC ATROPHY IN ~30% FA2H: less prominent than C19orf12 (50-70%); "
            "ophthalmology mandatory; "
            "(6) PROGRESSIVE COURSE: childhood onset; "
            "typically wheelchair by 20-40y; cognitive decline later in disease"
        ),
        "key_ddx": (
            "FA2H DDx: "
            "(1) PANK2 PKAN: eye-of-tiger; NO leukodystrophy; NO cerebellar atrophy early; "
            "(2) PLA2G6 INAD: cerebellar atrophy (similar) but NO leukodystrophy; axonal spheroids; "
            "(3) Metachromatic leukodystrophy (ARSA/MLD): leukodystrophy but NO iron accumulation; "
            "(4) Krabbe disease (GALC): leukodystrophy; corticospinal + peripheral involvement; NO iron; "
            "(5) Hereditary spastic paraplegia (other SPGs): NO iron accumulation; NO leukodystrophy in most; "
            "(6) Multiple sclerosis: demyelinating plaques; adults; relapsing; NO iron accumulation"
        ),
        "diet_treatment": "No disease-modifying therapy. Spasticity management: baclofen (oral/intrathecal), tizanidine. Dystonia: trihexyphenidyl, botulinum toxin. Physiotherapy. Orthoses. Wheelchair. Speech therapy. Vitamin E supplementation (antioxidant; anecdotal reports; not proven). Gene therapy is a research priority.",
        "gene_therapy_status": "No approved gene therapy. FA2H gene delivery preclinical. AAV9-mediated CNS delivery studied. Substrate replacement (exogenous 2-hydroxy fatty acid supplementation) under investigation. Enzyme replacement not feasible (ER enzyme, not secreted).",
        "critical_ci": (
            "CRITICAL: (1) Missing leukodystrophy component — always get MRI with FLAIR/T2 WM sequences in NBIA workup; "
            "(2) Classifying FA2H as 'pure HSP/SPG35' without recognising the NBIA component; "
            "(3) Missing SPG35 = FA2H connection — SPG35 is the same gene; "
            "(4) Confusing with demyelinating disease (MS-like leukodystrophy) — FA2H has iron + childhood onset + progressive"
        ),
        "nbs_marker": "No NBS marker. Diagnosis: MRI (leukodystrophy + iron on SWI) + FA2H sequencing. 2-hydroxy fatty acid analysis in plasma/skin fibroblasts (reduced/absent). FA2H enzyme activity in fibroblasts. Urine organic acids (nonspecific).",
        "key_biomarker": "MRI: leukodystrophy (T2 WM hyperintensity) + GP iron (SWI) + cerebellar atrophy. Plasma/fibroblast 2-hydroxy fatty acids (reduced). FA2H enzyme activity (fibroblasts). FA2H sequencing. Ophthalmology (optic atrophy ~30%).",
        "severity_spectrum": "Childhood onset (4-12y); progressive spastic-ataxic-dystonic syndrome; lose ambulation 20-40y; cognitive decline later; no direct effect on lifespan early but respiratory/swallowing complications later.",
        "founder_variant": "No single founder. Rare variants globally. p.Ala258Glu and p.Gly264Val described in European/Middle Eastern families. Regional clustering in Moroccan and Turkish families.",
        "key_variants": ["p.Ala258Glu", "p.Gly264Val", "c.270+1G>A", "p.Ser141Asn", "Exon 5 deletion"],
        "seed": SEED_BASE + 3,
    },
    # ── WDR45 — WD Repeat Domain 45 (BPAN) ──────────────────────────────────────
    {
        "gene": "WDR45", "alias": "WDR45 — BPAN: Beta-propeller Protein-Associated Neurodegeneration (OMIM #300894)",
        "aa": "360 aa", "kDa": "38 kDa",
        "gene_class": "WD40-repeat autophagy scaffolding protein: WIPI4 homologue — phosphatidylinositol-3-phosphate (PI3P)-binding; ATG12 interaction; autophagosome formation step",
        "nbia_subgroup": "Autophagy pathway (WDR45)",
        "locus": "Xp11.23", "omim_gene": 300526,
        "phenotype": "BPAN: X-LINKED DOMINANT (de novo); BIPHASIC — static childhood period (ID + seizures + absent speech) → ADULT-ONSET progressive parkinsonism-dementia; iron in SN/GP; females 4:1 males",
        "disease": (
            "WDR45 encodes WIPI4 (WD repeat domain containing phosphoinositide-interacting protein 4), "
            "a 360aa (38kDa) WD40-repeat β-propeller protein that functions as an autophagy scaffolding factor. "
            "WIPI4 binds phosphatidylinositol-3-phosphate (PI3P) generated by the PI3K complex at the "
            "phagophore, recruits autophagy factors (ATG2, ATG18 family), and is essential for "
            "autophagosome expansion and maturation. "
            "WDR45 loss → defective autophagy → abnormal lysosomal iron handling → iron accumulation in "
            "substantia nigra and globus pallidus. "
            "BPAN is UNIQUE among NBIA for its INHERITANCE PATTERN: "
            "X-LINKED DOMINANT — caused by de novo (spontaneous) mutations in WDR45 on Xp11.23. "
            "Because affected males (hemizygous) have lethal phenotype in utero (very rarely survive), "
            "virtually all BPAN patients are FEMALES with heterozygous de novo mutations. "
            "Males with WDR45 mutations are extremely rarely described (somatic mosaicism). "
            "Female:Male ratio ~4:1 (surviving males are mosaic or have hypomorphic variants). "
            "BIPHASIC CLINICAL COURSE — PATHOGNOMONIC: "
            "Phase 1 (childhood, years 1-20y): relatively STATIC intellectual disability (moderate-severe) + "
            "epilepsy (often drug-resistant, multiple seizure types) + absent or minimal speech + "
            "autistic features; MRI may appear relatively normal or show early GP/SN iron. "
            "Phase 2 (adult onset, 20-40y): ABRUPT progression to parkinsonism-dementia; "
            "bradykinesia, rigidity, tremor; rapid cognitive decline to dementia; "
            "MRI: striking iron accumulation in SN (halo pattern) + GP; T1 hyperintensity in SN (iron). "
            "The biphasic course (stable childhood → dramatic adult deterioration) is highly characteristic. "
            "Incidence: ~1/1,000,000. Sporadic (de novo); recurrence risk low (~1% for germline mosaicism)."
        ),
        "inheritance": "X-LINKED DOMINANT. De novo mutations. Xp11.23. Female:Male ~4:1 (hemizygous males usually lethal in utero). Recurrence risk low (de novo); germline mosaicism ~1%. Heterozygous females affected.",
        "hallmark": (
            "WDR45 HALLMARKS: "
            "(1) X-LINKED DOMINANT DE NOVO: "
            "only NBIA gene with X-linked dominant inheritance; "
            "test WDR45 in ANY female with unexplained ID + seizures + later movement disorder; "
            "de novo → no family history; "
            "(2) BIPHASIC COURSE PATHOGNOMONIC: "
            "stable childhood (ID + epilepsy + absent speech) → "
            "ABRUPT adult deterioration (parkinsonism-dementia) at 20-40y; "
            "this two-phase course is highly characteristic and should trigger WDR45 sequencing; "
            "(3) IRON IN SN T1 HYPERINTENSITY: "
            "T1 MRI shows SN hyperintensity (iron shortens T1); SWI shows SN + GP dark signal; "
            "SN involvement prominent early in BPAN vs GP in PKAN; "
            "(4) AUTOPHAGY DEFECT: "
            "WDR45/WIPI4 is an autophagy scaffold; "
            "connection to mTOR pathway means rapamycin/mTOR inhibitors are under investigation; "
            "(5) RETT SYNDROME-LIKE CHILDHOOD PHASE: "
            "BPAN females may resemble Rett (MECP2) in early phase; "
            "key differentiator: WDR45 has adult-phase parkinsonism; Rett does not; "
            "(6) FEMALE PREDOMINANCE DIAGNOSTIC CLUE: "
            "BPAN seen almost exclusively in females; "
            "sporadic (de novo); "
            "maternal germline mosaicism accounts for rare familial recurrence; "
            "(7) NO EYE-OF-TIGER: BPAN MRI ≠ PKAN MRI; "
            "SN predominant signal changes; GP changes later; never eye-of-tiger pattern"
        ),
        "key_ddx": (
            "WDR45 DDx: "
            "(1) Rett syndrome (MECP2): similar childhood phase; NO adult parkinsonism; "
            "(2) PANK2 PKAN: eye-of-tiger; AR; no biphasic course; "
            "(3) Angelman syndrome: similar ID + seizures; maternal 15q11 deletion; NO movement disorder phase; "
            "(4) CDKL5: X-linked; seizures + ID + no adult movement disorder; "
            "(5) Idiopathic PD: adult onset only; NO childhood static phase; NO X-linked dominant"
        ),
        "diet_treatment": "No disease-modifying therapy. mTOR inhibitors (rapamycin/sirolimus) — phase 2 trials ongoing (autophagy restoration hypothesis). Epilepsy: valproate (generally safe in BPAN), levetiracetam, clobazam. Parkinsonism: L-DOPA (partial benefit). Cognitive: supportive care, AAC devices for non-verbal patients. Feeding support.",
        "gene_therapy_status": "No approved gene therapy. WDR45 X-linked location and dominant mechanism make gene replacement complex (would need to silence mutant allele + add wild type). CRISPR base editing for de novo dominant-negative explored in research. Autophagy pharmacological restoration (rapamycin) is the main therapeutic hypothesis.",
        "critical_ci": (
            "CRITICAL: (1) Missing BPAN in female ID+epilepsy — WDR45 sequencing in all females with unexplained ID+seizures; "
            "(2) Not anticipating adult phase — families and clinicians must be counselled about inevitable adult deterioration; "
            "(3) Assuming AR inheritance — BPAN is X-LINKED DOMINANT de novo; no need for consanguinity; "
            "(4) Expecting full L-DOPA response — partial only in BPAN adult phase; "
            "(5) Confusing with Rett syndrome and stopping at MECP2 testing"
        ),
        "nbs_marker": "No NBS marker. Diagnosis: WDR45 sequencing (X-linked dominant; test females with ID+seizures+later movement disorder). MRI: SN T1 hyperintensity + SWI iron. EEG: epilepsy workup. The de novo nature means family sequencing often unrevealing; proband-only testing sufficient.",
        "key_biomarker": "WDR45 sequencing (de novo heterozygous mutation in females). MRI: SN/GP iron on T1 + SWI; biphasic signal changes. EEG (seizure characterisation). Autophagy biomarkers (p62, LC3-II in CSF/research only). mTOR pathway activity (research).",
        "severity_spectrum": "All cases progressive and severe long-term. Childhood phase relatively stable (years to decades). Adult phase: rapid decline to dementia + severe parkinsonism within 5-10y of adult phase onset. Life expectancy reduced to 40-60y in most cases.",
        "founder_variant": "De novo mutations — no founder variant (each family has unique mutation). Hotspot: exon 9 (WD40 repeat domain); missense and small deletions/insertions. p.Trp325Gly, p.Val216Gly, p.Gln330* described in multiple cases.",
        "key_variants": ["De novo heterozygous (females)", "p.Trp325Gly", "p.Val216Gly", "p.Gln330*", "Xp11.23 deletion"],
        "seed": SEED_BASE + 4,
    },
    # ── COASY — CoA Synthase (CoPAN) ─────────────────────────────────────────────
    {
        "gene": "COASY", "alias": "COASY — CoPAN: CoA Synthase Protein-Associated Neurodegeneration (OMIM #615643)",
        "aa": "578 aa (bifunctional enzyme)", "kDa": "65 kDa",
        "gene_class": "CoA synthase: bifunctional enzyme catalysing steps 4 and 5 of CoA biosynthesis (4'-phosphopantetheine → dephospho-CoA → CoA)",
        "nbia_subgroup": "CoA biosynthesis pathway (PANK2 · COASY)",
        "locus": "17q21.2", "omim_gene": 609855,
        "phenotype": "CoPAN: adult-onset (range childhood-adult) spastic paraplegia + dystonia + dysarthria + cognitive decline; iron in GP; same CoA synthesis pathway as PANK2; very rare",
        "disease": (
            "COASY encodes CoA synthase (578aa, 65kDa), a bifunctional enzyme located in the mitochondrial matrix "
            "that catalyses the final two steps of CoA biosynthesis: "
            "Step 4: 4'-phosphopantetheine + ATP → dephospho-CoA (PPAT domain of COASY). "
            "Step 5: dephospho-CoA + ATP → CoA (DPCK domain of COASY). "
            "COASY and PANK2 are in the SAME CoA biosynthesis pathway (PANK2 catalyses step 1; "
            "COASY catalyses steps 4-5). Both enzymes are mitochondrial. "
            "COASY deficiency → CoA deficiency in mitochondria → same downstream pathophysiology as PANK2: "
            "fatty acid β-oxidation impairment, TCA cycle dysfunction, and iron accumulation in GP. "
            "CoPAN is extremely rare (fewer than 10 reported families worldwide). "
            "Clinical features: onset variable (childhood to adulthood); "
            "spastic paraplegia (UMN signs predominant); dystonia; dysarthria; cognitive decline; "
            "psychiatric features in some; "
            "MRI: iron in GP (SWI dark signal); T2 GP changes (iron-related); "
            "NO eye-of-tiger (distinguishes from PANK2). "
            "The CoA pathway connection to PANK2 means pantethine/CoA precursors are rational therapeutic "
            "candidates for CoPAN as well as PKAN, but evidence is very limited given extreme rarity."
        ),
        "inheritance": "AR. 17q21.2. Both sexes equally. Extremely rare — fewer than 10 families worldwide reported.",
        "hallmark": (
            "COASY HALLMARKS: "
            "(1) SAME PATHWAY AS PANK2 — CoA biosynthesis: "
            "PANK2 = step 1; COASY = steps 4-5; "
            "both → CoA deficiency → iron accumulation in GP; "
            "PANK2 and COASY are pathway siblings; "
            "(2) EXTREMELY RARE — CoPAN IS THE RAREST NBIA GENE: "
            "fewer than 10 published families; "
            "diagnosis requires high index of suspicion + comprehensive NBIA gene panel; "
            "(3) NO EYE-OF-TIGER: "
            "despite same pathway as PANK2, COASY does NOT show eye-of-tiger sign; "
            "iron in GP but without the central T2 bright spot; "
            "(4) SPASTIC PARAPLEGIA PROMINENT: "
            "UMN signs (hyperreflexia, Babinski) + spasticity; "
            "similar to FA2H/SPG35 in this respect; "
            "(5) PANTETHINE RATIONAL TREATMENT: "
            "exogenous pantethine bypasses PANK2 but NOT steps 4-5 (COASY steps); "
            "less clear rationale than for PANK2 therapy; "
            "(6) COASY IS BIFUNCTIONAL: "
            "one gene encodes two enzymatic activities (PPAT + DPCK domains); "
            "mutations can selectively impair one domain vs both"
        ),
        "key_ddx": (
            "COASY DDx: "
            "(1) PANK2 PKAN: eye-of-tiger PRESENT (COASY absent); same CoA pathway; both AR; "
            "(2) FA2H FAHN/SPG35: leukodystrophy + iron; similar spastic phenotype; "
            "(3) C19orf12 MPAN: similar spastic + dystonic; optic atrophy in MPAN (not COASY); "
            "(4) Other hereditary spastic paraplegias: NO brain iron accumulation; "
            "(5) Friedreich ataxia: cerebellar + cardiomyopathy; GAA repeat; NO iron in GP"
        ),
        "diet_treatment": "No approved disease-modifying therapy. Pantethine supplementation rational (same pathway as PANK2) but not proven in CoPAN. Spasticity: baclofen, tizanidine. Dystonia: trihexyphenidyl. Physiotherapy. Given extreme rarity, evidence base is anecdotal.",
        "gene_therapy_status": "No approved gene therapy. COASY gene delivery is theoretically straightforward (578aa, well-defined bifunctional enzyme). Clinical development limited by extreme rarity.",
        "critical_ci": (
            "CRITICAL: (1) Missing COASY in NBIA panel — must include COASY in comprehensive NBIA gene panels; "
            "(2) Assuming PANK2 is the only CoA-pathway NBIA gene; "
            "(3) Expecting eye-of-tiger in COASY — ABSENT (that is PANK2 specific); "
            "(4) Treating CoPAN as PKAN without genotype confirmation"
        ),
        "nbs_marker": "No NBS marker. Diagnosis: comprehensive NBIA gene panel including COASY. MRI: GP iron on SWI. CoA synthase enzyme activity in fibroblasts (research). COASY sequencing.",
        "key_biomarker": "COASY sequencing (NBIA panel). MRI SWI: GP iron (no eye-of-tiger). CoA levels in fibroblasts (research). CoA synthase activity assay.",
        "severity_spectrum": "Highly variable given extreme rarity. Childhood to adult onset. Progressive spastic-dystonic syndrome. Cognitive decline variable. No mortality data from sufficient cohorts.",
        "founder_variant": "No founder variants identified (too rare). Each family has unique mutations. Primarily missense variants in PPAT or DPCK domains.",
        "key_variants": ["p.Arg499Gln (DPCK domain)", "p.Ala461Val", "p.Gly453Glu", "c.1489+2T>A", "p.Trp360*"],
        "seed": SEED_BASE + 5,
    },
    # ── ATP13A2 — ATPase type 13A2 (Kufor-Rakeb / PARK9) ─────────────────────────
    {
        "gene": "ATP13A2", "alias": "ATP13A2 — Kufor-Rakeb syndrome / PARK9: lysosomal ATPase deficiency (OMIM #606693)",
        "aa": "1180 aa", "kDa": "130 kDa",
        "gene_class": "P5B-type lysosomal ATPase: lysosomal membrane cation pump; polyamine transport; α-synuclein degradation facilitation",
        "nbia_subgroup": "Lysosomal pathway / PARK-NBIA (ATP13A2)",
        "locus": "1p36.13", "omim_gene": 610513,
        "phenotype": "Kufor-Rakeb / PARK9: juvenile-onset parkinsonism + pyramidal signs + supranuclear gaze palsy; L-DOPA responsive (wanes); iron in striatum; also causes Ceroid Lipofuscinosis 12 (CLN12) with biallelic null mutations",
        "disease": (
            "ATP13A2 encodes a 1180aa (130kDa) P5B-type ATPase located on the lysosomal membrane. "
            "ATP13A2 functions as a cation pump, transporting Mn2+, Zn2+, and polyamines across the "
            "lysosomal membrane into the cytosol. It facilitates lysosomal biogenesis, autophagy flux, "
            "and lysosomal protein degradation (including α-synuclein clearance via autophagy-lysosomal pathway). "
            "ATP13A2 deficiency → lysosomal dysfunction → α-synuclein accumulation (Lewy body-like pathology) "
            "+ mitochondrial dysfunction + metal dysregulation → neurodegeneration. "
            "Kufor-Rakeb syndrome (PARK9, OMIM #606693): named after a Jordanian village where first family described. "
            "Clinical features: JUVENILE parkinsonism onset 12-20y (range 7-25y); "
            "bradykinesia, rigidity, tremor — L-DOPA RESPONSIVE initially (response wanes over years); "
            "PYRAMIDAL SIGNS (hyperreflexia, Babinski) — distinguishes from idiopathic PD; "
            "SUPRANUCLEAR GAZE PALSY (limited vertical + horizontal gaze) — highly characteristic; "
            "facial-faucial-finger mini-myoclonus; dementia develops; "
            "MRI: iron in caudate, putamen, GP (SWI dark); "
            "cognitive decline progressive. "
            "Biallelic null mutations cause CLN12 (Ceroid Lipofuscinosis 12): "
            "more severe, with NCL-like storage material (ceroid). "
            "Incidence: rare, ~1/1,000,000+ (Jordan/Middle East higher due to founder). AR inheritance."
        ),
        "inheritance": "AR. 1p36.13. Both sexes. Jordanian founder. Kufor-Rakeb (PARK9) = missense; CLN12 = null alleles (more severe).",
        "hallmark": (
            "ATP13A2 HALLMARKS: "
            "(1) JUVENILE PARKINSONISM + PYRAMIDAL SIGNS: "
            "onset 12-20y; bradykinesia + rigidity + tremor (parkinsonism) "
            "PLUS hyperreflexia + Babinski (pyramidal); "
            "pyramidal signs in parkinsonism → Kufor-Rakeb / Wilson / rare metabolic PD; "
            "(2) SUPRANUCLEAR GAZE PALSY (SNGP): "
            "limited vertical + horizontal gaze; "
            "highly characteristic of Kufor-Rakeb; "
            "also seen in PSP (but adult, tau, no NBIA iron); "
            "SNGP + juvenile parkinsonism + pyramidal + iron → ATP13A2; "
            "(3) L-DOPA RESPONSIVE INITIALLY (THEN WANES): "
            "good initial L-DOPA response (distinguishes from PKAN dystonia); "
            "response declines over years as neurodegeneration progresses; "
            "dopaminergic neurons are lost; "
            "(4) LYSOSOMAL ATPase — α-SYNUCLEIN CONNECTION: "
            "ATP13A2 failure → α-synuclein accumulation (similar to idiopathic PD mechanism but earlier); "
            "NBIA + Lewy body pathology; "
            "links ATP13A2 to PARK pathophysiology; "
            "(5) CAUDATE/PUTAMEN/GP IRON (striatal pattern): "
            "iron in striatum (caudate + putamen + GP); "
            "different topography from PKAN (GP predominant); "
            "(6) CLN12 SEVERE FORM: "
            "biallelic null ATP13A2 → Ceroid Lipofuscinosis 12 (more severe); "
            "NCL storage in neurons; "
            "NCL-like histopathology"
        ),
        "key_ddx": (
            "ATP13A2 DDx: "
            "(1) Wilson disease: KF rings; hepatic disease; copper; responds to chelation (not L-DOPA); "
            "(2) Progressive supranuclear palsy (PSP): adults; tau pathology; NO NBIA iron; "
            "(3) PANK2 PKAN: eye-of-tiger (ATP13A2 absent); dystonia predominant (not parkinsonism); "
            "(4) Idiopathic PD: adults >50y; NO pyramidal signs; NO gaze palsy; "
            "(5) Huntington: CAG repeat; chorea; different brain iron distribution; "
            "(6) Ceroid Lipofuscinosis (other NCL genes): NCL storage; different gene"
        ),
        "diet_treatment": "No disease-modifying therapy. L-DOPA/Carbidopa: initial response good; wanes over years (mean useful period ~5-10y). Manage parkinsonism progression. Spasticity management. Cognitive support. DBS GPi for parkinsonism (case reports; limited evidence). Ophthalmological monitoring (gaze palsy). Lysosomal-targeting therapies in research.",
        "gene_therapy_status": "No approved gene therapy. ATP13A2 (1180aa) is a large gene, challenging for standard AAV packaging. Split-AAV approaches or lentiviral approaches explored. Lysosomal pathway modulation (mTOR, TFEB activation) as alternative.",
        "critical_ci": (
            "CRITICAL: (1) Missing ATP13A2 in juvenile parkinsonism workup — always include in PARK gene panel for <30y onset; "
            "(2) Not testing for supranuclear gaze palsy — specific clinical sign; "
            "(3) Labelling as idiopathic PD in a 15-year-old — juvenile onset MUST trigger genetic workup; "
            "(4) Expecting sustained L-DOPA benefit — benefit wanes; anticipate and counsel; "
            "(5) Missing CLN12 spectrum — null mutations cause NCL-like disease"
        ),
        "nbs_marker": "No NBS marker. Diagnosis: ATP13A2 sequencing (PARK panel or NBIA panel). MRI: striatal iron on SWI. Ophthalmology (gaze palsy, ERG for CLN12). EEG (seizures in CLN12). Skin biopsy for NCL electron microscopy (CLN12). Dopamine transporter SPECT (DAT scan): reduced uptake (dopaminergic neuron loss).",
        "key_biomarker": "ATP13A2 sequencing. DAT-SPECT (reduced). MRI SWI: striatal iron (caudate/putamen/GP). Ophthalmology (gaze palsy, ERG). Skin biopsy (curvilinear bodies in CLN12). CSF neurofilament light (elevated).",
        "severity_spectrum": "Progressive. Onset 12-20y; wheelchair-bound by 30-40y; dementia by 40s; severe disability. Biallelic null alleles (CLN12) are more severe with earlier onset and NCL features.",
        "founder_variant": "Jordanian founder: c.3176T>G p.Leu1059Arg (first Kufor-Rakeb family). Other founders in Chilean, Pakistani families. c.1306+5G>A splice variant (European). p.Gly504Arg (Pakistani).",
        "key_variants": ["c.3176T>G p.Leu1059Arg (Jordan founder)", "c.1306+5G>A (splice)", "p.Gly504Arg (Pakistan)", "p.Arg226Gln", "Exon 27 deletion"],
        "seed": SEED_BASE + 6,
    },
    # ── DCAF17 — DDB1 and CUL4 Associated Factor 17 (Woodhouse-Sakati) ───────────
    {
        "gene": "DCAF17", "alias": "DCAF17 — Woodhouse-Sakati syndrome (WSS): nucleolar protein with systemic features (OMIM #611237)",
        "aa": "591 aa", "kDa": "67 kDa",
        "gene_class": "DDB1-CUL4 associated factor 17 (DCAF17): nucleolar protein; CRL4-DCAF17 E3 ubiquitin ligase substrate receptor; RNA Pol I-associated; ribosome biogenesis",
        "nbia_subgroup": "Nucleolar / ubiquitin pathway (DCAF17)",
        "locus": "2q31.1", "omim_gene": 612515,
        "phenotype": "Woodhouse-Sakati syndrome: ONLY NBIA WITH SYSTEMIC FEATURES — hypogonadism (hypergonadotropic), diabetes mellitus, alopecia, sensorineural deafness + neurological (dystonia, dysarthria, cognitive decline) + brain iron in GP",
        "disease": (
            "DCAF17 encodes a 591aa (67kDa) nucleolar protein that functions as a substrate receptor "
            "for the CRL4-DCAF17 E3 ubiquitin ligase complex. DCAF17 localises to the nucleolus and "
            "associates with RNA polymerase I transcription machinery, implicating it in ribosome biogenesis. "
            "Loss of DCAF17 → nucleolar dysfunction → widespread protein synthesis impairment → "
            "neurodegeneration + endocrine dysfunction + hearing loss. "
            "Woodhouse-Sakati syndrome (OMIM #612514) is the ONLY NBIA disorder with prominent SYSTEMIC "
            "(non-neurological) features: "
            "HYPOGONADISM (hypergonadotropic): gonadal failure; primary hypogonadism with high LH/FSH and low "
            "oestrogen/testosterone; amenorrhoea in females; delayed/absent puberty; infertility. "
            "DIABETES MELLITUS: insulin-dependent type 1-like; onset variable. "
            "ALOPECIA: progressive scalp hair loss (not universal but common). "
            "SENSORINEURAL HEARING LOSS (SNHL): progressive. "
            "Neurological features (onset 10-30y): DYSTONIA (typically focal → generalised); "
            "DYSARTHRIA; cognitive decline; movement disorder. "
            "MRI: iron in GP (SWI dark signal); NO eye-of-tiger. "
            "First described by Woodhouse and Sakati in Saudi Arabia (1983). "
            "Strong founder in Saudi Arabian and other Gulf Arab populations: c.436delC p.Leu146Tyrfs*20. "
            "Incidence: very rare; highest in Arabian Peninsula. AR inheritance."
        ),
        "inheritance": "AR. 2q31.1. Both sexes. Saudi/Gulf Arab founder c.436delC (p.Leu146Tyrfs*20). Also found in Libyan, Turkish, Iranian families.",
        "hallmark": (
            "DCAF17 HALLMARKS: "
            "(1) ONLY NBIA GENE WITH SYSTEMIC FEATURES: "
            "ALL OTHER NBIA GENES CAUSE PURE NEUROLOGICAL DISEASE; "
            "DCAF17/WSS has HYPOGONADISM + DIABETES + ALOPECIA + DEAFNESS; "
            "systemic features may PRECEDE neurological features by years; "
            "any patient with hypogonadism + movement disorder = DCAF17 until proven otherwise; "
            "(2) HYPERGONADOTROPIC HYPOGONADISM (PRIMARY GONADAL FAILURE): "
            "high LH/FSH (pituitary responding appropriately); low gonadal hormones; "
            "NOT secondary/hypogonadotropic (pituitary is intact); "
            "amenorrhoea in females; azoospermia in males; infertility; "
            "hormone replacement therapy indicated; "
            "(3) SAUDI/GULF ARAB FOUNDER c.436delC: "
            "frameshift; prevalent in Arabian Peninsula; "
            "targeted testing justified in Saudi, Qatari, UAE patients; "
            "(4) NUCLEOLAR DYSFUNCTION — SYSTEMIC PROTEIN SYNTHESIS IMPAIRMENT: "
            "DCAF17 in nucleolus + CRL4 E3 ligase → ribosome biogenesis disruption → "
            "tissues with highest protein demand (gonads, endocrine, neurons) most affected; "
            "(5) GP IRON WITHOUT EYE-OF-TIGER: "
            "SWI shows GP dark signal; NO central T2 bright spot; "
            "(6) DYSTONIA ONSET TYPICALLY LATER (10-30y): "
            "systemic features (hypogonadism, diabetes) may be diagnosed first; "
            "neurological onset later"
        ),
        "key_ddx": (
            "DCAF17 DDx: "
            "(1) Alstrom syndrome (ALMS1): obesity + SNHL + vision loss + diabetes; NO hypogonadism typically; NO brain iron; "
            "(2) Bardet-Biedl syndrome: obesity + polydactyly + retinal dystrophy + hypogonadism; NO brain iron; "
            "(3) Wolfram syndrome (WFS1): diabetes insipidus + diabetes mellitus + optic atrophy + deafness (DIDMOAD); NO brain iron; "
            "(4) PANK2 PKAN: pure neurological; NO systemic features; eye-of-tiger; "
            "(5) Primary hypogonadism (other causes): NO neurological features; NO brain iron; "
            "(6) Turner syndrome (females): 45X; NO brain iron; NO dystonia"
        ),
        "diet_treatment": "Sex hormone replacement therapy (oestrogen for females, testosterone for males) for hypogonadism. Insulin for diabetes mellitus. Hearing aids for SNHL. Cochlear implant consideration. Dystonia: trihexyphenidyl, botulinum toxin, baclofen. DBS GPi for refractory dystonia (case reports). Fertility preservation counselling before gonadal failure complete.",
        "gene_therapy_status": "No approved gene therapy. DCAF17 (591aa) suitable for AAV packaging. No active clinical development. Systemic (non-CNS) features would also require systemic gene delivery.",
        "critical_ci": (
            "CRITICAL: (1) Missing WSS when seeing young hypogonadism + movement disorder — "
            "test DCAF17 in any patient with unexplained hypogonadism + neurological features; "
            "(2) Diagnosing PKAN without checking for systemic features (in patients from Gulf region especially); "
            "(3) Missing hormone replacement — hypogonadism is treatable; "
            "(4) Not offering fertility counselling — primary gonadal failure is progressive; early intervention matters; "
            "(5) Missing diabetes management — can be severe if untreated"
        ),
        "nbs_marker": "No NBS marker. Diagnosis: endocrine workup (LH/FSH/oestrogen/testosterone) + MRI (GP iron on SWI) + DCAF17 sequencing. Audiological assessment (SNHL). HbA1c/glucose (diabetes). Targeted c.436delC testing in Saudi/Gulf Arab patients.",
        "key_biomarker": "DCAF17 sequencing (targeted c.436delC in Saudi/Gulf). Endocrine: LH/FSH elevated; oestrogen/testosterone low. HbA1c/glucose (diabetes). Audiometry (SNHL). MRI SWI: GP iron. Dystonia: EMG, clinical assessment.",
        "severity_spectrum": "Progressive systemic + neurological disease. Endocrine features often first (delayed puberty, amenorrhoea). Neurological onset 10-30y. Progresses to generalised dystonia + cognitive decline. Lifespan reduced; hormone replacement and diabetes management crucial for quality of life.",
        "founder_variant": "c.436delC (p.Leu146Tyrfs*20) — Saudi Arabia/Gulf Arab founder; most common mutation worldwide. Also p.Arg408Gln (Libyan families), p.Asp349Asn (Turkish).",
        "key_variants": ["c.436delC p.Leu146Tyrfs*20 (Saudi/Gulf founder)", "p.Arg408Gln (Libyan)", "p.Asp349Asn (Turkish)", "p.Ala369Pro", "c.1028+1G>A"],
        "seed": SEED_BASE + 7,
    },
]


def _make_patients(gene_dict):
    """Generate 40 synthetic patient records for a given NBIA gene."""
    rng = random.Random(gene_dict["seed"])
    gene = gene_dict["gene"]

    # Phenotypic class probabilities per gene
    PHENO_PROBS = {
        "PANK2":    [0.60, 0.40, 0.00],   # Classic PKAN / Atypical PKAN
        "PLA2G6":   [0.55, 0.30, 0.15],   # INAD / PLAN / PARK14
        "C19orf12": [0.70, 0.30, 0.00],   # Typical MPAN / Mild/late-onset MPAN
        "FA2H":     [0.65, 0.35, 0.00],   # FAHN / Milder SPG35 predominant
        "WDR45":    [0.75, 0.25, 0.00],   # Classic BPAN / Mild BPAN
        "COASY":    [0.60, 0.40, 0.00],   # Classic CoPAN / Milder
        "ATP13A2":  [0.65, 0.25, 0.10],   # Kufor-Rakeb / CLN12 form / Mild
        "DCAF17":   [0.70, 0.20, 0.10],   # WSS classic / Neurological-predominant / Mild
    }
    CLASS_NAMES = {
        "PANK2":    ["Classic PKAN (<10y onset)", "Atypical PKAN (>10y onset)", "Variant"],
        "PLA2G6":   ["INAD (Infantile)", "PLAN/NBIA (Juvenile)", "PARK14 (Adult)"],
        "C19orf12": ["Typical MPAN", "Late-onset MPAN", "Mild"],
        "FA2H":     ["FAHN (Typical)", "SPG35 Predominant", "Mild"],
        "WDR45":    ["Classic BPAN", "Mild BPAN", "Variant"],
        "COASY":    ["Classic CoPAN", "Milder CoPAN", "Variant"],
        "ATP13A2":  ["Kufor-Rakeb (PARK9)", "CLN12 Form", "Mild"],
        "DCAF17":   ["WSS Classic", "Neurological Predominant", "Mild WSS"],
    }
    probs = PHENO_PROBS.get(gene, [0.50, 0.35, 0.15])
    classes = CLASS_NAMES.get(gene, ["Severe", "Moderate", "Mild"])

    # Age at diagnosis by gene and phenotypic class
    AGE_RANGES = {
        "PANK2":    [(3, 10), (10, 30), (20, 50)],
        "PLA2G6":   [(0.5, 2.5), (1, 8), (20, 45)],
        "C19orf12": [(10, 25), (20, 40), (30, 55)],
        "FA2H":     [(4, 12), (6, 20), (15, 35)],
        "WDR45":    [(1, 10), (5, 20), (1, 15)],
        "COASY":    [(5, 25), (10, 35), (20, 50)],
        "ATP13A2":  [(12, 22), (10, 18), (20, 35)],
        "DCAF17":   [(10, 25), (15, 30), (20, 40)],
    }

    patients = []
    for i in range(40):
        r = rng.random()
        if r < probs[0]:
            pheno_idx = 0
        elif r < probs[0] + probs[1]:
            pheno_idx = 1
        else:
            pheno_idx = 2
        pheno = classes[pheno_idx]

        age_range = AGE_RANGES.get(gene, [(5, 30), (10, 40), (20, 50)])[pheno_idx]
        age_dx = round(rng.uniform(*age_range), 1)

        # Sex assignment
        if gene == "WDR45":
            # X-linked dominant; females ~4:1
            sex = rng.choice(["F", "F", "F", "F", "M"])
        else:
            sex = rng.choice(["M", "F"])

        # Iron on MRI (SWI)
        iron_on_mri = True if pheno_idx <= 1 else rng.random() < 0.7

        # Eye-of-tiger (PANK2 only)
        eot = (gene == "PANK2") and (rng.random() < 0.95)

        # L-DOPA response
        if gene == "ATP13A2":
            ldopa_response = rng.choice(["Good initial response", "Moderate initial response", "Partial response"])
        elif gene in ("PANK2", "WDR45"):
            ldopa_response = rng.choice(["Partial", "Minimal", "None"])
        elif gene == "PLA2G6" and pheno_idx == 2:  # PARK14
            ldopa_response = rng.choice(["Good initial", "Moderate", "Partial"])
        else:
            ldopa_response = rng.choice(["Partial", "Minimal", "None"])

        # Distinctive features by gene
        if gene == "PANK2":
            presenting_feature = rng.choice(["Generalised dystonia", "Gait dystonia", "Dysarthria + dystonia", "Foot dystonia"])
            outcome_class = rng.choice(["Progressive dystonia", "Loss of ambulation by teenage years", "Stable on DBS"])
            retinopathy = rng.random() < 0.70
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting_feature,
                "eye_of_tiger_mri": eot,
                "iron_on_mri": iron_on_mri,
                "retinopathy": retinopathy,
                "l_dopa_response": ldopa_response,
                "outcome_class": outcome_class,
            })
        elif gene == "PLA2G6":
            presenting_feature = (
                rng.choice(["Hypotonia", "Cerebellar ataxia (infant)", "Strabismus + hypotonia"]) if pheno_idx == 0
                else rng.choice(["Dystonia + ataxia", "Regression", "Eye movement abnormality"]) if pheno_idx == 1
                else rng.choice(["Parkinsonism", "Psychiatric symptoms", "Resting tremor"])
            )
            axonal_spheroids = (pheno_idx == 0) and (rng.random() < 0.90)
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting_feature,
                "eye_of_tiger_mri": False,
                "iron_on_mri": iron_on_mri,
                "axonal_spheroids": axonal_spheroids,
                "l_dopa_response": ldopa_response,
            })
        elif gene == "C19orf12":
            presenting_feature = rng.choice(["Psychiatric symptoms", "Spastic gait", "Dystonia onset", "Optic atrophy noted"])
            optic_atrophy = rng.random() < 0.60
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting_feature,
                "eye_of_tiger_mri": False,
                "iron_on_mri": iron_on_mri,
                "optic_atrophy": optic_atrophy,
                "l_dopa_response": ldopa_response,
            })
        elif gene == "FA2H":
            presenting_feature = rng.choice(["Spastic gait", "Cerebellar ataxia", "Leukodystrophy on MRI", "Dysarthria"])
            leukodystrophy_on_mri = rng.random() < 0.90
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting_feature,
                "eye_of_tiger_mri": False,
                "iron_on_mri": iron_on_mri,
                "leukodystrophy_mri": leukodystrophy_on_mri,
                "l_dopa_response": ldopa_response,
            })
        elif gene == "WDR45":
            presenting_feature = rng.choice(["Intellectual disability", "Seizures", "Absent speech", "Autistic features"])
            adult_phase = (pheno_idx == 0) and (rng.random() < 0.80)
            de_novo = rng.random() < 0.97
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting_feature,
                "eye_of_tiger_mri": False,
                "iron_on_mri": iron_on_mri,
                "de_novo_mutation": de_novo,
                "adult_parkinsonism_dementia": adult_phase,
                "l_dopa_response": ldopa_response,
            })
        elif gene == "COASY":
            presenting_feature = rng.choice(["Spastic paraplegia", "Dystonia", "Dysarthria", "Cognitive decline"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting_feature,
                "eye_of_tiger_mri": False,
                "iron_on_mri": iron_on_mri,
                "l_dopa_response": ldopa_response,
            })
        elif gene == "ATP13A2":
            presenting_feature = rng.choice(["Juvenile parkinsonism", "Pyramidal signs + parkinsonism", "Gaze palsy noted", "Tremor + rigidity"])
            gaze_palsy = rng.random() < 0.75
            pyramidal_signs = rng.random() < 0.85
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting_feature,
                "eye_of_tiger_mri": False,
                "iron_on_mri": iron_on_mri,
                "supranuclear_gaze_palsy": gaze_palsy,
                "pyramidal_signs": pyramidal_signs,
                "l_dopa_response": ldopa_response,
            })
        elif gene == "DCAF17":
            presenting_feature = rng.choice(["Delayed puberty / amenorrhoea", "Hypogonadism detected", "Dystonia onset", "Alopecia noted"])
            hypogonadism = rng.random() < 0.95
            diabetes = rng.random() < 0.65
            alopecia = rng.random() < 0.60
            snhl = rng.random() < 0.70
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting_feature,
                "eye_of_tiger_mri": False,
                "iron_on_mri": iron_on_mri,
                "hypogonadism": hypogonadism,
                "diabetes_mellitus": diabetes,
                "alopecia": alopecia,
                "snhl": snhl,
                "l_dopa_response": ldopa_response,
            })
        else:
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "iron_on_mri": iron_on_mri,
                "eye_of_tiger_mri": False,
            })

    return patients


# ── Populate patient cohorts ──────────────────────────────────────────────────────
for _g in NBIA_GENES:
    _g["patients"] = _make_patients(_g)
    _g["n_patients"] = len(_g["patients"])

ALL_PATIENTS = [p for g in NBIA_GENES for p in g["patients"]]


# ─── API: get_overview ───────────────────────────────────────────────────────────
def get_overview():
    total = len(ALL_PATIENTS)

    gene_summary = []
    for g in NBIA_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "locus": g["locus"],
            "gene_class": g["gene_class"],
            "nbia_subgroup": g["nbia_subgroup"],
            "n_patients": g["n_patients"],
            "phenotype": g["phenotype"],
            "diet_treatment": g["diet_treatment"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })

    # Summary statistics
    n_eye_of_tiger = sum(1 for p in ALL_PATIENTS if p.get("eye_of_tiger_mri", False))
    n_iron_on_mri  = sum(1 for p in ALL_PATIENTS if p.get("iron_on_mri", False))
    n_optic_atrophy = sum(1 for p in ALL_PATIENTS if p.get("optic_atrophy", False))
    n_systemic      = sum(1 for p in ALL_PATIENTS if p.get("hypogonadism", False))
    n_de_novo       = sum(1 for p in ALL_PATIENTS if p.get("de_novo_mutation", False))

    return {
        "atlas": "NBIA-Atlas — Complete 8-Gene Neurodegeneration with Brain Iron Accumulation Atlas",
        "n_genes": len(NBIA_GENES),
        "n_patients": total,
        "seeds": [g["seed"] for g in NBIA_GENES],
        "genes_covered": [g["gene"] for g in NBIA_GENES],
        "gene_subgroups": {
            "CoA biosynthesis (PANK2 · COASY)": ["PANK2", "COASY"],
            "Phospholipid remodelling (PLA2G6 · FA2H)": ["PLA2G6", "FA2H"],
            "Autophagy pathway (WDR45)": ["WDR45"],
            "Lysosomal / PARK-NBIA (ATP13A2)": ["ATP13A2"],
            "Mitochondria-associated membrane (C19orf12)": ["C19orf12"],
            "Nucleolar / ubiquitin (DCAF17)": ["DCAF17"],
        },
        "n_eye_of_tiger": n_eye_of_tiger,
        "n_iron_on_mri": n_iron_on_mri,
        "n_optic_atrophy": n_optic_atrophy,
        "n_systemic_features": n_systemic,
        "n_de_novo_wdr45": n_de_novo,
        "critical_clinical_rules": [
            "EYE-OF-TIGER SIGN IS PANK2/PKAN-SPECIFIC: bilateral symmetric GP lesion with central T2 hyperintensity (gliosis) + peripheral T2 hypointensity (iron) is PATHOGNOMONIC for PKAN caused by PANK2 mutations; NO OTHER NBIA GENE produces the eye-of-tiger sign; presence = test PANK2 first; absence does NOT exclude NBIA (all other 7 genes lack it)",
            "WDR45 IS X-LINKED DOMINANT DE NOVO — TEST FEMALES: BPAN is caused by de novo heterozygous WDR45 mutations on Xp11.23; virtually all patients are FEMALE (hemizygous males lethal in utero); no family history expected (de novo); BIPHASIC COURSE (stable childhood intellectual disability + epilepsy → ABRUPT adult-onset parkinsonism-dementia) is pathognomonic; any female with unexplained ID + seizures needs WDR45 sequencing",
            "DCAF17/WSS IS THE ONLY NBIA WITH SYSTEMIC FEATURES: PANK2, PLA2G6, C19orf12, FA2H, WDR45, COASY, ATP13A2 all cause PURE neurological disease; DCAF17/Woodhouse-Sakati has HYPERGONADOTROPIC HYPOGONADISM + DIABETES MELLITUS + ALOPECIA + SENSORINEURAL DEAFNESS + neurological features; systemic features often precede the movement disorder by years; hypogonadism in a young patient + movement disorder = DCAF17",
            "FA2H IS THE ONLY NBIA GENE WITH LEUKODYSTROPHY + IRON: all other NBIA genes cause grey matter (basal ganglia) iron accumulation; FA2H causes BOTH leukodystrophy (T2 periventricular white matter hyperintensity from demyelination) AND striatal iron accumulation; the combination of leukodystrophy + iron on MRI in a child with dystonia/spastic paraplegia = FA2H until proven otherwise",
            "C19orf12 MPAN — OPTIC ATROPHY IN ~60% AND PSYCHIATRIC ONSET: optic atrophy (fundoscopic disc pallor + VEP delay) is present in 50-70% of MPAN and distinguishes it from PKAN and WDR45 (which do NOT have optic atrophy); psychiatric features (depression, psychosis) typically precede the movement disorder by years, leading to misdiagnosis as primary psychiatric disorder; fundoscopy + VEP mandatory in all NBIA workup",
            "ATP13A2 KUFOR-RAKEB — JUVENILE PARKINSONISM + PYRAMIDAL + GAZE PALSY TRIAD: the combination of parkinsonism onset <25y + UMN signs (hyperreflexia, Babinski) + supranuclear gaze palsy is highly characteristic; L-DOPA responsive initially but response wanes; pyramidal signs in a young patient with parkinsonism should trigger ATP13A2 sequencing; also causes CLN12 (ceroid lipofuscinosis) with biallelic null alleles",
            "PANK2/COASY SHARE CoA PATHWAY: PANK2 (step 1) and COASY (steps 4-5) are in the same CoA biosynthesis pathway; pantethine (CoA precursor) is a rational therapeutic for PKAN; deferiprone (iron chelation) shows partial benefit in PKAN specifically; neither has proven benefit for other NBIA genes; DBS GPi for palliation of dystonia (especially PKAN) is the main invasive option",
            "PLA2G6 THREE-PHENOTYPE SPECTRUM: INAD (infantile; axonal spheroids; cerebellar atrophy — NO iron early), PLAN/NBIA (juvenile; iron + ataxia), PARK14 (adult; L-DOPA-responsive PD-like + psychiatric); the infant with hypotonia + cerebellar atrophy and the young adult with PD + psychiatric features may both have PLA2G6 mutations; molecular diagnosis has replaced nerve biopsy for INAD",
            "COMPREHENSIVE NBIA GENE PANEL MANDATORY: clinical and MRI features alone cannot definitively diagnose individual NBIA genes (except PKAN with eye-of-tiger + PANK2 confirmation); all patients with basal ganglia iron accumulation on MRI require sequencing of the full NBIA gene panel: PANK2, PLA2G6, C19orf12, FA2H, WDR45, COASY, ATP13A2, DCAF17 (minimum); additional genes: MPAN, SENDA, BPAN-negative cases may have novel genes",
            "NO DISEASE-MODIFYING THERAPY APPROVED FOR ANY NBIA GENE: iron chelation (deferiprone) shows partial benefit in PKAN only (TIRCON trial data; not conclusive); DBS is palliative for dystonia; mTOR inhibitors for WDR45/BPAN are in clinical trials; pantethine for PKAN is experimental; symptomatic treatment (antispasticity, antidystonia, L-DOPA for parkinsonian forms) is the standard of care",
        ],
        "gene_summary": gene_summary,
        "mri_note": "SWI (susceptibility-weighted imaging) and GRE sequences are MANDATORY for iron detection — conventional T1/T2 may miss early iron; Eye-of-Tiger is a T2 sequence finding specific to PANK2/PKAN; iron dark on T2/SWI in all NBIA genes.",
    }


# ─── API: get_breakdown ──────────────────────────────────────────────────────────
def get_breakdown():
    gene_rows = []
    for g in NBIA_GENES:
        pts = g["patients"]
        n_eot = sum(1 for p in pts if p.get("eye_of_tiger_mri", False))
        n_iron = sum(1 for p in pts if p.get("iron_on_mri", False))
        gene_rows.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "gene_class": g["gene_class"],
            "nbia_subgroup": g["nbia_subgroup"],
            "n_patients": g["n_patients"],
            "seed": g["seed"],
            "phenotype": g["phenotype"],
            "inheritance": g["inheritance"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "diet_treatment": g["diet_treatment"],
            "gene_therapy_status": g["gene_therapy_status"],
            "critical_ci": g["critical_ci"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "key_variants": g["key_variants"],
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
            "n_eye_of_tiger": n_eot,
            "n_iron_mri": n_iron,
        })
    return {
        "genes": gene_rows,
        "total": len(NBIA_GENES),
        "total_patients": len(ALL_PATIENTS),
    }


# ─── API: get_definitions ─────────────────────────────────────────────────────
def get_definitions():
    return {
        "atlas": "NBIA-Atlas — Complete 8-Gene Neurodegeneration with Brain Iron Accumulation Atlas",
        "nbia_overview": {
            "full_name": "Neurodegeneration with Brain Iron Accumulation (NBIA) — heterogeneous group of inherited progressive neurodegenerative diseases characterised by iron accumulation predominantly in the basal ganglia (globus pallidus, substantia nigra) and neuronal degeneration",
            "genes_in_atlas": 8,
            "collective_incidence": "PANK2/PKAN: ~50% of all NBIA (~1-3/1,000,000); C19orf12/MPAN: ~20%; PLA2G6/PLAN: ~10%; FA2H/FAHN: ~5%; WDR45/BPAN: ~5%; ATP13A2: ~5%; COASY/CoPAN: <1%; DCAF17/WSS: ~1-2%",
            "mri_note": "SWI/GRE imaging MANDATORY for iron detection; eye-of-tiger ONLY in PANK2; other genes show iron without central T2 bright spot",
        },
        "definitions": [
            {
                "term": "NBIA — Neurodegeneration with Brain Iron Accumulation: Definition and Classification",
                "definition": "NBIA encompasses at least 15 genetic disorders defined by: (1) progressive neurological deterioration (movement disorder, cognitive decline), (2) iron accumulation in the basal ganglia (globus pallidus, substantia nigra) detected on MRI (SWI/GRE sequences), and (3) underlying genetic defect. Pathological iron accumulates in glia and neurons of the basal ganglia — iron acts as a reactive oxygen species (ROS) source (Fenton reaction) → oxidative stress → neuronal death. Common pathways disrupted: CoA biosynthesis (PANK2, COASY), phospholipid remodelling (PLA2G6, FA2H), autophagy (WDR45), lysosomal function (ATP13A2), mitochondria-associated membranes (C19orf12), nucleolar function (DCAF17). Not all NBIA genes are represented in this atlas (e.g., MPAN-related genes, SENDA/WDR45-related). Shared clinical features: extrapyramidal movement disorder (dystonia, parkinsonism, chorea), corticospinal tract signs (spasticity, pyramidal signs), progressive neurological decline; onset from infancy to adulthood.",
            },
            {
                "term": "Eye-of-Tiger Sign — PATHOGNOMONIC for PKAN (PANK2 Only)",
                "definition": "The Eye-of-Tiger sign is a T2-weighted MRI finding consisting of: (1) bilateral, symmetric hypointensity (dark signal) in the globus pallidus medialis (GPm) — corresponding to iron accumulation, and (2) a central focus of T2 hyperintensity (bright signal) within the GP hypointensity — corresponding to gliosis and/or cystic necrosis/vacuolisation. On axial T2 images, this produces an appearance resembling a tiger's eye. Key facts: PATHOGNOMONIC for PKAN (PANK2 mutations): no other NBIA gene reliably produces the eye-of-tiger sign; ~95% of PKAN cases show it; ~5% have iron-only GP changes (variant PKAN). The presence of eye-of-tiger = test PANK2 FIRST; if PANK2 sequencing is negative in a patient with eye-of-tiger, consider: (a) variant PKAN with deep intronic or promoter mutations; (b) repeat PANK2 sequencing with deletion/duplication analysis; (c) rarely, other causes. Not a feature of: PLA2G6, C19orf12, FA2H, WDR45, COASY, ATP13A2, DCAF17. SWI sequences are better than T2 for detecting iron; eye-of-tiger is specifically a T2 finding.",
            },
            {
                "term": "BPAN Biphasic Course — Childhood Static Phase → Adult Parkinsonism-Dementia",
                "definition": "BPAN (Beta-propeller protein-associated neurodegeneration) caused by WDR45 has a clinically distinctive biphasic disease course: Phase 1 — Childhood Static Phase (variable duration, typically 1-20+ years): Intellectual disability (moderate-severe), epilepsy (drug-resistant, multiple seizure types including absence, myoclonic, generalised tonic-clonic), absent or severely limited speech, autistic features, sleep disturbance (melatonin-related circadian rhythm disruption). MRI may show minimal or no iron early; subtle SN signal changes may be present. Neurological status is relatively stable (not rapidly progressive) in Phase 1 — hence 'static.' Families and clinicians may attribute the disability to a static encephalopathy (e.g., Rett syndrome, Angelman syndrome, cerebral palsy). Phase 2 — Adult Parkinsonism-Dementia Phase (onset typically 20-40y): Abrupt onset (over 1-2 years) of parkinsonism (bradykinesia, rigidity, tremor) + rapid cognitive decline to dementia. MRI: striking iron accumulation in SN (T1 hyperintensity; SWI hypointensity) + GP. The transition from stable childhood → sudden adult deterioration is the clinical signature of BPAN and should prompt immediate WDR45 sequencing in any patient previously labelled with 'static encephalopathy.'",
            },
            {
                "term": "PANK2-COASY CoA Biosynthesis Pathway — Pantothenate → CoA",
                "definition": "Coenzyme A (CoA) is an essential cofactor for >100 metabolic reactions: fatty acid β-oxidation, TCA cycle (acetyl-CoA, succinyl-CoA), fatty acid synthesis, amino acid metabolism. CoA is synthesised from pantothenate (vitamin B5) in 5 steps: Step 1 (PANK2): Pantothenate + ATP → 4'-phosphopantothenate. Rate-limiting step. Step 2 (PPCDC): 4'-phosphopantothenate → 4'-phosphopantothenoylcysteine. Step 3 (PPC): 4'-phosphopantothenoylcysteine → 4'-phosphopantetheine. Step 4 (COASY, PPAT domain): 4'-phosphopantetheine + ATP → dephospho-CoA. Step 5 (COASY, DPCK domain): dephospho-CoA + ATP → CoA. PANK2 (step 1) and COASY (steps 4-5) are the two NBIA genes in this pathway. Both are mitochondrial enzymes. CoA deficiency → cysteine accumulation (step 2 substrate backs up when step 1 blocked) → cysteine auto-oxidation + iron chelation → iron-cysteine complex deposition in GP → NBIA. Therapeutic rationale: pantethine (4'-phosphopantetheine precursor) bypasses step 1 (PANK2 block) and enters the pathway at step 2, providing CoA precursor downstream — rational for PANK2 but less so for COASY (steps 4-5 still blocked).",
            },
            {
                "term": "PLA2G6 Axonal Spheroids — Pathological Hallmark of INAD",
                "definition": "Axonal spheroids are focal swellings of neuronal axons filled with organelle debris — the pathological hallmark of INAD (Infantile Neuroaxonal Dystrophy) caused by PLA2G6 mutations. iPLA2β (PLA2G6) remodels mitochondrial and ER membrane phospholipids (removing damaged sn-2 fatty acids). Without iPLA2β: abnormal phospholipid accumulation → mitochondrial membrane breakdown → axons swell with dysfunctional mitochondria, myelin-like whorls, and membranous debris. Macroscopic: axonal enlargement visible on histology. Staining: PAS-positive (glycoprotein content); electron microscopy shows tubulovesicular bodies ('fingerprint profiles' or 'osmiophilic debris'). Distribution: ubiquitous in CNS axons; cerebellar cortex axons are most severely affected → cerebellar atrophy predominates on MRI. Peripheral nerves (sural nerve biopsy): spheroids detectable → historically used for diagnosis before molecular testing. Modern practice: PLA2G6 sequencing has largely replaced sural nerve biopsy for INAD diagnosis. Spheroids differentiate INAD from other cerebellar atrophy causes.",
            },
            {
                "term": "Deferiprone — Iron Chelation in NBIA",
                "definition": "Deferiprone (DFP, 3-hydroxy-1,2-dimethyl-4(1H)-pyridone) is an oral iron chelator that crosses the blood-brain barrier — unlike deferoxamine which does not. Mechanism: chelates Fe3+ → deferiprone-Fe3+ complex excreted renally. Rationale in NBIA: iron accumulation drives ROS-mediated neurodegeneration (Fenton reaction); chelating excess iron should slow neurodegeneration. Clinical evidence: TIRCON trial (deferiprone in PKAN; n=88; 18 months): primary endpoint (PKAN DRS score) not significantly different, but trend toward slowing. FAIR-PARK I (deferiprone in PD): showed MRI iron reduction. Evidence is weak and conflicting for NBIA. Current status: NOT yet standard of care for PKAN; considered experimental. PKAN is the only NBIA gene for which there is any clinical trial data for deferiprone. No evidence for other NBIA genes (PLA2G6, C19orf12, FA2H, WDR45, COASY, ATP13A2, DCAF17). Side effects: agranulocytosis (rare but serious; CBC monitoring mandatory), GI symptoms, arthropathy. Dose: typically 25-30 mg/kg/day in 3 divided doses.",
            },
            {
                "term": "Deep Brain Stimulation (DBS) in NBIA — Palliation, Not Cure",
                "definition": "DBS involves surgical implantation of electrodes in the globus pallidus internus (GPi) or subthalamic nucleus (STN), connected to an implantable pulse generator (IPG). For dystonia in NBIA: DBS GPi reduces dystonia severity and improves quality of life — principally studied in PKAN. Multiple case reports and small series show significant dystonia improvement with GPi-DBS in PKAN. DBS does NOT modify the underlying disease or stop neurodegeneration. Benefits: symptom palliation (reduced dystonia, pain from sustained muscle contractions, improved feeding). Who benefits: PKAN cases with severe drug-refractory dystonia; MPAN/C19orf12 (case reports); other NBIA genes (anecdotal). Limitations: progressive neurodegeneration continues; cognition and bulbar function decline independent of DBS; technical challenges in young patients. Pre-DBS evaluation: multidisciplinary (neurology, neurosurgery, neuropsychology); MRI for lead placement planning; GP iron accumulation must be considered in electrode placement.",
            },
            {
                "term": "Woodhouse-Sakati Syndrome (DCAF17) — The Only NBIA with Systemic Features",
                "definition": "Woodhouse-Sakati syndrome (WSS) is the only NBIA disorder with non-neurological systemic features: (1) Hypergonadotropic hypogonadism (primary gonadal failure): LH and FSH are ELEVATED (pituitary appropriately increases gonadotropins because the gonads fail to respond). Gonadal hormones (oestrogen, testosterone) are LOW. Presentation: delayed/absent puberty, amenorrhoea in females, azoospermia in males, infertility. Treatment: sex hormone replacement therapy (oestrogen or testosterone). (2) Diabetes mellitus: autoimmune-like type 1 presentation; insulin-dependent. (3) Alopecia: progressive scalp hair loss (not universal). (4) Sensorineural hearing loss (SNHL): bilateral, progressive. (5) Neurological features: dystonia (often the last feature to appear; focal → generalised), dysarthria, cognitive decline; MRI GP iron (SWI dark signal). DCAF17 encodes a nucleolar protein; ribosome biogenesis failure causes multi-tissue dysfunction. Saudi Arabian founder mutation c.436delC (p.Leu146Tyrfs*20) accounts for most cases. Key clinical pearl: systemic features (especially hypogonadism and diabetes) typically present BEFORE the neurological features — cardiologists, endocrinologists, and gynaecologists should be aware that hypogonadism + progressive neurological symptoms in a young patient may be WSS.",
            },
            {
                "term": "WDR45 Autophagy Mechanism — Why Iron Accumulates in BPAN",
                "definition": "WDR45 encodes WIPI4 (WD repeat domain, phosphoinositide-interacting protein 4), a member of the WIPI (WIPI1-4) family of autophagy proteins. WIPI proteins bind phosphatidylinositol-3-phosphate (PI3P) generated at the isolation membrane (phagophore) by the VPS34-Beclin1-ATG14 PI3 kinase complex. WIPI4/WDR45 function: (1) Binds PI3P at phagophores → recruits ATG2A/ATG2B (tethers phagophore to ER lipid source) + ATG18 → facilitates phagophore membrane expansion. (2) Interacts with ATG12-ATG5-ATG16L1 complex (ubiquitin-like conjugation for LC3 lipidation). Without WIPI4/WDR45: defective autophagosome formation → autophagic flux impairment → failure to clear damaged mitochondria (mitophagy failure) and protein aggregates. Iron connection: lysosomal iron storage (ferritin degradation → free iron release) is normally tightly coupled to autophagic flux; defective autophagy → dysregulated lysosomal iron handling → iron accumulation in SN/GP. Therapeutic relevance: rapamycin (mTOR inhibitor) can bypass the WDR45 defect by stimulating alternative autophagy initiation pathways → BPAN clinical trials with rapamycin/sirolimus ongoing.",
            },
            {
                "term": "ATP13A2 Lysosomal Dysfunction — Bridge between NBIA and Parkinson Disease",
                "definition": "ATP13A2 (PARK9) encodes a P5B-type lysosomal ATPase that transports Mn2+, Zn2+, and polyamines from the lysosomal lumen to the cytosol. Its role in lysosomal homeostasis: (1) regulates lysosomal pH and membrane integrity; (2) facilitates lysosomal protein degradation, including α-synuclein clearance via the autophagy-lysosomal pathway (ALP); (3) maintains lysosomal biogenesis and TFEB nuclear translocation. ATP13A2 loss → lysosomal dysfunction → α-synuclein accumulation (Lewy body-like pathology) + metal dyshomeostasis (Mn2+/Zn2+ accumulation) + mitochondrial dysfunction. This positions ATP13A2 at the intersection of NBIA (iron accumulation + neurodegeneration) and PARK (Parkinson disease genetics — α-synuclein, lysosomal pathway). Unlike sporadic PD which is adult-onset, Kufor-Rakeb (PARK9) begins in adolescence/young adulthood, L-DOPA responds initially, and includes pyramidal + gaze palsy features atypical for idiopathic PD. CLN12 (biallelic null ATP13A2): more severe lysosomal failure → NCL-like ceroid lipofuscinosis storage material on biopsy; more severe, earlier onset phenotype. Therapeutic: TFEB activation (mTOR inhibitors), lysosomal biogenesis enhancers under investigation.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== NBIA Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"Subgroups: {list(ov['gene_subgroups'].keys())}")
    print(f"Eye-of-Tiger: {ov['n_eye_of_tiger']}, Iron on MRI: {ov['n_iron_on_mri']}")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
    print("OK")
