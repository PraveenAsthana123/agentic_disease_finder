#!/usr/bin/env python3
"""Hereditary-Dystonia-Atlas — Complete 8-Gene Hereditary Dystonia Atlas.

TOR1A   (Torsin 1A; 332 aa; 9q34.11; AD;
          DYT-TOR1A (DYT1) — Early-Onset Generalized Dystonia;
          c.904_906delGAG (p.Glu302del) 90%+ of cases; penetrance 30%;
          onset <26 yr; GPi-DBS highly responsive; anticholinergics 1st line;
          seed SEED_BASE+0).
THAP1   (THAP domain containing protein 1 / THAP1; 213 aa; 8p11.21; AD;
          DYT-THAP1 (DYT6) — Mixed-Onset Primary Dystonia;
          cranial-cervical-laryngeal spread; onset 5-46 yr;
          botulinum toxin + DBS; reduced penetrance;
          seed SEED_BASE+1).
GCH1    (GTP cyclohydrolase 1; 250 aa; 14q22.2; AD;
          DYT-GCH1 (DRD/Segawa disease) — Dopa-Responsive Dystonia;
          MIRACULOUS levodopa response PATHOGNOMONIC; diurnal fluctuation;
          female predominance 3:1; phenylalanine loading test;
          seed SEED_BASE+2).
ATP1A3  (ATPase Na+/K+ alpha-3 subunit; 1013 aa; 19q13.2; AD;
          ATP1A3-related disorders — AHC/RDP/CAPOS triad;
          AHC (alternating hemiplegia of childhood); RDP (rapid-onset dystonia-parkinsonism);
          CAPOS (cerebellar-areflexia-pes cavus-optic atrophy-SNHL);
          seed SEED_BASE+3).
KMT2B   (Lysine methyltransferase 2B / MLL4; 2715 aa; 19q13.12; AD;
          DYT-KMT2B (DYT28) — Childhood-Onset Complex Dystonia;
          oculomotor abnormalities; mild ID; GPi-DBS highly responsive;
          de novo dominant; microdeletion detected only by CMA;
          seed SEED_BASE+4).
ADCY5   (Adenylate cyclase 5; 1261 aa; 3q21.3; AD;
          ADCY5-related hyperkinetic movement disorder;
          childhood onset; nocturnal dyskinesia; caffeine-sensitive;
          clonazepam + acetazolamide; facial hypotonia;
          seed SEED_BASE+5).
ANO3    (Anoctamin 3; 981 aa; 11p14.3; AD;
          DYT-ANO3 (DYT24) — Craniocervical Adult-Onset Focal Dystonia;
          onset 30-40 yr; cervical >> cranial; botulinum toxin responsive;
          DBS emerging; tremor prominent;
          seed SEED_BASE+6).
GNAL    (G protein subunit alpha L / Golf; 381 aa; 18p11.21; AD;
          DYT-GNAL (DYT25) — Primary Cranial / Spasmodic Dysphonia Dystonia;
          adult onset 40s; isolated cranial-laryngeal; botulinum toxin mainstay;
          striatal dopamine signalling; NOT DYT1;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2070-2077).
"""

import random

SEED_BASE = 2070

DYT_GENES = [
    # -- TOR1A — DYT-TOR1A (DYT1) --------------------------------------------------
    {
        "gene": "TOR1A",
        "alt_name": (
            "TOR1A (TOR1A-332aa-9q34.11 / AD — DYT-TOR1A-DYT1-Early-Onset-Generalized-Dystonia — "
            "c.904_906delGAG-p.Glu302del-90pct-Cases — Penetrance-30pct-NOT-100pct — "
            "GPi-DBS-Highly-Responsive — Anticholinergics-1st-Line — Onset-<26yr)"
        ),
        "protein": (
            "TOR1A -- 9q34.11 AD -- TOR1A-332aa -- "
            "Torsin-1A-AAA-Plus-ATPase-ER-Lumen-Nuclear-Envelope -- "
            "DYT-TOR1A-DYT1-Early-Onset-Primary-Generalised-Dystonia -- "
            "c.904_906delGAG-p.Glu302del-3bp-deletion-exon5-loss-glutamic-acid-302 -- "
            "Penetrance-30pct-NOT-fully-penetrant-modifier-genes-influence -- "
            "Autosomal-Dominant-Haploinsufficiency-Dominant-Negative -- "
            "Onset-<26yr-leg-foot-arm-CLASSIC-sequence-generalises -- "
            "GPi-DBS-Globus-Pallidus-Internus-Highly-Responsive-Axial-Sparing -- "
            "Anticholinergics-Trihexyphenidyl-1st-Line-High-Dose"
        ),
        "locus": "9q34.11",
        "protein_size": "332 aa",
        "inheritance": (
            "AD (autosomal dominant) — TOR1A haploinsufficiency / dominant-negative; "
            "Penetrance: 30% only (NOT 100%); major clinical implication — carrier ≠ affected; "
            "Ashkenazi Jewish founder: c.904_906delGAG — 1 in 2,000 AJ; "
            "Non-AJ prevalence: 1 in 10,000-20,000; "
            "90%+ of pathogenic TOR1A variants = single delGAG deletion; "
            "Siblings at risk: 50% inherit, 30% of those → affected = 15% sibling risk; "
            "Modifier: ΔGAG homozygous → higher penetrance; TOR1B modifier locus 9q34"
        ),
        "age_of_onset": (
            "Onset <26 yr (MANDATORY for DYT-TOR1A diagnosis); "
            "Classic onset 5-28 yr (peak 11-15 yr); "
            "Childhood-onset: foot/leg → arm → trunk → generalised within 5 years; "
            "Adult-onset: VERY rare in TOR1A — consider THAP1/KMT2B/other; "
            "Onset in leg during walking PATHOGNOMONIC; "
            "Task-specific onset: writer's cramp → generalises in TOR1A; "
            "Facial/pharyngeal spared in DYT1 (contrast DYT6/THAP1); "
            "After age 26: generalisation much less common; prognosis improves"
        ),
        "key_biomarker": (
            "Genetic: TOR1A sequencing — delGAG detected; "
            "MRI brain: NORMAL (essential to exclude secondary); "
            "No cerebellar signs, no parkinsonism, no myoclonus (pure dystonia); "
            "DATscan: NORMAL (distinguishes from dystonia-parkinsonism); "
            "Penetrance testing: positive genetic test does NOT predict affectedness; "
            "NMO, copper, ceruloplasmin, thyroid: exclude secondary; "
            "EEG normal (no epilepsy); EMG: overflow co-contraction pattern"
        ),
        "pathognomonic": (
            "ONSET IN LEG/FOOT UNDER AGE 26 = TOR1A until proven otherwise; "
            "DELGAG SINGLE DELETION = 90%+ of DYT1 pathogenic variants — targeted test FIRST; "
            "PENETRANCE 30% = positive test in asymptomatic sibling does NOT mean they will develop dystonia; "
            "FACE AND PHARYNX SPARED = critical DDx from DYT6/THAP1 (cranial involvement); "
            "GPi-DBS HIGHLY EFFECTIVE = 70-90% improvement with bilateral GPi; "
            "GENERALISED DYSTONIA IN CHILD/TEEN = TOR1A + KMT2B panel MANDATORY first; "
            "NORMAL MRI + NORMAL DATscan = primary dystonia confirmed"
        ),
        "treatment": (
            "1st line: Trihexyphenidyl (high-dose anticholinergic) — 6-80 mg/day; "
            "tolerated better in children (titrate slowly to minimise dry mouth/confusion); "
            "Baclofen: oral or intrathecal (ITB) for generalised; "
            "Benzodiazepines: clonazepam adjunct; "
            "Botulinum toxin A: focal bothersome muscle group — useful for writer's cramp; "
            "GPi-DBS: MOST EFFECTIVE — bilateral GPi; 70-90% improvement; "
            "DBS response predicts: young age at surgery, shorter duration, no fixed deformity; "
            "NOT levodopa-responsive (contrast DYT-GCH1); "
            "NOT tetrabenazine (worsens primary dystonia); "
            "Genetic counselling: penetrance 30% — counsel asymptomatic relatives carefully"
        ),
        "critical_flags": [
            "ONSET-LEG-FOOT-UNDER-26yr-PATHOGNOMONIC-DYT1",
            "PENETRANCE-30pct-NOT-100pct-POSITIVE-TEST-NEQV-DISEASE",
            "DELGAG-TARGETED-TEST-FIRST-90pct-CASES",
            "GPi-DBS-70-90pct-IMPROVEMENT-HIGHLY-EFFECTIVE",
            "FACE-PHARYNX-SPARED-DDx-DYT6-THAP1",
            "NOT-LEVODOPA-RESPONSIVE-CONTRAST-DRD-GCH1",
            "NORMAL-MRI-NORMAL-DATSCAN-PRIMARY-DYSTONIA",
            "ASHKENAZI-JEWISH-FOUNDER-1-IN-2000",
            "ANTICHOLINERGIC-TRIHEXYPHENIDYL-HIGH-DOSE-FIRST-LINE",
            "TETRABENAZINE-WORSENS-PRIMARY-DYSTONIA-AVOID",
        ],
    },

    # -- THAP1 — DYT-THAP1 (DYT6) --------------------------------------------------
    {
        "gene": "THAP1",
        "alt_name": (
            "THAP1 (THAP1-213aa-8p11.21 / AD — DYT-THAP1-DYT6-Mixed-Onset-Primary-Dystonia — "
            "Cranial-Cervical-Laryngeal-Spread-PATHOGNOMONIC — "
            "Botulinum-Toxin-A-1st-Line-Focal — DBS-Generalised — "
            "Onset-5-46yr-Bimodal)"
        ),
        "protein": (
            "THAP1 -- 8p11.21 AD -- THAP1-213aa -- "
            "THAP-Domain-Zinc-Finger-Transcription-Factor-E2F-Target-Gene-Repressor -- "
            "DYT-THAP1-DYT6-Mixed-Primary-Dystonia -- "
            "Cranial-Cervical-Arm-Onset-Laryngeal-Spread -- "
            "Reduced-Penetrance-60-pct-Modifier-Genes -- "
            "Autosomal-Dominant-Loss-of-Function -- "
            "Onset-5-46yr-Bimodal-Distribution -- "
            "Botulinum-Toxin-A-Focal-1st-Line -- "
            "GPi-DBS-Generalised-Forms"
        ),
        "locus": "8p11.21",
        "protein_size": "213 aa",
        "inheritance": (
            "AD (autosomal dominant) — THAP1 loss of function; "
            "Penetrance: ~60% (reduced, not as low as DYT1's 30%); "
            "Prevalence: 1 in 50,000-100,000; "
            "Bimodal onset: childhood (10-20 yr) and adult (30-50 yr); "
            "De novo variants account for ~20% of cases; "
            "Intragenic and splice-site mutations predominate; "
            "THAP domain (zinc finger, exon 1-2): most pathogenic variants cluster here; "
            "Coiled-coil domain mutations: milder phenotype"
        ),
        "age_of_onset": (
            "Bimodal: childhood (10-20 yr) and adult (30-50 yr); "
            "Overall range: 5-46 yr; "
            "Arm/hand onset: most common (writer's cramp, instrument dystonia); "
            "Cranial onset: blepharospasm, oromandibular dystonia, spasmodic dysphonia; "
            "Cervical onset: torticollis (20%); "
            "Generalisation: 50% with childhood onset generalise; "
            "Adult-onset: typically remains focal or segmental; "
            "Laryngeal involvement: 50-70% — DISTINGUISHES from DYT1; "
            "Spasmodic dysphonia: voice breaks, strained-strangled quality"
        ),
        "key_biomarker": (
            "Genetic: THAP1 sequencing — coding region + splice sites; "
            "THAP domain variants: most pathogenic; "
            "MRI brain: NORMAL; "
            "DATscan: NORMAL (no parkinsonism); "
            "CSF BH4/pterins: NORMAL (contrast DRD/GCH1); "
            "EMG: overflow, co-contraction; "
            "Levodopa trial: POOR response (contrast GCH1/DRD); "
            "Botulinum toxin response: GOOD for focal muscle groups"
        ),
        "pathognomonic": (
            "CRANIAL-CERVICAL-LARYNGEAL SPREAD = hallmark of DYT-THAP1; critical DDx from DYT1 (face/pharynx spared in DYT1); "
            "SPASMODIC DYSPHONIA (voice breaks) = 50-70% in THAP1 — uncommon in DYT1; "
            "BIMODAL ONSET = childhood generalised OR adult focal/segmental; "
            "REDUCED PENETRANCE 60% = positive test in asymptomatic relative does not guarantee disease; "
            "THAP DOMAIN VARIANTS = highest pathogenicity — cluster in exon 1-2; "
            "LEVODOPA POOR RESPONSE = distinguishes from DRD-GCH1 (levodopa MIRACULOUS in DRD); "
            "NORMAL MRI + NORMAL DATSCAN = primary dystonia, NOT parkinson-plus"
        ),
        "treatment": (
            "Focal dystonia: Botulinum toxin A (onabotulinumtoxinA / abobotulinumtoxinA); "
            "Cervical dystonia: BoNT-A 200-500 MU abobo every 12 weeks; "
            "Spasmodic dysphonia: BoNT-A laryngeal injection thyroarytenoid; "
            "Blepharospasm: BoNT-A orbicularis oculi; "
            "Generalised/segmental: Trihexyphenidyl + baclofen; "
            "GPi-DBS: effective for generalised forms (less consistent than TOR1A); "
            "STN-DBS: limited evidence; "
            "Levodopa trial: always try to exclude DRD (GCH1/SPR) — poor response expected in THAP1; "
            "Clonazepam: adjunct for anxiety-triggered worsening"
        ),
        "critical_flags": [
            "CRANIAL-CERVICAL-LARYNGEAL-SPREAD-PATHOGNOMONIC-THAP1",
            "SPASMODIC-DYSPHONIA-50-70pct-DDx-DYT1-FACE-SPARED",
            "PENETRANCE-60pct-NOT-FULLY-PENETRANT",
            "THAP-DOMAIN-EXON-1-2-HIGHEST-PATHOGENICITY",
            "LEVODOPA-POOR-RESPONSE-EXCLUDES-GCH1-DRD",
            "BOTULINUMTOXIN-A-FOCAL-1ST-LINE-CERVICAL-CRANIAL",
            "GPi-DBS-GENERALISED-FORMS",
            "NORMAL-MRI-NORMAL-DATSCAN-PRIMARY-DYSTONIA",
            "BIMODAL-ONSET-CHILDHOOD-OR-ADULT",
            "WRITER-CRAMP-ARM-ONSET-MOST-COMMON",
        ],
    },

    # -- GCH1 — DYT-GCH1 (DRD / Segawa) -------------------------------------------
    {
        "gene": "GCH1",
        "alt_name": (
            "GCH1 (GCH1-250aa-14q22.2 / AD — DYT-GCH1-DRD-Segawa-Disease — "
            "Levodopa-MIRACULOUS-RESPONSE-PATHOGNOMONIC — "
            "Diurnal-Fluctuation-Worse-Evening-PATHOGNOMONIC — "
            "Phenylalanine-Loading-Test — Female-Predominance-3:1 — "
            "Low-Dose-Levodopa-Life-Long)"
        ),
        "protein": (
            "GCH1 -- 14q22.2 AD -- GCH1-250aa -- "
            "GTP-Cyclohydrolase-1-Rate-Limiting-Enzyme-Tetrahydrobiopterin-BH4-Synthesis -- "
            "DYT-GCH1-Dopa-Responsive-Dystonia-DRD-Segawa-Disease -- "
            "BH4-Cofactor-for-Tyrosine-Hydroxylase-TH-Dopamine-Synthesis -- "
            "BH4-Deficiency-→-TH-Dysfunction-→-Striatal-Dopamine-Deficiency -- "
            "Autosomal-Dominant-Haploinsufficiency-BH4 -- "
            "Diurnal-Fluctuation-Worse-Evening-Better-Morning-Sleep -- "
            "Levodopa-MIRACULOUS-Dramatic-Response-Even-Low-Doses -- "
            "Female-Predominance-3:1-Hormonal-Oestrogen-BH4-Interaction"
        ),
        "locus": "14q22.2",
        "protein_size": "250 aa",
        "inheritance": (
            "AD (autosomal dominant) — GCH1 haploinsufficiency; "
            "Penetrance: female 87%; male 38% (sex-specific penetrance — unique feature); "
            "Prevalence: 1 in 2,000,000 (underdiagnosed due to diagnostic delay); "
            "Female predominance: 3:1 (oestrogen reduces residual GCH1 activity); "
            "AR biallelic GCH1: severe hyperphenylalaninaemia (HPA) + neurological crisis (different phenotype); "
            "Allelic: SPR (sepiapterin reductase) mutations → AR DRD with cerebrospinal biogenic amine deficiency; "
            "Median diagnostic delay: 10-15 yr (misdiagnosed as CP/spastic diplegia/functional)"
        ),
        "age_of_onset": (
            "Onset: 1-20 yr (peak 5-10 yr); "
            "Foot dystonia with equinus gait: CLASSIC presentation (misdiagnosed as spastic diplegia/CP); "
            "Morning normal — worsening through day — evening severe: DIURNAL FLUCTUATION PATHOGNOMONIC; "
            "After sleep: marked improvement (hours); "
            "Older age: may present with parkinsonism (without dystonia initially) — 'Parkinsonism with diurnal fluctuation'; "
            "Leg dystonia → generalised if untreated; "
            "Writer's cramp: adult presentation; "
            "Female: earlier onset, more severe if untreated"
        ),
        "key_biomarker": (
            "CSF neurotransmitter: LOW biopterin + LOW neopterin + LOW HVA (dopamine metabolite); "
            "Phenylalanine loading test: exaggerated rise in phenylalanine after oral load (GCH1 haploinsufficiency); "
            "Urine pterins: abnormal biopterin:neopterin ratio; "
            "Genetic: GCH1 sequencing — missense/nonsense/splice mutations; "
            "Levodopa therapeutic trial: MIRACULOUS response (1-3 mg/kg/day) = diagnostic; "
            "DATscan: NORMAL (striatal dopamine transporter NOT lost — contrast Parkinson's); "
            "Serum prolactin: elevated (dopamine deficiency disinhibits PRL); "
            "MRI: NORMAL; "
            "DO NOT MISS: levodopa trial in ALL childhood-onset dystonia with diurnal fluctuation"
        ),
        "pathognomonic": (
            "DIURNAL FLUCTUATION (worse evening, better morning after sleep) = DRD PATHOGNOMONIC — no other dystonia; "
            "LEVODOPA MIRACULOUS DRAMATIC RESPONSE = low dose 1-3 mg/kg/day → abolishes dystonia; "
            "CHILDHOOD LEG DYSTONIA MISDIAGNOSED AS CP/SPASTIC DIPLEGIA = DRD until levodopa trial proves otherwise; "
            "DATSCAN NORMAL = contrast Parkinson's (DATscan ABNORMAL in PD — GCH1/DRD DATscan NORMAL); "
            "FEMALE PREDOMINANCE 3:1 = oestrogen reduces residual GCH1; sex-specific penetrance; "
            "PHENYLALANINE LOADING TEST = GCH1 pathway test; abnormal BH4 metabolism; "
            "DO NOT MISS DRD = levodopa is curative and lifelong treatment is required"
        ),
        "treatment": (
            "Levodopa/carbidopa: FIRST LINE — miraculous response; "
            "Dose: start 1-2 mg/kg/day levodopa (with carbidopa 1:4 ratio); "
            "Typical adult dose: 100-300 mg levodopa/day in 3 divided doses; "
            "Life-long therapy required — never stop; "
            "NO wearing off / NO dyskinesia at therapeutic DRD doses (contrast Parkinson's); "
            "BH4 (sapropterin): alternative/adjunct especially in AR forms; "
            "DO NOT use botulinum toxin as primary therapy (misses diagnosis); "
            "DO NOT use DBS for DRD (levodopa restores function); "
            "Genetic counselling: 50% risk each child; sex-specific penetrance counselling; "
            "Folic acid: supplementation during pregnancy (BH4 pathway)"
        ),
        "critical_flags": [
            "LEVODOPA-MIRACULOUS-RESPONSE-PATHOGNOMONIC-DRD",
            "DIURNAL-FLUCTUATION-WORSE-EVENING-BETTER-MORNING-PATHOGNOMONIC",
            "CHILDHOOD-FOOT-DYSTONIA-CP-MISDIAGNOSIS-DRD-LEVODOPA-TRIAL-MANDATORY",
            "DATSCAN-NORMAL-CONTRAST-PARKINSONS-ABNORMAL",
            "FEMALE-PREDOMINANCE-3:1-SEX-SPECIFIC-PENETRANCE",
            "PHENYLALANINE-LOADING-TEST-GCH1-PATHWAY",
            "LIFELONG-LEVODOPA-NEVER-STOP",
            "NO-WEARING-OFF-NO-DYSKINESIA-AT-DRD-DOSES",
            "CSF-LOW-BIOPTERIN-NEOPTERIN-HVA",
            "DO-NOT-DBS-FOR-DRD-LEVODOPA-CURATIVE",
        ],
    },

    # -- ATP1A3 — AHC / RDP / CAPOS ------------------------------------------------
    {
        "gene": "ATP1A3",
        "alt_name": (
            "ATP1A3 (ATP1A3-1013aa-19q13.2 / AD — ATP1A3-Related-Neurological-Spectrum — "
            "AHC-Alternating-Hemiplegia-of-Childhood-Flunarizine-1st-Line — "
            "RDP-Rapid-Onset-Dystonia-Parkinsonism-Rostrocaudal-Gradient-PATHOGNOMONIC — "
            "CAPOS-Cerebellar-Areflexia-Pes-Cavus-Optic-Atrophy-SNHL — "
            "AVOID-Triggers-Fever-Emotional-Stress)"
        ),
        "protein": (
            "ATP1A3 -- 19q13.2 AD -- ATP1A3-1013aa -- "
            "ATPase-Na+/K+-Transporting-Alpha-3-Subunit-Neuron-Specific-Isoform -- "
            "Na+/K+-ATPase-α3-Electrogenics-Neuronal-Membrane-Potential -- "
            "Autosomal-Dominant-Gain-of-Function-Partial-Loss-of-Function -- "
            "Three-Clinical-Syndromes-AHC-RDP-CAPOS-Genotype-Phenotype -- "
            "AHC-pE815K-Alternating-Hemiplegia-Episodic-Paroxysmal -- "
            "RDP-pD801N-Rapid-Onset-Dystonia-Parkinsonism-Rostrocaudal-Spread -- "
            "CAPOS-pE818K-Febrile-Onset-Cerebellar-Optic-Auditory"
        ),
        "locus": "19q13.2",
        "protein_size": "1013 aa",
        "inheritance": (
            "AD (autosomal dominant) — gain/loss of function; "
            "Mostly de novo (AHC/RDP/CAPOS all predominantly de novo); "
            "Rare familial cases with variable expressivity; "
            "Genotype-phenotype correlations: "
            "AHC: p.E815K (most common), p.T672A, p.G947R; "
            "RDP: p.D801N (distinctive); "
            "CAPOS: p.E818K (highly specific — single variant causes CAPOS); "
            "Prevalence AHC: 1 in 1,000,000; "
            "No parental mosaicism testing required (de novo confirmed usually)"
        ),
        "age_of_onset": (
            "AHC: onset <18 months (DIAGNOSTIC CRITERION); "
            "AHC: hemiplegic episodes from age 3-18 months; "
            "AHC: nystagmus at birth/neonatal — earliest sign; "
            "AHC: episodes triggered by fever, emotional stress, water; "
            "AHC: bilateral hemiplegia = respiratory compromise EMERGENCY; "
            "RDP: onset 4-55 yr (most adolescent/young adult); "
            "RDP: ABRUPT onset within hours-4 weeks; NEVER gradual; "
            "CAPOS: onset during febrile illness age 1-6 yr"
        ),
        "key_biomarker": (
            "Genetic: ATP1A3 sequencing — targeted variant panels; "
            "AHC: clinical diagnosis (criteria: onset <18 months, alternating hemiplegia, "
            "normal MRI between episodes, improvement with sleep); "
            "EEG: ictal EEG during hemiplegia — NOT epileptiform (distinguishes from seizure); "
            "MRI: NORMAL between episodes (AHC/RDP); "
            "CSF: NORMAL (no inflammatory); "
            "CAPOS: VEP (visual evoked potential) abnormal; ABR (auditory brainstem response) abnormal; "
            "Ophthalmology: optic atrophy (CAPOS); "
            "Nerve conduction: ABSENT reflexes (CAPOS areflexia); "
            "DATscan: NORMAL (RDP — contrast Parkinson's; DAT intact)"
        ),
        "pathognomonic": (
            "ALTERNATING HEMIPLEGIA <18 MONTHS = ATP1A3-AHC until proven otherwise; "
            "BILATERAL HEMIPLEGIA = respiratory compromise EMERGENCY — CAREGIVER TRAINING MANDATORY; "
            "EPISODES RESOLVE WITH SLEEP = AHC PATHOGNOMONIC (hemiplegia disappears after sleep); "
            "NYSTAGMUS IN NEONATAL PERIOD = earliest ATP1A3-AHC sign; "
            "RAPID-ONSET DYSTONIA-PARKINSONISM ROSTROCAUDAL GRADIENT = RDP PATHOGNOMONIC (cranial > arm > leg); "
            "RDP ABRUPT ONSET HOURS-WEEKS THEN PLATEAU = NOT progressive (contrasts PD); "
            "CAPOS p.E818K = single variant causing entire syndrome; "
            "FEVER TRIGGERS AHC EPISODES = avoid antipyretics delay (treat fever aggressively); "
            "DATSCAN NORMAL IN RDP = contrasts Parkinson's (DAT ABNORMAL in PD)"
        ),
        "treatment": (
            "AHC: Flunarizine (calcium channel blocker) — reduces frequency/severity; "
            "AHC: 5-10 mg/day flunarizine; most effective preventive; "
            "AHC: Benzodiazepines (clonazepam/diazepam) PRN during episodes; "
            "AHC: Sleep induction aborts episodes (carry sleeping medication); "
            "AHC: AVOID triggers — fever (treat early), emotional stress, water; "
            "RDP: No effective treatment for core dystonia-parkinsonism (no levodopa response); "
            "RDP: Botulinum toxin for focal dystonia components; "
            "RDP: DBS limited evidence; "
            "CAPOS: Supportive — hearing aids (SNHL), low-vision support (optic atrophy); "
            "All ATP1A3: genetic counselling (de novo — sibling risk very low)"
        ),
        "critical_flags": [
            "AHC-ALTERNATING-HEMIPLEGIA-<18MONTHS-PATHOGNOMONIC",
            "BILATERAL-HEMIPLEGIA-RESPIRATORY-EMERGENCY-AHC",
            "EPISODES-RESOLVE-WITH-SLEEP-PATHOGNOMONIC-AHC",
            "NYSTAGMUS-NEONATAL-EARLIEST-AHC-SIGN",
            "RDP-ABRUPT-ONSET-HOURS-WEEKS-ROSTROCAUDAL-GRADIENT",
            "FEVER-TRIGGERS-AHC-TREAT-EARLY-AGGRESSIVELY",
            "FLUNARIZINE-FIRST-LINE-AHC-PREVENTION",
            "CAPOS-p.E818K-SINGLE-VARIANT-CEREBELLAR-SNHL-OPTIC",
            "DATSCAN-NORMAL-RDP-CONTRASTS-PARKINSONS",
            "NO-LEVODOPA-RESPONSE-RDP-CONTRAST-GCH1-DRD",
        ],
    },

    # -- KMT2B — DYT-KMT2B (DYT28) ------------------------------------------------
    {
        "gene": "KMT2B",
        "alt_name": (
            "KMT2B (KMT2B-2715aa-19q13.12 / AD — DYT-KMT2B-DYT28-Childhood-Complex-Dystonia — "
            "Oculomotor-Abnormalities-PATHOGNOMONIC-DDx-DYT1 — "
            "Mild-ID-Facial-Dysmorphism — "
            "GPi-DBS-HIGHLY-RESPONSIVE — "
            "De-Novo-Dominant-Microdeletion-CMA-Mandatory)"
        ),
        "protein": (
            "KMT2B -- 19q13.12 AD -- KMT2B-2715aa -- "
            "Lysine-Methyltransferase-2B-MLL4-Histone-H3K4-Methyltransferase -- "
            "DYT-KMT2B-DYT28-Childhood-Onset-Complex-Dystonia -- "
            "De-Novo-Dominant-Mostly-Truncating-Missense -- "
            "Oculomotor-Abnormalities-Supranuclear-Gaze-Palsy-Nystagmus -- "
            "Mild-Intellectual-Disability-Facial-Dysmorphism-Short-Stature -- "
            "GPi-DBS-Bilateral-Highly-Effective-Earlier-Is-Better -- "
            "Microdeletion-19q13.12-CMA-Required-Sequencing-May-Miss"
        ),
        "locus": "19q13.12",
        "protein_size": "2715 aa",
        "inheritance": (
            "AD (autosomal dominant) — mostly de novo; "
            "Loss of function: truncating variants (nonsense/frameshift/splice) predominate; "
            "Microdeletions of 19q13.12: up to 30% — sequencing ALONE misses these; "
            "CMA (chromosomal microarray) MANDATORY alongside sequencing; "
            "Rare familial cases with variable expressivity; "
            "No parental mosaicism data; "
            "Prevalence: rare, exact figure unknown; "
            "Constitutes ~3-6% of children with unexplained generalised dystonia"
        ),
        "age_of_onset": (
            "Onset: 1-12 yr (childhood mandatory for classical DYT28); "
            "Lower limbs first — then rapid generalisation; "
            "Oculomotor abnormalities early (supranuclear gaze palsy, nystagmus); "
            "Facial dystonia + oromandibular: distinguish from DYT1 (face spared in DYT1); "
            "Mild ID/developmental delay: present in most; "
            "Facial dysmorphism: wide forehead, broad nasal bridge; "
            "Short stature: 30-50%; "
            "Rapid progressive course without treatment; "
            "After GPi-DBS: improvement within weeks"
        ),
        "key_biomarker": (
            "Genetic: KMT2B sequencing + CMA for 19q13.12 microdeletion (both required); "
            "MRI brain: NORMAL (no structural lesion — primary dystonia); "
            "Ophthalmology: supranuclear gaze palsy, nystagmus assessment; "
            "Neuropsychology: mild ID documented; "
            "DATscan: NORMAL; "
            "CSF: NORMAL (no neurotransmitter deficiency); "
            "Levodopa trial: POOR response (contrast GCH1/DRD); "
            "EMG: overflow co-contraction; "
            "EEG: NORMAL (no epilepsy in typical DYT28)"
        ),
        "pathognomonic": (
            "CHILDHOOD GENERALISED DYSTONIA + OCULOMOTOR ABNORMALITIES = KMT2B until proven otherwise; "
            "OCULOMOTOR SIGNS (supranuclear gaze palsy, nystagmus) = critical DDx from DYT1 (no oculomotor in DYT1); "
            "MILD ID + FACIAL DYSMORPHISM + GENERALISED DYSTONIA = DYT28/KMT2B phenotype; "
            "MICRODELETION NOT DETECTED BY SEQUENCING ALONE = CMA MANDATORY alongside sequence; "
            "GPi-DBS HIGHLY EFFECTIVE = respond even better than DYT1 in some series; "
            "EARLIER DBS = BETTER OUTCOME — do not delay; "
            "LEVODOPA POOR RESPONSE = distinguishes from DRD/GCH1"
        ),
        "treatment": (
            "GPi-DBS: MOST EFFECTIVE — bilateral GPi; earlier surgery = better outcome; "
            "Do not delay DBS waiting for age threshold if disease is severe; "
            "Trihexyphenidyl: high-dose anticholinergic before DBS; "
            "Baclofen: oral or intrathecal adjunct; "
            "Botulinum toxin: focal/segmental burden reduction pre-DBS; "
            "Levodopa trial: perform to exclude DRD (poor response expected in KMT2B); "
            "Tetrabenazine: AVOID (worsens primary dystonia); "
            "Multidisciplinary: physiotherapy, OT, speech (oromandibular dystonia); "
            "Genetic counselling: de novo — sibling risk very low; "
            "Educational support for mild ID"
        ),
        "critical_flags": [
            "OCULOMOTOR-ABNORMALITIES-PATHOGNOMONIC-DDx-DYT1-NO-OCULOMOTOR",
            "CMA-MANDATORY-19q13.12-MICRODELETION-SEQUENCING-MISSES-30pct",
            "GPi-DBS-HIGHLY-EFFECTIVE-EARLIER-IS-BETTER",
            "MILD-ID-FACIAL-DYSMORPHISM-DISTINGUISH-DYT1",
            "DE-NOVO-DOMINANT-MOSTLY-TRUNCATING",
            "LEVODOPA-POOR-RESPONSE-CONTRAST-GCH1-DRD",
            "CHILDHOOD-ONSET-<12yr-GENERALISED-RAPID",
            "TETRABENAZINE-AVOID-WORSENS-PRIMARY-DYSTONIA",
            "NORMAL-MRI-PRIMARY-DYSTONIA-KMT2B",
            "FACIAL-OROMANDIBULAR-DYSTONIA-DISTINGUISH-DYT1",
        ],
    },

    # -- ADCY5 — ADCY5-Related Hyperkinetic Movement Disorder -----------------------
    {
        "gene": "ADCY5",
        "alt_name": (
            "ADCY5 (ADCY5-1261aa-3q21.3 / AD — ADCY5-Related-Hyperkinetic-Movement-Disorder — "
            "Nocturnal-Dyskinesia-Worsening-PATHOGNOMONIC — "
            "Facial-Hypotonia-Chorea-Dystonia-Myoclonus-Triad — "
            "Caffeine-ABSOLUTELY-CI — "
            "Clonazepam-Acetazolamide-1st-Line)"
        ),
        "protein": (
            "ADCY5 -- 3q21.3 AD -- ADCY5-1261aa -- "
            "Adenylate-Cyclase-5-Striatal-cAMP-Production-Dopamine-Receptor-Signal -- "
            "ADCY5-Related-Hyperkinetic-Movement-Disorder-ADCY5-RMD -- "
            "Gain-of-Function-Increased-cAMP-Striatal-Dysfunction -- "
            "Autosomal-Dominant-De-Novo-Variable-Expressivity -- "
            "Facial-Hypotonia-Prominent-Neonatal-Hypotonia -- "
            "Nocturnal-Worsening-Dyskinesia-Sleep-Disruption -- "
            "Chorea-Dystonia-Myoclonus-Mixed-Hyperkinetic-Disorder -- "
            "Caffeine-ABSOLUTELY-CI-Adenosine-Pathway-Interaction"
        ),
        "locus": "3q21.3",
        "protein_size": "1261 aa",
        "inheritance": (
            "AD (autosomal dominant) — gain of function; "
            "Mostly de novo; rare familial cases; "
            "Mosaicism: parental mosaicism documented — siblings may have lower-level mosaic variants; "
            "Gain-of-function mutation → increased striatal cAMP → dysregulated basal ganglia signalling; "
            "Two recurrent mutations: p.R418W, p.A726T (hotspot variants >50% of cases); "
            "Prevalence: rare, exact figure unknown; "
            "Variable expressivity: mild chorea → severe generalised mixed hyperkinetic disorder"
        ),
        "age_of_onset": (
            "Neonatal/infantile onset: hypotonia prominent first sign; "
            "Movement disorder: emerges 1-3 yr; "
            "Facial hypotonia: drooping, floppy facies — persists; "
            "Hyperkinetic movements: chorea, dystonia, myoclonus — all present (MIXED); "
            "NOCTURNAL WORSENING: episodes during non-REM sleep; violent limb movements; "
            "Daytime: variable; worsens with excitement/emotion; "
            "Respiratory dysfunction: nocturnal episodes may affect breathing; "
            "Course: non-progressive; may improve with age in some; "
            "Cognition: usually normal or borderline"
        ),
        "key_biomarker": (
            "Genetic: ADCY5 sequencing — target p.R418W, p.A726T first (recurrent hotspots); "
            "Parental mosaicism testing (blood AND saliva): if proband is de novo — test parents; "
            "MRI brain: NORMAL (or mild signal change — non-specific); "
            "Nocturnal video-EEG: movement episodes NOT epileptiform — EEG NORMAL during episodes; "
            "DATscan: NORMAL; "
            "Caffeine challenge: CONTRAINDICATED (may trigger severe episodes); "
            "Levodopa trial: POOR response (contrast DRD); "
            "Sleep study (polysomnography): nocturnal NREM episodes documented; "
            "EMG: irregular myoclonic bursts + choreoathetoid activity"
        ),
        "pathognomonic": (
            "NOCTURNAL DYSKINESIA WORSENING DURING SLEEP = ADCY5-RMD PATHOGNOMONIC; episodes during NREM sleep; "
            "FACIAL HYPOTONIA (drooping face) + CHILDHOOD HYPERKINETIC DISORDER = ADCY5 suspect; "
            "MIXED CHOREA + DYSTONIA + MYOCLONUS = triad typical of ADCY5 (other dystonia genes: usually pure dystonia); "
            "CAFFEINE ABSOLUTELY CONTRAINDICATED = adenosine A1 blockade by caffeine → worsens cAMP pathway GOF; "
            "EEG NORMAL DURING NOCTURNAL EPISODES = NOT epilepsy (critical DDx nocturnal frontal lobe epilepsy); "
            "p.R418W OR p.A726T = recurrent hotspot — targeted panel FIRST; "
            "DATSCAN NORMAL = contrasts parkinsonism (no presynaptic dopamine loss)"
        ),
        "treatment": (
            "Clonazepam: FIRST LINE — reduces frequency/severity of nocturnal dyskinesia; "
            "Acetazolamide: second line — carbonic anhydrase inhibitor, reduces cAMP; "
            "Combined clonazepam + acetazolamide: often effective; "
            "CAFFEINE ABSOLUTELY CONTRAINDICATED — educate patient/family; "
            "Tetrabenazine: some response for chorea component; "
            "Benzodiazepines: PRN for severe episodes; "
            "DBS: limited evidence in ADCY5-RMD — variable response; "
            "Physiotherapy: prevent contractures from dystonic posturing; "
            "Sleep hygiene: regular sleep schedule; "
            "Genetic counselling: mosaicism risk — sequence parents blood AND saliva"
        ),
        "critical_flags": [
            "NOCTURNAL-DYSKINESIA-NREM-SLEEP-PATHOGNOMONIC-ADCY5",
            "CAFFEINE-ABSOLUTELY-CONTRAINDICATED-cAMP-GOF",
            "FACIAL-HYPOTONIA-MIXED-CHOREA-DYSTONIA-MYOCLONUS-TRIAD",
            "EEG-NORMAL-DURING-EPISODES-NOT-EPILEPSY-DDx-NFLE",
            "p.R418W-p.A726T-RECURRENT-HOTSPOTS-50pct",
            "PARENTAL-MOSAICISM-TEST-BLOOD-AND-SALIVA",
            "CLONAZEPAM-ACETAZOLAMIDE-FIRST-LINE",
            "DATSCAN-NORMAL-NOT-PARKINSONISM",
            "LEVODOPA-POOR-RESPONSE-CONTRAST-DRD-GCH1",
            "TETRABENAZINE-PARTIAL-RESPONSE-CHOREA-COMPONENT",
        ],
    },

    # -- ANO3 — DYT-ANO3 (DYT24) ---------------------------------------------------
    {
        "gene": "ANO3",
        "alt_name": (
            "ANO3 (ANO3-981aa-11p14.3 / AD — DYT-ANO3-DYT24-Craniocervical-Adult-Focal-Dystonia — "
            "Cervical-Dystonia-Torticollis-Most-Common — "
            "Tremor-Prominent-Distinguish-ETor-CD — "
            "Botulinum-Toxin-A-1st-Line — "
            "DBS-Emerging-Generalised)"
        ),
        "protein": (
            "ANO3 -- 11p14.3 AD -- ANO3-981aa -- "
            "Anoctamin-3-Calcium-Activated-Chloride-Channel-TMEM16-Family -- "
            "DYT-ANO3-DYT24-Adult-Onset-Focal-Craniocervical-Dystonia -- "
            "Autosomal-Dominant-Reduced-Penetrance -- "
            "Cervical-Dystonia-Torticollis-Primary-Presentation -- "
            "Cranial-Dystonia-Blepharospasm-Oromandibular-Spread -- "
            "Tremor-Prominent-Often-Misdiagnosed-Essential-Tremor -- "
            "Botulinum-Toxin-A-Cervical-Cranial-Primary-Treatment"
        ),
        "locus": "11p14.3",
        "protein_size": "981 aa",
        "inheritance": (
            "AD (autosomal dominant) — loss of function; "
            "Penetrance: reduced (~60%); "
            "Founder effect: Northern European (Danish/British); "
            "ANO3 accounts for approximately 7-15% of familial adult-onset focal dystonia; "
            "Rare de novo cases reported; "
            "Expressivity: cervical dystonia most common; blepharospasm, oromandibular rare; "
            "Non-penetrant carriers: identified in genetic studies"
        ),
        "age_of_onset": (
            "Adult onset: 30-60 yr (peak 35-50 yr); "
            "Childhood onset: RARE — if present, consider other genes; "
            "Cervical dystonia: most common (torticollis/laterocollis/retrocollis); "
            "Tremor: head/neck tremor prominent — often precedes or accompanies dystonia; "
            "Cranial spread: blepharospasm (10%), oromandibular (5%) over time; "
            "Arm involvement: focal arm dystonia (15%); "
            "Generalisation: RARE in ANO3 (mostly stays focal/segmental); "
            "Course: slowly progressive; often plateaus after years"
        ),
        "key_biomarker": (
            "Genetic: ANO3 sequencing — look for missense variants in transmembrane domains; "
            "MRI brain: NORMAL (primary focal dystonia); "
            "DATscan: NORMAL (no parkinsonism); "
            "Sensory trick (geste antagoniste): PRESENT in cervical dystonia (touch chin area relieves spasm); "
            "EMG: antagonist co-contraction pattern in sternocleidomastoid; "
            "Botulinum toxin response: GOOD for cervical dystonia; "
            "Levodopa trial: POOR (not DRD); "
            "Family history: helpful — adult focal dystonia in multiple relatives suggests ANO3"
        ),
        "pathognomonic": (
            "ADULT-ONSET CERVICAL DYSTONIA + TREMOR + FAMILY HISTORY = ANO3 highly suspect; "
            "TREMOR PROMINENT IN CERVICAL DYSTONIA = distinguishes from pure torticollis — may resemble ET; "
            "SENSORY TRICK (geste antagoniste) = specific to cervical/focal dystonia (absent in Parkinson's); "
            "FOCAL CRANIOCERVICAL DISTRIBUTION = ANO3 rarely generalises (contrast DYT1/KMT2B childhood); "
            "FAMILY HISTORY OF ADULT FOCAL DYSTONIA = autosomal dominant reduced penetrance pedigree; "
            "BOTULINUM TOXIN RESPONSE = excellent for cervical/cranial component; "
            "DATSCAN NORMAL = not Parkinson's (cervical rigidity DDx PD — DATscan distinguishes)"
        ),
        "treatment": (
            "Botulinum toxin A (cervical dystonia): MAINSTAY — onabotulinumtoxinA 150-300 U or abobotulinumtoxinA 500-1000 MU; "
            "Injection every 12 weeks; sternocleidomastoid + splenius capitis targeted; "
            "Blepharospasm: BoNT-A orbicularis oculi; "
            "Clonazepam/baclofen: oral adjunct for pain/anxiety component; "
            "Trihexyphenidyl: some benefit for tremor component; "
            "Physiotherapy: head/neck stretching, sensory trick training; "
            "GPi-DBS: emerging evidence for medically refractory cervical dystonia; "
            "Levodopa: POOR response (not DRD); "
            "Genetic counselling: reduced penetrance — counselling about asymptomatic relatives"
        ),
        "critical_flags": [
            "ADULT-ONSET-CERVICAL-DYSTONIA-TREMOR-FAMILY-HX-ANO3",
            "SENSORY-TRICK-GESTE-ANTAGONISTE-FOCAL-DYSTONIA",
            "TREMOR-PROMINENT-MISDIAGNOSED-ESSENTIAL-TREMOR",
            "BOTULINUMTOXIN-A-MAINSTAY-CERVICAL-CRANIAL",
            "FOCAL-RARELY-GENERALISES-CONTRAST-DYT1-KMT2B",
            "DATSCAN-NORMAL-NOT-PARKINSONS-DDx-CERVICAL-PD",
            "REDUCED-PENETRANCE-60pct-FAMILY-COUNSELLING",
            "NORTHERN-EUROPEAN-FOUNDER-DANISH-BRITISH",
            "LEVODOPA-POOR-RESPONSE-NOT-DRD",
            "GPi-DBS-EMERGING-REFRACTORY-CERVICAL",
        ],
    },

    # -- GNAL — DYT-GNAL (DYT25) ---------------------------------------------------
    {
        "gene": "GNAL",
        "alt_name": (
            "GNAL (GNAL-381aa-18p11.21 / AD — DYT-GNAL-DYT25-Primary-Cranial-Spasmodic-Dysphonia — "
            "Isolated-Cranial-Laryngeal-Dystonia — "
            "Striatal-Golf-Dopamine-Adenylyl-Cyclase-Pathway — "
            "Botulinum-Toxin-A-Laryngeal-Mainstay — "
            "NOT-DYT1-Limbs-Spared)"
        ),
        "protein": (
            "GNAL -- 18p11.21 AD -- GNAL-381aa -- "
            "G-Protein-Subunit-Alpha-L-Golf-Striatal-G-Protein-D1-Receptor-Coupling -- "
            "DYT-GNAL-DYT25-Primary-Isolated-Cranial-Dystonia -- "
            "Striatal-cAMP-Signalling-Golf-Adenylyl-Cyclase-Pathway -- "
            "Autosomal-Dominant-Loss-of-Function-Haploinsufficiency -- "
            "Spasmodic-Dysphonia-Abductor-Adductor-Laryngeal-Dystonia -- "
            "Blepharospasm-Oromandibular-Spread-Cranial-Segmental -- "
            "Botulinum-Toxin-A-Laryngeal-Injection-Mainstay-Treatment"
        ),
        "locus": "18p11.21",
        "protein_size": "381 aa",
        "inheritance": (
            "AD (autosomal dominant) — GNAL haploinsufficiency; "
            "Reduced penetrance (~60%); "
            "Variable expressivity: spasmodic dysphonia, blepharospasm, craniocervical dystonia; "
            "GNAL accounts for ~7% of isolated focal cranial dystonia families; "
            "Predominantly female-predominant presentation; "
            "Rare familial cases with variable cranial phenotypes"
        ),
        "age_of_onset": (
            "Adult onset: 30-60 yr (peak 40-55 yr); "
            "Spasmodic dysphonia: strained-strangled voice (adductor) or breathy (abductor); "
            "Blepharospasm: involuntary eye closure, photophobia; "
            "Oromandibular: jaw opening/closing deviation; "
            "Cervical dystonia: minority; "
            "Limbs: SPARED in DYT-GNAL (contrast DYT1/DYT6); "
            "Generalisation: RARE; stays cranial/laryngeal; "
            "Course: slowly progressive; responsive to BoNT-A treatment"
        ),
        "key_biomarker": (
            "Genetic: GNAL sequencing — loss of function variants; "
            "Laryngoscopy: adductor/abductor laryngeal spasm (dystonic adduction during phonation); "
            "MRI brain: NORMAL; "
            "DATscan: NORMAL; "
            "Speech pathology assessment: voice quality, vocal breaks; "
            "Botulinum toxin injection (laryngeal) response: GOOD; "
            "Levodopa: POOR response; "
            "CSF: NORMAL; "
            "Acoustic analysis: strained-strangled pattern (adductor SD) or aphonic breaks (abductor SD)"
        ),
        "pathognomonic": (
            "SPASMODIC DYSPHONIA (strained-strangled or aphonic voice breaks) = GNAL suspect if familial adult-onset; "
            "ISOLATED CRANIAL-LARYNGEAL DISTRIBUTION = DYT-GNAL characteristic; LIMBS SPARED (contrast DYT1: limbs first); "
            "GOLF (Gαolf) = striatal G-protein coupling D1-receptor to adenylyl cyclase; "
            "GNAL LOSS-OF-FUNCTION → REDUCED cAMP → STRIATAL DOPAMINE PATHWAY DYSFUNCTION; "
            "ADULT FEMALE PREDOMINANT CRANIAL DYSTONIA = GNAL in differential; "
            "BOTULINUM TOXIN LARYNGEAL INJECTION = mainstay — thyroarytenoid injection for adductor SD; "
            "DATSCAN NORMAL = not PD/parkinson-plus; striatal dopamine transporter intact"
        ),
        "treatment": (
            "Spasmodic dysphonia: BoNT-A laryngeal injection — thyroarytenoid (adductor SD); "
            "Posterior cricoarytenoid injection (abductor SD); "
            "Dose: onabotulinumtoxinA 2.5-5 U bilaterally thyroarytenoid; "
            "Frequency: every 3-4 months; sustained benefit; "
            "Blepharospasm: BoNT-A orbicularis oculi 5-10 U per site; "
            "Oromandibular: BoNT-A masseter/pterygoid (jaw closing) or mylohyoid (jaw opening); "
            "Trihexyphenidyl: adjunct systemic benefit limited; "
            "DBS: STN/GPi — limited evidence for isolated cranial/laryngeal dystonia; "
            "Voice therapy: adjunct (not curative); "
            "Levodopa: POOR response"
        ),
        "critical_flags": [
            "SPASMODIC-DYSPHONIA-STRAINED-STRANGLED-OR-APHONIC-GNAL",
            "ISOLATED-CRANIAL-LARYNGEAL-LIMBS-SPARED-CONTRAST-DYT1",
            "BOTULINUMTOXIN-A-LARYNGEAL-INJECTION-MAINSTAY-THYROARYTENOID",
            "GNAL-GOLF-D1-RECEPTOR-STRIATAL-cAMP-PATHWAY",
            "REDUCED-PENETRANCE-60pct-ADULT-FEMALE-PREDOMINANT",
            "DATSCAN-NORMAL-NOT-PARKINSONS",
            "LEVODOPA-POOR-RESPONSE-NOT-DRD-GCH1",
            "ADULT-ONSET-40s-CRANIAL-DISTRIBUTION",
            "BLEPHAROSPASM-OROMANDIBULAR-SPREAD-SEGMENTAL",
            "BOTULINUM-TOXIN-3-4-MONTHS-REPEAT-EVERY-CYCLE",
        ],
    },
]


def _generate_cohort():
    """Generate 320-patient aggregate (8 × 40 patients, seeds 2070-2077)."""
    all_patients = []
    for idx, gene_data in enumerate(DYT_GENES):
        gene = gene_data["gene"]
        rng = random.Random(SEED_BASE + idx)
        for i in range(40):
            age = rng.randint(3, 72)
            if gene == "TOR1A":
                onset_age = rng.randint(5, 25)
                generalised = rng.random() < 0.70
                gpi_dbs = rng.random() < 0.45
                delgag = rng.random() < 0.92
                anticholinergic = rng.random() < 0.75
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "onset_age": onset_age,
                    "generalised_dystonia": generalised,
                    "gpi_dbs": gpi_dbs,
                    "delgag_variant": delgag,
                    "anticholinergic_use": anticholinergic,
                })
            elif gene == "THAP1":
                onset_age = rng.randint(5, 46)
                laryngeal = rng.random() < 0.55
                cervical = rng.random() < 0.45
                generalised = onset_age < 20 and rng.random() < 0.50
                bont_use = rng.random() < 0.70
                dbs = generalised and rng.random() < 0.35
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "onset_age": onset_age,
                    "laryngeal_involvement": laryngeal,
                    "cervical_involvement": cervical,
                    "generalised_dystonia": generalised,
                    "botulinum_toxin": bont_use,
                    "gpi_dbs": dbs,
                })
            elif gene == "GCH1":
                onset_age = rng.randint(1, 18)
                female = rng.random() < 0.75
                diurnal_fluctuation = True
                levodopa_response = True
                misdiagnosed_cp = onset_age < 10 and rng.random() < 0.55
                parkinsonism_later = age > 40 and rng.random() < 0.25
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "onset_age": onset_age,
                    "female": female,
                    "diurnal_fluctuation": diurnal_fluctuation,
                    "levodopa_dramatic_response": levodopa_response,
                    "misdiagnosed_cerebral_palsy": misdiagnosed_cp,
                    "parkinsonism_in_adult": parkinsonism_later,
                })
            elif gene == "ATP1A3":
                syndrome = rng.choice(["AHC", "RDP", "CAPOS"])
                onset_age = rng.randint(0, 55) if syndrome == "RDP" else (rng.randint(0, 18) if syndrome == "CAPOS" else rng.randint(0, 2))
                flunarizine = syndrome == "AHC" and rng.random() < 0.80
                bilateral_hemi = syndrome == "AHC" and rng.random() < 0.30
                capos_snhl = syndrome == "CAPOS"
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "syndrome": syndrome,
                    "onset_age": onset_age,
                    "flunarizine_use": flunarizine,
                    "bilateral_hemiplegia_event": bilateral_hemi,
                    "capos_snhl": capos_snhl,
                })
            elif gene == "KMT2B":
                onset_age = rng.randint(1, 12)
                oculomotor = rng.random() < 0.85
                mild_id = rng.random() < 0.80
                microdeletion = rng.random() < 0.28
                gpi_dbs = rng.random() < 0.60
                generalised = True
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "onset_age": onset_age,
                    "oculomotor_abnormality": oculomotor,
                    "mild_id": mild_id,
                    "microdeletion_detected": microdeletion,
                    "gpi_dbs": gpi_dbs,
                    "generalised_dystonia": generalised,
                })
            elif gene == "ADCY5":
                onset_age = rng.randint(0, 3)
                nocturnal_dyskinesia = True
                facial_hypotonia = rng.random() < 0.90
                chorea = rng.random() < 0.80
                dystonia = rng.random() < 0.75
                myoclonus = rng.random() < 0.65
                caffeine_exposure = rng.random() < 0.20
                clonazepam = rng.random() < 0.70
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "onset_age": onset_age,
                    "nocturnal_dyskinesia": nocturnal_dyskinesia,
                    "facial_hypotonia": facial_hypotonia,
                    "chorea": chorea,
                    "dystonia": dystonia,
                    "myoclonus": myoclonus,
                    "caffeine_exposure": caffeine_exposure,
                    "clonazepam_use": clonazepam,
                })
            elif gene == "ANO3":
                onset_age = rng.randint(30, 65)
                cervical = rng.random() < 0.85
                tremor = rng.random() < 0.75
                cranial = rng.random() < 0.30
                generalised = rng.random() < 0.05
                bont_use = rng.random() < 0.85
                sensory_trick = rng.random() < 0.70
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "onset_age": onset_age,
                    "cervical_dystonia": cervical,
                    "tremor_prominent": tremor,
                    "cranial_involvement": cranial,
                    "generalised": generalised,
                    "botulinum_toxin": bont_use,
                    "sensory_trick": sensory_trick,
                })
            elif gene == "GNAL":
                onset_age = rng.randint(30, 65)
                spasmodic_dysphonia = rng.random() < 0.75
                blepharospasm = rng.random() < 0.40
                oromandibular = rng.random() < 0.25
                cervical = rng.random() < 0.20
                limb_involvement = False  # spared
                bont_laryngeal = spasmodic_dysphonia and rng.random() < 0.88
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "onset_age": onset_age,
                    "spasmodic_dysphonia": spasmodic_dysphonia,
                    "blepharospasm": blepharospasm,
                    "oromandibular_dystonia": oromandibular,
                    "cervical_involvement": cervical,
                    "limb_involvement": limb_involvement,
                    "botulinum_toxin_laryngeal": bont_laryngeal,
                })
    return all_patients


def overview():
    """Return atlas-level aggregate overview."""
    patients = _generate_cohort()
    generalised = sum(1 for p in patients if p.get("generalised_dystonia") or p.get("generalised"))
    gpi_dbs = sum(1 for p in patients if p.get("gpi_dbs"))
    levodopa_response = sum(1 for p in patients if p.get("levodopa_dramatic_response"))
    bont_use = sum(1 for p in patients if p.get("botulinum_toxin") or p.get("botulinum_toxin_laryngeal"))
    nocturnal = sum(1 for p in patients if p.get("nocturnal_dyskinesia"))
    oculomotor = sum(1 for p in patients if p.get("oculomotor_abnormality"))
    misdiagnosed = sum(1 for p in patients if p.get("misdiagnosed_cerebral_palsy"))
    spasmodic_dysphonia = sum(1 for p in patients if p.get("spasmodic_dysphonia"))
    tremor_prominent = sum(1 for p in patients if p.get("tremor_prominent"))
    flunarizine = sum(1 for p in patients if p.get("flunarizine_use"))
    return {
        "atlas": "Hereditary-Dystonia-Atlas",
        "genes": [g["gene"] for g in DYT_GENES],
        "total_patients": len(patients),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "generalised_dystonia_patients": generalised,
        "gpi_dbs_patients": gpi_dbs,
        "levodopa_responsive_drd_patients": levodopa_response,
        "botulinum_toxin_patients": bont_use,
        "nocturnal_dyskinesia_patients": nocturnal,
        "oculomotor_abnormality_patients": oculomotor,
        "misdiagnosed_cerebral_palsy": misdiagnosed,
        "spasmodic_dysphonia_patients": spasmodic_dysphonia,
        "tremor_prominent_patients": tremor_prominent,
        "flunarizine_use_ahc": flunarizine,
    }


def breakdown():
    """Return per-gene breakdown with clinical details."""
    patients = _generate_cohort()
    result = {}
    for gene_data in DYT_GENES:
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
    """Return gene definitions, glossary and treatment protocols."""
    return {
        "genes": {g["gene"]: g["protein"] for g in DYT_GENES},
        "glossary": {
            "Primary Dystonia": "Dystonia as the only or predominant feature; no identifiable secondary cause (normal brain MRI, normal metabolic workup); includes TOR1A (DYT1), THAP1 (DYT6), ANO3 (DYT24), GNAL (DYT25)",
            "Dopa-Responsive Dystonia (DRD)": "Childhood-onset dystonia with MIRACULOUS levodopa response; caused by GCH1 (most common), TH, SPR mutations; BH4 biosynthesis deficiency → dopamine deficiency; diurnal fluctuation pathognomonic; low-dose levodopa curative",
            "DYT Nomenclature": "International Parkinson and Movement Disorder Society (IPMDS) classification: DYT-TOR1A, DYT-THAP1, DYT-GCH1, DYT-ATP1A3, DYT-KMT2B, DYT-ADCY5, DYT-ANO3, DYT-GNAL; replaces old DYT1-DYT25 numerical designations",
            "GPi-DBS (Globus Pallidus Interna Deep Brain Stimulation)": "Neurostimulation targeting GPi bilaterally; most effective surgical treatment for generalised primary dystonia; TOR1A (DYT1): 70-90% improvement; KMT2B (DYT28): highly responsive; THAP1: moderate; GCH1: CONTRA (levodopa sufficient); SNc-DBS rarely used",
            "Diurnal Fluctuation": "Worsening of symptoms during the day (afternoon/evening) with improvement after sleep; PATHOGNOMONIC for DRD (GCH1); absent in all other dystonia genes; if present, always trial levodopa before any other treatment",
            "Botulinum Toxin A (BoNT-A)": "Serotype A clostridial toxin; blocks acetylcholine release at NMJ; types: onabotulinumtoxinA (Botox), abobotulinumtoxinA (Dysport); used in focal/segmental dystonia — cervical, cranial, laryngeal, limb; repeat every 10-16 weeks; gold standard for focal dystonia",
            "Reduced Penetrance": "Pathogenic variant carrier does NOT always develop the disease; TOR1A: 30% penetrance; THAP1: 60%; GNAL: 60%; ANO3: 60%; GCH1: female 87%, male 38%; counselling critical — asymptomatic carrier test positive does not predict disease",
            "Sensory Trick (Geste Antagoniste)": "Tactile or proprioceptive manoeuvre that temporarily reduces dystonic posturing; classic: touching chin relieves cervical dystonia (torticollis); absent in Parkinson's cervical rigidity; pathognomonic for dystonia (not Parkinsonism)",
            "AHC (Alternating Hemiplegia of Childhood)": "ATP1A3 disorder; onset <18 months; paroxysmal alternating hemiplegias; episodes triggered by fever/stress/water; resolve with sleep (PATHOGNOMONIC); bilateral hemiplegia = respiratory emergency; flunarizine for prevention",
            "RDP (Rapid-Onset Dystonia-Parkinsonism)": "ATP1A3 disorder; ABRUPT onset hours to weeks; rostrocaudal gradient (cranial > arm > leg); plateau after onset — NOT progressive like PD; no levodopa response; DATscan NORMAL (contrasts PD)",
            "Spasmodic Dysphonia": "Focal laryngeal dystonia; adductor type: strained-strangled voice with pitch breaks; abductor type: breathy/aphonic breaks; GNAL and THAP1 genes; BoNT-A laryngeal injection (thyroarytenoid) primary treatment; highly effective",
            "Tetrabenazine": "VMAT2 inhibitor (depletes presynaptic monoamines); indicated for Huntington's chorea and secondary hyperkinesia; CONTRAINDICATED in primary dystonia (TOR1A/THAP1/KMT2B) — worsens dystonia by depleting dopamine in context of already dysregulated circuitry",
            "Nocturnal Dyskinesia (ADCY5)": "ADCY5-related hyperkinetic movement disorder; characterised by episodes during NREM sleep — chorea/dystonia/myoclonus; EEG NORMAL during episodes (NOT epilepsy); caffeine absolutely contraindicated; clonazepam and acetazolamide first-line treatment",
            "BH4 (Tetrahydrobiopterin)": "Essential cofactor for tyrosine hydroxylase (TH), phenylalanine hydroxylase (PAH), and nitric oxide synthase; GCH1 → BH4; BH4 deficiency → TH dysfunction → dopamine deficiency → DRD; phenylalanine loading test screens GCH1 pathway function",
            "Levodopa Trial in Childhood Dystonia": "MANDATORY in ALL children with unexplained dystonia before other treatments; GCH1/DRD responds miraculously — low dose 2-5 mg/kg/day; TOR1A/THAP1/KMT2B: poor response; ATP1A3/ADCY5: poor response; low cost and low side-effect profile — always try first",
        },
        "surveillance_protocols": {
            "TOR1A (DYT1)": "Genetic testing with specific delGAG targeted PCR first (90%+ of cases); full sequencing if delGAG negative; penetrance counselling (30%); initiate trihexyphenidyl if symptomatic; refer to DBS centre if pharmacotherapy inadequate; family cascade testing; antenatal counselling if requested",
            "THAP1 (DYT6)": "THAP1 sequencing (exons 1-3 priority — THAP domain); levodopa trial to exclude DRD; botulinum toxin clinic every 12 weeks if focal; DBS centre referral if generalised; voice assessment (spasmodic dysphonia); penetrance counselling",
            "GCH1 (DRD)": "GCH1 sequencing; CSF neurotransmitters (biopterin, HVA) if available; phenylalanine loading test; urine pterins; LEVODOPA TRIAL FIRST (1-2 mg/kg/day); DATscan if parkinsonism in adult presentation; lifelong levodopa; annual clinic; folate supplementation in pregnancy; cascade family testing",
            "ATP1A3 (AHC/RDP/CAPOS)": "ATP1A3 sequencing (targeted p.E815K/p.D801N/p.E818K first); AHC: flunarizine 5-10 mg/day; emergency protocol for bilateral hemiplegia; ABR and VEP for CAPOS; ophthalmology (optic atrophy); audiometry; trigger avoidance education; emergency travel card",
            "KMT2B (DYT28)": "KMT2B sequencing + CMA (chromosomal microarray — mandatory, not optional); ophthalmology (oculomotor assessment); neuropsychology (mild ID); levodopa trial; early GPi-DBS referral (do not delay); physiotherapy; educational support",
            "ADCY5 (ADCY5-RMD)": "ADCY5 sequencing (target p.R418W, p.A726T hotspots); parental mosaicism testing (blood AND saliva); nocturnal video-EEG (exclude epilepsy during episodes); sleep study (PSG); caffeine absolutely prohibited (patient/family education); clonazepam titration; acetazolamide if incomplete response",
            "ANO3 (DYT24)": "ANO3 sequencing; cervical dystonia botulinum toxin clinic every 12 weeks; DATscan if parkinsonism suspected; neurophysiology (EMG); physiotherapy; geste antagoniste training; family genetic counselling (reduced penetrance); DBS referral if refractory",
            "GNAL (DYT25)": "GNAL sequencing; ENT laryngoscopy (laryngeal dystonia assessment); speech pathology; botulinum toxin laryngeal clinic (thyroarytenoid injection); ophthalmology (blepharospasm); levodopa trial to exclude DRD; cascade family testing; voice therapy adjunct",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Generalised dystonia patients: {ov['generalised_dystonia_patients']}")
    print(f"GPi-DBS patients: {ov['gpi_dbs_patients']}")
    print(f"Levodopa-responsive (DRD) patients: {ov['levodopa_responsive_drd_patients']}")
    print(f"Botulinum toxin patients: {ov['botulinum_toxin_patients']}")
    print(f"Spasmodic dysphonia patients: {ov['spasmodic_dysphonia_patients']}")
