#!/usr/bin/env python3
"""Hereditary-Tremor-Ataxia-Atlas — Complete 8-Gene Hereditary Tremor & Episodic/Progressive Ataxia Atlas.

FMR1     (Fragile X Mental Retardation Protein; ~632 aa transcript / CGG repeat; Xq27.3; X-linked;
          FXTAS — Fragile X Tremor-Ataxia Syndrome; premutation 55-200 CGG repeats;
          INTENTION TREMOR + CEREBELLAR ATAXIA in males >50 yr;
          MRI: middle cerebellar peduncle (MCP) T2 hyperintensity PATHOGNOMONIC;
          Cognitive decline, parkinsonism, autonomic neuropathy late;
          FMR1 premutation testing + neuroimaging; NO disease-modifying Rx; seed SEED_BASE+0).
CACNA1A  (Voltage-gated calcium channel Cav2.1 alpha-1A; 2505 aa; 19p13.13; AD;
          Two disorders on one gene — EA2 (episodic attacks) + SCA6 (progressive CAG ≥20);
          EA2: paroxysmal cerebellar ataxia + NYSTAGMUS BETWEEN ATTACKS PATHOGNOMONIC;
          Acetazolamide Level B for EA2 — majority respond; SCA6: pure late-onset cerebellar;
          seed SEED_BASE+1).
FGF14    (Fibroblast Growth Factor 14; 252 aa; 13q33.1; AD;
          SCA27B — most common late-onset cerebellar ataxia recently recognised;
          GAA-TTC intronic repeat expansion >250 units pathogenic;
          Standard panels MISS IT — request LONG-READ PCR or repeat-primed PCR;
          4-aminopyridine (4-AP) SPECIFIC RESPONSE — Level B; seed SEED_BASE+2).
RFC1     (Replication Factor C subunit 1; 1148 aa; 4p14; AR biallelic;
          CANVAS — Cerebellar Ataxia + Neuropathy + Vestibular Areflexia Syndrome;
          Biallelic AAGGG pentanucleotide repeat expansion (pathogenic: >400 units each allele);
          PURE SENSORY neuropathy + vestibular areflexia TRIAD PATHOGNOMONIC;
          Cough (chronic dry) in >70% — under-recognised clue;
          Standard sequencing MISSES repeat — request repeat-primed PCR; seed SEED_BASE+3).
NOTCH2NLC (Neuronal intranuclear inclusion disease protein; 1XGG-repeat context; 1q22; AD;
          NIID — Neuronal Intranuclear Inclusion Disease;
          GGC repeat expansion >60 units pathogenic;
          SKIN BIOPSY intranuclear eosinophilic inclusions (p62+/ubiquitin+) PATHOGNOMONIC;
          DWI MRI: cortico-medullary junction hyperintensity — highly characteristic;
          Dementia + tremor + parkinsonism + peripheral neuropathy; seed SEED_BASE+4).
PRKCG    (Protein Kinase C Gamma; 697 aa; 19q13.42; AD;
          SCA14 — spinocerebellar ataxia type 14;
          ACTION TREMOR PROMINENT feature (distinguishes from other SCAs);
          Onset 20-40 yr; slow progression; near-normal life expectancy;
          Missense in kinase domain / C1 domain; cognitive preservation in most; seed SEED_BASE+5).
ITPR1    (Inositol 1,4,5-Trisphosphate Receptor Type 1; 2695 aa; 3p26.1; AD;
          SCA15 — pure cerebellar ataxia, VERY SLOW progression;
          Deletions common — MLPA MANDATORY alongside sequencing;
          Standard point-mutation panels MISS DELETIONS;
          Gaze-evoked nystagmus; no extracerebellar features; seed SEED_BASE+6).
ELOVL5   (ELOVL Fatty Acid Elongase 5; 299 aa; 6p12.3; AD;
          SCA38 — spinocerebellar ataxia type 38;
          Reduced serum DHA (docosahexaenoic acid) PATHOGNOMONIC metabolic signature;
          Dietary DHA supplementation (1g/day) — clinical trial evidence, well-tolerated;
          Pes cavus; mild sensory neuropathy; seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2094-2101).
"""

import random

SEED_BASE = 2094

TREMOR_ATAXIA_GENES = [
    # -- FMR1 — FXTAS ----------------------------------------------------------
    {
        "gene": "FMR1",
        "alt_name": (
            "FMR1 (FMR1-Xq27.3 / X-linked — FXTAS-Fragile-X-Tremor-Ataxia-Syndrome — "
            "Premutation-55-200-CGG-Repeats — Late-Onset-Males->50yr — "
            "MCP-Hyperintensity-PATHOGNOMONIC — No-Disease-Modifying-Rx)"
        ),
        "protein": (
            "FMR1 -- Xq27.3 X-linked -- FMR1-CGG-repeat -- "
            "Fragile-X-Mental-Retardation-Protein-FMRP-mRNA-Binding-Translational-Regulator -- "
            "Normal-<44-CGG; Premutation-55-200-CGG-FMR1-mRNA-TOXIC-GAIN-OF-FUNCTION -- "
            "Full-Mutation->200-CGG-Fragile-X-Syndrome-SEPARATE-DISORDER -- "
            "FXTAS-Premutation-Males-Penetrance-50pct-over-70yr-Females-20pct -- "
            "Intention-Tremor-FIRST-Then-Cerebellar-Ataxia-Progression -- "
            "MRI-MCP-T2-Hyperintensity-PATHOGNOMONIC-Bilateral-Middle-Cerebellar-Peduncle -- "
            "Cognitive-Decline-Executive-Dysfunction-Late -- "
            "Autonomic-Dysfunction-Parkinsonism-Peripheral-Neuropathy -- "
            "FMR1-mRNA-Elevated-But-FMRP-Protein-Normal-Or-Slightly-Reduced -- "
            "FXS-Fragile-X-Carriers-Premutation-50pct-Risk-Each-Son-Daughters -- "
            "Genetic-Counselling-Mandatory-Cascade-Testing"
        ),
        "locus": "Xq27.3",
        "protein_size": "CGG repeat / 632 aa FMRP",
        "inheritance": (
            "X-linked (FXTAS) — premutation 55-200 CGG; males predominantly affected >50 yr; "
            "females affected 20% (heterozygous carrier, more protected by normal X); "
            "Penetrance in males: 30% at 50-59yr → 75% at >80yr; "
            "Full mutation (>200 CGG) = Fragile X Syndrome — DIFFERENT disorder, childhood onset; "
            "FMR1 premutation transmission: female carrier → 50% sons have premutation/full mutation; "
            "Anticipation through maternal transmission (repeats expand) — track pedigree carefully; "
            "Standard FMR1 repeat PCR detects premutation; Southern blot for full mutation sizing"
        ),
        "pathognomonic": (
            "MRI MIDDLE CEREBELLAR PEDUNCLE (MCP) T2/FLAIR HYPERINTENSITY BILATERAL — neuroimaging PATHOGNOMONIC; "
            "FXTAS = Premutation 55-200 CGG + cerebellar tremor + ataxia + cognitive decline in male >50yr; "
            "Intranuclear inclusions (FMR1 mRNA aggregates) in neurons and astrocytes — autopsy/biopsy; "
            "FMR1 mRNA elevated on molecular testing (toxic gain-of-function mechanism); "
            "EEG: diffuse slowing late; EMG: axonal sensorimotor neuropathy; "
            "Autonomic tests: orthostatic hypotension, bowel/bladder; "
            "Memory impairment + executive dysfunction on neuropsychological testing"
        ),
        "treatment": (
            "NO disease-modifying therapy (no approved agent); "
            "Symptomatic: propranolol or primidone for tremor; "
            "Memantine (off-label cognitive support — limited evidence); "
            "Physiotherapy + OT (balance, falls prevention); "
            "National Fragile X Foundation registry + genetic counselling; "
            "Cascade FMR1 testing in all at-risk female relatives (premutation carriers); "
            "Annual neurological review + fall risk; "
            "Neuropsychiatric support (anxiety, depression common)"
        ),
        "critical_flags": [
            "MCP-HYPERINTENSITY-MRI-PATHOGNOMONIC",
            "PREMUTATION-55-200-CGG-NOT-FULL-MUTATION",
            "MALES->50yr-50pct-PENETRANCE",
            "CASCADE-FMR1-TESTING-FEMALE-RELATIVES",
            "ANTICIPATION-MATERNAL-TRANSMISSION-EXPANDS",
            "NO-DISEASE-MODIFYING-RX",
            "FXS-FULL-MUTATION-DIFFERENT-DISORDER",
        ],
        "age_of_onset": "Typically >50 years in males (mean onset 60-65yr)",
        "key_biomarker": "FMR1 CGG repeat 55-200 (premutation) + elevated FMR1 mRNA",
        "seed": SEED_BASE + 0,
    },
    # -- CACNA1A — EA2 / SCA6 -------------------------------------------------
    {
        "gene": "CACNA1A",
        "alt_name": (
            "CACNA1A (CACNA1A-2505aa-19p13.13 / AD — EA2-Episodic-Ataxia-Type-2-Acetazolamide-Level-B — "
            "SCA6-CAG≥20-Pure-Progressive-Late-Onset — "
            "Nystagmus-Between-Attacks-PATHOGNOMONIC-EA2 — Familial-Hemiplegic-Migraine-Type-1)"
        ),
        "protein": (
            "CACNA1A -- 19p13.13 AD -- CACNA1A-2505aa -- "
            "Voltage-Gated-P/Q-Type-Calcium-Channel-Cav2.1-Alpha-1A-Subunit -- "
            "Three-Allelic-Disorders-One-Gene: EA2-missense/splice + SCA6-CAG-repeat + FHM1-gain-of-function -- "
            "EA2-Loss-of-Function-Missense/Splice-Mutations-Dominant-Negative -- "
            "SCA6-CAG-Repeat->20-Abnormal-Normal-<19-Polyglutamine-Toxic-Aggregation -- "
            "FHM1-Gain-of-Function-Missense-Familial-Hemiplegic-Migraine -- "
            "Cav2.1-Purkinje-Cell-Predominant-Expression-Cerebellar-Output -- "
            "EA2-Paroxysmal-Attacks-Hours-Days-With-Interictal-Nystagmus -- "
            "SCA6-Pure-Cerebellar-No-Extracerebellar-Features-Onset->40yr -- "
            "Acetazolamide-Prevents-EA2-Attacks-Level-B-Evidence"
        ),
        "locus": "19p13.13",
        "protein_size": "2505 aa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance for EA2 (~90%); "
            "EA2: missense/splice loss-of-function; "
            "SCA6: CAG ≥20 repeat expansion — test SEPARATELY from point mutations; "
            "FHM1: gain-of-function missense (distinct phenotype); "
            "SCA6 very low repeat length effect (≥20 vs normal ≤18 — narrow window); "
            "Anticipation minimal in SCA6; "
            "De novo EA2 mutations described; "
            "Family history of episodic ataxia/migraine points to CACNA1A first"
        ),
        "pathognomonic": (
            "EA2: NYSTAGMUS BETWEEN ATTACKS PATHOGNOMONIC (down-beat or gaze-evoked — persists interictal); "
            "EA2 attacks: hours-to-days cerebellar ataxia + vertigo + nausea + headache; "
            "Interictal nystagmus differentiates EA2 from EA1 (EA1 = no interictal nystagmus); "
            "SCA6: CAG ≥20 repeat — molecular PATHOGNOMONIC; "
            "Acetazolamide trial: 75-90% attack frequency reduction in EA2; "
            "Cerebellar atrophy on MRI (vermis > hemisphere); "
            "Trigger-sensitive attacks: stress, exercise, alcohol, caffeine, illness; "
            "FHM1: hemiplegic migraine + cerebellar ataxia + sometimes coma"
        ),
        "treatment": (
            "EA2: Acetazolamide 250-1000 mg/day — Level B attack prevention; "
            "4-aminopyridine (4-AP) 5-10 mg TDS — alternative/addition (Kv channel stabilisation); "
            "Avoid known triggers (alcohol, caffeine, stress); "
            "SCA6: symptomatic only — physiotherapy, OT, speech therapy; "
            "Riluzole 50 mg BD (off-label cerebellar SCA benefit — modest evidence); "
            "VGF/CACNA1A gene therapy research ongoing; "
            "SCA Functional Index (SARA score) annually; "
            "CACNA1A European patient registry (Orphanet)"
        ),
        "critical_flags": [
            "INTERICTAL-NYSTAGMUS-PATHOGNOMONIC-EA2",
            "ACETAZOLAMIDE-LEVEL-B-EA2",
            "SCA6-CAG≥20-SEPARATE-TEST-FROM-SEQUENCING",
            "THREE-DISORDERS-ONE-GENE-EA2-SCA6-FHM1",
            "4-AP-ALTERNATIVE-EA2",
            "TRIGGERS-AVOID-ALCOHOL-CAFFEINE",
        ],
        "age_of_onset": "EA2: childhood-young adult (5-25yr); SCA6: late adult (>40yr, mean 52yr)",
        "key_biomarker": "EA2: CACNA1A pathogenic variant; SCA6: CAG ≥20 (repeat-primed PCR mandatory)",
        "seed": SEED_BASE + 1,
    },
    # -- FGF14 — SCA27B -------------------------------------------------------
    {
        "gene": "FGF14",
        "alt_name": (
            "FGF14 (FGF14-252aa-13q33.1 / AD — SCA27B-Most-Common-Late-Onset-Cerebellar-Ataxia — "
            "GAA-TTC-Intronic-Repeat->250-PATHOGNOMONIC — "
            "Standard-Panels-MISS-IT-Long-Read-PCR-Mandatory — "
            "4-Aminopyridine-SPECIFIC-RESPONSE-Level-B)"
        ),
        "protein": (
            "FGF14 -- 13q33.1 AD -- FGF14-252aa -- "
            "Fibroblast-Growth-Factor-14-Intracellular-FGF-Non-Secreted -- "
            "Nav1.6-Sodium-Channel-Modulator-Purkinje-Cell-Axon-Initial-Segment -- "
            "GAA-TTC-Intronic-Repeat-Intron-1-Expansion-Pathogenic->250-Units -- "
            "Normal-<250-Repeats-No-Overlap-Clean-Threshold -- "
            "Standard-Short-Read-Sequencing-AND-Standard-PCR-MISS-Repeat -- "
            "Long-Read-PCR-or-Repeat-Primed-PCR-MANDATORY -- "
            "Most-Common-Late-Onset-Cerebellar-Ataxia-Globally-Newly-Recognised -- "
            "4-Aminopyridine-Restores-Nav1.6-Purkinje-Output-SPECIFIC-MECHANISM"
        ),
        "locus": "13q33.1",
        "protein_size": "252 aa",
        "inheritance": (
            "AD (autosomal dominant); GAA-TTC intronic repeat expansion >250 units in intron 1; "
            "Penetrance high (>90% in older individuals); "
            "De novo expansions rare — familial in majority; "
            "Standard exome/genome sequencing misses this repeat — requires long-read or repeat-primed PCR; "
            "Prevalence: estimated 1/5,000 in late-onset ataxia cohorts — now considered most common; "
            "Anticipation possible (expanding through generations); "
            "Sporadic cases: check parents; repeat-primed PCR recommended in ALL unexplained late ataxia"
        ),
        "pathognomonic": (
            "GAA-TTC REPEAT >250 IN FGF14 INTRON 1 — molecular PATHOGNOMONIC for SCA27B; "
            "CLINICAL: late-onset (>50yr) pure or near-pure cerebellar ataxia + downbeat nystagmus; "
            "4-AMINOPYRIDINE (4-AP) CLINICAL RESPONSE — specific to SCA27B/FGF14 mechanism; "
            "Downbeat nystagmus (spontaneous or positional); "
            "Episodic component in early stages (falls, dizziness); "
            "Cerebellar vermis atrophy on MRI (progressive); "
            "Cognitive spared in most patients; "
            "STANDARD PANELS MISS IT — unexplained ataxia after negative exome → test FGF14 repeat"
        ),
        "treatment": (
            "4-Aminopyridine (4-AP, fampridine) 5-10 mg TDS — Level B SCA27B specific (Purkinje Nav1.6 restoration); "
            "Monitor for QTc prolongation with 4-AP (ECG baseline); "
            "4-AP also improves nystagmus component; "
            "Physiotherapy (balance, gait rehabilitation); "
            "Occupational therapy (falls prevention, aids); "
            "Vestibular rehabilitation (nystagmus); "
            "SCA27B patient registry (EuroAtaxia + FGFR14 consortium); "
            "Annual SARA (Scale for Assessment and Rating of Ataxia) monitoring"
        ),
        "critical_flags": [
            "STANDARD-PANELS-MISS-FGF14-REPEAT",
            "LONG-READ-PCR-MANDATORY",
            "4-AP-SPECIFIC-TREATMENT-LEVEL-B",
            "MOST-COMMON-LATE-ONSET-ATAXIA",
            "GAA-TTC->250-PATHOGNOMONIC",
            "DOWNBEAT-NYSTAGMUS-CLUE",
            "QTc-MONITOR-WITH-4-AP",
        ],
        "age_of_onset": "Late adult onset, typically >50 yr (range 45-75yr)",
        "key_biomarker": "FGF14 GAA-TTC repeat >250 (long-read PCR or repeat-primed PCR)",
        "seed": SEED_BASE + 2,
    },
    # -- RFC1 — CANVAS --------------------------------------------------------
    {
        "gene": "RFC1",
        "alt_name": (
            "RFC1 (RFC1-1148aa-4p14 / AR-biallelic — CANVAS-Cerebellar-Ataxia-Neuropathy-Vestibular-Areflexia — "
            "AAGGG-Pentanucleotide-Biallelic->400-PATHOGNOMONIC — "
            "Pure-Sensory-Neuropathy-Chronic-Cough-70pct — "
            "Standard-Sequencing-MISSES-Repeat-Primed-PCR-Mandatory)"
        ),
        "protein": (
            "RFC1 -- 4p14 AR -- RFC1-1148aa -- "
            "Replication-Factor-C-Subunit-1-DNA-Clamp-Loader-PCNA-Loading -- "
            "Intronic-AAGGG-Pentanucleotide-Repeat-Expansion-Biallelic-Pathogenic->400-Each-Allele -- "
            "Normal-AAAAG-Allele-Pathogenic-AAGGG-Different-Sequence-Not-Just-Length -- "
            "Several-Pathogenic-Motifs-AAGGG-ACAGG-AGAGG-Biallelic-Combinations -- "
            "CANVAS-Triad-Cerebellar-Ataxia-Sensory-Neuropathy-Vestibular-Areflexia -- "
            "Chronic-Dry-Cough->70pct-Under-Recognised-Clue-RFC1 -- "
            "Standard-Short-Read-Sequencing-Misses-Repeat -- "
            "Repeat-Primed-PCR-Plus-Flanking-Genotype-MANDATORY -- "
            "Slowly-Progressive-Middle-Age-Onset"
        ),
        "locus": "4p14",
        "protein_size": "1148 aa",
        "inheritance": (
            "AR (autosomal recessive) — biallelic AAGGG pentanucleotide repeat expansion; "
            "Both alleles must carry pathogenic motif (AAGGG or other pathogenic variants: ACAGG, AGAGG); "
            "Compound heterozygotes (AAGGG + ACAGG biallelic) also pathogenic; "
            "Normal allele: AAAAG motif (5-nucleotide unit); pathogenic: AAGGG; "
            "Prevalence: ~1/1,000 in European late-onset ataxia cohorts; "
            "Standard sequencing and exome MISS this repeat; "
            "Repeat-primed PCR + flanking size analysis required; "
            "Siblings: 25% risk; parents obligate carriers (asymptomatic)"
        ),
        "pathognomonic": (
            "CANVAS TRIAD PATHOGNOMONIC: Cerebellar Ataxia + Pure Sensory Neuropathy + Bilateral Vestibular Areflexia; "
            "PURE SENSORY neuropathy (no motor component) — distinguishes from Friedreich ataxia; "
            "BILATERAL VESTIBULAR AREFLEXIA — absent caloric response (video head impulse test VHIT); "
            "CHRONIC DRY COUGH >70% — pathognomonic clinical clue (cardiac neurogenic cough); "
            "Biallelic RFC1 AAGGG >400 units — molecular pathognomonic; "
            "Absent lower-limb reflexes + absent sural nerve SNAP on NCS; "
            "Cerebellar vermis atrophy + DRG (dorsal root ganglion) degeneration on imaging; "
            "Oscillopsia (vision blurs when walking) from vestibular areflexia"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "Vestibular rehabilitation (balance board, VR vestibular exercises); "
            "Physiotherapy (gait, walking aids, falls prevention); "
            "COUGH: ACE inhibitor cessation if prescribed; speech therapy for aspiration risk; "
            "Sensory aids: ankle-foot orthoses (drop foot), proprioceptive bracing; "
            "RFC1 patient registry (EUROCANIC study, EuroAtaxia); "
            "Hearing assessment (concurrent SNHL in some); "
            "Annual SARA + neurophysiology (NCS) monitoring"
        ),
        "critical_flags": [
            "CANVAS-TRIAD-PATHOGNOMONIC",
            "PURE-SENSORY-NEUROPATHY-NO-MOTOR",
            "BILATERAL-VESTIBULAR-AREFLEXIA-PATHOGNOMONIC",
            "CHRONIC-DRY-COUGH-70pct-CLUE",
            "STANDARD-SEQUENCING-MISSES-REPEAT",
            "REPEAT-PRIMED-PCR-MANDATORY",
            "AAGGG-BIALLELIC->400-MOLECULAR",
        ],
        "age_of_onset": "Middle age (40-60yr), rarely earlier",
        "key_biomarker": "Biallelic RFC1 AAGGG pentanucleotide repeat >400 per allele",
        "seed": SEED_BASE + 3,
    },
    # -- NOTCH2NLC — NIID ------------------------------------------------------
    {
        "gene": "NOTCH2NLC",
        "alt_name": (
            "NOTCH2NLC (NOTCH2NLC-1q22 / AD — NIID-Neuronal-Intranuclear-Inclusion-Disease — "
            "GGC-Repeat->60-PATHOGNOMONIC — "
            "Skin-Biopsy-Intranuclear-Inclusions-p62-Ubiquitin-PATHOGNOMONIC — "
            "DWI-Cortico-Medullary-Hyperintensity-PATHOGNOMONIC)"
        ),
        "protein": (
            "NOTCH2NLC -- 1q22 AD -- NOTCH2NLC-GGC-repeat -- "
            "Notch2-N-Terminal-Like-Protein-Poorly-Characterised-Function -- "
            "5-Prime-UTR-GGC-Repeat-Expansion-Pathogenic->60-Units-Normal-<40 -- "
            "Repeat-Codes-Polyglycine-Tract-Toxic-Gain-Of-Function-Aggregation -- "
            "NIID-Spectrum-Adult-Onset-Tremor-Ataxia-Parkinsonism-Dementia-Peripheral-Neuropathy -- "
            "Skin-Biopsy-p62-Ubiquitin-Positive-Intranuclear-Inclusions-In-Fibroblasts-Sweat-Gland-Adipocytes-PATHOGNOMONIC -- "
            "DWI-MRI-Cortico-Medullary-Junction-Linear-Hyperintensity-HALLMARK -- "
            "Leukoencephalopathy-Episodic-Encephalopathy-Encephalitis-Like-Episodes"
        ),
        "locus": "1q22",
        "protein_size": "GGC repeat / variable",
        "inheritance": (
            "AD (autosomal dominant) — GGC repeat expansion in 5' UTR of NOTCH2NLC; "
            "Normal <40 GGC repeats; pathogenic ≥60; grey zone 40-59; "
            "De novo expansions described; familial with variable expressivity; "
            "Standard sequencing misses repeat — request repeat-primed PCR; "
            "Anticipation through paternal transmission; "
            "Penetrance age-related; onset 40-70yr for neurological features; "
            "Asian (Japanese, Chinese) population predominantly described but worldwide"
        ),
        "pathognomonic": (
            "SKIN BIOPSY: p62+/ubiquitin+ intranuclear eosinophilic inclusions in fibroblasts, sweat gland cells, adipocytes — PATHOGNOMONIC (ante-mortem diagnosis); "
            "DWI MRI: bilateral cortico-medullary junction (subcortical white-grey border) linear hyperintensity — HIGHLY CHARACTERISTIC; "
            "GGC REPEAT >60 in NOTCH2NLC — molecular pathognomonic; "
            "Leukoencephalopathy on T2/FLAIR MRI (white matter changes); "
            "CLINICAL SPECTRUM: tremor + ataxia + parkinsonism + dementia + peripheral neuropathy + autonomic dysfunction + encephalopathy episodes; "
            "Encephalopathy episodes (fever-triggered) — sub-acute stroke-like onset; "
            "Peripheral neuropathy (axonal sensorimotor)"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "Symptomatic: levodopa trial for parkinsonism (partial response); "
            "Beta-blocker / primidone for tremor; "
            "Physiotherapy (ataxia management, Frenkel exercises); "
            "Encephalopathy episodes: avoid fever triggers, steroids controversial; "
            "NIID patient registry (Japan, OMIM 603472 research consortia); "
            "Genetic counselling (AD — 50% offspring risk); "
            "Annual neuropsychological assessment; "
            "Avoid nephrotoxic agents (autonomic renal impact)"
        ),
        "critical_flags": [
            "SKIN-BIOPSY-p62-UBIQUITIN-PATHOGNOMONIC",
            "DWI-CORTICO-MEDULLARY-HYPERINTENSITY",
            "GGC->60-MOLECULAR-PATHOGNOMONIC",
            "STANDARD-SEQUENCING-MISSES-REPEAT",
            "ENCEPHALOPATHY-EPISODES-FEVER-TRIGGERED",
            "LEVODOPA-TRIAL-PARTIAL-RESPONSE",
            "ANTICIPATION-PATERNAL-TRANSMISSION",
        ],
        "age_of_onset": "Adult onset (40-70 yr), broad spectrum",
        "key_biomarker": "NOTCH2NLC GGC repeat >60 + skin biopsy intranuclear inclusions",
        "seed": SEED_BASE + 4,
    },
    # -- PRKCG — SCA14 --------------------------------------------------------
    {
        "gene": "PRKCG",
        "alt_name": (
            "PRKCG (PRKCG-697aa-19q13.42 / AD — SCA14-Spinocerebellar-Ataxia-Type-14 — "
            "Action-Tremor-PROMINENT-Feature — Slow-Progression-Near-Normal-Life-Expectancy — "
            "Kinase-Domain-C1-Domain-Missense)"
        ),
        "protein": (
            "PRKCG -- 19q13.42 AD -- PRKCG-697aa -- "
            "Protein-Kinase-C-Gamma-Serine-Threonine-Kinase-Purkinje-Cell-Specific-High-Expression -- "
            "Regulatory-C1-Domain-Diacylglycerol-Binding-Mutations-Constitutive-Activation -- "
            "Catalytic-Kinase-Domain-Mutations-Impaired-Activity-Or-Misfolding -- "
            "SCA14-Onset-8-42yr-Mean-28yr-Cerebellar-Ataxia-Plus-Action-Tremor -- "
            "ACTION-TREMOR-PROMINENT-Differentiates-From-Other-SCAs -- "
            "Cognitive-Function-Largely-Preserved-Through-Disease -- "
            "Very-Slow-Progression-Ambulatory->20yr-After-Onset -- "
            "Cerebellar-Vermis-Hypoplasia-Early-Plus-Progressive-Atrophy"
        ),
        "locus": "19q13.42",
        "protein_size": "697 aa",
        "inheritance": (
            "AD (autosomal dominant); missense mutations in C1 domain (regulatory) or kinase domain (catalytic); "
            "High penetrance; variable expressivity (tremor-dominant vs ataxia-dominant); "
            "Prevalence: rare (SCA14 ~2-5% of SCAs in some cohorts); "
            "No repeat expansion — standard sequencing detects; "
            "De novo mutations reported; "
            "Family history may be absent in de novo; "
            "PRKCG sequencing on SCA gene panels"
        ),
        "pathognomonic": (
            "ACTION TREMOR PROMINENT distinguishes SCA14 from many other SCAs (most = pure ataxia); "
            "Onset 8-42 yr (mean ~28 yr) — younger than most AD SCAs; "
            "Near-normal life expectancy with very slow progression; "
            "Cerebellar vermis hypoplasia on MRI early (developmental component); "
            "Progressive cerebellar atrophy (vermis + hemispheres) on serial MRI; "
            "Nystagmus (gaze-evoked); "
            "Cognitive preservation (distinguishes from many SCAs); "
            "Action tremor may precede ataxia by years"
        ),
        "treatment": (
            "Symptomatic: propranolol / primidone for action tremor; "
            "Physiotherapy (gait, balance, Frenkel exercises); "
            "OT (tremor aids, writing aids); "
            "Riluzole 50 mg BD (off-label SCA — modest benefit reported in RCT); "
            "Acetazolamide NOT effective (no episodic component — unlike EA2); "
            "Annual SARA score monitoring; "
            "Genetic counselling (50% offspring risk); "
            "SCA patient registry (EUROSCA, SPATAX consortium)"
        ),
        "critical_flags": [
            "ACTION-TREMOR-PROMINENT-DISTINGUISHES-SCA14",
            "SLOW-PROGRESSION-NORMAL-LIFE-EXPECTANCY",
            "YOUNG-ONSET-MEAN-28yr",
            "COGNITIVE-PRESERVATION",
            "ACETAZOLAMIDE-NOT-EFFECTIVE-NO-EPISODIC",
            "RILUZOLE-OFF-LABEL-MODEST",
        ],
        "age_of_onset": "8-42 years (mean ~28 yr)",
        "key_biomarker": "PRKCG pathogenic missense (C1 domain or kinase domain)",
        "seed": SEED_BASE + 5,
    },
    # -- ITPR1 — SCA15 ---------------------------------------------------------
    {
        "gene": "ITPR1",
        "alt_name": (
            "ITPR1 (ITPR1-2695aa-3p26.1 / AD — SCA15-Pure-Cerebellar-Ataxia-Very-Slow-Progression — "
            "Deletions-Common-MLPA-MANDATORY — Standard-Point-Mutation-Panels-MISS-Deletions — "
            "Gaze-Evoked-Nystagmus-No-Extracerebellar-Features)"
        ),
        "protein": (
            "ITPR1 -- 3p26.1 AD -- ITPR1-2695aa -- "
            "Inositol-1-4-5-Trisphosphate-Receptor-Type-1-ER-Ca2+-Release-Channel -- "
            "Predominant-Expression-Purkinje-Cells-Cerebellum-High-Purkinje-Specificity -- "
            "Heterozygous-Missense-Plus-Heterozygous-Deletion-3p26.1-Both-Cause-SCA15 -- "
            "Deletions-Single-Exon-To-Whole-Gene-35-45pct-Of-SCA15-Alleles -- "
            "MLPA-Detects-Deletions-Standard-Sequencing-Only-Finds-Missense -- "
            "SCA15-Pure-Cerebellar-Ataxia-Onset-10-60yr-Mean-35yr -- "
            "VERY-SLOW-Progression-Ambulatory->20-30yr-After-Onset -- "
            "IP3-Receptor-Ca2+-Release-ER-Depletion-Purkinje-Dendritic-Loss"
        ),
        "locus": "3p26.1",
        "protein_size": "2695 aa",
        "inheritance": (
            "AD (autosomal dominant); haploinsufficiency mechanism; "
            "Both missense mutations AND intragenic/whole-gene deletions cause SCA15; "
            "Deletions comprise 35-45% of SCA15 alleles — MLPA mandatory alongside sequencing; "
            "Pure sequencing misses 35-45% of SCA15 alleles; "
            "High penetrance; variable expressivity; "
            "Prevalence: SCA15 ~1-3% of all AD SCAs; "
            "Onset 10-60 yr, very variable within families; "
            "Occasionally isolated cases (de novo deletion)"
        ),
        "pathognomonic": (
            "PURE CEREBELLAR ATAXIA — no extracerebellar signs (no pyramidal, no peripheral neuropathy, no cognitive impairment); "
            "VERY SLOW PROGRESSION — ambulatory 20-30+ years after onset; "
            "Gaze-evoked nystagmus (horizontal ± vertical); "
            "Cerebellar vermis + hemisphere atrophy on MRI (progressive, selective); "
            "Molecular: ITPR1 pathogenic missense OR heterozygous deletion — MLPA confirms deletion; "
            "No cardiac or systemic features (pure neurological); "
            "Excellent prognosis relative to other SCAs"
        ),
        "treatment": (
            "Symptomatic only; "
            "Physiotherapy (balance, gait, Frenkel exercises); "
            "Vestibular rehabilitation (nystagmus component); "
            "Riluzole 50 mg BD (off-label — some SCA benefit signal); "
            "Annual SARA score + MRI every 3 years; "
            "Genetic counselling (50% offspring risk); "
            "MLPA mandatory in diagnostic workup — communicate to ordering lab; "
            "SCA15 patient registry (EUROSCA); "
            "Driving assessment (early restriction if gait/nystagmus severe)"
        ),
        "critical_flags": [
            "DELETIONS-35-45pct-OF-ALLELES",
            "MLPA-MANDATORY-STANDARD-SEQUENCING-MISSES",
            "PURE-CEREBELLAR-NO-EXTRACEREBELLAR",
            "VERY-SLOW-PROGRESSION-AMBULATORY-20-30yr",
            "GAZE-EVOKED-NYSTAGMUS",
            "HAPLOINSUFFICIENCY-MECHANISM",
        ],
        "age_of_onset": "10-60 years (mean ~35 yr)",
        "key_biomarker": "ITPR1 pathogenic missense OR heterozygous deletion (MLPA)",
        "seed": SEED_BASE + 6,
    },
    # -- ELOVL5 — SCA38 -------------------------------------------------------
    {
        "gene": "ELOVL5",
        "alt_name": (
            "ELOVL5 (ELOVL5-299aa-6p12.3 / AD — SCA38-Spinocerebellar-Ataxia-Type-38 — "
            "DHA-Docosahexaenoic-Acid-Deficiency-PATHOGNOMONIC-Metabolic-Signature — "
            "Dietary-DHA-Supplementation-1g-Day-Clinical-Trial-Evidence — "
            "Pes-Cavus-Sensory-Neuropathy-Late-Onset)"
        ),
        "protein": (
            "ELOVL5 -- 6p12.3 AD -- ELOVL5-299aa -- "
            "ELOVL-Fatty-Acid-Elongase-5-ER-Membrane-Enzyme -- "
            "Elongates-C18-C20-Polyunsaturated-Fatty-Acids-Toward-DHA-EPA-Synthesis -- "
            "Loss-of-Function-Impairs-PUFA-Elongation-DHA-Docosahexaenoic-Acid-Deficiency -- "
            "DHA-Critical-For-Purkinje-Cell-Membrane-Synaptic-Vesicle-Function -- "
            "REDUCED-SERUM-DHA-PATHOGNOMONIC-Metabolic-Biomarker -- "
            "SCA38-Late-Onset-Cerebellar-Ataxia-Onset-35-60yr -- "
            "Pes-Cavus-Foot-Deformity-Sensory-Neuropathy-Associated -- "
            "Dietary-DHA-Omega-3-Fish-Oil-1g-Day-Reduces-Progression"
        ),
        "locus": "6p12.3",
        "protein_size": "299 aa",
        "inheritance": (
            "AD (autosomal dominant); missense mutations in transmembrane domain; "
            "Loss-of-function haploinsufficiency; "
            "Rare — Italian-origin families predominantly (founder effect suspected); "
            "Standard sequencing detects (no repeat expansion); "
            "Prevalence: SCA38 rare worldwide, probably underdiagnosed due to DHA not routinely measured; "
            "All unexplained late-onset ataxia with low serum DHA → test ELOVL5; "
            "De novo cases described"
        ),
        "pathognomonic": (
            "REDUCED SERUM DHA PATHOGNOMONIC METABOLIC SIGNATURE — fasting serum omega-3 fatty acid profile; "
            "DHA specifically low (EPA may be normal); "
            "Cerebellar ataxia + pes cavus foot deformity (high arch) + sensory peripheral neuropathy; "
            "Late onset (35-60yr); "
            "Cerebellar atrophy on MRI (progressive); "
            "Dietary DHA supplementation shows clinical stabilisation / improvement; "
            "ELOVL5 pathogenic missense on sequencing; "
            "No myoclonus, no pyramidal signs, no cognitive impairment typically"
        ),
        "treatment": (
            "DIETARY DHA SUPPLEMENTATION 1 g/day (omega-3 fish oil): Level C — clinical trial evidence, "
            "well-tolerated, biomarker improvement confirmed; "
            "Serum DHA monitoring (fasting omega-3 PUFA profile) annually; "
            "Physiotherapy (gait, balance); "
            "Orthotics for pes cavus; "
            "SARA score annually; "
            "Neuropathy management (gabapentin if painful); "
            "SCA38 patient registry (EuroAtaxia); "
            "Dietary counselling: increase oily fish intake; "
            "Genetic counselling (50% offspring risk)"
        ),
        "critical_flags": [
            "DHA-DEFICIENCY-PATHOGNOMONIC-METABOLIC",
            "DIETARY-DHA-1g-DAY-TREATMENT",
            "SERUM-OMEGA-3-PROFILE-MANDATORY",
            "PES-CAVUS-SENSORY-NEUROPATHY-ASSOCIATED",
            "LATE-ONSET-35-60yr",
            "STANDARD-SEQUENCING-DETECTS",
        ],
        "age_of_onset": "35-60 years",
        "key_biomarker": "Reduced serum DHA (fasting omega-3 fatty acid profile) + ELOVL5 pathogenic missense",
        "seed": SEED_BASE + 7,
    },
]


def _generate_cohort():
    """Generate 8×40 deterministic patient cohort (seeds 2094-2101)."""
    all_patients = []

    for gene_data in TREMOR_ATAXIA_GENES:
        gene = gene_data["gene"]
        seed = gene_data["seed"]
        rng = random.Random(seed)

        for i in range(40):
            pid = f"{gene}-{seed}-{i+1:03d}"

            if gene == "FMR1":
                sex = rng.choice(["Male", "Male", "Male", "Female"])  # males 75%
                mcp_sign = sex == "Male" and rng.random() < 0.80
                cognitive_decline = rng.random() < 0.60
                parkinsonism = rng.random() < 0.35
                autonomic = rng.random() < 0.40
                age_onset = rng.randint(52, 78)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "sex": sex,
                    "mcp_hyperintensity": mcp_sign,
                    "cognitive_decline": cognitive_decline,
                    "parkinsonism": parkinsonism,
                    "autonomic_dysfunction": autonomic,
                    "age_onset": age_onset,
                })

            elif gene == "CACNA1A":
                disorder = rng.choice(["EA2", "EA2", "SCA6"])  # EA2 more common
                acetazolamide_responder = disorder == "EA2" and rng.random() < 0.85
                interictal_nystagmus = disorder == "EA2" and rng.random() < 0.90
                four_ap_responder = disorder == "EA2" and not acetazolamide_responder and rng.random() < 0.60
                age_onset = rng.randint(5, 25) if disorder == "EA2" else rng.randint(40, 65)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "disorder_type": disorder,
                    "acetazolamide_responder": acetazolamide_responder,
                    "interictal_nystagmus": interictal_nystagmus,
                    "four_ap_responder": four_ap_responder,
                    "age_onset": age_onset,
                })

            elif gene == "FGF14":
                standard_panel_missed = rng.random() < 0.95  # usually missed on standard panels
                four_ap_responder = rng.random() < 0.70
                downbeat_nystagmus = rng.random() < 0.65
                age_onset = rng.randint(48, 75)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "standard_panel_missed": standard_panel_missed,
                    "four_ap_responder": four_ap_responder,
                    "downbeat_nystagmus": downbeat_nystagmus,
                    "age_onset": age_onset,
                })

            elif gene == "RFC1":
                pure_sensory_neuropathy = rng.random() < 0.95
                vestibular_areflexia = rng.random() < 0.90
                chronic_cough = rng.random() < 0.72
                standard_missed = rng.random() < 0.90
                age_onset = rng.randint(40, 62)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "pure_sensory_neuropathy": pure_sensory_neuropathy,
                    "bilateral_vestibular_areflexia": vestibular_areflexia,
                    "chronic_dry_cough": chronic_cough,
                    "standard_sequencing_missed": standard_missed,
                    "age_onset": age_onset,
                })

            elif gene == "NOTCH2NLC":
                skin_biopsy_inclusions = rng.random() < 0.90
                dwi_corticomedullary = rng.random() < 0.80
                encephalopathy_episode = rng.random() < 0.45
                levodopa_partial = rng.random() < 0.55
                age_onset = rng.randint(40, 72)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "skin_biopsy_inclusions": skin_biopsy_inclusions,
                    "dwi_corticomedullary_hyperintensity": dwi_corticomedullary,
                    "encephalopathy_episode": encephalopathy_episode,
                    "levodopa_partial_response": levodopa_partial,
                    "age_onset": age_onset,
                })

            elif gene == "PRKCG":
                action_tremor_prominent = rng.random() < 0.88
                cognitive_preserved = rng.random() < 0.85
                age_onset = rng.randint(8, 42)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "action_tremor_prominent": action_tremor_prominent,
                    "cognitive_preserved": cognitive_preserved,
                    "age_onset": age_onset,
                })

            elif gene == "ITPR1":
                pure_cerebellar = rng.random() < 0.92
                deletion_type = rng.random() < 0.40  # 40% deletions
                mlpa_required = deletion_type
                age_onset = rng.randint(10, 60)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "pure_cerebellar_no_extracerebellar": pure_cerebellar,
                    "deletion_not_missense": deletion_type,
                    "mlpa_required_for_diagnosis": mlpa_required,
                    "age_onset": age_onset,
                })

            elif gene == "ELOVL5":
                low_serum_dha = rng.random() < 0.93
                dha_supplement_benefit = rng.random() < 0.68
                pes_cavus = rng.random() < 0.60
                age_onset = rng.randint(35, 60)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "low_serum_dha": low_serum_dha,
                    "dha_supplementation_benefit": dha_supplement_benefit,
                    "pes_cavus": pes_cavus,
                    "age_onset": age_onset,
                })

    return all_patients


def overview():
    """Return atlas-level aggregate overview."""
    patients = _generate_cohort()
    mcp_sign = sum(1 for p in patients if p.get("mcp_hyperintensity"))
    acetazolamide_resp = sum(1 for p in patients if p.get("acetazolamide_responder"))
    four_ap_resp = sum(1 for p in patients if p.get("four_ap_responder"))
    canvas_cough = sum(1 for p in patients if p.get("chronic_dry_cough"))
    skin_biopsy_niid = sum(1 for p in patients if p.get("skin_biopsy_inclusions"))
    standard_missed = sum(1 for p in patients if p.get("standard_panel_missed") or p.get("standard_sequencing_missed"))
    low_dha = sum(1 for p in patients if p.get("low_serum_dha"))
    action_tremor = sum(1 for p in patients if p.get("action_tremor_prominent"))
    itpr1_deletions = sum(1 for p in patients if p.get("deletion_not_missense"))
    vestibular_areflexia = sum(1 for p in patients if p.get("bilateral_vestibular_areflexia"))
    return {
        "atlas": "Hereditary-Tremor-Ataxia-Atlas",
        "genes": [g["gene"] for g in TREMOR_ATAXIA_GENES],
        "total_patients": len(patients),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "fmr1_mcp_hyperintensity_patients": mcp_sign,
        "cacna1a_acetazolamide_responders": acetazolamide_resp,
        "fgf14_four_ap_responders": four_ap_resp,
        "rfc1_canvas_chronic_cough_patients": canvas_cough,
        "rfc1_vestibular_areflexia_patients": vestibular_areflexia,
        "notch2nlc_skin_biopsy_inclusions": skin_biopsy_niid,
        "standard_panels_missed_diagnosis_patients": standard_missed,
        "elovl5_low_serum_dha_patients": low_dha,
        "prkcg_action_tremor_patients": action_tremor,
        "itpr1_deletion_patients": itpr1_deletions,
    }


def breakdown():
    """Return per-gene breakdown with clinical details."""
    patients = _generate_cohort()
    result = {}
    for gene_data in TREMOR_ATAXIA_GENES:
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
        "genes": {g["gene"]: g["protein"] for g in TREMOR_ATAXIA_GENES},
        "glossary": {
            "FXTAS (Fragile X Tremor-Ataxia Syndrome)": (
                "Late-onset neurological syndrome in male FMR1 premutation carriers (55-200 CGG repeats); "
                "distinct from Fragile X Syndrome (full mutation >200 CGG, childhood onset, intellectual disability); "
                "FXTAS mechanism: FMR1 mRNA toxic gain-of-function — elevated mRNA sequesters RNA-binding proteins; "
                "Core features: intention tremor + cerebellar ataxia + cognitive decline (executive > memory); "
                "Associated: parkinsonism, autonomic neuropathy, white matter disease, peripheral neuropathy; "
                "MCP T2 hyperintensity pathognomonic on MRI; "
                "Penetrance in males: 17% at 50yr → 75% at >80yr; females 20% (heterozygous protection); "
                "No disease-modifying treatment; genetic counselling for female relatives critical (premutation carrier daughters)"
            ),
            "Episodic Ataxia Type 2 (EA2)": (
                "CACNA1A loss-of-function causing paroxysmal cerebellar attacks (hours-days); "
                "Interictal nystagmus (down-beat or horizontal) PERSISTS between attacks — pathognomonic DDx from EA1; "
                "Triggers: stress, exercise, alcohol, caffeine, illness; "
                "Acetazolamide Level B — 75-90% attack frequency reduction; "
                "4-aminopyridine (4-AP) alternative; "
                "Small SCA6-like progressive component develops with age in some EA2 patients; "
                "Avoid distinguishing terms: EA2 vs SCA6 are allelic — same gene, different mutation class"
            ),
            "SCA27B (FGF14-Related Ataxia)": (
                "Most common late-onset cerebellar ataxia now recognised worldwide; "
                "GAA-TTC intronic repeat expansion in FGF14; "
                "Standard exome, gene panel, and standard PCR ALL MISS this repeat; "
                "Long-read PCR or repeat-primed PCR required — must specify when ordering; "
                "4-aminopyridine (4-AP) specific treatment: restores Nav1.6 channel function in Purkinje cells; "
                "Downbeat nystagmus common; late onset; very slow progression; "
                "Consider in all unexplained late-onset ataxia with negative standard workup"
            ),
            "CANVAS (RFC1)": (
                "Cerebellar Ataxia + Neuropathy + Vestibular Areflexia Syndrome; "
                "Biallelic AAGGG pentanucleotide repeat expansion in RFC1 intron; "
                "Pure sensory neuropathy (no motor): absent sural nerve SNAP, preserved CMAPs on NCS; "
                "Bilateral vestibular areflexia: absent caloric response, abnormal vHIT; "
                "Chronic dry cough (unexplained) >70% — clue; "
                "Standard sequencing misses — repeat-primed PCR mandatory; "
                "Multiple pathogenic motifs (AAGGG, ACAGG, AGAGG) — check flanking sequence not just repeat size; "
                "No disease-modifying treatment; vestibular rehab"
            ),
            "NIID (NOTCH2NLC)": (
                "Neuronal Intranuclear Inclusion Disease; "
                "GGC repeat expansion >60 in NOTCH2NLC 5' UTR; polyglycine toxic gain-of-function; "
                "Skin biopsy: p62+/ubiquitin+ eosinophilic intranuclear inclusions in fibroblasts, sweat glands, adipocytes — PATHOGNOMONIC and ante-mortem accessible; "
                "DWI MRI: linear cortico-medullary junction hyperintensity — highly characteristic; "
                "Broad clinical spectrum: tremor, ataxia, parkinsonism, dementia, neuropathy, encephalopathy; "
                "Fever-triggered encephalopathy episodes — manage aggressively; "
                "Anticipation through paternal transmission (repeats expand)"
            ),
            "4-Aminopyridine (4-AP) in Cerebellar Ataxia": (
                "Broad-spectrum voltage-gated potassium channel blocker; "
                "Prolongs Purkinje cell action potentials → improves cerebellar output regularity; "
                "FDA-approved as Ampyra (sustained release 10 mg BD) for MS walking; "
                "Used off-label in EA2 and SCA27B/FGF14 (specific Nav1.6/FGF14 mechanism in SCA27B); "
                "Dose: 5-10 mg TDS (immediate release) or 10 mg BD (sustained release); "
                "Monitor: QTc prolongation (ECG baseline + 1 month); seizure threshold lowering; "
                "Contraindicated: history of seizures, severe renal impairment"
            ),
            "MLPA (Multiplex Ligation-dependent Probe Amplification)": (
                "Detects copy-number variants (deletions, duplications) in specific genes; "
                "Required for ITPR1/SCA15: deletions comprise 35-45% of pathogenic alleles; "
                "Standard sequencing panels ONLY detect point mutations (missense, splice, nonsense); "
                "When ordering SCA15/ITPR1 testing: explicitly request both sequencing AND MLPA; "
                "Applicable also to: PTEN, BRCA1, BRCA2, PRKN (large deletions common); "
                "Result format: dose quotient for each exon (0.5 = heterozygous deletion, 0 = homozygous)"
            ),
            "Repeat-Primed PCR (RP-PCR)": (
                "PCR technique using primer within the repeat itself — generates 'stutter' ladder on capillary electrophoresis; "
                "Detects presence of repeat expansion even when too large for standard PCR amplification; "
                "Required for: FGF14 (GAA-TTC), RFC1 (AAGGG), NOTCH2NLC (GGC), FMR1 (CGG); "
                "Does NOT give accurate sizing (use long-read PCR or Southern blot for exact repeat length); "
                "Gold standard for confirmation: long-read PCR (Oxford Nanopore, PacBio) or Southern blot; "
                "Order explicitly: 'repeat-primed PCR for FGF14/RFC1/NOTCH2NLC' — not included in standard exome/panel"
            ),
            "Acetazolamide (Carbonic Anhydrase Inhibitor)": (
                "First-line for EA2 (CACNA1A) — Level B evidence (attack reduction 75-90%); "
                "Mechanism uncertain: pH, membrane stabilisation, Na/K-ATPase; "
                "Dose: 250-500 mg BD (range 125-1000 mg/day); "
                "Side effects: paraesthesias, kidney stones (hydration important), hypokalemia; "
                "NOT effective for: EA1 (KCNA1), SCA14 (PRKCG), SCA15 (ITPR1), SCA38 (ELOVL5); "
                "Monthly serum potassium + bicarbonate monitoring; "
                "Contraindicated in sulphonamide allergy and severe renal impairment"
            ),
            "Downbeat Nystagmus": (
                "Eyes drift upward, corrective fast phase downward; "
                "Localises to: cervicomedullary junction, flocculus/paraflocculus, posterior fossa; "
                "Hereditary causes: FGF14 (SCA27B), CACNA1A (EA2/SCA6), ITPR1 (SCA15); "
                "Assessment: video-nystagmography (VNG) or infrared oculography in darkness; "
                "3,4-diaminopyridine or 4-AP can reduce downbeat nystagmus intensity; "
                "Head-up 20° tilting (Trendelenburg reverse) may transiently reduce downbeat nystagmus; "
                "Clonazepam, baclofen helpful in some"
            ),
            "DHA (Docosahexaenoic Acid) Deficiency (ELOVL5/SCA38)": (
                "22-carbon omega-3 PUFA essential for neuronal membrane function and synaptic vesicle dynamics; "
                "ELOVL5 elongase is required in the pathway from linolenic acid → EPA → DHA; "
                "SCA38 LOF → reduced DHA production → Purkinje cell membrane/synaptic dysfunction; "
                "Measure: fasting serum omega-3 fatty acid profile (DHA, EPA, DPA separately); "
                "Treatment: dietary DHA supplementation 1 g/day (algae-derived for vegetarians); "
                "Monitor: repeat serum DHA at 3 months; "
                "SARA score: clinical trial showed stabilisation vs progression without DHA"
            ),
        },
        "surveillance_protocols": {
            "FMR1 (FXTAS)": (
                "FMR1 CGG repeat PCR (premutation 55-200 — request specific assay not standard array); "
                "Brain MRI with T2/FLAIR (MCP hyperintensity); "
                "Neuropsychological battery (executive function, memory); "
                "EMG/NCS (peripheral neuropathy component); "
                "Autonomic screen (tilt table, bladder/bowel); "
                "Cascade FMR1 testing ALL female relatives (premutation carrier daughters → risk to grandsons); "
                "Falls prevention programme (physiotherapy); "
                "National Fragile X Foundation referral; "
                "Annual neurological review"
            ),
            "CACNA1A (EA2/SCA6)": (
                "CACNA1A full sequencing (point mutations) + CAG repeat assay (SCA6 ≥20 — SEPARATE test); "
                "Characterise: episodic (EA2) vs progressive (SCA6) vs mixed; "
                "EA2: acetazolamide 250 mg BD trial (K+ + bicarb monthly); "
                "SCA6: SARA score + cerebellar MRI annually; "
                "Oculomotor assessment (nystagmography — interictal nystagmus); "
                "Physiotherapy (balance); "
                "CACNA1A Registry"
            ),
            "FGF14 (SCA27B)": (
                "FGF14 repeat-primed PCR + long-read sequencing for GAA-TTC repeat sizing; "
                "Standard panels insufficient — explicitly request FGF14 repeat assay; "
                "Initiate 4-AP 5 mg OD → titrate to 10 mg TDS over 4 weeks; "
                "ECG baseline + 1 month (QTc); "
                "Downbeat nystagmus: VNG assessment; vestibular rehab; "
                "SARA score biannually; "
                "Cerebellar MRI at diagnosis + every 3 years; "
                "EuroAtaxia FGF14 registry"
            ),
            "RFC1 (CANVAS)": (
                "RFC1 repeat-primed PCR + flanking genotype (AAGGG vs AAAAG motif); "
                "NCS: confirm pure sensory neuropathy (absent SNAP, preserved CMAP); "
                "VHIT (video head impulse test) + caloric testing: bilateral vestibular areflexia; "
                "ENT/audiology (concurrent SNHL ~30%); "
                "Chronic cough: ACE-inhibitor cessation; aspiration risk (SLT); "
                "Vestibular rehabilitation; physiotherapy; "
                "Annual SARA + NCS + VHIT; "
                "EUROCANIC study registry"
            ),
            "NOTCH2NLC (NIID)": (
                "NOTCH2NLC GGC repeat PCR (repeat-primed PCR); "
                "Skin biopsy (punch biopsy axillary/abdominal — stain p62 + ubiquitin + EMA); "
                "Brain MRI DWI (cortico-medullary junction hyperintensity) + T2 (leukoencephalopathy); "
                "EMG/NCS (axonal sensorimotor neuropathy); "
                "Neuropsychological battery (executive, memory); "
                "Levodopa trial (parkinsonism component); "
                "Encephalopathy emergency plan (fever → urgent neurology); "
                "Annual neurological review; cascade testing"
            ),
            "PRKCG (SCA14)": (
                "PRKCG full gene sequencing; "
                "SARA score (gait/limb/speech/oculomotor) biannually; "
                "Propranolol / primidone for action tremor; "
                "Tremorographic assessment (spirography); "
                "MRI cerebellum at diagnosis + every 5 years; "
                "Physiotherapy (balance, Frenkel exercises); "
                "Driving assessment (tremor-dependent); "
                "SCA14 patient registry (EUROSCA)"
            ),
            "ITPR1 (SCA15)": (
                "ITPR1 SEQUENCING + MLPA (both mandatory — deletions 35-45% of alleles); "
                "State explicitly when ordering: 'SCA15/ITPR1 requires MLPA not just sequencing'; "
                "SARA score biannually (very slow progression); "
                "MRI cerebellum at diagnosis + every 5 years; "
                "Vestibular assessment (nystagmus); "
                "Physiotherapy; driving assessment; "
                "EUROSCA registry"
            ),
            "ELOVL5 (SCA38)": (
                "ELOVL5 gene sequencing; "
                "Fasting serum omega-3 fatty acid profile (DHA, EPA, DPA); "
                "Initiate DHA 1 g/day supplementation (algae-derived or fish oil); "
                "Repeat serum DHA at 3 months (confirm biomarker correction); "
                "SARA score biannually; "
                "Foot assessment (pes cavus — orthotics); "
                "NCS (sensory neuropathy component); "
                "MRI cerebellum; "
                "EuroAtaxia SCA38 registry"
            ),
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"FMR1 MCP hyperintensity: {ov['fmr1_mcp_hyperintensity_patients']}")
    print(f"CACNA1A acetazolamide responders: {ov['cacna1a_acetazolamide_responders']}")
    print(f"FGF14 4-AP responders: {ov['fgf14_four_ap_responders']}")
    print(f"RFC1 CANVAS cough: {ov['rfc1_canvas_chronic_cough_patients']}")
    print(f"NOTCH2NLC skin biopsy: {ov['notch2nlc_skin_biopsy_inclusions']}")
    print(f"Standard panels missed: {ov['standard_panels_missed_diagnosis_patients']}")
    print(f"ELOVL5 low DHA: {ov['elovl5_low_serum_dha_patients']}")
