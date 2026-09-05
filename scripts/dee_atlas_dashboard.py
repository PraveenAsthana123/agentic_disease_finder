#!/usr/bin/env python3
"""DEE-Atlas — Complete 8-Gene Developmental and Epileptic Encephalopathy Atlas
SCN1A  (Nav1.1; 2009 aa; 1p36.11; Dravet syndrome / GEFS+; LOF → inhibitory neuron loss; AVOID CBZ/OXC/PHT/LTG; stiripentol+VPA+clobazam; fenfluramine FDA 2020) ·
KCNQ2  (Kv7.2; 872 aa; 20q13.33; KCNQ2-DEE GOF severe / BFNS LOF benign neonatal; neonatal onset; carbamazepine LOF; quinidine GOF) ·
CDKL5  (CDKL5 kinase; 1030 aa; Xp22.13; CDKL5 deficiency disorder CDD; early infantile spasms <5 mo; Rett-like; ganaxolone FDA 2022) ·
ARX    (Aristaless-related homeobox; 562 aa; Xp21.3; EIEE1/Ohtahara males; West→LGS; X-linked; poly-A expansion → Proud syndrome) ·
STXBP1 (Munc18-1; 594 aa; 20p13; EIEE4/Ohtahara → West → LGS; vesicle fusion; hypotonia+dystonia; suppression-burst) ·
PCDH19 (Protocadherin 19; 1148 aa; Xq22.1; CELLULAR INTERFERENCE; hemizygous males UNAFFECTED; het females AFFECTED; fever-triggered clusters) ·
SCN8A  (Nav1.6; 1980 aa; 12q13.13; EIEE13/DEE13; GOF; HIGH SUDEP RISK; quinidine GOF-specific) ·
GRIN2A (GluN2A; 1464 aa; 16p13.2; epilepsy-aphasia spectrum; CSWS/ESES; LKS acquired aphasia; sulthiame first-line)
320-patient aggregate cohort (8 × 40, seeds 1070–1077)
"""

import random

SEED_BASE = 1070

DEE_GENES = [
    # ── SCN1A — Dravet syndrome / GEFS+ ─────────────────────────────────────
    {
        "gene": "SCN1A", "protein": "Voltage-Gated Sodium Channel Nav1.1 (SCN1A)",
        "alias": "SCN1A; OMIM gene 182389; 1p36.11; ~2009 aa; Dravet syndrome (OMIM #607208); GEFS+ (OMIM #604233); autosomal dominant / de novo LOF; inhibitory neuron loss; AVOID CBZ/OXC/PHT/LTG",
        "aa": "~2009 aa", "kDa": "~228 kDa",
        "mechanism": (
            "SCN1A encodes Nav1.1, the voltage-gated sodium channel alpha-subunit predominantly expressed "
            "in inhibitory GABAergic interneurons (parvalbumin-positive fast-spiking interneurons) of the "
            "cerebral cortex, hippocampus, and cerebellum. "
            "DRAVET MECHANISM: heterozygous LOF mutations in SCN1A → Nav1.1 haploinsufficiency → "
            "selective loss of sodium current in inhibitory interneurons (interneurons depend on Nav1.1 more "
            "than excitatory neurons for high-frequency firing) → GABAergic interneuron failure → "
            "loss of inhibition → network disinhibition → seizure threshold markedly lowered. "
            "EXCITATORY NEURONS: retain Nav1.2/Nav1.6 which compensate → excitatory drive preserved → "
            "net result: disinhibited excitatory network → seizures. "
            "TEMPERATURE SENSITIVITY: Nav1.1 function further impaired at elevated temperatures → "
            "febrile seizures and temperature-sensitive seizures are cardinal Dravet features. "
            "DRUG PARADOX (CRITICAL): sodium channel BLOCKERS (CBZ, OXC, PHT, LTG) further suppress "
            "residual Nav1.1 function in inhibitory interneurons → WORSEN Dravet syndrome paradoxically. "
            "GEFS+ SPECTRUM: less severe SCN1A variants → GEFS+ (generalised epilepsy with febrile "
            "seizures plus) — milder phenotype, good prognosis, same gene family. "
            "GENOTYPE-PHENOTYPE: truncating/splice = severe Dravet; missense in voltage-sensing domain = "
            "Dravet to GEFS+ spectrum; missense in linker regions = often GEFS+."
        ),
        "disease_type": "Dravet Syndrome / GEFS+ (AD de novo LOF)",
        "locus": "1p36.11", "omim_gene": 182389, "omim_disease": 607208,
        "inheritance": (
            "AUTOSOMAL DOMINANT — predominantly DE NOVO (>90% of Dravet). Inherited in GEFS+ families "
            "(milder spectrum). LOF mutations (truncating, splice-site, missense in functional domains). "
            "MOSAICISM: ~5–10% of Dravet parents carry somatic or germline mosaic SCN1A variant — "
            "recurrence risk in siblings not zero even if parents test negative. "
            "Genetic testing: full gene sequencing + MLPA (deletions account for ~5% Dravet). "
            "Functional assay or voltage-clamp studies for variants of uncertain significance."
        ),
        "phenotype": (
            "DRAVET SYNDROME: "
            "ONSET: First year of life (usually 5–8 months); FEBRILE STATUS EPILEPTICUS is the TYPICAL FIRST EVENT "
            "(prolonged febrile seizure >30 min in a previously normal infant). "
            "SEIZURE TYPES: prolonged febrile hemiclonic seizures → myoclonic → absence → focal → "
            "convulsive status epilepticus. "
            "TEMPERATURE SENSITIVITY: warm bath, fever, exercise → seizure trigger (pathognomonic cluster). "
            "DEVELOPMENT: initially NORMAL, then REGRESSION after seizure onset — intellectual disability "
            "(moderate-severe), ataxia, hyperkinesis. "
            "EEG: early normal → generalised/multifocal spike-wave; photosensitivity in some. "
            "DRUG SENSITIVITY: sodium channel blockers (CBZ, OXC, PHT, LTG) WORSEN seizure control — "
            "critical prescribing alert. "
            "GEFS+: febrile seizures beyond age 6 years; generalised seizures; good developmental outcome; "
            "seizures usually remit in adolescence. "
            "SUDEP RISK: elevated in Dravet (1–2% per year); multifactorial including nocturnal seizures."
        ),
        "treatment_options": [
            "Stiripentol + VPA + clobazam: TRIPLE THERAPY — first-line for Dravet syndrome (EU/FDA approved); "
            "stiripentol is a positive GABA-A modulator and CYP inhibitor that elevates clobazam metabolite; "
            "combination significantly reduces convulsive seizures",
            "Fenfluramine (Fintepla): FDA approved 2020 for Dravet syndrome; "
            "serotonin-releasing agent + sigma-1 receptor agonist; reduces convulsive seizures by ~54% in trials; "
            "REQUIRES cardiac monitoring (echocardiogram 6-monthly — historical valvulopathy risk at high doses)",
            "Cannabidiol (Epidiolex): FDA approved 2018 for Dravet; plant-derived CBD; "
            "reduces convulsive seizures; mechanism uncertain (possible TRPV1, GPR55 modulation); "
            "monitor liver function (LFT); interaction with VPA/clobazam",
            "Clobazam: 1,5-benzodiazepine; adjunctive; reduces seizure frequency; less tolerance than 1,4-BZDs; "
            "useful long-term in Dravet",
            "Sodium valproate (VPA): broad-spectrum; adjunctive in Dravet; monitor for hyperammonaemia "
            "and LFT; avoid in females of childbearing potential (teratogenicity, PCOS risk)",
            "Ketogenic diet: effective adjunctive therapy in Dravet; can reduce seizures by >50% in ~50% "
            "of patients; recommend when ≥2 ASMs failed; classic 4:1 or modified Atkins",
            "Topiramate: adjunctive; some evidence in Dravet; mechanism includes sodium channel blockade "
            "but does not worsen like CBZ/OXC (possibly because of additional GABA-potentiation)",
            "AVOID: carbamazepine (CBZ), oxcarbazepine (OXC), phenytoin (PHT), lamotrigine (LTG) — "
            "ALL worsen Dravet by blocking residual Nav1.1 in inhibitory neurons; "
            "AVOID vigabatrin (may worsen myoclonic seizures in Dravet)",
        ],
        "key_ddx": [
            "GEFS+ (milder SCN1A) — same gene; febrile seizures beyond 6y; no regression; better prognosis",
            "KCNQ2-DEE (GOF) — neonatal onset; suppression-burst; different seizure semiology",
            "Doose syndrome (MAE) — myoclonic-atonic seizures; normal early development; different EEG",
            "CDKL5 deficiency disorder — infantile spasms <5mo; female-predominant; Rett-like features",
            "Lennox-Gastaut (multiple causes) — tonic/atonic/absence triad; onset usually after 1 year; "
            "no temperature-sensitive seizures",
        ],
        "onset_range_y": (0.4, 1.0),
        "sex_female_prob": 0.50,
        "eeg_pattern": "Multifocal spikes / generalised spike-wave / photosensitivity",
        "seizure_type": "Febrile hemiclonic status / myoclonic / absence / convulsive status",
        "sudep_risk": "Elevated (1-2%/y)",
        "targeted_therapy_available": True,
        "severity_dist": {"Severe": 0.55, "Moderate": 0.35, "Mild": 0.10},
        "seizure_free_rate": 0.12,
        "status_hx_rate": 0.72,
        "drug_error_rate": 0.08,
        "first_line_drug": "Stiripentol + VPA + Clobazam",
        "critical_avoid": "ABSOLUTELY AVOID: CBZ, OXC, PHT, LTG (worsen Dravet by blocking residual Nav1.1 in inhibitory neurons)",
    },
    # ── KCNQ2 — KCNQ2-DEE / BFNS ──────────────────────────────────────────
    {
        "gene": "KCNQ2", "protein": "Voltage-Gated Potassium Channel Kv7.2 (KCNQ2)",
        "alias": "KCNQ2; OMIM gene 602235; 20q13.33; ~872 aa; KCNQ2-DEE (OMIM #613720, GOF severe) / BFNS Benign Familial Neonatal Seizures (OMIM #121200, LOF mild self-limiting); neonatal onset; carbamazepine (LOF); quinidine (GOF)",
        "aa": "~872 aa", "kDa": "~97 kDa",
        "mechanism": (
            "KCNQ2 encodes Kv7.2, a voltage-gated potassium channel subunit that co-assembles with Kv7.3 "
            "(KCNQ3) to form the M-current (IKM) — a slowly activating, non-inactivating potassium current "
            "critical for neuronal repolarisation and suppression of repetitive firing. "
            "M-CURRENT FUNCTION: activated at subthreshold voltages → dampens repetitive action potential firing → "
            "natural 'brake' on neuronal excitability. Highly expressed in hippocampus, neocortex, "
            "and spinal cord at the axon initial segment. "
            "BIMODAL PHENOTYPE — GENOTYPE-DRIVEN: "
            "LOF (BFNS): haploinsufficiency → reduced IKM → mild neonatal seizures → self-limiting "
            "(spontaneous remission by 2–6 months in most); normal development; autosomal dominant. "
            "GOF (KCNQ2-DEE): gain-of-function variants → constitutively active or altered Kv7.2 → "
            "paradoxical persistent depolarisation (M-current normally repolarises; constitutive GOF "
            "disrupts firing pattern; membrane bistability) → neonatal DEE; suppression-burst EEG; "
            "severe intellectual disability; de novo; NOT self-limiting. "
            "TREATMENT IMPLICATION: CBZ reduces neuronal firing → beneficial in LOF BFNS (compensates "
            "for loss of IKM brake); QUINIDINE blocks the constitutively active GOF Kv7.2 channel → "
            "potentially beneficial in GOF DEE (opposite mechanism to BFNS)."
        ),
        "disease_type": "KCNQ2-DEE (GOF, severe) / BFNS (LOF, benign neonatal, AD)",
        "locus": "20q13.33", "omim_gene": 602235, "omim_disease": 613720,
        "inheritance": (
            "GOF KCNQ2-DEE: DE NOVO dominant; severe. "
            "LOF BFNS: AUTOSOMAL DOMINANT (familial or de novo); benign self-limiting. "
            "Critical clinical rule: not all KCNQ2 mutations are the same — "
            "variant functional classification (GOF vs LOF) MANDATORY before choosing treatment. "
            "Electrophysiology studies (patch clamp) or established variant databases required. "
            "De novo + neonatal onset + suppression-burst EEG = KCNQ2-DEE until proven otherwise."
        ),
        "phenotype": (
            "KCNQ2-DEE (GOF, severe): "
            "ONSET: neonatal (first days to weeks of life). "
            "EEG: SUPPRESSION-BURST (high-voltage bursts alternating with electrocerebral suppression) — "
            "hallmark neonatal EEG in KCNQ2-DEE and other Ohtahara-spectrum conditions. "
            "SEIZURES: tonic seizures, focal motor seizures, apnoeic spells; status epilepticus common. "
            "DEVELOPMENT: severe intellectual disability, hypotonia, spasticity; no expected normalisation. "
            "EVOLUTION: suppression-burst → multifocal spikes → hypsarrhythmia (some); "
            "Lennox-Gastaut evolution in some cases. "
            "BFNS (LOF, benign): "
            "ONSET: neonatal (first 3 days); self-limiting (remits by 2–6 months in >90%). "
            "EEG: multifocal spikes; no suppression-burst. "
            "DEVELOPMENT: NORMAL (distinguishes from DEE). "
            "FAMILY HISTORY: often positive (autosomal dominant family); "
            "PROGNOSIS: excellent; small risk of febrile seizures later."
        ),
        "treatment_options": [
            "Carbamazepine (CBZ): DRUG OF CHOICE for LOF BFNS — reduces neuronal hyperexcitability "
            "to compensate for reduced IKM; rapid seizure control; short course (2-6 months); "
            "taper when seizure-free",
            "Phenobarbital: alternative first-line in neonates (especially when KCNQ2 status unknown); "
            "sedating; effective for neonatal seizures broadly",
            "Quinidine: sodium/potassium channel blocker; used in GOF KCNQ2-DEE (compassionate use); "
            "blocks constitutively active Kv7.2 GOF channel; case series show seizure reduction; "
            "NOT first-line without functional confirmation of GOF",
            "High-dose phenobarbital + phenytoin: second-line for neonatal status in KCNQ2-DEE "
            "when genetic result pending",
            "Levetiracetam: adjunctive in KCNQ2-DEE; well-tolerated neonatal profile; "
            "less specific but useful for add-on",
            "ACTH/prednisolone: for hypsarrhythmia evolution (West syndrome) in KCNQ2-DEE progression",
            "Genetic functional testing: early classification GOF vs LOF is CRITICAL for treatment selection; "
            "variant databases (ClinVar, KCNQ2 gene-specific databases) + electrophysiology consultation",
            "Family counselling: BFNS families — excellent prognosis, AD, re-assure on development; "
            "GOF DEE families — severe prognosis, de novo, low recurrence risk (<1% in siblings)",
        ],
        "key_ddx": [
            "SCN1A Dravet — onset 5–8 mo; febrile trigger; temperature-sensitive; not neonatal",
            "STXBP1-EIEE4 — neonatal suppression-burst; no bimodal GOF/LOF split; Munc18-1 mechanism",
            "ARX EIEE1 — X-linked males; neonatal; poly-A expansion → Proud syndrome; ARX gene",
            "Pyridoxine-dependent epilepsy (ALDH7A1) — dramatic pyridoxine response; CSF amino acids",
            "Hypoxic-ischaemic encephalopathy (HIE) — perinatal history; MRI pattern; not genetic",
        ],
        "onset_range_y": (0.0, 0.1),
        "sex_female_prob": 0.50,
        "eeg_pattern": "Suppression-burst (DEE) / Multifocal spikes (BFNS)",
        "seizure_type": "Tonic neonatal / focal motor / apnoeic spells",
        "sudep_risk": "Moderate (DEE) / Low (BFNS)",
        "targeted_therapy_available": True,
        "severity_dist": {"Severe": 0.60, "Moderate": 0.25, "Mild": 0.15},
        "seizure_free_rate": 0.18,
        "status_hx_rate": 0.65,
        "drug_error_rate": 0.07,
        "first_line_drug": "Carbamazepine (LOF-BFNS) / Quinidine (GOF-DEE)",
        "critical_avoid": "AVOID applying CBZ/OXC in GOF DEE (worsens); AVOID quinidine in LOF BFNS (unnecessary); classify GOF vs LOF BEFORE treatment",
    },
    # ── CDKL5 — CDKL5 Deficiency Disorder (CDD) ────────────────────────────
    {
        "gene": "CDKL5", "protein": "Cyclin-Dependent Kinase-Like 5 (CDKL5)",
        "alias": "CDKL5 (STK9); OMIM gene 300203; Xp22.13; ~1030 aa; CDKL5 Deficiency Disorder CDD (OMIM #300672); X-linked; early infantile spasms onset <5 mo; Rett-like features WITHOUT MECP2; ganaxolone FDA 2022 FIRST CDD-specific therapy",
        "aa": "~1030 aa", "kDa": "~115 kDa",
        "mechanism": (
            "CDKL5 encodes cyclin-dependent kinase-like 5, a serine-threonine kinase with an N-terminal "
            "kinase domain and a long C-terminal regulatory domain. CDKL5 is highly expressed in neurons "
            "(hippocampus, cortex, cerebellum) and is regulated by neuronal activity. "
            "CDKL5 SUBSTRATES AND FUNCTION: phosphorylates ARC (activity-regulated cytoskeleton-associated protein), "
            "MAP1S, NGL-1, and SHANK3 at synapses → regulates dendritic morphology, spine density, "
            "and excitatory synapse maturation. "
            "CDD MECHANISM: CDKL5 LOF mutations → loss of kinase activity → "
            "disrupted synaptic signalling → immature dendritic arbors → aberrant network connectivity → "
            "early infantile epileptic encephalopathy + Rett-like neurodevelopmental features. "
            "NOT CLASSIC RETT: CDKL5 mutations cause CDD — a DISTINCT disorder from MECP2 (classic Rett); "
            "CDD onset is EARLIER (spasms <5 months vs Rett onset after 6–18 months of normal development). "
            "GENDER EFFECT: X-linked; hemizygous males — usually more severely affected; "
            "heterozygous females — intermediate severity (Lyon inactivation mosaic); "
            "both sexes affected (unlike ARX where females usually unaffected). "
            "GANAXOLONE MECHANISM: synthetic analogue of neurosteroid allopregnanolone → positive allosteric "
            "modulator of synaptic and extrasynaptic GABA-A receptors → reduces seizure frequency; "
            "FDA approved March 2022 — FIRST-EVER CDD-specific therapy."
        ),
        "disease_type": "CDKL5 Deficiency Disorder / CDD (X-linked)",
        "locus": "Xp22.13", "omim_gene": 300203, "omim_disease": 300672,
        "inheritance": (
            "X-LINKED. Males: hemizygous — often severely affected (full LOF of single X allele). "
            "Females: heterozygous — affected (unlike typical XLR where females are unaffected carriers); "
            "females often less severely affected than males due to mosaicism from X-inactivation. "
            "DE NOVO in ~90% of cases (new mutation in the patient). "
            "Familial cases reported (mother as mosaic carrier). "
            "FEMALES PREDOMINATE in clinical series (70–80% female) — because severely affected males "
            "may not survive to diagnosis or females are more often referred. "
            "Genetic testing: full SCN1A and CDKL5 gene sequencing + MLPA (deletions common)."
        ),
        "phenotype": (
            "CDKL5 DEFICIENCY DISORDER (CDD): "
            "ONSET: infantile spasms (West syndrome-like) typically before 5 months of age "
            "(median ~5 months; mean 4.5 months) — EARLIER than classic West syndrome. "
            "EEG: hypsarrhythmia (chaotic high-amplitude slow waves + spikes) during spasms phase; "
            "multifocal spikes persistently; normal background early. "
            "RETT-LIKE FEATURES: hand stereotypies (hand-wringing, hand-mouthing), absent purposeful hand use, "
            "absent speech — BUT DIFFERENT ONSET PATTERN from classic Rett (no 6–18 month normal period). "
            "DEVELOPMENT: severe global delay; minimal language; limited hand function; "
            "hand stereotypies in 70–80%; gaze contact often preserved (unlike Rett). "
            "SEIZURES PERSIST throughout life in most (polytherapy required); "
            "tonic seizures, myoclonic-tonic, hypermotor, focal — variable. "
            "AUTONOMIC: drooling, constipation, cold feet common. "
            "MORTALITY: elevated; SUDEP risk reported; aspiration pneumonia."
        ),
        "treatment_options": [
            "Ganaxolone (Marinus Pharmaceuticals): FDA approved March 2022 for CDD in patients ≥2 years; "
            "FIRST CDD-SPECIFIC APPROVED THERAPY; positive GABA-A modulator (neurosteroid analogue); "
            "reduces motor seizures by ~30%; oral or IV formulation; dose: 1800 mg/day in 3 divided doses",
            "ACTH (adrenocorticotropic hormone) or prednisolone: first-line for infantile spasms (West syndrome phase); "
            "high-dose ACTH 150 IU/m²/day × 2 weeks, then taper; most effective for spasm cessation",
            "Vigabatrin: first-line for infantile spasms (especially TSC-associated, but used in CDD); "
            "GABA-transaminase inhibitor → irreversible; visual field constriction risk (ophthalmology monitoring mandatory)",
            "Valproate + clobazam: adjunctive combination; broad-spectrum; useful for ongoing seizures post-spasms",
            "Topiramate: adjunctive; some evidence in CDD; multiple mechanisms",
            "Ketogenic diet: effective adjunctive for refractory CDD seizures; ~50% responders in open-label series; "
            "classic 4:1 or modified Atkins; dietitian required",
            "Cannabidiol (Epidiolex): off-label adjunctive; evidence emerging from expanded access programs",
            "Multidisciplinary care: physiotherapy (hypotonia, motor), speech therapy (communication aids), "
            "OT (hand function), dietitian (feeding), ophthalmology (nystagmus, cortical visual impairment)",
        ],
        "key_ddx": [
            "Classic Rett syndrome (MECP2) — female only; 6–18 months NORMAL development then regression; later onset than CDD",
            "West syndrome (cryptogenic or other causes) — spasms onset 4–7 months; hypsarrhythmia; multiple causes",
            "SCN1A Dravet — febrile status as first event; temperature-sensitive; Nav1.1 mechanism",
            "STXBP1-EIEE4 — neonatal suppression-burst; presynaptic mechanism; not X-linked",
            "Angelman syndrome (UBE3A) — happy demeanour; ataxia; microcephaly; different EEG",
        ],
        "onset_range_y": (0.0, 0.42),
        "sex_female_prob": 0.78,
        "eeg_pattern": "Hypsarrhythmia (spasms phase) / Multifocal spikes (persistent)",
        "seizure_type": "Infantile spasms / tonic / myoclonic-tonic / focal hypermotor",
        "sudep_risk": "Elevated",
        "targeted_therapy_available": True,
        "severity_dist": {"Severe": 0.65, "Moderate": 0.30, "Mild": 0.05},
        "seizure_free_rate": 0.05,
        "status_hx_rate": 0.55,
        "drug_error_rate": 0.06,
        "first_line_drug": "Ganaxolone (CDD-specific FDA 2022) / ACTH or vigabatrin (spasms)",
        "critical_avoid": "Do NOT use lamotrigine alone (may worsen myoclonic component); AVOID missing ganaxolone as CDD-specific option",
    },
    # ── ARX — EIEE1 / Ohtahara / West / LGS ───────────────────────────────
    {
        "gene": "ARX", "protein": "Aristaless-Related Homeobox Transcription Factor (ARX)",
        "alias": "ARX; OMIM gene 300382; Xp21.3; ~562 aa; EIEE1/Ohtahara syndrome (OMIM #308350); West syndrome; LGS; X-linked; males severely affected; POLY-A expansion → Proud syndrome (dystonia + agenesis of CC)",
        "aa": "~562 aa", "kDa": "~63 kDa",
        "mechanism": (
            "ARX encodes Aristaless-related homeobox, a paired-type homeodomain transcription factor "
            "critical for interneuron development and migration. ARX is expressed in GABAergic "
            "interneuron progenitors in the ganglionic eminences during embryonic brain development. "
            "ARX FUNCTION: regulates differentiation, tangential migration, and survival of cortical "
            "interneurons (parvalbumin+, somatostatin+) and striatal interneurons → also critical for "
            "pancreatic glucagon-producing alpha-cell development. "
            "CDD MECHANISM: ARX LOF → loss of interneuron progenitor specification → "
            "reduced cortical GABAergic interneurons → disinhibition → early-onset epileptic encephalopathy. "
            "MUTATION TYPE DETERMINES PHENOTYPE (CRITICAL): "
            "TRUNCATING/SPLICE MUTATIONS → EIEE1/Ohtahara (most severe; neonatal suppression-burst). "
            "POLY-A EXPANSION (GCG repeat in exon 2, first or second polyalanine tract): "
            "  - FIRST POLY-A expansion (12→17+ alanines) → ISSX/West syndrome + intellectual disability. "
            "  - SECOND POLY-A expansion → PRTS (Partington syndrome: intellectual disability + dystonia). "
            "  - PROUD SYNDROME: point mutations + poly-A → intellectual disability + AGENESIS OF CORPUS CALLOSUM "
            "    + dystonia (specific ARX mutation phenotype). "
            "MISSENSE in homeodomain → Lissencephaly with ambiguous genitalia (XLAG — X-linked lissencephaly). "
            "GENDER: X-LINKED — hemizygous males severely affected; females usually ASYMPTOMATIC carriers "
            "(extremely rarely mildly affected)."
        ),
        "disease_type": "EIEE1 / Ohtahara Syndrome / West / LGS Spectrum (X-linked ARX)",
        "locus": "Xp21.3", "omim_gene": 300382, "omim_disease": 308350,
        "inheritance": (
            "X-LINKED. Hemizygous males — severely affected (Ohtahara, West, LGS, XLAG). "
            "Heterozygous females — usually ASYMPTOMATIC carriers (extremely rarely mild intellectual disability). "
            "DE NOVO or X-linked familial. "
            "IMPORTANT: multiple DIFFERENT phenotypes from different ARX mutation types in the SAME gene — "
            "full sequencing + polyalanine tract sizing MANDATORY. "
            "Female carrier testing critical for recurrence risk counselling (X-linked recurrence: "
            "50% of sons affected, 50% of daughters carriers)."
        ),
        "phenotype": (
            "EIEE1 / OHTAHARA SYNDROME (truncating ARX): "
            "ONSET: neonatal (first days of life). "
            "EEG: SUPPRESSION-BURST — alternating bursts of high-amplitude polyspike-slow waves "
            "with near-flat background suppression; continuous, including during sleep. "
            "SEIZURES: tonic spasms (erratic, frequent, 300–400/day); myoclonic; focal. "
            "DEVELOPMENT: profound intellectual disability; hypotonia; virtually no development. "
            "PROGRESSION: Ohtahara → West syndrome (hypsarrhythmia) → Lennox-Gastaut (diffuse slow spike-wave). "
            "WEST SYNDROME (poly-A expansion): "
            "Onset 4–8 months; hypsarrhythmia; infantile spasms; intellectual disability. "
            "PROUD SYNDROME: intellectual disability + AGENESIS OF CORPUS CALLOSUM + DYSTONIA — "
            "ARX-specific triad; corpus callosum absent on MRI. "
            "NO TARGETED THERAPY: currently only symptomatic treatment."
        ),
        "treatment_options": [
            "ACTH: first-line for spasms (West syndrome phase); "
            "high-dose ACTH 150 IU/m²/day tapering protocol; most effective spasm cessation in West syndrome",
            "Vigabatrin: first-line for spasms (alternative or combined with ACTH); "
            "ophthalmology monitoring mandatory (visual field constriction); preferred in TSC-West",
            "Phenobarbital: first-line for neonatal seizures (Ohtahara phase); widely available; "
            "can reduce tonic spasm frequency",
            "Valproate: broad-spectrum adjunctive; useful across seizure types; "
            "monitor LFT and haematology",
            "Clobazam: adjunctive benzodiazepine; reduces seizure frequency; tolerability good",
            "Topiramate / zonisamide: adjunctive; some evidence for LGS-evolved phenotype",
            "Corpus callosotomy: surgical option for drop attacks in LGS-evolved ARX encephalopathy",
            "Genetic counselling: X-linked; female carriers need testing; "
            "prenatal diagnosis by CVS/amniocentesis; establish exact ARX mutation type for phenotype prediction; "
            "NO disease-modifying therapy currently available",
        ],
        "key_ddx": [
            "STXBP1-EIEE4 — neonatal suppression-burst; autosomal de novo; presynaptic mechanism; both sexes",
            "KCNQ2-DEE — neonatal; M-current; bimodal GOF/LOF; carbamazepine/quinidine options",
            "Pyridoxine-dependent epilepsy (ALDH7A1) — pyridoxine IV trial in all neonatal refractory seizures",
            "Hypoxic-ischaemic encephalopathy (HIE) — perinatal history; MRI pattern; aEEG monitoring",
            "Aicardi syndrome (MECP2-related in some) — females; agenesis CC + infantile spasms; "
            "chorioretinal lacunae (pathognomonic)",
        ],
        "onset_range_y": (0.0, 0.1),
        "sex_female_prob": 0.05,
        "eeg_pattern": "Suppression-burst (Ohtahara) / Hypsarrhythmia (West) / Diffuse slow spike-wave (LGS)",
        "seizure_type": "Tonic neonatal spasms / infantile spasms / tonic-atonic / myoclonic",
        "sudep_risk": "High (Ohtahara, severe DEE)",
        "targeted_therapy_available": False,
        "severity_dist": {"Severe": 0.80, "Moderate": 0.18, "Mild": 0.02},
        "seizure_free_rate": 0.04,
        "status_hx_rate": 0.78,
        "drug_error_rate": 0.05,
        "first_line_drug": "ACTH or vigabatrin (spasms) / Phenobarbital (neonatal)",
        "critical_avoid": "No targeted therapy; AVOID delay in ACTH for spasms; MANDATORY: size ARX polyalanine tracts to predict phenotype",
    },
    # ── STXBP1 — EIEE4 / Ohtahara → West → LGS ─────────────────────────────
    {
        "gene": "STXBP1", "protein": "Syntaxin-Binding Protein 1 / Munc18-1 (STXBP1)",
        "alias": "STXBP1 (MUNC18-1); OMIM gene 602926; 20p13; ~594 aa; EIEE4/Ohtahara Syndrome (OMIM #612164); AD de novo LOF; vesicle fusion; GABA+glutamate release disrupted; movement disorder prominent; progression Ohtahara→West→LGS",
        "aa": "~594 aa", "kDa": "~68 kDa",
        "mechanism": (
            "STXBP1 encodes syntaxin-binding protein 1 (Munc18-1), a member of the SM (Sec1/Munc18) "
            "protein family critical for regulated neurotransmitter exocytosis. "
            "MUNC18-1 FUNCTION: interacts with syntaxin-1 (STX1A/STX1B) and the SNARE complex "
            "(syntaxin-1/SNAP-25/synaptobrevin-2) to facilitate synaptic vesicle docking and priming → "
            "essential for both inhibitory (GABA) and excitatory (glutamate) synaptic vesicle fusion. "
            "STXBP1-EIEE4 MECHANISM: de novo LOF mutations (haploinsufficiency) → Munc18-1 protein "
            "reduced by ~50% (haploinsufficiency) → impaired SNARE complex assembly → "
            "reduced BOTH GABA AND GLUTAMATE release → but net effect: inhibitory > excitatory deficit "
            "→ seizure disinhibition in developing brain + additional presynaptic vesicle trafficking disorder. "
            "MOVEMENT DISORDER: Munc18-1 also critical in striatal dopaminergic synapses → "
            "MOVEMENT DISORDER (tremor, dystonia, hyperkinesia) is a prominent co-feature of STXBP1-EIEE4. "
            "PROTEIN MISFOLDING: many missense mutations cause Munc18-1 misfolding + ER stress "
            "→ protein degradation (haploinsufficiency even from missense). "
            "EEG EVOLUTION: neonatal suppression-burst (Ohtahara) → hypsarrhythmia (West) → "
            "diffuse slow spike-wave (Lennox-Gastaut) as the child grows."
        ),
        "disease_type": "EIEE4 / Ohtahara Syndrome → West → LGS Progression (AD de novo STXBP1)",
        "locus": "20p13", "omim_gene": 602926, "omim_disease": 612164,
        "inheritance": (
            "AUTOSOMAL DOMINANT — DE NOVO (>95% of STXBP1-EIEE4). "
            "Both sexes equally affected (autosomal). "
            "Parental mosaicism: rare but reported — parental testing recommended. "
            "Recurrence risk: <1% for parents (de novo dominant), but parental testing first. "
            "STXBP1 mutations account for ~10–20% of all Ohtahara syndrome (neonatal suppression-burst DEE). "
            "Neonatal suppression-burst + hypotonia + dystonia + de novo = STXBP1 until proven otherwise."
        ),
        "phenotype": (
            "STXBP1-EIEE4: "
            "ONSET: neonatal (first days of life) or early infantile (first weeks). "
            "EEG: SUPPRESSION-BURST neonatal → hypsarrhythmia (West syndrome phase 3–12 months) → "
            "diffuse slow spike-wave (LGS evolution). "
            "SEIZURES: tonic spasms, myoclonic, focal clonic, epileptic spasms (West phase). "
            "DEVELOPMENT: severe intellectual disability; absent language; minimal purposeful movement. "
            "MOVEMENT DISORDER: PROMINENT — tremor (fine postural/action), dystonia, hyperkinesia; "
            "often the most disabling non-epileptic feature in older children. "
            "HYPOTONIA: universal at birth. "
            "AUTONOMIC: poor feeding; temperature dysregulation. "
            "NO TARGETED THERAPY: currently symptomatic only. "
            "SEIZURE EVOLUTION: often initial improvement in spasm frequency in West phase, "
            "then LGS pattern — never seizure-free in most."
        ),
        "treatment_options": [
            "ACTH: first-line for infantile spasms (West syndrome evolution); "
            "high-dose protocol; aims for spasm cessation and hypsarrhythmia resolution",
            "Vigabatrin: alternative first-line for spasms; GABA-transaminase inhibitor; "
            "visual field monitoring mandatory",
            "Phenobarbital: neonatal seizure first-line (Ohtahara phase); "
            "broad-spectrum; rapid IV loading available",
            "Valproate: adjunctive; broad-spectrum; effective across seizure types in STXBP1-EIEE4",
            "Clobazam: adjunctive benzodiazepine; useful for tonic/atonic seizures in LGS evolution",
            "Topiramate: adjunctive in LGS-evolved STXBP1; sodium channel + AMPA + carbonic anhydrase",
            "Rufinamide: FDA-approved adjunctive for LGS; reduces tonic and atonic (drop) seizures",
            "Deep brain stimulation (DBS) or VNS: for movement disorder component (tremor/dystonia) "
            "in older patients; VNS also for seizure reduction",
        ],
        "key_ddx": [
            "ARX EIEE1 — X-linked males; neonatal suppression-burst; poly-A expansion → Proud syndrome; ARX gene",
            "KCNQ2-DEE — neonatal M-current; bimodal GOF/LOF; quinidine option for GOF",
            "Early myoclonic encephalopathy (EME) — myoclonic > tonic; non-ketotic hyperglycinaemia cause common",
            "Pyridoxine-dependent epilepsy — IV pyridoxine response; ALDH7A1; CSF amino acids",
            "CDKL5-CDD — infantile spasms <5 months; X-linked; Rett-like; ganaxolone option",
        ],
        "onset_range_y": (0.0, 0.1),
        "sex_female_prob": 0.50,
        "eeg_pattern": "Suppression-burst (neonatal) → Hypsarrhythmia (West) → Diffuse slow spike-wave (LGS)",
        "seizure_type": "Tonic neonatal / epileptic spasms / focal clonic / tonic-atonic (LGS)",
        "sudep_risk": "High (severe DEE)",
        "targeted_therapy_available": False,
        "severity_dist": {"Severe": 0.75, "Moderate": 0.22, "Mild": 0.03},
        "seizure_free_rate": 0.03,
        "status_hx_rate": 0.80,
        "drug_error_rate": 0.05,
        "first_line_drug": "ACTH or vigabatrin (spasms) / Phenobarbital (neonatal)",
        "critical_avoid": "No targeted therapy; movement disorder is distinct from seizures — address separately; AVOID sedating polytherapy that masks motor disorder",
    },
    # ── PCDH19 — PCDH19 Epilepsy (Cellular Interference) ──────────────────
    {
        "gene": "PCDH19", "protein": "Protocadherin 19 (PCDH19)",
        "alias": "PCDH19; OMIM gene 300460; Xq22.1; ~1148 aa; PCDH19 epilepsy / DEE9 (OMIM #300088); CELLULAR INTERFERENCE — hemizygous males UNAFFECTED; heterozygous females AFFECTED; mosaic males can be affected; fever-triggered CLUSTERS; FEMALES ONLY (virtually)",
        "aa": "~1148 aa", "kDa": "~128 kDa",
        "mechanism": (
            "PCDH19 encodes protocadherin 19, a calcium-dependent cell-adhesion molecule of the "
            "delta-2 subfamily of non-clustered protocadherins. PCDH19 is expressed in developing "
            "cortical neurons and mediates cell-cell adhesion in neural circuits. "
            "CELLULAR INTERFERENCE MECHANISM — UNIQUE IN GENETICS: "
            "Unlike all other X-linked diseases, PCDH19 epilepsy follows CELLULAR INTERFERENCE (mosaic model): "
            "HEMIZYGOUS MALES (one X, null for PCDH19): ALL neurons express NO PCDH19 — "
            "homogeneous null → NO disease (neurons are all the same, circuits assemble normally). "
            "HETEROZYGOUS FEMALES (one normal X, one mutant X): due to random X-inactivation (Lyon), "
            "each neuron expresses EITHER wild-type OR mutant PCDH19 → a MOSAIC of cells with and without "
            "PCDH19 protein → CELLULAR INTERFERENCE: cells with and without PCDH19 cannot interact normally "
            "(PCDH19 is a homophilic adhesion molecule — requires same PCDH19 expression for binding) → "
            "aberrant circuit assembly → epilepsy. "
            "MOSAIC MALES: somatic mosaic males (some cells null, some expressing PCDH19) → SAME "
            "CELLULAR INTERFERENCE as females → affected (rare but documented). "
            "OBLIGATE CARRIER FATHER: hemizygous male transmits his SINGLE mutant X to ALL daughters "
            "(all daughters are obligate carriers and may be affected); sons receive Y, not X → unaffected."
        ),
        "disease_type": "PCDH19 Epilepsy / DEE9 — X-linked Cellular Interference; Females Only",
        "locus": "Xq22.1", "omim_gene": 300460, "omim_disease": 300088,
        "inheritance": (
            "X-LINKED — CELLULAR INTERFERENCE MECHANISM (unique, non-standard X-linked pattern). "
            "HETEROZYGOUS FEMALES: AFFECTED (epilepsy in >80% of female carriers). "
            "HEMIZYGOUS MALES: UNAFFECTED (no cellular interference in homogeneous null). "
            "MOSAIC MALES: rare — can be affected (somatic mosaicism creates interference). "
            "OBLIGATE CARRIER FATHER: hemizygous male → 100% of daughters are carriers (obligate); "
            "sons are 100% unaffected. "
            "INHERITANCE PEARL: a father with a documented PCDH19 mutation who is clinically normal "
            "transmits the mutation to ALL daughters — genetic counselling MANDATORY."
        ),
        "phenotype": (
            "PCDH19 EPILEPSY (DEE9): "
            "ONSET: 5 months to 3 years (median ~8–10 months); FEVER-TRIGGERED SEIZURE CLUSTERS pathognomonic. "
            "SEIZURE CLUSTERS: brief focal or bilateral tonic-clonic seizures occurring in CLUSTERS "
            "(multiple seizures over minutes to hours) with fever — BRIEF CLUSTER PATTERN IS PATHOGNOMONIC. "
            "ICTAL SEMIOLOGY: emotional hypermotor seizures (screaming, fear, laughing); "
            "focal clonic; bilateral clonic. "
            "EEG: ictal focal or generalised discharges; interictal may be normal between clusters. "
            "INTER-CLUSTER PERIOD: often SEIZURE-FREE for weeks to months between cluster episodes. "
            "DEVELOPMENT: intellectual disability in ~50–70%; autism/ADHD features; "
            "some have near-normal development with medication control. "
            "PSYCHIATRIC COMORBIDITY: anxiety, aggression, stereotyped behaviours prominent. "
            "EVOLUTION: seizures often improve in puberty (some enter prolonged remission in adolescence). "
            "TREATMENT: bromides (potassium bromide) historically used; VPA; benzodiazepines for clusters."
        ),
        "treatment_options": [
            "Clobazam: first-line adjunctive; reduces cluster severity and frequency; "
            "rescue dose for cluster onset is particularly effective",
            "Valproate: broad-spectrum; adjunctive first-line; reduces seizure frequency between clusters",
            "Potassium bromide (KBr): historically first-line for PCDH19 epilepsy; "
            "effective cluster suppression; nephrotoxicity risk (renal monitoring); "
            "sedation and rash; still used in refractory cases",
            "Progesterone / hormonotherapy: emerging evidence; progesterone receptors may modulate "
            "GABA-A sensitivity; improves seizure control in some PCDH19 females; "
            "not standard of care but used compassionately",
            "Steroids (prednisolone/methylprednisolone): some response reported for cluster suppression; "
            "short-course consideration in refractory clusters",
            "Levetiracetam / topiramate: adjunctive; variable response; commonly used in combination",
            "Emergency rescue: diastat (rectal diazepam) or intranasal midazolam for cluster abort; "
            "CRITICAL: recognise cluster early and treat promptly to prevent status epilepticus",
            "AVOID prolonged fever: fever prevention (antipyretics at first temperature elevation); "
            "education of parents on cluster recognition and rescue protocol; "
            "school emergency plan essential",
        ],
        "key_ddx": [
            "SCN1A Dravet — temperature-sensitive; but clusters less prominent; males also affected",
            "GEFS+ (SCN1A/SCN1B) — febrile seizures extended beyond 6 years; no cluster pattern; milder",
            "Doose syndrome (MAE) — myoclonic-atonic; onset 2–6 years; normal sex ratio; CSWS possible",
            "FIRES (febrile infection-related epilepsy syndrome) — status epilepticus; inflammatory; immune",
            "HHE syndrome (hemi-convulsion-hemiplegia-epilepsy) — prolonged febrile hemiconvulsion → hemiplegia; structural",
        ],
        "onset_range_y": (0.4, 3.0),
        "sex_female_prob": 0.92,
        "eeg_pattern": "Ictal focal/generalised discharges during clusters; normal interictal",
        "seizure_type": "Fever-triggered brief seizure clusters / hypermotor focal / bilateral clonic",
        "sudep_risk": "Low-moderate (clusters may be prolonged)",
        "targeted_therapy_available": False,
        "severity_dist": {"Severe": 0.30, "Moderate": 0.50, "Mild": 0.20},
        "seizure_free_rate": 0.20,
        "status_hx_rate": 0.40,
        "drug_error_rate": 0.06,
        "first_line_drug": "Clobazam + VPA / Bromides (KBr) for refractory clusters",
        "critical_avoid": "AVOID misdiagnosing unaffected father as unaffected transmitter — ALL daughters of hemizygous father are obligate carriers; AVOID fever without antipyretic pre-treatment",
    },
    # ── SCN8A — EIEE13 / DEE13 (GOF Nav1.6) ─────────────────────────────────
    {
        "gene": "SCN8A", "protein": "Voltage-Gated Sodium Channel Nav1.6 (SCN8A)",
        "alias": "SCN8A; OMIM gene 600702; 12q13.13; ~1980 aa; EIEE13/DEE13 (OMIM #614558); AD de novo GOF; persistent sodium current; HIGH SUDEP RISK up to 10%; quinidine (GOF-specific); early neonatal to 6 mo onset",
        "aa": "~1980 aa", "kDa": "~225 kDa",
        "mechanism": (
            "SCN8A encodes Nav1.6, a voltage-gated sodium channel alpha-subunit predominantly expressed "
            "in excitatory neurons (pyramidal cells, granule cells) and Purkinje cells of the cerebellum. "
            "Nav1.6 is the dominant sodium channel at the axon initial segment (AIS) and nodes of Ranvier "
            "of myelinated axons — critical for action potential generation and propagation. "
            "SCN8A-DEE MECHANISM — GOF: de novo gain-of-function (GOF) missense mutations → "
            "one or more of: incomplete inactivation (persistent sodium current, Ina-P), "
            "hyperpolarised activation voltage, slowed inactivation, accelerated recovery from inactivation → "
            "PERSISTENT SODIUM INWARD CURRENT at resting potential → sustained neuronal depolarisation → "
            "repetitive action potential firing → network hyperexcitability → severe epileptic encephalopathy. "
            "EXCITATORY BIAS: Nav1.6 is predominantly in excitatory neurons (unlike Nav1.1 in inhibitory) → "
            "GOF Nav1.6 directly drives excitatory neuronal hyperexcitability. "
            "SUDEP RISK: pathologically elevated Nav1.6 persistent current → "
            "cardiac and respiratory autonomic dysregulation + seizure-related apnoea → "
            "estimated SUDEP risk up to 10% in SCN8A-DEE (highest among DEE genes). "
            "QUINIDINE MECHANISM: open-channel blocker; preferentially blocks the persistent sodium current "
            "(Ina-P) generated by GOF Nav1.6 → reduces burst firing; GOF-specific benefit. "
            "AVOID LOF DRUGS: drugs that block residual channel would worsen a GOF mechanism if Nav1.1 "
            "pathway also involved; quinidine is specific to Nav1.6 persistent current."
        ),
        "disease_type": "EIEE13 / DEE13 — GOF Nav1.6 Epileptic Encephalopathy (AD de novo)",
        "locus": "12q13.13", "omim_gene": 600702, "omim_disease": 614558,
        "inheritance": (
            "AUTOSOMAL DOMINANT — DE NOVO GOF (>95% of SCN8A-DEE). "
            "Both sexes equally affected. "
            "Parental mosaicism documented (low-level somatic/germline). "
            "SCN8A accounts for ~1–2% of all DEE (significant given severe phenotype). "
            "Genotype-phenotype: GOF variants in DIII-S4/S5 linker (voltage sensor) = most severe; "
            "some LOF SCN8A variants → ataxia without epilepsy (different phenotype). "
            "GOF functional classification by patch-clamp or established variant databases is critical."
        ),
        "phenotype": (
            "SCN8A-DEE (EIEE13): "
            "ONSET: neonatal to 6 months (median ~4 months); among the earliest DEEs. "
            "EEG: multifocal spikes; focal ictal discharges; not classical suppression-burst (unlike KCNQ2/ARX); "
            "burst activity present in neonatal period in severe cases. "
            "SEIZURES: focal motor, focal tonic, bilateral tonic-clonic, spasms; "
            "STATUS EPILEPTICUS frequent (90%+ of patients); seizure clusters common. "
            "DEVELOPMENT: severe intellectual disability; absent language; hypotonia; spasticity; "
            "movement disorder (chorea, dystonia). "
            "SUDEP RISK: HIGH — estimated 8–10% lifetime SUDEP risk; "
            "sudden unexpected death has been reported in infants; nocturnal monitoring CRITICAL. "
            "EVOLUTION: intractable epilepsy throughout life; no spontaneous remission. "
            "QUINIDINE USE: case series show 40–60% responders in GOF SCN8A-DEE; "
            "must confirm GOF before prescribing; cardiac monitoring (QTc) mandatory during quinidine use."
        ),
        "treatment_options": [
            "Quinidine: sodium channel blocker targeting persistent Nav1.6 current; "
            "effective in GOF SCN8A-DEE (case series 40–60% responders); "
            "dose: 10–30 mg/kg/day in 3–4 doses; MANDATORY: baseline and monitoring ECG (QTc prolongation risk); "
            "confirm GOF variant before use; NOT appropriate for LOF SCN8A variants",
            "High-dose phenytoin (or fosphenytoin): sodium channel blocker with high affinity for Nav1.6; "
            "some efficacy in SCN8A-DEE; caution — can also worsen if LOF overlap; "
            "therapeutic drug monitoring required",
            "Phenobarbital: broad neonatal first-line; adjunctive in SCN8A-DEE",
            "Sodium valproate: broad-spectrum adjunctive; useful combination",
            "Tricyclic antidepressants (nortriptyline, carbamazepine): some case reports in SCN8A-DEE; "
            "sodium channel blocking effect; use with caution given cardiac effects",
            "Ketogenic diet: adjunctive; can reduce seizure burden; recommended early given refractory course",
            "VNS (vagus nerve stimulation): palliative seizure reduction; consider when ≥4 ASMs failed",
            "SUDEP PREVENTION: nocturnal seizure monitoring (pulse oximetry, apnoea monitor); "
            "room sharing for infants; SAMi mattress; avoid unsupervised swimming/bath; "
            "seizure detection devices; family education on rescue breathing",
        ],
        "key_ddx": [
            "SCN1A Dravet — temperature-sensitive; Nav1.1 LOF not GOF; onset 5–8 months; avoid CBZ/OXC",
            "KCNQ2-DEE — neonatal onset; suppression-burst; M-current; quinidine also used for GOF KCNQ2",
            "STXBP1-EIEE4 — neonatal suppression-burst; movement disorder; presynaptic; de novo",
            "Lennox-Gastaut syndrome (multiple causes) — older onset; tonic+atonic+absence triad",
            "Febrile infection-related epilepsy (FIRES) — school-age; inflammatory; MRI evolution",
        ],
        "onset_range_y": (0.0, 0.5),
        "sex_female_prob": 0.50,
        "eeg_pattern": "Multifocal spikes / focal ictal / burst activity (neonatal)",
        "seizure_type": "Focal motor / focal tonic / bilateral tonic-clonic / status epilepticus",
        "sudep_risk": "Very high (8-10% lifetime)",
        "targeted_therapy_available": True,
        "severity_dist": {"Severe": 0.75, "Moderate": 0.22, "Mild": 0.03},
        "seizure_free_rate": 0.04,
        "status_hx_rate": 0.88,
        "drug_error_rate": 0.07,
        "first_line_drug": "Quinidine (GOF-specific Nav1.6 blocker) / Phenobarbital (neonatal)",
        "critical_avoid": "AVOID quinidine without GOF functional confirmation; MANDATORY SUDEP monitoring (nocturnal); HIGH MORTALITY risk if unmonitored",
    },
    # ── GRIN2A — Epilepsy-Aphasia Spectrum (EAS) ─────────────────────────────
    {
        "gene": "GRIN2A", "protein": "Glutamate Ionotropic Receptor NMDA Subunit GluN2A (GRIN2A)",
        "alias": "GRIN2A; OMIM gene 138253; 16p13.2; ~1464 aa; Epilepsy-Aphasia Spectrum EAS (OMIM #245570); CSWS/ESES; Landau-Kleffner Syndrome LKS; ABPE; sleep-activated; language regression/aphasia PATHOGNOMONIC; sulthiame first-line; AD variable penetrance",
        "aa": "~1464 aa", "kDa": "~166 kDa",
        "mechanism": (
            "GRIN2A encodes GluN2A, the NR2A subunit of NMDA (N-methyl-D-aspartate) receptors — "
            "the major ionotropic glutamate receptors for synaptic plasticity (LTP and LTD). "
            "GluN2A FUNCTION: forms obligate heterotetramer with GluN1 (GRIN1) subunits; "
            "GluN2A-containing NMDA receptors are the predominant form in adult cortex — "
            "high expression in language-processing areas (perisylvian cortex, superior temporal gyrus). "
            "GRIN2A-EAS MECHANISM: GRIN2A mutations (LOF or GOF, both reported) → "
            "altered NMDA receptor kinetics in perisylvian cortex → "
            "disrupted synaptic plasticity in language networks → aberrant cortical synchrony → "
            "SLEEP-ACTIVATED EPILEPTIFORM ACTIVITY (CSWS/ESES) in slow-wave sleep → "
            "interference with sleep-dependent memory consolidation + language learning. "
            "CSWS/ESES DEFINITION: continuous spike-wave during slow-wave sleep = "
            "spike-wave index (SWI) >85% during NREM sleep on overnight EEG — "
            "DIAGNOSTIC REQUIREMENT for ESES. "
            "ACQUIRED EPILEPTIC APHASIA (Landau-Kleffner): LKS = acquired loss of language "
            "in a child with previously normal development + CSWS on EEG → GRIN2A most common single-gene cause. "
            "SULTHIAME MECHANISM: carbonic anhydrase inhibitor; "
            "reduces CSWS/ESES burden; first-line for GRIN2A-EAS."
        ),
        "disease_type": "Epilepsy-Aphasia Spectrum (EAS): CSWS/ESES, LKS, ABPE (AD GRIN2A, variable penetrance)",
        "locus": "16p13.2", "omim_gene": 138253, "omim_disease": 245570,
        "inheritance": (
            "AUTOSOMAL DOMINANT with VARIABLE PENETRANCE (~60% penetrance). "
            "Familial cases common (parent may have history of benign childhood epilepsy / language delay). "
            "De novo and familial. "
            "GRIN2A mutations found in 10–20% of EAS families (most common single-gene cause of LKS). "
            "Overall penetrance and expressivity variable — same family member may have ABPE, "
            "another CSWS/ESES, another LKS. "
            "Genetic testing: GRIN2A sequencing + CNV detection (deletions reported). "
            "EEG STUDY: overnight EEG (polysomnography-quality) required — awake/routine EEG misses CSWS."
        ),
        "phenotype": (
            "EPILEPSY-APHASIA SPECTRUM (EAS): "
            "ONSET: 4–12 years (school-age) — LATE ONSET compared to other DEE genes. "
            "EEG HALLMARK: CSWS/ESES — continuous spike-wave during slow-wave sleep; "
            "spike-wave index (SWI) >85% in NREM sleep; centrotemporal / perisylvian focus. "
            "ABPE (atypical benign partial epilepsy of childhood): rolandic-like seizures + "
            "some cognitive/language involvement; mild. "
            "CSWS (continuous spike-wave during slow-wave sleep): cognitive deterioration, "
            "language regression, behavioural disturbance — correlates with SWI; "
            "seizures secondary concern vs cognitive. "
            "LANDAU-KLEFFNER SYNDROME (LKS — acquired epileptic aphasia): "
            "ACQUIRED LANGUAGE LOSS (verbal agnosia → no speech comprehension) in a "
            "previously normal child (4–7 years) + CSWS on sleep EEG; "
            "LANGUAGE REGRESSION IS PATHOGNOMONIC; seizures may be minimal; "
            "auditory verbal agnosia (understands non-verbal sounds, not words). "
            "PROGNOSIS: variable; seizures remit in adolescence in most; "
            "language recovery depends on duration of CSWS and age at treatment. "
            "SULTHIAME: reduces SWI → language improvement in LKS; first-line EAS."
        ),
        "treatment_options": [
            "Sulthiame (STM): FIRST-LINE for GRIN2A-EAS and CSWS/ESES; "
            "carbonic anhydrase inhibitor → reduces CSWS burden (SWI); "
            "dose: 5–10 mg/kg/day in 2–3 doses; well tolerated; "
            "available in Europe/Israel; compassionate use in North America; "
            "language improvement in LKS with CSWS reduction",
            "Oral corticosteroids (prednisolone/methylprednisolone): for LKS with severe language regression; "
            "high-dose oral prednisolone (2 mg/kg/day) → CSWS reduction → language recovery; "
            "maintain ≥12 months; monitor for steroid side effects; "
            "most effective when started <2 years after language loss onset",
            "ACTH: IV ACTH for severe LKS — some evidence for CSWS suppression; "
            "used as bridge before oral steroids",
            "Clobazam: adjunctive; nocturnal benzodiazepine can transiently suppress CSWS; "
            "tolerance develops; useful short-term",
            "Levetiracetam: adjunctive; some evidence for CSWS reduction in ESES; "
            "well-tolerated; monitor for behavioural side effects (LEV rage)",
            "Valproate: avoid in LKS if possible — some case reports of VPA worsening language; "
            "use only when seizure burden outweighs language risk",
            "Multiple subpial transections (MST): surgical option for LKS with medically refractory CSWS; "
            "transects horizontal fibres in language cortex (Broca/Wernicke area); "
            "preserves function while interrupting seizure propagation; specialised centre only",
            "Speech/language therapy: MANDATORY in LKS — intensive language therapy during CSWS remission; "
            "AAC (augmentative and alternative communication) devices during non-verbal periods",
        ],
        "key_ddx": [
            "Idiopathic childhood occipital epilepsy (Panayiotopoulos/Gastaut) — occipital focus; no language regression",
            "CSWS from structural cause (cortical dysplasia, porencephaly) — MRI lesion present",
            "Autism spectrum disorder with language regression — no epileptiform activity during sleep; EEG normal",
            "Auditory processing disorder (APD) — no EEG changes; audiogram normal/different pattern",
            "Self-limited epilepsy with centrotemporal spikes (SELECTS/BECTS) — no CSWS; no language regression; self-limiting",
        ],
        "onset_range_y": (4.0, 12.0),
        "sex_female_prob": 0.45,
        "eeg_pattern": "CSWS/ESES — spike-wave index >85% in NREM sleep; centrotemporal/perisylvian focus",
        "seizure_type": "Focal motor (rolandic-like) / absence / language regression (LKS); nocturnal predominance",
        "sudep_risk": "Low (benign-spectrum seizures; school-age onset)",
        "targeted_therapy_available": False,
        "severity_dist": {"Severe": 0.15, "Moderate": 0.50, "Mild": 0.35},
        "seizure_free_rate": 0.45,
        "status_hx_rate": 0.15,
        "drug_error_rate": 0.08,
        "first_line_drug": "Sulthiame (CSWS/ESES/LKS first-line) / Corticosteroids (LKS language recovery)",
        "critical_avoid": "AVOID VPA as sole agent in LKS (may worsen language); MANDATORY: overnight EEG (not just routine) to detect CSWS; AVOID missing language therapy window",
    },
]


# ── Patient generator ───────────────────────────────────────────────────────

def _gen_patients(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    patients = []
    onset_lo, onset_hi = gene_data["onset_range_y"]

    for i in range(40):
        onset = round(rng.uniform(onset_lo, onset_hi), 2)

        # Severity
        r = rng.random()
        cumulative = 0.0
        sev = "Severe"
        for label, prob in gene_data["severity_dist"].items():
            cumulative += prob
            if r < cumulative:
                sev = label
                break

        # Sex
        sex = "F" if rng.random() < gene_data["sex_female_prob"] else "M"

        # Clinical booleans
        status_hx     = rng.random() < gene_data["status_hx_rate"]
        seizure_free  = rng.random() < gene_data["seizure_free_rate"]
        drug_error    = rng.random() < gene_data["drug_error_rate"]
        on_targeted   = gene_data["targeted_therapy_available"] and rng.random() < 0.55
        cognitive_imp = sev in ("Severe", "Moderate") and rng.random() < (
            0.95 if gene in ("ARX", "STXBP1", "SCN8A") else
            0.80 if gene in ("SCN1A", "CDKL5", "KCNQ2") else
            0.55 if gene == "PCDH19" else 0.30
        )
        sudep_high = gene_data["sudep_risk"].startswith("Very high") or (
            gene_data["sudep_risk"].startswith("High") and rng.random() < 0.60
        ) or (
            gene_data["sudep_risk"].startswith("Elevated") and rng.random() < 0.35
        )

        # EEG pattern (sampled from gene-specific pattern string)
        eeg = gene_data["eeg_pattern"].split(" / ")[0] if sev == "Severe" else gene_data["eeg_pattern"].split(" / ")[-1]

        # Treatment
        fl = gene_data["first_line_drug"].split(" / ")[0]
        if on_targeted:
            tx = fl + " (targeted)"
        elif drug_error:
            tx = "CONTRAINDICATED drug prescribed (error detected)"
        else:
            tx = fl + (" + adjunctive ASM" if rng.random() < 0.60 else "")

        age_at_dx = round(min(onset + rng.uniform(0.2, 2.0), onset_hi + 2.0), 2)

        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "onset_age_y": onset,
            "diagnosis_age_y": age_at_dx,
            "sex": sex,
            "severity": sev,
            "eeg_pattern": eeg,
            "seizure_type": gene_data["seizure_type"].split(" / ")[0],
            "status_epilepticus_hx": status_hx,
            "seizure_free": seizure_free,
            "drug_avoid_prescribed_error": drug_error,
            "on_targeted_therapy": on_targeted,
            "cognitive_impairment": cognitive_imp,
            "sudep_risk_high": sudep_high,
            "treatment": tx,
            "first_line_drug": gene_data["first_line_drug"],
            "critical_avoid": gene_data["critical_avoid"],
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gene_data in enumerate(DEE_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients(gene_data, seed))
    return all_pts


# ── Public API ──────────────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] += 1

    status_n      = sum(1 for p in patients if p["status_epilepticus_hx"])
    seizfree_n    = sum(1 for p in patients if p["seizure_free"])
    sudep_n       = sum(1 for p in patients if p["sudep_risk_high"])
    cog_n         = sum(1 for p in patients if p["cognitive_impairment"])
    targeted_n    = sum(1 for p in patients if p["on_targeted_therapy"])
    drug_err_n    = sum(1 for p in patients if p["drug_avoid_prescribed_error"])

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 2)
    mean_dx_age = round(sum(p["diagnosis_age_y"] for p in patients) / n, 2)

    return {
        "atlas": "DEE-Atlas",
        "full_name": "Complete 8-Gene Developmental and Epileptic Encephalopathy Atlas",
        "subtitle": (
            "SCN1A·KCNQ2·CDKL5·ARX·STXBP1·PCDH19·SCN8A·GRIN2A — "
            "320 patients (8×40, seeds 1070–1077)"
        ),
        "description": (
            "Comprehensive atlas of 8 major genetic developmental and epileptic encephalopathies (DEEs). "
            "Encompasses: DRAVET SYNDROME (SCN1A — Nav1.1 LOF, inhibitory interneuron loss; "
            "ABSOLUTELY AVOID CBZ/OXC/PHT/LTG; stiripentol+VPA+clobazam; fenfluramine FDA 2020; cannabidiol); "
            "KCNQ2-DEE/BFNS (KCNQ2 — M-current bimodal: GOF→DEE severe; LOF→benign neonatal self-limiting; "
            "carbamazepine LOF; quinidine GOF); "
            "CDKL5 DEFICIENCY DISORDER (CDKL5 — X-linked kinase; spasms <5mo; Rett-like; "
            "ganaxolone FDA 2022 FIRST CDD-specific therapy); "
            "EIEE1/Ohtahara (ARX — X-linked males; poly-A expansion → Proud syndrome; "
            "mutation type determines phenotype); "
            "EIEE4/Ohtahara→West→LGS (STXBP1 — Munc18-1; presynaptic; de novo; movement disorder prominent); "
            "PCDH19 EPILEPSY (PCDH19 — CELLULAR INTERFERENCE; females affected; "
            "hemizygous males UNAFFECTED; fever-triggered clusters PATHOGNOMONIC); "
            "EIEE13 (SCN8A — GOF Nav1.6; HIGH SUDEP RISK 8–10%; quinidine GOF-specific); "
            "EPILEPSY-APHASIA SPECTRUM (GRIN2A — CSWS/ESES sleep-activated; LKS acquired aphasia; "
            "sulthiame first-line; steroids for LKS language recovery). "
            "CRITICAL RULES: Dravet avoid sodium channel blockers; PCDH19 cellular interference unique X-linked pattern; "
            "SCN8A SUDEP monitoring mandatory; GRIN2A overnight EEG required; "
            "GOF vs LOF classification mandatory before KCNQ2 treatment."
        ),
        "total_patients": n,
        "genes_covered": len(DEE_GENES),
        "patients_per_gene": 40,
        "seed_range": "1070–1077",
        "gene_list": [g["gene"] for g in DEE_GENES],
        "disease_category_breakdown": {
            "Dravet Syndrome / GEFS+ (AD de novo SCN1A LOF — Nav1.1 inhibitory neuron loss; AVOID CBZ/OXC/LTG)": ["SCN1A"],
            "KCNQ2-DEE GOF severe / BFNS LOF benign neonatal (Kv7.2 M-current; carbamazepine LOF; quinidine GOF)": ["KCNQ2"],
            "CDKL5 Deficiency Disorder CDD (X-linked kinase; spasms <5mo; ganaxolone FDA 2022 first CDD-specific)": ["CDKL5"],
            "EIEE1/Ohtahara → West → LGS (X-linked ARX; males only; poly-A expansion → Proud syndrome)": ["ARX"],
            "EIEE4/Ohtahara → West → LGS (AD de novo STXBP1/Munc18-1; movement disorder prominent)": ["STXBP1"],
            "PCDH19 Epilepsy DEE9 (Cellular Interference; females affected; hemizygous males unaffected; fever clusters)": ["PCDH19"],
            "EIEE13/DEE13 (AD de novo GOF SCN8A/Nav1.6; SUDEP 8-10%; quinidine GOF-specific)": ["SCN8A"],
            "Epilepsy-Aphasia Spectrum EAS (AD GRIN2A; CSWS/ESES; LKS acquired aphasia; sulthiame first-line)": ["GRIN2A"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "mean_diagnosis_age_y": mean_dx_age,
        "kpis": [
            {"label": "Total Patients", "value": n, "color": "#1565c0"},
            {"label": "Genes Covered", "value": len(DEE_GENES), "color": "#2e7d32"},
            {"label": "Patients/Gene", "value": 40, "color": "#6a1b9a"},
            {"label": "High SUDEP Risk", "value": f"{round(100 * sudep_n / n, 1)}%", "color": "#b71c1c"},
            {"label": "Mean Onset (y)", "value": mean_onset, "color": "#e65100"},
            {"label": "Seeds", "value": "1070–1077", "color": "#37474f"},
        ],
        "clinical_features_prevalence": {
            "Status Epilepticus History": round(100 * status_n / n, 1),
            "Seizure-Free (current)": round(100 * seizfree_n / n, 1),
            "SUDEP Risk — High": round(100 * sudep_n / n, 1),
            "Cognitive Impairment": round(100 * cog_n / n, 1),
            "On Targeted Therapy": round(100 * targeted_n / n, 1),
            "Drug-Avoid Error Detected": round(100 * drug_err_n / n, 1),
        },
        "drug_alerts": [
            "SCN1A DRAVET: ABSOLUTELY AVOID carbamazepine (CBZ), oxcarbazepine (OXC), phenytoin (PHT), "
            "lamotrigine (LTG) — ALL worsen Dravet by blocking residual Nav1.1 in inhibitory interneurons; "
            "first-line: stiripentol + VPA + clobazam",
            "KCNQ2 CRITICAL: GOF vs LOF classification MANDATORY before treatment — carbamazepine is "
            "first-line for LOF-BFNS; quinidine is GOF-DEE specific; applying wrong drug worsens outcome",
            "PCDH19 CELLULAR INTERFERENCE (UNIQUE): hemizygous males are UNAFFECTED but transmit to ALL daughters "
            "(obligate carrier); heterozygous females AFFECTED — inverse of typical X-linked recessive",
            "SCN8A HIGH SUDEP RISK: 8–10% lifetime SUDEP risk — nocturnal monitoring (apnoea alert, pulse oximetry) "
            "MANDATORY; quinidine for GOF Nav1.6; ECG monitoring required during quinidine use (QTc)",
            "CDKL5 CDD: ganaxolone (FDA March 2022) is the FIRST CDD-specific approved therapy — "
            "ensure eligible patients are offered ganaxolone (≥2 years of age)",
            "GRIN2A LKS: AVOID valproate as sole agent in Landau-Kleffner (may worsen language); "
            "sulthiame first-line for CSWS; overnight EEG MANDATORY (routine EEG misses CSWS/ESES)",
        ],
        "diagnostic_pearls": [
            "SCN1A Dravet: FEBRILE STATUS EPILEPTICUS in normal infant 5–8 months + TEMPERATURE SENSITIVITY "
            "→ SCN1A immediately; do NOT prescribe CBZ/OXC pending result",
            "KCNQ2: neonatal suppression-burst + de novo → GOF-DEE until classified; "
            "quinidine only after GOF confirmation; carbamazepine for LOF-BFNS",
            "CDKL5: infantile spasms BEFORE 5 MONTHS + female-predominant + Rett-like features + X-linked → "
            "CDKL5 (not MECP2 — different onset); ganaxolone if ≥2 years",
            "ARX: neonatal suppression-burst in a MALE + X-linked pattern → ARX first (alongside STXBP1/KCNQ2); "
            "second poly-A expansion → Proud syndrome (agenesis CC + dystonia)",
            "STXBP1: neonatal suppression-burst + hypotonia + MOVEMENT DISORDER (tremor/dystonia) → "
            "STXBP1 top differential; both sexes; de novo",
            "PCDH19: GIRLS ONLY + FEVER-TRIGGERED SEIZURE CLUSTERS (brief, recurring) + "
            "normal father with family history → PCDH19 (father is unaffected hemizygous transmitter)",
            "SCN8A: early infantile onset + refractory + HIGH SUDEP risk + multifocal EEG → "
            "SCN8A GOF; quinidine + nocturnal monitoring",
            "GRIN2A LKS: child with ACQUIRED LANGUAGE LOSS + previous normal development + "
            "CSWS on OVERNIGHT EEG → GRIN2A; sulthiame + steroids; speech therapy urgent",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    breakdown = {}
    for gene_data in DEE_GENES:
        gene = gene_data["gene"]
        gene_pts = [p for p in patients if p["gene"] == gene]
        n = len(gene_pts)
        sev = {s: sum(1 for p in gene_pts if p["severity"] == s) for s in ("Mild", "Moderate", "Severe")}

        # Sex distribution
        f_n = sum(1 for p in gene_pts if p["sex"] == "F")

        breakdown[gene] = {
            "gene": gene,
            "protein": gene_data["protein"],
            "alias": gene_data["alias"],
            "n_patients": n,
            "disease_type": gene_data["disease_type"],
            "locus": gene_data["locus"],
            "omim_gene": gene_data["omim_gene"],
            "omim_disease": gene_data["omim_disease"],
            "inheritance": gene_data["inheritance"],
            "phenotype": gene_data["phenotype"],
            "treatment_options": gene_data["treatment_options"],
            "key_ddx": gene_data["key_ddx"],
            "mechanism": gene_data["mechanism"],
            "eeg_pattern": gene_data["eeg_pattern"],
            "seizure_type": gene_data["seizure_type"],
            "sudep_risk": gene_data["sudep_risk"],
            "first_line_drug": gene_data["first_line_drug"],
            "critical_avoid": gene_data["critical_avoid"],
            "severity_distribution": {
                "mild_pct": round(100 * sev["Mild"] / n, 1),
                "moderate_pct": round(100 * sev["Moderate"] / n, 1),
                "severe_pct": round(100 * sev["Severe"] / n, 1),
            },
            "mean_onset_age_y": round(sum(p["onset_age_y"] for p in gene_pts) / n, 2),
            "mean_diagnosis_age_y": round(sum(p["diagnosis_age_y"] for p in gene_pts) / n, 2),
            "sex_pct_female": round(100 * f_n / n, 1),
            "seizure_free_pct": round(100 * sum(1 for p in gene_pts if p["seizure_free"]) / n, 1),
            "status_hx_pct": round(100 * sum(1 for p in gene_pts if p["status_epilepticus_hx"]) / n, 1),
            "sudep_risk_high_pct": round(100 * sum(1 for p in gene_pts if p["sudep_risk_high"]) / n, 1),
            "drug_error_pct": round(100 * sum(1 for p in gene_pts if p["drug_avoid_prescribed_error"]) / n, 1),
            "cognitive_impairment_pct": round(100 * sum(1 for p in gene_pts if p["cognitive_impairment"]) / n, 1),
            "on_targeted_therapy_pct": round(100 * sum(1 for p in gene_pts if p["on_targeted_therapy"]) / n, 1),
        }
    return {
        "atlas": "DEE-Atlas",
        "subtitle": "Per-gene clinical breakdown — 320 patients (8×40, seeds 1070–1077)",
        "genes": breakdown,
        "gene_order": [g["gene"] for g in DEE_GENES],
    }


def get_definitions() -> dict:
    return {
        "atlas": "DEE-Atlas",
        "subtitle": "Clinical and genetic terminology definitions for DEE-Atlas",
        "definitions": {
            "Dravet Syndrome": (
                "Severe DEE caused by heterozygous SCN1A LOF mutations (>90% de novo). Onset in first year "
                "of life with febrile status epilepticus; temperature-sensitive seizures; progressive "
                "cognitive decline; refractory epilepsy. CRITICAL: sodium channel blockers (CBZ/OXC/PHT/LTG) "
                "worsen Dravet by blocking residual Nav1.1 in inhibitory neurons. First-line: "
                "stiripentol + VPA + clobazam; fenfluramine (FDA 2020); cannabidiol."
            ),
            "GEFS+ (Generalised Epilepsy with Febrile Seizures Plus)": (
                "Milder SCN1A spectrum — febrile seizures extending beyond 6 years + occasional afebrile "
                "generalised seizures; good prognosis; autosomal dominant; no cognitive regression. "
                "Same SCN1A gene as Dravet but less severe mutations (missense in linker regions)."
            ),
            "GOF (Gain-of-Function) Mutation": (
                "Mutation that confers a new or enhanced protein function. In ion channel genes: GOF often "
                "means persistent current, slowed inactivation, or constitutive activation → neuronal "
                "hyperexcitability. Examples: KCNQ2 GOF (DEE), SCN8A GOF (DEE13). Treated differently "
                "from LOF mutations — quinidine targets GOF persistent current."
            ),
            "LOF (Loss-of-Function) Mutation": (
                "Mutation that reduces or abolishes normal protein function. Haploinsufficiency = one "
                "functional allele is insufficient. Examples: SCN1A LOF (Dravet — Nav1.1 reduced in "
                "inhibitory neurons), KCNQ2 LOF (BFNS — IKM reduced), STXBP1 LOF (Munc18-1 haploinsufficiency). "
                "LOF vs GOF classification is MANDATORY before prescribing targeted therapies."
            ),
            "Suppression-Burst": (
                "EEG pattern of high-amplitude bursts of polyspike-slow waves alternating with near-flat "
                "(suppressed) background; occurs in severe neonatal epileptic encephalopathies. "
                "Pathognomonic of Ohtahara syndrome (EIEE). Genes: KCNQ2-DEE, ARX-EIEE1, STXBP1-EIEE4. "
                "Continuous even during sleep (distinguishes from burst suppression of sedation)."
            ),
            "Hypsarrhythmia": (
                "High-amplitude chaotic slow waves mixed with multifocal spikes and sharp waves — the "
                "interictal EEG hallmark of West syndrome (infantile spasms). Seen in CDKL5-CDD, "
                "STXBP1-EIEE4 (West evolution), ARX-EIEE1 (West evolution), and many other DEEs. "
                "Cessation of hypsarrhythmia after ACTH is the EEG treatment goal."
            ),
            "West Syndrome": (
                "Epileptic encephalopathy triad: infantile spasms + hypsarrhythmia on EEG + "
                "psychomotor regression. Onset 4–8 months. Many genetic causes (CDKL5, ARX, STXBP1, "
                "TSC1/2, chromosomal). First-line: ACTH or vigabatrin. "
                "Response within 2 weeks = remission achieved; non-responders need second-line."
            ),
            "LGS (Lennox-Gastaut Syndrome)": (
                "Severe DEE triad: multiple seizure types (tonic, atonic/drop attacks, atypical absence) + "
                "diffuse slow spike-wave on EEG (SWI <2.5 Hz) + intellectual disability. "
                "Onset 1–8 years. Often evolves from Ohtahara or West (STXBP1, ARX). "
                "Adjunctive options: rufinamide, clobazam, lamotrigine, VNS, corpus callosotomy."
            ),
            "CSWS/ESES (Continuous Spike-Wave During Slow-Wave Sleep / Electrical Status Epilepticus During Sleep)": (
                "EEG pattern of continuous (>85% NREM sleep time) spike-wave discharges during slow-wave sleep; "
                "spike-wave index (SWI) >85% is the diagnostic threshold. Hallmark of epilepsy-aphasia spectrum "
                "(GRIN2A). Associated with cognitive decline, language regression, behavioural disturbance. "
                "Requires OVERNIGHT polysomnography-quality EEG — routine EEG MISSES CSWS."
            ),
            "Cellular Interference (PCDH19)": (
                "Unique X-linked disease mechanism where disease occurs in MOSAIC cells — not in uniformly "
                "null cells. In PCDH19 epilepsy: heterozygous females have a mosaic of PCDH19-positive and "
                "PCDH19-negative neurons (due to X-inactivation) → these cells cannot interact normally "
                "(PCDH19 is homophilic) → aberrant circuit assembly → epilepsy. Hemizygous males "
                "(all cells null) are UNAFFECTED. Mosaic males can be affected. "
                "This is the OPPOSITE of typical X-linked recessive inheritance."
            ),
            "SUDEP (Sudden Unexpected Death in Epilepsy)": (
                "Unexplained death in a person with epilepsy, without evidence of drowning or trauma, "
                "with or without evidence of a seizure. Leading cause of death in drug-resistant epilepsy. "
                "Risk highest in: SCN8A-DEE (8–10% lifetime), Dravet syndrome (1–2%/y), STXBP1-EIEE. "
                "Prevention: nocturnal seizure monitoring, prone position avoidance, adherence to ASMs, "
                "SUDEP-specific wearable monitors."
            ),
            "Febrile Status Epilepticus": (
                "Prolonged (>30 minutes) or cluster seizure associated with fever without CNS infection. "
                "TYPICAL FIRST EVENT in Dravet syndrome (SCN1A LOF). Presentation: prolonged febrile hemiclonic "
                "seizure in a normal 5–8 month infant → EMERGENCY → SCN1A testing URGENTLY. "
                "Also associated with PCDH19 epilepsy (fever-triggered CLUSTERS)."
            ),
            "Stiripentol (STP) Mechanism": (
                "Stiripentol is a positive allosteric modulator of GABA-A receptors (direct effect at "
                "alpha3-containing receptors in developing brain) AND a potent CYP3A4/CYP2C19 inhibitor → "
                "elevates active N-desmethylclobazam metabolite levels (~5×) → enhanced clobazam efficacy. "
                "Licensed for Dravet syndrome in combination with VPA + clobazam. "
                "Monitor: sedation, neutropenia, LFT."
            ),
            "CDKL5 Deficiency Disorder (CDD)": (
                "X-linked DEE caused by CDKL5 mutations (kinase critical for synaptogenesis). "
                "Onset: infantile spasms before 5 months (distinguishes from classic Rett which has 6–18mo "
                "normal period). Features: Rett-like hand stereotypies, absent speech, absent purposeful hand use. "
                "Females predominate in clinical series. Ganaxolone (FDA March 2022) is the FIRST CDD-specific therapy."
            ),
            "Ohtahara Syndrome (EIEE — Early Infantile Epileptic Encephalopathy)": (
                "Severe neonatal epileptic encephalopathy: onset first 3 months (usually neonatal); "
                "suppression-burst EEG; tonic spasms; profound developmental impairment. "
                "Causes: ARX (X-linked males), STXBP1 (de novo AD), KCNQ2-DEE (GOF), structural, metabolic. "
                "Evolves to West syndrome (hypsarrhythmia) then LGS in most survivors."
            ),
            "Munc18-1 Function (STXBP1)": (
                "Munc18-1 (STXBP1) is an SM protein that binds syntaxin-1 and chaperones SNARE complex assembly "
                "(syntaxin-1/SNAP-25/synaptobrevin-2) → essential for synaptic vesicle docking, priming, and "
                "calcium-triggered fusion for BOTH inhibitory (GABA) AND excitatory (glutamate) release. "
                "Haploinsufficiency → global presynaptic vesicle fusion deficit → seizures + movement disorder."
            ),
            "EAS (Epilepsy-Aphasia Spectrum)": (
                "A spectrum of epilepsies unified by perisylvian/centrotemporal epileptiform activity "
                "and language involvement. Ranges from ABPE (atypical benign partial epilepsy), through "
                "CSWS/ESES (cognitive/language regression), to Landau-Kleffner syndrome (acquired aphasia). "
                "GRIN2A is the most common single-gene cause, found in 10–20% of EAS families."
            ),
            "LKS (Landau-Kleffner Syndrome — Acquired Epileptic Aphasia)": (
                "ACQUIRED LANGUAGE LOSS in a previously normal child (ages 4–7 years) caused by CSWS "
                "on sleep EEG. Key feature: verbal agnosia (child hears sounds but cannot process spoken words) → "
                "appears deaf. Seizures may be minimal. GRIN2A most common genetic cause. "
                "Treatment: sulthiame (first-line) + oral corticosteroids → language recovery if treated early. "
                "Language therapy MANDATORY during remission."
            ),
            "Fenfluramine Mechanism (Dravet)": (
                "Fenfluramine (Fintepla) is a serotonin-releasing agent and sigma-1 receptor agonist. "
                "Releases serotonin from presynaptic terminals AND inhibits its reuptake → elevated synaptic "
                "5-HT → activation of serotonin receptors (5-HT2C, 5-HT1A) → reduced excitability. "
                "Also modulates sigma-1 receptor. FDA approved 2020 for Dravet (≥2 years); also approved for "
                "LGS (FDA 2022). REQUIRES echocardiography monitoring (historical valvulopathy at high doses)."
            ),
            "Ganaxolone": (
                "Ganaxolone (Ztalmy, Marinus Pharmaceuticals) is a synthetic analogue of allopregnanolone "
                "(a neurosteroid metabolite of progesterone). Mechanism: positive allosteric modulator of both "
                "synaptic (gamma-subunit-containing) and extrasynaptic (delta-subunit-containing) GABA-A receptors. "
                "FDA approved March 2022 for CDKL5 deficiency disorder (CDD) in patients ≥2 years — "
                "the FIRST CDD-specific approved therapy. Reduces motor seizure frequency by ~30%."
            ),
            "Quinidine (SCN8A / KCNQ2 GOF)": (
                "Quinidine is a class Ia antiarrhythmic (sodium and potassium channel blocker). "
                "In GOF SCN8A-DEE: blocks the persistent sodium current (Ina-P) generated by GOF Nav1.6 "
                "→ reduces burst firing → seizure reduction (40–60% responders in case series). "
                "In GOF KCNQ2-DEE: blocks constitutively active Kv7.2 → reduces persistent depolarisation. "
                "MANDATORY: baseline ECG + QTc monitoring during use (QTc prolongation risk). "
                "NOT appropriate without GOF functional classification."
            ),
            "Sulthiame": (
                "Sulthiame (STM, Ospolot) is a carbonic anhydrase inhibitor antiseizure medication. "
                "Mechanism in EAS/CSWS: reduces CSWS burden (spike-wave index) → cognitive/language improvement. "
                "First-line for GRIN2A-EAS and CSWS/ESES of centrotemporal origin. "
                "Also effective in self-limited epilepsy with centrotemporal spikes (SELECTS/BECTS). "
                "Available in Europe and Israel; compassionate/investigational use in North America."
            ),
            "ARX Polyalanine Expansion": (
                "ARX gene contains two polyalanine (GCG repeat) tracts in exon 2. Expansion of these "
                "tracts causes disease with MUTATION TYPE → PHENOTYPE correspondence: "
                "First poly-A tract expansion: ISSX/Partington syndrome (intellectual disability + ataxia). "
                "Second poly-A tract expansion: PRTS (Partington syndrome: ID + dystonia). "
                "PROUD SYNDROME: specific ARX mutations → intellectual disability + AGENESIS OF CORPUS CALLOSUM "
                "+ DYSTONIA. Polyalanine TRACT SIZING is MANDATORY for phenotype prediction."
            ),
            "EIEE Classification (Early Infantile Epileptic Encephalopathy)": (
                "Numbered classification of early infantile epileptic encephalopathies by causative gene: "
                "EIEE1 = ARX (X-linked males); EIEE4 = STXBP1 (de novo AD, ~10-20% of Ohtahara); "
                "EIEE7 = KCNQ2 (GOF DEE); EIEE9 = PCDH19 (cellular interference, females); "
                "EIEE13 = SCN8A (GOF Nav1.6, HIGH SUDEP); CDKL5-DD formerly EIEE2 (now CDD). "
                "Numerical classification largely superseded by gene-specific terminology."
            ),
            "Nav1.1 Inhibitory Neuron Loss (Dravet Mechanism)": (
                "Nav1.1 (SCN1A) is preferentially expressed in fast-spiking GABAergic parvalbumin-positive "
                "interneurons. These interneurons require high Nav1.1 expression to sustain rapid firing "
                "for inhibitory tone. SCN1A LOF → Nav1.1 haploinsufficiency → interneuron firing failure → "
                "loss of cortical inhibition → network disinhibition → seizures. "
                "Excitatory neurons compensate with Nav1.2/Nav1.6 and are less affected."
            ),
            "Na-Channel Blockers in Dravet (WORSEN)": (
                "Sodium channel-blocking ASMs (carbamazepine, oxcarbazepine, phenytoin, lamotrigine) "
                "WORSEN Dravet syndrome paradoxically. Mechanism: these drugs block all voltage-gated "
                "sodium channels including residual Nav1.1 in inhibitory interneurons → further impair the "
                "already-reduced inhibitory tone → increased seizure frequency and severity. "
                "THIS IS A LIFE-THREATENING PRESCRIBING ERROR in Dravet. Must be avoided absolutely."
            ),
            "Cellular Interference — Obligate Carrier Father": (
                "In PCDH19 epilepsy, hemizygous males (one X with PCDH19 mutation) are clinically UNAFFECTED "
                "but transmit the mutation to 100% of their daughters (who receive his X chromosome). "
                "Each daughter is an obligate heterozygous carrier and has >80% risk of PCDH19 epilepsy. "
                "Sons of an affected mother have 50% risk of inheriting the mutant X — but will be unaffected "
                "hemizygous males (unless somatic mosaic). Genetic counselling MANDATORY."
            ),
        }
    }


# ── Standalone self-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    print("=== DEE-Atlas Overview ===")
    ov = get_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Full name: {ov['full_name']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Genes: {', '.join(ov['gene_list'])}")
    print(f"Mean onset: {ov['mean_onset_age_y']} y")
    print(f"Mean diagnosis age: {ov['mean_diagnosis_age_y']} y")
    print(f"Severity: mild={ov['severity']['mild_pct']}% mod={ov['severity']['moderate_pct']}% severe={ov['severity']['severe_pct']}%")
    print("Clinical features (%):")
    for k, v in ov['clinical_features_prevalence'].items():
        print(f"  {k}: {v}%")
    print("Drug alerts:")
    for a in ov['drug_alerts']:
        print(f"  - {a[:90]}...")
    bk = get_breakdown()
    print(f"\n=== Breakdown: {len(bk['genes'])} genes ===")
    for g, data in bk['genes'].items():
        print(
            f"  {g}: mean onset {data['mean_onset_age_y']}y "
            f"seizure-free={data['seizure_free_pct']}% "
            f"status_hx={data['status_hx_pct']}% "
            f"SUDEP_high={data['sudep_risk_high_pct']}% "
            f"drug_err={data['drug_error_pct']}% "
            f"sex_F={data['sex_pct_female']}%"
        )
    df = get_definitions()
    print(f"\n=== Definitions: {len(df['definitions'])} terms ===")
    for term in df['definitions']:
        print(f"  - {term}")
