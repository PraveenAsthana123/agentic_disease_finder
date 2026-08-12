"""
GLUT1 Deficiency Syndrome (GLUT1-DS / De Vivo Disease) Dashboard
=================================================================
41-patient cohort · SLC2A1 haploinsufficiency · BBB glucose transport defect
GLUT1-DS: SLC2A1 mutation → reduced GLUT1 transporter density at blood-brain barrier
→ chronic cerebral energy insufficiency → early-onset seizures + movement disorder +
cognitive impairment. KEY BIOMARKER: CSF glucose <45 mg/dL + CSF:serum ratio <0.40.
Ketogenic diet is FIRST-LINE DISEASE-MODIFYING therapy (not just seizure control).
VPA is ABSOLUTE CONTRAINDICATION — directly inhibits GLUT1 transporter.
"""

import random
from datetime import datetime

SEED = 7777
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "De novo SLC2A1 pathogenic variant (haploinsufficiency)",
        "n": 23, "pct": 56,
        "category": "De-Novo-SLC2A1",
        "mechanism": (
            "The most common cause of GLUT1-DS: a de novo pathogenic variant in SLC2A1 "
            "(chromosome 1p35-p31.3) encoding glucose transporter type 1. Haploinsufficiency "
            "reduces GLUT1 protein expression at the blood-brain barrier (BBB) endothelial cells "
            "and astrocytic end-feet by approximately 50%. The brain's obligate dependence on "
            "glucose (>90% of cerebral ATP) means that even partial GLUT1 reduction causes "
            "chronic cerebral energy insufficiency — particularly in the neonatal and early "
            "childhood period of rapid brain growth. Variant types: missense (60%), nonsense/stop "
            "(15%), splice-site (10%), small insertion/deletion (10%), large deletion (5%). "
            "Dominant-negative effects (certain missense variants) may produce more severe phenotype "
            "than simple haploinsufficiency. Over 900 pathogenic SLC2A1 variants catalogued "
            "in ClinVar. De novo occurrence: new family variant, parents phenotypically normal."
        ),
        "eeg_correlate": (
            "Interictal: multifocal spike-wave complexes; generalised 2.5–4 Hz spike-wave "
            "(absence-like); focal slowing and sharp waves. Ictal: generalised spike-wave "
            "during absence seizures; focal discharge with secondary generalisation. "
            "Characteristic feature: EEG normalises or improves significantly within weeks of "
            "ketogenic diet initiation (biomarker of metabolic rescue). Morning EEG (fasting) "
            "shows more epileptiform activity than post-prandial recording."
        ),
        "mri_finding": (
            "MRI often NORMAL early in disease — the absence of structural lesion in a child with "
            "early-onset epilepsy and movement disorder should raise GLUT1-DS suspicion. "
            "With disease progression: mild generalised volume loss, delayed myelination on T2 "
            "(hypomyelination pattern, particularly in periventricular white matter). "
            "MR spectroscopy: reduced NAA/Cr ratio reflects cortical neuronal dysfunction. "
            "FDG-PET: generalised hypometabolism with relative thalamic and basal ganglia "
            "preservation — 'inverse' pattern vs. focal epilepsies."
        ),
        "clinical_note": (
            "Classic clinical triad: (1) Early-onset seizures (4–6 months, onset range 1–24M); "
            "(2) Spastic-ataxic or dystonic movement disorder; (3) Developmental delay/intellectual "
            "disability. Acquired microcephaly by 2–3 years. Diurnal fluctuation of symptoms: "
            "worse in the morning (fasting CSF glucose nadir) — pathognomonic. Seizures and "
            "movement disorder triggered or worsened by intercurrent illness (metabolic stress). "
            "Lumbar puncture (after 4h fast): CSF glucose <45 mg/dL + CSF:serum glucose ratio "
            "<0.40 is diagnostic. SLC2A1 sequencing confirms in >95% of classic phenotype."
        ),
    },
    {
        "etiology": "Familial / inherited SLC2A1 variant (autosomal dominant)",
        "n": 8, "pct": 20,
        "category": "Familial-AD",
        "mechanism": (
            "Autosomal dominant inheritance with ~90% penetrance. An affected parent (often mildly "
            "affected, sometimes only with paroxysmal exercise-induced dyskinesia without overt "
            "epilepsy) transmits the pathogenic SLC2A1 variant to 50% of offspring. Variable "
            "expressivity is well-documented: identical SLC2A1 variants within a family can produce "
            "the full classic epileptic-encephalopathy phenotype in one sibling while another has "
            "only PED or absence epilepsy. Anticipation has not been consistently demonstrated. "
            "Familial cases often present at a younger age because heightened clinical awareness "
            "accelerates diagnosis."
        ),
        "eeg_correlate": (
            "Similar to de novo: absence-like generalised spike-wave 2.5–4 Hz; multifocal "
            "discharges; importantly, parental EEG may show only subtle photoparoxysmal response "
            "or mild background irregularity — screening parents' EEG is recommended. "
            "EEG improvement on KD is equally robust in familial cases."
        ),
        "mri_finding": (
            "Familial cases: similar MRI spectrum to de novo — often normal in early/mild cases; "
            "hypomyelination in moderate-severe; mild generalised atrophy in long-standing untreated "
            "cases. Mildly affected parents may have entirely normal brain MRI."
        ),
        "clinical_note": (
            "Family history screening is critical — parents and siblings of an index case should "
            "undergo SLC2A1 sequencing + lumbar puncture if symptomatic. Mildly affected parents "
            "(PED only, no epilepsy) may never have been diagnosed. Genetic counselling: 50% "
            "transmission risk per pregnancy. Pre-implantation genetic testing (PGT) available. "
            "Prenatal diagnosis: chorionic villus sampling or amniocentesis for known familial "
            "variant."
        ),
    },
    {
        "etiology": "SLC2A1 structural variant / large deletion (1p35-p31.3)",
        "n": 5, "pct": 12,
        "category": "Structural-Deletion",
        "mechanism": (
            "Large chromosomal deletions encompassing the SLC2A1 locus at 1p35.3 — detected by "
            "chromosomal microarray (CMA) or whole-genome sequencing (WGS), NOT by standard "
            "Sanger sequencing or gene panel that covers only coding regions and splice sites. "
            "Contiguous gene deletion syndromes may include additional neurodevelopmental genes "
            "flanking SLC2A1, producing a more severe phenotype: intellectual disability beyond "
            "that expected for GLUT1-DS alone, additional dysmorphic features, or autism spectrum "
            "disorder. SLC2A1 duplications (increased copy number) do NOT cause GLUT1-DS — only "
            "deletions / loss-of-function variants."
        ),
        "eeg_correlate": (
            "Structural deletion cases: typically more severe EEG abnormalities — higher "
            "epileptiform burden, more generalised slowing, may show hypsarrhythmia pattern in "
            "early infancy (mimicking West syndrome). Seizure semiology more heterogeneous. "
            "KD still improves EEG in the GLUT1-DS component, but additional co-morbid "
            "epileptic encephalopathy from contiguous gene effects may persist."
        ),
        "mri_finding": (
            "Structural deletion: more frequent white matter abnormalities including hypomyelination "
            "and periventricular leukomalacia pattern. Corpus callosum thinning or dysgenesis "
            "in larger deletions affecting midline structures. More pronounced cerebral atrophy."
        ),
        "clinical_note": (
            "If SLC2A1 sequencing is negative but clinical suspicion high: order CMA (chromosomal "
            "microarray, minimum 400K SNP array) or WGS. Contiguous deletion may explain why "
            "a child has a more severe course than expected. KD treatment is equally indicated "
            "regardless of deletion size — the GLUT1-DS component responds to ketosis."
        ),
    },
    {
        "etiology": "Non-classic / atypical GLUT1-DS (PED-predominant / absence-predominant)",
        "n": 3, "pct": 7,
        "category": "Non-Classic-Atypical",
        "mechanism": (
            "A clinically important minority of SLC2A1 variants produce attenuated phenotypes "
            "without the full classic triad. The two most recognised atypical presentations are: "
            "(1) Paroxysmal exercise-induced dyskinesia (PED) — involuntary dystonic/choreic movements "
            "of limbs triggered by prolonged walking/running, lasting 5–30 minutes, with no or "
            "minimal epilepsy; and (2) Childhood absence epilepsy (CAE) with or without ataxia — "
            "the SLC2A1 variant accounts for ~5% of CAE. A third rare atypical: familial "
            "spastic paraplegia with movement disorder without epilepsy. These attenuated phenotypes "
            "likely reflect SLC2A1 variants with partial GLUT1 function retention (≥30% residual "
            "transport capacity) — sufficient for resting brain glucose but inadequate for metabolic "
            "stress of exercise or focal neural activity bursts during absence seizures."
        ),
        "eeg_correlate": (
            "PED-predominant: EEG may be normal or show only mild nonspecific background changes "
            "without epileptiform discharges — EEG is unremarkable between attacks, confirming "
            "non-epileptic nature of the movement disorder. Absence-predominant: classic 3 Hz "
            "generalised spike-wave (indistinguishable from idiopathic CAE on EEG alone). "
            "CSF glucose ratio is the diagnostic differentiator."
        ),
        "mri_finding": (
            "Atypical/non-classic: brain MRI typically normal; no hypomyelination or atrophy "
            "in the mildest PED-only cases. Absence-predominant atypical GLUT1: may show minimal "
            "or no structural changes."
        ),
        "clinical_note": (
            "PED is pathognomonic of GLUT1-DS when correctly identified: distinguish from familial "
            "hemiplegic migraine, alternating hemiplegia of childhood, and channelopathies "
            "(KCNMA1, PRRT2). Key history: onset after 5–15 minutes of exercise, dystonic quality, "
            "no loss of consciousness, resolves with rest in 30–60 minutes. Even PED-only GLUT1-DS "
            "benefits from KD — motor attacks resolve on ketosis. Do NOT delay KD assuming "
            "PED is 'non-epileptic and non-urgent'."
        ),
    },
    {
        "etiology": "Cryptogenic GLUT1-DS (meets biochemical criteria, SLC2A1 sequencing negative)",
        "n": 2, "pct": 5,
        "category": "Cryptogenic-SLC2A1-Negative",
        "mechanism": (
            "A small but real group of patients with unambiguously low CSF glucose ratio (<0.40) "
            "and characteristic clinical syndrome who test negative on comprehensive SLC2A1 testing "
            "(Sanger + NGS panel + CMA). Possible explanations: (1) Deep intronic SLC2A1 variant "
            "affecting splicing (RNA studies required, not captured on DNA sequencing); "
            "(2) Somatic mosaic SLC2A1 variant (requires deep WGS or single-cell sequencing); "
            "(3) Regulatory region variant in SLC2A1 promoter or enhancer (captured only by WGS); "
            "(4) Pathogenic variants in other genes encoding GLUT1-associated proteins (TfR1, "
            "caveolin-1) — very rare. Treatment principle unchanged: KD is first-line regardless "
            "of genetic confirmation, as the biochemical defect drives management."
        ),
        "eeg_correlate": (
            "Indistinguishable from genetically confirmed GLUT1-DS. EEG response to KD is the "
            "same, providing indirect diagnostic confirmation — EEG improvement within 2–6 weeks "
            "of ketosis supports the diagnosis even when genetic confirmation is lacking."
        ),
        "mri_finding": (
            "MRI spectrum identical to confirmed cases. Hypomyelination and volume loss as per "
            "disease severity and duration of untreated energy deficit."
        ),
        "clinical_note": (
            "Do NOT withhold KD pending genetic confirmation in a child with CSF glucose ratio "
            "<0.40 and consistent phenotype. KD response serves as a therapeutic trial and provides "
            "strong diagnostic support. Refer to a metabolic genetics/inherited metabolic disease "
            "centre for RNA-based or WGS-based re-analysis. Document as 'GLUT1-DS biochemically "
            "confirmed, molecularly unresolved' — not as 'rule out GLUT1-DS'."
        ),
    },
]

# ── Seizure Types (4 types) ───────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Absence-like / Staring spells (with oculomotor features)",
        "prevalence_pct": 70,
        "eeg_correlate": (
            "2.5–4 Hz generalised spike-wave (often slower than classic 3 Hz idiopathic CAE); "
            "eye-opening artefact, oculomotor artefact (nystagmoid movements during ictus); "
            "focal or multifocal onset before secondary generalisation in some. "
            "Post-ictal slowing minimal. EEG markedly improved or normalised on KD."
        ),
        "clinical_tip": (
            "GLUT1-DS absence is distinguished from idiopathic CAE by: morning predominance "
            "(fasting), worsening with illness/fever, associated cognitive/movement disorder, "
            "and absence of response to ethosuximide as sole therapy. "
            "Eye movement abnormalities (opsoclonus, ocular flutter, nystagmus) during or "
            "between seizures are clinically distinctive. Check CSF glucose ratio in any "
            "child with 'CAE' that responds poorly to ETX or has atypical features."
        ),
    },
    {
        "type": "Focal motor / myoclonic seizures",
        "prevalence_pct": 55,
        "eeg_correlate": (
            "Focal sharp waves or spike-and-slow-wave over centroparietal or frontal regions; "
            "generalised myoclonic bursts with polyspike-wave; myoclonic-tonic sequences; "
            "EEG background may show continuous slowing between events, improving with ketosis."
        ),
        "clinical_tip": (
            "Stimulus-sensitive myoclonus common — movement, touch, or startle triggered. "
            "Negative myoclonus (brief lapses of postural tone without positive jerking) "
            "can cause drop attacks mimicking LGS or Doose MAE. "
            "Morning seizure clustering (fasting state) is the key distinguishing feature."
        ),
    },
    {
        "type": "Tonic-clonic / generalised convulsive seizures",
        "prevalence_pct": 40,
        "eeg_correlate": (
            "Generalised paroxysmal fast activity evolving to spike-wave during tonic phase; "
            "rhythmic 2–3 Hz spike-wave during clonic phase; post-ictal suppression. "
            "Background EEG: generalised slowing, improvement with KD initiation."
        ),
        "clinical_tip": (
            "GTCS in GLUT1-DS exacerbated by: fasting, fever, missed KD meals, intercurrent "
            "illness. IV glucose administration during status epilepticus is PARADOXICALLY "
            "harmful — provide ketone supplementation (MCT oil by NG tube if needed). "
            "Emergency KD protocol must be in place for hospitalised GLUT1-DS patients."
        ),
    },
    {
        "type": "Paroxysmal Exercise-Induced Dyskinesia (PED)",
        "prevalence_pct": 35,
        "eeg_correlate": (
            "NORMAL EEG during PED attacks — pathognomonic of non-epileptic mechanism. "
            "Video-EEG monitoring essential to document EEG silence during episodes. "
            "PED attacks are metabolic, not ictal — triggered by exercise-induced glucose "
            "depletion in motor cortex/basal ganglia serving energy-demanding motor circuits."
        ),
        "clinical_tip": (
            "PED attacks: involuntary dystonic or choreic movements of limbs (most often legs), "
            "onset after 5–30 minutes of sustained exercise (walking, running), lasting 5–30 "
            "minutes, resolving spontaneously with rest. No loss of consciousness. "
            "Distinguish from epileptic attacks (EEG monitoring), hemiplegic migraine "
            "(unilateral headache), PRRT2-channelopathy (shorter, fever-triggered). "
            "KD resolves PED in >90% of GLUT1-DS patients within weeks — dramatic response."
        ),
    },
]

# ── Triggers (8 triggers) ─────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fasting / prolonged time since last meal", "seizure_rate_pct": 100,
     "note": "CSF glucose falls progressively with fasting — morning pre-breakfast period is highest-risk window. All GLUT1-DS patients must NEVER fast beyond 8–10 hours. Hospital nil-by-mouth (NBO) orders are potentially harmful."},
    {"trigger": "Morning (before breakfast, peak CSF glucose nadir)", "seizure_rate_pct": 90,
     "note": "Diurnal fluctuation: CSF glucose lowest at 6–8 AM after overnight fast. Early-morning seizure clustering and movement difficulties are clinically diagnostic of GLUT1-DS. Symptoms improve dramatically after KD breakfast."},
    {"trigger": "Fever / intercurrent illness", "seizure_rate_pct": 75,
     "note": "Fever increases cerebral metabolic rate while illness reduces oral intake — double hit on already-limited glucose supply. Fever management: paracetamol (no fasting); maintain KD feeds via NG tube during illness; ketone monitoring required."},
    {"trigger": "Carbohydrate-containing meals without ketosis (pre-diagnosis)", "seizure_rate_pct": 65,
     "note": "Before KD initiation: glucose ingestion transiently raises blood glucose but CSF glucose remains low relative to demand. Post-glucose spike followed by relative CSF glucose trough worsens seizure threshold."},
    {"trigger": "Intercurrent illness / reduced oral intake", "seizure_rate_pct": 55,
     "note": "Illness-related reduced feeding without KD backup leads to metabolic crisis. Sick-day management plan essential: NG tube KD formula, electrolyte monitoring, hospital admission threshold lower than for other epilepsies."},
    {"trigger": "Prolonged exercise (without ketone supplementation)", "seizure_rate_pct": 40,
     "note": "Exercise depletes local cortical and basal ganglia glucose → PED and seizure risk. On KD: BHB provides exercise fuel; MCT supplementation 30 minutes before exercise further reduces PED. Non-KD patients: carry glucose-free snack with MCT oil supplement."},
    {"trigger": "Sleep deprivation", "seizure_rate_pct": 30,
     "note": "Sleep deprivation increases cerebral metabolic demand and reduces seizure threshold — additive to chronic energy deficiency of GLUT1-DS. Consistent sleep schedule mandatory. Overnight fasting management critical."},
    {"trigger": "Hospital nil-by-mouth (NBM) orders / surgical fasting", "seizure_rate_pct": 85,
     "note": "CRITICAL SAFETY ALERT: standard surgical NBM protocols (4–6h fasting) are DANGEROUS in GLUT1-DS. Hospital emergency protocol must specify: continue KD formula via NG tube, IV dextrose-free saline (NOT glucose-containing IV fluids), ketone monitoring every 2–4 hours. Alert ID bracelet mandatory."},
]

# ── Treatments (8 treatments) ─────────────────────────────────────────────────
TREATMENTS = [
    {
        "name": "Classic Ketogenic Diet (4:1 ratio)",
        "evidence_level": "Level A (ILAE 2018, AAN 2020)",
        "dose": "Fat:Protein+Carb ratio 4:1 (90% fat kcal); total carbohydrate <10g/day; protein 1–2g/kg/day; calories age-appropriate; fluid 1–1.2 mL/kcal",
        "moa": (
            "Ketone bodies (beta-hydroxybutyrate BHB + acetoacetate AcAc) cross the BBB via "
            "MCT1/MCT2 transporters (GLUT1-independent pathway) and serve as alternative "
            "cerebral fuel, bypassing the GLUT1 transport defect. BHB provides 4 kcal/g vs "
            "glucose 3.75 kcal/g and is efficiently metabolised via acetyl-CoA → TCA cycle. "
            "KD is DISEASE-MODIFYING in GLUT1-DS — not merely seizure control. Ketosis "
            "corrects the cerebral energy deficit, improving cognition, movement disorder, and "
            "EEG simultaneously. Target BHB: 2–4 mmol/L. Classical KD achieves highest ketosis "
            "but requires strict dietitian supervision."
        ),
        "efficacy": "Seizure freedom: 30–50%; ≥50% seizure reduction: 60–75%; PED resolution: >90%; EEG improvement: 70–85%; cognitive improvement: 50–65% (if initiated early)",
        "safety": "Growth monitoring (height/weight monthly), bone density (annual DEXA), renal ultrasound (kidney stones 5–8%), fasting lipids (q6M), selenium/zinc/carnitine levels, vitamin D supplementation mandatory",
        "monitoring": "Urinary ketones daily (target 3–4+), BHB blood meter twice weekly, growth charts monthly, lipid panel q6M, selenium/zinc/carnitine annually, DEXA q2Y",
        "contraindications_in_glut1ds": "None for classic KD in GLUT1-DS specifically — it is the TREATMENT",
        "clinical_alert": "KD must be initiated as early as possible — delay worsens cognitive and developmental outcomes. Pre-KD evaluation: metabolic bloodwork, renal USS, baseline lipids, nutritional assessment by specialist dietitian."
    },
    {
        "name": "Modified Atkins Diet (MAD)",
        "evidence_level": "Level B (Klepper 2020, ILAE 2018)",
        "dose": "Carbohydrate 10–20g/day (NOT weighed ratio); fat encouraged (60–65% kcal); protein unrestricted; total calorie unrestricted; no defined KD ratio",
        "moa": (
            "Carbohydrate restriction to 10–20g/day sufficient to induce nutritional ketosis "
            "(BHB 1–3 mmol/L) in most patients, providing alternative cerebral fuel via MCT1/2. "
            "Less ketogenic than 4:1 classical KD (lower BHB levels) but more flexible, "
            "allowing greater dietary variety and easier adherence — particularly in adolescents "
            "and adults with atypical GLUT1-DS (PED-predominant). Transition from classic KD "
            "to MAD at school age or adolescence is a common management strategy."
        ),
        "efficacy": "≥50% seizure reduction: 50–65% (slightly less than classic KD); PED resolution: 75–85%; EEG improvement: 60–75%; cognitive stability maintenance",
        "safety": "Easier to maintain than 4:1 KD; similar monitoring required; lower risk of extreme hyperlipidaemia vs. 4:1 ratio",
        "monitoring": "BHB blood meter twice weekly, lipid panel q6M, growth monitoring, carnitine/selenium/zinc annually",
        "contraindications_in_glut1ds": "None — MAD is an acceptable alternative to classic KD, particularly for older children and adults",
        "clinical_alert": "MAD may be insufficient in infants with severe GLUT1-DS — classic 4:1 KD preferred in the first 3 years of life for maximum disease-modifying effect."
    },
    {
        "name": "MCT (Medium-Chain Triglyceride) Ketogenic Diet",
        "evidence_level": "Level B (Neal 2009, ILAE 2018)",
        "dose": "MCT oil provides 45–60% of total calories; remaining calories as regular food; lower overall fat ratio vs. classic KD; MCT:LCT ratio adjusted by dietitian for GI tolerance",
        "moa": (
            "MCT (C8 caprylic acid + C10 capric acid) are more rapidly absorbed and converted "
            "to ketone bodies in the liver vs. long-chain triglycerides (LCT). The increased "
            "ketogenic efficiency of MCT allows a lower overall fat:carbohydrate ratio in the "
            "diet, permitting more protein and carbohydrate than classic KD — improving palatability "
            "and dietary variety. Caprylic acid (C8) has additional direct anticonvulsant effects "
            "independent of ketosis. C10 has anti-inflammatory CNS effects."
        ),
        "efficacy": "Similar efficacy to classic KD for seizure control; preferred in some patients for tolerability; PED resolution comparable",
        "safety": "GI side effects (nausea, diarrhoea, vomiting) at high MCT loads — dose-titration required; avoid >60% MCT of calories",
        "monitoring": "Ketone monitoring as for classic KD; GI tolerance diary; lipids q6M",
        "contraindications_in_glut1ds": "None specific; GI intolerance limits dose in some patients",
        "clinical_alert": "MCT diet requires specialised paediatric dietitian experienced in ketogenic therapies. Do not self-prescribe MCT oil without dietitian supervision — ratio miscalculation can cause hypo- or insufficient ketosis."
    },
    {
        "name": "Low Glycemic Index Treatment (LGIT)",
        "evidence_level": "Level C (Pfeifer 2008, expert consensus)",
        "dose": "Glycemic index <50 foods; fat 60% kcal; carbohydrate 40–60g/day (low GI only); protein unrestricted; no ratio weighing; calories unrestricted",
        "moa": (
            "Low glycemic index carbohydrates produce slower, lower-amplitude glucose peaks — "
            "reducing post-prandial glucose surges and troughs in CSF. Induces mild-to-moderate "
            "nutritional ketosis (BHB 0.5–1.5 mmol/L — lower than classic KD). Less effective "
            "for seizure control but may adequately manage PED in non-classic GLUT1-DS and "
            "improve quality of life in patients unable to sustain strict KD. Useful as "
            "'bridge therapy' during KD breaks or for partial responders."
        ),
        "efficacy": "Less effective than KD for seizures; suitable for PED management and cognitive support in non-classic/atypical phenotype; low adherence dropout",
        "safety": "Best-tolerated dietary approach; minimal dietary restrictions vs. classic KD; similar nutritional monitoring needed",
        "monitoring": "Urine/blood ketones weekly, lipids q6M, carnitine/selenium annually",
        "contraindications_in_glut1ds": "LGIT alone insufficient for classic severe GLUT1-DS — use only for non-classic phenotype or transitional management",
        "clinical_alert": "Do not use LGIT as first-line in classic GLUT1-DS with epileptic encephalopathy. Reserve for: (1) atypical PED-only phenotype, (2) transition between KD protocols, (3) dietary adherence-failure bridge."
    },
    {
        "name": "Triheptanoin (C7 odd-chain fatty acid / anaplerotic therapy)",
        "evidence_level": "Level C (Mochel 2016, compassionate/investigational)",
        "dose": "1–3 g/kg/day in 4 divided doses orally; titrate over 2–4 weeks for GI tolerance; administered with meals",
        "moa": (
            "Triheptanoin provides C5 ketone bodies (beta-hydroxypentanoate + beta-ketopentanoate) "
            "as well as propionyl-CoA — an anaplerotic substrate that replenishes TCA cycle "
            "intermediates (succinyl-CoA, oxaloacetate). In GLUT1-DS, glucose deprivation depletes "
            "not only acetyl-CoA but also TCA cycle carbon — triheptanoin specifically corrects "
            "the anaplerotic deficit, complementing even-chain MCT/KD therapy. "
            "Provides alternative mitochondrial fuel without glucose dependency. "
            "Investigational — limited evidence base but promising compassionate use data "
            "in severe refractory GLUT1-DS."
        ),
        "efficacy": "Limited controlled data; case series suggest improvement in movement disorder and fatigue even when KD-insufficient; C5 ketones cross BBB effectively",
        "safety": "GI side effects (nausea, cramping, diarrhoea); liver enzyme monitoring required; avoid in carnitine deficiency (correct first)",
        "monitoring": "Liver enzymes q3M, free carnitine (supplement if low), urine C5 ketones, GI symptom diary",
        "contraindications_in_glut1ds": "Carnitine deficiency (relative CI — correct first); not a substitute for KD in classic GLUT1-DS",
        "clinical_alert": "Use as adjunct to KD, not replacement. Investigational status varies by jurisdiction — check local regulatory requirements. Enrol in registry/study if available."
    },
    {
        "name": "Levetiracetam (LEV) — adjunct only",
        "evidence_level": "Level C (off-label adjunct; Leen 2010 case series)",
        "dose": "10–40 mg/kg/day divided twice daily; standard paediatric AED dosing",
        "moa": (
            "SV2A synaptic vesicle glycoprotein 2A modulation reduces abnormal neuronal "
            "hypersynchrony without affecting basal metabolic pathways. Does NOT inhibit GLUT1 "
            "transporter. Safe to combine with KD — no pharmacokinetic interactions with "
            "ketone bodies. LEV provides marginal seizure-control benefit as add-on when KD "
            "alone is insufficient, particularly for myoclonic component."
        ),
        "efficacy": "Modest add-on benefit; does not address underlying GLUT1 energy deficit; cognitive/behavioural side effects (irritability, aggression) more frequent in developmental encephalopathies",
        "safety": "Behavioural side effects: irritability 15%, aggression 10%, depression 5% — monitor with CBCL/Conners scales; renal dosing in renal impairment; no hepatotoxicity",
        "monitoring": "CBCL/PHQ-9 behavioural monitoring q3M, renal function annually",
        "contraindications_in_glut1ds": "Not a substitute for KD — do not add LEV and delay KD initiation",
        "clinical_alert": "In GLUT1-DS, LEV may be considered as a bridge while KD is being established, or for residual seizures on KD. NEVER as monotherapy — insufficient for underlying metabolic deficit."
    },
    {
        "name": "Clonazepam (CLN) — adjunct for myoclonic component only",
        "evidence_level": "Level C (expert consensus, Klepper 2020 review)",
        "dose": "0.01–0.05 mg/kg/day divided twice daily; target lowest effective dose",
        "moa": (
            "GABA-A receptor positive allosteric modulator — enhances chloride channel opening "
            "frequency, hyperpolarising post-synaptic neuron. Effective for myoclonic component "
            "of GLUT1-DS seizures. Does not interact with GLUT1 transporter. Tolerance develops "
            "with chronic use. Combination with KD: no pharmacokinetic interaction; KD may allow "
            "lower CLN dose over time."
        ),
        "efficacy": "Effective for myoclonic-tonic component; tolerance limits long-term use; KD allows dose reduction/taper in responders",
        "safety": "Sedation, hypotonia, drooling, tolerance, withdrawal risk; respiratory depression with high doses; avoid abrupt discontinuation",
        "monitoring": "UMRS myoclonus rating scale, Berg Balance Scale, drowsiness diary, respiratory assessment; taper schedule at 6–12M if seizure control achieved on KD",
        "contraindications_in_glut1ds": "Use only for persistent myoclonic component after KD optimisation; long-term clonazepam without concurrent KD is inadequate management",
        "clinical_alert": "GLUT1-DS myoclonus should respond to KD within 4–8 weeks. If myoclonus persists on adequate KD (BHB 2–4 mmol/L), add CLN as adjunct. Plan taper at 12M KD anniversary if seizure-free."
    },
    {
        "name": "Valproate (VPA) — ABSOLUTE CONTRAINDICATION",
        "evidence_level": "ABSOLUTE CONTRAINDICATION (Class I evidence for harm)",
        "dose": "NEVER USE IN GLUT1-DS",
        "moa": (
            "Valproate directly inhibits GLUT1 glucose transporter — pharmacological mechanism "
            "established in vitro (Adkison 1994 J Pharmacol Exp Ther) and in vivo human studies. "
            "VPA reduces erythrocyte glucose transport by 20–40% and cerebrospinal fluid GLUT1 "
            "activity by 15–30%. In a patient already haploinsufficient for GLUT1, VPA "
            "adds a second functional insult to an already critically compromised transporter — "
            "precipitating acute metabolic crisis with status epilepticus paradoxically worsened "
            "by the prescribed anticonvulsant. Mechanism: VPA binds to GLUT1 extracellular "
            "glucose-binding pocket, competitively inhibiting glucose translocation."
        ),
        "efficacy": "Paradoxically HARMFUL — worsens the GLUT1-DS-defining metabolic defect despite anticonvulsant properties",
        "safety": "ABSOLUTE CONTRAINDICATION: acute worsening of seizures, metabolic encephalopathy, irreversible neurological deterioration reported with VPA in confirmed GLUT1-DS",
        "monitoring": "N/A — do not prescribe. If inadvertently started, STOP IMMEDIATELY and initiate KD.",
        "contraindications_in_glut1ds": "ABSOLUTE CONTRAINDICATION — must be documented prominently in all medical records, allergy/drug alert systems, and patient ID bracelet",
        "clinical_alert": "⚠️ CRITICAL SAFETY ALERT: VPA is the most dangerous drug in GLUT1-DS. Any neurologist prescribing VPA for a seizure disorder in childhood/young adulthood should first consider GLUT1-DS and check CSF glucose ratio. MEDIC ALERT bracelet: 'GLUT1-DS — NO VALPROATE'."
    },
]

# ── Monitoring Protocol (5 items) ─────────────────────────────────────────────
MONITORING_PROTOCOL = [
    {
        "item": "Blood BHB ketone monitoring",
        "frequency": "Twice weekly (fasting morning + 2h post-meal) on KD; daily during illness",
        "target": "2–4 mmol/L (adequate therapeutic ketosis); <1 mmol/L = insufficient; >6 mmol/L = risk of acidosis",
        "rationale": "BHB is the therapeutic biomarker — confirms adequate alternative cerebral fuel delivery. Ketosis target is non-negotiable in classic GLUT1-DS."
    },
    {
        "item": "CSF glucose ratio monitoring",
        "frequency": "At diagnosis (lumbar puncture after 4h fast); repeat at 12M if initial equivocal; not required for ongoing KD monitoring",
        "target": "<0.40 (diagnostic); <0.35 (definitive); >0.45 (non-diagnostic — reconsider diagnosis)",
        "rationale": "CSF:serum glucose ratio is the gold-standard diagnostic test. Normal serum glucose at time of LP is mandatory — hyperglycaemia will falsely elevate ratio."
    },
    {
        "item": "Growth and nutritional monitoring",
        "frequency": "Height/weight monthly for first year; q3M thereafter; dietitian review q3M",
        "target": "Growth on or above pre-KD trajectory; no acceleration of height loss relative to peers",
        "rationale": "Classic 4:1 KD is calorie-dense but protein-restricted — risk of growth deceleration. Monthly monitoring allows early dietary adjustment."
    },
    {
        "item": "Lipid panel, selenium, zinc, carnitine",
        "frequency": "Lipids q6M; selenium, zinc, free carnitine, 25-OH vitamin D — annually",
        "target": "LDL <3.5 mmol/L; selenium >80 µg/L; zinc >9.2 µmol/L; free carnitine >15 µmol/L; Vit D >50 nmol/L",
        "rationale": "High-fat diet raises LDL — statin not routinely needed in children but monitor. Selenium deficiency on KD causes cardiomyopathy (supplement routinely). Carnitine required for long-chain fat oxidation — supplement if deficient."
    },
    {
        "item": "Renal ultrasound + uric acid",
        "frequency": "Annually; earlier if haematuria, loin pain, or recurrent UTI",
        "target": "No nephrolithiasis; uric acid <450 µmol/L; urine Ca:Cr ratio <0.8",
        "rationale": "Kidney stones (calcium oxalate + uric acid) occur in 5–8% of KD patients. Citrate supplementation (potassium citrate) reduces stone risk. Adequate hydration (minimum 1.2 mL/kcal) is mandatory."
    },
]

# ── 6-Window Lifecycle ─────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Neonatal / Early Infancy (0–6 months)",
        "phase": "Pre-diagnostic / prodromal",
        "description": (
            "GLUT1-DS rarely presents at birth — the neonatal period may show only subtle signs: "
            "abnormal eye movements (opsoclonus, ocular flutter, nystagmus), feeding difficulties, "
            "and irritability. Seizures typically begin at 3–6 months. The delay between birth and "
            "GLUT1-DS symptom onset reflects the transition from neonatal cerebral metabolism "
            "(uses lactate as primary fuel) to glucose-dependent mature brain metabolism. "
            "If GLUT1-DS suspected (positive family history): early SLC2A1 testing + CSF LP + "
            "prophylactic KD before seizure onset dramatically improves cognitive outcomes."
        ),
        "key_actions": "Family history screen; SLC2A1 testing in at-risk infants; baseline developmental assessment; KD initiation if confirmed"
    },
    {
        "window": "Early Childhood (6 months – 3 years)",
        "phase": "Seizure onset / diagnosis / KD initiation",
        "description": (
            "Peak diagnostic period — majority of classic GLUT1-DS diagnosed here. Characteristic "
            "seizure clusters in the morning (fasting), oculomotor abnormalities, developmental "
            "slowing, and acquired microcephaly. Lumbar puncture after 4h fast shows CSF glucose "
            "<45 mg/dL + ratio <0.40. SLC2A1 sequencing confirms. URGENT KD initiation — "
            "every month of delay at this age costs cognitive developmental opportunity. "
            "Classic 4:1 KD with specialist dietitian. Family education on emergency protocols, "
            "NBM danger, and VPA prohibition."
        ),
        "key_actions": "Urgent CSF LP + SLC2A1 sequencing; dietitian-supervised classic 4:1 KD; emergency protocol card; VPA prohibition documented; multidisciplinary team (neurology + dietitian + developmental paediatrics)"
    },
    {
        "window": "School Age (3–12 years)",
        "phase": "Active KD management / cognitive support",
        "description": (
            "KD established — focus on maintenance, growth monitoring, and educational support. "
            "Seizure freedom or significant reduction expected in 30–50% of children by this phase. "
            "Cognitive outcomes strongly correlated with age of KD initiation and adequacy of "
            "ketosis: children starting KD before 2 years achieve higher IQ than those starting "
            "after 3 years. Learning support (individual education plan, IEP) required — "
            "processing speed, working memory, and attention most affected. Annual neuropsychological "
            "assessment. KD ratio may be liberalised from 4:1 to 3:1 as brain growth slows. "
            "Sports participation: ketone supplementation before physical education."
        ),
        "key_actions": "Neuropsychological assessment annually; IEP in school; KD adherence monitoring; BHB targets maintained; sports protocol (MCT supplement pre-exercise); DEXA bone density"
    },
    {
        "window": "Adolescence (12–18 years)",
        "phase": "Dietary adherence challenges / KD liberalisation / PED management",
        "description": (
            "Dietary adherence is the greatest challenge — peer pressure, social eating, and "
            "increased caloric needs of puberty. Transition from classic 4:1 KD to Modified Atkins "
            "Diet (MAD) or LGIT may improve quality of life while maintaining adequate ketosis. "
            "PED becomes more problematic with increased physical education and social sports. "
            "Cognitive challenges: executive function, memory, and processing speed continue to "
            "affect academic performance. Mental health monitoring: adolescents with chronic dietary "
            "restrictions at higher risk of depression and anxiety. Driving: avoid until ≥12M "
            "seizure-free + neurologist clearance. Transition planning for adult services begins."
        ),
        "key_actions": "KD→MAD transition if indicated; adherence counselling; PED exercise protocol; mental health screen (PHQ-9); transition planning; driving restriction counselling"
    },
    {
        "window": "Young Adult (18–35 years)",
        "phase": "Long-term KD maintenance / adult transition / reproductive counselling",
        "description": (
            "Adults with GLUT1-DS require lifelong dietary management — KD/MAD/LGIT depending on "
            "phenotype severity. The atypical GLUT1-DS adult with PED-predominant phenotype may "
            "manage on LGIT alone. Classic phenotype requires sustained ketosis. Adult neurology "
            "transition: adult neurologists familiar with GLUT1-DS rare — may require specialist "
            "centre follow-up. Reproductive counselling: 50% autosomal dominant transmission risk; "
            "KD safe in pregnancy (limited data, specialist supervision essential — fetal ketone "
            "effects uncertain); genetic counselling and PGT available. Employment: cognitive "
            "impairment and fatigue may affect work capacity — occupational assessment recommended."
        ),
        "key_actions": "Adult neurology transition; reproductive/genetic counselling; MAD or LGIT maintenance; occupational health assessment; driving review annually; SLC2A1 testing for partner if family planning"
    },
    {
        "window": "Older Adult (35+ years)",
        "phase": "Chronic management / cognitive trajectory / comorbidity",
        "description": (
            "Limited long-term data in adults with GLUT1-DS (condition described 1991, first "
            "cohort only now reaching middle age). Available evidence suggests: cognitive outcomes "
            "plateau in adulthood — determined primarily by age of KD initiation; PED may "
            "improve spontaneously in some adults; seizure control maintained on dietary therapy "
            "in majority. Emerging concerns: premature cognitive decline? Long-term KD "
            "cardiovascular effects? Bone density on lifelong KD? Annual metabolic monitoring "
            "essential. Polypharmacy review: any new prescription must be checked against GLUT1 "
            "transport inhibition (VPA absolute CI remains lifelong)."
        ),
        "key_actions": "Annual metabolic review; DEXA bone density q2Y; cognitive assessment; cardiovascular risk monitoring; medication review for GLUT1 inhibitors; palliative care planning for severely affected"
    },
]

# ── 14 Concepts ──────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "GLUT1 (Glucose Transporter Type 1)", "definition": "SLC2A1-encoded integral membrane protein responsible for facilitated glucose transport across the blood-brain barrier (BBB) endothelial cells and astrocytic end-feet. GLUT1 operates by an alternating-access mechanism, transporting glucose down its concentration gradient. The brain isoform (55 kDa) is constitutively expressed at the BBB. Haploinsufficiency reduces transporter density by ~50%, insufficient for the cerebral energy demands of a developing child."},
    {"term": "De Vivo Disease", "definition": "Eponym for GLUT1-DS, named after Darryl De Vivo (Columbia University), who first described the syndrome in 1991 in two children with early-onset seizures, hypoglycorrhachia, and movement disorder. The original NEJM paper established CSF:serum glucose ratio <0.4 as the diagnostic biomarker."},
    {"term": "Haploinsufficiency", "definition": "State in which a single functional copy of a gene is insufficient to produce the required amount of protein for normal cellular function. GLUT1-DS is a haploinsufficiency syndrome — one SLC2A1 allele is pathogenic, producing ~50% of normal GLUT1 transporter density at the BBB. This is below the functional threshold for adequate cerebral glucose delivery."},
    {"term": "Hypoglycorrhachia", "definition": "Low glucose in cerebrospinal fluid (CSF) in the absence of hypoglycaemia. In GLUT1-DS: CSF glucose <45 mg/dL + CSF:serum glucose ratio <0.40 (after 4h fasting, with normal serum glucose). Hypoglycorrhachia in the context of normal serum glucose specifically implicates impaired glucose transport across the BBB — the biochemical hallmark of GLUT1-DS."},
    {"term": "CSF:Serum Glucose Ratio", "definition": "Diagnostic ratio calculated as (CSF glucose mg/dL) ÷ (simultaneous serum glucose mg/dL). Normal: 0.45–0.85. GLUT1-DS diagnostic threshold: <0.40. Method: lumbar puncture after minimum 4-hour fast (longer fast lowers ratio further — improves sensitivity); serum glucose drawn within 30 minutes of LP; avoid glucose-containing IV fluids before LP. Values 0.40–0.45 are equivocal — repeat LP or proceed to SLC2A1 sequencing."},
    {"term": "Ketone Bodies (BHB, AcAc)", "definition": "Beta-hydroxybutyrate (BHB) and acetoacetate (AcAc) produced by hepatic fatty acid oxidation during carbohydrate restriction. Cross the BBB via MCT1/MCT2 monocarboxylate transporters (GLUT1-independent pathway). BHB is the primary circulating ketone on ketogenic diet (60–75% of total ketones). AcAc is the spontaneously measured urinary ketone. On KD in GLUT1-DS, BHB serves as the primary cerebral fuel, correcting the energy deficit caused by impaired GLUT1-mediated glucose import. Target BHB: 2–4 mmol/L."},
    {"term": "Paroxysmal Exercise-Induced Dyskinesia (PED)", "definition": "Non-epileptic movement disorder: involuntary dystonic, choreic, or ballistic movements of the limbs triggered by sustained physical exercise (walking, running). Onset after 5–30 minutes of exercise; duration 5–30 minutes; resolution with rest. EEG normal during attacks. In GLUT1-DS, PED is caused by exercise-induced glucose depletion in glucose-starved motor cortex and basal ganglia. Pathognomonic feature: responds dramatically to ketogenic diet (>90% attack freedom). Distinguish from epileptic motor seizures by video-EEG monitoring."},
    {"term": "Ketogenic Diet (KD)", "definition": "High-fat, low-carbohydrate, adequate-protein dietary therapy that induces sustained nutritional ketosis. In GLUT1-DS, KD is DISEASE-MODIFYING (not merely anticonvulsant) — by providing BHB as alternative cerebral fuel, KD corrects the pathophysiological cerebral energy deficit. ILAE 2018 and AAN 2020 classify KD as Level A evidence for GLUT1-DS — the highest evidence level for any dietary therapy in any epilepsy syndrome. Variants: Classic 4:1, MCT, Modified Atkins, LGIT."},
    {"term": "Anaplerosis", "definition": "Replenishment of TCA (Krebs cycle) intermediate metabolites — particularly oxaloacetate, α-ketoglutarate, and succinyl-CoA — to maintain TCA cycle capacity. In GLUT1-DS, glucose deprivation depletes not only acetyl-CoA but also TCA carbon skeletons. Triheptanoin (C7 odd-chain fatty acid) provides propionyl-CoA as an anaplerotic substrate, specifically correcting TCA intermediate depletion beyond what even-chain ketones can provide."},
    {"term": "Opsoclonus / Oculomotor Apraxia", "definition": "Opsoclonus: rapid, conjugate, multi-directional saccadic eye movements ('dancing eyes') occurring in bursts. Oculomotor apraxia: inability to initiate voluntary horizontal saccades while reflex saccades (vestibulo-ocular) are preserved. Both are common in GLUT1-DS, particularly in the pre-diagnostic period and during fasting-triggered metabolic crises. They represent brainstem/cerebellar glucose starvation affecting the oculomotor control circuitry. Resolve or markedly improve with KD."},
    {"term": "Acquired Microcephaly", "definition": "Progressive reduction in head circumference (OFC) below the 3rd centile, occurring AFTER normal birth head circumference. In GLUT1-DS, acquired microcephaly results from impaired synaptic proliferation, dendritic arborisation, and myelination due to chronic cerebral energy deficiency in the critical early brain growth window (0–3 years). Rate of OFC deceleration correlates with KD delay — early KD initiation prevents acquired microcephaly or reduces its severity."},
    {"term": "Beta-hydroxybutyrate (BHB) Target Range", "definition": "The therapeutic BHB level on ketogenic diet for GLUT1-DS: 2–4 mmol/L. Measurement: fingerpick blood ketone meter (Precision Xtra or equivalent), most accurate method; urine ketone dipstick provides proxy but less precise. BHB <1 mmol/L = insufficient ketosis (increase dietary fat ratio, check for hidden carbohydrates). BHB >6 mmol/L = risk of acidosis (reduce fat ratio, check for illness/dehydration). Monitoring frequency: twice weekly fasting BHB; daily during illness."},
    {"term": "SUDEP Risk in GLUT1-DS", "definition": "Sudden Unexpected Death in Epilepsy (SUDEP). GLUT1-DS patients with uncontrolled seizures carry standard epilepsy SUDEP risk factors (nocturnal GTCS, drug-resistant epilepsy). A specific GLUT1-DS consideration: nocturnal glucose deprivation (overnight fasting) may exacerbate nocturnal seizures and postictal respiratory depression. KD with adequate overnight ketosis should provide continuous cerebral fuel supply during sleep, potentially reducing SUDEP risk. SUDEP rate in well-controlled KD-treated GLUT1-DS: likely lower than uncontrolled epilepsy population, but specific data limited."},
    {"term": "Valproate-GLUT1 Inhibition", "definition": "Valproic acid (VPA, valproate) directly inhibits GLUT1 glucose transporter function by binding to the extracellular glucose-binding domain. Pharmacological studies (Adkison 1994; Klepper 2001) demonstrate VPA reduces erythrocyte GLUT1-mediated glucose transport by 20–40% in vitro and reduces CSF glucose ratio in vivo. In GLUT1-DS patients (already 50% GLUT1-deficient), VPA adds a second inhibitory insult, causing acute worsening of cerebral glucose deficiency — paradoxically provoking status epilepticus and metabolic encephalopathy. ABSOLUTE CONTRAINDICATION in all GLUT1-DS patients, lifelong."},
]

# ── 6 Standards ────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE 2022 Epilepsy Syndrome Classification", "domain": "Diagnosis", "relevance": "Classifies GLUT1-DS as a metabolic epilepsy syndrome with specific aetiological (SLC2A1) and biomarker (CSF:serum ratio) criteria. Distinguishes from structural, immune, and genetic generalised epilepsies requiring different management."},
    {"name": "NICE NG217 Epilepsies (2022)", "domain": "Clinical Guideline", "relevance": "Recommends ketogenic diet as first-line treatment for GLUT1-DS and PDHD; specifies urgent referral to paediatric KD specialist; mandates VPA avoidance in GLUT1-DS."},
    {"name": "ILAE Dietary Therapies Commission 2018 (van der Louw et al.)", "domain": "KD Protocol Standard", "relevance": "Defines evidence levels for dietary therapies by epilepsy syndrome. Assigns Level A evidence (highest) to ketogenic diet for GLUT1-DS and PDHD — only two syndromes with Level A KD evidence. Specifies pre-KD evaluation, initiation, and monitoring protocols."},
    {"name": "American Academy of Neurology (AAN) KD Practice Parameter 2020", "domain": "Practice Guideline", "relevance": "Level A evidence for KD in GLUT1-DS. Specifies KD as first-line disease-modifying therapy; details monitoring requirements including BHB targets, nutritional parameters, and growth monitoring."},
    {"name": "GLUT1 Deficiency Syndrome International Study Group Consensus (Leen 2010, Ann Neurol)", "domain": "Diagnostic Criteria", "relevance": "Establishes CSF:serum glucose ratio <0.40 as primary diagnostic criterion; defines clinical phenotypic spectrum (classic, non-classic, PED-only); specifies SLC2A1 testing pathway; emergency protocol for hospitalised patients."},
    {"name": "ACMG/AMP Variant Interpretation Guidelines (Richards 2015)", "domain": "Genetic Variant Classification", "relevance": "Framework for classifying SLC2A1 variants as pathogenic/likely pathogenic/uncertain/benign. Essential for: distinguishing causative GLUT1-DS variants from polymorphisms; counselling families about variant significance; resolving cryptogenic cases with variants of uncertain significance (VUS)."},
]

# ── 8 Thresholds ───────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "CSF glucose <45 mg/dL", "category": "Diagnosis", "action": "GLUT1-DS diagnostic — proceed to SLC2A1 sequencing + urgent KD initiation"},
    {"name": "CSF:serum glucose ratio <0.40", "category": "Diagnosis", "action": "Diagnostic threshold (measured after 4h fast with simultaneous serum glucose; 0.40–0.45 equivocal — repeat)"},
    {"name": "Blood BHB 2–4 mmol/L", "category": "KD Therapeutic Target", "action": "Adequate therapeutic ketosis range on KD; <1 mmol/L = insufficient (increase fat ratio); >6 mmol/L = acidosis risk"},
    {"name": "KD ratio 4:1 (fat:carb+protein, g:g)", "category": "KD Protocol", "action": "Classic KD ratio for maximum ketosis in infants/toddlers; may reduce to 3:1 after age 3–4Y; specialist dietitian adjustment"},
    {"name": "Carbohydrate <20g/day (Modified Atkins)", "category": "MAD Protocol", "action": "Minimum carbohydrate restriction for MAD ketosis in school-age/adolescent/adult GLUT1-DS; higher fat liberalises carb threshold"},
    {"name": "Seizure-free ≥2 years on KD", "category": "Liberalisation Trigger", "action": "Consider KD ratio reduction (4:1→3:1→MAD) with ongoing BHB monitoring; do NOT discontinue KD entirely in classic GLUT1-DS"},
    {"name": "Driving restriction: 12M seizure-free + neurologist clearance", "category": "Safety", "action": "Standard seizure-freedom driving threshold; GLUT1-DS patients must also demonstrate adequate fasting KD management before driving clearance"},
    {"name": "Hospital fasting protocol: KD formula via NG if NPO >4h", "category": "Emergency Safety", "action": "Hospital NBM orders are DANGEROUS in GLUT1-DS — mandatory NG KD formula continuation, dextrose-free IV fluids (0.9% saline), BHB monitoring q4h during any surgical or nil-by-mouth period"},
]

# ── 6 References ────────────────────────────────────────────────────────────────
REFERENCES = [
    {"authors": "De Vivo DC, Trifiletti RR, Jacobson RI, Ronen GM, Behmand RA, Harik SI", "year": 1991, "title": "Defective glucose transport across the blood-brain barrier as a cause of persistent hypoglycorrhachia, seizures, and developmental delay", "journal": "N Engl J Med", "vol": "325", "pages": "703–709", "pmid": "1714544", "note": "Original description of GLUT1-DS — established hypoglycorrhachia biomarker and clinical triad"},
    {"authors": "Leen WG, Klepper J, Verbeek MM, et al.", "year": 2010, "title": "Glucose transporter-1 deficiency syndrome: the expanding clinical and genetic spectrum of a treatable disorder", "journal": "Brain", "vol": "133", "pages": "655–670", "pmid": "20200115", "note": "Largest natural history study — 132 patients; defines phenotypic spectrum (classic/non-classic/PED); confirms CSF ratio diagnostic criteria"},
    {"authors": "Klepper J, Akman C, Armeno M, et al.", "year": 2020, "title": "Glut1 Deficiency Syndrome (Glut1DS): State of the art in 2020 and recommendations of the international Glut1DS study group", "journal": "Epilepsia Open", "vol": "5", "pages": "354–365", "pmid": "32913944", "note": "Current international consensus recommendations — KD first-line, monitoring protocols, VPA absolute CI, phenotypic classification"},
    {"authors": "Wang D, Pascual JM, Yang H, Engelstad K, Mao X, Truong J, De Vivo DC", "year": 2005, "title": "A mouse model for Glut-1 haploinsufficiency", "journal": "Hum Mol Genet", "vol": "14", "pages": "1869–1883", "pmid": "15888480", "note": "Foundational SLC2A1 haploinsufficiency mouse model — validated KD rescue of neurological phenotype, underpins KD mechanism"},
    {"authors": "Pearson TS, Akman C, Hinton VJ, Engelstad K, De Vivo DC", "year": 2013, "title": "Phenotypic spectrum of glucose transporter type 1 deficiency syndrome (Glut1 DS)", "journal": "Curr Neurol Neurosci Rep", "vol": "13", "pages": "342", "pmid": "23417430", "note": "Comprehensive phenotypic spectrum description — cognitive outcomes correlation with KD initiation age; PED natural history; atypical presentations"},
    {"authors": "Ramm-Pettersen A, Stabell KE, Nakken KO, Selmer KK", "year": 2014, "title": "Does ketogenic diet improve cognitive function in patients with GLUT1-DS? A 6- to 17-month follow-up study including neuropsychological testing", "journal": "Epilepsy Behav", "vol": "39", "pages": "111–115", "pmid": "25150668", "note": "Longitudinal cognitive outcomes on KD — demonstrates cognitive improvement correlates with ketosis adequacy (BHB levels) and duration of untreated disease"},
]

# ── Patient Table (N=41 synthetic) ────────────────────────────────────────────
_SEXES = ["M", "F"]
_ETIOLOGIES = [e["category"] for e in ETIOLOGY_CATALOG]
_ETIOLOGY_WEIGHTS = [e["pct"] for e in ETIOLOGY_CATALOG]
_SEIZURE_TYPES_BRIEF = ["Absence-like", "Focal-myoclonic", "Tonic-clonic", "PED"]
_PHASES = ["early-childhood", "school-age", "adolescence", "young-adult", "adult"]
_TREATMENTS_BRIEF = ["Classic-4:1-KD", "MAD", "MCT-KD", "LGIT", "Classic-KD+LEV", "MAD+CLN"]
_CONTROL = ["seizure-free", "partial-response", "minimal-response", "PED-free-only"]


def _weighted_choice(items, weights):
    total = sum(weights)
    r = random.random() * total
    running = 0
    for item, w in zip(items, weights):
        running += w
        if r < running:
            return item
    return items[-1]


def _make_patients():
    pts = []
    for i in range(1, 42):
        sex = random.choice(_SEXES)
        age = random.randint(4, 45)
        onset_age_m = random.randint(2, 36)  # months
        etiology = _weighted_choice(_ETIOLOGIES, _ETIOLOGY_WEIGHTS)
        seizure_types = random.sample(_SEIZURE_TYPES_BRIEF, k=random.randint(1, 3))
        phase = random.choice(_PHASES)
        treatment = random.choice(_TREATMENTS_BRIEF)
        control = random.choice(_CONTROL)
        csf_glucose = round(random.uniform(22, 44), 1)
        csf_ratio = round(random.uniform(0.25, 0.39), 2)
        bhb_on_kd = round(random.uniform(1.5, 4.5), 1)
        kd_duration_m = random.randint(3, 144)
        ped_present = "Y" if "PED" in seizure_types or random.random() < 0.35 else "N"
        pts.append({
            "id": f"P{i:03d}",
            "age": age,
            "sex": sex,
            "onset_age_months": onset_age_m,
            "etiology": etiology,
            "seizure_types": ", ".join(seizure_types),
            "disease_phase": phase,
            "current_treatment": treatment,
            "seizure_control": control,
            "csf_glucose_mg_dl": csf_glucose,
            "csf_serum_ratio": csf_ratio,
            "bhb_mmol_l": bhb_on_kd,
            "kd_duration_months": kd_duration_m,
            "ped_present": ped_present,
        })
    return pts


PATIENTS = _make_patients()

# ── KPI Summary ────────────────────────────────────────────────────────────────
N = 41
SEIZURE_FREE_PCT = round(
    100 * sum(1 for p in PATIENTS if p["seizure_control"] == "seizure-free") / N, 1
)
AVG_ONSET_M = round(sum(p["onset_age_months"] for p in PATIENTS) / N, 1)
AVG_CSF_RATIO = round(sum(p["csf_serum_ratio"] for p in PATIENTS) / N, 2)
AVG_BHB = round(sum(p["bhb_mmol_l"] for p in PATIENTS) / N, 1)
PED_N = sum(1 for p in PATIENTS if p["ped_present"] == "Y")


def overview():
    etiology_dist = {}
    for e in ETIOLOGY_CATALOG:
        etiology_dist[e["category"]] = {"n": e["n"], "pct": e["pct"]}

    trigger_rates = {t["trigger"][:45]: t["seizure_rate_pct"] for t in TRIGGERS}

    return {
        "dashboard": "GLUT1 Deficiency Syndrome (GLUT1-DS / De Vivo Disease)",
        "syndrome_class": "Metabolic Epilepsy — SLC2A1 haploinsufficiency",
        "gene": "SLC2A1 (chromosome 1p35.3)",
        "mechanism": "Reduced GLUT1 transporter density at BBB → chronic cerebral energy insufficiency → seizures + movement disorder + cognitive impairment",
        "key_biomarker": "CSF:serum glucose ratio < 0.40 (after 4h fast) + CSF glucose < 45 mg/dL",
        "first_line_treatment": "Ketogenic Diet (Level A evidence — DISEASE-MODIFYING, not just anticonvulsant)",
        "absolute_contraindication": "Valproate / VPA — directly inhibits GLUT1 transporter",
        "n_patients": N,
        "kpis": {
            "seizure_free_pct": SEIZURE_FREE_PCT,
            "avg_onset_age_months": AVG_ONSET_M,
            "avg_csf_serum_ratio": AVG_CSF_RATIO,
            "avg_bhb_mmol_l": AVG_BHB,
            "ped_present_n": PED_N,
            "ped_present_pct": round(100 * PED_N / N, 1),
            "kd_on_diet_n": sum(1 for p in PATIENTS if "KD" in p["current_treatment"] or "MAD" in p["current_treatment"] or "LGIT" in p["current_treatment"]),
        },
        "etiology_distribution": etiology_dist,
        "trigger_seizure_rates": trigger_rates,
        "seizure_type_prevalence": {s["type"][:50]: s["prevalence_pct"] for s in SEIZURE_TYPES},
        "clinical_alerts": [
            "⚠️ ABSOLUTE CI: Valproate (VPA) inhibits GLUT1 transporter — worsens GLUT1-DS, may cause metabolic crisis",
            "⚠️ HOSPITAL NBM ORDERS ARE DANGEROUS — maintain KD formula via NG tube; use dextrose-FREE IV fluids",
            "⚠️ URGENT KD INITIATION — each month of delay costs cognitive developmental opportunity",
            "⚠️ CSF LP after 4h fast mandatory — normal serum glucose at time of LP required for valid ratio",
            "⚠️ PED attacks are NON-EPILEPTIC (normal EEG) — do not treat with AEDs; treat with KD",
        ],
        "standards": [s["name"] for s in STANDARDS],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def breakdown():
    return {
        "dashboard": "GLUT1-DS Breakdown — per-patient + etiology + seizures + triggers + treatments + monitoring + lifecycle",
        "patients": PATIENTS,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "monitoring_protocol": MONITORING_PROTOCOL,
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def definitions():
    return {
        "dashboard": "GLUT1-DS Definitions — concepts, contraindications, thresholds, references",
        "concepts": CONCEPTS,
        "absolute_contraindications": [
            {
                "drug": "Valproate (VPA) — ALL formulations",
                "scope": "ALL GLUT1-DS patients — lifelong",
                "mechanism": "Directly inhibits GLUT1 transporter by binding extracellular glucose-binding domain; reduces GLUT1 transport capacity 20–40% in an already 50% haploinsufficient transporter system — precipitates metabolic encephalopathy and paradoxical seizure worsening",
                "consequence": "Acute metabolic crisis, status epilepticus, irreversible neurological deterioration; case reports of death",
                "action": "STOP IMMEDIATELY if inadvertently prescribed; initiate KD; MEDIC ALERT bracelet 'GLUT1-DS — NO VALPROATE' mandatory",
            },
            {
                "drug": "Hospital nil-by-mouth (NBM) / surgical fasting without KD protocol",
                "scope": "ALL GLUT1-DS patients requiring surgical or procedural fasting",
                "mechanism": "Standard NBM (4–8h) depletes already-limited cerebral glucose reserve → breakthrough seizures and metabolic crisis in GLUT1-DS patients not receiving alternate fuel",
                "consequence": "Breakthrough seizures, prolonged seizure clusters, metabolic encephalopathy during peri-operative period",
                "action": "KD formula via NG tube throughout NBM period; dextrose-FREE IV fluids (0.9% NaCl); BHB monitoring q4h; anaesthesia team briefed pre-operatively",
            },
            {
                "drug": "IV glucose bolus / dextrose-containing IV fluids during acute GLUT1-DS crisis",
                "scope": "Acute management of GLUT1-DS metabolic crisis or seizure breakthrough",
                "mechanism": "IV glucose provides substrate that GLUT1-deficient BBB cannot transport adequately to the brain — peripheral hyperglycaemia without proportional CSF glucose rise; may paradoxically deepen metabolic crisis by suppressing compensatory ketogenesis",
                "consequence": "Failure to treat underlying cerebral energy deficit; may worsen acute encephalopathy if ketone supply is simultaneously withdrawn",
                "action": "Provide MCT oil via NG tube as emergency ketone substrate; IV ketone solutions investigational; treat seizures with benzodiazepines + KD formula",
            },
            {
                "drug": "Acetazolamide",
                "scope": "ALL GLUT1-DS patients — relative-to-absolute contraindication",
                "mechanism": "Carbonic anhydrase inhibition reduces cerebrospinal fluid production and modestly inhibits GLUT1-mediated transport — additive inhibitory effect on already-deficient transporter",
                "consequence": "Worsening of CSF glucose ratio and cerebral energy supply; metabolic acidosis exacerbated (potentiates KD-related acidosis risk)",
                "action": "Avoid in GLUT1-DS; if diuresis required perioperatively, use alternative (furosemide); if prescribed in error, discontinue and reassess ketosis",
            },
        ],
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
