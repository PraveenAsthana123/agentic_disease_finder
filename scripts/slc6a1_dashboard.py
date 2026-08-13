"""
SLC6A1 Epilepsy -- Myoclonic-Atonic Epilepsy (MAE / Doose Syndrome / SLC6A1-DEE)
==================================================================================
41-patient cohort . SLC6A1 (3p25.3) . GABA Transporter 1 (GAT-1)
SLC6A1 LOF variants are the most common single-gene cause of Myoclonic-Atonic Epilepsy
(MAE, formerly Doose Syndrome) -- causing GABAergic reuptake failure --> network hyperexcitability
--> drop attacks, absence, myoclonic seizures, intellectual disability.
"""

import random
from datetime import datetime

SEED = 9184  # dashboard 184
random.seed(SEED)

# Etiology Distribution (5 classes, N=41)
ETIOLOGY_CATALOG = [
    {
        "etiology": "De novo SLC6A1 truncating (nonsense / frameshift / large deletion) -- MAE-DEE severe",
        "n": 16, "pct": 39,
        "category": "SLC6A1-de-novo-truncating-LOF-MAE-DEE-severe",
        "mechanism": (
            "Most prevalent class (~39%): de novo nonsense mutations (premature stop codon via UAA/UAG/UGA), "
            "out-of-frame insertions/deletions, or large intragenic deletions spanning SLC6A1 exons -- "
            "all producing NMD-mediated haploinsufficiency or dominant-negative GAT-1 absence. PVS1 "
            "criterion applies (LOF in SLC6A1 causes disease by haploinsufficiency -- GAT-1 transport "
            "null leads to >2-fold reduction in GABA reuptake at GABAergic synapses). Mechanistically: "
            "absent GAT-1 protein --> complete failure of Na+/Cl-coupled GABA reuptake at presynaptic "
            "terminals + astrocytic processes --> extracellular GABA accumulation --> sustained delta-subunit "
            "extrasynaptic GABA_A tonic current --> shunting inhibition + interneuron fatigue --> loss of "
            "effective phasic inhibitory drive --> generalised spike-wave discharges and myoclonic-atonic "
            "bursts. Severe phenotype: drug-resistant MAE-DEE with profound intellectual disability, "
            "autistic features (SLC6A1 also expressed in striatum/cerebellum). ACMG: Pathogenic "
            "(PVS1+PM2+PP2+PP3). De novo confirmed in >90% of truncating class."
        ),
        "eeg_signature": (
            "Ictal: Generalised 2-2.5 Hz irregular spike-wave or polyspike-wave complexes -- onset "
            "in frontoparietal leads, synchronous bilateral, lasting 1-5 seconds per drop attack. "
            "Myoclonic component: fast polyspike (>4 Hz) burst immediately preceding atonic phase. "
            "Absence: 3 Hz regular spike-wave (shorter, <10 sec, abrupt onset/offset). "
            "Pathognomonic interictal feature: DOOSE THETA -- generalised 4-7 Hz rhythmic theta bursts, "
            "maximal over frontoparietal and central regions, appearing as monomorphic sinusoidal trains "
            "lasting 2-6 seconds. This theta-dominant interictal pattern distinguishes MAE from Lennox-"
            "Gastaut syndrome (which shows slow spike-wave <2.5 Hz + diffuse slowing, no theta bursts). "
            "Background: Normal in early onset; progressive background slowing in DEE-severe subgroup."
        ),
        "mri": (
            "Normal MRI in >92% of SLC6A1 MAE-DEE truncating class. No cortical malformation, "
            "no white matter signal change, no focal structural lesion. Normal MRI is expected "
            "and diagnostically reassuring in confirmed SLC6A1 -- do not pursue further structural "
            "workup unless: focal neurological signs, asymmetric semiology, or focal EEG onset."
        ),
        "clinical_note": (
            "HELMET MANDATORY from drop attack onset. Truncating variants confirmed pathogenic by "
            "PVS1 -- no further functional studies needed. Start KD discussion at >=2 AED failures. "
            "Autism screening (M-CHAT, ADOS-2) at diagnosis -- ASD in ~30-40% of DEE-severe. "
            "VPA first-line (POLG excluded). Avoid all sodium channel blockers (CBZ/OXC/PHT)."
        ),
    },
    {
        "etiology": "De novo SLC6A1 missense -- functional LOF confirmed -- MAE moderate",
        "n": 12, "pct": 29,
        "category": "SLC6A1-de-novo-missense-LOF-confirmed-MAE-moderate",
        "mechanism": (
            "Second class (~29%): de novo missense variants in functionally critical residues -- "
            "transmembrane helices (TM1-TM12) or substrate-binding pocket of GAT-1 -- with functional "
            "studies (Xenopus oocyte electrophysiology, HEK293 transport assays) confirming null or "
            ">70% transport reduction. Phenotype: MAE, moderate-severe intellectual disability, absence "
            "and myoclonic-atonic seizures; less severe than truncating class in ~60% -- possibly "
            "reflecting partial residual transport activity in some missense variants. "
            "ACMG: PS3+PM2+PP2+PP3 = Likely Pathogenic/Pathogenic."
        ),
        "eeg_signature": (
            "Similar to truncating class: generalised 2-2.5 Hz spike-wave, polyspike-wave during "
            "myoclonic-atonic episodes. DOOSE THETA present in 80% (slightly lower than truncating). "
            "Absence ictal: 3 Hz spike-wave. Some missense patients show predominantly absence-dominant "
            "EEG pattern (3 Hz regular spike-wave) with infrequent atonic episodes."
        ),
        "mri": "Normal MRI in ~95% of missense-confirmed LOF class. No structural abnormality expected.",
        "clinical_note": (
            "Request functional studies through a reference laboratory if variant is novel -- "
            "critical for PM1/PS3 classification. Treatment ladder: VPA --> add ETH (if absence-dominant) "
            "--> KD at >=2 failures. Neuropsychological tracking every 12M: approximately 60% have "
            "mild-moderate ID, 40% have near-normal cognition."
        ),
    },
    {
        "etiology": "De novo SLC6A1 missense -- absence-dominant, near-normal cognition (partial LOF)",
        "n": 6, "pct": 15,
        "category": "SLC6A1-de-novo-missense-partial-LOF-absence-dominant",
        "mechanism": (
            "Third class (~15%): de novo missense variants outside the core transmembrane transport "
            "machinery -- affecting cytoplasmic C-terminal regulatory sequences or remote allosteric "
            "residues -- with only partial (30-60%) reduction in GABA transport capacity. These "
            "variants retain significant residual GAT-1 function, shifting the phenotype toward "
            "absence-dominant epilepsy rather than the full MAE-DEE syndrome. Intellectual disability "
            "is mild or absent in ~60% of this class -- diagnostically important for genetic counselling. "
            "ACMG: PM1+PM2+PP2+PP3+PS3 (if functional data partial-LOF) = Likely Pathogenic."
        ),
        "eeg_signature": (
            "Predominantly typical 3 Hz generalised spike-wave (CAE-like), with shorter absence "
            "bursts (3-15 sec). Myoclonic-atonic seizures infrequent or absent. DOOSE THETA may "
            "be minimal or absent. Background EEG: normal in majority."
        ),
        "mri": "Normal MRI in 100% of this partial-LOF class. No structural abnormality.",
        "clinical_note": (
            "Closest differential: Childhood Absence Epilepsy (CAE) with genetic susceptibility. "
            "Key distinguishing feature: de novo variant in SLC6A1 with functional evidence of "
            "transport reduction. Treatment: ETH first-line for absence; VPA if myoclonic features emerge."
        ),
    },
    {
        "etiology": "SLC6A1 splice-site variant -- MAE-DEE, intermediate severity",
        "n": 4, "pct": 10,
        "category": "SLC6A1-splice-site-MAE-DEE-intermediate",
        "mechanism": (
            "Fourth class (~10%): canonical (plus/minus 1, 2) or near-canonical splice-site variants "
            "causing exon skipping or intron retention -- typically producing an in-frame or "
            "out-of-frame mRNA product. In-frame exon skips may yield partially functional "
            "truncated protein (intermediate phenotype); out-of-frame products undergo NMD (severe LOF). "
            "RNA studies (cDNA sequencing from lymphocyte or fibroblast RNA) are needed for confirmation. "
            "Phenotype: intermediate between truncating and missense-moderate classes -- MAE with "
            "moderate ID, mixed seizure burden. ACMG: PVS1 if out-of-frame by RNA confirmation."
        ),
        "eeg_signature": (
            "Generalised 2-2.5 Hz spike-wave with DOOSE THETA (present in ~70% of splice class). "
            "Mixed myoclonic-atonic + absence pattern. Background: mild-moderate slowing at "
            "disease peak (3-7 years)."
        ),
        "mri": "Normal MRI. RNA splice confirmation recommended for pathogenicity classification.",
        "clinical_note": (
            "Order RNA studies to confirm splice variant consequence. Treat as full MAE-DEE "
            "until RNA data available -- implement helmet, VPA, and KD pathway if failing AEDs."
        ),
    },
    {
        "etiology": "Clinical SLC6A1-negative MAE phenocopy (SYNGAP1, NEXMIF, DNM1, other)",
        "n": 3, "pct": 7,
        "category": "SLC6A1-negative-MAE-phenocopy",
        "mechanism": (
            "Fifth class (~7%): clinical MAE phenotype (Doose theta, drop attacks, absence) without "
            "pathogenic SLC6A1 variant on comprehensive gene panel or exome. These represent genetic "
            "MAE phenocopies: SYNGAP1 (Ras-ERK pathway -- myoclonic-atonic + eyelid myoclonia), "
            "NEXMIF/KIAA2022 (X-linked -- females with severe MAE-like DEE), "
            "DNM1 (dynamin 1 -- MAE + severe DEE), KCNQ5 (Kv7.5 -- MAE-like + DEE), "
            "GABRA1/GABRB3 (GABA receptor subunits -- absence + MAE). In ~30% of clinical MAE "
            "no genetic cause is identified even after exome/genome sequencing -- suggesting "
            "polygenic architecture, deep intronic variants, or mosaicism requiring multi-tissue testing."
        ),
        "eeg_signature": (
            "DOOSE THETA present in subset -- EEG-based MAE diagnosis remains valid regardless of "
            "genetic result. Absence + myoclonic-atonic pattern identical to SLC6A1-positive cases. "
            "Broad EEG-based phenotyping guides treatment before genetic confirmation."
        ),
        "mri": "Normal MRI in majority. Consider exome/genome reanalysis if MRI shows subtle findings.",
        "clinical_note": (
            "Exome/genome reanalysis after 12-18 months -- yield increases with improved bioinformatics "
            "pipelines. Consider deep intronic SLC6A1 variant testing (RNA or long-read sequencing) "
            "if clinical MAE phenotype is compelling. Treat clinically as MAE -- VPA + KD pathway "
            "regardless of genetic result."
        ),
    },
]

# Seizure Types (4)
SEIZURE_TYPES = [
    {
        "type": "Myoclonic-Atonic Seizure (Drop Attack)",
        "prevalence_pct": 88,
        "eeg_correlate": "Generalised polyspike (>4 Hz) immediately followed by slow-wave (1-1.5 Hz) -- "
                          "polyspike = myoclonic component; slow wave = atonic component. Duration: 0.5-3 sec.",
        "semiology": "Sudden myoclonic jerk of trunk/limbs immediately followed by brief loss of postural "
                     "tone --> patient falls forward or backward (drop attack). Consciousness preserved "
                     "or very briefly impaired. Recovery instantaneous. High injury risk (falls, head trauma). "
                     "Pathognomonic of MAE -- distinguishes from LGS tonic-atonic falls (different EEG correlate).",
        "frequency": "1-50+ drops/day in uncontrolled period; nocturnal clustering common",
        "clinical_tip": "HELMET MANDATORY -- head and facial injury risk from falls is the primary acute harm. "
                         "Recommend protective headgear (soft foam helmet) until >=6 months drop-attack freedom. "
                         "Home video essential for parent documentation of drop-attack count.",
        "treatment_priority": "KD (Level A), VPA (Level B), CLB adjunct, Rufinamide adjunct"
    },
    {
        "type": "Absence Seizure (Typical and Atypical)",
        "prevalence_pct": 75,
        "eeg_correlate": "Typical absence: 3 Hz regular generalised spike-wave, abrupt onset/offset, 3-20 sec. "
                          "Atypical absence: 2-2.5 Hz irregular spike-wave, longer, gradual onset/offset.",
        "semiology": "Brief behavioural arrest, staring, eyelid flutter. Atypical absence (more common in "
                     "SLC6A1-DEE): longer, eye deviation, subtle automatisms, incomplete awareness. "
                     "Distinguishing from CAE: SLC6A1 absence co-exists with drop attacks + DOOSE THETA "
                     "interictal pattern + intellectual disability.",
        "frequency": "Multiple per day (10-200+ in active periods); photically induced in ~42%",
        "clinical_tip": "ETH is absence-specific. VPA covers absence + myoclonic/atonic. "
                         "Monitor for absence status (prolonged absence cluster --> subtle confusion for hours) -- "
                         "treat with IV lorazepam.",
        "treatment_priority": "ETH (Level B for absence-dominant), VPA (Level B), KD"
    },
    {
        "type": "Myoclonic Seizure (without atonic phase)",
        "prevalence_pct": 65,
        "eeg_correlate": "Generalised polyspike burst (3-6 Hz, 0.5-2 sec), maximal frontoparietal. No slow-wave "
                          "follow -- distinguishing from myoclonic-atonic correlate.",
        "semiology": "Sudden brief (0.1-0.5 sec) jerk of bilateral upper limbs, neck, trunk -- without loss "
                     "of posture. May cause dropped objects, stumbling. Often worse in early morning. "
                     "Worsened by sleep deprivation and sodium channel blockers.",
        "frequency": "5-50/day in active phase; worsened by sleep deprivation and sodium channel blockers",
        "clinical_tip": "Myoclonic worsening after starting CBZ/OXC/PHT confirms sodium channel blocker "
                         "aggravation -- STOP the offending AED immediately. VPA + LEV combination effective.",
        "treatment_priority": "VPA (Level B), LEV adjunct (Level C), CLB (Level C)"
    },
    {
        "type": "Generalised Tonic-Clonic Seizure (GTCS)",
        "prevalence_pct": 40,
        "eeg_correlate": "Generalised fast activity (tonic phase, 10-20 Hz), then generalised polyspike-wave "
                          "(clonic phase, 3-5 Hz decelerating). Postictal suppression.",
        "semiology": "Classical tonic-clonic. Often nocturnal. SUDEP risk -- sleep-related GTCS in "
                     "drug-resistant SLC6A1-DEE confers elevated SUDEP risk (counselling mandatory).",
        "frequency": "1-4/month in typical MAE; more frequent in drug-resistant DEE",
        "clinical_tip": "SUDEP counselling for families. Sleep surveillance: nocturnal GTCS monitoring "
                         "(mattress monitor or video). VPA + broad-spectrum coverage essential.",
        "treatment_priority": "VPA (Level B), LEV adjunct (Level C), KD for drug-resistant"
    },
]

# Triggers (8)
TRIGGERS = [
    {"trigger": "Sleep deprivation", "prevalence_pct": 78,
     "mechanism": "Reduced slow-wave sleep --> impaired cortical synchronisation normalisation --> "
                  "increased spike-wave propensity in generalised epilepsies.",
     "management": "Strict sleep schedule. School accommodations for morning seizure burden. "
                   "Avoid all-nighters. Optimise sleep hygiene."},
    {"trigger": "Fever / intercurrent illness", "prevalence_pct": 72,
     "mechanism": "Fever increases metabolic demand + GABAergic interneuron dysfunction --> "
                  "heightened seizure threshold reduction in SLC6A1-GAT1 deficiency.",
     "management": "Antipyretics promptly. Rescue medication plan (diazepam rectal/nasal midazolam) "
                   "for seizure clusters >=3 in 24h during febrile illness. Written fever action plan."},
    {"trigger": "Missed AED dose", "prevalence_pct": 65,
     "mechanism": "Abrupt reduction in GABAergic drug levels --> rebound hyperexcitability.",
     "management": "Blister pack or electronic pill dispenser. MMAS-8 adherence scale every visit. "
                   "Parents: do NOT double the next dose if missed -- give missed dose within 2h only."},
    {"trigger": "Psychosocial stress / emotional arousal", "prevalence_pct": 55,
     "mechanism": "HPA axis activation --> corticotropin-releasing hormone reduces GABA inhibition "
                  "--> spike-wave threshold lowering.",
     "management": "Psychological support + cognitive-behavioural strategies. School inclusion plan. "
                   "Occupational therapist for environmental modifications."},
    {"trigger": "Photic stimulation (photosensitivity)", "prevalence_pct": 42,
     "mechanism": "Generalised epilepsy photosensitivity -- flickering lights trigger occipital-onset "
                  "spike-wave propagation. Higher prevalence in SLC6A1 than other MAE causes.",
     "management": "Photosensitivity testing on EEG (IPS protocol). Polarised lenses for outdoor use. "
                   "Blue-light filter on screens. Limit direct sunlight flickering."},
    {"trigger": "Eye closure sensitivity (ECIPA)", "prevalence_pct": 35,
     "mechanism": "Eye closure --> visual cortex de-afferentation --> alpha/occipital release --> "
                  "spike-wave facilitation in generalised epilepsies. More common in absence-dominant class.",
     "management": "Eye-closure sensitivity test on EEG. Dim room lighting reduces provocation. "
                   "Distinguish from SYNGAP1 eyelid myoclonia (ECIPA-signature)."},
    {"trigger": "Post-prandial / hypoglycaemia", "prevalence_pct": 25,
     "mechanism": "Post-meal insulin surge --> transient hypoglycaemia --> reduced brain glucose --> "
                  "increased neuronal excitability. Particularly relevant during KD transition.",
     "management": "Regular small meals. Monitor blood glucose during KD initiation. "
                   "Target KD ketones beta-OHB 2-4 mmol/L -- not starvation. KD team dietary counselling."},
    {"trigger": "Sodium channel blocker AED (iatrogenic aggravation)", "prevalence_pct": 95,
     "mechanism": "CBZ, OXC, PHT, LCM enhance Na+ channel inactivation --> preferentially "
                  "suppresses tonically-firing inhibitory interneurons --> paradoxical increase in "
                  "myoclonic and atonic seizures. Well-documented class effect in generalised epilepsies.",
     "management": "CONTRAINDICATED. Stop immediately if inadvertently started. Taper over "
                   "2-4 weeks to avoid withdrawal seizures while crossing over to VPA. "
                   "Document AED aggravation clearly in notes and referral letters."},
]

# Treatments (8)
TREATMENTS = [
    {
        "drug": "Valproate (VPA) -- Epilim, Depakote",
        "level": "Level B",
        "role": "First-line broad-spectrum -- MAE and SLC6A1-DEE",
        "dose": "20-60 mg/kg/day in 2-3 divided doses; target serum level 50-100 mcg/mL",
        "moa": "Sodium channel stabilisation + GABA-T inhibition (increases GABA synthesis) + "
               "T-type Ca2+ channel block + HDAC inhibition. Broad-spectrum: covers absence, "
               "myoclonic, atonic, and GTCS seizure types in generalised epilepsies.",
        "efficacy": "50-60% >=50% seizure reduction in MAE; 30-40% seizure freedom in SLC6A1-MAE "
                    "with VPA monotherapy. Best evidence for absence + myoclonic-atonic combination.",
        "safety": "Teratogenicity (MHRA 2024 mandatory PREVENT programme -- valproate REMS for "
                  "females >=4 years; annual review mandatory). Weight gain, tremor, hair thinning, "
                  "thrombocytopaenia, hyperammonaemia (especially with polypharmacy). "
                  "Hepatotoxicity risk: highest <2 years on polytherapy -- monitor LFTs q3M.",
        "monitoring": "VPA serum level (TDM) q6M; LFT, FBC, ammonia, weight q3M. "
                      "POLG mutation exclusion MANDATORY before starting in any child with suspected "
                      "mitochondrial disease -- VPA absolutely contraindicated in POLG/Alpers.",
        "contraindication_note": "POLG exclusion before use; MHRA PREVENT programme compliance mandatory; "
                                  "urea cycle disorders contraindicated (hyperammonaemia).",
    },
    {
        "drug": "Ethosuximide (ETH) -- Zarontin",
        "level": "Level B",
        "role": "Absence-dominant SLC6A1 -- add-on or monotherapy in absence-only subgroup",
        "dose": "15-40 mg/kg/day in 2 divided doses; target 40-100 mcg/mL",
        "moa": "T-type voltage-gated Ca2+ channel blockade (Cav3.1/Cav3.2) in thalamic reticular "
               "neurons --> disrupts thalamocortical 3 Hz spike-wave oscillation --> absence control. "
               "No effect on myoclonic or atonic seizures -- hence adjunct, not monotherapy in MAE.",
        "efficacy": "70-80% absence freedom in CAE; ~55-65% in SLC6A1 absence-dominant subgroup. "
                    "Combine with VPA for myoclonic-atonic + absence combination.",
        "safety": "Generally well-tolerated. GI side effects (nausea, vomiting -- take with food). "
                  "Rare: lupus-like syndrome, Stevens-Johnson syndrome. CBC q6M.",
        "monitoring": "ETH serum level TDM q6M (40-100 mcg/mL); LFTs baseline. CBC q6M.",
        "contraindication_note": "Not effective for myoclonic-atonic seizures -- do not use as sole agent in MAE.",
    },
    {
        "drug": "Ketogenic Diet (KD) -- classical 4:1 or modified Atkins diet (MAD)",
        "level": "Level A",
        "role": "Drug-resistant drop attacks in MAE/SLC6A1-DEE -- highest-evidence treatment",
        "dose": "Classical KD: 4:1 fat:protein+carbohydrate ratio, initiated by metabolic dietitian. "
                "MAD: <20g carb/day (more flexible, equivalent efficacy). "
                "Target: beta-OHB 2-4 mmol/L (blood ketones). Trial period: minimum 3 months.",
        "moa": "Mechanisms include: (1) direct anticonvulsant effect of beta-hydroxybutyrate "
               "(GABA_B agonism + gap junction modulation); (2) altered GABA/glutamate balance; "
               "(3) adenosine A1R activation; (4) mTOR inhibition; (5) epigenetic histone "
               "deacetylase modulation. Particularly effective for myoclonic-atonic (drop attacks).",
        "efficacy": "MAE drop attacks: ~50-60% achieve >=50% reduction; ~30-35% achieve >90% reduction. "
                    "KD is the only Level A evidence therapy for MAE drop attacks (Neal et al. 2008 RCT).",
        "safety": "Growth impairment (monitor height/weight q3M). Dyslipidaemia (lipid profile q6M). "
                  "Renal stones (citrate supplementation; hydration). Constipation. "
                  "Selenium/zinc/vitamin D deficiency. Metabolic acidosis at initiation. DEXA annually.",
        "monitoring": "beta-OHB ketones: point-of-care twice daily at home (target 2-4 mmol/L). "
                      "Metabolic panel q3M: glucose, electrolytes, lipids, LFTs, renal function. "
                      "Selenium, zinc, vitamin D, B vitamins monthly at initiation then q6M. "
                      "DEXA bone density annual. KD specialist dietitian every 3M.",
        "contraindication_note": "Contraindicated in: POLG, carnitine deficiency, fatty acid oxidation disorders, "
                                  "pyruvate carboxylase deficiency. Metabolic workup mandatory before KD.",
    },
    {
        "drug": "Clobazam (CLB) -- Frisium, Onfi",
        "level": "Level C",
        "role": "Adjunct for myoclonic-atonic and absence -- intermediate tolerability",
        "dose": "0.25-1.0 mg/kg/day in 2 divided doses; max 40 mg/day",
        "moa": "Benzodiazepine -- positive allosteric modulator of GABA_A receptors (alpha1, alpha2, alpha5). "
               "Reduces burst firing in thalamocortical circuits. CLB preferentially targets alpha2/alpha5 "
               "(less sedating than traditional BDZ targeting alpha1).",
        "efficacy": "40-55% >=50% reduction in drop attacks as add-on. Tolerance develops in 30-50% "
                    "within 3-12 months -- norclobazam accumulation contributes.",
        "safety": "Sedation, ataxia, hyperactivity in children. Tolerance and dependence with long-term use.",
        "monitoring": "CLB TDM (CLB + norclobazam): norclobazam target 50-300 ng/mL. "
                      "Sedation assessment every visit.",
        "contraindication_note": "Gradual taper required if discontinuing -- do NOT abruptly stop.",
    },
    {
        "drug": "Levetiracetam (LEV) -- Keppra",
        "level": "Level C",
        "role": "Adjunct for GTCS and myoclonic components in MAE",
        "dose": "20-60 mg/kg/day in 2 divided doses; max 3000 mg/day",
        "moa": "SV2A synaptic vesicle protein binding --> reduces Ca2+-dependent neurotransmitter release "
               "at presynaptic terminal. Also modulates GABA_A receptor trafficking.",
        "efficacy": "Moderate efficacy for myoclonic seizures and GTCS in generalised epilepsies. "
                    "30-45% >=50% reduction in myoclonic/GTCS burden. Less effective for atonic drop attacks than KD.",
        "safety": "Behavioural adverse effects: irritability, aggression, mood dysregulation (BIRB) "
                  "in ~15-25%. Pyridoxine (B6) 50-100mg/day may reduce irritability.",
        "monitoring": "Behavioural monitoring: Conners' Parent Rating Scale q3M. Renal function q12M.",
        "contraindication_note": "Not first-line for drop attacks. Behavioural effects may worsen "
                                  "pre-existing ASD behaviour in SLC6A1-DEE -- monitor closely.",
    },
    {
        "drug": "Rufinamide -- Banzel / Inovelon",
        "level": "Level C",
        "role": "Third/fourth-line adjunct for drop attacks in drug-resistant SLC6A1-MAE",
        "dose": "Children: 10-45 mg/kg/day in 2 doses; max 3200 mg/day. "
                "Start low (10 mg/kg/day) and titrate over 2 weeks.",
        "moa": "Sodium channel stabilisation via a novel mechanism (prolonged inactivated state, "
               "distinct from CBZ binding site). Shows relative selectivity for high-frequency firing "
               "(as in atonic-myoclonic bursts) vs normal neuronal activity.",
        "efficacy": "LGS: 42% >=50% drop attack reduction (RCT-based). MAE/SLC6A1: extrapolated from "
                    "LGS evidence; clinical experience suggests similar responder rate for atonic drops. "
                    "NOT effective for absence.",
        "safety": "QTc shortening (rare) -- ECG at baseline. Nausea/vomiting at initiation (transient). "
                  "Dizziness and somnolence. No hepatotoxicity.",
        "monitoring": "ECG at baseline and dose escalation. Seizure diary: track drop count weekly.",
        "contraindication_note": "Avoid in familial short QT syndrome. Carbapenem antibiotics "
                                  "reduce rufinamide levels significantly -- alert in hospitalised patients.",
    },
    {
        "drug": "Fenfluramine (FFA) -- Fintepla (Investigational for MAE/SLC6A1)",
        "level": "Level C",
        "role": "Investigational -- FDA/EMA approved for Dravet Syndrome and CDKL5-DEE; "
                "off-label/trial for SLC6A1-MAE with preliminary evidence",
        "dose": "FFA: 0.1-0.35 mg/kg/day in 2 doses (max 26 mg/day per REMS). "
                "Requires REMS enrolment (cardiac monitoring -- valvulopathy/PAH risk).",
        "moa": "Serotonin (5-HT1D, 5-HT2C) receptor agonism --> activation of sigma1 receptor + "
               "downstream GABA_B sensitization. FFA also activates TREK channels (K+ leak). "
               "The serotonergic enhancement of cortical inhibitory interneuron function partially "
               "compensates for GAT-1 reuptake failure.",
        "efficacy": "DS: 54% responder rate. CDKL5-DEE: 50% (FDA label). "
                    "SLC6A1-MAE: case series + open-label data show 35-55% >=50% drop attack reduction.",
        "safety": "MANDATORY: cardiac echocardiogram every 6 months (valvulopathy + PAH risk from "
                  "5-HT2B cardiac receptor activation). Weight monitoring. BP monitoring. "
                  "Fintepla REMS programme mandatory.",
        "monitoring": "Echo q6M (REMS-mandated). Weight, BP, height monthly during titration.",
        "contraindication_note": "DO NOT use outside REMS programme. Contraindicated with MAOIs. "
                                  "Contraindicated with severe hepatic impairment.",
    },
    {
        "drug": "ACTH / Prednisolone -- acute cluster / status protocol",
        "level": "Level C",
        "role": "Short-course for acute myoclonic-atonic status or refractory cluster event",
        "dose": "ACTH: 40-60 IU/m2/day IM for 2 weeks then taper. "
                "Prednisolone: 2 mg/kg/day for 2-4 weeks then taper.",
        "moa": "Neurosteroid synthesis stimulation (ACTH --> adrenal cortex --> cortisol/neurosteroids). "
               "Neurosteroids are positive allosteric GABA_A modulators -- acute increase in "
               "inhibitory tone breaks refractory spike-wave / myoclonic status.",
        "efficacy": "Variable -- clinical series: 50-70% short-term response. Not for chronic use. "
                    "Best used as bridge while optimising long-term regimen.",
        "safety": "Hypertension, electrolyte imbalance, immunosuppression, adrenal suppression, "
                  "growth impairment, behavioural change. BP daily during course.",
        "monitoring": "BP, electrolytes (Na/K), glucose daily during acute course. Infection surveillance.",
        "contraindication_note": "Short course only. Chronic corticosteroid use contraindicated in children.",
    },
]

# Contraindications (4)
CONTRAINDICATIONS = [
    {
        "item": "Sodium Channel Blockers -- CBZ / OXC / PHT / LCM -- ABSOLUTE CONTRAINDICATION in MAE",
        "severity": "ABSOLUTE",
        "detail": (
            "Carbamazepine (CBZ), oxcarbazepine (OXC), phenytoin (PHT), and lacosamide (LCM) "
            "are sodium channel-blocking AEDs with demonstrated seizure-aggravating effect in "
            "generalised epilepsies with myoclonic and atonic components, including SLC6A1-MAE. "
            "Mechanism: Na+ channel blockers preferentially suppress fast-spiking GABAergic "
            "interneurons --> paradoxical disinhibition --> increase in myoclonic bursts and "
            "drop attacks. Clinical consequence: initiating CBZ/OXC/PHT in a child with MAE --> "
            "acute worsening of drop attacks (often 2-5x increase) within days --> risk of "
            "myoclonic or tonic status epilepticus. This is the single most common iatrogenic "
            "harm in SLC6A1-MAE. TAPER AND DISCONTINUE over 2-4 weeks while cross-titrating to VPA. "
            "Document contraindication in EVERY future referral letter and medication summary."
        ),
    },
    {
        "item": "VPA in POLG / Mitochondrial Disease -- ABSOLUTE CONTRAINDICATION",
        "severity": "ABSOLUTE",
        "detail": (
            "Valproate (VPA) is ABSOLUTELY CONTRAINDICATED in POLG-related disorders (POLG1/POLG2 "
            "mutations -- Alpers-Huttenlocher syndrome, MELAS, MERRF, CPEO). VPA inhibits "
            "mitochondrial beta-oxidation and complexes I/IV of the respiratory chain -- in "
            "POLG patients this precipitates: acute hepatic failure, acute-on-chronic lactic "
            "acidosis, and fatal hepatic encephalopathy. ALL children starting VPA must have POLG "
            "excluded BEFORE first dose if ANY feature of mitochondrial disease is present: "
            "lactic acidosis, hepatic involvement, myopathy, ophthalmological abnormality, "
            "family history of mitochondrial disease, or rapidly progressive encephalopathy. "
            "Rapid POLG sequencing turnaround (1-2 weeks). "
            "Bridge: use LEV or CLB while awaiting POLG result -- do NOT start VPA until POLG negative."
        ),
    },
    {
        "item": "Fenfluramine outside REMS / without echocardiogram monitoring",
        "severity": "HIGH",
        "detail": (
            "Fenfluramine (Fintepla) requires mandatory enrolment in the Fintepla REMS programme "
            "(USA) or equivalent risk management programme (EU) before prescribing. Cardiac "
            "risk of 5-HT2B receptor-mediated cardiac valvulopathy and pulmonary arterial "
            "hypertension mandates echocardiographic monitoring every 6 months throughout treatment. "
            "Prescribing fenfluramine without REMS enrolment or skipping mandatory echo monitoring "
            "is a patient safety violation. Do not use FFA in combination with MAOIs (serotonin syndrome risk)."
        ),
    },
    {
        "item": "Ketogenic Diet without metabolic disorder exclusion",
        "severity": "HIGH",
        "detail": (
            "Ketogenic diet (KD) is contraindicated in: primary carnitine deficiency, "
            "carnitine palmitoyltransferase I/II deficiency, acyl-CoA dehydrogenase deficiencies "
            "(MCAD, VLCAD, LCAD -- impaired fatty acid beta-oxidation), POLG/mitochondrial OXPHOS "
            "defects, and pyruvate carboxylase deficiency. A comprehensive metabolic screen "
            "(plasma amino acids, acylcarnitine profile, urine organic acids, lactate, carnitine) "
            "must be completed BEFORE KD initiation. SLC6A1-MAE patients typically have normal "
            "metabolism -- screening confirms KD safety."
        ),
    },
]

# Monitoring Items (8)
MONITORING = [
    {"item": "VPA TDM + LFT + FBC + ammonia", "schedule": "q3M",
     "detail": "Target VPA serum level: 50-100 mcg/mL (trough, pre-dose). LFT: ALT, AST, ALP, GGT -- "
               "stop VPA if ALT/AST >3x ULN with symptoms. FBC: thrombocytopaenia in ~5% (VPA). "
               "Ammonia: elevated in ~10-15% with polytherapy -- if >2x ULN + symptoms: reduce VPA."},
    {"item": "KD metabolic monitoring (beta-OHB, lipids, metabolic panel)", "schedule": "q3M",
     "detail": "beta-OHB blood ketones: point-of-care twice daily at home. Target: 2-4 mmol/L. "
               "Metabolic panel q3M: glucose, Na, K, Cl, bicarbonate, BUN, creatinine, ALT, AST, TG, cholesterol. "
               "Selenium, zinc, vitamin D, B-complex: q6M. DEXA bone density: annually. "
               "Growth velocity: height + weight q3M."},
    {"item": "EEG (awake + sleep) for Doose theta monitoring and spike-wave burden", "schedule": "q6M",
     "detail": "Quantitative spike-wave index (SWI) on awake EEG. DOOSE THETA (4-7 Hz generalised "
               "theta bursts) quantification: persistence correlates with ongoing encephalopathy burden. "
               "Sleep EEG: spike-wave index during NREM -- if >85%: consider CSWS complication. "
               "Photosensitivity IPS protocol: re-test q12M."},
    {"item": "Neuropsychological assessment -- cognitive + adaptive + ASD", "schedule": "q12M",
     "detail": "Bayley-III or WPPSI-IV (preschool); WISC-V (school age). VABS-III adaptive behavior. "
               "ADOS-2 + SCQ for autism spectrum -- SLC6A1-DEE has ~30-40% ASD co-occurrence. "
               "BRIEF-2 (executive function) in school-age. Results guide: special educational "
               "needs placement, school support plan, occupational therapy referral."},
    {"item": "Drop attack helmet compliance + fall diary", "schedule": "Every clinic visit",
     "detail": "Protective headgear mandatory for ALL children with active drop attacks. "
               "Falls diary: daily count of drop attacks by parent/carer -- reported as number per week at clinic review. "
               "Helmet compliance assessed: days/week worn, adherence barriers. "
               "Discontinue helmet only after >=6 months of confirmed drop attack freedom (diary + EEG)."},
    {"item": "POLG / mitochondrial disease exclusion (before VPA)", "schedule": "Once at diagnosis",
     "detail": "POLG1 and POLG2 gene sequencing + plasma lactate/pyruvate + urine organic acids "
               "before starting VPA. If any feature of mitochondrial disease: ophthalmology (CPEO), "
               "muscle biopsy (respiratory chain enzymes), genetic panel. "
               "Document POLG negative result in medication record."},
    {"item": "Fenfluramine echocardiogram (REMS-mandated cardiac monitoring)", "schedule": "q6M on FFA",
     "detail": "Transthoracic echocardiogram every 6 months during fenfluramine therapy -- "
               "valvulopathy (mitral/aortic/tricuspid regurgitation) and PAH screening. "
               "Discontinue FFA if: significant valvulopathy or mean PAP >25 mmHg. "
               "Weight, BP, heart rate monthly during titration."},
    {"item": "SUDEP risk assessment + safety planning", "schedule": "Annually + after each GTCS",
     "detail": "SUDEP-7 Inventory at each annual review. Nocturnal GTCS: highest SUDEP risk -- "
               "mattress movement sensor or seizure-alert device recommended. Sleep position "
               "(prone sleeping after GTCS increases SUDEP risk). Family education: SUDEP risk, "
               "rescue medication, seizure diary. Written Emergency Care Plan reviewed annually."},
]

# Lifecycle Windows (6)
LIFECYCLE = [
    {"window": "Genetic discovery / suspected diagnosis", "age_range": "0-18 months",
     "focus": "Molecular confirmation (trio exome or gene panel). Metabolic screen (POLG exclusion). "
              "EEG baseline. Parental counselling: de novo vs inherited. Cascade testing if inherited.",
     "action": "Genetic testing referral. Metabolic workup. Neurologist assessment. "
                "Parental psychosocial support."},
    {"window": "Seizure onset + initial AED", "age_range": "7 months - 3 years",
     "focus": "MAE seizure onset (peak 2-4Y for drop attacks). First AED: VPA first-line. "
              "AVOID Na+ channel blockers. HELMET at drop attack onset. Seizure diary initiated.",
     "action": "VPA start after POLG exclusion. Helmet fitting. Seizure diary. "
                "Developmental assessment (Bayley-III). Fever action plan. Rescue medication."},
    {"window": "AED optimisation + KD consideration", "age_range": "2-6 years",
     "focus": "Assess response at 6M per AED. If >=2 AED failures: KD referral (Level A). "
              "Add ETH if absence-dominant. EEG DOOSE THETA tracking. School readiness.",
     "action": "KD metabolic workup + dietitian referral. Add second AED if partial response. "
                "Educational assessment + special needs identification. OT/SLT referral."},
    {"window": "Drug-resistant phase + KD", "age_range": "4-10 years",
     "focus": "KD implementation and optimisation. Rufinamide / CLB add-on for drop attacks. "
              "Fenfluramine consideration (REMS enrolment). Epilepsy surgery evaluation "
              "(callosotomy may reduce drop attacks in drug-resistant MAE -- ~60% drop reduction).",
     "action": "KD 3-month trial assessment. Corpus callosotomy discussion at tertiary centre. "
                "FFA REMS enrolment if eligible. Neuropsychological review + school plan update."},
    {"window": "School-age stabilisation", "age_range": "8-14 years",
     "focus": "Seizure frequency often decreases in mid-childhood (natural history of MAE). "
              "EEG Doose theta diminishes. Cognitive plateau or slow improvement. "
              "Social inclusion and community participation goals.",
     "action": "Consider AED rationalisation if seizure-free >2 years. Vocational planning starts. "
                "ASD/ADHD-targeted support. Transition planning to adult epilepsy services."},
    {"window": "Adolescence and adulthood", "age_range": "14 years+",
     "focus": "Transition to adult neurology. Driving assessment (typically requires >=1 year seizure freedom). "
              "Contraception + VPA REMS compliance for females. Independent living support for moderate-severe ID.",
     "action": "Formal adult transition (NICE NG217 Transition standard). VPA PREVENT programme "
                "annual acknowledgement form for females >=4Y. Driving licence seizure freedom rules. "
                "Supported employment pathway for moderate ID."},
]

# Definitions (14 key concepts)
DEFINITIONS = [
    {
        "term": "SLC6A1 / GAT-1 / GABA Transporter 1",
        "definition": (
            "SLC6A1 (Solute Carrier Family 6 Member 1, 3p25.3) encodes GAT-1 (GABA Transporter 1), "
            "the dominant Na+/Cl-coupled GABA reuptake transporter at central GABAergic synapses. "
            "GAT-1 is expressed on presynaptic interneuron terminals and perisynaptic astrocytic "
            "processes throughout the cortex, basal ganglia, hippocampus, and cerebellum. "
            "It transports 1 GABA molecule per cycle coupled to 2 Na+ and 1 Cl-. GAT-1 terminates "
            "synaptic GABA signalling by clearing the synaptic cleft after GABA release. "
            "LOF --> elevated extracellular GABA --> tonic delta-subunit GABA_A receptor activation "
            "--> receptor desensitisation --> net failure of phasic inhibitory transmission --> MAE."
        ),
    },
    {
        "term": "MAE -- Myoclonic-Atonic Epilepsy (Doose Syndrome)",
        "definition": (
            "Myoclonic-Atonic Epilepsy (MAE), historically known as Doose Syndrome, is a "
            "generalised developmental and epileptic encephalopathy characterised by: "
            "(1) Myoclonic-atonic seizures (drop attacks) -- hallmark. "
            "(2) Absence seizures. (3) Myoclonic seizures. (4) GTCS. "
            "Onset: 7 months to 6 years (peak 2-4 years). EEG: DOOSE THETA (4-7 Hz generalised "
            "theta bursts, pathognomonic interictal pattern) + generalised 2-2.5 Hz spike-wave. "
            "Structural MRI: normal. Intellectual disability: 60-80%. SLC6A1 is the most commonly "
            "identified single-gene cause (~3-17% of clinical MAE)."
        ),
    },
    {
        "term": "SLC6A1-DEE (Developmental and Epileptic Encephalopathy)",
        "definition": (
            "SLC6A1-DEE is the severe end of the SLC6A1 phenotypic spectrum -- distinguished from "
            "MAE by: more profound intellectual disability (severe-profound, IQ <40), earlier "
            "seizure onset (<12 months), drug-resistant course from the outset, autistic features "
            "in >50%, and often biallelic or severe dominant-LOF variants. SLC6A1-DEE "
            "overlaps clinically with Lennox-Gastaut syndrome in severe drug-resistant cases -- "
            "distinguishing features: DOOSE THETA (absent in LGS), EEG background (faster in MAE "
            "than LGS slow <1.5 Hz background), and normal structural MRI."
        ),
    },
    {
        "term": "Myoclonic-Atonic Drop Attack (Pathognomonic seizure of MAE)",
        "definition": (
            "The defining seizure type of MAE: a brief (0.5-3 second) event comprising two "
            "phases: (1) Myoclonic phase -- sudden bilateral symmetric jerk of axial muscles "
            "+ proximal limbs; EEG correlate: generalised polyspike burst (>4 Hz). "
            "(2) Atonic phase -- immediate loss of postural tone following the jerk --> fall. "
            "EEG correlate: slow-wave (1-1.5 Hz) following polyspike. Consciousness is preserved "
            "or very briefly impaired. Recovery is instantaneous. Injury risk is high. "
            "Distinguishing from LGS tonic-atonic: LGS tonic seizures have EEG fast activity "
            "(>10 Hz) during tonic phase, no preceding myoclonic jerk. Video-EEG is gold standard."
        ),
    },
    {
        "term": "DOOSE THETA -- Pathognomonic EEG Interictal Pattern of MAE",
        "definition": (
            "DOOSE THETA refers to the characteristic interictal EEG finding of generalised "
            "4-7 Hz rhythmic theta burst activity, maximal over frontoparietal and central "
            "regions, appearing as trains of monomorphic sinusoidal oscillations lasting 2-6 "
            "seconds, occurring independently of ictal events. Named after Hermann Doose who "
            "first described this EEG signature. Pathognomonic value: DOOSE THETA is ABSENT "
            "in Lennox-Gastaut Syndrome (which shows slow background + diffuse slow spike-wave). "
            "Its presence distinguishes MAE from LGS. Persistence correlates with epileptic "
            "encephalopathy burden -- reduction/resolution often precedes clinical improvement."
        ),
    },
    {
        "term": "Sodium Channel Blocker Aggravation in Generalised Epilepsy / MAE",
        "definition": (
            "A class effect of voltage-gated sodium channel-blocking AEDs (CBZ, OXC, PHT, LCM, "
            "high-dose LTG) whereby these drugs paradoxically WORSEN seizures in patients with "
            "generalised epilepsies featuring myoclonic, atonic, or absence components. "
            "Mechanism: Na+ channel blockers preferentially suppress the high-frequency tonic "
            "firing of GABAergic fast-spiking interneurons --> disinhibition of excitatory networks "
            "--> increased myoclonic bursts, drop attack frequency, and absence duration. "
            "Clinical presentation: within days to 2 weeks of starting CBZ/OXC --> 2-5x "
            "increase in drop attack frequency, or new tonic status epilepticus. "
            "ACTION: STOP offending AED immediately and taper over 2-4 weeks."
        ),
    },
    {
        "term": "Ketogenic Diet Level A Evidence for MAE Drop Attacks",
        "definition": (
            "The ketogenic diet (KD) -- classical 4:1 fat:protein+carbohydrate ratio -- is the "
            "only treatment with Level A evidence (Class I randomised trial data) for reduction "
            "of drop attacks in MAE. Neal et al. (Lancet Neurology 2008) demonstrated ~62% of "
            "children on KD achieved >=50% seizure reduction vs ~37% on AED alone. For MAE, "
            "drop attacks show the highest KD responder rate (~50-60% >=50% reduction). "
            "KD implementation requires: metabolic disorder exclusion, paediatric dietitian-led "
            "initiation, close monitoring of growth, lipids, renal stones, and bone density. "
            "Modified Atkins Diet (MAD) offers equivalent efficacy with greater dietary flexibility."
        ),
    },
    {
        "term": "POLG Exclusion before VPA -- Safety Protocol",
        "definition": (
            "POLG (Polymerase Gamma 1, POLG1/POLG2 -- mitochondrial DNA polymerase) mutations "
            "cause Alpers-Huttenlocher syndrome and other mtDNA depletion syndromes. "
            "Valproate (VPA) is absolutely contraindicated in POLG disorders because: VPA "
            "inhibits mitochondrial beta-oxidation --> fatty acid accumulation --> directly toxic "
            "to already-dysfunctional POLG mitochondria --> acute hepatic failure, lactic "
            "acidosis crisis, and fatal hepatic encephalopathy. "
            "MANDATORY PROTOCOL: ALL children considered for VPA must be screened for POLG "
            "if ANY feature of mitochondrial disease present. POLG sequencing turnaround: "
            "1-2 weeks (NHS Genomic Medicine Service). Bridge therapy: use LEV or CLB while awaiting result."
        ),
    },
    {
        "term": "De Novo SLC6A1 Variant -- Genetics and Recurrence Risk",
        "definition": (
            "~85-90% of pathogenic SLC6A1 variants in MAE arise de novo -- new variants not "
            "present in either biological parent (confirmed by trio sequencing). This high de novo "
            "rate reflects the severe fitness effect of SLC6A1 haploinsufficiency. "
            "Recurrence risk for parents of a de novo proband: <1% (germline mosaicism risk). "
            "For the proband themselves: 50% transmission risk to offspring (autosomal dominant). "
            "SLC6A1 mosaicism: ~5% of SLC6A1-MAE cases show somatic mosaicism (15-40% VAF "
            "in blood) -- phenotypically milder. Deep sequencing (>500x coverage) or "
            "multi-tissue testing (saliva, urine) needed to detect mosaicism."
        ),
    },
    {
        "term": "Tonic Inhibition vs Phasic Inhibition -- GAT-1 Physiology",
        "definition": (
            "GABA_A receptor-mediated inhibition occurs in two distinct modes: "
            "(1) PHASIC INHIBITION: rapid (1-20 ms), high-amplitude IPSC generated by "
            "synaptic GABA released from presynaptic vesicles --> activating synaptic GABA_A "
            "receptors (typically gamma2-subunit containing, rapidly desensitising). "
            "Terminated by GAT-1 reuptake -- the dominant mechanism. "
            "(2) TONIC INHIBITION: sustained, low-amplitude current generated by ambient "
            "extracellular GABA activating extrasynaptic GABA_A receptors (delta-subunit, "
            "high-affinity, non-desensitising). In SLC6A1 LOF: extracellular GABA rises --> "
            "sustained tonic GABA --> delta-receptor desensitisation --> paradoxical net "
            "reduction in effective inhibitory drive --> seizures. This counter-intuitive "
            "mechanism explains why 'more GABA' leads to more seizures in GAT-1 deficiency."
        ),
    },
    {
        "term": "VPA Valproate REMS (PREVENT Programme -- Females >=4 Years)",
        "definition": (
            "Valproate (VPA) is the most potent known human teratogen among AEDs -- major "
            "congenital malformations in 10-11% of first-trimester-exposed pregnancies, "
            "and 30-40% neurodevelopmental impairment in children of VPA-exposed mothers. "
            "MHRA (UK) 2024 PREVENT Programme: mandatory risk acknowledgement for all females "
            ">=4 years prescribed valproate -- annual Review of Pregnancy Prevention Programme "
            "form, patient information card, and contraception counselling. "
            "FDA REMS (USA): equivalent mandatory programme since 2023. "
            "Prescribers cannot renew VPA prescription without documented annual PREVENT/REMS review. "
            "Alternative AED considered in all females of childbearing potential."
        ),
    },
    {
        "term": "SLC6A1 Alliance -- Patient Support Organisation",
        "definition": (
            "The SLC6A1 Alliance (formerly SLC6A1 Connect) is the primary international "
            "patient advocacy and research support organisation for SLC6A1-MAE and SLC6A1-DEE. "
            "Based in the USA; supports international families. Activities include: natural "
            "history registry, biobank, research grant funding, annual research symposium, "
            "and family conferences. Gene therapy and ASO (antisense oligonucleotide) "
            "programmes in development. Equivalent EU resources: EpiCARE (European Reference "
            "Network for rare and complex epilepsies) -- SLC6A1 included. "
            "Patient-family connection to SLC6A1 Alliance recommended at diagnosis."
        ),
    },
    {
        "term": "Protective Helmet for Drop Attacks -- Mandatory Safety Equipment",
        "definition": (
            "A protective helmet (foam safety helmet, cycling helmet, or medical-grade "
            "seizure-protection headgear) is mandatory for ALL children with active drop attacks. "
            "Rationale: drop attacks cause high-velocity falls with head impact --> skull fractures, "
            "subdural haematomas, facial lacerations, and dental injuries are documented "
            "complications. A protective helmet reduces head injury severity by >80% "
            "(observational data from MAE cohorts). Prescription via occupational therapist. "
            "Discontinuation criteria: sustained drop attack freedom for >=6 months confirmed "
            "by parent seizure diary AND clinic EEG reassessment."
        ),
    },
    {
        "term": "Corpus Callosotomy for Drug-Resistant MAE Drop Attacks",
        "definition": (
            "Corpus callosotomy (CC) is a palliative epilepsy surgery that sections the corpus "
            "callosum (complete or anterior 2/3) to interrupt interhemispheric synchronisation "
            "of epileptiform discharges -- reducing generalised atonic/myoclonic-atonic seizures "
            "without curative intent. In drug-resistant MAE/SLC6A1-DEE with predominant drop "
            "attacks: CC reduces drop attack frequency by >=50% in ~60-70% of patients. "
            "CC does not cure MAE or improve cognition -- it is palliative (prevents injury). "
            "Patient selection: failed >=3 AEDs and KD, bilateral spike-wave (not focal onset). "
            "MRI-guided LITT callosotomy is an emerging minimally invasive alternative."
        ),
    },
]

# Clinical Alerts (top 3)
ALERTS = [
    "WARNING: SODIUM CHANNEL BLOCKERS ABSOLUTELY CONTRAINDICATED: CBZ/OXC/PHT/LCM WORSEN drop attacks "
    "and myoclonic seizures in MAE/SLC6A1-DEE. This is the most common iatrogenic harm. "
    "Stop immediately if inadvertently started. Document contraindication in ALL future referral letters.",

    "WARNING: VPA + POLG ABSOLUTE CONTRAINDICATION: EXCLUDE POLG MUTATIONS BEFORE STARTING VPA. "
    "VPA in POLG/Alpers --> fatal hepatic failure. POLG sequencing mandatory if any mitochondrial feature. "
    "Bridge with LEV or CLB while awaiting POLG result.",

    "WARNING: HELMET MANDATORY: ALL children with active drop attacks MUST wear protective headgear. "
    "Drop attack head injuries cause skull fractures and subdural haematomas. "
    "Discontinue only after >=6 months confirmed drop attack freedom by diary + EEG.",
]

# Standards / Guidelines (8)
STANDARDS = [
    "ILAE-2022 -- Myoclonic-Atonic Epilepsy (MAE) classification and diagnosis",
    "NICE-NG217 -- Epilepsy in children and adults (UK guideline, 2022) -- ketogenic diet section",
    "ILAE-Dietary-Therapies-2018 -- Ketogenic diet clinical evidence review (Level A: drop attacks)",
    "MHRA-PREVENT-VPA-2024 -- Mandatory valproate pregnancy prevention programme (females >=4Y)",
    "FDA-Fintepla-REMS-2022 -- Fenfluramine risk evaluation and mitigation strategy",
    "ACMG-AMP-2015 -- Variant classification standards (PVS1 for SLC6A1 LOF)",
    "ACNS-EEG-2021 -- EEG recording and reporting standards (Doose theta characterisation)",
    "Carvill-2015-NatGenet -- SLC6A1 discovery as MAE gene; de novo variants",
]

# Thresholds (10)
THRESHOLDS = [
    "MAE-onset-7M-6Y -- outside window: reconsider diagnosis (LGS if later, EIDEE if earlier)",
    "2-AED-failures-KD-referral -- KD mandatory discussion after 2 failed AEDs",
    "VPA-TDM-50-100-mcg-mL -- below 50: underdosed; above 100: toxicity monitoring",
    "VPA-LFT-3xULN-STOP -- ALT/AST >3x upper limit of normal with symptoms: stop VPA",
    "KD-beta-OHB-target-2-4-mmol-L -- ketosis target; <2: insufficient; >6: excess acidosis risk",
    "Drop-attack-helmet-mandatory -- discontinue only after >=6 months confirmed freedom",
    "POLG-exclusion-mandatory-before-VPA -- do NOT start VPA until POLG negative",
    "FFA-echo-q6M-REMS -- echocardiogram every 6 months on fenfluramine (mandatory)",
    "Na-channel-blocker-CONTRAINDICATION -- CBZ/OXC/PHT/LCM NEVER in MAE/SLC6A1",
    "SUDEP-counselling-first-visit -- nocturnal GTCS: sleep monitoring device recommended",
]

# References (6)
REFERENCES = [
    {
        "citation": "Carvill GL et al. (2015) Nat Genet 47:170-172",
        "title": "SLC6A1 mutation and epilepsy with myoclonic-atonic seizures",
        "relevance": "Discovery paper -- identified SLC6A1 as causative gene for MAE via trio exome sequencing; "
                     "established de novo LOF variants as the primary genetic mechanism",
    },
    {
        "citation": "Johannesen KM et al. (2018) Ann Neurol 84:905-917",
        "title": "Phenotypic spectrum of SLC6A1: from myoclonic-atonic epilepsy to intellectual disability and autism",
        "relevance": "Largest SLC6A1 cohort (n=68); defined full phenotypic spectrum from absence-only to severe DEE; "
                     "missense vs truncating genotype-phenotype correlations",
    },
    {
        "citation": "Neal EG et al. (2008) Lancet Neurol 7:500-506",
        "title": "The ketogenic diet for the treatment of childhood epilepsy: a randomised controlled trial",
        "relevance": "Landmark KD RCT -- Level A evidence for >=50% seizure reduction in MAE; "
                     "drop attacks highest KD-responsive seizure type; established KD Level A for MAE",
    },
    {
        "citation": "Doose H (1992) Epilepsia 33(suppl):105-114",
        "title": "Myoclonic-astatic epilepsy of early childhood",
        "relevance": "Original Doose Syndrome description -- clinical characterisation of MAE, "
                     "Doose theta EEG pattern definition, natural history, and diagnostic criteria",
    },
    {
        "citation": "Lemke JR et al. (2016) Neurology 86:1015-1024",
        "title": "Delineating the GABA transporter SLC6A1 encephalopathy",
        "relevance": "Functional characterisation of SLC6A1 missense variants; GABA transport assay "
                     "data establishing LOF criteria; genotype-phenotype correlation for MAE severity",
    },
    {
        "citation": "Vlasnik A et al. (2022) Epilepsia 63:2512-2525",
        "title": "SLC6A1-related epilepsy: natural history and treatment outcomes from an international cohort",
        "relevance": "International multi-centre SLC6A1 natural history; KD and treatment response; "
                     "SUDEP rate; genotype-phenotype refinement in 120 patients",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    """SLC6A1 Epilepsy (MAE / SLC6A1-DEE) -- overview endpoint."""
    total = sum(e["n"] for e in ETIOLOGY_CATALOG)
    return {
        "syndrome": "SLC6A1 Epilepsy -- Myoclonic-Atonic Epilepsy (MAE / Doose Syndrome / SLC6A1-DEE)",
        "gene": "SLC6A1 -- 3p25.3 -- GABA Transporter 1 (GAT-1)",
        "protein_function": "GAT-1 drives Na+/Cl-coupled GABA reuptake at presynaptic terminals + astrocytes",
        "lof_consequence": "SLC6A1 LOF --> impaired GABA reuptake --> extracellular GABA accumulation --> "
                           "tonic GABA_A desensitization --> net failure of inhibitory drive --> MAE",
        "inheritance": "De novo ~85-90%; autosomal recessive (biallelic) rare; mosaicism ~5%",
        "most_common_gene_mea": "SLC6A1 is the most commonly identified single-gene cause of MAE (~3-17%)",
        "cohort": total,
        "etiology_classes": len(ETIOLOGY_CATALOG),
        "seizure_types": len(SEIZURE_TYPES),
        "triggers": len(TRIGGERS),
        "treatments": len(TREATMENTS),
        "contraindications": len(CONTRAINDICATIONS),
        "monitoring_items": len(MONITORING),
        "lifecycle_windows": len(LIFECYCLE),
        "concepts": len(DEFINITIONS),
        "standards": len(STANDARDS),
        "thresholds": len(THRESHOLDS),
        "references": len(REFERENCES),
        "kd_efficacy_drop_attacks": "Level A -- ~50-60% achieve >=50% drop attack reduction",
        "hallmark_eeg": "DOOSE THETA (4-7 Hz generalised theta bursts) -- pathognomonic for MAE",
        "key_safety": "Na-channel blockers (CBZ/OXC/PHT) ABSOLUTELY CONTRAINDICATED; "
                      "POLG exclusion before VPA; HELMET mandatory for drop attacks",
        "top_alerts": ALERTS[:3],
        "generated_at": datetime.now().isoformat(),
        "dashboard_id": 184,
    }


def get_breakdown():
    """SLC6A1 Epilepsy (MAE / SLC6A1-DEE) -- breakdown endpoint (full clinical detail)."""
    return {
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "alerts": ALERTS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "generated_at": datetime.now().isoformat(),
    }


def get_definitions():
    """SLC6A1 Epilepsy (MAE / SLC6A1-DEE) -- definitions endpoint (14 key concepts)."""
    return {
        "syndrome": "SLC6A1 Epilepsy -- Myoclonic-Atonic Epilepsy (MAE / Doose Syndrome / SLC6A1-DEE)",
        "definitions": DEFINITIONS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "generated_at": datetime.now().isoformat(),
    }
