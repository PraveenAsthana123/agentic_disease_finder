"""
PRRT2 Epilepsy Spectrum — Benign Familial Infantile Epilepsy / Paroxysmal Kinesigenic Dyskinesia
=================================================================================================
41-patient cohort · PRRT2 (16p11.2) · Proline-Rich Transmembrane Protein 2
PRRT2-related disorders: the most common cause of Benign Familial Infantile Epilepsy (BFIE),
Paroxysmal Kinesigenic Dyskinesia (PKD), and the Infantile Convulsions and Choreoathetosis
(ICCA) combined syndrome. PRRT2 LOF variants → disrupted SNAP25 interaction → sodium channel
dysregulation → episodic neuronal hyperexcitability.

PRRT2 BIOLOGY: PRRT2 (Proline-Rich Transmembrane Protein 2, 16p11.2) encodes a single-pass
transmembrane protein with a large proline-rich N-terminal cytoplasmic domain. PRRT2 is
expressed in glutamatergic neurons, localises to presynaptic terminals, and directly interacts
with SNAP25 (synaptosomal-associated protein 25kDa) — a SNARE complex protein critical for
synaptic vesicle fusion. The PRRT2-SNAP25 interaction negatively modulates voltage-gated
sodium channel (Nav1.2/Nav1.6) surface expression and kinetics at the presynaptic membrane.
LOF in PRRT2 → reduced sodium channel inhibition → transiently increased Nav surface density →
neuronal hyperexcitability → episodic clinical attacks (seizures in infancy; movement-triggered
dyskinesia in later life).

HOTSPOT VARIANT: c.649dupC (p.Arg217Profs*8) accounts for ~80% of all pathogenic PRRT2
variants — a single-nucleotide frameshift (cytosine duplication at position 649 in exon 2)
within a poly-C stretch (homopolymer run), causing premature stop codon at codon 224 and
NMD-mediated haploinsufficiency. This variant is pathogenic by ACMG PVS1+PM2+PP1+PP3.

PHENOTYPIC SPECTRUM — PRRT2-Related Disorders:
① BFIE (Benign Familial Infantile Epilepsy / BFIS): onset 3-12 months, brief focal or
  generalised tonic-clonic or clonic seizures, often in clusters. Self-limited by 18-24 months.
  Normal development, normal EEG. Autosomal dominant; PRRT2 accounts for ~90% of familial BFIE.
② PKD (Paroxysmal Kinesigenic Dyskinesia): sudden movement-triggered brief (<60s) attacks of
  dystonia, chorea, or athetosis — typically onset in adolescence (6-16 years). Not true
  epileptic seizures (no EEG correlate). PRRT2 is the most common genetic cause of PKD (~80%).
  Excellent CBZ response (near-100% control at low doses 50-200mg/day).
③ ICCA (Infantile Convulsions and Choreoathetosis): BFIE + PKD in same individual or same family.
  PRRT2 c.649dupC accounts for majority of ICCA pedigrees.
④ Hemiplegic Migraine: rare PRRT2 phenotype — attacks of migraine with aura + transient
  hemiplegia. Overlaps with FHM (Familial Hemiplegic Migraine) spectrum.

NATURAL HISTORY:
- BFIE: Excellent prognosis. Most infants seizure-free by 18-24 months without or with brief AED.
  Normal cognitive development in ~95%. No AED needed in most after age 2.
- PKD: Excellent CBZ/OXC response — near-complete control with low-dose CBZ (100-200mg/day).
  Most adults can taper CBZ in mid-adulthood when PKD attacks diminish spontaneously.
- ICCA: Same excellent prognosis for both components when correctly identified.

INHERITANCE: Autosomal dominant, ~60-80% penetrance. c.649dupC shows de novo occurrence in
~15% of cases alongside familial transmission. 16p11.2 microdeletion (spanning PRRT2) causes
more complex phenotype (intellectual disability + autism spectrum + BFIE/PKD).

KEY SAFETY PEARLS:
• CBZ HLA-B*15:02: Mandatory testing in SE Asian ancestry before CBZ/OXC (SJS/TEN risk).
• MISDIAGNOSIS RISK: PKD attacks are frequently misdiagnosed as focal motor seizures → unnecessary
  chronic AED polytherapy. Key distinguishing feature: normal ictal EEG during PKD episode +
  movement-triggering pattern + <60 second duration + consciousness preserved.
• BFIE OVERTREATMENT: Most BFIE self-limits by 24 months — long-term AED maintenance after
  spontaneous remission is unnecessary.
• GENETIC COUNSELLING: c.649dupC hotspot → cascade testing in all first-degree relatives.
"""

import random
from datetime import datetime

SEED = 9183  # dashboard 183
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "PRRT2 truncating frameshift c.649dupC (p.Arg217Profs*8) — BFIE+PKD/ICCA (de novo or familial)",
        "n": 18, "pct": 44,
        "category": "PRRT2-truncating-c649dupC-BFIE-PKD-ICCA",
        "mechanism": (
            "Most prevalent class (~44%): cytosine duplication at the homopolymer poly-C stretch "
            "(exon 2, position 649) creates p.Arg217Profs*8 — a frameshift leading to a premature "
            "stop codon at residue 224 with NMD-mediated haploinsufficiency. This hotspot accounts "
            "for ~80% of all PRRT2 pathogenic variants globally. PRRT2 haploinsufficiency disrupts "
            "the presynaptic PRRT2-SNAP25 protein-protein interaction: SNAP25 normally suppresses "
            "Nav1.2/Nav1.6 surface trafficking at the presynaptic terminal; loss of PRRT2 → reduced "
            "SNAP25-Nav interaction → excess sodium channel membrane insertion → transiently "
            "elevated sodium conductance → action potential threshold lowering → episodic "
            "neuronal burst firing. In infancy (CNS developmental window) this manifests as "
            "clustered focal/generalised seizures (BFIE); in adolescence, as the Nav maturation "
            "shifts and kinesiogenic triggers (sudden movement) produce brief cortical-subcortical "
            "dysrhythmia without full EEG ictal correlate — manifesting as PKD. The c.649dupC "
            "variant is pathogenic by ACMG PVS1 (null variant in PRRT2 causes disease by "
            "haploinsufficiency) + PM2 (absent from gnomAD) + PP1 (familial co-segregation). "
            "Penetrance ~60-80% (incomplete); de novo in ~15% of probands. ClinVar: Pathogenic."
        ),
        "eeg_signature": (
            "BFIE phase: Focal onset ictal discharge — typically unilateral centroparietal or "
            "occipital rhythmic theta evolving to spike-wave. Clusters of 1-10 brief seizures "
            "(mean 3-5 per cluster per day). Interictal EEG normal in 85-90% of BFIE patients "
            "between clusters — absence of persistent interictal epileptiform discharges is a "
            "key diagnostic feature. Background: normal for age. "
            "PKD phase (adolescence/adulthood): EEG during PKD episode is NORMAL — no ictal "
            "correlate. This is diagnostic: movement-triggered paroxysmal episodes with completely "
            "normal ictal scalp EEG confirms PKD vs focal motor seizure. Video-EEG is the gold "
            "standard for this distinction. Background: always normal."
        ),
        "mri": (
            "Normal MRI in >98% of PRRT2 BFIE/PKD patients — no structural abnormality. "
            "Routine MRI not mandatory if clinical BFIE with positive PRRT2 molecular diagnosis "
            "in a classic presentation (normal development, cluster onset 3-12M, self-limited). "
            "MRI warranted if: atypical features (persistent beyond 24M, developmental delay, "
            "abnormal EEG background, asymmetric motor exam)."
        ),
        "clinical_note": (
            "HOTSPOT VARIANT: c.649dupC accounts for ~80% of PRRT2 pathogenic variants — "
            "single-gene panel testing or targeted sequencing of exon 2 is cost-effective first-line "
            "approach. CASCADE TESTING: all first-degree relatives. BFIE: reassure families "
            "— excellent prognosis, self-limited. PKD: CBZ 100-200mg/day produces near-complete "
            "control; start low and titrate to response."
        ),
    },
    {
        "etiology": "PRRT2 truncating nonsense / canonical splice-site — isolated BFIE or isolated PKD (de novo or familial)",
        "n": 11, "pct": 27,
        "category": "PRRT2-truncating-nonsense-splice-BFIE-PKD-isolated",
        "mechanism": (
            "Second most common class (~27%): nonsense mutations (stop codon), canonical splice-site "
            "variants (affecting ±1/±2 splice donor/acceptor), or small deletions/duplications "
            "not involving the poly-C hotspot — all producing PRRT2 haploinsufficiency via the "
            "same NMD pathway as c.649dupC. Phenotypically: isolated BFIE (without later PKD) "
            "or isolated PKD (without prior infantile seizures) — these partial phenotypes reflect "
            "incomplete penetrance and phenotypic variability within the PRRT2 spectrum. "
            "Mechanism identical to class 1: PRRT2 LOF → SNAP25-Nav dysregulation → episodic "
            "hyperexcitability. Pathogenic by PVS1 criteria. May present as sporadic BFIE with "
            "subsequent family testing revealing affected relatives with PKD (ICCA family in retrospect). "
            "Genotype-phenotype correlation: no reliable predictor of isolated vs combined "
            "BFIE+PKD — depends on incomplete penetrance + additional genetic/environmental modifiers."
        ),
        "eeg_signature": (
            "Isolated BFIE: Same as class 1 — focal centroparietal/occipital ictal onset during "
            "cluster, normal interictal. Isolated PKD: EEG entirely normal (no BFIE history, "
            "no interictal IED, normal ictal EEG during kinesiogenic attack). Awareness is "
            "preserved during PKD episodes (patient can converse during attack) — unlike "
            "typical focal seizures."
        ),
        "mri": "Normal MRI in >97%; no structural cause identified.",
        "clinical_note": (
            "Key diagnostic challenge: isolated PKD without BFIE history — differentiate from "
            "focal motor epilepsy. Triad: movement-triggered, <60s, normal ictal EEG. "
            "Genetic testing: clinical genetics referral + PRRT2 sequencing. "
            "Family history: often positive for BFIE in parent/sibling even if PKD-only proband."
        ),
    },
    {
        "etiology": "PRRT2 missense LOF variant — partial phenotype / attenuated BFIE",
        "n": 5, "pct": 12,
        "category": "PRRT2-missense-LOF-partial",
        "mechanism": (
            "Third class (~12%): missense variants at functionally critical residues — primarily "
            "within the proline-rich cytoplasmic domain and transmembrane region where PRRT2-SNAP25 "
            "interaction occurs. These variants reduce (rather than abolish) PRRT2 function — "
            "partial LOF. Clinically: often attenuated BFIE (fewer seizures, shorter clusters, "
            "earlier remission) or sub-clinical presentation. Some missense variants classified "
            "as VUS initially; functional assay (sodium current patch-clamp, co-IP with SNAP25) "
            "required for definitive pathogenicity assignment. The missense class may have higher "
            "non-penetrance than truncating variants. Key missense hotspots: p.Arg217Cys (different "
            "from c.649dupC at same codon), p.Asp347Tyr, variants in TM domain. ACMG classification "
            "typically LP or VUS pending segregation + functional data."
        ),
        "eeg_signature": (
            "If symptomatic BFIE: same focal centroparietal ictal morphology as truncating class "
            "but fewer clusters, shorter duration, earlier resolution (often by 12-15M). "
            "PKD: if present, typically milder (less frequent kinesiogenic attacks, often spontaneously "
            "remitting by late adolescence without CBZ). Normal interictal EEG."
        ),
        "mri": "Normal MRI — no structural abnormality.",
        "clinical_note": (
            "VUS missense in PRRT2: request functional assay or await further clinical evidence. "
            "Segregation data (family testing) most powerful reclassification tool. "
            "Treat clinically: if child has BFIE and carries PRRT2 missense in same family "
            "as BFIE/PKD, treat clinical picture — don't withhold diagnosis pending molecular certainty."
        ),
    },
    {
        "etiology": "16p11.2 microdeletion (including PRRT2) — complex neurodevelopmental phenotype + BFIE/PKD",
        "n": 4, "pct": 10,
        "category": "16p11.2-microdeletion-PRRT2-NDD",
        "mechanism": (
            "Fourth class (~10%): chromosomal microdeletion at 16p11.2 (breakpoint region 4-5, "
            "~593kb recurrent deletion mediated by segmental duplication BPKR4-BPKR5) — the most "
            "common autism-associated CNV affecting ~1/3000 births. Deletion includes PRRT2 plus "
            "25-28 additional genes. Unlike isolated PRRT2 LOF (excellent prognosis), 16p11.2 "
            "deletion causes a more complex phenotype: mild-moderate intellectual disability "
            "(IQ 70-85), autism spectrum disorder (15-25%), speech delay, macrocephaly, and "
            "psychiatric comorbidities. The BFIE/PKD component is attributed specifically to "
            "PRRT2 haploinsufficiency within the deletion interval. Additional deletion genes "
            "(TBX6 vertebral abnormality, ALDOA, MAPK3) contribute to the broader phenotype. "
            "Detected by chromosomal microarray (CMA), not by standard sequencing. "
            "Parent-of-origin effect: de novo deletions most severe; inherited deletion "
            "associated with better cognitive outcome."
        ),
        "eeg_signature": (
            "BFIE component: same focal onset, cluster pattern as isolated PRRT2 BFIE. "
            "May be more prolonged or refractory than isolated PRRT2 BFIE due to broader "
            "neurodevelopmental disruption. Interictal EEG may show mild generalised background "
            "slowing reflecting global NDD. PKD component (if present): same normal ictal EEG."
        ),
        "mri": (
            "Normal structural MRI (no cortical malformation). "
            "May show mild white matter volume changes on quantitative MRI in some 16p11.2 deletion carriers."
        ),
        "clinical_note": (
            "Detection requires chromosomal microarray (CMA) — not identified by gene panel "
            "sequencing alone. All patients with BFIE + developmental delay/autism should have CMA. "
            "Multidisciplinary management: neurology (BFIE/PKD) + developmental paediatrics (NDD) "
            "+ speech therapy + psychology. Family: recurrence risk 50% if inherited."
        ),
    },
    {
        "etiology": "Clinical PRRT2-negative BFIE/PKD phenocopy — sporadic or unknown aetiology",
        "n": 3, "pct": 7,
        "category": "Clinical-PRRT2-negative-phenocopy",
        "mechanism": (
            "Fifth class (~7%): clinically classic BFIE or PKD with negative PRRT2 sequencing "
            "and negative chromosomal microarray — genetic aetiology unknown. Possible explanations: "
            "(a) deep intronic variant or promoter variant not captured by standard sequencing; "
            "(b) structural variant (inversion) disrupting PRRT2 without CNV; (c) KCNQ2/KCNQ3 "
            "variant (alternative BFIE cause, benign neonatal/infantile epilepsy); (d) SCN8A "
            "mild variant causing sporadic BFIE; (e) truly sporadic BFIE without identified gene. "
            "For PKD phenocopy: other rare causes include SLC2A1 (GLUT1), ATP1A3 (alternating "
            "hemiplegia), KCNA1 (episodic ataxia 1 with kinesigenic component). "
            "Management: treat clinically; consider repeat genetic testing with upgraded panel/WES "
            "in 2-3 years as variant interpretation evolves."
        ),
        "eeg_signature": (
            "Clinically indistinguishable from PRRT2 BFIE/PKD by EEG — normal interictal, "
            "focal ictal for BFIE, normal ictal for PKD. Genetic negative status does not change "
            "EEG interpretation or clinical management."
        ),
        "mri": "Normal MRI — same as PRRT2 positive cases.",
        "clinical_note": (
            "Inform family: currently unresolved; not a missed diagnosis but a knowledge gap. "
            "Prognosis same as PRRT2-confirmed BFIE/PKD (excellent). "
            "Re-test with WES in 2-3 years. Treat PKD with CBZ (response confirms PKD phenotype)."
        ),
    },
]

# ── Seizure Types (4) ─────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal onset tonic / clonic / tonic-clonic seizures (BFIE — Benign Familial Infantile Epilepsy)",
        "frequency_pct": 92,
        "age_window": "3-12 months (peak 5-7 months)",
        "eeg_correlate": (
            "Focal ictal onset: centroparietal or occipital rhythmic theta/alpha evolving to "
            "spike-wave discharge; 10-40 sec duration; often bilateral spread but focal onset clearly "
            "identifiable. Occur in clusters (3-10 per day, lasting 1-7 days). Interictal EEG: "
            "NORMAL in 85-90% — absence of persistent interictal IED is a defining BFIE feature."
        ),
        "clinical_tip": (
            "Classic cluster pattern: 5-6 brief focal seizures over 12-24 hours → seizure-free "
            "for days/weeks → another cluster. Consciousness: usually retained (aware but "
            "unresponsive during tonic phase); post-ictal period brief (<5 minutes). "
            "Key distinguishing features from non-benign infantile epilepsy: "
            "(1) normal development before and after seizures; "
            "(2) no developmental plateau during cluster; "
            "(3) seizures self-limit by 18-24 months; "
            "(4) positive family history (BFIE in first-degree relative or PKD in parent); "
            "(5) normal interictal EEG between clusters; "
            "(6) rapid PRRT2 genetic confirmation. "
            "Emergency: clusters can last >30min cumulatively — if >3 seizures in 24h or any "
            "seizure >5min → diazepam rescue and neurology review."
        ),
    },
    {
        "type": "Paroxysmal Kinesigenic Dyskinesia (PKD) — movement-triggered episodic involuntary movements",
        "frequency_pct": 73,
        "age_window": "Onset 6-16 years (adolescence); may persist to adulthood",
        "eeg_correlate": (
            "EEG during PKD episode: NORMAL — no ictal discharge. This is the diagnostic cornerstone. "
            "Video-EEG: movement artifact at onset (kinesiogenic trigger), then brief 10-60 sec "
            "involuntary movement (dystonia, chorea, athetosis or mixed), NORMAL EEG throughout, "
            "return to baseline. Normal background EEG before and after episode. "
            "WARNING: do not misinterpret muscle artifact as epileptiform activity."
        ),
        "clinical_tip": (
            "PKD diagnostic criteria (Bruno 2004, updated IRLSSG 2011): "
            "(1) kinesiogenic trigger (sudden voluntary movement onset); "
            "(2) brief attacks <60 seconds (typically 10-30s); "
            "(3) no loss of consciousness; "
            "(4) ictal EEG normal; "
            "(5) attacks responsive to low-dose CBZ (near 100%). "
            "Semiology: typically one limb → contralateral arm tonic posturing or writhing "
            "(dystonia) or rapid irregular movements (chorea); face involvement (dysarthria, "
            "grimacing); consciousness always preserved — patient can speak during episode. "
            "Attack frequency: 1-100/day in untreated severe cases; CBZ reduces to 0-1/day. "
            "Precipitants: arising from chair, starting to walk, startling, change in movement speed. "
            "MISDIAGNOSIS ALERT: frequently confused with focal motor seizures — key difference "
            "is the normal ictal EEG and kinesiogenic trigger pattern."
        ),
    },
    {
        "type": "Febrile Seizures / BFIE with fever (fever-triggered infantile clusters)",
        "frequency_pct": 35,
        "age_window": "3-18 months (overlap with BFIE window)",
        "eeg_correlate": (
            "Same focal onset ictal morphology as afebrile BFIE clusters. Fever lowers seizure "
            "threshold — typical febrile seizure EEG (post-ictal slowing). Crucially: unlike "
            "Dravet syndrome, PRRT2 BFIE does NOT evolve to febrile seizure plus (FS+) pattern "
            "or to Dravet syndrome — normal development maintained through febrile illness."
        ),
        "clinical_tip": (
            "PRRT2 BFIE with febrile seizures can initially mimic Dravet syndrome "
            "(SCN1A-haploinsufficiency) — critical to distinguish early: "
            "(1) PRRT2 BFIE: seizures self-limit by 18-24 months, no developmental regression; "
            "(2) Dravet: progressive developmental regression, prolonged febrile seizures > 15 min, "
            "onset typically 5-6 months at first prolonged febrile seizure, alternating hemiclonic. "
            "Genetic testing resolves ambiguity: PRRT2 vs SCN1A. "
            "Action: if febrile seizure >5min → SCN1A/Dravet protocol first "
            "(sodium channel blockers contraindicated in Dravet) → confirm PRRT2 before CBZ if Dravet excluded."
        ),
    },
    {
        "type": "FBTCS (Focal to Bilateral Tonic-Clonic Seizure) — secondary generalisation from BFIE",
        "frequency_pct": 18,
        "age_window": "During BFIE phase (3-18 months) — cluster evolution",
        "eeg_correlate": (
            "Focal onset evolving to bilateral tonic-clonic — bilateral synchrony on EEG after "
            "focal discharge spread. Occurs in ~18% of BFIE clusters (typically longer or more "
            "severe clusters). Full post-ictal slowing. Resolves with BFIE remission."
        ),
        "clinical_tip": (
            "FBTCS in BFIE context: not an indicator of worse prognosis — still self-limited by "
            "24 months. Emergency plan: rectal/buccal diazepam for seizures >5 minutes. "
            "Hospitalisation for prolonged cluster (>3 FBTCS in 24h). "
            "Once PRRT2 confirmed: reassure family about prognosis."
        ),
    },
]

# ── Triggers (8) ──────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Sudden voluntary movement / change in movement speed (PKD-specific)", "pct": 95,
     "detail": "The defining kinesiogenic trigger — sudden voluntary movement onset (e.g., standing "
               "from chair, starting to walk/run, reaching rapidly, turning quickly) is the universal "
               "precipitant of PKD. Gradual or sustained movement does NOT trigger — only the "
               "sudden velocity change. This kinesiogenic selectivity is pathognomonic."},
    {"trigger": "Fever / intercurrent illness (BFIE-specific)", "pct": 68,
     "detail": "Fever lowers seizure threshold in infants via sodium channel dynamics and "
               "neuronal temperature sensitivity. Most BFIE clusters are triggered by fever "
               "and intercurrent viral illness — explains clustering with respiratory and "
               "gastrointestinal infections in infancy. Fever management (paracetamol) does "
               "NOT reliably prevent BFIE clusters."},
    {"trigger": "Startle / sudden unexpected stimulus (PKD)", "pct": 62,
     "detail": "Unexpected loud sounds, sudden touch, or sudden visual stimuli can trigger "
               "PKD attacks — the 'startle' variant. Overlaps with kinesigenic trigger in "
               "that both involve sudden sensory-motor transitions. EEG: normal during "
               "startle-triggered PKD (distinguishes from startle epilepsy where ictal EEG is abnormal)."},
    {"trigger": "Sleep deprivation / fatigue (PKD + BFIE)", "pct": 48,
     "detail": "Sleep deprivation lowers the threshold for both PKD attacks and BFIE clusters. "
               "Particularly relevant in adolescents with PKD — irregular sleep-wake cycles "
               "(school exam periods, social activities) can increase PKD frequency."},
    {"trigger": "Caffeine / stimulants (PKD)", "pct": 38,
     "detail": "High caffeine intake (energy drinks, coffee) may increase PKD attack frequency "
               "in some adolescents via adenosine receptor antagonism and sympathomimetic "
               "effects increasing movement initiation reflexes. Practical advice: limit "
               "energy drinks; no prescription stimulant restrictions in ADHD-comorbid patients."},
    {"trigger": "Missed AED dose (CBZ/OXC — PKD phase)", "pct": 35,
     "detail": "Discontinuation or missed doses of CBZ/OXC in established PKD patients triggers "
               "cluster rebound of kinesiogenic attacks. The effect is typically apparent within "
               "12-24 hours of a missed dose. Reinforces strict adherence during CBZ therapy."},
    {"trigger": "Hyperventilation (PKD)", "pct": 28,
     "detail": "Sustained hyperventilation (exercise, anxiety, singing) may lower the kinesiogenic "
               "threshold via hypocapnic alkalosis effects on Nav channel inactivation kinetics. "
               "Less consistent than movement trigger but reported in ~28% of PKD patients."},
    {"trigger": "Emotional stress / anxiety (PKD + BFIE)", "pct": 22,
     "detail": "Psychological stress increases sympathetic tone and movement initiation frequency "
               "— indirectly increasing PKD attack rate. Direct stress-seizure pathway less "
               "prominent than in other epilepsies. Anxiety management (therapy, mindfulness) "
               "may reduce perceived attack burden."},
]

# ── Treatments (8) ──────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Carbamazepine (CBZ)",
        "brand": "Tegretol / generic",
        "evidence": "Level A — PKD first-line (near-universal response)",
        "dose_pediatric": "2-5 mg/kg/day (PKD usually responsive at low doses 50-150mg/day)",
        "dose_adult": "100-400mg/day (PKD control typically 100-200mg/day — much lower than epilepsy dosing)",
        "titration": "Start 50-100mg/day; increase by 50mg/week to response. Most PKD patients "
                     "achieve complete control at very low doses — avoid over-treating.",
        "moa": (
            "Voltage-gated Na+ channel blocker (Nav1.1/1.2/1.6): binds inactivated state, "
            "prolongs inactivation → stabilises neuronal membrane → reduces the excessive Nav "
            "surface expression caused by PRRT2 LOF. CBZ at low doses effective for PKD "
            "because kinesiogenic attacks involve transient nav-driven episodic hyperexcitability "
            "rather than sustained epileptic discharge — lower channel-block threshold needed."
        ),
        "efficacy": "Near-complete PKD control in >90% at low doses. Response typically within "
                    "1-2 weeks. Most patients achieve attack-free status. Can usually taper "
                    "after age 25-30 years as PKD naturally remits.",
        "safety": "Rash (5-10%); rare SJS/TEN (HLA-B*15:02 risk); diplopia/ataxia at higher doses "
                  "(not typically seen at low PKD doses); hyponatraemia (SIADH); teratogen "
                  "(folic acid supplementation + alternative if pregnant). "
                  "DRUG INTERACTIONS: CYP3A4 inducer → reduces levels of many drugs.",
        "monitoring": "HLA-B*15:02 before prescribing in SE Asian ancestry. "
                       "TDM not routinely needed at low PKD doses — clinical response guides. "
                       "FBC at baseline; LFTs annually. ECG if pre-existing cardiac conduction disease.",
        "notes": "BFIE: NOT routinely indicated — BFIE is self-limited. Reserve CBZ for "
                 "prolonged BFIE clusters >18-24 months or clinician/family preference. "
                 "PKD: first-line, near-mandatory — dramatic quality-of-life benefit.",
    },
    {
        "drug": "Oxcarbazepine (OXC)",
        "brand": "Trileptal / generic",
        "evidence": "Level B — PKD alternative to CBZ",
        "dose_pediatric": "5-10 mg/kg/day",
        "dose_adult": "150-600mg/day (PKD control usually at low end)",
        "titration": "Start 150mg/day; increase by 150mg/week. Monitor sodium levels at start.",
        "moa": (
            "Pro-drug: rapidly metabolised to MHD (monohydroxy-derivative) — active moiety. "
            "Same Nav-channel inactivation mechanism as CBZ. Better tolerability profile "
            "than CBZ (lower drug-interaction burden, less auto-induction, lower rash incidence "
            "in non-Asian populations). Preferred when CBZ rash risk or drug interaction concern."
        ),
        "efficacy": "Near-equivalent to CBZ for PKD. Complete control in ~85-90%. "
                    "Better tolerated in some patients (fewer CNS side effects at equivalent doses).",
        "safety": "Hyponatraemia (>10% at higher doses — monitor Na+); cross-rash with CBZ (~50% "
                  "if CBZ rash occurred); rash overall lower than CBZ. HLA-B*15:02: CPIC states "
                  "avoid OXC also in carriers (SJS/TEN risk, lower than CBZ but not zero).",
        "monitoring": "Serum sodium at baseline and after dose changes (SIADH). "
                       "HLA-B*15:02 before prescribing. MHD TDM: therapeutic range 12-35 µg/mL.",
        "notes": "Preferred over CBZ if: (1) significant drug interactions (OXC less inducing); "
                 "(2) CBZ tolerability issues; (3) female of childbearing age (OXC may be "
                 "marginally better teratogenic profile vs CBZ, though both require folate).",
    },
    {
        "drug": "Observation / watchful waiting (BFIE — no AED)",
        "brand": "N/A — no medication",
        "evidence": "Level B — BFIE management standard (most cases need no chronic AED)",
        "dose_pediatric": "N/A",
        "dose_adult": "N/A",
        "titration": "N/A",
        "moa": (
            "BFIE is self-limited by 18-24 months in >95% of PRRT2 cases — the natural history "
            "of the condition. Prophylactic AED for BFIE is not evidence-based and exposes "
            "the infant to drug side effects unnecessarily. Management: parental seizure first aid "
            "education, rescue medication for prolonged cluster, fever management, and "
            "scheduled follow-up. Most international guidelines do not recommend prophylactic "
            "AED for typical BFIE."
        ),
        "efficacy": "N/A — expectant management. Outcome: excellent. >95% seizure-free by 24M "
                    "without chronic AED. Neurodevelopmental outcome: normal in >95%.",
        "safety": "N/A — avoids AED side-effect exposure in developing infant brain.",
        "monitoring": "Clinical follow-up every 3-6M. EEG if seizures persist beyond 18M or "
                       "if developmental concerns arise.",
        "notes": "SHORT-TERM AED: If BFIE clusters are very frequent or severe, short-term "
                 "phenobarbitone or LEV for 3-6 months during peak cluster phase (6-12 months) "
                 "is acceptable, with planned taper by 18-24 months. RESCUE: diazepam rectal/buccal "
                 "0.3-0.5 mg/kg for cluster >3 seizures in 24h or any seizure >5 min.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "brand": "Keppra / generic",
        "evidence": "Level C — BFIE short-term (if AED selected for BFIE phase)",
        "dose_pediatric": "20-40 mg/kg/day in 2 divided doses",
        "dose_adult": "250-500mg BD",
        "titration": "Start 10 mg/kg/day; increase to 20-40 over 2 weeks.",
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) modulator — reduces neurotransmitter "
            "release. Does NOT affect Nav channels directly. Less mechanistically aligned "
            "with PRRT2 LOF than CBZ but broadly effective for focal epilepsy and safer "
            "in infants than some alternatives."
        ),
        "efficacy": "Moderate — reduces BFIE cluster frequency and duration when AED chosen. "
                    "Less dramatically effective for PKD than CBZ (PKD requires Nav blockade). "
                    "Useful for BFIE if CBZ/OXC not preferred.",
        "safety": "Behavioural side effects (irritability, aggression — 'Keppra rage') in 10-15% "
                  "of children. Otherwise good safety profile. No HLA testing needed.",
        "monitoring": "Clinical — no TDM routinely needed. Monitor for behavioural changes.",
        "notes": "NOT recommended as primary PKD treatment (insufficient Nav effect). "
                 "Acceptable BFIE AED when CBZ avoided (e.g. HLA-B*15:02 positive) or as "
                 "bridge therapy during cluster phase.",
    },
    {
        "drug": "Phenobarbitone (PB)",
        "brand": "Luminal / generic",
        "evidence": "Level C — BFIE historical first-line (now superseded)",
        "dose_pediatric": "3-5 mg/kg/day",
        "dose_adult": "60-180mg/day",
        "titration": "Start 3 mg/kg/day; titrate to response.",
        "moa": "GABAA receptor positive allosteric modulator — prolongs chloride channel opening → "
               "hyperpolarisation. Broad-spectrum anti-seizure effect. Historically used for BFIE "
               "before genetic characterisation; largely replaced by LEV in modern practice.",
        "efficacy": "Effective for BFIE cluster suppression. Not effective for PKD (different mechanism). "
                    "Only used when other agents fail or in resource-limited settings.",
        "safety": "Sedation, hypnotic dependence, cognitive effects (concern in infants — developmental "
                  "neurotoxicity risk at high doses); enzyme inducer (CYP450). "
                  "Withdrawal syndrome if abrupt discontinuation.",
        "monitoring": "TDM: 15-40 µg/mL. Sedation monitoring. Liver enzymes.",
        "notes": "Use is declining for BFIE — prefer LEV or watchful waiting. "
                 "PB may be used in NICU for prolonged neonatal seizures but not standard BFIE treatment.",
    },
    {
        "drug": "Valproic acid / valproate (VPA)",
        "brand": "Epilim / Depakote",
        "evidence": "Level C — BFIE adjunct (use with caution in girls/women)",
        "dose_pediatric": "20-40 mg/kg/day in 2-3 divided doses",
        "dose_adult": "400-1000mg/day",
        "titration": "Start 10 mg/kg/day; increase slowly.",
        "moa": "Multi-modal: Nav inactivation + GABA transaminase inhibition → increased GABA + "
               "T-type calcium channel block. Broad-spectrum. Not Nav-selective — less optimal "
               "for PKD than CBZ.",
        "efficacy": "Effective for BFIE focal/generalised seizures. Not recommended as PKD first-line "
                    "(CBZ superior for kinesiogenic component).",
        "safety": "TERATOGEN (major congenital malformations ~10%, spina bifida, cognitive impairment "
                  "in children exposed in utero) — ABSOLUTELY CONTRAINDICATED in pregnancy and "
                  "women of childbearing potential without PREVENT programme counselling. "
                  "Weight gain, polycystic ovary syndrome, pancreatitis, hepatotoxicity. "
                  "POLG mitochondrial disease exclusion before use.",
        "monitoring": "TDM 50-100 µg/mL; LFTs + FBC + ammonia; weight. "
                       "Females: PREVENT programme, effective contraception, pregnancy register.",
        "notes": "Avoid in girls >10 years or women of childbearing potential unless no alternative "
                 "(UK MHRA/EMA valproate safety restrictions). POLG exclusion before use in infants.",
    },
    {
        "drug": "Genetic counselling (all PRRT2 patients)",
        "brand": "N/A — genetic medicine service",
        "evidence": "Level A — mandatory component of PRRT2 management",
        "dose_pediatric": "N/A",
        "dose_adult": "N/A",
        "titration": "N/A",
        "moa": (
            "Not pharmacological. PRRT2 is autosomal dominant (~60-80% penetrance). "
            "First-degree relatives (parents, siblings, children of affected individuals) "
            "should be offered PRRT2 testing. Preconception counselling: 50% transmission "
            "risk per pregnancy; prenatal genetic diagnosis available. "
            "Recurrent variant (c.649dupC in 80%): targeted testing cost-effective "
            "for at-risk family members."
        ),
        "efficacy": "Enables family screening, early BFIE diagnosis (avoid unnecessary workup), "
                    "early PKD diagnosis and treatment, informed family planning.",
        "safety": "N/A — no pharmacological risk. Psychosocial impact of genetic diagnosis "
                  "must be addressed.",
        "monitoring": "Cascade genetic testing every 3 months in identified AD family — "
                       "particularly before reproductive decision-making.",
        "notes": "Referral to clinical genetics at diagnosis. Molecular result turnaround: "
                 "typically 4-6 weeks for targeted PRRT2 sequencing; 6-12 weeks for panel/WES.",
    },
    {
        "drug": "Diazepam (rescue — BFIE clusters)",
        "brand": "Stesolid rectal / Epistatus buccal",
        "evidence": "Level B — acute BFIE cluster rescue",
        "dose_pediatric": "0.3-0.5 mg/kg rectal (max 10mg); or 0.1-0.3 mg/kg buccal midazolam",
        "dose_adult": "N/A (BFIE is infantile)",
        "titration": "Single dose; can repeat once after 10 min if seizure continues.",
        "moa": "GABAA receptor positive allosteric modulator — chloride influx → rapid seizure termination.",
        "efficacy": "Highly effective for cluster termination in BFIE. Not a prophylactic.",
        "safety": "Respiratory depression (rare at recommended doses); sedation. "
                  "Do not give if seizure has already terminated (respiratory risk outweighs benefit).",
        "monitoring": "Parental training mandatory. Written seizure action plan with clear "
                       "indications: >3 seizures in 24h, or any seizure >5 minutes.",
        "notes": "BFIE rescue essential for all PRRT2 BFIE families regardless of whether "
                 "chronic AED is prescribed. Review action plan at every clinical visit.",
    },
]

# ── Contraindications (4) ─────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "item": "CBZ/OXC in HLA-B*15:02 carriers — ABSOLUTE CONTRAINDICATION (SJS/TEN)",
        "reason": (
            "HLA-B*15:02 allele (Southeast Asian, Han Chinese, Thai, Vietnamese, Filipino ancestry) "
            "carriers have >15-fold increased risk of CBZ-induced Stevens-Johnson Syndrome (SJS) "
            "and Toxic Epidermal Necrolysis (TEN) — potentially fatal. OXC also implicated (lower "
            "risk than CBZ but CPIC Level A: avoid both). CPIC guideline 2023 (HLA-B and CBZ/OXC): "
            "Genotype before prescribing. If HLA-B*15:02 positive: use LCM, LEV, or LTG as "
            "alternative for BFIE. PKD in HLA-B*15:02 carrier: consider LCM (effective for PKD "
            "in case series) or OXC with extreme caution + written consent. "
            "Mandatory testing before CBZ/OXC in any patient of SE Asian descent."
        ),
    },
    {
        "item": "Valproate in girls ≥10 years / women of childbearing potential — RESTRICTED",
        "reason": (
            "VPA is a known major teratogen: ~10% major congenital malformation risk, 6-9-fold "
            "increased neural tube defect risk, and dose-dependent cognitive impairment in "
            "children exposed in utero (IQ reduction 7-10 points vs unexposed). UK MHRA/EMA "
            "valproate safety restrictions: must not be prescribed for epilepsy in girls ≥10 "
            "years or women without Pregnancy Prevention Programme (PPP) enrolment. "
            "For PRRT2: BFIE is self-limited → VPA almost never needed in girls/women; "
            "for PKD, CBZ/OXC are first-line and VPA has no role. "
            "If VPA clinically necessary despite restrictions: PPP mandatory (effective "
            "contraception + annual review + signed agreement)."
        ),
    },
    {
        "item": "Long-term AED maintenance for BFIE beyond 18-24 months — AVOID (overtreatment)",
        "reason": (
            "BFIE self-limits by 18-24 months in >95% of PRRT2 patients. Continuing AED "
            "indefinitely for a self-limited condition exposes children to unnecessary "
            "drug side effects (cognitive, behavioural, systemic), polypharmacy risk, "
            "and medicalization of a benign condition. Once a child is >24 months and "
            "has been seizure-free for 6-12 months, AED taper and discontinuation is "
            "strongly recommended. Failure to taper = iatrogenic harm. Exception: "
            "if BFIE persists beyond 24 months (atypical — consider alternative diagnosis)."
        ),
    },
    {
        "item": "Sodium channel blockers (CBZ/OXC/PHT/LCM) in phenotypically uncertain infant — DEFER until Dravet excluded",
        "reason": (
            "Dravet syndrome (SCN1A haploinsufficiency) in infancy can mimic BFIE in early stages "
            "(febrile clusters 5-9 months). Sodium channel blockers (CBZ, OXC, PHT, LCM) are "
            "CONTRAINDICATED in Dravet and can precipitate status epilepticus or developmental "
            "regression. Before PRRT2 confirmation: if any feature of Dravet is present "
            "(prolonged febrile seizure >15min, alternating hemiclonic semiology, fever sensitivity "
            "at very first seizure, family history of Dravet), obtain SCN1A testing before starting "
            "Nav blocker. For typical short BFIE clusters without Dravet features, PRRT2 testing "
            "can proceed with CBZ start after 1-2 week turnaround; or use LEV/PB in interim."
        ),
    },
]

# ── Monitoring Items (8) ──────────────────────────────────────────────────────
MONITORING = [
    {"item": "HLA-B*15:02 genotype (CPIC Level A — pre-CBZ/OXC)", "schedule": "Once before CBZ/OXC prescription",
     "detail": "Mandatory in patients of SE Asian ancestry (Han Chinese, Thai, Vietnamese, Filipino, "
               "Korean, Malaysian, Indonesian). Commercial test turnaround 2-5 days. If positive: "
               "do NOT use CBZ/OXC — use LCM or LEV as alternative. Result documented in notes."},
    {"item": "Video-EEG (to confirm PKD vs focal seizure)", "schedule": "Once at PKD diagnosis",
     "detail": "Gold standard: capture a kinesiogenic attack on video with simultaneous EEG. "
               "Normal EEG during attack confirms PKD; epileptiform discharge refutes it. "
               "Induction: ask patient to make a sudden movement (stand from chair). "
               "Reports: document attack morphology, duration, EEG findings."},
    {"item": "CBZ clinical response monitoring (PKD)", "schedule": "2-4 weeks after initiation; then q6M",
     "detail": "At-home attack diary: date/time/duration/trigger/severity for each PKD episode. "
               "Target: complete attack control within 2-4 weeks of therapeutic CBZ dose. "
               "If inadequate response at 200mg/day: confirm diagnosis (video-EEG), then increase. "
               "TDM rarely needed at low PKD doses — clinical response guides titration."},
    {"item": "EEG (BFIE — during and between clusters)", "schedule": "At diagnosis; repeat if atypical course",
     "detail": "Routine EEG: normal interictal in 85-90% of BFIE between clusters (confirms benign "
               "nature). If BFIE EEG shows persistent IED or generalised background slowing: "
               "rethink diagnosis — consider SCN1A/Dravet, KCNQ2-DEE, pyridoxine-dependent epilepsy. "
               "Video-EEG during cluster: documents focal onset pattern for diagnosis."},
    {"item": "Neurodevelopmental assessment (BFIE follow-up)", "schedule": "At 12 months, 24 months, 5 years",
     "detail": "Bayley Scales of Infant Development (Bayley-III/Bayley-4) at 12M and 24M. "
               "WPPSI or developmental milestones at 3-5 years. "
               "Expected: normal in >95% of isolated PRRT2 BFIE. "
               "If developmental delay detected: extend genetic workup (16p11.2 CMA, broader panel)."},
    {"item": "Chromosomal Microarray (CMA) — if BFIE + developmental delay", "schedule": "Once at diagnosis if atypical",
     "detail": "Routine PRRT2 gene sequencing does NOT detect 16p11.2 microdeletion. "
               "If BFIE patient has developmental delay, autism features, macrocephaly, "
               "or dysmorphic features: add CMA. The 16p11.2 deletion is the second most "
               "common cause of BFIE after isolated PRRT2 variants — and has distinct management."},
    {"item": "Family cascade genetic testing (PRRT2)", "schedule": "Within 3 months of index case confirmation",
     "detail": "Autosomal dominant — first-degree relatives (parents, siblings >5 years, adult children) "
               "offered targeted PRRT2 testing. c.649dupC hotspot: commercially available targeted "
               "assay (rapid, low cost). Positive relatives: assess for subclinical BFIE history, "
               "current PKD symptoms, neurology referral if symptomatic."},
    {"item": "Seizure and PKD attack diary", "schedule": "Ongoing from diagnosis",
     "detail": "Paper or app-based diary: BFIE clusters (date, number of seizures, duration, "
               "fever, emergency medication use) and PKD episodes (date, trigger, duration, "
               "frequency, severity). Guides AED adjustment and documents remission. "
               "Recommended apps: EpiDiary, Seizure Tracker, or Neurology clinic portal."},
]

# ── Lifecycle Windows (6) ─────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Pre-symptomatic / cascade-detected (any age — familial or prenatal testing)",
        "age": "Variable — positive family history or prenatal genetic diagnosis",
        "focus": (
            "PRRT2 carrier identified via family cascade testing (sibling/parent testing after "
            "index case) or prenatal diagnosis. Pre-symptomatic counselling: "
            "(1) 60-80% penetrance → not all carriers develop BFIE or PKD; "
            "(2) if infant: monitor for BFIE clusters at 3-12 months — have rescue diazepam ready; "
            "(3) adolescent: watch for kinesiogenic attacks — early CBZ if PKD develops. "
            "No prophylactic treatment. Reassure: excellent prognosis for both BFIE and PKD."
        ),
        "key_actions": [
            "Genetic confirmation (PRRT2 c.649dupC targeted test or full gene sequencing)",
            "Parental BFIE first-aid education + rescue diazepam prescription",
            "Paediatric neurology referral",
            "Document in medical record — inform future providers",
        ],
    },
    {
        "window": "Infantile BFIE phase (peak symptomatic period)",
        "age": "3-18 months (peak 5-7 months)",
        "focus": (
            "Classic BFIE presentation: focal seizure clusters (3-10 per day, lasting 1-5 days). "
            "Priority: (1) confirm PRRT2 diagnosis (rapid genetic testing); "
            "(2) exclude Dravet syndrome if any atypical features; "
            "(3) reassure family — self-limited prognosis; "
            "(4) provide rescue medication plan; "
            "(5) decide on short-term AED (watchful waiting vs 3-6 month LEV/PB). "
            "Monitoring: neurodevelopment at each visit — any regression = re-evaluate diagnosis."
        ),
        "key_actions": [
            "PRRT2 gene sequencing (rapid 2-4 week turnaround)",
            "If atypical: SCN1A testing before CBZ (Dravet exclusion)",
            "Rescue diazepam + written seizure action plan",
            "EEG (during cluster if possible — video-EEG preferred)",
            "Short-term AED decision (most: observation or 3-6M LEV/PB)",
        ],
    },
    {
        "window": "BFIE remission / seizure-free period",
        "age": "18 months — 5 years",
        "focus": (
            "BFIE resolves spontaneously — confirmed seizure-free period. "
            "Priority: (1) taper and discontinue AED if prescribed; "
            "(2) confirm normal neurodevelopment (Bayley/WPPSI); "
            "(3) educate family on future PKD risk in adolescence; "
            "(4) schedule follow-up at 5 and 10 years for PKD surveillance. "
            "Most families: reassure and discharge from routine neurology to GP "
            "(but maintain genetics follow-up and PKD awareness)."
        ),
        "key_actions": [
            "AED taper and discontinuation (if prescribed — typically by 24-30 months)",
            "Neurodevelopmental assessment (WPPSI, school readiness)",
            "PKD education: what to look for in adolescence, triggers",
            "Genetic counselling update: reproductive planning for parents",
        ],
    },
    {
        "window": "School age — latent period",
        "age": "5-10 years",
        "focus": (
            "Typically asymptomatic period between BFIE remission and PKD onset. "
            "Key: ensure family knows PKD may emerge (sudden movement-triggered attacks, "
            "NOT loss of consciousness) and that immediate neurology referral is appropriate. "
            "Neurodevelopment: normal at school-age cognitive assessment. "
            "No AED during latent period unless EEG/clinical evidence of continued epilepsy."
        ),
        "key_actions": [
            "Annual clinical review (or GP surveillance) for PKD early symptoms",
            "School notification: PRRT2 diagnosis, action plan if seizures resume",
            "Cognitive assessment if any academic concerns",
        ],
    },
    {
        "window": "Adolescent PKD onset",
        "age": "10-18 years (typical onset 10-14 years)",
        "focus": (
            "PKD onset: movement-triggered brief (<60s) episodic dystonia/chorea. "
            "Priority: (1) video-EEG to confirm PKD (normal ictal EEG = confirms PKD, not seizure); "
            "(2) start CBZ 50-100mg/day titrated to complete control; "
            "(3) HLA-B*15:02 genotype before CBZ (SE Asian ancestry); "
            "(4) school/sports notification — PKD attacks during PE class a significant concern; "
            "(5) driving advice: no driving until attack-free on treatment (varies by jurisdiction). "
            "Psychosocial: PKD can cause significant social embarrassment — address mental health."
        ),
        "key_actions": [
            "Video-EEG for PKD confirmation",
            "HLA-B*15:02 genotype",
            "CBZ initiation (50-200mg/day)",
            "School/PE teacher notification — PKD is NOT epilepsy but appears similar",
            "Driving: defer until confirmed attack-free on treatment",
            "Mental health screening (PKD stigma, peer awareness)",
        ],
    },
    {
        "window": "Adulthood — maintained remission / CBZ taper",
        "age": "25+ years",
        "focus": (
            "PKD naturally remits in many adults by mid-adulthood (25-35 years). "
            "Priority: (1) assess PKD frequency — if attack-free for 2+ years: offer CBZ taper; "
            "(2) manage reproductive issues (CBZ teratogen — discuss with women of childbearing age); "
            "(3) cascade family screening continues as new relatives born; "
            "(4) driving licence: if attack-free and CBZ tapered, UK DVLA requires 1-year "
            "attack-free before notification of successful management; "
            "(5) employment: inform occupational health if job involves operating machinery. "
            "Most adults achieve complete PKD remission by age 30-40 without AED."
        ),
        "key_actions": [
            "CBZ taper trial if attack-free ≥2 years",
            "Women: contraception counselling if on CBZ (enzyme inducer → OCP failure)",
            "Driving: jurisdiction-specific attack-free period compliance",
            "Cascade genetic testing for children of affected individual",
            "Occupational health notification if relevant",
        ],
    },
]

# ── Key Definitions (14 concepts) ─────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "PRRT2 (Proline-Rich Transmembrane Protein 2)",
        "definition": (
            "PRRT2 encodes a single-pass transmembrane protein (340 amino acids) with a large "
            "proline-rich N-terminal cytoplasmic domain and a short C-terminal extracellular "
            "domain. Located at chromosome 16p11.2 (within the autism-associated 16p11.2 region). "
            "PRRT2 is preferentially expressed in glutamatergic neurons of hippocampus, cortex, "
            "cerebellum, and basal ganglia. At presynaptic terminals, PRRT2 directly binds "
            "SNAP25 (synaptosomal-associated protein 25kDa) — a component of the SNARE complex "
            "required for synaptic vesicle fusion. The PRRT2-SNAP25 interaction restrains "
            "voltage-gated Na+ channel (Nav1.2, Nav1.6) surface trafficking. PRRT2 LOF removes "
            "this brake → excess Nav surface expression → episodic sodium-driven hyperexcitability."
        ),
    },
    {
        "term": "BFIE / BFIS (Benign Familial Infantile Epilepsy / Seizures)",
        "definition": (
            "BFIE (OMIM #605751): autosomal dominant condition characterised by brief focal or "
            "generalised tonic-clonic seizures in clusters, onset 3-12 months, normal EEG "
            "between clusters, normal neurodevelopment, and spontaneous remission by 18-24 months. "
            "PRRT2 accounts for ~90% of familial BFIE. Benign: excellent prognosis without "
            "significant neurodevelopmental sequelae. ILAE 2022: classified as a self-limited "
            "focal epilepsy of infancy. KEY: does NOT evolve to epileptic encephalopathy."
        ),
    },
    {
        "term": "PKD (Paroxysmal Kinesigenic Dyskinesia)",
        "definition": (
            "PKD (OMIM #128200): episodic movement disorder characterised by brief (<60 seconds), "
            "frequent (up to 100/day in untreated severe cases), sudden movement-triggered attacks "
            "of involuntary movements (dystonia, chorea, athetosis, or mixed). Consciousness "
            "preserved throughout. No EEG correlate (not epileptic). PRRT2 is the most common "
            "identified genetic cause (~80% of familial PKD). Onset: typically 6-16 years "
            "(adolescence). Excellent CBZ/OXC response (>90% attack-free at low doses). "
            "NOT epilepsy — different mechanism and different treatment context, though often "
            "coexists with BFIE in PRRT2 carriers."
        ),
    },
    {
        "term": "ICCA (Infantile Convulsions and Choreoathetosis)",
        "definition": (
            "ICCA (OMIM #602066): combined phenotype of BFIE in infancy and PKD in adolescence "
            "occurring in the same individual or in the same family. The combination was first "
            "described in 1997 (Szepetowski) and linkage-mapped to 16p11.2 before PRRT2 was "
            "identified in 2011 as the causative gene. ICCA is now understood as the full-spectrum "
            "phenotypic expression of PRRT2 LOF. Many apparent BFIE families have PKD members "
            "not previously linked — family history must explicitly enquire about movement disorders."
        ),
    },
    {
        "term": "c.649dupC (p.Arg217Profs*8) — PRRT2 Hotspot Variant",
        "definition": (
            "The single most common pathogenic PRRT2 variant globally — a cytosine duplication "
            "at position 649 of exon 2 within a poly-C (7-cytosine) homopolymer run. This creates "
            "a frameshift leading to a premature stop codon at residue 224 (8 codons downstream "
            "of the frameshift) with subsequent NMD-mediated transcript degradation → PRRT2 "
            "haploinsufficiency. Accounts for ~80% of all pathogenic PRRT2 variants. The poly-C "
            "stretch predisposes to replication slippage — explaining the variant's high frequency "
            "and recurrent de novo occurrence rate (~15%). ACMG classification: Pathogenic "
            "(PVS1 + PM2 + PP1)."
        ),
    },
    {
        "term": "Kinesiogenic Trigger",
        "definition": (
            "The defining feature of PKD: sudden, voluntary movement initiation triggers the "
            "dyskinetic attack. Specifically: abrupt onset of voluntary movement (not gradual or "
            "sustained) — e.g., standing from a sitting position, starting to walk from standing, "
            "reaching suddenly, turning rapidly. The movement velocity change is critical — "
            "slow, steady movements do not trigger. Neurophysiological basis: sudden motor "
            "cortex activation → transient Nav-channel opening surge (normally dampened by "
            "PRRT2-SNAP25 interaction) → brief dysrhythmic basal ganglia-cortical discharge "
            "→ involuntary movement. Diagnostic criterion: >3 PKD diagnostic criteria must be "
            "met including kinesiogenic trigger."
        ),
    },
    {
        "term": "SNAP25 (Synaptosomal-Associated Protein 25kDa)",
        "definition": (
            "SNAP25 is a Q-SNARE protein at the presynaptic terminal — component of the "
            "VAMP2/SNAP25/Syntaxin-1 SNARE complex that drives synaptic vesicle fusion and "
            "neurotransmitter release. PRRT2 directly binds SNAP25 at the presynaptic membrane. "
            "This PRRT2-SNAP25 interaction modulates Nav channel (Nav1.2/Nav1.6) surface "
            "trafficking — PRRT2 LOF → reduced SNAP25-Nav coupling → Nav channel overexpression "
            "at presynaptic membrane → episodic sodium-driven neuronal hyperexcitability. "
            "SNAP25 variants themselves cause Neurodevelopmental Disorder with Hyperkinetic "
            "Movements and Learning Difficulties (NEDHML) — separate condition."
        ),
    },
    {
        "term": "16p11.2 Microdeletion (including PRRT2)",
        "definition": (
            "Recurrent ~593kb copy number variant at chromosome 16p11.2 (breakpoint region 4-5, "
            "BP4-BP5) — the most common autism-associated CNV (frequency ~1/3000 births). "
            "Spans ~25 genes including PRRT2, TBX6, ALDOA, MAPK3, TAOK2. Unlike isolated "
            "PRRT2 LOF (excellent prognosis), 16p11.2 deletion causes: mild-moderate intellectual "
            "disability, autism spectrum disorder, speech delay, macrocephaly, obesity. "
            "BFIE/PKD component attributed to PRRT2 haploinsufficiency within the deletion. "
            "Detected by CMA not sequencing. De novo in ~60% of probands; inherited in ~40% "
            "(parent often has milder phenotype — variable expressivity)."
        ),
    },
    {
        "term": "CBZ (Carbamazepine) Precision in PKD",
        "definition": (
            "CBZ's near-complete efficacy for PKD at low doses (50-200mg/day — much less than "
            "epilepsy doses 400-1600mg/day) reflects the different pathophysiology: "
            "PKD = transient episodic Nav excess → low-level Nav blockade sufficient for "
            "prevention. This dose-response contrast is diagnostically useful: complete PKD "
            "control at 100mg/day confirms the diagnosis. The PKD dose is sub-therapeutic "
            "for epilepsy — if typical epilepsy doses needed, reconsider whether diagnosis is "
            "PKD or focal seizures."
        ),
    },
    {
        "term": "HLA-B*15:02 / CBZ Pharmacogenomics (CPIC Level A)",
        "definition": (
            "HLA-B*15:02 allele is strongly associated with CBZ/OXC-induced Stevens-Johnson "
            "Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) — both potentially fatal "
            "cutaneous adverse drug reactions. Prevalence: 5-15% of Han Chinese, Thai, Vietnamese, "
            "Filipino, Malaysian populations; <1% in European, Japanese. CPIC 2023 guideline: "
            "AVOID CBZ and OXC in HLA-B*15:02 carriers (Level A recommendation). "
            "Test before prescribing in any patient of SE Asian ancestry. "
            "Commercial test turnaround: 2-5 days. If carrier: use LCM, LEV, or VPA (with "
            "appropriate safety monitoring) as alternative."
        ),
    },
    {
        "term": "Video-EEG Differentiation (PKD vs Focal Motor Seizure)",
        "definition": (
            "The gold standard for confirming PKD vs focal motor seizure. Protocol: "
            "(1) EEG electrode placement as standard 10-20; "
            "(2) patient asked to make sudden movements (stand from chair) during recording; "
            "(3) video captures attack semiology simultaneously. "
            "PKD: normal EEG throughout — no ictal discharge. Focal motor seizure: "
            "focal EEG ictal onset (typically frontal or centroparietal) with evolving discharge. "
            "Critical clinical implication: PKD confirmed → low-dose CBZ; "
            "focal seizure confirmed → standard epilepsy workup including MRI and AED titration. "
            "Misdiagnosis of focal seizure as PKD (or vice versa) leads to wrong treatment."
        ),
    },
    {
        "term": "AD Incomplete Penetrance in PRRT2",
        "definition": (
            "PRRT2 pathogenic variants (including the common c.649dupC hotspot) demonstrate "
            "autosomal dominant inheritance with incomplete penetrance (~60-80%). "
            "Consequence: some confirmed PRRT2 variant carriers (from molecular testing of "
            "family members) have no clinical BFIE or PKD history. This non-penetrance "
            "is not a variant reclassification criterion — penetrance <100% does not "
            "indicate benign variant. Clinical implication: absence of symptoms in a parent "
            "carrying the same variant does NOT mean the child is unaffected. "
            "Conversely, a clinically affected relative strengthens pathogenicity classification."
        ),
    },
    {
        "term": "Dravet Syndrome Exclusion (before CBZ in BFIE)",
        "definition": (
            "Dravet syndrome (SCN1A haploinsufficiency) is the critical differential for BFIE: "
            "onset ~5-9 months, febrile convulsions, focal or generalised. "
            "CBZ, OXC, PHT (Na+ channel blockers) are CONTRAINDICATED in Dravet — "
            "can worsen seizures and trigger status epilepticus. "
            "If any Dravet feature present: (prolonged febrile seizure >15 min, "
            "alternating hemiclonic semiology, very high fever sensitivity from first seizure, "
            "developmental stagnation after 2nd seizure year) → test SCN1A first. "
            "PRRT2 testing can proceed simultaneously but do NOT start CBZ until Dravet excluded. "
            "Use LEV or PB as interim AED if treatment urgent."
        ),
    },
    {
        "term": "PRRT2 Alliance / Patient Support",
        "definition": (
            "PRRT2-related disorders patient support resources: "
            "International League Against Epilepsy (ILAE) Genetic Epilepsy Commission — "
            "PRRT2 is included in ILAE 2022 framework for self-limited infantile epilepsies. "
            "Rare Epilepsy Network (REN) — patient registry includes BFIE. "
            "Movement Disorder Society (MDS) — PKD consensus guidelines. "
            "PRRT2 families benefit from connection with PKD-specific organisations: "
            "Dystonia UK (for PKD), EpiCARENetwork (European Reference Network for rare epilepsies). "
            "Genetic testing support: Shire Genetics, Blueprint Genetics, Invitae, GeneDx "
            "all offer PRRT2 gene-specific testing as well as epilepsy gene panels."
        ),
    },
]

# ── Clinical Alerts (top 3 for overview) ──────────────────────────────────────
ALERTS = [
    "⚠ MISDIAGNOSIS RISK: PKD is NOT epilepsy — normal ictal EEG is diagnostic. "
    "Do NOT prescribe chronic AED for epilepsy when PKD attacks explain all events. "
    "Video-EEG is mandatory before chronic AED in suspected PKD.",

    "⚠ CBZ HLA-B*15:02: Mandatory genotype testing before CBZ/OXC in SE Asian patients. "
    "Carriers: use LCM or LEV for BFIE. For PKD: LCM is an effective alternative.",

    "⚠ DRAVET EXCLUSION: NEVER start CBZ/OXC without SCN1A exclusion if any Dravet feature "
    "is present (prolonged febrile seizure >15 min, alternating hemiclonic). "
    "Na+ blockers CONTRAINDICATED in Dravet — can precipitate SE.",
]

# ── Standards / Guidelines (8) ────────────────────────────────────────────────
STANDARDS = [
    "ILAE-2022 — Self-limited focal epilepsies of infancy (BFIE classification)",
    "NICE-NG217 — Epilepsy in adults/children (UK guideline, 2022)",
    "EAN-MDS-PKD-Guidelines-2020 — European PKD management consensus",
    "CPIC-HLA-B-CBZ-OXC-2023 — Carbamazepine/HLA-B*15:02 pharmacogenomics (Level A)",
    "ACMG-AMP-2015 — Variant classification standards",
    "ACNS-EEG-2021 — EEG recording standards",
    "Chen-2011-NatGenet — PRRT2 discovery in PKD",
    "Heron-2012-NatGenet — PRRT2 in BFIE",
]

# ── Thresholds (10) ───────────────────────────────────────────────────────────
THRESHOLDS = [
    "BFIE-onset-3-12M — outside window: reconsider diagnosis",
    "BFIE-remission-24M — if seizures persist beyond 24M: re-evaluate (atypical BFIE)",
    "PKD-attack-duration-<60s — attack >60s: reconsider PKD diagnosis",
    "CBZ-low-dose-PKD-100-200mg-day — high doses not needed for PKD",
    "CBZ-TDM-4-12-mcg-mL — clinical response guides at low PKD doses",
    "HLA-B1502-CBZ-OXC-CI — mandatory test in SE Asian ancestry",
    "Video-EEG-normal-ictal-confirms-PKD — epileptiform = not PKD",
    "2-AED-failures-neurology-tertiary-referral — re-evaluate diagnosis if 2 AEDs fail",
    "Family-cascade-3M-from-index — all first-degree relatives tested within 3 months",
    "Dravet-exclusion-before-CBZ — SCN1A negative before Na+ blocker in atypical infant",
]

# ── References (6) ────────────────────────────────────────────────────────────
REFERENCES = [
    {
        "citation": "Chen WJ et al. (2011) Nat Genet 43:1252-1253",
        "title": "Exome sequencing identifies truncating mutations in PRRT2 that cause paroxysmal kinesigenic dyskinesia",
        "relevance": "PRRT2 discovery paper — identified PRRT2 as the causative gene for PKD via exome sequencing",
    },
    {
        "citation": "Heron SE et al. (2012) Nat Genet 44:1151-1153",
        "title": "PRRT2 mutations cause benign familial infantile epilepsy and infantile convulsions with choreoathetosis syndrome",
        "relevance": "Established PRRT2 as the gene for BFIE and ICCA — unified the PRRT2 epilepsy-dyskinesia spectrum",
    },
    {
        "citation": "Ebrahimi-Fakhari D et al. (2015) Neurology 85:1386-1391",
        "title": "The spectrum of movement disorders in childhood-onset genetic epilepsies",
        "relevance": "PRRT2 movement disorder spectrum review — PKD phenotypic characterisation and natural history",
    },
    {
        "citation": "Ono S et al. (2017) Brain 140:3219-3234",
        "title": "PRRT2 missense mutation cluster at C-terminal causes familial infantile convulsions with paroxysmal choreoathetosis",
        "relevance": "PRRT2 missense variant functional characterisation and genotype-phenotype correlations",
    },
    {
        "citation": "Gardiner AR et al. (2012) Brain 135:2528-2536",
        "title": "The clinical and genetic heterogeneity of paroxysmal dyskinesias",
        "relevance": "Clinical heterogeneity of PKD spectrum; PRRT2 in broader PKD genetic context",
    },
    {
        "citation": "Liu XR et al. (2012) Nat Genet 44:1147-1151",
        "title": "The PRRT2 gene mutations in patients with infantile convulsions and paroxysmal kinesigenic dyskinesia",
        "relevance": "Independent replication of PRRT2-ICCA association; hotspot c.649dupC prevalence",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    """PRRT2 Epilepsy Spectrum — overview endpoint."""
    total = sum(e["n"] for e in ETIOLOGY_CATALOG)
    return {
        "syndrome": "PRRT2 Epilepsy Spectrum (BFIE / PKD / ICCA)",
        "gene": "PRRT2 — 16p11.2 — Proline-Rich Transmembrane Protein 2",
        "protein_function": "PRRT2 binds SNAP25 → modulates Nav1.2/Nav1.6 surface expression at presynaptic terminal",
        "lof_consequence": "PRRT2 LOF → disrupted SNAP25-Nav interaction → episodic neuronal hyperexcitability",
        "inheritance": "Autosomal dominant — incomplete penetrance (~60-80%)",
        "hotspot_variant": "c.649dupC (p.Arg217Profs*8) — accounts for ~80% of pathogenic PRRT2 variants",
        "phenotypes": {
            "BFIE": "Benign Familial Infantile Epilepsy — focal seizure clusters 3-12 months, self-limited by 24M",
            "PKD": "Paroxysmal Kinesigenic Dyskinesia — movement-triggered brief attacks, CBZ first-line",
            "ICCA": "Infantile Convulsions + Choreoathetosis — BFIE + PKD in same individual/family",
        },
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
        "cbz_efficacy_pkd": "Near-complete control (>90%) at low doses 100-200mg/day",
        "bfie_prognosis": "Excellent — self-limited by 18-24 months in >95%; normal neurodevelopment",
        "key_safety": "HLA-B*15:02 before CBZ (SE Asian); Dravet exclusion before Na+ blockers",
        "top_alerts": ALERTS[:3],
        "generated_at": datetime.now().isoformat(),
        "dashboard_id": 183,
    }


def get_breakdown():
    """PRRT2 Epilepsy Spectrum — breakdown endpoint (full clinical detail)."""
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
    """PRRT2 Epilepsy Spectrum — definitions endpoint (14 key concepts)."""
    return {
        "syndrome": "PRRT2 Epilepsy Spectrum (BFIE / PKD / ICCA)",
        "definitions": DEFINITIONS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "generated_at": datetime.now().isoformat(),
    }
