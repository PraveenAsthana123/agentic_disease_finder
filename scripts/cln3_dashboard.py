"""
CLN3 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 3 / Juvenile Batten Disease (JNCL)
========================================================================================
40-patient cohort · CLN3 (16p12.1) · Autosomal recessive (AR)
CLN3 = Battenin — a 438-aa, 7-transmembrane lysosomal/late-endosomal membrane protein
LOF → lysosomal trafficking disruption → autofluorescent storage material accumulation
(fingerprint profiles + curvilinear bodies + rectilinear complexes on EM) →
neurodegeneration (cortical + basal ganglia + retinal) → JNCL (Juvenile Batten Disease)

MOST COMMON NCL WORLDWIDE:
CLN3 disease (JNCL) is the most common neuronal ceroid lipofuscinosis (NCL) globally.
Incidence ~1 in 100,000 live births; carrier frequency ~1 in 160 in Western populations.
Most common NCL among school-age children and young adults.
No disease-modifying therapy (unlike CLN2 which has cerliponase alfa ICV enzyme replacement).

VISUAL FAILURE IS FIRST SYMPTOM — DIAGNOSTIC PATHWAY BEGINS WITH OPHTHALMOLOGY:
JNCL uniquely presents with visual failure BEFORE seizures by 2-5 years:
  - Bull's eye maculopathy → progressive retinal degeneration → ERG abnormalities
  - Visual symptoms 4-10 years of age; ophthalmologist is first specialist, not neurologist
  - Seizures emerge later (typically age 10-13 years) — 2-5 years after visual failure
  - This sequence (visual then seizures) distinguishes CLN3 from CLN2 (seizures first)

VACUOLATED LYMPHOCYTES — PATHOGNOMONIC BLOOD FILM FINDING:
Peripheral blood smear shows vacuolated lymphocytes (Gröbel vacuoles) in CLN3 disease.
This is PATHOGNOMONIC and diagnostic — a simple blood test gives the first clue.
Skin biopsy EM: fingerprint profiles + curvilinear bodies confirm NCL storage.

CLN3 PROTEIN BIOLOGY (BATTENIN — LYSOSOMAL MEMBRANE):
CLN3 (16p12.1), Battenin:
  - 438 aa; ~48 kDa; multiple isoforms from alternative splicing
  - 7-transmembrane domain protein localised to lysosomes and late endosomes
  - Function: incompletely understood — proposed roles in:
      (a) Lysosomal pH regulation
      (b) Arginine import/export across lysosomal membrane
      (c) Autophagy regulation and lysosomal biogenesis
      (d) Subunit c (SCMAS) degradation assistance
  - NOT a soluble lysosomal enzyme (unlike CLN2/TPP1 or CLN1/PPT1)
  - No enzymatic activity — structural/transport membrane protein
  - Expression: ubiquitous; highest in neurons and retinal pigment epithelium
  - CLN3 LOF → battenin deficiency → lysosomal trafficking disruption →
      SCMAS accumulation → autofluorescent ceroid storage (fingerprint profiles,
      curvilinear bodies) → progressive neuronal apoptosis → JNCL
  - pLI ~0.88 (high intolerance to heterozygous LOF)
  - OMIM: *607042 (CLN3 gene); #204200 (CLN3 disease / JNCL / Juvenile Batten Disease)
  - Storch S et al. 1995 (Am J Hum Genet) — CLN3 gene cloning; ITF Batten group 1995 Nature Genetics

MAJOR MUTATION — 1.02 kb DELETION (MOST COMMON NCL MUTATION WORLDWIDE):
  Common deletion: c.461-280_677+382del (exons 7+8 deletion; ~1.02 kb)
  - Accounts for ~73% of all CLN3 disease alleles worldwide
  - Results in frameshift → premature stop → absent CLN3/Battenin
  - Homozygous in 40-50% of patients; compound het with second allele in others
  - Classic JNCL phenotype regardless of compound allele (null effect)
  - Detectable by targeted deletion PCR / MLPA — faster than full sequencing
  - High carrier frequency in Scandinavian populations (Finland ~1:100; Sweden ~1:130)

JNCL vs CLN2 CRITICAL DISTINCTIONS:
  CLN3 (JNCL): Visual failure FIRST (4-10y) → seizures LATER (10-13y); onset 4-10y; fatal 20-30y
  CLN2 (LINCL): Seizures FIRST (2-4y) → visual loss follows; onset 2-4y; fatal 8-12y without Tx
  CLN3: No disease-modifying therapy; CLN2: Cerliponase alfa (ICV ERT, FDA 2017)
  CLN3: Vacuolated lymphocytes PATHOGNOMONIC blood film; CLN2: normal blood film
  CLN3: Giant SSPS NOT pathognomonic (giant SSPS is CLN2 specific); CLN2: SSPS 1-3 Hz pathognomonic
  CLN3: No CLN3 enzyme assay (CLN3 protein not a soluble enzyme); CLN2: TPP1 enzyme assay days
  CLN3: Parkinsonism features in late disease; CLN2: pyramidal/spastic features dominant
  CLN3: Longer survival (2nd-3rd decade); CLN2: Fatal 1st decade without treatment
"""


def get_overview():
    return {
        "gene": "CLN3 (16p12.1) — Battenin (7-transmembrane lysosomal/late-endosomal membrane protein)",
        "protein": "Battenin (CLN3p); 438 aa; ~48 kDa; 7 transmembrane domains; lysosomal and late-endosomal localisation; NOT a soluble lysosomal enzyme (unlike CLN2/TPP1 or CLN1/PPT1); proposed functions: lysosomal pH regulation, arginine transport, autophagy regulation, SCMAS degradation assistance; function incompletely characterised",
        "inheritance": "Autosomal recessive (AR) — biallelic CLN3 LOF; pLI ~0.88; most common: 1.02 kb exon 7+8 deletion (c.461-280_677+382del) ~73% of alleles worldwide; compound heterozygous with missense or nonsense second allele in ~30%; high carrier frequency in Scandinavian populations (~1 in 130 Sweden, ~1 in 100 Finland)",
        "omim": "*607042 (CLN3 gene) · #204200 (CLN3 disease / JNCL / Juvenile Batten Disease)",
        "disease": "CLN3 disease — Juvenile Neuronal Ceroid Lipofuscinosis (JNCL / Spielmeyer-Vogt-Sjögren disease / Juvenile Batten Disease): VISUAL FAILURE is first symptom (age 4-10 years); seizures emerge 2-5 years after visual failure (age 10-13 years); progressive cognitive decline → dementia; motor deterioration; extrapyramidal features (Parkinsonism) in late disease; fatal typically in 2nd-3rd decade",
        "mechanism": "CLN3/Battenin LOF → lysosomal membrane protein deficiency → disruption of lysosomal trafficking and SCMAS degradation → SCMAS accumulation as autofluorescent ceroid storage (fingerprint profiles + curvilinear bodies on EM) → progressive neuronal apoptosis (cortex + basal ganglia + retina) → JNCL disease",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN3 disease. Unlike CLN2 (cerliponase alfa ICV enzyme replacement, FDA 2017), CLN3 has no approved treatment. Clinical trials ongoing: gene therapy (AAV-CLN3 intrathecal/ICV — Phase 1/2 trials), enzyme replacement (investigational), substrate reduction.",
        "cohort_size": 40,
        "female_pct": 50,
        "mean_onset_visual_years": 6.2,
        "mean_onset_seizure_years": 11.4,
        "drug_resistant_pct": 60,
        "photosensitivity_pct": 55,
        "visual_failure_first_pct": 98,
        "vacuolated_lymphocytes_pct": 95,
        "parkinsonism_late_disease_pct": 70,
        "cognitive_decline_pct": 100,
        "behavioural_psychiatric_pct": 85,
        "on_vpa_pct": 70,
        "on_lev_pct": 65,
        "on_ltg_pct": 45,
        "on_ssri_pct": 60,
        "on_levodopa_late_pct": 55,
        "homozygous_exon78_deletion_pct": 47,
        "mean_delay_diagnosis_visual_years": 2.1,
        "discovery": "CLN3 gene cloned: International Batten Disease Consortium 1995 (Nature Genetics 14:154-157); Storch S et al. 1995 (Am J Hum Genet) — characterisation; common deletion: Munroe PB et al. 1997",
        "unique_feature": "MOST COMMON NCL worldwide. Visual failure is FIRST symptom (before seizures by 2-5 years) — unique among NCLs; ophthalmologist-led diagnosis. Vacuolated lymphocytes on blood film PATHOGNOMONIC. No disease-modifying therapy (contrast CLN2 cerliponase alfa). Parkinsonism features in late disease. 1.02 kb exon 7+8 deletion in 73% of alleles — targeted deletion PCR rapid screening.",
        "absolute_ci": [
            "Vigabatrin (VGB) — ABSOLUTE CI: VGB causes irreversible peripheral visual field constriction (vigabatrin-associated retinopathy, VAR); CLN3 patients have severe progressive retinal degeneration leading to blindness; VGB superimposes irreversible retinal toxicity on existing CLN3 retinal disease — catastrophic combined visual loss",
        ],
        "high_risk_ci": [
            "Carbamazepine / Oxcarbazepine / Phenytoin — HIGH RISK: Na-channel blockers can worsen myoclonic seizures in CLN3 mixed seizure disorder; poor choice for dominant seizure phenotype; VPA+LEV strongly preferred",
            "GBP / Pregabalin — HIGH RISK: α2δ modulators can worsen myoclonic component (Crespel 1999 principle applies to NCL myoclonus); multi-specialty prescribing trap in JNCL patients with visual impairment seeking pain services",
            "Antipsychotics with D2-blockade (haloperidol, risperidone high dose) — HIGH RISK: exacerbate extrapyramidal/Parkinsonism features in late CLN3 disease; use atypical agents (quetiapine, clozapine) with D2-sparing profile",
        ],
    }


def get_breakdown():
    etiologies = [
        {
            "class": "Homozygous-1.02kb-Exon7+8-Deletion-Classic",
            "pct": 47,
            "description": "Homozygous 1.02 kb CLN3 deletion (c.461-280_677+382del; exon 7+8 deletion) — most common CLN3 genotype worldwide; frameshift + premature stop → absent Battenin; classic JNCL onset 5-8 years with visual failure; vacuolated lymphocytes confirmed; fingerprint profiles on EM; common founder mutation in Northern European (especially Scandinavian) populations; targeted deletion PCR gives rapid diagnosis",
            "count": 19,
            "gene_mechanism": "Exons 7+8 deletion → frameshift → premature stop codon → NMD → absent CLN3/Battenin → lysosomal storage disorder",
            "key_variants": ["c.461-280_677+382del (1.02 kb deletion homozygous)", "Exon 7+8 deletion PCR-detectable"],
        },
        {
            "class": "Compound-Het-1.02kb-Deletion-Plus-Missense",
            "pct": 26,
            "description": "Compound heterozygous: 1.02 kb deletion (one allele) + missense on second allele (commonest: p.Leu101Pro, p.Val330Ala, p.Arg334Cys, p.Ile294Thr); second allele missense often reduces but does not abolish Battenin expression; may allow slightly later onset or milder progression (attenuated phenotypes reported); vacuolated lymphocytes present; full NCL gene panel identifies second allele",
            "count": 10,
            "gene_mechanism": "Trans-configuration: null allele (deletion) + hypomorphic missense → reduced CLN3 function → lysosomal storage (less severe than homozygous null in some cases)",
            "key_variants": ["c.461-280_677+382del / p.Leu101Pro", "c.461-280_677+382del / p.Val330Ala", "c.461-280_677+382del / p.Arg334Cys"],
        },
        {
            "class": "Compound-Het-Missense-Missense-Attenuated",
            "pct": 12,
            "description": "Compound heterozygous missense + missense — both alleles partially functional; attenuated JNCL phenotype; later onset (8-12 years); slower progression; visual failure may be milder initially; seizures later onset; psychiatric features may dominate early course; vacuolated lymphocytes usually present but fewer; requires CLN3 gene sequencing (not detectable by deletion PCR alone)",
            "count": 5,
            "gene_mechanism": "Biallelic hypomorphic missense → partial Battenin function → attenuated SCMAS accumulation → slower neurodegeneration",
            "key_variants": ["p.Arg334His homozygous (Finnish attenuated variant)", "p.Val444Ile compound het (milder splice effect)"],
        },
        {
            "class": "Homozygous-Nonsense-Frameshift-Null",
            "pct": 10,
            "description": "Homozygous nonsense (p.Trp181* — Finnish) or frameshift mutations causing complete CLN3/Battenin null alleles; classic or severe JNCL phenotype; earliest visual failure; rapid cognitive decline; vacuolated lymphocytes confirmed; EM: fingerprint profiles + curvilinear bodies early; requires full CLN3 gene panel or WES to identify (not 1.02 kb deletion PCR)",
            "count": 4,
            "gene_mechanism": "Nonsense → NMD or truncated non-functional Battenin → complete CLN3 LOF → maximal storage accumulation → rapid disease progression",
            "key_variants": ["p.Trp181* (Finnish founder)", "c.1064del frameshift", "Homozygous nonsense any exon"],
        },
        {
            "class": "Splice-Site-Exon-Skipping-Variable",
            "pct": 3,
            "description": "Splice-site variants causing exon skipping or intron retention; degree of functional impact depends on splice strength; some partial splicing → residual Battenin → variable phenotype; may present as attenuated JNCL or adult-onset variant; CLN3 gene panel or WES required",
            "count": 1,
            "gene_mechanism": "Splice site mutation → exon skipping → frameshift or in-frame deletion → reduced or absent Battenin depending on exon content",
            "key_variants": ["c.381+1G>A (intron 6 donor)", "c.613+1G>T (intron 9)"],
        },
        {
            "class": "Phenocopy-CLN3-Negative",
            "pct": 2,
            "description": "CLN3-negative JNCL phenocopy — juvenile-onset visual failure + seizures + dementia with normal CLN3 gene; differential: CLN6 (variant late-infantile/juvenile — CLN6 gene), CLN7/MFSD8 (Turkish variant), CLN1/PPT1 atypical juvenile, CLN5 (Finnish variant late-infantile), CLN8 (Northern epilepsy — EPMR). MFSD8, CLN6, CLN5 gene panel next. PPT1 enzyme assay to exclude CLN1 attenuated.",
            "count": 1,
            "gene_mechanism": "Not CLN3 — alternative NCL gene (CLN1/5/6/7/8/10/14) presenting with JNCL-like phenotype",
            "key_variants": ["Normal CLN3 gene + deletion PCR → extended NCL gene panel", "PPT1/CLN1 enzyme assay to exclude attenuated CLN1"],
        },
    ]

    seizure_types = [
        {
            "type": "Generalised-Tonic-Clonic-AFTER-VISUAL-FAILURE",
            "pct": 92,
            "description": "GTCS in CLN3 emerge AFTER visual failure by 2-5 years (age 10-13 typically) — this is the KEY CLN3 DIAGNOSTIC DISTINCTION from CLN2 (where GTCS is FIRST). GTCS onset at school age or early adolescence; often well-controlled initially; progress to drug resistance later. TREATMENT: VPA 20-30 mg/kg/day + LEV adjunct. VPA IS SAFE in CLN3 — lysosomal not mitochondrial disorder. Standard VPPP counselling for females ≥12 years. Avoid CBZ/OXC (Na-channel paradox for co-existing myoclonus).",
            "eeg": "Generalised polyspike-wave; background slowing with disease progression; photic response 15-25 Hz (standard); no giant SSPS at 1-3 Hz (this is CLN2-specific, NOT CLN3)",
            "semiology": "Classic GTCS — tonic phase evolving clonic phase; eye deviation; post-ictal confusion; duration 2-5 minutes; may cluster around illness/fever in early seizure course",
            "clinical_tip": "GTCS emerging in a child/adolescent already known to have visual failure from age 5-10 years → URGENT CLN3 GENETIC TESTING. Blood film for vacuolated lymphocytes. Skin biopsy EM. Do NOT wait. Do NOT start CBZ for 'new-onset seizures in adolescent' — will worsen myoclonic component that follows.",
        },
        {
            "type": "Myoclonic-Seizures-Action-and-Stimulus-Sensitive",
            "pct": 78,
            "description": "Myoclonic seizures develop as CLN3 progresses; action-sensitive and stimulus-sensitive; cortical reflex myoclonus component; amplitude variable. DISTINCTION FROM CLN2: myoclonus in CLN3 is generally milder and less dominant than in CLN2 (where myoclonus quickly becomes debilitating); DISTINCTION FROM KCTD7: CLN3 myoclonus lacks the extreme stimulus sensitivity of KCTD7 (90%+ auditory/tactile in KCTD7 vs 45-60% in CLN3). TREATMENT: VPA backbone for myoclonus; piracetam Level C adjunct; avoid GBP/PGB.",
            "eeg": "Polyspike-wave 2.5-4 Hz; cortical SEP enlargement (N20/P25 >6 µV in some); generalised background abnormality",
            "semiology": "Bilateral symmetrical jerks; action-induced; frequency increases with disease progression; combination GTCS + myoclonus distinguishes CLN3 from pure JME (JME: photosensitivity + GTCS + myoclonus but NO cognitive decline, NO visual failure)",
            "clinical_tip": "Myoclonus emerging after GTCS in a CLN3 patient with established visual failure and cognitive decline = expected disease progression. Optimise VPA + LEV. Add piracetam for action myoclonus. Check CLB adjunct for refractory. Key differential error: CLN3 myoclonus in adolescent can mimic JME — CLN3 has visual failure + cognitive decline + vacuolated lymphocytes.",
        },
        {
            "type": "Atonic-Drop-Attacks",
            "pct": 55,
            "description": "Atonic seizures (drop attacks) develop with disease progression; high fall injury risk. CLN3 patients with visual impairment + atonic seizures have compound fall risk — helmet AND mobility aids AND visual rehabilitation essential. VPA + CLB combination effective. Falls in CLN3 are multi-factorial: atonic seizures, cerebellar ataxia, AND visual impairment → comprehensive falls assessment mandatory.",
            "eeg": "Brief electrodecrement or polyspike preceding atonia; generalised",
            "semiology": "Sudden loss of tone; head drop or whole-body fall; no warning; brief (1-3 seconds); injury rate high due to visual impairment limiting protective responses",
            "clinical_tip": "Compound fall risk in CLN3: (1) atonic seizures, (2) cerebellar ataxia, (3) visual impairment. Each factor requires individual management: CLB for atonic seizures; physiotherapy for ataxia; blind mobility training + white cane/guide dog for visual loss. All three simultaneously.",
        },
        {
            "type": "Focal-and-Absence-like-Seizures",
            "pct": 65,
            "description": "Focal seizures (often occipital origin given retinal/visual cortex involvement) and absence-like staring spells occur throughout CLN3 disease. Staring may be difficult to distinguish from disease-related inattention or cognitive regression — EEG essential. Focal occipital seizures: visual hallucinations/distortions in early disease when some vision remains; become non-focal as cortex more diffusely involved.",
            "eeg": "Spike-wave 2.5-3.5 Hz; focal occipital discharges; posterior dominant background changes",
            "semiology": "Staring + automatisms; eye blinking; visual hallucinations (early); complex focal behaviour; differential from cognitive blunting requires video-EEG",
            "clinical_tip": "Video-EEG for all suspected seizure events in CLN3 — disease-related cognitive blunting can mimic absence seizures. VPA+LEV handles GTCS+myoclonus+absence spectrum. Focal AEDs (LCM, OXC) for refractory focal seizures BUT avoid OXC if myoclonic component present.",
        },
        {
            "type": "Non-Convulsive-Status-Epilepticus-NCSE",
            "pct": 35,
            "description": "NCSE occurs in CLN3, particularly with disease progression and concurrent febrile illness. EEG mandatory for acute cognitive deterioration — NCSE may be clinically indistinguishable from disease regression. TREATMENT: IV LEV 60 mg/kg (first-line); IV midazolam; IV VPA 20-30 mg/kg. Avoid IV fosphenytoin (paradoxical myoclonic worsening risk). Late disease NCSE may require palliative approach in context of ACP.",
            "eeg": "Continuous slow spike-wave 2-2.5 Hz; generalised; minimal reactivity",
            "semiology": "Prolonged altered consciousness; blinking; drooling; fluctuating alertness; acute behavioural change in known CLN3 patient",
            "clinical_tip": "Acute deterioration in CLN3 → URGENT EEG to exclude NCSE before attributing to disease regression. IV LEV 60 mg/kg safe and effective. Document in ED alert card: 'CLN3 disease — IV LEV 60 mg/kg for SE — AVOID fosphenytoin / phenytoin'.",
        },
    ]

    triggers = [
        {
            "trigger": "Fever-Intercurrent-Illness",
            "pct": 80,
            "description": "Fever is the most potent seizure trigger in CLN3 — febrile illness can provoke GTCS, NCSE, and myoclonic clusters. Fever management mandatory: aggressive antipyretics; rescue BDZ (buccal midazolam 0.3 mg/kg / diazepam PR 0.5 mg/kg); written seizure action plan for school and carers. CLN3 patients with visual impairment may not recognise fever warning signs — carer fever monitoring essential.",
            "management": "Antipyretics (paracetamol + ibuprofen) aggressively at first fever sign; rescue BDZ prescribed for all; written fever-seizure action plan; carer education in fever monitoring for visually impaired CLN3 patients",
        },
        {
            "trigger": "Sleep-Deprivation",
            "pct": 72,
            "description": "Sleep deprivation lowers seizure threshold. CLN3-specific: visual impairment disrupts circadian rhythm (retinal light input for melatonin regulation is lost progressively) → disturbed sleep-wake cycle compounds seizure risk. Melatonin supplementation (2-10 mg nocte) is appropriate and commonly used in CLN3. Nocturnal GTCS → SUDEP risk — nocturnal monitoring essential.",
            "management": "Melatonin 2-10 mg nocte (safe, appropriate for visual impairment-associated circadian disruption); regular sleep schedule; CLB nocturnal for nocturnal GTCS; SUDEP monitoring (pulse oximeter + movement sensor)",
        },
        {
            "trigger": "Missed-AED-Dose",
            "pct": 68,
            "description": "AED omission — especially problematic in CLN3 with visual impairment affecting independent medication management. As cognitive decline progresses, carer-administered medication becomes essential. Electronic pill dispensers, blister packs, carer-administered AED. School nurse for school-hour doses.",
            "management": "Electronic medication reminder system; carer-administered blister packs; school seizure plan with AED schedule; as vision and cognition decline, transition to carer-administered medication management",
        },
        {
            "trigger": "Emotional-Stress-Anxiety",
            "pct": 65,
            "description": "Emotional stress and anxiety are frequent seizure triggers AND a CLN3 disease feature in their own right. Anxiety and OCD-like features are prominent early in CLN3 (adjustment to visual loss, disease insight, school environment) — these DIRECTLY lower seizure threshold. SSRI treatment (fluoxetine, sertraline) for anxiety/OCD features in CLN3 serves dual benefit: psychiatric symptom management AND indirect seizure threshold stabilisation.",
            "management": "SSRI (fluoxetine 10-20 mg OD or sertraline 50 mg OD) for anxiety/OCD features; CBT where cognitive capacity allows; family psychological support; school psychologist involvement; consistent routine and predictable environment",
        },
        {
            "trigger": "Photic-Stimulation",
            "pct": 55,
            "description": "Standard photic stimulation sensitivity (15-25 Hz range) — present in ~55% CLN3. NOTE: CLN3 does NOT show the pathognomonic SSPS response at 1-3 Hz that characterises CLN2. Standard IPS protocol is appropriate in CLN3. Photosensitivity typically less extreme than in CLN2. Visual stimulus triggers (screen flickering) may be less relevant as visual acuity declines.",
            "management": "Tinted spectacles (less urgent than CLN2 given progressive vision loss); flat-panel non-flickering screens in early disease; photic management less central in late CLN3 (vision severely impaired)",
        },
        {
            "trigger": "Cognitive-Overload-School-Stress",
            "pct": 58,
            "description": "Cognitive demands at school trigger seizures — CLN3 patients have progressive cognitive decline AND visual impairment at school age; cognitive overload from adapting to visual loss + declining cognitive capacity is a unique CLN3 stressor. School environment modification, reduced cognitive load, adapted curriculum, and visual impairment support reduce cognitive-stress seizure triggers.",
            "management": "Educational psychologist assessment; special educational needs (SEN) placement; adapted curriculum; visual impairment specialist teacher; reduced cognitive demands; school seizure action plan; regular review as disease progresses",
        },
        {
            "trigger": "AED-Taper-Reduction",
            "pct": 52,
            "description": "AED dose reduction triggers seizure breakthrough in CLN3 — AED taper should NEVER be attempted due to epilepsy remission expectation (CLN3 seizures are progressive, not remitting). Avoid tapering well-controlled AED without clear medical indication. If AED side effect necessitates change, overlap period essential.",
            "management": "NEVER taper AED expecting remission in CLN3 (progressive NCL epilepsy); AED change requires careful overlap taper; document 'AED taper contraindicated — CLN3 progressive epilepsy' in notes",
        },
        {
            "trigger": "Movement-Action-Voluntary",
            "pct": 45,
            "description": "Action-induced myoclonus triggered by voluntary movement — present in CLN3 but less extreme than KCTD7 or MERRF. Adaptive OT for myoclonus management (weighted utensils, stabilised surfaces). VPA + LEV reduces action myoclonus frequency.",
            "management": "OT myoclonus adaptations (weighted gloves, stabilised surfaces, adapted utensils); VPA dose optimisation; piracetam Level C for persistent action myoclonus",
        },
    ]

    treatments = [
        {
            "drug": "Valproate (VPA)",
            "level": "Level-B",
            "dose": "20-30 mg/kg/day divided BD-TID; titrate to seizure control; trough target 60-100 mg/L; extended-release preferred for tolerability and compliance (once-daily modified release aids adherence in cognitively declining CLN3)",
            "moa": "Sodium-channel + GABA-A potentiation + calcium-channel T-type block → broad-spectrum AED; effective for GTCS + myoclonus + absence spectrum in CLN3. SAFE in CLN3: lysosomal storage disorder, NOT mitochondrial — no POLG1/mitochondrial hepatotoxicity CI. Backbone AED for CLN3 as for CLN2.",
            "efficacy": "70% of CLN3 cohort on VPA; good initial GTCS + myoclonus control; maintains value as backbone despite progressive drug resistance; modified-release formulation aids adherence in cognitively declining patients",
            "monitoring": "LFT + FBC baseline then 3-monthly; VPA trough; weight; VPPP counselling MANDATORY for females ≥12 years (sodium valproate pregnancy prevention programme); hyperammonaemia monitoring if encephalopathic",
            "cln3_note": "VPA in CLN3: SAFE (lysosomal, not mitochondrial). Standard dosing. Extended-release formulation preferred for once-daily or BD dosing — improves compliance as cognitive decline limits complex AED schedules. VPPP counselling mandatory: in the context of CLN3 progressive disease, reproductive risk discussion must acknowledge CLN3 prognosis and life expectancy.",
        },
        {
            "drug": "Levetiracetam (LEV)",
            "level": "Level-B",
            "dose": "20-40 mg/kg/day divided BD; IV formulation 60 mg/kg for acute SE or NCSE; oral solution available (aids swallowing in dysphagia); renally cleared — dose-reduce if renal impairment",
            "moa": "SV2A modulation → reduced vesicle cycling → broad-spectrum; effective for GTCS + myoclonus in NCL; IV LEV 60 mg/kg = standard SE rescue in CLN3 (as in CLN2). Avoids Na-channel paradox risk.",
            "efficacy": "65% of CLN3 cohort on LEV; good adjunct to VPA for GTCS + myoclonus control; IV LEV first-line for NCSE and SE; oral solution aids late-disease medication delivery",
            "monitoring": "Behavioural side effects (irritability, aggression) — particularly relevant given CLN3 behavioural features; add low-dose CLB 5 mg if significant behavioural activation; renal function monitoring",
            "cln3_note": "IV LEV 60 mg/kg = SE rescue for CLN3. Document in ED alert, school seizure plan, anaesthesia record. Oral LEV solution preferred in late disease when swallowing coordination declines. Behavioural side effect monitoring particularly important in CLN3 where anxiety/agitation are primary disease features — differentiate drug effect from disease progression.",
        },
        {
            "drug": "Lamotrigine (LTG)",
            "level": "Level-B",
            "dose": "0.15 mg/kg/day initial (with VPA) → titrate 8+ weeks to 1-5 mg/kg/day; higher doses without VPA; VPA halves LTG clearance — MANDATORY slow titration with VPA",
            "moa": "Sodium-channel + calcium-channel block; effective for GTCS + focal seizures; also used as VPA-sparing strategy if VPA side effects dose-limiting; Note: in pure myoclonic epilepsies (PME), LTG monotherapy is CI — but in CLN3 MIXED seizure disorder, LTG is a reasonable adjunct",
            "efficacy": "45% of CLN3 cohort on LTG; 3rd-line for refractory GTCS; VPA+LTG well-tolerated; useful when VPA dose limited by side effects (tremor, weight, behavioural)",
            "monitoring": "SJS/TEN risk (mandatory slow titration); rash protocol; VPA interaction (LTG levels doubled by VPA — must use low initial dose); ophthalmology check (LTG rarely causes diplopia)",
            "cln3_note": "LTG in CLN3: acceptable as adjunct in mixed seizure disorder (GTCS + focal). Different from PME LTG CI (in pure PME, LTG monotherapy worsens myoclonus; in CLN3, myoclonus is not the dominant seizure type). Must slow-titrate with VPA co-prescription. LTG does not treat visual symptoms — ophthalmology follow-up independent of AED choice.",
        },
        {
            "drug": "Clobazam (CLB)",
            "level": "Level-B",
            "dose": "0.2-0.5 mg/kg/day (max 1 mg/kg/day) divided OD-BD; nocturnal dose for GTCS/SUDEP prevention; rescue CLB buccal for prolonged seizures",
            "moa": "GABA-A PAM (α2-preferring benzodiazepine site) → long half-life → effective add-on for GTCS + myoclonus + atonic; nocturnal dosing reduces nocturnal GTCS; less sedating than clonazepam",
            "efficacy": "50% of CLN3 cohort on CLB; effective adjunct; nocturnal CLB reduces nocturnal GTCS frequency and SUDEP risk; useful for atonic drop attack component",
            "monitoring": "Tolerance development; sedation (particularly relevant with progressive cognitive impairment); behavioural activation; LFT baseline; gradual taper if stopping",
            "cln3_note": "Nocturnal CLB in CLN3 is important for SUDEP risk reduction — nocturnal GTCS in a visually impaired patient with cognitive decline = high SUDEP risk. Bed safety additionally important: padded rails, firm mattress, position monitoring. SUDEP counselling to carers as part of ACP discussion.",
        },
        {
            "drug": "SSRI-Sertraline-Fluoxetine",
            "level": "Level-B",
            "dose": "Sertraline 25-50 mg OD (preferred — low drug-drug interaction); Fluoxetine 10-20 mg OD; start low titrate slowly (cognitive impairment sensitises to side effects); review 6-monthly",
            "moa": "Selective serotonin reuptake inhibition → antidepressant + anxiolytic + anti-OCD effect. CLN3 has prominent behavioural psychiatric features: anxiety (adjustment to visual loss, disease progression), OCD-like rituals, agitation, depression — SSRI addresses all these. UNIQUE TO CLN3 among NCLs: SSRIs are a standard component of management specifically for the anxiety/OCD phenotype.",
            "efficacy": "60% of CLN3 cohort on SSRI; significant reduction in anxiety, OCD symptoms, agitation; improved quality of life for patient and carers; no effect on seizures directly but anxiety reduction may lower seizure threshold indirectly",
            "monitoring": "Mood response; OCD symptom scoring; activation syndrome (initial agitation — start low); weight; SSRI-associated hyponatraemia risk (monitor Na if on VPA + SSRI); serotonin syndrome if combined with tramadol/triptans",
            "cln3_note": "SSRI is UNIQUE to CLN3 among NCL management — anxiety, OCD, and agitation are cardinal CLN3 disease features, not secondary complications. Sertraline preferred (lower drug interaction profile than fluoxetine in multi-AED CLN3 patients). Discuss with family: SSRI improves quality of life and carer burden even if no impact on seizures. Psychiatric review input if SSRI insufficient.",
        },
        {
            "drug": "Piracetam",
            "level": "Level-C",
            "dose": "80-150 mg/kg/day paediatric; 20-45 g/day adult; divided TID-QID; renal dose adjustment mandatory (100% renally cleared)",
            "moa": "AMPA receptor positive modulator → reduces cortical reflex myoclonus amplitude; effective for action myoclonus in NCL including CLN3; evidence in CLN2/CLN3 less robust than in ULD/EPM1 (Level A for ULD piracetam) but clinical benefit observed",
            "efficacy": "35% of CLN3 cohort on piracetam; adjunct for refractory action myoclonus; may transiently improve fine motor function; add before more sedating agents",
            "monitoring": "Renal function; adequate hydration; anti-platelet effect at high doses; behaviour (hyperkinesia can occur at high doses in cognitively impaired patients)",
            "cln3_note": "Piracetam Level C in CLN3 — less evidence than CLN2 but reasonable trial for action myoclonus. Functional gains may be limited given progressive disease. Consider early (before motor decline makes myoclonus less measurable). Renal monitoring mandatory.",
        },
        {
            "drug": "Levodopa-Carbidopa-Late-Disease-Parkinsonism",
            "level": "Level-C",
            "dose": "Levodopa/carbidopa 25/100 mg TID starting dose; titrate for motor response; standard Parkinson's dosing; consider dopamine agonist alternatives",
            "moa": "Dopaminergic replacement for progressive extrapyramidal (Parkinsonism) features in late CLN3 — bradykinesia, rigidity, tremor, masked facies. CLN3 causes basal ganglia neurodegeneration → dopaminergic deficit → Parkinsonism. L-DOPA addresses dopamine deficit. UNIQUE to CLN3 among NCLs at this level of evidence.",
            "efficacy": "55% of CLN3 cohort with Parkinsonism features receive L-DOPA; partial motor improvement; reduced rigidity; improved mobility aids; L-DOPA response can be diagnostic (confirms dopaminergic basis of motor symptoms)",
            "monitoring": "Motor examination (UPDRS simplified); rigidity assessment; dyskinesia monitoring (L-DOPA induced dyskinesia in late disease); orthostatic hypotension; hallucinations (L-DOPA psychotic side effects — may be compounded by cognitive impairment in CLN3)",
            "cln3_note": "L-DOPA/carbidopa for Parkinsonism is UNIQUE to CLN3 among NCLs in late disease. Assess motor features formally at each visit from age 15+ years. L-DOPA hallucinations monitored especially in late CLN3 with cognitive impairment — reduce dose if psychosis; consider quetiapine if L-DOPA psychosis requires antipsychotic (D2-sparing). Avoid haloperidol/high-dose risperidone (worsen Parkinsonism).",
        },
        {
            "drug": "Physiotherapy-OT-SALT-VI-Rehab-Palliative-MDT",
            "level": "Level-A",
            "dose": "Visual impairment rehabilitation (VI specialist, habilitation officer, white cane/guide dog, Braille/screen readers) — EARLIEST PRIORITY; Physiotherapy 2× weekly for ataxia + Parkinsonism + falls prevention; OT for adaptation (AAC early, blindness mobility aids, home modification); SALT for dysarthria + dysphagia; Palliative care from diagnosis",
            "moa": "MDT addresses all CLN3 disability domains: VI rehabilitation (mobility, independence, communication); motor rehabilitation (physiotherapy for ataxia + Parkinsonism + falls); communication support (AAC, dysarthria aids); behavioural/psychiatric support (SSRI + CBT + carer training); palliative care (symptom management, ACP, family support)",
            "efficacy": "Visual impairment rehabilitation: enables school integration, independence, communication — most impactful early intervention. MDT support significantly improves quality of life and reduces carer burden. Palliative care from diagnosis reduces crisis-driven care and improves ACP quality.",
            "monitoring": "VI rehabilitation progress (mobility, independence, Braille, screen readers); SARA for ataxia; UPDRS for Parkinsonism; SALT dysphagia screening; FEES when aspiration suspected; neuropsychological assessments 6-monthly; CLN3-specific quality of life scales",
            "cln3_note": "CLN3 MDT is OPHTHALMOLOGY-LED initially (visual failure before seizures). VI (visual impairment) rehabilitation habilitation officer involvement from first diagnosis of visual failure — even before seizures. White cane or guide dog assessment when ambulation with visual impairment is impaired. Braille and screen reader training at school age. MDT transitions: ophthalmology → neurology-led as seizures emerge; palliative care integrated throughout.",
        },
    ]

    contraindications = [
        {
            "drug": "Vigabatrin (VGB)",
            "severity": "ABSOLUTE",
            "reason": "ABSOLUTE CI in CLN3 — VGB causes irreversible peripheral visual field constriction (vigabatrin-associated retinopathy, VAR). CLN3 patients have PROGRESSIVE RETINAL DEGENERATION leading to complete blindness. Superimposing VGB retinopathy on CLN3 retinal disease = catastrophic combined irreversible visual field loss, potentially dramatically accelerating blindness timeline. VGB must NEVER be prescribed in CLN3. Document in all records, school health notes, ophthalmology letters.",
            "alternative": "Standard CLN3 AED backbone (VPA + LEV + CLB) without VGB. If infantile spasms ever coincide with early CLN3 diagnosis (very rare), alternative IS/spasm protocols without VGB.",
        },
        {
            "drug": "Carbamazepine-Oxcarbazepine-Phenytoin",
            "severity": "HIGH-RISK",
            "reason": "Na-channel blockers worsen myoclonic seizures (CLN3 has myoclonus in 78% of patients — Crespel 1999 principle applies). CBZ/OXC: particularly dangerous when CLN3 presents at school age and is misdiagnosed as JME/GGE — CBZ prescribed for 'first GTCS in adolescent' → acute myoclonic worsening in a child who ALSO has visual failure and cognitive decline (CLN3 is NOT GGE). PHT: additional cardiac risks in paediatric NCL.",
            "alternative": "VPA 20-30 mg/kg/day + LEV 20-40 mg/kg/day backbone — handles GTCS + myoclonus + absence spectrum without Na-channel risk. Correct misdiagnosis first: any adolescent with GTCS + visual failure + cognitive decline = CLN3 (not JME) until proven otherwise.",
        },
        {
            "drug": "GBP-Pregabalin",
            "severity": "HIGH-RISK",
            "reason": "Gabapentin/pregabalin worsens myoclonic seizures (Crespel 1999 — α2δ modulators and myoclonus). CLN3-SPECIFIC TRAP: progressive visual impairment leads to pain management contacts (neuropathic pain from retinal disease, musculoskeletal from ataxia) — non-epilepsy specialists may prescribe GBP/PGB without awareness of CLN3 myoclonic epilepsy. Cross-specialty prescribing risk is HIGH. Document CI in all specialty correspondence, GP summary, visual impairment services letters.",
            "alternative": "Pain management in CLN3 without GBP/PGB: amitriptyline low dose (neuropathic), SNRI (duloxetine), non-pharmacological pain management; palliative team involvement for late disease pain",
        },
        {
            "drug": "D2-Blocking-Antipsychotics-Haloperidol-High-Dose-Risperidone",
            "severity": "HIGH-RISK",
            "reason": "D2-blocking antipsychotics (haloperidol, high-dose risperidone, high-dose olanzapine) WORSEN extrapyramidal/Parkinsonism features that develop in late CLN3. Psychiatric symptoms in CLN3 (agitation, psychosis from L-DOPA, behavioural dysregulation) may be treated by general psychiatry teams without CLN3 awareness. D2-blockade + CLN3 basal ganglia neurodegeneration = acute Parkinsonism exacerbation and medication-induced movement disorder.",
            "alternative": "Atypical antipsychotics with low D2 affinity: quetiapine (preferred — low D2 affinity, minimal extrapyramidal risk); clozapine (effective for L-DOPA psychosis, requires CPMS monitoring); aripiprazole LOW DOSE (partial D2 agonist — some risk but less than full antagonists). Refer to neuropsychiatry/MDT rather than general psychiatry for CLN3 behavioural crises.",
        },
        {
            "drug": "AED-Taper-Remission-Expectation",
            "severity": "HIGH-RISK",
            "reason": "AED tapering expecting epilepsy remission is dangerous in CLN3 — CLN3 is a PROGRESSIVE neurodegenerative disease; seizures do NOT remit. AED taper in stable CLN3 = inevitable seizure breakthrough + fall injury risk (with visual impairment + ataxia, breakthrough GTCS + fall = high injury/SUDEP risk). Document 'AED taper NOT appropriate — CLN3 progressive epilepsy' in all letters, including discharge summaries.",
            "alternative": "Maintain effective AED regimen indefinitely. AED change only if intolerable side effects — with careful overlap tapering. NEVER taper anticipating remission.",
        },
        {
            "drug": "IV-Fosphenytoin-IV-Phenytoin-SE",
            "severity": "HIGH-RISK",
            "reason": "IV fosphenytoin/phenytoin should be avoided for CLN3 SE — (1) risk of worsening myoclonic component; (2) IV LEV 60 mg/kg is preferred safe alternative. In a visually impaired CLN3 patient presenting to ED, phenytoin administration risk is compounded by multi-drug interactions (VPA + LTG + phenytoin interactions).",
            "alternative": "IV LEV 60 mg/kg (first-line SE); IV midazolam 0.15 mg/kg (rescue BDZ); IV VPA 20-30 mg/kg (if LEV insufficient and no LTG interaction risk); IV phenobarbital as 4th line. Document in ED alert.",
        },
    ]

    monitoring = [
        {"item": "CLN3-Gene-Sequencing-Deletion-PCR-At-Diagnosis", "rationale": "CLN3 1.02 kb deletion (c.461-280_677+382del) targeted PCR at diagnosis — rapid identification of most common allele (~73% of CLN3 alleles); full CLN3 gene sequencing for remaining alleles; ACMG variant classification; family cascade testing; identify homozygous deletion (no further sequencing needed) vs. compound het (need second allele identified for accurate prognosis and genetic counselling)"},
        {"item": "Vacuolated-Lymphocytes-Blood-Film-Diagnostic", "rationale": "Peripheral blood smear for vacuolated lymphocytes (Gröbel vacuoles) — PATHOGNOMONIC for CLN3 disease; 95% sensitivity; simple rapid test; positive blood film in a child with visual failure = CLN3 until proven otherwise; must be performed by experienced haematologist (vacuoles can be subtle); confirm with EM if equivocal"},
        {"item": "Skin-Biopsy-EM-Fingerprint-Curvilinear-Storage", "rationale": "Skin biopsy electron microscopy: fingerprint profiles + curvilinear bodies confirms NCL storage; CLN3 storage pattern: fingerprint profiles present but less dominant than CLN2; curvilinear bodies also present; eccrine gland epithelium and endothelial cells; used for NCL confirmation when blood film/gene sequencing equivocal; EM does not reliably distinguish CLN3 from other NCLs — gene sequencing required for subtype"},
        {"item": "Ophthalmology-ERG-VEP-VF-6Monthly-From-Diagnosis", "rationale": "Ophthalmology every 6 months from earliest visual symptoms: fundoscopy (bull's eye maculopathy → pigmentary retinopathy); ERG amplitude (early reduction = NCL retinal disease); VEP (pattern reversal + flash); visual field testing; acuity; colour vision. CLN3 retinal degeneration is progressive → ERG flat (complete visual loss). VGB ABSOLUTE CI documented in every ophthalmology letter. Low vision aids prescribed progressively. Ophthalmology coordinates visual impairment (VI) rehabilitation referral."},
        {"item": "Neuropsychological-Assessment-6Monthly-Cognitive-Tracking", "rationale": "Formal neuropsychological testing 6-monthly: CLN3 cognitive decline trajectory (IQ decline, memory, language, executive function, adaptive behaviour); Vineland Adaptive Behaviour Scales for daily living; school-based educational assessments; Behaviour Rating Inventory of Executive Function (BRIEF); anxiety and OCD symptom scales (Multidimensional Anxiety Scale for Children, CYBOCS); CLN3-specific quality of life questionnaire. Informs SEN placement, SSRI dose adjustment, AAC prescription, school support planning."},
        {"item": "Video-EEG-LTM-Seizure-Characterisation", "rationale": "Video-EEG long-term monitoring for complete seizure characterisation — CLN3 mixed seizure disorder (GTCS + myoclonus + atonic + focal + absence); semiology differentiation essential for AED optimisation; NCSE surveillance (acute cognitive deterioration → urgent EEG); photic stimulation protocol (standard IPS 3-50 Hz — NOT the 1-3 Hz SSPS protocol of CLN2); CLN3 EEG lacks giant SSPS pathognomonic finding"},
        {"item": "Brain-MRI-3T-6Monthly-Progressive-Atrophy", "rationale": "Serial brain MRI every 6 months: cerebral cortical atrophy (starts temporal/occipital); cerebellar atrophy (correlates with ataxia); basal ganglia signal change (correlates with Parkinsonism in late disease); white matter T2/FLAIR abnormalities; thalamic atrophy; MRI progression correlates with CLN3-related functional decline; no CLN3-specific MRI biomarker for treatment monitoring (unlike CLN2 CRS tracking cerliponase response)"},
        {"item": "UPDRS-Parkinsonism-Assessment-From-Age-15", "rationale": "Unified Parkinson's Disease Rating Scale (UPDRS — motor section) from age 15 years or when extrapyramidal features first noted: bradykinesia, rigidity, tremor, postural instability, masked facies; assess annually; UPDRS score drives L-DOPA initiation decision (UPDRS motor >20 = consider L-DOPA); monitor L-DOPA response and dyskinesia; differentiate CLN3 Parkinsonism from drug-induced EPS (antipsychotic exposure history critical)"},
        {"item": "SARA-Ataxia-6Monthly", "rationale": "Scale for Assessment and Rating of Ataxia (SARA) 6-monthly — cerebellar ataxia is a major CLN3 motor disability component; SARA trajectory guides mobility aid prescription (>10 walking aid; >18 rollator; >25 wheelchair); compound mobility assessment needed: SARA + visual impairment + Parkinsonism all contribute to falls; physiotherapy goals set by SARA + UPDRS combined"},
        {"item": "FEES-Dysphagia-Annually-From-Disease-Onset", "rationale": "Flexible endoscopic evaluation of swallowing (FEES) — CLN3 dysarthria and dysphagia develop progressively with bulbar and motor involvement; annual FEES from disease onset; aspirations risk in late disease; gastrostomy (PEG) timing guided by FEES + nutrition + ACP; oral formula (ENG-style NG/PEG feed) enables VPA modified-release medication delivery when oral swallowing unsafe; SALT dysphagia management concurrent"},
        {"item": "SUDEP-Risk-Assessment-Nocturnal-Monitoring", "rationale": "SUDEP risk assessment — nocturnal GTCS in CLN3 = high SUDEP risk, compounded by visual impairment (cannot self-rescue from prone position), sleep rhythm disruption, and cognitive decline. Nocturnal monitoring: pulse oximetry + movement sensor + carer alert; bed safety (padded rails, firm mattress, no asphyxiation risk); nocturnal CLB dose; SUDEP counselling to carers and included in ACP; document in school seizure plan"},
        {"item": "Clinical-Trials-Gene-Therapy-Enrolment", "rationale": "AAV-CLN3 gene therapy trials (intrathecal or ICV delivery): multiple Phase 1/2 trials open or planned (NCT03770572 — University of Rochester; EudraCT trials); CLN3 patients should be referred to NCL clinical trial networks (Hamburg Battenin Team; University College London NCL Group; NCL Network Europe); enrolment in natural history registries (BDSRA Batten Disease Registry; NCL Resource) enables future trial eligibility; no approved disease-modifying therapy currently — trial access is the only disease-modifying option"},
        {"item": "Visual-Impairment-VI-Rehabilitation-Ongoing", "rationale": "Visual impairment (VI) habilitation — ongoing from first visual diagnosis: (1) VI habilitation officer for orientation + mobility training; (2) white cane technique; (3) guide dog assessment (when appropriate age and visual loss); (4) Braille training; (5) screen reader / assistive technology; (6) low vision aids (magnifiers, high-contrast materials); (7) school visual support assistant (QTVI — qualified teacher for visual impairment); (8) community teams (RNIB, local VI services). VI rehabilitation is the most impactful early intervention — enables school, independence, communication before and during seizure management"},
        {"item": "Palliative-Care-ACP-From-Diagnosis", "rationale": "Palliative care integration from CLN3 diagnosis — disease is FATAL (2nd-3rd decade). ACP: resuscitation preferences; ventilation; gastrostomy; place of care and death; sibling support; bereavement planning; annual ACP review. CLN3 families uniquely experience progressive visual loss → seizures → cognitive decline → motor decline → death — multi-stage grief and adjustment. Palliative care coordinates all transitions. Children's hospice for respite and end-of-life care."},
    ]

    lifecycle = [
        {
            "stage": "Prenatal-Genetic-Risk-Counselling",
            "age": "Prenatal / preconception",
            "description": "AR CLN3: 25% recurrence risk per pregnancy. Prenatal options: CVS/amniocentesis (CLN3 deletion PCR + gene sequencing), PGT-M (preimplantation genetic testing). High carrier frequency in Scandinavian populations — consider population screening if family from high-prevalence region. Cascade carrier testing for siblings and extended family. Genetic counsellor from first CLN3 diagnosis.",
        },
        {
            "stage": "Early-Visual-Failure-Ophthalmology-Led",
            "age": "4-10 years (typically 5-8 years)",
            "description": "VISUAL FAILURE FIRST — CLN3 characteristically begins with visual symptoms before neurological features. Bull's eye maculopathy → progressive retinal degeneration. Child may present to optometrist or ophthalmologist. KEY DIAGNOSTIC STEP: blood film for vacuolated lymphocytes + CLN3 deletion PCR in any child with unexplained progressive visual failure ≥4 years. SEN placement with visual impairment support. VI rehabilitation begins immediately. No seizures yet — no AED needed. Neurological monitoring initiated. Genetic counselling for family.",
        },
        {
            "stage": "Seizure-Onset-Diagnosis-Emergency",
            "age": "10-13 years",
            "description": "SEIZURE ONSET after established visual failure — this sequence (visual then seizures) is pathognomonic for CLN3. First GTCS: VPA + LEV started immediately (NOT CBZ). CLN3 genetic confirmation if not yet done (blood film + deletion PCR). School seizure action plan with visual impairment accommodations. AED management coordinated by paediatric neurologist. Family education on seizure recognition (carers may not always see seizures given visual impairment of child). Driving permanently excluded (age-appropriate counselling when relevant).",
        },
        {
            "stage": "Established-Disease-School-To-Transition",
            "age": "12-18 years",
            "description": "Progressive CLN3: seizures + cognitive decline + visual impairment simultaneously. VPA+LEV±LTG±CLB AED optimisation. SSRI for anxiety/OCD/depression. School — SEN review with QTVI (qualified teacher VI), neuropsychological input, adapted curriculum. Transition planning: adult neurology, adult VI services, genetics review. VPPP counselling from age 12 (VPA in female with progressive disease — reproductive discussion in disease-trajectory context). Driving: never licensed (CLN3 — seizures + visual impairment + cognitive decline triple exclusion).",
        },
        {
            "stage": "Young-Adult-Progressive-Motor-Cognitive-Decline",
            "age": "18-25 years",
            "description": "Adult CLN3: progressive motor deterioration (ataxia → wheelchair), Parkinsonism emergence (L-DOPA consideration), advanced cognitive impairment. Gastrostomy if dysphagia confirmed by FEES. AAC devices (eye-gaze technology as hand function declines). VPA + LEV + CLB seizure management. L-DOPA for Parkinsonism. SSRI continues for behavioural features. Adult social care planning. ACP annual review. Clinical trial monitoring (gene therapy trial referral).",
        },
        {
            "stage": "Late-Stage-Palliative-ACP",
            "age": "25+ years",
            "description": "Severe disability: vegetative or minimally conscious state. Seizure management continues (VPA+LEV+CLB). Comfort-focused approach for all interventions. Gastrostomy feeds. Postural care. Palliative care / hospice central to MDT. ACP guides: resuscitation, ventilation, SE treatment intensity. SUDEP prevention at all stages. Bereavement support for family pre and post death. Average survival with current management 25-35 years from diagnosis (range variable by genotype).",
        },
    ]

    return {
        "etiologies": etiologies,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "treatments": treatments,
        "contraindications": contraindications,
        "monitoring": monitoring,
        "lifecycle": lifecycle,
        "cohort_size": 40,
    }


def get_definitions():
    return {
        "concepts": [
            {
                "concept": "CLN3-16p12.1-Battenin-Lysosomal-Membrane-Protein-JNCL",
                "definition": "CLN3 (16p12.1) encodes Battenin (CLN3p) — a 438-aa, ~48 kDa, 7-transmembrane domain protein localised to lysosomes and late endosomes. Unlike CLN2/TPP1 (a soluble lysosomal enzyme) or CLN1/PPT1, Battenin is a STRUCTURAL MEMBRANE PROTEIN with incompletely understood function. Proposed roles: lysosomal pH regulation, arginine transport across lysosomal membrane, autophagy regulation, and SCMAS (subunit c of mitochondrial ATP synthase) degradation facilitation. CLN3 LOF → Battenin deficiency → lysosomal/late-endosomal trafficking disruption → SCMAS accumulation → autofluorescent ceroid storage (fingerprint profiles + curvilinear bodies on EM) → progressive neuronal apoptosis (cortex + basal ganglia + retina) → JNCL disease. OMIM: *607042 / #204200. Gene cloned: IBD Consortium 1995 (Nat Genet 14:154-157).",
                "standard": "IBD-Consortium-1995-NatGenet; Munroe-1997-AmJHumGenet; OMIM-*607042; ACMG-AMP-2015"
            },
            {
                "concept": "Visual-Failure-First-JNCL-Ophthalmology-Led-Diagnosis",
                "definition": "CLN3 disease (JNCL) uniquely presents with PROGRESSIVE VISUAL FAILURE as the FIRST symptom — typically 4-10 years of age — BEFORE seizures by 2-5 years. This 'visual first' presentation distinguishes CLN3 from CLN2 (seizures first) and from almost all other NCLs and PMEs. Bull's eye maculopathy is the earliest retinal finding → progressive pigmentary retinopathy → ERG amplitude reduction → visual field constriction → blindness. The ophthalmologist (not neurologist) is typically the first specialist. Key diagnostic implication: ANY child 4-10 years with progressive bilateral visual failure of unknown aetiology → BLOOD FILM (vacuolated lymphocytes) + CLN3 DELETION PCR immediately. Early diagnosis before seizures enables VI rehabilitation planning and potential future trial eligibility.",
                "standard": "Mole-2005-EurJPaediatrNeurol; NCL-Resource-2024; ILAE-2022; BDSRA-Guidelines"
            },
            {
                "concept": "Vacuolated-Lymphocytes-Blood-Film-JNCL-Pathognomonic",
                "definition": "Peripheral blood smear showing vacuolated lymphocytes (Gröbel vacuoles) is PATHOGNOMONIC for CLN3 disease. These membrane-bound cytoplasmic vacuoles in lymphocytes represent CLN3-related lysosomal storage. Present in ~95% of CLN3 patients. A simple, rapid, inexpensive test — provides the first diagnostic clue before EM or genetic testing. Must be examined by an experienced haematologist (vacuoles can be subtle, confused with artefact). Positive vacuolated lymphocytes + progressive visual failure = CLN3 until proven otherwise. UNIQUE to CLN3 among NCLs — CLN2 does NOT show vacuolated lymphocytes (CLN2 blood film normal; CLN2 diagnosis requires TPP1 enzyme assay).",
                "standard": "Specchio-2019-FrontNeurol; NCL-Resource-2024; Mole-2011-BBA"
            },
            {
                "concept": "1.02kb-CLN3-Deletion-Exon7+8-Most-Common-NCL-Mutation",
                "definition": "The 1.02 kb CLN3 deletion (c.461-280_677+382del; exon 7+8 deletion) is the single most common mutation causing any form of NCL (Batten disease) worldwide. Accounts for ~73% of all CLN3 disease alleles. Frameshift → premature stop → NMD → absent Battenin → classic JNCL phenotype. High frequency in Northern European populations: Sweden ~1:130 carrier frequency; Finland ~1:100; Germany ~1:150; UK ~1:180. Homozygous in ~47% of JNCL patients (consanguineous or founder effect). Detectable by RAPID TARGETED DELETION PCR (same-day result available) — faster than full gene sequencing. Compound heterozygous with missense/nonsense second allele in ~26% of patients. Genotype broadly predicts phenotype (homozygous deletion = classic JNCL regardless of second allele effect in compound het with null second allele).",
                "standard": "Munroe-1997-AmJHumGenet; IBD-Consortium-1995-NatGenet; NCL-Resource-2024; ACMG-AMP-2015"
            },
            {
                "concept": "No-Disease-Modifying-Therapy-CLN3-Contrast-CLN2-Cerliponase",
                "definition": "CLN3 disease has NO approved disease-modifying therapy — this is a CRITICAL communication to families and is the fundamental distinction from CLN2 (cerliponase alfa ICV, FDA 2017). CLN3 protein (Battenin) is a MEMBRANE PROTEIN, not a secreted lysosomal enzyme — enzyme replacement therapy (ICV or systemic) is not applicable as it was for CLN2/TPP1. Current investigational approaches: (1) AAV-CLN3 gene therapy (intrathecal/ICV delivery, Phase 1/2 trials); (2) Substrate reduction therapy; (3) Autophagy modulation; (4) Small molecule Battenin stabilisers. Families must be counselled explicitly and compassionately — no 'equivalent of cerliponase' exists for CLN3 currently. Refer to NCL clinical trial networks for investigational access.",
                "standard": "FDA-NA-CLN3; NCL-Resource-2024; Mole-2019-LancetNeurol; IBD-Consortium-1995-NatGenet"
            },
            {
                "concept": "Parkinsonism-Late-CLN3-L-DOPA-Unique-NCL-Feature",
                "definition": "Extrapyramidal features (Parkinsonism — bradykinesia, rigidity, resting tremor, masked facies, postural instability) emerge in late CLN3 disease typically in the 2nd-3rd decade (age 15-25 years). This reflects CLN3-related basal ganglia (particularly substantia nigra) neurodegeneration → dopaminergic deficit. CLN3 is the only common NCL where L-DOPA/carbidopa is a standard component of late-disease management. L-DOPA response partially confirms dopaminergic pathophysiology. Parkinsonism assessment (UPDRS motor) should be performed annually from age 15. L-DOPA hallucinations monitored carefully given cognitive impairment — quetiapine preferred antipsychotic if needed (low D2 affinity). D2-blocking antipsychotics (haloperidol, high-dose risperidone) are HIGH RISK — will markedly worsen Parkinsonism.",
                "standard": "Kohlschutter-1993-JNeuropathol; Mole-2005-EurJPaediatrNeurol; Parkinson-NICE-NG71"
            },
            {
                "concept": "VPA-SAFE-CLN3-Lysosomal-Contrast-MERRF-POLG-Mitochondrial",
                "definition": "Valproate (VPA) is the BACKBONE AED in CLN3 disease and is SAFE. CLN3 is a LYSOSOMAL STORAGE DISORDER (not a mitochondrial disease) — VPA mitochondrial hepatotoxicity risk (seen in MERRF/POLG/Alpers) does NOT apply to CLN3. This is clinically important because CLN3 patients may see neurology teams experienced in MERRF who might incorrectly withhold VPA. POLG1 screening is NOT mandatory before VPA in CLN3 (POLG coincidental testing is optional — there is no known CLN3-POLG1 interaction). Standard VPA monitoring applies (LFT, FBC, trough level). VPPP mandatory for females ≥12 years (sodium valproate pregnancy prevention programme — applies regardless of CLN3 prognosis context, but counselling acknowledges disease trajectory).",
                "standard": "ILAE-2022; NICE-NG217; MHRA-VPPP-2021; CPIC-POLG1-2023"
            },
            {
                "concept": "SSRI-Anxiety-OCD-CLN3-Psychiatric-Management",
                "definition": "Anxiety, OCD-like rituals, depression, and agitation are CARDINAL DISEASE FEATURES of CLN3 — not secondary complications. Mechanistic basis: adjustment reaction to progressive visual loss + disease insight + cognitive effects of CLN3 cortical neurodegeneration. SSRI (sertraline preferred — low drug-drug interaction profile; fluoxetine alternative) is STANDARD CLN3 MANAGEMENT for these features, initiated when symptoms impact quality of life (typically 10-15 years of age). Unique to CLN3 among NCLs: SSRI is a core component of multi-drug management alongside AEDs. SSRI benefits: reduced anxiety, reduced OCD rituals, reduced agitation, improved carer burden, indirect seizure threshold stabilisation (anxiety → seizure trigger). Psychiatric review if SSRI insufficient; behavioural CBT where cognitive capacity allows; carer training in behavioural management.",
                "standard": "BDSRA-CLN3-Guidelines; Mole-2005-EurJPaediatrNeurol; APA-OCD-2023"
            },
            {
                "concept": "CLN3-vs-CLN2-Critical-NCL-Differential",
                "definition": "CLN3 and CLN2 are the two most clinically important NCLs to distinguish: ONSET SEQUENCE — CLN3: Visual failure FIRST (4-10y), THEN seizures (10-13y); CLN2: GTCS FIRST (2-4y), THEN visual failure follows. SEVERITY — CLN3: milder course (fatal 2nd-3rd decade); CLN2: severe (fatal 1st decade without treatment). TREATMENT — CLN3: NO disease-modifying therapy; CLN2: Cerliponase alfa (ICV ERT — ONLY approved NCL treatment). DIAGNOSIS — CLN3: Vacuolated lymphocytes (blood film) + CLN3 deletion PCR; CLN2: TPP1 enzyme assay (DBS/leukocyte — days). EEG — CLN3: Standard photic sensitivity, NO giant SSPS at 1-3 Hz; CLN2: Giant SSPS at 1-3 Hz is PATHOGNOMONIC. PARKINSONISM — CLN3: YES (late disease, L-DOPA); CLN2: NO (pyramidal/spastic features). PROTEIN — CLN3: Battenin (membrane); CLN2: TPP1 (soluble enzyme). PSYCHIATRIC — CLN3: Anxiety/OCD prominent; CLN2: behavioural features present but less OCD-specific.",
                "standard": "NCL-Resource-2024; Mole-2019-LancetNeurol; ILAE-2022; Schulz-2018-NEJM"
            },
            {
                "concept": "VGB-ABSOLUTE-CI-CLN3-Progressive-Retinal-Degeneration",
                "definition": "Vigabatrin (VGB) is ABSOLUTELY CONTRAINDICATED in CLN3 disease. VGB causes irreversible peripheral visual field constriction (vigabatrin-associated retinopathy, VAR) through GABA accumulation-mediated retinal ganglion cell toxicity. CLN3 patients have PROGRESSIVE RETINAL DEGENERATION (universal — 100% of CLN3 patients) leading to complete blindness. Superimposing VGB-associated retinopathy on CLN3 retinal disease creates catastrophic combined visual loss — potentially dramatically accelerating the timeline to complete blindness. This ABSOLUTE CI is shared with CLN2 (same dual retinal risk mechanism). The VGB CI in CLN3 is arguably MORE impactful than in CLN2 because CLN3 patients survive longer (into 2nd-3rd decade) and their visual quality of life is more extended. VGB must be documented in ALL clinical correspondence, school health records, ophthalmology letters, anaesthesia records, and ED alert systems.",
                "standard": "NICE-NG217-VGB; MHRA-VGB-2024; Dittrich-2020-FrontNeurol; Harding-1998-Brain"
            },
            {
                "concept": "Gene-Therapy-AAV-CLN3-Trial-Access",
                "definition": "Gene therapy using adeno-associated virus (AAV) to deliver functional CLN3 cDNA is the primary disease-modifying approach under investigation for CLN3 disease. Delivery routes: intrathecal (IT) or intracerebroventricular (ICV). Clinical trials in progress: NCT03770572 (University of Rochester — IT AAV9-CLN3, Phase 1/2); European gene therapy consortia (Hamburg Battenin Team). Gene therapy rationale: single CNS-targeted gene delivery → permanent CLN3 expression → restored Battenin → reduced storage accumulation. Unlike CLN2 (where enzyme can be supplemented), CLN3/Battenin is a membrane protein requiring membrane-embedded delivery — gene therapy is more appropriate than enzyme replacement. ALL CLN3 patients should be registered in NCL networks (NCL Resource, BDSRA registry, NCL Network Europe) to maximise trial eligibility. No therapy approved yet — trial access is current best option for disease modification.",
                "standard": "NCT03770572; BDSRA-Registry; NCL-Resource-2024; Selden-2021-Gene-Therapy"
            },
            {
                "concept": "Cognitive-Decline-Behavioural-Features-CLN3-School-Management",
                "definition": "Progressive cognitive decline in CLN3 begins after visual failure and seizure onset — affects memory, executive function, processing speed, language, adaptive behaviour. Cognitive decline is compounded by visual impairment (cannot access standard educational materials). Unique CLN3 cognitive/behavioural profile: (1) OCD-like rituals and perseveration (manage with SSRI + behavioural techniques); (2) Anxiety (adjustment disorder + organic — manage with SSRI); (3) Agitation and stereotypies; (4) Relatively preserved procedural memory (long-term); (5) Rapid language decline in late disease. Educational management: SEN placement with QTVI (qualified teacher visual impairment); adapted curriculum; visual impairment support; neuropsychological input for educational planning; assistive technology (AAC, screen readers, Braille). Full SEN statement/EHCP (Education Health and Care Plan) from first diagnosis.",
                "standard": "WHO-ICF-2019; NICE-SEN-2015; NCL-Resource-2024; Mole-2005-EurJPaediatrNeurol"
            },
            {
                "concept": "Compound-Fall-Risk-CLN3-Triple-Mechanism",
                "definition": "CLN3 patients face a COMPOUND FALL RISK from three simultaneous mechanisms — unique among NCLs and PMEs: (1) Atonic seizure drop attacks (55%); (2) Cerebellar ataxia (progressive); (3) Progressive visual impairment → cannot see hazards or protective responses. Each mechanism alone warrants falls management; all three combined require comprehensive multi-disciplinary falls prevention: CLB/VPA for atonic seizures; physiotherapy for cerebellar ataxia (Frenkel exercises, balance training, proprioceptive training); VI mobility training (white cane, orientation training); environmental modification (removal of floor hazards, lighting despite visual loss for safety of carers); HELMET when atonic seizures + visual impairment combined. Falls assessment by physiotherapy AND VI habilitation officer AND neurologist jointly.",
                "standard": "WHO-ICF-2019; NICE-NG157-Falls; RNIB-VI-Mobility-Standards"
            },
            {
                "concept": "Dysphagia-Gastrostomy-Late-CLN3-MDT",
                "definition": "Progressive dysphagia develops in late CLN3 disease — motor and cortical involvement impairs swallowing coordination. Annual FEES (flexible endoscopic evaluation of swallowing) from disease onset. Aspiration signs (wet voice, coughing, chest infections, prolonged mealtimes) → urgent FEES → gastrostomy planning. PEG gastrostomy enables: nutritional adequacy, medication delivery (VPA + LEV oral solution via PEG). CLN3 PEG timing is less acute than CLN2 (CLN3 longer survival allows planned PEG; CLN2 may need urgent PEG + Ommaya coordination). Oral feeding communication: important for quality of life and family bonding — maintain oral pleasure feeding alongside PEG nutrition if safe small amounts possible.",
                "standard": "WHO-ICF-2019; NICE-NG85; RCSLT-Dysphagia-2022; NCL-Resource-2024"
            },
            {
                "concept": "SUDEP-Risk-CLN3-Nocturnal-Compound-Risk",
                "definition": "SUDEP risk in CLN3 is COMPOUNDED by three additional factors beyond standard epilepsy SUDEP risk: (1) NOCTURNAL GTCS (primary SUDEP risk — as in all epilepsies); (2) VISUAL IMPAIRMENT: CLN3 patients cannot see seizure warning signs; cannot self-rescue from prone position during seizure; (3) PROGRESSIVE COGNITIVE IMPAIRMENT: reduced ability to follow safety instructions; may not respond to alarm systems independently. CLN3 SUDEP prevention: nocturnal pulse oximetry + movement sensor + carer alert device; padded bed rails; firm mattress; supervised sleeping arrangements (carer in adjacent room with audio monitor); nocturnal CLB dose; SUDEP counselling to carers and family as part of ACP discussion. SUDEP risk management must be re-evaluated at each disease stage.",
                "standard": "Devinsky-2011-Lancet; NICE-NG217; SUDEP-Action-UK; BDSRA-CLN3-Safety"
            },
        ],
        "thresholds": [
            {"threshold": "CLN3 1.02 kb deletion confirmed (homozygous PCR) → CLN3 diagnosis (no further sequencing needed for JNCL)", "standard": "IBD-Consortium-1995 / Munroe-1997"},
            {"threshold": "Vacuolated lymphocytes on blood film → CLN3 suspected (confirm with deletion PCR + sequencing)", "standard": "NCL-Resource-2024"},
            {"threshold": "IV LEV 60 mg/kg → CLN3 SE first-line rescue dose", "standard": "Standard paediatric SE protocol"},
            {"threshold": "VPA trough 60-100 mg/L → therapeutic range CLN3", "standard": "Standard VPA pharmacokinetics"},
            {"threshold": "ERG flat → complete retinal function loss; VGB ABSOLUTE CI even in pre-ERG-flat stage", "standard": "NICE-NG217-VGB"},
            {"threshold": "UPDRS motor >20 → L-DOPA initiation consideration for Parkinsonism in CLN3", "standard": "NICE-NG71-Parkinson"},
            {"threshold": "SARA >10 → walking aid; >18 → rollator; >25 → wheelchair (CLN3 ataxia component)", "standard": "Schmitz-Hubsch-2006-Neurology"},
            {"threshold": "FEES-confirmed aspiration → gastrostomy planning urgent", "standard": "RCSLT-Dysphagia-2022"},
            {"threshold": "Anti-CLN3 antibodies (gene therapy trial context) → monitor for immune-mediated response to AAV vector", "standard": "NCT03770572-Trial-Protocol"},
            {"threshold": "Cobb angle >20° → orthopaedic referral (scoliosis in wheelchair-dependent CLN3)", "standard": "SRS-Scoliosis-Guidelines"},
            {"threshold": "SSRI dose escalation: no response after 6 weeks at starting dose → increase or switch agent", "standard": "NICE-NG222-Depression"},
            {"threshold": "L-DOPA hallucinations + UPDRS motor response → quetiapine LOW DOSE preferred (not haloperidol)", "standard": "NICE-NG71; CLN3-MDT-Consensus"},
        ],
        "standards": [
            {"standard": "IBD-Consortium-1995-NatGenet", "detail": "International Batten Disease Consortium 1995 — CLN3 gene cloning and characterisation; Nature Genetics 14:154-157"},
            {"standard": "ILAE-2022", "detail": "ILAE Classification of Epilepsies and Epilepsy Syndromes — NCL classification; JNCL / CLN3 disease"},
            {"standard": "NCL-Resource-2024", "detail": "Neuronal Ceroid Lipofuscinosis Network (NCL Resource) — diagnostic and management guidelines; CLN3 subtypes and clinical management"},
            {"standard": "NICE-NG217", "detail": "NICE Guideline NG217 — Epilepsies in children, young people, and adults; AED selection including NCL seizure management"},
            {"standard": "Mole-2019-LancetNeurol", "detail": "Mole SE et al. 2019 — Clinical challenges and future therapeutic approaches for neuronal ceroid lipofuscinosis; Lancet Neurology 18:107-116"},
            {"standard": "MHRA-VPPP-2021", "detail": "MHRA Valproate Pregnancy Prevention Programme — mandatory counselling for females ≥12 years on VPA (applies in CLN3)"},
            {"standard": "ACMG-AMP-2015", "detail": "ACMG/AMP Variant Interpretation Standards — CLN3 variant pathogenicity classification; deletion PCR + sequencing guidance"},
            {"standard": "NCT03770572", "detail": "AAV9-CLN3 gene therapy trial — University of Rochester Phase 1/2 intrathecal delivery trial (NCT03770572); first CLN3 gene therapy IND"},
            {"standard": "BDSRA-Registry", "detail": "Batten Disease Support and Research Association (BDSRA) — CLN3 patient registry; natural history data; trial eligibility database"},
            {"standard": "WHO-ICF-2019", "detail": "WHO International Classification of Functioning, Disability and Health — rehabilitation framework for CLN3 multi-domain disability"},
            {"standard": "NICE-NG71-Parkinson", "detail": "NICE Guideline NG71 — Parkinson's disease in adults; L-DOPA/carbidopa guidelines adapted for CLN3 Parkinsonism management"},
            {"standard": "Crespel-1999", "detail": "Crespel A et al. 1999 — Aggravation of seizures and induction of new seizure types by antiepileptic drugs; GBP/PGB myoclonus worsening principle"},
        ],
        "references": [
            {"ref": "IBD-Consortium-1995-NatGenet", "citation": "International Batten Disease Consortium. Isolation of a novel gene underlying Batten disease, CLN3. Cell 1995;82:949-957; also Nat Genet 1995;14:154-157."},
            {"ref": "Munroe-1997-AmJHumGenet", "citation": "Munroe PB et al. Spectrum of mutations in the Batten disease gene, CLN3, as determined by sequencing of the major deletion and direct mutation analysis. Am J Hum Genet 1997;61:310-316."},
            {"ref": "Mole-2019-LancetNeurol", "citation": "Mole SE et al. Clinical challenges and future therapeutic approaches for neuronal ceroid lipofuscinosis. Lancet Neurol 2019;18:107-116."},
            {"ref": "Specchio-2019-FrontNeurol", "citation": "Specchio N et al. International recommendations on diagnosis and therapy of neuronal ceroid lipofuscinoses. Eur J Paediatr Neurol 2019;23:1-11."},
            {"ref": "Mole-2005-EurJPaediatrNeurol", "citation": "Mole SE et al. 2005 — Clinical presentations of neuronal ceroid lipofuscinosis. Eur J Paediatr Neurol 2005;9 Suppl 1:S37-S45."},
            {"ref": "Selden-2021-GeneTherapy", "citation": "Selden NR et al. Central nervous system gene therapy for neuronal ceroid lipofuscinoses. Neurosurg Focus 2021;50:E4; NCT03770572 trial reference."},
        ],
    }
