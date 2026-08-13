"""
KCNA2 Epilepsy — Developmental and Epileptic Encephalopathy (KCNA2-DEE)
=========================================================================
41-patient cohort · KCNA2 (1p13.3) · Kv1.2 Voltage-Gated Potassium Channel α-subunit
KCNA2-DEE: a dual-phenotype channelopathy where both loss-of-function (LOF) and gain-of-function
(GOF) variants in KCNA2 cause distinct forms of DEE with different EEG signatures, treatment
responses, and precision therapy opportunities (4-aminopyridine / quinidine for GOF variants).

KCNA2 BIOLOGY: KCNA2 (Potassium Channel Voltage-Gated Shaker-Related Subfamily A Member 2,
1p13.3) encodes Kv1.2, a delayed-rectifier voltage-gated potassium channel α-subunit of the
Shaker (Kv1) family. Kv1.2 homotetramers or heterotetramers with Kv1.1/Kv1.4/Kv1.6 form
pores at axon initial segments and juxtaparanodal regions of myelinated axons, where they
repolarise action potentials and regulate firing patterns. Kv1.2 is expressed widely in
cortical pyramidal neurons, cerebellar Purkinje cells, and hippocampal interneurons.

DUAL-MECHANISM CHANNELOPATHY — the uniquely complex feature of KCNA2:
① LOSS-OF-FUNCTION (LOF) variants: Missense or protein-truncating variants that reduce or
  abolish Kv1.2 current. Mechanism: haploinsufficiency OR dominant-negative effect via
  heterotetramer assembly. Reduced K+ repolarisation → prolonged action potentials →
  neuronal hyperexcitability → focal or multifocal DEE. Phenotype: severe DEE with
  treatment-refractory focal/multifocal seizures, developmental plateau, often hypotonia.
② GAIN-OF-FUNCTION (GOF) variants: Missense variants that shift the activation curve to
  more negative membrane potentials (hyperpolarised shift), producing a "leak"-like
  persistent K+ current at resting potential. Paradoxical excitatory effect via: dominant
  negative suppression of K+ currents in interneurons → loss of inhibitory neuron firing
  → disinhibition of pyramidal cells → network hyperexcitability. Phenotype: DEE with
  prominent cerebellar ataxia + epilepsy — a distinctive KCNA2-GOF signature.

PRECISION THERAPY FOR GOF:
- 4-Aminopyridine (4-AP, dalfampridine): broad-spectrum K+ channel blocker that partially
  rescues GOF current by reducing the aberrant "leak" → documented seizure/ataxia
  improvement in KCNA2-GOF case series (Syrbe 2015, Masnada 2017). Dose: 1-10 mg/day
  (neurologist-guided), pharmacokinetically validated in clinical use for MS fatigue.
- Quinidine: Na+/K+ blocker with reported benefit in GOF KCNA2 variants in isolated cases.
- Standard AEDs: generally poor efficacy in KCNA2-GOF DEE.

LOF TREATMENT: Conventional AEDs (VPA, LEV, CLB, KD) with modest efficacy; ketogenic diet
shows responder rates ~40% in refractory KCNA2-LOF DEE.

EPIDEMIOLOGY: ~60+ patients reported in literature (Syrbe 2015 discovery cohort: 20 patients;
Masnada 2017 expanded cohort: 31 patients; ongoing KCNA2 registry). Estimated prevalence
~1/100,000. De novo in ~95% of cases.

INHERITANCE: De novo in >95% (both LOF and GOF). Autosomal dominant — not reported as
autosomal recessive, unlike some other channelopathies. Mosaicism reported in rare cases.

KEY SAFETY PEARLS:
• GOF vs LOF distinction is CRITICAL — directs precision therapy (4-AP). Functional assay
  (Xenopus oocyte voltage-clamp or HEK cell patch-clamp) is the gold standard for ambiguous
  variants. Computational prediction alone insufficient for GOF/LOF assignment.
• 4-AP RISK: can worsen seizures in LOF KCNA2 patients (further reduces K+ repolarisation) —
  NEVER use 4-AP without confirmed GOF functional data.
• Cerebellar atrophy: progressive in GOF KCNA2 variants — annual MRI follow-up mandatory.
• Somatosensory epilepsy: pericentral somatosensory seizures (arm/face twitching) are a
  characteristic KCNA2-DEE semiology — do not misclassify as benign.
"""

import random
from datetime import datetime

SEED = 9185  # dashboard 185
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ───────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": (
            "KCNA2 missense LOF variant — dominant-negative / haploinsufficiency DEE "
            "(de novo, focal/multifocal severe)"
        ),
        "n": 17, "pct": 41,
        "category": "KCNA2-missense-LOF-dominant-negative-DEE",
        "mechanism": (
            "Most prevalent class (~41%): de novo missense variants at residues critical for Kv1.2 "
            "pore formation, voltage-sensing domain (S1-S4), or channel tetramerisation interface. "
            "Mechanism: dominant-negative assembly — the mutant Kv1.2 α-subunit co-assembles with "
            "wild-type Kv1.1/Kv1.2/Kv1.6 α-subunits into heterotetrameric channels; if ≥1 "
            "subunit is non-functional, the heterotetramer current is severely reduced (>>50% "
            "loss exceeding haploinsufficiency). Key hot-spot residues: T252 (voltage sensor S2), "
            "L298 (pore helix), P405 (S6 gate), I402 (inner pore). Electrophysiology (Xenopus "
            "oocyte): >70% reduction in peak K+ current compared to WT; dominant-negative "
            "confirmed by co-injection rescue experiment. Consequence: reduced Kv1.2 repolarisation "
            "at axon initial segments → prolonged action potential duration → lowered firing "
            "threshold → neuronal hyperexcitability → focal/multifocal seizures and DEE. "
            "ACMG pathogenicity: PS2 (de novo) + PM2 (absent gnomAD) + PS3 (functional LOF assay) "
            "+ PM1 (hotspot domain) → Pathogenic in validated cases."
        ),
        "eeg_signature": (
            "Multifocal epileptiform discharges: independent spikes or sharp waves from multiple "
            "lobes — centrotemporal, frontoparietal, occipital (no single predominant focus). "
            "This multifocal pattern distinguishes KCNA2-LOF from most monofocal genetic "
            "epilepsies. Background: moderate-severe generalised slowing (1-3 Hz delta "
            "activity) reflecting diffuse encephalopathy. During sleep: enhancement of "
            "multifocal spiking; NCSZ (non-convulsive seizures) common during NREM. "
            "Absence of CSWS/SWI pattern (unlike DEPDC5, KCNT1). "
            "Tonic seizures: may show low-amplitude beta recruitment pattern. "
            "High-frequency oscillations (HFOs): γ-band (80-250 Hz) ripples co-localised "
            "with multifocal spikes on intracranial EEG — reflect axon initial segment hyperexcitability."
        ),
        "mri": (
            "Normal at onset in ~60% of KCNA2-LOF patients. Progressive findings in 40%: "
            "mild-moderate cortical atrophy (diffuse), thin corpus callosum. "
            "NO cerebellar atrophy (this is the KEY MRI differentiator from GOF: "
            "cerebellar atrophy = GOF; normal cerebellum = LOF). "
            "T2/FLAIR: occasional periventricular white matter signal changes (non-specific). "
            "No focal cortical dysplasia, no tubers, no pachygyria."
        ),
        "clinical_note": (
            "FUNCTIONAL ASSAY MANDATORY: LOF confirmation essential before considering any "
            "precision therapy. 4-AP is CONTRAINDICATED in LOF — further reduces K+ repolarisation "
            "and risks acute seizure worsening. KD (ketogenic diet) is the recommended "
            "adjunct for LOF-DEE with ≥2 AED failures: ~40% achieve ≥50% seizure reduction. "
            "Request referral to KCNA2 registry (Hecke lab, Hamburg) for variant curation."
        ),
    },
    {
        "etiology": (
            "KCNA2 missense GOF variant — inhibitory neuron disinhibition DEE + cerebellar ataxia "
            "(de novo, precision therapy eligible)"
        ),
        "n": 14, "pct": 34,
        "category": "KCNA2-missense-GOF-interneuron-disinhibition-DEE-ataxia",
        "mechanism": (
            "Second class (~34%): de novo missense variants causing gain-of-function: hyperpolarised "
            "shift of Kv1.2 activation curve (by -10 to -30 mV) or slowed inactivation → persistent "
            "'leak' K+ current at resting membrane potentials where Kv1.2 is normally closed. "
            "PARADOXICAL EXCITATION via interneuron-selective vulnerability: GABAergic interneurons "
            "(parvalbumin+ basket cells, chandelier cells) express high Kv1.2 density at their "
            "axon initial segments for precise tonic firing control. GOF 'leak' current in "
            "interneurons → K+ efflux at rest → hyperpolarisation → reduced interneuron firing "
            "→ loss of surround inhibition of pyramidal cells → network disinhibition → seizures. "
            "Additionally: GOF Kv1.2 in cerebellar Purkinje cells (highest Kv1.2 expression in "
            "CNS) → Purkinje cell output dysfunction → cerebellar ataxia + subsequent Purkinje "
            "cell death → progressive cerebellar atrophy on MRI. Key hot-spot residues: "
            "P405L (most recurrent GOF, >10 independent de novo cases), L298F, V408A, "
            "T252A (hyperpolarised activation), R294H. P405L: 'gain' of persistent current "
            "+250% compared to WT, located at S6 pore gate — removes slow inactivation. "
            "PRECISION THERAPY: 4-aminopyridine (4-AP) blocks the aberrant GOF 'leak' current "
            "→ reduces both seizures and ataxia in KCNA2-GOF (documented clinical improvement "
            "in Syrbe 2015, Masnada 2017, Semmler 2020 case reports). "
            "ACMG: PS2+PS3(GOF functional)+PM1+PM2 → Pathogenic."
        ),
        "eeg_signature": (
            "Generalised epileptiform activity predominating: irregular spike-wave, polyspike-wave "
            "bursts (2.5-4 Hz). Multifocal component also present (centrotemporal > occipital). "
            "EEG background: moderate slowing; cerebellar involvement may produce 'cerebellar "
            "tremor' artifact at 3-5 Hz blending with epileptiform activity. "
            "Sleep EEG: NREM generalised polyspike-wave enhancement (may resemble JME but "
            "with background slowing and multifocal component distinguishing it). "
            "Photosensitivity (PPR) reported in ~25% of GOF patients (photic-driven "
            "generalised polyspike-wave 3-4 Hz at IPS frequencies 6-20 Hz). "
            "Progressive EEG deterioration correlating with cerebellar atrophy progression. "
            "After 4-AP treatment: documented reduction in epileptiform discharge frequency "
            "on serial EEGs in responder cases."
        ),
        "mri": (
            "CEREBELLAR ATROPHY: hallmark of GOF phenotype — present in 80-90% of KCNA2-GOF. "
            "Progressive on serial MRI: mild vermis + hemisphere atrophy at onset → moderate "
            "pan-cerebellar atrophy by adolescence. Reflects Purkinje cell loss from persistent "
            "GOF K+ current disrupting Purkinje cell output. "
            "Cortex: usually normal or mild cerebral atrophy. No FCD. "
            "MRI COMPARISON WITH LOF: cerebellar atrophy = GOF; normal cerebellum = LOF. "
            "This is the single most reliable imaging differentiator of GOF vs LOF before "
            "functional assay results return. "
            "T2/FLAIR: progressive cerebellar T2 signal change in severe cases. "
            "Serial MRI q12M mandatory to track cerebellar progression."
        ),
        "clinical_note": (
            "4-AP TRIAL: after GOF confirmation by functional assay, consider supervised 4-AP trial. "
            "Starting dose: 1-2 mg/day (paediatric — off-label use requires ethics/consent). "
            "Titrate cautiously; seizure diary + ataxia rating scale (SARA) before and after. "
            "If responder: maintain; if no response by 12 weeks, discontinue. "
            "Register in KCNA2-GOF international registry. MRI q12M for cerebellar atrophy tracking. "
            "Counsel family: progressive cerebellar component requires multidisciplinary rehab."
        ),
    },
    {
        "etiology": (
            "KCNA2 protein-truncating variant (frameshift / nonsense) — haploinsufficiency DEE "
            "(de novo, pure LOF, severe)"
        ),
        "n": 5, "pct": 12,
        "category": "KCNA2-truncating-haploinsufficiency-DEE",
        "mechanism": (
            "Third class (~12%): frameshift (small insertion/deletion), nonsense (stop-gain), "
            "or canonical splice-site variants → premature stop codon → NMD-mediated transcript "
            "degradation → true haploinsufficiency (50% reduction in Kv1.2 protein). "
            "Unlike dominant-negative missense LOF, pure haploinsufficiency produces "
            "somewhat less severe reduction in total K+ current (50% vs >70% in dominant-negative "
            "missense) — however, clinically both subclasses present with severe DEE. "
            "Mechanism: reduced Kv1.2 at axon initial segments → partial loss of "
            "repolarisation capacity → elevated network excitability. "
            "Haploinsufficiency confirmed: Kv1.2 heterozygous knockout mice develop "
            "spontaneous seizures (Smart 1998) — genetic validation of loss-of-function mechanism. "
            "All truncating variants are GOF-negative by definition — 4-AP contraindicated. "
            "ACMG: PVS1+PS2+PM2 → Pathogenic."
        ),
        "eeg_signature": (
            "Similar to missense LOF: multifocal epileptiform discharges, generalised slowing. "
            "May be slightly less severe than dominant-negative missense on quantitative "
            "EEG metrics (fewer IED/hour). Background: moderate slowing. "
            "Sleep enhancement of spiking. NCSZ possible. "
            "Some truncating cases show predominant temporal lobe focus mimicking focal "
            "structural epilepsy — intracranial EEG (SEEG) if surgical evaluation considered."
        ),
        "mri": (
            "Normal MRI most common (70%). Mild cortical atrophy in 30%. "
            "NO cerebellar atrophy (truncating = LOF = no Purkinje cell GOF vulnerability). "
            "Confirm: absence of cerebellar atrophy on initial MRI helps confirm LOF "
            "functional class pending formal electrophysiology."
        ),
        "clinical_note": (
            "PVS1 applies — all KCNA2 truncating variants in context of DEE phenotype are "
            "Pathogenic pending review. No need for functional assay to confirm LOF — "
            "NMD transcript degradation implies pure haploinsufficiency. "
            "Do NOT trial 4-AP. Consider KD early (after 2 AED failures). "
            "Genetic counselling: de novo; recurrence risk <1% (germline mosaicism "
            "counselling if parental testing not done)."
        ),
    },
    {
        "etiology": (
            "KCNA2 missense — functional ambiguity (VUS, GOF/LOF unresolved), "
            "clinically moderate DEE"
        ),
        "n": 3, "pct": 7,
        "category": "KCNA2-missense-VUS-functional-ambiguous-moderate-DEE",
        "mechanism": (
            "Fourth class (~7%): de novo missense variants in KCNA2 at positions without "
            "prior functional characterisation — classified initially as Variant of Uncertain "
            "Significance (VUS). Clinical phenotype: DEE (moderate severity: seizures + "
            "developmental delay) consistent with KCNA2-DEE but without confirmed GOF/LOF "
            "status. Cannot safely assign 4-AP (requires confirmed GOF). "
            "Management pathway: (1) refer variant to functional assay laboratory "
            "(Xenopus oocyte voltage-clamp, HEK293 patch-clamp); (2) computational prediction "
            "(AlphaMissense, EVE, CADD >20) as interim evidence only; (3) await ClinVar "
            "reclassification from international KCNA2 curation group; (4) check KCNA2 "
            "variant database (Hecke lab / ClinVar 'KCNA2' expert panel submissions). "
            "ACMG starting: PM2 + PM1 (domain) + PS2 (de novo) + PP4 (phenotype match) → LP; "
            "upgrade to P with functional PS3 data."
        ),
        "eeg_signature": (
            "Variable: may show multifocal (LOF-type) or generalised polyspike-wave "
            "(GOF-type) or mixed pattern. EEG pattern itself may help inform "
            "functional class assignment in ambiguous variants. "
            "Cerebellar atrophy absent → favour LOF. Cerebellar atrophy present → favour GOF. "
            "Await formal functional assay before precision therapy decision."
        ),
        "mri": (
            "Variable: normal OR mild cerebellar atrophy (helps functional class assignment). "
            "Serial MRI: if cerebellar atrophy progresses over 12M → strong GOF signal → "
            "expedite functional assay referral."
        ),
        "clinical_note": (
            "Do not trial 4-AP in functionally ambiguous cases — risk of worsening if LOF. "
            "Treat seizures with broad-spectrum conventional AED (VPA/LEV/CLB) as bridge. "
            "Prioritise functional assay referral (turnaround 3-6 months at specialist centres). "
            "Annual EEG + MRI for phenotype progression monitoring."
        ),
    },
    {
        "etiology": (
            "Clinical KCNA2-negative DEE phenocopy — negative KCNA2 with KCNA2-like phenotype"
        ),
        "n": 2, "pct": 6,
        "category": "Clinical-KCNA2-negative-DEE-phenocopy",
        "mechanism": (
            "Fifth class (~6%): clinically suspected KCNA2-DEE (focal/multifocal DEE ± "
            "cerebellar ataxia) with negative KCNA2 sequencing and CNV negative. "
            "Differential diagnosis includes: (a) KCNA1 variant (episodic ataxia type 1 + "
            "epilepsy, closely related Kv1.1 channel); (b) KCNB1-DEE (Kv2.1, multifocal "
            "DEE similar phenotype); (c) GRIN2A or GRIN2B if EEG shows "
            "sleep-associated pattern; (d) SCN8A-DEE if pericentral somatosensory semiology; "
            "(e) deep intronic KCNA2 variant not captured by standard sequencing — "
            "long-read sequencing (Oxford Nanopore) may be indicated. "
            "Management: WES/WGS in unresolved DEE; treat clinically as DEE."
        ),
        "eeg_signature": (
            "Multifocal spikes + generalised slowing — clinically indistinguishable from "
            "KCNA2-positive DEE without genetic confirmation. "
            "Genetic diagnosis needed before any precision therapy consideration."
        ),
        "mri": (
            "Variable — depends on true underlying genetic aetiology. "
            "If cerebellar atrophy + negative KCNA2: evaluate KCNA1, CACNA1A, ATP1A3."
        ),
        "clinical_note": (
            "Upgrade to WGS if panel and WES negative. Check KCNA1 (Kv1.1) — "
            "KCNA1 variants cause episodic ataxia type 1 (EA1) with epilepsy, "
            "sometimes phenotypically very similar. KCNA1 mutation → acetazolamide response "
            "— important treatment implication if KCNA1 confirmed."
        ),
    },
]

# ── Seizure Types (4) ─────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal onset tonic / focal tonic-clonic seizures (pericentral somatosensory predominance)",
        "frequency_pct": 88,
        "age_window": "Onset 0-24 months (median 8 months)",
        "eeg_correlate": (
            "Focal onset at pericentral (centroparietal, Rolandic) or frontocentral regions: "
            "rhythmic theta / beta (12-20 Hz) recruiting discharge evolving to "
            "spike-wave complex; 20-60 sec duration. Multifocal: independent foci from "
            "temporal, occipital, frontal regions in 65% of patients — no single dominant focus. "
            "Interictal: persistent multifocal spikes (≥2 independent foci) between seizures; "
            "background generalised slowing (delta 1-3 Hz) proportional to encephalopathy severity. "
            "NREM sleep enhancement: spiking density increases ≥3-fold during NREM stage N2/N3 "
            "compared to wakefulness. No CSWS/ESES pattern (unlike GRIN2A/DEPDC5)."
        ),
        "clinical_tip": (
            "SOMATOSENSORY SEMIOLOGY: characteristic pericentral focus produces focal tonic "
            "posturing or clonic jerking of arm/hand/face contralateral to EEG focus. "
            "Awareness partially retained (focal aware or focal impaired awareness). "
            "Key clinical discriminator: pericentral onset + multifocal background spikes "
            "in a DEE child should prompt KCNA2 gene panel testing. "
            "LOF vs GOF distinction cannot be made from seizure semiology alone. "
            "Seizure clustering (multiple per day) in the first years: refractory to most AEDs "
            "in KCNA2-DEE; do not repeatedly escalate sodium channel blockers if seizures "
            "persist — consider KD early. Prolonged seizures (>5 min): treat with "
            "intranasal midazolam rescue (0.2-0.3 mg/kg)."
        ),
    },
    {
        "type": "Generalised tonic-clonic seizures (GTCS)",
        "frequency_pct": 71,
        "age_window": "Throughout childhood; may be first presentation in GOF",
        "eeg_correlate": (
            "Generalised onset: synchronous high-amplitude spike-wave or polyspike-wave "
            "burst preceding tonic/clonic evolution. GOF variant EEG: more generalised at "
            "onset; LOF variant EEG: secondary generalisation from focal onset. "
            "Duration 1-3 minutes. Post-ictal: generalised attenuation 1-5 minutes, "
            "then slow delta recovery. "
            "GOF: may show 3 Hz spike-wave bursts between focal onset and secondary "
            "generalisation resembling JME — but background slowing and age of onset "
            "distinguishes KCNA2-GOF from JME."
        ),
        "clinical_tip": (
            "GTCS in KCNA2-DEE: not an isolated finding but part of a broader seizure "
            "repertoire (usually alongside focal tonic/clonic and/or spasms). "
            "Do not use sodium channel blockers (CBZ/PHT) as first-line — potential "
            "worsening in multifocal DEE context. VPA is preferred first-line for GTCS "
            "component (broad-spectrum). SUDEP risk counselling: mandatory for all "
            "patients with GTCS + DEE — provide seizure safety guidance, "
            "nighttime monitoring, rescue medication training."
        ),
    },
    {
        "type": "Epileptic spasms (infantile spasms / West syndrome phase)",
        "frequency_pct": 32,
        "age_window": "Onset 3-12 months; transitions to focal + GTCS by age 2-3 years",
        "eeg_correlate": (
            "Hypsarrhythmia or modified hypsarrhythmia: high-amplitude (>300 µV) chaotic "
            "asynchronous multifocal spikes + slow waves between spasms. During spasm: "
            "electrodecrement (high-amplitude spike → voltage attenuation) — "
            "the classic electrodecrement of infantile spasms. "
            "Modified hypsarrhythmia (asymmetric or fragmented) in 40% of KCNA2-spasm "
            "cases — reflects underlying genetic/structural heterogeneity. "
            "KCNA2-DEE patients presenting as infantile spasms may respond partially to "
            "ACTH/vigabatrin (infantile spasm first-line) but often relapse or evolve "
            "to a more complex DEE pattern."
        ),
        "clinical_tip": (
            "INFANTILE SPASMS in KCNA2-DEE: treat per infantile spasms protocol first "
            "(ACTH or vigabatrin first-line; UKISS/ICISS trial protocols). "
            "Genetic testing should proceed concurrently — do not delay KCNA2 gene panel "
            "awaiting infantile spasms treatment response. "
            "If spasms remit but EEG shows residual multifocal epileptiform activity "
            "+ developmental plateau → high suspicion for KCNA2-DEE or other genetic DEE. "
            "Vigabatrin in KCNA2-LOF: watch for irreversible visual field defect "
            "(REMS monitoring required). ACTH course: 4-6 weeks, then taper."
        ),
    },
    {
        "type": "Atonic / myoclonic-atonic seizures (drop attacks)",
        "frequency_pct": 24,
        "age_window": "Ages 2-6 years (after spasm/focal phase transitions)",
        "eeg_correlate": (
            "Atonic seizures: high-amplitude generalised spike or polyspike immediately "
            "followed by slow wave → sudden postural tone loss. Duration: 0.5-2 seconds. "
            "Myoclonic-atonic: brief myoclonic jerk (spike) immediately preceding atonic "
            "drop (slow wave) — classic Doose-like pattern but in context of genetic DEE. "
            "May appear on EEG as generalised 2-4 Hz spike-wave but patient semiology is "
            "atonic (drop) rather than pure absence. "
            "Background: generalised slowing ≥50% of record, indicating severe encephalopathy "
            "when atonic seizures are prominent."
        ),
        "clinical_tip": (
            "DROP ATTACKS in KCNA2-DEE: HELMET mandatory — immediate injury prevention. "
            "Treatment: VPA + CLB combination frequently used; KD if 2 AEDs fail. "
            "DO NOT use CBZ, OXC, or PHT for drop attacks — may worsen atonic seizures. "
            "Rufinamide: Level C evidence for drop attacks in DEE (Lennox-Gastaut pattern). "
            "Corpus callosotomy: consider if KD + ≥3 AEDs fail to control drop attacks "
            "— palliative but evidence-based for atonic seizure reduction."
        ),
    },
]

# ── Seizure Triggers (8) ──────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever / febrile illness",
        "prevalence_pct": 78,
        "mechanism": (
            "Fever increases neuronal metabolism and Na+/K+-ATPase demand; also accelerates "
            "HCN channel kinetics lowering inhibitory threshold. In KCNA2-DEE, reduced Kv1.2 "
            "repolarisation capacity (LOF) or increased GOF interneuron suppression at higher "
            "temperatures further compromises seizure threshold — fever is the most penetrant "
            "KCNA2-DEE trigger and first recognisable trigger in many infants."
        ),
        "clinical_management": (
            "Written fever-seizure action plan mandatory. Aggressive fever management "
            "(paracetamol 15 mg/kg q4-6h, ibuprofen alternating). Rescue: "
            "intranasal midazolam 0.2-0.3 mg/kg, repeat once if no response after 10 min. "
            "Hospital admission if: age <18M + febrile seizure, >2 seizures in 24h, "
            "duration >5 min, or KCNA2-DEE (all febrile seizures warrant ER review). "
            "Consider prophylactic diazepam 0.3 mg/kg/day during febrile illness "
            "for high-frequency febrile responders (seizures every fever event)."
        ),
    },
    {
        "trigger": "Sleep deprivation",
        "prevalence_pct": 72,
        "mechanism": (
            "NREM sleep consolidation phase (N2/N3) normally enhances Kv1.2 surface expression "
            "at axon initial segments (membrane trafficking upregulation during slow-wave sleep). "
            "Sleep deprivation prevents this upregulation → acute reduction in Kv1.2 "
            "at critical axonal sites → increased seizure susceptibility. "
            "EEG: sleep deprivation activates latent epileptiform foci in KCNA2-DEE "
            "disproportionately — spiking rate 4-5× higher in sleep-deprived NREM vs "
            "normal NREM. Sleep architecture often abnormal in KCNA2-DEE "
            "(fragmented NREM, reduced slow-wave sleep, increased arousals)."
        ),
        "clinical_management": (
            "Strict sleep hygiene protocol: fixed bedtime/wake time, dark quiet room, "
            "no screens 1h before bed, caregiver alert to nocturnal seizures. "
            "Melatonin 0.5-3 mg at bedtime for sleep onset insomnia (common in DEE). "
            "Nocturnal seizure monitoring: camera + pulse oximetry for SUDEP risk reduction. "
            "School accommodations: protected nap time if needed; flexible start time. "
            "Sleep study (PSG) if daytime somnolence or obstructive breathing suspected."
        ),
    },
    {
        "trigger": "Sudden movement / startling (GOF-specific)",
        "prevalence_pct": 45,
        "mechanism": (
            "GOF-specific trigger: in KCNA2-GOF, Kv1.2 aberrant leak current in cerebellar "
            "Purkinje cells and basal ganglia disrupts movement-onset computation. "
            "Sudden kinesiogenic stimuli (abrupt movement, acoustic startle) trigger brief "
            "myoclonic or focal motor seizures in GOF patients — a phenotypic overlap with "
            "PRRT2-PKD but with EEG ictal correlate (unlike PKD). "
            "LOF patients: less prominent movement-trigger; fever and sleep deprivation "
            "dominate. Startle epilepsy: generalised startle response triggering a tonic "
            "seizure — documented in KCNA2-GOF in paediatric series."
        ),
        "clinical_management": (
            "GOF-specific: 4-AP trial may reduce startle-triggered myoclonus + seizures. "
            "Clonazepam 0.02-0.05 mg/kg/day for acute startle-triggered myoclonus control. "
            "Environmental: reduce sudden loud noises where possible; avoid startling patient "
            "from sleep. Inform school: unexpected startle → seizure → protocol applies. "
            "Video-EEG of startle event confirms ictal correlate (distinguishes from "
            "non-epileptic startle response)."
        ),
    },
    {
        "trigger": "Missed AED dose",
        "prevalence_pct": 68,
        "mechanism": (
            "AED discontinuity → acute AED level drop → loss of seizure threshold elevation "
            "provided by AED mechanism. In KCNA2-DEE, AED efficacy is limited "
            "(most patients remain pharmacoresistant) but even partial AED suppression is "
            "critically important — missing a dose removes this partial protection. "
            "VPA half-life ~14h; CLB active metabolite (norclobazam) half-life ~40h; "
            "shorter half-life AEDs (LEV, LTG) produce faster threshold drop on missed dose."
        ),
        "clinical_management": (
            "Extended-release formulations where available (VPA XR, LEV XR) — reduce "
            "peak-trough fluctuation. Alarm-based adherence (MMAS-8 assessment + phone reminder). "
            "Written sick-day protocol: if oral AED not tolerated (vomiting) → "
            "rectal diazepam/midazolam rescue + hospital review within 2h. "
            "Simplify regimen: ≤2 AEDs if possible (polypharmacy ↑ missed dose risk). "
            "Carer training: recognise pre-ictal signs and have rescue ready."
        ),
    },
    {
        "trigger": "Photic stimulation / photosensitivity (GOF predominantly)",
        "prevalence_pct": 28,
        "mechanism": (
            "Photosensitivity (PPR — photoparoxysmal response) reported in ~25-30% of "
            "KCNA2-GOF patients (vs ~5% in LOF). Mechanism: GOF K+ leak in visual cortex "
            "inhibitory interneurons → reduced visual cortex GABAergic tone → enhanced "
            "visual cortex entrainment to rhythmic photic stimulation → generalised polyspike-wave "
            "discharge at IPS rates 6-20 Hz. "
            "Standard IPS protocol at 3-15 Hz most provocative. "
            "Avoid in GOF: strobe lights, video games (if PPR positive), flickering sunlight."
        ),
        "clinical_management": (
            "Formal IPS protocol at diagnosis: 3-60 Hz testing with eye-open/closed conditions. "
            "If PPR positive: polarised/tinted glasses outdoors; game/screen restrictions; "
            "no strobe/dance lights. VPA suppresses PPR in ~60% of patients. "
            "If PPR dominant pattern: confirm GOF status — 4-AP may reduce PPR in GOF "
            "(case-level evidence). Inform school re: projector/flicker light restrictions."
        ),
    },
    {
        "trigger": "Stress / emotional upset",
        "prevalence_pct": 55,
        "mechanism": (
            "Acute psychological stress activates HPA axis → cortisol surge → "
            "reduced GABA-A receptor sensitivity → lower seizure threshold. "
            "Noradrenaline (stress neurotransmitter) also modulates Kv1.2 trafficking: "
            "β-adrenergic receptor activation → PKA phosphorylation of Kv1.2 → "
            "reduced surface expression → acute seizure threshold lowering. "
            "Particularly relevant in KCNA2-DEE children experiencing school/social stress "
            "or medical procedure anxiety."
        ),
        "clinical_management": (
            "Psychology referral for adaptive coping strategies (CBT-adapted for "
            "developmental level). Pre-procedure benzodiazepine for high-anxiety "
            "medical events (blood draw, MRI sedation). Caregiver stress also impacts "
            "child — family support services and carer respite. "
            "Routine: maintain predictable schedules to reduce unpredictable stressors. "
            "School: SEP (special educational plan) with emotional support provisions."
        ),
    },
    {
        "trigger": "Hyperventilation (HV) — GOF ataxia component",
        "prevalence_pct": 22,
        "mechanism": (
            "Hyperventilation → hypocapnia → cerebral vasoconstriction → cortical hypoperfusion "
            "→ reduced seizure threshold. Additionally: alkalosis shifts ion channel equilibria — "
            "Kv1.2 activation curve is sensitive to extracellular pH; alkalosis (pH ↑) "
            "further enhances GOF Kv1.2 leak current in interneurons → greater disinhibition. "
            "HV also exacerbates cerebellar ataxia component in GOF patients "
            "(hypocapnia reduces cerebellar perfusion transiently)."
        ),
        "clinical_management": (
            "Avoid prolonged HV (exercise should not be restricted but paced activity "
            "is preferable to anaerobic bursts). Breathing technique training: "
            "slow diaphragmatic breathing for anxiety-related HV. "
            "Standard HV activation during routine EEG: may elicit subclinical generalised "
            "spike-wave in KCNA2-GOF — use to characterise epileptiform pattern. "
            "Do not restrict swimming/exercise — benefits outweigh trigger risk."
        ),
    },
    {
        "trigger": "AED iatrogenic worsening (sodium channel blockers in LOF/GOF)",
        "prevalence_pct": 38,
        "mechanism": (
            "Sodium channel blockers (CBZ, OXC, LTG, PHT, LAC): reduce Na+ current → "
            "reduce action potential firing. In KCNA2-DEE context: these agents may "
            "paradoxically worsen atonic/myoclonic seizure types by disinhibiting inhibitory "
            "interneuron-mediated networks (same paradoxical effect seen in SCN1A/Dravet). "
            "LTG + GTCS: LTG may be safe for GTCS but risky for myoclonic/atonic component. "
            "Drug-drug interactions: CBZ induces CYP3A4 → reduces VPA and CLB levels "
            "→ loss of VPA-mediated seizure control in polypharmacy."
        ),
        "clinical_management": (
            "AVOID CBZ/OXC/PHT in KCNA2-DEE — especially if myoclonic or atonic "
            "component present. LTG: use with caution and monitor closely. "
            "Preferred AEDs: VPA (first-line broad-spectrum), LEV, CLB, rufinamide "
            "(for Lennox-like pattern), KD. "
            "Drug interaction check: CYP3A4 interactions if polypharmacy. "
            "Review AED list at each visit: remove ineffective AEDs systematically."
        ),
    },
]

# ── Treatments (8) ────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproic Acid (VPA)",
        "evidence_level": "Level B — first-line broad-spectrum for KCNA2-DEE",
        "dose": "20-60 mg/kg/day divided BD-TID; target TDM 50-100 mg/L",
        "moa": (
            "Multi-mechanism: enhances GABA-A receptor function + inhibits GABA transaminase "
            "(↑ GABA availability) + reduces T-type Ca2+ current (spike-wave suppression) "
            "+ modulates Na+ channel inactivation. Broad-spectrum activity relevant to "
            "KCNA2-DEE multiple seizure types (GTCS, focal, atonic, myoclonic, spasms)."
        ),
        "efficacy": (
            "GTCS suppression: 50-70% ≥50% reduction in genetic DEE. Atonic seizures: "
            "40-60%. Focal tonic: moderate. Infantile spasms: 20-30% (not first-line). "
            "KCNA2-specific: no RCT data; LOF series (Masnada 2017): most common AED used "
            "with partial response. Combination with CLB improves efficacy in "
            "drug-resistant KCNA2-DEE."
        ),
        "safety": (
            "Weight gain (30%), tremor (dose-dependent), alopecia, thrombocytopenia. "
            "CRITICAL: POLG1 mutation screening MANDATORY before VPA (risk of "
            "VPA-induced hepatotoxicity + Alpers disease in POLG1 biallelic carriers). "
            "Teratogen: PREVENT programme consent for females of childbearing age "
            "(UK MHRA 2024). Neural tube defect risk → folic acid 5 mg/day. "
            "Hyperammonaemia: check serum ammonia if drowsiness/cognitive change. "
            "Hepatotoxicity: LFT q3M for first year, then q6M."
        ),
        "monitoring": "TDM q3M; LFT + FBC + ammonia q3M; weight q3M; POLG1 pre-VPA; female: PREVENT consent",
    },
    {
        "drug": "Clobazam (CLB)",
        "evidence_level": "Level B — adjunct for focal/GTCS/drop attacks in KCNA2-DEE",
        "dose": "0.1-0.3 mg/kg/day divided BD; max 1 mg/kg/day",
        "moa": (
            "Positive allosteric modulator of GABA-A receptor (benzodiazepine site) → "
            "enhanced Cl- conductance → hyperpolarisation. 1,5-benzodiazepine structure "
            "with lower sedation potential than 1,4-benzodiazepines (diazepam, clonazepam). "
            "Active metabolite norclobazam (t½ ~40h) provides sustained efficacy."
        ),
        "efficacy": (
            "Lennox-Gastaut syndrome: Level A for drop attacks (FDA-approved 2011). "
            "KCNA2-DEE extrapolated: adjunct for focal + GTCS + atonic component. "
            "Norclobazam TDM (target 50-300 ng/mL) guides dose adjustment. "
            "Tolerance: may develop over 6-12M — drug holiday (1 week taper off, "
            "2 weeks off, re-start) partially restores efficacy."
        ),
        "safety": (
            "Sedation, drooling, irritability, behavioural disinhibition (especially in "
            "ID + ASD patients). Rare: Stevens-Johnson syndrome — check HLA-B*15:02 "
            "if SE Asian ancestry before prescribing. Dependence if abrupt discontinuation "
            "→ must taper slowly (5-10% per week). "
            "Paradoxical excitation in 5-10% of DEE patients."
        ),
        "monitoring": "Norclobazam TDM q3M (target 50-300 ng/mL); sedation scale; behaviour diary",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence_level": "Level C — adjunct for focal/GTCS in KCNA2-DEE",
        "dose": "20-60 mg/kg/day divided BD; max 3000 mg/day",
        "moa": (
            "Binds SV2A (synaptic vesicle protein 2A) → reduces glutamate release "
            "at active synapses. Does not modulate GABA-A or Na+ channels directly. "
            "Mechanism independent of Kv1.2 — mechanistically neutral for KCNA2 GOF/LOF."
        ),
        "efficacy": (
            "Focal seizures: 40-50% responder rate in genetic focal DEE. "
            "GTCS: moderate efficacy. Myoclonic: variable — some worsening reported "
            "(less than LTG, but monitor). "
            "KCNA2-specific: used in majority of patients in Masnada 2017 series; "
            "provides partial seizure control as component of polytherapy."
        ),
        "safety": (
            "BEHAVIOURAL ADVERSE EFFECTS: irritability, aggression, emotional lability "
            "in 10-20% — especially in DEE with ASD/ID comorbidity (KCNA2-DEE). "
            "B6 (pyridoxine) supplementation 50 mg/day may reduce LEV behavioural effects. "
            "Somnolence. Generally well-tolerated. No organ toxicity. "
            "Renal dose adjustment if eGFR <60."
        ),
        "monitoring": "Behaviour diary; neurodevelopmental assessment q6M; renal function annually",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "evidence_level": "Level B — for drug-resistant KCNA2-DEE (LOF and GOF) after ≥2 AED failures",
        "dose": (
            "Classic KD: 4:1 fat:carbohydrate+protein ratio; carbohydrates <10g/day; "
            "protein 1-1.5 g/kg/day; individualised by metabolic dietitian"
        ),
        "moa": (
            "Ketone bodies (β-hydroxybutyrate, acetoacetate) → ATP-sensitive K+ channel "
            "activation → membrane hyperpolarisation → reduced neuronal excitability. "
            "Additionally: mTOR pathway suppression; GABA synthesis enhancement "
            "(acetyl-CoA → citrate → GABA precursor); reduced ROS. "
            "In KCNA2-LOF: ATP-K channel activation partially compensates for reduced Kv1.2 "
            "repolarisation — plausible mechanistic rationale for KD benefit."
        ),
        "efficacy": (
            "Non-specific DEE: 50-60% achieve ≥50% seizure reduction (Neal 2008 RCT). "
            "KCNA2-DEE (LOF series): ~40% responder rate in observational data. "
            "Drop attacks most KD-responsive seizure type. "
            "GOF: KD evidence less studied — theoretically less mechanistically targeted "
            "but clinical responders reported."
        ),
        "safety": (
            "Dyslipidaemia (LDL ↑, HDL ↓ or ↑ depending on fat source). "
            "Growth faltering if protein inadequate. Renal stones (urine pH acidification): "
            "hydration + potassium citrate. Selenium/zinc/carnitine supplementation. "
            "Constipation: fibre supplementation. Bone mineral density: calcium/vitamin D "
            "supplementation (DXA at 2 years). Metabolic screen before initiating: "
            "exclude mitochondrial disorder, pyruvate carboxylase deficiency, porphyria."
        ),
        "monitoring": (
            "Beta-OH-butyrate POCT target 2-4 mmol/L; lipids q3M; "
            "growth/height q3M; renal function + urine Ca/Cr ratio q3M; "
            "bone density DXA at 2y; neuropsychology assessment q12M"
        ),
    },
    {
        "drug": "4-Aminopyridine (4-AP / dalfampridine) — GOF KCNA2 ONLY",
        "evidence_level": "Level C — precision therapy for confirmed KCNA2-GOF (case series evidence, Syrbe 2015, Masnada 2017, Semmler 2020)",
        "dose": (
            "Adults (MS indication): 10 mg BD standard dalfampridine. "
            "Paediatric KCNA2-GOF: 1-10 mg/day (off-label, neurologist-guided, "
            "start 1 mg/day and titrate monthly under seizure diary + SARA scale monitoring)"
        ),
        "moa": (
            "4-AP: broad-spectrum voltage-gated K+ channel blocker (Kv1 family including Kv1.2). "
            "In KCNA2-GOF: blocks the aberrant hyperpolarisation-shifted 'leak' current "
            "produced by GOF variants → partially restores normal Kv1.2 gating → "
            "reduces excess K+ efflux in interneurons → partially restores GABAergic tone → "
            "reduces disinhibition-driven seizures and cerebellar ataxia. "
            "Mechanism is GOF-selective: in LOF patients, 4-AP would FURTHER reduce "
            "the already-deficient K+ repolarisation → dangerous worsening."
        ),
        "efficacy": (
            "Syrbe 2015 (Nat Genet discovery paper): 2/3 GOF patients improved on 4-AP "
            "(seizure + ataxia). Masnada 2017 (BRAIN 31-patient cohort): 4-AP responders "
            "in GOF subgroup — ataxia scale improvement + seizure frequency reduction. "
            "Semmler 2020 (EJPN case series): P405L (most recurrent GOF hotspot) responders "
            "to 4-AP, including ataxia reversal. Not all GOF patients respond. "
            "Trial duration: 12 weeks minimum before declaring non-responder."
        ),
        "safety": (
            "SEIZURE RISK: 4-AP lowers seizure threshold in populations with epilepsy at "
            "supra-therapeutic doses. Use lowest effective dose, seizure diary mandatory. "
            "ABSOLUTE CONTRAINDICATION IN LOF: further reduces Kv1.2-mediated repolarisation. "
            "CNS: dizziness, insomnia, headache, tremor. Renal: 4-AP renally excreted — "
            "reduce dose in renal impairment (eGFR <30: avoid). "
            "QTc prolongation: baseline and 4-weekly ECG during titration. "
            "Ethics/consent: off-label paediatric use — document functional assay "
            "GOF confirmation before prescribing; IRB/ethics approval for formal trial."
        ),
        "monitoring": (
            "Seizure diary q2W; SARA (Scale for Assessment and Rating of Ataxia) q4W; "
            "EEG at 4W and 12W (document epileptiform change); ECG q4W during titration; "
            "renal function q3M; GOF confirmation document on file before prescribing"
        ),
    },
    {
        "drug": "Vigabatrin (VGB) — infantile spasms phase ONLY",
        "evidence_level": "Level B — for KCNA2-DEE presenting as infantile spasms (VGB standard IS protocol)",
        "dose": "100-150 mg/kg/day divided BD (infantile spasms); max 3g/day",
        "moa": (
            "Irreversible GABA transaminase inhibitor → prevents GABA catabolism → "
            "accumulation of synaptic GABA → enhanced GABAergic tone → infantile spasm suppression. "
            "Evidence strongest for TSC-associated infantile spasms (UKISS 2008). "
            "Extrapolated to genetic IS including KCNA2-DEE presenting as IS."
        ),
        "efficacy": (
            "Infantile spasms: 36-90% hypsarrhythmia cessation depending on aetiology "
            "(TSC: highest; cryptogenic: lower; genetic DEE: intermediate). "
            "KCNA2-DEE infantile spasms: may partially respond; relapse after "
            "VGB cessation common → evolution to focal/GTCS/atonic pattern."
        ),
        "safety": (
            "IRREVERSIBLE VISUAL FIELD CONSTRICTION: occurs in 30-40% of patients on "
            "prolonged VGB — REMS programme mandatory (baseline VEP + visual field, "
            "q3M monitoring). Reduce use to minimal duration (IS treatment course: 6-12M max). "
            "Intramyelinic oedema: MRI DWI signal in BG/thalami/brainstem in infants "
            "(usually reversible on discontinuation). Sedation, hypotonia."
        ),
        "monitoring": (
            "VEP + visual field testing before start and q3M (REMS); MRI DWI at 3M; "
            "EEG q4W during IS treatment; growth/weight q4W"
        ),
    },
    {
        "drug": "Rufinamide",
        "evidence_level": "Level C — for Lennox-Gastaut-like atonic/drop attack pattern in KCNA2-DEE",
        "dose": "10-45 mg/kg/day divided BD; max 3200 mg/day",
        "moa": (
            "Prolongs Na+ channel inactivation (prolongation of inactive state) → "
            "reduced high-frequency repetitive firing. Mechanism orthogonal to Kv1.2 — "
            "does not directly worsen KCNA2-DEE mechanism. "
            "Evidence-based for drop attacks in Lennox-Gastaut syndrome (FDA 2008)."
        ),
        "efficacy": (
            "LGS drop attacks: ~42% ≥50% reduction (NEJM Glauser 2008). "
            "KCNA2-DEE with atonic component: Level C extrapolation — "
            "used in refractory KCNA2-DEE with significant drop attack burden "
            "as part of rational polytherapy with VPA + CLB."
        ),
        "safety": (
            "QTc shortening (do not use with antiarrhythmics that shorten QTc). "
            "Somnolence, dizziness, nausea. Hypersensitivity rash (rare). "
            "VPA interaction: VPA increases rufinamide plasma levels ~25% — reduce "
            "rufinamide dose or monitor. Titrate slowly over 2 weeks."
        ),
        "monitoring": "ECG before start (QTc baseline); LFT + FBC q3M; seizure/fall diary q2W",
    },
    {
        "drug": "Corpus Callosotomy (CC) — surgical palliative for refractory drop attacks",
        "evidence_level": "Level B — palliative for drug-resistant atonic seizures in KCNA2-DEE",
        "dose": "Anterior 2/3 CC; posterior extension if incomplete response",
        "moa": (
            "Disconnects interhemispheric propagation of epileptic discharge — "
            "prevents synchronised bilateral cortical activation required for "
            "atonic seizures (generalised tone loss). Does not cure the underlying "
            "KCNA2 channelopathy but eliminates the propagation pathway for drop attacks. "
            "No curative intent — strictly palliative for seizure safety."
        ),
        "efficacy": (
            "Atonic seizure reduction: 50-90% reduction in generalised atonic seizures "
            "post-CC in Lennox-Gastaut-like DEE. KCNA2-specific: no dedicated series — "
            "extrapolated from LGS CC data. Indication: ≥3 AEDs + KD failure with "
            "significant daily drop attack burden and injury risk. "
            "Not suitable for all: neuropsychological evaluation prerequisite."
        ),
        "safety": (
            "Disconnection syndrome: transient mutism, left-hand apraxia. "
            "Surgical risk: bleeding, infection, venous injury (<1% major). "
            "Partial CC: lower disconnection syndrome risk. "
            "Does not cure seizures — focal seizures often unchanged or increased. "
            "Quality-of-life-weighted decision: injury reduction vs surgical risk."
        ),
        "monitoring": (
            "Pre-op: VEEG + MRI + neuropsychology + family consent. "
            "Post-op: EEG at 1M, 3M, 6M; fall diary; neuropsychology at 6M"
        ),
    },
]

# ── Contraindications (4) ─────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug_or_action": "Sodium Channel Blockers (CBZ, OXC, PHT, LTG) in KCNA2-DEE with myoclonic/atonic component",
        "risk_level": "HIGH",
        "reason": (
            "Sodium channel blockers may paradoxically worsen myoclonic and atonic seizures "
            "in genetic DEE contexts — same mechanism as SCN1A/Dravet syndrome aggravation. "
            "In KCNA2-DEE: if myoclonic-atonic component present, CBZ/OXC/PHT carry "
            "documented worsening risk. LTG is safer for pure GTCS but monitor closely "
            "for myoclonic worsening. Phenytoin: additional cerebellar toxicity risk "
            "in KCNA2-GOF patients with existing cerebellar atrophy."
        ),
        "alternative": "VPA + CLB + LEV as preferred AED combination; avoid if myoclonic/atonic present",
    },
    {
        "drug_or_action": "4-Aminopyridine (4-AP) in KCNA2-LOF or uncharacterised KCNA2 variant",
        "risk_level": "ABSOLUTE CONTRAINDICATION",
        "reason": (
            "4-AP blocks Kv1.2 and related K+ channels — in KCNA2-LOF patients who already "
            "have critically reduced Kv1.2 repolarisation, 4-AP further impairs K+ conductance "
            "→ acute worsening of neuronal hyperexcitability → seizure escalation, "
            "potential status epilepticus. 4-AP is ONLY safe in confirmed GOF variants "
            "where the therapeutic target is the aberrant GOF leak current (not the WT current). "
            "GOF functional assay MUST be documented before 4-AP prescription."
        ),
        "alternative": "KD + VPA + CLB for LOF-DEE; never trial 4-AP without GOF confirmation",
    },
    {
        "drug_or_action": "VPA without POLG1 exclusion",
        "risk_level": "HIGH",
        "reason": (
            "VPA-induced Alpers-Huttenlocher syndrome: rare but fatal — occurs in patients "
            "with biallelic POLG1 mutations (mtDNA polymerase). Presentation: rapid-onset "
            "liver failure, encephalopathy, and death within weeks of VPA initiation. "
            "POLG1 phenotype overlap: progressive encephalopathy + epilepsy in POLG1 "
            "can mimic KCNA2-DEE. POLG1 testing MANDATORY before VPA in any child "
            "with progressive encephalopathy, especially if hepatic enzyme elevation, "
            "ataxia, or family history of mitochondrial disease."
        ),
        "alternative": "Confirm POLG1 biallelic variant absence before VPA; if POLG1 positive: use LEV/CLB/KD instead",
    },
    {
        "drug_or_action": "Withholding 4-AP trial in confirmed KCNA2-GOF without documented clinical reason",
        "risk_level": "HIGH (risk of suboptimal care)",
        "reason": (
            "KCNA2-GOF patients are rare but precision therapy (4-AP) is the only "
            "mechanistically targeted intervention available. Failure to offer 4-AP "
            "to a confirmed GOF patient (after functional assay documentation) represents "
            "a missed precision medicine opportunity — the patient may continue with "
            "refractory DEE + progressive cerebellar atrophy when 4-AP could substantially "
            "reduce both seizures and ataxia. International KCNA2-GOF registry mandates "
            "4-AP trial documentation in all GOF patients without medical contraindication."
        ),
        "alternative": "Document GOF functional assay result → initiate 4-AP trial with monitoring protocol → KCNA2 registry enrolment",
    },
]

# ── Monitoring Items (8) ──────────────────────────────────────────────────────
MONITORING = [
    {
        "item": "VPA TDM (target 50-100 mg/L) + LFT + FBC + ammonia",
        "frequency": "Every 3 months",
        "rationale": (
            "VPA TDM guides dose: sub-therapeutic (<50 mg/L) → inadequate seizure control; "
            "supra-therapeutic (>100 mg/L) → toxicity (tremor, thrombocytopenia, hepatotoxicity). "
            "LFT: VPA hepatotoxicity screening; FBC: thrombocytopenia (VPA-induced platelet "
            "aggregation defect); ammonia: VPA-induced hyperammonaemia — if drowsiness "
            "or cognitive regression → check ammonia + consider L-carnitine supplementation."
        ),
    },
    {
        "item": "4-AP seizure diary + SARA (Scale for Assessment and Rating of Ataxia) — GOF only",
        "frequency": "Every 4 weeks during 4-AP titration; every 3 months once stable",
        "rationale": (
            "4-AP efficacy monitoring in GOF patients: seizure diary quantifies seizure "
            "frequency change. SARA (0-40 scale, higher = more ataxia) tracks cerebellar "
            "ataxia response — 4-AP expected to reduce both seizures and SARA score in "
            "responders. ECG at 4W and 12W: QTc monitoring during 4-AP titration. "
            "Document GOF functional assay result in medical record before every 4-AP prescription."
        ),
    },
    {
        "item": "Brain MRI (with volumetric cerebellar assessment)",
        "frequency": "At diagnosis, then annually (GOF: every 6M if progressive atrophy)",
        "rationale": (
            "Serial MRI tracks cerebellar atrophy progression in GOF patients: "
            "volumetric cerebellar assessment (vermis volume, hemisphere volume) — "
            "validated tools include FSL's FAST/FIRST or FreeSurfer parcellation. "
            "Cortical atrophy in LOF: tracks global encephalopathy progression. "
            "KCNA2-DEE without cerebellar atrophy at 2-year mark → reassign towards LOF class. "
            "Cerebellar atrophy progression rate correlates with clinical deterioration in GOF."
        ),
    },
    {
        "item": "Neuropsychological assessment (developmental quotient / IQ + adaptive behaviour)",
        "frequency": "At diagnosis, then every 12 months",
        "rationale": (
            "Developmental trajectory monitoring: DQ (Griffiths/Bayley-III) in preschool, "
            "IQ (WISC-V) + Vineland-3 adaptive behaviour in school age. "
            "KCNA2-DEE: developmental plateau is the rule — trajectory tracks "
            "encephalopathy severity and AED cognitive burden. "
            "If rapid developmental regression: check seizure burden (NCSZ), AED toxicity, "
            "thyroid, and repeat functional assay if variant functional class uncertain. "
            "Guides educational support intensity and therapy frequency (SLP/OT/PT)."
        ),
    },
    {
        "item": "EEG (routine awake + sleep + HV + IPS)",
        "frequency": "At diagnosis; 3-monthly during first year; 6-monthly thereafter; after any AED change",
        "rationale": (
            "EEG tracks multifocal spike burden, background slowing severity, "
            "sleep-wake distribution of IEDs, and photosensitivity (PPR on IPS). "
            "Quantitative IED/hour measure: declining → AED efficacy. "
            "Increasing → AED resistance / disease progression. "
            "Prolonged EEG (48h ambulatory): detect NCSZ episodes that may explain "
            "developmental regression not accounted for by clinical seizures. "
            "After 4-AP in GOF: EEG at 4W and 12W to document epileptiform response."
        ),
    },
    {
        "item": "POLG1 biallelic mutation exclusion before VPA",
        "frequency": "Once, before VPA initiation",
        "rationale": (
            "VPA-induced Alpers-Huttenlocher syndrome in POLG1 patients is fatal — "
            "LFT elevation within weeks of VPA onset → fulminant hepatic failure → death. "
            "POLG1 sequencing (next-generation sequencing of POLG gene) is straightforward "
            "and must be completed or pending-negative before VPA prescribing in any child "
            "with encephalopathy + progressive epilepsy. Report POLG1 result in clinical "
            "notes and prescribing record. If POLG1 biallelic confirmed: NEVER prescribe VPA."
        ),
    },
    {
        "item": "KCNA2 functional assay result documentation and GOF/LOF classification",
        "frequency": "Once, at diagnosis confirmation; re-review if new clinical data",
        "rationale": (
            "GOF vs LOF classification directs precision therapy: only GOF → 4-AP eligible. "
            "Functional assay (Xenopus oocyte voltage-clamp or HEK cell patch-clamp) "
            "is the gold standard — laboratories: Hecke group (Hamburg), Lerche group "
            "(Tübingen), Bhatt lab (Cincinnati). Turnaround 3-6 months. "
            "Computational prediction (AlphaMissense, CADD, REVEL) provides interim support "
            "but is not sufficient for 4-AP prescribing decision alone. "
            "Document functional assay accession/report reference in KCNA2 registry."
        ),
    },
    {
        "item": "SUDEP risk assessment + caregiver safety training",
        "frequency": "Annually; at any change in seizure pattern",
        "rationale": (
            "KCNA2-DEE with refractory GTCS carries elevated SUDEP risk (estimated "
            "1:100-1:1000 patient-years in severe DEE). SUDEP risk factors: nocturnal "
            "unwitnessed GTCS, prone sleeping, AED sub-therapeutic levels, no rescue "
            "medication. Mitigation: nighttime monitoring (camera/apnoea monitor), "
            "supine sleeping position, rescue midazolam training, AED optimisation, "
            "no unsupervised bathing/swimming. SUDEP Action plan documentation + "
            "caregiver training annually per NICE NG217 standard."
        ),
    },
]

# ── Lifecycle Windows (6) ─────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "phase": "Neonatal / Early Infancy (0-3 months)",
        "focus": "Genetic testing initiation; infantile spasm vigilance; AED-naive baseline EEG",
        "milestones": [
            "Clinical genetics referral at first seizure presentation",
            "Targeted gene panel (KCNA2 + DEE panel) sent urgently",
            "EEG: hypsarrhythmia or multifocal IED baseline",
            "POLG1 exclusion before any VPA consideration",
            "MRI brain at diagnosis: cerebellar assessment (GOF vs LOF signal)",
        ],
    },
    {
        "phase": "Infantile Epilepsy Phase (3-24 months)",
        "focus": "Infantile spasms treatment; AED initiation; functional assay referral",
        "milestones": [
            "Infantile spasms: ACTH or vigabatrin (IS-protocol) ± VPA adjunct",
            "KCNA2 genetic result review: confirm GOF/LOF status",
            "Functional assay referral if variant ambiguous",
            "Developmental surveillance: Griffiths DQ every 6M",
            "Family genetic counselling: de novo recurrence risk <1%",
        ],
    },
    {
        "phase": "Toddler / Preschool (2-5 years)",
        "focus": "AED optimisation; KD trial if ≥2 AED failures; 4-AP initiation if GOF confirmed",
        "milestones": [
            "GOF confirmed → initiate 4-AP trial with monitoring (SARA + seizure diary)",
            "LOF: KD initiation after 2nd AED failure",
            "Drop attacks: helmet prescription; callosotomy consideration if severe",
            "Neuropsychological assessment (Griffiths/Bayley-III) at age 2-3",
            "Enrolment in KCNA2 international patient registry",
        ],
    },
    {
        "phase": "School Age (5-12 years)",
        "focus": "Educational support; seizure safety; AED rationalisation; cerebellar monitoring",
        "milestones": [
            "IEP (Individualised Education Plan): SEP with seizure action plan + PT/OT/SLP",
            "Annual MRI: cerebellar volume quantification (GOF patients)",
            "SARA scale every 6M (GOF): cerebellar ataxia progression",
            "AED rationalisation: remove ineffective AEDs (polypharmacy review)",
            "Puberty approaching: VPA PREVENT programme re-consent (females)",
        ],
    },
    {
        "phase": "Adolescence (12-18 years)",
        "focus": "Transition planning; SUDEP counselling; reproductive health; driving restrictions",
        "milestones": [
            "Transition to adult neurology: structured handover document",
            "SUDEP counselling: formal discussion per NICE NG217",
            "Driving: seizure-free period requirement (national guidelines)",
            "VPA: PREVENT programme consent and contraception counselling (females)",
            "Independence assessment: self-medication readiness; school-leaving support",
        ],
    },
    {
        "phase": "Adulthood (18+ years)",
        "focus": "Long-term AED management; annual review; independent living; genetic counselling",
        "milestones": [
            "Annual neurologist review: seizure control, AED levels, drug interactions",
            "4-AP (GOF): review efficacy annually; cerebellar MRI q12-18M",
            "Genetic counselling: AD inheritance (de novo); offspring risk <50% (if de novo) vs 50% (if inherited)",
            "Employment/independent living: occupational therapy assessment",
            "SUDEP risk: ongoing safety plan; nighttime monitoring if unwitnessed nocturnal seizures",
        ],
    },
]

# ── Clinical Alerts (6) ───────────────────────────────────────────────────────
ALERTS = [
    "⚠️ 4-AP ABSOLUTELY CONTRAINDICATED in KCNA2-LOF — worsens seizures. GOF functional assay MUST be documented before prescribing.",
    "⚠️ POLG1 EXCLUSION MANDATORY before VPA — biallelic POLG1 + VPA = Alpers syndrome (fatal hepatic failure).",
    "⚠️ CEREBELLAR ATROPHY = GOF signal — serial MRI q12M; KCNA2-GOF causes progressive Purkinje cell loss.",
    "⚠️ SODIUM CHANNEL BLOCKERS (CBZ/OXC/PHT) — risk of myoclonic/atonic worsening in KCNA2-DEE with drop attacks.",
    "⚠️ SUDEP RISK — refractory GTCS in KCNA2-DEE: nocturnal monitoring + supine position + rescue midazolam training.",
    "⚠️ KCNA2-GOF PHENOTYPE OVERLAP WITH JME — GOF may show 3 Hz spike-wave resembling JME; background slowing + ataxia + early onset distinguishes KCNA2-GOF.",
]

# ── Key Concepts / Definitions (14) ──────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "KCNA2 (Kv1.2)",
        "definition": (
            "KCNA2 encodes Kv1.2, the voltage-gated potassium channel α-subunit of the "
            "Shaker (Kv1) subfamily — 1p13.3. Kv1.2 is a delayed-rectifier K+ channel "
            "critical for action potential repolarisation at axon initial segments and "
            "juxtaparanodal regions. Kv1.2 homotetramers or heterotetramers "
            "(with Kv1.1/Kv1.4/Kv1.6) form the primary repolarising current in cortical "
            "pyramidal neurons and cerebellar Purkinje cells. De novo pathogenic KCNA2 "
            "variants cause KCNA2-DEE via two distinct mechanisms: LOF or GOF."
        ),
    },
    {
        "term": "KCNA2-DEE (Developmental and Epileptic Encephalopathy)",
        "definition": (
            "ILAE 2022 recognised genetic DEE caused by de novo KCNA2 pathogenic variants. "
            "Characterised by early-onset refractory focal/multifocal or generalised seizures, "
            "developmental delay/plateau, and in GOF cases, progressive cerebellar ataxia + "
            "cerebellar atrophy. Two mechanistic subtypes: KCNA2-LOF-DEE (dominant-negative "
            "or haploinsufficiency → reduced K+ repolarisation → neuronal hyperexcitability) "
            "and KCNA2-GOF-DEE (hyperpolarised K+ leak in interneurons → disinhibition → "
            "seizures + ataxia). Estimated prevalence ~1/100,000. ClinGen: gene-disease "
            "validity DEFINITIVE for KCNA2 ↔ DEE."
        ),
    },
    {
        "term": "Dual-Mechanism Channelopathy",
        "definition": (
            "KCNA2 is the prototypical dual-mechanism epilepsy gene: both LOF and GOF "
            "variants cause DEE via opposite biophysical mechanisms yet similar clinical "
            "phenotypes (early-onset DEE). This creates the critical clinical challenge "
            "of GOF/LOF distinction — because precision therapy (4-AP) is GOF-selective. "
            "Other dual-mechanism epilepsy genes: KCNB1, HCN1 (limited GOF data), CACNA1A. "
            "KCNA2 is unique in having a documented precision therapy (4-AP) specifically "
            "for its GOF mechanism."
        ),
    },
    {
        "term": "GOF (Gain-of-Function) Kv1.2 variant",
        "definition": (
            "GOF KCNA2 variants produce a hyperpolarised shift in Kv1.2 voltage activation "
            "(-10 to -30 mV vs WT) or slowed inactivation — creating a 'leak' K+ current "
            "at resting membrane potentials. In interneurons (high Kv1.2 density): GOF K+ "
            "leak → interneuron hyperpolarisation → loss of GABAergic inhibition → network "
            "disinhibition → seizures. In Purkinje cells (highest CNS Kv1.2): GOF → "
            "Purkinje cell output dysregulation → cerebellar ataxia + progressive atrophy. "
            "Most recurrent GOF hotspot: P405L (>10 de novo cases globally). "
            "Precision therapy: 4-AP blocks GOF leak current."
        ),
    },
    {
        "term": "LOF (Loss-of-Function) Kv1.2 variant — dominant-negative or haploinsufficiency",
        "definition": (
            "LOF KCNA2 variants reduce or abolish K+ channel current via dominant-negative "
            "heterotetramer assembly (missense LOF: >70% current reduction) or NMD-mediated "
            "haploinsufficiency (truncating: 50% reduction). Reduced Kv1.2 repolarisation → "
            "prolonged action potentials → neuronal hyperexcitability → focal/multifocal DEE. "
            "Phenotype: severe DEE without cerebellar atrophy (distinguishes from GOF). "
            "Precision therapy: 4-AP CONTRAINDICATED — would further reduce K+ conductance. "
            "Recommended: VPA + KD after ≥2 AED failures."
        ),
    },
    {
        "term": "4-Aminopyridine (4-AP) / dalfampridine — KCNA2-GOF precision therapy",
        "definition": (
            "4-AP is a broad-spectrum voltage-gated K+ channel blocker used clinically for "
            "walking speed improvement in MS (dalfampridine, FDA 2010). In KCNA2-GOF: "
            "4-AP blocks the aberrant GOF 'leak' Kv1.2 current → partially restores "
            "normal interneuron firing → reduces disinhibition-driven seizures and "
            "cerebellar ataxia. Level C evidence (Syrbe 2015 Nat Genet, Masnada 2017 BRAIN, "
            "Semmler 2020 EJPN). GOF functional assay confirmation is MANDATORY before "
            "prescribing — LOF patients risk acute deterioration on 4-AP."
        ),
    },
    {
        "term": "Pericentral Somatosensory Seizures",
        "definition": (
            "Characteristic KCNA2-DEE seizure semiology: focal tonic or clonic movements "
            "of arm/hand/face arising from pericentral (centroparietal/Rolandic) cortex "
            "— the region of highest Kv1.2 expression in neocortex. EEG: rhythmic "
            "centroparietal or frontocentral beta discharge. Semiology: contralateral "
            "arm tonic extension, hand posturing, or facial twitching. "
            "Key differential: benign Rolandic epilepsy (SeLECTS/BECTS) — but KCNA2-DEE "
            "has background slowing, multifocal IEDs, and developmental impairment "
            "not seen in BECTS."
        ),
    },
    {
        "term": "Cerebellar Atrophy — KCNA2-GOF marker",
        "definition": (
            "Progressive cerebellar atrophy is the hallmark MRI finding in KCNA2-GOF variants: "
            "Purkinje cells express the highest CNS Kv1.2 density; GOF K+ leak → "
            "Purkinje cell output dysregulation → eventual Purkinje cell death → "
            "progressive vermis + hemisphere volume loss. Present in 80-90% of GOF patients "
            "by school age. ABSENCE of cerebellar atrophy in KCNA2-DEE suggests LOF. "
            "MRI discrimination: cerebellar atrophy = GOF → 4-AP eligible. "
            "Normal cerebellum = LOF → 4-AP contraindicated. "
            "Serial MRI q12M for volumetric tracking."
        ),
    },
    {
        "term": "SARA (Scale for Assessment and Rating of Ataxia)",
        "definition": (
            "Standardised 40-point clinical scale for cerebellar ataxia quantification "
            "(0=normal, 40=most severe): 8 subscores (gait, stance, sitting, speech, "
            "finger chase, nose-finger test, fast alternating hand movements, heel-shin). "
            "Used for KCNA2-GOF monitoring: baseline SARA before 4-AP, repeat at 4W, 12W, "
            "then 3-monthly. Clinically meaningful change: ≥4 points improvement. "
            "Validated in cerebellar ataxias and paediatric neurology. "
            "Combined with seizure diary for 4-AP efficacy assessment in GOF patients."
        ),
    },
    {
        "term": "Multifocal Epileptiform Discharges — KCNA2-DEE EEG signature",
        "definition": (
            "Independent epileptiform spikes or sharp-slow waves arising from ≥2 non-contiguous "
            "cortical regions in the same patient — characteristic KCNA2-LOF EEG pattern. "
            "Unlike focal structural epilepsy (one predominant focus) or generalised genetic "
            "epilepsy (symmetric generalised discharge), KCNA2-LOF EEG shows centrotemporal, "
            "occipital, and frontal independent foci with no single dominant region. "
            "NREM sleep: multifocal spiking density 3-5× higher than wakefulness. "
            "Multifocal IEDs + background generalised slowing = KCNA2-DEE red flag → gene panel."
        ),
    },
    {
        "term": "Dominant-Negative Effect (Kv1.2 channel)",
        "definition": (
            "Kv1.2 functions as a tetramer: 4 α-subunits assemble to form one functional "
            "channel pore. In dominant-negative KCNA2 missense LOF: one mutant α-subunit "
            "in a heterotetramer inactivates the entire channel — producing >75% current "
            "reduction (far greater than simple haploinsufficiency at 50%). This is why "
            "KCNA2 LOF missense variants cause more severe DEE than would be predicted "
            "by haploinsufficiency alone: dominant-negative mechanism amplifies the "
            "functional loss. Key examples: T252I, P405L (LOF), I402T. "
            "Confirmed by co-injection rescue experiment in Xenopus oocytes."
        ),
    },
    {
        "term": "KCNA2-GOF P405L Hotspot",
        "definition": (
            "P405L (c.1214C>T) in KCNA2 exon 5 (S6 transmembrane gate) is the most recurrent "
            "de novo GOF variant globally — >10 independent de novo cases reported across "
            "multiple KCNA2 cohorts. P405 is located at the S6 inner pore gate: leucine "
            "substitution removes the conserved proline kink required for fast inactivation → "
            "channels remain open longer → GOF leak current. Functional patch-clamp: "
            "P405L produces +250% persistent K+ current compared to WT at resting potential. "
            "4-AP documented responder rate for P405L cases: ~60% in published series. "
            "ClinVar classification: Pathogenic. ACMG: PS2+PS3+PM1+PM2+PP4."
        ),
    },
    {
        "term": "KCNA2-Alliance / Patient Registry",
        "definition": (
            "International patient registry for KCNA2-DEE families: coordinates variant "
            "submissions, functional assay referrals, 4-AP trial data, and natural history "
            "data collection. Scientific leads: Hecke group (Hamburg University Medical Centre), "
            "Lerche group (Tübingen), Bhatt lab (Cincinnati Children's). "
            "Enrolment: all KCNA2-DEE patients recommended at diagnosis. "
            "Database feeds international KCNA2 genotype-phenotype analysis and "
            "future trial design. Families: KCNA2 Foundation patient advocacy group "
            "provides family support and 4-AP access pathways."
        ),
    },
    {
        "term": "Ketogenic Diet (KD) for KCNA2-LOF-DEE",
        "definition": (
            "Classic KD (4:1 fat:carbohydrate+protein) is the primary non-pharmacological "
            "adjunct for KCNA2-LOF-DEE after ≥2 AED failures. Mechanistic rationale: "
            "ketone bodies activate ATP-sensitive K+ channels (Kir6.2/SUR1) → membrane "
            "hyperpolarisation → partially compensates for reduced Kv1.2 repolarisation "
            "in LOF patients. Observational data: ~40% achieve ≥50% seizure reduction "
            "in KCNA2-LOF series. KD also reduces drop attack frequency — "
            "combined with CLB as rational LOF polytherapy. Metabolic screen mandatory "
            "before initiation (POLG, pyruvate carboxylase deficiency exclusion)."
        ),
    },
]

# ── Standards (8) ─────────────────────────────────────────────────────────────
STANDARDS = [
    {
        "standard": "ILAE-2022",
        "full": "ILAE 2022 Classification of Epilepsies — KCNA2-DEE as recognised genetic DEE",
        "relevance": "KCNA2-DEE classified as genetic DEE; GOF/LOF distinction acknowledged",
    },
    {
        "standard": "NICE-NG217",
        "full": "NICE Guideline NG217 (2022) — Epilepsies: diagnosis and management",
        "relevance": "Genetic testing in DEE; SUDEP counselling; AED choice in DEE",
    },
    {
        "standard": "ClinGen-KCNA2-DEE-Definitive",
        "full": "ClinGen Gene-Disease Validity: KCNA2 ↔ DEE — DEFINITIVE (2023)",
        "relevance": "Gene-disease validity confirmed; mandates KCNA2 on all DEE panels",
    },
    {
        "standard": "ACMG-AMP-2015",
        "full": "ACMG/AMP 2015 Standards for Interpretation of Sequence Variants",
        "relevance": "Pathogenicity classification; PS3 (functional assay) for GOF/LOF confirmation",
    },
    {
        "standard": "ACNS-EEG-2021",
        "full": "ACNS Guideline: Standardised Terminology for Seizures and Periodic Patterns",
        "relevance": "Multifocal IED definition; NCSZ detection criteria; hypsarrhythmia classification",
    },
    {
        "standard": "MHRA-PREVENT-VPA-2024",
        "full": "MHRA PREVENT Programme (2024) — Valproate in females of childbearing age",
        "relevance": "VPA teratogenicity PREVENT consent for KCNA2-DEE females; annual consent renewal",
    },
    {
        "standard": "Syrbe-2015-NatGenet",
        "full": "Syrbe S et al. (2015) Nat Genet 47:393-399 — KCNA2 discovery paper",
        "relevance": "Discovery of GOF/LOF KCNA2 variants as cause of DEE; first 4-AP trial data",
    },
    {
        "standard": "Masnada-2017-BRAIN",
        "full": "Masnada S et al. (2017) BRAIN 140:2321-2337 — KCNA2-DEE cohort",
        "relevance": "31-patient cohort; GOF/LOF phenotype delineation; 4-AP outcomes; cerebellar atrophy data",
    },
]

# ── Thresholds (10) ───────────────────────────────────────────────────────────
THRESHOLDS = [
    {
        "threshold": "GOF functional assay required before 4-AP prescribing",
        "value": "MANDATORY — no 4-AP without documented GOF confirmation",
        "rationale": "4-AP WORSENS LOF patients — functional classification prevents iatrogenic harm",
    },
    {
        "threshold": "VPA TDM target",
        "value": "50-100 mg/L (therapeutic range)",
        "rationale": "Sub-therapeutic <50: seizure breakthrough; supra-therapeutic >100: toxicity (tremor/hepatic)",
    },
    {
        "threshold": "KD after AED failures",
        "value": "≥2 AED failures → KD trial (KCNA2-LOF-DEE)",
        "rationale": "KD is level B adjunct for LOF-DEE; earlier initiation correlates with better seizure control",
    },
    {
        "threshold": "POLG1 exclusion before VPA",
        "value": "MANDATORY — biallelic POLG1 + VPA = Alpers syndrome (fatal)",
        "rationale": "POLG1 mutation causes mitochondrial DNA polymerase deficiency; VPA inhibits mtDNA replication",
    },
    {
        "threshold": "4-AP trial duration (GOF)",
        "value": "12 weeks minimum before declaring non-responder",
        "rationale": "Kv1.2 membrane trafficking adaptation requires 8-12 weeks for full 4-AP pharmacodynamic effect",
    },
    {
        "threshold": "Cerebellar atrophy → GOF classification",
        "value": "Cerebellar atrophy on MRI = GOF signal; normal cerebellum = LOF",
        "rationale": "MRI differentiator pending formal functional assay; guides 4-AP eligibility interim decision",
    },
    {
        "threshold": "SARA clinically meaningful change",
        "value": "≥4 points improvement on SARA (0-40)",
        "rationale": "Validated MCID for cerebellar ataxia scale; below this = within measurement error",
    },
    {
        "threshold": "MRI surveillance (GOF)",
        "value": "Annual MRI; every 6M if active cerebellar atrophy progression",
        "rationale": "Cerebellar volume loss tracking guides GOF severity classification and 4-AP urgency",
    },
    {
        "threshold": "SUDEP counselling requirement",
        "value": "All KCNA2-DEE patients with refractory GTCS; annually per NICE NG217",
        "rationale": "GTCS + DEE: SUDEP risk ~1:100-1:1000 patient-years; night monitoring + supine position",
    },
    {
        "threshold": "Norclobazam TDM (CLB active metabolite)",
        "value": "50-300 ng/mL",
        "rationale": "Norclobazam (t½ ~40h) primary active metabolite; TDM correlates better with efficacy than CLB parent",
    },
]

# ── References (6) ────────────────────────────────────────────────────────────
REFERENCES = [
    {
        "citation": "Syrbe S et al. (2015) Nat Genet 47:393-399",
        "title": "De novo loss- or gain-of-function mutations in KCNA2 cause epileptic encephalopathy",
        "relevance": (
            "Discovery paper — identified KCNA2 as a dual-mechanism DEE gene; "
            "characterised GOF vs LOF electrophysiology; first clinical description of "
            "4-AP benefit in 2/3 GOF patients (seizure + ataxia improvement). "
            "Established the GOF/LOF classification framework for KCNA2-DEE."
        ),
    },
    {
        "citation": "Masnada S et al. (2017) BRAIN 140:2321-2337",
        "title": "Clinical spectrum and genotype-phenotype associations of KCNA2-related encephalopathies",
        "relevance": (
            "Largest published KCNA2-DEE cohort (N=31); detailed GOF vs LOF phenotype "
            "delineation; cerebellar atrophy documented as GOF hallmark; 4-AP response "
            "data in GOF subgroup; functional assay data for each variant; "
            "natural history from infancy to adolescence."
        ),
    },
    {
        "citation": "Semmler A et al. (2020) Eur J Paediatr Neurol 26:98-105",
        "title": "4-aminopyridine in KCNA2 gain-of-function mutations: clinical observations",
        "relevance": (
            "Case series documenting 4-AP clinical outcomes in confirmed KCNA2-GOF patients; "
            "P405L hotspot responders; ataxia scale (SARA) and seizure frequency pre/post; "
            "dose-response observations; adverse effect monitoring protocol."
        ),
    },
    {
        "citation": "Allen NM et al. (2014) Neurology 82:1917-1924",
        "title": "Apparent SUDEP in a child with KCNA2 epileptic encephalopathy",
        "relevance": (
            "First KCNA2-DEE SUDEP case report; highlights elevated SUDEP risk in KCNA2-DEE; "
            "established SUDEP counselling as mandatory in KCNA2-DEE management."
        ),
    },
    {
        "citation": "Smart SL et al. (1998) Neuron 20:809-819",
        "title": "Deletion of the Kv1.2 potassium channel causes seizures in mice",
        "relevance": (
            "Foundational animal model validation: Kv1.2 heterozygous knockout mice develop "
            "spontaneous seizures — genetic proof-of-principle for KCNA2 haploinsufficiency "
            "as seizure mechanism; established Kv1.2 as epilepsy gene before human variant discovery."
        ),
    },
    {
        "citation": "Lehmann-Horn F & Jurkat-Rott K (1999) Physiol Rev 79:1317-1372",
        "title": "Voltage-gated ion channels and hereditary disease",
        "relevance": (
            "Foundational review of channelopathy mechanisms including Kv1 family; "
            "LOF vs GOF electrophysiological principles; tetramerisation and dominant-negative "
            "mechanisms; remains the conceptual framework for all Kv1.2 channelopathy interpretation."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    """KCNA2-DEE (Kv1.2 Channelopathy) — overview endpoint."""
    total = sum(e["n"] for e in ETIOLOGY_CATALOG)
    gof_n = next(e["n"] for e in ETIOLOGY_CATALOG if "GOF" in e["category"])
    lof_n = sum(e["n"] for e in ETIOLOGY_CATALOG if "LOF" in e["category"] or "truncating" in e["category"])
    return {
        "syndrome": "KCNA2-DEE — Developmental and Epileptic Encephalopathy (Kv1.2 Channelopathy)",
        "gene": "KCNA2 — 1p13.3 — Kv1.2 Voltage-Gated Potassium Channel α-subunit",
        "protein_function": "Kv1.2: delayed-rectifier K+ channel at axon initial segments + juxtaparanodes → action potential repolarisation",
        "dual_mechanism": "LOF → dominant-negative/haploinsufficiency → neuronal hyperexcitability → focal/multifocal DEE | GOF → interneuron K+ leak → disinhibition → DEE + cerebellar ataxia",
        "precision_therapy": "4-Aminopyridine (4-AP / dalfampridine) — GOF-ONLY (contraindicated in LOF)",
        "inheritance": "De novo >95%; autosomal dominant; mosaicism rare",
        "cohort": total,
        "gof_patients": gof_n,
        "lof_patients": lof_n,
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
        "gof_hallmark_mri": "Cerebellar atrophy (Purkinje cell loss) — progressive, serial MRI q12M",
        "lof_hallmark_eeg": "Multifocal epileptiform discharges — ≥2 independent foci, generalised background slowing",
        "key_safety": (
            "4-AP ABSOLUTE CI in LOF — functional assay mandatory before prescribing | "
            "POLG1 exclusion before VPA | CBZ/OXC risk myoclonic/atonic worsening"
        ),
        "top_alerts": ALERTS[:3],
        "generated_at": datetime.now().isoformat(),
        "dashboard_id": 185,
    }


def get_breakdown():
    """KCNA2-DEE (Kv1.2 Channelopathy) — breakdown endpoint (full clinical detail)."""
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
    """KCNA2-DEE (Kv1.2 Channelopathy) — definitions endpoint (14 key concepts)."""
    return {
        "syndrome": "KCNA2-DEE — Developmental and Epileptic Encephalopathy (Kv1.2 Channelopathy)",
        "definitions": DEFINITIONS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "generated_at": datetime.now().isoformat(),
    }
