"""
EEF1A2 Epilepsy (Developmental and Epileptic Encephalopathy-5 / eEF1A2 / Ribosomal Translation)
================================================================================================
40-patient cohort · EEF1A2 (20q13.33) · Eukaryotic Translation Elongation Factor 1-Alpha 2
EEF1A2 encodes the neuron/muscle-specific isoform of the eukaryotic translation elongation factor
1-alpha, which delivers aminoacyl-tRNA to the ribosomal A-site in a GTP-dependent manner.
De novo dominant variants cause DEE-5 (OMIM #616577) — a severe, early-onset epileptic
encephalopathy with infantile spasms, Lennox-Gastaut evolution, and profound intellectual disability.

EEF1A2 BIOLOGY:
EEF1A2 (20q13.33) encodes eEF1A2, a GTP-binding protein (463 aa, 50.1 kDa) expressed
exclusively in neurons and muscle cells. Three functional domains:
  Domain I (G-domain, aa 1–235): GTP binding and hydrolysis. Binds GTP → recruits
    aminoacyl-tRNA → deposits aa-tRNA at ribosomal A-site → GTP hydrolysis → eEF1A2·GDP
    released → recycled by eEF1Bα/β/γ GEF complex.
  Domain II (aa 236–335): tRNA accommodation and interactions with ribosomal decoding centre.
  Domain III (aa 336–463): Actin-bundling — non-canonical cytoskeletal role; bundles F-actin
    filaments important for dendritic spine morphology and synaptic vesicle trafficking.

CRITICAL MECHANISTIC PEARL — THE POSTNATAL eEF1A1→eEF1A2 SWITCH:
  eEF1A1 is the ubiquitous isoform expressed in all cells including fetal neurons (94% AA identity).
  eEF1A2 is the POSTNATAL neuronal isoform: expression is essentially ABSENT in fetal neurons and
  begins ~3–4 weeks postnatal in mice (equivalent ~3–6 months in humans).
  The switch: eEF1A1 mRNA is actively degraded in mature neurons via eIF2α-mediated repression;
  simultaneously eEF1A2 transcription is activated by neuron-specific enhancers (Bhatt et al. 1999).
  WHY THIS EXPLAINS THE PHENOTYPE:
    ① No in-utero seizures (unlike PNPO/ALDH7A1): fetal neurons use eEF1A1 — disease onset
       only after the switch, typically 3–9 months postnatal.
    ② Tissue specificity: non-neuronal cells are spared (retain eEF1A1 permanently) → NO systemic
       or cardiac involvement (unlike some DEEs).
    ③ 'Window of vulnerability': the transition period (3–9 months) when eEF1A1 is shutting down
       but eEF1A2 LOF cannot compensate → neuronal translation collapse → epilepsy onset.
    ④ Muscle is also affected (eEF1A2 expressed in skeletal muscle) → hypotonia is near-universal.
  Therapeutic implication: AAV-based eEF1A2 replacement is theoretically tractable (neuron-specific
  promoter already known; small coding sequence). No precision therapy yet approved (2026).

INHERITANCE: De novo dominant — almost all variants are de novo heterozygous missense or LOF.
  Rare biallelic AR LOF: p.Arg396Gln/p.Arg396Gln — documented in 2 consanguineous families.
  Prevalence: ~150–200 patients reported worldwide (rare DEE; WES is required for diagnosis).
  First described: Nakajima et al. 2015 Hum Mutat; Lam et al. 2016 Neurogenetics.
"""

import random
from datetime import datetime

SEED = 9251  # dashboard 251
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=40) ───────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": (
            "EEF1A2 LOF missense — G-domain / GTP-binding impaired "
            "(de novo dominant, domain I missense)"
        ),
        "n": 16, "pct": 40,
        "category": "LOF-missense-G-domain-GTPase-impaired",
        "functional_class": "LOF-G-domain",
        "mechanism": (
            "Most prevalent class (~40%): de novo heterozygous missense variants in the G-domain "
            "(domain I, aa 1–235) of eEF1A2 — e.g. p.Val216Ala, p.Thr191Ala, p.Ser200Pro, "
            "p.Ala399Thr, p.Ile190Thr. These substitutions reduce GTPase activity or impair "
            "GTP/GDP exchange, disrupting aminoacyl-tRNA delivery to the ribosomal A-site. "
            "Dominant negative: mutant eEF1A2·GDP cannot be recycled efficiently, trapping the "
            "elongation cycle at the A-site accommodation step → ribosome stalling → proteotoxic "
            "stress → mTOR activation → UPR → neuronal apoptosis/dysfunction. Onset: infantile "
            "spasms at 4–8 months (coincides with eEF1A1→eEF1A2 switch). EEG: hypsarrhythmia → "
            "multifocal. MRI: thin corpus callosum (40%) or polymicrogyria (20%). Outcome: "
            "severe ID, non-verbal. ACTH/VGB initial response ~50%. Most evolve to LGS."
        ),
        "variant_examples": ["p.Val216Ala", "p.Thr191Ala", "p.Ser200Pro", "p.Ala399Thr"],
        "response_to_acth": "50% spasm cessation (EEG hypsarrhythmia resolution in 6 weeks)",
        "response_to_kd": "40–55% >50% seizure reduction (best evidence in refractory LGS phase)",
    },
    {
        "etiology": (
            "EEF1A2 LOF missense — Domain II/III impaired "
            "(tRNA accommodation / actin bundling disrupted)"
        ),
        "n": 10, "pct": 25,
        "category": "LOF-missense-domain-II-III-tRNA-actin",
        "functional_class": "LOF-domain-II-III",
        "mechanism": (
            "Second class (~25%): de novo missense variants in domains II or III of eEF1A2 — "
            "e.g. p.Glu462Lys (C-terminal actin bundling), p.Asn358Ile, p.Leu303Arg (domain II). "
            "Domain II variants impair tRNA accommodation into the A-site without affecting GTPase "
            "activity — ribosomal decoding is slower, increasing translational error rate. "
            "Domain III variants disrupt the non-canonical actin-bundling activity of eEF1A2: "
            "actin filament organisation in dendritic spines is impaired → synaptopathy → "
            "dendritic arbour simplification (visible on Golgi staining). Clinically: slightly "
            "later onset than G-domain class (6–10 months), more myoclonic-predominant seizure "
            "pattern, autistic features prominent (~70%), hypotonia often severe. MRI: simplified "
            "gyral pattern or pachygyria (15%). Some domain III variants may have partial GOF "
            "(aberrant actin cross-linking) — preclinical data only."
        ),
        "variant_examples": ["p.Glu462Lys", "p.Asn358Ile", "p.Leu303Arg", "p.Arg396Gln"],
        "response_to_acth": "45% spasm response; myoclonic burden often persists post-ACTH",
        "response_to_kd": "50% response rate; especially effective for myoclonic component",
    },
    {
        "etiology": (
            "EEF1A2 LOF truncating — haploinsufficiency "
            "(frameshift / nonsense / splice-site de novo)"
        ),
        "n": 6, "pct": 15,
        "category": "LOF-truncating-haploinsufficiency",
        "functional_class": "LOF-truncating-haploinsufficiency",
        "mechanism": (
            "Third class (~15%): de novo truncating variants (frameshift, nonsense, essential "
            "splice-site) causing premature termination of eEF1A2 translation → NMD → true "
            "haploinsufficiency. Examples: p.Arg396Ter, p.Gln7fs, splice c.785+1G>A. "
            "Haploinsufficiency in neurons: after the postnatal eEF1A1→eEF1A2 switch, the "
            "remaining single functional allele cannot sustain the high translational demand of "
            "mature neurons (which have the highest ribosomal density of any cell type) → "
            "selective neuronal vulnerability. Hypotonia and motor delay are often the first "
            "manifestation (before seizures) because muscle is similarly affected. MRI: more "
            "frequently normal (45%) than missense classes. Outcome: similar severe phenotype. "
            "Biallelic AR truncating variants have been reported (two consanguineous families) → "
            "neonatal/infantile-onset with earlier, more severe encephalopathy."
        ),
        "variant_examples": ["p.Arg396Ter", "p.Gln7fs", "c.785+1G>A (splice)"],
        "response_to_acth": "55% IS response; lower relapse rate than missense classes",
        "response_to_kd": "35–45% response; seizure burden often less refractory",
    },
    {
        "etiology": (
            "EEF1A2 GOF missense — suspected toxic gain of function "
            "(actin hyperbundling / mTOR dysregulation)"
        ),
        "n": 4, "pct": 10,
        "category": "GOF-missense-toxic-actin-mTOR",
        "functional_class": "GOF-toxic",
        "mechanism": (
            "Fourth class (~10%): de novo missense variants with in vitro evidence of gain-of-function "
            "— aberrant actin hyperbundling or constitutive GTP-independence. Examples: "
            "p.Met396Ile (reduced GTP hydrolysis → constitutively active), p.His432Arg (C-terminal "
            "actin-crosslinking domain — hyperactivated bundling). These variants show increased "
            "co-immunoprecipitation with F-actin and reduced eEF1Bα interaction (defective "
            "recycling → constitutive GTP-bound active state → persistent tRNA delivery attempts "
            "with aberrant peptide elongation → ribosomal collisions → ribotoxic stress response). "
            "Clinically: often slightly worse developmental outcome; epileptic spasm onset earlier "
            "(3–5 months). mTOR pathway activation has been demonstrated in patient-derived "
            "neurons (Lam 2016 Neurogenetics) — theoretical implication for rapamycin, but no "
            "clinical evidence yet. This subset is important: mTOR inhibition (everolimus) is NOT "
            "established for EEF1A2 (unlike MTOR/TSC where somatic GOF is the mechanism)."
        ),
        "variant_examples": ["p.Met396Ile", "p.His432Arg"],
        "response_to_acth": "35% — lowest response rate; early ACTH failure common",
        "response_to_kd": "50–60% response; KD may mitigate mTOR/ribotoxic stress via BHB/adenosine",
    },
    {
        "etiology": (
            "Phenocopy — DEE overlapping EEF1A2 "
            "(CDKL5 / ARX / KCNQ2 / SCN1A / STXBP1 initially suspected)"
        ),
        "n": 4, "pct": 10,
        "category": "Phenocopy-DEE-CDKL5-ARX-KCNQ2",
        "functional_class": "Phenocopy",
        "mechanism": (
            "Fifth class (~10%): patients with clinical DEE phenotype initially attributed to "
            "EEF1A2 before confirmatory genetic testing; ultimately reclassified. Common sources: "
            "CDKL5 deficiency (X-linked, more females, hypotonic seizures with hand stereotypies), "
            "ARX (males, dystonia + spasms), KCNQ2 neonatal-onset (earlier than EEF1A2 switch), "
            "SCN1A Dravet (febrile seizure trigger at 5–9 months, temperature-sensitive). "
            "Important diagnostic pitfall: EEF1A2 onset timing (4–9 months) overlaps with multiple "
            "DEE syndromes. WES/panel is essential; single-gene testing often misses EEF1A2. "
            "Key distinguishing features: EEF1A2 — NO febrile seizure trigger early (unlike SCN1A); "
            "NO hand stereotypies (unlike CDKL5); Male + Female equally; 20q13.33 locus (distinct "
            "from all common DEE genes). eEF1A1→eEF1A2 switch timing as 'diagnostic clock.'"
        ),
        "variant_examples": ["N/A — phenocopy (different gene confirmed)"],
        "response_to_acth": "Variable — depends on true underlying etiology",
        "response_to_kd": "Variable",
    },
]

# ── Patient Registry (N=40) ────────────────────────────────────────────────────
_FIRST = [
    "Aiden", "Bella", "Carlos", "Diya", "Ethan", "Fatima", "Gabriel", "Hannah",
    "Ishaan", "Jaya", "Kai", "Leila", "Marco", "Nora", "Oscar", "Priya",
    "Quinn", "Rania", "Soren", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yasmine", "Zara", "Anton", "Bree", "Chen", "Danica", "Eli", "Freya",
    "Gus", "Hana", "Ivan", "Jasmine", "Kevin", "Lily", "Musa", "Nina",
]
_LAST = [
    "Nakamura", "Patel", "Reyes", "Singh", "Torres", "Ueda", "Vargas", "Wang",
    "Xavier", "Yılmaz", "Zhang", "Ahmed", "Becker", "Cruz", "Dubois", "Ellis",
    "Fischer", "García", "Hassan", "Ibrahim", "Jensen", "Kim", "López", "Müller",
    "Ndiaye", "Okafor", "Perrin", "Qin", "Romano", "Santos", "Thiaw", "Unger",
    "Volkov", "Walsh", "Xu", "Yamamoto", "Zoltan", "Adeyemi", "Beaumont", "Callahan",
]
_ETIOLOGY_KEYS = [e["category"] for e in ETIOLOGY_CATALOG]
_ETIOLOGY_NAMES = {e["category"]: e["etiology"][:50] for e in ETIOLOGY_CATALOG}
_ETIOLOGY_DIST = [0] * 40
idx = 0
for e in ETIOLOGY_CATALOG:
    for _ in range(e["n"]):
        _ETIOLOGY_DIST[idx] = e["category"]
        idx += 1
random.shuffle(_ETIOLOGY_DIST)

_VARIANT_POOL = {
    "LOF-missense-G-domain-GTPase-impaired": ["p.Val216Ala", "p.Thr191Ala", "p.Ser200Pro", "p.Ala399Thr", "p.Ile190Thr", "p.Gly42Glu"],
    "LOF-missense-domain-II-III-tRNA-actin": ["p.Glu462Lys", "p.Asn358Ile", "p.Leu303Arg", "p.Arg270His", "p.Trp290Ser"],
    "LOF-truncating-haploinsufficiency": ["p.Arg396Ter", "p.Gln7fs*14", "c.785+1G>A", "p.Lys23Ter", "p.Glu120fs*6"],
    "GOF-missense-toxic-actin-mTOR": ["p.Met396Ile", "p.His432Arg", "p.Gly46Cys"],
    "Phenocopy-DEE-CDKL5-ARX-KCNQ2": ["N/A (phenocopy)"],
}
_ONSET_POOL = ["3 months", "4 months", "5 months", "6 months", "7 months", "8 months", "9 months", "10 months", "12 months"]
_IS_RESPONSE = ["ACTH — IS resolved, relapsed 4m", "ACTH + VGB — IS resolved", "VGB — partial 50% reduction",
                "ACTH failed; KD started", "IS refractory to ACTH/VGB; VNS trial", "ACTH — brief response; LGS evolved"]

PATIENTS = []
for i in range(40):
    pid = f"EEF-{i+1:03d}"
    name = f"{_FIRST[i]} {_LAST[i]}"
    sex = "F" if i % 2 == 0 else "M"
    age_mo = random.randint(18, 84)
    onset = random.choice(_ONSET_POOL)
    etiol = _ETIOLOGY_DIST[i]
    vpool = _VARIANT_POOL[etiol]
    variant = random.choice(vpool)
    is_resp = random.choice(_IS_RESPONSE)
    seizure_free = random.random() < 0.12  # ~12% seizure-free
    PATIENTS.append({
        "id": pid,
        "name": name,
        "sex": sex,
        "age_months": age_mo,
        "onset": onset,
        "etiology_short": etiol.replace("-", " ").split(" ")[0].upper(),
        "etiology_category": etiol,
        "eef1a2_variant": variant,
        "is_response": is_resp,
        "currently_seizure_free": seizure_free,
        "current_aeds": random.choice([
            "VPA + CLB", "VPA + LEV + KD", "VPA + CLB + KD", "LEV + CLB",
            "VPA + TPM", "KD alone", "VGB + VPA", "CLB + KD + LEV",
        ]),
        "mri": random.choice([
            "Thin CC + simplified gyri", "Polymicrogyria (posterior)", "Normal",
            "Pachygyria (posterior)", "Normal (50%)", "Thin CC only",
        ]),
    })

# ── Seizure Types (5) ──────────────────────────────────────────────────────────
SEIZURE_CATALOG = [
    {
        "type": "Infantile Spasms (West Syndrome)",
        "pct": 88,
        "eeg": (
            "Hypsarrhythmia (chaotic high-amplitude delta + multifocal spikes, interhemispheric "
            "asynchrony). Classic ictal EEG: generalised electrodecrement (voltage attenuation) "
            "coinciding with spasm. Modified hypsarrhythmia (interhemispheric synchrony or with "
            "burst-suppression elements) in ~30%."
        ),
        "semiology": (
            "Flexion > extension clusters (Salaam attacks). Neck flexion, arm abduction/adduction, "
            "leg extension. Clusters on awakening from sleep. Subtle forms: head nods, eye "
            "deviation, facial grimace. Onset: 4–9 months — coincides with eEF1A1→eEF1A2 switch."
        ),
        "clinical_tip": (
            "The 4–9 month onset timing is the KEY diagnostic clock for EEF1A2 — it corresponds "
            "precisely to the postnatal eEF1A1→eEF1A2 switch in human neurons. Any infant with IS "
            "at this age should have WES/gene panel including EEF1A2. Pre-switch (0–3 months): "
            "eEF1A1 active → seizures absent. ACTH + VGB is first-line (UKISS trial protocol)."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "pct": 75,
        "eeg": (
            "Multifocal interictal discharges (temporal, frontal, central); generalised 3 Hz "
            "slow spike-and-wave (LGS pattern); ictal: fast recruiting rhythm → generalised "
            "clonic; post-ictal suppression 2–10 min. EEG normalisation between seizures is "
            "uncommon in EEF1A2 DEE — background is persistently slow."
        ),
        "semiology": (
            "Tonic phase 10–30s → clonic phase 30–60s; may be brief or prolonged; asymmetric "
            "onset in 40% (focal to bilateral tonic-clonic); post-ictal coma/stupor up to 30 min. "
            "GTCS are a major SUDEP risk factor in EEF1A2 — overnight monitoring is important."
        ),
        "clinical_tip": (
            "Prolonged GTCS (>5 min) should trigger emergency benzodiazepine protocol. SUDEP risk "
            "in EEF1A2 is high (estimated 1/200/year in refractory DEE cohorts). Seizure alarm "
            "systems (e.g. Empatica E4, Neurobit) should be discussed at diagnosis. Night-time "
            "prone positioning must be avoided."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "pct": 62,
        "eeg": (
            "Generalised polyspike-and-slow-wave (2.5–4 Hz); often stimulus-sensitive. "
            "Ictal: brief (<500ms) generalised polyspike burst time-locked to myoclonic jerk. "
            "Background slowing between episodes. May coexist with electroclinical dissociation "
            "(EEG myoclonic discharge without visible jerk in 20%)."
        ),
        "semiology": (
            "Brief (<1s) bilateral synchronous jerks — upper extremities > lower; head drops. "
            "Clusters on awakening. Stimulus-sensitive (startle, touch, noise) in 38%. "
            "Distinguish from myoclonus of cortical origin (consistent EEG correlate) vs "
            "subcortical (no correlate — relevant for domain III GOF variants affecting "
            "actin/synapse — non-ictal myoclonus possible)."
        ),
        "clinical_tip": (
            "Avoid sodium channel blockers (PHT/CBZ/OXC/LTG) — these worsen myoclonic burden "
            "in EEF1A2 (as in Dravet, LGS). VPA is the broad-spectrum first choice for "
            "myoclonus (after POLG1 screen). CLB reduces myoclonus effectively as adjunct. "
            "Video-EEG essential to determine whether myoclonus is ictal vs non-ictal."
        ),
    },
    {
        "type": "Focal Impaired Awareness Seizures (FIAS)",
        "pct": 48,
        "eeg": (
            "Focal temporal (>frontal) ictal onset — rhythmic theta/alpha discharge with "
            "contralateral spread; focal spikes + slow waves interictally. Temporal lobe "
            "predominance may reflect hippocampal vulnerability to postnatal eEF1A2 switch "
            "timing (high eEF1A2 expression in CA1/CA3 pyramidal neurons)."
        ),
        "semiology": (
            "Staring + behavioral arrest, automatisms (lip-smacking, hand gestures), "
            "post-ictal confusion. Duration 30–90s. In LGS phase: may become shorter/more "
            "frequent. Distinguish from absence-like episodes (shorter, no post-ictal "
            "state, 3 Hz SW pattern) — both present in EEF1A2."
        ),
        "clinical_tip": (
            "FIAS that evolve to bilateral tonic-clonic carry highest SUDEP risk — document on "
            "emergency care plan. Temporal lobe involvement in EEF1A2 may respond partially "
            "to CLB; SEEG evaluation in selected DRE cases has revealed resectable foci in "
            "~15% (posterior temporal/occipital) — pre-surgical evaluation warranted in DRE."
        ),
    },
    {
        "type": "Atonic / Drop Attacks (Lennox-Gastaut Spectrum)",
        "pct": 32,
        "eeg": (
            "LGS pattern: slow (<2.5 Hz) generalised spike-and-wave; paroxysmal fast activity "
            "during tonic seizures; diffuse background slowing. Drop attacks: brief generalised "
            "EMG burst (tonic <500ms) or sudden EMG silence (atonic) — both produce falls. "
            "Nocturnal tonic seizures common in LGS phase of EEF1A2."
        ),
        "semiology": (
            "Sudden loss of postural tone (atonic) or brief tonic stiffening (tonic drop) "
            "causing falls and head injuries. Helmets mandatory. Evolution from West syndrome "
            "to LGS pattern occurs in ~60% of EEF1A2 patients who had IS. Age at LGS "
            "evolution: typically 18 months–4 years."
        ),
        "clinical_tip": (
            "CLB (clobazam) reduces drop attack frequency in LGS phase (~40–55% responders). "
            "VPA is the backbone broad-spectrum AED. KD is highly recommended for LGS-phase "
            "EEF1A2 — multi-site observational data show 55% >50% drop attack reduction. "
            "Rufinamide (Level B) is approved for LGS drop attacks (check dosing in pediatrics). "
            "Protective helmets are MANDATORY once drop attacks are identified."
        ),
    },
]

# ── Triggers (8) ──────────────────────────────────────────────────────────────
TRIGGER_CATALOG = [
    {
        "trigger": "Fever / Intercurrent Illness",
        "pct": 85,
        "mechanism": (
            "Fever induces heat shock response — eEF1A2 is recruited to stress granules, "
            "reducing available translational capacity. In neurons already stressed by "
            "eEF1A2 LOF, fever further depletes functional elongation factor → ribosomal "
            "stalling → epileptic threshold lowered. NOT temperature-sensitive in the same "
            "NaV1.1 mechanism as SCN1A/Dravet — antipyretic use during fever is appropriate "
            "(unlike Dravet where antipyretics do not reliably prevent seizures)."
        ),
        "clinical_note": (
            "Fever management: paracetamol/ibuprofen at fever onset; written fever protocol "
            "essential. Key distinction from Dravet: in EEF1A2, the fever trigger is not a "
            "temperature-channel effect (unlike SCN1A NaV1.1 desensitisation) — brief "
            "febrile seizures are not a reliable diagnostic trigger for EEF1A2."
        ),
    },
    {
        "trigger": "Sleep Deprivation / Sleep Disruption",
        "pct": 72,
        "mechanism": (
            "Sleep promotes synaptic homeostasis (synaptic downscaling); disrupted sleep "
            "increases synaptic drive and reduces seizure threshold. In EEF1A2 DEE, "
            "sleep architecture is severely disrupted — high-amplitude slow-wave sleep "
            "is reduced, and interictal discharges are most frequent in NREM stages "
            "N2/N3. Sleep-deprived EEG often captures more abundant discharges."
        ),
        "clinical_note": (
            "Regular sleep schedule is critical. Carers should not wake child during "
            "cluster-free nights. Melatonin (0.5–3 mg at bedtime) improves sleep "
            "latency and is safe in EEF1A2. Avoid: stimulants, screen time pre-sleep. "
            "PSG/video-EEG during sleep recommended at LGS diagnosis — tonic seizures "
            "during NREM are a common trigger for respiratory compromise."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 65,
        "mechanism": (
            "AED levels drop below therapeutic range — synaptic inhibition decreases "
            "while excitatory drive is unchanged → breakthrough seizures. In EEF1A2 "
            "the 'background noise' of cortical hyperexcitability is already high; "
            "even brief gaps in coverage trigger prolonged seizure episodes. VPA has "
            "a long half-life (12–16h) providing some buffer; CLB shorter (18–22h)."
        ),
        "clinical_note": (
            "Family education: keep 24h emergency supply at school/respite. Written "
            "emergency protocol for missed doses. VPA levels: check TDM if breakthrough "
            "occurs — autoinduction is NOT a feature of VPA (unlike CBZ) but intercurrent "
            "illness reduces VPA levels by reducing protein binding. LEV levels less "
            "relevant (renal clearance, no protein binding issues)."
        ),
    },
    {
        "trigger": "Metabolic Stress / Catabolism",
        "pct": 58,
        "mechanism": (
            "Fasting, acute metabolic illness, surgery, or prolonged vomiting increases "
            "catabolic stress. eEF1A2 is involved in the cellular response to nutrient "
            "deprivation — under metabolic stress, eEF1A2 is preferentially directed to "
            "stress-response mRNA translation (HSP70, HSP90) at the expense of neuronal "
            "maintenance proteins. In LOF: this redirection of already-scarce functional "
            "eEF1A2 depletes neuronal maintenance → seizure threshold drops acutely."
        ),
        "clinical_note": (
            "Pre-surgical fasting protocols should be minimised in EEF1A2 patients. "
            "IV glucose/dextrose maintenance during nil-by-mouth periods. IV AEDs "
            "(VPA IV, LEV IV, PHB IV) should be prescribed for any nil-by-mouth period >4h. "
            "Inform anaesthesiologist: EEF1A2 DEE — high seizure risk peri-operatively."
        ),
    },
    {
        "trigger": "Sensory Overstimulation",
        "pct": 52,
        "mechanism": (
            "Excessive visual/auditory/tactile stimulation can provoke myoclonic or "
            "absence-like events. In EEF1A2, cortical excitability is tonically elevated "
            "due to impaired translational homeostasis — sensory cortices are more "
            "susceptible to spreading excitation. Photosensitivity (IPS-triggered discharges) "
            "is present in ~25% — check on standard EEG; if present, flickering lights "
            "and video games require avoidance/restriction."
        ),
        "clinical_note": (
            "Environmental modifications: dim lighting during cluster periods; ear protection "
            "if loud environments provoke. School accommodations: quiet room access, sensory "
            "breaks. Sensory integration therapy (OT) should be offered — not as seizure "
            "prevention but for comorbid sensory processing disorder (~55% in EEF1A2 DEE). "
            "Photosensitivity: screen with IPS on EEG annually."
        ),
    },
    {
        "trigger": "Startle",
        "pct": 38,
        "mechanism": (
            "Startle epilepsy — a subset of EEF1A2 patients have stimulus-sensitive myoclonus "
            "or generalised tonic-clonic seizures triggered by unexpected auditory or tactile "
            "stimuli. Mechanism: cortical reflex epilepsy — the startle network (amygdala-"
            "reticulothalamic pathway) has low threshold in EEF1A2 due to impaired GABA "
            "ergic translation (GAD2 mRNA is selectively dependent on efficient elongation). "
            "EEG: brief generalised polyspike burst time-locked to startle."
        ),
        "clinical_note": (
            "Document startled-triggered seizures on seizure diary. CLB reduces startle "
            "reflex severity. Avoid sudden loud noises in sleep. Startle epilepsy in EEF1A2 "
            "should NOT prompt phenytoin (which worsens myoclonic burden). CLB + VPA is "
            "the preferred regimen when startle is prominent."
        ),
    },
    {
        "trigger": "AED Taper / Rapid Weaning",
        "pct": 35,
        "mechanism": (
            "EEF1A2 DEE is a refractory epilepsy — AED weaning is high-risk. Rapid taper "
            "(>25% per 2 weeks) removes inhibitory tone faster than the already-compromised "
            "neuronal translation machinery can compensate. Even partially effective AEDs "
            "should not be withdrawn unless clear evidence of futility or serious side effects. "
            "ACTH taper after IS is a particularly high-risk period — ~30% of EEF1A2 patients "
            "relapse within 8 weeks of ACTH discontinuation."
        ),
        "clinical_note": (
            "ACTH taper protocol: extend over 8–12 weeks (not 4 weeks as in non-EEF1A2 IS). "
            "Start KD during ACTH taper if IS have responded — bridging KD before ACTH "
            "discontinuation reduces relapse rate. Any AED discontinuation: maximum 10% per "
            "4 weeks with diary monitoring. Never discontinue >1 AED simultaneously in EEF1A2."
        ),
    },
    {
        "trigger": "Catamenial / Hormonal Fluctuations",
        "pct": 18,
        "mechanism": (
            "In adolescent/adult females with EEF1A2 DEE. Perimenstrual progesterone withdrawal "
            "reduces neurosteroid (allopregnanolone) levels — reduced positive GABA-A modulation "
            "→ transient increase in seizure frequency. Relevant in later adolescence. "
            "Relatively lower prevalence than in genetic focal epilepsies (18% vs 40% in "
            "JME/focal epilepsy) — EEF1A2 seizures are multifactorial and hormonal contribution "
            "is less dominant given the severity of baseline cortical hyperexcitability."
        ),
        "clinical_note": (
            "Catamenial pattern: seizure diary stratified by menstrual cycle for 3+ cycles. "
            "If confirmed: perimenstrual CLB pulse therapy (2–3× daily CLB dose for 5 days "
            "perimenstrually) can reduce cluster frequency. Progesterone supplementation "
            "(natural, not synthetic progestin) has limited evidence but is sometimes tried. "
            "Refer to epilepsy + reproductive endocrinology specialist."
        ),
    },
]

# ── Treatments (8) ────────────────────────────────────────────────────────────
TREATMENT_CATALOG = [
    {
        "treatment": "ACTH (Tetracosactide / Acthar)",
        "level": "Level A",
        "role": "First-line for infantile spasms (IS); initiated as soon as IS diagnosed",
        "dose": (
            "Tetracosactide: 0.5–1.0 mg/1.5 mg/kg IM alternate days (UK UKISS protocol) "
            "OR synthetic ACTH: 20–40 IU/day IM for 14 days → slow taper over 6–8 weeks. "
            "Alternative: high-dose prednisolone 10 mg QDS (40 mg/day) for 14 days (UKISS "
            "equivalent). Do NOT use low-dose ACTH — dose-dependent response."
        ),
        "moa": (
            "ACTH acts via MC2R on adrenal cortex → cortisol production → broad "
            "anti-inflammatory + CRH suppression. Direct CNS effect via MC1/MC3/MC4R: "
            "melanocortin receptors on neurons suppress CRH-driven excitability. "
            "Reduces hypsarrhythmia via cortisol-mediated GABA-A receptor upregulation "
            "and NMDA receptor downregulation. eEF1A2-specific: ACTH does not restore "
            "translation — it suppresses excitability while the clinical phenotype persists."
        ),
        "efficacy": (
            "IS cessation at 14 days: ~50% in EEF1A2 (lower than idiopathic IS ~80%). "
            "Hypsarrhythmia resolution (EEG criterion): ~55%. Relapse within 6 months: "
            "~30% in EEF1A2 (higher than structural IS). Best outcomes: early diagnosis "
            "and ACTH initiation within 4 weeks of IS onset — CRITICAL for neurodevelopment."
        ),
        "monitoring": (
            "BP (daily first 2 weeks — hypertension risk with ACTH). Blood glucose (ACTH → "
            "hyperglycemia). Electrolytes (Na/K — mineralocorticoid effect). Infection "
            "screen (immunosuppression). Weight/BMI. Mood/behavioural changes. "
            "EEG at 14 days and 28 days (hypsarrhythmia resolution criterion)."
        ),
        "eef1a2_specific_notes": (
            "EEF1A2 IS have a higher ACTH failure rate (~50%) than idiopathic/structural IS. "
            "If IS continue at 14 days → do NOT extend ACTH alone; add VGB or switch protocol. "
            "Start bridging KD DURING ACTH taper (not after) to reduce relapse rate. "
            "Document genetic diagnosis on emergency plan — EEF1A2 IS recurrence post-ACTH "
            "taper (~30%) should prompt early KD institution."
        ),
    },
    {
        "treatment": "Vigabatrin (VGB)",
        "level": "Level A",
        "role": "First-line IS (alternative/add-on to ACTH); especially effective in TSC-IS (use first); second IS first-line in EEF1A2",
        "dose": (
            "Infants: start 50 mg/kg/day in 2 divided doses; increase to 100–150 mg/kg/day "
            "over 1–2 weeks. Maximum 200 mg/kg/day. Adults: 1–3 g/day. "
            "REMS Program: VGB is available in USA under SHARE REMS (Risk Evaluation and "
            "Mitigation Strategy) — prescribers must be enrolled; ERG every 3 months mandatory."
        ),
        "moa": (
            "Irreversible GABA-T (GABA transaminase) inhibitor → GABA accumulation in brain "
            "synaptic terminals and extracellular space → enhanced GABAergic inhibition. "
            "Particularly effective in infantile spasms where GABAergic tone is deficient. "
            "Irreversible binding: discontinuation does not restore GABA-T for 5–7 days; "
            "visual field defect risk is cumulative and dose-dependent (VFD in 30–50% "
            "long-term users — irreversible peripheral constriction)."
        ),
        "efficacy": (
            "IS cessation: ~45% in EEF1A2 (similar to ACTH). Combination ACTH + VGB "
            "(UKISS trial) → 73% spasm cessation in TSC-IS; in non-TSC structural/genetic "
            "IS (which includes EEF1A2): 44% combined vs 42% ACTH alone (no additive benefit "
            "demonstrated statistically). NICE recommends: ACTH OR VGB as first-line (not "
            "mandatory combined). Start one → add other if no response at 14 days."
        ),
        "monitoring": (
            "ERG (electroretinogram) every 3 months in infants; fundoscopy 6-monthly. "
            "REMS enrollment (USA). Drowsiness/hypotonia — dose-reduce if severe. "
            "MRI: increased signal in basal ganglia/thalami/brainstem (reversible VGB-related "
            "signal change — MRI-RS) in 20–30% of infants — monitor, usually resolves. "
            "Discontinue VGB if bilateral visual field constriction confirmed on ERG."
        ),
        "eef1a2_specific_notes": (
            "VGB is a reasonable first-line for EEF1A2 IS if ACTH is unavailable or patient "
            "cannot tolerate ACTH side effects. In EEF1A2, the underlying translational defect "
            "means GABAergic augmentation (VGB mechanism) provides symptomatic relief only. "
            "Consider early transition off VGB once IS controlled (minimise cumulative VFD risk). "
            "VGB MRI-RS: increased T2 signal in thalami/BG — document at diagnosis MRI "
            "to distinguish from eEF1A2 structural abnormalities."
        ),
    },
    {
        "treatment": "Valproate (VPA)",
        "level": "Level B",
        "role": "Broad-spectrum AED — backbone for post-IS multifocal DEE and LGS phase; POLG1 screen MANDATORY before starting",
        "dose": (
            "Children: 15–40 mg/kg/day in 2–3 divided doses (chronopharmacology: peak TDM "
            "2h post-dose). Target TDM: 50–100 mg/L (seizure; titrate to response). "
            "Maximum: 60 mg/kg/day. IV VPA available for acute management. "
            "Sodium valproate preferred over VPA acid (better tolerated in children)."
        ),
        "moa": (
            "Multiple mechanisms: (1) Na+ channel inactivation (fast) — reduces high-frequency "
            "firing; (2) Ca2+ channel blockade (T-type) — absence/IS mechanism; (3) GABA-T "
            "inhibition → GABA ↑ (weaker than VGB but reversible); (4) GABA-A potentiation; "
            "(5) HDAC inhibition — epigenetic modulation of neuronal gene expression; "
            "(6) Branched-chain fatty acid metabolism effects on mitochondrial function. "
            "Broadest spectrum of any AED — effective for IS, myoclonic, tonic, atonic, focal."
        ),
        "efficacy": (
            "Myoclonic: 60–70% ≥50% reduction. Drop attacks (LGS): 40% response. "
            "Post-IS multifocal DEE: ~55% meaningful seizure reduction. "
            "Not typically seizure-free but markedly improves seizure control. "
            "Combination VPA + CLB is the most common effective regimen in EEF1A2 LGS phase."
        ),
        "monitoring": (
            "POLG1 gene testing BEFORE starting (POLG1 biallelic LOF + VPA → fatal hepatotoxicity "
            "— Alpers-Huttenlocher syndrome). LFT + FBC + ammonia at baseline, 4 weeks, 3 months, "
            "then 6-monthly. TDM (target 50–100 mg/L). Weight/appetite. Hair thinning (biotin "
            "supplementation helpful). Menstrual irregularity/PCOS risk (females): discuss hormonal "
            "monitoring. Teratogenicity: folic acid 5 mg/day in all females of reproductive age."
        ),
        "eef1a2_specific_notes": (
            "VPA is the BACKBONE of EEF1A2 management post-IS — broadest spectrum for the "
            "multifocal DEE/LGS phenotype. POLG1 screen is MANDATORY before initiating — "
            "this applies to ALL DEE patients regardless of suspected etiology. If POLG1 "
            "mutation found: VPA is ABSOLUTE CI → use LEV + CLB + KD instead. "
            "VPA + KD combination: monitor for additive effects (KD already reduces glycolysis; "
            "VPA inhibits beta-oxidation → monitor liver function closely if combined)."
        ),
    },
    {
        "treatment": "Ketogenic Diet (KD)",
        "level": "Level B",
        "role": "Drug-resistant epilepsy (DRE) in EEF1A2; highly recommended for IS→LGS transition phase; consider early",
        "dose": (
            "Classic KD: 4:1 (fat:carb+protein) or 3:1 ratio. Modified Atkins: 20 g/day "
            "carbohydrate. MCT (medium-chain triglyceride) diet alternative. "
            "Initiation: gradual (outpatient safe); fasting initiation not required. "
            "Target BHB: 2–5 mmol/L (urine ketones 4+). Duration: minimum 3 months "
            "to assess response; continue ≥2 years if effective."
        ),
        "moa": (
            "Multiple mechanisms relevant to EEF1A2: (1) BHB as alternative energy substrate "
            "reduces glycolytic dependence (important in neuronal metabolic stress); (2) "
            "Adenosine A1 receptor activation → neuronal inhibition; (3) mTOR pathway "
            "inhibition (relevant for GOF actin-mTOR class); (4) GABA synthesis enhancement "
            "(BHB → increased GABA:glutamate ratio); (5) HDAC inhibition (BHB as HDAC inhibitor "
            "— epigenetic modulation); (6) Mitochondrial biogenesis improvement; "
            "(7) Reduces reactive oxygen species in hyperexcitable neurons."
        ),
        "efficacy": (
            "IS cessation (KD initiated during/after ACTH): 35–45% — important bridging therapy. "
            "LGS-phase drop attack reduction: 55% ≥50% reduction (best evidence class). "
            "Multifocal myoclonic: 50–60% responder rate. Seizure-free: ~8% (consistent with "
            "other DEE studies). EEF1A2-specific meta-analysis: n=12 patients, KD initiated "
            "mean age 18 months → 58% ≥50% reduction at 12 months (Lam 2016 Neurogenetics)."
        ),
        "monitoring": (
            "BHB (urine ketones 4x/day; serum BHB weekly initially). Lipid profile (3-monthly). "
            "Kidney ultrasound (nephrolithiasis risk: hydration mandatory). "
            "LFT (VPA combination: monitor closely). Growth/weight (dietitian quarterly). "
            "Electrolytes + BMP (Ca, P, Mg, K, Na). Trace minerals (Se, Zn) annually. "
            "Carnitine levels if carnitine-supplemented. Bone density (DEXA if >2 years on KD)."
        ),
        "eef1a2_specific_notes": (
            "EARLY KD initiation is recommended in EEF1A2 (as early as IS diagnosis if ACTH "
            "fails or after ACTH course). Start KD DURING ACTH taper to bridge IS relapse risk. "
            "GOF mTOR class: KD may provide additional benefit via mTOR suppression. "
            "KD + VPA combination: monitor hepatic function monthly (additive metabolic effects). "
            "KD initiation does NOT require hospitalisation in stable EEF1A2 patients; "
            "refer to specialist ketogenic dietitian before start."
        ),
    },
    {
        "treatment": "Clobazam (CLB)",
        "level": "Level B",
        "role": "Adjunct for LGS-phase drop attacks, focal seizures, and myoclonus; second most effective adjunct in EEF1A2 after VPA",
        "dose": (
            "Children <30 kg: 0.1–0.3 mg/kg/day in 1–2 doses (max 10 mg/day). "
            "Children >30 kg / adults: 10–40 mg/day in 2 divided doses. "
            "Titrate slowly (tolerance develops at 6–12 months in 30–40% of patients). "
            "Intermittent use (perimenstrual pulse) can delay tolerance development."
        ),
        "moa": (
            "1,5-benzodiazepine (not 1,4) — acts at the GABA-A benzodiazepine binding site "
            "→ increased Cl− flux → neuronal hyperpolarisation. Less sedating than 1,4-BDZs "
            "(nitrazepam, diazepam) — more selective for α2 and α3 subunits vs α1 (which "
            "mediates sedation/amnesia). Active metabolite N-desmethylclobazam (t½ 71h) "
            "contributes significantly to effect. CLB reduces CYP2C19-metabolised AEDs "
            "including phenytoin (interaction: monitor TDM if combined)."
        ),
        "efficacy": (
            "Drop attacks (LGS): 40–55% ≥50% reduction. Myoclonic: 45% responder. "
            "Focal seizures: 38% responder. Ongoing responders at 2 years: ~55% "
            "(better retention than most AEDs in LGS). Tolerance: dose increase needed "
            "in 30–40% at 12 months; drug holiday (4–6 weeks off) can partially restore "
            "sensitivity in some patients."
        ),
        "monitoring": (
            "Sedation/drowsiness (dose-limiting in EEF1A2 DEE — may impair communication). "
            "Salivary hypersecretion (common in DEE). Weight changes. Respiratory depression "
            "risk in prolonged GTCS — ensure family has rescue BDZ protocol. "
            "CYP2C19 genotype affects N-desmethylclobazam levels — poor metabolisers: "
            "half the usual dose may be effective. Check EFT/LFT (mild hepatic metabolism)."
        ),
        "eef1a2_specific_notes": (
            "CLB is the most valuable adjunct AED in EEF1A2 LGS-phase management after VPA. "
            "Combination VPA + CLB is the clinical workhorse for EEF1A2 multifocal DEE. "
            "Startle epilepsy in EEF1A2: CLB reduces startle reflex effectively. "
            "Beware: CLB causes salivary hypersecretion in non-verbal patients → aspiration "
            "risk if swallowing is already compromised. Salivation management: "
            "glycopyrrolate or hyoscine patches if severe."
        ),
    },
    {
        "treatment": "Levetiracetam (LEV)",
        "level": "Level B",
        "role": "Adjunct AED; second-line for myoclonic and focal seizures; CAUTION for severe behavioural side effects in EEF1A2 DEE",
        "dose": (
            "Children: 10 mg/kg/day increasing to 20–40 mg/kg/day in 2 divided doses. "
            "Maximum 60 mg/kg/day. IV formulation available for acute management. "
            "Renal dose adjustment required (GFR-based). No hepatic induction. "
            "Extended-release once-daily available for older children."
        ),
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) binding → modulates neurotransmitter "
            "release from synaptic vesicles (reduces vesicle docking/fusion). Additional: "
            "inhibits negative modulators of GABA-A/glycine receptors (Zn2+, β-carbolines). "
            "Reduces high-frequency neuronal firing without blocking Na+ channels. "
            "Active as both enantiomers but L-enantiomer (levetiracetam) is pharmacologically "
            "active. Minimal drug-drug interactions (not CYP-metabolised)."
        ),
        "efficacy": (
            "Myoclonic seizures: 40–55% ≥50% reduction. Focal seizures: 35–50%. "
            "IS (infantile spasms): NOT indicated — no evidence of IS efficacy. "
            "In EEF1A2, LEV is used as adjunct when VPA/CLB are inadequate for focal component. "
            "Seizure-free rate: <5% in EEF1A2 DEE."
        ),
        "monitoring": (
            "BEHAVIOURAL TOXICITY: aggression, irritability, hyperactivity are reported in "
            "15–35% of DEE patients (HIGHER than general epilepsy — ~15%). In EEF1A2 DEE, "
            "behavioural escalation severely impacts quality of life. Monitor with parent-reported "
            "behavioural checklists monthly during first 3 months. If irritability worsens: "
            "dose-reduce or switch to brivaracetam (SV2A-selective, lower behavioural burden). "
            "BMP (renal function — dose adjustment). CBC."
        ),
        "eef1a2_specific_notes": (
            "BEHAVIOURAL CAUTION: EEF1A2 DEE patients often have pre-existing irritability and "
            "autistic features — LEV can markedly worsen behavioural profile ('levetiracetam rage'). "
            "Start at low dose (10 mg/kg/day); titrate slowly over 4 weeks. If behavioural "
            "deterioration: first consider brivaracetam (BRV) — same SV2A mechanism but lower "
            "irritability rate (~5%). Pyridoxine 50 mg/day may mitigate LEV-induced irritability "
            "(anecdotal, no RCT in EEF1A2 specifically). NOT a first-choice AED in EEF1A2."
        ),
    },
    {
        "treatment": "Topiramate (TPM)",
        "level": "Level C",
        "role": "Third/fourth-line for DRE; LGS-phase drop attacks; significant cognitive side effects limit use in EEF1A2 DEE",
        "dose": (
            "Children ≥2 years: start 0.5–1 mg/kg/day; increase by 0.5–1 mg/kg/week "
            "to 3–9 mg/kg/day in 2 divided doses. Adults: 100–400 mg/day. "
            "Slow titration is CRITICAL to minimise cognitive side effects. "
            "Avoid in patients with kidney stones history (carbonic anhydrase inhibition → "
            "nephrolithiasis risk). Adequate hydration mandatory."
        ),
        "moa": (
            "Multiple: (1) Na+ channel inactivation — reduces high-frequency firing; "
            "(2) AMPA/kainate receptor antagonism — blocks excitatory glutamate transmission; "
            "(3) GABA-A positive allosteric modulation at a non-BDZ site; "
            "(4) Carbonic anhydrase II/IV inhibition — mild metabolic acidosis (may contribute "
            "to anti-seizure effect via changes in neuronal pH); (5) Voltage-gated Ca2+ channel "
            "modulation. Broad-spectrum but constrained by cognitive tolerability."
        ),
        "efficacy": (
            "LGS drop attacks: ~35% ≥50% reduction (Level C evidence). "
            "Generalised seizures: ~30% responder. Very limited data specific to EEF1A2. "
            "Cognitive impact: SIGNIFICANT — IQ reduction documented in controlled trials "
            "(word-finding, verbal fluency) — particularly problematic in EEF1A2 DEE "
            "patients who are already at the limit of communicative function."
        ),
        "monitoring": (
            "Cognitive testing (ADOS-2 behaviour rating, clinician global impression) at baseline "
            "and 3 months. Weight (significant weight loss in 5–10% — monitor in underweight "
            "children). Metabolic acidosis: bicarb levels (TPM inhibits carbonic anhydrase → "
            "metabolic acidosis can worsen growth and respiratory function). "
            "Nephrolithiasis: fluid intake ≥1.5L/day; urinalysis annually. "
            "Acute myopia/closed-angle glaucoma: ophthalmology urgently if eye pain."
        ),
        "eef1a2_specific_notes": (
            "COGNITIVE CAUTION: EEF1A2 DEE patients are already severely cognitively impaired — "
            "TPM further reduces communicative ability, which families find particularly distressing. "
            "Use ONLY after VPA, CLB, LEV, and KD have been trialled. Slow titration (maximum "
            "0.5 mg/kg/week) reduces cognitive toxicity. If trial > 3 months without ≥30% "
            "seizure reduction: discontinue (evidence threshold). "
            "LGS drop attack: consider rufinamide instead (Level B, less cognitive burden)."
        ),
    },
    {
        "treatment": "Fenfluramine (FFA)",
        "level": "Level C",
        "role": "Emerging DRE option in EEF1A2 DEE; off-label beyond Dravet/LGS; FDA/EMA approved only for Dravet (2020) and LGS (2022)",
        "dose": (
            "Currently approved dosing for Dravet/LGS: 0.1–0.35 mg/kg/day (max 26 mg/day) "
            "in 2 divided doses. REMS Program (Fintepla REMS): mandatory echocardiography "
            "before start, at 6 months, then annually (cardiac valvulopathy/PHT risk). "
            "Not approved in children <2 years. Must be used with risk agreement. "
            "Off-label for EEF1A2: weight of evidence insufficient for routine use."
        ),
        "moa": (
            "FFA releases serotonin from presynaptic terminals (potent 5-HT releasing agent) "
            "AND activates sigma-1 receptors (modulates NMDA and sigma-1/IP3R axis) — dual "
            "anti-seizure mechanism distinct from all other AEDs. Serotonergic pathway reduces "
            "cortical excitability via 5-HT1A and 5-HT2C receptor activation. Sigma-1R: "
            "reduces IP3R-mediated Ca2+ release from ER — reduces cytoplasmic Ca2+ oscillations "
            "contributing to seizure initiation. Originally used as an appetite suppressant "
            "(withdrawn 1997 due to cardiac valvulopathy risk); reformulated at lower doses for "
            "epilepsy (Zogenix)."
        ),
        "efficacy": (
            "Dravet syndrome (FINTEPLA trial): 62% ≥50% reduction; 6.5% seizure-free. "
            "LGS (FDA Brabio trial): 26% ≥50% reduction (placebo 13%). "
            "EEF1A2 DEE: case series n=4 (non-published) — 2/4 >50% reduction in GTCS. "
            "Insufficient evidence for routine use in EEF1A2 — consider only after "
            "VPA/CLB/LEV/KD/VGB have all failed and cardiac evaluation is normal."
        ),
        "monitoring": (
            "REMS PROGRAM: echocardiography at baseline, 6m, then annually (pulmonary "
            "arterial hypertension + cardiac valvulopathy risk). Appetite suppression/weight "
            "loss: monitor weekly initially. Serotonin syndrome risk if combined with SSRIs/MAOIs. "
            "Blood pressure (serotonergic stimulation). NOT for patients with cardiac structural "
            "abnormalities. Prescriber must be enrolled in FINTEPLA REMS."
        ),
        "eef1a2_specific_notes": (
            "EXPERIMENTAL in EEF1A2 — use only after all evidence-based options exhausted. "
            "The serotonergic mechanism of FFA is mechanistically plausible in EEF1A2 "
            "(serotonin pathways interact with eEF1A2's actin/dendritic spine roles), but "
            "no controlled data exists. Off-label use requires ethics committee/IRB consideration. "
            "Cardiac monitoring is non-negotiable — the risk/benefit decision must include "
            "a paediatric cardiologist. NOT recommended as standard EEF1A2 therapy (2026 evidence)."
        ),
    },
]

# ── Contraindications (6) ─────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Phenytoin (PHT) / Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Lamotrigine (LTG)",
        "severity": "HIGH RISK",
        "reason": (
            "Sodium channel blockers (PHT, CBZ, OXC) WORSEN generalised seizure types in EEF1A2 DEE — "
            "particularly myoclonic seizures and infantile spasms. Mechanism: PHT/CBZ selectively "
            "suppress gamma-aminobutyric acid (GABA) interneuron activity preferentially in high-"
            "frequency context, paradoxically increasing generalised spread. LTG (Na+ channel blocker "
            "with AMPA modulation) also worsens myoclonic seizures in LGS/DEE context. "
            "Pattern: prescribers unfamiliar with DEE sometimes use PHT/CBZ as 'default AED' — "
            "this can catastrophically worsen seizure burden in EEF1A2."
        ),
        "alternative": "VPA (broad-spectrum) + CLB (adjunct) + KD — safe combination for EEF1A2 multifocal DEE.",
    },
    {
        "drug": "Tiagabine (TGB)",
        "severity": "ABSOLUTE CONTRAINDICATION",
        "reason": (
            "TGB (GAT-1 GABA-reuptake inhibitor) causes non-convulsive status epilepticus (NCSE) "
            "in generalised epilepsies including DEE. Mechanism: in normal synapses, GAT-1 limits "
            "GABA duration; blockade → extracellular GABA accumulation → GABA-A and GABA-B "
            "receptor desensitisation/internalization → paradoxical excitation + NCSE. "
            "This risk is HIGHEST in children with pre-existing generalised epilepsy (EEF1A2, "
            "Lennox-Gastaut, Dravet). NEVER use TGB in EEF1A2 DEE. NCSE caused by TGB "
            "may not be identified without continuous EEG — it presents as cognitive decline or "
            "behavioural worsening rather than convulsive seizures."
        ),
        "alternative": "CLB (GABA-A positive modulator, not reuptake inhibitor) is safe. VGB for IS.",
    },
    {
        "drug": "Valproate (VPA) without prior POLG1 screening",
        "severity": "ABSOLUTE CONTRAINDICATION (if POLG1 not cleared)",
        "reason": (
            "POLG1 biallelic pathogenic variants + VPA → Alpers-Huttenlocher syndrome: acute "
            "hepatic failure, often fatal. EEF1A2 DEE patients have NOT been systematically "
            "excluded from POLG1 carrier status — POLG1 mutations affect ~1% of the general "
            "population and ~10% of refractory paediatric epilepsies. VPA + undetected POLG1 "
            "biallelic disease = potentially lethal hepatotoxicity within 3–6 months. "
            "POLG1 testing must be complete (both alleles) before any VPA prescription. "
            "In urgent situations: start LEV + CLB while awaiting POLG1 result."
        ),
        "alternative": "LEV + CLB while awaiting POLG1 result (7–14 days turnaround). After negative POLG1: VPA safe to start.",
    },
    {
        "drug": "Phenobarbitone (PHB) as sole IS treatment",
        "severity": "NOT ADEQUATE for infantile spasms — use as BRIDGE ONLY",
        "reason": (
            "PHB is NOT a first-line treatment for infantile spasms in EEF1A2 (or any etiology). "
            "PHB suppresses EEG discharge amplitude (may superficially appear to reduce spasms) "
            "without resolving hypsarrhythmia — clinicians may falsely believe IS are 'controlled'. "
            "Continued PHB without ACTH/VGB delays the effective treatment window → worse "
            "neurodevelopmental outcome. PHB IS used as a BRIDGE (neonatal SE protocol, ICU "
            "management of acute status) but MUST be followed by ACTH/VGB within 48 hours. "
            "PHB MONOTHERAPY for IS = inadequate treatment — unacceptable in 2026."
        ),
        "alternative": "ACTH (0.5–1 mg/1.5 mg alternate days) OR Vigabatrin (50→150 mg/kg/day) as first-line.",
    },
    {
        "drug": "Levetiracetam (LEV) — HIGH BEHAVIOURAL RISK in EEF1A2 DEE",
        "severity": "HIGH RISK — behavioural toxicity",
        "reason": (
            "NOT a formal absolute contraindication but a HIGH-RISK drug class requiring explicit "
            "consent in EEF1A2 DEE. LEV causes aggression, irritability, and severe behavioural "
            "deterioration in 15–35% of DEE patients — substantially higher than in focal epilepsy "
            "(3–8%). EEF1A2 patients have pre-existing irritability, autistic features, and "
            "behavioural dysregulation; LEV can create a catastrophic escalation that is mistaken "
            "for disease progression. Families must be explicitly counselled before initiating: "
            "'This drug can cause severe mood and behaviour changes — contact us immediately if "
            "your child becomes aggressive or inconsolable.' Consider brivaracetam (BRV) as "
            "preferred SV2A alternative in EEF1A2 DEE (lower irritability rate)."
        ),
        "alternative": "Brivaracetam (BRV) — same SV2A mechanism, substantially lower irritability rate (~5%).",
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) + HLA-B*15:02 without screening in high-risk ancestry",
        "severity": "HLA-B*15:02 SCREENING REQUIRED in South/Southeast Asian ancestry before CBZ/OXC",
        "reason": (
            "Although CBZ/OXC are AVOIDED in EEF1A2 DEE due to seizure-worsening risk (CI #1), "
            "if ever prescribed (e.g. error, or for comorbid pain management), HLA-B*15:02 screening "
            "is mandatory in South Asian, Southeast Asian, and Han Chinese patients. HLA-B*15:02 "
            "carriers have ~6% risk of Stevens-Johnson syndrome / toxic epidermal necrolysis (SJS/TEN) "
            "with CBZ/OXC. CPIC 2023 guideline: if HLA-B*15:02 positive → NEVER prescribe CBZ/OXC/PHT. "
            "While this CI overlaps with the seizure-worsening CI, it is listed separately as it "
            "applies to ANY context where CBZ might be considered (e.g. carers with EEF1A2 comorbidity)."
        ),
        "alternative": "Avoid CBZ/OXC entirely in EEF1A2. Carbamazepine has no role in EEF1A2 management.",
    },
]

# ── Monitoring (14 items) ─────────────────────────────────────────────────────
MONITORING_ITEMS = [
    "WES (Whole Exome Sequencing) / Epilepsy gene panel including EEF1A2 — diagnostic confirmation; maternal and paternal TRIO-WES preferred to confirm de novo status",
    "POLG1 gene testing (biallelic LOF) BEFORE starting VPA — mandatory screening protocol; turnaround 7–14 days; initiate LEV+CLB bridge while awaiting",
    "EEG at IS onset (hypsarrhythmia baseline), at 14 days post-ACTH (response criterion), at 28 days, and every 6 months (LGS evolution monitoring) — continuous EEG during acute SE",
    "VGB ERG (electroretinogram) and fundoscopy every 3 months in infants on VGB; visual field testing 6-monthly in older children — REMS-mandated; discontinue VGB if bilateral VFD confirmed",
    "MRI brain 3T with epilepsy protocol (axial T1, FLAIR, DWI, T2*GRE) at diagnosis — cortical malformations (polymicrogyria, pachygyria, thin CC); repeat at LGS transition or if new focal deficit",
    "Developmental assessment: Bayley Scales of Infant and Toddler Development (Bayley-4) in <42 months; Vineland Adaptive Behavior Scales-3 (VABS-3) 6-monthly — track developmental trajectory",
    "Ophthalmology assessment: cortical visual impairment (CVI), nystagmus, strabismus, optic atrophy — at diagnosis and annually; VGB ERG at same appointment if on VGB",
    "ACTH monitoring during induction: BP (daily), blood glucose (BD), electrolytes (Na/K) weekly, weight, infection screen — document response at Day 14 (spasm cessation, EEG hypsarrhythmia resolution)",
    "VPA therapeutic drug monitoring (TDM): target 50–100 mg/L; LFT + FBC + serum ammonia at baseline, 4 weeks, 3 months, then 6-monthly; hair thinning — biotin supplementation",
    "KD monitoring (if on ketogenic diet): BHB (serum/urine 4×/day initially), lipid profile 3-monthly, kidney ultrasound annually, electrolytes (K, Na, Mg, Ca, P), trace minerals (Se, Zn) annually, growth/weight monthly with dietitian",
    "Audiology: hearing screen (ABR/OAE) at diagnosis — cortical auditory processing disorder common; school-age: cortical auditory processing assessment",
    "Genetics counselling: de novo EEF1A2 variants → recurrence risk <1% (de novo rate) for future siblings; siblings NOT at elevated risk unless germline/somatic mosaicism found in parent; reproductive planning discussion",
    "SUDEP risk stratification: nocturnal convulsive seizure monitoring (seizure alarm system — Empatica, SAMi-3, NightWatch); avoid prone sleeping; rescue benzodiazepine training for caregivers; SUDEP conversation documented",
    "Driving/VPPP/transition (adolescence): DVLA/Transport Canada regulations; VGB visual impairment may affect eligibility; VPPP (valproate pregnancy prevention programme) for all EEF1A2 females initiated on VPA",
]

# ── Lifecycle Stages (6) ──────────────────────────────────────────────────────
LIFECYCLE_STAGES = [
    {
        "stage": "Stage 1: Pre-Switch Window (Neonatal–3 months)",
        "age_range": "0–3 months",
        "description": (
            "eEF1A1 is ACTIVE — fetal/neonatal neurons use the ubiquitous eEF1A1 isoform. "
            "In EEF1A2 LOF, neonates appear NORMAL (unlike PNPO/ALDH7A1 where neonatal seizures occur). "
            "KEY OPPORTUNITY: This is the 'pre-symptomatic window'. If EEF1A2 is identified through "
            "sibling diagnosis or incidental NICU WES, this window allows: (1) baseline "
            "developmental assessment, (2) family education, (3) ACTH/KD readiness planning, "
            "(4) participation in natural history studies. Future: pre-switch AAV gene therapy "
            "(theoretically possible, under investigation). Goals: genetic confirmation, "
            "neurodevelopment baseline, family support, anticipatory guidance."
        ),
    },
    {
        "stage": "Stage 2: IS Onset (3–12 months)",
        "age_range": "3–12 months",
        "description": (
            "eEF1A1→eEF1A2 postnatal switch occurs ~3–6 months → eEF1A2 LOF causes neuronal "
            "translation failure → Infantile Spasms (West syndrome) onset. Hypsarrhythmia on EEG. "
            "CRITICAL: diagnose IS within 2 weeks of onset; initiate ACTH (first-line) within 4 weeks "
            "of IS onset — neurodevelopmental outcome is time-sensitive. Order TRIO-WES + POLG1 "
            "simultaneously with ACTH initiation. Start KD during ACTH taper. "
            "Goals: spasm cessation, hypsarrhythmia resolution, developmental rescue, "
            "prevent epileptic encephalopathy progression."
        ),
    },
    {
        "stage": "Stage 3: Post-IS Multifocal DEE (12–36 months)",
        "age_range": "12–36 months",
        "description": (
            "IS resolved (partially or fully) but multifocal epileptic encephalopathy evolves. "
            "Seizure types: myoclonic + GTCS + focal. VPA is the backbone AED. KD continued. "
            "Developmental delay becomes pronounced: motor (hypotonia, delayed walking >18 months), "
            "language (non-verbal in 80%), autistic features emerge (early ABA therapy recommended). "
            "MDT: neurology + paediatrics + physiotherapy + OT + SALT + psychology. "
            "Goals: optimise AED regimen, KD continuation, early intervention therapy, "
            "caregiver training (seizure rescue, SUDEP awareness), specialist school placement."
        ),
    },
    {
        "stage": "Stage 4: LGS Evolution (4–12 years)",
        "age_range": "4–12 years",
        "description": (
            "Lennox-Gastaut syndrome pattern: slow spike-wave (<2.5 Hz), tonic seizures (NREM), "
            "drop attacks (atonic + tonic). Helmets mandatory. VPA + CLB backbone; consider "
            "rufinamide for drop attacks (Level B LGS evidence). KD continuation or re-trial if "
            "discontinued. Pre-surgical evaluation in selected DRE patients (SEEG, fMRI, MEG). "
            "VNS consideration for DRE (Level B evidence in LGS). School: specialist educational "
            "provision, AAC (augmentative and alternative communication) for non-verbal patients. "
            "Goals: minimise drop attack injury, optimise cognitive function, school inclusion, "
            "caregiver psychological support."
        ),
    },
    {
        "stage": "Stage 5: Adolescence — DRE Management (13–18 years)",
        "age_range": "13–18 years",
        "description": (
            "Continued DRE; seizure severity may plateau or worsen at puberty. SUDEP risk is "
            "highest in this age group (unsupervised nights, puberty-related sleep disruption). "
            "Key issues: puberty (catamenial pattern if female — perimenstrual CLB pulse), "
            "VPPP (valproate pregnancy prevention for females on VPA), transition to adult services "
            "planning, driving regulations (unlikely to qualify but must be counselled), "
            "VGB visual fields formal testing annually. Carer strain peaks — respite care referral "
            "essential. Consider fenfluramine (off-label) if all else failed. "
            "Goals: SUDEP prevention, transition planning, caregiver wellbeing."
        ),
    },
    {
        "stage": "Stage 6: Adulthood — Chronic DEE (18+ years)",
        "age_range": "18+ years",
        "description": (
            "Profound intellectual disability (most EEF1A2 adults are non-verbal, dependent care). "
            "Seizures typically persist but may reduce in frequency. AED continuation: VPA + CLB ± "
            "LEV ± KD. Transition to adult neurology clinic (epilepsy + learning disabilities team). "
            "Residential care/day placement planning. Gastrostomy consideration if swallowing "
            "impaired (aspiration pneumonia risk). Legal: capacity assessment, power of attorney, "
            "best interests decisions. Palliative care planning (SUDEP, aspiration, status). "
            "Caregiver support: rare disease support groups (EEF1A2 Research Foundation). "
            "Goals: quality of life, SUDEP prevention, caregiver support, end-of-life planning."
        ),
    },
]

# ── Thresholds (12) ───────────────────────────────────────────────────────────
THRESHOLDS = {
    "IS_treatment_window": "≤4 weeks from IS onset to ACTH/VGB initiation — critical for neurodevelopment (UKISS trial)",
    "ACTH_response_criterion": "EEG hypsarrhythmia resolution AND clinical spasm cessation at Day 14 (not Day 7)",
    "EEF1A2_onset_window": "3–9 months (eEF1A1→eEF1A2 postnatal switch timing — diagnostic clock)",
    "VPA_TDM_target": "50–100 mg/L (seizure suppression); >100 mg/L → toxicity risk (encephalopathy, tremor)",
    "KD_target_BHB": "2–5 mmol/L serum BHB (urine ketones 4+ equivalent); check at 2 weeks post-initiation",
    "VGB_dose_max": "150–200 mg/kg/day in infants (above this: sedation > efficacy)",
    "ACTH_BP_limit": "Systolic >130 mmHg in infant → dose-reduce ACTH; >150 mmHg → hold and treat",
    "POLG1_turnaround": "7–14 days for POLG1 gene result — bridge with LEV+CLB during this window",
    "SUDEP_high_risk_threshold": "≥3 nocturnal GTCS/year in EEF1A2 DEE → seizure alarm system MANDATORY",
    "KD_lipids_threshold": "LDL >4.5 mmol/L (174 mg/dL) → KD modification (reduce saturated fat ratio)",
    "VGB_VFD_discontinue": "Bilateral VFD on ERG → discontinue VGB regardless of seizure status",
    "CLB_tolerance_escalation": "Dose increase >50% from original effective dose = tolerance — consider drug holiday (4–6 weeks) to restore sensitivity",
}

# ── Evidence Standards (12) ───────────────────────────────────────────────────
EVIDENCE_STANDARDS = [
    "ILAE-2022 Definition and Classification of Epilepsy Syndromes — DEE classification framework",
    "NICE NG217 (2022) Epilepsies in Children, Young People and Adults — UK IS management protocol",
    "UKISS Trial (Lux AL 2004 Lancet; O'Callaghan 2017 Lancet Neurol) — ACTH vs prednisolone for IS; definitive first-line evidence",
    "Nakajima J, et al. 2015 Hum Mutat — First description of EEF1A2 de novo mutations in DEE; series of 4 patients",
    "Lam WW, et al. 2016 Neurogenetics — EEF1A2 phenotypic expansion; KD response data; IS→LGS evolution",
    "Bhatt DL, et al. 1999 J Neurosci — eEF1A1→eEF1A2 postnatal switch mechanism in murine neurons (foundational mechanistic paper)",
    "CPIC HLA-B*15:02 and Carbamazepine Guideline 2023 — HLA typing mandatory before CBZ/OXC in high-risk ancestry",
    "CPIC POLG1 and Valproate 2023 — POLG1 testing before VPA in paediatric epilepsy",
    "FDA FINTEPLA REMS (Fenfluramine) — cardiac monitoring protocol for fenfluramine use",
    "MHRA VPPP (Valproate Pregnancy Prevention Programme) 2021 — mandatory female VPA monitoring",
    "ACMG/AMP Variant Classification Standards 2015 (Richards et al.) — de novo EEF1A2 pathogenicity framework",
    "NICE NG224 (2023) Epilepsies: Update — SUDEP risk reduction; seizure alarms; rescue medication protocols",
]

# ── References (6) ────────────────────────────────────────────────────────────
REFERENCES = [
    {
        "citation": "Nakajima J, et al. A de novo missense mutation in EEF1A2 identified by whole exome sequencing in intractable epilepsy with intellectual disabilities. Am J Med Genet A. 2015;167(11):2706-2711.",
        "pmid": "26178227",
        "key_finding": "First report of EEF1A2 de novo missense in DEE. Whole exome sequencing identified causal variant. Clinical description of infantile spasms → LGS evolution. Founding paper for EEF1A2 epilepsy.",
    },
    {
        "citation": "Lam WW, et al. Novel de novo EEF1A2 missense mutations causing epileptic encephalopathy and intellectual disability. Neurogenetics. 2016;17(3):165-176.",
        "pmid": "27041368",
        "key_finding": "Series expansion (n=9); phenotypic breadth defined; KD response data; IS onset timing correlated with eEF1A1→eEF1A2 switch; domain II/III variants associated with actin-bundling defects.",
    },
    {
        "citation": "Bhatt DL, et al. Expression of neuronal elongation factor EEF1A2 is switch-like and restricted to post-mitotic cells. J Neurosci. 1999;19(4):1396-1404.",
        "pmid": "9952415",
        "key_finding": "Definitive characterisation of the postnatal eEF1A1→eEF1A2 switch in murine CNS. Timing: 2–4 weeks postnatal in mice (equivalent ~3–6 months human). Explains why EEF1A2 disease is postnatal and tissue-specific.",
    },
    {
        "citation": "de Ligt J, et al. Diagnostic exome sequencing in persons with severe intellectual disability. N Engl J Med. 2012;367(20):1921-1929.",
        "pmid": "23033978",
        "key_finding": "Large WES study confirming de novo mutations as the primary cause of severe ID and DEE including EEF1A2; EEF1A2 identified in cohort; established WES as diagnostic standard for DEE.",
    },
    {
        "citation": "Lux AL, et al.; UKISS Collaborative Group. The United Kingdom Infantile Spasms Study (UKISS): a multicentre, randomised trial. Lancet Neurol. 2005;4(7):395-405.",
        "pmid": "15978524",
        "key_finding": "ACTH and hormonal treatment for IS: spasm cessation at 14 days 74% (hormones) vs 40% (VGB). Definitive evidence for ACTH superiority in non-TSC IS. Standard protocol used in EEF1A2 IS management.",
    },
    {
        "citation": "Symonds JD, et al. Incidence and phenotypes of childhood-onset genetic epilepsies: a prospective population-based national cohort. Brain. 2019;142(8):2303-2318.",
        "pmid": "31292444",
        "key_finding": "Population incidence of genetic DEEs including EEF1A2; confirms rarity (0.02–0.05/100,000/year); WES yield in DEE 47%; EEF1A2 among top 15 de novo DEE genes. Evidence for WES-first diagnostic approach.",
    },
]

# ── Key Concepts (15) ────────────────────────────────────────────────────────
KEY_CONCEPTS = {
    "EEF1A2-20q13.33-DEE-5": (
        "EEF1A2 (Eukaryotic Translation Elongation Factor 1-Alpha 2) maps to chromosome 20q13.33. "
        "Pathogenic de novo variants cause DEE-5 (OMIM #616577). Gene: 1.4 kb CDS, 9 exons, 463 aa, 50.1 kDa protein."
    ),
    "eEF1A1-eEF1A2-Postnatal-Neuronal-Switch": (
        "eEF1A1 (ubiquitous) → eEF1A2 (neuron+muscle only) switch occurs postnatally at ~3–6 months in human neurons. "
        "Before the switch: eEF1A1 compensates → no seizures. After switch: eEF1A2 LOF → translation failure → IS onset. "
        "The switch timing IS the diagnostic clock for EEF1A2 epilepsy."
    ),
    "GTP-Binding-Aminoacyl-tRNA-Delivery": (
        "eEF1A2 binds GTP and aminoacyl-tRNA → delivers aa-tRNA to the ribosomal A-site (codon decoding step). "
        "After GTP hydrolysis, eEF1A2·GDP is recycled by the eEF1B complex. Domain I (G-domain) contains the "
        "GTP-binding P-loop (GXXXXGKS) essential for GTPase cycle."
    ),
    "Actin-Bundling-Non-Canonical-Synaptopathy": (
        "Domain III of eEF1A2 bundles F-actin filaments in dendritic spines — a non-canonical moonlighting function. "
        "LOF/missense in domain III disrupts dendritic spine morphology → synaptic plasticity impaired → "
        "synaptopathy contributing to intellectual disability independent of seizure burden."
    ),
    "DEE-5-OMIM-616577": (
        "DEE-5 = Developmental and Epileptic Encephalopathy type 5 (OMIM #616577). Hallmarks: IS onset 3–9 months; "
        "multifocal DEE; LGS evolution; profound ID; ASD features; non-verbal (80%). Rare: ~150–200 cases worldwide (2026)."
    ),
    "Hypsarrhythmia-West-IS-Diagnostic-Clock": (
        "Infantile spasms with hypsarrhythmia: pathognomonic EEG of West syndrome. In EEF1A2, hypsarrhythmia onset "
        "at 3–9 months coincides exactly with the eEF1A1→eEF1A2 postnatal switch — this timing IS the 'diagnostic clock'. "
        "EEG: chaotic, high-amplitude multifocal spikes + slow waves, interhemispheric asynchrony."
    ),
    "Polymicrogyria-Cortical-Malformation": (
        "Cortical malformations in ~30% of EEF1A2 patients: polymicrogyria (most common), pachygyria, thin corpus "
        "callosum, simplified gyral pattern. eEF1A2 is expressed during cortical plate formation — LOF during "
        "neuronal migration/differentiation may cause structural malformations in addition to functional epilepsy. "
        "MRI 3T with epilepsy protocol mandatory."
    ),
    "ACTH-First-Line-IS-Level-A": (
        "ACTH (tetracosactide) or high-dose prednisolone is Level A first-line for infantile spasms (UKISS trial). "
        "IS cessation at 14 days: 50% in EEF1A2 (lower than idiopathic IS due to underlying translational defect). "
        "Initiation within 4 weeks of IS onset is time-critical for neurodevelopment."
    ),
    "VGB-REMS-ERG-Mandatory-Monitoring": (
        "Vigabatrin (VGB) causes irreversible peripheral visual field defects (VFD) in 30–50% long-term users "
        "via retinal cone/rod degeneration. REMS program mandatory in USA. ERG every 3 months in infants. "
        "Bilateral VFD on ERG = discontinue VGB regardless of seizure status."
    ),
    "LGS-Evolution-Drop-Attacks": (
        "~60% of EEF1A2 IS patients evolve to Lennox-Gastaut syndrome at age 18m–4y. LGS features: slow "
        "spike-wave <2.5 Hz, tonic nocturnal seizures, atonic/tonic drop attacks (MANDATORY helmet). "
        "VPA + CLB backbone; rufinamide/fenfluramine Level B-C evidence for drops. KD highly effective."
    ),
    "POLG1-VPA-Mandatory-Screening": (
        "POLG1 biallelic variants + VPA = Alpers-Huttenlocher syndrome (fatal hepatotoxicity). "
        "POLG1 testing is MANDATORY before VPA in ALL children with refractory epilepsy regardless of suspected etiology. "
        "CPIC 2023: if POLG1 positive → VPA is ABSOLUTE CONTRAINDICATION. Bridge with LEV+CLB while awaiting result."
    ),
    "PHT-CBZ-OXC-Worsen-Myoclonic-DEE": (
        "Sodium channel blockers (phenytoin, carbamazepine, oxcarbazepine) paradoxically WORSEN generalised and "
        "myoclonic seizures in DEE/LGS patients. Avoid in EEF1A2. LTG similarly worsens myoclonic burden. "
        "Document in emergency care plan: 'AVOID PHT/CBZ/OXC/LTG — contraindicated in this patient's epilepsy.'"
    ),
    "LEV-Behavioural-Toxicity-DEE": (
        "Levetiracetam causes severe behavioural toxicity (aggression, irritability, 'levetiracetam rage') in "
        "15–35% of DEE patients — substantially higher than in focal epilepsy. In EEF1A2 with pre-existing "
        "autistic features, LEV can cause catastrophic behavioural deterioration. "
        "Preferred alternative: brivaracetam (same SV2A mechanism, lower irritability ~5%)."
    ),
    "TGB-Absolute-CI-NCSE": (
        "Tiagabine (GAT-1 inhibitor) is an ABSOLUTE CONTRAINDICATION in EEF1A2 and ALL generalised epilepsies. "
        "Mechanism: GABA reuptake blockade → receptor desensitisation → paradoxical excitation → NCSE. "
        "NCSE in EEF1A2 may be missed without EEG (presents as cognitive decline, not convulsions)."
    ),
    "SUDEP-High-Risk-EEF1A2": (
        "SUDEP (Sudden Unexpected Death in Epilepsy) risk in EEF1A2 DEE is HIGH (~1/200/year in DRE cohorts). "
        "Risk factors: uncontrolled nocturnal GTCS, prone sleeping, seizure-induced respiratory depression. "
        "Mandatory: seizure alarm systems (NightWatch, SAMi-3, Empatica E4), carer education, "
        "avoid prone sleeping, rescue BDZ protocol, SUDEP conversation documented in notes."
    ),
}


# ── Public API Functions ───────────────────────────────────────────────────────
def get_overview() -> dict:
    total = len(PATIENTS)
    seizure_free = sum(1 for p in PATIENTS if p.get("currently_seizure_free"))
    with_poly = sum(1 for p in PATIENTS if "Polymicrogyria" in (p.get("mri") or ""))
    is_pct = round(SEIZURE_CATALOG[0]["pct"])
    return {
        "dashboard": "EEF1A2 Epilepsy — Developmental and Epileptic Encephalopathy-5 (DEE-5)",
        "gene": "EEF1A2",
        "locus": "20q13.33",
        "omim_gene": "*602959",
        "omim_phenotype": "#616577",
        "protein": (
            "Eukaryotic Translation Elongation Factor 1-Alpha 2 (eEF1A2) — GTP-binding protein "
            "(463 aa, 50.1 kDa); neuron+muscle specific; delivers aminoacyl-tRNA to ribosomal A-site; "
            "replaces eEF1A1 postnatally in neurons at ~3–6 months; domain III bundles F-actin. 20q13.33."
        ),
        "syndrome": "DEE-5 / EEF1A2-Related Developmental and Epileptic Encephalopathy",
        "inheritance": "De novo dominant (>90%); rare biallelic AR (consanguineous families)",
        "total_patients": total,
        "infantile_spasms_pct": is_pct,
        "seizure_free_pct": round(seizure_free / total * 100),
        "polymicrogyria_pct": round(with_poly / total * 100),
        "non_verbal_pct": 80,
        "etiology_breakdown": [{"label": e["category"], "n": e["n"], "pct": e["pct"]} for e in ETIOLOGY_CATALOG],
        "critical_pearl": (
            "The eEF1A1→eEF1A2 postnatal neuronal switch (~3–6 months) IS the diagnostic clock: "
            "neonates are normal (eEF1A1 active); IS onset at 3–9 months coincides exactly with the "
            "switch. NO in-utero seizures (unlike PNPO/ALDH7A1). ACTH must be initiated within 4 weeks "
            "of IS onset — delays in diagnosis worsen neurodevelopmental outcome irreversibly."
        ),
        "treatment_hierarchy": [
            "ACTH (IS — first-line)", "VGB (IS alternative)", "VPA (post-IS DEE backbone)",
            "Ketogenic Diet (DRE + IS bridge)", "CLB (adjunct LGS)", "LEV (adjunct — behavioural caution)",
        ],
        "absolute_CIs": ["TGB (NCSE)", "VPA without POLG1 screen", "PHT/CBZ/OXC (worsens myoclonus)"],
        "key_biomarkers": [
            "EEF1A2 de novo variant on WES (diagnostic)", "eEF1A2 protein level on Western blot (research)",
            "MRI cortical malformations (30%)", "EEG hypsarrhythmia (IS phase)",
        ],
        "etiologies": [{"class": e["category"], "pct": e["pct"]} for e in ETIOLOGY_CATALOG],
    }


def get_breakdown() -> dict:
    return {
        "patients": PATIENTS,
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_CATALOG,
        "triggers": TRIGGER_CATALOG,
        "treatments": TREATMENT_CATALOG,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING_ITEMS,
        "lifecycle": LIFECYCLE_STAGES,
        "thresholds": THRESHOLDS,
        "evidence_standards": EVIDENCE_STANDARDS,
        "references": REFERENCES,
    }


def get_definitions() -> dict:
    return {
        "gene": "EEF1A2",
        "full_name": "Eukaryotic Translation Elongation Factor 1-Alpha 2",
        "locus": "20q13.33",
        "omim_gene": "*602959",
        "omim_phenotype": "#616577",
        "protein": (
            "GTP-binding translation elongation factor (463 aa, 50.1 kDa); neuron and skeletal muscle specific; "
            "delivers aminoacyl-tRNA to the ribosomal A-site in the elongation cycle; three domains: "
            "G-domain (GTP binding/hydrolysis), Domain II (tRNA accommodation), Domain III (actin bundling). "
            "Replaces eEF1A1 postnatally at ~3–6 months in human neurons."
        ),
        "enzyme_class": "GTPase (EC 3.6.5.3) — translation elongation factor",
        "syndrome": {
            "DEE-5": "Developmental and Epileptic Encephalopathy-5 (OMIM #616577) — de novo dominant EEF1A2 variants",
            "West_Syndrome_phase": "Infantile spasms with hypsarrhythmia at 3–9 months — coincides with eEF1A1→eEF1A2 switch",
            "LGS_evolution": "~60% of EEF1A2 IS patients evolve to Lennox-Gastaut syndrome (slow SW <2.5 Hz, tonic, drop attacks)",
            "DEE_class": "Non-syndromic DEE — no consistent dysmorphic features; diagnosis requires WES/gene panel",
        },
        "concepts": KEY_CONCEPTS,
        "thresholds": THRESHOLDS,
        "evidence_standards": EVIDENCE_STANDARDS,
        "key_pharmacological_distinctions": [
            "eEF1A1→eEF1A2 POSTNATAL SWITCH IS THE DIAGNOSTIC CLOCK: neonates appear normal (eEF1A1 active); "
            "IS onset at 3–9 months = switch timing. Unlike PNPO (neonatal onset) or SCN1A (febrile trigger at 5–9 months). "
            "Any infant with IS onset in this window and no structural/metabolic cause → WES including EEF1A2 mandatory.",
            "ACTH IS FIRST-LINE but lower efficacy in EEF1A2 (~50%) than idiopathic IS (~80%): the underlying "
            "translational defect persists after ACTH — ACTH suppresses excitability but does NOT restore eEF1A2 function. "
            "Do NOT extend ACTH beyond 14 days without EEG evidence of hypsarrhythmia resolution.",
            "POLG1 SCREEN BEFORE VPA IS NON-NEGOTIABLE: EEF1A2 patients are NOT pre-selected to be POLG1-negative. "
            "POLG1 biallelic disease + VPA = Alpers-Huttenlocher hepatotoxicity (fatal). Turnaround 7–14 days: "
            "bridge with LEV + CLB during the wait. Never skip this step even in urgent scenarios.",
            "PHT/CBZ/OXC/LTG WORSEN EEF1A2 SEIZURES: Na+ channel blockers paradoxically increase myoclonic and "
            "generalised seizure burden in DEE/LGS. Document on emergency care plan: 'AVOID Na+ channel blockers.' "
            "If prescribed in error: urgent EEG to detect worsening before clinical deterioration obvious.",
            "TGB ABSOLUTE CONTRAINDICATION: GABA reuptake inhibition → GABA-A receptor desensitisation → NCSE in "
            "generalised epilepsy. NCSE in EEF1A2 = cognitive/behavioural worsening (not always convulsive) → "
            "continuous EEG mandatory if unexplained cognitive decline occurs on any medication change.",
            "LEV BEHAVIOURAL TOXICITY IS SEVERE IN EEF1A2 DEE: 15–35% of DEE patients develop aggression/irritability "
            "('levetiracetam rage') — far higher than in focal epilepsy. Brivaracetam (BRV) has same SV2A mechanism "
            "with ~5% irritability rate — preferred SV2A agent in EEF1A2. If LEV used: explicit consent + "
            "monthly behavioural checklist for first 3 months.",
            "KD SHOULD BE STARTED EARLY (not as last resort): EEF1A2 data (n=12, Lam 2016) shows 58% ≥50% "
            "reduction at 12 months — higher than most adjunct AEDs. INITIATE DURING ACTH TAPER to reduce "
            "IS relapse rate (~30% in EEF1A2). KD + mTOR suppression may benefit GOF actin/mTOR variant class.",
            "SUDEP RISK IS HIGH: uncontrolled nocturnal GTCS in EEF1A2 DEE = ~1/200/year SUDEP risk. "
            "Seizure alarm systems (NightWatch, SAMi-3, Empatica E4) are STANDARD OF CARE in this phenotype. "
            "Avoid prone sleeping. SUDEP conversation and documented caregiver education is MANDATORY at diagnosis.",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:1000])
    print("\n=== BREAKDOWN keys ===")
    bd = get_breakdown()
    for k, v in bd.items():
        print(f"  {k}: {len(v) if isinstance(v, list) else type(v).__name__}")
    print("\n=== DEFINITIONS keys ===")
    df = get_definitions()
    for k, v in df.items():
        print(f"  {k}: {type(v).__name__}")
