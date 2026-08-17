"""
DNM1 Epilepsy — Developmental and Epileptic Encephalopathy / Infantile Spasms / DEE31
=======================================================================================
40-patient cohort · DNM1 (9q34.11) · Dynamin-1 GTPase · AD de novo · ACTH / KD
OMIM: EPILEPTIC ENCEPHALOPATHY, EARLY INFANTILE, 31 (EIEE31/DEE31) #617107

DNM1 BIOLOGY:
DNM1 (9q34.11) encodes Dynamin-1 — a 870-amino acid, ~96 kDa large GTPase expressed
predominantly in neurons. Dynamin-1 is the principal mechanochemical enzyme driving
clathrin-mediated endocytosis (CME) at the presynaptic terminal, pinching recycled
synaptic vesicle (SV) membrane off the plasma membrane after vesicle fusion (exocytosis).

KEY POINTS:
  1. DNM1 PROTEIN ARCHITECTURE (5 domains):
     GTPase domain (aa 1–330): binds and hydrolyses GTP → ~150 GTP/min basal; >1000 GTP/min
       assembled. GDP release is rate-limiting; GTP hydrolysis provides the power stroke.
     Middle domain (aa 331–490): stalk; self-assembly interface; drives oligomerisation
       into rings (~12–16-mer) and helical collars (~26-mer per turn) around membrane necks.
     Pleckstrin Homology (PH) domain (aa 511–633): targets DNM1 to PI(4,5)P₂-rich
       membrane invaginations (clathrin-coated pits). PIP₂ binding positions DNM1 at
       the neck of the budding vesicle.
     GTPase Effector Domain (GED, aa 694–747): forms intramolecular contacts with GTPase
       domain → stimulates GTPase activity (cis-activation). Also forms trans contacts
       in the assembled collar (further GAP-like stimulation of adjacent protomers).
     Proline-Rich Domain (PRD, aa 751–870): SH3-domain docking; amphiphysin-1/2 and
       endophilin recruit DNM1 to the clathrin-coated pit via PRD; this concentrating
       step is essential for efficient CME at high firing-rate synapses.

  2. SYNAPTIC VESICLE RECYCLING CYCLE:
     (a) SV exocytosis: vesicle fuses with plasma membrane (SNARE complex, Ca²⁺).
     (b) Clathrin coat assembly: clathrin + AP2 + FCHo (bin-amphiphysin-rvs/BAR)
         nucleate a coated pit around the exocytosed membrane patch.
     (c) DNM1 recruitment: amphiphysin-1/2 (SH3 domain) binds DNM1 PRD → concentrates
         DNM1 at the neck of the coated pit. PI(4,5)P₂ ↑ at neck → PH-domain docking.
     (d) DNM1 oligomerisation: DNM1 monomers assemble into ~4 rings (26-mer/turn) around
         the neck (12–16 nm diameter). GTP binding stabilises the collar.
     (e) GTP hydrolysis → power stroke: GED-stimulated GTPase activity → conformational
         change → DNM1 collar CONSTRICTS the neck (from ~14 nm to ~3 nm) → membrane
         FISSION → SV released into cytoplasm.
     (f) SV recycling: uncoating (Hsc70/auxilin), refilling (VAChT/VGLUT/VGAT),
         re-docking at active zone. Full cycle: ~5–20 sec.

  3. WHY DNM1 LOF CAUSES SEIZURES:
     High-frequency neuronal firing demands rapid SV recycling (tens of SV/sec per active
     zone). DNM1 is rate-limiting for recycling at fast synapses. LOF → fission failure
     → SV pool depletion during sustained activity. TWO CONSEQUENCES:
     (a) Reduced INHIBITORY neurotransmission preferentially: GABAergic interneurons fire
         at higher rates (100–600 Hz, parvalbumin/fast-spiking) than excitatory pyramidal
         neurons (10–80 Hz). PV+ interneurons depend on rapid CME via DNM1 for SV
         replenishment — they are preferentially vulnerable to DNM1 LOF. PV+ interneuron
         failure → cortical disinhibition → hyperexcitability → seizures.
     (b) Reduced excitatory vesicle recycling (secondary): glutamatergic terminals also
         fail at high frequency but PV+ failure dominates the acute excitatory/inhibitory
         imbalance, tipping toward excess excitation.
     DOMINANT NEGATIVE MECHANISM: recurrent missense variants (especially at middle-domain
     self-assembly interface, e.g. p.R237W) form non-functional oligomers that TRAP WT
     DNM1 in inactive conformations → haploinsufficiency is amplified beyond simple 50%
     reduction → more severe phenotype than pure truncating variants in some cases.

  4. RECURRENT VARIANTS AND HOTSPOTS (EuroEPINOMICS 2014, Bhatt 2023):
     p.Arg237Trp (R237W) — MOST RECURRENT; middle domain stalk; disrupts self-assembly
       kinetics; dominant negative; severe DEE; infantile spasms; hypsarrhythmia.
     p.Arg237Gln (R237Q) — same hotspot codon; slightly milder than R237W in some series.
     p.Ala395Val (A395V) — middle domain; reduces GTPase-stimulated assembly.
     p.Gly401Glu (G401E) — middle domain; reduces endocytosis rate.
     p.Lys562Met (K562M) — PH domain; impairs PI(4,5)P₂ binding; reduced membrane targeting.
     Truncating variants (frameshift, nonsense) — haploinsufficiency; variable severity;
       NMD-escaping transcripts may produce truncated dominant-negative protein.

  5. GENOTYPE–PHENOTYPE (EuroEPINOMICS 2014, Marsh 2018):
     Middle-domain missense (R237W/Q, A395V): SEVERE — neonatal/early infantile spasms;
       profound ID; non-verbal; spastic quadriplegia in majority. pLI ~0.98.
     PH-domain missense (K562M): MODERATE — infantile spasms; variable ID; some attain
       limited expressive language.
     GTPase domain: variable; GTPase hypomorphs may have less severe DEE.
     Truncating (haploinsufficiency): moderate-severe; some attain limited speech;
       less severe than dominant-negative missense in most cases.
     All: AD de novo >95%; parental gonadal mosaicism ~1-2%.

  6. POPULATION GENETICS:
     DNM1 locus: 9q34.11 (same arm as TSC1 at 9q34.13 — different genes, often confused
     on chromosomal karyotype but distinct on molecular analysis).
     pLI ~0.98 (extreme haploinsufficiency intolerance).
     Estimated ~200–400 confirmed cases worldwide (as of 2025 — underdiagnosis is significant).
     No effective disease-modifying or precision therapy as of 2025 — management is supportive +
     symptomatic seizure control (contrast with mTOR-pathway genes where everolimus is used).

CLINICAL DIAGNOSIS:
  - Onset: neonatal period or first 6 months (earlier than most DEE genes)
  - Seizures at birth or within weeks: tonic, myoclonic, epileptic spasms, focal motor
  - EEG: burst suppression (neonates) → evolves to hypsarrhythmia (infantile spasms phase)
  - MRI: normal or non-specific atrophy; no focal cortical dysplasia (unlike TSC)
  - Hypotonia (marked) + spasticity develops later; feeding difficulties (NG/PEG)
  - Profound to severe intellectual disability; absent speech in >90%
  - Movement disorder: choreoathetosis, dystonia in subset
  - No dysmorphic features (unlike chromosomal DEE syndromes)
  - Key diagnostic clue: severe early-onset DEE + normal MRI + de novo on trio-WES

INHERITANCE AND GENETICS:
  AUTOSOMAL DOMINANT. De novo: >95%. Familial: <5% (rare gonadal mosaicism).
  Locus: 9q34.11. Gene: DNM1 (22 exons; 870 aa protein; large GTPase).
  pLI ~0.98. Recurrent hotspot: p.Arg237 (middle domain stalk).
  Key de novo variants: p.R237W (most common) · p.R237Q · p.A395V · p.G401E · p.K562M.

KEY REFERENCES:
  EuroEPINOMICS-RES Consortium 2014 Am J Hum Genet 95(4):360-370 — DNM1 DEE discovery
  Bhatt et al. 2023 Epilepsia — gene-epilepsy classification reference; DNM1 definitive
  Marsh et al. 2018 Clin Genet 95(1):144-150 — DNM1 cohort expansion, genotype-phenotype
  Bhattacharya & Bhatt 2020 Front Neurol — precision therapy framework for genetic epilepsies
  ILAE Commission on Classification and Terminology 2022
  ACMG-AMP 2015 Genet Med 17(5):405-424 — variant classification standard
"""
import random

random.seed(43)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "DNM1-middle-domain-missense-severe",
        "pct": 40,
        "etiology": "DNM1 middle-domain missense (R237W/Q, A395V) — severe DEE / dominant negative / infantile spasms",
        "mechanism": (
            "Missense variants at the middle-domain stalk interface (hotspot: p.Arg237) disrupt "
            "DNM1 self-assembly into functional ring/helical collars around the vesicle neck. "
            "p.R237W/R237Q disrupt charge-dependent inter-protomer contacts → misassembly of "
            "oligomeric collar → GTP hydrolysis is uncoupled from membrane fission. "
            "DOMINANT NEGATIVE MECHANISM: mutant protomers co-polymerise with WT DNM1 → "
            "mixed oligomers have <20% of normal fission activity → effective >80% LOF despite "
            "only 50% mutant protein. PV+ fast-spiking interneurons preferentially depleted → "
            "cortical disinhibition → early infantile spasms / hypsarrhythmia."
        ),
        "typical_variants": "p.R237W (most recurrent) · p.R237Q · p.A395V · p.G401E",
        "onset_age_months": 2,
        "outcome": "Severe — refractory infantile spasms; profound ID; non-verbal in >90%; spastic quadriplegia; NG/PEG feeding in >60%",
    },
    {
        "category": "DNM1-GTPase-domain-missense-moderate",
        "pct": 28,
        "etiology": "DNM1 GTPase domain missense — moderate DEE / impaired GTP hydrolysis / infantile spasms",
        "mechanism": (
            "Missense variants in the GTPase domain (aa 1–330) reduce intrinsic GTPase activity "
            "or GTP binding affinity → slower power stroke during membrane fission → reduced fission "
            "rate rather than complete abrogation. Moderate haploinsufficiency: ~50% reduction in "
            "fission capacity tolerated at low firing rates but fails under high-frequency activity. "
            "GED intramolecular stimulation is preserved in some variants → residual 30–40% "
            "endocytosis activity. Clinical: infantile spasms onset 3–6 months (slightly later than "
            "middle-domain dominant negatives); partial ACTH response (40–60%)."
        ),
        "typical_variants": "p.K44A-like (GTP-binding); p.R15S; p.A395V-adjacent; catalytic loop variants",
        "onset_age_months": 4,
        "outcome": "Moderate-severe — infantile spasms; ACTH partial response; severe ID; limited speech in 20%; LGS evolution common",
    },
    {
        "category": "DNM1-PH-domain-missense-intermediate",
        "pct": 18,
        "etiology": "DNM1 PH-domain missense — intermediate DEE / impaired membrane targeting / infantile spasms",
        "mechanism": (
            "Missense variants in the PH domain (aa 511–633) reduce PI(4,5)P₂ binding affinity → "
            "impaired targeting of DNM1 to the neck of clathrin-coated pits (PI(4,5)P₂-rich zone). "
            "DNM1 is synthesised and GTPase activity preserved, but recruitment to the correct "
            "membrane site is defective → inefficient vesicle fission. Less severe than middle-domain "
            "dominant negatives because no trapping of WT protein in non-functional assemblies — "
            "pure hypomorphic LOF. p.K562M is the canonical PH domain variant: reduces PI(4,5)P₂ "
            "binding by ~60%. Phenotype: infantile spasms onset 3–8 months; some attain limited "
            "single words (10-15%); moderate-severe ID."
        ),
        "typical_variants": "p.K562M (canonical) · PH loop variants · PH-GED interface variants",
        "onset_age_months": 5,
        "outcome": "Intermediate — some ACTH response; moderate-severe ID; limited language in 15%; focal epilepsy often persists post-IS",
    },
    {
        "category": "DNM1-truncating-haploinsufficiency",
        "pct": 10,
        "etiology": "DNM1 truncating / frameshift — haploinsufficiency DEE (50% protein reduction)",
        "mechanism": (
            "Nonsense or frameshift variants → nonsense-mediated mRNA decay (NMD) → 50% reduction "
            "in DNM1 protein. Pure haploinsufficiency (no dominant negative). 50% DNM1 is insufficient "
            "for full-speed SV recycling at PV+ interneurons under high-frequency demands, but less "
            "catastrophic than dominant-negative missense (no WT protein trapping). Neonatal/infantile "
            "onset but sometimes later than dominant-negative variants. ACTH response rate higher "
            "(~50%) than middle-domain missense group. Some NMD-escape transcripts may produce "
            "C-terminal truncated proteins that act as partial dominant negatives — variable severity."
        ),
        "typical_variants": "p.Q734* · c.1245+2T>G (splice) · c.1048delA · p.R15* · p.W148fs",
        "onset_age_months": 4,
        "outcome": "Moderate-severe — infantile spasms; ACTH response ~50%; severe ID; no speech in majority; some focal epilepsy",
    },
    {
        "category": "DNM1-negative-phenocopy",
        "pct": 4,
        "etiology": "DNM1-negative DEE phenocopy — clinically DNM1-like but no pathogenic variant identified",
        "mechanism": (
            "Severe early-onset DEE with infantile spasms, hypsarrhythmia, normal MRI and "
            "profound ID — clinically mimicking DNM1 — but DNM1 sequencing negative. "
            "Differential: STXBP1 (most common, synaptic), CDKL5 (X-linked, consider females), "
            "ARX (males, X-linked), KCNQ2 (neonatal NaV), DNM1L (mitochondrial fission, distinct), "
            "AP2M1 (adaptor protein 2, mu-1; same endocytic pathway), CLATHRIN (CLTC variants). "
            "Consider SV endocytosis pathway genes: ITSN1, SH3GL2 (endophilin A1) — rare. "
            "Functional endocytosis assay (pHluorin imaging of SV cycling) for VUS classification."
        ),
        "typical_variants": "VUS on DNM1 · panel-negative with similar SV-endocytosis phenotype · AP2M1/CLTC variants",
        "onset_age_months": 5,
        "outcome": "Variable — depends on underlying aetiology; full synaptic gene panel + DNM1L/mitochondrial testing recommended",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES  (5)
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Epileptic Spasms / Infantile Spasms (West Syndrome)",
        "pct": 82,
        "description": (
            "Epileptic spasms — the dominant seizure type in DNM1 encephalopathy, occurring in "
            "80–90% of patients. Brief (0.5–2 sec) symmetric or asymmetric flexion/extension of "
            "neck, trunk, and proximal limbs. Occur in clusters (5–50 per cluster) on waking from "
            "sleep or nap. EEG: hypsarrhythmia (classic or modified). Ictal correlate: electrodecremental "
            "response (abrupt EEG amplitude reduction during each spasm). Onset: typically 1–6 months "
            "(neonatal onset possible — earlier than most genetic IS). "
            "REFRACTORY in >70% — DNM1 IS are among the most pharmacoresistant of all genetic IS."
        ),
        "eeg": (
            "Interictal: hypsarrhythmia — continuous high-amplitude (>300 µV) chaotic background "
            "with multifocal spikes and slow waves; no organised rhythms. Modified hypsarrhythmia "
            "(asymmetric or synchronised bursts) common in older infants. "
            "Ictal: each spasm = electrodecrement (abrupt 1–3 sec amplitude suppression) preceded "
            "by high-amplitude slow wave. Post-cluster: temporary EEG flattening then hypsarrhythmia resumes."
        ),
        "semiology": (
            "Sudden flexion (Salaam attack): head drops, arms flex, knees drawn up. Or extension: "
            "back arches, limbs extend. Or mixed. Lasts 0.5–2 sec. Clusters of 5–50 on waking. "
            "Cry or facial grimacing during or after cluster. No post-ictal confusion."
        ),
        "clinical_tips": (
            "Video-EEG is ESSENTIAL — home video misses 40% of spasms, especially subtle forms. "
            "Start ACTH + Vigabatrin within 48-72 hours of EEG confirmation (delay worsens outcome). "
            "DNM1 IS are particularly refractory: expect only 30–45% sustained spasm cessation with ACTH+VGB "
            "(lower than idiopathic IS). Escalate to KD promptly if ACTH+VGB fails at 4 weeks. "
            "Pyridoxine trial mandatory before ACTH — rule out ALDH7A1/PDE."
        ),
    },
    {
        "type": "Focal Motor Seizures (Neonatal / Infantile Onset)",
        "pct": 72,
        "description": (
            "Focal motor seizures — prominent in DNM1, often preceding infantile spasms or co-existing. "
            "Neonatal onset possible: clonic jerking of one limb, version, eye deviation. "
            "In infants: focal clonic (arm or leg), tonic posturing of one side, autonomic features. "
            "Multifocal in many — focal seizures shift between hemispheres or lobes (multifocal "
            "independent spike discharges on EEG). Reflects diffuse cortical hyperexcitability "
            "from PV+ interneuron failure without anatomical lesion (unlike structural focal epilepsy)."
        ),
        "eeg": (
            "Neonatal: burst-suppression background; focal clonic seizures have rhythmic focal "
            "alpha/beta build-up. Infantile: multifocal sharp waves and spikes (temporal, frontal, "
            "parietal independently). Ictal focal: rhythmic theta/alpha over onset region → spreads. "
            "EEG may show independent bilateral onset — 'pseudobilateral' from multifocal generators."
        ),
        "semiology": (
            "Neonatal: subtle lip-smacking, eye deviation, clonic arm — may be missed without EEG. "
            "Infant: head turn, arm stiffening or clonic jerking (one side). Duration: 1–3 min. "
            "Autonomic: colour change, apnoea, tachycardia. Post-ictal: focal weakness (Todd's paresis)."
        ),
        "clinical_tips": (
            "Neonatal seizures + DNM1 de novo → DO NOT treat as neonatal seizures only; full "
            "genetic work-up essential (trio-WES if possible). "
            "LCM (lacosamide) is preferred for focal DNM1 seizures post-IS phase — slow Na+ "
            "inactivation; IV formulation available for acute management. "
            "Avoid OXC/CBZ if any generalised or myoclonic component."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "pct": 55,
        "description": (
            "Myoclonic seizures — brief bilateral or multifocal myoclonic jerks in DNM1 encephalopathy. "
            "May occur independently or as part of myoclonic-tonic sequence. "
            "Axial myoclonus (head drops) and limb myoclonus. Often worsened by CBZ/OXC, PHT. "
            "Cortical myoclonus: EEG-timed, back-averaged MEG/EEG shows cortical spike preceding jerk. "
            "Subcortical myoclonus can occur in severe cases (brainstem origin). "
            "In older DNM1 patients (>3Y): myoclonus may emerge as a dominant seizure type "
            "as spasms evolve, overlapping with progressive myoclonic epilepsy (PME) features."
        ),
        "eeg": (
            "Generalised or multifocal polyspike burst time-locked to each myoclonic jerk. "
            "Amplitude: 200–500 µV. Duration: 50–200 ms. Background: slow, disorganised. "
            "Interictal: multifocal sharp waves. May show 3-Hz generalised SWD in older children."
        ),
        "semiology": (
            "Sudden bilateral arm/leg jerk. Axial: sudden head nod. Eyelid myoclonia (blinking). "
            "Massive myoclonia (whole-body). Very brief (100–500 ms). No post-ictal state. "
            "May cause falls — high injury risk. Worsened by fatigue and intercurrent illness."
        ),
        "clinical_tips": (
            "AVOID CBZ/OXC/PHT — these Na-channel blockers can precipitate or worsen myoclonus. "
            "VPA is drug of choice for myoclonic seizures (after POLG1 exclusion). "
            "ESM can be added for pure myoclonic or myoclonic-absence component. "
            "KD is effective for myoclonic-atonic overlap. "
            "Clobazam (CLB) as adjunct for myoclonic clusters during illness."
        ),
    },
    {
        "type": "Tonic Seizures (Nocturnal / Lennox-Gastaut Evolution)",
        "pct": 40,
        "description": (
            "Tonic seizures emerge as DNM1 epilepsy evolves from infantile spasms to Lennox-Gastaut "
            "spectrum (LGS) by age 2–5 years (~45% of DNM1 patients). Sudden sustained muscle "
            "stiffening (5–20 sec), often nocturnal. Axial tonic (neck + trunk) or bilateral tonic "
            "(arms + legs). High SUDEP risk when tonic-clonic seizures are frequent. "
            "LGS criteria: tonic + slow SWD (<2.5 Hz) + cognitive impairment. "
            "DNM1-LGS is among the most refractory — KD and rufinamide are key therapies."
        ),
        "eeg": (
            "Paroxysmal fast activity (PFA): 10–20 Hz low-amplitude discharge during tonic phase. "
            "Interictal: diffuse slow spike-wave (1.5–2.5 Hz) on slow, disorganised background. "
            "Sleep: paroxysmal fast bursts in NREM (pathognomonic LGS EEG pattern). "
            "No electrodecrement (tonic → full EEG involvement, not decrement)."
        ),
        "semiology": (
            "Sudden axial or global muscle stiffening. Tonic eye deviation upward. Brief cry at "
            "onset. Duration 5–20 sec. High fall risk. Autonomic features (pallor, apnoea). "
            "No post-ictal confusion (unlike GTCS). Nocturnal predominance."
        ),
        "clinical_tips": (
            "Tonic seizures + slow SWD + ID → LGS criteria met → rufinamide is specifically "
            "indicated. Rufinamide + VPA (reduce rufinamide dose 30% when co-prescribed with VPA "
            "due to pharmacokinetic interaction). KD reduces tonic seizure burden in DNM1-LGS. "
            "Padded bed, floor-level sleeping, door/stair gates mandatory for fall prevention."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "pct": 35,
        "description": (
            "GTCS emerge in older DNM1 patients (>2Y) during LGS evolution or as independent "
            "generalised seizure type. Bilateral symmetric tonic → clonic. Duration 1–3 min. "
            "Post-ictal: prolonged drowsiness (20–60 min). GTCS are a major SUDEP risk factor — "
            "nocturnal GTCS in particular. DNM1 patients often have GTCS during febrile illnesses. "
            "Frequency: often weekly to monthly in refractory cases."
        ),
        "eeg": (
            "Ictal: generalised polyspike run (tonic phase) → polyspike-wave (clonic phase) → "
            "terminal diffuse slowing. Background between GTCS: slow, disorganised, diffuse SWD. "
            "Post-ictal EEG: diffuse delta, burst-like suppression."
        ),
        "semiology": (
            "Tonic phase (10–20 sec): rigid extension of all limbs, tongue bite risk, cyanosis. "
            "Clonic phase (30–60 sec): rhythmic bilateral jerking, decelerating frequency. "
            "Post-ictal: unresponsive, snoring, 20–60 min recovery. Incontinence common."
        ),
        "clinical_tips": (
            "GTCS in DNM1 signals LGS-level severity → ensure SUDEP prevention plan: "
            "bed alarm (e.g. SAMNet), prone sleep avoidance, safe sleeping surface, rescue protocol. "
            "Buccal midazolam 0.3 mg/kg for GTCS >2 min. "
            "VPA ± CLB ± rufinamide for GTCS in LGS context. Annual SUDEP counselling mandatory."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS  (8)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Waking from Sleep / Sleep-Wake Transition",
        "pct": 88,
        "notes": (
            "MOST FREQUENT TRIGGER for infantile spasms in DNM1 DEE — clusters occur predominantly "
            "on waking from sleep (morning, post-nap). Mechanism: NREM→wake transition activates "
            "thalamo-cortical circuits; in DNM1 DEE with hypsarrhythmia, this activation hits a "
            "hypersynchronous background → triggers spasm cluster. "
            "CLINICAL IMPORTANCE: Educate parents to observe clusters 10–30 min after waking; "
            "morning is the highest-yield video-EEG recording window. "
            "In LGS phase: nocturnal tonic seizures have the opposite pattern — occur during NREM."
        ),
    },
    {
        "trigger": "Fever / Febrile Illness",
        "pct": 72,
        "notes": (
            "Fever (≥38°C) triggers acute escalation in spasm frequency and can precipitate "
            "febrile status epilepticus (FSE) in DNM1 DEE. "
            "Mechanism: fever increases metabolic demand → neuronal firing rate ↑ → SV depletion "
            "accelerated in PV+ interneurons already compromised by DNM1 LOF → net disinhibition. "
            "Also: inflammatory cytokines (IL-1β, TNF-α) lower seizure threshold directly. "
            "MANAGEMENT: Acetaminophen/ibuprofen early (do not wait for high fever). "
            "Maintain AEDs during illness; IV formulation if oral route unavailable. "
            "Emergency letter for A&E with AED protocol and rescue medication instructions."
        ),
    },
    {
        "trigger": "Intercurrent Illness / Metabolic Stress",
        "pct": 65,
        "notes": (
            "GI illness (vomiting, diarrhoea) disrupts oral AED absorption → drug level drop → "
            "breakthrough seizures. GI illness is also a direct neuroinflammatory trigger "
            "(gut-brain axis). Dehydration can worsen KD-induced acidosis (dangerous if on KD). "
            "Electrolyte disturbance (Na+, K+ loss with vomiting) further lowers seizure threshold. "
            "During any GI illness: check electrolytes; consider IV AED if vomiting ×2; "
            "KD families must have sick-day protocol (reduce fat ratio, maintain hydration). "
            "Hospital protocol for IV access should be agreed with local paediatrics team in advance."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 58,
        "notes": (
            "Missed VPA, LEV, or CLB dose typically triggers breakthrough seizures within 12–24 hr "
            "(depending on half-life). Missed ACTH dose during active IS course can trigger rebound "
            "spasm clusters within hours. "
            "DNM1 patients have MULTIPLE AEDs (polypharmacy) → risk of complex drug schedules → "
            "dose errors. Recommend: blister packs, electronic dispensers, caregiver training. "
            "If ACTH is missed: contact neurologist immediately; do NOT double-dose. "
            "ACTH taper must be adhered to strictly — sudden ACTH withdrawal = adrenal crisis risk."
        ),
    },
    {
        "trigger": "Sleep Deprivation",
        "pct": 50,
        "notes": (
            "Sleep deprivation triggers focal seizures and GTCS in older DNM1 patients (>2Y) "
            "with LGS-evolution epilepsy. In infants: reduced sleep quality (pain, illness) → "
            "increased spasm frequency on subsequent waking. "
            "MECHANISM: sleep deprivation reduces GABA release (GABA synthesis requires sleep); "
            "in DNM1 with already-impaired PV+ interneuron function → further E/I imbalance. "
            "Melatonin 1–3 mg at bedtime (safe, improves sleep onset latency) is commonly used. "
            "Strict sleep schedule; minimize nocturnal caregiving disruptions."
        ),
    },
    {
        "trigger": "Hyperthermia (Non-Febrile: Hot Bath, Overheating)",
        "pct": 40,
        "notes": (
            "Non-febrile hyperthermia (hot bath, warm car, overheating in pram) can trigger "
            "spasm clusters — similar to SCN1A/Dravet thermolability but via different mechanism "
            "(accelerated SV depletion at higher metabolic rate rather than NaV gain-of-function). "
            "Advise: lukewarm baths (≤37°C); car A/C in summer; avoid prolonged sun exposure; "
            "keep room temperature <22°C. Parents should have rectal thermometer and check "
            "temperature whenever seizures worsen acutely."
        ),
    },
    {
        "trigger": "Sensory Stimulation (Tactile, Auditory, Visual)",
        "pct": 30,
        "notes": (
            "Subset of DNM1 infants show stimulus-sensitive spasms — tactile (touch, bathing), "
            "sudden sound, or bright light can trigger individual spasms or clusters. "
            "Mechanism: sensory activation → thalamocortical arousal → on hypsarrhythmic "
            "background → synchronised discharge → spasm. "
            "CLINICAL: Minimise stimulation during spasm clusters. Dim lighting; reduce noise. "
            "Photosensitivity formal testing (IPS) if suspected — included in EEG protocol."
        ),
    },
    {
        "trigger": "Vaccination / Post-Vaccination Fever",
        "pct": 22,
        "notes": (
            "Post-vaccination fever (routinely given MMR, DTaP, etc.) can trigger acute spasm "
            "exacerbation in DNM1 infants who are in the active IS phase. This is a fever-mediated "
            "trigger, NOT a contraindication to vaccination — all standard vaccines recommended. "
            "Pre-medication with acetaminophen 15 mg/kg 1 hr before vaccination reduces "
            "post-vaccination fever and seizure risk. "
            "Observe for 4 hours post-vaccination if in active IS phase. "
            "Document in vaccination record that DNM1 DEE increases seizure risk with any fever."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS  (8)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "ACTH (Adrenocorticotropic Hormone / Tetracosactide / Synacthen Depot)",
        "level": "Level A",
        "indication": "First-line: infantile spasms / epileptic spasms — per UKISS protocol; lower response rate in DNM1 vs idiopathic IS",
        "dose": (
            "Tetracosactide (Synacthen Depot, UK/EU): 0.5 mg/day IM × 14 days → 8-week taper. "
            "Natural ACTH (Acthar Gel, USA): 150 IU/m²/day IM in 2 divided doses × 14 days → taper. "
            "UKISS protocol: ACTH + VGB simultaneously (not sequential) from day 1."
        ),
        "moa": (
            "ACTH binds melanocortin receptors (MC2R adrenal; MC3R/MC4R/MC5R neuronal) → "
            "cortisol production + direct neuronal anti-epileptic effect. "
            "Reduces CRH-driven limbic hyperexcitability. Neurosteroid induction (allopregnanolone) → "
            "GABA-A potentiation. Anti-inflammatory suppression of neuroinflammatory cytokines. "
            "DNM1-SPECIFIC: No precision rationale (unlike GNB1 where CRH axis is mechanistically "
            "linked). ACTH is empirical IS treatment — works via generic anti-spasm mechanisms, "
            "not by restoring DNM1-mediated endocytosis. Response rate ~35–45% (lower than idiopathic)."
        ),
        "efficacy": "IS cessation: 35–45% in DNM1 at 14 days (lower than TSC/idiopathic ~65%). Hypsarrhythmia resolution: ~40%. Combined with VGB: ~50%.",
        "safety": (
            "Immunosuppression (serious infection risk: CMV, PCP, HSV). Cushingoid features "
            "(moon face — resolves post-taper). Hypertension (BP 3×/week). "
            "Hyperglycaemia (glucose daily). Hypokalaemia (electrolytes weekly). "
            "Irritability, insomnia. Adrenal suppression — MANDATORY taper (never abrupt stop). "
            "No live vaccines during ACTH course."
        ),
        "monitoring": "BP 3×/week · Blood glucose daily · Electrolytes weekly · Infection surveillance · Growth q4W · Early morning cortisol 4W post-stop",
        "dnm1_note": (
            "DNM1 IS are typically MORE REFRACTORY than idiopathic IS — expect ~40% response. "
            "Do NOT wait beyond 4 weeks for ACTH response assessment — escalate to KD promptly. "
            "Pyridoxine (B6) trial MANDATORY before ACTH — rule out ALDH7A1/PDE. "
            "Simultaneous ACTH + VGB (UKISS protocol) is standard — do not use sequential approach. "
            "If rebound spasms at taper → second ACTH course OR switch to KD rather than repeated ACTH."
        ),
    },
    {
        "drug": "Prednisolone (High-Dose Oral)",
        "level": "Level A",
        "indication": "First-line: infantile spasms — equivalent to ACTH (UKISS trial); preferred if ACTH not available or oral route preferred",
        "dose": "4 mg/kg/day (max 40 mg/day) oral × 14 days → 4-week taper. Dexamethasone alternative: 0.3 mg/kg/day × 6 weeks.",
        "moa": (
            "Synthetic glucocorticoid → GR transrepression → reduced neuroinflammatory cytokines "
            "(IL-1β, TNF-α, IL-6) → reduced CRH mRNA transcription → reduced limbic hyperexcitability. "
            "Neurosteroid induction (as with ACTH). Less mineralocorticoid effect than ACTH → "
            "lower electrolyte disturbance. UKISS trial: prednisolone non-inferior to ACTH for IS."
        ),
        "efficacy": "IS cessation: 40–50% in DNM1 (slightly higher or equivalent to ACTH; clinical equipoise). Hypsarrhythmia resolution: ~40%.",
        "safety": (
            "Similar to ACTH: immunosuppression, cushingoid, hypertension, hyperglycaemia, irritability. "
            "Less severe cushingoid features than ACTH. Mandatory taper (adrenal suppression). "
            "Avoid NSAIDs during prednisolone (GI bleed risk). Monitor BP, glucose, growth."
        ),
        "monitoring": "BP twice daily · Blood glucose daily · Growth q4W · Adrenal recovery (early morning cortisol 4W post-stop)",
        "dnm1_note": (
            "Preferred over ACTH when: parent prefers oral route (IM ACTH is burdensome); "
            "ACTH unavailable; second IS episode (prednisolone for second course reduces "
            "cumulative immunosuppression). "
            "For DNM1 IS: simultaneously add VGB per UKISS protocol regardless of corticosteroid choice."
        ),
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "Level A",
        "indication": "First-line combination: infantile spasms (UKISS ACTH+VGB protocol); adjunct only — not monotherapy in DNM1",
        "dose": "50–150 mg/kg/day (divided BD); max 3 g/day. Start 50 mg/kg/day → 100–150 mg/kg/day over 7 days.",
        "moa": (
            "Irreversible GABA transaminase inhibitor → prevents GABA catabolism → elevated synaptic "
            "and extrasynaptic GABA. In DNM1: VGB acts on metabolic GABA homeostasis rather than "
            "presynaptic vesicle release — partially compensates for DNM1-impaired GABAergic vesicle "
            "recycling by increasing the ambient GABA available for tonic (extrasynaptic) inhibition. "
            "Synergy with ACTH: UKISS trial showed ACTH+VGB 73% IS cessation vs ACTH alone 55% "
            "(non-DNM1 genetic IS); synergy less dramatic in DNM1 but still additive."
        ),
        "efficacy": "DNM1 IS with ACTH+VGB: ~50% cessation (vs ~40% ACTH alone). VGB monotherapy: ~25% in DNM1 (low).",
        "safety": (
            "VISUAL FIELD DEFECT (VFD): irreversible bilateral concentric VFD, 30–50% with "
            "cumulative dose >100 g. ERG monitoring at baseline and q3 months. "
            "SHARE REMS (USA): mandatory pre-treatment ERG. Restrict duration to IS course "
            "(typically 4–6 months total). Transient MRI T2 signal in basal ganglia (reverses on stopping)."
        ),
        "monitoring": "ERG baseline · ERG q3M during VGB · VF perimetry when age-appropriate (>5Y) · Limit total VGB duration to 6 months",
        "dnm1_note": (
            "VGB in DNM1 is COMBINATION therapy with ACTH/prednisolone — not monotherapy. "
            "Start VGB simultaneously with ACTH on day 1 (UKISS protocol). "
            "VFD risk: counsel parents at start; document informed consent. "
            "DISCONTINUE VGB after IS course (4–6 months) — do NOT continue as maintenance AED. "
            "Switch to LEV or LCM for ongoing focal seizures after IS phase."
        ),
    },
    {
        "drug": "Valproate (VPA)",
        "level": "Level B",
        "indication": "Post-IS phase: focal epilepsy · GTCS · LGS transition · myoclonic seizures in DNM1",
        "dose": "20–40 mg/kg/day (BD-TDS); TDM target: 50–100 mg/L (trough). IV loading available.",
        "moa": (
            "Broad-spectrum: Na+ channel slow inactivation, T-type Ca²⁺ block, GABA-T inhibition "
            "(↑ GABA synthesis), HDAC inhibition (epigenetic). Covers all post-IS DNM1 seizure types. "
            "In DNM1: VPA increases GABA by the metabolic route (GABA-T inhibition) rather than "
            "vesicular release — mechanistically complementary to DNM1 endocytosis defect. "
            "Good choice for LGS transition: covers tonic, GTCS, and myoclonic with one drug."
        ),
        "efficacy": "50–65% seizure reduction in post-IS DNM1 focal/GTCS. Myoclonic: 55% responder. LGS tonic: 40% reduction.",
        "safety": (
            "POLG1 = ABSOLUTE CONTRAINDICATION (fatal Alpers hepatotoxicity). "
            "VPPP mandatory for all females ≥9Y (MHRA 2021). "
            "Hepatotoxicity (esp. infants <2Y — monitor LFT q3M). "
            "Hyperammonaemia. Teratogenicity (neural tube defects — VPPP). Weight gain. Hair loss."
        ),
        "monitoring": "POLG1 before start · VPA TDM q3M · LFT + FBC + ammonia q3M · VPPP females ≥9Y",
        "dnm1_note": (
            "POLG1 TESTING MANDATORY before VPA in any DNM1 infant. "
            "Do NOT start VPA while on ACTH (additive hepatotoxicity risk) — bridge with LEV "
            "during ACTH course, then transition to VPA as ACTH tapers. "
            "VPA is the backbone AED for post-IS DNM1 epilepsy if POLG1 negative. "
            "Monitor LFT closely in first 6 months (infantile liver is particularly vulnerable)."
        ),
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": "Focal seizures · GTCS adjunct · preferred if POLG1 positive or VPA not tolerated — safe in DNM1 infants",
        "dose": "20–60 mg/kg/day (BD). IV formulation available (equivalent dose). No TDM required.",
        "moa": (
            "SV2A modulator — reduces Ca²⁺-dependent vesicle exocytosis presynaptically. "
            "In DNM1 context: SV2A targets the vesicle fusion step (exocytosis), which is "
            "downstream of the DNM1 endocytosis defect. By reducing excessive excitatory exocytosis, "
            "LEV partially corrects E/I imbalance from a different angle. "
            "Does not worsen myoclonic or absence seizures. Safe for use during ACTH course."
        ),
        "efficacy": "55–60% responder for focal DNM1 seizures. GTCS adjunct: 45% reduction.",
        "safety": "Behavioural: irritability, aggression (10-15%); monitor closely in DNM1 DEE (pre-existing behavioural impairment). No hepatotoxicity. No POLG interaction. Renal dosing.",
        "monitoring": "Behaviour assessment q3M · Renal function baseline · CBCL q6M",
        "dnm1_note": (
            "LEV is the preferred ACTH-compatible AED during the IS course (safe alongside ACTH/VGB). "
            "Also first-choice POLG1+ alternative to VPA. "
            "IV LEV for acute cluster management during illness (when oral route unavailable — "
            "important for DNM1 who frequently have GI illness-triggered clusters). "
            "Behavioural SEs may be more pronounced in DNM1 DEE — try low-dose CLB addition "
            "if LEV-induced aggression is problematic."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B",
        "indication": "Refractory IS (ACTH+VGB failed) · LGS transition · post-IS refractory epilepsy — HIGH PRIORITY in DNM1",
        "dose": "Classical 4:1 fat:carbohydrate. MAD for older children. Dietitian + metabolic team mandatory. Initiate in hospital for infants.",
        "moa": (
            "Ketone bodies (β-hydroxybutyrate, acetoacetate) → multiple anti-epileptic mechanisms: "
            "↑ GABA (via glutamate decarboxylase upregulation), ↓ glutamate vesicular loading, "
            "KATP channel opening (neuronal hyperpolarisation), AMPK activation → mTOR suppression, "
            "reduced neuroinflammation. "
            "DNM1-SPECIFIC RELEVANCE: KATP channel opening and GABA synthesis upregulation "
            "address E/I imbalance by pathways independent of SV endocytosis. "
            "KD does NOT restore DNM1 endocytosis, but compensates downstream. "
            "ADDITIONAL: KD reduces mitochondrial ROS (oxidative stress in high-firing interneurons "
            "with compromised SV recycling is significant) → neuroprotective component."
        ),
        "efficacy": "DNM1 IS refractory to ACTH+VGB: 40–55% ≥50% spasm reduction with KD. Post-IS focal/GTCS: 45–55% ≥50% seizure reduction.",
        "safety": "Metabolic acidosis, hyperlipidaemia, nephrolithiasis, growth impairment. Carnitine supplementation. Monitor lipids, Ca, phosphate, renal US q6M.",
        "monitoring": "Ketones BD (blood BHB target 2–4 mmol/L) · Lipid panel q6M · Renal US q12M · Growth q3M · Carnitine level q6M · HCO3 weekly in first month",
        "dnm1_note": (
            "KD is PRIORITISED in DNM1 — initiate after first ACTH+VGB failure (do not wait for "
            "3 AED failures as per general guidance). DNM1 IS are typically refractory and early "
            "KD initiation (within weeks of ACTH failure) may prevent LGS evolution. "
            "For DNM1 infants on NG/PEG: KD formula (KetoCal 4:1) delivered via tube — "
            "discuss with dietitian before starting if NG/PEG already in place. "
            "Target 2 years continuous KD if effective; re-evaluate every 6 months."
        ),
    },
    {
        "drug": "Rufinamide",
        "level": "Level B",
        "indication": "LGS transition: tonic + atonic seizures in DNM1 evolving to Lennox-Gastaut spectrum",
        "dose": "Child: start 10 mg/kg/day (BD) → 45 mg/kg/day (max 3200 mg/day). Always titrate slowly.",
        "moa": (
            "Sodium channel modulator (prolongs inactive state). Specifically approved for "
            "tonic and atonic seizures in LGS. Reduces paroxysmal fast activity on EEG. "
            "No direct interaction with SV endocytosis pathway. "
            "VPA INTERACTION: VPA inhibits rufinamide metabolism → ↑ rufinamide levels by ~25%; "
            "when co-prescribing with VPA, reduce rufinamide target dose by 30%."
        ),
        "efficacy": "LGS tonic/atonic: 30–40% seizure reduction in phase III trial. DNM1-LGS specific data limited but consistent with general LGS population.",
        "safety": "Nausea, vomiting, somnolence, diplopia. QTc shortening (caution with other QT drugs). VPA interaction (dose reduce rufinamide by 30%).",
        "monitoring": "ECG at baseline · Titration diary · QTc monitoring if co-prescribed QT-modifying drugs",
        "dnm1_note": (
            "Introduce rufinamide when LGS criteria are met: tonic seizures + slow SWD + cognitive impairment. "
            "Check VPA co-prescription — start rufinamide at 5 mg/kg/day if on VPA and titrate slowly. "
            "Rufinamide is most effective for tonic/atonic component specifically — adds to VPA's "
            "myoclonic/GTCS coverage. Add to regimen sequentially, not simultaneously with other new AEDs."
        ),
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B",
        "indication": "Adjunct: focal seizures · myoclonic clusters · acute rescue during illness — versatile adjunct in DNM1",
        "dose": "Child: 0.25–1 mg/kg/day (BD; max 20 mg BD). Acute cluster rescue: 0.1 mg/kg single dose (buccal off-label or oral).",
        "moa": (
            "1,5-benzodiazepine: binds GABA-A receptor at the BZD site (α2/α3 subunit interface) → "
            "positive allosteric modulation → ↑ Cl⁻ conductance → enhanced GABAergic inhibition. "
            "1,5-BZD (vs classical 1,4-BZD): less tolerance development (slower CLB tolerance vs "
            "diazepam/lorazepam). α2/α3-selective (CLB has relatively higher α2/α3 affinity vs α1). "
            "In DNM1: enhances residual GABAergic tone to partially offset PV+ interneuron endocytosis "
            "failure. Useful as 'top-up' inhibition during periods of high seizure risk (illness, taper)."
        ),
        "efficacy": "Focal adjunct: 45–55% responder in 3-month trials. Catamenial cycles: effective adjunct. Rescue: 80-90% cluster termination.",
        "safety": "Sedation (dose-dependent). Tolerance (less than 1,4-BZD). Withdrawal: do not stop abruptly (titrate down over weeks). Drooling, hypotonia (in infants).",
        "monitoring": "Sedation assessment · Behaviour q3M · Tolerance monitoring (re-assess efficacy at 3M)",
        "dnm1_note": (
            "CLB is a useful 'bridging' adjunct during ACTH taper and VGB taper phases in DNM1. "
            "During intercurrent illness: CLB 0.1 mg/kg PRN (rescue) can abort myoclonic clusters. "
            "Tolerance develops in ~25% over 3 months — 'drug holiday' strategy (2-week break) "
            "can restore efficacy. Drooling and hypotonia in DNM1 infants (pre-existing "
            "hypotonia) can be worsened by CLB — use lowest effective dose."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS  (5)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "level": "HIGH CAUTION — avoid if any myoclonic or generalised component",
        "reason": (
            "CBZ/OXC are fast NaV inactivation enhancers — appropriate for pure focal epilepsy but "
            "may PARADOXICALLY WORSEN myoclonic seizures and absence episodes in DNM1 DEE if there is "
            "a generalised epilepsy component (myoclonic, GTCS from LGS-evolution). "
            "OXC causes SIADH (hyponatraemia in 25–30%) — particularly dangerous in DNM1 infants "
            "on ACTH (ACTH-related mineralocorticoid already affects Na+ balance). "
            "HLA-B*15:02 screening mandatory before CBZ in South/Southeast Asian populations. "
            "In PURELY focal DNM1 (rare, blade-6-like variants without myoclonic): CBZ/OXC "
            "acceptable but LCM preferred (no electrolyte risk, no HLA-B required, IV available)."
        ),
        "alternative": "LCM (focal, no SIADH, IV available) · LEV (focal/GTCS, safe) · VPA (full-spectrum, POLG1 mandatory)",
        "icon": "⚠️",
    },
    {
        "drug": "Tiagabine (TGB)",
        "level": "ABSOLUTE CONTRAINDICATION — NCSE risk in generalised epilepsy",
        "reason": (
            "TGB (GABA reuptake inhibitor via GAT-1 blockade) can precipitate non-convulsive "
            "status epilepticus (NCSE) and absence status in ANY patient with a generalised "
            "epilepsy component — including the evolving LGS spectrum of DNM1 DEE. "
            "DNM1 patients are non-verbal in >90% — NCSE may be completely clinically silent "
            "and missed without urgent EEG. NCSE in DNM1 can cause irreversible neuronal injury "
            "in an already-vulnerable brain. ABSOLUTE CONTRAINDICATION in all DNM1 patients."
        ),
        "alternative": "LEV · LCM · VPA · CLB",
        "icon": "🚫",
    },
    {
        "drug": "Valproate — POLG1 Mutation / Mitochondrial Disease",
        "level": "ABSOLUTE CONTRAINDICATION if POLG1 pathogenic variant present",
        "reason": (
            "POLG1 (mitochondrial DNA polymerase gamma) pathogenic variants + VPA → Alpers-Huttenlocher "
            "syndrome: fatal hepatoencephalopathy with liver failure within months of VPA exposure. "
            "DNM1 DEE patients undergo extensive genetic testing; POLG1 co-mutation MUST be excluded "
            "before ANY VPA prescription. Screen POLG1 BEFORE VPA in every DEE infant. "
            "KEY: DNM1 DEE infants are often on VPA long-term from infancy — POLG1 testing at "
            "diagnosis prevents catastrophic Alpers development. "
            "If POLG1 positive → LEV + KD as VPA alternative (same seizure types covered)."
        ),
        "alternative": "LEV · KD · LCM (focal) · CLB (adjunct)",
        "icon": "🚫",
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin — Long-Term Use",
        "level": "HIGH CAUTION — avoid chronic use in DNM1 DEE",
        "reason": (
            "PHT may worsen myoclonic seizures (paradoxical myoclonus from Na-channel block in "
            "GABAergic interneurons at toxic plasma levels). Zero-order kinetics → narrow TI → "
            "toxicity risk in infants. Long-term PHT: gingival hypertrophy, cerebellar atrophy, "
            "cognitive blunting — unacceptable in DNM1 DEE with baseline profound ID. "
            "IV fosphenytoin can be used ACUTELY for status epilepticus (SE) refractory to "
            "first-line benzodiazepine, but is NOT for chronic maintenance. "
            "Preferred IV SE alternatives: IV LEV (20 mg/kg load) · IV VPA (loading dose 20 mg/kg) · "
            "IV LCM (for focal SE)."
        ),
        "alternative": "IV LEV · IV VPA · Buccal midazolam (acute rescue) · IV LCM (focal SE)",
        "icon": "⚠️",
    },
    {
        "drug": "Vigabatrin (VGB) — Long-Term Maintenance (beyond IS course)",
        "level": "HIGH CAUTION — VFD risk with prolonged use; restrict to IS course only",
        "reason": (
            "VGB is appropriate for the infantile spasms treatment phase (4–6 months total). "
            "PROLONGED use beyond the IS phase carries irreversible VFD risk (30–50% with "
            ">100 g cumulative dose; bilateral concentric VFD). "
            "DNM1 DEE patients often develop ongoing focal/LGS epilepsy requiring long-term AEDs — "
            "VGB is NOT the right long-term AED for this; switch to LCM/LEV/VPA after IS course. "
            "SHARE REMS (USA): prescribers must be enrolled and document ERG every 3 months. "
            "If family refuses discontinuation: full REMS documentation and ERG every 3 months mandatory."
        ),
        "alternative": "LEV or LCM for focal seizures post-IS; VPA for GTCS/LGS spectrum",
        "icon": "⚠️",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING  (14 items)
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG1 Screening (before VPA)", "freq": "Once (at diagnosis, before VPA)", "notes": "Full POLG1 sequencing or common variant panel (p.W748S, p.A467T) before starting VPA. Document permanently. ABSOLUTE CONTRAINDICATION if POLG1 pathogenic."},
    {"item": "ACTH — Blood Pressure", "freq": "3×/week during ACTH", "notes": "Hypertension from ACTH mineralocorticoid effect. Target: <95th centile for age. Amlodipine if >99th centile. DNM1 infants may have baseline hypotonia — monitor orthostatic BP."},
    {"item": "ACTH — Blood Glucose", "freq": "Daily during ACTH", "notes": "Hyperglycaemia (ACTH→cortisol→gluconeogenesis). Target fasting: <7 mmol/L. Sliding scale insulin if >10 mmol/L × 2. Reduce ACTH if persistent."},
    {"item": "ACTH — Electrolytes (Na+, K+)", "freq": "Weekly during ACTH; q4W post-stop", "notes": "Hypokalaemia (ACTH mineralocorticoid). Check Na+ simultaneously — OXC/CBZ SIADH risk if added. Early morning cortisol at 4W post-ACTH to exclude adrenal suppression."},
    {"item": "VGB — ERG (Electroretinogram)", "freq": "Baseline · q3M during VGB", "notes": "Visual field defect (VFD) surveillance. b-wave amplitude reduction = early VFD marker. Irreversible — stop VGB if b-wave amplitude falls >10% vs baseline. SHARE REMS (USA): mandatory enrollment."},
    {"item": "VPA — TDM (Therapeutic Drug Monitoring)", "freq": "q3M (trough, pre-dose)", "notes": "Target: 50–100 mg/L. Adjust for illness, growth, drug interactions. CLB and rufinamide interact with VPA levels (VPA ↑ CLB active metabolite; VPA ↑ rufinamide by 25%)."},
    {"item": "VPA — LFT + FBC + Ammonia", "freq": "q3M", "notes": "ALT, AST, GGT, bilirubin, ammonia. Stop VPA if ammonia >2×ULN with encephalopathy. Extra caution in first 6 months of VPA in DNM1 infants (high-risk age for Alpers even with negative POLG1 if VUS present)."},
    {"item": "Developmental Assessment (Bayley-4 / GMDS-ER)", "freq": "q3M (infant) · q6M (child)", "notes": "Bayley Scales of Infant Development (0–42M). Griffiths GMDS-ER for locomotion, language, eye-hand. DNM1: most show profound delay baseline — track for regression (urgent EEG if regression)."},
    {"item": "EEG (hypsarrhythmia + evolution tracking)", "freq": "At diagnosis · 2W post-ACTH · q12M", "notes": "2-week EEG after ACTH start is MANDATORY to assess hypsarrhythmia resolution. Annual EEG tracks evolution: IS phase → LGS transition (look for slow SWD, PFA in sleep)."},
    {"item": "MRI Brain", "freq": "At diagnosis · 12M if normal at baseline", "notes": "Usually normal in DNM1 (no structural lesion, unlike TSC). Repeat at 12–18M: look for myelination delay, non-specific atrophy. If cortical dysplasia found → re-examine DNM1 diagnosis vs TSC1/TSC2."},
    {"item": "Feeding Assessment (NG/PEG)", "freq": "q1M (infancy) · q3M (child)", "notes": "DNM1 DEE: feeding difficulties in >60% (hypotonia + oromotor dyspraxia). NG tube in infancy → consider PEG placement if persistent feeding failure. Dietitian: ensure caloric adequacy especially if on KD."},
    {"item": "VPPP (females on VPA)", "freq": "Annual (from age 9Y)", "notes": "Valproate Pregnancy Prevention Programme — mandatory for all females ≥9Y on VPA (MHRA 2021). Annual form signed. Not immediately relevant in infancy but document policy for future clinicians."},
    {"item": "SUDEP Risk Counselling", "freq": "Annual (once GTCS present)", "notes": "SUDEP risk increases with GTCS frequency. Bed alarm (SAMNet/Emfit), floor-level sleeping, prone avoidance, buccal midazolam for GTCS >2 min. Annual SUDEP discussion from age 2Y if GTCS present."},
    {"item": "Genetic Counselling", "freq": "At diagnosis · q2Y", "notes": "De novo >95%: very low parental recurrence risk (~1-2% gonadal mosaicism). Parental testing recommended. Prenatal testing options (PGT-M, CVS/amnio) discussed for subsequent pregnancies."},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE  (6 windows)
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Neonatal / Early Infancy (0–3M)",
        "key_events": "Neonatal seizures (subset) · Hypotonia · Genetic diagnosis · Feeding assessment",
        "priorities": (
            "DNM1 can present with neonatal seizures (focal motor, clonic) before infantile spasms. "
            "If DNM1 known from prenatal or neonatal trio-WES: alert neonatology/neurology team. "
            "Hypotonia: physiotherapy referral immediately. Feeding: swallowing assessment (SLT); "
            "NG tube if poor oral feeding. POLG1 test sent at diagnosis (before future VPA need). "
            "Counsel parents on spasm recognition (clusters on waking = red flag). "
            "Video-EEG window: if neonatal seizures present, confirm with EEG and initiate treatment."
        ),
    },
    {
        "window": "Infancy — Spasm Phase (1–8M)",
        "key_events": "Infantile spasms onset · ACTH + VGB · Hypsarrhythmia · Developmental impact",
        "priorities": (
            "EMERGENCY RESPONSE at spasm onset: Video-EEG confirm → B6 trial → ACTH + VGB within "
            "48-72 hours. DNM1: expect lower response rate (~40%) — plan KD escalation early. "
            "Monitor ACTH side effects intensively (BP, glucose, electrolytes). ERG at VGB start. "
            "Developmental team: physiotherapy + occupational therapy + speech-language therapy (SLT). "
            "Target: hypsarrhythmia resolution at 2-week EEG. If not resolved → escalate urgently."
        ),
    },
    {
        "window": "Late Infancy (8–24M)",
        "key_events": "ACTH taper · VGB taper · KD initiation (if refractory) · Transition to VPA/LEV",
        "priorities": (
            "ACTH taper: complete by 12 weeks from start; bridge with LEV during taper. "
            "VGB: discontinue at 6 months total. If IS refractory to ACTH+VGB → INITIATE KD. "
            "KD in NG/PEG-fed infants: KetoCal 4:1 via tube — ensure dietitian involvement. "
            "Post-IS epilepsy monitoring: EEG at 12M and 18M to track evolution. "
            "Neurodevelopmental: Bayley-4 at 12M and 18M. "
            "PEG assessment if NG dependency continues beyond 12 months."
        ),
    },
    {
        "window": "Early Childhood (2–5Y)",
        "key_events": "LGS evolution · KD continuation/escalation · Focal epilepsy · AED optimisation",
        "priorities": (
            "45% of DNM1 patients evolve to LGS spectrum by age 3Y. "
            "EEG at age 2Y and 3Y: look for slow SWD (<2.5 Hz) and paroxysmal fast activity (LGS criteria). "
            "If LGS evolves: add rufinamide to VPA backbone. Continue KD if effective. "
            "Neurodevelopmental review: GMDS-ER at 3Y and 5Y. SEN assessment — special school placement. "
            "Adaptive equipment: wheelchair, orthotics if spasticity develops. Gastrostomy review. "
            "SUDEP plan initiated: bed alarm, rescue protocol established."
        ),
    },
    {
        "window": "School Age (5–12Y)",
        "key_events": "Special education plan · Epilepsy review · ASD assessment · AED rationalisation",
        "priorities": (
            "Special education: SEN statement with 1:1 support. "
            "Formal ASD assessment (ASD features in ~40% DNM1). "
            "Review AED polypharmacy — target 2–3 AEDs max; re-assess KD continuation. "
            "Annual SUDEP discussion (GTCS likely present by school age). "
            "Rescue medication plan at school: buccal midazolam with staff training. "
            "Annual EEG; MRI if new seizure type or regression. "
            "Orthopaedic assessment if spasticity (hip surveillance, scoliosis screening from age 7Y)."
        ),
    },
    {
        "window": "Adolescence / Adult (12Y+)",
        "key_events": "VPPP (females) · Adult transition · Social care · Long-term AED management",
        "priorities": (
            "VPPP from age 9Y if female on VPA: mandatory annual form + contraception confirmation. "
            "Transition planning: begin at 14Y with joint paediatric-adult clinic; handover by 18Y. "
            "Most DNM1 patients with severe DEE: unable to drive; supported living assessment. "
            "Bone density (DEXA): enzyme-inducing AED effect on bone (PHB, PHT if used historically). "
            "SUDEP annual counselling: bed alarm, safe sleeping, carer awareness of nocturnal GTCS. "
            "Carer/parent mental health: burnout significant by adolescence — social work referral. "
            "KD: reassess need at 10Y; some discontinue after age 7–8Y (less effective in older patients)."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS  (15 concepts)
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "DNM1 (9q34.11)",
        "definition": (
            "Dynamin-1 gene; 9q34.11; 22 exons; 870 aa; ~96 kDa large GTPase. "
            "Essential for clathrin-mediated synaptic vesicle endocytosis (membrane fission). "
            "pLI ~0.98 (extreme intolerance); AD de novo >95%. "
            "OMIM EIEE31/DEE31 #617107. Discovery: EuroEPINOMICS-RES Consortium 2014."
        ),
    },
    {
        "term": "Synaptic Vesicle Endocytosis (Clathrin-Mediated)",
        "definition": (
            "Post-exocytosis membrane retrieval pathway: clathrin + AP2 coat forms pit → "
            "BAR-domain proteins (amphiphysin, endophilin) form tubule → DNM1 oligomerises as ring "
            "around neck → GTP hydrolysis → constriction → membrane fission → SV released → "
            "uncoating → refilling → re-docking. DNM1 is rate-limiting — ~5-20 sec per cycle. "
            "High-frequency PV+ interneurons (100–600 Hz) most vulnerable to DNM1 LOF."
        ),
    },
    {
        "term": "DNM1 Middle Domain / Stalk (Self-Assembly Interface)",
        "definition": (
            "DNM1 amino acids 331–490; drives oligomerisation into rings (12–16 mer) and helical "
            "collars (26 mer/turn) around vesicle necks. Hotspot for pathogenic variants: p.R237W "
            "(most recurrent) disrupts inter-stalk electrostatic contacts → misassembly → dominant "
            "negative trapping of WT DNM1 → >80% functional LOF despite 50% mutant protein."
        ),
    },
    {
        "term": "Dominant Negative Mechanism (DNM1)",
        "definition": (
            "Recurrent missense DNM1 variants (especially p.R237W/Q at stalk interface) form "
            "non-functional oligomers that co-polymerise with WT DNM1 → mixed rings with <20% "
            "fission activity → amplified LOF beyond simple haploinsufficiency. "
            "Contrast: truncating variants → pure haploinsufficiency (50% loss) → often milder phenotype."
        ),
    },
    {
        "term": "PV+ Interneuron Vulnerability (DNM1 Mechanism)",
        "definition": (
            "Parvalbumin-positive (PV+) fast-spiking interneurons fire at 100–600 Hz — the highest "
            "firing rates in cortex. Their synaptic vesicle pools require rapid DNM1-mediated "
            "recycling. DNM1 LOF preferentially depletes PV+ inhibitory terminals → cortical "
            "disinhibition → E/I imbalance → seizures. Pyramidal excitatory neurons (10–80 Hz) "
            "are relatively less affected, amplifying the net inhibitory deficit."
        ),
    },
    {
        "term": "West Syndrome (Infantile Spasms)",
        "definition": (
            "DEE syndrome triad (Gibbs & Gibbs 1950): epileptic spasms + hypsarrhythmia (EEG) "
            "+ developmental regression/arrest. Onset 3–12 months. DNM1 is one of the most "
            "refractory genetic causes (IS cessation ~40–50% vs ~65% for idiopathic). "
            "Differential includes STXBP1, ARX, CDKL5, KCNQ2, TSC1/TSC2, structural (cortical dysplasia)."
        ),
    },
    {
        "term": "Hypsarrhythmia",
        "definition": (
            "Interictal EEG hallmark of West syndrome: high-amplitude (>300 µV), chaotic, "
            "disorganised background with multifocal asynchronous spikes and slow waves. "
            "Classic (Gibbs) vs Modified (synchronised bursts, hemispheric asymmetry, suppression periods). "
            "Ictal correlate: electrodecrement (sudden amplitude suppression during each spasm). "
            "Resolution of hypsarrhythmia at 2-week EEG after ACTH = primary treatment response endpoint."
        ),
    },
    {
        "term": "LGS (Lennox-Gastaut Syndrome) Evolution",
        "definition": (
            "DNM1 DEE evolves to LGS spectrum in ~45% by age 3–5Y. "
            "LGS triad: multiple seizure types (tonic + atonic + GTCS + absence) + slow SWD (<2.5 Hz) "
            "on EEG + cognitive/behavioural impairment. DNM1-LGS is particularly refractory. "
            "Treatment: VPA backbone + rufinamide (tonic/atonic) + CLB + KD."
        ),
    },
    {
        "term": "ACTH (Adrenocorticotropic Hormone)",
        "definition": (
            "39-aa pituitary peptide; binds MC2R (adrenal, cortisol) and MC3/4/5R (brain, direct neuronal). "
            "First-line for infantile spasms per UKISS/NICE guidelines. "
            "Tetracosactide (Synacthen Depot, synthetic ACTH 1-24, UK/EU) or Acthar Gel (natural porcine "
            "ACTH, USA). In DNM1: empirical IS treatment, no precision rationale (unlike GNB1/ALDH7A1). "
            "Response rate ~40–45% (lower than idiopathic IS ~65%)."
        ),
    },
    {
        "term": "VFD (Visual Field Defect) / VGB Retinopathy",
        "definition": (
            "Irreversible bilateral concentric VFD from vigabatrin toxicity to GABA-A receptors "
            "in retinal cone photoreceptors (peripheral > central). Risk: cumulative dose >100 g, "
            "prolonged duration, young age. ERG (b-wave amplitude reduction): earliest detectable "
            "marker. SHARE REMS (USA): mandatory ERG q3M. Restrict VGB to IS course (≤6 months)."
        ),
    },
    {
        "term": "POLG1 (DNA Polymerase Gamma)",
        "definition": (
            "Mitochondrial DNA polymerase gamma; pathogenic variants (p.W748S, p.A467T most common) "
            "+ VPA → Alpers-Huttenlocher syndrome: fatal hepatoencephalopathy. "
            "Mandatory POLG1 screening before VPA in ANY DEE infant. Full gene sequencing preferred. "
            "If POLG1 positive: LEV + KD as VPA alternative."
        ),
    },
    {
        "term": "VPPP (Valproate Pregnancy Prevention Programme)",
        "definition": (
            "MHRA 2021 UK mandatory programme for females ≥9Y on VPA: annual form signed, "
            "effective contraception confirmed, patient information card issued. Cannot prescribe "
            "VPA without VPPP documentation from age 9Y. Relevant for DNM1 patients on long-term "
            "VPA through childhood and adolescence."
        ),
    },
    {
        "term": "SUDEP (Sudden Unexpected Death in Epilepsy)",
        "definition": (
            "Death in epilepsy patient without anatomical or toxicological cause; most commonly "
            "nocturnal post-GTCS (SUDEP in Childhood Surveillance Study). Risk in DNM1: frequent "
            "GTCS, LGS evolution, nocturnal seizures, polypharmacy. Prevention: bed alarm, "
            "safe sleeping position (supine), rescue protocol, GTCS frequency minimisation. "
            "Annual SUDEP counselling mandatory from first GTCS."
        ),
    },
    {
        "term": "ACMG-AMP 2015 Variant Classification",
        "definition": (
            "Richards et al. 2015 Genet Med — framework for sequence variant pathogenicity: "
            "Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign. "
            "DNM1 recurrent de novo missense (p.R237W/Q, p.A395V) = Pathogenic (strong criteria: "
            "PM2 population rarity, PS2 de novo confirmed, PS4 prevalence, PM1 critical domain). "
            "Novel missense at non-hotspot: functional endocytosis assay (pHluorin SV cycling "
            "assay, dynamin GTPase activity) required for LP/P classification."
        ),
    },
    {
        "term": "Gonadal Mosaicism",
        "definition": (
            "DNM1 is de novo in >95% of cases, BUT parental gonadal mosaicism (GNM) occurs in "
            "~1–2% — parent unaffected but carries DNM1 variant in some germ cells → recurrence "
            "risk for siblings is ~1-2% (not zero). Parental sequencing of blood DNA usually "
            "negative in gonadal mosaicism. Prenatal testing (CVS/amniocentesis) or PGT-M "
            "discussed for future pregnancies regardless of negative parental blood testing."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "ACTH response window", "value": "Assess at 14 days; escalate plan at 4 weeks if no response", "action": "If spasms persist at 4 weeks → initiate KD or second hormonal course; do not add third AED without KD trial"},
    {"name": "BP during ACTH", "value": ">99th centile for age (×2 readings)", "action": "Start amlodipine; escalate antihypertensive if persistent; contact paediatric cardiology if severe"},
    {"name": "Blood glucose (ACTH)", "value": ">10 mmol/L × 2 readings", "action": "Insulin sliding scale; reduce ACTH dose if persistent hyperglycaemia (not controlled by insulin)"},
    {"name": "VPA TDM target", "value": "50–100 mg/L (trough, pre-dose)", "action": "Increase dose if <50 (check compliance first); reduce/hold if >100 mg/L or clinical toxicity (tremor, ataxia, drowsiness)"},
    {"name": "VGB total duration", "value": "≤6 months for IS course", "action": "Taper and discontinue after IS control achieved; switch to LEV/LCM for ongoing focal seizures"},
    {"name": "VFD ERG signal", "value": "b-wave amplitude reduction >10% vs baseline", "action": "Stop VGB immediately; urgent ophthalmology review; document VFD in records; do not restart VGB"},
    {"name": "Ketosis target (KD)", "value": "Blood BHB 2–4 mmol/L; urine ketones 3+", "action": "Adjust fat ratio (titrate up to 4:1 if below); ensure adequate fat intake; hospitalise for KD initiation if <12M"},
    {"name": "LFT trigger (VPA)", "value": "ALT >3×ULN or ammonia >2×ULN with clinical symptoms", "action": "Hold VPA; check POLG1 urgently if not previously done; hepatology review; do not restart VPA without specialist input"},
    {"name": "Developmental red flag", "value": "Loss of previously acquired skills at ANY age", "action": "Urgent EEG (sub-clinical SE); AED review (toxicity?); metabolic screen; MRI if new neurological sign"},
    {"name": "GTCS rescue threshold", "value": "GTCS duration >2 min", "action": "Administer buccal midazolam 0.3 mg/kg; if no response at 5 min → call 999/112; do not give second midazolam dose unless instructed"},
    {"name": "KD side effect thresholds", "value": "Serum HCO3 <15 mmol/L; Ca <2.1 mmol/L; LDL >3.5 mmol/L", "action": "HCO3 low → reduce KD ratio; Ca low → supplement; LDL high → dietary fat composition review (less saturated fat)"},
    {"name": "SUDEP review trigger", "value": "≥1 nocturnal GTCS in past 12 months", "action": "Introduce bed alarm; prone sleep avoidance; GTCS rescue buccal midazolam; SUDEP annual counselling; consider GTCS-reducing AED escalation"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE 2022 Classification", "relevance": "International League Against Epilepsy: DEE and self-limited epilepsy classification; DNM1 encephalopathy = DEE31 (early infantile); infantile spasms = epileptic spasms (preferred ILAE 2022 term)."},
    {"name": "NICE NG217 (Epilepsy, 2022)", "relevance": "UK NICE: infantile spasms — ACTH (tetracosactide) or high-dose prednisolone first-line; VGB adjunct (UKISS protocol); KD for refractory IS after 2 hormonal failures."},
    {"name": "UKISS Trial (2004)", "relevance": "UK Infantile Spasms Study: ACTH vs prednisolone vs VGB. ACTH+prednisolone superior to VGB alone. Combined ACTH+VGB is UK standard of care (73% cessation vs 55% ACTH alone in non-DNM1 IS)."},
    {"name": "EuroEPINOMICS-RES Consortium 2014 Am J Hum Genet", "relevance": "DNM1 discovery paper: identified recurrent de novo DNM1 variants in infantile spasms/DEE cohort. Established R237W/Q as hotspot missense; functional endocytosis assay showed impaired SV recycling."},
    {"name": "Marsh et al. 2018 Clin Genet", "relevance": "DNM1 cohort expansion (n>30): genotype-phenotype correlations; middle-domain variants = severe DEE; PH-domain variants = intermediate phenotype; first formal DNM1 gene-epilepsy classification attempt."},
    {"name": "Bhatt et al. 2023 Epilepsia", "relevance": "Gene-epilepsy classification reference: DNM1 classified as definitive epilepsy gene (strong clinical and functional evidence). Treatment framework for genetic DEE including DNM1."},
    {"name": "CPIC POLG-VPA 2023", "relevance": "Clinical Pharmacogenomics Implementation Consortium: POLG pathogenic variants = absolute contraindication to VPA. Mandatory testing before VPA in any DEE patient with unexplained encephalopathy."},
    {"name": "SHARE REMS (Vigabatrin / Sabril)", "relevance": "US FDA Risk Evaluation and Mitigation Strategy for vigabatrin: mandatory ERG q3M; prescriber + pharmacy enrolment required. Applies to all DNM1 patients receiving VGB in the USA."},
    {"name": "MHRA VPPP 2021", "relevance": "UK MHRA Valproate Pregnancy Prevention Programme: mandatory for females ≥9Y on VPA. Annual signed consent form, effective contraception confirmed, patient information card issued."},
    {"name": "ACMG-AMP 2015 Variant Classification", "relevance": "Standards for interpreting sequence variants: DNM1 recurrent de novo missense at hotspot codons = Pathogenic. Novel missense: functional endocytosis assay (pHluorin SV cycling) required for LP classification."},
    {"name": "ILAE Diet Therapies 2018", "relevance": "International consensus for dietary treatments: KD is indicated for refractory IS after 2 failed IS therapies. DNM1: KD high-priority (initiate early, not as last resort). Classical 4:1 and MAD protocols."},
    {"name": "WHO ICF 2019", "relevance": "International Classification of Functioning, Disability and Health: frames DNM1 DEE impact across body functions (seizure/cognition), activities (communication/self-care), participation (education/social). Guides rehabilitation goals."},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES  (6)
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"pmid": "PMID 25363760", "citation": "EuroEPINOMICS-RES Consortium 2014. De novo mutations in synaptic transmission genes including DNM1 cause epileptic encephalopathies. Am J Hum Genet 95(4):360-370."},
    {"pmid": "PMID 29236316", "citation": "Marsh et al. 2018. Expanding the phenotype associated with DNM1 variants: Additional cases of DNM1-related epileptic encephalopathy. Clin Genet 95(1):144-150."},
    {"pmid": "PMID 35524059", "citation": "Bhatt et al. 2023. Epilepsy gene panels: current state and future directions. Epilepsia 64(3):543-562."},
    {"pmid": "PMID 32966943", "citation": "Bhattacharya & Bhatt 2020. Precision medicine approaches for the treatment of epilepsy. Front Neurol 11:563026."},
    {"pmid": "PMID 15151455", "citation": "Lux et al. 2004 (UKISS). Hormonal treatment versus vigabatrin for infantile spasms: a multicentre randomised controlled trial. Lancet Neurol 3(5):289-295."},
    {"pmid": "PMID 25356970", "citation": "Ferguson et al. 2017. Dynamin, a membrane-remodelling GTPase. Nat Rev Mol Cell Biol 13(2):75-88."},
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT  (40 patients, 15 shown)
# ─────────────────────────────────────────────────────────────────────────────
NAMES_FEMALE = ["Aisha", "Priya", "Sofia", "Mei", "Amara", "Layla", "Nina", "Chloe", "Ananya", "Rania", "Sana", "Leila", "Nadia", "Zoe", "Aria"]
NAMES_MALE   = ["Arjun", "Leo", "Omar", "Ethan", "Noah", "Dev", "Kai", "Rayan", "Luca", "Sam", "Amir", "Jay", "Felix", "Rohan", "Max"]
ETIOL_DIST   = {
    "middle-domain-missense-severe": 16,
    "GTPase-domain-moderate": 11,
    "PH-domain-intermediate": 7,
    "truncating-haploinsufficiency": 4,
    "DNM1-negative-phenocopy": 2,
}
ACTH_STATUS = [
    "ACTH + VGB — partial spasm response (ongoing)",
    "ACTH + VGB + KD — spasm controlled",
    "Prednisolone + VGB — spasms ceased",
    "KD initiated (ACTH+VGB failed)",
    "VPA + LEV (post-IS phase)",
]
VARIANT_BY_ETIOL = {
    "middle-domain-missense-severe": ["p.R237W", "p.R237Q", "p.A395V", "p.G401E", "p.L239P"],
    "GTPase-domain-moderate":        ["p.R15S", "p.G60D", "p.K44E", "p.D128N", "p.T141R"],
    "PH-domain-intermediate":        ["p.K562M", "p.I533T", "p.G587R", "p.N618D", "p.L562P"],
    "truncating-haploinsufficiency":  ["p.Q734*", "p.R15*", "c.1048delA", "c.1245+2T>G"],
    "DNM1-negative-phenocopy":        ["VUS p.V200I (DNM1)", "Panel-negative / AP2M1"],
}


def _gen_patients():
    patients = []
    pid = 1
    etiol_pool = []
    for et, count in ETIOL_DIST.items():
        etiol_pool.extend([et] * count)
    random.shuffle(etiol_pool)
    for i, et in enumerate(etiol_pool[:15]):
        sex = "F" if i % 2 == 0 else "M"
        name = (NAMES_FEMALE if sex == "F" else NAMES_MALE)[i % 15]
        vlist = VARIANT_BY_ETIOL[et]
        variant = vlist[i % len(vlist)]
        onset = {
            "middle-domain-missense-severe": 2,
            "GTPase-domain-moderate": 4,
            "PH-domain-intermediate": 5,
            "truncating-haploinsufficiency": 4,
            "DNM1-negative-phenocopy": 5,
        }[et]
        tx = random.choice(ACTH_STATUS)
        seizure_free = et not in ("middle-domain-missense-severe", "DNM1-negative-phenocopy") or random.random() > 0.7
        polg_tested = True
        vppp = sex == "F" and random.random() > 0.2
        patients.append({
            "id": f"DNM{pid:02d}",
            "name": name,
            "sex": sex,
            "age_at_diagnosis_months": round(onset + random.uniform(-0.5, 2.0), 1),
            "etiology_class": et,
            "variant": variant,
            "seizure_free": seizure_free,
            "current_treatment": tx,
            "polg1_tested": polg_tested,
            "vppp_enrolled": vppp,
        })
        pid += 1
    return patients


PATIENTS = _gen_patients()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    seizure_free_count = sum(1 for p in PATIENTS if p["seizure_free"])
    return {
        "gene": "DNM1",
        "locus": "9q34.11",
        "omim": "DEE31 / EIEE31 (OMIM #617107)",
        "protein": "Dynamin-1 — Large GTPase / Synaptic Vesicle Endocytosis (CME fission)",
        "inheritance": "AD de novo (>95%)",
        "pli": "~0.98",
        "cohort_size": 40,
        "seizure_free_pct": round(seizure_free_count / 15 * 100),
        "top_trigger": "Waking from sleep / sleep-wake transition (88%)",
        "top_treatment": "ACTH + Vigabatrin (Level A — UKISS protocol) + Ketogenic Diet (early escalation)",
        "etiology_distribution": {ec["category"]: int(40 * ec["pct"] / 100) for ec in ETIOLOGY_CATALOG},
        "mechanism": (
            "DNM1 (9q34.11) encodes Dynamin-1, the principal GTPase driving clathrin-mediated "
            "synaptic vesicle endocytosis. After SV fusion: DNM1 oligomerises as helical collar "
            "around the vesicle neck → GTP hydrolysis → constriction → membrane fission → SV "
            "recycled. LOF → SV pool depletion during high-frequency firing. PV+ fast-spiking "
            "interneurons (100–600 Hz) are most vulnerable — inhibitory circuit failure → cortical "
            "disinhibition → early infantile spasms / DEE31. Dominant negative mechanism (p.R237W): "
            "mutant protomers trap WT dynamin in non-functional collars → >80% functional LOF "
            "despite 50% mutant protein. No precision therapy available (2025) — ACTH + KD."
        ),
        "key_clinical_pearl": (
            "DNM1 IS are among the most refractory genetic infantile spasms (~40% ACTH response). "
            "ESCALATE TO KD EARLY — do not wait for 3 AED failures. "
            "Pyridoxine (B6) trial MANDATORY before ACTH. POLG1 screening before VPA. "
            "Simultaneous ACTH + VGB (UKISS protocol) from day 1 — not sequential. "
            "VGB: restrict to 6 months (VFD risk). Monitor ERG q3M while on VGB. "
            "LGS evolution in ~45% by age 3Y — add rufinamide when LGS criteria met."
        ),
        "key_contraindication": (
            "ABSOLUTE CI: Tiagabine (NCSE in LGS/generalised component) · VPA + POLG1 (Alpers fatal). "
            "HIGH CAUTION: CBZ/OXC if myoclonic component (aggravates) · VGB long-term (VFD). "
            "PHT: avoid chronic use (myoclonic aggravation, zero-order kinetics toxicity risk). "
            "POLG1 MANDATORY before VPA. Do NOT continue VGB as maintenance AED."
        ),
        "key_references": [
            "EuroEPINOMICS-RES Consortium 2014 Am J Hum Genet 95(4):360 — DNM1 discovery",
            "Marsh et al. 2018 Clin Genet 95(1):144 — DNM1 genotype-phenotype expansion",
            "Bhatt et al. 2023 Epilepsia — gene-epilepsy reference; DNM1 definitive",
            "UKISS 2004 Lancet Neurol 3(5):289 — ACTH+VGB standard for IS",
            "NICE NG217 2022 — UK epilepsy guideline; IS algorithm",
            "ACMG-AMP 2015 Genet Med 17(5):405 — variant classification standard",
        ],
    }


def get_breakdown():
    seizure_free_count = sum(1 for p in PATIENTS if p["seizure_free"])
    return {
        "etiology_catalog": ETIOLOGY_CATALOG,
        "patients_sample": PATIENTS,
        "summary_stats": {
            "seizure_free": seizure_free_count,
            "etiology_classes": len(ETIOLOGY_CATALOG),
            "seizure_types_count": len(SEIZURE_TYPES),
            "treatments_count": len(TREATMENTS),
            "contraindications_count": len(CONTRAINDICATIONS),
            "monitoring_items": len(MONITORING),
            "lifecycle_windows": len(LIFECYCLE),
        },
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    return {
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
