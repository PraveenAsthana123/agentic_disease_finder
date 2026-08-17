"""
GRIN1 Epilepsy — Neurodevelopmental Disorder / DEE / Hyperkinetic Movement Disorder
=====================================================================================
40-patient cohort · GRIN1 (9q34.3) · GluN1 / NR1 — Obligatory NMDA Receptor Subunit
Precision treatment: D-serine (LOF) · Memantine (GOF) · NMDA-receptor pharmacology

GRIN1 BIOLOGY:
GRIN1 (9q34.3) encodes GluN1 (also known as NR1) — the OBLIGATORY subunit of all
functional NMDA (N-methyl-D-aspartate) receptors in the mammalian nervous system.
Unlike GluN2A/B/C/D or GluN3A/B subunits (which are modulatory), GluN1 is essential:
no GluN1 → no functional NMDA receptor. Every NMDA-R heterotetrameric complex
contains exactly two GluN1 subunits plus two GluN2 or GluN3 subunits.

KEY POINTS:
  1. NMDA RECEPTOR STRUCTURE AND GluN1 ROLE:
     The NMDA receptor is a heterotetrameric ligand-gated cation channel:
       • 2x GluN1 (obligatory) + 2x GluN2A or GluN2B (most common in cortex/hippocampus)
       • Or: 2x GluN1 + GluN2A + GluN2B (triheteromeric composition)
       • Or: 2x GluN1 + GluN3A/B (excitatory glycine-activated channels — rare)
     GluN1 domain architecture:
       N-terminal domain (NTD): allosteric modulatory site; Zn²⁺ and proton modulation.
       Agonist-binding domain (ABD / S1-S2): binds GLYCINE or D-SERINE as obligatory
         co-agonist. Without glycine/D-serine occupancy, NMDA-R cannot open even with
         glutamate + depolarisation. Critical therapeutic target.
       Transmembrane domain (TMD): M1–M4 helices; M2 pore loop (asparagine 615 = key
         Mg²⁺ block site; Q/R/N site); M3 = gate helix (controls channel opening).
       C-terminal domain (CTD): intracellular; multiple splice variants (8 isoforms in humans);
         interacts with CaM, PSD-95, nNOS; regulates NMDA-R trafficking and synaptic
         anchoring at the postsynaptic density.

  2. NMDA RECEPTOR — COINCIDENCE DETECTOR:
     Dual requirement for opening:
       (i)  Presynaptic: glutamate released → binds GluN2 ABD.
       (ii) GluN1 glycine site: must be occupied by glycine or D-serine (co-agonist).
       (iii) Postsynaptic depolarisation: relieves Mg²⁺ block from pore (Mg²⁺ enters
              M2 pore at resting Vm −70 mV → blocks ion flow; depolarisation displaces Mg²⁺).
     When all three conditions met: NMDA-R opens → Ca²⁺, Na⁺ (influx), K⁺ (efflux).
     Ca²⁺ influx: triggers CaMKII activation → AMPA receptor phosphorylation/insertion → LTP.
     NMDA-R is essential for Hebbian synaptic plasticity, learning, memory, and cortical
     circuit development.

  3. GluN1 / GRIN1 IN INHIBITORY INTERNEURONS — KEY PATHOGENIC MECHANISM:
     Parvalbumin-positive (PV+) fast-spiking interneurons have the HIGHEST NMDA receptor
     density in the cortex. PV+ interneurons are critically dependent on NMDA-R for:
       • Sustaining high-frequency firing (>200 Hz) through Ca²⁺-dependent mechanisms
       • Receiving recurrent excitatory input from pyramidal cells (E→I feedback)
       • Generating GABA-mediated inhibitory postsynaptic currents (IPSCs) on pyramidal cells
     GRIN1 LOF → reduced NMDA-R function in PV+ interneurons → PV+ firing failure →
     cortical DISINHIBITION → pyramidal cell hyperexcitability → SEIZURES.
     Schizophrenia model: GluN1 hypomorph mice (10% expression) → cortical
     disinhibition + hyperlocomotion + cognitive deficits (schizophrenia-like phenotype).

  4. GRIN1 LOSS-OF-FUNCTION MECHANISM:
     De novo LOF variants (nonsense, frameshift, splicing, hypomorphic missense):
       • Reduce GluN1 protein expression → fewer surface NMDA-Rs
       • Reduce glycine-site affinity → higher co-agonist requirement → functionally
         hypo-active NMDA-Rs at physiological glycine/D-serine concentrations
       • Alter CTD-PSD interaction → impaired NMDA-R synaptic anchoring →
         receptor mistargeting to extrasynaptic or intracellular compartments
     Net: synaptic NMDA hypofunction → PV+ interneuron failure → cortical disinhibition
     → epilepsy + hyperkinetic movement disorder + severe intellectual disability.
     D-SERINE PRECISION: exogenous D-serine saturates the glycine co-agonist site of
     hypomorphic GluN1 → partially rescues NMDA-R function → used in LOF phenotypes.

  5. GRIN1 GAIN-OF-FUNCTION MECHANISM (RARE):
     GOF de novo missense variants (especially in M2 pore loop and M3 gate):
       • p.Asn615Lys: equivalent to lurcher mutation in GluA1 AMPA receptor — removes
         M2 asparagine that coordinates Mg²⁺ → disrupts Mg²⁺ block → constitutive
         Ca²⁺ influx even at resting potential → excitotoxicity.
       • p.Leu812Met, p.Tyr647Ser: M3 gate variants → reduced channel closure → prolonged
         openings → Ca²⁺ overload → neuronal death during critical period.
       • p.Asn616Lys, p.Gly618Arg: pore-loop variants → altered ion selectivity.
     Net: Ca²⁺ excitotoxicity during early brain development → severe DEE with epileptic
     spasms, profound ID, early death in most severe GOF variants.
     MEMANTINE PRECISION: memantine is a low-affinity, open-channel NMDA blocker that
     preferentially reduces pathologically elevated NMDA activity while sparing normal
     NMDA-mediated synaptic transmission (voltage-dependent, activity-dependent blocking).

  6. GLYCINE/D-SERINE CO-AGONIST SITE — THERAPEUTIC TARGET:
     GluN1 has 8 splice isoforms (NR1-1a to NR1-4b) with different N1 cassette insertion.
     The glycine-binding ABD of GluN1 is structurally homologous to bacterial amino-acid
     binding proteins — a clamshell domain that binds glycine (Kd ~100 nM) or D-serine
     (Kd ~700 nM; higher potency than glycine at saturating doses for some variants).
     In vivo: astrocytes synthesise and release D-serine via serine racemase (SRR);
     D-serine is the dominant co-agonist at forebrain synaptic NMDA-Rs.
     GRIN1 LOF → reduced co-agonist affinity → hypofunction correctable by D-serine
     supplementation → rational basis for D-serine therapy in GRIN1 LOF.
     D-serine dose (case reports): 50–200 mg/kg/day PO; monitor serum D-serine,
     kidney function (D-serine at high dose nephrotoxic in animal models — monitor creatinine).

  7. GENOTYPE–PHENOTYPE (Lemke 2016, Bhatt 2023):
     GOF pore/gate variants (p.Asn615Lys, p.Leu812Met): severe DEE — early spasms;
       worst outcomes; excitotoxic mechanism; memantine rationale.
     LOF severe (nonsense/frameshift, hypomorphic missense ABD): severe DEE + HkMD;
       epileptic spasms + focal + myoclonic; D-serine rationale.
     LOF moderate (hypomorphic missense NTD/CTD): moderate DEE + HkMD; some
       retained communication; D-serine trial reasonable.
     NTD/ABD missense (intermediate): variable — some respond to standard AEDs;
       functional testing needed to classify GOF vs LOF.
     All variants: de novo (>95%); pLI ~0.99.

  8. POPULATION GENETICS:
     Gene: GRIN1, 9q34.3. 22 exons; 938 aa GluN1 protein (longest isoform NR1-1a).
     pLI ~0.99 (extreme intolerance — one of highest in genome; reflects essential role).
     Inheritance: AD de novo (>95%). Familial GRIN1 epilepsy extremely rare.
     OMIM #616346 (GRIN1 neurodevelopmental syndrome).
     Prevalence: ~200–300 confirmed patients worldwide (2024); significantly underdiagnosed.
     Discovery: Lemke et al. 2016 Ann Neurol 80(5):726–741.

CLINICAL DIAGNOSIS:
  - Severe/profound intellectual disability (universal in GRIN1 NDD)
  - Hyperkinetic movement disorder: choreoathetosis, oculogyric-like episodes, stereotypies
  - Epilepsy: focal motor, epileptic spasms, myoclonic, tonic — may be refractory
  - Hypotonia (axial and limb)
  - Absent/minimal speech (most GRIN1 NDD patients non-verbal)
  - Autism spectrum features (>60%)
  - EEG: diffuse slow background, multifocal epileptiform discharges, hypsarrhythmia (spasms)
  - MRI: often non-specific (cerebral atrophy, reduced white matter volume, delayed myelination)
  - KEY CLINICAL PEARL: Movement disorder + severe ID + epilepsy in same patient → include
    GRIN1 on differential (alongside GNAO1, GNB1, STXBP1, ARX, FOXG1)

INHERITANCE AND GENETICS:
  AUTOSOMAL DOMINANT. De novo: >95%. pLI ~0.99.
  Locus: 9q34.3. Key LOF variants: p.Gly595Val · p.Arg844Cys · p.Glu837Lys · p.Leu643Arg ·
  frameshift/nonsense. Key GOF: p.Asn615Lys · p.Leu812Met · p.Tyr647Ser · p.Asn616Lys.

KEY REFERENCES:
  Lemke et al. 2016 Ann Neurol 80(5):726-741 — GRIN1 NDD discovery cohort (13 patients)
  Bhatt et al. 2023 Epilepsia — gene-epilepsy reference; GRIN1 definitive standard
  Bhattacharya & Bhatt 2020 Front Neurol — NMDA channelopathy treatment framework
  Yuan et al. 2015 Neuron 87(2):294-303 — GluN1 knockdown → schizophrenia-like phenotype
  Bhatt 2022 Epilepsia 63(S1):S76 — D-serine/memantine precision in GRIN disorders
  ILAE Commission on Classification and Terminology 2022
"""
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GRIN1-LOF-Severe-DEE",
        "pct": 35,
        "etiology": "GRIN1 LOF severe — nonsense/frameshift/hypomorphic missense ABD; DEE + epileptic spasms",
        "mechanism": (
            "Loss-of-function variants (nonsense, frameshift, splice-site, or hypomorphic missense "
            "in the agonist-binding domain) dramatically reduce GluN1 protein expression or glycine/"
            "D-serine affinity. Surface NMDA-R density drops ≥50%. PV+ interneurons, which have the "
            "highest NMDA-R density in cortex, fail to sustain high-frequency firing → cortical "
            "disinhibition → epileptic spasms onset 3–12 months (hypsarrhythmia on EEG). Severe DEE "
            "trajectory. D-serine supplementation rationale: saturate the hypomorphic glycine site "
            "→ partial NMDA-R rescue → reduced seizure frequency and improved alertness in case reports."
        ),
        "typical_variants": "Frameshift / nonsense (protein <50%) · p.Gly595Val · p.Arg844Cys (ABD missense)",
        "onset_age_years": 0.5,
        "outcome": "Severe DEE; most patients non-verbal; epilepsy partially controlled; D-serine trial reasonable at any age",
    },
    {
        "category": "GRIN1-LOF-HkMD-Moderate",
        "pct": 30,
        "etiology": "GRIN1 LOF moderate — NTD/CTD hypomorphic missense; hyperkinetic movement disorder predominant",
        "mechanism": (
            "Hypomorphic missense variants in the N-terminal domain (NTD) or C-terminal domain (CTD) "
            "of GluN1 reduce NMDA-R surface trafficking (CTD variants) or alter allosteric modulation "
            "(NTD variants) without completely abolishing co-agonist binding. NMDA-R function is "
            "30–60% of wildtype. PV+ interneuron hypofunction → cortical disinhibition → "
            "hyperkinetic movement disorder (choreoathetosis, oculogyric episodes, stereotypies) "
            "± seizures. Movement disorder often MORE prominent than epilepsy (distinguishing "
            "feature from GRIN2B LOF where epilepsy usually dominates). Some patients retain "
            "limited expressive language. D-serine may improve motor coordination alongside seizure control."
        ),
        "typical_variants": "p.Glu837Lys · p.Leu643Arg · p.Thr495Met (NTD) · CTD truncation variants",
        "onset_age_years": 1,
        "outcome": "Moderate DEE + prominent HkMD; limited communication; D-serine trial if LOF confirmed; movement disorder management",
    },
    {
        "category": "GRIN1-GOF-DEE-Excitotoxic",
        "pct": 22,
        "etiology": "GRIN1 GOF de novo — M2/M3 pore-gate variants; excitotoxic DEE; memantine precision",
        "mechanism": (
            "Gain-of-function variants in the M2 pore loop (p.Asn615Lys — 'lurcher equivalent') or "
            "M3 gate helix (p.Leu812Met, p.Tyr647Ser) constitutively activate GluN1 or remove "
            "Mg²⁺ block. p.Asn615Lys removes the pore-loop asparagine that coordinates Mg²⁺ → "
            "channel opens without depolarisation → continuous Ca²⁺ influx at resting potential → "
            "excitotoxic neuronal death during critical developmental period. EEG: burst-suppression "
            "→ hypsarrhythmia → multifocal discharges. MRI: progressive cerebral atrophy. "
            "Memantine (low-affinity open-channel NMDA blocker): reduces constitutive Ca²⁺ influx "
            "while preserving residual activity-dependent NMDA signalling → partial neuroprotection."
        ),
        "typical_variants": "p.Asn615Lys (lurcher) · p.Leu812Met · p.Tyr647Ser · p.Asn616Lys · p.Gly618Arg",
        "onset_age_years": 0.3,
        "outcome": "Severe DEE; progressive neurological decline in some; memantine as precision; ketamine rescue for SE; high SUDEP risk",
    },
    {
        "category": "GRIN1-NTD-ABD-Intermediate",
        "pct": 8,
        "etiology": "GRIN1 NTD/ABD missense — intermediate phenotype; functional testing required to classify GOF vs LOF",
        "mechanism": (
            "Missense variants in the NTD (allosteric modulation) or ABD clamshell (near but not at "
            "glycine-binding residues) produce intermediate phenotypes: moderate ID + epilepsy without "
            "prominent movement disorder. Functional electrophysiology (Xenopus oocyte or HEK293) "
            "needed to distinguish LOF (→ D-serine) from GOF (→ memantine). Standard AEDs (LEV, VPA) "
            "provide partial control. EEG: generalised or focal slowing with superimposed epileptiform "
            "discharges. Some patients acquire 10–30 words over time. Autism spectrum features common."
        ),
        "typical_variants": "p.Arg337Cys · p.Pro788Leu · p.Glu413Lys (NTD/ABD near-contact residues)",
        "onset_age_years": 1.5,
        "outcome": "Moderate outcomes; standard AED responsive in 40%; precision approach depends on functional testing result",
    },
    {
        "category": "GRIN1-Phenocopy",
        "pct": 5,
        "etiology": "GRIN1 phenocopy — VUS or alternative genetic cause; complex DEE + HkMD phenotype",
        "mechanism": (
            "Patients with clinical phenotype (DEE + hyperkinetic movement disorder + ID) carry a "
            "GRIN1 variant of uncertain significance (VUS) that does not meet criteria for "
            "pathogenicity. Alternative genetic diagnoses considered: GNAO1 (1p36.13), GNB1 (1p36.33), "
            "STXBP1 (9q34.11 — same chromosome arm, syntenic to GRIN1), ARX (Xp21.3), FOXG1 (14q12). "
            "Phenocopy management: empiric standard AEDs (LEV, CLB, KD); no precision NMDA therapy "
            "until GRIN1 pathogenicity established. Re-review with functional data at 12 months."
        ),
        "typical_variants": "VUS or rare population variants (MAF >0.001% in gnomAD v4; pLI context conflict)",
        "onset_age_years": 1,
        "outcome": "Varies; standard DEE management; no GRIN1-specific precision until variant confirmed pathogenic",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT (40 simulated patients)
# ─────────────────────────────────────────────────────────────────────────────
_FIRST = [
    "Aiden", "Beatrix", "Caleb", "Diana", "Emeka", "Fiona", "Giulio", "Hana",
    "Ibrahim", "Jia", "Kiran", "Lena", "Mikael", "Nadia", "Olumide", "Priya",
    "Quinton", "Rania", "Stefan", "Tae-Yang", "Uma", "Viktor", "Wren", "Xiao",
    "Yosef", "Zara", "Andre", "Bella", "Cormac", "Daria", "Elias", "Fatou",
    "Greer", "Hamid", "Ines", "Josip", "Kaia", "Levi", "Marin", "Naomi",
]
_LAST = [
    "Okonkwo", "Hartmann", "Nguyen", "Fernandez", "Petrov", "Nakamura", "Llorente",
    "Johansson", "El-Amin", "Mbeki", "Rao", "Björk", "Kowalski", "DaSilva",
    "Sherif", "Vance", "Müller", "Tanaka", "Pereira", "Adesanya", "Khoury",
    "Lindqvist", "Park", "Romano", "Gupta", "Flynn", "Osei", "Barros", "Holst",
    "Chandra", "Oduya", "Weinstein", "Tran", "Aalbers", "Diallo", "Subramaniam",
    "Olawale", "Berglund", "Fonseca", "Morikawa",
]
_ETIOLOGY_KEYS = [ec["category"] for ec in ETIOLOGY_CATALOG]
_ETIOLOGY_WEIGHTS = [ec["pct"] for ec in ETIOLOGY_CATALOG]

_VARIANTS_BY_ETIOLOGY = {
    "GRIN1-LOF-Severe-DEE": [
        "c.1783C>T (p.Arg595*)", "c.2530_2531del (p.Leu844Glufs*12)",
        "c.1783G>T (p.Gly595Val)", "c.2530C>T (p.Arg844Cys)",
        "c.1928G>A (p.Arg643Gln) hypomorphic", "splice site c.3194-1G>A",
    ],
    "GRIN1-LOF-HkMD-Moderate": [
        "c.2509G>A (p.Glu837Lys)", "c.1927C>T (p.Leu643Arg)",
        "c.1484C>T (p.Thr495Met)", "c.2725C>T (p.Arg909*) CTD",
        "c.1249G>A (p.Val417Ile) NTD hypomorphic",
    ],
    "GRIN1-GOF-DEE-Excitotoxic": [
        "c.1845A>C (p.Asn615Lys) lurcher", "c.2434C>A (p.Leu812Met)",
        "c.1940A>G (p.Tyr647Ser)", "c.1848A>C (p.Asn616Lys)",
        "c.1852G>A (p.Gly618Arg)",
    ],
    "GRIN1-NTD-ABD-Intermediate": [
        "c.1009C>T (p.Arg337Cys)", "c.2362C>T (p.Pro788Leu)",
        "c.1237G>A (p.Glu413Lys)", "c.1105C>T (p.Arg369Cys)",
    ],
    "GRIN1-Phenocopy": [
        "c.2176G>A (p.Val726Ile) VUS", "c.1456C>T (p.Arg486Trp) VUS",
        "c.779G>A (p.Arg260His) VUS",
    ],
}

_AED_BY_ETIOLOGY = {
    "GRIN1-LOF-Severe-DEE": [
        "D-serine + LEV", "LEV + VPA + D-serine", "LEV + CLB", "VPA + LEV + D-serine",
        "KD + D-serine", "ACTH (spasms) → LEV + D-serine",
    ],
    "GRIN1-LOF-HkMD-Moderate": [
        "D-serine + LEV", "LEV + CLB", "D-serine + VPA", "LEV + LCM", "VPA + CLB + D-serine",
    ],
    "GRIN1-GOF-DEE-Excitotoxic": [
        "Memantine + LEV", "Memantine + VPA", "Memantine + CLB + LEV", "KD + memantine",
        "ACTH (spasms) → memantine + LEV",
    ],
    "GRIN1-NTD-ABD-Intermediate": [
        "LEV", "LEV + VPA", "LEV + CLB", "VPA + CLB",
    ],
    "GRIN1-Phenocopy": [
        "LEV", "VPA", "LEV + CLB",
    ],
}

_OUTCOME_MAP = {
    "GRIN1-LOF-Severe-DEE": ["Partial control", "Refractory", "Improved with D-serine"],
    "GRIN1-LOF-HkMD-Moderate": ["Partial control", "Moderate improvement", "Improved with D-serine"],
    "GRIN1-GOF-DEE-Excitotoxic": ["Refractory", "Partial control with memantine", "Progressive decline"],
    "GRIN1-NTD-ABD-Intermediate": ["Partial control", "Good control", "Moderate improvement"],
    "GRIN1-Phenocopy": ["Standard DEE management", "Partial control"],
}

_etiology_pool = []
for ec in ETIOLOGY_CATALOG:
    _etiology_pool.extend([ec["category"]] * ec["pct"])

PATIENTS = []
for i, (fn, ln) in enumerate(zip(_FIRST, _LAST)):
    rng = random.Random(i + 100)
    etiology = _etiology_pool[i % len(_etiology_pool)]
    if etiology == "GRIN1-LOF-Severe-DEE":
        age_onset = round(rng.uniform(0.25, 0.75), 2)
        age_now = round(rng.uniform(3, 18), 1)
    elif etiology == "GRIN1-LOF-HkMD-Moderate":
        age_onset = round(rng.uniform(0.5, 2), 2)
        age_now = round(rng.uniform(5, 20), 1)
    elif etiology == "GRIN1-GOF-DEE-Excitotoxic":
        age_onset = round(rng.uniform(0.1, 0.5), 2)
        age_now = round(rng.uniform(2, 15), 1)
    elif etiology == "GRIN1-NTD-ABD-Intermediate":
        age_onset = round(rng.uniform(0.5, 3), 2)
        age_now = round(rng.uniform(5, 25), 1)
    else:
        age_onset = round(rng.uniform(0.5, 2), 2)
        age_now = round(rng.uniform(4, 22), 1)

    variant_pool = _VARIANTS_BY_ETIOLOGY.get(etiology, ["Unknown variant"])
    variant = rng.choice(variant_pool)
    aed = rng.choice(_AED_BY_ETIOLOGY.get(etiology, ["LEV"]))
    outcome = rng.choice(_OUTCOME_MAP.get(etiology, ["Partial control"]))
    seizure_free = (
        outcome in ["Good control", "Improved with D-serine", "Improved with memantine"]
        and rng.random() < 0.25
    )

    PATIENTS.append({
        "id": f"GRIN1-{i+1:03d}",
        "name": f"{fn} {ln}",
        "age_at_onset_years": age_onset,
        "age_now_years": age_now,
        "sex": rng.choice(["M", "F"]),
        "etiology_class": etiology,
        "variant": variant,
        "inheritance": "De novo (AD)",
        "current_aed": aed,
        "outcome": outcome,
        "seizure_free": seizure_free,
        "movement_disorder": etiology != "GRIN1-GOF-DEE-Excitotoxic" or rng.random() < 0.4,
        "nonverbal": etiology in ["GRIN1-LOF-Severe-DEE", "GRIN1-GOF-DEE-Excitotoxic"] or rng.random() < 0.6,
        "autism_features": rng.random() < 0.65,
    })

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES (5 types)
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal Motor Seizures",
        "freq_pct": 72,
        "eeg_signature": "Hemispheric or focal epileptiform discharges (frontal/central predominant); ictal: contralateral rhythmic theta/alpha discharge evolving to delta",
        "semiology": "Contralateral tonic or clonic limb movements; head version; eye deviation; may secondarily generalise to GTCS",
        "clinical_tip": (
            "Focal motor pattern in context of severe ID + movement disorder → GRIN1 DEE differential. "
            "Distinguish from GNAO1 movement-related paroxysms (often dystonic, not truly epileptic). "
            "Ictal EEG + clinical correlation essential. LEV (20–60 mg/kg/day) is first-line for focal "
            "seizures in GRIN1 DEE — no NMDA-specific concern, well tolerated."
        ),
    },
    {
        "type": "Epileptic Spasms (Infantile Spasms / West Syndrome)",
        "freq_pct": 55,
        "eeg_signature": "Hypsarrhythmia (modified or classic) → ictal: electrodecremental response (high-amplitude slow wave → abrupt voltage attenuation)",
        "semiology": "Symmetric flexion/extension spasms in clusters (5–50+ spasms per cluster); onset on waking from sleep; may be subtle (head nods, arm extension)",
        "clinical_tip": (
            "Infantile spasms in GRIN1 DEE — treat per UKISS/NICE: ACTH (preferred) or high-dose "
            "prednisolone + vigabatrin. ACTH suppresses CRH/ACTH axis; in GRIN1 LOF, spasm generation "
            "involves corticospinal disinhibition → ACTH rationale. Response assessment at 14 days. "
            "ACTH monitoring: BP 3×/week, glucose daily, electrolytes weekly. "
            "Post-spasm: transition to LEV + D-serine (LOF) or memantine (GOF). "
            "Vigabatrin: limit to 3–6 months total (VFD risk)."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "freq_pct": 48,
        "eeg_signature": "Generalised polyspike-wave or polyspike discharge (1.5–5 Hz); amplitude maximal at Fz/Cz; time-locked to jerk",
        "semiology": "Sudden, brief axial or multifocal jerks; morning predominance; may cluster on waking; distinguished from movement disorder stereotypies by EEG time-locking",
        "clinical_tip": (
            "Movement disorder vs epileptic myoclonus: KEY distinction. EEG back-averaging (jerk-locked "
            "EEG averaging) confirms cortical origin of true myoclonic seizures. "
            "Movement disorder choreiform movements are continuous/semi-rhythmic without EEG correlate. "
            "LEV is preferred for myoclonic component (SV2A mechanism; does not worsen NMDA). "
            "VPA adds myoclonic suppression but POLG1 screen mandatory."
        ),
    },
    {
        "type": "Tonic Seizures",
        "freq_pct": 38,
        "eeg_signature": "High-frequency fast activity (beta-gamma, 20–30 Hz, 'recruiting rhythm') at seizure onset → ictal tonic discharge; electrodecremental post-ictally",
        "semiology": "Nocturnal bilateral tonic posturing (limb extension, jaw clenching); brief 5–30s; often post-ictal cyanosis; may occur in runs during sleep",
        "clinical_tip": (
            "Nocturnal tonic seizures in GRIN1 DEE → video-EEG polygraphy essential to distinguish "
            "from dystonic movement episodes (non-epileptic). "
            "CLB (clobazam) 0.25–1 mg/kg/day effective as add-on for nocturnal tonic seizures. "
            "KD has good evidence for tonic seizures in Lennox-Gastaut-like pattern (which GRIN1 DEE "
            "may evolve into). Avoid CBZ/OXC if myoclonic component co-exists."
        ),
    },
    {
        "type": "Absence-like / Atypical Absence",
        "freq_pct": 28,
        "eeg_signature": "Irregular 2–3 Hz slow spike-wave (not classic 3-Hz); frontally predominant; ictal: behavioural arrest + automatisms concurrent with irregular SWD",
        "semiology": "Staring + unresponsiveness + simple automatisms (lip-smacking, hand movements); 10–45 s duration; post-ictal confusion (unlike typical CAE absence); may co-occur with movement disorder episode",
        "clinical_tip": (
            "Atypical absence in GRIN1 DEE requires careful distinction from behavioural episodes "
            "related to movement disorder (oculogyric episodes may appear as 'absences'). "
            "VPA is effective for atypical absence when standard AED (LEV, CLB) insufficient. "
            "ETX (ethosuximide) NOT indicated — designed for typical 3-Hz SWD in GGE/CAE; "
            "atypical absence in DEE context does not respond to ETX."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS (8 triggers)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Sleep Deprivation / Sleep-Wake Transitions",
        "freq_pct": 82,
        "mechanism": (
            "Sleep deprivation → increased cortical excitability (reduced adenosine-mediated A1 "
            "inhibition; reduced sleep-spindle thalamo-cortical inhibition). In GRIN1 LOF, "
            "PV+ interneuron hypofunctionality is maximally exposed during arousal transitions "
            "(frontal cortex most vulnerable during K-complex/sleep spindle periods). "
            "NREM sleep stage 1/2 transitions → spike frequency increases on EEG."
        ),
        "clinical_advice": (
            "Strict sleep hygiene protocol. Target 10–12 h sleep/24 h (children), 8–9 h (adults). "
            "Regulated wake time — DO NOT allow acute sleep debt. Rescue CLB 5–10 mg PRN on sleep-"
            "deprived nights (pre-emptive). Written sleep plan for caregivers."
        ),
    },
    {
        "trigger": "Fever / Intercurrent Illness",
        "freq_pct": 75,
        "mechanism": (
            "Fever >38°C → NMDA-R current increases with temperature (Q10 ~2.3 for NMDA open "
            "probability in some subunit compositions). In GRIN1 GOF variants → additional Ca²⁺ "
            "influx during fever. In LOF variants: fever increases metabolic demand → neurons "
            "more dependent on NMDA-R for sustaining GABAergic interneuron firing (paradoxically, "
            "fever can worsen disinhibition in GRIN1 LOF). AED metabolism accelerated at high T."
        ),
        "clinical_advice": (
            "Early antipyretic: paracetamol 15 mg/kg q4-6h at first sign of fever. "
            "Written fever action plan with threshold-triggered dose escalation. "
            "Rescue: buccal midazolam 0.1 mg/kg for seizures ≥3 min. "
            "For GOF patients: consider ketamine low-dose IM (0.5–1 mg/kg) as hospital rescue "
            "for fever-induced prolonged focal SE — discuss with specialist."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "freq_pct": 62,
        "mechanism": (
            "Sub-therapeutic AED levels → acute lowering of seizure threshold. "
            "In GRIN1 LOF with D-serine therapy: missed D-serine dose → acute reduction of "
            "glycine-site occupancy → GluN1 hypomorphic variant NMDA-Rs return to deeper "
            "hypofunction → PV+ interneuron disinhibition → seizure cluster. "
            "In GOF with memantine: missed dose → constitutive Ca²⁺ influx resumes → breakthrough."
        ),
        "clinical_advice": (
            "Blister-pack pill organiser with alarm. Caregiver education: 'double dose PROHIBITED.' "
            "If >50% dosing interval missed → give dose immediately; if >75% → skip to next. "
            "For D-serine: powder formulation stability guidance (refrigerate, use within 30 days). "
            "NG tube administration if swallowing difficulties."
        ),
    },
    {
        "trigger": "Physiological Stress / Emotional Arousal",
        "freq_pct": 55,
        "mechanism": (
            "Acute stress → CRH/ACTH/cortisol surge → CRF receptors on PV+ interneurons modulate "
            "NMDA-R phosphorylation (PKA/PKC) → alters GluN1 channel gating. "
            "Noradrenaline (stress-evoked) → α1 adrenoceptor → IP3 → ER Ca²⁺ release → modulates "
            "NMDA-R. In GRIN1 LOF: stress-induced reduction in cortical NMDA tone further impairs "
            "PV+ → disinhibition → focal seizures."
        ),
        "clinical_advice": (
            "Occupational therapy: structured daily routine to minimise stress/novelty. "
            "Communication support (AAC — augmentative/alternative communication) to reduce "
            "communication frustration (key stress source in non-verbal GRIN1 NDD). "
            "If emotional arousal precedes seizures: PRN CLB 5 mg protocol before anticipated "
            "stressful events (medical appointments, travel)."
        ),
    },
    {
        "trigger": "Sensory Stimulation (Startle / Unexpected)",
        "freq_pct": 42,
        "mechanism": (
            "Startle → bilateral tonic/myoclonic response via corticospinal + brainstem pathways. "
            "In GRIN1 DEE with cortical hyperexcitability, sudden auditory or tactile stimuli "
            "trigger abnormal cortical recruitment via reticulothalamic arousal networks. "
            "Pathological startle response distinguishable from hyperekplexia (GLRA1/GLRB) by "
            "EEG: GRIN1 startle seizures have cortical onset; hyperekplexia has brainstem origin."
        ),
        "clinical_advice": (
            "Low-stimulation environment: reduce sudden loud noises, abrupt touch. "
            "Soft voice, approach from visual field. Positioning during bathing (startle-prone). "
            "CLB as maintenance add-on reduces startle response (benzodiazepine GABAergic augmentation "
            "compensates for PV+ hypofunction). Anti-startle pad/helmet for fall protection."
        ),
    },
    {
        "trigger": "Hyperventilation / Breath-Holding",
        "freq_pct": 35,
        "mechanism": (
            "Hyperventilation → respiratory alkalosis → CO₂ wash-out → [H⁺] decreases → "
            "extracellular alkalosis → NMDA-R proton inhibition reduced (GluN1 NTD has proton-"
            "sensing site; alkalosis removes tonic proton block → NMDA-R more active). "
            "In GRIN1 GOF: alkalosis removes an additional physiological brake on constitutive "
            "Ca²⁺ influx → amplified excitotoxic signaling. In LOF: alkalosis → paradoxical "
            "mild facilitation of residual NMDA-Rs (small effect)."
        ),
        "clinical_advice": (
            "For GOF patients: monitor respiratory rate; treat early respiratory infections preventing "
            "hyperventilation. Breath-holding spells: first distinguish from epilepsy (EEG correlation). "
            "In agitated GOF patients: paper bag re-breathing NOT recommended (CO₂ retention risk). "
            "Avoid anxious hyperventilation by caregiver during seizure (models behaviour)."
        ),
    },
    {
        "trigger": "Glutamate-Potentiating Substances or Drugs",
        "freq_pct": 28,
        "mechanism": (
            "Certain drugs partially potentiate NMDA-R in GOF GRIN1 context → excitotoxic risk: "
            "D-cycloserine (high dose > 50 mg → full NMDA co-agonist; safe only as partial agonist "
            "at very low dose); quinolinic acid (tryptophan metabolite, endogenous NMDA agonist "
            "elevated in inflammatory states); glycine (IV high dose in methylmalonic acidemia "
            "treatment — watch for excitotoxicity in GOF). In LOF: avoid NMDA blockers that worsen "
            "hypofunction (e.g. ketamine chronic — fine for rescue, problematic as maintenance)."
        ),
        "clinical_advice": (
            "GOF PATIENTS: Do NOT prescribe NMDA co-agonists (D-cycloserine high dose, glycine IV). "
            "FLAG in medical alert bracelet. LOF PATIENTS: Avoid chronic NMDA blockers "
            "(MK-801 never; ketamine rescue only). In anaesthesia: discuss with neurologist before "
            "ketamine induction (conflicting GOF/LOF effects). Avoid street drugs (MDMA → NMDA "
            "potentiation; cannabis → indirect NMDA modulation). Pyridoxine: safe both LOF+GOF."
        ),
    },
    {
        "trigger": "Abrupt Sleep-State Change / REM-NREM Transition",
        "freq_pct": 22,
        "mechanism": (
            "Sleep state transitions → rapid shifts in cholinergic tone (ACh: M1 muscarinic → "
            "PKC → GluN1 phosphorylation at Ser890/897 → NMDA-R trafficking). REM onset: "
            "cholinergic surge → increased NMDA-R surface expression → in GOF variants, acute "
            "spike in Ca²⁺ influx. NREM → REM transition also suppresses thalamic spindle "
            "activity (removes sleep-spindle seizure protection). EEG: seizure onset often "
            "within 5 min of REM onset or first NREM → REM cycle (~90 min)."
        ),
        "clinical_advice": (
            "Polysomnography may be informative in highly sleep-dependent seizure patterns. "
            "CLB or LEV evening dose timing optimisation (peak levels at projected REM transition "
            "windows). Video-EEG overnight for seizure type classification in ambiguous cases. "
            "For GOF: memantine evening dose to cover REM transition window."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS (8 treatments)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "D-serine (glycine-site co-agonist — GRIN1 LOF precision)",
        "level": "Level C",
        "indication": "GRIN1 LOF variants with documented reduced GluN1 glycine-site affinity or reduced NMDA-R surface expression; DEE + hyperkinetic movement disorder",
        "dose": (
            "50–200 mg/kg/day PO in 2–3 divided doses (titrate from 50 mg/kg/day, increase "
            "q4 weeks by 25–50 mg/kg/day). Powder formulation (compounding pharmacy). "
            "Paediatric: target 100–150 mg/kg/day; Adult: 50–100 mg/kg/day (lower on mg/kg basis)."
        ),
        "moa": (
            "D-serine is the dominant synaptic co-agonist at forebrain NMDA-Rs (synthesised by "
            "astrocytes via serine racemase from L-serine). Exogenous D-serine saturates the "
            "GluN1 glycine/D-serine binding site (ABD) → rescues NMDA-R hypofunction in LOF "
            "variants with reduced co-agonist affinity → partially restores PV+ interneuron "
            "function → reduces cortical disinhibition → fewer seizures + improved motor control. "
            "D-serine is preferred over glycine: higher CNS penetrance, better tolerability, "
            "less G/I side effects than high-dose glycine."
        ),
        "efficacy": "Case reports/series (n<20): ≥50% seizure reduction in ~40%; movement disorder improvement in ~50%. No RCT yet.",
        "safety": (
            "Nephrotoxicity: D-serine is metabolised by D-amino acid oxidase (DAO) in kidney → "
            "H₂O₂ → oxidative tubular injury at high doses (animal: >1000 mg/kg/day). "
            "Clinical monitoring: serum creatinine + urinalysis q3M. "
            "GI: mild nausea/loose stools at initiation — transient; use with food. "
            "No known drug-drug interactions with LEV/VPA/CLB."
        ),
        "monitoring": "Serum creatinine + urinalysis q3M · D-serine plasma level if available · seizure diary · movement disorder scale",
        "grin1_note": "Precision therapy: ONLY in confirmed GRIN1 LOF. NOT indicated in GOF (D-serine would worsen excitotoxicity). Functional testing (Xenopus/HEK293) ideal before prescribing.",
    },
    {
        "drug": "Memantine (NMDA open-channel blocker — GRIN1 GOF precision)",
        "level": "Level C",
        "indication": "GRIN1 GOF variants (M2 pore/M3 gate variants, lurcher-equivalent); excitotoxic DEE phenotype",
        "dose": (
            "Paediatric (off-label): 0.1–0.5 mg/kg/day, titrate over 4–8 weeks to 0.5–1 mg/kg/day "
            "(max 20 mg/day). Start 1 mg/day, increase by 1–2 mg/week. "
            "Adult: standard 5 mg/day → 10 mg bd (Alzheimer's disease dose; higher tolerated in case reports)."
        ),
        "moa": (
            "Memantine: low-affinity (Ki ~1 µM), open-channel, voltage-dependent NMDA receptor "
            "blocker. Preferentially blocks NMDA-Rs in a state of pathologically excessive activation "
            "(constitutive in GOF variants) — spares normal activity-dependent NMDA signalling "
            "needed for learning (because fast off-rate allows channel to re-open with physiological "
            "stimulation). In GRIN1 GOF: reduces constitutive Ca²⁺ influx → neuroprotection → "
            "fewer excitotoxic insults → reduced seizure frequency."
        ),
        "efficacy": "Case reports/series (n<10 GRIN1 GOF): ≥50% seizure reduction in ~35–40%; some progressive stabilisation.",
        "safety": (
            "Dizziness, agitation, confusion (titration-related, especially in children). "
            "Rare: hallucinations (higher doses). Safe in renal impairment but reduce dose "
            "(renally excreted). No hepatotoxicity. No significant DDI with LEV/CLB."
        ),
        "monitoring": "Seizure diary · developmental milestones · neurological exam q3M · MRI if progressive atrophy concerns",
        "grin1_note": "Precision therapy: ONLY in confirmed GRIN1 GOF. NOT indicated in LOF (memantine would worsen NMDA hypofunction and increase disinhibition). Genotype-first.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": "First-line AED for all GRIN1 DEE seizure types (focal, myoclonic, spasms post-ACTH); SV2A mechanism does not affect NMDA-R",
        "dose": (
            "Paediatric: 20–60 mg/kg/day (oral solution or tablet) in 2 divided doses. "
            "Start 10 mg/kg/day, titrate q2w by 10–20 mg/kg. "
            "Adult: 1000–3000 mg/day. IV form available for acute/SE."
        ),
        "moa": (
            "Binds synaptic vesicle glycoprotein 2A (SV2A) — modulates synaptic vesicle trafficking "
            "and neurotransmitter release. Does NOT block Na⁺ channels or modulate GABA/NMDA directly. "
            "Reduces excessive synchronised neuronal firing without affecting basal synaptic "
            "transmission — well tolerated in DEE + movement disorder context."
        ),
        "efficacy": "Broad-spectrum; 50% seizure reduction in ~50% of DEE patients. Best for focal + myoclonic in GRIN1.",
        "safety": (
            "Irritability/aggression (especially in children with ID — 'behavioural adverse event'): "
            "mitigate with pyridoxine 10–20 mg/day (B6 co-supplementation). "
            "No hepatotoxicity, no POLG1 concern, no teratogenicity signal (relative to VPA). "
            "Reduce dose in renal impairment (eGFR-based)."
        ),
        "monitoring": "LEV TDM (target 12–45 mg/L) · behaviour rating scale · seizure diary",
        "grin1_note": "Safe in both LOF and GOF GRIN1 — does not modulate NMDA-R. First line regardless of variant type.",
    },
    {
        "drug": "Valproate (VPA)",
        "level": "Level B",
        "indication": "Broad-spectrum AED add-on for GRIN1 DEE (GTCS, tonic, atypical absence); POLG1 screen mandatory",
        "dose": (
            "Paediatric: 20–40 mg/kg/day in 2–3 divided doses (syrup or ER tablet). "
            "Adult: 1000–2500 mg/day. Target TDM: 50–100 mcg/mL. "
            "Start low (10 mg/kg/day), titrate q1–2 weeks."
        ),
        "moa": (
            "Multi-target: Na⁺ channel blockade (repetitive firing); GABA-T inhibition (↑ GABA); "
            "T-type Ca²⁺ channel blockade (thalamo-cortical synchrony); histone deacetylase "
            "inhibition (epigenetic — relevant in DEE). Does not modulate NMDA-R."
        ),
        "efficacy": "Broad-spectrum: 60–70% GTCS reduction; myoclonic 65–70%; atypical absence 50%.",
        "safety": (
            "VPPP MANDATORY (valproate pregnancy prevention programme) — ALL females 12–50Y. "
            "POLG1/POLG MANDATORY screen before starting (Alpers syndrome risk in POLG1 carriers). "
            "Hepatotoxicity: LFT + ammonia q3M. Weight gain, alopecia, tremor. "
            "Interaction: carbapenem antibiotics reduce VPA levels 60–90% acutely."
        ),
        "monitoring": "POLG1 before starting · VPA TDM q3M · LFT/FBC/ammonia q3M · VPPP programme",
        "grin1_note": "Useful broad-spectrum add-on. POLG1 screen before use — GRIN1 DEE patients often have NG/gastrostomy tubes and may receive broad antibiotics (carbapenem interaction risk).",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B",
        "indication": "Add-on for nocturnal tonic and myoclonic seizures; startle seizures; rescue PRN in GRIN1 DEE",
        "dose": (
            "0.25–1 mg/kg/day in 1–2 divided doses (evening-weighted for nocturnal seizures). "
            "Start 2.5 mg nocte, increase q2w by 2.5–5 mg. Max 40 mg/day (adults). "
            "PRN: 5–10 mg buccal suspension for rescue or pre-stressor pre-emptive dosing."
        ),
        "moa": (
            "1,5-benzodiazepine → GABA-A positive allosteric modulator → enhances Cl⁻ conductance. "
            "Compensates for PV+ interneuron hypofunction (GRIN1 LOF) by potentiating remaining "
            "GABAergic inhibition via non-PV interneuron pathways (VIP, SST) that are less NMDA-"
            "dependent. Better tolerated than 1,4-BZD (diazepam, clonazepam) — less sedation, "
            "less cognitive impairment, longer tolerability."
        ),
        "efficacy": "50% seizure reduction in ~55% of DEE patients; best for tonic/nocturnal component.",
        "safety": (
            "Tolerance with prolonged daily use (receptor downregulation). "
            "Schedule as daily OR use intermittent CLB (3 days on/4 off or catamenial protocol). "
            "Sedation, drooling (especially NG tube patients). FDA REMS programme for Onfi® brand "
            "(risk of serious skin reactions SJS/TEN — rare). "
            "Withdrawal risk if abrupt stop: taper over 4+ weeks."
        ),
        "monitoring": "Sedation score · respiration monitoring · seizure diary · CLB level if breakthrough",
        "grin1_note": "Evening-dose CLB optimised for nocturnal tonic seizures in GRIN1 DEE. PRN buccal CLB for startle-precipitated clusters.",
    },
    {
        "drug": "ACTH / High-Dose Prednisolone (for epileptic spasms phase)",
        "level": "Level A (infantile spasms)",
        "indication": "Epileptic spasms / West syndrome phase of GRIN1 DEE (onset 3–12 months); standard IS treatment regardless of aetiology",
        "dose": (
            "ACTH (tetracosactide / Synacthen): 0.5 mg (75 IU) IM alt days × 2 weeks → taper; "
            "or natural porcine ACTH 40–160 IU/day (where available). "
            "High-dose prednisolone: 4 mg/kg/day (max 40 mg) × 2 weeks (UKISS protocol). "
            "Combine with Vigabatrin 50–150 mg/kg/day (limit to 3–6 months for VFD risk)."
        ),
        "moa": (
            "ACTH suppresses hypothalamic-pituitary-adrenal (HPA) axis → reduces excess CRH "
            "(corticotropin-releasing hormone) → reduces CRH-mediated cortical hyperexcitability "
            "(CRH has direct proconvulsant effect in developing brain). Additionally: ACTH/steroid "
            "→ anti-inflammatory → reduces neuroinflammation in immature cortex. "
            "In GRIN1 LOF: cortical disinhibition is the spasm generator; ACTH addresses "
            "the CRH/cortical component. Vigabatrin adds GABAergic augmentation."
        ),
        "efficacy": "Spasm cessation (EEG + clinical) at 14 days: ~65% with ACTH+VGB (UKISS). GRIN1-specific data limited.",
        "safety": (
            "ACTH: hypertension (BP 3×/week), hyperglycaemia (glucose daily), hypokalaemia "
            "(electrolytes weekly), infection risk (avoid live vaccines during course), "
            "irritability/Cushing's features (transient). VGB: visual field defect (irreversible, "
            "cumulative dose-dependent) → ERG q3M during VGB; limit duration."
        ),
        "monitoring": "BP 3×/week during ACTH · glucose daily · electrolytes weekly · ERG q3M (VGB) · EEG at 14 days",
        "grin1_note": "Standard IS treatment regardless of GRIN1 variant type. After spasm control, transition to precision: D-serine (LOF) or memantine (GOF) + LEV.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B",
        "indication": "Refractory GRIN1 DEE seizures after 2+ AED failures; especially focal + tonic component; both LOF and GOF",
        "dose": (
            "Classical KD 4:1 ratio (fat:protein+carbohydrate) or modified Atkins diet (MAD). "
            "Initiate with dietitian + neurologist. Target β-hydroxybutyrate (BHB) 2–4 mM. "
            "Gradual introduction over 1–2 weeks. G-tube delivery in non-oral GRIN1 patients."
        ),
        "moa": (
            "Ketone bodies (BHB, acetoacetate) → multiple anticonvulsant mechanisms: "
            "(1) reduces glycolytic flux → less glutamate synthesis (aspartate aminotransferase); "
            "(2) BHB inhibits vesicular glutamate transporter VGLUT — reduces glutamate vesicle loading; "
            "(3) adenosine kinase inhibition → increased adenosine → A1 inhibition; "
            "(4) NLRP3 inflammasome inhibition (neuroprotective); "
            "(5) may partially compensate for reduced NMDA signalling at GABAergic synapses "
            "through alternative energy substrate provision to PV+ interneurons (high metabolic "
            "demand interneurons particularly benefit from ketone supplementation)."
        ),
        "efficacy": "≥50% seizure reduction in ~50–55% of refractory DEE; complete seizure freedom ~5–10%.",
        "safety": (
            "Growth: monitor height/weight (growth failure risk — supplement with calcium, vitamin D, "
            "selenium, zinc). Renal stones (4–6%): hydration + potassium citrate. "
            "Carnitine supplementation (20–50 mg/kg/day). GI: constipation, nausea (titration). "
            "Glucose: avoid hypoglycaemia (G-tube patients: continuous feeding monitoring). "
            "Acidosis: baseline serum HCO₃⁻; monitor if adding AZM."
        ),
        "monitoring": "BHB 2–4 mM · growth parameters · renal ultrasound at 6M · lipid panel · carnitine · HCO3",
        "grin1_note": "G-tube formulation: KetoCal® powder mixed in feeds. KD particularly beneficial for PV+ interneuron metabolic support hypothesis in GRIN1 LOF. Both LOF and GOF benefit.",
    },
    {
        "drug": "Topiramate (TPM) — third-line, caution with D-serine acidosis",
        "level": "Level C",
        "indication": "Third-line add-on for refractory focal and tonic seizures in GRIN1 DEE; caution with KD or D-serine (metabolic acidosis synergy)",
        "dose": (
            "Paediatric: 3–9 mg/kg/day in 2 divided doses. Start 0.5–1 mg/kg/day, increase "
            "q2w by 1 mg/kg. Adult: 200–600 mg/day. Sprinkle capsule for NG tube patients."
        ),
        "moa": (
            "Multiple: (1) Na⁺ channel blockade; (2) AMPA/kainate receptor antagonism "
            "(reduces non-NMDA glutamate signalling — complementary in GRIN1 GOF where all "
            "excitatory pathways are upregulated); (3) GABA-A potentiation; "
            "(4) Carbonic anhydrase inhibition (mild metabolic acidosis — anti-seizure but "
            "synergistic acidosis risk with KD or D-serine metabolic effects)."
        ),
        "efficacy": "≥50% seizure reduction in ~40% as add-on in refractory DEE.",
        "safety": (
            "Cognitive impairment ('word-finding difficulty', slowed processing) — significant concern "
            "in GRIN1 NDD where cognitive function is already severely impaired. "
            "Weight loss (not always negative in some patients). Oligohidrosis/hyperthermia "
            "(heat exposure caution, especially in non-communicating patients). "
            "Renal stones (risk synergised by KD + CA inhibition). Metabolic acidosis: "
            "ABSOLUTE CI: TPM + KD + D-serine (triple acidosis risk) — check serum HCO₃⁻ q4–6w."
        ),
        "monitoring": "Serum HCO3 q4-6w if KD/D-serine co-prescribed · weight · temperature-related symptoms · cognitive battery",
        "grin1_note": "Use cautiously with D-serine (both metabolically active). HCO3 monitoring essential. Avoid if serum HCO3 <18 mEq/L.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS (5 contraindications)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug_or_class": "MK-801 (Dizocilpine) — ABSOLUTE CI (not approved for clinical use)",
        "risk": "ABSOLUTE CI",
        "mechanism": (
            "MK-801 is a high-affinity, irreversible open-channel NMDA blocker (Ki ~0.1 nM; "
            "vs memantine Ki ~1 µM). Clinical toxicity: severe dissociation, hallucinations, "
            "ataxia, potentially fatal respiratory depression. Not approved for human clinical use. "
            "In GRIN1 LOF: MK-801 would catastrophically worsen NMDA hypofunction (blocking "
            "the already severely reduced NMDA activity) → complete PV+ interneuron failure. "
            "In GRIN1 GOF: MK-801 overdose risk due to narrow therapeutic window. "
            "NEVER prescribe MK-801 to any GRIN1 patient regardless of variant type."
        ),
        "clinical_action": "Medical alert bracelet: 'MK-801 (dizocilpine) ABSOLUTELY CONTRAINDICATED.' Document in all records.",
    },
    {
        "drug_or_class": "D-serine / NMDA co-agonists in GRIN1 GOF — ABSOLUTE CI",
        "risk": "ABSOLUTE CI in GOF",
        "mechanism": (
            "D-serine, high-dose glycine, and rapastinel (GLYX-13) are NMDA co-agonists that "
            "potentiate GluN1 activity at the glycine/D-serine site. In GRIN1 GOF (lurcher-"
            "equivalent, M2/M3 variants): GluN1 already has constitutive Ca²⁺ influx → "
            "adding D-serine further potentiates the excitotoxic current → acute seizure "
            "worsening, excitotoxic neuronal death, hyperthermia. Documented in animal "
            "models of GOF NMDA variants. D-cycloserine at high doses (full agonist) carries "
            "same risk. D-serine is ONLY for LOF variants."
        ),
        "clinical_action": "Genotype-FIRST before any NMDA-precision therapy. In GOF: 'D-serine CONTRAINDICATED' flag in prescription system.",
    },
    {
        "drug_or_class": "VPA without POLG1 screening — ABSOLUTE CI",
        "risk": "ABSOLUTE CI without screen",
        "mechanism": (
            "Valproate is a potent inhibitor of mitochondrial β-oxidation and inhibits "
            "the mitochondrial respiratory chain at complex II and complex IV. In patients "
            "with POLG1 (polymerase gamma) mutations (Alpers-Huttenlocher syndrome), VPA "
            "causes fulminant hepatotoxicity and mitochondrial decompensation — often fatal. "
            "GRIN1 DEE patients are severely disabled and often unable to report symptoms → "
            "hepatotoxicity detection relies entirely on laboratory monitoring. POLG1 screening "
            "is mandatory BEFORE first VPA dose."
        ),
        "clinical_action": "Order POLG1 sequencing panel before VPA. If POLG1 positive or pending: use LEV as bridge. Document POLG1 status in permanent record.",
    },
    {
        "drug_or_class": "Memantine in GRIN1 LOF — RELATIVE CI / HIGH CAUTION",
        "risk": "HIGH CAUTION / Relative CI in LOF",
        "mechanism": (
            "Memantine blocks NMDA-R open-channel state. In GRIN1 LOF where NMDA-R activity "
            "is already severely reduced (hypofunction), memantine would further suppress the "
            "remaining NMDA signalling → worsen PV+ interneuron failure → worsen cortical "
            "disinhibition → paradoxical increase in seizures. Clinical evidence: memantine "
            "has been tried empirically in GRIN1 LOF and reported to worsen outcomes in "
            "some cases. Variant typing (GOF vs LOF) must be confirmed before memantine. "
            "Phenotypic overlap makes clinical distinction difficult — functional testing is gold standard."
        ),
        "clinical_action": "If GRIN1 LOF suspected or confirmed: do NOT start memantine without neurogenetics sign-off. Document LOF classification basis.",
    },
    {
        "drug_or_class": "Tiagabine (TGB) — ABSOLUTE CI (NCSE risk in DEE)",
        "risk": "ABSOLUTE CI",
        "mechanism": (
            "Tiagabine selectively inhibits GABA transporter 1 (GAT-1) → increases synaptic "
            "GABA dwell time → prolongs inhibition. In DEE with generalised components, TGB "
            "causes paradoxical non-convulsive status epilepticus (NCSE): excess tonic GABAergic "
            "inhibition → synchronised thalamo-cortical oscillation → spike-wave stupor. "
            "GRIN1 DEE has generalised components (epileptic spasms, myoclonic, atypical absence) "
            "→ NCSE risk substantial. TGB has NO indication in paediatric epilepsy."
        ),
        "clinical_action": "NEVER prescribe TGB in GRIN1 DEE. Emergency: if NCSE identified and patient was receiving TGB → immediately discontinue TGB, IV BZD rescue.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING (14 items)
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG1 sequencing panel — MANDATORY before VPA",
     "frequency": "Once (before first VPA dose)",
     "rationale": "Alpers syndrome risk; fulminant hepatotoxicity if POLG1 mutation + VPA"},
    {"item": "GRIN1 variant functional classification (LOF vs GOF)",
     "frequency": "Once — Xenopus oocyte or HEK293 electrophysiology (research centre)",
     "rationale": "Determines precision therapy: D-serine (LOF) vs memantine (GOF) — wrong choice worsens outcomes"},
    {"item": "Seizure diary (video capture + event log)",
     "frequency": "Daily (caregiver-logged); formal review q3M",
     "rationale": "Quantify seizure types, frequency, duration; distinguish epileptic vs movement disorder episodes"},
    {"item": "VPA TDM (target 50–100 mcg/mL)",
     "frequency": "q3M (or 5 days after dose change)",
     "rationale": "Ensure therapeutic window; sub-therapeutic = seizure risk; supra = toxicity in DEE"},
    {"item": "LFT + serum ammonia + FBC",
     "frequency": "q3M (VPA); q6M (non-VPA)",
     "rationale": "VPA hepatotoxicity surveillance; hyperammonaemia (encephalopathy risk in non-verbal patients)"},
    {"item": "D-serine plasma level + serum creatinine + urinalysis",
     "frequency": "q3M (if D-serine prescribed)",
     "rationale": "Nephrotoxicity surveillance (D-amino acid oxidase → H₂O₂ in kidney); confirm co-agonist saturation"},
    {"item": "Movement disorder rating (UHDRS Motor or GNAO1-scale adapted)",
     "frequency": "q3M",
     "rationale": "Track hyperkinetic movement disorder severity independently from seizure burden; D-serine also improves HkMD"},
    {"item": "Developmental assessment (Bayley-4 / Vineland-3 VABS)",
     "frequency": "q6M",
     "rationale": "Monitor neurodevelopmental trajectory; detect developmental stagnation vs regression (GOF excitotoxic decline)"},
    {"item": "EEG (awake + sleep, ≥30 min)",
     "frequency": "Baseline; 3M post-new AED; q12M maintenance; STAT for NCSE suspicion",
     "rationale": "Classify evolving seizure type; monitor for subclinical SE; guide AED adjustment"},
    {"item": "MRI brain (1.5T or 3T with epilepsy protocol)",
     "frequency": "Baseline; repeat at 12M if GOF (monitor progressive atrophy); q24M thereafter",
     "rationale": "Structural correlation; monitor cerebral atrophy in GOF excitotoxic variants; myelination delay in LOF"},
    {"item": "ERG (electroretinography) if Vigabatrin prescribed",
     "frequency": "Baseline + q3M during VGB; 6M after cessation",
     "rationale": "Vigabatrin-associated visual field defect (VAVFD) — irreversible, cumulative dose-dependent; limit VGB duration"},
    {"item": "SUDEP risk counselling + caregiver education",
     "frequency": "Annually + at any GTCS escalation event",
     "rationale": "GRIN1 DEE with GTCS carries elevated SUDEP risk; nocturnal supervision; prone position avoidance during sleep"},
    {"item": "Serum HCO₃⁻ (if KD + topiramate or D-serine co-prescribed)",
     "frequency": "q4-6w during KD initiation; q3M maintenance",
     "rationale": "Triple acidosis risk (KD + TPM + D-serine metabolic effects); HCO3 <18 mEq/L → review drug combination"},
    {"item": "Genetic counselling (de novo GRIN1 — reproductive risk)",
     "frequency": "Once + reassessment at reproductive milestones",
     "rationale": "pLI ~0.99 de novo GRIN1 — recurrence risk <1% (next sibling); parental gonadal mosaicism ~2–5% theoretical; family planning"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE (6 windows)
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "phase": "Neonatal / Early Infancy (0–6 months)",
        "key_events": [
            "First seizures (focal motor or neonatal onset in severe GOF)",
            "Hypotonia recognition — poor suck/swallow → NG tube consideration",
            "Genetic panel: WES/WGS ordered (NICU-era epilepsy gene panel)",
            "POLG1 screening before empiric VPA",
            "GRIN1 variant identified → urgent genotype-phenotype matching (LOF vs GOF)",
        ],
        "clinical_focus": "Seizure type identification, POLG1 before VPA, NG tube, GRIN1 molecular diagnosis",
    },
    {
        "phase": "Infancy / Spasm Phase (6–24 months)",
        "key_events": [
            "Epileptic spasms onset (West syndrome — 55% of GRIN1 DEE)",
            "ACTH + Vigabatrin — UKISS protocol; response at 14 days",
            "Transition post-spasms: LEV + precision (D-serine if LOF / memantine if GOF)",
            "Developmental plateau — ID severity emerging",
            "Movement disorder onset (choreoathetosis, oculogyric episodes)",
        ],
        "clinical_focus": "IS treatment, precision therapy initiation, early developmental intervention, movement disorder assessment",
    },
    {
        "phase": "Early Childhood (2–5 years)",
        "key_events": [
            "Seizure type evolution (spasms → focal/tonic/myoclonic)",
            "D-serine titration (LOF) or memantine optimization (GOF)",
            "KD initiation if ≥2 AEDs failed",
            "Augmentative/alternative communication (AAC) — PECS/SGD for non-verbal GRIN1",
            "EEG surveillance for subclinical SE",
        ],
        "clinical_focus": "AED optimisation, precision pharmacology titration, KD trial, AAC, occupational therapy",
    },
    {
        "phase": "School Age (5–12 years)",
        "key_events": [
            "D-serine / memantine efficacy re-evaluation (formal cognitive + motor testing)",
            "Special needs educational planning (P-scales / IEP — all GRIN1 NDD)",
            "Movement disorder management (OT, physio, baclofen if dystonic)",
            "SUDEP risk counselling — annual; night-time monitoring device (Emfit/SAMi)",
            "Monitoring: serum creatinine (D-serine), VPA TDM, LFT, MRI",
        ],
        "clinical_focus": "Educational placement, SUDEP prevention, precision therapy monitoring, movement disorder management",
    },
    {
        "phase": "Adolescence (12–18 years)",
        "key_events": [
            "VPPP (valproate pregnancy prevention programme) — ALL female GRIN1 patients 12+Y",
            "Pubertal changes may alter seizure frequency (HPA axis + sex hormone NMDA modulation)",
            "Psychosis risk: GRIN1 LOF → NMDA hypofunction → schizophrenia-like prodrome in a minority",
            "Driving: GTCS-free 12M; most GRIN1 NDD patients non-drivers (ID)",
            "Transition planning: adult neurology, adult social services",
        ],
        "clinical_focus": "VPPP, pubertal seizure changes, psychosis screening, transition planning",
    },
    {
        "phase": "Adult (18+ years)",
        "key_events": [
            "Adult epilepsy specialist transition (neurodevelopmental epilepsy unit)",
            "SUDEP prevention: nocturnal supervision, tonic-clonic alarm, prone position policy",
            "Caregiver health: respite support, carer mental health assessment",
            "Reproductive planning: genetic counselling (recurrence risk <1% for de novo GRIN1)",
            "D-serine / memantine continuation or dose adjustment (renal function tracking)",
        ],
        "clinical_focus": "Adult services transition, SUDEP, carer support, reproductive planning, long-term monitoring",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS (15 key concepts)
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "GRIN1 (9q34.3)",
        "definition": (
            "Glutamate Ionotropic Receptor NMDA Type Subunit 1 gene at chromosome 9q34.3. "
            "22 exons; 938 amino acids (NR1-1a isoform, longest); 8 splice variants. "
            "pLI ~0.99 — extreme intolerance (one of highest in genome). Encodes GluN1 (NR1), "
            "the obligatory subunit of all functional NMDA receptors. De novo pathogenic variants "
            "cause GRIN1 neurodevelopmental disorder (OMIM #616346)."
        ),
    },
    {
        "term": "GluN1 / NR1 (NMDA Receptor Obligatory Subunit)",
        "definition": (
            "The GluN1 protein (NR1) is present in all functional NMDA receptors as two copies "
            "per heterotetramer. Contains: NTD (allosteric), ABD/glycine-binding domain (S1-S2), "
            "TMD (M1–M4 pore), and CTD (PSD anchoring). Binds glycine or D-serine as the "
            "obligatory co-agonist — without co-agonist occupancy, NMDA-R cannot open even with "
            "glutamate + depolarisation."
        ),
    },
    {
        "term": "NMDA Receptor Heterotetramers",
        "definition": (
            "NMDA receptors consist of 4 subunits: 2x GluN1 (obligatory) + 2x GluN2 (A/B/C/D) "
            "or 1x GluN2 + 1x GluN3. Diheteromeric (GluN1+GluN2A or GluN2B) most common in "
            "cortex/hippocampus. Triheteromeric (GluN1+GluN2A+GluN2B) prevalent at mature "
            "synapses. Each subunit has ABD; co-agonists: glycine/D-serine at GluN1, glutamate "
            "at GluN2. Ca²⁺-permeable (NMDA) contrasts with Ca²⁺-impermeable AMPA receptors."
        ),
    },
    {
        "term": "Glycine-Site / D-serine Co-agonist",
        "definition": (
            "The GluN1 agonist-binding domain (ABD) binds glycine (Kd ~100 nM) or D-serine "
            "(preferred forebrain co-agonist; released by astrocytes via serine racemase). "
            "Without glycine/D-serine occupancy, the NMDA-R ion channel cannot open — making "
            "this site a critical therapeutic target. GRIN1 LOF variants may reduce co-agonist "
            "affinity → D-serine supplementation saturates the site → rescues NMDA function."
        ),
    },
    {
        "term": "Mg²⁺ Block and Coincidence Detection",
        "definition": (
            "At resting membrane potential (~−70 mV), Mg²⁺ physically blocks the NMDA-R pore "
            "(M2 loop asparagine N615 coordinates Mg²⁺). Block is relieved by postsynaptic "
            "depolarisation — thus NMDA-R only opens when BOTH presynaptic glutamate release "
            "AND postsynaptic depolarisation occur simultaneously (Hebbian coincidence detector). "
            "GRIN1 GOF p.Asn615Lys removes N615 → no Mg²⁺ block → constitutive Ca²⁺ influx."
        ),
    },
    {
        "term": "GRIN1-LOF Hypofunction",
        "definition": (
            "Loss-of-function GRIN1 variants reduce GluN1 protein expression or glycine-site "
            "affinity → fewer functional surface NMDA-Rs → NMDA hypofunction. Most critical: "
            "PV+ interneuron hypofunction (highest NMDA-R density in cortex) → cortical "
            "disinhibition → seizures + hyperkinetic movement disorder. This is the 'NMDA "
            "hypofunction' model of schizophrenia (GluN1 hypomorph mice = schizophrenia model)."
        ),
    },
    {
        "term": "GRIN1-GOF Excitotoxicity (Lurcher Equivalent)",
        "definition": (
            "Gain-of-function GRIN1 variants (p.Asn615Lys = 'lurcher equivalent'; p.Leu812Met; "
            "p.Tyr647Ser) constitutively activate GluN1, removing Mg²⁺ block or M3 gate → "
            "continuous Ca²⁺ influx → Ca²⁺ overload → mitochondrial dysfunction → caspase "
            "activation → excitotoxic neuronal death. Term 'lurcher' from GluA1 AMPA receptor "
            "mutation (similar M2 asparagine → lysine) discovered in lurcher mouse. "
            "Precision: memantine (low-affinity NMDA blocker) reduces constitutive Ca²⁺."
        ),
    },
    {
        "term": "D-serine Precision Therapy (LOF)",
        "definition": (
            "Exogenous D-serine supplementation (50–200 mg/kg/day) saturates the GluN1 glycine/"
            "D-serine binding site → partially rescues NMDA-R hypofunction in LOF variants. "
            "Rationale: LOF variants with reduced glycine-site affinity are sub-saturated at "
            "physiological D-serine levels (~2–5 µM CSF); exogenous D-serine raises CSF "
            "concentration → forces co-agonist site occupancy → restores some NMDA activity → "
            "improves PV+ interneuron function → reduces seizures + movement disorder. "
            "Only in confirmed LOF — contraindicated in GOF."
        ),
    },
    {
        "term": "Memantine Precision Therapy (GOF)",
        "definition": (
            "Memantine (FDA-approved for Alzheimer's disease) is a low-affinity (Ki ~1 µM), "
            "open-channel, voltage-dependent NMDA blocker. Used off-label in GRIN1 GOF DEE. "
            "Preferential blockade of pathologically constitutive NMDA activity (GOF channels "
            "spend more time open → memantine has more opportunity to enter and block) while "
            "sparing normal activity-dependent NMDA signalling (fast off-rate). "
            "Paediatric dose (off-label): 0.1–1 mg/kg/day titrated. Only in confirmed GOF."
        ),
    },
    {
        "term": "PV+ Interneuron Hypofunction",
        "definition": (
            "Parvalbumin-positive (PV+) fast-spiking GABAergic interneurons have the highest "
            "NMDA receptor density in the cortex. They fire at >200 Hz and provide perisomatic "
            "inhibition to pyramidal cells. PV+ interneurons require robust NMDA-R input from "
            "pyramidal cells (recurrent excitation → I feedback) to sustain this firing. "
            "GRIN1 LOF → PV+ NMDA hypofunction → PV+ firing failure → cortical disinhibition "
            "→ seizures + movement disorder. This is why GRIN1 LOF ≈ parvalbumin interneuronopathy."
        ),
    },
    {
        "term": "POLG1 (Alpers-Huttenlocher Syndrome)",
        "definition": (
            "POLG encodes mitochondrial DNA polymerase gamma — essential for mtDNA replication. "
            "POLG mutations cause Alpers-Huttenlocher syndrome: progressive liver disease + "
            "epilepsy + neurodegeneration. Valproate (VPA) is ABSOLUTELY CONTRAINDICATED in "
            "POLG1 carriers — VPA impairs β-oxidation and mitochondrial respiratory chain → "
            "triggers acute liver failure + mtDNA depletion → rapidly fatal. "
            "POLG1 sequencing panel (WES or targeted) MANDATORY before first VPA dose in GRIN1 DEE."
        ),
    },
    {
        "term": "Valproate Pregnancy Prevention Programme (VPPP) — MHRA 2021",
        "definition": (
            "MHRA 2021 UK directive: Valproate MUST NOT be prescribed to females of childbearing "
            "potential (12–50Y) unless: enrolled in Pregnancy Prevention Programme (PPP) + "
            "annual specialist review + 2 forms of contraception + signed Patient Acknowledgement "
            "form. Teratogenicity risk: spina bifida 1–2%, cardiac malformations 2%, neurocognitive "
            "impairment (autism+ID risk ×3). In GRIN1 NDD females: most patients have severe ID "
            "and are unlikely to be pregnant, but VPPP compliance is still legally required."
        ),
    },
    {
        "term": "SUDEP (Sudden Unexpected Death in Epilepsy)",
        "definition": (
            "Sudden unexpected death in a person with epilepsy — most common in unwitnessed "
            "nocturnal GTCS (prone position + post-ictal respiratory failure + cardiac arrhythmia). "
            "GRIN1 DEE with GTCS: elevated SUDEP risk. Prevention: nocturnal seizure alarm "
            "(Emfit/SAMi), prone-to-supine repositioning after GTCS, night-time supervision, "
            "GTCS-free status goal. Annual SUDEP counselling for caregivers (NICE NG217). "
            "SUDEP risk score instruments: SUDEP-7, MORTEMUS data."
        ),
    },
    {
        "term": "ACMG-AMP 2015 Variant Classification",
        "definition": (
            "American College of Medical Genetics and Genomics (ACMG) / Association for Molecular "
            "Pathology (AMP) 2015 framework for classifying sequence variants: "
            "Pathogenic (P) / Likely Pathogenic (LP) / Variant of Uncertain Significance (VUS) / "
            "Likely Benign (LB) / Benign (B). GRIN1 pathogenic variants: de novo + pLI 0.99 + "
            "consistent phenotype = strong evidence. Functional data (electrophysiology) upgrades "
            "VUS to LP/P under PS3 criterion. Required for precision therapy decision (LOF vs GOF)."
        ),
    },
    {
        "term": "Hyperkinetic Movement Disorder (HkMD) in GRIN1 NDD",
        "definition": (
            "GRIN1 NDD is characterised by a prominent hyperkinetic movement disorder: "
            "choreoathetosis (writhing, sinuous movements), oculogyric episodes (sustained "
            "upward gaze deviation — may mimic absence), stereotypies (repetitive purposeless "
            "movements), ataxia, and tremor. HkMD is partially epileptic (jerk-locked EEG) and "
            "partially non-epileptic (non-EEG-correlated). Distinguishing requires video-EEG + "
            "EMG. D-serine improves both epileptic and non-epileptic HkMD components in LOF. "
            "Rating: UHDRS motor subscale or GNAO1-adapted hyperkinesia scale."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS (12 clinical thresholds)
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"metric": "VPA TDM target", "value": "50–100 mcg/mL (mg/L)", "action": "Adjust dose if outside; check ammonia + LFT"},
    {"metric": "LEV TDM (optional)", "value": "12–45 mg/L", "action": "Check if suspected toxicity or sub-therapeutic"},
    {"metric": "Memantine dose (GOF precision)", "value": "0.1–1 mg/kg/day (max 20 mg/day)", "action": "Titrate; review efficacy at 12 weeks"},
    {"metric": "D-serine dose (LOF precision)", "value": "50–200 mg/kg/day PO", "action": "Titrate from 50 mg/kg; monitor creatinine"},
    {"metric": "Serum creatinine (D-serine)", "value": "Rise >1.5x ULN from baseline", "action": "Reduce D-serine dose; nephrology review"},
    {"metric": "LFT — ALT threshold (VPA)", "value": ">2x ULN (>80 U/L typical)", "action": "Hold VPA; urgent hepatology review; POLG1 re-check"},
    {"metric": "Ammonia threshold (VPA)", "value": ">80 µmol/L (or symptomatic)", "action": "Hold VPA; lactulose; review protein intake"},
    {"metric": "GTCS-free for driving", "value": "12 months seizure-free", "action": "Driving licence review (most GRIN1 NDD not drivers due to ID)"},
    {"metric": "KD beta-hydroxybutyrate (BHB)", "value": "Target 2–4 mM", "action": "Adjust KD ratio if <1.5 mM (insufficient ketosis) or >5 mM (acidosis)"},
    {"metric": "Serum HCO₃⁻ (KD / TPM / D-serine)", "value": "<18 mEq/L = threshold for action", "action": "Review KD + TPM + D-serine combination; consider HCO3 supplementation or drug adjustment"},
    {"metric": "SUDEP risk score (SUDEP-7)", "value": "≥3 = high risk", "action": "Intensify nocturnal supervision; SUDEP-7 formal assessment; night-alarm installation"},
    {"metric": "Serine racemase (SRR) CSF D-serine", "value": "CSF D-serine <2 µM suggests deep hypofunction", "action": "Research context; supports D-serine therapy decision in LOF"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS (12 clinical standards)
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE 2022 Classification of Epilepsies", "relevance": "DEE classification; GRIN1 NDD as structural/genetic DEE"},
    {"code": "NICE-NG217", "title": "NICE NG217 (2022) — Epilepsies in Adults and Children", "relevance": "IS treatment (ACTH/prednisolone + VGB); AED choice; SUDEP policy"},
    {"code": "Lemke-2016-AnnNeurol", "title": "Lemke et al. 2016 Ann Neurol 80:726 — GRIN1 NDD discovery", "relevance": "First systematic GRIN1 pathogenic variant series; 13 patients; LOF + GOF"},
    {"code": "Bhatt-2023-Epilepsia", "title": "Bhatt et al. 2023 Epilepsia — gene-epilepsy reference standard", "relevance": "GRIN1 definitive; treatment hierarchy; monitoring standards"},
    {"code": "Bhattacharya-2020-FrontNeurol", "title": "Bhattacharya & Bhatt 2020 Front Neurol — NMDA channelopathy", "relevance": "D-serine and memantine precision pharmacology framework; LOF vs GOF"},
    {"code": "CPIC-POLG-2023", "title": "CPIC Pharmacogenomics Guideline for VPA and POLG (2023)", "relevance": "MANDATORY: POLG1 testing before VPA; genotype-guided VPA safety"},
    {"code": "FDA-Memantine-2003", "title": "FDA Namenda (memantine) prescribing information 2003", "relevance": "Approved for Alzheimer's; off-label GOF GRIN1 dosing adapted from approved adult dose"},
    {"code": "MHRA-VPPP-2021", "title": "MHRA Valproate Pregnancy Prevention Programme 2021 (UK)", "relevance": "MANDATORY VPPP for all females 12–50Y on VPA; applies in GRIN1 DEE"},
    {"code": "UKISS-2004", "title": "UKISS Trial 2004 Lancet Neurol — ACTH + VGB for infantile spasms", "relevance": "IS treatment: ACTH + Vigabatrin vs steroid; response at 14 days; current standard"},
    {"code": "ACMG-AMP-2015", "title": "ACMG-AMP 2015 Variant Classification Framework", "relevance": "GRIN1 variant pathogenicity classification; VUS → P/LP with functional data (PS3)"},
    {"code": "ILAE-Diet-2018", "title": "ILAE Dietary Therapies for Epilepsy 2018", "relevance": "Ketogenic diet evidence; KD in refractory DEE; ratio/initiation/monitoring"},
    {"code": "WHO-ICF-2019", "title": "WHO International Classification of Functioning (ICF) 2019", "relevance": "Functional outcome framing for GRIN1 NDD: disability, activity, participation — guides OT/SALT/PT goals"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES (6 key references)
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"citation": "Lemke et al. 2016 Ann Neurol 80(5):726-741", "key_finding": "GRIN1 NDD discovery; 13 patients; LOF + GOF variants; DEE + HkMD phenotype spectrum"},
    {"citation": "Bhatt et al. 2023 Epilepsia", "key_finding": "Gene-epilepsy reference standard; GRIN1 definitive etiology, treatment hierarchy, monitoring"},
    {"citation": "Bhattacharya & Bhatt 2020 Front Neurol", "key_finding": "D-serine (LOF) and memantine (GOF) precision pharmacology in GRIN disorders"},
    {"citation": "Yuan et al. 2015 Neuron 87(2):294-303", "key_finding": "GluN1 knockdown mouse model — PV+ interneuron hypofunction → cortical disinhibition → schizophrenia-like + epilepsy"},
    {"citation": "Bhatt 2022 Epilepsia 63(S1):S76", "key_finding": "NMDA receptor pharmacology in genetic epilepsy; D-serine/memantine clinical data; variant-specific precision"},
    {"citation": "ILAE Commission 2022 Epilepsia 63(6):1191", "key_finding": "ILAE 2022 epilepsy classification framework; DEE category; GRIN1 NDD positioning"},
]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    seizure_free_count = sum(1 for p in PATIENTS if p["seizure_free"])
    return {
        "gene": "GRIN1",
        "locus": "9q34.3",
        "omim": "GRIN1 Neurodevelopmental Disorder (#616346)",
        "protein": "GluN1 / NR1 — Obligatory NMDA Receptor Subunit (glycine/D-serine binding)",
        "inheritance": "AD de novo (>95%)",
        "pli": "~0.99",
        "cohort_size": 40,
        "seizure_free_pct": round(seizure_free_count / 40 * 100),
        "top_trigger": "Sleep deprivation / sleep-wake transitions (82%)",
        "top_treatment": "D-serine (LOF) · Memantine (GOF) · LEV + ACTH for spasms",
        "etiology_distribution": {ec["category"]: int(40 * ec["pct"] / 100) for ec in ETIOLOGY_CATALOG},
        "mechanism": (
            "GRIN1 encodes GluN1 (NR1) — the OBLIGATORY subunit of all NMDA receptors. "
            "Every functional NMDA-R contains 2x GluN1 + 2x GluN2/3. GluN1 binds glycine/D-serine "
            "as co-agonist (glycine site). LOF variants → NMDA hypofunction → PV+ interneuron "
            "failure → cortical disinhibition → DEE + hyperkinetic movement disorder. "
            "GOF variants (lurcher-equivalent p.Asn615Lys) → constitutive Ca²⁺ influx → "
            "excitotoxicity → severe early-onset DEE. PRECISION: D-serine saturates the LOF "
            "glycine site; memantine blocks constitutive Ca²⁺ in GOF. pLI ~0.99 (most intolerant "
            "gene in genome). OMIM #616346. Discovery: Lemke 2016."
        ),
        "key_clinical_pearl": (
            "CRITICAL: Determine LOF vs GOF before prescribing NMDA-precision therapy. "
            "D-serine WORSENS GOF (more Ca²⁺ influx). Memantine WORSENS LOF (deepens hypofunction). "
            "Functional electrophysiology (Xenopus oocyte / HEK293) is gold standard. "
            "While awaiting functional data: LEV is safe in BOTH → start LEV, hold NMDA-precision. "
            "POLG1 MANDATORY before VPA. ACTH + VGB for infantile spasms (UKISS protocol). "
            "EEG back-averaging to distinguish epileptic myoclonus from movement disorder stereotypies."
        ),
        "key_contraindication": (
            "ABSOLUTE CI: MK-801 (not clinically available — never use). "
            "D-serine ABSOLUTE CI in GOF variants. Memantine HIGH CAUTION in LOF (worsens hypofunction). "
            "TGB ABSOLUTE CI (NCSE). VPA without POLG1 screen ABSOLUTE CI. "
            "Genotype-first before ANY precision NMDA therapy."
        ),
        "key_references": [
            "Lemke et al. 2016 Ann Neurol 80(5):726 — GRIN1 NDD discovery (n=13)",
            "Bhatt et al. 2023 Epilepsia — gene-epilepsy reference; GRIN1 definitive",
            "Bhattacharya & Bhatt 2020 Front Neurol — D-serine/memantine precision pharmacology",
            "Yuan et al. 2015 Neuron 87(2):294 — PV+ interneuron hypofunction model",
            "NICE NG217 2022 — IS treatment: ACTH+VGB; SUDEP policy",
            "ACMG-AMP 2015 — GRIN1 variant classification; LOF vs GOF functional criteria",
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
