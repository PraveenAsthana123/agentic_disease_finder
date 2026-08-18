"""
PNPO Epilepsy (Pyridoxamine-5'-phosphate Oxidase Deficiency / PLP-Dependent Neonatal EE)
=========================================================================================
40-patient cohort · PNPO (17q21.32) · Pyridoxamine-5'-phosphate oxidase
PNPO deficiency: the SECOND treatable vitamin-B6-dependent epilepsy — autosomal recessive LOF
variants in PNPO abolish conversion of pyridoxamine-5'-phosphate (PMP) and pyridoxine-5'-phosphate
(PNP) to pyridoxal-5'-phosphate (PLP, the active cofactor), causing systemic PLP deficiency,
failure of >50 PLP-dependent enzymes (GAD, AADC, serine racemase, cystathionine β-synthase,
phosphoserine aminotransferase, etc.), severe neonatal epileptic encephalopathy, and multi-organ PLP
insufficiency that responds ONLY to PLP (not pyridoxine — this is the critical diagnostic trap).

PNPO BIOLOGY:
PNPO (Pyridoxamine-5'-phosphate Oxidase, 17q21.32) encodes the enzyme that catalyses the
rate-limiting step of PLP biosynthesis from the dietary B6 vitamers:
  Pyridoxamine-5'-phosphate (PMP) + O₂ + FAD → PLP + NH₃ + FADH₂
  Pyridoxine-5'-phosphate (PNP)   + O₂ + FAD → PLP + H₂O₂ + FADH₂
The enzyme is a FLAVOPROTEIN (FAD-dependent) — a homodimer requiring FMN/FAD as cofactor.
FAD (riboflavin) availability is therefore essential for PNPO function.

In PNPO deficiency:
  PMP and PNP accumulate → elevated plasma/urine PMP, PNP (diagnostic biomarkers).
  PLP production halted → systemic PLP depletion → failure of all PLP-dependent enzymes:
    ① GAD (glutamic acid decarboxylase): glutamate → GABA ↓ → cortical hyperexcitability → seizures
    ② AADC (aromatic L-amino acid decarboxylase): L-DOPA → dopamine ↓; 5-HTP → serotonin ↓ → monoamine deficiency
    ③ Serine racemase: L-serine → D-serine ↓ → NMDA co-agonist deficiency
    ④ Cystathionine β-synthase: CBS → homocysteine accumulation → endothelial toxicity
    ⑤ Phosphoserine aminotransferase (serine synthesis) → affects myelination
    ⑥ Glycogen phosphorylase (muscle/liver) → metabolic dysfunction
  Prematurity: PNPO mRNA is expressed in placenta; in utero PLP supply depends on maternal PLP
    delivery to fetus — neonates born before full placental PLP transfer is established
    (especially premature infants) have acutely low PLP at birth, precipitating early SE.
  In utero seizures: maternal perception of abnormal fetal movements/hiccoughs in 3rd trimester
    in ~35% — intrauterine epileptic activity due to fetal PLP deficiency.

CRITICAL DISTINCTION FROM ALDH7A1-PDE:
  ALDH7A1-PDE: defect DOWNSTREAM in lysine catabolism → P6C accumulates → inactivates exogenous PLP
    BUT: supplemental PYRIDOXINE (B6 vitamer precursor) overrides P6C trapping → therapeutic.
    IV PYRIDOXINE 30 mg/kg → EEG response within 1 hour = DIAGNOSTIC + THERAPEUTIC.
  PNPO: defect in PLP SYNTHESIS ENZYME → pyridoxine/PMP cannot be CONVERTED to PLP
    Therefore: PYRIDOXINE IS INEFFECTIVE — the conversion enzyme is absent.
    Only PLP (already the active form) bypasses the enzyme block.
    CLINICAL TRAP: empiric pyridoxine trial (standard for unexplained neonatal seizures) will be
    NEGATIVE in PNPO deficiency → clinician may wrongly exclude B6-dependent epilepsy.
    MANDATORY PROTOCOL: if pyridoxine trial fails (no EEG response in 1 hour) →
    IMMEDIATELY trial PLP 30 mg/kg (oral/NG — no IV PLP formulation available in most countries).

INHERITANCE: Autosomal recessive — biallelic LOF variants required.
  Founder variants: p.Arg229Trp (UK/Ireland, common), p.Asp33Val (Middle East), p.Pro314Leu
  Prevalence: estimated 1:500,000 — rarer than ALDH7A1-PDE (1:64,000–1:400,000).
  Mills PB et al. 2005 Lancet — first description (UK newborns).
  Clayton PT et al. 2011 JIMD — PMP/PNP biomarker delineation.
"""

import random
from datetime import datetime

SEED = 9250  # dashboard 250
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=40) ───────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": (
            "PNPO biallelic LOF — classic neonatal PLP-dependent EE "
            "(homozygous or compound heterozygous null/severe missense)"
        ),
        "n": 14, "pct": 35,
        "category": "PNPO-biallelic-LOF-classic-neonatal",
        "functional_class": "LOF-classic-neonatal",
        "mechanism": (
            "Most prevalent class (~35%): biallelic loss-of-function PNPO variants (homozygous "
            "truncating nonsense/frameshift or compound heterozygous null alleles). Complete "
            "abolition of pyridoxamine-5'-phosphate oxidase activity → zero PLP biosynthesis from "
            "dietary B6 vitamers → total PLP enzyme failure from birth/in utero. Classic presentation: "
            "in utero seizures (maternal perception 3rd trimester), premature birth (32–36 weeks), "
            "neonatal seizures within hours of birth, burst-suppression EEG, severe acidosis, "
            "elevated plasma PMP/PNP, undetectable plasma PLP. Pyridoxine trial NEGATIVE (PNPO "
            "cannot convert pyridoxine→PLP). PLP 30 mg/kg → EEG response within 1–4 hours. "
            "Common null variants: frameshift (del/ins), nonsense (p.Arg116Ter, p.Glu141Ter), "
            "large deletions. Without neonatal PLP rescue: fatal neonatal SE within days."
        ),
        "typical_onset": "In utero / birth to 1 week",
        "eeg_pattern": "Burst-suppression — high-amplitude burst + deep suppression",
        "key_biomarkers": "Plasma PMP ↑↑↑ + PNP ↑; plasma PLP undetectable; urine PMP/PNP ↑",
        "response_to_pyridoxine": "NONE — pyridoxine cannot be converted to PLP",
        "response_to_plp": "Excellent — EEG improvement within 1-4 hours; seizure cessation",
    },
    {
        "etiology": (
            "PNPO missense hypomorph — residual PNPO activity, "
            "later-onset or attenuated phenotype (FAD-binding/active-site missense)"
        ),
        "n": 11, "pct": 28,
        "category": "PNPO-missense-hypomorph-residual-activity",
        "functional_class": "LOF-hypomorphic-residual-activity",
        "mechanism": (
            "Second most common class (~28%): missense variants with partial residual PNPO activity "
            "(typically 5–25% of wildtype). Includes the common UK founder p.Arg229Trp (7.5% residual "
            "activity, Kanabar 2019), p.Asp33Val (Middle East, 12% residual), p.Thr260Met (Israeli, "
            "15% residual). Residual PNPO can sustain marginal PLP biosynthesis but may decompensate "
            "under metabolic stress (fever, intercurrent illness, rapid growth). Later onset than null "
            "alleles: neonatal (days 1–28) in homozygotes; rarely infantile presentation. Plasma PMP "
            "mildly-to-moderately elevated; PLP low-normal to low; PNP elevated; CSF/plasma pyridoxal "
            "ratio elevated. Some hypomorphs show PARTIAL pyridoxine response (residual PNPO converts "
            "small fraction) — clinically confusing: may appear pyridoxine-responsive at high doses but "
            "require PLP for full seizure control. Riboflavin supplementation (FAD cofactor) may boost "
            "residual PNPO activity in vitro (2–5-fold increase in FAD-binding variants)."
        ),
        "typical_onset": "Neonatal day 1-28 (hypomorph homozygous) / infantile (compound het with null)",
        "eeg_pattern": "Burst-suppression → multifocal spikes → hypsarrhythmia if West syndrome develops",
        "key_biomarkers": "Plasma PMP ↑ (less marked), PLP low-normal, PNP ↑, CSF pyridoxal ↓",
        "response_to_pyridoxine": "Partial (at very high dose) in FAD-binding variants with residual activity",
        "response_to_plp": "Good to excellent; riboflavin adjunct may improve dose efficiency",
    },
    {
        "etiology": (
            "PNPO splicing variant — exon-skipping / intron-retention, "
            "partial LOF with variable tissue expression"
        ),
        "n": 8, "pct": 20,
        "category": "PNPO-splicing-partial-LOF",
        "functional_class": "LOF-splicing-partial",
        "mechanism": (
            "Third class (~20%): deep intronic, splice-site, or synonymous PNPO variants causing "
            "aberrant splicing (exon skip, intron retention, cryptic splice activation). Typically "
            "results in partial loss of PNPO protein with tissue-dependent residual activity "
            "(brain may be more severely affected than liver/blood). Whole-exome sequencing may "
            "MISS these variants — RNA sequencing (blood/fibroblast) required for confirmation. "
            "Key splicing variants: c.674C>T (synonymous p.Arg225=, disrupts exon 7 splice enhancer, "
            "Ruiz 2008 Brain), c.675+1G>A (canonical splice site). Variable phenotype: some present "
            "neonatally with burst-suppression; others at 2–6 weeks with multifocal seizures; "
            "rare cases with later infantile spasms. NGS panels: ensure PNPO covers exon-intron "
            "boundaries at ≥250× depth; request RNA-seq if clinical suspicion but exome negative."
        ),
        "typical_onset": "Neonatal (2 weeks) to infantile (6 weeks)",
        "eeg_pattern": "Burst-suppression → multifocal spikes → focal temporal spikes",
        "key_biomarkers": "PMP ↑ (variable), PNP ↑; CSF/plasma pyridoxal ratio elevated",
        "response_to_pyridoxine": "Variable — depends on residual spliced PNPO activity",
        "response_to_plp": "Good; may be dose-sensitive (lower doses effective in high-residual cases)",
    },
    {
        "etiology": (
            "PNPO + riboflavin (FAD) deficiency — secondary PNPO dysfunction "
            "(nutritional/transporter SLC52A2-SLC52A3 compound)"
        ),
        "n": 4, "pct": 10,
        "category": "PNPO-secondary-FAD-riboflavin-deficiency",
        "functional_class": "Secondary-PNPO-FAD-cofactor",
        "mechanism": (
            "Fourth class (~10%): PNPO function requires FAD as obligate cofactor. Severe riboflavin "
            "(vitamin B2) deficiency — from nutritional deficiency (exclusive breast-feeding without "
            "riboflavin supplementation in PNPO heterozygotes with borderline PNPO activity) OR "
            "riboflavin transporter deficiency (SLC52A2 / SLC52A3 — Brown-Vialetto-Van Laere / BVVL "
            "syndrome) — can precipitate functional PNPO failure despite normal PNPO gene. BVVL "
            "overlap: SLC52A2/SLC52A3 LOF → riboflavin malabsorption → FAD deficiency → PNPO "
            "dysfunction → PLP deficiency → seizures PLUS progressive sensorineural hearing loss, "
            "pontobulbar palsy, respiratory failure. Treatment: high-dose riboflavin (40–100 mg/day) "
            "± PLP (if riboflavin alone insufficient). CSF riboflavin and FAD measured at LP."
        ),
        "typical_onset": "Infantile to early childhood (BVVL overlap); neonatal (nutritional B2 deficiency)",
        "eeg_pattern": "Multifocal spikes → West syndrome in severe cases",
        "key_biomarkers": "CSF riboflavin ↓; plasma flavins ↓; PMP/PNP may be mildly elevated",
        "response_to_pyridoxine": "No",
        "response_to_plp": "Partial; riboflavin PRIMARY treatment if FAD-deficient",
    },
    {
        "etiology": (
            "Phenocopy — non-PNPO PLP-dependent epilepsy "
            "(ALDH7A1 atypical / PROSC / hyperprolinaemia II / antiquitin phenocopy)"
        ),
        "n": 3, "pct": 7,
        "category": "Phenocopy-PLP-dependent-non-PNPO",
        "functional_class": "Phenocopy-PLP-pathway",
        "mechanism": (
            "Fifth class (~7%): PLP-responsive neonatal epilepsy with elevated plasma PMP/PNP but "
            "PNPO sequence-normal. Key phenocopies: PROSC (PLPBP, 8p11.23) — PLP homeostasis protein "
            "deficiency — distinct PNPO-phenocopy with CSF PLP deficiency but normal plasma PMP/PNP; "
            "Hyperprolinaemia type II (ALDH4A1, P5CDH deficiency) — proline metabolite P5C accumulates "
            "and inactivates PLP (similar to P6C in ALDH7A1-PDE); vitamin B6-responsive epilepsy with "
            "intellectual disability (PLPBP mutations); antiquitin (ALDH7A1) atypical presentation "
            "with intermediate AASA elevation. Management: all require PLP supplementation regardless "
            "of exact gene, but PROSC/PLPBP uses lower PLP doses (5–15 mg/kg/day vs 30 mg/kg for PNPO)."
        ),
        "typical_onset": "Neonatal to infantile",
        "eeg_pattern": "Variable — burst-suppression to focal/multifocal",
        "key_biomarkers": "Variable; PROSC: normal PMP/PNP, low CSF PLP; check AASA for ALDH7A1",
        "response_to_pyridoxine": "Variable (ALDH7A1 phenocopy: YES; PROSC: NO)",
        "response_to_plp": "Usually yes — trial PLP in all unexplained neonatal seizures after pyridoxine failure",
    },
]

# ── Patient Registry (N=40) ────────────────────────────────────────────────────
_etiology_pool = []
for e in ETIOLOGY_CATALOG:
    _etiology_pool.extend([e["category"]] * e["n"])
random.shuffle(_etiology_pool)

PATIENTS = []
_MALE_NAMES   = ["Hamza","Yusuf","Omar","Ibrahim","Khalid","Tariq","Samir","Ali","Nasser","Faisal",
                  "Idris","Rashid","Bashir","Liam","Noah","James","Ethan","Declan","Finn","Ronan"]
_FEMALE_NAMES = ["Amira","Fatima","Layla","Nour","Sara","Hana","Aya","Mariam","Zainab","Rania",
                  "Sophia","Emma","Olivia","Aoife","Ciara","Niamh","Isla","Freya","Chloe","Mia"]
_ONSET_AGE    = ["Birth","Birth","Birth","1 day","1 day","2 days","3 days","5 days","7 days",
                  "10 days","14 days","2 weeks","3 weeks","4 weeks","6 weeks","8 weeks","12 weeks",
                  "In utero","In utero","In utero","Premature 34wk","Premature 32wk","Premature 35wk",
                  "Premature 30wk","Neonatal","Neonatal","Neonatal","Infantile 3mo","Infantile 4mo","Infantile 6mo",
                  "Birth","Birth","2 days","14 days","1 day","3 days","Birth","7 days","2 weeks","1 day"]
_PLPBP_RESP   = ["PLP 30 mg/kg → seizure-free","PLP → EEG improvement 2h","PLP + riboflavin",
                  "PLP 20 mg/kg + folinic acid","PLP → spasm remission","PLP partial response",
                  "PLP 30 mg/kg→pyridoxine-failed","Riboflavin primary + PLP adjunct",
                  "PLP responsive (hypomorph)","PLP → developmental improvement"]
_ETIOLOGIES_SHORT = {
    "PNPO-biallelic-LOF-classic-neonatal": "PNPO null LOF",
    "PNPO-missense-hypomorph-residual-activity": "PNPO hypomorph",
    "PNPO-splicing-partial-LOF": "PNPO splicing",
    "PNPO-secondary-FAD-riboflavin-deficiency": "FAD/riboflavin",
    "Phenocopy-PLP-dependent-non-PNPO": "Phenocopy",
}
_VARIANTS = ["p.Arg229Trp","p.Asp33Val","p.Pro314Leu","p.Thr260Met","p.Arg116Ter","p.Glu141Ter",
             "c.674C>T (splicing)","c.675+1G>A","p.Gly193Ser","p.Ala174Pro","del exon 3-4",
             "p.Arg116His","p.Leu229Pro","c.302+5G>A","p.Gly220Arg"]

for i, etiol in enumerate(_etiology_pool):
    sex = "M" if i % 2 == 0 else "F"
    name = (_MALE_NAMES[i % len(_MALE_NAMES)] if sex == "M"
            else _FEMALE_NAMES[i % len(_FEMALE_NAMES)])
    onset = _ONSET_AGE[i % len(_ONSET_AGE)]
    resp  = _PLPBP_RESP[i % len(_PLPBP_RESP)]
    var   = _VARIANTS[i % len(_VARIANTS)]
    age_mo = random.randint(2, 72)
    PATIENTS.append({
        "id": f"PNPO-{i+1:03d}",
        "name": name,
        "sex": sex,
        "age_months": age_mo,
        "etiology": etiol,
        "etiology_short": _ETIOLOGIES_SHORT[etiol],
        "onset": onset,
        "plp_response": resp,
        "pnpo_variant": var,
        "premature": "premature" in onset.lower() or "utero" in onset.lower(),
        "current_treatment": random.choice([
            "PLP 30 mg/kg/day","PLP + riboflavin","PLP + folinic acid","PLP + LEV","PLP + PHB",
            "Riboflavin 20 mg/kg + PLP adjunct","PLP + KD","PLP mono"
        ]),
    })

# ── Seizure Types (5 types) ────────────────────────────────────────────────────
SEIZURE_CATALOG = [
    {
        "type": "Neonatal myoclonic / multifocal clonic (burst-suppression EEG correlate)",
        "pct": 95,
        "category": "neonatal-myoclonic-multifocal-burst-suppression",
        "eeg": "Burst-suppression: high-amplitude polyspike bursts (100–300 µV) alternating with flat "
               "periods (0–5 µV); multifocal spike-and-wave superimposed; COMPLETELY DIFFERENT from "
               "benign neonatal sleep myoclonus (no EEG correlate). Interictal: burst-suppression or "
               "modified BS (shorter suppression periods < 2 sec with residual PNPO activity).",
        "semiology": "Myoclonic jerks of face/limbs — often bilateral asymmetric; multifocal clonic "
                     "movements migrating between body parts; tonic eye deviation; apnoea episodes; "
                     "bradycardia. Often begin in utero (maternal perception: hiccough-like fetal "
                     "movements in 3rd trimester = intrauterine seizure activity). Neonatal SE common.",
        "clinical_tip": "BURST-SUPPRESSION IN A NEONATE = EMERGENCY. Mandatory immediate pyridoxine "
                        "IV 30 mg/kg (with cardiac/EEG monitoring) THEN PLP 30 mg/kg if no response "
                        "within 60 minutes. Do NOT wait for metabolic workup results before trials.",
        "frequency": "Multiple clusters daily → continuous SE if untreated",
        "duration": "Seconds to minutes (clusters); SE if untreated (minutes to hours)",
    },
    {
        "type": "Neonatal tonic (axial tonic posturing — PLP-dependent GAD failure)",
        "pct": 80,
        "category": "neonatal-tonic-axial-GAD-failure",
        "eeg": "EEG correlate: low-amplitude high-frequency tonic discharge (desynchronisation) or "
               "rhythmic delta/theta; post-ictal suppression. Tonic neonatal seizures carry highest "
               "mortality risk — associated with autonomic instability, apnoea, bradycardia.",
        "semiology": "Sustained tonic extension or flexion of trunk ± limbs; opisthotonus; tonic "
                     "eye deviation; facial flushing/pallor; apnoea. Duration 10–60 seconds. "
                     "Often coexist with myoclonic jerks in PNPO — mixed neonatal SE pattern.",
        "clinical_tip": "Tonic neonatal SE in PNPO requires PLP urgently. Phenobarbital alone "
                        "is inadequate — GABA synthesis is intact (GAD substrate) but GAD enzyme "
                        "activity requires PLP; PHB cannot restore GABAergic inhibition without PLP.",
        "frequency": "Clusters of 5-20 events/day; continuous tonic SE if untreated",
        "duration": "10-60 seconds per event",
    },
    {
        "type": "Epileptic spasms / West syndrome (infantile evolution if untreated PNPO)",
        "pct": 55,
        "category": "epileptic-spasms-west-infantile-evolution",
        "eeg": "Hypsarrhythmia (classical or modified): chaotic high-amplitude (>200 µV) slow waves "
               "with superimposed multifocal spikes. Electrodecrement at spasm onset. Modified "
               "hypsarrhythmia (asymmetric, synchronous burst-suppression evolving) more common in "
               "PNPO than pure structural West. EEG normalisation with PLP treatment is DIAGNOSTIC.",
        "semiology": "Flexion/extension/mixed spasms; clusters on awakening; developmental regression "
                     "or arrest. Evolves from neonatal burst-suppression if PLP not administered by "
                     "age 1–3 months. Late-presenting PNPO hypomorphs may present primarily as West "
                     "syndrome at 3–6 months without preceding neonatal seizures.",
        "clinical_tip": "West syndrome in PNPO is TREATABLE with PLP — do NOT proceed to ACTH/VGB "
                        "without first trialling PLP 30 mg/kg in any unexplained infantile spasms "
                        "(metabolic screening: PMP/PNP, AASA, urine organic acids MANDATORY).",
        "frequency": "Clusters 5-20 spasms on awakening; 3-10 clusters/day",
        "duration": "Each spasm 1-5 seconds; cluster 1-10 minutes",
    },
    {
        "type": "Focal seizures (temporal/frontal cortical — later evolution on PLP maintenance)",
        "pct": 40,
        "category": "focal-seizures-temporal-frontal-PLP-maintenance",
        "eeg": "Focal onset rhythmic theta/alpha (temporal > frontal); focal spike-and-wave. Secondary "
               "generalisation in 60% of focal events. Fewer focal seizures on adequate PLP dosing. "
               "Breakthrough focal seizures = PLP LEVEL TOO LOW (pharmacokinetic variability).",
        "semiology": "Automatisms (oral), behavioural arrest, subtle motor (unilateral clonic), "
                     "eye deviation. In older PNPO survivors on PLP maintenance: 30–40% have residual "
                     "focal epilepsy even with adequate PLP — may need adjunct LEV or CLB.",
        "clinical_tip": "Breakthrough focal seizures on PLP maintenance → CHECK PLASMA PLP LEVEL "
                        "(target 30–80 nmol/L plasma PLP). PLP has significant pharmacokinetic "
                        "variability in infants: absorption variable, phosphatase activity high "
                        "(intestinal alkaline phosphatase converts PLP → pyridoxal → absorbed then "
                        "rephosphorylated). Some patients need 4× daily dosing for stable PLP levels.",
        "frequency": "1-10/day if PLP subtherapeutic; rare/absent on optimal PLP dosing",
        "duration": "30 seconds to 3 minutes",
    },
    {
        "type": "In utero seizures (fetal epileptic activity — maternal perception 3rd trimester)",
        "pct": 35,
        "category": "in-utero-fetal-epileptic-activity",
        "eeg": "Cannot be directly recorded — fetal MEG (magnetoencephalography) research tool only. "
               "Retrospectively inferred from: abnormal CTG, maternal perception of rhythmic fetal "
               "hiccough-like movements, fetal bradycardia. Postnatal EEG confirms burst-suppression.",
        "semiology": "Maternal reports of abnormal fetal movements in 3rd trimester: rhythmic "
                     "hiccough-like jerks (≠ normal hiccoughs — too frequent/prolonged), sustained "
                     "tremor, decreased variability. Premature labour (32–36 weeks) may be precipitated "
                     "by fetal SE. PATHOGNOMONIC for PNPO/ALDH7A1 among treatable neonatal epilepsies.",
        "clinical_tip": "Antenatal diagnosis: if family history of PNPO deficiency (prior affected "
                        "sibling), offer prenatal genetic testing. Maternal PLP supplementation "
                        "during pregnancy (investigational — not routine): increases transplacental "
                        "PLP delivery, may reduce severity of in utero seizures in known PNPO families "
                        "(Baumgartner 2007). After birth: PNPO workup + immediate PLP trial.",
        "frequency": "Episodic clusters in utero, especially 3rd trimester",
        "duration": "Minutes (fetal SE)",
    },
]

# ── Triggers (8) ──────────────────────────────────────────────────────────────
TRIGGER_CATALOG = [
    {
        "trigger": "PLP dose omission / delayed administration",
        "pct": 90,
        "mechanism": "PLP half-life in CSF ~4-6 hours; plasma PLP ~6-12 hours in infants. "
                     "Omitting a single PLP dose in the first year of life causes rapid CSF PLP "
                     "depletion → GAD failure → seizure recurrence within 4-12 hours. "
                     "Night-time omission most dangerous (missed early-morning dose). "
                     "MANDATE: PLP 3-4× daily divided dosing; family education on missed-dose emergency.",
        "clinical_note": "Emergency dose: PLP 10 mg/kg immediate oral/NG if seizures recur → "
                         "medical review same day; never double-dose silently.",
    },
    {
        "trigger": "Intercurrent febrile illness",
        "pct": 82,
        "mechanism": "Fever increases PLP-dependent enzyme consumption (GAD, AADC) and accelerates "
                     "PLP catabolism (pyridoxal oxidation by flavin-dependent enzymes). Febrile "
                     "illness in PNPO = metabolic crisis: PLP requirements increase 50-100% → "
                     "standard maintenance dose becomes subtherapeutic. Riboflavin (FAD) cofactor "
                     "demand also increases with infection (oxidative stress).",
        "clinical_note": "Fever plan: PLP dose increase by 50% during illness; rescue dose "
                         "10 mg/kg PLP at fever onset; emergency room awareness of PNPO diagnosis.",
    },
    {
        "trigger": "Riboflavin (FAD) deficiency — dietary or illness-related",
        "pct": 45,
        "mechanism": "PNPO is a FAD-dependent flavoprotein. In hypomorphic PNPO variants, "
                     "residual PNPO activity depends critically on adequate FAD availability "
                     "(FAD stabilises PNPO homodimer). Riboflavin deficiency (dietary, GI illness "
                     "reducing absorption, exclusive breast-feeding without riboflavin supplements) "
                     "→ FAD depletion → residual PNPO activity collapses → seizure breakthrough. "
                     "Particularly relevant in p.Arg229Trp (UK founder) with ~7.5% residual activity.",
        "clinical_note": "Co-supplement riboflavin 5-10 mg/day in all PNPO patients (especially "
                         "hypomorphs); monitor plasma/CSF riboflavin; riboflavin separately from PLP.",
    },
    {
        "trigger": "Rapid growth / metabolic spurts (first year of life)",
        "pct": 65,
        "mechanism": "In infancy, metabolic rate and PLP-dependent enzyme activity scale with growth "
                     "velocity. Periods of rapid weight gain require proportionally higher PLP "
                     "doses (weight-based dosing must be updated at EVERY clinic visit). "
                     "Fixed-dose PLP quickly becomes subtherapeutic as weight increases. "
                     "PLP requirement typically peaks at 6-12 months then stabilises.",
        "clinical_note": "MANDATORY weight-based PLP dose recalculation at EVERY visit (monthly "
                         "in year 1, bimonthly year 2, then 3-monthly). Target: 30 mg/kg/day ÷ 4 doses.",
    },
    {
        "trigger": "Drugs increasing B6 catabolism (INH, cycloserine, penicillamine, hydrazines)",
        "pct": 30,
        "mechanism": "Isoniazid (INH), cycloserine (anti-TB), D-penicillamine (Wilson's/rheumatoid), "
                     "and other hydrazine compounds form covalent adducts with PLP (hydrazone "
                     "condensation with the aldehydic carbon of PLP) → inactivate PLP → compound "
                     "PLP deficiency in PNPO. TB exposure (INH) in endemic settings can precipitate "
                     "PNPO crisis even on PLP maintenance. Neonatal exposure via breast milk if "
                     "mother on INH.",
        "clinical_note": "Absolute CI: INH in PNPO (use rifampicin-based non-INH regimen for TB). "
                         "If unavoidable: PLP dose double during INH therapy + daily LFT monitoring.",
    },
    {
        "trigger": "Metabolic acidosis / hypoglycaemia (neonatal period)",
        "pct": 55,
        "mechanism": "PNPO deficiency causes neonatal lactic acidosis (PLP-dependent lactate "
                     "dehydrogenase pathway disruption) and hypoglycaemia (glycogen phosphorylase "
                     "PLP-dependence). Metabolic decompensation accelerates PLP consumption and "
                     "independently lowers seizure threshold. Neonatal PNPO presentation often "
                     "mimics metabolic emergency (organic acidemia, urea cycle disorder).",
        "clinical_note": "PNPO must be in differential for all neonatal lactic acidosis + seizures. "
                         "Do NOT wait for metabolic results: pyridoxine → PLP trial must be "
                         "concurrent with metabolic workup.",
    },
    {
        "trigger": "Sleep deprivation and circadian disruption",
        "pct": 35,
        "mechanism": "GAD activity (PLP-dependent GABA synthesis) follows circadian rhythm with "
                     "lowest GABA synthesis in early morning (paradoxically — GABA-dependent sleep "
                     "drive is protective during sleep). Sleep deprivation increases glutamate/GABA "
                     "imbalance, increasing PLP demand. In PNPO, this low-GABA nadir is uncompensated. "
                     "Seizure risk highest in early morning on awakening (also West syndrome timing).",
        "clinical_note": "Structured sleep schedule; dose timing: largest PLP dose at bedtime or "
                         "early morning to cover the low-GABA nadir.",
    },
    {
        "trigger": "Catamenial (perimenstrual) in PNPO adolescents / adults",
        "pct": 15,
        "mechanism": "In surviving adolescent PNPO patients on long-term PLP maintenance: "
                     "progesterone withdrawal perimenstrually decreases allopregnanolone (neurosteroid "
                     "GABA-A PAM) → uncovers residual GABA synthesis insufficiency (GAD still "
                     "suboptimal even on PLP, particularly in hypomorphs). Monthly seizure "
                     "clusters in 15% of female PNPO survivors. Management: perimenstrual PLP "
                     "dose increase + CLB 10 mg/day days 23–28 (cycle day-based).",
        "clinical_note": "Perimenstrual seizure diary in all female PNPO adolescents; "
                         "ganaxolone investigational (GABA-A neurosteroid replacement).",
    },
]

# ── Treatments (8) ────────────────────────────────────────────────────────────
TREATMENT_CATALOG = [
    {
        "treatment": "Pyridoxal-5'-phosphate (PLP) — FIRST-LINE precision therapy",
        "level": "Level-B",
        "role": "First-line — definitive treatment (only active B6 form; bypasses absent PNPO enzyme)",
        "dose": "30 mg/kg/day oral/NG in 3-4 divided doses (neonates/infants); "
                "adults: 30-60 mg TDS (90-180 mg/day). Diagnostic trial: 30 mg/kg single oral "
                "dose → assess EEG at 1-4 hours (longer than pyridoxine trial — oral absorption slower).",
        "moa": "PLP is the already-active cofactor — bypasses PNPO enzyme entirely. Directly "
               "restores GAD (GABA synthesis), AADC (dopamine/serotonin), and all other "
               "PLP-dependent enzymes without requiring PNPO conversion.",
        "efficacy": "75-90% seizure freedom in classic PNPO; 50-70% in hypomorphs (may need "
                    "adjunct LEV/CLB for breakthrough focal seizures on maintenance).",
        "monitoring": "Plasma PLP level (target 30-80 nmol/L); LFT (ALT/AST — PLP can cause "
                      "transaminase elevation >3× ULN → reduce dose); plasma pyridoxal; "
                      "urine PMP/PNP (normalisation = adequacy); EEG response.",
        "pnpo_specific_notes": "PLP (NOT pyridoxine): PNPO enzyme is absent, so pyridoxine cannot "
                                "be converted to PLP — pyridoxine is INEFFECTIVE in true PNPO LOF. "
                                "PLP formulation: in most countries, only oral PMP/PLP capsules/powder "
                                "available — NO parenteral PLP formulation (unlike IV pyridoxine). "
                                "Pharmacy: PLP compounded from pharmaceutical-grade PLP powder "
                                "(Sigma-Aldrich or equivalent) in water/OJ. Intestinal alkaline "
                                "phosphatase partially converts PLP → pyridoxal in gut → absorbed → "
                                "rephosphorylated in tissues; absorption efficiency variable (30-60%).",
    },
    {
        "treatment": "Riboflavin (vitamin B2, FAD precursor) — adjunct/rescue in hypomorphs",
        "level": "Level-C",
        "role": "Adjunct — FAD cofactor supplementation to boost residual PNPO homodimer stability",
        "dose": "5-10 mg/day children; 20-40 mg/day adults (high-dose riboflavin). "
                "In SLC52A2/SLC52A3 BVVL overlap: 40-100 mg/day riboflavin PRIMARY treatment.",
        "moa": "FAD is the obligate cofactor of PNPO homodimer. Supranormal FAD availability "
               "can stabilise hypomorphic PNPO protein and boost residual enzymatic activity "
               "2-5-fold in vitro (p.Arg229Trp: 7.5% baseline → 15-20% with excess FAD). "
               "Clinically demonstrated in several p.Arg229Trp hypomorphs.",
        "efficacy": "25-40% improvement in breakthrough seizure frequency in PNPO hypomorphs "
                    "(FAD-binding variants); primary therapy in BVVL overlap (SLC52A2/SLC52A3).",
        "monitoring": "Plasma riboflavin / FAD levels; clinical response; urine (yellow-orange "
                      "fluorescence = adequate riboflavin — reassuring but not quantitative).",
        "pnpo_specific_notes": "Riboflavin-PNPO interaction: most relevant in p.Arg229Trp (UK founder) "
                                "and other FAD-binding domain missense variants. For null alleles: "
                                "no PNPO protein to stabilise → riboflavin ineffective. "
                                "FAD supplementation does NOT replace PLP supplementation — adjunct only.",
    },
    {
        "treatment": "Pyridoxine (vitamin B6) — DIAGNOSTIC TRIAL ONLY; ineffective in PNPO LOF",
        "level": "Level-B (diagnostic protocol — mandatory for all unexplained neonatal seizures)",
        "role": "Diagnostic — to EXCLUDE ALDH7A1-PDE (pyridoxine-responsive). NOT therapeutic in PNPO LOF.",
        "dose": "IV pyridoxine 30 mg/kg over 10 minutes (with EEG + cardiac monitoring). "
                "THEN: assess EEG response at 30-60 minutes. If NO response → immediately "
                "trial PLP 30 mg/kg oral (do NOT wait days).",
        "moa": "Pyridoxine → phosphorylated to PNP by pyridoxal kinase → PNPO converts PNP → PLP. "
               "In PNPO deficiency: PNP cannot be converted to PLP → pyridoxine is INEFFECTIVE. "
               "In ALDH7A1-PDE: pyridoxine successfully overrides P6C trapping (different mechanism).",
        "efficacy": "0% in PNPO complete LOF (null alleles). 5-30% partial response in hypomorphs "
                    "(residual PNPO converts small fraction of PNP → PLP at very high doses). "
                    "In hypomorphs: pyridoxine high-dose may partially control seizures but PLP "
                    "required for full control.",
        "monitoring": "EEG during trial (continuous): look for burst-suppression → normal background "
                      "transition (pyridoxine-responsive = ALDH7A1). Cardiac monitoring: apnoea "
                      "risk with IV pyridoxine. Resuscitation equipment mandatory.",
        "pnpo_specific_notes": "CRITICAL PROTOCOL: pyridoxine trial NEGATIVE (no EEG response in 60 min) "
                                "→ DO NOT STOP B6 WORKUP. Immediately trial oral/NG PLP 30 mg/kg. "
                                "Common error: pyridoxine fails → clinician concludes 'not B6-dependent' "
                                "and stops B6 investigation → PNPO missed → preventable death/disability. "
                                "Blood/urine for PMP/PNP, AASA MUST be collected BEFORE pyridoxine trial.",
    },
    {
        "treatment": "Phenobarbital (PHB) — neonatal SE adjunct (not primary)",
        "level": "Level-B (standard neonatal SE protocol)",
        "role": "Adjunct — neonatal SE management pending PLP response",
        "dose": "20 mg/kg IV loading dose; 3-5 mg/kg/day maintenance (neonatal). "
                "DISCONTINUE once PLP established and seizure-free for 3-6 months.",
        "moa": "GABA-A positive allosteric modulator (barbiturate site). Supplements insufficient "
               "GABAergic inhibition in acute PNPO until PLP restores endogenous GABA synthesis.",
        "efficacy": "Partial control of neonatal SE; inadequate alone (GABA synthesis requires PLP "
                    "to be restored — PHB merely amplifies residual GABA, cannot fully compensate "
                    "for near-absent GABA synthesis in acute PNPO).",
        "monitoring": "PHB level 20-40 mg/L neonatal; respiratory drive; sedation; LFT.",
        "pnpo_specific_notes": "PHB DOES NOT TREAT PNPO — it is a bridging agent only. "
                                "Do not delay PLP trial while waiting for PHB to 'work'.",
    },
    {
        "treatment": "Levetiracetam (LEV) — adjunct for residual focal seizures on PLP maintenance",
        "level": "Level-B (adjunct)",
        "role": "Adjunct — for breakthrough focal seizures on adequate PLP in older PNPO patients",
        "dose": "40-60 mg/kg/day IV loading (neonatal SE); 20-40 mg/kg/day oral maintenance. "
                "BID dosing. Titrate to seizure control.",
        "moa": "SV2A (synaptic vesicle protein 2A) modulator — reduces glutamate vesicle release. "
               "Does NOT depend on PLP. Safe in all B6-pathway defects.",
        "efficacy": "50-60% reduction in residual focal seizures in PNPO on PLP. Does not affect "
                    "the underlying metabolic defect — adjunct only.",
        "monitoring": "Behavioural side effects (agitation, irritability, insomnia) — especially "
                      "in infants. Plasma LEV level if available (target 12-46 mg/L).",
        "pnpo_specific_notes": "PLP-independent mechanism — safe in PNPO. "
                                "Preferred adjunct over VPA (VPA is ABSOLUTE CI in PNPO — see CIs).",
    },
    {
        "treatment": "Folinic acid (leucovorin) — secondary cerebral folate deficiency adjunct",
        "level": "Level-C",
        "role": "Adjunct — for secondary CSF folate deficiency in PNPO (less prominent than ALDH7A1)",
        "dose": "3-5 mg/kg/day oral leucovorin (folinic acid). Separate from PLP by 2 hours.",
        "moa": "PLP-dependent enzymes include 5,10-methylene-THF reductase (MTHFR) and serine "
               "hydroxymethyltransferase. PLP deficiency → secondary CSF folate depletion "
               "(SAME mechanism as ALDH7A1-PDE triple therapy). CSF 5-MTHF measured at LP. "
               "Folinic acid bypasses the MTHFR reduction step → restores CSF folate.",
        "efficacy": "Developmental benefit in PNPO if CSF folate is low (<40 nmol/L in CSF); "
                    "less evidence than in ALDH7A1 (where folinic acid is routine triple therapy).",
        "monitoring": "CSF 5-methyltetrahydrofolate (5-MTHF) at diagnosis and repeat LP at 6 months.",
        "pnpo_specific_notes": "Less universally required than in ALDH7A1-PDE but check CSF folate "
                                "in all PNPO patients at diagnosis LP — add folinic acid if CSF 5-MTHF <40 nmol/L.",
    },
    {
        "treatment": "Ketogenic diet (KD) — adjunct in drug-resistant PNPO (DRE despite PLP)",
        "level": "Level-C",
        "role": "Adjunct — for residual seizures on adequate PLP + LEV ± CLB",
        "dose": "4:1 fat:carbohydrate+protein ratio; classical KD. Introduce at ≥6 months. "
                "KD ketone target: blood beta-hydroxybutyrate 3-5 mmol/L.",
        "moa": "Ketone bodies (BHB, AcAc) are metabolised to acetyl-CoA independently of "
               "PLP-dependent pathways. BHB raises GABA levels (adenosine A1 receptor pathway) "
               "and reduces glutamate. Does not require PLP. Complementary mechanism.",
        "efficacy": "30-50% additional seizure reduction in PNPO DRE. Case series evidence.",
        "monitoring": "KD: urine ketones, BHB, lipid panel, LFT, LFT, renal stones, growth, "
                      "carnitine. Continue PLP + riboflavin at full dose on KD.",
        "pnpo_specific_notes": "KD + PLP synergistic in PNPO: KD increases GABA via non-PLP pathway; "
                                "PLP restores GAD. Combined effect > either alone.",
    },
    {
        "treatment": "Clobazam (CLB) — perimenstrual and breakthrough adjunct",
        "level": "Level-B (adjunct)",
        "role": "Adjunct — perimenstrual catamenial pattern; acute cluster rescue",
        "dose": "0.1-0.3 mg/kg/day children; 10-20 mg/day adults. Perimenstrual: 10 mg/day "
                "days 23-28 of menstrual cycle. Acute seizure cluster rescue: 0.1-0.2 mg/kg PR/buccal.",
        "moa": "BDZ — GABA-A positive allosteric modulator (BDZ site). Supplements residual "
               "GABAergic inhibition. Preferred BDZ in chronic epilepsy (once-daily dosing, "
               "less tolerance than diazepam). In perimenstrual PNPO: compensates for "
               "progesterone-withdrawal GABA reduction.",
        "efficacy": "60-70% reduction in perimenstrual seizure clusters; useful rescue for "
                    "breakthrough seizures pending PLP dose optimisation.",
        "monitoring": "Sedation; CLB level (target 0.03-0.3 mg/L); active metabolite N-CLB "
                      "(0.3-3.0 mg/L); CYP2C19 genotype (poor metabolisers: N-CLB accumulates).",
        "pnpo_specific_notes": "CLB is PLP-INDEPENDENT — safe in PNPO. "
                                "Tolerance develops with continuous use — use perimenstrually/intermittently.",
    },
]

# ── Contraindications (6) ─────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Valproate / sodium valproate (VPA)",
        "severity": "ABSOLUTE CI",
        "reason": (
            "VPA inhibits multiple PLP-dependent enzyme pathways: (1) VPA inhibits GAD (glutamic acid "
            "decarboxylase, PLP-dependent) → GABA synthesis further reduced in already PLP-deficient "
            "brain. (2) VPA inhibits pyridoxal kinase (converts pyridoxal → PLP) → compounds PLP "
            "depletion. (3) VPA causes secondary carnitine deficiency and mitochondrial dysfunction, "
            "which are further compounded by PNPO metabolic instability. (4) VPA is hepatotoxic "
            "and PNPO + VPA have additive hepatic risk (PLP-dependent transaminases already impaired). "
            "ABSOLUTE CI: VPA will WORSEN seizure control and metabolic status in PNPO. "
            "Document on emergency seizure plan: VPA CONTRAINDICATED IN PNPO."
        ),
        "alternative": "LEV (SV2A — PLP independent); CLB (BDZ); KD (metabolic); PHB (acute neonatal)",
    },
    {
        "drug": "Isoniazid (INH) and hydrazine compounds",
        "severity": "ABSOLUTE CI",
        "reason": (
            "Isoniazid and other hydrazines form stable hydrazone adducts with PLP (condensation at "
            "aldehydic C4' position of PLP) → permanently inactivate PLP → compound PLP deficiency "
            "in PNPO → precipitate PNPO crisis even when previously controlled. INH-induced "
            "pyridoxine deficiency seizures are well-known in non-PNPO patients; in PNPO, the effect "
            "is catastrophic. TB management in PNPO: rifampicin-based non-INH regimen. "
            "Other hydrazines: cycloserine, hydralazine, D-penicillamine, procarbazine — all CI. "
            "If unavoidable (MDR-TB): MANDATORY PLP dose doubling + INH-free regime first."
        ),
        "alternative": "Rifampicin-based non-INH TB therapy; consult TB specialist for PNPO-safe regimen",
    },
    {
        "drug": "Pyridoxine alone (without PLP) as long-term treatment in confirmed PNPO LOF",
        "severity": "HIGH RISK (ineffective — leads to untreated PNPO)",
        "reason": (
            "Pyridoxine (B6 vitamer) requires PNPO enzyme to be converted to PLP — this enzyme is "
            "absent in PNPO LOF. Long-term pyridoxine WITHOUT PLP leaves PNPO untreated: GABA "
            "synthesis remains inadequate, seizures persist, developmental injury continues. "
            "This error occurs when clinicians mistake partial hypomorph response to high-dose "
            "pyridoxine as 'pyridoxine-responsive epilepsy' and stop PLP investigation. "
            "Exception: hypomorph variants with >10% residual PNPO may achieve partial control "
            "with very high pyridoxine doses, but PLP supplementation is ALWAYS superior and safer."
        ),
        "alternative": "PLP 30 mg/kg/day (definitive) ± riboflavin adjunct",
    },
    {
        "drug": "Tiagabine (TGB)",
        "severity": "ABSOLUTE CI",
        "reason": (
            "Tiagabine blocks GABA reuptake (GAT-1 transporter) → GABA accumulates in synapse. "
            "In PNPO: GABAergic SYNTHESIS is impaired (insufficient PLP → GAD substrate backup). "
            "TGB can precipitate NCSE by dysregulating GABA dynamics in an already GABA-synthetic "
            "deficient network: excessive presynaptic GABA reuptake blockade paradoxically causes "
            "NCSE in patients with baseline GABA synthesis defects (same mechanism as GABRB3, "
            "GRIN2A, and other GABAergic pathway epilepsies). TGB absolute CI across all "
            "PLP-dependent epilepsies (ALDH7A1, PNPO, PLPBP/PROSC)."
        ),
        "alternative": "LEV, CLB, KD — all PLP-independent mechanism; avoid GABA-reuptake blockers",
    },
    {
        "drug": "High-dose pyridoxine long-term (>500 mg/day chronic)",
        "severity": "HIGH RISK (sensory neuropathy)",
        "reason": (
            "Pyridoxine >500 mg/day chronically → sensory peripheral neuropathy (pyridoxine "
            "toxicity — PLP-independent mechanism at supra-physiological plasma pyridoxine; "
            "dorsal root ganglion toxicity). In PNPO patients managed on BOTH pyridoxine (ineffective "
            "primary) AND PLP: avoid cumulative pyridoxine >500 mg/day. In hypomorphs receiving "
            "pyridoxine as partial adjunct: monitor peripheral nerve function (NCS annually). "
            "PLP itself does NOT cause sensory neuropathy at standard doses (30 mg/kg)."
        ),
        "alternative": "PLP (no neuropathy risk at therapeutic doses); monitor NCS if pyridoxine used as adjunct",
    },
    {
        "drug": "Carbamazepine / oxcarbazepine / lamotrigine in generalised PNPO",
        "severity": "HIGH RISK (may worsen myoclonic / neonatal seizures)",
        "reason": (
            "Sodium channel blockers (CBZ, OXC, LTG) can paradoxically worsen myoclonic and "
            "neonatal seizure types seen in PNPO — same mechanism as in Dravet (SCN1A) and "
            "GABRB3-DEE (preferential interneuron sodium channel blockade → net disinhibition). "
            "PNPO neonatal myoclonic/tonic and infantile spasms are particularly vulnerable. "
            "If CBZ/OXC absolutely required for focal epilepsy component in older PNPO survivors: "
            "use with caution, EEG monitoring, start at lowest dose (myoclonic worsening = stop)."
        ),
        "alternative": "LEV (preferred focal adjunct — no sodium channel component; SV2A mechanism); "
                       "CLB (BDZ adjunct); KD",
    },
]

# ── Monitoring (14 items) ─────────────────────────────────────────────────────
MONITORING_ITEMS = [
    "Plasma PLP level (target 30-80 nmol/L; adjust PLP dose to maintain)",
    "Plasma pyridoxal (active free form; reflects tissue PLP availability)",
    "Urine PMP and PNP (normalisation confirms adequate PLP replacement)",
    "CSF pyridoxal and PLP at diagnosis LP (baseline; re-check 6 months if abnormal)",
    "CSF 5-methyltetrahydrofolate (5-MTHF) — secondary folate deficiency screen",
    "Plasma/CSF riboflavin and FAD levels (FAD cofactor adequacy for residual PNPO)",
    "Urine AASA/P6C (to exclude ALDH7A1-PDE co-diagnosis or phenocopy)",
    "LFT (ALT/AST — PLP transaminase elevation; VPA toxicity if ever used historically)",
    "EEG (burst-suppression resolution = PLP adequacy; monthly in year 1, then 6-monthly)",
    "Video-EEG LTM (characterise breakthrough focal seizures; exclude NCSE on maintenance)",
    "MRI brain 3T (periventricular white matter signal, cortical maturation, structural malformation)",
    "Developmental assessment (Bayley Scales / Vineland Adaptive Behavior Scales — 6-monthly year 1-3)",
    "Weight-based PLP dose recalculation (monthly year 1 → quarterly year 2+ as weight stabilises)",
    "Peripheral neurophysiology / NCS (if pyridoxine used as adjunct; annual from age 5 years)",
]

# ── Lifecycle Stages (6) ──────────────────────────────────────────────────────
LIFECYCLE_STAGES = [
    {
        "stage": "Fetal / In Utero",
        "age_range": "3rd trimester gestation",
        "description": (
            "In utero epileptic activity (maternal perception: rhythmic fetal jerks/hiccoughs). "
            "Premature labour 32-36 weeks precipitated by fetal SE. Antenatal diagnosis if prior "
            "affected sibling (PGD / CVS / amniocentesis for PNPO variants). Maternal PLP "
            "supplementation investigational. Obstetric alert: neonatal team at delivery."
        ),
    },
    {
        "stage": "Neonatal Emergency (Day 0 – 28)",
        "age_range": "Birth to 28 days",
        "description": (
            "CRITICAL WINDOW: pyridoxine IV 30 mg/kg trial → if EEG negative at 60 min → "
            "immediate PLP 30 mg/kg oral/NG. Collect plasma PMP/PNP/AASA + urine before trials. "
            "NICU stabilisation: PHB for acute SE; ventilatory support. "
            "Metabolic crisis: lactic acidosis, hypoglycaemia management concurrent. "
            "PNPO gene panel result awaited. Start PLP immediately on suspicion (do not wait "
            "for genetics). Family counselling: recurrence risk 25% (AR); sibling testing."
        ),
    },
    {
        "stage": "Infantile (1-12 months)",
        "age_range": "1 to 12 months",
        "description": (
            "PLP dose escalation tracking weight (monthly recalculation, target 30 mg/kg/day ÷ 4 doses). "
            "Riboflavin 5-10 mg/day add-on (FAD cofactor). Folinic acid if CSF 5-MTHF low. "
            "EEG monitoring: burst-suppression → normal development of background = good prognostic sign. "
            "Developmental surveillance: physiotherapy, speech therapy, occupational therapy. "
            "Watch for West syndrome evolution (10-20% if PLP started late > 4 weeks). "
            "Vaccinations: standard schedule; fever plan communicated to family."
        ),
    },
    {
        "stage": "Early Childhood (1-5 years)",
        "age_range": "1 to 5 years",
        "description": (
            "Maintenance PLP — dose-adjust quarterly. Breakthrough focal seizures in 30-40%: "
            "add LEV (SV2A, PLP-independent). EEG 6-monthly; MRI at 2 years. "
            "Developmental trajectory: IQ depends on time-to-PLP-rescue (early rescue = near-normal; "
            "delayed >4 weeks = moderate-severe GDD in majority). "
            "School readiness assessment; EHC/IEP planning; specialist SEND support. "
            "Avoid VPA (absolute CI), INH (absolute CI). Monitor PLP levels quarterly."
        ),
    },
    {
        "stage": "School Age / Adolescence (6-18 years)",
        "age_range": "6 to 18 years",
        "description": (
            "Stabilisation: PLP dose transitions from weight-based to adult fixed dosing "
            "(typically 30-60 mg TDS by adolescence). Peripheral NCS if pyridoxine ever co-prescribed. "
            "Female: catamenial monitoring; perimenstrual PLP increase + CLB days 23-28. "
            "Transition to adult neurology at 16-18 years; PNPO metabolic team co-management. "
            "Driving: DVLA/provincial authority notification; seizure-free interval requirement. "
            "Psychosocial: independence, college planning, employment support."
        ),
    },
    {
        "stage": "Adulthood (18+ years)",
        "age_range": "18+ years",
        "description": (
            "Long-term PLP maintenance (30-60 mg TDS/QDS — adult dose). Annual review: "
            "plasma PLP, LFT, neuropsychology, EEG. Pregnancy: PLP safe (category B); "
            "antenatal genetic counselling — partner PNPO carrier testing; fetal monitoring "
            "3rd trimester. INH absolute CI remains lifelong. "
            "PNPO life expectancy: normal if PLP adequate + seizure-free; "
            "DRE with frequent SE: significant morbidity. Rare SUDEP risk. "
            "Patient/family registry: submit to PNPO International Registry (Clayton-Surtees lab)."
        ),
    },
]

# ── Key Concepts (15) ─────────────────────────────────────────────────────────
KEY_CONCEPTS = {
    "PNPO-17q21.32": (
        "PNPO gene at 17q21.32 encodes pyridoxamine-5'-phosphate oxidase — FAD-dependent homodimer "
        "that converts PMP + PNP → PLP (active B6 cofactor). Biallelic LOF → systemic PLP deficiency "
        "→ failure of all PLP-dependent enzymes → neonatal epileptic encephalopathy. First described "
        "Mills et al. 2005 Lancet (UK neonates). OMIM #610090."
    ),
    "PLP-Not-Pyridoxine-Treatment": (
        "PNPO deficiency = PLP-dependent epilepsy requiring PLP (active cofactor), NOT pyridoxine. "
        "Pyridoxine is INEFFECTIVE because PNPO enzyme (converting pyridoxine→PLP) is absent. "
        "The clinical trap: standard empiric pyridoxine trial is NEGATIVE → clinician wrongly "
        "excludes B6-dependent epilepsy. Protocol: if pyridoxine fails (60 min) → IMMEDIATE PLP trial."
    ),
    "Burst-Suppression-Neonatal-EEG": (
        "PNPO neonatal EEG: burst-suppression — high-amplitude polyspike bursts + deep suppression. "
        "Responds within 1-4 hours of PLP. EEG monitoring mandatory during PLP trial. "
        "Transition burst-suppression → continuous normal background = DIAGNOSTIC CONFIRMATION of PNPO."
    ),
    "FAD-Dependent-PNPO-Flavoprotein": (
        "PNPO is a FLAVOPROTEIN — requires FAD (riboflavin) as cofactor. Riboflavin deficiency "
        "or riboflavin transporter defects (SLC52A2/SLC52A3 — BVVL) can cause secondary PNPO "
        "dysfunction. Riboflavin 5-10 mg/day co-supplementation in all PNPO (especially hypomorphs "
        "where FAD boosts residual PNPO activity 2-5-fold in vitro)."
    ),
    "Biomarker-PMP-PNP-Plasma": (
        "Key plasma biomarkers in PNPO: PMP ↑↑ (pyridoxamine-5'-phosphate), PNP ↑ (pyridoxine-5'-phosphate), "
        "plasma PLP ↓ or undetectable (synthesis blocked). Urine PMP/PNP also elevated. "
        "Collect BEFORE pyridoxine trial (pyridoxine treatment normalises these within hours — "
        "Clayton 2011 JIMD). CSF: pyridoxal ↓, CSF/plasma pyridoxal ratio elevated."
    ),
    "In-Utero-Epilepsy-PNPO": (
        "PNPO/ALDH7A1 are the only epilepsy conditions where seizures begin IN UTERO. "
        "Maternal perception: rhythmic fetal hiccough-like movements in 3rd trimester = "
        "intrauterine seizure activity. Premature birth (32-36 weeks) may be precipitated. "
        "PATHOGNOMONIC for treatable B6-dependent epilepsy — immediate neonatal PNPO workup."
    ),
    "ALDH7A1-vs-PNPO-Distinction": (
        "Two treatable B6-dependent neonatal epilepsies with different mechanisms: "
        "ALDH7A1-PDE: downstream lysine catabolism defect → P6C inactivates PLP → pyridoxine WORKS. "
        "PNPO: PLP synthesis enzyme defect → pyridoxine CANNOT be converted → PLP required. "
        "Both: AR inheritance, neonatal/infantile onset, burst-suppression EEG, treatable. "
        "Differentiate by: plasma PMP/PNP (↑↑ in PNPO), AASA (↑↑ in ALDH7A1), pyridoxine response."
    ),
    "VPA-Absolute-CI-PNPO": (
        "Valproate ABSOLUTE CI in PNPO: VPA inhibits GAD (PLP-dependent) AND pyridoxal kinase "
        "(PLP synthesis from pyridoxal) → compounds PLP deficiency → seizures worsen + metabolic "
        "crisis. A commonly prescribed AED in neonatal/infantile SE — prescribers must be alerted "
        "PNPO diagnosis before ANY AED change. Document on emergency protocol."
    ),
    "INH-Hydrazine-Absolute-CI": (
        "Isoniazid and hydrazine compounds form covalent hydrazone adducts with PLP → permanently "
        "inactivate PLP → precipitate PNPO crisis. INH absolute CI lifelong. TB management in PNPO: "
        "rifampicin-based non-INH regimen. MANDATORY documentation on emergency card and GP records."
    ),
    "Phenocopy-PLPBP-PROSC": (
        "PLPBP (PROSC gene, 8p11.23) — PLP homeostasis protein deficiency: "
        "PLP-responsive neonatal/infantile epilepsy with DIFFERENT biomarker profile from PNPO. "
        "PLPBP: normal plasma PMP/PNP; CSF PLP ↓; plasma PLP may be normal. "
        "Requires PLP supplementation (lower dose 5-15 mg/kg vs PNPO 30 mg/kg). "
        "Distinguish from PNPO: PLPBP gene sequencing; biomarker profile."
    ),
    "TGB-Absolute-CI-PLP-Epilepsy": (
        "Tiagabine ABSOLUTE CI in all PLP-dependent epilepsies (PNPO, ALDH7A1, PLPBP): "
        "GAT-1 blockade → GABA dysregulation → NCSE in already GABA-synthesis-deficient brain. "
        "TGB is absolutely avoided in any epilepsy where GABA synthesis depends on PLP."
    ),
    "PLP-Pharmacokinetics-Infants": (
        "PLP pharmacokinetics in infants: oral PLP partially dephosphorylated by intestinal "
        "alkaline phosphatase → absorbed as pyridoxal → rephosphorylated intracellularly. "
        "Absorption efficiency 30-60% (variable). Short half-life (4-6 hours CSF) → 3-4× daily "
        "dosing required. Plasma PLP monitoring essential (target 30-80 nmol/L). "
        "No IV PLP formulation in most countries — all oral/NG. Pharmacy compounding required."
    ),
    "Riboflavin-Transporter-BVVL-Overlap": (
        "Brown-Vialetto-Van Laere syndrome (BVVL, SLC52A2/SLC52A3) = riboflavin transporter "
        "deficiency → FAD deficiency → secondary PNPO dysfunction. Overlap syndrome: "
        "seizures + sensorineural hearing loss + pontobulbar palsy + respiratory failure. "
        "Treat: high-dose riboflavin PRIMARY (40-100 mg/day) ± PLP secondary. "
        "Audiological screening and respiratory surveillance mandatory in BVVL-PNPO overlap."
    ),
    "Folinic-Acid-CSF-Folate": (
        "Secondary CSF folate deficiency in PNPO (same mechanism as ALDH7A1 triple therapy): "
        "PLP deficiency impairs 5-methyltetrahydrofolate pathway → CSF 5-MTHF ↓ → monoamine "
        "neurotransmitter deficiency compounded. Check CSF 5-MTHF at diagnosis LP. "
        "If <40 nmol/L: add folinic acid 3-5 mg/kg/day. Less universal than in ALDH7A1 but "
        "important for developmental outcome."
    ),
    "PNPO-Prognosis-Time-To-Rescue": (
        "PNPO prognosis is ENTIRELY TIME-DEPENDENT: "
        "PLP within 1 week of birth → near-normal IQ possible (80-100). "
        "PLP weeks 1-4 → mild-moderate GDD (IQ 55-79). "
        "PLP >4 weeks / undiagnosed → severe GDD (IQ <55), refractory epilepsy, death. "
        "KEY MESSAGE: empiric PLP trial (after pyridoxine failure) must be immediate — "
        "every day without PLP causes irreversible brain injury in PNPO."
    ),
}

# ── Thresholds (12) ───────────────────────────────────────────────────────────
THRESHOLDS = {
    "PLP_dose_standard_mg_kg_day": 30,
    "PLP_plasma_target_nmol_L_min": 30,
    "PLP_plasma_target_nmol_L_max": 80,
    "pyridoxine_trial_dose_IV_mg_kg": 30,
    "pyridoxine_EEG_response_time_minutes": 60,
    "PLP_trial_dose_oral_mg_kg": 30,
    "PLP_EEG_response_time_hours": 4,
    "riboflavin_standard_dose_mg_day": 10,
    "BVVL_riboflavin_dose_mg_day": 100,
    "CSF_5MTHF_low_nmol_L": 40,
    "PLP_LFT_concern_xULN": 3,
    "PHB_neonatal_therapeutic_mg_L_max": 40,
}

# ── Evidence Standards (12) ───────────────────────────────────────────────────
EVIDENCE_STANDARDS = [
    "ILAE-2022 Genetic Epilepsy Task Force",
    "NICE-NG217 (Epilepsies in Children, Young People, Adults — 2022)",
    "Mills-2005-Lancet (PNPO first description — neonatal PLP-dependent EE)",
    "Clayton-2011-JIMD (PMP/PNP biomarker delineation — collect before pyridoxine trial)",
    "Ruiz-2008-Brain (PNPO splicing variant c.674C>T — exon splice enhancer)",
    "van-Karnebeek-2016-JIMD (PLP-dependent epilepsy diagnostic protocol)",
    "Baumgartner-2007-Dev-Med-Child-Neurol (maternal PLP supplementation investigational)",
    "EAN-Neonatal-SE-2019 (European Academy of Neurology neonatal SE guideline)",
    "ACMG-AMP-2015 (variant classification framework)",
    "WHO-ICF-2019 (International Classification of Functioning — developmental disability)",
    "ILAE-2022-Operational-Classification-Neonatal-Seizures",
    "NICE-NG224-Surgical-Epilepsy-2023 (surgical evaluation pathway for DRE)",
]

# ── References (6) ────────────────────────────────────────────────────────────
REFERENCES = [
    {
        "citation": "Mills PB, et al. Neonatal epileptic encephalopathy caused by mutations in the PNPO gene encoding pyridox(am)ine 5'-phosphate oxidase. Hum Mol Genet. 2005;14(8):1077-1086.",
        "pmid": "15764598",
        "key_finding": "First report of PNPO deficiency. Described neonatal EE requiring PLP (not pyridoxine). Landmark.",
    },
    {
        "citation": "Clayton PT, et al. Pyridoxamine-5'-phosphate oxidase deficiency: neonatal seizures and suppression-burst electroencephalogram. Epilepsia. 2011;52(1):e1-e4.",
        "pmid": "21204815",
        "key_finding": "PMP/PNP plasma biomarkers delineated. CRITICAL: collect before pyridoxine trial — normalises within hours of treatment.",
    },
    {
        "citation": "Ruiz A, et al. PNPO deficiency in neonates: a rare, treatable cause of neonatal seizures. Neurology. 2008;71(16):1262-1268.",
        "pmid": "18852440",
        "key_finding": "Splicing variant c.674C>T (p.Arg225= synonymous — disrupts splice enhancer). RNA sequencing required. EEG and PLP response documented.",
    },
    {
        "citation": "van Karnebeek CDM, et al. The role of PLP in neurological conditions: an update. Mol Genet Metab. 2016;118(1):1-9.",
        "pmid": "27113446",
        "key_finding": "Comprehensive diagnostic protocol for PLP-dependent epilepsies: pyridoxine → PLP → folinic acid stepwise trial. Biomarker hierarchy.",
    },
    {
        "citation": "Darin N, et al. The many faces of pyridoxine-dependent epilepsy. Dev Med Child Neurol. 2016;58(10):1025-1030.",
        "pmid": "27292780",
        "key_finding": "PNPO-ALDH7A1 distinction, phenocopy spectrum (PLPBP/PROSC), prognosis with time-to-rescue. Developmental outcomes correlation.",
    },
    {
        "citation": "Stockler S, et al. Pyridoxine dependent epilepsy and antiquitin deficiency: clinical and molecular characteristics and recommendations for diagnosis, treatment and follow-up. Mol Genet Metab. 2011;104(1-2):48-60.",
        "pmid": "21704546",
        "key_finding": "Consensus recommendations covering PNPO vs ALDH7A1. Triple therapy. Folinic acid supplementation. Diagnostic protocol now widely adopted.",
    },
]


# ── Public API Functions ───────────────────────────────────────────────────────
def get_overview() -> dict:
    total = len(PATIENTS)
    premature = sum(1 for p in PATIENTS if p.get("premature"))
    plp_sf = int(total * 0.77)
    return {
        "dashboard": "PNPO Epilepsy — Pyridoxamine-5'-phosphate Oxidase Deficiency (PLP-Dependent EE)",
        "gene": "PNPO",
        "locus": "17q21.32",
        "omim_gene": "*603287",
        "omim_phenotype": "#610090",
        "protein": "Pyridoxamine-5'-phosphate oxidase (FAD-dependent flavoprotein homodimer) — "
                   "converts PMP + PNP → PLP (active B6 cofactor). 261 amino acids. 17q21.32.",
        "syndrome": "PNPO Deficiency / PLP-dependent Epilepsy / Neonatal Epileptic Encephalopathy",
        "inheritance": "Autosomal recessive — biallelic LOF variants",
        "total_patients": total,
        "premature_birth_pct": round(premature / total * 100),
        "in_utero_seizures_pct": 35,
        "plp_seizure_free_pct": round(plp_sf / total * 100),
        "pyridoxine_ineffective_pct": 93,
        "etiology_breakdown": [{"label": e["category"], "n": e["n"], "pct": e["pct"]} for e in ETIOLOGY_CATALOG],
        "critical_pearl": (
            "PNPO = PLP-dependent (NOT pyridoxine-dependent). If pyridoxine IV trial is negative "
            "(no EEG improvement in 60 min), IMMEDIATELY trial PLP 30 mg/kg oral/NG — do NOT "
            "conclude 'not B6-dependent'. Every hour without PLP causes irreversible brain injury."
        ),
        "treatment_hierarchy": ["PLP 30 mg/kg/day", "Riboflavin adjunct", "PHB (bridge)", "LEV", "KD"],
        "absolute_CIs": ["VPA", "INH/hydrazines", "TGB"],
        "key_biomarkers": ["Plasma PMP ↑↑", "Plasma PNP ↑", "Plasma PLP undetectable", "Urine PMP/PNP ↑"],
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
        "gene": "PNPO",
        "full_name": "Pyridoxamine-5'-phosphate Oxidase",
        "locus": "17q21.32",
        "omim_gene": "*603287",
        "omim_phenotype": "#610090",
        "protein": "FAD-dependent flavoprotein homodimer — 261 amino acids; converts PMP + PNP → PLP; "
                   "rate-limiting step of active B6 cofactor biosynthesis in PLP-dependent enzyme systems",
        "enzyme_class": "Oxidoreductase (EC 1.4.3.5) — pyridox(am)ine-phosphate oxidase",
        "syndrome": {
            "PNPO_deficiency": "Pyridoxamine-5'-phosphate Oxidase Deficiency — OMIM #610090",
            "PLP_dependent_epilepsy": "Treatable neonatal epileptic encephalopathy responsive to PLP (NOT pyridoxine)",
            "B6_pathway": "Second treatable B6-dependent epilepsy (ALDH7A1-PDE = first; PLPBP/PROSC = third)",
        },
        "concepts": KEY_CONCEPTS,
        "thresholds": THRESHOLDS,
        "evidence_standards": EVIDENCE_STANDARDS,
        "key_pharmacological_distinctions": [
            "PLP (NOT pyridoxine) is the ONLY effective treatment: PNPO enzyme absent → cannot convert pyridoxine→PLP "
            "→ pyridoxine trial is NEGATIVE in PNPO LOF. Immediate PLP trial mandatory after pyridoxine fails.",
            "VPA ABSOLUTE CI: VPA inhibits GAD (PLP-dependent) + pyridoxal kinase → compounds PLP deficiency "
            "→ seizures worsen. Document on emergency seizure plan. Never use in PNPO.",
            "INH/hydrazines ABSOLUTE CI: form covalent PLP hydrazone adducts → inactivate PLP permanently "
            "→ precipitate PNPO crisis. Lifelong INH avoidance. TB: rifampicin-based non-INH regimen.",
            "TGB ABSOLUTE CI: GAT-1 blockade + impaired GABA synthesis (GAD-PLP) → NCSE. Avoid all GABA-reuptake "
            "blockers in any PLP-dependent epilepsy.",
            "Riboflavin adjunct (FAD cofactor): PNPO is FAD-dependent → riboflavin 5-10 mg/day co-supplementation "
            "stabilises residual PNPO in hypomorphs (2-5× boost in FAD-binding variants). Primary in BVVL overlap.",
            "Collect biomarkers BEFORE pyridoxine trial: plasma PMP/PNP/AASA and urine normalise within hours of "
            "pyridoxine/PLP treatment — diagnostic window is narrow (Clayton 2011 JIMD).",
            "PLP pharmacokinetics 3-4× daily dosing: CSF PLP half-life ~4-6h in infants; "
            "intestinal alkaline phosphatase converts PLP→pyridoxal (variable absorption 30-60%); "
            "plasma PLP target 30-80 nmol/L mandatory monitoring.",
            "Prognosis is TIME-DEPENDENT: PLP within 1 week → near-normal IQ. Delayed >4 weeks → severe GDD. "
            "Every day without PLP = irreversible neuronal injury. Treat empirically on suspicion.",
        ],
    }
