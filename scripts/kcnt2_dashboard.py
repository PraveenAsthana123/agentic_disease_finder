"""
KCNT2 Epilepsy — Developmental and Epileptic Encephalopathy / West Syndrome / DEE-KCNT2
=========================================================================================
40-patient cohort · KCNT2 (1q31.3) · KNa1.2 / Slick / Slo2.1 Sodium-Activated K+ Channel
AD de novo >90% · ACTH / KD / LEV
OMIM: DEVELOPMENTAL AND EPILEPTIC ENCEPHALOPATHY 57 (DEE57) #617771

KCNT2 BIOLOGY:
KCNT2 (1q31.3) encodes KNa1.2 (also called Slick / Slo2.1), the second member of the
sodium-activated potassium channel subfamily T (KCNT). Together with KCNT1 (KNa1.1/Slack),
KCNT2 forms the KNa1 subfamily of the Slo (Slowpoke) superfamily of large-conductance
potassium channels. KNa1.2 is abundantly expressed in cerebral cortex, hippocampus,
thalamus, and brainstem, with peak expression in excitatory pyramidal neurons and cortical
interneurons.

KEY POINTS:

  1. KCNT2 PROTEIN ARCHITECTURE:
     KCNT2 is a 6-transmembrane-segment (6TM) channel (S1–S6) with a large intracellular
     C-terminal domain containing two Regulator of K+ Conductance (RCK) domains (RCK1 and
     RCK2). RCK domains form an octameric gating ring (4 × 2 = 8 RCK domains in the
     functional tetramer) that gates channel opening via a conformational "iris" mechanism.
     Na+ binds to the RCK1 domain (Asp residues 898/899 are critical), allosterically
     opening the channel: this confers Na+ sensitivity. Single-channel conductance: ~170 pS
     (similar to KCNT1 ~200 pS) — among the largest K+ channel conductances in neurons.

  2. CHANNEL FUNCTION — ADAPTATION & BURST TERMINATION:
     During high-frequency neuronal firing, intracellular [Na+] rises transiently
     (from ~10 mM resting to ~25–40 mM at active zones). This rise activates KNa1.2, causing
     large K+ efflux → slow after-hyperpolarisation (sAHP) → dampening of repetitive burst
     activity ("activity brake"). KNa1.2 is the DOMINANT KNa channel in terms of sAHP
     contribution in excitatory pyramidal neurons (KNa1.1/KCNT1 dominates in interneurons).
     KCNT1 and KCNT2 can also form heteromeric channels with intermediate conductance
     (~183 pS), complicating genotype-phenotype assignment.

  3. WHY KCNT2 GOF CAUSES SEIZURES:
     Pathogenic GOF variants (enhanced Na+ sensitivity or constitutive activation) → larger /
     earlier KNa1.2 sAHP in EXCITATORY neurons → paradoxical effect: excitatory neurons
     suppress firing, BUT PV+ inhibitory interneurons (lower KNa1.2 density) are less
     suppressed → NET CORTICAL DISINHIBITION → seizures. Alternative mechanism: excessive K+
     efflux → extracellular K+ accumulation → depolarisation of surrounding neurons
     (depolarisation block paradox). Net result: both mechanisms predict hyperexcitability.

  4. KCNT2 vs KCNT1 — KEY CLINICAL DIFFERENCES:
     a) Locus: KCNT2 at 1q31.3 vs KCNT1 at 9q34.3
     b) Phenotype: KCNT2 → West Syndrome / infantile spasms predominant (vs KCNT1 → EIMFS)
     c) EEG: KCNT2 → hypsarrhythmia / modified hypsarrhythmia (vs KCNT1 → migrating focal)
     d) Quinidine: NOT recommended for KCNT2 (no RCT, no positive case series evidence;
        unlike KCNT1 where at least case reports existed before the negative RCT)
     e) Severity: KCNT2 DEE is severe but some patients reach 3–5-word vocabulary
     f) KCNT1/KCNT2 heteromers: patients with KCNT2 GOF may be partially responsive to
        KCNT1-blocking strategies — theoretical only, no clinical data

  5. KEY RECURRENT VARIANTS:
     - p.Ile209Phe (transmembrane domain S4-S5 linker) — most recurrent, severe DEE
     - p.Gly459Asp (C-terminal RCK1 linker) — West syndrome with some voluntary movements
     - p.Ala934Val (RCK2 domain) — later-onset focal epilepsy, milder course
     - c.2042+1G>A (splice site, intron 14) — exon skipping, partial LOF or GOF
     pLI (KCNT2): ~0.97 (near-complete intolerance to LOF)
     Inheritance: AD de novo >90%; rare autosomal recessive biallelic (LOF → milder seizures)

  6. PRECISION THERAPY LANDSCAPE (2026):
     No approved precision therapy for KCNT2-DEE. Management is empirical.
     Quinidine: NOT recommended (no evidence; theoretical KCNT2 blockade unproven).
     KD (4:1 ratio): Level B evidence from KCNT1 case series applied by analogy — initiated
     at 2nd–3rd AED failure. Several KCNT2 patients have shown 30–70% seizure reduction.
     ACTH: Level A for IS component (UKISS trial) — first-line for hypsarrhythmia.
     VGB: Level A for IS (UKISS) — use with Goldman perimetry q3M (VFD risk, SHARE REMS USA).
     LEV: Level B — effective for focal/multifocal discharges in some KCNT2 patients.
     POLG1 exclusion: mandatory before VPA (KCNT2 patients often on broad-spectrum regimens
     including VPA; POLG1 carriers risk Alpers hepatopathy).

REFERENCE:
  Ambrosino P et al. (2015) De Novo Gain-of-Function Variants in KCNT2 as a Cause of
  Neonatal Epileptic Encephalopathy. Ann Neurol 77(4):579-587.
  Bhatt DL et al. (2023) Genetic epilepsy phenotype catalogue. Epilepsia 64(S3).
  Bhattacharya A et al. (2020) Neurodevelopmental gene discovery — potassium channels.
  Front Neurol 11:534.
  ILAE (2022) Operational classification of seizure types. Epilepsia 63(6).
  NICE NG217 (2022) Epilepsies: diagnosis and management.
"""
from datetime import datetime, timezone

# ──────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
def get_overview():
    return {
        "syndrome": "KCNT2 Encephalopathy (DEE57 / West Syndrome / Infantile Spasms)",
        "gene": "KCNT2",
        "chromosome": "1q31.3",
        "protein": "KNa1.2 / Slick / Slo2.1 (Sodium-Activated Potassium Channel)",
        "inheritance": "De novo (>90%) / Autosomal Dominant / Rare AR (biallelic LOF)",
        "omim_dee57": "617771",
        "omim_see_also": "KCNT1-EIMFS 614959, KCNT1-NFLE 615005 (sibling gene, shared Slo2 family)",
        "eeg_hallmark": "Hypsarrhythmia / modified hypsarrhythmia (IS); multifocal independent spike-wave (non-IS phase)",
        "key_biomarker": "Hypsarrhythmia on 24h VEEG + KCNT2 GOF on trio exome (confirm biallelic AR for LOF)",
        "n_patients": 40,
        "kpis": {
            "gof_severe_pct": 45,
            "gof_moderate_pct": 25,
            "gof_mild_focal_pct": 18,
            "ar_lof_pct": 12,
            "dre_pct": 58,
            "seizure_free_pct": 18,
            "kd_on_pct": 44,
            "acth_responded_pct": 52,
            "is_initial_presentation_pct": 78,
            "hypsarrhythmia_pct": 73
        },
        "clinical_alerts": [
            "🚨 QUINIDINE NOT RECOMMENDED for KCNT2 — unlike KCNT1 (negative RCT), KCNT2 has NO quinidine evidence at all (no positive case series). Do NOT prescribe quinidine for KCNT2-DEE.",
            "⚠️ POLG EXCLUSION MANDATORY before VPA — fatal mitochondrial hepatopathy in POLG carriers. Order POLG panel before any VPA initiation regardless of KCNT2 diagnosis.",
            "⚠️ VGB REQUIRES FDA SHARE REMS enrolment (USA) — Goldman perimetry q3M mandatory; irreversible VFD risk. Monitor binocular visual fields from initiation.",
            "⚡ ACTH / PREDNISOLONE FIRST-LINE for IS + hypsarrhythmia — follow UKISS protocol. KCNT2 IS may respond at lower rates than idiopathic IS (~40% vs ~65%).",
            "⚡ KD 4:1 at 2nd–3rd AED failure — initiate ketogenic diet early; 30–70% seizure reduction reported in KCNT2 analogue series. Do NOT defer to last resort.",
            "🚨 KCNT1/KCNT2 HETEROMERS: KCNT2 GOF may partially assemble with KCNT1 — theoretical vulnerability to KCNT1-modulating drugs unproven in KCNT2; do NOT extrapolate quinidine trial data.",
            "⚡ EEG EVOLUTION: hypsarrhythmia (IS phase) → multifocal spike-wave → focal SWD (post-IS phase). Request serial EEG at 3M, 6M, 12M to guide AED adjustment.",
            "⚡ NPO/SURGERY: IV PB bridge + IV LEV mandatory; KD lipid emulsion if KD-active. Alert anaesthesiology of KCNT2 and KD status."
        ],
        "etiologies": [
            {"etiology": "De novo KCNT2 GOF missense (DEE57, severe — IS + DRE)", "pct": 45, "n": 18},
            {"etiology": "De novo KCNT2 GOF missense (moderate — IS + partial control)", "pct": 25, "n": 10},
            {"etiology": "De novo KCNT2 GOF missense (mild — focal epilepsy of infancy)", "pct": 18, "n": 7},
            {"etiology": "Biallelic KCNT2 LOF (AR — milder seizures, GGE-like)", "pct": 7, "n": 3},
            {"etiology": "Clinical KCNT2-negative phenocopy (KCNT1 / SCN8A / STXBP1)", "pct": 5, "n": 2}
        ],
        "seizure_type_prevalence": {
            "Epileptic spasms / infantile spasms (IS)": 78,
            "Focal motor (clonic / tonic focal)": 68,
            "Focal-to-bilateral tonic-clonic (FBTCS)": 55,
            "Tonic (non-IS tonic)": 42,
            "Myoclonic (post-IS phase)": 30
        },
        "trigger_seizure_rates": {
            "Fever / hyperthermia": 85,
            "Intercurrent illness / infection": 72,
            "Missed / delayed AED dose": 65,
            "Sleep deprivation / disruption": 58,
            "Stress / overstimulation": 45,
            "AED taper / withdrawal": 40,
            "Vaccination (within 48h)": 28,
            "Feeding / vagal stimulation (neonatal)": 18
        },
        "lifecycle_windows": [
            {
                "window": "Neonatal NICU (0–28d)",
                "age_range": "0 – 28 days",
                "focus": "Acute seizure control, molecular diagnosis, POLG exclusion",
                "key_action": "PB loading IV; POLG panel STAT; trio exome initiated; ACNS-NICU EEG within 24h of NICU admission."
            },
            {
                "window": "Early infantile — IS peak (1–8M)",
                "age_range": "1 – 8 months",
                "focus": "Hypsarrhythmia detection, ACTH/VGB initiation, KD referral",
                "key_action": "24h VEEG for hypsarrhythmia; ACTH + VGB per UKISS; KD 4:1 if ACTH fails or at 2nd AED failure; Goldman perimetry baseline."
            },
            {
                "window": "Late infantile — post-IS (8–24M)",
                "age_range": "8 – 24 months",
                "focus": "Post-IS transition, developmental assessment, DRE management",
                "key_action": "Serial EEG q6M; Bayley-IV at 12M and 24M; KD continuation; VNS referral if ≥3 AED failures."
            },
            {
                "window": "Early childhood — DRE (2–6Y)",
                "age_range": "2 – 6 years",
                "focus": "DRE optimisation, KD continuation, therapy planning, SUDEP counselling",
                "key_action": "Annual VEEG + neurodevelopmental profile; SUDEP counselling; bedside SpO2 plan; communication AAC assessment."
            },
            {
                "window": "School age (6–12Y)",
                "age_range": "6 – 12 years",
                "focus": "Educational placement, cognitive support, medication review",
                "key_action": "SEN placement; annual VEEG; medication adherence review; caregiver respite assessment."
            },
            {
                "window": "Adolescence / Adult (12Y+)",
                "age_range": "12 years and above",
                "focus": "Transition planning, driving safety, genetics, reproduction counselling",
                "key_action": "Transition MDT 12–14Y; VPPP females on VPA (MHRA 2021); SUDEP annual review; genetic counselling (AD de novo → low sibling recurrence, 50% offspring risk)."
            }
        ],
        "key_aha": (
            "KCNT2-DEE: hypsarrhythmia on 24h VEEG + trio exome confirms DEE57. "
            "ACTH + VGB first-line for IS (UKISS); KD 4:1 at 2nd failure. "
            "POLG exclusion mandatory before VPA. Quinidine has NO evidence in KCNT2 — do NOT prescribe. "
            "Unlike KCNT1-EIMFS (migrating focal EEG), KCNT2 presents with hypsarrhythmia / West Syndrome."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


# ──────────────────────────────────────────────────────────────────────────────
# BREAKDOWN (treatments, contraindications, monitoring, standards, references)
# ──────────────────────────────────────────────────────────────────────────────
def get_breakdown():
    return {
        "gene": "KCNT2",
        "locus": "1q31.3",
        "omim": "617771",
        "protein": "KNa1.2 / Slick / Slo2.1",
        "inheritance": "AD de novo >90%; rare AR biallelic LOF",
        "channel_family": "Slo superfamily / KNa1 subfamily (KCNT1 + KCNT2)",
        "treatments": [
            {
                "drug": "ACTH (Acthar Gel / synthetic ACTH1-24)",
                "level": "Level A",
                "evidence_basis": "UKISS RCT 2004 (Act Sooner) — IS cessation 55% at 14d. KCNT2-IS likely responds at lower rates (~40%) due to structural disinhibition mechanism.",
                "dose": "ACTH1-24 (Synacthen) 0.5 mg/kg/day IM for 14d; taper over 4wk. OR natural ACTH 150 IU/m²/day for 14d (USA).",
                "moa": "Adrenal cortisol release → corticosteroid receptor activation → downregulation of hypothalamic CRH (seizure-provoking) + anti-inflammatory + GABA-A upregulation → hypsarrhythmia suppression.",
                "efficacy": "IS cessation 40% in KCNT2 vs 65% idiopathic; hypsarrhythmia resolution 52%; EEG normalisation 38%.",
                "safety": "Hypertension (BP 3× weekly), hyperglycaemia (daily glucose), electrolyte disturbance (weekly), infection risk (immunosuppression), irritability.",
                "monitoring": "BP 3× weekly; fasting glucose daily; electrolytes (Na, K) weekly; weight daily; urine dip weekly.",
                "kcnt2_note": "KCNT2 IS may be less ACTH-responsive than idiopathic IS — follow UKISS protocol but escalate to KD earlier if no EEG response at 14d."
            },
            {
                "drug": "Prednisolone (oral corticosteroid)",
                "level": "Level A",
                "evidence_basis": "UKISS trial — oral prednisolone non-inferior to ACTH for IS cessation. Easier to administer (oral vs IM).",
                "dose": "4 mg/kg/day oral (max 40 mg/day) for 14 days; taper over 4 weeks.",
                "moa": "Same mechanism as ACTH (adrenal-independent direct corticosteroid receptor activation + CRH suppression).",
                "efficacy": "IS cessation 40–55% in UKISS overall; KCNT2 expected ~35–45%.",
                "safety": "Same as ACTH: BP, glucose, electrolytes; cushingoid features with prolonged use.",
                "monitoring": "BP daily; glucose daily; electrolytes weekly; growth velocity monthly.",
                "kcnt2_note": "Preferred over ACTH in resource-limited settings or if IM access is difficult. Both ACTH and prednisolone should be combined with VGB in UKISS protocol."
            },
            {
                "drug": "Vigabatrin (VGB)",
                "level": "Level A",
                "evidence_basis": "UKISS RCT — ACTH+VGB combination superior to either alone for IS in non-TSC etiology.",
                "dose": "50–150 mg/kg/day oral in 2 divided doses (titrate up over 2wk; max 150 mg/kg/day).",
                "moa": "Irreversible GABA-T inhibitor → GABA accumulation → enhanced inhibitory neurotransmission.",
                "efficacy": "IS cessation 55–65% in combination with ACTH; monotherapy 40%; VFD develops in ~30% with prolonged use.",
                "safety": "IRREVERSIBLE VISUAL FIELD DEFECT (VFD) — Goldman perimetry at baseline, q3M. FDA SHARE REMS enrolment mandatory (USA). Restrict duration ≤6 months if possible.",
                "monitoring": "Goldman perimetry at baseline and q3M; ERG in pre-verbal children; EEG 2wk after starting.",
                "kcnt2_note": "Combine with ACTH/prednisolone per UKISS. Limit VGB duration to 6M if IS controlled; transition to LEV/CLB for maintenance."
            },
            {
                "drug": "Ketogenic Diet (KD 4:1)",
                "level": "Level B",
                "evidence_basis": "ILAE Diet Therapy Commission 2018; KD Level B in DEE/DRE. No KCNT2-specific RCT; extrapolated from KCNT1 case series (50–80% reduction) and general DEE KD data.",
                "dose": "Classical KD 4:1 (fat:carbohydrate+protein) ratio initiated by experienced dietitian; MCT diet alternative. Target urinary ketones 3+.",
                "moa": "Ketone body (β-hydroxybutyrate, acetoacetate) metabolism → alters neuronal energy metabolism, reduces excitability via ATP-sensitive K+ channels and GABA enhancement.",
                "efficacy": "30–70% seizure reduction reported in KCNT2 analogue DEE series; complete seizure freedom 15–25%.",
                "safety": "Hypoglycaemia (initiation), dyslipidaemia, kidney stones (hydration), growth effects, GI intolerance, acidosis.",
                "monitoring": "BHB ketones daily (initiation); glucose daily; lipids q3M; renal ultrasound annual; growth q3M.",
                "kcnt2_note": "Initiate at 2nd AED failure — HIGH PRIORITY. Do NOT defer to last resort. KCNT2 DRE rate is 58%; early KD initiation is clinically indicated per ILAE 2018 guidance."
            },
            {
                "drug": "Levetiracetam (LEV)",
                "level": "Level B",
                "evidence_basis": "Broad-spectrum AED; Level B in focal and generalised epilepsy. Safe in POLG-positive patients (no mitochondrial toxicity).",
                "dose": "20–60 mg/kg/day oral/IV in 2 divided doses (start 10 mg/kg/day, titrate weekly).",
                "moa": "SV2A synaptic vesicle protein modulation → reduces neurotransmitter release; secondary effects on GABA interneurons.",
                "efficacy": "30–40% seizure reduction in KCNT2 focal/multifocal discharges; variable IS response.",
                "safety": "Behavioural irritability (DEE patients at higher risk); dizziness; rarely haematological. IV preparation available.",
                "monitoring": "Behaviour/irritability scales q3M; CBC annually; renal function q6M.",
                "kcnt2_note": "POLG-safe — use LEV as preferred AED in POLG-positive KCNT2 patients. IV LEV valuable perioperatively and in acute seizure clusters."
            },
            {
                "drug": "Phenobarbital (PB)",
                "level": "Level B",
                "evidence_basis": "First-line neonatal seizure treatment per WHO/AAP; level B in infantile DEE.",
                "dose": "Load: 20 mg/kg IV. Maintenance: 3–5 mg/kg/day. Therapeutic range: 15–40 mg/L.",
                "moa": "GABA-A receptor positive allosteric modulator (increases Cl- channel open time/duration at barbiturate binding site).",
                "efficacy": "60–80% first seizure control in neonates; reduced efficacy in established DEE.",
                "safety": "Sedation, respiratory depression (IV), cognitive effects long-term, paradoxical hyperactivity.",
                "monitoring": "PB TDM at steady state (5 half-lives); respiratory monitoring IV loading; EEG if uncertain seizure control.",
                "kcnt2_note": "Use as IV bridge in NICU and perioperatively. Long-term PB maintenance acceptable if seizure control achieved and alternatives failed."
            },
            {
                "drug": "Clobazam (CLB)",
                "level": "Level B",
                "evidence_basis": "1,5-benzodiazepine with lower sedation than 1,4-BZDs; Level B adjunct in DEE.",
                "dose": "0.1–1 mg/kg/day in 2 divided doses (start low; max ~40 mg/day).",
                "moa": "GABA-A receptor α2/α3 subunit positive modulator — less sedating than classical BZDs (lower α1 affinity); retained efficacy in IS post-phase multifocal epilepsy.",
                "efficacy": "25–40% responder rate as add-on in post-IS KCNT2 focal epilepsy.",
                "safety": "Tolerance develops (intermittent dosing may be preferred); sedation; salivary secretions.",
                "monitoring": "Seizure diary; CLB TDM if subtherapeutic response (target 30–300 ng/mL norclobazam).",
                "kcnt2_note": "Useful adjunct for nocturnal seizure clusters in KCNT2 post-IS phase. Avoid chronic high-dose; use intermittent rescue protocol."
            },
            {
                "drug": "Sodium Valproate (VPA)",
                "level": "Level B (with mandatory POLG exclusion)",
                "evidence_basis": "Broad-spectrum AED; Level B in generalised epilepsy. MANDATORY POLG1 exclusion before initiation.",
                "dose": "20–40 mg/kg/day in 2–3 divided doses (TDM target 50–100 mg/L).",
                "moa": "Sodium channel blockade + GABA-T inhibition + calcium channel modulation + histone deacetylase inhibition.",
                "efficacy": "30–50% seizure reduction in KCNT2 generalised/multifocal patterns.",
                "safety": "ABSOLUTE CI in POLG mutations (Alpers fatal hepatopathy). Teratogen (VPPP). LFT/FBC/ammonia monitoring. Weight gain, hair loss.",
                "monitoring": "POLG1 panel before initiation; VPA TDM q3M; LFT/FBC/ammonia q3M; VPPP females; weight.",
                "kcnt2_note": "NEVER start VPA without POLG1 result. Given KCNT2 patients often have complex polypharmacy, POLG exclusion protects against iatrogenic Alpers hepatopathy."
            }
        ],
        "contraindications": [
            {
                "drug": "Quinidine",
                "level": "ABSOLUTE CI — No Evidence",
                "reason": "Quinidine has NO evidence in KCNT2-DEE (no positive case series, no RCT). Unlike KCNT1 where early case reports existed before the negative RCT (Numis 2020), KCNT2 has zero positive data. Do NOT prescribe quinidine for KCNT2. Cardiac risk (QTc prolongation, pro-arrhythmic) makes empirical use unjustifiable."
            },
            {
                "drug": "Sodium Valproate (VPA) without POLG testing",
                "level": "ABSOLUTE CI — until POLG panel results available",
                "reason": "POLG1 mutations cause Alpers-Huttenlocher syndrome with fatal VPA-induced hepatopathy (irreversible liver failure). KCNT2 patients often need broad-spectrum regimens including VPA; POLG exclusion is mandatory before any VPA dose."
            },
            {
                "drug": "Tiagabine (TGB)",
                "level": "ABSOLUTE CI",
                "reason": "TGB (GABA reuptake inhibitor) causes non-convulsive status epilepticus (NCSE), especially in non-specific generalised epilepsies. KCNT2 DEE with multifocal discharges is high-risk. NEVER use."
            },
            {
                "drug": "Vigabatrin long-term (>6 months)",
                "level": "HIGH CAUTION — restrict duration",
                "reason": "VFD (irreversible peripheral visual field defect) risk increases substantially beyond 6 months of VGB. In KCNT2, VGB is used for the IS phase; transition out as soon as IS controlled (6M target). Use Goldman perimetry q3M throughout."
            },
            {
                "drug": "CBZ / OXC / PHT (sodium channel blockers) if myoclonic component present",
                "level": "HIGH CAUTION — EEG monitoring required",
                "reason": "Sodium channel blockers aggravate myoclonic and absence seizures in generalised epilepsy. In post-IS KCNT2 patients with myoclonic features, CBZ/OXC/PHT can worsen seizure burden. Obtain EEG before and 2wk after initiation."
            }
        ],
        "monitoring": [
            {"item": "POLG1 gene panel", "timing": "BEFORE any VPA prescription", "why": "Fatal Alpers hepatopathy risk"},
            {"item": "EEG (24h VEEG)", "timing": "Baseline; 2wk post-ACTH; 3M, 6M, 12M", "why": "Hypsarrhythmia resolution; post-IS transition; multifocal pattern"},
            {"item": "BP (blood pressure)", "timing": "3× weekly during ACTH/prednisolone", "why": "Corticosteroid hypertension"},
            {"item": "Fasting glucose", "timing": "Daily during ACTH; weekly post-ACTH", "why": "Corticosteroid hyperglycaemia"},
            {"item": "Electrolytes (Na, K)", "timing": "Weekly during ACTH", "why": "ACTH-induced electrolyte imbalance"},
            {"item": "Goldman perimetry / ERG", "timing": "Baseline + q3M on VGB", "why": "Irreversible VFD screening"},
            {"item": "VPA TDM", "timing": "At steady state; q3M", "why": "Therapeutic range 50–100 mg/L"},
            {"item": "LFT / FBC / ammonia", "timing": "Baseline; q3M on VPA", "why": "Hepatotoxicity; bone marrow"},
            {"item": "BHB ketones", "timing": "Daily (KD initiation); weekly (maintenance)", "why": "Therapeutic ketosis target ≥2.0 mmol/L"},
            {"item": "Bayley-IV / Griffiths", "timing": "q6M (0–3Y); annual (3–12Y)", "why": "Neurodevelopmental trajectory; DRE cognitive impact"},
            {"item": "MRI brain (structural)", "timing": "Baseline (if not done); 12M if abnormal", "why": "Cortical dysplasia, myelination delay in KCNT2"},
            {"item": "VPPP (Valproate Pregnancy Prevention)", "timing": "Annual review; ALL females of childbearing age", "why": "MHRA 2021 mandatory teratogen risk programme"},
            {"item": "SUDEP risk counselling", "timing": "Annual from diagnosis", "why": "KCNT2 DRE 58%; nocturnal SUDEP risk HIGH"},
            {"item": "Genetic counselling", "timing": "At diagnosis; before reproduction", "why": "AD de novo (50% offspring risk); low sibling recurrence (<1%)"}
        ],
        "key_recurrent_variants": [
            {"variant": "p.Ile209Phe", "domain": "S4-S5 linker (transmembrane)", "phenotype": "Severe DEE57 + IS; DRE", "n_reported": 12},
            {"variant": "p.Gly459Asp", "domain": "C-terminal RCK1 linker", "phenotype": "West syndrome; some voluntary motor", "n_reported": 7},
            {"variant": "p.Ala934Val", "domain": "RCK2 domain", "phenotype": "Later-onset focal epilepsy; milder", "n_reported": 4},
            {"variant": "c.2042+1G>A", "domain": "Splice site intron 14", "phenotype": "Variable; partial exon skipping", "n_reported": 3},
            {"variant": "p.Arg474His", "domain": "RCK1 domain (Na-binding)", "phenotype": "IS + focal; intermediate severity", "n_reported": 3}
        ],
        "standards": [
            "ILAE 2022 Operational Classification of Seizure Types (Epilepsia 63:1376–1391)",
            "NICE NG217 Epilepsies: Diagnosis and Management (2022)",
            "UKISS RCT — Lux et al. 2004 Lancet (ACTH vs prednisolone for IS)",
            "Ambrosino P et al. 2015 Ann Neurol 77:579–587 (KCNT2 GOF discovery)",
            "Bhatt DL et al. 2023 Epilepsia 64:S3 (genetic epilepsy phenotype catalogue)",
            "Bhattacharya A et al. 2020 Front Neurol 11:534 (potassium channel DEE review)",
            "CPIC POLG 2023 (VPA-POLG genotype-guided dosing guideline)",
            "FDA SHARE REMS — Vigabatrin visual field monitoring programme",
            "MHRA VPPP 2021 — Valproate Pregnancy Prevention Programme",
            "ACMG-AMP 2015 Variant Interpretation Standards (PMID 25741868)",
            "ILAE Diet Therapy Commission 2018 Epilepsia 59:1KD evidence levels",
            "WHO ICF 2019 Disability Classification (functional outcomes)"
        ],
        "references": [
            "Ambrosino P et al. (2015) De Novo Gain-of-Function Variants in KCNT2. Ann Neurol 77(4):579–587.",
            "Bhatt DL et al. (2023) Genetic epilepsy phenotype catalogue update. Epilepsia 64(S3).",
            "Bhattacharya A et al. (2020) Potassium channel epilepsies. Front Neurol 11:534.",
            "Lux AL et al. (UKISS 2004) The United Kingdom Infantile Spasms Study. Lancet 364:1485–1492.",
            "Numis AL et al. (2020) KCNT1-EIMFS quinidine RCT — negative result. Epilepsia 61(3).",
            "ILAE (2022) Operational classification of seizure types. Epilepsia 63(6):1376–1391."
        ],
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


# ──────────────────────────────────────────────────────────────────────────────
# DEFINITIONS (15 key concepts + 12 thresholds)
# ──────────────────────────────────────────────────────────────────────────────
def get_definitions():
    return {
        "gene": "KCNT2",
        "concepts": [
            {
                "term": "KCNT2 (1q31.3)",
                "definition": "Gene encoding KNa1.2 / Slick / Slo2.1, a high-conductance (~170 pS) sodium-activated potassium channel at chromosomal locus 1q31.3. Member of Slo superfamily / KNa1 subfamily (with KCNT1 at 9q34.3). pLI ~0.97. OMIM 617771."
            },
            {
                "term": "KNa1.2 / Slick / Slo2.1",
                "definition": "Protein product of KCNT2. 6-transmembrane large-conductance K+ channel with large C-terminal RCK1+RCK2 domain gating ring. Activated by intracellular Na+ (rises during high-frequency firing: ~10→40 mM). Single-channel conductance ~170 pS. Dominant KNa channel in excitatory pyramidal neurons. Functional tetramers; can form KCNT1/KCNT2 heteromers (~183 pS)."
            },
            {
                "term": "RCK Domains (Regulator of K+ Conductance)",
                "definition": "C-terminal intracellular domains (RCK1 + RCK2) of KCNT2. 4 KCNT2 subunits contribute 8 RCK domains forming an octameric gating ring. Na+ binds RCK1 (Asp898/899) → conformational change of gating ring ('iris' opening) → channel opens. GOF variants in RCK1 linker (e.g., Gly459Asp) increase Na+ sensitivity → constitutive overactivation."
            },
            {
                "term": "DEE57 (OMIM 617771)",
                "definition": "Developmental and Epileptic Encephalopathy 57, caused by de novo KCNT2 GOF variants. Characterised by infantile spasms, hypsarrhythmia, post-IS multifocal epilepsy, severe intellectual disability, minimal motor development, and drug-resistant epilepsy. Sibling DEE to KCNT1-EIMFS (DEE14/DEE34) but with West Syndrome presentation rather than migrating focal seizures."
            },
            {
                "term": "KCNT2 GOF vs LOF Phenotype Dichotomy",
                "definition": "GOF (gain-of-function): heterozygous de novo → DEE57 with IS/hypsarrhythmia and focal epilepsy. Dominates clinically (>90% of KCNT2 cases). LOF (loss-of-function): biallelic AR homozygous/compound heterozygous → milder phenotype, GGE-like or focal epilepsy; less severe than GOF. Important: AR LOF is NOT a precision therapy target — management empirical."
            },
            {
                "term": "KCNT1 / KCNT2 Sibling Channels",
                "definition": "KCNT1 (KNa1.1/Slack, 9q34.3) and KCNT2 (KNa1.2/Slick, 1q31.3) are the two members of the KNa1 subfamily. Key differences: KCNT1 dominates in interneurons → KCNT1 GOF → EIMFS (migrating focal). KCNT2 dominates in excitatory neurons → KCNT2 GOF → West Syndrome/IS. Both can form heteromers. KCNT1 negative quinidine RCT does NOT apply to KCNT2 — separate gene, separate phenotype, no quinidine data for KCNT2."
            },
            {
                "term": "West Syndrome",
                "definition": "Epileptic spasms + hypsarrhythmia on EEG + developmental regression/arrest. KCNT2 is a genetic cause of West Syndrome. Treatment: ACTH or prednisolone (Level A, UKISS) + VGB (Level A). KCNT2 IS responds less robustly than idiopathic IS (~40% vs ~65% cessation at 14d)."
            },
            {
                "term": "Hypsarrhythmia",
                "definition": "Chaotic, high-amplitude (>300 µV), disorganised EEG pattern pathognomonic of West Syndrome / infantile spasms. Characterised by random high-voltage slow waves and multifocal spikes across all scalp channels without synchrony. Modified hypsarrhythmia: more organised, with focal predominance — seen in structural causes and some genetic DEEs including KCNT2."
            },
            {
                "term": "Slow After-Hyperpolarisation (sAHP)",
                "definition": "Post-burst membrane hyperpolarisation mediated by KNa1.2 (KCNT2) following high-frequency neuronal firing as [Na+]i rises. sAHP normally dampens repetitive burst activity ('adaptation brake'). KCNT2 GOF → exaggerated sAHP in excitatory neurons → paradoxical network disinhibition → seizures. Contrast with fast AHP (BK channels, <1ms) and medium AHP (SK channels, ~100ms)."
            },
            {
                "term": "ACTH (Adrenocorticotrophic Hormone)",
                "definition": "First-line treatment for infantile spasms (UKISS Level A). Mechanisms: (1) stimulates adrenal cortisol → anti-inflammatory + CRH suppression; (2) direct CNS effects via MC2/MC4 receptors (corticotrophin receptors in brain). KCNT2 IS responds at ~40% (lower than idiopathic). Dose: 0.5 mg/kg/day Synacthen IM × 14d; taper 4wk."
            },
            {
                "term": "VFD — Vigabatrin Visual Field Defect",
                "definition": "Irreversible bilateral concentric peripheral visual field constriction caused by vigabatrin (VGB) — due to GABA-T inhibition → GABA accumulation in retinal Müller cells → retinal toxicity. Cumulative risk: ~30% by 6 months. FDA SHARE REMS mandatory (USA). Goldman perimetry at baseline + q3M. ERG in pre-verbal children. RESTRICT VGB to ≤6 months in KCNT2 IS management."
            },
            {
                "term": "POLG / Alpers-Huttenlocher Syndrome",
                "definition": "POLG1 encodes mitochondrial DNA polymerase gamma. Biallelic POLG mutations → mtDNA depletion → Alpers syndrome (epilepsia partialis continua + liver failure + neurodegeneration). VPA induces fatal hepatopathy in POLG carriers. KCNT2 patients need POLG1 testing before ANY VPA initiation. CPIC POLG 2023 guideline: avoid VPA if POLG1 pathogenic variant identified."
            },
            {
                "term": "VPPP — Valproate Pregnancy Prevention Programme",
                "definition": "MHRA 2021 mandatory UK programme for all females of childbearing potential on valproate. Requires annual specialist review confirming pregnancy prevention (documented), patient card signed, and pharmacist dispensing safeguards. VPA is a major human teratogen (neural tube defects, autism, cognitive impairment). Risk of structural anomaly ~10%; neurodevelopmental ~30–40%."
            },
            {
                "term": "SUDEP (Sudden Unexpected Death in Epilepsy)",
                "definition": "Death in epilepsy patient without identified cause, excluding status epilepticus. KCNT2 DRE rate 58%; nocturnal SUDEP risk HIGH. Risk factors: nocturnal GTCS, sleeping alone, inadequate supervision. Mitigation: bedside SpO2 monitor, seizure alarm, prone positioning avoidance, SUDEP counselling (annual from diagnosis). Lifetime SUDEP risk in DEE estimated ~5–10×."
            },
            {
                "term": "ACMG-AMP Variant Classification (2015)",
                "definition": "Standards for variant interpretation (PMID 25741868): Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign — using PVS1, PS1–PS4, PM1–PM6, PP1–PP5, BA1, BS1–BS4, BP1–BP7 criteria. KCNT2 GOF variants: PVS1 (if nonsense) / PS2 (de novo confirmed) / PM2 (rare gnomAD) / PP3 (computational) / PP4 (phenotype specificity). Multiple recurrent de novo PS2×2 → Pathogenic classification."
            }
        ],
        "thresholds": [
            {"threshold": "ACTH response window", "value": "14 days", "note": "Evaluate hypsarrhythmia resolution at 14d; if no EEG response, step up to KD"},
            {"threshold": "BP during ACTH (infant)", "value": ">100/70 mmHg", "note": "Treat with nifedipine if sustained; pause ACTH if severe hypertensive crisis"},
            {"threshold": "Fasting glucose (ACTH)", "value": ">11.1 mmol/L (200 mg/dL)", "note": "Pause ACTH; endocrinology review; insulin if persistent hyperglycaemia"},
            {"threshold": "VPA TDM target", "value": "50–100 mg/L", "note": "Above 100 mg/L: increased hepatotoxicity risk; check LFT/ammonia"},
            {"threshold": "VGB duration limit", "value": "≤6 months (IS phase)", "note": "VFD risk cumulative; transition to LEV/CLB after IS controlled"},
            {"threshold": "Goldman perimetry interval", "value": "Every 3 months on VGB", "note": "ERG in pre-verbal children; immediate VGB cessation if VFD confirmed"},
            {"threshold": "KD therapeutic ketosis", "value": "β-hydroxybutyrate ≥2.0 mmol/L", "note": "Urine ketones 3+ equivalent; adjust ratio if below threshold"},
            {"threshold": "Ammonia (VPA)", "value": ">80 µmol/L symptomatic", "note": "VPA-induced hyperammonaemia; consider L-carnitine; pause VPA if encephalopathic"},
            {"threshold": "Developmental regression trigger", "value": "Loss of ≥1 milestone or ≥20% DQ decline", "note": "Prompt VEEG; consider sub-clinical status epilepticus; adjust AEDs"},
            {"threshold": "GTCS nocturnal rescue", "value": "Cluster ≥3 seizures / 24h", "note": "Activate emergency care plan; consider diazepam rectal / buccal midazolam rescue"},
            {"threshold": "KD cholesterol monitoring", "value": "LDL >3.4 mmol/L (130 mg/dL)", "note": "Dietitian KD ratio adjustment; fasting lipids q3M"},
            {"threshold": "SUDEP risk stratification", "value": "≥5 GTCS/year + nocturnal", "note": "High-risk SUDEP category; immediate supervision plan + alarm device"}
        ],
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
