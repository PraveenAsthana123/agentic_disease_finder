"""
CACNA1C Epilepsy — Timothy Syndrome / DEE + LQTS8 / Cav1.2 L-type HVA Ca²⁺ Channel / GOF / 12p13.33
======================================================================================================
40-patient cohort · CACNA1C (12p13.33) · Cav1.2 L-type Ca²⁺ Channel · AD GOF de novo
OMIM #601005 Timothy Syndrome — GOF G406R (TS1) / G402S (TS2-splice-variant) / non-syndromic DEE

KEY CACNA1C BIOLOGY — L-TYPE HIGH-VOLTAGE-ACTIVATED Ca²⁺ CHANNEL (Cav1.2):
CACNA1C (12p13.33) encodes Cav1.2 (α1C), the dominant L-type (dihydropyridine-sensitive)
high-voltage-activated (HVA) voltage-gated calcium channel in cardiac muscle and neurons.
Cav1 subfamily (L-type HVA):
  · Cav1.1 (CACNA1S, 1q32.1): skeletal muscle EC-coupling; malignant hyperthermia / hypoKPP2
  · Cav1.2 (CACNA1C, 12p13.33): cardiac + neuronal; Timothy Syndrome LQTS8 DEE autism
  · Cav1.3 (CACNA1D, 3p14.3): cochlear/neuronal; SANDD syndrome DEE + deafness
  · Cav1.4 (CACNA1F, Xp11.23): retinal photoreceptors; CSNB2 (congenital stationary night blindness)

KEY CLINICAL NOTES:
  1. GOF MECHANISM — IMPAIRED VOLTAGE-DEPENDENT INACTIVATION (VDI):
     - CACNA1C GOF variants (G406R TS1; G402S TS2 alternative splice; non-syndromic DEE GOF)
       → dramatically slowed / abolished VDI → prolonged window current at −40 to −20 mV
       → persistent L-type Ca²⁺ influx during action-potential plateau in cardiac cells
         → QTc prolongation (LQTS8) → 2:1 AV block → fatal arrhythmia risk
       → excess neuronal Ca²⁺ → cortical/hippocampal hyperexcitability → DEE + autism/ID
  2. BIOPHYSICS — L-TYPE HVA DISTINCT FROM T-TYPE AND R-TYPE:
     - Cav1.2 activates at −40 to −20 mV (HVA; higher threshold than T-type LVA −80 to −55 mV)
     - PHARMACOLOGY: Dihydropyridine (DHP) SENSITIVE: nifedipine/amlodipine (vasodilator DHPs)
       · Phenylalkylamine: verapamil (Cav1.2-preferring intracellular blocker)
       · Benzothiazepine: diltiazem — all three block L-type
     - CDI (calcium-dependent inactivation): CaM-IQ domain feedback — reduces pathological Ca²⁺
     - SUBUNIT: α1C (2221 aa, cardiac isoform; 2138 aa neuronal isoform) + β2/β3 + α2δ1
  3. PRECISION THERAPY — VERAPAMIL:
     - Jacobs et al. 2006 (Ann Clin Biochem): verapamil + mexiletine → QTc shortening + seizure
       reduction in TS1 (G406R) patient. Mechanism: intracellular block of Cav1.2 → reduces
       GOF Ca²⁺ → both cardiac (QTc) and neuronal benefit.
     - DOSE: 1–3 mg/kg/day oral; IV 0.1 mg/kg for acute arrhythmia
     - MONITORING: PR interval + QRS widening; avoid rapid IV in infants (bradycardia risk)
     - Roscovitine (CDK5 inhibitor): preclinical (Yarotsky 2009) — restores VDI of G406R Cav1.2;
       not yet in clinical practice
  4. CARDIAC INVOLVEMENT — MANDATORY MONITORING:
     - QTc >500 ms: immediate cardiology review · 2:1 AV block risk → pacemaker in severe TS
     - Avoid ALL QT-prolonging agents: see CONTRAINDICATIONS
     - Holter + echo 6-monthly
  5. ABSOLUTE CONTRAINDICATIONS:
     - Class Ia/III antiarrhythmics (quinidine, amiodarone, sotalol): prolong QT → fatal arrhythmia
     - TGB: NCSE risk (all epilepsies); also can prolong QT
     - VPA+POLG1: Alpers hepatotoxicity
     - OXC/CBZ/PHT: can prolong QT interval; use with cardiac monitoring only
"""

import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG — 5-CLASS GOF SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GOF-Timothy-Syndrome-Classic-TS1",
        "pct": 40,
        "mechanism": "G406R de novo in exon 8A (TS1 alternative splice exon); profound VDI abolition",
        "phenotype": "Full TS triad: DEE+West + LQTS8 (QTc >500 ms) + autism + cutaneous syndactyly 2-3",
        "eeg_pattern": "Hypsarrhythmia in infancy → multifocal with focal cortical onset; 3-5 Hz SWD components",
        "severity": "Severe DEE + potentially fatal cardiac arrhythmia (2:1 AV block)",
        "reference": "Splawski 2004 Cell 119:19 — 13 TS1 probands (G406R exon 8A); de novo GOF",
    },
    {
        "category": "GOF-Timothy-Syndrome-TS2-Splice",
        "pct": 20,
        "mechanism": "G402S in constitutively expressed exon 8 (TS2); absent in ~50% cells → mosaic GOF",
        "phenotype": "TS2: severe LQTS8 + DEE; facial dysmorphisms variable; syndactyly may be absent",
        "eeg_pattern": "West syndrome / infantile spasms; hypsarrhythmia; multifocal IEDs",
        "severity": "Severe; cardiac mortality risk from 2:1 AV block even without full syndactyly",
        "reference": "Splawski 2005 PNAS 102:18143 — TS2 G402S constitutive exon 8",
    },
    {
        "category": "GOF-DEE-Cardiac-Moderate",
        "pct": 22,
        "mechanism": "Novel CACNA1C GOF beyond G406R/G402S → partial VDI impairment; moderate QTc 460-500 ms",
        "phenotype": "DEE + moderate LQTS (QTc 460-500 ms); autism common; syndactyly absent",
        "eeg_pattern": "West syndrome infantile spasms; focal seizures; lennox-like evolution possible",
        "severity": "Moderate-severe DEE; cardiac risk lower than TS1/TS2 but monitoring mandatory",
        "reference": "Barel 2008; Gillis 2012 NEJM; non-syndromic CACNA1C GOF DEE registry",
    },
    {
        "category": "GOF-Neurodevelopmental-Only",
        "pct": 12,
        "mechanism": "CACNA1C GOF primarily affecting neuronal isoform; cardiac QTc borderline 440-460 ms",
        "phenotype": "DEE + autism/ID without classic LQTS or syndactyly; resembles CACNA1C-DEE (OMIM #618485)",
        "eeg_pattern": "Focal epilepsy of infancy; BECTS-like; multifocal; frontal predominance",
        "severity": "Moderate DEE; neurodevelopmental burden; cardiac monitoring still mandatory",
        "reference": "Scholl 2013; Damaj 2015 — CACNA1C DEE without syndromic cardiac phenotype",
    },
    {
        "category": "Phenocopy-DEE-No-CACNA1C",
        "pct": 6,
        "mechanism": "Clinical overlap with TS; no pathogenic CACNA1C variant on comprehensive panel",
        "phenotype": "DEE + cardiac QTc borderline; KCNH2/SCN5A/KCNQ1 variants found in some",
        "eeg_pattern": "Variable; West/Lennox-like; resembles TS phenotypically",
        "severity": "Moderate DEE; genetic re-analysis often needed",
        "reference": "Multi-gene LQTS panel + epilepsy panel recommended in suspected TS without CACNA1C",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES — 5-TYPE PROFILE
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Epileptic-Spasms (West Syndrome / IS)",
        "pct": 75,
        "onset_age": "2–8 months (peak 3–6 months in TS1; up to 12M in non-syndromic)",
        "eeg": "Hypsarrhythmia (modified in TS); burst-suppression in neonatal cases; multifocal high-amplitude IEDs",
        "semiology": "Clusters of flexion/extension spasms; crying post-cluster; eye deviation; hypotonia",
        "clinical_tip": "ACTH Level A per UKISS 2004 (73% IS freedom). Cardiac monitoring MANDATORY before ACTH (BP + QTc). Prednisolone Level A oral alternative.",
    },
    {
        "type": "Focal Impaired Awareness",
        "pct": 58,
        "onset_age": "Any age post-infancy; predominates after West syndrome resolution",
        "eeg": "Temporal/frontal-onset focal discharges; secondary generalisation common",
        "semiology": "Behavioural arrest, staring, oro-alimentary automatisms; focal motor component",
        "clinical_tip": "Respond partially to LEV/VPA. Consider cortical mapping if DRE — focal cortical dysplasia possible in Cav1.2 GOF.",
    },
    {
        "type": "Tonic",
        "pct": 52,
        "onset_age": "Childhood (2–8 years) especially nocturnal; Lennox-Gastaut-like evolution",
        "eeg": "EEG flattening + low-amplitude fast activity (LAFA) during tonic; interictal slow spike-wave",
        "semiology": "Nocturnal tonic stiffening; falls risk; apnoea during prolonged events",
        "clinical_tip": "KD Level B high priority. VNS adjunct. AVOID CBZ/OXC (QT-prolonging risk + seizure aggravation risk in some GGE-like features).",
    },
    {
        "type": "Myoclonic",
        "pct": 35,
        "onset_age": "Infancy–early childhood; often post-West; myoclonic-tonic common",
        "eeg": "Polyspike-wave 2-5 Hz; photosensitivity in ~28%",
        "semiology": "Sudden axial myoclonus; drop attacks; myoclonic-tonic sequences",
        "clinical_tip": "LEV or CLB adjunct. Monitor for myoclonic-tonic evolution (Lennox-Gastaut). KD evidence for myoclonic.",
    },
    {
        "type": "Absence-like / Atypical Absence",
        "pct": 20,
        "onset_age": "2–8 years; post-West evolution; often atypical (variable duration, incomplete)",
        "eeg": "Irregular 2-2.5 Hz slow spike-wave (atypical); may generalise but slower than classical 3-Hz CAE",
        "semiology": "Staring with incomplete awareness impairment; postictal fatigue; motor components common",
        "clinical_tip": "ETX may help if 3-Hz component. VPA Level B. KD evidence for atypical absence.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS — 8 TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever / Febrile Illness",
        "pct": 88,
        "mechanism": "Fever → Cav1.2 GOF channel kinetics worsen at elevated temperature → greater VDI impairment → increased window current → both seizure threshold lowered AND QTc further prolonged",
        "management": "Antipyretics PROMPTLY (paracetamol); avoid aspirin (cardiac risk). ECG during febrile illness. ICU-level monitoring in TS1/TS2 with fever >38.5°C.",
    },
    {
        "trigger": "Infection",
        "pct": 80,
        "mechanism": "Systemic inflammation + fever + metabolic stress → cardiac arrhythmia risk during infection",
        "management": "Hospital admission in TS1/TS2 for any significant infection; continuous cardiac monitoring.",
    },
    {
        "trigger": "Sleep Deprivation",
        "pct": 72,
        "mechanism": "Sleep deprivation → increased cortical excitability; also triggers cardiac arrhythmia in LQTS via adrenergic surge",
        "management": "Strict sleep hygiene; melatonin adjunct evidence for DEE + autism co-management.",
    },
    {
        "trigger": "Missed AED / Verapamil Dose",
        "pct": 65,
        "mechanism": "Missed L-type blocker (verapamil) → rebound increase in Cav1.2 window current → seizure cluster AND QTc prolongation spike",
        "management": "Strict medication adherence; caregiver-administered rescue CLB buccal/nasal for clusters; immediate ECG if missed verapamil.",
    },
    {
        "trigger": "Stress / Emotional Upset",
        "pct": 58,
        "mechanism": "Adrenergic surge → PKA phosphorylation of Cav1.2 → increased L-type current → sensitises GOF channel further; cardiac sympathetic drive worsens LQTS",
        "management": "Calm environment; beta-blocker (propranolol) as adjunct cardiac/seizure management evidence in LQTS.",
    },
    {
        "trigger": "QT-Prolonging Drug Exposure",
        "pct": 38,
        "mechanism": "Any drug prolonging QTc (see CONTRAINDICATIONS) → additive to baseline LQTS8 → potentially fatal 2:1 AV block or VF",
        "management": "Comprehensive drug review before ANY new prescription. CredibleMeds check MANDATORY. Avoid PHT/CBZ unless absolutely necessary with continuous cardiac monitoring.",
    },
    {
        "trigger": "AED Taper / Rapid Dose Change",
        "pct": 42,
        "mechanism": "Rapid AED taper → seizure withdrawal phenomenon in DEE; particularly dangerous in TS with concurrent cardiac instability",
        "management": "VERY SLOW tapers (>4 weeks minimum); never abrupt in TS. Cardiac monitoring during taper.",
    },
    {
        "trigger": "Startle / Loud Noise",
        "pct": 22,
        "mechanism": "Startle response → sympathetic discharge → adrenergic LQTS trigger; startle-epilepsy possible in frontal seizures",
        "management": "Quiet environment; benzodiazepine rescue for startle-triggered clusters.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS — 8 AGENTS
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Verapamil (Calan / Isoptin) — L-type Cav1.2 Blocker (CACNA1C Precision)",
        "level": "Level B — case series (Jacobs 2006; Napolitano 2010; n<15)",
        "moa": "Phenylalkylamine intracellular L-type Ca²⁺ channel blocker; preferentially blocks Cav1.2 open/inactivated state → reduces GOF Ca²⁺ influx → dual cardiac (QTc shortening) + neuronal (seizure reduction) benefit",
        "dose": "1–3 mg/kg/day oral BID/TID (cardiac dose 2-6 mg/kg/day); IV 0.1 mg/kg over 5 min (acute arrhythmia; AVOID rapid IV in infants <1 year → profound bradycardia)",
        "efficacy": "QTc reduction 40-80 ms reported; seizure frequency reduction ~40-60% in case series; combined verapamil+mexiletine (Jacobs 2006) most documented",
        "safety": "Negative chronotrope/inotrope → monitor PR interval + heart rate; constipation; grapefruit interaction (CYP3A4); avoid in severe LV dysfunction",
        "monitoring": "ECG (PR interval, QRS) before initiation and after dose increase; avoid concomitant beta-blockers in infants",
        "cacna1c_note": "ONLY agent with direct Cav1.2 blocking MOA; mechanistically ideal precision therapy. Initiate under paediatric cardiology co-management.",
    },
    {
        "drug": "ACTH (Adrenocorticotrophic Hormone / Acthar Gel)",
        "level": "Level A — West syndrome / Infantile Spasms (UKISS 2004; United Kingdom Infantile Spasms Study; n=107 RCT)",
        "moa": "ACTH → MC4R-mediated suppression of hypothalamic CRH → reduced cortical excitability → hypsarrhythmia resolution; independent of aetiology. In CACNA1C: addresses IS regardless of L-type mechanism.",
        "dose": "40-60 IU/day IM for 2 weeks then taper; synthetic ACTH (tetracosactide) 0.5-1.5 mg alternate days UK; Acthar Gel 150 IU/m²/day USA (max 4 weeks)",
        "efficacy": "73% IS freedom at 14 days (UKISS 2004); superior to VGB for non-TSC aetiology. EEG hypsarrhythmia resolution correlates with IS cessation.",
        "safety": "Hypertension (MANDATORY BP q2 weeks); hyperglycaemia; infection susceptibility; electrolyte imbalance; cardiac monitoring — QTc may change with ACTH-driven adrenergic effects",
        "monitoring": "BP q2 weeks; glucose q2 weeks; electrolytes; WEIGHT; CARDIAC ECG mandatory in CACNA1C (LQTS8 + ACTH adrenergic risk = heightened cardiac monitoring)",
        "cacna1c_note": "First-line for West/IS regardless of CACNA1C; HEIGHTENED cardiac monitoring vs standard IS protocol given baseline LQTS8 risk.",
    },
    {
        "drug": "Prednisolone (Oral Corticosteroid)",
        "level": "Level A — IS (UKISS 2004 oral arm; United Kingdom Infantile Spasms Study; non-inferior to ACTH in combined analysis)",
        "moa": "Glucocorticoid → anti-inflammatory + MC4R-independent cortical excitability suppression; oral alternative to ACTH injection",
        "dose": "4 mg/kg/day oral (max 60 mg/day) for 2 weeks then 2-week taper",
        "efficacy": "76% IS freedom at 14 days (UKISS); similar to ACTH in combined analysis. Preferred where injection not feasible.",
        "safety": "Similar to ACTH: hypertension, hyperglycaemia, behavioural change, immunosuppression; QTc monitoring in CACNA1C",
        "monitoring": "BP; glucose; weight; ECG in CACNA1C (same heightened cardiac protocol)",
        "cacna1c_note": "Oral alternative to ACTH for IS; maintain cardiac surveillance throughout corticosteroid course.",
    },
    {
        "drug": "Vigabatrin / VGB (GABA-T Inhibitor)",
        "level": "Level A — IS (SHARE REMS; FDA-approved infantile spasms; particularly high efficacy in TSC-aetiology [84%]; CACNA1C: used but less TSC-specific benefit)",
        "moa": "Irreversible GABA-transaminase inhibitor → ↑ synaptic GABA → anticonvulsant; reduces GABA degradation in hippocampus/cortex",
        "dose": "100-150 mg/kg/day oral in 2 divided doses (IS); max 3000 mg/day",
        "efficacy": "53-65% IS freedom in non-TSC; inferior to ACTH in non-TSC but additive combination (COMBO-IS). Focal epilepsy adjunct.",
        "safety": "VISUAL FIELD DEFECT (VFD) 30-50% with >2 years exposure — IRREVERSIBLE peripheral constriction; MRI T2 changes in infants (reversible); sedation",
        "monitoring": "ERG q3M (SHARE REMS mandatory for FDA); ophthalmology annual; duration limit <2 years preferred in CACNA1C unless seizure control demands continuation",
        "cacna1c_note": "HIGH RISK: VFD with prolonged use. ERG every 3 months mandatory (SHARE REMS). Preferred shorter duration; transition to alternative if seizure freedom achieved.",
    },
    {
        "drug": "Valproate / VPA (Broad-Spectrum)",
        "level": "Level B — broad DEE evidence; POLG screening MANDATORY",
        "moa": "GABA augmentation + INaP reduction + T-type Ca²⁺ block (modest) + HDAC inhibition; broad-spectrum",
        "dose": "20-60 mg/kg/day oral (serum TDM 50-100 μg/mL; some DEE 100-120 μg/mL); slow-release preferred",
        "efficacy": "Moderate IS/tonic/myoclonic/focal seizure control; monotherapy or combination",
        "safety": "POLG1 screening mandatory; teratogenicity (VPPP MHRA 2021 — MANDATORY for females of reproductive potential); weight gain; hepatotoxicity; thrombocytopaenia; hyperammonaemia",
        "monitoring": "POLG1 BEFORE prescribing; LFT + FBC + ammonia q3M; VPA TDM q3M; VPPP mandatory for females; QTc — VPA has mild QT effects, monitor in LQTS8 context",
        "cacna1c_note": "Use with caution in CACNA1C given baseline LQTS8 — VPA has mild QTc-shortening effect (potentially beneficial) but unpredictable; POLG1 mandatory.",
    },
    {
        "drug": "Levetiracetam / LEV (SV2A Ligand)",
        "level": "Level B — adjunct DEE/IS evidence; POLG-safe (no mitochondrial toxicity)",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) binding → modulates vesicle priming → reduces glutamate/GABA exocytosis; exact mechanism not fully characterised",
        "dose": "20-60 mg/kg/day oral or IV BID (IV in clusters); titrate 2-week intervals",
        "efficacy": "Focal + generalised seizure adjunct; IS post-hormonal therapy maintenance; good tolerability",
        "safety": "Behavioural side effects (irritability, aggression) — especially in ID/autism (dose-reduce or switch); no hepatotoxicity; no QTc effect",
        "monitoring": "Behavioural monitoring (CACNA1C autism comorbidity heightens LEV behavioural risk); renal dosing in reduced GFR",
        "cacna1c_note": "PREFERRED adjunct in CACNA1C — POLG-safe, no significant QTc effect, no CYP interactions (avoids verapamil pharmacokinetic complications). Watch behavioural side effects given autism.",
    },
    {
        "drug": "Ketogenic Diet / KD (High-Fat Low-Carbohydrate)",
        "level": "Level B — DRE ≥2 AED failures; ILAE Diet Therapies Task Force 2018",
        "moa": "Beta-hydroxybutyrate (BHB) → multiple mechanisms: mitochondrial ROS reduction; ATP-sensitive K+ channel (KATP) opening → membrane hyperpolarisation; GABA upregulation; acetoacetate adenosine receptor activation; potential Cav1 window-current modulation via membrane cholesterol",
        "dose": "3:1 or 4:1 (fat:protein+carb) ratio; modified Atkins 0.5-1 g/kg/day carb; RD-supervised initiation",
        "efficacy": "50% seizure reduction in 50-55% of DRE DEE; IS post-ACTH maintenance; Lennox-Gastaut evidence Level B",
        "safety": "Kidney stones (7%); dyslipidaemia; growth effects; constipation; selenium/zinc deficiency; cardiomyopathy risk in selenium-deficient KD (monitor in CACNA1C cardiac background)",
        "monitoring": "Lipid panel q6M; renal USS annually; selenium/zinc q6M; ECHOCARDIOGRAM annually (CACNA1C + KD cardiomyopathy risk); dietitian monthly initially",
        "cacna1c_note": "HIGH PRIORITY in CACNA1C DRE ≥2 AED failures. MANDATORY cardiac echo annually on KD — selenium-deficient KD cardiomyopathy could compound LQTS8 cardiac risk.",
    },
    {
        "drug": "Clobazam / CLB (1,5-Benzodiazepine)",
        "level": "Level B — adjunct DEE/LGS/focal evidence; Lennox-Gastaut FDA-approved",
        "moa": "Positive allosteric GABA-A receptor modulator (BZD-binding site); α2 subunit preferential; anxiolytic + anticonvulsant",
        "dose": "0.1-0.3 mg/kg/day oral BID; max 40 mg/day (FDA LGS approval up to 40 mg)",
        "efficacy": "Drop attack / tonic-clonic adjunct; nocturnal tonic seizures; IS cluster-break adjunct",
        "safety": "Sedation; tolerance (4-6 months); withdrawal risk — taper slowly; respiratory depression in combination",
        "monitoring": "Sedation assessment; tolerance review q6M; behavioural co-management in CACNA1C autism; no significant QTc effect",
        "cacna1c_note": "Useful nocturnal tonic seizure adjunct. CLB has minimal QTc effect — safer adjunct than many alternatives in LQTS8 background.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS — 5 CI WITH CACNA1C-SPECIFIC RATIONALE
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Class Ia/III Antiarrhythmics (Quinidine / Amiodarone / Sotalol / Dofetilide)",
        "risk": "ABSOLUTE CI — QT LETHAL",
        "mechanism": "All directly block hERG (KCNH2) IKr current → additive to LQTS8 (hERG + CACNA1C GOF dual QTc prolongation) → risk of torsades de pointes + 2:1 AV block → sudden cardiac death",
        "cacna1c_note": "ANY antiarrhythmic prescribed for incidental cardiac arrhythmia must be verified against CredibleMeds LQTS risk category. Beta-blockers (propranolol/nadolol) preferred LQTS cardiac management.",
        "alternative": "Beta-blockers (propranolol 1-2 mg/kg/day) for LQTS8 rate control; verapamil for L-type precision; pacemaker for severe 2:1 AV block",
    },
    {
        "drug": "TGB — Tiagabine (GABA Reuptake Inhibitor)",
        "risk": "ABSOLUTE CI — NCSE",
        "mechanism": "TGB → excess synaptic GABA → paradoxical absence-like NCSE in focal epilepsy/GGE backgrounds; also reports of QTc prolongation in TGB case series",
        "cacna1c_note": "TGB is ABSOLUTE CI in all CACNA1C epilepsy: (1) NCSE risk; (2) possible QTc effect (double cardiac risk in LQTS8).",
        "alternative": "CLB / LEV / KD for seizure control",
    },
    {
        "drug": "VPA + POLG1 Variant (Valproate + Mitochondrial Polymerase Gamma Deficiency)",
        "risk": "ABSOLUTE CI — ALPERS HEPATOTOXICITY",
        "mechanism": "POLG1 pathogenic variant (AR) → mitochondrial DNA depletion syndrome (Alpers disease). VPA in POLG1 → fulminant hepatic failure + status epilepticus. POLG1 DEE may clinically resemble CACNA1C DEE → POLG1 MUST BE EXCLUDED before any VPA.",
        "cacna1c_note": "POLG1 screening MANDATORY before VPA in any DEE presentation. CACNA1C + VPA = acceptable IF POLG1 negative. CACNA1C + VPA + undetected POLG1 = LETHAL.",
        "alternative": "LEV (POLG-safe) as preferred adjunct if POLG status unknown",
    },
    {
        "drug": "CBZ / OXC / PHT — Sodium Channel Blockers (in LQTS8 background)",
        "risk": "HIGH RISK — QT + CARDIAC CONDUCTION EFFECTS",
        "mechanism": "Phenytoin: QTc shortening at therapeutic but QRS widening / AV block in toxicity. Carbamazepine: cardiac conduction slowing (AV block, sinus bradycardia); OXC similar. All interact with verapamin via CYP3A4 enzyme induction → reduce verapamil levels → lose precision cardiac/seizure benefit",
        "cacna1c_note": "CBZ/OXC/PHT markedly REDUCE verapamil plasma levels via CYP3A4 induction → loss of Cav1.2 precision therapy. Avoid unless no alternative. If used: increase verapamil dose and monitor ECG closely.",
        "alternative": "LEV / VPA(POLG-screened) / KD — avoid CYP3A4-inducing AEDs in CACNA1C on verapamil",
    },
    {
        "drug": "VGB long-term >12M (Vigabatrin)",
        "risk": "HIGH RISK — VISUAL FIELD DEFECT (VFD)",
        "mechanism": "Irreversible VFD in 30-50% with >2 years exposure; MRI T2 basal ganglia changes in infants (usually reversible). In CACNA1C autism, VFD adds significant quality-of-life burden to already-impaired child.",
        "cacna1c_note": "Use <12 months preferred; ERG mandatory q3M (SHARE REMS). If VFD detected → immediate cessation discussion with neurology + cardiology co-management.",
        "alternative": "ACTH/prednisolone for IS; KD for maintenance DRE",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING — 14 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {
        "item": "ECG (QTc + PR interval) — BASELINE + EVERY CLINIC VISIT",
        "frequency": "Every 2-4 weeks (active titration); every 3 months (stable); every clinic visit minimum",
        "rationale": "LQTS8 hallmark: QTc >500 ms → immediate cardiology; 2:1 AV block monitoring mandatory; verapamil → PR interval widening monitoring",
        "threshold": "QTc >500 ms → cardiology review same day; PR >200 ms on verapamil → dose review",
    },
    {
        "item": "Holter Monitor (24-48h Ambulatory ECG)",
        "frequency": "6-monthly (minimum); after any dose change of verapamil/cardiac agent; after febrile illness",
        "rationale": "Identifies paroxysmal arrhythmias (2:1 AV block episodes) not captured on spot ECG; nocturnal QTc pattern",
        "threshold": "2:1 AV block episodes → pacemaker evaluation; VT/VF episodes → ICU transfer",
    },
    {
        "item": "POLG1 Gene Testing (BEFORE VPA)",
        "frequency": "ONCE (pre-VPA; comprehensive mitochondrial gene panel preferred)",
        "rationale": "POLG1 DEE mimics CACNA1C DEE; VPA in POLG1 = fulminant Alpers hepatic failure. Non-negotiable prior to any VPA prescription.",
        "threshold": "POLG1 pathogenic variant found → VPA ABSOLUTELY CONTRAINDICATED",
    },
    {
        "item": "ACTH / Prednisolone Monitoring (BP + Glucose)",
        "frequency": "Every 2 weeks during IS treatment; weekly in neonatal cases",
        "rationale": "Corticosteroid hypertension + hyperglycaemia risk; LQTS8 patients may have exaggerated adrenergic response to corticosteroid hypertension",
        "threshold": "BP >95th centile → dose review; glucose >12 mmol/L → endocrinology",
    },
    {
        "item": "VGB ERG (Electroretinogram) — SHARE REMS Mandatory",
        "frequency": "Every 3 months (MANDATORY per SHARE REMS protocol for USA; every 6M minimum elsewhere)",
        "rationale": "VFD 30-50% with long-term VGB; autism in CACNA1C makes self-report unreliable → ERG is objective",
        "threshold": "ERG amplitude reduction >30% vs baseline → immediate VGB cessation review with neurology",
    },
    {
        "item": "Echocardiogram (Cardiac Morphology + Function)",
        "frequency": "Baseline; 6-monthly in first 2 years; annually thereafter; ANNUALLY if on KD",
        "rationale": "Cav1.2 GOF → potential cardiomyopathy risk; KD selenium-deficiency cardiomyopathy risk independent of LQTS8",
        "threshold": "Ejection fraction <50% → cardiology; new cardiomyopathy → suspend KD; verapamil dose review",
    },
    {
        "item": "VPA TDM (Therapeutic Drug Monitoring)",
        "frequency": "Every 3 months (stable); 2 weeks after any dose change",
        "rationale": "VPA narrow therapeutic range; toxicity risk in DEE; ammonia monitoring in parallel",
        "threshold": "VPA >120 μg/mL + ammonia >60 μmol/L → dose reduce; LFT 3× ULN → consider discontinuation",
    },
    {
        "item": "LFT + FBC + Ammonia (VPA Safety)",
        "frequency": "Every 3 months (on VPA); monthly in first 6 months",
        "rationale": "VPA hepatotoxicity monitoring; FBC thrombocytopaenia; hyperammonaemia encephalopathy risk",
        "threshold": "LFT ALT/AST >3× ULN → POLG1 re-check + discontinue; ammonia >80 → VPA hold",
    },
    {
        "item": "EEG (Baseline + Annual Review)",
        "frequency": "Baseline; every 6-12 months (active DEE); after major seizure change",
        "rationale": "Track evolution from hypsarrhythmia → multifocal → possible LGS pattern; NCSE surveillance (especially if TGB was ever tried)",
        "threshold": "NCSE on EEG → IV BZD; hypsarrhythmia resolution confirms IS response to ACTH",
    },
    {
        "item": "Cognitive / Developmental Assessment",
        "frequency": "Every 6 months (structured; Bayley / Griffiths / VABS-III)",
        "rationale": "DEE trajectory monitoring; autism severity (ADOS); communication aids; educational planning",
        "threshold": "Developmental regression → MRI + metabolic review + epilepsy re-assessment",
    },
    {
        "item": "MRI Brain (Baseline + As Indicated)",
        "frequency": "Baseline (sedation with cardiac monitoring in CACNA1C); repeat if regression/seizure change",
        "rationale": "Cortical malformation (FCD) underlying focal seizures; Leigh-like patterns to exclude mitochondrial overlap",
        "threshold": "FCD or structural lesion → epilepsy surgery workup; Leigh-like → POLG/mitochondrial re-workup",
    },
    {
        "item": "VPPP (Valproate Pregnancy Prevention Programme — MHRA 2021)",
        "frequency": "Every annual review for females of reproductive potential on VPA; at transition to adolescence",
        "rationale": "Valproate teratogenicity (neural tube, cognitive); MHRA 2021 mandatory programme in UK; applies globally as best practice",
        "threshold": "VPA in fertile female without VPPP documentation → MANDATORY enrolment or switch to alternative AED",
    },
    {
        "item": "SUDEP Risk Counselling (Annual)",
        "frequency": "Annual (at each annual review); after any seizure cluster or GTCS",
        "rationale": "SUDEP: cardiac arrhythmia is a leading mechanism — PARTICULARLY HIGH RISK in CACNA1C (LQTS8 + ictal autonomic dysregulation); nocturnal supervision; seizure alert devices",
        "threshold": "Unwitnessed nocturnal GTCS + LQTS8 → high-risk SUDEP counselling; cardiac implantable device discussion (ICD vs pacemaker in severe LQTS8)",
    },
    {
        "item": "Genetic Counselling (Family)",
        "frequency": "At diagnosis; at each reproductive planning discussion; prenatal counselling",
        "rationale": "De novo >95% in TS1/TS2; recurrence risk ~1-3% (germline mosaicism); first-degree relative cardiac screening (QTc) recommended given LQTS8 penetrance possible even without syndromic phenotype",
        "threshold": "Sibling QTc >460 ms (male) />470 ms (female) → CACNA1C sequencing; cardiac monitoring in asymptomatic siblings",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE — 6 WINDOWS
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Neonatal-Cardiac-0-4wk",
        "key_issues": "QTc prolongation at birth; 2:1 AV block possible; pacemaker evaluation in severe TS1; antiepileptic approach secondary to cardiac stabilisation; neonatal ICU co-management with cardiology + neurology",
    },
    {
        "window": "West-Syndrome-IS-2-12M",
        "key_issues": "Infantile spasms + hypsarrhythmia peak; ACTH/prednisolone Level A with HEIGHTENED cardiac monitoring; VGB Level A (SHARE REMS ERG q3M); verapamil introduction under cardiology co-management; POLG1 before VPA",
    },
    {
        "window": "Post-IS-Transition-12M-3Y",
        "key_issues": "IS resolution assessment; residual multifocal epilepsy; KD initiation if DRE; cognitive development tracking; autism assessment (ADOS); verapamil dose optimisation; cardiac monitoring continuation",
    },
    {
        "window": "Childhood-DEE-3-12Y",
        "key_issues": "Lennox-Gastaut-like evolution in some; focal epilepsy predominance; KD continuation; LEV + CLB combination; cortical mapping if DRE with structural lesion; educational support; cardiac Holter q6M",
    },
    {
        "window": "Adolescence-Transition-12-25Y",
        "key_issues": "AED transition (paediatric to adult); VPPP counselling (females on VPA); cardiac ICD evaluation in persistent severe LQTS8; psychosocial support; employment/independence planning; SUDEP counselling",
    },
    {
        "window": "Adult-Severe-DEE-25Y+",
        "key_issues": "Residential care in severe cases; polypharmacy rationalisation; cardiac device management (pacemaker/ICD); SUDEP prevention; palliative care discussion in end-stage DEE; genetic family counselling for siblings",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY CONCEPTS / DEFINITIONS — 15
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "CACNA1C (12p13.33)",
        "definition": "Calcium Voltage-Gated Channel Subunit Alpha1 C — encodes Cav1.2 (α1C), the dominant L-type HVA Ca²⁺ channel in cardiac muscle, smooth muscle, and brain. 2221 aa (cardiac isoform), 49 exons. pLI ~0.99 (highly intolerant to LOF). GOF mutations cause Timothy Syndrome; common GWAS risk locus for schizophrenia/bipolar.",
    },
    {
        "term": "Cav1.2 L-type HVA",
        "definition": "L-type (Long-lasting) high-voltage-activated Ca²⁺ channel — Cav1.2. Activates at −40 to −20 mV. Key features: dihydropyridine (DHP) sensitive; calcium-dependent inactivation (CDI) via CaM-IQ domain; slow voltage-dependent inactivation (VDI). Principal cardiac sarcolemmal Ca²⁺ channel for excitation-contraction coupling. In brain: cortical/hippocampal dendritic integration, synaptic plasticity (LTP).",
    },
    {
        "term": "Timothy Syndrome (TS)",
        "definition": "Rare multisystem disorder caused by CACNA1C GOF. TS1 (G406R, exon 8A alternative): cardiac (LQTS8, 2:1 AV block) + DEE + autism + cutaneous syndactyly 2-3 + facial dysmorphisms. TS2 (G402S, constitutive exon 8): more severe cardiac; syndactyly may be absent. Described by Timothy 2004; Splawski 2004 Cell 119:19. OMIM #601005. ~20 classic cases published worldwide.",
    },
    {
        "term": "LQTS8 (Long-QT Syndrome Type 8)",
        "definition": "Cardiac channelopathy component of Timothy Syndrome. CACNA1C GOF → L-type Ca²⁺ current prolonged into repolarisation → delayed ventricular repolarisation → QTc prolongation (typically >500 ms) → torsades de pointes (TdP) → ventricular fibrillation → sudden cardiac death. Distinct from LQTS1 (KCNQ1), LQTS2 (KCNH2), LQTS3 (SCN5A) in mechanism.",
    },
    {
        "term": "GOF Mechanism — Cav1.2 VDI Impairment",
        "definition": "CACNA1C GOF variants (G406R, G402S) → dramatically impaired voltage-dependent inactivation (VDI). Normally Cav1.2 rapidly inactivates after depolarisation. GOF → channel stays OPEN → enlarged window current at −40 to −20 mV → persistent Ca²⁺ influx during action-potential plateau (cardiac) → QTc prolongation; in neurons → excess Ca²⁺ → cortical hyperexcitability → DEE.",
    },
    {
        "term": "VDI (Voltage-Dependent Inactivation)",
        "definition": "Mechanism by which voltage-gated Ca²⁺ channels close during sustained depolarisation regardless of intracellular Ca²⁺. Mediated by the IS6 segment hydrophobic residues (I-F-M motif) + β-subunit auxiliary. CACNA1C G406R/G402S abolish VDI → channel remains constitutively open at depolarised potentials → pathological Ca²⁺ entry.",
    },
    {
        "term": "CDI (Calcium-Dependent Inactivation)",
        "definition": "Secondary inactivation mechanism of Cav1.2 driven by rising intracellular Ca²⁺ binding to calmodulin (CaM) at the IQ domain of the C-terminus. CDI provides negative feedback limiting Cav1.2 activity. In GOF CACNA1C: CDI is partially preserved but VDI abolition dominates → net persistent Ca²⁺ overload despite CDI.",
    },
    {
        "term": "DHP (Dihydropyridine) Sensitivity",
        "definition": "L-type hallmark: Cav1.1/1.2/1.3/1.4 are all blocked by dihydropyridines (nifedipine, amlodipine, felodipine) — binding to domain III S5-S6. DHPs are vasodilator-class L-type blockers. Verapamil (phenylalkylamine) and diltiazem (benzothiazepine) also block Cav1.2 via different sites. DHP sensitivity distinguishes L-type from T-type (insensitive) and R-type (insensitive to nimodipine) channels.",
    },
    {
        "term": "Verapamil Precision (CACNA1C)",
        "definition": "Verapamil: phenylalkylamine class L-type Ca²⁺ channel blocker. Binds Cav1.2 intracellular DIVA site (domain IV S6). Reduces GOF Ca²⁺ influx → dual benefit: cardiac QTc shortening (Jacobs 2006) + seizure frequency reduction. PRECISION rationale: directly blocks the gain-of-function channel. Dose: 1-3 mg/kg/day. Monitoring: PR interval, HR. AVOID rapid IV in infants. CYP3A4 inhibition — avoid CYP3A4-inducing AEDs (CBZ/OXC/PHT).",
    },
    {
        "term": "West Syndrome / IS",
        "definition": "Epilepsy syndrome of infantile spasms + hypsarrhythmia (chaotic high-amplitude EEG) + developmental arrest/regression. Peak onset 3-8 months. Multiple aetiologies including CACNA1C GOF. ACTH/prednisolone Level A evidence (UKISS 2004: 73% IS-freedom). Vigabatrin Level A (SHARE REMS — ERG q3M). EEG-driven diagnosis: hypsarrhythmia must resolve with treatment.",
    },
    {
        "term": "VPPP (Valproate Pregnancy Prevention Programme — MHRA 2021)",
        "definition": "MHRA UK mandatory programme for all females of reproductive potential on valproate. Background: VPA causes neural tube defects (1-2%), cognitive impairment in offspring (~30-40 IQ points), autism risk. VPPP requires: annual specialist review; effective contraception documented; patient information card; form signatures. Applies to CACNA1C patients on VPA globally as best practice regardless of jurisdiction.",
    },
    {
        "term": "POLG1 / Alpers Syndrome",
        "definition": "POLG1 (mitochondrial DNA polymerase gamma): biallelic pathogenic variants cause Alpers-Huttenlocher syndrome — progressive mitochondrial encephalopathy + liver failure + intractable seizures. VPA in POLG1 mutation → fulminant hepatic failure. POLG1-DEE mimics CACNA1C-DEE. MANDATORY POLG1 testing before VPA in any DEE. CPIC guideline 2023: POLG1 screening required before VPA.",
    },
    {
        "term": "Syndactyly (Timothy Syndrome Cutaneous)",
        "definition": "Cutaneous (soft tissue) syndactyly of digits 2-3 and/or 2-3-4 of hands and feet is PATHOGNOMONIC for TS1. Bony structure normal (unlike genetic skeletal dysplasias). Present at birth — prenatal ultrasound can detect. Correctable surgically. Absence in TS2 and non-syndromic CACNA1C DEE variants makes clinical diagnosis harder — molecular testing essential.",
    },
    {
        "term": "SUDEP (Sudden Unexpected Death in Epilepsy)",
        "definition": "Sudden death in a person with epilepsy without identifiable cause. Mechanism: cardiac (ictal bradycardia/asystole), respiratory (post-ictal apnoea), autonomic dysregulation. IN CACNA1C: SUDEP risk is PARTICULARLY HIGH due to dual cardiac (LQTS8 arrhythmia risk) + epilepsy (ictal autonomic dysregulation). ICD (implantable cardioverter-defibrillator) discussion mandatory in severe LQTS8 with QTc >520 ms or prior arrhythmia events.",
    },
    {
        "term": "ACMG/AMP 2015 Variant Classification",
        "definition": "American College of Medical Genetics and Genomics + Association for Molecular Pathology 2015 classification. 5-tier: Pathogenic (P) / Likely Pathogenic (LP) / Variant of Uncertain Significance (VUS) / Likely Benign (LB) / Benign (B). CACNA1C G406R = Pathogenic. Novel CACNA1C variants in DEE + functional assay (in vitro VDI impairment) = LP/P. VUS variants require functional data before precision verapamil therapy.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS — 12 CLINICAL DECISION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"param": "QTc (QTcF Fridericia)", "action_threshold": ">500 ms → same-day cardiology review; >520 ms + symptoms → ICD evaluation", "unit": "ms"},
    {"param": "PR Interval on Verapamil", "action_threshold": ">200 ms → verapamil dose reduction; >250 ms → cardiology review", "unit": "ms"},
    {"param": "VPA Serum Level (TDM)", "action_threshold": "Target 50-100 μg/mL; >120 μg/mL with toxicity → dose reduce", "unit": "μg/mL"},
    {"param": "ETX Serum Level (if used for atypical absence)", "action_threshold": "Target 40-100 μg/mL; toxicity (GI, hiccup) if >100", "unit": "μg/mL"},
    {"param": "VGB ERG Amplitude", "action_threshold": ">30% reduction vs baseline → immediate VGB cessation discussion", "unit": "μV amplitude ratio"},
    {"param": "ALT/AST (Liver Enzymes on VPA)", "action_threshold": ">3× ULN → re-check POLG1; consider VPA discontinuation", "unit": "× ULN"},
    {"param": "Serum Ammonia", "action_threshold": ">80 μmol/L + encephalopathy → VPA hold; consider L-carnitine supplementation", "unit": "μmol/L"},
    {"param": "Blood Glucose (on ACTH/prednisolone)", "action_threshold": ">12 mmol/L → endocrinology review; insulin if persistent", "unit": "mmol/L"},
    {"param": "Blood Pressure (on ACTH/prednisolone)", "action_threshold": ">95th centile for age/height → antihypertensive; dose review", "unit": "centile"},
    {"param": "Echocardiogram EF (KD)", "action_threshold": "Ejection fraction <50% on KD → selenium check + KD suspension", "unit": "% (EF)"},
    {"param": "Sibling QTc (Family Screening)", "action_threshold": "Male sibling QTc >460 ms; female >470 ms → CACNA1C sequencing", "unit": "ms"},
    {"param": "Fever Threshold (Cardiac Alert)", "action_threshold": ">38.5°C → hospital admission in TS1/TS2 for continuous cardiac monitoring", "unit": "°C"},
]

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE STANDARDS — 12
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022", "title": "International League Against Epilepsy Operational Classification and Terminology 2022"},
    {"code": "NICE-NG217", "title": "NICE NG217 — Epilepsies: diagnosis and management (UK, 2022)"},
    {"code": "Splawski-2004-Cell", "title": "Splawski I et al. 2004 Cell 119:19 — CACNA1C G406R mutations cause Timothy Syndrome (TS1, 13 probands)"},
    {"code": "Splawski-2005-PNAS", "title": "Splawski I et al. 2005 PNAS 102:18143 — Timothy Syndrome Type 2 (TS2, G402S constitutive exon 8)"},
    {"code": "Jacobs-2006-Verapamil", "title": "Jacobs A et al. 2006 Ann Clin Biochem 43:475 — Verapamil+mexiletine in Timothy Syndrome (QTc shortening + seizure reduction)"},
    {"code": "UKISS-2004", "title": "Lux AL et al. 2004 Lancet 364:2075 — UKISS RCT (n=107): ACTH 73% IS-freedom; prednisolone equivalent"},
    {"code": "CPIC-POLG-2023", "title": "CPIC Guideline 2023 — POLG variants and valproic acid: mandatory POLG1 screening before VPA in any DEE"},
    {"code": "MHRA-VPPP-2021", "title": "MHRA 2021 — Valproate Pregnancy Prevention Programme (VPPP): mandatory annual review + contraception + patient card for fertile females"},
    {"code": "SHARE-REMS-VGB", "title": "FDA SHARE REMS — Vigabatrin Risk Evaluation and Mitigation Strategy: ERG q3M mandatory; cumulative dose/duration tracking"},
    {"code": "ACMG-AMP-2015", "title": "Richards S et al. 2015 Genet Med 17:405 — ACMG/AMP 5-tier variant classification (P/LP/VUS/LB/B)"},
    {"code": "ILAE-Diet-2018", "title": "ILAE Dietary Therapies Task Force 2018 — Ketogenic diet position statement: DRE ≥2 AED failures"},
    {"code": "WHO-ICF-2019", "title": "WHO ICF (International Classification of Functioning, Disability and Health) 2019 — multidimensional DEE outcome framework"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES — 6 KEY PAPERS
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"key": "Splawski-2004", "citation": "Splawski I et al. (2004) CACNA1C mutations disrupt Cav1.2 inactivation and cause Timothy syndrome. Cell 119(1):19-31."},
    {"key": "Splawski-2005", "citation": "Splawski I et al. (2005) Severe arrhythmia disorder caused by cardiac L-type calcium channel mutations. PNAS 102(23):8089-96."},
    {"key": "Jacobs-2006", "citation": "Jacobs A et al. (2006) Successful treatment of severe arrhythmia associated with Timothy syndrome. Ann Clin Biochem 43:475-478."},
    {"key": "Napolitano-2010", "citation": "Napolitano C et al. (2010) Genetics of ventricular arrhythmias and sudden death. European Heart Journal 31:1945-1952."},
    {"key": "ILAE-2022", "citation": "ILAE Classification and Terminology 2022. Epilepsia 63(6):1233-1254."},
    {"key": "NICE-NG217", "citation": "NICE NG217. Epilepsies: diagnosis and management. London: NICE, 2022."},
]

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC PATIENT COHORT — 40 PATIENTS
# ─────────────────────────────────────────────────────────────────────────────
_ETIOLOGY_POOL = [
    ("GOF-Timothy-Syndrome-Classic-TS1", 16),
    ("GOF-Timothy-Syndrome-TS2-Splice", 8),
    ("GOF-DEE-Cardiac-Moderate", 9),
    ("GOF-Neurodevelopmental-Only", 5),
    ("Phenocopy-DEE-No-CACNA1C", 2),
]

_cohort = []
_pid = 1
for etio, count in _ETIOLOGY_POOL:
    for _ in range(count):
        is_ts1 = etio == "GOF-Timothy-Syndrome-Classic-TS1"
        is_ts2 = etio == "GOF-Timothy-Syndrome-TS2-Splice"
        is_ts = is_ts1 or is_ts2
        is_moderate = etio == "GOF-DEE-Cardiac-Moderate"
        is_neuroonly = etio == "GOF-Neurodevelopmental-Only"
        is_phenocopy = etio == "Phenocopy-DEE-No-CACNA1C"

        age = random.randint(1, 14)
        if is_ts:
            age = random.randint(1, 10)  # shorter survival / severe
        elif is_phenocopy:
            age = random.randint(3, 18)

        # seizure freedom: low in severe TS
        seizure_free = random.random() < (0.08 if is_ts1 else 0.12 if is_ts2 else 0.22 if is_moderate else 0.35 if is_neuroonly else 0.40)
        drug_resistant = random.random() < (0.85 if is_ts1 else 0.78 if is_ts2 else 0.65 if is_moderate else 0.45 if is_neuroonly else 0.30)

        acth_received = random.random() < (0.92 if is_ts else 0.78 if is_moderate else 0.60 if is_neuroonly else 0.45)
        spasms_present = random.random() < (0.90 if is_ts else 0.72 if is_moderate else 0.55 if is_neuroonly else 0.40)
        lqts_present = random.random() < (0.98 if is_ts1 else 0.95 if is_ts2 else 0.75 if is_moderate else 0.30 if is_neuroonly else 0.10)
        verapamil_tried = random.random() < (0.88 if is_ts else 0.60 if is_moderate else 0.25 if is_neuroonly else 0.10)
        kd_on = random.random() < (0.45 if drug_resistant else 0.20)
        autism_asd = random.random() < (0.92 if is_ts1 else 0.88 if is_ts2 else 0.75 if is_moderate else 0.70 if is_neuroonly else 0.35)
        syndactyly = random.random() < (0.98 if is_ts1 else 0.20 if is_ts2 else 0.0)
        cardiac_device = random.random() < (0.35 if is_ts1 else 0.20 if is_ts2 else 0.08 if is_moderate else 0.0)

        qtc_ms = (
            random.randint(510, 580) if is_ts1 else
            random.randint(500, 570) if is_ts2 else
            random.randint(470, 515) if is_moderate else
            random.randint(450, 480) if is_neuroonly else
            random.randint(420, 460)
        )

        _cohort.append({
            "id": f"CACNA1C-{_pid:03d}",
            "etiology": etio,
            "current_age": age,
            "seizure_free": seizure_free,
            "drug_resistant": drug_resistant,
            "acth_received": acth_received,
            "spasms_present": spasms_present,
            "lqts_present": lqts_present,
            "verapamil_tried": verapamil_tried,
            "kd_on": kd_on,
            "autism_asd": autism_asd,
            "syndactyly": syndactyly,
            "cardiac_device": cardiac_device,
            "qtc_ms": qtc_ms,
            "variant": (
                "G406R (exon 8A)" if is_ts1 else
                "G402S (exon 8)" if is_ts2 else
                "Novel GOF" if is_moderate else
                "Novel GOF (neuronal)" if is_neuroonly else
                "No CACNA1C (phenocopy)"
            ),
        })
        _pid += 1


def get_overview():
    n = len(_cohort)
    seizure_free_n = sum(1 for p in _cohort if p["seizure_free"])
    drug_resistant_n = sum(1 for p in _cohort if p["drug_resistant"])
    acth_n = sum(1 for p in _cohort if p["acth_received"])
    spasms_n = sum(1 for p in _cohort if p["spasms_present"])
    lqts_n = sum(1 for p in _cohort if p["lqts_present"])
    verapamil_n = sum(1 for p in _cohort if p["verapamil_tried"])
    kd_n = sum(1 for p in _cohort if p["kd_on"])
    autism_n = sum(1 for p in _cohort if p["autism_asd"])
    syndactyly_n = sum(1 for p in _cohort if p["syndactyly"])
    cardiac_device_n = sum(1 for p in _cohort if p["cardiac_device"])

    etiology_dist = []
    for e in ETIOLOGY_CATALOG:
        count = sum(1 for p in _cohort if p["etiology"] == e["category"])
        etiology_dist.append({
            "category": e["category"],
            "count": count,
            "pct": round(count / n * 100, 1),
        })

    seizure_summary = [{"type": st["type"], "pct": st["pct"]} for st in SEIZURE_TYPES]
    treatment_summary = [
        {"drug": t["drug"].split(" (")[0].split(" /")[0][:35], "level": t["level"].split(" —")[0]}
        for t in TREATMENTS
    ]
    monitoring_summary = [{"item": m["item"][:55], "frequency": m["frequency"][:60]} for m in MONITORING[:6]]
    lifecycle_summary = [{"window": lc["window"], "key": lc["key_issues"][:80] + "…"} for lc in LIFECYCLE]

    return {
        "kpis": {
            "n_patients": n,
            "seizure_free_pct": round(seizure_free_n / n * 100, 1),
            "drug_resistant_pct": round(drug_resistant_n / n * 100, 1),
            "acth_received_n": acth_n,
            "spasms_n": spasms_n,
            "lqts_present_n": lqts_n,
            "verapamil_tried_n": verapamil_n,
            "kd_on_n": kd_n,
            "autism_asd_n": autism_n,
            "syndactyly_n": syndactyly_n,
            "cardiac_device_n": cardiac_device_n,
            "avg_age_years": round(sum(p["current_age"] for p in _cohort) / n, 1),
        },
        "etiology_distribution": etiology_dist,
        "seizure_summary": seizure_summary,
        "treatments_summary": treatment_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": lifecycle_summary,
        "thresholds": THRESHOLDS[:6],
        "contraindications_summary": [
            {"drug": ci["drug"].split(" (")[0].split(" —")[0].split(" /")[0][:45], "risk": ci["risk"]}
            for ci in CONTRAINDICATIONS[:5]
        ],
    }


def get_breakdown():
    return {
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "patients": _cohort,
    }


def get_definitions():
    return {
        "gene_summary": {
            "gene": "CACNA1C",
            "full_name": "Calcium Voltage-Gated Channel Subunit Alpha1 C",
            "chromosome": "12p13.33",
            "protein": "Cav1.2 (α1C) — L-type High-Voltage-Activated Ca²⁺ Channel",
            "size": "2221 aa (cardiac isoform) / 2138 aa (neuronal) · 49 exons · α1C + β2/β3 + α2δ1",
            "channel_type": "L-type (dihydropyridine-sensitive); HVA (high-voltage-activated; −40 to −20 mV)",
            "activation_threshold": "−40 to −20 mV (HVA; between T-type LVA −80–55 mV and PQ/R-type −30–10 mV)",
            "inactivation_kinetics": "GOF: dramatically impaired VDI (voltage-dependent inactivation) → prolonged window current → excess Ca²⁺ in both cardiac + neurons",
            "primary_location": "Cardiac myocytes (L-type EC coupling) · Dendritic spines cortex/hippocampus (synaptic plasticity) · Smooth muscle (vasomotor)",
            "cav1_subfamily": "Cav1.1/CACNA1S (1q32.1 skeletal/MH) · Cav1.2/CACNA1C (12p13.33 cardiac-neuronal/TS) · Cav1.3/CACNA1D (3p14.3 cochlear/SANDD) · Cav1.4/CACNA1F (Xp11.23 retinal/CSNB2)",
            "inheritance": "AD GOF de novo (>95% TS1/TS2); germline mosaicism ~1-3% recurrence; pLI ~0.99",
            "omim": "OMIM #601005 Timothy Syndrome · *114205 CACNA1C gene · #618485 non-syndromic CACNA1C DEE",
            "precision_treatment": "Verapamil (L-type blocker) Level B — direct Cav1.2 block; dual cardiac + neuronal benefit (Jacobs 2006)",
            "absolute_ci": "Class Ia/III antiarrhythmics (fatal arrhythmia) · TGB (NCSE + QTc) · VPA+POLG1 (Alpers) · CBZ/OXC/PHT (reduce verapamil via CYP3A4 induction)",
        },
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
