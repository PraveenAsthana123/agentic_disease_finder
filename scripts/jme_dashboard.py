"""Juvenile Myoclonic Epilepsy (JME) Dashboard — the most common genetic generalised
epilepsy, accounting for 5–10% of all epilepsies. Onset typically 12–18 years.

Classic triad: myoclonic jerks on awakening + generalised tonic-clonic seizures (GTCS)
               + absence seizures in ~30%.
EEG hallmark: 3–6 Hz generalised (poly)spike-wave, markedly activated by sleep deprivation
              and early awakening; photosensitivity in 30–40%.
Genetics: GABRA1, EFHC1, CLCN2, CACNB4; complex polygenic + rare Mendelian variants.
Treatment: lifelong — 80% relapse on AED withdrawal.
ABSOLUTE CONTRAINDICATIONS: Carbamazepine · Oxcarbazepine · Phenytoin · Phenobarbital
                             Vigabatrin · Tiagabine (all aggravate myoclonic jerks/absence).
WARNING: Lamotrigine — may worsen myoclonus in ~20%; use with caution.

References:
  - Kasteleijn-Nolst Trenité 2012 Epilepsia (JME photosensitivity)
  - Camfield 2009 Epilepsia (JME long-term prognosis)
  - Baykan 2008 Acta Neurol (JME genetics)
  - Genton 2000 Brain (JME & valproate withdrawal)
  - Wheless 2018 Epileptic Disord (JME treatment guidelines)
Data: live clinical.db (41 epilepsy patients, deterministic JME overlay)
      + curated JME pharmacology / genetics / seizure-type / trigger catalogs."""

import sqlite3
import json
from pathlib import Path
from datetime import date

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"
_PROJECT = Path(__file__).resolve().parent.parent


# ─── helpers ────────────────────────────────────────────────────────────────

def _db_rows(sql, params=()):
    try:
        con = sqlite3.connect(DB)
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _seed(pid):
    """Deterministic hash from patient_id string."""
    h = 0
    for c in str(pid):
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


# ─── Genetic / variant catalog ───────────────────────────────────────────────

_GENETICS = [
    {
        "gene": "GABRA1",
        "locus": "5q34",
        "variant_type": "Heterozygous missense / LOF",
        "pct": 18,
        "mechanism": "GABA-A receptor α1 subunit LOF → reduced inhibitory postsynaptic currents → cortical hyperexcitability; autosomal dominant JME sub-type (Cossette 2002)",
        "clinical_note": "Classic JME phenotype; valproate typically effective; earlier onset (mean 11 years)"
    },
    {
        "gene": "EFHC1",
        "locus": "6p12.3",
        "variant_type": "Heterozygous missense",
        "pct": 9,
        "mechanism": "EF-hand containing protein 1; localises to mitotic spindle and cilia; destabilises R-type Ca²⁺ channels → hyperexcitability; linked to myoclonic jerk initiation",
        "clinical_note": "More prominent myoclonia; mild GTCS; photosensitivity less common; some familial JME pedigrees"
    },
    {
        "gene": "CLCN2",
        "locus": "3q27.1",
        "variant_type": "Heterozygous LOF / gain-of-function",
        "pct": 7,
        "mechanism": "Chloride channel ClC-2; LOF → intracellular Cl⁻ accumulation → depolarising GABA responses in cortical neurons → paradoxical excitation",
        "clinical_note": "JME + IGE phenotypic spectrum; some CLCN2 variants linked to febrile seizures (GEFS+); acetazolamide sometimes adjunctive"
    },
    {
        "gene": "CACNB4",
        "locus": "2q22–23",
        "variant_type": "Heterozygous missense / truncation",
        "pct": 5,
        "mechanism": "Voltage-gated Ca²⁺ channel β4-subunit; regulates P/Q-type channel gating → thalamocortical oscillation dysregulation; JME + episodic ataxia overlap",
        "clinical_note": "Rare; JME phenotype with occasional cerebellar features; ethosuximide less effective"
    },
    {
        "gene": "SCN1A / SCN2A / GABRD",
        "locus": "Multiple",
        "variant_type": "Heterozygous missense (minor)",
        "pct": 4,
        "mechanism": "Sodium channel Nav1.1/Nav1.2 and GABA-A δ-subunit variants observed in JME-spectrum families; lower penetrance than classic channelopathy phenotypes",
        "clinical_note": "JME-like phenotype; genetic counselling important; broader IGE spectrum referral"
    },
    {
        "gene": "Polygenic / Complex (no single gene identified)",
        "locus": "Multiple GWAS loci",
        "variant_type": "Common variant risk alleles",
        "pct": 57,
        "mechanism": "GWAS identifies risk variants near EFHC1, BRD7, IMMP2L, NRXN3, and multiple ion-channel gene regions; no single causal gene identified in majority; complex inheritance",
        "clinical_note": "Majority of JME; standard phenotypic diagnosis; genetic testing low yield but recommended for atypical cases or family planning"
    },
]

# ─── Seizure types ───────────────────────────────────────────────────────────

_SEIZURE_TYPES = [
    {
        "type": "Myoclonic Jerks (MJ)",
        "prevalence_pct": 100,
        "description": "Sudden, brief (50–200 ms), involuntary bilateral synchronous limb jerks; predominantly upper limbs/shoulders; characteristic on awakening within 30–60 min of waking",
        "eeg": "Synchronous generalised poly-spike (2–20 Hz) or polyspike-wave discharge; coincides with jerk; interictal 3–6 Hz GSW",
        "clinical_tip": "Ask about 'breakfast-time clumsiness' — dropped cups, toothbrush accidents. Often dismissed as nervousness before JME diagnosis.",
        "triggers": "Sleep deprivation (90%), alcohol (65%), early awakening, excitement, menstrual cycle",
        "first_line_aed": "Valproate / Levetiracetam"
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "prevalence_pct": 95,
        "description": "Bilateral GTCS, usually 30 s–3 min; most commonly triggered by morning myoclonic jerks that escalate; occasional nocturnal GTCS from sleep deprivation",
        "eeg": "Generalised 3–6 Hz (poly)spike-wave before tonic phase; widespread voltage attenuation post-ictally",
        "clinical_tip": "GTCS in JME: typically follows unchecked myoclonic jerk clusters on awakening — educate patient/family to lie down after morning jerks",
        "triggers": "Sleep deprivation, missed AED dose, alcohol the night before, menstrual cycle (catamenial overlap)",
        "first_line_aed": "Valproate / Levetiracetam / Lamotrigine (CAUTION for myoclonus)"
    },
    {
        "type": "Absence Seizures (AS)",
        "prevalence_pct": 30,
        "description": "Brief (5–30 sec) staring episodes with impaired awareness; no automatisms (unlike CAE); often unnoticed by patient; may precede JME diagnosis by years",
        "eeg": "3–4 Hz regular spike-wave (slower than CAE 3 Hz); absence duration 5–15 sec; generalised",
        "clinical_tip": "Absence in JME: less prominent than CAE; if ethosuximide is trialled (absence only) it will NOT protect against GTCS — levetiracetam or valproate preferred",
        "triggers": "Hyperventilation, sleep deprivation, photosensitivity (some)",
        "first_line_aed": "Valproate (covers all 3 types) / Levetiracetam+Lamotrigine combination"
    },
    {
        "type": "Photosensitive Responses (EEG/Clinical)",
        "prevalence_pct": 35,
        "description": "Intermittent photic stimulation (IPS) induces generalised (poly)spike-wave; 30–40% of JME patients photosensitive; clinical photosensitive GTCS in ~15%",
        "eeg": "Photoparoxysmal response (PPR) Grades II–IV on IPS at 14–20 Hz; generalised discharges with/without myoclonus",
        "clinical_tip": "Advise polarised sunglasses (UV400), avoid strobe environments. Sodium valproate and levetiracetam reduce photosensitivity; carbamazepine WORSENS it",
        "triggers": "Television screens, video games, flickering sunlight, strobes",
        "first_line_aed": "Valproate (best photosensitivity reduction) / Levetiracetam"
    },
]

# ─── Seizure triggers ────────────────────────────────────────────────────────

_TRIGGERS = [
    {"trigger": "Sleep Deprivation", "frequency_pct": 90, "mechanism": "Reduced cortical inhibition; homeostatic sleep pressure alters thalamocortical oscillations; myoclonic threshold lowered", "mitigation": "Regular sleep schedule (7–9h); alarm no earlier than 7 AM; nap scheduling; educate patient"},
    {"trigger": "Alcohol Consumption", "frequency_pct": 65, "mechanism": "Acute GABAergic rebound excitability 12–24h after alcohol; dehydration; disrupted sleep architecture", "mitigation": "Complete alcohol abstinence recommended; even 1–2 units significantly raises next-morning risk"},
    {"trigger": "Missed AED Dose", "frequency_pct": 55, "mechanism": "Plasma AED level drop below therapeutic range → loss of seizure threshold protection", "mitigation": "Dose-reminder alarm; blister pack; travel supply; double-dose protocol discussion with neurologist"},
    {"trigger": "Stress / Anxiety", "frequency_pct": 45, "mechanism": "Cortisol-mediated HPA axis → increased neuronal excitability; impaired sleep quality", "mitigation": "CBT, mindfulness, stress management; assess for comorbid anxiety/depression (common in JME)"},
    {"trigger": "Photosensitivity (flash/flicker)", "frequency_pct": 35, "mechanism": "Cortical visual hyperexcitability → generalised discharge propagation; IPS threshold lowered in JME", "mitigation": "Blue-tinted or polarised lenses; cover one eye; avoid gaming at night; screen brightness reduction"},
    {"trigger": "Menstrual Cycle / Hormonal", "frequency_pct": 30, "mechanism": "Perimenstrual oestrogen peak → increased glutamate/NMDA sensitivity; progesterone withdrawal reduces GABA-A efficacy", "mitigation": "Catamenial diary; consider clobazam supplementation (10 mg/day days 22–2 of cycle) — neurologist decision"},
    {"trigger": "Fever / Intercurrent Illness", "frequency_pct": 20, "mechanism": "Fever lowers seizure threshold; sodium channel gating alteration; sleep disruption from illness", "mitigation": "Antipyretics early; maintain AED adherence; seek medical care for prolonged illness"},
    {"trigger": "Caffeine / Stimulant excess", "frequency_pct": 15, "mechanism": "Adenosine A1 receptor blockade → reduced inhibitory tone; disrupted sleep architecture if consumed late", "mitigation": "Moderate caffeine (< 300 mg/day); avoid energy drinks; no caffeine after 2 PM"},
]

# ─── Treatments ─────────────────────────────────────────────────────────────

_TREATMENTS = [
    {
        "drug": "Valproate (Depakote / Epilim)",
        "fda_status": "FDA-approved (generalised epilepsy, 1978); REMS for childbearing-age women (REMS 2013 — pregnancy prevention)",
        "dose": "1000–2000 mg/day in 2 divided doses (adults); therapeutic trough 50–100 µg/mL; slow-release formulation preferred",
        "moa": "Multiple: enhances GABA-T inhibition (↑ GABA), blocks voltage-gated Na⁺ and T-type Ca²⁺ channels, reduces glutamate release; broadest spectrum anti-epileptic",
        "efficacy": "Seizure-free in 80–85% (all JME seizure types); covers myoclonus + GTCS + absence; reduces photosensitivity",
        "safety": "Teratogenicity (Category X — spina bifida 1–2%, fetal valproate syndrome); weight gain; tremor; PCOS; hepatotoxicity (rare); pancreatitis (rare); REMS mandatory for women of childbearing age",
        "evidence_level": "ILAE Level A (all JME seizure types); first-line in men and post-menopausal women; AVOID or minimise in women of childbearing age (NICE NG217 2022)"
    },
    {
        "drug": "Levetiracetam (Keppra)",
        "fda_status": "FDA-approved (myoclonic seizures in JME ≥12 years, 2005); IV available for acute use",
        "dose": "1000–3000 mg/day in 2 divided doses; titrate over 2–4 weeks; renal dose adjustment required",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulation → reduces vesicular neurotransmitter release; no effect on Na⁺/Ca²⁺ channels; unique target",
        "efficacy": "Myoclonic jerk reduction 76–85%; GTCS reduction 73%; lower absence efficacy than valproate; comparative to valproate in SANAD-II (Marson 2021)",
        "safety": "Irritability/mood changes ('Keppra rage') in 10–20% (B6 supplementation may help); drowsiness; headache; safe in pregnancy (EURAP data: lowest teratogenic risk of all AEDs); no significant drug-drug interactions",
        "evidence_level": "ILAE Level A (myoclonic JME); preferred first-line for women of childbearing age (NICE NG217); SANAD-II non-inferior to valproate for seizure control at 2 years"
    },
    {
        "drug": "Lamotrigine (Lamictal)",
        "fda_status": "FDA-approved (partial-onset + primary GTCS ≥2 years)",
        "dose": "100–400 mg/day; slow titration mandatory (25 mg/14 days to avoid SJS); lower dose with valproate co-medication",
        "moa": "Voltage-gated Na⁺ channel blocker + Ca²⁺ (N-type) modulation; stabilises presynaptic neuronal membranes; reduces glutamate release",
        "efficacy": "GTCS reduction 65–75%; absence reduction 60–70%; CAUTION: may WORSEN myoclonic jerks in 10–20% of JME patients (paradoxical myoclonus exacerbation)",
        "safety": "Stevens-Johnson syndrome (SJS) risk (0.1%) — slow titration essential; rash (5–10%); dizziness; insomnia; relatively safe in pregnancy (Category C) — lower teratogenic risk than valproate",
        "evidence_level": "ILAE Level B (JME GTCS/absence); use with CAUTION for myoclonus — monitor for worsening; preferred adjunct to levetiracetam in women of childbearing age"
    },
    {
        "drug": "Topiramate (Topamax)",
        "fda_status": "FDA-approved (primary GTCS ≥10 years); adjunctive use in JME",
        "dose": "100–400 mg/day in 2 doses; slow titration (25 mg/week); target dose over 8 weeks",
        "moa": "Blocks voltage-gated Na⁺ channels; GABA-A potentiation; AMPA/kainate receptor blockade; carbonic anhydrase inhibition",
        "efficacy": "GTCS reduction 65–75%; myoclonic jerk reduction 55–65%; cognitive side effects limit dose escalation; adjunct preferred",
        "safety": "Cognitive dulling ('dopamax') — word-finding difficulties, slowed processing; weight loss; nephrolithiasis (2–4%); oligohidrosis/hyperthermia; teratogenic (cleft lip risk — avoid in pregnancy or use contraception)",
        "evidence_level": "ILAE Level B (adjunct JME); useful in obese patients (weight-neutral/loss); prefer levetiracetam for better cognitive profile"
    },
    {
        "drug": "Zonisamide (Zonegran)",
        "fda_status": "FDA-approved (adjunctive partial seizures ≥16 years); used off-label in JME",
        "dose": "100–500 mg/day once daily; titrate 100 mg/4 weeks; low protein-binding simplifies co-medication",
        "moa": "Blocks voltage-gated Na⁺ and T-type Ca²⁺ channels; carbonic anhydrase inhibition; free-radical scavenging; reduces high-frequency neuronal firing",
        "efficacy": "Myoclonic jerk reduction 70–80% in open-label JME studies; GTCS reduction 65–75%; well-tolerated once-daily dosing improves adherence",
        "safety": "Weight loss; drowsiness; nephrolithiasis (2–5%); cognitive effects less prominent than topiramate; oligohidrosis; teratogenic data limited — use contraception",
        "evidence_level": "ILAE Level C (JME adjunct); good for adherence (once daily); effective alternative if levetiracetam mood effects intolerable"
    },
    {
        "drug": "Clobazam (Onfi)",
        "fda_status": "FDA-approved (Lennox-Gastaut adjunct 2011); off-label for JME",
        "dose": "10–40 mg/day in 1–2 doses; catamenial supplementation: 10 mg/day days 22–2 of cycle",
        "moa": "Positive allosteric modulator of GABA-A receptor (1,5-benzodiazepine); less sedating than clonazepam; broader anti-epileptic profile",
        "efficacy": "Catamenial JME seizure control (premenstrual exacerbation) 70% improvement with cyclic clobazam; continuous use limited by tolerance (6–12 months); useful PRN for sleep deprivation periods",
        "safety": "Sedation; tolerance development (limit continuous use to < 3 months); mood change; teratogenic (avoid in first trimester); physical dependence risk — taper on discontinuation",
        "evidence_level": "ILAE Level B (catamenial supplementation); PRN use in high-risk situations (travel, exam stress, sleep-deprived period)"
    },
    {
        "drug": "Perampanel (Fycompa)",
        "fda_status": "FDA-approved (myoclonic seizures in JME ≥12 years, 2018); once-daily",
        "dose": "2–12 mg/day at bedtime; titrate 2 mg/2 weeks; 8 mg typical effective dose",
        "moa": "First-in-class selective non-competitive AMPA receptor antagonist; reduces post-synaptic glutamate-mediated excitation; unique mechanism complements GABAergic AEDs",
        "efficacy": "PMID 30297088 (ELEVATE trial): 64.5% myoclonic jerk responder rate; GTCS-free 47%; bedtime dosing minimises CNS side effects; sustained long-term efficacy",
        "safety": "Dizziness, somnolence, irritability/aggression (5–10%); psychiatric history: increased caution; weight gain; CYP3A4 induction (reduces perampanel levels — adjust with EIAEDs); REMS program required (risk of serious psychiatric events)",
        "evidence_level": "ILAE Level A (myoclonic JME, 2018 FDA approval); excellent option for treatment-resistant myoclonic jerks; REMS mandatory"
    },
    {
        "drug": "Clonazepam (Klonopin)",
        "fda_status": "FDA-approved (myoclonic seizures, 1975); controlled substance Schedule IV",
        "dose": "0.5–4 mg/day in 2–3 doses; start low 0.25 mg/day; NOT recommended for long-term monotherapy",
        "moa": "Benzodiazepine GABA-A positive allosteric modulator; enhances Cl⁻ influx → rapid myoclonic jerk suppression",
        "efficacy": "Rapid (days) myoclonic suppression 85–90%; tolerance develops within 3–6 months significantly reducing efficacy; not suitable for primary maintenance AED",
        "safety": "Sedation; tolerance and dependence; withdrawal seizures on abrupt discontinuation; falls risk in elderly; avoid in respiratory disease; teratogenic (Category D)",
        "evidence_level": "ILAE Level C (rescue or adjunct); reserved for add-on in refractory morning myoclonus; short-course use only — tolerance limits long-term utility"
    },
]

# ─── ABSOLUTE CONTRAINDICATIONS ─────────────────────────────────────────────

_CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (Tegretol)",
        "risk_level": "ABSOLUTE CONTRAINDICATION",
        "mechanism": "Na⁺ channel blocker that REDUCES generalised spike-wave inhibition → worsens myoclonic jerks (acute myoclonic exacerbation), increases absence seizures, may precipitate absence status",
        "consequence": "Dramatic worsening of myoclonic jerks (within days of starting); new-onset absence status epilepticus; paradoxical seizure exacerbation",
        "evidence": "Ferrie 1995 / Genton 2000: multiple case series of severe myoclonic worsening; standard of care contraindication worldwide"
    },
    {
        "drug": "Oxcarbazepine (Trileptal / Oxtellar)",
        "risk_level": "ABSOLUTE CONTRAINDICATION",
        "mechanism": "Same mechanism as carbamazepine (Na⁺ channel blocker); may be slightly less severe but same class-effect myoclonic aggravation",
        "consequence": "Myoclonic jerk exacerbation; increased GTCS frequency; absence worsening; potential absence status epilepticus",
        "evidence": "Guerreiro 1997; class-effect extension from carbamazepine data; widely recognised in JME literature"
    },
    {
        "drug": "Phenytoin (Dilantin / Phenytek)",
        "risk_level": "ABSOLUTE CONTRAINDICATION",
        "mechanism": "Na⁺ channel stabiliser preferentially acting on focal activity; worsens 3–6 Hz generalised spike-wave → increased myoclonic jerk frequency and severity",
        "consequence": "Myoclonic worsening (50% of patients); may paradoxically increase GTCS via absence status → GTCS cascade; cerebellar toxicity with chronic use",
        "evidence": "Panayiotopoulos 2005 'The Epilepsies' atlas; standard JME contraindication; historical documentation"
    },
    {
        "drug": "Phenobarbital (Luminal)",
        "risk_level": "ABSOLUTE CONTRAINDICATION",
        "mechanism": "Although GABAergic, phenobarbital worsens absence and myoclonic jerks via unclear mechanism (possible disruption of cortical inhibitory interneuron timing); sedation masks myoclonus without treating it",
        "consequence": "Absence worsening; myoclonic jerk persistence or increase; cognitive sedation; dependence; rarely paradoxical disinhibition",
        "evidence": "Berkovic 1991; ILAE guidelines; historical case series — still occasionally misused in LMIC settings"
    },
    {
        "drug": "Vigabatrin (Sabril)",
        "risk_level": "ABSOLUTE CONTRAINDICATION",
        "mechanism": "GABA-T irreversible inhibition → increased GABA; paradoxically worsens generalised spike-wave (mechanism unclear — possibly alters cortical inhibitory/excitatory balance via extrasynaptic GABA accumulation)",
        "consequence": "Severe myoclonic exacerbation; de novo absence status; may trigger convulsive status epilepticus; in addition — permanent visual field defects",
        "evidence": "Ferrie 1995; Beran 1996; universal JME contraindication; compounded by permanent visual toxicity"
    },
    {
        "drug": "Tiagabine (Gabitril)",
        "risk_level": "ABSOLUTE CONTRAINDICATION",
        "mechanism": "GABA re-uptake inhibitor (GAT-1 blocker) → prolonged GABA effects but paradoxically produces non-convulsive status epilepticus (NCSE) in generalised epilepsies via unclear mechanism",
        "consequence": "Non-convulsive status epilepticus (NCSE) risk; myoclonic exacerbation; absence worsening; NCSE may be mistaken for drug toxicity",
        "evidence": "Ettinger 1999; FDA post-marketing cases; universally contraindicated in IGE/JME"
    },
]

# ─── AED monitoring requirements ─────────────────────────────────────────────

_AED_MONITORING = [
    {
        "drug": "Valproate",
        "category": "REMS + TERATOGENICITY + TDM",
        "risk": "Spina bifida 1–2%, fetal valproate syndrome (craniofacial/limb/cognitive effects); PCOS; weight gain; hepatotoxicity; pancreatitis; tremor",
        "monitoring": "REMS (Valproate REMS 2013): pregnancy prevention counselling every 6 months for women of childbearing potential; LFT/ammonia at baseline + 6M; trough level 50–100 µg/mL; BMI; platelets (thrombocytopenia risk)",
        "mitigation": "Folic acid 5 mg/day pre-conception; contraception mandatory if prescribed; transition to levetiracetam before planned pregnancy; Extended-release formulation for tremor/tolerance",
        "evidence": "FDA Valproate REMS 2013; EURAP Registry (Tomson 2019); NICE NG217 2022; MONEAD study (Meador 2020)"
    },
    {
        "drug": "Levetiracetam",
        "category": "MOOD / BEHAVIOUR MONITORING",
        "risk": "Irritability, aggression ('Keppra rage') in 10–20%; suicidal ideation (class warning); depression; anxiety; hostility especially in adolescents",
        "monitoring": "Mood questionnaire (GAD-7/PHQ-9) at each visit; patient/caregiver interview for irritability/behavioural change; renal function (dose-adjust for GFR <50 mL/min); plasma levels not routinely needed",
        "mitigation": "Pyridoxine (B6) 50–200 mg/day may reduce irritability (Asadi-Pooya 2008); switch to alternative if intolerable mood effects; gradual titration; consider clobazam PRN instead if mood-sensitive",
        "evidence": "Mula 2003 (levetiracetam psychiatric effects); FDA class warning AEDs + suicidality; SANAD-II Marson 2021"
    },
    {
        "drug": "Lamotrigine",
        "category": "SJS RASH MONITORING + MYOCLONUS WATCH",
        "risk": "Stevens-Johnson syndrome/TEN (0.1% — highest in first 8 weeks); maculopapular rash 5–10%; paradoxical myoclonic worsening (10–20% JME patients); dizziness; insomnia",
        "monitoring": "Rash surveillance protocol: patient instructed to seek ER for any rash in first 8 weeks; slow titration mandatory (25 mg/14 days then 50 mg/14 days etc.); myoclonic jerk frequency diary at each visit (first 3 months)",
        "mitigation": "NEVER increase lamotrigine dose in JME patient reporting worsening morning jerks — reduce or stop; HALT lamotrigine immediately for any rash; halve dose if adding valproate (drug-drug interaction doubles LTG level)",
        "evidence": "NICE NG217 2022; Biraben 2000 (LTG myoclonus); SJS FDA label warning; Brodie 2000"
    },
    {
        "drug": "Perampanel",
        "category": "REMS + PSYCHIATRIC MONITORING",
        "risk": "Dizziness; aggression/psychiatric effects (hostility, suicidal ideation); falls; CYP3A4 induction by EIAEDs (40% level reduction with carbamazepine — avoid combination in JME)",
        "monitoring": "Perampanel REMS: patient must be counselled on risk of serious psychiatric events; PHQ-9/mood assessment at each visit; avoid in active psychosis; plasma levels useful if co-prescribed enzyme inducers; adjust for hepatic impairment",
        "mitigation": "Bedtime administration minimises CNS side-effects; start 2 mg/night; titrate q2 weeks; avoid co-prescription of EIAEDs (carbamazepine contraindicated in JME anyway)",
        "evidence": "ELEVATE trial (French 2015 Epilepsia); FDA approval 2018; Perampanel REMS; NICE 2014 TA307"
    },
]

# ─── Developmental / lifecycle trajectory ───────────────────────────────────

_LIFECYCLE = [
    {
        "age_window": "Childhood (before onset: 6–11 years)",
        "typical_profile": "Normal development; no seizure history; school performance appropriate; family history of epilepsy/febrile seizures common in JME kindreds",
        "jme_indicators": "Occasional 'eye-fluttering' episodes (unrecognised absence in ~30%); school teacher reports of 'blanking'; no formal diagnosis yet",
        "action": "Family history screening; if eye-fluttering reported → EEG (3 Hz GSW may be seen pre-JME); educate family about genetic epilepsy risk"
    },
    {
        "age_window": "Adolescent onset (12–18 years — peak JME onset)",
        "typical_profile": "Puberty/hormonal changes; exam stress; reduced sleep; alcohol experimentation; driving aspirations; identity development",
        "jme_indicators": "Morning myoclonic jerks (may dismiss as tiredness); first GTCS often precipitated by sleep deprivation + alcohol; diagnosis typically made after first witnessed GTCS",
        "action": "URGENT EEG (3–6 Hz GSW ± polyspike-wave); START levetiracetam or valproate (if male/post-menopausal female); EDUCATION: sleep, alcohol, driving restrictions; 12-month seizure-free before driving consideration"
    },
    {
        "age_window": "Young Adult (18–30 years) — high-risk lifestyle period",
        "typical_profile": "University, new employment, independent living, alcohol exposure, irregular sleep, relationship formation, driving; women: pregnancy planning",
        "jme_indicators": "High relapse risk: peer pressure → alcohol + late nights → GTCS at 23:00 or 07:00 AM; AED non-adherence common; breakthrough seizures",
        "action": "AED adherence counselling (blister packs); contraception discussion; switch valproate → levetiracetam before pregnancy; SUDEP risk counselling; driving regulations (national rules vary: 12 months seizure-free); alcohol harm reduction"
    },
    {
        "age_window": "Childbearing years (women 18–45 years)",
        "typical_profile": "Pregnancy planning; concerns about AED teratogenicity; breastfeeding decisions; hormonal contraception effects on AED levels",
        "jme_indicators": "Valproate REMS consultation mandatory; hormonal contraception: enzyme-inducing AEDs reduce OCP efficacy (valproate/levetiracetam do NOT induce CYP450); catamenial exacerbation in 30%",
        "action": "Preconception counselling: switch to levetiracetam + folic acid 5 mg/day before conception; EURAP registry enrolment; gestational AED monitoring (levetiracetam dose may need ↑ 50% in pregnancy due to renal clearance); postpartum: breastfeeding generally safe with levetiracetam"
    },
    {
        "age_window": "Middle Adult (30–60 years)",
        "typical_profile": "Established lifestyle; career; family; most patients achieve seizure control; occasional breakthrough from stress/illness/poor sleep",
        "jme_indicators": "80% relapse if AED withdrawn → lifelong treatment recommended; career decisions (heights, machinery, swimming); SUDEP risk ongoing; psychiatric comorbidity (anxiety 20%, depression 15%)",
        "action": "Reinforce lifelong treatment message; annual review (comorbidity screening PHQ-9/GAD-7); occupational assessment; no AED withdrawal trials unless very exceptional circumstances; SUDEP discussion"
    },
    {
        "age_window": "Older Adult (>60 years)",
        "typical_profile": "Retirement; polypharmacy risk; metabolic changes; reduced hepatic/renal function; cognitive changes; caregiver dependency potential",
        "jme_indicators": "JME typically well-controlled by this age if adherent; new drug-drug interactions from other medications; falls risk from AED side-effects; rare late-onset myoclonic re-emergence",
        "action": "Review AED dosing for age-related pharmacokinetic changes (reduce levetiracetam dose for declining renal function); polypharmacy review; falls prevention (avoid clonazepam if fall risk); cognition screening"
    },
]

# ─── Key definitions ─────────────────────────────────────────────────────────

_DEFINITIONS = [
    {"term": "Juvenile Myoclonic Epilepsy (JME)", "definition": "Most common genetic generalised epilepsy (5–10% of all epilepsies). Onset 12–18 years. Classic triad: myoclonic jerks on awakening + GTCS + absence seizures (30%). Requires lifelong AED therapy (80% relapse on withdrawal). ILAE 2022 classified as Genetic Generalised Epilepsy (GGE)."},
    {"term": "Myoclonic Jerk (MJ)", "definition": "Brief (50–200 ms) bilateral synchronous involuntary muscle contraction; predominantly upper limbs/shoulders on awakening. EEG correlate: generalised polyspike or polyspike-wave. Pathognomonic morning jerks (breakfast clumsiness) suggest JME."},
    {"term": "Generalised (Poly)spike-wave (GSW/GPSW)", "definition": "EEG hallmark of JME: 3–6 Hz generalised spike-wave and polyspike-wave discharges; most prominent 20–30 min after awakening; augmented by sleep deprivation, photosensitivity, hyperventilation. Distinguishes JME from focal epilepsy."},
    {"term": "Photoparoxysmal Response (PPR)", "definition": "EEG response to intermittent photic stimulation (IPS): generalised spike-wave induced by flickering light at 14–20 Hz. Present in 30–40% of JME patients. Valproate and levetiracetam reduce PPR; carbamazepine INCREASES PPR."},
    {"term": "Genetic Generalised Epilepsy (GGE)", "definition": "ILAE 2022 classification superseding 'idiopathic generalised epilepsy (IGE)': epilepsies presumed genetic basis with generalised EEG patterns and no structural lesion. JME is a GGE subtype alongside CAE, JAE, GTCS-alone."},
    {"term": "SANAD-II Trial", "definition": "Marson 2021 Lancet: UK RCT comparing valproate vs levetiracetam vs zonisamide in newly diagnosed GGE. Valproate most effective (57% 12-month remission); levetiracetam non-inferior in seizure control but inferior on composite outcome. Reinforced levetiracetam preference for women of childbearing age."},
    {"term": "ELEVATE Trial", "definition": "French 2015 Epilepsia: Phase III RCT of adjunctive perampanel in JME. 64.5% myoclonic jerk responder rate; 47% GTCS-free; led to FDA approval of perampanel for myoclonic seizures in JME (2018)."},
    {"term": "Valproate REMS", "definition": "FDA Risk Evaluation and Mitigation Strategy (2013): mandatory counselling every 6 months for women of childbearing age about fetal valproate syndrome and neural tube defects. Healthcare provider must confirm no effective contraception before prescribing."},
    {"term": "Sleep Deprivation Seizure Threshold", "definition": "JME-specific pathophysiology: sleep deprivation lowers myoclonic seizure threshold by altering thalamocortical oscillations and reducing cortical GABAergic tone. Clinical instruction: minimum 7–9 hours sleep nightly; no napping as sleep substitute."},
    {"term": "Catamenial JME", "definition": "Perimenstrual seizure exacerbation in ~30% of women with JME: oestrogen peak (days 12–14) and progesterone withdrawal (days 25–27) alter neuronal excitability. Management: cyclic clobazam 10 mg/day days 22–2; progesterone supplementation (limited evidence)."},
    {"term": "GABRA1 Variant", "definition": "α1-subunit of GABA-A receptor; LOF variants (A322D, D219N) cause JME via reduced Cl⁻ conductance → disinhibited cortical neurons. Autosomal dominant JME sub-type (Cossette 2002 Nature Genetics). First JME gene identified in a large French-Canadian pedigree."},
    {"term": "AED Aggravation (JME)", "definition": "Paradoxical worsening of seizures by sodium channel blockers (carbamazepine, oxcarbazepine, phenytoin) and vigabatrin in JME: these drugs fail to suppress generalised spike-wave and may increase myoclonic jerk frequency and precipitate absence status epilepticus."},
    {"term": "Lifelong Treatment Principle", "definition": "JME requires lifelong AED therapy: 80% of patients relapse within 1–2 years of AED withdrawal (Camfield 2009; Genton 2000). Remission criteria are not established. AED cessation only considered in exceptional circumstances (prolonged seizure freedom >5 years + no EEG discharges + low-risk lifestyle)."},
    {"term": "SUDEP in JME", "definition": "Sudden Unexpected Death in Epilepsy: estimated 1:1000 patient-years in JME (lower than DRE but higher than general population). Risk factors: frequent nocturnal GTCS, prone sleeping position, solitary environment, poor AED adherence. Safety counselling mandatory."},
]

# ─── Key standards ────────────────────────────────────────────────────────────

_STANDARDS = [
    {"name": "ILAE JME Classification 2022", "body": "International League Against Epilepsy", "scope": "JME classified as Genetic Generalised Epilepsy (GGE); triad definition; EEG criteria; treatment hierarchy; monitoring"},
    {"name": "UK NICE Guideline NG217 (Epilepsies 2022)", "body": "National Institute for Health and Care Excellence", "scope": "Valproate REMS in women of childbearing age; levetiracetam preferred in women; lifelong treatment endorsement; seizure aggravation drug list"},
    {"name": "FDA Valproate REMS 2013", "body": "US Food and Drug Administration", "scope": "Mandatory pregnancy prevention counselling every 6 months; healthcare provider/patient agreement; contraception documentation"},
    {"name": "FDA Perampanel Approval (JME) 2018", "body": "US Food and Drug Administration", "scope": "Perampanel indicated for myoclonic seizures in patients ≥12 years with JME; adjunctive therapy; REMS for psychiatric risk"},
    {"name": "SANAD-II RCT 2021 (Marson)", "body": "UK MRC", "scope": "Comparative effectiveness valproate vs levetiracetam vs zonisamide in GGE; established evidence hierarchy for GGE including JME"},
    {"name": "EAN/EFNS JME Consensus 2013 (Steinhoff)", "body": "European Academy of Neurology", "scope": "JME treatment algorithm; contraindicated AEDs list; women-specific guidance; lifestyle modification protocol"},
]

_THRESHOLDS = [
    {"parameter": "Valproate REMS counselling interval", "threshold": "Every 6 months for women of childbearing age", "unit": "months", "significance": "Mandatory FDA REMS requirement — prescriber, patient, pharmacy all must comply; prescriber at risk of liability if skipped"},
    {"parameter": "Lamotrigine starting dose (without valproate)", "threshold": "25 mg/day × 14 days → 50 mg/day × 14 days then titrate", "unit": "mg/14 days", "significance": "Slow titration mandatory — rapid dose escalation is primary risk factor for Stevens-Johnson syndrome"},
    {"parameter": "Lamotrigine dose adjustment with valproate", "threshold": "Halve all lamotrigine doses when co-prescribed with valproate (VPA inhibits LTG glucuronidation → doubles plasma LTG)", "unit": "Dose halved", "significance": "Drug-drug interaction causes 2× LTG exposure → SJS risk; start LTG 12.5 mg/day with VPA"},
    {"parameter": "JME relapse risk on AED withdrawal", "threshold": "80% relapse within 1–2 years of stopping AED", "unit": "% relapse", "significance": "Lifelong treatment standard — counsel all newly diagnosed JME patients; only exceptional cases (>5 year seizure-free, normal EEG, low-risk lifestyle) consider withdrawal"},
    {"parameter": "Seizure-free before driving (UK/CAN)", "threshold": "12 months seizure-free (no seizures including myoclonic jerks) for standard driving licence", "unit": "months", "significance": "Myoclonic jerks ALONE disqualify driving if they could cause a road accident; patient must declare to DVLA/transport authority"},
    {"parameter": "Folic acid dose (pre-conception, valproate use)", "threshold": "5 mg/day (high-dose) starting 3 months before planned conception, continue through first trimester", "unit": "mg/day", "significance": "Standard 0.4 mg/day insufficient when taking valproate (teratogenic AED); high-dose folate reduces but does not eliminate neural tube defect risk"},
]

_REFERENCES = [
    {"citation": "Genton P et al. Brain 2000;123:2495-2505.", "key_finding": "JME relapse after AED withdrawal: 80% relapse within 2 years; established lifelong treatment necessity"},
    {"citation": "Camfield P & Camfield C. Epilepsia 2009;50(Suppl 5):22-24.", "key_finding": "JME long-term prognosis: 90% seizure-free on appropriate AEDs; 80% relapse on withdrawal; social outcome generally good with adherence"},
    {"citation": "Marson A et al. (SANAD-II) Lancet 2021;397:1016-1026.", "key_finding": "Valproate most effective for GGE/JME (57% 12M remission); levetiracetam non-inferior for seizure control; endorsed levetiracetam for women"},
    {"citation": "French JA et al. (ELEVATE) Epilepsia 2015;56:1291-1302.", "key_finding": "Perampanel: 64.5% myoclonic jerk responder rate in JME; FDA approval 2018 for myoclonic seizures in JME"},
    {"citation": "Kasteleijn-Nolst Trenité D et al. Epilepsia 2012;53(Suppl 2):9-14.", "key_finding": "JME photosensitivity in 30–40% of patients; IPS protocol standardisation; valproate and levetiracetam reduce PPR"},
    {"citation": "Baykan B et al. Acta Neurol Scand 2008;117:23-30.", "key_finding": "JME genetics: GABRA1 18% prevalence; EFHC1 9%; complex polygenic majority; counselling framework"},
]


# ─── patient overlay (deterministic from clinical.db) ──────────────────────

def _get_patients():
    rows = _db_rows("""
        SELECT p.patient_id, p.age, p.gender, p.disease
        FROM patients p
        ORDER BY p.patient_id
    """)
    _SEIZURE_SUBTYPES = ["MJ only", "MJ + GTCS", "MJ + GTCS + Absence", "GTCS + Absence", "MJ + Absence"]
    _AED_REGIMEN = [
        "Levetiracetam mono",
        "Valproate mono",
        "Levetiracetam + Lamotrigine",
        "Valproate + Clobazam",
        "Levetiracetam + Perampanel",
        "Zonisamide + Levetiracetam",
        "Topiramate + Levetiracetam",
    ]
    _CONTROL = ["Seizure-free", "Improved (>50% reduction)", "Partial response", "Drug-resistant"]
    result = []
    for i, r in enumerate(rows[:41]):
        sd = _seed(r["patient_id"])
        gene_idx = sd % len(_GENETICS)
        onset_age = 12 + (sd % 7)  # 12–18
        subtype_idx = (sd >> 4) % len(_SEIZURE_SUBTYPES)
        aed_idx = (sd >> 8) % len(_AED_REGIMEN)
        photosensitive = bool((sd >> 12) % 3 == 0)  # ~33%
        catamenial = r.get("gender", "M") == "F" and bool((sd >> 14) % 3 == 0)
        control_w = [50, 25, 15, 10]
        control_idx = 0
        rv = (sd >> 16) % 100
        acc = 0
        for ci, w in enumerate(control_w):
            acc += w
            if rv < acc:
                control_idx = ci
                break
        years_on_aed = max(1, (r.get("age", 30) or 30) - onset_age)
        result.append({
            "patient_id": r["patient_id"],
            "sex": r.get("gender", "Unknown"),
            "onset_age_years": onset_age,
            "seizure_types": _SEIZURE_SUBTYPES[subtype_idx],
            "genetic_variant": _GENETICS[gene_idx]["gene"],
            "aed_regimen": _AED_REGIMEN[aed_idx],
            "seizure_control": _CONTROL[control_idx],
            "photosensitive": photosensitive,
            "catamenial": catamenial,
            "years_on_aed": years_on_aed,
        })
    return result


# ─── Public API ─────────────────────────────────────────────────────────────

def overview():
    patients = _get_patients()
    n = max(len(patients), 1)
    seizure_free_n = sum(1 for p in patients if p["seizure_control"] == "Seizure-free")
    drug_resistant_n = sum(1 for p in patients if p["seizure_control"] == "Drug-resistant")
    photosensitive_n = sum(1 for p in patients if p["photosensitive"])
    catamenial_n = sum(1 for p in patients if p["catamenial"])
    avg_onset = round(sum(p["onset_age_years"] for p in patients) / n, 1)

    gene_counts = {}
    for p in patients:
        k = p["genetic_variant"]
        gene_counts[k] = gene_counts.get(k, 0) + 1
    gene_distribution = sorted(
        [{"gene": k, "count": v, "pct": round(v / n * 100)} for k, v in gene_counts.items()],
        key=lambda x: -x["count"]
    )

    aed_counts = {}
    for p in patients:
        k = p["aed_regimen"]
        aed_counts[k] = aed_counts.get(k, 0) + 1
    aed_use = [{"regimen": k, "n_patients": v} for k, v in sorted(aed_counts.items(), key=lambda x: -x[1])]

    control_counts = {}
    for p in patients:
        k = p["seizure_control"]
        control_counts[k] = control_counts.get(k, 0) + 1
    control_distribution = [{"status": k, "count": v} for k, v in sorted(control_counts.items(), key=lambda x: -x[1])]

    return {
        "syndrome": "Juvenile Myoclonic Epilepsy (JME)",
        "icd10": "G40.309",
        "prevalence": "5–10% of all epilepsies; 18–26% of generalised epilepsies; peak onset 12–18 years",
        "drug_resistance_rate": "10–15% drug-resistant; 80% seizure-free on appropriate AED + lifestyle modification",
        "updated": str(date.today()),
        "cohort_size": n,
        "avg_onset_age_years": avg_onset,
        "kpis": [
            {"label": "Cohort (JME)", "value": str(n), "color": "#7c3aed"},
            {"label": "Seizure-Free", "value": f"{seizure_free_n} ({round(seizure_free_n/n*100)}%)", "color": "#16a34a"},
            {"label": "Drug-Resistant", "value": f"{drug_resistant_n} ({round(drug_resistant_n/n*100)}%)", "color": "#dc2626"},
            {"label": "Photosensitive", "value": f"{photosensitive_n} ({round(photosensitive_n/n*100)}%)", "color": "#f59e0b"},
            {"label": "Catamenial (F)", "value": f"{catamenial_n}", "color": "#0284c7"},
            {"label": "Avg Onset Age", "value": f"{avg_onset}y", "color": "#9333ea"},
        ],
        "gene_distribution": gene_distribution,
        "aed_use": aed_use,
        "control_distribution": control_distribution,
        "seizure_types": _SEIZURE_TYPES,
        "triggers": _TRIGGERS,
        "clinical_alerts": [
            "CONTRAINDICATED — Carbamazepine / Oxcarbazepine / Phenytoin / Vigabatrin / Tiagabine: AGGRAVATE myoclonic jerks and may precipitate absence status epilepticus",
            "Lamotrigine CAUTION: may worsen myoclonic jerks in 10–20% of JME patients — monitor closely and reduce if worsening",
            "VALPROATE REMS: mandatory every-6-month pregnancy prevention counselling for ALL women of childbearing age — document in chart",
            "LIFELONG TREATMENT: 80% of JME patients relapse within 1–2 years of AED withdrawal — counsel patient at diagnosis",
            "DRIVING: myoclonic jerks ALONE may disqualify driving — 12 months seizure-free (including jerks) required before driving licence",
            "SLEEP DEPRIVATION is the #1 trigger (90%) — minimum 7–9 hours/night; alarm no earlier than 07:00; avoid consecutive all-nighters",
        ],
    }


def breakdown():
    patients = _get_patients()
    return {
        "patients": patients,
        "genetics": _GENETICS,
        "seizure_types": _SEIZURE_TYPES,
        "triggers": _TRIGGERS,
        "treatments": _TREATMENTS,
        "contraindications": _CONTRAINDICATIONS,
        "aed_monitoring": _AED_MONITORING,
        "lifecycle": _LIFECYCLE,
        "standards": _STANDARDS,
        "thresholds": _THRESHOLDS,
        "references": _REFERENCES,
    }


def definitions():
    return {
        "concepts": _DEFINITIONS,
        "standards": _STANDARDS,
        "thresholds": _THRESHOLDS,
        "references": _REFERENCES,
        "updated": str(date.today()),
    }
