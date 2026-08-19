"""
SGSH Epilepsy — Mucopolysaccharidosis Type IIIA (Sanfilippo Syndrome A)
========================================================================
40-patient cohort · SGSH (17q25.3) · Autosomal Recessive (AR) biallelic LOF
SGSH encodes heparan sulfate sulfamidase (N-sulphoglucosamine sulphohydrolase,
  sulphamidase, 502 aa, ~56 kDa):
  SGSH cleaves the N-sulfate group from N-sulfoglucosamine (GlcNS) residues at
  the non-reducing end of heparan sulfate (HS) chains in lysosomes.
  SGSH LOF → incomplete HS degradation → lysosomal HS accumulation → CNS-dominant
  progressive neurodegeneration; MPS IIIA / Sanfilippo Syndrome Type A.

DISEASE — MPS IIIA / SANFILIPPO SYNDROME TYPE A (OMIM 252900):
  Incidence ~1:100,000 live births in NW Europe (HIGHER than MPS IIIB; MPS IIIA
  is the MOST COMMON MPS III subtype globally — ~52% of all MPS III in Europe).
  Most severe neurological course among all MPS-III subtypes (faster decline than
  MPS IIIB, IIIC, IIID).
  Behavioral regression onset age 2–4 years (earlier than MPS IIIB) → cognitive
  decline → motor deterioration → death typically in the second decade.
  Epilepsy: 85–90% lifetime prevalence — HIGHEST among all MPS-III subtypes.

PATHOGNOMONIC FEATURES:
  (1) URINE HEPARAN SULFATE (HS) — QUANTITATIVE (PATHOGNOMONIC FIRST-LINE):
      Urine GAG quantification (dimethylmethylene blue, DMMB) + HS-specific HPLC/
      tandem-MS → elevated isolated HS with normal/low CS, DS, KS.
      MPS IIIA: pure HS elevation (same pattern as IIIB/C/D — enzyme confirmation
      essential to distinguish subtype A from B/C/D).
  (2) LEUKOCYTE SGSH ENZYME ACTIVITY <1% CONTROL (CONFIRMATORY):
      Heparin substrate or 4-methylumbelliferyl assay; MPS III enzyme panel
      (SGSH + NAGLU + HGSNAT + GNS) confirms type A specifically.
      SGSH activity <1% in leukocytes = diagnostic for MPS IIIA.
  (3) BIALLELIC SGSH VARIANTS (MOLECULAR CONFIRMATION):
      WES/WGS; >150 pathogenic variants; p.R206P (c.617G>C) most common
      Caucasian founder worldwide (~25% alleles); p.S298P NW European founder;
      compound heterozygous most common in non-consanguineous populations.
      Genotype–phenotype: null/null → fastest, most severe decline.
  (4) BRAIN MRI — WHITE MATTER CHANGES (SUPPORTIVE):
      Periventricular and subcortical white matter T2 hyperintensity + progressive
      cerebral atrophy; identical MRI pattern to MPS IIIB — CANNOT distinguish
      IIIA from IIIB by MRI alone; enzyme confirmation ESSENTIAL.

EPILEPSY — SGSH-SPECIFIC:
  Prevalence: 85–90% lifetime (HIGHEST in all MPS-III subtypes); onset late
  childhood (7–12 years typical); MOST SEVERE neurological disease in MPS III.
  Seizure types: GTCS (80%), myoclonic (60%), tonic (45%), atypical absence (30%),
  epileptic spasms (late, 15%).
  Drug resistance: 55–65% (higher than MPS IIIB — most severe CNS HS burden).
  PME-like pattern: action myoclonus + cognitive decline + EEG generalised SW;
  HS accumulation mechanism differs from sphingolipid PME.
  EEG: generalised slow spike-wave, background slowing, multifocal sharp waves;
  photosensitivity 15–25%; PME-like pattern present from mid-childhood.

DISEASE-MODIFYING THERAPY: NO APPROVED THERAPY (2026):
  No ERT approved (sulphamidase enzyme delivery to CNS limited by BBB — ongoing
  intrathecal/intracerebroventricular approaches under investigation).
  Gene therapy: OAV-101 (tralesinagene aparvovec, LYS-SAF302) — intracranial
  AAV10 vector, Lysogene; ACMENA Phase II/III trial (NCT02716246); 4-site
  intracranial injection; promising CSF HS reduction; pre-symptomatic preferred.
  SRT: miglustat off-label, genistein — no RCT support for either.
  HSCT: NOT RECOMMENDED for MPS IIIA — neurological decline continues post-HSCT
  (unlike MPS I/II where HSCT is beneficial); occasional pre-symptomatic NBS
  cases only discussed individually.

DRUG SAFETY (AED-SPECIFIC):
  CBZ/OXC: CAUTION (not absolute CI) — subclinical axonal neuropathy in late-stage
    MPS IIIA; hyponatraemia risk (SIADH); sedation compounds cognitive decline;
    monitor NCS if used >6 months; different from MLD/ARSA where absolute CI.
  VPA: SAFE (HS pathway, not mitochondrial); POLG1 exclusion MANDATORY
    (CPIC Level A — MPS IIIA clinical phenotype overlaps POLG/MERRF superficially);
    dual benefit: anti-seizure + mood stabiliser in MPS IIIA behavioral phase.
  VGB: RELATIVE CI — visual field monitoring impossible in severe ID; retinal
    toxicity cannot be monitored → relative CI; functional vision quarterly.
  Typical Antipsychotics: HIGH RISK — MPS IIIA has the MOST SEVERE behavioral
    phase among all MPS; haloperidol/chlorpromazine → EPS + markedly lowered
    seizure threshold; atypical antipsychotics MANDATORY (risperidone ≤1 mg/day,
    aripiprazole) for behavioral symptoms.
  PHT/Fosphenytoin: AVOID (relative CI) — IV LEV preferred in SE; PHT enzyme
    induction disrupts antipsychotic and melatonin levels; peripheral neuropathy
    additive in late-stage disease; fosphenytoin cardiac risk.
  Melatonin: STRONGLY INDICATED — MPS IIIA has the MOST SEVERE sleep disorder
    of all MPS-III subtypes (hypothalamic HS accumulation from age 2–4 years);
    circadian disruption is often the first presenting symptom before behavioral
    regression; melatonin 2–10 mg at bedtime, Level B.
"""

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

GENE        = "SGSH (N-sulphoglucosamine sulphohydrolase / heparan sulfate sulfamidase)"
LOCUS       = "17q25.3"
OMIM        = "605270 (SGSH gene); 252900 (MPS IIIA / Sanfilippo Syndrome A)"
INHERITANCE = "Autosomal Recessive (AR) — biallelic LOF; null/null → fastest, most severe course"
COHORT_SIZE = 40
COLOR       = "#1b5e20"   # dark green — distinct from NAGLU teal

ETIOLOGIES = [
    {
        "name": "Compound Heterozygous — p.R206P + Null",
        "pct": 32,
        "onset": "Early childhood — rapid (2–4 years behavioral onset)",
        "notes": (
            "Most common genotype in Caucasian non-consanguineous populations; p.R206P (c.617G>C) — "
            "most common worldwide founder SGSH variant (~25% alleles) paired with a truncating "
            "(frameshift/nonsense) null allele; SGSH activity <2% leukocytes; "
            "severe rapid behavioral regression (hyperactivity, aggression, sleep disorder) by age 2–4; "
            "language loss by 5–7 years; seizure onset 7–12 years in 88%; drug resistance 60%; "
            "fastest disease course in compound het subgroup; earliest death in second decade."
        ),
        "key_finding": "p.R206P + null compound het; SGSH <2%; rapid neurodegeneration; seizures 88%; DRE 60%",
    },
    {
        "name": "Homozygous p.R206P Caucasian Founder",
        "pct": 22,
        "onset": "Childhood (3–6 years behavioral onset); slightly attenuated vs null/null",
        "notes": (
            "p.R206P (c.617G>C) homozygous — most prevalent SGSH pathogenic variant globally; "
            "1–3% residual SGSH activity in homozygotes; slightly attenuated neurological progression "
            "vs biallelic null or compound het with null; cognitive plateau phase may extend ~1 year; "
            "seizures develop in 80% of patients; language regression age 5–8 years; "
            "NBS increasingly identifies this founder variant; behavioral onset typically 3–5 years."
        ),
        "key_finding": "p.R206P Caucasian founder (homozygous); SGSH 1–3% residual; moderately severe; seizures 80%",
    },
    {
        "name": "Biallelic Null — Rapid Severe",
        "pct": 20,
        "onset": "Severe early childhood (2–3 years behavioral onset); fastest decline in MPS IIIA",
        "notes": (
            "Biallelic frameshift, nonsense, or large deletion variants; SGSH activity undetectable (<1%); "
            "most severe and rapidly progressive MPS IIIA phenotype; behavioral regression by age 2–3 years; "
            "seizure onset 5–9 years in 92%; drug-resistant epilepsy in 68%; "
            "non-ambulatory by age 8–12 years; earliest death in entire MPS IIIA spectrum; "
            "NBS or sibling cascade detection offers the only pre-symptomatic gene therapy window."
        ),
        "key_finding": "Biallelic null; SGSH undetectable; fastest MPS IIIA decline; seizures 92%; DRE 68%",
    },
    {
        "name": "p.S298P European Founder (Compound Het or Hom)",
        "pct": 16,
        "onset": "Childhood (3–7 years behavioral onset); moderately severe",
        "notes": (
            "p.S298P (NW European / Dutch enriched founder variant); homozygous or compound het with "
            "p.R206P or other pathogenic variant; moderately severe phenotype; SGSH activity 1–5%; "
            "behavioral regression age 3–6 years; seizures 82% of patients; drug resistance 52%; "
            "distinguishable from p.R206P homozygotes only by genotyping — similar clinical course; "
            "NBS important to identify before symptom onset."
        ),
        "key_finding": "p.S298P NW European/Dutch founder; moderately severe; seizures 82%; DRE 52%",
    },
    {
        "name": "Attenuated — Missense/Missense (Residual Activity)",
        "pct": 10,
        "onset": "Late childhood / adolescence (5–12 years behavioral onset); extended course",
        "notes": (
            "Biallelic missense variants preserving 5–20% residual SGSH activity; attenuated rare phenotype "
            "(<10% of MPS IIIA); extended plateau phase; seizures in 55%; intellectual disability "
            "initially mild-moderate; may survive into the third decade; "
            "HIGHEST misdiagnosis risk — initial labels of ADHD, autism spectrum disorder, or "
            "psychiatric disease common; potentially highest benefit from future gene therapy "
            "given preserved CNS substrate at diagnosis."
        ),
        "key_finding": "Residual SGSH 5–20%; attenuated course; misdiagnosis risk highest; seizures 55%",
    },
]

SEIZURE_TYPES = [
    {
        "type": "Generalised Tonic-Clonic (GTCS)",
        "pct": 80,
        "subtype": (
            "Most common seizure type; onset late childhood (7–12 years) as HS accumulation progresses; "
            "LEV and VPA first-line; drug resistance 55–65% overall (highest in MPS-III series); "
            "EEG: generalised irregular spike-wave with background slow delta; "
            "febrile seizure clusters common (3–5 GTCS in 24 hours during intercurrent illness); "
            "photosensitivity co-trigger in 15–25%"
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "pct": 60,
        "subtype": (
            "Action myoclonus (cortical) — highest prevalence in MPS-IIIA vs other MPS-III subtypes; "
            "triggered by voluntary movement, startle (startle component), and photic stimulation; "
            "PME-like syndrome in severe cases (action myoclonus + dementia + EEG generalised SW); "
            "LEV most effective; piracetam Level C adjunct (2.4–4.8 g/day in adults); "
            "distinguish from MERRF/sphingolipid PME by urine HS and SGSH enzyme assay"
        ),
    },
    {
        "type": "Tonic Seizures",
        "pct": 45,
        "subtype": (
            "Tonic posturing; nocturnal predominance; drug-resistant in 60% of tonic-type; "
            "falls and injury risk — padding, helmet use; clobazam PRN nocturnal clusters; "
            "rufinamide Level C for tonic-atonic cluster sequences; "
            "EEG: electrodecremental response + high-amplitude tonic discharge; "
            "drop attacks warrant urgent epilepsy specialist review"
        ),
    },
    {
        "type": "Atypical Absence / Staring Spells",
        "pct": 30,
        "subtype": (
            "Atypical slow spike-wave (<3 Hz); clinically indistinguishable from cognitive fluctuation "
            "of advancing MPS IIIA dementia without video-EEG; "
            "video-EEG MANDATORY to distinguish ictal atypical absence from disease-related cognitive dips; "
            "ethosuximide NOT indicated (atypical SW ≠ typical 3Hz absence; myoclonic + tonic components); "
            "VPA adjunct covers atypical absence + myoclonus + GTCS spectrum"
        ),
    },
    {
        "type": "Epileptic Spasms (Late-onset)",
        "pct": 15,
        "subtype": (
            "Late-onset spasm clusters (age 8–15 years) in severe MPS IIIA; not infantile hypsarrhythmia "
            "pathway — late HS burden disrupts cortical inhibitory networks differently; "
            "ACTH trial Level C (late-onset, small case series); vigabatrin RELATIVE CI "
            "(visual field monitoring impossible in severe ID); "
            "clobazam + VPA adjunct approach; distinguish from myoclonic clusters by EEG"
        ),
    },
]

TRIGGERS = [
    {
        "trigger": "Sleep deprivation / disrupted sleep (circadian dysfunction)",
        "pct": 90,
        "notes": (
            "CARDINAL TRIGGER — MPS IIIA has the MOST SEVERE sleep disorder of ALL MPS-III subtypes: "
            "hypothalamic HS accumulation disrupts SCN circadian pacemaker from age 2–4 years; "
            "sleep disorder often presents BEFORE behavioral regression (earliest symptom); "
            "severe sleep fragmentation → night waking, reversed sleep-wake cycle → seizure clusters; "
            "melatonin 2–10 mg at bedtime STRONGLY INDICATED (Level B); "
            "sleep-EEG: abnormal architecture (↓slow-wave sleep, ↓REM, fragmented NREM); "
            "treating sleep disorder reduces seizure frequency AND behavioral agitation — dual benefit; "
            "caregiver burnout from sleep disorder is the PRIMARY burden reported by MPS IIIA families"
        ),
    },
    {
        "trigger": "Febrile illness / intercurrent infection",
        "pct": 75,
        "notes": (
            "Fever lowers seizure threshold in HS-laden CNS; "
            "MPS IIIA patients are immunocompetent (no organomegaly-based immunosuppression unlike "
            "MPS I/II/VI) but recurrent otitis media + sinusitis common (GAG-filled macrophages "
            "in mucous membranes → mucosal barrier dysfunction); "
            "febrile seizure clusters may occur at modest temperatures (38°C); "
            "rescue: buccal midazolam; paracetamol aggressive fever management; "
            "MPS IIIA families should have written febrile illness action plan"
        ),
    },
    {
        "trigger": "Behavioral agitation / emotional stress",
        "pct": 70,
        "notes": (
            "MPS IIIA has the MOST SEVERE behavioral phase of all MPS-III subtypes: "
            "extreme hyperactivity, aggression, impulsivity, self-injurious behavior; "
            "autonomic arousal during agitation → lowers seizure threshold → seizure cluster; "
            "interlinked: treating sleep disorder ↓ agitation ↓ seizure frequency (virtuous cycle); "
            "behavioral medications: atypical antipsychotics ONLY (risperidone ≤0.5–1 mg/day, "
            "aripiprazole); typical antipsychotics HIGH RISK — must not prescribe in MPS IIIA; "
            "VPA has secondary mood-stabiliser benefit in MPS IIIA behavioral phase"
        ),
    },
    {
        "trigger": "Missed AED dose",
        "pct": 62,
        "notes": (
            "Complex polypharmacy regimen (AEDs + melatonin + behavioral medications); "
            "severe intellectual disability + extreme behavioral disorder in MPS IIIA → "
            "medication adherence dependent entirely on caregiver; "
            "simplified regimen essential (once-daily AEDs preferred — zonisamide advantage); "
            "liquid formulations for dysphagia in late-stage MPS IIIA; "
            "written medication schedule + caregiver training essential; "
            "pill-counting + monthly pharmacy review to detect adherence gaps"
        ),
    },
    {
        "trigger": "Physical exertion / action myoclonus trigger",
        "pct": 55,
        "notes": (
            "Cortical action myoclonus triggered by voluntary movement and physical exertion; "
            "most prominent in MPS IIIA (highest myoclonus prevalence in MPS-III); "
            "startle component to myoclonus — sudden sounds also trigger; "
            "physiotherapy modification: balance training + gentle movement protocols; "
            "protective equipment (helmet, padded vest) for falls from myoclonus + tonic drop attacks; "
            "distinguish ictal myoclonic events from movement artifact on long-term EEG monitoring"
        ),
    },
    {
        "trigger": "Photosensitivity (light stimulation)",
        "pct": 22,
        "notes": (
            "Photo-paroxysmal response on EEG in 15–25% of MPS IIIA patients (IPS testing); "
            "VPA is the most effective AED for photosensitivity — also covers GTCS + myoclonus; "
            "standard photosensitivity precautions: polarised glasses, screen filters, "
            "TV viewing distance >2 m; "
            "EEG-IPS testing recommended at diagnosis; re-test if clinical suspicion of "
            "photic-triggered seizures; photosensitivity may wane in late-stage disease"
        ),
    },
    {
        "trigger": "Antipsychotic medication (typical) initiation/change",
        "pct": 42,
        "notes": (
            "Behavioral phase of MPS IIIA (most severe in MPS-III series) frequently prompts "
            "antipsychotic prescription by non-specialist clinicians; "
            "CRITICAL RISK: typical antipsychotics (haloperidol, chlorpromazine, pimozide) → "
            "markedly lower seizure threshold + EPS + QTc prolongation in MPS IIIA; "
            "ATYPICAL ANTIPSYCHOTICS MANDATORY: risperidone ≤0.5–1 mg/day, aripiprazole; "
            "any antipsychotic initiation → EEG monitoring for 4–6 weeks; "
            "educate all prescribers (paediatricians, GPs) about typical antipsychotic HIGH RISK in MPS IIIA"
        ),
    },
]

TREATMENTS = [
    {
        "treatment": "Levetiracetam (LEV)",
        "level": "B",
        "indication": (
            "GTCS, myoclonic, focal seizures; broad-spectrum first-line in MPS IIIA; "
            "IV formulation preferred over IV PHT for SE management; "
            "renal excretion (safe in hepatic disease); all SGSH phenotypes; "
            "preferred in advanced disease (no hepatic metabolism, minimal enzyme interactions)"
        ),
        "mechanism": (
            "SV2A synaptic vesicle protein modulation; reduces cortical myoclonus + generalised bursts; "
            "no CYP enzyme interactions — critical in MPS IIIA polypharmacy (antipsychotics + melatonin + VPA); "
            "rapid oral + IV titration possible"
        ),
        "monitoring": (
            "Behavioural side-effects (irritability 10–20%) — CRITICAL in MPS IIIA where behavioral "
            "disorder is the primary symptom; LEV irritability easily misattributed to disease progression; "
            "renal dose adjustment (eGFR); CBC annually; psychiatric review at each visit; "
            "if LEV worsens agitation: add clobazam 5 mg or switch to ZNS/VPA"
        ),
        "caution": (
            "Behavioral side-effects may aggravate MPS IIIA agitation + aggression in behavioral phase; "
            "worst case: LEV psychosis-like agitation — switch to ZNS if dose reduction insufficient; "
            "IV LEV preferred over IV PHT/fosphenytoin in status epilepticus — PHT peripheral neuropathy "
            "additive; cardiac arrhythmia risk with IV fosphenytoin in progressive CNS disease"
        ),
    },
    {
        "treatment": "Valproic Acid (VPA)",
        "level": "B",
        "indication": (
            "GTCS + myoclonic + tonic seizures; photosensitivity; atypical absence component; "
            "POLG1 exclusion MANDATORY (CPIC Level A) before initiation; "
            "dual benefit: antiseizure + mood stabiliser in MPS IIIA behavioral phase; "
            "photosensitivity pattern — VPA most effective single agent"
        ),
        "mechanism": (
            "Sodium channel modulation + GABA augmentation + T-type calcium channel inhibition; "
            "anti-myoclonic (most effective for cortical myoclonus); HS pathway — no mechanistic conflict; "
            "mood stabiliser component (GABA augmentation) provides secondary behavioral benefit in "
            "MPS IIIA aggressive/hyperactive phase; VPA serum level target 50–100 mg/L"
        ),
        "monitoring": (
            "LFT every 3 months; ammonia if encephalopathy (distinguish VPA hyperammonaemia from "
            "HS encephalopathy — different treatment); POLG1 exclusion MANDATORY before initiation; "
            "VPPP monitoring in females ≥12 years (teratogenicity counselling); "
            "weight gain monitoring (additive with behavioral medication weight gain in MPS IIIA)"
        ),
        "caution": (
            "POLG1 exclusion mandatory — MERRF is key MPS IIIA phenocopy (PME + mitochondrial); "
            "VPA hyperammonaemia: check ammonia if acute encephalopathy on VPA; "
            "L-carnitine supplementation if VPA-hyperammonaemia confirmed; "
            "liver failure risk in POLG1 carriers → mandatory exclusion at diagnosis"
        ),
    },
    {
        "treatment": "Clobazam (CLB)",
        "level": "B",
        "indication": (
            "ADJUNCT for seizure clusters, tonic seizures, and nocturnal rescue; "
            "sleep-associated nocturnal tonic clusters (clobazam 5–10 mg at bedtime); "
            "behavioral agitation overlap with seizure clusters; "
            "STATUS EPILEPTICUS: buccal midazolam first-line rescue in community"
        ),
        "mechanism": (
            "1,5-benzodiazepine; GABA-A positive allosteric modulator; "
            "longer half-life than clonazepam; less sedating than diazepam; "
            "N-desmethylclobazam (active metabolite) — CYP2C19 polymorphism (slow metabolisers "
            "have 5× higher N-desmethyl-CLB exposure)"
        ),
        "monitoring": (
            "Sedation (additive with antipsychotics, melatonin, clonidine in MPS IIIA polypharmacy); "
            "PARADOXICAL AGITATION documented in MPS IIIA — reduce dose if agitation worsens; "
            "tolerance development (taper to lowest effective dose after initial seizure control); "
            "CYP2C19 genotyping if unexpected high sedation or toxicity"
        ),
        "caution": (
            "Paradoxical agitation documented in SGSH/MPS IIIA — reduce dose or withdraw if agitation "
            "worsens acutely after clobazam initiation; "
            "sedation additive with melatonin + clonidine + antipsychotic polypharmacy; "
            "abrupt withdrawal → seizure cluster; taper over 4–6 weeks; "
            "respiratory depression risk in late-stage MPS IIIA with bulbar dysfunction"
        ),
    },
    {
        "treatment": "Zonisamide (ZNS)",
        "level": "C",
        "indication": (
            "Adjunct in drug-resistant epilepsy (DRE); myoclonic component; "
            "once-daily dosing advantage (adherence benefit in MPS IIIA — simplified regimen); "
            "useful if LEV behavioural side-effects preclude continuation; "
            "carbonic anhydrase activity → metabolic acidosis monitoring required"
        ),
        "mechanism": (
            "Sodium + T-type calcium channel blockade; carbonic anhydrase inhibition; "
            "modulates dopaminergic/serotonergic pathways (potentially relevant to MPS IIIA behavioral "
            "symptoms); once-daily dosing simplifies adherence in severe intellectual disability"
        ),
        "monitoring": (
            "Metabolic acidosis (serum bicarbonate 3-monthly); renal stones (hydration counselling); "
            "OLIGOHIDROSIS / HYPERTHERMIA — HIGH RISK in non-verbal MPS IIIA patients "
            "(cannot report heat intolerance); caregiver temperature monitoring MANDATORY; "
            "weight loss (additive with late-stage disease-related weight decline)"
        ),
        "caution": (
            "Oligohidrosis (reduced sweating → hyperthermia) in non-verbal MPS IIIA patients — "
            "caregiver must monitor body temperature, especially in warm environments; "
            "anorexia worsens disease-related dysphagia in late-stage MPS IIIA; "
            "metabolic acidosis may impair cognition (difficult to assess in advanced ID)"
        ),
    },
    {
        "treatment": "Piracetam",
        "level": "C",
        "indication": (
            "Action myoclonus — UNIQUE INDICATION for MPS IIIA (highest myoclonus prevalence); "
            "PME-like cortical myoclonus; adjunct to LEV; "
            "2.4–4.8 g/day in adults; 50–100 mg/kg/day in children; "
            "renal monitoring required; not an AED (no anti-GTCS/tonic effect)"
        ),
        "mechanism": (
            "Cyclic GABA derivative; unknown precise mechanism; enhances cortical excitability "
            "modulation at AMPA receptors; myoclonic suppression in PME; "
            "no enzyme interactions; renal excretion; "
            "does NOT address tonic, GTCS, or absence components — adjunct only"
        ),
        "monitoring": (
            "Renal function (eGFR) — dose reduction required in renal impairment; "
            "clinical myoclonus assessment (motor diary, action tremor scoring); "
            "no routine plasma level monitoring; reassess benefit at 3 months; "
            "discontinue if no myoclonus benefit at maximum tolerated dose"
        ),
        "caution": (
            "Not a broad-spectrum AED — piracetam does NOT protect against GTCS/tonic; "
            "always combine with LEV or VPA for full seizure type coverage in MPS IIIA; "
            "behavioral agitation occasionally worsened (discontinue if significant); "
            "renal dose adjustment essential (eGFR <60: reduce dose by 50%)"
        ),
    },
    {
        "treatment": "Melatonin (sleep-seizure cluster prevention)",
        "level": "B",
        "indication": (
            "STRONGLY INDICATED in ALL MPS IIIA patients — not an AED but seizure cluster prevention "
            "via sleep normalisation; MPS IIIA has MOST SEVERE sleep disorder in MPS-III series; "
            "2–10 mg at bedtime; prolonged-release preferred for sleep maintenance; "
            "sleep disorder often precedes behavioral regression — melatonin should be initiated early"
        ),
        "mechanism": (
            "MT1/MT2 receptor agonism in suprachiasmatic nucleus (SCN — hypothalamic circadian pacemaker); "
            "resynchronises circadian rhythm disrupted by hypothalamic HS accumulation in MPS IIIA; "
            "reduces sleep fragmentation → reduces nocturnal seizure clusters secondary to sleep deprivation; "
            "no dependency, no withdrawal seizures, no worsening of behavioral symptoms"
        ),
        "monitoring": (
            "Sleep diary (caregiver-reported: nights per week of waking, estimated total sleep hours); "
            "dose titration from 2 mg → 10 mg based on sleep diary response; "
            "prolonged-release formulation preferred (matches sleep-maintenance problem of MPS IIIA); "
            "no hepatic/renal monitoring required; no plasma level monitoring"
        ),
        "caution": (
            "Not a substitute for AED optimisation; "
            "daytime sedation at >10 mg — split dosing not recommended; titrate carefully; "
            "high-dose (>20 mg) — no safety signal but no evidence; avoid; "
            "mild CYP1A2 interaction with fluvoxamine — rarely co-prescribed in MPS IIIA"
        ),
    },
    {
        "treatment": "OAV-101 / LYS-SAF302 Gene Therapy (investigational — intracranial delivery)",
        "level": "Investigational",
        "indication": (
            "ACMENA Phase II/III (NCT02716246 — intracranial AAV10; Lysogene); "
            "OAV-101 (tralesinagene aparvovec) — 4-site intracranial injection delivers SGSH transgene; "
            "CSF HS reduction achieved in early cohorts; pre-symptomatic / early symptomatic preferred; "
            "NOT approved 2026 — patient access via compassionate use / expanded access programmes"
        ),
        "mechanism": (
            "AAV10 serotype with strong CNS neurotropism; SGSH transgene delivered to cortical + "
            "subcortical neurons at 4 stereotactic intracranial sites; "
            "cross-correction mechanism: secreted SGSH taken up by neighbouring cells via M6P receptor; "
            "intracranial delivery circumvents BBB limitation of systemic IV enzyme delivery; "
            "peripheral HS accumulation (non-CNS) may persist — limited peripheral vector effect"
        ),
        "monitoring": (
            "CSF HS quantification (primary biomarker — target normalisation toward age-normal range); "
            "CSF/leukocyte SGSH enzyme activity; neurocognitive battery (Vineland, Bayley, Griffiths); "
            "pre-dose AAV10 antibody titre (pre-existing capsid immunity may exclude); "
            "MRI brain (vector inflammatory reaction, spread); "
            "post-procedural CSF monitoring for infection/haemorrhage"
        ),
        "caution": (
            "NOT APPROVED — enrol in ACMENA trial only; do not administer off-trial; "
            "intracranial procedure: neurosurgical risk (infection, haemorrhage, anaesthesia in MPS); "
            "AAV10 capsid immune response (corticosteroid prophylaxis protocol required); "
            "CNS inflammatory response post-vector delivery (CSF pleocytosis + protein monitoring); "
            "best outcome pre-symptomatic — advocate NBS + early referral to gene therapy centre"
        ),
    },
    {
        "treatment": "HSCT (NOT recommended in established MPS IIIA)",
        "level": "D (not recommended)",
        "indication": (
            "HSCT NOT recommended for MPS IIIA after symptoms appear — neurological decline "
            "continues post-HSCT in MPS III (unlike MPS I/II where HSCT halts neurodegeneration); "
            "occasional discussion in pre-symptomatic NBS-detected infants (case series only, no RCT); "
            "always discuss with specialist MPS + transplant team before considering; "
            "do NOT extrapolate from MPS I/II HSCT success data"
        ),
        "mechanism": (
            "Donor haematopoietic cells → CNS microglia replacement → enzyme cross-correction; "
            "in MPS IIIA: CNS HS accumulation rate vastly exceeds donor macrophage cross-correction "
            "capacity; unlike MPS I (moderate CNS HS) — MPS IIIA CNS HS burden is overwhelming; "
            "no convincing neurological stabilisation demonstrated in published MPS IIIA HSCT series"
        ),
        "monitoring": "N/A (not recommended for established MPS IIIA)",
        "caution": (
            "HSCT morbidity/mortality not justified in MPS IIIA — no demonstrated neurological benefit; "
            "families may request HSCT based on MPS I success stories — clear, compassionate counselling "
            "essential; direct families to OAV-101 gene therapy trial (NCT02716246) instead; "
            "pre-symptomatic NBS cases: individual specialist discussion only"
        ),
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, Pimozide)",
        "level": "HIGH RISK",
        "reason": (
            "MPS IIIA has the MOST SEVERE behavioral phase in all MPS-III subtypes — clinicians most "
            "likely to be pressured to prescribe antipsychotics; HS accumulation in striatum/basal "
            "ganglia → dopaminergic dysfunction → extreme EPS sensitivity (tardive dyskinesia, dystonia, "
            "rigidity); significant seizure threshold lowering; QTc prolongation risk; "
            "atypical antipsychotics mandatory: risperidone ≤0.5–1 mg/day, aripiprazole"
        ),
        "safe_alternative": "Risperidone ≤0.5–1 mg/day; aripiprazole; melatonin for sleep; VPA for mood stabilisation",
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin",
        "level": "AVOID / RELATIVE CI (use IV LEV in SE)",
        "reason": (
            "Progressive CNS degeneration → narrow and unpredictable therapeutic window; "
            "CYP2C9/3A4 enzyme induction reduces levels of antipsychotics and melatonin — "
            "both critical medications in MPS IIIA management; "
            "IV PHT infusion risk (cardiac arrhythmia, hypotension) in progressive CNS disease; "
            "PHT peripheral neuropathy additive in late-stage MPS IIIA axonal degeneration; "
            "IV LEV is safer SE option in established MPS IIIA"
        ),
        "safe_alternative": "IV Levetiracetam for SE; LEV/VPA/CLB for chronic management",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "RELATIVE CI",
        "reason": (
            "Visual field testing IMPOSSIBLE in severe intellectual disability (MPS IIIA patients "
            "cannot cooperate with Humphrey automated perimetry); VGB retinal toxicity risk "
            "cannot be monitored safely → relative CI; "
            "functional vision assessment only (optokinetic drum, preferential looking) if VGB used; "
            "no retinal NCL-type ganglion cell pathology (unlike NCL/CLN diseases where absolute CI)"
        ),
        "safe_alternative": "LEV, VPA, CLB; ACTH Level C for late epileptic spasms",
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "level": "CAUTION (not absolute CI)",
        "reason": (
            "Subclinical axonal neuropathy in late-stage MPS IIIA (NCS monitoring required >6 months); "
            "CBZ/OXC aggravate pre-existing neuropathy in advanced disease; "
            "severe sedation compounds cognitive decline (difficult to distinguish drug effect from "
            "disease progression in non-verbal MPS IIIA patients); "
            "hyponatraemia (SIADH) — difficult to detect in non-verbal patients; "
            "enzyme induction reduces antipsychotic + VPA levels; "
            "HLA-B*15:02 MANDATORY before use in SE Asian ancestry (SJS/TEN risk)"
        ),
        "safe_alternative": "LEV first-line; VPA if myoclonic/tonic; CLB adjunct",
    },
    {
        "drug": "Phenobarbital",
        "level": "CAUTION (second-line only)",
        "reason": (
            "Sedation compounds progressive cognitive decline in MPS IIIA — difficult to distinguish "
            "drug-induced cognitive slowing from disease progression; "
            "CYP enzyme induction reduces antipsychotic, VPA, and melatonin levels; "
            "second-line consideration only in LMIC settings where IV LEV is unavailable for SE; "
            "IV phenobarbital is acceptable SE alternative if IV LEV unavailable"
        ),
        "safe_alternative": "IV LEV preferred for SE; LEV/VPA/CLB for chronic epilepsy",
    },
    {
        "drug": "Ethosuximide",
        "level": "NOT INDICATED",
        "reason": (
            "Designed for pure 3Hz typical absence epilepsy; atypical staring spells in MPS IIIA "
            "have mixed mechanism (slow spike-wave + myoclonic + tonic components); "
            "ethosuximide monotherapy → worsening of GTCS, tonic, and myoclonic components; "
            "VPA covers the full MPS IIIA seizure spectrum (absence-like + myoclonus + GTCS)"
        ),
        "safe_alternative": "VPA (full spectrum coverage); CLB adjunct for atypical absence",
    },
]

MONITORING = [
    "Urine GAG quantification (DMMB + HS-specific HPLC-MS) — annual biomarker tracking",
    "Leukocyte SGSH enzyme activity — baseline + response to investigational gene therapy",
    "SGSH gene panel / WES — biallelic variants confirmed at diagnosis; family cascade",
    "Video-EEG with sleep — at diagnosis; 2-hour minimum; annual or after seizure type change",
    "Sleep diary + actigraphy (caregiver-reported) — monthly (sleep disorder is #1 trigger; earliest symptom)",
    "Behavioural assessment (ABAS-3, Vineland-3) — 6-monthly (disease staging + AED impact)",
    "Brain MRI — baseline + every 2 years (white matter progression, cortical atrophy staging)",
    "ECG — baseline + on antipsychotics (QTc prolongation) + if IVP PHT considered",
    "LFT + ammonia — 3-monthly on VPA (hepatic monitoring + hyperammonaemia screen)",
    "VPA trough level — 3-monthly; VPPP compliance annually in females ≥12 years",
    "POLG1 sequencing — MANDATORY before VPA initiation at diagnosis",
    "HLA-B*15:02 — MANDATORY before CBZ/OXC in SE Asian ancestry",
    "Nerve conduction study (NCS) — baseline if CBZ/OXC initiated; 6-monthly if continuing",
    "Serum bicarbonate — 3-monthly on zonisamide (metabolic acidosis monitoring)",
    "Ophthalmology — annual (fundoscopy; no cherry-red spot expected; monitor for retinal deposits)",
    "Body temperature monitoring (caregiver) — daily if on zonisamide (oligohidrosis risk in non-verbal patients)",
    "Dysphagia screen (SALT referral) — 6-monthly in late stage (aspiration pneumonia risk)",
    "Neurocognitive battery — 6-monthly (Griffiths/Bayley/Vineland; gene therapy eligibility + staging)",
]

THRESHOLDS = [
    {"parameter": "SGSH enzyme activity (diagnostic)", "threshold": "<1% control activity", "action": "Diagnostic for MPS IIIA; confirm with biallelic SGSH gene variants"},
    {"parameter": "Urine HS (DMMB)", "threshold": ">2× upper normal for age", "action": "Send MPS III enzyme panel (SGSH, NAGLU, GNS, HGSNAT) — distinguish IIIA from B/C/D"},
    {"parameter": "VPA trough level", "threshold": "50–100 mg/L (350–700 µmol/L)", "action": "If below range: increase dose; if >120 mg/L: reduce (hepatotoxicity risk)"},
    {"parameter": "VPA ALT", "threshold": ">3× ULN on VPA", "action": "Reduce/hold VPA; check ammonia; discuss alternative (LEV, CLB)"},
    {"parameter": "CBZ sodium (SIADH)", "threshold": "<130 mmol/L on CBZ/OXC", "action": "Hold CBZ/OXC; fluid restriction; switch to LEV"},
    {"parameter": "Antipsychotic QTc", "threshold": "QTc >500 ms on antipsychotic", "action": "Reduce/stop antipsychotic; switch to aripiprazole (minimal QTc effect)"},
    {"parameter": "Melatonin dose", "threshold": "Start 2 mg; titrate to 10 mg", "action": "If no sleep improvement at 10 mg: add low-dose clonidine 25 mcg at bedtime"},
    {"parameter": "Seizure cluster (emergency)", "threshold": "≥3 seizures / 24 hours OR any seizure >5 min", "action": "Buccal midazolam rescue; call emergency services if not responsive within 5 min"},
    {"parameter": "Drug resistance definition", "threshold": "Failure of 2 adequate AED trials (ILAE 2010)", "action": "MPS specialist review; OAV-101 gene therapy trial referral (NCT02716246)"},
    {"parameter": "ZNS bicarbonate", "threshold": "<18 mmol/L on zonisamide", "action": "Reduce ZNS dose; bicarbonate supplementation if symptomatic acidosis"},
    {"parameter": "Body temperature on ZNS", "threshold": "≥38.5°C unexplained on zonisamide", "action": "Suspect oligohidrosis; remove ZNS; cool patient; hydration; check creatinine"},
    {"parameter": "SGSH NBS screen", "threshold": "Enzyme activity <10% on DBS newborn screen", "action": "Confirmatory leukocyte SGSH + urine HS + SGSH gene panel; gene therapy referral urgent"},
]

STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE Operational Classification of Seizure Types 2022"},
    {"code": "NICE-NG17", "title": "NICE NG17: Diagnosis and Management of MPS (UK 2016, updated 2022)"},
    {"code": "ACMG-AMP-2015", "title": "ACMG/AMP Variant Classification Standards (SGSH VUS interpretation)"},
    {"code": "CPIC-POLG-2023", "title": "CPIC Guideline: VPA and POLG1 (Level A — mandatory exclusion before VPA)"},
    {"code": "CPIC-HLA-B1502-CBZ", "title": "CPIC Guideline: CBZ and HLA-B*15:02 (Level A — SE Asian ancestry)"},
    {"code": "MHRA-VPPP-2021", "title": "MHRA Valproate Pregnancy Prevention Programme (females ≥12 years)"},
    {"code": "Wijburg-2013", "title": "Wijburg FA et al. MPS IIIA European natural history cohort (J Inherit Metab Dis 2013)"},
    {"code": "Tardieu-2014", "title": "Tardieu M et al. Intracerebral gene therapy (OAV-101) for MPS IIIA — Phase I/II (Sci Transl Med 2014)"},
    {"code": "NCT02716246", "title": "NCT02716246: ACMENA Phase II/III intracranial AAV10 OAV-101 gene therapy for SGSH/MPS IIIA"},
    {"code": "Lavery-2016", "title": "Lavery C et al. MPS IIIA UK natural history cohort (Orphanet J Rare Dis 2016)"},
    {"code": "Ruijter-2008", "title": "Ruijter GJ et al. MPS IIIA genotype-phenotype correlations (J Inherit Metab Dis 2008)"},
    {"code": "WHO-ICF-2019", "title": "WHO ICF Framework for MPS III disability classification"},
    {"code": "ILAE-Genetic-2018", "title": "ILAE Report: Genetic epilepsies — classification and diagnosis"},
]

DEFINITIONS = [
    {"term": "SGSH (heparan sulfate sulfamidase)", "definition": "N-sulphoglucosamine sulphohydrolase; lysosomal enzyme cleaving N-sulfate groups from N-sulfoglucosamine (GlcNS) at the non-reducing end of heparan sulfate chains; 502 aa, ~56 kDa; SGSH gene at 17q25.3; biallelic LOF → MPS IIIA."},
    {"term": "MPS IIIA (Sanfilippo A)", "definition": "Mucopolysaccharidosis type IIIA; SGSH deficiency; most common MPS III subtype (~52% of MPS III in Europe); MOST SEVERE neurological course of all MPS-III subtypes; pure CNS-dominant HS storage; epilepsy 85–90%; no approved disease-modifying therapy (2026)."},
    {"term": "Heparan Sulfate (HS)", "definition": "N-sulfated glycosaminoglycan; CNS-abundant; attached to neuronal cell-surface proteoglycans; SGSH LOF → incomplete HS degradation → lysosomal HS accumulation → neuronal dysfunction + death; urine and CSF HS are primary biomarkers."},
    {"term": "GAG (glycosaminoglycan)", "definition": "Polysaccharide chains degraded in lysosomes; urine HS elevated in all MPS-III subtypes (A/B/C/D); DS, CS, KS normal or low; MPS III enzyme panel required to confirm subtype A (SGSH) versus B (NAGLU), C (HGSNAT), D (GNS)."},
    {"term": "ACMENA trial (NCT02716246)", "definition": "Phase II/III randomised controlled trial of OAV-101 (tralesinagene aparvovec, LYS-SAF302) — intracranial AAV10 gene therapy for MPS IIIA; 4-site stereotactic intracranial injection; primary endpoint CSF HS normalisation; conducted by Lysogene."},
    {"term": "OAV-101 (tralesinagene aparvovec)", "definition": "AAV10 vector encoding human SGSH (heparan sulfate sulfamidase); intracranial delivery to 4 cortical/subcortical sites; circumvents BBB limitation of systemic enzyme delivery; cross-correction potential; ACMENA Phase II/III (NCT02716246)."},
    {"term": "Sleep disorder in MPS IIIA", "definition": "Most severe sleep disorder of all MPS-III subtypes; hypothalamic HS accumulation disrupts SCN circadian pacemaker; onset age 2–4 years — EARLIEST presenting symptom (before behavioral regression); severe sleep fragmentation; reversed sleep-wake cycle; melatonin Level B strongly indicated."},
    {"term": "PME-like phenotype in MPS IIIA", "definition": "Progressive myoclonic epilepsy-like syndrome: action myoclonus + cognitive dementia + generalized spike-wave EEG; MPS IIIA has highest myoclonus prevalence in MPS-III; mechanism is HS-mediated CNS dysfunction (not sphingolipid PME); piracetam Level C for myoclonus adjunct."},
    {"term": "POLG1 exclusion (mandatory before VPA)", "definition": "POLG1 encodes mitochondrial DNA polymerase gamma; POLG1 pathogenic variants → POLG disease (MERRF, Alpers); VPA in POLG disease → fatal hepatotoxicity; MPS IIIA phenocopy of PME/POLG superficially; POLG1 sequencing MANDATORY before VPA in ALL MPS IIIA patients."},
    {"term": "Drug resistance in MPS IIIA", "definition": "Failure of ≥2 adequate AED trials (ILAE 2010); 55–65% DRE rate in MPS IIIA — highest in MPS-III series; driven by broad HS-mediated synaptic dysfunction across multiple receptor systems simultaneously; OAV-101 gene therapy may modify seizure biology by reducing HS burden."},
]

LIFECYCLE = [
    {
        "stage": "Pre-symptomatic (0–2 years)",
        "description": (
            "Newborn screening (NBS) increasingly detects SGSH deficiency via multiplex enzyme assay; "
            "confirmatory leukocyte SGSH + urine HS + SGSH gene panel; "
            "no clinical symptoms at this stage; EARLIEST and highest-benefit gene therapy window; "
            "family cascade genetic counselling; HSCT discussion (evidence very limited for MPS IIIA); "
            "melatonin considered prophylactically in confirmed cases (sleep disorder often first symptom)."
        ),
    },
    {
        "stage": "Behavioral Phase (2–5 years)",
        "description": (
            "First presenting symptoms: severe hyperactivity, aggression, self-injurious behavior, "
            "sleep disorder (circadian disruption, reversed sleep-wake cycle); language regression; "
            "MPS IIIA behavioral phase is the most severe of all MPS-III subtypes; "
            "commonly misdiagnosed as ADHD, autism spectrum disorder, or attachment disorder; "
            "seizures absent in 90% during early behavioral phase; "
            "melatonin initiated for sleep disorder; atypical antipsychotics for behavior "
            "(risperidone — NOT typical antipsychotics)."
        ),
    },
    {
        "stage": "Cognitive Decline Phase (5–10 years)",
        "description": (
            "Progressive intellectual deterioration; language lost completely; comprehension declining; "
            "seizures emerge in majority (75–88%) during this phase — GTCS + myoclonic predominate; "
            "EEG: generalised slow spike-wave; AED initiation (LEV/VPA first-line); "
            "PME-like pattern: action myoclonus + dementia + EEG SW; "
            "SGSH gene therapy referral (NCT02716246 — narrowing window)."
        ),
    },
    {
        "stage": "Motor Deterioration Phase (8–15 years)",
        "description": (
            "Loss of ambulation (mean age 10–14 years — earlier than MPS IIIB); dysphagia onset; "
            "epilepsy peaks in frequency + drug resistance (DRE 55–65%); "
            "tonic seizures + drop attacks emerge; clobazam + rufinamide adjunct; "
            "SALT referral for dysphagia assessment (NG/PEG consideration); "
            "seizure rescue plan essential; non-verbal communication strategies."
        ),
    },
    {
        "stage": "Palliative Phase (12+ years)",
        "description": (
            "Bed-bound; fully dependent care; dysphagia + aspiration pneumonia — primary mortality cause; "
            "epilepsy frequency may stabilise as cortical atrophy reduces excitable tissue mass; "
            "palliative AED approach (comfort and dignity, not seizure freedom target); "
            "hospice planning; advance care planning discussion with family; SUDEP counselling; "
            "death typically in second decade in severe MPS IIIA (earlier than MPS IIIB)."
        ),
    },
]

KEY_CONCEPTS = [
    "MPS IIIA (Sanfilippo A): NO approved disease-modifying therapy (2026) — most severe MPS-III; highest unmet need",
    "SGSH at 17q25.3: biallelic LOF → heparan sulfate sulfamidase deficiency → HS accumulation → CNS neurodegeneration",
    "Most common MPS-III in Europe (~52%); most severe neurological course; earliest death in second decade",
    "Epilepsy prevalence 85–90%: HIGHEST in all MPS-III subtypes; onset 7–12 years; drug resistance 55–65%",
    "Sleep disorder EARLIEST symptom: circadian disruption from age 2–4 years; melatonin Level B — treat early",
    "Typical antipsychotics: HIGH RISK in MPS IIIA — most severe behavioral phase → pressure to prescribe; atypicals mandatory",
    "PHT/Fosphenytoin: avoid in SE; IV LEV preferred; PHT enzyme induction disrupts antipsychotic + melatonin levels",
    "VGB: relative CI — visual field monitoring impossible in severe ID; functional vision only if VGB considered",
    "CBZ/OXC: caution (not absolute CI) — late-stage axonal neuropathy + SIADH; NCS monitoring if used >6 months",
    "POLG1 exclusion MANDATORY before VPA — MPS IIIA phenotype overlaps MERRF/PME superficially",
    "HLA-B*15:02 MANDATORY before CBZ/OXC in SE Asian ancestry — SJS/TEN fatal risk (CPIC Level A)",
    "OAV-101 gene therapy (NCT02716246): intracranial AAV10; ACMENA Phase II/III; enrol pre-symptomatic if NBS detected",
    "HSCT NOT recommended in established MPS IIIA — unlike MPS I/II; neurological decline continues post-transplant",
    "Clobazam paradoxical agitation: documented in MPS IIIA — reduce dose if agitation acutely worsens on CLB",
    "Piracetam Level C for action myoclonus — highest myoclonus prevalence in MPS-III; adjunct to LEV/VPA",
    "p.R206P Caucasian founder (~25% alleles worldwide): most common SGSH variant; WES confirms; NBS detectable",
]

DIFFERENTIAL_DIAGNOSIS = [
    {
        "condition": "MPS IIIB (NAGLU / Sanfilippo B)",
        "distinguishing": "Urine HS identical; NAGLU enzyme (not SGSH) low; NAGLU gene at 17q21.2; clinically indistinguishable from MPS IIIA — ENZYME PANEL mandatory; MPS IIIB slightly less severe; different gene therapy approaches",
    },
    {
        "condition": "MPS IIIC (HGSNAT / Sanfilippo C)",
        "distinguishing": "HGSNAT enzyme low (not SGSH); HS elevated; slower clinical course than IIIA (attenuated forms common); HGSNAT gene at 8p11.21; adult-onset attenuated phenotype more common",
    },
    {
        "condition": "MPS IIID (GNS / Sanfilippo D)",
        "distinguishing": "GNS enzyme low; rarest MPS-III subtype; clinical phenotype similar to IIIA/B; GNS gene at 12q14.3; enzyme panel essential to distinguish from SGSH/NAGLU deficiency",
    },
    {
        "condition": "MPS II (Iduronate-2-Sulfatase / IDS) — X-linked",
        "distinguishing": "HS + DS both elevated (NOT HS alone); X-linked recessive (males predominantly); prominent somatic features (coarse facies, hepatosplenomegaly, cardiac valve disease); ERT available (idursulfase/iduronic acid)",
    },
    {
        "condition": "MERRF / POLG disease (PME, mitochondrial)",
        "distinguishing": "Key exclusion before VPA; lactate elevated; POLG1 variants; mtDNA deletion (muscle biopsy); ragged red fibres; VPA CONTRAINDICATED in POLG; urine HS normal; SGSH enzyme normal",
    },
    {
        "condition": "Autism Spectrum Disorder (ASD) / ADHD",
        "distinguishing": "MPS IIIA behavioral phase misdiagnosed as ASD/ADHD in 50–70%; urine GAG screen distinguishes; SGSH enzyme assay confirms; progressive regression (not static) is RED FLAG for metabolic disease",
    },
    {
        "condition": "NCL (Neuronal Ceroid Lipofuscinosis — CLN2, CLN3)",
        "distinguishing": "Urine HS normal; ceroid/lipofuscin on EM; CLN enzyme panel (TPP1); VGB ABSOLUTE CI in NCL (NOT relative CI as in MPS IIIA); macular cherry-red spot possible in CLN1",
    },
    {
        "condition": "Angelman Syndrome / Rett Syndrome",
        "distinguishing": "Early childhood developmental regression mimics MPS IIIA behavioral phase; urine HS normal; specific genetic abnormalities (UBE3A, MECP2); non-progressive beyond initial regression in Angelman; MPS IIIA is progressive",
    },
]


# ---------------------------------------------------------------------------
# API FUNCTIONS
# ---------------------------------------------------------------------------

def get_overview():
    """Return SGSH/MPS IIIA overview data for /api/sgsh/overview."""
    by_etiology = {e["name"]: e["pct"] for e in ETIOLOGIES}
    by_seizure  = {s["type"]: s["pct"] for s in SEIZURE_TYPES}
    by_trigger  = {t["trigger"]: t["pct"] for t in TRIGGERS}

    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "cohort_size": COHORT_SIZE,
        "color": COLOR,
        "disease_name": "MPS IIIA — Sanfilippo Syndrome Type A",
        "disease_mechanism": (
            "SGSH biallelic LOF → heparan sulfate sulfamidase deficiency → "
            "lysosomal heparan sulfate (HS) accumulation → CNS-predominant progressive "
            "neurodegeneration; behavioral regression (2–4 years) → cognitive decline → "
            "motor failure → death in second decade."
        ),
        "epilepsy_prevalence_pct": 88,
        "epilepsy_onset": "Late childhood — 7–12 years (peaks in cognitive decline phase)",
        "drug_resistance_pct": 60,
        "no_approved_disease_therapy": True,
        "gene_therapy_phase": "Phase II/III (NCT02716246 — ACMENA; OAV-101 intracranial AAV10)",
        "pme_risk": "High — action myoclonus + cognitive decline PME-like pattern; highest myoclonus prevalence in MPS-III",
        "cardinal_seizure_types": ["GTCS", "Myoclonic", "Tonic", "Atypical Absence", "Late Spasms"],
        "cardinal_trigger": "Sleep deprivation (most severe sleep disorder in MPS-III — SCN hypothalamic HS accumulation)",
        "absolute_ci_drugs": ["Typical antipsychotics (EPS + seizure threshold)", "PHT/Fosphenytoin (SE — use IV LEV)"],
        "key_safe_aeds": [
            "Levetiracetam (LEV)",
            "Valproic Acid (VPA — POLG1 excluded)",
            "Clobazam (CLB adjunct)",
            "Zonisamide (ZNS — once-daily adherence)",
            "Piracetam (Level C — action myoclonus)",
        ],
        "key_warning": (
            "NO approved disease-modifying therapy (2026). MOST SEVERE MPS-III — death in second decade. "
            "Enrol in OAV-101 gene therapy trial (NCT02716246). POLG1 exclusion MANDATORY before VPA. "
            "HSCT NOT recommended in established disease. Melatonin strongly indicated — sleep disorder "
            "is earliest symptom and primary seizure trigger."
        ),
        "by_etiology": by_etiology,
        "by_seizure_type": by_seizure,
        "top_triggers": dict(list(by_trigger.items())[:5]),
        "etiologies": ETIOLOGIES,
        "lifecycle": LIFECYCLE,
        "key_concepts": KEY_CONCEPTS,
    }


def get_breakdown():
    """Return SGSH/MPS IIIA detailed breakdown for /api/sgsh/breakdown."""
    import random
    rng = random.Random(42)

    etiology_pool  = ETIOLOGIES
    seizure_pool   = SEIZURE_TYPES
    treatment_pool = TREATMENTS
    drug_resp      = ["Controlled", "Partially controlled", "Drug-resistant"]
    drug_resp_w    = [0.12, 0.28, 0.60]   # MPS IIIA: higher DRE than NAGLU
    stages         = [lc["stage"] for lc in LIFECYCLE]

    patients = []
    for i in range(1, COHORT_SIZE + 1):
        etiology = rng.choices(etiology_pool, weights=[e["pct"] for e in etiology_pool])[0]
        sz_types = [
            s["type"]
            for s in rng.choices(
                seizure_pool,
                k=rng.randint(1, 3),
                weights=[s["pct"] for s in seizure_pool],
            )
        ]
        sz_types = list(dict.fromkeys(sz_types))  # unique, preserve order

        primary_aed = rng.choice(treatment_pool[:6])
        adjunct_aed = rng.choice(treatment_pool[:6])
        while adjunct_aed["treatment"] == primary_aed["treatment"]:
            adjunct_aed = rng.choice(treatment_pool[:6])

        response          = rng.choices(drug_resp, weights=drug_resp_w)[0]
        age               = rng.randint(5, 22)
        onset_age         = rng.randint(6, 13)
        sex               = rng.choice(["M", "F"])
        trigger_primary   = rng.choice(TRIGGERS)
        seizure_freq      = rng.randint(0, 20) if response != "Controlled" else rng.randint(0, 2)
        last_seizure_days = rng.randint(1, 365) if response == "Controlled" else rng.randint(1, 30)
        sleep_disorder    = rng.random() < 0.90   # 90% MPS IIIA sleep disorder
        gt_candidate      = (
            rng.random() < 0.40
            and age <= 10
            and response in ("Partially controlled", "Drug-resistant")
        )
        stage_idx = min(int((i - 1) / 8), len(stages) - 1)

        patients.append({
            "patient_id":           f"SGSH-{i:02d}",
            "age":                  age,
            "sex":                  sex,
            "onset_age":            onset_age,
            "etiology":             etiology["name"],
            "seizure_types":        sz_types,
            "drug_resistance":      response,
            "primary_aed":          primary_aed["treatment"],
            "adjunct_aed":          adjunct_aed["treatment"],
            "trigger_primary":      trigger_primary["trigger"],
            "seizure_freq_monthly": seizure_freq,
            "last_seizure_days":    last_seizure_days,
            "sleep_disorder":       sleep_disorder,
            "gene_therapy_candidate": gt_candidate,
        })

    drug_resistant_n  = sum(1 for p in patients if p["drug_resistance"] == "Drug-resistant")
    sleep_disorder_n  = sum(1 for p in patients if p["sleep_disorder"])
    gt_candidate_n    = sum(1 for p in patients if p["gene_therapy_candidate"])

    return {
        "gene":             GENE,
        "cohort_size":      COHORT_SIZE,
        "etiologies":       ETIOLOGIES,
        "seizure_types":    SEIZURE_TYPES,
        "triggers":         TRIGGERS,
        "treatments":       TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring":       MONITORING,
        "thresholds":       THRESHOLDS,
        "lifecycle":        LIFECYCLE,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "cohort_summary": {
            "total_patients":       COHORT_SIZE,
            "drug_resistant_n":     drug_resistant_n,
            "drug_resistant_pct":   round(drug_resistant_n / COHORT_SIZE * 100, 1),
            "sleep_disorder_n":     sleep_disorder_n,
            "sleep_disorder_pct":   round(sleep_disorder_n / COHORT_SIZE * 100, 1),
            "gene_therapy_candidate_n":   gt_candidate_n,
            "gene_therapy_candidate_pct": round(gt_candidate_n / COHORT_SIZE * 100, 1),
        },
        "patients": patients,
    }


def get_definitions():
    """Return SGSH/MPS IIIA definitions for /api/sgsh/definitions."""
    return {
        "gene":   GENE,
        "locus":  LOCUS,
        "omim":   OMIM,
        "definitions":  DEFINITIONS,
        "key_concepts": KEY_CONCEPTS,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "standards": STANDARDS,
        "diagnostic_algorithm": [
            "Step 1: Urine GAG quantification (DMMB + HS-specific HPLC-MS) — elevated isolated HS; CS/DS/KS normal",
            "Step 2: MPS III enzyme panel (SGSH + NAGLU + HGSNAT + GNS leukocyte assays) — confirm SGSH deficiency",
            "Step 3: Leukocyte SGSH enzyme activity <1% of control — confirms MPS IIIA",
            "Step 4: POLG1 sequencing / mtDNA depletion panel — mandatory exclusion before VPA (PME/MERRF phenocopy)",
            "Step 5: Biallelic SGSH variant confirmation (WES/WGS; screen p.R206P first in Caucasian patients)",
            "Step 6: HLA-B*15:02 genotyping — mandatory before CBZ/OXC in South/SE Asian ancestry (SJS risk)",
            "Step 7: Brain MRI — periventricular T2 hyperintensity + progressive atrophy (supportive; cannot distinguish IIIA vs IIIB)",
            "Step 8: Video-EEG with sleep — seizure type characterisation; PME-like pattern; photo-paroxysmal response; nocturnal clustering",
        ],
        "pharmacological_distinctions": [
            "POLG1 mandatory exclusion before VPA — MPS IIIA is phenocopy of PME/MERRF (not mitochondrial but superficially similar)",
            "IV LEV preferred over IV PHT in status epilepticus — PHT enzyme induction; cardiac + neuropathy risk",
            "Typical antipsychotics HIGH RISK — MPS IIIA has MOST SEVERE behavioral phase; clinician pressure highest; atypicals mandatory",
            "VGB relative CI (not absolute) — visual monitoring impossible in severe ID; functional vision assessment quarterly if used",
            "CBZ/OXC: CAUTION (not absolute CI) — different from MLD/ARSA/GALC where absolute CI; NCS monitoring if used >6 months",
            "Piracetam Level C: MPS IIIA has highest myoclonus prevalence in MPS-III — unique indication; adjunct to LEV",
            "Melatonin Level B: MOST SEVERE sleep disorder in MPS-III; earliest symptom from age 2–4 years; treat early",
            "HSCT NOT recommended for established MPS IIIA — unlike MPS I/II; neurological decline continues post-transplant",
            "OAV-101 (NCT02716246) — intracranial AAV10; ACMENA Phase II/III; enrol pre-symptomatic NBS-detected patients",
            "Clobazam paradoxical agitation — documented in SGSH/MPS IIIA; reduce dose if agitation worsens acutely on CLB",
            "No approved ERT — contrast with MPS I (laronidase), MPS II (idursulfase), MPS IVA (elosulfase), MPS VI (galsulfase)",
            "VPA dual benefit in MPS IIIA: antiseizure + mood stabiliser — ONLY MPS-III where behavioral phase demands both simultaneously",
        ],
    }
