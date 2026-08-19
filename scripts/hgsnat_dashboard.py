"""
HGSNAT Epilepsy — Mucopolysaccharidosis Type IIIC (Sanfilippo Syndrome C)
==========================================================================
40-patient cohort · HGSNAT (8p11.21) · Autosomal Recessive (AR) biallelic LOF
HGSNAT encodes Heparan Alpha-Glucosaminide N-Acetyltransferase (HGSNAT, 666 aa, ~73 kDa):
  HGSNAT is a multi-pass transmembrane lysosomal enzyme that transfers an acetyl
  group from cytosolic acetyl-CoA to the free amino group of the non-reducing
  terminal α-glucosamine residues of heparan sulfate (HS) chains.
  HGSNAT LOF → HS acetylation step blocked → HS accumulation → MPS IIIC / Sanfilippo C.
  HS storage predominantly affects neurons (CNS-dominant LSD); systemic features mild.

DISEASE — MPS IIIC / SANFILIPPO SYNDROME TYPE C (OMIM 252930):
  Incidence ~1:1,400,000 live births (European); 3rd most common MPS III subtype
  (~15–20% of all Sanfilippo cases; after IIIA/SGSH and IIIB/NAGLU).
  Slightly slower neurological progression than MPS IIIA/IIIB; some patients survive
  into 4th decade; behavioral phenotype equally severe in early phase.
  Epilepsy: 70–80% lifetime prevalence (lower than IIIA/IIIB; slower HS accumulation).
  Drug resistance: 40–50% (lower than MPS IIIA/IIIB — residual acetylation in some).

PATHOGNOMONIC FEATURES:
  (1) URINE HEPARAN SULFATE (HS) — QUANTITATIVE (PATHOGNOMONIC FIRST-LINE):
      Urine GAG quantification (DMMB) + HS-specific HPLC/tandem-MS → isolated HS
      elevation (normal/low CS, DS, KS); indistinguishable from MPS IIIA/IIIB/IIID
      on urine alone — enzyme panel MANDATORY to subtype.
  (2) LEUKOCYTE HGSNAT ENZYME ACTIVITY <1–5% CONTROL (CONFIRMATORY):
      Acetyltransferase assay using [14C]-acetyl-CoA or fluorescent substrate;
      technically demanding — specialized LSD labs; pseudodeficiency NOT described
      for HGSNAT. Normal leukocyte HGSNAT in heterozygous carriers (>50% control).
  (3) BIALLELIC HGSNAT VARIANTS (MOLECULAR CONFIRMATION):
      WES/WGS; >70 pathogenic variants; French-Canadian founder p.Arg344Cys
      (~20% Quebec alleles); Dutch founder p.Trp649stop (~15%); Italian founder
      p.Pro524Leu; compound het most common in outbred populations.
      Genotype–phenotype: null/null → faster decline; missense/missense → attenuated.
  (4) BRAIN MRI — WHITE MATTER CHANGES (SUPPORTIVE):
      Periventricular T2 hyperintensity + progressive cerebral atrophy; findings
      identical to MPS IIIA/IIIB on MRI — enzyme/gene confirmation essential.
      No cherry-red spot; no corpus callosum thinning as early feature (unlike MLD).

EPILEPSY — HGSNAT-SPECIFIC:
  Prevalence: 70–80% lifetime; onset typically late childhood (8–14 years);
  later onset than MPS IIIA/IIIB reflecting slower HS accumulation kinetics.
  Seizure types: GTCS (65%), myoclonic (45%), tonic (35%), absence-like (30%),
  epileptic spasms (late, 15%).
  Drug resistance: 40–50% (moderate; lower than MPS IIIA [60%] and IIIB [55%]).
  PME-like pattern: action myoclonus + cognitive decline + EEG generalised SW
  mimics PME but HS mechanism differs from sphingolipidoses.
  EEG: generalised slow spike-wave, background slowing; photosensitivity 15–25%.

DISEASE-MODIFYING THERAPY: NO APPROVED THERAPY (2026):
  No ERT approved (HGSNAT acetyltransferase requires acetyl-CoA from cytosol —
  recombinant ERT cannot reproduce this reaction in lysosomes; ERT replacement
  strategy fundamentally limited).
  Gene therapy: AAV9-HGSNAT intrathecal Phase I — CSF HS reduction reported;
  NCATS-funded preclinical programs (mouse HGSNAT KO models show efficacy).
  SRT: miglustat off-label; genistein negative RCT (all MPS III).
  HSCT: NOT RECOMMENDED for MPS IIIC in established disease — similar to IIIA/IIIB;
  occasional pre-symptomatic NBS-detected cases may be offered; evidence very limited.

DRUG SAFETY (AED-SPECIFIC):
  CBZ/OXC: CAUTION (not absolute CI) — subclinical peripheral neuropathy described
    in late MPS IIIC; less prominent than MLD/GALC/ARSA where ABSOLUTE CI; SIADH
    risk; hyponatraemia monitoring mandatory; not same as sulfatide/galactocerebroside
    demyelinating mechanism.
  VPA: SAFE (HS pathway, not mitochondrial); POLG1 exclusion MANDATORY (CPIC A);
    MPS IIIC phenocopy of PME superficially — POLG/MERRF must be excluded;
    hepatic monitoring + VPA-protein binding in albumin-depleted severe disease.
  VGB: RELATIVE CI — visual field monitoring impossible in severe ID; functional
    vision assessment quarterly if VGB used; photosensitivity 15–25% in HGSNAT.
  Typical Antipsychotics: HIGH RISK — HS-laden basal ganglia → EPS + lowered
    seizure threshold; behavioral phase most severely affected (hyperactivity,
    aggression) → clinical pressure to prescribe; atypical antipsychotics ONLY
    (aripiprazole, risperidone low-dose, olanzapine).
  PHT/Fosphenytoin: AVOID in SE — use IV LEV; PHT enzyme induction disrupts
    antipsychotic/melatonin levels; cardiac risk in degenerating myocardium.
  Melatonin: LEVEL B — sleep disorder 85–90% (cardinal; slightly less severe onset
    than MPS IIIA but equally disabling); sleep-seizure nexus documented in HGSNAT;
    initiate early at behavioral phase onset (circadian disruption 2–5 years).
  Clobazam: PARADOXICAL AGITATION documented in MPS IIIC (same as IIIA/IIIB);
    reduce dose or discontinue if behavioral agitation acutely worsens on CLB.
  HSCT: NOT RECOMMENDED for established MPS IIIC — distinct from MPS I/II policy.

§57.7 HONESTY: All patient data generated from validated clinical distributions
  (HGSNAT/MPS IIIC literature: Ruijter et al. 2008, Fedele et al. 2009,
  Valstar et al. 2010, Crawley et al. 2017, Shapiro et al. 2016).
  No real patient records used; 40-patient cohort is synthetic research simulation.
"""

# ─── GENE / DISEASE CONSTANTS ────────────────────────────────────────────────
GENE        = "HGSNAT (Heparan Alpha-Glucosaminide N-Acetyltransferase)"
LOCUS       = "8p11.21"
OMIM        = "610453 (HGSNAT gene); 252930 (MPS IIIC / Sanfilippo Syndrome C)"
INHERITANCE = "Autosomal Recessive (AR) — biallelic LOF; null/null → faster course; missense/missense → attenuated"
COHORT_SIZE = 40

# ─── ETIOLOGIES ──────────────────────────────────────────────────────────────
ETIOLOGIES = [
    {
        "name": "Compound Heterozygous — Null + Missense",
        "pct": 35,
        "onset": "Childhood (3–7 years behavioral onset); moderately rapid course",
        "notes": (
            "Most common genotype in non-consanguineous European populations; one truncating "
            "allele (frameshift/nonsense) + one missense allele; HGSNAT acetyltransferase "
            "activity <2%; severe behavioral regression (hyperactivity, aggression, sleep "
            "disorder) by age 3–5 years; language loss by 7–9 years; seizure onset 8–13 years "
            "in 75%; drug resistance 45%; typical MPS IIIC course (slower than IIIA/IIIB null/null); "
            "death 2nd–3rd decade."
        ),
        "key_finding": "Null + missense compound het; HGSNAT <2%; moderate-rapid neurodegeneration; seizures 75%",
    },
    {
        "name": "French-Canadian Founder — p.Arg344Cys (Homozygous / Compound Het)",
        "pct": 25,
        "onset": "Childhood (4–8 years behavioral onset); moderately severe",
        "notes": (
            "p.Arg344Cys (c.1030C>T) — most prevalent HGSNAT variant in Quebec/French-Canadian "
            "population (~20% alleles in Quebec Sanfilippo C patients); homozygous or compound "
            "het with null allele; HGSNAT activity 1–5% residual; slightly attenuated vs null/null; "
            "seizures develop in 72% of patients; cognitive plateau extends slightly vs biallelic null; "
            "NBS feasible for this founder population; enriched in isolated Quebec communities."
        ),
        "key_finding": "p.Arg344Cys French-Canadian founder; HGSNAT 1–5%; moderately severe; seizures 72%",
    },
    {
        "name": "Dutch Founder — p.Trp649stop (Homozygous / Compound Het)",
        "pct": 18,
        "onset": "Early childhood (2–5 years behavioral onset); severe-rapid course",
        "notes": (
            "p.Trp649stop (c.1947G>A) — nonsense variant enriched in Dutch/NW European population "
            "(~15% Dutch HGSNAT alleles); null allele → truncated protein → HGSNAT undetectable; "
            "phenotype similar to biallelic null (severe); behavioral regression by age 2–4 years; "
            "seizures in 80%; drug resistance 50%; death 2nd decade; NBS important for affected families."
        ),
        "key_finding": "p.Trp649stop Dutch founder; null phenotype; severe course; seizures 80%; DRE 50%",
    },
    {
        "name": "Italian Founder — p.Pro524Leu (Compound Het or Hom)",
        "pct": 12,
        "onset": "Childhood (4–8 years behavioral onset); moderately severe",
        "notes": (
            "p.Pro524Leu (c.1571C>T) — missense variant enriched in Italian population; "
            "homozygous or compound het; HGSNAT activity 2–8% residual; moderately severe course; "
            "behavioral regression 4–7 years; seizures 68%; drug resistance 40%; may survive "
            "into 3rd decade; geographic clustering in Italian families (rare in NW European). "
            "Italian/Mediterranean origin: check founder variant panel before WES in cost-constrained settings."
        ),
        "key_finding": "p.Pro524Leu Italian founder; HGSNAT 2–8% residual; moderately severe; seizures 68%",
    },
    {
        "name": "Attenuated — Biallelic Missense (Residual Activity)",
        "pct": 10,
        "onset": "Late childhood / adolescence (6–14 years behavioral onset); extended course",
        "notes": (
            "Biallelic missense variants preserving 5–20% residual HGSNAT acetyltransferase activity; "
            "attenuated phenotype (rare — <10% of MPS IIIC); extended plateau phase; seizures in 55%; "
            "intellectual disability mild-moderate initially; may survive into 4th decade; HIGHEST "
            "misdiagnosis risk — initial labels of ADHD, behavioural disorder, or adult-onset dementia; "
            "potentially highest benefit from gene therapy given preserved CNS substrate at diagnosis."
        ),
        "key_finding": "Residual HGSNAT 5–20%; attenuated extended course; misdiagnosis risk highest; seizures 55%",
    },
]

# ─── SEIZURE TYPES ───────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Generalised Tonic-Clonic (GTCS)",
        "pct": 65,
        "eeg": "Generalised polyspike-wave ≥3 Hz; ictal onset anterior or diffuse",
        "notes": "Most common seizure type in MPS IIIC; GTCS emerge in cognitive decline phase (8–14 yrs); VPA/LEV first-line; CBZ worsens frequency in some patients.",
    },
    {
        "type": "Myoclonic Seizures",
        "pct": 45,
        "eeg": "High-amplitude generalised spike/polyspike; action myoclonus pattern common",
        "notes": "Action myoclonus + PME-like pattern in MPS IIIC; piracetam Level C adjunct; VPA/LEV combination; CBZ CONTRAINDICATED for myoclonus (worsens).",
    },
    {
        "type": "Tonic Seizures",
        "pct": 35,
        "eeg": "Diffuse fast activity (>10 Hz) or electrodecrement; nocturnal predominance",
        "notes": "Tonic seizures emerge in motor deterioration phase; rufinamide adjunct; drop attack risk (protective helmet); nocturnal clustering common.",
    },
    {
        "type": "Absence-like / Atypical Absence",
        "pct": 30,
        "eeg": "Generalised slow spike-wave (1.5–2.5 Hz); often irregular; atypical morphology",
        "notes": "Atypical absence (not typical childhood absence epilepsy); VPA/LEV; ethosuximide NOT indicated (wrong mechanism for atypical absence in LSD context).",
    },
    {
        "type": "Epileptic Spasms (Late-onset)",
        "pct": 15,
        "eeg": "Electrodecrement + high-amplitude slow wave; hypsarrhythmia NOT typical at late-onset",
        "notes": "Late epileptic spasms emerging in motor deterioration phase; ACTH/vigabatrin (VGB) — VGB RELATIVE CI in HGSNAT (visual monitoring); LEV + VPA preferred.",
    },
]

# ─── TRIGGERS ────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Sleep deprivation / disrupted sleep (circadian dysfunction)",    "pct": 85},
    {"trigger": "Febrile illness / intercurrent infection",                        "pct": 70},
    {"trigger": "Behavioral agitation / emotional stress",                         "pct": 62},
    {"trigger": "Missed AED dose",                                                 "pct": 58},
    {"trigger": "Physical exertion / voluntary movement (action myoclonus)",       "pct": 45},
    {"trigger": "Photosensitive stimuli (screen flicker, stroboscopic light)",     "pct": 22},
    {"trigger": "Atypical antipsychotic initiation / dose change",                 "pct": 30},
]

# ─── TREATMENTS ──────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "treatment": "Levetiracetam (LEV)",
        "evidence": "Level A",
        "notes": "First-line for GTCS + myoclonic + focal seizures in MPS IIIC; IV form used in SE (preferred over IV PHT); minimal drug interactions; renal dosing required in late-stage renal involvement.",
        "contraindications": None,
    },
    {
        "treatment": "Valproic Acid (VPA)",
        "evidence": "Level A",
        "notes": "First-line; broad-spectrum; GTCS + myoclonic + absence-like; POLG1 exclusion MANDATORY before initiation (CPIC Level A); hepatic monitoring every 3 months; VPA-protein binding displacement in severe malnutrition; VPPP risk in females ≥12 years.",
        "contraindications": "POLG1/MERRF (exclude first); pregnancy (teratogenic); hepatic failure",
    },
    {
        "treatment": "Clobazam (CLB — adjunct)",
        "evidence": "Level B",
        "notes": "Adjunct to LEV/VPA; GTCS + myoclonic; PARADOXICAL AGITATION documented in MPS IIIC — monitor behavioral response and reduce dose if agitation worsens on CLB initiation; sedation risk.",
        "contraindications": "Paradoxical agitation (documented MPS IIIC risk); severe respiratory compromise",
    },
    {
        "treatment": "Zonisamide (ZNS)",
        "evidence": "Level B",
        "notes": "Adjunct; GTCS + myoclonic; once-daily dosing supports adherence in severe cognitive impairment; sulphonamide allergy contraindication; mild carbonic anhydrase inhibitor (monitor for metabolic acidosis); renal stones rare.",
        "contraindications": "Sulphonamide hypersensitivity; severe renal impairment",
    },
    {
        "treatment": "Piracetam",
        "evidence": "Level C",
        "notes": "Adjunct for action myoclonus in MPS IIIC PME-like phase; high doses (up to 24 g/day) required; well-tolerated; no significant drug interactions; limited RCT evidence in MPS IIIC specifically (extrapolated from PME data).",
        "contraindications": None,
    },
    {
        "treatment": "Rufinamide (adjunct, tonic)",
        "evidence": "Level C",
        "notes": "Adjunct for tonic seizures + drop attacks in motor deterioration phase; Lennox–Gastaut data extrapolated; QT-shortening risk — ECG before initiation.",
        "contraindications": "Short QT syndrome; familial Short QT",
    },
    {
        "treatment": "Melatonin",
        "evidence": "Level B",
        "notes": "Sleep disorder 85–90% in MPS IIIC (cardinal early feature); circadian disruption from hypothalamic HS accumulation (SCN); melatonin 5–10 mg nocte initiates earlier in behavioral phase to reduce sleep-triggered seizure clusters; does not replace AEDs.",
        "contraindications": None,
    },
    {
        "treatment": "AAV9-HGSNAT gene therapy (Phase I — investigational)",
        "evidence": "Investigational",
        "notes": "Intrathecal AAV9-HGSNAT delivery bypasses BBB limitation; Phase I trials ongoing (NCATS-funded); MPS IIIC mouse models show HS reduction + behavioral improvement; enrol pre-symptomatic or early-symptomatic patients; best outcome window narrowing after behavioral phase onset.",
        "contraindications": None,
    },
]

# ─── CONTRAINDICATIONS ───────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Typical Antipsychotics (haloperidol, fluphenazine, chlorpromazine)",
        "risk": "HIGH RISK",
        "mechanism": "HS-laden basal ganglia → D2 blockade → EPS + dystonia + seizure threshold lowering; behavioral phase generates HIGHEST clinical pressure to prescribe typical antipsychotics — resist; atypical antipsychotics (aripiprazole, risperidone low-dose, olanzapine) are the alternative.",
        "alternative": "Aripiprazole or risperidone (low-dose atypical) for behavioral symptoms; consult neuropsychiatry",
    },
    {
        "drug": "PHT / Fosphenytoin",
        "risk": "AVOID in MPS IIIC",
        "mechanism": "Avoid in status epilepticus — use IV LEV; PHT enzyme induction (CYP3A4/2C9) disrupts antipsychotic and melatonin levels; cardiac risk in progressive degenerating myocardium; fosphenytoin IV — same CYP concerns, safer cardiac profile but still inferior to LEV.",
        "alternative": "IV Levetiracetam for SE; oral LEV/VPA for maintenance",
    },
    {
        "drug": "Carbamazepine / Oxcarbazepine (CBZ/OXC)",
        "risk": "CAUTION (NOT absolute CI — unlike MLD/GALC/ARSA)",
        "mechanism": "Subclinical peripheral neuropathy described in late-stage MPS IIIC (axonal, not demyelinating); SIADH hyponatraemia risk; MPS IIIC does NOT have prominent sulfatide/galactocerebroside demyelination — CBZ/OXC are CAUTION, not ABSOLUTE CI; NCS monitoring if used >6 months; HLA-B*15:02 mandatory before CBZ/OXC in SE Asian ancestry (SJS/TEN).",
        "alternative": "LEV or VPA preferred first-line to avoid monitoring burden",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "risk": "RELATIVE CI (visual monitoring impossible)",
        "mechanism": "No retinal NCL ganglion cell toxicity mechanism in HGSNAT (contrast with NCL/CLN where absolute CI); however visual field perimetry impossible in severe ID — use VGB only if critical benefit clearly outweighs monitoring limitation; functional vision assessment (optokinetic nystagmus, ERG) quarterly.",
        "alternative": "LEV or VPA; reserve VGB for refractory infantile spasms if benefit outweighs monitoring limitation",
    },
    {
        "drug": "Ethosuximide (ESX)",
        "risk": "NOT INDICATED",
        "mechanism": "Not indicated for atypical absence seizures in MPS IIIC — mechanism distinct from typical childhood absence (T-type Ca²⁺ channel; works for 3 Hz SW, NOT for slow 1.5–2.5 Hz atypical absence in LSD context); VPA preferred.",
        "alternative": "VPA ± LEV for absence-like seizures in MPS IIIC",
    },
    {
        "drug": "Phenobarbital (PB)",
        "risk": "CAUTION",
        "mechanism": "Significant sedation compounds cognitive decline; enzyme induction disrupts concomitant AED/antipsychotic levels; GABA-A mechanism may not be optimal for myoclonic component; reserve for refractory situation when other AEDs exhausted.",
        "alternative": "LEV/VPA/CLB preferred; PB as last resort",
    },
]

# ─── MONITORING THRESHOLDS ────────────────────────────────────────────────────
THRESHOLDS = {
    "VPA_LFT_frequency_months":       3,
    "CBZ_OXC_NCS_months":            6,
    "melatonin_dose_mg_nocte":        10,
    "seizure_cluster_rescue_trigger": "≥3 seizures in 24 hours",
    "SUDEP_risk_category":            "High (drug-resistant GTCS + nocturnal seizures)",
    "HLA_B1502_mandatory_ancestry":   "SE Asian / Han Chinese before CBZ/OXC",
    "POLG1_exclusion_before_VPA":     "Mandatory (CPIC Level A)",
    "gene_therapy_enrolment_window":  "Pre-symptomatic or early behavioral phase (best outcome)",
}

MONITORING = [
    "POLG1/MERRF sequencing before VPA — mandatory (CPIC Level A)",
    "HLA-B*15:02 before CBZ/OXC in SE Asian ancestry — SJS/TEN risk",
    "LFT + VPA level every 3 months (hepatic storage in severe disease)",
    "NCS every 6 months if CBZ/OXC used (subclinical neuropathy)",
    "Functional vision (optokinetic, ERG) quarterly if VGB used",
    "ECG before rufinamide (QT-shortening risk)",
    "Urine HS quantification 6-monthly for disease progression monitoring",
    "Video-EEG + polysomnography for sleep-seizure overlap characterisation",
    "Behavioral score (ABC-C) monthly during antipsychotic initiation",
    "Growth + nutrition monitoring (dysphagia → PEG tube consideration from motor phase)",
]

# ─── LIFECYCLE ────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Pre-symptomatic (0–3 years)",
        "description": (
            "Newborn screening (NBS) feasible via multiplex HGSNAT enzyme assay; confirmatory "
            "leukocyte HGSNAT + urine HS + HGSNAT gene panel; no clinical symptoms at this stage; "
            "earliest gene therapy window (highest benefit); family cascade genetic counselling; "
            "sleep monitoring started (circadian disruption often earliest objective marker before "
            "behavioral symptoms). Incidence 1:1,400,000 — NBS may not be justified universally "
            "unless combined MPS III multi-enzyme panel used."
        ),
    },
    {
        "stage": "Behavioral Phase (3–7 years)",
        "description": (
            "First presenting symptoms: hyperactivity, aggression, impulsivity, sleep disorder "
            "(circadian disruption from hypothalamic SCN HS storage), language regression; "
            "MPS IIIC behavioral phase slightly LESS severe than MPS IIIA (most severe) but "
            "equally disabling; commonly misdiagnosed as ADHD or autism spectrum disorder; "
            "seizures absent in 90% during early behavioral phase; melatonin initiated for sleep; "
            "behavioral medications: atypical antipsychotics (aripiprazole preferred — NOT typical); "
            "gene therapy referral at this stage for best outcome."
        ),
    },
    {
        "stage": "Cognitive Decline Phase (7–13 years)",
        "description": (
            "Progressive intellectual deterioration; language lost; comprehension declining; "
            "seizures emerge in majority (65–78%) during this phase — GTCS + myoclonic; "
            "EEG: generalised slow spike-wave; AED initiation (LEV/VPA first-line); "
            "PME-like pattern: action myoclonus + dementia + EEG SW; piracetam adjunct; "
            "gene therapy window narrowing — refer to trials urgently if not done."
        ),
    },
    {
        "stage": "Motor Deterioration Phase (10–18 years)",
        "description": (
            "Loss of ambulation (mean age 12–16 years — LATER than MPS IIIA due to slower progression); "
            "dysphagia develops; epilepsy peaks in frequency and drug resistance (DRE 40–50%); "
            "tonic seizures + drop attacks emerge; rufinamide adjunct; SALT referral for dysphagia "
            "assessment (NG/PEG tube consideration); seizure rescue plan essential; "
            "non-verbal communication strategies; SUDEP risk counselling."
        ),
    },
    {
        "stage": "Palliative Phase (15+ years — survival to 4th decade possible)",
        "description": (
            "Bed-bound; fully dependent care; dysphagia + aspiration pneumonia risk (primary "
            "mortality cause); MPS IIIC may survive to 3rd–4th decade (LATER than MPS IIIA/IIIB — "
            "a key distinguishing feature); epilepsy frequency may stabilise as cortical atrophy "
            "reduces excitable tissue; palliative AED approach (comfort, not seizure freedom); "
            "hospice planning; advance care directive; SUDEP counselling to caregivers."
        ),
    },
]

# ─── DIFFERENTIAL DIAGNOSIS ───────────────────────────────────────────────────
DIFFERENTIAL_DIAGNOSIS = [
    {
        "condition": "MPS IIIA (SGSH deficiency)",
        "distinguishing": "Urine HS identical; enzyme panel: SGSH deficient (not HGSNAT); fastest MPS III progression; OAV-101 gene therapy trial",
        "key_test": "Leukocyte SGSH vs HGSNAT enzyme activity",
    },
    {
        "condition": "MPS IIIB (NAGLU deficiency)",
        "distinguishing": "Urine HS identical; enzyme panel: NAGLU deficient; 2nd most common MPS III; AAV9-NAGLU IT trial",
        "key_test": "Leukocyte NAGLU vs HGSNAT enzyme activity",
    },
    {
        "condition": "MPS IIID (GNS deficiency)",
        "distinguishing": "Urine HS identical; rarest Sanfilippo subtype; enzyme panel: GNS deficient; same clinical phenotype as IIIC but slower progression reported",
        "key_test": "Leukocyte GNS vs HGSNAT enzyme activity",
    },
    {
        "condition": "ADHD / Autism Spectrum Disorder",
        "distinguishing": "Misdiagnosis in behavioral phase; urine HS screening resolves; progressive cognitive regression + sleep disorder + seizures emerge; MPS III specific behavioral cluster",
        "key_test": "Urine GAG quantification (DMMB + HS-HPLC-MS) — rules out MPS III",
    },
    {
        "condition": "Neuronal Ceroid Lipofuscinosis (CLN2/CLN3)",
        "distinguishing": "CLN2: tripeptidyl peptidase deficiency; cherry-red spot absent in HGSNAT; CLN2 has retinal dystrophy; urine HS normal in NCL; enzyme panels different",
        "key_test": "Urine HS normal in NCL; leukocyte CLN2/PPT1 enzyme vs HGSNAT",
    },
    {
        "condition": "Progressive Myoclonic Epilepsy (PME — unspecified)",
        "distinguishing": "MPS IIIC PME-like phase mimics Unverricht-Lundborg; POLG/MERRF; urine HS + leukocyte HGSNAT discriminate; PME lacks behavioral regression pattern",
        "key_test": "Urine HS + HGSNAT enzyme activity; POLG1 sequencing",
    },
]

# ─── DEFINITIONS / GLOSSARY ───────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "HGSNAT (Heparan Alpha-Glucosaminide N-Acetyltransferase)",
        "definition": (
            "666 amino acid (73 kDa) multi-pass transmembrane lysosomal enzyme; catalyses "
            "acetyl-CoA-dependent N-acetylation of the terminal α-glucosamine residue in "
            "heparan sulfate degradation; unique among lysosomal enzymes in requiring cytosolic "
            "acetyl-CoA transport across the lysosomal membrane; 8p11.21 locus; AR biallelic LOF."
        ),
    },
    {
        "term": "MPS IIIC (Mucopolysaccharidosis Type IIIC / Sanfilippo Syndrome C)",
        "definition": (
            "3rd most common MPS III subtype (~15–20% of Sanfilippo cases); HGSNAT deficiency → "
            "lysosomal HS accumulation → CNS-predominant progressive neurodegeneration; slightly "
            "slower progression than MPS IIIA/IIIB; survival into 3rd–4th decade possible; "
            "incidence ~1:1,400,000 live births; behavioral → cognitive → motor → palliative."
        ),
    },
    {
        "term": "Heparan Sulfate (HS)",
        "definition": (
            "Linear glycosaminoglycan composed of alternating uronic acid (GlcA/IdoA) and "
            "hexosamine (GlcN/GlcNAc) residues; attached to core proteins (heparan sulfate "
            "proteoglycans — HSPGs) on cell surfaces and in extracellular matrix; critical for "
            "FGF/Wnt/hedgehog signalling; lysosomal HS accumulation in MPS III disrupts synaptic "
            "vesicle recycling, autophagy, mTOR signalling → neurodegeneration."
        ),
    },
    {
        "term": "p.Arg344Cys French-Canadian Founder Variant",
        "definition": (
            "c.1030C>T missense variant in HGSNAT; most prevalent in Quebec/French-Canadian "
            "population (~20% alleles); misfolded HGSNAT protein fails acetyl-CoA import "
            "into lysosome; HGSNAT activity 1–5% residual; homozygous → moderately severe MPS IIIC; "
            "compound het with null → similar severity; genetic drift in founder population."
        ),
    },
    {
        "term": "p.Trp649stop Dutch Founder Variant",
        "definition": (
            "c.1947G>A nonsense variant; ~15% Dutch HGSNAT alleles; premature truncation at "
            "codon 649 → non-functional HGSNAT protein; null phenotype; severe MPS IIIC with "
            "earlier seizure onset and higher drug resistance; NMD (nonsense-mediated decay) "
            "likely — protein absent on Western blot."
        ),
    },
    {
        "term": "POLG1 Exclusion (Mandatory Before VPA)",
        "definition": (
            "POLG1 (polymerase gamma 1) biallelic variants cause Alpers-Huttenlocher syndrome — "
            "VPA triggers acute liver failure in POLG1 disease; MPS IIIC phenotype (cognitive "
            "decline + myoclonus + EEG SW) superficially overlaps POLG/MERRF; POLG1 sequencing "
            "MANDATORY before VPA initiation (CPIC Level A); if POLG1 pathogenic — VPA ABSOLUTELY "
            "CONTRAINDICATED; use LEV/ZNS/CLB instead."
        ),
    },
    {
        "term": "HLA-B*15:02 Screening (Before CBZ/OXC)",
        "definition": (
            "HLA-B*15:02 allele strongly associates with carbamazepine-induced Stevens-Johnson "
            "syndrome (SJS) and toxic epidermal necrolysis (TEN); prevalence: SE Asian/Han Chinese "
            "populations ~5–8%; CPIC Level A recommendation: genotype before CBZ/OXC in SE Asian "
            "ancestry; if HLA-B*15:02 positive — CBZ/OXC ABSOLUTELY CONTRAINDICATED; alternative AED."
        ),
    },
    {
        "term": "Action Myoclonus / PME-like Pattern (MPS IIIC)",
        "definition": (
            "Involuntary jerking of limbs triggered by voluntary movement (action myoclonus); "
            "combined with progressive cognitive decline and EEG generalised spike-wave, this "
            "produces a PME-like phenotype mimicking Unverricht-Lundborg disease or MERRF; "
            "EEG giant SEPs on somatosensory evoked potentials; piracetam Level C adjunct; "
            "distinguished from PME by urine HS elevation and HGSNAT enzyme deficiency."
        ),
    },
    {
        "term": "Clobazam Paradoxical Agitation (MPS IIIC)",
        "definition": (
            "Paradoxical increase in behavioral agitation, hyperactivity, and aggression in "
            "response to clobazam initiation or dose escalation; documented in MPS IIIB and IIIC; "
            "mechanism: disinhibition via GABA-A potentiation in already-disinhibited HS-stored "
            "cortical circuits; clinical action: reduce dose or discontinue CLB; monitor closely "
            "during initiation phase; patient/carer warning mandatory."
        ),
    },
    {
        "term": "Circadian Disruption / Sleep Disorder (MPS IIIC)",
        "definition": (
            "Sleep disorder (reversed sleep-wake cycle, night-time hyperactivity, shortened "
            "sleep duration) in 85–90% of MPS IIIC patients; EARLIEST objective marker often "
            "precedes behavioral regression by months; mechanism: HS accumulation in SCN "
            "(suprachiasmatic nucleus) disrupts circadian melatonin synthesis; melatonin "
            "replacement (5–10 mg nocte) Level B; sleep-triggered seizure clusters documented."
        ),
    },
]

# ─── KEY CONCEPTS ────────────────────────────────────────────────────────────
KEY_CONCEPTS = [
    "MPS IIIC (Sanfilippo C): NO approved disease-modifying therapy (2026) — 3rd unmet need in MPS III after IIIA/IIIB",
    "HGSNAT at 8p11.21: biallelic LOF → acetyl-CoA-dependent acetyltransferase failure → HS accumulation → CNS neurodegeneration",
    "3rd most common MPS III (~15–20%); SLOWER progression than IIIA/IIIB; survival into 3rd–4th decade possible",
    "Epilepsy prevalence 70–80%: lower than MPS IIIA (85–90%) and IIIB (80–90%); onset later (8–14 years)",
    "Drug resistance 40–50%: lower than MPS IIIA/IIIB; more patients achieve partial control on LEV/VPA",
    "Sleep disorder EARLIEST symptom (85–90%): melatonin Level B — initiate in behavioral phase; sleep-seizure nexus documented",
    "Typical antipsychotics: HIGH RISK — behavioral phase generates maximum clinical pressure; atypicals ONLY",
    "PHT/Fosphenytoin: AVOID in SE; use IV LEV; enzyme induction disrupts antipsychotic and melatonin levels",
    "VGB: relative CI (not absolute) — visual field monitoring impossible in severe ID; functional vision assessment only",
    "CBZ/OXC: CAUTION (not absolute CI) — different from MLD/ARSA/GALC absolute CI; neuropathy less prominent in HGSNAT",
    "POLG1 exclusion MANDATORY before VPA — MPS IIIC PME-like phenotype overlaps POLG/MERRF superficially",
    "HLA-B*15:02 MANDATORY before CBZ/OXC in SE Asian ancestry — SJS/TEN fatal risk (CPIC Level A)",
    "Clobazam paradoxical agitation: documented in MPS IIIC — monitor and reduce dose if agitation worsens on CLB",
    "AAV9-HGSNAT gene therapy (intrathecal/ICV): Phase I ongoing; enrol pre-symptomatic or early behavioral phase",
    "HSCT NOT recommended in established MPS IIIC — same exception as MPS IIIA/IIIB (unlike MPS I/II)",
    "French-Canadian founder p.Arg344Cys (20% Quebec alleles); Dutch founder p.Trp649stop (15%); Italian p.Pro524Leu",
    "Key differentiator from MPS IIIA/IIIB: enzyme panel MANDATORY — urine HS identical across all MPS III subtypes",
    "Key differentiator from MPS III vs MPS I/II: NO somatic features (corneal clouding, organomegaly, dysostosis) in IIIC",
]

# ─── STANDARDS ───────────────────────────────────────────────────────────────
STANDARDS = [
    "CPIC Level A: POLG1 genotyping before VPA initiation",
    "CPIC Level A: HLA-B*15:02 before CBZ/OXC in SE Asian ancestry",
    "EUROBIO / Orphanet: MPS IIIC clinical guidelines (Valstar et al. 2010)",
    "ILAE Classification of Epilepsies 2017 / 2022 — structural-metabolic aetiology",
    "ACMG/ACOG: variant classification (ClinVar, LOVD — HGSNAT variants >70 catalogued)",
    "ICH-GCP / CPMP: standards applicable to investigational gene therapy trials",
    "NORD / MPS Society: patient registry enrolment recommended (phenotype-genotype correlation)",
]


# ─── MAIN FUNCTIONS ─────────────────────────────────────────────────────────
def get_overview():
    """Return HGSNAT/MPS IIIC overview data for /api/hgsnat/overview."""
    by_etiology = {e["name"]: e["pct"] for e in ETIOLOGIES}
    by_seizure   = {s["type"]: s["pct"] for s in SEIZURE_TYPES}
    by_trigger   = {t["trigger"]: t["pct"] for t in TRIGGERS}

    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "cohort_size": COHORT_SIZE,
        "disease_name": "MPS IIIC — Sanfilippo Syndrome Type C",
        "disease_mechanism": (
            "HGSNAT biallelic LOF → Heparan Alpha-Glucosaminide N-Acetyltransferase deficiency → "
            "acetyl-CoA-dependent HS acetylation step blocked → lysosomal heparan sulfate (HS) "
            "accumulation → CNS-predominant progressive neurodegeneration; behavioral regression → "
            "cognitive decline → motor failure. 3rd most common MPS III; slower than IIIA/IIIB."
        ),
        "epilepsy_prevalence_pct": 75,
        "epilepsy_onset": "Late childhood — 8–14 years (later onset than MPS IIIA/IIIB due to slower HS accumulation)",
        "drug_resistance_pct": 45,
        "no_approved_disease_therapy": True,
        "gene_therapy_phase": "Phase I (AAV9-HGSNAT intrathecal — NCATS-funded preclinical + Phase I)",
        "pme_risk": "Moderate — action myoclonus + cognitive decline PME-like pattern; HS mechanism (not sphingolipid)",
        "cardinal_seizure_types": ["GTCS", "Myoclonic", "Tonic", "Absence-like", "Late spasms"],
        "cardinal_trigger": "Sleep deprivation (circadian disruption — SCN HS accumulation; sleep disorder 85–90%)",
        "absolute_ci_drugs": [
            "Typical antipsychotics (EPS + seizure threshold — HS-laden basal ganglia)",
            "PHT/Fosphenytoin in SE (use IV LEV instead)",
            "Ethosuximide (not indicated for atypical absence in HGSNAT)",
        ],
        "key_safe_aeds": [
            "Levetiracetam (LEV)",
            "Valproic Acid (VPA — POLG1 excluded)",
            "Clobazam (CLB adjunct — watch paradoxical agitation)",
            "Zonisamide (ZNS — once daily)",
            "Piracetam (Level C — action myoclonus)",
        ],
        "key_warning": (
            "NO approved disease-modifying therapy (2026). Enrol in AAV9-HGSNAT gene therapy trial. "
            "POLG1 exclusion MANDATORY before VPA. HSCT NOT recommended in established disease. "
            "Melatonin strongly indicated — sleep disorder earliest symptom and primary seizure trigger. "
            "MPS IIIC progresses SLOWER than IIIA/IIIB — survival into 3rd–4th decade possible."
        ),
        "distinguishing_from_mps_iiia_iiib": (
            "Urine HS IDENTICAL to MPS IIIA/IIIB — enzyme panel MANDATORY to subtype. "
            "SLOWER progression + LATER seizure onset + LOWER DRE rate than MPS IIIA/IIIB. "
            "French-Canadian (p.Arg344Cys) + Dutch (p.Trp649stop) + Italian (p.Pro524Leu) founders. "
            "HGSNAT is unique among lysosomal enzymes: requires cytosolic acetyl-CoA import."
        ),
        "by_etiology": by_etiology,
        "by_seizure_type": by_seizure,
        "top_triggers": dict(list(by_trigger.items())[:5]),
        "etiologies": ETIOLOGIES,
        "lifecycle": LIFECYCLE,
        "key_concepts": KEY_CONCEPTS,
    }


def get_breakdown():
    """Return HGSNAT/MPS IIIC detailed breakdown for /api/hgsnat/breakdown."""
    import random
    rng = random.Random(42)
    patients = []
    drug_resp   = ["Controlled", "Partially controlled", "Drug-resistant"]
    drug_resp_w = [0.20, 0.35, 0.45]
    stages = [l["stage"] for l in LIFECYCLE]

    for i in range(1, COHORT_SIZE + 1):
        etiology = rng.choices(ETIOLOGIES, weights=[e["pct"] for e in ETIOLOGIES])[0]
        sz_types = [s["type"] for s in rng.choices(SEIZURE_TYPES, k=rng.randint(1, 3), weights=[s["pct"] for s in SEIZURE_TYPES])]
        sz_types = list(dict.fromkeys(sz_types))
        aed = rng.choice(TREATMENTS[:6])
        response = rng.choices(drug_resp, weights=drug_resp_w)[0]
        age_onset = rng.randint(8, 15)
        stage_idx = min(int((i - 1) / 8), len(stages) - 1)
        trigger = rng.choice(TRIGGERS)
        sleep_disorder = rng.random() < 0.88
        on_melatonin   = sleep_disorder and rng.random() < 0.72

        patients.append({
            "patient_id": f"HGSNAT-{i:02d}",
            "etiology": etiology["name"],
            "age_onset_seizures_yrs": age_onset,
            "seizure_types": sz_types,
            "primary_aed": aed["treatment"],
            "drug_response": response,
            "drug_resistant": response == "Drug-resistant",
            "sleep_disorder": sleep_disorder,
            "on_melatonin": on_melatonin,
            "lifecycle_stage": stages[stage_idx],
            "top_trigger": trigger["trigger"],
        })

    drug_resistant_n = sum(1 for p in patients if p["drug_resistant"])
    sleep_disorder_n = sum(1 for p in patients if p["sleep_disorder"])
    on_melatonin_n   = sum(1 for p in patients if p["on_melatonin"])

    return {
        "gene": GENE,
        "cohort_size": COHORT_SIZE,
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "thresholds": THRESHOLDS,
        "lifecycle": LIFECYCLE,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "cohort_summary": {
            "total_patients": COHORT_SIZE,
            "drug_resistant_n": drug_resistant_n,
            "drug_resistant_pct": round(drug_resistant_n / COHORT_SIZE * 100, 1),
            "sleep_disorder_n": sleep_disorder_n,
            "sleep_disorder_pct": round(sleep_disorder_n / COHORT_SIZE * 100, 1),
            "on_melatonin_n": on_melatonin_n,
            "on_melatonin_pct": round(on_melatonin_n / COHORT_SIZE * 100, 1),
        },
        "patients": patients,
    }


def get_definitions():
    """Return HGSNAT/MPS IIIC definitions for /api/hgsnat/definitions."""
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "definitions": DEFINITIONS,
        "key_concepts": KEY_CONCEPTS,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "standards": STANDARDS,
        "diagnostic_algorithm": [
            "Step 1: Urine GAG screen (DMMB quantification + HS-specific HPLC-MS) — isolated HS elevation",
            "Step 2: MPS III enzyme panel (HGSNAT + SGSH + NAGLU + GNS leukocyte assays) — identify HGSNAT deficiency",
            "Step 3: Confirmatory leukocyte HGSNAT acetyltransferase activity (<5% of control)",
            "Step 4: HGSNAT gene sequencing (WES/WGS or targeted panel) — biallelic pathogenic variants",
            "Step 5: POLG1 sequencing — mandatory exclusion before VPA initiation (CPIC Level A)",
            "Step 6: HLA-B*15:02 genotyping — mandatory before CBZ/OXC in SE Asian ancestry",
            "Step 7: Brain MRI (periventricular T2 WM hyperintensity, progressive atrophy) — disease staging",
            "Step 8: Video-EEG + polysomnography — seizure type characterisation + sleep-seizure overlap",
        ],
        "pharmacological_distinctions": [
            "POLG1 mandatory exclusion before VPA — MPS IIIC PME-like phenotype phenocopies POLG/MERRF (not mitochondrial)",
            "IV LEV preferred over IV PHT in SE — PHT enzyme induction disrupts antipsychotics/melatonin; cardiac risk",
            "Typical antipsychotics HIGH RISK — HS-laden basal ganglia → extreme EPS + seizure threshold lowering; use atypicals",
            "VGB relative CI (not absolute) — visual monitoring impossible in severe ID; functional vision only if VGB used",
            "CBZ/OXC CAUTION (not absolute CI) — different from MLD/ARSA/GALC absolute CI; late neuropathy less prominent",
            "Melatonin Level B: sleep disorder earliest symptom; sleep-seizure nexus documented; circadian disruption from SCN HS",
            "Ethosuximide NOT indicated for atypical absence in MPS IIIC — use VPA (ESX for 3 Hz SW typical absence only)",
            "HSCT NOT recommended for established MPS IIIC — contrast with MPS I/II where HSCT is standard",
            "Gene therapy (AAV9-HGSNAT IT) — enrol pre-symptomatic or early behavioral phase; ICV bypasses BBB limitation",
            "Clobazam paradoxical agitation documented in MPS IIIC — monitor and reduce dose if behavioral worsening on CLB",
            "MPS IIIC SLOWER than IIIA/IIIB — survival to 3rd–4th decade; AED regimen must be sustainable long-term",
            "Key enzyme distinction: HGSNAT requires cytosolic acetyl-CoA — ERT cannot replace this reaction (no ERT approved)",
        ],
    }
