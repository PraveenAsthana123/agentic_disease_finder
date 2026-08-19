"""
GLA Epilepsy — Fabry Disease / Anderson-Fabry Disease
======================================================
40-patient cohort · GLA (Xq22.1) · X-linked (XL) — hemizygous males classically affected
GLA encodes Alpha-galactosidase A (α-Gal A, 429 aa, ~46 kDa, GH27 family, EC 3.2.1.22):
  α-Gal A cleaves terminal galactose residues from Gb3 (globotriaosylceramide/GL-3)
  and Lyso-Gb3 (globotriaosylsphingosine).
  GLA hemizygous/biallelic LOF → α-Gal A enzyme deficient → Gb3 + Lyso-Gb3 accumulate
  in endothelium, DRG neurons, cardiomyocytes, podocytes, renal tubular cells →
  multi-organ small-vessel vasculopathy.

DISEASE — FABRY DISEASE / ANDERSON-FABRY DISEASE (OMIM *300644 gene / #301500 disease):
  X-linked lysosomal storage disorder. NOT autosomal recessive (unique in lysosomal series).
  Classic males (hemizygous, <1% α-Gal A): onset childhood — acroparesthesias (4–8y),
    angiokeratomas, corneal verticillata, hypohidrosis, HCM, proteinuria, stroke/TIA.
  Late-onset males (hemizygous missense, 1–10% α-Gal A): cardiac/renal dominant; seizures rare.
  Heterozygous females: highly variable phenotype via lyonization (X-inactivation mosaicism);
    corneal verticillata in 70–95%; seizures in 8%; cardiac/renal involvement variable.
  NO CARRIER STATE — all heterozygous females are at risk for disease manifestations.

PATHOGNOMONIC FEATURES:
  (1) CORNEAL VERTICILLATA (whorl-like corneal opacity): 95% classic males, 70% females —
      visible on slit-lamp examination; pathognomonic for Fabry disease even in heterozygous females.
  (2) ANGIOKERATOMAS (skin): 66% classic males — buttocks/genitalia/umbilicus distribution;
      dark red-purple punctate lesions; PATHOGNOMONIC distribution pattern.
  (3) POSTERIOR CIRCULATION STROKES (vertebrobasilar territory): 88% of Fabry strokes —
      PATHOGNOMONIC pattern; young stroke + posterior circulation → always test Lyso-Gb3.
  (4) LYSO-GB3 PLASMA BIOMARKER: primary biomarker; elevated males (>2 nM) and carrier
      females (>0.8 nM); sensitivity 98%/97% (male); correlates with disease burden.

EPILEPSY — GLA/FABRY-SPECIFIC:
  All seizures are SECONDARY to cerebrovascular disease — no primary Fabry epilepsy.
  Stroke-related focal seizures: 55% (posterior circulation — vertebrobasilar territory).
  Thalamic/basal ganglia focal seizures: 25%.
  Convulsive SE at stroke onset: 8%.
  GTCS post-stroke: 35%.
  Drug resistance: 40% (all secondary to cerebrovascular burden).

SUBTYPES:
  Classic Male (hemizygous null/missense, <1% α-Gal A): 55% of cohort — full phenotype.
  Late-Onset Male (missense, 1–10% α-Gal A): 15% — cardiac/renal dominant; seizures rare.
  Classic Female (heterozygous, variable 10–80% α-Gal A via lyonization): 30% — variable.

DRUG SAFETY:
  CBZ/OXC: CAUTION — hyponatremia risk (Fabry has renal impairment/SIADH); sodium monitoring
    mandatory; worsens neuropathy via Na-channel blockade; not absolute CI (no myoclonus).
  PHT (Phenytoin): RELATIVE-CI — peripheral neuropathy additive (DRG ganglionopathy + PHT
    neuropathy); IV LEV replaces fosphenytoin in SE.
  VPA: SAFE — GLA is lysosomal glycoside hydrolase (NOT mitochondrial); POLG1 exclusion
    mandatory as standard of care (CPIC-POLG1-2023).
  GBP/PGB: DUAL USE — UNIQUE IN THIS SERIES: treats BOTH neuropathic pain (acroparesthesias)
    AND acts as antiseizure agent; Level B for both indications; preferred in Fabry.
  Typical Antipsychotics (Haloperidol, Chlorpromazine): HIGH RISK — QTc prolongation
    (Fabry cardiomyopathy + antipsychotic QTc → Torsades risk); ECG mandatory.
  Migalastat: AMENABLE MUTATIONS ONLY — pharmacogenomics mandatory (Fabry Variants Database).
  CBZ drug interaction: CYP3A4 inducer → may reduce agalsidase plasma levels.
  NOAC/Warfarin: AED selection must minimize drug interactions; LEV preferred (minimal).
"""

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

GENE        = "GLA (Alpha-Galactosidase A / α-Gal A)"
LOCUS       = "Xq22.1"
OMIM        = "*300644 (GLA gene); #301500 (Fabry Disease / Anderson-Fabry Disease)"
INHERITANCE = (
    "X-linked (XL) — hemizygous males classically affected; heterozygous females variable "
    "(mosaicism via lyonization / X-inactivation); NO CARRIER STATE — all heterozygous females at risk; "
    "UNIQUE IN LYSOSOMAL SERIES — not autosomal recessive"
)
COHORT_SIZE = 40

ETIOLOGIES = [
    {
        "name": "Classic Male (hemizygous null/missense, <1% α-Gal A)",
        "pct": 55,
        "onset": "Childhood (acroparesthesias 4–8 years; stroke/HCM adolescence–adulthood)",
        "notes": (
            "Hemizygous GLA null or severe missense; α-Gal A <1% normal; full classic phenotype: "
            "acroparesthesias (burning pain extremities, 4–8y onset), angiokeratomas (buttocks/genitalia/umbilicus, 66%), "
            "corneal verticillata (95% on slit-lamp), hypohidrosis/anhidrosis, hypertrophic cardiomyopathy (HCM), "
            "proteinuria → progressive renal failure, posterior circulation stroke/TIA (88% of Fabry strokes), "
            "psychiatric features (anxiety/panic attacks — 25% misdiagnosed); "
            "Gb3 accumulation in endothelium, DRG neurons, cardiomyocytes, podocytes"
        ),
        "key_finding": (
            "Corneal verticillata (slit-lamp PATHOGNOMONIC) + angiokeratoma (distribution PATHOGNOMONIC) + "
            "Lyso-Gb3 >2 nM + α-Gal A <1%; ERT initiation before age 18 recommended"
        ),
    },
    {
        "name": "Late-Onset Male (hemizygous missense, 1–10% α-Gal A)",
        "pct": 15,
        "onset": "Adult (30s–50s; cardiac/renal dominant presentation)",
        "notes": (
            "Hemizygous GLA missense with partial residual activity (1–10% normal α-Gal A); "
            "cardiac-dominant or renal-dominant phenotype; HCM ± renal failure; "
            "angiokeratomas absent or minimal; corneal verticillata less prevalent (~50%); "
            "acroparesthesias mild or absent; seizures rare (stroke risk lower but present); "
            "often diagnosed incidentally during family screening or cardiac workup; "
            "Lyso-Gb3 moderately elevated (1–5 nM); ERT/migalastat per amenability"
        ),
        "key_finding": (
            "Adult HCM + Lyso-Gb3 1–5 nM + partial α-Gal A (1–10%); "
            "check Fabry Variants Database for migalastat amenability; GFR monitoring essential"
        ),
    },
    {
        "name": "Classic Female (heterozygous, variable 10–80% α-Gal A via lyonization)",
        "pct": 30,
        "onset": "Variable (childhood to adult; highly dependent on X-inactivation pattern)",
        "notes": (
            "Heterozygous GLA mutation; X-inactivation (lyonization) creates tissue-level mosaicism; "
            "phenotype ranges from asymptomatic to as severe as classic male; "
            "corneal verticillata PATHOGNOMONIC in females (70% of heterozygotes by slit-lamp); "
            "cardiac involvement (HCM, arrhythmia) 60%; renal involvement 40%; "
            "acroparesthesias 60%; hypohidrosis 40%; stroke/TIA 25%; seizures 8%; "
            "Lyso-Gb3 >0.8 nM diagnostic in females (sensitivity 82%, specificity 89%); "
            "NO CARRIER STATE — all heterozygous females are at risk"
        ),
        "key_finding": (
            "Corneal verticillata (slit-lamp) PATHOGNOMONIC in females; Lyso-Gb3 >0.8 nM; "
            "all daughters of affected males are heterozygous (obligate); ERT when symptomatic"
        ),
    },
]

SEIZURE_TYPES = [
    {
        "type": "Stroke-related focal seizures (posterior circulation)",
        "pct": 55,
        "subtype": (
            "Vertebrobasilar territory strokes PATHOGNOMONIC in Fabry; "
            "posterior circulation — occipital, cerebellar, brainstem, thalamic involvement; "
            "focal onset seizures from ischemic cortex; LEV Level B post-stroke AED of choice; "
            "anticoagulation/antiplatelet for secondary stroke prevention (Level A)"
        ),
    },
    {
        "type": "Thalamic/basal ganglia focal seizures",
        "pct": 25,
        "subtype": (
            "Thalamic and basal ganglia Gb3 deposition → focal seizures; "
            "deep grey matter involvement; LEV or LTG preferred; "
            "VPA safe but monitor for drug interactions with anticoagulants (enhanced INR monitoring)"
        ),
    },
    {
        "type": "Convulsive Status Epilepticus (SE) at stroke onset",
        "pct": 8,
        "subtype": (
            "Acute symptomatic SE at time of posterior circulation stroke; "
            "IV LEV first-line (preferred over fosphenytoin — DRG neuropathy risk); "
            "IV lorazepam + IV LEV per SE protocol; "
            "avoid fosphenytoin (PHT RELATIVE-CI — additive DRG neuropathy)"
        ),
    },
    {
        "type": "GTCS post-stroke (secondary generalization)",
        "pct": 35,
        "subtype": (
            "Generalised tonic-clonic seizures following stroke; "
            "VPA safe (lysosomal, NOT mitochondrial; POLG1 exclusion mandatory); "
            "LEV Level B; GBP/PGB dual role — neuropathic pain + seizure prevention; "
            "LTG also safe in Fabry (minimal drug interactions with anticoagulants)"
        ),
    },
    {
        "type": "TIA-related transient neurological attacks",
        "pct": 62,
        "subtype": (
            "TIA/lacunar infarct events may include transient focal symptoms mimicking seizures; "
            "DWI MRI mandatory to distinguish TIA from seizure; "
            "antiplatelet (aspirin) + anticoagulation per cardiology; "
            "EEG to confirm seizure activity; posterior circulation TIA highly characteristic"
        ),
    },
]

TRIGGERS = [
    {
        "trigger": "Posterior circulation stroke",
        "pct": 88,
        "notes": (
            "Most potent seizure trigger in Fabry disease — Gb3 endothelial accumulation → "
            "vertebrobasilar territory vasculopathy → stroke → acute symptomatic seizures; "
            "PATHOGNOMONIC stroke territory for Fabry; young stroke + posterior circulation → test Lyso-Gb3"
        ),
    },
    {
        "trigger": "TIA / lacunar infarct",
        "pct": 62,
        "notes": (
            "Transient ischemic attacks from Fabry vasculopathy; multiple TIAs → cumulative seizure risk; "
            "posterior circulation TIAs characteristic; dual antiplatelet or anticoagulation per cardiology "
            "reduces TIA burden and secondary seizure risk"
        ),
    },
    {
        "trigger": "Dehydration / fever (hyperthermia triggers stroke in Fabry)",
        "pct": 45,
        "notes": (
            "Dehydration + hyperthermia → vasomotor instability → stroke trigger in Fabry; "
            "hypohidrosis/anhidrosis (sweating impairment) → heat retention → hyperthermia crisis; "
            "dehydration exacerbates small-vessel vasculopathy; emergency hydration protocol essential"
        ),
    },
    {
        "trigger": "Hyperthermia crisis (heat intolerance due to hypohidrosis)",
        "pct": 35,
        "notes": (
            "Fabry hypohidrosis/anhidrosis → inability to thermoregulate → hyperthermia crisis → "
            "vasoconstrictive stroke trigger → seizures; management: cooling, hydration, avoid heat; "
            "GBP/PGB reduces acroparesthesias + helps prevent hyperthermia-triggered pain crises"
        ),
    },
    {
        "trigger": "Exercise (heat intolerance → crisis)",
        "pct": 38,
        "notes": (
            "Exercise-induced heat accumulation + hypohidrosis → acroparesthesias crisis + vasomotor crisis; "
            "heat-triggered small-vessel events → stroke risk → seizures; "
            "patients advised to avoid strenuous exercise in heat; GBP/PGB for exercise-induced neuropathic pain"
        ),
    },
    {
        "trigger": "Missed ERT dose",
        "pct": 25,
        "notes": (
            "ERT interruption → rapid Gb3/Lyso-Gb3 re-accumulation → increased stroke risk; "
            "missed agalsidase dose correlates with clinical deterioration; "
            "ensure adherence to q2w infusion schedule; CBZ/OXC interaction — CYP3A4 induction "
            "may reduce agalsidase plasma levels"
        ),
    },
    {
        "trigger": "Fever (non-hyperthermia crisis)",
        "pct": 40,
        "notes": (
            "Febrile illness → systemic inflammatory state → vasomotor destabilization → "
            "increased stroke risk in Fabry; fever management paramount; "
            "any febrile episode should trigger reassessment of stroke risk and AED levels"
        ),
    },
    {
        "trigger": "Exertion / physical stress",
        "pct": 38,
        "notes": (
            "Physical stress in the setting of HCM → cardiac arrhythmia → cardioembolism → stroke → seizures; "
            "cardiac monitoring (ECG, Holter) essential; avoid QTc-prolonging drugs (typical antipsychotics); "
            "GBP/PGB dual role: reduces acroparesthesias triggered by exertion AND antiseizure activity"
        ),
    },
]

TREATMENTS = [
    {
        "treatment": "Agalsidase-alfa (Replagal, Shire/Takeda)",
        "level": "A",
        "indication": (
            "Fabry disease — all symptomatic patients; first-line ERT; "
            "EMA approval 2001; 0.2 mg/kg IV every 2 weeks; not FDA-approved (EMA only)"
        ),
        "mechanism": (
            "Recombinant human α-Gal A from human fibroblasts; replaces deficient enzyme; "
            "reduces Gb3 in endothelial cells, plasma, urine; "
            "0.2 mg/kg q2w dose (lower than agalsidase-beta)"
        ),
        "monitoring": (
            "Lyso-Gb3 plasma (q6 months); urine Gb3/creatinine ratio; eGFR; "
            "echocardiogram (HCM monitoring); infusion reactions (premedicate with antihistamines/paracetamol); "
            "anti-drug antibodies (especially in null mutations)"
        ),
        "caution": (
            "CBZ/OXC (CYP3A4 inducers) may theoretically reduce agalsidase plasma levels — monitor ERT response; "
            "infusion reactions in 5–15%; anti-agalsidase IgG antibodies in classic males with null mutations; "
            "EMA-approved only (not FDA)"
        ),
    },
    {
        "treatment": "Agalsidase-beta (Fabrazyme, Sanofi Genzyme)",
        "level": "A",
        "indication": (
            "Fabry disease — all symptomatic patients; first-line ERT; "
            "FDA approval 2003, EMA approval 2001; 1 mg/kg IV every 2 weeks"
        ),
        "mechanism": (
            "Recombinant human α-Gal A from CHO cells; replaces deficient enzyme; "
            "reduces Gb3 in kidney, heart, skin, plasma; "
            "1 mg/kg q2w dose (5× higher than agalsidase-alfa); FDA + EMA approved"
        ),
        "monitoring": (
            "Lyso-Gb3 plasma (q6 months); eGFR annually; echocardiogram; "
            "LV mass index (HCM response); urine protein; "
            "infusion reactions; anti-agalsidase antibodies"
        ),
        "caution": (
            "CBZ/OXC (CYP3A4 inducers) may theoretically reduce agalsidase-beta plasma levels; "
            "infusion reactions 15–20% (mild-moderate); IgE-mediated anaphylaxis rare; "
            "premedication with antihistamines; neutralising antibodies in some null mutations reduce efficacy"
        ),
    },
    {
        "treatment": "Migalastat (Galafold, Amicus Therapeutics)",
        "level": "A",
        "indication": (
            "Fabry disease — AMENABLE MUTATIONS ONLY (must be on Fabry Variants Database amenable list); "
            "FDA approval 2018, EMA approval 2016; 123 mg oral every other day (QOD); "
            "oral chaperone therapy — pharmacogenomics MANDATORY before prescribing"
        ),
        "mechanism": (
            "Pharmacological chaperone — binds and stabilises misfolded α-Gal A in ER; "
            "promotes trafficking to lysosome; increases α-Gal A activity in amenable mutations only; "
            "does NOT work for null/truncating mutations (no protein to stabilise); "
            "oral administration (advantage over IV ERT)"
        ),
        "monitoring": (
            "Confirm mutation in Fabry Variants Database BEFORE prescribing; "
            "α-Gal A enzyme activity (WBC); Lyso-Gb3 plasma; eGFR; echocardiogram; "
            "must NOT test α-Gal A activity within 24h of migalastat dose (chaperone effect gives false-high)"
        ),
        "caution": (
            "AMENABLE MUTATIONS ONLY — prescribing to non-amenable variant → NO BENEFIT; "
            "check fabry-database.org MANDATORY before prescribing; "
            "do not test α-Gal A within 24h of dose; "
            "not suitable for null mutations (no protein to stabilise)"
        ),
    },
    {
        "treatment": "Pegunigalsidase-alfa (Elfabrio, Chiesi)",
        "level": "A",
        "indication": (
            "Fabry disease — adult patients; next-generation pegylated ERT; "
            "FDA approval 2023, EMA approval 2023; 1 mg/kg IV every 2 weeks; "
            "extended plasma half-life vs agalsidase-alfa/beta"
        ),
        "mechanism": (
            "Pegylated recombinant human α-Gal A (PRX-102); PEGylation extends plasma half-life; "
            "reduced immunogenicity vs standard ERT; "
            "replaces deficient enzyme; reduces Gb3 in target organs"
        ),
        "monitoring": (
            "Lyso-Gb3 plasma; eGFR; echocardiogram; infusion reactions; "
            "anti-drug antibodies (lower immunogenicity expected vs standard ERT); "
            "MODIFY trial data (2023) for long-term monitoring guidance"
        ),
        "caution": (
            "Approved 2023 — post-marketing surveillance ongoing; "
            "CBZ/OXC CYP3A4 interaction theoretical; "
            "infusion reactions; hypersensitivity; "
            "limited long-term comparative data vs agalsidase-beta"
        ),
    },
    {
        "treatment": "LEV (Levetiracetam) — Post-Stroke AED",
        "level": "B",
        "indication": (
            "Post-stroke seizures in Fabry disease; SE management (IV formulation); "
            "focal + GTCS; minimal drug interactions with anticoagulants (preferred in Fabry)"
        ),
        "mechanism": (
            "SV2A vesicle protein modulation; broad-spectrum antiseizure; "
            "IV formulation available for SE (replaces fosphenytoin in Fabry); "
            "renal excretion; no hepatic CYP interactions (advantage with NOAC/warfarin)"
        ),
        "monitoring": (
            "Renal function (dose adjust for GFR <50 mL/min — common in Fabry nephropathy); "
            "behavioural side-effects; CBC annually; "
            "critical: GFR monitoring in Fabry renal disease"
        ),
        "caution": (
            "Renal dose adjustment MANDATORY in Fabry (renal impairment common); "
            "behavioural side-effects (irritability); "
            "preferred over fosphenytoin in SE — avoids PHT DRG neuropathy additive risk"
        ),
    },
    {
        "treatment": "GBP/PGB (Gabapentin/Pregabalin) — DUAL USE in Fabry",
        "level": "B",
        "indication": (
            "DUAL USE — UNIQUE IN LYSOSOMAL SERIES: "
            "(1) Neuropathic pain (acroparesthesias) — burning extremity pain from DRG Gb3 accumulation; "
            "(2) Antiseizure for post-stroke focal/GTCS; "
            "Level B for both indications in Fabry disease"
        ),
        "mechanism": (
            "Voltage-gated calcium channel α2δ subunit ligand; "
            "reduces neuropathic pain signal transmission in DRG neurons (DRG Gb3 accumulation in Fabry); "
            "antiseizure effect via same calcium channel mechanism; "
            "dual therapeutic role makes GBP/PGB uniquely valuable in Fabry"
        ),
        "monitoring": (
            "Renal dose adjustment (Fabry nephropathy — GFR monitoring essential); "
            "sedation; ataxia (less concern than in AR lysosomal diseases without ataxia); "
            "efficacy for both pain and seizure control"
        ),
        "caution": (
            "Renal dose adjustment required (Fabry renal impairment); "
            "sedation + ataxia in high doses; "
            "UNIQUE: treats acroparesthesias AND prevents seizures — dual prescribing advantage; "
            "no QTc concern (unlike typical antipsychotics)"
        ),
    },
    {
        "treatment": "VPA (Valproic Acid)",
        "level": "B",
        "indication": (
            "GTCS + focal seizures post-stroke; broad-spectrum; "
            "GLA is lysosomal glycoside hydrolase — NOT mitochondrial; VPA generally safe"
        ),
        "mechanism": (
            "Sodium channel + GABA + T-type calcium; broad antiseizure; "
            "lysosomal pathway — no direct conflict; "
            "POLG1 exclusion mandatory as standard of care before initiation"
        ),
        "monitoring": (
            "POLG1 exclusion MANDATORY (CPIC-POLG1-2023) before VPA; "
            "LFT 3-monthly (standard monitoring); "
            "INR monitoring if co-prescribed with warfarin (VPA + warfarin → enhanced anticoagulation); "
            "ammonia if encephalopathy suspected"
        ),
        "caution": (
            "POLG1 exclusion mandatory before VPA (even though GLA is lysosomal not mitochondrial); "
            "VPA + warfarin → enhanced INR monitoring; "
            "LFT 3-monthly standard; "
            "not first-line if stroke prevention anticoagulation is warfarin (INR interaction)"
        ),
    },
    {
        "treatment": "Anticoagulation/Antiplatelet (Stroke Prevention)",
        "level": "A",
        "indication": (
            "Secondary stroke prevention in Fabry disease — Level A guideline; "
            "aspirin, warfarin, or NOAC per cardiology assessment; "
            "AED selection must minimise drug interactions with anticoagulants"
        ),
        "mechanism": (
            "Antiplatelet (aspirin, clopidogrel) reduces TIA/stroke recurrence; "
            "anticoagulation (warfarin or NOAC) for cardioembolic stroke (HCM-related arrhythmia); "
            "NOAC preferred over warfarin for lower drug-drug interaction profile"
        ),
        "monitoring": (
            "INR (warfarin); renal function (NOAC dose); "
            "bleeding risk; LFT; "
            "AED interactions: LEV/LTG minimal; VPA + warfarin → enhanced INR; "
            "CBZ/OXC (CYP3A4 inducers) may reduce warfarin/NOAC levels"
        ),
        "caution": (
            "CBZ/OXC CYP3A4 induction may reduce warfarin/NOAC plasma levels → "
            "inadequate stroke prevention; "
            "PHT similar CYP induction concern; "
            "LEV and LTG preferred AEDs for minimal anticoagulant interactions"
        ),
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, etc.)",
        "risk": "HIGH RISK",
        "mechanism": (
            "QTc prolongation — Fabry cardiomyopathy (HCM) + antipsychotic QTc-prolonging effect → "
            "Torsades de Pointes risk; D2 blockade → extrapyramidal effects; "
            "25% Fabry patients misdiagnosed with anxiety/panic (acroparesthesias mimicking panic) → "
            "typical antipsychotics incorrectly prescribed; ECG MANDATORY before any antipsychotic use"
        ),
        "alternative": (
            "Atypical antipsychotics (quetiapine — lowest QTc risk) if psychiatric symptoms require treatment; "
            "treat acroparesthesias (GBP/PGB) and anxiety components separately; "
            "ECG before initiation + QTc monitoring"
        ),
        "evidence": (
            "Fabry HCM + QTc-prolonging drugs: Weidemann 2013 (FOS registry); "
            "cardiac arrhythmia risk in HCM: NICE TA; psychiatric misdiagnosis 25%: Schiffmann 2018; "
            "QTc threshold monitoring: >450ms (male) / >470ms (female) → avoid QTc-prolonging AEDs"
        ),
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "risk": "CAUTION (not absolute CI)",
        "mechanism": (
            "Hyponatremia risk — Fabry has renal impairment + SIADH susceptibility; "
            "CBZ/OXC-induced hyponatremia compounded by Fabry renal disease; "
            "sodium channel blockade may worsen DRG ganglionopathy-related neuropathy; "
            "CYP3A4 induction (CBZ) → reduces agalsidase plasma levels (ERT interaction); "
            "NOT absolute CI — no myoclonus component in Fabry"
        ),
        "alternative": (
            "LEV (no hyponatremia, no CYP3A4 induction, no neuropathy worsening); "
            "LTG (minimal drug interactions); "
            "GBP/PGB (dual use: neuropathic pain + antiseizure, no CYP3A4 interaction); "
            "sodium monitoring mandatory if CBZ/OXC used"
        ),
        "evidence": (
            "CBZ/OXC hyponatremia + renal impairment: NICE NG217; "
            "CYP3A4 induction + ERT: pharmacokinetic interaction (theoretical, monitor ERT biomarkers); "
            "DRG neuropathy worsening: Fabry specialist consensus; "
            "sodium monitoring: electrolytes every 3 months if CBZ/OXC continued"
        ),
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin (IV)",
        "risk": "RELATIVE-CI",
        "mechanism": (
            "Peripheral neuropathy additive — Fabry DRG ganglionopathy (Gb3 in dorsal root ganglion neurons) "
            "plus PHT neuropathy (axonal, length-dependent); cumulative neuropathic burden worsened; "
            "IV fosphenytoin risky in acute SE — IV LEV replaces in Fabry SE management"
        ),
        "alternative": (
            "IV LEV for SE (replaces fosphenytoin); "
            "oral LEV or LTG for chronic post-stroke seizure management; "
            "GBP/PGB dual use (neuropathic pain + antiseizure — preferred in DRG neuropathy context)"
        ),
        "evidence": (
            "PHT peripheral neuropathy: Perucca 2006; DRG Gb3 accumulation: Schiffmann 2018; "
            "IV LEV in SE replacing fosphenytoin: Neurocritical Care Society guidelines 2012; "
            "Fabry DRG ganglionopathy: Kaye 2010 — additive neuropathy risk confirmed"
        ),
    },
    {
        "drug": "Migalastat — Non-Amenable Mutations (Prescribing Error)",
        "risk": "NO BENEFIT / PRESCRIBING ERROR",
        "mechanism": (
            "Migalastat is a pharmacological chaperone — only stabilises MISSENSE mutations that produce "
            "misfolded but translatable α-Gal A protein; null/truncating mutations produce no protein to stabilise; "
            "prescribing migalastat to non-amenable variant → no α-Gal A increase → no clinical benefit; "
            "patients receive oral treatment believing it works but disease progresses"
        ),
        "alternative": (
            "MANDATORY: check GLA variant against Fabry Variants Database (fabry-database.org) BEFORE prescribing; "
            "non-amenable mutations → agalsidase-alfa or agalsidase-beta (IV ERT); "
            "pharmacogenomics consultation mandatory"
        ),
        "evidence": (
            "Migalastat Phase III (ATTRACT trial): Schiffmann 2018 — amenable mutations only; "
            "FDA 2018 label: amenable mutations required; "
            "Fabry Variants Database (Amicus/FDA validated): fabry-database.org; "
            "NICE TA694 (2021) — migalastat amenable mutations criteria"
        ),
    },
    {
        "drug": "CBZ (CYP3A4 Induction) — ERT Drug Interaction",
        "risk": "CAUTION — ERT Efficacy Reduction",
        "mechanism": (
            "CBZ is a potent CYP3A4 inducer; agalsidase-alfa and agalsidase-beta are glycoproteins "
            "metabolised in part via hepatic pathways; theoretical reduction of ERT plasma levels; "
            "reduced ERT bioavailability → suboptimal Gb3 clearance → disease progression; "
            "monitor Lyso-Gb3 response if CBZ co-prescribed"
        ),
        "alternative": (
            "LEV (no CYP3A4 interaction); LTG (minimal CYP interaction); "
            "GBP/PGB (no CYP3A4 interaction, dual use benefit); "
            "if CBZ essential: monitor Lyso-Gb3 closely and consider ERT dose adjustment"
        ),
        "evidence": (
            "CYP3A4 induction + ERT: pharmacokinetic interaction (theoretical, no prospective RCT); "
            "agalsidase metabolism: product monographs; "
            "clinical monitoring recommended: Lyso-Gb3 every 3–6 months if CBZ co-prescribed; "
            "Fabry specialist consensus"
        ),
    },
    {
        "drug": "VPA + Warfarin (Anticoagulant Interaction)",
        "risk": "CAUTION — Enhanced Anticoagulation",
        "mechanism": (
            "VPA displaces warfarin from plasma protein binding + inhibits CYP2C9 (warfarin metabolism); "
            "enhanced anticoagulation → bleeding risk; "
            "Fabry patients on warfarin for stroke prevention + VPA for seizures → "
            "mandatory INR enhanced monitoring"
        ),
        "alternative": (
            "LEV or LTG preferred AEDs when patient on warfarin (minimal protein binding, no CYP interaction); "
            "if VPA required: weekly INR monitoring during initiation, monthly when stable; "
            "consider NOAC instead of warfarin (fewer VPA interactions)"
        ),
        "evidence": (
            "VPA + warfarin interaction: Levy 1999; CYP2C9 inhibition: Margolis 1987; "
            "protein displacement: clinical pharmacology textbooks; "
            "Fabry stroke prevention warfarin: European Fabry Working Group guidelines; "
            "INR monitoring: mandatory per anticoagulation clinic"
        ),
    },
    {
        "drug": "VPA (without POLG1 exclusion)",
        "risk": "PROTOCOL VIOLATION",
        "mechanism": (
            "POLG1 mutation → mitochondrial DNA depletion syndrome; VPA inhibits mtDNA polymerase gamma → "
            "POLG1 patients → fatal hepatotoxicity; GLA is lysosomal (NOT mitochondrial) BUT "
            "POLG1 exclusion before VPA is MANDATORY standard of care regardless of primary diagnosis"
        ),
        "alternative": (
            "POLG1 gene sequencing BEFORE VPA initiation (CPIC-POLG1-2023 Level A); "
            "if POLG1 confirmed → VPA absolutely contraindicated; "
            "LEV or LTG as alternatives if POLG1 positive"
        ),
        "evidence": (
            "CPIC-POLG1 guideline 2023 — Level A; VPA hepatotoxicity in POLG1: Naviaux 1999; "
            "mandatory exclusion regardless of primary diagnosis: CPIC implementation guidance; "
            "OMIM POLG: 174763"
        ),
    },
    {
        "drug": "Fosphenytoin IV (Acute SE)",
        "risk": "RELATIVE-CI — Use IV LEV Instead",
        "mechanism": (
            "Fosphenytoin is a prodrug of PHT; peripheral neuropathy risk (DRG additive in Fabry); "
            "IV LEV is the preferred alternative for SE in Fabry — avoids DRG neuropathy worsening; "
            "same efficacy as fosphenytoin in SE with better Fabry-specific safety profile"
        ),
        "alternative": (
            "IV LEV (60 mg/kg over 15 min) replaces fosphenytoin in Fabry SE; "
            "IV lorazepam first-line for SE initiation (standard protocol); "
            "IV valproate as alternative (POLG1 excluded)"
        ),
        "evidence": (
            "IV LEV vs fosphenytoin SE equivalence: NCS 2012, ESETT trial 2019; "
            "DRG neuropathy in Fabry: Kaye 2010; PHT peripheral neuropathy additive: Perucca 2006; "
            "IV LEV preferred in neuropathic conditions: Fabry specialist consensus"
        ),
    },
]

THRESHOLDS = [
    {
        "name": "Lyso-Gb3 — male diagnostic threshold",
        "value": ">2 nM",
        "significance": "Sensitivity 98%, specificity 97% for Fabry disease in males; primary screening biomarker; "
                        "correlates with disease burden and ERT response",
    },
    {
        "name": "Lyso-Gb3 — female (heterozygous) diagnostic threshold",
        "value": ">0.8 nM",
        "significance": "Sensitivity 82%, specificity 89% for Fabry disease in heterozygous females; "
                        "lower threshold due to variable lyonization; best biomarker in females",
    },
    {
        "name": "α-Gal A — Classic male (diagnostic)",
        "value": "<1% normal",
        "significance": "Classic male phenotype; nil/trace α-Gal A activity; Gb3 accumulates in all organs; "
                        "ERT initiation before age 18 recommended in confirmed cases",
    },
    {
        "name": "α-Gal A — Late-onset male",
        "value": "1–10% normal",
        "significance": "Partial residual α-Gal A activity; cardiac/renal dominant phenotype; "
                        "migalastat amenability must be checked regardless of residual activity",
    },
    {
        "name": "Urine Gb3 (GL-3) — elevated",
        "value": ">1.5 μg/mg creatinine",
        "significance": "Secondary biomarker; elevated in classic Fabry males; "
                        "useful for monitoring ERT response (Gb3 normalisation with effective ERT)",
    },
    {
        "name": "ERT initiation age — males",
        "value": "≤18 years (pre-symptomatic after confirmed diagnosis)",
        "significance": "Guideline recommendation: start ERT before organ damage occurs; "
                        "prevents irreversible nephropathy, HCM, and stroke; "
                        "initiation age strongly predictive of renal outcome (Weidemann 2013)",
    },
    {
        "name": "GFR threshold for ERT monitoring",
        "value": "GFR <45 mL/min",
        "significance": "Progressive Fabry nephropathy threshold — ERT slows GFR decline; "
                        "GFR <45 mL/min signals advanced nephropathy; "
                        "LEV dose reduction required at this threshold (renal excretion)",
    },
    {
        "name": "HCM threshold for ERT cardiology",
        "value": "LV wall thickness >12 mm",
        "significance": "HCM threshold requiring combined ERT + cardiology management; "
                        "avoid QTc-prolonging drugs (typical antipsychotics) at this stage; "
                        "ECG + echo monitoring q6 months",
    },
    {
        "name": "Migalastat — amenable mutation requirement",
        "value": "GLA variant on Fabry Variants Database amenable list",
        "significance": "MANDATORY pharmacogenomics check before migalastat; "
                        "non-amenable variants → no benefit; fabry-database.org (FDA validated); "
                        "prescribing error if given to non-amenable mutation",
    },
    {
        "name": "Seizure prophylaxis threshold",
        "value": "≥2 seizures OR 1 SE episode",
        "significance": "Post-stroke seizure prophylaxis indicated; LEV or GBP/PGB preferred; "
                        "single seizure → observation vs prophylaxis based on stroke extent and recurrence risk",
    },
    {
        "name": "QTc monitoring threshold (avoid QTc-prolonging AEDs)",
        "value": "QTc >450ms (male) / >470ms (female)",
        "significance": "Fabry HCM + QTc-prolonging drugs → Torsades de Pointes risk; "
                        "mandatory ECG monitoring; avoid typical antipsychotics above threshold; "
                        "Holter monitoring in HCM patients on any QTc-active medication",
    },
    {
        "name": "POLG1 exclusion before VPA",
        "value": "Mandatory protocol (CPIC-POLG1-2023 Level A)",
        "significance": "POLG1 sequencing required before any VPA initiation in Fabry patients; "
                        "GLA is lysosomal (not mitochondrial) but POLG1 exclusion is standard of care; "
                        "POLG1 positive → VPA absolutely contraindicated",
    },
]

STANDARDS = [
    "Anderson H (1898) — First description of angiokeratoma corporis diffusum (simultaneous with Fabry)",
    "Fabry J (1898) — Simultaneous independent description of angiokeratoma corporis diffusum; eponym disease",
    "Brady RO et al. (1967) — α-Galactosidase A enzyme deficiency identified as cause of Fabry disease",
    "Desnick RJ et al. (2001) — Fabry disease molecular genetics and ERT foundation",
    "Eng CM et al. (2001) — Agalsidase-alfa/beta first clinical trials (Phase III ERT Fabry)",
    "Hughes DA et al. (2008) — Fabry Outcome Survey (FOS) — natural history and ERT outcomes registry",
    "Weidemann F et al. (2013) — Fabry Outcome Survey: cardiac outcomes and ERT timing (ERT age predicts renal outcome)",
    "Schiffmann R et al. (2018) — Migalastat Phase III ATTRACT trial — amenable mutations pharmacogenomics",
    "Lenders M et al. (2023) — Pegunigalsidase-alfa (Elfabrio) Phase III BALANCE trial",
    "ILAE 2022 — Classification of Epilepsies: Structural (post-stroke) epilepsy framework",
    "NICE TA694 (2021) — Migalastat for treating Fabry disease (amenable mutations only)",
    "EMA 2001 (agalsidase-alfa) / FDA 2003 (agalsidase-beta) / FDA 2018 (migalastat) / FDA+EMA 2023 (pegunigalsidase) approvals",
]

KEY_CONCEPTS = [
    {
        "term": "X-linked inheritance — UNIQUE IN LYSOSOMAL EPILEPSY SERIES",
        "definition": (
            "Fabry disease is X-linked — hemizygous males classically affected; heterozygous females "
            "have variable disease via lyonization (X-inactivation mosaicism). Unlike ALL other lysosomal "
            "diseases in this series (MLD/ARSA, MPS/GALNS, Krabbe/GALC, Gaucher/GBA, NPD/SMPD1, "
            "Farber/ASAH1, Prosaposin/PSAP, MSD/SUMF1 — all autosomal recessive), Fabry is X-LINKED. "
            "NO CARRIER STATE: all heterozygous females are at risk for disease manifestations. "
            "All daughters of affected males are obligate heterozygotes. Inheritance pattern counselling differs."
        ),
    },
    {
        "term": "GLA — single enzyme, two substrates (Gb3 + Lyso-Gb3)",
        "definition": (
            "α-Gal A cleaves terminal α-galactosyl residues from Gb3 (globotriaosylceramide/GL-3) and "
            "Lyso-Gb3 (globotriaosylsphingosine). GLA LOF → both substrates accumulate. "
            "Lyso-Gb3 is the neurotoxic biomarker: plasma Lyso-Gb3 >2 nM (males) / >0.8 nM (females) "
            "is the primary diagnostic and monitoring biomarker. Unlike SUMF1 (all 17 sulfatases), "
            "GLA deficiency is single-enzyme — more targetable with enzyme replacement and chaperone therapy."
        ),
    },
    {
        "term": "Corneal verticillata — PATHOGNOMONIC",
        "definition": (
            "Whorl-like corneal opacity (cornea verticillata / vortex keratopathy) visible on slit-lamp "
            "examination: 95% of classic males, 70% of heterozygous females. PATHOGNOMONIC for Fabry disease. "
            "Caused by Gb3 deposition in corneal epithelial cells. Does NOT affect vision. "
            "Slit-lamp examination is mandatory in all suspected Fabry cases. "
            "First observable sign in children as young as 3–4 years — important for early detection."
        ),
    },
    {
        "term": "Posterior circulation strokes — PATHOGNOMONIC seizure trigger",
        "definition": (
            "88% of Fabry strokes occur in the vertebrobasilar (posterior circulation) territory — "
            "occipital, cerebellar, brainstem, thalamic involvement. This distribution is PATHOGNOMONIC "
            "for Fabry disease. Young stroke + posterior circulation → always test Lyso-Gb3 plasma. "
            "All seizures in Fabry are SECONDARY to cerebrovascular disease — no primary Fabry epilepsy. "
            "AED selection must minimise drug interactions with anticoagulants used for stroke prevention."
        ),
    },
    {
        "term": "Agalsidase-alfa vs agalsidase-beta — same mechanism, different dosing",
        "definition": (
            "Both are recombinant human α-Gal A ERT, Level A. Key differences: "
            "Agalsidase-alfa (Replagal): 0.2 mg/kg q2w, human fibroblast-derived, EMA-approved only; "
            "Agalsidase-beta (Fabrazyme): 1 mg/kg q2w, CHO-derived, FDA + EMA approved. "
            "Head-to-head trials show similar efficacy; dose difference is 5-fold. "
            "Both can elicit anti-drug antibodies especially in null mutations — neutralising antibodies "
            "reduce ERT efficacy and require tolerance induction protocols."
        ),
    },
    {
        "term": "Migalastat amenable mutations — pharmacogenomics mandatory",
        "definition": (
            "Migalastat is a pharmacological chaperone that ONLY works for amenable missense mutations "
            "(GLA variant produces misfolded but translatable α-Gal A protein that migalastat can stabilise). "
            "Null/truncating mutations produce no protein → no chaperone benefit. "
            "MANDATORY: check GLA variant against Fabry Variants Database (fabry-database.org) before prescribing. "
            "Prescribing migalastat to non-amenable variant = prescribing error. "
            "Pharmacogenomics consultation mandatory before migalastat initiation."
        ),
    },
    {
        "term": "GBP/PGB dual use — UNIQUE IN LYSOSOMAL SERIES (neuropathic pain + antiseizure)",
        "definition": (
            "Gabapentin/Pregabalin has a UNIQUE DUAL ROLE in Fabry disease: "
            "(1) Neuropathic pain: treats acroparesthesias (burning extremity pain from DRG Gb3 accumulation) "
            "— Level B evidence; "
            "(2) Antiseizure: post-stroke focal/GTCS seizure prevention — Level B; "
            "This dual therapeutic role makes GBP/PGB uniquely valuable in Fabry disease. "
            "No other lysosomal disease in this series benefits from GBP/PGB for both indications. "
            "Renal dose adjustment required (Fabry nephropathy)."
        ),
    },
    {
        "term": "Typical antipsychotics — HIGH RISK (QTc + HCM)",
        "definition": (
            "Fabry cardiomyopathy (HCM) + typical antipsychotic QTc prolongation → Torsades de Pointes risk. "
            "25% of Fabry patients are misdiagnosed with anxiety/panic attacks (acroparesthesias + "
            "hypohidrosis crisis mimicking panic) → typical antipsychotics incorrectly prescribed. "
            "ECG mandatory before any antipsychotic use. QTc threshold: >450ms (male) / >470ms (female). "
            "Preferred: treat acroparesthesias with GBP/PGB, anxiety with CBT; "
            "atypical antipsychotics (quetiapine) if psychiatric treatment essential."
        ),
    },
    {
        "term": "DBS Lyso-Gb3 — neonatal screening",
        "definition": (
            "Dried blood spot (DBS) Lyso-Gb3 assay is the preferred neonatal screening biomarker for Fabry disease. "
            "DBS α-Gal A alone misses heterozygous females (enzyme activity overlaps normal range in females). "
            "DBS Lyso-Gb3 detects affected males reliably and some heterozygous females. "
            "Newborn screening programs for Fabry (Taiwan, Austria, Italy pilot programs) use DBS Lyso-Gb3 + "
            "GLA gene sequencing for confirmation."
        ),
    },
    {
        "term": "Pegunigalsidase-alfa (Elfabrio) — third-generation pegylated ERT (FDA/EMA 2023)",
        "definition": (
            "Pegunigalsidase-alfa (PRX-102, Elfabrio, Chiesi) is the newest ERT for Fabry disease: "
            "pegylated recombinant human α-Gal A; PEGylation extends plasma half-life; "
            "potentially reduced immunogenicity vs standard ERT; 1 mg/kg q2w IV; "
            "FDA approval August 2023, EMA approval 2023. BALANCE trial (Phase III vs agalsidase-beta) "
            "showed non-inferiority. Post-marketing surveillance ongoing."
        ),
    },
    {
        "term": "NOAC/anticoagulation + AED interactions — LEV/LTG preferred",
        "definition": (
            "Fabry disease stroke prevention requires anticoagulation/antiplatelet therapy. "
            "AED selection must minimise drug-drug interactions with anticoagulants: "
            "LEV: no CYP interactions → preferred AED in Fabry (no effect on warfarin/NOAC levels); "
            "LTG: glucuronidation only → safe with anticoagulants; "
            "VPA + warfarin → enhanced INR (protein displacement + CYP2C9 inhibition) → weekly INR; "
            "CBZ/OXC: CYP3A4 induction → reduces warfarin/NOAC levels → stroke risk; "
            "PHT: CYP induction → similar concern; avoid if on anticoagulants."
        ),
    },
    {
        "term": "Hyperthermia crisis — triggers stroke → seizures → emergency management",
        "definition": (
            "Fabry hypohidrosis/anhidrosis (absent sweating due to Gb3 in sweat gland autonomic neurons) → "
            "inability to thermoregulate → hyperthermia crisis → vasomotor instability → "
            "acute posterior circulation stroke → acute symptomatic seizures. "
            "Emergency management: cooling, IV hydration, cardiac monitoring. "
            "Prevention: GBP/PGB for acroparesthesias; avoid heat/exertion; "
            "ERT reduces hypohidrosis burden over time."
        ),
    },
    {
        "term": "Psychiatric misdiagnosis — 25% of Fabry patients",
        "definition": (
            "25% of Fabry patients (especially adult females) are initially misdiagnosed with anxiety, "
            "panic disorder, or depression. Acroparesthesias (burning extremity pain), autonomic symptoms "
            "(hypohidrosis, palpitations), and fatigue mimic panic attacks. "
            "Consequence: typical antipsychotics prescribed → HIGH RISK (QTc + HCM). "
            "Red flags for Fabry in 'psychiatric' patients: young onset anxiety + heat intolerance + "
            "acroparesthesias + family history → test Lyso-Gb3."
        ),
    },
]

DIAGNOSTIC_ALGORITHM = [
    "Step 1 — Suspect Fabry: Young posterior circulation stroke/TIA + acroparesthesias + "
    "corneal verticillata (slit-lamp) + HCM + angiokeratomas → test Lyso-Gb3 plasma immediately; "
    "any young stroke (<50y) + posterior circulation territory → Fabry screening mandatory",
    "Step 2 — DBS α-Gal A enzyme assay (males): DBS α-Gal A <1% normal → Fabry confirmed (males); "
    "caution: DBS α-Gal A UNRELIABLE in females (lyonization overlap with normal) → use Lyso-Gb3 in females",
    "Step 3 — Lyso-Gb3 plasma: >2 nM males (sens 98%, spec 97%) / >0.8 nM females (sens 82%, spec 89%); "
    "Lyso-Gb3 is primary biomarker for all patients; correlates with disease burden; monitor ERT response",
    "Step 4 — GLA gene sequencing (Xq22.1): confirm hemizygous (males) or heterozygous (females) GLA variant; "
    "classify variant as null/missense; check Fabry Variants Database for migalastat amenability",
    "Step 5 — Fabry Variants Database (fabry-database.org) amenability check: MANDATORY before any treatment; "
    "amenable mutation → migalastat eligible; non-amenable → agalsidase-alfa or agalsidase-beta; "
    "pharmacogenomics consultation mandatory",
    "Step 6 — POLG1 exclusion: MANDATORY before VPA initiation (CPIC-POLG1-2023 Level A); "
    "even though GLA is lysosomal (not mitochondrial), POLG1 exclusion is standard of care",
    "Step 7 — Multidisciplinary plan: Neurology (post-stroke seizure management — LEV/GBP preferred; "
    "avoid typical antipsychotics QTc; avoid fosphenytoin DRG neuropathy) + Cardiology (HCM/ECG/QTc monitoring + "
    "stroke prevention anticoagulation — NOAC preferred) + Nephrology (GFR/proteinuria) + Ophthalmology "
    "(corneal verticillata slit-lamp) + Genetics (GLA WES + POLG1) + Metabolic Medicine (ERT or migalastat) "
    "+ Dermatology (angiokeratoma) + Physiotherapy (acroparesthesias management with GBP/PGB)",
]

GLOSSARY = {
    "α-Gal A (Alpha-Galactosidase A)": (
        "429-aa ~46 kDa lysosomal enzyme (GH27 family, EC 3.2.1.22) encoded by GLA (Xq22.1); "
        "cleaves terminal α-galactosyl residues from Gb3 and Lyso-Gb3; deficient in Fabry disease"
    ),
    "Gb3 (Globotriaosylceramide / GL-3)": (
        "Primary substrate of α-Gal A; glycosphingolipid accumulating in endothelium, "
        "cardiomyocytes, podocytes, DRG neurons in Fabry disease; causes multi-organ vasculopathy"
    ),
    "Lyso-Gb3 (Globotriaosylsphingosine)": (
        "Deacylated form of Gb3; primary Fabry biomarker (plasma); diagnostic: >2 nM males, >0.8 nM females; "
        "directly neurotoxic to DRG neurons; correlates with disease burden and ERT response"
    ),
    "Corneal verticillata": (
        "Whorl-like corneal opacity (vortex keratopathy) — PATHOGNOMONIC for Fabry; "
        "slit-lamp examination: 95% males, 70% females; Gb3 in corneal epithelium; does not impair vision"
    ),
    "Angiokeratoma": (
        "Dark red-purple punctate skin lesions — PATHOGNOMONIC distribution (buttocks/genitalia/umbilicus) "
        "in 66% classic males; Gb3 in cutaneous capillary endothelium; "
        "distribution pattern distinguishes Fabry from isolated angiokeratoma (Fordyce)"
    ),
    "Posterior circulation stroke (vertebrobasilar)": (
        "88% of Fabry strokes in vertebrobasilar territory — PATHOGNOMONIC pattern; "
        "occipital, cerebellar, brainstem, thalamic involvement; "
        "young stroke + posterior circulation → always test Lyso-Gb3"
    ),
    "Migalastat (amenable mutations)": (
        "Oral pharmacological chaperone for amenable GLA missense mutations only; "
        "stabilises misfolded α-Gal A in ER; promotes lysosomal trafficking; "
        "MANDATORY: fabry-database.org amenability check before prescribing; FDA 2018 / EMA 2016"
    ),
    "Agalsidase (ERT)": (
        "Enzyme replacement therapy: agalsidase-alfa (Replagal, 0.2 mg/kg q2w, EMA 2001) and "
        "agalsidase-beta (Fabrazyme, 1 mg/kg q2w, FDA 2003); recombinant human α-Gal A; Level A"
    ),
    "Pegunigalsidase-alfa": (
        "Elfabrio (Chiesi); pegylated recombinant human α-Gal A; third-generation ERT; "
        "extended half-life; reduced immunogenicity; 1 mg/kg q2w IV; FDA/EMA 2023; BALANCE trial"
    ),
    "Lyonization": (
        "X-inactivation (Lyon hypothesis) — random inactivation of one X chromosome per cell in females; "
        "creates tissue mosaicism in heterozygous Fabry females; "
        "accounts for variable phenotype severity in females (10–80% α-Gal A activity)"
    ),
    "DBS Lyso-Gb3 (Neonatal Screening)": (
        "Dried blood spot Lyso-Gb3 assay — preferred neonatal screening biomarker for Fabry; "
        "detects affected males reliably; DBS α-Gal A alone misses heterozygous females; "
        "Taiwan/Austria NBS programs use DBS Lyso-Gb3 + GLA sequencing"
    ),
    "GBP/PGB dual use (Fabry)": (
        "Gabapentin/Pregabalin — UNIQUE DUAL ROLE in Fabry: "
        "(1) neuropathic pain (acroparesthesias from DRG Gb3 accumulation) Level B; "
        "(2) antiseizure (post-stroke focal/GTCS) Level B; "
        "no CYP3A4 interaction; renal dose adjustment required in Fabry nephropathy"
    ),
}

# ---------------------------------------------------------------------------
# DATA-LAYER FUNCTIONS
# ---------------------------------------------------------------------------

def get_overview():
    """Return top-level clinical overview dict for GLA / Fabry Disease."""
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "cohort_size": COHORT_SIZE,
        "disease": (
            "GLA (Alpha-Galactosidase A / α-Gal A, EC 3.2.1.22, GH27 family, 429 aa, ~46 kDa) encodes "
            "the lysosomal enzyme that cleaves terminal α-galactosyl residues from Gb3 "
            "(globotriaosylceramide/GL-3) and Lyso-Gb3 (globotriaosylsphingosine). "
            "GLA hemizygous/biallelic LOF → α-Gal A deficient → Gb3 + Lyso-Gb3 accumulate → "
            "multi-organ small-vessel vasculopathy in endothelium, DRG neurons, cardiomyocytes, podocytes. "
            "Fabry Disease / Anderson-Fabry Disease (OMIM *300644/#301500). "
            "X-LINKED — UNIQUE IN LYSOSOMAL SERIES (all others AR). NO CARRIER STATE. "
            "3 patient subgroups: Classic Male 55% (<1% α-Gal A), Late-Onset Male 15% (1–10%), "
            "Classic Female 30% (10–80% via lyonization). "
            "4 approved ERT/chaperone treatments: agalsidase-alfa (EMA 2001), agalsidase-beta (FDA 2003), "
            "migalastat (FDA 2018, AMENABLE ONLY), pegunigalsidase-alfa (FDA/EMA 2023). "
            "All seizures secondary to cerebrovascular disease (no primary Fabry epilepsy). "
            "Drug resistance 40%. GBP/PGB DUAL USE (neuropathic pain + antiseizure) — UNIQUE. "
            "Typical antipsychotics HIGH RISK (QTc + HCM). 40-patient cohort."
        ),
        "protein": (
            "α-Gal A (Alpha-Galactosidase A) — 429 amino acids, ~46 kDa, "
            "GH27 (glycoside hydrolase family 27), EC 3.2.1.22; "
            "lysosomal hydrolase cleaving terminal α-D-galactosyl residues from "
            "Gb3 (globotriaosylceramide/GL-3) and Lyso-Gb3 (globotriaosylsphingosine); "
            "homodimer; requires cofactor saposin B for optimal activity in some contexts; "
            "GLA gene Xq22.1 — X-linked (not autosomal recessive)"
        ),
        "mechanism": (
            "GLA hemizygous/biallelic LOF → α-Gal A enzyme deficient → "
            "Gb3 (globotriaosylceramide) + Lyso-Gb3 (globotriaosylsphingosine) accumulate in: "
            "vascular endothelial cells → small-vessel vasculopathy → stroke/TIA; "
            "DRG neurons → acroparesthesias (burning extremity pain); "
            "cardiomyocytes → hypertrophic cardiomyopathy (HCM); "
            "podocytes/renal tubular cells → proteinuria → renal failure; "
            "corneal epithelium → corneal verticillata (PATHOGNOMONIC); "
            "sweat gland autonomic neurons → hypohidrosis/anhidrosis → hyperthermia crisis"
        ),
        "pathognomonic_note": (
            "CORNEAL VERTICILLATA (whorl-like corneal opacity): 95% males, 70% females — "
            "PATHOGNOMONIC for Fabry disease; slit-lamp examination mandatory. "
            "ANGIOKERATOMAS (dark red-purple skin lesions, buttocks/genitalia/umbilicus): "
            "66% classic males — distribution PATHOGNOMONIC. "
            "POSTERIOR CIRCULATION STROKES (vertebrobasilar territory): 88% of Fabry strokes — "
            "young stroke + posterior circulation → always test Lyso-Gb3."
        ),
        "diagnostic_hierarchy": (
            "1. Young posterior circulation stroke/TIA + acroparesthesias + corneal verticillata → "
            "Lyso-Gb3 plasma immediately (primary biomarker). "
            "2. DBS α-Gal A (males) or Lyso-Gb3 (females + males) for diagnosis. "
            "3. GLA gene sequencing (Xq22.1) — hemizygous/heterozygous variant confirmation. "
            "4. Fabry Variants Database amenability check MANDATORY before treatment selection. "
            "5. POLG1 exclusion before VPA (CPIC Level A). "
            "6. ECG/echo for QTc/HCM monitoring — mandatory before any QTc-active drug."
        ),
        "unique_features": (
            "X-linked NOT autosomal recessive — unique in lysosomal epilepsy series; "
            "NO carrier state (all heterozygous females at risk); "
            "GBP/PGB dual use (neuropathic pain + antiseizure) — unique in lysosomal series; "
            "migalastat amenable mutations only (pharmacogenomics mandatory); "
            "typical antipsychotics HIGH RISK (QTc + HCM); "
            "4 approved ERT/chaperone treatments; "
            "pegunigalsidase-alfa (Elfabrio) — third-generation pegylated ERT (2023); "
            "all seizures secondary to cerebrovascular disease (no primary epilepsy); "
            "posterior circulation stroke PATHOGNOMONIC (88% vertebrobasilar)"
        ),
        "key_pharmacological_distinctions": {
            "GBP_PGB_dual_use": "UNIQUE — treats acroparesthesias (DRG Gb3 neuropathic pain) AND post-stroke seizures; Level B both",
            "typical_antipsychotics": "HIGH RISK — QTc prolongation + Fabry HCM → Torsades; 25% misdiagnosed with panic/anxiety",
            "migalastat_amenable_only": "PHARMACOGENOMICS MANDATORY — fabry-database.org check before prescribing; non-amenable = no benefit",
            "cbz_oxc": "CAUTION (not absolute CI) — hyponatremia + renal impairment + CYP3A4 → ERT interaction; sodium monitoring",
            "pht_relative_ci": "RELATIVE-CI — DRG neuropathy additive (Gb3 + PHT neuropathy); IV LEV replaces fosphenytoin in SE",
            "vpa_safe": "SAFE — GLA lysosomal (NOT mitochondrial); POLG1 exclusion mandatory; VPA + warfarin → enhanced INR",
            "lev_preferred": "PREFERRED — minimal drug interactions with anticoagulants (NOAC/warfarin); renal dose adjust in Fabry nephropathy",
            "ert_cbz_interaction": "CBZ CYP3A4 induction may reduce agalsidase plasma levels — monitor Lyso-Gb3 if co-prescribed",
            "pegunigalsidase": "Third-generation pegylated ERT (Elfabrio) — FDA/EMA 2023; extended half-life; BALANCE trial",
        },
        "discovery": "Anderson H (1898) + Fabry J (1898) — simultaneous independent description; "
                     "Brady RO et al. (1967) — α-Gal A enzyme deficiency identified",
        "pathognomonic_features": [
            "Corneal verticillata (whorl-like corneal opacity): 95% males, 70% females — slit-lamp PATHOGNOMONIC",
            "Angiokeratomas (buttocks/genitalia/umbilicus): 66% classic males — distribution PATHOGNOMONIC",
            "Posterior circulation strokes (vertebrobasilar territory): 88% of Fabry strokes — PATHOGNOMONIC pattern",
            "Lyso-Gb3 plasma >2 nM (males) / >0.8 nM (females) — primary biomarker (sensitivity 98%/82%)",
        ],
        "treatment_highlights": [
            "Agalsidase-alfa (Replagal) 0.2 mg/kg q2w IV — Level A ERT (EMA 2001)",
            "Agalsidase-beta (Fabrazyme) 1 mg/kg q2w IV — Level A ERT (FDA/EMA 2003)",
            "Migalastat (Galafold) 123 mg oral QOD — Level A Chaperone (FDA 2018) — AMENABLE MUTATIONS ONLY",
            "Pegunigalsidase-alfa (Elfabrio) 1 mg/kg q2w IV — Level A Pegylated ERT (FDA/EMA 2023)",
            "GBP/PGB — Level B DUAL USE: neuropathic pain (acroparesthesias) + antiseizure",
            "LEV — Level B AED: preferred (minimal drug interactions with anticoagulants)",
            "VPA — Level B (POLG1 exclusion mandatory; enhanced INR monitoring with warfarin)",
            "Anticoagulation/antiplatelet — Level A stroke prevention",
        ],
        "seizure_pct_overall": 55,
        "seizure_pct_classic_male": 55,
        "seizure_pct_late_onset_male": 8,
        "seizure_pct_classic_female": 8,
        "drug_resistant_pct": 40,
        "posterior_circulation_stroke_pct": 88,
        "corneal_verticillata_males_pct": 95,
        "corneal_verticillata_females_pct": 70,
        "angiokeratoma_pct": 66,
        "hcm_pct": 60,
        "drg_neuropathy_pct": 70,
        "psychiatric_misdiagnosis_pct": 25,
        "on_ert_pct": 72,
        "on_migalastat_pct": 18,
        "on_lev_pct": 55,
        "on_gbp_pgb_pct": 48,
        "standards": STANDARDS,
    }


def get_breakdown():
    """Return detailed breakdown: etiologies, seizure types, triggers, treatments, CIs, thresholds."""
    return {
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "treatment_hierarchy": [
            "Step 1: ERT — agalsidase-alfa (0.2 mg/kg q2w) OR agalsidase-beta (1 mg/kg q2w) — Level A; "
            "check Fabry Variants Database → if amenable: migalastat (Galafold) 123 mg QOD oral",
            "Step 2: Anticoagulation/antiplatelet — Level A stroke prevention (aspirin, NOAC, warfarin per cardiology); "
            "NOAC preferred (fewer AED interactions); AED selection must minimise anticoagulant interactions",
            "Step 3: GBP/PGB (Gabapentin/Pregabalin) — Level B DUAL USE: "
            "neuropathic pain (acroparesthesias) + antiseizure (post-stroke focal/GTCS); "
            "renal dose adjust (Fabry nephropathy); preferred first AED in Fabry",
            "Step 4: LEV (Levetiracetam) — Level B for post-stroke seizures; IV formulation for SE; "
            "minimal anticoagulant interactions; renal dose adjust (GFR monitoring mandatory)",
            "Step 5: VPA — Level B for GTCS; POLG1 exclusion mandatory; INR monitoring if warfarin co-prescribed",
            "Step 6: LTG — safe with anticoagulants (glucuronidation only); focal/GTCS adjunct; "
            "minimal CYP3A4 interaction (no ERT interaction)",
            "AVOID: Typical Antipsychotics HIGH RISK (QTc + HCM → Torsades); "
            "Fosphenytoin RELATIVE-CI (DRG neuropathy + IV LEV preferred in SE); "
            "CBZ/OXC CAUTION (hyponatremia + CYP3A4 → ERT interaction + neuropathy); "
            "migalastat to non-amenable mutations (prescribing error)",
        ],
        "gla_biomarker_summary": {
            "Lyso_Gb3_plasma_male": {"threshold": ">2 nM", "sensitivity": "98%", "specificity": "97%", "use": "Primary diagnostic + ERT monitoring"},
            "Lyso_Gb3_plasma_female": {"threshold": ">0.8 nM", "sensitivity": "82%", "specificity": "89%", "use": "Primary biomarker in heterozygous females"},
            "alpha_Gal_A_DBS_male": {"threshold": "<1% normal", "sensitivity": "99%", "specificity": "99%", "use": "NBS/diagnosis in males (unreliable in females)"},
            "urine_Gb3": {"threshold": ">1.5 μg/mg Cr", "sensitivity": "90%", "specificity": "85%", "use": "Secondary biomarker; ERT monitoring"},
            "GLA_WES": {"threshold": "Hemizygous/heterozygous GLA pathogenic variant", "sensitivity": "100%", "specificity": "100%", "use": "Confirmatory + amenability check"},
        },
        "subtype_severity_matrix": {
            "Classic Male (hemizygous <1% α-Gal A)": (
                "55% cohort · full phenotype · acroparesthesias 4–8y · angiokeratoma 66% · "
                "corneal verticillata 95% · HCM 60% · stroke 88% posterior circulation · seizures 55%"
            ),
            "Late-Onset Male (hemizygous 1–10% α-Gal A)": (
                "15% cohort · cardiac/renal dominant · HCM ± renal failure · "
                "angiokeratoma minimal · seizures rare · Lyso-Gb3 1–5 nM"
            ),
            "Classic Female (heterozygous, lyonization)": (
                "30% cohort · variable (lyonization) · corneal verticillata 70% (PATHOGNOMONIC) · "
                "cardiac 60% · renal 40% · acroparesthesias 60% · seizures 8% · Lyso-Gb3 >0.8 nM"
            ),
        },
    }


def get_definitions():
    """Return definitions, diagnostic algorithm, glossary, key concepts, and standards."""
    return {
        "diagnostic_algorithm": DIAGNOSTIC_ALGORITHM,
        "gla_glossary": GLOSSARY,
        "key_concepts": KEY_CONCEPTS,
        "standards": STANDARDS,
        "differential_diagnosis": {
            "Fabry_vs_isolated_Gaucher_Type3_GBA": (
                "Gaucher Type 3 (GBA): AR; horizontal saccade palsy PATHOGNOMONIC; action myoclonus 75%; "
                "ERT crosses visceral only (not BBB); no posterior circulation stroke; "
                "GBA biallelic mutation; Lyso-Gb1 biomarker (not Lyso-Gb3) → "
                "distinguished by different biomarkers, inheritance, and seizure type"
            ),
            "Fabry_vs_multiple_sclerosis_MS": (
                "MS: demyelinating; posterior circulation white matter lesions on MRI; "
                "CSF oligoclonal bands; no corneal verticillata; no angiokeratoma; "
                "no Lyso-Gb3 elevation; no family history consistent with X-linked; "
                "MS lesions disseminated in space/time vs Fabry vasculopathy → "
                "Fabry: Lyso-Gb3 + corneal verticillata + angiokeratoma + family history"
            ),
            "Fabry_vs_cryptogenic_stroke_young": (
                "Cryptogenic stroke <50y: always test Lyso-Gb3 to exclude Fabry; "
                "Fabry-specific: posterior circulation territory (88%), acroparesthesias, "
                "corneal verticillata, HCM, family history (X-linked pattern), male sex; "
                "Fabry identified in 0.5–1% of young cryptogenic stroke — important treatable cause"
            ),
            "Fabry_vs_CADASIL": (
                "CADASIL (NOTCH3): autosomal dominant; migraine with aura; lacunar infarcts; "
                "deep white matter lesions + anterior temporal pole (PATHOGNOMONIC for CADASIL); "
                "no corneal verticillata; no angiokeratoma; no Lyso-Gb3 elevation; "
                "NOTCH3 mutation; GOM deposits on skin biopsy electron microscopy → "
                "Fabry: Lyso-Gb3 + Xq22.1 mutation + corneal verticillata distinguishes"
            ),
            "Fabry_vs_isolated_angiokeratoma_Fordyce": (
                "Fordyce angiokeratoma: benign isolated vascular lesion; no systemic disease; "
                "normal α-Gal A; normal Lyso-Gb3; no corneal verticillata; no HCM; no stroke; "
                "distribution differs from Fabry (scrotal/labial vs Fabry: bathing suit distribution); "
                "Fabry PATHOGNOMONIC: angiokeratoma + Lyso-Gb3 >2 nM + corneal verticillata"
            ),
        },
    }
