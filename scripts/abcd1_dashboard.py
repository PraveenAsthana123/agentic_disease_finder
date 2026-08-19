#!/usr/bin/env python3
"""ABCD1 / X-linked Adrenoleukodystrophy (X-ALD) Epilepsy Dashboard — seed data module.

X-ALD: ABCD1 LOF → peroxisomal VLCFA beta-oxidation failure → Very Long Chain Fatty Acids (C26:0, C24:0)
accumulate in CNS white matter, adrenal cortex, Leydig cells, and all tissues.
ABCD1 (Xq28) encodes Adrenoleukodystrophy Protein (ALDP) — peroxisomal membrane half-transporter (PMP);
745 aa; 6 transmembrane helices; homodimerizes; transports VLCFA-CoA esters into peroxisome for beta-oxidation.
X-LINKED: males hemizygous (fully affected); females heterozygous (65% develop AMN-like myelopathy by age 60).

PHENOTYPIC SPECTRUM (MALES):
(1) Childhood Cerebral ALD (CCALD): 35% of males; onset 4-8yr; rapid neuroinflammatory demyelination starting
    in splenium of corpus callosum + parieto-occipital WM → anterior progression; gadolinium enhancement =
    ACTIVE neuroinflammation = INTERVENTION WINDOW; seizures 90%+; fatal if untreated; genotype DOES NOT
    predict phenotype (same mutation → CCALD in one brother, AMN in another).
(2) Adolescent Cerebral: ~7%; onset 10-21yr; similar to CCALD but slightly slower progression.
(3) Adult Cerebral: ~5%; usually AMN converting to cerebral form; MRI surveillance essential.
(4) AMN (Adrenomyeloneuropathy): 45% of males; onset 20-40yr; progressive spastic paraparesis + peripheral
    neuropathy; VLCFA accumulation in cervical cord + Schwann cells; seizures 15-20% (usually cerebral
    conversion); NO HSCT benefit for established AMN axonopathy (arrests only inflammatory form).
(5) Addison-only: 10% of males; isolated adrenal insufficiency; NO primary CNS disease at presentation;
    seizures ONLY from adrenal crisis (cortisol deficiency → hypoglycemia/hyponatremia/hypotension).
(6) Female heterozygotes: 65% develop AMN-like myelopathy by age 60; adrenal insufficiency rare (<1%);
    cerebral involvement rare (<2%); seizures 5-10%; plasma VLCFA elevated in 85% (NBS insufficient).

ADRENAL INSUFFICIENCY (71% OF MALES):
  Cortisol + aldosterone deficiency from VLCFA accumulation in adrenal cortex (zona fasciculata + reticularis);
  ACTH rises to compensate → progressive adrenal cortex destruction; PRIMARY ADRENAL INSUFFICIENCY;
  Adrenal crisis triggered by ANY stress (infection, surgery, fasting, trauma) → LIFE-THREATENING:
  cortisol deficiency → hypoglycemia + hypotension + hyponatremia → metabolic SEIZURES + coma.
  Hydrocortisone (cortisol replacement) + fludrocortisone (aldosterone) MANDATORY in ALL males.
  Perioperative STRESS DOSING (100mg IV hydrocortisone at induction + 50mg q6h × 24-48h) MANDATORY.
  Enzyme-inducing AEDs (PHT, CBZ, OXC, PB) REDUCE corticosteroid levels → adrenal crisis RISK.

LOES SCORE / NRS (INTERVENTION ELIGIBILITY):
  Loes Score 0-34: MRI white matter severity; 0=normal; ≤9 = eligible for HSCT/gene therapy;
  10-34 = advanced disease; gadolinium enhancement = active BBB breakdown = window for intervention.
  NRS (Neurological Disability Scale) 0-25: 0=normal; ≤1 = HSCT/GT eligible; ≥2 = diminishing benefit.
  Serial MRI every 6 months for males age 3-12yr; immediately at any neurological symptom.

TREATMENT LANDSCAPE:
  (1) ALLOGENEIC HSCT (Level A): gold standard for CCALD Loes ≤9 + NRS ≤1 + Gd-enhancement;
      matched sibling donor preferred; matched unrelated acceptable; arrests neuroinflammation
      permanently; 5-yr survival 92% (early) vs 45% (late); NO benefit for AMN axonopathy.
  (2) GENE THERAPY — Skysona (elivaldogene autotemcel, eli-cel): FDA approved Aug 2022 for CCALD ≤17yr
      without HLA-matched sibling donor; autologous CD34+ HSC transduced with LentiGlobin-ABCD1 lentiviral
      vector; REMS required (malignancy risk: AML/MDS in ~40% at 7yr follow-up — BOXED WARNING);
      Phase 2/3 ALD-102 trial; monitoring q3M CBC + bone marrow biopsy annually.
  (3) ADRENAL HORMONE REPLACEMENT (MANDATORY): hydrocortisone 15-20mg/m²/day (adult: 15-25mg divided
      TID) + fludrocortisone 0.05-0.2mg daily; MANDATORY regardless of CNS phenotype; stress dosing protocol.
  (4) LORENZO'S OIL (Level C, presymptomatic only): 4:1 glyceryl trioleate:glyceryl trierucate (GTO:GTE);
      normalizes plasma VLCFA C26:0 within 4 weeks; does NOT benefit established cerebral ALD or AMN;
      Level C evidence: delays cerebral conversion in presymptomatic males (Moser 2005 Johns Hopkins series);
      side effects: thrombocytopenia (platelet count monthly), GI intolerance; low-fat diet required.
  (5) LEV (FIRST-LINE AED, Level C): preferred AED; NO enzyme induction; no adrenal interaction;
      minimal drug interactions with cortisol; IV/PO; effective CCALD focal + GTCS seizures.
  (6) POLG1 EXCLUSION BEFORE VPA (CPIC A): mandatory WES/mtDNA screening before valproate; VPA-induced
      hepatotoxicity risk; also adrenal function monitoring if VPA used (hepatic VPA metabolism).
  (7) NEWBORN SCREENING (NBS): C26:0-lyso-phosphatidylcholine on DBS; implemented ~30 US states;
      ~2/100,000 NBS positive; detects presymptomatic ABCD1 LOF; allows early surveillance + Lorenzo's Oil.
"""
import random

GENE = "ABCD1"
LOCUS = "Xq28"
OMIM_GENE = "300371"
OMIM_DISEASE = "300100"
INHERITANCE = (
    "X-linked (XL) — males hemizygous (fully affected); females heterozygous "
    "(65% develop AMN-like myelopathy by age 60; adrenal insufficiency rare <1%; "
    "cerebral involvement rare <2%; seizures 5-10%); NO carrier females unaffected lifelong."
)
COHORT_SIZE = 40
DISEASE_MECHANISM = (
    "ABCD1 LOF → Adrenoleukodystrophy Protein (ALDP) deficiency → peroxisomal VLCFA "
    "beta-oxidation failure → Very Long Chain Fatty Acids (C26:0, C24:0, C25:0) accumulate "
    "in ALL tissues but primarily CNS white matter + adrenal cortex. CNS: VLCFA triggers "
    "neuroinflammatory cascade in oligodendrocytes/microglia → demyelination (CCALD) and/or "
    "axonal degeneration (AMN). Adrenal: VLCFA destroys zona fasciculata + reticularis → "
    "primary adrenal insufficiency (cortisol + aldosterone deficiency, 71% of males). "
    "Adrenal crisis (cortisol deficiency → hypoglycemia/hyponatremia/hypotension) = "
    "DOMINANT METABOLIC SEIZURE TRIGGER. Genotype does NOT predict phenotype: same ABCD1 "
    "mutation → CCALD in one brother, AMN in another, Addison-only in a third. "
    "Peroxisomal disease (NOT lysosomal): VLCFAs oxidized ONLY in peroxisome; "
    "mitochondria cannot compensate; no lysosomal enzyme replacement possible."
)

ETIOLOGIES = [
    {
        "name": "Childhood Cerebral ALD (CCALD)",
        "pct": 35,
        "n": 14,
        "sex": "Males (hemizygous)",
        "onset_age": "4-8 years",
        "seizure_risk": "90-95% (most common form; seizures often PRESENTING symptom)",
        "eeg": (
            "Posterior-predominant delta/theta slowing (parieto-occipital WM lesions); "
            "focal occipital/parietal spikes; as disease advances → bilateral synchrony + "
            "diffuse slowing; SE risk high; normal EEG early in presymptomatic period"
        ),
        "mri": "Posterior parieto-occipital WM → corpus callosum splenium; Gd+ = active inflammation; Loes 4-18",
        "loes_range": "4-18 at diagnosis; ≤9 = HSCT/GT eligible",
        "hsct_eligible": True,
        "gt_eligible": True,
        "ert_available": False,
        "variant_detail": (
            "Any ABCD1 LOF; phenotype UNPREDICTABLE from genotype; missense/nonsense/frameshift/deletion all reported; "
            "early-onset 4-6yr = most common; rapid progression (months from symptom to nonverbal); Gd-enhancement "
            "appears BEFORE neurological deficit (NBS detection allows surveillance)"
        ),
    },
    {
        "name": "Adolescent Cerebral ALD",
        "pct": 7,
        "n": 3,
        "sex": "Males (hemizygous)",
        "onset_age": "10-21 years",
        "seizure_risk": "85-90% (similar to CCALD but slightly slower)",
        "eeg": "Posterior-predominant slowing + focal spikes; SE risk; frontal involvement later",
        "mri": "Similar to CCALD but may start frontally in some; Loes 3-15 at intervention",
        "loes_range": "3-15",
        "hsct_eligible": True,
        "gt_eligible": True,
        "ert_available": False,
        "variant_detail": "Same spectrum as CCALD; unclear why adolescent vs childhood onset; phenotypic modifier genes suspected",
    },
    {
        "name": "Adult Cerebral ALD (AMN→Cerebral Conversion)",
        "pct": 5,
        "n": 2,
        "sex": "Males (hemizygous)",
        "onset_age": "21-50+ years",
        "seizure_risk": "70-80% (usually converting from AMN; seizures mark cerebral involvement)",
        "eeg": "Frontal or multifocal slowing + spikes; posterior changes if WM distribution; SE risk",
        "mri": "Variable distribution; frontal or parieto-occipital; Gd+ in active phase",
        "loes_range": "Variable",
        "hsct_eligible": False,
        "gt_eligible": False,
        "ert_available": False,
        "variant_detail": "AMN-onset male developing cerebral lesions; HSCT data limited; LEV first-line AED; adrenal replacement mandatory",
    },
    {
        "name": "AMN — Adrenomyeloneuropathy",
        "pct": 40,
        "n": 16,
        "sex": "Males (hemizygous)",
        "onset_age": "20-40 years",
        "seizure_risk": "15-20% (seizures indicate cerebral conversion; NOT from axonopathy alone)",
        "eeg": (
            "Usually normal or mildly diffusely slow; focal changes only if cerebral lesions develop; "
            "peripheral neuropathy on NCS (NOT EEG finding)"
        ),
        "mri": "Spinal cord T2 hyperintensity (dorsal columns, corticospinal tracts); Loes 0-2 unless cerebral",
        "loes_range": "0-4 (spinal dominant)",
        "hsct_eligible": False,
        "gt_eligible": False,
        "ert_available": False,
        "variant_detail": (
            "Progressive spastic paraparesis + peripheral neuropathy; axonal degeneration (NOT demyelination); "
            "HSCT does NOT arrest AMN axonopathy; 40-50% AMN males will develop cerebral involvement → "
            "MRI surveillance every 12 months; adrenal insufficiency 71%"
        ),
    },
    {
        "name": "Addison-Only (Adrenal Insufficiency, No CNS)",
        "pct": 10,
        "n": 4,
        "sex": "Males (hemizygous)",
        "onset_age": "Any age (often childhood/adolescence for AI)",
        "seizure_risk": "AI-crisis-only: 15-20% (metabolic seizures from cortisol deficiency — not epileptic in origin)",
        "eeg": "Normal interictal EEG; during crisis: diffuse slowing (metabolic); resolves with cortisol",
        "mri": "NORMAL brain MRI; MRI q6M surveillance mandatory (cerebral conversion 30-40% over time)",
        "loes_range": "0 at diagnosis",
        "hsct_eligible": False,
        "gt_eligible": False,
        "ert_available": False,
        "variant_detail": (
            "Isolated primary adrenal insufficiency (Addison disease); ABCD1 most common single-gene "
            "cause of Addison disease in males; seizures = adrenal crisis symptoms (cortisol/aldosterone "
            "deficiency → metabolic derangement); MANDATORY: hydrocortisone + fludrocortisone + "
            "VLCFA + MRI surveillance (30-40% eventually develop AMN or cerebral ALD)"
        ),
    },
    {
        "name": "Female Heterozygous (AMN-like Myelopathy)",
        "pct": 3,
        "n": 1,
        "sex": "Females (heterozygous)",
        "onset_age": "40-60 years (AMN-like)",
        "seizure_risk": "5-10% (mild; usually AMN myelopathy with spasticity, not seizures)",
        "eeg": "Usually normal or mildly slow; cerebral involvement rare; seizures rare",
        "mri": "Spinal cord predominant if symptomatic; brain MRI usually normal",
        "loes_range": "0-2",
        "hsct_eligible": False,
        "gt_eligible": False,
        "ert_available": False,
        "variant_detail": (
            "65% develop AMN-like myelopathy by age 60; adrenal insufficiency rare (<1%); "
            "VLCFA elevated in 85%; brain MRI usually normal; follow as AMN; no established "
            "disease-modifying therapy; Lorenzo's Oil used empirically"
        ),
    },
]

SEIZURE_TYPES = [
    {
        "type": "Focal Onset with Posterior Predominance (Occipital/Parietal)",
        "pct": 60,
        "eeg": (
            "Posterior (O1/O2, P3/P4) spike-wave + focal slowing; maps to parieto-occipital "
            "WM lesion; visual aura/amaurosis common (posterior cortex involvement); "
            "secondary GTCS frequent"
        ),
    },
    {
        "type": "Generalized Tonic-Clonic (GTCS)",
        "pct": 45,
        "eeg": "Bilateral synchronous spike-wave; often secondary generalisation from focal onset",
    },
    {
        "type": "Status Epilepticus (SE)",
        "pct": 25,
        "eeg": (
            "High risk in acute CCALD exacerbation (WM oedema + Gd+ = active inflammation); "
            "refractory SE in advanced disease; IV LEV + midazolam first-line; "
            "avoid PHT IV → adrenal crisis risk"
        ),
    },
    {
        "type": "Tonic Seizures",
        "pct": 20,
        "eeg": "Frontal fast activity + EMG artifact; bilateral motor involvement",
    },
    {
        "type": "Metabolic Seizures (Adrenal Crisis)",
        "pct": 18,
        "eeg": (
            "Diffuse slow background during crisis (cortisol deficiency → "
            "hypoglycemia/hyponatremia → metabolic encephalopathy); resolves with "
            "IV hydrocortisone + dextrose; NOT epileptic in origin — correct underlying "
            "metabolic derangement FIRST before adding AED"
        ),
    },
    {
        "type": "Absence-like / Atypical Absence",
        "pct": 10,
        "eeg": "Generalised 2-3Hz spike-wave; frontal WM lesion contribution",
    },
]

TRIGGERS = [
    {
        "trigger": "Adrenal Crisis (Cortisol Deficiency)",
        "pct": 45,
        "note": (
            "ANY stressor (infection, surgery, vomiting, fasting, trauma) → cortisol deficiency "
            "in AI males → hypoglycemia + hyponatremia + hypotension → metabolic seizure. "
            "Action: IV 100mg hydrocortisone STAT + dextrose + saline → most AI-related seizures "
            "resolve with cortisol. Perioperative stress dosing MANDATORY."
        ),
    },
    {
        "trigger": "Fever / Intercurrent Infection",
        "pct": 40,
        "note": (
            "Neuroinflammatory exacerbation in CCALD: fever accelerates WM lesion expansion "
            "(BBB disruption + oxidative stress) → seizure threshold lowered; adrenal crisis "
            "simultaneously; aggressive antibiotic + cortisol + temperature management"
        ),
    },
    {
        "trigger": "Perioperative / Surgical Stress",
        "pct": 35,
        "note": (
            "EXTREME HAZARD: adrenal crisis during anaesthesia → cortisol deficiency → "
            "haemodynamic collapse + seizures + death. MANDATORY: 100mg hydrocortisone IV "
            "at induction + 50mg q6h × 24-48h postoperatively; alert anaesthesia team BEFORE procedure. "
            "Special risk: HSCT/gene therapy conditioning — close adrenal monitoring."
        ),
    },
    {
        "trigger": "Rapid WM Lesion Expansion (Active CCALD)",
        "pct": 30,
        "note": (
            "Gadolinium-enhancing lesion expansion = acute neuroinflammatory burst → seizure risk; "
            "immediate MRI + HSCT/GT evaluation; avoid seizure-aggravating AEDs (PHT ABSOLUTE CI). "
            "IV dexamethasone may temporarily reduce oedema (NOT disease-modifying)."
        ),
    },
    {
        "trigger": "Missed Corticosteroid Dose",
        "pct": 28,
        "note": (
            "Hydrocortisone missed → relative cortisol deficiency → fatigue + nausea + risk of crisis. "
            "Education: patient/family must carry emergency hydrocortisone injection kit (100mg IM kit). "
            "Sick-day rules: double/triple dose for illness + fever."
        ),
    },
    {
        "trigger": "Head Trauma / Falls",
        "pct": 20,
        "note": (
            "Seizures pre-dispose to falls; spastic gait (AMN) increases fall risk; head trauma "
            "may exacerbate neuroinflammation in CCALD; adrenal crisis triggered by trauma stress"
        ),
    },
    {
        "trigger": "Sleep Deprivation / Circadian Disruption",
        "pct": 18,
        "note": "Standard seizure trigger; may be amplified by adrenal cortisol circadian rhythm disruption",
    },
    {
        "trigger": "Enzyme-Inducing AED Introduction / Dose Increase",
        "pct": 15,
        "note": (
            "PHT/CBZ/OXC/PB initiation → CYP3A4 induction → accelerated cortisol catabolism → "
            "relative adrenal insufficiency in borderline-sufficient males → clinical adrenal crisis; "
            "ACTION: corticosteroid levels q2W × 3M after any enzyme inducer start; "
            "PHT ABSOLUTE CI regardless"
        ),
    },
]

TREATMENTS = [
    {
        "drug": "Allogeneic HSCT (Hematopoietic Stem Cell Transplantation)",
        "class": "Disease-Modifying — Level A (Early CCALD)",
        "evidence": "Level A (CCALD Loes ≤9, NRS ≤1, Gd-enhancement); Eichler et al. NEJM 2017; Orchard series",
        "dose": (
            "Conditioning: myeloablative (busulfan + cyclophosphamide ± fludarabine); "
            "matched sibling donor preferred (lowest GVHD); matched unrelated acceptable; "
            "cord blood possible (slower engraftment); 5-yr overall survival 92% (early) vs 45% (late)"
        ),
        "moa": (
            "Donor haematopoietic cells → microglia reconstitution → restore VLCFA metabolism "
            "in CNS; arrests neuroinflammatory cascade; DOES NOT reverse established axonal damage; "
            "benefit seen only when Gd+ (active inflammation) present at transplant"
        ),
        "monitoring": "MRI q3M × 2yr post-HSCT; NRS monthly × 6M; CBC (engraftment); cortisol (preserve AI monitoring)",
        "ci": "Loes >9 or NRS >1 → survival benefit lost; AMN without cerebral involvement → not indicated",
    },
    {
        "drug": "Skysona (Elivaldogene Autotemcel / Eli-cel)",
        "class": "Gene Therapy — FDA Approved Aug 2022 (CCALD, ≤17yr, no HLA match)",
        "evidence": "Phase 2/3 ALD-102; FDA accelerated approval Aug 16 2022; REMS required (BOXED WARNING malignancy)",
        "dose": (
            "Autologous CD34+ HSC + myeloablative conditioning + LentiGlobin-ALD lentiviral vector; "
            "single infusion; manufacturing 2-3 months; only at qualified REMS centres; "
            "no suitable HLA-matched sibling donor required to qualify"
        ),
        "moa": (
            "Autologous CD34+ HSC transduced with lentiviral vector carrying functional ABCD1 gene; "
            "avoids GVHD (autologous); restores ALDP in microglia/macrophages; arrests neuroinflammation; "
            "SAME mechanism as HSCT (microglial repopulation) but without alloreactivity"
        ),
        "monitoring": (
            "REMS: CBC q1M × 2yr + q3M thereafter; bone marrow biopsy annually × 15yr; "
            "insert site analysis (retroviral integration monitoring); MRI q3M × 2yr; NRS q3M"
        ),
        "ci": "CCALD >17yr (FDA label); Loes >9 or NRS >1 (diminishing benefit); gene therapy not approved for AMN",
    },
    {
        "drug": "Hydrocortisone + Fludrocortisone (Adrenal Replacement — MANDATORY)",
        "class": "Adrenal Hormone Replacement — Mandatory All Males",
        "evidence": "Level A for adrenal insufficiency; standard endocrine care",
        "dose": (
            "Hydrocortisone: 10-12 mg/m²/day divided TID (adult: 15-25mg/day); "
            "fludrocortisone: 0.05-0.2mg once daily; "
            "STRESS DOSING: 100mg IV hydrocortisone at anaesthetic induction + 50mg q6h × 24-48h; "
            "sick day rules: 3× dose for fever/vomiting/surgery; emergency kit (100mg IM syringe) MANDATORY"
        ),
        "moa": "Cortisol (glucocorticoid) + fludrocortisone (mineralocorticoid) replacement; prevents adrenal crisis",
        "monitoring": "Morning cortisol + ACTH q6-12M; electrolytes q3M; growth velocity (paediatric); BP",
        "ci": "None (mandatory); interaction: enzyme-inducing AEDs (PHT/CBZ) reduce effective cortisol levels",
    },
    {
        "drug": "Lorenzo's Oil (GTO:GTE 4:1)",
        "class": "VLCFA Substrate Reduction — Level C (Presymptomatic Only)",
        "evidence": (
            "Level C: Moser 2005 (Johns Hopkins cohort — delays cerebral conversion in "
            "presymptomatic Loes 0 males); NOT effective in established CCALD or AMN"
        ),
        "dose": (
            "Glyceryl trioleate:glyceryl trierucate 4:1 mixture; dose: 2-3mL/kg/day with food; "
            "low-fat diet (< 15% calories from fat) required to reduce endogenous VLCFA synthesis; "
            "normalises plasma C26:0 within 4-8 weeks; GI side effects (nausea, cramps)"
        ),
        "moa": (
            "GTO (oleic acid C18:1) competes with VLCFA for elongase enzymes → reduces C26:0 synthesis; "
            "GTE (erucic acid C22:1) further suppresses C24:0; does NOT fix peroxisomal beta-oxidation; "
            "VLCFA reduced in plasma (diagnostic marker) but CNS levels not reliably reduced — "
            "explains lack of benefit in symptomatic disease"
        ),
        "monitoring": "Platelet count monthly (thrombocytopenia risk — stop if PLT < 100); VLCFA q3M; LFT q6M",
        "ci": "Established cerebral ALD or AMN (no benefit; may cause false reassurance); thrombocytopenia",
    },
    {
        "drug": "Levetiracetam (LEV) — First-Line AED",
        "class": "AED — Level C (first-line in CCALD/AMN+cerebral)",
        "evidence": "Level C; expert consensus; preferred AED in X-ALD due to NO enzyme induction + NO adrenal interaction",
        "dose": "250-1500mg BD (adult); 20-40mg/kg/day paediatric; IV formulation available (SE management)",
        "moa": "SV2A modulation; no CYP induction; NO reduction of corticosteroid levels",
        "monitoring": "Neuropsychiatric adverse effects (especially in CCALD with cognitive decline); renal function",
        "ci": "Renal failure (dose reduction); psychiatric history caution (can worsen in CCALD behavioural phase)",
    },
    {
        "drug": "VPA (Sodium Valproate) — Adjunct (POLG1 Exclusion MANDATORY)",
        "class": "AED — Level C adjunct (exclude POLG1 first)",
        "evidence": "Level C adjunct; POLG1 WES/mtDNA exclusion mandatory (CPIC A) before initiating",
        "dose": "Standard dosing; LFT quarterly; cortisol interaction: monitor adrenal function",
        "moa": "Broad-spectrum; GABA augmentation + Na+ channel modulation",
        "monitoring": "LFT q3M; POLG1 status mandatory pre-initiation; cortisol levels if any adrenal compromise",
        "ci": "POLG1 mutation (fatal hepatotoxicity); hepatic impairment; thrombocytopenia (Lorenzo's Oil co-use)",
    },
    {
        "drug": "Clobazam",
        "class": "AED — Level C adjunct (mild enzyme inducer — low risk vs CBZ/PHT)",
        "evidence": "Level C; relatively safe enzyme inducer; weaker CYP induction than CBZ/PHT",
        "dose": "5-20mg/day (adult); 0.5-1mg/kg/day (paediatric)",
        "moa": "GABA-A positive allosteric modulator (1,5-benzodiazepine)",
        "monitoring": "Cortisol levels q1M × 3M after initiation (enzyme induction, though weak)",
        "ci": "Sedation worsens CCALD cognitive decline; monitor respiratory function (spastic AMN)",
    },
    {
        "drug": "IV Midazolam + IV LEV (SE Protocol)",
        "class": "Acute seizure management — Status Epilepticus",
        "evidence": "Level A for SE management (midazolam); IV LEV preferred over IV PHT in X-ALD",
        "dose": (
            "Midazolam: 0.1-0.2mg/kg IV/IM/intranasal; IV LEV: 20-30mg/kg over 15min; "
            "NEVER IV phenytoin in X-ALD (adrenal crisis risk); IV lacosamide as 3rd-line alternative"
        ),
        "moa": "Midazolam: GABA-A; LEV: SV2A; lacosamide: Na+ slow inactivation",
        "monitoring": "Cortisol level + blood glucose in SE (exclude adrenal crisis trigger)",
        "ci": "PHT/Fosphenytoin IV: ABSOLUTE CI in X-ALD (enzyme induction → immediate cortisol drop → haemodynamic collapse)",
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "PHT / Phenytoin",
        "level": "ABSOLUTE CI",
        "reason": (
            "CYP3A4 strong inducer → accelerated cortisol catabolism → effective cortisol drop → "
            "adrenal crisis in ALL X-ALD males with adrenal insufficiency (71%). "
            "In SE: Fosphenytoin (IV PHT pro-drug) ALSO ABSOLUTE CI — IV LEV replaces. "
            "Also worsens neuroinflammation in CCALD. NEVER use in X-ALD."
        ),
        "alternative": "IV LEV (first-line) or IV lacosamide (third-line SE)",
    },
    {
        "drug": "CBZ / Carbamazepine",
        "level": "RELATIVE CI (Enhanced Monitoring)",
        "reason": (
            "CYP3A4 moderate inducer → reduces cortisol levels ~30-40% → sub-clinical adrenal crisis "
            "risk in borderline adrenal function. Also impairs VLCFA metabolism (minor). "
            "IF used: cortisol level q2W × 3M after starting; dose of hydrocortisone may need +30% increase."
        ),
        "alternative": "LEV (preferred) or LTG (weaker inducer); OXC also RELATIVE CI (weaker than CBZ)",
    },
    {
        "drug": "OXC / Oxcarbazepine",
        "level": "RELATIVE CI (Lower Risk than CBZ)",
        "reason": (
            "Moderate CYP inducer (weaker than CBZ but non-negligible); cortisol levels may fall; "
            "adrenal monitoring required. Risk lower than CBZ/PHT but still RELATIVE CI in X-ALD."
        ),
        "alternative": "LEV preferred; if OXC needed: cortisol q1M × 6M",
    },
    {
        "drug": "Phenobarbitone (PB)",
        "level": "RELATIVE CI",
        "reason": (
            "CYP3A4 + CYP2C9 induction → reduces cortisol clearance (paradoxically complex); "
            "risk lower than PHT in acute setting but chronic use reduces adrenal reserve. "
            "Sedation worsens CCALD cognitive decline."
        ),
        "alternative": "LEV or LTG or CLB (weak inducer)",
    },
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine)",
        "level": "HIGH RISK",
        "reason": (
            "EPS worsens spastic/dystonic CCALD neurological decline; D2 blockade disrupts "
            "adrenal stress response axis; additive sedation in encephalopathy. "
            "Psychiatric features in CCALD (behavioural change, psychosis) MUST NOT be treated "
            "with typical antipsychotics — CCALD is the underlying disease."
        ),
        "alternative": "Atypical antipsychotics (risperidone low dose / quetiapine) if psychiatric features; treat CCALD aggressively",
    },
    {
        "drug": "Anaesthesia / Surgery — EXTREME HAZARD",
        "level": "EXTREME HAZARD (mandatory stress protocol)",
        "reason": (
            "Perioperative stress → cortisol deficiency → haemodynamic collapse → metabolic seizures → "
            "death. MANDATORY PROTOCOL: 100mg hydrocortisone IV at anaesthetic induction + "
            "50mg q6h × 24-48h postoperatively; emergency hydrocortisone kit with patient at all times. "
            "Alert ALL surgical/anaesthetic teams about X-ALD adrenal insufficiency BEFORE any procedure."
        ),
        "alternative": "Mandatory stress dosing; continue baseline hydrocortisone orally or IV throughout",
    },
    {
        "drug": "Corticosteroid (High-Dose IV Steroids) for Acute CCALD",
        "level": "NOT DISEASE-MODIFYING (limit use)",
        "reason": (
            "IV dexamethasone may temporarily reduce WM oedema/BBB permeability (short-term) but "
            "does NOT arrest neuroinflammation; may paradoxically accelerate lesion expansion in some; "
            "HSCT/gene therapy is the ONLY disease-modifying approach; steroids at high dose worsen "
            "adrenal suppression long-term (iatrogenic secondary AI)."
        ),
        "alternative": "HSCT (Level A) or gene therapy (FDA 2022); LEV for seizures; dex only if acute oedema/herniation",
    },
    {
        "drug": "VGB / Vigabatrin",
        "level": "NOT INDICATED",
        "reason": (
            "No evidence of benefit in X-ALD seizures; visual field monitoring impossible in severe "
            "CCALD (cognitive decline); permanent visual field defects unacceptable in already "
            "visually-compromised disease"
        ),
        "alternative": "LEV first-line; CLB or VPA (with POLG1 exclusion) as adjuncts",
    },
]

KEY_CONCEPTS = [
    "ABCD1 — ATP-Binding Cassette Subfamily D Member 1 (Xq28); 745 aa; peroxisomal membrane half-transporter (PMP70-like); 6 TM helices; homodimerizes; transports VLCFA-CoA esters into peroxisome",
    "X-ALD — X-linked Adrenoleukodystrophy; most common inherited peroxisomal disorder (1:17,000 males; 1:16,800 females); peroxisomal (NOT lysosomal) — no ERT possible",
    "VLCFA — Very Long Chain Fatty Acids; C22:0–C26:0 saturated FAs; C26:0 (hexacosanoic acid) is PRIMARY diagnostic marker; oxidised ONLY in peroxisome (NOT mitochondria)",
    "CCALD — Childhood Cerebral ALD; age 4-8yr; rapid neuroinflammatory demyelination; posterior WM dominant; MOST SEVERE phenotype; seizures 90%+; fatal without HSCT/GT if Loes ≤9 window missed",
    "AMN — Adrenomyeloneuropathy; age 20-40yr; axonal degeneration (NOT demyelination); spinal cord + peripheral neuropathy; HSCT does NOT arrest established AMN axonopathy",
    "Loes Score — MRI WM severity scale 0-34; 0=normal; ≤9+NRS≤1 = HSCT/GT eligible window; Gd-enhancement = active neuroinflammation = best treatment window",
    "NRS — Neurological Disability Scale 0-25; 0=normal; ≤1 = optimal HSCT/GT eligibility; serial assessment q3-6M in surveillance",
    "Skysona / Elivaldogene autotemcel (eli-cel) — FDA approved Aug 2022 for CCALD ≤17yr without HLA-matched sibling; autologous CD34+ + LentiGlobin-ABCD1; REMS required (malignancy BOXED WARNING)",
    "Adrenal Insufficiency — 71% of X-ALD males at some point; primary AI (cortisol + aldosterone); MANDATORY hydrocortisone + fludrocortisone; adrenal crisis = dominant METABOLIC seizure trigger",
    "Gadolinium Enhancement — MRI T1+Gd; BBB breakdown = active neuroinflammation = INTERVENTION WINDOW for HSCT/gene therapy; disappears in burnt-out disease",
    "NBS (Newborn Screening) — C26:0-lyso-PC on DBS; ~30 US states; detects presymptomatic ABCD1 LOF; ~2/100,000 positive; enables surveillance + Lorenzo's Oil intervention",
    "Lorenzo's Oil — 4:1 GTO:GTE; normalises plasma VLCFA; Level C prevention only in presymptomatic Loes 0 males; NO benefit in established CCALD or AMN; thrombocytopenia risk",
    "PHT ABSOLUTE CI — CYP3A4 induction → cortisol catabolism → adrenal crisis; also worsens CCALD neuroinflammation; IV Fosphenytoin ALSO ABSOLUTE CI in SE (IV LEV replaces)",
    "CYP3A4 Induction Risk — PHT > CBZ ≈ OXC > PB > CLB; all enzyme-inducing AEDs reduce effective cortisol → relative AI → adrenal crisis risk; LEV (non-inducer) is first-line",
    "POLG1 Exclusion — CPIC A: mandatory WES/mtDNA before VPA initiation; also: VPA hepatotoxicity risk increased in adrenal-compromised state",
    "Genotype-Phenotype Dissociation — SAME ABCD1 mutation → CCALD in one brother + AMN in another + Addison-only in a third; phenotype NOT predictable from genotype; modifier genes suspected",
]

STANDARDS = [
    "Engelen M et al. Orphanet J Rare Dis 2012 (X-ALD guidelines — comprehensive review)",
    "Eichler FS et al. NEJM 2017 (HSCT outcomes + Loes score eligibility criteria)",
    "Orchard PJ et al. NEJM 2022 / Blood 2023 (elivaldogene autotemcel Phase 2/3 ALD-102)",
    "Moser HW et al. Ann Neurol 2005 (Lorenzo's Oil presymptomatic prevention — Johns Hopkins cohort)",
    "Loes DJ et al. AJNR 1994 (Loes MRI scoring system — definitive description)",
    "FDA 2022 Skysona approval + REMS programme (malignancy boxed warning for eli-cel)",
    "ACMG NBS guidelines (C26:0-lyso-PC DBS newborn screening expansion)",
    "Kemp S et al. Ann Neurol 2012 (X-ALD biology, epidemiology and treatment landscape)",
    "van Geel BM et al. Brain 2001 (AMN natural history — long-term follow-up)",
    "POLG1 pharmacogenomics: CPIC guidelines for VPA and mitochondrial disease genes",
    "Cartier N et al. Science 2009 (gene therapy proof-of-concept — lentiviral ABCD1)",
    "Kennedy Krieger Institute X-ALD database (xald.org) — >760 ABCD1 variant repository",
]

THRESHOLDS = [
    {"parameter": "Loes Score", "threshold": "≤9 + Gd+ = HSCT/GT eligible", "action": "IMMEDIATE HSCT or gene therapy referral"},
    {"parameter": "NRS", "threshold": "≤1 = optimal eligibility", "action": "HSCT/GT now; delay = lost window"},
    {"parameter": "Plasma C26:0", "threshold": "Elevated (>0.64 μg/mL typical)", "action": "Confirm ABCD1 sequencing; start surveillance MRI"},
    {"parameter": "MRI Gd-enhancement", "threshold": "Any Gd+ in presymptomatic male", "action": "HSCT/GT consultation within 1 week"},
    {"parameter": "Morning Cortisol", "threshold": "<83 nmol/L (peak <500 on stim test)", "action": "Start hydrocortisone + fludrocortisone"},
    {"parameter": "Platelet Count (Lorenzo's Oil)", "threshold": "<100 × 10⁹/L", "action": "Withhold Lorenzo's Oil; haematology review"},
    {"parameter": "Cortisol Level (on enzyme-inducing AED)", "threshold": "Fall >30% from baseline", "action": "Increase hydrocortisone dose or switch AED to LEV"},
    {"parameter": "HSCT eligibility window", "threshold": "Loes >9 or NRS ≥2", "action": "HSCT unlikely beneficial; consider palliative/LEV-based seizure management"},
]

MONITORING = [
    "MRI brain q6M (age 3-12yr presymptomatic males); q12M adolescent/adult; immediately if ANY neurological symptom",
    "Loes score + NRS at every MRI (specialist review)",
    "Plasma VLCFA (C26:0, C24:0/C22:0 ratio) q6-12M; q3M if on Lorenzo's Oil",
    "Morning cortisol + ACTH stimulation test q6-12M; electrolytes q3M",
    "Neuropsychological assessment annually (CCALD: detailed cognitive + behavioural)",
    "MRI spinal cord in AMN q12-24M (T2 cord hyperintensity progression)",
    "Audiometry + nerve conduction studies q12M (AMN peripheral neuropathy)",
    "EEG: baseline at first seizure; annual in active CCALD; earlier if SE or new symptoms",
    "CBC monthly × 2yr post-HSCT/GT; bone marrow biopsy annually (gene therapy REMS)",
    "Lorenzo's Oil: platelet count monthly; LFT q6M; VLCFA q3M",
    "Enzyme-inducing AED initiation: cortisol level q2W × 3M",
]

LIFECYCLE = [
    {
        "stage": "NBS / Presymptomatic (0-3yr)",
        "features": "C26:0-lyso-PC elevated on DBS; ABCD1 confirmed; NO neurological symptoms; adrenal function testing; brain MRI NORMAL; Loes 0",
        "action": "Adrenal hormone replacement if AI; Lorenzo's Oil (Level C); MRI q6M from age 3; family counselling; genetic cascade testing",
    },
    {
        "stage": "CCALD Surveillance Window (3-12yr)",
        "features": "MRI Loes 0→evolving; serial NRS; no/minimal symptoms; ANY Gd+ = emergency",
        "action": "Immediate HSCT or gene therapy referral if Gd+ appears; Lorenzo's Oil ongoing; cortisol management; school support",
    },
    {
        "stage": "Active CCALD (4-10yr typical)",
        "features": "Rapid neurological regression; seizures 90%+; posterior WM expansion; Gd+; Loes rising",
        "action": "URGENT HSCT (if Loes ≤9, NRS ≤1) or Skysona; LEV first-line AED; cortisol stress protocol; avoid PHT/CBZ; school/behaviour support",
    },
    {
        "stage": "AMN Adult Phase (20-40yr)",
        "features": "Progressive spastic paraparesis; peripheral neuropathy; spinal cord T2 lesions; seizures 15-20% (cerebral conversion)",
        "action": "Physiotherapy; orthotics; pain management; baclofen (spasticity); LEV if seizures; Lorenzo's Oil; cortisol replacement; sexual dysfunction management",
    },
    {
        "stage": "AMN + Cerebral Conversion",
        "features": "AMN + new Gd+ cerebral lesions; rapid deterioration; seizures escalating",
        "action": "HSCT less effective but may be considered; intensive AED (LEV ± CLB); avoid ALL enzyme inducers; cortisol management; palliative planning",
    },
    {
        "stage": "Late Stage / Palliative",
        "features": "Nonverbal; wheelchair; frequent seizures; aspiration risk; respiratory compromise",
        "action": "Palliative AED (CLB/LEV PO or rectal); enteral nutrition; respiratory support; comfort care; advance directives",
    },
]

DIFFERENTIAL_DIAGNOSIS = [
    {
        "condition": "MLD (ARSA — Metachromatic Leukodystrophy)",
        "distinction": "AR (not X-linked); arylsulfatase A; sulfatide accumulation; WM but different distribution; VLCFA normal; urine sulfatide elevated; no adrenal insufficiency",
    },
    {
        "condition": "Krabbe Disease (GALC)",
        "distinction": "AR; galactocerebrosidase; early infantile onset; Gd+ at specific WM sites; no adrenal disease; VLCFA normal",
    },
    {
        "condition": "Addison Disease (Autoimmune)",
        "distinction": "Most common Addison cause; VLCFA normal; ABCD1 sequencing rules out X-ALD; 21-hydroxylase antibodies +",
    },
    {
        "condition": "MS (Multiple Sclerosis)",
        "distinction": "Relapsing-remitting or progressive; Gd+ lesions common; no adrenal disease; no VLCFA elevation; demyelination different distribution",
    },
    {
        "condition": "NPC1/NPC2 (Niemann-Pick C)",
        "distinction": "AR; lysosomal cholesterol; VSGP + gelastic cataplexy PATHOGNOMONIC; oxysterol biomarkers; VLCFA normal",
    },
    {
        "condition": "POLG1-related leukodystrophy",
        "distinction": "Mitochondrial; lactic acidosis; Alpers syndrome; VLCFA normal; X-ALD screen before VPA",
    },
    {
        "condition": "CLN (Neuronal Ceroid Lipofuscinosis)",
        "distinction": "AR (various genes); visual failure (PPT1/CLN1); EM vacuolar lymphocytes; ceroid accumulation; VLCFA normal",
    },
    {
        "condition": "Zellweger Spectrum (PEX genes)",
        "distinction": "AR; generalised peroxisomal dysfunction; VLCFA elevated (same finding!) but also phytanic acid, pristanic, very long chain fatty alcohol; severe early onset; ZSD-specific MRI",
    },
]

# ── Cohort simulation (deterministic 40 patients) ─────────────────────────────
random.seed(42)
_COHORT = []
for i in range(COHORT_SIZE):
    phenotype = random.choices(
        ["CCALD", "Adolescent-Cerebral", "Adult-Cerebral", "AMN", "Addison-only", "Female-het-AMN"],
        weights=[35, 7, 5, 40, 10, 3],
        k=1
    )[0]
    is_ccald = phenotype in ("CCALD", "Adolescent-Cerebral", "Adult-Cerebral")
    is_amn = phenotype == "AMN"
    is_addison = phenotype == "Addison-only"
    has_ai = phenotype not in ("Female-het-AMN",) and random.random() < (0.9 if is_addison else 0.55 if is_ccald else 0.65)
    has_seizures = (
        random.random() < 0.92 if is_ccald else
        random.random() < 0.17 if is_amn else
        random.random() < 0.17 if is_addison else
        random.random() < 0.08
    )
    loes = (
        random.randint(4, 18) if phenotype == "CCALD" else
        random.randint(2, 12) if phenotype in ("Adolescent-Cerebral", "Adult-Cerebral") else
        random.randint(0, 3)
    )
    on_hsct = phenotype == "CCALD" and loes <= 9 and random.random() < 0.6
    on_gt = phenotype == "CCALD" and loes <= 9 and not on_hsct and random.random() < 0.3
    on_lorenzo = phenotype not in ("CCALD",) and random.random() < 0.4
    aed = None
    response = None
    if has_seizures:
        aed = random.choices(["LEV", "VPA+LEV", "CLB+LEV", "VPA", "CBZ (pre-diagnosis)"], weights=[55, 20, 10, 10, 5], k=1)[0]
        response = random.choices(
            ["Well controlled", "Partially controlled", "Drug-resistant"],
            weights=[35, 40, 25] if is_ccald else [65, 25, 10],
            k=1
        )[0]
    _COHORT.append({
        "patient_id": f"ABCD1-{i+1:03d}",
        "phenotype": phenotype,
        "sex": "F" if phenotype == "Female-het-AMN" else "M",
        "loes_score": loes,
        "has_ai": has_ai,
        "has_seizures": has_seizures,
        "on_hsct": on_hsct,
        "on_gt": on_gt,
        "on_lorenzo": on_lorenzo,
        "primary_aed": aed,
        "drug_response": response,
    })

N = len(_COHORT)


def get_overview():
    """Return ABCD1/X-ALD overview for /api/abcd1/overview."""
    seizure_n = sum(1 for p in _COHORT if p["has_seizures"])
    ccald_n = sum(1 for p in _COHORT if p["phenotype"] in ("CCALD", "Adolescent-Cerebral", "Adult-Cerebral"))
    amn_n = sum(1 for p in _COHORT if p["phenotype"] == "AMN")
    ai_n = sum(1 for p in _COHORT if p["has_ai"])
    hsct_n = sum(1 for p in _COHORT if p["on_hsct"])
    gt_n = sum(1 for p in _COHORT if p["on_gt"])
    dr_n = sum(1 for p in _COHORT if p["drug_response"] == "Drug-resistant")
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "inheritance": INHERITANCE,
        "disease_mechanism": DISEASE_MECHANISM,
        "cohort_size": N,
        "seizure_pct": round(100 * seizure_n / N),
        "ccald_pct": round(100 * ccald_n / N),
        "amn_pct": round(100 * amn_n / N),
        "adrenal_insufficiency_pct": round(100 * ai_n / N),
        "on_hsct_pct": round(100 * hsct_n / N),
        "on_gt_pct": round(100 * gt_n / N),
        "drug_resistance_pct": round(100 * dr_n / seizure_n) if seizure_n else 0,
        "vlcfa_diagnostic_c26_sensitivity_pct": 100,
        "nbs_positive_rate": "~2 per 100,000 newborns",
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "key_concepts": KEY_CONCEPTS[:8],
        "standards": STANDARDS[:6],
    }


def get_breakdown():
    """Return ABCD1/X-ALD breakdown for /api/abcd1/breakdown."""
    return {
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "thresholds": THRESHOLDS,
        "lifecycle": LIFECYCLE,
        "patients": _COHORT,
    }


def get_definitions():
    """Return ABCD1/X-ALD definitions for /api/abcd1/definitions."""
    return {
        "key_concepts": KEY_CONCEPTS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "pharmacological_distinctions": [
            "PHT/Phenytoin — ABSOLUTE CI: CYP3A4 induction → cortisol drop → adrenal crisis (ALL X-ALD males); IV Fosphenytoin ALSO ABSOLUTE CI in SE → use IV LEV",
            "CBZ/Carbamazepine — RELATIVE CI: CYP3A4 moderate induction → cortisol falls 30-40%; cortisol monitoring q2W × 3M after starting; prefer LEV",
            "OXC/Oxcarbazepine — RELATIVE CI: weaker inducer than CBZ but same concern; cortisol q1M × 6M",
            "PB/Phenobarbitone — RELATIVE CI: chronic CYP induction; sedation worsens CCALD cognitive decline",
            "Typical antipsychotics — HIGH RISK: EPS worsens CCALD neurological decline; disrupt adrenal axis; use atypical only",
            "Anaesthesia — EXTREME HAZARD: mandatory 100mg IV hydrocortisone at induction + 50mg q6h × 24-48h perioperative stress dosing",
            "VPA — RELATIVE CI: POLG1 exclusion mandatory (CPIC A); LFT q3M; cortisol monitoring (adrenal compromise); LEV preferred",
            "VGB — NOT INDICATED: no benefit; visual monitoring impossible in advanced CCALD",
            "Corticosteroids (high-dose for CCALD) — NOT disease-modifying: temporarily reduces oedema only; HSCT/GT is curative intervention",
            "Lorenzo's Oil — PRESYMPTOMATIC ONLY (Level C): normalises VLCFA plasma; NO benefit in established CCALD/AMN; thrombocytopenia monitoring monthly",
            "LEV — FIRST-LINE AED (Level C): NO enzyme induction; NO adrenal interaction; IV formulation available; preferred in ALL X-ALD phenotypes",
            "HSCT (Level A) — CCALD Loes ≤9 + NRS ≤1 + Gd+: arrests neuroinflammation permanently; no benefit established AMN; gene therapy alternative if no HLA match",
        ],
        "diagnostic_algorithm": [
            "Step 1: Any male with unexplained neurological regression / white matter disease / adrenal insufficiency → check plasma VLCFA (C26:0, C24:0/C22:0 ratio) IMMEDIATELY",
            "Step 2: VLCFA elevated → confirm with ABCD1 sequencing (WES/targeted panel + CNV/MLPA for large deletions ~5%)",
            "Step 3: Brain MRI + Loes score; spinal cord MRI if AMN phenotype; gadolinium-enhanced sequences MANDATORY",
            "Step 4: ACTH stimulation test → morning cortisol (exclude/confirm adrenal insufficiency); electrolytes; ACTH level",
            "Step 5: Assess NRS (neurological disability score 0-25); EEG if seizures",
            "Step 6: HSCT eligibility assessment — Loes ≤9 + NRS ≤1 + Gd+ = IMMEDIATE REFERRAL to transplant centre",
            "Step 7: If no HLA-matched sibling → gene therapy (Skysona) evaluation (age ≤17yr, CCALD)",
            "Step 8: Start hydrocortisone + fludrocortisone if AI confirmed; give emergency cortisol kit",
            "Step 9: Cascade testing all first-degree male relatives (maternal uncles, brothers) + female carriers; NBS if available",
            "Step 10: Surveillance MRI q6M (age 3-12yr males); q12M adolescent/adult; NRS q3-6M; VLCFA q6-12M",
        ],
    }
