#!/usr/bin/env python3
"""GCDH (Glutaryl-CoA Dehydrogenase) Deficiency — Glutaric Aciduria Type 1 (GA1) Dashboard.

GCDH encodes glutaryl-CoA dehydrogenase, a mitochondrial FAD-dependent enzyme:
  Glutaryl-CoA + FAD  →  [GCDH]  →  Crotonyl-CoA + FADH₂ + CO₂
  (step in lysine/hydroxylysine/tryptophan catabolism — saccharopine pathway → glutaryl-CoA)

GCDH LOF → Glutaryl-CoA CANNOT be dehydrogenated → accumulates:
  → Glutaric acid (GA): VERY HIGH (primary biomarker; urine >>100 mmol/mol Cr)
  → 3-Hydroxyglutaric acid (3-HGA): ELEVATED (the primary NEUROTOXIN)
  → Glutarylcarnitine (C5DC): ELEVATED (newborn screening marker)
  → Secondary carnitine depletion (free carnitine LOW)

OMIM Disease: #231670 (Glutaric Aciduria, Type I / GA1)
OMIM Gene:   *608801 (GCDH)
Chromosome: 19p13.2
Inheritance: Autosomal Recessive (AR)
Protein: 438 aa; mitochondrial matrix; homotetrameric; FAD-dependent acyl-CoA dehydrogenase
Prevalence: ~1:100,000 general; ~1:360 Old Order Amish (P196S); ~1:300 Ojibway-Cree (P196T)

MECHANISM — 3-HYDROXYGLUTARIC ACID NEUROTOXICITY + STRIATAL VULNERABILITY:
  Normal GCDH: converts glutaryl-CoA (from Lys/Trp catabolism) to crotonyl-CoA
  GCDH LOF: glutaryl-CoA hydrolyses → glutaric acid (GA) + accumulates 3-hydroxyglutaric acid
  3-HGA (3-hydroxyglutaric acid) = PRIMARY NEUROTOXIN:
    → Structural analogue of glutamate → NMDA receptor agonist → excitotoxicity
    → Inhibits succinate dehydrogenase (Complex II) → energy deficit in striatum
    → Striatum (caudate + putamen) is disproportionately vulnerable:
       high metabolic demand + dense glutamate receptors + blood-brain barrier immaturity
  CRITICAL PERIOD: First 6 years of life (myelination period = peak vulnerability)
  After age 6: further encephalopathic crises unlikely to cause new striatal injury
  BUT: pre-existing striatal damage → lifelong chronic dystonia

ENCEPHALOPATHIC CRISIS:
  Trigger: febrile illness (infection, surgery, vaccination) → catabolic state →
           increased Lys/Trp catabolism → MORE glutaryl-CoA → MORE 3-HGA →
           acute striatal excitotoxic injury → acute onset dystonia/dyskinesia
  Timing: usually within first 6 years; peak 6 months – 3 years
  Outcome: each crisis can cause PERMANENT striatal damage → progressive dystonia
  Prevention: EMERGENCY PROTOCOL during illness (glucose + carnitine → anti-catabolic)

MACROCEPHALY:
  90% of GA1 patients: OFC > 98th percentile
  Frontotemporal atrophy + widened Sylvian fissures (CSF-filled spaces) = characteristic MRI
  Can appear before clinical onset — macrocephaly in infancy → investigate GA1
  SUBDURAL HAEMORRHAGE can occur (child abuse mimicry — bridging vein stretch in macrocephaly)

LOW EXCRETOR VARIANT:
  ~15% of GA1 patients: LOW or even NORMAL glutaric acid on urine/plasma metabolomics
  C5DC (glutarylcarnitine) also borderline → MISSED by NBS in low excretors
  3-HGA may also be low in low excretors → requires gene panel for diagnosis
  IMPORTANT: Normal NBS does NOT exclude GA1 (low excretor variant)
  p.Arg402Trp = predominant low excretor allele (Amish: homozygous → ~1:360)

POSITION IN LYSINE CATABOLISM:
  L-Lysine → saccharopine pathway → glutaryl-CoA → [GCDH] → crotonyl-CoA → acetyl-CoA
  GCDH LOF: glutaryl-CoA accumulates → spontaneous hydrolysis → glutaric acid + 3-HGA
  → Cannot progress to crotonyl-CoA/acetyl-CoA → TCA cycle entry impaired for Lys
"""

import random

SEED       = 241      # next in series after SLC25A13 (seed 235)
N_PATIENTS = 40

# Phenotypic classes
PHENOTYPE_CLASSES = [
    {"name": "Classic (crisis + striatal injury)", "pct": 60, "note": "Encephalopathic crisis → acute striatal injury → chronic dystonia; macrocephaly; most common in unscreened"},
    {"name": "NBS-detected presymptomatic",        "pct": 25, "note": "Identified by newborn screening before crisis; best outcomes with emergency protocol compliance"},
    {"name": "Low-excretor / late-onset",          "pct": 15, "note": "Normal/mild GA on metabolomics; often missed by NBS; diagnosed by gene panel; milder phenotype"},
]

# Top pathogenic variants
VARIANTS = [
    {"variant": "p.Arg402Trp (c.1204C>T)",    "freq": 8,  "domain": "FAD-binding domain — substrate channel",    "phenotype": "Low excretor; Amish (homozygous ~1:360)",  "note": "Most common worldwide; low excretor → borderline NBS; homozygous in Old Order Amish population; FAD binding impaired but partial enzyme activity retained"},
    {"variant": "p.Ala421Val (c.1262C>T)",    "freq": 7,  "domain": "FAD-binding domain",                        "phenotype": "Classic excretor; high GA",                "note": "Second most common worldwide; high GA excretor; classic presentation with encephalopathic crisis risk; broad ethnic distribution"},
    {"variant": "p.Glu414Ala (c.1241A>C)",    "freq": 6,  "domain": "C-terminal FAD-binding",                   "phenotype": "Classic; Northern European",               "note": "Northern European population; classic excretor; macrocephaly; crisis in febrile illness"},
    {"variant": "p.Pro196Thr (c.586C>A)",     "freq": 5,  "domain": "Substrate-binding loop",                   "phenotype": "Ojibway-Cree founder (~1:300)",            "note": "Indigenous Canadian (Ojibway-Cree) founder allele; ~1:300 prevalence in this community; high excretor; classic presentation"},
    {"variant": "p.Pro196Ser (c.586C>T)",     "freq": 5,  "domain": "Substrate-binding loop",                   "phenotype": "Amish low excretor variant",               "note": "Old Order Amish; similar position to P196T; low excretor; NBS may miss; gene panel required"},
    {"variant": "c.IVS10+5G>A",               "freq": 5,  "domain": "Splice site — intron 10",                  "phenotype": "Classic; European; null",                  "note": "Splice donor disruption → exon 10 skipping → premature stop; null allele; high excretor; broad European distribution"},
    {"variant": "p.Arg402Gln (c.1205G>A)",    "freq": 4,  "domain": "FAD-binding — Arg402 locus",               "phenotype": "Mild; intermediate excretor",              "note": "Same position as R402W but different substitution; partial FAD binding retained; mild-to-moderate phenotype; NBS may detect"},
    {"variant": "p.Val400Met (c.1198G>A)",    "freq": 3,  "domain": "FAD-binding domain",                       "phenotype": "Classic; moderate excretor",               "note": "Moderate excretor; classic encephalopathic crisis risk in first 6 years"},
]

# Biomarker panel
BIOMARKERS = {
    "ga_urine":  {"label": "Urine Glutaric Acid",           "normal": "<8 mmol/mol Cr",     "status": "VERY HIGH (>100 classic; 10-100 low excretor)",  "direction": "↑↑↑ CRITICAL",  "color": "danger"},
    "hga_urine": {"label": "Urine 3-Hydroxyglutaric Acid",  "normal": "<2 mmol/mol Cr",     "status": "ELEVATED (>10 classic) — PRIMARY NEUROTOXIN",    "direction": "↑↑ NEUROTOXIN", "color": "danger"},
    "c5dc":      {"label": "Plasma C5DC (Glutarylcarnitine)","normal": "<0.3 µmol/L",        "status": "ELEVATED (NBS marker; may be borderline low ex)", "direction": "↑ NBS marker",  "color": "warning"},
    "carnitine": {"label": "Free Carnitine (plasma)",        "normal": "25–60 µmol/L",       "status": "LOW (secondary depletion; glutaryl-carnitine)",   "direction": "↓ LOW",         "color": "warning"},
    "macroceph": {"label": "OFC / Head Circumference",       "normal": "<98th percentile",   "status": "MACROCEPHALY (>98th; 90% patients)",             "direction": "↑ MACRO",       "color": "warning"},
    "mri_brain": {"label": "Brain MRI",                      "normal": "Normal",              "status": "Frontotemporal atrophy + widened Sylvian fissures + striatal changes",
                                                                                                                                "direction": "⚠ ABNORMAL",    "color": "warning"},
    "ga_plasma": {"label": "Plasma Glutaric Acid",           "normal": "<1 µmol/L",          "status": "ELEVATED (less specific than urine)",            "direction": "↑ ELEVATED",    "color": "warning"},
    "nh3":       {"label": "Plasma Ammonia",                  "normal": "<50 µmol/L",         "status": "NORMAL — KEY NEGATIVE vs UCDs",                  "direction": "→ NORMAL",      "color": "success"},
    "citrulline":{"label": "Plasma Citrulline",               "normal": "15–35 µmol/L",       "status": "NORMAL — KEY NEGATIVE vs UCDs",                  "direction": "→ NORMAL",      "color": "success"},
    "orotic":    {"label": "Urine Orotic Acid",               "normal": "<6 µmol/mol Cr",     "status": "NORMAL — KEY NEGATIVE vs UCDs/OTC/CPS1",         "direction": "→ NORMAL",      "color": "success"},
    "thcy":      {"label": "Total Homocysteine",              "normal": "<15 µmol/L",         "status": "NORMAL — KEY NEGATIVE vs CBS/MTHFR",             "direction": "→ NORMAL",      "color": "success"},
    "mma":       {"label": "Methylmalonic Acid (MMA)",        "normal": "<0.4 µmol/L",        "status": "NORMAL — KEY NEGATIVE vs MMUT/cblA/cblB",        "direction": "→ NORMAL",      "color": "success"},
    "vlcfa":     {"label": "VLCFA",                           "normal": "Normal",              "status": "NORMAL — KEY NEGATIVE vs peroxisomal",           "direction": "→ NORMAL",      "color": "success"},
}

# Treatments (evidence-graded)
TREATMENTS = [
    {
        "therapy": "Lysine restriction",
        "level": "A",
        "dose": "Natural protein 0.8–1.5 g/kg/day; Lys intake <100 mg/kg/day; Lys-free amino acid supplement for protein target",
        "rationale": "Lysine is the primary precursor of glutaryl-CoA (tryptophan contributes minimally after infancy). Restricting Lys reduces substrate for GCDH-pathway → less glutaric acid + 3-HGA produced. Arginine competes with lysine for transport at intestinal (SLC7A9) and blood-brain barrier (y+LAT) transporters → arginine supplementation also reduces Lys entry into brain.",
        "class": "Dietary (primary — natural protein restriction)",
    },
    {
        "therapy": "L-Carnitine supplementation",
        "level": "A",
        "dose": "100–200 mg/kg/day oral; IV in acute crises; titrate to normalise free carnitine (>25 µmol/L)",
        "rationale": "GCDH LOF → glutaryl-CoA + carnitine → glutarylcarnitine (C5DC) → carnitine depleted. Secondary carnitine deficiency worsens metabolic crisis (impaired fatty acid oxidation → energy failure). Carnitine supplementation: (1) replenishes free carnitine pool; (2) enhances urinary excretion of glutaric acid as C5DC (detoxification); (3) protects cardiac and skeletal muscle.",
        "class": "Carnitine supplementation (primary)",
    },
    {
        "therapy": "Riboflavin (riboflavin-responsive GA1)",
        "level": "B",
        "dose": "100–300 mg/day (pharmacological dose); trial for 3 months; assess biochemical response (GA reduction >50%)",
        "rationale": "GCDH is FAD-dependent. Some missense variants (particularly FAD-binding domain mutations like p.Arg402Trp) retain partial apoenzyme that can be stabilised/activated by pharmacological FAD doses (riboflavin → riboflavin-5-phosphate → FAD). Not all patients respond; biochemical response (reduced urinary GA) predicts clinical response. Still recommended to add Lys restriction even in responders.",
        "class": "Cofactor therapy (riboflavin-responsive subset)",
    },
    {
        "therapy": "Emergency protocol during febrile illness",
        "level": "A",
        "dose": "High-dose glucose (10% dextrose IV or oral glucose polymer at 10 mg/kg/min); IV carnitine 100 mg/kg; withhold protein 24–48h; continue if fever >38°C",
        "rationale": "Febrile catabolism → increased Lys turnover → MORE glutaryl-CoA → MORE 3-HGA → acute striatal excitotoxic crisis. High-dose glucose: (1) anti-catabolic — suppresses endogenous Lys catabolism; (2) provides energy substrate bypassing GCDH block. Emergency carnitine: replenishes depleted pool. TIME-CRITICAL: crisis prevention requires starting within hours of fever onset. Emergency action cards provided to all GA1 families.",
        "class": "Emergency (crisis prevention — CRITICAL protocol)",
    },
    {
        "therapy": "Arginine supplementation",
        "level": "B",
        "dose": "200–400 mg/kg/day; competes with Lys at intestinal + BBB transporters",
        "rationale": "Arginine and lysine share the cationic amino acid transporters (SLC7A9, CAT1/SLC7A1). High arginine: (1) competitively inhibits Lys absorption in gut (reduces Lys entry); (2) reduces Lys transport across blood-brain barrier → less substrate entering brain for GCDH pathway → less 3-HGA generated in neurons. Synergistic with Lys restriction. Also supports carbamoyl-P and urea cycle (no CI here unlike in UCDs).",
        "class": "Amino acid (adjunct)",
    },
    {
        "therapy": "LEV (levetiracetam)",
        "level": "B",
        "dose": "10–40 mg/kg/day; first-line AED for true seizures",
        "rationale": "True seizures (vs dystonic spasms) occur in ~30-40% GA1 patients after striatal injury. LEV: no hepatotoxicity, no significant carnitine depletion, no interaction with amino acid metabolism. SV2A mechanism does not aggravate glutamatergic toxicity. First-line AED in GA1.",
        "class": "AED — first-line",
    },
    {
        "therapy": "Benzodiazepines (acute dystonic crisis)",
        "level": "A",
        "dose": "Diazepam 0.2–0.3 mg/kg IV/rectal (acute); clonazepam chronic (0.02–0.1 mg/kg/day)",
        "rationale": "Acute encephalopathic crisis → dystonic spasms often misidentified as epileptic seizures. Benzodiazepines: (1) GABAergic inhibition reduces acute striatal excitotoxicity (reduces 3-HGA-mediated NMDA activation); (2) anti-dystonic (baclofen as adjunct). Critical: EEG distinguishes dystonic spasms from true seizures; treat appropriately. Benzodiazepines + emergency protocol together for crisis management.",
        "class": "Acute crisis management — dystonia/seizure",
    },
    {
        "therapy": "Valproate (VPA)",
        "level": "MODERATE RISK",
        "dose": "Avoid preferentially; if essential, supplement carnitine, monitor liver closely",
        "rationale": "VPA in GA1: (1) inhibits carnitine biosynthesis + renal reabsorption → worsens ALREADY depleted carnitine (secondary GA1 deficiency); (2) hepatotoxic potential (not as severe as UCDs but caution with pre-existing metabolic liver stress); (3) VPA-CoA competes with fatty acid oxidation. Not absolute CI (unlike UCDs) but AVOID preferentially. Use LEV instead.",
        "class": "Moderate-risk AED (prefer alternatives)",
    },
    {
        "therapy": "High-protein / Lysine-rich foods",
        "level": "AVOID",
        "dose": "Monitor dietary Lys; avoid high-Lys foods (meat, legumes, dairy in excess) during catabolic crises",
        "rationale": "Lysine is the principal substrate for GCDH pathway. Excess Lys → excess glutaryl-CoA → more 3-HGA → increased striatal toxicity. Especially critical during febrile illness (avoid high-Lys loads). Long-term: Lys restriction reduces cumulative 3-HGA exposure.",
        "class": "Dietary restriction",
    },
]

# Seizure types
SEIZURE_TYPES = [
    {"type": "Dystonic spasms (crisis — misidentified as seizures)", "pct": 55, "note": "Acute striatal injury → dystonia/dyskinesia; looks like tonic-clonic but EEG often non-epileptic; benzodiazepines + emergency protocol"},
    {"type": "Focal / complex partial seizures",                      "pct": 35, "note": "Post-striatal injury epilepsy; frontal/temporal from cortical spread; LEV first-line"},
    {"type": "GTCS (generalised tonic-clonic)",                       "pct": 25, "note": "Acute encephalopathic crisis; may coexist with dystonic spasms; EEG distinguishes"},
    {"type": "Infantile spasms (acute crisis phase)",                 "pct": 15, "note": "Some infants: encephalopathic crisis → West syndrome-like spasms; ACTH trialled with emergency metabolic protocol"},
    {"type": "Drug-resistant epilepsy (DRE)",                         "pct": 20, "note": "Chronic striatal damage → refractory focal epilepsy; structural substrate; multi-AED; seizure surgery evaluation"},
    {"type": "Absence / myoclonic",                                   "pct": 10, "note": "Less common; thalamic involvement in severe cases"},
]

# Systemic features
SYSTEMIC_FEATURES = [
    {"feature": "Macrocephaly (OFC >98th percentile)",      "pct": 90, "note": "PATHOGNOMONIC hallmark: enlarged head circumference; frontotemporal atrophy + widened Sylvian fissures (CSF spaces); apparent at birth or within first months"},
    {"feature": "Chronic dystonia (post-striatal injury)",  "pct": 75, "note": "Caudate + putamen degeneration → generalised/segmental dystonia; progressive; worsens with crises; defines long-term disability"},
    {"feature": "Encephalopathic crisis (febrile trigger)", "pct": 70, "note": "Fever → acute striatal excitotoxic injury → acute dystonia onset; usually 6 months–3 years (peak); each crisis = potential permanent damage"},
    {"feature": "Intellectual disability (post-crisis)",    "pct": 55, "note": "Severity proportional to number/severity of crises; NBS-detected + emergency protocol → normal IQ in 70%"},
    {"feature": "Subdural haemorrhage (SDH)",               "pct": 20, "note": "Stretched bridging veins over enlarged Sylvian spaces; trauma (minor) → SDH; can mimic non-accidental injury (child abuse); awareness critical"},
    {"feature": "Speech delay / dysarthria",                "pct": 50, "note": "Striatal + cortical spread → dysarthria; aphasia; especially post-crisis; augmentative communication often needed"},
    {"feature": "Hypotonia (neonatal/early)",               "pct": 35, "note": "Generalised hypotonia in infancy; precedes crisis; early clue with macrocephaly"},
    {"feature": "Frontotemporal atrophy (MRI)",             "pct": 85, "note": "Widened Sylvian fissures; 'bat wing' temporal lobes; enlarged intracranial CSF spaces; bridging vein stretching"},
]


def _generate_cohort():
    random.seed(SEED)
    patients = []
    phenotype_dist = []
    for cls in PHENOTYPE_CLASSES:
        n = round(N_PATIENTS * cls["pct"] / 100)
        phenotype_dist.extend([cls["name"]] * n)
    while len(phenotype_dist) < N_PATIENTS:
        phenotype_dist.append("Classic (crisis + striatal injury)")
    phenotype_dist = phenotype_dist[:N_PATIENTS]
    random.shuffle(phenotype_dist)

    for i in range(N_PATIENTS):
        phenotype = phenotype_dist[i]
        if phenotype == "NBS-detected presymptomatic":
            age_onset_months = round(random.uniform(0, 1), 2)       # detected at birth
            ga_urine         = random.randint(50, 250)               # high but controlled
            hga_urine        = round(random.uniform(8, 40), 1)
            c5dc             = round(random.uniform(0.5, 2.5), 2)
            free_carnitine   = random.randint(12, 25)
            ofc_zscore       = round(random.uniform(1.5, 3.5), 1)   # mild macrocephaly
            crisis_count     = 0
            dystonia         = False
            dre              = False
        elif phenotype == "Low-excretor / late-onset":
            age_onset_months = round(random.uniform(6, 120), 1)
            ga_urine         = random.randint(5, 50)                 # low/normal excretor
            hga_urine        = round(random.uniform(1, 15), 1)
            c5dc             = round(random.uniform(0.2, 0.8), 2)   # may be borderline
            free_carnitine   = random.randint(15, 35)
            ofc_zscore       = round(random.uniform(0.5, 2.5), 1)
            crisis_count     = random.randint(0, 1)
            dystonia         = random.random() < 0.30
            dre              = random.random() < 0.15
        else:  # Classic
            age_onset_months = round(random.uniform(6, 48), 1)       # peak 6 mo–3 yr
            ga_urine         = random.randint(120, 800)
            hga_urine        = round(random.uniform(20, 120), 1)
            c5dc             = round(random.uniform(1.0, 6.0), 2)
            free_carnitine   = random.randint(5, 20)
            ofc_zscore       = round(random.uniform(2.0, 5.0), 1)
            crisis_count     = random.randint(1, 5)
            dystonia         = random.random() < 0.85
            dre              = random.random() < 0.25

        macrocephaly = ofc_zscore >= 2.0
        seizures     = random.random() < 0.40 or dre
        idd          = phenotype == "Classic (crisis + striatal injury)" and crisis_count >= 2 and random.random() < 0.60
        sdh          = macrocephaly and random.random() < 0.20
        riboflavin_r = random.random() < 0.15  # ~15% riboflavin-responsive
        v            = random.choice(VARIANTS)

        patients.append({
            "id":                  f"GCDH-{i+1:03d}",
            "phenotype":           phenotype,
            "age_onset_months":    age_onset_months,
            "ga_urine_mmol_mol_cr":ga_urine,
            "hga_urine_mmol_mol_cr":hga_urine,
            "c5dc_umol_l":         c5dc,
            "free_carnitine":      free_carnitine,
            "ofc_zscore":          ofc_zscore,
            "macrocephaly":        macrocephaly,
            "crisis_count":        crisis_count,
            "dystonia":            dystonia,
            "seizures":            seizures,
            "dre":                 dre,
            "idd":                 idd,
            "sdh":                 sdh,
            "riboflavin_responsive": riboflavin_r,
            "variant":             v["variant"],
        })
    return patients


COHORT = _generate_cohort()


def get_overview():
    n             = len(COHORT)
    n_macro       = sum(1 for p in COHORT if p["macrocephaly"])
    n_dystonia    = sum(1 for p in COHORT if p["dystonia"])
    n_seizures    = sum(1 for p in COHORT if p["seizures"])
    n_dre         = sum(1 for p in COHORT if p["dre"])
    n_idd         = sum(1 for p in COHORT if p["idd"])
    n_sdh         = sum(1 for p in COHORT if p["sdh"])
    n_classic     = sum(1 for p in COHORT if p["phenotype"] == "Classic (crisis + striatal injury)")
    n_nbs         = sum(1 for p in COHORT if p["phenotype"] == "NBS-detected presymptomatic")
    n_low_ex      = sum(1 for p in COHORT if p["phenotype"] == "Low-excretor / late-onset")
    avg_ga        = round(sum(p["ga_urine_mmol_mol_cr"] for p in COHORT) / n)
    avg_hga       = round(sum(p["hga_urine_mmol_mol_cr"] for p in COHORT) / n, 1)
    avg_c5dc      = round(sum(p["c5dc_umol_l"] for p in COHORT) / n, 2)
    avg_carnitine = round(sum(p["free_carnitine"] for p in COHORT) / n, 1)

    return {
        "disease": "Glutaric Aciduria Type 1 (GA1) — GCDH Deficiency",
        "omim_gene": "608801",
        "omim_disease": "231670",
        "gene": "GCDH",
        "alias": "Glutaryl-CoA Dehydrogenase / GA1",
        "chromosome": "19p13.2",
        "inheritance": "Autosomal Recessive (AR)",
        "protein": "438 aa; mitochondrial matrix homotetrameric enzyme; FAD-dependent acyl-CoA dehydrogenase",
        "prevalence": "~1:100,000 general; ~1:360 Old Order Amish (P196S); ~1:300 Ojibway-Cree First Nations (P196T)",
        "mechanism": "Glutaryl-CoA CANNOT be dehydrogenated → GA + 3-HGA accumulate; 3-HGA is NMDA agonist + Complex II inhibitor → striatal excitotoxicity during febrile illness",
        "n_patients": n,
        "kpi": {
            "n_patients":    {"value": n,                               "label": "Cohort size",                           "color": "#1a237e"},
            "ga_avg":        {"value": f"{avg_ga} mmol/mol Cr",         "label": "Mean Urine GA (VERY HIGH classic)",     "color": "#b71c1c"},
            "hga_avg":       {"value": f"{avg_hga} mmol/mol Cr",        "label": "Mean Urine 3-HGA (neurotoxin)",         "color": "#c62828"},
            "c5dc_avg":      {"value": f"{avg_c5dc} µmol/L",            "label": "Mean C5DC (NBS marker)",                "color": "#e65100"},
            "carnitine_avg": {"value": f"{avg_carnitine} µmol/L",       "label": "Mean Free Carnitine (LOW)",             "color": "#6a1b9a"},
            "macro_pct":     {"value": f"{round(n_macro/n*100)}%",      "label": "Macrocephaly (>98th percentile)",       "color": "#0d47a1"},
            "dystonia_pct":  {"value": f"{round(n_dystonia/n*100)}%",   "label": "Chronic Dystonia (post-striatal)",      "color": "#37474f"},
            "seizures_pct":  {"value": f"{round(n_seizures/n*100)}%",   "label": "Seizures (%)",                          "color": "#b71c1c"},
            "sdh_pct":       {"value": f"{round(n_sdh/n*100)}%",        "label": "Subdural Haemorrhage (%)",              "color": "#880e4f"},
            "dre_pct":       {"value": f"{round(n_dre/n*100)}%",        "label": "Drug-Resistant Epilepsy (%)",           "color": "#bf360c"},
        },
        "phenotype_dist": [
            {"class": "Classic (crisis + striatal injury)", "n": n_classic, "pct": round(n_classic/n*100)},
            {"class": "NBS-detected presymptomatic",        "n": n_nbs,     "pct": round(n_nbs/n*100)},
            {"class": "Low-excretor / late-onset",          "n": n_low_ex,  "pct": round(n_low_ex/n*100)},
        ],
        "hallmark_biomarker": (
            "Urine Glutaric Acid VERY HIGH (>100 mmol/mol Cr classic; may be low in low-excretor variant); "
            "Urine 3-HGA ELEVATED (primary neurotoxin — NMDA agonist); "
            "C5DC (glutarylcarnitine) ELEVATED on NBS (may be borderline low-excretor); "
            "Free Carnitine LOW (secondary depletion)"
        ),
        "hallmark_clinical": "Macrocephaly + Encephalopathic crisis (febrile trigger → striatal injury → dystonia) during first 6 years of life",
        "low_excretor_warning": (
            "LOW EXCRETOR VARIANT (~15%): Urine GA may be NORMAL or only mildly elevated. "
            "C5DC on NBS may also be borderline. CRITICAL: Normal metabolomics does NOT exclude GA1. "
            "p.Arg402Trp = main low-excretor allele (Old Order Amish). Gene panel required for diagnosis."
        ),
        "crisis_protocol_note": (
            "EMERGENCY PROTOCOL (time-critical): Start within hours of fever >38°C. "
            "High-dose glucose (anti-catabolic) + IV/oral L-carnitine + protein restriction 24–48h. "
            "Prevents acute striatal excitotoxic injury. Family must have written action plan and "
            "emergency medication kit. Crisis = permanent disability risk."
        ),
        "nbs_note": (
            "NBS detects C5DC (glutarylcarnitine) on tandem mass spectrometry. "
            "Low-excretor variant may have borderline or normal C5DC → may be missed. "
            "Macrocephaly in NBS-era infant → investigate even if NBS was normal."
        ),
        "sdh_note": (
            "SUBDURAL HAEMORRHAGE (GA1): Macrocephalic brain → bridging veins stretched over enlarged "
            "Sylvian spaces → minor trauma → SDH. Mimics non-accidental injury. "
            "All infants with SDH + macrocephaly should be screened for GA1 (urine OA + C5DC)."
        ),
    }


def get_breakdown():
    return {
        "biomarkers": BIOMARKERS,
        "variants": VARIANTS,
        "treatments": TREATMENTS,
        "seizure_types": SEIZURE_TYPES,
        "systemic_features": SYSTEMIC_FEATURES,
        "phenotype_classes": PHENOTYPE_CLASSES,
        "cohort_preview": COHORT[:10],
        "enzyme_mechanism": {
            "gene":         "GCDH (Glutaryl-CoA Dehydrogenase)",
            "function":     "FAD-dependent mitochondrial enzyme; catalyses dehydrogenation of glutaryl-CoA in Lys/Trp catabolism",
            "reaction":     "Glutaryl-CoA + FAD → Crotonyl-CoA + FADH₂ + CO₂ (decarboxylating step)",
            "block":        "GCDH LOF → glutaryl-CoA CANNOT be oxidised → spontaneous hydrolysis → glutaric acid + 3-HGA accumulate",
            "neurotoxin":   "3-Hydroxyglutaric acid (3-HGA): structural glutamate analogue → NMDA receptor agonist + Complex II (succinate dehydrogenase) inhibitor → striatal excitotoxicity",
            "striatum_vuln":"Caudate + putamen: highest glutamate receptor density + high metabolic demand + immature BBB (0-6 years) → disproportionate vulnerability to 3-HGA",
            "critical_period": "First 6 years of life (myelination window); crisis after age 6 rarely causes new striatal damage",
            "lys_pathway":  "L-Lysine → saccharopine → α-aminoadipic semialdehyde → 2-oxoadipic acid → glutaryl-CoA → [GCDH blocked] → GA + 3-HGA accumulate",
        },
        "differential_diagnosis": {
            "vs_d2hga": {
                "key_diff": "D-2-Hydroxyglutaric aciduria (D2HGDH/IDH2): 2-HGA elevated, NOT glutaric + 3-HGA",
                "ga1": "GA1: glutaric acid VERY HIGH + 3-HGA elevated; 2-HGA normal",
                "d2hga": "D2HGA: D-2-hydroxyglutaric acid elevated; glutaric acid normal; cardiomyopathy + leukodystrophy",
            },
            "vs_ga2": {
                "key_diff": "Glutaric Aciduria Type 2 (MADD — ETFA/ETFB/ETFDH): MULTIPLE acylcarnitines + GA; not isolated GA",
                "ga1": "GA1: ISOLATED elevation of GA + 3-HGA + C5DC; other acylcarnitines normal",
                "ga2_madd": "GA2: C5DC elevated + PLUS multiple others (C4, C5, C5-OH, C6, C8, C10, C12) + non-ketotic hypoglycaemia + cardiomyopathy",
            },
            "vs_l2hga": {
                "key_diff": "L-2-Hydroxyglutaric aciduria (L2HGDH): L-2-HGA, not glutaric acid",
                "ga1": "GA1: glutaric + 3-HGA; no 2-HGA",
                "l2hga": "L2HGDH: L-2-HGA elevated; progressive cerebellar atrophy + cortical leukodystrophy; no macrocephaly",
            },
            "vs_non_accidental_injury": {
                "key_diff": "Subdural haemorrhage in GA1 mimics non-accidental injury",
                "ga1": "GA1: macrocephaly + SDH + glutaric aciduria on urine OA; bridging vein stretching mechanism",
                "nai": "NAI: no metabolic abnormalities; SDH pattern may differ but overlap. Screen ALL SDH + macrocephaly infants with urine OA + NBS repeat",
            },
        },
    }


def get_definitions():
    return {
        "gene_function": (
            "GCDH (Glutaryl-CoA Dehydrogenase, OMIM *608801) encodes a 438-amino-acid mitochondrial "
            "matrix enzyme. GCDH is a FAD-dependent acyl-CoA dehydrogenase (ACAD family) that forms "
            "a homotetramer. It catalyses the oxidative decarboxylation of glutaryl-CoA to crotonyl-CoA: "
            "  Glutaryl-CoA + FAD → Crotonyl-CoA + FADH₂ + CO₂ "
            "This is a critical step in the catabolism of L-lysine, L-hydroxylysine, and L-tryptophan. "
            "The enzyme passes electrons via electron-transfer flavoprotein (ETF/ETFA-ETFB) → ETFDH → "
            "Complex III of the respiratory chain. "
            "Chromosome 19p13.2; autosomal recessive; >200 pathogenic variants known."
        ),
        "pathomechanism": (
            "GCDH LOF → glutaryl-CoA CANNOT undergo dehydrogenation. Glutaryl-CoA undergoes spontaneous "
            "hydrolysis → free glutaric acid (GA) + 3-hydroxyglutaric acid (3-HGA) accumulate. "
            "3-HGA is the PRIMARY neurotoxin via two mechanisms: "
            "(1) NMDA receptor agonism: 3-HGA is a structural analogue of glutamate → agonises NMDA "
            "    receptors → excitotoxic calcium influx → neuronal death (especially striatum). "
            "(2) Complex II (succinate dehydrogenase) inhibition: 3-HGA inhibits SDH → energy deficit "
            "    in high-metabolic-demand cells → amplifies excitotoxic vulnerability. "
            "Striatal vulnerability: Caudate nucleus + putamen have the highest density of NMDA receptors, "
            "highest metabolic demand, and immature BBB in the first 6 years → disproportionately damaged "
            "during febrile illness when Lys catabolism surges (catabolic state). "
            "The CRITICAL PERIOD is the first 6 years of life (myelination window). After age 6, "
            "new striatal damage from encephalopathic crises is rare — but existing damage is permanent."
        ),
        "encephalopathic_crisis_mechanism": (
            "Encephalopathic crisis = the pivotal GA1 event. Trigger: any febrile illness, surgery, "
            "or even vaccination causing a catabolic state. "
            "Mechanism: catabolic state → increased protein turnover → increased Lys catabolism → "
            "MORE glutaryl-CoA → MORE 3-HGA produced (GCDH cannot clear it) → acute surge of 3-HGA "
            "into the brain → acute striatal excitotoxicity → ACUTE ONSET dystonia (within hours). "
            "Clinical presentation: infant/toddler has fever, then suddenly develops hypotonia, "
            "loss of motor milestones, acute-onset movement disorder resembling seizures or stroke. "
            "THIS IS NOT A SEIZURE — it is an encephalopathic crisis mimicking acute striatal injury. "
            "Outcome: each crisis can permanently destroy portions of the caudate/putamen → "
            "progressive irreversible dystonia. Multiple crises = progressive disability accumulation. "
            "PREVENTION: emergency protocol within hours of ANY fever (not waiting for 'bad fever'). "
            "TIMING IS CRITICAL — delay of 6-12 hours increases risk of permanent striatal damage."
        ),
        "macrocephaly_mechanism": (
            "Macrocephaly (OFC >98th percentile) is the hallmark physical finding in GA1, present in ~90% "
            "of patients. Pathomechanism: "
            "(1) Frontotemporal atrophy: chronic low-level 3-HGA toxicity in frontal and temporal cortex → "
            "    neuronal loss → compensatory expansion of CSF spaces (ex vacuo enlargement). "
            "(2) Widened Sylvian fissures: temporal lobe hypoplasia/atrophy → enlarged perisylvian CSF spaces "
            "    ('bat-wing' temporal lobes on MRI). "
            "(3) Subdural haematoma risk: enlarged intracranial CSF spaces → bridging veins stretched over "
            "    the cortical surface → minor head trauma → subdural bleed. "
            "Macrocephaly may be the ONLY finding before the first crisis → any infant with macrocephaly "
            "should have urine organic acids (GA + 3-HGA) and plasma C5DC (glutarylcarnitine) checked. "
            "MEDICO-LEGAL: GA1 macrocephaly + SDH can exactly mimic non-accidental injury. "
            "All SDH cases with macrocephaly MUST be screened for GA1 before concluding NAI."
        ),
        "low_excretor_variant": (
            "Approximately 15% of GA1 patients are LOW EXCRETORS — their urinary glutaric acid is "
            "NORMAL or only minimally elevated (<10–30 mmol/mol Cr), and plasma C5DC may be borderline "
            "or normal on newborn screening. "
            "KEY ALLELE: p.Arg402Trp (R402W) — the predominant low-excretor variant. Homozygous in "
            "Old Order Amish (~1:360 prevalence). This variant retains partial GCDH enzyme activity, "
            "producing less metabolic overflow, but still has significant neurological risk. "
            "Implications: (1) NBS may miss low excretors (C5DC borderline); "
            "(2) Urine OA in a febrile low excretor may be only mildly elevated — FALSE REASSURANCE; "
            "(3) Brain MRI (frontotemporal atrophy + macrocephaly) may be the diagnostic clue; "
            "(4) Gene panel is required for definitive diagnosis; "
            "(5) Emergency protocol applies equally — low excretor does NOT mean low risk."
        ),
        "treatment_rationale": (
            "GA1 treatment targets: reduce 3-HGA production + replenish carnitine + prevent crisis. "
            "(1) Lysine restriction: Lys is the primary substrate for the GCDH pathway. "
            "    Natural protein restriction to 0.8–1.5 g/kg/day + Lys-free amino acid supplement "
            "    for protein adequacy. Target urine GA reduction by >50%. "
            "(2) L-Carnitine (100–200 mg/kg/day): replenishes secondary carnitine deficiency; "
            "    enhances C5DC excretion (detoxification); protects cardiac muscle. "
            "(3) Riboflavin (100–300 mg/day): pharmacological FAD supplementation; stabilises "
            "    residual enzyme activity in FAD-binding domain variants; 15–20% of patients respond "
            "    with >50% reduction in urinary GA. "
            "(4) Emergency protocol: HIGH PRIORITY — withhold natural protein + high-dose glucose "
            "    (10 mg/kg/min) + IV carnitine at onset of fever >38°C. Anti-catabolic strategy "
            "    prevents the critical surge of 3-HGA during illness. Written emergency plans for "
            "    all GA1 families; emergency letter for A&E departments."
        ),
        "seizure_vs_dystonia": (
            "CRITICAL DISTINCTION in GA1: Seizures vs Dystonic Spasms (encephalopathic crisis). "
            "DYSTONIC CRISIS (NOT seizure): acute-onset abnormal posturing/movements during/after "
            "febrile illness; may look like tonic posturing; EEG often NORMAL or non-epileptic. "
            "Treatment: benzodiazepines + emergency metabolic protocol (glucose + carnitine). "
            "TRUE SEIZURES: occur in ~35-40% post-striatal injury; focal or GTCS; EEG epileptiform. "
            "Treatment: AEDs (LEV first-line). "
            "MUST distinguish: (1) EEG during episode; (2) metabolic emergency protocol even if "
            "movements are dystonic (not epileptic) because crisis can damage striatum regardless; "
            "(3) False labelling as epilepsy → AEDs used instead of metabolic protocol → crisis "
            "    continues → more striatal damage. "
            "AED choice: LEV (safe in GA1). VPA: MODERATE RISK (carnitine depletion, hepatotoxicity "
            "potential) — prefer alternatives. NOT absolute CI unlike UCDs."
        ),
    }
